#include "kvcache.h"
#include "log.h"
#include "memory.h"
#include "recipe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline int kvcache_kv_heads_stride(const kvcache *c) {
	return c->n_kv_heads_max;
}

static int kvcache_device_of(const kvcache *c, const backend *b) {
	if (b == c->backend)
		return 0;
	for (int i = 0; i < c->n_mirrors; i++)
		if (c->mirror[i].be == b)
			return i + 1;
	return -1;
}

int kvcache_layer_device(const kvcache *c, const model *m, int layer) {
	backend *lb = c->backend;
	if (m && layer >= 0 && layer < m->n_layer_backends && m->layer_backends[layer])
		lb = m->layer_backends[layer];
	return kvcache_device_of(c, lb);
}

static int kvcache_device_view(const kvcache *c, int dev, int layer, buffer **k, buffer **v,
							   int *slot) {
	if (dev <= 0) {
		*k		   = (buffer *)&c->k;
		*v		   = (buffer *)&c->v;
		int mapped = (c->primary_remap && layer >= 0 && layer < c->n_primary_remap)
						 ? c->primary_remap[layer]
						 : layer;
		*slot	   = mapped >= 0 ? mapped : layer;
		return mapped >= 0 ? 0 : -1;
	}
	const kv_mirror *mir	= &c->mirror[dev - 1];
	int				 mapped = (layer >= 0 && layer < mir->n_remap) ? mir->remap[layer] : -1;
	*k						= (buffer *)&mir->k;
	*v						= (buffer *)&mir->v;
	*slot					= mapped >= 0 ? mapped : layer;
	return mapped >= 0 ? 0 : -1;
}

void kvcache_attn_view(const kvcache *c, const model *m, int exec_layer, int kv_layer, buffer **k,
					   buffer **v, int *slot) {
	int dev = kvcache_layer_device(c, m, exec_layer);
	kvcache_device_view(c, dev, kv_layer, k, v, slot);
}

int kvcache_device_count(const kvcache *c) {
	return 1 + c->n_mirrors;
}

backend *kvcache_device_backend(const kvcache *c, int dev) {
	return dev <= 0 ? c->backend : c->mirror[dev - 1].be;
}

int kvcache_device_slots(const kvcache *c, int dev) {
	if (dev > 0)
		return c->mirror[dev - 1].n_slots;
	return c->primary_remap ? c->n_primary_slots : c->n_kv_layers;
}

size_t kvcache_device_bytes(const kvcache *c, const model *m, int dev) {
	if (c->is_mla || c->hybrid)
		return 0;
	backend *be		   = kvcache_device_backend(c, dev);
	int		 compacted = backend_has_cap(be, BCAP_KV_POS_CAP);
	size_t	 total	   = 0;
	for (int i = 0; i < c->n_kv_layers; i++) {
		int present = dev > 0 ? (c->slot_devices ? (c->slot_devices[i] >> dev) & 1u : 0)
							  : (!c->slot_devices || (c->slot_devices[i] & 1u));
		if (present)
			total += model_kv_slot_bytes(m, c->n_ctx, c->kv_quant, compacted, i);
	}
	return total;
}

static status_code kvcache_ensure_transfer_buf(kvcache *c, size_t need_floats);

static status_code kvcache_map_slots(kvcache *c, const model *m);
static status_code kvcache_alloc_slots(kvcache *c, const model *m, int dev, backend *be, buffer *k,
									   buffer *v, int **remap_out, int *n_slots_out);
static status_code kvcache_alloc_mla_buffer(kvcache *c, const model *m, backend *be, int n_layers,
											buffer *kv);
static status_code kvcache_alloc_primary_slots(kvcache *c, const model *m, backend *be, bool *held);

static status_code kvcache_alloc_layer_state(backend *be, buffer *out, size_t floats) {
	if (floats == 0)
		return OK;
	return buffer_alloc_f32(be, floats, out);
}

static status_code kvcache_alloc_kda_state(kvcache_hybrid *q, const model *m) {
	const size_t n = (size_t)m->n_layers;
	q->conv_buf	   = xcalloc(n, sizeof(buffer));
	q->state_buf   = xcalloc(n, sizeof(buffer));
	q->latent_buf  = xcalloc(n, sizeof(buffer));
	for (int li = 0; li < m->n_layers; li++) {
		backend	   *be		  = model_layer_backend(m, li);
		const int	recurrent = model_layer_is_recurrent(m, li);
		status_code s =
			recurrent ? kvcache_alloc_layer_state(be, &q->conv_buf[li], q->conv_stride)
					  : kvcache_alloc_layer_state(be, &q->latent_buf[li], q->mla_latent_stride);
		if (s == OK && recurrent)
			s = kvcache_alloc_layer_state(be, &q->state_buf[li], q->recurrent_stride);
		if (s != OK)
			return s;
	}
	return OK;
}

static void kvcache_free_layer_states(kvcache_hybrid *q) {
	buffer *sets[3] = {q->conv_buf, q->state_buf, q->latent_buf};
	for (int i = 0; i < 3; i++) {
		for (int li = 0; sets[i] && li < q->n_layers; li++)
			buffer_release(&sets[i][li]);
		free(sets[i]);
	}
}

status_code kvcache_init(kvcache *c, const model *m, int n_ctx, kv_quant_type kv_quant) {
	memset(c, 0, sizeof(*c));
	c->n_ctx	= n_ctx;
	c->n_pos	= 0;
	c->backend	= m->backend;
	c->kv_quant = kv_quant;

	int is_mla	  = m->arch_info->is_mla;
	int is_hybrid = m->arch_info->is_hybrid_recurrent;

	if (kv_quant != KV_QUANT_F16 && is_mla) {
		ERROR("kvcache: quantized KV cache is not supported for MLA models");
		return ERR_UNSUPPORTED;
	}

	c->head_dim_max	  = m->head_dim;
	c->n_kv_heads_max = m->n_kv_heads;
	if (m->arch_info->has_variable_layer_dims) {
		for (int i = 0; i < m->n_layers; i++) {
			int hd = model_layer_head_dim(m, i);
			if (hd > c->head_dim_max)
				c->head_dim_max = hd;
			int kvh = model_layer_kv_heads(m, i);
			if (kvh > c->n_kv_heads_max)
				c->n_kv_heads_max = kvh;
		}
	}

	int n_kv_layers =
		m->layer_dims.n_layer_kv_from_start > 0 ? m->layer_dims.n_layer_kv_from_start : m->n_layers;
	c->n_kv_layers = n_kv_layers;

	if (is_mla && !m->arch_info->is_kda) {
		c->is_mla = true;
		if (!m->mixed_backend_mode)
			return kvcache_alloc_mla_buffer(c, m, m->backend, m->n_layers, &c->k);
		bool held;
		return kvcache_alloc_primary_slots(c, m, m->backend, &held);
	}

	if (is_hybrid) {
		int				conv_state_only = m->arch_info->hybrid_shortconv;
		size_t			chans = m->hybrid.conv_channels > 0 ? (size_t)m->hybrid.conv_channels
															: (size_t)m->hybrid.conv_dim;
		kvcache_hybrid *q	  = xcalloc(1, sizeof(*q));
		q->conv_stride		  = chans * (size_t)(m->hybrid.conv_kernel - 1);
		q->recurrent_stride = conv_state_only
								  ? 0
								  : (size_t)m->hybrid.n_value_heads * (size_t)m->hybrid.state_size *
										(size_t)m->hybrid.value_head_dim;
		q->conv_total		= (size_t)m->n_layers * q->conv_stride;
		q->recurrent_total	= (size_t)m->n_layers * q->recurrent_stride;
		q->n_layers			= m->n_layers;
		c->hybrid			= q;
		if (m->arch_info->is_kda) {
			q->mla_latent_stride = (size_t)n_ctx * (size_t)m->mla.kv_lora;
			return kvcache_alloc_kda_state(q, m);
		}
		q->conv_state = xcalloc((size_t)m->n_layers, q->conv_stride * sizeof(float));
		if (q->recurrent_stride > 0)
			q->recurrent_state = xcalloc((size_t)m->n_layers, q->recurrent_stride * sizeof(float));
	}

	backend *kv_backend = c->backend->kv_alloc ? c->backend : backend_host();
	if (kv_backend != c->backend)
		backend_report_host_fallback(
			c->backend, "kv_alloc", HFB_CAPABILITY,
			"backend '%s' has no kv_alloc; kv cache allocated in host (cpu) memory",
			c->backend->name);

	if (kv_quant == KV_QUANT_Q8_0 && !backend_has_cap(kv_backend, BCAP_KV_QUANT_Q8_0)) {
		ERROR("kvcache: backend '%s' does not support Q8_0 quantized KV cache; "
			  "use --kv-quant f16 or select a backend that supports it",
			  kv_backend->name);
		return ERR_UNSUPPORTED;
	}

	if (m->mixed_backend_mode && !is_hybrid) {
		bool		held;
		status_code s = kvcache_alloc_primary_slots(c, m, kv_backend, &held);
		if (s != OK || held) {
			if (s == OK)
				s = kvcache_ensure_transfer_buf(c, (size_t)2 * (size_t)c->n_kv_heads_max *
													   (size_t)c->head_dim_max);
			return s;
		}
	}

	int *layer_head_dim	  = NULL;
	int *layer_n_kv_heads = NULL;
	if ((m->arch_info->has_variable_layer_dims || is_hybrid) &&
		backend_has_cap(kv_backend, BCAP_IS_HOST)) {
		layer_head_dim	 = xcalloc(m->n_layers, sizeof(int));
		layer_n_kv_heads = xcalloc(m->n_layers, sizeof(int));
		for (int i = 0; i < m->n_layers; i++) {
			layer_head_dim[i] = model_layer_head_dim(m, i);
			int kvh			  = model_layer_kv_heads(m, i);
			if (!m->arch_info->has_variable_layer_dims && m->layer_dims.n_kv_heads_per_layer &&
				i < m->n_layers)
				kvh = m->layer_dims.n_kv_heads_per_layer[i] > 0
						  ? m->layer_dims.n_kv_heads_per_layer[i]
						  : 0;
			layer_n_kv_heads[i] = kvh;
		}
	}

	int *layer_pos_cap = NULL;
	if (backend_has_cap(kv_backend, BCAP_KV_POS_CAP) && n_kv_layers > 0 && m->sliding_window > 0) {
		layer_pos_cap = xcalloc((size_t)n_kv_layers, sizeof(int));
		for (int i = 0; i < n_kv_layers; i++)
			layer_pos_cap[i] = model_kv_layer_pos_cap(m, n_ctx, i);
	}

	kv_desc desc = {
		.n_layers		  = m->n_layers,
		.n_kv_layers	  = n_kv_layers,
		.n_kv_heads		  = c->n_kv_heads_max,
		.head_dim		  = c->head_dim_max,
		.n_ctx			  = n_ctx,
		.kv_quant		  = kv_quant,
		.layer_head_dim	  = layer_head_dim,
		.layer_n_kv_heads = layer_n_kv_heads,
		.layer_pos_cap	  = layer_pos_cap,
	};
	status_code s = kv_backend->kv_alloc(kv_backend, &desc, &c->k, &c->v);
	free(layer_head_dim);
	free(layer_n_kv_heads);
	free(layer_pos_cap);
	if (s != OK)
		return s;
	if (m->mixed_backend_mode || kv_backend != c->backend) {
		s = kvcache_ensure_transfer_buf(c, (size_t)2 * (size_t)c->n_kv_heads_max *
											   (size_t)c->head_dim_max);
		if (s != OK)
			return s;
	}
	return OK;
}

static void kv_pair_free(backend *be, buffer *k, buffer *v, bool mla) {
	if (!mla && be && be->kv_free) {
		be->kv_free(be, k, v);
		return;
	}
	buffer_release(k);
	buffer_release(v);
}

void kvcache_free(kvcache *c) {
	if (!c->backend)
		return;
	if (!(c->hybrid && c->hybrid->conv_buf)) {
		bool mla = c->is_mla;
		kv_pair_free(c->backend, &c->k, &c->v, mla);
		for (int i = 0; i < c->n_mirrors; i++) {
			kv_mirror *mir = &c->mirror[i];
			kv_pair_free(mir->be, &mir->k, &mir->v, mla);
			free(mir->remap);
		}
		c->n_mirrors = 0;
	}
	c->is_mla = false;
	if (c->hybrid) {
		kvcache_free_layer_states(c->hybrid);
		free(c->hybrid->conv_state);
		free(c->hybrid->recurrent_state);
		free(c->hybrid);
		c->hybrid = NULL;
	}
	free(c->primary_remap);
	c->primary_remap   = NULL;
	c->n_primary_remap = 0;
	free(c->slot_devices);
	c->slot_devices	  = NULL;
	c->n_slot_devices = 0;
	free(c->kv_transfer_buf);
	c->kv_transfer_buf = NULL;
	c->kv_transfer_cap = 0;
	memset(c, 0, sizeof(*c));
}

static int kvcache_register_device(kvcache *c, backend *b) {
	int dev = kvcache_device_of(c, b);
	if (dev >= 0)
		return dev;
	if (c->n_mirrors >= KV_MAX_MIRRORS)
		return -1;
	c->mirror[c->n_mirrors].be = b;
	return ++c->n_mirrors;
}

static status_code kvcache_map_slots(kvcache *c, const model *m) {
	if (c->slot_devices)
		return OK;
	c->slot_devices	  = xcalloc((size_t)m->n_layers, sizeof(unsigned));
	c->n_slot_devices = m->n_layers;
	for (int i = 0; i < m->n_layers; i++) {
		backend *lb	 = model_layer_backend(m, i);
		int		 dev = kvcache_register_device(c, lb ? lb : c->backend);
		if (dev < 0) {
			ERROR("kvcache: more than %d devices hold layers", KV_MAX_MIRRORS + 1);
			return ERR_UNSUPPORTED;
		}
		int slot = i;
		if (m->recipe)
			slot = model_layer_kv_layer(m, i);
		if (slot >= 0 && slot < m->n_layers)
			c->slot_devices[slot] |= 1u << dev;
		if (i < c->n_kv_layers)
			c->slot_devices[i] |= 1u << dev;
	}
	return OK;
}

static int kvcache_slot_remap(const kvcache *c, const model *m, unsigned bit, int *remap) {
	int next = 0;
	for (int i = 0; i < m->n_layers; i++)
		remap[i] = (c->slot_devices[i] & bit) ? next++ : -1;
	return next;
}

static status_code kvcache_alloc_mla_buffer(kvcache *c, const model *m, backend *be, int n_layers,
											buffer *kv) {
	bool	 native = be->kv_alloc_mla && be->attention_mla && be->kv_put_mla;
	backend *owner	= native ? be : backend_host();
	if (!owner || !owner->kv_alloc_mla) {
		ERROR("kvcache: model is MLA but no backend can allocate its KV cache");
		return ERR_UNSUPPORTED;
	}
	if (!native)
		backend_report_host_fallback(
			be, "kv_alloc_mla", HFB_CAPABILITY,
			"backend '%s' has no native MLA; MLA kv cache allocated in host (cpu) memory",
			be->name);
	return owner->kv_alloc_mla(owner, n_layers, c->n_ctx, m->mla.kv_lora, m->mla.qk_rope, kv);
}

static status_code kvcache_alloc_primary_slots(kvcache *c, const model *m, backend *be,
											   bool *held) {
	*held		  = false;
	status_code s = kvcache_map_slots(c, m);
	if (s != OK)
		return s;
	for (int i = 0; i < m->n_layers; i++)
		*held |= (c->slot_devices[i] & 1u) != 0;
	if (!*held)
		return OK;
	s = kvcache_alloc_slots(c, m, 0, be, &c->k, &c->v, &c->primary_remap, &c->n_primary_slots);
	c->n_primary_remap = m->n_layers;
	return s;
}

static status_code kvcache_alloc_slots(kvcache *c, const model *m, int dev, backend *be, buffer *k,
									   buffer *v, int **remap_out, int *n_slots_out) {
	unsigned bit	 = 1u << dev;
	int		*remap	 = xcalloc((size_t)m->n_layers, sizeof(int));
	int		 n_slots = kvcache_slot_remap(c, m, bit, remap);
	if (n_slots == 0) {
		free(remap);
		ERROR("kvcache: backend '%s' holds no KV slots", be->name);
		return ERR_INVALID_ARG;
	}

	if (c->is_mla) {
		status_code s = kvcache_alloc_mla_buffer(c, m, be, n_slots, k);
		if (s != OK) {
			free(remap);
			return s;
		}
		*remap_out	 = remap;
		*n_slots_out = n_slots;
		return OK;
	}

	if (!be->kv_alloc) {
		free(remap);
		ERROR("kvcache: backend '%s' has no kv_alloc; cannot hold KV for its layers", be->name);
		return ERR_UNSUPPORTED;
	}
	if (c->kv_quant == KV_QUANT_Q8_0 && !backend_has_cap(be, BCAP_KV_QUANT_Q8_0)) {
		free(remap);
		ERROR("kvcache: backend '%s' does not support Q8_0 quantized KV cache; "
			  "use --kv-quant f16",
			  be->name);
		return ERR_UNSUPPORTED;
	}

	bool per_layer_dims =
		(m->arch_info->has_variable_layer_dims || m->arch_info->is_hybrid_recurrent) &&
		backend_has_cap(be, BCAP_IS_HOST);
	int *slot_list		= xcalloc((size_t)n_slots, sizeof(int));
	int *layer_head_dim = per_layer_dims ? xcalloc((size_t)n_slots, sizeof(int)) : NULL;
	int *layer_n_kv		= per_layer_dims ? xcalloc((size_t)n_slots, sizeof(int)) : NULL;
	for (int i = 0; i < m->n_layers; i++) {
		if (remap[i] < 0)
			continue;
		int next		= remap[i];
		slot_list[next] = i;
		if (per_layer_dims) {
			int kvh = model_layer_kv_heads(m, i);
			if (!m->arch_info->has_variable_layer_dims && m->layer_dims.n_kv_heads_per_layer)
				kvh = m->layer_dims.n_kv_heads_per_layer[i] > 0
						  ? m->layer_dims.n_kv_heads_per_layer[i]
						  : 0;
			layer_head_dim[next] = model_layer_head_dim(m, i);
			layer_n_kv[next]	 = kvh;
		}
	}

	int *pos_cap = NULL;
	if (backend_has_cap(be, BCAP_KV_POS_CAP) && m->sliding_window > 0) {
		pos_cap = xcalloc((size_t)n_slots, sizeof(int));
		for (int i = 0; i < n_slots; i++)
			pos_cap[i] = model_kv_layer_pos_cap(m, c->n_ctx, slot_list[i]);
	}

	kv_desc desc = {
		.n_layers		  = n_slots,
		.n_kv_layers	  = n_slots,
		.n_kv_heads		  = c->n_kv_heads_max,
		.head_dim		  = c->head_dim_max,
		.n_ctx			  = c->n_ctx,
		.kv_quant		  = c->kv_quant,
		.layer_head_dim	  = layer_head_dim,
		.layer_n_kv_heads = layer_n_kv,
		.layer_pos_cap	  = pos_cap,
	};
	status_code s = be->kv_alloc(be, &desc, k, v);
	free(layer_head_dim);
	free(layer_n_kv);
	free(pos_cap);
	free(slot_list);
	if (s != OK) {
		free(remap);
		return s;
	}
	*remap_out	 = remap;
	*n_slots_out = n_slots;
	return OK;
}

status_code kvcache_alloc_mirrors(kvcache *c, const model *m) {
	if (!c || !m || !m->mixed_backend_mode || (c->hybrid && c->hybrid->conv_buf))
		return OK;
	status_code s = kvcache_map_slots(c, m);
	for (int i = 0; s == OK && i < c->n_mirrors; i++) {
		kv_mirror *mir = &c->mirror[i];
		if (mir->remap)
			continue;
		s = kvcache_alloc_slots(c, m, i + 1, mir->be, &mir->k, &mir->v, &mir->remap, &mir->n_slots);
		mir->n_remap = m->n_layers;
	}
	if (s != OK)
		return s;
	return kvcache_ensure_transfer_buf(c, (size_t)2 * (size_t)c->n_kv_heads_max *
											  (size_t)c->head_dim_max);
}

void kvcache_reset(kvcache *c) {
	c->n_pos = 0;
	if (c->hybrid) {
		if (c->hybrid->conv_state && c->hybrid->conv_total > 0)
			memset(c->hybrid->conv_state, 0, c->hybrid->conv_total * sizeof(float));
		if (c->hybrid->recurrent_state && c->hybrid->recurrent_total > 0)
			memset(c->hybrid->recurrent_state, 0, c->hybrid->recurrent_total * sizeof(float));
	}
}

static status_code kvcache_ensure_transfer_buf(kvcache *c, size_t need_floats) {
	if (c->kv_transfer_cap >= need_floats)
		return OK;
	c->kv_transfer_buf = xrealloc(c->kv_transfer_buf, need_floats * sizeof(float));
	c->kv_transfer_cap = need_floats;
	return OK;
}

static void kv_host_pair_bufs(kvcache *c, int k_floats, buffer *k_out, buffer *v_out) {
	memset(k_out, 0, sizeof(*k_out));
	memset(v_out, 0, sizeof(*v_out));
	k_out->handle	= c->kv_transfer_buf;
	k_out->host_ptr = c->kv_transfer_buf;
	k_out->size		= (size_t)k_floats * sizeof(float);
	k_out->offset	= 0;
	k_out->owner	= backend_host();
	v_out->handle	= (char *)c->kv_transfer_buf + (size_t)k_floats * sizeof(float);
	v_out->host_ptr = v_out->handle;
	v_out->size		= (size_t)k_floats * sizeof(float);
	v_out->offset	= 0;
	v_out->owner	= backend_host();
}

static status_code kvcache_put_device(kvcache *c, const model *m, int dev, int layer, int pos,
									  const buffer *k_in, const buffer *v_in) {
	int hd		   = model_layer_head_dim(m, layer);
	int kvh_stride = kvcache_kv_heads_stride(c);
	int kvh_active = model_layer_kv_heads(m, layer);

	buffer *kb, *vb;
	int		slot;
	kvcache_device_view(c, dev, layer, &kb, &vb, &slot);
	backend *dev_backend = dev > 0 ? c->mirror[dev - 1].be : c->backend;
	backend *kv_backend	 = kb->owner ? kb->owner : dev_backend;
	backend *k_in_owner	 = k_in->owner ? k_in->owner : kv_backend;
	backend *v_in_owner	 = v_in->owner ? v_in->owner : kv_backend;

	if (k_in_owner != kv_backend || v_in_owner != kv_backend) {
		ensure_sync(k_in_owner);
		if (v_in_owner != k_in_owner)
			ensure_sync(v_in_owner);

		int			k_floats = kvh_active * hd;
		status_code st		 = kvcache_ensure_transfer_buf(c, (size_t)k_floats * 2);
		if (st != OK)
			return st;
		buffer k_host_buf, v_host_buf;
		kv_host_pair_bufs(c, k_floats, &k_host_buf, &v_host_buf);

		if (k_in_owner && k_in_owner->buffer_read_f32) {
			st = k_in_owner->buffer_read_f32(k_in_owner, k_in, c->kv_transfer_buf, k_floats);
			if (st != OK)
				return st;
		} else if (k_in->host_ptr) {
			memcpy(c->kv_transfer_buf, k_in->host_ptr, (size_t)k_floats * sizeof(float));
		} else {
			return ERR_UNSUPPORTED;
		}
		if (v_in_owner && v_in_owner->buffer_read_f32) {
			st =
				v_in_owner->buffer_read_f32(v_in_owner, v_in, (float *)v_host_buf.handle, k_floats);
			if (st != OK)
				return st;
		} else if (v_in->host_ptr) {
			memcpy(v_host_buf.handle, v_in->host_ptr, (size_t)k_floats * sizeof(float));
		} else {
			return ERR_UNSUPPORTED;
		}
		return kv_backend->kv_put(kv_backend, kb, vb, slot, pos, &k_host_buf, &v_host_buf,
								  kvh_stride, hd, c->n_ctx, kvh_active);
	}

	if (kv_backend != c->backend)
		ensure_sync(kv_backend);
	return kv_backend->kv_put(kv_backend, kb, vb, slot, pos, k_in, v_in, kvh_stride, hd, c->n_ctx,
							  kvh_active);
}

static unsigned kvcache_put_mask(const kvcache *c, const model *m, int layer, int *owner) {
	int dev = kvcache_layer_device(c, m, layer);
	if (dev < 0)
		dev = 0;
	*owner		  = dev;
	unsigned mask = c->slot_devices && layer < c->n_slot_devices ? c->slot_devices[layer] : 1u;
	return mask | (1u << dev);
}

status_code kvcache_put(kvcache *c, const model *m, int layer, int pos, const buffer *k_in,
						const buffer *v_in) {
	if (pos < 0 || pos >= c->n_ctx)
		return ERR_INVALID_ARG;
	if (layer < 0 || layer >= m->n_layers)
		return ERR_INVALID_ARG;

	int			owner;
	unsigned	mask = kvcache_put_mask(c, m, layer, &owner);
	status_code st	 = kvcache_put_device(c, m, owner, layer, pos, k_in, v_in);
	for (int dev = 0; st == OK && dev <= c->n_mirrors; dev++)
		if (dev != owner && (mask & (1u << dev)))
			st = kvcache_put_device(c, m, dev, layer, pos, k_in, v_in);
	return st;
}

static unsigned kvcache_slot_mask(const kvcache *c, int slot) {
	return c->slot_devices && slot < c->n_slot_devices ? c->slot_devices[slot] : 1u;
}

static backend *kvcache_shift_backend(const kvcache *c, int dev, const buffer *kb) {
	if (kb && kb->owner)
		return kb->owner;
	return kvcache_device_backend(c, dev);
}

bool kvcache_can_shift(const kvcache *c, const model *m) {
	if (!c || !m || c->is_mla || c->hybrid || c->n_kv_layers <= 0)
		return false;
	if (!m->arch_info->has_variable_layer_dims || !m->recipe)
		return false;
	for (int li = 0; li < c->n_kv_layers; li++) {
		unsigned mask = kvcache_slot_mask(c, li);
		for (int dev = 0; dev <= c->n_mirrors; dev++) {
			if (!(mask & (1u << dev)))
				continue;
			buffer *kb, *vb;
			int		slot;
			if (kvcache_device_view(c, dev, li, &kb, &vb, &slot) < 0)
				return false;
			backend *be = kvcache_shift_backend(c, dev, kb);
			if (!be || !be->kv_shift || !backend_has_cap(be, BCAP_KV_SHIFT))
				return false;
		}
	}
	return true;
}

static void kvcache_shift_tables(const model *m, int is_global, int head_dim,
								 const kv_shift_chunk *chunks, int n_chunks, float *cos_tbl,
								 float *sin_tbl) {
	const int	 half  = head_dim / 2;
	const float	 theta = is_global ? m->layer_dims.rope_theta_global : m->layer_dims.rope_theta_swa;
	const float *ff	   = is_global ? m->rope_freqs : NULL;
	const double theta_scale = pow((double)theta, -2.0 / (double)head_dim);
	for (int i = 0; i < n_chunks; i++) {
		double delta = (double)(chunks[i].dst_pos - chunks[i].src_pos);
		double freq	 = 1.0;
		for (int j = 0; j < half; j++, freq *= theta_scale) {
			double factor = 1.0;
			if (ff)
				factor = (ff[j] >= 1e10f || ff[j] == 0.0f) ? 0.0 : (double)ff[j];
			double angle				  = delta * freq * factor;
			cos_tbl[(size_t)i * half + j] = (float)cos(angle);
			sin_tbl[(size_t)i * half + j] = (float)sin(angle);
		}
	}
}

status_code kvcache_shift(kvcache *c, const model *m, const kv_shift_chunk *chunks, int n_chunks) {
	for (int li = 0; li < c->n_kv_layers; li++) {
		int is_global = m->layers[li].is_global_layer;
		int head_dim  = model_layer_head_dim(m, li);
		if (is_global && (!m->rope_freqs || m->rope_freqs_count < head_dim / 2))
			return ERR_UNSUPPORTED;
		size_t tbl_floats = (size_t)n_chunks * (size_t)(head_dim / 2);
		float *tables	  = xmalloc(2 * tbl_floats * sizeof(float));
		kvcache_shift_tables(m, is_global, head_dim, chunks, n_chunks, tables, tables + tbl_floats);
		unsigned	mask = kvcache_slot_mask(c, li);
		status_code st	 = OK;
		for (int dev = 0; dev <= c->n_mirrors && st == OK; dev++) {
			if (!(mask & (1u << dev)))
				continue;
			buffer *kb, *vb;
			int		slot;
			kvcache_device_view(c, dev, li, &kb, &vb, &slot);
			backend *be	  = kvcache_shift_backend(c, dev, kb);
			be->rope_neox = m->arch_info->uses_neox_rope;
			if (be != c->backend)
				ensure_sync(be);
			st = be->kv_shift(be, kb, vb, slot, chunks, n_chunks, tables, tables + tbl_floats,
							  kvcache_kv_heads_stride(c), head_dim, c->n_ctx,
							  model_layer_kv_heads(m, li));
			if (st == OK && be != c->backend)
				ensure_sync(be);
		}
		free(tables);
		if (st != OK)
			return st;
	}
	return OK;
}

status_code kvcache_put_batch(kvcache *c, const model *m, int layer, int pos_start,
							  const buffer *k_in, const buffer *v_in, int kv_row_stride,
							  int n_rows) {
	if (!c || !m || n_rows <= 0)
		return ERR_INVALID_ARG;
	if (layer < 0 || layer >= m->n_layers)
		return ERR_INVALID_ARG;
	if (pos_start < 0 || pos_start + n_rows > c->n_ctx)
		return ERR_INVALID_ARG;

	int hd		   = model_layer_head_dim(m, layer);
	int kvh_stride = kvcache_kv_heads_stride(c);
	int kvh_active = model_layer_kv_heads(m, layer);

	int		 owner;
	unsigned mask = kvcache_put_mask(c, m, layer, &owner);

	backend *kv_backend = c->k.owner ? c->k.owner : c->backend;
	backend *k_owner	= k_in->owner ? k_in->owner : kv_backend;
	backend *v_owner	= v_in->owner ? v_in->owner : kv_backend;

	bool same_backend = (k_owner == kv_backend && v_owner == kv_backend);
	bool fast_ok = same_backend && kv_backend->kv_put_batch != NULL && owner == 0 && mask == 1u;
	if (fast_ok) {
		buffer *kb, *vb;
		int		slot;
		kvcache_device_view(c, 0, layer, &kb, &vb, &slot);
		return kv_backend->kv_put_batch(kv_backend, kb, vb, slot, pos_start, k_in, v_in,
										kv_row_stride, kvh_stride, hd, c->n_ctx, kvh_active,
										n_rows);
	}

	for (int row = 0; row < n_rows; row++) {
		buffer krow = *k_in;
		buffer vrow = *v_in;
		size_t off	= (size_t)row * (size_t)kv_row_stride * sizeof(float);
		krow.offset += off;
		vrow.offset += off;
		status_code st = kvcache_put(c, m, layer, pos_start + row, &krow, &vrow);
		if (st != OK)
			return st;
	}
	return OK;
}
