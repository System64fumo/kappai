#include "kvcache.h"
#include "log.h"
#include "recipe.h"

#include <stdlib.h>
#include <string.h>

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

	if (is_mla) {
		if (!m->backend->kv_alloc_mla) {
			ERROR("kvcache: model is MLA but backend has no kv_alloc_mla");
			return ERR_UNSUPPORTED;
		}
		c->mla				= xcalloc(1, sizeof(kvcache_mla));
		c->mla->lora_dim	= m->mla.kv_lora;
		c->mla->qk_rope_dim = m->mla.qk_rope;
		return m->backend->kv_alloc_mla(m->backend, m->n_layers, n_ctx, c->mla->lora_dim,
										c->mla->qk_rope_dim, &c->mla->kv);
	}

	if (is_hybrid) {
		kvcache_hybrid *q	= xcalloc(1, sizeof(*q));
		q->conv_stride		= (size_t)m->hybrid.conv_dim * (size_t)(m->hybrid.conv_kernel - 1);
		q->recurrent_stride = (size_t)m->hybrid.n_value_heads * (size_t)m->hybrid.state_size *
							  (size_t)m->hybrid.value_head_dim;
		q->conv_total		= (size_t)m->n_layers * q->conv_stride;
		q->recurrent_total	= (size_t)m->n_layers * q->recurrent_stride;
		q->conv_state		= xcalloc((size_t)m->n_layers, q->conv_stride * sizeof(float));
		q->recurrent_state	= xcalloc((size_t)m->n_layers, q->recurrent_stride * sizeof(float));
		c->hybrid			= q;
	}

	backend *kv_backend = c->backend->kv_alloc ? c->backend : backend_host();

	if (kv_quant == KV_QUANT_Q8_0 && !backend_has_cap(kv_backend, BCAP_KV_QUANT_Q8_0)) {
		ERROR("kvcache: backend '%s' does not support Q8_0 quantized KV cache; "
			  "use --kv-quant f16 or select a backend that supports it",
			  kv_backend->name);
		return ERR_UNSUPPORTED;
	}

	int *layer_head_dim	  = NULL;
	int *layer_n_kv_heads = NULL;
	if (m->arch_info->has_variable_layer_dims) {
		layer_head_dim	 = xcalloc(m->n_layers, sizeof(int));
		layer_n_kv_heads = xcalloc(m->n_layers, sizeof(int));
		for (int i = 0; i < m->n_layers; i++) {
			layer_head_dim[i]	= model_layer_head_dim(m, i);
			layer_n_kv_heads[i] = model_layer_kv_heads(m, i);
		}
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
	};
	status_code s = kv_backend->kv_alloc(kv_backend, &desc, &c->k, &c->v);
	free(layer_head_dim);
	free(layer_n_kv_heads);
	return s;
}

void kvcache_free(kvcache *c) {
	if (!c->backend)
		return;
	if (c->mla) {
		if (c->mla->kv.owner)
			c->mla->kv.owner->buffer_free(c->mla->kv.owner, &c->mla->kv);
		if (c->mla->host_alloced && c->mla->kv_host.owner)
			c->mla->kv_host.owner->buffer_free(c->mla->kv_host.owner, &c->mla->kv_host);
		free(c->mla);
	} else {
		if (c->backend->kv_free) {
			c->backend->kv_free(c->backend, &c->k, &c->v);
		} else {
			if (c->k.owner)
				c->k.owner->buffer_free(c->k.owner, &c->k);
			if (c->v.owner)
				c->v.owner->buffer_free(c->v.owner, &c->v);
		}
		if (c->has_host_kv) {
			backend *host = backend_host();
			if (host && host->kv_free) {
				host->kv_free(host, &c->k_host, &c->v_host);
			} else {
				if (c->k_host.owner)
					c->k_host.owner->buffer_free(c->k_host.owner, &c->k_host);
				if (c->v_host.owner)
					c->v_host.owner->buffer_free(c->v_host.owner, &c->v_host);
			}
			c->has_host_kv = 0;
		}
	}
	if (c->hybrid) {
		free(c->hybrid->conv_state);
		free(c->hybrid->recurrent_state);
		free(c->hybrid);
		c->hybrid = NULL;
	}
	free(c->kv_slot_on_host);
	c->kv_slot_on_host = NULL;
	c->n_kv_slot_flags = 0;
	free(c->kv_transfer_buf);
	c->kv_transfer_buf = NULL;
	c->kv_transfer_cap = 0;
	memset(c, 0, sizeof(*c));
}

status_code kvcache_alloc_host_mirror(kvcache *c, const model *m) {
	if (!c || !m || !m->mixed_backend_mode)
		return OK;
	if (c->has_host_kv)
		return OK;

	backend *host = backend_host();
	if (!host)
		return ERR_UNSUPPORTED;

	if (m->arch_info->is_mla) {
		if (!host->kv_alloc_mla) {
			ERROR("kvcache: host backend has no kv_alloc_mla; cannot mirror MLA KV");
			return ERR_UNSUPPORTED;
		}
		if (!c->mla) {
			ERROR("kvcache: MLA state missing; cannot mirror KV");
			return ERR_INVALID_ARG;
		}
		status_code s = host->kv_alloc_mla(host, m->n_layers, c->n_ctx, c->mla->lora_dim,
										   c->mla->qk_rope_dim, &c->mla->kv_host);
		if (s != OK)
			return s;
		c->mla->host_alloced = 1;
		c->has_host_kv		 = 1;
	} else {
		if (!host->kv_alloc) {
			ERROR("kvcache: host backend has no kv_alloc; cannot mirror KV");
			return ERR_UNSUPPORTED;
		}
		if (c->kv_quant == KV_QUANT_Q8_0 && !backend_has_cap(host, BCAP_KV_QUANT_Q8_0)) {
			ERROR("kvcache: host backend does not support Q8_0 quantized KV cache; "
				  "use --kv-quant f16 or run all layers on the accelerator backend");
			return ERR_UNSUPPORTED;
		}

		int *layer_head_dim	  = NULL;
		int *layer_n_kv_heads = NULL;
		if (m->arch_info->has_variable_layer_dims) {
			layer_head_dim	 = xcalloc(m->n_layers, sizeof(int));
			layer_n_kv_heads = xcalloc(m->n_layers, sizeof(int));
			for (int i = 0; i < m->n_layers; i++) {
				layer_head_dim[i]	= model_layer_head_dim(m, i);
				layer_n_kv_heads[i] = model_layer_kv_heads(m, i);
			}
		}

		kv_desc desc = {
			.n_layers		  = m->n_layers,
			.n_kv_layers	  = c->n_kv_layers,
			.n_kv_heads		  = c->n_kv_heads_max,
			.head_dim		  = c->head_dim_max,
			.n_ctx			  = c->n_ctx,
			.kv_quant		  = c->kv_quant,
			.layer_head_dim	  = layer_head_dim,
			.layer_n_kv_heads = layer_n_kv_heads,
		};
		status_code s = host->kv_alloc(host, &desc, &c->k_host, &c->v_host);
		free(layer_head_dim);
		free(layer_n_kv_heads);
		if (s != OK)
			return s;
		c->has_host_kv = 1;
	}

	if (!c->kv_slot_on_host) {
		c->kv_slot_on_host = xcalloc((size_t)m->n_layers, sizeof(int));
		c->n_kv_slot_flags = m->n_layers;
	}
	for (int i = 0; i < m->n_layers; i++) {
		backend *lb = model_layer_backend(m, i);
		if (lb && lb != m->backend && backend_has_cap(lb, BCAP_IS_HOST)) {
			int slot = i;
			if (m->recipe && m->recipe->layer_ctx)
				slot = m->recipe->layer_ctx[i].kv_layer;
			if (slot >= 0 && slot < m->n_layers)
				c->kv_slot_on_host[slot] = 1;
			c->kv_slot_on_host[i] = 1;
		}
	}
	int n_mirrored = 0;
	for (int i = 0; i < m->n_layers; i++)
		if (c->kv_slot_on_host[i])
			n_mirrored++;
	INFO("mixed backend KV mirror: %d of %d KV slot(s) on host", n_mirrored, m->n_layers);
	return OK;
}

void kvcache_reset(kvcache *c) {
	c->n_pos = 0;
	if (c->hybrid) {
		memset(c->hybrid->conv_state, 0, c->hybrid->conv_total * sizeof(float));
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

status_code kvcache_put(kvcache *c, const model *m, int layer, int pos, const buffer *k_in,
						const buffer *v_in) {
	if (pos < 0 || pos >= c->n_ctx)
		return ERR_INVALID_ARG;
	if (layer < 0 || layer >= m->n_layers)
		return ERR_INVALID_ARG;
	int		 hd				 = model_layer_head_dim(m, layer);
	int		 kvh_stride		 = kvcache_kv_heads_stride(c);
	int		 kvh_active		 = model_layer_kv_heads(m, layer);
	int		 layer_on_host	 = kvcache_layer_uses_host_kv(c, m, layer);
	int		 slot_needs_host = kvcache_slot_on_host(c, layer);
	buffer	*kb				 = layer_on_host ? &c->k_host : &c->k;
	buffer	*vb				 = layer_on_host ? &c->v_host : &c->v;
	backend *kv_backend = kb->owner ? kb->owner : (layer_on_host ? backend_host() : c->backend);
	backend *k_in_owner = k_in->owner ? k_in->owner : kv_backend;
	backend *v_in_owner = v_in->owner ? v_in->owner : kv_backend;

	if (k_in_owner != kv_backend || v_in_owner != kv_backend) {
		if (k_in_owner && k_in_owner->synchronize)
			k_in_owner->synchronize(k_in_owner);
		if (v_in_owner && v_in_owner != k_in_owner && v_in_owner->synchronize)
			v_in_owner->synchronize(v_in_owner);

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
		return kv_backend->kv_put(kv_backend, kb, vb, layer, pos, &k_host_buf, &v_host_buf,
								  kvh_stride, hd, c->n_ctx, kvh_active);
	}

	if (k_in_owner != c->backend && c->backend->synchronize)
		c->backend->synchronize(c->backend);
	status_code st = kv_backend->kv_put(kv_backend, kb, vb, layer, pos, k_in, v_in, kvh_stride, hd,
										c->n_ctx, kvh_active);

	if (st == OK && !layer_on_host && slot_needs_host && c->has_host_kv) {
		backend *host = backend_host();
		if (host && c->k_host.owner && c->v_host.owner && host->kv_put) {
			int			k_floats = kvh_active * hd;
			status_code ts		 = kvcache_ensure_transfer_buf(c, (size_t)k_floats * 2);
			if (ts != OK)
				return ts;
			buffer k_host_buf, v_host_buf;
			kv_host_pair_bufs(c, k_floats, &k_host_buf, &v_host_buf);
			if (kv_backend->buffer_read_f32) {
				ts = kv_backend->buffer_read_f32(kv_backend, k_in, c->kv_transfer_buf, k_floats);
				if (ts != OK)
					return ts;
				ts = kv_backend->buffer_read_f32(kv_backend, v_in, (float *)v_host_buf.handle,
												 k_floats);
				if (ts != OK)
					return ts;
			}
			return host->kv_put(host, &c->k_host, &c->v_host, layer, pos, &k_host_buf, &v_host_buf,
								kvh_stride, hd, c->n_ctx, kvh_active);
		}
	}
	return st;
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

	int hd				= model_layer_head_dim(m, layer);
	int kvh_stride		= kvcache_kv_heads_stride(c);
	int kvh_active		= model_layer_kv_heads(m, layer);
	int slot_needs_host = kvcache_slot_on_host(c, layer);

	backend *kv_backend = c->k.owner ? c->k.owner : c->backend;
	backend *k_owner	= k_in->owner ? k_in->owner : kv_backend;
	backend *v_owner	= v_in->owner ? v_in->owner : kv_backend;

	bool same_backend = (k_owner == kv_backend && v_owner == kv_backend);
	bool fast_ok	  = same_backend && kv_backend->kv_put_batch != NULL &&
						!kvcache_layer_uses_host_kv(c, m, layer) && !slot_needs_host;
	if (fast_ok) {
		return kv_backend->kv_put_batch(kv_backend, &c->k, &c->v, layer, pos_start, k_in, v_in,
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
