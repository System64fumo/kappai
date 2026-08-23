#include "compute.h"
#include "backend/cpu/scalar/quants.h"
#include "log.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void compute_scratch_init(compute_scratch *s) {
	memset(s, 0, sizeof(*s));
}

static void compute_scratch_set_router_bufs(compute_scratch *s) {
	buffer *rid	  = &s->slots[RECIPE_SLOT_ROUTER_IDS];
	buffer *rw	  = &s->slots[RECIPE_SLOT_ROUTER_W];
	rid->handle	  = s->router_ids_host;
	rid->host_ptr = s->router_ids_host;
	rid->size	  = sizeof(s->router_ids_host);
	rid->owner	  = NULL;
	rw->handle	  = s->router_w_host;
	rw->host_ptr  = s->router_w_host;
	rw->size	  = sizeof(s->router_w_host);
	rw->owner	  = NULL;
}

typedef struct {
	float *cos_out;
	float *sin_out;
	int	   n_ctx;
	int	   half;
	double theta_scale;
} rope_table_job;

#define ROPE_TABLE_BLOCK 4096

static void rope_table_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	rope_table_job *j = ctx;
	for (int col = begin; col < end; col++) {
		double freq		= pow(j->theta_scale, (double)col);
		double step_cos = cos(freq);
		double step_sin = sin(freq);
		float *cos_col	= j->cos_out + col;
		float *sin_col	= j->sin_out + col;
		for (int base = 0; base < j->n_ctx; base += ROPE_TABLE_BLOCK) {
			int	   stop	 = base + ROPE_TABLE_BLOCK > j->n_ctx ? j->n_ctx : base + ROPE_TABLE_BLOCK;
			double angle = freq * (double)base;
			double c	 = cos(angle);
			double s	 = sin(angle);
			cos_col[(ptrdiff_t)base * j->half] = (float)c;
			sin_col[(ptrdiff_t)base * j->half] = (float)s;
			for (int pos = base + 1; pos < stop; pos++) {
				double nc						  = (c * step_cos) - (s * step_sin);
				s								  = (c * step_sin) + (s * step_cos);
				c								  = nc;
				cos_col[(ptrdiff_t)pos * j->half] = (float)c;
				sin_col[(ptrdiff_t)pos * j->half] = (float)s;
			}
		}
	}
}

static void build_rope_table(backend *a, float *cos_out, float *sin_out, int n_ctx, int head_dim,
							 float theta) {
	const int	   half		   = head_dim / 2;
	const double   theta_scale = pow(theta, -2.0 / (double)head_dim);
	rope_table_job job		   = {.cos_out	   = cos_out,
								  .sin_out	   = sin_out,
								  .n_ctx	   = n_ctx,
								  .half		   = half,
								  .theta_scale = theta_scale};

	tpool *pool = (a && a->get_pool) ? a->get_pool(a) : NULL;
	if (pool && tpool_n_threads(pool) > 1 && half >= tpool_n_threads(pool) &&
		tpool_current_tid() < 0) {
		tpool_parallel_for(pool, half, 1, rope_table_chunk, &job);
		return;
	}
	rope_table_chunk(0, half, -1, &job);
}

static void free_buf(buffer *b) {
	if (!b->owner)
		return;
	b->owner->buffer_free(b->owner, b);
}

static void scratch_free_device_buffers(compute_scratch *s) {
	free_buf(&s->slots[RECIPE_SLOT_X]);
	free_buf(&s->slots[RECIPE_SLOT_XB]);
	free_buf(&s->slots[RECIPE_SLOT_XB2]);
	free_buf(&s->slots[RECIPE_SLOT_ATTN_OUT]);
	free_buf(&s->slots[RECIPE_SLOT_Q]);
	free_buf(&s->slots[RECIPE_SLOT_K]);
	free_buf(&s->slots[RECIPE_SLOT_V]);
	free_buf(&s->slots[RECIPE_SLOT_FFN_GATE_UP]);
	free_buf(&s->slots[RECIPE_SLOT_FFN_ACT]);
	free_buf(&s->slots[RECIPE_SLOT_LOGITS]);
	free_buf(&s->slots[RECIPE_SLOT_HYB_PROJ]);
	free_buf(&s->slots[RECIPE_SLOT_HYB_GATE]);
	free_buf(&s->slots[RECIPE_SLOT_HYB_ALPHA]);
	free_buf(&s->slots[RECIPE_SLOT_HYB_BETA]);
	free_buf(&s->ple_inp);
	free_buf(&s->ple_slice);
	free_buf(&s->ple_all);
	free_buf(&s->ple_proj_gpu);
	free_buf(&s->ple_proj_norm_w_gpu);
	free_buf(&s->router_softmax_inp_gpu);
	free_buf(&s->router_logits_gpu);

	free_buf(&s->mirror_slots[RECIPE_SLOT_X]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_XB]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_XB2]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_ATTN_OUT]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_Q]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_K]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_V]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_FFN_GATE_UP]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_FFN_ACT]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_LOGITS]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_HYB_PROJ]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_HYB_GATE]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_HYB_ALPHA]);
	free_buf(&s->mirror_slots[RECIPE_SLOT_HYB_BETA]);
	for (int i = 0; i < RECIPE_SLOT_MAX; i++) {
		if (i == RECIPE_SLOT_FFN_GATE || i == RECIPE_SLOT_FFN_UP)
			continue;
		s->mirror_slots[i].owner	= NULL;
		s->mirror_slots[i].handle	= NULL;
		s->mirror_slots[i].host_ptr = NULL;
		s->mirror_slots[i].size		= 0;
		s->mirror_slots[i].offset	= 0;
	}
	s->mirror_slots_alloced = 0;
	s->mirror_backend		= NULL;
	s->active_backend		= NULL;
	s->active_is_mirror		= 0;
	free(s->transfer_buf);
	s->transfer_buf		= NULL;
	s->transfer_buf_cap = 0;
	if (!s->logits_alias)
		free(s->logits_host);
	s->logits_host	= NULL;
	s->logits_alias = 0;
}

static void scratch_free_host_buffers(compute_scratch *s) {
	free(s->rope_cos);
	free(s->rope_sin);
	free(s->rope_cos_swa);
	free(s->rope_sin_swa);
	free(s->ple_buf);
	free(s->ple_inp_host);
	s->ple_inp_host = NULL;
	free(s->cur_host);
	s->cur_host = NULL;
	free(s->out_proj_host);
	s->out_proj_host = NULL;
	free(s->vf_host);
	s->vf_host = NULL;
	free(s->xf_host);
	s->xf_host				   = NULL;
	s->small_host_dim		   = 0;
	s->small_host_intermediate = 0;
	s->small_host_kv_out	   = 0;
	free(s->ple_proj_host.p);
	free(s->inpL_host.p);
	s->ple_proj_host.p	 = NULL;
	s->ple_proj_host.cap = 0;
	s->inpL_host.p		 = NULL;
	s->inpL_host.cap	 = 0;
	free(s->moe_all_scratch.p);
	free(s->moe_all_outs.p);
	free(s->moe_scratch.p);
	free(s->moe_xb_f.p);
	free(s->moe_shared_y.p);
	free(s->hybrid_host.p);
	s->moe_all_scratch.p = NULL;
	s->moe_all_outs.p	 = NULL;
	s->moe_scratch.p	 = NULL;
	s->moe_xb_f.p		 = NULL;
	s->moe_shared_y.p	 = NULL;
	s->hybrid_host.p	 = NULL;
	s->hybrid_host.cap	 = 0;

	if (s->bs) {
		batch_scratch_free(s->bs);
		free(s->bs);
		s->bs = NULL;
	}
	free(s->batch_logits_tmp.p);
	s->batch_logits_tmp.cap = 0;
}

void compute_scratch_free(compute_scratch *s) {
	scratch_free_device_buffers(s);
	scratch_free_host_buffers(s);
	memset(s, 0, sizeof(*s));
}

void compute_small_host_ensure(compute_scratch *s, int dim, int intermediate, int kv_out) {
	if (s->small_host_dim >= dim && s->small_host_intermediate >= intermediate &&
		s->small_host_kv_out >= kv_out)
		return;

	int new_dim = s->small_host_dim < dim ? dim : s->small_host_dim;
	int new_int =
		s->small_host_intermediate < intermediate ? intermediate : s->small_host_intermediate;
	int new_kv				   = s->small_host_kv_out < kv_out ? kv_out : s->small_host_kv_out;
	s->ple_inp_host			   = xrealloc(s->ple_inp_host, (size_t)new_int * sizeof(float));
	s->cur_host				   = xrealloc(s->cur_host, (size_t)new_dim * sizeof(float));
	s->out_proj_host		   = xrealloc(s->out_proj_host, (size_t)new_dim * sizeof(float));
	s->vf_host				   = xrealloc(s->vf_host, (size_t)new_kv * sizeof(float));
	s->xf_host				   = xrealloc(s->xf_host, (size_t)new_dim * sizeof(float));
	s->small_host_dim		   = new_dim;
	s->small_host_intermediate = new_int;
	s->small_host_kv_out	   = new_kv;
}

static int layer_max_kv_heads(const model *m) {
	int best = 0;
	for (int i = 0; i < m->n_layers; i++) {
		int v = model_layer_kv_heads(m, i);
		if (v > best)
			best = v;
	}
	return best;
}

static int layer_max_intermediate(const model *m) {
	int best = 0;
	for (int i = 0; i < m->n_layers; i++) {
		int v = model_layer_intermediate(m, i);
		if (v > best)
			best = v;
	}
	return best;
}

static status_code scratch_alloc(backend *a, buffer *dst, size_t size, const char *name) {
	status_code st = a->buffer_alloc_scratch(a, size, dst);
	if (st != OK) {
		ERROR(
			"compute_scratch_ensure: failed to allocate scratch buffer '%s' (%zu bytes, status=%d)",
			name, size, st);
	}
	return st;
}

typedef struct {
	int attn_buf_size;
	int q_out;
	int kv_out;
	int max_head_dim;
	int max_kv_heads;
	int max_intermediate;
	int ffn_act_size;
} compute_layout;

static void compute_layout_init(const model *m, compute_layout *L) {
	L->max_head_dim = m->head_dim;
	L->max_kv_heads = m->n_kv_heads;
	if (m->arch_info->has_variable_layer_dims) {
		L->max_head_dim = MAX(m->layer_dims.head_dim_global, m->layer_dims.head_dim_swa);
		L->max_kv_heads = layer_max_kv_heads(m);
		if (L->max_kv_heads == 0)
			L->max_kv_heads = m->n_kv_heads;
	}

	L->attn_buf_size = m->dim;
	if (m->arch_info->has_variable_layer_dims) {
		int q_out_max = m->n_heads * L->max_head_dim;
		if (q_out_max > L->attn_buf_size)
			L->attn_buf_size = q_out_max;
	}
	if (m->arch_info->is_mla) {
		int mla_out = m->n_heads * m->mla.v_head;
		if (mla_out > L->attn_buf_size)
			L->attn_buf_size = mla_out;
	}
	if (m->arch_info->is_hybrid_recurrent && m->hybrid.value_dim > L->attn_buf_size)
		L->attn_buf_size = m->hybrid.value_dim;

	L->q_out  = m->n_heads * L->max_head_dim;
	L->kv_out = L->max_kv_heads * L->max_head_dim;

	L->max_intermediate = m->intermediate;
	if (m->arch_info->has_variable_layer_dims) {
		int layer_max = layer_max_intermediate(m);
		if (layer_max > L->max_intermediate)
			L->max_intermediate = layer_max;
	}
	L->ffn_act_size = L->max_intermediate > m->dim ? L->max_intermediate : m->dim;
}

status_code compute_scratch_ensure(compute_scratch *s, const model *m, int n_ctx) {
	if (!compute_model_changed(s, m, n_ctx))
		return OK;

	scratch_free_device_buffers(s);
	scratch_free_host_buffers(s);

	s->backend = m->backend;
	backend *a = s->backend;

	compute_layout L;
	compute_layout_init(m, &L);

	status_code _st;
	_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_X], (size_t)L.attn_buf_size * sizeof(float), "X");
	if (_st != OK)
		return _st;
	if (m->arch_info->is_hybrid_recurrent) {
		_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_HYB_PROJ],
							(size_t)model_hybrid_proj_size(m) * sizeof(float), "hybrid projection");
		if (_st != OK)
			return _st;
		_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_HYB_GATE],
							(size_t)model_hybrid_gate_size(m) * sizeof(float), "hybrid gate");
		if (_st != OK)
			return _st;
		_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_HYB_ALPHA],
							(size_t)m->hybrid.n_value_heads * sizeof(float), "hybrid alpha");
		if (_st != OK)
			return _st;
		_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_HYB_BETA],
							(size_t)m->hybrid.n_value_heads * sizeof(float), "hybrid beta");
		if (_st != OK)
			return _st;
	}
	_st =
		scratch_alloc(a, &s->slots[RECIPE_SLOT_XB], (size_t)L.attn_buf_size * sizeof(float), "XB");
	if (_st != OK)
		return _st;
	_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_XB2], (size_t)L.attn_buf_size * sizeof(float),
						"XB2");
	if (_st != OK)
		return _st;
	_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_ATTN_OUT], (size_t)L.attn_buf_size * sizeof(float),
						"ATTN_OUT");
	if (_st != OK)
		return _st;
	_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_Q], (size_t)L.q_out * sizeof(float), "Q");
	if (_st != OK)
		return _st;
	_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_K], (size_t)L.kv_out * sizeof(float), "K");
	if (_st != OK)
		return _st;
	_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_V], (size_t)L.kv_out * sizeof(float), "V");
	if (_st != OK)
		return _st;
	_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_FFN_GATE_UP],
						(size_t)L.max_intermediate * 2 * sizeof(float), "FFN_GATE_UP");
	if (_st != OK)
		return _st;
	s->slots[RECIPE_SLOT_FFN_GATE] = buffer_slice(&s->slots[RECIPE_SLOT_FFN_GATE_UP], 0,
												  (size_t)L.max_intermediate * sizeof(float));
	s->slots[RECIPE_SLOT_FFN_UP] =
		buffer_slice(&s->slots[RECIPE_SLOT_FFN_GATE_UP], (size_t)L.max_intermediate * sizeof(float),
					 (size_t)L.max_intermediate * sizeof(float));
	_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_FFN_ACT], (size_t)L.ffn_act_size * sizeof(float),
						"FFN_ACT");
	if (_st != OK)
		return _st;
	_st = scratch_alloc(a, &s->slots[RECIPE_SLOT_LOGITS], (size_t)m->vocab_size * sizeof(float),
						"LOGITS");
	if (_st != OK)
		return _st;

	if (backend_has_cap(a, BCAP_IS_HOST)) {
		buffer *lb		= &s->slots[RECIPE_SLOT_LOGITS];
		s->logits_host	= (float *)((char *)lb->handle + lb->offset);
		s->logits_alias = 1;
	} else {
		s->logits_host	= xmalloc((size_t)m->vocab_size * sizeof(float));
		s->logits_alias = 0;
	}

	compute_scratch_set_router_bufs(s);

	if (m->arch_info->has_variable_layer_dims) {
		const int half_swa = m->layer_dims.head_dim_swa / 2;
		s->rope_cos_swa	   = xmalloc((size_t)n_ctx * half_swa * sizeof(float));
		s->rope_sin_swa	   = xmalloc((size_t)n_ctx * half_swa * sizeof(float));
		build_rope_table(a, s->rope_cos_swa, s->rope_sin_swa, n_ctx, m->layer_dims.head_dim_swa,
						 m->layer_dims.rope_theta_swa);

		const int half_global = m->layer_dims.head_dim_global / 2;
		s->rope_cos			  = xmalloc((size_t)n_ctx * half_global * sizeof(float));
		s->rope_sin			  = xmalloc((size_t)n_ctx * half_global * sizeof(float));
		build_rope_table(a, s->rope_cos, s->rope_sin, n_ctx, m->layer_dims.head_dim_global,
						 m->layer_dims.rope_theta_global);

		if (m->has_per_layer_embeddings) {
			s->ple_buf =
				xmalloc((size_t)m->layer_dims.n_embd_per_layer * m->n_layers * sizeof(float));
			{
				status_code _st = scratch_alloc(
					a, &s->ple_inp, (size_t)m->layer_dims.n_embd_per_layer * sizeof(float),
					"&s->ple_inp");
				if (_st != OK)
					return _st;
			}
			{
				status_code _st = scratch_alloc(
					a, &s->ple_slice, (size_t)m->layer_dims.n_embd_per_layer * sizeof(float),
					"&s->ple_slice");
				if (_st != OK)
					return _st;
			}
		}
	} else {
		const int rope_head_dim =
			m->arch_info->is_mla ? m->mla.qk_rope
								 : (m->arch_info->is_hybrid_recurrent ? m->rope_dim : m->head_dim);
		const int half = rope_head_dim / 2;
		s->rope_cos	   = xmalloc((size_t)n_ctx * half * sizeof(float));
		s->rope_sin	   = xmalloc((size_t)n_ctx * half * sizeof(float));
		build_rope_table(a, s->rope_cos, s->rope_sin, n_ctx, rope_head_dim, m->rope_theta);
	}

	s->allocated_n_ctx = n_ctx;
	s->last_model	   = m;
	s->last_n_ctx	   = n_ctx;

	return OK;
}
status_code compute_forward(model *m, kvcache *cache, compute_scratch *s, int token, int pos,
							int flash_attn, float *logits_out) {
	status_code st = compute_scratch_ensure(s, m, cache->n_ctx);
	if (st != OK)
		return st;
	return compute_forward_recipe(m, cache, s, token, pos, flash_attn, logits_out);
}

status_code compute_forward_batch(model *m, kvcache *cache, compute_scratch *s,
								  const int32_t *tokens, int n_tokens, int pos_start,
								  int flash_attn, float *logits_out) {
	status_code st = compute_scratch_ensure(s, m, cache->n_ctx);
	if (st != OK)
		return st;
	return compute_forward_batch_recipe(m, cache, s, tokens, n_tokens, pos_start, flash_attn,
										logits_out);
}

void compute_set_layer_progress_cb(compute_scratch *s, layer_progress_cb cb, void *ud) {
	s->layer_cb	   = cb;
	s->layer_cb_ud = ud;
}

static status_code ensure_transfer_buf(compute_scratch *s, size_t need_floats) {
	if (s->transfer_buf_cap >= need_floats)
		return OK;
	s->transfer_buf		= xrealloc(s->transfer_buf, need_floats * sizeof(float));
	s->transfer_buf_cap = need_floats;
	return OK;
}

status_code compute_scratch_ensure_mirror(compute_scratch *s, const model *m, int n_ctx) {
	(void)n_ctx;
	if (s->mirror_slots_alloced && s->mirror_backend)
		return OK;
	if (!m || !m->mixed_backend_mode)
		return OK;
	backend *host = backend_host();
	if (!host)
		return ERR_UNSUPPORTED;

	s->mirror_backend = host;

	compute_layout L;
	compute_layout_init(m, &L);

	status_code st;
#define MIRROR_ALLOC(slot, size)                                                                   \
	do {                                                                                           \
		st = scratch_alloc(host, &s->mirror_slots[slot], (size), "mirror_slots[" #slot "]");       \
		if (st != OK)                                                                              \
			return st;                                                                             \
	} while (0)

	MIRROR_ALLOC(RECIPE_SLOT_X, (size_t)L.attn_buf_size * sizeof(float));
	if (m->arch_info->is_hybrid_recurrent) {
		MIRROR_ALLOC(RECIPE_SLOT_HYB_PROJ, (size_t)model_hybrid_proj_size(m) * sizeof(float));
		MIRROR_ALLOC(RECIPE_SLOT_HYB_GATE, (size_t)model_hybrid_gate_size(m) * sizeof(float));
		MIRROR_ALLOC(RECIPE_SLOT_HYB_ALPHA, (size_t)m->hybrid.n_value_heads * sizeof(float));
		MIRROR_ALLOC(RECIPE_SLOT_HYB_BETA, (size_t)m->hybrid.n_value_heads * sizeof(float));
	}
	MIRROR_ALLOC(RECIPE_SLOT_XB, (size_t)L.attn_buf_size * sizeof(float));
	MIRROR_ALLOC(RECIPE_SLOT_XB2, (size_t)L.attn_buf_size * sizeof(float));
	MIRROR_ALLOC(RECIPE_SLOT_ATTN_OUT, (size_t)L.attn_buf_size * sizeof(float));
	MIRROR_ALLOC(RECIPE_SLOT_Q, (size_t)L.q_out * sizeof(float));
	MIRROR_ALLOC(RECIPE_SLOT_K, (size_t)L.kv_out * sizeof(float));
	MIRROR_ALLOC(RECIPE_SLOT_V, (size_t)L.kv_out * sizeof(float));
	MIRROR_ALLOC(RECIPE_SLOT_FFN_GATE_UP, (size_t)L.max_intermediate * 2 * sizeof(float));
	s->mirror_slots[RECIPE_SLOT_FFN_GATE] = buffer_slice(
		&s->mirror_slots[RECIPE_SLOT_FFN_GATE_UP], 0, (size_t)L.max_intermediate * sizeof(float));
	s->mirror_slots[RECIPE_SLOT_FFN_UP] = buffer_slice(&s->mirror_slots[RECIPE_SLOT_FFN_GATE_UP],
													   (size_t)L.max_intermediate * sizeof(float),
													   (size_t)L.max_intermediate * sizeof(float));
	MIRROR_ALLOC(RECIPE_SLOT_FFN_ACT, (size_t)L.ffn_act_size * sizeof(float));
	MIRROR_ALLOC(RECIPE_SLOT_LOGITS, (size_t)m->vocab_size * sizeof(float));
#undef MIRROR_ALLOC

	s->mirror_slots[RECIPE_SLOT_ROUTER_IDS].handle	 = s->router_ids_host;
	s->mirror_slots[RECIPE_SLOT_ROUTER_IDS].host_ptr = s->router_ids_host;
	s->mirror_slots[RECIPE_SLOT_ROUTER_IDS].size	 = sizeof(s->router_ids_host);
	s->mirror_slots[RECIPE_SLOT_ROUTER_IDS].offset	 = 0;
	s->mirror_slots[RECIPE_SLOT_ROUTER_IDS].owner	 = NULL;
	s->mirror_slots[RECIPE_SLOT_ROUTER_W].handle	 = s->router_w_host;
	s->mirror_slots[RECIPE_SLOT_ROUTER_W].host_ptr	 = s->router_w_host;
	s->mirror_slots[RECIPE_SLOT_ROUTER_W].size		 = sizeof(s->router_w_host);
	s->mirror_slots[RECIPE_SLOT_ROUTER_W].offset	 = 0;
	s->mirror_slots[RECIPE_SLOT_ROUTER_W].owner		 = NULL;

	s->mirror_slots_alloced = 1;
	return OK;
}

status_code compute_switch_active_backend(compute_scratch *s, backend *target, int dim) {
	if (!target)
		target = s->backend;
	backend *current = compute_active_backend(s);
	if (target == current) {
		s->active_backend	= current;
		s->active_is_mirror = (current == s->mirror_backend);
		return OK;
	}

	if (target != s->backend && target != s->mirror_backend) {
		ERROR("compute_switch_active_backend: target backend '%s' is neither primary nor mirror",
			  target->name);
		return ERR_UNSUPPORTED;
	}

	buffer *src_x =
		s->active_is_mirror ? &s->mirror_slots[RECIPE_SLOT_X] : &s->slots[RECIPE_SLOT_X];
	buffer *dst_x =
		(target == s->mirror_backend) ? &s->mirror_slots[RECIPE_SLOT_X] : &s->slots[RECIPE_SLOT_X];

	backend *src_be = current;
	if (src_be && src_be->synchronize)
		src_be->synchronize(src_be);

	status_code st = compute_copy_buffer_cross(s, src_x, dst_x, dim);
	if (st != OK)
		return st;

	s->active_backend	= target;
	s->active_is_mirror = (target == s->mirror_backend);
	return OK;
}

status_code compute_copy_buffer_cross(compute_scratch *s, const buffer *src, buffer *dst, int n) {
	if (!src || !dst || n <= 0)
		return ERR_INVALID_ARG;

	backend *src_owner = src->owner;
	backend *dst_owner = dst->owner;

	if (src_owner == dst_owner) {
		if (src_owner && src_owner->copy_buffer)
			return src_owner->copy_buffer(src_owner, src, dst, n);
		if (src_owner && src_owner->buffer_read_f32 && src_owner->buffer_write_f32) {
			status_code st = ensure_transfer_buf(s, (size_t)n);
			if (st != OK)
				return st;
			st = src_owner->buffer_read_f32(src_owner, src, s->transfer_buf, n);
			if (st != OK)
				return st;
			return src_owner->buffer_write_f32(src_owner, dst, s->transfer_buf, n);
		}
		if (src->host_ptr && dst->host_ptr) {
			memcpy((void *)dst->host_ptr, src->host_ptr, (size_t)n * sizeof(float));
			return OK;
		}
		return ERR_UNSUPPORTED;
	}

	if (src_owner && src_owner->synchronize)
		src_owner->synchronize(src_owner);

	status_code st = ensure_transfer_buf(s, (size_t)n);
	if (st != OK)
		return st;

	if (src_owner && src_owner->buffer_read_f32) {
		st = src_owner->buffer_read_f32(src_owner, src, s->transfer_buf, n);
		if (st != OK)
			return st;
	} else if (src->host_ptr) {
		memcpy(s->transfer_buf, src->host_ptr, (size_t)n * sizeof(float));
	} else {
		return ERR_UNSUPPORTED;
	}

	if (dst_owner && dst_owner->buffer_write_f32) {
		return dst_owner->buffer_write_f32(dst_owner, dst, s->transfer_buf, n);
	} else if (dst->host_ptr) {
		memcpy((void *)dst->host_ptr, s->transfer_buf, (size_t)n * sizeof(float));
		return OK;
	}
	return ERR_UNSUPPORTED;
}