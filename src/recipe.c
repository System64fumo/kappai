#include "recipe.h"
#include "backend/backend.h"
#include "backend/cpu/scalar/quants.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "moe/moe_common.h"
#include "moe/moe_stream.h"
#include "monitor.h"

#include <assert.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RECIPE_MAX_ARCHES 32
#define RECIPE_COALESCE_MAX 4
#define MLA_Q_A_STACK_CAP 4096
#define MOE_MAX_DIM_STACK 8192
#define MLA_KV_A_STACK_CAP 2048
static inline void recipe_unused(int dummy, ...) {
	(void)dummy;
}
#define UNUSED(...) recipe_unused(0, __VA_ARGS__)

static backend *cached_host_backend = NULL;

static inline backend *op_host_backend(void) {
	backend *b = cached_host_backend;
	if (!b) {
		b					= backend_host();
		cached_host_backend = b;
	}
	return b;
}
#define OP_BACKEND(field) ((a->field) ? a : (a == op_host_backend() ? a : op_backend_fallback(a)))
static status_code	 exec_op(const recipe_op *op, struct model *m, struct kvcache *cache,
							 struct compute_scratch *s, int token, int pos, int li, int flash_attn,
							 float *logits_out);
static buffer		*batch_slot(batch_scratch *bs, uint8_t slot);
static buffer		 batch_row_view(const buffer *whole, int row, int row_elems);
static inline float *batch_buf_ptr(const buffer *b);
static status_code	 ple_build_batch(exec_ctx *ctx);
static status_code	 ple_proj_inject_batch(exec_ctx *ctx);
static status_code	 matmul_multi_batch_body(exec_ctx *ctx);
static void			 moe_batch_expert_chunk(int begin, int end, int tid, void *ctx);
static status_code	 batch_copy_k_to_v(model *m, batch_scratch *bs, int li, int n_tokens);
static uint32_t		 op_batch_slot_mask(const recipe_op *op);

static inline buffer *exec_slot(const exec_ctx *ctx, uint8_t idx) {
	if (ctx->bs)
		return batch_slot(ctx->bs, idx);
	if (ctx->s && ctx->s->active_is_mirror)
		return &ctx->s->mirror_slots[idx];
	return &ctx->s->slots[idx];
}

float *recipe_slot_f32(const exec_ctx *ctx, uint8_t idx) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return NULL;
	if (b->handle)
		return (float *)((char *)b->handle + b->offset);
	return (float *)b->host_ptr;
}

static inline int slot_is_host_resident(const buffer *b) {
	if (!b->handle)
		return 1;
	return b->owner && backend_has_cap(b->owner, BCAP_IS_HOST);
}

const float *recipe_slot_read_f32(const exec_ctx *ctx, uint8_t idx, float_buf *stage, int n) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return NULL;
	if (slot_is_host_resident(b))
		return (const float *)(b->handle ? (char *)b->handle + b->offset : b->host_ptr);
	float *dst = float_buf_ensure_nocopy(stage, (size_t)n, 64);
	if (!dst || b->owner->buffer_read_f32(b->owner, b, dst, n) != OK)
		return NULL;
	return dst;
}

float *recipe_slot_write_stage(const exec_ctx *ctx, uint8_t idx, float_buf *stage, int n) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return NULL;
	if (slot_is_host_resident(b))
		return (float *)(b->handle ? (char *)b->handle + b->offset : b->host_ptr);
	return float_buf_ensure_nocopy(stage, (size_t)n, 64);
}

status_code recipe_slot_write_commit(const exec_ctx *ctx, uint8_t idx, const float *staged, int n) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return ERR_INVALID_ARG;
	if (slot_is_host_resident(b))
		return OK;
	return b->owner->buffer_write_f32(b->owner, b, staged, n);
}

float *recipe_slot_rw_f32(const exec_ctx *ctx, uint8_t idx, float_buf *stage, int n) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return NULL;
	if (slot_is_host_resident(b))
		return (float *)(b->handle ? (char *)b->handle + b->offset : b->host_ptr);
	float *dst = float_buf_ensure_nocopy(stage, (size_t)n, 64);
	if (!dst || b->owner->buffer_read_f32(b->owner, b, dst, n) != OK)
		return NULL;
	return dst;
}
static inline buffer *exec_slots(const exec_ctx *ctx) {
	if (ctx->s && ctx->s->active_is_mirror)
		return ctx->s->mirror_slots;
	return ctx->s->slots;
}
static inline int exec_is_batch(const exec_ctx *ctx) {
	return ctx->bs != NULL;
}

static inline backend *exec_active_backend(const exec_ctx *ctx) {
	if (ctx->s && ctx->s->active_backend)
		return ctx->s->active_backend;
	return ctx->m->backend;
}

static inline backend *exec_layer_backend(const exec_ctx *ctx) {
	if (ctx->li >= 0 && ctx->m && ctx->m->mixed_backend_mode)
		return model_layer_backend(ctx->m, ctx->li);
	return exec_active_backend(ctx);
}

static inline int exec_pos_for_row(const exec_ctx *ctx, int row) {
	return exec_is_batch(ctx) ? ctx->pos_start + row : ctx->pos;
}

static status_code ple_ensure_norm_w(backend *a, compute_scratch *s, model *m,
									 int n_embd_per_layer) {
	if (s->ple_proj_norm_w_uploaded)
		return OK;
	status_code st =
		buffer_ensure_scratch(a, &s->ple_proj_norm_w, (size_t)n_embd_per_layer * sizeof(float));
	if (st != OK)
		return st;
	st = a->buffer_write_f32(a, &s->ple_proj_norm_w, m->layer_dims.per_layer_proj_norm_w.host_ptr,
							 n_embd_per_layer);
	if (st != OK)
		return st;
	s->ple_proj_norm_w_uploaded = 1;
	return OK;
}

typedef struct {
	char			  arch_gguf_name[64];
	recipe_builder_fn builder;
} recipe_reg_entry;

static recipe_reg_entry g_recipe_registry[RECIPE_MAX_ARCHES];
static int				g_recipe_registry_count = 0;

void recipe_register(const char *arch_gguf_name, recipe_builder_fn builder) {
	if (!arch_gguf_name || !builder)
		return;
	if (g_recipe_registry_count >= RECIPE_MAX_ARCHES) {
		ERROR("recipe_register: registry full (%d entries)", RECIPE_MAX_ARCHES);
		return;
	}
	for (int i = 0; i < g_recipe_registry_count; i++) {
		if (strcmp(g_recipe_registry[i].arch_gguf_name, arch_gguf_name) == 0) {
			WARN("recipe_register: arch '%s' already registered, replacing", arch_gguf_name);
			g_recipe_registry[i].builder = builder;
			return;
		}
	}
	recipe_reg_entry *e = &g_recipe_registry[g_recipe_registry_count++];
	snprintf(e->arch_gguf_name, sizeof(e->arch_gguf_name), "%s", arch_gguf_name);
	e->builder = builder;
}

const recipe_builder_fn *recipe_lookup(const char *arch_gguf_name) {
	if (!arch_gguf_name)
		return NULL;
	for (int i = 0; i < g_recipe_registry_count; i++) {
		if (strcmp(g_recipe_registry[i].arch_gguf_name, arch_gguf_name) == 0) {
			return &g_recipe_registry[i].builder;
		}
	}
	return NULL;
}

static inline int layer_has_kv(const model *m, int li) {
	if (!m->arch_info->has_variable_layer_dims)
		return 1;
	if (m->layer_dims.n_layer_kv_from_start <= 0)
		return 1;
	return li < m->layer_dims.n_layer_kv_from_start;
}

static void resolve_matmul_dims(const layer_ctx_entry *lc, const model *m, uint8_t w_idx, int *n,
								int *k) {
	int head_dim = lc->head_dim;
	switch ((weight_idx)w_idx) {
	case WIDX_WQ:
		*n = m->n_heads * head_dim;
		break;
	case WIDX_WK:
	case WIDX_WV:
		*n = lc->n_kv_heads * head_dim;
		break;
	case WIDX_WO:
		*n = m->dim;
		*k = m->n_heads * head_dim;
		break;
	case WIDX_DOWN:
		*n = m->dim;
		*k = lc->intermediate;
		break;
	case WIDX_GATE:
	case WIDX_UP:
		*n = lc->intermediate;
		break;
	default:
		break;
	}
}

static void recipe_coalesce_matmul_runs(recipe_op *ops, int n_ops) {
	for (int j = 0; j < n_ops; j++) {
		recipe_op *op = &ops[j];
		if (op->kind != OP_MATMUL || op->u.matmul.n == 0 || op->u.matmul.k == 0)
			continue;
		int run = 1;
		while (run < RECIPE_COALESCE_MAX && j + run < n_ops) {
			recipe_op *next = &ops[j + run];
			if (next->kind != OP_MATMUL || next->u.matmul.n == 0 || next->u.matmul.k == 0)
				break;
			if (next->in[0] != op->in[0] || next->u.matmul.k != op->u.matmul.k)
				break;
			run++;
		}
		op->coalesce_run_len = (uint8_t)run;
	}
}

static int model_has_sliding_layers(const model *m);

static int recipe_uses_mla(const model_recipe *r) {
	for (int i = 0; i < r->layer.n_ops; i++)
		if (r->layer.ops[i].kind == OP_ATTENTION_MLA)
			return 1;
	return 0;
}

static status_code recipe_require_capability(const struct model *m, const backend *a,
											 const char *op_name, int native, int fallback_native,
											 int allow_fallback) {
	if (native)
		return OK;
	if (allow_fallback && fallback_native) {
		WARN("backend '%s': falling back to cpu for %s", a->name, op_name);
		return OK;
	}
	ERROR("recipe_build: '%s' needs %s, unsupported on backend '%s'%s", m->arch_info->gguf_name,
		  op_name, a->name, allow_fallback ? " (no CPU fallback)" : "");
	return allow_fallback ? ERR_FORMAT : ERR_UNSUPPORTED;
}

static status_code recipe_check_backend_capabilities(const model_recipe *r, const struct model *m) {
	const backend *a = m->backend;
	status_code	   s;

	if (recipe_uses_mla(r)) {
		s = recipe_require_capability(m, a, "attention_mla", a->attention_mla != NULL, 0, 0);
		if (s != OK)
			return s;
	}
	if (model_has_sliding_layers(m) && !a->attention_swa) {
		backend *host = backend_host();
		s = recipe_require_capability(m, a, "attention_swa", 0,
									  host && host != a && host->attention_swa != NULL, 1);
		if (s != OK)
			return s;
	}
	if (m->moe.n_experts > 0 && !a->matmul_thread_local &&
		!(backend_has_cap(a, BCAP_MOE_EXPERT_RESIDENT) && a->moe_experts_batch &&
		  a->moe_expert_ffn && m->moe.experts_resident)) {
		backend *host = backend_host();
		s = recipe_require_capability(m, a, "matmul_thread_local", 0,
									  host && host != a && host->matmul_thread_local != NULL, 1);
		if (s != OK)
			return s;
	}

	return OK;
}

model_recipe *recipe_build(const struct model *m) {
	if (!m || !m->arch_info)
		return NULL;
	const recipe_builder_fn *fb = recipe_lookup(m->arch_info->gguf_name);
	if (!fb) {
		DEBUG("recipe_build: no recipe builder registered for arch '%s' "
			  "(will fall back to legacy compute path)",
			  m->arch_info->gguf_name);
		return NULL;
	}
	model_recipe *r = (*fb)(m);
	if (!r)
		return NULL;

	assert(m->moe.n_experts <= MOE_MAX_K);
	if (m->moe.n_experts > MOE_MAX_K) {
		ERROR("recipe_build: n_experts=%d exceeds supported maximum MOE_MAX_K=%d", m->moe.n_experts,
			  MOE_MAX_K);
		recipe_free(r);
		return NULL;
	}
	if (m->moe.n_experts > 0 && m->moe.n_experts_used > MOE_MAX_TOPK)
		ERROR("recipe_build: n_experts_used=%d exceeds supported top-k %d; execution will fail "
			  "rather than silently truncate",
			  m->moe.n_experts_used, MOE_MAX_TOPK);

	int has_vld = m->arch_info->has_variable_layer_dims;

	if (!has_vld) {
		recipe_coalesce_matmul_runs(r->layer.ops, r->layer.n_ops);
		if (r->per_layer_ops) {
			for (int li = 0; li < m->n_layers; li++)
				recipe_coalesce_matmul_runs(&r->per_layer_ops[(size_t)li * r->layer.n_ops],
											r->layer.n_ops);
		}
	}

	r->layer_ctx = xcalloc((size_t)m->n_layers, sizeof(layer_ctx_entry));
	for (int li = 0; li < m->n_layers; li++) {
		layer_ctx_entry *lc = &r->layer_ctx[li];
		lc->head_dim		= model_layer_head_dim(m, li);
		lc->n_kv_heads		= model_layer_kv_heads(m, li);
		lc->intermediate	= model_layer_intermediate(m, li);
		lc->has_kv			= layer_has_kv(m, li);
		lc->has_own_v		= model_layer_has_own_v(m, li);
		lc->is_global		= m->layers[li].is_global_layer;
		lc->kv_row_stride	= lc->n_kv_heads * lc->head_dim;
		lc->q_row_stride	= m->n_heads * lc->head_dim;
		lc->kv_layer		= li;
		if (has_vld && !lc->has_kv)
			lc->kv_layer =
				lc->is_global ? m->layer_dims.kv_layer_global : m->layer_dims.kv_layer_swa;
	}

	{
		int max_inter = r->max_intermediate;
		int max_hd	  = r->max_head_dim;
		int max_kvh	  = r->max_kv_heads;
		for (int li = 0; li < m->n_layers; li++) {
			const layer_ctx_entry *lc = &r->layer_ctx[li];
			if (lc->intermediate > max_inter)
				max_inter = lc->intermediate;
			if (lc->head_dim > max_hd)
				max_hd = lc->head_dim;
			if (lc->n_kv_heads > max_kvh)
				max_kvh = lc->n_kv_heads;
		}
		if (m->moe.n_experts > 0 || m->moe.moe_intermediate > 0) {
			if (m->moe.moe_intermediate > max_inter)
				max_inter = m->moe.moe_intermediate;
			int sh_inter = m->moe.moe_intermediate * m->moe.n_shared_experts;
			if (sh_inter > max_inter)
				max_inter = sh_inter;
		}
		r->max_intermediate = max_inter;
		r->max_head_dim		= max_hd;
		r->max_kv_heads		= max_kvh;
	}

	if (has_vld) {
		int n_ops		 = r->layer.n_ops;
		r->per_layer_ops = xcalloc((size_t)m->n_layers * n_ops, sizeof(recipe_op));
		for (int li = 0; li < m->n_layers; li++) {
			const layer_ctx_entry *lc  = &r->layer_ctx[li];
			recipe_op			  *dst = &r->per_layer_ops[(size_t)li * n_ops];
			memcpy(dst, r->layer.ops, (size_t)n_ops * sizeof(recipe_op));
			for (int j = 0; j < n_ops; j++) {
				recipe_op *op = &dst[j];
				if (!lc->has_kv) {
					if ((op->w_idx == WIDX_WK || op->w_idx == WIDX_WV) || op->kind == OP_KV_PUT ||
						(op->kind == OP_ROPE_EXT && op->in[0] == RECIPE_SLOT_K) ||
						(op->kind == OP_RMSNORM_PER_HEAD && op->w_idx == WIDX_ATTN_K_NORM) ||
						op->kind == OP_RMSNORM_NOWEIGHT) {
						op->kind = OP_NONE;
						continue;
					}
				}
				if (op->kind == OP_MATMUL || op->kind == OP_MATMUL_RESIDUAL ||
					op->kind == OP_MATMUL_FUSED_GATEUP) {
					if (op->u.matmul.n == 0 || op->u.matmul.k == 0)
						resolve_matmul_dims(lc, m, op->w_idx, &op->u.matmul.n, &op->u.matmul.k);
				}
				if (op->kind == OP_MATMUL_FFN_DOWN) {
					if (op->u.matmul_ffn_down.n == 0)
						op->u.matmul_ffn_down.n = m->dim;
					if (op->u.matmul_ffn_down.k == 0)
						op->u.matmul_ffn_down.k = lc->intermediate;
				}
				if (op->kind == OP_MATMUL_MULTI && op->w_idx == WIDX_GATE) {
					if (op->u.matmul_multi.n_out[0] == 0)
						op->u.matmul_multi.n_out[0] = lc->intermediate;
					if (op->u.matmul_multi.n_out[1] == 0)
						op->u.matmul_multi.n_out[1] = lc->intermediate;
				}
				if (op->kind == OP_MATMUL_MULTI && op->w_idx == WIDX_WQ) {
					op->u.matmul_multi.n_out[0] = m->n_heads * lc->head_dim;
					op->u.matmul_multi.n_out[1] = lc->n_kv_heads * lc->head_dim;
					op->u.matmul_multi.n_out[2] = lc->n_kv_heads * lc->head_dim;
				}
				if (op->kind == OP_ATTENTION) {
					op->u.attention.head_dim		  = lc->head_dim;
					op->u.attention.n_kv_heads		  = lc->n_kv_heads;
					op->u.attention.n_kv_heads_active = lc->n_kv_heads;
					op->u.attention.kv_layer		  = lc->kv_layer;
				}
			}
			recipe_coalesce_matmul_runs(dst, n_ops);
		}
	}

	uint32_t bs_mask = 0;
	for (int i = 0; i < r->n_pre_ops; i++)
		bs_mask |= op_batch_slot_mask(&r->pre_ops[i]);
	if (r->per_layer_ops && m->n_layers > 0 && r->layer.n_ops > 0) {
		for (int li = 0; li < m->n_layers; li++) {
			const recipe_op *lops = &r->per_layer_ops[(size_t)li * r->layer.n_ops];
			for (int j = 0; j < r->layer.n_ops; j++)
				bs_mask |= op_batch_slot_mask(&lops[j]);
		}
	} else {
		for (int j = 0; j < r->layer.n_ops; j++)
			bs_mask |= op_batch_slot_mask(&r->layer.ops[j]);
	}
	r->bs_slot_mask = bs_mask;

	if (recipe_check_backend_capabilities(r, m) != OK) {
		recipe_free(r);
		return NULL;
	}

	return r;
}

void recipe_free(model_recipe *r) {
	if (!r)
		return;
	free(r->pre_ops);
	free(r->layer.ops);
	free(r->post_ops);
	free(r->per_layer_ops);
	free(r->layer_ctx);
	memset(r, 0, sizeof(*r));
	free(r);
}
static const weight_ref WEIGHT_REF_NONE = {0};

static weight_ref *resolve_weight_ref(const model *m, int li, uint8_t w_idx) {
	if (w_idx == RECIPE_NO_WEIGHT || w_idx == WIDX_NONE)
		return (weight_ref *)&WEIGHT_REF_NONE;
	if (!m || !m->wrefs_by_layer || m->n_layers <= 0)
		return (weight_ref *)&WEIGHT_REF_NONE;
	int			row = (li >= 0 && li < m->n_layers) ? li : 0;
	weight_ref *w	= m->wrefs_by_layer[(size_t)row * WIDX_COUNT + w_idx];
	return w ? w : (weight_ref *)&WEIGHT_REF_NONE;
}

static const buffer *resolve_weight(const model *m, int li, uint8_t w_idx) {
	return &resolve_weight_ref(m, li, w_idx)->buf;
}

static status_code copy_k_to_v_slot(model *m, struct compute_scratch *s, int li) {
	int		 kv_out = m->recipe->layer_ctx[li].kv_row_stride;
	backend *a		= model_layer_backend(m, li);
	buffer	*kb = s->active_is_mirror ? &s->mirror_slots[RECIPE_SLOT_K] : &s->slots[RECIPE_SLOT_K];
	buffer	*vb = s->active_is_mirror ? &s->mirror_slots[RECIPE_SLOT_V] : &s->slots[RECIPE_SLOT_V];
	if (a->copy_buffer && kb->owner == vb->owner) {
		return a->copy_buffer(a, kb, vb, kv_out);
	}
	if (a->synchronize)
		a->synchronize(a);
	compute_small_host_ensure(s, m->dim, 0, kv_out);
	float	   *tmp		= s->vf_host;
	backend	   *k_owner = kb->owner;
	backend	   *v_owner = vb->owner;
	status_code st		= k_owner->buffer_read_f32(k_owner, kb, tmp, kv_out);
	if (st == OK)
		st = v_owner->buffer_write_f32(v_owner, vb, tmp, kv_out);
	return st;
}

static status_code op_matmul_multi_qkv(const recipe_op *op, model *m, kvcache *cache,
									   compute_scratch *s, int li, buffer *slots) {
	(void)cache;
	profile				*prof = &s->prof;
	profile_scope		 ps;
	status_code			 st;
	backend				*a			= model_layer_backend(m, li);
	const layer_weights *L			= &m->layers[li];
	int					 k			= op->u.matmul_multi.k;
	const buffer		*w_list[3]	= {&L->wq.buf, &L->wk.buf, &L->wv.buf};
	uint32_t			 w_types[3] = {L->wq.type, L->wk.type, L->wv.type};
	buffer				*y_list[3]	= {&slots[op->out], &slots[op->out + 1], &slots[op->out + 2]};
	const int			*n_out		= op->u.matmul_multi.n_out;
	const int			 has_kv		= model_layer_has_kv(m, li);
	const int			 has_own_v	= model_layer_has_own_v(m, li);
	int					 n_mats		= (has_kv && has_own_v) ? 3 : (has_kv ? 2 : 1);
	int					 n			= has_kv ? op->u.matmul_multi.n : 1;

	if (!a->matmul_multi || n == 1 || n_mats < n) {
		ps = profile_begin(prof, STAGE_MATMUL_QKV);
		st = a->matmul(a, w_list[0], w_types[0], &slots[op->in[0]], y_list[0], n_out[0], k);
		if (st == OK && n_mats >= 2)
			st = a->matmul(a, w_list[1], w_types[1], &slots[op->in[0]], y_list[1], n_out[1], k);
		if (st == OK && n_mats >= 3)
			st = a->matmul(a, w_list[2], w_types[2], &slots[op->in[0]], y_list[2], n_out[2], k);
		profile_end(prof, &ps);
		if (st == OK && has_kv && !has_own_v)
			st = copy_k_to_v_slot(m, s, li);
		return st;
	}

	ps = profile_begin(prof, STAGE_MATMUL_QKV);
	st = a->matmul_multi(a, w_list, w_types, &slots[op->in[0]], y_list, n_out, k, n_mats);
	profile_end(prof, &ps);
	if (st == OK && has_kv && !has_own_v)
		st = copy_k_to_v_slot(m, s, li);
	return st;
}

static status_code op_matmul_multi_kv(const recipe_op *op, model *m, kvcache *cache,
									  compute_scratch *s, int li, buffer *slots) {
	(void)cache;
	profile				  *prof = &s->prof;
	profile_scope		   ps;
	status_code			   st;
	backend				  *a  = model_layer_backend(m, li);
	const layer_weights	  *L  = &m->layers[li];
	const layer_ctx_entry *lc = &m->recipe->layer_ctx[li];
	int					   k  = op->u.matmul_multi.k;

	int kv_out	  = lc->kv_row_stride;
	int has_own_v = lc->has_own_v;

	if (!has_own_v) {
		ps = profile_begin(prof, op->stage);
		st = a->matmul(a, &L->wk.buf, L->wk.type, &slots[op->in[0]], &slots[RECIPE_SLOT_K], kv_out,
					   k);
		if (st == OK)
			st = copy_k_to_v_slot(m, s, li);
		profile_end(prof, &ps);
		return st;
	}

	const buffer *w_list[2]		 = {&L->wk.buf, &L->wv.buf};
	uint32_t	  w_types[2]	 = {L->wk.type, L->wv.type};
	buffer		 *y_list[2]		 = {&slots[RECIPE_SLOT_K], &slots[RECIPE_SLOT_V]};
	int			  n_out_local[2] = {kv_out, kv_out};

	if (!a->matmul_multi) {
		ps = profile_begin(prof, op->stage);
		st = a->matmul(a, w_list[0], w_types[0], &slots[op->in[0]], y_list[0], kv_out, k);
		if (st == OK)
			st = a->matmul(a, w_list[1], w_types[1], &slots[op->in[0]], y_list[1], kv_out, k);
		profile_end(prof, &ps);
		return st;
	}

	ps = profile_begin(prof, op->stage);
	st = a->matmul_multi(a, w_list, w_types, &slots[op->in[0]], y_list, n_out_local, k, 2);
	profile_end(prof, &ps);
	return st;
}

static status_code op_matmul_multi_gateup(const recipe_op *op, model *m, kvcache *cache,
										  compute_scratch *s, int li, buffer *slots) {
	(void)cache;
	profile				*prof = &s->prof;
	profile_scope		 ps;
	status_code			 st;
	backend				*a		 = model_layer_backend(m, li);
	const layer_weights *L		 = &m->layers[li];
	int					 k		 = op->u.matmul_multi.k;
	int					 n		 = op->u.matmul_multi.n;
	int			  n_out_local[2] = {op->u.matmul_multi.n_out[0], op->u.matmul_multi.n_out[1]};
	const buffer *w_list[2]		 = {&L->gate_w.buf, &L->up_w.buf};
	uint32_t	  w_types[2]	 = {L->gate_w.type, L->up_w.type};
	buffer		 *y_list[2]		 = {&slots[op->out], &slots[op->out + 1]};

	if (!a->matmul_multi) {
		ps = profile_begin(prof, op->stage);
		st = a->matmul(a, w_list[0], w_types[0], &slots[op->in[0]], y_list[0], n_out_local[0], k);
		if (st == OK)
			st = a->matmul(a, w_list[1], w_types[1], &slots[op->in[0]], y_list[1], n_out_local[1],
						   k);
		profile_end(prof, &ps);
		return st;
	}

	ps = profile_begin(prof, op->stage);
	st = a->matmul_multi(a, w_list, w_types, &slots[op->in[0]], y_list, n_out_local, k, n);
	profile_end(prof, &ps);
	return st;
}

static stage matmul_substage(uint8_t w_idx) {
	switch ((weight_idx)w_idx) {
	case WIDX_WQ:
	case WIDX_WK:
	case WIDX_WV:
	case WIDX_MLA_Q_A:
	case WIDX_MLA_Q_B:
	case WIDX_MLA_KV_A:
	case WIDX_MLA_K_B:
	case WIDX_MLA_V_B:
		return STAGE_MATMUL_QKV;
	case WIDX_WO:
		return STAGE_MATMUL_ATTN_OUT;
	case WIDX_GATE:
	case WIDX_UP:
	case WIDX_DOWN:
	case WIDX_GATE_UP:
	case WIDX_SHEXP_GATE:
	case WIDX_SHEXP_UP:
	case WIDX_SHEXP_DOWN:
		return STAGE_MATMUL_FFN;
	case WIDX_OUTPUT_W:
		return STAGE_LOGITS_MATMUL;
	default:
		return STAGE_MATMUL;
	}
}

static int exec_matmul_run_coalesced(const recipe_op *ops, int n_ops, model *m, compute_scratch *s,
									 int li, buffer *slots, status_code *out_status) {
	backend *a = m->backend;
	if (!a->matmul_multi || n_ops < 2) {
		*out_status = OK;
		return 0;
	}

	int n = n_ops > RECIPE_COALESCE_MAX ? RECIPE_COALESCE_MAX : n_ops;
	for (int i = 0; i < n; i++) {
		if (ops[i].w_idx == WIDX_WV && li >= 0 && !model_layer_has_own_v(m, li)) {
			n = i;
			break;
		}
	}
	if (n < 2) {
		*out_status = OK;
		return 0;
	}
	const buffer *w_list[RECIPE_COALESCE_MAX];
	uint32_t	  w_types[RECIPE_COALESCE_MAX];
	buffer		 *y_list[RECIPE_COALESCE_MAX];
	int			  n_out[RECIPE_COALESCE_MAX];
	int			  k = ops[0].u.matmul.k;

	for (int i = 0; i < n; i++) {
		weight_ref *w = resolve_weight_ref(m, li, ops[i].w_idx);
		w_list[i]	  = &w->buf;
		w_types[i]	  = w->type;
		y_list[i]	  = &slots[ops[i].out];
		n_out[i]	  = ops[i].u.matmul.n;
	}

	profile		 *prof = &s->prof;
	profile_scope ps   = profile_begin(prof, matmul_substage(ops[0].w_idx));
	status_code st = a->matmul_multi(a, w_list, w_types, &slots[ops[0].in[0]], y_list, n_out, k, n);
	profile_end(prof, &ps);

	*out_status = st;
	return n;
}

static int exec_matmul_run_coalesced_batch(const recipe_op *ops, int n_ops, model *m,
										   compute_scratch *s, batch_scratch *bs, int li,
										   int n_rows, status_code *out_status) {
	backend *a = m->backend;
	if (!a->matmul_multi_batch || n_ops < 2) {
		*out_status = OK;
		return 0;
	}

	int n = n_ops > RECIPE_COALESCE_MAX ? RECIPE_COALESCE_MAX : n_ops;
	for (int i = 0; i < n; i++) {
		if (ops[i].w_idx == WIDX_WV && li >= 0 && !model_layer_has_own_v(m, li)) {
			n = i;
			break;
		}
	}
	if (n < 2) {
		*out_status = OK;
		return 0;
	}
	const buffer *w_list[RECIPE_COALESCE_MAX];
	uint32_t	  w_types[RECIPE_COALESCE_MAX];
	buffer		 *y_list[RECIPE_COALESCE_MAX];
	int			  n_out[RECIPE_COALESCE_MAX];
	int			  k = ops[0].u.matmul.k;

	for (int i = 0; i < n; i++) {
		weight_ref *w = resolve_weight_ref(m, li, ops[i].w_idx);
		w_list[i]	  = &w->buf;
		w_types[i]	  = w->type;
		y_list[i]	  = batch_slot(bs, ops[i].out);
		n_out[i]	  = ops[i].u.matmul.n;
	}

	profile		 *prof = &s->prof;
	profile_scope ps   = profile_begin(prof, matmul_substage(ops[0].w_idx));
	status_code st = a->matmul_multi_batch(a, w_list, w_types, batch_slot(bs, ops[0].in[0]), y_list,
										   n_out, k, n, n_rows);
	profile_end(prof, &ps);

	*out_status = st;
	return n;
}

static int exec_matmul_run_qonly(const recipe_op *ops, int n_ops, model *m, buffer *xb,
								 buffer **y_list, int li, int n_rows, status_code *out_status) {
	backend *a	= m->backend;
	*out_status = OK;
	if (n_ops < 2 || !backend_has_cap(a, BCAP_MATMUL_QONLY) || !a->prequantize_x ||
		!a->matmul_qonly)
		return 0;

	int n = n_ops > RECIPE_COALESCE_MAX ? RECIPE_COALESCE_MAX : n_ops;
	for (int i = 0; i < n; i++) {
		if (ops[i].w_idx == WIDX_WV && li >= 0 && !model_layer_has_own_v(m, li)) {
			n = i;
			break;
		}
	}
	if (n < 2)
		return 0;

	int		 k	= ops[0].u.matmul.k;
	uint32_t q8 = wtype_to_q8type(resolve_weight_ref(m, li, ops[0].w_idx)->type);
	if (!q8)
		return 0;
	for (int i = 1; i < n; i++) {
		if (ops[i].u.matmul.k != k ||
			wtype_to_q8type(resolve_weight_ref(m, li, ops[i].w_idx)->type) != q8) {
			n = i;
			break;
		}
	}
	if (n < 2)
		return 0;

	buffer		xq = {0};
	status_code st = a->prequantize_x(a, xb, k, q8, &xq);
	if (st != OK) {
		if (st == ERR_UNSUPPORTED || st == ERR_INVALID_ARG) {
			*out_status = OK;
			return 0;
		}
		*out_status = st;
		return -1;
	}

	for (int i = 0; i < n; i++) {
		weight_ref *w = resolve_weight_ref(m, li, ops[i].w_idx);
		status_code ost =
			a->matmul_qonly(a, &w->buf, w->type, &xq, q8, y_list[i], ops[i].u.matmul.n, k, n_rows);
		if (ost != OK) {
			*out_status = ost;
			return -1;
		}
	}
	*out_status = OK;
	return n;
}

static inline backend *op_backend_fallback(backend *a) {
	if (a->synchronize)
		a->synchronize(a);
	return op_host_backend();
}

static inline void backend_sync_rope(backend *t, backend *a) {
	if (t != a) {
		t->rope_neox  = a->rope_neox;
		t->rope_theta = a->rope_theta;
	}
}

static status_code op_ple_build(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return ple_build_batch(ctx);
	const recipe_op		   *op	  = ctx->op;
	struct model		   *m	  = ctx->m;
	struct compute_scratch *s	  = ctx->s;
	int						token = ctx->token;
	backend				   *a	  = exec_layer_backend(ctx);
	profile				   *prof  = &ctx->s->prof;
	buffer				   *slots = exec_slots(ctx);
	profile_scope			ps;
	status_code				st;
	const int				n_embd_per_layer = m->layer_dims.n_embd_per_layer;
	const int				n_layers		 = m->n_layers;
	const int				total_ple		 = n_embd_per_layer * n_layers;
	const int				ple_dim			 = m->dim;
	const float				eps				 = m->norm_eps;
	const float				n_embd_sqrt		 = sqrtf((float)n_embd_per_layer);
	const float				inv_sqrt_ple	 = m->dim_sqrt > 0 ? 1.0f / m->dim_sqrt : 0.0f;

	size_t		   row_stride = ggml_row_size(m->layer_dims.per_layer_tok_embd.type, total_ple);
	const uint8_t *embd =
		(const uint8_t *)m->layer_dims.per_layer_tok_embd.host_ptr + ((size_t)token * row_stride);
	float *ple = s->ple_buf;

	dequant_row_dispatch(m->layer_dims.per_layer_tok_embd.type, embd, total_ple, ple);

	{
		float scale = n_embd_sqrt;
		for (int i = 0; i < total_ple; i++)
			ple[i] *= scale;
	}

	int dev_path_ok = (a->matmul && a->scale_inplace && a->rmsnorm && a->ple_combine &&
					   m->layer_dims.per_layer_model_proj.buf.handle);

	if (dev_path_ok) {
		st = buffer_ensure_scratch(a, &s->ple_all, (size_t)total_ple * sizeof(float));
		if (st != OK)
			return st;
		st = buffer_ensure_scratch(a, &s->ple_proj, (size_t)total_ple * sizeof(float));
		if (st != OK)
			return st;

		st = a->buffer_write_f32(a, &s->ple_all, ple, total_ple);
		if (st != OK)
			return st;

		buffer *xb = &slots[RECIPE_SLOT_X];
		st		   = a->matmul(a, &m->layer_dims.per_layer_model_proj.buf,
							   m->layer_dims.per_layer_model_proj.type, xb, &s->ple_proj, total_ple,
							   ple_dim);
		if (st != OK)
			return st;

		float inv_sqrt_dim = inv_sqrt_ple;
		st				   = a->scale_inplace(a, &s->ple_proj, inv_sqrt_dim, total_ple);
		if (st != OK)
			return st;

		st = ple_ensure_norm_w(a, s, m, n_embd_per_layer);
		if (st != OK)
			return st;

		for (int l = 0; l < n_layers; l++) {
			buffer x_slice = s->ple_proj;
			x_slice.offset = (size_t)l * n_embd_per_layer * sizeof(float);
			ps			   = profile_begin(prof, op->stage);
			st = a->rmsnorm(a, &x_slice, &s->ple_proj_norm_w, &x_slice, n_embd_per_layer, eps);
			profile_end(prof, &ps);
			if (st != OK)
				return st;
		}

		float combine_scale = 0.70710678118654752f;
		st = a->ple_combine(a, &s->ple_all, &s->ple_proj, total_ple, combine_scale);
		if (st != OK)
			return st;

		return OK;
	}

	if (a->synchronize)
		a->synchronize(a);

	float *ple_proj	 = float_buf_ensure(&s->ple_proj_host, (size_t)total_ple);
	float *inpL_host = float_buf_ensure(&s->inpL_host, (size_t)ple_dim);

	buffer	*xb		 = &slots[RECIPE_SLOT_X];
	backend *x_owner = xb->owner;
	st				 = x_owner->buffer_read_f32(x_owner, xb, inpL_host, ple_dim);
	if (st != OK)
		return st;

	matmul_generic_f32(m->layer_dims.per_layer_model_proj.host_ptr,
					   m->layer_dims.per_layer_model_proj.type, inpL_host, ple_proj, total_ple,
					   ple_dim);

	backend *host = backend_host();

	{
		buffer proj_view = {0};
		proj_view.handle = ple_proj;
		proj_view.owner	 = host;
		host->scale_inplace(host, &proj_view, inv_sqrt_ple, total_ple);
	}

	{
		buffer w_view = {0};
		w_view.handle = (void *)m->layer_dims.per_layer_proj_norm_w.host_ptr;
		w_view.owner  = host;
		for (int l = 0; l < n_layers; l++) {
			buffer row_view = {0};
			row_view.handle = ple_proj + ((size_t)l * n_embd_per_layer);
			row_view.owner	= host;
			host->rmsnorm(host, &row_view, &w_view, &row_view, n_embd_per_layer, eps);
		}
	}

	{
		buffer ple_view	 = {0};
		ple_view.handle	 = ple;
		ple_view.owner	 = host;
		buffer proj_view = {0};
		proj_view.handle = ple_proj;
		proj_view.owner	 = host;
		host->ple_combine(host, &ple_view, &proj_view, total_ple, 0.70710678118654752f);
	}

	if (a->copy_buffer) {
		st = buffer_ensure_scratch(a, &s->ple_all, (size_t)total_ple * sizeof(float));
		if (st != OK)
			return st;
		st = a->buffer_write_f32(a, &s->ple_all, ple, total_ple);
		if (st != OK)
			return st;
	}

	return OK;
}

static status_code op_ple_proj_inject(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return ple_proj_inject_batch(ctx);
	const struct layer_weights *L =
		(ctx->li >= 0 && ctx->li < ctx->m->n_layers) ? &ctx->m->layers[ctx->li] : NULL;
	const recipe_op		   *op	  = ctx->op;
	struct model		   *m	  = ctx->m;
	struct compute_scratch *s	  = ctx->s;
	int						li	  = ctx->li;
	backend				   *a	  = exec_layer_backend(ctx);
	profile				   *prof  = &ctx->s->prof;
	buffer				   *slots = exec_slots(ctx);
	profile_scope			ps;
	status_code				st;
	if (li < 0 || !L)
		return ERR_INVALID_ARG;
	const int	n_embd_per_layer = m->layer_dims.n_embd_per_layer;
	const int	inj_dim			 = m->dim;
	const float eps				 = m->norm_eps;
	float	   *ple				 = s->ple_buf;
	float	   *ple_slice		 = ple + ((size_t)li * n_embd_per_layer);

	backend *t = a;

	buffer *ple_slice_buf = &s->ple_slice;
	buffer *ple_inp_buf	  = &s->ple_inp;

	st = buffer_ensure_scratch(t, ple_slice_buf, (size_t)n_embd_per_layer * sizeof(float));
	if (st != OK)
		return st;
	st = buffer_ensure_scratch(t, ple_inp_buf, (size_t)n_embd_per_layer * sizeof(float));
	if (st != OK)
		return st;

	if (s->ple_all.handle) {
		buffer ple_src = s->ple_all;
		ple_src.offset += (size_t)li * n_embd_per_layer * sizeof(float);
		st = compute_copy_buffer_cross(s, &ple_src, ple_slice_buf, n_embd_per_layer);
		if (st != OK)
			return st;
	} else {
		st = t->buffer_write_f32(t, ple_slice_buf, ple_slice, n_embd_per_layer);
	}
	if (st != OK)
		return st;

	buffer *xb2b = &slots[op->in[0]];
	st = t->matmul(t, &L->ple_inp_gate_w.buf, GGML_TYPE_F32, xb2b, ple_inp_buf, n_embd_per_layer,
				   inj_dim);
	if (st != OK)
		return st;

	st = t->ffn_activate_ex(t, ple_inp_buf, ple_slice_buf, ple_inp_buf, n_embd_per_layer, 1);
	if (st != OK)
		return st;

	buffer *aob = &slots[op->in[1]];
	st			= t->matmul(t, &L->ple_proj_w.buf, GGML_TYPE_F32, ple_inp_buf, aob, inj_dim,
							n_embd_per_layer);
	if (st != OK)
		return st;

	ps			= profile_begin(prof, op->stage);
	backend *t2 = OP_BACKEND(rmsnorm);
	st			= t2->rmsnorm(t2, aob, &L->ple_post_norm_w.buf, aob, inj_dim, eps);
	profile_end(prof, &ps);
	if (st != OK)
		return st;

	ps			= profile_begin(prof, op->stage);
	backend *t3 = OP_BACKEND(add_inplace);
	st			= t3->add_inplace(t3, xb2b, aob, inj_dim);
	profile_end(prof, &ps);
	return st;
}

static status_code op_ffn_activate_ex(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps;
	status_code	  st;
	buffer		 *slots		   = exec_slots(ctx);
	int			  intermediate = ctx->op->u.matmul.n;
	int			  activation   = ctx->op->u.ffn_act.activation;
	if (exec_is_batch(ctx)) {
		ps = profile_begin(prof, ctx->op->stage);
		st = a->ffn_activate_batch(a, exec_slot(ctx, ctx->op->in[0]),
								   exec_slot(ctx, ctx->op->in[1]), exec_slot(ctx, ctx->op->out),
								   intermediate, activation, ctx->n_rows);
		profile_end(prof, &ps);
		return st;
	}
	UNUSED(ctx->m, ctx->li);
	if (a->ffn_activate_ex) {
		ps = profile_begin(prof, ctx->op->stage);
		st = a->ffn_activate_ex(a, &slots[ctx->op->in[0]], &slots[ctx->op->in[1]],
								&slots[ctx->op->out], intermediate, activation);
		profile_end(prof, &ps);
		return st;
	}
	float	   *buf		= float_buf_ensure(&ctx->s->moe_scratch, (size_t)intermediate * 3);
	float	   *g		= buf;
	float	   *u		= buf + intermediate;
	float	   *o		= buf + intermediate * 2;
	backend	   *g_owner = slots[ctx->op->in[0]].owner;
	backend	   *u_owner = slots[ctx->op->in[1]].owner;
	status_code r_g = g_owner->buffer_read_f32(g_owner, &slots[ctx->op->in[0]], g, intermediate);
	if (r_g != OK)
		return r_g;
	status_code r_u = u_owner->buffer_read_f32(u_owner, &slots[ctx->op->in[1]], u, intermediate);
	if (r_u != OK)
		return r_u;
	ps = profile_begin(prof, ctx->op->stage);
	if (activation == ACTIVATION_GELU) {
		for (int i = 0; i < intermediate; i++)
			o[i] = gelu_tanh(g[i]) * u[i];
	} else {
		for (int i = 0; i < intermediate; i++)
			o[i] = silu(g[i]) * u[i];
	}
	profile_end(prof, &ps);
	backend *o_owner = slots[ctx->op->out].owner;
	st				 = o_owner->buffer_write_f32(o_owner, &slots[ctx->op->out], o, intermediate);
	return st;
}

static status_code op_rope_ext(exec_ctx *ctx) {
	const recipe_op		   *op	  = ctx->op;
	struct model		   *m	  = ctx->m;
	struct compute_scratch *s	  = ctx->s;
	int						li	  = ctx->li;
	backend				   *a	  = exec_layer_backend(ctx);
	profile				   *prof  = &ctx->s->prof;
	buffer				   *slots = exec_slots(ctx);
	if (li < 0)
		return ERR_INVALID_ARG;
	const layer_ctx_entry *lc		 = &m->recipe->layer_ctx[li];
	int					   is_global = lc->is_global;
	int					   n_heads	 = op->u.rope_ext.n_heads;
	int					   head_dim	 = lc->head_dim;
	if (n_heads == 0) {
		n_heads = (op->in[0] == RECIPE_SLOT_Q) ? m->n_heads : lc->n_kv_heads;
	}
	const float *rope_cos;
	const float *rope_sin;
	const float *freq_factors;
	if (is_global) {
		rope_cos	 = s->rope_cos;
		rope_sin	 = s->rope_sin;
		freq_factors = op->u.rope_ext.use_freq_factors ? m->rope_freqs : NULL;
	} else {
		rope_cos	 = s->rope_cos_swa;
		rope_sin	 = s->rope_sin_swa;
		freq_factors = NULL;
	}
	a->rope_neox  = op->u.rope_ext.rope_neox;
	a->rope_theta = is_global ? m->layer_dims.rope_theta_global : m->layer_dims.rope_theta_swa;

	profile_scope ps			 = profile_begin(prof, op->stage);
	status_code	  st			 = OK;
	int			  is_batch		 = exec_is_batch(ctx);
	int			  pos			 = is_batch ? ctx->pos_start : ctx->pos;
	int			  n_applications = is_batch ? ctx->n_rows : 1;

	if (is_batch && a->rope_ext_batch && a->rope_ext) {
		buffer *whole = batch_slot(ctx->bs, op->in[0]);
		st = a->rope_ext_batch(a, whole, n_heads, head_dim, pos, rope_cos, rope_sin, freq_factors,
							   ctx->n_rows);
		profile_end(prof, &ps);
		return st;
	}

	for (int r = 0; r < n_applications; r++) {
		buffer vec;
		if (is_batch) {
			int row_stride = (op->in[0] == RECIPE_SLOT_Q) ? lc->q_row_stride : lc->kv_row_stride;
			vec			   = batch_row_view(batch_slot(ctx->bs, op->in[0]), r, row_stride);
		} else {
			vec = slots[op->in[0]];
		}
		backend *t = is_batch ? (a->rope_ext ? a : OP_BACKEND(rope)) : OP_BACKEND(rope_ext);
		backend_sync_rope(t, a);
		if (t->rope_ext) {
			st = t->rope_ext(t, &vec, n_heads, head_dim, pos + r, rope_cos, rope_sin, freq_factors);
		} else {
			backend *t2 = OP_BACKEND(rope);
			backend_sync_rope(t2, a);
			st = t2->rope(t2, &vec, n_heads, head_dim, pos + r, rope_cos, rope_sin);
		}
		if (st != OK)
			break;
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_attention_impl(const recipe_op *op, struct model *m, struct kvcache *cache,
									 buffer *slots, int li, int pos, int flash_attn, backend *a,
									 profile *prof, int kv_layer, int n_heads, int n_kv_heads,
									 int head_dim, int n_ctx, float scale, int sliding_window,
									 int n_kv_heads_active, int allow_backend_fallback) {
	int has_swa = a->attention_swa != NULL || (allow_backend_fallback && backend_host() != a &&
											   backend_host()->attention_swa != NULL);
	int use_swa = (sliding_window > 0) && (li >= 0) && model_layer_is_sliding(m, li) && has_swa;
	profile_scope ps = profile_begin(prof, op->stage);
	status_code	  st;
	if (cache->mirror_remap && kvcache_slot_on_host(cache, kv_layer))
		kv_layer = kvcache_mirror_layer(cache, kv_layer);
	buffer *kb = kvcache_k_for_layer(cache, m, kv_layer);
	buffer *vb = kvcache_v_for_layer(cache, m, kv_layer);
	if (use_swa) {
		backend *t = allow_backend_fallback ? OP_BACKEND(attention_swa) : a;
		st = t->attention_swa(t, &slots[op->in[0]], kb, vb, &slots[op->out], kv_layer, pos, n_heads,
							  n_kv_heads, head_dim, n_ctx, flash_attn, scale, sliding_window,
							  n_kv_heads_active);
	} else {
		backend *t = allow_backend_fallback ? OP_BACKEND(attention) : a;
		st = t->attention(t, &slots[op->in[0]], kb, vb, &slots[op->out], kv_layer, pos, n_heads,
						  n_kv_heads, head_dim, n_ctx, flash_attn, scale, n_kv_heads_active);
	}
	profile_end(prof, &ps);
	return st;
}

typedef status_code (*op_handler)(exec_ctx *ctx);

static status_code op_embd_lookup(exec_ctx *ctx) {
	backend		 *a		= exec_layer_backend(ctx);
	profile		 *prof	= &ctx->s->prof;
	buffer		 *slots = exec_slots(ctx);
	profile_scope ps;
	status_code	  st;
	const int	  dim		= ctx->m->dim;
	weight_ref	 *embd		= resolve_weight_ref(ctx->m, ctx->li, ctx->op->w_idx);
	const buffer *embd_w	= &embd->buf;
	uint32_t	  embd_type = embd->type;
	backend		 *t			= OP_BACKEND(embd_lookup);
	ps						= profile_begin(prof, ctx->op->stage);
	st = t->embd_lookup(t, embd_w, embd_type, ctx->token, dim, &slots[ctx->op->out]);
	profile_end(prof, &ps);
	return st;
}

static status_code op_scale_embeddings(exec_ctx *ctx) {
	backend		 *a		= exec_layer_backend(ctx);
	profile		 *prof	= &ctx->s->prof;
	buffer		 *slots = exec_slots(ctx);
	profile_scope ps;
	status_code	  st;
	const int	  dim	= ctx->m->dim;
	buffer		 *xb	= &slots[ctx->op->out];
	const float	  scale = ctx->m->dim_sqrt;
	if (a->scale_inplace) {
		ps = profile_begin(prof, ctx->op->stage);
		st = a->scale_inplace(a, xb, scale, dim);
		profile_end(prof, &ps);
		return st;
	}
	if (a->synchronize)
		a->synchronize(a);
	float  stackbuf[2048];
	float *buf =
		(dim <= (int)ARRAY_LEN(stackbuf)) ? stackbuf : xmalloc((size_t)dim * sizeof(float));
	backend *owner = xb->owner;
	st			   = owner->buffer_read_f32(owner, xb, buf, dim);
	if (st == OK) {
		for (int i = 0; i < dim; i++)
			buf[i] *= scale;
		st = owner->buffer_write_f32(owner, xb, buf, dim);
	}
	if (buf != stackbuf)
		free(buf);
	return st;
}

static status_code op_rmsnorm(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	const buffer *w	   = resolve_weight(ctx->m, ctx->li, ctx->op->w_idx);
	status_code	  st;
	if (exec_is_batch(ctx)) {
		st = a->rmsnorm_batch(a, exec_slot(ctx, ctx->op->in[0]), w, exec_slot(ctx, ctx->op->out),
							  ctx->m->dim, ctx->op->u.rmsnorm.eps, ctx->n_rows);
	} else {
		backend *t = OP_BACKEND(rmsnorm);
		st		   = t->rmsnorm(t, exec_slot(ctx, ctx->op->in[0]), w, exec_slot(ctx, ctx->op->out),
								ctx->m->dim, ctx->op->u.rmsnorm.eps);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_matmul(exec_ctx *ctx) {
	const recipe_op *op = ctx->op;
	int				 li = ctx->li;
	if (op->w_idx == WIDX_WV && li >= 0 && li < ctx->m->n_layers &&
		!ctx->m->recipe->layer_ctx[li].has_own_v) {
		if (exec_is_batch(ctx))
			return batch_copy_k_to_v(ctx->m, ctx->bs, li, ctx->n_rows);
		return copy_k_to_v_slot(ctx->m, ctx->s, li);
	}
	weight_ref	 *wref = resolve_weight_ref(ctx->m, li, op->w_idx);
	int			  n = op->u.matmul.n, k = op->u.matmul.k;
	profile_scope ps = profile_begin(&ctx->s->prof, matmul_substage(op->w_idx));
	status_code	  st;
	backend		 *a = exec_layer_backend(ctx);
	if (exec_is_batch(ctx)) {
		st = a->matmul_batch(a, &wref->buf, wref->type, exec_slot(ctx, op->in[0]),
							 exec_slot(ctx, op->out), n, k, ctx->n_rows);
	} else {
		backend *t = OP_BACKEND(matmul);
		st		   = t->matmul(t, &wref->buf, wref->type, exec_slot(ctx, op->in[0]),
							   exec_slot(ctx, op->out), n, k);
	}
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code op_matmul_residual(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, matmul_substage(ctx->op->w_idx));
	status_code	  st;
	weight_ref	 *wref = resolve_weight_ref(ctx->m, ctx->li, ctx->op->w_idx);
	const buffer *w	   = &wref->buf;
	uint32_t	  wt   = wref->type;
	int			  n	   = ctx->op->u.matmul.n;
	int			  k	   = ctx->op->u.matmul.k;
	if (exec_is_batch(ctx)) {
		buffer *tmp = batch_slot(ctx->bs, RECIPE_SLOT_RESID_TMP);
		st = a->matmul_batch(a, w, wt, exec_slot(ctx, ctx->op->in[0]), tmp, n, k, ctx->n_rows);
		if (st == OK)
			st = a->add_batch(a, exec_slot(ctx, ctx->op->out), tmp, n, ctx->n_rows);
	} else {
		if (!a->matmul_residual)
			return ERR_UNSUPPORTED;
		st = a->matmul_residual(a, w, wt, exec_slot(ctx, ctx->op->in[0]),
								exec_slot(ctx, ctx->op->in[1]), exec_slot(ctx, ctx->op->out), n, k);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_matmul_multi(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return matmul_multi_batch_body(ctx);
	buffer				*slots = exec_slots(ctx);
	const layer_weights *L =
		(ctx->li >= 0 && ctx->li < ctx->m->n_layers) ? &ctx->m->layers[ctx->li] : NULL;
	if (ctx->op->w_idx == WIDX_WQ && L)
		return op_matmul_multi_qkv(ctx->op, ctx->m, ctx->cache, ctx->s, ctx->li, slots);
	if (ctx->op->w_idx == WIDX_WK && L)
		return op_matmul_multi_kv(ctx->op, ctx->m, ctx->cache, ctx->s, ctx->li, slots);
	if (ctx->op->w_idx == WIDX_GATE && L)
		return op_matmul_multi_gateup(ctx->op, ctx->m, ctx->cache, ctx->s, ctx->li, slots);
	return ERR_INVALID_ARG;
}

static status_code op_matmul_gateup(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, matmul_substage(ctx->op->w_idx));
	status_code	  st;
	weight_ref	 *wref = resolve_weight_ref(ctx->m, ctx->li, ctx->op->w_idx);
	int			  n	   = ctx->op->u.matmul.n;
	int			  k	   = ctx->op->u.matmul.k;
	if (exec_is_batch(ctx)) {
		st = a->matmul_batch(a, &wref->buf, wref->type, exec_slot(ctx, ctx->op->in[0]),
							 batch_slot(ctx->bs, RECIPE_SLOT_FFN_GATE_UP), n, k, ctx->n_rows);
	} else {
		backend *t = OP_BACKEND(matmul);
		st		   = t->matmul(t, &wref->buf, wref->type, exec_slot(ctx, ctx->op->in[0]),
							   exec_slot(ctx, ctx->op->out), n, k);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_matmul_ffn_down(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps;
	status_code	  st;
	weight_ref	 *wref = resolve_weight_ref(ctx->m, ctx->li, ctx->op->w_idx);
	const buffer *w	   = &wref->buf;
	uint32_t	  wt   = wref->type;
	int			  n	   = ctx->op->u.matmul_ffn_down.n;
	int			  k	   = ctx->op->u.matmul_ffn_down.k;
	int			  act  = ctx->op->u.matmul_ffn_down.activation;
	if (n == 0)
		n = ctx->m->dim;
	if (exec_is_batch(ctx)) {
		ps = profile_begin(prof, matmul_substage(ctx->op->w_idx));
		st =
			a->ffn_activate_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
								  batch_slot(ctx->bs, RECIPE_SLOT_FFN_ACT), k, act, ctx->n_rows);
		if (st == OK)
			st = a->matmul_batch(a, w, wt, batch_slot(ctx->bs, RECIPE_SLOT_FFN_ACT),
								 exec_slot(ctx, ctx->op->out), n, k, ctx->n_rows);
		profile_end(prof, &ps);
		return st;
	}
	if (a->matmul_ffn_down && backend_has_cap(a, BCAP_MATMUL_FFN_DOWN)) {
		ps = profile_begin(prof, matmul_substage(ctx->op->w_idx));
		st = a->matmul_ffn_down(a, w, wt, exec_slot(ctx, ctx->op->in[0]),
								exec_slot(ctx, ctx->op->in[1]), exec_slot(ctx, ctx->op->out), n, k,
								act);
		profile_end(prof, &ps);
		if (st == OK)
			return st;
	}
	{
		int		 intermediate = k;
		backend *t			  = OP_BACKEND(ffn_activate_ex);
		if (!t->ffn_activate_ex)
			t = OP_BACKEND(ffn_activate);
		ps = profile_begin(prof, matmul_substage(ctx->op->w_idx));
		if (t->ffn_activate_ex) {
			st = t->ffn_activate_ex(t, exec_slot(ctx, ctx->op->in[0]),
									exec_slot(ctx, ctx->op->in[1]),
									exec_slot(ctx, RECIPE_SLOT_FFN_ACT), intermediate, act);
		} else {
			if (act != ACTIVATION_SILU)
				return ERR_UNSUPPORTED;
			st = t->ffn_activate(t, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
								 exec_slot(ctx, RECIPE_SLOT_FFN_ACT), intermediate);
		}
		if (st == OK) {
			backend *tm = OP_BACKEND(matmul);
			st			= tm->matmul(tm, w, wt, exec_slot(ctx, RECIPE_SLOT_FFN_ACT),
									 exec_slot(ctx, ctx->op->out), n, k);
		}
		profile_end(prof, &ps);
		return st;
	}
}

static status_code op_rope(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	int			  n_heads  = ctx->op->u.rope.n_heads;
	int			  head_dim = ctx->op->u.rope.head_dim;
	a->rope_neox		   = ctx->op->u.rope.rope_neox;
	if (exec_is_batch(ctx)) {
		st = a->rope_batch(a, exec_slot(ctx, ctx->op->in[0]), n_heads, head_dim, ctx->pos_start,
						   ctx->s->rope_cos, ctx->s->rope_sin, ctx->n_rows);
	} else {
		backend *t = OP_BACKEND(rope);
		backend_sync_rope(t, a);
		st = t->rope(t, exec_slot(ctx, ctx->op->in[0]), n_heads, head_dim, ctx->pos,
					 ctx->s->rope_cos, ctx->s->rope_sin);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_rope_qk_fused(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	int			  n_heads	 = ctx->op->u.rope.n_heads;
	int			  n_kv_heads = ctx->op->u.rope.n_kv_heads;
	int			  head_dim	 = ctx->op->u.rope.head_dim;
	a->rope_neox			 = ctx->op->u.rope.rope_neox;
	if (exec_is_batch(ctx)) {
		st = a->rope_qk_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
							  n_heads, n_kv_heads, head_dim, ctx->pos_start, ctx->s->rope_cos,
							  ctx->s->rope_sin, ctx->n_rows);
	} else {
		backend *t = a->rope_qk ? a : OP_BACKEND(rope);
		backend_sync_rope(t, a);
		if (t->rope_qk) {
			st = t->rope_qk(t, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
							n_heads, n_kv_heads, head_dim, ctx->pos, ctx->s->rope_cos,
							ctx->s->rope_sin);
		} else {
			st = t->rope(t, exec_slot(ctx, ctx->op->in[0]), n_heads, head_dim, ctx->pos,
						 ctx->s->rope_cos, ctx->s->rope_sin);
			if (st == OK)
				st = t->rope(t, exec_slot(ctx, ctx->op->in[1]), n_kv_heads, head_dim, ctx->pos,
							 ctx->s->rope_cos, ctx->s->rope_sin);
		}
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_kv_put(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st;
	if (exec_is_batch(ctx)) {
		int kv_row_stride = ctx->m->recipe->layer_ctx[ctx->li].kv_row_stride;
		return kvcache_put_batch(ctx->cache, ctx->m, ctx->li, ctx->pos_start,
								 batch_slot(ctx->bs, ctx->op->in[0]),
								 batch_slot(ctx->bs, ctx->op->in[1]), kv_row_stride, ctx->n_rows);
	}
	st = kvcache_put(ctx->cache, ctx->m, ctx->li, ctx->pos, exec_slot(ctx, ctx->op->in[0]),
					 exec_slot(ctx, ctx->op->in[1]));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code op_attention(exec_ctx *ctx) {
	backend *a				   = exec_layer_backend(ctx);
	profile *prof			   = &ctx->s->prof;
	buffer	*slots			   = exec_slots(ctx);
	int		 n_heads		   = ctx->op->u.attention.n_heads;
	int		 n_kv_heads		   = ctx->op->u.attention.n_kv_heads;
	int		 head_dim		   = ctx->op->u.attention.head_dim;
	int		 n_ctx			   = ctx->cache->n_ctx;
	float	 scale			   = ctx->op->u.attention.scale;
	int		 sliding_window	   = ctx->op->u.attention.sliding_window;
	int		 n_kv_heads_active = ctx->op->u.attention.n_kv_heads_active;
	int		 kv_layer =
		ctx->m->arch_info->has_variable_layer_dims ? ctx->op->u.attention.kv_layer : ctx->li;

	if (exec_is_batch(ctx)) {
		int use_swa = (sliding_window > 0) && (ctx->li >= 0) &&
					  model_layer_is_sliding(ctx->m, ctx->li) && a->attention_swa_batch != NULL;
		if (use_swa)
			return a->attention_swa_batch(a, batch_slot(ctx->bs, ctx->op->in[0]), &ctx->cache->k,
										  &ctx->cache->v, batch_slot(ctx->bs, ctx->op->out),
										  kv_layer, ctx->pos_start, n_heads, n_kv_heads, head_dim,
										  n_ctx, ctx->flash_attn, scale, sliding_window,
										  n_kv_heads_active, ctx->n_rows);
		return a->attention_batch(a, batch_slot(ctx->bs, ctx->op->in[0]), &ctx->cache->k,
								  &ctx->cache->v, batch_slot(ctx->bs, ctx->op->out), kv_layer,
								  ctx->pos_start, n_heads, n_kv_heads, head_dim, n_ctx,
								  ctx->flash_attn, scale, n_kv_heads_active, ctx->n_rows);
	}

	return op_attention_impl(ctx->op, ctx->m, ctx->cache, slots, ctx->li, ctx->pos, ctx->flash_attn,
							 a, prof, kv_layer, n_heads, n_kv_heads, head_dim, n_ctx, scale,
							 sliding_window, n_kv_heads_active, 1);
}

static status_code op_add(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	const int	  dim = ctx->m->dim;
	if (exec_is_batch(ctx)) {
		st = a->add_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]), dim,
						  ctx->n_rows);
	} else {
		backend *t = OP_BACKEND(add_inplace);
		st = t->add_inplace(t, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]), dim);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_swap(exec_ctx *ctx) {
	buffer *s0	= exec_slot(ctx, ctx->op->in[0]);
	buffer *s1	= exec_slot(ctx, ctx->op->in[1]);
	buffer	tmp = *s0;
	*s0			= *s1;
	*s1			= tmp;
	return OK;
}

static status_code op_ffn_activate(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	int			  n = ctx->op->u.ffn_act.n;
	if (exec_is_batch(ctx)) {
		st =
			a->ffn_activate_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
								  exec_slot(ctx, ctx->op->out), n, 0, ctx->n_rows);
	} else {
		backend *t = OP_BACKEND(ffn_activate);
		st = t->ffn_activate(t, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
							 exec_slot(ctx, ctx->op->out), n);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_ffn_activate_fused(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	int			  n = ctx->op->u.ffn_act.n;
	if (exec_is_batch(ctx)) {
		backend	 *a			   = exec_layer_backend(ctx);
		int		  act		   = ctx->op->u.ffn_act.activation;
		const int fused_stride = 2 * n;
		int		  n_rows	   = ctx->n_rows;

		if (!backend_has_cap(a, BCAP_IS_HOST)) {
			buffer *fused_buf  = batch_slot(ctx->bs, ctx->op->in[0]);
			buffer *out_buf	   = batch_slot(ctx->bs, ctx->op->out);
			float  *fused_host = xmalloc((size_t)n_rows * fused_stride * sizeof(float));
			float  *out_host   = xmalloc((size_t)n_rows * n * sizeof(float));
			st = a->buffer_read_f32(a, fused_buf, fused_host, n_rows * fused_stride);
			if (st != OK) {
				free(fused_host);
				free(out_host);
				return st;
			}
			for (int row = 0; row < n_rows; row++) {
				const float *g = fused_host + (size_t)row * fused_stride;
				const float *u = g + n;
				float		*o = out_host + (size_t)row * n;
				if (act == ACTIVATION_GELU) {
					for (int i = 0; i < n; i++)
						o[i] = gelu_tanh(g[i]) * u[i];
				} else {
					for (int i = 0; i < n; i++)
						o[i] = silu(g[i]) * u[i];
				}
			}
			st = a->buffer_write_f32(a, out_buf, out_host, n_rows * n);
			free(fused_host);
			free(out_host);
			return st;
		}
		const float *fused = batch_buf_ptr(batch_slot(ctx->bs, ctx->op->in[0]));
		float		*out   = batch_buf_ptr(batch_slot(ctx->bs, ctx->op->out));
		for (int row = 0; row < ctx->n_rows; row++) {
			const float *g = fused + (size_t)row * fused_stride;
			const float *u = g + n;
			float		*o = out + (size_t)row * n;
			if (act == ACTIVATION_GELU) {
				for (int i = 0; i < n; i++)
					o[i] = gelu_tanh(g[i]) * u[i];
			} else {
				for (int i = 0; i < n; i++)
					o[i] = silu(g[i]) * u[i];
			}
		}
		st = OK;
	} else {
		backend *t	   = OP_BACKEND(ffn_activate);
		buffer	*slots = exec_slots(ctx);
		st = t->ffn_activate(t, &slots[RECIPE_SLOT_FFN_GATE], &slots[RECIPE_SLOT_FFN_UP],
							 &slots[ctx->op->out], n);
	}
	profile_end(prof, &ps);
	return st;
}

typedef struct {
	float *logits;
	int	   vocab;
	float  inv_cap;
	float  cap;
} softcap_job;

static void softcap_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	softcap_job *j = ctx;
	for (int i = begin; i < end; i++)
		j->logits[i] = j->cap * tanhf(j->logits[i] * j->inv_cap);
}

static status_code op_softcap(exec_ctx *ctx) {
	float *logits_out = ctx->logits_out;
	if (!logits_out)
		return OK;
	float cap = ctx->op->u.softcap.cap;
	if (cap <= 0.0f)
		return OK;
	int		 vocab	 = ctx->m->vocab_size;
	float	 inv_cap = 1.0f / cap;
	backend *a		 = exec_layer_backend(ctx);
	tpool	*pool	 = (a && a->get_pool) ? a->get_pool(a) : NULL;
	if (pool && tpool_n_threads(pool) > 1 && vocab >= 2 * 4096 && tpool_current_tid() < 0) {
		softcap_job job = {.logits = logits_out, .vocab = vocab, .inv_cap = inv_cap, .cap = cap};
		tpool_parallel_for(pool, vocab, 4096, softcap_chunk, &job);
		return OK;
	}
	for (int i = 0; i < vocab; i++)
		logits_out[i] = cap * tanhf(logits_out[i] * inv_cap);
	return OK;
}

static status_code op_logits_readback(exec_ctx *ctx) {
	backend		 *a			 = exec_layer_backend(ctx);
	profile		 *prof		 = &ctx->s->prof;
	buffer		 *slots		 = exec_slots(ctx);
	float		 *logits_out = ctx->logits_out;
	profile_scope ps;
	status_code	  st;
	if (!logits_out)
		return OK;
	if (ctx->s->logits_alias && logits_out == ctx->s->logits_host &&
		backend_has_cap(a, BCAP_IS_HOST))
		return OK;
	buffer	*lb	   = &slots[ctx->op->in[0]];
	int		 vocab = ctx->m->vocab_size;
	backend *owner = lb->owner;
	if (a->synchronize)
		a->synchronize(a);
	ps = profile_begin(prof, ctx->op->stage);
	st = owner->buffer_read_f32(owner, lb, logits_out, vocab);
	profile_end(prof, &ps);
	return st;
}

static status_code op_rmsnorm_per_head(exec_ctx *ctx) {
	backend				  *a	= exec_layer_backend(ctx);
	profile				  *prof = &ctx->s->prof;
	profile_scope		   ps	= profile_begin(prof, ctx->op->stage);
	status_code			   st	= OK;
	const layer_ctx_entry *lc	= &ctx->m->recipe->layer_ctx[ctx->li];
	const buffer		  *w	= resolve_weight(ctx->m, ctx->li, ctx->op->w_idx);
	int n_heads	   = (ctx->op->w_idx == WIDX_ATTN_Q_NORM) ? ctx->m->n_heads : lc->n_kv_heads;
	int row_stride = (ctx->op->w_idx == WIDX_ATTN_Q_NORM) ? lc->q_row_stride : lc->kv_row_stride;
	backend *t	   = OP_BACKEND(rmsnorm_per_head);
	if (exec_is_batch(ctx)) {
		if (t->rmsnorm_per_head_batch) {
			st = t->rmsnorm_per_head_batch(t, exec_slot(ctx, ctx->op->in[0]), w,
										   exec_slot(ctx, ctx->op->in[0]), n_heads, lc->head_dim,
										   ctx->m->norm_eps, ctx->n_rows);
		} else {
			for (int row = 0; row < ctx->n_rows; row++) {
				buffer rowb = batch_row_view(batch_slot(ctx->bs, ctx->op->in[0]), row, row_stride);
				st			= t->rmsnorm_per_head(t, &rowb, w, &rowb, n_heads, lc->head_dim,
												  ctx->m->norm_eps);
				if (st != OK)
					break;
			}
		}
	} else {
		st = t->rmsnorm_per_head(t, exec_slot(ctx, ctx->op->in[0]), w, exec_slot(ctx, ctx->op->out),
								 n_heads, lc->head_dim, ctx->m->norm_eps);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_rmsnorm_noweight(exec_ctx *ctx) {
	backend				  *a	= exec_layer_backend(ctx);
	profile				  *prof = &ctx->s->prof;
	profile_scope		   ps;
	status_code			   st;
	const layer_ctx_entry *lc = &ctx->m->recipe->layer_ctx[ctx->li];
	backend				  *t  = OP_BACKEND(rmsnorm_noweight_per_head);
	backend_sync_rope(t, a);
	ps = profile_begin(prof, ctx->op->stage);
	if (exec_is_batch(ctx)) {
		if (t->rmsnorm_noweight_per_head_batch) {
			st = t->rmsnorm_noweight_per_head_batch(t, exec_slot(ctx, ctx->op->in[0]),
													exec_slot(ctx, ctx->op->in[0]), lc->n_kv_heads,
													lc->head_dim, ctx->m->norm_eps, ctx->n_rows);
		} else if (t->rmsnorm_noweight_batch) {
			st = t->rmsnorm_noweight_batch(t, exec_slot(ctx, ctx->op->in[0]),
										   exec_slot(ctx, ctx->op->in[0]), lc->kv_row_stride,
										   ctx->m->norm_eps, ctx->n_rows);
		} else if (t->rmsnorm_noweight_per_head) {
			for (int row = 0; row < ctx->n_rows; row++) {
				buffer rowb =
					batch_row_view(batch_slot(ctx->bs, ctx->op->in[0]), row, lc->kv_row_stride);
				st = t->rmsnorm_noweight_per_head(t, &rowb, &rowb, lc->n_kv_heads, lc->head_dim,
												  ctx->m->norm_eps);
				if (st != OK)
					break;
			}
		} else {
			backend *t2 = OP_BACKEND(rmsnorm_noweight);
			for (int row = 0; row < ctx->n_rows; row++) {
				buffer rowb =
					batch_row_view(batch_slot(ctx->bs, ctx->op->in[0]), row, lc->kv_row_stride);
				st = t2->rmsnorm_noweight(t2, &rowb, &rowb, lc->kv_row_stride, ctx->m->norm_eps);
				if (st != OK)
					break;
			}
		}
	} else {
		int n_kv_heads = lc->n_kv_heads;
		int head_dim   = lc->head_dim;
		int kv_out	   = n_kv_heads * head_dim;
		if (t->rmsnorm_noweight_per_head) {
			st = t->rmsnorm_noweight_per_head(t, exec_slot(ctx, ctx->op->in[0]),
											  exec_slot(ctx, ctx->op->in[0]), n_kv_heads, head_dim,
											  ctx->m->norm_eps);
		} else {
			t  = OP_BACKEND(rmsnorm_noweight);
			st = t->rmsnorm_noweight(t, exec_slot(ctx, ctx->op->in[0]),
									 exec_slot(ctx, ctx->op->in[0]), kv_out, ctx->m->norm_eps);
		}
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_rmsnorm_add(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	const buffer *w = resolve_weight(ctx->m, ctx->li, ctx->op->w_idx);
	int			  n = ctx->m->dim;
	if (exec_is_batch(ctx)) {
		if (a->rmsnorm_add && backend_has_cap(a, BCAP_RMSNORM_ADD) && ctx->n_rows < 4) {
			for (int row = 0; row < ctx->n_rows; row++) {
				buffer xrow	  = batch_row_view(exec_slot(ctx, ctx->op->in[0]), row, n);
				buffer resrow = batch_row_view(exec_slot(ctx, ctx->op->in[1]), row, n);
				buffer outrow = batch_row_view(exec_slot(ctx, ctx->op->out), row, n);
				st = a->rmsnorm_add(a, &xrow, w, &resrow, &outrow, n, ctx->op->u.rmsnorm.eps);
				if (st != OK)
					break;
			}
		} else {
			st =
				a->rmsnorm_batch(a, exec_slot(ctx, ctx->op->in[0]), w, exec_slot(ctx, ctx->op->out),
								 n, ctx->op->u.rmsnorm.eps, ctx->n_rows);
			if (st == OK)
				st = a->add_batch(a, exec_slot(ctx, ctx->op->out), exec_slot(ctx, ctx->op->in[1]),
								  n, ctx->n_rows);
		}
	} else {
		if (a->rmsnorm_add && backend_has_cap(a, BCAP_RMSNORM_ADD)) {
			st =
				a->rmsnorm_add(a, exec_slot(ctx, ctx->op->in[0]), w, exec_slot(ctx, ctx->op->in[1]),
							   exec_slot(ctx, ctx->op->out), n, ctx->op->u.rmsnorm.eps);
		} else {
			backend *t = OP_BACKEND(rmsnorm);
			st = t->rmsnorm(t, exec_slot(ctx, ctx->op->in[0]), w, exec_slot(ctx, ctx->op->out), n,
							ctx->op->u.rmsnorm.eps);
			if (st == OK) {
				backend *ta = OP_BACKEND(add_inplace);
				st			= ta->add_inplace(ta, exec_slot(ctx, ctx->op->out),
											  exec_slot(ctx, ctx->op->in[1]), n);
			}
		}
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_scale(exec_ctx *ctx) {
	backend					   *a	 = exec_layer_backend(ctx);
	profile					   *prof = &ctx->s->prof;
	profile_scope				ps	 = profile_begin(prof, ctx->op->stage);
	status_code					st	 = OK;
	const struct layer_weights *L	 = &ctx->m->layers[ctx->li];
	float						sc	 = ctx->op->u.scale.scale;
	if (ctx->m->arch_info->has_layer_output_scale && L && L->layer_out_scale != 1.0f)
		sc = L->layer_out_scale;
	if (sc == 1.0f)
		return OK;
	int scale_dim = ctx->m->dim;
	if (exec_is_batch(ctx)) {
		if (a->scale_inplace) {
			st = a->scale_inplace(a, exec_slot(ctx, ctx->op->in[0]), sc, scale_dim * ctx->n_rows);
		} else {
			if (a->synchronize)
				a->synchronize(a);
			float *base = (float *)exec_slot(ctx, ctx->op->in[0])->host_ptr;
			for (int row = 0; row < ctx->n_rows; row++) {
				float *xf = base + (size_t)row * scale_dim;
				for (int i = 0; i < scale_dim; i++)
					xf[i] *= sc;
			}
		}
	} else {
		if (a->scale_inplace) {
			st = a->scale_inplace(a, exec_slot(ctx, ctx->op->in[0]), sc, scale_dim);
		} else {
			if (a->synchronize)
				a->synchronize(a);
			compute_small_host_ensure(ctx->s, scale_dim, 0, 0);
			float	*xf	   = ctx->s->xf_host;
			buffer	*xb	   = exec_slot(ctx, ctx->op->in[0]);
			backend *owner = xb->owner;
			st			   = owner->buffer_read_f32(owner, xb, xf, scale_dim);
			if (st == OK) {
				for (int i = 0; i < scale_dim; i++)
					xf[i] *= sc;
				st = owner->buffer_write_f32(owner, xb, xf, scale_dim);
			}
		}
	}
	profile_end(prof, &ps);
	return st;
}

static status_code compute_forward_recipe_one(struct model *m, struct kvcache *cache,
											  struct compute_scratch *s, int token, int pos,
											  int flash_attn, float *logits_out, int own_batch,
											  int need_logits) {
	if (!m || !m->recipe)
		return ERR_INVALID_ARG;

	const model_recipe *r = m->recipe;
	status_code			st;

	backend	  *bk		 = m->backend;
	const bool has_begin = (own_batch && bk->begin_batch);
	const bool has_end	 = (own_batch && bk->end_batch);

	const bool mixed = model_mixed_backend_mode(m);
	if (mixed) {
		st = compute_scratch_ensure_mirror(s, m, cache->n_ctx);
		if (st != OK)
			return st;
	}
	s->active_backend	= NULL;
	s->active_is_mirror = 0;

	if (has_begin)
		bk->begin_batch(bk);

	for (int i = 0; i < r->n_pre_ops; i++) {
		st = exec_op(&r->pre_ops[i], m, cache, s, token, pos, -1, flash_attn, logits_out);
		if (st != OK) {
			if (has_end)
				bk->end_batch(bk);
			goto fail;
		}
	}

	int dev_ple_build = (bk->ple_combine && bk->matmul && bk->scale_inplace && bk->rmsnorm &&
						 m->layer_dims.per_layer_model_proj.buf.handle);
	if (!dev_ple_build) {
		if (has_end)
			bk->end_batch(bk);
		if (has_begin)
			bk->begin_batch(bk);
	}

	const recipe_op *ops_base	= r->per_layer_ops ? r->per_layer_ops : r->layer.ops;
	int				 ops_stride = r->per_layer_ops ? r->layer.n_ops : 0;

	for (int li = 0; li < m->n_layers; li++) {
		if (mixed) {
			backend *layer_be = model_layer_backend(m, li);
			st				  = compute_switch_active_backend(s, layer_be, m->dim);
			if (st != OK) {
				if (has_end)
					bk->end_batch(bk);
				goto fail;
			}
		}

		const recipe_op *lops = &ops_base[(size_t)li * ops_stride];
		int				 j	  = 0;
		while (j < r->layer.n_ops) {
			const recipe_op *rop = &lops[j];

			if (rop->kind == OP_NONE) {
				j++;
				continue;
			}

			if (rop->coalesce_run_len > 1 && !mixed) {
				int			run_len = rop->coalesce_run_len;
				status_code coalesced_status;
				int consumed = exec_matmul_run_coalesced(&lops[j], run_len, m, s, li, s->slots,
														 &coalesced_status);
				if (consumed == 0 && coalesced_status == OK && !s->active_is_mirror) {
					buffer *ys[RECIPE_COALESCE_MAX];
					for (int q = 0; q < run_len; q++)
						ys[q] = &s->slots[lops[j + q].out];
					consumed = exec_matmul_run_qonly(&lops[j], run_len, m, &s->slots[lops[j].in[0]],
													 ys, li, 1, &coalesced_status);
				}
				if (consumed < 0) {
					if (has_end)
						bk->end_batch(bk);
					st = coalesced_status;
					goto fail;
				}
				if (consumed > 0) {
					if (coalesced_status != OK) {
						if (has_end)
							bk->end_batch(bk);
						st = coalesced_status;
						goto fail;
					}
					j += consumed;
					continue;
				}
			}

			st = exec_op(rop, m, cache, s, token, pos, li, flash_attn, logits_out);
			if (st != OK) {
				if (has_end)
					bk->end_batch(bk);
				goto fail;
			}
			j++;
		}

		if (s->layer_cb)
			s->layer_cb(li + 1, m->n_layers, -1, 0, s->layer_cb_ud);

		if (s->interrupt && *s->interrupt) {
			if (has_end)
				bk->end_batch(bk);
			st = ERR_INTERRUPTED;
			goto fail;
		}
	}

	if (mixed) {
		st = compute_switch_active_backend(s, m->backend, m->dim);
		if (st != OK) {
			if (has_end)
				bk->end_batch(bk);
			goto fail;
		}
	}

	if (has_end)
		bk->end_batch(bk);

	if (need_logits) {
		for (int i = 0; i < r->n_post_ops; i++) {
			st = exec_op(&r->post_ops[i], m, cache, s, token, pos, -1, flash_attn, logits_out);
			if (st != OK)
				goto fail;
		}
	}

	return OK;

fail:
	if (logits_out)
		memset(logits_out, 0, (size_t)m->vocab_size * sizeof(float));
	return st;
}

status_code compute_forward_recipe(struct model *m, struct kvcache *cache,
								   struct compute_scratch *s, int token, int pos, int flash_attn,
								   float *logits_out) {
	return compute_forward_recipe_one(m, cache, s, token, pos, flash_attn, logits_out, 1, 1);
}

typedef struct {
	float_buf fb;
	buffer	  b;
} bs_pair;

typedef struct {
	int cnt;
	int u;
} moe_order_pair;

static int moe_order_pair_cmp(const void *pa, const void *pb) {
	const moe_order_pair *a = pa, *b = pb;
	if (a->cnt != b->cnt)
		return b->cnt - a->cnt;
	return a->u - b->u;
}

struct batch_scratch {
	bs_pair pair[RECIPE_SLOT_MAX];

	int	  *moe_router_ids;
	float *moe_router_w;
	int	  *moe_union_ids;
	int	   moe_n_union;

	int *moe_union_rows;
	int *moe_union_kidx;
	int *moe_union_offsets;
	int *moe_union_cursor;
	int *moe_union_count;

	int *moe_expert_union_idx;
	int *moe_expert_seen_gen;
	int	 moe_expert_gen;
	int	 moe_expert_alloc;

	int *moe_union_order;
	int	 moe_union_order_cap;
	int *moe_union_sorted_ids;
	int	 moe_union_sorted_ids_cap;

	moe_order_pair *moe_order_pairs;
	int				moe_order_pairs_cap;
	int				moe_router_ids_cap_tokens;
	int				moe_router_ids_cap_k;

	float_buf moe_expert_out;
	float_buf moe_out;
	float_buf moe_router_logits;
	float_buf moe_router_inp;
	float_buf moe_xb_f;

	float_buf moe_gather_x;
	float_buf moe_exp_gu;
	float_buf moe_up_scratch;
	float_buf moe_exp_act;
	float_buf moe_exp_y;

	moe_expert_slot *moe_per_token_slots;
	int				 moe_per_token_slots_cap;

	moe_expert_slot *moe_union_slots;
	int				 moe_union_slots_cap;
	int				 moe_union_pending_n;

	buffer moe_xb_q8_gate;
	int	   moe_xb_q8_gate_ok;

	const int32_t *tokens;
	int			   n_tokens_stashed;

	float_buf ple_buf;
	buffer	  ple_all;
	float_buf ple_inp;
	float_buf ple_proj;
	float_buf ple_slice;
};

static const uint8_t bs_slot_valid[RECIPE_SLOT_MAX] = {
	[RECIPE_SLOT_X] = 1,		[RECIPE_SLOT_XB] = 1,		   [RECIPE_SLOT_XB2] = 1,
	[RECIPE_SLOT_ATTN_OUT] = 1, [RECIPE_SLOT_Q] = 1,		   [RECIPE_SLOT_K] = 1,
	[RECIPE_SLOT_V] = 1,		[RECIPE_SLOT_FFN_GATE] = 1,	   [RECIPE_SLOT_FFN_UP] = 1,
	[RECIPE_SLOT_FFN_ACT] = 1,	[RECIPE_SLOT_FFN_GATE_UP] = 1, [RECIPE_SLOT_RESID_TMP] = 1,
	[RECIPE_SLOT_HYB_PROJ] = 1, [RECIPE_SLOT_HYB_GATE] = 1,	   [RECIPE_SLOT_HYB_ALPHA] = 1,
	[RECIPE_SLOT_HYB_BETA] = 1,
};

static uint32_t bs_slot_bit(uint8_t slot) {
	return (slot < RECIPE_SLOT_MAX) ? (1u << slot) : 0;
}

static uint32_t op_batch_slot_mask(const recipe_op *op) {
	uint32_t m = 0;
	for (int i = 0; i < 3; i++)
		if (op->in[i] != RECIPE_SLOT_NONE)
			m |= bs_slot_bit(op->in[i]);
	if (op->out != RECIPE_SLOT_NONE)
		m |= bs_slot_bit(op->out);

	switch (op->kind) {
	case OP_MATMUL:
	case OP_MATMUL_MULTI:
		m |= bs_slot_bit(RECIPE_SLOT_K) | bs_slot_bit(RECIPE_SLOT_V);
		break;
	case OP_MATMUL_RESIDUAL:
		m |= bs_slot_bit(RECIPE_SLOT_RESID_TMP);
		break;
	case OP_MATMUL_FUSED_GATEUP:
		m |= bs_slot_bit(RECIPE_SLOT_FFN_GATE_UP);
		break;
	case OP_MATMUL_FFN_DOWN:
		m |= bs_slot_bit(RECIPE_SLOT_FFN_ACT);
		break;
	case OP_SPLIT_QGATE:
		m |= bs_slot_bit(RECIPE_SLOT_HYB_PROJ) | bs_slot_bit(RECIPE_SLOT_Q) |
			 bs_slot_bit(RECIPE_SLOT_HYB_GATE);
		break;
	case OP_PARTIAL_ROPE_QK:
		m |= bs_slot_bit(RECIPE_SLOT_Q) | bs_slot_bit(RECIPE_SLOT_K);
		break;
	case OP_ATTN_OUTPUT_GATE:
		m |= bs_slot_bit(RECIPE_SLOT_HYB_GATE);
		break;
	case OP_GATED_DELTA_NET:
		m |= bs_slot_bit(RECIPE_SLOT_HYB_PROJ) | bs_slot_bit(RECIPE_SLOT_HYB_GATE) |
			 bs_slot_bit(RECIPE_SLOT_HYB_ALPHA) | bs_slot_bit(RECIPE_SLOT_HYB_BETA) |
			 bs_slot_bit(RECIPE_SLOT_XB2);
		break;
	case OP_MOE_ROUTER:
	case OP_MOE_EXPERTS:
	case OP_MOE_SHARED:
		m |= bs_slot_bit(RECIPE_SLOT_XB) | bs_slot_bit(RECIPE_SLOT_XB2) |
			 bs_slot_bit(RECIPE_SLOT_FFN_GATE) | bs_slot_bit(RECIPE_SLOT_FFN_UP) |
			 bs_slot_bit(RECIPE_SLOT_FFN_ACT) | bs_slot_bit(RECIPE_SLOT_FFN_GATE_UP) |
			 bs_slot_bit(RECIPE_SLOT_RESID_TMP);
		break;
	default:
		break;
	}
	return m & 0x1ffffu;
}

static void bs_ensure_slot(batch_scratch *bs, backend *owner, uint8_t slot, size_t n_elems) {
	bs_pair *p	   = &bs->pair[slot];
	size_t	 bytes = n_elems * sizeof(float);

	if (!backend_has_cap(owner, BCAP_IS_HOST)) {
		if (p->b.handle && p->b.owner)
			p->b.owner->buffer_free(p->b.owner, &p->b);
		memset(&p->b, 0, sizeof(p->b));
		owner->buffer_alloc_scratch(owner, bytes, &p->b);
	} else {
		float_buf_ensure_nocopy(&p->fb, n_elems, 64);
		p->b.handle	  = p->fb.p;
		p->b.host_ptr = p->fb.p;
		p->b.size	  = bytes;
		p->b.offset	  = 0;
		p->b.owner	  = owner;
	}
}

static size_t bs_slot_elems(uint8_t slot, int m, int q_out, int kv_out, int intermediate,
							int attn_buf_size, int ffn_act_size) {
	switch (slot) {
	case RECIPE_SLOT_X:
	case RECIPE_SLOT_XB:
	case RECIPE_SLOT_XB2:
	case RECIPE_SLOT_ATTN_OUT:
	case RECIPE_SLOT_RESID_TMP:
		return (size_t)m * (size_t)attn_buf_size;
	case RECIPE_SLOT_Q:
		return (size_t)m * (size_t)q_out;
	case RECIPE_SLOT_K:
	case RECIPE_SLOT_V:
		return (size_t)m * (size_t)kv_out;
	case RECIPE_SLOT_FFN_GATE:
	case RECIPE_SLOT_FFN_UP:
		return (size_t)m * (size_t)intermediate;
	case RECIPE_SLOT_FFN_GATE_UP:
		return (size_t)m * 2 * (size_t)intermediate;
	case RECIPE_SLOT_FFN_ACT:
		return (size_t)m * (size_t)ffn_act_size;
	default:
		return 0;
	}
}

static void batch_scratch_alloc(batch_scratch *bs, backend *owner, int m, int dim, int q_out,
								int kv_out, int intermediate, int attn_buf_size,
								uint32_t slot_mask) {
	int ffn_act_size = intermediate > dim ? intermediate : dim;
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (!bs_slot_valid[slot])
			continue;
		if (slot_mask && !(slot_mask & bs_slot_bit((uint8_t)slot)))
			continue;
		size_t elems = bs_slot_elems((uint8_t)slot, m, q_out, kv_out, intermediate, attn_buf_size,
									 ffn_act_size);
		if (elems)
			bs_ensure_slot(bs, owner, (uint8_t)slot, elems);
	}
}

void batch_scratch_free(batch_scratch *bs) {
	if (!bs)
		return;
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (bs->pair[slot].b.owner && !backend_has_cap(bs->pair[slot].b.owner, BCAP_IS_HOST))
			bs->pair[slot].b.owner->buffer_free(bs->pair[slot].b.owner, &bs->pair[slot].b);
		free(bs->pair[slot].fb.p);
		memset(&bs->pair[slot], 0, sizeof(bs->pair[slot]));
	}
	free(bs->moe_router_ids);
	free(bs->moe_router_w);
	free(bs->moe_union_ids);
	free(bs->moe_union_rows);
	free(bs->moe_union_kidx);
	free(bs->moe_union_offsets);
	free(bs->moe_union_cursor);
	free(bs->moe_union_count);
	free(bs->moe_expert_union_idx);
	free(bs->moe_expert_seen_gen);
	free(bs->moe_union_order);
	free(bs->moe_union_sorted_ids);
	free(bs->moe_order_pairs);
	free(bs->moe_expert_out.p);
	free(bs->moe_out.p);
	free(bs->moe_router_logits.p);
	free(bs->moe_router_inp.p);
	free(bs->moe_xb_f.p);
	free(bs->moe_gather_x.p);
	free(bs->moe_exp_gu.p);
	free(bs->moe_up_scratch.p);
	free(bs->moe_exp_act.p);
	free(bs->moe_exp_y.p);
	free(bs->moe_per_token_slots);
	free(bs->moe_union_slots);
	if (bs->moe_xb_q8_gate.owner)
		bs->moe_xb_q8_gate.owner->buffer_free(bs->moe_xb_q8_gate.owner, &bs->moe_xb_q8_gate);
	free(bs->ple_buf.p);
	if (bs->ple_all.owner)
		bs->ple_all.owner->buffer_free(bs->ple_all.owner, &bs->ple_all);
	free(bs->ple_inp.p);
	free(bs->ple_proj.p);
	free(bs->ple_slice.p);
	memset(bs, 0, sizeof(*bs));
}

static buffer *batch_slot(batch_scratch *bs, uint8_t slot) {
	if (slot >= RECIPE_SLOT_MAX || !bs_slot_valid[slot])
		return NULL;
	return &bs->pair[slot].b;
}

static buffer batch_row_view(const buffer *whole, int row, int row_elems) {
	buffer v = *whole;
	v.offset = whole->offset + ((size_t)row * row_elems * sizeof(float));
	return v;
}

static inline float *batch_buf_ptr(const buffer *b) {
	if (b->host_ptr)
		return (float *)((const char *)b->host_ptr + b->offset);
	return (float *)((char *)b->handle + b->offset);
}

static inline void batch_sync(const exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (a && a->synchronize && !backend_has_cap(a, BCAP_IS_HOST))
		a->synchronize(a);
}

typedef struct {
	model				  *m;
	int					   li;
	int					   top_k, inter, dim;
	int					   any_fused, use_gelu;
	int					   xb_q8_gate_ok;
	uint32_t			   gate_q8_type;
	const moe_expert_slot *slot_buf;
	const int			  *expert_ids;
	const float			  *weights;
	const float			  *xb_f;
	const buffer		  *xb_q8_gate;
	backend				  *a;
	float				  *all_scratch;
	size_t				   per_thread_scratch;
	float				  *all_outs;
	size_t				   per_thread_out;
	_Atomic int			   first_err;
	_Atomic int			   n_err;
} moe_par_job;

typedef struct {
	model			*m;
	backend			*a;
	batch_scratch	*bs;
	moe_expert_slot *slot_buf;
	int				 li, top_k, inter, dim, n_tokens, use_gelu;
	const float		*xb_base;
	float			*all_scratch;
	float			*all_outs;
	size_t			 per_thread_scratch;
	size_t			 per_thread_out;
} moe_batch_job;

int moe_topk_select(const float *scores, int n_experts, int top_k, int *top_idx, float *top_score) {
	if (top_k <= 0)
		return 0;
	if (top_k > MOE_MAX_TOPK)
		top_k = MOE_MAX_TOPK;
	int hn = topk_heap_select(scores, n_experts, top_k, top_score, top_idx);
	for (int k = hn; k < top_k; k++) {
		top_idx[k]	 = -1;
		top_score[k] = -1e30f;
	}
	return top_k;
}

static int moe_grouped_topk_select(const float *scores, int E, int K, int n_group, int topk_group,
								   int *top_idx, float *top_score) {
	if (n_group <= 1 || topk_group <= 0 || topk_group >= n_group || n_group > E)
		return moe_topk_select(scores, E, K, top_idx, top_score);

	enum { GROUP_STACK_CAP = 256, CAND_STACK_CAP = 2048 };
	float  group_sum_stack[GROUP_STACK_CAP];
	float  group_kept_stack[GROUP_STACK_CAP];
	int	   group_sel_stack[GROUP_STACK_CAP];
	float  cand_score_stack[CAND_STACK_CAP];
	int	   cand_eid_stack[CAND_STACK_CAP];
	float *group_sum  = group_sum_stack;
	float *group_kept = group_kept_stack;
	int	  *group_sel  = group_sel_stack;
	float *cand_score = cand_score_stack;
	int	  *cand_eid	  = cand_eid_stack;
	if (n_group > GROUP_STACK_CAP) {
		group_sum  = xmalloc((size_t)n_group * sizeof(float));
		group_kept = xmalloc((size_t)n_group * sizeof(float));
		group_sel  = xmalloc((size_t)n_group * sizeof(int));
	}
	if (E > CAND_STACK_CAP) {
		cand_score = xmalloc((size_t)E * sizeof(float));
		cand_eid   = xmalloc((size_t)E * sizeof(int));
	}

	int epg		= E / n_group;
	int rem		= E % n_group;
	int g_start = 0;
	for (int g = 0; g < n_group; g++) {
		int	  g_end	 = g_start + epg + (g < rem ? 1 : 0);
		float best	 = -1e30f;
		float second = -1e30f;
		for (int e = g_start; e < g_end; e++) {
			if (scores[e] > best) {
				second = best;
				best   = scores[e];
			} else if (scores[e] > second) {
				second = scores[e];
			}
		}
		group_sum[g] = best + second;
		g_start		 = g_end;
	}

	int n_kept = topk_heap_select(group_sum, n_group, topk_group, group_kept, group_sel);

	for (int i = 1; i < n_kept; i++) {
		int g = group_sel[i];
		int j = i - 1;
		while (j >= 0 && group_sel[j] > g) {
			group_sel[j + 1] = group_sel[j];
			j--;
		}
		group_sel[j + 1] = g;
	}

	int n_cand = 0;
	for (int i = 0; i < n_kept; i++) {
		int g  = group_sel[i];
		int gs = g * epg + (g < rem ? g : rem);
		int ge = gs + epg + (g < rem ? 1 : 0);
		for (int e = gs; e < ge; e++) {
			cand_score[n_cand] = scores[e];
			cand_eid[n_cand]   = e;
			n_cand++;
		}
	}

	int sel = moe_topk_select(cand_score, n_cand, K, top_idx, top_score);
	for (int k = 0; k < sel; k++)
		if (top_idx[k] >= 0)
			top_idx[k] = cand_eid[top_idx[k]];

	if (group_sum != group_sum_stack) {
		free(group_sum);
		free(group_kept);
		free(group_sel);
	}
	if (cand_score != cand_score_stack) {
		free(cand_score);
		free(cand_eid);
	}
	return sel;
}

void moe_apply_weights(float *weight, int n_k, int norm_topk, float routed_scale) {
	if (norm_topk) {
		float sum = 0.0f;
		for (int k = 0; k < n_k; k++)
			sum += weight[k];
		if (sum > 1e-20f) {
			float inv = routed_scale / sum;
			for (int k = 0; k < n_k; k++)
				weight[k] *= inv;
			return;
		}
	}
	for (int k = 0; k < n_k; k++)
		weight[k] *= routed_scale;
}

void moe_activate(float *act, const float *gate, const float *up, int n_i, float gs, float us,
				  int use_gelu) {
	if (use_gelu) {
		for (int i = 0; i < n_i; i++)
			act[i] = gelu_tanh(gate[i] * gs) * (up[i] * us);
	} else {
		for (int i = 0; i < n_i; i++)
			act[i] = silu(gate[i] * gs) * (up[i] * us);
	}
}

void moe_router_normalize_input(const struct model *m, const struct layer_weights *L, int dim,
								float *router_input) {
	float ss = 0.0f;
	for (int i = 0; i < dim; i++)
		ss += router_input[i] * router_input[i];
	float inv		 = 1.0f / sqrtf((ss / (float)dim) + m->norm_eps);
	float base_scale = inv * m->moe.router_dim_scale;
	if (L->router_scale_w.host_ptr) {
		const float *scale_w = (const float *)L->router_scale_w.host_ptr;
		for (int i = 0; i < dim; i++)
			router_input[i] *= base_scale * scale_w[i];
	} else {
		for (int i = 0; i < dim; i++)
			router_input[i] *= base_scale;
	}
}

static int moe_router_logits_finite(const float *logits, int n) {
	for (int e = 0; e < n; e++)
		if (!isfinite(logits[e]))
			return 0;
	return 1;
}

int moe_router_emit(int E, int K, int use_softmax, int norm_topk, float routed_scale, float *logits,
					const float *bias, float *scores_scratch, int *ids_out, float *w_out) {
	return moe_router_emit_ex(E, K, use_softmax, norm_topk, routed_scale, 1, 0, logits, bias,
							  scores_scratch, ids_out, w_out);
}

int moe_router_emit_ex(int E, int K, int use_softmax, int norm_topk, float routed_scale,
					   int n_group, int topk_group, float *logits, const float *bias,
					   float *scores_scratch, int *ids_out, float *w_out) {
	int	  top_idx[MOE_MAX_TOPK];
	float top_score[MOE_MAX_TOPK];
	float weight[MOE_MAX_TOPK];

	if (use_softmax) {
		float max_logit = logits[0];
		for (int e = 1; e < E; e++)
			if (logits[e] > max_logit)
				max_logit = logits[e];
		float sum_exp = 0.0f;
		for (int e = 0; e < E; e++) {
			logits[e] = expf(logits[e] - max_logit);
			sum_exp += logits[e];
		}
		for (int e = 0; e < E; e++)
			logits[e] /= sum_exp;
	} else {
		float *scores = scores_scratch;
		float  scores_stack[8192];
		if (!scores)
			scores = (E <= (int)ARRAY_LEN(scores_stack)) ? scores_stack
														 : xmalloc((size_t)E * sizeof(float));
		for (int e = 0; e < E; e++)
			scores[e] = (1.0f / (1.0f + expf(-logits[e]))) + (bias ? bias[e] : 0.0f);
		K = moe_grouped_topk_select(scores, E, K, n_group, topk_group, top_idx, top_score);
		if (scores != scores_scratch && scores != scores_stack)
			free(scores);
		for (int k = 0; k < K; k++)
			weight[k] = (top_idx[k] >= 0) ? top_score[k] - (bias ? bias[top_idx[k]] : 0.0f) : 0.0f;
		moe_apply_weights(weight, K, norm_topk, routed_scale);
		for (int k = 0; k < K; k++) {
			ids_out[k] = top_idx[k];
			w_out[k]   = weight[k];
		}
		return K;
	}

	K = moe_topk_select(logits, E, K, top_idx, top_score);
	for (int k = 0; k < K; k++)
		weight[k] = top_score[k];
	moe_apply_weights(weight, K, norm_topk, routed_scale);
	for (int k = 0; k < K; k++) {
		ids_out[k] = top_idx[k];
		w_out[k]   = weight[k];
	}
	return K;
}

static status_code op_moe_router_softmax(backend *a, layer_weights *L, int E, int K, int dim,
										 int norm_topk, float routed_scale, float *router_input,
										 struct compute_scratch *s, buffer *slots) {
	float  logits_fallback[8192];
	float *logits = logits_fallback;
	if (E > (int)(sizeof(logits_fallback) / sizeof(float)))
		logits = xmalloc((size_t)E * sizeof(float));
	status_code st;

	buffer *inp_buf = &s->router_softmax_inp;
	st				= buffer_ensure_scratch(a, inp_buf, (size_t)dim * sizeof(float));
	if (st != OK) {
		if (logits != logits_fallback)
			free(logits);
		return st;
	}
	status_code st2 = a->buffer_write_f32(a, inp_buf, router_input, dim);
	if (st2 != OK) {
		if (logits != logits_fallback)
			free(logits);
		return st2;
	}
	status_code st3 = buffer_ensure_scratch(a, &s->router_logits, (size_t)E * sizeof(float));
	if (st3 != OK) {
		if (logits != logits_fallback)
			free(logits);
		return st3;
	}

	st = a->matmul(a, &L->router_w.buf, L->router_w.type, inp_buf, &s->router_logits, E, dim);
	if (st == OK)
		st = a->buffer_read_f32(a, &s->router_logits, logits, E);
	if (st != OK) {
		if (logits != logits_fallback)
			free(logits);
		return st;
	}

	if (!moe_router_logits_finite(logits, E)) {
		ERROR("op_moe_router_softmax: non-finite router logit (E=%d)", E);
		if (logits != logits_fallback)
			free(logits);
		return ERR_FORMAT;
	}

	int	  *idx_out = (int *)slots[RECIPE_SLOT_ROUTER_IDS].handle;
	float *w_out   = (float *)slots[RECIPE_SLOT_ROUTER_W].handle;
	moe_router_emit(E, K, 1, norm_topk, routed_scale, logits, NULL, NULL, idx_out, w_out);
	if (logits != logits_fallback)
		free(logits);
	return OK;
}

static status_code op_moe_router_direct(backend *a, const model *m, layer_weights *L, int E, int K,
										int dim, int norm_topk, float routed_scale, buffer *inp,
										struct compute_scratch *s, buffer *slots) {
	float  logits_fallback[8192];
	float *logits = logits_fallback;
	if (E > (int)(sizeof(logits_fallback) / sizeof(float)))
		logits = xmalloc((size_t)E * sizeof(float));

	status_code st;
	if (backend_has_cap(a, BCAP_IS_HOST)) {
		buffer logits_buf	= {0};
		logits_buf.handle	= logits;
		logits_buf.host_ptr = logits;
		logits_buf.size		= (size_t)E * sizeof(float);
		st = a->matmul(a, &L->router_w.buf, L->router_w.type, inp, &logits_buf, E, dim);
	} else {
		buffer	   *logits_buf = &s->router_logits;
		status_code ast		   = buffer_ensure_scratch(a, logits_buf, (size_t)E * sizeof(float));
		if (ast != OK) {
			if (logits != logits_fallback)
				free(logits);
			return ast;
		}
		st = a->matmul(a, &L->router_w.buf, L->router_w.type, inp, logits_buf, E, dim);
		if (st == OK)
			st = a->buffer_read_f32(a, logits_buf, logits, E);
	}
	if (st != OK) {
		if (logits != logits_fallback)
			free(logits);
		return st;
	}

	if (!moe_router_logits_finite(logits, E)) {
		ERROR("op_moe_router_direct: non-finite router logit (E=%d)", E);
		if (logits != logits_fallback)
			free(logits);
		return ERR_FORMAT;
	}

	const float *bias	 = (const float *)L->router_bias.host_ptr;
	int			*idx_out = (int *)slots[RECIPE_SLOT_ROUTER_IDS].handle;
	float		*w_out	 = (float *)slots[RECIPE_SLOT_ROUTER_W].handle;
	moe_router_emit_ex(E, K, 0, norm_topk, routed_scale, m->moe.n_group, m->moe.topk_group, logits,
					   bias, NULL, idx_out, w_out);
	if (logits != logits_fallback)
		free(logits);
	return OK;
}

typedef struct {
	batch_scratch *bs;
	layer_weights *L;
	int			   E, K, rows;
	int			   uses_softmax, norm_topk;
	float		   routed_scale;
	int			   n_group, topk_group;
} moe_router_emit_job;

static void moe_router_emit_chunk(int begin, int end, int tid, void *v) {
	(void)tid;
	moe_router_emit_job *j = v;
	for (int row = begin; row < end; row++) {
		float		*logits	 = j->bs->moe_router_logits.p + (size_t)row * j->E;
		int			*row_ids = j->bs->moe_router_ids + (size_t)row * j->K;
		float		*row_w	 = j->bs->moe_router_w + (size_t)row * j->K;
		const float *bias	 = (const float *)j->L->router_bias.host_ptr;
		int			 K_row	 = moe_router_emit_ex(
			j->E, j->K, j->uses_softmax, j->norm_topk, j->routed_scale, j->n_group, j->topk_group,
			logits, bias, j->bs->moe_router_logits.p + (size_t)j->rows * j->E + (size_t)row * j->E,
			row_ids, row_w);
		for (int k = K_row; k < j->K; k++) {
			row_ids[k] = -1;
			row_w[k]   = 0.0f;
		}
	}
}

static status_code moe_router_batch(exec_ctx *ctx) {

	backend *a = exec_layer_backend(ctx);
	if (!model_layer_is_moe(ctx->m, ctx->li))
		return OK;

	int			   E			= ctx->m->moe.n_experts;
	int			   K			= ctx->m->moe.n_experts_used;
	int			   router_dim	= ctx->op->u.matmul.k;
	float		   routed_scale = ctx->m->moe.routed_scale;
	int			   norm_topk	= ctx->m->moe.norm_topk_prob;
	int			   uses_softmax = ctx->m->arch_info->uses_moe_softmax_router;
	layer_weights *L			= &ctx->m->layers[ctx->li];

	if (ctx->bs->moe_router_ids_cap_tokens < ctx->n_rows || ctx->bs->moe_router_ids_cap_k < K) {
		ctx->bs->moe_router_ids =
			xrealloc(ctx->bs->moe_router_ids, (size_t)ctx->n_rows * K * sizeof(int));
		ctx->bs->moe_router_w =
			xrealloc(ctx->bs->moe_router_w, (size_t)ctx->n_rows * K * sizeof(float));
		ctx->bs->moe_union_ids =
			xrealloc(ctx->bs->moe_union_ids, (size_t)ctx->n_rows * K * sizeof(int));

		size_t slot_cap			= (size_t)ctx->n_rows * K;
		ctx->bs->moe_union_rows = xrealloc(ctx->bs->moe_union_rows, slot_cap * sizeof(int));
		ctx->bs->moe_union_kidx = xrealloc(ctx->bs->moe_union_kidx, slot_cap * sizeof(int));
		ctx->bs->moe_union_offsets =
			xrealloc(ctx->bs->moe_union_offsets, (slot_cap + 1) * sizeof(int));
		ctx->bs->moe_union_cursor = xrealloc(ctx->bs->moe_union_cursor, slot_cap * sizeof(int));
		ctx->bs->moe_union_count  = xrealloc(ctx->bs->moe_union_count, slot_cap * sizeof(int));
		ctx->bs->moe_router_ids_cap_tokens = ctx->n_rows;
		ctx->bs->moe_router_ids_cap_k	   = K;
	}
	if (ctx->bs->moe_expert_alloc < E) {
		ctx->bs->moe_expert_union_idx =
			xrealloc(ctx->bs->moe_expert_union_idx, (size_t)E * sizeof(int));
		ctx->bs->moe_expert_seen_gen =
			xrealloc(ctx->bs->moe_expert_seen_gen, (size_t)E * sizeof(int));
		memset(ctx->bs->moe_expert_seen_gen, 0, (size_t)E * sizeof(int));
		ctx->bs->moe_expert_gen	  = 0;
		ctx->bs->moe_expert_alloc = E;
	}
	float_buf_ensure(&ctx->bs->moe_router_logits, (size_t)ctx->n_rows * E * 2);
	float_buf_ensure(&ctx->bs->moe_router_inp, (size_t)ctx->n_rows * router_dim);
	float_buf_ensure(&ctx->bs->moe_xb_f, (size_t)ctx->n_rows * router_dim);

	buffer *xb = batch_slot(ctx->bs, ctx->op->in[0]);

	int			n_rows	= ctx->n_rows;
	float	   *xb_host = xmalloc((size_t)n_rows * router_dim * sizeof(float));
	status_code st		= a->buffer_read_f32(a, xb, xb_host, n_rows * router_dim);
	if (st != OK) {
		free(xb_host);
		return st;
	}

	for (int row = 0; row < n_rows; row++) {
		const float *router_input_src = xb_host + (size_t)row * router_dim;
		float		*router_input	  = ctx->bs->moe_router_inp.p + (size_t)row * router_dim;
		memcpy(router_input, router_input_src, (size_t)router_dim * sizeof(float));

		if (uses_softmax)
			moe_router_normalize_input(ctx->m, L, router_dim, router_input);
	}
	free(xb_host);

	buffer router_inp_buf = {0};
	st = a->buffer_alloc_scratch(a, (size_t)ctx->n_rows * router_dim * sizeof(float),
								 &router_inp_buf);
	if (st != OK)
		return st;

	st = a->buffer_write_f32(a, &router_inp_buf, ctx->bs->moe_router_inp.p,
							 ctx->n_rows * router_dim);
	if (st != OK) {
		a->buffer_free(a, &router_inp_buf);
		return st;
	}

	buffer router_logits_buf = {0};
	st = a->buffer_alloc_scratch(a, (size_t)ctx->n_rows * E * sizeof(float), &router_logits_buf);
	if (st != OK) {
		a->buffer_free(a, &router_inp_buf);
		return st;
	}

	st = a->matmul_batch(a, &L->router_w.buf, L->router_w.type, &router_inp_buf, &router_logits_buf,
						 E, router_dim, ctx->n_rows);
	if (st != OK) {
		a->buffer_free(a, &router_inp_buf);
		a->buffer_free(a, &router_logits_buf);
		return st;
	}

	st = a->buffer_read_f32(a, &router_logits_buf, ctx->bs->moe_router_logits.p, ctx->n_rows * E);
	a->buffer_free(a, &router_inp_buf);
	if (st != OK) {
		a->buffer_free(a, &router_logits_buf);
		return st;
	}

	if (!moe_router_logits_finite(ctx->bs->moe_router_logits.p, (size_t)ctx->n_rows * E)) {
		ERROR("moe_router_batch: non-finite router logit (layer=%d, rows=%d, E=%d)", ctx->li,
			  ctx->n_rows, E);
		a->buffer_free(a, &router_logits_buf);
		return ERR_FORMAT;
	}

	tpool *router_pool = (ctx->m->backend && ctx->m->backend->get_pool)
							 ? ctx->m->backend->get_pool(ctx->m->backend)
							 : NULL;
	if (router_pool && ctx->n_rows >= 8 && E >= 8 && tpool_current_tid() < 0) {
		moe_router_emit_job emit_job = {.bs			  = ctx->bs,
										.L			  = L,
										.E			  = E,
										.K			  = K,
										.rows		  = ctx->n_rows,
										.uses_softmax = uses_softmax,
										.norm_topk	  = norm_topk,
										.routed_scale = routed_scale,
										.n_group	  = ctx->m->moe.n_group,
										.topk_group	  = ctx->m->moe.topk_group};
		tpool_parallel_for(router_pool, ctx->n_rows, 1, moe_router_emit_chunk, &emit_job);
	} else {
		for (int row = 0; row < ctx->n_rows; row++) {
			float		*logits	 = ctx->bs->moe_router_logits.p + (size_t)row * E;
			int			*row_ids = ctx->bs->moe_router_ids + (size_t)row * K;
			float		*row_w	 = ctx->bs->moe_router_w + (size_t)row * K;
			const float *bias	 = (const float *)L->router_bias.host_ptr;
			int			 K_row	 = moe_router_emit_ex(
				E, K, uses_softmax, norm_topk, routed_scale, ctx->m->moe.n_group,
				ctx->m->moe.topk_group, logits, bias,
				ctx->bs->moe_router_logits.p + (size_t)ctx->n_rows * E + (size_t)row * E, row_ids,
				row_w);
			for (int k = K_row; k < K; k++) {
				row_ids[k] = -1;
				row_w[k]   = 0.0f;
			}
		}
	}

	int gen = ++ctx->bs->moe_expert_gen;

	int n_union = 0;
	for (int row = 0; row < ctx->n_rows; row++) {
		const int *row_ids = ctx->bs->moe_router_ids + (size_t)row * K;
		for (int k = 0; k < K; k++) {
			int eid = row_ids[k];
			if (eid < 0)
				continue;
			int union_idx;
			if (ctx->bs->moe_expert_seen_gen[eid] == gen) {
				union_idx = ctx->bs->moe_expert_union_idx[eid];
			} else {
				union_idx							= n_union++;
				ctx->bs->moe_expert_seen_gen[eid]	= gen;
				ctx->bs->moe_expert_union_idx[eid]	= union_idx;
				ctx->bs->moe_union_ids[union_idx]	= eid;
				ctx->bs->moe_union_count[union_idx] = 0;
			}
			ctx->bs->moe_union_count[union_idx]++;
		}
	}
	ctx->bs->moe_n_union = n_union;

	ctx->bs->moe_union_offsets[0] = 0;
	for (int u = 0; u < n_union; u++)
		ctx->bs->moe_union_offsets[u + 1] =
			ctx->bs->moe_union_offsets[u] + ctx->bs->moe_union_count[u];

	{
		int *cursor = ctx->bs->moe_union_cursor;
		for (int u = 0; u < n_union; u++)
			cursor[u] = ctx->bs->moe_union_offsets[u];
		for (int row = 0; row < ctx->n_rows; row++) {
			const int *row_ids = ctx->bs->moe_router_ids + (size_t)row * K;
			for (int k = 0; k < K; k++) {
				if (row_ids[k] < 0)
					continue;
				int u						  = ctx->bs->moe_expert_union_idx[row_ids[k]];
				int slot					  = cursor[u]++;
				ctx->bs->moe_union_rows[slot] = row;
				ctx->bs->moe_union_kidx[slot] = k;
			}
		}
	}

	{
		if (ctx->bs->moe_union_order_cap < n_union) {
			free(ctx->bs->moe_union_order);
			ctx->bs->moe_union_order	 = xmalloc((size_t)n_union * sizeof(int));
			ctx->bs->moe_union_order_cap = n_union;
		}
		if (ctx->bs->moe_union_sorted_ids_cap < n_union) {
			free(ctx->bs->moe_union_sorted_ids);
			ctx->bs->moe_union_sorted_ids	  = xmalloc((size_t)n_union * sizeof(int));
			ctx->bs->moe_union_sorted_ids_cap = n_union;
		}
		if (ctx->bs->moe_order_pairs_cap < n_union) {
			free(ctx->bs->moe_order_pairs);
			ctx->bs->moe_order_pairs	 = xmalloc((size_t)n_union * sizeof(moe_order_pair));
			ctx->bs->moe_order_pairs_cap = n_union;
		}
		for (int u = 0; u < n_union; u++)
			ctx->bs->moe_order_pairs[u] =
				(moe_order_pair){.cnt = ctx->bs->moe_union_count[u], .u = u};
		qsort(ctx->bs->moe_order_pairs, (size_t)n_union, sizeof(moe_order_pair),
			  moe_order_pair_cmp);
		for (int i = 0; i < n_union; i++) {
			ctx->bs->moe_union_order[i] = ctx->bs->moe_order_pairs[i].u;
			ctx->bs->moe_union_sorted_ids[i] =
				ctx->bs->moe_union_ids[ctx->bs->moe_order_pairs[i].u];
		}
	}

	{
		int total_slots = ctx->n_rows * K;
		if (ctx->bs->moe_per_token_slots_cap < total_slots) {
			free(ctx->bs->moe_per_token_slots);
			ctx->bs->moe_per_token_slots = xmalloc((size_t)total_slots * sizeof(moe_expert_slot));
			ctx->bs->moe_per_token_slots_cap = total_slots;
		}
	}

	if (ctx->bs->moe_union_slots_cap < n_union) {
		free(ctx->bs->moe_union_slots);
		ctx->bs->moe_union_slots	 = xcalloc((size_t)n_union, sizeof(moe_expert_slot));
		ctx->bs->moe_union_slots_cap = n_union;
	}

	status_code rst = moe_stream_resolve(ctx->m, ctx->li, ctx->bs->moe_union_sorted_ids, n_union,
										 ctx->bs->moe_union_slots);
	if (rst != OK) {
		ERROR("moe_router_batch: moe_stream_resolve failed (layer=%d, union=%d, st=%d)", ctx->li,
			  n_union, rst);
		ctx->bs->moe_n_union		 = 0;
		ctx->bs->moe_union_pending_n = 0;
		a->buffer_free(a, &router_logits_buf);
		return ERR_INTERNAL;
	}
	ctx->bs->moe_union_pending_n = n_union;
	a->buffer_free(a, &router_logits_buf);
	return OK;
}

static status_code moe_experts_grouped(exec_ctx *ctx, backend *a, int dim, int K, int I,
									   int use_gelu) {
	batch_scratch *bs	   = ctx->bs;
	int			   n_rows  = ctx->n_rows;
	int			   n_union = bs->moe_n_union;

	buffer	   *xb_dev	= batch_slot(bs, ctx->op->in[0]);
	float	   *xb_host = xmalloc((size_t)n_rows * dim * sizeof(float));
	status_code st		= a->buffer_read_f32(a, xb_dev, xb_host, n_rows * dim);
	if (st != OK) {
		free(xb_host);
		return st;
	}
	const float *xb_base = xb_host;
	float		*out	 = bs->moe_out.p;

	float_buf_ensure(&bs->moe_gather_x, (size_t)n_rows * dim);
	float_buf_ensure(&bs->moe_exp_act, (size_t)n_rows * I);
	float_buf_ensure(&bs->moe_exp_y, (size_t)n_rows * dim);

	for (int i = 0; i < n_union; i++) {
		int				 u	  = bs->moe_union_order[i];
		int				 cnt  = bs->moe_union_count[u];
		const int		*rows = bs->moe_union_rows + bs->moe_union_offsets[u];
		const int		*kidx = bs->moe_union_kidx + bs->moe_union_offsets[u];
		moe_expert_slot *es	  = &bs->moe_union_slots[i];
		if (cnt <= 0 || es->eid < 0 || !es->gate_w)
			continue;
		moe_stream_wait_slot(es);

		float *X = bs->moe_gather_x.p;
		for (int c = 0; c < cnt; c++)
			memcpy(X + (size_t)c * dim, xb_base + (size_t)rows[c] * dim,
				   (size_t)dim * sizeof(float));
		float *A = bs->moe_exp_act.p;

		if (es->gate_up_fused) {
			float *GU = float_buf_ensure(&bs->moe_exp_gu, (size_t)cnt * I * 2);
			for (int c = 0; c < cnt; c++)
				matmul_generic_f32(es->gate_w, es->gate_type, X + (size_t)c * dim,
								   GU + (size_t)c * I * 2, I * 2, dim);
			for (int c = 0; c < cnt; c++)
				moe_activate(A + (size_t)c * I, GU + (size_t)c * I * 2, GU + (size_t)c * I * 2 + I,
							 I, es->gate_scale, es->up_scale, use_gelu);
		} else {
			float *G = float_buf_ensure(&bs->moe_exp_gu, (size_t)cnt * I);
			float *U = float_buf_ensure(&bs->moe_up_scratch, (size_t)cnt * I);
			for (int c = 0; c < cnt; c++) {
				matmul_generic_f32(es->gate_w, es->gate_type, X + (size_t)c * dim,
								   G + (size_t)c * I, I, dim);
				matmul_generic_f32(es->up_w, es->up_type, X + (size_t)c * dim, U + (size_t)c * I, I,
								   dim);
			}
			for (int c = 0; c < cnt; c++)
				moe_activate(A + (size_t)c * I, G + (size_t)c * I, U + (size_t)c * I, I,
							 es->gate_scale, es->up_scale, use_gelu);
		}

		float *Y = bs->moe_exp_y.p;
		for (int c = 0; c < cnt; c++)
			matmul_generic_f32(es->down_w, es->down_type, A + (size_t)c * I, Y + (size_t)c * dim,
							   dim, I);

		float ds = es->down_scale;
		for (int c = 0; c < cnt; c++) {
			float		 w = bs->moe_router_w[(size_t)rows[c] * K + kidx[c]];
			const float *y = Y + (size_t)c * dim;
			float		*o = out + (size_t)rows[c] * dim;
			if (ds == 1.0f) {
				for (int d = 0; d < dim; d++)
					o[d] += w * y[d];
			} else {
				for (int d = 0; d < dim; d++)
					o[d] += w * (ds * y[d]);
			}
		}
	}
	free(xb_host);
	return OK;
}

static status_code moe_experts_resident_grouped(exec_ctx *ctx, backend *a, int dim, int K, int I,
												int use_gelu) {
	batch_scratch *bs	   = ctx->bs;
	int			   n_rows  = ctx->n_rows;
	int			   n_union = bs->moe_n_union;

	buffer *xb_dev = batch_slot(bs, ctx->op->in[0]);
	buffer *out_b  = batch_slot(bs, ctx->op->out);
	if (xb_dev->owner != a || out_b->owner != a)
		return ERR_UNSUPPORTED;

	int counts[MOE_MAX_TOPK * 8];
	if (n_union > (int)(sizeof(counts) / sizeof(counts[0])))
		return ERR_UNSUPPORTED;

	moe_resident_expert *experts		= xcalloc((size_t)n_union, sizeof(*experts));
	int					*rows_packed	= xcalloc((size_t)n_rows * K, sizeof(int));
	float				*weights_packed = xcalloc((size_t)n_rows * K, sizeof(float));
	int					 packed			= 0;
	for (int i = 0; i < n_union; i++) {
		int				 u	  = bs->moe_union_order[i];
		int				 cnt  = bs->moe_union_count[u];
		const int		*rows = bs->moe_union_rows + bs->moe_union_offsets[u];
		const int		*kidx = bs->moe_union_kidx + bs->moe_union_offsets[u];
		moe_expert_slot *es	  = &bs->moe_union_slots[i];
		if (cnt <= 0 || es->eid < 0 || !es->gate_w || !es->dev_ready) {
			counts[i] = 0;
			continue;
		}
		counts[i]				 = cnt;
		experts[i].gate_w		 = &es->dev_gate;
		experts[i].up_w			 = &es->dev_up;
		experts[i].down_w		 = &es->dev_down;
		experts[i].gate_type	 = es->gate_type;
		experts[i].up_type		 = es->up_type;
		experts[i].down_type	 = es->down_type;
		experts[i].gate_up_fused = es->gate_up_fused;
		experts[i].use_gelu		 = use_gelu;
		experts[i].gate_scale	 = es->gate_scale;
		experts[i].up_scale		 = es->up_scale;
		experts[i].down_scale	 = es->down_scale;
		for (int c = 0; c < cnt; c++) {
			rows_packed[packed]	   = rows[c];
			weights_packed[packed] = ctx->bs->moe_router_w[(size_t)rows[c] * K + kidx[c]];
			packed++;
		}
	}

	status_code st = a->scale_inplace(a, out_b, 0.0f, n_rows * dim);
	if (st == OK)
		st = a->moe_experts_batch(a, xb_dev, out_b, n_rows, dim, I, use_gelu, n_union, experts,
								  counts, rows_packed, weights_packed);

	free(experts);
	free(rows_packed);
	free(weights_packed);
	return st;
}

static status_code moe_experts_batch(exec_ctx *ctx) {

	backend	 *a	  = exec_layer_backend(ctx);
	const int dim = ctx->m->dim;
	if (!model_layer_is_moe(ctx->m, ctx->li)) {
		buffer *dst = batch_slot(ctx->bs, ctx->op->out);

		float	   *zeros = xcalloc((size_t)ctx->n_rows * dim, sizeof(float));
		status_code r	  = a->buffer_write_f32(a, dst, zeros, ctx->n_rows * dim);
		free(zeros);
		return r;
	}

	int K		 = ctx->m->moe.n_experts_used;
	int I		 = ctx->m->moe.moe_intermediate;
	int use_gelu = ctx->m->arch_info->uses_gelu_activation;

	int n_union = ctx->bs->moe_union_pending_n;
	for (int i = 0; i < ctx->n_rows * K; i++)
		ctx->bs->moe_per_token_slots[i].eid = -1;
	for (int i = 0; i < n_union; i++) {
		int		   u	= ctx->bs->moe_union_order[i];
		int		   cnt	= ctx->bs->moe_union_count[u];
		const int *rows = ctx->bs->moe_union_rows + ctx->bs->moe_union_offsets[u];
		const int *kidx = ctx->bs->moe_union_kidx + ctx->bs->moe_union_offsets[u];
		for (int c = 0; c < cnt; c++) {
			moe_expert_slot slot = ctx->bs->moe_union_slots[i];
			if (c > 0) {
				slot.owned	  = 0;
				slot.heap_buf = NULL;
			}
			ctx->bs->moe_per_token_slots[(size_t)rows[c] * K + kidx[c]] = slot;
		}
		if (cnt > 1)
			moe_stream_record_extra_hits(ctx->m, ctx->li, cnt - 1, cnt - 1,
										 ctx->bs->moe_union_slots[i].pinned);
	}

	for (int row = 0; row < ctx->n_rows; row++) {
		const int	*row_ids = ctx->bs->moe_router_ids + (size_t)row * K;
		const float *row_w	 = ctx->bs->moe_router_w + (size_t)row * K;
		monitor_emit_moe_experts(g_monitor, ctx->li, -1, row_ids, row_w, K);
	}

	float_buf_ensure(&ctx->bs->moe_out, (size_t)ctx->n_rows * dim);
	memset(ctx->bs->moe_out.p, 0, ctx->n_rows * dim * sizeof(float));

	status_code st = OK;
	if (ctx->n_rows > 1 && a->matmul_batch) {
		if (backend_has_cap(a, BCAP_MOE_EXPERT_RESIDENT) && a->moe_experts_batch &&
			ctx->m->moe.experts_resident) {
			st = moe_experts_resident_grouped(ctx, a, dim, K, I, use_gelu);
			if (st == OK) {
				for (int i = 0; i < n_union; i++) {
					moe_expert_slot *us = &ctx->bs->moe_union_slots[i];
					if (us->owned && us->heap_buf) {
						free(us->heap_buf);
						us->heap_buf = NULL;
						us->eid		 = -1;
					}
				}
				moe_stream_release_slots(ctx->m, ctx->li, ctx->bs->moe_union_slots, n_union);
				for (int i = 0; i < n_union; i++)
					ctx->bs->moe_union_slots[i].eid = -1;
				for (int i = 0; i < ctx->n_rows * K; i++) {
					moe_expert_slot *es = &ctx->bs->moe_per_token_slots[i];
					es->eid				= -1;
					es->owned			= 0;
					es->heap_buf		= NULL;
					es->gate_w			= NULL;
					es->up_w			= NULL;
					es->down_w			= NULL;
				}
				return OK;
			}
			WARN("moe resident expert batch failed (st=%d) -- falling back to cpu", (int)st);
		}
		st = moe_experts_grouped(ctx, a, dim, K, I, use_gelu);
		goto finish;
	}

	tpool *pool			 = (ctx->m->backend && ctx->m->backend->get_pool)
							   ? ctx->m->backend->get_pool(ctx->m->backend)
							   : NULL;
	int	   total_experts = ctx->n_rows * K;

	if (pool && total_experts >= 2) {
		int n_threads = tpool_n_threads(pool);
		if (n_threads < 1)
			n_threads = 1;
		if (n_threads > total_experts)
			n_threads = total_experts;

		size_t per_thread_scratch = (size_t)I * 3 + (size_t)dim + (size_t)I * 2;
		float *all_scratch =
			float_buf_ensure(&ctx->s->moe_all_scratch, (size_t)n_threads * per_thread_scratch);
		size_t per_thread_out = (size_t)ctx->n_rows * dim;
		float *all_outs =
			float_buf_ensure(&ctx->s->moe_all_outs, (size_t)n_threads * per_thread_out);
		for (int t = 0; t < n_threads; t++) {
			float *o = all_outs + (size_t)t * per_thread_out;
			memset(o, 0, per_thread_out * sizeof(float));
		}

		moe_batch_job job;
		job.m		 = ctx->m;
		job.a		 = ctx->m->backend;
		job.bs		 = ctx->bs;
		job.slot_buf = ctx->bs->moe_per_token_slots;
		job.li		 = ctx->li;
		job.top_k	 = K;
		job.inter	 = I;
		job.dim		 = dim;
		job.n_tokens = ctx->n_rows;
		job.use_gelu = use_gelu;
		{
			buffer *xb_b = batch_slot(ctx->bs, ctx->op->in[0]);
			float  *xb_h = xmalloc((size_t)ctx->n_rows * dim * sizeof(float));
			st			 = a->buffer_read_f32(a, xb_b, xb_h, ctx->n_rows * dim);
			if (st != OK) {
				free(xb_h);
				return st;
			}
			job.xb_base = xb_h;
		}
		job.all_scratch		   = all_scratch;
		job.all_outs		   = all_outs;
		job.per_thread_scratch = per_thread_scratch;
		job.per_thread_out	   = per_thread_out;

		tpool_parallel_for(pool, total_experts, 1, moe_batch_expert_chunk, &job);
		free((void *)job.xb_base);

		for (int t = 0; t < n_threads; t++) {
			float *o = all_outs + (size_t)t * per_thread_out;
			for (int i = 0; i < ctx->n_rows * dim; i++)
				ctx->bs->moe_out.p[i] += o[i];
		}
	} else {
		float *scratch =
			float_buf_ensure(&ctx->s->moe_scratch, ((size_t)I * 3 + (size_t)dim + (size_t)I * 2));
		float *gate_h = scratch;
		float *up_h	  = gate_h + I;
		float *act_h  = up_h + I;
		float *y_h	  = act_h + I;
		float *gu_h	  = y_h + dim;

		buffer *xb_b	= batch_slot(ctx->bs, ctx->op->in[0]);
		float  *xb_host = xmalloc((size_t)ctx->n_rows * dim * sizeof(float));
		st				= a->buffer_read_f32(a, xb_b, xb_host, ctx->n_rows * dim);
		if (st != OK) {
			free(xb_host);
			return st;
		}

		for (int row = 0; row < ctx->n_rows; row++) {
			const float *xb_row = xb_host + (size_t)row * dim;
			for (int k = 0; k < K; k++) {
				moe_expert_slot *es = &ctx->bs->moe_per_token_slots[(size_t)row * K + k];
				if (es->eid < 0 || !es->gate_w)
					continue;

				if (es->gate_up_fused) {
					matmul_generic_f32(es->gate_w, es->gate_type, xb_row, gu_h, I * 2, dim);
					moe_activate(act_h, gu_h, gu_h + I, I, es->gate_scale, es->up_scale, use_gelu);
				} else {
					matmul_generic_f32(es->gate_w, es->gate_type, xb_row, gate_h, I, dim);
					matmul_generic_f32(es->up_w, es->up_type, xb_row, up_h, I, dim);
					moe_activate(act_h, gate_h, up_h, I, es->gate_scale, es->up_scale, use_gelu);
				}

				matmul_generic_f32(es->down_w, es->down_type, act_h, y_h, dim, I);
				if (es->down_scale != 1.0f) {
					float ds = es->down_scale;
					for (int d = 0; d < dim; d++)
						y_h[d] *= ds;
				}

				float  w	   = ctx->bs->moe_router_w[(size_t)row * K + k];
				float *out_row = ctx->bs->moe_out.p + (size_t)row * dim;
				for (int d = 0; d < dim; d++)
					out_row[d] += w * y_h[d];
			}
		}
		free(xb_host);
	}

finish:
	buffer *xb2 = batch_slot(ctx->bs, ctx->op->out);
	st			= a->buffer_write_f32(a, xb2, ctx->bs->moe_out.p, ctx->n_rows * dim);
	if (st != OK)
		return st;

	for (int i = 0; i < n_union; i++) {
		moe_expert_slot *us = &ctx->bs->moe_union_slots[i];
		if (us->owned && us->heap_buf) {
			free(us->heap_buf);
			us->heap_buf = NULL;
			us->eid		 = -1;
		}
	}
	moe_stream_release_slots(ctx->m, ctx->li, ctx->bs->moe_union_slots, n_union);

	for (int i = 0; i < n_union; i++)
		ctx->bs->moe_union_slots[i].eid = -1;
	for (int i = 0; i < ctx->n_rows * K; i++) {
		moe_expert_slot *es = &ctx->bs->moe_per_token_slots[i];
		es->eid				= -1;
		es->owned			= 0;
		es->heap_buf		= NULL;
		es->gate_w			= NULL;
		es->up_w			= NULL;
		es->down_w			= NULL;
	}
	return st;
}

static status_code moe_shared_batch(exec_ctx *ctx) {

	backend	 *a		 = exec_layer_backend(ctx);
	const int dim	 = ctx->m->dim;
	int		  is_moe = model_layer_is_moe(ctx->m, ctx->li);
	if (is_moe && ctx->m->moe.n_shared_experts == 0)
		return OK;

	int			   sh_inter = is_moe ? (ctx->m->moe.moe_intermediate * ctx->m->moe.n_shared_experts)
									 : ctx->m->intermediate;
	layer_weights *L		= &ctx->m->layers[ctx->li];
	buffer		  *xb		= batch_slot(ctx->bs, ctx->op->in[0]);

	const buffer *gate_w_buf = is_moe ? &L->shexp_gate_w.buf : &L->gate_w.buf;
	uint32_t	  gate_wt	 = is_moe ? L->shexp_gate_w.type : L->gate_w.type;
	const buffer *up_w_buf	 = is_moe ? &L->shexp_up_w.buf : &L->up_w.buf;
	uint32_t	  up_wt		 = is_moe ? L->shexp_up_w.type : L->up_w.type;
	const buffer *down_w_buf = is_moe ? &L->shexp_down_w.buf : &L->down_w.buf;
	uint32_t	  down_wt	 = is_moe ? L->shexp_down_w.type : L->down_w.type;

	if (!is_moe && L->gate_up_fused) {
		bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE_UP, (size_t)ctx->n_rows * 2 * sh_inter);

		status_code st = a->matmul_batch(a, &L->gate_up_w.buf, L->gate_up_w.type, xb,
										 &ctx->bs->pair[RECIPE_SLOT_FFN_GATE_UP].b, 2 * sh_inter,
										 dim, ctx->n_rows);
		if (st != OK)
			return st;

		int	   use_gelu = ctx->m->arch_info->uses_gelu_activation;
		int	   n_rows	= ctx->n_rows;
		float *gu_host	= xmalloc((size_t)n_rows * 2 * sh_inter * sizeof(float));
		float *act_host = xmalloc((size_t)n_rows * sh_inter * sizeof(float));
		st				= a->buffer_read_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE_UP].b, gu_host,
											 n_rows * 2 * sh_inter);
		if (st != OK) {
			free(gu_host);
			free(act_host);
			return st;
		}
		for (int row = 0; row < n_rows; row++) {
			const float *gu = gu_host + (size_t)row * 2 * sh_inter;
			float		*o	= act_host + (size_t)row * sh_inter;
			moe_activate(o, gu, gu + sh_inter, sh_inter, 1.0f, 1.0f, use_gelu);
		}
		float_buf_ensure(&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].fb, (size_t)n_rows * sh_inter);
		st = a->buffer_write_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, act_host,
								 n_rows * sh_inter);
		free(gu_host);
		free(act_host);
		if (st != OK)
			return st;
	} else {
		bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE, (size_t)ctx->n_rows * sh_inter);
		bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_UP, (size_t)ctx->n_rows * sh_inter);

		status_code st;
		if (a->matmul_multi_batch) {
			const buffer *ws[2]	   = {gate_w_buf, up_w_buf};
			uint32_t	  wts[2]   = {gate_wt, up_wt};
			buffer		 *ys[2]	   = {&ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b,
									  &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b};
			int			  n_out[2] = {sh_inter, sh_inter};
			st = a->matmul_multi_batch(a, ws, wts, xb, ys, n_out, dim, 2, ctx->n_rows);
		} else {
			st = a->matmul_batch(a, gate_w_buf, gate_wt, xb, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b,
								 sh_inter, dim, ctx->n_rows);
			if (st == OK)
				st = a->matmul_batch(a, up_w_buf, up_wt, xb, &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b,
									 sh_inter, dim, ctx->n_rows);
		}
		if (st != OK)
			return st;

		int	   use_gelu = ctx->m->arch_info->uses_gelu_activation;
		int	   n_rows2	= ctx->n_rows;
		float *g_host	= xmalloc((size_t)n_rows2 * sh_inter * sizeof(float));
		float *u_host	= xmalloc((size_t)n_rows2 * sh_inter * sizeof(float));
		float *act_host = xmalloc((size_t)n_rows2 * sh_inter * sizeof(float));
		st				= a->buffer_read_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b, g_host,
											 n_rows2 * sh_inter);
		if (st == OK)
			st = a->buffer_read_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b, u_host,
									n_rows2 * sh_inter);
		if (st != OK) {
			free(g_host);
			free(u_host);
			free(act_host);
			return st;
		}
		for (int row = 0; row < n_rows2; row++) {
			const float *g = g_host + (size_t)row * sh_inter;
			const float *u = u_host + (size_t)row * sh_inter;
			float		*o = act_host + (size_t)row * sh_inter;
			moe_activate(o, g, u, sh_inter, 1.0f, 1.0f, use_gelu);
		}
		st = a->buffer_write_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, act_host,
								 n_rows2 * sh_inter);
		free(g_host);
		free(u_host);
		free(act_host);
		if (st != OK)
			return st;
	}

	ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b.size = (size_t)ctx->n_rows * sh_inter * sizeof(float);

	status_code st =
		a->matmul_batch(a, down_w_buf, down_wt, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b,
						&ctx->bs->pair[RECIPE_SLOT_RESID_TMP].b, dim, sh_inter, ctx->n_rows);
	if (st != OK)
		return st;

	if (a->copy_buffer) {
		st = a->copy_buffer(a, &ctx->bs->pair[RECIPE_SLOT_RESID_TMP].b,
							&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, ctx->n_rows * dim);
		if (st != OK)
			return st;
	} else {
		batch_sync(ctx);
		memcpy(batch_buf_ptr(&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b),
			   batch_buf_ptr(&ctx->bs->pair[RECIPE_SLOT_RESID_TMP].b),
			   (size_t)ctx->n_rows * dim * sizeof(float));
	}
	ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b.size = (size_t)ctx->n_rows * dim * sizeof(float);
	return OK;
}

static status_code matmul_multi_batch_body(exec_ctx *ctx) {

	backend				  *a  = exec_layer_backend(ctx);
	const layer_weights	  *lw = &ctx->m->layers[ctx->li];
	const layer_ctx_entry *lc = &ctx->m->recipe->layer_ctx[ctx->li];
	int					   k  = ctx->op->u.matmul_multi.k;
	if (ctx->op->w_idx == WIDX_WQ) {
		const int *n_out = ctx->op->u.matmul_multi.n_out;
		if (!lc->has_kv) {
			return a->matmul_batch(a, &lw->wq.buf, lw->wq.type, batch_slot(ctx->bs, ctx->op->in[0]),
								   batch_slot(ctx->bs, ctx->op->out), n_out[0], k, ctx->n_rows);
		}
		int n_mats = lc->has_own_v ? 3 : 2;
		if (a->matmul_multi_batch) {
			const buffer *ws[3]	 = {&lw->wq.buf, &lw->wk.buf, &lw->wv.buf};
			uint32_t	  wts[3] = {lw->wq.type, lw->wk.type, lw->wv.type};
			buffer		 *ys[3]	 = {batch_slot(ctx->bs, ctx->op->out),
									batch_slot(ctx->bs, ctx->op->out + 1),
									batch_slot(ctx->bs, ctx->op->out + 2)};
			status_code st = a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]),
												   ys, n_out, k, n_mats, ctx->n_rows);
			if (st == OK && !lc->has_own_v)
				st = batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->n_rows);
			return st;
		}
		status_code st =
			a->matmul_batch(a, &lw->wq.buf, lw->wq.type, batch_slot(ctx->bs, ctx->op->in[0]),
							batch_slot(ctx->bs, ctx->op->out), n_out[0], k, ctx->n_rows);
		if (st == OK)
			st = a->matmul_batch(a, &lw->wk.buf, lw->wk.type, batch_slot(ctx->bs, ctx->op->in[0]),
								 batch_slot(ctx->bs, ctx->op->out + 1), n_out[1], k, ctx->n_rows);
		if (st == OK && lc->has_own_v)
			st = a->matmul_batch(a, &lw->wv.buf, lw->wv.type, batch_slot(ctx->bs, ctx->op->in[0]),
								 batch_slot(ctx->bs, ctx->op->out + 2), n_out[2], k, ctx->n_rows);
		if (st == OK && !lc->has_own_v)
			st = batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->n_rows);
		return st;
	}
	if (ctx->op->w_idx == WIDX_WK) {
		int kv_out = lc->kv_row_stride;
		if (!lc->has_own_v) {
			status_code st =
				a->matmul_batch(a, &lw->wk.buf, lw->wk.type, batch_slot(ctx->bs, ctx->op->in[0]),
								&ctx->bs->pair[RECIPE_SLOT_K].b, kv_out, k, ctx->n_rows);
			if (st != OK)
				return st;
			return batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->n_rows);
		}
		if (a->matmul_multi_batch) {
			const buffer *ws[2]	 = {&lw->wk.buf, &lw->wv.buf};
			uint32_t	  wts[2] = {lw->wk.type, lw->wv.type};
			buffer *ys[2]	 = {&ctx->bs->pair[RECIPE_SLOT_K].b, &ctx->bs->pair[RECIPE_SLOT_V].b};
			int		n_out[2] = {kv_out, kv_out};
			return a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]), ys, n_out,
										 k, 2, ctx->n_rows);
		}
		status_code st =
			a->matmul_batch(a, &lw->wk.buf, lw->wk.type, batch_slot(ctx->bs, ctx->op->in[0]),
							&ctx->bs->pair[RECIPE_SLOT_K].b, kv_out, k, ctx->n_rows);
		if (st != OK)
			return st;
		return a->matmul_batch(a, &lw->wv.buf, lw->wv.type, batch_slot(ctx->bs, ctx->op->in[0]),
							   &ctx->bs->pair[RECIPE_SLOT_V].b, kv_out, k, ctx->n_rows);
	}
	if (ctx->op->w_idx == WIDX_GATE) {
		const int *n_out = ctx->op->u.matmul_multi.n_out;
		if (a->matmul_multi_batch) {
			const buffer *ws[2]	 = {&lw->gate_w.buf, &lw->up_w.buf};
			uint32_t	  wts[2] = {lw->gate_w.type, lw->up_w.type};
			buffer		 *ys[2]	 = {batch_slot(ctx->bs, ctx->op->out),
									batch_slot(ctx->bs, ctx->op->out + 1)};
			return a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]), ys, n_out,
										 k, 2, ctx->n_rows);
		}
		status_code st = a->matmul_batch(
			a, &lw->gate_w.buf, lw->gate_w.type, batch_slot(ctx->bs, ctx->op->in[0]),
			batch_slot(ctx->bs, ctx->op->out), n_out[0], k, ctx->n_rows);
		if (st == OK)
			st = a->matmul_batch(a, &lw->up_w.buf, lw->up_w.type,
								 batch_slot(ctx->bs, ctx->op->in[0]),
								 batch_slot(ctx->bs, ctx->op->out + 1), n_out[1], k, ctx->n_rows);
		return st;
	}
	return ERR_INVALID_ARG;
}

static status_code ple_build_batch(exec_ctx *ctx) {

	if (!ctx->m->has_per_layer_embeddings)
		return OK;
	struct model *m				   = ctx->m;
	backend		 *a				   = exec_layer_backend(ctx);
	const int	  n_embd_per_layer = m->layer_dims.n_embd_per_layer;
	const int	  n_layers		   = m->n_layers;
	const int	  total_ple		   = n_embd_per_layer * n_layers;
	const int	  dim			   = m->dim;
	const int	  n_rows		   = ctx->n_rows;
	const float	  eps			   = m->norm_eps;
	const float	  n_embd_sqrt	   = sqrtf((float)n_embd_per_layer);
	const float	  inv_sqrt_ple	   = m->dim_sqrt > 0 ? 1.0f / m->dim_sqrt : 0.0f;
	const float	  combine_scale	   = 0.70710678118654752f;
	status_code	  st;

	float_buf_ensure(&ctx->bs->ple_buf, (size_t)n_rows * total_ple);
	st = buffer_ensure_scratch(a, &ctx->bs->ple_all, (size_t)n_rows * total_ple * sizeof(float));
	if (st != OK)
		return st;
	float_buf_ensure(&ctx->bs->ple_proj, (size_t)n_rows * total_ple);

	float *ple = ctx->bs->ple_buf.p;
	for (int row = 0; row < n_rows; row++) {
		int			   token	  = ctx->bs->tokens ? ctx->bs->tokens[row] : 0;
		size_t		   row_stride = ggml_row_size(m->layer_dims.per_layer_tok_embd.type, total_ple);
		const uint8_t *embd		  = (const uint8_t *)m->layer_dims.per_layer_tok_embd.host_ptr +
									((size_t)token * row_stride);
		float		  *ple_row	  = ple + (size_t)row * total_ple;
		dequant_row_dispatch(m->layer_dims.per_layer_tok_embd.type, embd, total_ple, ple_row);
		for (int i = 0; i < total_ple; i++)
			ple_row[i] *= n_embd_sqrt;
	}

	int dev_path_ok = (a->matmul_batch && a->scale_inplace && a->rmsnorm_batch && a->ple_combine &&
					   m->layer_dims.per_layer_model_proj.buf.handle);

	if (dev_path_ok) {
		status_code st = a->buffer_write_f32(a, &ctx->bs->ple_all, ple, n_rows * total_ple);
		if (st != OK)
			return st;

		buffer *xb = batch_slot(ctx->bs, RECIPE_SLOT_X);

		buffer proj_buf = {0};
		st = a->buffer_alloc_scratch(a, (size_t)n_rows * total_ple * sizeof(float), &proj_buf);
		if (st != OK)
			return st;

		st = a->matmul_batch(a, &m->layer_dims.per_layer_model_proj.buf,
							 m->layer_dims.per_layer_model_proj.type, xb, &proj_buf, total_ple, dim,
							 n_rows);
		if (st != OK)
			goto ple_build_cleanup;

		st = a->scale_inplace(a, &proj_buf, inv_sqrt_ple, n_rows * total_ple);
		if (st != OK)
			goto ple_build_cleanup;

		st = ple_ensure_norm_w(a, ctx->s, m, n_embd_per_layer);
		if (st != OK)
			goto ple_build_cleanup;

		for (int l = 0; l < n_layers; l++) {
			for (int row = 0; row < n_rows; row++) {
				buffer x_slice = proj_buf;
				x_slice.offset =
					((size_t)row * total_ple + (size_t)l * n_embd_per_layer) * sizeof(float);
				st = a->rmsnorm(a, &x_slice, &ctx->s->ple_proj_norm_w, &x_slice, n_embd_per_layer,
								eps);
				if (st != OK)
					goto ple_build_cleanup;
			}
		}

		st = a->ple_combine(a, &ctx->bs->ple_all, &proj_buf, n_rows * total_ple, combine_scale);
	ple_build_cleanup:
		if (a->synchronize)
			a->synchronize(a);
		a->buffer_free(a, &proj_buf);
		return st;
	}

	if (a->synchronize)
		a->synchronize(a);

	float *ple_proj	 = float_buf_ensure(&ctx->s->ple_proj_host, (size_t)n_rows * total_ple);
	float *inpL_host = float_buf_ensure(&ctx->s->inpL_host, (size_t)n_rows * dim);

	buffer	*xb		 = batch_slot(ctx->bs, RECIPE_SLOT_X);
	backend *x_owner = xb->owner;
	st				 = x_owner->buffer_read_f32(x_owner, xb, inpL_host, n_rows * dim);
	if (st != OK)
		return st;

	backend *host = backend_host();
	for (int row = 0; row < n_rows; row++)
		matmul_generic_f32(m->layer_dims.per_layer_model_proj.host_ptr,
						   m->layer_dims.per_layer_model_proj.type, inpL_host + (size_t)row * dim,
						   ple_proj + (size_t)row * total_ple, total_ple, dim);

	{
		buffer proj_view = {0};
		proj_view.handle = ple_proj;
		proj_view.owner	 = host;
		host->scale_inplace(host, &proj_view, inv_sqrt_ple, n_rows * total_ple);
	}

	{
		buffer w_view = {0};
		w_view.handle = (void *)m->layer_dims.per_layer_proj_norm_w.host_ptr;
		w_view.owner  = host;
		for (int row = 0; row < n_rows; row++) {
			for (int l = 0; l < n_layers; l++) {
				buffer row_view = {0};
				row_view.handle =
					ple_proj + ((size_t)row * total_ple) + ((size_t)l * n_embd_per_layer);
				row_view.owner = host;
				host->rmsnorm(host, &row_view, &w_view, &row_view, n_embd_per_layer, eps);
			}
		}
	}

	{
		buffer ple_view	 = {0};
		ple_view.handle	 = ple;
		ple_view.owner	 = host;
		buffer proj_view = {0};
		proj_view.handle = ple_proj;
		proj_view.owner	 = host;
		host->ple_combine(host, &ple_view, &proj_view, n_rows * total_ple, combine_scale);
	}

	if (a->copy_buffer) {
		st = a->buffer_write_f32(a, &ctx->bs->ple_all, ple, n_rows * total_ple);
		if (st != OK)
			return st;
	}

	return OK;
}

static status_code ple_proj_inject_batch(exec_ctx *ctx) {

	if (ctx->li < 0)
		return ERR_INVALID_ARG;
	if (!ctx->m->has_per_layer_embeddings)
		return OK;
	struct model			   *m				 = ctx->m;
	const struct layer_weights *L				 = &m->layers[ctx->li];
	backend					   *a				 = exec_layer_backend(ctx);
	const int					n_embd_per_layer = m->layer_dims.n_embd_per_layer;
	const int					total_ple		 = n_embd_per_layer * m->n_layers;
	const int					inj_dim			 = m->dim;
	const int					n_rows			 = ctx->n_rows;
	const float					eps				 = m->norm_eps;

	float_buf_ensure(&ctx->bs->ple_inp, (size_t)n_rows * n_embd_per_layer);
	float_buf_ensure(&ctx->bs->ple_slice, (size_t)n_rows * n_embd_per_layer);

	status_code st;
	buffer		ple_inp_b = {0};
	st = a->buffer_alloc_scratch(a, (size_t)n_rows * n_embd_per_layer * sizeof(float), &ple_inp_b);
	if (st != OK)
		return st;

	buffer ple_slice_b = {0};
	st =
		a->buffer_alloc_scratch(a, (size_t)n_rows * n_embd_per_layer * sizeof(float), &ple_slice_b);
	if (st != OK) {
		a->buffer_free(a, &ple_inp_b);
		return st;
	}

	if (ctx->bs->ple_all.handle) {
		for (int row = 0; row < n_rows; row++) {
			buffer ple_src = ctx->bs->ple_all;
			ple_src.offset +=
				((size_t)row * total_ple + (size_t)ctx->li * n_embd_per_layer) * sizeof(float);
			buffer dst_row = batch_row_view(&ple_slice_b, row, n_embd_per_layer);
			st = compute_copy_buffer_cross(ctx->s, &ple_src, &dst_row, n_embd_per_layer);
			if (st != OK) {
				ERROR("ple_proj_inject_batch: copy_buffer_cross(ple_src->slice) failed (st=%d)",
					  (int)st);
				goto ple_cleanup;
			}
		}
	} else {
		for (int row = 0; row < n_rows; row++) {
			float *ple_row =
				ctx->bs->ple_buf.p + (size_t)row * total_ple + (size_t)ctx->li * n_embd_per_layer;
			buffer dst_row = batch_row_view(&ple_slice_b, row, n_embd_per_layer);
			st			   = a->buffer_write_f32(a, &dst_row, ple_row, n_embd_per_layer);
			if (st != OK)
				goto ple_cleanup;
		}
	}

	buffer *xb2b = batch_slot(ctx->bs, RECIPE_SLOT_XB2);
	st			 = a->matmul_batch(a, &L->ple_inp_gate_w.buf, GGML_TYPE_F32, xb2b, &ple_inp_b,
								   n_embd_per_layer, inj_dim, n_rows);
	if (st != OK) {
		ERROR("ple_proj_inject_batch: matmul_batch(ple_inp_gate) failed (st=%d)", (int)st);
		goto ple_cleanup;
	}

	st = a->ffn_activate_batch(a, &ple_inp_b, &ple_slice_b, &ple_inp_b, n_embd_per_layer,
							   ACTIVATION_GELU, n_rows);
	if (st != OK) {
		ERROR("ple_proj_inject_batch: ffn_activate_batch failed (st=%d)", (int)st);
		goto ple_cleanup;
	}

	buffer *aob = batch_slot(ctx->bs, RECIPE_SLOT_ATTN_OUT);
	st			= a->matmul_batch(a, &L->ple_proj_w.buf, GGML_TYPE_F32, &ple_inp_b, aob, inj_dim,
								  n_embd_per_layer, n_rows);
	if (st != OK) {
		ERROR("ple_proj_inject_batch: matmul_batch(ple_proj) failed (st=%d)", (int)st);
		goto ple_cleanup;
	}

	st = a->rmsnorm_batch(a, aob, &L->ple_post_norm_w.buf, aob, inj_dim, eps, n_rows);
	if (st != OK) {
		ERROR("ple_proj_inject_batch: rmsnorm_batch failed (st=%d)", (int)st);
		goto ple_cleanup;
	}

	st = a->add_batch(a, xb2b, aob, inj_dim, n_rows);
	if (st != OK)
		ERROR("ple_proj_inject_batch: add_batch failed (st=%d)", (int)st);

ple_cleanup:
	if (a->synchronize)
		a->synchronize(a);
	a->buffer_free(a, &ple_inp_b);
	a->buffer_free(a, &ple_slice_b);
	return st;
}

static status_code op_moe_router(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return moe_router_batch(ctx);
	const recipe_op		   *op	  = ctx->op;
	struct model		   *m	  = ctx->m;
	struct compute_scratch *s	  = ctx->s;
	int						li	  = ctx->li;
	buffer				   *slots = compute_slots_array(ctx->s);
	if (!model_layer_is_moe(m, li))
		return OK;

	backend		  *a			= model_layer_backend(ctx->m, ctx->li);
	layer_weights *L			= &m->layers[li];
	int			   E			= m->moe.n_experts;
	int			   K			= m->moe.n_experts_used;
	int			   dim			= op->u.matmul.k;
	float		   routed_scale = m->moe.routed_scale;
	int			   norm_topk	= m->moe.norm_topk_prob;
	int			   uses_softmax = m->arch_info->uses_moe_softmax_router;

	buffer *inp = &slots[op->in[0]];

	if (uses_softmax) {
		float *router_input = float_buf_ensure(&s->moe_xb_f, (size_t)dim);
		a->synchronize(a);
		backend	   *ib = inp->owner ? inp->owner : a;
		status_code rs = ib->buffer_read_f32(ib, inp, router_input, dim);
		if (rs != OK)
			return rs;

		moe_router_normalize_input(m, L, dim, router_input);

		return op_moe_router_softmax(a, L, E, K, dim, norm_topk, routed_scale, router_input, s,
									 slots);
	}

	return op_moe_router_direct(a, m, L, E, K, dim, norm_topk, routed_scale, inp, s, slots);
}

static void moe_expert_chunk(int begin, int end, int tid, void *ctx) {
	moe_par_job *j	 = ctx;
	int			 dim = j->dim;

	moe_expert_ctx cx = {
		.a			  = j->a,
		.inter		  = j->inter,
		.dim		  = dim,
		.use_gelu	  = j->use_gelu,
		.q8_gate_ok	  = j->xb_q8_gate_ok,
		.q8_gate_type = j->gate_q8_type,
		.xb_q8_gate	  = j->xb_q8_gate,
		.out		  = (float *)((char *)j->all_outs + ((size_t)tid * j->per_thread_out)),
	};
	cx.scratch = (float *)((char *)j->all_scratch + ((size_t)tid * j->per_thread_scratch));

	for (int k = begin; k < end; k++) {
		int eid = j->expert_ids[k];
		if (eid < 0 || eid >= j->m->moe.n_experts)
			continue;
		status_code local_st = moe_expert_exec(&cx, &j->slot_buf[k], j->xb_f, j->weights[k], tid);
		if (local_st != OK) {
			if (atomic_fetch_add(&j->n_err, 1) == 0)
				atomic_store(&j->first_err, local_st);
		}
	}
}

typedef struct {
	moe_stream_op *op;
	moe_par_job	  *job;
} moe_fused_job;

static void moe_fused_chunk(int begin, int end, int tid, void *ctx) {
	moe_fused_job *fj = (moe_fused_job *)ctx;
	for (int i = begin; i < end; i++) {
		int k = moe_stream_op_compute_k(fj->op, i);
		if (k < 0)
			moe_stream_op_fill_run(fj->op, i, tid);
		else
			moe_expert_chunk(k, k + 1, tid, fj->job);
	}
}

static status_code moe_experts_run_parallel(model *m, int li, int K, int dim, backend *a,
											moe_expert_slot *slot_buf, const int *expert_ids,
											const float *weights, buffer *xb, compute_scratch *s,
											int I, int any_fused, int use_gelu, int xb_q8_gate_ok,
											uint32_t gate_q8_type, const buffer *xb_q8_gate,
											size_t scratch_need, tpool *pool, int interleave,
											moe_stream_op *moe_op, float *outf) {
	int n_threads = tpool_n_threads(pool);
	if (n_threads < 1)
		n_threads = 1;
	if (!(interleave && moe_op) && n_threads > K)
		n_threads = K;

	size_t per_thread_scratch = scratch_need;
	size_t total_scratch	  = (size_t)n_threads * per_thread_scratch;
	float *all_scratch = float_buf_ensure(&s->moe_all_scratch, total_scratch / sizeof(float));

	size_t per_thread_out = (size_t)dim * sizeof(float);
	size_t total_out	  = (size_t)n_threads * per_thread_out;
	float *all_outs		  = float_buf_ensure(&s->moe_all_outs, total_out / sizeof(float));
	for (int t = 0; t < n_threads; t++) {
		float *o = (float *)((char *)all_outs + ((size_t)t * per_thread_out));
		for (int d = 0; d < dim; d++)
			o[d] = 0.0f;
	}
	for (int d = 0; d < dim; d++)
		outf[d] = 0.0f;

	float *xb_f = float_buf_ensure(&s->moe_xb_f, (size_t)dim);
	{
		backend *xb_owner = xb->owner ? xb->owner : a;
		if (xb_owner->synchronize)
			xb_owner->synchronize(xb_owner);
		status_code rs = xb_owner->buffer_read_f32(xb_owner, xb, xb_f, dim);
		if (rs != OK)
			return rs;
	}

	moe_par_job job = {.m				   = m,
					   .li				   = li,
					   .top_k			   = K,
					   .inter			   = I,
					   .dim				   = dim,
					   .any_fused		   = any_fused,
					   .use_gelu		   = use_gelu,
					   .xb_q8_gate_ok	   = xb_q8_gate_ok,
					   .gate_q8_type	   = gate_q8_type,
					   .slot_buf		   = slot_buf,
					   .expert_ids		   = expert_ids,
					   .weights			   = weights,
					   .xb_f			   = xb_f,
					   .xb_q8_gate		   = xb_q8_gate,
					   .a				   = a,
					   .all_scratch		   = all_scratch,
					   .per_thread_scratch = per_thread_scratch,
					   .all_outs		   = all_outs,
					   .per_thread_out	   = per_thread_out};
	atomic_store(&job.first_err, OK);
	atomic_store(&job.n_err, 0);

	if (interleave && moe_op) {
		moe_stream_op_set_compute_hook(moe_op, moe_expert_chunk, &job);
		moe_fused_job fj = {.op = moe_op, .job = &job};
		tpool_parallel_for(pool, moe_stream_op_n_items(moe_op), 1, moe_fused_chunk, &fj);
	} else {
		tpool_parallel_for(pool, K, 1, moe_expert_chunk, &job);
	}

	for (int t = 0; t < n_threads; t++) {
		float *o = (float *)((char *)all_outs + ((size_t)t * per_thread_out));
		for (int d = 0; d < dim; d++)
			outf[d] += o[d];
	}

	if (atomic_load(&job.n_err) > 0) {
		status_code st = atomic_load(&job.first_err);
		if (st == OK)
			st = ERR_INTERNAL;
		return st;
	}
	return OK;
}

static status_code moe_experts_run_sequential(model *m, int li, int K, int dim, backend *a,
											  moe_expert_slot *slot_buf, const int *expert_ids,
											  const float *weights, buffer *xb, compute_scratch *s,
											  int I, int use_gelu, int xb_q8_gate_ok,
											  uint32_t gate_q8_type, const buffer *xb_q8_gate,
											  size_t scratch_need, float *outf) {
	(void)li;
	float *scratch = float_buf_ensure(&s->moe_scratch, scratch_need / sizeof(float));

	float	*xb_f	  = float_buf_ensure(&s->moe_xb_f, dim);
	backend *xb_owner = xb->owner ? xb->owner : a;
	if (xb_owner->synchronize)
		xb_owner->synchronize(xb_owner);
	status_code st = xb_owner->buffer_read_f32(xb_owner, xb, xb_f, dim);
	if (st != OK)
		return st;

	for (int d = 0; d < dim; d++)
		outf[d] = 0.0f;

	moe_expert_ctx cx = {
		.a			  = a,
		.inter		  = I,
		.dim		  = dim,
		.use_gelu	  = use_gelu,
		.q8_gate_ok	  = xb_q8_gate_ok,
		.q8_gate_type = gate_q8_type,
		.xb_q8_gate	  = xb_q8_gate,
		.scratch	  = scratch,
		.out		  = outf,
	};

	for (int k = 0; k < K; k++) {
		int eid = expert_ids[k];
		if (eid < 0 || eid >= m->moe.n_experts)
			continue;
		st = moe_expert_exec(&cx, &slot_buf[k], xb_f, weights[k], 0);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code op_moe_experts(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return moe_experts_batch(ctx);
	struct model		   *m		= ctx->m;
	struct compute_scratch *s		= ctx->s;
	int						li		= ctx->li;
	buffer				   *slots	= compute_slots_array(ctx->s);
	buffer				   *out_buf = &slots[RECIPE_SLOT_XB2];
	int						dim		= m->dim;

	int	   out_is_host = out_buf->host_ptr != NULL || !out_buf->owner ||
						 backend_has_cap(out_buf->owner, BCAP_IS_HOST);
	float  local_outf[MOE_MAX_DIM_STACK];
	float *heap_outf = NULL;
	float *outf;
	if (out_is_host) {
		outf = (float *)(out_buf->host_ptr ? out_buf->host_ptr : out_buf->handle);
		if (!outf) {
			ERROR("op_moe_experts: output slot XB2 has no host-accessible pointer (backend=%s)",
				  m->backend && m->backend->name ? m->backend->name : "?");
			return ERR_UNSUPPORTED;
		}
	} else if (dim <= MOE_MAX_DIM_STACK) {
		outf = local_outf;
	} else {
		heap_outf = xmalloc((size_t)dim * sizeof(float));
		outf	  = heap_outf;
	}
	if (!model_layer_is_moe(m, li)) {
		for (int d = 0; d < dim; d++)
			outf[d] = 0.0f;
		if (!out_is_host) {
			status_code wst = out_buf->owner->buffer_write_f32(out_buf->owner, out_buf, outf, dim);
			free(heap_outf);
			return wst;
		}
		return OK;
	}

	backend *a = model_layer_backend(ctx->m, ctx->li);
	int		 K = m->moe.n_experts_used;
	int		 I = m->moe.moe_intermediate;

	const int	*expert_ids = (const int *)slots[RECIPE_SLOT_ROUTER_IDS].handle;
	const float *weights	= (const float *)slots[RECIPE_SLOT_ROUTER_W].handle;
	if (!expert_ids || !weights) {
		ERROR("op_moe_experts: router slot handles are NULL (layer=%d)", li);
		for (int d = 0; d < dim; d++)
			outf[d] = 0.0f;
		if (!out_is_host) {
			out_buf->owner->buffer_write_f32(out_buf->owner, out_buf, outf, dim);
			free(heap_outf);
		}
		return ERR_INVALID_ARG;
	}

	moe_expert_slot *slot_buf = s->moe_slot_buf;
	if (K > MOE_MAX_TOPK) {
		ERROR("op_moe_experts: n_experts_used=%d exceeds supported maximum %d "
			  "(router scratch capacity); refusing silent truncation",
			  K, MOE_MAX_TOPK);
		if (!out_is_host) {
			out_buf->owner->buffer_write_f32(out_buf->owner, out_buf, outf, dim);
			free(heap_outf);
		}
		return ERR_FORMAT;
	}

	buffer *xb_dev = &slots[RECIPE_SLOT_XB];
	if (!exec_is_batch(ctx) && backend_has_cap(a, BCAP_MOE_EXPERT_RESIDENT) && a->moe_expert_ffn &&
		m->moe.experts_resident && xb_dev->owner == a && out_buf->owner == a) {
		status_code gst	   = moe_stream_resolve(m, li, expert_ids, K, slot_buf);
		int			dev_ok = (gst == OK);
		for (int k = 0; dev_ok && k < K; k++)
			if (!slot_buf[k].dev_ready)
				dev_ok = 0;
		if (dev_ok) {
			int use_gelu = m->arch_info->uses_gelu_activation;
			gst			 = a->scale_inplace(a, out_buf, 0.0f, dim);
			for (int k = 0; gst == OK && k < K; k++) {
				if (slot_buf[k].eid < 0 || !slot_buf[k].gate_w)
					continue;
				moe_resident_expert ge = {
					.gate_w		   = &slot_buf[k].dev_gate,
					.up_w		   = &slot_buf[k].dev_up,
					.down_w		   = &slot_buf[k].dev_down,
					.gate_type	   = slot_buf[k].gate_type,
					.up_type	   = slot_buf[k].up_type,
					.down_type	   = slot_buf[k].down_type,
					.gate_up_fused = slot_buf[k].gate_up_fused,
					.use_gelu	   = use_gelu,
					.gate_scale	   = slot_buf[k].gate_scale,
					.up_scale	   = slot_buf[k].up_scale,
					.down_scale	   = slot_buf[k].down_scale,
					.weight		   = weights[k],
				};
				gst = a->moe_expert_ffn(a, xb_dev, out_buf, &ge, dim, I);
			}
			if (gst == OK) {
				moe_stream_release_slots(m, li, slot_buf, K);
				return OK;
			}
			WARN("moe resident expert ffn failed (st=%d) -- falling back to cpu", (int)gst);
			moe_stream_release_slots(m, li, slot_buf, K);
		}
	}

	monitor_emit_moe_experts(g_monitor, li, -1, expert_ids, weights, K);

	tpool *pool		  = (a->get_pool && a->matmul_thread_local) ? a->get_pool(a) : NULL;
	int	   par		  = (pool && a->matmul_thread_local && K >= 2);
	int	   interleave = par;

	profile_scope  io_ps  = profile_begin(&s->prof, STAGE_MOE_IO_WAIT);
	moe_stream_op *moe_op = NULL;
	if (interleave) {
		moe_op = moe_stream_resolve_prep(m, li, expert_ids, K, slot_buf);
	} else {
		status_code st_w = moe_stream_resolve(m, li, expert_ids, K, slot_buf);
		if (st_w != OK) {
			profile_end(&s->prof, &io_ps);
			ERROR("op_moe_experts: moe_stream_resolve failed (layer=%d, st=%d)", li, st_w);
			for (int d = 0; d < dim; d++)
				outf[d] = 0.0f;
			if (!out_is_host) {
				out_buf->owner->buffer_write_f32(out_buf->owner, out_buf, outf, dim);
				free(heap_outf);
			}
			return st_w;
		}
	}
	profile_end(&s->prof, &io_ps);

	profile_scope compute_ps = profile_begin(&s->prof, STAGE_MATMUL_FFN);

	int use_gelu = m->arch_info->uses_gelu_activation;

	size_t scratch_need = ((size_t)I * sizeof(float) * 3) + ((size_t)dim * sizeof(float));
	int	   any_fused	= m->layers[li].any_fused_experts;
	if (any_fused)
		scratch_need += (size_t)I * 2 * sizeof(float);

	buffer	   *xb = &slots[RECIPE_SLOT_XB];
	status_code st = OK;

	int	   has_qonly = backend_has_cap(a, BCAP_MATMUL_QONLY) && a->matmul_qonly && a->prequantize_x;
	buffer xb_q8_gate	  = {0};
	int	   xb_q8_gate_ok  = 0;
	uint32_t gate_q8_type = has_qonly ? m->layers[li].gate_q8_type : 0;
	if (has_qonly) {
		if (gate_q8_type) {
			st = a->prequantize_x(a, xb, dim, gate_q8_type, &xb_q8_gate);
			if (st == OK)
				xb_q8_gate_ok = 1;
			else if (st != ERR_UNSUPPORTED)
				goto cleanup;
		}
	}

	if (par) {
		st = moe_experts_run_parallel(m, li, K, dim, backend_host(), slot_buf, expert_ids, weights,
									  xb, s, I, any_fused, use_gelu, xb_q8_gate_ok, gate_q8_type,
									  &xb_q8_gate, scratch_need, pool, interleave, moe_op, outf);
	} else {
		st = moe_experts_run_sequential(m, li, K, dim, backend_host(), slot_buf, expert_ids,
										weights, xb, s, I, use_gelu, xb_q8_gate_ok, gate_q8_type,
										&xb_q8_gate, scratch_need, outf);
	}

cleanup:
	profile_end(&s->prof, &compute_ps);

	if (moe_op) {
		status_code fst = moe_stream_op_finish(moe_op);
		if (fst != OK) {
			ERROR("op_moe_experts: moe_stream_op_finish failed (layer=%d, st=%d)", li, fst);
			if (st == OK)
				st = ERR_INTERNAL;
		}
		moe_stream_op_free(moe_op);
		moe_op = NULL;
	}

	for (int k = 0; k < K; k++) {
		if (slot_buf[k].owned && slot_buf[k].heap_buf) {
			free(slot_buf[k].heap_buf);
			slot_buf[k].heap_buf = NULL;
			slot_buf[k].owned	 = 0;
		}
	}
	moe_stream_release_slots(m, li, slot_buf, K);

	if (!out_is_host) {
		status_code wst = out_buf->owner->buffer_write_f32(out_buf->owner, out_buf, outf, dim);
		free(heap_outf);
		if (st == OK)
			st = wst;
	}

	return st;
}

static status_code op_moe_shared(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return moe_shared_batch(ctx);
	struct model		   *m	   = ctx->m;
	struct compute_scratch *s	   = ctx->s;
	int						li	   = ctx->li;
	buffer				   *slots  = compute_slots_array(ctx->s);
	backend				   *a	   = model_layer_backend(ctx->m, ctx->li);
	layer_weights		   *L	   = &m->layers[li];
	int						dim	   = m->dim;
	int						is_moe = model_layer_is_moe(m, li);

	int sh_inter = is_moe ? (m->moe.moe_intermediate * m->moe.n_shared_experts) : m->intermediate;
	if (is_moe && m->moe.n_shared_experts == 0)
		return OK;

	profile		 *prof = &s->prof;
	profile_scope ps   = profile_begin(prof, STAGE_MOE_SHARED);

	buffer *xb		 = &slots[RECIPE_SLOT_XB];
	buffer *gate_buf = &slots[RECIPE_SLOT_FFN_GATE];
	buffer *up_buf	 = &slots[RECIPE_SLOT_FFN_UP];
	buffer *act_buf	 = &slots[RECIPE_SLOT_FFN_ACT];

	uint32_t gate_type = is_moe ? L->shexp_gate_w.type : L->gate_w.type;
	uint32_t up_type   = is_moe ? L->shexp_up_w.type : L->up_w.type;
	int has_qonly	 = backend_has_cap(a, BCAP_MATMUL_QONLY) && a->matmul_qonly && a->prequantize_x;
	uint32_t gate_q8 = has_qonly ? wtype_to_q8type(gate_type) : 0;
	uint32_t up_q8	 = has_qonly ? wtype_to_q8type(up_type) : 0;
	int		 share_q8 = gate_q8 && gate_q8 == up_q8;

	buffer		xb_q8 = {0};
	status_code st;

	if (is_moe) {
		if (share_q8) {
			st = a->prequantize_x(a, xb, dim, gate_q8, &xb_q8);
			if (st != OK)
				share_q8 = 0;
		}
		if (share_q8) {
			st = a->matmul_qonly(a, &L->shexp_gate_w.buf, gate_type, &xb_q8, gate_q8, gate_buf,
								 sh_inter, dim, 1);
		} else {
			st = a->matmul(a, &L->shexp_gate_w.buf, L->shexp_gate_w.type, xb, gate_buf, sh_inter,
						   dim);
		}
		if (st != OK)
			goto done;
		buffer up_w_buf =
			buffer_slice(&L->shexp_up_w.buf, 0, L->shexp_up_w.buf.size - L->shexp_up_w.buf.offset);
		if (share_q8) {
			st = a->matmul_qonly(a, &up_w_buf, up_type, &xb_q8, up_q8, up_buf, sh_inter, dim, 1);
		} else {
			st = a->matmul(a, &up_w_buf, L->shexp_up_w.type, xb, up_buf, sh_inter, dim);
		}
		if (st != OK)
			goto done;
	} else {
		if (L->gate_up_fused) {
			st = a->matmul(a, &L->gate_up_w.buf, L->gate_up_w.type, xb,
						   &slots[RECIPE_SLOT_FFN_GATE_UP], 2 * sh_inter, dim);
			if (st != OK)
				goto done;
			*gate_buf =
				buffer_slice(&slots[RECIPE_SLOT_FFN_GATE_UP], 0, (size_t)sh_inter * sizeof(float));
			*up_buf =
				buffer_slice(&slots[RECIPE_SLOT_FFN_GATE_UP], (size_t)sh_inter * sizeof(float),
							 (size_t)sh_inter * sizeof(float));
		} else {
			if (share_q8) {
				st = a->prequantize_x(a, xb, dim, gate_q8, &xb_q8);
				if (st != OK)
					share_q8 = 0;
			}
			if (share_q8) {
				st = a->matmul_qonly(a, &L->gate_w.buf, gate_type, &xb_q8, gate_q8, gate_buf,
									 sh_inter, dim, 1);
			} else {
				st = a->matmul(a, &L->gate_w.buf, L->gate_w.type, xb, gate_buf, sh_inter, dim);
			}
			if (st != OK)
				goto done;
			if (share_q8) {
				st = a->matmul_qonly(a, &L->up_w.buf, up_type, &xb_q8, up_q8, up_buf, sh_inter, dim,
									 1);
			} else {
				st = a->matmul(a, &L->up_w.buf, L->up_w.type, xb, up_buf, sh_inter, dim);
			}
			if (st != OK)
				goto done;
		}
	}

	float *g = (float *)((char *)gate_buf->handle + gate_buf->offset);
	float *u = (float *)((char *)up_buf->handle + up_buf->offset);
	float *o = (float *)((char *)act_buf->handle + act_buf->offset);
	moe_activate(o, g, u, sh_inter, 1.0f, 1.0f, m->arch_info->uses_gelu_activation);

	float *yp	   = float_buf_ensure(&s->moe_shared_y, (size_t)dim);
	buffer y_buf   = {0};
	y_buf.handle   = yp;
	y_buf.host_ptr = yp;
	y_buf.owner	   = NULL;
	y_buf.size	   = (size_t)dim * sizeof(float);

	if (is_moe) {
		st = a->matmul(a, &L->shexp_down_w.buf, L->shexp_down_w.type, act_buf, &y_buf, dim,
					   sh_inter);
	} else {
		st = a->matmul(a, &L->down_w.buf, L->down_w.type, act_buf, &y_buf, dim, sh_inter);
	}
	if (st != OK)
		goto done;
	memcpy(act_buf->handle, yp, (size_t)dim * sizeof(float));
	st = OK;

done:
	profile_end(prof, &ps);
	return st;
}

static void moe_batch_expert_chunk(int begin, int end, int tid, void *ctx) {
	moe_batch_job *j	   = ctx;
	float		  *scratch = j->all_scratch + (size_t)tid * j->per_thread_scratch;
	float		  *out	   = j->all_outs + (size_t)tid * j->per_thread_out;

	moe_expert_ctx cx = {
		.a		  = j->a,
		.inter	  = j->inter,
		.dim	  = j->dim,
		.use_gelu = j->use_gelu,
		.scratch  = scratch,
	};

	for (int idx = begin; idx < end; idx++) {
		int row = idx / j->top_k;
		int k	= idx % j->top_k;

		moe_expert_slot *es = &j->slot_buf[(size_t)row * j->top_k + k];
		if (es->eid < 0 || !es->gate_w)
			continue;

		const float *xb_row = j->xb_base + (size_t)row * j->dim;
		float		*w		= &j->bs->moe_router_w[(size_t)row * j->top_k + k];
		cx.out				= out + (size_t)row * j->dim;
		(void)moe_expert_exec(&cx, es, xb_row, *w, tid);
	}
}

static status_code batch_copy_k_to_v(model *m, batch_scratch *bs, int li, int n_tokens) {
	const layer_ctx_entry *lc			 = &m->recipe->layer_ctx[li];
	int					   kv_out		 = lc->kv_row_stride;
	backend				  *a			 = m->backend;
	buffer				  *kb			 = &bs->pair[RECIPE_SLOT_K].b;
	buffer				  *vb			 = &bs->pair[RECIPE_SLOT_V].b;
	int					   kv_row_stride = lc->kv_row_stride;

	if (a->copy_buffer && kb->owner == vb->owner) {
		for (int row = 0; row < n_tokens; row++) {
			buffer		krow = batch_row_view(kb, row, kv_row_stride);
			buffer		vrow = batch_row_view(vb, row, kv_row_stride);
			status_code st	 = a->copy_buffer(a, &krow, &vrow, kv_out);
			if (st != OK)
				return st;
		}
		return OK;
	}
	if (a->synchronize)
		a->synchronize(a);
	for (int row = 0; row < n_tokens; row++) {
		const float *k =
			(const float *)((const char *)kb->host_ptr + kb->offset) + (size_t)row * kv_row_stride;
		float *v = (float *)((char *)vb->host_ptr + vb->offset) + (size_t)row * kv_row_stride;
		memcpy(v, k, (size_t)kv_out * sizeof(float));
	}
	return OK;
}

static buffer mla_scratch_buf_alloc(float *stack, int stack_cap, int n, int *out_heap) {
	float *ptr	 = (n <= stack_cap) ? stack : xmalloc((size_t)n * sizeof(float));
	*out_heap	 = (ptr != stack);
	buffer buf	 = {0};
	buf.handle	 = ptr;
	buf.size	 = (size_t)n * sizeof(float);
	buf.host_ptr = ptr;
	buf.owner	 = NULL;
	return buf;
}

static void mla_scratch_buf_free(buffer *buf, int heap) {
	if (heap)
		free(buf->handle);
}

static status_code op_mla_qkv_proj_fused(exec_ctx *ctx) {
	if (exec_is_batch(ctx)) {

		backend		  *a		 = exec_layer_backend(ctx);
		const int	   dim		 = ctx->m->dim;
		layer_weights *L		 = &ctx->m->layers[ctx->li];
		int			   q_lora	 = ctx->m->mla.q_lora;
		int			   kv_lora	 = ctx->m->mla.kv_lora;
		int			   qk_rope	 = ctx->m->mla.qk_rope;
		int			   q_b_rows	 = ctx->m->n_heads * ctx->m->mla.qk_head;
		int			   kv_a_rows = kv_lora + qk_rope;
		buffer		  *xb		 = batch_slot(ctx->bs, ctx->op->in[0]);

		float_buf_ensure(&ctx->bs->pair[RECIPE_SLOT_FFN_GATE].fb, (size_t)ctx->n_rows * q_lora);
		float_buf_ensure(&ctx->bs->pair[RECIPE_SLOT_FFN_UP].fb, (size_t)ctx->n_rows * kv_a_rows);

		status_code st;
		buffer		q_a_b = {0};
		st = a->buffer_alloc_scratch(a, (size_t)ctx->n_rows * q_lora * sizeof(float), &q_a_b);
		if (st != OK)
			return st;
		buffer kv_a_b = {0};
		st = a->buffer_alloc_scratch(a, (size_t)ctx->n_rows * kv_a_rows * sizeof(float), &kv_a_b);
		if (st != OK) {
			a->buffer_free(a, &q_a_b);
			return st;
		}

		if (a->matmul_multi_batch) {
			const buffer *ws[2]	   = {&L->q_a_w.buf, &L->kv_a_w.buf};
			uint32_t	  wts[2]   = {L->q_a_w.type, L->kv_a_w.type};
			buffer		 *ys[2]	   = {&q_a_b, &kv_a_b};
			int			  n_out[2] = {q_lora, kv_a_rows};
			st = a->matmul_multi_batch(a, ws, wts, xb, ys, n_out, dim, 2, ctx->n_rows);
		} else {
			st = a->matmul_batch(a, &L->q_a_w.buf, L->q_a_w.type, xb, &q_a_b, q_lora, dim,
								 ctx->n_rows);
			if (st == OK)
				st = a->matmul_batch(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, &kv_a_b, kv_a_rows, dim,
									 ctx->n_rows);
		}
		if (st != OK)
			goto mla_qkv_cleanup;

		st = a->rmsnorm_batch(a, &q_a_b, &L->q_a_norm_w.buf, &q_a_b, q_lora, ctx->m->norm_eps,
							  ctx->n_rows);
		if (st != OK)
			goto mla_qkv_cleanup;

		buffer *q_buf = batch_slot(ctx->bs, RECIPE_SLOT_Q);
		st = a->matmul_batch(a, &L->q_b_w.buf, L->q_b_w.type, &q_a_b, q_buf, q_b_rows, q_lora,
							 ctx->n_rows);
		if (st != OK)
			goto mla_qkv_cleanup;

		for (int row = 0; row < ctx->n_rows; row++) {
			buffer kv_a_row = batch_row_view(&kv_a_b, row, kv_a_rows);
			st = a->kv_put_mla(a, &ctx->cache->mla->kv, ctx->li, ctx->pos_start + row, &kv_a_row,
							   &L->kv_a_norm_w.buf, kv_lora, qk_rope, ctx->cache->n_ctx,
							   ctx->m->norm_eps);
			if (st != OK)
				goto mla_qkv_cleanup;
		}
	mla_qkv_cleanup:
		if (a->synchronize)
			a->synchronize(a);
		a->buffer_free(a, &q_a_b);
		a->buffer_free(a, &kv_a_b);
		return st;
	}

	status_code		st;
	struct model   *m		  = ctx->m;
	struct kvcache *cache	  = ctx->cache;
	int				pos		  = ctx->pos;
	int				li		  = ctx->li;
	buffer		   *slots	  = compute_slots_array(ctx->s);
	backend		   *a		  = model_layer_backend(m, ctx->li);
	layer_weights  *L		  = &m->layers[li];
	const int		dim		  = m->dim;
	const int		q_lora	  = m->mla.q_lora;
	const int		kv_lora	  = m->mla.kv_lora;
	const int		qk_rope	  = m->mla.qk_rope;
	const int		q_b_rows  = m->n_heads * m->mla.qk_head;
	const int		kv_a_rows = kv_lora + qk_rope;

	buffer *xb = &slots[RECIPE_SLOT_XB];

	float  q_a_stack[MLA_Q_A_STACK_CAP];
	int	   q_a_heap;
	buffer q_a_buf = mla_scratch_buf_alloc(q_a_stack, MLA_Q_A_STACK_CAP, q_lora, &q_a_heap);

	float  kv_a_stack[MLA_KV_A_STACK_CAP];
	int	   kv_a_heap;
	buffer kv_a_buf = mla_scratch_buf_alloc(kv_a_stack, MLA_KV_A_STACK_CAP, kv_a_rows, &kv_a_heap);

	if (a->matmul_multi) {
		const buffer *w_list[2]	 = {&L->q_a_w.buf, &L->kv_a_w.buf};
		uint32_t	  w_types[2] = {L->q_a_w.type, L->kv_a_w.type};
		buffer		 *y_list[2]	 = {&q_a_buf, &kv_a_buf};
		int			  n_list[2]	 = {q_lora, kv_a_rows};
		st						 = a->matmul_multi(a, w_list, w_types, xb, y_list, n_list, dim, 2);
	} else {
		st = a->matmul(a, &L->q_a_w.buf, L->q_a_w.type, xb, &q_a_buf, q_lora, dim);
		if (st == OK)
			st = a->matmul(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, &kv_a_buf, kv_a_rows, dim);
	}
	if (st != OK) {
		mla_scratch_buf_free(&q_a_buf, q_a_heap);
		mla_scratch_buf_free(&kv_a_buf, kv_a_heap);
		return st;
	}

	st = a->rmsnorm(a, &q_a_buf, &L->q_a_norm_w.buf, &q_a_buf, q_lora, m->norm_eps);
	if (st == OK) {
		buffer *q_buf = &slots[RECIPE_SLOT_Q];
		st = a->matmul(a, &L->q_b_w.buf, L->q_b_w.type, &q_a_buf, q_buf, q_b_rows, q_lora);
	}
	mla_scratch_buf_free(&q_a_buf, q_a_heap);

	if (st != OK) {
		mla_scratch_buf_free(&kv_a_buf, kv_a_heap);
		return st;
	}

	st = a->kv_put_mla(a, &cache->mla->kv, li, pos, &kv_a_buf, &L->kv_a_norm_w.buf, kv_lora,
					   qk_rope, cache->n_ctx, m->norm_eps);
	mla_scratch_buf_free(&kv_a_buf, kv_a_heap);
	return st;
}

static status_code op_mla_q_proj(exec_ctx *ctx) {
	if (exec_is_batch(ctx)) {

		backend		  *a		= exec_layer_backend(ctx);
		const int	   dim		= ctx->m->dim;
		layer_weights *L		= &ctx->m->layers[ctx->li];
		int			   q_lora	= ctx->m->mla.q_lora;
		int			   q_b_rows = ctx->m->n_heads * ctx->m->mla.qk_head;
		buffer		  *xb		= batch_slot(ctx->bs, ctx->op->in[0]);

		float_buf_ensure(&ctx->bs->pair[RECIPE_SLOT_FFN_GATE].fb, (size_t)ctx->n_rows * q_lora);

		status_code st;
		buffer		q_a_b = {0};
		st = a->buffer_alloc_scratch(a, (size_t)ctx->n_rows * q_lora * sizeof(float), &q_a_b);
		if (st != OK)
			return st;

		st = a->matmul_batch(a, &L->q_a_w.buf, L->q_a_w.type, xb, &q_a_b, q_lora, dim, ctx->n_rows);
		if (st != OK)
			goto mla_q_cleanup;
		st = a->rmsnorm_batch(a, &q_a_b, &L->q_a_norm_w.buf, &q_a_b, q_lora, ctx->m->norm_eps,
							  ctx->n_rows);
		if (st != OK)
			goto mla_q_cleanup;
		buffer *q_buf = batch_slot(ctx->bs, RECIPE_SLOT_Q);
		st = a->matmul_batch(a, &L->q_b_w.buf, L->q_b_w.type, &q_a_b, q_buf, q_b_rows, q_lora,
							 ctx->n_rows);
	mla_q_cleanup:
		if (a->synchronize)
			a->synchronize(a);
		a->buffer_free(a, &q_a_b);
		return st;
	}

	struct model  *m		= ctx->m;
	int			   li		= ctx->li;
	buffer		  *slots	= compute_slots_array(ctx->s);
	backend		  *a		= model_layer_backend(m, ctx->li);
	layer_weights *L		= &m->layers[li];
	const int	   dim		= m->dim;
	const int	   q_lora	= m->mla.q_lora;
	const int	   q_b_rows = m->n_heads * m->mla.qk_head;

	buffer *xb = &slots[RECIPE_SLOT_XB];
	float	q_a_stack[MLA_Q_A_STACK_CAP];
	int		q_a_heap;
	buffer	q_a_buf = mla_scratch_buf_alloc(q_a_stack, MLA_Q_A_STACK_CAP, q_lora, &q_a_heap);

	status_code st = a->matmul(a, &L->q_a_w.buf, L->q_a_w.type, xb, &q_a_buf, q_lora, dim);
	if (st == OK)
		st = a->rmsnorm(a, &q_a_buf, &L->q_a_norm_w.buf, &q_a_buf, q_lora, m->norm_eps);
	if (st == OK) {
		buffer *q_buf = &slots[RECIPE_SLOT_Q];
		st = a->matmul(a, &L->q_b_w.buf, L->q_b_w.type, &q_a_buf, q_buf, q_b_rows, q_lora);
	}
	mla_scratch_buf_free(&q_a_buf, q_a_heap);
	return st;
}

static status_code op_mla_kv_proj(exec_ctx *ctx) {
	if (exec_is_batch(ctx)) {

		backend		  *a		 = exec_layer_backend(ctx);
		const int	   dim		 = ctx->m->dim;
		layer_weights *L		 = &ctx->m->layers[ctx->li];
		int			   kv_lora	 = ctx->m->mla.kv_lora;
		int			   qk_rope	 = ctx->m->mla.qk_rope;
		int			   kv_a_rows = kv_lora + qk_rope;
		buffer		  *xb		 = batch_slot(ctx->bs, ctx->op->in[0]);
		float_buf_ensure(&ctx->bs->pair[RECIPE_SLOT_FFN_UP].fb, (size_t)ctx->n_rows * kv_a_rows);

		status_code st;
		buffer		kv_a_b = {0};
		st = a->buffer_alloc_scratch(a, (size_t)ctx->n_rows * kv_a_rows * sizeof(float), &kv_a_b);
		if (st != OK)
			return st;

		st = a->matmul_batch(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, &kv_a_b, kv_a_rows, dim,
							 ctx->n_rows);
		if (st != OK)
			goto mla_kv_cleanup;
		for (int row = 0; row < ctx->n_rows; row++) {
			buffer kv_a_row = batch_row_view(&kv_a_b, row, kv_a_rows);
			st = a->kv_put_mla(a, &ctx->cache->mla->kv, ctx->li, ctx->pos_start + row, &kv_a_row,
							   &L->kv_a_norm_w.buf, kv_lora, qk_rope, ctx->cache->n_ctx,
							   ctx->m->norm_eps);
			if (st != OK)
				goto mla_kv_cleanup;
		}
	mla_kv_cleanup:
		if (a->synchronize)
			a->synchronize(a);
		a->buffer_free(a, &kv_a_b);
		return st;
	}

	struct model   *m		  = ctx->m;
	struct kvcache *cache	  = ctx->cache;
	int				pos		  = ctx->pos;
	int				li		  = ctx->li;
	buffer		   *slots	  = compute_slots_array(ctx->s);
	backend		   *a		  = model_layer_backend(m, ctx->li);
	layer_weights  *L		  = &m->layers[li];
	const int		dim		  = m->dim;
	const int		kv_lora	  = m->mla.kv_lora;
	const int		qk_rope	  = m->mla.qk_rope;
	const int		kv_a_rows = kv_lora + qk_rope;

	float  kv_a_stack[MLA_KV_A_STACK_CAP];
	int	   kv_a_heap;
	buffer kv_a_buf = mla_scratch_buf_alloc(kv_a_stack, MLA_KV_A_STACK_CAP, kv_a_rows, &kv_a_heap);

	buffer	   *xb = &slots[RECIPE_SLOT_XB];
	status_code st = a->matmul(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, &kv_a_buf, kv_a_rows, dim);
	if (st == OK)
		st = a->kv_put_mla(a, &cache->mla->kv, li, pos, &kv_a_buf, &L->kv_a_norm_w.buf, kv_lora,
						   qk_rope, cache->n_ctx, m->norm_eps);
	mla_scratch_buf_free(&kv_a_buf, kv_a_heap);
	return st;
}

static status_code op_attention_mla(exec_ctx *ctx) {
	if (exec_is_batch(ctx)) {
		backend		  *a		= exec_layer_backend(ctx);
		int			   n_heads	= ctx->op->u.attention.n_heads;
		int			   qk_head	= ctx->m->mla.qk_head;
		int			   qk_rope	= ctx->m->mla.qk_rope;
		int			   qk_nope	= ctx->m->mla.qk_nope;
		int			   v_head	= ctx->m->mla.v_head;
		int			   kv_lora	= ctx->m->mla.kv_lora;
		float		   scale	= ctx->op->u.attention.scale;
		layer_weights *L		= &ctx->m->layers[ctx->li];
		buffer		  *q_buf	= batch_slot(ctx->bs, ctx->op->in[0]);
		int			   q_b_rows = ctx->m->n_heads * qk_head;
		int			   wo_in	= n_heads * v_head;

		buffer *out_buf = batch_slot(ctx->bs, ctx->op->out);
		for (int row = 0; row < ctx->n_rows; row++) {
			buffer		q_row	= batch_row_view(q_buf, row, q_b_rows);
			buffer		out_row = batch_row_view(out_buf, row, wo_in);
			status_code st		= a->attention_mla(
				a, &q_row, &ctx->cache->mla->kv, &L->k_b_w.buf, &L->v_b_w.buf, &out_row, ctx->li,
				ctx->pos_start + row, n_heads, qk_head, qk_rope, qk_nope, v_head, kv_lora,
				ctx->cache->n_ctx, ctx->s->rope_cos, ctx->s->rope_sin, scale);
			if (st != OK)
				return st;
		}
		return OK;
	}

	const recipe_op		   *op	  = ctx->op;
	struct model		   *m	  = ctx->m;
	struct kvcache		   *cache = ctx->cache;
	struct compute_scratch *s	  = ctx->s;
	int						li	  = ctx->li;
	int						pos	  = ctx->pos;
	buffer				   *slots = compute_slots_array(ctx->s);
	(void)ctx->flash_attn;
	backend		  *a	   = model_layer_backend(m, ctx->li);
	layer_weights *L	   = &m->layers[li];
	int			   n_heads = op->u.attention.n_heads;
	int			   qk_head = m->mla.qk_head;
	int			   qk_rope = m->mla.qk_rope;
	int			   qk_nope = m->mla.qk_nope;
	int			   v_head  = m->mla.v_head;
	int			   kv_lora = m->mla.kv_lora;
	float		   scale   = op->u.attention.scale;

	return a->attention_mla(a, &slots[RECIPE_SLOT_Q], &cache->mla->kv, &L->k_b_w.buf, &L->v_b_w.buf,
							&slots[op->out], li, pos, n_heads, qk_head, qk_rope, qk_nope, v_head,
							kv_lora, cache->n_ctx, s->rope_cos, s->rope_sin, scale);
}

typedef struct {
	const float *mixed;
	float		*q, *gate;
	int			 n_heads, head_dim, n_rows;
} qwen_split_job;

static void qwen_split_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	qwen_split_job *j			 = ctx;
	int				q_out		 = j->n_heads * j->head_dim;
	int				mixed_stride = 2 * q_out;
	for (int row = begin; row < end; row++) {
		const float *src = j->mixed + (size_t)row * mixed_stride;
		float		*qd	 = j->q + (size_t)row * q_out;
		float		*gd	 = j->gate + (size_t)row * q_out;
		for (int h = 0; h < j->n_heads; h++, src += 2 * j->head_dim) {
			for (int jj = 0; jj < j->head_dim; jj++) {
				qd[jj] = src[jj];
				gd[jj] = src[jj + j->head_dim];
			}
			qd += j->head_dim;
			gd += j->head_dim;
		}
	}
}

static void split_qgate_rows(tpool *pool, const float *mixed, float *q, float *gate, int n_heads,
							 int head_dim, int n_rows) {
	qwen_split_job job = {.mixed	= mixed,
						  .q		= q,
						  .gate		= gate,
						  .n_heads	= n_heads,
						  .head_dim = head_dim,
						  .n_rows	= n_rows};
	if (pool && n_rows > 1 && tpool_current_tid() < 0)
		tpool_parallel_for(pool, n_rows, 1, qwen_split_chunk, &job);
	else
		qwen_split_chunk(0, n_rows, -1, &job);
}

status_code op_split_qgate(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	model		*m		  = ctx->m;
	int			 head_dim = m->head_dim;
	int			 n_heads  = m->n_heads;
	int			 n_rows	  = recipe_exec_is_batch(ctx) ? ctx->n_rows : 1;
	int			 q_out	  = n_heads * head_dim;
	int			 mixed_n  = n_rows * 2 * q_out;
	int			 out_n	  = n_rows * q_out;
	const float *mixed =
		recipe_slot_read_f32(ctx, RECIPE_SLOT_HYB_PROJ, &ctx->s->hybrid_host, mixed_n);
	float *q	= recipe_slot_write_stage(ctx, RECIPE_SLOT_Q, &ctx->s->hybrid_host2, out_n);
	float *gate = recipe_slot_write_stage(ctx, RECIPE_SLOT_HYB_GATE, &ctx->s->hybrid_host3, out_n);
	if (!mixed || !q || !gate)
		return ERR_INVALID_ARG;
	split_qgate_rows(model_get_pool(m), mixed, q, gate, n_heads, head_dim, n_rows);
	status_code st = recipe_slot_write_commit(ctx, RECIPE_SLOT_Q, q, out_n);
	if (st == OK)
		st = recipe_slot_write_commit(ctx, RECIPE_SLOT_HYB_GATE, gate, out_n);
	return st;
}

typedef struct {
	float		*q, *k;
	const float *cos_base, *sin_base;
	int			 qn, kn, half, rope_dim, n_heads, n_kv_heads, head_dim, pos0, rows;
} qwen_rope_job;

static void qwen_partial_rope_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	qwen_rope_job *j = ctx;
	for (int row = begin; row < end; row++) {
		const float *cosv = j->cos_base + (size_t)(j->pos0 + row) * j->half;
		const float *sinv = j->sin_base + (size_t)(j->pos0 + row) * j->half;
		rope_rotate_neox(j->q + (size_t)row * j->qn, j->n_heads, j->head_dim, j->rope_dim, cosv,
						 sinv);
		rope_rotate_neox(j->k + (size_t)row * j->kn, j->n_kv_heads, j->head_dim, j->rope_dim, cosv,
						 sinv);
	}
}

status_code op_partial_rope_qk(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s || ctx->pos < 0)
		return ERR_INVALID_ARG;
	model *m	= ctx->m;
	int	   qn	= m->n_heads * m->head_dim;
	int	   kn	= m->n_kv_heads * m->head_dim;
	int	   half = m->rope_dim / 2;
	int	   rows = recipe_exec_is_batch(ctx) ? ctx->n_rows : 1;
	int	   pos0 = recipe_exec_is_batch(ctx) ? ctx->pos_start : ctx->pos;
	int	   qn_n = rows * qn;
	int	   kn_n = rows * kn;

	float *q = recipe_slot_rw_f32(ctx, RECIPE_SLOT_Q, &ctx->s->hybrid_host, qn_n);
	float *k = recipe_slot_rw_f32(ctx, RECIPE_SLOT_K, &ctx->s->hybrid_host2, kn_n);
	if (!q || !k)
		return ERR_INVALID_ARG;

	qwen_rope_job job  = {.q		  = q,
						  .k		  = k,
						  .cos_base	  = ctx->s->rope_cos,
						  .sin_base	  = ctx->s->rope_sin,
						  .qn		  = qn,
						  .kn		  = kn,
						  .half		  = half,
						  .rope_dim	  = m->rope_dim,
						  .n_heads	  = m->n_heads,
						  .n_kv_heads = m->n_kv_heads,
						  .head_dim	  = m->head_dim,
						  .pos0		  = pos0,
						  .rows		  = rows};
	tpool		 *pool = model_get_pool(m);
	if (pool && rows > 1 && tpool_current_tid() < 0)
		tpool_parallel_for(pool, rows, 1, qwen_partial_rope_chunk, &job);
	else
		qwen_partial_rope_chunk(0, rows, -1, &job);

	status_code st = recipe_slot_write_commit(ctx, RECIPE_SLOT_Q, q, qn_n);
	if (st == OK)
		st = recipe_slot_write_commit(ctx, RECIPE_SLOT_K, k, kn_n);
	return st;
}

typedef struct {
	float		*out;
	const float *gate;
	int			 n, rows;
} qwen_gate_job;

static void qwen_output_gate_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	qwen_gate_job *j = ctx;
	for (int row = begin; row < end; row++) {
		float		*o = j->out + (size_t)row * j->n;
		const float *g = j->gate + (size_t)row * j->n;
		for (int i = 0; i < j->n; i++)
			o[i] *= sigmoidf(g[i]);
	}
}

status_code op_attn_output_gate(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	int			 n		 = ctx->m->n_heads * ctx->m->head_dim;
	int			 rows	 = recipe_exec_is_batch(ctx) ? ctx->n_rows : 1;
	int			 n_total = rows * n;
	float		*out	 = recipe_slot_rw_f32(ctx, ctx->op->in[0], &ctx->s->hybrid_host, n_total);
	const float *gate =
		recipe_slot_read_f32(ctx, RECIPE_SLOT_HYB_GATE, &ctx->s->hybrid_host2, n_total);
	if (!out || !gate)
		return ERR_INVALID_ARG;

	qwen_gate_job job  = {.out = out, .gate = gate, .n = n, .rows = rows};
	tpool		 *pool = model_get_pool(ctx->m);
	if (pool && rows > 1 && tpool_current_tid() < 0)
		tpool_parallel_for(pool, rows, 1, qwen_output_gate_chunk, &job);
	else
		qwen_output_gate_chunk(0, rows, -1, &job);

	return recipe_slot_write_commit(ctx, ctx->op->in[0], out, n_total);
}

typedef struct {
	float		*conv_out, *conv_state;
	const float *mixed, *conv_w;
	int			 conv_dim, conv_kernel, n_tokens, mixed_stride, history;
} gdn_conv_job;

static void gdn_conv_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	gdn_conv_job *j		  = ctx;
	int			  history = j->history;
	for (int c = begin; c < end; c++) {
		const float *w	   = j->conv_w + (size_t)c * j->conv_kernel;
		float		*hist  = j->conv_state + (size_t)c * history;
		const float *mix_c = j->mixed + c;
		if (history == 3) {
			float h0 = hist[0], h1 = hist[1], h2 = hist[2];
			for (int t = 0; t < j->n_tokens; t++) {
				float m	  = mix_c[(size_t)t * j->mixed_stride];
				float sum = h0 * w[0] + h1 * w[1] + h2 * w[2] + m * w[3];
				h0		  = h1;
				h1		  = h2;
				h2		  = m;
				j->conv_out[(size_t)t * j->conv_dim + c] = silu(sum);
			}
			hist[0] = h0;
			hist[1] = h1;
			hist[2] = h2;
		} else {
			for (int t = 0; t < j->n_tokens; t++) {
				const float *mix = j->mixed + (size_t)t * j->mixed_stride;
				float		*oc	 = j->conv_out + (size_t)t * j->conv_dim;
				float		 sum = mix[c] * w[history];
				if (history > 0) {
					for (int jj = 0; jj < history; jj++)
						sum += hist[jj] * w[jj];
					if (history > 1)
						memmove(hist, hist + 1, (size_t)(history - 1) * sizeof(float));
					hist[history - 1] = mix[c];
				}
				oc[c] = silu(sum);
			}
		}
	}
}

static void gdn_conv_tokens(tpool *pool, float *conv_out, float *conv_state, const float *mixed,
							const float *conv_w, int conv_dim, int conv_kernel, int n_tokens,
							int mixed_stride) {
	gdn_conv_job job = {.conv_out	  = conv_out,
						.conv_state	  = conv_state,
						.mixed		  = mixed,
						.conv_w		  = conv_w,
						.conv_dim	  = conv_dim,
						.conv_kernel  = conv_kernel,
						.n_tokens	  = n_tokens,
						.mixed_stride = mixed_stride,
						.history	  = conv_kernel - 1};
	if (pool && conv_dim > 8 && tpool_current_tid() < 0) {
		tpool_parallel_for(pool, conv_dim, 8, gdn_conv_chunk, &job);
		return;
	}
	gdn_conv_chunk(0, conv_dim, -1, &job);
}

typedef struct {
	float		*state;
	const float *conv;
	const float *z;
	const float *alpha;
	const float *beta;
	float		*out;
	const float *dt;
	const float *a;
	const float *norm_w;
	float		*scratch;
	int			 n_tokens;
	int			 conv_stride;
	int			 z_stride;
	int			 alpha_stride;
	int			 beta_stride;
	int			 out_stride;
	int			 nkh;
	int			 kd;
	int			 vd;
	int			 key_dim;
	int			 scratch_stride;
	float		 eps;
} gdn_job;

static void gdn_heads(int vh0, int vh1, const gdn_job *j, float *scratch) {
	int	   kd		  = j->kd;
	int	   vd		  = j->vd;
	float  q_scale	  = 1.0f / sqrtf((float)kd);
	float *k_s		  = scratch;
	float *q_s		  = k_s + kd;
	float *mem		  = q_s + kd;
	float *delta	  = mem + vd;
	int	   state_head = kd * vd;

	for (int vh = vh0; vh < vh1; vh++) {
		int	   kh	 = vh % j->nkh;
		float *shead = j->state + (size_t)vh * state_head;
		for (int t = 0; t < j->n_tokens; t++) {
			const float *conv_t	 = j->conv + (size_t)t * j->conv_stride;
			const float *q		 = conv_t + (size_t)kh * kd;
			const float *k		 = conv_t + j->key_dim + (size_t)kh * kd;
			const float *v		 = conv_t + 2 * j->key_dim + (size_t)vh * vd;
			const float *z_t	 = j->z + (size_t)t * j->z_stride + (size_t)vh * vd;
			float		*y		 = j->out + (size_t)t * j->out_stride + (size_t)vh * vd;
			float		 alpha_t = j->alpha[(size_t)t * j->alpha_stride + vh];
			float		 beta_t	 = j->beta[(size_t)t * j->beta_stride + vh];

			float decay = expf(j->a[vh] * softplusf(alpha_t + j->dt[vh]));
			float b		= sigmoidf(beta_t);

			float qn = j->eps;
			float kn = j->eps;
			for (int d = 0; d < kd; d++) {
				qn += q[d] * q[d];
				kn += k[d] * k[d];
			}
			qn = q_scale / sqrtf(qn);
			kn = 1.0f / sqrtf(kn);
			for (int d = 0; d < kd; d++) {
				q_s[d] = q[d] * qn;
				k_s[d] = k[d] * kn;
			}

			memset(mem, 0, (size_t)vd * sizeof(float));
			for (int d = 0; d < kd; d++) {
				float *row = shead + (size_t)d * vd;
				float  ks  = k_s[d];
				for (int jj = 0; jj < vd; jj++) {
					row[jj] *= decay;
					mem[jj] += row[jj] * ks;
				}
			}
			for (int jj = 0; jj < vd; jj++)
				delta[jj] = (v[jj] - mem[jj]) * b;
			memset(y, 0, (size_t)vd * sizeof(float));
			for (int d = 0; d < kd; d++) {
				float *row = shead + (size_t)d * vd;
				float  ks  = k_s[d];
				float  qs  = q_s[d];
				for (int jj = 0; jj < vd; jj++) {
					row[jj] += ks * delta[jj];
					y[jj] += row[jj] * qs;
				}
			}

			float mean_sq = j->eps;
			for (int jj = 0; jj < vd; jj++)
				mean_sq += y[jj] * y[jj] / (float)vd;
			float inv_rms = 1.0f / sqrtf(mean_sq);
			for (int jj = 0; jj < vd; jj++)
				y[jj] = y[jj] * inv_rms * j->norm_w[jj] * silu(z_t[jj]);
		}
	}
}

static void gdn_chunk(int begin, int end, int tid, void *ctx) {
	gdn_job *j = (gdn_job *)ctx;
	gdn_heads(begin, end, j, j->scratch + (size_t)tid * j->scratch_stride);
}

static status_code gdn_run(exec_ctx *ctx, const float *mixed, const float *z, const float *alpha,
						   const float *beta, float *out, float *ws, tpool *pool) {
	model					  *m	  = ctx->m;
	const model_hybrid_params *p	  = &m->hybrid;
	layer_weights			  *L	  = &m->layers[ctx->li];
	const float				  *conv_w = (const float *)L->ssm_conv1d_w.host_ptr;
	const float				  *dt	  = (const float *)L->ssm_dt_b.host_ptr;
	const float				  *a	  = (const float *)L->ssm_a.host_ptr;
	const float				  *norm_w = (const float *)L->ssm_norm_w.host_ptr;
	if (!conv_w || !dt || !a || !norm_w)
		return ERR_FORMAT;

	kvcache_hybrid *cache	   = ctx->cache->hybrid;
	float		   *conv_state = cache->conv_state + (size_t)ctx->li * cache->conv_stride;
	float		   *state	   = cache->recurrent_state + (size_t)ctx->li * cache->recurrent_stride;
	if (ctx->pos_start == 0) {
		memset(conv_state, 0, cache->conv_stride * sizeof(float));
		memset(state, 0, cache->recurrent_stride * sizeof(float));
	}

	int	   n_tokens		  = ctx->n_rows > 0 ? ctx->n_rows : 1;
	int	   scratch_stride = 2 * p->state_size + 2 * p->value_head_dim;
	float *conv			  = ws;
	float *scratch		  = ws + (size_t)n_tokens * p->conv_dim;

	gdn_conv_tokens(pool, conv, conv_state, mixed, conv_w, p->conv_dim, p->conv_kernel, n_tokens,
					p->conv_dim);

	gdn_job job = {
		.state			= state,
		.conv			= conv,
		.z				= z,
		.alpha			= alpha,
		.beta			= beta,
		.out			= out,
		.dt				= dt,
		.a				= a,
		.norm_w			= norm_w,
		.scratch		= scratch,
		.n_tokens		= n_tokens,
		.conv_stride	= p->conv_dim,
		.z_stride		= p->value_dim,
		.alpha_stride	= p->n_value_heads,
		.beta_stride	= p->n_value_heads,
		.out_stride		= p->value_dim,
		.nkh			= p->n_key_heads,
		.kd				= p->state_size,
		.vd				= p->value_head_dim,
		.key_dim		= p->key_dim,
		.scratch_stride = scratch_stride,
		.eps			= m->norm_eps,
	};
	if (pool && p->n_value_heads > 1)
		tpool_parallel_for(pool, p->n_value_heads, 1, gdn_chunk, &job);
	else
		gdn_heads(0, p->n_value_heads, &job, scratch);
	return OK;
}

status_code op_gated_delta_net(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->cache || !ctx->s || !ctx->cache->hybrid)
		return ERR_INVALID_ARG;
	profile_scope			   ps		= profile_begin(&ctx->s->prof, ctx->op->stage);
	const model_hybrid_params *p		= &ctx->m->hybrid;
	int						   n_tokens = ctx->n_rows > 0 ? ctx->n_rows : 1;

	tpool *pool		 = model_get_pool(ctx->m);
	int	   n_threads = pool ? tpool_n_threads(pool) : 1;
	if (n_threads < 1)
		n_threads = 1;

	size_t conv_need	= (size_t)n_tokens * p->conv_dim;
	size_t scratch_need = (size_t)n_threads * (2 * p->state_size + 2 * p->value_head_dim);

	int mixed_n = n_tokens * p->conv_dim;
	int z_n		= n_tokens * p->value_dim;
	int alpha_n = n_tokens * p->n_value_heads;
	int beta_n	= n_tokens * p->n_value_heads;
	int out_n	= n_tokens * p->value_dim;

	const float *mixed =
		recipe_slot_read_f32(ctx, RECIPE_SLOT_HYB_PROJ, &ctx->s->hybrid_host, mixed_n);
	const float *z = recipe_slot_read_f32(ctx, RECIPE_SLOT_HYB_GATE, &ctx->s->gdn_z_host, z_n);
	const float *alpha =
		recipe_slot_read_f32(ctx, RECIPE_SLOT_HYB_ALPHA, &ctx->s->gdn_alpha_host, alpha_n);
	const float *beta =
		recipe_slot_read_f32(ctx, RECIPE_SLOT_HYB_BETA, &ctx->s->gdn_beta_host, beta_n);
	float *out = recipe_slot_write_stage(ctx, ctx->op->out, &ctx->s->gdn_out_host, out_n);
	if (!mixed || !z || !alpha || !beta || !out)
		return ERR_INVALID_ARG;

	float	   *ws = float_buf_ensure(&ctx->s->gdn_ws_host, conv_need + scratch_need);
	status_code st = gdn_run(ctx, mixed, z, alpha, beta, out, ws, pool);
	if (st == OK)
		st = recipe_slot_write_commit(ctx, ctx->op->out, out, out_n);
	profile_end(&ctx->s->prof, &ps);
	return st;
}
static int lfm_attn_buf(const model *m) {
	int buf	  = m->dim;
	int q_out = m->n_heads * m->head_dim;
	return buf > q_out ? buf : q_out;
}

typedef struct {
	float		*out;
	float		*state;
	const float *mixed;
	const float *conv_w;
	int			 dim, kernel, history;
	int			 in_stride, out_stride;
	int			 n_rows;
} shortconv_job;

static void shortconv_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	shortconv_job *j = ctx;
	const int	   K = j->kernel;
	for (int c = begin; c < end; c++) {
		const float *w	   = j->conv_w + (size_t)c * K;
		float		*hist  = j->state + (size_t)c * j->history;
		const float *b_col = j->mixed + c;
		const float *c_col = j->mixed + (size_t)j->dim + c;
		const float *x_col = j->mixed + 2 * (size_t)j->dim + c;
		float		*o_col = j->out + c;

		if (K == 3) {
			float h0 = hist[0], h1 = hist[1];
			for (int t = 0; t < j->n_rows; t++) {
				size_t off						 = (size_t)t * j->in_stride;
				float  bx						 = b_col[off] * x_col[off];
				float  sum						 = h0 * w[0] + h1 * w[1] + bx * w[2];
				h0								 = h1;
				h1								 = bx;
				o_col[(size_t)t * j->out_stride] = c_col[off] * sum;
			}
			hist[0] = h0;
			hist[1] = h1;
		} else {
			for (int t = 0; t < j->n_rows; t++) {
				size_t off = (size_t)t * j->in_stride;
				float  bx  = b_col[off] * x_col[off];
				float  sum = bx * w[K - 1];
				for (int k = 0; k < j->history; k++)
					sum += hist[k] * w[k];
				memmove(hist, hist + 1, (size_t)(j->history - 1) * sizeof(float));
				hist[j->history - 1]			 = bx;
				o_col[(size_t)t * j->out_stride] = c_col[off] * sum;
			}
		}
	}
}

status_code op_shortconv(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->cache || !ctx->cache->hybrid || !ctx->s)
		return ERR_INVALID_ARG;
	model					  *m	  = ctx->m;
	layer_weights			  *L	  = &m->layers[ctx->li];
	kvcache_hybrid			  *hc	  = ctx->cache->hybrid;
	const model_hybrid_params *p	  = &m->hybrid;
	const float				  *conv_w = (const float *)L->ssm_conv1d_w.host_ptr;
	if (!conv_w || p->conv_kernel < 2 || ctx->li < 0 || ctx->li >= m->n_layers)
		return ERR_FORMAT;

	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);

	int	   n_rows = recipe_exec_is_batch(ctx) ? ctx->n_rows : 1;
	float *state  = hc->conv_state + (size_t)ctx->li * hc->conv_stride;
	if (ctx->pos_start == 0)
		memset(state, 0, (size_t)m->dim * (size_t)(p->conv_kernel - 1) * sizeof(float));

	int			 in_stride	= model_hybrid_proj_size(m);
	int			 out_stride = lfm_attn_buf(m);
	int			 mixed_n	= n_rows * in_stride;
	int			 out_n		= n_rows * out_stride;
	const float *mixed = recipe_slot_read_f32(ctx, ctx->op->in[0], &ctx->s->hybrid_host, mixed_n);
	float		*out   = recipe_slot_write_stage(ctx, ctx->op->out, &ctx->s->hybrid_host2, out_n);

	shortconv_job job = {
		.out		= out,
		.state		= state,
		.mixed		= mixed,
		.conv_w		= conv_w,
		.dim		= m->dim,
		.kernel		= p->conv_kernel,
		.history	= p->conv_kernel - 1,
		.in_stride	= in_stride,
		.out_stride = out_stride,
		.n_rows		= n_rows,
	};
	if (!job.out || !job.mixed)
		return ERR_INVALID_ARG;

	tpool *pool = model_get_pool(m);
	if (pool && m->dim > 8 && tpool_current_tid() < 0)
		tpool_parallel_for(pool, m->dim, 8, shortconv_chunk, &job);
	else
		shortconv_chunk(0, m->dim, -1, &job);

	status_code st = recipe_slot_write_commit(ctx, ctx->op->out, out, out_n);

	profile_end(&ctx->s->prof, &ps);
	return st;
}

static const op_handler g_op_dispatch[OP_KIND_COUNT] = {
	[OP_EMBD_LOOKUP]		 = op_embd_lookup,
	[OP_SCALE_EMBEDDINGS]	 = op_scale_embeddings,
	[OP_RMSNORM]			 = op_rmsnorm,
	[OP_RMSNORM_PER_HEAD]	 = op_rmsnorm_per_head,
	[OP_RMSNORM_NOWEIGHT]	 = op_rmsnorm_noweight,
	[OP_RMSNORM_ADD]		 = op_rmsnorm_add,
	[OP_MATMUL]				 = op_matmul,
	[OP_MATMUL_RESIDUAL]	 = op_matmul_residual,
	[OP_MATMUL_MULTI]		 = op_matmul_multi,
	[OP_MATMUL_FUSED_GATEUP] = op_matmul_gateup,
	[OP_MATMUL_FFN_DOWN]	 = op_matmul_ffn_down,
	[OP_ROPE]				 = op_rope,
	[OP_ROPE_QK_FUSED]		 = op_rope_qk_fused,
	[OP_ROPE_EXT]			 = op_rope_ext,
	[OP_KV_PUT]				 = op_kv_put,
	[OP_ATTENTION]			 = op_attention,
	[OP_ATTENTION_SWA]		 = op_attention,
	[OP_ADD]				 = op_add,
	[OP_SWAP]				 = op_swap,
	[OP_FFN_ACTIVATE]		 = op_ffn_activate,
	[OP_FFN_ACTIVATE_EX]	 = op_ffn_activate_ex,
	[OP_FFN_ACTIVATE_FUSED]	 = op_ffn_activate_fused,
	[OP_SCALE]				 = op_scale,
	[OP_SOFTCAP]			 = op_softcap,
	[OP_LOGITS_READBACK]	 = op_logits_readback,
	[OP_PLE_BUILD]			 = op_ple_build,
	[OP_PLE_PROJ_INJECT]	 = op_ple_proj_inject,
	[OP_MLA_Q_PROJ]			 = op_mla_q_proj,
	[OP_MLA_KV_PROJ]		 = op_mla_kv_proj,
	[OP_MLA_QKV_PROJ_FUSED]	 = op_mla_qkv_proj_fused,
	[OP_ATTENTION_MLA]		 = op_attention_mla,
	[OP_MOE_ROUTER]			 = op_moe_router,
	[OP_MOE_EXPERTS]		 = op_moe_experts,
	[OP_MOE_SHARED]			 = op_moe_shared,
	[OP_SPLIT_QGATE]		 = op_split_qgate,
	[OP_PARTIAL_ROPE_QK]	 = op_partial_rope_qk,
	[OP_ATTN_OUTPUT_GATE]	 = op_attn_output_gate,
	[OP_GATED_DELTA_NET]	 = op_gated_delta_net,
	[OP_SHORTCONV]			 = op_shortconv,
};

static int recipe_ops_are_batchable(const recipe_op *ops, int n) {
	if (!ops || n <= 0)
		return 0;
	for (int j = 0; j < n; j++) {
		if (ops[j].kind == OP_NONE)
			continue;
		if (ops[j].kind < 0 || ops[j].kind >= OP_KIND_COUNT || !g_op_dispatch[ops[j].kind])
			return 0;
	}
	return 1;
}

static int recipe_is_batchable(const model *m) {
	const model_recipe *r = m->recipe;
	if (!r)
		return 0;
	if (r->per_layer_ops) {
		if (r->layer.n_ops <= 0)
			return 0;
		for (int li = 0; li < m->n_layers; li++) {
			if (!recipe_ops_are_batchable(&r->per_layer_ops[(size_t)li * r->layer.n_ops],
										  r->layer.n_ops))
				return 0;
		}
	} else if (!recipe_ops_are_batchable(r->layer.ops, r->layer.n_ops)) {
		return 0;
	}
	for (int i = 0; i < r->n_pre_ops; i++) {
		op_kind k = r->pre_ops[i].kind;
		if (k == OP_EMBD_LOOKUP || k == OP_SCALE_EMBEDDINGS || k == OP_NONE)
			continue;
		if (k < 0 || k >= OP_KIND_COUNT || !g_op_dispatch[k])
			return 0;
	}
	for (int i = 0; i < r->n_post_ops; i++) {
		op_kind k = r->post_ops[i].kind;
		if (k == OP_NONE)
			continue;
		if (k < 0 || k >= OP_KIND_COUNT || !g_op_dispatch[k])
			return 0;
	}
	return 1;
}

static status_code exec_op(const recipe_op *op, struct model *m, struct kvcache *cache,
						   struct compute_scratch *s, int token, int pos, int li, int flash_attn,
						   float *logits_out) {
	if (op->kind < 0 || op->kind >= OP_KIND_COUNT)
		return ERR_INVALID_ARG;
	op_handler h = g_op_dispatch[op->kind];
	if (!h)
		return ERR_UNSUPPORTED;
	exec_ctx ctx = {
		.op			= op,
		.m			= m,
		.cache		= cache,
		.s			= s,
		.bs			= NULL,
		.token		= token,
		.pos		= pos,
		.li			= li,
		.flash_attn = flash_attn,
		.n_rows		= 1,
		.pos_start	= pos,
		.logits_out = logits_out,
	};
	return h(&ctx);
}

static int model_has_sliding_layers(const model *m) {
	for (int i = 0; i < m->n_layers; i++)
		if (m->layers[i].is_sliding)
			return 1;
	return 0;
}

static int backend_supports_batch_ops(const model *m) {
	const backend *a = m->backend;
	if (!(a->matmul_batch && a->rmsnorm_batch && a->add_batch && a->ffn_activate_batch &&
		  a->rope_batch && a->rope_qk_batch && a->attention_batch))
		return 0;
	if (model_has_sliding_layers(m) && !a->attention_swa_batch)
		return 0;
	return 1;
}

static status_code exec_op_batch(const recipe_op *op, model *m, kvcache *cache, compute_scratch *s,
								 batch_scratch *bs, int pos_start, int n_tokens, int li,
								 int flash_attn) {
	if (op->kind < 0 || op->kind >= OP_KIND_COUNT)
		return ERR_INVALID_ARG;
	op_handler h = g_op_dispatch[op->kind];
	if (!h)
		return ERR_INVALID_ARG;
	exec_ctx ctx = {
		.op			= op,
		.m			= m,
		.cache		= cache,
		.s			= s,
		.bs			= bs,
		.token		= 0,
		.pos		= pos_start,
		.li			= li,
		.flash_attn = flash_attn,
		.n_rows		= n_tokens,
		.pos_start	= pos_start,
		.logits_out = NULL,
	};
	return h(&ctx);
}

typedef struct {
	model		  *m;
	backend		  *a;
	const int32_t *tokens;
	buffer		  *xs;
	int			   dim;
	int			   scale_embd;
	_Atomic int	  *first_err;
} embd_fill_job;

static void embd_fill_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	embd_fill_job *j = ctx;
	for (int row = begin; row < end; row++) {
		buffer		xrow = batch_row_view(j->xs, row, j->dim);
		status_code st	 = j->a->embd_lookup(j->a, &j->m->tok_embd.buf, j->m->tok_embd.type,
											 j->tokens[row], j->dim, &xrow);
		if (st != OK) {
			atomic_store_explicit(j->first_err, st, memory_order_relaxed);
			continue;
		}
		if (j->scale_embd) {
			float *xf = batch_buf_ptr(&xrow);
			for (int i = 0; i < j->dim; i++)
				xf[i] *= j->m->dim_sqrt;
		}
	}
}

static status_code compute_forward_batch_recipe_fast(struct model *m, struct kvcache *cache,
													 struct compute_scratch *s,
													 const int32_t *tokens, int n_tokens,
													 int pos_start, int flash_attn,
													 float *logits_out) {
	if (n_tokens < 2)
		return ERR_FALLBACK;
	if (model_mixed_backend_mode(m))
		return ERR_FALLBACK;
	if (!backend_supports_batch_ops(m))
		return ERR_FALLBACK;

	if (m->batchable < 0)
		m->batchable = recipe_is_batchable(m) ? 1 : 0;
	if (!m->batchable)
		return ERR_FALLBACK;

	backend			   *a	= m->backend;
	const model_recipe *r	= m->recipe;
	const int			dim = m->dim;

	const int max_head_dim = r->max_head_dim;
	const int max_kv_heads = r->max_kv_heads;
	const int max_inter	   = r->max_intermediate;
	const int q_out		   = m->n_heads * max_head_dim;
	const int kv_out	   = max_kv_heads * max_head_dim;

	int attn_buf_size = dim;
	if (q_out > attn_buf_size)
		attn_buf_size = q_out;
	if (m->arch_info->is_mla) {
		int mla_out = m->n_heads * m->mla.v_head;
		if (mla_out > attn_buf_size)
			attn_buf_size = mla_out;
	}
	if (m->arch_info->is_hybrid_recurrent && m->hybrid.value_dim > attn_buf_size)
		attn_buf_size = m->hybrid.value_dim;

	if (!s->bs) {
		s->bs = xcalloc(1, sizeof(batch_scratch));
	}
	batch_scratch *bs	 = s->bs;
	bs->tokens			 = tokens;
	bs->n_tokens_stashed = n_tokens;
	uint32_t slot_mask	 = r->bs_slot_mask;
	if (!slot_mask)
		slot_mask = 0xffffffffu;
	batch_scratch_alloc(bs, a, n_tokens, dim, q_out, kv_out, max_inter, attn_buf_size, slot_mask);
	if (m->arch_info->is_hybrid_recurrent) {
		bs_ensure_slot(bs, a, RECIPE_SLOT_HYB_PROJ, (size_t)n_tokens * model_hybrid_proj_size(m));
		bs_ensure_slot(bs, a, RECIPE_SLOT_HYB_GATE, (size_t)n_tokens * model_hybrid_gate_size(m));
		bs_ensure_slot(bs, a, RECIPE_SLOT_HYB_ALPHA, (size_t)n_tokens * m->hybrid.n_value_heads);
		bs_ensure_slot(bs, a, RECIPE_SLOT_HYB_BETA, (size_t)n_tokens * m->hybrid.n_value_heads);
	}

	status_code st = OK;

	tpool *embd_pool = (a->get_pool) ? a->get_pool(a) : NULL;
	if (embd_pool && n_tokens >= 16 && backend_has_cap(a, BCAP_IS_HOST) &&
		tpool_current_tid() < 0 && tpool_n_threads(embd_pool) > 1) {
		_Atomic int	  embd_err = OK;
		embd_fill_job efj	   = {.m		  = m,
								  .a		  = a,
								  .tokens	  = tokens,
								  .xs		  = &bs->pair[RECIPE_SLOT_X].b,
								  .dim		  = dim,
								  .scale_embd = m->arch_info->has_scale_embeddings,
								  .first_err  = &embd_err};
		tpool_parallel_for(embd_pool, n_tokens, 4, embd_fill_chunk, &efj);
		st = atomic_load_explicit(&embd_err, memory_order_relaxed);
		if (st != OK)
			goto done;
	} else {
		for (int row = 0; row < n_tokens; row++) {
			buffer xrow = batch_row_view(&bs->pair[RECIPE_SLOT_X].b, row, dim);
			st = a->embd_lookup(a, &m->tok_embd.buf, m->tok_embd.type, tokens[row], dim, &xrow);
			if (st != OK)
				goto done;
		}
		if (m->arch_info->has_scale_embeddings) {
			if (a->scale_inplace) {
				st = a->scale_inplace(a, &bs->pair[RECIPE_SLOT_X].b, m->dim_sqrt, n_tokens * dim);
			} else {
				if (a->synchronize)
					a->synchronize(a);
				float  scale = m->dim_sqrt;
				float *xf	 = batch_buf_ptr(&bs->pair[RECIPE_SLOT_X].b);
				for (int i = 0; i < n_tokens * dim; i++)
					xf[i] *= scale;
			}
		}
	}

	for (int i = 0; i < r->n_pre_ops; i++) {
		const recipe_op *pop = &r->pre_ops[i];
		if (pop->kind == OP_EMBD_LOOKUP || pop->kind == OP_SCALE_EMBEDDINGS)
			continue;
		st = exec_op_batch(pop, m, cache, s, bs, pos_start, n_tokens, -1, flash_attn);
		if (st != OK) {
			ERROR("batch fast-path: pre_op[%d] kind=%d (stage=%d) failed with status=%d", i,
				  pop->kind, pop->stage, (int)st);
			goto done;
		}
	}

	if (a->begin_batch)
		a->begin_batch(a);

	const recipe_op *ops_base	= r->per_layer_ops ? r->per_layer_ops : r->layer.ops;
	int				 ops_stride = r->per_layer_ops ? r->layer.n_ops : 0;

	for (int li = 0; li < m->n_layers; li++) {
		const recipe_op *ops = &ops_base[(size_t)li * ops_stride];
		int				 j	 = 0;
		while (j < r->layer.n_ops) {
			if (ops[j].kind == OP_NONE) {
				j++;
				continue;
			}
			if (ops[j].coalesce_run_len > 1) {
				status_code coalesced_status;
				int			consumed = exec_matmul_run_coalesced_batch(
					&ops[j], ops[j].coalesce_run_len, m, s, bs, li, n_tokens, &coalesced_status);
				if (consumed == 0 && coalesced_status == OK) {
					buffer *ys[RECIPE_COALESCE_MAX];
					for (int q = 0; q < ops[j].coalesce_run_len; q++)
						ys[q] = batch_slot(bs, ops[j + q].out);
					consumed = exec_matmul_run_qonly(&ops[j], ops[j].coalesce_run_len, m,
													 batch_slot(bs, ops[j].in[0]), ys, li, n_tokens,
													 &coalesced_status);
				}
				if (consumed < 0) {
					st = coalesced_status;
					goto done;
				}
				if (consumed > 0) {
					if (coalesced_status != OK) {
						st = coalesced_status;
						goto done;
					}
					j += consumed;
					continue;
				}
			}
			st = exec_op_batch(&ops[j], m, cache, s, bs, pos_start, n_tokens, li, flash_attn);
			if (st != OK) {
				ERROR("batch fast-path: layer %d op[%d] kind=%d (stage=%d) "
					  "w_idx=%d failed with status=%d",
					  li, j, ops[j].kind, ops[j].stage, ops[j].w_idx, (int)st);
				goto done;
			}
			j++;
		}
		if (s->layer_cb)
			s->layer_cb(li + 1, m->n_layers, -1, 1, s->layer_cb_ud);

		if (s->interrupt && *s->interrupt) {
			st = ERR_INTERRUPTED;
			goto done;
		}
	}

	if (a->end_batch)
		a->end_batch(a);

	if (a->synchronize)
		a->synchronize(a);

	{
		buffer	 last_x	 = batch_row_view(&bs->pair[RECIPE_SLOT_X].b, n_tokens - 1, dim);
		backend *owner_x = s->slots[RECIPE_SLOT_X].owner;
		float	*tmp	 = float_buf_ensure(&s->batch_logits_tmp, (size_t)dim);
		st				 = a->buffer_read_f32(a, &last_x, tmp, dim);
		if (st != OK)
			goto done;
		st = owner_x->buffer_write_f32(owner_x, &s->slots[RECIPE_SLOT_X], tmp, dim);
		if (st != OK)
			goto done;

		for (int i = 0; i < r->n_post_ops; i++) {
			st = exec_op(&r->post_ops[i], m, cache, s, tokens[n_tokens - 1],
						 pos_start + n_tokens - 1, -1, flash_attn, logits_out);
			if (st != OK)
				goto done;
		}
	}

done:
	if (st != OK && logits_out)
		memset(logits_out, 0, (size_t)m->vocab_size * sizeof(float));
	return st;
}

status_code compute_forward_batch_recipe(struct model *m, struct kvcache *cache,
										 struct compute_scratch *s, const int32_t *tokens,
										 int n_tokens, int pos_start, int flash_attn,
										 float *logits_out) {
	status_code fast_status = compute_forward_batch_recipe_fast(m, cache, s, tokens, n_tokens,
																pos_start, flash_attn, logits_out);
	if (fast_status != ERR_FALLBACK && fast_status != ERR_UNSUPPORTED)
		return fast_status;

	if (fast_status == ERR_UNSUPPORTED)
		WARN("batch fast-path returned ERR_UNSUPPORTED — falling back to "
			 "per-token dispatch (see ERROR above for the failing op). "
			 "Performance will be reduced.");

	if (n_tokens <= 0)
		return OK;
	if (!m || !m->recipe)
		return ERR_INVALID_ARG;

	backend *bk = m->backend;

	int chunked = (n_tokens > 1) && bk->begin_batch && bk->end_batch;

	if (chunked)
		bk->begin_batch(bk);

	for (int i = 0; i < n_tokens; i++) {
		int			is_last = (i == n_tokens - 1);
		float	   *out		= is_last ? logits_out : NULL;
		status_code st		= compute_forward_recipe_one(m, cache, s, tokens[i], pos_start + i,
														 flash_attn, out, !chunked, is_last);
		if (st != OK) {
			if (chunked && bk->end_batch)
				bk->end_batch(bk);
			if (logits_out)
				memset(logits_out, 0, (size_t)m->vocab_size * sizeof(float));
			return st;
		}
	}

	if (chunked && bk->end_batch)
		bk->end_batch(bk);
	return OK;
}
recipe_op mk_rmsnorm(uint8_t in, uint8_t out, uint8_t widx, float eps, stage stage) {
	recipe_op op = {
		.kind	   = OP_RMSNORM,
		.in		   = {in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	   = out,
		.w_idx	   = widx,
		.stage	   = stage,
		.u.rmsnorm = {.eps = eps},
	};
	return op;
}

recipe_op mk_rmsnorm_add(uint8_t in, uint8_t residual, uint8_t out, uint8_t widx, float eps,
						 stage stage) {
	recipe_op op = {
		.kind	   = OP_RMSNORM_ADD,
		.in		   = {in, residual, RECIPE_SLOT_NONE},
		.out	   = out,
		.w_idx	   = widx,
		.stage	   = stage,
		.u.rmsnorm = {.eps = eps},
	};
	return op;
}

recipe_op mk_matmul(uint8_t in, uint8_t out, uint8_t widx, int n, int k, stage stage) {
	recipe_op op = {
		.kind	  = OP_MATMUL,
		.in		  = {in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	  = out,
		.w_idx	  = widx,
		.stage	  = stage,
		.u.matmul = {.n = n, .k = k},
	};
	return op;
}

recipe_op mk_matmul_multi2(uint8_t in, uint8_t out, uint8_t widx, int k, int n0, int n1) {
	recipe_op op = {
		.kind			= OP_MATMUL_MULTI,
		.in				= {in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out			= out,
		.w_idx			= widx,
		.stage			= STAGE_MATMUL,
		.u.matmul_multi = {.n = 2, .k = k, .n_out = {n0, n1}},
	};
	return op;
}

recipe_op mk_add(uint8_t in0, uint8_t in1, stage stage) {
	recipe_op op = {
		.kind  = OP_ADD,
		.in	   = {in0, in1, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = stage,
	};
	return op;
}

recipe_op mk_swap(uint8_t in0, uint8_t in1, stage stage) {
	recipe_op op = {
		.kind  = OP_SWAP,
		.in	   = {in0, in1, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = stage,
	};
	return op;
}

recipe_op mk_kvput(uint8_t k_in, uint8_t v_in) {
	recipe_op op = {
		.kind  = OP_KV_PUT,
		.in	   = {k_in, v_in, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_KVPUT,
	};
	return op;
}

recipe_op mk_attention(uint8_t q_in, uint8_t out, int n_heads, int n_kv_heads, int head_dim,
					   int n_ctx, float scale, int sliding_window) {
	recipe_op op = {
		.kind  = OP_ATTENTION,
		.in	   = {q_in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = out,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
		.u.attention =
			{
				.n_heads		   = n_heads,
				.n_kv_heads		   = n_kv_heads,
				.head_dim		   = head_dim,
				.n_ctx			   = n_ctx,
				.scale			   = scale,
				.sliding_window	   = sliding_window,
				.n_kv_heads_active = n_kv_heads,
			},
	};
	return op;
}

recipe_op mk_rope(uint8_t in, int n_heads, int head_dim, int rope_neox) {
	recipe_op op = {
		.kind	= OP_ROPE,
		.in		= {in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	= RECIPE_SLOT_NONE,
		.w_idx	= RECIPE_NO_WEIGHT,
		.stage	= STAGE_ROPE,
		.u.rope = {.n_heads = n_heads, .head_dim = head_dim, .rope_neox = rope_neox},
	};
	return op;
}

recipe_op mk_rope_qk_fused(int n_heads, int n_kv_heads, int head_dim, int rope_neox) {
	recipe_op op = {
		.kind	= OP_ROPE_QK_FUSED,
		.in		= {RECIPE_SLOT_Q, RECIPE_SLOT_K, RECIPE_SLOT_NONE},
		.out	= RECIPE_SLOT_NONE,
		.w_idx	= RECIPE_NO_WEIGHT,
		.stage	= STAGE_ROPE,
		.u.rope = {.n_heads	   = n_heads,
				   .n_kv_heads = n_kv_heads,
				   .head_dim   = head_dim,
				   .rope_neox  = rope_neox},
	};
	return op;
}

recipe_op mk_rope_ext(uint8_t in, int rope_neox) {
	recipe_op op = {
		.kind		= OP_ROPE_EXT,
		.in			= {in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out		= RECIPE_SLOT_NONE,
		.w_idx		= RECIPE_NO_WEIGHT,
		.stage		= STAGE_ROPE,
		.u.rope_ext = {.n_heads = 0, .head_dim = 0, .use_freq_factors = 1, .rope_neox = rope_neox},
	};
	return op;
}

recipe_op mk_partial_rope_qk(void) {
	recipe_op op = {
		.kind  = OP_PARTIAL_ROPE_QK,
		.in	   = {RECIPE_SLOT_Q, RECIPE_SLOT_K, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ROPE,
	};
	return op;
}

int recipe_append_dense_ffn(recipe_op *ops, int i, const struct model *m, int li) {
	int	  dim			   = m->dim;
	int	  inter			   = m->intermediate;
	float eps			   = m->norm_eps;
	int	  has_matmul_multi = backend_has_cap(m->backend, BCAP_MULTI_MATMUL);
	int	  gate_up_fused	   = li >= 0 && m->layers[li].gate_up_fused;

	ops[i++] = mk_add(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD);
	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, eps, STAGE_RMSNORM);

	if (gate_up_fused) {
		ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP, 2 * inter, dim,
							 STAGE_MATMUL);
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE_FUSED,
			.in		   = {RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	} else {
		if (has_matmul_multi) {
			ops[i++] = mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, dim, inter,
										inter);
		} else {
			ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter, dim,
								 STAGE_MATMUL);
			ops[i++] =
				mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim, STAGE_MATMUL);
		}
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE,
			.in		   = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	}
	ops[i++] = mk_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim, inter, STAGE_MATMUL);
	ops[i++] = mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD);
	return i;
}

int recipe_append_moe_ffn(recipe_op *ops, int i, const struct model *m, uint8_t router_in_slot,
						  uint8_t experts_in_slot, uint8_t out_slot) {
	int dim		  = m->dim;
	int moe_inter = m->moe.moe_intermediate;

	if (m->moe.n_shared_experts > 0) {
		int sh_inter = moe_inter * m->moe.n_shared_experts;
		ops[i++]	 = (recipe_op){
			.kind	  = OP_MOE_SHARED,
			.in		  = {experts_in_slot, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_FFN_ACT,
			.w_idx	  = WIDX_NONE,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = sh_inter, .k = dim},
		};
	}

	ops[i++] = (recipe_op){
		.kind	  = OP_MOE_ROUTER,
		.in		  = {router_in_slot, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	  = RECIPE_SLOT_ROUTER_IDS,
		.w_idx	  = WIDX_FFN_GATE_INP,
		.stage	  = STAGE_MATMUL,
		.u.matmul = {.n = m->moe.n_experts, .k = dim},
	};

	ops[i++] = (recipe_op){
		.kind	  = OP_MOE_EXPERTS,
		.in		  = {experts_in_slot, RECIPE_SLOT_ROUTER_IDS, RECIPE_SLOT_ROUTER_W},
		.out	  = out_slot,
		.w_idx	  = WIDX_NONE,
		.stage	  = STAGE_MATMUL,
		.u.matmul = {.n = dim, .k = moe_inter},
	};

	return i;
}

void recipe_build_pre_ops(model_recipe *r, const model *m) {
	int n = 1;
	if (m->arch_info->has_scale_embeddings)
		n++;
	if (m->has_per_layer_embeddings)
		n++;
	recipe_op *ops = xcalloc((size_t)n, sizeof(recipe_op));
	int		   i   = 0;

	ops[i++] = (recipe_op){
		.kind  = OP_EMBD_LOOKUP,
		.in	   = {RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_X,
		.w_idx = WIDX_TOK_EMBD,
		.stage = STAGE_EMBD,
	};
	if (m->arch_info->has_scale_embeddings) {
		ops[i++] = (recipe_op){
			.kind  = OP_SCALE_EMBEDDINGS,
			.in	   = {RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_X,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_EMBD,
		};
	}
	if (m->has_per_layer_embeddings) {
		ops[i++] = (recipe_op){
			.kind  = OP_PLE_BUILD,
			.in	   = {RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_NONE,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_EMBD,
		};
	}
	r->pre_ops	 = ops;
	r->n_pre_ops = i;
}

void recipe_build_post_ops(model_recipe *r, const model *m) {
	const int	dim = m->dim;
	const float eps = m->norm_eps;
	int			cap = 5;
	recipe_op  *ops = xcalloc(cap, sizeof(recipe_op));
	int			i	= 0;

	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_OUTPUT_NORM, eps, STAGE_LOGITS_NORM);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_LOGITS, WIDX_OUTPUT_W, m->vocab_size, dim,
						 STAGE_LOGITS_MATMUL);

	ops[i++] = (recipe_op){
		.kind  = OP_LOGITS_READBACK,
		.in	   = {RECIPE_SLOT_LOGITS, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_LOGITS_READBACK,
	};

	if (m->final_logit_softcap > 0.0f) {
		ops[i++] = (recipe_op){
			.kind	   = OP_SOFTCAP,
			.in		   = {RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_NONE,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_LOGITS_READBACK,
			.u.softcap = {.cap = m->final_logit_softcap},
		};
	}

	r->post_ops	  = ops;
	r->n_post_ops = i;
}
