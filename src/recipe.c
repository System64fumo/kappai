#include "recipe.h"
#include "backend/backend.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "moe/moe_common.h"
#include "moe/moe_exec.h"
#include "moe/moe_stream.h"
#include "monitor.h"
#include "plan.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int moe_router_emit_ex(int E, int K, int use_softmax, int norm_topk, float routed_scale,
							  int n_group, int topk_group, float *logits, const float *bias,
							  float *scores_scratch, int *ids_out, float *w_out);

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
#define OP_BACKEND(be, field) op_backend_pick((be), ((be)->field != NULL), #field)
static inline backend *op_backend_fallback(backend *a, const char *op_name);
static inline backend *op_backend_pick(backend *a, int has_field, const char *op_name) {
	if (has_field)
		return a;
	if (backend_has_cap(a, BCAP_IS_HOST))
		return a;
	return op_backend_fallback(a, op_name);
}
static stage		 matmul_substage(uint8_t w_idx);
static void			 profile_weight_bytes(profile *prof, stage s, uint32_t w_type, int rows, int k);
static status_code	 exec_op_batch(const recipe_op *op, struct model *m, struct kvcache *cache,
								   struct compute_scratch *s, batch_scratch *bs, int pos_start,
								   int n_tokens, int li, int flash_attn);
static buffer		*batch_slot(batch_scratch *bs, uint8_t slot);
static buffer		 batch_row_view(const buffer *whole, int row, int row_elems);
static inline float *batch_buf_ptr(const buffer *b);
static status_code	 ple_build_batch(exec_ctx *ctx);
static buffer		 bs_cached_view(float_buf *fb, backend *owner, size_t n_elems);
static status_code	 ple_proj_inject_batch(exec_ctx *ctx);
static status_code	 scale_inplace_fallback(struct compute_scratch *s, backend *sync_be, buffer *xb,
											int n, float scale);
static status_code	 matmul_multi_batch_body(exec_ctx *ctx);
static void			 moe_batch_expert_chunk(int begin, int end, int tid, void *ctx);
static status_code	 batch_copy_k_to_v(model *m, batch_scratch *bs, int li, int n_tokens);
static uint32_t		 op_batch_slot_mask(const recipe_op *op);
static status_code	 ffn_activate_fused_device_fallback(exec_ctx *ctx, backend *a, int n, int act,
														int fused_stride, int n_rows);
static void			 recipe_optimize(model_recipe *r, const struct model *m);

static inline buffer *exec_slot(const exec_ctx *ctx, uint8_t idx) {
	return batch_slot(ctx->bs, idx);
}

static inline int slot_is_host_resident(const buffer *b) {
	if (!b->handle)
		return 1;
	return b->owner && backend_has_cap(b->owner, BCAP_IS_HOST);
}

static const float *recipe_slot_read_f32(const exec_ctx *ctx, uint8_t idx, float_buf *stage,
										 int n) {
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

static float *recipe_slot_write_stage(const exec_ctx *ctx, uint8_t idx, float_buf *stage, int n) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return NULL;
	if (slot_is_host_resident(b))
		return (float *)(b->handle ? (char *)b->handle + b->offset : b->host_ptr);
	return float_buf_ensure_nocopy(stage, (size_t)n, 64);
}

static status_code recipe_slot_write_commit(const exec_ctx *ctx, uint8_t idx, const float *staged,
											int n) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return ERR_INVALID_ARG;
	if (slot_is_host_resident(b)) {
		float *dst = (float *)(b->handle ? (char *)b->handle + b->offset : (char *)b->host_ptr);
		if (dst && (const float *)dst != staged)
			memcpy(dst, staged, (size_t)n * sizeof(float));
		return OK;
	}
	return b->owner->buffer_write_f32(b->owner, b, staged, n);
}

typedef struct {
	int n_calls;
	int rows;
} exec_segments;

static inline exec_segments exec_segments_of(const exec_ctx *ctx) {
	if (ctx->rows.pos)
		return (exec_segments){ctx->rows.n_rows, 1};
	return (exec_segments){1, ctx->rows.n_rows};
}

static inline int exec_segment_pos(const exec_ctx *ctx, int call) {
	return ctx->rows.pos ? ctx->rows.pos[call] : ctx->rows.pos_start;
}

static inline int exec_row_pos(const exec_ctx *ctx, int row) {
	return ctx->rows.pos ? ctx->rows.pos[row] : ctx->rows.pos_start + row;
}

static inline kvcache *exec_segment_cache(const exec_ctx *ctx, int call) {
	return ctx->rows.caches ? ctx->rows.caches[call] : ctx->rows.cache;
}

static inline backend *exec_active_backend(const exec_ctx *ctx) {
	if (ctx->s && ctx->s->active_backend)
		return ctx->s->active_backend;
	return ctx->m->backend;
}

static inline backend *exec_layer_backend(const exec_ctx *ctx) {
	if (ctx->layer_be)
		return ctx->layer_be;
	if (ctx->li >= 0 && ctx->m && ctx->m->mixed_backend_mode)
		return model_layer_backend(ctx->m, ctx->li);
	return exec_active_backend(ctx);
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

static const recipe_builder_fn *recipe_lookup(const char *arch_gguf_name) {
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

void recipe_coalesce_matmul_runs(recipe_op *ops, int n_ops) {
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

const recipe_op *recipe_layer_ops(const model_recipe *r, int li) {
	return r->per_layer_ops ? &r->per_layer_ops[(size_t)li * r->layer.n_ops] : r->layer.ops;
}

static int recipe_uses_op(const model_recipe *r, const struct model *m, op_kind kind) {
	const int n_variants = r->per_layer_ops ? m->n_layers : 1;
	for (int li = 0; li < n_variants; li++) {
		const recipe_op *ops = recipe_layer_ops(r, li);
		for (int i = 0; i < r->layer.n_ops; i++)
			if (ops[i].kind == kind)
				return 1;
	}
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

static status_code recipe_require_native(const model_recipe *r, const struct model *m,
										 op_kind kind) {
	if (!recipe_uses_op(r, m, kind))
		return OK;
	const char *missing = recipe_op_missing_fn(m->backend, kind);
	return missing ? recipe_require_capability(m, m->backend, missing, 0, 0, 0) : OK;
}

static status_code recipe_check_backend_capabilities(const model_recipe *r, const struct model *m) {
	const backend *a = m->backend;
	status_code	   s;

	if (recipe_uses_op(r, m, OP_ATTENTION_MLA)) {
		backend *host = backend_host();
		s = recipe_require_capability(m, a, "attention_mla", a->attention_mla != NULL,
									  host && host != a && host->attention_mla != NULL, 1);
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
	static const op_kind NATIVE_REQUIRED_OPS[] = {OP_HC_INIT,  OP_HC_PRE, OP_HC_POST,
												  OP_HC_FINAL, OP_KDA,	  OP_MLA_ABSORBED};
	for (size_t i = 0; i < ARRAY_LEN(NATIVE_REQUIRED_OPS); i++) {
		s = recipe_require_native(r, m, NATIVE_REQUIRED_OPS[i]);
		if (s != OK)
			return s;
	}
	if (m->moe.n_experts > 0 && !a->matmul_thread_local &&
		!(backend_has_cap(a, BCAP_MOE_EXPERT_RESIDENT) && a->moe_experts_batch &&
		  a->moe_expert_ffn && a->buffer_alloc_from_host)) {
		backend *host = backend_host();
		s = recipe_require_capability(m, a, "matmul_thread_local", 0,
									  host && host != a && host->matmul_thread_local != NULL, 1);
		if (s != OK)
			return s;
	}

	return OK;
}

typedef struct {
	const char *fn_name;
	size_t		slot0;
	size_t		slot1;
	uint32_t	batch_extra_mask;
	int8_t		reads[4];
	int8_t		writes[4];
	uint8_t		inplace_out_none;
} op_requirement;

#define OP_NO_SLOT ((size_t)-1)
#define OP_FN_SLOT(fn) offsetof(backend, fn)

static const op_requirement OP_REQUIREMENTS[OP_KIND_COUNT] =
	{
		[OP_NONE]		 = {},
		[OP_PLE_BUILD]	 = {},
		[OP_EMBD_LOOKUP] = {.fn_name = "embd_lookup",
							.slot0	 = OP_FN_SLOT(embd_lookup),
							.slot1	 = OP_NO_SLOT,
							.writes	 = {OP_TOUCH_OUT}},
		[OP_SCALE_EMBEDDINGS] =
			{
				.fn_name		  = "scale_inplace",
				.slot0			  = OP_FN_SLOT(scale_inplace),
				.slot1			  = OP_NO_SLOT,
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
				.inplace_out_none = 1,
			},
		[OP_SCALE] =
			{
				.fn_name		  = "scale_inplace",
				.slot0			  = OP_FN_SLOT(scale_inplace),
				.slot1			  = OP_NO_SLOT,
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
				.inplace_out_none = 1,
			},
		[OP_RMSNORM] =
			{
				.fn_name = "rmsnorm",
				.slot0	 = OP_FN_SLOT(rmsnorm),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_RMSNORM_PER_HEAD] =
			{
				.fn_name		  = "rmsnorm_per_head",
				.slot0			  = OP_FN_SLOT(rmsnorm_per_head),
				.slot1			  = OP_NO_SLOT,
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
				.inplace_out_none = 1,
			},
		[OP_RMSNORM_NOWEIGHT] =
			{
				.fn_name		  = "rmsnorm_noweight",
				.slot0			  = OP_FN_SLOT(rmsnorm_noweight),
				.slot1			  = OP_NO_SLOT,
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
				.inplace_out_none = 1,
			},
		[OP_RMSNORM_ADD] =
			{
				.fn_name = "rmsnorm_add",
				.slot0	 = OP_FN_SLOT(rmsnorm_add),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_MATMUL] =
			{
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_K) | (1u << RECIPE_SLOT_V),
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT, OP_TOUCH_OUT_GATE, OP_TOUCH_OUT_UP},
			},
		[OP_MATMUL_MULTI] =
			{
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_K) | (1u << RECIPE_SLOT_V),
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT, OP_TOUCH_OUT1, OP_TOUCH_OUT2_WQ},
			},
		[OP_MATMUL_RESIDUAL] =
			{
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_RESID_TMP),
				.reads			  = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes			  = {OP_TOUCH_OUT},
			},
		[OP_MATMUL_FFN_DOWN] =
			{
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_FFN_ACT),
				.reads			  = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes			  = {OP_TOUCH_OUT},
			},
		[OP_MLA_Q_PROJ] =
			{
				.fn_name = "matmul",
				.slot0	 = OP_FN_SLOT(matmul),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_SLOT(RECIPE_SLOT_FFN_GATE)},
				.writes	 = {OP_TOUCH_OUT, OP_TOUCH_SLOT(RECIPE_SLOT_FFN_GATE)},
			},
		[OP_MLA_KV_PROJ] =
			{
				.fn_name = "matmul",
				.slot0	 = OP_FN_SLOT(matmul),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_SLOT(RECIPE_SLOT_FFN_UP)},
				.writes	 = {OP_TOUCH_SLOT(RECIPE_SLOT_FFN_UP)},
			},
		[OP_MLA_QKV_PROJ_FUSED] =
			{
				.fn_name = "matmul",
				.slot0	 = OP_FN_SLOT(matmul),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_SLOT(RECIPE_SLOT_FFN_GATE),
							OP_TOUCH_SLOT(RECIPE_SLOT_FFN_UP)},
				.writes	 = {OP_TOUCH_OUT, OP_TOUCH_SLOT(RECIPE_SLOT_FFN_GATE),
							OP_TOUCH_SLOT(RECIPE_SLOT_FFN_UP)},
			},
		[OP_MOE_ROUTER] =
			{
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_XB) | (1u << RECIPE_SLOT_XB2) |
									(1u << RECIPE_SLOT_FFN_ACT) | (1u << RECIPE_SLOT_RESID_TMP),
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT, OP_TOUCH_SLOT(RECIPE_SLOT_ROUTER_W)},
			},
		[OP_ROPE] =
			{
				.fn_name		  = "rope",
				.slot0			  = OP_FN_SLOT(rope),
				.slot1			  = OP_NO_SLOT,
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
				.inplace_out_none = 1,
			},
		[OP_ROPE_QK_FUSED] =
			{
				.fn_name = "rope_qk",
				.slot0	 = OP_FN_SLOT(rope_qk),
				.slot1	 = OP_FN_SLOT(rope),
				.reads	 = {OP_TOUCH_SLOT(RECIPE_SLOT_Q), OP_TOUCH_SLOT(RECIPE_SLOT_K)},
				.writes	 = {OP_TOUCH_SLOT(RECIPE_SLOT_Q), OP_TOUCH_SLOT(RECIPE_SLOT_K)},
			},
		[OP_ROPE_EXT] =
			{
				.fn_name		  = "rope_ext",
				.slot0			  = OP_FN_SLOT(rope_ext),
				.slot1			  = OP_FN_SLOT(rope),
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
				.inplace_out_none = 1,
			},
		[OP_KV_PUT] =
			{
				.fn_name = "kv_put",
				.slot0	 = OP_FN_SLOT(kv_put),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
			},
		[OP_ATTENTION] =
			{
				.fn_name = "attention",
				.slot0	 = OP_FN_SLOT(attention),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_ATTENTION_MLA] =
			{
				.fn_name = "attention",
				.slot0	 = OP_FN_SLOT(attention),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_MLA_ABSORBED] =
			{
				.fn_name = "mla_absorbed",
				.slot0	 = OP_FN_SLOT(mla_absorbed),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_ADD] =
			{
				.fn_name = "add_inplace",
				.slot0	 = OP_FN_SLOT(add_inplace),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_IN0},
			},
		[OP_SWAP] =
			{
				.reads	= {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes = {OP_TOUCH_IN0, OP_TOUCH_IN1},
			},
		[OP_FFN_ACTIVATE] =
			{
				.fn_name = "ffn_activate",
				.slot0	 = OP_FN_SLOT(ffn_activate),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_FFN_ACTIVATE_EX] =
			{
				.fn_name = "ffn_activate_ex",
				.slot0	 = OP_FN_SLOT(ffn_activate_ex),
				.slot1	 = OP_FN_SLOT(ffn_activate),
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_FFN_ACTIVATE_FUSED] =
			{
				.fn_name = "ffn_activate",
				.slot0	 = OP_FN_SLOT(ffn_activate),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_SLOT(RECIPE_SLOT_FFN_GATE),
							OP_TOUCH_SLOT(RECIPE_SLOT_FFN_UP)},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_SOFTCAP] =
			{
				.fn_name		  = "softcap",
				.slot0			  = OP_FN_SLOT(softcap),
				.slot1			  = OP_NO_SLOT,
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
				.inplace_out_none = 1,
			},
		[OP_SPLIT_QGATE] =
			{
				.fn_name		  = "split_qgate",
				.slot0			  = OP_FN_SLOT(split_qgate),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_HYB_PROJ) |
									(1u << RECIPE_SLOT_Q) | (1u << RECIPE_SLOT_HYB_GATE),
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT, OP_TOUCH_SLOT(RECIPE_SLOT_HYB_GATE)},
			},
		[OP_PARTIAL_ROPE_QK] =
			{
				.fn_name		  = "partial_rope_qk",
				.slot0			  = OP_FN_SLOT(partial_rope_qk),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_Q) | (1u << RECIPE_SLOT_K),
				.reads			  = {OP_TOUCH_SLOT(RECIPE_SLOT_Q), OP_TOUCH_SLOT(RECIPE_SLOT_K)},
				.writes			  = {OP_TOUCH_SLOT(RECIPE_SLOT_Q), OP_TOUCH_SLOT(RECIPE_SLOT_K)},
			},
		[OP_ATTN_OUTPUT_GATE] =
			{
				.fn_name		  = "attn_output_gate",
				.slot0			  = OP_FN_SLOT(attn_output_gate),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_HYB_GATE),
				.reads			  = {OP_TOUCH_IN0, OP_TOUCH_SLOT(RECIPE_SLOT_HYB_GATE)},
				.writes			  = {OP_TOUCH_IN0},
			},
		[OP_GATED_DELTA_NET] =
			{
				.fn_name		  = "gated_delta_net",
				.slot0			  = OP_FN_SLOT(gated_delta_net),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_HYB_PROJ) | (1u << RECIPE_SLOT_HYB_GATE) |
									(1u << RECIPE_SLOT_HYB_ALPHA) | (1u << RECIPE_SLOT_HYB_BETA) |
									(1u << RECIPE_SLOT_XB2),
				.reads			  = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2,
									 OP_TOUCH_SLOT(RECIPE_SLOT_HYB_BETA)},
				.writes			  = {OP_TOUCH_OUT},
			},
		[OP_SHORTCONV] =
			{
				.reads	= {OP_TOUCH_IN0},
				.writes = {OP_TOUCH_OUT},
			},
		[OP_HC_INIT] =
			{
				.fn_name = "hc_init",
				.slot0	 = OP_FN_SLOT(hc_init),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_HC_PRE] =
			{
				.fn_name		  = "hc_pre",
				.slot0			  = OP_FN_SLOT(hc_pre),
				.slot1			  = OP_NO_SLOT,
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_SLOT(RECIPE_SLOT_HC_COEF), OP_TOUCH_OUT},
				.batch_extra_mask = (1u << RECIPE_SLOT_HC_COEF),
			},
		[OP_HC_POST] =
			{
				.fn_name		  = "hc_post",
				.slot0			  = OP_FN_SLOT(hc_post),
				.slot1			  = OP_NO_SLOT,
				.reads			  = {OP_TOUCH_IN0, OP_TOUCH_SLOT(RECIPE_SLOT_HC_COEF)},
				.writes			  = {OP_TOUCH_OUT},
				.batch_extra_mask = (1u << RECIPE_SLOT_HC_COEF),
			},
		[OP_HC_FINAL] =
			{
				.fn_name = "hc_final",
				.slot0	 = OP_FN_SLOT(hc_final),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_KDA] =
			{
				.fn_name = "kda",
				.slot0	 = OP_FN_SLOT(kda),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_LOGITS_READBACK] =
			{
				.reads = {OP_TOUCH_IN0, OP_TOUCH_IN1},
			},
		[OP_PLE_PROJ_INJECT] =
			{
				.reads	= {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes = {OP_TOUCH_IN0, OP_TOUCH_IN1},
			},
		[OP_MOE_SHARED] =
			{
				.fn_name		  = "moe_activate",
				.slot0			  = OP_FN_SLOT(moe_activate),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_XB) | (1u << RECIPE_SLOT_XB2) |
									(1u << RECIPE_SLOT_FFN_ACT) | (1u << RECIPE_SLOT_RESID_TMP),
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
			},
		[OP_MOE_EXPERTS] =
			{
				.fn_name		  = "moe_experts",
				.slot0			  = OP_FN_SLOT(moe_experts_batch),
				.slot1			  = OP_FN_SLOT(matmul_thread_local),
				.batch_extra_mask = (1u << RECIPE_SLOT_XB) | (1u << RECIPE_SLOT_XB2) |
									(1u << RECIPE_SLOT_FFN_ACT) | (1u << RECIPE_SLOT_RESID_TMP),
				.reads			  = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2},
				.writes			  = {OP_TOUCH_OUT},
			},
};

const char *recipe_op_missing_fn(const backend *a, op_kind k) {
	if ((int)k < 0 || k >= OP_KIND_COUNT)
		return NULL;
	const op_requirement *req = &OP_REQUIREMENTS[k];
	if (!req->fn_name)
		return NULL;
	const char *base = (const char *)a;
	if (*(const void *const *)(base + req->slot0))
		return NULL;
	if (req->slot1 != OP_NO_SLOT && *(const void *const *)(base + req->slot1))
		return NULL;
	return req->fn_name;
}

static void touch_push(uint8_t *list, int *n, uint8_t slot) {
	if (slot == RECIPE_SLOT_NONE || *n >= OP_TOUCH_MAX)
		return;
	for (int i = 0; i < *n; i++)
		if (list[i] == slot)
			return;
	list[(*n)++] = slot;
}

static uint8_t touch_resolve(int8_t t, const recipe_op *op, uint8_t inplace_out_none) {
	if (t < 0)
		return (uint8_t)(-t - 1);
	if (t <= OP_TOUCH_IN2)
		return op->in[t - 1];
	if (t == OP_TOUCH_OUT) {
		uint8_t slot = op->out;
		if (slot == RECIPE_SLOT_NONE && inplace_out_none)
			slot = op->in[0];
		return slot;
	}
	if (t == OP_TOUCH_OUT1) {
		if (op->w_idx == WIDX_GATE_UP)
			return RECIPE_SLOT_FFN_GATE;
		return (uint8_t)(op->out + 1);
	}
	if (t == OP_TOUCH_OUT2_WQ) {
		if (op->w_idx == WIDX_GATE_UP)
			return RECIPE_SLOT_FFN_UP;
		if (op->w_idx != WIDX_WQ)
			return RECIPE_SLOT_NONE;
		return (uint8_t)(op->out + 2);
	}
	if (op->out != RECIPE_SLOT_FFN_GATE_UP)
		return RECIPE_SLOT_NONE;
	if (t == OP_TOUCH_OUT_GATE)
		return RECIPE_SLOT_FFN_GATE;
	if (t == OP_TOUCH_OUT_UP)
		return RECIPE_SLOT_FFN_UP;
	return RECIPE_SLOT_NONE;
}

int recipe_op_touch(const recipe_op *op, uint8_t *reads, int *n_reads, uint8_t *writes,
					int *n_writes) {
	if (op->kind < 0 || op->kind >= OP_KIND_COUNT)
		return 0;
	const op_requirement *req = &OP_REQUIREMENTS[op->kind];
	for (int i = 0; i < 4 && req->reads[i]; i++)
		touch_push(reads, n_reads, touch_resolve(req->reads[i], op, req->inplace_out_none));
	for (int i = 0; i < 4 && req->writes[i]; i++)
		touch_push(writes, n_writes, touch_resolve(req->writes[i], op, req->inplace_out_none));
	return 1;
}

static int matmul_type_ok(const backend *a, uint32_t t) {
	return !a->matmul_type_native || a->matmul_type_native((backend *)a, t);
}

static int ple_build_dev_path_cached(const struct model *m, backend *a, int batched) {
	if (!m->mixed_backend_mode && a == m->backend && m->recipe)
		return batched ? m->recipe->ple_build_dev_ok_batch : m->recipe->ple_build_dev_ok;
	return ple_build_dev_path_ok(a, m, batched);
}

void recipe_fuse_output_scale(recipe_op *ops, int n_ops, const struct model *m, int li) {
	for (int j = 0; j < n_ops; j++) {
		recipe_op *op = &ops[j];
		if (op->kind != OP_SCALE)
			continue;
		uint8_t slot = op->in[0];
		float	sc	 = op->u.scale.scale;
		if (m->arch_info->has_layer_output_scale && li >= 0 && li < m->n_layers &&
			m->layers[li].layer_out_scale != 1.0f)
			sc = m->layers[li].layer_out_scale;
		for (int k = j - 1; k >= 0; k--) {
			recipe_op *prev = &ops[k];
			if (prev->kind == OP_NONE)
				continue;
			if (prev->kind == OP_RMSNORM_ADD && (prev->out == slot || prev->in[0] == slot)) {
				prev->u.rmsnorm.out_scale *= sc;
				op->kind = OP_NONE;
				break;
			}
			if (prev->kind == OP_PLE_PROJ_INJECT && prev->in[0] == slot) {
				if (m->has_per_layer_embeddings) {
					op->kind = OP_NONE;
					break;
				}
				continue;
			}
			if (prev->out == slot)
				break;
			int touches = 0;
			for (int q = 0; q < 3; q++)
				touches |= (prev->in[q] == slot);
			if (touches)
				break;
		}
	}
}

model_recipe *recipe_build(const struct model *m) {
	if (!m || !m->arch_info)
		return NULL;
	const recipe_builder_fn *fb = recipe_lookup(m->arch_info->gguf_name);
	if (!fb) {
		ERROR("recipe_build: no recipe builder registered for arch '%s'", m->arch_info->gguf_name);
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
		int max_inter = 0;
		int max_hd	  = 0;
		int max_kvh	  = 0;
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
		if (m->arch_info->is_mla) {
			if (m->mla.qk_head > max_hd)
				max_hd = m->mla.qk_head;
			if (m->n_heads > max_kvh)
				max_kvh = m->n_heads;
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
				if (op->kind == OP_MATMUL || op->kind == OP_MATMUL_RESIDUAL) {
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
		}
	}

	r->has_sliding_layers	  = model_has_sliding_layers(m);
	r->ple_build_dev_ok		  = ple_build_dev_path_ok(m->backend, m, 0);
	r->ple_build_dev_ok_batch = ple_build_dev_path_ok(m->backend, m, 1);

	recipe_optimize(r, m);
	plan_fuse_residuals(r, m);

	if (recipe_check_backend_capabilities(r, m) != OK) {
		recipe_free(r);
		return NULL;
	}

	plan_attach(r, m);
	backend_fallback_report();

	return r;
}

static void recipe_optimize(model_recipe *r, const struct model *m) {
	recipe_refine_matmul_stages(r->layer.ops, r->layer.n_ops);
	recipe_coalesce_matmul_runs(r->layer.ops, r->layer.n_ops);
	for (int i = 0; i < r->n_pre_ops; i++)
		recipe_refine_matmul_stages(&r->pre_ops[i], 1);
	for (int i = 0; i < r->n_post_ops; i++)
		recipe_refine_matmul_stages(&r->post_ops[i], 1);
	if (r->per_layer_ops && m->n_layers > 0 && r->layer.n_ops > 0) {
		for (int li = 0; li < m->n_layers; li++) {
			recipe_op *dst = &r->per_layer_ops[(size_t)li * r->layer.n_ops];
			recipe_fuse_output_scale(dst, r->layer.n_ops, m, li);
			recipe_refine_matmul_stages(dst, r->layer.n_ops);
			recipe_coalesce_matmul_runs(dst, r->layer.n_ops);
		}
	}
	recipe_batch_slot_mask(r, m);
}

void recipe_batch_slot_mask(model_recipe *r, const struct model *m) {
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
}

void recipe_free(model_recipe *r) {
	if (!r)
		return;
	free(r->pre_ops);
	free(r->layer.ops);
	free(r->post_ops);
	free(r->per_layer_ops);
	free(r->layer_ctx);
	plan_free(r->plan);
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

static inline weight_ref *exec_weight_ref(const exec_ctx *ctx, uint8_t w_idx) {
	if (w_idx == RECIPE_NO_WEIGHT || w_idx == WIDX_NONE)
		return (weight_ref *)&WEIGHT_REF_NONE;
	if (ctx->wtab_row) {
		weight_ref *w = ctx->wtab_row[w_idx];
		return w ? w : (weight_ref *)&WEIGHT_REF_NONE;
	}
	return resolve_weight_ref(ctx->m, ctx->li, w_idx);
}

static inline const buffer *exec_weight(const exec_ctx *ctx, uint8_t w_idx) {
	return &exec_weight_ref(ctx, w_idx)->buf;
}

static status_code copy_k_to_v_buffers(backend *a, buffer *kb, buffer *vb, int kv_out,
									   int kv_stride, int n_rows) {
	if (a->copy_buffer && kb->owner == vb->owner) {
		for (int row = 0; row < n_rows; row++) {
			buffer		krow = batch_row_view(kb, row, kv_stride);
			buffer		vrow = batch_row_view(vb, row, kv_stride);
			status_code st	 = a->copy_buffer(a, &krow, &vrow, kv_out);
			if (st != OK)
				return st;
		}
		return OK;
	}
	ensure_sync(a);
	if (!kb->host_ptr || !vb->host_ptr) {
		float	   *tmp = xmalloc((size_t)kv_out * sizeof(float));
		status_code st	= OK;
		for (int row = 0; row < n_rows && st == OK; row++) {
			buffer krow = batch_row_view(kb, row, kv_stride);
			buffer vrow = batch_row_view(vb, row, kv_stride);
			st			= kb->owner->buffer_read_f32(kb->owner, &krow, tmp, kv_out);
			if (st == OK)
				st = vb->owner->buffer_write_f32(vb->owner, &vrow, tmp, kv_out);
		}
		free(tmp);
		return st;
	}
	for (int row = 0; row < n_rows; row++) {
		const float *k =
			(const float *)((const char *)kb->host_ptr + kb->offset) + (size_t)row * kv_stride;
		float *v = (float *)((char *)vb->host_ptr + vb->offset) + (size_t)row * kv_stride;
		memcpy(v, k, (size_t)kv_out * sizeof(float));
	}
	return OK;
}

typedef struct {
	const weight_ref *w;
	buffer			 *y;
	int				  n;
} matmul_fallback_item;

static int matmul_op_uses_substage(op_kind k) {
	return k == OP_MATMUL || k == OP_MATMUL_MULTI || k == OP_MATMUL_RESIDUAL ||
		   k == OP_MATMUL_FFN_DOWN;
}

void recipe_refine_matmul_stages(recipe_op *ops, int n_ops) {
	for (int i = 0; i < n_ops; i++) {
		recipe_op *op = &ops[i];
		if (!matmul_op_uses_substage(op->kind) || op->stage != STAGE_MATMUL)
			continue;
		op->stage = matmul_substage(op->w_idx);
	}
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
	case WIDX_GATE_UP:
	case WIDX_SHEXP_GATE:
	case WIDX_SHEXP_UP:
		return STAGE_MATMUL_FFN_GATE_UP;
	case WIDX_DOWN:
	case WIDX_SHEXP_DOWN:
		return STAGE_MATMUL_FFN_DOWN;
	case WIDX_OUTPUT_W:
		return STAGE_LOGITS_MATMUL;
	default:
		return STAGE_MATMUL;
	}
}

static void profile_weight_bytes(profile *prof, stage s, uint32_t w_type, int rows, int k) {
	if (rows <= 0 || k <= 0)
		return;
	size_t rb = ggml_row_size(w_type, (size_t)k);
	profile_add_bytes(prof, s, (uint64_t)rb * (uint64_t)rows);
}

static int exec_matmul_run_coalesced_impl(const recipe_op *ops, int n_ops, model *m,
										  compute_scratch *s, int li, buffer *xb, buffer *ys[],
										  int batched, int n_rows, status_code *out_status) {
	backend *a = m->backend;
	if ((batched ? !a->matmul_multi_batch : !a->matmul_multi) || n_ops < 2) {
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
		y_list[i]	  = ys[i];
		n_out[i]	  = ops[i].u.matmul.n;
	}

	profile		 *prof = &s->prof;
	profile_scope ps   = profile_begin(prof, ops[0].stage);
	for (int i = 0; i < n; i++)
		profile_weight_bytes(prof, ops[0].stage, w_types[i], n_out[i], k);
	status_code st =
		batched ? a->matmul_multi_batch(a, w_list, w_types, xb, y_list, n_out, k, n, n_rows)
				: a->matmul_multi(a, w_list, w_types, xb, y_list, n_out, k, n);
	profile_end(prof, &ps);

	*out_status = st;
	return n;
}

static int exec_matmul_run_coalesced_batch(const recipe_op *ops, int n_ops, model *m,
										   compute_scratch *s, int li, buffer *xb, buffer *ys[],
										   int n_rows, status_code *out_status) {
	return exec_matmul_run_coalesced_impl(ops, n_ops, m, s, li, xb, ys, 1, n_rows, out_status);
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

#define OP_FALLBACK_SEEN_MAX 64

typedef struct {
	const backend *be;
	const char	  *op;
} op_fallback_seen;

static op_fallback_seen g_op_fallback_seen[OP_FALLBACK_SEEN_MAX];
static _Atomic int		g_op_fallback_n_seen;

static pthread_mutex_t g_op_fallback_mtx = PTHREAD_MUTEX_INITIALIZER;

static bool op_fallback_seen_before(const backend *a, const char *op_name, int n) {
	for (int i = 0; i < n && i < OP_FALLBACK_SEEN_MAX; i++)
		if (g_op_fallback_seen[i].be == a && strcmp(g_op_fallback_seen[i].op, op_name) == 0)
			return true;
	return false;
}

static void op_fallback_note(backend *a, const char *op_name) {
	if (op_fallback_seen_before(a, op_name, atomic_load(&g_op_fallback_n_seen)))
		return;
	pthread_mutex_lock(&g_op_fallback_mtx);
	int n = atomic_load(&g_op_fallback_n_seen);
	if (n >= OP_FALLBACK_SEEN_MAX || op_fallback_seen_before(a, op_name, n)) {
		pthread_mutex_unlock(&g_op_fallback_mtx);
		return;
	}
	g_op_fallback_seen[n].be = a;
	g_op_fallback_seen[n].op = op_name;
	atomic_store(&g_op_fallback_n_seen, n + 1);
	pthread_mutex_unlock(&g_op_fallback_mtx);
	backend_report_host_fallback(a, op_name, HFB_OP_NOT_NATIVE,
								 "backend has no native '%s'; running on the host fallback",
								 op_name);
}

static inline backend *op_backend_fallback(backend *a, const char *op_name) {
	op_fallback_note(a, op_name);
	ensure_sync(a);
	return op_host_backend();
}

static inline void backend_sync_rope(backend *t, backend *a) {
	if (t != a) {
		t->rope_neox  = a->rope_neox;
		t->rope_theta = a->rope_theta;
	}
}

typedef struct {
	struct model *m;
	backend		 *host;
	backend		 *dev;
	int			  total_ple;
	int			  dim;
	int			  n_embd_per_layer;
	int			  n_layers;
	float		  n_embd_sqrt;
	float		  inv_sqrt_ple;
	float		  eps;
} ple_build_plan;

static void ple_build_plan_fill(ple_build_plan *p, struct model *m, backend *dev) {
	p->m				= m;
	p->host				= NULL;
	p->dev				= dev;
	p->n_embd_per_layer = m->layer_dims.n_embd_per_layer;
	p->n_layers			= m->n_layers;
	p->total_ple		= p->n_embd_per_layer * p->n_layers;
	p->dim				= m->dim;
	p->n_embd_sqrt		= sqrtf((float)p->n_embd_per_layer);
	p->inv_sqrt_ple		= m->dim_sqrt > 0 ? 1.0f / m->dim_sqrt : 0.0f;
	p->eps				= m->norm_eps;
}

static int ple_build_dev_caps_ok(backend *a, const struct model *m, int batched) {
	int has_proj = batched ? (a->matmul_batch != NULL) : (a->matmul != NULL);
	int has_norm = batched ? (a->rmsnorm_batch != NULL) : (a->rmsnorm != NULL);
	return a->embd_lookup && has_proj && a->scale_inplace && has_norm && a->ple_combine &&
		   matmul_type_ok(a, m->layer_dims.per_layer_model_proj.type);
}

int ple_build_dev_path_ok(backend *a, const struct model *m, int batched) {
	return ple_build_dev_caps_ok(a, m, batched) && m->layer_dims.per_layer_model_proj.buf.handle &&
		   m->layer_dims.per_layer_tok_embd.buf.handle;
}

int ple_device_path_possible(const struct model *m) {
	return !m->mixed_backend_mode && m->backend && ple_build_dev_caps_ok(m->backend, m, 0) &&
		   ple_build_dev_caps_ok(m->backend, m, 1);
}

static status_code ple_host_decode_row(const ple_build_plan *p, int token, float *ple_row) {
	size_t row_stride	= ggml_row_size(p->m->layer_dims.per_layer_tok_embd.type, p->total_ple);
	const uint8_t *embd = (const uint8_t *)p->m->layer_dims.per_layer_tok_embd.host_ptr +
						  ((size_t)token * row_stride);
	backend		  *dq	= (p->dev && p->dev->dequant_row) ? p->dev : NULL;
	if (!dq) {
		ple_build_plan *mp = (ple_build_plan *)p;
		if (!mp->host)
			mp->host = backend_host();
		dq = mp->host;
	}
	if (!dq || !dq->dequant_row)
		return ERR_UNSUPPORTED;
	status_code st =
		dq->dequant_row(dq, p->m->layer_dims.per_layer_tok_embd.type, embd, p->total_ple, ple_row);
	if (st != OK)
		return st;
	for (int i = 0; i < p->total_ple; i++)
		ple_row[i] *= p->n_embd_sqrt;
	return OK;
}

static status_code ple_norm_layers(backend *a, const buffer *norm_w, buffer *x, int n_layers,
								   int slice_elems, float eps) {
	if (a->rmsnorm_batch && n_layers > 1)
		return a->rmsnorm_batch(a, x, norm_w, x, slice_elems, eps, n_layers);
	for (int l = 0; l < n_layers; l++) {
		buffer xs	   = *x;
		xs.offset	   = x->offset + (size_t)l * slice_elems * sizeof(float);
		status_code st = a->rmsnorm(a, &xs, norm_w, &xs, slice_elems, eps);
		if (st != OK)
			return st;
	}
	return OK;
}

static void ple_host_project_row(const ple_build_plan *p, const float *inp_row, float *proj_row) {
	host_matmul_generic(p->m->layer_dims.per_layer_model_proj.host_ptr,
						p->m->layer_dims.per_layer_model_proj.type, inp_row, proj_row, p->total_ple,
						p->dim);
}

static void ple_host_finish(const ple_build_plan *p, float *ple, float *proj, int n_rows) {
	ple_build_plan *mp = (ple_build_plan *)p;
	if (!mp->host)
		mp->host = backend_host();
	backend *host	 = mp->host;
	buffer proj_view = buffer_host_view(host, proj, (size_t)n_rows * p->total_ple * sizeof(float));
	host->scale_inplace(host, &proj_view, p->inv_sqrt_ple, n_rows * p->total_ple);
	buffer w_view = buffer_host_view(host, (void *)p->m->layer_dims.per_layer_proj_norm_w.host_ptr,
									 (size_t)p->n_embd_per_layer * sizeof(float));
	for (int row = 0; row < n_rows; row++) {
		buffer row_buf = buffer_host_view(host, proj + (size_t)row * p->total_ple,
										  (size_t)p->total_ple * sizeof(float));
		ple_norm_layers(host, &w_view, &row_buf, p->n_layers, p->n_embd_per_layer, p->eps);
	}
	buffer ple_view	  = buffer_host_view(host, ple, (size_t)n_rows * p->total_ple * sizeof(float));
	buffer proj_view2 = buffer_host_view(host, proj, (size_t)n_rows * p->total_ple * sizeof(float));
	host->ple_combine(host, &ple_view, &proj_view2, n_rows * p->total_ple, 0.70710678118654752f);
}

typedef struct {
	struct model			   *m;
	const struct layer_weights *L;
	int							li;
	int							n_embd_per_layer;
	int							total_ple;
	int							inj_dim;
	float						eps;
	const buffer			   *gate_w;
	const buffer			   *proj_w;
	const buffer			   *post_norm_w;
	uint32_t					gate_type;
	uint32_t					proj_type;
} ple_inject_plan;

static int ple_inject_plan_fill(ple_inject_plan *p, struct model *m, int li) {
	if (li < 0 || li >= m->n_layers)
		return 0;
	p->m				= m;
	p->L				= &m->layers[li];
	p->li				= li;
	p->n_embd_per_layer = m->layer_dims.n_embd_per_layer;
	p->total_ple		= p->n_embd_per_layer * m->n_layers;
	p->inj_dim			= m->dim;
	p->eps				= m->norm_eps;
	p->gate_w			= &p->L->ple_inp_gate_w.buf;
	p->proj_w			= &p->L->ple_proj_w.buf;
	p->post_norm_w		= &p->L->ple_post_norm_w.buf;
	p->gate_type		= p->L->ple_inp_gate_w.type;
	p->proj_type		= p->L->ple_proj_w.type;
	return 1;
}

static status_code op_ffn_activate_ex(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps;
	status_code	  st;
	int			  intermediate = ctx->op->u.matmul.n;
	int			  activation   = ctx->op->u.ffn_act.activation;
	ps						   = profile_begin(prof, ctx->op->stage);
	st = a->ffn_activate_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
							   exec_slot(ctx, ctx->op->out), intermediate, activation,
							   ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st;
}

static status_code op_rope_ext(exec_ctx *ctx) {
	const recipe_op		   *op	 = ctx->op;
	struct model		   *m	 = ctx->m;
	struct compute_scratch *s	 = ctx->s;
	int						li	 = ctx->li;
	backend				   *a	 = exec_layer_backend(ctx);
	profile				   *prof = &ctx->s->prof;
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
	int			  pos			 = ctx->rows.pos_start;
	int			  n_applications = ctx->rows.n_rows;
	int			  row_stride = (op->in[0] == RECIPE_SLOT_Q) ? lc->q_row_stride : lc->kv_row_stride;

	if (a->rope_ext_batch && a->rope_ext) {
		buffer		 *whole = batch_slot(ctx->bs, op->in[0]);
		exec_segments seg	= exec_segments_of(ctx);
		for (int c = 0; c < seg.n_calls && st == OK; c++) {
			buffer view = batch_row_view(whole, c * seg.rows, row_stride);
			st = a->rope_ext_batch(a, &view, n_heads, head_dim, exec_segment_pos(ctx, c), rope_cos,
								   rope_sin, freq_factors, seg.rows);
		}
		profile_end(prof, &ps);
		return st;
	}

	backend *t = a->rope_ext ? a : OP_BACKEND(a, rope);
	backend_sync_rope(t, a);
	backend *t2 = NULL;
	if (!t->rope_ext) {
		t2 = OP_BACKEND(a, rope);
		backend_sync_rope(t2, a);
	}

	for (int r = 0; r < n_applications; r++) {
		int	   row_pos = ctx->rows.pos ? ctx->rows.pos[r] : pos + r;
		buffer vec	   = batch_row_view(batch_slot(ctx->bs, op->in[0]), r, row_stride);
		if (t->rope_ext) {
			st = t->rope_ext(t, &vec, n_heads, head_dim, row_pos, rope_cos, rope_sin, freq_factors);
		} else {
			st = t2->rope(t2, &vec, n_heads, head_dim, row_pos, rope_cos, rope_sin);
		}
		if (st != OK)
			break;
	}
	profile_end(prof, &ps);
	return st;
}

typedef status_code (*op_handler)(exec_ctx *ctx);

static const op_handler g_op_dispatch[OP_KIND_COUNT];

static status_code scale_inplace_fallback(struct compute_scratch *s, backend *sync_be, buffer *xb,
										  int n, float scale) {
	ensure_sync(sync_be);
	compute_small_host_ensure(s, n, 0, 0);
	float	   *xf	  = s->xf_host;
	backend	   *owner = xb->owner;
	status_code st	  = owner->buffer_read_f32(owner, xb, xf, n);
	if (st == OK) {
		for (int i = 0; i < n; i++)
			xf[i] *= scale;
		st = owner->buffer_write_f32(owner, xb, xf, n);
	}
	return st;
}

static status_code op_rmsnorm(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	const buffer *w	   = exec_weight(ctx, ctx->op->w_idx);
	status_code	  st;
	st = a->rmsnorm_batch(a, exec_slot(ctx, ctx->op->in[0]), w, exec_slot(ctx, ctx->op->out),
						  ctx->m->dim, ctx->op->u.rmsnorm.eps, ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st;
}

static status_code op_matmul(exec_ctx *ctx) {
	const recipe_op *op = ctx->op;
	int				 li = ctx->li;
	if (op->w_idx == WIDX_WV && li >= 0 && li < ctx->m->n_layers &&
		!ctx->m->recipe->layer_ctx[li].has_own_v) {
		return batch_copy_k_to_v(ctx->m, ctx->bs, li, ctx->rows.n_rows);
	}
	weight_ref	 *wref = exec_weight_ref(ctx, op->w_idx);
	int			  n = op->u.matmul.n, k = op->u.matmul.k;
	profile_scope ps = profile_begin(&ctx->s->prof, op->stage);
	profile_weight_bytes(&ctx->s->prof, op->stage, wref->type, n, k);
	status_code st;
	backend	   *a = exec_layer_backend(ctx);
	st			  = a->matmul_batch(a, &wref->buf, wref->type, exec_slot(ctx, op->in[0]),
									exec_slot(ctx, op->out), n, k, ctx->rows.n_rows);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code op_matmul_residual(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	weight_ref	 *wref = exec_weight_ref(ctx, ctx->op->w_idx);
	const buffer *w	   = &wref->buf;
	uint32_t	  wt   = wref->type;
	int			  n	   = ctx->op->u.matmul.n;
	int			  k	   = ctx->op->u.matmul.k;
	profile_weight_bytes(prof, ctx->op->stage, wt, n, k);
	buffer *tmp = batch_slot(ctx->bs, RECIPE_SLOT_RESID_TMP);
	st = a->matmul_batch(a, w, wt, exec_slot(ctx, ctx->op->in[0]), tmp, n, k, ctx->rows.n_rows);
	if (st == OK)
		st = a->add_batch(a, exec_slot(ctx, ctx->op->out), tmp, n, ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st;
}

static status_code ffn_activate_ex_rows(backend *t, const float *fused, float *out, int n, int act,
										int n_rows, int fused_stride);

static status_code op_matmul_ffn_down(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps;
	status_code	  st;
	weight_ref	 *wref = exec_weight_ref(ctx, ctx->op->w_idx);
	const buffer *w	   = &wref->buf;
	uint32_t	  wt   = wref->type;
	int			  n	   = ctx->op->u.matmul_ffn_down.n;
	int			  k	   = ctx->op->u.matmul_ffn_down.k;
	int			  act  = ctx->op->u.matmul_ffn_down.activation;
	if (n == 0)
		n = ctx->m->dim;
	profile_weight_bytes(&ctx->s->prof, ctx->op->stage, wt, n, k);
	ps = profile_begin(prof, ctx->op->stage);
	int fused_batch =
		(a->matmul_ffn_down_batch && backend_has_cap(a, BCAP_MATMUL_FFN_DOWN)) ? 1 : 0;
	if (fused_batch) {
		st = a->matmul_ffn_down_batch(a, w, wt, exec_slot(ctx, ctx->op->in[0]),
									  exec_slot(ctx, ctx->op->in[1]), exec_slot(ctx, ctx->op->out),
									  n, k, act, ctx->rows.n_rows);
	} else {
		st = a->ffn_activate_batch(
			a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
			batch_slot(ctx->bs, RECIPE_SLOT_FFN_ACT), k, act, ctx->rows.n_rows);
		if (st == OK)
			st = a->matmul_batch(a, w, wt, batch_slot(ctx->bs, RECIPE_SLOT_FFN_ACT),
								 exec_slot(ctx, ctx->op->out), n, k, ctx->rows.n_rows);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_rope(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	int			  n_heads  = ctx->op->u.rope.n_heads;
	int			  head_dim = ctx->op->u.rope.head_dim;
	a->rope_neox		   = ctx->op->u.rope.rope_neox;
	exec_segments seg	   = exec_segments_of(ctx);
	buffer		 *whole	   = exec_slot(ctx, ctx->op->in[0]);
	st					   = OK;
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer view = batch_row_view(whole, c * seg.rows, n_heads * head_dim);
		st = a->rope_batch(a, &view, n_heads, head_dim, exec_segment_pos(ctx, c), ctx->s->rope_cos,
						   ctx->s->rope_sin, seg.rows);
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
	exec_segments seg		 = exec_segments_of(ctx);
	buffer		 *q_all		 = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *k_all		 = exec_slot(ctx, ctx->op->in[1]);
	st						 = OK;
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer q_view = batch_row_view(q_all, c * seg.rows, n_heads * head_dim);
		buffer k_view = batch_row_view(k_all, c * seg.rows, n_kv_heads * head_dim);
		st = a->rope_qk_batch(a, &q_view, &k_view, n_heads, n_kv_heads, head_dim,
							  exec_segment_pos(ctx, c), ctx->s->rope_cos, ctx->s->rope_sin,
							  seg.rows);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_kv_put(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st;
	int			  kv_row_stride = ctx->m->recipe->layer_ctx[ctx->li].kv_row_stride;
	exec_segments seg			= exec_segments_of(ctx);
	buffer		 *k_all			= batch_slot(ctx->bs, ctx->op->in[0]);
	buffer		 *v_all			= batch_slot(ctx->bs, ctx->op->in[1]);
	for (int c = 0; c < seg.n_calls; c++) {
		buffer k_view = batch_row_view(k_all, c * seg.rows, kv_row_stride);
		buffer v_view = batch_row_view(v_all, c * seg.rows, kv_row_stride);
		st = kvcache_put_batch(exec_segment_cache(ctx, c), ctx->m, ctx->li,
							   exec_segment_pos(ctx, c), &k_view, &v_view, kv_row_stride, seg.rows);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code op_attention(exec_ctx *ctx) {
	backend *a				   = exec_layer_backend(ctx);
	profile *prof			   = &ctx->s->prof;
	int		 n_heads		   = ctx->op->u.attention.n_heads;
	int		 n_kv_heads		   = ctx->op->u.attention.n_kv_heads;
	int		 head_dim		   = ctx->op->u.attention.head_dim;
	int		 n_ctx			   = ctx->rows.cache->n_ctx;
	float	 scale			   = ctx->op->u.attention.scale;
	int		 sliding_window	   = ctx->op->u.attention.sliding_window;
	int		 n_kv_heads_active = ctx->op->u.attention.n_kv_heads_active;
	int		 kv_layer =
		ctx->m->arch_info->has_variable_layer_dims ? ctx->op->u.attention.kv_layer : ctx->li;

	int use_swa			= (sliding_window > 0) && (ctx->li >= 0) &&
						  model_layer_is_sliding(ctx->m, ctx->li) && a->attention_swa_batch != NULL;
	profile_scope ps	= profile_begin(prof, ctx->op->stage);
	exec_segments seg	= exec_segments_of(ctx);
	buffer		 *q_all = batch_slot(ctx->bs, ctx->op->in[0]);
	buffer		 *o_all = batch_slot(ctx->bs, ctx->op->out);
	status_code	  st	= OK;
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer	q_view = batch_row_view(q_all, c * seg.rows, n_heads * head_dim);
		buffer	o_view = batch_row_view(o_all, c * seg.rows, n_heads * head_dim);
		buffer *kb;
		buffer *vb;
		int		kv_slot;
		kvcache_attn_view(exec_segment_cache(ctx, c), ctx->m, ctx->li, kv_layer, &kb, &vb,
						  &kv_slot);
		if (use_swa)
			st = a->attention_swa_batch(a, &q_view, kb, vb, &o_view, kv_slot,
										exec_segment_pos(ctx, c), n_heads, n_kv_heads, head_dim,
										n_ctx, ctx->flash_attn, scale, sliding_window,
										n_kv_heads_active, seg.rows);
		else
			st = a->attention_batch(a, &q_view, kb, vb, &o_view, kv_slot, exec_segment_pos(ctx, c),
									n_heads, n_kv_heads, head_dim, n_ctx, ctx->flash_attn, scale,
									n_kv_heads_active, seg.rows);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_add(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	const int	  dim = ctx->m->dim;
	st = a->add_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]), dim,
					  ctx->rows.n_rows);
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
	int			  n	  = ctx->op->u.ffn_act.n;
	int			  act = ctx->op->u.ffn_act.activation;
	st = a->ffn_activate_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
							   exec_slot(ctx, ctx->op->out), n, act, ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st;
}

static status_code op_ffn_activate_fused(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	int			  n			   = ctx->op->u.ffn_act.n;
	int			  act		   = ctx->op->u.ffn_act.activation;
	const int	  fused_stride = 2 * n;
	int			  n_rows	   = ctx->rows.n_rows;

	if (a->ffn_activate_fused_batch) {
		st = a->ffn_activate_fused_batch(a, batch_slot(ctx->bs, ctx->op->in[0]),
										 batch_slot(ctx->bs, ctx->op->out), n, act, n_rows);
		if (st != ERR_UNSUPPORTED) {
			profile_end(prof, &ps);
			return st;
		}
	}
	int dev_fallback = !backend_has_cap(a, BCAP_IS_HOST);
	if (dev_fallback) {
		backend_report_host_fallback(a, "ffn_activate_fused", HFB_BATCH_DESIGN,
									 "batch prefill stages gate/up activation through host memory");
		st = ffn_activate_fused_device_fallback(ctx, a, n, act, fused_stride, n_rows);
		return st;
	}
	const float *fused = batch_buf_ptr(batch_slot(ctx->bs, ctx->op->in[0]));
	float		*out   = batch_buf_ptr(batch_slot(ctx->bs, ctx->op->out));
	backend		*t	   = op_host_backend();
	if (!t->ffn_activate_ex)
		return ERR_UNSUPPORTED;
	st = ffn_activate_ex_rows(t, fused, out, n, act, ctx->rows.n_rows, fused_stride);
	profile_end(prof, &ps);
	return st;
}

static status_code op_softcap(exec_ctx *ctx) {
	float cap = ctx->op->u.softcap.cap;
	if (cap <= 0.0f)
		return OK;
	backend		 *a		= exec_layer_backend(ctx);
	profile		 *prof	= &ctx->s->prof;
	backend		 *t		= OP_BACKEND(a, softcap);
	buffer		 *lb	= exec_slot(ctx, RECIPE_SLOT_LOGITS);
	int			  total = ctx->m->vocab_size * ctx->rows.n_rows;
	profile_scope ps	= profile_begin(prof, ctx->op->stage);
	status_code	  st	= t->softcap(t, lb, cap, total);
	profile_end(prof, &ps);
	return st;
}

static status_code op_logits_readback(exec_ctx *ctx) {
	(void)ctx;
	return OK;
}

typedef status_code (*batch_row_fn)(buffer *row, void *ud);

static status_code batch_row_each(exec_ctx *ctx, uint8_t slot, int row_elems, batch_row_fn fn,
								  void *ud) {
	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer		rowb = batch_row_view(batch_slot(ctx->bs, slot), row, row_elems);
		status_code st	 = fn(&rowb, ud);
		if (st != OK)
			return st;
	}
	return OK;
}

typedef struct {
	backend		 *t;
	const buffer *w;
	int			  n_heads;
	int			  head_dim;
	float		  eps;
} rmsnorm_ph_row_ud;

static status_code rmsnorm_ph_row_fn(buffer *rowb, void *v) {
	rmsnorm_ph_row_ud *u = v;
	return u->t->rmsnorm_per_head(u->t, rowb, u->w, rowb, u->n_heads, u->head_dim, u->eps);
}

typedef struct {
	backend *t;
	float	 eps;
	int		 n;
} rmsnorm_nw_row_ud;

static status_code rmsnorm_nw_row_fn(buffer *rowb, void *v) {
	rmsnorm_nw_row_ud *u = v;
	return u->t->rmsnorm_noweight(u->t, rowb, rowb, u->n, u->eps);
}

static status_code op_rmsnorm_per_head(exec_ctx *ctx) {
	backend				  *a	= exec_layer_backend(ctx);
	profile				  *prof = &ctx->s->prof;
	profile_scope		   ps	= profile_begin(prof, ctx->op->stage);
	status_code			   st	= OK;
	const layer_ctx_entry *lc	= &ctx->m->recipe->layer_ctx[ctx->li];
	const buffer		  *w	= exec_weight(ctx, ctx->op->w_idx);
	int n_heads	   = (ctx->op->w_idx == WIDX_ATTN_Q_NORM) ? ctx->m->n_heads : lc->n_kv_heads;
	int row_stride = (ctx->op->w_idx == WIDX_ATTN_Q_NORM) ? lc->q_row_stride : lc->kv_row_stride;
	backend *t	   = OP_BACKEND(a, rmsnorm_per_head);
	if (t->rmsnorm_per_head_batch) {
		st = t->rmsnorm_per_head_batch(t, exec_slot(ctx, ctx->op->in[0]), w,
									   exec_slot(ctx, ctx->op->in[0]), n_heads, lc->head_dim,
									   ctx->m->norm_eps, ctx->rows.n_rows);
	} else {
		rmsnorm_ph_row_ud ud = {t, w, n_heads, lc->head_dim, ctx->m->norm_eps};
		st = batch_row_each(ctx, ctx->op->in[0], row_stride, rmsnorm_ph_row_fn, &ud);
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
	backend				  *t  = OP_BACKEND(a, rmsnorm_noweight_per_head);
	ps						  = profile_begin(prof, ctx->op->stage);
	if (t->rmsnorm_noweight_per_head_batch) {
		st = t->rmsnorm_noweight_per_head_batch(t, exec_slot(ctx, ctx->op->in[0]),
												exec_slot(ctx, ctx->op->in[0]), lc->n_kv_heads,
												lc->head_dim, ctx->m->norm_eps, ctx->rows.n_rows);
	} else if (t->rmsnorm_noweight_batch) {
		st = t->rmsnorm_noweight_batch(t, exec_slot(ctx, ctx->op->in[0]),
									   exec_slot(ctx, ctx->op->in[0]), lc->kv_row_stride,
									   ctx->m->norm_eps, ctx->rows.n_rows);
	} else if (t->rmsnorm_noweight_per_head) {
		for (int row = 0; row < ctx->rows.n_rows; row++) {
			buffer rowb =
				batch_row_view(batch_slot(ctx->bs, ctx->op->in[0]), row, lc->kv_row_stride);
			st = t->rmsnorm_noweight_per_head(t, &rowb, &rowb, lc->n_kv_heads, lc->head_dim,
											  ctx->m->norm_eps);
			if (st != OK)
				break;
		}
	} else {
		backend			 *t2 = OP_BACKEND(a, rmsnorm_noweight);
		rmsnorm_nw_row_ud ud = {t2, ctx->m->norm_eps, lc->kv_row_stride};
		st = batch_row_each(ctx, ctx->op->in[0], lc->kv_row_stride, rmsnorm_nw_row_fn, &ud);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_rmsnorm_add(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	const buffer *w			= exec_weight(ctx, ctx->op->w_idx);
	int			  n			= ctx->m->dim;
	float		  out_scale = ctx->op->u.rmsnorm.out_scale;
	if (a->rmsnorm_add_batch) {
		st = a->rmsnorm_add_batch(a, exec_slot(ctx, ctx->op->in[0]), w,
								  exec_slot(ctx, ctx->op->in[1]), exec_slot(ctx, ctx->op->out), n,
								  ctx->op->u.rmsnorm.eps, out_scale, ctx->rows.n_rows);
	} else if (a->rmsnorm_add && backend_has_cap(a, BCAP_RMSNORM_ADD) && ctx->rows.n_rows < 4) {
		for (int row = 0; row < ctx->rows.n_rows; row++) {
			buffer xrow	  = batch_row_view(exec_slot(ctx, ctx->op->in[0]), row, n);
			buffer resrow = batch_row_view(exec_slot(ctx, ctx->op->in[1]), row, n);
			buffer outrow = batch_row_view(exec_slot(ctx, ctx->op->out), row, n);
			st =
				a->rmsnorm_add(a, &xrow, w, &resrow, &outrow, n, ctx->op->u.rmsnorm.eps, out_scale);
			if (st != OK)
				break;
		}
	} else {
		st = a->rmsnorm_batch(a, exec_slot(ctx, ctx->op->in[0]), w, exec_slot(ctx, ctx->op->out), n,
							  ctx->op->u.rmsnorm.eps, ctx->rows.n_rows);
		if (st == OK)
			st = a->add_batch(a, exec_slot(ctx, ctx->op->out), exec_slot(ctx, ctx->op->in[1]), n,
							  ctx->rows.n_rows);
		if (st == OK && out_scale != 1.0f) {
			buffer *outb = exec_slot(ctx, ctx->op->out);
			if (a->scale_inplace)
				st = a->scale_inplace(a, outb, out_scale, n * ctx->rows.n_rows);
			else
				st = scale_inplace_fallback(ctx->s, a, outb, n * ctx->rows.n_rows, out_scale);
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
	if (a->scale_inplace) {
		st = a->scale_inplace(a, exec_slot(ctx, ctx->op->in[0]), sc, scale_dim * ctx->rows.n_rows);
	} else {
		st = scale_inplace_fallback(ctx->s, a, exec_slot(ctx, ctx->op->in[0]),
									scale_dim * ctx->rows.n_rows, sc);
	}
	profile_end(prof, &ps);
	return st;
}

typedef struct {
	struct model		   *m;
	row_desc				rows;
	struct compute_scratch *s;
	batch_scratch		   *bs;
	int						li;
	int						flash_attn;
} op_walker;

static status_code walk_layer_ops(op_walker *w, const recipe_op *lops, int n_ops) {
	model	*m		  = w->m;
	int		 li		  = w->li;
	backend *layer_be = NULL;
	if (m->mixed_backend_mode)
		layer_be = model_layer_backend(m, li);
	else if (w->s->active_backend)
		layer_be = w->s->active_backend;
	else
		layer_be = m->backend;
	weight_ref **wtab_row = NULL;
	if (m->wrefs_by_layer && m->n_layers > 0) {
		int row	 = (li >= 0 && li < m->n_layers) ? li : 0;
		wtab_row = &m->wrefs_by_layer[(size_t)row * WIDX_COUNT];
	}
	exec_ctx ctx = {
		.op			= NULL,
		.m			= m,
		.rows		= w->rows,
		.s			= w->s,
		.bs			= w->bs,
		.li			= li,
		.flash_attn = w->flash_attn,
		.layer_be	= layer_be,
		.wtab_row	= wtab_row,
	};
	int j = 0;
	while (j < n_ops) {
		const recipe_op *rop = &lops[j];

		if (rop->kind == OP_NONE) {
			j++;
			continue;
		}

		int run_len = (int)rop->coalesce_run_len;
		if (run_len > 1) {
			status_code coalesced_status;
			buffer	   *ys[RECIPE_COALESCE_MAX];
			for (int q = 0; q < run_len; q++)
				ys[q] = batch_slot(w->bs, lops[j + q].out);
			int consumed = exec_matmul_run_coalesced_batch(&lops[j], run_len, m, w->s, li,
														   batch_slot(w->bs, rop->in[0]), ys,
														   w->rows.n_rows, &coalesced_status);
			if (consumed == 0 && coalesced_status == OK)
				consumed =
					exec_matmul_run_qonly(&lops[j], run_len, m, batch_slot(w->bs, rop->in[0]), ys,
										  li, w->rows.n_rows, &coalesced_status);
			if (consumed < 0)
				return coalesced_status;
			if (consumed > 0) {
				if (coalesced_status != OK)
					return coalesced_status;
				j += consumed;
				continue;
			}
		}

		if (rop->kind < 0 || rop->kind >= OP_KIND_COUNT || !g_op_dispatch[rop->kind])
			return ERR_INVALID_ARG;
		ctx.op		   = rop;
		status_code st = g_op_dispatch[rop->kind](&ctx);
		if (st != OK) {
			ERROR("layer %d op[%d] kind=%d (stage=%d) w_idx=%d failed: %s", li, j, rop->kind,
				  rop->stage, rop->w_idx, status_str(st));
			return st;
		}
		j++;
	}

	if (w->s->layer_cb)
		w->s->layer_cb(li + 1, m->n_layers, -1, 1, w->s->layer_cb_ud);

	if (w->s->interrupt && *w->s->interrupt)
		return ERR_INTERRUPTED;
	return OK;
}

status_code compute_forward_recipe(struct model *m, struct kvcache *cache,
								   struct compute_scratch *s, int token, int pos, int flash_attn,
								   float *logits_out) {
	const int32_t tok = token;
	return compute_forward_decode_batch_recipe(m, &cache, s, &tok, &pos, NULL, 1, flash_attn,
											   logits_out);
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
	int				moe_rowslots_cap;

	float_buf moe_expert_out;
	float_buf moe_out;
	float_buf moe_router_logits;
	float_buf moe_router_inp;
	buffer	  moe_router_logits_dev;
	buffer	  moe_router_inp_dev;
	float_buf moe_xb_f;

	float_buf moe_gather_x;
	float_buf moe_exp_gu;
	float_buf moe_up_scratch;
	float_buf moe_exp_act;
	float_buf moe_exp_y;
	float_buf moe_fused_host;
	float_buf moe_fact_out;
	float_buf moe_zeros;

	moe_expert_slot *moe_per_token_slots;
	int				 moe_per_token_slots_cap;

	moe_expert_slot *moe_union_slots;
	int				 moe_union_slots_cap;
	int				 moe_union_pending_n;

	const int32_t *tokens;

	float_buf ple_buf;
	buffer	  ple_all;
	float_buf ple_inp;
	float_buf ple_proj;
	float_buf ple_slice;
	buffer	  ple_inp_dev;
	buffer	  ple_slice_dev;
};

enum {
	BS_SIZE_NONE = 0,
	BS_SIZE_ATTN_BUF,
	BS_SIZE_Q_OUT,
	BS_SIZE_KV_OUT,
	BS_SIZE_INTER,
	BS_SIZE_INTER_X2,
	BS_SIZE_ACT,
	BS_SIZE_HC,
	BS_SIZE_HC_COEF,
} bs_slot_size_kind;

static const struct {
	uint8_t valid;
	uint8_t size_kind;
} bs_slot_info[RECIPE_SLOT_MAX] = {
	[RECIPE_SLOT_X]			  = {1, BS_SIZE_ATTN_BUF},
	[RECIPE_SLOT_XB]		  = {1, BS_SIZE_ATTN_BUF},
	[RECIPE_SLOT_XB2]		  = {1, BS_SIZE_ATTN_BUF},
	[RECIPE_SLOT_ATTN_OUT]	  = {1, BS_SIZE_ATTN_BUF},
	[RECIPE_SLOT_Q]			  = {1, BS_SIZE_Q_OUT},
	[RECIPE_SLOT_K]			  = {1, BS_SIZE_KV_OUT},
	[RECIPE_SLOT_V]			  = {1, BS_SIZE_KV_OUT},
	[RECIPE_SLOT_FFN_GATE]	  = {1, BS_SIZE_INTER},
	[RECIPE_SLOT_FFN_UP]	  = {1, BS_SIZE_INTER},
	[RECIPE_SLOT_FFN_ACT]	  = {1, BS_SIZE_ACT},
	[RECIPE_SLOT_FFN_GATE_UP] = {1, BS_SIZE_INTER_X2},
	[RECIPE_SLOT_RESID_TMP]	  = {1, BS_SIZE_ATTN_BUF},
	[RECIPE_SLOT_HYB_PROJ]	  = {1, BS_SIZE_NONE},
	[RECIPE_SLOT_HYB_GATE]	  = {1, BS_SIZE_NONE},
	[RECIPE_SLOT_HYB_ALPHA]	  = {1, BS_SIZE_NONE},
	[RECIPE_SLOT_HYB_BETA]	  = {1, BS_SIZE_NONE},
	[RECIPE_SLOT_LOGITS]	  = {1, BS_SIZE_NONE},
	[RECIPE_SLOT_HC]		  = {1, BS_SIZE_HC},
	[RECIPE_SLOT_HC_COEF]	  = {1, BS_SIZE_HC_COEF},
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

	if ((int)op->kind >= 0 && op->kind < OP_KIND_COUNT)
		m |= OP_REQUIREMENTS[op->kind].batch_extra_mask;
	return m & ((RECIPE_SLOT_MAX >= 32) ? 0xffffffffu : ((1u << RECIPE_SLOT_MAX) - 1u));
}

static int bs_ensure_slot(batch_scratch *bs, backend *owner, uint8_t slot, size_t n_elems) {
	bs_pair *p	   = &bs->pair[slot];
	size_t	 bytes = n_elems * sizeof(float);

	if (!backend_has_cap(owner, BCAP_IS_HOST)) {
		if (p->b.owner == owner && p->b.handle && p->b.size >= bytes)
			return 0;
		if (p->b.handle && p->b.owner && !backend_has_cap(p->b.owner, BCAP_IS_HOST))
			p->b.owner->buffer_free(p->b.owner, &p->b);
		memset(&p->b, 0, sizeof(p->b));
		owner->buffer_alloc_scratch(owner, bytes, &p->b);
		return 1;
	}
	if (p->b.handle && p->b.owner && p->b.owner != owner &&
		!backend_has_cap(p->b.owner, BCAP_IS_HOST)) {
		p->b.owner->buffer_free(p->b.owner, &p->b);
		memset(&p->b, 0, sizeof(p->b));
	}
	int	   existed = p->b.handle != NULL;
	size_t old_cap = p->fb.cap;
	float_buf_ensure_nocopy(&p->fb, n_elems, 64);
	int fresh	  = !existed || p->fb.cap > old_cap;
	p->b.handle	  = p->fb.p;
	p->b.host_ptr = p->fb.p;
	p->b.size	  = bytes;
	p->b.offset	  = 0;
	p->b.owner	  = owner;
	return fresh;
}

static size_t bs_slot_elems(uint8_t slot, int m, int q_out, int kv_out, int intermediate,
							int attn_buf_size, int ffn_act_size, int hc_flat, int hc_coef) {
	if (slot >= RECIPE_SLOT_MAX)
		return 0;
	switch (bs_slot_info[slot].size_kind) {
	case BS_SIZE_ATTN_BUF:
		return (size_t)m * (size_t)attn_buf_size;
	case BS_SIZE_Q_OUT:
		return (size_t)m * (size_t)q_out;
	case BS_SIZE_KV_OUT:
		return (size_t)m * (size_t)kv_out;
	case BS_SIZE_INTER:
		return (size_t)m * (size_t)intermediate;
	case BS_SIZE_INTER_X2:
		return (size_t)m * 2 * (size_t)intermediate;
	case BS_SIZE_ACT:
		return (size_t)m * (size_t)ffn_act_size;
	case BS_SIZE_HC:
		return (size_t)m * (size_t)hc_flat;
	case BS_SIZE_HC_COEF:
		return (size_t)m * (size_t)hc_coef;
	case BS_SIZE_NONE:
	default:
		return 0;
	}
}

static void batch_scratch_alloc(batch_scratch *bs, backend *owner, int m, int dim, int q_out,
								int kv_out, int intermediate, int attn_buf_size, uint32_t slot_mask,
								int hc_flat, int hc_coef) {
	int	   ffn_act_size = intermediate > dim ? intermediate : dim;
	size_t alloc_bytes = 0, total_bytes = 0;
	int	   fresh_alloc = 0;
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (!bs_slot_info[slot].valid)
			continue;
		if (slot_mask && !(slot_mask & bs_slot_bit((uint8_t)slot)))
			continue;
		size_t elems = bs_slot_elems((uint8_t)slot, m, q_out, kv_out, intermediate, attn_buf_size,
									 ffn_act_size, hc_flat, hc_coef);
		if (!elems)
			continue;
		total_bytes += elems * sizeof(float);
		if (bs_ensure_slot(bs, owner, (uint8_t)slot, elems)) {
			fresh_alloc = 1;
			alloc_bytes += elems * sizeof(float);
		}
	}
	if (fresh_alloc)
		INFO("batch scratch: %zu B new, %zu B total (%zu MiB) for %d rows", alloc_bytes,
			 total_bytes, total_bytes >> 20, m);
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
	if (bs->moe_router_logits_dev.owner)
		bs->moe_router_logits_dev.owner->buffer_free(bs->moe_router_logits_dev.owner,
													 &bs->moe_router_logits_dev);
	if (bs->moe_router_inp_dev.owner)
		bs->moe_router_inp_dev.owner->buffer_free(bs->moe_router_inp_dev.owner,
												  &bs->moe_router_inp_dev);
	free(bs->moe_xb_f.p);
	free(bs->moe_gather_x.p);
	free(bs->moe_exp_gu.p);
	free(bs->moe_up_scratch.p);
	free(bs->moe_exp_act.p);
	free(bs->moe_exp_y.p);
	free(bs->moe_fused_host.p);
	free(bs->moe_fact_out.p);
	free(bs->moe_zeros.p);
	free(bs->moe_per_token_slots);
	free(bs->moe_union_slots);
	free(bs->ple_buf.p);
	if (bs->ple_all.owner)
		bs->ple_all.owner->buffer_free(bs->ple_all.owner, &bs->ple_all);
	free(bs->ple_inp.p);
	free(bs->ple_proj.p);
	free(bs->ple_slice.p);
	if (bs->ple_inp_dev.owner)
		bs->ple_inp_dev.owner->buffer_free(bs->ple_inp_dev.owner, &bs->ple_inp_dev);
	if (bs->ple_slice_dev.owner)
		bs->ple_slice_dev.owner->buffer_free(bs->ple_slice_dev.owner, &bs->ple_slice_dev);
	memset(bs, 0, sizeof(*bs));
}

static buffer *batch_slot(batch_scratch *bs, uint8_t slot) {
	if (slot >= RECIPE_SLOT_MAX || !bs_slot_info[slot].valid)
		return NULL;
	return &bs->pair[slot].b;
}

static buffer batch_row_view(const buffer *whole, int row, int row_elems) {
	buffer v	 = *whole;
	size_t delta = (size_t)row * row_elems * sizeof(float);
	v.offset	 = whole->offset + delta;
	if (whole->size >= delta)
		v.size = whole->size - delta;
	return v;
}

static inline float *batch_buf_ptr(const buffer *b) {
	if (b->host_ptr)
		return (float *)((const char *)b->host_ptr + b->offset);
	return (float *)((char *)b->handle + b->offset);
}

typedef struct {
	backend		  *a;
	const buffer  *w;
	uint32_t	   w_type;
	const int32_t *tokens;
	buffer		  *xs;
	int			   dim;
	float		   scale;
	_Atomic int	  *first_err;
} embd_fill_job;

static void embd_fill_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	embd_fill_job *j = ctx;
	for (int row = begin; row < end; row++) {
		buffer		xrow = batch_row_view(j->xs, row, j->dim);
		int			tok	 = j->tokens ? j->tokens[row] : 0;
		status_code st	 = j->a->embd_lookup(j->a, j->w, j->w_type, tok, j->dim, &xrow);
		if (st != OK) {
			atomic_store_explicit(j->first_err, st, memory_order_relaxed);
			continue;
		}
		if (j->scale != 1.0f) {
			float *xf = batch_buf_ptr(&xrow);
			for (int i = 0; i < j->dim; i++)
				xf[i] *= j->scale;
		}
	}
}

static status_code embd_fill_rows(backend *a, const buffer *w, uint32_t w_type,
								  const int32_t *tokens, buffer *xs, int n_rows, int dim,
								  float scale) {
	tpool *pool = backend_get_pool(a);
	if (pool && n_rows >= 16 && backend_has_cap(a, BCAP_IS_HOST) && tpool_current_tid() < 0 &&
		tpool_n_threads(pool) > 1) {
		_Atomic int	  err = OK;
		embd_fill_job j	  = {.a			= a,
							 .w			= w,
							 .w_type	= w_type,
							 .tokens	= tokens,
							 .xs		= xs,
							 .dim		= dim,
							 .scale		= scale,
							 .first_err = &err};
		tpool_parallel_for(pool, n_rows, 4, embd_fill_chunk, &j);
		return atomic_load_explicit(&err, memory_order_relaxed);
	}

	for (int row = 0; row < n_rows; row++) {
		buffer		xrow = batch_row_view(xs, row, dim);
		int			tok	 = tokens ? tokens[row] : 0;
		status_code st	 = a->embd_lookup(a, w, w_type, tok, dim, &xrow);
		if (st != OK)
			return st;
	}
	if (scale == 1.0f)
		return OK;
	if (a->scale_inplace)
		return a->scale_inplace(a, xs, scale, n_rows * dim);
	ensure_sync(a);
	float *xf = batch_buf_ptr(xs);
	for (int i = 0; i < n_rows * dim; i++)
		xf[i] *= scale;
	return OK;
}

static status_code ffn_activate_ex_rows(backend *t, const float *fused, float *out, int n, int act,
										int n_rows, int fused_stride) {
	for (int row = 0; row < n_rows; row++) {
		const float *g	= fused + (size_t)row * fused_stride;
		const float *u	= g + n;
		float		*o	= out + (size_t)row * n;
		buffer		 gb = buffer_host_view(t, (void *)g, (size_t)n * sizeof(float));
		buffer		 ub = buffer_host_view(t, (void *)u, (size_t)n * sizeof(float));
		buffer		 ob = buffer_host_view(t, o, (size_t)n * sizeof(float));
		status_code	 st = t->ffn_activate_ex(t, &gb, &ub, &ob, n, act);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code ffn_activate_fused_device_fallback(exec_ctx *ctx, backend *a, int n, int act,
													  int fused_stride, int n_rows) {
	buffer *fused_buf = batch_slot(ctx->bs, ctx->op->in[0]);
	buffer *out_buf	  = batch_slot(ctx->bs, ctx->op->out);
	float  *fused_host =
		float_buf_ensure_nocopy(&ctx->bs->moe_fused_host, (size_t)n_rows * fused_stride, 64);
	float	   *out_host = float_buf_ensure_nocopy(&ctx->bs->moe_fact_out, (size_t)n_rows * n, 64);
	status_code st		 = a->buffer_read_f32(a, fused_buf, fused_host, n_rows * fused_stride);
	if (st != OK)
		return st;
	backend *t = op_host_backend();
	if (!t->ffn_activate_ex)
		return ERR_UNSUPPORTED;
	st = ffn_activate_ex_rows(t, fused_host, out_host, n, act, n_rows, fused_stride);
	if (st != OK)
		return st;
	return a->buffer_write_f32(a, out_buf, out_host, n_rows * n);
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
	size_t				   per_expert_out;
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

status_code moe_activate(backend *a, float *act, const float *gate, const float *up, int n_i,
						 float gs, float us, int use_gelu) {
	if (!(a && a->moe_activate && backend_has_cap(a, BCAP_IS_HOST)))
		backend_report_host_fallback(a, "moe_activate", HFB_OP_NOT_NATIVE,
									 "moe activation executes on the host (cpu) fallback");
	backend *h = (a && a->moe_activate && backend_has_cap(a, BCAP_IS_HOST)) ? a : backend_host();
	if (!h || !h->moe_activate)
		return ERR_UNSUPPORTED;
	buffer gb = buffer_host_view(h, (void *)gate, (size_t)n_i * sizeof(float));
	buffer ub = buffer_host_view(h, (void *)up, (size_t)n_i * sizeof(float));
	buffer ob = buffer_host_view(h, act, (size_t)n_i * sizeof(float));
	return h->moe_activate(h, &gb, &ub, &ob, n_i, gs, us, use_gelu);
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

		K = moe_topk_select(logits, E, K, top_idx, top_score);
		for (int k = 0; k < K; k++)
			weight[k] = top_score[k];
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
	}

	moe_apply_weights(weight, K, norm_topk, routed_scale);
	for (int k = 0; k < K; k++) {
		ids_out[k] = top_idx[k];
		w_out[k]   = weight[k];
	}
	return K;
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

static void moe_batch_ensure_rowslots(batch_scratch *bs, int rows, int K) {
	size_t need = (size_t)rows * (size_t)K;
	if ((size_t)bs->moe_rowslots_cap < need) {
		bs->moe_router_ids	  = xrealloc(bs->moe_router_ids, need * sizeof(int));
		bs->moe_router_w	  = xrealloc(bs->moe_router_w, need * sizeof(float));
		bs->moe_union_ids	  = xrealloc(bs->moe_union_ids, need * sizeof(int));
		bs->moe_union_rows	  = xrealloc(bs->moe_union_rows, need * sizeof(int));
		bs->moe_union_kidx	  = xrealloc(bs->moe_union_kidx, need * sizeof(int));
		bs->moe_union_offsets = xrealloc(bs->moe_union_offsets, (need + 1) * sizeof(int));
		bs->moe_union_cursor  = xrealloc(bs->moe_union_cursor, need * sizeof(int));
		bs->moe_union_count	  = xrealloc(bs->moe_union_count, need * sizeof(int));
		bs->moe_rowslots_cap  = (int)need;
	}
}

static void moe_batch_ensure_experts(batch_scratch *bs, int E) {
	if (bs->moe_expert_alloc < E) {
		bs->moe_expert_union_idx = xrealloc(bs->moe_expert_union_idx, (size_t)E * sizeof(int));
		bs->moe_expert_seen_gen	 = xrealloc(bs->moe_expert_seen_gen, (size_t)E * sizeof(int));
		memset(bs->moe_expert_seen_gen, 0, (size_t)E * sizeof(int));
		bs->moe_expert_gen	 = 0;
		bs->moe_expert_alloc = E;
	}
}

static void moe_union_slots_ensure(batch_scratch *bs, int n_union) {
	if (bs->moe_union_slots_cap < n_union) {
		moe_expert_slot *grown =
			xrealloc(bs->moe_union_slots, (size_t)n_union * sizeof(moe_expert_slot));
		size_t old_n = (size_t)bs->moe_union_slots_cap;
		memset(grown + old_n, 0, ((size_t)n_union - old_n) * sizeof(moe_expert_slot));
		bs->moe_union_slots		= grown;
		bs->moe_union_slots_cap = n_union;
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

	moe_batch_ensure_rowslots(ctx->bs, ctx->rows.n_rows, K);
	moe_batch_ensure_experts(ctx->bs, E);
	float_buf_ensure(&ctx->bs->moe_router_logits, (size_t)ctx->rows.n_rows * E * 2);
	float_buf_ensure(&ctx->bs->moe_router_inp, (size_t)ctx->rows.n_rows * router_dim);
	float_buf_ensure(&ctx->bs->moe_xb_f, (size_t)ctx->rows.n_rows * router_dim);

	buffer *xb = batch_slot(ctx->bs, ctx->op->in[0]);

	int			n_rows	= ctx->rows.n_rows;
	float	   *xb_host = ctx->bs->moe_xb_f.p;
	status_code st		= a->buffer_read_f32(a, xb, xb_host, n_rows * router_dim);
	if (st != OK)
		return st;

	for (int row = 0; row < n_rows; row++) {
		const float *router_input_src = xb_host + (size_t)row * router_dim;
		float		*router_input	  = ctx->bs->moe_router_inp.p + (size_t)row * router_dim;
		memcpy(router_input, router_input_src, (size_t)router_dim * sizeof(float));

		if (uses_softmax)
			moe_router_normalize_input(ctx->m, L, router_dim, router_input);
	}

	buffer *router_inp_buf = &ctx->bs->moe_router_inp_dev;
	st = buffer_ensure_scratch(a, router_inp_buf,
							   (size_t)ctx->rows.n_rows * router_dim * sizeof(float));
	if (st != OK)
		return st;

	st = a->buffer_write_f32(a, router_inp_buf, ctx->bs->moe_router_inp.p,
							 ctx->rows.n_rows * router_dim);
	if (st != OK)
		return st;

	buffer *router_logits_buf = &ctx->bs->moe_router_logits_dev;
	st = buffer_ensure_scratch(a, router_logits_buf, (size_t)ctx->rows.n_rows * E * sizeof(float));
	if (st != OK)
		return st;

	st = a->matmul_batch(a, &L->router_w.buf, L->router_w.type, router_inp_buf, router_logits_buf,
						 E, router_dim, ctx->rows.n_rows);
	if (st != OK)
		return st;

	st = a->buffer_read_f32(a, router_logits_buf, ctx->bs->moe_router_logits.p,
							ctx->rows.n_rows * E);
	if (st != OK)
		return st;

	if (!moe_router_logits_finite(ctx->bs->moe_router_logits.p, (size_t)ctx->rows.n_rows * E)) {
		ERROR("moe_router_batch: non-finite router logit (layer=%d, rows=%d, E=%d)", ctx->li,
			  ctx->rows.n_rows, E);
		return ERR_FORMAT;
	}

	tpool			   *router_pool = model_get_pool(ctx->m);
	moe_router_emit_job emit_job	= {.bs			 = ctx->bs,
									   .L			 = L,
									   .E			 = E,
									   .K			 = K,
									   .rows		 = ctx->rows.n_rows,
									   .uses_softmax = uses_softmax,
									   .norm_topk	 = norm_topk,
									   .routed_scale = routed_scale,
									   .n_group		 = ctx->m->moe.n_group,
									   .topk_group	 = ctx->m->moe.topk_group};
	if (router_pool && ctx->rows.n_rows >= 8 && E >= 8 && tpool_current_tid() < 0) {
		tpool_parallel_for(router_pool, ctx->rows.n_rows, 1, moe_router_emit_chunk, &emit_job);
	} else {
		moe_router_emit_chunk(0, ctx->rows.n_rows, 0, &emit_job);
	}

	int gen = ++ctx->bs->moe_expert_gen;

	int n_union = 0;
	for (int row = 0; row < ctx->rows.n_rows; row++) {
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
		for (int row = 0; row < ctx->rows.n_rows; row++) {
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
		ARR_ENSURE(ctx->bs->moe_union_order, n_union, ctx->bs->moe_union_order_cap);
		ARR_ENSURE(ctx->bs->moe_union_sorted_ids, n_union, ctx->bs->moe_union_sorted_ids_cap);
		ARR_ENSURE(ctx->bs->moe_order_pairs, n_union, ctx->bs->moe_order_pairs_cap);
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
		int total_slots = ctx->rows.n_rows * K;
		ARR_ENSURE(ctx->bs->moe_per_token_slots, total_slots, ctx->bs->moe_per_token_slots_cap);
	}

	moe_union_slots_ensure(ctx->bs, n_union);

	status_code rst = moe_stream_resolve(ctx->m, ctx->li, ctx->bs->moe_union_sorted_ids, n_union,
										 ctx->bs->moe_union_slots);
	if (rst != OK) {
		ERROR("moe_router_batch: moe_stream_resolve failed (layer=%d, union=%d, st=%d)", ctx->li,
			  n_union, rst);
		ctx->bs->moe_n_union		 = 0;
		ctx->bs->moe_union_pending_n = 0;
		return ERR_INTERNAL;
	}
	ctx->bs->moe_union_pending_n = n_union;
	return OK;
}

static status_code moe_experts_grouped(exec_ctx *ctx, backend *a, int dim, int K, int I,
									   int use_gelu) {
	batch_scratch *bs	   = ctx->bs;
	int			   n_rows  = ctx->rows.n_rows;
	int			   n_union = bs->moe_n_union;

	buffer	   *xb_dev	= batch_slot(bs, ctx->op->in[0]);
	float	   *xb_host = float_buf_ensure(&bs->moe_xb_f, (size_t)n_rows * dim);
	status_code st		= a->buffer_read_f32(a, xb_dev, xb_host, n_rows * dim);
	if (st != OK)
		return st;
	const float *xb_base = xb_host;
	float		*out	 = bs->moe_out.p;

	float_buf_ensure(&bs->moe_gather_x, (size_t)n_rows * dim);
	float_buf_ensure(&bs->moe_exp_act, (size_t)n_rows * I);
	float_buf_ensure(&bs->moe_exp_y, (size_t)n_rows * dim);
	float *contrib = float_buf_ensure(&bs->moe_expert_out, (size_t)n_rows * K * dim);
	memset(contrib, 0, (size_t)n_rows * K * dim * sizeof(float));

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
				host_matmul_generic(es->gate_w, es->gate_type, X + (size_t)c * dim,
									GU + (size_t)c * I * 2, I * 2, dim);
			for (int c = 0; c < cnt; c++) {
				st = moe_activate(backend_host(), A + (size_t)c * I, GU + (size_t)c * I * 2,
								  GU + (size_t)c * I * 2 + I, I, es->gate_scale, es->up_scale,
								  use_gelu);
				if (st != OK)
					return st;
			}
		} else {
			float *G = float_buf_ensure(&bs->moe_exp_gu, (size_t)cnt * I);
			float *U = float_buf_ensure(&bs->moe_up_scratch, (size_t)cnt * I);
			for (int c = 0; c < cnt; c++) {
				host_matmul_generic(es->gate_w, es->gate_type, X + (size_t)c * dim,
									G + (size_t)c * I, I, dim);
				host_matmul_generic(es->up_w, es->up_type, X + (size_t)c * dim, U + (size_t)c * I,
									I, dim);
			}
			for (int c = 0; c < cnt; c++) {
				st = moe_activate(backend_host(), A + (size_t)c * I, G + (size_t)c * I,
								  U + (size_t)c * I, I, es->gate_scale, es->up_scale, use_gelu);
				if (st != OK)
					return st;
			}
		}

		float *Y = bs->moe_exp_y.p;
		for (int c = 0; c < cnt; c++)
			host_matmul_generic(es->down_w, es->down_type, A + (size_t)c * I, Y + (size_t)c * dim,
								dim, I);

		float ds = es->down_scale;
		for (int c = 0; c < cnt; c++) {
			float		 w = bs->moe_router_w[(size_t)rows[c] * K + kidx[c]];
			const float *y = Y + (size_t)c * dim;
			float		*o = contrib + ((size_t)rows[c] * K + kidx[c]) * dim;
			if (ds == 1.0f) {
				for (int d = 0; d < dim; d++)
					o[d] = w * y[d];
			} else {
				for (int d = 0; d < dim; d++)
					o[d] = w * (ds * y[d]);
			}
		}
	}
	for (int row = 0; row < n_rows; row++) {
		float *o = out + (size_t)row * dim;
		for (int k = 0; k < K; k++) {
			const float *c = contrib + ((size_t)row * K + k) * dim;
			for (int d = 0; d < dim; d++)
				o[d] += c[d];
		}
	}
	return OK;
}

static status_code moe_experts_resident_grouped(exec_ctx *ctx, backend *a, int dim, int K, int I,
												int use_gelu) {
	batch_scratch *bs	   = ctx->bs;
	int			   n_rows  = ctx->rows.n_rows;
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
		if (cnt <= 0) {
			counts[i] = 0;
			continue;
		}
		if (es->eid < 0 || !es->dev_ready || es->dev_gate.owner != a) {
			free(experts);
			free(rows_packed);
			free(weights_packed);
			return ERR_UNSUPPORTED;
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

static void moe_experts_batch_split(exec_ctx *ctx, backend *a, int dim, int K, int I,
									int use_gelu) {
	batch_scratch *bs	   = ctx->bs;
	int			   n_rows  = ctx->rows.n_rows;
	int			   n_union = bs->moe_n_union;
	if (n_union <= 0)
		return;

	backend **owners = xcalloc((size_t)n_union, sizeof(*owners));
	int		  any	 = 0;
	for (int i = 0; i < n_union; i++) {
		int				 u	= bs->moe_union_order[i];
		moe_expert_slot *es = &bs->moe_union_slots[i];
		if (bs->moe_union_count[u] > 0 && es->eid >= 0 && es->dev_ready) {
			owners[i] = es->dev_gate.owner;
			any |= owners[i] != NULL;
		}
	}
	size_t total = (size_t)n_rows * dim;
	float *xh	 = float_buf_ensure(&bs->moe_xb_f, total);
	if (!any || a->buffer_read_f32(a, batch_slot(bs, ctx->op->in[0]), xh, (int)total) != OK) {
		free(owners);
		return;
	}

	moe_resident_expert *experts		= xcalloc((size_t)n_union, sizeof(*experts));
	int					*counts			= xcalloc((size_t)n_union, sizeof(int));
	int					*idx			= xcalloc((size_t)n_union, sizeof(int));
	int					*rows_packed	= xcalloc((size_t)n_rows * K, sizeof(int));
	float				*weights_packed = xcalloc((size_t)n_rows * K, sizeof(float));

	for (int i0 = 0; i0 < n_union; i0++) {
		backend *eb = owners[i0];
		if (!eb)
			continue;
		int n_e	   = 0;
		int packed = 0;
		for (int i = i0; i < n_union; i++) {
			if (owners[i] != eb)
				continue;
			int				 u		   = bs->moe_union_order[i];
			int				 cnt	   = bs->moe_union_count[u];
			const int		*rows	   = bs->moe_union_rows + bs->moe_union_offsets[u];
			const int		*kidx	   = bs->moe_union_kidx + bs->moe_union_offsets[u];
			moe_expert_slot *es		   = &bs->moe_union_slots[i];
			idx[n_e]				   = i;
			counts[n_e]				   = cnt;
			experts[n_e].gate_w		   = &es->dev_gate;
			experts[n_e].up_w		   = &es->dev_up;
			experts[n_e].down_w		   = &es->dev_down;
			experts[n_e].gate_type	   = es->gate_type;
			experts[n_e].up_type	   = es->up_type;
			experts[n_e].down_type	   = es->down_type;
			experts[n_e].gate_up_fused = es->gate_up_fused;
			experts[n_e].use_gelu	   = use_gelu;
			experts[n_e].gate_scale	   = es->gate_scale;
			experts[n_e].up_scale	   = es->up_scale;
			experts[n_e].down_scale	   = es->down_scale;
			for (int c = 0; c < cnt; c++) {
				rows_packed[packed]	   = rows[c];
				weights_packed[packed] = bs->moe_router_w[(size_t)rows[c] * K + kidx[c]];
				packed++;
			}
			n_e++;
		}

		status_code		  st = OK;
		expert_dev_entry *e	 = NULL;
		if (!eb->moe_experts_batch || n_e > MOE_MAX_TOPK * 8)
			st = ERR_UNSUPPORTED;
		else if (!(e = expert_dev_scratch_get(&ctx->s->expert_dev, eb, total)))
			st = ERR_OUT_OF_MEMORY;
		if (st == OK)
			st = eb->buffer_write_f32(eb, &e->x, xh, (int)total);
		if (st == OK)
			st = eb->scale_inplace(eb, &e->out, 0.0f, (int)total);
		if (st == OK)
			st = eb->moe_experts_batch(eb, &e->x, &e->out, n_rows, dim, I, use_gelu, n_e, experts,
									   counts, rows_packed, weights_packed);
		if (st == OK) {
			float *tmp = float_buf_ensure(&ctx->s->expert_dev.tmp, total);
			st		   = eb->buffer_read_f32(eb, &e->out, tmp, (int)total);
			if (st == OK)
				for (size_t j = 0; j < total; j++)
					bs->moe_out.p[j] += tmp[j];
		}
		if (st == OK) {
			for (int j = 0; j < n_e; j++)
				bs->moe_union_count[bs->moe_union_order[idx[j]]] = 0;
		} else {
			WARN("moe expert batch on '%s' failed (st=%d) -- falling back to cpu",
				 eb->name ? eb->name : "?", (int)st);
		}
		for (int i = i0; i < n_union; i++)
			if (owners[i] == eb)
				owners[i] = NULL;
	}

	free(experts);
	free(counts);
	free(idx);
	free(rows_packed);
	free(weights_packed);
	free(owners);
}

static void moe_release_batch_slots(model *m, int li, batch_scratch *bs, int n_rows, int K,
									int n_union) {
	for (int i = 0; i < n_union; i++) {
		moe_expert_slot *us = &bs->moe_union_slots[i];
		if (us->owned && us->heap_buf) {
			free(us->heap_buf);
			us->heap_buf = NULL;
			us->eid		 = -1;
		}
	}
	moe_stream_release_slots(m, li, bs->moe_union_slots, n_union);

	for (int i = 0; i < n_union; i++)
		bs->moe_union_slots[i].eid = -1;
	for (int i = 0; i < n_rows * K; i++) {
		moe_expert_slot *es = &bs->moe_per_token_slots[i];
		es->eid				= -1;
		es->owned			= 0;
		es->heap_buf		= NULL;
		es->gate_w			= NULL;
		es->up_w			= NULL;
		es->down_w			= NULL;
	}
}

typedef struct {
	int dim;
	int K;
	int I;
	int use_gelu;
} moe_exec_plan;

static void moe_exec_plan_fill(moe_exec_plan *p, struct model *m) {
	p->dim		= m->dim;
	p->K		= m->moe.n_experts_used;
	p->I		= m->moe.moe_intermediate;
	p->use_gelu = m->arch_info->uses_gelu_activation;
}

static status_code moe_experts_batch(exec_ctx *ctx) {

	backend	 *a	  = exec_layer_backend(ctx);
	const int dim = ctx->m->dim;
	if (!model_layer_is_moe(ctx->m, ctx->li)) {
		buffer *dst	  = batch_slot(ctx->bs, ctx->op->out);
		size_t	need  = (size_t)ctx->rows.n_rows * dim;
		size_t	old	  = ctx->bs->moe_zeros.cap;
		float  *zeros = float_buf_ensure(&ctx->bs->moe_zeros, need);
		if (need > old)
			memset(zeros + old, 0, (need - old) * sizeof(float));
		return a->buffer_write_f32(a, dst, zeros, ctx->rows.n_rows * dim);
	}

	moe_exec_plan plan;
	moe_exec_plan_fill(&plan, ctx->m);
	int K		 = plan.K;
	int I		 = plan.I;
	int use_gelu = plan.use_gelu;

	int n_union = ctx->bs->moe_union_pending_n;
	for (int i = 0; i < ctx->rows.n_rows * K; i++)
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

	for (int row = 0; row < ctx->rows.n_rows; row++) {
		const int	*row_ids = ctx->bs->moe_router_ids + (size_t)row * K;
		const float *row_w	 = ctx->bs->moe_router_w + (size_t)row * K;
		monitor_emit_moe_experts(g_monitor, ctx->li, -1, row_ids, row_w, K);
	}

	float_buf_ensure(&ctx->bs->moe_out, (size_t)ctx->rows.n_rows * dim);
	memset(ctx->bs->moe_out.p, 0, ctx->rows.n_rows * dim * sizeof(float));

	status_code st = OK;
	if (ctx->rows.n_rows > 1 && a->matmul_batch) {
		if (backend_has_cap(a, BCAP_MOE_EXPERT_RESIDENT) && a->moe_experts_batch &&
			ctx->m->layers[ctx->li].experts_resident) {
			st = moe_experts_resident_grouped(ctx, a, dim, K, I, use_gelu);
			if (st == OK) {
				moe_release_batch_slots(ctx->m, ctx->li, ctx->bs, ctx->rows.n_rows, K, n_union);
				return OK;
			}
			if (st != ERR_UNSUPPORTED)
				WARN("moe resident expert batch failed (st=%d) -- falling back to cpu", (int)st);
		}
		if (ctx->m->layers[ctx->li].experts_resident)
			moe_experts_batch_split(ctx, a, dim, K, I, use_gelu);
		moe_experts_grouped(ctx, a, dim, K, I, use_gelu);
		goto finish;
	}

	tpool *pool			 = model_get_pool(ctx->m);
	int	   total_experts = ctx->rows.n_rows * K;

	if (pool && total_experts >= 2) {
		int n_threads = tpool_n_threads(pool);
		if (n_threads < 1)
			n_threads = 1;
		if (n_threads > total_experts)
			n_threads = total_experts;

		size_t per_thread_scratch = (size_t)I * 3 + (size_t)dim + (size_t)I * 2;
		float *all_scratch =
			float_buf_ensure(&ctx->s->moe_all_scratch, (size_t)n_threads * per_thread_scratch);
		size_t per_thread_out = (size_t)ctx->rows.n_rows * dim;
		float *all_outs =
			float_buf_ensure(&ctx->s->moe_all_outs, (size_t)n_threads * per_thread_out);
		for (int t = 0; t < n_threads; t++) {
			float *o = all_outs + (size_t)t * per_thread_out;
			memset(o, 0, per_thread_out * sizeof(float));
		}

		moe_batch_job job;
		job.m		 = ctx->m;
		job.a		 = backend_host();
		job.bs		 = ctx->bs;
		job.slot_buf = ctx->bs->moe_per_token_slots;
		job.li		 = ctx->li;
		job.top_k	 = K;
		job.inter	 = I;
		job.dim		 = dim;
		job.n_tokens = ctx->rows.n_rows;
		job.use_gelu = use_gelu;
		{
			buffer *xb_b = batch_slot(ctx->bs, ctx->op->in[0]);
			float  *xb_h = float_buf_ensure(&ctx->bs->moe_xb_f, (size_t)ctx->rows.n_rows * dim);
			st			 = a->buffer_read_f32(a, xb_b, xb_h, ctx->rows.n_rows * dim);
			if (st != OK)
				return st;
			job.xb_base = xb_h;
		}
		job.all_scratch		   = all_scratch;
		job.all_outs		   = all_outs;
		job.per_thread_scratch = per_thread_scratch;
		job.per_thread_out	   = per_thread_out;

		tpool_parallel_for(pool, total_experts, 1, moe_batch_expert_chunk, &job);

		for (int t = 0; t < n_threads; t++) {
			float *o = all_outs + (size_t)t * per_thread_out;
			for (int i = 0; i < ctx->rows.n_rows * dim; i++)
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
		float  *xb_host = float_buf_ensure(&ctx->bs->moe_xb_f, (size_t)ctx->rows.n_rows * dim);
		st				= a->buffer_read_f32(a, xb_b, xb_host, ctx->rows.n_rows * dim);
		if (st != OK)
			return st;

		for (int row = 0; row < ctx->rows.n_rows; row++) {
			const float *xb_row = xb_host + (size_t)row * dim;
			for (int k = 0; k < K; k++) {
				moe_expert_slot *es = &ctx->bs->moe_per_token_slots[(size_t)row * K + k];
				if (es->eid < 0 || !es->gate_w)
					continue;

				if (es->gate_up_fused) {
					host_matmul_generic(es->gate_w, es->gate_type, xb_row, gu_h, I * 2, dim);
					st = moe_activate(backend_host(), act_h, gu_h, gu_h + I, I, es->gate_scale,
									  es->up_scale, use_gelu);
				} else {
					host_matmul_generic(es->gate_w, es->gate_type, xb_row, gate_h, I, dim);
					host_matmul_generic(es->up_w, es->up_type, xb_row, up_h, I, dim);
					st = moe_activate(backend_host(), act_h, gate_h, up_h, I, es->gate_scale,
									  es->up_scale, use_gelu);
				}
				if (st != OK)
					return st;

				host_matmul_generic(es->down_w, es->down_type, act_h, y_h, dim, I);
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
	}

finish:
	buffer *xb2 = batch_slot(ctx->bs, ctx->op->out);
	st			= a->buffer_write_f32(a, xb2, ctx->bs->moe_out.p, ctx->rows.n_rows * dim);
	if (st != OK)
		return st;

	moe_release_batch_slots(ctx->m, ctx->li, ctx->bs, ctx->rows.n_rows, K, n_union);
	return st;
}

typedef struct {
	int			  is_moe;
	int			  sh_inter;
	int			  use_gelu;
	int			  gate_up_fused;
	const buffer *gate_w;
	uint32_t	  gate_type;
	const buffer *up_w;
	uint32_t	  up_type;
	const buffer *down_w;
	uint32_t	  down_type;
	const buffer *gate_up_w;
	uint32_t	  gate_up_type;
} moe_shared_plan;

static int moe_shared_plan_fill(moe_shared_plan *p, struct model *m, int li) {
	int is_moe = model_layer_is_moe(m, li);
	if (is_moe && m->moe.n_shared_experts == 0)
		return 0;
	layer_weights *L = &m->layers[li];
	p->is_moe		 = is_moe;
	p->sh_inter = is_moe ? (m->moe.moe_intermediate * m->moe.n_shared_experts) : m->intermediate;
	p->use_gelu = m->arch_info->uses_gelu_activation;
	p->gate_up_fused = (!is_moe && L->gate_up_fused) ? 1 : 0;
	if (is_moe) {
		p->gate_w		= &L->shexp_gate_w.buf;
		p->gate_type	= L->shexp_gate_w.type;
		p->up_w			= &L->shexp_up_w.buf;
		p->up_type		= L->shexp_up_w.type;
		p->down_w		= &L->shexp_down_w.buf;
		p->down_type	= L->shexp_down_w.type;
		p->gate_up_w	= NULL;
		p->gate_up_type = 0;
	} else {
		p->gate_w		= &L->gate_w.buf;
		p->gate_type	= L->gate_w.type;
		p->up_w			= &L->up_w.buf;
		p->up_type		= L->up_w.type;
		p->down_w		= &L->down_w.buf;
		p->down_type	= L->down_w.type;
		p->gate_up_w	= &L->gate_up_w.buf;
		p->gate_up_type = L->gate_up_w.type;
	}
	return 1;
}

static status_code moe_shared_batch(exec_ctx *ctx) {

	backend		   *a	= exec_layer_backend(ctx);
	const int		dim = ctx->m->dim;
	moe_shared_plan plan;
	if (!moe_shared_plan_fill(&plan, ctx->m, ctx->li))
		return OK;
	int sh_inter = plan.sh_inter;

	buffer *xb = batch_slot(ctx->bs, ctx->op->in[0]);

	const buffer *gate_w_buf = plan.gate_w;
	uint32_t	  gate_wt	 = plan.gate_type;
	const buffer *up_w_buf	 = plan.up_w;
	uint32_t	  up_wt		 = plan.up_type;
	const buffer *down_w_buf = plan.down_w;
	uint32_t	  down_wt	 = plan.down_type;

	if (plan.gate_up_fused) {
		bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE_UP,
					   (size_t)ctx->rows.n_rows * 2 * sh_inter);

		status_code st = a->matmul_batch(a, plan.gate_up_w, plan.gate_up_type, xb,
										 &ctx->bs->pair[RECIPE_SLOT_FFN_GATE_UP].b, 2 * sh_inter,
										 dim, ctx->rows.n_rows);
		if (st != OK)
			return st;

		int use_gelu = plan.use_gelu;
		int n_rows	 = ctx->rows.n_rows;
		if (a->ffn_activate_fused_batch) {
			bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_ACT, (size_t)n_rows * sh_inter);
			status_code ast = a->ffn_activate_fused_batch(
				a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE_UP].b, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b,
				sh_inter, use_gelu ? 1 : 0, n_rows);
			if (ast == OK)
				goto down_batch;
			if (ast != ERR_UNSUPPORTED)
				return ast;
		}
		float *gu_host	= float_buf_ensure(&ctx->bs->moe_exp_gu, (size_t)n_rows * 2 * sh_inter);
		float *act_host = float_buf_ensure(&ctx->bs->moe_exp_act, (size_t)n_rows * sh_inter);
		st				= a->buffer_read_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE_UP].b, gu_host,
											 n_rows * 2 * sh_inter);
		if (st != OK)
			return st;
		for (int row = 0; row < n_rows; row++) {
			const float *gu = gu_host + (size_t)row * 2 * sh_inter;
			float		*o	= act_host + (size_t)row * sh_inter;
			st				= moe_activate(a, o, gu, gu + sh_inter, sh_inter, 1.0f, 1.0f, use_gelu);
			if (st != OK)
				return st;
		}
		float_buf_ensure(&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].fb, (size_t)n_rows * sh_inter);
		st = a->buffer_write_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, act_host,
								 n_rows * sh_inter);
		if (st != OK)
			return st;
	} else {
		bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE, (size_t)ctx->rows.n_rows * sh_inter);
		bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_UP, (size_t)ctx->rows.n_rows * sh_inter);

		status_code st;
		if (a->matmul_multi_batch) {
			const buffer *ws[2]	   = {gate_w_buf, up_w_buf};
			uint32_t	  wts[2]   = {gate_wt, up_wt};
			buffer		 *ys[2]	   = {&ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b,
									  &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b};
			int			  n_out[2] = {sh_inter, sh_inter};
			st = a->matmul_multi_batch(a, ws, wts, xb, ys, n_out, dim, 2, ctx->rows.n_rows);
		} else {
			st = a->matmul_batch(a, gate_w_buf, gate_wt, xb, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b,
								 sh_inter, dim, ctx->rows.n_rows);
			if (st == OK)
				st = a->matmul_batch(a, up_w_buf, up_wt, xb, &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b,
									 sh_inter, dim, ctx->rows.n_rows);
		}
		if (st != OK)
			return st;

		if (a->ffn_activate_batch) {
			int n_rows2 = ctx->rows.n_rows;
			bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_ACT, (size_t)n_rows2 * sh_inter);
			status_code ast = a->ffn_activate_batch(
				a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b, &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b,
				&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, sh_inter, plan.use_gelu ? 1 : 0, n_rows2);
			if (ast == OK)
				goto down_batch;
			if (ast != ERR_UNSUPPORTED)
				return ast;
		}

		int	   use_gelu = plan.use_gelu;
		int	   n_rows2	= ctx->rows.n_rows;
		float *g_host	= float_buf_ensure(&ctx->bs->moe_exp_gu, (size_t)n_rows2 * sh_inter);
		float *u_host	= float_buf_ensure(&ctx->bs->moe_up_scratch, (size_t)n_rows2 * sh_inter);
		float *act_host = float_buf_ensure(&ctx->bs->moe_exp_act, (size_t)n_rows2 * sh_inter);
		st				= a->buffer_read_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b, g_host,
											 n_rows2 * sh_inter);
		if (st == OK)
			st = a->buffer_read_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b, u_host,
									n_rows2 * sh_inter);
		if (st != OK)
			return st;
		for (int row = 0; row < n_rows2; row++) {
			const float *g = g_host + (size_t)row * sh_inter;
			const float *u = u_host + (size_t)row * sh_inter;
			float		*o = act_host + (size_t)row * sh_inter;
			st			   = moe_activate(a, o, g, u, sh_inter, 1.0f, 1.0f, use_gelu);
			if (st != OK)
				return st;
		}
		st = a->buffer_write_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, act_host,
								 n_rows2 * sh_inter);
		if (st != OK)
			return st;
	}

down_batch:
	ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b.size = (size_t)ctx->rows.n_rows * sh_inter * sizeof(float);

	status_code st =
		a->matmul_batch(a, down_w_buf, down_wt, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b,
						&ctx->bs->pair[RECIPE_SLOT_RESID_TMP].b, dim, sh_inter, ctx->rows.n_rows);
	if (st != OK)
		return st;

	if (a->copy_buffer) {
		st = a->copy_buffer(a, &ctx->bs->pair[RECIPE_SLOT_RESID_TMP].b,
							&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, ctx->rows.n_rows * dim);
		if (st != OK)
			return st;
	} else {
		batch_sync(ctx);
		memcpy(batch_buf_ptr(&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b),
			   batch_buf_ptr(&ctx->bs->pair[RECIPE_SLOT_RESID_TMP].b),
			   (size_t)ctx->rows.n_rows * dim * sizeof(float));
	}
	ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b.size = (size_t)ctx->rows.n_rows * dim * sizeof(float);
	return OK;
}

static status_code matmul_multi_batch_body_impl(exec_ctx *ctx);

static status_code matmul_multi_batch_body(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = matmul_multi_batch_body_impl(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code matmul_multi_batch_body_impl(exec_ctx *ctx) {

	backend				  *a  = exec_layer_backend(ctx);
	const layer_weights	  *lw = &ctx->m->layers[ctx->li];
	const layer_ctx_entry *lc = &ctx->m->recipe->layer_ctx[ctx->li];
	int					   k  = ctx->op->u.matmul_multi.k;
	if (ctx->op->w_idx == WIDX_WQ) {
		const int *n_out = ctx->op->u.matmul_multi.n_out;
		if (!lc->has_kv) {
			profile_weight_bytes(&ctx->s->prof, ctx->op->stage, lw->wq.type, n_out[0], k);
			return a->matmul_batch(a, &lw->wq.buf, lw->wq.type, batch_slot(ctx->bs, ctx->op->in[0]),
								   batch_slot(ctx->bs, ctx->op->out), n_out[0], k,
								   ctx->rows.n_rows);
		}
		int n_mats = lc->has_own_v ? 3 : 2;
		if (a->matmul_multi_batch) {
			const buffer *ws[3]	 = {&lw->wq.buf, &lw->wk.buf, &lw->wv.buf};
			uint32_t	  wts[3] = {lw->wq.type, lw->wk.type, lw->wv.type};
			int			  nb	 = n_mats;
			for (int i = 0; i < nb; i++)
				profile_weight_bytes(&ctx->s->prof, ctx->op->stage, wts[i], n_out[i], k);
			buffer	   *ys[3] = {batch_slot(ctx->bs, ctx->op->out),
								 batch_slot(ctx->bs, ctx->op->out + 1),
								 batch_slot(ctx->bs, ctx->op->out + 2)};
			status_code st = a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]),
												   ys, n_out, k, n_mats, ctx->rows.n_rows);
			if (st == OK && !lc->has_own_v)
				st = batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->rows.n_rows);
			return st;
		}
		status_code st =
			a->matmul_batch(a, &lw->wq.buf, lw->wq.type, batch_slot(ctx->bs, ctx->op->in[0]),
							batch_slot(ctx->bs, ctx->op->out), n_out[0], k, ctx->rows.n_rows);
		if (st == OK)
			st = a->matmul_batch(a, &lw->wk.buf, lw->wk.type, batch_slot(ctx->bs, ctx->op->in[0]),
								 batch_slot(ctx->bs, ctx->op->out + 1), n_out[1], k,
								 ctx->rows.n_rows);
		if (st == OK && lc->has_own_v)
			st = a->matmul_batch(a, &lw->wv.buf, lw->wv.type, batch_slot(ctx->bs, ctx->op->in[0]),
								 batch_slot(ctx->bs, ctx->op->out + 2), n_out[2], k,
								 ctx->rows.n_rows);
		if (st == OK && !lc->has_own_v)
			st = batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->rows.n_rows);
		return st;
	}
	if (ctx->op->w_idx == WIDX_WK) {
		int kv_out = lc->kv_row_stride;
		if (!lc->has_own_v) {
			status_code st =
				a->matmul_batch(a, &lw->wk.buf, lw->wk.type, batch_slot(ctx->bs, ctx->op->in[0]),
								&ctx->bs->pair[RECIPE_SLOT_K].b, kv_out, k, ctx->rows.n_rows);
			if (st != OK)
				return st;
			return batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->rows.n_rows);
		}
		if (a->matmul_multi_batch) {
			const buffer *ws[2]	 = {&lw->wk.buf, &lw->wv.buf};
			uint32_t	  wts[2] = {lw->wk.type, lw->wv.type};
			buffer *ys[2]	 = {&ctx->bs->pair[RECIPE_SLOT_K].b, &ctx->bs->pair[RECIPE_SLOT_V].b};
			int		n_out[2] = {kv_out, kv_out};
			profile_weight_bytes(&ctx->s->prof, ctx->op->stage, wts[0], n_out[0], k);
			profile_weight_bytes(&ctx->s->prof, ctx->op->stage, wts[1], n_out[1], k);
			return a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]), ys, n_out,
										 k, 2, ctx->rows.n_rows);
		}
		status_code st =
			a->matmul_batch(a, &lw->wk.buf, lw->wk.type, batch_slot(ctx->bs, ctx->op->in[0]),
							&ctx->bs->pair[RECIPE_SLOT_K].b, kv_out, k, ctx->rows.n_rows);
		if (st != OK)
			return st;
		return a->matmul_batch(a, &lw->wv.buf, lw->wv.type, batch_slot(ctx->bs, ctx->op->in[0]),
							   &ctx->bs->pair[RECIPE_SLOT_V].b, kv_out, k, ctx->rows.n_rows);
	}
	if (ctx->op->w_idx == WIDX_GATE) {
		const int *n_out = ctx->op->u.matmul_multi.n_out;
		if (a->matmul_multi_batch) {
			const buffer *ws[2]	 = {&lw->gate_w.buf, &lw->up_w.buf};
			uint32_t	  wts[2] = {lw->gate_w.type, lw->up_w.type};
			profile_weight_bytes(&ctx->s->prof, ctx->op->stage, wts[0], n_out[0], k);
			profile_weight_bytes(&ctx->s->prof, ctx->op->stage, wts[1], n_out[1], k);
			buffer *ys[2] = {batch_slot(ctx->bs, ctx->op->out),
							 batch_slot(ctx->bs, ctx->op->out + 1)};
			return a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]), ys, n_out,
										 k, 2, ctx->rows.n_rows);
		}
		status_code st = a->matmul_batch(
			a, &lw->gate_w.buf, lw->gate_w.type, batch_slot(ctx->bs, ctx->op->in[0]),
			batch_slot(ctx->bs, ctx->op->out), n_out[0], k, ctx->rows.n_rows);
		if (st == OK)
			st = a->matmul_batch(
				a, &lw->up_w.buf, lw->up_w.type, batch_slot(ctx->bs, ctx->op->in[0]),
				batch_slot(ctx->bs, ctx->op->out + 1), n_out[1], k, ctx->rows.n_rows);
		return st;
	}
	return ERR_INVALID_ARG;
}

static status_code ple_build_batch(exec_ctx *ctx) {

	if (!ctx->m->has_per_layer_embeddings)
		return OK;
	struct model  *m = ctx->m;
	backend		  *a = exec_layer_backend(ctx);
	ple_build_plan plan;
	ple_build_plan_fill(&plan, m, a);
	const int	total_ple = plan.total_ple;
	const int	n_layers  = plan.n_layers;
	const int	dim		  = plan.dim;
	const int	n_rows	  = ctx->rows.n_rows;
	status_code st;

	st = buffer_ensure_scratch(a, &ctx->bs->ple_all, (size_t)n_rows * total_ple * sizeof(float));
	if (st != OK)
		return st;

	int dev_path_ok = ple_build_dev_path_cached(m, a, 1);

	if (dev_path_ok) {
		st = embd_fill_rows(a, &m->layer_dims.per_layer_tok_embd.buf,
							m->layer_dims.per_layer_tok_embd.type, ctx->bs->tokens,
							&ctx->bs->ple_all, n_rows, total_ple, 1.0f);
		if (st != OK)
			return st;
		st = a->scale_inplace(a, &ctx->bs->ple_all, plan.n_embd_sqrt, n_rows * total_ple);
		if (st != OK)
			return st;

		buffer *xb = batch_slot(ctx->bs, RECIPE_SLOT_X);

		size_t n_proj	= (size_t)n_rows * total_ple;
		buffer proj_buf = {0};
		int	   heap		= !backend_has_cap(a, BCAP_IS_HOST);
		float_buf_ensure(&ctx->bs->ple_proj, n_proj);
		if (heap) {
			st = a->buffer_alloc_scratch(a, n_proj * sizeof(float), &proj_buf);
			if (st != OK)
				return st;
		} else {
			proj_buf = bs_cached_view(&ctx->bs->ple_proj, a, n_proj);
		}

		st = a->matmul_batch(a, &m->layer_dims.per_layer_model_proj.buf,
							 m->layer_dims.per_layer_model_proj.type, xb, &proj_buf, total_ple, dim,
							 n_rows);
		if (st != OK)
			goto ple_build_cleanup;

		st = a->scale_inplace(a, &proj_buf, plan.inv_sqrt_ple, n_rows * total_ple);
		if (st != OK)
			goto ple_build_cleanup;

		st = ple_ensure_norm_w(a, ctx->s, m, plan.n_embd_per_layer);
		if (st != OK)
			goto ple_build_cleanup;

		for (int row = 0; row < n_rows; row++) {
			buffer row_view = proj_buf;
			row_view.offset = (size_t)row * total_ple * sizeof(float);
			st				= ple_norm_layers(a, &ctx->s->ple_proj_norm_w, &row_view, n_layers,
											  plan.n_embd_per_layer, plan.eps);
			if (st != OK)
				goto ple_build_cleanup;
		}

		st = a->ple_combine(a, &ctx->bs->ple_all, &proj_buf, n_rows * total_ple,
							0.70710678118654752f);
	ple_build_cleanup:
		ensure_sync(a);
		if (heap)
			a->buffer_free(a, &proj_buf);
		return st;
	}

	ensure_sync(a);

	float_buf_ensure(&ctx->bs->ple_buf, (size_t)n_rows * total_ple);
	float_buf_ensure(&ctx->bs->ple_proj, (size_t)n_rows * total_ple);
	float *ple = ctx->bs->ple_buf.p;
	backend_report_host_fallback(a, "ple_build", HFB_WEIGHT_TYPE,
								 "per-layer token embeddings are dequantized on the host (cpu) "
								 "before the projection runs");
	for (int row = 0; row < n_rows; row++) {
		int token = ctx->bs->tokens ? ctx->bs->tokens[row] : 0;
		st		  = ple_host_decode_row(&plan, token, ple + (size_t)row * total_ple);
		if (st != OK)
			return st;
	}

	float *ple_proj	 = float_buf_ensure(&ctx->s->ple_proj_host, (size_t)n_rows * total_ple);
	float *inpL_host = float_buf_ensure(&ctx->s->inpL_host, (size_t)n_rows * dim);

	buffer	*xb		 = batch_slot(ctx->bs, RECIPE_SLOT_X);
	backend *x_owner = xb->owner;
	st				 = x_owner->buffer_read_f32(x_owner, xb, inpL_host, n_rows * dim);
	if (st != OK)
		return st;

	backend_report_host_fallback(a, "ple_proj_matmul", HFB_OP_NOT_NATIVE,
								 "device backend lacks a batched per-layer projection path; "
								 "projection matmul executes on host (cpu)");
	for (int row = 0; row < n_rows; row++)
		ple_host_project_row(&plan, inpL_host + (size_t)row * dim,
							 ple_proj + (size_t)row * total_ple);

	ple_host_finish(&plan, ple, ple_proj, n_rows);

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
	ple_inject_plan plan;
	if (!ple_inject_plan_fill(&plan, ctx->m, ctx->li))
		return ERR_INVALID_ARG;
	backend		  *a				= exec_layer_backend(ctx);
	const int	   n_embd_per_layer = plan.n_embd_per_layer;
	const int	   total_ple		= plan.total_ple;
	const int	   inj_dim			= plan.inj_dim;
	const int	   n_rows			= ctx->rows.n_rows;
	const float	   eps				= plan.eps;
	batch_scratch *ple_src_bs		= ctx->s->bs;

	float_buf_ensure(&ctx->bs->ple_inp, (size_t)n_rows * n_embd_per_layer);
	float_buf_ensure(&ctx->bs->ple_slice, (size_t)n_rows * n_embd_per_layer);

	status_code st;
	size_t		n_ple	  = (size_t)n_rows * n_embd_per_layer;
	buffer		ple_inp_b = {0}, ple_slice_b = {0};
	int			heap = !backend_has_cap(a, BCAP_IS_HOST);
	if (heap) {
		st = buffer_ensure_scratch(a, &ctx->bs->ple_inp_dev, n_ple * sizeof(float));
		if (st != OK)
			return st;
		st = buffer_ensure_scratch(a, &ctx->bs->ple_slice_dev, n_ple * sizeof(float));
		if (st != OK)
			return st;
		ple_inp_b	= ctx->bs->ple_inp_dev;
		ple_slice_b = ctx->bs->ple_slice_dev;
	} else {
		ple_inp_b	= bs_cached_view(&ctx->bs->ple_inp, a, n_ple);
		ple_slice_b = bs_cached_view(&ctx->bs->ple_slice, a, n_ple);
	}

	if (ple_src_bs->ple_all.handle) {
		for (int row = 0; row < n_rows; row++) {
			buffer ple_src = ple_src_bs->ple_all;
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
			float *ple_row = ple_src_bs->ple_buf.p + (size_t)row * total_ple +
							 (size_t)ctx->li * n_embd_per_layer;
			buffer dst_row = batch_row_view(&ple_slice_b, row, n_embd_per_layer);
			st			   = a->buffer_write_f32(a, &dst_row, ple_row, n_embd_per_layer);
			if (st != OK)
				goto ple_cleanup;
		}
	}

	buffer		 *xb2b = batch_slot(ctx->bs, RECIPE_SLOT_XB2);
	profile_scope pps  = profile_begin(&ctx->s->prof, ctx->op->stage);
	profile_weight_bytes(&ctx->s->prof, ctx->op->stage, plan.gate_type, inj_dim, n_embd_per_layer);
	profile_weight_bytes(&ctx->s->prof, ctx->op->stage, plan.proj_type, n_embd_per_layer, inj_dim);
	st = a->matmul_batch(a, plan.gate_w, plan.gate_type, xb2b, &ple_inp_b, n_embd_per_layer,
						 inj_dim, n_rows);
	if (st != OK) {
		ERROR("ple_proj_inject_batch: matmul_batch(ple_inp_gate) failed (st=%d)", (int)st);
		goto ple_cleanup;
	}

	buffer *aob = batch_slot(ctx->bs, RECIPE_SLOT_ATTN_OUT);
	if (a->matmul_ffn_down_batch) {
		st = a->matmul_ffn_down_batch(a, plan.proj_w, plan.proj_type, &ple_inp_b, &ple_slice_b, aob,
									  inj_dim, n_embd_per_layer, ACTIVATION_GELU, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: matmul_ffn_down_batch failed (st=%d)", (int)st);
			goto ple_cleanup;
		}
	} else {
		st = a->ffn_activate_batch(a, &ple_inp_b, &ple_slice_b, &ple_inp_b, n_embd_per_layer,
								   ACTIVATION_GELU, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: ffn_activate_batch failed (st=%d)", (int)st);
			goto ple_cleanup;
		}
		st = a->matmul_batch(a, plan.proj_w, plan.proj_type, &ple_inp_b, aob, inj_dim,
							 n_embd_per_layer, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: matmul_batch(ple_proj) failed (st=%d)", (int)st);
			goto ple_cleanup;
		}
	}
	profile_end(&ctx->s->prof, &pps);

	if (a->rmsnorm_add_batch) {
		float out_scale = 1.0f;
		if (ctx->m->arch_info->has_layer_output_scale && ctx->li >= 0 &&
			ctx->li < ctx->m->n_layers && ctx->m->layers[ctx->li].layer_out_scale != 1.0f)
			out_scale = ctx->m->layers[ctx->li].layer_out_scale;
		st = a->rmsnorm_add_batch(a, aob, plan.post_norm_w, xb2b, xb2b, inj_dim, eps, out_scale,
								  n_rows);
		if (st != OK)
			ERROR("ple_proj_inject_batch: rmsnorm_add_batch failed (st=%d)", (int)st);
	} else {
		st = a->rmsnorm_batch(a, aob, plan.post_norm_w, aob, inj_dim, eps, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: rmsnorm_batch failed (st=%d)", (int)st);
			goto ple_cleanup;
		}

		st = a->add_batch(a, xb2b, aob, inj_dim, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: add_batch failed (st=%d)", (int)st);
			goto ple_cleanup;
		}
		float out_scale = 1.0f;
		if (ctx->m->arch_info->has_layer_output_scale && ctx->li >= 0 &&
			ctx->li < ctx->m->n_layers && ctx->m->layers[ctx->li].layer_out_scale != 1.0f)
			out_scale = ctx->m->layers[ctx->li].layer_out_scale;
		if (out_scale != 1.0f) {
			if (a->scale_inplace)
				st = a->scale_inplace(a, xb2b, out_scale, inj_dim * n_rows);
			else
				st = scale_inplace_fallback(ctx->s, a, xb2b, inj_dim * n_rows, out_scale);
			if (st != OK)
				ERROR("ple_proj_inject_batch: scale fallback failed (st=%d)", (int)st);
		}
	}

ple_cleanup:
	return st;
}

typedef struct {
	moe_stream_op *op;
	moe_par_job	  *job;
} moe_fused_job;

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

	return copy_k_to_v_buffers(a, kb, vb, kv_out, kv_row_stride, n_tokens);
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

static buffer bs_cached_view(float_buf *fb, backend *owner, size_t n_elems) {
	buffer b   = {0};
	b.handle   = fb->p;
	b.host_ptr = fb->p;
	b.size	   = n_elems * sizeof(float);
	b.owner	   = owner;
	return b;
}

static void mla_scratch_buf_free(buffer *buf, int heap) {
	if (heap)
		free(buf->handle);
}

static buffer *mla_kv_view(const exec_ctx *ctx, kvcache *cache, int *slot) {
	buffer *k, *v;
	kvcache_attn_view(cache, ctx->m, ctx->li, ctx->li, &k, &v, slot);
	return k;
}

static const buffer *mla_host_weight(const weight_ref *w, size_t bytes, backend *host,
									 buffer *view) {
	if (!w->buf.owner || backend_has_cap(w->buf.owner, BCAP_IS_HOST))
		return &w->buf;
	*view = buffer_host_view(host, (void *)w->host_ptr, bytes);
	return view;
}

static status_code mla_kv_put_row(exec_ctx *ctx, kvcache *cache, backend *a, const buffer *kv_a_row,
								  int pos) {
	const model	  *m	   = ctx->m;
	layer_weights *L	   = &m->layers[ctx->li];
	const int	   kv_lora = m->mla.kv_lora;
	const int	   qk_rope = m->mla.qk_rope;
	int			   slot;
	buffer		  *kv = mla_kv_view(ctx, cache, &slot);
	if (kv->owner == a)
		return a->kv_put_mla(a, kv, slot, pos, kv_a_row, &L->kv_a_norm_w.buf, kv_lora, qk_rope,
							 cache->n_ctx, m->norm_eps);

	backend		 *host = op_backend_fallback(a, "kv_put_mla");
	buffer		  norm_view;
	const buffer *norm_w =
		mla_host_weight(&L->kv_a_norm_w, (size_t)kv_lora * sizeof(float), host, &norm_view);
	int			heap;
	buffer		staged = mla_scratch_buf_alloc(NULL, 0, kv_lora + qk_rope, &heap);
	status_code st	   = a->buffer_read_f32(a, kv_a_row, staged.handle, kv_lora + qk_rope);
	if (st == OK)
		st = host->kv_put_mla(host, kv, slot, pos, &staged, norm_w, kv_lora, qk_rope, cache->n_ctx,
							  m->norm_eps);
	mla_scratch_buf_free(&staged, heap);
	return st;
}

static status_code mla_attention_row(exec_ctx *ctx, kvcache *cache, backend *a, const buffer *q,
									 buffer *out, int pos) {
	const model	  *m	   = ctx->m;
	layer_weights *L	   = &m->layers[ctx->li];
	const int	   n_heads = ctx->op->u.attention.n_heads;
	const float	   scale   = ctx->op->u.attention.scale;
	int			   slot;
	buffer		  *kv = mla_kv_view(ctx, cache, &slot);
	if (kv->owner == a)
		return a->attention_mla(a, q, kv, &L->k_b_w.buf, &L->v_b_w.buf, out, slot, pos, n_heads,
								m->mla.qk_head, m->mla.qk_rope, m->mla.qk_nope, m->mla.v_head,
								m->mla.kv_lora, cache->n_ctx, ctx->s->rope_cos, ctx->s->rope_sin,
								L->k_b_w.type, L->v_b_w.type, scale);

	backend		 *host = op_backend_fallback(a, "attention_mla");
	buffer		  k_b_view, v_b_view;
	const buffer *k_b_w = mla_host_weight(
		&L->k_b_w, (size_t)n_heads * m->mla.qk_nope * m->mla.kv_lora * sizeof(float), host,
		&k_b_view);
	const buffer *v_b_w =
		mla_host_weight(&L->v_b_w, (size_t)n_heads * m->mla.kv_lora * m->mla.v_head * sizeof(float),
						host, &v_b_view);
	int			q_n	  = n_heads * m->mla.qk_head;
	int			out_n = n_heads * m->mla.v_head;
	int			q_heap, out_heap;
	buffer		q_host	 = mla_scratch_buf_alloc(NULL, 0, q_n, &q_heap);
	buffer		out_host = mla_scratch_buf_alloc(NULL, 0, out_n, &out_heap);
	status_code st		 = a->buffer_read_f32(a, q, q_host.handle, q_n);
	if (st == OK)
		st = host->attention_mla(host, &q_host, kv, k_b_w, v_b_w, &out_host, slot, pos, n_heads,
								 m->mla.qk_head, m->mla.qk_rope, m->mla.qk_nope, m->mla.v_head,
								 m->mla.kv_lora, cache->n_ctx, ctx->s->rope_cos, ctx->s->rope_sin,
								 L->k_b_w.type, L->v_b_w.type, scale);
	if (st == OK)
		st = a->buffer_write_f32(a, out, out_host.handle, out_n);
	mla_scratch_buf_free(&q_host, q_heap);
	mla_scratch_buf_free(&out_host, out_heap);
	return st;
}

static status_code op_mla_qkv_proj_fused(exec_ctx *ctx) {

	backend		  *a		 = exec_layer_backend(ctx);
	const int	   dim		 = ctx->m->dim;
	layer_weights *L		 = &ctx->m->layers[ctx->li];
	int			   q_lora	 = ctx->m->mla.q_lora;
	int			   kv_lora	 = ctx->m->mla.kv_lora;
	int			   qk_rope	 = ctx->m->mla.qk_rope;
	int			   q_b_rows	 = ctx->m->n_heads * ctx->m->mla.qk_head;
	int			   kv_a_rows = kv_lora + qk_rope;
	buffer		  *xb		 = batch_slot(ctx->bs, ctx->op->in[0]);

	bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE, (size_t)ctx->rows.n_rows * q_lora);
	bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_UP, (size_t)ctx->rows.n_rows * kv_a_rows);
	status_code st	   = OK;
	buffer	   *q_a_b  = &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b;
	buffer	   *kv_a_b = &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b;

	if (a->matmul_multi_batch) {
		const buffer *ws[2]	   = {&L->q_a_w.buf, &L->kv_a_w.buf};
		uint32_t	  wts[2]   = {L->q_a_w.type, L->kv_a_w.type};
		buffer		 *ys[2]	   = {q_a_b, kv_a_b};
		int			  n_out[2] = {q_lora, kv_a_rows};
		st = a->matmul_multi_batch(a, ws, wts, xb, ys, n_out, dim, 2, ctx->rows.n_rows);
	} else {
		st = a->matmul_batch(a, &L->q_a_w.buf, L->q_a_w.type, xb, q_a_b, q_lora, dim,
							 ctx->rows.n_rows);
		if (st == OK)
			st = a->matmul_batch(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, kv_a_b, kv_a_rows, dim,
								 ctx->rows.n_rows);
	}
	if (st != OK)
		goto mla_qkv_cleanup;

	if (q_lora > 0)
		st = a->rmsnorm_batch(a, q_a_b, &L->q_a_norm_w.buf, q_a_b, q_lora, ctx->m->norm_eps,
							  ctx->rows.n_rows);
	if (st != OK)
		goto mla_qkv_cleanup;

	buffer *q_buf = batch_slot(ctx->bs, RECIPE_SLOT_Q);
	st			  = a->matmul_batch(a, &L->q_b_w.buf, L->q_b_w.type, q_a_b, q_buf, q_b_rows, q_lora,
									ctx->rows.n_rows);
	if (st != OK)
		goto mla_qkv_cleanup;

	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer kv_a_row = batch_row_view(kv_a_b, row, kv_a_rows);
		st =
			mla_kv_put_row(ctx, exec_segment_cache(ctx, row), a, &kv_a_row, exec_row_pos(ctx, row));
		if (st != OK)
			goto mla_qkv_cleanup;
	}
mla_qkv_cleanup:
	ensure_sync(a);
	return st;
}

static status_code op_mla_q_proj(exec_ctx *ctx) {

	backend		  *a		= exec_layer_backend(ctx);
	const int	   dim		= ctx->m->dim;
	layer_weights *L		= &ctx->m->layers[ctx->li];
	int			   q_lora	= ctx->m->mla.q_lora;
	int			   q_b_rows = ctx->m->n_heads * ctx->m->mla.qk_head;
	buffer		  *xb		= batch_slot(ctx->bs, ctx->op->in[0]);

	bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE, (size_t)ctx->rows.n_rows * q_lora);
	status_code st	  = OK;
	buffer	   *q_a_b = &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b;

	st = a->matmul_batch(a, &L->q_a_w.buf, L->q_a_w.type, xb, q_a_b, q_lora, dim, ctx->rows.n_rows);
	if (st != OK)
		goto mla_q_cleanup;
	if (q_lora > 0)
		st = a->rmsnorm_batch(a, q_a_b, &L->q_a_norm_w.buf, q_a_b, q_lora, ctx->m->norm_eps,
							  ctx->rows.n_rows);
	if (st != OK)
		goto mla_q_cleanup;
	buffer *q_buf = batch_slot(ctx->bs, RECIPE_SLOT_Q);
	st			  = a->matmul_batch(a, &L->q_b_w.buf, L->q_b_w.type, q_a_b, q_buf, q_b_rows, q_lora,
									ctx->rows.n_rows);
mla_q_cleanup:
	ensure_sync(a);
	return st;
}

static status_code op_mla_kv_proj(exec_ctx *ctx) {

	backend		  *a		 = exec_layer_backend(ctx);
	const int	   dim		 = ctx->m->dim;
	layer_weights *L		 = &ctx->m->layers[ctx->li];
	int			   kv_lora	 = ctx->m->mla.kv_lora;
	int			   qk_rope	 = ctx->m->mla.qk_rope;
	int			   kv_a_rows = kv_lora + qk_rope;
	buffer		  *xb		 = batch_slot(ctx->bs, ctx->op->in[0]);
	bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_UP, (size_t)ctx->rows.n_rows * kv_a_rows);
	status_code st	   = OK;
	buffer	   *kv_a_b = &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b;

	st = a->matmul_batch(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, kv_a_b, kv_a_rows, dim,
						 ctx->rows.n_rows);
	if (st != OK)
		goto mla_kv_cleanup;
	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer kv_a_row = batch_row_view(kv_a_b, row, kv_a_rows);
		st =
			mla_kv_put_row(ctx, exec_segment_cache(ctx, row), a, &kv_a_row, exec_row_pos(ctx, row));
		if (st != OK)
			goto mla_kv_cleanup;
	}
mla_kv_cleanup:
	ensure_sync(a);
	return st;
}

static status_code op_attention_mla(exec_ctx *ctx) {
	backend	 *a		   = exec_layer_backend(ctx);
	const int q_b_rows = ctx->m->n_heads * ctx->m->mla.qk_head;
	const int wo_in	   = ctx->op->u.attention.n_heads * ctx->m->mla.v_head;
	buffer	 *q_buf	   = batch_slot(ctx->bs, ctx->op->in[0]);
	buffer	 *out_buf  = batch_slot(ctx->bs, ctx->op->out);
	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer		q_row	= batch_row_view(q_buf, row, q_b_rows);
		buffer		out_row = batch_row_view(out_buf, row, wo_in);
		status_code st = mla_attention_row(ctx, exec_segment_cache(ctx, row), a, &q_row, &out_row,
										   exec_row_pos(ctx, row));
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code op_split_qgate(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	model		 *m		   = ctx->m;
	backend		 *a		   = exec_layer_backend(ctx);
	profile		 *prof	   = &ctx->s->prof;
	backend		 *t		   = OP_BACKEND(a, split_qgate);
	int			  n_heads  = m->n_heads;
	int			  head_dim = m->head_dim;
	int			  n_rows   = ctx->rows.n_rows;
	buffer		 *mixed	   = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *q		   = exec_slot(ctx, ctx->op->out);
	buffer		 *gate	   = exec_slot(ctx, RECIPE_SLOT_HYB_GATE);
	profile_scope ps	   = profile_begin(prof, ctx->op->stage);
	status_code	  st	   = t->split_qgate(t, mixed, q, gate, n_heads, head_dim, n_rows);
	profile_end(prof, &ps);
	return st;
}

static status_code op_partial_rope_qk(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	model		 *m			 = ctx->m;
	backend		 *a			 = exec_layer_backend(ctx);
	profile		 *prof		 = &ctx->s->prof;
	backend		 *t			 = OP_BACKEND(a, partial_rope_qk);
	int			  n_heads	 = m->n_heads;
	int			  n_kv_heads = m->n_kv_heads;
	int			  head_dim	 = m->head_dim;
	int			  rope_dim	 = m->rope_dim;
	buffer		 *q			 = exec_slot(ctx, RECIPE_SLOT_Q);
	buffer		 *k			 = exec_slot(ctx, RECIPE_SLOT_K);
	profile_scope ps		 = profile_begin(prof, ctx->op->stage);
	status_code	  st		 = OK;
	exec_segments seg		 = exec_segments_of(ctx);
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer q_view = batch_row_view(q, c * seg.rows, n_heads * head_dim);
		buffer k_view = batch_row_view(k, c * seg.rows, n_kv_heads * head_dim);
		int	   pos	  = exec_segment_pos(ctx, c);
		st = t->partial_rope_qk(t, &q_view, &k_view, n_heads, n_kv_heads, head_dim, rope_dim, pos,
								ctx->s->rope_cos, ctx->s->rope_sin, seg.rows);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_attn_output_gate(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	backend		 *a		 = exec_layer_backend(ctx);
	profile		 *prof	 = &ctx->s->prof;
	backend		 *t		 = OP_BACKEND(a, attn_output_gate);
	int			  n		 = ctx->m->n_heads * ctx->m->head_dim;
	int			  n_rows = ctx->rows.n_rows;
	buffer		 *out	 = exec_slot(ctx, ctx->op->in[0]);
	const buffer *gate	 = exec_slot(ctx, RECIPE_SLOT_HYB_GATE);
	profile_scope ps	 = profile_begin(prof, ctx->op->stage);
	status_code	  st	 = t->attn_output_gate(t, out, gate, n, n_rows);
	profile_end(prof, &ps);
	return st;
}

static status_code gdn_run(exec_ctx *ctx, backend *t, kvcache *kc, int n_tokens, bool reset,
						   const float *mixed, const float *z, const float *alpha,
						   const float *beta, float *out, float *ws) {
	model					  *m	  = ctx->m;
	const model_hybrid_params *p	  = &m->hybrid;
	layer_weights			  *L	  = &m->layers[ctx->li];
	const float				  *conv_w = (const float *)L->ssm_conv1d_w.host_ptr;
	const float				  *dt	  = (const float *)L->ssm_dt_b.host_ptr;
	const float				  *a	  = (const float *)L->ssm_a.host_ptr;
	const float				  *norm_w = (const float *)L->ssm_norm_w.host_ptr;
	if (!conv_w || !dt || !a || !norm_w)
		return ERR_FORMAT;
	if (!t->gated_delta_net)
		return ERR_UNSUPPORTED;

	kvcache_hybrid *cache	   = kc->hybrid;
	float		   *conv_state = cache->conv_state + (size_t)ctx->li * cache->conv_stride;
	float		   *state	   = cache->recurrent_state + (size_t)ctx->li * cache->recurrent_stride;
	if (reset) {
		memset(conv_state, 0, cache->conv_stride * sizeof(float));
		memset(state, 0, cache->recurrent_stride * sizeof(float));
	}

	gdn_desc desc = {
		.n_tokens		= n_tokens,
		.n_value_heads	= p->n_value_heads,
		.n_key_heads	= p->n_key_heads,
		.conv_dim		= p->conv_dim,
		.conv_kernel	= p->conv_kernel,
		.key_dim		= p->key_dim,
		.state_size		= p->state_size,
		.value_head_dim = p->value_head_dim,
		.value_dim		= p->value_dim,
		.eps			= m->norm_eps,
		.mixed			= mixed,
		.z				= z,
		.alpha			= alpha,
		.beta			= beta,
		.conv_w			= conv_w,
		.dt				= dt,
		.a_vec			= a,
		.norm_w			= norm_w,
		.conv_state		= conv_state,
		.state			= state,
		.out			= out,
		.ws				= ws,
	};
	return t->gated_delta_net(t, &desc);
}

static status_code op_gated_delta_net(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->rows.cache || !ctx->s || !ctx->rows.cache->hybrid)
		return ERR_INVALID_ARG;
	profile_scope			   ps		= profile_begin(&ctx->s->prof, ctx->op->stage);
	const model_hybrid_params *p		= &ctx->m->hybrid;
	int						   n_tokens = ctx->rows.n_rows > 0 ? ctx->rows.n_rows : 1;
	backend					  *a		= exec_layer_backend(ctx);
	if (!(a && a->gated_delta_net && backend_has_cap(a, BCAP_IS_HOST)))
		backend_report_host_fallback(a, "gated_delta_net", HFB_OP_NOT_NATIVE,
									 "gated delta net executes on the host (cpu) fallback");
	backend *t =
		(a && a->gated_delta_net && backend_has_cap(a, BCAP_IS_HOST)) ? a : op_host_backend();
	if (!t->gated_delta_net) {
		profile_end(&ctx->s->prof, &ps);
		return ERR_UNSUPPORTED;
	}

	tpool *bpool	 = backend_get_pool(t);
	int	   n_threads = bpool ? tpool_n_threads(bpool) : 1;
	if (n_threads < 1)
		n_threads = 1;

	size_t conv_need	= (size_t)n_tokens * p->conv_dim;
	size_t scratch_need = (size_t)n_threads * (2 * p->state_size + 2 * p->value_head_dim);

	int mixed_n = n_tokens * p->conv_dim;
	int z_n		= n_tokens * p->value_dim;
	int alpha_n = n_tokens * p->n_value_heads;
	int beta_n	= n_tokens * p->n_value_heads;
	int out_n	= n_tokens * p->value_dim;

	const float *mixed = recipe_slot_read_f32(ctx, ctx->op->in[0], &ctx->s->hybrid_host, mixed_n);
	const float *z	   = recipe_slot_read_f32(ctx, ctx->op->in[1], &ctx->s->gdn_z_host, z_n);
	const float *alpha =
		recipe_slot_read_f32(ctx, ctx->op->in[2], &ctx->s->gdn_alpha_host, alpha_n);
	const float *beta =
		recipe_slot_read_f32(ctx, RECIPE_SLOT_HYB_BETA, &ctx->s->gdn_beta_host, beta_n);
	float *out = recipe_slot_write_stage(ctx, ctx->op->out, &ctx->s->gdn_out_host, out_n);
	if (!mixed || !z || !alpha || !beta || !out)
		return ERR_INVALID_ARG;

	float		 *ws  = float_buf_ensure(&ctx->s->gdn_ws_host, conv_need + scratch_need);
	exec_segments seg = exec_segments_of(ctx);
	status_code	  st  = OK;
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		size_t row0 = (size_t)c * seg.rows;
		st = gdn_run(ctx, t, exec_segment_cache(ctx, c), seg.rows, exec_segment_pos(ctx, c) == 0,
					 mixed + row0 * p->conv_dim, z + row0 * p->value_dim,
					 alpha + row0 * p->n_value_heads, beta + row0 * p->n_value_heads,
					 out + row0 * p->value_dim, ws);
	}
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

static status_code op_shortconv(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->rows.cache || !ctx->rows.cache->hybrid || !ctx->s)
		return ERR_INVALID_ARG;
	model					  *m	  = ctx->m;
	layer_weights			  *L	  = &m->layers[ctx->li];
	const model_hybrid_params *p	  = &m->hybrid;
	const float				  *conv_w = (const float *)L->ssm_conv1d_w.host_ptr;
	if (!conv_w || p->conv_kernel < 2 || ctx->li < 0 || ctx->li >= m->n_layers)
		return ERR_FORMAT;

	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);

	int n_rows = ctx->rows.n_rows;

	int			 in_stride	= model_hybrid_proj_size(m);
	int			 out_stride = lfm_attn_buf(m);
	int			 mixed_n	= n_rows * in_stride;
	int			 out_n		= n_rows * out_stride;
	const float *mixed = recipe_slot_read_f32(ctx, ctx->op->in[0], &ctx->s->hybrid_host, mixed_n);
	float		*out   = recipe_slot_write_stage(ctx, ctx->op->out, &ctx->s->hybrid_host2, out_n);

	if (!out || !mixed)
		return ERR_INVALID_ARG;

	tpool		 *pool = model_get_pool(m);
	exec_segments seg  = exec_segments_of(ctx);
	for (int c = 0; c < seg.n_calls; c++) {
		kvcache_hybrid *hc	  = exec_segment_cache(ctx, c)->hybrid;
		float		   *state = hc->conv_state + (size_t)ctx->li * hc->conv_stride;
		if (exec_segment_pos(ctx, c) == 0)
			memset(state, 0, (size_t)m->dim * (size_t)(p->conv_kernel - 1) * sizeof(float));
		size_t		  row0 = (size_t)c * seg.rows;
		shortconv_job job  = {
			.out		= out + row0 * out_stride,
			.state		= state,
			.mixed		= mixed + row0 * in_stride,
			.conv_w		= conv_w,
			.dim		= m->dim,
			.kernel		= p->conv_kernel,
			.history	= p->conv_kernel - 1,
			.in_stride	= in_stride,
			.out_stride = out_stride,
			.n_rows		= seg.rows,
		};
		if (pool && m->dim > 8 && tpool_current_tid() < 0)
			tpool_parallel_for(pool, m->dim, 8, shortconv_chunk, &job);
		else
			shortconv_chunk(0, m->dim, -1, &job);
	}

	status_code st = recipe_slot_write_commit(ctx, ctx->op->out, out, out_n);

	profile_end(&ctx->s->prof, &ps);
	return st;
}

static backend_weight weight_view(const weight_ref *w) {
	return (backend_weight){&w->buf, w->type};
}

static status_code op_hc_init(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (!a->hc_init)
		return ERR_UNSUPPORTED;
	const model	 *m		 = ctx->m;
	exec_segments seg	 = exec_segments_of(ctx);
	buffer		 *x_all	 = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *hc_all = exec_slot(ctx, ctx->op->out);
	status_code	  st	 = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int			 row  = call * seg.rows + t;
			buffer		 xrow = batch_row_view(x_all, row, m->dim);
			buffer		 hrow = batch_row_view(hc_all, row, m->kda.hc_count * m->dim);
			hc_init_desc d	  = {.dim = m->dim, .n_hc = m->kda.hc_count, .x = &xrow, .hc = &hrow};
			st				  = a->hc_init(a, &d);
		}
	}
	return st;
}

static status_code op_hc_final(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (!a->hc_final)
		return ERR_UNSUPPORTED;
	const model	 *m		 = ctx->m;
	exec_segments seg	 = exec_segments_of(ctx);
	buffer		 *hc_all = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *x_all	 = exec_slot(ctx, ctx->op->out);
	status_code	  st	 = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int			  row  = call * seg.rows + t;
			buffer		  hrow = batch_row_view(hc_all, row, m->kda.hc_count * m->dim);
			buffer		  xrow = batch_row_view(x_all, row, m->dim);
			hc_final_desc d	   = {.dim = m->dim, .n_hc = m->kda.hc_count, .hc = &hrow, .x = &xrow};
			st				   = a->hc_final(a, &d);
		}
	}
	return st;
}

static status_code op_hc_pre(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (!a->hc_pre)
		return ERR_UNSUPPORTED;
	const model			*m			= ctx->m;
	const layer_weights *L			= &m->layers[ctx->li];
	const bool			 attn		= ctx->op->w_idx == WIDX_HC_ATTN_FN;
	const weight_ref	*fn			= attn ? &L->hc_attn_fn_w : &L->hc_ffn_fn_w;
	const weight_ref	*base		= attn ? &L->hc_attn_base_w : &L->hc_ffn_base_w;
	const weight_ref	*scale		= attn ? &L->hc_attn_scale_w : &L->hc_ffn_scale_w;
	exec_segments		 seg		= exec_segments_of(ctx);
	buffer				*stream_all = exec_slot(ctx, ctx->op->in[0]);
	buffer				*x_all		= exec_slot(ctx, ctx->op->out);
	buffer				*coef_all	= exec_slot(ctx, RECIPE_SLOT_HC_COEF);
	const int			 flat		= m->kda.hc_count * m->dim;
	const int			 coef_elems = m->kda.hc_count * (m->kda.hc_count + 1);
	hc_pre_desc			 d			= {
		.dim			= m->dim,
		.n_hc			= m->kda.hc_count,
		.sinkhorn_iters = m->kda.hc_sinkhorn_iters,
		.norm_eps		= m->norm_eps,
		.eps			= m->kda.hc_eps,
		.fn				= weight_view(fn),
		.base			= &base->buf,
		.scale			= &scale->buf,
	};
	if (backend_has_cap(a, BCAP_IS_HOST))
		d.ws = float_buf_ensure_nocopy(&ctx->s->op_ws, hc_pre_ws_floats(&d), 64);
	status_code st = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int	   row	= call * seg.rows + t;
			buffer srow = batch_row_view(stream_all, row, flat);
			buffer xrow = batch_row_view(x_all, row, m->dim);
			buffer crow = batch_row_view(coef_all, row, coef_elems);
			d.stream	= &srow;
			d.x			= &xrow;
			d.coef		= &crow;
			st			= a->hc_pre(a, &d);
		}
	}
	return st;
}

static status_code op_hc_post(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (!a->hc_post)
		return ERR_UNSUPPORTED;
	const model	 *m			 = ctx->m;
	exec_segments seg		 = exec_segments_of(ctx);
	buffer		 *sub_all	 = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *hc_all	 = exec_slot(ctx, ctx->op->out);
	buffer		 *coef_all	 = exec_slot(ctx, RECIPE_SLOT_HC_COEF);
	const int	  flat		 = m->kda.hc_count * m->dim;
	const int	  coef_elems = m->kda.hc_count * (m->kda.hc_count + 1);
	status_code	  st		 = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int			 row	= call * seg.rows + t;
			buffer		 subrow = batch_row_view(sub_all, row, m->dim);
			buffer		 crow	= batch_row_view(coef_all, row, coef_elems);
			buffer		 hrow	= batch_row_view(hc_all, row, flat);
			hc_post_desc d		= {
				.dim  = m->dim,
				.n_hc = m->kda.hc_count,
				.sub  = &subrow,
				.coef = &crow,
				.hc	  = &hrow,
			};
			st = a->hc_post(a, &d);
		}
	}
	return st;
}

static status_code op_kda(exec_ctx *ctx) {
	backend		   *a = exec_layer_backend(ctx);
	kvcache_hybrid *c = ctx->rows.cache ? ctx->rows.cache->hybrid : NULL;
	if (!a->kda || !c || !c->conv_buf || !c->state_buf)
		return ERR_UNSUPPORTED;
	const model			*m = ctx->m;
	const layer_weights *L = &m->layers[ctx->li];

	kda_desc d = {
		.dim			  = m->dim,
		.n_heads		  = m->n_heads,
		.head_dim		  = m->kda.head_dim,
		.conv_kernel	  = m->hybrid.conv_kernel,
		.gate_lower_bound = m->kda.gate_lower_bound,
		.norm_eps		  = m->norm_eps,
		.wq				  = weight_view(&L->wq),
		.wk				  = weight_view(&L->wk),
		.wv				  = weight_view(&L->wv),
		.wo				  = weight_view(&L->wo),
		.f_a			  = weight_view(&L->kda_f_a_w),
		.f_b			  = weight_view(&L->kda_f_b_w),
		.g_a			  = weight_view(&L->kda_g_a_w),
		.g_b			  = weight_view(&L->kda_g_b_w),
		.beta			  = weight_view(&L->ssm_beta_w),
		.conv_q			  = &L->kda_conv_q_w.buf,
		.conv_k			  = &L->kda_conv_k_w.buf,
		.conv_v			  = &L->kda_conv_v_w.buf,
		.dt_bias		  = &L->ssm_dt_b.buf,
		.ssm_a			  = &L->ssm_a.buf,
		.out_norm		  = &L->ssm_norm_w.buf,
	};
	if (backend_has_cap(a, BCAP_IS_HOST))
		d.ws = float_buf_ensure_nocopy(&ctx->s->op_ws, kda_ws_floats(&d), 64);

	exec_segments seg	  = exec_segments_of(ctx);
	buffer		 *x_all	  = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *out_all = exec_slot(ctx, ctx->op->out);
	status_code	  st	  = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		kvcache		   *rcache = exec_segment_cache(ctx, call);
		kvcache_hybrid *rc	   = rcache ? rcache->hybrid : c;
		if (!rc || !rc->conv_buf || !rc->state_buf)
			return ERR_UNSUPPORTED;
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int	   row	 = call * seg.rows + t;
			buffer xrow	 = batch_row_view(x_all, row, m->dim);
			buffer orow	 = batch_row_view(out_all, row, m->dim);
			d.x			 = &xrow;
			d.out		 = &orow;
			d.conv_state = &rc->conv_buf[ctx->li];
			d.state		 = &rc->state_buf[ctx->li];
			d.reset		 = exec_row_pos(ctx, row) == 0;
			st			 = a->kda(a, &d);
		}
	}
	return st;
}

static status_code op_mla_absorbed(exec_ctx *ctx) {
	backend		   *a = exec_layer_backend(ctx);
	kvcache_hybrid *c = ctx->rows.cache ? ctx->rows.cache->hybrid : NULL;
	if (!a->mla_absorbed || !c || !c->latent_buf)
		return ERR_UNSUPPORTED;
	const model			*m = ctx->m;
	const layer_weights *L = &m->layers[ctx->li];
	if (L->k_b_w.type != GGML_TYPE_F32 || L->v_b_w.type != GGML_TYPE_F32)
		return ERR_UNSUPPORTED;

	tpool *pool		 = backend_get_pool(a);
	int	   n_threads = pool ? tpool_n_threads(pool) : 1;

	mla_absorbed_desc d = {
		.dim	   = m->dim,
		.n_heads   = m->n_heads,
		.kv_lora   = m->mla.kv_lora,
		.qk_head   = m->mla.qk_head,
		.v_head	   = m->mla.v_head,
		.q_lora	   = m->mla.q_lora,
		.n_threads = n_threads < 1 ? 1 : n_threads,
		.norm_eps  = m->norm_eps,
		.q_a	   = weight_view(&L->q_a_w),
		.q_b	   = weight_view(&L->q_b_w),
		.kv_a	   = weight_view(&L->kv_a_w),
		.wo		   = weight_view(&L->wo),
		.q_a_norm  = &L->q_a_norm_w.buf,
		.kv_a_norm = &L->kv_a_norm_w.buf,
		.k_b	   = &L->k_b_w.buf,
		.v_b	   = &L->v_b_w.buf,
	};
	if (backend_has_cap(a, BCAP_IS_HOST))
		d.ws = float_buf_ensure_nocopy(&ctx->s->op_ws, mla_absorbed_ws_floats(&d), 64);

	exec_segments seg	  = exec_segments_of(ctx);
	buffer		 *x_all	  = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *out_all = exec_slot(ctx, ctx->op->out);
	status_code	  st	  = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		kvcache		   *rcache = exec_segment_cache(ctx, call);
		kvcache_hybrid *rc	   = rcache ? rcache->hybrid : c;
		if (!rc || !rc->latent_buf)
			return ERR_UNSUPPORTED;
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int	   row	   = call * seg.rows + t;
			buffer xrow	   = batch_row_view(x_all, row, m->dim);
			buffer orow	   = batch_row_view(out_all, row, m->dim);
			d.latent_cache = &rc->latent_buf[ctx->li];
			d.x			   = &xrow;
			d.out		   = &orow;
			d.pos		   = exec_row_pos(ctx, row);
			st			   = a->mla_absorbed(a, &d);
		}
	}
	return st;
}

static const op_handler g_op_dispatch[OP_KIND_COUNT] = {
	[OP_RMSNORM]			= op_rmsnorm,
	[OP_RMSNORM_PER_HEAD]	= op_rmsnorm_per_head,
	[OP_RMSNORM_NOWEIGHT]	= op_rmsnorm_noweight,
	[OP_RMSNORM_ADD]		= op_rmsnorm_add,
	[OP_MATMUL]				= op_matmul,
	[OP_MATMUL_RESIDUAL]	= op_matmul_residual,
	[OP_MATMUL_MULTI]		= matmul_multi_batch_body,
	[OP_MATMUL_FFN_DOWN]	= op_matmul_ffn_down,
	[OP_ROPE]				= op_rope,
	[OP_ROPE_QK_FUSED]		= op_rope_qk_fused,
	[OP_ROPE_EXT]			= op_rope_ext,
	[OP_KV_PUT]				= op_kv_put,
	[OP_ATTENTION]			= op_attention,
	[OP_ADD]				= op_add,
	[OP_SWAP]				= op_swap,
	[OP_FFN_ACTIVATE]		= op_ffn_activate,
	[OP_FFN_ACTIVATE_EX]	= op_ffn_activate_ex,
	[OP_FFN_ACTIVATE_FUSED] = op_ffn_activate_fused,
	[OP_SCALE]				= op_scale,
	[OP_SOFTCAP]			= op_softcap,
	[OP_LOGITS_READBACK]	= op_logits_readback,
	[OP_PLE_BUILD]			= ple_build_batch,
	[OP_PLE_PROJ_INJECT]	= ple_proj_inject_batch,
	[OP_MLA_Q_PROJ]			= op_mla_q_proj,
	[OP_MLA_KV_PROJ]		= op_mla_kv_proj,
	[OP_MLA_QKV_PROJ_FUSED] = op_mla_qkv_proj_fused,
	[OP_ATTENTION_MLA]		= op_attention_mla,
	[OP_MOE_ROUTER]			= moe_router_batch,
	[OP_MOE_EXPERTS]		= moe_experts_batch,
	[OP_MOE_SHARED]			= moe_shared_batch,
	[OP_SPLIT_QGATE]		= op_split_qgate,
	[OP_PARTIAL_ROPE_QK]	= op_partial_rope_qk,
	[OP_ATTN_OUTPUT_GATE]	= op_attn_output_gate,
	[OP_GATED_DELTA_NET]	= op_gated_delta_net,
	[OP_SHORTCONV]			= op_shortconv,
	[OP_HC_INIT]			= op_hc_init,
	[OP_HC_PRE]				= op_hc_pre,
	[OP_HC_POST]			= op_hc_post,
	[OP_HC_FINAL]			= op_hc_final,
	[OP_KDA]				= op_kda,
	[OP_MLA_ABSORBED]		= op_mla_absorbed,
};

static recipe_op mk_op_base(op_kind kind, uint8_t in0, uint8_t in1, uint8_t in2, uint8_t out,
							uint8_t widx, stage stg);

recipe_op mk_backend_op(op_kind kind, uint8_t in, uint8_t out, uint8_t widx, stage stage) {
	return mk_op_base(kind, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, widx, stage);
}

static int model_has_sliding_layers(const model *m) {
	for (int i = 0; i < m->n_layers; i++)
		if (m->layers[i].is_sliding)
			return 1;
	return 0;
}

static int backend_has_batch_ops(const model *m, const backend *a) {
	if (!(a->matmul_batch && a->rmsnorm_batch && a->add_batch && a->ffn_activate_batch &&
		  a->rope_batch && a->rope_qk_batch && a->attention_batch))
		return 0;
	int has_sliding = m->recipe ? m->recipe->has_sliding_layers : model_has_sliding_layers(m);
	if (has_sliding && !a->attention_swa_batch)
		return 0;
	return 1;
}

static int backend_supports_batch_ops(const model *m) {
	if (!model_mixed_backend_mode(m))
		return backend_has_batch_ops(m, m->backend);
	for (int li = 0; li < m->n_layers; li++)
		if (!backend_has_batch_ops(m, model_layer_backend(m, li)))
			return 0;
	return backend_has_batch_ops(m, m->backend);
}

static status_code batch_carry_rows(compute_scratch *s, const model *m, batch_scratch *from,
									batch_scratch *to, int n_rows) {
	status_code st = compute_copy_buffer_cross(s, batch_slot(from, RECIPE_SLOT_X),
											   batch_slot(to, RECIPE_SLOT_X), n_rows * m->dim);
	if (st != OK || !m->arch_info->has_hyper_connections)
		return st;
	return compute_copy_buffer_cross(s, batch_slot(from, RECIPE_SLOT_HC),
									 batch_slot(to, RECIPE_SLOT_HC),
									 n_rows * m->kda.hc_count * m->dim);
}

static status_code batch_switch_device(model *m, compute_scratch *s, batch_scratch **cur,
									   backend *target, int n_rows) {
	int dev = compute_device_index(s, target);
	if (dev < 0) {
		ERROR("batch walker: backend '%s' is not a device of this model", target->name);
		return ERR_UNSUPPORTED;
	}
	batch_scratch *next = dev == 0 ? s->bs : s->mirror[dev - 1].bs;
	if (next != *cur) {
		status_code st = batch_carry_rows(s, m, *cur, next, n_rows);
		if (st != OK)
			return st;
		*cur = next;
	}
	s->active_backend = target;
	s->active_dev	  = dev;
	return OK;
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
		.rows		= {.cache = cache, .pos_start = pos_start, .n_rows = n_tokens},
		.s			= s,
		.bs			= bs,
		.li			= li,
		.flash_attn = flash_attn,
	};
	return h(&ctx);
}

static status_code batch_run_post_ops(model *m, kvcache *cache, compute_scratch *s,
									  batch_scratch *bs, int n_rows, int flash_attn,
									  float *logits_out) {
	backend			   *a	  = m->backend;
	const model_recipe *r	  = m->recipe;
	const int			total = n_rows * m->vocab_size;
	bs_ensure_slot(bs, a, RECIPE_SLOT_LOGITS, (size_t)total);
	for (int i = 0; i < r->n_post_ops; i++) {
		if (r->post_ops[i].kind == OP_LOGITS_READBACK)
			continue;
		status_code st = exec_op_batch(&r->post_ops[i], m, cache, s, bs, 0, n_rows, -1, flash_attn);
		if (st != OK)
			return st;
	}
	if (!logits_out)
		return OK;
	ensure_sync(a);
	buffer *logits = batch_slot(bs, RECIPE_SLOT_LOGITS);
	return logits->owner->buffer_read_f32(logits->owner, logits, logits_out, total);
}

static int rows_n_logit_rows(const row_desc *rows) {
	if (!rows->needs_logits)
		return rows->n_rows;
	int n = 0;
	for (int i = 0; i < rows->n_rows; i++)
		n += rows->needs_logits[i] != 0;
	return n;
}

static status_code batch_gather_logit_rows(model *m, compute_scratch *s, batch_scratch *bs,
										   const row_desc *rows) {
	if (!rows->needs_logits)
		return OK;
	const int	dim		= m->dim;
	const int	hc_flat = m->arch_info->has_hyper_connections ? m->kda.hc_count * dim : 0;
	int			dst		= 0;
	status_code st		= OK;
	for (int src = 0; src < rows->n_rows && st == OK; src++) {
		if (!rows->needs_logits[src])
			continue;
		if (src != dst) {
			buffer x_src = batch_row_view(batch_slot(bs, RECIPE_SLOT_X), src, dim);
			buffer x_dst = batch_row_view(batch_slot(bs, RECIPE_SLOT_X), dst, dim);
			st			 = compute_copy_buffer_cross(s, &x_src, &x_dst, dim);
			if (st == OK && hc_flat) {
				buffer hc_src = batch_row_view(batch_slot(bs, RECIPE_SLOT_HC), src, hc_flat);
				buffer hc_dst = batch_row_view(batch_slot(bs, RECIPE_SLOT_HC), dst, hc_flat);
				st			  = compute_copy_buffer_cross(s, &hc_src, &hc_dst, hc_flat);
			}
		}
		dst++;
	}
	return st;
}

static batch_scratch *batch_scratch_prepare_dev(model *m, compute_scratch *s, int n_rows, int dev) {
	backend			   *a	 = dev == 0 ? m->backend : s->mirror[dev - 1].be;
	batch_scratch	  **slot = dev == 0 ? &s->bs : &s->mirror[dev - 1].bs;
	const model_recipe *r	 = m->recipe;
	const int			dim	 = m->dim;

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

	if (!*slot)
		*slot = xcalloc(1, sizeof(batch_scratch));
	batch_scratch *bs		 = *slot;
	uint32_t	   slot_mask = r->bs_slot_mask;
	if (!slot_mask)
		slot_mask = 0xffffffffu;
	batch_scratch_alloc(
		bs, a, n_rows, dim, q_out, kv_out, max_inter, attn_buf_size, slot_mask,
		m->arch_info->has_hyper_connections ? m->kda.hc_count * m->dim : 0,
		m->arch_info->has_hyper_connections ? m->kda.hc_count * (m->kda.hc_count + 1) : 0);
	if (m->arch_info->is_hybrid_recurrent) {
		bs_ensure_slot(bs, a, RECIPE_SLOT_HYB_PROJ, (size_t)n_rows * model_hybrid_proj_size(m));
		bs_ensure_slot(bs, a, RECIPE_SLOT_HYB_GATE, (size_t)n_rows * model_hybrid_gate_size(m));
		bs_ensure_slot(bs, a, RECIPE_SLOT_HYB_ALPHA, (size_t)n_rows * m->hybrid.n_value_heads);
		bs_ensure_slot(bs, a, RECIPE_SLOT_HYB_BETA, (size_t)n_rows * m->hybrid.n_value_heads);
	}
	return bs;
}

static batch_scratch *batch_scratch_prepare(model *m, compute_scratch *s, int n_rows) {
	for (int d = 0; d < s->n_mirrors; d++)
		batch_scratch_prepare_dev(m, s, n_rows, d + 1);
	return batch_scratch_prepare_dev(m, s, n_rows, 0);
}

buffer *compute_logits_device(compute_scratch *s) {
	return &s->bs->pair[RECIPE_SLOT_LOGITS].b;
}

void compute_scratch_reserve_rows(model *m, compute_scratch *s, int n_rows) {
	if (n_rows < 1 || !recipe_is_decode_batchable(m))
		return;
	batch_scratch_prepare(m, s, n_rows);
}

static status_code compute_forward_rows(struct model *m, const row_desc *rows,
										struct compute_scratch *s, const int32_t *tokens,
										int flash_attn, float *logits_out) {
	struct kvcache *cache	  = rows->cache;
	const int		n_tokens  = rows->n_rows;
	const int		pos_start = rows->pos_start;
	if (n_tokens < 1)
		return OK;
	if (!recipe_is_decode_batchable(m)) {
		ERROR("model cannot run on the batch walker: a backend holding layers lacks batch ops");
		return ERR_UNSUPPORTED;
	}

	backend			   *a	  = m->backend;
	const model_recipe *r	  = m->recipe;
	const int			dim	  = m->dim;
	const int			n_out = rows_n_logit_rows(rows);

	const int mixed = model_mixed_backend_mode(m);

	status_code st		   = OK;
	int			batch_open = 0;

	if (mixed) {
		st = compute_scratch_ensure_mirror(s, m);
		if (st != OK)
			return st;
	}
	s->active_backend = NULL;
	s->active_dev	  = 0;

	batch_scratch *bs  = batch_scratch_prepare(m, s, n_tokens);
	batch_scratch *cur = bs;
	bs->tokens		   = tokens;

	st = embd_fill_rows(a, &m->tok_embd.buf, m->tok_embd.type, tokens, &bs->pair[RECIPE_SLOT_X].b,
						n_tokens, dim, m->arch_info->has_scale_embeddings ? m->dim_sqrt : 1.0f);
	if (st != OK)
		goto done;

	for (int i = 0; i < r->n_pre_ops; i++) {
		const recipe_op *pop = &r->pre_ops[i];
		if (pop->kind == OP_EMBD_LOOKUP || pop->kind == OP_SCALE_EMBEDDINGS)
			continue;
		st = exec_op_batch(pop, m, cache, s, bs, pos_start, n_tokens, -1, flash_attn);
		if (st != OK) {
			ERROR("batch fast-path: pre_op[%d] kind=%d (stage=%d) failed: %s", i, pop->kind,
				  pop->stage, status_str(st));
			goto done;
		}
	}

	if (a->begin_batch) {
		a->begin_batch(a);
		batch_open = 1;
	}

	const recipe_op *ops_base	= r->per_layer_ops ? r->per_layer_ops : r->layer.ops;
	int				 ops_stride = r->per_layer_ops ? r->layer.n_ops : 0;

	for (int li = 0; li < m->n_layers; li++) {
		if (mixed) {
			st = batch_switch_device(m, s, &cur, model_layer_backend(m, li), n_tokens);
			if (st != OK)
				goto done;
			cur->tokens = tokens;
		}
		const recipe_op *ops = &ops_base[(size_t)li * ops_stride];
		op_walker		 w	 = {
			.m = m, .rows = *rows, .s = s, .bs = cur, .li = li, .flash_attn = flash_attn};
		st = walk_layer_ops(&w, ops, r->layer.n_ops);
		if (st != OK)
			goto done;
	}

	if (mixed) {
		st = batch_switch_device(m, s, &cur, m->backend, n_tokens);
		if (st != OK)
			goto done;
		s->active_backend = NULL;
		s->active_dev	  = 0;
	}

	if (a->end_batch) {
		a->end_batch(a);
		batch_open = 0;
	}

	ensure_sync(a);

	st = batch_gather_logit_rows(m, s, bs, rows);
	if (st != OK)
		goto done;
	st = batch_run_post_ops(m, cache, s, bs, n_out, flash_attn, logits_out);

done:
	s->active_backend = NULL;
	s->active_dev	  = 0;
	if (batch_open && a->end_batch) {
		a->end_batch(a);
	}
	if (st != OK && logits_out)
		memset(logits_out, 0, (size_t)n_out * m->vocab_size * sizeof(float));
	return st;
}

int recipe_is_decode_batchable(const model *m) {
	return m->recipe && backend_supports_batch_ops(m);
}

status_code compute_forward_decode_batch_recipe(struct model *m, struct kvcache **caches,
												struct compute_scratch *s, const int32_t *tokens,
												const int *positions, const uint8_t *needs_logits,
												int n_seqs, int flash_attn, float *logits_out) {
	row_desc rows = {.n_rows	   = n_seqs,
					 .pos_start	   = positions[0],
					 .pos		   = positions,
					 .cache		   = caches[0],
					 .caches	   = caches,
					 .needs_logits = needs_logits};
	return compute_forward_rows(m, &rows, s, tokens, flash_attn, logits_out);
}

status_code compute_forward_batch_recipe(struct model *m, struct kvcache *cache,
										 struct compute_scratch *s, const int32_t *tokens,
										 int n_tokens, int pos_start, int flash_attn,
										 float *logits_out) {
	if (n_tokens < 1)
		return OK;
	uint8_t *last_only		= xcalloc((size_t)n_tokens, 1);
	last_only[n_tokens - 1] = 1;
	row_desc rows			= {
		.n_rows = n_tokens, .pos_start = pos_start, .cache = cache, .needs_logits = last_only};
	status_code st = compute_forward_rows(m, &rows, s, tokens, flash_attn, logits_out);
	free(last_only);
	return st;
}

static recipe_op mk_op_base(op_kind kind, uint8_t in0, uint8_t in1, uint8_t in2, uint8_t out,
							uint8_t widx, stage stg) {
	recipe_op op = {
		.kind  = kind,
		.in	   = {in0, in1, in2},
		.out   = out,
		.w_idx = widx,
		.stage = stg,
	};
	return op;
}

recipe_op mk_rmsnorm(uint8_t in, uint8_t out, uint8_t widx, float eps, stage stage) {
	recipe_op op = mk_op_base(OP_RMSNORM, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, widx, stage);
	op.u.rmsnorm.eps = eps;
	return op;
}

recipe_op mk_rmsnorm_add(uint8_t in, uint8_t residual, uint8_t out, uint8_t widx, float eps,
						 stage stage) {
	recipe_op op	 = mk_op_base(OP_RMSNORM_ADD, in, residual, RECIPE_SLOT_NONE, out, widx, stage);
	op.u.rmsnorm.eps = eps;
	op.u.rmsnorm.out_scale = 1.0f;
	return op;
}

recipe_op mk_matmul(uint8_t in, uint8_t out, uint8_t widx, int n, int k, stage stage) {
	recipe_op op  = mk_op_base(OP_MATMUL, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, widx, stage);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

recipe_op mk_matmul_multi2(uint8_t in, uint8_t out, uint8_t widx, int n0, int n1, int k) {
	recipe_op op = mk_op_base(OP_MATMUL_MULTI, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, widx,
							  STAGE_MATMUL);
	op.u.matmul_multi.n		   = 2;
	op.u.matmul_multi.k		   = k;
	op.u.matmul_multi.n_out[0] = n0;
	op.u.matmul_multi.n_out[1] = n1;
	return op;
}

recipe_op mk_matmul_multi3(uint8_t in, uint8_t out, uint8_t widx, int n0, int n1, int n2, int k) {
	recipe_op op = mk_op_base(OP_MATMUL_MULTI, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, widx,
							  STAGE_MATMUL);
	op.u.matmul_multi.n		   = 3;
	op.u.matmul_multi.k		   = k;
	op.u.matmul_multi.n_out[0] = n0;
	op.u.matmul_multi.n_out[1] = n1;
	op.u.matmul_multi.n_out[2] = n2;
	return op;
}

recipe_op mk_matmul_residual(uint8_t in, uint8_t residual, uint8_t out, uint8_t widx, int n,
							 int k) {
	recipe_op op =
		mk_op_base(OP_MATMUL_RESIDUAL, in, residual, RECIPE_SLOT_NONE, out, widx, STAGE_MATMUL);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

recipe_op mk_matmul_ffn_down(uint8_t gate_in, uint8_t up_in, uint8_t out, uint8_t widx, int n,
							 int k, int activation) {
	recipe_op op =
		mk_op_base(OP_MATMUL_FFN_DOWN, gate_in, up_in, RECIPE_SLOT_NONE, out, widx, STAGE_MATMUL);
	op.u.matmul_ffn_down.n			= n;
	op.u.matmul_ffn_down.k			= k;
	op.u.matmul_ffn_down.activation = activation;
	return op;
}

recipe_op mk_ffn_activate(uint8_t gate_in, uint8_t up_in, uint8_t out, int n, int activation) {
	recipe_op op			= mk_op_base(OP_FFN_ACTIVATE, gate_in, up_in, RECIPE_SLOT_NONE, out,
										 RECIPE_NO_WEIGHT, STAGE_FFN_ACT);
	op.u.ffn_act.n			= n;
	op.u.ffn_act.activation = activation;
	return op;
}

recipe_op mk_ffn_activate_fused(uint8_t in, uint8_t out, int n, int activation) {
	recipe_op op   = mk_op_base(OP_FFN_ACTIVATE_FUSED, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out,
								RECIPE_NO_WEIGHT, STAGE_FFN_ACT);
	op.u.ffn_act.n = n;
	op.u.ffn_act.activation = activation;
	return op;
}

recipe_op mk_rmsnorm_per_head(uint8_t io, uint8_t widx, float eps, int n_heads) {
	recipe_op op = mk_op_base(OP_RMSNORM_PER_HEAD, io, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, io, widx,
							  STAGE_RMSNORM);
	op.u.rmsnorm.eps	 = eps;
	op.u.rmsnorm.n_heads = n_heads;
	return op;
}

recipe_op mk_rmsnorm_noweight(uint8_t in) {
	return mk_op_base(OP_RMSNORM_NOWEIGHT, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_NO_WEIGHT, STAGE_RMSNORM);
}

recipe_op mk_split_qgate(void) {
	return mk_op_base(OP_SPLIT_QGATE, RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_SLOT_Q, RECIPE_NO_WEIGHT, STAGE_MATMUL);
}

recipe_op mk_attn_output_gate(uint8_t attn_in, uint8_t gate_in, uint8_t out) {
	return mk_op_base(OP_ATTN_OUTPUT_GATE, attn_in, gate_in, RECIPE_SLOT_NONE, out,
					  RECIPE_NO_WEIGHT, STAGE_ATTN);
}

recipe_op mk_gated_delta_net(uint8_t proj_in, uint8_t gate_in, uint8_t alpha_in, uint8_t out) {
	return mk_op_base(OP_GATED_DELTA_NET, proj_in, gate_in, alpha_in, out, RECIPE_NO_WEIGHT,
					  STAGE_ATTN);
}

recipe_op mk_shortconv(uint8_t in, uint8_t out) {
	return mk_op_base(OP_SHORTCONV, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, RECIPE_NO_WEIGHT,
					  STAGE_ATTN);
}

recipe_op mk_mla_qkv_proj_fused(uint8_t in, uint8_t out, int n, int k) {
	recipe_op op  = mk_op_base(OP_MLA_QKV_PROJ_FUSED, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out,
							   WIDX_NONE, STAGE_MATMUL);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

recipe_op mk_attention_mla(uint8_t q_in, uint8_t out, int n_heads, int head_dim, int n_ctx,
						   float scale) {
	recipe_op op = mk_op_base(OP_ATTENTION_MLA, q_in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out,
							  WIDX_NONE, STAGE_ATTN);
	op.u.attention.n_heads			 = n_heads;
	op.u.attention.n_kv_heads		 = n_heads;
	op.u.attention.head_dim			 = head_dim;
	op.u.attention.n_ctx			 = n_ctx;
	op.u.attention.scale			 = scale;
	op.u.attention.sliding_window	 = 0;
	op.u.attention.n_kv_heads_active = n_heads;
	return op;
}

recipe_op mk_scale(uint8_t io, uint8_t widx, float scale) {
	recipe_op op = mk_op_base(OP_SCALE, io, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
							  widx, STAGE_ADD);
	op.u.scale.scale = scale;
	return op;
}

recipe_op mk_ple_proj_inject(uint8_t in, uint8_t residual, uint8_t widx) {
	return mk_op_base(OP_PLE_PROJ_INJECT, in, residual, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, widx,
					  STAGE_PLE);
}

recipe_op mk_add(uint8_t in0, uint8_t in1, stage stage) {
	return mk_op_base(OP_ADD, in0, in1, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT,
					  stage);
}

recipe_op mk_swap(uint8_t in0, uint8_t in1, stage stage) {
	return mk_op_base(OP_SWAP, in0, in1, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT,
					  stage);
}

recipe_op mk_kvput(uint8_t k_in, uint8_t v_in) {
	return mk_op_base(OP_KV_PUT, k_in, v_in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT,
					  STAGE_KVPUT);
}

recipe_op mk_attention(uint8_t q_in, uint8_t out, int n_heads, int n_kv_heads, int head_dim,
					   int n_ctx, float scale, int sliding_window) {
	recipe_op op		   = mk_op_base(OP_ATTENTION, q_in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out,
										RECIPE_NO_WEIGHT, STAGE_ATTN);
	op.u.attention.n_heads = n_heads;
	op.u.attention.n_kv_heads		 = n_kv_heads;
	op.u.attention.head_dim			 = head_dim;
	op.u.attention.n_ctx			 = n_ctx;
	op.u.attention.scale			 = scale;
	op.u.attention.sliding_window	 = sliding_window;
	op.u.attention.n_kv_heads_active = n_kv_heads;
	return op;
}

recipe_op mk_attention_default_scale(uint8_t q_in, uint8_t out, int n_heads, int n_kv_heads,
									 int head_dim, int n_ctx, int sliding_window) {
	return mk_attention(q_in, out, n_heads, n_kv_heads, head_dim, n_ctx,
						1.0f / sqrtf((float)head_dim), sliding_window);
}

recipe_op mk_rope(uint8_t in, int n_heads, int head_dim, int rope_neox) {
	recipe_op op = mk_op_base(OP_ROPE, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
							  RECIPE_NO_WEIGHT, STAGE_ROPE);
	op.u.rope.n_heads	= n_heads;
	op.u.rope.head_dim	= head_dim;
	op.u.rope.rope_neox = rope_neox;
	return op;
}

recipe_op mk_rope_qk_fused(int n_heads, int n_kv_heads, int head_dim, int rope_neox) {
	recipe_op op	  = mk_op_base(OP_ROPE_QK_FUSED, RECIPE_SLOT_Q, RECIPE_SLOT_K, RECIPE_SLOT_NONE,
								   RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT, STAGE_ROPE);
	op.u.rope.n_heads = n_heads;
	op.u.rope.n_kv_heads = n_kv_heads;
	op.u.rope.head_dim	 = head_dim;
	op.u.rope.rope_neox	 = rope_neox;
	return op;
}

recipe_op mk_rope_ext(uint8_t in, int rope_neox) {
	recipe_op op = mk_op_base(OP_ROPE_EXT, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
							  RECIPE_NO_WEIGHT, STAGE_ROPE);
	op.u.rope_ext.n_heads		   = 0;
	op.u.rope_ext.head_dim		   = 0;
	op.u.rope_ext.use_freq_factors = 1;
	op.u.rope_ext.rope_neox		   = rope_neox;
	return op;
}

recipe_op mk_partial_rope_qk(void) {
	return mk_op_base(OP_PARTIAL_ROPE_QK, RECIPE_SLOT_Q, RECIPE_SLOT_K, RECIPE_SLOT_NONE,
					  RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT, STAGE_ROPE);
}

static recipe_op mk_embd_lookup(void) {
	return mk_op_base(OP_EMBD_LOOKUP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_SLOT_X, WIDX_TOK_EMBD, STAGE_EMBD);
}

static recipe_op mk_scale_embeddings(void) {
	return mk_op_base(OP_SCALE_EMBEDDINGS, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_SLOT_X, RECIPE_NO_WEIGHT, STAGE_EMBD);
}

static recipe_op mk_ple_build(void) {
	return mk_op_base(OP_PLE_BUILD, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT, STAGE_EMBD);
}

static recipe_op mk_softcap(uint8_t in, float cap) {
	recipe_op op = mk_op_base(OP_SOFTCAP, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
							  RECIPE_NO_WEIGHT, STAGE_LOGITS_READBACK);
	op.u.softcap.cap = cap;
	return op;
}

static recipe_op mk_logits_readback(void) {
	return mk_op_base(OP_LOGITS_READBACK, RECIPE_SLOT_LOGITS, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT, STAGE_LOGITS_READBACK);
}

static recipe_op mk_moe_router(uint8_t in, int n_experts, int k) {
	recipe_op op  = mk_op_base(OP_MOE_ROUTER, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
							   RECIPE_SLOT_ROUTER_IDS, WIDX_FFN_GATE_INP, STAGE_MATMUL);
	op.u.matmul.n = n_experts;
	op.u.matmul.k = k;
	return op;
}

static recipe_op mk_moe_experts(uint8_t in, uint8_t out, int n, int k) {
	recipe_op op = mk_op_base(OP_MOE_EXPERTS, in, RECIPE_SLOT_ROUTER_IDS, RECIPE_SLOT_ROUTER_W, out,
							  WIDX_NONE, STAGE_MATMUL);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

static recipe_op mk_moe_shared(uint8_t in, int n, int k) {
	recipe_op op  = mk_op_base(OP_MOE_SHARED, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
							   RECIPE_SLOT_FFN_ACT, WIDX_NONE, STAGE_MATMUL);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

int recipe_append_dense_ffn(recipe_op *ops, int i, const struct model *m, int li) {
	return recipe_append_dense_ffn_ex(ops, i, m, li, WIDX_FFN_NORM);
}

int recipe_append_dense_ffn_ex(recipe_op *ops, int i, const struct model *m, int li,
							   uint8_t norm_widx) {
	int	  dim			   = m->dim;
	int	  inter			   = m->intermediate;
	float eps			   = m->norm_eps;
	int	  has_matmul_multi = backend_has_cap(m->backend, BCAP_MULTI_MATMUL);
	int	  gate_up_fused	   = li >= 0 && m->layers[li].gate_up_fused;
	int	  act =
		m->arch_info && m->arch_info->uses_gelu_activation ? ACTIVATION_GELU : ACTIVATION_SILU;

	ops[i++] = mk_add(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD);
	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, norm_widx, eps, STAGE_RMSNORM);

	if (gate_up_fused) {
		ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP, 2 * inter, dim,
							 STAGE_MATMUL);
		ops[i++] = mk_ffn_activate_fused(RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_FFN_ACT, inter, act);
	} else {
		if (has_matmul_multi) {
			ops[i++] = mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter,
										inter, dim);
		} else {
			ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter, dim,
								 STAGE_MATMUL);
			ops[i++] =
				mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim, STAGE_MATMUL);
		}
		ops[i++] = mk_ffn_activate(RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_FFN_ACT,
								   inter, act);
	}
	ops[i++] = mk_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim, inter, STAGE_MATMUL);
	ops[i++] = mk_add(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD);
	return i;
}

int recipe_append_moe_ffn(recipe_op *ops, int i, const struct model *m, uint8_t router_in_slot,
						  uint8_t experts_in_slot, uint8_t out_slot) {
	int dim		  = m->dim;
	int moe_inter = m->moe.moe_intermediate;

	ops[i++] = mk_moe_router(router_in_slot, m->moe.n_experts, dim);

	if (m->moe.n_shared_experts > 0) {
		int sh_inter = moe_inter * m->moe.n_shared_experts;
		ops[i++]	 = mk_moe_shared(experts_in_slot, sh_inter, dim);
	}

	ops[i++] = mk_moe_experts(experts_in_slot, out_slot, dim, moe_inter);

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

	ops[i++] = mk_embd_lookup();
	if (m->arch_info->has_scale_embeddings) {
		ops[i++] = mk_scale_embeddings();
	}
	if (m->has_per_layer_embeddings) {
		ops[i++] = mk_ple_build();
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

	if (m->final_logit_softcap > 0.0f) {
		ops[i++] = mk_softcap(RECIPE_SLOT_LOGITS, m->final_logit_softcap);
	}

	ops[i++] = mk_logits_readback();

	r->post_ops	  = ops;
	r->n_post_ops = i;
}
