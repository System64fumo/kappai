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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RECIPE_MAX_ARCHES 32
#define RECIPE_COALESCE_MAX 4
static inline void recipe_unused(int dummy, ...) {
	(void)dummy;
}
#define UNUSED(...) recipe_unused(0, __VA_ARGS__)

static backend *cached_host_backend = NULL;

void recipe_init(void) {
	if (!cached_host_backend)
		cached_host_backend = backend_host();
}

static inline backend *op_host_backend(void) {
	backend *b = cached_host_backend;
	if (!b) {
		b					= backend_host();
		cached_host_backend = b;
	}
	return b;
}
#define OP_BACKEND(field) ((a == op_host_backend() || (a->field)) ? a : op_backend_fallback(a))

static buffer *batch_slot(batch_scratch *bs, uint8_t slot);
static buffer  batch_row_view(const buffer *whole, int row, int row_elems);

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

static status_code op_batch_matmul_impl(exec_ctx *ctx);
static status_code op_batch_matmul_multi_impl(exec_ctx *ctx);
static status_code op_batch_rope_ext_impl(exec_ctx *ctx);
static status_code op_batch_attention_impl(exec_ctx *ctx);
static status_code op_batch_ple_build_impl(exec_ctx *ctx);
static status_code op_batch_ple_proj_inject_impl(exec_ctx *ctx);
static status_code op_batch_moe_router_impl(exec_ctx *ctx);
static status_code op_batch_moe_experts_impl(exec_ctx *ctx);
static status_code op_batch_moe_shared_impl(exec_ctx *ctx);
static status_code op_batch_mla_q_proj_impl(exec_ctx *ctx);
static status_code op_batch_mla_kv_proj_impl(exec_ctx *ctx);
static status_code op_batch_mla_qkv_proj_fused_impl(exec_ctx *ctx);
static status_code op_batch_attention_mla_impl(exec_ctx *ctx);
static status_code op_batch_kv_put_impl(exec_ctx *ctx);
static status_code op_batch_ffn_activate_fused_impl(exec_ctx *ctx);
static status_code op_mla_q_proj_unified(exec_ctx *ctx);
static status_code op_mla_kv_proj_unified(exec_ctx *ctx);
static status_code op_mla_qkv_proj_fused_unified(exec_ctx *ctx);
static status_code op_attention_mla_unified(exec_ctx *ctx);
static status_code op_moe_router_unified(exec_ctx *ctx);
static status_code op_moe_experts_unified(exec_ctx *ctx);
static status_code op_moe_shared_unified(exec_ctx *ctx);

static inline int exec_pos_for_row(const exec_ctx *ctx, int row) {
	return exec_is_batch(ctx) ? ctx->pos_start + row : ctx->pos;
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
					if (op->kind == OP_MATMUL_MULTI && op->w_idx == WIDX_WQ) {
						op->kind		 = OP_MATMUL;
						op->u.matmul.n = lc->q_row_stride;
						op->u.matmul.k = m->dim;
					}
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
					op->u.attention.n_kv_heads		  = m->n_kv_heads;
					op->u.attention.n_kv_heads_active = lc->n_kv_heads;
					op->u.attention.kv_layer		  = lc->kv_layer;
				}
			}
			recipe_coalesce_matmul_runs(dst, n_ops);
		}
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
	if (li >= 0 && li < m->n_layers && m->wrefs_by_layer) {
		weight_ref *w = m->wrefs_by_layer[(size_t)li * WIDX_COUNT + w_idx];
		return w ? w : (weight_ref *)&WEIGHT_REF_NONE;
	}
	layer_weights *L = (li >= 0 && li < m->n_layers) ? &m->layers[li] : NULL;
	switch ((weight_idx)w_idx) {
	case WIDX_TOK_EMBD:
		return (weight_ref *)&m->tok_embd;
	case WIDX_OUTPUT_NORM:
		return (weight_ref *)&m->output_norm_w;
	case WIDX_OUTPUT_W:
		return (weight_ref *)&m->output_w;
	case WIDX_ATTN_NORM:
		return L ? &L->attn_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_WQ:
		return L ? &L->wq : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_WK:
		return L ? &L->wk : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_WV:
		return L ? &L->wv : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_WO:
		return L ? &L->wo : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_FFN_NORM:
		return L ? &L->ffn_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_GATE:
		return L ? &L->gate_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_UP:
		return L ? &L->up_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_DOWN:
		return L ? &L->down_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_GATE_UP:
		return L ? &L->gate_up_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_POST_ATTN_NORM:
		return L ? &L->post_attn_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_POST_FFN_NORM:
		return L ? &L->post_ffn_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_ATTN_Q_NORM:
		return L ? &L->attn_q_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_ATTN_K_NORM:
		return L ? &L->attn_k_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_PLE_POST_NORM:
		return L ? &L->ple_post_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_PLE_INP_GATE:
		return L ? &L->ple_inp_gate_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_PLE_PROJ:
		return L ? &L->ple_proj_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_LAYER_OUT_SCALE:
		return L ? &L->layer_out_scale_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_ROPE_FREQS:
		return m->rope_freqs_count > 0 ? (weight_ref *)&m->rope_freqs_w
									   : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_PER_LAYER_TOK_EMBD:
		return (weight_ref *)&m->layer_dims.per_layer_tok_embd;
	case WIDX_PER_LAYER_MODEL_PROJ:
		return (weight_ref *)&m->layer_dims.per_layer_model_proj;
	case WIDX_PER_LAYER_PROJ_NORM:
		return (weight_ref *)&m->layer_dims.per_layer_proj_norm_w;
	case WIDX_FFN_GATE_INP:
		return L ? &L->router_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_EXP_PROBS_BIAS:
		return L ? &L->router_bias : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_FFN_GATE_INP_S:
		return L ? &L->router_scale_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_FFN_PRE_NORM_2:
		return L ? &L->ffn_pre_norm_2_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_FFN_POST_NORM_1:
		return L ? &L->ffn_post_norm_1_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_FFN_POST_NORM_2:
		return L ? &L->ffn_post_norm_2_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SHEXP_GATE:
		return L ? &L->shexp_gate_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SHEXP_UP:
		return L ? &L->shexp_up_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SHEXP_DOWN:
		return L ? &L->shexp_down_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_MLA_Q_A:
		return L ? &L->q_a_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_MLA_Q_B:
		return L ? &L->q_b_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_MLA_Q_A_NORM:
		return L ? &L->q_a_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_MLA_KV_A:
		return L ? &L->kv_a_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_MLA_K_B:
		return L ? &L->k_b_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_MLA_V_B:
		return L ? &L->v_b_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_MLA_KV_A_NORM:
		return L ? &L->kv_a_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_ATTN_QKV:
		return L ? &L->attn_qkv_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_ATTN_GATE:
		return L ? &L->attn_gate_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SSM_CONV1D:
		return L ? &L->ssm_conv1d_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SSM_DT:
		return L ? &L->ssm_dt_b : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SSM_A:
		return L ? &L->ssm_a : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SSM_BETA:
		return L ? &L->ssm_beta_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SSM_ALPHA:
		return L ? &L->ssm_alpha_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SSM_NORM:
		return L ? &L->ssm_norm_w : (weight_ref *)&WEIGHT_REF_NONE;
	case WIDX_SSM_OUT:
		return L ? &L->ssm_out_w : (weight_ref *)&WEIGHT_REF_NONE;
	default:
		return (weight_ref *)&WEIGHT_REF_NONE;
	}
}

static const buffer *resolve_weight(const model *m, int li, uint8_t w_idx) {
	return &resolve_weight_ref(m, li, w_idx)->buf;
}

static stage matmul_substage(uint8_t w_idx);

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
	const layer_ctx_entry *lc		= &m->recipe->layer_ctx[li];
	int					 k			= op->u.matmul_multi.k;
	int					 n			= lc->has_own_v ? 3 : 2;
	const buffer		*w_list[3]	= {&L->wq.buf, &L->wk.buf, &L->wv.buf};
	uint32_t			 w_types[3] = {L->wq.type, L->wk.type, L->wv.type};
	buffer				*y_list[3]	= {&slots[op->out], &slots[op->out + 1], &slots[op->out + 2]};
	const int			*n_out		= op->u.matmul_multi.n_out;

	if (!a->matmul_multi) {
		ps = profile_begin(prof, STAGE_MATMUL_QKV);
		st = a->matmul(a, w_list[0], w_types[0], &slots[op->in[0]], y_list[0], n_out[0], k);
		if (st == OK)
			st = a->matmul(a, w_list[1], w_types[1], &slots[op->in[0]], y_list[1], n_out[1], k);
		if (st == OK && lc->has_own_v)
			st = a->matmul(a, w_list[2], w_types[2], &slots[op->in[0]], y_list[2], n_out[2], k);
		if (st == OK && !lc->has_own_v)
			st = copy_k_to_v_slot(m, s, li);
		profile_end(prof, &ps);
		return st;
	}

	ps = profile_begin(prof, STAGE_MATMUL_QKV);
	st = a->matmul_multi(a, w_list, w_types, &slots[op->in[0]], y_list, n_out, k, n);
	if (st == OK && !lc->has_own_v)
		st = copy_k_to_v_slot(m, s, li);
	profile_end(prof, &ps);
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
		ps = profile_begin(prof, matmul_substage(op->w_idx));
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
		ps = profile_begin(prof, matmul_substage(op->w_idx));
		st = a->matmul(a, w_list[0], w_types[0], &slots[op->in[0]], y_list[0], kv_out, k);
		if (st == OK)
			st = a->matmul(a, w_list[1], w_types[1], &slots[op->in[0]], y_list[1], kv_out, k);
		profile_end(prof, &ps);
		return st;
	}

	ps = profile_begin(prof, matmul_substage(op->w_idx));
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
		ps = profile_begin(prof, matmul_substage(op->w_idx));
		st = a->matmul(a, w_list[0], w_types[0], &slots[op->in[0]], y_list[0], n_out_local[0], k);
		if (st == OK)
			st = a->matmul(a, w_list[1], w_types[1], &slots[op->in[0]], y_list[1], n_out_local[1],
						   k);
		profile_end(prof, &ps);
		return st;
	}

	ps = profile_begin(prof, matmul_substage(op->w_idx));
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
	case WIDX_PLE_INP_GATE:
	case WIDX_PLE_PROJ:
		return STAGE_PLE_INJECT;
	case WIDX_PER_LAYER_MODEL_PROJ:
		return STAGE_PLE_BUILD;
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

	int			  n = n_ops > RECIPE_COALESCE_MAX ? RECIPE_COALESCE_MAX : n_ops;
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

	int			  n = n_ops > RECIPE_COALESCE_MAX ? RECIPE_COALESCE_MAX : n_ops;
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
	status_code st =
		a->matmul_multi_batch(a, w_list, w_types, batch_slot(bs, ops[0].in[0]), y_list, n_out, k, n,
							  n_rows);
	profile_end(prof, &ps);

	*out_status = st;
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

static status_code op_ple_build_one(exec_ctx *ctx) {
	struct model		   *m	  = ctx->m;
	struct compute_scratch *s	  = ctx->s;
	int						token = ctx->token;
	backend				   *a	  = exec_layer_backend(ctx);
	buffer				   *slots = exec_slots(ctx);
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

	int gpu_path_ok = (a->matmul && a->scale_inplace && a->rmsnorm && a->ple_combine &&
					   m->layer_dims.per_layer_model_proj.buf.handle);

	if (gpu_path_ok) {
		if (s->ple_all.size < (size_t)total_ple * sizeof(float)) {
			if (s->ple_all.owner)
				s->ple_all.owner->buffer_free(s->ple_all.owner, &s->ple_all);
			st = a->buffer_alloc_scratch(a, (size_t)total_ple * sizeof(float), &s->ple_all);
			if (st != OK)
				return st;
		}
		if (s->ple_proj_gpu.size < (size_t)total_ple * sizeof(float)) {
			if (s->ple_proj_gpu.owner)
				s->ple_proj_gpu.owner->buffer_free(s->ple_proj_gpu.owner, &s->ple_proj_gpu);
			st = a->buffer_alloc_scratch(a, (size_t)total_ple * sizeof(float), &s->ple_proj_gpu);
			if (st != OK)
				return st;
		}

		st = a->buffer_write_f32(a, &s->ple_all, ple, total_ple);
		if (st != OK)
			return st;

		buffer *xb = &slots[RECIPE_SLOT_X];
		st = a->matmul(a, &m->layer_dims.per_layer_model_proj.buf,
					   m->layer_dims.per_layer_model_proj.type, xb, &s->ple_proj_gpu, total_ple,
					   ple_dim);
		if (st != OK)
			return st;

		float inv_sqrt_dim = inv_sqrt_ple;
		st				   = a->scale_inplace(a, &s->ple_proj_gpu, inv_sqrt_dim, total_ple);
		if (st != OK)
			return st;

		if (!s->ple_proj_norm_w_uploaded) {
			if (s->ple_proj_norm_w_gpu.size < (size_t)n_embd_per_layer * sizeof(float)) {
				if (s->ple_proj_norm_w_gpu.owner)
					s->ple_proj_norm_w_gpu.owner->buffer_free(s->ple_proj_norm_w_gpu.owner,
															  &s->ple_proj_norm_w_gpu);
				st = a->buffer_alloc_scratch(a, (size_t)n_embd_per_layer * sizeof(float),
											 &s->ple_proj_norm_w_gpu);
				if (st != OK)
					return st;
			}
			st =
				a->buffer_write_f32(a, &s->ple_proj_norm_w_gpu,
									m->layer_dims.per_layer_proj_norm_w.host_ptr, n_embd_per_layer);
			if (st != OK)
				return st;
			s->ple_proj_norm_w_uploaded = 1;
		}

		for (int l = 0; l < n_layers; l++) {
			buffer x_slice = s->ple_proj_gpu;
			x_slice.offset = (size_t)l * n_embd_per_layer * sizeof(float);
			st = a->rmsnorm(a, &x_slice, &s->ple_proj_norm_w_gpu, &x_slice, n_embd_per_layer, eps);
			if (st != OK)
				return st;
		}

		float combine_scale = 0.70710678118654752f;
		st = a->ple_combine(a, &s->ple_all, &s->ple_proj_gpu, total_ple, combine_scale);
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
		if (s->ple_all.size < (size_t)total_ple * sizeof(float)) {
			if (s->ple_all.owner)
				s->ple_all.owner->buffer_free(s->ple_all.owner, &s->ple_all);
			st = a->buffer_alloc_scratch(a, (size_t)total_ple * sizeof(float), &s->ple_all);
			if (st != OK)
				return st;
		}
		st = a->buffer_write_f32(a, &s->ple_all, ple, total_ple);
		if (st != OK)
			return st;
	}

	return OK;
}

static status_code op_ple_build(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, STAGE_PLE_BUILD);
	status_code st = exec_is_batch(ctx) ? op_batch_ple_build_impl(ctx) : op_ple_build_one(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code op_ple_proj_inject_one(exec_ctx *ctx) {
	const struct layer_weights *L =
		(ctx->li >= 0 && ctx->li < ctx->m->n_layers) ? &ctx->m->layers[ctx->li] : NULL;
	const recipe_op		   *op	  = ctx->op;
	struct model		   *m	  = ctx->m;
	struct compute_scratch *s	  = ctx->s;
	int						li	  = ctx->li;
	backend				   *a	  = exec_layer_backend(ctx);
	buffer				   *slots = exec_slots(ctx);
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

	if (ple_slice_buf->size < (size_t)n_embd_per_layer * sizeof(float) ||
		ple_slice_buf->owner != t) {
		if (ple_slice_buf->owner)
			ple_slice_buf->owner->buffer_free(ple_slice_buf->owner, ple_slice_buf);
		memset(ple_slice_buf, 0, sizeof(*ple_slice_buf));
		st = t->buffer_alloc_scratch(t, (size_t)n_embd_per_layer * sizeof(float), ple_slice_buf);
		if (st != OK)
			return st;
	}
	if (ple_inp_buf->size < (size_t)n_embd_per_layer * sizeof(float) || ple_inp_buf->owner != t) {
		if (ple_inp_buf->owner)
			ple_inp_buf->owner->buffer_free(ple_inp_buf->owner, ple_inp_buf);
		memset(ple_inp_buf, 0, sizeof(*ple_inp_buf));
		st = t->buffer_alloc_scratch(t, (size_t)n_embd_per_layer * sizeof(float), ple_inp_buf);
		if (st != OK)
			return st;
	}

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
	st = t->matmul(t, &L->ple_inp_gate_w.buf, L->ple_inp_gate_w.type, xb2b, ple_inp_buf, n_embd_per_layer,
				   inj_dim);
	if (st != OK)
		return st;

	st = t->ffn_activate_ex(t, ple_inp_buf, ple_slice_buf, ple_inp_buf, n_embd_per_layer, 1);
	if (st != OK)
		return st;

	buffer *aob = &slots[op->in[1]];
	st			= t->matmul(t, &L->ple_proj_w.buf, L->ple_proj_w.type, ple_inp_buf, aob, inj_dim,
							n_embd_per_layer);
	if (st != OK)
		return st;

	backend *t2 = OP_BACKEND(rmsnorm);
	st			= t2->rmsnorm(t2, aob, &L->ple_post_norm_w.buf, aob, inj_dim, eps);
	if (st != OK)
		return st;

	backend *t3 = OP_BACKEND(add_inplace);
	st			= t3->add_inplace(t3, xb2b, aob, inj_dim);
	return st;
}

static status_code op_ple_proj_inject(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, STAGE_PLE_INJECT);
	status_code st =
		exec_is_batch(ctx) ? op_batch_ple_proj_inject_impl(ctx) : op_ple_proj_inject_one(ctx);
	profile_end(&ctx->s->prof, &ps);
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
	if (exec_is_batch(ctx))
		return op_batch_rope_ext_impl(ctx);
	const recipe_op		   *op	  = ctx->op;
	struct model		   *m	  = ctx->m;
	struct compute_scratch *s	  = ctx->s;
	int						li	  = ctx->li;
	int						pos	  = ctx->pos;
	backend				   *a	  = exec_layer_backend(ctx);
	profile				   *prof  = &ctx->s->prof;
	buffer				   *slots = exec_slots(ctx);
	profile_scope			ps;
	status_code				st;
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
	ps			  = profile_begin(prof, op->stage);
	backend *t	  = OP_BACKEND(rope_ext);
	backend_sync_rope(t, a);
	if (t->rope_ext) {
		st = t->rope_ext(t, &slots[op->in[0]], n_heads, head_dim, pos, rope_cos, rope_sin,
						 freq_factors);
	} else {
		backend *t2 = OP_BACKEND(rope);
		backend_sync_rope(t2, a);
		st = t2->rope(t2, &slots[op->in[0]], n_heads, head_dim, pos, rope_cos, rope_sin);
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
	buffer		 *kb = kvcache_k_for_layer(cache, m, kv_layer);
	buffer		 *vb = kvcache_v_for_layer(cache, m, kv_layer);
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
			return op_batch_matmul_impl(ctx);
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
		return op_batch_matmul_multi_impl(ctx);
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
		st = op_batch_kv_put_impl(ctx);
	} else {
		st = kvcache_put(ctx->cache, ctx->m, ctx->li, ctx->pos, exec_slot(ctx, ctx->op->in[0]),
						 exec_slot(ctx, ctx->op->in[1]));
	}
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code op_attention(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return op_batch_attention_impl(ctx);
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
		st = op_batch_ffn_activate_fused_impl(ctx);
	} else {
		backend *t	   = OP_BACKEND(ffn_activate);
		buffer	*slots = exec_slots(ctx);
		st = t->ffn_activate(t, &slots[RECIPE_SLOT_FFN_GATE], &slots[RECIPE_SLOT_FFN_UP],
							 &slots[ctx->op->out], n);
	}
	profile_end(prof, &ps);
	return st;
}

static status_code op_softcap(exec_ctx *ctx) {
	float *logits_out = ctx->logits_out;
	if (!logits_out)
		return OK;
	float cap = ctx->op->u.softcap.cap;
	if (cap <= 0.0f)
		return OK;
	int	  vocab	  = ctx->m->vocab_size;
	float inv_cap = 1.0f / cap;
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
		for (int row = 0; row < ctx->n_rows; row++) {
			buffer rowb = batch_row_view(batch_slot(ctx->bs, ctx->op->in[0]), row, row_stride);
			st = t->rmsnorm_per_head(t, &rowb, w, &rowb, n_heads, lc->head_dim, ctx->m->norm_eps);
			if (st != OK)
				break;
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
		if (t->rmsnorm_noweight_per_head) {
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
		if (a->rmsnorm_add && backend_has_cap(a, BCAP_RMSNORM_ADD)) {
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
			for (int row = 0; row < ctx->n_rows; row++) {
				buffer rowb = batch_row_view(exec_slot(ctx, ctx->op->in[0]), row, scale_dim);
				st			= a->scale_inplace(a, &rowb, sc, scale_dim);
				if (st != OK)
					break;
			}
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
	[OP_MLA_Q_PROJ]			 = op_mla_q_proj_unified,
	[OP_MLA_KV_PROJ]		 = op_mla_kv_proj_unified,
	[OP_MLA_QKV_PROJ_FUSED]	 = op_mla_qkv_proj_fused_unified,
	[OP_ATTENTION_MLA]		 = op_attention_mla_unified,
	[OP_MOE_ROUTER]			 = op_moe_router_unified,
	[OP_MOE_EXPERTS]		 = op_moe_experts_unified,
	[OP_MOE_SHARED]			 = op_moe_shared_unified,
	[OP_QWEN_SPLIT_QGATE]	 = op_qwen_split_qgate,
	[OP_QWEN_PARTIAL_ROPE_QK] = op_qwen_partial_rope_qk,
	[OP_QWEN_ATTN_GATE]		 = op_qwen_attn_gate,
	[OP_QWEN_GATED_DELTA_NET] = op_qwen_gated_delta_net,
};

static status_code op_mla_q_proj_unified(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return op_batch_mla_q_proj_impl(ctx);
	return op_mla_q_proj(ctx);
}
static status_code op_mla_kv_proj_unified(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return op_batch_mla_kv_proj_impl(ctx);
	return op_mla_kv_proj(ctx);
}
static status_code op_mla_qkv_proj_fused_unified(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return op_batch_mla_qkv_proj_fused_impl(ctx);
	return op_mla_qkv_proj_fused(ctx);
}
static status_code op_attention_mla_unified(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return op_batch_attention_mla_impl(ctx);
	return op_attention_mla(ctx);
}
static status_code op_moe_router_unified(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return op_batch_moe_router_impl(ctx);
	return op_moe_router(ctx);
}
static status_code op_moe_experts_unified(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return op_batch_moe_experts_impl(ctx);
	return op_moe_experts(ctx);
}
static status_code op_moe_shared_unified(exec_ctx *ctx) {
	if (exec_is_batch(ctx))
		return op_batch_moe_shared_impl(ctx);
	return op_moe_shared(ctx);
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

	int gpu_ple_build = (bk->ple_combine && bk->matmul && bk->scale_inplace && bk->rmsnorm &&
						 m->layer_dims.per_layer_model_proj.buf.handle);
	if (!gpu_ple_build) {
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

struct batch_scratch {
	float_buf x, xb, xb2, attn_out;
	float_buf q, k, v;
	float_buf ffn_gate, ffn_up, ffn_gate_up, ffn_act;
	float_buf resid_tmp;
	float_buf qwen_proj, qwen_gate, qwen_alpha, qwen_beta;
	buffer	  x_b, xb_b, xb2_b, attn_out_b, q_b, k_b, v_b;
	buffer	  ffn_gate_b, ffn_up_b, ffn_gate_up_b, ffn_act_b, resid_tmp_b;
	buffer	  qwen_proj_b, qwen_gate_b, qwen_alpha_b, qwen_beta_b;

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
	int	 moe_router_ids_cap_tokens;
	int	 moe_router_ids_cap_k;

	float_buf moe_expert_out;
	float_buf moe_out;
	float_buf moe_router_logits;
	float_buf moe_router_inp;
	float_buf moe_xb_f;

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
	buffer	  ple_all, ple_proj, ple_slice, ple_inp;
};

static status_code batch_backend_buffer_ensure(buffer *buf, backend *owner, size_t size) {
	if (buf->handle && buf->owner == owner && buf->size >= size)
		return OK;
	if (buf->owner)
		buf->owner->buffer_free(buf->owner, buf);
	memset(buf, 0, sizeof(*buf));
	return owner->buffer_alloc_scratch(owner, size, buf);
}

static void bs_ensure_wrap(batch_scratch *bs, float_buf *fb, buffer *buf, size_t n_elems,
						   backend *ow) {
	(void)bs;
	float_buf_ensure(fb, n_elems);
	buf->handle	  = fb->p;
	buf->host_ptr = fb->p;
	buf->size	  = n_elems * sizeof(float);
	buf->offset	  = 0;
	buf->owner	  = ow;
}

static int batch_scratch_alloc(batch_scratch *bs, backend *owner, int m, int dim, int q_out,
							   int kv_out, int intermediate, int attn_buf_size) {
	int ffn_act_size = intermediate > dim ? intermediate : dim;
	bs_ensure_wrap(bs, &bs->x, &bs->x_b, (size_t)m * attn_buf_size, owner);
	bs_ensure_wrap(bs, &bs->xb, &bs->xb_b, (size_t)m * attn_buf_size, owner);
	bs_ensure_wrap(bs, &bs->xb2, &bs->xb2_b, (size_t)m * attn_buf_size, owner);
	bs_ensure_wrap(bs, &bs->attn_out, &bs->attn_out_b, (size_t)m * attn_buf_size, owner);
	bs_ensure_wrap(bs, &bs->q, &bs->q_b, (size_t)m * q_out, owner);
	bs_ensure_wrap(bs, &bs->k, &bs->k_b, (size_t)m * kv_out, owner);
	bs_ensure_wrap(bs, &bs->v, &bs->v_b, (size_t)m * kv_out, owner);
	bs_ensure_wrap(bs, &bs->ffn_gate, &bs->ffn_gate_b, (size_t)m * intermediate, owner);
	bs_ensure_wrap(bs, &bs->ffn_up, &bs->ffn_up_b, (size_t)m * intermediate, owner);
	bs_ensure_wrap(bs, &bs->ffn_gate_up, &bs->ffn_gate_up_b, (size_t)m * 2 * intermediate, owner);
	bs_ensure_wrap(bs, &bs->ffn_act, &bs->ffn_act_b, (size_t)m * ffn_act_size, owner);
	bs_ensure_wrap(bs, &bs->resid_tmp, &bs->resid_tmp_b, (size_t)m * attn_buf_size, owner);
	return 1;
}

void batch_scratch_free(batch_scratch *bs) {
	if (!bs)
		return;
	free(bs->x.p);
	free(bs->xb.p);
	free(bs->xb2.p);
	free(bs->attn_out.p);
	free(bs->q.p);
	free(bs->k.p);
	free(bs->v.p);
	free(bs->ffn_gate.p);
	free(bs->ffn_up.p);
	free(bs->ffn_gate_up.p);
	free(bs->ffn_act.p);
	free(bs->resid_tmp.p);
	free(bs->qwen_proj.p);
	free(bs->qwen_gate.p);
	free(bs->qwen_alpha.p);
	free(bs->qwen_beta.p);
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
	free(bs->moe_expert_out.p);
	free(bs->moe_out.p);
	free(bs->moe_router_logits.p);
	free(bs->moe_router_inp.p);
	free(bs->moe_xb_f.p);
	free(bs->moe_per_token_slots);
	free(bs->moe_union_slots);
	if (bs->moe_xb_q8_gate.owner)
		bs->moe_xb_q8_gate.owner->buffer_free(bs->moe_xb_q8_gate.owner, &bs->moe_xb_q8_gate);
	free(bs->ple_buf.p);
	if (bs->ple_all.owner)
		bs->ple_all.owner->buffer_free(bs->ple_all.owner, &bs->ple_all);
	if (bs->ple_proj.owner)
		bs->ple_proj.owner->buffer_free(bs->ple_proj.owner, &bs->ple_proj);
	if (bs->ple_slice.owner)
		bs->ple_slice.owner->buffer_free(bs->ple_slice.owner, &bs->ple_slice);
	if (bs->ple_inp.owner)
		bs->ple_inp.owner->buffer_free(bs->ple_inp.owner, &bs->ple_inp);
	memset(bs, 0, sizeof(*bs));
}

static buffer *batch_slot(batch_scratch *bs, uint8_t slot) {
	switch (slot) {
	case RECIPE_SLOT_X:
		return &bs->x_b;
	case RECIPE_SLOT_XB:
		return &bs->xb_b;
	case RECIPE_SLOT_XB2:
		return &bs->xb2_b;
	case RECIPE_SLOT_ATTN_OUT:
		return &bs->attn_out_b;
	case RECIPE_SLOT_Q:
		return &bs->q_b;
	case RECIPE_SLOT_K:
		return &bs->k_b;
	case RECIPE_SLOT_V:
		return &bs->v_b;
	case RECIPE_SLOT_FFN_GATE:
		return &bs->ffn_gate_b;
	case RECIPE_SLOT_FFN_UP:
		return &bs->ffn_up_b;
	case RECIPE_SLOT_FFN_ACT:
		return &bs->ffn_act_b;
	case RECIPE_SLOT_FFN_GATE_UP:
		return &bs->ffn_gate_up_b;
	case RECIPE_SLOT_RESID_TMP:
		return &bs->resid_tmp_b;
	case RECIPE_SLOT_QWEN_PROJ:
		return &bs->qwen_proj_b;
	case RECIPE_SLOT_QWEN_GATE:
		return &bs->qwen_gate_b;
	case RECIPE_SLOT_QWEN_ALPHA:
		return &bs->qwen_alpha_b;
	case RECIPE_SLOT_QWEN_BETA:
		return &bs->qwen_beta_b;
	default:
		return NULL;
	}
}

static buffer batch_row_view(const buffer *whole, int row, int row_elems) {
	buffer v = *whole;
	v.offset = whole->offset + ((size_t)row * row_elems * sizeof(float));
	return v;
}

static inline float *batch_buf_ptr(const buffer *b) {
	return (float *)((char *)b->handle + b->offset);
}

static const op_handler g_op_dispatch[OP_KIND_COUNT];

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

static void moe_batch_expert_chunk(int begin, int end, int tid, void *ctx) {
	moe_batch_job *j	   = (moe_batch_job *)ctx;
	float		  *scratch = j->all_scratch + (size_t)tid * j->per_thread_scratch;
	float		  *gate_h  = scratch;
	float		  *up_h	   = gate_h + j->inter;
	float		  *act_h   = up_h + j->inter;
	float		  *y_h	   = act_h + j->inter;
	float		  *gu_h	   = y_h + j->dim;
	float		  *out	   = j->all_outs + (size_t)tid * j->per_thread_out;

	for (int idx = begin; idx < end; idx++) {
		int row = idx / j->top_k;
		int k	= idx % j->top_k;

		moe_expert_slot *es = &j->slot_buf[(size_t)row * j->top_k + k];
		if (es->eid < 0 || !es->gate_w)
			continue;

		const float *xb_row = j->xb_base + (size_t)row * j->dim;

		if (es->gate_up_fused) {
			j->a->matmul_thread_local(j->a, es->gate_w, es->gate_type, xb_row, gu_h, j->inter * 2,
									  j->dim, tid);
			moe_activate(act_h, gu_h, gu_h + j->inter, j->inter, es->gate_scale, es->up_scale,
						 j->use_gelu);
		} else {
			j->a->matmul_thread_local(j->a, es->gate_w, es->gate_type, xb_row, gate_h, j->inter,
									  j->dim, tid);
			j->a->matmul_thread_local(j->a, es->up_w, es->up_type, xb_row, up_h, j->inter, j->dim,
									  tid);
			moe_activate(act_h, gate_h, up_h, j->inter, es->gate_scale, es->up_scale, j->use_gelu);
		}

		j->a->matmul_thread_local(j->a, es->down_w, es->down_type, act_h, y_h, j->dim, j->inter,
								  tid);
		if (es->down_scale != 1.0f) {
			float ds = es->down_scale;
			for (int d = 0; d < j->dim; d++)
				y_h[d] *= ds;
		}

		float  w	   = j->bs->moe_router_w[(size_t)row * j->top_k + k];
		float *out_row = out + (size_t)row * j->dim;
		for (int d = 0; d < j->dim; d++)
			out_row[d] += w * y_h[d];
	}
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
	if (m->has_per_layer_embeddings &&
		!(a->scale_inplace && a->ple_combine && a->buffer_write_f32 && a->copy_buffer))
		return 0;
	if (model_has_sliding_layers(m) && !a->attention_swa_batch)
		return 0;
	return 1;
}

static status_code batch_copy_k_to_v(model *m, batch_scratch *bs, int li, int n_tokens) {
	const layer_ctx_entry *lc			 = &m->recipe->layer_ctx[li];
	int					   kv_out		 = lc->kv_row_stride;
	backend				  *a			 = m->backend;
	buffer				  *kb			 = &bs->k_b;
	buffer				  *vb			 = &bs->v_b;
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
		const float *k = (const float *)kb->host_ptr + (size_t)row * kv_row_stride;
		float		*v = (float *)vb->host_ptr + (size_t)row * kv_row_stride;
		memcpy(v, k, (size_t)kv_out * sizeof(float));
	}
	return OK;
}

static status_code op_batch_matmul_impl(exec_ctx *ctx) {

	if (ctx->op->w_idx == WIDX_WV && ctx->li >= 0 && ctx->li < ctx->m->n_layers &&
		!ctx->m->recipe->layer_ctx[ctx->li].has_own_v)
		return batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->n_rows);
	backend	   *a	 = exec_layer_backend(ctx);
	weight_ref *wref = resolve_weight_ref(ctx->m, ctx->li, ctx->op->w_idx);
	return a->matmul_batch(a, &wref->buf, wref->type, batch_slot(ctx->bs, ctx->op->in[0]),
						   batch_slot(ctx->bs, ctx->op->out), ctx->op->u.matmul.n,
						   ctx->op->u.matmul.k, ctx->n_rows);
}

static status_code op_batch_matmul_multi_impl(exec_ctx *ctx) {

	backend				  *a  = exec_layer_backend(ctx);
	const layer_weights	  *lw = &ctx->m->layers[ctx->li];
	const layer_ctx_entry *lc = &ctx->m->recipe->layer_ctx[ctx->li];
	int					   k  = ctx->op->u.matmul_multi.k;
	if (ctx->op->w_idx == WIDX_WQ) {
		const int *n_out = ctx->op->u.matmul_multi.n_out;
		int		   n_w	 = lc->has_own_v ? 3 : 2;
		if (a->matmul_multi_batch) {
			const buffer *ws[3]	 = {&lw->wq.buf, &lw->wk.buf, &lw->wv.buf};
			uint32_t	  wts[3] = {lw->wq.type, lw->wk.type, lw->wv.type};
			buffer		 *ys[3]	 = {batch_slot(ctx->bs, ctx->op->out),
									batch_slot(ctx->bs, ctx->op->out + 1),
									batch_slot(ctx->bs, ctx->op->out + 2)};
			status_code st =
				a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]), ys, n_out, k,
									  n_w, ctx->n_rows);
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
								&ctx->bs->k_b, kv_out, k, ctx->n_rows);
			if (st != OK)
				return st;
			return batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->n_rows);
		}
		if (a->matmul_multi_batch) {
			const buffer *ws[2]	   = {&lw->wk.buf, &lw->wv.buf};
			uint32_t	  wts[2]   = {lw->wk.type, lw->wv.type};
			buffer		 *ys[2]	   = {&ctx->bs->k_b, &ctx->bs->v_b};
			int			  n_out[2] = {kv_out, kv_out};
			return a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]), ys, n_out,
										 k, 2, ctx->n_rows);
		}
		status_code st =
			a->matmul_batch(a, &lw->wk.buf, lw->wk.type, batch_slot(ctx->bs, ctx->op->in[0]),
							&ctx->bs->k_b, kv_out, k, ctx->n_rows);
		if (st != OK)
			return st;
		return a->matmul_batch(a, &lw->wv.buf, lw->wv.type, batch_slot(ctx->bs, ctx->op->in[0]),
							   &ctx->bs->v_b, kv_out, k, ctx->n_rows);
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

static status_code op_batch_kv_put_impl(exec_ctx *ctx) {

	int kv_row_stride = ctx->m->recipe->layer_ctx[ctx->li].kv_row_stride;
	for (int row = 0; row < ctx->n_rows; row++) {
		buffer		krow = batch_row_view(batch_slot(ctx->bs, ctx->op->in[0]), row, kv_row_stride);
		buffer		vrow = batch_row_view(batch_slot(ctx->bs, ctx->op->in[1]), row, kv_row_stride);
		status_code st =
			kvcache_put(ctx->cache, ctx->m, ctx->li, ctx->pos_start + row, &krow, &vrow);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code op_batch_attention_impl(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	int		 kv_layer =
		ctx->m->arch_info->has_variable_layer_dims ? ctx->op->u.attention.kv_layer : ctx->li;
	int sliding_window = ctx->op->u.attention.sliding_window;
	int use_swa		   = (sliding_window > 0) && (ctx->li >= 0) &&
						 model_layer_is_sliding(ctx->m, ctx->li) && a->attention_swa_batch != NULL;
	if (use_swa)
		return a->attention_swa_batch(a, batch_slot(ctx->bs, ctx->op->in[0]), &ctx->cache->k,
									  &ctx->cache->v, batch_slot(ctx->bs, ctx->op->out), kv_layer,
									  ctx->pos_start, ctx->op->u.attention.n_heads,
									  ctx->op->u.attention.n_kv_heads,
									  ctx->op->u.attention.head_dim, ctx->cache->n_ctx,
									  ctx->flash_attn, ctx->op->u.attention.scale, sliding_window,
									  ctx->op->u.attention.n_kv_heads_active, ctx->n_rows);
	return a->attention_batch(a, batch_slot(ctx->bs, ctx->op->in[0]), &ctx->cache->k,
							  &ctx->cache->v, batch_slot(ctx->bs, ctx->op->out), kv_layer,
							  ctx->pos_start, ctx->op->u.attention.n_heads,
							  ctx->op->u.attention.n_kv_heads, ctx->op->u.attention.head_dim,
							  ctx->cache->n_ctx, ctx->flash_attn, ctx->op->u.attention.scale,
							  ctx->op->u.attention.n_kv_heads_active, ctx->n_rows);
}

static status_code op_batch_ffn_activate_fused_impl(exec_ctx *ctx) {

	int			 n			  = ctx->op->u.ffn_act.n;
	int			 act		  = ctx->op->u.ffn_act.activation;
	const float *fused		  = batch_buf_ptr(batch_slot(ctx->bs, ctx->op->in[0]));
	float		*out		  = batch_buf_ptr(batch_slot(ctx->bs, ctx->op->out));
	const int	 fused_stride = 2 * n;
	for (int row = 0; row < ctx->n_rows; row++) {
		const float *g = fused + (size_t)row * fused_stride;
		const float *u = g + n;
		float		*o = out + (size_t)row * n;
		if (act == 1) {
			for (int i = 0; i < n; i++)
				o[i] = 0.5f * g[i] * (1.0f + erff(g[i] * 0.7071067811865475f)) * u[i];
		} else {
			for (int i = 0; i < n; i++) {
				float gv = g[i];
				o[i]	 = gv / (1.0f + expf(-gv)) * u[i];
			}
		}
	}
	return OK;
}

static status_code op_batch_moe_shared_impl(exec_ctx *ctx) {

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
		bs_ensure_wrap(ctx->bs, &ctx->bs->ffn_gate_up, &ctx->bs->ffn_gate_up_b,
					   (size_t)ctx->n_rows * 2 * sh_inter, a);

		status_code st = a->matmul_batch(a, &L->gate_up_w.buf, L->gate_up_w.type, xb,
										 &ctx->bs->ffn_gate_up_b, 2 * sh_inter, dim, ctx->n_rows);
		if (st != OK)
			return st;

		int use_gelu = ctx->m->arch_info->uses_gelu_activation;
		float_buf_ensure(&ctx->bs->ffn_act, (size_t)ctx->n_rows * sh_inter);
		for (int row = 0; row < ctx->n_rows; row++) {
			float *gu = ctx->bs->ffn_gate_up.p + (size_t)row * 2 * sh_inter;
			float *o  = ctx->bs->ffn_act.p + (size_t)row * sh_inter;
			moe_activate(o, gu, gu + sh_inter, sh_inter, 1.0f, 1.0f, use_gelu);
		}
	} else {
		bs_ensure_wrap(ctx->bs, &ctx->bs->ffn_gate, &ctx->bs->ffn_gate_b,
					   (size_t)ctx->n_rows * sh_inter, a);
		bs_ensure_wrap(ctx->bs, &ctx->bs->ffn_up, &ctx->bs->ffn_up_b,
					   (size_t)ctx->n_rows * sh_inter, a);

		status_code st = a->matmul_batch(a, gate_w_buf, gate_wt, xb, &ctx->bs->ffn_gate_b, sh_inter,
										 dim, ctx->n_rows);
		if (st != OK)
			return st;
		st =
			a->matmul_batch(a, up_w_buf, up_wt, xb, &ctx->bs->ffn_up_b, sh_inter, dim, ctx->n_rows);
		if (st != OK)
			return st;

		int use_gelu = ctx->m->arch_info->uses_gelu_activation;
		float_buf_ensure(&ctx->bs->ffn_act, (size_t)ctx->n_rows * sh_inter);
		for (int row = 0; row < ctx->n_rows; row++) {
			float *g = ctx->bs->ffn_gate.p + (size_t)row * sh_inter;
			float *u = ctx->bs->ffn_up.p + (size_t)row * sh_inter;
			float *o = ctx->bs->ffn_act.p + (size_t)row * sh_inter;
			moe_activate(o, g, u, sh_inter, 1.0f, 1.0f, use_gelu);
		}
	}

	bs_ensure_wrap(ctx->bs, &ctx->bs->resid_tmp, &ctx->bs->resid_tmp_b, (size_t)ctx->n_rows * dim,
				   a);

	ctx->bs->ffn_act_b.size = (size_t)ctx->n_rows * sh_inter * sizeof(float);

	status_code st = a->matmul_batch(a, down_w_buf, down_wt, &ctx->bs->ffn_act_b,
									 &ctx->bs->resid_tmp_b, dim, sh_inter, ctx->n_rows);
	if (st != OK)
		return st;

	memcpy(ctx->bs->ffn_act.p, ctx->bs->resid_tmp.p, (size_t)ctx->n_rows * dim * sizeof(float));
	ctx->bs->ffn_act_b.size = (size_t)ctx->n_rows * dim * sizeof(float);
	return OK;
}

static status_code op_batch_moe_router_impl(exec_ctx *ctx) {

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

	buffer *xb	 = batch_slot(ctx->bs, ctx->op->in[0]);
	float  *xb_f = batch_buf_ptr(xb);

	for (int row = 0; row < ctx->n_rows; row++) {
		const float *router_input_src = xb_f + (size_t)row * router_dim;
		float		*router_input	  = ctx->bs->moe_router_inp.p + (size_t)row * router_dim;
		memcpy(router_input, router_input_src, (size_t)router_dim * sizeof(float));

		if (uses_softmax)
			moe_router_normalize_input(ctx->m, L, router_dim, router_input);
	}

	buffer router_inp_buf	   = {0};
	router_inp_buf.handle	   = ctx->bs->moe_router_inp.p;
	router_inp_buf.host_ptr	   = ctx->bs->moe_router_inp.p;
	router_inp_buf.size		   = (size_t)ctx->n_rows * router_dim * sizeof(float);
	router_inp_buf.owner	   = a;
	buffer router_logits_buf   = {0};
	router_logits_buf.handle   = ctx->bs->moe_router_logits.p;
	router_logits_buf.host_ptr = ctx->bs->moe_router_logits.p;
	router_logits_buf.size	   = (size_t)ctx->n_rows * E * sizeof(float);
	router_logits_buf.owner	   = a;

	status_code st = a->matmul_batch(a, &L->router_w.buf, L->router_w.type, &router_inp_buf,
									 &router_logits_buf, E, router_dim, ctx->n_rows);
	if (st != OK)
		return st;

	for (int row = 0; row < ctx->n_rows; row++) {
		float		*logits	 = ctx->bs->moe_router_logits.p + (size_t)row * E;
		int			*row_ids = ctx->bs->moe_router_ids + (size_t)row * K;
		float		*row_w	 = ctx->bs->moe_router_w + (size_t)row * K;
		const float *bias	 = (const float *)L->router_bias.host_ptr;
		int K_row = moe_router_emit(E, K, uses_softmax, norm_topk, routed_scale, logits, bias,
									ctx->bs->moe_router_logits.p + (size_t)ctx->n_rows * E +
										(size_t)row * E,
									row_ids, row_w);
		for (int k = K_row; k < K; k++) {
			row_ids[k] = -1;
			row_w[k]   = 0.0f;
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
		for (int u = 0; u < n_union; u++)
			ctx->bs->moe_union_order[u] = u;
		for (int i = 1; i < n_union; i++) {
			int ord = ctx->bs->moe_union_order[i];
			int cnt = ctx->bs->moe_union_count[ord];
			int j	= i - 1;
			while (j >= 0 && ctx->bs->moe_union_count[ctx->bs->moe_union_order[j]] < cnt) {
				ctx->bs->moe_union_order[j + 1] = ctx->bs->moe_union_order[j];
				j--;
			}
			ctx->bs->moe_union_order[j + 1] = ord;
		}
		for (int i = 0; i < n_union; i++)
			ctx->bs->moe_union_sorted_ids[i] = ctx->bs->moe_union_ids[ctx->bs->moe_union_order[i]];
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
		ctx->bs->moe_union_slots	 = xmalloc((size_t)n_union * sizeof(moe_expert_slot));
		ctx->bs->moe_union_slots_cap = n_union;
	}

	moe_stream_resolve(ctx->m, ctx->li, ctx->bs->moe_union_sorted_ids, n_union,
					   ctx->bs->moe_union_slots);
	ctx->bs->moe_union_pending_n = n_union;
	return OK;
}

static status_code op_batch_moe_experts_impl(exec_ctx *ctx) {

	backend	 *a	  = exec_layer_backend(ctx);
	const int dim = ctx->m->dim;
	if (!model_layer_is_moe(ctx->m, ctx->li))
		return OK;

	int K		 = ctx->m->moe.n_experts_used;
	int I		 = ctx->m->moe.moe_intermediate;
	int use_gelu = ctx->m->arch_info->uses_gelu_activation;

	int n_union = ctx->bs->moe_union_pending_n;
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
		job.m				   = ctx->m;
		job.a				   = ctx->m->backend;
		job.bs				   = ctx->bs;
		job.slot_buf		   = ctx->bs->moe_per_token_slots;
		job.li				   = ctx->li;
		job.top_k			   = K;
		job.inter			   = I;
		job.dim				   = dim;
		job.n_tokens		   = ctx->n_rows;
		job.use_gelu		   = use_gelu;
		job.xb_base			   = batch_buf_ptr(batch_slot(ctx->bs, ctx->op->in[0]));
		job.all_scratch		   = all_scratch;
		job.all_outs		   = all_outs;
		job.per_thread_scratch = per_thread_scratch;
		job.per_thread_out	   = per_thread_out;

		tpool_parallel_for(pool, total_experts, 1, moe_batch_expert_chunk, &job);

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

		for (int row = 0; row < ctx->n_rows; row++) {
			const float *xb_row =
				batch_buf_ptr(batch_slot(ctx->bs, ctx->op->in[0])) + (size_t)row * dim;
			for (int k = 0; k < K; k++) {
				moe_expert_slot *es = &ctx->bs->moe_per_token_slots[(size_t)row * K + k];
				if (es->eid < 0 || !es->gate_w)
					continue;

				if (es->gate_up_fused) {
					a->matmul_thread_local(a, es->gate_w, es->gate_type, xb_row, gu_h, I * 2, dim,
										   0);
					moe_activate(act_h, gu_h, gu_h + I, I, es->gate_scale, es->up_scale, use_gelu);
				} else {
					a->matmul_thread_local(a, es->gate_w, es->gate_type, xb_row, gate_h, I, dim, 0);
					a->matmul_thread_local(a, es->up_w, es->up_type, xb_row, up_h, I, dim, 0);
					moe_activate(act_h, gate_h, up_h, I, es->gate_scale, es->up_scale, use_gelu);
				}

				a->matmul_thread_local(a, es->down_w, es->down_type, act_h, y_h, dim, I, 0);
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

	buffer *xb2		= batch_slot(ctx->bs, ctx->op->out);
	float  *xb2_ptr = batch_buf_ptr(xb2);
	memcpy(xb2_ptr, ctx->bs->moe_out.p, (size_t)ctx->n_rows * dim * sizeof(float));

	for (int row = 0; row < ctx->n_rows; row++) {
		for (int k = 0; k < K; k++) {
			moe_expert_slot *es = &ctx->bs->moe_per_token_slots[(size_t)row * K + k];
			if (es->eid < 0)
				continue;
			if (es->owned && es->heap_buf) {
				free(es->heap_buf);
				es->heap_buf = NULL;
			} else if (!es->owned) {
				moe_stream_release_slot(ctx->m, ctx->li, es);
			}
			es->eid		 = -1;
			es->owned	 = 0;
			es->heap_buf = NULL;
			es->gate_w	 = NULL;
			es->up_w	 = NULL;
			es->down_w	 = NULL;
		}
	}
	return OK;
}

static status_code op_batch_mla_qkv_proj_fused_impl(exec_ctx *ctx) {

	backend		  *a		 = exec_layer_backend(ctx);
	const int	   dim		 = ctx->m->dim;
	layer_weights *L		 = &ctx->m->layers[ctx->li];
	int			   q_lora	 = ctx->m->mla.q_lora;
	int			   kv_lora	 = ctx->m->mla.kv_lora;
	int			   qk_rope	 = ctx->m->mla.qk_rope;
	int			   q_b_rows	 = ctx->m->n_heads * ctx->m->mla.qk_head;
	int			   kv_a_rows = kv_lora + qk_rope;
	buffer		  *xb		 = batch_slot(ctx->bs, ctx->op->in[0]);

	float_buf_ensure(&ctx->bs->ffn_gate, (size_t)ctx->n_rows * q_lora);
	float_buf_ensure(&ctx->bs->ffn_up, (size_t)ctx->n_rows * kv_a_rows);
	buffer q_a_b	= {0};
	q_a_b.handle	= ctx->bs->ffn_gate.p;
	q_a_b.host_ptr	= ctx->bs->ffn_gate.p;
	q_a_b.size		= (size_t)ctx->n_rows * q_lora * sizeof(float);
	q_a_b.owner		= a;
	buffer kv_a_b	= {0};
	kv_a_b.handle	= ctx->bs->ffn_up.p;
	kv_a_b.host_ptr = ctx->bs->ffn_up.p;
	kv_a_b.size		= (size_t)ctx->n_rows * kv_a_rows * sizeof(float);
	kv_a_b.owner	= a;

	status_code st =
		a->matmul_batch(a, &L->q_a_w.buf, L->q_a_w.type, xb, &q_a_b, q_lora, dim, ctx->n_rows);
	if (st != OK)
		return st;
	st = a->matmul_batch(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, &kv_a_b, kv_a_rows, dim,
						 ctx->n_rows);
	if (st != OK)
		return st;

	st = a->rmsnorm_batch(a, &q_a_b, &L->q_a_norm_w.buf, &q_a_b, q_lora, ctx->m->norm_eps,
						  ctx->n_rows);
	if (st != OK)
		return st;

	buffer *q_buf = batch_slot(ctx->bs, RECIPE_SLOT_Q);
	st = a->matmul_batch(a, &L->q_b_w.buf, L->q_b_w.type, &q_a_b, q_buf, q_b_rows, q_lora,
						 ctx->n_rows);
	if (st != OK)
		return st;

	for (int row = 0; row < ctx->n_rows; row++) {
		buffer kv_a_row = batch_row_view(&kv_a_b, row, kv_a_rows);
		st = a->kv_put_mla(a, &ctx->cache->mla->kv, ctx->li, ctx->pos_start + row, &kv_a_row,
						   &L->kv_a_norm_w.buf, kv_lora, qk_rope, ctx->cache->n_ctx,
						   ctx->m->norm_eps);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code op_batch_mla_q_proj_impl(exec_ctx *ctx) {

	backend		  *a		= exec_layer_backend(ctx);
	const int	   dim		= ctx->m->dim;
	layer_weights *L		= &ctx->m->layers[ctx->li];
	int			   q_lora	= ctx->m->mla.q_lora;
	int			   q_b_rows = ctx->m->n_heads * ctx->m->mla.qk_head;
	buffer		  *xb		= batch_slot(ctx->bs, ctx->op->in[0]);

	float_buf_ensure(&ctx->bs->ffn_gate, (size_t)ctx->n_rows * q_lora);
	buffer q_a_b   = {0};
	q_a_b.handle   = ctx->bs->ffn_gate.p;
	q_a_b.host_ptr = ctx->bs->ffn_gate.p;
	q_a_b.size	   = (size_t)ctx->n_rows * q_lora * sizeof(float);
	q_a_b.owner	   = a;

	status_code st =
		a->matmul_batch(a, &L->q_a_w.buf, L->q_a_w.type, xb, &q_a_b, q_lora, dim, ctx->n_rows);
	if (st != OK)
		return st;
	st = a->rmsnorm_batch(a, &q_a_b, &L->q_a_norm_w.buf, &q_a_b, q_lora, ctx->m->norm_eps,
						  ctx->n_rows);
	if (st != OK)
		return st;
	buffer *q_buf = batch_slot(ctx->bs, RECIPE_SLOT_Q);
	return a->matmul_batch(a, &L->q_b_w.buf, L->q_b_w.type, &q_a_b, q_buf, q_b_rows, q_lora,
						   ctx->n_rows);
}

static status_code op_batch_mla_kv_proj_impl(exec_ctx *ctx) {

	backend		  *a		 = exec_layer_backend(ctx);
	const int	   dim		 = ctx->m->dim;
	layer_weights *L		 = &ctx->m->layers[ctx->li];
	int			   kv_lora	 = ctx->m->mla.kv_lora;
	int			   qk_rope	 = ctx->m->mla.qk_rope;
	int			   kv_a_rows = kv_lora + qk_rope;
	buffer		  *xb		 = batch_slot(ctx->bs, ctx->op->in[0]);
	float_buf_ensure(&ctx->bs->ffn_up, (size_t)ctx->n_rows * kv_a_rows);
	buffer kv_a_b	= {0};
	kv_a_b.handle	= ctx->bs->ffn_up.p;
	kv_a_b.host_ptr = ctx->bs->ffn_up.p;
	kv_a_b.size		= (size_t)ctx->n_rows * kv_a_rows * sizeof(float);
	kv_a_b.owner	= a;

	status_code st = a->matmul_batch(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, &kv_a_b, kv_a_rows, dim,
									 ctx->n_rows);
	if (st != OK)
		return st;
	for (int row = 0; row < ctx->n_rows; row++) {
		buffer kv_a_row = batch_row_view(&kv_a_b, row, kv_a_rows);
		st = a->kv_put_mla(a, &ctx->cache->mla->kv, ctx->li, ctx->pos_start + row, &kv_a_row,
						   &L->kv_a_norm_w.buf, kv_lora, qk_rope, ctx->cache->n_ctx,
						   ctx->m->norm_eps);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code op_batch_attention_mla_impl(exec_ctx *ctx) {
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

static status_code op_batch_rope_ext_impl(exec_ctx *ctx) {

	if (ctx->li < 0)
		return ERR_INVALID_ARG;
	const layer_ctx_entry *lc		= &ctx->m->recipe->layer_ctx[ctx->li];
	backend				  *a		= exec_layer_backend(ctx);
	int					   n_heads	= ctx->op->u.rope_ext.n_heads;
	int					   head_dim = lc->head_dim;
	if (n_heads == 0)
		n_heads = (ctx->op->in[0] == RECIPE_SLOT_Q) ? ctx->m->n_heads : lc->n_kv_heads;
	const float *rope_cos, *rope_sin, *freq_factors = NULL;
	if (lc->is_global) {
		rope_cos	 = ctx->s->rope_cos;
		rope_sin	 = ctx->s->rope_sin;
		freq_factors = ctx->op->u.rope_ext.use_freq_factors ? ctx->m->rope_freqs : NULL;
	} else {
		rope_cos = ctx->s->rope_cos_swa;
		rope_sin = ctx->s->rope_sin_swa;
	}
	a->rope_neox = ctx->op->u.rope_ext.rope_neox;
	a->rope_theta =
		lc->is_global ? ctx->m->layer_dims.rope_theta_global : ctx->m->layer_dims.rope_theta_swa;
	int		 row_stride = (ctx->op->in[0] == RECIPE_SLOT_Q) ? lc->q_row_stride : lc->kv_row_stride;
	backend *t			= a->rope_ext ? a : OP_BACKEND(rope);
	backend_sync_rope(t, a);
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = OK;
	for (int row = 0; row < ctx->n_rows; row++) {
		buffer rowb = batch_row_view(batch_slot(ctx->bs, ctx->op->in[0]), row, row_stride);
		if (t->rope_ext)
			st = t->rope_ext(t, &rowb, n_heads, head_dim, ctx->pos_start + row, rope_cos, rope_sin,
							 freq_factors);
		else {
			backend *t2 = OP_BACKEND(rope);
			backend_sync_rope(t2, a);
			st = t2->rope(t2, &rowb, n_heads, head_dim, ctx->pos_start + row, rope_cos, rope_sin);
		}
		if (st != OK)
			break;
	}
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code op_batch_ple_build_impl(exec_ctx *ctx) {
	if (!ctx->m->has_per_layer_embeddings)
		return OK;
	const int n_embd_per_layer = ctx->m->layer_dims.n_embd_per_layer;
	const int total_ple		   = n_embd_per_layer * ctx->m->n_layers;
	const int n_rows			   = ctx->n_rows;
	backend	 *a				   = exec_layer_backend(ctx);

	float		   *ple			= float_buf_ensure(&ctx->bs->ple_buf, (size_t)n_rows * total_ple);
	const float		embd_scale = sqrtf((float)n_embd_per_layer);
	const uint32_t embd_type	= ctx->m->layer_dims.per_layer_tok_embd.type;
	const size_t	row_stride = ggml_row_size(embd_type, total_ple);
	const uint8_t *embd_base = ctx->m->layer_dims.per_layer_tok_embd.host_ptr;
	for (int row = 0; row < n_rows; row++) {
		float *ple_row = ple + (size_t)row * total_ple;
		int	   token	 = ctx->bs->tokens ? ctx->bs->tokens[row] : 0;
		dequant_row_dispatch(embd_type, embd_base + (size_t)token * row_stride, total_ple,
							 ple_row);
		for (int i = 0; i < total_ple; i++)
			ple_row[i] *= embd_scale;
	}

	const size_t ple_bytes = (size_t)n_rows * total_ple * sizeof(float);
	status_code  st		 = batch_backend_buffer_ensure(&ctx->bs->ple_all, a, ple_bytes);
	if (st != OK)
		return st;
	st = batch_backend_buffer_ensure(&ctx->bs->ple_proj, a, ple_bytes);
	if (st != OK)
		return st;
	st = a->buffer_write_f32(a, &ctx->bs->ple_all, ple, n_rows * total_ple);
	if (st != OK)
		return st;

	st = a->matmul_batch(a, &ctx->m->layer_dims.per_layer_model_proj.buf,
						 ctx->m->layer_dims.per_layer_model_proj.type, &ctx->bs->x_b,
						 &ctx->bs->ple_proj, total_ple, ctx->m->dim, n_rows);
	if (st != OK)
		return st;
	st = a->scale_inplace(a, &ctx->bs->ple_proj,
						ctx->m->dim_sqrt > 0 ? 1.0f / ctx->m->dim_sqrt : 0.0f,
						n_rows * total_ple);
	if (st != OK)
		return st;
	st = a->rmsnorm_batch(a, &ctx->bs->ple_proj,
						  &ctx->m->layer_dims.per_layer_proj_norm_w.buf, &ctx->bs->ple_proj,
						  n_embd_per_layer, ctx->m->norm_eps, n_rows * ctx->m->n_layers);
	if (st != OK)
		return st;
	return a->ple_combine(a, &ctx->bs->ple_all, &ctx->bs->ple_proj, n_rows * total_ple,
						  0.70710678118654752f);
}

static status_code op_batch_ple_proj_inject_impl(exec_ctx *ctx) {
	if (ctx->li < 0)
		return ERR_INVALID_ARG;
	if (!ctx->m->has_per_layer_embeddings)
		return OK;
	const int n_embd_per_layer = ctx->m->layer_dims.n_embd_per_layer;
	const int total_ple		   = n_embd_per_layer * ctx->m->n_layers;
	const int n_rows			   = ctx->n_rows;
	const int	 inj_dim		= ctx->m->dim;
	backend		*a				= exec_layer_backend(ctx);
	layer_weights *L			= &ctx->m->layers[ctx->li];
	const size_t	 slice_bytes = (size_t)n_rows * n_embd_per_layer * sizeof(float);

	status_code st = batch_backend_buffer_ensure(&ctx->bs->ple_slice, a, slice_bytes);
	if (st != OK)
		return st;
	st = batch_backend_buffer_ensure(&ctx->bs->ple_inp, a, slice_bytes);
	if (st != OK)
		return st;

	for (int row = 0; row < n_rows; row++) {
		buffer src = ctx->bs->ple_all;
		buffer dst = ctx->bs->ple_slice;
		src.offset += ((size_t)row * total_ple + (size_t)ctx->li * n_embd_per_layer) *
					  sizeof(float);
		dst.offset += (size_t)row * n_embd_per_layer * sizeof(float);
		st = compute_copy_buffer_cross(ctx->s, &src, &dst, n_embd_per_layer);
		if (st != OK)
			return st;
	}

	buffer *xb2		 = batch_slot(ctx->bs, ctx->op->in[0]);
	buffer *attn_out = batch_slot(ctx->bs, ctx->op->in[1]);
	st = a->matmul_batch(a, &L->ple_inp_gate_w.buf, L->ple_inp_gate_w.type, xb2,
						 &ctx->bs->ple_inp, n_embd_per_layer, inj_dim, n_rows);
	if (st != OK)
		return st;
	st = a->ffn_activate_batch(a, &ctx->bs->ple_inp, &ctx->bs->ple_slice,
							   &ctx->bs->ple_inp, n_embd_per_layer, 1, n_rows);
	if (st != OK)
		return st;
	st = a->matmul_batch(a, &L->ple_proj_w.buf, L->ple_proj_w.type, &ctx->bs->ple_inp,
						 attn_out, inj_dim, n_embd_per_layer, n_rows);
	if (st != OK)
		return st;
	st = a->rmsnorm_batch(a, attn_out, &L->ple_post_norm_w.buf, attn_out, inj_dim,
						  ctx->m->norm_eps, n_rows);
	if (st != OK)
		return st;
	return a->add_batch(a, xb2, attn_out, inj_dim, n_rows);
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
	if (m->arch_info->is_hybrid_recurrent && m->qwen35.value_dim > attn_buf_size)
		attn_buf_size = m->qwen35.value_dim;

	if (!s->bs) {
		s->bs = xcalloc(1, sizeof(batch_scratch));
	}
	batch_scratch *bs	 = s->bs;
	bs->tokens			 = tokens;
	bs->n_tokens_stashed = n_tokens;
	batch_scratch_alloc(bs, a, n_tokens, dim, q_out, kv_out, max_inter, attn_buf_size);
	if (m->arch_info->is_hybrid_recurrent) {
		int q_attn	  = m->n_heads * m->head_dim;
		int proj_size = m->qwen35.conv_dim > 2 * q_attn ? m->qwen35.conv_dim : 2 * q_attn;
		int gate_size = m->qwen35.value_dim > q_attn ? m->qwen35.value_dim : q_attn;
		bs_ensure_wrap(bs, &bs->qwen_proj, &bs->qwen_proj_b, (size_t)n_tokens * proj_size, a);
		bs_ensure_wrap(bs, &bs->qwen_gate, &bs->qwen_gate_b, (size_t)n_tokens * gate_size, a);
		bs_ensure_wrap(bs, &bs->qwen_alpha, &bs->qwen_alpha_b,
					   (size_t)n_tokens * m->qwen35.n_value_heads, a);
		bs_ensure_wrap(bs, &bs->qwen_beta, &bs->qwen_beta_b,
					   (size_t)n_tokens * m->qwen35.n_value_heads, a);
	}

	status_code st = OK;

	for (int row = 0; row < n_tokens; row++) {
		buffer xrow = batch_row_view(&bs->x_b, row, dim);
		st = a->embd_lookup(a, &m->tok_embd.buf, m->tok_embd.type, tokens[row], dim, &xrow);
		if (st != OK)
			goto done;
		if (m->arch_info->has_scale_embeddings) {
			float  scale = m->dim_sqrt;
			float *xf	 = batch_buf_ptr(&xrow);
			for (int i = 0; i < dim; i++)
				xf[i] *= scale;
		}
	}

	for (int i = 0; i < r->n_pre_ops; i++) {
		const recipe_op *pop = &r->pre_ops[i];
		if (pop->kind == OP_EMBD_LOOKUP || pop->kind == OP_SCALE_EMBEDDINGS)
			continue;
		st = exec_op_batch(pop, m, cache, s, bs, pos_start, n_tokens, -1, flash_attn);
		if (st != OK)
			goto done;
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
			if (st != OK)
				goto done;
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

	{
		buffer	 last_x	 = batch_row_view(&bs->x_b, n_tokens - 1, dim);
		backend *owner_x = s->slots[RECIPE_SLOT_X].owner;
		float	*tmp	 = float_buf_ensure(&s->batch_logits_tmp, (size_t)dim);
		memcpy(tmp, batch_buf_ptr(&last_x), (size_t)dim * sizeof(float));
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
	if (fast_status != ERR_FALLBACK)
		return fast_status;

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
