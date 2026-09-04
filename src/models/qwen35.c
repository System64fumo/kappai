#include "backend/backend.h"
#include "backend/cpu/scalar/quants.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "recipe.h"

#include <math.h>

static recipe_op qwen_matmul(uint8_t in, uint8_t out, uint8_t weight, int n, int k) {
	return mk_matmul(in, out, weight, n, k, STAGE_MATMUL);
}

static int qwen_append_ffn(recipe_op *ops, int i, const model *m) {
	int	  dim			   = m->dim;
	int	  inter			   = m->intermediate;
	float eps			   = m->norm_eps;
	int	  has_matmul_multi = backend_has_cap(m->backend, BCAP_MULTI_MATMUL);
	ops[i++]			   = mk_add(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++]			   = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD);
	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_POST_ATTN_NORM, eps, STAGE_RMSNORM);
	if (m->layers[0].gate_up_fused) {
		ops[i++] =
			qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP, 2 * inter, dim);
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE_FUSED,
			.in		   = {RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	} else if (has_matmul_multi) {
		ops[i++] =
			mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, dim, inter, inter);
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE,
			.in		   = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	} else {
		ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter, dim);
		ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim);
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE,
			.in		   = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	}
	ops[i++] = qwen_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim, inter);
	ops[i++] = mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD);
	return i;
}

static int qwen_append_recurrent(recipe_op *ops, const model *m) {
	int						   i   = 0;
	int						   dim = m->dim;
	const model_hybrid_params *p   = &m->hybrid;
	ops[i++] =
		mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_ATTN_QKV, p->conv_dim, dim);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_GATE, WIDX_ATTN_GATE, p->value_dim, dim);
	ops[i++] =
		qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_ALPHA, WIDX_SSM_ALPHA, p->n_value_heads, dim);
	ops[i++] =
		qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_BETA, WIDX_SSM_BETA, p->n_value_heads, dim);
	ops[i++] = (recipe_op){
		.kind  = OP_GATED_DELTA_NET,
		.in	   = {RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_HYB_GATE, RECIPE_SLOT_HYB_ALPHA},
		.out   = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
	};
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_SSM_OUT, dim, p->value_dim);
	return qwen_append_ffn(ops, i, m);
}

static int qwen_append_full_attention(recipe_op *ops, const model *m) {
	int i	   = 0;
	int dim	   = m->dim;
	int q_out  = m->n_heads * m->head_dim;
	int kv_out = m->n_kv_heads * m->head_dim;
	ops[i++] =
		mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_WQ, 2 * q_out, dim);
	ops[i++] = (recipe_op){
		.kind  = OP_SPLIT_QGATE,
		.in	   = {RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_Q,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_MATMUL,
	};
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim);
	ops[i++] = (recipe_op){
		.kind	   = OP_RMSNORM_PER_HEAD,
		.in		   = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	   = RECIPE_SLOT_Q,
		.w_idx	   = WIDX_ATTN_Q_NORM,
		.stage	   = STAGE_RMSNORM,
		.u.rmsnorm = {.eps = m->norm_eps, .n_heads = m->n_heads},
	};
	ops[i++] = (recipe_op){
		.kind	   = OP_RMSNORM_PER_HEAD,
		.in		   = {RECIPE_SLOT_K, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	   = RECIPE_SLOT_K,
		.w_idx	   = WIDX_ATTN_K_NORM,
		.stage	   = STAGE_RMSNORM,
		.u.rmsnorm = {.eps = m->norm_eps, .n_heads = m->n_kv_heads},
	};
	ops[i++] = mk_partial_rope_qk();
	ops[i++] = mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V);
	ops[i++] = mk_attention(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, m->n_heads, m->n_kv_heads, m->head_dim,
							m->n_ctx, 1.0f / sqrtf((float)m->head_dim), 0);
	ops[i++] = (recipe_op){
		.kind  = OP_ATTN_OUTPUT_GATE,
		.in	   = {RECIPE_SLOT_XB2, RECIPE_SLOT_HYB_GATE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
	};
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out);
	return qwen_append_ffn(ops, i, m);
}

enum { QWEN35_MAX_OPS_PER_LAYER = 24 };

static model_recipe *build_qwen35_recipe(const model *m) {
	model_recipe *r		= xcalloc(1, sizeof(*r));
	r->max_intermediate = m->intermediate;
	r->max_head_dim		= m->head_dim;
	r->max_kv_heads		= m->n_kv_heads;
	recipe_build_pre_ops(r, m);

	r->layer.ops	 = xcalloc(QWEN35_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	r->layer.n_ops	 = QWEN35_MAX_OPS_PER_LAYER;
	r->per_layer_ops = xcalloc((size_t)m->n_layers * QWEN35_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	for (int li = 0; li < m->n_layers; li++) {
		recipe_op *ops = r->per_layer_ops + (size_t)li * QWEN35_MAX_OPS_PER_LAYER;
		int		   n   = model_layer_is_recurrent(m, li) ? qwen_append_recurrent(ops, m)
														 : qwen_append_full_attention(ops, m);
		if (n > QWEN35_MAX_OPS_PER_LAYER) {
			ERROR("qwen35: layer %d produced %d recipe ops but capacity is %d -- "
				  "raise QWEN35_MAX_OPS_PER_LAYER",
				  li, n, QWEN35_MAX_OPS_PER_LAYER);
			free(r->pre_ops);
			free(r->layer.ops);
			free(r->per_layer_ops);
			free(r);
			return NULL;
		}
	}
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(qwen35, "qwen35", build_qwen35_recipe)
