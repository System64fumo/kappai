#include "backend/backend.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "recipe.h"

static recipe_op qwen_matmul(uint8_t in, uint8_t out, uint8_t weight, int n, int k) {
	return mk_matmul(in, out, weight, n, k, STAGE_MATMUL);
}

static void qwen_append_recurrent(op_emitter *e, const model *m, int li) {
	int						   dim = m->dim;
	const model_hybrid_params *p   = &m->hybrid;
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM));
	OP_EMIT(e, qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_ATTN_QKV, p->conv_dim, dim));
	OP_EMIT(e,
			qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_GATE, WIDX_ATTN_GATE, p->value_dim, dim));
	OP_EMIT(e, qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_ALPHA, WIDX_SSM_ALPHA, p->n_value_heads,
						   dim));
	OP_EMIT(
		e, qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_BETA, WIDX_SSM_BETA, p->n_value_heads, dim));
	OP_EMIT(e, mk_gated_delta_net(RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_HYB_GATE, RECIPE_SLOT_HYB_ALPHA,
								  RECIPE_SLOT_XB2));
	OP_EMIT(e, qwen_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_SSM_OUT, dim, p->value_dim));
	e->count = recipe_append_dense_ffn_ex(e->ops, e->count, m, li, WIDX_POST_ATTN_NORM);
}

static void qwen_append_full_attention(op_emitter *e, const model *m, int li) {
	int dim	   = m->dim;
	int q_out  = m->n_heads * m->head_dim;
	int kv_out = m->n_kv_heads * m->head_dim;
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM));
	OP_EMIT(e, qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_WQ, 2 * q_out, dim));
	OP_EMIT(e, mk_split_qgate());
	OP_EMIT(e, qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim));
	OP_EMIT(e, qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim));
	OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_Q, WIDX_ATTN_Q_NORM, m->norm_eps, m->n_heads));
	OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_K, WIDX_ATTN_K_NORM, m->norm_eps, m->n_kv_heads));
	OP_EMIT(e, mk_partial_rope_qk());
	OP_EMIT(e, mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V));
	OP_EMIT(e, mk_attention_default_scale(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, m->n_heads, m->n_kv_heads,
										  m->head_dim, m->n_ctx, 0));
	OP_EMIT(e, mk_attn_output_gate(RECIPE_SLOT_XB2, RECIPE_SLOT_HYB_GATE, RECIPE_SLOT_XB2));
	OP_EMIT(e, qwen_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out));
	e->count = recipe_append_dense_ffn_ex(e->ops, e->count, m, li, WIDX_POST_ATTN_NORM);
}

enum { QWEN35_MAX_OPS_PER_LAYER = 24 };

static model_recipe *build_qwen35_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(*r));
	recipe_build_pre_ops(r, m);

	r->layer.ops	 = xcalloc(QWEN35_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	r->layer.n_ops	 = QWEN35_MAX_OPS_PER_LAYER;
	r->per_layer_ops = xcalloc((size_t)m->n_layers * QWEN35_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	for (int li = 0; li < m->n_layers; li++) {
		recipe_op *ops = r->per_layer_ops + (size_t)li * QWEN35_MAX_OPS_PER_LAYER;
		op_emitter e   = op_emitter_make(ops, QWEN35_MAX_OPS_PER_LAYER, "qwen35");
		if (model_layer_is_recurrent(m, li))
			qwen_append_recurrent(&e, m, li);
		else
			qwen_append_full_attention(&e, m, li);
	}
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(qwen35, "qwen35", build_qwen35_recipe)
