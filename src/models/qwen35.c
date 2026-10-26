#include "backend/backend.h"
#include "common.h"
#include "kvcache.h"
#include "model.h"
#include "recipe.h"

static void qwen_append_recurrent(op_emitter *e, const model *m, int li) {
	int						   dim = m->dim;
	const model_hybrid_params *p   = &m->hybrid;
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_ATTN_QKV, p->conv_dim, dim,
						 STAGE_MATMUL));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_GATE, WIDX_ATTN_GATE, p->value_dim, dim,
						 STAGE_MATMUL));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_ALPHA, WIDX_SSM_ALPHA, p->n_value_heads,
						 dim, STAGE_MATMUL));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_BETA, WIDX_SSM_BETA, p->n_value_heads, dim,
						 STAGE_MATMUL));
	OP_EMIT(e, mk_gated_delta_net(RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_HYB_GATE, RECIPE_SLOT_HYB_ALPHA,
								  RECIPE_SLOT_HYB_BETA, RECIPE_SLOT_XB2));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_SSM_OUT, dim, p->value_dim,
						 STAGE_MATMUL));
	recipe_emit_dense_ffn(e, m, li, WIDX_POST_ATTN_NORM);
}

static void qwen_append_full_attention(op_emitter *e, const model *m, int li) {
	int dim	   = m->dim;
	int q_out  = m->n_heads * m->head_dim;
	int kv_out = m->n_kv_heads * m->head_dim;
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM));
	OP_EMIT(e,
			mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_WQ, 2 * q_out, dim, STAGE_MATMUL));
	OP_EMIT(e, mk_split_qgate(RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_Q, RECIPE_SLOT_HYB_GATE));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim, STAGE_MATMUL));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim, STAGE_MATMUL));
	OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_Q, WIDX_ATTN_Q_NORM, m->norm_eps, m->n_heads));
	OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_K, WIDX_ATTN_K_NORM, m->norm_eps, m->n_kv_heads));
	OP_EMIT(e, mk_partial_rope_qk(RECIPE_SLOT_Q, RECIPE_SLOT_K));
	OP_EMIT(e, mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V));
	OP_EMIT(e, mk_attention_default_scale(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, m->n_heads, m->n_kv_heads,
										  m->head_dim, m->n_ctx, 0));
	OP_EMIT(e, mk_attn_output_gate(RECIPE_SLOT_XB2, RECIPE_SLOT_HYB_GATE));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out, STAGE_MATMUL));
	recipe_emit_dense_ffn(e, m, li, WIDX_POST_ATTN_NORM);
}

enum { QWEN35_MAX_OPS_PER_LAYER = 24 };

static void qwen_append_layer(op_emitter *e, const model *m, int li) {
	if (model_layer_is_recurrent(m, li))
		qwen_append_recurrent(e, m, li);
	else
		qwen_append_full_attention(e, m, li);
}

static model_recipe *build_qwen35_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(*r));
	recipe_build_pre_ops(r, m);

	recipe_build_layers(r, m, QWEN35_MAX_OPS_PER_LAYER, "qwen35", qwen_append_layer);
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(qwen35, "qwen35", build_qwen35_recipe)
