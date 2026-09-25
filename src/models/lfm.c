#include "backend/backend.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "recipe.h"

static void lfm_append_conv_block(op_emitter *e, const model *m, int li) {
	int						   dim = m->dim;
	const model_hybrid_params *p   = &m->hybrid;

	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_ATTN_QKV, p->conv_dim, dim,
						 STAGE_MATMUL));
	OP_EMIT(e, mk_shortconv(RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_XB2));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_SSM_OUT, dim, p->value_dim,
						 STAGE_MATMUL));
	e->count = recipe_append_dense_ffn(e->ops, e->count, m, li);
}

static void lfm_append_attention_block(op_emitter *e, const model *m, int li) {
	int dim	   = m->dim;
	int q_out  = m->n_heads * m->head_dim;
	int kv_out = m->n_kv_heads * m->head_dim;
	int neox   = m->arch_info->uses_neox_rope;

	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, q_out, dim, STAGE_MATMUL));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim, STAGE_MATMUL));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim, STAGE_MATMUL));

	OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_Q, WIDX_ATTN_Q_NORM, m->norm_eps, m->n_heads));
	OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_K, WIDX_ATTN_K_NORM, m->norm_eps, m->n_kv_heads));
	OP_EMIT(e, mk_rope(RECIPE_SLOT_Q, m->n_heads, m->head_dim, neox));
	OP_EMIT(e, mk_rope(RECIPE_SLOT_K, m->n_kv_heads, m->head_dim, neox));
	OP_EMIT(e, mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V));
	OP_EMIT(e, mk_attention_default_scale(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, m->n_heads, m->n_kv_heads,
										  m->head_dim, m->n_ctx, m->sliding_window));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out, STAGE_MATMUL));
	e->count = recipe_append_dense_ffn(e->ops, e->count, m, li);
}

enum { LFM_MAX_OPS_PER_LAYER = 24 };

static model_recipe *build_lfm2_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(*r));
	recipe_build_pre_ops(r, m);

	r->layer.ops	 = xcalloc(LFM_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	r->layer.n_ops	 = LFM_MAX_OPS_PER_LAYER;
	r->per_layer_ops = xcalloc((size_t)m->n_layers * LFM_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	for (int li = 0; li < m->n_layers; li++) {
		recipe_op *ops = r->per_layer_ops + (size_t)li * LFM_MAX_OPS_PER_LAYER;
		op_emitter e   = op_emitter_make(ops, LFM_MAX_OPS_PER_LAYER, "lfm2");
		if (model_layer_is_recurrent(m, li))
			lfm_append_conv_block(&e, m, li);
		else
			lfm_append_attention_block(&e, m, li);
	}
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(lfm2, "lfm2", build_lfm2_recipe)
