#include "backend/backend.h"
#include "common.h"
#include "kvcache.h"
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
	recipe_emit_dense_ffn(e, m, li, WIDX_FFN_NORM);
}

enum { LFM_MAX_OPS_PER_LAYER = 24 };

static void lfm_append_layer(op_emitter *e, const model *m, int li) {
	if (model_layer_is_recurrent(m, li))
		lfm_append_conv_block(e, m, li);
	else
		recipe_emit_qk_norm_attention(e, m, li);
}

static model_recipe *build_lfm2_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(*r));
	recipe_build_pre_ops(r, m);

	recipe_build_layers(r, m, LFM_MAX_OPS_PER_LAYER, "lfm2", lfm_append_layer);
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(lfm2, "lfm2", build_lfm2_recipe)
