#include "backend/backend.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "recipe.h"

#include <math.h>

static int lfm_append_ffn(recipe_op *ops, int i, const model *m, int li) {
	return recipe_append_dense_ffn(ops, i, m, li);
}

static int lfm_append_conv_block(recipe_op *ops, const model *m, int li) {
	int						   i   = 0;
	int						   dim = m->dim;
	const model_hybrid_params *p   = &m->hybrid;

	ops[i++] =
		mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_ATTN_QKV, p->conv_dim, dim,
						 STAGE_MATMUL);
	ops[i++] = (recipe_op){
		.kind  = OP_SHORTCONV,
		.in	   = {RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
	};
	ops[i++] = mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_SSM_OUT, dim, p->value_dim,
						 STAGE_MATMUL);
	return lfm_append_ffn(ops, i, m, li);
}

static int lfm_append_attention_block(recipe_op *ops, const model *m, int li) {
	int i	   = 0;
	int dim	   = m->dim;
	int q_out  = m->n_heads * m->head_dim;
	int kv_out = m->n_kv_heads * m->head_dim;
	int neox   = m->arch_info->uses_neox_rope;

	ops[i++] =
		mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, q_out, dim, STAGE_MATMUL);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim, STAGE_MATMUL);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim, STAGE_MATMUL);

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
	ops[i++] = mk_rope(RECIPE_SLOT_Q, m->n_heads, m->head_dim, neox);
	ops[i++] = mk_rope(RECIPE_SLOT_K, m->n_kv_heads, m->head_dim, neox);
	ops[i++] = mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V);
	ops[i++] = mk_attention(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, m->n_heads, m->n_kv_heads, m->head_dim,
							m->n_ctx, 1.0f / sqrtf((float)m->head_dim), m->sliding_window);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out, STAGE_MATMUL);
	return lfm_append_ffn(ops, i, m, li);
}

enum { LFM_MAX_OPS_PER_LAYER = 24 };

static model_recipe *build_lfm2_recipe(const model *m) {
	model_recipe *r		= xcalloc(1, sizeof(*r));
	r->max_intermediate = m->intermediate;
	r->max_head_dim		= m->head_dim;
	r->max_kv_heads		= m->n_kv_heads;
	recipe_build_pre_ops(r, m);

	r->layer.ops	 = xcalloc(LFM_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	r->layer.n_ops	 = LFM_MAX_OPS_PER_LAYER;
	r->per_layer_ops = xcalloc((size_t)m->n_layers * LFM_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	for (int li = 0; li < m->n_layers; li++) {
		recipe_op *ops = r->per_layer_ops + (size_t)li * LFM_MAX_OPS_PER_LAYER;
		int		   n   = model_layer_is_recurrent(m, li) ? lfm_append_conv_block(ops, m, li)
														 : lfm_append_attention_block(ops, m, li);
		if (n > LFM_MAX_OPS_PER_LAYER) {
			ERROR("lfm2: layer %d produced %d recipe ops but capacity is %d -- "
				  "raise LFM_MAX_OPS_PER_LAYER",
				  li, n, LFM_MAX_OPS_PER_LAYER);
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

RECIPE_REGISTER(lfm2, "lfm2", build_lfm2_recipe)
