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

static int qwen_append_ffn(recipe_op *ops, int i, const model *m) {
	int start = i;
	i		  = recipe_append_dense_ffn(ops, i, m, 0);
	for (int j = start; j < i; j++) {
		if (ops[j].kind == OP_RMSNORM && ops[j].w_idx == WIDX_FFN_NORM)
			ops[j].w_idx = WIDX_POST_ATTN_NORM;
	}
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
	ops[i++] = mk_gated_delta_net(RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_HYB_GATE, RECIPE_SLOT_HYB_ALPHA,
								  RECIPE_SLOT_XB2);
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
	ops[i++] = mk_split_qgate();
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim);
	ops[i++] = mk_rmsnorm_per_head(RECIPE_SLOT_Q, WIDX_ATTN_Q_NORM, m->norm_eps, m->n_heads);
	ops[i++] = mk_rmsnorm_per_head(RECIPE_SLOT_K, WIDX_ATTN_K_NORM, m->norm_eps, m->n_kv_heads);
	ops[i++] = mk_partial_rope_qk();
	ops[i++] = mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V);
	ops[i++] = mk_attention_default_scale(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, m->n_heads, m->n_kv_heads,
										  m->head_dim, m->n_ctx, 0);
	ops[i++] = mk_attn_output_gate(RECIPE_SLOT_XB2, RECIPE_SLOT_HYB_GATE, RECIPE_SLOT_XB2);
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
