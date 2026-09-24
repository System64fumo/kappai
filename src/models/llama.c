#include "backend/backend.h"
#include "common.h"
#include "log.h"
#include "model.h"
#include "recipe.h"

static model_recipe *build_standard_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	backend	   *a			 = m->backend;
	const int	dim			 = m->dim;
	const int	n_heads		 = m->n_heads;
	const int	n_kv_heads	 = m->n_kv_heads;
	const int	head_dim	 = m->head_dim;
	const int	intermediate = m->intermediate;
	const int	q_out		 = n_heads * head_dim;
	const int	kv_out		 = n_kv_heads * head_dim;
	const int	n_ctx		 = m->n_ctx;
	const float eps			 = m->norm_eps;
	const int	rope_neox	 = m->arch_info->uses_neox_rope;

	const int has_matmul_multi	  = backend_has_cap(a, BCAP_MULTI_MATMUL);
	const int has_matmul_residual = backend_has_cap(a, BCAP_MATMUL_RESIDUAL);
	const int has_rope_qk		  = backend_has_cap(a, BCAP_ROPE_QK_FUSED);

	const int can_fuse_attn_residual = has_matmul_residual && !m->arch_info->has_attn_post_norm;
	const int can_fuse_ffn_residual	 = has_matmul_residual && !m->arch_info->has_ffn_post_norm;

	r->max_intermediate = intermediate;
	r->max_head_dim		= head_dim;
	r->max_kv_heads		= n_kv_heads;

	recipe_build_pre_ops(r, m);

	{
		int		   cap = 24;
		recipe_op *ops = xcalloc(cap, sizeof(recipe_op));
		int		   i   = 0;

		ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, eps, STAGE_RMSNORM);

		if (has_matmul_multi) {
			ops[i++] = mk_matmul_multi3(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, dim, q_out, kv_out,
										kv_out);
		} else {
			ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, q_out, dim, STAGE_MATMUL);
			ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim, STAGE_MATMUL);
			ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim, STAGE_MATMUL);
		}

		if (has_rope_qk) {
			ops[i++] = mk_rope_qk_fused(n_heads, n_kv_heads, head_dim, rope_neox);
		} else {
			ops[i++] = mk_rope(RECIPE_SLOT_Q, n_heads, head_dim, rope_neox);
			ops[i++] = mk_rope(RECIPE_SLOT_K, n_kv_heads, head_dim, rope_neox);
		}

		ops[i++] = mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V);

		ops[i++] = mk_attention_default_scale(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, n_heads, n_kv_heads,
											  head_dim, n_ctx, m->sliding_window);

		if (can_fuse_attn_residual) {
			ops[i++] = mk_matmul_residual(RECIPE_SLOT_XB2, RECIPE_SLOT_X, RECIPE_SLOT_X, WIDX_WO,
										  dim, q_out);
		} else {
			ops[i++] =
				mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out, STAGE_MATMUL);
			if (m->arch_info->has_attn_post_norm) {
				ops[i++] = mk_rmsnorm(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_ATTN_OUT,
									  WIDX_POST_ATTN_NORM, eps, STAGE_RMSNORM);
			}
			ops[i++] = mk_add(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, STAGE_ADD);
			ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD);
		}

		ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, eps, STAGE_RMSNORM);

		if (m->layers[0].gate_up_fused) {
			ops[i++] = mk_matmul_fused_gateup(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP,
											  2 * intermediate, dim);
			ops[i++] = mk_ffn_activate_fused(RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_FFN_ACT,
											 intermediate, 0);
		} else if (has_matmul_multi) {
			ops[i++] = mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, dim,
										intermediate, intermediate);
		} else {
			ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, intermediate, dim,
								 STAGE_MATMUL);
			ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, intermediate, dim,
								 STAGE_MATMUL);
		}

		if (!m->layers[0].gate_up_fused) {
			ops[i++] = mk_ffn_activate(RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP,
									   RECIPE_SLOT_FFN_ACT, intermediate, 0);
		}

		if (can_fuse_ffn_residual) {
			ops[i++] = mk_matmul_residual(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_X, RECIPE_SLOT_X,
										  WIDX_DOWN, dim, intermediate);
		} else {
			ops[i++] = mk_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim, intermediate,
								 STAGE_MATMUL);
			if (m->arch_info->has_ffn_post_norm) {
				ops[i++] = mk_rmsnorm(RECIPE_SLOT_XB2, RECIPE_SLOT_XB2, WIDX_POST_FFN_NORM, eps,
									  STAGE_RMSNORM);
			}
			ops[i++] = mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_X, STAGE_ADD);
			ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD);
		}

		r->layer.ops   = ops;
		r->layer.n_ops = i;
	}

	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(llama, "llama", build_standard_recipe)
