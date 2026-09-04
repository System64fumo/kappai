#include "backend/backend.h"
#include "common.h"
#include "log.h"
#include "model.h"
#include "recipe.h"

#include <math.h>

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
	const float attn_scale	 = 1.0f / sqrtf((float)head_dim);
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
			ops[i++] = (recipe_op){
				.kind			= OP_MATMUL_MULTI,
				.in				= {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out			= RECIPE_SLOT_Q,
				.w_idx			= WIDX_WQ,
				.stage			= STAGE_MATMUL,
				.u.matmul_multi = {.n = 3, .k = dim, .n_out = {q_out, kv_out, kv_out}},
			};
		} else {
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_Q,
				.w_idx	  = WIDX_WQ,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = q_out, .k = dim},
			};
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_K,
				.w_idx	  = WIDX_WK,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = kv_out, .k = dim},
			};
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_V,
				.w_idx	  = WIDX_WV,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = kv_out, .k = dim},
			};
		}

		if (has_rope_qk) {
			ops[i++] = mk_rope_qk_fused(n_heads, n_kv_heads, head_dim, rope_neox);
		} else {
			ops[i++] = mk_rope(RECIPE_SLOT_Q, n_heads, head_dim, rope_neox);
			ops[i++] = mk_rope(RECIPE_SLOT_K, n_kv_heads, head_dim, rope_neox);
		}

		ops[i++] = mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V);

		ops[i++] = mk_attention(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, n_heads, n_kv_heads, head_dim,
								n_ctx, attn_scale, m->sliding_window);

		if (can_fuse_attn_residual) {
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL_RESIDUAL,
				.in		  = {RECIPE_SLOT_XB2, RECIPE_SLOT_X, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_X,
				.w_idx	  = WIDX_WO,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = dim, .k = q_out},
			};
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
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL_FUSED_GATEUP,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_FFN_GATE_UP,
				.w_idx	  = WIDX_GATE_UP,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = 2 * intermediate, .k = dim},
			};
			ops[i++] = (recipe_op){
				.kind	   = OP_FFN_ACTIVATE_FUSED,
				.in		   = {RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	   = RECIPE_SLOT_FFN_ACT,
				.w_idx	   = RECIPE_NO_WEIGHT,
				.stage	   = STAGE_FFN_ACT,
				.u.ffn_act = {.n = intermediate, .activation = 0},
			};
		} else if (has_matmul_multi) {
			ops[i++] = mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, dim,
										intermediate, intermediate);
		} else {
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_FFN_GATE,
				.w_idx	  = WIDX_GATE,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = intermediate, .k = dim},
			};
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_FFN_UP,
				.w_idx	  = WIDX_UP,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = intermediate, .k = dim},
			};
		}

		if (!m->layers[0].gate_up_fused) {
			ops[i++] = (recipe_op){
				.kind	   = OP_FFN_ACTIVATE,
				.in		   = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
				.out	   = RECIPE_SLOT_FFN_ACT,
				.w_idx	   = RECIPE_NO_WEIGHT,
				.stage	   = STAGE_FFN_ACT,
				.u.ffn_act = {.n = intermediate, .activation = 0},
			};
		}

		if (can_fuse_ffn_residual) {
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL_RESIDUAL,
				.in		  = {RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_X, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_X,
				.w_idx	  = WIDX_DOWN,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = dim, .k = intermediate},
			};
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
