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
	int			act = m->arch_info->uses_gelu_activation ? ACTIVATION_GELU : ACTIVATION_SILU;

	const int has_matmul_multi	  = backend_has_cap(a, BCAP_MULTI_MATMUL);
	const int has_matmul_residual = backend_has_cap(a, BCAP_MATMUL_RESIDUAL);
	const int has_rope_qk		  = backend_has_cap(a, BCAP_ROPE_QK_FUSED);

	const int can_fuse_attn_residual = has_matmul_residual && !m->arch_info->has_attn_post_norm;
	const int can_fuse_ffn_residual	 = has_matmul_residual && !m->arch_info->has_ffn_post_norm;

	recipe_build_pre_ops(r, m);

	{
		enum { LLAMA_MAX_OPS = 24 };
		recipe_op *ops = xcalloc(LLAMA_MAX_OPS, sizeof(recipe_op));
		op_emitter e   = op_emitter_make(ops, LLAMA_MAX_OPS, "llama");

		OP_EMIT(&e, mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, eps, STAGE_RMSNORM));

		if (has_matmul_multi) {
			OP_EMIT(&e, mk_matmul_multi3(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, q_out, kv_out,
										 kv_out, dim));
		} else {
			OP_EMIT(&e,
					mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, q_out, dim, STAGE_MATMUL));
			OP_EMIT(&e,
					mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim, STAGE_MATMUL));
			OP_EMIT(&e,
					mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim, STAGE_MATMUL));
		}

		if (has_rope_qk) {
			OP_EMIT(&e, mk_rope_qk_fused(n_heads, n_kv_heads, head_dim, rope_neox));
		} else {
			OP_EMIT(&e, mk_rope(RECIPE_SLOT_Q, n_heads, head_dim, rope_neox));
			OP_EMIT(&e, mk_rope(RECIPE_SLOT_K, n_kv_heads, head_dim, rope_neox));
		}

		OP_EMIT(&e, mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V));

		OP_EMIT(&e, mk_attention_default_scale(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, n_heads, n_kv_heads,
											   head_dim, n_ctx, m->sliding_window));

		if (can_fuse_attn_residual) {
			OP_EMIT(&e, mk_matmul_residual(RECIPE_SLOT_XB2, RECIPE_SLOT_X, RECIPE_SLOT_X, WIDX_WO,
										   dim, q_out));
		} else {
			OP_EMIT(&e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out,
								  STAGE_MATMUL));
			if (m->arch_info->has_attn_post_norm) {
				OP_EMIT(&e, mk_rmsnorm(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_ATTN_OUT,
									   WIDX_POST_ATTN_NORM, eps, STAGE_RMSNORM));
			}
			OP_EMIT(&e, mk_add(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD));
		}

		OP_EMIT(&e, mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, eps, STAGE_RMSNORM));

		if (m->layers[0].gate_up_fused) {
			OP_EMIT(&e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP,
								  2 * intermediate, dim, STAGE_MATMUL));
			OP_EMIT(&e, mk_ffn_activate_fused(RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_FFN_ACT,
											  intermediate, act));
		} else if (has_matmul_multi) {
			OP_EMIT(&e, mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE,
										 intermediate, intermediate, dim));
		} else {
			OP_EMIT(&e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, intermediate,
								  dim, STAGE_MATMUL));
			OP_EMIT(&e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, intermediate, dim,
								  STAGE_MATMUL));
		}

		if (!m->layers[0].gate_up_fused) {
			OP_EMIT(&e, mk_ffn_activate(RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP,
										RECIPE_SLOT_FFN_ACT, intermediate, act));
		}

		if (can_fuse_ffn_residual) {
			OP_EMIT(&e, mk_matmul_residual(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_X, RECIPE_SLOT_X,
										   WIDX_DOWN, dim, intermediate));
		} else {
			OP_EMIT(&e, mk_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim,
								  intermediate, STAGE_MATMUL));
			if (m->arch_info->has_ffn_post_norm) {
				OP_EMIT(&e, mk_rmsnorm(RECIPE_SLOT_XB2, RECIPE_SLOT_XB2, WIDX_POST_FFN_NORM, eps,
									   STAGE_RMSNORM));
			}
			OP_EMIT(&e, mk_add(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD));
		}

		r->layer.ops   = ops;
		r->layer.n_ops = e.count;
	}

	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(llama, "llama", build_standard_recipe)
