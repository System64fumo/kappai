#include "backend/backend.h"
#include "common.h"
#include "log.h"
#include "model.h"
#include "moe/moe_stream.h"
#include "recipe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static model_recipe *build_glm_dsa_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	const int	dim		= m->dim;
	const int	n_heads = m->n_heads;
	const int	n_ctx	= m->n_ctx;
	const float eps		= m->norm_eps;

	const int qk_head  = m->mla.qk_head;
	const int v_head   = m->mla.v_head;
	const int q_b_rows = n_heads * qk_head;
	const int wo_in	   = n_heads * v_head;

	recipe_build_pre_ops(r, m);

	{
		enum { GLM_DSA_MAX_OPS = 32 };
		recipe_op *ops = xcalloc(GLM_DSA_MAX_OPS, sizeof(recipe_op));
		op_emitter e   = op_emitter_make(ops, GLM_DSA_MAX_OPS, "glm-dsa");

		OP_EMIT(&e, mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, eps, STAGE_RMSNORM));

		OP_EMIT(&e, mk_mla_qkv_proj_fused(RECIPE_SLOT_XB, RECIPE_SLOT_Q, q_b_rows, dim));

		OP_EMIT(&e, mk_attention_mla(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, n_heads, qk_head, n_ctx,
									 1.0f / sqrtf((float)qk_head)));

		if (backend_has_cap(m->backend, BCAP_MATMUL_RESIDUAL)) {
			OP_EMIT(&e, mk_matmul_residual(RECIPE_SLOT_XB2, RECIPE_SLOT_X, RECIPE_SLOT_X, WIDX_WO,
										   dim, wo_in));
		} else {
			OP_EMIT(&e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, wo_in,
								  STAGE_MATMUL));
			OP_EMIT(&e, mk_add(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD));
		}

		OP_EMIT(&e, mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, eps, STAGE_RMSNORM));

		e.count = recipe_append_moe_ffn(e.ops, e.count, m, RECIPE_SLOT_XB, RECIPE_SLOT_XB,
										RECIPE_SLOT_XB2);

		OP_EMIT(&e, mk_add(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD));
		if (m->moe.n_shared_experts > 0)
			OP_EMIT(&e, mk_add(RECIPE_SLOT_X, RECIPE_SLOT_FFN_ACT, STAGE_ADD));

		r->layer.ops   = ops;
		r->layer.n_ops = e.count;
	}

	recipe_build_post_ops(r, m);

	moe_stream_cache_init((struct model *)m);
	return r;
}

RECIPE_REGISTER(glm_dsa, "glm-dsa", build_glm_dsa_recipe)