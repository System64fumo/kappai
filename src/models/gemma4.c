#include "backend/backend.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "moe/moe_common.h"
#include "moe/moe_stream.h"
#include "recipe.h"

#include <math.h>

static void recipe_append_gemma4_attn_block(op_emitter *e, const model *m) {
	backend	   *a				 = m->backend;
	const int	dim				 = m->dim;
	const int	n_heads			 = m->n_heads;
	const int	n_kv_heads		 = m->n_kv_heads;
	const int	head_dim		 = m->head_dim;
	const int	n_ctx			 = m->n_ctx;
	const float eps				 = m->norm_eps;
	const float attn_scale		 = 1.0f;
	const int	rope_neox		 = m->arch_info->uses_neox_rope;
	const int	has_matmul_multi = backend_has_cap(a, BCAP_MULTI_MATMUL);

	OP_EMIT(e, mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, eps, STAGE_RMSNORM));

	const int fuse_qkv = has_matmul_multi;

	if (fuse_qkv) {
		OP_EMIT(e, mk_matmul_multi3(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, n_heads * head_dim,
									n_kv_heads * head_dim, n_kv_heads * head_dim, dim));
	} else {
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, 0, dim, STAGE_MATMUL));

		if (has_matmul_multi) {
			OP_EMIT(e, mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, 0, 0, dim));
		} else {
			OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, 0, dim, STAGE_MATMUL));
			OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, 0, dim, STAGE_MATMUL));
		}
	}

	if (m->arch_info->has_qk_norm) {
		OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_Q, WIDX_ATTN_Q_NORM, eps, 0));
		OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_K, WIDX_ATTN_K_NORM, eps, 0));
	}

	if (m->arch_info->uses_norm_v_without_weight) {
		OP_EMIT(e, mk_rmsnorm_noweight(RECIPE_SLOT_V));
	}

	OP_EMIT(e, mk_rope_ext(RECIPE_SLOT_Q, rope_neox));
	OP_EMIT(e, mk_rope_ext(RECIPE_SLOT_K, rope_neox));

	OP_EMIT(e, mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V));

	OP_EMIT(e, mk_attention(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, n_heads, n_kv_heads, head_dim, n_ctx,
							attn_scale, m->sliding_window));

	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, 0, STAGE_MATMUL));

	OP_EMIT(e, mk_rmsnorm_add(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT,
							  WIDX_POST_ATTN_NORM, eps, STAGE_ADD));
}

static void build_gemma4_ffn_prefix(op_emitter *e, const model *m, int dim, float eps) {
	const int has_matmul_multi = backend_has_cap(m->backend, BCAP_MULTI_MATMUL);
	OP_EMIT(e, mk_rmsnorm(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB, WIDX_FFN_NORM, eps, STAGE_RMSNORM));
	if (has_matmul_multi) {
		OP_EMIT(e, mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, 0, 0, dim));
	} else {
		OP_EMIT(e,
				mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, 0, dim, STAGE_MATMUL));
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, 0, dim, STAGE_MATMUL));
	}
	OP_EMIT(e, mk_matmul_ffn_down(RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_XB2,
								  WIDX_DOWN, dim, 0, ACTIVATION_GELU));
}

static void build_gemma4_ffn_tail(op_emitter *e, const model *m, float eps, model_recipe *r) {
	OP_EMIT(e, mk_rmsnorm_add(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB2,
							  WIDX_POST_FFN_NORM, eps, STAGE_ADD));
	if (m->has_per_layer_embeddings) {
		OP_EMIT(e, mk_ple_proj_inject(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_PLE_INP_GATE));
	}
	if (m->arch_info->has_layer_output_scale) {
		OP_EMIT(e, mk_scale(RECIPE_SLOT_XB2, WIDX_LAYER_OUT_SCALE, 1.0f));
	}
	OP_EMIT(e, mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD));
	r->layer.ops   = e->ops;
	r->layer.n_ops = e->count;
}

static model_recipe *build_gemma4_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	const int	dim = m->dim;
	const float eps = m->norm_eps;

	recipe_build_pre_ops(r, m);

	{
		enum { GEMMA4_MAX_OPS = 28 };
		recipe_op *ops = xcalloc(GEMMA4_MAX_OPS, sizeof(recipe_op));
		op_emitter e   = op_emitter_make(ops, GEMMA4_MAX_OPS, "gemma4");
		recipe_append_gemma4_attn_block(&e, m);
		build_gemma4_ffn_prefix(&e, m, dim, eps);
		build_gemma4_ffn_tail(&e, m, eps, r);
	}

	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(gemma4, "gemma4", build_gemma4_recipe)

static model_recipe *build_gemma4_moe_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	const int	dim = m->dim;
	const float eps = m->norm_eps;

	recipe_build_pre_ops(r, m);

	{
		enum { GEMMA4_MOE_MAX_OPS = 40 };
		recipe_op *ops = xcalloc(GEMMA4_MOE_MAX_OPS, sizeof(recipe_op));
		op_emitter e   = op_emitter_make(ops, GEMMA4_MOE_MAX_OPS, "gemma4_moe");
		recipe_append_gemma4_attn_block(&e, m);
		build_gemma4_ffn_prefix(&e, m, dim, eps);

		OP_EMIT(&e, mk_rmsnorm(RECIPE_SLOT_XB2, RECIPE_SLOT_FFN_ACT, WIDX_FFN_POST_NORM_1, eps,
							   STAGE_RMSNORM));
		OP_EMIT(&e, mk_rmsnorm(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB, WIDX_FFN_PRE_NORM_2, eps,
							   STAGE_RMSNORM));
		e.count = recipe_append_moe_ffn(e.ops, e.count, m, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB,
										RECIPE_SLOT_XB2);
		OP_EMIT(&e, mk_rmsnorm(RECIPE_SLOT_XB2, RECIPE_SLOT_XB2, WIDX_FFN_POST_NORM_2, eps,
							   STAGE_RMSNORM));
		OP_EMIT(&e, mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_FFN_ACT, STAGE_ADD));

		build_gemma4_ffn_tail(&e, m, eps, r);
	}

	recipe_build_post_ops(r, m);
	moe_stream_cache_init((struct model *)m);
	return r;
}

RECIPE_REGISTER(gemma4_moe, "gemma4_moe", build_gemma4_moe_recipe)