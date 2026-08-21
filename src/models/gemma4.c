#include "backend/backend.h"
#include "backend/cpu/scalar/quants.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "moe/moe_common.h"
#include "moe/moe_stream.h"
#include "recipe.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void recipe_build_gemma4_pre_ops(model_recipe *r, const model *m) {
	recipe_build_pre_ops(r, m);
}

static int recipe_append_gemma4_attn_block(recipe_op *ops, int i, const model *m) {
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

	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, eps, STAGE_RMSNORM);

	const int fuse_qkv = has_matmul_multi;

	if (fuse_qkv) {
		ops[i++] = (recipe_op){
			.kind			= OP_MATMUL_MULTI,
			.in				= {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out			= RECIPE_SLOT_Q,
			.w_idx			= WIDX_WQ,
			.stage			= STAGE_MATMUL,
			.u.matmul_multi = {.n	  = 3,
							   .k	  = dim,
							   .n_out = {n_heads * head_dim, n_kv_heads * head_dim,
										 n_kv_heads * head_dim}},
		};
	} else {
		ops[i++] = (recipe_op){
			.kind	  = OP_MATMUL,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_Q,
			.w_idx	  = WIDX_WQ,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = 0, .k = dim},
		};

		if (has_matmul_multi) {
			ops[i++] = (recipe_op){
				.kind			= OP_MATMUL_MULTI,
				.in				= {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out			= RECIPE_SLOT_K,
				.w_idx			= WIDX_WK,
				.stage			= STAGE_MATMUL,
				.u.matmul_multi = {.n = 2, .k = dim, .n_out = {0, 0}},
			};
		} else {
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_K,
				.w_idx	  = WIDX_WK,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = 0, .k = dim},
			};
			ops[i++] = (recipe_op){
				.kind	  = OP_MATMUL,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_V,
				.w_idx	  = WIDX_WV,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = 0, .k = dim},
			};
		}
	}

	if (m->arch_info->has_qk_norm) {
		ops[i++] = (recipe_op){
			.kind  = OP_RMSNORM_PER_HEAD,
			.in	   = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_Q,
			.w_idx = WIDX_ATTN_Q_NORM,
			.stage = STAGE_RMSNORM,
		};
		ops[i++] = (recipe_op){
			.kind  = OP_RMSNORM_PER_HEAD,
			.in	   = {RECIPE_SLOT_K, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_K,
			.w_idx = WIDX_ATTN_K_NORM,
			.stage = STAGE_RMSNORM,
		};
	}

	if (m->arch_info->uses_norm_v_without_weight) {
		ops[i++] = (recipe_op){
			.kind  = OP_RMSNORM_NOWEIGHT,
			.in	   = {RECIPE_SLOT_V, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_NONE,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_RMSNORM,
		};
	}

	ops[i++] = (recipe_op){
		.kind		= OP_ROPE_EXT,
		.in			= {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out		= RECIPE_SLOT_NONE,
		.w_idx		= RECIPE_NO_WEIGHT,
		.stage		= STAGE_ROPE,
		.u.rope_ext = {.n_heads = 0, .head_dim = 0, .use_freq_factors = 1, .rope_neox = rope_neox},
	};
	ops[i++] = (recipe_op){
		.kind		= OP_ROPE_EXT,
		.in			= {RECIPE_SLOT_K, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out		= RECIPE_SLOT_NONE,
		.w_idx		= RECIPE_NO_WEIGHT,
		.stage		= STAGE_ROPE,
		.u.rope_ext = {.n_heads = 0, .head_dim = 0, .use_freq_factors = 1, .rope_neox = rope_neox},
	};

	ops[i++] = (recipe_op){
		.kind  = OP_KV_PUT,
		.in	   = {RECIPE_SLOT_K, RECIPE_SLOT_V, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_KVPUT,
	};

	ops[i++] = (recipe_op){
		.kind  = OP_ATTENTION,
		.in	   = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
		.u.attention =
			{
				.n_heads		   = n_heads,
				.n_kv_heads		   = n_kv_heads,
				.head_dim		   = head_dim,
				.n_ctx			   = n_ctx,
				.scale			   = attn_scale,
				.sliding_window	   = m->sliding_window,
				.n_kv_heads_active = n_kv_heads,
			},
	};

	ops[i++] = mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, 0, STAGE_MATMUL);

	ops[i++] = mk_rmsnorm_add(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT,
							  WIDX_POST_ATTN_NORM, eps, STAGE_ADD);

	return i;
}

static int build_gemma4_ffn_prefix(recipe_op *ops, int i, const model *m, int dim, float eps) {
	const int has_matmul_multi = backend_has_cap(m->backend, BCAP_MULTI_MATMUL);
	ops[i++] = mk_rmsnorm(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB, WIDX_FFN_NORM, eps, STAGE_RMSNORM);
	if (has_matmul_multi) {
		ops[i++] = (recipe_op){
			.kind			= OP_MATMUL_MULTI,
			.in				= {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out			= RECIPE_SLOT_FFN_GATE,
			.w_idx			= WIDX_GATE,
			.stage			= STAGE_MATMUL,
			.u.matmul_multi = {.n = 2, .k = dim, .n_out = {0, 0}},
		};
	} else {
		ops[i++] = (recipe_op){
			.kind	  = OP_MATMUL,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_FFN_GATE,
			.w_idx	  = WIDX_GATE,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = 0, .k = dim},
		};
		ops[i++] = (recipe_op){
			.kind	  = OP_MATMUL,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_FFN_UP,
			.w_idx	  = WIDX_UP,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = 0, .k = dim},
		};
	}
	ops[i++] = (recipe_op){
		.kind			   = OP_MATMUL_FFN_DOWN,
		.in				   = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
		.out			   = RECIPE_SLOT_XB2,
		.w_idx			   = WIDX_DOWN,
		.stage			   = STAGE_MATMUL,
		.u.matmul_ffn_down = {.n = dim, .k = 0, .activation = ACTIVATION_GELU},
	};
	return i;
}

static int build_gemma4_ffn_tail(recipe_op *ops, int i, const model *m, float eps,
								 model_recipe *r) {
	ops[i++] = mk_rmsnorm_add(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB2,
							  WIDX_POST_FFN_NORM, eps, STAGE_ADD);
	if (m->has_per_layer_embeddings) {
		ops[i++] = (recipe_op){
			.kind  = OP_PLE_PROJ_INJECT,
			.in	   = {RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_NONE,
			.w_idx = WIDX_PLE_INP_GATE,
			.stage = STAGE_FFN_ACT,
		};
	}
	if (m->arch_info->has_layer_output_scale) {
		ops[i++] = (recipe_op){
			.kind	 = OP_SCALE,
			.in		 = {RECIPE_SLOT_XB2, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	 = RECIPE_SLOT_NONE,
			.w_idx	 = WIDX_LAYER_OUT_SCALE,
			.stage	 = STAGE_ADD,
			.u.scale = {.scale = 1.0f},
		};
	}
	ops[i++]	   = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD);
	r->layer.ops   = ops;
	r->layer.n_ops = i;
	return i;
}

static model_recipe *build_gemma4_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	const int	dim	   = m->dim;
	const float eps	   = m->norm_eps;
	const int	max_hd = m->layer_dims.head_dim_global > m->layer_dims.head_dim_swa
							 ? m->layer_dims.head_dim_global
							 : m->layer_dims.head_dim_swa;

	r->max_intermediate = m->intermediate;
	r->max_head_dim		= max_hd;
	r->max_kv_heads		= m->n_kv_heads;

	recipe_build_gemma4_pre_ops(r, m);

	{
		int		   cap = 28;
		recipe_op *ops = xcalloc(cap, sizeof(recipe_op));
		int		   i   = recipe_append_gemma4_attn_block(ops, 0, m);
		i			   = build_gemma4_ffn_prefix(ops, i, m, dim, eps);
		build_gemma4_ffn_tail(ops, i, m, eps, r);
	}

	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(gemma4, "gemma4", build_gemma4_recipe)

static model_recipe *build_gemma4_moe_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	const int	dim			 = m->dim;
	const float eps			 = m->norm_eps;
	const int	intermediate = m->intermediate;
	const int	moe_inter	 = m->moe.moe_intermediate;
	const int	max_hd		 = m->layer_dims.head_dim_global > m->layer_dims.head_dim_swa
								   ? m->layer_dims.head_dim_global
								   : m->layer_dims.head_dim_swa;

	r->max_intermediate = intermediate > moe_inter ? intermediate : moe_inter;
	r->max_head_dim		= max_hd;
	r->max_kv_heads		= m->n_kv_heads;

	recipe_build_gemma4_pre_ops(r, m);

	{
		int		   cap = 40;
		recipe_op *ops = xcalloc(cap, sizeof(recipe_op));
		int		   i   = recipe_append_gemma4_attn_block(ops, 0, m);
		i			   = build_gemma4_ffn_prefix(ops, i, m, dim, eps);

		ops[i++] = (recipe_op){
			.kind	   = OP_RMSNORM,
			.in		   = {RECIPE_SLOT_XB2, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = WIDX_FFN_POST_NORM_1,
			.stage	   = STAGE_RMSNORM,
			.u.rmsnorm = {.eps = eps, .n_heads = 0},
		};
		ops[i++] = (recipe_op){
			.kind	   = OP_RMSNORM,
			.in		   = {RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_XB,
			.w_idx	   = WIDX_FFN_PRE_NORM_2,
			.stage	   = STAGE_RMSNORM,
			.u.rmsnorm = {.eps = eps, .n_heads = 0},
		};
		ops[i++] = (recipe_op){
			.kind	  = OP_MOE_ROUTER,
			.in		  = {RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_ROUTER_IDS,
			.w_idx	  = WIDX_FFN_GATE_INP,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = m->moe.n_experts, .k = dim},
		};
		ops[i++] = (recipe_op){
			.kind	  = OP_MOE_EXPERTS,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_ROUTER_IDS, RECIPE_SLOT_ROUTER_W},
			.out	  = RECIPE_SLOT_XB2,
			.w_idx	  = WIDX_NONE,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = dim, .k = moe_inter},
		};
		ops[i++] = (recipe_op){
			.kind	   = OP_RMSNORM,
			.in		   = {RECIPE_SLOT_XB2, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_XB2,
			.w_idx	   = WIDX_FFN_POST_NORM_2,
			.stage	   = STAGE_RMSNORM,
			.u.rmsnorm = {.eps = eps, .n_heads = 0},
		};
		ops[i++] = (recipe_op){
			.kind  = OP_ADD,
			.in	   = {RECIPE_SLOT_XB2, RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_XB2,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_ADD,
		};

		build_gemma4_ffn_tail(ops, i, m, eps, r);
	}

	recipe_build_post_ops(r, m);
	moe_stream_cache_init((struct model *)m);
	return r;
}

RECIPE_REGISTER(gemma4_moe, "gemma4_moe", build_gemma4_moe_recipe)