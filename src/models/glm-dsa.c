#include "backend/backend.h"
#include "common.h"
#include "log.h"
#include "model.h"
#include "moe/moe_stream.h"
#include "recipe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MLA_Q_A_STACK_CAP 4096
#define MLA_KV_A_STACK_CAP 2048

static model_recipe *build_glm_dsa_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	const int	dim		  = m->dim;
	const int	n_heads	  = m->n_heads;
	const int	n_ctx	  = m->n_ctx;
	const float eps		  = m->norm_eps;
	const int	rope_neox = m->arch_info->uses_neox_rope;

	const int qk_head  = m->mla.qk_head;
	const int v_head   = m->mla.v_head;
	const int q_b_rows = n_heads * qk_head;
	const int wo_in	   = n_heads * v_head;

	r->max_intermediate =
		m->moe.moe_intermediate > m->intermediate ? m->moe.moe_intermediate : m->intermediate;
	r->max_head_dim = qk_head;
	r->max_kv_heads = n_heads;

	recipe_build_pre_ops(r, m);

	{
		int		   cap = 32;
		recipe_op *ops = xcalloc(cap, sizeof(recipe_op));
		int		   i   = 0;

		ops[i++] = (recipe_op){
			.kind	   = OP_RMSNORM,
			.in		   = {RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_XB,
			.w_idx	   = WIDX_ATTN_NORM,
			.stage	   = STAGE_RMSNORM,
			.u.rmsnorm = {.eps = eps, .n_heads = 0},
		};

		ops[i++] = (recipe_op){
			.kind	  = OP_MLA_QKV_PROJ_FUSED,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_Q,
			.w_idx	  = WIDX_NONE,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = q_b_rows, .k = dim},
		};

		ops[i++] = (recipe_op){
			.kind  = OP_ATTENTION_MLA,
			.in	   = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_XB2,
			.w_idx = WIDX_NONE,
			.stage = STAGE_ATTN,
			.u.attention =
				{
					.n_heads		   = n_heads,
					.n_kv_heads		   = n_heads,
					.head_dim		   = qk_head,
					.n_ctx			   = n_ctx,
					.scale			   = 1.0f / sqrtf((float)qk_head),
					.sliding_window	   = 0,
					.n_kv_heads_active = n_heads,
				},
		};

		ops[i++] = (recipe_op){
			.kind	  = OP_MATMUL_RESIDUAL,
			.in		  = {RECIPE_SLOT_XB2, RECIPE_SLOT_X, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_X,
			.w_idx	  = WIDX_WO,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = dim, .k = wo_in},
		};

		ops[i++] = (recipe_op){
			.kind	   = OP_RMSNORM,
			.in		   = {RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_XB,
			.w_idx	   = WIDX_FFN_NORM,
			.stage	   = STAGE_RMSNORM,
			.u.rmsnorm = {.eps = eps, .n_heads = 0},
		};

		if (m->moe.n_shared_experts > 0) {
			int sh_inter = m->moe.moe_intermediate * m->moe.n_shared_experts;
			ops[i++]	 = (recipe_op){
				.kind	  = OP_MOE_SHARED,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_FFN_ACT,
				.w_idx	  = WIDX_NONE,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = sh_inter, .k = dim},
			};
		}

		ops[i++] = (recipe_op){
			.kind	  = OP_MOE_ROUTER,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
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
			.u.matmul = {.n = dim, .k = m->moe.moe_intermediate},
		};

		ops[i++] = (recipe_op){
			.kind  = OP_ADD,
			.in	   = {RECIPE_SLOT_X, RECIPE_SLOT_XB2, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_NONE,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_ADD,
		};
		if (m->moe.n_shared_experts > 0) {
			ops[i++] = (recipe_op){
				.kind  = OP_ADD,
				.in	   = {RECIPE_SLOT_X, RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_NONE},
				.out   = RECIPE_SLOT_NONE,
				.w_idx = RECIPE_NO_WEIGHT,
				.stage = STAGE_ADD,
			};
		}

		r->layer.ops   = ops;
		r->layer.n_ops = i;
	}

	recipe_build_post_ops(r, m);

	moe_stream_cache_init((struct model *)m);
	(void)rope_neox;
	return r;
}

RECIPE_REGISTER(glm_dsa, "glm-dsa", build_glm_dsa_recipe)