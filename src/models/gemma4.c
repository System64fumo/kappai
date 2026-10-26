#include "backend/backend.h"
#include "common.h"
#include "kvcache.h"
#include "model.h"
#include "moe/moe_common.h"
#include "moe/moe_stream.h"
#include "recipe.h"

#include <math.h>

static void gemma4_append_attn_block(op_emitter *e, const model *m, int li, int has_matmul_multi) {
	const int	dim		   = m->dim;
	const int	n_heads	   = m->n_heads;
	const int	n_kv_heads = model_layer_kv_heads(m, li);
	const int	head_dim   = model_layer_head_dim(m, li);
	const int	q_out	   = n_heads * head_dim;
	const int	kv_out	   = n_kv_heads * head_dim;
	const int	has_kv	   = model_layer_has_kv(m, li);
	const int	own_v	   = model_layer_has_own_v(m, li);
	const float eps		   = m->norm_eps;
	const int	rope_neox  = m->arch_info->uses_neox_rope;

	OP_EMIT(e, mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, eps, STAGE_RMSNORM));

	if (has_matmul_multi && has_kv) {
		recipe_op qkv = mk_matmul_multi3(RECIPE_SLOT_XB, RECIPE_SLOT_Q, RECIPE_SLOT_K,
										 RECIPE_SLOT_V, WIDX_WQ, q_out, kv_out, kv_out, dim);
		qkv.u.matmul_multi.v_from_k = !own_v;
		OP_EMIT(e, qkv);
	} else {
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, q_out, dim, STAGE_MATMUL));
		if (has_kv) {
			OP_EMIT(e,
					mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim, STAGE_MATMUL));
			OP_EMIT(e, own_v ? mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim,
										 STAGE_MATMUL)
							 : mk_copy_k_to_v(RECIPE_SLOT_K, RECIPE_SLOT_V));
		}
	}

	const int fuse_qk_norm_rope =
		m->arch_info->has_qk_norm && backend_has_cap(m->backend, BCAP_QK_NORM_ROPE);

	if (m->arch_info->has_qk_norm && !fuse_qk_norm_rope) {
		OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_Q, WIDX_ATTN_Q_NORM, eps, 0));
		if (has_kv)
			OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_K, WIDX_ATTN_K_NORM, eps, 0));
	}

	if (has_kv && m->arch_info->uses_norm_v_without_weight)
		OP_EMIT(e, mk_rmsnorm_noweight(RECIPE_SLOT_V));

	if (fuse_qk_norm_rope) {
		OP_EMIT(e, mk_qk_norm_rope(RECIPE_SLOT_Q, has_kv ? RECIPE_SLOT_K : RECIPE_SLOT_NONE, eps,
								   rope_neox));
	} else {
		OP_EMIT(e, mk_rope_ext(RECIPE_SLOT_Q, 1, rope_neox));
		if (has_kv)
			OP_EMIT(e, mk_rope_ext(RECIPE_SLOT_K, 0, rope_neox));
	}

	if (has_kv)
		OP_EMIT(e, mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V));

	recipe_op attention = mk_attention(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, n_heads, n_kv_heads,
									   head_dim, m->n_ctx, 1.0f, m->sliding_window);
	attention.u.attention.kv_layer = model_layer_kv_layer(m, li);
	OP_EMIT(e, attention);

	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out, STAGE_MATMUL));

	OP_EMIT(e, mk_rmsnorm_add(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT,
							  WIDX_POST_ATTN_NORM, eps, STAGE_ADD));
}

static void gemma4_append_ffn_prefix(op_emitter *e, const model *m, int li, int has_matmul_multi) {
	const int	dim			 = m->dim;
	const int	intermediate = model_layer_intermediate(m, li);
	const float eps			 = m->norm_eps;

	OP_EMIT(e, mk_rmsnorm(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB, WIDX_FFN_NORM, eps, STAGE_RMSNORM));
	if (has_matmul_multi) {
		OP_EMIT(e, mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP,
									WIDX_GATE, intermediate, intermediate, dim));
	} else {
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, intermediate, dim,
							 STAGE_MATMUL));
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, intermediate, dim,
							 STAGE_MATMUL));
	}
	OP_EMIT(e, mk_matmul_ffn_down(RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_XB2,
								  WIDX_DOWN, dim, intermediate, ACTIVATION_GELU));
}

static void gemma4_append_ffn_tail(op_emitter *e, const model *m) {
	const float eps = m->norm_eps;
	OP_EMIT(e, mk_rmsnorm_add(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB2,
							  WIDX_POST_FFN_NORM, eps, STAGE_ADD));
	if (m->has_per_layer_embeddings) {
		OP_EMIT(e, mk_ple_proj_inject(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_PLE_INP_GATE));
	}
	if (m->arch_info->has_layer_output_scale) {
		OP_EMIT(e, mk_scale(RECIPE_SLOT_XB2, WIDX_LAYER_OUT_SCALE, 1.0f));
	}
	OP_EMIT(e, mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD));
}

static void gemma4_append_dense_layer(op_emitter *e, const model *m, int li, int has_matmul_multi) {
	gemma4_append_attn_block(e, m, li, has_matmul_multi);
	gemma4_append_ffn_prefix(e, m, li, has_matmul_multi);
	gemma4_append_ffn_tail(e, m);
}

static void gemma4_append_moe_layer(op_emitter *e, const model *m, int li, int has_matmul_multi) {
	const float eps = m->norm_eps;

	gemma4_append_attn_block(e, m, li, has_matmul_multi);
	gemma4_append_ffn_prefix(e, m, li, has_matmul_multi);

	OP_EMIT(e, mk_rmsnorm(RECIPE_SLOT_XB2, RECIPE_SLOT_FFN_ACT, WIDX_FFN_POST_NORM_1, eps,
						  STAGE_RMSNORM));
	OP_EMIT(e, mk_rmsnorm(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB, WIDX_FFN_PRE_NORM_2, eps,
						  STAGE_RMSNORM));
	e->count = recipe_append_moe_ffn(e->ops, e->count, m, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_XB,
									 RECIPE_SLOT_XB2);
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_XB2, RECIPE_SLOT_XB2, WIDX_FFN_POST_NORM_2, eps, STAGE_RMSNORM));
	OP_EMIT(e, mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_FFN_ACT, STAGE_ADD));

	gemma4_append_ffn_tail(e, m);
}

typedef void (*gemma4_layer_fn)(op_emitter *e, const model *m, int li, int has_matmul_multi);

static void gemma4_build_layers(model_recipe *r, const model *m, int cap, const char *arch_name,
								gemma4_layer_fn append_layer) {
	const int has_mm = backend_has_cap(m->backend, BCAP_MULTI_MATMUL);

	r->layer.ops	 = xcalloc((size_t)cap, sizeof(recipe_op));
	r->layer.n_ops	 = cap;
	r->per_layer_ops = xcalloc((size_t)m->n_layers * cap, sizeof(recipe_op));
	int max_ops		 = 0;
	for (int li = 0; li < m->n_layers; li++) {
		op_emitter e = op_emitter_make(r->per_layer_ops + (size_t)li * cap, cap, arch_name);
		append_layer(&e, m, li, has_mm);
		if (e.count > max_ops)
			max_ops = e.count;
	}
	recipe_pack_per_layer_ops(r, m->n_layers, cap, max_ops);
}

static model_recipe *build_gemma4_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	recipe_build_pre_ops(r, m);
	gemma4_build_layers(r, m, 28, "gemma4", gemma4_append_dense_layer);
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(gemma4, "gemma4", build_gemma4_recipe)

static model_recipe *build_gemma4_moe_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	recipe_build_pre_ops(r, m);
	gemma4_build_layers(r, m, 40, "gemma4_moe", gemma4_append_moe_layer);
	recipe_build_post_ops(r, m);
	if (moe_stream_cache_init((struct model *)m) != OK) {
		ERROR("recipe: failed to init MoE expert cache");
		recipe_free(r);
		return NULL;
	}
	return r;
}

RECIPE_REGISTER(gemma4_moe, "gemma4_moe", build_gemma4_moe_recipe)
