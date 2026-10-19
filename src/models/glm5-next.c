#include "common.h"
#include "model.h"
#include "moe/moe_stream.h"
#include "recipe.h"

#include <string.h>

enum {
	GLM5_NEXT_MAX_OPS_PER_LAYER = 24,
};

static void append_dense_ffn(op_emitter *e, const model *m, int li) {
	const int dim = m->dim, inter = m->intermediate;
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, m->norm_eps, STAGE_RMSNORM));
	if (m->layers[li].gate_up_fused) {
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP, 2 * inter, dim,
							 STAGE_MATMUL));
		OP_EMIT(e, mk_ffn_activate_fused(RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_FFN_ACT, inter,
										 ACTIVATION_SILU));
	} else {
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter, dim,
							 STAGE_MATMUL));
		OP_EMIT(e,
				mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim, STAGE_MATMUL));
		OP_EMIT(e, mk_ffn_activate(RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_FFN_ACT,
								   inter, ACTIVATION_SILU));
	}
	OP_EMIT(e,
			mk_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim, inter, STAGE_MATMUL));
}

static void append_moe_ffn(op_emitter *e, const model *m) {
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, m->norm_eps, STAGE_RMSNORM));
	e->count =
		recipe_append_moe_ffn(e->ops, e->count, m, RECIPE_SLOT_XB, RECIPE_SLOT_XB, RECIPE_SLOT_XB2);
	OP_EMIT(e, mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_FFN_ACT, STAGE_ADD));
}

static void append_layer(op_emitter *e, const model *m, int li) {
	OP_EMIT(e,
			mk_backend_op(OP_HC_PRE, RECIPE_SLOT_HC, RECIPE_SLOT_X, WIDX_HC_ATTN_FN, STAGE_ATTN));
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM));
	OP_EMIT(e, mk_backend_op(model_layer_is_recurrent(m, li) ? OP_KDA : OP_MLA_ABSORBED,
							 RECIPE_SLOT_XB, RECIPE_SLOT_ATTN_OUT, WIDX_NONE, STAGE_ATTN));
	OP_EMIT(e,
			mk_backend_op(OP_HC_POST, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_HC, WIDX_NONE, STAGE_ADD));
	OP_EMIT(e, mk_backend_op(OP_HC_PRE, RECIPE_SLOT_HC, RECIPE_SLOT_X, WIDX_HC_FFN_FN, STAGE_ATTN));
	if (model_layer_is_moe(m, li))
		append_moe_ffn(e, m);
	else
		append_dense_ffn(e, m, li);
	OP_EMIT(e, mk_backend_op(OP_HC_POST, RECIPE_SLOT_XB2, RECIPE_SLOT_HC, WIDX_NONE, STAGE_ADD));
}

static model_recipe *build_glm5_next_recipe(const model *m) {
	if (m->mla.qk_rope != 0) {
		ERROR("glm5-next: MLA with qk_rope=%d is unsupported; the absorbed kernel is rope-free",
			  m->mla.qk_rope);
		return NULL;
	}
	for (int li = 0; li < m->n_layers; li++) {
		if (!model_layer_is_recurrent(m, li) && (m->layers[li].k_b_w.type != GGML_TYPE_F32 ||
												 m->layers[li].v_b_w.type != GGML_TYPE_F32)) {
			ERROR("glm5-next: layer %d MLA k_b/v_b could not be dequantized to f32", li);
			return NULL;
		}
	}

	model_recipe *r = xcalloc(1, sizeof(*r));
	recipe_build_pre_ops(r, m);
	r->pre_ops = xrealloc(r->pre_ops, (size_t)(r->n_pre_ops + 1) * sizeof(recipe_op));
	r->pre_ops[r->n_pre_ops++] =
		mk_backend_op(OP_HC_INIT, RECIPE_SLOT_X, RECIPE_SLOT_HC, WIDX_NONE, STAGE_ADD);

	r->layer.ops   = xcalloc(GLM5_NEXT_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	r->layer.n_ops = GLM5_NEXT_MAX_OPS_PER_LAYER;
	r->per_layer_ops =
		xcalloc((size_t)m->n_layers * GLM5_NEXT_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	for (int li = 0; li < m->n_layers; li++) {
		recipe_op *ops = r->per_layer_ops + (size_t)li * GLM5_NEXT_MAX_OPS_PER_LAYER;
		op_emitter e   = op_emitter_make(ops, GLM5_NEXT_MAX_OPS_PER_LAYER, "glm5-next");
		append_layer(&e, m, li);
	}

	recipe_build_post_ops(r, m);
	r->post_ops = xrealloc(r->post_ops, (size_t)(r->n_post_ops + 1) * sizeof(recipe_op));
	memmove(r->post_ops + 1, r->post_ops, (size_t)r->n_post_ops * sizeof(recipe_op));
	r->post_ops[0] =
		mk_backend_op(OP_HC_FINAL, RECIPE_SLOT_HC, RECIPE_SLOT_X, WIDX_NONE, STAGE_ADD);
	r->n_post_ops++;

	if (moe_stream_cache_init((struct model *)m) != OK) {
		ERROR("recipe: failed to init MoE expert cache");
		recipe_free(r);
		return NULL;
	}
	return r;
}

RECIPE_REGISTER(glm5_next, "glm5-next", build_glm5_next_recipe)
