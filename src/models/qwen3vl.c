#include "common.h"
#include "model.h"
#include "recipe.h"

enum { QWEN3VL_MAX_OPS_PER_LAYER = 24 };

static model_recipe *build_qwen3vl_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(*r));
	recipe_build_pre_ops(r, m);
	recipe_build_layers(r, m, QWEN3VL_MAX_OPS_PER_LAYER, "qwen3vl", recipe_emit_qk_norm_attention);
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(qwen3vl, "qwen3vl", build_qwen3vl_recipe)
