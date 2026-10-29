#include "decision.h"
#include "recipe.h"

#include <math.h>
#include <stdio.h>

#define OMNI_MAX_TOKENS 16384
#define OMNI_MAX_OPTIONS 255

static const char *const OMNI_SEP	   = "<<d1omni:sep>>";
static const char *const OMNI_STATE	   = "<<d1omni:state>>";
static const char *const OMNI_QUESTION = "<<d1omni:question>>";
static const char *const OMNI_OPTION   = "<<d1omni:option>>";

typedef struct {
	int	  encoder_layers;
	float norm_eps;
} omni_params;

typedef struct {
	uint8_t		 index;
	const char	*name;
	int			 n_dims;
	uint64_t	 d0, d1;
	weight_class wc;
} omni_tensor;

static status_code omni_bind_tensors(model *m, int li, const omni_tensor *tensors, size_t n) {
	for (size_t i = 0; i < n; i++) {
		const omni_tensor *t = &tensors[i];
		char			   name[128];
		if (li >= 0)
			snprintf(name, sizeof(name), "blk.%d.%s", li, t->name);
		else
			snprintf(name, sizeof(name), "%s", t->name);
		status_code st = model_bind_tensor(m, li, t->index, name, t->n_dims, t->d0, t->d1, t->wc);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code omni_configure(model *m) {
	int32_t n_head = 0;
	float	eps	   = 0.0f;
	int		causal = 1;
	if (m->arch != ARCH_LFM2 ||
		gguf_get_i32(&m->gctx, "lfm2.decision.block_count", &n_head) != OK || n_head < 1 ||
		n_head >= m->n_layers || m->hybrid.conv_kernel != 3 ||
		gguf_get_bool(&m->gctx, "lfm2.attention.causal", &causal) != OK || causal ||
		gguf_get_f32(&m->gctx, "lfm2.attention.layer_norm_epsilon", &eps) != OK || !isfinite(eps) ||
		eps <= 0.0f || m->n_heads * m->head_dim != m->dim) {
		ERROR("lfm2-d1-omni: invalid encoder or decision head metadata");
		return ERR_FORMAT;
	}
	omni_params *p				= xcalloc(1, sizeof(*p));
	p->encoder_layers			= m->n_layers - n_head;
	p->norm_eps					= eps;
	m->variant_data				= p;
	m->full_sequence			= true;
	m->output_size				= 1;
	m->n_ctx					= MIN(m->n_ctx, OMNI_MAX_TOKENS);
	int				  dim		= m->dim;
	const omni_tensor tensors[] = {
		{WIDX_TYPE_EMBD, "token_types.weight", 2, dim, 3, WCLASS_MISC},
		{WIDX_CLS_NORM, "cls.norm.weight", 1, dim, 0, WCLASS_NORM},
		{WIDX_CLS_NORM_B, "cls.norm.bias", 1, dim, 0, WCLASS_NORM},
		{WIDX_CLS, "cls.weight", 2, dim, dim, WCLASS_MATMUL},
		{WIDX_CLS_B, "cls.bias", 1, dim, 0, WCLASS_MISC},
		{WIDX_CLS_OUT, "cls.output.weight", 1, dim, 0, WCLASS_MATMUL},
		{WIDX_CLS_OUT_B, "cls.output.bias", 1, 1, 0, WCLASS_MISC},
	};
	return omni_bind_tensors(m, -1, tensors, ARRAY_LEN(tensors));
}

static status_code omni_load_layer(model *m, int li) {
	const omni_params *p = m->variant_data;
	if (li < p->encoder_layers)
		return ERR_NOT_FOUND;
	layer_weights *L = &m->layers[li];
	if (L->is_recurrent || L->n_kv_heads != m->n_heads || L->intermediate < 1)
		return ERR_FORMAT;
	int				  dim		= m->dim;
	int				  inter		= L->intermediate;
	const omni_tensor tensors[] = {
		{WIDX_ATTN_NORM, "attn_norm.weight", 1, dim, 0, WCLASS_NORM},
		{WIDX_ATTN_NORM_B, "attn_norm.bias", 1, dim, 0, WCLASS_NORM},
		{WIDX_ATTN_QKV, "attn_qkv.weight", 2, dim, 3 * dim, WCLASS_MATMUL},
		{WIDX_ATTN_QKV_B, "attn_qkv.bias", 1, 3 * dim, 0, WCLASS_MISC},
		{WIDX_WO, "attn_output.weight", 2, dim, dim, WCLASS_MATMUL},
		{WIDX_WO_B, "attn_output.bias", 1, dim, 0, WCLASS_MISC},
		{WIDX_FFN_NORM, "ffn_norm.weight", 1, dim, 0, WCLASS_NORM},
		{WIDX_FFN_NORM_B, "ffn_norm.bias", 1, dim, 0, WCLASS_NORM},
		{WIDX_UP, "ffn_up.weight", 2, dim, inter, WCLASS_MATMUL},
		{WIDX_UP_B, "ffn_up.bias", 1, inter, 0, WCLASS_MISC},
		{WIDX_DOWN, "ffn_down.weight", 2, inter, dim, WCLASS_MATMUL},
		{WIDX_DOWN_B, "ffn_down.bias", 1, dim, 0, WCLASS_MISC},
	};
	return omni_bind_tensors(m, li, tensors, ARRAY_LEN(tensors));
}

static void omni_append_ffn(op_emitter *e, const model *m, int li) {
	const omni_params *p	 = m->variant_data;
	int				   dim	 = m->dim;
	int				   inter = model_layer_intermediate(m, li);
	OP_EMIT(e, mk_layernorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, WIDX_FFN_NORM_B, dim,
							p->norm_eps));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim, STAGE_MATMUL));
	OP_EMIT(e, mk_add_bias(RECIPE_SLOT_FFN_UP, WIDX_UP_B, inter));
	OP_EMIT(e, mk_activate(RECIPE_SLOT_FFN_UP, inter, ACTIVATION_RELU));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_FFN_UP, RECIPE_SLOT_XB2, WIDX_DOWN, dim, inter, STAGE_MATMUL));
	OP_EMIT(e, mk_add_bias(RECIPE_SLOT_XB2, WIDX_DOWN_B, dim));
	OP_EMIT(e, mk_add(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD));
}

static void omni_append_layer(op_emitter *e, const model *m, int li) {
	const omni_params *p = m->variant_data;
	if (li < p->encoder_layers) {
		recipe_emit_lfm_layer(e, m, li);
		return;
	}
	int dim = m->dim;
	if (li == p->encoder_layers) {
		OP_EMIT(e, mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_X, WIDX_OUTPUT_NORM, m->norm_eps,
							  STAGE_RMSNORM));
		OP_EMIT(e, mk_add_type_embd(RECIPE_SLOT_X, WIDX_TYPE_EMBD, dim, 3));
	}
	OP_EMIT(e, mk_layernorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, WIDX_ATTN_NORM_B, dim,
							p->norm_eps));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_ATTN_QKV, 3 * dim, dim,
						 STAGE_MATMUL));
	OP_EMIT(e, mk_add_bias(RECIPE_SLOT_HYB_PROJ, WIDX_ATTN_QKV_B, 3 * dim));
	OP_EMIT(e, mk_split_qkv(RECIPE_SLOT_HYB_PROJ, dim));
	recipe_op attn = mk_attention_default_scale(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, m->n_heads,
												m->n_heads, m->head_dim, m->n_ctx, 0);
	attn.kind	   = OP_ATTENTION_FULL;
	attn.in[1]	   = RECIPE_SLOT_K;
	attn.in[2]	   = RECIPE_SLOT_V;
	OP_EMIT(e, attn);
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, dim, STAGE_MATMUL));
	OP_EMIT(e, mk_add_bias(RECIPE_SLOT_ATTN_OUT, WIDX_WO_B, dim));
	OP_EMIT(e, mk_add(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD));
	if (li + 1 < m->n_layers || model_layer_backend(m, li) != m->backend)
		omni_append_ffn(e, m, li);
}

static model_recipe *omni_recipe(const model *m) {
	const omni_params *p = m->variant_data;
	model_recipe	  *r = xcalloc(1, sizeof(*r));
	recipe_build_pre_ops(r, m);
	recipe_build_layers(r, m, 26, "lfm2-d1-omni", omni_append_layer);
	r->post_ops		= xcalloc(14, sizeof(recipe_op));
	op_emitter e	= op_emitter_make(r->post_ops, 14, "lfm2-d1-omni");
	int		   last = m->n_layers - 1;
	if (model_layer_backend(m, last) == m->backend) {
		r->post_weight_layer = last;
		omni_append_ffn(&e, m, last);
	}
	OP_EMIT(&e, mk_layernorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_CLS_NORM, WIDX_CLS_NORM_B, m->dim,
							 p->norm_eps));
	OP_EMIT(&e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_XB2, WIDX_CLS, m->dim, m->dim,
						  STAGE_LOGITS_MATMUL));
	OP_EMIT(&e, mk_add_bias(RECIPE_SLOT_XB2, WIDX_CLS_B, m->dim));
	OP_EMIT(&e, mk_activate(RECIPE_SLOT_XB2, m->dim, ACTIVATION_GELU_ERF));
	OP_EMIT(&e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_LOGITS, WIDX_CLS_OUT, 1, m->dim,
						  STAGE_LOGITS_MATMUL));
	OP_EMIT(&e, mk_add_bias(RECIPE_SLOT_LOGITS, WIDX_CLS_OUT_B, 1));
	OP_EMIT(&e, mk_logits_readback());
	r->n_post_ops = e.count;
	return r;
}

static void omni_limits(const decision_question *q, decision_limits *lim) {
	int budget	  = MAX(96, MIN(24 * q->n_options + 32, OMNI_MAX_TOKENS / 2));
	lim->plain	  = OMNI_MAX_TOKENS;
	lim->question = MAX(16, budget);
	lim->option	  = MAX(2, (budget - 3 * q->n_options) / q->n_options);
	lim->total	  = OMNI_MAX_TOKENS;
}

static const model_variant omni_variant = {
	.configure	= omni_configure,
	.load_layer = omni_load_layer,
	.recipe		= omni_recipe,
};

static const decision_ops omni_ops = {
	.type		   = "lfm2-d1-omni",
	.max_options   = OMNI_MAX_OPTIONS,
	.variant	   = &omni_variant,
	.escape		   = true,
	.sep		   = OMNI_SEP,
	.state_mark	   = OMNI_STATE,
	.question_mark = OMNI_QUESTION,
	.option_mark   = OMNI_OPTION,
	.score_mark	   = "<|mask|>",
	.limits		   = omni_limits,
	.forward	   = decision_forward_masked,
};

DECISION_REGISTER(lfm2_d1_omni, &omni_ops)
