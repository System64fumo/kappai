#include "decision.h"
#include "recipe.h"

#include <ctype.h>
#include <limits.h>
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

static char *omni_escape(const char *text) {
	str_builder out = {0};
	for (const char *p = text; *p;) {
		if (!strncmp(p, "<<d1omni:", 9)) {
			sb_puts(&out, "<<d1omni ");
			p += 9;
			continue;
		}
		if (p[0] == '<' && p[1] == '|') {
			const char *end = p + 2;
			while (isalnum((unsigned char)*end) || *end == '_')
				end++;
			if (end > p + 2 && end[0] == '|' && end[1] == '>') {
				sb_puts(&out, "<¦");
				sb_putb(&out, p + 2, (size_t)(end - p - 2));
				sb_puts(&out, "¦>");
				p = end + 2;
				continue;
			}
		}
		sb_putb(&out, p++, 1);
	}
	return out.p ? out.p : xstrdup("");
}

static void omni_escape_values(jinja_value *v) {
	if (v->type == JV_STRING) {
		char *escaped = omni_escape(v->as.s);
		free(v->as.s);
		v->as.s = escaped;
	} else if (v->type == JV_LIST) {
		for (uint32_t i = 0; i < v->as.list.n; i++)
			omni_escape_values(v->as.list.items[i]);
	} else if (v->type == JV_DICT) {
		for (jinja_dict_entry *e = v->as.dict; e; e = e->next) {
			char *key = omni_escape(e->key);
			free(e->key);
			e->key = key;
			omni_escape_values(e->val);
		}
	}
}

typedef struct {
	int32_t *ids;
	int		 n, cap;
} omni_tokens;

static status_code omni_tokenize(tokenizer *tok, const char *text, omni_tokens *tokens, int limit) {
	for (;;) {
		int n = tokenizer_encode_with_specials(tok, text, 0, tokens->ids, tokens->cap, NULL);
		if (n >= 0) {
			tokens->n = MIN(n, limit);
			return OK;
		}
		size_t bytes = strlen(text);
		if (bytes >= INT_MAX || tokens->cap > (int)bytes)
			return ERR_INVALID_ARG;
		tokens->cap = tokens->cap > (int)bytes / 2 ? (int)bytes + 1 : tokens->cap * 2;
		tokens->ids = xrealloc(tokens->ids, (size_t)tokens->cap * sizeof(int32_t));
	}
}

static status_code omni_prepare(context *c, jinja_program *tmpl, const decision_request *req,
								const decision_question *q, omni_tokens *out) {
	jinja_value *g = decision_prompt_values(req, q, NULL);
	omni_escape_values(g);
	jinja_dict_set(g, "audio", jinja_bool(false));
	jinja_dict_set(g, "sep", jinja_string(OMNI_SEP));
	jinja_dict_set(g, "mark_state", jinja_string(OMNI_STATE));
	jinja_dict_set(g, "mark_question", jinja_string(OMNI_QUESTION));
	jinja_dict_set(g, "mark_option", jinja_string(OMNI_OPTION));
	char	   *prompt = NULL;
	char		err[256];
	status_code st = jinja_render(tmpl, g, &prompt, err, sizeof(err));
	jinja_value_free(g);
	if (st != OK) {
		ERROR("lfm2-d1-omni: %s", err);
		return st;
	}
	int32_t	   *prefix	 = xmalloc(OMNI_MAX_TOKENS * sizeof(int32_t));
	int32_t	   *state	 = xmalloc(OMNI_MAX_TOKENS * sizeof(int32_t));
	int			n_prefix = 0, n_state = 0, n_tail = 0;
	int			budget	   = MAX(96, MIN(24 * q->n_options + 32, OMNI_MAX_TOKENS / 2));
	int			option_max = MAX(2, (budget - 3 * q->n_options) / q->n_options);
	omni_tokens span	   = {.ids = xmalloc(256 * sizeof(int32_t)), .cap = 256};
	bool		has_state  = false;
	out->ids			   = xmalloc(OMNI_MAX_TOKENS * sizeof(int32_t));
	for (char *piece = prompt; piece;) {
		char *next = strstr(piece, OMNI_SEP);
		if (next)
			*next = '\0';
		int	 limit	  = OMNI_MAX_TOKENS;
		bool is_state = !strncmp(piece, OMNI_STATE, strlen(OMNI_STATE));
		if (is_state)
			piece += strlen(OMNI_STATE);
		else if (!strncmp(piece, OMNI_QUESTION, strlen(OMNI_QUESTION))) {
			piece += strlen(OMNI_QUESTION);
			limit = MAX(16, budget);
		} else if (!strncmp(piece, OMNI_OPTION, strlen(OMNI_OPTION))) {
			piece += strlen(OMNI_OPTION);
			limit = option_max;
		}
		st = omni_tokenize(&c->tok, piece, &span, limit);
		if (st != OK)
			break;
		if (is_state) {
			if (has_state) {
				st = ERR_FORMAT;
				break;
			}
			memcpy(state, span.ids, (size_t)span.n * sizeof(int32_t));
			n_state	  = span.n;
			has_state = true;
		} else {
			int		*n	 = has_state ? &n_tail : &n_prefix;
			int32_t *dst = has_state ? out->ids : prefix;
			if (*n + span.n > OMNI_MAX_TOKENS) {
				st = ERR_INVALID_ARG;
				break;
			}
			memcpy(dst + *n, span.ids, (size_t)span.n * sizeof(int32_t));
			*n += span.n;
		}
		piece = next ? next + strlen(OMNI_SEP) : NULL;
	}
	if (!has_state || n_prefix + n_tail > OMNI_MAX_TOKENS)
		st = ERR_FORMAT;
	n_state = MIN(n_state, MAX(0, OMNI_MAX_TOKENS - n_prefix - n_tail));
	out->n	= n_prefix + n_state + n_tail;
	if (out->n > c->n_ctx)
		st = ERR_INVALID_ARG;
	if (st == OK) {
		memmove(out->ids + n_prefix + n_state, out->ids, (size_t)n_tail * sizeof(int32_t));
		memcpy(out->ids, prefix, (size_t)n_prefix * sizeof(int32_t));
		memcpy(out->ids + n_prefix, state, (size_t)n_state * sizeof(int32_t));
	}
	free(span.ids);
	free(prefix);
	free(state);
	free(prompt);
	return st;
}

static status_code omni_scores(context *c, jinja_program *tmpl, const decision_request *req,
							   const decision_question *q, float *scores, int *n_tokens) {
	omni_tokens tokens = {0};
	status_code st	   = omni_prepare(c, tmpl, req, q, &tokens);
	if (st != OK) {
		free(tokens.ids);
		return st;
	}
	int		 marker	 = tokenizer_find_token(&c->tok, "<|mask|>");
	uint8_t *outputs = xcalloc((size_t)tokens.n, 1);
	int		 n		 = 0;
	for (int i = 0; i < tokens.n; i++) {
		outputs[i] = tokens.ids[i] == marker;
		n += outputs[i];
	}
	if (marker < 0 || n != q->n_options)
		st = ERR_FORMAT;
	else if (c->interrupt)
		st = ERR_INTERRUPTED;
	else
		st = compute_forward_sequence_recipe(&c->m, &c->scratch, tokens.ids, tokens.n, outputs,
											 q->type, c->flash_attn, scores);
	if (st == OK)
		*n_tokens += tokens.n;
	free(outputs);
	free(tokens.ids);
	return st;
}

static const model_variant omni_variant = {
	.configure	= omni_configure,
	.load_layer = omni_load_layer,
	.recipe		= omni_recipe,
};

static const decision_ops omni_ops = {
	.type		 = "lfm2-d1-omni",
	.max_options = OMNI_MAX_OPTIONS,
	.variant	 = &omni_variant,
	.scores		 = omni_scores,
};

DECISION_REGISTER(lfm2_d1_omni, &omni_ops)
