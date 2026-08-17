#include "chat_template.h"

#include "log.h"

#include <ctype.h>
#include <string.h>

typedef struct {
	const char *start;
	const char *end;
} think_marker_pair;

static const think_marker_pair think_marker_pairs[] = {
	{"<think>", "</think>"},
	{"<|channel>", "<channel|>"},
	{"<|think|>", "<|/think|>"},
	{"<start_of_thought>", "<|end_of_thought|>"},
};

status_code chat_template_init(chat_template_state *cts, const gguf_ctx *g, const tokenizer *tok) {
	memset(cts, 0, sizeof(*cts));

	const char *tmpl_src = NULL;
	status_code s		 = gguf_get_str(g, "tokenizer.chat_template", &tmpl_src);
	if (s != OK) {
		ERROR("model GGUF has no 'tokenizer.chat_template'; this build requires it");
		return ERR_NOT_FOUND;
	}

	char errbuf[512];
	cts->prog = jinja_compile(tmpl_src, errbuf, sizeof(errbuf));
	if (!cts->prog) {
		ERROR("failed to compile tokenizer.chat_template: %s", errbuf);
		return ERR_FORMAT;
	}

	if (tok->bos_id >= 0 && (size_t)tok->bos_id < tok->n_tokens)
		cts->bos_token = xstrdup(tok->tokens[tok->bos_id].text);
	else
		cts->bos_token = xstrdup("");

	if (tok->eos_id >= 0 && (size_t)tok->eos_id < tok->n_tokens)
		cts->eos_token = xstrdup(tok->tokens[tok->eos_id].text);
	else
		cts->eos_token = xstrdup("");

	cts->think_start_id	  = -1;
	cts->think_end_id	  = -1;
	cts->think_start_text = NULL;
	for (size_t i = 0; i < ARRAY_LEN(think_marker_pairs); i++) {
		int32_t start_id = tokenizer_find_token(tok, think_marker_pairs[i].start);
		int32_t end_id	 = tokenizer_find_token(tok, think_marker_pairs[i].end);
		if (start_id >= 0 && end_id >= 0) {
			cts->think_start_id	  = start_id;
			cts->think_end_id	  = end_id;
			cts->think_start_text = think_marker_pairs[i].start;
			break;
		}
	}

	DEBUG("chat_template: think_start_id=%d think_end_id=%d", cts->think_start_id,
		  cts->think_end_id);

	cts->think_open		 = false;
	cts->last_render	 = xstrdup("");
	cts->enable_thinking = true;
	return OK;
}

void chat_template_clear_messages(chat_template_state *cts) {
	for (size_t i = 0; i < cts->n_messages; i++) {
		free(cts->messages[i].role);
		free(cts->messages[i].content);
	}
	free(cts->messages);
	cts->messages	  = NULL;
	cts->n_messages	  = 0;
	cts->cap_messages = 0;
	free(cts->last_render);
	cts->last_render = xstrdup("");
}

void chat_template_free(chat_template_state *cts) {
	if (!cts)
		return;
	for (size_t i = 0; i < cts->n_messages; i++) {
		free(cts->messages[i].role);
		free(cts->messages[i].content);
	}
	free(cts->messages);
	free(cts->last_render);
	free(cts->bos_token);
	free(cts->eos_token);
	jinja_program_free(cts->prog);
	memset(cts, 0, sizeof(*cts));
}

static jinja_value *build_globals(chat_template_state *cts, int add_generation_prompt) {
	jinja_value *g = jinja_dict();

	jinja_value *msgs = jinja_list();
	for (size_t i = 0; i < cts->n_messages; i++) {
		jinja_value *m = jinja_dict();
		jinja_dict_set(m, "role", jinja_string(cts->messages[i].role));
		jinja_dict_set(m, "content", jinja_string(cts->messages[i].content));
		jinja_list_append(msgs, m);
	}
	jinja_dict_set(g, "messages", msgs);
	jinja_dict_set(g, "add_generation_prompt", jinja_bool(add_generation_prompt));
	jinja_dict_set(g, "enable_thinking", jinja_bool(cts->enable_thinking));
	jinja_dict_set(g, "bos_token", jinja_string(cts->bos_token));
	jinja_dict_set(g, "eos_token", jinja_string(cts->eos_token));
	jinja_dict_set(g, "strftime_now", jinja_bool(1));
	return g;
}

status_code chat_template_add_turn(chat_template_state *cts, const char *role, const char *content,
								   int add_generation_prompt, char **out, char *errbuf,
								   size_t errbuf_len) {
	ARR_RESERVE(cts->messages, cts->n_messages, cts->cap_messages);
	cts->messages[cts->n_messages].role	   = xstrdup(role);
	cts->messages[cts->n_messages].content = xstrdup(content);
	cts->n_messages++;

	jinja_value *globals = build_globals(cts, add_generation_prompt);
	char		*rendered;
	status_code	 rc = jinja_render(cts->prog, globals, &rendered, errbuf, errbuf_len);
	jinja_value_free(globals);
	if (rc != OK)
		return rc;

	size_t prev_len = strlen(cts->last_render);
	size_t new_len	= strlen(rendered);

	const char *diff;
	size_t		diff_len;
	if (new_len < prev_len || strncmp(rendered, cts->last_render, prev_len) != 0) {
		diff	 = rendered;
		diff_len = new_len;
	} else {
		diff	 = rendered + prev_len;
		diff_len = new_len - prev_len;
	}
	*out = xstrdup(diff);

	cts->think_open = false;
	if (add_generation_prompt && cts->think_start_text) {
		size_t l = strlen(cts->think_start_text);
		if (l <= diff_len && strcmp(diff + diff_len - l, cts->think_start_text) == 0)
			cts->think_open = true;
	}

	free(cts->last_render);
	cts->last_render = rendered;
	return OK;
}