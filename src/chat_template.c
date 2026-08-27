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
			cts->think_end_text	  = think_marker_pairs[i].end;
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

static char *strip_thinking_span(chat_template_state *cts, const char *content) {
	if (!cts->think_start_text || !cts->think_end_text || !content)
		return xstrdup(content ? content : "");

	const char *start = strstr(content, cts->think_start_text);
	if (!start)
		return xstrdup(content);

	const char *after_start = start + strlen(cts->think_start_text);
	const char *end			= strstr(after_start, cts->think_end_text);
	if (!end)
		return xstrdup(content);

	const char *tail	 = end + strlen(cts->think_end_text);
	size_t		head_len = (size_t)(start - content);
	size_t		tail_len = strlen(tail);
	char	   *stripped = xmalloc(head_len + tail_len + 1);
	memcpy(stripped, content, head_len);
	memcpy(stripped + head_len, tail, tail_len);
	stripped[head_len + tail_len] = '\0';
	return stripped;
}

void chat_template_add_message(chat_template_state *cts, const char *role, const char *content) {
	ARR_RESERVE(cts->messages, cts->n_messages, cts->cap_messages);
	char *clean							   = strip_thinking_span(cts, content);
	cts->messages[cts->n_messages].role	   = xstrdup(role ? role : "");
	cts->messages[cts->n_messages].content = clean;
	cts->n_messages++;
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

static jinja_value *build_globals(chat_template_state *cts, const chat_message *extra,
								  size_t n_extra, int add_generation_prompt) {
	jinja_value *g	  = jinja_dict();
	jinja_value *msgs = jinja_list();

	for (size_t i = 0; i < cts->n_messages; i++) {
		jinja_value *m = jinja_dict();
		jinja_dict_set(m, "role", jinja_string(cts->messages[i].role));
		jinja_dict_set(m, "content", jinja_string(cts->messages[i].content));
		jinja_list_append(msgs, m);
	}
	for (size_t i = 0; i < n_extra; i++) {
		jinja_value *m = jinja_dict();
		jinja_dict_set(m, "role", jinja_string(extra[i].role));
		jinja_dict_set(m, "content", jinja_string(extra[i].content));
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

static size_t lcp_len(const char *a, const char *b) {
	size_t i = 0;
	while (a[i] && b[i] && a[i] == b[i])
		i++;
	return i;
}

status_code chat_template_preview_next_turn(chat_template_state *cts, const char *role,
											const char *content, int add_generation_prompt,
											char **out, char *errbuf, size_t errbuf_len) {
	if (!cts || !cts->prog || !out)
		return ERR_INVALID_ARG;
	*out = NULL;

	chat_message extra = {.role	   = (char *)(role ? role : "user"),
						  .content = (char *)(content ? content : "")};
	jinja_value *g	   = build_globals(cts, &extra, 1, add_generation_prompt);
	status_code	 rc	   = jinja_render(cts->prog, g, out, errbuf, errbuf_len);
	jinja_value_free(g);
	return rc == OK ? OK : ERR_FORMAT;
}

size_t chat_template_detect_static_prefix(chat_template_state *cts, const char *system) {
	if (!cts || !cts->prog)
		return 0;

	const char	*sys	 = (system && *system) ? system : "";
	chat_message sys_msg = {.role = (char *)"system", .content = (char *)sys};

	char  *r1 = NULL, *r2 = NULL;
	size_t prefix_len = 0;

	jinja_set_time_shift(0);
	jinja_value *g1	 = build_globals(cts, &sys_msg, 1, 0);
	status_code	 rc1 = jinja_render(cts->prog, g1, &r1, NULL, 0);
	jinja_value_free(g1);
	if (rc1 != OK)
		goto out;

	jinja_set_time_shift(86400);
	jinja_value *g2	 = build_globals(cts, &sys_msg, 1, 0);
	status_code	 rc2 = jinja_render(cts->prog, g2, &r2, NULL, 0);
	jinja_value_free(g2);
	if (rc2 != OK)
		goto out;

	prefix_len = lcp_len(r1, r2);
	DEBUG("static prefix: %zu bytes", prefix_len);

out:
	jinja_set_time_shift(0);
	free(r1);
	free(r2);
	return prefix_len;
}

status_code chat_template_add_turn(chat_template_state *cts, const char *role, const char *content,
								   int add_generation_prompt, char **out, char *errbuf,
								   size_t errbuf_len) {
	chat_template_add_message(cts, role, content);

	jinja_value *globals = build_globals(cts, NULL, 0, add_generation_prompt);
	char		*rendered;
	status_code	 rc = jinja_render(cts->prog, globals, &rendered, errbuf, errbuf_len);
	jinja_value_free(globals);
	if (rc != OK)
		return rc;

	size_t new_len = strlen(rendered);
	size_t common  = lcp_len(rendered, cts->last_render);

	const char *diff	 = rendered + common;
	size_t		diff_len = new_len - common;
	cts->think_open		 = false;
	if (add_generation_prompt && cts->think_start_text) {
		size_t l   = strlen(cts->think_start_text);
		size_t end = diff_len;
		while (end > 0 && isspace((unsigned char)diff[end - 1]))
			end--;
		if (l <= end && memcmp(diff + end - l, cts->think_start_text, l) == 0)
			cts->think_open = true;
	}

	*out = xstrdup(diff);

	free(cts->last_render);
	cts->last_render = rendered;
	return OK;
}
