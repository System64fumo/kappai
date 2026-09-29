#include "chat_template.h"

#include "json_helpers.h"
#include "log.h"

#include <ctype.h>
#include <string.h>

static char *patch_template_source(const char *src) {
	const char *marker		= "last_query_index=messages|length - 1";
	const char *replacement = "last_query_index=0";
	const char *update_line = "{%- set ns.last_query_index = index %}";

	char *patched = xstrdup(src);

	char *pos = strstr(patched, marker);
	if (pos) {
		size_t marker_len = strlen(marker);
		size_t repl_len	  = strlen(replacement);
		char  *new_str	  = xmalloc(strlen(patched) - marker_len + repl_len + 1);
		size_t before	  = pos - patched;
		memcpy(new_str, patched, before);
		memcpy(new_str + before, replacement, repl_len);
		memcpy(new_str + before + repl_len, pos + marker_len, strlen(pos + marker_len) + 1);
		free(patched);
		patched = new_str;
		DEBUG("chat_template: patched last_query_index to stable value 0");
	}

	pos = strstr(patched, update_line);
	if (pos) {
		char *line_start = pos;
		while (line_start > patched && line_start[-1] != '\n')
			line_start--;
		char *line_end = pos + strlen(update_line);
		while (*line_end && *line_end != '\n')
			line_end++;
		if (*line_end == '\n')
			line_end++;
		size_t before_len = line_start - patched;
		size_t after_len  = strlen(line_end);
		char  *new_str	  = xmalloc(before_len + after_len + 1);
		memcpy(new_str, patched, before_len);
		memcpy(new_str + before_len, line_end, after_len + 1);
		free(patched);
		patched = new_str;
		DEBUG("chat_template: removed last_query_index update from scan loop");
	}

	return patched;
}

static jinja_value *json_to_jinja(const json_object *jo) {
	if (!jo || json_object_is_type(jo, json_type_null))
		return jinja_none();
	switch (json_object_get_type(jo)) {
	case json_type_boolean:
		return jinja_bool(json_object_get_boolean(jo) ? 1 : 0);
	case json_type_int:
		return jinja_int(json_object_get_int64(jo));
	case json_type_double:
		return jinja_float(json_object_get_double(jo));
	case json_type_string:
		return jinja_string(json_object_get_string((json_object *)jo));
	case json_type_array: {
		jinja_value	 *out = jinja_list();
		json_arr_iter it  = json_arr_begin((json_object *)jo);
		json_object	 *elem;
		while (json_arr_next(&it, &elem))
			jinja_list_append(out, json_to_jinja(elem));
		return out;
	}
	case json_type_object: {
		jinja_value *out = jinja_dict();
		json_object_object_foreach((json_object *)jo, key, val)
			jinja_dict_set(out, key, json_to_jinja(val));
		return out;
	}
	default:
		return jinja_none();
	}
}

static jinja_value *tool_calls_to_jinja(const json_object *tool_calls) {
	jinja_value *out = jinja_list();
	if (!tool_calls || !json_object_is_type(tool_calls, json_type_array))
		return out;
	json_arr_iter it = json_arr_begin((json_object *)tool_calls);
	json_object	 *tc;
	while (json_arr_next(&it, &tc)) {
		if (!json_object_is_type(tc, json_type_object))
			continue;
		jinja_value *d = jinja_dict();

		jinja_dict_set(d, "id", jinja_string(json_get_str(tc, "id", "")));

		jinja_dict_set(d, "type", jinja_string(json_get_str(tc, "type", "function")));

		jinja_value *fn	 = jinja_dict();
		json_object *jfn = json_get_obj(tc, "function");
		if (jfn) {
			jinja_dict_set(fn, "name", jinja_string(json_get_str(jfn, "name", "")));
			json_object *args = json_get(jfn, "arguments");
			if (args) {
				if (json_object_is_type(args, json_type_string)) {
					json_object *parsed = json_tokener_parse(json_object_get_string(args));
					if (parsed) {
						jinja_dict_set(fn, "arguments", json_to_jinja(parsed));
						json_object_put(parsed);
					} else {
						jinja_dict_set(fn, "arguments", jinja_string(json_object_get_string(args)));
					}
				} else {
					jinja_dict_set(fn, "arguments", json_to_jinja(args));
				}
			} else {
				jinja_dict_set(fn, "arguments", jinja_dict());
			}
		}
		jinja_dict_set(d, "function", fn);
		jinja_list_append(out, d);
	}
	return out;
}

status_code chat_template_init(chat_template_state *cts, const gguf_ctx *g, const tokenizer *tok) {
	memset(cts, 0, sizeof(*cts));

	const char *tmpl_src = NULL;
	status_code s		 = gguf_get_str(g, "tokenizer.chat_template", &tmpl_src);
	if (s != OK) {
		ERROR("model GGUF has no 'tokenizer.chat_template'; this build requires it");
		return ERR_NOT_FOUND;
	}

	char *tmpl_src_patched = patch_template_source(tmpl_src);

	char errbuf[512];
	cts->prog = jinja_compile(tmpl_src_patched, errbuf, sizeof(errbuf));
	free(tmpl_src_patched);
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

	const char		  *tmpl_src_for_probe = tmpl_src;
	const marker_pair *think			  = marker_probe_text(tmpl_src_for_probe, MARKER_THINKING);
	if (!think)
		think = marker_probe(tok, MARKER_THINKING);
	if (think) {
		cts->think_start_id	  = tokenizer_find_token(tok, think->open);
		cts->think_end_id	  = tokenizer_find_token(tok, think->close);
		cts->think_start_text = think->open;
		cts->think_end_text	  = think->close;
		cts->think_label_line = think->label_line;
	}

	cts->tool_fmt = marker_probe_text(tmpl_src_for_probe, MARKER_TOOL_CALL);
	if (!cts->tool_fmt)
		cts->tool_fmt = marker_probe(tok, MARKER_TOOL_CALL);

	DEBUG("chat_template: think_start_id=%d think_end_id=%d tool_fmt=%s", cts->think_start_id,
		  cts->think_end_id, cts->tool_fmt ? cts->tool_fmt->open : "none");

	cts->think_open		 = false;
	cts->last_render	 = xstrdup("");
	cts->enable_thinking = true;
	return OK;
}

static void notify_render_replaced(chat_template_state *cts) {
	if (cts->invalidate_cb)
		cts->invalidate_cb(cts->invalidate_ud);
}

void chat_template_clear_messages(chat_template_state *cts) {
	for (size_t i = 0; i < cts->n_messages; i++) {
		free(cts->messages[i].role);
		free(cts->messages[i].content);
		free(cts->messages[i].reasoning_content);
		free(cts->messages[i].tool_call_id);
		free(cts->messages[i].name);
		if (cts->messages[i].tool_calls)
			json_object_put(cts->messages[i].tool_calls);
	}
	free(cts->messages);
	cts->messages	  = NULL;
	cts->n_messages	  = 0;
	cts->cap_messages = 0;
	free(cts->last_render);
	cts->last_render = xstrdup("");
	notify_render_replaced(cts);
}

void chat_template_set_tools(chat_template_state *cts, json_object *tools,
							 const char *tool_choice) {
	if (cts->tools) {
		json_object_put(cts->tools);
		cts->tools = NULL;
	}
	free(cts->tool_choice);
	cts->tool_choice = NULL;
	if (tools && (!tool_choice || strcmp(tool_choice, "none") != 0)) {
		cts->tools		 = json_object_get(tools);
		cts->tool_choice = xstrdup(tool_choice ? tool_choice : "auto");
	}
	notify_render_replaced(cts);
}

void chat_template_free(chat_template_state *cts) {
	if (!cts)
		return;
	chat_template_clear_messages(cts);
	chat_template_set_tools(cts, NULL, NULL);
	free(cts->last_render);
	free(cts->bos_token);
	free(cts->eos_token);
	jinja_program_free(cts->prog);
	memset(cts, 0, sizeof(*cts));
}

static char *strip_thinking_spans(chat_template_state *cts, const char *content) {
	if (!content)
		return xstrdup("");
	if (!cts->think_start_text || !cts->think_end_text)
		return xstrdup(content);
	if (cts->keep_thinking_in_history)
		return xstrdup(content);

	const size_t slen = strlen(cts->think_start_text);
	const size_t elen = strlen(cts->think_end_text);
	char		*out  = xmalloc(strlen(content) + 1);
	size_t		 o	  = 0;
	const char	*p	  = content;

	for (;;) {
		const char *s = strstr(p, cts->think_start_text);
		if (!s)
			break;
		memcpy(out + o, p, (size_t)(s - p));
		o += (size_t)(s - p);
		const char *e = strstr(s + slen, cts->think_end_text);
		if (!e) {
			out[o] = '\0';
			return out;
		}
		p = e + elen;
	}
	strcpy(out + o, p);
	return out;
}

static const char *skip_newlines(const char *p) {
	while (*p == '\n')
		p++;
	return p;
}

void chat_template_split_thinking(const chat_template_state *cts, const char *raw,
								  char **out_reasoning, char **out_content) {
	*out_reasoning = NULL;
	*out_content   = xstrdup(raw ? raw : "");
	if (!raw || !cts->think_start_text || !cts->think_end_text)
		return;

	const char *e = strstr(raw, cts->think_end_text);
	if (!e)
		return;

	const char *rstart = raw;
	if (strncmp(rstart, cts->think_start_text, strlen(cts->think_start_text)) == 0)
		rstart += strlen(cts->think_start_text);
	if (cts->think_label_line) {
		const char *nl = strchr(rstart, '\n');
		rstart		   = nl ? nl + 1 : e;
	} else {
		rstart = skip_newlines(rstart);
	}
	if (rstart > e)
		rstart = e;

	free(*out_content);
	*out_content   = xstrdup(skip_newlines(e + strlen(cts->think_end_text)));
	*out_reasoning = xstrndup(rstart, (size_t)(e - rstart));
}

void chat_template_set_invalidate_cb(chat_template_state *cts, void (*cb)(void *), void *ud) {
	cts->invalidate_cb = cb;
	cts->invalidate_ud = ud;
}

static void simple_message(chat_message *m, const char *role, const char *content) {
	memset(m, 0, sizeof(*m));
	m->role	   = (char *)(role ? role : "");
	m->content = (char *)(content ? content : "");
}

static jinja_value *build_globals(chat_template_state *cts, const chat_message *extra,
								  size_t n_extra, int add_generation_prompt);

static status_code render_with_globals(chat_template_state *cts, const chat_message *extra,
									   size_t n_extra, int add_generation_prompt, char **out,
									   char *errbuf, size_t errbuf_len) {
	jinja_value *g	= build_globals(cts, extra, n_extra, add_generation_prompt);
	status_code	 rc = jinja_render(cts->prog, g, out, errbuf, errbuf_len);
	jinja_value_free(g);
	return rc == OK ? OK : ERR_FORMAT;
}

void chat_template_add_message(chat_template_state *cts, const char *role, const char *content) {
	chat_message m;
	simple_message(&m, role, content);
	chat_template_add_message_ex(cts, &m);
}

void chat_template_add_message_ex(chat_template_state *cts, const chat_message *msg) {
	ARR_RESERVE(cts->messages, cts->n_messages, cts->cap_messages);
	char		 *clean = msg->role && strcmp(msg->role, "assistant") == 0
							  ? strip_thinking_spans(cts, msg->content)
							  : xstrdup(msg->content ? msg->content : "");
	chat_message *dst	= &cts->messages[cts->n_messages];
	memset(dst, 0, sizeof(*dst));
	dst->role			   = xstrdup(msg->role ? msg->role : "");
	dst->content		   = clean;
	dst->reasoning_content = msg->reasoning_content ? xstrdup(msg->reasoning_content) : NULL;
	dst->tool_calls		   = msg->tool_calls ? json_object_get(msg->tool_calls) : NULL;
	dst->tool_call_id	   = msg->tool_call_id ? xstrdup(msg->tool_call_id) : NULL;
	dst->name			   = msg->name ? xstrdup(msg->name) : NULL;
	cts->n_messages++;
}

static void message_to_jinja(jinja_value *m, const chat_message *msg) {
	jinja_dict_set(m, "role", jinja_string(msg->role ? msg->role : ""));
	jinja_dict_set(m, "content", jinja_string(msg->content ? msg->content : ""));
	if (msg->reasoning_content)
		jinja_dict_set(m, "reasoning_content", jinja_string(msg->reasoning_content));

	if (msg->tool_calls)
		jinja_dict_set(m, "tool_calls", tool_calls_to_jinja(msg->tool_calls));
	if (msg->tool_call_id)
		jinja_dict_set(m, "tool_call_id", jinja_string(msg->tool_call_id));
	if (msg->name)
		jinja_dict_set(m, "name", jinja_string(msg->name));
}

static jinja_value *build_globals(chat_template_state *cts, const chat_message *extra,
								  size_t n_extra, int add_generation_prompt) {
	jinja_value *g	  = jinja_dict();
	jinja_value *msgs = jinja_list();

	for (size_t i = 0; i < cts->n_messages; i++) {
		jinja_value *m = jinja_dict();
		message_to_jinja(m, &cts->messages[i]);
		jinja_list_append(msgs, m);
	}
	for (size_t i = 0; i < n_extra; i++) {
		jinja_value *m = jinja_dict();
		message_to_jinja(m, &extra[i]);
		jinja_list_append(msgs, m);
	}

	jinja_dict_set(g, "messages", msgs);
	jinja_dict_set(g, "add_generation_prompt", jinja_bool(add_generation_prompt));
	jinja_dict_set(g, "enable_thinking", jinja_bool(cts->enable_thinking));
	jinja_dict_set(g, "bos_token", jinja_string(cts->bos_token));
	jinja_dict_set(g, "eos_token", jinja_string(cts->eos_token));
	jinja_dict_set(g, "strftime_now", jinja_bool(1));
	if (cts->tools)
		jinja_dict_set(g, "tools", json_to_jinja(cts->tools));
	return g;
}

status_code chat_template_render(chat_template_state *cts, int add_generation_prompt, char **out,
								 char *errbuf, size_t errbuf_len) {
	if (!cts || !cts->prog || !out)
		return ERR_INVALID_ARG;
	*out = NULL;
	return render_with_globals(cts, NULL, 0, add_generation_prompt, out, errbuf, errbuf_len);
}

status_code chat_template_preview_next_turn(chat_template_state *cts, const char *role,
											const char *content, int add_generation_prompt,
											char **out, char *errbuf, size_t errbuf_len) {
	if (!cts || !cts->prog || !out)
		return ERR_INVALID_ARG;
	*out = NULL;

	chat_message extra;
	simple_message(&extra, role ? role : "user", content);
	return render_with_globals(cts, &extra, 1, add_generation_prompt, out, errbuf, errbuf_len);
}

size_t chat_template_detect_static_prefix(chat_template_state *cts, const char *system) {
	if (!cts || !cts->prog)
		return 0;

	const char	*sys	 = (system && *system) ? system : "";
	chat_message sys_msg = {.role = (char *)"system", .content = (char *)sys};

	char  *r1 = NULL, *r2 = NULL;
	size_t prefix_len = 0;

	jinja_set_time_shift(0);
	status_code rc1 = render_with_globals(cts, &sys_msg, 1, 0, &r1, NULL, 0);
	if (rc1 != OK)
		goto out;

	jinja_set_time_shift(86400);
	status_code rc2 = render_with_globals(cts, &sys_msg, 1, 0, &r2, NULL, 0);
	if (rc2 != OK)
		goto out;

	prefix_len = str_lcp_len(r1, r2);
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
	chat_message m;
	simple_message(&m, role, content);
	return chat_template_add_turn_ex(cts, &m, add_generation_prompt, out, errbuf, errbuf_len);
}

status_code chat_template_add_turn_ex(chat_template_state *cts, const chat_message *msg,
									  int add_generation_prompt, char **out, char *errbuf,
									  size_t errbuf_len) {
	chat_template_add_message_ex(cts, msg);

	char	   *rendered;
	status_code rc =
		render_with_globals(cts, NULL, 0, add_generation_prompt, &rendered, errbuf, errbuf_len);
	if (rc != OK)
		return rc;

	size_t new_len = strlen(rendered);
	size_t common  = str_lcp_len(rendered, cts->last_render);

	const char *diff	 = rendered + common;
	size_t		diff_len = new_len - common;
	cts->think_open		 = false;
	if (add_generation_prompt && cts->think_start_text) {
		const char *last_open = NULL, *last_close = NULL, *p;
		p = diff;
		while ((p = strstr(p, cts->think_start_text)) != NULL) {
			last_open = p;
			p += strlen(cts->think_start_text);
		}
		if (cts->think_end_text) {
			p = diff;
			while ((p = strstr(p, cts->think_end_text)) != NULL) {
				last_close = p;
				p += strlen(cts->think_end_text);
			}
		}
		cts->think_open = last_open != NULL && (last_close == NULL || last_open > last_close);
	}

	*out = xstrdup(diff);

	free(cts->last_render);
	cts->last_render = rendered;
	return OK;
}

void chat_template_rewrite_last_assistant(chat_template_state *cts, const char *content,
										  const char *reasoning, json_object *tool_calls) {
	if (!cts || cts->n_messages == 0)
		return;
	chat_message *last = &cts->messages[cts->n_messages - 1];
	if (!last->role || strcmp(last->role, "assistant") != 0)
		return;

	free(last->content);
	last->content = xstrdup(content ? content : "");
	free(last->reasoning_content);
	last->reasoning_content = reasoning && reasoning[0] ? xstrdup(reasoning) : NULL;
	if (last->tool_calls) {
		json_object_put(last->tool_calls);
		last->tool_calls = NULL;
	}
	if (tool_calls)
		last->tool_calls = json_object_get(tool_calls);

	char *rendered = NULL;
	if (chat_template_render(cts, 0, &rendered, NULL, 0) == OK && rendered) {
		free(cts->last_render);
		cts->last_render = rendered;
	}
	cts->think_open = false;
	notify_render_replaced(cts);
}
