#include "chat_template.h"

#include "log.h"

#include <ctype.h>
#include <string.h>

static jinja_value *json_to_jinja(const json_object *jo) {
	if (!jo || json_object_is_type(jo, json_type_null))
		return jinja_none();
	switch (json_object_get_type(jo)) {
	case json_type_boolean:
		return jinja_bool(json_object_get_boolean(jo) ? 1 : 0);
	case json_type_int: {
		char buf[32];
		snprintf(buf, sizeof(buf), "%lld", (long long)json_object_get_int64(jo));
		return jinja_string(buf);
	}
	case json_type_double: {
		char buf[40];
		snprintf(buf, sizeof(buf), "%.17g", json_object_get_double(jo));
		return jinja_string(buf);
	}
	case json_type_string:
		return jinja_string(json_object_get_string((json_object *)jo));
	case json_type_array: {
		jinja_value *out = jinja_list();
		size_t		 n	 = json_object_array_length(jo);
		for (size_t i = 0; i < n; i++)
			jinja_list_append(out, json_to_jinja(json_object_array_get_idx(jo, i)));
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
	size_t n = json_object_array_length(tool_calls);
	for (size_t i = 0; i < n; i++) {
		json_object *tc = json_object_array_get_idx(tool_calls, i);
		if (!json_object_is_type(tc, json_type_object))
			continue;
		jinja_value *d = jinja_dict();

		json_object *id;
		jinja_dict_set(d, "id",
					   json_object_object_get_ex(tc, "id", &id) &&
							   json_object_is_type(id, json_type_string)
						   ? jinja_string(json_object_get_string(id))
						   : jinja_string(""));

		json_object *type;
		jinja_dict_set(d, "type",
					   json_object_object_get_ex(tc, "type", &type) &&
							   json_object_is_type(type, json_type_string)
						   ? jinja_string(json_object_get_string(type))
						   : jinja_string("function"));

		jinja_value *fn = jinja_dict();
		json_object *jfn;
		if (json_object_object_get_ex(tc, "function", &jfn) &&
			json_object_is_type(jfn, json_type_object)) {
			json_object *name;
			jinja_dict_set(fn, "name",
						   json_object_object_get_ex(jfn, "name", &name) &&
								   json_object_is_type(name, json_type_string)
							   ? jinja_string(json_object_get_string(name))
							   : jinja_string(""));
			json_object *args;
			if (json_object_object_get_ex(jfn, "arguments", &args)) {
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

	const char		  *tmpl_src_for_probe = tmpl_src;
	const marker_pair *think			  = marker_probe_text(tmpl_src_for_probe, MARKER_THINKING);
	if (!think)
		think = marker_probe(tok, MARKER_THINKING);
	if (think) {
		cts->think_start_id	  = tokenizer_find_token(tok, think->open);
		cts->think_end_id	  = tokenizer_find_token(tok, think->close);
		cts->think_start_text = think->open;
		cts->think_end_text	  = think->close;
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

void chat_template_add_message(chat_template_state *cts, const char *role, const char *content) {
	chat_message m = {.role	   = (char *)(role ? role : ""),
					  .content = (char *)(content ? content : "")};
	chat_template_add_message_ex(cts, &m);
}

void chat_template_add_message_ex(chat_template_state *cts, const chat_message *msg) {
	ARR_RESERVE(cts->messages, cts->n_messages, cts->cap_messages);
	char		 *clean = strip_thinking_spans(cts, msg->content);
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

static jinja_value *build_globals(chat_template_state *cts, const chat_message *extra,
								  size_t n_extra, int add_generation_prompt) {
	jinja_value *g	  = jinja_dict();
	jinja_value *msgs = jinja_list();

	for (size_t i = 0; i < cts->n_messages; i++) {
		jinja_value *m = jinja_dict();
		jinja_dict_set(m, "role", jinja_string(cts->messages[i].role));
		jinja_dict_set(m, "content", jinja_string(cts->messages[i].content));
		if (cts->messages[i].reasoning_content)
			jinja_dict_set(m, "reasoning_content",
						   jinja_string(cts->messages[i].reasoning_content));
		if (cts->messages[i].tool_calls)
			jinja_dict_set(m, "tool_calls", tool_calls_to_jinja(cts->messages[i].tool_calls));
		if (cts->messages[i].tool_call_id)
			jinja_dict_set(m, "tool_call_id", jinja_string(cts->messages[i].tool_call_id));
		if (cts->messages[i].name)
			jinja_dict_set(m, "name", jinja_string(cts->messages[i].name));
		jinja_list_append(msgs, m);
	}
	for (size_t i = 0; i < n_extra; i++) {
		jinja_value *m = jinja_dict();
		jinja_dict_set(m, "role", jinja_string(extra[i].role));
		jinja_dict_set(m, "content", jinja_string(extra[i].content));
		if (extra[i].reasoning_content)
			jinja_dict_set(m, "reasoning_content", jinja_string(extra[i].reasoning_content));
		if (extra[i].tool_calls)
			jinja_dict_set(m, "tool_calls", tool_calls_to_jinja(extra[i].tool_calls));
		if (extra[i].tool_call_id)
			jinja_dict_set(m, "tool_call_id", jinja_string(extra[i].tool_call_id));
		if (extra[i].name)
			jinja_dict_set(m, "name", jinja_string(extra[i].name));
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
	*out				 = NULL;
	jinja_value *globals = build_globals(cts, NULL, 0, add_generation_prompt);
	status_code	 rc		 = jinja_render(cts->prog, globals, out, errbuf, errbuf_len);
	jinja_value_free(globals);
	return rc == OK ? OK : ERR_FORMAT;
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
	chat_message m = {.role	   = (char *)(role ? role : ""),
					  .content = (char *)(content ? content : "")};
	return chat_template_add_turn_ex(cts, &m, add_generation_prompt, out, errbuf, errbuf_len);
}

status_code chat_template_add_turn_ex(chat_template_state *cts, const chat_message *msg,
									  int add_generation_prompt, char **out, char *errbuf,
									  size_t errbuf_len) {
	chat_template_add_message_ex(cts, msg);

	jinja_value *globals = build_globals(cts, NULL, 0, add_generation_prompt);
	char		*rendered;
	status_code	 rc = jinja_render(cts->prog, globals, &rendered, errbuf, errbuf_len);
	jinja_value_free(globals);
	if (rc != OK)
		return rc;

	size_t new_len = strlen(rendered);
	size_t common  = str_lcp_len(rendered, cts->last_render);

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
}
