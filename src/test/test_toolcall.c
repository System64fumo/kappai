#include "test_core.h"
#include "toolcall.h"

#include <json-c/json.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define record_tc(label, ok, ...) record_resultf(OPFAM_TOOLCALL, (label), (ok), __VA_ARGS__)

static int parse_str(const marker_pair *fmt, const char *text, char **out_name,
					 json_object **out_args) {
	size_t		used	  = 0;
	int			saw_close = 0;
	status_code st = toolcall_parse(fmt, text, strlen(text), out_name, out_args, &used, &saw_close);
	if (st != OK)
		return 0;
	return 1;
}

static int json_eq(json_object *a, json_object *b) {
	if (!a && !b)
		return 1;
	if (!a || !b)
		return 0;
	const char *sa = json_object_to_json_string_ext(a, JSON_C_TO_STRING_PLAIN);
	const char *sb = json_object_to_json_string_ext(b, JSON_C_TO_STRING_PLAIN);
	return strcmp(sa, sb) == 0;
}

static void test_probe_text_qwen(void) {
	const char		  *tmpl = "some template <|tool_call> stuff <|tool_response> noise";
	const marker_pair *mp	= marker_probe_text(tmpl, MARKER_TOOL_CALL);
	int ok = mp && strcmp(mp->open, "<|tool_call>") == 0 &&
			 strcmp(mp->close, "<tool_call|>") == 0 && mp->payload == PAYLOAD_CALLCOLON;
	record_tc("marker.probe_text_qwen_callcolon", ok, "got open='%s' payload=%d",
			  mp ? mp->open : "(null)", mp ? (int)mp->payload : -1);
}

static void test_probe_text_auto_fallback(void) {
	const char		  *tmpl = "no markers here at all";
	const marker_pair *mp	= marker_probe_text(tmpl, MARKER_TOOL_CALL);
	int				   ok	= mp && mp->open[0] == '\0' && mp->payload == PAYLOAD_AUTO &&
							  strcmp(mp->stop, "<|eot_id|>") == 0;
	record_tc("marker.probe_text_auto_fallback", ok, "got payload=%d stop='%s'",
			  mp ? (int)mp->payload : -1, mp && mp->stop ? mp->stop : "(null)");
}

static void test_probe_text_thinking(void) {
	const char		  *tmpl = "chat template uses <think>...</think> blocks";
	const marker_pair *mp	= marker_probe_text(tmpl, MARKER_THINKING);
	int ok = mp && strcmp(mp->open, "<think>") == 0 && strcmp(mp->close, "</think>") == 0;
	record_tc("marker.probe_text_thinking", ok, "got open='%s'", mp ? mp->open : "(null)");
}

static void test_parse_auto_object_args(void) {
	marker_pair fmt = {
		.open		= "",
		.close		= "",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_AUTO,
		.stop		= "<|eot_id|>",
		.probe_hint = NULL,
	};
	const char	*text	 = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}";
	char		*name	 = NULL;
	json_object *args	 = NULL;
	int			 ok		 = parse_str(&fmt, text, &name, &args);
	int			 name_ok = ok && name && strcmp(name, "get_weather") == 0;
	int			 args_ok = ok && args && json_object_is_type(args, json_type_object);
	const char	*as		 = args ? json_object_to_json_string_ext(args, JSON_C_TO_STRING_PLAIN) : "";
	int			 city_ok = args_ok && json_object_object_get_ex(args, "city", NULL);
	record_tc("parse.auto_object_args", name_ok && args_ok && city_ok, "name='%s' args=%s",
			  name ? name : "(null)", as);
	free(name);
	if (args)
		json_object_put(args);
}

static void test_parse_auto_string_args(void) {
	marker_pair fmt = {
		.open		= "",
		.close		= "",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_AUTO,
		.stop		= "<|eot_id|>",
		.probe_hint = NULL,
	};
	const char	*text	 = "{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}";
	char		*name	 = NULL;
	json_object *args	 = NULL;
	int			 ok		 = parse_str(&fmt, text, &name, &args);
	int			 name_ok = ok && name && strcmp(name, "get_weather") == 0;
	int			 args_ok = ok && args && json_object_is_type(args, json_type_object);
	int			 city_ok = args_ok && json_object_object_get_ex(args, "city", NULL);
	record_tc("parse.auto_string_args", name_ok && args_ok && city_ok,
			  "name='%s' args_ok=%d city_ok=%d", name ? name : "(null)", args_ok, city_ok);
	free(name);
	if (args)
		json_object_put(args);
}

static void test_parse_auto_parameters(void) {
	marker_pair fmt = {
		.open		= "",
		.close		= "",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_AUTO,
		.stop		= "<|eot_id|>",
		.probe_hint = NULL,
	};
	const char	*text	 = "{\"name\":\"search\",\"parameters\":{\"q\":\"hello\"}}";
	char		*name	 = NULL;
	json_object *args	 = NULL;
	int			 ok		 = parse_str(&fmt, text, &name, &args);
	int			 name_ok = ok && name && strcmp(name, "search") == 0;
	int			 q_ok	 = ok && args && json_object_object_get_ex(args, "q", NULL);
	record_tc("parse.auto_parameters_spelling", name_ok && q_ok, "name='%s' q_ok=%d",
			  name ? name : "(null)", q_ok);
	free(name);
	if (args)
		json_object_put(args);
}

static void test_parse_auto_name_then_obj(void) {
	marker_pair fmt = {
		.open		= "",
		.close		= "",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_AUTO,
		.stop		= "<|eot_id|>",
		.probe_hint = NULL,
	};
	const char	*text	 = "search{\"q\":\"hi\",\"limit\":3}";
	char		*name	 = NULL;
	json_object *args	 = NULL;
	int			 ok		 = parse_str(&fmt, text, &name, &args);
	int			 name_ok = ok && name && strcmp(name, "search") == 0;
	int			 q_ok	 = ok && args && json_object_object_get_ex(args, "q", NULL);
	int			 lim_ok	 = ok && args && json_object_object_get_ex(args, "limit", NULL);
	record_tc("parse.auto_name_then_obj", name_ok && q_ok && lim_ok, "name='%s'",
			  name ? name : "(null)");
	free(name);
	if (args)
		json_object_put(args);
}

static void test_parse_funcargs_whitespace_name(void) {
	marker_pair fmt = {
		.open		= "<|tool_call_start|>",
		.close		= "<|tool_call_end|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_FUNCARGS,
		.stop		= "<|tool_call_end|>",
		.probe_hint = NULL,
	};
	const char	*text	  = "<|tool_call_start|>my func(key=\"hello\", count=3)<|tool_call_end|>";
	char		*name	  = NULL;
	json_object *args	  = NULL;
	int			 ok		  = parse_str(&fmt, text, &name, &args);
	int			 name_ok  = ok && name && strcmp(name, "my func") == 0;
	int			 key_ok	  = ok && args && json_object_object_get_ex(args, "key", NULL);
	int			 count_ok = ok && args && json_object_object_get_ex(args, "count", NULL);
	record_tc("parse.funcargs_whitespace_name", name_ok && key_ok && count_ok,
			  "name='%s' key_ok=%d count_ok=%d", name ? name : "(null)", key_ok, count_ok);
	free(name);
	if (args)
		json_object_put(args);
}

static void test_parse_funcargs_basic(void) {
	marker_pair fmt = {
		.open		= "<|tool_call_start|>",
		.close		= "<|tool_call_end|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_FUNCARGS,
		.stop		= "<|tool_call_end|>",
		.probe_hint = NULL,
	};
	const char	*text	 = "<|tool_call_start|>add(a=1, b=2)<|tool_call_end|>";
	char		*name	 = NULL;
	json_object *args	 = NULL;
	int			 ok		 = parse_str(&fmt, text, &name, &args);
	int			 name_ok = ok && name && strcmp(name, "add") == 0;
	int			 a_ok	 = ok && args && json_object_object_get_ex(args, "a", NULL);
	int			 b_ok	 = ok && args && json_object_object_get_ex(args, "b", NULL);
	record_tc("parse.funcargs_basic", name_ok && a_ok && b_ok, "name='%s' a_ok=%d b_ok=%d",
			  name ? name : "(null)", a_ok, b_ok);
	free(name);
	if (args)
		json_object_put(args);
}

static void test_parse_xmlfunc(void) {
	marker_pair fmt = {
		.open		= "<function=",
		.close		= "</function>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_XMLFUNC,
		.stop		= "<|im_end|>",
		.probe_hint = NULL,
	};
	const char	*text	 = "<function=get_weather><parameter=city>SF</parameter><parameter=unit>F</"
						   "parameter></function>";
	char		*name	 = NULL;
	json_object *args	 = NULL;
	int			 ok		 = parse_str(&fmt, text, &name, &args);
	int			 name_ok = ok && name && strcmp(name, "get_weather") == 0;
	int			 city_ok = ok && args && json_object_object_get_ex(args, "city", NULL);
	int			 unit_ok = ok && args && json_object_object_get_ex(args, "unit", NULL);
	record_tc("parse.xmlfunc_basic", name_ok && city_ok && unit_ok,
			  "name='%s' city_ok=%d unit_ok=%d", name ? name : "(null)", city_ok, unit_ok);
	free(name);
	if (args)
		json_object_put(args);
}

static void test_parse_callcolon(void) {
	marker_pair fmt = {
		.open		= "<|tool_call>",
		.close		= "<tool_call|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_CALLCOLON,
		.stop		= "<|tool_response>",
		.probe_hint = NULL,
	};
	const char *text =
		"<|tool_call>call: get_weather\n{city: <|\"|>SF<|\"|>, unit: <|\"|>F<|\"|>}<tool_call|>";
	char		*name	 = NULL;
	json_object *args	 = NULL;
	int			 ok		 = parse_str(&fmt, text, &name, &args);
	int			 name_ok = ok && name && strcmp(name, "get_weather") == 0;
	int			 city_ok = ok && args && json_object_object_get_ex(args, "city", NULL);
	int			 unit_ok = ok && args && json_object_object_get_ex(args, "unit", NULL);
	record_tc("parse.callcolon_basic", name_ok && city_ok && unit_ok,
			  "name='%s' city_ok=%d unit_ok=%d", name ? name : "(null)", city_ok, unit_ok);
	free(name);
	if (args)
		json_object_put(args);
}

static void test_parse_malformed_returns_error(void) {
	marker_pair fmt = {
		.open		= "<|tool_call>",
		.close		= "<tool_call|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_CALLCOLON,
		.stop		= "<|tool_response>",
		.probe_hint = NULL,
	};
	const char	*text	 = "<|tool_call>not_a_call<tool_call|>";
	char		*name	 = NULL;
	json_object *args	 = NULL;
	int			 ok		 = parse_str(&fmt, text, &name, &args);
	int			 fail_ok = !ok && name == NULL && args == NULL;
	record_tc("parse.malformed_returns_error", fail_ok, "ok=%d name=%p args=%p", ok, (void *)name,
			  (void *)args);
	free(name);
	if (args)
		json_object_put(args);
}

static void test_parse_auto_no_name_field(void) {
	marker_pair fmt = {
		.open		= "",
		.close		= "",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_AUTO,
		.stop		= "<|eot_id|>",
		.probe_hint = NULL,
	};
	const char	*text	 = "{\"args\":[1,2,3]}";
	char		*name	 = NULL;
	json_object *args	 = NULL;
	int			 ok		 = parse_str(&fmt, text, &name, &args);
	int			 fail_ok = !ok;
	record_tc("parse.auto_no_name_rejected", fail_ok, "ok=%d (expected 0)", ok);
	free(name);
	if (args)
		json_object_put(args);
}

static void test_scanner_single_call(void) {
	marker_pair fmt = {
		.open		= "<|tool_call>",
		.close		= "<tool_call|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_CALLCOLON,
		.stop		= "<|tool_response>",
		.probe_hint = NULL,
	};
	toolcall_buf	  content = {0};
	const char		 *text = "prelude\n<|tool_call>call: foo\n{k: <|\"|>v<|\"|>}<tool_call|>tail";
	toolcall_scanner *sc   = toolcall_scanner_new(&fmt, &content, NULL, NULL, NULL);
	toolcall_buf_append(&content, text, strlen(text));
	toolcall_scanner_feed(sc);
	toolcall_scanner_finish(sc);

	size_t		 n_calls	= toolcall_scanner_n_calls(sc);
	json_object *calls		= toolcall_scanner_calls(sc);
	int			 suppressed = toolcall_scanner_suppressed(sc);

	int n_ok		  = (n_calls == 1);
	int suppressed_ok = (suppressed == 0);
	int name_ok		  = 0;
	if (n_ok && calls) {
		json_object *first = json_object_array_get_idx(calls, 0);
		json_object *fn	   = NULL;
		if (json_object_object_get_ex(first, "function", &fn)) {
			json_object *nm = NULL;
			if (json_object_object_get_ex(fn, "name", &nm) &&
				json_object_is_type(nm, json_type_string)) {
				name_ok = (strcmp(json_object_get_string(nm), "foo") == 0);
			}
		}
	}

	record_tc("scanner.single_call", n_ok && suppressed_ok && name_ok,
			  "n_calls=%zu suppressed=%d name_ok=%d", n_calls, suppressed, name_ok);

	toolcall_scanner_free(sc);
	free(content.p);
}

static void test_scanner_multiple_calls(void) {
	marker_pair fmt = {
		.open		= "<|tool_call>",
		.close		= "<tool_call|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_CALLCOLON,
		.stop		= "<|tool_response>",
		.probe_hint = NULL,
	};
	toolcall_buf	  content = {0};
	const char		 *text	  = "<|tool_call>call: a\n{x: <|\"|>1<|\"|>}<tool_call|>"
								"middle text"
								"<|tool_call>call: b\n{y: <|\"|>2<|\"|>}<tool_call|>";
	toolcall_scanner *sc	  = toolcall_scanner_new(&fmt, &content, NULL, NULL, NULL);
	toolcall_buf_append(&content, text, strlen(text));
	toolcall_scanner_feed(sc);
	toolcall_scanner_finish(sc);

	size_t n_calls = toolcall_scanner_n_calls(sc);
	int	   n_ok	   = (n_calls == 2);

	int			 names_ok = 0;
	json_object *calls	  = toolcall_scanner_calls(sc);
	if (n_ok && calls) {
		const char *want[2] = {"a", "b"};
		names_ok			= 1;
		for (int i = 0; i < 2; i++) {
			json_object *tc = json_object_array_get_idx(calls, i);
			json_object *fn = NULL;
			if (!json_object_object_get_ex(tc, "function", &fn)) {
				names_ok = 0;
				break;
			}
			json_object *nm = NULL;
			if (!json_object_object_get_ex(fn, "name", &nm) ||
				!json_object_is_type(nm, json_type_string) ||
				strcmp(json_object_get_string(nm), want[i]) != 0) {
				names_ok = 0;
				break;
			}
		}
	}

	record_tc("scanner.multiple_calls", n_ok && names_ok, "n_calls=%zu (want 2)", n_calls);

	toolcall_scanner_free(sc);
	free(content.p);
}

static void test_scanner_truncated_salvage(void) {
	marker_pair fmt = {
		.open		= "<|tool_call>",
		.close		= "<tool_call|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_CALLCOLON,
		.stop		= "<|tool_response>",
		.probe_hint = NULL,
	};
	toolcall_buf	  content = {0};
	const char		 *text	  = "<|tool_call>call: salvaged\n{k: <|\"|>v<|\"|>}";
	toolcall_scanner *sc	  = toolcall_scanner_new(&fmt, &content, NULL, NULL, NULL);
	toolcall_buf_append(&content, text, strlen(text));
	toolcall_scanner_feed(sc);
	toolcall_scanner_finish(sc);

	size_t n_calls = toolcall_scanner_n_calls(sc);
	int	   n_ok	   = (n_calls == 1);

	int			 name_ok = 0;
	json_object *calls	 = toolcall_scanner_calls(sc);
	if (n_ok && calls) {
		json_object *tc = json_object_array_get_idx(calls, 0);
		json_object *fn = NULL;
		if (json_object_object_get_ex(tc, "function", &fn)) {
			json_object *nm = NULL;
			if (json_object_object_get_ex(fn, "name", &nm) &&
				json_object_is_type(nm, json_type_string)) {
				name_ok = (strcmp(json_object_get_string(nm), "salvaged") == 0);
			}
		}
	}

	record_tc("scanner.truncated_salvage", n_ok && name_ok, "n_calls=%zu name_ok=%d", n_calls,
			  name_ok);

	toolcall_scanner_free(sc);
	free(content.p);
}

static void test_scanner_suppressed_by_stop(void) {
	marker_pair fmt = {
		.open		= "<|tool_call>",
		.close		= "<tool_call|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_CALLCOLON,
		.stop		= "<|tool_response>",
		.probe_hint = NULL,
	};
	toolcall_buf content = {0};
	const char	*text = "<|tool_call>call: foo\n{k: <|\"|>v<|\"|>}<tool_call|><|tool_response>fake";
	toolcall_scanner *sc = toolcall_scanner_new(&fmt, &content, NULL, NULL, NULL);
	toolcall_buf_append(&content, text, strlen(text));
	toolcall_scanner_feed(sc);
	toolcall_scanner_finish(sc);

	int suppressed	  = toolcall_scanner_suppressed(sc);
	int suppressed_ok = (suppressed == 1);

	int content_ok = (content.len < strlen(text));

	record_tc("scanner.suppressed_by_stop", suppressed_ok && content_ok,
			  "suppressed=%d content_len=%zu (text_len=%zu)", suppressed, content.len,
			  strlen(text));

	toolcall_scanner_free(sc);
	free(content.p);
}

struct content_capture {
	char   buf[1024];
	size_t len;
};

static void on_content_capture(void *ud, const char *piece, size_t n) {
	struct content_capture *c = (struct content_capture *)ud;
	if (n + c->len + 1 > sizeof(c->buf)) {
		size_t take = sizeof(c->buf) - c->len - 1;
		if (take == 0)
			return;
		memcpy(c->buf + c->len, piece, take);
		c->len += take;
		c->buf[c->len] = '\0';
		return;
	}
	memcpy(c->buf + c->len, piece, n);
	c->len += n;
	c->buf[c->len] = '\0';
}

static void test_scanner_content_callback(void) {
	marker_pair fmt = {
		.open		= "<|tool_call>",
		.close		= "<tool_call|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_CALLCOLON,
		.stop		= "<|tool_response>",
		.probe_hint = NULL,
	};
	toolcall_buf		   content = {0};
	struct content_capture cap	   = {0};
	const char			  *text = "BEFORE<|tool_call>call: x\n{a: <|\"|>b<|\"|>}<tool_call|>AFTER";
	toolcall_scanner *sc = toolcall_scanner_new(&fmt, &content, on_content_capture, NULL, &cap);
	toolcall_buf_append(&content, text, strlen(text));
	toolcall_scanner_feed(sc);
	toolcall_scanner_finish(sc);

	int content_ok = (strcmp(cap.buf, "BEFOREAFTER") == 0);
	record_tc("scanner.content_callback", content_ok, "content='%s' (want 'BEFOREAFTER')", cap.buf);

	toolcall_scanner_free(sc);
	free(content.p);
}

static void test_scanner_id_uniqueness(void) {
	marker_pair fmt = {
		.open		= "<|tool_call>",
		.close		= "<tool_call|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_CALLCOLON,
		.stop		= "<|tool_response>",
		.probe_hint = NULL,
	};
	const char *text = "<|tool_call>call: x\n{k: <|\"|>v<|\"|>}<tool_call|>";

	toolcall_buf	  c1  = {0};
	toolcall_scanner *sc1 = toolcall_scanner_new(&fmt, &c1, NULL, NULL, NULL);
	toolcall_buf_append(&c1, text, strlen(text));
	toolcall_scanner_feed(sc1);
	toolcall_scanner_finish(sc1);

	toolcall_buf	  c2  = {0};
	toolcall_scanner *sc2 = toolcall_scanner_new(&fmt, &c2, NULL, NULL, NULL);
	toolcall_buf_append(&c2, text, strlen(text));
	toolcall_scanner_feed(sc2);
	toolcall_scanner_finish(sc2);

	json_object *calls1 = toolcall_scanner_calls(sc1);
	json_object *calls2 = toolcall_scanner_calls(sc2);
	int			 ok		= 0;
	if (calls1 && calls2 && json_object_array_length(calls1) == 1 &&
		json_object_array_length(calls2) == 1) {
		json_object *tc1 = json_object_array_get_idx(calls1, 0);
		json_object *tc2 = json_object_array_get_idx(calls2, 0);
		json_object *id1 = NULL, *id2 = NULL;
		json_object_object_get_ex(tc1, "id", &id1);
		json_object_object_get_ex(tc2, "id", &id2);
		if (id1 && id2 && json_object_is_type(id1, json_type_string) &&
			json_object_is_type(id2, json_type_string)) {
			const char *s1 = json_object_get_string(id1);
			const char *s2 = json_object_get_string(id2);
			ok			   = strcmp(s1, s2) != 0;
		}
	}

	record_tc("scanner.id_uniqueness", ok, "two scanners produced %s IDs",
			  ok ? "distinct" : "colliding");

	toolcall_scanner_free(sc1);
	toolcall_scanner_free(sc2);
	free(c1.p);
	free(c2.p);
}

static void test_scanner_streamed_capture(void) {
	marker_pair fmt = {
		.open		= "<|tool_call>",
		.close		= "<tool_call|>",
		.role		= MARKER_TOOL_CALL,
		.payload	= PAYLOAD_CALLCOLON,
		.stop		= "<|tool_response>",
		.probe_hint = NULL,
	};
	toolcall_buf content_streamed = {0};
	toolcall_buf content_oneshot  = {0};
	const char	*text			  = "<|tool_call>call: streamed\n{k: <|\"|>v<|\"|>}<tool_call|>";

	toolcall_scanner *sc_oneshot = toolcall_scanner_new(&fmt, &content_oneshot, NULL, NULL, NULL);
	toolcall_buf_append(&content_oneshot, text, strlen(text));
	toolcall_scanner_feed(sc_oneshot);
	toolcall_scanner_finish(sc_oneshot);

	toolcall_scanner *sc_streamed = toolcall_scanner_new(&fmt, &content_streamed, NULL, NULL, NULL);
	size_t			  tlen		  = strlen(text);
	for (size_t i = 0; i < tlen; i++) {
		if (toolcall_scanner_in_capture(sc_streamed))
			toolcall_scanner_feed_capture(sc_streamed, text + i, 1);
		else {
			toolcall_buf_append(&content_streamed, text + i, 1);
			toolcall_scanner_feed(sc_streamed);
		}
	}
	toolcall_scanner_finish(sc_streamed);

	json_object *calls_oneshot	= toolcall_scanner_calls(sc_oneshot);
	json_object *calls_streamed = toolcall_scanner_calls(sc_streamed);
	int			 ok =
		(toolcall_scanner_n_calls(sc_oneshot) == 1 && toolcall_scanner_n_calls(sc_streamed) == 1);
	if (ok && calls_oneshot && calls_streamed) {
		json_object *fn1 = NULL, *fn2 = NULL;
		json_object_object_get_ex(json_object_array_get_idx(calls_oneshot, 0), "function", &fn1);
		json_object_object_get_ex(json_object_array_get_idx(calls_streamed, 0), "function", &fn2);
		ok = fn1 && fn2 && json_eq(fn1, fn2);
	}

	record_tc("scanner.streamed_equals_oneshot", ok, "oneshot/streamed parity: %s",
			  ok ? "match" : "differ");

	toolcall_scanner_free(sc_oneshot);
	toolcall_scanner_free(sc_streamed);
	free(content_oneshot.p);
	free(content_streamed.p);
}

void run_toolcall_tests(void) {
	test_probe_text_qwen();
	test_probe_text_auto_fallback();
	test_probe_text_thinking();

	test_parse_auto_object_args();
	test_parse_auto_string_args();
	test_parse_auto_parameters();
	test_parse_auto_name_then_obj();
	test_parse_funcargs_basic();
	test_parse_funcargs_whitespace_name();
	test_parse_xmlfunc();
	test_parse_callcolon();

	test_parse_malformed_returns_error();
	test_parse_auto_no_name_field();

	test_scanner_single_call();
	test_scanner_multiple_calls();
	test_scanner_truncated_salvage();
	test_scanner_suppressed_by_stop();
	test_scanner_content_callback();
	test_scanner_id_uniqueness();
	test_scanner_streamed_capture();
}
