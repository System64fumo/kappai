#include "jinja.h"
#include "chat_template.h"
#include "test_core.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static jinja_value *mk_globals(const char *extra_key, jinja_value *extra_val) {
	jinja_value *g = jinja_dict();
	jinja_dict_set(g, "messages", jinja_list());
	jinja_dict_set(g, "add_generation_prompt", jinja_bool(0));
	if (extra_key)
		jinja_dict_set(g, extra_key, extra_val);
	return g;
}

static int render_ok(const char *tmpl, jinja_value *globals, char *out, size_t out_cap) {
	char errbuf[512];
	errbuf[0]			= '\0';
	jinja_program *prog = jinja_compile(tmpl, errbuf, sizeof(errbuf));
	if (!prog) {
		jinja_value_free(globals);
		snprintf(out, out_cap, "COMPILE-ERROR: %s", errbuf);
		return 0;
	}
	char	   *rendered = NULL;
	status_code rc		 = jinja_render(prog, globals, &rendered, errbuf, sizeof(errbuf));
	jinja_program_free(prog);
	jinja_value_free(globals);
	if (rc != OK || !rendered) {
		snprintf(out, out_cap, "RENDER-ERROR: %s", errbuf);
		return 0;
	}
	snprintf(out, out_cap, "%s", rendered);
	free(rendered);
	return 1;
}

static void record_jinja(const char *label, int ok, const char *detail_fmt, ...) {
	char	detail[256];
	va_list ap;
	va_start(ap, detail_fmt);
	vsnprintf(detail, sizeof(detail), detail_fmt, ap);
	va_end(ap);
	record_result(OPFAM_EDGE_CASE, label, ok ? V_PASS : V_FAIL, detail);
}

static void test_replace_method(void) {
	char out[256];
	int	 ok = render_ok("{{ 'abc'.replace('b', 'X') }}", mk_globals(NULL, NULL), out, sizeof(out));
	int	 match = ok && strcmp(out, "aXc") == 0;
	record_jinja("jinja.replace_string_method", match, "'abc'.replace('b','X') -> '%s'", out);

	ok	  = render_ok("{{ 'a-a-a'.replace('-', '+') }}", mk_globals(NULL, NULL), out, sizeof(out));
	match = ok && strcmp(out, "a+a+a") == 0;
	record_jinja("jinja.replace_all_occurrences", match, "'a-a-a'.replace -> '%s'", out);
}

static void test_is_null(void) {
	char out[256];

	int ok = render_ok("{% if x is null %}yes{% else %}no{% endif %}", mk_globals(NULL, NULL), out,
					   sizeof(out));
	int none_case = ok && strcmp(out, "yes") == 0;

	int ok2		   = render_ok("{% if x is null %}yes{% else %}no{% endif %}",
							   mk_globals("x", jinja_string("v")), out, sizeof(out));
	int bound_case = ok2 && strcmp(out, "no") == 0;
	record_jinja("jinja.is_null_test", none_case && bound_case, "unbound->'%s' bound->'%s'",
				 none_case ? "yes" : "?", bound_case ? "no" : "?");

	ok = render_ok("{% if x is none %}y{% endif %}", mk_globals(NULL, NULL), out, sizeof(out));
	record_jinja("jinja.is_none_alias_still_works", ok && strcmp(out, "y") == 0, "");
}

static void test_for_over_unbound(void) {
	char out[256];

	int ok = render_ok("A{% for t in tools %}[{{ t }}]{% endfor %}B", mk_globals(NULL, NULL), out,
					   sizeof(out));
	int unbound = ok && strcmp(out, "AB") == 0;

	int ok2		  = render_ok("A{% for t in tools %}[{{ t }}]{% endfor %}B",
							  mk_globals("tools", jinja_none()), out, sizeof(out));
	int none_case = ok2 && strcmp(out, "AB") == 0;

	jinja_value *tools = jinja_list();
	jinja_list_append(tools, jinja_string("t1"));
	jinja_list_append(tools, jinja_string("t2"));
	int ok3	  = render_ok("A{% for t in tools %}[{{ t }}]{% endfor %}B", mk_globals("tools", tools),
						  out, sizeof(out));
	int bound = ok3 && strcmp(out, "A[t1][t2]B") == 0;
	record_jinja("jinja.for_over_unbound_empty_loop", unbound && none_case && bound,
				 "unbound/none empty=%d%d bound->'%s'", unbound, none_case, out);
}

static void test_set_in_for_scope(void) {
	jinja_value *g	   = mk_globals(NULL, NULL);
	jinja_value *items = jinja_list();
	jinja_list_append(items, jinja_string("a"));
	jinja_list_append(items, jinja_string("b"));
	jinja_dict_set(g, "items", items);
	char out[256];
	int	 ok = render_ok(
		"{% set acc = 'outer' %}{% for i in items %}{% set acc = i %}{% endfor %}[{{ acc }}]", g,
		out, sizeof(out));
	int match = ok && strcmp(out, "[outer]") == 0;
	record_jinja("jinja.set_in_for_no_leak", match, "set-in-for non-persistence -> '%s'", out);
}

static void test_depth_cap(void) {
	size_t cap	= 4096;
	char  *tmpl = malloc(cap);
	tmpl[0]		= '\0';
	for (int i = 0; i < 200; i++)
		strncat(tmpl, "{% if true %}", cap - strlen(tmpl) - 1);
	strncat(tmpl, "x", cap - strlen(tmpl) - 1);

	char errbuf[512];
	errbuf[0]			= '\0';
	jinja_program *prog = jinja_compile(tmpl, errbuf, sizeof(errbuf));
	free(tmpl);
	int failed_cleanly = prog == NULL && errbuf[0] != '\0';
	if (prog)
		jinja_program_free(prog);
	record_jinja("jinja.depth_cap_parse_error", failed_cleanly,
				 failed_cleanly ? errbuf : "deep nesting compiled or crashed");

	jinja_value *g = mk_globals(NULL, NULL);
	char		 out[256];
	int ok = render_ok("{% if true %}{% if true %}ok{% endif %}{% endif %}", g, out, sizeof(out));
	record_jinja("jinja.nesting_under_cap_still_works", ok && strcmp(out, "ok") == 0, "");
}

static void test_range_cap(void) {
	jinja_value *g = mk_globals(NULL, NULL);
	char		 errbuf[512];
	errbuf[0]			= '\0';
	jinja_program *prog = jinja_compile("{% for i in range(0, 100000000) %}{{ i }}{% endfor %}",
										errbuf, sizeof(errbuf));
	if (!prog) {
		jinja_value_free(g);
		record_jinja("jinja.range_cap_error", 0, "template failed to compile");
		return;
	}
	char	   *rendered = NULL;
	status_code rc		 = jinja_render(prog, g, &rendered, errbuf, sizeof(errbuf));
	jinja_program_free(prog);
	jinja_value_free(g);
	int capped = rc != OK && rendered == NULL && errbuf[0] != '\0';
	free(rendered);
	record_jinja("jinja.range_cap_error", capped, capped ? errbuf : "huge range rendered");

	char out[256];
	int	 ok = render_ok("{% for i in range(3) %}{{ i }}{% endfor %}", mk_globals(NULL, NULL), out,
						sizeof(out));
	record_jinja("jinja.small_range_unaffected", ok && strcmp(out, "012") == 0, "");
}

static void test_for_over_dict(void) {
	jinja_value *g = mk_globals(NULL, NULL);
	jinja_value *d = jinja_dict();
	jinja_dict_set(d, "one", jinja_string("1"));
	jinja_dict_set(d, "two", jinja_string("2"));
	jinja_dict_set(d, "three", jinja_string("3"));
	jinja_dict_set(g, "d", d);

	char out[256];
	int	 ok	   = render_ok("{% for k in d %}{{ k }},{% endfor %}", g, out, sizeof(out));
	int	 match = ok && strcmp(out, "three,two,one,") == 0;
	record_jinja("jinja.for_over_dict_keys", match, "dict keys -> '%s'", out);
}

static void test_for_over_string(void) {
	jinja_value *g = mk_globals("s", jinja_string("abc"));
	char		 out[256];
	int			 ok	   = render_ok("{% for c in s %}[{{ c }}]{% endfor %}", g, out, sizeof(out));
	int			 match = ok && strcmp(out, "[a][b][c]") == 0;
	record_jinja("jinja.for_over_string_chars", match, "string chars -> '%s'", out);
}

static void test_loop_prev_next(void) {
	jinja_value *g	   = mk_globals(NULL, NULL);
	jinja_value *items = jinja_list();
	jinja_list_append(items, jinja_string("a"));
	jinja_list_append(items, jinja_string("b"));
	jinja_list_append(items, jinja_string("c"));
	jinja_dict_set(g, "items", items);

	char out[256];
	int	 ok = render_ok(
		"{% for x in items %}{{ x }}|{{ loop.previtem }}|{{ loop.nextitem }},{% endfor %}", g, out,
		sizeof(out));
	int match = ok && strcmp(out, "a||b,b|a|c,c|b|,") == 0;
	record_jinja("jinja.loop_prev_next", match, "prev/next items -> '%s'", out);
}

static void test_for_over_empty_dict(void) {
	jinja_value *g = mk_globals("d", jinja_dict());
	char		 out[256];
	int			 ok	   = render_ok("A{% for k in d %}[{{ k }}]{% endfor %}B", g, out, sizeof(out));
	int			 match = ok && strcmp(out, "AB") == 0;
	record_jinja("jinja.for_over_empty_dict", match, "empty dict -> '%s'", out);
}

static void test_logical_operands(void) {
	static const struct { const char *expr, *expected; } cases[] = {
		{"'hello' or 'fallback'", "hello"},
		{"'' or 'fallback'", "fallback"},
		{"'' or ''", ""},
		{"'hello' and 'kept'", "kept"},
		{"'' and 'kept'", ""},
		{"(['v'] or [])|tojson", "[\"v\"]"},
		{"([] and ['v'])|tojson", "[]"},
		{"true or raise_exception('unreachable')", "True"},
		{"false and raise_exception('unreachable')", "False"},
	};
	int pass = 1;
	for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
		char tmpl[256], out[256];
		snprintf(tmpl, sizeof(tmpl), "{{ %s }}", cases[i].expr);
		int ok = render_ok(tmpl, mk_globals(NULL, NULL), out, sizeof(out));
		if (!ok || strcmp(out, cases[i].expected)) {
			pass = 0;
			record_jinja("jinja.logical_operand_case", 0, "%s -> %s", cases[i].expr, out);
		}
	}
	record_jinja("jinja.logical_operands_and_short_circuit", pass, "9 operand/short-circuit cases");
}

static chat_template_state test_template_state(const char *tmpl) {
	chat_template_state c = {0};
	char err[256] = {0};
	c.prog = jinja_compile(tmpl, err, sizeof(err));
	c.bos_token = xstrdup("");
	c.eos_token = xstrdup("");
	c.last_render = xstrdup("");
	c.think_start_text = "<think>";
	c.think_end_text = "</think>";
	c.enable_thinking = true;
	return c;
}

static int typed_leaf(json_object *root, const char *key, enum json_type type) {
	json_object *v = NULL;
	return json_object_object_get_ex(root, key, &v) && json_object_get_type(v) == type;
}

static void test_json_numeric_ingestion(void) {
	const char *tmpl = "{{ tools[0].function.parameters|tojson }}|"
		"{{ messages[0].tool_calls[0].function.arguments|tojson }}";
	const char *payload = "{\"zero\":0,\"neg\":-7,\"max\":9223372036854775807,"
		"\"rate\":-1.25,\"whole_float\":1.0,\"nested\":[0,{\"x\":0.5}],"
		"\"yes\":true,\"no\":false,\"nil\":null,\"empty\":\"\","
		"\"digits\":\"007\",\"strzero\":\"0\"}";
	for (int string_args = 0; string_args < 2; string_args++) {
		chat_template_state c = test_template_state(tmpl);
		json_object *tools = json_tokener_parse("[{\"function\":{\"parameters\":{}}}]");
		json_object *params = json_tokener_parse(payload);
		json_object *fn = json_object_object_get(json_object_array_get_idx(tools, 0), "function");
		json_object_object_add(fn, "parameters", json_object_get(params));
		chat_template_set_tools(&c, tools, "auto");
		json_object *calls = json_tokener_parse("[{\"function\":{\"name\":\"f\"}}]");
		fn = json_object_object_get(json_object_array_get_idx(calls, 0), "function");
		json_object_object_add(fn, "arguments", string_args ? json_object_new_string(payload)
											 : json_object_get(params));
		chat_message msg = {.role = "assistant", .content = "", .tool_calls = calls};
		chat_template_add_message_ex(&c, &msg);
		char *rendered = NULL, err[256] = {0};
		int pass = chat_template_render(&c, 0, &rendered, err, sizeof(err)) == OK && rendered;
		if (pass) {
			char *sep = strchr(rendered, '|');
			pass = sep != NULL;
			if (pass) {
				*sep++ = '\0';
				json_object *left = json_tokener_parse(rendered);
				json_object *right = json_tokener_parse(sep);
				json_object *nested = NULL, *item = NULL;
				pass = left && right;
				json_object *sides[] = {left, right};
				for (size_t i = 0; pass && i < 2; i++) {
					json_object *o = sides[i];
					pass = typed_leaf(o,"zero",json_type_int) &&
						json_object_get_int(json_object_object_get(o,"zero")) == 0 &&
						typed_leaf(o,"neg",json_type_int) &&
						json_object_get_int(json_object_object_get(o,"neg")) == -7 &&
						typed_leaf(o,"max",json_type_int) &&
						json_object_get_int64(json_object_object_get(o,"max")) == INT64_MAX &&
						typed_leaf(o,"rate",json_type_double) &&
						json_object_get_double(json_object_object_get(o,"rate")) == -1.25 &&
						typed_leaf(o,"whole_float",json_type_double) &&
						typed_leaf(o,"yes",json_type_boolean) &&
						typed_leaf(o,"no",json_type_boolean) &&
						typed_leaf(o,"nil",json_type_null) &&
						typed_leaf(o,"empty",json_type_string) &&
						typed_leaf(o,"digits",json_type_string) &&
						!strcmp(json_object_get_string(json_object_object_get(o,"digits")), "007") &&
						typed_leaf(o,"strzero",json_type_string) &&
						json_object_object_get_ex(o,"nested",&nested) &&
						json_object_is_type(nested,json_type_array) &&
						json_object_is_type(json_object_array_get_idx(nested,0),json_type_int) &&
						json_object_object_get_ex(json_object_array_get_idx(nested,1),"x",&item) &&
						json_object_is_type(item,json_type_double);
				}
				if (left) json_object_put(left);
				if (right) json_object_put(right);
			}
		}
		record_jinja(string_args ? "template.numeric_string_arguments" :
			"template.numeric_object_arguments", pass, "typed schema and nested arguments %s", pass ? "ok" : err);
		free(rendered);
		json_object_put(params);
		json_object_put(calls);
		json_object_put(tools);
		chat_template_free(&c);
	}
}

static void test_numeric_expression_semantics(void) {
	char out[512];
	jinja_value *g = mk_globals("n", jinja_int(0));
	jinja_dict_set(g, "f", jinja_float(1.5));
	jinja_dict_set(g, "s", jinja_string("0"));
	const char *tmpl = "{{ [n, f, 3, 1.25, 0.1, 2+3, 4-2, range(2)|first, '007']|tojson }}|"
		"{{ n is number }}{{ f is float }}{{ s is string }}"
		"{{ n == 0 }}{{ n == s }}{{ f > 1.25 }}";
	int ok = render_ok(tmpl, g, out, sizeof(out));
	record_jinja("jinja.numeric_expressions_are_typed", ok &&
		!strcmp(out, "[0, 1.5, 3, 1.25, 0.1, 5, 2, 0, \"007\"]|TrueTrueTrueTrueFalseTrue"),
		"numeric literal/filter/comparison: %s", out);
}

static void test_nonassistant_thinking_text(void) {
	const char *roles[] = {"user", "system", "tool"};
	const char *contents[] = {"Explain <think>x</think> literally.",
		"Explain <think> literally.", "A </think> marker", "hello", ""};
	int pass = 1;
	for (int enabled = 0; enabled < 2; enabled++)
		for (size_t r = 0; r < 3; r++)
			for (size_t t = 0; t < 5; t++) {
				chat_template_state c = test_template_state("{{ messages[0].content }}");
				c.enable_thinking = enabled;
				char *preview = NULL, *diff = NULL, err[256] = {0};
				chat_template_preview_next_turn(&c, roles[r], contents[t], 0, &preview, err, sizeof(err));
				chat_template_add_turn(&c, roles[r], contents[t], 0, &diff, err, sizeof(err));
				if (!preview || !diff || strcmp(c.messages[0].content, contents[t]) ||
					strcmp(preview, c.last_render) || strcmp(diff, c.last_render))
					pass = 0;
				free(preview);
				free(diff);
				chat_template_free(&c);
			}
	record_jinja("template.nonassistant_markers_preview_commit", pass,
			"3 roles x 5 contents x 2 thinking settings");
}

void run_jinja_tests(void) {
	test_logical_operands();
	test_json_numeric_ingestion();
	test_numeric_expression_semantics();
	test_nonassistant_thinking_text();
	test_replace_method();
	test_is_null();
	test_for_over_unbound();
	test_for_over_dict();
	test_for_over_string();
	test_loop_prev_next();
	test_for_over_empty_dict();
	test_set_in_for_scope();
	test_depth_cap();
	test_range_cap();
	flush_family(OPFAM_EDGE_CASE);
}
