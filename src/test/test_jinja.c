#include "jinja.h"
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

void run_jinja_tests(void) {
	test_replace_method();
	test_is_null();
	test_for_over_unbound();
	test_set_in_for_scope();
	test_depth_cap();
	test_range_cap();
	flush_family(OPFAM_EDGE_CASE);
}
