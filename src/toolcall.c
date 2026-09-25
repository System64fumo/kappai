#define _GNU_SOURCE
#include "toolcall.h"

#include "json_helpers.h"
#include "log.h"

#include <ctype.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static atomic_ullong g_call_seq = ATOMIC_VAR_INIT(0);

struct toolcall_scanner {
	const marker_pair *fmt;

	toolcall_buf *content;

	toolcall_content_fn on_content;
	toolcall_call_fn	on_call;
	void			   *ud;

	bool		 in_call;
	bool		 suppressed;
	toolcall_buf buf;
	size_t		 scan_pos;
	size_t		 close_scan;
	size_t		 auto_stop_end;
	bool		 auto_stop_found;
	size_t		 auto_dead_i;
	size_t		 auto_dead_n;
	bool		 force_json_attempt;
	size_t		 n_calls;
	json_object *calls;
};

static long find_marker(const char *s, size_t len, size_t from, const char *marker) {
	if (!marker || !marker[0])
		return -1;
	size_t mlen = strlen(marker);
	if (len < mlen || from > len - mlen)
		return -1;
	const char *hit = memmem(s + from, len - from, marker, mlen);
	return hit ? (long)(hit - s) : -1;
}

static size_t partial_open_hold(const char *s, size_t len, const marker_pair *fmt) {
	size_t h1 = marker_tail_len(s, len, fmt->open);
	size_t h2 = fmt->stop ? marker_tail_len(s, len, fmt->stop) : 0;
	return h1 > h2 ? h1 : h2;
}

#define CC_STR_DELIM "<|\"|>"

typedef struct {
	const char *p;
	size_t		len;
	size_t		pos;
} cc_lex;

static void cc_ws(cc_lex *s) {
	while (s->pos < s->len && isspace((unsigned char)s->p[s->pos]))
		s->pos++;
}

static int cc_peek(cc_lex *s) {
	return s->pos < s->len ? (unsigned char)s->p[s->pos] : -1;
}

static bool cc_lit(cc_lex *s, const char *lit) {
	size_t l = strlen(lit);
	if (s->pos + l <= s->len && memcmp(s->p + s->pos, lit, l) == 0) {
		s->pos += l;
		return true;
	}
	return false;
}

static json_object *cc_string(cc_lex *s) {
	if (!cc_lit(s, CC_STR_DELIM))
		return NULL;
	const char *start = s->p + s->pos;
	size_t		dlen  = strlen(CC_STR_DELIM);
	const char *hit	  = NULL;
	for (size_t i = 0; i + dlen <= s->len - s->pos; i++) {
		if (memcmp(start + i, CC_STR_DELIM, dlen) == 0) {
			hit = start + i;
			s->pos += i + dlen;
			break;
		}
	}
	if (!hit)
		return NULL;
	return json_object_new_string_len(start, (int)(hit - start));
}

static json_object *parse_scalar_number(const char *s, size_t len) {
	char buf[64];
	if (len == 0 || len >= sizeof(buf))
		return NULL;
	memcpy(buf, s, len);
	buf[len]  = '\0';
	char *end = NULL;
	if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
		long long v = strtoll(buf, &end, 10);
		if (end && *end == '\0' && end != buf)
			return json_object_new_int64(v);
	}
	double d = strtod(buf, &end);
	if (end && *end == '\0' && end != buf)
		return json_object_new_double(d);
	return NULL;
}

static json_object *cc_number(cc_lex *s) {
	size_t start = s->pos;
	while (s->pos < s->len) {
		char ch = s->p[s->pos];
		if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' ||
			ch == 'E')
			s->pos++;
		else
			break;
	}
	if (s->pos == start)
		return NULL;
	return parse_scalar_number(s->p + start, s->pos - start);
}

static int cc_value(cc_lex *s, int depth, json_object **out);

static int cc_dict(cc_lex *s, int depth, json_object **out) {
	*out = NULL;
	if (!cc_lit(s, "{"))
		return 0;
	json_object *dict = json_object_new_object();
	cc_ws(s);
	if (cc_peek(s) == '}') {
		s->pos++;
		*out = dict;
		return 1;
	}
	for (;;) {
		cc_ws(s);
		size_t kstart = s->pos;
		while (s->pos < s->len && s->p[s->pos] != ':' && s->p[s->pos] != '}')
			s->pos++;
		if (s->pos >= s->len || s->p[s->pos] != ':')
			goto fail;
		size_t kend = s->pos;
		while (kend > kstart && isspace((unsigned char)s->p[kend - 1]))
			kend--;
		while (kstart < kend && isspace((unsigned char)s->p[kstart]))
			kstart++;
		s->pos++;
		json_object *val = NULL;
		if (!cc_value(s, depth + 1, &val))
			goto fail;
		char *key = xstrndup(s->p + kstart, kend - kstart);
		json_object_object_add(dict, key, val);
		free(key);
		cc_ws(s);
		if (cc_lit(s, ",")) {
			cc_ws(s);
			if (cc_peek(s) == '}') {
				s->pos++;
				*out = dict;
				return 1;
			}
			continue;
		}
		if (cc_lit(s, "}")) {
			*out = dict;
			return 1;
		}
		goto fail;
	}
fail:
	json_object_put(dict);
	return 0;
}

static int cc_array(cc_lex *s, int depth, json_object **out) {
	*out = NULL;
	if (!cc_lit(s, "["))
		return 0;
	json_object *arr = json_object_new_array();
	cc_ws(s);
	if (cc_peek(s) == ']') {
		s->pos++;
		*out = arr;
		return 1;
	}
	for (;;) {
		cc_ws(s);
		json_object *val = NULL;
		if (!cc_value(s, depth + 1, &val)) {
			json_object_put(arr);
			return 0;
		}
		json_object_array_add(arr, val ? val : json_object_new_null());
		cc_ws(s);
		if (cc_lit(s, ",")) {
			cc_ws(s);
			if (cc_peek(s) == ']') {
				s->pos++;
				*out = arr;
				return 1;
			}
			continue;
		}
		if (cc_lit(s, "]")) {
			*out = arr;
			return 1;
		}
		json_object_put(arr);
		return 0;
	}
}

static int cc_value(cc_lex *s, int depth, json_object **out) {
	*out = NULL;
	if (depth > 32)
		return 0;
	cc_ws(s);
	int ch = cc_peek(s);
	if (ch == '<') {
		*out = cc_string(s);
		return *out != NULL;
	}
	if (ch == '{')
		return cc_dict(s, depth, out);
	if (ch == '[')
		return cc_array(s, depth, out);
	if (cc_lit(s, "true")) {
		*out = json_object_new_boolean(1);
		return 1;
	}
	if (cc_lit(s, "false")) {
		*out = json_object_new_boolean(0);
		return 1;
	}
	if (cc_lit(s, "null"))
		return 1;
	*out = cc_number(s);
	return *out != NULL;
}

static int payload_callcolon(const char *p, size_t len, size_t pos, char **name_out,
							 json_object **args_out, size_t *end_out) {
	cc_lex s = {.p = p, .len = len, .pos = pos};
	cc_ws(&s);
	if (!cc_lit(&s, "call:"))
		return 0;

	cc_ws(&s);
	size_t nstart = s.pos;
	while (s.pos < s.len && s.p[s.pos] != '{' && !isspace((unsigned char)s.p[s.pos]))
		s.pos++;
	size_t nend = s.pos;
	cc_ws(&s);
	if (s.pos >= s.len || s.p[s.pos] != '{' || nend == nstart)
		return 0;
	char		*name = xstrndup(p + nstart, nend - nstart);
	json_object *args = NULL;
	if (!cc_dict(&s, 0, &args)) {
		free(name);
		return 0;
	}
	*name_out = name;
	*args_out = args;
	*end_out  = s.pos;
	return 1;
}

static json_object *json_slice(const char *s, size_t len, size_t *used) {
	return json_parse_len_ex(s, len, used, NULL);
}

static json_object *json_args_of(json_object *obj) {
	if (!obj)
		return json_object_new_object();
	json_object *jargs = json_get(obj, "arguments");
	if (!jargs)
		jargs = json_get(obj, "parameters");
	if (jargs) {
		if (json_object_is_type(jargs, json_type_object))
			return json_object_get(jargs);
		if (json_object_is_type(jargs, json_type_string)) {
			json_object *parsed = json_tokener_parse(json_object_get_string(jargs));
			if (parsed)
				return parsed;
		}
	}
	return json_object_new_object();
}

static json_object *json_obj_as_args(json_object *obj) {
	json_object *args = json_object_new_object();
	if (obj) {
		json_object_object_foreach(obj, key, val) {
			if (strcmp(key, "name") == 0)
				continue;
			json_object_object_add(args, key, json_object_get(val));
		}
	}
	return args;
}

static int payload_auto(const char *p, size_t len, size_t pos, char **name_out,
						json_object **args_out, size_t *end_out) {
	while (pos < len && isspace((unsigned char)p[pos]))
		pos++;
	if (pos >= len)
		return 0;

	size_t used = 0;
	if (p[pos] == '{') {
		json_object *obj = json_slice(p + pos, len - pos, &used);
		if (!obj)
			return 0;
		const char *jname = json_get_str(obj, "name", NULL);
		if (!jname) {
			json_object_put(obj);
			return 0;
		}
		*name_out = xstrdup(jname);
		*args_out = json_args_of(obj);
		json_object_put(obj);
		*end_out = pos + used;
		return 1;
	}

	size_t nstart = pos;
	while (pos < len && p[pos] != '{')
		pos++;
	size_t nend = pos;
	while (nend > nstart && (isspace((unsigned char)p[nend - 1]) || p[nend - 1] == ':'))
		nend--;
	while (nstart < nend && isspace((unsigned char)p[nstart]))
		nstart++;
	json_object *obj = json_slice(p + pos, len - pos, &used);
	if (!obj || nend == nstart) {
		json_object_put(obj);
		return 0;
	}
	char *name = xstrndup(p + nstart, nend - nstart);
	*name_out  = name;
	*args_out  = json_obj_as_args(obj);
	json_object_put(obj);
	*end_out = pos + used;
	return 1;
}

static int parse_funcarg_value(const char *s, size_t len, json_object **out) {
	while (len > 0 && isspace((unsigned char)*s)) {
		s++;
		len--;
	}
	while (len > 0 && isspace((unsigned char)s[len - 1]))
		len--;
	if (len == 0)
		return 0;
	if (*s == '"' || *s == '\'') {
		char q = *s;
		if (len < 2 || s[len - 1] != q)
			return 0;
		*out = json_object_new_string_len(s + 1, (int)(len - 2));
		return 1;
	}
	if (len == 4 && memcmp(s, "true", 4) == 0) {
		*out = json_object_new_boolean(1);
		return 1;
	}
	if (len == 5 && memcmp(s, "false", 5) == 0) {
		*out = json_object_new_boolean(0);
		return 1;
	}
	if (len == 4 && memcmp(s, "null", 4) == 0) {
		*out = json_object_new_null();
		return 1;
	}
	if (*s == '{' || *s == '[') {
		json_object *obj = json_slice(s, len, NULL);
		if (obj) {
			*out = obj;
			return 1;
		}
	}
	{
		json_object *num = parse_scalar_number(s, len);
		if (num) {
			*out = num;
			return 1;
		}
	}
	return 0;
}

static int payload_funcargs(const char *p, size_t len, size_t pos, char **name_out,
							json_object **args_out, size_t *end_out) {
	while (pos < len && isspace((unsigned char)p[pos]))
		pos++;
	if (pos < len && p[pos] == '[')
		pos++;
	while (pos < len && isspace((unsigned char)p[pos]))
		pos++;
	size_t nstart = pos;
	while (pos < len && p[pos] != '(')
		pos++;
	if (pos >= len)
		return 0;
	size_t nend = pos;
	while (nend > nstart && isspace((unsigned char)p[nend - 1]))
		nend--;
	while (nstart < nend && isspace((unsigned char)p[nstart]))
		nstart++;
	if (nend == nstart)
		return 0;
	pos++;
	char		*name	= xstrndup(p + nstart, nend - nstart);
	json_object *args	= json_object_new_object();
	size_t		 depth	= 0;
	size_t		 vstart = pos;
	char		*key	= NULL;
	for (;;) {
		if (pos >= len)
			goto done;
		char ch = p[pos];
		if (ch == '(' || ch == '{' || ch == '[') {
			depth++;
			pos++;
			continue;
		}
		if (ch == ')' || ch == '}' || ch == ']') {
			if (depth == 0) {
				pos++;
				goto done;
			}
			depth--;
			pos++;
			continue;
		}
		if (depth == 0 && ch == '=') {
			size_t klen = pos - vstart;
			while (vstart < pos && isspace((unsigned char)p[vstart])) {
				vstart++;
				klen--;
			}
			while (klen > 0 && isspace((unsigned char)p[vstart + klen - 1]))
				klen--;
			key = xstrndup(p + vstart, klen);
			pos++;
			vstart = pos;
			while (pos < len) {
				char c = p[pos];
				if (c == '(' || c == '{' || c == '[') {
					depth++;
				} else if (c == ')' || c == '}' || c == ']') {
					if (depth == 0)
						break;
					depth--;
				} else if (depth == 0 && (c == ',' || c == ')')) {
					break;
				}
				pos++;
			}
			json_object *val = NULL;
			if (parse_funcarg_value(p + vstart, pos - vstart, &val)) {
				json_object_object_add(args, key, val);
			}
			free(key);
			key = NULL;
			if (pos < len && p[pos] == ',') {
				pos++;
				vstart = pos;
			}
			continue;
		}
		pos++;
	}
done:
	*name_out = name;
	*args_out = args;
	*end_out  = pos;
	return 1;
}

static int payload_xmlfunc(const char *p, size_t len, size_t pos, char **name_out,
						   json_object **args_out, size_t *end_out) {
	while (pos < len && isspace((unsigned char)p[pos]))
		pos++;
	if (pos + 10 <= len && memcmp(p + pos, "<function=", 10) == 0)
		pos += 10;
	size_t nstart = pos;
	while (pos < len && p[pos] != '>')
		pos++;
	if (pos >= len)
		return 0;
	size_t nend = pos;
	while (nend > nstart && isspace((unsigned char)p[nend - 1]))
		nend--;
	while (nstart < nend && isspace((unsigned char)p[nstart]))
		nstart++;
	if (nend == nstart)
		return 0;
	char *name = xstrndup(p + nstart, nend - nstart);
	pos++;
	json_object *args = json_object_new_object();
	while (pos < len) {
		while (pos < len && isspace((unsigned char)p[pos]))
			pos++;
		if (pos + 10 <= len && memcmp(p + pos, "</function", 10) == 0)
			break;
		if (pos >= len || p[pos] != '<')
			break;
		pos++;
		size_t tag_start = pos;
		while (pos < len && p[pos] != '>' && p[pos] != '=')
			pos++;
		if (pos >= len)
			break;
		size_t tag_end = pos;
		while (tag_end > tag_start && isspace((unsigned char)p[tag_end - 1]))
			tag_end--;
		char   tag[64];
		size_t tag_len = tag_end - tag_start;
		if (tag_len >= sizeof(tag))
			tag_len = sizeof(tag) - 1;
		memcpy(tag, p + tag_start, tag_len);
		tag[tag_len] = '\0';
		if (p[pos] == '=') {
			pos++;
			size_t kstart = pos;
			while (pos < len && p[pos] != '>')
				pos++;
			if (pos >= len)
				break;
			size_t kend = pos;
			while (kend > kstart && isspace((unsigned char)p[kstart]))
				kstart++;
			while (kend > kstart && isspace((unsigned char)p[kend - 1]))
				kend--;
			pos++;
			if (kend == kstart)
				continue;
			char  *key	  = xstrndup(p + kstart, kend - kstart);
			size_t vstart = pos;
			char   close_buf[256];
			snprintf(close_buf, sizeof(close_buf), "</%s>", tag);
			size_t clen = strlen(close_buf);
			size_t vlen = 0;
			while (pos + clen <= len) {
				if (memcmp(p + pos, close_buf, clen) == 0)
					break;
				pos++;
				vlen++;
			}
			if (pos + clen > len) {
				free(key);
				break;
			}
			str_span	 vs	 = span_trim(p + vstart, vlen);
			json_object *val = NULL;
			if (vs.len > 0 && (vs.p[0] == '{' || vs.p[0] == '[')) {
				const char *perr;
				val = json_parse_len(vs.p, vs.len, &perr);
			}
			if (!val)
				val = json_object_new_string_len(vs.p, (int)vs.len);
			json_object_object_add(args, key, val);
			free(key);
			pos += clen;
		} else {
			pos++;
		}
	}
	*name_out = name;
	*args_out = args;
	*end_out  = pos;
	return 1;
}

static int payload_xmlargs(const char *p, size_t len, size_t pos, char **name_out,
						   json_object **args_out, size_t *end_out) {
	while (pos < len && isspace((unsigned char)p[pos]))
		pos++;
	size_t nstart = pos;
	while (pos < len && p[pos] != '<')
		pos++;
	if (pos >= len)
		return 0;
	size_t nend = pos;
	while (nend > nstart && isspace((unsigned char)p[nend - 1]))
		nend--;
	while (nstart < nend && isspace((unsigned char)p[nstart]))
		nstart++;
	if (nend == nstart)
		return 0;
	char *name = xstrndup(p + nstart, nend - nstart);

	json_object *args = json_object_new_object();
	for (;;) {
		if (pos + 9 <= len && memcmp(p + pos, "<arg_key>", 9) == 0) {
			pos += 9;
			size_t kstart = pos;
			long   kend	  = find_marker(p, len, pos, "</arg_key>");
			if (kend < 0)
				break;
			char *key = xstrndup(p + kstart, (size_t)kend - kstart);
			pos		  = (size_t)kend + strlen("</arg_key>");

			if (pos + 11 > len || memcmp(p + pos, "<arg_value>", 11) != 0) {
				free(key);
				break;
			}
			pos += 11;
			size_t vstart = pos;
			long   vend	  = find_marker(p, len, pos, "</arg_value>");
			if (vend < 0) {
				free(key);
				break;
			}
			size_t		 vraw = (size_t)vend - vstart;
			json_object *val  = NULL;
			if (vraw > 0 && (p[vstart] == '{' || p[vstart] == '[')) {
				const char *perr;
				val = json_parse_len(p + vstart, vraw, &perr);
			}
			if (!val)
				val = json_object_new_string_len(p + vstart, (int)vraw);
			json_object_object_add(args, key, val);
			free(key);
			pos = (size_t)vend + strlen("</arg_value>");
			continue;
		}
		break;
	}

	*name_out = name;
	*args_out = args;
	*end_out  = pos;
	return 1;
}

toolcall_scanner *toolcall_scanner_new(const marker_pair *fmt, toolcall_buf *content,
									   toolcall_content_fn on_content, toolcall_call_fn on_call,
									   void *ud) {
	if (!fmt || fmt->role != MARKER_TOOL_CALL)
		return NULL;
	toolcall_scanner *sc = xcalloc(1, sizeof(*sc));
	sc->fmt				 = fmt;
	sc->content			 = content;
	sc->on_content		 = on_content;
	sc->on_call			 = on_call;
	sc->ud				 = ud;
	sc->calls			 = json_object_new_array();
	return sc;
}

void toolcall_scanner_free(toolcall_scanner *sc) {
	if (!sc)
		return;
	free(sc->buf.p);
	if (sc->calls)
		json_object_put(sc->calls);
	free(sc);
}

bool toolcall_scanner_in_capture(const toolcall_scanner *sc) {
	return sc && sc->in_call;
}

bool toolcall_scanner_suppressed(const toolcall_scanner *sc) {
	return sc && sc->suppressed;
}

size_t toolcall_scanner_n_calls(const toolcall_scanner *sc) {
	return sc ? sc->n_calls : 0;
}

json_object *toolcall_scanner_calls(const toolcall_scanner *sc) {
	return sc ? sc->calls : NULL;
}

static void toolcall_scanner_sync(toolcall_scanner *sc) {
	if (!sc || !sc->content)
		return;
	if (sc->content->len < sc->scan_pos)
		sc->scan_pos = sc->content->len;
}

static void emit_content(toolcall_scanner *sc, size_t from, size_t to) {
	if (sc->on_content && to > from)
		sc->on_content(sc->ud, sc->content->p + from, to - from);
}

static void record_call(toolcall_scanner *sc, const char *name, json_object *args) {
	json_object *tc = json_object_new_object();
	json_set_int(tc, "index", (int)sc->n_calls);
	uint64_t seq = atomic_fetch_add_explicit(&g_call_seq, 1, memory_order_relaxed);
	char	 idbuf[64];
	snprintf(idbuf, sizeof(idbuf), "call_%llx_%016llx%03zu", (unsigned long long)time(NULL),
			 (unsigned long long)seq, sc->n_calls);
	json_set_str(tc, "id", idbuf);
	json_set_str(tc, "type", "function");
	json_object *fn = json_object_new_object();
	json_set_str(fn, "name", name);
	const char *args_str =
		args ? json_object_to_json_string_ext(args, JSON_C_TO_STRING_PLAIN) : "{}";
	json_set_str(fn, "arguments", args_str);
	json_object_object_add(tc, "function", fn);
	json_object_array_add(sc->calls, tc);
	if (sc->on_call)
		sc->on_call(sc->ud, (int)sc->n_calls, idbuf, name, args_str);
	sc->n_calls++;
}

static void advance(toolcall_scanner *sc) {
	const marker_pair *fmt = sc->fmt;
	for (int guard = 0; guard < 8192; guard++) {
		if (sc->suppressed)
			return;
		if (!fmt->open[0] && !sc->in_call) {
			toolcall_buf *c = sc->content;
			if (!c)
				return;
			if (c->len < sc->scan_pos) {
				sc->scan_pos = c->len;
				return;
			}
			long   k_stop;
			size_t slen = fmt->stop ? strlen(fmt->stop) : 0;
			if (slen > 0 && !sc->auto_stop_found && sc->auto_stop_end >= sc->scan_pos &&
				sc->auto_stop_end <= c->len) {
				size_t resume = sc->auto_stop_end > slen - 1 ? sc->auto_stop_end - (slen - 1) : 0;
				size_t from	  = resume > sc->scan_pos ? resume : sc->scan_pos;
				k_stop		  = find_marker(c->p, c->len, from, fmt->stop);
			} else {
				k_stop = fmt->stop ? find_marker(c->p, c->len, sc->scan_pos, fmt->stop) : -1;
			}
			sc->auto_stop_end	= c->len;
			sc->auto_stop_found = k_stop >= 0;
			size_t search_end	= k_stop >= 0 ? (size_t)k_stop : c->len;
			size_t tail			= c->len;
			while (tail > sc->scan_pos && isspace((unsigned char)c->p[tail - 1]))
				tail--;
			bool tail_closable =
				sc->force_json_attempt || (tail > sc->scan_pos && c->p[tail - 1] == '}');
			for (size_t i = sc->scan_pos; i < search_end; i++) {
				if (c->p[i] != '{')
					continue;
				if (!tail_closable) {
					emit_content(sc, sc->scan_pos, i);
					sc->scan_pos = i;
					return;
				}
				size_t		 avail = c->len - i;
				size_t		 used  = 0;
				json_object *obj   = NULL;
				if (i != sc->auto_dead_i || avail > sc->auto_dead_n) {
					enum json_tokener_error perr = json_tokener_success;
					obj = json_parse_len_ex_code(c->p + i, avail, &used, &perr);
					if (!obj && perr != json_tokener_continue) {
						sc->auto_dead_i = i;
						sc->auto_dead_n = avail;
					}
				}
				if (!obj) {
					emit_content(sc, sc->scan_pos, i);
					sc->scan_pos = i;
					return;
				}
				const char *jname = json_get_str(obj, "name", NULL);
				if (!jname) {
					json_object_put(obj);
					emit_content(sc, sc->scan_pos, i + used);
					sc->scan_pos = i + used;
					continue;
				}
				emit_content(sc, sc->scan_pos, i);
				char		*name = xstrdup(jname);
				json_object *args = json_args_of(obj);
				json_object_put(obj);
				record_call(sc, name, args);
				free(name);
				json_object_put(args);
				sc->scan_pos = i + used;
				if (k_stop >= 0 && sc->scan_pos >= (size_t)k_stop) {
					c->len = (size_t)k_stop;
					if (c->p)
						c->p[c->len] = '\0';
					sc->scan_pos   = (size_t)k_stop;
					sc->suppressed = true;
					return;
				}
				break;
			}
			if (k_stop >= 0) {
				emit_content(sc, sc->scan_pos, (size_t)k_stop);
				c->len = (size_t)k_stop;
				if (c->p)
					c->p[c->len] = '\0';
				sc->scan_pos   = (size_t)k_stop;
				sc->suppressed = true;
				return;
			}
			emit_content(sc, sc->scan_pos, search_end);
			sc->scan_pos = search_end;
			return;
		}
		if (!sc->in_call) {
			toolcall_buf *c = sc->content;
			if (!c)
				return;
			if (c->len < sc->scan_pos) {
				sc->scan_pos = c->len;
				return;
			}
			long k_open = find_marker(c->p, c->len, sc->scan_pos, fmt->open);
			long k_stop = fmt->stop ? find_marker(c->p, c->len, sc->scan_pos, fmt->stop) : -1;
			long k;
			bool is_stop;
			if (k_open >= 0 && (k_stop < 0 || k_open <= k_stop)) {
				k		= k_open;
				is_stop = false;
			} else if (k_stop >= 0) {
				k		= k_stop;
				is_stop = true;
			} else {
				k		= -1;
				is_stop = false;
			}
			if (k >= 0) {
				emit_content(sc, sc->scan_pos, (size_t)k);
				if (is_stop) {
					c->len = (size_t)k;
					if (c->p)
						c->p[c->len] = '\0';
					sc->scan_pos   = (size_t)k;
					sc->suppressed = true;
					return;
				}
				toolcall_buf_append(&sc->buf, c->p + k, c->len - (size_t)k);
				c->len = (size_t)k;
				if (c->p)
					c->p[c->len] = '\0';
				sc->in_call	   = true;
				sc->close_scan = 0;
				sc->scan_pos   = 0;
				continue;
			}
			size_t hold		= partial_open_hold(c->p, c->len, fmt);
			size_t emit_end = c->len - hold;
			if (emit_end < sc->scan_pos)
				emit_end = sc->scan_pos;
			emit_content(sc, sc->scan_pos, emit_end);
			sc->scan_pos = emit_end;
			return;
		} else {
			if (!fmt->close)
				return;
			long k = find_marker(sc->buf.p, sc->buf.len, sc->close_scan, fmt->close);
			if (k < 0) {
				size_t mlen	   = strlen(fmt->close);
				size_t keep	   = mlen > 0 ? mlen - 1 : 0;
				sc->close_scan = sc->buf.len > keep ? sc->buf.len - keep : 0;
				return;
			}
			size_t total		= (size_t)k + strlen(fmt->close);
			size_t buf_consumed = 0;
			bool   any_ok		= false;
			while (buf_consumed < (size_t)k) {
				char		*name	   = NULL;
				json_object *args	   = NULL;
				size_t		 consumed  = 0;
				int			 saw_close = 0;
				status_code	 st;
				if (any_ok) {
					size_t plen = (size_t)k - buf_consumed;
					int	   ok	= payload_funcargs(sc->buf.p + buf_consumed, plen, 0, &name, &args,
												   &consumed);
					st			= ok ? OK : ERR_FORMAT;
					saw_close	= 0;
				} else {
					st = toolcall_parse(fmt, sc->buf.p + buf_consumed, (size_t)k - buf_consumed,
										&name, &args, &consumed, &saw_close);
				}
				if (st != OK) {
					if (!any_ok)
						WARN("unparseable tool call block (%zu bytes); treating it as content", k);
					break;
				}
				record_call(sc, name, args);
				free(name);
				json_object_put(args);
				any_ok = true;
				buf_consumed += consumed;
				while (buf_consumed < (size_t)k &&
					   (sc->buf.p[buf_consumed] == ',' ||
						isspace((unsigned char)sc->buf.p[buf_consumed])))
					buf_consumed++;
			}
			if (!any_ok && sc->content)
				toolcall_buf_append(sc->content, sc->buf.p, total);
			memmove(sc->buf.p, sc->buf.p + total, sc->buf.len - total);
			sc->buf.len -= total;
			if (sc->buf.p)
				sc->buf.p[sc->buf.len] = '\0';
			sc->in_call	   = false;
			sc->close_scan = 0;
			sc->scan_pos   = sc->content ? sc->content->len : 0;
			if (sc->buf.len > 0 && sc->content) {
				toolcall_buf_append(sc->content, sc->buf.p, sc->buf.len);
				toolcall_buf_reset(&sc->buf);
			}
			continue;
		}
	}
	WARN("toolcall scanner: iteration guard fired (input likely malformed; "
		 "fmt payload=%d in_call=%d scan_pos=%zu buf_len=%zu content_len=%zu)",
		 (int)fmt->payload, (int)sc->in_call, sc->scan_pos, sc->buf.len,
		 sc->content ? sc->content->len : (size_t)0);
}

void toolcall_scanner_feed_capture(toolcall_scanner *sc, const char *piece, size_t n) {
	if (!sc || !sc->in_call)
		return;
	toolcall_buf_append(&sc->buf, piece, n);
	advance(sc);
}

void toolcall_scanner_feed(toolcall_scanner *sc) {
	if (!sc)
		return;
	advance(sc);
}

void toolcall_scanner_finish(toolcall_scanner *sc) {
	if (!sc)
		return;
	if (sc->suppressed) {
		toolcall_buf_reset(&sc->buf);
		sc->in_call = false;
		return;
	}
	if (sc->in_call && sc->buf.len > 0) {
		char		*name	   = NULL;
		json_object *args	   = NULL;
		size_t		 consumed  = 0;
		int			 saw_close = 0;
		status_code	 st =
			toolcall_parse(sc->fmt, sc->buf.p, sc->buf.len, &name, &args, &consumed, &saw_close);
		if (st == OK) {
			DEBUG("salvaged truncated tool call '%s' (no closing marker)", name);
			record_call(sc, name, args);
		} else {
			WARN("incomplete tool call at end of generation; emitting it as content");
			size_t before = sc->content ? sc->content->len : 0;
			if (sc->content) {
				toolcall_buf_append(sc->content, sc->buf.p, sc->buf.len);
				emit_content(sc, before, sc->content->len);
			}
		}
		free(name);
		json_object_put(args);
		toolcall_buf_reset(&sc->buf);
		sc->in_call = false;
	}
	if (!sc->in_call && sc->content) {
		toolcall_scanner_sync(sc);
		if (!sc->fmt->open[0] && sc->scan_pos < sc->content->len) {
			sc->force_json_attempt = true;
			advance(sc);
			sc->force_json_attempt = false;
		}
		if (sc->content->len > sc->scan_pos) {
			emit_content(sc, sc->scan_pos, sc->content->len);
			sc->scan_pos = sc->content->len;
		}
	}
}

status_code toolcall_parse(const marker_pair *fmt, const char *text, size_t len, char **name_out,
						   json_object **args_out, size_t *consumed, int *saw_close) {
	if (!fmt || !text || !name_out || !args_out || !consumed || !saw_close)
		return ERR_INVALID_ARG;
	*name_out  = NULL;
	*args_out  = NULL;
	*consumed  = 0;
	*saw_close = 0;

	size_t olen = strlen(fmt->open);
	if (len < olen || memcmp(text, fmt->open, olen) != 0)
		return ERR_FORMAT;

	char		*name = NULL;
	json_object *args = NULL;
	size_t		 end  = 0;
	int			 ok;
	switch (fmt->payload) {
	case PAYLOAD_CALLCOLON:
		ok = payload_callcolon(text, len, olen, &name, &args, &end);
		break;
	case PAYLOAD_FUNCARGS:
		ok = payload_funcargs(text, len, olen, &name, &args, &end);
		break;
	case PAYLOAD_XMLFUNC:
		ok = payload_xmlfunc(text, len, olen, &name, &args, &end);
		break;
	case PAYLOAD_XMLARGS:
		ok = payload_xmlargs(text, len, olen, &name, &args, &end);
		break;
	default:
		ok = payload_auto(text, len, olen, &name, &args, &end);
		break;
	}
	if (!ok)
		return ERR_FORMAT;

	size_t pos = end;
	while (pos < len && isspace((unsigned char)text[pos]))
		pos++;
	size_t clen	  = fmt->close ? strlen(fmt->close) : 0;
	int	   closed = clen > 0 && pos + clen <= len && memcmp(text + pos, fmt->close, clen) == 0;

	*name_out  = name;
	*args_out  = args;
	*consumed  = closed ? pos + clen : end;
	*saw_close = closed;
	return OK;
}
