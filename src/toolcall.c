#include "toolcall.h"

#include "log.h"

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
	size_t		 n_calls;
	json_object *calls;
};

void toolcall_buf_append(toolcall_buf *b, const char *s, size_t n) {
	if (!b || n == 0)
		return;
	if (b->len + n + 1 > b->cap) {
		b->cap = b->cap ? b->cap * 2 : 128;
		while (b->len + n + 1 > b->cap)
			b->cap *= 2;
		b->p = xrealloc(b->p, b->cap);
	}
	memcpy(b->p + b->len, s, n);
	b->len += n;
	b->p[b->len] = '\0';
}

void toolcall_buf_reset(toolcall_buf *b) {
	if (!b)
		return;
	b->len = 0;
	if (b->p)
		b->p[0] = '\0';
}

static size_t partial_marker_len(const char *s, size_t len, const char *marker) {
	if (!marker)
		return 0;
	size_t mlen = strlen(marker);
	if (mlen <= 1)
		return 0;
	size_t max = len < mlen - 1 ? len : mlen - 1;
	for (size_t k = max; k > 0; k--)
		if (memcmp(s + len - k, marker, k) == 0)
			return k;
	return 0;
}

static long find_marker(const char *s, size_t len, size_t from, const char *marker) {
	if (!marker || !marker[0])
		return -1;
	size_t mlen = strlen(marker);
	if (len < mlen)
		return -1;
	size_t last = len - mlen;
	if (from > last)
		return -1;
	for (size_t i = from; i <= last; i++)
		if (s[i] == marker[0] && memcmp(s + i, marker, mlen) == 0)
			return (long)i;
	return -1;
}

static size_t partial_open_hold(const char *s, size_t len, const marker_pair *fmt) {
	size_t h1 = partial_marker_len(s, len, fmt->open);
	size_t h2 = fmt->stop ? partial_marker_len(s, len, fmt->stop) : 0;
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
	char   buf[64];
	size_t n = s->pos - start;
	if (n >= sizeof(buf))
		return NULL;
	memcpy(buf, s->p + start, n);
	buf[n]	  = '\0';
	char *end = NULL;
	if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
		long long v = strtoll(buf, &end, 10);
		if (end && *end == '\0' && end != buf)
			return json_object_new_int64(v);
		return NULL;
	}
	double d = strtod(buf, &end);
	if (end && *end == '\0' && end != buf)
		return json_object_new_double(d);
	return NULL;
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
		char *key = xmalloc(kend - kstart + 1);
		memcpy(key, s->p + kstart, kend - kstart);
		key[kend - kstart] = '\0';
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
	size_t		 n	 = 0;
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
		if (val)
			json_object_array_add(arr, val);
		else
			json_object_array_put_idx(arr, n, NULL);
		n++;
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
	size_t nstart = s.pos;
	while (s.pos < s.len && s.p[s.pos] != '{' && !isspace((unsigned char)s.p[s.pos]))
		s.pos++;
	size_t nend = s.pos;
	cc_ws(&s);
	if (s.pos >= s.len || s.p[s.pos] != '{' || nend == nstart)
		return 0;
	char *name = xmalloc(nend - nstart + 1);
	memcpy(name, p + nstart, nend - nstart);
	name[nend - nstart] = '\0';
	json_object *args	= NULL;
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
	if (len > (size_t)INT_MAX)
		return NULL;
	json_tokener *tk  = json_tokener_new();
	json_object	 *obj = json_tokener_parse_ex(tk, s, (int)len);
	if (json_tokener_get_error(tk) != json_tokener_success) {
		json_tokener_free(tk);
		return NULL;
	}
	if (used)
		*used = (size_t)json_tokener_get_parse_end(tk);
	json_tokener_free(tk);
	return obj;
}

static json_object *json_args_of(json_object *obj) {
	json_object *jargs = NULL;
	if (obj &&
		(json_object_object_get_ex(obj, "arguments", &jargs) ||
		 json_object_object_get_ex(obj, "parameters", &jargs)) &&
		json_object_is_type(jargs, json_type_object))
		return json_object_get(jargs);
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
		json_object *jname	  = NULL;
		int			 has_name = json_object_object_get_ex(obj, "name", &jname) &&
								json_object_is_type(jname, json_type_string);
		if (!has_name) {
			json_object_put(obj);
			return 0;
		}
		*name_out = xstrdup(json_object_get_string(jname));
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
	char *name = xmalloc(nend - nstart + 1);
	memcpy(name, p + nstart, nend - nstart);
	name[nend - nstart] = '\0';
	*name_out			= name;
	*args_out			= json_obj_as_args(obj);
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
		char buf[64];
		if (len < sizeof(buf)) {
			memcpy(buf, s, len);
			buf[len]  = '\0';
			char *end = NULL;
			if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
				long long v = strtoll(buf, &end, 10);
				if (end && *end == '\0') {
					*out = json_object_new_int64(v);
					return 1;
				}
			}
			double d = strtod(buf, &end);
			if (end && *end == '\0') {
				*out = json_object_new_double(d);
				return 1;
			}
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
	while (pos < len && p[pos] != '(' && !isspace((unsigned char)p[pos]))
		pos++;
	size_t nend = pos;
	while (pos < len && p[pos] != '(')
		pos++;
	if (pos >= len || nend == nstart)
		return 0;
	pos++;
	char *name = xmalloc(nend - nstart + 1);
	memcpy(name, p + nstart, nend - nstart);
	name[nend - nstart] = '\0';
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
			while (klen > 0 && isspace((unsigned char)p[vstart + klen - 1]))
				klen--;
			while (vstart < pos && isspace((unsigned char)p[vstart]))
				vstart++;
			klen = pos - vstart;
			key	 = xmalloc(klen + 1);
			memcpy(key, p + vstart, klen);
			key[klen] = '\0';
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
	char *name = xmalloc(nend - nstart + 1);
	memcpy(name, p + nstart, nend - nstart);
	name[nend - nstart] = '\0';
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
			char *key = xmalloc(kend - kstart + 1);
			memcpy(key, p + kstart, kend - kstart);
			key[kend - kstart] = '\0';
			size_t vstart	   = pos;
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
			const char *vptr = p + vstart;
			size_t		vraw = vlen;
			while (vraw > 0 && isspace((unsigned char)*vptr)) {
				vptr++;
				vraw--;
			}
			while (vraw > 0 && isspace((unsigned char)vptr[vraw - 1]))
				vraw--;
			json_object *val = NULL;
			if (vraw > 0 && (vptr[0] == '{' || vptr[0] == '[')) {
				json_tokener *tk	 = json_tokener_new();
				json_object	 *parsed = json_tokener_parse_ex(tk, vptr, (int)vraw);
				if (json_tokener_get_error(tk) == json_tokener_success && parsed) {
					val = parsed;
				} else if (parsed) {
					json_object_put(parsed);
				}
				json_tokener_free(tk);
			}
			if (!val)
				val = json_object_new_string_len(vptr, (int)vraw);
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
	json_object_object_add(tc, "index", json_object_new_int((int)sc->n_calls));
	char idbuf[48];
	snprintf(idbuf, sizeof(idbuf), "call_%llx%03zu", (unsigned long long)time(NULL), sc->n_calls);
	json_object_object_add(tc, "id", json_object_new_string(idbuf));
	json_object_object_add(tc, "type", json_object_new_string("function"));
	json_object *fn = json_object_new_object();
	json_object_object_add(fn, "name", json_object_new_string(name));
	const char *args_str =
		args ? json_object_to_json_string_ext(args, JSON_C_TO_STRING_PLAIN) : "{}";
	json_object_object_add(fn, "arguments", json_object_new_string(args_str));
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
			long   k_stop	  = fmt->stop ? find_marker(c->p, c->len, sc->scan_pos, fmt->stop) : -1;
			size_t search_end = k_stop >= 0 ? (size_t)k_stop : c->len;
			for (size_t i = sc->scan_pos; i < search_end; i++) {
				if (c->p[i] != '{')
					continue;
				size_t		 used = 0;
				json_object *obj  = json_slice(c->p + i, c->len - i, &used);
				if (!obj) {
					emit_content(sc, sc->scan_pos, i);
					sc->scan_pos = i;
					return;
				}
				json_object *jname = NULL;
				if (!json_object_object_get_ex(obj, "name", &jname) ||
					!json_object_is_type(jname, json_type_string)) {
					json_object_put(obj);
					emit_content(sc, sc->scan_pos, i + used);
					sc->scan_pos = i + used;
					continue;
				}
				emit_content(sc, sc->scan_pos, i);
				char		*name = xstrdup(json_object_get_string(jname));
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
				sc->in_call	 = true;
				sc->scan_pos = 0;
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
			long k = find_marker(sc->buf.p, sc->buf.len, 0, fmt->close);
			if (k < 0)
				return;
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
			sc->in_call	 = false;
			sc->scan_pos = sc->content ? sc->content->len : 0;
			if (sc->buf.len > 0 && sc->content) {
				toolcall_buf_append(sc->content, sc->buf.p, sc->buf.len);
				toolcall_buf_reset(&sc->buf);
			}
			continue;
		}
	}
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
