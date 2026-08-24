#include "jinja.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct {
	char  *p;
	size_t len;
	size_t cap;
} strbuf;

typedef struct {
	char **msgs;
	size_t n;
	size_t cap;
} strlist;

static void strlist_add(strlist *sl, char *msg) {
	ARR_RESERVE(sl->msgs, sl->n, sl->cap);
	sl->msgs[sl->n++] = msg;
}

static void strlist_free(strlist *sl) {
	if (!sl)
		return;
	for (size_t i = 0; i < sl->n; i++)
		free(sl->msgs[i]);
	free(sl->msgs);
	sl->msgs = NULL;
	sl->n = sl->cap = 0;
}

static void sb_init(strbuf *b) {
	b->cap	= 64;
	b->len	= 0;
	b->p	= xmalloc(b->cap);
	b->p[0] = '\0';
}

static void sb_append(strbuf *b, const char *s, size_t n) {
	if (b->len + n + 1 > b->cap) {
		while (b->len + n + 1 > b->cap)
			b->cap *= 2;
		b->p = xrealloc(b->p, b->cap);
	}
	memcpy(b->p + b->len, s, n);
	b->len += n;
	b->p[b->len] = '\0';
}

static void sb_append_str(strbuf *b, const char *s) {
	sb_append(b, s, strlen(s));
}

static void strlist_report(const strlist *sl, char *errbuf, size_t errbuf_len) {
	strbuf sb;
	sb_init(&sb);
	for (size_t i = 0; i < sl->n; i++) {
		if (i)
			sb_append_str(&sb, "\n");
		sb_append_str(&sb, sl->msgs[i]);
	}
	if (errbuf_len)
		snprintf(errbuf, errbuf_len, "%s", sb.p);
	free(sb.p);
}

typedef struct {
	jinja_value **vals;
	size_t		  n;
	size_t		  cap;
} jinja_arena;

static _Thread_local jinja_arena *g_render_arena	 = NULL;
static _Thread_local time_t		  g_jinja_time_shift = 0;

void jinja_set_time_shift(time_t seconds) {
	g_jinja_time_shift = seconds;
}

static void arena_free_shallow(jinja_value *v) {
	switch (v->type) {
	case JV_STRING:
		free(v->as.s);
		break;
	case JV_DICT: {
		jinja_dict_entry *e = v->as.dict;
		while (e) {
			jinja_dict_entry *next = e->next;
			free(e->key);
			free(e);
			e = next;
		}
		break;
	}
	case JV_LIST:
		free(v->as.list.items);
		break;
	default:
		break;
	}
	free(v);
}

static void arena_free_all(jinja_arena *a) {
	if (!a)
		return;
	for (size_t i = 0; i < a->n; i++)
		arena_free_shallow(a->vals[i]);
	free(a->vals);
}

static void arena_track(jinja_value *v) {
	jinja_arena *a = g_render_arena;
	if (!a)
		return;
	if (!a->cap) {
		a->cap	= 64;
		a->vals = xrealloc(a->vals, a->cap * sizeof(*a->vals));
	}
	ARR_RESERVE(a->vals, a->n, a->cap);
	a->vals[a->n++] = v;
}

jinja_value *jinja_none(void) {
	jinja_value *v = xmalloc(sizeof(*v));
	v->type		   = JV_NONE;
	arena_track(v);
	return v;
}

jinja_value *jinja_bool(int b) {
	jinja_value *v = xmalloc(sizeof(*v));
	v->type		   = JV_BOOL;
	v->as.b		   = !!b;
	arena_track(v);
	return v;
}

jinja_value *jinja_string(const char *s) {
	jinja_value *v = xmalloc(sizeof(*v));
	v->type		   = JV_STRING;
	v->as.s		   = xstrdup(s ? s : "");
	arena_track(v);
	return v;
}

jinja_value *jinja_string_n(const char *s, size_t n) {
	jinja_value *v = xmalloc(sizeof(*v));
	v->type		   = JV_STRING;
	v->as.s		   = xmalloc(n + 1);
	if (n)
		memcpy(v->as.s, s, n);
	v->as.s[n] = '\0';
	arena_track(v);
	return v;
}

jinja_value *jinja_dict(void) {
	jinja_value *v = xmalloc(sizeof(*v));
	v->type		   = JV_DICT;
	v->as.dict	   = NULL;
	arena_track(v);
	return v;
}

jinja_value *jinja_list(void) {
	jinja_value *v	 = xmalloc(sizeof(*v));
	v->type			 = JV_LIST;
	v->as.list.items = NULL;
	v->as.list.n	 = 0;
	arena_track(v);
	return v;
}

void jinja_dict_set(jinja_value *d, const char *key, jinja_value *val) {
	for (jinja_dict_entry *e = d->as.dict; e; e = e->next) {
		if (!strcmp(e->key, key)) {
			e->val = val;
			return;
		}
	}
	jinja_dict_entry *e = xmalloc(sizeof(*e));
	e->key				= xstrdup(key);
	e->val				= val;
	e->next				= d->as.dict;
	d->as.dict			= e;
}

void jinja_list_append(jinja_value *l, jinja_value *val) {
	l->as.list.items = xrealloc(l->as.list.items, (l->as.list.n + 1) * sizeof(jinja_value *));
	l->as.list.items[l->as.list.n++] = val;
}

void jinja_value_free(jinja_value *v) {
	if (!v)
		return;
	switch (v->type) {
	case JV_STRING:
		free(v->as.s);
		break;
	case JV_DICT: {
		jinja_dict_entry *e = v->as.dict;
		while (e) {
			jinja_dict_entry *next = e->next;
			free(e->key);
			jinja_value_free(e->val);
			free(e);
			e = next;
		}
		break;
	}
	case JV_LIST:
		for (size_t i = 0; i < v->as.list.n; i++)
			jinja_value_free(v->as.list.items[i]);
		free(v->as.list.items);
		break;
	default:
		break;
	}
	free(v);
}

static jinja_value *dict_get(const jinja_value *d, const char *key) {
	if (!d || d->type != JV_DICT)
		return NULL;
	for (jinja_dict_entry *e = d->as.dict; e; e = e->next)
		if (!strcmp(e->key, key))
			return e->val;
	return NULL;
}

static int truthy(const jinja_value *v) {
	if (!v)
		return 0;
	switch (v->type) {
	case JV_NONE:
		return 0;
	case JV_BOOL:
		return v->as.b;
	case JV_STRING:
		return v->as.s[0] != '\0';
	case JV_LIST:
		return v->as.list.n > 0;
	case JV_DICT:
		return v->as.dict != NULL;
	case JV_MACRO:
		return 1;
	}
	return 0;
}

static const char *value_as_cstr(const jinja_value *v) {
	if (!v)
		return "";
	if (v->type == JV_STRING)
		return v->as.s;
	if (v->type == JV_NONE)
		return "";
	if (v->type == JV_BOOL)
		return v->as.b ? "True" : "False";
	return "";
}

typedef enum {
	TOK_TEXT,
	TOK_EXPR_OPEN,
	TOK_EXPR_CLOSE,
	TOK_STMT_OPEN,
	TOK_STMT_CLOSE,
	TOK_IDENT,
	TOK_STRING,
	TOK_NUMBER,
	TOK_DOT,
	TOK_LBRACKET,
	TOK_RBRACKET,
	TOK_LPAREN,
	TOK_RPAREN,
	TOK_COMMA,
	TOK_COLON,
	TOK_OP,
	TOK_PIPE,
	TOK_UNKNOWN,
	TOK_EOF,
} tok_type;

typedef struct {
	tok_type	type;
	const char *start;
	size_t		len;
	int			strip_left;
	int			strip_right;
} token;

typedef struct {
	const char *src;
	size_t		len;
	size_t		pos;
	token	   *toks;
	size_t		n_toks;
	size_t		cap_toks;
	strlist		diagnostics;
	char	   *errbuf;
	size_t		errbuf_len;
	size_t		text_start;
	int			next_strip_left;
	int			next_trim_nl;
	size_t		tag_start;
	int			tag_is_expr;
	int			tag_strip_after;
} lexer;

static void lex_push(lexer *lx, tok_type type, const char *start, size_t len, int strip_left,
					 int strip_right) {
	if (lx->n_toks == lx->cap_toks) {
		lx->cap_toks = lx->cap_toks ? lx->cap_toks * 2 : 64;
		lx->toks	 = xrealloc(lx->toks, lx->cap_toks * sizeof(token));
	}
	token *t	   = &lx->toks[lx->n_toks++];
	t->type		   = type;
	t->start	   = start;
	t->len		   = len;
	t->strip_left  = strip_left;
	t->strip_right = strip_right;
}

static size_t lex_lstrip_text_end(const char *src, size_t tstart, size_t tag_start) {
	size_t e = tag_start;
	while (e > tstart && (src[e - 1] == ' ' || src[e - 1] == '\t'))
		e--;
	if (e == tag_start)
		return tag_start;
	if (e > tstart)
		return src[e - 1] == '\n' ? e : tag_start;
	if (tstart == 0 || src[tstart - 1] == '\n')
		return e;
	return tag_start;
}

static void lex_flush_text(lexer *lx, const char *src, size_t tag_start,
						   int strip_left_of_text_before, int apply_lstrip) {
	size_t tstart = lx->text_start;
	size_t tend	  = tag_start;
	if (!lx->next_strip_left && lx->next_trim_nl && tend > tstart && src[tstart] == '\n')
		tstart++;
	lx->next_trim_nl = 0;
	if (apply_lstrip && !lx->next_strip_left)
		tend = lex_lstrip_text_end(src, tstart, tag_start);
	if (tend > tstart || strip_left_of_text_before)
		lex_push(lx, TOK_TEXT, src + tstart, tend - tstart, lx->next_strip_left,
				 strip_left_of_text_before);
}

static int is_ident_start(char c) {
	return isalpha((unsigned char)c) || c == '_';
}
static int is_ident_char(char c) {
	return isalnum((unsigned char)c) || c == '_';
}

static const char *g_op_words[] = {"and", "or", "not", "in", "is", NULL};

static int lex_word_is_op(const char *s, size_t n) {
	for (int i = 0; g_op_words[i]; i++)
		if (strlen(g_op_words[i]) == n && !strncmp(g_op_words[i], s, n))
			return 1;
	return 0;
}

static status_code lex_comment(lexer *lx, const char *src) {
	size_t start = lx->pos;
	lx->pos += 2;
	int strip_left_of_text_before = 0;
	if (lx->pos < lx->len && src[lx->pos] == '-') {
		strip_left_of_text_before = 1;
		lx->pos++;
	}
	int strip_left_of_text_after = 0;
	for (;;) {
		if (lx->pos >= lx->len) {
			snprintf(lx->errbuf, lx->errbuf_len, "unterminated comment at offset %zu", start);
			return ERR_FORMAT;
		}
		if (src[lx->pos] == '-' && !strncmp(src + lx->pos + 1, "#}", 2)) {
			lx->pos += 3;
			strip_left_of_text_after = 1;
			break;
		}
		if (!strncmp(src + lx->pos, "#}", 2)) {
			lx->pos += 2;
			break;
		}
		lx->pos++;
	}
	lex_flush_text(lx, src, start, strip_left_of_text_before, 1);
	lx->text_start		= lx->pos;
	lx->next_strip_left = strip_left_of_text_after;
	lx->next_trim_nl	= !strip_left_of_text_after;
	return OK;
}

static status_code lex_tag_inner(lexer *lx, const char *src) {
	const char *close = lx->tag_is_expr ? "}}" : "%}";
	for (;;) {
		while (lx->pos < lx->len && isspace((unsigned char)src[lx->pos]))
			lx->pos++;
		if (lx->pos >= lx->len) {
			snprintf(lx->errbuf, lx->errbuf_len, "unterminated tag at offset %zu", lx->tag_start);
			return ERR_FORMAT;
		}
		if (src[lx->pos] == '-' && !strncmp(src + lx->pos + 1, close, 2)) {
			lex_push(lx, lx->tag_is_expr ? TOK_EXPR_CLOSE : TOK_STMT_CLOSE, src + lx->pos, 3, 1, 0);
			lx->pos += 3;
			lx->tag_strip_after = 1;
			break;
		}
		if (!strncmp(src + lx->pos, close, 2)) {
			lex_push(lx, lx->tag_is_expr ? TOK_EXPR_CLOSE : TOK_STMT_CLOSE, src + lx->pos, 2, 0, 0);
			lx->pos += 2;
			break;
		}
		char c = src[lx->pos];
		if (c == '\'' || c == '"') {
			char   q = c;
			size_t s = ++lx->pos;
			while (lx->pos < lx->len && src[lx->pos] != q) {
				if (src[lx->pos] == '\\' && lx->pos + 1 < lx->len)
					lx->pos++;
				lx->pos++;
			}
			lex_push(lx, TOK_STRING, src + s, lx->pos - s, 0, 0);
			if (lx->pos < lx->len)
				lx->pos++;
		} else if (isdigit((unsigned char)c)) {
			size_t s = lx->pos;
			while (lx->pos < lx->len && isdigit((unsigned char)src[lx->pos]))
				lx->pos++;
			lex_push(lx, TOK_NUMBER, src + s, lx->pos - s, 0, 0);
		} else if (is_ident_start(c)) {
			size_t s = lx->pos;
			while (lx->pos < lx->len && is_ident_char(src[lx->pos]))
				lx->pos++;
			size_t n = lx->pos - s;
			lex_push(lx, lex_word_is_op(src + s, n) ? TOK_OP : TOK_IDENT, src + s, n, 0, 0);
		} else if (c == '.') {
			lex_push(lx, TOK_DOT, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else if (c == '[') {
			lex_push(lx, TOK_LBRACKET, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else if (c == ']') {
			lex_push(lx, TOK_RBRACKET, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else if (c == '(') {
			lex_push(lx, TOK_LPAREN, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else if (c == ')') {
			lex_push(lx, TOK_RPAREN, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else if (c == ',') {
			lex_push(lx, TOK_COMMA, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else if (c == ':') {
			lex_push(lx, TOK_COLON, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else if (c == '|') {
			lex_push(lx, TOK_PIPE, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else if (c == '=' && src[lx->pos + 1] == '=') {
			lex_push(lx, TOK_OP, src + lx->pos, 2, 0, 0);
			lx->pos += 2;
		} else if (c == '!' && src[lx->pos + 1] == '=') {
			lex_push(lx, TOK_OP, src + lx->pos, 2, 0, 0);
			lx->pos += 2;
		} else if ((c == '>' || c == '<') && src[lx->pos + 1] == '=') {
			lex_push(lx, TOK_OP, src + lx->pos, 2, 0, 0);
			lx->pos += 2;
		} else if (c == '>' || c == '<') {
			lex_push(lx, TOK_OP, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else if (c == '+' || c == '-' || c == '=' || c == '~') {
			lex_push(lx, TOK_OP, src + lx->pos, 1, 0, 0);
			lx->pos++;
		} else {
			char msg[128];
			snprintf(msg, sizeof(msg), "unsupported character '%c' at offset %zu", c, lx->pos);
			strlist_add(&lx->diagnostics, xstrdup(msg));
			lex_push(lx, TOK_UNKNOWN, src + lx->pos, 1, 0, 0);
			lx->pos++;
		}
	}
	return OK;
}

static status_code jinja_lex(const char *src, lexer *lx, char *errbuf, size_t errbuf_len) {
	lx->src = src;
	lx->len = strlen(src);
	if (lx->len > 0 && src[lx->len - 1] == '\n')
		lx->len--;
	lx->pos				= 0;
	lx->toks			= NULL;
	lx->n_toks			= 0;
	lx->cap_toks		= 0;
	lx->errbuf			= errbuf;
	lx->errbuf_len		= errbuf_len;
	lx->text_start		= 0;
	lx->next_strip_left = 0;
	lx->next_trim_nl	= 0;

	while (lx->pos < lx->len) {
		if (!strncmp(src + lx->pos, "{#", 2)) {
			status_code sc = lex_comment(lx, src);
			if (sc != OK)
				return sc;
			continue;
		}
		if (!strncmp(src + lx->pos, "{{", 2) || !strncmp(src + lx->pos, "{%", 2)) {
			int	   is_expr = src[lx->pos + 1] == '{';
			size_t start   = lx->pos;
			lx->pos += 2;
			int strip_left_of_text_before = 0;
			if (lx->pos < lx->len && src[lx->pos] == '-') {
				strip_left_of_text_before = 1;
				lx->pos++;
			}
			if (start > lx->text_start || lx->next_strip_left)
				lex_flush_text(lx, src, start, strip_left_of_text_before, !is_expr);
			lex_push(lx, is_expr ? TOK_EXPR_OPEN : TOK_STMT_OPEN, src + start, lx->pos - start, 0,
					 0);

			lx->tag_start		= start;
			lx->tag_is_expr		= is_expr;
			lx->tag_strip_after = 0;
			status_code sc		= lex_tag_inner(lx, src);
			if (sc != OK)
				return sc;

			lx->text_start		= lx->pos;
			lx->next_strip_left = lx->tag_strip_after;
			lx->next_trim_nl	= !is_expr && !lx->tag_strip_after;
		} else {
			lx->pos++;
		}
	}
	if (lx->len > lx->text_start || lx->next_strip_left)
		lex_flush_text(lx, src, lx->len, 0, 0);
	lex_push(lx, TOK_EOF, src + lx->len, 0, 0, 0);
	return OK;
}

typedef enum {
	EX_STRING,
	EX_IDENT,
	EX_ATTR,
	EX_INDEX,
	EX_SLICE,
	EX_BINOP,
	EX_NOT,
	EX_FILTER,
	EX_ISDEFINED,
	EX_CALL,
	EX_LIST,
	EX_TERNARY,
	EX_METHODCALL,
	EX_DYNINDEX,
} expr_kind;

typedef struct expr_arg {
	char			 *name;
	struct expr_node *val;
	struct expr_arg	 *next;
} expr_arg;

typedef struct expr_node {
	expr_kind		  kind;
	char			 *str;
	char			 *op;
	struct expr_node *a, *b, *c, *step;

	expr_arg *args;

	struct expr_node **items;
	size_t			   n_items;
} expr_node;

typedef enum {
	ST_TEXT,
	ST_OUTPUT,
	ST_IF,
	ST_FOR,
	ST_LIST,
	ST_SET,
	ST_MACRO,
	ST_GENERATION,
	ST_NOOP,
} stmt_kind;

typedef struct macro_param {
	char			   *name;
	struct expr_node   *default_val;
	struct macro_param *next;
} macro_param;

typedef struct stmt_node {
	stmt_kind kind;

	char *text;

	expr_node *expr;

	expr_node		 *cond;
	struct stmt_node *then_body;
	struct stmt_node *else_body;

	char			 *loop_var;
	char			 *loop_var2;
	expr_node		 *iterable;
	struct stmt_node *body;

	struct stmt_node **items;
	size_t			   n_items;

	char	  *set_name;
	char	  *set_attr;
	expr_node *set_val;

	char			 *macro_name;
	macro_param		 *macro_params;
	struct stmt_node *macro_body;

	struct stmt_node *next;
} stmt_node;

typedef struct {
	lexer  *lx;
	size_t	pos;
	char   *errbuf;
	size_t	errbuf_len;
	int		failed;
	strlist diagnostics;
	strlist features;
} parser;

static token *pcur(parser *p) {
	return &p->lx->toks[p->pos];
}
static token *padvance(parser *p) {
	token *t = pcur(p);
	if (t->type != TOK_EOF)
		p->pos++;
	return t;
}
static int ptok_is(parser *p, tok_type t) {
	return pcur(p)->type == t;
}
static int ptok_ident_is(parser *p, const char *s) {
	token *t = pcur(p);
	return t->type == TOK_IDENT && strlen(s) == t->len && !strncmp(t->start, s, t->len);
}
static int ptok_op_is(parser *p, const char *s) {
	token *t = pcur(p);
	return t->type == TOK_OP && strlen(s) == t->len && !strncmp(t->start, s, t->len);
}
static char *tok_dup(token *t) {
	char *s = xmalloc(t->len + 1);
	memcpy(s, t->start, t->len);
	s[t->len] = '\0';
	return s;
}

static char *str_unescape(const char *s, size_t len) {
	char  *out = xmalloc(len + 1);
	size_t oi  = 0;
	for (size_t i = 0; i < len; i++) {
		if (s[i] != '\\' || i + 1 >= len) {
			out[oi++] = s[i];
			continue;
		}
		char e = s[++i];
		switch (e) {
		case 'n':
			out[oi++] = '\n';
			break;
		case 't':
			out[oi++] = '\t';
			break;
		case 'r':
			out[oi++] = '\r';
			break;
		case 'b':
			out[oi++] = '\b';
			break;
		case 'f':
			out[oi++] = '\f';
			break;
		case 'v':
			out[oi++] = '\v';
			break;
		case 'a':
			out[oi++] = '\a';
			break;
		case '\\':
			out[oi++] = '\\';
			break;
		case '\'':
			out[oi++] = '\'';
			break;
		case '"':
			out[oi++] = '"';
			break;
		case '0':
			out[oi++] = '\0';
			break;
		default:
			out[oi++] = e;
			break;
		}
	}
	out[oi] = '\0';
	return out;
}

static void perr(parser *p, const char *msg) {
	char buf[256];
	snprintf(buf, sizeof(buf), "parse error: %s", msg);
	strlist_add(&p->diagnostics, xstrdup(buf));
	p->failed = 1;
}

static void expect_stmt_close(parser *p, const char *context) {
	if (!ptok_is(p, TOK_STMT_CLOSE)) {
		char buf[256];
		snprintf(buf, sizeof(buf), "expected '%%}' after %s", context);
		perr(p, buf);
	} else {
		padvance(p);
	}
}

static expr_node *parse_expr(parser *p);

static expr_arg *parse_call_args(parser *p) {
	expr_arg *head = NULL, *tail = NULL;
	padvance(p);
	while (!ptok_is(p, TOK_RPAREN) && !ptok_is(p, TOK_EOF)) {
		expr_arg *arg = xmalloc(sizeof(*arg));
		memset(arg, 0, sizeof(*arg));
		if (ptok_is(p, TOK_IDENT)) {
			size_t save = p->pos;
			char  *nm	= tok_dup(pcur(p));
			padvance(p);
			if (ptok_op_is(p, "=")) {
				padvance(p);
				arg->name = nm;
				arg->val  = parse_expr(p);
			} else {
				free(nm);
				p->pos	 = save;
				arg->val = parse_expr(p);
			}
		} else {
			arg->val = parse_expr(p);
		}
		if (!head)
			head = tail = arg;
		else {
			tail->next = arg;
			tail	   = arg;
		}
		if (ptok_is(p, TOK_COMMA))
			padvance(p);
		else
			break;
	}
	if (!ptok_is(p, TOK_RPAREN))
		perr(p, "expected ')'");
	else
		padvance(p);
	return head;
}

static expr_node *parse_primary(parser *p) {
	if (ptok_op_is(p, "-")) {
		padvance(p);
		expr_node *operand = parse_primary(p);
		expr_node *zero	   = xmalloc(sizeof(*zero));
		memset(zero, 0, sizeof(*zero));
		zero->kind	 = EX_STRING;
		zero->str	 = xstrdup("0");
		expr_node *n = xmalloc(sizeof(*n));
		memset(n, 0, sizeof(*n));
		n->kind = EX_BINOP;
		n->op	= xstrdup("-");
		n->a	= zero;
		n->b	= operand;
		return n;
	}
	expr_node *n = xmalloc(sizeof(*n));
	memset(n, 0, sizeof(*n));
	if (ptok_is(p, TOK_STRING)) {
		n->kind = EX_STRING;
		n->str	= str_unescape(pcur(p)->start, pcur(p)->len);
		padvance(p);
	} else if (ptok_is(p, TOK_NUMBER)) {
		n->kind = EX_STRING;
		n->str	= tok_dup(pcur(p));
		padvance(p);
	} else if (ptok_is(p, TOK_LBRACKET)) {
		padvance(p);
		n->kind = EX_LIST;
		while (!ptok_is(p, TOK_RBRACKET) && !ptok_is(p, TOK_EOF)) {
			expr_node *item		   = parse_expr(p);
			n->items			   = xrealloc(n->items, (n->n_items + 1) * sizeof(expr_node *));
			n->items[n->n_items++] = item;
			if (ptok_is(p, TOK_COMMA))
				padvance(p);
			else
				break;
		}
		if (!ptok_is(p, TOK_RBRACKET))
			perr(p, "expected ']'");
		else
			padvance(p);
	} else if (ptok_is(p, TOK_IDENT)) {
		n->str = tok_dup(pcur(p));
		padvance(p);
		if (ptok_is(p, TOK_LPAREN)) {
			n->kind = EX_CALL;
			n->args = parse_call_args(p);
		} else {
			n->kind = EX_IDENT;
		}
	} else if (ptok_is(p, TOK_LPAREN)) {
		padvance(p);
		free(n);
		n = parse_expr(p);
		if (ptok_is(p, TOK_COMMA)) {
			expr_node *tuple = xmalloc(sizeof(*tuple));
			memset(tuple, 0, sizeof(*tuple));
			tuple->kind					   = EX_LIST;
			tuple->items				   = xrealloc(NULL, sizeof(expr_node *));
			tuple->items[tuple->n_items++] = n;
			while (ptok_is(p, TOK_COMMA)) {
				padvance(p);
				if (ptok_is(p, TOK_RPAREN))
					break;
				expr_node *item = parse_expr(p);
				tuple->items = xrealloc(tuple->items, (tuple->n_items + 1) * sizeof(expr_node *));
				tuple->items[tuple->n_items++] = item;
			}
			n = tuple;
		}
		if (!ptok_is(p, TOK_RPAREN))
			perr(p, "expected ')'");
		else
			padvance(p);
	} else {
		perr(p, "expected expression");
		n->kind = EX_STRING;
		n->str	= xstrdup("");
	}

	for (;;) {
		if (ptok_is(p, TOK_DOT)) {
			padvance(p);
			if (!ptok_is(p, TOK_IDENT) && !ptok_is(p, TOK_NUMBER)) {
				perr(p, "expected identifier after '.'");
				break;
			}
			expr_node *attr = xmalloc(sizeof(*attr));
			memset(attr, 0, sizeof(*attr));
			attr->str = tok_dup(pcur(p));
			attr->a	  = n;
			padvance(p);
			if (ptok_is(p, TOK_LPAREN)) {
				attr->kind = EX_METHODCALL;
				attr->args = parse_call_args(p);
			} else {
				attr->kind = EX_ATTR;
			}
			n = attr;
		} else if (ptok_is(p, TOK_LBRACKET)) {
			padvance(p);
			expr_node *attr = xmalloc(sizeof(*attr));
			memset(attr, 0, sizeof(*attr));
			if (ptok_is(p, TOK_COLON)) {
				attr->kind = EX_SLICE;
				padvance(p);
				if (!ptok_is(p, TOK_RBRACKET) && !ptok_is(p, TOK_COLON))
					attr->c = parse_expr(p);
			} else if (ptok_is(p, TOK_STRING)) {
				attr->kind = EX_ATTR;
				attr->str  = str_unescape(pcur(p)->start, pcur(p)->len);
				padvance(p);
			} else if (ptok_is(p, TOK_NUMBER)) {
				expr_node *start = xmalloc(sizeof(*start));
				memset(start, 0, sizeof(*start));
				start->kind = EX_STRING;
				start->str	= tok_dup(pcur(p));
				padvance(p);
				if (ptok_is(p, TOK_COLON)) {
					attr->kind = EX_SLICE;
					attr->b	   = start;
					padvance(p);
					if (!ptok_is(p, TOK_RBRACKET) && !ptok_is(p, TOK_COLON))
						attr->c = parse_expr(p);
				} else {
					attr->kind = EX_INDEX;
					attr->str  = start->str;
					free(start);
				}
			} else {
				expr_node *start = parse_expr(p);
				if (ptok_is(p, TOK_COLON)) {
					attr->kind = EX_SLICE;
					attr->b	   = start;
					padvance(p);
					if (!ptok_is(p, TOK_RBRACKET) && !ptok_is(p, TOK_COLON))
						attr->c = parse_expr(p);
				} else {
					attr->kind = EX_DYNINDEX;
					attr->b	   = start;
				}
			}
			if (attr->kind == EX_SLICE && ptok_is(p, TOK_COLON)) {
				padvance(p);
				if (!ptok_is(p, TOK_RBRACKET))
					attr->step = parse_expr(p);
			}
			attr->a = n;
			if (!ptok_is(p, TOK_RBRACKET))
				perr(p, "expected ']'");
			else
				padvance(p);
			n = attr;
		} else {
			break;
		}
	}
	return n;
}

static expr_node *parse_filter(parser *p) {
	expr_node *n = parse_primary(p);
	while (ptok_is(p, TOK_PIPE)) {
		padvance(p);
		if (!ptok_is(p, TOK_IDENT)) {
			perr(p, "expected filter name");
			break;
		}
		expr_node *f = xmalloc(sizeof(*f));
		memset(f, 0, sizeof(*f));
		f->kind = EX_FILTER;
		f->str	= tok_dup(pcur(p));
		f->a	= n;
		padvance(p);
		if (ptok_is(p, TOK_LPAREN))
			f->args = parse_call_args(p);
		n = f;
	}
	return n;
}

static expr_node *parse_concat(parser *p) {
	expr_node *n = parse_filter(p);
	for (;;) {
		if (ptok_op_is(p, "+") || ptok_op_is(p, "~")) {
			padvance(p);
			expr_node *rhs = parse_filter(p);
			expr_node *b   = xmalloc(sizeof(*b));
			memset(b, 0, sizeof(*b));
			b->kind = EX_BINOP;
			b->op	= xstrdup("+");
			b->a	= n;
			b->b	= rhs;
			n		= b;
		} else if (ptok_op_is(p, "-")) {
			padvance(p);
			expr_node *rhs = parse_filter(p);
			expr_node *b   = xmalloc(sizeof(*b));
			memset(b, 0, sizeof(*b));
			b->kind = EX_BINOP;
			b->op	= xstrdup("-");
			b->a	= n;
			b->b	= rhs;
			n		= b;
		} else {
			break;
		}
	}
	return n;
}

static expr_node *parse_comparison(parser *p) {
	if (ptok_op_is(p, "not")) {
		padvance(p);
		expr_node *n = xmalloc(sizeof(*n));
		memset(n, 0, sizeof(*n));
		n->kind = EX_NOT;
		n->a	= parse_comparison(p);
		return n;
	}
	expr_node *n = parse_concat(p);
	for (;;) {
		const char *op = NULL;
		if (ptok_op_is(p, "=="))
			op = "==";
		else if (ptok_op_is(p, "!="))
			op = "!=";
		else if (ptok_op_is(p, "in"))
			op = "in";
		else if (ptok_op_is(p, "not")) {
			size_t save = p->pos;
			padvance(p);
			if (ptok_op_is(p, "in")) {
				padvance(p);
				expr_node *rhs	   = parse_concat(p);
				expr_node *in_node = xmalloc(sizeof(*in_node));
				memset(in_node, 0, sizeof(*in_node));
				in_node->kind	= EX_BINOP;
				in_node->op		= xstrdup("in");
				in_node->a		= n;
				in_node->b		= rhs;
				expr_node *notn = xmalloc(sizeof(*notn));
				memset(notn, 0, sizeof(*notn));
				notn->kind = EX_NOT;
				notn->a	   = in_node;
				n		   = notn;
				continue;
			}
			p->pos = save;
			break;
		} else if (ptok_op_is(p, ">"))
			op = ">";
		else if (ptok_op_is(p, "<"))
			op = "<";
		else if (ptok_op_is(p, ">="))
			op = ">=";
		else if (ptok_op_is(p, "<="))
			op = "<=";
		else if (ptok_op_is(p, "is")) {
			padvance(p);
			int negate = 0;
			if (ptok_op_is(p, "not")) {
				negate = 1;
				padvance(p);
			}
			char *test_name = NULL;
			if (ptok_is(p, TOK_IDENT)) {
				test_name = tok_dup(pcur(p));
				padvance(p);
			}
			expr_node *b = xmalloc(sizeof(*b));
			memset(b, 0, sizeof(*b));
			b->kind = EX_ISDEFINED;
			b->a	= n;
			b->b	= NULL;
			b->str	= test_name;
			if (negate) {
				expr_node *notn = xmalloc(sizeof(*notn));
				memset(notn, 0, sizeof(*notn));
				notn->kind = EX_NOT;
				notn->a	   = b;
				n		   = notn;
			} else {
				n = b;
			}
			continue;
		} else
			break;
		padvance(p);
		expr_node *rhs = parse_concat(p);
		expr_node *b   = xmalloc(sizeof(*b));
		memset(b, 0, sizeof(*b));
		b->kind = EX_BINOP;
		b->op	= xstrdup(op);
		b->a	= n;
		b->b	= rhs;
		n		= b;
	}
	return n;
}

static expr_node *parse_binop_left_assoc(parser		*p, expr_node *(*next)(parser *),
										 const char *op_token) {
	expr_node *n = next(p);
	while (ptok_op_is(p, op_token)) {
		padvance(p);
		expr_node *rhs = next(p);
		expr_node *b   = xmalloc(sizeof(*b));
		memset(b, 0, sizeof(*b));
		b->kind = EX_BINOP;
		b->op	= xstrdup(op_token);
		b->a	= n;
		b->b	= rhs;
		n		= b;
	}
	return n;
}

static expr_node *parse_and(parser *p) {
	return parse_binop_left_assoc(p, parse_comparison, "and");
}

static expr_node *parse_or(parser *p) {
	return parse_binop_left_assoc(p, parse_and, "or");
}

static expr_node *parse_expr(parser *p) {
	expr_node *n = parse_or(p);
	if (ptok_ident_is(p, "if")) {
		padvance(p);
		expr_node *cond		= parse_or(p);
		expr_node *else_val = NULL;
		if (ptok_ident_is(p, "else")) {
			padvance(p);
			else_val = parse_expr(p);
		}
		expr_node *t = xmalloc(sizeof(*t));
		memset(t, 0, sizeof(*t));
		t->kind = EX_TERNARY;
		t->a	= n;
		t->c	= cond;
		t->b	= else_val;
		n		= t;
	}
	return n;
}

static stmt_node *new_stmt(stmt_kind k) {
	stmt_node *s = xmalloc(sizeof(*s));
	memset(s, 0, sizeof(*s));
	s->kind = k;
	return s;
}

static stmt_node *parse_block(parser *p);

static int stmt_kw_is(parser *p, const char *kw) {
	return ptok_is(p, TOK_IDENT) && ptok_ident_is(p, kw);
}

static void parse_if_tail(parser *p, stmt_node *s) {
	if (stmt_kw_is(p, "elif")) {
		padvance(p);
		stmt_node *inner = new_stmt(ST_IF);
		inner->cond		 = parse_expr(p);
		expect_stmt_close(p, "elif condition");
		inner->then_body = parse_block(p);
		parse_if_tail(p, inner);
		s->else_body		   = new_stmt(ST_LIST);
		s->else_body->items	   = xmalloc(sizeof(stmt_node *));
		s->else_body->items[0] = inner;
		s->else_body->n_items  = 1;
	} else if (stmt_kw_is(p, "else")) {
		padvance(p);
		expect_stmt_close(p, "else");
		s->else_body = parse_block(p);
		if (!stmt_kw_is(p, "endif"))
			perr(p, "expected 'endif'");
		else
			padvance(p);
	} else if (stmt_kw_is(p, "endif")) {
		padvance(p);
	} else {
		perr(p, "expected 'endif'");
	}
}

static stmt_node *parse_if_stmt(parser *p) {
	padvance(p);
	stmt_node *s = new_stmt(ST_IF);
	s->cond		 = parse_expr(p);
	expect_stmt_close(p, "if condition");
	s->then_body = parse_block(p);
	parse_if_tail(p, s);
	expect_stmt_close(p, "endif");
	return s;
}

static stmt_node *parse_for_stmt(parser *p) {
	padvance(p);
	stmt_node *s = new_stmt(ST_FOR);
	if (!ptok_is(p, TOK_IDENT))
		perr(p, "expected loop variable");
	s->loop_var = tok_dup(pcur(p));
	padvance(p);
	if (ptok_is(p, TOK_COMMA)) {
		padvance(p);
		if (!ptok_is(p, TOK_IDENT))
			perr(p, "expected second loop variable");
		s->loop_var2 = tok_dup(pcur(p));
		padvance(p);
	}
	if (!ptok_op_is(p, "in"))
		perr(p, "expected 'in'");
	else
		padvance(p);
	s->iterable = parse_expr(p);
	expect_stmt_close(p, "for");
	s->body = parse_block(p);
	if (!stmt_kw_is(p, "endfor"))
		perr(p, "expected 'endfor'");
	else
		padvance(p);
	expect_stmt_close(p, "endfor");
	return s;
}

static stmt_node *parse_set_stmt(parser *p) {
	padvance(p);
	if (!ptok_is(p, TOK_IDENT)) {
		perr(p, "expected variable name after set");
		while (!ptok_is(p, TOK_STMT_CLOSE) && !ptok_is(p, TOK_EOF))
			padvance(p);
		if (ptok_is(p, TOK_STMT_CLOSE))
			padvance(p);
		return new_stmt(ST_NOOP);
	}
	stmt_node *s = new_stmt(ST_SET);
	s->set_name	 = tok_dup(pcur(p));
	padvance(p);
	if (ptok_is(p, TOK_DOT)) {
		padvance(p);
		if (!ptok_is(p, TOK_IDENT))
			perr(p, "expected attribute name after '.'");
		else {
			s->set_attr = tok_dup(pcur(p));
			padvance(p);
		}
	}
	if (ptok_op_is(p, "=")) {
		padvance(p);
		s->set_val = parse_expr(p);
		expect_stmt_close(p, "set");
	} else if (ptok_is(p, TOK_STMT_CLOSE)) {
		padvance(p);
		s->body = parse_block(p);
		if (!stmt_kw_is(p, "endset"))
			perr(p, "expected 'endset'");
		else
			padvance(p);
		expect_stmt_close(p, "endset");
	} else {
		perr(p, "expected '=' after set target");
	}
	return s;
}

static stmt_node *parse_macro_stmt(parser *p) {
	padvance(p);
	if (!ptok_is(p, TOK_IDENT)) {
		perr(p, "expected macro name");
	}
	stmt_node *s  = new_stmt(ST_MACRO);
	s->macro_name = tok_dup(pcur(p));
	padvance(p);
	if (ptok_is(p, TOK_LPAREN)) {
		padvance(p);
		macro_param *ptail = NULL;
		while (!ptok_is(p, TOK_RPAREN) && !ptok_is(p, TOK_EOF)) {
			if (!ptok_is(p, TOK_IDENT)) {
				perr(p, "expected parameter name");
				break;
			}
			macro_param *mp = xmalloc(sizeof(*mp));
			memset(mp, 0, sizeof(*mp));
			mp->name = tok_dup(pcur(p));
			padvance(p);
			if (ptok_op_is(p, "=")) {
				padvance(p);
				mp->default_val = parse_expr(p);
			}
			if (!s->macro_params)
				s->macro_params = mp;
			else
				ptail->next = mp;
			ptail = mp;
			if (ptok_is(p, TOK_COMMA))
				padvance(p);
			else
				break;
		}
		if (!ptok_is(p, TOK_RPAREN))
			perr(p, "expected ')' after macro params");
		else
			padvance(p);
	}
	expect_stmt_close(p, "macro signature");
	s->macro_body = parse_block(p);
	if (!stmt_kw_is(p, "endmacro"))
		perr(p, "expected 'endmacro'");
	else
		padvance(p);
	expect_stmt_close(p, "endmacro");
	return s;
}

static stmt_node *parse_statement_tag(parser *p, int *is_block_end, const char **end_kw) {
	*is_block_end = 0;
	*end_kw		  = NULL;

	if (stmt_kw_is(p, "if"))
		return parse_if_stmt(p);

	if (stmt_kw_is(p, "for"))
		return parse_for_stmt(p);

	if (stmt_kw_is(p, "endif") || stmt_kw_is(p, "endfor") || stmt_kw_is(p, "else") ||
		stmt_kw_is(p, "elif") || stmt_kw_is(p, "endmacro") || stmt_kw_is(p, "endset") ||
		stmt_kw_is(p, "endgeneration")) {
		*is_block_end = 1;
		*end_kw		  = "end";
		return NULL;
	}

	if (stmt_kw_is(p, "set"))
		return parse_set_stmt(p);

	if (stmt_kw_is(p, "generation")) {
		stmt_node *s = new_stmt(ST_GENERATION);
		padvance(p);
		expect_stmt_close(p, "generation tag");
		s->body = parse_block(p);
		if (!stmt_kw_is(p, "endgeneration"))
			perr(p, "expected 'endgeneration'");
		else
			padvance(p);
		expect_stmt_close(p, "endgeneration");
		return s;
	}

	if (stmt_kw_is(p, "macro"))
		return parse_macro_stmt(p);

	perr(p, "unsupported statement");
	while (!ptok_is(p, TOK_STMT_CLOSE) && !ptok_is(p, TOK_EOF))
		padvance(p);
	if (ptok_is(p, TOK_STMT_CLOSE))
		padvance(p);
	return new_stmt(ST_NOOP);
}

static stmt_node *parse_block(parser *p) {
	stmt_node *head = NULL, *tail = NULL;
	for (;;) {
		if (ptok_is(p, TOK_EOF))
			break;
		if (ptok_is(p, TOK_TEXT)) {
			token *t = pcur(p);
			padvance(p);
			size_t start = 0, len = t->len;
			if (t->strip_left)
				while (start < len && isspace((unsigned char)t->start[start]))
					start++;
			while (t->strip_right && len > start && isspace((unsigned char)t->start[len - 1]))
				len--;
			stmt_node *s = new_stmt(ST_TEXT);
			s->text		 = xmalloc(len - start + 1);
			memcpy(s->text, t->start + start, len - start);
			s->text[len - start] = '\0';
			if (!head)
				head = tail = s;
			else {
				tail->next = s;
				tail	   = s;
			}
			continue;
		}
		if (ptok_is(p, TOK_EXPR_OPEN)) {
			padvance(p);
			stmt_node *s = new_stmt(ST_OUTPUT);
			s->expr		 = parse_expr(p);
			if (!ptok_is(p, TOK_EXPR_CLOSE))
				perr(p, "expected '}}'");
			else
				padvance(p);
			if (!head)
				head = tail = s;
			else {
				tail->next = s;
				tail	   = s;
			}
			continue;
		}
		if (ptok_is(p, TOK_STMT_OPEN)) {
			padvance(p);
			int			is_end;
			const char *end_kw;
			stmt_node  *s = parse_statement_tag(p, &is_end, &end_kw);
			if (is_end)
				return head;
			if (!head)
				head = tail = s;
			else {
				tail->next = s;
				tail	   = s;
			}
			continue;
		}
		perr(p, "unexpected token");
		padvance(p);
	}
	return head;
}

static const char *supported_filters[] = {"trim",	  "default", "length", "upper",		 "lower",
										  "dictsort", "list",	 "map",	   "capitalize", "replace",
										  "tojson",	  "int",	 "first",  "last",		 "join",
										  "safe",	  "string",	 "items",  NULL};

static const char *supported_methods[] = {
	"get",	 "split",	   "strip",	  "lstrip", "rstrip",	  "trim",	  "join", "lower",
	"upper", "capitalize", "replace", "items",	"startswith", "endswith", NULL};

static const char *supported_tests[] = {"none",	   "null",	   "defined", "undefined", "string",
										"mapping", "iterable", "number",  "integer",   "sequence",
										"boolean", "true",	   "false",	  "dict",	   NULL};

static int list_has(const char **list, const char *s) {
	for (int i = 0; list[i]; i++)
		if (!strcmp(list[i], s))
			return 1;
	return 0;
}

static void scan_expr(parser *p, expr_node *e) {
	if (!e)
		return;
	switch (e->kind) {
	case EX_FILTER:
		if (e->str && !list_has(supported_filters, e->str)) {
			char buf[128];
			snprintf(buf, sizeof(buf), "unsupported filter '%s'", e->str);
			strlist_add(&p->features, xstrdup(buf));
		}
		scan_expr(p, e->a);
		for (expr_arg *a = e->args; a; a = a->next)
			scan_expr(p, a->val);
		break;
	case EX_METHODCALL:
		if (e->str && !list_has(supported_methods, e->str)) {
			char buf[128];
			snprintf(buf, sizeof(buf), "unsupported method '%s()'", e->str);
			strlist_add(&p->features, xstrdup(buf));
		}
		scan_expr(p, e->a);
		for (expr_arg *a = e->args; a; a = a->next)
			scan_expr(p, a->val);
		break;
	case EX_ISDEFINED:
		if (e->str && !list_has(supported_tests, e->str)) {
			char buf[128];
			snprintf(buf, sizeof(buf), "unsupported test 'is %s'", e->str);
			strlist_add(&p->features, xstrdup(buf));
		}
		scan_expr(p, e->a);
		break;
	case EX_BINOP: {
		static const char *supported_ops[] = {
			"and", "or", "+", "-", "==", "!=", ">", "<", ">=", "<=", "in", NULL};
		if (e->op && !list_has(supported_ops, e->op)) {
			char buf[128];
			snprintf(buf, sizeof(buf), "unsupported operator '%s'", e->op);
			strlist_add(&p->features, xstrdup(buf));
		}
		scan_expr(p, e->a);
		scan_expr(p, e->b);
		break;
	}
	case EX_NOT:
	case EX_TERNARY:
	case EX_ATTR:
	case EX_INDEX:
	case EX_DYNINDEX:
	case EX_CALL:
		scan_expr(p, e->a);
		scan_expr(p, e->b);
		scan_expr(p, e->c);
		for (expr_arg *a = e->args; a; a = a->next)
			scan_expr(p, a->val);
		break;
	case EX_SLICE:
		scan_expr(p, e->a);
		scan_expr(p, e->b);
		scan_expr(p, e->c);
		scan_expr(p, e->step);
		break;
	case EX_LIST:
		for (size_t i = 0; i < e->n_items; i++)
			scan_expr(p, e->items[i]);
		break;
	default:
		break;
	}
}

static void scan_stmt(parser *p, stmt_node *s) {
	for (; s; s = s->next) {
		scan_expr(p, s->expr);
		scan_expr(p, s->cond);
		scan_stmt(p, s->then_body);
		scan_stmt(p, s->else_body);
		scan_expr(p, s->iterable);
		scan_stmt(p, s->body);
		scan_expr(p, s->set_val);
		scan_stmt(p, s->macro_body);
		for (size_t i = 0; i < s->n_items; i++)
			scan_stmt(p, s->items[i]);
	}
}

jinja_program *jinja_compile(const char *template_src, char *errbuf, size_t errbuf_len) {
	lexer	lx		= {0};
	strlist lx_diag = {0};
	if (jinja_lex(template_src, &lx, errbuf, errbuf_len) != OK) {
		strlist_free(&lx_diag);
		free(lx.toks);
		return NULL;
	}

	parser p	 = {0};
	p.lx		 = &lx;
	p.pos		 = 0;
	p.errbuf	 = errbuf;
	p.errbuf_len = errbuf_len;
	p.failed	 = 0;

	for (size_t i = 0; i < lx.diagnostics.n; i++)
		strlist_add(&p.diagnostics, xstrdup(lx.diagnostics.msgs[i]));
	strlist_free(&lx.diagnostics);

	if (p.diagnostics.n)
		p.failed = 1;

	jinja_program *prog = parse_block(&p);
	scan_stmt(&p, prog);

	strlist combined = {0};
	for (size_t i = 0; i < p.diagnostics.n; i++)
		strlist_add(&combined, xstrdup(p.diagnostics.msgs[i]));
	for (size_t i = 0; i < p.features.n; i++)
		strlist_add(&combined, xstrdup(p.features.msgs[i]));

	if (combined.n) {
		strlist_report(&combined, errbuf, errbuf_len);
		strlist_free(&combined);
		strlist_free(&p.diagnostics);
		strlist_free(&p.features);
		free(lx.toks);
		jinja_program_free(prog);
		return NULL;
	}

	strlist_free(&p.diagnostics);
	strlist_free(&p.features);
	free(lx.toks);
	return prog;
}

static void expr_free(expr_node *e) {
	if (!e)
		return;
	free(e->str);
	free(e->op);
	expr_free(e->a);
	expr_free(e->b);
	expr_free(e->c);
	expr_free(e->step);
	for (expr_arg *arg = e->args; arg;) {
		expr_arg *next = arg->next;
		free(arg->name);
		expr_free(arg->val);
		free(arg);
		arg = next;
	}
	for (size_t i = 0; i < e->n_items; i++)
		expr_free(e->items[i]);
	free(e->items);
	free(e);
}

static void macro_params_free(macro_param *mp) {
	while (mp) {
		macro_param *next = mp->next;
		free(mp->name);
		expr_free(mp->default_val);
		free(mp);
		mp = next;
	}
}

static void stmt_free(stmt_node *s) {
	while (s) {
		stmt_node *next = s->next;
		free(s->text);
		expr_free(s->expr);
		expr_free(s->cond);
		stmt_free(s->then_body);
		stmt_free(s->else_body);
		free(s->loop_var);
		free(s->loop_var2);
		expr_free(s->iterable);
		stmt_free(s->body);
		for (size_t i = 0; i < s->n_items; i++)
			stmt_free(s->items[i]);
		free(s->items);
		free(s->set_name);
		free(s->set_attr);
		expr_free(s->set_val);
		free(s->macro_name);
		macro_params_free(s->macro_params);
		stmt_free(s->macro_body);
		free(s);
		s = next;
	}
}

void jinja_program_free(jinja_program *prog) {
	if (!prog)
		return;
	stmt_free(prog);
}

typedef struct scope_entry {
	char			   *name;
	jinja_value		   *val;
	struct scope_entry *next;
} scope_entry;

typedef struct scope {
	scope_entry	 *entries;
	struct scope *parent;
} scope;

static jinja_value *scope_lookup(scope *sc, const char *name) {
	for (scope *s = sc; s; s = s->parent)
		for (scope_entry *e = s->entries; e; e = e->next)
			if (!strcmp(e->name, name))
				return e->val;
	return NULL;
}

static void scope_assign(scope *sc, const char *name, jinja_value *val) {
	for (scope_entry *e = sc->entries; e; e = e->next) {
		if (!strcmp(e->name, name)) {
			e->val = val;
			return;
		}
	}
	scope_entry *e = xmalloc(sizeof(*e));
	e->name		   = xstrdup(name);
	e->val		   = val;
	e->next		   = sc->entries;
	sc->entries	   = e;
}

static void scope_free_entries(scope *sc) {
	scope_entry *e = sc->entries;
	while (e) {
		scope_entry *next = e->next;
		free(e->name);
		free(e);
		e = next;
	}
}

typedef struct {
	scope *sc;
	char  *errbuf;
	size_t errbuf_len;
	int	   failed;
	int	   call_depth;
} eval_ctx;

static void everr(eval_ctx *ctx, const char *msg) {
	if (!ctx->failed)
		snprintf(ctx->errbuf, ctx->errbuf_len, "render error: %s", msg);
	ctx->failed = 1;
}

static void			exec_stmts(eval_ctx *ctx, stmt_node *s, strbuf *out);
static jinja_value *eval_expr(eval_ctx *ctx, expr_node *e);

static void bind_one_param(scope *call_sc, macro_param *mp, expr_arg *args, expr_arg **cur_pos,
						   eval_ctx *ctx) {
	jinja_value *bound = NULL;
	int			 found = 0;
	for (expr_arg *a = args; a; a = a->next) {
		if (a->name && !strcmp(a->name, mp->name)) {
			bound = eval_expr(ctx, a->val);
			found = 1;
			break;
		}
	}
	if (!found && *cur_pos && !(*cur_pos)->name) {
		bound = eval_expr(ctx, (*cur_pos)->val);
		found = 1;
		do {
			*cur_pos = (*cur_pos)->next;
		} while (*cur_pos && (*cur_pos)->name);
	}
	if (!found && mp->default_val)
		bound = eval_expr(ctx, mp->default_val);
	if (!found && !mp->default_val)
		bound = jinja_none();
	scope_assign(call_sc, mp->name, bound);
}

static jinja_value *call_macro(eval_ctx *ctx, jinja_value *macro_val, expr_arg *args) {
	stmt_node *def = macro_val->as.macro;
	if (ctx->call_depth > 200) {
		everr(ctx, "macro recursion too deep");
		return jinja_string("");
	}
	scope call_sc;
	call_sc.entries = NULL;
	call_sc.parent	= ctx->sc;

	macro_param *mp		 = def->macro_params;
	expr_arg	*cur_pos = args;
	while (cur_pos && cur_pos->name)
		cur_pos = cur_pos->next;

	for (; mp; mp = mp->next)
		bind_one_param(&call_sc, mp, args, &cur_pos, ctx);

	scope *saved = ctx->sc;
	ctx->sc		 = &call_sc;
	ctx->call_depth++;
	strbuf sb;
	sb_init(&sb);
	exec_stmts(ctx, def->macro_body, &sb);
	ctx->call_depth--;
	ctx->sc = saved;
	scope_free_entries(&call_sc);

	jinja_value *out = jinja_string(sb.p);
	free(sb.p);
	return out;
}

static jinja_value *dictsort_value(const jinja_value *base) {
	jinja_value *out = jinja_list();
	if (!base || base->type != JV_DICT)
		return out;
	size_t n = 0;
	for (jinja_dict_entry *e = base->as.dict; e; e = e->next)
		n++;
	jinja_dict_entry **arr = xmalloc(n * sizeof(*arr));
	size_t			   i   = 0;
	for (jinja_dict_entry *e = base->as.dict; e; e = e->next)
		arr[i++] = e;
	for (size_t a = 0; a < n; a++)
		for (size_t b = a + 1; b < n; b++)
			if (strcmp(arr[a]->key, arr[b]->key) > 0) {
				jinja_dict_entry *tmp = arr[a];
				arr[a]				  = arr[b];
				arr[b]				  = tmp;
			}
	for (i = 0; i < n; i++) {
		jinja_value *pair = jinja_list();
		jinja_list_append(pair, jinja_string(arr[i]->key));
		jinja_list_append(pair, arr[i]->val);
		jinja_list_append(out, pair);
	}
	free(arr);
	return out;
}

static void json_append_str(strbuf *sb, const char *s) {
	sb_append(sb, "\"", 1);
	for (const char *c = s; *c; c++) {
		switch (*c) {
		case '"':
			sb_append(sb, "\\\"", 2);
			break;
		case '\\':
			sb_append(sb, "\\\\", 2);
			break;
		case '\n':
			sb_append(sb, "\\n", 2);
			break;
		case '\r':
			sb_append(sb, "\\r", 2);
			break;
		case '\t':
			sb_append(sb, "\\t", 2);
			break;
		default:
			sb_append(sb, c, 1);
			break;
		}
	}
	sb_append(sb, "\"", 1);
}

static void json_to_json(strbuf *sb, const jinja_value *v) {
	if (!v) {
		sb_append_str(sb, "null");
		return;
	}
	switch (v->type) {
	case JV_NONE:
		sb_append_str(sb, "null");
		break;
	case JV_BOOL:
		sb_append_str(sb, v->as.b ? "true" : "false");
		break;
	case JV_STRING:
		json_append_str(sb, v->as.s);
		break;
	case JV_DICT: {
		sb_append(sb, "{", 1);
		int first = 1;
		for (jinja_dict_entry *e = v->as.dict; e; e = e->next) {
			if (!first)
				sb_append_str(sb, ", ");
			first = 0;
			json_append_str(sb, e->key);
			sb_append_str(sb, ": ");
			json_to_json(sb, e->val);
		}
		sb_append(sb, "}", 1);
		break;
	}
	case JV_LIST: {
		sb_append(sb, "[", 1);
		for (size_t i = 0; i < v->as.list.n; i++) {
			if (i)
				sb_append_str(sb, ", ");
			json_to_json(sb, v->as.list.items[i]);
		}
		sb_append(sb, "]", 1);
		break;
	}
	case JV_MACRO:
		json_append_str(sb, value_as_cstr(v));
		break;
	}
}

static int is_int_str(const char *s) {
	if (!s || !*s)
		return 0;
	const char *c = s;
	if (*c == '-')
		c++;
	if (!*c)
		return 0;
	for (; *c; c++)
		if (!isdigit((unsigned char)*c))
			return 0;
	return 1;
}

static jinja_value *eval_expr(eval_ctx *ctx, expr_node *e);
static jinja_value *eval_filter(eval_ctx *ctx, expr_node *e) {
	jinja_value *base = eval_expr(ctx, e->a);
	if (!strcmp(e->str, "trim")) {
		const char *s = value_as_cstr(base);
		while (*s && isspace((unsigned char)*s))
			s++;
		char  *dup = xstrdup(s);
		size_t n   = strlen(dup);
		while (n > 0 && isspace((unsigned char)dup[n - 1]))
			dup[--n] = '\0';
		jinja_value *out = jinja_string(dup);
		free(dup);
		return out;
	}
	if (!strcmp(e->str, "default")) {
		if (truthy(base))
			return base;
		return e->args ? eval_expr(ctx, e->args->val) : jinja_string("");
	}
	if (!strcmp(e->str, "length")) {
		size_t n = 0;
		if (base && base->type == JV_LIST)
			n = base->as.list.n;
		else if (base && base->type == JV_STRING)
			n = strlen(base->as.s);
		char buf[32];
		snprintf(buf, sizeof(buf), "%zu", n);
		return jinja_string(buf);
	}
	if (!strcmp(e->str, "upper")) {
		char *dup = xstrdup(value_as_cstr(base));
		for (char *c = dup; *c; c++)
			*c = toupper((unsigned char)*c);
		jinja_value *out = jinja_string(dup);
		free(dup);
		return out;
	}
	if (!strcmp(e->str, "capitalize")) {
		char *dup = xstrdup(value_as_cstr(base));
		if (dup[0])
			dup[0] = toupper((unsigned char)dup[0]);
		for (char *c = dup + 1; *c; c++)
			*c = tolower((unsigned char)*c);
		jinja_value *out = jinja_string(dup);
		free(dup);
		return out;
	}
	if (!strcmp(e->str, "replace")) {
		const char *old_s = e->args ? value_as_cstr(eval_expr(ctx, e->args->val)) : "";
		const char *new_s =
			e->args && e->args->next ? value_as_cstr(eval_expr(ctx, e->args->next->val)) : "";
		const char *str	  = value_as_cstr(base);
		size_t		old_n = strlen(old_s);
		strbuf		sb;
		sb_init(&sb);
		if (!old_n) {
			sb_append_str(&sb, str);
		} else {
			const char *remaining = str;
			for (;;) {
				const char *hit = strstr(remaining, old_s);
				if (!hit) {
					sb_append_str(&sb, remaining);
					break;
				}
				sb_append(&sb, remaining, (size_t)(hit - remaining));
				sb_append_str(&sb, new_s);
				remaining = hit + old_n;
			}
		}
		jinja_value *out = jinja_string(sb.p);
		free(sb.p);
		return out;
	}
	if (!strcmp(e->str, "lower")) {
		char *dup = xstrdup(value_as_cstr(base));
		for (char *c = dup; *c; c++)
			*c = tolower((unsigned char)*c);
		jinja_value *out = jinja_string(dup);
		free(dup);
		return out;
	}
	if (!strcmp(e->str, "dictsort"))
		return dictsort_value(base);
	if (!strcmp(e->str, "int")) {
		long v = atol(value_as_cstr(base));
		char buf[32];
		snprintf(buf, sizeof(buf), "%ld", v);
		return jinja_string(buf);
	}
	if (!strcmp(e->str, "first")) {
		if (base && base->type == JV_LIST && base->as.list.n > 0)
			return base->as.list.items[0];
		return jinja_none();
	}
	if (!strcmp(e->str, "last")) {
		if (base && base->type == JV_LIST && base->as.list.n > 0)
			return base->as.list.items[base->as.list.n - 1];
		return jinja_none();
	}
	if (!strcmp(e->str, "join")) {
		const char *sep = e->args ? value_as_cstr(eval_expr(ctx, e->args->val)) : "";
		strbuf		sb;
		sb_init(&sb);
		if (base && base->type == JV_LIST) {
			for (size_t i = 0; i < base->as.list.n; i++) {
				if (i)
					sb_append_str(&sb, sep);
				sb_append_str(&sb, value_as_cstr(base->as.list.items[i]));
			}
		}
		jinja_value *out = jinja_string(sb.p);
		free(sb.p);
		return out;
	}
	if (!strcmp(e->str, "tojson")) {
		strbuf sb;
		sb_init(&sb);
		json_to_json(&sb, base);
		jinja_value *out = jinja_string(sb.p);
		free(sb.p);
		return out;
	}
	if (!strcmp(e->str, "list"))
		return base;
	if (!strcmp(e->str, "safe"))
		return base;
	if (!strcmp(e->str, "string"))
		return jinja_string(value_as_cstr(base));
	if (!strcmp(e->str, "items")) {
		jinja_value *out = jinja_list();
		if (base && base->type == JV_DICT) {
			for (jinja_dict_entry *en = base->as.dict; en; en = en->next) {
				jinja_value *pair = jinja_list();
				jinja_list_append(pair, jinja_string(en->key));
				jinja_list_append(pair, en->val);
				jinja_list_append(out, pair);
			}
		}
		return out;
	}
	if (!strcmp(e->str, "map")) {
		const char	*fname = e->args ? value_as_cstr(eval_expr(ctx, e->args->val)) : "";
		jinja_value *out   = jinja_list();
		if (base && base->type == JV_LIST) {
			for (size_t i = 0; i < base->as.list.n; i++) {
				jinja_value *item = base->as.list.items[i];
				if (!strcmp(fname, "upper")) {
					char *dup = xstrdup(value_as_cstr(item));
					for (char *c = dup; *c; c++)
						*c = toupper((unsigned char)*c);
					jinja_list_append(out, jinja_string(dup));
					free(dup);
				} else if (!strcmp(fname, "lower")) {
					char *dup = xstrdup(value_as_cstr(item));
					for (char *c = dup; *c; c++)
						*c = tolower((unsigned char)*c);
					jinja_list_append(out, jinja_string(dup));
					free(dup);
				} else {
					jinja_list_append(out, item);
				}
			}
		}
		return out;
	}
	return base;
}

static jinja_value *eval_binop(eval_ctx *ctx, expr_node *e) {
	if (!strcmp(e->op, "and"))
		return jinja_bool(truthy(eval_expr(ctx, e->a)) && truthy(eval_expr(ctx, e->b)));
	if (!strcmp(e->op, "or"))
		return jinja_bool(truthy(eval_expr(ctx, e->a)) || truthy(eval_expr(ctx, e->b)));
	if (!strcmp(e->op, "+")) {
		jinja_value *a		  = eval_expr(ctx, e->a);
		jinja_value *b		  = eval_expr(ctx, e->b);
		const char	*as		  = value_as_cstr(a);
		const char	*bs		  = value_as_cstr(b);
		int			 is_int_a = is_int_str(as);
		int			 is_int_b = is_int_str(bs);
		if (is_int_a && is_int_b) {
			char buf[32];
			snprintf(buf, sizeof(buf), "%ld", atol(as) + atol(bs));
			return jinja_string(buf);
		}
		strbuf sb;
		sb_init(&sb);
		sb_append_str(&sb, as);
		sb_append_str(&sb, bs);
		jinja_value *out = jinja_string(sb.p);
		free(sb.p);
		return out;
	}
	if (!strcmp(e->op, "-")) {
		jinja_value *a	= eval_expr(ctx, e->a);
		jinja_value *b	= eval_expr(ctx, e->b);
		long		 la = atol(value_as_cstr(a));
		long		 lb = atol(value_as_cstr(b));
		char		 buf[32];
		snprintf(buf, sizeof(buf), "%ld", la - lb);
		return jinja_string(buf);
	}
	if (!strcmp(e->op, "==") || !strcmp(e->op, "!=")) {
		jinja_value *a	= eval_expr(ctx, e->a);
		jinja_value *b	= eval_expr(ctx, e->b);
		int			 eq = !strcmp(value_as_cstr(a), value_as_cstr(b));
		return jinja_bool(!strcmp(e->op, "==") ? eq : !eq);
	}
	if (!strcmp(e->op, ">") || !strcmp(e->op, "<") || !strcmp(e->op, ">=") ||
		!strcmp(e->op, "<=")) {
		jinja_value *a	= eval_expr(ctx, e->a);
		jinja_value *b	= eval_expr(ctx, e->b);
		long		 la = atol(value_as_cstr(a));
		long		 lb = atol(value_as_cstr(b));
		int			 r;
		if (!strcmp(e->op, ">"))
			r = la > lb;
		else if (!strcmp(e->op, "<"))
			r = la < lb;
		else if (!strcmp(e->op, ">="))
			r = la >= lb;
		else
			r = la <= lb;
		return jinja_bool(r);
	}
	if (!strcmp(e->op, "in")) {
		jinja_value *needle = eval_expr(ctx, e->a);
		jinja_value *hay	= eval_expr(ctx, e->b);
		if (hay && hay->type == JV_DICT) {
			const char *needle_s = value_as_cstr(needle);
			for (jinja_dict_entry *en = hay->as.dict; en; en = en->next)
				if (!strcmp(en->key, needle_s))
					return jinja_bool(1);
			return jinja_bool(0);
		}
		if (hay && hay->type == JV_LIST) {
			for (size_t i = 0; i < hay->as.list.n; i++)
				if (!strcmp(value_as_cstr(hay->as.list.items[i]), value_as_cstr(needle)))
					return jinja_bool(1);
			return jinja_bool(0);
		}
		return jinja_bool(strstr(value_as_cstr(hay), value_as_cstr(needle)) != NULL);
	}
	everr(ctx, "unsupported operator");
	return jinja_none();
}

static jinja_value *eval_expr(eval_ctx *ctx, expr_node *e) {
	if (ctx->failed || !e)
		return jinja_none();
	switch (e->kind) {
	case EX_STRING:
		return jinja_string(e->str);
	case EX_IDENT: {
		static const struct {
			const char *name;
			int			kind;
		} kw[] = {{"true", 1}, {"True", 1}, {"false", 0}, {"False", 0},
				  {"none", 2}, {"None", 2}, {NULL, 0}};
		for (int i = 0; kw[i].name; i++) {
			if (!strcmp(e->str, kw[i].name)) {
				if (kw[i].kind == 1)
					return jinja_bool(1);
				if (kw[i].kind == 0)
					return jinja_bool(0);
				return jinja_none();
			}
		}
		jinja_value *v = scope_lookup(ctx->sc, e->str);
		return v ? v : jinja_none();
	}
	case EX_LIST: {
		jinja_value *out = jinja_list();
		for (size_t i = 0; i < e->n_items; i++)
			jinja_list_append(out, eval_expr(ctx, e->items[i]));
		return out;
	}
	case EX_TERNARY: {
		if (truthy(eval_expr(ctx, e->c)))
			return eval_expr(ctx, e->a);
		return e->b ? eval_expr(ctx, e->b) : jinja_none();
	}
	case EX_ATTR: {
		jinja_value *base = eval_expr(ctx, e->a);
		if (base && base->type == JV_LIST && e->str[0] &&
			e->str[strspn(e->str, "0123456789")] == '\0') {
			long idx = atol(e->str);
			long n	 = (long)base->as.list.n;
			if (idx < 0)
				idx += n;
			if (idx < 0 || idx >= n)
				return jinja_none();
			return base->as.list.items[idx];
		}
		jinja_value *v = dict_get(base, e->str);
		return v ? v : jinja_none();
	}
	case EX_INDEX: {
		jinja_value *base = eval_expr(ctx, e->a);
		long		 idx  = atol(e->str);
		if (!base || base->type != JV_LIST || idx < 0 || (size_t)idx >= base->as.list.n)
			return jinja_none();
		return base->as.list.items[idx];
	}
	case EX_SLICE: {
		jinja_value *base = eval_expr(ctx, e->a);
		if (!base)
			return jinja_none();
		long n;
		if (base->type == JV_STRING)
			n = (long)strlen(base->as.s);
		else if (base->type == JV_LIST)
			n = (long)base->as.list.n;
		else
			return jinja_none();

		long step = 1;
		if (e->step)
			step = atol(value_as_cstr(eval_expr(ctx, e->step)));
		if (step == 0)
			step = 1;

		long start = step > 0 ? 0 : n - 1;
		long end   = step > 0 ? n : -1;
		if (e->b) {
			start = atol(value_as_cstr(eval_expr(ctx, e->b)));
			if (start < 0)
				start += n;
			start = start < 0 ? (step > 0 ? 0 : -1) : (start > n ? n : start);
			if (step < 0 && start >= n)
				start = n - 1;
		}
		if (e->c) {
			end = atol(value_as_cstr(eval_expr(ctx, e->c)));
			if (end < 0)
				end += n;
			if (step > 0)
				end = end < 0 ? 0 : (end > n ? n : end);
			else
				end = end < -1 ? -1 : (end >= n ? n - 1 : end);
		}

		jinja_value *out  = base->type == JV_STRING ? NULL : jinja_list();
		size_t		 cap  = base->type == JV_STRING ? (size_t)(n > 0 ? n : 0) + 1 : 0;
		char		*sbuf = base->type == JV_STRING ? xmalloc(cap) : NULL;
		size_t		 slen = 0;
		if (step > 0) {
			for (long i = start; i < end; i += step) {
				if (base->type == JV_STRING)
					sbuf[slen++] = base->as.s[i];
				else
					jinja_list_append(out, base->as.list.items[i]);
			}
		} else {
			for (long i = start; i > end; i += step) {
				if (base->type == JV_STRING)
					sbuf[slen++] = base->as.s[i];
				else
					jinja_list_append(out, base->as.list.items[i]);
			}
		}
		if (base->type == JV_STRING) {
			sbuf[slen]		 = '\0';
			jinja_value *res = jinja_string_n(sbuf, slen);
			free(sbuf);
			return res;
		}
		return out;
	}
	case EX_DYNINDEX: {
		jinja_value *base = eval_expr(ctx, e->a);
		jinja_value *key  = eval_expr(ctx, e->b);
		if (base && base->type == JV_STRING) {
			long idx = atol(value_as_cstr(key));
			long n	 = (long)strlen(base->as.s);
			if (idx < 0)
				idx += n;
			if (idx < 0 || idx >= n)
				return jinja_none();
			return jinja_string_n(base->as.s + idx, 1);
		}
		if (base && base->type == JV_LIST) {
			long idx = atol(value_as_cstr(key));
			long n	 = (long)base->as.list.n;
			if (idx < 0)
				idx += n;
			if (idx < 0 || idx >= n)
				return jinja_none();
			return base->as.list.items[idx];
		}
		if (base && base->type == JV_DICT) {
			jinja_value *v = dict_get(base, value_as_cstr(key));
			return v ? v : jinja_none();
		}
		return jinja_none();
	}
	case EX_NOT:
		return jinja_bool(!truthy(eval_expr(ctx, e->a)));
	case EX_CALL: {
		if (!strcmp(e->str, "raise_exception")) {
			everr(ctx, "template raised an exception");
			return jinja_none();
		}
		if (!strcmp(e->str, "namespace")) {
			jinja_value *out = jinja_dict();
			for (expr_arg *a = e->args; a; a = a->next)
				if (a->name)
					jinja_dict_set(out, a->name, eval_expr(ctx, a->val));
			return out;
		}
		if (!strcmp(e->str, "range")) {
			long	  lo = 0, hi = 0;
			expr_arg *a0 = e->args;
			expr_arg *a1 = a0 ? a0->next : NULL;
			if (a0 && a1) {
				lo = atol(value_as_cstr(eval_expr(ctx, a0->val)));
				hi = atol(value_as_cstr(eval_expr(ctx, a1->val)));
			} else if (a0) {
				hi = atol(value_as_cstr(eval_expr(ctx, a0->val)));
			}
			long	  step = 1;
			expr_arg *a2   = a1 ? a1->next : NULL;
			if (a2)
				step = atol(value_as_cstr(eval_expr(ctx, a2->val)));
			jinja_value *out = jinja_list();
			if (step > 0)
				for (long i = lo; i < hi; i += step) {
					char buf[32];
					snprintf(buf, sizeof(buf), "%ld", i);
					jinja_list_append(out, jinja_string(buf));
				}
			else if (step < 0)
				for (long i = lo; i > hi; i += step) {
					char buf[32];
					snprintf(buf, sizeof(buf), "%ld", i);
					jinja_list_append(out, jinja_string(buf));
				}
			return out;
		}
		if (!strcmp(e->str, "strftime_now")) {
			const char *fmt = e->args ? value_as_cstr(eval_expr(ctx, e->args->val)) : "%d %b %Y";
			time_t		now = time(NULL) + g_jinja_time_shift;
			struct tm	tm;
			localtime_r(&now, &tm);
			char buf[128];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
			strftime(buf, sizeof(buf), fmt, &tm);
#pragma GCC diagnostic pop
			return jinja_string(buf);
		}
		jinja_value *macro = scope_lookup(ctx->sc, e->str);
		if (macro && macro->type == JV_MACRO)
			return call_macro(ctx, macro, e->args);
		return jinja_none();
	}
	case EX_METHODCALL: {
		jinja_value *base = eval_expr(ctx, e->a);
		if (!strcmp(e->str, "get")) {
			const char	*key = e->args ? value_as_cstr(eval_expr(ctx, e->args->val)) : "";
			jinja_value *v	 = dict_get(base, key);
			if (v)
				return v;
			if (e->args && e->args->next)
				return eval_expr(ctx, e->args->next->val);
			return jinja_none();
		}
		if (!strcmp(e->str, "split")) {
			const char	*sep = e->args ? value_as_cstr(eval_expr(ctx, e->args->val)) : NULL;
			const char	*s	 = value_as_cstr(base);
			jinja_value *out = jinja_list();
			if (!sep || !*sep) {
				jinja_list_append(out, jinja_string(s));
				return out;
			}
			size_t		seplen = strlen(sep);
			const char *cursor = s;
			for (;;) {
				const char *hit = strstr(cursor, sep);
				if (!hit) {
					jinja_list_append(out, jinja_string(cursor));
					break;
				}
				size_t partlen = (size_t)(hit - cursor);
				jinja_list_append(out, jinja_string_n(cursor, partlen));
				cursor = hit + seplen;
			}
			return out;
		}
		if (!strcmp(e->str, "strip") || !strcmp(e->str, "lstrip") || !strcmp(e->str, "rstrip") ||
			!strcmp(e->str, "trim")) {
			const char *s = value_as_cstr(base);
			if (strcmp(e->str, "rstrip"))
				while (*s && isspace((unsigned char)*s))
					s++;
			char  *dup = xstrdup(s);
			size_t n   = strlen(dup);
			if (strcmp(e->str, "lstrip"))
				while (n > 0 && isspace((unsigned char)dup[n - 1]))
					dup[--n] = '\0';
			jinja_value *out = jinja_string(dup);
			free(dup);
			return out;
		}
		if (!strcmp(e->str, "items")) {
			jinja_value *out = jinja_list();
			if (base && base->type == JV_DICT) {
				for (jinja_dict_entry *en = base->as.dict; en; en = en->next) {
					jinja_value *pair = jinja_list();
					jinja_list_append(pair, jinja_string(en->key));
					jinja_list_append(pair, en->val);
					jinja_list_append(out, pair);
				}
			}
			return out;
		}
		if (!strcmp(e->str, "join")) {
			const char *sep = e->args ? value_as_cstr(eval_expr(ctx, e->args->val)) : "";
			strbuf		sb;
			sb_init(&sb);
			if (base && base->type == JV_LIST) {
				for (size_t i = 0; i < base->as.list.n; i++) {
					if (i)
						sb_append_str(&sb, sep);
					sb_append_str(&sb, value_as_cstr(base->as.list.items[i]));
				}
			}
			jinja_value *out = jinja_string(sb.p);
			free(sb.p);
			return out;
		}
		if (!strcmp(e->str, "lower") || !strcmp(e->str, "upper") || !strcmp(e->str, "capitalize")) {
			char *dup = xstrdup(value_as_cstr(base));
			if (!strcmp(e->str, "lower")) {
				for (char *c = dup; *c; c++)
					*c = tolower((unsigned char)*c);
			} else if (!strcmp(e->str, "upper")) {
				for (char *c = dup; *c; c++)
					*c = toupper((unsigned char)*c);
			} else {
				if (dup[0])
					dup[0] = toupper((unsigned char)dup[0]);
				for (char *c = dup + 1; *c; c++)
					*c = tolower((unsigned char)*c);
			}
			jinja_value *out = jinja_string(dup);
			free(dup);
			return out;
		}
		if (!strcmp(e->str, "startswith") || !strcmp(e->str, "endswith")) {
			const char *prefix = e->args ? value_as_cstr(eval_expr(ctx, e->args->val)) : "";
			const char *s	   = value_as_cstr(base);
			size_t		n	   = strlen(prefix);
			int			res	   = !strcmp(e->str, "startswith")
									 ? !strncmp(s, prefix, n)
									 : (strlen(s) >= n && !strcmp(s + strlen(s) - n, prefix));
			return jinja_bool(res);
		}
		everr(ctx, "unsupported method call");
		return jinja_none();
	}
	case EX_ISDEFINED: {
		jinja_value *v	  = eval_expr(ctx, e->a);
		const char	*test = e->str ? e->str : "defined";
		if (!strcmp(test, "none") || !strcmp(test, "Null")) {
			int is_none = !v || v->type == JV_NONE;
			return jinja_bool(is_none);
		}
		if (!strcmp(test, "undefined") || !strcmp(test, "defined")) {
			int is_none = !v || v->type == JV_NONE;
			return jinja_bool(!strcmp(test, "undefined") ? is_none : !is_none);
		}
		if (!strcmp(test, "string"))
			return jinja_bool(v && v->type == JV_STRING);
		if (!strcmp(test, "mapping") || !strcmp(test, "dict"))
			return jinja_bool(v && v->type == JV_DICT);
		if (!strcmp(test, "iterable"))
			return jinja_bool(v && (v->type == JV_LIST || v->type == JV_STRING));
		if (!strcmp(test, "number") || !strcmp(test, "integer"))
			return jinja_bool(v && v->type == JV_STRING && value_as_cstr(v)[0] &&
							  isdigit((unsigned char)value_as_cstr(v)[0]));
		if (!strcmp(test, "sequence"))
			return jinja_bool(v && v->type == JV_LIST);
		if (!strcmp(test, "boolean"))
			return jinja_bool(v && v->type == JV_BOOL);
		if (!strcmp(test, "true"))
			return jinja_bool(v && v->type == JV_BOOL && v->as.b);
		if (!strcmp(test, "false"))
			return jinja_bool(v && v->type == JV_BOOL && !v->as.b);
		everr(ctx, "unsupported test");
		return jinja_none();
	}
	case EX_FILTER:
		return eval_filter(ctx, e);
	case EX_BINOP:
		return eval_binop(ctx, e);
	}
	return jinja_none();
}

static void exec_stmts(eval_ctx *ctx, stmt_node *s, strbuf *out);

static void exec_stmt(eval_ctx *ctx, stmt_node *s, strbuf *out) {
	if (ctx->failed)
		return;
	switch (s->kind) {
	case ST_TEXT:
		sb_append_str(out, s->text);
		return;
	case ST_OUTPUT: {
		jinja_value *v = eval_expr(ctx, s->expr);
		sb_append_str(out, value_as_cstr(v));
		return;
	}
	case ST_IF:
		if (truthy(eval_expr(ctx, s->cond)))
			exec_stmts(ctx, s->then_body, out);
		else if (s->else_body)
			exec_stmts(ctx, s->else_body, out);
		return;
	case ST_FOR: {
		jinja_value *iter = eval_expr(ctx, s->iterable);
		if (!iter || iter->type != JV_LIST) {
			everr(ctx, "for loop over non-list");
			return;
		}
		size_t n = iter->as.list.n;
		for (size_t i = 0; i < n && !ctx->failed; i++) {
			scope loop_sc;
			loop_sc.entries = NULL;
			loop_sc.parent	= ctx->sc;

			jinja_value *item = iter->as.list.items[i];
			if (s->loop_var2 && item && item->type == JV_LIST && item->as.list.n >= 2) {
				scope_assign(&loop_sc, s->loop_var, item->as.list.items[0]);
				scope_assign(&loop_sc, s->loop_var2, item->as.list.items[1]);
			} else {
				scope_assign(&loop_sc, s->loop_var, item);
			}

			jinja_value *loop_obj = jinja_dict();
			char		 buf[32];
			snprintf(buf, sizeof(buf), "%zu", i);
			jinja_dict_set(loop_obj, "index0", jinja_string(buf));
			snprintf(buf, sizeof(buf), "%zu", i + 1);
			jinja_dict_set(loop_obj, "index", jinja_string(buf));
			jinja_dict_set(loop_obj, "first", jinja_bool(i == 0));
			jinja_dict_set(loop_obj, "last", jinja_bool(i == n - 1));
			scope_assign(&loop_sc, "loop", loop_obj);

			scope *saved = ctx->sc;
			ctx->sc		 = &loop_sc;
			exec_stmts(ctx, s->body, out);
			ctx->sc = saved;
			scope_free_entries(&loop_sc);
		}
		return;
	}
	case ST_SET: {
		jinja_value *val;
		if (s->set_val) {
			val = eval_expr(ctx, s->set_val);
		} else {
			strbuf sb;
			sb_init(&sb);
			exec_stmts(ctx, s->body, &sb);
			val = jinja_string(sb.p);
			free(sb.p);
		}
		if (s->set_attr) {
			jinja_value *target = scope_lookup(ctx->sc, s->set_name);
			if (!target || target->type != JV_DICT) {
				everr(ctx, "set attribute on non-namespace value");
				return;
			}
			jinja_dict_set(target, s->set_attr, val);
		} else {
			scope_assign(ctx->sc, s->set_name, val);
		}
		return;
	}
	case ST_MACRO: {
		jinja_value *mval = xmalloc(sizeof(*mval));
		mval->type		  = JV_MACRO;
		mval->as.macro	  = s;
		arena_track(mval);
		scope_assign(ctx->sc, s->macro_name, mval);
		return;
	}
	case ST_LIST:
		for (size_t i = 0; i < s->n_items && !ctx->failed; i++)
			exec_stmt(ctx, s->items[i], out);
		return;
	case ST_GENERATION:
		exec_stmts(ctx, s->body, out);
		return;
	case ST_NOOP:
		return;
	}
}

static void exec_stmts(eval_ctx *ctx, stmt_node *s, strbuf *out) {
	for (; s && !ctx->failed; s = s->next)
		exec_stmt(ctx, s, out);
}

status_code jinja_render(jinja_program *prog, jinja_value *globals, char **out, char *errbuf,
						 size_t errbuf_len) {
	jinja_arena	 arena = {0};
	jinja_arena *outer = g_render_arena;
	g_render_arena	   = &arena;

	scope global_sc;
	global_sc.entries = NULL;
	global_sc.parent  = NULL;
	if (globals && globals->type == JV_DICT) {
		for (jinja_dict_entry *e = globals->as.dict; e; e = e->next) {
			scope_entry *se	  = xmalloc(sizeof(*se));
			se->name		  = xstrdup(e->key);
			se->val			  = e->val;
			se->next		  = global_sc.entries;
			global_sc.entries = se;
		}
	}

	eval_ctx ctx;
	ctx.sc		   = &global_sc;
	ctx.errbuf	   = errbuf;
	ctx.errbuf_len = errbuf_len;
	ctx.failed	   = 0;
	ctx.call_depth = 0;

	strbuf sb;
	sb_init(&sb);
	exec_stmts(&ctx, prog, &sb);

	scope_free_entries(&global_sc);

	arena_free_all(&arena);
	g_render_arena = outer;

	if (ctx.failed) {
		free(sb.p);
		*out = NULL;
		return ERR_FORMAT;
	}

	*out = sb.p;
	return OK;
}