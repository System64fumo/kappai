#include "openai.h"

#include "common.h"
#include "json-c/json.h"
#include "log.h"
#include "microhttpd.h"
#include "sampler.h"

#include <errno.h>
#include <limits.h>
#include <netdb.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_BODY_BYTES (32 * 1024 * 1024)
#define SSE_POLL_SEC 1

typedef struct {
	char *role;
	char *content;
} oa_message;

typedef struct {
	bool	 stream;
	bool	 include_usage;
	float	 temperature;
	float	 top_p;
	float	 min_p;
	int		 top_k;
	int		 max_tokens;
	bool	 has_seed;
	uint64_t seed;

	json_object *stop;
	size_t		 n_stop;

	oa_message *messages;
	size_t		n_messages;

	char *prompt;
} oa_req_params;

typedef struct {
	char  *p;
	size_t len;
	size_t cap;
} oa_buf;

static void oa_buf_append(oa_buf *b, const char *s, size_t n) {
	if (b->len + n + 1 > b->cap) {
		b->cap = b->cap ? b->cap * 2 : 128;
		b->p   = xrealloc(b->p, b->cap);
	}
	memcpy(b->p + b->len, s, n);
	b->len += n;
	b->p[b->len] = '\0';
}

typedef struct {
	openai_state  *st;
	oa_req_params *params;

	bool streaming;
	bool chat_api;

	pthread_mutex_t q_mtx;
	pthread_cond_t	q_cv;
	char		  **queue;
	size_t		   *frame_lens;
	size_t			q_len;
	size_t			q_cap;
	size_t			q_off;
	bool			producer_done;
	bool			client_gone;

	oa_buf content;
	oa_buf reasoning;

	bool in_thinking;
	bool first_token;
	bool skip_label;

	bool stopped_by_stop;
	bool sent_any_chunk;

	int	 generated;
	int	 prompt_tokens;
	char id[64];
} oa_gen;

typedef struct req_ctx req_ctx;

struct req_ctx {
	openai_state *st;
	atomic_int	  refs;

	char  *body;
	size_t body_len;
	size_t body_cap;
	bool   too_large;

	oa_req_params params;
	json_object	 *body_json;
	oa_gen		  gen;
};

typedef struct {
	req_ctx *rc;
	bool	 chat_api;
} gen_job;

struct openai_state {
	context *ctx;

	char *model_id;
	char *api_key;

	long		  created_at;
	atomic_ullong req_counter;

	pthread_mutex_t gen_mtx;

	struct MHD_Daemon *daemon;
	struct sockaddr	  *bind_addr;
	socklen_t		   bind_len;
	atomic_bool		   shutting_down;
};

static sigset_t g_signal_mask;

static char *derive_model_id(const char *model_path) {
	const char *base = strrchr(model_path, '/');
	base			 = base ? base + 1 : model_path;
	size_t len		 = strlen(base);
	if (len > 5 && !strcasecmp(base + len - 5, ".gguf"))
		len -= 5;
	char *id = xmalloc(len + 1);
	memcpy(id, base, len);
	id[len] = '\0';
	return id;
}

static void add_cors(struct MHD_Response *resp) {
	MHD_add_response_header(resp, "Access-Control-Allow-Origin", "*");
}

static struct MHD_Response *json_response(json_object *body) {
	const char			*s = json_object_to_json_string_ext(body, JSON_C_TO_STRING_PLAIN);
	struct MHD_Response *resp =
		MHD_create_response_from_buffer(strlen(s), (void *)s, MHD_RESPMEM_MUST_COPY);
	if (resp) {
		MHD_add_response_header(resp, MHD_HTTP_HEADER_CONTENT_TYPE, "application/json");
		add_cors(resp);
	}
	return resp;
}

static enum MHD_Result respond_json(struct MHD_Connection *conn, unsigned int status,
									json_object *body) {
	struct MHD_Response *resp = json_response(body);
	json_object_put(body);
	enum MHD_Result r = MHD_queue_response(conn, status, resp);
	MHD_destroy_response(resp);
	return r;
}

static json_object *error_body(const char *message, const char *type) {
	json_object *root = json_object_new_object();
	json_object *err  = json_object_new_object();
	json_object_object_add(err, "message", json_object_new_string(message));
	json_object_object_add(err, "type", json_object_new_string(type));
	json_object_object_add(err, "param", NULL);
	json_object_object_add(err, "code", NULL);
	json_object_object_add(root, "error", err);
	return root;
}

static bool check_auth(openai_state *st, struct MHD_Connection *conn) {
	if (!st->api_key)
		return true;
	const char *auth = MHD_lookup_connection_value(conn, MHD_HEADER_KIND, "Authorization");
	return auth && !strncmp(auth, "Bearer ", 7) && !strcmp(auth + 7, st->api_key);
}

static bool shutting_down(openai_state *st) {
	return atomic_load_explicit(&st->shutting_down, memory_order_relaxed);
}

static char *extract_message_content(json_object *msg) {
	json_object *c;
	if (!json_object_object_get_ex(msg, "content", &c) || json_object_is_type(c, json_type_null))
		return xstrdup("");
	if (json_object_is_type(c, json_type_string))
		return xstrdup(json_object_get_string(c));
	if (json_object_is_type(c, json_type_array)) {
		char  *out = NULL;
		size_t len = 0, cap = 0;
		size_t n = json_object_array_length(c);
		for (size_t i = 0; i < n; i++) {
			json_object *part = json_object_array_get_idx(c, i);
			if (!json_object_is_type(part, json_type_object))
				continue;
			json_object *type, *text;
			const char	*ts = "";
			if (json_object_object_get_ex(part, "type", &type) &&
				json_object_is_type(type, json_type_string))
				ts = json_object_get_string(type);
			if (*ts && strcmp(ts, "text") != 0)
				continue;
			if (!json_object_object_get_ex(part, "text", &text) ||
				!json_object_is_type(text, json_type_string))
				continue;
			const char *t	 = json_object_get_string(text);
			size_t		tlen = strlen(t);
			if (len + tlen + 1 > cap) {
				cap = (len + tlen + 1) * 2;
				out = xrealloc(out, cap);
			}
			memcpy(out + len, t, tlen);
			len += tlen;
			out[len] = '\0';
		}
		return out ? out : xstrdup("");
	}
	return xstrdup("");
}

static double get_num_or(json_object *root, const char *key, double fallback) {
	json_object *v;
	if (!json_object_object_get_ex(root, key, &v))
		return fallback;
	return json_object_get_double(v);
}

static bool get_bool_or(json_object *root, const char *key, bool fallback) {
	json_object *v;
	if (!json_object_object_get_ex(root, key, &v) || !json_object_is_type(v, json_type_boolean))
		return fallback;
	return json_object_get_boolean(v);
}

static int64_t get_int_or(json_object *root, const char *key, int64_t fallback) {
	json_object *v;
	if (!json_object_object_get_ex(root, key, &v))
		return fallback;
	return json_object_get_int64(v);
}

static const char *parse_stop(json_object *root, oa_req_params *p) {
	json_object *stop;
	if (!json_object_object_get_ex(root, "stop", &stop))
		return NULL;
	if (json_object_is_type(stop, json_type_string)) {
		p->stop	  = stop;
		p->n_stop = json_object_get_string_len(stop) ? 1 : 0;
		return NULL;
	}
	if (json_object_is_type(stop, json_type_array)) {
		p->stop	 = stop;
		size_t n = json_object_array_length(stop);
		for (size_t i = 0; i < n; i++) {
			json_object *item = json_object_array_get_idx(stop, i);
			if (!json_object_is_type(item, json_type_string))
				continue;
			if (json_object_get_string_len(item) == 0)
				continue;
			p->n_stop++;
		}
		return NULL;
	}
	return "'stop' must be a string or an array of strings";
}

static void parse_sampling_params(json_object *root, oa_req_params *p) {
	p->stream		 = get_bool_or(root, "stream", false);
	p->include_usage = false;
	json_object *so;
	if (json_object_object_get_ex(root, "stream_options", &so) &&
		json_object_is_type(so, json_type_object))
		p->include_usage = get_bool_or(so, "include_usage", false);

	double d	   = get_num_or(root, "temperature", 0.80);
	p->temperature = d < 0 ? 0 : (d > 2 ? 2 : (float)d);
	d			   = get_num_or(root, "top_p", 0.90);
	p->top_p	   = d <= 0 ? 1e-3f : (d > 1 ? 1 : (float)d);
	d			   = get_num_or(root, "min_p", 0.10);
	p->min_p	   = d < 0 ? 0 : (d > 1 ? 1 : (float)d);

	int64_t k = get_int_or(root, "top_k", 40);
	p->top_k  = k < 0 ? 0 : (int)(k > INT_MAX ? INT_MAX : k);

	int64_t mt = -1;
	mt		   = get_int_or(root, "max_completion_tokens", mt);
	if (mt < 0)
		mt = get_int_or(root, "max_tokens", -1);
	p->max_tokens = (int)(mt > 100000000 ? 100000000 : mt);

	p->has_seed = false;
	json_object *seed;
	if (json_object_object_get_ex(root, "seed", &seed) &&
		json_object_is_type(seed, json_type_int)) {
		int64_t s = json_object_get_int64(seed);
		if (s >= 0) {
			p->has_seed = true;
			p->seed		= (uint64_t)s;
		}
	}

	parse_stop(root, p);
}

static const char *parse_chat_request(json_object *root, oa_req_params *p) {
	parse_sampling_params(root, p);

	json_object *msgs;
	if (!json_object_object_get_ex(root, "messages", &msgs) ||
		!json_object_is_type(msgs, json_type_array))
		return "'messages' must be a non-empty array";
	size_t n = json_object_array_length(msgs);
	if (n == 0)
		return "'messages' must be a non-empty array";

	p->messages = xcalloc(n, sizeof(*p->messages));
	for (size_t i = 0; i < n; i++) {
		json_object *m = json_object_array_get_idx(msgs, i);
		if (!json_object_is_type(m, json_type_object))
			return "every element of 'messages' must be an object";
		json_object *role;
		const char	*r = "user";
		if (json_object_object_get_ex(m, "role", &role) &&
			json_object_is_type(role, json_type_string))
			r = json_object_get_string(role);
		p->messages[i].role	   = xstrdup(r);
		p->messages[i].content = extract_message_content(m);
	}
	p->n_messages = n;
	return NULL;
}

static const char *parse_completion_request(json_object *root, oa_req_params *p) {
	parse_sampling_params(root, p);

	json_object *prompt;
	if (!json_object_object_get_ex(root, "prompt", &prompt))
		return "'prompt' is required";
	if (json_object_is_type(prompt, json_type_string)) {
		p->prompt = xstrdup(json_object_get_string(prompt));
		return NULL;
	}
	if (json_object_is_type(prompt, json_type_array)) {
		size_t n = json_object_array_length(prompt);
		for (size_t i = 0; i < n; i++)
			if (!json_object_is_type(json_object_array_get_idx(prompt, i), json_type_string))
				return "'prompt' array elements must be strings";
		size_t cap = 256, len = 0;
		char  *out = xmalloc(cap);
		out[0]	   = '\0';
		for (size_t i = 0; i < n; i++) {
			const char *s  = json_object_get_string(json_object_array_get_idx(prompt, i));
			size_t		sl = strlen(s);
			while (len + sl + 2 > cap) {
				cap *= 2;
				out = xrealloc(out, cap);
			}
			if (i)
				out[len++] = '\n';
			memcpy(out + len, s, sl);
			len += sl;
			out[len] = '\0';
		}
		p->prompt = out;
		return NULL;
	}
	return "'prompt' must be a string";
}

static void free_req_params(oa_req_params *p) {
	for (size_t i = 0; i < p->n_messages; i++) {
		free(p->messages[i].role);
		free(p->messages[i].content);
	}
	free(p->messages);
	free(p->prompt);
	memset(p, 0, sizeof(*p));
}

static void rc_ref(req_ctx *rc) {
	atomic_fetch_add_explicit(&rc->refs, 1, memory_order_relaxed);
}

static void rc_unref(req_ctx *rc) {
	if (!rc || atomic_fetch_sub_explicit(&rc->refs, 1, memory_order_acq_rel) != 1)
		return;
	free(rc->body);
	json_object_put(rc->body_json);
	for (size_t i = 0; i < rc->gen.q_len; i++)
		free(rc->gen.queue[i]);
	free(rc->gen.queue);
	free(rc->gen.frame_lens);
	free(rc->gen.content.p);
	free(rc->gen.reasoning.p);
	pthread_mutex_destroy(&rc->gen.q_mtx);
	pthread_cond_destroy(&rc->gen.q_cv);
	free_req_params(&rc->params);
	free(rc);
}

static req_ctx *rc_new(openai_state *st) {
	req_ctx *rc = xcalloc(1, sizeof(*rc));
	rc->st		= st;
	atomic_store_explicit(&rc->refs, 1, memory_order_relaxed);
	pthread_mutex_init(&rc->gen.q_mtx, NULL);
	pthread_cond_init(&rc->gen.q_cv, NULL);
	rc->gen.st	   = st;
	rc->gen.params = &rc->params;
	return rc;
}

static void gen_queue_push(oa_gen *g, const char *payload, size_t len) {
	pthread_mutex_lock(&g->q_mtx);
	if (!g->client_gone && !shutting_down(g->st)) {
		ARR_RESERVE(g->queue, g->q_len, g->q_cap);
		if (g->q_cap && !g->frame_lens)
			g->frame_lens = xmalloc(g->q_cap * sizeof(*g->frame_lens));
		else if (g->frame_lens && g->q_len == g->q_cap)
			g->frame_lens = xrealloc(g->frame_lens, g->q_cap * sizeof(*g->frame_lens));
		char *frame = xmalloc(6 + len + 2);
		memcpy(frame, "data: ", 6);
		memcpy(frame + 6, payload, len);
		frame[6 + len]			= '\n';
		frame[6 + len + 1]		= '\n';
		g->queue[g->q_len]		= frame;
		g->frame_lens[g->q_len] = len + 8;
		g->q_len++;
	}
	pthread_cond_broadcast(&g->q_cv);
	pthread_mutex_unlock(&g->q_mtx);
}

static void gen_queue_done(oa_gen *g) {
	pthread_mutex_lock(&g->q_mtx);
	g->producer_done = true;
	pthread_cond_broadcast(&g->q_cv);
	pthread_mutex_unlock(&g->q_mtx);
}

static void gen_mark_client_gone(oa_gen *g) {
	pthread_mutex_lock(&g->q_mtx);
	g->client_gone = true;
	pthread_cond_broadcast(&g->q_cv);
	pthread_mutex_unlock(&g->q_mtx);
}

static void make_id(oa_gen *g) {
	uint64_t seq = atomic_fetch_add_explicit(&g->st->req_counter, 1, memory_order_relaxed);
	snprintf(g->id, sizeof(g->id), "%s-%llx-%llu", g->chat_api ? "chatcmpl" : "cmpl",
			 (unsigned long long)time(NULL), (unsigned long long)seq);
}

static const char *compute_finish_reason(const oa_gen *g) {
	if (g->stopped_by_stop)
		return "stop";
	if (g->params->max_tokens > 0 && g->generated >= g->params->max_tokens)
		return "length";
	return "stop";
}

static json_object *chunk_head(oa_gen *g, const char *object) {
	json_object *chunk = json_object_new_object();
	json_object_object_add(chunk, "id", json_object_new_string(g->id));
	json_object_object_add(chunk, "object", json_object_new_string(object));
	json_object_object_add(chunk, "created", json_object_new_int64((int64_t)time(NULL)));
	json_object_object_add(chunk, "model", json_object_new_string(g->st->model_id));
	return chunk;
}

static void append_usage(json_object *root, int prompt_tokens, int completion_tokens) {
	json_object *usage = json_object_new_object();
	json_object_object_add(usage, "prompt_tokens", json_object_new_int(prompt_tokens));
	json_object_object_add(usage, "completion_tokens", json_object_new_int(completion_tokens));
	json_object_object_add(usage, "total_tokens",
						   json_object_new_int(prompt_tokens + completion_tokens));
	json_object_object_add(root, "usage", usage);
}

static json_object *chat_choice_delta(json_object *delta, const char *finish_reason) {
	json_object *choice = json_object_new_object();
	json_object_object_add(choice, "index", json_object_new_int(0));
	json_object_object_add(choice, "delta", delta ? delta : json_object_new_object());
	json_object_object_add(choice, "logprobs", NULL);
	json_object_object_add(choice, "finish_reason",
						   finish_reason ? json_object_new_string(finish_reason) : NULL);
	return choice;
}

static json_object *text_choice_delta(const char *piece, size_t n, const char *finish_reason) {
	json_object *choice = json_object_new_object();
	json_object_object_add(choice, "text", json_object_new_string_len(piece ? piece : "", (int)n));
	json_object_object_add(choice, "index", json_object_new_int(0));
	json_object_object_add(choice, "logprobs", NULL);
	json_object_object_add(choice, "finish_reason",
						   finish_reason ? json_object_new_string(finish_reason) : NULL);
	return choice;
}

static json_object *build_completion_body(req_ctx *rc, bool chat_api) {
	oa_gen		 *g	   = &rc->gen;
	openai_state *st   = rc->st;
	json_object	 *root = json_object_new_object();
	json_object_object_add(root, "id", json_object_new_string(g->id));
	json_object_object_add(
		root, "object", json_object_new_string(chat_api ? "chat.completion" : "text_completion"));
	json_object_object_add(root, "created", json_object_new_int64((int64_t)time(NULL)));
	json_object_object_add(root, "model", json_object_new_string(st->model_id));

	json_object *choices = json_object_new_array();
	json_object *choice	 = json_object_new_object();
	json_object_object_add(choice, "index", json_object_new_int(0));
	if (chat_api) {
		json_object *msg = json_object_new_object();
		json_object_object_add(msg, "role", json_object_new_string("assistant"));
		json_object_object_add(msg, "content",
							   json_object_new_string(g->content.p ? g->content.p : ""));
		if (g->reasoning.len > 0)
			json_object_object_add(msg, "reasoning_content",
								   json_object_new_string(g->reasoning.p));
		json_object_object_add(choice, "message", msg);
	} else {
		json_object_object_add(choice, "text",
							   json_object_new_string(g->content.p ? g->content.p : ""));
	}
	json_object_object_add(choice, "logprobs", NULL);
	json_object_object_add(choice, "finish_reason",
						   json_object_new_string(compute_finish_reason(g)));
	json_object_array_add(choices, choice);
	json_object_object_add(root, "choices", choices);
	append_usage(root, g->prompt_tokens, g->generated);
	return root;
}

static json_object *health_body(void) {
	json_object *root = json_object_new_object();
	json_object_object_add(root, "status", json_object_new_string("ok"));
	return root;
}

static json_object *root_body(openai_state *st) {
	json_object *root = json_object_new_object();
	json_object_object_add(root, "status", json_object_new_string("ok"));
	json_object_object_add(root, "model", json_object_new_string(st->model_id));
	json_object *eps = json_object_new_array();
	json_object_array_add(eps, json_object_new_string("GET /health"));
	json_object_array_add(eps, json_object_new_string("GET /v1/models"));
	json_object_array_add(eps, json_object_new_string("POST /v1/chat/completions"));
	json_object_array_add(eps, json_object_new_string("POST /v1/completions"));
	json_object_object_add(root, "endpoints", eps);
	return root;
}

static json_object *models_body(openai_state *st) {
	json_object *root = json_object_new_object();
	json_object_object_add(root, "object", json_object_new_string("list"));
	json_object *list = json_object_new_array();
	json_object *m	  = json_object_new_object();
	json_object_object_add(m, "id", json_object_new_string(st->model_id));
	json_object_object_add(m, "object", json_object_new_string("model"));
	json_object_object_add(m, "created", json_object_new_int64((int64_t)st->created_at));
	json_object_object_add(m, "owned_by", json_object_new_string("kappai"));
	json_object_array_add(list, m);
	json_object_object_add(root, "data", list);
	return root;
}

static void sse_push_chunk(oa_gen *g, json_object *chunk) {
	const char *s = json_object_to_json_string_ext(chunk, JSON_C_TO_STRING_PLAIN);
	gen_queue_push(g, s, strlen(s));
	json_object_put(chunk);
}

static void sse_chat_delta(oa_gen *g, json_object *delta, const char *finish_reason) {
	json_object *chunk	 = chunk_head(g, "chat.completion.chunk");
	json_object *choices = json_object_new_array();
	json_object_array_add(choices, chat_choice_delta(delta, finish_reason));
	json_object_object_add(chunk, "choices", choices);
	if (g->params->include_usage)
		json_object_object_add(chunk, "usage", NULL);
	sse_push_chunk(g, chunk);
}

static void sse_text_delta(oa_gen *g, const char *piece, size_t n, const char *finish_reason) {
	json_object *chunk	 = chunk_head(g, "text_completion");
	json_object *choices = json_object_new_array();
	json_object_array_add(choices, text_choice_delta(piece, n, finish_reason));
	json_object_object_add(chunk, "choices", choices);
	sse_push_chunk(g, chunk);
}

static void sse_send_role_chunk(oa_gen *g) {
	json_object *delta = json_object_new_object();
	json_object_object_add(delta, "role", json_object_new_string("assistant"));
	json_object_object_add(delta, "content", json_object_new_string(""));
	sse_chat_delta(g, delta, NULL);
}

static void sse_send_delta(oa_gen *g, bool reasoning, const char *piece, size_t n) {
	if (!g->sent_any_chunk) {
		g->sent_any_chunk = true;
		if (g->chat_api)
			sse_send_role_chunk(g);
		else
			sse_text_delta(g, "", 0, NULL);
	}
	if (g->client_gone || shutting_down(g->st))
		return;
	if (g->chat_api) {
		json_object *delta = json_object_new_object();
		json_object_object_add(delta, reasoning ? "reasoning_content" : "content",
							   json_object_new_string_len(piece, (int)n));
		sse_chat_delta(g, delta, NULL);
	} else {
		sse_text_delta(g, piece, n, NULL);
	}
}

static void check_stop_sequences(oa_gen *g) {
	if (g->stopped_by_stop || g->params->n_stop == 0)
		return;
	if (!g->content.p && !g->reasoning.p)
		return;

	size_t rlen	 = g->reasoning.len;
	size_t clen	 = g->content.len;
	size_t total = rlen + clen;
	if (total == 0)
		return;

	bool   is_string	= json_object_is_type(g->params->stop, json_type_string);
	size_t needle_count = is_string ? 1 : json_object_array_length(g->params->stop);
	size_t max_needle	= 0;

	for (size_t i = 0; i < needle_count; i++) {
		json_object *item =
			is_string ? g->params->stop : json_object_array_get_idx(g->params->stop, i);
		if (!json_object_is_type(item, json_type_string))
			continue;
		size_t l = (size_t)json_object_get_string_len(item);
		if (l > max_needle)
			max_needle = l;
	}
	if (max_needle == 0 || total < max_needle)
		return;

	size_t window_len	= 2 * max_needle - 1;
	size_t window_start = total > window_len ? total - window_len : 0;
	size_t wlen			= total - window_start;
	char  *window		= xmalloc(wlen + 1);
	if (window_start < rlen) {
		memcpy(window, g->reasoning.p + window_start, rlen - window_start);
		memcpy(window + (rlen - window_start), g->content.p, clen);
	} else {
		memcpy(window, g->content.p + (window_start - rlen), wlen);
	}
	window[wlen] = '\0';

	for (size_t i = 0; i < needle_count && !g->stopped_by_stop; i++) {
		json_object *item =
			is_string ? g->params->stop : json_object_array_get_idx(g->params->stop, i);
		if (!json_object_is_type(item, json_type_string))
			continue;
		const char *needle = json_object_get_string(item);
		if (!needle[0])
			continue;
		const char *hit = strstr(window, needle);
		if (!hit)
			continue;

		g->stopped_by_stop = true;
		size_t cut		   = window_start + (size_t)(hit - window);
		if (cut <= rlen) {
			g->reasoning.len = cut;
			g->content.len	 = 0;
		} else {
			g->content.len = cut - rlen;
		}
		if (g->reasoning.p)
			g->reasoning.p[g->reasoning.len] = '\0';
		if (g->content.p)
			g->content.p[g->content.len] = '\0';
	}
	free(window);
}

static void gen_on_token(int32_t id, const char *piece, int n, void *ud) {
	oa_gen *g = (oa_gen *)ud;

	if (g->client_gone || shutting_down(g->st)) {
		g->st->ctx->interrupt = 1;
		return;
	}

	if (!g->chat_api) {
		oa_buf_append(&g->content, piece, (size_t)n);
		check_stop_sequences(g);
		if (g->stopped_by_stop) {
			g->st->ctx->interrupt = 1;
			return;
		}
		if (g->streaming)
			sse_send_delta(g, false, piece, (size_t)n);
		return;
	}

	if (g->first_token && g->st->ctx->chat.think_open)
		g->in_thinking = true;
	g->first_token = false;

	if (id == g->st->ctx->chat.think_start_id) {
		g->in_thinking = true;
		g->skip_label  = true;
		return;
	}
	if (id == g->st->ctx->chat.think_end_id) {
		g->in_thinking = false;
		return;
	}

	if (g->skip_label) {
		const char *nl = memchr(piece, '\n', (size_t)n);
		if (!nl)
			return;
		n			  = n - (int)(nl - piece) - 1;
		piece		  = nl + 1;
		g->skip_label = false;
	}
	if (n <= 0)
		return;

	oa_buf *dst = g->in_thinking ? &g->reasoning : &g->content;
	oa_buf_append(dst, piece, (size_t)n);
	check_stop_sequences(g);
	if (g->stopped_by_stop) {
		g->st->ctx->interrupt = 1;
		return;
	}

	if (g->streaming)
		sse_send_delta(g, g->in_thinking, piece, (size_t)n);
}

static void sse_finish_stream(oa_gen *g) {
	const char *finish = compute_finish_reason(g);

	if (!g->client_gone && !shutting_down(g->st)) {
		if (g->chat_api) {
			if (!g->sent_any_chunk)
				sse_send_role_chunk(g);
			sse_chat_delta(g, NULL, finish);
		} else {
			if (!g->sent_any_chunk)
				sse_text_delta(g, "", 0, NULL);
			sse_text_delta(g, "", 0, finish);
		}

		if (g->params->include_usage) {
			json_object *chunk =
				chunk_head(g, g->chat_api ? "chat.completion.chunk" : "text_completion");
			json_object_object_add(chunk, "choices", json_object_new_array());
			append_usage(chunk, g->prompt_tokens, g->generated);
			sse_push_chunk(g, chunk);
		}

		gen_queue_push(g, "[DONE]", 6);
	}
	gen_queue_done(g);
}

static void run_generation(req_ctx *rc, bool chat_api) {
	openai_state *st = rc->st;
	oa_gen		 *g	 = &rc->gen;
	context		 *c	 = st->ctx;

	g->streaming = rc->params.stream;
	g->chat_api	 = chat_api;
	make_id(g);

	sampler_params sp = {rc->params.temperature, rc->params.top_k, rc->params.top_p,
						 rc->params.min_p};

	pthread_mutex_lock(&st->gen_mtx);
	{
		if (rc->params.has_seed) {
			sampler_free(&c->samp);
			sampler_init(&c->samp, rc->params.seed);
			sampler_set_vocab(&c->samp, c->m.vocab_size);
		}
		context_reset(c);

		if (chat_api) {
			for (size_t i = 0; i + 1 < rc->params.n_messages; i++)
				chat_template_add_message(&c->chat, rc->params.messages[i].role,
										  rc->params.messages[i].content);
			g->generated = context_chat_turn(c, rc->params.messages[rc->params.n_messages - 1].role,
											 rc->params.messages[rc->params.n_messages - 1].content,
											 true, rc->params.max_tokens, &sp, gen_on_token, g, "");
		} else {
			g->generated = context_completion(c, rc->params.prompt, rc->params.max_tokens, &sp,
											  gen_on_token, g);
		}
		c->interrupt	 = 0;
		g->prompt_tokens = c->last_prompt_tokens;
	}
	pthread_mutex_unlock(&st->gen_mtx);

	if (g->streaming)
		sse_finish_stream(g);
}

static ssize_t sse_read_cb(void *cls, uint64_t pos, char *buf, size_t max) {
	req_ctx *rc = (req_ctx *)cls;
	oa_gen	*g	= &rc->gen;
	(void)pos;

	pthread_mutex_lock(&g->q_mtx);
	while (g->q_len == 0 && !g->producer_done && !g->client_gone && !shutting_down(g->st)) {
		struct timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);
		ts.tv_sec += SSE_POLL_SEC;
		pthread_cond_timedwait(&g->q_cv, &g->q_mtx, &ts);
	}

	ssize_t out;
	if (g->q_len > 0) {
		char  *f	  = g->queue[0];
		size_t remain = g->frame_lens[0] - g->q_off;
		size_t n	  = remain < max ? remain : max;
		memcpy(buf, f + g->q_off, n);
		g->q_off += n;
		if (g->q_off == g->frame_lens[0]) {
			free(f);
			memmove(g->queue, g->queue + 1, (g->q_len - 1) * sizeof(*g->queue));
			memmove(g->frame_lens, g->frame_lens + 1, (g->q_len - 1) * sizeof(*g->frame_lens));
			g->q_len--;
			g->q_off = 0;
		}
		out = (ssize_t)n;
	} else if (g->producer_done) {
		out = MHD_CONTENT_READER_END_OF_STREAM;
	} else {
		out = MHD_CONTENT_READER_END_WITH_ERROR;
	}
	pthread_mutex_unlock(&g->q_mtx);
	return out;
}

static void sse_free_cb(void *cls) {
	rc_unref((req_ctx *)cls);
}

static void request_completed(void *cls, struct MHD_Connection *conn, void **con_cls,
							  enum MHD_RequestTerminationCode toe) {
	(void)cls;
	(void)conn;
	(void)toe;
	req_ctx *rc = *(req_ctx **)con_cls;
	if (!rc)
		return;
	gen_mark_client_gone(&rc->gen);
	*(req_ctx **)con_cls = NULL;
	rc_unref(rc);
}

static void *gen_thread_main(void *arg) {
	gen_job *job = (gen_job *)arg;
	uint64_t t0	 = time_us();
	run_generation(job->rc, job->chat_api);
	INFO("gen: %s prompt=%d tokens generated=%d (%.1f s, %s)",
		 job->chat_api ? "chat" : "completion", job->rc->gen.prompt_tokens, job->rc->gen.generated,
		 (double)(time_us() - t0) / 1e6,
		 job->rc->gen.client_gone ? "client disconnected" : "completed");
	rc_unref(job->rc);
	free(job);
	return NULL;
}

static enum MHD_Result handle_post(openai_state *st, struct MHD_Connection *conn, req_ctx *rc,
								   const char *url) {
	if (!check_auth(st, conn))
		return respond_json(conn, MHD_HTTP_UNAUTHORIZED,
							error_body("Invalid API key", "authentication_error"));

	json_object *body = json_tokener_parse(rc->body ? rc->body : "");
	if (!body)
		return respond_json(conn, MHD_HTTP_BAD_REQUEST,
							error_body("Invalid JSON in request body", "invalid_request_error"));
	if (!json_object_is_type(body, json_type_object)) {
		json_object_put(body);
		return respond_json(
			conn, MHD_HTTP_BAD_REQUEST,
			error_body("Request body must be a JSON object", "invalid_request_error"));
	}

	bool		chat_api = strcmp(url, "/v1/chat/completions") == 0;
	const char *err		 = chat_api ? parse_chat_request(body, &rc->params)
									: parse_completion_request(body, &rc->params);
	if (err) {
		json_object_put(body);
		return respond_json(conn, MHD_HTTP_BAD_REQUEST, error_body(err, "invalid_request_error"));
	}
	rc->body_json = body;

	rc->gen.streaming = rc->params.stream;

	if (rc->gen.streaming) {
		gen_job *job  = xmalloc(sizeof(*job));
		job->rc		  = rc;
		job->chat_api = chat_api;
		pthread_t tid;
		rc_ref(rc);
		if (pthread_create(&tid, NULL, gen_thread_main, job) != 0) {
			rc_unref(rc);
			free(job);
			return respond_json(conn, MHD_HTTP_SERVICE_UNAVAILABLE,
								error_body("server overloaded", "server_error"));
		}
		rc_ref(rc);
		struct MHD_Response *resp =
			MHD_create_response_from_callback(MHD_SIZE_UNKNOWN, 4096, sse_read_cb, rc, sse_free_cb);
		if (!resp) {
			gen_mark_client_gone(&rc->gen);
			rc_unref(rc);
			return MHD_NO;
		}
		MHD_add_response_header(resp, MHD_HTTP_HEADER_CONTENT_TYPE, "text/event-stream");
		MHD_add_response_header(resp, "Cache-Control", "no-cache");
		MHD_add_response_header(resp, "X-Accel-Buffering", "no");
		add_cors(resp);
		enum MHD_Result r = MHD_queue_response(conn, MHD_HTTP_OK, resp);
		MHD_destroy_response(resp);
		return r;
	}

	run_generation(rc, chat_api);

	if (rc->gen.generated < 0) {
		INFO("gen: %s failed after %d tokens", chat_api ? "chat" : "completion", rc->gen.generated);
		return respond_json(conn, MHD_HTTP_INTERNAL_SERVER_ERROR,
							error_body("generation failed (prompt too long or compute "
									   "error)",
									   "server_error"));
	}

	INFO("gen: %s prompt=%d tokens generated=%d finish=%s", chat_api ? "chat" : "completion",
		 rc->gen.prompt_tokens, rc->gen.generated, compute_finish_reason(&rc->gen));

	return respond_json(conn, MHD_HTTP_OK, build_completion_body(rc, chat_api));
}

static enum MHD_Result handle_request(void *cls, struct MHD_Connection *conn, const char *url,
									  const char *method, const char *version,
									  const char *upload_data, size_t *upload_data_size,
									  void **con_cls) {
	openai_state *st = (openai_state *)cls;
	(void)version;

	if (*con_cls == NULL) {
		*con_cls = rc_new(st);
		return MHD_YES;
	}
	req_ctx *rc = *(req_ctx **)con_cls;

	if (*upload_data_size > 0) {
		if (rc->body_len + *upload_data_size > MAX_BODY_BYTES) {
			rc->too_large = true;
		} else {
			if (rc->body_len + *upload_data_size + 1 > rc->body_cap) {
				rc->body_cap = (rc->body_len + *upload_data_size + 1) * 2;
				rc->body	 = xrealloc(rc->body, rc->body_cap);
			}
			memcpy(rc->body + rc->body_len, upload_data, *upload_data_size);
			rc->body_len += *upload_data_size;
			rc->body[rc->body_len] = '\0';
		}
		*upload_data_size = 0;
		return MHD_YES;
	}

	if (strcmp(method, "OPTIONS") == 0) {
		struct MHD_Response *resp =
			MHD_create_response_from_buffer(0, (void *)"", MHD_RESPMEM_PERSISTENT);
		MHD_add_response_header(resp, "Access-Control-Allow-Origin", "*");
		MHD_add_response_header(resp, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
		MHD_add_response_header(resp, "Access-Control-Allow-Headers",
								"Content-Type, Authorization");
		MHD_add_response_header(resp, "Access-Control-Max-Age", "86400");
		enum MHD_Result r = MHD_queue_response(conn, MHD_HTTP_NO_CONTENT, resp);
		MHD_destroy_response(resp);
		return r;
	}

	if (strcmp(url, "/health") == 0 && strcmp(method, "GET") == 0)
		return respond_json(conn, MHD_HTTP_OK, health_body());

	if (strcmp(url, "/") == 0 && strcmp(method, "GET") == 0)
		return respond_json(conn, MHD_HTTP_OK, root_body(st));

	if ((strcmp(url, "/v1/chat/completions") == 0 || strcmp(url, "/v1/completions") == 0) &&
		strcmp(method, "POST") == 0) {
		if (rc->too_large)
			return respond_json(conn, MHD_HTTP_PAYLOAD_TOO_LARGE,
								error_body("request body too large", "invalid_request_error"));
		uint64_t		t0 = time_us();
		enum MHD_Result r  = handle_post(st, conn, rc, url);
		INFO("%s %s (%.1f ms)", method, url, (double)(time_us() - t0) / 1000.0);
		return r;
	}

	if (strcmp(url, "/v1/models") == 0) {
		if (strcmp(method, "GET") != 0)
			return respond_json(conn, MHD_HTTP_METHOD_NOT_ALLOWED,
								error_body("Method not allowed", "invalid_request_error"));
		if (!check_auth(st, conn))
			return respond_json(conn, MHD_HTTP_UNAUTHORIZED,
								error_body("Invalid API key", "authentication_error"));
		return respond_json(conn, MHD_HTTP_OK, models_body(st));
	}

	char msgbuf[192];
	snprintf(msgbuf, sizeof(msgbuf), "Unknown endpoint: %s %s", method, url);
	json_object *root = json_object_new_object();
	json_object_object_add(root, "error", error_body(msgbuf, "invalid_request_error"));
	return respond_json(conn, MHD_HTTP_NOT_FOUND, root);
}

openai_state *openai_init(context *ctx, const cli_args *args) {
	openai_state *st = xcalloc(1, sizeof(*st));
	st->ctx			 = ctx;
	st->model_id	 = derive_model_id(config_get()->model ? config_get()->model : "unknown");
	st->api_key		 = args->server_api_key ? xstrdup(args->server_api_key) : NULL;
	st->created_at	 = (long)time(NULL);
	atomic_store_explicit(&st->req_counter, 0, memory_order_relaxed);
	atomic_store_explicit(&st->shutting_down, false, memory_order_relaxed);
	pthread_mutex_init(&st->gen_mtx, NULL);
	INFO("openai: model=%s api-key=%s", st->model_id, st->api_key ? "set" : "unset");
	return st;
}

bool openai_bind(openai_state *st, const char *host, int port) {
	if (!host || !host[0])
		host = "127.0.0.1";
	char portbuf[16];
	snprintf(portbuf, sizeof(portbuf), "%d", port);

	struct addrinfo hints;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family		= AF_UNSPEC;
	hints.ai_socktype	= SOCK_STREAM;
	hints.ai_flags		= AI_PASSIVE;
	struct addrinfo *ai = NULL;
	int				 gr = getaddrinfo(host, portbuf, &hints, &ai);
	if (gr != 0 || !ai) {
		ERROR("failed to resolve '%s': %s", host, gr ? gai_strerror(gr) : "no results");
		if (ai)
			freeaddrinfo(ai);
		return false;
	}
	st->bind_addr = xmalloc(ai->ai_addrlen);
	memcpy(st->bind_addr, ai->ai_addr, ai->ai_addrlen);
	st->bind_len = ai->ai_addrlen;
	freeaddrinfo(ai);
	return true;
}

bool openai_serve(openai_state *st, char *errbuf, size_t errbuf_len) {
	unsigned flags =
		MHD_USE_THREAD_PER_CONNECTION | MHD_USE_INTERNAL_POLLING_THREAD | MHD_USE_ERROR_LOG;
	st->daemon =
		MHD_start_daemon(flags, 0, NULL, NULL, handle_request, st, MHD_OPTION_SOCK_ADDR,
						 st->bind_addr, MHD_OPTION_CONNECTION_TIMEOUT, (unsigned int)600,
						 MHD_OPTION_NOTIFY_COMPLETED, request_completed, st, MHD_OPTION_END);
	if (!st->daemon) {
		snprintf(errbuf, errbuf_len, "MHD_start_daemon failed (%s)", strerror(errno));
		return false;
	}
	return true;
}

void openai_install_signals(void) {
	sigemptyset(&g_signal_mask);
	sigaddset(&g_signal_mask, SIGINT);
	sigaddset(&g_signal_mask, SIGTERM);
	pthread_sigmask(SIG_BLOCK, &g_signal_mask, NULL);
}

void openai_wait_for_signal(void) {
	int sig = 0;
	sigwait(&g_signal_mask, &sig);
	INFO("received signal %d; shutting down", sig);
}

void openai_stop(openai_state *st) {
	if (!st)
		return;
	atomic_store_explicit(&st->shutting_down, true, memory_order_relaxed);
	if (st->daemon)
		MHD_stop_daemon(st->daemon);
	st->daemon = NULL;
}

void openai_free(openai_state *st) {
	if (!st)
		return;
	openai_stop(st);
	free(st->bind_addr);
	free(st->model_id);
	free(st->api_key);
	pthread_mutex_destroy(&st->gen_mtx);
	free(st);
}
