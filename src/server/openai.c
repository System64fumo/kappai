#include "openai.h"

#include "common.h"
#include "json-c/json.h"
#include "json_helpers.h"
#include "log.h"
#include "microhttpd.h"
#include "sampler.h"
#include "toolcall.h"

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
	char		*role;
	char		*content;
	char		*reasoning_content;
	json_object *tool_calls;
	const char	*tool_call_id;
	const char	*name;
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

	json_object *tools;
	char		*tool_choice;
	char		*forced_function;

	char *prompt;
} oa_req_params;

typedef struct {
	openai_state  *st;
	oa_req_params *params;

	bool	 streaming;
	bool	 chat_api;
	bool	 same_session;
	uint64_t session_key_at_start;
	bool	 generation_done;

	pthread_mutex_t q_mtx;
	pthread_cond_t	q_cv;
	char		  **queue;
	size_t		   *frame_lens;
	size_t			q_len;
	size_t			q_cap;
	size_t			q_off;
	bool			producer_done;
	bool			client_gone;

	toolcall_buf content;
	toolcall_buf reasoning;

	bool in_thinking;
	bool first_token;
	bool skip_label;

	bool stopped_by_stop;
	bool sent_any_chunk;
	bool tool_fill_logged;

	toolcall_scanner *tsc;

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

	bool	  has_session;
	uint64_t *prefix_hashes;
	size_t	  prefix_len;

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
	json_set_str(err, "message", message);
	json_set_str(err, "type", type);
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
	json_object *c = json_get(msg, "content");
	if (!c || json_object_is_type(c, json_type_null))
		return xstrdup("");
	if (json_object_is_type(c, json_type_string))
		return xstrdup(json_object_get_string(c));
	if (json_object_is_type(c, json_type_array)) {
		str_builder sb;
		sb_init(&sb);
		size_t n = json_object_array_length(c);
		for (size_t i = 0; i < n; i++) {
			json_object *part = json_object_array_get_idx(c, i);
			if (!json_object_is_type(part, json_type_object))
				continue;
			const char *ts = json_get_str(part, "type", "");
			if (*ts && strcmp(ts, "text") != 0)
				continue;
			sb_puts(&sb, json_get_str(part, "text", NULL));
		}
		return sb_finish(&sb);
	}
	return xstrdup("");
}

static const char *parse_stop(json_object *root, oa_req_params *p) {
	json_object *stop = json_get(root, "stop");
	if (!stop)
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
	p->stream		 = json_get_bool(root, "stream", false);
	p->include_usage = false;
	json_object *so	 = json_get_obj(root, "stream_options");
	if (so)
		p->include_usage = json_get_bool(so, "include_usage", false);

	double d	   = json_get_num(root, "temperature", 0.80);
	p->temperature = d < 0 ? 0 : (d > 2 ? 2 : (float)d);
	d			   = json_get_num(root, "top_p", 0.90);
	p->top_p	   = d <= 0 ? 1e-3f : (d > 1 ? 1 : (float)d);
	d			   = json_get_num(root, "min_p", 0.10);
	p->min_p	   = d < 0 ? 0 : (d > 1 ? 1 : (float)d);

	int64_t k = json_get_int(root, "top_k", 40);
	p->top_k  = k < 0 ? 0 : (int)(k > INT_MAX ? INT_MAX : k);

	int64_t mt = -1;
	mt		   = json_get_int(root, "max_completion_tokens", mt);
	if (mt < 0)
		mt = json_get_int(root, "max_tokens", -1);
	p->max_tokens = (int)(mt > 100000000 ? 100000000 : mt);

	p->has_seed = false;
	int64_t s	= json_get_int(root, "seed", -1);
	if (s >= 0) {
		p->has_seed = true;
		p->seed		= (uint64_t)s;
	}

	parse_stop(root, p);
}

static uint64_t oa_message_hash(uint64_t h, const oa_message *m) {
	h = fnv1a_update_str(h, m->role);
	h = fnv1a_update_str(h, "\x1f");
	h = fnv1a_update_str(h, m->content);
	h = fnv1a_update_str(h, m->reasoning_content ? m->reasoning_content : "");
	if (m->tool_calls)
		h = fnv1a_update_str(h, json_object_to_json_string(m->tool_calls));
	h = fnv1a_update_str(h, "\x1f");
	h = fnv1a_update_str(h, m->tool_call_id ? m->tool_call_id : "");
	h = fnv1a_update_str(h, "\x1f");
	h = fnv1a_update_str(h, m->name ? m->name : "");
	h = fnv1a_update_str(h, "\x1e");
	return h;
}

static uint64_t *session_prefix_hash_chain(const oa_req_params *p, size_t *out_len) {
	size_t	  len	= p->n_messages > 0 ? p->n_messages - 1 : 0;
	uint64_t *chain = xmalloc((len + 1) * sizeof(uint64_t));
	uint64_t  h		= FNV1A_OFFSET_BASIS;
	if (p->tools)
		h = fnv1a_update_str(h, json_object_to_json_string(p->tools));
	h		 = fnv1a_update_str(h, "\x1f");
	h		 = fnv1a_update_str(h, p->tool_choice ? p->tool_choice : "");
	h		 = fnv1a_update_str(h, p->forced_function ? p->forced_function : "");
	h		 = fnv1a_update_str(h, "\x1e");
	chain[0] = h;
	for (size_t i = 0; i < len; i++) {
		chain[i + 1] = oa_message_hash(chain[i], &p->messages[i]);
	}
	*out_len = len;
	return chain;
}

static const char *parse_tools_array(json_object *root, oa_req_params *p) {
	json_object *tools = json_get(root, "tools");
	if (!tools || json_object_is_type(tools, json_type_null))
		return NULL;
	if (!json_object_is_type(tools, json_type_array))
		return "'tools' must be an array";
	size_t n = json_object_array_length(tools);
	for (size_t i = 0; i < n; i++) {
		json_object *t	= json_object_array_get_idx(tools, i);
		json_object *fn = json_get_obj(t, "function");
		const char	*nm = fn ? json_get_str(fn, "name", "") : "";
		if (!json_object_is_type(t, json_type_object) || !fn || !*nm)
			return "'tools[]' must be {type:'function', function:{name:'<non-empty>', ...}}";
	}
	p->tools = tools;
	return NULL;
}

static const char *parse_tool_choice(json_object *root, oa_req_params *p) {
	p->tool_choice	= xstrdup("auto");
	json_object *tc = json_get(root, "tool_choice");
	if (!tc || json_object_is_type(tc, json_type_null))
		return NULL;

	if (json_object_is_type(tc, json_type_string)) {
		const char *s = json_object_get_string(tc);
		if (strcmp(s, "none") && strcmp(s, "auto") && strcmp(s, "required"))
			return "'tool_choice' must be 'none', 'auto', 'required', or an object";
		free(p->tool_choice);
		p->tool_choice = xstrdup(s);
		return NULL;
	}

	if (json_object_is_type(tc, json_type_object)) {
		const char	*t2 = json_get_str(tc, "type", "");
		json_object *f2 = json_get_obj(tc, "function");
		const char	*n2 = f2 ? json_get_str(f2, "name", NULL) : NULL;
		if (strcmp(t2, "function") || !f2 || !n2)
			return "'tool_choice' object must be {type:'function', function:{name:...}}";
		free(p->tool_choice);
		p->tool_choice	   = xstrdup("required");
		p->forced_function = xstrdup(n2);
		return NULL;
	}

	return "'tool_choice' must be a string or an object";
}

static const char *parse_tools(json_object *root, oa_req_params *p) {
	const char *err = parse_tools_array(root, p);
	if (err)
		return err;
	if (!p->tools)
		return NULL;
	return parse_tool_choice(root, p);
}

static const char *parse_chat_request(json_object *root, oa_req_params *p) {
	parse_sampling_params(root, p);

	const char *terr = parse_tools(root, p);
	if (terr)
		return terr;

	json_object *msgs = json_get_arr(root, "messages");
	if (!msgs)
		return "'messages' must be a non-empty array";
	size_t n = json_object_array_length(msgs);
	if (n == 0)
		return "'messages' must be a non-empty array";

	p->messages = xcalloc(n, sizeof(*p->messages));
	for (size_t i = 0; i < n; i++) {
		json_object *m = json_object_array_get_idx(msgs, i);
		if (!json_object_is_type(m, json_type_object))
			return "every element of 'messages' must be an object";
		p->messages[i].role	   = xstrdup(json_get_str(m, "role", "user"));
		p->messages[i].content = extract_message_content(m);

		const char *rc = json_get_str(m, "reasoning_content", NULL);
		if (rc)
			p->messages[i].reasoning_content = xstrdup(rc);

		json_object *tcs = json_get_arr(m, "tool_calls");
		if (tcs)
			p->messages[i].tool_calls = tcs;

		p->messages[i].tool_call_id = json_get_str(m, "tool_call_id", NULL);
		p->messages[i].name			= json_get_str(m, "name", NULL);
	}
	p->n_messages = n;
	return NULL;
}

static const char *parse_completion_request(json_object *root, oa_req_params *p) {
	parse_sampling_params(root, p);

	json_object *prompt = json_get(root, "prompt");
	if (!prompt)
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
		str_builder sb;
		sb_init(&sb);
		for (size_t i = 0; i < n; i++) {
			if (i)
				sb_putc(&sb, '\n');
			sb_puts(&sb, json_object_get_string(json_object_array_get_idx(prompt, i)));
		}
		p->prompt = sb_finish(&sb);
		return NULL;
	}
	return "'prompt' must be a string";
}

static void free_req_params(oa_req_params *p) {
	for (size_t i = 0; i < p->n_messages; i++) {
		free(p->messages[i].role);
		free(p->messages[i].content);
		free(p->messages[i].reasoning_content);
	}
	free(p->messages);
	free(p->prompt);
	free(p->tool_choice);
	free(p->forced_function);
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
	toolcall_scanner_free(rc->gen.tsc);
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
	bool was_in_flight = !g->generation_done;
	g->client_gone	   = true;
	pthread_cond_broadcast(&g->q_cv);
	pthread_mutex_unlock(&g->q_mtx);
	if (was_in_flight && g->st && g->st->ctx)
		g->st->ctx->interrupt = 1;
}

static void make_id(oa_gen *g) {
	uint64_t seq = atomic_fetch_add_explicit(&g->st->req_counter, 1, memory_order_relaxed);
	snprintf(g->id, sizeof(g->id), "%s-%llx-%llu", g->chat_api ? "chatcmpl" : "cmpl",
			 (unsigned long long)time(NULL), (unsigned long long)seq);
}

static const char *compute_finish_reason(const oa_gen *g) {
	if (g->stopped_by_stop)
		return "stop";
	if (g->tsc && toolcall_scanner_n_calls(g->tsc) > 0)
		return "tool_calls";
	if (g->params->max_tokens > 0 && g->generated >= g->params->max_tokens)
		return "length";
	return "stop";
}

static json_object *chunk_head(oa_gen *g, const char *object) {
	json_object *chunk = json_object_new_object();
	json_set_str(chunk, "id", g->id);
	json_set_str(chunk, "object", object);
	json_object_object_add(chunk, "created", json_object_new_int64((int64_t)time(NULL)));
	json_set_str(chunk, "model", g->st->model_id);
	return chunk;
}

static void append_usage(json_object *root, int prompt_tokens, int completion_tokens) {
	json_object *usage = json_object_new_object();
	json_set_int(usage, "prompt_tokens", prompt_tokens);
	json_set_int(usage, "completion_tokens", completion_tokens);
	json_set_int(usage, "total_tokens", prompt_tokens + completion_tokens);
	json_object_object_add(root, "usage", usage);
}

static json_object *chat_choice_delta(json_object *delta, const char *finish_reason) {
	json_object *choice = json_object_new_object();
	json_set_int(choice, "index", 0);
	json_object_object_add(choice, "delta", delta ? delta : json_object_new_object());
	json_object_object_add(choice, "logprobs", NULL);
	json_object_object_add(choice, "finish_reason",
						   finish_reason ? json_object_new_string(finish_reason) : NULL);
	return choice;
}

static json_object *text_choice_delta(const char *piece, size_t n, const char *finish_reason) {
	json_object *choice = json_object_new_object();
	json_object_object_add(choice, "text", json_object_new_string_len(piece ? piece : "", (int)n));
	json_set_int(choice, "index", 0);
	json_object_object_add(choice, "logprobs", NULL);
	json_object_object_add(choice, "finish_reason",
						   finish_reason ? json_object_new_string(finish_reason) : NULL);
	return choice;
}

static json_object *build_completion_body(req_ctx *rc, bool chat_api) {
	oa_gen		 *g	   = &rc->gen;
	openai_state *st   = rc->st;
	json_object	 *root = json_object_new_object();
	json_set_str(root, "id", g->id);
	json_set_str(root, "object", chat_api ? "chat.completion" : "text_completion");
	json_object_object_add(root, "created", json_object_new_int64((int64_t)time(NULL)));
	json_set_str(root, "model", st->model_id);

	json_object *choices = json_object_new_array();
	json_object *choice	 = json_object_new_object();
	json_set_int(choice, "index", 0);
	if (chat_api) {
		json_object *msg = json_object_new_object();
		json_set_str(msg, "role", "assistant");
		json_set_str(msg, "content", g->content.p ? g->content.p : "");
		if (g->reasoning.len > 0)
			json_set_str(msg, "reasoning_content", g->reasoning.p);
		json_object *calls = g->tsc ? toolcall_scanner_calls(g->tsc) : NULL;
		if (calls && json_object_array_length(calls) > 0)
			json_object_object_add(msg, "tool_calls", json_object_get(calls));
		json_object_object_add(choice, "message", msg);
	} else {
		json_set_str(choice, "text", g->content.p ? g->content.p : "");
	}
	json_object_object_add(choice, "logprobs", NULL);
	json_set_str(choice, "finish_reason", compute_finish_reason(g));
	json_object_array_add(choices, choice);
	json_object_object_add(root, "choices", choices);
	append_usage(root, g->prompt_tokens, g->generated);
	return root;
}

static json_object *health_body(void) {
	json_object *root = json_object_new_object();
	json_set_str(root, "status", "ok");
	return root;
}

static json_object *root_body(openai_state *st) {
	json_object *root = json_object_new_object();
	json_set_str(root, "status", "ok");
	json_set_str(root, "model", st->model_id);
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
	json_set_str(root, "object", "list");
	json_object *list = json_object_new_array();
	json_object *m	  = json_object_new_object();
	json_set_str(m, "id", st->model_id);
	json_set_str(m, "object", "model");
	json_object_object_add(m, "created", json_object_new_int64((int64_t)st->created_at));
	json_set_str(m, "owned_by", "kappai");
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
	json_set_str(delta, "role", "assistant");
	json_set_str(delta, "content", "");
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

static void sse_send_tool_call(oa_gen *g, int index, const char *id, const char *name,
							   const char *args_json) {
	if (!g->sent_any_chunk) {
		g->sent_any_chunk = true;
		if (g->chat_api)
			sse_send_role_chunk(g);
		else
			sse_text_delta(g, "", 0, NULL);
	}
	if (g->client_gone || shutting_down(g->st))
		return;

	json_object *tcd = json_object_new_object();
	json_set_int(tcd, "index", index);
	json_set_str(tcd, "id", id);
	json_set_str(tcd, "type", "function");
	json_object *fn = json_object_new_object();
	json_set_str(fn, "name", name);
	json_set_str(fn, "arguments", args_json);
	json_object_object_add(tcd, "function", fn);

	json_object *arr = json_object_new_array();
	json_object_array_add(arr, tcd);
	json_object *delta = json_object_new_object();
	json_object_object_add(delta, "tool_calls", arr);
	sse_chat_delta(g, delta, NULL);
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

static void tool_on_content(void *ud, const char *piece, size_t n) {
	oa_gen *g = (oa_gen *)ud;
	if (g->streaming && !g->client_gone && !shutting_down(g->st))
		sse_send_delta(g, false, piece, n);
}

static void tool_on_call(void *ud, int index, const char *id, const char *name,
						 const char *args_json) {
	oa_gen *g = (oa_gen *)ud;
	if (g->streaming)
		sse_send_tool_call(g, index, id, name, args_json);
}

static void gen_on_token(int32_t id, const char *piece, int n, void *ud) {
	oa_gen *g = (oa_gen *)ud;

	if (g->client_gone || shutting_down(g->st)) {
		g->st->ctx->interrupt = 1;
		return;
	}

	if (!g->chat_api) {
		toolcall_buf_append(&g->content, piece, (size_t)n);
		check_stop_sequences(g);
		if (g->stopped_by_stop) {
			g->st->ctx->interrupt = 1;
			return;
		}
		if (g->streaming)
			sse_send_delta(g, false, piece, (size_t)n);
		return;
	}

	if (g->tsc && toolcall_scanner_suppressed(g->tsc)) {
		g->st->ctx->interrupt = 1;
		return;
	}

	if (g->tsc && toolcall_scanner_in_capture(g->tsc)) {
		toolcall_scanner_feed_capture(g->tsc, piece, (size_t)n);
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

	toolcall_buf *dst = g->in_thinking ? &g->reasoning : &g->content;
	toolcall_buf_append(dst, piece, (size_t)n);
	check_stop_sequences(g);
	if (g->stopped_by_stop) {
		g->st->ctx->interrupt = 1;
		return;
	}

	if (g->tsc && dst == &g->content) {
		toolcall_scanner_feed(g->tsc);
		if (toolcall_scanner_suppressed(g->tsc)) {
			if (!g->tool_fill_logged) {
				g->tool_fill_logged = true;
				WARN("model filled in the tool response slot itself; suppressed "
					 "the rest of the generation");
			}
			g->st->ctx->interrupt = 1;
			return;
		}
	} else if (g->streaming) {
		sse_send_delta(g, g->in_thinking, piece, (size_t)n);
	}
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

static chat_message oa_message_to_view(const oa_message *m) {
	chat_message cm = {
		.role			   = m->role,
		.content		   = m->content,
		.reasoning_content = m->reasoning_content,
		.tool_calls		   = m->tool_calls,
		.tool_call_id	   = (char *)m->tool_call_id,
		.name			   = (char *)m->name,
	};
	return cm;
}

static void run_generation(req_ctx *rc, bool chat_api) {
	openai_state *st = rc->st;
	oa_gen		 *g	 = &rc->gen;
	context		 *c	 = st->ctx;

	g->streaming = rc->params.stream;
	g->chat_api	 = chat_api;
	make_id(g);

	g->in_thinking		= false;
	g->first_token		= true;
	g->skip_label		= false;
	g->stopped_by_stop	= false;
	g->sent_any_chunk	= false;
	g->tool_fill_logged = false;
	g->generation_done	= false;
	g->prompt_tokens	= 0;
	g->generated		= 0;
	toolcall_buf_reset(&g->content);
	toolcall_buf_reset(&g->reasoning);

	sampler_params sp = {.temperature	 = rc->params.temperature,
						 .top_k			 = rc->params.top_k,
						 .top_p			 = rc->params.top_p,
						 .min_p			 = rc->params.min_p,
						 .repeat_penalty = 1.0f,
						 .repeat_last_n	 = 0};

	pthread_mutex_lock(&st->gen_mtx);
	{
		if (rc->params.has_seed) {
			sampler_free(&c->samp);
			sampler_init(&c->samp, rc->params.seed);
			sampler_set_vocab(&c->samp, c->m.vocab_size);
		}
		if (!g->same_session || c->session_poisoned)
			context_reset(c);

		if (chat_api) {
			chat_template_set_tools(&c->chat, rc->params.tools, rc->params.tool_choice);
			if (rc->params.forced_function)
				INFO("tool_choice: function '%s' requested; running in best-effort "
					 "mode (no grammar forcing)",
					 rc->params.forced_function);

			g->tsc = (c->chat.tools && c->chat.tool_fmt)
						 ? toolcall_scanner_new(c->chat.tool_fmt, &g->content, tool_on_content,
												tool_on_call, g)
						 : NULL;

			size_t start = c->chat.n_messages;
			for (size_t i = start; i + 1 < rc->params.n_messages; i++) {
				chat_message cm = oa_message_to_view(&rc->params.messages[i]);
				chat_template_add_message_ex(&c->chat, &cm);
			}
			{
				chat_message cm =
					oa_message_to_view(&rc->params.messages[rc->params.n_messages - 1]);
				g->generated = context_chat_turn_msg(c, &cm, true, rc->params.max_tokens, &sp,
													 gen_on_token, g, "");
			}

			if (g->tsc)
				toolcall_scanner_finish(g->tsc);

			json_object *calls		= g->tsc ? toolcall_scanner_calls(g->tsc) : NULL;
			bool		 suppressed = g->tsc && toolcall_scanner_suppressed(g->tsc);
			if ((calls && json_object_array_length(calls) > 0) || suppressed)
				chat_template_rewrite_last_assistant(&c->chat, g->content.p ? g->content.p : "",
													 g->reasoning.p ? g->reasoning.p : "", calls);
		} else {
			g->generated = context_completion(c, rc->params.prompt, rc->params.max_tokens, &sp,
											  gen_on_token, g);
		}
		c->interrupt	 = 0;
		g->prompt_tokens = c->last_prompt_tokens;
		pthread_mutex_lock(&g->q_mtx);
		g->generation_done = true;
		pthread_mutex_unlock(&g->q_mtx);
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
	req_ctx *rc = *(req_ctx **)con_cls;
	if (!rc)
		return;
	DEBUG("connection closed (termination_code=%d) for session=%llx interrupt_before=%d", (int)toe,
		  (unsigned long long)rc->gen.session_key_at_start,
		  (int)(rc->st->ctx ? rc->st->ctx->interrupt : -1));
	gen_mark_client_gone(&rc->gen);
	DEBUG("connection closed: interrupt_after=%d",
		  (int)(rc->st->ctx ? rc->st->ctx->interrupt : -1));
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

	DEBUG("request body [%s] (%zu bytes): %.4096s", url, rc->body_len,
		  rc->body ? rc->body : "(empty)");

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

	if (chat_api) {
		size_t	  new_len;
		uint64_t *new_chain = session_prefix_hash_chain(&rc->params, &new_len);
		uint64_t  key		= new_chain[new_len];

		pthread_mutex_lock(&st->gen_mtx);
		bool matched = st->has_session && new_len >= st->prefix_len &&
					   new_chain[st->prefix_len] == st->prefix_hashes[st->prefix_len];
		if (st->has_session && !matched)
			WARN("session=%016llx new session detected (previous session's history no "
				 "longer matches; previous_len=%zu new_len=%zu) -- previous session is "
				 "now broken; n_messages=%zu",
				 (unsigned long long)key, st->prefix_len, new_len, rc->params.n_messages);
		else
			INFO("session=%016llx %s; n_messages=%zu", (unsigned long long)key,
				 matched ? "continuing existing session" : "starting first session",
				 rc->params.n_messages);
		rc->gen.same_session		 = matched;
		rc->gen.session_key_at_start = key;
		st->has_session				 = true;
		free(st->prefix_hashes);
		st->prefix_hashes = new_chain;
		st->prefix_len	  = new_len;
		pthread_mutex_unlock(&st->gen_mtx);
	}

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

static enum MHD_Result route_health(openai_state *st, struct MHD_Connection *conn, req_ctx *rc,
									const char *url, const char *method) {
	(void)st;
	(void)rc;
	(void)url;
	(void)method;
	return respond_json(conn, MHD_HTTP_OK, health_body());
}

static enum MHD_Result route_root(openai_state *st, struct MHD_Connection *conn, req_ctx *rc,
								  const char *url, const char *method) {
	(void)rc;
	(void)url;
	(void)method;
	return respond_json(conn, MHD_HTTP_OK, root_body(st));
}

static enum MHD_Result route_completions(openai_state *st, struct MHD_Connection *conn, req_ctx *rc,
										 const char *url, const char *method) {
	if (rc->too_large)
		return respond_json(conn, MHD_HTTP_PAYLOAD_TOO_LARGE,
							error_body("request body too large", "invalid_request_error"));
	uint64_t		t0 = time_us();
	enum MHD_Result r  = handle_post(st, conn, rc, url);
	INFO("%s %s (%.1f ms)", method, url, (double)(time_us() - t0) / 1000.0);
	return r;
}

static enum MHD_Result route_models(openai_state *st, struct MHD_Connection *conn, req_ctx *rc,
									const char *url, const char *method) {
	(void)rc;
	(void)url;
	(void)method;
	if (!check_auth(st, conn))
		return respond_json(conn, MHD_HTTP_UNAUTHORIZED,
							error_body("Invalid API key", "authentication_error"));
	return respond_json(conn, MHD_HTTP_OK, models_body(st));
}

typedef enum MHD_Result (*route_fn)(openai_state *st, struct MHD_Connection *conn, req_ctx *rc,
									const char *url, const char *method);

typedef struct {
	const char *method;
	const char *path;
	route_fn	fn;
} route;

static const route routes[] = {
	{"GET", "/health", route_health},
	{"GET", "/", route_root},
	{"POST", "/v1/chat/completions", route_completions},
	{"POST", "/v1/completions", route_completions},
	{"GET", "/v1/models", route_models},
};

static enum MHD_Result handle_request(void *cls, struct MHD_Connection *conn, const char *url,
									  const char *method, const char *version,
									  const char *upload_data, size_t *upload_data_size,
									  void **con_cls) {
	openai_state *st = (openai_state *)cls;
	(void)version;

	if (*con_cls == NULL) {
		*con_cls = rc_new(st);
		DEBUG("connection opened: %s %s", method, url);
		return MHD_YES;
	}
	req_ctx *rc = *(req_ctx **)con_cls;

	if (*upload_data_size > 0) {
		if (rc->body_len + *upload_data_size > MAX_BODY_BYTES) {
			rc->too_large = true;
		} else {
			if (rc->body_len + *upload_data_size + 1 > rc->body_cap) {
				size_t need	 = rc->body_len + *upload_data_size + 1;
				size_t want	 = need * 2 > MAX_BODY_BYTES + 1 ? MAX_BODY_BYTES + 1 : need * 2;
				rc->body_cap = want > need ? want : need;
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

	for (size_t i = 0; i < sizeof(routes) / sizeof(routes[0]); i++) {
		if (strcmp(method, routes[i].method) != 0)
			continue;
		if (strcmp(url, routes[i].path) != 0)
			continue;
		return routes[i].fn(st, conn, rc, url, method);
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
	if (st->ctx)
		st->ctx->interrupt = 1;
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
	free(st->prefix_hashes);
	pthread_mutex_destroy(&st->gen_mtx);
	free(st);
}
