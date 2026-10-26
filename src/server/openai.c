#include "openai.h"

#include "common.h"
#include "engine.h"
#include "json-c/json.h"
#include "json_helpers.h"
#include "log.h"
#include "memconfig.h"
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
#include <sys/mman.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#define MAX_BODY_BYTES (32 * 1024 * 1024)
#define SSE_POLL_SEC 1
#define TASK_STACK_BYTES ((size_t)16 << 20)

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

	size_t		 n_stop;
	const char **stop_needles;
	size_t		*stop_lens;
	size_t		 stop_max;

	chat_message *messages;
	size_t		  n_messages;

	json_object *tools;
	char		*tool_choice;
	char		*forced_function;

	char *prompt;
} oa_req_params;

typedef struct {
	char  *data;
	size_t len;
} sse_frame;

typedef struct {
	openai_state	  *st;
	_Atomic(context *) sctx;
	oa_req_params	  *params;

	bool		streaming;
	bool		chat_api;
	bool		same_session;
	uint64_t	session_key_at_start;
	atomic_bool generation_done;

	pthread_mutex_t q_mtx;
	pthread_cond_t	q_cv;
	sse_frame	   *queue;
	size_t			q_len;
	size_t			q_cap;
	size_t			q_off;
	bool			producer_done;
	atomic_bool		client_gone;

	str_builder content;
	str_builder reasoning;

	char  *stop_window;
	size_t stop_window_cap;

	think_filter think;

	bool stopped_by_stop;
	bool sent_any_chunk;
	bool tool_fill_logged;

	toolcall_scanner *tsc;

	int	 generated;
	int	 prompt_tokens;
	char id[64];
} oa_gen;

typedef struct req_ctx req_ctx;
typedef struct oa_slot oa_slot;
typedef struct oa_job  oa_job;

struct req_ctx {
	openai_state *st;
	atomic_int	  refs;

	str_builder body;
	bool		too_large;

	oa_req_params params;
	json_object	 *body_json;
	oa_gen		  gen;
};

struct oa_job {
	oa_job	 *next;
	req_ctx	 *rc;
	bool	  chat_api;
	uint64_t *chain;
	size_t	  chain_len;
	uint64_t  session_key;
	uint64_t  t_start_us;
};

struct oa_slot {
	context	 sess;
	bool	 owns;
	bool	 used;
	bool	 busy;
	uint64_t full_hash;
	size_t	 n_msgs;
	uint64_t last_ns;
};

typedef struct {
	openai_state *st;
	oa_job		 *job;
	oa_slot		 *slot;
	ucontext_t	  uc;
	void		 *stack;
	bool		  done;

	bool		decode_pending;
	int32_t		decode_token;
	int			decode_pos;
	float	   *decode_logits;
	status_code decode_status;
} oa_task;

typedef struct {
	oa_task **tasks;
	int32_t	 *tokens;
	int		 *positions;
	kvcache **caches;
	float	 *logits;
	size_t	  logits_cap;
	int		  n;
} oa_decode_batch;

struct openai_state {
	context *ctx;

	char *model_id;
	char *api_key;

	long		  created_at;
	atomic_ullong req_counter;

	int		 slots_cap;
	int		 n_slots;
	oa_slot *slots;

	pthread_mutex_t sch_mtx;
	pthread_cond_t	sch_cv;

	oa_job *q_head;
	oa_job *q_tail;

	pthread_t worker;

	ucontext_t		sched_uc;
	oa_task		  **tasks;
	int				n_tasks;
	oa_decode_batch decode;

	struct MHD_Daemon *daemon;
	struct sockaddr	  *bind_addr;
	atomic_bool		   shutting_down;
};

static sigset_t g_signal_mask;

typedef enum MHD_Result (*route_fn)(openai_state *st, struct MHD_Connection *conn, req_ctx *rc,
									const char *url, const char *method);

typedef struct {
	const char *method;
	const char *path;
	route_fn	fn;
} route;

static oa_task *task_entering;

static char *derive_model_id(const char *model_path) {
	const char *base = path_basename(model_path);
	size_t		len	 = strlen(base);
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

static inline bool gen_dead(const oa_gen *g) {
	return atomic_load_explicit(&g->client_gone, memory_order_acquire) || shutting_down(g->st);
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
		json_arr_iter it = json_arr_begin(c);
		json_object	 *part;
		while (json_arr_next(&it, &part)) {
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
		size_t l = (size_t)json_object_get_string_len(stop);
		if (l == 0)
			return NULL;
		p->stop_needles	   = xmalloc(sizeof(*p->stop_needles));
		p->stop_lens	   = xmalloc(sizeof(*p->stop_lens));
		p->stop_needles[0] = json_object_get_string(stop);
		p->stop_lens[0]	   = l;
		p->stop_max		   = l;
		p->n_stop		   = 1;
		return NULL;
	}
	if (json_object_is_type(stop, json_type_array)) {
		json_arr_iter it = json_arr_begin(stop);
		json_object	 *item;
		size_t		  cap = 0;
		size_t		  n	  = 0;
		while (json_arr_next(&it, &item)) {
			if (!json_object_is_type(item, json_type_string))
				continue;
			size_t l = (size_t)json_object_get_string_len(item);
			if (l == 0)
				continue;
			if (n >= cap) {
				cap				= cap ? cap * 2 : 4;
				p->stop_needles = xrealloc(p->stop_needles, cap * sizeof(*p->stop_needles));
				p->stop_lens	= xrealloc(p->stop_lens, cap * sizeof(*p->stop_lens));
			}
			p->stop_needles[n] = json_object_get_string(item);
			p->stop_lens[n]	   = l;
			if (l > p->stop_max)
				p->stop_max = l;
			n++;
		}
		p->n_stop = n;
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

static uint64_t oa_message_hash(uint64_t h, const chat_message *m) {
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
	size_t	  len	= p->n_messages;
	uint64_t *chain = xmalloc((len + 1) * sizeof(uint64_t));
	uint64_t  h		= FNV1A_OFFSET_BASIS;
	if (p->tools)
		h = fnv1a_update_str(h, json_object_to_json_string(p->tools));
	h		 = fnv1a_update_str(h, "\x1f");
	h		 = fnv1a_update_str(h, p->tool_choice ? p->tool_choice : "");
	h		 = fnv1a_update_str(h, p->forced_function ? p->forced_function : "");
	h		 = fnv1a_update_str(h, "\x1e");
	chain[0] = h;
	for (size_t i = 0; i < len; i++)
		chain[i + 1] = oa_message_hash(chain[i], &p->messages[i]);
	*out_len = len;
	return chain;
}

static const char *parse_tools_array(json_object *root, oa_req_params *p) {
	json_object *tools = json_get(root, "tools");
	if (!tools || json_object_is_type(tools, json_type_null))
		return NULL;
	if (!json_object_is_type(tools, json_type_array))
		return "'tools' must be an array";
	json_arr_iter it = json_arr_begin(tools);
	json_object	 *t;
	while (json_arr_next(&it, &t)) {
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

		const char *tcid			= json_get_str(m, "tool_call_id", NULL);
		const char *nm				= json_get_str(m, "name", NULL);
		p->messages[i].tool_call_id = tcid ? xstrdup(tcid) : NULL;
		p->messages[i].name			= nm ? xstrdup(nm) : NULL;
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
		free(p->messages[i].tool_call_id);
		free(p->messages[i].name);
	}
	free(p->messages);
	free(p->prompt);
	free(p->tool_choice);
	free(p->forced_function);
	free(p->stop_needles);
	free(p->stop_lens);
	memset(p, 0, sizeof(*p));
}

static void rc_ref(req_ctx *rc) {
	atomic_fetch_add_explicit(&rc->refs, 1, memory_order_relaxed);
}

static void rc_unref(req_ctx *rc) {
	if (!rc || atomic_fetch_sub_explicit(&rc->refs, 1, memory_order_acq_rel) != 1)
		return;
	sb_free(&rc->body);
	json_object_put(rc->body_json);
	for (size_t i = 0; i < rc->gen.q_len; i++)
		free(rc->gen.queue[i].data);
	free(rc->gen.queue);
	free(rc->gen.content.p);
	free(rc->gen.reasoning.p);
	free(rc->gen.stop_window);
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
	if (!gen_dead(g)) {
		ARR_RESERVE(g->queue, g->q_len, g->q_cap);
		char *frame = xmalloc(6 + len + 2);
		memcpy(frame, "data: ", 6);
		memcpy(frame + 6, payload, len);
		frame[6 + len]			= '\n';
		frame[6 + len + 1]		= '\n';
		g->queue[g->q_len].data = frame;
		g->queue[g->q_len].len	= len + 8;
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
	atomic_store_explicit(&g->client_gone, true, memory_order_release);
	pthread_cond_broadcast(&g->q_cv);
	pthread_mutex_unlock(&g->q_mtx);
	if (was_in_flight && g->sctx)
		g->sctx->interrupt = 1;
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

static json_object *stream_choice(const char *payload_key, json_object *payload,
								  const char *finish_reason) {
	json_object *choice = json_object_new_object();
	json_set_int(choice, "index", 0);
	json_object_object_add(choice, payload_key, payload ? payload : json_object_new_object());
	json_object_object_add(choice, "logprobs", NULL);
	json_object_object_add(choice, "finish_reason",
						   finish_reason ? json_object_new_string(finish_reason) : NULL);
	return choice;
}

static json_object *chat_choice_delta(json_object *delta, const char *finish_reason) {
	return stream_choice("delta", delta, finish_reason);
}

static json_object *text_choice_delta(const char *piece, size_t n, const char *finish_reason) {
	return stream_choice("text", json_object_new_string_len(piece ? piece : "", (int)n),
						 finish_reason);
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

static void sse_ensure_preamble(oa_gen *g) {
	if (g->sent_any_chunk)
		return;
	g->sent_any_chunk = true;
	if (g->chat_api)
		sse_send_role_chunk(g);
	else
		sse_text_delta(g, "", 0, NULL);
}

static void sse_send_delta(oa_gen *g, bool reasoning, const char *piece, size_t n) {
	sse_ensure_preamble(g);
	if (gen_dead(g))
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
	sse_ensure_preamble(g);
	if (gen_dead(g))
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

	size_t max_needle = g->params->stop_max;
	if (max_needle == 0 || total < max_needle)
		return;

	size_t window_len	= 2 * max_needle - 1;
	size_t window_start = total > window_len ? total - window_len : 0;
	size_t wlen			= total - window_start;
	ARR_ENSURE(g->stop_window, wlen + 1, g->stop_window_cap);
	char *window = g->stop_window;
	if (window_start < rlen) {
		memcpy(window, g->reasoning.p + window_start, rlen - window_start);
		memcpy(window + (rlen - window_start), g->content.p, clen);
	} else {
		memcpy(window, g->content.p + (window_start - rlen), wlen);
	}
	window[wlen] = '\0';

	for (size_t i = 0; i < g->params->n_stop && !g->stopped_by_stop; i++) {
		const char *needle = g->params->stop_needles[i];
		const char *hit	   = strstr(window, needle);
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
}

static void tool_on_content(void *ud, const char *piece, size_t n) {
	oa_gen *g = (oa_gen *)ud;
	if (g->streaming && !gen_dead(g))
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

	if (gen_dead(g)) {
		g->sctx->interrupt = 1;
		return;
	}

	if (!g->chat_api) {
		sb_putb(&g->content, piece, (size_t)n);
		check_stop_sequences(g);
		if (g->stopped_by_stop) {
			g->sctx->interrupt = 1;
			return;
		}
		if (g->streaming)
			sse_send_delta(g, false, piece, (size_t)n);
		return;
	}

	if (g->tsc && toolcall_scanner_suppressed(g->tsc)) {
		g->sctx->interrupt = 1;
		return;
	}

	if (g->tsc && toolcall_scanner_in_capture(g->tsc)) {
		toolcall_scanner_feed_capture(g->tsc, piece, (size_t)n);
		return;
	}

	think_filter_event tev =
		think_filter_feed(&g->think, id, g->sctx->chat.think_start_id, g->sctx->chat.think_end_id,
						  g->sctx->chat.think_open, g->sctx->chat.think_label_line, &piece, &n);
	if (tev != THINK_EMIT)
		return;
	if (n <= 0)
		return;

	str_builder *dst = g->think.in_thinking ? &g->reasoning : &g->content;
	sb_putb(dst, piece, (size_t)n);
	check_stop_sequences(g);
	if (g->stopped_by_stop) {
		g->sctx->interrupt = 1;
		return;
	}

	if (g->tsc && dst == &g->content) {
		toolcall_scanner_feed(g->tsc);
		if (toolcall_scanner_suppressed(g->tsc)) {
			if (!g->tool_fill_logged) {
				g->tool_fill_logged = true;
				DEBUG("tool response marker reached; suppressed the rest of the generation");
			}
			g->sctx->interrupt = 1;
			return;
		}
	} else if (g->streaming) {
		sse_send_delta(g, g->think.in_thinking, piece, (size_t)n);
	}
}

static void sse_finish_stream(oa_gen *g) {
	const char *finish = compute_finish_reason(g);

	if (!gen_dead(g)) {
		if (g->chat_api) {
			sse_ensure_preamble(g);
			sse_chat_delta(g, NULL, finish);
		} else {
			sse_ensure_preamble(g);
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

static void run_generation(req_ctx *rc, oa_slot *slot) {
	oa_gen	*g		  = &rc->gen;
	context *c		  = &slot->sess;
	bool	 chat_api = rc->gen.chat_api;
	g->sctx			  = c;

	if (gen_dead(g)) {
		pthread_mutex_lock(&g->q_mtx);
		g->generation_done = true;
		pthread_cond_broadcast(&g->q_cv);
		pthread_mutex_unlock(&g->q_mtx);
		sse_finish_stream(g);
		return;
	}

	g->streaming = rc->params.stream;
	g->chat_api	 = chat_api;
	make_id(g);

	g->think.in_thinking = false;
	g->think.first_token = true;
	g->think.skip_label	 = false;
	g->stopped_by_stop	 = false;
	g->sent_any_chunk	 = false;
	g->tool_fill_logged	 = false;
	g->generation_done	 = false;
	g->prompt_tokens	 = 0;
	g->generated		 = 0;
	sb_reset(&g->content);
	sb_reset(&g->reasoning);

	sampler_params sp = {.temperature	 = rc->params.temperature,
						 .top_k			 = rc->params.top_k,
						 .top_p			 = rc->params.top_p,
						 .min_p			 = rc->params.min_p,
						 .repeat_penalty = 1.0f,
						 .repeat_last_n	 = 0};

	{
		c->interrupt = 0;
		if (rc->params.has_seed) {
			sampler_free(&c->samp);
			sampler_init(&c->samp, rc->params.seed);
			sampler_set_vocab(&c->samp, c->m.vocab_size);
		}
		if (!g->same_session || c->session_poisoned) {
			if (c->session_poisoned || c->kv.hybrid)
				context_reset(c);
			else
				context_reset_session(c);
		}

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
				chat_template_add_message_ex(&c->chat, &rc->params.messages[i]);
			}
			{
				g->generated =
					context_chat_turn_msg(c, &rc->params.messages[rc->params.n_messages - 1], true,
										  rc->params.max_tokens, &sp, gen_on_token, g, "");
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
		pthread_cond_broadcast(&g->q_cv);
		pthread_mutex_unlock(&g->q_mtx);
	}
	backend_release_thread(c->backend);

	if (g->streaming)
		sse_finish_stream(g);
}

static ssize_t sse_read_cb(void *cls, uint64_t pos, char *buf, size_t max) {
	req_ctx *rc = (req_ctx *)cls;
	oa_gen	*g	= &rc->gen;
	(void)pos;

	pthread_mutex_lock(&g->q_mtx);
	while (g->q_len == 0 && !g->producer_done && !gen_dead(g)) {
		struct timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);
		ts.tv_sec += SSE_POLL_SEC;
		pthread_cond_timedwait(&g->q_cv, &g->q_mtx, &ts);
	}

	ssize_t out;
	if (g->q_len > 0) {
		char  *f	  = g->queue[0].data;
		size_t remain = g->queue[0].len - g->q_off;
		size_t n	  = remain < max ? remain : max;
		memcpy(buf, f + g->q_off, n);
		g->q_off += n;
		if (g->q_off == g->queue[0].len) {
			free(f);
			memmove(g->queue, g->queue + 1, (g->q_len - 1) * sizeof(*g->queue));
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
		  (int)(rc->gen.sctx ? rc->gen.sctx->interrupt : -1));
	gen_mark_client_gone(&rc->gen);
	DEBUG("connection closed: interrupt_after=%d",
		  (int)(rc->gen.sctx ? rc->gen.sctx->interrupt : -1));
	*(req_ctx **)con_cls = NULL;
	rc_unref(rc);
}

static int32_t slot_token_common(const context *c, const int32_t *ids, int n) {
	int32_t common_max = (int32_t)MIN(c->fed_ids.n, n);
	int32_t reuse	   = 0;
	while (reuse < common_max && c->fed_ids.p[reuse] == ids[reuse])
		reuse++;
	return reuse;
}

static void slot_warm(context *c, bool warmup) {
	engine_prepare_system(c, "", warmup);
	context_reset_session(c);
	if (c->session_poisoned)
		context_reset(c);
}

static oa_slot *slot_create(openai_state *st) {
	if (st->n_slots >= st->slots_cap)
		return NULL;

	backend		 *kv_owner = st->ctx->backend->kv_alloc ? st->ctx->backend : backend_host();
	kv_quant_type kv_quant = st->ctx->kv.kv_quant;
	size_t		  kv_bytes = model_kv_cache_bytes_quant(&st->ctx->m, st->ctx->n_ctx, kv_quant,
														backend_has_cap(kv_owner, BCAP_KV_POS_CAP));
	size_t		  avail	   = mem_topology_pool_available(&st->ctx->topo, kv_owner);
	if (!config_get()->disable_failsafes && avail > 0 && kv_bytes > avail) {
		ERROR("session slot KV cache (ctx=%d, %.1f MB) exceeds what is free on backend "
			  "'%s' (%.1f MB); refusing to allocate slot",
			  st->ctx->n_ctx, kv_bytes / (1024.0 * 1024.0), kv_owner->name,
			  avail / (1024.0 * 1024.0));
		return NULL;
	}
	oa_slot *s = &st->slots[st->n_slots];
	memset(s, 0, sizeof(*s));
	s->owns = true;
	if (context_slot_init(&s->sess, st->ctx) != OK) {
		ERROR("server: failed to allocate session slot (KV cache too large?)");
		context_slot_free(&s->sess);
		return NULL;
	}
	s->used	   = true;
	s->last_ns = time_us();
	st->n_slots++;
	INFO("server: session slot %d allocated", st->n_slots);
	return s;
}

static oa_slot *slot_pick_chat(openai_state *st, uint64_t *chain, size_t chain_len,
							   bool *matched_out) {
	oa_slot *best		  = NULL;
	oa_slot *volatile lru = NULL;
	bool   best_match	  = false;
	size_t best_reuse	  = 0;
	for (int i = 0; i < st->n_slots; i++) {
		oa_slot *volatile s = &st->slots[i];
		if (s->busy)
			continue;
		if (!lru || s->last_ns < lru->last_ns)
			lru = s;
		bool   matched = s->n_msgs <= chain_len && chain[s->n_msgs] == s->full_hash;
		size_t reuse   = matched ? s->n_msgs : 0;
		if (!best || reuse > best_reuse) {
			best	   = s;
			best_reuse = reuse;
			best_match = matched;
		}
		if (matched)
			break;
	}
	if (best_match) {
		*matched_out = true;
		return best;
	}
	*matched_out			= false;
	oa_slot *volatile fresh = slot_create(st);
	return fresh ? fresh : lru;
}

static oa_slot *slot_pick_text(openai_state *st, const int32_t *ids, int n, bool *matched_out) {
	oa_slot *volatile best = NULL;
	oa_slot *volatile lru  = NULL;
	int32_t best_reuse	   = -1;
	for (int i = 0; i < st->n_slots; i++) {
		oa_slot *volatile s = &st->slots[i];
		if (s->busy)
			continue;
		if (!lru || s->last_ns < lru->last_ns)
			lru = s;
		int32_t reuse = slot_token_common(&s->sess, ids, n);
		if (reuse > best_reuse) {
			best	   = s;
			best_reuse = reuse;
		}
	}
	*matched_out = best_reuse > 0;
	if (best_reuse > 0)
		return best;
	oa_slot *volatile fresh = slot_create(st);
	return fresh ? fresh : lru;
}

static void slot_record_catalog(openai_state *st, oa_slot *s) {
	(void)st;
	s->last_ns = time_ns();
	s->busy	   = false;
}

static void job_free(openai_state *st, oa_job *j) {
	(void)st;
	if (!j)
		return;
	INFO("job_free j=%p chain=%p", (void *)j, (void *)j->chain);
	if (j->chain)
		free(j->chain);
	rc_unref(j->rc);
	free(j);
}

static void slot_adopt_chat_chain(oa_slot *s, oa_job *j) {
	uint64_t				   h   = j->chain[0];
	const chat_template_state *cts = &s->sess.chat;
	for (size_t i = 0; i < cts->n_messages; i++)
		h = oa_message_hash(h, &cts->messages[i]);
	s->full_hash = h;
	free(j->chain);
	j->chain  = NULL;
	s->n_msgs = cts->n_messages;
}

static void fail_job(openai_state *st, oa_job *j) {
	oa_gen *g	 = &j->rc->gen;
	g->generated = -1;
	pthread_mutex_lock(&g->q_mtx);
	g->generation_done = true;
	pthread_cond_broadcast(&g->q_cv);
	pthread_mutex_unlock(&g->q_mtx);
	sse_finish_stream(g);
	job_free(st, j);
}

static oa_slot *job_pick_slot(openai_state *st, oa_job *j, bool *matched) {
	req_ctx *rc = j->rc;
	if (j->chat_api)
		return slot_pick_chat(st, j->chain, j->chain_len, matched);

	int		 cap = st->ctx->n_ctx;
	int32_t *ids = xmalloc(sizeof(int32_t) * (size_t)(cap + 2));
	int		 n	 = 0;
	if (st->ctx->tok.add_bos && st->ctx->tok.bos_id >= 0)
		ids[n++] = st->ctx->tok.bos_id;
	int r = tokenizer_encode_with_specials(&st->ctx->tok, rc->params.prompt, 0, ids + n, cap - n,
										   &st->ctx->scratch.prof);
	if (r > 0)
		n += r;
	oa_slot *slot = slot_pick_text(st, ids, n, matched);
	free(ids);
	return slot;
}

static void task_yield(void *ud) {
	oa_task *t = ud;
	swapcontext(&t->uc, &t->st->sched_uc);
}

static status_code task_decode_step(void *ud, int32_t token, int pos, float *logits_out) {
	oa_task *t		  = ud;
	t->decode_token	  = token;
	t->decode_pos	  = pos;
	t->decode_logits  = logits_out;
	t->decode_pending = true;
	swapcontext(&t->uc, &t->st->sched_uc);
	return t->decode_status;
}

static void decode_batch_reserve(openai_state *st) {
	oa_decode_batch *b = &st->decode;
	if (!b->tasks) {
		b->tasks	 = xcalloc((size_t)st->slots_cap, sizeof(*b->tasks));
		b->tokens	 = xcalloc((size_t)st->slots_cap, sizeof(*b->tokens));
		b->positions = xcalloc((size_t)st->slots_cap, sizeof(*b->positions));
		b->caches	 = xcalloc((size_t)st->slots_cap, sizeof(*b->caches));
		compute_scratch_reserve_rows(&st->slots[0].sess.m, &st->slots[0].sess.scratch,
									 st->slots_cap);
	}
	size_t need = (size_t)st->slots_cap * (size_t)st->ctx->m.vocab_size;
	if (b->logits_cap < need) {
		b->logits	  = xrealloc(b->logits, need * sizeof(float));
		b->logits_cap = need;
	}
}

static void decode_batch_collect(openai_state *st) {
	oa_decode_batch *b = &st->decode;
	b->n			   = 0;
	for (int i = 0; i < st->n_tasks; i++) {
		oa_task *t = st->tasks[i];
		if (!t->decode_pending)
			continue;
		t->decode_pending = false;
		context *c		  = &t->slot->sess;
		if (c->interrupt) {
			t->decode_status = ERR_INTERRUPTED;
			continue;
		}
		b->tasks[b->n]	   = t;
		b->tokens[b->n]	   = t->decode_token;
		b->positions[b->n] = t->decode_pos;
		b->caches[b->n]	   = &c->kv;
		b->n++;
	}
}

static void run_decode_step(openai_state *st) {
	oa_decode_batch *b = &st->decode;
	decode_batch_reserve(st);
	decode_batch_collect(st);
	if (b->n == 0)
		return;

	context	   *lead = &st->slots[0].sess;
	status_code batch_status =
		compute_forward_decode_batch(&lead->m, b->caches, &lead->scratch, b->tokens, b->positions,
									 NULL, b->n, lead->flash_attn, b->logits);

	const size_t vocab = (size_t)st->ctx->m.vocab_size;
	for (int i = 0; i < b->n; i++) {
		oa_task *t		 = b->tasks[i];
		context *c		 = &t->slot->sess;
		t->decode_status = batch_status;
		if (batch_status != OK)
			continue;
		memcpy(t->decode_logits ? t->decode_logits : c->scratch.logits_host,
			   b->logits + (size_t)i * vocab, vocab * sizeof(float));
		c->logits_on_host = true;
	}
}

static void task_main(void) {
	oa_task		 *t	 = task_entering;
	openai_state *st = t->st;
	oa_job		 *j	 = t->job;
	oa_gen		 *g	 = &j->rc->gen;

	run_generation(j->rc, t->slot);
	if (j->chat_api)
		slot_adopt_chat_chain(t->slot, j);
	j->chain = NULL;
	slot_record_catalog(st, t->slot);
	INFO("gen: %s prompt=%d tokens generated=%d (%.1f s, %s)", j->chat_api ? "chat" : "completion",
		 g->prompt_tokens, g->generated, (double)(time_us() - j->t_start_us) / 1e6,
		 atomic_load_explicit(&g->client_gone, memory_order_acquire) ? "client disconnected"
																	 : "completed");
	job_free(st, j);
	t->done = true;
}

static bool task_start(openai_state *st, oa_job *j) {
	bool	 matched = false;
	oa_slot *slot	 = job_pick_slot(st, j, &matched);
	if (!slot) {
		if (st->n_tasks > 0)
			return false;
		ERROR("server: no session slot available");
		fail_job(st, j);
		return true;
	}

	oa_gen *g				= &j->rc->gen;
	g->session_key_at_start = j->session_key;
	g->same_session			= matched;
	INFO("session=%016llx slot=%d %s; n_messages=%zu", (unsigned long long)j->session_key,
		 (int)(slot - st->slots), matched ? "continuing existing session" : "fresh start",
		 j->rc->params.n_messages);

	void *stack = mmap(NULL, TASK_STACK_BYTES, PROT_READ | PROT_WRITE,
					   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK, -1, 0);
	if (stack == MAP_FAILED) {
		ERROR("server: failed to allocate task stack");
		fail_job(st, j);
		return true;
	}
	mprotect(stack, (size_t)sysconf(_SC_PAGESIZE), PROT_NONE);

	oa_task *t = xcalloc(1, sizeof(*t));
	t->st	   = st;
	t->job	   = j;
	t->slot	   = slot;
	t->stack   = stack;
	getcontext(&t->uc);
	t->uc.uc_stack.ss_sp   = stack;
	t->uc.uc_stack.ss_size = TASK_STACK_BYTES;
	t->uc.uc_link		   = &st->sched_uc;
	makecontext(&t->uc, task_main, 0);

	slot->busy				  = true;
	slot->sess.logits_on_host = false;
	slot->sess.yield		  = task_yield;
	slot->sess.yield_ud		  = t;
	slot->sess.decode_step	  = task_decode_step;
	slot->sess.decode_step_ud = t;
	st->tasks[st->n_tasks++]  = t;
	return true;
}

static void task_finish(openai_state *st, int idx) {
	oa_task *t					 = st->tasks[idx];
	t->slot->sess.yield			 = NULL;
	t->slot->sess.yield_ud		 = NULL;
	t->slot->sess.decode_step	 = NULL;
	t->slot->sess.decode_step_ud = NULL;
	munmap(t->stack, TASK_STACK_BYTES);
	free(t);
	st->tasks[idx] = st->tasks[--st->n_tasks];
}

static void admit_jobs(openai_state *st) {
	while (st->n_tasks < st->slots_cap && !shutting_down(st)) {
		pthread_mutex_lock(&st->sch_mtx);
		oa_job *j = st->q_head;
		if (j) {
			st->q_head = j->next;
			if (!st->q_head)
				st->q_tail = NULL;
		}
		pthread_mutex_unlock(&st->sch_mtx);
		if (!j)
			return;
		if (task_start(st, j))
			continue;
		pthread_mutex_lock(&st->sch_mtx);
		j->next	   = st->q_head;
		st->q_head = j;
		if (!st->q_tail)
			st->q_tail = j;
		pthread_mutex_unlock(&st->sch_mtx);
		return;
	}
}

static void slots_idle_prefill_wait(openai_state *st) {
	for (int i = 0; i < st->n_slots; i++)
		context_idle_prefill_wait(&st->slots[i].sess);
}

static void *worker_main(void *arg) {
	openai_state *st = (openai_state *)arg;
	for (;;) {
		pthread_mutex_lock(&st->sch_mtx);
		while (!st->q_head && st->n_tasks == 0 && !shutting_down(st))
			pthread_cond_wait(&st->sch_cv, &st->sch_mtx);
		pthread_mutex_unlock(&st->sch_mtx);
		if (st->n_tasks == 0 && shutting_down(st))
			break;

		slots_idle_prefill_wait(st);
		admit_jobs(st);

		for (int i = 0; i < st->n_tasks;) {
			oa_task *t	  = st->tasks[i];
			task_entering = t;
			swapcontext(&st->sched_uc, &t->uc);
			if (t->done)
				task_finish(st, i);
			else
				i++;
		}
		run_decode_step(st);
	}
	backend_release_thread(st->ctx->backend);
	return NULL;
}

static void job_push(openai_state *st, oa_job *j) {
	pthread_mutex_lock(&st->sch_mtx);
	j->next = NULL;
	if (st->q_tail)
		st->q_tail->next = j;
	else
		st->q_head = j;
	st->q_tail = j;
	pthread_cond_signal(&st->sch_cv);
	pthread_mutex_unlock(&st->sch_mtx);
}

static void job_wait_done(req_ctx *rc) {
	pthread_mutex_lock(&rc->gen.q_mtx);
	while (!rc->gen.generation_done &&
		   !atomic_load_explicit(&rc->gen.client_gone, memory_order_acquire))
		pthread_cond_wait(&rc->gen.q_cv, &rc->gen.q_mtx);
	pthread_mutex_unlock(&rc->gen.q_mtx);
}

static void jobs_cancel(openai_state *st) {
	while (st->q_head) {
		oa_job *j  = st->q_head;
		st->q_head = j->next;
		pthread_mutex_lock(&j->rc->gen.q_mtx);
		j->rc->gen.generation_done = true;
		pthread_cond_broadcast(&j->rc->gen.q_cv);
		pthread_mutex_unlock(&j->rc->gen.q_mtx);
		job_free(st, j);
	}
	st->q_tail = NULL;
}

static enum MHD_Result respond_auth_failure(openai_state *st, struct MHD_Connection *conn) {
	(void)st;
	return respond_json(conn, MHD_HTTP_UNAUTHORIZED,
						error_body("Invalid API key", "authentication_error"));
}

static enum MHD_Result handle_post(openai_state *st, struct MHD_Connection *conn, req_ctx *rc,
								   bool chat_api) {
	if (!check_auth(st, conn))
		return respond_auth_failure(st, conn);

	DEBUG("request body [%s] (%zu bytes): %.4096s",
		  chat_api ? "/v1/chat/completions" : "/v1/completions", rc->body.len,
		  rc->body.p ? rc->body.p : "(empty)");

	json_object *body = json_tokener_parse(rc->body.p ? rc->body.p : "");
	if (!body)
		return respond_json(conn, MHD_HTTP_BAD_REQUEST,
							error_body("Invalid JSON in request body", "invalid_request_error"));
	if (!json_object_is_type(body, json_type_object)) {
		json_object_put(body);
		return respond_json(
			conn, MHD_HTTP_BAD_REQUEST,
			error_body("Request body must be a JSON object", "invalid_request_error"));
	}

	const char *err = chat_api ? parse_chat_request(body, &rc->params)
							   : parse_completion_request(body, &rc->params);
	if (err) {
		json_object_put(body);
		return respond_json(conn, MHD_HTTP_BAD_REQUEST, error_body(err, "invalid_request_error"));
	}
	rc->body_json = body;

	uint64_t *new_chain = NULL;
	size_t	  new_len	= 0;
	uint64_t  key		= 0;
	if (chat_api) {
		new_chain = session_prefix_hash_chain(&rc->params, &new_len);
		key		  = new_chain[new_len];
	}

	rc->gen.streaming		= rc->params.stream;
	rc->gen.chat_api		= chat_api;
	rc->gen.generation_done = false;

	oa_job *j = xmalloc(sizeof(*j));
	*j		  = (oa_job){.rc		  = rc,
						 .chat_api	  = chat_api,
						 .chain		  = new_chain,
						 .chain_len	  = new_len,
						 .session_key = key};
	rc_ref(rc);
	INFO("push j=%p chain=%p", (void *)j, (void *)j->chain);
	job_push(st, j);

	if (rc->gen.streaming) {
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

	job_wait_done(rc);

	if (rc->gen.generated < 0) {
		INFO("gen: %s failed after %d tokens", chat_api ? "chat" : "completion", rc->gen.generated);
		return respond_json(conn, MHD_HTTP_SERVICE_UNAVAILABLE,
							error_body("generation failed (prompt too long, compute "
									   "error, or out of memory)",
									   "server_error"));
	}

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
										 const char *url, const char *method, bool chat_api) {
	if (rc->too_large)
		return respond_json(conn, MHD_HTTP_PAYLOAD_TOO_LARGE,
							error_body("request body too large", "invalid_request_error"));
	uint64_t		t0 = time_us();
	enum MHD_Result r  = handle_post(st, conn, rc, chat_api);
	INFO("%s %s (%.1f ms)", method, url, (double)(time_us() - t0) / 1000.0);
	return r;
}

static enum MHD_Result route_chat_completions(openai_state *st, struct MHD_Connection *conn,
											  req_ctx *rc, const char *url, const char *method) {
	return route_completions(st, conn, rc, url, method, true);
}

static enum MHD_Result route_text_completions(openai_state *st, struct MHD_Connection *conn,
											  req_ctx *rc, const char *url, const char *method) {
	return route_completions(st, conn, rc, url, method, false);
}

static enum MHD_Result route_models(openai_state *st, struct MHD_Connection *conn, req_ctx *rc,
									const char *url, const char *method) {
	(void)rc;
	(void)url;
	(void)method;
	if (!check_auth(st, conn))
		return respond_auth_failure(st, conn);
	return respond_json(conn, MHD_HTTP_OK, models_body(st));
}

static const route routes[] = {
	{"GET", "/health", route_health},
	{"GET", "/", route_root},
	{"POST", "/v1/chat/completions", route_chat_completions},
	{"POST", "/v1/completions", route_text_completions},
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
		if (rc->body.len + *upload_data_size > MAX_BODY_BYTES) {
			rc->too_large = true;
		} else {
			sb_putb(&rc->body, upload_data, *upload_data_size);
		}
		*upload_data_size = 0;
		return MHD_YES;
	}

	if (strcmp(method, "OPTIONS") == 0) {
		struct MHD_Response *resp =
			MHD_create_response_from_buffer(0, (void *)"", MHD_RESPMEM_PERSISTENT);
		add_cors(resp);
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
	return respond_json(conn, MHD_HTTP_NOT_FOUND, error_body(msgbuf, "invalid_request_error"));
}

openai_state *openai_init(context *ctx, const cli_args *args) {
	openai_state *st = xcalloc(1, sizeof(*st));
	st->ctx			 = ctx;
	st->model_id	 = derive_model_id(config_get()->model ? config_get()->model : "unknown");
	st->api_key		 = args->server_api_key ? xstrdup(args->server_api_key) : NULL;
	st->created_at	 = (long)time(NULL);
	atomic_store_explicit(&st->req_counter, 0, memory_order_relaxed);
	atomic_store_explicit(&st->shutting_down, false, memory_order_relaxed);
	pthread_mutex_init(&st->sch_mtx, NULL);
	pthread_cond_init(&st->sch_cv, NULL);

	st->slots_cap = args->server_slots > 0 ? args->server_slots : 1;
	st->slots	  = xcalloc((size_t)st->slots_cap, sizeof(oa_slot));
	st->tasks	  = xcalloc((size_t)st->slots_cap, sizeof(oa_task *));
	st->n_slots	  = 1;
	if (context_slot_init(&st->slots[0].sess, st->ctx) != OK) {
		ERROR("openai: failed to allocate base session slot");
		free(st->slots);
		free(st->tasks);
		free(st->model_id);
		free(st->api_key);
		pthread_mutex_destroy(&st->sch_mtx);
		pthread_cond_destroy(&st->sch_cv);
		free(st);
		return NULL;
	}
	st->slots[0].used	 = true;
	st->slots[0].busy	 = false;
	st->slots[0].owns	 = true;
	st->slots[0].last_ns = time_us();
	slot_warm(&st->slots[0].sess, args->warmup);
	if (args->warmup && !st->slots[0].sess.session_poisoned)
		context_idle_prefill_start(&st->slots[0].sess);
	backend_release_thread(st->ctx->backend);

	if (pthread_create(&st->worker, NULL, worker_main, st) != 0) {
		ERROR("openai: failed to start scheduler worker");
		free(st->slots);
		free(st->tasks);
		free(st->model_id);
		free(st->api_key);
		pthread_mutex_destroy(&st->sch_mtx);
		pthread_cond_destroy(&st->sch_cv);
		free(st);
		return NULL;
	}
	INFO("openai: model=%s api-key=%s slots=%d", st->model_id, st->api_key ? "set" : "unset",
		 st->slots_cap);
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
	pthread_mutex_lock(&st->sch_mtx);
	pthread_cond_broadcast(&st->sch_cv);
	pthread_mutex_unlock(&st->sch_mtx);
	for (int i = 0; i < st->n_slots; i++)
		st->slots[i].sess.interrupt = 1;
	if (st->daemon)
		MHD_stop_daemon(st->daemon);
	st->daemon = NULL;
}

void openai_free(openai_state *st) {
	if (!st)
		return;
	openai_stop(st);
	pthread_join(st->worker, NULL);
	for (int i = 0; i < st->n_slots; i++)
		if (st->slots[i].owns)
			context_slot_free(&st->slots[i].sess);
	free(st->slots);
	free(st->tasks);
	free(st->decode.tasks);
	free(st->decode.tokens);
	free(st->decode.positions);
	free(st->decode.caches);
	free(st->decode.logits);
	jobs_cancel(st);
	free(st->bind_addr);
	free(st->model_id);
	free(st->api_key);
	pthread_mutex_destroy(&st->sch_mtx);
	pthread_cond_destroy(&st->sch_cv);
	free(st);
}
