#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "coop.h"
#include "compute.h"
#include "kvcache.h"
#include "model.h"
#include "profile.h"
#include "threadpool.h"

#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define COOP_MIN_ROWS 32
#define COOP_MIN_TOKENS 64
#define COOP_GAP_PIECE_ROWS 512
#define COOP_STAGE_MAX_BYTES ((size_t)256 << 20)
#define COOP_MAX_LANES (CONFIG_MAX_LAYER_DEVICES + 1)
#define COOP_WAIT_NS 2000000L
#define COOP_CALIB_ROWS 32
#define COOP_CALIB_WARM_ROWS 16
#define COOP_CALIB_US 1200000
#define COOP_RANGES_PER_RUN 5

typedef struct {
	int			pos_start;
	int			n;
	int			tok_off;
	int			producer;
	bool		last;
	_Atomic int layers_done;
	void	  **stage;
	void	  **state;
	int		   *left;
} coop_range;

typedef struct coop_lane {
	backend			  *be;
	model			  *m;
	kvcache			  *kv;
	compute_scratch	  *scratch;
	model			   own_m;
	kvcache			   own_kv;
	compute_scratch	   own_scratch;
	bool			   owned;
	int32_t			  *ids;
	int				   n_ids;
	int				   valid;
	double			   rate;
	int				  *cursor;
	int				   idx;
	int				   run_tokens;
	uint64_t		   compute_us;
	uint64_t		   sync_us;
	uint64_t		   export_us;
	uint64_t		   busy_until_us;
	_Atomic int		   flight_rows;
	_Atomic int		   flight_layers;
	struct coop_board *board;
	pthread_t		   thread;
	status_code		   result;
} coop_lane;

struct coop {
	coop_lane lanes[COOP_MAX_LANES];
	int		  n_lanes;
	bool	  disabled;
};

typedef struct coop_board {
	pthread_mutex_t				mu;
	pthread_cond_t				cv;
	coop					   *k;
	const int32_t			   *tokens;
	const _Atomic sig_atomic_t *interrupt;
	progress				   *prog;
	layer_progress_cb			layer_cb;
	void					   *layer_cb_ud;
	int							n_tokens;
	int							pos;
	int							flash_attn;
	int							base_rows;
	int							max_rows;
	int							hard_rows;
	int							n_layers;
	int							n_consumers;
	int							next_tok;
	int							n_ranges;
	int							cap_ranges;
	_Atomic int					tokens_done;
	coop_range				   *ranges;
	_Atomic int					failed;
} coop_board;

typedef struct {
	coop_board *board;
	coop_lane  *lane;
	int			range;
	uint64_t	started_us;
} coop_gate;

typedef struct {
	bool engaged;
#if defined(__linux__)
	cpu_set_t mask;
#endif
} coop_thread_state;

static void coop_lane_enter(const backend *be, coop_thread_state *st) {
	st->engaged = false;
	if (backend_has_cap(be, BCAP_IS_HOST)) {
		backend_bind_phase(TPOOL_PHASE_PREFILL);
		return;
	}
#if defined(__linux__)
	if (sched_getaffinity(0, sizeof(st->mask), &st->mask) != 0)
		return;
	cpu_set_t all;
	CPU_ZERO(&all);
	long n_cpus = sysconf(_SC_NPROCESSORS_CONF);
	for (long i = 0; i < n_cpus && i < CPU_SETSIZE; i++)
		CPU_SET(i, &all);
	if (sched_setaffinity(0, sizeof(all), &all) != 0)
		return;
	st->engaged = true;
#endif
}

static void coop_lane_leave(const coop_thread_state *st) {
#if defined(__linux__)
	if (!st->engaged)
		return;
	sched_setaffinity(0, sizeof(st->mask), &st->mask);
#else
	(void)st;
#endif
}

static int coop_hard_rows(const model *m) {
	if (m->sliding_window <= 0)
		return INT_MAX;
	return MAX(8, m->sliding_window / 4) + 1;
}

static const char *coop_model_blocker(const model *m) {
	const arch_info *ai = m->arch_info;
	if (!m->recipe)
		return "model recipe is not built";
	if (m->full_sequence)
		return "full-sequence (embedding) models are not supported";
	if (model_mixed_backend_mode(m))
		return "layer-split (mixed backend) mode is active";
	if (ai->is_moe)
		return "mixture-of-experts models are not supported";
	if (ai->is_kda)
		return "KDA models are not supported";
	if (ai->is_mla)
		return "MLA models are not supported";
	if (coop_hard_rows(m) < COOP_MIN_ROWS)
		return "sliding window is too small to split into ranges";
	if (!recipe_is_decode_batchable(m))
		return "the device cannot batch every operation of this model";
	return NULL;
}

static const char *coop_backend_blocker(const backend *be, const model *m) {
	const arch_info *ai = m->arch_info;
	if (ai->hybrid_shortconv && !be->shortconv)
		return "device lacks shortconv";
	if (ai->is_hybrid_recurrent && !ai->hybrid_shortconv && !be->gated_delta_net)
		return "device lacks gated_delta_net";
	return NULL;
}

static void coop_fail(coop_board *b, status_code st) {
	int expected = OK;
	pthread_mutex_lock(&b->mu);
	atomic_compare_exchange_strong(&b->failed, &expected, (int)st);
	pthread_cond_broadcast(&b->cv);
	pthread_mutex_unlock(&b->mu);
}

static coop_range *coop_new_range(coop_board *b, int pos_start, int n, int tok_off, int producer) {
	coop_range *r = &b->ranges[b->n_ranges++];
	r->pos_start  = pos_start;
	r->n		  = n;
	r->tok_off	  = tok_off;
	r->producer	  = producer;
	r->last		  = false;
	atomic_init(&r->layers_done, 0);
	r->stage = xcalloc((size_t)b->n_layers, sizeof(void *));
	r->left	 = xcalloc((size_t)b->n_layers, sizeof(int));
	r->state = xcalloc((size_t)b->n_layers, sizeof(void *));
	return r;
}

static void coop_report_progress(coop_board *b, const coop_lane *lane) {
	if ((!b->prog && !b->layer_cb) || lane->idx != 0)
		return;
	double done = (double)atomic_load(&b->tokens_done);
	for (int i = 0; i < b->k->n_lanes; i++)
		done += (double)atomic_load(&b->k->lanes[i].flight_rows) *
				(double)atomic_load(&b->k->lanes[i].flight_layers) / (double)b->n_layers;
	int done_tokens = (int)MIN(done, (double)b->n_tokens);
	if (b->prog)
		progress_update(b->prog, (uint64_t)done_tokens);
	if (b->layer_cb)
		b->layer_cb(atomic_load(&lane->flight_layers), b->n_layers, done_tokens, 1, b->layer_cb_ud);
}

static status_code coop_export_layer(coop_board *b, coop_lane *lane, coop_range *r, int layer,
									 bool with_state) {
	void  *buf	 = NULL;
	size_t bytes = kvcache_range_bytes(lane->kv, lane->m, layer, r->n);
	if (bytes) {
		buf			   = xmalloc(bytes);
		status_code st = kvcache_export_range(lane->kv, lane->m, layer, r->pos_start, r->n, buf);
		if (st != OK) {
			free(buf);
			return st;
		}
	}
	void  *state	   = NULL;
	size_t state_bytes = with_state ? kvcache_state_bytes(lane->kv, lane->m, layer) : 0;
	if (state_bytes) {
		state		   = xmalloc(state_bytes);
		status_code st = kvcache_export_state(lane->kv, lane->m, layer, state);
		if (st != OK) {
			free(buf);
			free(state);
			return st;
		}
	}
	pthread_mutex_lock(&b->mu);
	r->state[layer] = state;
	r->stage[layer] = buf;
	r->left[layer]	= buf ? b->n_consumers : 0;
	atomic_store(&r->layers_done, layer + 1);
	pthread_cond_broadcast(&b->cv);
	pthread_mutex_unlock(&b->mu);
	return OK;
}

static status_code coop_wait_layer(coop_board *b, coop_lane *lane, coop_range *r, int layer) {
	pthread_mutex_lock(&b->mu);
	while (atomic_load(&r->layers_done) <= layer && atomic_load(&b->failed) == OK) {
		if (b->interrupt && *b->interrupt) {
			int expected = OK;
			atomic_compare_exchange_strong(&b->failed, &expected, (int)ERR_INTERRUPTED);
			pthread_cond_broadcast(&b->cv);
			break;
		}
		struct timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);
		ts.tv_nsec += COOP_WAIT_NS;
		if (ts.tv_nsec >= 1000000000L) {
			ts.tv_sec++;
			ts.tv_nsec -= 1000000000L;
		}
		pthread_cond_timedwait(&b->cv, &b->mu, &ts);
		coop_report_progress(b, lane);
	}
	status_code st = (status_code)atomic_load(&b->failed);
	pthread_mutex_unlock(&b->mu);
	return st;
}

static void coop_consume(coop_board *b, coop_range *r, int layer) {
	pthread_mutex_lock(&b->mu);
	if (--r->left[layer] == 0) {
		free(r->stage[layer]);
		r->stage[layer] = NULL;
	}
	pthread_mutex_unlock(&b->mu);
}

static status_code coop_import_upto(coop_board *b, coop_lane *lane, int layer, int upto) {
	while (lane->cursor[layer] < upto) {
		coop_range *r = &b->ranges[lane->cursor[layer]];
		if (r->producer != lane->idx) {
			status_code st = coop_wait_layer(b, lane, r, layer);
			if (st != OK)
				return st;
			if (r->stage[layer]) {
				st = kvcache_import_range(lane->kv, lane->m, layer, r->pos_start, r->n,
										  r->stage[layer]);
				if (st != OK)
					return st;
				coop_consume(b, r, layer);
			}
			if (lane->cursor[layer] == upto - 1 && r->state[layer]) {
				st = kvcache_import_state(lane->kv, lane->m, layer, r->state[layer]);
				if (st != OK)
					return st;
			}
		}
		lane->cursor[layer]++;
	}
	return OK;
}

static status_code coop_before(void *ud, int layer) {
	coop_gate  *g  = ud;
	status_code st = (status_code)atomic_load(&g->board->failed);
	if (st != OK)
		return st;
	uint64_t sync_start = time_us();
	st					= coop_import_upto(g->board, g->lane, layer, g->range);
	g->started_us		= time_us();
	g->lane->sync_us += g->started_us - sync_start;
	return st;
}

static status_code coop_after(void *ud, int layer) {
	coop_gate *g			= ud;
	uint64_t   export_start = time_us();
	g->lane->compute_us += export_start - g->started_us;
	status_code st = coop_export_layer(g->board, g->lane, &g->board->ranges[g->range], layer, true);
	if (st == OK && g->range > 0) {
		coop_range *prev = &g->board->ranges[g->range - 1];
		pthread_mutex_lock(&g->board->mu);
		free(prev->state[layer]);
		prev->state[layer] = NULL;
		pthread_mutex_unlock(&g->board->mu);
	}
	g->lane->export_us += time_us() - export_start;
	atomic_store(&g->lane->flight_layers, layer + 1);
	coop_report_progress(g->board, g->lane);
	return st;
}

static int coop_rows_for(const coop_board *b, const coop_lane *lane) {
	double fastest = 0;
	double slowest = 0;
	double total   = 0;
	for (int i = 0; i < b->k->n_lanes; i++) {
		double rate = b->k->lanes[i].rate;
		if (rate <= 0)
			continue;
		fastest = MAX(fastest, rate);
		slowest = slowest > 0 ? MIN(slowest, rate) : rate;
		total += rate;
	}
	if (fastest <= 0)
		return b->base_rows;
	double rate	   = lane->rate > 0 ? lane->rate : fastest;
	double seconds = (double)b->n_tokens / total / COOP_RANGES_PER_RUN;
	seconds		   = MAX(seconds, (double)COOP_MIN_ROWS / slowest);
	seconds		   = MIN(seconds, (double)b->max_rows / fastest);
	int rows	   = (int)(rate * seconds);
	return MIN(MAX(rows, COOP_MIN_ROWS), b->max_rows);
}

static bool coop_tail_declined(const coop_board *b, const coop_lane *lane, int rows, bool last,
							   uint64_t now) {
	double self_done = (double)now + (double)rows / lane->rate * 1e6;
	for (int i = 0; i < b->k->n_lanes; i++) {
		const coop_lane *other = &b->k->lanes[i];
		if (other == lane || other->rate <= lane->rate || (last && other->idx != 0))
			continue;
		double free_at = MAX((double)now, (double)other->busy_until_us);
		if (free_at + (double)rows / other->rate * 1e6 < self_done)
			return true;
	}
	return false;
}

static bool coop_claim(coop_board *b, coop_lane *lane, int *idx) {
	pthread_mutex_lock(&b->mu);
	if (atomic_load(&b->failed) != OK || b->next_tok >= b->n_tokens) {
		pthread_mutex_unlock(&b->mu);
		return false;
	}
	int left = b->n_tokens - b->next_tok;
	int rows = coop_rows_for(b, lane);
	if (rows >= left || (left - rows < COOP_MIN_ROWS && left <= b->hard_rows))
		rows = left;
	bool last = rows == left;
	if (last && lane->idx != 0) {
		rows = left - COOP_MIN_ROWS;
		last = false;
		if (rows < COOP_MIN_ROWS) {
			pthread_mutex_unlock(&b->mu);
			return false;
		}
	}
	uint64_t now = time_us();
	if (left <= b->max_rows && lane->rate > 0 && coop_tail_declined(b, lane, rows, last, now)) {
		pthread_mutex_unlock(&b->mu);
		return false;
	}
	lane->busy_until_us = now + (lane->rate > 0 ? (uint64_t)((double)rows / lane->rate * 1e6) : 0);
	coop_range *r		= coop_new_range(b, b->pos + b->next_tok, rows, b->next_tok, lane->idx);
	r->last				= last;
	*idx				= b->n_ranges - 1;
	b->next_tok += rows;
	pthread_mutex_unlock(&b->mu);
	return true;
}

static status_code coop_run_range(coop_board *b, coop_lane *lane, int idx) {
	coop_range		 *r			= &b->ranges[idx];
	coop_gate		  gate_ctx	= {b, lane, idx, 0};
	recipe_layer_gate gate		= {coop_before, coop_after, &gate_ctx};
	float			 *logits	= r->last ? lane->scratch->logits_host : NULL;
	uint64_t		  before_us = lane->compute_us;
	atomic_store(&lane->flight_layers, 0);
	atomic_store(&lane->flight_rows, r->n);

	status_code st = compute_prefill_range(lane->m, lane->kv, lane->scratch, b->tokens + r->tok_off,
										   r->n, r->pos_start, b->flash_attn, &gate, logits);
	if (st != OK)
		return st;

	double spent = (double)(lane->compute_us - before_us) / 1e6;
	pthread_mutex_lock(&b->mu);
	if (spent > 0) {
		double measured = (double)r->n / spent;
		lane->rate		= lane->rate > 0 ? 0.5 * lane->rate + 0.5 * measured : measured;
	}
	lane->run_tokens += r->n;
	atomic_fetch_add(&b->tokens_done, r->n);
	atomic_store(&lane->flight_rows, 0);
	pthread_mutex_unlock(&b->mu);
	coop_report_progress(b, lane);
	return OK;
}

static status_code coop_drain(coop_board *b, coop_lane *lane) {
	for (int layer = 0; layer < b->n_layers; layer++) {
		status_code st = coop_import_upto(b, lane, layer, b->n_ranges);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code coop_lane_run(coop_board *b, coop_lane *lane) {
	status_code st = OK;
	int			idx;
	while (st == OK && coop_claim(b, lane, &idx))
		st = coop_run_range(b, lane, idx);
	if (st == OK && atomic_load(&b->failed) == OK)
		st = coop_drain(b, lane);
	if (st != OK)
		coop_fail(b, st);
	return st;
}

static void *coop_thread_main(void *arg) {
	coop_lane		 *lane = arg;
	coop_thread_state state;
	coop_lane_enter(lane->be, &state);
	lane->result = coop_lane_run(lane->board, lane);
	backend_release_thread(lane->be);
	return NULL;
}

static status_code coop_stage_gap(coop_board *b, coop_lane *primary, int gap_start, int pos) {
	for (int off = gap_start; off < pos; off += COOP_GAP_PIECE_ROWS) {
		coop_range *r = coop_new_range(b, off, MIN(COOP_GAP_PIECE_ROWS, pos - off), -1, 0);
		for (int layer = 0; layer < b->n_layers; layer++) {
			status_code st = coop_export_layer(b, primary, r, layer, off + r->n >= pos);
			if (st != OK)
				return st;
		}
	}
	return OK;
}

static status_code coop_prepare_lanes(coop *k, coop_board *b, int rows) {
	for (int i = 0; i < k->n_lanes; i++) {
		coop_lane *lane		= &k->lanes[i];
		lane->board			= b;
		lane->idx			= i;
		lane->run_tokens	= 0;
		lane->compute_us	= 0;
		lane->sync_us		= 0;
		lane->export_us		= 0;
		lane->busy_until_us = 0;
		atomic_store(&lane->flight_rows, 0);
		atomic_store(&lane->flight_layers, 0);
		free(lane->cursor);
		lane->cursor   = xcalloc((size_t)b->n_layers, sizeof(int));
		status_code st = compute_scratch_ensure(lane->scratch, lane->m, lane->kv->n_ctx);
		if (st != OK)
			return st;
		compute_scratch_reserve_rows(lane->m, lane->scratch, rows);
	}
	return OK;
}

static void coop_release_worker_lanes(coop *k) {
	for (int i = 1; i < k->n_lanes; i++)
		backend_release_thread(k->lanes[i].be);
}

static status_code coop_run_lanes(coop *k, coop_board *b) {
	bool started[COOP_MAX_LANES] = {false};
	coop_release_worker_lanes(k);
	for (int i = 1; i < k->n_lanes; i++) {
		if (pthread_create(&k->lanes[i].thread, NULL, coop_thread_main, &k->lanes[i]) != 0) {
			coop_fail(b, ERR_COMPUTE_FAIL);
			break;
		}
		started[i] = true;
	}
	coop_thread_state state;
	coop_lane_enter(k->lanes[0].be, &state);
	coop_lane_run(b, &k->lanes[0]);
	coop_lane_leave(&state);
	for (int i = 1; i < k->n_lanes; i++)
		if (started[i])
			pthread_join(k->lanes[i].thread, NULL);
	return (status_code)atomic_load(&b->failed);
}

static void coop_release_board(coop *k, coop_board *b) {
	for (int i = 0; i < b->n_ranges; i++) {
		for (int layer = 0; layer < b->n_layers; layer++) {
			free(b->ranges[i].stage[layer]);
			free(b->ranges[i].state[layer]);
		}
		free(b->ranges[i].stage);
		free(b->ranges[i].state);
		free(b->ranges[i].left);
	}
	free(b->ranges);
	for (int i = 0; i < k->n_lanes; i++) {
		free(k->lanes[i].cursor);
		k->lanes[i].cursor = NULL;
	}
	pthread_mutex_destroy(&b->mu);
	pthread_cond_destroy(&b->cv);
}

static void coop_forget_helpers(coop *k) {
	for (int i = 1; i < k->n_lanes; i++)
		k->lanes[i].n_ids = 0;
}

static void coop_record_helpers(coop *k, const context *c, const int32_t *tokens, int n, int pos,
								int avail) {
	for (int i = 1; i < k->n_lanes; i++) {
		coop_lane *lane = &k->lanes[i];
		if (avail < pos) {
			lane->n_ids = 0;
			continue;
		}
		lane->ids = xrealloc(lane->ids, (size_t)(pos + n) * sizeof(int32_t));
		memcpy(lane->ids, c->fed_ids.p, (size_t)pos * sizeof(int32_t));
		memcpy(lane->ids + pos, tokens, (size_t)n * sizeof(int32_t));
		lane->n_ids = pos + n;
	}
}

static void coop_log_run(const coop *k, int n, uint64_t t0_us) {
	double secs = (double)(time_us() - t0_us) / 1e6;
	DEBUG("cooperative prefill: %d tokens in %.2f s (%.1f t/s)", n, secs,
		  secs > 0 ? n / secs : 0.0);
	for (int i = 0; i < k->n_lanes; i++)
		DEBUG(
			"  lane '%s': %d tokens, %.1f t/s compute, compute %.2f s, sync %.2f s, export %.2f s",
			k->lanes[i].be->name, k->lanes[i].run_tokens, k->lanes[i].rate,
			(double)k->lanes[i].compute_us / 1e6, (double)k->lanes[i].sync_us / 1e6,
			(double)k->lanes[i].export_us / 1e6);
}

status_code coop_prefill(coop *k, context *c, const int32_t *tokens, int n, int pos, int base_rows,
						 progress *prog, layer_progress_cb layer_cb, void *layer_cb_ud) {
	const int n_layers	= c->m.n_layers;
	size_t	  per_token = 0;
	for (int layer = 0; layer < n_layers; layer++)
		per_token += kvcache_range_bytes(&c->kv, &c->m, layer, 1);

	const int avail		= MIN(pos, c->fed_ids.n);
	int		  gap_start = pos;
	for (int i = 1; i < k->n_lanes; i++) {
		coop_lane *lane	 = &k->lanes[i];
		int		   limit = MIN(lane->n_ids, avail);
		lane->valid		 = 0;
		while (lane->valid < limit && lane->ids[lane->valid] == c->fed_ids.p[lane->valid])
			lane->valid++;
		gap_start = MIN(gap_start, lane->valid);
	}
	if ((size_t)(pos - gap_start) * per_token > COOP_STAGE_MAX_BYTES)
		return ERR_UNSUPPORTED;

	uint64_t   t0_us	  = time_us();
	int		   gap_pieces = (pos - gap_start + COOP_GAP_PIECE_ROWS - 1) / COOP_GAP_PIECE_ROWS;
	coop_board b = {.k			 = k,
					.tokens		 = tokens,
					.interrupt	 = &c->interrupt,
					.prog		 = prog,
					.layer_cb	 = layer_cb,
					.layer_cb_ud = layer_cb_ud,
					.n_tokens	 = n,
					.pos		 = pos,
					.flash_attn	 = c->flash_attn,
					.base_rows	 = base_rows,
					.max_rows  = MIN(MAX(2 * base_rows, 2 * COOP_MIN_ROWS), coop_hard_rows(&c->m)),
					.hard_rows = coop_hard_rows(&c->m),
					.n_layers  = n_layers,
					.n_consumers = k->n_lanes - 1,
					.cap_ranges	 = gap_pieces + n / COOP_MIN_ROWS + 2};
	pthread_mutex_init(&b.mu, NULL);
	pthread_cond_init(&b.cv, NULL);
	atomic_init(&b.failed, OK);
	b.ranges = xcalloc((size_t)b.cap_ranges, sizeof(coop_range));

	status_code st = coop_stage_gap(&b, &k->lanes[0], gap_start, pos);
	if (st == OK)
		st = coop_prepare_lanes(k, &b, b.max_rows + COOP_MIN_ROWS);
	if (st == OK)
		st = coop_run_lanes(k, &b);

	if (st == OK) {
		coop_record_helpers(k, c, tokens, n, pos, avail);
		coop_log_run(k, n, t0_us);
	} else
		coop_forget_helpers(k);
	coop_release_board(k, &b);

	if (st == OK || st == ERR_INTERRUPTED)
		return st;
	WARN("cooperative prefill failed (%s); using single-device prefill", status_str(st));
	k->disabled = true;
	return ERR_UNSUPPORTED;
}

bool coop_usable(const coop *k, const context *c, int n_tokens) {
	return k && !k->disabled && !c->yield && n_tokens >= COOP_MIN_TOKENS;
}

static void coop_lane_free(coop_lane *lane) {
	if (lane->owned) {
		compute_scratch_free(&lane->own_scratch);
		kvcache_free(&lane->own_kv);
		model_free(&lane->own_m);
	}
	free(lane->ids);
	free(lane->cursor);
	memset(lane, 0, sizeof(*lane));
}

void coop_close(coop *k) {
	if (!k)
		return;
	for (int i = 0; i < k->n_lanes; i++)
		coop_lane_free(&k->lanes[i]);
	free(k);
}

static status_code coop_lane_load(coop_lane *lane, const context *c, const config *cfg, backend *be,
								  const char **why) {
	*why		  = NULL;
	lane->be	  = be;
	lane->m		  = &lane->own_m;
	lane->kv	  = &lane->own_kv;
	lane->scratch = &lane->own_scratch;
	compute_scratch_init(&lane->own_scratch);

	status_code st =
		model_load_parse(&lane->own_m, cfg->model, be, cfg->use_mmap, cfg->repack, NULL);
	if (st != OK)
		return st;
	lane->owned					 = true;
	lane->own_m.skip_output_head = true;
	if (cfg->fuse && cfg->fuse[0] && strcmp(cfg->fuse, "none") != 0)
		lane->own_m.fuse_config = cfg->fuse;
	be->rope_neox = lane->own_m.arch_info->uses_neox_rope;
	st			  = model_upload_weights(&lane->own_m);
	if (st == OK)
		st = model_build_recipe(&lane->own_m);
	if (st == OK) {
		*why = coop_model_blocker(&lane->own_m);
		if (!*why)
			*why = coop_backend_blocker(be, &lane->own_m);
		if (*why)
			st = ERR_UNSUPPORTED;
	}
	if (st == OK)
		st = kvcache_init(&lane->own_kv, &lane->own_m, c->n_ctx, c->kv.kv_quant);
	if (st == OK)
		st = kvcache_adopt_device_conv_state(&lane->own_kv, &lane->own_m);
	lane->own_scratch.interrupt		= &c->interrupt;
	lane->own_scratch.model.backend = be;
	return st;
}

static bool coop_backend_taken(const coop *k, const backend *be) {
	for (int i = 0; i < k->n_lanes; i++)
		if (k->lanes[i].be == be ||
			(backend_has_cap(k->lanes[i].be, BCAP_IS_HOST) && backend_has_cap(be, BCAP_IS_HOST)))
			return true;
	return false;
}

typedef struct {
	coop_lane	*lane;
	int			 flash_attn;
	int			 n_lanes;
	uint64_t	 deadline_us;
	_Atomic int *first_done;
	int			 tokens;
	uint64_t	 compute_us;
} coop_calib;

static void coop_warm(coop_lane *lane, int flash_attn) {
	int32_t			  tokens[COOP_CALIB_WARM_ROWS] = {0};
	coop_thread_state state;
	coop_lane_enter(lane->be, &state);
	if (compute_scratch_ensure(lane->scratch, lane->m, lane->kv->n_ctx) == OK) {
		compute_scratch_reserve_rows(lane->m, lane->scratch, COOP_CALIB_ROWS);
		if (compute_prefill_range(lane->m, lane->kv, lane->scratch, tokens, COOP_CALIB_WARM_ROWS, 0,
								  flash_attn, NULL, NULL) == OK)
			ensure_sync(lane->be);
	}
	coop_lane_leave(&state);
}

static void coop_calib_run(coop_calib *cb) {
	int32_t tokens[COOP_CALIB_ROWS] = {0};
	bool	timed					= false;
	while (!timed || time_us() < cb->deadline_us || atomic_load(cb->first_done) < cb->n_lanes) {
		uint64_t start_us = time_us();
		if (compute_prefill_range(cb->lane->m, cb->lane->kv, cb->lane->scratch, tokens,
								  COOP_CALIB_ROWS, 0, cb->flash_attn, NULL, NULL) != OK)
			break;
		ensure_sync(cb->lane->be);
		cb->compute_us += time_us() - start_us;
		cb->tokens += COOP_CALIB_ROWS;
		if (!timed) {
			timed = true;
			atomic_fetch_add(cb->first_done, 1);
		}
	}
	if (!timed)
		atomic_fetch_add(cb->first_done, 1);
}

static void *coop_calib_thread(void *arg) {
	coop_calib		 *cb = arg;
	coop_thread_state state;
	coop_lane_enter(cb->lane->be, &state);
	coop_calib_run(cb);
	backend_release_thread(cb->lane->be);
	return NULL;
}

static void coop_calibrate(coop *k, int flash_attn) {
	for (int i = 0; i < k->n_lanes; i++)
		coop_warm(&k->lanes[i], flash_attn);
	coop_release_worker_lanes(k);

	_Atomic int first_done = 0;
	coop_calib	calib[COOP_MAX_LANES];
	bool		started[COOP_MAX_LANES] = {false};
	uint64_t	deadline_us				= time_us() + COOP_CALIB_US;
	for (int i = 0; i < k->n_lanes; i++)
		calib[i] = (coop_calib){.lane		 = &k->lanes[i],
								.flash_attn	 = flash_attn,
								.n_lanes	 = k->n_lanes,
								.deadline_us = deadline_us,
								.first_done	 = &first_done};
	for (int i = 1; i < k->n_lanes; i++) {
		if (pthread_create(&k->lanes[i].thread, NULL, coop_calib_thread, &calib[i]) == 0)
			started[i] = true;
		else
			atomic_fetch_add(&first_done, 1);
	}
	coop_thread_state state;
	coop_lane_enter(k->lanes[0].be, &state);
	coop_calib_run(&calib[0]);
	coop_lane_leave(&state);
	for (int i = 1; i < k->n_lanes; i++)
		if (started[i])
			pthread_join(k->lanes[i].thread, NULL);

	for (int i = 0; i < k->n_lanes; i++) {
		if (calib[i].compute_us == 0)
			continue;
		k->lanes[i].rate = (double)calib[i].tokens / ((double)calib[i].compute_us / 1e6);
		INFO("prefill lane '%s': %.1f t/s", k->lanes[i].be->name, k->lanes[i].rate);
	}
}

status_code coop_open(coop **out, context *c, const config *cfg, backend *const *devices,
					  int n_devices) {
	*out = NULL;
	if (n_devices < 1)
		return OK;
	const char *blocker = coop_model_blocker(&c->m);
	if (blocker) {
		WARN("cooperative prefill disabled: %s", blocker);
		return OK;
	}
	coop *k				= xcalloc(1, sizeof(*k));
	k->lanes[0].be		= c->backend;
	k->lanes[0].m		= &c->m;
	k->lanes[0].kv		= &c->kv;
	k->lanes[0].scratch = &c->scratch;
	k->n_lanes			= 1;

	for (int i = 0; i < n_devices && k->n_lanes < COOP_MAX_LANES; i++) {
		if (coop_backend_taken(k, devices[i])) {
			WARN("cooperative prefill: device '%s' skipped: it shares a backend with another lane",
				 devices[i]->name);
			continue;
		}
		coop_lane  *lane = &k->lanes[k->n_lanes];
		const char *why	 = NULL;
		status_code st	 = coop_lane_load(lane, c, cfg, devices[i], &why);
		if (st != OK) {
			WARN("cooperative prefill: device '%s' skipped: %s", devices[i]->name,
				 why ? why : status_str(st));
			coop_lane_free(lane);
			continue;
		}
		k->n_lanes++;
	}
	if (k->n_lanes < 2) {
		WARN("cooperative prefill disabled: no usable prefill device");
		coop_close(k);
		return OK;
	}
	coop_calibrate(k, c->flash_attn);
	*out = k;
	return OK;
}
