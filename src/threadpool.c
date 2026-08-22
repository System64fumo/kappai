#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "threadpool.h"
#include "common.h"
#include "config.h"
#include "log.h"
#include "profile.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define TPOOL_SPIN_BUDGET_MIN_NS 5000
#define TPOOL_SPIN_BUDGET_MAX_NS 40000
#define TPOOL_SPIN_BUDGET_INITIAL_NS 50000
#define TPOOL_WAIT_EPOCH_SPIN_BUDGET_NS 20000
#define TPOOL_SPIN_CHECK_EVERY 128
#define TPOOL_MAX_THREADS 256
#define TPOOL_SPIN_EMA_NUM 1
#define TPOOL_SPIN_EMA_DEN 4
#define TPOOL_SPIN_BUDGET_FRACTION_NUM 1
#define TPOOL_SPIN_BUDGET_FRACTION_DEN 2

typedef struct tlocal_slot {
	void			  **ptr;
	struct tlocal_slot *next;
} tlocal_slot;

static void tlocal_cleanup(void *v) {
	tlocal_slot *s = v;
	while (s) {
		if (s->ptr && *s->ptr) {
			free(*s->ptr);
			*s->ptr = NULL;
		}
		tlocal_slot *n = s->next;
		free(s);
		s = n;
	}
}

static pthread_once_t tlocal_once = PTHREAD_ONCE_INIT;
static pthread_key_t  tlocal_key  = 0;

static void tlocal_key_make(void) {
	pthread_key_create(&tlocal_key, tlocal_cleanup);
}

void tlocal_register(void **tls_ptr) {
	pthread_once(&tlocal_once, tlocal_key_make);
	tlocal_slot *s = malloc(sizeof(*s));
	s->ptr		   = tls_ptr;
	s->next		   = pthread_getspecific(tlocal_key);
	pthread_setspecific(tlocal_key, s);
}

typedef struct {
	_Atomic uint64_t total_items;
	_Atomic uint64_t total_busy_ns;
	_Atomic uint64_t total_wait_ns;
	_Atomic uint64_t total_parked_ns;
	char			 _pad[64];
} __attribute__((aligned(64))) tpool_slot;

typedef struct {
	tpool *pool;
	long   tid;
} tpool_thread_arg;

static inline void tpool_run_timed(tpool_slot *slot, tpool_chunk_fn fn, int begin, int end, int tid,
								   void *ctx, int stats_enabled) {
	int n = end - begin;
	if (n <= 0)
		return;
	if (!stats_enabled) {
		fn(begin, end, tid, ctx);
		return;
	}
	long long t0 = time_ns();
	fn(begin, end, tid, ctx);
	long long t1 = time_ns();
	atomic_fetch_add_explicit(&slot->total_items, (uint64_t)n, memory_order_relaxed);
	atomic_fetch_add_explicit(&slot->total_busy_ns, (uint64_t)(t1 - t0), memory_order_relaxed);
}

typedef struct {
	tpool_chunk_fn job_fn;
	void		  *job_ctx;
	int			   job_end;
	int			   chunk_size;
	_Atomic int	   n_workers;
	_Atomic int	   cursor;
	_Atomic int	   job_epoch;
} __attribute__((aligned(64))) tpool_job_block;

typedef struct {
	_Atomic int epoch;
	_Atomic int active_workers;
	_Atomic int active_remaining;
	_Atomic int parked_workers;
	_Atomic int shutdown;
	_Atomic int spin_budget_ns;
} __attribute__((aligned(64))) tpool_sync_block;

struct tpool {
	tpool_job_block			job;
	tpool_sync_block		sync;
	tpool_slot			   *slot;
	tpool_worker_cleanup_fn worker_cleanup;
	pthread_mutex_t			wake_mtx;
	pthread_cond_t			wake_cv;
	pthread_t				threads[TPOOL_MAX_THREADS];
	int						n_threads;
	int						stats_enabled;
};

static __thread int tpool_cur_tid = -1;

int tpool_current_tid(void) {
	return tpool_cur_tid;
}

static int tpool_wait_epoch(tpool *pool, int last_seen_epoch, int tid) {
	uint64_t t_wait_start = time_ns();
	uint64_t t_spin_start = t_wait_start;
	int		 spins		  = 0;
	int		 budget_ns	  = TPOOL_WAIT_EPOCH_SPIN_BUDGET_NS;
	for (;;) {
		int e = atomic_load_explicit(&pool->sync.epoch, memory_order_acquire);
		if (e != last_seen_epoch) {
			if (tid >= 0)
				atomic_fetch_add_explicit(&pool->slot[tid].total_wait_ns,
										  (time_ns() - t_wait_start), memory_order_relaxed);
			return e;
		}
		if (atomic_load_explicit(&pool->sync.shutdown, memory_order_acquire))
			return last_seen_epoch;

		spins++;
		if (spins % TPOOL_SPIN_CHECK_EVERY == 0 && time_ns() - t_spin_start > (uint64_t)budget_ns)
			break;

		cpu_relax();
	}

	uint64_t t_park_start = time_ns();
	atomic_fetch_add_explicit(&pool->sync.parked_workers, 1, memory_order_relaxed);
	pthread_mutex_lock(&pool->wake_mtx);
	int e;
	for (;;) {
		e = atomic_load_explicit(&pool->sync.epoch, memory_order_acquire);
		if (e != last_seen_epoch)
			break;
		if (atomic_load_explicit(&pool->sync.shutdown, memory_order_acquire))
			break;

		struct timespec ts;
		uint64_t		deadline = time_ns() + 200000;
		ts.tv_sec				 = deadline / 1000000000ULL;
		ts.tv_nsec				 = deadline % 1000000000ULL;
		pthread_cond_timedwait(&pool->wake_cv, &pool->wake_mtx, &ts);
	}
	pthread_mutex_unlock(&pool->wake_mtx);
	atomic_fetch_sub_explicit(&pool->sync.parked_workers, 1, memory_order_relaxed);

	if (tid >= 0) {
		uint64_t now = time_ns();
		atomic_fetch_add_explicit(&pool->slot[tid].total_wait_ns, (now - t_wait_start),
								  memory_order_relaxed);
		atomic_fetch_add_explicit(&pool->slot[tid].total_parked_ns, (now - t_park_start),
								  memory_order_relaxed);
	}
	return e;
}

static int tpool_affinity_count(void) {
	cpu_set_t set;
	CPU_ZERO(&set);
	if (sched_getaffinity(0, sizeof(set), &set) != 0)
		return 0;
	int n = CPU_COUNT(&set);
	return n > 0 ? n : 0;
}

static int tpool_drain_work(tpool *pool, int tid, int expected_epoch) {
	tpool_slot *s = &pool->slot[tid];
	int			cur;

	for (;;) {
		int live_epoch = atomic_load_explicit(&pool->job.job_epoch, memory_order_acquire);
		if (live_epoch != expected_epoch)
			return 0;

		tpool_chunk_fn fn	 = pool->job.job_fn;
		void		  *ctx	 = pool->job.job_ctx;
		int			   end	 = pool->job.job_end;
		int			   chunk = pool->job.chunk_size;
		if (!fn)
			return 1;

		cur = atomic_load_explicit(&pool->job.cursor, memory_order_relaxed);
		if (cur >= end)
			return 1;

		int next = cur + chunk;
		if (next > end)
			next = end;

		if (atomic_compare_exchange_weak_explicit(&pool->job.cursor, &cur, next,
												  memory_order_acq_rel, memory_order_relaxed)) {
			int confirm_epoch = atomic_load_explicit(&pool->job.job_epoch, memory_order_acquire);
			if (confirm_epoch != expected_epoch) {
				return 1;
			}
			tpool_run_timed(s, fn, cur, next, tid, ctx, pool->stats_enabled);
		}
	}
}

static void *tpool_worker_main(void *arg) {
	tpool_thread_arg *a	   = (tpool_thread_arg *)arg;
	tpool			 *pool = a->pool;
	int				  tid  = (int)a->tid;
	free(a);

	int last_seen = 0;
	for (;;) {
		int new_epoch = tpool_wait_epoch(pool, last_seen, tid);
		if (new_epoch == last_seen) {
			if (pool->worker_cleanup)
				pool->worker_cleanup();
			return NULL;
		}
		last_seen = new_epoch;

		int job_epoch_now = atomic_load_explicit(&pool->job.job_epoch, memory_order_acquire);
		if (job_epoch_now != new_epoch)
			continue;
		int n_workers_now = atomic_load_explicit(&pool->job.n_workers, memory_order_relaxed);

		if (tid <= n_workers_now) {
			tpool_cur_tid	  = tid;
			int fully_drained = tpool_drain_work(pool, tid, new_epoch);
			tpool_cur_tid	  = -1;
			if (fully_drained)
				atomic_fetch_sub_explicit(&pool->sync.active_remaining, 1, memory_order_release);
		}
	}
}

int tpool_default_n_threads(void) {
	int v = config_get()->n_threads;
	if (v > 0)
		return v > TPOOL_MAX_THREADS ? TPOOL_MAX_THREADS : v;

	int n = tpool_affinity_count();
	if (n < 1) {
		long online = sysconf(_SC_NPROCESSORS_ONLN);
		n			= online < 1 ? 1 : (int)online;
	}
	if (n > TPOOL_MAX_THREADS)
		n = TPOOL_MAX_THREADS;
	return n;
}

tpool *tpool_create(int n_threads) {
	if (n_threads < 1)
		n_threads = 1;
	if (n_threads > TPOOL_MAX_THREADS)
		n_threads = TPOOL_MAX_THREADS;

	tpool *pool = xmalloc_aligned(sizeof(*pool), 64);
	memset(pool, 0, sizeof(*pool));
	pool->n_threads = n_threads;
	pool->slot		= xcalloc((size_t)n_threads, sizeof(tpool_slot));
	pthread_mutex_init(&pool->wake_mtx, NULL);
	{
		pthread_condattr_t cattr;
		pthread_condattr_init(&cattr);
		pthread_condattr_setclock(&cattr, CLOCK_MONOTONIC);
		pthread_cond_init(&pool->wake_cv, &cattr);
		pthread_condattr_destroy(&cattr);
	}

	atomic_store_explicit(&pool->sync.epoch, 0, memory_order_relaxed);
	atomic_store_explicit(&pool->sync.active_workers, 0, memory_order_relaxed);
	atomic_store_explicit(&pool->sync.active_remaining, 0, memory_order_relaxed);
	atomic_store_explicit(&pool->sync.shutdown, 0, memory_order_relaxed);
	atomic_store_explicit(&pool->sync.parked_workers, 0, memory_order_relaxed);
	atomic_store_explicit(&pool->sync.spin_budget_ns, TPOOL_SPIN_BUDGET_INITIAL_NS,
						  memory_order_relaxed);
	pool->job.chunk_size = 1;
	pool->job.job_fn	 = NULL;
	pool->job.job_ctx	 = NULL;
	atomic_store_explicit(&pool->job.cursor, 0, memory_order_relaxed);
	atomic_store_explicit(&pool->job.job_epoch, 0, memory_order_relaxed);
	atomic_store_explicit(&pool->job.n_workers, 0, memory_order_relaxed);
	pool->stats_enabled = getenv("TPOOL_STATS") != NULL;

	int started = 1;
	for (int i = 1; i < n_threads; i++) {
		tpool_thread_arg *arg = malloc(sizeof(*arg));
		int				  rc  = 0;
		if (arg) {
			arg->pool = pool;
			arg->tid  = i;
			rc		  = pthread_create(&pool->threads[i], NULL, tpool_worker_main, arg);
			if (rc != 0)
				free(arg);
		}
		if (!arg || rc != 0) {
			atomic_store_explicit(&pool->sync.shutdown, 1, memory_order_release);
			pthread_mutex_lock(&pool->wake_mtx);
			pthread_cond_broadcast(&pool->wake_cv);
			pthread_mutex_unlock(&pool->wake_mtx);
			for (int j = 1; j < i; j++)
				pthread_join(pool->threads[j], NULL);
			pool->n_threads = started;
			atomic_store_explicit(&pool->sync.shutdown, 0, memory_order_relaxed);
			return pool;
		}
		started++;
	}

	return pool;
}

static void tpool_dump_stats(const tpool *pool) {
	if (!pool->stats_enabled)
		return;
	fprintf(stderr, "\n[tpool] per-thread stats (%d threads):\n", pool->n_threads);
	fprintf(stderr, "%-4s %12s %12s %12s %12s\n", "tid", "items", "busy_ms", "wait_ms",
			"parked_ms");
	for (int i = 0; i < pool->n_threads; i++) {
		const tpool_slot *s		  = &pool->slot[i];
		uint64_t		  items	  = atomic_load_explicit(&s->total_items, memory_order_relaxed);
		uint64_t		  busy_ns = atomic_load_explicit(&s->total_busy_ns, memory_order_relaxed);
		uint64_t		  wait_ns = atomic_load_explicit(&s->total_wait_ns, memory_order_relaxed);
		uint64_t parked_ns		  = atomic_load_explicit(&s->total_parked_ns, memory_order_relaxed);
		fprintf(stderr, "%-4d %12llu %12.2f %12.2f %12.2f\n", i, (unsigned long long)items,
				(double)busy_ns / 1.0e6, (double)wait_ns / 1.0e6, (double)parked_ns / 1.0e6);
	}
}

void tpool_destroy(tpool *pool) {
	if (!pool)
		return;
	tpool_dump_stats(pool);
	atomic_store_explicit(&pool->sync.shutdown, 1, memory_order_release);
	pthread_mutex_lock(&pool->wake_mtx);
	pthread_cond_broadcast(&pool->wake_cv);
	pthread_mutex_unlock(&pool->wake_mtx);
	for (int i = 1; i < pool->n_threads; i++)
		pthread_join(pool->threads[i], NULL);
	pthread_mutex_destroy(&pool->wake_mtx);
	pthread_cond_destroy(&pool->wake_cv);
	free(pool->slot);
	free(pool);
}

int tpool_n_threads(const tpool *pool) {
	return pool ? pool->n_threads : 1;
}

void tpool_set_worker_cleanup(tpool *pool, tpool_worker_cleanup_fn fn) {
	if (pool)
		pool->worker_cleanup = fn;
}

static void tpool_update_spin_budget(tpool *pool, long long wall_ns) {
	long target =
		(long)((double)wall_ns * TPOOL_SPIN_BUDGET_FRACTION_NUM / TPOOL_SPIN_BUDGET_FRACTION_DEN);
	if (target < TPOOL_SPIN_BUDGET_MIN_NS)
		target = TPOOL_SPIN_BUDGET_MIN_NS;
	if (target > TPOOL_SPIN_BUDGET_MAX_NS)
		target = TPOOL_SPIN_BUDGET_MAX_NS;

	int cur		= atomic_load_explicit(&pool->sync.spin_budget_ns, memory_order_relaxed);
	int updated = (int)(cur + ((target - cur) * TPOOL_SPIN_EMA_NUM / TPOOL_SPIN_EMA_DEN));
	atomic_store_explicit(&pool->sync.spin_budget_ns, updated, memory_order_relaxed);
}

void tpool_parallel_for(tpool *pool, int n_items, int min_items_per_thread, tpool_chunk_fn fn,
						void *ctx) {
	if (n_items <= 0)
		return;

	int n_threads = pool ? pool->n_threads : 1;
	if (min_items_per_thread < 1)
		min_items_per_thread = 1;

	int usable = n_items / min_items_per_thread;
	if (usable < 1)
		usable = 1;
	if (usable > n_threads)
		usable = n_threads;

	if (usable <= 1 || !pool) {
		fn(0, n_items, 0, ctx);
		return;
	}

	int chunk_size			  = min_items_per_thread;
	int min_chunks_per_thread = 4;
	while (chunk_size > 1 && n_items / chunk_size < usable * min_chunks_per_thread)
		chunk_size /= 2;
	if (chunk_size < 1)
		chunk_size = 1;

	pool->job.job_end	 = n_items;
	pool->job.chunk_size = chunk_size;
	pool->job.job_fn	 = fn;
	pool->job.job_ctx	 = ctx;
	atomic_store_explicit(&pool->job.cursor, 0, memory_order_relaxed);

	int n_workers = usable - 1;
	atomic_store_explicit(&pool->job.n_workers, n_workers, memory_order_relaxed);
	atomic_store_explicit(&pool->sync.active_workers, n_workers, memory_order_release);
	atomic_store_explicit(&pool->sync.active_remaining, n_workers, memory_order_release);

	int new_epoch = atomic_load_explicit(&pool->sync.epoch, memory_order_relaxed) + 1;
	atomic_store_explicit(&pool->job.job_epoch, new_epoch, memory_order_release);
	if (atomic_load_explicit(&pool->sync.parked_workers, memory_order_relaxed) == 0) {
		atomic_store_explicit(&pool->sync.epoch, new_epoch, memory_order_release);
	} else {
		pthread_mutex_lock(&pool->wake_mtx);
		atomic_store_explicit(&pool->sync.epoch, new_epoch, memory_order_release);
		pthread_cond_broadcast(&pool->wake_cv);
		pthread_mutex_unlock(&pool->wake_mtx);
	}

	long long t_wall_start = time_ns();

	tpool_drain_work(pool, 0, new_epoch);

	int		 budget_ns	  = atomic_load_explicit(&pool->sync.spin_budget_ns, memory_order_relaxed);
	uint64_t t_spin_start = time_ns();
	int		 spins		  = 0;
	while (atomic_load_explicit(&pool->sync.active_remaining, memory_order_acquire) > 0) {
		spins++;
		if (spins % TPOOL_SPIN_CHECK_EVERY == 0 && time_ns() - t_spin_start > (uint64_t)budget_ns) {
			sched_yield();
		} else {
			cpu_relax();
		}
	}

	tpool_update_spin_budget(pool, time_ns() - t_wall_start);
}