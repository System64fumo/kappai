#ifndef CPU_THREADPOOL_H
#include <stdbool.h>

#define CPU_THREADPOOL_H

#include <stddef.h>

typedef void (*tpool_chunk_fn)(int begin, int end, int tid, void *ctx);
typedef void (*tpool_worker_cleanup_fn)(void);

#define TPOOL_MAX_THREADS 256

#define TPOOL_PHASE_PREFILL 1u
#define TPOOL_PHASE_DECODE 2u
#define TPOOL_PHASES_ALL (TPOOL_PHASE_PREFILL | TPOOL_PHASE_DECODE)

typedef struct {
	int		 cpu;
	unsigned phases;
} tpool_member;

typedef struct tpool tpool;

tpool *tpool_create(const tpool_member *members, int n_threads);
void   tpool_destroy(tpool *pool);
int	   tpool_current_tid(void);
bool   tpool_phase_is_prefill(void);
int	   tpool_default_n_threads(void);
tpool *tpool_fetch_pool(void);
void   tpool_bind_phase(tpool *pool, unsigned phase);

void tpool_parallel_for(tpool *pool, int n_items, int min_items_per_thread, tpool_chunk_fn fn,
						void *ctx);

int tpool_n_threads(const tpool *pool);

void tpool_set_worker_cleanup(tpool *pool, tpool_worker_cleanup_fn fn);

void tlocal_register(void **tls_ptr);

void tlocal_free_all(void);

#endif
