#ifndef CPU_THREADPOOL_H
#define CPU_THREADPOOL_H

#include <stddef.h>

typedef void (*tpool_chunk_fn)(int begin, int end, int tid, void *ctx);
typedef void (*tpool_worker_cleanup_fn)(void);

typedef struct tpool tpool;

tpool *tpool_create(int n_threads);
void   tpool_destroy(tpool *pool);
int	   tpool_current_tid(void);
int	   tpool_default_n_threads(void);

void tpool_parallel_for(tpool *pool, int n_items, int min_items_per_thread, tpool_chunk_fn fn,
						void *ctx);

int tpool_n_threads(const tpool *pool);

void tpool_set_worker_cleanup(tpool *pool, tpool_worker_cleanup_fn fn);

void tlocal_register(void **tls_ptr);

#endif