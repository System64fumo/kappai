#define _GNU_SOURCE
#include "moe/moe_stream.h"
#include "common.h"
#include "config.h"
#include "log.h"
#include "memconfig.h"
#include "model.h"
#include "moe/moe_common.h"
#include "moe/moe_tier.h"
#include "monitor.h"
#include "profile.h"
#include "threadpool.h"

#include <errno.h>
#include <fcntl.h>
#if HAVE_MALLOC
#include <malloc.h>
#endif
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MOE_CHUNK_TARGET_BYTES (8 * 1024 * 1024)

#define PIN_COPY_MAX_WORKERS 8
#define MOE_PRELOAD_MAX_THREADS 32
#define LFRU_DECAY_INTERVAL 256
#define TIER_SWEEP_INTERVAL 8
#define TIER_MOVES_PER_SWEEP 4

static pthread_mutex_t g_fetch_pool_mtx = PTHREAD_MUTEX_INITIALIZER;

struct moe_stream_layer {
	pthread_mutex_t	 mtx;
	int				 lru_cap;
	int				 n_lru;
	moe_expert_slot *lru_slots;
	int				 n_pinned;
	moe_expert_slot *pinned_slots;
	uint32_t		*lru_freq;
	uint32_t		 decay_counter;
	int				*eid_to_lru;
	int				*eid_to_pinned;

	int *heap_idx;
	int *heap_pos;
	int	 heap_n;

	int				 tier_cap;
	moe_expert_slot *tier_slots;
	int				*eid_to_tier;
	moe_tier_state	*tstate;
	uint64_t		 step;
	uint64_t		 last_sweep;
};

struct moe_stream_cache {
	struct moe_stream_layer *layers;
	int						 n_layers;
	_Atomic uint64_t		 clock;
	_Atomic uint64_t		 stat_requests;
	_Atomic uint64_t		 stat_hits;
	_Atomic uint64_t		 stat_misses;
	_Atomic uint64_t		 stat_pin_hits;
	_Atomic uint64_t		 stat_lru_hits;
	int						 n_pinned;
	backend				   **pin_be;
	backend					*tier_be;
	moe_tier_policy			 tier_policy;
	const void				*map_base;
	size_t					 map_size;
	long					 page_size;
	int						 backing_fd;

	int	   direct_io_fd;
	size_t direct_io_align;
	size_t backing_file_size;

	int		 use_mmap;
	int		 nomap_fd;
	int		 nomap_direct_fd;
	size_t	 nomap_direct_align;
	uint64_t data_file_offset;
	uint64_t nomap_file_size;

	_Atomic uint64_t stat_direct_ok;
	_Atomic uint64_t stat_direct_fallback;
	_Atomic uint64_t stat_nomap_aligned;
	_Atomic uint64_t stat_nomap_bounce;

	_Atomic uint64_t stat_fill_nomap;
	_Atomic uint64_t stat_fill_direct_aligned;
	_Atomic uint64_t stat_fill_direct_bounce;
	_Atomic uint64_t stat_fill_direct_fallback_copy;
	_Atomic uint64_t stat_fill_memcpy;
	_Atomic uint64_t stat_fill_bytes;
};

typedef struct {
	struct model	 *model;
	moe_stream_cache *cache;
	moe_expert_slot **slots;
	int				 *layers;
	_Atomic int		  next;
	int				  n_total;
	_Atomic size_t	  total_copied;
	_Atomic int		  n_copied;
	_Atomic int		  n_err;
	_Atomic int		  first_err;
	_Atomic int		  workers_running;
} pin_copy_job;

typedef struct {
	const void *ptr;
	size_t		bytes;
} moe_preload_region;

typedef struct {
	const moe_preload_region *regions;
	size_t					  begin, end;
	long					  page_size;
} moe_preload_job;

typedef struct {
	moe_expert_slot *slot;
	void			*reuse_buf;
	size_t			 reuse_size;
	uint64_t		 gen;
	moe_expert_slot	 tmp;
	int				 k;
	int				 eid;
	int				 dep;
	status_code		 st;
	_Atomic int		 io_err;
	_Atomic int		 remaining;
	_Atomic int		 finalized;
} moe_miss_entry;

typedef struct moe_fetch_job {
	struct model	 *model;
	moe_stream_cache *cache;
	int				  layer;
	moe_miss_entry	 *misses;
	int				  n_misses;
	_Atomic int		  next;
} moe_fetch_job;

typedef struct {
	int chunk;
	int k;
} moe_op_item;

typedef struct moe_stream_op {
	struct model			  *model;
	moe_stream_cache		  *cache;
	struct moe_stream_layer	  *slayer;
	int						   layer;
	uint64_t				   now;
	int						   n_k;
	moe_expert_slot			  *out_slots;
	moe_miss_entry			  *misses;
	int						   n_misses;
	moe_fill_chunk			  *chunks;
	int						   n_chunks;
	int						  *chunk_miss_idx;
	moe_op_item				  *items;
	int						   n_items;
	status_code				   st;
	moe_stream_compute_hook_fn hook;
	void					  *hook_ctx;
} moe_stream_op;

static size_t tensor_bytes_2d(uint32_t type, int d0, int d1) {
	return ggml_row_size(type, (size_t)d0 * (size_t)d1);
}

moe_expert_bytes moe_calc_expert_bytes(const struct model *m, int layer, int eid) {
	moe_expert_bytes r = {0, 0, 0, 0};
	if (layer < 0 || layer >= m->n_layers || !m->layers[layer].experts)
		return r;
	if (eid < 0)
		eid = 0;
	if (eid >= m->moe.n_experts)
		return r;

	const struct expert_desc *e		= &m->layers[layer].experts[eid];
	int						  fused = e->gate_up_fused;
	r.gate_b = fused ? tensor_bytes_2d(e->gate_type, m->moe.moe_intermediate * 2, m->dim)
					 : tensor_bytes_2d(e->gate_type, m->moe.moe_intermediate, m->dim);
	r.up_b	 = fused ? 0 : tensor_bytes_2d(e->up_type, m->moe.moe_intermediate, m->dim);
	r.down_b = tensor_bytes_2d(e->down_type, m->dim, m->moe.moe_intermediate);
	r.total	 = r.gate_b + r.up_b + r.down_b;
	return r;
}

static inline void slot_from_expert_desc(moe_expert_slot *s, int eid, const struct expert_desc *e) {
	s->eid			 = eid;
	s->gate_w		 = e->gate_w;
	s->up_w			 = e->up_w;
	s->down_w		 = e->down_w;
	s->gate_type	 = e->gate_type;
	s->up_type		 = e->up_type;
	s->down_type	 = e->down_type;
	s->gate_up_fused = e->gate_up_fused;
	s->gate_scale	 = e->gate_scale;
	s->up_scale		 = e->up_scale;
	s->down_scale	 = e->down_scale;
	s->gate_off		 = e->gate_off;
	s->up_off		 = e->up_off;
	s->down_off		 = e->down_off;
	s->owned		 = 0;
	s->heap_buf		 = NULL;
	s->heap_size	 = 0;
	s->pinned		 = 0;
}

static void moe_direct_io_probe(moe_stream_cache *c) {
	int fd = dup(c->backing_fd);
	if (fd < 0)
		return;

	size_t align = 0;
	if (direct_io_probe_fd(fd, "moe direct-io", 0, 1, &align) != OK)
		return;

	c->direct_io_fd	   = fd;
	c->direct_io_align = align;
}

static void moe_direct_io_close(moe_stream_cache *c) {
	if (c->direct_io_fd >= 0) {
		close(c->direct_io_fd);
		c->direct_io_fd = -1;
	}
}

static void moe_nomap_open(moe_stream_cache *c, const char *path) {
	c->nomap_fd			  = -1;
	c->nomap_direct_fd	  = -1;
	c->nomap_direct_align = 0;
	c->nomap_file_size	  = 0;
	if (!path)
		return;

	c->nomap_fd = open(path, O_RDONLY);
	if (c->nomap_fd < 0) {
		WARN("moe no-mmap: could not open '%s' for expert streaming (%s)", path, strerror(errno));
		return;
	}
	posix_fadvise(c->nomap_fd, 0, 0, POSIX_FADV_RANDOM);

	struct stat nst;
	if (fstat(c->nomap_fd, &nst) == 0 && S_ISREG(nst.st_mode) && nst.st_size > 0)
		c->nomap_file_size = (uint64_t)nst.st_size;

	int fd = open(path, O_RDONLY | O_DIRECT);
	if (fd < 0) {
		DEBUG("moe no-mmap: O_DIRECT unavailable (%s), using buffered reads", strerror(errno));
		return;
	}

	size_t align = 0;
	if (direct_io_probe_fd(fd, "moe no-mmap", 0, 1, &align) != OK)
		return;

	c->nomap_direct_fd	  = fd;
	c->nomap_direct_align = align;
}

static void moe_nomap_close(moe_stream_cache *c) {
	if (c->nomap_direct_fd >= 0) {
		close(c->nomap_direct_fd);
		c->nomap_direct_fd = -1;
	}
	if (c->nomap_fd >= 0) {
		close(c->nomap_fd);
		c->nomap_fd = -1;
	}
}

static __thread void  *tl_bounce_buf   = NULL;
static __thread size_t tl_bounce_cap   = 0;
static __thread size_t tl_bounce_align = 0;

static void *bounce_buf_get(size_t align, size_t need) {
	if (tl_bounce_buf && tl_bounce_align == align && tl_bounce_cap >= need)
		return tl_bounce_buf;
	free(tl_bounce_buf);
	tl_bounce_buf = NULL;
	if (posix_memalign(&tl_bounce_buf, align, need) != 0) {
		tl_bounce_buf	= NULL;
		tl_bounce_cap	= 0;
		tl_bounce_align = 0;
		return NULL;
	}
	tl_bounce_cap	= need;
	tl_bounce_align = align;
	return tl_bounce_buf;
}

void moe_stream_thread_cleanup(void) {
	free(tl_bounce_buf);
	tl_bounce_buf	= NULL;
	tl_bounce_cap	= 0;
	tl_bounce_align = 0;
}

static int bounce_read_aligned(int fd, uint64_t aligned_off, size_t aligned_len, size_t head_slop,
							   size_t len, uint8_t *dst, size_t align) {
	void *bounce = bounce_buf_get(align, aligned_len);
	if (!bounce)
		return -1;
	ssize_t got = gguf_pread_full(fd, bounce, aligned_len, (off_t)aligned_off);
	if ((size_t)got < head_slop + len)
		return -1;
	memcpy(dst, (uint8_t *)bounce + head_slop, len);
	return 0;
}

static int aligned_pread_bounce(int fd, size_t align, uint64_t file_off, size_t len, uint8_t *dst,
								size_t backing_file_size) {
	uintptr_t am		  = ~((uintptr_t)align - 1);
	uint64_t  aligned_off = file_off & am;
	size_t	  head_slop	  = (size_t)(file_off - aligned_off);
	size_t	  aligned_len = (head_slop + len + align - 1) & ~(align - 1);

	if (backing_file_size > 0 && (size_t)aligned_off + aligned_len > backing_file_size) {
		size_t max_len = backing_file_size - (size_t)aligned_off;
		aligned_len	   = (max_len + align - 1) & ~(align - 1);
	}
	if (aligned_len < head_slop + len)
		return -1;

	return bounce_read_aligned(fd, aligned_off, aligned_len, head_slop, len, dst, align);
}

static int moe_nomap_read(moe_stream_cache *c, uint64_t file_off, size_t len, uint8_t *dst,
						  int precheck_aligned) {
	if (len == 0)
		return 0;

	if (c->nomap_direct_fd >= 0 && c->nomap_direct_align > 0) {
		size_t align = c->nomap_direct_align;

		if (precheck_aligned &&
			(size_t)gguf_pread_full(c->nomap_direct_fd, dst, len, (off_t)file_off) >= len) {
			atomic_fetch_add_explicit(&c->stat_direct_ok, 1, memory_order_relaxed);
			atomic_fetch_add_explicit(&c->stat_nomap_aligned, 1, memory_order_relaxed);
			return 0;
		}

		if (aligned_pread_bounce(c->nomap_direct_fd, align, file_off, len, dst, 0) == 0) {
			atomic_fetch_add_explicit(&c->stat_direct_ok, 1, memory_order_relaxed);
			atomic_fetch_add_explicit(&c->stat_nomap_bounce, 1, memory_order_relaxed);
			return 0;
		}
		WARN("moe_nomap_read: O_DIRECT bounce fallback at off=%llu len=%zu",
			 (unsigned long long)file_off, len);
	}

	if (c->nomap_fd < 0)
		return -1;
	atomic_fetch_add_explicit(&c->stat_direct_fallback, 1, memory_order_relaxed);
	if ((size_t)gguf_pread_full(c->nomap_fd, dst, len, (off_t)file_off) < len)
		return -1;
	posix_fadvise(c->nomap_fd, (off_t)file_off, (off_t)len, POSIX_FADV_DONTNEED);
	return 0;
}

static int moe_direct_io_read(moe_stream_cache *c, off_t file_off, size_t len, uint8_t *dst,
							  int precheck_aligned) {
	if (c->direct_io_fd < 0 || len == 0)
		return -1;

	size_t align = c->direct_io_align;

	if (precheck_aligned && (size_t)gguf_pread_full(c->direct_io_fd, dst, len, file_off) >= len)
		return 0;

	return aligned_pread_bounce(c->direct_io_fd, align, (uint64_t)file_off, len, dst,
								c->backing_file_size);
}

static int moe_push_chunks(moe_fill_chunk *out, int cap, int n, moe_stream_cache *c, uint8_t *dst,
						   uint64_t nomap_off, const uint8_t *mmap_src, size_t len,
						   int direct_io_eligible, size_t align, _Atomic int *err_flag) {
	if (len == 0)
		return n;
	size_t chunk_bytes = MOE_CHUNK_TARGET_BYTES;
	if (align > 0) {
		chunk_bytes = (chunk_bytes / align) * align;
		if (chunk_bytes == 0)
			chunk_bytes = align;
	}
	size_t off = 0;
	while (off < len) {
		if (n >= cap)
			return -1;
		size_t piece = len - off;
		if (piece > chunk_bytes)
			piece = chunk_bytes;
		uint8_t *piece_dst = dst + (ptrdiff_t)off;
		uint64_t piece_off = nomap_off + off;
		int		 aligned   = 0;
		if (align > 0) {
			aligned = ((((uintptr_t)piece_dst & (align - 1)) == 0) &&
					   ((piece_off & (align - 1)) == 0) && ((piece & (align - 1)) == 0));
		}
		out[n].cache			  = c;
		out[n].dst				  = piece_dst;
		out[n].nomap_off		  = piece_off;
		out[n].mmap_src			  = mmap_src ? mmap_src + (ptrdiff_t)off : NULL;
		out[n].len				  = piece;
		out[n].direct_io_eligible = direct_io_eligible;
		out[n].precheck_aligned	  = aligned;
		out[n].err_flag			  = err_flag;
		n++;
		off += piece;
	}
	return n;
}

static int moe_fill_chunk_do_read(moe_fill_chunk *ch) {
	moe_stream_cache *c = ch->cache;
	if (!c->use_mmap) {
		int ok = (moe_nomap_read(c, ch->nomap_off, ch->len, ch->dst, ch->precheck_aligned) == 0);
		atomic_fetch_add_explicit(&c->stat_fill_nomap, 1, memory_order_relaxed);
		atomic_fetch_add_explicit(&c->stat_fill_bytes, ch->len, memory_order_relaxed);
		return ok;
	}
	if (ch->direct_io_eligible && c->direct_io_fd >= 0) {
		int ok = (moe_direct_io_read(c, (off_t)ch->nomap_off, ch->len, ch->dst,
									 ch->precheck_aligned) == 0);
		if (ch->precheck_aligned)
			atomic_fetch_add_explicit(&c->stat_fill_direct_aligned, 1, memory_order_relaxed);
		else
			atomic_fetch_add_explicit(&c->stat_fill_direct_bounce, 1, memory_order_relaxed);
		if (!ok) {
			memcpy(ch->dst, ch->mmap_src, ch->len);
			atomic_fetch_add_explicit(&c->stat_fill_direct_fallback_copy, 1, memory_order_relaxed);
		}
		atomic_fetch_add_explicit(&c->stat_fill_bytes, ch->len, memory_order_relaxed);
		return ok;
	}
	memcpy(ch->dst, ch->mmap_src, ch->len);
	atomic_fetch_add_explicit(&c->stat_fill_memcpy, 1, memory_order_relaxed);
	atomic_fetch_add_explicit(&c->stat_fill_bytes, ch->len, memory_order_relaxed);
	return 1;
}

static void moe_fill_chunk_run(int begin, int end, int tid, void *ctx) {
	(void)tid;
	moe_fill_chunk *chunks = (moe_fill_chunk *)ctx;
	for (int i = begin; i < end; i++) {
		moe_fill_chunk *ch = &chunks[i];
		if (ch->err_flag && atomic_load_explicit(ch->err_flag, memory_order_relaxed))
			continue;

		if (!moe_fill_chunk_do_read(ch) && ch->err_flag)
			atomic_store_explicit(ch->err_flag, 1, memory_order_relaxed);
	}
}

static status_code moe_pin_prepare_slot(struct model *m, moe_stream_cache *c, int layer,
										moe_expert_slot *s, void *reuse_buf, size_t reuse_size,
										moe_fill_chunk *chunks, int *n, int cap,
										_Atomic int *err_flag) {
	if (s->heap_buf)
		return OK;
	if (!m->layers[layer].experts)
		return ERR_INVALID_ARG;

	int e = s->eid;
	if (e < 0 || e >= m->moe.n_experts)
		return ERR_INVALID_ARG;

	int				 fused	= m->layers[layer].experts[e].gate_up_fused;
	moe_expert_bytes eb		= moe_calc_expert_bytes(m, layer, e);
	size_t			 gate_b = eb.gate_b;
	size_t			 up_b	= eb.up_b;
	size_t			 down_b = eb.down_b;
	size_t			 total	= eb.total;
	if (total == 0)
		return ERR_INVALID_ARG;

	size_t	 alloc_align = c->nomap_direct_align > 0 ? c->nomap_direct_align : sizeof(void *);
	uint8_t *buf;
	if (reuse_buf && reuse_size >= total && ((uintptr_t)reuse_buf & (alloc_align - 1)) == 0) {
		buf = (uint8_t *)reuse_buf;
	} else {
		if (reuse_buf)
			free(reuse_buf);
		if (posix_memalign((void **)&buf, alloc_align, total) != 0)
			return ERR_OUT_OF_MEMORY;
	}

	s->gate_w	 = buf;
	s->up_w		 = fused ? NULL : (buf + gate_b);
	s->down_w	 = buf + gate_b + up_b;
	s->heap_buf	 = buf;
	s->heap_size = total;

	if (!c->use_mmap) {
		const struct expert_desc *ed	   = &m->layers[layer].experts[e];
		uint64_t				  base_off = c->data_file_offset;
		size_t					  align	   = c->nomap_direct_align;

		*n = moe_push_chunks(chunks, cap, *n, c, buf, base_off + ed->gate_off, NULL, gate_b, 0,
							 align, err_flag);
		if (*n >= 0 && !fused)
			*n = moe_push_chunks(chunks, cap, *n, c, buf + gate_b, base_off + ed->up_off, NULL,
								 up_b, 0, align, err_flag);
		if (*n >= 0)
			*n = moe_push_chunks(chunks, cap, *n, c, buf + gate_b + up_b, base_off + ed->down_off,
								 NULL, down_b, 0, align, err_flag);
		if (*n < 0)
			goto pin_fail;
		return OK;
	}

	{
		const uint8_t *src_gate = (const uint8_t *)m->layers[layer].experts[e].gate_w;
		const uint8_t *src_up	= (const uint8_t *)m->layers[layer].experts[e].up_w;
		const uint8_t *src_down = (const uint8_t *)m->layers[layer].experts[e].down_w;

		uintptr_t base = (uintptr_t)c->map_base;
		uintptr_t stop = base + c->map_size;

		const struct {
			const uint8_t *src;
			size_t		   len;
			uint8_t		  *dst;
		} regions[3] = {
			{src_gate, gate_b, buf},
			{fused ? NULL : src_up, up_b, buf + gate_b},
			{src_down, down_b, buf + gate_b + up_b},
		};

		if (c->backing_fd >= 0 && c->map_base) {
			for (int i = 0; i < 3; i++) {
				if (!regions[i].src || regions[i].len == 0)
					continue;
				uintptr_t a = (uintptr_t)regions[i].src;
				uintptr_t b = a + regions[i].len;
				if (a < base || b > stop || b < a)
					continue;
				if (c->direct_io_fd >= 0 && a >= base && b <= stop)
					continue;
				madvise((void *)regions[i].src, regions[i].len, MADV_WILLNEED);
			}
		}

		for (int i = 0; i < 3; i++) {
			if (!regions[i].src || regions[i].len == 0)
				continue;
			uintptr_t a = (uintptr_t)regions[i].src;
			uintptr_t b = a + regions[i].len;
			int		  direct_ok =
				(c->direct_io_fd >= 0 && c->map_base && a >= base && b <= stop && b >= a);
			uint64_t nomap_off = direct_ok ? (uint64_t)(a - base) : 0;
			*n = moe_push_chunks(chunks, cap, *n, c, regions[i].dst, nomap_off, regions[i].src,
								 regions[i].len, direct_ok, c->direct_io_align, err_flag);
			if (*n < 0)
				goto pin_fail;
		}

		if (c->map_base && c->map_size > 0) {
			for (int i = 0; i < 3; i++) {
				if (!regions[i].src || regions[i].len == 0)
					continue;
				page_span r = page_span_for(regions[i].src, regions[i].len, c->page_size);
				if (!page_span_clamp(&r, base, c->map_size) || r.len == 0)
					continue;
				madvise((void *)r.start, r.len, MADV_DONTNEED);
			}
		}

		return OK;
	}

pin_fail:
	free(buf);
	s->gate_w = s->up_w = s->down_w = s->heap_buf = NULL;
	s->heap_size								  = 0;
	return ERR_OUT_OF_MEMORY;
}

static status_code moe_pin_copy_slot(struct model *m, moe_stream_cache *c, int layer,
									 moe_expert_slot *s, void *reuse_buf, size_t reuse_size) {
	moe_fill_chunk chunks[512];
	int			   n   = 0;
	_Atomic int	   err = 0;
	status_code	   st =
		moe_pin_prepare_slot(m, c, layer, s, reuse_buf, reuse_size, chunks, &n, 512, &err);
	if (st != OK)
		return st;
	if (n == 0)
		return OK;

	moe_fill_chunk_run(0, n, 0, chunks);

	if (atomic_load_explicit(&err, memory_order_relaxed)) {
		free(s->heap_buf);
		s->heap_buf	 = NULL;
		s->heap_size = 0;
		s->gate_w = s->up_w = s->down_w = NULL;
		return ERR_IO;
	}
	return OK;
}

static void *pin_copy_worker(void *arg) {
	pin_copy_job *j = (pin_copy_job *)arg;
	for (;;) {
		int idx = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed);
		if (idx >= j->n_total)
			break;
		moe_expert_slot *s	   = j->slots[idx];
		int				 layer = j->layers[idx];
		status_code		 st	   = moe_pin_copy_slot(j->model, j->cache, layer, s, NULL, 0);
		if (st == OK) {
			atomic_store_explicit(&s->io_ready, 1, memory_order_release);
			atomic_fetch_add_explicit(&j->total_copied, s->heap_size, memory_order_relaxed);
			atomic_fetch_add_explicit(&j->n_copied, 1, memory_order_relaxed);
		} else {
			WARN("moe pin-copy: layer=%d eid=%d failed (st=%d)", layer, s->eid, st);
			s->heap_buf	 = NULL;
			s->heap_size = 0;
			s->gate_w = s->up_w = s->down_w = NULL;
			s->eid							= -1;
			atomic_store_explicit(&s->io_ready, 1, memory_order_release);
			atomic_fetch_add_explicit(&j->n_err, 1, memory_order_relaxed);
			if (atomic_load_explicit(&j->first_err, memory_order_relaxed) == OK)
				atomic_store_explicit(&j->first_err, st, memory_order_relaxed);
		}
	}
	atomic_fetch_sub_explicit(&j->workers_running, 1, memory_order_release);
	moe_stream_thread_cleanup();
	return NULL;
}

static void moe_slot_free_dev_buffers(moe_expert_slot *sl) {
	if (sl->dev_gate.owner)
		sl->dev_gate.owner->buffer_free(sl->dev_gate.owner, &sl->dev_gate);
	if (sl->dev_up.owner)
		sl->dev_up.owner->buffer_free(sl->dev_up.owner, &sl->dev_up);
	if (sl->dev_down.owner)
		sl->dev_down.owner->buffer_free(sl->dev_down.owner, &sl->dev_down);
	sl->dev_gate  = (buffer){0};
	sl->dev_up	  = (buffer){0};
	sl->dev_down  = (buffer){0};
	sl->dev_ready = 0;
}

typedef struct {
	int		  n_pinned;
	int		  lru_cap;
	backend **pin_be;
	backend	 *tier_be;
	int		  tier_cap;
	int		  tier_pool;
} moe_hcs_plan;

static int moe_backend_resident_capable(backend *b) {
	return b && !backend_has_cap(b, BCAP_IS_HOST) && b->moe_expert_ffn &&
		   b->buffer_alloc_from_host && (b->caps & BCAP_MOE_EXPERT_RESIDENT);
}

static backend *moe_pin_dest(const struct model *m, const moe_stream_cache *c, int layer, int pi,
							 int dev_upload) {
	if (!dev_upload)
		return NULL;
	backend *b = c->pin_be ? c->pin_be[pi] : model_pin_backend(m, layer, pi);
	return moe_backend_resident_capable(b) ? b : NULL;
}

static int moe_tier_part_pool(const struct model *m, int part) {
	backend *be = m->expert_parts[part].be;
	if (!m->topo || !moe_backend_resident_capable(be))
		return -1;
	int pool = mem_topology_pool_of(m->topo, be);
	return pool >= 0 && !m->topo->pools[pool].is_host ? pool : -1;
}

static int moe_pick_tier_part(const struct model *m, const config *cfg) {
	if (!cfg->moe_stream)
		return -1;
	int best		  = -1;
	int best_priority = m->host_priority;
	for (int i = 0; i < m->n_expert_parts; i++) {
		int pool = m->expert_parts[i].count > 0 ? moe_tier_part_pool(m, i) : -1;
		if (pool < 0)
			continue;
		int priority = m->expert_parts[i].priority >= 0 ? m->expert_parts[i].priority : pool;
		if (priority > best_priority) {
			best		  = i;
			best_priority = priority;
		}
	}
	return best;
}

static status_code moe_plan_hcs(const struct model *m, const config *cfg, moe_hcs_plan *p) {
	int n_exp	  = m->moe.n_experts;
	int first_moe = m->moe.first_dense_layer;
	if (first_moe < 0 || first_moe >= m->n_layers)
		first_moe = 0;
	int	   moe_layers = m->n_layers - first_moe;
	size_t per_expert = moe_calc_expert_bytes(m, first_moe, 0).total;

	int tier_part = moe_pick_tier_part(m, cfg);
	if (tier_part >= 0) {
		p->tier_pool = moe_tier_part_pool(m, tier_part);
		p->tier_cap =
			moe_tier_pool_capacity(&m->topo->pools[p->tier_pool], per_expert * (size_t)moe_layers,
								   m->expert_parts[tier_part].count);
		if (p->tier_cap > 0)
			p->tier_be = m->expert_parts[tier_part].be;
		else
			tier_part = -1;
	}

	int n_dev  = 0;
	int n_host = 0;
	for (int i = 0; i < m->n_expert_parts; i++) {
		int cnt = m->expert_parts[i].count;
		if (i == tier_part)
			continue;
		if (moe_backend_resident_capable(m->expert_parts[i].be))
			n_dev += MIN(cnt, n_exp - n_dev);
		else
			n_host += cnt;
	}
	n_host		   = MIN(n_host, n_exp - n_dev);
	int unassigned = n_exp - n_dev - n_host;

	size_t avail   = m->topo ? mem_pool_available(&m->topo->pools[0]) : get_available_memory();
	size_t budget  = moe_tier_pool_budget(avail);
	size_t layer_b = per_expert * (size_t)moe_layers;
	int	   fit	   = moe_tier_fit(budget, layer_b, n_exp);

	int pin_host = n_host;
	int lru_need = unassigned;
	if (tier_part >= 0) {
		pin_host = 0;
		lru_need += n_host;
	} else if (n_host > fit) {
		if (!cfg->moe_stream && !cfg->disable_failsafes) {
			ERROR("MoE: %d host expert(s)/layer (%.1f GB) exceed the %.1f GB RAM budget and "
				  "streaming is off -- enable --moe-stream or move experts to a device",
				  n_host, (double)(layer_b * (size_t)n_host) / (1024.0 * 1024.0 * 1024.0),
				  (double)budget / (1024.0 * 1024.0 * 1024.0));
			return ERR_OUT_OF_MEMORY;
		}
		if (cfg->moe_stream) {
			pin_host = 0;
			lru_need += n_host;
		}
	}

	int lru_budget = MAX(fit - pin_host, 0);
	int lru_cap	   = MIN(lru_need, lru_budget);
	if (lru_need > 0 && lru_cap < m->moe.n_experts_used)
		lru_cap = MIN(lru_need, m->moe.n_experts_used);
	if (lru_cap < 1)
		lru_cap = 1;
	if (lru_cap > 1024)
		lru_cap = 1024;

	p->n_pinned = n_dev + pin_host;
	p->lru_cap	= lru_cap;
	p->pin_be	= xcalloc((size_t)(p->n_pinned > 0 ? p->n_pinned : 1), sizeof(backend *));
	int idx		= 0;
	int left	= n_dev;
	for (int i = 0; i < m->n_expert_parts; i++) {
		if (i == tier_part || !moe_backend_resident_capable(m->expert_parts[i].be))
			continue;
		int cnt = MIN(m->expert_parts[i].count, left);
		for (int k = 0; k < cnt; k++)
			p->pin_be[idx++] = m->expert_parts[i].be;
		left -= cnt;
	}

	INFO("MoE placement per layer: %d on device, %d pinned in RAM, %d streamed via LRU "
		 "(cap %d); RAM budget %.1f GB",
		 n_dev, pin_host, lru_need, lru_need > 0 ? lru_cap : 0,
		 (double)budget / (1024.0 * 1024.0 * 1024.0));
	if (p->tier_be)
		INFO("MoE tiering: pool0 -> pool%d ('%s'), %d expert slot(s)/layer", p->tier_pool,
			 p->tier_be->name ? p->tier_be->name : "?", p->tier_cap);
	return OK;
}

typedef struct {
	moe_expert_slot *slot;
	moe_expert_slot	 tmp;
	backend			*be;
	int				 layer;
	status_code		 st;
	_Atomic int		 io_err;
} moe_upload_item;

static void moe_slot_to_host_fallback(struct model *m, moe_stream_cache *c, int layer,
									  moe_expert_slot *sl) {
	status_code st = moe_pin_copy_slot(m, c, layer, sl, NULL, 0);
	if (st != OK) {
		WARN("moe upload fallback: layer=%d eid=%d host pin failed (st=%d)", layer, sl->eid,
			 (int)st);
		sl->gate_w = sl->up_w = sl->down_w = NULL;
		sl->eid							   = -1;
	}
	atomic_store_explicit(&sl->io_ready, 1, memory_order_release);
}

static int moe_upload_one(struct model *m, moe_upload_item *it, size_t *dev_bytes) {
	moe_expert_slot *sl = it->slot;
	backend			*lb = it->be;
	moe_expert_bytes eb = moe_calc_expert_bytes(m, it->layer, sl->eid);
	status_code		 st = lb->buffer_alloc_from_host(lb, it->tmp.gate_w, eb.gate_b, &sl->dev_gate);
	if (st == OK && !sl->gate_up_fused)
		st = lb->buffer_alloc_from_host(lb, it->tmp.up_w, eb.up_b, &sl->dev_up);
	if (st == OK)
		st = lb->buffer_alloc_from_host(lb, it->tmp.down_w, eb.down_b, &sl->dev_down);
	if (st != OK) {
		ERROR("MoE: expert upload failed for layer %d on '%s' (st=%d) -- keeping its experts "
			  "in RAM",
			  it->layer, lb->name ? lb->name : "?", (int)st);
		moe_slot_free_dev_buffers(sl);
		return 0;
	}
	sl->dev_ready = 1;
	if (!m->use_mmap)
		sl->gate_w = sl->up_w = sl->down_w = NULL;
	atomic_store_explicit(&sl->io_ready, 1, memory_order_release);
	*dev_bytes += eb.total;
	return 1;
}

typedef struct {
	struct model	 *m;
	moe_stream_cache *c;
	moe_upload_item	 *items;
	moe_fill_chunk	 *chunks;
	int				  chunk_cap;
	void			**bufs;
	size_t			 *buf_sz;
	tpool			 *pool;
	backend			**failed;
	int				 *n_failed;
	char			 *layer_dev;
	char			 *layer_partial;
	int				 *uploaded;
	size_t			 *dev_bytes;
	progress		 *prog;
	int				  n_items;
	int				  n_done;
} moe_upload_ctx;

static int moe_upload_backend_failed(const moe_upload_ctx *u, const backend *b) {
	for (int f = 0; f < *u->n_failed; f++)
		if (u->failed[f] == b)
			return 1;
	return 0;
}

static void moe_upload_flush(moe_upload_ctx *u) {
	int n_chunks = 0;
	for (int j = 0; j < u->n_items; j++) {
		moe_upload_item *it = &u->items[j];
		it->tmp				= *it->slot;
		int start			= n_chunks;
		it->st = moe_pin_prepare_slot(u->m, u->c, it->layer, &it->tmp, u->bufs[j], u->buf_sz[j],
									  u->chunks, &n_chunks, u->chunk_cap, &it->io_err);
		if (it->tmp.heap_buf) {
			u->bufs[j]	 = it->tmp.heap_buf;
			u->buf_sz[j] = it->tmp.heap_size;
		} else if (it->st != ERR_INVALID_ARG) {
			u->bufs[j]	 = NULL;
			u->buf_sz[j] = 0;
		}
		if (it->st != OK)
			n_chunks = start;
	}
	if (n_chunks > 0) {
		if (u->pool && pthread_mutex_trylock(&g_fetch_pool_mtx) == 0) {
			tpool_parallel_for(u->pool, n_chunks, 1, moe_fill_chunk_run, u->chunks);
			pthread_mutex_unlock(&g_fetch_pool_mtx);
		} else {
			moe_fill_chunk_run(0, n_chunks, 0, u->chunks);
		}
	}
	for (int j = 0; j < u->n_items; j++, u->n_done++) {
		moe_upload_item *it = &u->items[j];
		progress_update(u->prog, (uint64_t)u->n_done);
		int ok = it->st == OK && !atomic_load_explicit(&it->io_err, memory_order_relaxed);
		if (ok)
			ok = moe_upload_one(u->m, it, u->dev_bytes);
		if (ok) {
			(*u->uploaded)++;
			u->layer_dev[it->layer] = 1;
			continue;
		}
		if (*u->n_failed < 16)
			u->failed[(*u->n_failed)++] = it->be;
		u->layer_partial[it->layer] = 1;
		moe_slot_to_host_fallback(u->m, u->c, it->layer, it->slot);
	}
	u->n_items = 0;
}

static status_code moe_upload_resident(struct model *m, moe_stream_cache *c) {
	int n_total = 0;
	for (int i = 0; i < m->n_layers; i++)
		for (int pi = 0; pi < c->layers[i].n_pinned; pi++)
			n_total += moe_pin_dest(m, c, i, pi, 1) != NULL;
	if (n_total == 0)
		return OK;

	int first_moe = m->moe.first_dense_layer;
	if (first_moe < 0 || first_moe >= m->n_layers)
		first_moe = 0;
	size_t max_expert = 0;
	for (int i = first_moe; i < m->n_layers; i++)
		max_expert = MAX(max_expert, moe_calc_expert_bytes(m, i, 0).total);
	if (max_expert == 0)
		return ERR_INVALID_ARG;

	int				window = (int)MIN((size_t)16, MAX((size_t)1, ((size_t)512 << 20) / max_expert));
	size_t			chunk_cap	   = (size_t)window * (max_expert / MOE_CHUNK_TARGET_BYTES + 6);
	moe_fill_chunk *chunks		   = xmalloc(chunk_cap * sizeof(*chunks));
	moe_upload_item *items		   = xcalloc((size_t)window, sizeof(*items));
	void		   **bufs		   = xcalloc((size_t)window, sizeof(void *));
	size_t			*buf_sz		   = xcalloc((size_t)window, sizeof(size_t));
	char			*layer_dev	   = xcalloc((size_t)m->n_layers, 1);
	char			*layer_partial = xcalloc((size_t)m->n_layers, 1);

	tpool *pool = tpool_fetch_pool();

	backend *failed[16] = {0};
	int		 n_failed	= 0;
	int		 uploaded	= 0;
	size_t	 dev_bytes	= 0;
	uint64_t t0			= time_us();

	progress prog;
	progress_start(&prog, "Uploading experts", (uint64_t)n_total);

	moe_upload_ctx uc = {
		.m			   = m,
		.c			   = c,
		.items		   = items,
		.chunks		   = chunks,
		.chunk_cap	   = (int)chunk_cap,
		.bufs		   = bufs,
		.buf_sz		   = buf_sz,
		.pool		   = pool,
		.failed		   = failed,
		.n_failed	   = &n_failed,
		.layer_dev	   = layer_dev,
		.layer_partial = layer_partial,
		.uploaded	   = &uploaded,
		.dev_bytes	   = &dev_bytes,
		.prog		   = &prog,
	};
	for (int i = 0; i < m->n_layers; i++) {
		for (int pi = 0; pi < c->layers[i].n_pinned; pi++) {
			backend *lb = moe_pin_dest(m, c, i, pi, 1);
			if (!lb)
				continue;
			moe_expert_slot *sl = &c->layers[i].pinned_slots[pi];
			if (moe_upload_backend_failed(&uc, lb)) {
				moe_slot_to_host_fallback(m, c, i, sl);
				layer_partial[i] = 1;
				uc.n_done++;
				continue;
			}
			moe_upload_item *it = &items[uc.n_items++];
			it->slot			= sl;
			it->be				= lb;
			it->layer			= i;
			atomic_init(&it->io_err, 0);
			if (uc.n_items == window)
				moe_upload_flush(&uc);
		}
	}
	moe_upload_flush(&uc);
	progress_finish(&prog);

	int moe_layers	= 0;
	int full_layers = 0;
	for (int i = 0; i < m->n_layers; i++) {
		if (c->layers[i].n_pinned <= 0)
			continue;
		moe_layers++;
		if (layer_dev[i])
			m->layers[i].experts_resident = 1;
		full_layers += !layer_partial[i];
	}
	if (uploaded > 0) {
		double mb	= (double)dev_bytes / (1024.0 * 1024.0);
		double secs = (double)(time_us() - t0) / 1e6;
		if (full_layers != moe_layers)
			WARN("MoE: %d expert(s) device-resident (%.1f MB) in %.1f s -- some experts fell "
				 "back to RAM",
				 uploaded, mb, secs);
		else
			INFO("MoE: %d experts resident on device (%.1f MB) in %.1f s, no host copy kept",
				 uploaded, mb, secs);
		m->moe.experts_resident = (full_layers == moe_layers);
	}

	for (int j = 0; j < window; j++)
		free(bufs[j]);
	free(bufs);
	free(buf_sz);
	free(items);
	free(chunks);
	free(layer_dev);
	free(layer_partial);
	return OK;
}

status_code moe_stream_cache_init(struct model *m) {
	if (m->moe_cache) {
		return OK;
	}
	if (!m->arch_info->is_moe) {
		WARN("moe_stream: arch '%s' is not MoE; cache is a no-op", m->arch_info->gguf_name);
		return OK;
	}
	const config *cfg			= config_get();
	int			  full_resident = !m->use_mmap && cfg->moe_stream == 0;
	int			  hcs_parts		= m->n_expert_parts > 0;
	int			  dev_upload	= hcs_parts || (full_resident && m->backend);
	m->moe_stream_enabled		= (cfg->moe_stream != 0) || !m->use_mmap || hcs_parts;
	moe_hcs_plan plan			= {0};
	if (hcs_parts) {
		status_code ps = moe_plan_hcs(m, cfg, &plan);
		if (ps != OK)
			return ps;
	}

#if HAVE_MALLOC
	if (!m->use_mmap) {
		mallopt(M_MMAP_THRESHOLD, 128 * 1024);
		mallopt(M_TRIM_THRESHOLD, 128 * 1024);
		DEBUG("moe no-mmap: heap pinned at 128 KB");
	}
#endif

	moe_stream_cache *c = xcalloc(1, sizeof(*c));
	c->n_layers			= m->n_layers;
	c->layers			= xcalloc(m->n_layers, sizeof(*c->layers));
	int lru_cap			= cfg->moe_cache_cap > 0 ? cfg->moe_cache_cap : MOE_DEFAULT_CACHE_CAP;
	if (hcs_parts)
		lru_cap = plan.lru_cap;
	if (lru_cap > 1024)
		lru_cap = 1024;
	if (m->moe.n_experts_used > 0 && lru_cap < m->moe.n_experts_used) {
		DEBUG("moe_stream: --moe-cache %d is below topk=%d (will evict every step)", lru_cap,
			  m->moe.n_experts_used);
	}

	c->n_pinned = hcs_parts ? plan.n_pinned : (full_resident ? m->moe.n_experts : cfg->moe_pin);
	c->pin_be	= plan.pin_be;
	c->tier_be	= plan.tier_be;
	if (c->tier_be)
		c->tier_policy = moe_tier_policy_default();
	if (c->n_pinned < 0)
		c->n_pinned = 0;
	if (c->n_pinned > m->moe.n_experts)
		c->n_pinned = m->moe.n_experts;

	int *pin_list	= xmalloc((size_t)m->moe.n_experts * sizeof(int));
	int	 n_pin_list = 0;
	if (!full_resident && !hcs_parts && cfg->moe_pin_list && cfg->moe_pin_list[0]) {
		const char *p = cfg->moe_pin_list;
		while (*p) {
			const char *seg_start = p;
			char	   *parse_end = NULL;
			long		eid		  = strtol(seg_start, &parse_end, 10);

			const char *seg_stop = seg_start;
			while (*seg_stop && *seg_stop != ',')
				seg_stop++;
			size_t seg_len = (size_t)(seg_stop - seg_start);
			p			   = (*seg_stop == ',') ? seg_stop + 1 : seg_stop;

			if (seg_len == 0 || parse_end == seg_start) {
				WARN("--moe-pin-list: empty segment ignored");
				continue;
			}
			for (const char *q = parse_end; q < seg_stop; q++) {
				if (*q != ' ' && *q != '\t') {
					WARN("--moe-pin-list: trailing junk in segment \"%.*s\" (parsed id %ld)",
						 (int)seg_len, seg_start, eid);
					break;
				}
			}
			if (eid >= 0 && eid < m->moe.n_experts) {
				if (n_pin_list < m->moe.n_experts)
					pin_list[n_pin_list++] = (int)eid;
				else
					WARN("--moe-pin-list: more entries than n_experts (%d), ignoring rest",
						 m->moe.n_experts);
			} else {
				WARN("--moe-pin-list: expert id %ld out of range [0,%d), skipping", eid,
					 m->moe.n_experts);
			}
		}
		c->n_pinned = n_pin_list;
	}

	c->map_base	  = m->gctx.map;
	c->map_size	  = m->gctx.map_size;
	c->backing_fd = m->gctx.fd;
	c->page_size  = sysconf(_SC_PAGESIZE);
	if (c->page_size <= 0)
		c->page_size = 4096;
	c->direct_io_fd		  = -1;
	c->direct_io_align	  = 0;
	c->backing_file_size  = m->gctx.map_size;
	c->use_mmap			  = m->use_mmap;
	c->data_file_offset	  = m->gctx.data_file_offset;
	c->nomap_fd			  = -1;
	c->nomap_direct_fd	  = -1;
	c->nomap_direct_align = 0;
	if (!c->use_mmap) {
		moe_nomap_open(c, m->model_path);
	} else {
		moe_direct_io_probe(c);
	}
	atomic_store_explicit(&c->clock, 1, memory_order_release);

	for (int i = 0; i < m->n_layers; i++) {
		struct moe_stream_layer *L = &c->layers[i];
		pthread_mutex_init(&L->mtx, NULL);
		L->lru_cap		 = lru_cap;
		L->n_lru		 = 0;
		L->lru_slots	 = xcalloc(lru_cap, sizeof(moe_expert_slot));
		L->lru_freq		 = xcalloc(lru_cap, sizeof(uint32_t));
		L->decay_counter = 0;
		L->eid_to_lru	 = xmalloc((size_t)m->moe.n_experts * sizeof(int));
		L->eid_to_pinned = xmalloc((size_t)m->moe.n_experts * sizeof(int));
		L->heap_idx		 = xmalloc((size_t)lru_cap * sizeof(int));
		L->heap_pos		 = xmalloc((size_t)lru_cap * sizeof(int));
		L->heap_n		 = 0;
		for (int s = 0; s < lru_cap; s++)
			L->heap_pos[s] = -1;
		for (int e = 0; e < m->moe.n_experts; e++)
			L->eid_to_lru[e] = -1;
		for (int e = 0; e < m->moe.n_experts; e++)
			L->eid_to_pinned[e] = -1;
		for (int s = 0; s < lru_cap; s++) {
			L->lru_slots[s].eid		  = -1;
			L->lru_slots[s].pinned	  = 0;
			L->lru_slots[s].heap_buf  = NULL;
			L->lru_slots[s].heap_size = 0;
			atomic_init(&L->lru_slots[s].io_ready, 1);
			atomic_init(&L->lru_slots[s].inuse, 0);
		}

		if (c->tier_be) {
			L->tstate	   = xcalloc((size_t)m->moe.n_experts, sizeof(*L->tstate));
			L->eid_to_tier = xmalloc((size_t)m->moe.n_experts * sizeof(int));
			for (int e = 0; e < m->moe.n_experts; e++)
				L->eid_to_tier[e] = -1;
			if (m->layers[i].experts) {
				L->tier_cap	  = plan.tier_cap;
				L->tier_slots = xcalloc((size_t)L->tier_cap, sizeof(moe_expert_slot));
				for (int t = 0; t < L->tier_cap; t++) {
					L->tier_slots[t].eid = -1;
					atomic_init(&L->tier_slots[t].io_ready, 1);
					atomic_init(&L->tier_slots[t].inuse, 0);
				}
				m->layers[i].experts_resident = 1;
			}
		}

		int n_pin_this_layer = c->n_pinned;

		L->n_pinned		= 0;
		L->pinned_slots = NULL;
		if (n_pin_this_layer > 0 && m->layers[i].experts) {
			L->pinned_slots = xcalloc(n_pin_this_layer, sizeof(moe_expert_slot));
			for (int pi = 0; pi < n_pin_this_layer; pi++) {
				int e = (n_pin_list > 0) ? pin_list[pi] : pi;
				if (e < 0 || e >= m->moe.n_experts)
					continue;
				moe_expert_slot *s = &L->pinned_slots[L->n_pinned++];
				if (L->eid_to_pinned[e] < 0)
					L->eid_to_pinned[e] = (int)(s - L->pinned_slots);
				slot_from_expert_desc(s, e, &m->layers[i].experts[e]);
				s->last_used = 0;
				s->pinned	 = 1;
				s->heap_buf	 = NULL;
				s->heap_size = 0;
				atomic_init(&s->inuse, 0);
				atomic_init(&s->io_ready, 0);
			}
		}
	}
	free(pin_list);
	m->moe_cache = c;

	{
		int first_moe = m->moe.first_dense_layer;
		if (first_moe < 0)
			first_moe = 0;
		if (first_moe >= m->n_layers)
			first_moe = 0;
		if (m->n_layers > 0 && m->layers[first_moe].experts) {
			int	   moe_layers = m->n_layers - first_moe;
			size_t per_expert = moe_calc_expert_bytes(m, first_moe, 0).total;
			size_t total	  = per_expert * (size_t)m->moe.n_experts * (size_t)moe_layers;
			double total_gb	  = total / (1024.0 * 1024.0 * 1024.0);
			INFO("MoE: %.2f GB total, mode=%s, cache lru=%d, pinned=%d/layer", total_gb,
				 full_resident ? "full-resident" : (m->moe_stream_enabled ? "streaming" : "direct"),
				 lru_cap, c->n_pinned);
		}
	}

	if (m->use_mmap && m->gctx.map && m->gctx.map_size > 0) {
		madvise(m->gctx.map, m->gctx.map_size, MADV_RANDOM);
#ifdef MADV_DONTDUMP
		madvise(m->gctx.map, m->gctx.map_size, MADV_DONTDUMP);
#endif
	}

	{
		int n_total = 0;
		for (int i = 0; i < m->n_layers; i++)
			for (int pi = 0; pi < c->layers[i].n_pinned; pi++)
				n_total += moe_pin_dest(m, c, i, pi, dev_upload) == NULL;

		if (n_total > 0) {
			moe_expert_slot **slots = (moe_expert_slot **)xmalloc((size_t)n_total * sizeof(*slots));
			int				 *layers = (int *)xmalloc((size_t)n_total * sizeof(int));
			int				  idx	 = 0;
			for (int i = 0; i < m->n_layers; i++) {
				for (int pi = 0; pi < c->layers[i].n_pinned; pi++) {
					if (moe_pin_dest(m, c, i, pi, dev_upload))
						continue;
					slots[idx]	= &c->layers[i].pinned_slots[pi];
					layers[idx] = i;
					idx++;
				}
			}

			int nw = 4;
			if (nw > PIN_COPY_MAX_WORKERS)
				nw = PIN_COPY_MAX_WORKERS;
			if (nw > n_total)
				nw = n_total;

			pin_copy_job job;
			job.model	= m;
			job.cache	= c;
			job.slots	= slots;
			job.layers	= layers;
			job.n_total = n_total;
			atomic_store(&job.next, 0);
			atomic_store(&job.total_copied, (size_t)0);
			atomic_store(&job.n_copied, 0);
			atomic_store(&job.n_err, 0);
			atomic_store(&job.first_err, OK);

			uint64_t t0 = time_us();

			if (g_monitor && g_monitor->fd >= 0) {
				monitor_emit_load_pin_copy_start(g_monitor, n_total, nw);
			}

			pthread_t th[PIN_COPY_MAX_WORKERS];
			atomic_store(&job.workers_running, nw);
			int started = 0;
			for (; started < nw; started++) {
				if (pthread_create(&th[started], NULL, pin_copy_worker, &job) != 0)
					break;
			}
			atomic_fetch_sub(&job.workers_running, nw - started);
			if (started == 0) {
				atomic_store(&job.workers_running, 1);
				pin_copy_worker(&job);
			}

			progress prog;
			progress_start(&prog, "Pinning experts", (uint64_t)n_total);
			while (atomic_load_explicit(&job.workers_running, memory_order_acquire) > 0) {
				progress_update(&prog,
								(uint64_t)(atomic_load(&job.n_copied) + atomic_load(&job.n_err)));
				usleep(20000);
			}
			progress_finish(&prog);
			for (int i = 0; i < started; i++)
				pthread_join(th[i], NULL);

			uint64_t elapsed_us = time_us() - t0;
			size_t	 copied		= atomic_load(&job.total_copied);
			int		 n_copied	= atomic_load(&job.n_copied);
			double	 mb			= (double)copied / (1024.0 * 1024.0);
			double	 secs		= (double)elapsed_us / 1e6;
			double	 mbps		= secs > 0 ? mb / secs : 0;

			INFO("moe pin-copy: %d experts, %.1f MB in %.1f s (%.0f MB/s)", n_copied, mb, secs,
				 mbps);

			if (g_monitor && g_monitor->fd >= 0) {
				monitor_emit_load_pin_copy_done(g_monitor, n_copied, mb, elapsed_us / 1000);
			}

			free((void *)slots);
			free((void *)layers);
		}
	}
	if (dev_upload) {
		status_code us = moe_upload_resident(m, c);
		if (us != OK)
			return us;
	}

	return OK;
}

void moe_stream_cache_free(moe_stream_cache *c) {
	if (!c)
		return;

	for (int i = 0; i < c->n_layers; i++) {
		for (int s = 0; s < c->layers[i].n_pinned; s++)
			moe_slot_free_dev_buffers(&c->layers[i].pinned_slots[s]);
		for (int s = 0; s < c->layers[i].n_pinned; s++) {
			free(c->layers[i].pinned_slots[s].heap_buf);
		}
		for (int s = 0; s < c->layers[i].n_lru; s++) {
			free(c->layers[i].lru_slots[s].heap_buf);
		}
		for (int s = 0; s < c->layers[i].tier_cap; s++)
			moe_slot_free_dev_buffers(&c->layers[i].tier_slots[s]);
		free(c->layers[i].tier_slots);
		free(c->layers[i].eid_to_tier);
		free(c->layers[i].tstate);
		free(c->layers[i].lru_slots);
		free(c->layers[i].lru_freq);
		free(c->layers[i].pinned_slots);
		free(c->layers[i].eid_to_lru);
		free(c->layers[i].eid_to_pinned);
		free(c->layers[i].heap_idx);
		free(c->layers[i].heap_pos);
		pthread_mutex_destroy(&c->layers[i].mtx);
	}
	free(c->layers);
	free(c->pin_be);
	moe_direct_io_close(c);
	moe_nomap_close(c);
	free(c);
}

static void fault_hint(const moe_stream_cache *c, const void *ptr, size_t bytes) {
	if (!c->use_mmap)
		return;
	if (!c->map_base || c->map_size == 0 || !ptr || bytes == 0)
		return;
	page_span r = page_span_for(ptr, bytes, c->page_size);
	if (!page_span_clamp(&r, (uintptr_t)c->map_base, c->map_size) || r.len == 0)
		return;
	madvise((void *)r.start, r.len, MADV_WILLNEED);
}

static void fault_wait(const void *ptr, size_t bytes, long page_size) {
	page_span r = page_span_for(ptr, bytes, (size_t)page_size);
	if (r.len == 0)
		return;
	volatile uint8_t acc = 0;
	for (uintptr_t p = r.start; p < r.start + r.len; p += (uintptr_t)page_size) {
		acc |= *((const volatile uint8_t *)p);
	}
	(void)acc;
}

static void drop_pages(const moe_stream_cache *c, const void *ptr, size_t bytes) {
	if (!c->map_base || c->map_size == 0 || !ptr || bytes == 0)
		return;
	page_span r = page_span_for(ptr, bytes, c->page_size);
	if (!page_span_clamp(&r, (uintptr_t)c->map_base, c->map_size) || r.len == 0)
		return;
	madvise((void *)r.start, r.len, MADV_DONTNEED);
}

static inline uint64_t lfru_score(uint32_t freq, uint64_t last_used, uint64_t clock) {
	uint64_t age	= clock - last_used;
	uint32_t recent = (age < 256) ? (256 - (uint32_t)age) : 0;
	return ((uint64_t)freq << 8) | recent;
}

static inline uint64_t lfru_score_idx(struct moe_stream_layer *sl, int idx, uint64_t now) {
	return lfru_score(sl->lru_freq[idx], sl->lru_slots[idx].last_used, now);
}

static void heap_swap(struct moe_stream_layer *sl, int a, int b) {
	int ia = sl->heap_idx[a], ib = sl->heap_idx[b];
	sl->heap_idx[a]	 = ib;
	sl->heap_idx[b]	 = ia;
	sl->heap_pos[ia] = b;
	sl->heap_pos[ib] = a;
}

static void heap_sift_down(struct moe_stream_layer *sl, int pos, uint64_t now) {
	int n = sl->heap_n;
	for (;;) {
		int l = 2 * pos + 1, r = 2 * pos + 2, smallest = pos;
		if (l < n && lfru_score_idx(sl, sl->heap_idx[l], now) <
						 lfru_score_idx(sl, sl->heap_idx[smallest], now))
			smallest = l;
		if (r < n && lfru_score_idx(sl, sl->heap_idx[r], now) <
						 lfru_score_idx(sl, sl->heap_idx[smallest], now))
			smallest = r;
		if (smallest == pos)
			return;
		heap_swap(sl, pos, smallest);
		pos = smallest;
	}
}

static void heap_sift_up(struct moe_stream_layer *sl, int pos, uint64_t now) {
	while (pos > 0) {
		int parent = (pos - 1) / 2;
		if (lfru_score_idx(sl, sl->heap_idx[parent], now) <=
			lfru_score_idx(sl, sl->heap_idx[pos], now))
			return;
		heap_swap(sl, pos, parent);
		pos = parent;
	}
}

static void heap_touch(struct moe_stream_layer *sl, int idx, uint64_t now) {
	int pos = sl->heap_pos[idx];
	if (pos < 0 || pos >= sl->heap_n)
		return;
	heap_sift_down(sl, pos, now);
	heap_sift_up(sl, pos, now);
}

static void heap_push(struct moe_stream_layer *sl, int idx, uint64_t now) {
	int pos			  = sl->heap_n++;
	sl->heap_idx[pos] = idx;
	sl->heap_pos[idx] = pos;
	heap_sift_up(sl, pos, now);
}

static moe_expert_slot *layer_find_tier(struct moe_stream_layer *sl, int eid) {
	if (!sl->eid_to_tier)
		return NULL;
	int ti = sl->eid_to_tier[eid];
	if (ti >= 0 && ti < sl->tier_cap && sl->tier_slots[ti].eid == eid)
		return &sl->tier_slots[ti];
	return NULL;
}

static moe_expert_slot *layer_find(struct moe_stream_layer *sl, int eid, uint64_t now) {
	if (eid >= 0 && sl->eid_to_pinned) {
		int pi = sl->eid_to_pinned[eid];
		if (pi >= 0 && pi < sl->n_pinned && sl->pinned_slots[pi].eid == eid) {
			sl->pinned_slots[pi].last_used = now;
			return &sl->pinned_slots[pi];
		}
	}
	moe_expert_slot *tier = layer_find_tier(sl, eid);
	if (tier) {
		tier->last_used = now;
		return tier;
	}
	int i = sl->eid_to_lru[eid];
	if (i >= 0 && i < sl->n_lru && sl->lru_slots[i].eid == eid) {
		sl->lru_slots[i].last_used = now;
		if (sl->lru_freq[i] < UINT32_MAX)
			sl->lru_freq[i]++;
		heap_touch(sl, i, now);
		return &sl->lru_slots[i];
	}
	return NULL;
}

static moe_expert_slot *layer_insert_lru(struct model *m, moe_stream_cache *c, int layer,
										 struct moe_stream_layer *sl, int eid, uint64_t now,
										 void **out_freed_buf, size_t *out_freed_size) {
	int idx;
	if (sl->n_lru < sl->lru_cap) {
		idx = sl->n_lru++;
		heap_push(sl, idx, now);
	} else {
		int worst_idx = -1;
		int requeue[1024];
		int n_requeue	= 0;
		int requeue_cap = sl->lru_cap < 1024 ? sl->lru_cap : 1024;

		while (sl->heap_n > 0) {
			int cand = sl->heap_idx[0];
			if (atomic_load_explicit(&sl->lru_slots[cand].inuse, memory_order_acquire) > 0) {
				if (n_requeue < requeue_cap)
					requeue[n_requeue++] = cand;
				heap_swap(sl, 0, sl->heap_n - 1);
				sl->heap_n--;
				heap_sift_down(sl, 0, now);
				continue;
			}
			worst_idx = cand;
			break;
		}

		for (int i = 0; i < n_requeue; i++)
			heap_push(sl, requeue[i], now);

		if (worst_idx < 0)
			return NULL;

		idx = worst_idx;
		heap_swap(sl, sl->heap_pos[idx], sl->heap_n - 1);
		sl->heap_n--;
		heap_sift_down(sl, sl->heap_pos[sl->heap_idx[0]], now);

		atomic_fetch_add_explicit(&sl->lru_slots[idx].gen, 1, memory_order_acq_rel);

		moe_expert_slot *victim = &sl->lru_slots[idx];
		if (victim->eid >= 0 && sl->eid_to_lru[victim->eid] == idx)
			sl->eid_to_lru[victim->eid] = -1;
		if (victim->heap_buf) {
			if (out_freed_buf) {
				*out_freed_buf	= victim->heap_buf;
				*out_freed_size = victim->heap_size;
			} else {
				free(victim->heap_buf);
			}
		} else if (victim->eid >= 0 && m && c) {
			moe_expert_bytes eb = moe_calc_expert_bytes(m, layer, victim->eid);
			drop_pages(c, victim->gate_w, eb.gate_b);
			if (!victim->gate_up_fused)
				drop_pages(c, victim->up_w, eb.up_b);
			drop_pages(c, victim->down_w, eb.down_b);
		}
		sl->lru_slots[idx].heap_buf	 = NULL;
		sl->lru_slots[idx].heap_size = 0;
		heap_push(sl, idx, now);
	}
	moe_expert_slot *s = &sl->lru_slots[idx];
	s->eid			   = eid;
	s->last_used	   = now;
	s->pinned		   = 0;
	s->heap_buf		   = NULL;
	s->heap_size	   = 0;
	atomic_store_explicit(&s->io_ready, 0, memory_order_relaxed);
	sl->eid_to_lru[eid] = idx;
	sl->lru_freq[idx]	= 1;
	heap_touch(sl, idx, now);
	sl->decay_counter++;
	if (sl->decay_counter >= LFRU_DECAY_INTERVAL) {
		sl->decay_counter = 0;
		for (int i = 0; i < sl->n_lru; i++)
			sl->lru_freq[i] >>= 1;
		for (int i = 0; i < sl->n_lru; i++)
			heap_touch(sl, i, now);
	}
	return s;
}

static void slot_zero(moe_expert_slot *s) {
	s->gate_w	 = NULL;
	s->up_w		 = NULL;
	s->down_w	 = NULL;
	s->last_used = 0;
	s->heap_buf	 = NULL;
	s->heap_size = 0;
	atomic_init(&s->gen, 0);
	s->gate_off		 = 0;
	s->up_off		 = 0;
	s->down_off		 = 0;
	s->gate_type	 = 0;
	s->up_type		 = 0;
	s->down_type	 = 0;
	s->dev_gate		 = (buffer){0};
	s->dev_up		 = (buffer){0};
	s->dev_down		 = (buffer){0};
	s->dev_ready	 = 0;
	s->eid			 = -1;
	s->gate_up_fused = 0;
	s->gate_scale	 = 0.0f;
	s->up_scale		 = 0.0f;
	s->down_scale	 = 0.0f;
	s->pinned		 = 0;
	atomic_init(&s->io_ready, 0);
	atomic_init(&s->inuse, 0);
	s->owned = 0;
}

static void slot_mark_invalid(moe_expert_slot *out) {
	slot_zero(out);
	out->eid = -1;
	atomic_store_explicit(&out->io_ready, 1, memory_order_release);
}

static void slot_take_ready(moe_expert_slot *out, moe_expert_slot *s) {
	*out	   = *s;
	out->owned = 0;
	atomic_fetch_add_explicit(&s->inuse, 1, memory_order_acq_rel);
	atomic_store_explicit(&out->io_ready, 1, memory_order_release);
}

static int layer_take_ready_stats(moe_stream_cache *c, moe_expert_slot *s, moe_expert_slot *dst) {
	if (!s || !atomic_load_explicit(&s->io_ready, memory_order_acquire))
		return 0;
	slot_take_ready(dst, s);
	atomic_fetch_add_explicit(&c->stat_hits, 1, memory_order_relaxed);
	if (s->pinned)
		atomic_fetch_add_explicit(&c->stat_pin_hits, 1, memory_order_relaxed);
	else
		atomic_fetch_add_explicit(&c->stat_lru_hits, 1, memory_order_relaxed);
	return 1;
}

static void miss_fill_desc(moe_miss_entry *me, int k, int eid, void *freed_buf, size_t freed_size,
						   const struct expert_desc *desc) {
	moe_expert_slot tmp;
	slot_zero(&tmp);
	slot_from_expert_desc(&tmp, eid, desc);
	tmp.heap_buf  = NULL;
	tmp.heap_size = 0;
	me->k		  = k;
	me->eid		  = eid;
	me->dep		  = -1;
	me->slot	  = NULL;
	me->gen		  = 0;
	me->tmp		  = tmp;
	me->st		  = OK;
	atomic_store_explicit(&me->io_err, 0, memory_order_relaxed);
	me->reuse_buf  = freed_buf;
	me->reuse_size = freed_size;
}

static void miss_fill_slot(moe_miss_entry *me, int k, int eid, moe_expert_slot *s, uint64_t gen,
						   void *freed_buf, size_t freed_size) {
	me->k			  = k;
	me->eid			  = eid;
	me->dep			  = -1;
	me->slot		  = s;
	me->gen			  = gen;
	me->tmp			  = *s;
	me->tmp.heap_buf  = NULL;
	me->tmp.heap_size = 0;
	me->st			  = OK;
	atomic_store_explicit(&me->io_err, 0, memory_order_relaxed);
	me->reuse_buf  = freed_buf;
	me->reuse_size = freed_size;
}

static void moe_resolve_no_cache(const struct model *m, const struct layer_weights *Lw,
								 const int *expert_ids, int n_k, moe_expert_slot *out_slots) {
	for (int k = 0; k < n_k; k++) {
		int eid = expert_ids[k];
		if (eid < 0 || eid >= m->moe.n_experts) {
			slot_mark_invalid(&out_slots[k]);
			continue;
		}
		slot_from_expert_desc(&out_slots[k], eid, &Lw->experts[eid]);
		out_slots[k].last_used = 0;
		atomic_store_explicit(&out_slots[k].io_ready, 1, memory_order_relaxed);
	}
}

static size_t moe_miss_chunk_budget(const struct model *m, int layer, const moe_miss_entry *misses,
									int n_misses) {
	size_t need = 0;
	for (int i = 0; i < n_misses; i++) {
		int eid = misses[i].eid;
		if (eid < 0 || eid >= m->moe.n_experts)
			continue;
		moe_expert_bytes eb = moe_calc_expert_bytes(m, layer, eid);
		need += (eb.total / (size_t)MOE_CHUNK_TARGET_BYTES) + 3;
	}
	if (need < 16)
		need = 16;
	return need;
}

static status_code miss_entry_status(const moe_miss_entry *me) {
	status_code st = me->st;
	if (st == OK && atomic_load_explicit(&me->io_err, memory_order_relaxed))
		st = ERR_IO;
	return st;
}

static void miss_commit(moe_expert_slot *out_slot, moe_miss_entry *me, struct moe_stream_layer *L,
						int layer, const char *tag) {
	status_code		 fetch_st = me->st;
	moe_expert_slot *s		  = me->slot;
	uint64_t		 my_gen	  = me->gen;
	moe_expert_slot *tmp	  = &me->tmp;

	if (fetch_st == OK && atomic_load_explicit(&me->io_err, memory_order_relaxed))
		fetch_st = ERR_IO;

	pthread_mutex_lock(&L->mtx);
	if (fetch_st != OK || !tmp->heap_buf) {
		if (tmp->heap_buf) {
			free(tmp->heap_buf);
			tmp->heap_buf = NULL;
		}
		if (s && atomic_load_explicit(&s->gen, memory_order_acquire) == my_gen) {
			if (atomic_load_explicit(&s->inuse, memory_order_acquire) == 0) {
				if (!s->pinned && s->eid >= 0) {
					int idx = (int)(s - L->lru_slots);
					if (idx >= 0 && idx < L->n_lru && L->eid_to_lru[s->eid] == idx)
						L->eid_to_lru[s->eid] = -1;
					s->eid = -1;
				}
				free(s->heap_buf);
				s->heap_buf	 = NULL;
				s->heap_size = 0;
				s->gate_w = s->up_w = s->down_w = NULL;
			}
			atomic_store_explicit(&s->io_ready, 1, memory_order_release);
		}
		slot_mark_invalid(out_slot);
		if (fetch_st != OK)
			WARN("%s: fetch failed for layer=%d eid=%d (st=%d)", tag, layer, me->eid, fetch_st);
		pthread_mutex_unlock(&L->mtx);
		return;
	}
	if (s && atomic_load_explicit(&s->gen, memory_order_acquire) == my_gen &&
		atomic_load_explicit(&s->inuse, memory_order_acquire) == 0) {
		if (s->heap_buf && s->heap_buf != tmp->heap_buf)
			free(s->heap_buf);
		*s = *tmp;
		atomic_store_explicit(&s->io_ready, 1, memory_order_release);
		*out_slot		= *tmp;
		out_slot->owned = 0;
		atomic_fetch_add_explicit(&s->inuse, 1, memory_order_acq_rel);
	} else {
		*out_slot		= *tmp;
		out_slot->owned = 1;
	}
	pthread_mutex_unlock(&L->mtx);
	atomic_store_explicit(&out_slot->io_ready, 1, memory_order_release);
}

static moe_expert_slot *live_slot_find(struct moe_stream_layer *L, const moe_expert_slot *slot) {
	if (slot->eid < 0)
		return NULL;
	if (L->eid_to_pinned) {
		int pi = L->eid_to_pinned[slot->eid];
		if (pi >= 0 && pi < L->n_pinned && L->pinned_slots[pi].eid == slot->eid)
			return &L->pinned_slots[pi];
	}
	moe_expert_slot *tier = layer_find_tier(L, slot->eid);
	if (tier)
		return tier;
	int i = L->eid_to_lru[slot->eid];
	if (i >= 0 && i < L->n_lru && L->lru_slots[i].eid == slot->eid)
		return &L->lru_slots[i];
	return NULL;
}

static void live_slot_unuse(moe_expert_slot *live) {
	int prev = atomic_fetch_sub_explicit(&live->inuse, 1, memory_order_acq_rel);
	if (prev <= 0)
		atomic_store_explicit(&live->inuse, 0, memory_order_release);
}

static uint64_t tier_clock(const moe_stream_cache *c) {
	return atomic_load_explicit(&c->clock, memory_order_relaxed);
}

static float tier_score(const moe_stream_cache *c, const struct moe_stream_layer *L, int eid) {
	return moe_tier_score_at(&c->tier_policy, &L->tstate[eid].score, L->step);
}

static void tier_note_move(struct moe_stream_layer *L, int eid) {
	L->tstate[eid].last_move = L->step;
}

static void tier_note_load(const moe_stream_cache *c, struct moe_stream_layer *L, int eid) {
	if (c->tier_be)
		tier_note_move(L, eid);
}

static void tier_score_requests(const moe_stream_cache *c, struct moe_stream_layer *L,
								int n_experts, const int *expert_ids, int n_k) {
	L->step++;
	for (int k = 0; k < n_k; k++) {
		int eid = expert_ids[k];
		if (eid >= 0 && eid < n_experts)
			moe_tier_score_hit(&c->tier_policy, &L->tstate[eid].score, L->step);
	}
}

static int tier_movable(const moe_stream_cache *c, const struct moe_stream_layer *L,
						const moe_expert_slot *s) {
	return s->eid >= 0 && atomic_load_explicit(&s->io_ready, memory_order_acquire) &&
		   atomic_load_explicit(&s->inuse, memory_order_acquire) == 0 &&
		   moe_tier_may_move(&c->tier_policy, L->tstate[s->eid].last_move, L->step);
}

static void lru_slot_clear(moe_stream_cache *c, struct moe_stream_layer *L, int idx) {
	moe_expert_slot *s = &L->lru_slots[idx];
	atomic_fetch_add_explicit(&s->gen, 1, memory_order_acq_rel);
	if (s->eid >= 0 && L->eid_to_lru[s->eid] == idx)
		L->eid_to_lru[s->eid] = -1;
	s->heap_buf	 = NULL;
	s->heap_size = 0;
	s->gate_w = s->up_w = s->down_w = NULL;
	s->eid							= -1;
	s->last_used					= 0;
	L->lru_freq[idx]				= 0;
	heap_touch(L, idx, tier_clock(c));
}

static void tier_evict(moe_stream_cache *c, struct moe_stream_layer *L, int idx) {
	moe_expert_slot *s = &L->lru_slots[idx];
	tier_note_move(L, s->eid);
	free(s->heap_buf);
	lru_slot_clear(c, L, idx);
}

static int lru_slot_refill(struct model *m, moe_stream_cache *c, int layer,
						   struct moe_stream_layer *L, int idx, int eid) {
	moe_expert_slot *s		  = &L->lru_slots[idx];
	void			*reuse	  = s->heap_buf;
	size_t			 reuse_sz = s->heap_size;
	moe_expert_slot	 tmp	  = *s;
	slot_from_expert_desc(&tmp, eid, &m->layers[layer].experts[eid]);
	status_code st = moe_pin_copy_slot(m, c, layer, &tmp, reuse, reuse_sz);
	lru_slot_clear(c, L, idx);
	if (st != OK)
		return 0;
	*s				   = tmp;
	s->last_used	   = tier_clock(c);
	L->eid_to_lru[eid] = idx;
	L->lru_freq[idx]   = 1;
	atomic_store_explicit(&s->io_ready, 1, memory_order_release);
	heap_touch(L, idx, s->last_used);
	tier_note_move(L, eid);
	return 1;
}

static int tier_upload(struct model *m, moe_stream_cache *c, int layer, const moe_expert_slot *host,
					   moe_expert_slot *out) {
	slot_zero(out);
	slot_from_expert_desc(out, host->eid, &m->layers[layer].experts[host->eid]);
	moe_upload_item it	  = {.slot = out, .be = c->tier_be, .layer = layer, .tmp = *host};
	size_t			bytes = 0;
	if (moe_upload_one(m, &it, &bytes))
		return 1;
	out->eid = -1;
	return 0;
}

static void tier_release(moe_expert_slot *t) {
	moe_slot_free_dev_buffers(t);
	t->gate_w = t->up_w = t->down_w = NULL;
	t->eid							= -1;
	t->last_used					= 0;
}

static int tier_free_index(const struct moe_stream_layer *L) {
	for (int t = 0; t < L->tier_cap; t++)
		if (L->tier_slots[t].eid < 0)
			return t;
	return -1;
}

static int tier_weakest_index(const moe_stream_cache *c, const struct moe_stream_layer *L) {
	int	  weakest = -1;
	float low	  = 0.0f;
	for (int t = 0; t < L->tier_cap; t++) {
		const moe_expert_slot *ts = &L->tier_slots[t];
		if (!tier_movable(c, L, ts))
			continue;
		float score = tier_score(c, L, ts->eid);
		if (weakest < 0 || score < low) {
			weakest = t;
			low		= score;
		}
	}
	return weakest;
}

static int tier_promote(struct model *m, moe_stream_cache *c, int layer, struct moe_stream_layer *L,
						int idx) {
	moe_expert_slot *host	= &L->lru_slots[idx];
	int				 eid	= host->eid;
	int				 dst	= tier_free_index(L);
	int				 victim = -1;
	if (dst < 0) {
		victim = tier_weakest_index(c, L);
		if (victim < 0 || !moe_tier_beats(&c->tier_policy, tier_score(c, L, eid),
										  tier_score(c, L, L->tier_slots[victim].eid)))
			return 0;
		dst = victim;
	}

	moe_expert_slot placed;
	if (!tier_upload(m, c, layer, host, &placed)) {
		tier_note_move(L, eid);
		return 0;
	}

	if (victim < 0) {
		free(host->heap_buf);
		lru_slot_clear(c, L, idx);
	} else {
		int victim_eid = L->tier_slots[victim].eid;
		if (!lru_slot_refill(m, c, layer, L, idx, victim_eid)) {
			moe_slot_free_dev_buffers(&placed);
			tier_note_move(L, eid);
			return 0;
		}
		L->eid_to_tier[victim_eid] = -1;
		tier_release(&L->tier_slots[victim]);
	}

	L->tier_slots[dst]	= placed;
	L->eid_to_tier[eid] = dst;
	tier_note_move(L, eid);
	return 1;
}

static int tier_demote(struct model *m, moe_stream_cache *c, int layer, struct moe_stream_layer *L,
					   int tier_idx) {
	moe_expert_slot *t		  = &L->tier_slots[tier_idx];
	int				 eid	  = t->eid;
	void			*freed	  = NULL;
	size_t			 freed_sz = 0;
	moe_expert_slot *s = layer_insert_lru(m, c, layer, L, eid, tier_clock(c), &freed, &freed_sz);
	if (!s)
		return 0;

	int				idx = (int)(s - L->lru_slots);
	moe_expert_slot tmp = *s;
	slot_from_expert_desc(&tmp, eid, &m->layers[layer].experts[eid]);
	status_code st = moe_pin_copy_slot(m, c, layer, &tmp, freed, freed_sz);
	if (st != OK) {
		lru_slot_clear(c, L, idx);
		atomic_store_explicit(&s->io_ready, 1, memory_order_release);
		tier_note_move(L, eid);
		return 0;
	}
	*s = tmp;
	atomic_store_explicit(&s->io_ready, 1, memory_order_release);
	L->eid_to_tier[eid] = -1;
	tier_release(t);
	tier_note_move(L, eid);
	return 1;
}

static void tier_sweep(struct model *m, moe_stream_cache *c, int layer,
					   struct moe_stream_layer *L) {
	const moe_tier_policy *p	 = &c->tier_policy;
	int					   moves = 0;
	for (int t = 0; t < L->tier_cap && moves < TIER_MOVES_PER_SWEEP; t++) {
		const moe_expert_slot *ts = &L->tier_slots[t];
		if (!tier_movable(c, L, ts) ||
			moe_tier_decide(p, tier_score(c, L, ts->eid), 1, 1) != MOE_TIER_DEMOTE)
			continue;
		moves += tier_demote(m, c, layer, L, t);
	}
	for (int i = 0; i < L->n_lru && moves < TIER_MOVES_PER_SWEEP; i++) {
		moe_expert_slot *s = &L->lru_slots[i];
		if (!tier_movable(c, L, s))
			continue;
		moe_tier_action action = moe_tier_decide(p, tier_score(c, L, s->eid), 0, 1);
		if (action == MOE_TIER_EVICT)
			tier_evict(c, L, i);
		else if (action == MOE_TIER_PROMOTE)
			moves += tier_promote(m, c, layer, L, i);
	}
}

static void tier_maintain(struct model *m, moe_stream_cache *c, int layer,
						  struct moe_stream_layer *L) {
	if (!c->tier_be || L->tier_cap <= 0)
		return;
	pthread_mutex_lock(&L->mtx);
	if (L->step - L->last_sweep >= TIER_SWEEP_INTERVAL) {
		L->last_sweep = L->step;
		tier_sweep(m, c, layer, L);
	}
	pthread_mutex_unlock(&L->mtx);
}

static void *moe_fetch_worker(void *arg) {
	moe_fetch_job *j = (moe_fetch_job *)arg;
	for (;;) {
		int idx = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed);
		if (idx >= j->n_misses)
			break;
		j->misses[idx].st = moe_pin_copy_slot(j->model, j->cache, j->layer, &j->misses[idx].tmp,
											  j->misses[idx].reuse_buf, j->misses[idx].reuse_size);
	}
	moe_stream_thread_cleanup();
	return NULL;
}

static void resolve_scan_hits(moe_stream_cache *c, struct moe_stream_layer *L, struct model *m,
							  const int *expert_ids, int n_k, uint64_t now,
							  moe_expert_slot *out_slots, int *need_fetch, int *n_need,
							  int sync_wait_not_ready) {
	pthread_mutex_lock(&L->mtx);
	if (c->tier_be)
		tier_score_requests(c, L, m->moe.n_experts, expert_ids, n_k);
	for (int k = 0; k < n_k; k++) {
		int eid = expert_ids[k];
		if (eid < 0 || eid >= m->moe.n_experts) {
			slot_mark_invalid(&out_slots[k]);
			continue;
		}
		moe_expert_slot *s = layer_find(L, eid, now);
		if (s) {
			if (layer_take_ready_stats(c, s, &out_slots[k])) {
			} else if (sync_wait_not_ready) {
				pthread_mutex_unlock(&L->mtx);
				moe_stream_wait_slot(s);
				pthread_mutex_lock(&L->mtx);
				s = layer_find(L, eid, now);
				if (!layer_take_ready_stats(c, s, &out_slots[k]))
					slot_mark_invalid(&out_slots[k]);
			} else {
				slot_zero(&out_slots[k]);
				need_fetch[(*n_need)++] = k;
			}
		} else {
			slot_zero(&out_slots[k]);
			need_fetch[(*n_need)++] = k;
			atomic_fetch_add_explicit(&c->stat_misses, 1, memory_order_relaxed);
		}
		atomic_fetch_add_explicit(&c->stat_requests, 1, memory_order_relaxed);
	}
	pthread_mutex_unlock(&L->mtx);
}

static void resolve_collect_misses(moe_stream_cache *c, struct moe_stream_layer *L, struct model *m,
								   int layer, const int *expert_ids, uint64_t now,
								   moe_expert_slot *out_slots, const int *need_fetch, int n_need,
								   moe_miss_entry *misses, int *n_misses, int *wait_needed,
								   int *n_wait_needed) {
	pthread_mutex_lock(&L->mtx);
	for (int i = 0; i < n_need; i++) {
		int k	= need_fetch[i];
		int eid = expert_ids[k];
		if (eid < 0 || eid >= m->moe.n_experts) {
			atomic_store_explicit(&out_slots[k].io_ready, 1, memory_order_release);
			continue;
		}
		moe_expert_slot *s = layer_find(L, eid, now);
		if (layer_take_ready_stats(c, s, &out_slots[k]))
			continue;
		if (s) {
			if (wait_needed) {
				wait_needed[(*n_wait_needed)++] = k;
				continue;
			}
			int prim = -1;
			for (int mi = 0; mi < *n_misses; mi++) {
				if (misses[mi].eid == eid && misses[mi].dep < 0) {
					prim = mi;
					break;
				}
			}
			if (prim >= 0) {
				moe_miss_entry *me = &misses[*n_misses];
				miss_fill_slot(me, k, eid, s, atomic_load_explicit(&s->gen, memory_order_acquire),
							   NULL, 0);
				me->dep = prim;
				atomic_store_explicit(&me->remaining, 1, memory_order_relaxed);
				(*n_misses)++;
				continue;
			}
			pthread_mutex_unlock(&L->mtx);
			moe_stream_wait_slot(s);
			pthread_mutex_lock(&L->mtx);
			s = layer_find(L, eid, now);
			if (!layer_take_ready_stats(c, s, &out_slots[k]))
				slot_mark_invalid(&out_slots[k]);
			continue;
		}

		void  *freed_buf  = NULL;
		size_t freed_size = 0;
		s				  = layer_insert_lru(m, c, layer, L, eid, now, &freed_buf, &freed_size);
		if (!s) {
			miss_fill_desc(&misses[*n_misses], k, eid, freed_buf, freed_size,
						   &m->layers[layer].experts[eid]);
			(*n_misses)++;
			continue;
		}
		slot_from_expert_desc(s, eid, &m->layers[layer].experts[eid]);
		tier_note_load(c, L, eid);
		uint64_t my_gen = atomic_load_explicit(&s->gen, memory_order_acquire);

		miss_fill_slot(&misses[*n_misses], k, eid, s, my_gen, freed_buf, freed_size);
		(*n_misses)++;
	}
	pthread_mutex_unlock(&L->mtx);
}

status_code moe_stream_resolve(struct model *m, int layer, const int *expert_ids, int n_k,
							   moe_expert_slot *out_slots) {
	moe_stream_cache *c = m->moe_cache;
	if (layer < 0 || layer >= m->n_layers) {
		return ERR_INVALID_ARG;
	}
	struct layer_weights *Lw = &m->layers[layer];
	if (!Lw->experts) {
		return ERR_INVALID_ARG;
	}

	if (n_k < 0) {
		ERROR("moe_stream_resolve: invalid n_k=%d", n_k);
		return ERR_INVALID_ARG;
	}

	if (!c || !m->moe_stream_enabled) {
		moe_resolve_no_cache(m, Lw, expert_ids, n_k, out_slots);
		return OK;
	}

	struct moe_stream_layer *L = &c->layers[layer];
	tier_maintain(m, c, layer, L);
	uint64_t	now = atomic_fetch_add_explicit(&c->clock, 1, memory_order_relaxed) + 1;
	status_code rc	= OK;

	int				need_fetch_stack[MOE_MAX_TOPK];
	moe_miss_entry	misses_stack[MOE_MAX_TOPK];
	int				wait_needed_stack[MOE_MAX_TOPK];
	void		   *resolve_heap = NULL;
	int			   *need_fetch	 = need_fetch_stack;
	moe_miss_entry *misses		 = misses_stack;
	int			   *wait_needed	 = wait_needed_stack;
	if (n_k > MOE_MAX_TOPK) {
		resolve_heap = xmalloc((size_t)n_k * (sizeof(moe_miss_entry) + 2 * sizeof(int)));
		misses		 = resolve_heap;
		need_fetch	 = (int *)((char *)resolve_heap + (size_t)n_k * sizeof(moe_miss_entry));
		wait_needed	 = need_fetch + n_k;
	}
	int n_need = 0;

	resolve_scan_hits(c, L, m, expert_ids, n_k, now, out_slots, need_fetch, &n_need, 0);

	if (n_need == 0) {
		free(resolve_heap);
		return OK;
	}

	int n_misses = 0;

	int n_wait_needed = 0;

	resolve_collect_misses(c, L, m, layer, expert_ids, now, out_slots, need_fetch, n_need, misses,
						   &n_misses, wait_needed, &n_wait_needed);

	for (int wi = 0; wi < n_wait_needed; wi++) {
		int k	= wait_needed[wi];
		int eid = expert_ids[k];

		pthread_mutex_lock(&L->mtx);
		moe_expert_slot *s = layer_find(L, eid, now);
		if (layer_take_ready_stats(c, s, &out_slots[k])) {
			pthread_mutex_unlock(&L->mtx);
			continue;
		}
		if (s) {
			pthread_mutex_unlock(&L->mtx);
			continue;
		}
		void  *freed_buf  = NULL;
		size_t freed_size = 0;
		s				  = layer_insert_lru(m, c, layer, L, eid, now, &freed_buf, &freed_size);
		if (!s) {
			pthread_mutex_unlock(&L->mtx);
			miss_fill_desc(&misses[n_misses], k, eid, freed_buf, freed_size, &Lw->experts[eid]);
			n_misses++;
			continue;
		}
		slot_from_expert_desc(s, eid, &Lw->experts[eid]);
		tier_note_load(c, L, eid);
		uint64_t my_gen = atomic_load_explicit(&s->gen, memory_order_acquire);
		pthread_mutex_unlock(&L->mtx);

		miss_fill_slot(&misses[n_misses], k, eid, s, my_gen, freed_buf, freed_size);
		n_misses++;
	}

	if (n_misses > 0) {
		tpool *pool = tpool_fetch_pool();

		int got_pool = pool && pthread_mutex_trylock(&g_fetch_pool_mtx) == 0;

		if (got_pool) {
			size_t			need_chunks = moe_miss_chunk_budget(m, layer, misses, n_misses);
			moe_fill_chunk *chunks		= xmalloc(need_chunks * sizeof(moe_fill_chunk));
			int				n_chunks	= 0;

			for (int i = 0; i < n_misses; i++) {
				atomic_store_explicit(&misses[i].io_err, 0, memory_order_relaxed);
				int start_chunk = n_chunks;
				misses[i].st	= moe_pin_prepare_slot(
					m, c, layer, &misses[i].tmp, misses[i].reuse_buf, misses[i].reuse_size, chunks,
					&n_chunks, (int)need_chunks, &misses[i].io_err);
				if (misses[i].st != OK)
					n_chunks = start_chunk;
			}

			if (n_chunks > 0)
				tpool_parallel_for(pool, n_chunks, 1, moe_fill_chunk_run, chunks);

			free(chunks);

			for (int i = 0; i < n_misses; i++) {
				if (misses[i].st == OK &&
					atomic_load_explicit(&misses[i].io_err, memory_order_relaxed))
					misses[i].st = ERR_IO;
			}
			pthread_mutex_unlock(&g_fetch_pool_mtx);
		} else {
			moe_fetch_job fjob = {
				.model	  = m,
				.cache	  = c,
				.layer	  = layer,
				.misses	  = misses,
				.n_misses = n_misses,
			};
			atomic_store(&fjob.next, 0);

			int n_workers = n_misses;
			if (n_workers > 8)
				n_workers = 8;
			if (n_workers < 1)
				n_workers = 1;

			pthread_t threads[8];
			for (int t = 1; t < n_workers; t++) {
				if (pthread_create(&threads[t], NULL, moe_fetch_worker, &fjob) != 0) {
					threads[t] = 0;
					n_workers  = t;
					break;
				}
			}
			moe_fetch_worker(&fjob);
			for (int t = 1; t < n_workers; t++) {
				if (threads[t])
					pthread_join(threads[t], NULL);
			}
		}
	}

	for (int i = 0; i < n_misses; i++) {
		miss_commit(&out_slots[misses[i].k], &misses[i], L, layer, "moe_stream_resolve");
		if (rc == OK)
			rc = miss_entry_status(&misses[i]);
	}

	for (int wi = 0; wi < n_wait_needed; wi++) {
		int k	= wait_needed[wi];
		int eid = expert_ids[k];

		pthread_mutex_lock(&L->mtx);
		moe_expert_slot *s = layer_find(L, eid, now);
		if (layer_take_ready_stats(c, s, &out_slots[k])) {
			pthread_mutex_unlock(&L->mtx);
			continue;
		}
		moe_expert_slot *live = s;
		pthread_mutex_unlock(&L->mtx);

		if (live)
			moe_stream_wait_slot(live);

		pthread_mutex_lock(&L->mtx);
		s = layer_find(L, eid, now);
		if (!layer_take_ready_stats(c, s, &out_slots[k])) {
			slot_mark_invalid(&out_slots[k]);
			if (rc == OK)
				rc = ERR_IO;
		}
		pthread_mutex_unlock(&L->mtx);
	}
	free(resolve_heap);
	return rc;
}

static void moe_stream_op_finalize_miss(moe_stream_op *op, int mi) {
	moe_miss_entry	*me		  = &op->misses[mi];
	int				 k		  = me->k;
	moe_expert_slot *out_slot = &op->out_slots[k];

	int expect = 0;
	if (!atomic_compare_exchange_strong_explicit(&me->finalized, &expect, 1, memory_order_acq_rel,
												 memory_order_relaxed))
		return;

	if (me->dep >= 0) {
		pthread_mutex_lock(&op->slayer->mtx);
		moe_expert_slot *s = layer_find(op->slayer, me->eid, op->now);
		if (!layer_take_ready_stats(op->cache, s, out_slot))
			slot_mark_invalid(out_slot);
		pthread_mutex_unlock(&op->slayer->mtx);
		return;
	}

	miss_commit(out_slot, me, op->slayer, op->layer, "moe_stream");

	for (int j = 0; j < op->n_misses; j++) {
		if (j != mi && op->misses[j].dep == mi)
			moe_stream_op_finalize_miss(op, j);
	}
}

moe_stream_op *moe_stream_resolve_prep(struct model *m, int layer, const int *expert_ids, int n_k,
									   moe_expert_slot *out_slots) {
	moe_stream_cache *c = m->moe_cache;
	if (layer < 0 || layer >= m->n_layers || !m->layers[layer].experts)
		return NULL;
	if (n_k < 0)
		return NULL;

	if (!c || !m->moe_stream_enabled) {
		moe_resolve_no_cache(m, &m->layers[layer], expert_ids, n_k, out_slots);
		return NULL;
	}

	struct moe_stream_layer *L = &c->layers[layer];
	tier_maintain(m, c, layer, L);
	uint64_t now = atomic_fetch_add_explicit(&c->clock, 1, memory_order_relaxed) + 1;
	int		 need_fetch_stack[MOE_MAX_TOPK];
	int		*need_fetch = need_fetch_stack;
	void	*prep_heap	= NULL;
	if (n_k > MOE_MAX_TOPK) {
		prep_heap  = xmalloc((size_t)n_k * sizeof(int));
		need_fetch = prep_heap;
	}
	int n_need = 0;

	resolve_scan_hits(c, L, m, expert_ids, n_k, now, out_slots, need_fetch, &n_need, 1);

	if (n_need == 0) {
		free(prep_heap);
		return NULL;
	}

	moe_stream_op *op  = xcalloc(1, sizeof(*op));
	op->model		   = m;
	op->cache		   = c;
	op->slayer		   = L;
	op->layer		   = layer;
	op->n_k			   = n_k;
	op->out_slots	   = out_slots;
	op->now			   = now;
	op->st			   = OK;
	op->n_chunks	   = 0;
	op->chunks		   = NULL;
	op->chunk_miss_idx = NULL;
	op->items		   = NULL;
	op->n_items		   = 0;
	op->n_misses	   = 0;
	op->misses		   = xcalloc((size_t)n_k, sizeof(moe_miss_entry));

	resolve_collect_misses(c, L, m, layer, expert_ids, op->now, out_slots, need_fetch, n_need,
						   op->misses, &op->n_misses, NULL, NULL);

	free(prep_heap);
	need_fetch = NULL;

	if (op->n_misses == 0) {
		moe_stream_op_free(op);
		return NULL;
	}

	size_t need_chunks = moe_miss_chunk_budget(m, layer, op->misses, op->n_misses);

	op->chunks		   = xmalloc(need_chunks * sizeof(moe_fill_chunk));
	op->chunk_miss_idx = xmalloc(need_chunks * sizeof(int));
	size_t chunk_cap   = need_chunks;

	int n_chunks = 0;
	for (int i = 0; i < op->n_misses; i++) {
		moe_miss_entry *me = &op->misses[i];
		if (me->dep >= 0) {
			atomic_store_explicit(&me->io_err, 0, memory_order_relaxed);
			continue;
		}
		atomic_store_explicit(&me->io_err, 0, memory_order_relaxed);
		int start_chunk = n_chunks;
		me->st			= moe_pin_prepare_slot(m, c, layer, &me->tmp, me->reuse_buf, me->reuse_size,
											   op->chunks, &n_chunks, (int)chunk_cap, &me->io_err);
		if (me->st != OK) {
			n_chunks = start_chunk;
			atomic_store_explicit(&me->remaining, 0, memory_order_relaxed);
			moe_stream_op_finalize_miss(op, i);
			continue;
		}
		int n_added = n_chunks - start_chunk;
		for (int j = start_chunk; j < n_chunks; j++)
			op->chunk_miss_idx[j] = i;
		if (n_added == 0) {
			atomic_store_explicit(&me->remaining, 0, memory_order_relaxed);
			moe_stream_op_finalize_miss(op, i);
		} else {
			atomic_store_explicit(&me->remaining, n_added, memory_order_relaxed);
		}
	}
	op->n_chunks = n_chunks;

	if (op->n_chunks == 0) {
		moe_stream_op_free(op);
		return NULL;
	}

	{
		char  pending_stack[MOE_MAX_TOPK];
		char *pending	   = pending_stack;
		void *pending_heap = NULL;
		if (n_k > MOE_MAX_TOPK) {
			pending_heap = xmalloc((size_t)n_k);
			pending		 = pending_heap;
		}
		memset(pending, 0, (size_t)n_k);
		for (int i = 0; i < op->n_misses; i++) {
			moe_miss_entry *me = &op->misses[i];
			if (me->k >= 0 && me->k < n_k)
				if (me->dep >= 0 || atomic_load_explicit(&me->remaining, memory_order_relaxed) > 0)
					pending[me->k] = 1;
		}

		int n_items = op->n_chunks + n_k;
		op->items	= xcalloc((size_t)n_items, sizeof(moe_op_item));
		int it		= 0;

		for (int k = 0; k < n_k; k++)
			if (!pending[k])
				op->items[it++] = (moe_op_item){.chunk = -1, .k = k};

		for (int ci = 0; ci < op->n_chunks; ci++)
			op->items[it++] = (moe_op_item){.chunk = ci, .k = -1};

		op->n_items = it;
		free(pending_heap);
	}
	return op;
}

void moe_stream_op_set_compute_hook(moe_stream_op *op, moe_stream_compute_hook_fn fn, void *ctx) {
	if (!op)
		return;
	op->hook	 = fn;
	op->hook_ctx = ctx;
}

int moe_stream_op_n_items(const moe_stream_op *op) {
	return op ? op->n_items : 0;
}

int moe_stream_op_compute_k(const moe_stream_op *op, int item) {
	if (!op || item < 0 || item >= op->n_items)
		return -1;
	return op->items[item].k;
}

void moe_stream_op_fill_run(moe_stream_op *op, int item, int tid) {
	(void)tid;
	if (item < 0 || item >= op->n_items)
		return;
	int ci = op->items[item].chunk;
	if (ci < 0 || ci >= op->n_chunks)
		return;
	{
		moe_fill_chunk *ch = &op->chunks[ci];
		if (ch->err_flag && atomic_load_explicit(ch->err_flag, memory_order_relaxed))
			return;
		if (!moe_fill_chunk_do_read(ch) && ch->err_flag)
			atomic_store_explicit(ch->err_flag, 1, memory_order_relaxed);
		int mi	 = op->chunk_miss_idx[ci];
		int prev = atomic_fetch_sub_explicit(&op->misses[mi].remaining, 1, memory_order_relaxed);
		if (prev == 1) {
			moe_stream_op_finalize_miss(op, mi);
			if (op->hook) {
				op->hook(op->misses[mi].k, op->misses[mi].k + 1, tid, op->hook_ctx);
				for (int j = 0; j < op->n_misses; j++)
					if (op->misses[j].dep == mi)
						op->hook(op->misses[j].k, op->misses[j].k + 1, tid, op->hook_ctx);
			}
		}
	}
}

status_code moe_stream_op_finish(moe_stream_op *op) {
	if (!op)
		return OK;
	for (int i = 0; i < op->n_misses; i++) {
		if (atomic_load_explicit(&op->misses[i].remaining, memory_order_relaxed) > 0)
			moe_stream_op_finalize_miss(op, i);
	}
	op->st = OK;
	for (int i = 0; i < op->n_misses; i++) {
		status_code mst = miss_entry_status(&op->misses[i]);
		if (mst != OK) {
			op->st = mst;
			break;
		}
	}
	return op->st;
}

void moe_stream_op_free(moe_stream_op *op) {
	if (!op)
		return;
	free(op->chunks);
	free(op->chunk_miss_idx);
	free(op->items);
	free(op->misses);
	free(op);
}

static int moe_slot_ready_pred(void *ud) {
	const moe_expert_slot *slot = ud;
	return atomic_load_explicit(&slot->io_ready, memory_order_acquire) != 0;
}

void moe_stream_wait_slot(const moe_expert_slot *slot) {
	if (!slot)
		return;
	spin_wait_relax(moe_slot_ready_pred, (void *)slot);
}

void moe_stream_release_slot(struct model *m, int layer, const moe_expert_slot *slot) {
	if (!m || !m->moe_cache || !slot || slot->eid < 0)
		return;
	moe_stream_cache *c = m->moe_cache;
	if (layer < 0 || layer >= c->n_layers)
		return;

	struct moe_stream_layer *L = &c->layers[layer];
	pthread_mutex_lock(&L->mtx);
	moe_expert_slot *live = live_slot_find(L, slot);
	if (live)
		live_slot_unuse(live);
	pthread_mutex_unlock(&L->mtx);
}

void moe_stream_release_slots(struct model *m, int layer, const moe_expert_slot *slots, int n) {
	if (!m || !m->moe_cache || !slots || n <= 0)
		return;
	moe_stream_cache *c = m->moe_cache;
	if (layer < 0 || layer >= c->n_layers)
		return;

	struct moe_stream_layer *L = &c->layers[layer];
	pthread_mutex_lock(&L->mtx);
	for (int j = 0; j < n; j++) {
		const moe_expert_slot *slot = &slots[j];
		if (slot->eid < 0)
			continue;
		moe_expert_slot *live = live_slot_find(L, slot);
		if (live)
			live_slot_unuse(live);
	}
	pthread_mutex_unlock(&L->mtx);
}

void moe_stream_get_stats(const moe_stream_cache *c, moe_stream_stats *out) {
	if (!c || !out)
		return;
	out->requests			= atomic_load_explicit(&c->stat_requests, memory_order_relaxed);
	out->cache_hits			= atomic_load_explicit(&c->stat_hits, memory_order_relaxed);
	out->cache_misses		= atomic_load_explicit(&c->stat_misses, memory_order_relaxed);
	out->pin_hits			= atomic_load_explicit(&c->stat_pin_hits, memory_order_relaxed);
	out->lru_hits			= atomic_load_explicit(&c->stat_lru_hits, memory_order_relaxed);
	out->direct_io_ok		= atomic_load_explicit(&c->stat_direct_ok, memory_order_relaxed);
	out->direct_io_fallback = atomic_load_explicit(&c->stat_direct_fallback, memory_order_relaxed);
}

void moe_stream_summarize(const struct model *m, moe_stats_summary *out) {
	int has_moe = m->arch_info && m->arch_info->is_moe && m->moe_cache;
	if (!has_moe) {
		memset(out, 0, sizeof(*out));
		return;
	}
	out->has_moe = 1;
	moe_stream_stats s;
	moe_stream_get_stats(m->moe_cache, &s);
	out->hit_rate	  = s.requests > 0 ? 100.0 * (double)s.cache_hits / (double)s.requests : 0.0;
	out->pin_rate	  = s.requests > 0 ? 100.0 * (double)s.pin_hits / (double)s.requests : 0.0;
	out->lru_rate	  = s.requests > 0 ? 100.0 * (double)s.lru_hits / (double)s.requests : 0.0;
	out->cache_misses = s.cache_misses;
	out->direct_io_ok = s.direct_io_ok;
	out->direct_io_fallback = s.direct_io_fallback;
}

void moe_stream_record_extra_hits(struct model *m, int layer, int n_extra_hits,
								  int n_extra_requests, int is_pinned) {
	if (!m || !m->moe_cache || layer < 0 || layer >= m->n_layers)
		return;
	if (n_extra_hits <= 0 && n_extra_requests <= 0)
		return;
	moe_stream_cache *c = m->moe_cache;
	if (n_extra_requests > 0)
		atomic_fetch_add_explicit(&c->stat_requests, (uint64_t)n_extra_requests,
								  memory_order_relaxed);
	if (n_extra_hits > 0) {
		atomic_fetch_add_explicit(&c->stat_hits, (uint64_t)n_extra_hits, memory_order_relaxed);
		if (is_pinned)
			atomic_fetch_add_explicit(&c->stat_pin_hits, (uint64_t)n_extra_hits,
									  memory_order_relaxed);
		else
			atomic_fetch_add_explicit(&c->stat_lru_hits, (uint64_t)n_extra_hits,
									  memory_order_relaxed);
	}
}

static void *moe_preload_worker(void *arg) {
	moe_preload_job *j = arg;
	for (size_t i = j->begin; i < j->end; i++)
		fault_wait(j->regions[i].ptr, j->regions[i].bytes, j->page_size);
	return NULL;
}

static void moe_preload_push_region(moe_preload_region **regions, size_t *n, size_t *cap,
									const void *ptr, size_t bytes) {
	if (!ptr || bytes == 0)
		return;
	ARR_RESERVE(*regions, *n, *cap);
	(*regions)[*n].ptr	 = ptr;
	(*regions)[*n].bytes = bytes;
	(*n)++;
}

status_code moe_stream_preload_all(struct model *m) {
	if (!m->arch_info->is_moe || !m->moe_cache)
		return OK;
	moe_stream_cache *c = m->moe_cache;
	if (!c->use_mmap) {
		DEBUG("moe_stream: preload_all is a no-op under --mmap off");
		return OK;
	}
	int total_experts = 0;
	int total_layers  = 0;

	size_t				n_regions	= 0;
	size_t				cap_regions = 1024;
	moe_preload_region *regions		= xmalloc(cap_regions * sizeof(*regions));

	for (int i = 0; i < m->n_layers; i++) {
		layer_weights *Lw = &m->layers[i];
		if (!Lw->experts || !Lw->is_moe_layer)
			continue;
		for (int e = 0; e < m->moe.n_experts; e++) {
			int				 fused = Lw->experts[e].gate_up_fused;
			moe_expert_bytes eb	   = moe_calc_expert_bytes(m, i, e);
			fault_hint(c, Lw->experts[e].gate_w, eb.gate_b);
			if (!fused)
				fault_hint(c, Lw->experts[e].up_w, eb.up_b);
			fault_hint(c, Lw->experts[e].down_w, eb.down_b);

			moe_preload_push_region(&regions, &n_regions, &cap_regions, Lw->experts[e].gate_w,
									eb.gate_b);
			if (!fused)
				moe_preload_push_region(&regions, &n_regions, &cap_regions, Lw->experts[e].up_w,
										eb.up_b);
			moe_preload_push_region(&regions, &n_regions, &cap_regions, Lw->experts[e].down_w,
									eb.down_b);
			total_experts++;
		}
		total_layers++;
	}

	long n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
	if (n_cpu < 1)
		n_cpu = 1;
	int n_threads = (int)n_cpu;
	if (n_threads > MOE_PRELOAD_MAX_THREADS)
		n_threads = MOE_PRELOAD_MAX_THREADS;
	if ((size_t)n_threads > n_regions)
		n_threads = n_regions > 0 ? (int)n_regions : 1;

	if (n_threads <= 1 || n_regions == 0) {
		for (size_t i = 0; i < n_regions; i++)
			fault_wait(regions[i].ptr, regions[i].bytes, c->page_size);
	} else {
		pthread_t		threads[MOE_PRELOAD_MAX_THREADS];
		moe_preload_job jobs[MOE_PRELOAD_MAX_THREADS];
		size_t			base_n = n_regions / (size_t)n_threads;
		size_t			off	   = 0;

		for (int t = 0; t < n_threads; t++) {
			size_t n = (t == n_threads - 1) ? (n_regions - off) : base_n;
			jobs[t]	 = (moe_preload_job){
				.regions = regions, .begin = off, .end = off + n, .page_size = c->page_size};
			off += n;
		}
		for (int t = 1; t < n_threads; t++) {
			if (pthread_create(&threads[t], NULL, moe_preload_worker, &jobs[t]) != 0)
				threads[t] = 0;
		}
		moe_preload_worker(&jobs[0]);
		for (int t = 1; t < n_threads; t++) {
			if (threads[t])
				pthread_join(threads[t], NULL);
		}
	}

	free(regions);

	INFO("moe_stream: preloaded %d experts across %d layers", total_experts, total_layers);
	return OK;
}

backend *moe_stream_expert_owner(const struct model *m, int layer, int eid) {
	if (!m || !m->moe_cache || layer < 0 || layer >= m->n_layers || eid < 0 ||
		eid >= m->moe.n_experts)
		return NULL;
	struct moe_stream_layer *L = &m->moe_cache->layers[layer];
	if (L->eid_to_pinned) {
		int pi = L->eid_to_pinned[eid];
		if (pi >= 0 && pi < L->n_pinned) {
			const moe_expert_slot *sl = &L->pinned_slots[pi];
			return sl->dev_ready ? sl->dev_gate.owner : NULL;
		}
	}
	if (!L->eid_to_tier)
		return NULL;
	backend *owner = NULL;
	pthread_mutex_lock(&L->mtx);
	const moe_expert_slot *live = layer_find_tier(L, eid);
	if (live && live->dev_ready)
		owner = live->dev_gate.owner;
	pthread_mutex_unlock(&L->mtx);
	return owner;
}
