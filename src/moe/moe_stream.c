#define _GNU_SOURCE
#include "moe/moe_stream.h"
#include "common.h"
#include "config.h"
#include "log.h"
#include "model.h"
#include "monitor.h"
#include "profile.h"
#include "threadpool.h"

#include <errno.h>
#include <fcntl.h>
#if HAVE_MALLOC
#include <malloc.h>
#endif
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MOE_DEFAULT_CACHE_CAP 64
#define MOE_DIRECT_IO_FALLBACK_ALIGN 4096
#define MOE_CHUNK_TARGET_BYTES (8 * 1024 * 1024)

#define PIN_COPY_MAX_WORKERS 8
#define MOE_PRELOAD_MAX_THREADS 32
#define LFRU_DECAY_INTERVAL 256

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

	int *heap_idx;
	int *heap_pos;
	int	 heap_n;
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
	moe_miss_entry			   misses[MOE_MAX_K];
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

	int flags = fcntl(fd, F_GETFL);
	if (flags < 0 || fcntl(fd, F_SETFL, flags | O_DIRECT) != 0) {
		close(fd);
		DEBUG("moe direct-io: O_DIRECT unsupported, using buffered reads");
		return;
	}

	long		blk = MOE_DIRECT_IO_FALLBACK_ALIGN;
	struct stat st;
	if (fstat(fd, &st) == 0 && st.st_blksize > 0)
		blk = st.st_blksize;

	size_t align = (size_t)blk;
	void  *probe_buf;
	if (posix_memalign(&probe_buf, align, align) != 0) {
		close(fd);
		return;
	}
	ssize_t rc = pread(fd, probe_buf, align, 0);
	free(probe_buf);
	if (rc < 0) {
		close(fd);
		DEBUG("moe direct-io: O_DIRECT probe failed (%s), using buffered reads", strerror(errno));
		return;
	}

	c->direct_io_fd	   = fd;
	c->direct_io_align = align;
	posix_fadvise(c->direct_io_fd, 0, 0, POSIX_FADV_RANDOM);
	DEBUG("moe direct-io: enabled (align=%zu)", align);
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

	long		blk = MOE_DIRECT_IO_FALLBACK_ALIGN;
	struct stat st;
	if (fstat(fd, &st) == 0 && st.st_blksize > 0)
		blk = st.st_blksize;
	size_t align = (size_t)blk;

	void *probe_buf;
	if (posix_memalign(&probe_buf, align, align) != 0) {
		close(fd);
		DEBUG("moe no-mmap: O_DIRECT probe alloc failed, using buffered reads");
		return;
	}
	ssize_t rc = pread(fd, probe_buf, align, 0);
	free(probe_buf);
	if (rc < 0) {
		close(fd);
		DEBUG("moe no-mmap: O_DIRECT probe failed (%s), using buffered reads", strerror(errno));
		return;
	}

	c->nomap_direct_fd	  = fd;
	c->nomap_direct_align = align;
	posix_fadvise(c->nomap_direct_fd, 0, 0, POSIX_FADV_RANDOM);
	DEBUG("moe no-mmap: O_DIRECT enabled (align=%zu)", align);
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

static ssize_t pread_full(int fd, uint8_t *dst, size_t len, off_t off) {
	size_t total = 0;
	while (total < len) {
		ssize_t n = pread(fd, dst + total, len - total, off + (off_t)total);
		if (n < 0) {
			if (errno == EINTR)
				continue;
			return -1;
		}
		if (n == 0)
			break;
		total += (size_t)n;
	}
	return (ssize_t)total;
}

static int bounce_read_aligned(int fd, uint64_t aligned_off, size_t aligned_len, size_t head_slop,
							   size_t len, uint8_t *dst, size_t align) {
	void *bounce = bounce_buf_get(align, aligned_len);
	if (!bounce)
		return -1;
	ssize_t got = pread_full(fd, bounce, aligned_len, (off_t)aligned_off);
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

	if (bounce_read_aligned(fd, aligned_off, aligned_len, head_slop, len, dst, align) != 0)
		return -1;
	posix_fadvise(fd, (off_t)aligned_off, (off_t)aligned_len, POSIX_FADV_DONTNEED);
	return 0;
}

static int moe_nomap_read(moe_stream_cache *c, uint64_t file_off, size_t len, uint8_t *dst,
						  int precheck_aligned) {
	if (len == 0)
		return 0;

	if (c->nomap_direct_fd >= 0 && c->nomap_direct_align > 0) {
		size_t align = c->nomap_direct_align;

		if (precheck_aligned &&
			(size_t)pread_full(c->nomap_direct_fd, dst, len, (off_t)file_off) >= len) {
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
	if ((size_t)pread_full(c->nomap_fd, dst, len, (off_t)file_off) < len)
		return -1;
	posix_fadvise(c->nomap_fd, (off_t)file_off, (off_t)len, POSIX_FADV_DONTNEED);
	return 0;
}

static int moe_direct_io_read(moe_stream_cache *c, off_t file_off, size_t len, uint8_t *dst,
							  int precheck_aligned) {
	if (c->direct_io_fd < 0 || len == 0)
		return -1;

	size_t align = c->direct_io_align;

	if (precheck_aligned && (size_t)pread_full(c->direct_io_fd, dst, len, file_off) >= len) {
		posix_fadvise(c->direct_io_fd, file_off, (off_t)len, POSIX_FADV_DONTNEED);
		return 0;
	}

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
			uintptr_t ps = c->page_size;
			uintptr_t pm = ~(ps - 1);
			for (int i = 0; i < 3; i++) {
				if (!regions[i].src || regions[i].len == 0)
					continue;
				uintptr_t a = (uintptr_t)regions[i].src;
				uintptr_t b = a + regions[i].len;
				if (a < base || b > stop || b < a)
					continue;
				uintptr_t pstart = a & pm;
				uintptr_t pend	 = (b + ps - 1) & pm;
				madvise((void *)pstart, pend - pstart, MADV_DONTNEED);
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
	moe_stream_thread_cleanup();
	return NULL;
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
	m->moe_stream_enabled		= (cfg->moe_stream != 0) || !m->use_mmap;

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
	if (lru_cap > 1024)
		lru_cap = 1024;
	if (m->moe.n_experts_used > 0 && lru_cap < m->moe.n_experts_used) {
		DEBUG("moe_stream: --moe-cache %d is below topk=%d (will evict every step)", lru_cap,
			  m->moe.n_experts_used);
	}

	c->n_pinned = full_resident ? m->moe.n_experts : cfg->moe_pin;
	if (c->n_pinned < 0)
		c->n_pinned = 0;
	if (c->n_pinned > m->moe.n_experts)
		c->n_pinned = m->moe.n_experts;

	int *pin_list	= xmalloc((size_t)m->moe.n_experts * sizeof(int));
	int	 n_pin_list = 0;
	if (!full_resident && cfg->moe_pin_list && cfg->moe_pin_list[0]) {
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
		L->heap_idx		 = xmalloc((size_t)lru_cap * sizeof(int));
		L->heap_pos		 = xmalloc((size_t)lru_cap * sizeof(int));
		L->heap_n		 = 0;
		for (int s = 0; s < lru_cap; s++)
			L->heap_pos[s] = -1;
		for (int e = 0; e < m->moe.n_experts; e++)
			L->eid_to_lru[e] = -1;
		for (int s = 0; s < lru_cap; s++) {
			L->lru_slots[s].eid		  = -1;
			L->lru_slots[s].pinned	  = 0;
			L->lru_slots[s].heap_buf  = NULL;
			L->lru_slots[s].heap_size = 0;
			atomic_init(&L->lru_slots[s].io_ready, 1);
			atomic_init(&L->lru_slots[s].inuse, 0);
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
			n_total += c->layers[i].n_pinned;

		if (n_total > 0) {
			moe_expert_slot **slots = (moe_expert_slot **)xmalloc((size_t)n_total * sizeof(*slots));
			int				 *layers = (int *)xmalloc((size_t)n_total * sizeof(int));
			int				  idx	 = 0;
			for (int i = 0; i < m->n_layers; i++) {
				for (int pi = 0; pi < c->layers[i].n_pinned; pi++) {
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
				monitor_send(g_monitor,
							 "{\"type\":\"load\",\"phase\":\"pin_copy_start\","
							 "\"n_experts\":%d,\"n_workers\":%d}",
							 n_total, nw);
				monitor_poll(g_monitor);
			}

			pthread_t th[PIN_COPY_MAX_WORKERS];
			for (int i = 0; i < nw; i++) {
				if (pthread_create(&th[i], NULL, pin_copy_worker, &job) != 0) {
					th[i] = 0;
					nw	  = i;
					break;
				}
			}
			pin_copy_worker(&job);
			for (int i = 0; i < nw; i++) {
				if (th[i])
					pthread_join(th[i], NULL);
			}

			uint64_t elapsed_us = time_us() - t0;
			size_t	 copied		= atomic_load(&job.total_copied);
			int		 n_copied	= atomic_load(&job.n_copied);
			double	 mb			= (double)copied / (1024.0 * 1024.0);
			double	 secs		= (double)elapsed_us / 1e6;
			double	 mbps		= secs > 0 ? mb / secs : 0;

			INFO("moe pin-copy: %d experts, %.1f MB in %.1f s (%.0f MB/s)", n_copied, mb, secs,
				 mbps);

			if (g_monitor && g_monitor->fd >= 0) {
				monitor_send(g_monitor,
							 "{\"type\":\"load\",\"phase\":\"pin_copy_done\","
							 "\"n_experts\":%d,\"mb\":%.1f,\"ms\":%llu}",
							 n_copied, mb, (unsigned long long)(elapsed_us / 1000));
				monitor_poll(g_monitor);
			}

			free((void *)slots);
			free((void *)layers);
		}
	}
	if (full_resident && m->backend && m->backend->moe_expert_ffn &&
		m->backend->buffer_alloc_from_host && (m->backend->caps & BCAP_MOE_EXPERT_RESIDENT)) {
		int		 uploaded  = 0;
		int		 failed	   = 0;
		size_t	 dev_bytes = 0;
		uint64_t t0		   = time_us();
		for (int i = 0; i < m->n_layers && !failed; i++) {
			for (int pi = 0; pi < c->layers[i].n_pinned; pi++) {
				moe_expert_slot *sl = &c->layers[i].pinned_slots[pi];
				moe_expert_bytes eb = moe_calc_expert_bytes(m, i, sl->eid);
				status_code		 st = m->backend->buffer_alloc_from_host(m->backend, sl->gate_w,
																		 eb.gate_b, &sl->dev_gate);
				if (st == OK && !sl->gate_up_fused)
					st = m->backend->buffer_alloc_from_host(m->backend, sl->up_w, eb.up_b,
															&sl->dev_up);
				if (st == OK)
					st = m->backend->buffer_alloc_from_host(m->backend, sl->down_w, eb.down_b,
															&sl->dev_down);
				if (st != OK) {
					failed = 1;
					break;
				}
				dev_bytes += eb.total;
				sl->dev_ready = 1;
				uploaded++;
			}
		}
		if (failed) {
			for (int i = 0; i < m->n_layers; i++) {
				for (int pi = 0; pi < c->layers[i].n_pinned; pi++) {
					moe_expert_slot *sl = &c->layers[i].pinned_slots[pi];
					if (sl->dev_ready && sl->dev_gate.owner)
						sl->dev_gate.owner->buffer_free(sl->dev_gate.owner, &sl->dev_gate);
					if (sl->dev_up.owner)
						sl->dev_up.owner->buffer_free(sl->dev_up.owner, &sl->dev_up);
					if (sl->dev_down.owner)
						sl->dev_down.owner->buffer_free(sl->dev_down.owner, &sl->dev_down);
					sl->dev_ready = 0;
				}
			}
			INFO("MoE: expert residency unavailable (allocation failed) -- keeping CPU path");
		} else if (uploaded > 0) {
			m->moe.experts_resident = 1;
			INFO("MoE: %d experts resident on device (%.1f MB) in %.1f s", uploaded,
				 (double)dev_bytes / (1024.0 * 1024.0), (double)(time_us() - t0) / 1e6);
		}
	}

	return OK;
}

void moe_stream_cache_free(moe_stream_cache *c) {
	if (!c)
		return;

	for (int i = 0; i < c->n_layers; i++) {
		for (int s = 0; s < c->layers[i].n_pinned; s++) {
			moe_expert_slot *sl = &c->layers[i].pinned_slots[s];
			if (sl->dev_gate.owner)
				sl->dev_gate.owner->buffer_free(sl->dev_gate.owner, &sl->dev_gate);
			if (sl->dev_up.owner)
				sl->dev_up.owner->buffer_free(sl->dev_up.owner, &sl->dev_up);
			if (sl->dev_down.owner)
				sl->dev_down.owner->buffer_free(sl->dev_down.owner, &sl->dev_down);
			sl->dev_ready = 0;
		}
		for (int s = 0; s < c->layers[i].n_pinned; s++) {
			free(c->layers[i].pinned_slots[s].heap_buf);
		}
		for (int s = 0; s < c->layers[i].n_lru; s++) {
			free(c->layers[i].lru_slots[s].heap_buf);
		}
		free(c->layers[i].lru_slots);
		free(c->layers[i].lru_freq);
		free(c->layers[i].pinned_slots);
		free(c->layers[i].eid_to_lru);
		free(c->layers[i].heap_idx);
		free(c->layers[i].heap_pos);
		pthread_mutex_destroy(&c->layers[i].mtx);
	}
	free(c->layers);
	moe_direct_io_close(c);
	moe_nomap_close(c);
	free(c);
}

static void fault_hint(const moe_stream_cache *c, const void *ptr, size_t bytes) {
	if (!c->use_mmap)
		return;
	if (!c->map_base || c->map_size == 0 || !ptr || bytes == 0)
		return;
	uintptr_t addr = (uintptr_t)ptr;
	uintptr_t end  = addr + bytes;
	uintptr_t base = (uintptr_t)c->map_base;
	uintptr_t stop = base + c->map_size;
	if (addr < base || end > stop || end < addr)
		return;
	uintptr_t page_mask = ~((uintptr_t)c->page_size - 1);
	uintptr_t pstart	= addr & page_mask;
	uintptr_t pend		= (end + c->page_size - 1) & page_mask;
	size_t	  count		= pend - pstart;
	if (count == 0)
		return;

	madvise((void *)pstart, count, MADV_WILLNEED);
}

static void fault_wait(const void *ptr, size_t bytes, long page_size) {
	if (!ptr || bytes == 0)
		return;
	uintptr_t addr		= (uintptr_t)ptr;
	uintptr_t end		= addr + bytes;
	uintptr_t page_mask = ~((uintptr_t)page_size - 1);
	uintptr_t pstart	= addr & page_mask;
	uintptr_t pend		= (end + (uintptr_t)page_size - 1) & page_mask;

	volatile uint8_t acc = 0;
	for (uintptr_t p = pstart; p < pend; p += (uintptr_t)page_size) {
		acc |= *((const volatile uint8_t *)p);
	}
	(void)acc;
}

static void drop_pages(const moe_stream_cache *c, const void *ptr, size_t bytes) {
	if (!c->map_base || c->map_size == 0 || !ptr || bytes == 0)
		return;
	uintptr_t addr = (uintptr_t)ptr;
	uintptr_t end  = addr + bytes;
	uintptr_t base = (uintptr_t)c->map_base;
	uintptr_t stop = base + c->map_size;
	if (addr < base || end > stop)
		return;
	uintptr_t page_mask = ~((uintptr_t)c->page_size - 1);
	uintptr_t pstart	= addr & page_mask;
	uintptr_t pend		= (end + c->page_size - 1) & page_mask;
	madvise((void *)pstart, pend - pstart, MADV_DONTNEED);
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

static moe_expert_slot *layer_find(struct moe_stream_layer *sl, int eid, uint64_t now) {
	for (int i = 0; i < sl->n_pinned; i++) {
		if (sl->pinned_slots[i].eid == eid) {
			sl->pinned_slots[i].last_used = now;
			return &sl->pinned_slots[i];
		}
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

static void miss_fill_desc(moe_miss_entry *me, int k, int eid, void *freed_buf, size_t freed_size,
						   const struct expert_desc *desc) {
	moe_expert_slot tmp;
	slot_zero(&tmp);
	slot_from_expert_desc(&tmp, eid, desc);
	tmp.heap_buf   = NULL;
	tmp.heap_size  = 0;
	me->k		   = k;
	me->eid		   = eid;
	me->dep		   = -1;
	me->slot	   = NULL;
	me->gen		   = 0;
	me->tmp		   = tmp;
	me->st		   = OK;
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
	me->reuse_buf	  = freed_buf;
	me->reuse_size	  = freed_size;
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
	for (int i = 0; i < L->n_pinned; i++) {
		if (L->pinned_slots[i].eid == slot->eid)
			return &L->pinned_slots[i];
	}
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

static void *moe_fetch_worker(void *arg) {
	moe_fetch_job *j = (moe_fetch_job *)arg;
	for (;;) {
		int idx = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed);
		if (idx >= j->n_misses)
			break;
		j->misses[idx].st = moe_pin_copy_slot(j->model, j->cache, j->layer, &j->misses[idx].tmp,
											  j->misses[idx].reuse_buf, j->misses[idx].reuse_size);
	}
	return NULL;
}

static void resolve_scan_hits(moe_stream_cache *c, struct moe_stream_layer *L, struct model *m,
							  const int *expert_ids, int n_k, uint64_t now,
							  moe_expert_slot *out_slots, int *need_fetch, int *n_need,
							  int sync_wait_not_ready) {
	pthread_mutex_lock(&L->mtx);
	for (int k = 0; k < n_k; k++) {
		int eid = expert_ids[k];
		if (eid < 0 || eid >= m->moe.n_experts) {
			slot_mark_invalid(&out_slots[k]);
			continue;
		}
		moe_expert_slot *s = layer_find(L, eid, now);
		if (s) {
			int ready = atomic_load_explicit(&s->io_ready, memory_order_acquire);
			atomic_fetch_add_explicit(&c->stat_hits, 1, memory_order_relaxed);
			if (s->pinned)
				atomic_fetch_add_explicit(&c->stat_pin_hits, 1, memory_order_relaxed);
			else
				atomic_fetch_add_explicit(&c->stat_lru_hits, 1, memory_order_relaxed);
			if (ready) {
				slot_take_ready(&out_slots[k], s);
			} else if (sync_wait_not_ready) {
				pthread_mutex_unlock(&L->mtx);
				moe_stream_wait_slot(s);
				pthread_mutex_lock(&L->mtx);
				s = layer_find(L, eid, now);
				if (s && atomic_load_explicit(&s->io_ready, memory_order_acquire))
					slot_take_ready(&out_slots[k], s);
				else
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
		if (s && atomic_load_explicit(&s->io_ready, memory_order_acquire)) {
			slot_take_ready(&out_slots[k], s);
			continue;
		}
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
			if (s && atomic_load_explicit(&s->io_ready, memory_order_acquire))
				slot_take_ready(&out_slots[k], s);
			else
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

	if (n_k > MOE_MAX_K) {
		ERROR("moe_stream_resolve: n_k=%d exceeds MOE_MAX_K=%d buffer", n_k, MOE_MAX_K);
		return ERR_INVALID_ARG;
	}

	if (!c || !m->moe_stream_enabled) {
		moe_resolve_no_cache(m, Lw, expert_ids, n_k, out_slots);
		return OK;
	}

	struct moe_stream_layer *L = &c->layers[layer];
	uint64_t	now			   = atomic_fetch_add_explicit(&c->clock, 1, memory_order_relaxed) + 1;
	status_code rc			   = OK;

	int need_fetch[MOE_MAX_K];
	int n_need = 0;

	resolve_scan_hits(c, L, m, expert_ids, n_k, now, out_slots, need_fetch, &n_need, 0);

	if (n_need == 0)
		return OK;

	moe_miss_entry misses[MOE_MAX_K];
	int			   n_misses = 0;

	int wait_needed[MOE_MAX_K];
	int n_wait_needed = 0;

	resolve_collect_misses(c, L, m, layer, expert_ids, now, out_slots, need_fetch, n_need, misses,
						   &n_misses, wait_needed, &n_wait_needed);

	for (int wi = 0; wi < n_wait_needed; wi++) {
		int k	= wait_needed[wi];
		int eid = expert_ids[k];

		pthread_mutex_lock(&L->mtx);
		moe_expert_slot *s = layer_find(L, eid, now);
		if (s && atomic_load_explicit(&s->io_ready, memory_order_acquire)) {
			slot_take_ready(&out_slots[k], s);
			atomic_fetch_add_explicit(&c->stat_hits, 1, memory_order_relaxed);
			atomic_fetch_add_explicit(&c->stat_lru_hits, 1, memory_order_relaxed);
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
		uint64_t my_gen = atomic_load_explicit(&s->gen, memory_order_acquire);
		pthread_mutex_unlock(&L->mtx);

		miss_fill_slot(&misses[n_misses], k, eid, s, my_gen, freed_buf, freed_size);
		n_misses++;
	}

	if (n_misses > 0) {
		backend *host_be = backend_host();
		tpool	*pool	 = (host_be && host_be->get_pool) ? host_be->get_pool(host_be) : NULL;

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
		if (s && atomic_load_explicit(&s->io_ready, memory_order_acquire)) {
			slot_take_ready(&out_slots[k], s);
			atomic_fetch_add_explicit(&c->stat_hits, 1, memory_order_relaxed);
			atomic_fetch_add_explicit(&c->stat_lru_hits, 1, memory_order_relaxed);
			pthread_mutex_unlock(&L->mtx);
			continue;
		}
		moe_expert_slot *live = s;
		pthread_mutex_unlock(&L->mtx);

		if (live)
			moe_stream_wait_slot(live);

		pthread_mutex_lock(&L->mtx);
		s = layer_find(L, eid, now);
		if (s && atomic_load_explicit(&s->io_ready, memory_order_acquire)) {
			slot_take_ready(&out_slots[k], s);
			atomic_fetch_add_explicit(&c->stat_hits, 1, memory_order_relaxed);
			atomic_fetch_add_explicit(&c->stat_lru_hits, 1, memory_order_relaxed);
		} else {
			slot_mark_invalid(&out_slots[k]);
			if (rc == OK)
				rc = ERR_IO;
		}
		pthread_mutex_unlock(&L->mtx);
	}
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
		if (s && atomic_load_explicit(&s->io_ready, memory_order_acquire)) {
			slot_take_ready(out_slot, s);
		} else {
			slot_mark_invalid(out_slot);
		}
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
	if (n_k > MOE_MAX_K) {
		ERROR("moe_stream_resolve_prep: n_k=%d exceeds MOE_MAX_K=%d buffer", n_k, MOE_MAX_K);
		return NULL;
	}

	if (!c || !m->moe_stream_enabled) {
		moe_resolve_no_cache(m, &m->layers[layer], expert_ids, n_k, out_slots);
		return NULL;
	}

	moe_stream_op *op  = xcalloc(1, sizeof(*op));
	op->model		   = m;
	op->cache		   = c;
	op->slayer		   = &c->layers[layer];
	op->layer		   = layer;
	op->n_k			   = n_k;
	op->out_slots	   = out_slots;
	op->now			   = atomic_fetch_add_explicit(&c->clock, 1, memory_order_relaxed) + 1;
	op->st			   = OK;
	op->n_chunks	   = 0;
	op->chunks		   = NULL;
	op->chunk_miss_idx = NULL;
	op->items		   = NULL;
	op->n_items		   = 0;
	op->n_misses	   = 0;

	struct moe_stream_layer *L = op->slayer;
	int						 need_fetch[MOE_MAX_K];
	int						 n_need = 0;

	resolve_scan_hits(c, L, m, expert_ids, n_k, op->now, out_slots, need_fetch, &n_need, 1);

	if (n_need == 0) {
		moe_stream_op_free(op);
		return NULL;
	}

	resolve_collect_misses(c, L, m, layer, expert_ids, op->now, out_slots, need_fetch, n_need,
						   op->misses, &op->n_misses, NULL, NULL);

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
		char pending[MOE_MAX_K];
		memset(pending, 0, sizeof(pending));
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
	free(op);
}

void moe_stream_wait_slot(const moe_expert_slot *slot) {
	if (!slot)
		return;
	int spins = 0;
	while (!atomic_load_explicit(&slot->io_ready, memory_order_acquire)) {
		if (spins < 10000) {
			cpu_relax();
			spins++;
		} else {
			sched_yield();
		}
	}
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
	if (*n == *cap) {
		*cap *= 2;
		*regions = xrealloc(*regions, *cap * sizeof(**regions));
	}
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