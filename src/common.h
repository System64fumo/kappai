#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__GLIBC__) && defined(__linux__)
#define HAVE_MALLOC 1
#else
#define HAVE_MALLOC 0
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

#define FNV1A_OFFSET_BASIS 0xcbf29ce484222325ULL

#define HEAD_DIM_MAX 512

typedef enum {
	OK				  = 0,
	ERR_IO			  = -1,
	ERR_FORMAT		  = -2,
	ERR_NOT_FOUND	  = -3,
	ERR_UNSUPPORTED	  = -4,
	ERR_OUT_OF_MEMORY = -5,
	ERR_INVALID_ARG	  = -6,
	ERR_INTERNAL	  = -7,
	ERR_INTERRUPTED	  = -8,
	ERR_FALLBACK	  = -9,
	ERR_COMPUTE_FAIL  = -10,
} status_code;

void  oom_abort(size_t bytes);
void *xmalloc(size_t n);
void *xmalloc_aligned(size_t n, size_t align);
void *xcalloc(size_t n, size_t sz);
void *xrealloc(void *p, size_t n);
char *xstrdup(const char *s);
char *xstrndup(const char *s, size_t n);

typedef int (*spin_until_fn)(void *ud);

void cpu_relax(void);
void spin_wait_relax(spin_until_fn pred, void *ud);

static inline void madvise_hugepage(void *ptr, size_t bytes) {
	if (!ptr || bytes == 0)
		return;
#ifdef MADV_HUGEPAGE
	madvise(ptr, bytes, MADV_HUGEPAGE);
#endif
}

static inline long page_size_cached(void) {
	static long ps = 0;
	if (ps <= 0) {
		ps = sysconf(_SC_PAGESIZE);
		if (ps <= 0)
			ps = 4096;
	}
	return ps;
}

static inline void prefault(void *ptr, size_t bytes) {
	if (!ptr || bytes == 0)
		return;
	size_t		   ps = (size_t)page_size_cached();
	volatile char *p  = (volatile char *)ptr;
	for (size_t off = 0; off < bytes; off += ps)
		p[off] = p[off];
	p[bytes - 1] = p[bytes - 1];
}

size_t		str_lcp_len(const char *a, const char *b);
const char *path_basename(const char *path);

typedef struct {
	uintptr_t start;
	size_t	  len;
} page_span;

static inline page_span page_span_for(const void *ptr, size_t bytes, size_t page_size) {
	page_span r = {0, 0};
	if (!ptr || bytes == 0 || page_size == 0)
		return r;
	uintptr_t addr = (uintptr_t)ptr;
	uintptr_t end  = addr + bytes;
	if (end < addr)
		return r;
	uintptr_t mask	 = ~((uintptr_t)page_size - 1);
	uintptr_t pstart = addr & mask;
	uintptr_t pend	 = (end + page_size - 1) & mask;
	if (pend < pstart)
		return r;
	r.start = pstart;
	r.len	= pend - pstart;
	return r;
}

static inline int page_span_clamp(page_span *r, uintptr_t base, size_t size) {
	if (!r || r->len == 0)
		return 0;
	if (r->start < base || r->start + r->len < r->start)
		return 0;
	uintptr_t stop = base + size;
	if (r->start + r->len > stop || stop < base)
		return 0;
	return 1;
}

typedef struct {
	const char *p;
	size_t		len;
} str_span;

str_span span_trim(const char *s, size_t len);
size_t	 marker_tail_len(const char *s, size_t len, const char *marker);

typedef struct {
	float *p;
	size_t cap;
} float_buf;

float *float_buf_ensure_nocopy(float_buf *b, size_t need, size_t align);
float *float_buf_ensure(float_buf *b, size_t need);

typedef struct {
	char  *p;
	size_t len;
	size_t cap;
} str_builder;

void  sb_reserve(str_builder *b, size_t extra);
void  sb_init(str_builder *b);
void  sb_putb(str_builder *b, const char *s, size_t n);
void  sb_puts(str_builder *b, const char *s);
void  sb_putc(str_builder *b, char c);
void  sb_reset(str_builder *b);
char *sb_finish(str_builder *b);
void  sb_free(str_builder *b);
void  str_replace_all(str_builder *out, const char *s, const char *from, const char *to);

typedef struct {
	char **chunks;
	size_t n_chunks;
	char  *cur;
	char  *end;
	size_t next_size;
} str_arena;

void  str_arena_init(str_arena *a);
char *str_arena_alloc(str_arena *a, size_t n);
void  str_arena_reserve(str_arena *a, size_t n);
char *str_arena_dup(str_arena *a, const char *s, size_t len);
void  str_arena_free(str_arena *a);

static inline void topk_heap_sift_down(float *score, int *idx, int n, int pos) {
	for (;;) {
		int l = 2 * pos + 1, r = 2 * pos + 2, smallest = pos;
		if (l < n && score[l] < score[smallest])
			smallest = l;
		if (r < n && score[r] < score[smallest])
			smallest = r;
		if (smallest == pos)
			return;
		float ts		= score[pos];
		score[pos]		= score[smallest];
		score[smallest] = ts;
		int ti			= idx[pos];
		idx[pos]		= idx[smallest];
		idx[smallest]	= ti;
		pos				= smallest;
	}
}

static inline int topk_heap_select(const float *scores, int n_scores, int k, float *out_score,
								   int *out_idx) {
	int hn = 0;
	for (int e = 0; e < n_scores && hn < k; e++) {
		out_idx[hn]	  = e;
		out_score[hn] = scores[e];
		int pos		  = hn++;
		while (pos > 0) {
			int parent = (pos - 1) / 2;
			if (out_score[parent] <= out_score[pos])
				break;
			float ts		  = out_score[pos];
			out_score[pos]	  = out_score[parent];
			out_score[parent] = ts;
			int ti			  = out_idx[pos];
			out_idx[pos]	  = out_idx[parent];
			out_idx[parent]	  = ti;
			pos				  = parent;
		}
	}
	for (int e = hn; e < n_scores; e++) {
		if (scores[e] > out_score[0]) {
			out_score[0] = scores[e];
			out_idx[0]	 = e;
			topk_heap_sift_down(out_score, out_idx, hn, 0);
		}
	}
	return hn;
}

static inline uint64_t fnv1a(const char *s, size_t n) {
	uint64_t h = FNV1A_OFFSET_BASIS;
	for (size_t i = 0; i < n; i++) {
		h ^= (uint8_t)s[i];
		h *= 0x100000001b3ULL;
	}
	return h;
}

static inline uint64_t fnv1a_str(const char *s) {
	return fnv1a(s, strlen(s));
}

static inline uint64_t fnv1a_update(uint64_t h, const char *s, size_t n) {
	for (size_t i = 0; i < n; i++) {
		h ^= (uint8_t)s[i];
		h *= 0x100000001b3ULL;
	}
	return h;
}

static inline uint64_t fnv1a_update_str(uint64_t h, const char *s) {
	return s ? fnv1a_update(h, s, strlen(s)) : h;
}

#define ARR_RESERVE(items, n, cap)                                                                 \
	do {                                                                                           \
		if ((n) == (cap)) {                                                                        \
			(cap)	= (cap) ? (cap) * 2 : 8;                                                       \
			(items) = xrealloc((items), (cap) * sizeof(*(items)));                                 \
		}                                                                                          \
	} while (0)

#define ARR_ENSURE(items, need, cap)                                                               \
	do {                                                                                           \
		if ((cap) < (need)) {                                                                      \
			(cap)	= (need);                                                                      \
			(items) = xrealloc((items), (size_t)(cap) * sizeof(*(items)));                         \
		}                                                                                          \
	} while (0)

#endif
