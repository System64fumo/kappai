#ifndef COMMON_H
#define COMMON_H

#include <execinfo.h>
#include <limits.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
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
} status_code;

typedef struct {
	float *p;
	size_t cap;
} float_buf;

static inline void madvise_hugepage(void *ptr, size_t bytes) {
	if (!ptr || bytes == 0)
		return;
#ifdef MADV_HUGEPAGE
	madvise(ptr, bytes, MADV_HUGEPAGE);
#endif
}

static inline void prefault(void *ptr, size_t bytes) {
	if (!ptr || bytes == 0)
		return;
	long ps = sysconf(_SC_PAGESIZE);
	if (ps <= 0)
		ps = 4096;
	volatile char *p = (volatile char *)ptr;
	for (size_t off = 0; off < bytes; off += (size_t)ps)
		p[off] = p[off];
	p[bytes - 1] = p[bytes - 1];
}

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
	__builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
	__asm__ __volatile__("isb" ::: "memory");
#else
	sched_yield();
#endif
}

static inline void oom_abort(size_t bytes) {
	fprintf(stderr, "fatal: out of memory (%zu bytes)\n", bytes);
	void *frames[32];
	int	  n = backtrace(frames, 32);
	backtrace_symbols_fd(frames, n, 2);
	abort();
}

static inline void *xmalloc(size_t n) {
	if (n == 0)
		n = 1;
	void *p = malloc(n);
	if (!p)
		oom_abort(n);
	return p;
}

static inline void *xmalloc_aligned(size_t n, size_t align) {
	if (align < sizeof(void *))
		align = sizeof(void *);
	void *p = NULL;
	if (posix_memalign(&p, align, n) != 0)
		oom_abort(n);
	return p;
}

static inline void *xcalloc(size_t n, size_t sz) {
	if (n == 0 || sz == 0)
		return NULL;
	if (n > SIZE_MAX / sz)
		oom_abort(n * sz);
	void *p = calloc(n, sz);
	if (!p)
		oom_abort(n * sz);
	return p;
}

static inline void *xrealloc(void *p, size_t n) {
	if (n == 0) {
		free(p);
		return NULL;
	}
	void *q = realloc(p, n);
	if (!q)
		oom_abort(n);
	return q;
}

static inline char *xstrdup(const char *s) {
	char *p = strdup(s);
	if (!p)
		oom_abort(strlen(s));
	return p;
}

static inline float *float_buf_ensure(float_buf *b, size_t need) {
	if (need > b->cap) {
		b->p   = xrealloc(b->p, need * sizeof(float));
		b->cap = need;
	}
	return b->p;
}

static inline uint64_t fnv1a(const char *s, size_t n) {
	uint64_t h = 0xcbf29ce484222325ULL;
	for (size_t i = 0; i < n; i++) {
		h ^= (uint8_t)s[i];
		h *= 0x100000001b3ULL;
	}
	return h;
}

static inline uint64_t fnv1a_str(const char *s) {
	return fnv1a(s, strlen(s));
}

#define ARR_RESERVE(items, n, cap)                                                                 \
	do {                                                                                           \
		if ((n) == (cap)) {                                                                        \
			(cap)	= (cap) ? (cap) * 2 : 8;                                                       \
			(items) = xrealloc((items), (cap) * sizeof(*(items)));                                 \
		}                                                                                          \
	} while (0)

#endif
