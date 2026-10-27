#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#include "common.h"

#include <ctype.h>
#include <fcntl.h>
#if defined(__GLIBC__) || defined(__APPLE__)
#include <execinfo.h>
#endif
#include <sched.h>
#include <stdio.h>
#include <string.h>

const char *status_str(status_code s) {
	switch (s) {
	case OK:
		return "ok";
	case ERR_IO:
		return "io error";
	case ERR_FORMAT:
		return "bad format";
	case ERR_NOT_FOUND:
		return "not found";
	case ERR_UNSUPPORTED:
		return "unsupported";
	case ERR_OUT_OF_MEMORY:
		return "out of memory";
	case ERR_INVALID_ARG:
		return "invalid argument";
	case ERR_INTERNAL:
		return "internal error";
	case ERR_INTERRUPTED:
		return "interrupted";
	case ERR_FALLBACK:
		return "fallback";
	case ERR_COMPUTE_FAIL:
		return "compute failed";
	default:
		return "unknown";
	}
}

void oom_abort(size_t bytes) {
	fprintf(stderr, "fatal: out of memory (%zu bytes)\n", bytes);
#if defined(__GLIBC__) || defined(__APPLE__)
	void *frames[32];
	int	  n = backtrace(frames, 32);
	backtrace_symbols_fd(frames, n, 2);
#endif
	abort();
}

void *xmalloc(size_t n) {
	if (n == 0)
		n = 1;
	void *p = malloc(n);
	if (!p)
		oom_abort(n);
	return p;
}

void *xmalloc_aligned(size_t n, size_t align) {
	if (align < sizeof(void *))
		align = sizeof(void *);
	void *p = NULL;
	if (posix_memalign(&p, align, n) != 0)
		oom_abort(n);
	return p;
}

void *xcalloc(size_t n, size_t sz) {
	if (n == 0 || sz == 0)
		return NULL;
	if (n > SIZE_MAX / sz)
		oom_abort(n * sz);
	void *p = calloc(n, sz);
	if (!p)
		oom_abort(n * sz);
	return p;
}

void *xrealloc(void *p, size_t n) {
	if (n == 0) {
		free(p);
		return NULL;
	}
	void *q = realloc(p, n);
	if (!q)
		oom_abort(n);
	return q;
}

char *xstrdup(const char *s) {
	size_t n = strlen(s) + 1;
	char  *p = xmalloc(n);
	memcpy(p, s, n);
	return p;
}

char *xstrndup(const char *s, size_t n) {
	char *p = xmalloc(n + 1);
	memcpy(p, s, n);
	p[n] = '\0';
	return p;
}

void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
	__builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
	__asm__ __volatile__("yield" ::: "memory");
#else
	sched_yield();
#endif
}

void spin_wait_relax(spin_until_fn pred, void *ud) {
	unsigned spins = 0;
	while (!pred(ud)) {
		if (++spins % 1024 == 0)
			sched_yield();
		else
			cpu_relax();
	}
}

void file_advise(int fd, off_t offset, off_t length, file_advice advice) {
	if (fd < 0 || offset < 0 || length < 0)
		return;
#if defined(POSIX_FADV_RANDOM) && defined(POSIX_FADV_DONTNEED)
	int native_advice;
	switch (advice) {
	case FILE_ADVICE_RANDOM:
		native_advice = POSIX_FADV_RANDOM;
		break;
	case FILE_ADVICE_RELEASE:
		native_advice = POSIX_FADV_DONTNEED;
		break;
	default:
		return;
	}
	posix_fadvise(fd, offset, length, native_advice);
#else
	(void)advice;
#endif
}

const char *str_find_n(const char *s, size_t len, const char *needle, size_t needle_len) {
	if (needle_len == 0)
		return s;
	if (needle_len > len)
		return NULL;
#if defined(__GLIBC__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) ||    \
	defined(__NetBSD__) || defined(__DragonFly__)
	return memmem(s, len, needle, needle_len);
#else
	size_t remaining = len - needle_len + 1;
	while (remaining > 0) {
		const char *p = memchr(s, (unsigned char)needle[0], remaining);
		if (!p)
			return NULL;
		if (memcmp(p, needle, needle_len) == 0)
			return p;
		remaining -= (size_t)(p - s) + 1;
		s = p + 1;
	}
	return NULL;
#endif
}

size_t str_lcp_len(const char *a, const char *b) {
	size_t i = 0;
	while (a[i] && b[i] && a[i] == b[i])
		i++;
	return i;
}

const char *path_basename(const char *path) {
	if (!path)
		return "";
	const char *slash = strrchr(path, '/');
	return slash ? slash + 1 : path;
}

str_span span_trim(const char *s, size_t len) {
	while (len > 0 && isspace((unsigned char)*s)) {
		s++;
		len--;
	}
	while (len > 0 && isspace((unsigned char)s[len - 1]))
		len--;
	str_span sp = {s, len};
	return sp;
}

size_t marker_tail_len(const char *s, size_t len, const char *marker) {
	if (!marker)
		return 0;
	size_t mlen = strlen(marker);
	if (mlen <= 1)
		return 0;
	size_t max = len < mlen - 1 ? len : mlen - 1;
	for (size_t k = max; k > 0; k--)
		if (memcmp(s + len - k, marker, k) == 0)
			return k;
	return 0;
}

float *float_buf_ensure(float_buf *b, size_t need) {
	if (need > b->cap) {
		size_t bytes = need * sizeof(float);
		size_t align = bytes >= (2u << 20) ? 4096 : 64;
		float *np	 = xmalloc_aligned(bytes, align);
		if (bytes >= (2u << 20))
			madvise_hugepage(np, bytes);
		if (b->p && b->cap > 0)
			memcpy(np, b->p, b->cap * sizeof(float));
		free(b->p);
		b->p   = np;
		b->cap = need;
	}
	return b->p;
}

float *float_buf_ensure_nocopy(float_buf *b, size_t need, size_t align) {
	if (need > b->cap) {
		size_t bytes = need * sizeof(float);
		if (bytes >= (2u << 20))
			align = align < 4096 ? 4096 : align;
		free(b->p);
		b->p = xmalloc_aligned(bytes, align);
		if (bytes >= (2u << 20))
			madvise_hugepage(b->p, bytes);
		b->cap = need;
	}
	return b->p;
}

void sb_reserve(str_builder *b, size_t extra) {
	size_t need = b->len + extra + 1;
	if (need <= b->cap)
		return;
	size_t cap = b->cap ? b->cap : 128;
	while (cap < need)
		cap *= 2;
	b->p   = xrealloc(b->p, cap);
	b->cap = cap;
	if (b->len == 0)
		b->p[0] = '\0';
}

void sb_init(str_builder *b) {
	b->len	= 0;
	b->cap	= 128;
	b->p	= xmalloc(b->cap);
	b->p[0] = '\0';
}

void sb_putb(str_builder *b, const char *s, size_t n) {
	if (!b || !s || n == 0)
		return;
	sb_reserve(b, n);
	memcpy(b->p + b->len, s, n);
	b->len += n;
	b->p[b->len] = '\0';
}

void sb_puts(str_builder *b, const char *s) {
	if (s)
		sb_putb(b, s, strlen(s));
}

void sb_putc(str_builder *b, char c) {
	sb_reserve(b, 1);
	b->p[b->len++] = c;
	b->p[b->len]   = '\0';
}

void sb_reset(str_builder *b) {
	if (!b)
		return;
	b->len = 0;
	if (b->p)
		b->p[0] = '\0';
}

char *sb_finish(str_builder *b) {
	sb_reserve(b, 0);
	char *p = b->p ? b->p : xstrdup("");
	memset(b, 0, sizeof(*b));
	return p;
}

void sb_free(str_builder *b) {
	free(b->p);
	memset(b, 0, sizeof(*b));
}

void str_replace_all(str_builder *out, const char *s, const char *from, const char *to) {
	size_t flen = strlen(from);
	if (flen == 0) {
		sb_puts(out, s);
		return;
	}
	const char *cur;
	while ((cur = strstr(s, from)) != NULL) {
		sb_putb(out, s, (size_t)(cur - s));
		sb_puts(out, to);
		s = cur + flen;
	}
	sb_puts(out, s);
}

void str_arena_init(str_arena *a) {
	memset(a, 0, sizeof(*a));
	a->next_size = 1u << 16;
}

char *str_arena_alloc(str_arena *a, size_t n) {
	if (n == 0)
		n = 1;
	if (!a->cur || (size_t)(a->end - a->cur) < n) {
		size_t sz = a->next_size;
		if (sz < n)
			sz = (n + 63u) & ~(size_t)63u;
		a->chunks			   = (char **)xrealloc(a->chunks, (a->n_chunks + 1) * sizeof(char *));
		a->chunks[a->n_chunks] = (char *)xmalloc(sz);
		a->n_chunks++;
		a->cur		 = a->chunks[a->n_chunks - 1];
		a->end		 = a->cur + sz;
		a->next_size = sz < (1u << 24) ? sz * 2 : sz;
	}
	char *p = a->cur;
	a->cur += n;
	return p;
}

void str_arena_reserve(str_arena *a, size_t n) {
	if (n == 0)
		return;
	if (a->cur && (size_t)(a->end - a->cur) >= n)
		return;
	a->chunks			   = (char **)xrealloc(a->chunks, (a->n_chunks + 1) * sizeof(char *));
	a->chunks[a->n_chunks] = (char *)xmalloc(n);
	a->n_chunks++;
	a->cur = a->chunks[a->n_chunks - 1];
	a->end = a->cur + n;
}

char *str_arena_dup(str_arena *a, const char *s, size_t len) {
	char *p = str_arena_alloc(a, len + 1);
	memcpy(p, s, len);
	p[len] = '\0';
	return p;
}

void str_arena_free(str_arena *a) {
	for (size_t i = 0; i < a->n_chunks; i++)
		free(a->chunks[i]);
	free(a->chunks);
	memset(a, 0, sizeof(*a));
}

static void advise_source(const void *ptr, size_t bytes, int advice) {
	page_span r = page_span_for(ptr, bytes, (size_t)page_size_cached());
	if (r.len > 0)
		madvise((void *)r.start, r.len, advice);
}

void weight_source_prefetch(const void *ptr, size_t bytes) {
	advise_source(ptr, bytes, MADV_WILLNEED);
}

void weight_source_release(const void *ptr, size_t bytes) {
	advise_source(ptr, bytes, MADV_DONTNEED);
}

void copy_weight_chunked(void *dst, const void *src, size_t bytes, int release_src) {
	if (!release_src) {
		memcpy(dst, src, bytes);
		return;
	}
	weight_source_prefetch(src, MIN(bytes, WEIGHT_COPY_CHUNK_BYTES));
	for (size_t off = 0; off < bytes; off += WEIGHT_COPY_CHUNK_BYTES) {
		size_t len	= MIN(bytes - off, WEIGHT_COPY_CHUNK_BYTES);
		size_t next = off + len;
		if (next < bytes)
			weight_source_prefetch((const char *)src + next,
								   MIN(bytes - next, WEIGHT_COPY_CHUNK_BYTES));
		memcpy((char *)dst + off, (const char *)src + off, len);
		weight_source_release((const char *)src + off, len);
	}
}
