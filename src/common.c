#include "common.h"

#include <ctype.h>
#include <execinfo.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>

void oom_abort(size_t bytes) {
	fprintf(stderr, "fatal: out of memory (%zu bytes)\n", bytes);
	void *frames[32];
	int	  n = backtrace(frames, 32);
	backtrace_symbols_fd(frames, n, 2);
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
	char *p = strdup(s);
	if (!p)
		oom_abort(strlen(s));
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
	__asm__ __volatile__("isb" ::: "memory");
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

static float *float_buf_ensure_aligned(float_buf *b, size_t need, size_t align) {
	if (need > b->cap) {
		size_t bytes = need * sizeof(float);
		if (bytes >= (2u << 20))
			align = align < 4096 ? 4096 : align;
		float *np = xmalloc_aligned(bytes, align);
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

float *float_buf_ensure(float_buf *b, size_t need) {
	return float_buf_ensure_aligned(b, need, 64);
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
