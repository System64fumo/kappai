#include "config.h"
#include "context.h"
#include "gguf.h"
#include "sampler.h"
#include "test_core.h"
#include "tokenizer.h"

typedef int (*bench_backend_fn)(backend *b, const char *name, void *ud);

static int bench_for_each_backend(int argc, char **argv, backend_info *infos, int n_backends,
								  bench_backend_fn fn, void *ud) {
	int do_all	   = wants_all(argc, argv);
	int has_filter = do_all;
	for (int ai = 1; !has_filter && ai < argc; ai++) {
		if (argv[ai][0] == '-')
			continue;
		for (int bi = 0; bi < n_backends; bi++)
			if (strcmp(argv[ai], infos[bi].name) == 0)
				has_filter = 1;
	}
	int n_run = 0;
	for (int bi = 0; bi < n_backends; bi++) {
		int want = do_all || matches_name(argc, argv, infos[bi].name) ||
				   (!has_filter && (infos[bi].caps & BCAP_IS_HOST));
		if (!want)
			continue;
		if (!infos[bi].available) {
			printf("\n=== %s: SKIPPED (not available) ===\n", infos[bi].name);
			continue;
		}
		backend *b = NULL;
		if (backend_create(infos[bi].name, 0, &b) != OK) {
			printf("\n=== %s: SKIPPED (failed to init) ===\n", infos[bi].name);
			continue;
		}
		n_run += fn(b, infos[bi].name, ud) == 0;
		backend_destroy(b);
	}
	return n_run;
}

typedef struct {
	int n, k, iters;
	int ms[8], n_ms;
} matmul_bench_ud;

static double bench_mul_batch_once(backend *b, const buffer *w, uint32_t w_type, const buffer *xb,
								   buffer *yb, int n, int k, int m) {
	uint64_t	t0 = time_us();
	status_code s  = b->matmul_batch(b, w, w_type, xb, yb, n, k, m);
	if (b->synchronize)
		b->synchronize(b);
	(void)s;
	return (double)(time_us() - t0);
}

static double bench_mul_gflops(backend *b, const qtype_info *qt, int n, int k, int m, int iters) {
	seed_test_rng((0xBEEFULL * (qt->type + 1) * 1000003ULL) + (uint64_t)n);

	size_t weight_bytes = 0;
	void  *weight_buf	= test_make_weight(b, qt, n, k, &weight_bytes);
	if (!weight_buf) {
		fprintf(stderr, "  [bench setup failed for %s]\n", qt->name);
		return 0.0;
	}

	float *x = xmalloc((size_t)k * (size_t)m * sizeof(float));
	for (int t = 0; t < m; t++)
		fill_random_f32(x + (size_t)t * k, k, 1.0f);

	tensor_desc wd = {.host_data = weight_buf, .type = qt->type, .n_dims = 2, .dims = {k, n}};
	buffer		w = {0}, xb = {0}, yb = {0};
	if (b->buffer_alloc_weight(b, &wd, &w) != OK ||
		b->buffer_alloc_scratch(b, (size_t)k * (size_t)m * sizeof(float), &xb) != OK ||
		b->buffer_alloc_scratch(b, (size_t)n * (size_t)m * sizeof(float), &yb) != OK ||
		b->buffer_write_f32(b, &xb, x, k * m) != OK) {
		fprintf(stderr, "  [bench setup failed for %s]\n", qt->name);
		if (w.owner)
			b->buffer_free(b, &w);
		if (xb.owner)
			b->buffer_free(b, &xb);
		if (yb.owner)
			b->buffer_free(b, &yb);
		free(x);
		free(weight_buf);
		return 0.0;
	}

	for (int i = 0; i < 2; i++)
		bench_mul_batch_once(b, &w, qt->type, &xb, &yb, n, k, m);

	double best_us = 1e300;
	for (int i = 0; i < iters; i++) {
		double us = bench_mul_batch_once(b, &w, qt->type, &xb, &yb, n, k, m);
		if (us < best_us)
			best_us = us;
	}

	b->buffer_free(b, &w);
	b->buffer_free(b, &xb);
	b->buffer_free(b, &yb);
	free(x);
	free(weight_buf);

	return (2.0 * (double)n * (double)k * (double)m) / best_us * 1e-3;
}

static int bench_matmul_one(backend *b, const char *name, void *ud) {
	matmul_bench_ud *u = ud;
	printf("\n[%s]  N=%d K=%d  best-of-%d\n", name, u->n, u->k, u->iters);
	if (!b->matmul_batch) {
		printf("  SKIPPED: backend has no matmul_batch\n");
		return 0;
	}
	printf("  %-10s", "quant");
	for (int mi = 0; mi < u->n_ms; mi++)
		printf("   M=%-6d", u->ms[mi]);
	printf("\n");
	for (int qi = 0; qi < QTYPES_N; qi++) {
		if (u->k % QTYPES[qi].block != 0)
			continue;
		if (b->matmul_type_native && !b->matmul_type_native(b, QTYPES[qi].type))
			continue;
		uint32_t repack_base;
		if (test_repack_base_type(QTYPES[qi].type, &repack_base) && u->n % 8 != 0)
			continue;
		printf("  %-10s", QTYPES[qi].name);
		fflush(stdout);
		for (int mi = 0; mi < u->n_ms; mi++) {
			double g = bench_mul_gflops(b, &QTYPES[qi], u->n, u->k, u->ms[mi], u->iters);
			printf("   %7.1f", g);
			fflush(stdout);
		}
		printf("\n");
		fflush(stdout);
	}
	return 0;
}

typedef struct {
	int heads, kv_heads, head_dim, n_ctx, pos, flash;
} attn_bench_cfg;

static double bench_attn_once(backend *b, const buffer *q, const buffer *kc, const buffer *vc,
							  buffer *out, const attn_bench_cfg *c) {
	uint64_t	t0	  = time_us();
	float		scale = 1.0f / sqrtf((float)c->head_dim);
	status_code s = b->attention(b, q, kc, vc, out, 0, c->pos, c->heads, c->kv_heads, c->head_dim,
								 c->n_ctx, c->flash, scale, c->kv_heads);
	if (b->synchronize)
		b->synchronize(b);
	(void)s;
	return (double)(time_us() - t0);
}

static int bench_attn_one(backend *b, const char *name, void *ud) {
	int							iters  = *(int *)ud;
	static const attn_bench_cfg cfgs[] = {
		{8, 4, 64, 256, 127, 0},
		{8, 4, 64, 256, 127, 1},
		{32, 8, 128, 2048, 511, 0},
		{8, 1, 256, 4096, 2047, 0},
	};
	printf("\n[%s]  attention latency  best-of-%d\n", name, iters);
	if (!b->attention || !b->kv_alloc || !b->kv_put) {
		printf("  SKIPPED: backend has no attention/kv ops\n");
		return 0;
	}
	printf("  %-22s %10s %10s\n", "config", "us/call", "GB/s");
	for (unsigned ci = 0; ci < sizeof(cfgs) / sizeof(cfgs[0]); ci++) {
		const attn_bench_cfg *c	  = &cfgs[ci];
		int					  n	  = c->heads * c->head_dim;
		int					  nkv = c->kv_heads * c->head_dim;
		int					  nt  = c->pos + 1;

		seed_test_rng(0xA77EULL + ((uint64_t)c->heads * 131) + (uint64_t)c->pos);
		float *q = xmalloc((size_t)n * sizeof(float));
		fill_random_f32(q, n, 1.0f);

		kv_desc kvd = {.n_ctx		= c->n_ctx,
					   .n_kv_heads	= c->kv_heads,
					   .head_dim	= c->head_dim,
					   .n_layers	= 1,
					   .n_kv_layers = 1};
		buffer	kc = {0}, vc = {0}, ki = {0}, vi = {0}, qb = {0}, ob = {0};
		if (b->kv_alloc(b, &kvd, &kc, &vc) != OK ||
			b->buffer_alloc_scratch(b, (size_t)nkv * sizeof(float), &ki) != OK ||
			b->buffer_alloc_scratch(b, (size_t)nkv * sizeof(float), &vi) != OK ||
			b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &qb) != OK ||
			b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &ob) != OK) {
			printf("  h=%d kv=%d d=%d pos=%d flash=%d  SETUP-FAIL\n", c->heads, c->kv_heads,
				   c->head_dim, c->pos, c->flash);
			free(q);
			continue;
		}
		float *kv_one = xmalloc((size_t)nkv * sizeof(float));
		for (int t = 0; t < nt; t++) {
			fill_random_f32(kv_one, nkv, 1.0f);
			b->buffer_write_f32(b, &ki, kv_one, nkv);
			fill_random_f32(kv_one, nkv, 1.0f);
			b->buffer_write_f32(b, &vi, kv_one, nkv);
			b->kv_put(b, &kc, &vc, 0, t, &ki, &vi, c->kv_heads, c->head_dim, c->n_ctx, c->kv_heads);
		}
		free(kv_one);
		b->buffer_write_f32(b, &qb, q, n);
		free(q);

		for (int i = 0; i < 2; i++)
			bench_attn_once(b, &qb, &kc, &vc, &ob, c);
		double best_us = 1e300;
		for (int i = 0; i < iters; i++) {
			double us = bench_attn_once(b, &qb, &kc, &vc, &ob, c);
			if (us < best_us)
				best_us = us;
		}
		double kv_bytes = (double)nt * (double)c->kv_heads * (double)c->head_dim * 2.0 * 2.0;
		double gbs		= best_us > 0 ? kv_bytes / best_us / 1e3 : 0.0;
		printf("  h=%-3d kv=%-2d d=%-4d pos=%-5d fl=%d %10.1f %10.1f\n", c->heads, c->kv_heads,
			   c->head_dim, c->pos, c->flash, best_us, gbs);
		fflush(stdout);

		b->buffer_free(b, &kc);
		b->buffer_free(b, &vc);
		b->buffer_free(b, &ki);
		b->buffer_free(b, &vi);
		b->buffer_free(b, &qb);
		b->buffer_free(b, &ob);
	}
	return 0;
}

static int bench_dequant_one(backend *b, const char *name, void *ud) {
	int iters = *(int *)ud;
	printf("\n[%s]  dequant throughput  best-of-%d\n", name, iters);
	if (!b->dequant_row) {
		printf("  SKIPPED: backend has no dequant_row\n");
		return 0;
	}
	printf("  %-10s %10s %10s\n", "quant", "GB/s", "Gelem/s");
	for (int qi = 0; qi < QTYPES_N; qi++) {
		const qtype_info *qt = &QTYPES[qi];
		if (qt->type == GGML_TYPE_F32 || qt->type == GGML_TYPE_F16 || qt->type == GGML_TYPE_BF16)
			continue;
		int	   n_elems	= qt->block * 1024;
		int	   n_blocks = n_elems / qt->block;
		void  *blocks	= xcalloc((size_t)n_blocks, qt->bytes);
		float *dst		= xmalloc((size_t)n_elems * sizeof(float));
		seed_test_rng(0xDE9ULL + qt->type);
		fill_random_blocks(blocks, n_blocks, qt->bytes, qt->type);
		if (b->dequant_row(b, qt->type, blocks, n_elems, dst) != OK) {
			printf("  %-10s  SKIPPED (no native dequant)\n", qt->name);
			free(blocks);
			free(dst);
			continue;
		}
		for (int i = 0; i < 2; i++)
			b->dequant_row(b, qt->type, blocks, n_elems, dst);
		if (b->synchronize)
			b->synchronize(b);
		uint64_t best_us = UINT64_MAX;
		for (int i = 0; i < iters; i++) {
			uint64_t t0 = time_us();
			b->dequant_row(b, qt->type, blocks, n_elems, dst);
			if (b->synchronize)
				b->synchronize(b);
			uint64_t dt = time_us() - t0;
			if (dt < best_us)
				best_us = dt;
		}
		size_t in_bytes = (size_t)n_blocks * qt->bytes;
		double gbs		= best_us > 0 ? (double)in_bytes / (double)best_us / 1e3 : 0.0;
		double gels		= best_us > 0 ? (double)n_elems / (double)best_us / 1e3 : 0.0;
		printf("  %-10s %10.1f %10.1f\n", qt->name, gbs, gels);
		fflush(stdout);
		free(blocks);
		free(dst);
	}
	return 0;
}

static size_t tok_corpus_repetitive(char *out, size_t need) {
	static const char *words[] = {
		"The quick brown fox jumps over the lazy dog. ",
		"Please remember this exact phrase and continue with more words. ",
		"0123456789 !@#$ Oxcafe\xc3\xa9 \xe6\xb5\xaa\xe6\xbc\xab ",
		"Implementation details matter when measuring throughput. ",
		"\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86 benchmark suite. ",
	};
	size_t	 at = 0;
	unsigned wi = 0;
	while (at < need) {
		const char *wseg = words[wi++ % ARRAY_LEN(words)];
		size_t		wl	 = strlen(wseg);
		if (at + wl > need)
			break;
		memcpy(out + at, wseg, wl);
		at += wl;
	}
	out[at] = '\0';
	return at;
}

static size_t tok_corpus_high_entropy(char *out, size_t need) {
	static const char alpha[] =
		"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,'-\n";
	const size_t na = sizeof(alpha) - 1;
	uint64_t	 st = 0x9E3779B97F4A7C15ULL;
	size_t		 at = 0;
	while (at < need) {
		st ^= st << 13;
		st ^= st >> 7;
		st ^= st << 17;
		out[at++] = alpha[st % na];
	}
	out[at] = '\0';
	return at;
}

static int bench_tok_corpus(tokenizer *tok, const char *label, const char *corpus, size_t at,
							int iters) {
	int32_t *ids = xmalloc((at * 2 + 16) * sizeof(int32_t));
	int		 n	 = tokenizer_encode_with_specials(tok, corpus, 0, ids, (int)(at * 2 + 16), NULL);
	if (n <= 0) {
		fprintf(stderr, "tokenizer bench: encode failed\n");
		free(ids);
		return 1;
	}
	uint64_t enc_best = UINT64_MAX;
	for (int i = 0; i < 2 + iters; i++) {
		uint64_t t0 = time_us();
		n			= tokenizer_encode_with_specials(tok, corpus, 0, ids, (int)(at * 2 + 16), NULL);
		uint64_t dt = time_us() - t0;
		if (i >= 2 && dt < enc_best)
			enc_best = dt;
	}
	char	*rt		  = xmalloc(at + 64);
	int		 dl		  = -1;
	uint64_t dec_best = UINT64_MAX;
	for (int i = 0; i < 2 + iters; i++) {
		uint64_t t0 = time_us();
		dl			= tokenizer_decode(tok, ids, n, rt, (int)(at + 64), NULL);
		uint64_t dt = time_us() - t0;
		if (i >= 2 && dt < dec_best)
			dec_best = dt;
	}
	printf(
		"  %-12s %5.0f KiB -> %7d ids   encode %8.1f us %7.1f MB/s %8.0f ids/s   decode %7.1f us "
		"%6.1f MB/s  (roundtrip %s)\n",
		label, (double)at / 1024.0, n, (double)enc_best, (double)at / (double)enc_best,
		(double)n / (double)enc_best * 1e6, (double)dec_best, (double)at / (double)dec_best,
		(dl == (int)at && memcmp(rt, corpus, at) == 0) ? "exact" : "MISMATCH");
	free(ids);
	free(rt);
	return 0;
}

static int bench_tokenizer(int iters) {
	synth_suite_common_init();

	gguf_ctx  g;
	tokenizer tok;
	memset(&g, 0, sizeof(g));
	memset(&tok, 0, sizeof(tok));
	if (gguf_load(&g, synth_chat_model_path) != OK || tokenizer_init(&tok, &g) != OK) {
		fprintf(stderr, "tokenizer bench: failed to load %s\n", synth_chat_model_path);
		return 1;
	}

	size_t need	  = 256 * 1024;
	char  *corpus = xmalloc(need + 1);
	int	   rc	  = 0;

	printf("\n[tokenizer] best-of-%d\n", iters);
	rc |= bench_tok_corpus(&tok, "repetitive", corpus, tok_corpus_repetitive(corpus, need), iters);
	rc |= bench_tok_corpus(&tok, "high-entropy", corpus, tok_corpus_high_entropy(corpus, need),
						   iters);

	free(corpus);
	tokenizer_free(&tok);
	gguf_free(&g);
	return rc;
}

static void bench_noop_token(int32_t id, const char *piece, int pn, void *ud) {
	(void)id;
	(void)piece;
	(void)pn;
	(void)ud;
}

typedef struct {
	int n_prefill, n_decode;
} e2e_bench_ud;

static int bench_e2e_one(backend *b, const char *name, void *ud) {
	e2e_bench_ud *u = ud;
	(void)b;
	synth_suite_common_init();

	config cfg	   = config_defaults();
	cfg.device	   = name;
	cfg.model	   = synth_chat_model_path;
	cfg.use_mmap   = false;
	cfg.flash_attn = false;
	cfg.ctx_size   = u->n_prefill + u->n_decode + 64;
	cfg.seed	   = 7;

	context c;
	memset(&c, 0, sizeof(c));
	if (context_init(&c, &cfg) != OK) {
		printf("\n[%s] e2e: context_init failed\n", name);
		return 1;
	}

	char prompt[2048];
	snprintf(prompt, sizeof(prompt), "user: please remember this exact phrase %d\nassistant:", 42);
	int32_t ids[2048];
	int		n = tokenizer_encode_with_specials(&c.tok, prompt, 0, ids, 2048, NULL);
	if (n <= 0)
		n = 64;
	if (n > u->n_prefill)
		n = u->n_prefill;

	sampler_params sp;
	sp.temperature	  = 0.0f;
	sp.top_k		  = 1;
	sp.top_p		  = 1.0f;
	sp.min_p		  = 0.0f;
	sp.repeat_penalty = 1.0f;
	sp.repeat_last_n  = 64;

	prefill_result pf		   = context_prefill_tokens(&c, ids, n, "prefill", true);
	double		   prefill_tps = pf.rc >= 0 ? pf.tps : 0.0;

	context_reset(&c);
	uint64_t t0		 = time_us();
	int		 gen	 = context_completion(&c, prompt, u->n_decode, &sp, bench_noop_token, NULL);
	uint64_t turn_us = time_us() - t0;

	double turn_tps = (gen > 0 && turn_us > 0) ? (double)gen / (double)turn_us * 1e6 : 0.0;
	printf("\n[%s] e2e prefill=%d gen=%d\n", name, n, gen);
	printf("  prefill: %8.1f tok/s\n", prefill_tps);
	printf("  turn:    %8.1f tok/s (prefill+%d gen)\n", turn_tps, gen >= 0 ? gen : 0);

	context_free(&c);
	return 0;
}

int run_bench_mode(int argc, char **argv, backend_info *infos, int n_backends) {
	int		  n = 4096, k = 4096, iters = 5;
	int		  ms[8], n_ms = 0;
	const int def_ms[] = {1, 32, 128};
	for (unsigned i = 0; i < sizeof(def_ms) / sizeof(def_ms[0]); i++)
		ms[n_ms++] = def_ms[i];
	int n_prefill = 128, n_decode = 32;

	const char *which = "matmul";
	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--bench") == 0 || strcmp(argv[ai], "--all") == 0)
			continue;
		if (strcmp(argv[ai], "--n") == 0 && ai + 1 < argc) {
			n = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--k") == 0 && ai + 1 < argc) {
			k = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--iters") == 0 && ai + 1 < argc) {
			iters = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--n-prefill") == 0 && ai + 1 < argc) {
			n_prefill = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--n-decode") == 0 && ai + 1 < argc) {
			n_decode = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--m") == 0) {
			n_ms = 0;
			while (ai + 1 < argc && argv[ai + 1][0] != '-') {
				if (n_ms < 8)
					ms[n_ms++] = atoi(argv[++ai]);
				else
					ai++;
			}
		} else if (strcmp(argv[ai], "matmul") == 0 || strcmp(argv[ai], "attn") == 0 ||
				   strcmp(argv[ai], "tok") == 0 || strcmp(argv[ai], "tokenize") == 0 ||
				   strcmp(argv[ai], "dequant") == 0 || strcmp(argv[ai], "e2e") == 0 ||
				   strcmp(argv[ai], "all") == 0) {
			which = strcmp(argv[ai], "tokenize") == 0 ? "tok" : argv[ai];
		} else if (argv[ai][0] == '-' && strcmp(argv[ai], "--bench") != 0 &&
				   strcmp(argv[ai], "--all") != 0) {
			fprintf(stderr, "unknown bench option: %s\n", argv[ai]);
			usage(argv[0]);
			return 1;
		} else if (argv[ai][0] != '-') {
			int known = 0;
			for (int bi = 0; bi < n_backends; bi++)
				if (strcmp(argv[ai], infos[bi].name) == 0)
					known = 1;
			if (!known) {
				fprintf(stderr, "unknown bench target '%s'\n", argv[ai]);
				usage(argv[0]);
				return 1;
			}
		}
	}

	if (strcmp(which, "tok") == 0)
		return bench_tokenizer(iters);

	if (strcmp(which, "all") == 0) {
		int rc = bench_tokenizer(iters);
		if (rc != 0)
			return rc;
	}

	matmul_bench_ud mu = {.n = n, .k = k, .ms = {0}, .n_ms = n_ms, .iters = iters};
	for (int i = 0; i < n_ms; i++)
		mu.ms[i] = ms[i];
	if (strcmp(which, "matmul") == 0 || strcmp(which, "all") == 0) {
		printf("\n=== matmul batch GFLOPS  N=%d K=%d  best-of-%d ----\n", n, k, iters);
		bench_for_each_backend(argc, argv, infos, n_backends, bench_matmul_one, &mu);
	}
	if (strcmp(which, "attn") == 0 || strcmp(which, "all") == 0)
		bench_for_each_backend(argc, argv, infos, n_backends, bench_attn_one, &iters);
	if (strcmp(which, "dequant") == 0 || strcmp(which, "all") == 0)
		bench_for_each_backend(argc, argv, infos, n_backends, bench_dequant_one, &iters);
	if (strcmp(which, "e2e") == 0 || strcmp(which, "all") == 0) {
		e2e_bench_ud eu = {.n_prefill = n_prefill, .n_decode = n_decode};
		bench_for_each_backend(argc, argv, infos, n_backends, bench_e2e_one, &eu);
	}
	return 0;
}
