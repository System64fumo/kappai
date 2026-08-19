#include "test_core.h"

static double bench_mul_batch_once(backend *b, const buffer *w, uint32_t w_type, const buffer *xb,
								   buffer *yb, int n, int k, int m) {
	uint64_t	t0 = time_us();
	status_code s  = b->matmul_batch(b, w, w_type, xb, yb, n, k, m);
	if (b->synchronize)
		b->synchronize(b);
	(void)s;
	return (double)(time_us() - t0);
}

typedef void (*repack_fn)(const void *src, void *dst, int n_rows, int k);

static repack_fn bench_repack_fn(uint32_t type, uint32_t *base_type_out) {
	switch (type) {
	case GGML_TYPE_Q4_0_R8:
		*base_type_out = GGML_TYPE_Q4_0;
		return repack_q4_0_to_q4_0_r8;
	case GGML_TYPE_Q8_0_R8:
		*base_type_out = GGML_TYPE_Q8_0;
		return repack_q8_0_to_q8_0_r8;
	case GGML_TYPE_IQ4_NL_R8:
		*base_type_out = GGML_TYPE_IQ4_NL;
		return repack_iq4_nl_to_iq4_nl_r8;
	case GGML_TYPE_IQ3_S_RE:
		*base_type_out = GGML_TYPE_IQ3_S;
		return repack_iq3_s;
	case GGML_TYPE_IQ3_S_RE8:
		*base_type_out = GGML_TYPE_IQ3_S;
		return repack_iq3_s_to_iq3_s_re8;
	default:
		*base_type_out = type;
		return NULL;
	}
}

static void fill_random_f16(uint16_t *x, int n) {
	for (int i = 0; i < n; i++) {
		int32_t r = (int32_t)(next_u32() % 2001) - 1000;
		x[i]	  = f32_to_f16((float)r / 1000.0f);
	}
}

static void fill_random_bf16(uint16_t *x, int n) {
	for (int i = 0; i < n; i++) {
		int32_t	 r = (int32_t)(next_u32() % 2001) - 1000;
		float	 f = (float)r / 1000.0f;
		uint32_t bits;
		memcpy(&bits, &f, sizeof(bits));
		x[i] = (uint16_t)(bits >> 16);
	}
}

static double bench_mul_gflops(backend *b, const qtype_info *qt, int n, int k, int m, int iters) {
	seed_test_rng((0xBEEFULL * (qt->type + 1) * 1000003ULL) + (uint64_t)n);

	uint32_t  base_type = qt->type;
	repack_fn repack	= bench_repack_fn(qt->type, &base_type);

	void  *weight_buf;
	size_t weight_bytes;

	if (qt->type == GGML_TYPE_F32) {
		weight_bytes = (size_t)n * (size_t)k * sizeof(float);
		weight_buf	 = xmalloc(weight_bytes);
		fill_random_f32(weight_buf, n * k, 1.0f);
	} else if (qt->type == GGML_TYPE_F16) {
		weight_bytes = (size_t)n * (size_t)k * sizeof(uint16_t);
		weight_buf	 = xmalloc(weight_bytes);
		fill_random_f16(weight_buf, n * k);
	} else if (qt->type == GGML_TYPE_BF16) {
		weight_bytes = (size_t)n * (size_t)k * sizeof(uint16_t);
		weight_buf	 = xmalloc(weight_bytes);
		fill_random_bf16(weight_buf, n * k);
	} else {
		int	  n_blocks = n * (k / qt->block);
		void *base	   = xcalloc((size_t)n_blocks, qt->bytes);
		fill_random_blocks(base, n_blocks, qt->bytes, base_type);

		if (repack) {
			weight_bytes = (size_t)n * ggml_row_size(qt->type, (size_t)k);
			weight_buf	 = xmalloc(weight_bytes);
			repack(base, weight_buf, n, k);
			free(base);
		} else {
			weight_bytes = (size_t)n_blocks * qt->bytes;
			weight_buf	 = base;
		}
	}

	float *x = xmalloc((size_t)k * (size_t)m * sizeof(float));
	for (int t = 0; t < m; t++)
		fill_random_f32(x + (size_t)t * k, k, 1.0f);

	tensor_desc wd = {.host_data = weight_buf, .type = qt->type, .n_dims = 2, .dims = {k, n}};
	buffer		w = {0}, xb = {0}, yb = {0};
	b->buffer_alloc_weight(b, &wd, &w);
	b->buffer_alloc_scratch(b, (size_t)k * (size_t)m * sizeof(float), &xb);
	b->buffer_alloc_scratch(b, (size_t)n * (size_t)m * sizeof(float), &yb);
	b->buffer_write_f32(b, &xb, x, k * m);

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

int run_matmul_bench_mode(int argc, char **argv, backend_info *infos, int n_backends) {
	int		  n = 4096, k = 4096, iters = 5;
	int		  ms[8], n_ms = 0;
	const int def_ms[] = {1, 32, 128};
	for (unsigned i = 0; i < sizeof(def_ms) / sizeof(def_ms[0]); i++)
		ms[n_ms++] = def_ms[i];

	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--bench") == 0 || strcmp(argv[ai], "--all") == 0)
			continue;
		if (strcmp(argv[ai], "--n") == 0 && ai + 1 < argc) {
			n = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--k") == 0 && ai + 1 < argc) {
			k = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--iters") == 0 && ai + 1 < argc) {
			iters = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--m") == 0) {
			n_ms = 0;
			while (ai + 1 < argc && argv[ai + 1][0] != '-') {
				if (n_ms < 8)
					ms[n_ms++] = atoi(argv[++ai]);
				else
					ai++;
			}
		} else if (argv[ai][0] == '-' && strcmp(argv[ai], "--bench") != 0 &&
				   strcmp(argv[ai], "--all") != 0) {
			fprintf(stderr, "unknown bench option: %s\n", argv[ai]);
			usage(argv[0]);
			return 1;
		}
	}

	printf("\n=== matmul batch GFLOPS  N=%d K=%d  best-of-%d ----\n", n, k, iters);

	int do_all = wants_all(argc, argv);
	for (int bi = 0; bi < n_backends; bi++) {
		int want = do_all || strcmp(infos[bi].name, "cpu") == 0 ||
				   matches_name(argc, argv, infos[bi].name);
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
		printf("\n[%s]  N=%d K=%d  best-of-%d\n", infos[bi].name, n, k, iters);
		if (!b->matmul_batch) {
			printf("  SKIPPED: backend has no matmul_batch\n");
			backend_destroy(b);
			continue;
		}
		printf("  %-10s", "quant");
		for (int mi = 0; mi < n_ms; mi++) {
			printf("   M=%-6d", ms[mi]);
		}
		printf("\n");
		for (int qi = 0; qi < QTYPES_N; qi++) {
			if (k % QTYPES[qi].block != 0)
				continue;
			if (b->matmul_type_native && !b->matmul_type_native(b, QTYPES[qi].type))
				continue;
			uint32_t repack_base;
			if (bench_repack_fn(QTYPES[qi].type, &repack_base) && n % 8 != 0)
				continue;
			printf("  %-10s", QTYPES[qi].name);
			fflush(stdout);
			for (int mi = 0; mi < n_ms; mi++) {
				double g = bench_mul_gflops(b, &QTYPES[qi], n, k, ms[mi], iters);
				printf("   %7.1f", g);
				fflush(stdout);
			}
			printf("\n");
			fflush(stdout);
		}
		backend_destroy(b);
	}
	return 0;
}