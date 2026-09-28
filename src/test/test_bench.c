#include "test_core.h"
#ifdef BACKEND_CUDA
#include "backend/cuda/cuda_repack.h"
#endif

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
	case GGML_TYPE_Q4_K_R8:
		*base_type_out = GGML_TYPE_Q4_K;
		return repack_q4_k_to_q4_k_r8;
	case GGML_TYPE_Q5_K_R8:
		*base_type_out = GGML_TYPE_Q5_K;
		return repack_q5_k_to_q5_k_r8;
	case GGML_TYPE_Q6_K_R8:
		*base_type_out = GGML_TYPE_Q6_K;
		return repack_q6_k_to_q6_k_r8;
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
			weight_buf = base;
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

/* GEMV decode bench (G0 rig for GEMV efficiency work).                */
/*                                                                     */
/* --bench exercises matmul_batch with plain layouts (prefill/WMMA    */
/* kernels). Production single-token decode instead flows through     */
/* b->matmul / matmul_multi / matmul_residual / rmsnorm_matmul_multi  */
/* / matmul_ffn_down with QM-repacked weights (v2/multi decode       */
/* kernels). This rig measures those true decode paths at production  */
/* shapes, reporting us/call + weight-GB/s, plus a per-token FFN +    */
/* logits rollup for the gemma4-E2B arch mix (15x6k + 20x12k layers). */
/* Timing only; correctness is gated separately (suite + agreement).  */
/* ------------------------------------------------------------------ */

typedef struct {
	int			n, k;
	const char *tag;
} gemv_shape;

static const gemv_shape GEMV_PROD[] = {
	{6144, 1536, "ffn-up-6k"},
	{12288, 1536, "ffn-up-12k"},
	{1536, 6144, "ffn-dn-6k"},
	{1536, 12288, "ffn-dn-12k"},
	{262144, 1536, "logits"},
};
#define GEMV_PROD_N ((int)(sizeof(GEMV_PROD) / sizeof(GEMV_PROD[0])))

/* Deterministic plausible Q8_0 weights (timing only; d=1.0 avoids inf). */
static void gemv_fill_q8(q8_0_block *b0, int n, int nb) {
	for (int i = 0; i < n * nb; i++) {
		b0[i].d = 0x3C00;
		for (int j = 0; j < 32; j++)
			b0[i].qs[j] = (int8_t)((i * 31 + j * 7) % 64 - 32);
	}
}

static void gemv_upload_qm(backend *b, int n, int k, buffer *w_out) {
	int nb = k / 32;
	q8_0_block *b0 = xmalloc((size_t)n * (size_t)nb * sizeof(q8_0_block));
	gemv_fill_q8(b0, n, nb);
	/* QM rows are nb*34 bytes (32 groups of 34B). */
	uint8_t *bq = xmalloc((size_t)n * (size_t)nb * 34u);
#ifdef BACKEND_CUDA
	cuda_repack_q8_0_qm_rows(b0, bq, 0, n, k);
#else
	memcpy(bq, b0, (size_t)n * (size_t)nb * sizeof(q8_0_block));
#endif
	free(b0);
	tensor_desc wd = {.host_data = bq, .type = GGML_TYPE_Q8_0_QM, .n_dims = 2,
					  .dims = {(uint64_t)k, (uint64_t)n}};
	b->buffer_alloc_weight(b, &wd, w_out);
	free(bq);
}

static void gemv_fill_x(backend *b, int k, buffer *x_out) {
	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);
	b->buffer_alloc_scratch(b, (size_t)k * sizeof(float), x_out);
	b->buffer_write_f32(b, x_out, x, k);
	free(x);
}

typedef status_code (*gemv_op_fn)(void *ctx, backend *b);

typedef struct {
	backend *b;
	buffer *w, *x, *y;
	int n, k;
} gemv_mm_ctx;

static status_code gemv_op_mm(void *ctx, backend *b) {
	gemv_mm_ctx *c = ctx;
	return b->matmul(b, c->w, GGML_TYPE_Q8_0_QM, c->x, c->y, c->n, c->k);
}

typedef struct {
	backend *b;
	const buffer *wcg[2];
	buffer *ycg[2];
	const uint32_t *types;
	const int *n_list;
	buffer *x;
	int k;
} gemv_multi_ctx;

static status_code gemv_op_multi(void *ctx, backend *b) {
	gemv_multi_ctx *c = ctx;
	return b->matmul_multi(b, c->wcg, c->types, c->x, c->ycg, c->n_list, c->k, 2);
}

typedef struct {
	backend *b;
	buffer *w, *x, *r, *y;
	int n, k;
} gemv_res_ctx;

static status_code gemv_op_res(void *ctx, backend *b) {
	gemv_res_ctx *c = ctx;
	return b->matmul_residual(b, c->w, GGML_TYPE_Q8_0_QM, c->x, c->r, c->y, c->n, c->k);
}

typedef struct {
	backend *b;
	buffer *nw;
	const buffer *wcg[2];
	buffer *ycg[2];
	const uint32_t *types;
	const int *n_list;
	buffer *x;
	int k;
	float eps;
} gemv_rmm_ctx;

static status_code gemv_op_rmm(void *ctx, backend *b) {
	gemv_rmm_ctx *c = ctx;
	return b->rmsnorm_matmul_multi(b, c->nw, c->eps, c->wcg, c->types, c->x, c->ycg,
								   c->n_list, c->k, 2);
}

typedef struct {
	backend *b;
	buffer *w, *g, *u, *y;
	int n, k, act;
} gemv_fdn_ctx;

static status_code gemv_op_fdn(void *ctx, backend *b) {
	gemv_fdn_ctx *c = ctx;
	return b->matmul_ffn_down(b, c->w, GGML_TYPE_Q8_0_QM, c->g, c->u, c->y, c->n, c->k,
							  c->act);
}

/* Warmup 3 + reps timed (op + synchronize each, matching production
 * stream semantics). Returns mean us/call, or -1 on error. */
static double gemv_time(backend *b, gemv_op_fn op, void *ctx, int reps) {
	for (int i = 0; i < 3; i++) {
		if (op(ctx, b) != OK)
			return -1.0;
	}
	if (b->synchronize)
		b->synchronize(b);
	uint64_t t0 = time_us();
	for (int i = 0; i < reps; i++) {
		if (op(ctx, b) != OK)
			return -1.0;
		if (b->synchronize)
			b->synchronize(b);
	}
	return (double)(time_us() - t0) / (double)reps;
}

/* One finite check over y (catches catastrophic breakage while tuning). */
static int gemv_finite(backend *b, buffer *y, int n) {
	float *h = xmalloc((size_t)n * sizeof(float));
	if (b->buffer_read_f32(b, y, h, n) != OK) {
		free(h);
		return 0;
	}
	int ok = 1;
	for (int i = 0; i < n; i += (n / 32 + 1)) {
		if (!(h[i] == h[i]) || h[i] > 1e30f || h[i] < -1e30f) {
			ok = 0;
			break;
		}
	}
	free(h);
	return ok;
}

static double gemv_gbs(size_t wbytes, double us) {
	/* bytes/us == MB/s numerically; report GB/s. */
	return us > 0 ? (double)wbytes / us / 1000.0 : 0.0;
}

int run_gemv_bench_mode(int argc, char **argv, backend_info *infos, int n_backends) {
	int reps = 20;
	gemv_shape extra[8];
	int n_extra = 0;
	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--gemv") == 0 || strcmp(argv[ai], "--all") == 0)
			continue;
		if (strcmp(argv[ai], "--reps") == 0 && ai + 1 < argc) {
			reps = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--shape") == 0 && ai + 1 < argc && n_extra < 8) {
			int n = 0, k = 0;
			if (sscanf(argv[++ai], "%d,%d", &n, &k) == 2 && n > 0 && k > 0 &&
				k % 32 == 0) {
				extra[n_extra].n = n;
				extra[n_extra].k = k;
				extra[n_extra].tag = "custom";
				n_extra++;
			}
		} else if (argv[ai][0] != '-') {
			continue; /* backend name filter, handled below */
		} else {
			fprintf(stderr, "unknown gemv option: %s\n", argv[ai]);
			return 1;
		}
	}
	if (reps < 1)
		reps = 1;

	printf("\n=== gemv decode bench (QM Q8_0, true decode entry points, reps=%d) ===\n", reps);

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
		if (!b->matmul || !b->matmul_type_native ||
			!b->matmul_type_native(b, GGML_TYPE_Q8_0_QM)) {
			printf("\n=== %s: SKIPPED (no QM-native matmul) ===\n", infos[bi].name);
			backend_destroy(b);
			continue;
		}
		printf("\n[%s] QM decode GEMV (us/call, weight-GB/s)\n", infos[bi].name);
		printf("  %-12s %6s %6s  %-34s %-34s %-34s\n", "shape", "n", "k", "matmul",
			   "multi/rmm(x2)", "residual/ffn_down");

		/* Heat pass (untimed): GPU clocks ramp over the first seconds of
		 * load; without this, early shapes measure 1.5-2x slow and even
		 * identical shapes disagree within a run. */
		for (int hi = 0; hi < 2; hi++) {
			gemv_shape hs = GEMV_PROD[hi == 0 ? 1 : 4];
			buffer HW = {0}, HX = {0}, HY = {0};
			gemv_upload_qm(b, hs.n, hs.k, &HW);
			gemv_fill_x(b, hs.k, &HX);
			b->buffer_alloc_scratch(b, (size_t)hs.n * sizeof(float), &HY);
			gemv_mm_ctx hmc = {b, &HW, &HX, &HY, hs.n, hs.k};
			gemv_time(b, gemv_op_mm, &hmc, 5);
			b->buffer_free(b, &HW);
			b->buffer_free(b, &HX);
			b->buffer_free(b, &HY);
		}

		double rmm6 = -1, rmm12 = -1, fdn6 = -1, fdn12 = -1, logt = -1;
		for (int si = 0; si < GEMV_PROD_N + n_extra; si++) {
			gemv_shape s = si < GEMV_PROD_N ? GEMV_PROD[si] : extra[si - GEMV_PROD_N];
			int n = s.n, k = s.k, nb = k / 32;
			size_t wbytes = (size_t)n * (size_t)nb * 34u;

			/* Shared operands. */
			buffer W = {0}, X = {0}, Y = {0}, R = {0}, G = {0}, U = {0}, NW = {0};
			buffer W2 = {0}, Y2a = {0}, Y2b = {0};
			gemv_upload_qm(b, n, k, &W);
			gemv_upload_qm(b, n, k, &W2);
			gemv_fill_x(b, k, &X);
			gemv_fill_x(b, k, &G);
			gemv_fill_x(b, k, &U);
			b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &Y);
			b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &R);
			b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &Y2a);
			b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &Y2b);
			float *ones = xmalloc((size_t)k * sizeof(float));
			for (int i = 0; i < k; i++)
				ones[i] = 1.0f;
			b->buffer_alloc_scratch(b, (size_t)k * sizeof(float), &NW);
			b->buffer_write_f32(b, &NW, ones, k);
			free(ones);

			char mm[40], mu[40], rs[40];
			double t_mm = -1, t_mu = -1, t_rmm = -1, t_rs = -1, t_fdn = -1;

			gemv_mm_ctx mmc = {b, &W, &X, &Y, n, k};
			t_mm = gemv_time(b, gemv_op_mm, &mmc, reps);
			snprintf(mm, sizeof(mm), t_mm < 0 ? "ERR" : "%.1fus (%.0f)",
					 t_mm, gemv_gbs(wbytes, t_mm));

			if (b->matmul_multi && n <= 65536) {
				const buffer *wc[2] = {&W, &W2};
				buffer *yc[2] = {&Y2a, &Y2b};
				static const uint32_t t2[2] = {GGML_TYPE_Q8_0_QM, GGML_TYPE_Q8_0_QM};
				int nl[2] = {n, n};
				gemv_multi_ctx muc = {b, {wc[0], wc[1]}, {yc[0], yc[1]}, t2, nl, &X, k};
				t_mu = gemv_time(b, gemv_op_multi, &muc, reps);
			}
			if (b->rmsnorm_matmul_multi && n <= 65536) {
				const buffer *wc[2] = {&W, &W2};
				buffer *yc[2] = {&Y2a, &Y2b};
				static const uint32_t t2[2] = {GGML_TYPE_Q8_0_QM, GGML_TYPE_Q8_0_QM};
				int nl[2] = {n, n};
				gemv_rmm_ctx rmc = {b, &NW, {wc[0], wc[1]}, {yc[0], yc[1]}, t2, nl, &X, k,
									1e-5f};
				t_rmm = gemv_time(b, gemv_op_rmm, &rmc, reps);
			}
			{
				char a[18], c[18];
				snprintf(a, sizeof(a), t_mu >= 0 ? "%.1f (%.0f)" : "n/a",
						 t_mu, gemv_gbs(2 * wbytes, t_mu));
				snprintf(c, sizeof(c), t_rmm >= 0 ? "%.1f (%.0f)" : "n/a",
						 t_rmm, gemv_gbs(2 * wbytes, t_rmm));
				snprintf(mu, sizeof(mu), "%s / %s", a, c);
			}

			if (b->matmul_residual) {
				gemv_res_ctx rsc = {b, &W, &X, &R, &Y, n, k};
				t_rs = gemv_time(b, gemv_op_res, &rsc, reps);
			}
			if (b->matmul_ffn_down && k <= 12288) {
				gemv_fdn_ctx fdc = {b, &W, &G, &U, &Y, n, k, ACTIVATION_GELU};
				t_fdn = gemv_time(b, gemv_op_fdn, &fdc, reps);
			}
			if (t_rs < 0 && t_fdn < 0)
				snprintf(rs, sizeof(rs), "n/a");
			else {
				char a[18], c[18];
				if (t_rs >= 0)
					snprintf(a, sizeof(a), "%.1f (%.0f)", t_rs,
							 gemv_gbs(wbytes, t_rs));
				else
					snprintf(a, sizeof(a), "n/a");
				if (t_fdn >= 0)
					snprintf(c, sizeof(c), "%.1f (%.0f)", t_fdn,
							 gemv_gbs(wbytes, t_fdn));
				else
					snprintf(c, sizeof(c), "n/a");
				snprintf(rs, sizeof(rs), "%s / %s", a, c);
			}

			int fin = gemv_finite(b, &Y, n);
			printf("  %-12s %6d %6d  %-34s %-34s %-34s %s\n", s.tag, n, k, mm, mu,
				   rs, fin ? "" : "NONFINITE");

			/* Rollup inputs (best fused option per class). */
			if (n == 6144 && k == 1536)
				rmm6 = t_rmm >= 0 ? t_rmm : (t_mu >= 0 ? t_mu : 2 * t_mm);
			if (n == 12288 && k == 1536)
				rmm12 = t_rmm >= 0 ? t_rmm : (t_mu >= 0 ? t_mu : 2 * t_mm);
			if (n == 1536 && k == 6144)
				fdn6 = t_fdn >= 0 ? t_fdn : t_mm;
			if (n == 1536 && k == 12288)
				fdn12 = t_fdn >= 0 ? t_fdn : t_mm;
			if (n == 262144 && k == 1536)
				logt = t_mm;

			b->buffer_free(b, &W);
			b->buffer_free(b, &W2);
			b->buffer_free(b, &X);
			b->buffer_free(b, &Y);
			b->buffer_free(b, &R);
			b->buffer_free(b, &G);
			b->buffer_free(b, &U);
			b->buffer_free(b, &NW);
			b->buffer_free(b, &Y2a);
			b->buffer_free(b, &Y2b);
		}
		if (rmm6 > 0 && rmm12 > 0 && fdn6 > 0 && fdn12 > 0) {
			double ffn_ms = (15.0 * (rmm6 + fdn6) + 20.0 * (rmm12 + fdn12)) / 1000.0;
			printf("  per-token FFN GEMV: %.2fms (15x6k + 20x12k, fused-best)%s\n", ffn_ms,
				   logt > 0 ? "" : " (logits n/a)");
			if (logt > 0)
				printf("  per-token FFN+logits GEMV: %.2fms\n", ffn_ms + logt / 1000.0);
		}
		backend_destroy(b);
	}
	return 0;
}
