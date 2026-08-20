#include "test_core.h"

typedef void (*dequant_fn)(const void *blocks, size_t n_blocks, float *dst);

static dequant_fn get_dequant_fn(uint32_t type) {
	switch (type) {
	case GGML_TYPE_Q4_0:
		return dequant_q4_0_row;
	case GGML_TYPE_Q4_1:
		return dequant_q4_1_row;
	case GGML_TYPE_Q5_0:
		return dequant_q5_0_row;
	case GGML_TYPE_Q5_1:
		return dequant_q5_1_row;
	case GGML_TYPE_Q8_0:
		return dequant_q8_0_row;
	case GGML_TYPE_IQ4_NL:
		return dequant_iq4_nl_row;
	case GGML_TYPE_IQ4_XS:
		return dequant_iq4_xs_row;
	case GGML_TYPE_IQ2_XXS:
		return dequant_iq2_xxs_row;
	case GGML_TYPE_IQ2_XS:
		return dequant_iq2_xs_row;
	case GGML_TYPE_IQ2_S:
		return dequant_iq2_s_row;
	case GGML_TYPE_IQ3_XXS:
		return dequant_iq3_xxs_row;
	case GGML_TYPE_IQ1_S:
		return dequant_iq1_s_row;
	case GGML_TYPE_IQ1_M:
		return dequant_iq1_m_row;
	case GGML_TYPE_Q2_K:
		return dequant_q2_k_row;
	case GGML_TYPE_Q3_K:
		return dequant_q3_k_row;
	case GGML_TYPE_Q4_K:
		return dequant_q4_k_row;
	case GGML_TYPE_Q5_K:
		return dequant_q5_k_row;
	case GGML_TYPE_Q6_K:
		return dequant_q6_k_row;
	case GGML_TYPE_IQ3_S:
		return dequant_iq3_s_row;
	default:
		return NULL;
	}
}

typedef struct {
	uint32_t	src_type;
	uint32_t	rtype;
	int			k_mult;
	int			n_align;
	size_t		src_block;
	size_t		dst_block;
	const char *tag;
} repack_spec;

void test_quant_determinism(const qtype_info *qt) {
	dequant_fn dq = get_dequant_fn(qt->type);
	if (!dq)
		return;

	int	   n_blocks = 8;
	int	   n		= n_blocks * qt->block;
	void  *blocks	= xcalloc((size_t)n_blocks, qt->bytes);
	float *dst1		= xmalloc((size_t)n * sizeof(float));
	float *dst2		= xmalloc((size_t)n * sizeof(float));

	seed_test_rng(0xD0DEULL + qt->type);
	fill_random_blocks(blocks, n_blocks, qt->bytes, qt->type);

	dq(blocks, n_blocks, dst1);
	dq(blocks, n_blocks, dst2);

	int mismatch = 0;
	for (int i = 0; i < n; i++) {
		if (dst1[i] != dst2[i])
			mismatch++;
	}

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "%s dequant determinism", qt->name);
	if (mismatch == 0) {
		snprintf(detail, sizeof(detail), "%d elements, bit-exact", n);
		record_result(OPFAM_QUANT, label, V_PASS, detail);
	} else {
		if (mismatch > 0)
			compute_debug(dst1, dst2, n);
		snprintf(detail, sizeof(detail), "%d/%d elements differ", mismatch, n);
		record_result(OPFAM_QUANT, label, V_FAIL, detail);
	}

	free(blocks);
	free(dst1);
	free(dst2);
}

void test_quant_finiteness(const qtype_info *qt) {
	dequant_fn dq = get_dequant_fn(qt->type);
	if (!dq)
		return;

	int	   n_blocks = 32;
	int	   n		= n_blocks * qt->block;
	void  *blocks	= xcalloc((size_t)n_blocks, qt->bytes);
	float *dst		= xmalloc((size_t)n * sizeof(float));

	seed_test_rng(0xF1F0ULL + qt->type);
	fill_random_blocks(blocks, n_blocks, qt->bytes, qt->type);

	dq(blocks, n_blocks, dst);

	int	 nf = count_nonfinite(dst, n);
	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "%s dequant finiteness", qt->name);
	if (nf == 0) {
		float mx = max_abs_val(dst, n);
		snprintf(detail, sizeof(detail), "%d elements, |max|=%.4e", n, mx);
		record_result(OPFAM_QUANT, label, V_PASS, detail);
	} else {
		float *zeros = xcalloc((size_t)n, sizeof(float));
		compute_debug(zeros, dst, n);
		free(zeros);
		snprintf(detail, sizeof(detail), "%d/%d non-finite values", nf, n);
		record_result(OPFAM_QUANT, label, V_FAIL, detail);
	}

	free(blocks);
	free(dst);
}

void test_quant_q8_0_roundtrip(void) {
	int			n		 = 256;
	int			n_blocks = n / 32;
	float	   *src		 = xmalloc((size_t)n * sizeof(float));
	q8_0_block *qblocks	 = xmalloc((size_t)n_blocks * sizeof(q8_0_block));
	float	   *dst		 = xmalloc((size_t)n * sizeof(float));

	seed_test_rng(0xB80ULL);
	fill_random_f32(src, n, 2.0f);

	quantize_q8_0(src, qblocks, n);
	dequant_q8_0_row(qblocks, n_blocks, dst);

	char  label[96];
	char  detail[256];
	int	  at	  = -1;
	float max_abs = max_abs_diff_at(src, dst, n, &at);

	float max_step = 0;
	for (int bi = 0; bi < n_blocks; bi++) {
		float d = f16_to_f32(qblocks[bi].d);
		if (d > max_step)
			max_step = d;
	}
	float theory_max_err = max_step * 0.5f;

	snprintf(label, sizeof(label), "q8_0 round-trip (float->q8->float)");
	snprintf(detail, sizeof(detail), "max_abs=%.4e@%d src=%+.6f dst=%+.6f theory_max=%.4e", max_abs,
			 at, at >= 0 ? src[at] : 0, at >= 0 ? dst[at] : 0, theory_max_err);

	if (max_abs <= theory_max_err + 1e-6f) {
		record_result(OPFAM_QUANT, label, V_PASS, detail);
	} else if (max_abs <= theory_max_err * 2.0f) {
		compute_debug(src, dst, n);
		record_result(OPFAM_QUANT, label, V_LOSSY, detail);
	} else {
		compute_debug(src, dst, n);
		record_result(OPFAM_QUANT, label, V_FAIL, detail);
	}

	free(src);
	free(qblocks);
	free(dst);
}

static void test_repack_parity_compare(const char *label, const float *y_ref, const float *y_got,
									   int n, const char *tol_kind) {
	char	detail[256];
	verdict v = classify_output(tol_kind, y_ref, y_got, n, OK, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	record_result(OPFAM_REPACK_PARITY, label, v, detail);
}

static void test_repack_fallback_generic_parity_q8_0_r8(int n, int k) {
	if (k % 32 != 0 || n % Q8_0_R8_ROWS != 0)
		return;

	const int	 blocks_per_row = k / 32;
	const size_t n_blocks		= (size_t)n * (size_t)blocks_per_row;

	void *src	   = xcalloc(n_blocks, sizeof(q8_0_block));
	void *repacked = xmalloc(n_blocks * sizeof(q8_0_block));
	seed_test_rng((0xFAB1ULL * n) + (uint64_t)k + 7);
	fill_random_blocks(src, (int)n_blocks, sizeof(q8_0_block), GGML_TYPE_Q8_0);
	repack_q8_0_to_q8_0_r8(src, repacked, n, k);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 0.75f);
	q8_0_block *xq = xmalloc((size_t)blocks_per_row * sizeof(q8_0_block));
	quantize_q8_0(x, xq, k);

	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_fb	 = xmalloc((size_t)n * sizeof(float));

	matmul_q8_0_r8_q8_qonly_f32(repacked, xq, 0, y_ref, n, n, k, 1);
	matmul_generic_f32(repacked, GGML_TYPE_Q8_0_R8, x, y_fb, n, k);

	char label[128];
	char detail[256];
	snprintf(label, sizeof(label), "q8_0_r8 generic matmul fallback N=%d K=%d", n, k);

	float max_abs_p = 0;
	int	  n_nonzero = 0;
	int	  worst_abs = 0;
	for (int i = 0; i < n; i++) {
		float v = fabsf(y_fb[i]);
		if (v > max_abs_p)
			max_abs_p = v;
		if (fabsf(y_fb[i]) > 0.0f)
			n_nonzero++;
		float dev	= fabsf(y_fb[i] - y_ref[i]);
		int	  units = (int)(dev * 1e3f);
		if (units > worst_abs)
			worst_abs = units;
	}

	verdict v = V_PASS;
	if (n_nonzero == 0 || max_abs_p == 0.0f) {
		v = V_FAIL;
		snprintf(detail, sizeof(detail), "%d/%d nonzero rows -> all-zero output", n_nonzero, n);
		compute_debug(y_ref, y_fb, n);
	} else {
		verdict pv = classify_output("loose", y_ref, y_fb, n, OK, detail, sizeof(detail));
		if (pv == V_FAIL) {
			v = V_FAIL;
			compute_debug(y_ref, y_fb, n);
		}
	}
	int dl = (int)strlen(detail);
	snprintf(detail + dl, sizeof(detail) - dl, " | max dev=%.0f units", (float)worst_abs / 1000.0f);

	record_result(OPFAM_REPACK_PARITY, label, v, detail);

	free(x);
	free(xq);
	free(src);
	free(repacked);
	free(y_ref);
	free(y_fb);
}

static void test_repack_parity_q8_0_r8(int n, int k) {
	if (k % 32 != 0 || n % Q8_0_R8_ROWS != 0)
		return;

	const int	 blocks_per_row = k / 32;
	const size_t n_blocks		= (size_t)n * (size_t)blocks_per_row;

	void *src	   = xcalloc(n_blocks, sizeof(q8_0_block));
	void *repacked = xmalloc(n_blocks * sizeof(q8_0_block));
	seed_test_rng((0x5110ULL * n) + (uint64_t)k + 1);
	fill_random_blocks(src, (int)n_blocks, sizeof(q8_0_block), GGML_TYPE_Q8_0);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);
	q8_0_block *xq = xmalloc((size_t)blocks_per_row * sizeof(q8_0_block));
	quantize_q8_0(x, xq, k);

	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_got = xmalloc((size_t)n * sizeof(float));

	matmul_q8_0_q8_qonly_f32(src, xq, 0, y_ref, n, n, k, 1);
	repack_q8_0_to_q8_0_r8(src, repacked, n, k);
	matmul_q8_0_r8_q8_qonly_f32(repacked, xq, 0, y_got, n, n, k, 1);

	char label[128];
	snprintf(label, sizeof(label), "q8_0_r8 parity N=%d K=%d", n, k);
	test_repack_parity_compare(label, y_ref, y_got, n, "loose");

	free(x);
	free(xq);
	free(src);
	free(repacked);
	free(y_ref);
	free(y_got);
}

static void test_repack_fallback_generic_parity_q4_0_r8(int n, int k) {
	if (k % 32 != 0 || n % Q4_0_R8_ROWS != 0)
		return;

	const int	 blocks_per_row = k / 32;
	const size_t n_blocks		= (size_t)n * (size_t)blocks_per_row;

	void *src	   = xcalloc(n_blocks, sizeof(q4_0_block));
	void *repacked = xmalloc(n_blocks * sizeof(q4_0_block));
	seed_test_rng((0xFAB2ULL * n) + (uint64_t)k + 7);
	fill_random_blocks(src, (int)n_blocks, sizeof(q4_0_block), GGML_TYPE_Q4_0);
	repack_q4_0_to_q4_0_r8(src, repacked, n, k);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 0.75f);
	q8_0_block *xq = xmalloc((size_t)blocks_per_row * sizeof(q8_0_block));
	quantize_q8_0(x, xq, k);

	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_fb	 = xmalloc((size_t)n * sizeof(float));

	matmul_q4_0_r8_q8_qonly_f32(repacked, xq, 0, y_ref, n, n, k, 1);
	matmul_generic_f32(repacked, GGML_TYPE_Q4_0_R8, x, y_fb, n, k);

	char label[128];
	char detail[256];
	snprintf(label, sizeof(label), "q4_0_r8 generic matmul fallback N=%d K=%d", n, k);

	float max_abs_p = 0;
	int	  n_nonzero = 0;
	int	  worst_abs = 0;
	for (int i = 0; i < n; i++) {
		float v = fabsf(y_fb[i]);
		if (v > max_abs_p)
			max_abs_p = v;
		if (fabsf(y_fb[i]) > 0.0f)
			n_nonzero++;
		float dev	= fabsf(y_fb[i] - y_ref[i]);
		int	  units = (int)(dev * 1e3f);
		if (units > worst_abs)
			worst_abs = units;
	}

	verdict v = V_PASS;
	if (n_nonzero == 0 || max_abs_p == 0.0f) {
		v = V_FAIL;
		snprintf(detail, sizeof(detail), "%d/%d nonzero rows -> all-zero output", n_nonzero, n);
		compute_debug(y_ref, y_fb, n);
	} else {
		verdict pv = classify_output("loose", y_ref, y_fb, n, OK, detail, sizeof(detail));
		if (pv == V_FAIL) {
			v = V_FAIL;
			compute_debug(y_ref, y_fb, n);
		}
	}
	int dl = (int)strlen(detail);
	snprintf(detail + dl, sizeof(detail) - dl, " | max dev=%.0f units", (float)worst_abs / 1000.0f);

	record_result(OPFAM_REPACK_PARITY, label, v, detail);

	free(x);
	free(xq);
	free(src);
	free(repacked);
	free(y_ref);
	free(y_fb);
}

static void test_repack_parity_q4_0_r8(int n, int k) {
	if (k % 32 != 0 || n % Q4_0_R8_ROWS != 0)
		return;

	const int	 blocks_per_row = k / 32;
	const size_t n_blocks		= (size_t)n * (size_t)blocks_per_row;

	void *src	   = xcalloc(n_blocks, sizeof(q4_0_block));
	void *repacked = xmalloc(n_blocks * sizeof(q4_0_block));
	seed_test_rng((0x4110ULL * n) + (uint64_t)k + 1);
	fill_random_blocks(src, (int)n_blocks, sizeof(q4_0_block), GGML_TYPE_Q4_0);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);
	q8_0_block *xq = xmalloc((size_t)blocks_per_row * sizeof(q8_0_block));
	quantize_q8_0(x, xq, k);

	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_got = xmalloc((size_t)n * sizeof(float));

	matmul_q4_q8_qonly_f32(src, xq, 0, y_ref, n, n, k, 1);
	repack_q4_0_to_q4_0_r8(src, repacked, n, k);
	matmul_q4_0_r8_q8_qonly_f32(repacked, xq, 0, y_got, n, n, k, 1);

	char label[128];
	snprintf(label, sizeof(label), "q4_0_r8 parity N=%d K=%d", n, k);
	test_repack_parity_compare(label, y_ref, y_got, n, "loose");

	free(x);
	free(xq);
	free(src);
	free(repacked);
	free(y_ref);
	free(y_got);
}

static void test_repack_parity_iq4_nl_to_q8_0(int n, int k) {
	if (k % 32 != 0)
		return;

	const int	 blocks_per_row = k / 32;
	const size_t n_blocks		= (size_t)n * (size_t)blocks_per_row;

	void *src	   = xcalloc(n_blocks, sizeof(iq4_nl_block));
	void *repacked = xmalloc(n_blocks * sizeof(q8_0_block));
	seed_test_rng((0x4C11ULL * n) + (uint64_t)k + 1);
	fill_random_blocks(src, (int)n_blocks, sizeof(iq4_nl_block), GGML_TYPE_IQ4_NL);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);
	q8_0_block *xq = xmalloc((size_t)blocks_per_row * sizeof(q8_0_block));
	quantize_q8_0(x, xq, k);

	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_got = xmalloc((size_t)n * sizeof(float));

	matmul_iq4_nl_q8_qonly_f32(src, xq, 0, y_ref, n, n, k, 1);
	repack_iq4_nl_to_q8_0(src, repacked, n, k);
	matmul_q8_0_q8_qonly_f32(repacked, xq, 0, y_got, n, n, k, 1);

	char label[128];
	snprintf(label, sizeof(label), "iq4_nl->q8_0 parity N=%d K=%d", n, k);
	test_repack_parity_compare(label, y_ref, y_got, n, "bits");

	free(x);
	free(xq);
	free(src);
	free(repacked);
	free(y_ref);
	free(y_got);
}

static void test_repack_parity_iq3_s_re8(int n, int k) {
	if (k % 256 != 0 || n % IQ3_S_RE8_ROWS != 0)
		return;

	const int	 blocks_per_row = k / 256;
	const size_t n_blocks		= (size_t)n * (size_t)blocks_per_row;

	void *src	   = xcalloc(n_blocks, sizeof(iq3_s_block));
	void *repacked = xmalloc((size_t)n * (size_t)blocks_per_row * IQ3_S_RE8_GROUP_BYTES);
	seed_test_rng((0x3538ULL * n) + (uint64_t)k + 1);
	fill_random_blocks(src, (int)n_blocks, sizeof(iq3_s_block), GGML_TYPE_IQ3_S);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);
	q8_k_block *xq = xmalloc((size_t)blocks_per_row * sizeof(q8_k_block));
	quantize_q8_k(x, xq, k);

	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_got = xmalloc((size_t)n * sizeof(float));

	matmul_iq3_s_q8_k_qonly_f32(src, xq, 0, y_ref, n, n, k, 1);
	repack_iq3_s_to_iq3_s_re8(src, repacked, n, k);
	matmul_iq3_s_re8_q8_k_qonly_f32(repacked, xq, 0, y_got, n, n, k, 1);

	char label[128];
	snprintf(label, sizeof(label), "iq3_s_re8 parity N=%d K=%d", n, k);
	test_repack_parity_compare(label, y_ref, y_got, n, "loose");

	free(x);
	free(xq);
	free(src);
	free(repacked);
	free(y_ref);
	free(y_got);
}

static void test_repack_parity_iq3_s_re(int n, int k) {
	if (k % 256 != 0)
		return;

	const int	 blocks_per_row = k / 256;
	const size_t n_blocks		= (size_t)n * (size_t)blocks_per_row;

	void *src	   = xcalloc(n_blocks, sizeof(iq3_s_block));
	void *repacked = xmalloc((size_t)n * (size_t)blocks_per_row * IQ3_S_RE_BLOCK_BYTES);
	seed_test_rng((0x3531ULL * n) + (uint64_t)k + 1);
	fill_random_blocks(src, (int)n_blocks, sizeof(iq3_s_block), GGML_TYPE_IQ3_S);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);
	q8_k_block *xq = xmalloc((size_t)blocks_per_row * sizeof(q8_k_block));
	quantize_q8_k(x, xq, k);

	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_got = xmalloc((size_t)n * sizeof(float));

	matmul_iq3_s_q8_k_qonly_f32(src, xq, 0, y_ref, n, n, k, 1);
	repack_iq3_s(src, repacked, n, k);
	matmul_iq3_s_re_q8_k_qonly_f32(repacked, xq, 0, y_got, n, n, k, 1);

	char label[128];
	snprintf(label, sizeof(label), "iq3_s_re parity N=%d K=%d", n, k);
	test_repack_parity_compare(label, y_ref, y_got, n, "loose");

	free(x);
	free(xq);
	free(src);
	free(repacked);
	free(y_ref);
	free(y_got);
}

static const repack_spec REPACK_SPECS[] = {
	{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0_R8, 32, Q8_0_R8_ROWS, sizeof(q8_0_block), sizeof(q8_0_block),
	 "q8_0_r8"},
	{GGML_TYPE_Q4_0, GGML_TYPE_Q4_0_R8, 32, Q4_0_R8_ROWS, sizeof(q4_0_block), sizeof(q4_0_block),
	 "q4_0_r8"},
	{GGML_TYPE_Q4_K, GGML_TYPE_Q4_K_R8, 256, Q4_K_R8_ROWS, sizeof(q4_k_block), sizeof(q4_k_block),
	 "q4_k_r8"},
	{GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_S_RE8, 256, IQ3_S_RE8_ROWS, sizeof(iq3_s_block),
	 IQ3_S_RE8_GROUP_BYTES / IQ3_S_RE8_ROWS, "iq3_s_re8"},
};
#define N_REPACK_SPECS ((int)(sizeof(REPACK_SPECS) / sizeof(REPACK_SPECS[0])))

static void repack_spec_do(const repack_spec *s, const void *src, void *dst, int n_rows, int k) {
	if (s->src_type == GGML_TYPE_Q8_0)
		repack_q8_0_to_q8_0_r8(src, dst, n_rows, k);
	else if (s->src_type == GGML_TYPE_Q4_0)
		repack_q4_0_to_q4_0_r8(src, dst, n_rows, k);
	else if (s->src_type == GGML_TYPE_Q4_K)
		repack_q4_k_to_q4_k_r8(src, dst, n_rows, k);
	else
		repack_iq3_s_to_iq3_s_re8(src, dst, n_rows, k);
}

static void repack_spec_alloc(const repack_spec *s, int n_rows, int k, void **src_out,
							  void **dst_out) {
	int	   bpr	   = k / s->k_mult;
	size_t row_dst = (size_t)bpr * s->dst_block;
	*src_out	   = xcalloc((size_t)n_rows * bpr, s->src_block);
	*dst_out	   = xmalloc((size_t)n_rows * row_dst);
	seed_test_rng((0x9E9ULL * s->src_type * 1009ULL) + ((uint64_t)n_rows * 41) + (uint64_t)k);
	fill_random_blocks(*src_out, n_rows * bpr, s->src_block, s->src_type);
}

static void test_repack_backend_matmul_parity(backend *cpu, const repack_spec *s, int n, int k) {
	char label[160];
	if (!cpu->matmul) {
		snprintf(label, sizeof(label), "%s backend matmul N=%d K=%d", s->tag, n, k);
		record_result(OPFAM_REPACK_PARITY, label, V_SKIP, "backend has no matmul");
		return;
	}

	void *src = NULL, *dst = NULL;
	repack_spec_alloc(s, n, k, &src, &dst);
	repack_spec_do(s, src, dst, n, k);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);

	tensor_desc wd_src = {.host_data = src, .type = s->src_type, .n_dims = 2, .dims = {k, n}};
	tensor_desc wd_dst = {.host_data = dst, .type = s->rtype, .n_dims = 2, .dims = {k, n}};

	buffer w_src = {0}, w_dst = {0}, xb = {0}, yb = {0};
	cpu->buffer_alloc_weight(cpu, &wd_src, &w_src);
	cpu->buffer_alloc_weight(cpu, &wd_dst, &w_dst);
	cpu->buffer_alloc_scratch(cpu, (size_t)k * sizeof(float), &xb);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &yb);
	cpu->buffer_write_f32(cpu, &xb, x, k);

	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_got = xmalloc((size_t)n * sizeof(float));
	cpu->matmul(cpu, &w_src, s->src_type, &xb, &yb, n, k);
	cpu->buffer_read_f32(cpu, &yb, y_ref, n);
	float *poison = xmalloc((size_t)n * sizeof(float));
	for (int i = 0; i < n; i++)
		poison[i] = (float)NAN;
	cpu->buffer_write_f32(cpu, &yb, poison, n);
	memset(y_got, 0, (size_t)n * sizeof(float));
	cpu->matmul(cpu, &w_dst, s->rtype, &xb, &yb, n, k);
	cpu->buffer_read_f32(cpu, &yb, y_got, n);
	snprintf(label, sizeof(label), "%s backend matmul parity N=%d K=%d", s->tag, n, k);
	test_repack_parity_compare(label, y_ref, y_got, n, "loose");

	free(src);
	free(dst);
	free(x);
	free(y_ref);
	free(y_got);
	free(poison);
	cpu->buffer_free(cpu, &xb);
	cpu->buffer_free(cpu, &yb);
}

static void test_repack_backend_multi_parity(backend *cpu, const repack_spec *s, int n1, int n2,
											 int k) {
	char label[160];
	if (!cpu->matmul_multi) {
		snprintf(label, sizeof(label), "%s backend multi N=%d+%d K=%d", s->tag, n1, n2, k);
		record_result(OPFAM_REPACK_PARITY, label, V_SKIP, "backend has no matmul_multi");
		return;
	}

	int	   total   = n1 + n2;
	int	   bpr	   = k / s->k_mult;
	size_t row_src = (size_t)bpr * s->src_block;
	size_t row_dst = (size_t)bpr * s->dst_block;

	void *src = xcalloc((size_t)total * bpr, s->src_block);
	void *dst = xmalloc((size_t)total * row_dst);
	seed_test_rng((0xBEEFULL * s->src_type) + ((uint64_t)total * 17) + (uint64_t)k);
	fill_random_blocks(src, total * bpr, s->src_block, s->src_type);
	repack_spec_do(s, src, dst, total, k);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);

	tensor_desc wd_s0 = {.host_data = src, .type = s->src_type, .n_dims = 2, .dims = {k, total}};
	tensor_desc wd_s1 = {.host_data = (uint8_t *)src + (size_t)n1 * row_src,
						 .type		= s->src_type,
						 .n_dims	= 2,
						 .dims		= {k, n2}};
	tensor_desc wd_d0 = {.host_data = dst, .type = s->rtype, .n_dims = 2, .dims = {k, total}};
	tensor_desc wd_d1 = {.host_data = (uint8_t *)dst + (size_t)n1 * row_dst,
						 .type		= s->rtype,
						 .n_dims	= 2,
						 .dims		= {k, n2}};

	buffer w_src0 = {0}, w_src1 = {0}, w_dst0 = {0}, w_dst1 = {0}, xb = {0}, y0 = {0}, y1 = {0};
	cpu->buffer_alloc_weight(cpu, &wd_s0, &w_src0);
	cpu->buffer_alloc_weight(cpu, &wd_s1, &w_src1);
	cpu->buffer_alloc_weight(cpu, &wd_d0, &w_dst0);
	cpu->buffer_alloc_weight(cpu, &wd_d1, &w_dst1);
	cpu->buffer_alloc_scratch(cpu, (size_t)k * sizeof(float), &xb);
	cpu->buffer_alloc_scratch(cpu, (size_t)n1 * sizeof(float), &y0);
	cpu->buffer_alloc_scratch(cpu, (size_t)n2 * sizeof(float), &y1);
	cpu->buffer_write_f32(cpu, &xb, x, k);

	const buffer *w_src_list[2] = {&w_src0, &w_src1};
	const buffer *w_dst_list[2] = {&w_dst0, &w_dst1};
	uint32_t	  wt_src[2]		= {s->src_type, s->src_type};
	uint32_t	  wt_dst[2]		= {s->rtype, s->rtype};
	buffer		 *y_src_list[2] = {&y0, &y1};
	buffer		 *y_dst_list[2] = {&y0, &y1};
	int			  n_out[2]		= {n1, n2};

	float *y_ref = xmalloc((size_t)total * sizeof(float));
	float *y_got = xmalloc((size_t)total * sizeof(float));
	cpu->matmul_multi(cpu, w_src_list, wt_src, &xb, y_src_list, n_out, k, 2);
	cpu->buffer_read_f32(cpu, &y0, y_ref, n1);
	cpu->buffer_read_f32(cpu, &y1, y_ref + n1, n2);
	float *poison = xmalloc((size_t)total * sizeof(float));
	for (int i = 0; i < total; i++)
		poison[i] = (float)NAN;
	cpu->buffer_write_f32(cpu, &y0, poison, n1);
	cpu->buffer_write_f32(cpu, &y1, poison + n1, n2);
	cpu->matmul_multi(cpu, w_dst_list, wt_dst, &xb, y_dst_list, n_out, k, 2);
	cpu->buffer_read_f32(cpu, &y0, y_got, n1);
	cpu->buffer_read_f32(cpu, &y1, y_got + n1, n2);
	snprintf(label, sizeof(label), "%s backend multi parity N=%d+%d K=%d", s->tag, n1, n2, k);
	test_repack_parity_compare(label, y_ref, y_got, total, "loose");

	free(src);
	free(dst);
	free(x);
	free(y_ref);
	free(y_got);
	free(poison);
	cpu->buffer_free(cpu, &xb);
	cpu->buffer_free(cpu, &y0);
	cpu->buffer_free(cpu, &y1);
}

static void test_repack_backend_batch_parity(backend *cpu, const repack_spec *s, int n, int k,
											 int m) {
	char label[160];
	if (!cpu->matmul_batch) {
		snprintf(label, sizeof(label), "%s backend batch N=%d K=%d M=%d", s->tag, n, k, m);
		record_result(OPFAM_REPACK_PARITY, label, V_SKIP, "backend has no matmul_batch");
		return;
	}

	void *src = NULL, *dst = NULL;
	repack_spec_alloc(s, n, k, &src, &dst);
	repack_spec_do(s, src, dst, n, k);

	float *x = xmalloc((size_t)k * (size_t)m * sizeof(float));
	for (int t = 0; t < m; t++)
		fill_random_f32(x + (size_t)t * k, k, 1.0f);

	tensor_desc wd_src = {.host_data = src, .type = s->src_type, .n_dims = 2, .dims = {k, n}};
	tensor_desc wd_dst = {.host_data = dst, .type = s->rtype, .n_dims = 2, .dims = {k, n}};

	buffer w_src = {0}, w_dst = {0}, xb = {0}, yb = {0};
	cpu->buffer_alloc_weight(cpu, &wd_src, &w_src);
	cpu->buffer_alloc_weight(cpu, &wd_dst, &w_dst);
	cpu->buffer_alloc_scratch(cpu, (size_t)k * (size_t)m * sizeof(float), &xb);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * (size_t)m * sizeof(float), &yb);
	cpu->buffer_write_f32(cpu, &xb, x, k * m);

	float *y_ref = xmalloc((size_t)n * (size_t)m * sizeof(float));
	float *y_got = xmalloc((size_t)n * (size_t)m * sizeof(float));
	cpu->matmul_batch(cpu, &w_src, s->src_type, &xb, &yb, n, k, m);
	cpu->buffer_read_f32(cpu, &yb, y_ref, n * m);
	float *poison = xmalloc((size_t)n * (size_t)m * sizeof(float));
	for (int i = 0; i < n * m; i++)
		poison[i] = (float)NAN;
	cpu->buffer_write_f32(cpu, &yb, poison, n * m);
	memset(y_got, 0, (size_t)n * (size_t)m * sizeof(float));
	cpu->matmul_batch(cpu, &w_dst, s->rtype, &xb, &yb, n, k, m);
	cpu->buffer_read_f32(cpu, &yb, y_got, n * m);
	snprintf(label, sizeof(label), "%s backend batch parity N=%d K=%d M=%d", s->tag, n, k, m);
	test_repack_parity_compare(label, y_ref, y_got, n * m, "loose");

	free(src);
	free(dst);
	free(x);
	free(y_ref);
	free(y_got);
	free(poison);
	cpu->buffer_free(cpu, &xb);
	cpu->buffer_free(cpu, &yb);
}

void run_repack_parity_tests(backend *cpu) {
	int shapes[][2] = {{32, 256}, {64, 512}, {64, 2048}};
	for (int sh = 0; sh < (int)(sizeof(shapes) / sizeof(shapes[0])); sh++) {
		int n = shapes[sh][0];
		int k = shapes[sh][1];
		test_repack_parity_q8_0_r8(n, k);
		test_repack_parity_q4_0_r8(n, k);
		test_repack_parity_iq4_nl_to_q8_0(n, k);
		test_repack_parity_iq3_s_re8(n, k);
		test_repack_parity_iq3_s_re(n, k);
		test_repack_fallback_generic_parity_q8_0_r8(n, k);
		test_repack_fallback_generic_parity_q4_0_r8(n, k);
	}

	for (int s = 0; s < N_REPACK_SPECS; s++) {
		for (int sh = 0; sh < (int)(sizeof(shapes) / sizeof(shapes[0])); sh++) {
			int n = shapes[sh][0];
			int k = shapes[sh][1];
			if (k % REPACK_SPECS[s].k_mult != 0 || n % REPACK_SPECS[s].n_align != 0)
				continue;
			test_repack_backend_matmul_parity(cpu, &REPACK_SPECS[s], n, k);
		}
	}

	int multi_shapes[][3] = {{32, 32, 256}, {64, 64, 256}, {128, 128, 2048}};
	for (int s = 0; s < N_REPACK_SPECS; s++) {
		for (int sh = 0; sh < (int)(sizeof(multi_shapes) / sizeof(multi_shapes[0])); sh++) {
			int n1 = multi_shapes[sh][0];
			int n2 = multi_shapes[sh][1];
			int k  = multi_shapes[sh][2];
			if (k % REPACK_SPECS[s].k_mult != 0 || n1 % REPACK_SPECS[s].n_align != 0 ||
				n2 % REPACK_SPECS[s].n_align != 0)
				continue;
			test_repack_backend_multi_parity(cpu, &REPACK_SPECS[s], n1, n2, k);
		}
	}

	int batch_shapes[][3] = {{64, 256, 2}, {64, 256, 8}, {128, 2048, 4}};
	for (int s = 0; s < N_REPACK_SPECS; s++) {
		for (int sh = 0; sh < (int)(sizeof(batch_shapes) / sizeof(batch_shapes[0])); sh++) {
			int n = batch_shapes[sh][0];
			int k = batch_shapes[sh][1];
			int m = batch_shapes[sh][2];
			if (k % REPACK_SPECS[s].k_mult != 0 || n % REPACK_SPECS[s].n_align != 0)
				continue;
			test_repack_backend_batch_parity(cpu, &REPACK_SPECS[s], n, k, m);
		}
	}
}

void test_dequant_parity_cross(backend *cpu, backend *tgt, const qtype_info *qt, int dim,
							   int n_rows) {
	if (!tgt->embd_lookup) {
		char label[128];
		snprintf(label, sizeof(label), "%s dequant_parity dim=%d rows=%d", qt->name, dim, n_rows);
		record_result(OPFAM_DEQUANT_PARITY, label, V_SKIP, "backend has no native embd_lookup");
		return;
	}
	if (dim % qt->block != 0)
		return;
	if (!test_type_per_row(qt->type)) {
		char label[128];
		snprintf(label, sizeof(label), "%s dequant_parity dim=%d rows=%d (skip)", qt->name, dim,
				 n_rows);
		record_result(OPFAM_DEQUANT_PARITY, label, V_SKIP,
					  "group-repacked type has no per-row lookup");
		return;
	}

	int	  vocab = n_rows;
	void *blocks;
	seed_test_rng(0xDE44ULL + qt->type + ((uint64_t)dim * 17));
	blocks					= test_make_weight(qt, vocab, dim, NULL);
	const size_t row_stride = ggml_row_size(qt->type, (size_t)dim);

	tensor_desc wd = {
		.host_data = blocks,
		.type	   = qt->type,
		.n_dims	   = 2,
		.dims	   = {dim, vocab},
	};

	float *ref_all = xmalloc((size_t)vocab * dim * sizeof(float));
	for (int row = 0; row < vocab; row++) {
		const void *row_blocks = (const uint8_t *)blocks + (size_t)row * row_stride;
		dequant_row_dispatch(qt->type, row_blocks, dim, ref_all + (row * dim));
	}

	buffer w_tgt   = {0};
	buffer out_tgt = {0};
	tgt->buffer_alloc_weight(tgt, &wd, &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)dim * sizeof(float), &out_tgt);

	float *tgt_all = xmalloc((size_t)vocab * dim * sizeof(float));

	int			worst_row	   = 0;
	float		worst_max_abs  = -1.0f;
	int			n_failing_rows = 0;
	status_code s_tgt		   = OK;
	for (int row = 0; row < vocab; row++) {

		s_tgt = tgt->embd_lookup(tgt, &w_tgt, qt->type, row, dim, &out_tgt);
		if (tgt->synchronize)
			tgt->synchronize(tgt);

		if (s_tgt != OK)
			break;

		tgt->buffer_read_f32(tgt, &out_tgt, tgt_all + (row * dim), dim);

		float *ref_row = ref_all + (row * dim);
		float *tgt_row = tgt_all + (row * dim);
		int	   at;
		float  row_max = max_abs_diff_at(ref_row, tgt_row, dim, &at);
		if (row_max > worst_max_abs) {
			worst_max_abs = row_max;
			worst_row	  = row;
		}
		float ratio = max_combined_ratio_at(ref_row, tgt_row, dim, ATOL_LOOSE, RTOL_LOOSE, NULL);
		if (ratio > 1.0f)
			n_failing_rows++;
	}

	if (s_tgt != OK) {
		char label[128];
		char detail[256];
		snprintf(label, sizeof(label), "%s dequant_parity dim=%d rows=%d", qt->name, dim, n_rows);
		verdict v = (s_tgt == ERR_UNSUPPORTED) ? V_SKIP : V_FAIL;
		if (v == V_SKIP)
			snprintf(detail, sizeof(detail), "missing native implementation");
		else
			snprintf(detail, sizeof(detail), "embd_lookup status=%d", s_tgt);
		record_result(OPFAM_DEQUANT_PARITY, label, v, detail);
		free(blocks);
		free(ref_all);
		free(tgt_all);
		tgt->buffer_free(tgt, &w_tgt);
		tgt->buffer_free(tgt, &out_tgt);
		return;
	}

	for (int row = 0; row < vocab; row++) {
		const void *row_blocks = (const uint8_t *)blocks + (size_t)row * row_stride;
		float		tmp[4096];
		dequant_row_dispatch(qt->type, row_blocks, dim, tmp);
	}
	float *ref_worst = ref_all + (worst_row * dim);
	float *tgt_worst = tgt_all + (worst_row * dim);

	char label[128];
	char detail[256];
	snprintf(label, sizeof(label), "%s dequant_parity dim=%d rows=%d", qt->name, dim, n_rows);

	verdict v = classify_output("loose", ref_worst, tgt_worst, dim, OK, detail, sizeof(detail));
	if (v != V_PASS)
		compute_debug(ref_worst, tgt_worst, dim);
	int dl = (int)strlen(detail);
	snprintf(detail + dl, sizeof(detail) - dl, " | worst_row=%d/%d failing max_abs=%.4e", worst_row,
			 n_failing_rows, worst_max_abs);

	record_result(OPFAM_DEQUANT_PARITY, label, v, detail);

	free(ref_all);
	free(tgt_all);
	free(blocks);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &out_tgt);
	(void)cpu;
}
