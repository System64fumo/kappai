#include "test_core.h"

static int quant_dequant_supported(backend *ref, uint32_t type) {
	(void)type;
	return ref && ref->dequant_row;
}

static void test_dequant_twice(backend *ref, const qtype_info *qt, const void *blocks, int n,
							   float *dst1, float *dst2) {
	ref->dequant_row(ref, qt->type, blocks, n, dst1);
	ref->dequant_row(ref, qt->type, blocks, n, dst2);
}

void test_quant_determinism(backend *ref, const qtype_info *qt) {
	uint32_t base;
	if (test_repack_base_type(qt->type, &base))
		return;
	if (!quant_dequant_supported(ref, qt->type))
		return;

	int	   n_blocks = 8;
	int	   n		= n_blocks * qt->block;
	void  *blocks	= xcalloc((size_t)n_blocks, qt->bytes);
	float *dst1		= xmalloc((size_t)n * sizeof(float));
	float *dst2		= xmalloc((size_t)n * sizeof(float));

	seed_test_rng(0xD0DEULL + qt->type);
	fill_random_blocks(blocks, n_blocks, qt->bytes, qt->type);

	test_dequant_twice(ref, qt, blocks, n, dst1, dst2);

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

void test_quant_finiteness(backend *ref, const qtype_info *qt) {
	uint32_t base;
	if (test_repack_base_type(qt->type, &base))
		return;
	if (!quant_dequant_supported(ref, qt->type))
		return;

	int	   n_blocks = 32;
	int	   n		= n_blocks * qt->block;
	void  *blocks	= xcalloc((size_t)n_blocks, qt->bytes);
	float *dst		= xmalloc((size_t)n * sizeof(float));

	seed_test_rng(0xF1F0ULL + qt->type);
	fill_random_blocks(blocks, n_blocks, qt->bytes, qt->type);

	ref->dequant_row(ref, qt->type, blocks, n, dst);

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

static void test_quantize_q8_0(const float *x, void *dst, int n) {
	uint8_t *bp = dst;
	for (int i = 0; i < n; i += 32) {
		float amax = 0.0f;
		for (int j = 0; j < 32; j++) {
			float v = fabsf(x[i + j]);
			if (v > amax)
				amax = v;
		}
		float	 d	 = amax / 127.0f;
		float	 id	 = d ? 1.0f / d : 0.0f;
		uint16_t d16 = test_f32_to_f16(d);
		memcpy(bp, &d16, 2);
		for (int j = 0; j < 32; j++)
			bp[2 + j] = (int8_t)roundf(x[i + j] * id);
		bp += 34;
	}
}

void test_quant_q8_0_roundtrip(backend *ref) {
	if (!ref || !ref->dequant_row) {
		record_result(OPFAM_QUANT, "q8_0 round-trip (float->q8->float)", V_SKIP,
					  "backend has no dequant_row");
		return;
	}
	int	   n		= 256;
	int	   n_blocks = n / 32;
	float *src		= xmalloc((size_t)n * sizeof(float));
	void  *qblocks	= xmalloc((size_t)n_blocks * 34);
	float *dst		= xmalloc((size_t)n * sizeof(float));

	seed_test_rng(0xB80ULL);
	fill_random_f32(src, n, 2.0f);

	test_quantize_q8_0(src, qblocks, n);
	ref->dequant_row(ref, GGML_TYPE_Q8_0, qblocks, n, dst);

	char  label[96];
	char  detail[256];
	int	  at	  = -1;
	float max_abs = max_abs_diff_at(src, dst, n, &at);

	float max_step = 0;
	for (int bi = 0; bi < n_blocks; bi++) {
		uint16_t d16;
		memcpy(&d16, (uint8_t *)qblocks + (size_t)bi * 34, 2);
		float d = test_f16_to_f32(d16);
		if (d > max_step)
			max_step = d;
	}
	float half_step_err		  = max_step * 0.5f;
	float f16_scale_err		  = max_step * 127.0f / 2048.0f;
	float roundtrip_err_limit = half_step_err + f16_scale_err;

	snprintf(label, sizeof(label), "q8_0 round-trip (float->q8->float)");
	snprintf(detail, sizeof(detail), "max_abs=%.4e@%d src=%+.6f dst=%+.6f limit=%.4e", max_abs, at,
			 at >= 0 ? src[at] : 0, at >= 0 ? dst[at] : 0, roundtrip_err_limit);

	if (max_abs <= roundtrip_err_limit) {
		record_result(OPFAM_QUANT, label, V_PASS, detail);
	} else {
		compute_debug(src, dst, n);
		record_result(OPFAM_QUANT, label, V_FAIL, detail);
	}

	free(src);
	free(qblocks);
	free(dst);
}

typedef struct {
	uint32_t	src_type;
	uint32_t	rtype;
	int			k_mult;
	size_t		src_block;
	size_t		dst_block;
	const char *tag;
} repack_spec;

static const repack_spec REPACK_SPECS[] = {
	{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0_R8, 32, 34, 34, "q8_0_r8"},
	{GGML_TYPE_Q4_0, GGML_TYPE_Q4_0_R8, 32, 18, 18, "q4_0_r8"},
	{GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL_R8, 32, 18, 18, "iq4_nl_r8"},
	{GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_S_RE8, 256, 110, 134, "iq3_s_re8"},
	{GGML_TYPE_Q4_K, GGML_TYPE_Q4_K_R8, 256, 144, 148, "q4_k_r8"},
	{GGML_TYPE_Q5_K, GGML_TYPE_Q5_K_R8, 256, 176, 180, "q5_k_r8"},
	{GGML_TYPE_Q6_K, GGML_TYPE_Q6_K_R8, 256, 210, 210, "q6_k_r8"},
};
#define N_REPACK_SPECS ((int)(sizeof(REPACK_SPECS) / sizeof(REPACK_SPECS[0])))

static void repack_spec_alloc(const repack_spec *s, int n_rows, int k, void **src_out,
							  void **dst_out) {
	int	   bpr	   = k / s->k_mult;
	size_t row_dst = (size_t)bpr * s->dst_block;
	*src_out	   = xcalloc((size_t)n_rows * bpr, s->src_block);
	*dst_out	   = xmalloc((size_t)n_rows * row_dst);
	seed_test_rng((0x9E9ULL * s->src_type * 1009ULL) + ((uint64_t)n_rows * 41) + (uint64_t)k);
	fill_random_blocks(*src_out, n_rows * bpr, s->src_block, s->src_type);
}

static void test_repack_backend_matmul_parity(backend *ref, const repack_spec *s, int n, int k) {
	char label[160];
	if (!ref->matmul) {
		snprintf(label, sizeof(label), "%s backend matmul N=%d K=%d", s->tag, n, k);
		record_result(OPFAM_REPACK_PARITY, label, V_SKIP, "backend has no matmul");
		return;
	}

	void *src = NULL, *dst = NULL;
	repack_spec_alloc(s, n, k, &src, &dst);
	if (test_repack(ref, s->src_type, src, dst, n, k) != OK) {
		snprintf(label, sizeof(label), "%s backend matmul N=%d K=%d", s->tag, n, k);
		record_result(OPFAM_REPACK_PARITY, label, V_SKIP, "backend has no repack_weight");
		free(src);
		free(dst);
		return;
	}

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);

	tensor_desc wd_src = {.host_data = src, .type = s->src_type, .n_dims = 2, .dims = {k, n}};
	tensor_desc wd_dst = {.host_data = dst, .type = s->rtype, .n_dims = 2, .dims = {k, n}};

	buffer w_src = {0}, w_dst = {0}, xb = {0}, yb = {0};
	ref->buffer_alloc_weight(ref, &wd_src, &w_src);
	ref->buffer_alloc_weight(ref, &wd_dst, &w_dst);
	ref->buffer_alloc_scratch(ref, (size_t)k * sizeof(float), &xb);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &yb);
	ref->buffer_write_f32(ref, &xb, x, k);

	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_got = xmalloc((size_t)n * sizeof(float));
	ref->matmul(ref, &w_src, s->src_type, &xb, &yb, n, k);
	ref->buffer_read_f32(ref, &yb, y_ref, n);
	float *poison = xmalloc((size_t)n * sizeof(float));
	for (int i = 0; i < n; i++)
		poison[i] = (float)NAN;
	ref->buffer_write_f32(ref, &yb, poison, n);
	memset(y_got, 0, (size_t)n * sizeof(float));
	ref->matmul(ref, &w_dst, s->rtype, &xb, &yb, n, k);
	ref->buffer_read_f32(ref, &yb, y_got, n);
	snprintf(label, sizeof(label), "%s backend matmul parity N=%d K=%d", s->tag, n, k);
	test_parity_compare(OPFAM_REPACK_PARITY, label, y_ref, y_got, n, "loose");

	free(src);
	free(dst);
	free(x);
	free(y_ref);
	free(y_got);
	free(poison);
	ref->buffer_free(ref, &xb);
	ref->buffer_free(ref, &yb);
}

static void test_repack_backend_multi_parity(backend *ref, const repack_spec *s, int n1, int n2,
											 int k) {
	char label[160];
	if (!ref->matmul_multi) {
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
	if (test_repack(ref, s->src_type, src, dst, total, k) != OK) {
		snprintf(label, sizeof(label), "%s backend multi N=%d+%d K=%d", s->tag, n1, n2, k);
		record_result(OPFAM_REPACK_PARITY, label, V_SKIP, "backend has no repack_weight");
		free(src);
		free(dst);
		return;
	}

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
	ref->buffer_alloc_weight(ref, &wd_s0, &w_src0);
	ref->buffer_alloc_weight(ref, &wd_s1, &w_src1);
	ref->buffer_alloc_weight(ref, &wd_d0, &w_dst0);
	ref->buffer_alloc_weight(ref, &wd_d1, &w_dst1);
	ref->buffer_alloc_scratch(ref, (size_t)k * sizeof(float), &xb);
	ref->buffer_alloc_scratch(ref, (size_t)n1 * sizeof(float), &y0);
	ref->buffer_alloc_scratch(ref, (size_t)n2 * sizeof(float), &y1);
	ref->buffer_write_f32(ref, &xb, x, k);

	const buffer *w_src_list[2] = {&w_src0, &w_src1};
	const buffer *w_dst_list[2] = {&w_dst0, &w_dst1};
	uint32_t	  wt_src[2]		= {s->src_type, s->src_type};
	uint32_t	  wt_dst[2]		= {s->rtype, s->rtype};
	buffer		 *y_src_list[2] = {&y0, &y1};
	buffer		 *y_dst_list[2] = {&y0, &y1};
	int			  n_out[2]		= {n1, n2};

	float *y_ref = xmalloc((size_t)total * sizeof(float));
	float *y_got = xmalloc((size_t)total * sizeof(float));
	ref->matmul_multi(ref, w_src_list, wt_src, &xb, y_src_list, n_out, k, 2);
	ref->buffer_read_f32(ref, &y0, y_ref, n1);
	ref->buffer_read_f32(ref, &y1, y_ref + n1, n2);
	float *poison = xmalloc((size_t)total * sizeof(float));
	for (int i = 0; i < total; i++)
		poison[i] = (float)NAN;
	ref->buffer_write_f32(ref, &y0, poison, n1);
	ref->buffer_write_f32(ref, &y1, poison + n1, n2);
	ref->matmul_multi(ref, w_dst_list, wt_dst, &xb, y_dst_list, n_out, k, 2);
	ref->buffer_read_f32(ref, &y0, y_got, n1);
	ref->buffer_read_f32(ref, &y1, y_got + n1, n2);
	snprintf(label, sizeof(label), "%s backend multi parity N=%d+%d K=%d", s->tag, n1, n2, k);
	test_parity_compare(OPFAM_REPACK_PARITY, label, y_ref, y_got, total, "loose");

	free(src);
	free(dst);
	free(x);
	free(y_ref);
	free(y_got);
	free(poison);
	ref->buffer_free(ref, &xb);
	ref->buffer_free(ref, &y0);
	ref->buffer_free(ref, &y1);
}

static void test_repack_backend_batch_parity(backend *ref, const repack_spec *s, int n, int k,
											 int m) {
	char label[160];
	if (!ref->matmul_batch) {
		snprintf(label, sizeof(label), "%s backend batch N=%d K=%d M=%d", s->tag, n, k, m);
		record_result(OPFAM_REPACK_PARITY, label, V_SKIP, "backend has no matmul_batch");
		return;
	}

	void *src = NULL, *dst = NULL;
	repack_spec_alloc(s, n, k, &src, &dst);
	if (test_repack(ref, s->src_type, src, dst, n, k) != OK) {
		snprintf(label, sizeof(label), "%s backend batch N=%d K=%d M=%d", s->tag, n, k, m);
		record_result(OPFAM_REPACK_PARITY, label, V_SKIP, "backend has no repack_weight");
		free(src);
		free(dst);
		return;
	}

	float *x = xmalloc((size_t)k * (size_t)m * sizeof(float));
	for (int t = 0; t < m; t++)
		fill_random_f32(x + (size_t)t * k, k, 1.0f);

	tensor_desc wd_src = {.host_data = src, .type = s->src_type, .n_dims = 2, .dims = {k, n}};
	tensor_desc wd_dst = {.host_data = dst, .type = s->rtype, .n_dims = 2, .dims = {k, n}};

	buffer w_src = {0}, w_dst = {0}, xb = {0}, yb = {0};
	ref->buffer_alloc_weight(ref, &wd_src, &w_src);
	ref->buffer_alloc_weight(ref, &wd_dst, &w_dst);
	ref->buffer_alloc_scratch(ref, (size_t)k * (size_t)m * sizeof(float), &xb);
	ref->buffer_alloc_scratch(ref, (size_t)n * (size_t)m * sizeof(float), &yb);
	ref->buffer_write_f32(ref, &xb, x, k * m);

	float *y_ref = xmalloc((size_t)n * (size_t)m * sizeof(float));
	float *y_got = xmalloc((size_t)n * (size_t)m * sizeof(float));
	ref->matmul_batch(ref, &w_src, s->src_type, &xb, &yb, n, k, m);
	ref->buffer_read_f32(ref, &yb, y_ref, n * m);
	float *poison = xmalloc((size_t)n * (size_t)m * sizeof(float));
	for (int i = 0; i < n * m; i++)
		poison[i] = (float)NAN;
	ref->buffer_write_f32(ref, &yb, poison, n * m);
	memset(y_got, 0, (size_t)n * (size_t)m * sizeof(float));
	ref->matmul_batch(ref, &w_dst, s->rtype, &xb, &yb, n, k, m);
	ref->buffer_read_f32(ref, &yb, y_got, n * m);
	snprintf(label, sizeof(label), "%s backend batch parity N=%d K=%d M=%d", s->tag, n, k, m);
	test_parity_compare(OPFAM_REPACK_PARITY, label, y_ref, y_got, n * m, "loose");

	free(src);
	free(dst);
	free(x);
	free(y_ref);
	free(y_got);
	free(poison);
	ref->buffer_free(ref, &xb);
	ref->buffer_free(ref, &yb);
}

static void test_matmul_multi3_parity(backend *ref, backend *tgt, int n0, int n1, int n2, int k,
									  int m) {
	char label[176];
	snprintf(label, sizeof(label), "matmul_multi3_batch n=%d+%d+%d k=%d m=%d", n0, n1, n2, k, m);
	if (!ref->matmul_multi_batch || !tgt->matmul_multi_batch) {
		record_result(OPFAM_REPACK_PARITY, label, V_SKIP, "backend has no matmul_multi_batch");
		return;
	}

	const int	   ns[3]  = {n0, n1, n2};
	const uint32_t type	  = GGML_TYPE_Q4_0;
	const size_t   row_sz = ggml_row_size(type, k);
	const size_t   blk	  = row_sz / (size_t)(k / 32);
	const int	   total  = (n0 + n1 + n2) * m;
	float		  *x	  = xmalloc((size_t)k * (size_t)m * sizeof(float));
	float		  *y_ref  = xmalloc((size_t)total * sizeof(float));
	float		  *y_got  = xmalloc((size_t)total * sizeof(float));
	float		  *poison = xmalloc((size_t)total * sizeof(float));
	for (int i = 0; i < total; i++)
		poison[i] = (float)NAN;
	for (int t = 0; t < m; t++)
		fill_random_f32(x + (size_t)t * k, k, 1.0f);

	backend *bs[2] = {ref, tgt};
	for (int side = 0; side < 2; side++) {
		backend *b = bs[side];
		void	*wraw[3];
		buffer	 w[3] = {0}, xb = {0}, y[3] = {0};
		for (int i = 0; i < 3; i++) {
			wraw[i] = xmalloc((size_t)ns[i] * row_sz);
			seed_test_rng(0xA17C0000ULL + ((uint64_t)i * 7919ULL) + ((uint64_t)k << 8) +
						  (uint64_t)m);
			fill_random_blocks(wraw[i], ns[i] * (k / 32), blk, type);
			tensor_desc wd = {.host_data = wraw[i], .type = type, .n_dims = 2, .dims = {k, ns[i]}};
			b->buffer_alloc_weight(b, &wd, &w[i]);
		}
		b->buffer_alloc_scratch(b, (size_t)k * (size_t)m * sizeof(float), &xb);
		b->buffer_write_f32(b, &xb, x, k * m);
		for (int i = 0; i < 3; i++)
			b->buffer_alloc_scratch(b, (size_t)ns[i] * (size_t)m * sizeof(float), &y[i]);

		const buffer *w_list[3] = {&w[0], &w[1], &w[2]};
		uint32_t	  wt[3]		= {type, type, type};
		buffer		 *y_list[3] = {&y[0], &y[1], &y[2]};

		if (side == 0) {
			b->matmul_multi_batch(b, w_list, wt, &xb, y_list, (int *)ns, k, 3, m);
			int off = 0;
			for (int i = 0; i < 3; i++) {
				b->buffer_read_f32(b, &y[i], y_ref + off, ns[i] * m);
				off += ns[i] * m;
			}
		} else {
			int so = 0;
			for (int i = 0; i < 3; i++) {
				b->buffer_write_f32(b, &y[i], poison + so, ns[i] * m);
				so += ns[i] * m;
			}
			b->matmul_multi_batch(b, w_list, wt, &xb, y_list, (int *)ns, k, 3, m);
			int off = 0;
			for (int i = 0; i < 3; i++) {
				b->buffer_read_f32(b, &y[i], y_got + off, ns[i] * m);
				off += ns[i] * m;
			}
		}

		for (int i = 0; i < 3; i++) {
			free(wraw[i]);
			b->buffer_free(b, &w[i]);
			b->buffer_free(b, &y[i]);
		}
		b->buffer_free(b, &xb);
	}

	test_parity_compare(OPFAM_REPACK_PARITY, label, y_ref, y_got, total, "loose");

	free(x);
	free(y_ref);
	free(y_got);
	free(poison);
}

void run_repack_parity_tests(backend *ref, backend *tgt) {
	int shapes[][2] = {{32, 256}, {64, 512}, {64, 2048}, {512, 512}};
	for (int s = 0; s < N_REPACK_SPECS; s++) {
		for (int sh = 0; sh < (int)(sizeof(shapes) / sizeof(shapes[0])); sh++) {
			int n = shapes[sh][0];
			int k = shapes[sh][1];
			if (k % REPACK_SPECS[s].k_mult != 0 ||
				n % test_repack_n_align(REPACK_SPECS[s].rtype) != 0)
				continue;
			test_repack_backend_matmul_parity(ref, &REPACK_SPECS[s], n, k);
		}
	}

	int multi_shapes[][3] = {{32, 32, 256}, {64, 64, 256}, {128, 128, 2048}};
	for (int s = 0; s < N_REPACK_SPECS; s++) {
		for (int sh = 0; sh < (int)(sizeof(multi_shapes) / sizeof(multi_shapes[0])); sh++) {
			int n1 = multi_shapes[sh][0];
			int n2 = multi_shapes[sh][1];
			int k  = multi_shapes[sh][2];
			if (k % REPACK_SPECS[s].k_mult != 0 ||
				n1 % test_repack_n_align(REPACK_SPECS[s].rtype) != 0 ||
				n2 % test_repack_n_align(REPACK_SPECS[s].rtype) != 0)
				continue;
			test_repack_backend_multi_parity(ref, &REPACK_SPECS[s], n1, n2, k);
		}
	}

	int batch_shapes[][3] = {{64, 256, 2}, {64, 256, 8}, {128, 2048, 4}};
	for (int s = 0; s < N_REPACK_SPECS; s++) {
		for (int sh = 0; sh < (int)(sizeof(batch_shapes) / sizeof(batch_shapes[0])); sh++) {
			int n = batch_shapes[sh][0];
			int k = batch_shapes[sh][1];
			int m = batch_shapes[sh][2];
			if (k % REPACK_SPECS[s].k_mult != 0 ||
				n % test_repack_n_align(REPACK_SPECS[s].rtype) != 0)
				continue;
			test_repack_backend_batch_parity(ref, &REPACK_SPECS[s], n, k, m);
		}
	}

	{
		int triples[][5] = {
			{32, 32, 32, 256, 1}, {64, 64, 64, 512, 1},	 {128, 128, 128, 2048, 1},
			{96, 32, 64, 512, 1}, {32, 64, 96, 1024, 1}, {128, 32, 32, 2048, 4},
		};
		for (int sh = 0; sh < (int)(sizeof(triples) / sizeof(triples[0])); sh++)
			test_matmul_multi3_parity(ref, tgt, triples[sh][0], triples[sh][1], triples[sh][2],
									  triples[sh][3], triples[sh][4]);
	}
}

void test_dequant_parity_cross(backend *ref, backend *tgt, const qtype_info *qt, int dim,
							   int n_rows) {
	if (!tgt->embd_lookup) {
		char label[128];
		snprintf(label, sizeof(label), "%s dequant_parity dim=%d rows=%d", qt->name, dim, n_rows);
		record_result(OPFAM_DEQUANT_PARITY, label, V_SKIP, "backend has no native embd_lookup");
		return;
	}
	if (!ref || !ref->dequant_row) {
		char label[128];
		snprintf(label, sizeof(label), "%s dequant_parity dim=%d rows=%d", qt->name, dim, n_rows);
		record_result(OPFAM_DEQUANT_PARITY, label, V_SKIP, "reference has no dequant_row");
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
	blocks = test_make_weight(ref, qt, vocab, dim, NULL);
	if (!blocks) {
		char label[128];
		snprintf(label, sizeof(label), "%s dequant_parity dim=%d rows=%d", qt->name, dim, n_rows);
		record_result(OPFAM_DEQUANT_PARITY, label, V_SKIP, "reference cannot repack weight");
		return;
	}
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
		ref->dequant_row(ref, qt->type, row_blocks, dim, ref_all + (row * dim));
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
}

void test_dequant_weight_parity(backend *ref, backend *tgt, const qtype_info *qt, int k,
								int n_rows) {
	char label[160];
	snprintf(label, sizeof(label), "%s dequant_weight k=%d rows=%d", qt->name, k, n_rows);
	if (!tgt->dequant_weight) {
		record_result(OPFAM_DEQUANT_WEIGHT, label, V_SKIP, "backend has no bulk dequant_weight");
		return;
	}
	if (!ref || !ref->dequant_row) {
		record_result(OPFAM_DEQUANT_WEIGHT, label, V_SKIP, "reference has no dequant_row");
		return;
	}
	if (k % qt->block != 0)
		return;

	seed_test_rng(0xDE77ULL + qt->type + ((uint64_t)k * 31) + ((uint64_t)n_rows * 7));
	void *blocks = test_make_weight(ref, qt, n_rows, k, NULL);
	if (!blocks) {
		record_result(OPFAM_DEQUANT_WEIGHT, label, V_SKIP, "reference cannot repack weight");
		return;
	}

	const size_t row_stride = ggml_row_size(qt->type, (size_t)k);
	float		*ref_all	= xmalloc((size_t)n_rows * k * sizeof(float));
	for (int r = 0; r < n_rows; r++)
		ref->dequant_row(ref, qt->type, (const uint8_t *)blocks + (size_t)r * row_stride, k,
						 ref_all + (size_t)r * k);

	buffer		out = {0};
	status_code s	= tgt->dequant_weight(tgt, qt->type, blocks, n_rows, k, &out);
	if (s == OK && tgt->synchronize)
		tgt->synchronize(tgt);

	char detail[256];
	if (s != OK) {
		verdict v = (s == ERR_UNSUPPORTED) ? V_SKIP : V_FAIL;
		snprintf(detail, sizeof(detail), "dequant_weight status=%d", s);
		record_result(OPFAM_DEQUANT_WEIGHT, label, v, detail);
		free(ref_all);
		free(blocks);
		tgt->buffer_free(tgt, &out);
		return;
	}

	float *tgt_all = xmalloc((size_t)n_rows * k * sizeof(float));
	tgt->buffer_read_f32(tgt, &out, tgt_all, n_rows * k);

	int	  worst_row = 0;
	float worst_max = -1.0f;
	for (int r = 0; r < n_rows; r++) {
		int	  at;
		float mx = max_abs_diff_at(ref_all + (size_t)r * k, tgt_all + (size_t)r * k, k, &at);
		if (mx > worst_max) {
			worst_max = mx;
			worst_row = r;
		}
	}

	verdict v = classify_output("loose", ref_all + (size_t)worst_row * k,
								tgt_all + (size_t)worst_row * k, k, OK, detail, sizeof(detail));
	if (v != V_PASS)
		compute_debug(ref_all + (size_t)worst_row * k, tgt_all + (size_t)worst_row * k, k);
	record_result(OPFAM_DEQUANT_WEIGHT, label, v, detail);

	free(tgt_all);
	free(ref_all);
	free(blocks);
	tgt->buffer_free(tgt, &out);
}
