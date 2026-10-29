#include "backend/cpu/quants.h"
#include "config.h"
#include "test_core.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static void rotate_reference(float *values, int k) {
	int block = 1;
	while (block < 512 && k % (block * 2) == 0)
		block *= 2;
	float scale = 1.0f / sqrtf((float)block);
	for (int j = 0; j < k; j++) {
		uint32_t bits = (uint32_t)j + 0x9e3779b9u;
		bits ^= bits >> 16;
		bits *= 0x85ebca6bu;
		bits ^= bits >> 13;
		values[j] *= bits & 1 ? -scale : scale;
	}
	for (int base = 0; base < k; base += block)
		for (int half = 1; half < block; half *= 2)
			for (int start = 0; start < block; start += half * 2)
				for (int j = 0; j < half; j++) {
					float a							= values[base + start + j];
					float b							= values[base + start + half + j];
					values[base + start + j]		= a + b;
					values[base + start + half + j] = a - b;
				}
}

static void quantize_reference(float *values, int8_t *codes, float *scales, int k, int tile_k) {
	rotate_reference(values, k);
	for (int start = 0, chunk = 0; start < k; start += tile_k, chunk++) {
		int	  count	  = k - start < tile_k ? k - start : tile_k;
		float maximum = 0.0f;
		for (int j = 0; j < count; j++)
			maximum = fmaxf(maximum, fabsf(values[start + j]));
		scales[chunk] = maximum / 127.0f;
		float inverse = maximum ? 127.0f / maximum : 0.0f;
		for (int j = 0; j < count; j++) {
			long code		 = lroundf(values[start + j] * inverse);
			codes[start + j] = (int8_t)(code < -127 ? -127 : code > 127 ? 127 : code);
		}
	}
}

static int32_t dot_reference(const int8_t *a, const int8_t *b, int k) {
#if defined(__ARM_FEATURE_DOTPROD)
	int32x4_t sums = vdupq_n_s32(0);
	for (int j = 0; j < k; j += 16)
		sums = vdotq_s32(sums, vld1q_s8(a + j), vld1q_s8(b + j));
	return vaddvq_s32(sums);
#else
	int32_t sum = 0;
	for (int j = 0; j < k; j++)
		sum += (int32_t)a[j] * b[j];
	return sum;
#endif
}

static void test_int8_matmul(backend *ref, backend *tgt, int n, int k, int m, int zero) {
	int			chunks = (k + 511) / 512;
	int			tile_k = ((k + chunks - 1) / chunks + 31) / 32 * 32;
	size_t		blocks = (size_t)n * (k / 32), elements = (size_t)m * n;
	q8_0_block *weights	 = xmalloc(blocks * sizeof(*weights));
	float	   *input	 = xmalloc((size_t)m * k * sizeof(float));
	float	   *expected = xmalloc(elements * sizeof(float));
	float	   *actual	 = xmalloc(elements * sizeof(float));
	int8_t	   *wc = xmalloc((size_t)n * k), *xc = xmalloc((size_t)m * k);
	float	   *ws	   = xmalloc((size_t)n * chunks * sizeof(float));
	float	   *xs	   = xmalloc((size_t)m * chunks * sizeof(float));
	float	   *values = xmalloc((size_t)k * sizeof(float));
	seed_test_rng(0x5388ULL + (uint64_t)n * 31 + k);
	for (size_t b = 0; b < blocks; b++) {
		weights[b].d = test_f32_to_f16(0.001f * (1 + next_u32() % 500));
		for (int j = 0; j < 32; j++)
			weights[b].qs[j] = (int8_t)((int)(next_u32() >> 24) - 128);
	}
	for (int col = 0; col < n; col++) {
		const q8_0_block *row = weights + (size_t)col * (k / 32);
		for (int j = 0; j < k; j++)
			values[j] = test_f16_to_f32(row[j / 32].d) * row[j / 32].qs[j % 32];
		quantize_reference(values, wc + (size_t)col * k, ws + (size_t)col * chunks, k, tile_k);
	}
	buffer w = buffer_host_view(tgt, weights, blocks * sizeof(*weights));
	for (int reuse = 0; reuse < 2; reuse++) {
		fill_random_f32(input, m * k, 2.0f);
		for (size_t j = 0; j < (size_t)m * k; j++) {
			if (j % 257 == 0)
				input[j] *= 48.0f;
			if (zero && !reuse)
				input[j] = 0.0f;
		}
		for (size_t j = 0; j < elements; j++)
			actual[j] = NAN;
		buffer		x  = buffer_host_view(tgt, input, (size_t)m * k * sizeof(float));
		buffer		y  = buffer_host_view(tgt, actual, elements * sizeof(float));
		status_code st = tgt->matmul_batch(tgt, &w, GGML_TYPE_Q8_0, &x, &y, n, k, m);
		ensure_sync(tgt);
		if (m < 32) {
			buffer		rw = buffer_host_view(ref, weights, w.size);
			buffer		rx = buffer_host_view(ref, input, x.size);
			buffer		ry = buffer_host_view(ref, expected, y.size);
			status_code rs = ref->matmul_batch(ref, &rw, GGML_TYPE_Q8_0, &rx, &ry, n, k, m);
			if (rs != OK)
				st = rs;
		} else {
			for (int row = 0; row < m; row++) {
				memcpy(values, input + (size_t)row * k, (size_t)k * sizeof(float));
				quantize_reference(values, xc + (size_t)row * k, xs + (size_t)row * chunks, k,
								   tile_k);
				for (int col = 0; col < n; col++) {
					float sum = 0.0f;
					for (int b = 0; b < chunks; b++) {
						int		start = b * tile_k, count = k - start < tile_k ? k - start : tile_k;
						int32_t integer = dot_reference(xc + (size_t)row * k + start,
														wc + (size_t)col * k + start, count);
						float	scale = xs[(size_t)row * chunks + b] * ws[(size_t)col * chunks + b];
						sum			  = fmaf((float)integer, scale, sum);
					}
					expected[(size_t)row * n + col] = sum;
				}
			}
		}
		char label[128], detail[256];
		snprintf(label, sizeof(label), "rocket INT8 N=%d K=%d M=%d reuse=%d", n, k, m, reuse);
		verdict v =
			classify_output("exact", expected, actual, (int)elements, st, detail, sizeof(detail));
		if (m >= 32 && v == V_PASS)
			for (size_t j = 0; j < elements; j++)
				if (expected[j] != actual[j]) {
					v = V_FAIL;
					snprintf(detail, sizeof(detail),
							 "integer reference differs at %zu: %.9g vs %.9g", j, expected[j],
							 actual[j]);
					break;
				}
		record_result(OPFAM_BATCH_PARITY, label, v, detail);
	}
	tgt->buffer_free(tgt, &w);
	free(weights);
	free(input);
	free(expected);
	free(actual);
	free(wc);
	free(xc);
	free(ws);
	free(xs);
	free(values);
}

static void test_weight_invalidation(backend *tgt) {
	q8_0_block weights[32];
	for (int b = 0; b < 32; b++) {
		weights[b].d = test_f32_to_f16(0.125f);
		for (int j = 0; j < 32; j++)
			weights[b].qs[j] = (int8_t)((b + j) % 31 - 15);
	}
	float input[33 * 32], expected[33 * 32], actual[33 * 32] = {0};
	seed_test_rng(0x3588);
	fill_random_f32(input, 33 * 32, 1.0f);
	buffer		x	 = buffer_host_view(tgt, input, sizeof(input));
	buffer		y	 = buffer_host_view(tgt, actual, sizeof(actual));
	tensor_desc desc = {
		.host_data = weights, .type = GGML_TYPE_Q8_0, .n_dims = 2, .dims = {32, 32}};
	buffer		w  = {0};
	status_code st = tgt->buffer_alloc_weight(tgt, &desc, &w);
	if (st == OK)
		st = tgt->matmul_batch(tgt, &w, GGML_TYPE_Q8_0, &x, &y, 32, 32, 33);
	memcpy(expected, actual, sizeof(expected));
	float zero[sizeof(weights) / sizeof(float)] = {0};
	for (int copy = 0; copy < 2; copy++) {
		if (st == OK) {
			if (copy) {
				buffer source = buffer_host_view(tgt, weights, sizeof(weights));
				st			  = tgt->copy_buffer(tgt, &source, &w, sizeof(weights) / sizeof(float));
			} else {
				st = tgt->buffer_write_f32(tgt, &w, zero, sizeof(weights) / sizeof(float));
			}
		}
		for (size_t i = 0; i < ARRAY_LEN(actual); i++)
			actual[i] = NAN;
		if (st == OK)
			st = tgt->matmul_batch(tgt, &w, GGML_TYPE_Q8_0, &x, &y, 32, 32, 33);
		char	detail[256];
		float	zeros[33 * 32] = {0};
		verdict v = classify_output("bits", copy ? expected : zeros, actual, 33 * 32, st, detail,
									sizeof(detail));
		record_result(OPFAM_BATCH_PARITY,
					  copy ? "rocket INT8 copy invalidates weights"
						   : "rocket INT8 write invalidates weights",
					  v, detail);
	}
	tgt->buffer_free(tgt, &w);
}

static void test_contiguous_multi(backend *ref, backend *tgt, const char *mode) {
	char label[128];
	snprintf(label, sizeof(label), "rocket %s contiguous multi", mode);
	const int		 n[]	 = {17, 23, 57};
	const uint32_t	 types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};
	const int		 k = 1024, m = 33;
	const qtype_info qt = {
		.name = "Q8_0", .type = GGML_TYPE_Q8_0, .block = 32, .bytes = sizeof(q8_0_block)};
	seed_test_rng(0x358803);
	void *weights = test_make_weight(ref, &qt, 97, k, NULL);
	if (!weights) {
		record_result(OPFAM_BATCH_PARITY, label, V_FAIL, "weight creation failed");
		return;
	}
	tensor_desc desc = {.host_data = weights, .type = GGML_TYPE_Q8_0, .n_dims = 2, .dims = {k, 97}};
	buffer		allocation = {0}, parts[3], expected_buf[3], actual_buf[3];
	const buffer *ws[3];
	buffer		 *ys[3];
	float		 *input	   = xmalloc((size_t)m * k * sizeof(float));
	float		 *expected = xmalloc((size_t)m * 97 * sizeof(float));
	float		 *actual   = xmalloc((size_t)m * 97 * sizeof(float));
	fill_random_f32(input, m * k, 1.0f);
	buffer		x	= buffer_host_view(tgt, input, (size_t)m * k * sizeof(float));
	status_code st	= tgt->buffer_alloc_weight(tgt, &desc, &allocation);
	int			col = 0;
	for (int i = 0; i < 3 && st == OK; i++) {
		parts[i] = buffer_slice(&allocation, (size_t)col * ggml_row_size(GGML_TYPE_Q8_0, k),
								(size_t)n[i] * ggml_row_size(GGML_TYPE_Q8_0, k));
		expected_buf[i] =
			buffer_host_view(tgt, expected + (size_t)m * col, (size_t)m * n[i] * sizeof(float));
		actual_buf[i] =
			buffer_host_view(tgt, actual + (size_t)m * col, (size_t)m * n[i] * sizeof(float));
		ws[i] = &parts[i];
		ys[i] = &actual_buf[i];
		st	  = tgt->matmul_batch(tgt, ws[i], types[i], &x, &expected_buf[i], n[i], k, m);
		col += n[i];
	}
	if (st == OK)
		st = tgt->matmul_multi_batch(tgt, ws, types, &x, ys, n, k, 3, m);
	char	detail[256];
	verdict v = classify_output("bits", expected, actual, m * 97, st, detail, sizeof(detail));
	record_result(OPFAM_BATCH_PARITY, label, v, detail);
	tgt->buffer_free(tgt, &allocation);
	free(weights);
	free(input);
	free(expected);
	free(actual);
}

void run_rocket_tests(void) {
	config	 saved = *config_get();
	backend *ref   = NULL;
	if (backend_create_host(&ref) != OK) {
		record_result(OPFAM_BATCH_PARITY, "rocket reference", V_FAIL, "host initialization failed");
		return;
	}
	const int shapes[][3] = {
		{17, 32, 33},	   {8257, 256, 33}, {17, 8224, 33},	 {384, 4096, 128}, {1024, 4608, 128},
		{9216, 4608, 128}, {24609, 32, 33}, {97, 1024, 129}, {97, 1024, 31},   {17, 32, 33},
	};
	for (int int8 = 0; int8 < 2; int8++) {
		config cfg	 = saved;
		cfg.npu_int8 = int8;
		config_init(&cfg);
		backend	   *tgt = NULL;
		status_code st	= backend_create("rocket", 0, &tgt);
		config_init(&saved);
		if (st != OK) {
			record_result(OPFAM_BATCH_PARITY, "rocket init", V_FAIL,
						  "backend initialization failed");
			continue;
		}
		if (int8) {
			for (size_t i = 0; i < ARRAY_LEN(shapes); i++)
				test_int8_matmul(ref, tgt, shapes[i][0], shapes[i][1], shapes[i][2], i == 0);
			test_weight_invalidation(tgt);
		}
		test_contiguous_multi(ref, tgt, int8 ? "INT8" : "FP16");
		backend_destroy(tgt);
	}
	backend_destroy(ref);
	flush_family(OPFAM_BATCH_PARITY);
}
