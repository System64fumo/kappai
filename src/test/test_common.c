#include "test_core.h"

#include <unistd.h>

static op_stat g_stats[OPFAM_COUNT];
static int	   g_use_color = 1;

const char *verdict_str(verdict v) {
	static const char *names[] = {
		[V_PASS]  = "PASS",
		[V_LOSSY] = "LOSSY",
		[V_FAIL]  = "FAIL",
		[V_SKIP]  = "SKIP",
	};
	if (v >= V_COUNT)
		return "?";
	return names[v];
}

void color_init(void) {
	g_use_color = isatty(STDOUT_FILENO) ? 1 : 0;
}

const char *verdict_color(verdict v) {
	if (!g_use_color)
		return "";
	static const char *colors[] = {
		[V_PASS]  = "\033[32m",
		[V_LOSSY] = "\033[33m",
		[V_FAIL]  = "\033[31m",
		[V_SKIP]  = "\033[90m",
	};
	if (v >= V_COUNT)
		return "";
	return colors[v];
}

const char *color_reset(void) {
	return g_use_color ? "\033[0m" : "";
}

const char *color_dim(void) {
	return g_use_color ? "\033[2m" : "";
}

const char *color_bold(void) {
	return g_use_color ? "\033[1m" : "";
}

const char *op_family_name(op_family f) {
	static const char *names[] = {
		[OPFAM_MATMUL]			 = "matmul",
		[OPFAM_EMBD_LOOKUP]		 = "embd_lookup",
		[OPFAM_RMSNORM]			 = "rmsnorm",
		[OPFAM_RMSNORM_PER_HEAD] = "rmsnorm_per_head",
		[OPFAM_RMSNORM_NOWEIGHT] = "rmsnorm_noweight",
		[OPFAM_ROPE]			 = "rope",
		[OPFAM_ROPE_EXT]		 = "rope_ext",
		[OPFAM_ADD_INPLACE]		 = "add_inplace",
		[OPFAM_FFN_ACTIVATE]	 = "ffn_activate",
		[OPFAM_FFN_ACTIVATE_EX]	 = "ffn_activate_ex",
		[OPFAM_ATTENTION]		 = "attention",
		[OPFAM_ATTENTION_SWA]	 = "attention_swa",
		[OPFAM_KV_PUT]			 = "kv_put",
		[OPFAM_ARGMAX]			 = "argmax",
		[OPFAM_ARCH_LAYER]		 = "arch.single_layer",
		[OPFAM_ARCH_PIPELINE]	 = "arch.pipeline",
		[OPFAM_QUANT]			 = "quant",
		[OPFAM_DEQUANT_PARITY]	 = "dequant_parity",
		[OPFAM_ARCH_DECODE]		 = "arch.decode_chain",
		[OPFAM_ARCH_COMPOUND]	 = "arch.error_compound",
		[OPFAM_ARCH_GENERATE]	 = "arch.generate",
		[OPFAM_MATMUL_RESIDUAL]	 = "matmul_residual",
		[OPFAM_ROPE_QK]			 = "rope_qk",
		[OPFAM_EDGE_CASE]		 = "edge_case",
		[OPFAM_REPACK_PARITY]	 = "repack_parity",
		[OPFAM_KV_QUANT_PARITY]	 = "kv_quant_parity",
	};
	if (f >= OPFAM_COUNT)
		return "?";
	return names[f];
}

static result_entry g_results[OPFAM_COUNT][MAX_RESULTS_PER_FAMILY];
static int			g_results_count[OPFAM_COUNT];

int g_pass = 0, g_lossy = 0, g_fail = 0, g_skip = 0;

static char g_debug_buf[512];

void stats_reset(void) {
	memset(g_stats, 0, sizeof(g_stats));
	memset(g_results_count, 0, sizeof(g_results_count));
	g_pass = g_lossy = g_fail = g_skip = 0;
	g_debug_buf[0]					   = '\0';
}

void compute_debug(const float *y_ref, const float *y_got, int n) {
	g_debug_buf[0] = '\0';
	if (!y_ref || !y_got || n <= 0)
		return;

	float  r_min	= 0;
	float  r_max	= 0;
	float  r_maxabs = 0;
	double r_sum	= 0;
	int	   r_nf		= 0;
	int	   r_valid	= 0;
	for (int i = 0; i < n; i++) {
		float v = y_ref[i];
		if (!isfinite(v)) {
			r_nf++;
			continue;
		}
		if (!r_valid) {
			r_min = r_max = v;
			r_maxabs	  = fabsf(v);
			r_valid		  = 1;
		} else {
			if (v < r_min)
				r_min = v;
			if (v > r_max)
				r_max = v;
			if (fabsf(v) > r_maxabs)
				r_maxabs = fabsf(v);
		}
		r_sum += (double)v;
	}
	float r_mean = r_valid ? (float)(r_sum / n) : 0.0f;

	float  g_min	= 0;
	float  g_max	= 0;
	float  g_maxabs = 0;
	double g_sum	= 0;
	int	   g_nf		= 0;
	int	   g_valid	= 0;
	for (int i = 0; i < n; i++) {
		float v = y_got[i];
		if (!isfinite(v)) {
			g_nf++;
			continue;
		}
		if (!g_valid) {
			g_min = g_max = v;
			g_maxabs	  = fabsf(v);
			g_valid		  = 1;
		} else {
			if (v < g_min)
				g_min = v;
			if (v > g_max)
				g_max = v;
			if (fabsf(v) > g_maxabs)
				g_maxabs = fabsf(v);
		}
		g_sum += (double)v;
	}
	float g_mean = g_valid ? (float)(g_sum / n) : 0.0f;

	int	   n_valid_diff = 0;
	float *diffs		= xmalloc((size_t)n * sizeof(float));
	for (int i = 0; i < n; i++) {
		if (!isfinite(y_ref[i]) || !isfinite(y_got[i]))
			continue;
		diffs[n_valid_diff++] = fabsf(y_ref[i] - y_got[i]);
	}
	for (int i = 1; i < n_valid_diff; i++) {
		float key = diffs[i];
		int	  j	  = i - 1;
		while (j >= 0 && diffs[j] > key) {
			diffs[j + 1] = diffs[j];
			j--;
		}
		diffs[j + 1] = key;
	}
	float p50  = n_valid_diff > 0 ? diffs[n_valid_diff / 2] : 0;
	float p90  = n_valid_diff > 0 ? diffs[(n_valid_diff * 90) / 100] : 0;
	float p99  = n_valid_diff > 0 ? diffs[(n_valid_diff * 99) / 100] : 0;
	float pmax = n_valid_diff > 0 ? diffs[n_valid_diff - 1] : 0;
	free(diffs);

	const int dbg_cap = (int)sizeof(g_debug_buf);

	int off = snprintf(g_debug_buf, sizeof(g_debug_buf),
					   "CPU:  min=%+.4e  max=%+.4e  mean=%+.4e  |max|=%.4e  nf=%d/%d\n"
					   "TGT:  min=%+.4e  max=%+.4e  mean=%+.4e  |max|=%.4e  nf=%d/%d\n"
					   "err:  p50=%.3e  p90=%.3e  p99=%.3e  max=%.3e  (%d valid)",
					   r_min, r_max, r_mean, r_maxabs, r_nf, n, g_min, g_max, g_mean, g_maxabs,
					   g_nf, n, p50, p90, p99, pmax, n_valid_diff);
	if (off < 0)
		off = 0;
	if (off > dbg_cap)
		off = dbg_cap;

	if (g_nf > 0) {
		for (int i = 0; i < n; i++) {
			if (!isfinite(y_got[i])) {
				int rem = dbg_cap - off;
				if (rem > 0)
					off += snprintf(g_debug_buf + off, (size_t)rem,
									"\nfirst NaN/Inf: TGT[%d]=%f (CPU=%f)", i, y_got[i],
									i < n ? y_ref[i] : 0.0f);
				if (off > dbg_cap)
					off = dbg_cap;
				break;
			}
		}
	}

	int shown = 0;
	for (int i = 0; i < n && shown < 3; i++) {
		if (!isfinite(y_got[i]) || !isfinite(y_ref[i]))
			continue;
		float diff	= fabsf(y_ref[i] - y_got[i]);
		float scale = fmaxf(fabsf(y_ref[i]), fabsf(y_got[i]));
		if (scale > 1e-8f && diff / scale > 0.01f) {
			int rem = dbg_cap - off;
			if (rem > 0)
				off += snprintf(g_debug_buf + off, (size_t)rem, "%s[%d] CPU=%+.6f TGT=%+.6f",
								shown == 0 ? "\nmismatch: " : ", ", i, y_ref[i], y_got[i]);
			if (off > dbg_cap)
				off = dbg_cap;
			shown++;
		}
	}

	if (n <= 16 && n > 0) {
		int rem = dbg_cap - off;
		if (rem > 0)
			off += snprintf(g_debug_buf + off, (size_t)rem, "\nelement dump:");
		if (off > dbg_cap)
			off = dbg_cap;
		for (int i = 0; i < n; i++) {
			int rem2 = dbg_cap - off;
			if (rem2 <= 0)
				break;
			off += snprintf(g_debug_buf + off, (size_t)rem2,
							"\n  [%2d] CPU=%+.6e  TGT=%+.6e  diff=%.3e", i, y_ref[i], y_got[i],
							isfinite(y_ref[i]) && isfinite(y_got[i]) ? fabsf(y_ref[i] - y_got[i])
																	 : (float)INFINITY);
			if (off > dbg_cap)
				off = dbg_cap;
		}
	}
}

void record_result(op_family fam, const char *label, verdict v, const char *detail) {
	int idx = g_results_count[fam];
	if (idx < MAX_RESULTS_PER_FAMILY) {
		result_entry *e = &g_results[fam][idx];
		snprintf(e->label, sizeof(e->label), "%s", label);
		e->v = v;
		snprintf(e->detail, sizeof(e->detail), "%s", detail ? detail : "");
		snprintf(e->debug, sizeof(e->debug), "%s", g_debug_buf);
		g_results_count[fam]++;
		g_debug_buf[0] = '\0';
	}

	g_stats[fam].calls++;
	g_stats[fam].cnt[v]++;
	switch (v) {
	case V_PASS:
		g_pass++;
		break;
	case V_LOSSY:
		g_lossy++;
		break;
	case V_FAIL:
		g_fail++;
		break;
	case V_SKIP:
		g_skip++;
		break;
	default:
		break;
	}
}

void flush_family(op_family fam) {
	int n = g_results_count[fam];
	if (n == 0)
		return;

	int pass  = g_stats[fam].cnt[V_PASS];
	int lossy = g_stats[fam].cnt[V_LOSSY];
	int fail  = g_stats[fam].cnt[V_FAIL];
	int skip  = g_stats[fam].cnt[V_SKIP];

	printf("\n%s:\n", op_family_name(fam));

	for (int i = 0; i < n; i++) {
		result_entry *e = &g_results[fam][i];
		if (e->v == V_PASS)
			continue;
		printf("  %s[%-5s]%s %-44s\n", verdict_color(e->v), verdict_str(e->v), color_reset(),
			   e->label);
		if (e->detail[0])
			printf("  %s       %s%s\n", color_dim(), e->detail, color_reset());
		if (e->debug[0]) {
			char *line = e->debug;
			char *nl;
			while ((nl = strchr(line, '\n')) != NULL) {
				*nl = '\0';
				printf("  %s       %s%s\n", color_dim(), line, color_reset());
				*nl	 = '\n';
				line = nl + 1;
			}
			if (*line)
				printf("  %s       %s%s\n", color_dim(), line, color_reset());
		}
	}
	printf("  %s-- %d tests:%s ", color_bold(), n, color_reset());
	if (fail == 0 && lossy == 0 && skip == 0) {
		printf("%sall PASS%s", verdict_color(V_PASS), color_reset());
	} else {
		printf("%s%d PASS%s", verdict_color(V_PASS), pass, color_reset());
		if (lossy)
			printf(", %s%d LOSSY%s", verdict_color(V_LOSSY), lossy, color_reset());
		if (fail)
			printf(", %s%d FAIL%s", verdict_color(V_FAIL), fail, color_reset());
		if (skip)
			printf(", %s%d SKIP%s", verdict_color(V_SKIP), skip, color_reset());
	}
	printf("%s\n", color_reset());

	g_results_count[fam] = 0;
}

float max_abs_diff_at(const float *a, const float *b, int n, int *at) {
	float mx = 0.0f;
	int	  mi = -1;
	for (int i = 0; i < n; i++) {
		float d = fabsf(a[i] - b[i]);
		if (d > mx) {
			mx = d;
			mi = i;
		}
	}
	if (at)
		*at = mi;
	return mx;
}

float max_abs_val(const float *a, int n) {
	float mx = 0.0f;
	for (int i = 0; i < n; i++) {
		float v = fabsf(a[i]);
		if (v > mx)
			mx = v;
	}
	return mx;
}

int count_nonfinite(const float *a, int n) {
	int c = 0;
	for (int i = 0; i < n; i++)
		if (!isfinite(a[i]))
			c++;
	return c;
}

float max_combined_ratio_at(const float *a, const float *b, int n, float atol, float rtol,
							int *at) {
	float mx = 0.0f;
	int	  mi = -1;
	for (int i = 0; i < n; i++) {
		float diff	= fabsf(a[i] - b[i]);
		float scale = fmaxf(fabsf(a[i]), fabsf(b[i]));
		float denom = atol + (rtol * scale);
		float ratio = denom > 0.0f ? diff / denom : (diff > 0.0f ? INFINITY : 0.0f);
		if (ratio > mx) {
			mx = ratio;
			mi = i;
		}
	}
	if (at)
		*at = mi;
	return mx;
}

verdict classify_output(const char *tol_kind, const float *y_ref, const float *y_got, int n,
						status_code tgt_status, char *detail, size_t detail_sz) {
	if (tgt_status == ERR_UNSUPPORTED) {
		snprintf(detail, detail_sz, "missing native implementation");
		return V_SKIP;
	}
	if (tgt_status != OK) {
		snprintf(detail, detail_sz, "op status=%d (returned error)", tgt_status);
		return V_FAIL;
	}

	int nf = count_nonfinite(y_got, n);
	if (nf > 0) {
		snprintf(detail, detail_sz, "%d/%d non-finite (NaN/Inf) values in output", nf, n);
		return V_FAIL;
	}

	float ref_mag = max_abs_val(y_ref, n);
	float got_mag = max_abs_val(y_got, n);
	if (got_mag < NULL_MAGNITUDE && ref_mag > REF_MAGNITUDE_MIN) {
		snprintf(detail, detail_sz, "output is all-zeros/null (got_mag=%g, ref_mag=%g)", got_mag,
				 ref_mag);
		return V_FAIL;
	}

	if (tol_kind && strcmp(tol_kind, "bits") == 0) {
		int ndiff = 0;
		for (int i = 0; i < n; i++) {
			uint32_t a, b;
			memcpy(&a, &y_ref[i], sizeof(a));
			memcpy(&b, &y_got[i], sizeof(b));
			if (a != b)
				ndiff++;
		}
		if (ndiff == 0) {
			snprintf(detail, detail_sz, "bit-identical (%d values)", n);
			return V_PASS;
		}
		snprintf(detail, detail_sz, "%d/%d values differ in low bits (not bit-identical)", ndiff,
				 n);
		return V_FAIL;
	}

	float atol;
	float rtol;
	int	  is_exact_band = (tol_kind && tol_kind[0] == 'e');
	int	  is_kv_quant	= (tol_kind && strcmp(tol_kind, "kv_quant") == 0);
	if (is_exact_band) {
		atol = EPS_EXACT;
		rtol = 0.0f;
	} else if (is_kv_quant) {
		atol = ATOL_KV_QUANT;
		rtol = RTOL_KV_QUANT;
	} else {
		atol = ATOL_LOOSE;
		rtol = RTOL_LOOSE;
	}

	int	  at	  = -1;
	float ratio	  = max_combined_ratio_at(y_ref, y_got, n, atol, rtol, &at);
	float abs_err = max_abs_diff_at(y_ref, y_got, n, NULL);

	if (is_kv_quant) {
		if (ratio <= 1.0f) {
			snprintf(detail, detail_sz,
					 "max_abs=%.3e@%d ref=%+.6f got=%+.6f tol_ratio=%.3f "
					 "(within quantized-cache band, expected quantization loss)",
					 abs_err, at, at >= 0 ? y_ref[at] : 0.0f, at >= 0 ? y_got[at] : 0.0f, ratio);
			return V_PASS;
		}
		snprintf(detail, detail_sz,
				 "max_abs=%.3e@%d ref=%+.6f got=%+.6f tol_ratio=%.3f "
				 "(exceeds quantized-cache lossy band, unexpectedly lossy)",
				 abs_err, at, at >= 0 ? y_ref[at] : 0.0f, at >= 0 ? y_got[at] : 0.0f, ratio);
		return V_LOSSY;
	}

	float exact_atol;
	float exact_rtol;
	if (is_exact_band) {
		exact_atol = EPS_EXACT;
		exact_rtol = 0.0f;
	} else {
		exact_atol = ATOL_EXACT;
		exact_rtol = RTOL_EXACT;
	}
	float ratio_exact = max_combined_ratio_at(y_ref, y_got, n, exact_atol, exact_rtol, NULL);
	if (ratio_exact <= 1.0f) {
		snprintf(detail, detail_sz, "max_abs=%.3e@%d ref=%+.6f got=%+.6f (exact match)", abs_err,
				 at, at >= 0 ? y_ref[at] : 0.0f, at >= 0 ? y_got[at] : 0.0f);
		return V_PASS;
	}

	if (ratio <= 1.0f) {
		snprintf(detail, detail_sz,
				 "max_abs=%.3e@%d ref=%+.6f got=%+.6f tol_ratio=%.3f "
				 "(within loose band, acceptable hw loss)",
				 abs_err, at, at >= 0 ? y_ref[at] : 0.0f, at >= 0 ? y_got[at] : 0.0f, ratio);
		return V_LOSSY;
	}

	snprintf(detail, detail_sz,
			 "max_abs=%.3e@%d ref=%+.6f got=%+.6f tol_ratio=%.3f "
			 "(exceeds loose band, too lossy)",
			 abs_err, at, at >= 0 ? y_ref[at] : 0.0f, at >= 0 ? y_got[at] : 0.0f, ratio);
	return V_FAIL;
}

#define TIME_OP(call_expr, sync_b, out_us)                                                         \
	do {                                                                                           \
		uint64_t	_t0 = time_us();                                                               \
		status_code _s	= (call_expr);                                                             \
		if ((sync_b) && (sync_b)->synchronize)                                                     \
			(sync_b)->synchronize(sync_b);                                                         \
		*(out_us) = time_us() - _t0;                                                               \
		(void)_s;                                                                                  \
	} while (0)

static uint64_t g_seed;
uint32_t		next_u32(void) {
	g_seed ^= g_seed << 13;
	g_seed ^= g_seed >> 7;
	g_seed ^= g_seed << 17;
	return (uint32_t)(g_seed >> 16);
}
void seed_test_rng(uint64_t s) {
	g_seed = s ? s : 0x9E3779B97F4A7C15ULL;
}

void fill_random_blocks(void *blocks, int n_blocks, size_t block_bytes, uint32_t type) {
	uint8_t *bp = blocks;
	for (int i = 0; i < n_blocks; i++) {
		for (size_t j = 0; j < block_bytes; j++)
			bp[j] = (uint8_t)(next_u32() & 0xFF);

		float	 d_val = 0.0005f + (0.02f * ((next_u32() % 997) / 997.0f));
		uint16_t d16   = f32_to_f16(d_val);
		if (type == GGML_TYPE_Q6_K) {
			memcpy(bp + 208, &d16, 2);
		} else {
			memcpy(bp, &d16, 2);
			if (type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1) {
				float	 m_val = -0.01f + (0.02f * ((next_u32() % 997) / 997.0f));
				uint16_t m16   = f32_to_f16(m_val);
				memcpy(bp + 2, &m16, 2);
			} else if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
				float	 dmin_val = 0.0002f + (0.005f * ((next_u32() % 997) / 997.0f));
				uint16_t dmin16	  = f32_to_f16(dmin_val);
				memcpy(bp + 2, &dmin16, 2);
			}
		}
		bp += block_bytes;
	}
}

void fill_random_f32(float *x, int n, float scale) {
	for (int i = 0; i < n; i++) {
		int32_t r = (int32_t)(next_u32() % 2001) - 1000;
		x[i]	  = (float)r * (scale / 1000.0f);
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

typedef void (*test_repack_fn)(const void *src, void *dst, int n_rows, int k);

static test_repack_fn test_repack_for_type(uint32_t type, uint32_t *base_type_out) {
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

int test_type_per_row(uint32_t type) {
	switch (type) {
	case GGML_TYPE_Q4_0_R8:
	case GGML_TYPE_Q8_0_R8:
	case GGML_TYPE_IQ4_NL_R8:
	case GGML_TYPE_IQ3_S_RE8:
		return 0;
	default:
		return 1;
	}
}

void *test_make_weight(const qtype_info *qt, int n_rows, int k, size_t *out_bytes) {
	int bpr = k / qt->block;

	uint32_t	   base_type;
	test_repack_fn repack = test_repack_for_type(qt->type, &base_type);
	if (repack) {
		int	   n_pad	  = (n_rows + 7) & ~7;
		size_t base_bytes = ggml_row_size(base_type, (size_t)k) / (size_t)bpr;
		int	   n_base	  = n_pad * bpr;
		void  *base		  = xcalloc((size_t)n_base, base_bytes);
		fill_random_blocks(base, n_base, base_bytes, base_type);

		size_t bytes = (size_t)n_pad * ggml_row_size(qt->type, (size_t)k);
		void  *buf	 = xmalloc(bytes);
		repack(base, buf, n_pad, k);
		free(base);
		if (out_bytes)
			*out_bytes = bytes;
		return buf;
	}

	if (qt->type == GGML_TYPE_F32) {
		size_t bytes = (size_t)n_rows * (size_t)k * sizeof(float);
		float *buf	 = xmalloc(bytes);
		fill_random_f32(buf, n_rows * k, 1.0f);
		if (out_bytes)
			*out_bytes = bytes;
		return buf;
	}
	if (qt->type == GGML_TYPE_F16) {
		size_t	  bytes = (size_t)n_rows * (size_t)k * sizeof(uint16_t);
		uint16_t *buf	= xmalloc(bytes);
		fill_random_f16(buf, n_rows * k);
		if (out_bytes)
			*out_bytes = bytes;
		return buf;
	}
	if (qt->type == GGML_TYPE_BF16) {
		size_t	  bytes = (size_t)n_rows * (size_t)k * sizeof(uint16_t);
		uint16_t *buf	= xmalloc(bytes);
		fill_random_bf16(buf, n_rows * k);
		if (out_bytes)
			*out_bytes = bytes;
		return buf;
	}

	int	   n_blocks = n_rows * bpr;
	size_t bytes	= (size_t)n_blocks * qt->bytes;
	void  *buf		= xcalloc((size_t)n_blocks, qt->bytes);
	fill_random_blocks(buf, n_blocks, qt->bytes, qt->type);
	if (out_bytes)
		*out_bytes = bytes;
	return buf;
}

const qtype_info QTYPES[] = {
	{"q4_0", GGML_TYPE_Q4_0, 32, sizeof(q4_0_block)},
	{"q4_1", GGML_TYPE_Q4_1, 32, sizeof(q4_1_block)},
	{"q5_0", GGML_TYPE_Q5_0, 32, sizeof(q5_0_block)},
	{"q5_1", GGML_TYPE_Q5_1, 32, sizeof(q5_1_block)},
	{"q8_0", GGML_TYPE_Q8_0, 32, sizeof(q8_0_block)},
	{"q4_K", GGML_TYPE_Q4_K, 256, sizeof(q4_k_block)},
	{"q5_K", GGML_TYPE_Q5_K, 256, sizeof(q5_k_block)},
	{"q6_K", GGML_TYPE_Q6_K, 256, sizeof(q6_k_block)},
	{"iq4_nl", GGML_TYPE_IQ4_NL, 32, sizeof(iq4_nl_block)},
	{"iq3_s", GGML_TYPE_IQ3_S, 256, sizeof(iq3_s_block)},
	{"q4_0_r8", GGML_TYPE_Q4_0_R8, 32, sizeof(q4_0_block)},
	{"q8_0_r8", GGML_TYPE_Q8_0_R8, 32, sizeof(q8_0_block)},
	{"iq4_nl_r8", GGML_TYPE_IQ4_NL_R8, 32, sizeof(iq4_nl_block)},
	{"iq3_s_re", GGML_TYPE_IQ3_S_RE, 256, sizeof(iq3_s_block)},
	{"iq3_s_re8", GGML_TYPE_IQ3_S_RE8, 256, sizeof(iq3_s_block)},
	{"f16", GGML_TYPE_F16, 1, sizeof(uint16_t)},
	{"bf16", GGML_TYPE_BF16, 1, sizeof(uint16_t)},
	{"f32", GGML_TYPE_F32, 1, sizeof(float)},
};
const int QTYPES_N = (int)(sizeof(QTYPES) / sizeof(QTYPES[0]));
#define QTYPES_N ((int)(sizeof(QTYPES) / sizeof(QTYPES[0])))

void print_summary_table(void) {
	int total_pass	= 0;
	int total_lossy = 0;
	int total_fail	= 0;
	int total_skip	= 0;
	for (int i = 0; i < OPFAM_COUNT; i++) {
		total_pass += g_stats[i].cnt[V_PASS];
		total_lossy += g_stats[i].cnt[V_LOSSY];
		total_fail += g_stats[i].cnt[V_FAIL];
		total_skip += g_stats[i].cnt[V_SKIP];
	}

	printf("\n========================================\n");
	printf("%sPer-op summary%s\n", color_bold(), color_reset());
	printf("========================================\n");
	printf("  %-20s  %4s %5s %4s %4s  %6s\n", "op", "pass", "lossy", "fail", "skip", "rate");
	printf("  %-20s  %4s %5s %4s %4s  %6s\n", "----", "----", "-----", "----", "----", "----");
	for (int i = 0; i < OPFAM_COUNT; i++) {
		if (g_stats[i].calls == 0)
			continue;
		int	   pass	 = g_stats[i].cnt[V_PASS];
		int	   lossy = g_stats[i].cnt[V_LOSSY];
		int	   fail	 = g_stats[i].cnt[V_FAIL];
		int	   skip	 = g_stats[i].cnt[V_SKIP];
		int	   total = g_stats[i].calls;
		double rate	 = total > 0 ? 100.0 * pass / total : 0.0;

		const char *op_color;
		if (fail > 0)
			op_color = verdict_color(V_FAIL);
		else if (lossy > 0)
			op_color = verdict_color(V_LOSSY);
		else if (skip > 0)
			op_color = verdict_color(V_SKIP);
		else
			op_color = verdict_color(V_PASS);

		printf("  %s%-20s%s  %s%4d%s %s%5d%s %s%4d%s %s%4d%s  %5.1f%%\n", op_color,
			   op_family_name((op_family)i), color_reset(), pass ? verdict_color(V_PASS) : "", pass,
			   color_reset(), lossy ? verdict_color(V_LOSSY) : "", lossy, color_reset(),
			   fail ? verdict_color(V_FAIL) : "", fail, color_reset(),
			   skip ? verdict_color(V_SKIP) : "", skip, color_reset(), rate);
	}

	int	   total_tests = total_pass + total_lossy + total_fail + total_skip;
	double total_rate  = total_tests > 0 ? 100.0 * total_pass / total_tests : 0.0;
	printf("  %-20s  %s%4d%s %s%5d%s %s%4d%s %s%4d%s  %5.1f%%\n", "TOTAL", verdict_color(V_PASS),
		   total_pass, color_reset(), verdict_color(V_LOSSY), total_lossy, color_reset(),
		   verdict_color(V_FAIL), total_fail, color_reset(), verdict_color(V_SKIP), total_skip,
		   color_reset(), total_rate);
}

void print_final_results(void) {
	print_summary_table();
}

int matches_name(int argc, char **argv, const char *name) {
	for (int ai = 1; ai < argc; ai++)
		if (argv[ai][0] != '-' && strcmp(argv[ai], name) == 0)
			return 1;
	return 0;
}

int wants_all(int argc, char **argv) {
	for (int ai = 1; ai < argc; ai++)
		if (strcmp(argv[ai], "--all") == 0)
			return 1;
	return 0;
}

void usage(const char *prog) {
	fprintf(stderr,
			"Usage:\n"
			"  %s [--all | <backend>...]         per-op + combined-op validation vs CPU\n"
			"  %s --bench [--all | <backend>]   per-quant matmul GFLOPS per backend\n"
			"  %s --model <path> [--all | <b>...]  real-model greedy-decode cross-check\n"
			"\n"
			"Modes:\n"
			"  (default)    per-op correctness (each op vs CPU) plus combined-op tests\n"
			"               (single layer, multi-layer prefill, decode chains, and a full\n"
			"               prompt-processing + %d-token generation sweep across "
			"architectures) --\n"
			"               catches compounding errors across op chains; CPU errors or\n"
			"               NaN/Inf at any step are always reported as a failure\n"
			"  --bench      per-quant matmul GFLOPS, every M row count, each backend\n"
			"  --model      load a real GGUF model and cross-validate greedy decode\n"
			"\n"
			"Options:\n"
			"  --all              run against every available backend\n"
			"  --n-prefill N      override prefill token count (model mode)\n"
			"  --n-decode  N      override decode token count (model mode)\n"
			"  -h, --help         this message\n"
			"\n"
			"Available backends: ",
			prog, prog, prog, ARCH_GENERATE_N_DECODE);
	backend_info infos[BACKEND_MAX];
	int			 n = backend_list(infos, BACKEND_MAX);
	for (int i = 0; i < n; i++)
		fprintf(stderr, "%s%s%s", i ? ", " : "", infos[i].name,
				infos[i].available ? "" : "(unavailable)");
	fprintf(stderr, "\n");
}