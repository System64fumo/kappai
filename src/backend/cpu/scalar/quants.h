#ifndef QUANTS_H
#define QUANTS_H

#include "common.h"
#include "gguf.h"
#include <math.h>

#define Q8_0_R8_ROWS 8
#define Q4_0_R8_ROWS 8
#define IQ4_NL_R8_ROWS 8
#define IQ3_S_RE_BLOCK_BYTES 134
#define IQ3_S_RE8_ROWS 8
#define IQ3_S_RE8_GROUP_BYTES                                                                      \
	(IQ3_S_RE8_ROWS * sizeof(uint16_t) + IQ3_S_RE8_ROWS * 4 + IQ3_S_RE8_ROWS * 128)

typedef struct {
	uint16_t d;
	uint8_t	 qs[16];
} q4_0_block;
_Static_assert(sizeof(q4_0_block) == 18, "");

typedef struct {
	uint16_t d;
	uint16_t m;
	uint8_t	 qs[16];
} q4_1_block;
_Static_assert(sizeof(q4_1_block) == 20, "");

typedef struct {
	uint16_t d;
	uint8_t	 qh[4];
	uint8_t	 qs[16];
} q5_0_block;
_Static_assert(sizeof(q5_0_block) == 22, "");

typedef struct {
	uint16_t d;
	uint16_t m;
	uint8_t	 qh[4];
	uint8_t	 qs[16];
} q5_1_block;
_Static_assert(sizeof(q5_1_block) == 24, "");

typedef struct {
	uint16_t d;
	uint8_t	 qs[16];
} iq4_nl_block;
_Static_assert(sizeof(iq4_nl_block) == 18, "");

typedef struct {
	uint16_t d;
	uint16_t scales_h;
	uint8_t	 scales_l[4];
	uint8_t	 qs[128];
} iq4_xs_block;
_Static_assert(sizeof(iq4_xs_block) == 136, "");

typedef struct {
	uint16_t d;
	uint16_t qs[32];
} iq2_xxs_block;
_Static_assert(sizeof(iq2_xxs_block) == 66, "");

typedef struct {
	uint16_t d;
	uint16_t qs[32];
	uint8_t	 scales[8];
} iq2_xs_block;
_Static_assert(sizeof(iq2_xs_block) == 74, "");

typedef struct {
	uint16_t d;
	uint8_t	 qs[64];
	uint8_t	 qh[8];
	uint8_t	 scales[8];
} iq2_s_block;
_Static_assert(sizeof(iq2_s_block) == 82, "");

typedef struct {
	uint16_t d;
	uint8_t	 qs[96];
} iq3_xxs_block;
_Static_assert(sizeof(iq3_xxs_block) == 98, "");

typedef struct {
	uint16_t d;
	uint8_t	 qs[32];
	uint16_t qh[8];
} iq1_s_block;
_Static_assert(sizeof(iq1_s_block) == 50, "");

typedef struct {
	uint8_t qs[32];
	uint8_t qh[16];
	uint8_t scales[8];
} iq1_m_block;
_Static_assert(sizeof(iq1_m_block) == 56, "");

typedef struct {
	uint8_t	 scales[16];
	uint8_t	 qs[64];
	uint16_t d;
	uint16_t dmin;
} q2_k_block;
_Static_assert(sizeof(q2_k_block) == 84, "");

typedef struct {
	uint16_t d;
	uint8_t	 qs[64];
	uint8_t	 qh[8];
	uint8_t	 signs[32];
	uint8_t	 scales[4];
} iq3_s_block;
_Static_assert(sizeof(iq3_s_block) == 110, "wrong iq3s block size");

typedef struct {
	uint8_t	 hmask[32];
	uint8_t	 qs[64];
	uint8_t	 scales[12];
	uint16_t d;
} q3_k_block;
_Static_assert(sizeof(q3_k_block) == 110, "");

typedef struct {
	uint16_t d;
	uint16_t dmin;
	uint8_t	 scales[12];
	uint8_t	 qs[128];
} q4_k_block;
_Static_assert(sizeof(q4_k_block) == 144, "");

typedef struct {
	uint16_t d;
	uint16_t dmin;
	uint8_t	 scales[12];
	uint8_t	 qh[32];
	uint8_t	 qs[128];
} q5_k_block;
_Static_assert(sizeof(q5_k_block) == 176, "");

typedef struct {
	uint8_t	 ql[128];
	uint8_t	 qh[64];
	int8_t	 scales[16];
	uint16_t d;
} q6_k_block;
_Static_assert(sizeof(q6_k_block) == 210, "");

typedef struct {
	uint16_t d;
	int8_t	 qs[32];
} q8_0_block;
_Static_assert(sizeof(q8_0_block) == 34, "");

typedef struct {
	uint16_t d;
	uint16_t s;
	int8_t	 qs[32];
} q8_1_block;
_Static_assert(sizeof(q8_1_block) == 36, "");

typedef struct {
	float	d;
	int8_t	qs[256];
	int16_t bsums[16];
} q8_k_block;
_Static_assert(sizeof(q8_k_block) == 292, "");

typedef struct {
	void  *q8_buf;
	size_t q8_buf_elems;
} quant_scratch;

extern const uint32_t iq3s_grid[512];
extern const uint8_t  ksigns_iq2xs[128];
extern const uint64_t iq2xxs_grid[256];
extern const uint64_t iq2xs_grid[512];
extern const uint64_t iq2s_grid[1024];
extern const uint32_t iq3xxs_grid[256];
extern const uint64_t iq1s_grid[2048];

float	 f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);

void dequant_q4_0_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_q4_1_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_q5_0_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_q5_1_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_q8_0_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_q2_k_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_q3_k_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_q4_k_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_q5_k_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_q6_k_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_iq4_nl_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_iq4_xs_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_iq2_xxs_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_iq2_xs_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_iq2_s_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_iq3_xxs_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_iq1_s_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_iq1_m_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_iq3_s_row(const void *blocks, size_t n_blocks, float *dst);
void dequant_f16_row(const void *src, int n, float *dst);
void dequant_bf16_row(const void *src, int n, float *dst);

void dequant_row_dispatch(uint32_t type, const void *src, int n_elems, float *dst);

float bf16_to_f32(uint16_t h);

static inline uint32_t wtype_to_q8type(uint32_t w_type) {
	switch (w_type) {
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_Q8_0:
		return GGML_TYPE_Q8_0;
	case GGML_TYPE_Q4_1:
		return GGML_TYPE_Q8_1;
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ3_S_RE:
	case GGML_TYPE_IQ4_XS:
	case GGML_TYPE_Q2_K:
	case GGML_TYPE_IQ2_XXS:
	case GGML_TYPE_IQ2_XS:
	case GGML_TYPE_IQ2_S:
	case GGML_TYPE_IQ3_XXS:
	case GGML_TYPE_IQ1_S:
	case GGML_TYPE_IQ1_M:
		return GGML_TYPE_Q8_K;
	default:
		return 0;
	}
}

void matmul_generic_f32(const void *w, uint32_t w_type, const float *x, float *y, int n, int k);

void quantize_q8_0(const float *x, q8_0_block *dst, int n);
void quantize_q8_1(const float *x, void *dst, int n);
void quantize_q8_k(const float *x, q8_k_block *y, int n);
void quant_scratch_ensure(quant_scratch *qs, size_t need);

void matmul_q4_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
					  quant_scratch *qs);
void matmul_q8_0_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						quant_scratch *qs);
void matmul_q4_1_q8_f32(const void *w, const float *x, float *y, int n, int k, quant_scratch *qs);
void matmul_iq4_nl_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						  quant_scratch *qs);
void matmul_iq4_xs_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
							quant_scratch *qs);
void matmul_q2_k_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						  quant_scratch *qs);
void matmul_iq2_xxs_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
							 quant_scratch *qs);
void matmul_iq2_xs_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
							quant_scratch *qs);
void matmul_iq2_s_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						   quant_scratch *qs);
void matmul_iq3_xxs_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
							 quant_scratch *qs);
void matmul_iq1_s_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						   quant_scratch *qs);
void matmul_iq1_m_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						   quant_scratch *qs);
void matmul_q6_k_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						quant_scratch *qs);
void matmul_q4_k_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						  quant_scratch *qs);
void matmul_q5_k_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						  quant_scratch *qs);
void matmul_q8_0_r8_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						   quant_scratch *qs);
void matmul_q4_0_r8_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						   quant_scratch *qs);
void matmul_iq3_s_re8_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n,
							   int k, quant_scratch *qs);

void repack_iq4_nl_to_q8_0(const void *src, void *dst, int n_rows, int k);
void repack_iq4_nl_to_q8_0_rows(const void *src, void *dst, int row_begin, int row_end, int k);

void repack_iq4_nl_to_iq4_nl_r8(const void *src, void *dst, int n_rows, int k);
void repack_iq4_nl_to_iq4_nl_r8_rows(const void *src, void *dst, int row_begin, int row_end, int k);
void matmul_iq4_nl_r8_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								   size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								   int n, int k, int m);
void matmul_iq4_nl_r8_q8_f32(const void *w, const float *restrict x, float *restrict y, int n,
							 int k, quant_scratch *qs);

void repack_q8_0_to_q8_0_r8(const void *src, void *dst, int n_rows, int k);
void repack_q8_0_to_q8_0_r8_rows(const void *src, void *dst, int row_begin, int row_end, int k);
void matmul_q8_0_r8_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m);

void repack_q4_0_to_q4_0_r8(const void *src, void *dst, int n_rows, int k);
void repack_q4_0_to_q4_0_r8_rows(const void *src, void *dst, int row_begin, int row_end, int k);
void matmul_q4_0_r8_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m);

void repack_iq3_s(const void *src, void *dst, int n_rows, int k);
void repack_iq3_s_rows(const void *src, void *dst, int row_begin, int row_end, int k);

void repack_iq3_s_to_iq3_s_re8(const void *src, void *dst, int n_rows, int k);
void repack_iq3_s_to_iq3_s_re8_rows(const void *src, void *dst, int row_begin, int row_end, int k);

void matmul_iq3_s_re8_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
									 size_t		 xq_row_stride_blocks, float *restrict y,
									 int y_row_stride, int n, int k, int m);

void matmul_iq3_s_re_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n,
							  int k, quant_scratch *qs);
void matmul_iq3_s_re_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
									size_t		xq_row_stride_blocks, float *restrict y,
									int y_row_stride, int n, int k, int m);
void dequant_iq3_s_re_row(const void *repacked_buf, size_t n_blocks, float *dst);

void matmul_iq3_s_q8_k_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						   quant_scratch *qs);

void matmul_q4_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							size_t xq_row_stride_blocks, float *restrict y, int y_row_stride, int n,
							int k, int m);
void matmul_q8_0_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m);
void matmul_iq4_nl_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m);
void matmul_iq4_xs_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								  int n, int k, int m);
void matmul_q2_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m);
void matmul_iq2_xxs_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								   size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								   int n, int k, int m);
void matmul_iq2_xs_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								  int n, int k, int m);
void matmul_iq2_s_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m);
void matmul_iq3_xxs_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								   size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								   int n, int k, int m);
void matmul_iq1_s_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m);
void matmul_iq1_m_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m);
void matmul_q4_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m);
void matmul_q5_0_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m);
void matmul_q5_0_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						quant_scratch *qs);
void matmul_q5_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m);
void matmul_q5_1_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						quant_scratch *qs);
void matmul_q6_k_q8_qonly_f32(const void *w, const q8_k_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m);
void matmul_q4_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m);
void matmul_q5_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m);
void matmul_iq3_s_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m);

void matmul_f32_f32(const float *restrict w, const float *restrict x, float *restrict y, int n,
					int k);
void matmul_f32_f32_batch(const float *restrict w, const float *restrict x, float *restrict y,
						  int n, int k, int m, int x_row_stride, int y_row_stride);
void matmul_f16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					int k);
void matmul_f16_f32_batch(const void *restrict w, const float *restrict x, float *restrict y, int n,
						  int k, int m, int x_row_stride, int y_row_stride);
void matmul_bf16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					 int k);
void matmul_bf16_f32_batch(const void *restrict w, const float *restrict x, float *restrict y,
						   int n, int k, int m, int x_row_stride, int y_row_stride);

float dot_f32(const float *restrict a, const float *restrict b, int n);

void rmsnorm(const float *x, const float *w, float *y, int n, float eps);

static inline float silu(float x) {
	return x / (1.0f + expf(-x));
}

static inline float gelu_tanh(float x) {
	const float c	  = 0.7978845608028654f;
	float		x3	  = x * x * x;
	float		inner = c * (x + (0.044715f * x3));
	return 0.5f * x * (1.0f + tanhf(inner));
}

void moe_activate_silu(float *restrict act, const float *restrict gate, const float *restrict up,
					   int n, float gs, float us);
void moe_activate_gelu(float *restrict act, const float *restrict gate, const float *restrict up,
					   int n, float gs, float us);

void softmax_masked(float *restrict scores, int n_valid);

void rmsnorm_noweight(const float *x, float *y, int n, float eps);

void rmsnorm_per_head(const float *x, const float *w, float *y, int n_heads, int head_dim,
					  float eps);

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
	if (j < 4) {
		*d = q[j] & 63;
		*m = q[j + 4] & 63;
	} else {
		*d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
		*m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
	}
}

static const int8_t kvalues_iq4nl[16] = {
	-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

#endif