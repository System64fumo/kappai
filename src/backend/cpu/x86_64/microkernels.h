#ifndef CPU_X86_64_MICROKERNELS_H
#define CPU_X86_64_MICROKERNELS_H

#include "backend/cpu/quants.h"
#include "backend/cpu/x86_64/common.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

static inline void	  q5_0_unpack(const q5_0_block *b, __m128i *lo, __m128i *hi);
static inline void	  q5_1_unpack(const q5_1_block *b, __m128i *lo, __m128i *hi);
static inline int32_t q5_dot(__m128i lo, __m128i hi, __m256i xq);
static inline void	  q4k_block_dot(const q4_k_block *b, const q8_k_block *xb, int32_t *sumi_out,
									int32_t *summ_out);
static inline void	  q5k_block_dot(const q5_k_block *b, const q8_k_block *xb, int32_t *sumi_out,
									int32_t *summ_out);
static inline void	  q6_k_unpack_ymm(const uint8_t *restrict ql, const uint8_t *restrict qh,
									  __m256i out[8]);
static inline int32_t q6k_dot_ymm(const __m256i a[8], const __m256i xv[8], const int8_t *sc);
static inline int32_t iq3s_block_dot(const iq3s_block *restrict b, const int8_t *restrict q8);

static inline float mk_x86_q4_0_acc(const void *wb, const void *xb, float acc) {
	const q4_0_block *b	   = wb;
	const q8_0_block *x	   = xb;
	const __m128i	  q	   = _mm_loadu_si128((const __m128i *)(b->qs));
	const __m128i	  lo_u = _mm_and_si128(q, _mm_set1_epi8(0x0F));
	const __m128i	  hi_u = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0F));
	const __m128i	  lo   = _mm_sub_epi8(lo_u, _mm_set1_epi8(8));
	const __m128i	  hi   = _mm_sub_epi8(hi_u, _mm_set1_epi8(8));
	const __m256i	  q32  = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
	const __m256i	  xq32 = _mm256_loadu_si256((const __m256i *)(x->qs));
	const int32_t	  sumi = vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq32));
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)sumi, acc);
}

static inline float mk_x86_q4_1_acc(const void *wb, const void *xb, float acc) {
	const q4_1_block *b	   = wb;
	const q8_1_block *x	   = xb;
	const __m128i	  q	   = _mm_loadu_si128((const __m128i *)(b->qs));
	const __m128i	  lo   = _mm_and_si128(q, _mm_set1_epi8(0x0F));
	const __m128i	  hi   = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0F));
	const __m256i	  q32  = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
	const __m256i	  xq32 = _mm256_loadu_si256((const __m256i *)(x->qs));
	const int32_t	  sumi = vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq32));
	acc					   = fmaf(f16_to_f32_fast(b->m), f16_to_f32_fast(x->s), acc);
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)sumi, acc);
}

static inline float mk_x86_q5_0_acc(const void *wb, const void *xb, float acc) {
	const q5_0_block *b = wb;
	const q8_0_block *x = xb;
	__m128i			  lo, hi;
	q5_0_unpack(b, &lo, &hi);
	const __m256i xq32 = _mm256_loadu_si256((const __m256i *)(x->qs));
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)q5_dot(lo, hi, xq32), acc);
}

static inline float mk_x86_q5_1_acc(const void *wb, const void *xb, float acc) {
	const q5_1_block *b = wb;
	const q8_1_block *x = xb;
	__m128i			  lo, hi;
	q5_1_unpack(b, &lo, &hi);
	const __m256i xq32 = _mm256_loadu_si256((const __m256i *)(x->qs));
	acc				   = fmaf(f16_to_f32_fast(b->m), f16_to_f32_fast(x->s), acc);
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)q5_dot(lo, hi, xq32), acc);
}

static inline float mk_x86_q8_0_acc(const void *wb, const void *xb, float acc) {
	const q8_0_block *b	   = wb;
	const q8_0_block *x	   = xb;
	const __m256i	  q32  = _mm256_loadu_si256((const __m256i *)(b->qs));
	const __m256i	  xq32 = _mm256_loadu_si256((const __m256i *)(x->qs));
	const int32_t	  sumi = vreduce_add_epi32(dotprod_s8_s8_i32_noclamp(q32, xq32));
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)sumi, acc);
}

static inline float mk_x86_iq4_nl_acc(const void *wb, const void *xb, float acc) {
	const iq4_nl_block *b		  = wb;
	const q8_0_block   *x		  = xb;
	const __m128i		kvalues_u = _mm_loadu_si128((const __m128i *)(kvalues_iq4nl));
	const __m128i		lo_mask	  = _mm_set1_epi8(0x0F);
	const __m128i		q		  = _mm_loadu_si128((const __m128i *)(b->qs));
	const __m128i		lo_idx	  = _mm_and_si128(q, lo_mask);
	const __m128i		hi_idx	  = _mm_and_si128(_mm_srli_epi16(q, 4), lo_mask);
	const __m128i		lo		  = _mm_shuffle_epi8(kvalues_u, lo_idx);
	const __m128i		hi		  = _mm_shuffle_epi8(kvalues_u, hi_idx);
	const __m256i		q32		  = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
	const __m256i		xq32	  = _mm256_loadu_si256((const __m256i *)(x->qs));
	const int32_t		sumi	  = vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq32));
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)sumi, acc);
}

static inline float mk_x86_q4_k_acc(const void *wb, const void *xb, float acc) {
	const q4_k_block *b = wb;
	const q8_k_block *x = xb;
	int32_t			  sumi, summ;
	q4k_block_dot(b, x, &sumi, &summ);
	const float d	 = f16_to_f32_fast(b->d);
	const float dmin = f16_to_f32_fast(b->dmin);
	return fmaf(x->d, fmaf(d, (float)sumi, -(dmin * (float)summ)), acc);
}

static inline float mk_x86_q5_k_acc(const void *wb, const void *xb, float acc) {
	const q5_k_block *b = wb;
	const q8_k_block *x = xb;
	int32_t			  sumi, summ;
	q5k_block_dot(b, x, &sumi, &summ);
	const float d	 = f16_to_f32_fast(b->d);
	const float dmin = f16_to_f32_fast(b->dmin);
	return fmaf(x->d, fmaf(d, (float)sumi, -(dmin * (float)summ)), acc);
}

static inline float mk_x86_q6_k_acc(const void *wb, const void *xb, float acc) {
	const q6_k_block *b = wb;
	const q8_k_block *x = xb;
	__m256i			  ay[8];
	__m256i			  xv[8];
	q6_k_unpack_ymm(b->ql, b->qh, ay);
	for (int j = 0; j < 8; j++)
		xv[j] = _mm256_loadu_si256((const __m256i *)(x->qs + j * 32));
	return fmaf(f16_to_f32_fast(b->d) * x->d, (float)q6k_dot_ymm(ay, xv, b->scales), acc);
}

static inline float mk_x86_iq3_s_acc(const void *wb, const void *xb, float acc) {
	const iq3s_block *b = wb;
	const q8_k_block *x = xb;
	return fmaf(f16_to_f32_fast(b->d) * x->d, (float)iq3s_block_dot(b, x->qs), acc);
}

extern uint8_t iq3s_sign_pattern[256][8];

static inline __m128i iq_flip8_u64(uint64_t g, uint8_t signs) {
	__m128i gv = _mm_cvtsi64_si128((long long)g);
	__m128i m  = _mm_loadl_epi64((const __m128i *)iq3s_sign_pattern[signs]);
	return _mm_sub_epi8(_mm_xor_si128(gv, m), m);
}

static inline int32_t iq_dot16_avx2(const int8_t *vals, const int8_t *q8) {
	__m128i v  = _mm_loadu_si128((const __m128i *)vals);
	__m128i q  = _mm_loadu_si128((const __m128i *)q8);
	__m256i vv = _mm256_inserti128_si256(_mm256_setzero_si256(), v, 0);
	__m256i qq = _mm256_inserti128_si256(_mm256_setzero_si256(), q, 0);
	return vreduce_add_epi32(dotprod_s8_s8_i32_noclamp(vv, qq));
}

static inline int32_t iq_dot32_avx2(const int8_t *vals, const int8_t *q8) {
	__m256i v = _mm256_loadu_si256((const __m256i *)vals);
	__m256i q = _mm256_loadu_si256((const __m256i *)q8);
	return vreduce_add_epi32(dotprod_s8_s8_i32_noclamp(v, q));
}

static inline int32_t iq_dot32_dual16_avx2(const int8_t *vals, const int8_t *q8, int m0, int m1) {
	__m256i v  = _mm256_loadu_si256((const __m256i *)vals);
	__m256i q  = _mm256_loadu_si256((const __m256i *)q8);
	__m256i d  = dotprod_s8_s8_i32_noclamp(v, q);
	__m256i sc = _mm256_setr_epi32(m0, m0, m0, m0, m1, m1, m1, m1);
	return vreduce_add_epi32(_mm256_mullo_epi32(d, sc));
}

static inline __m256i iq_pack16(__m128i lo, __m128i hi) {
	return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static int32_t iq2_xxs_block_tot(const void *b_, const int8_t *restrict q8) {
	const iq2_xxs_block *b	 = b_;
	const uint8_t		*qs	 = (const uint8_t *)b->qs;
	int32_t				 tot = 0;
	for (int ib = 0; ib < 8; ib++) {
		uint32_t aux[2];
		memcpy(aux, qs + (8 * ib), 8);
		const uint8_t *a  = (const uint8_t *)aux;
		uint32_t	   s  = aux[1];
		__m256i		   wv = iq_pack16(
			_mm_unpacklo_epi64(
				iq_flip8_u64(ggml_iq2xxs_grid[a[0]], ggml_ksigns_iq2xs[s & 127]),
				iq_flip8_u64(ggml_iq2xxs_grid[a[1]], ggml_ksigns_iq2xs[(s >> 7) & 127])),
			_mm_unpacklo_epi64(
				iq_flip8_u64(ggml_iq2xxs_grid[a[2]], ggml_ksigns_iq2xs[(s >> 14) & 127]),
				iq_flip8_u64(ggml_iq2xxs_grid[a[3]], ggml_ksigns_iq2xs[(s >> 21) & 127])));
		tot += (1 + 2 * (int)(s >> 28)) * iq_dot32_avx2((const int8_t *)&wv, q8 + (ib * 32));
	}
	return tot;
}

static int32_t iq3_xxs_block_tot(const void *b_, const int8_t *restrict q8) {
	const iq3_xxs_block *b	 = b_;
	const uint8_t		*qs	 = b->qs;
	const uint8_t		*ss	 = b->qs + 64;
	int32_t				 tot = 0;
	for (int ib = 0; ib < 8; ib++) {
		uint32_t saux;
		memcpy(&saux, ss + (4 * ib), 4);
		const uint8_t *q = qs + (8 * ib);
		uint64_t c0 = (uint64_t)ggml_iq3xxs_grid[q[0]] | ((uint64_t)ggml_iq3xxs_grid[q[1]] << 32);
		uint64_t c1 = (uint64_t)ggml_iq3xxs_grid[q[2]] | ((uint64_t)ggml_iq3xxs_grid[q[3]] << 32);
		uint64_t c2 = (uint64_t)ggml_iq3xxs_grid[q[4]] | ((uint64_t)ggml_iq3xxs_grid[q[5]] << 32);
		uint64_t c3 = (uint64_t)ggml_iq3xxs_grid[q[6]] | ((uint64_t)ggml_iq3xxs_grid[q[7]] << 32);
		__m256i	 wv =
			iq_pack16(_mm_unpacklo_epi64(iq_flip8_u64(c0, ggml_ksigns_iq2xs[saux & 127]),
										 iq_flip8_u64(c1, ggml_ksigns_iq2xs[(saux >> 7) & 127])),
					  _mm_unpacklo_epi64(iq_flip8_u64(c2, ggml_ksigns_iq2xs[(saux >> 14) & 127]),
										 iq_flip8_u64(c3, ggml_ksigns_iq2xs[(saux >> 21) & 127])));
		tot += (1 + 2 * (int)(saux >> 28)) * iq_dot32_avx2((const int8_t *)&wv, q8 + (ib * 32));
	}
	return tot;
}

static int32_t iq2_s_block_tot(const void *b_, const int8_t *restrict q8) {
	const iq2_s_block *b	 = b_;
	const uint8_t	  *qs	 = b->qs;
	const uint8_t	  *signs = b->qs + 32;
	const uint8_t	  *qh	 = b->qh;
	int32_t			   tot	 = 0;
	for (int ib = 0; ib < 8; ib++) {
		uint8_t sb	= b->scales[ib];
		__m128i w01 = _mm_unpacklo_epi64(
			iq_flip8_u64(ggml_iq2s_grid[qs[0] | ((qh[ib] << 8) & 0x300)], signs[0]),
			iq_flip8_u64(ggml_iq2s_grid[qs[1] | ((qh[ib] << 6) & 0x300)], signs[1]));
		__m128i w23 = _mm_unpacklo_epi64(
			iq_flip8_u64(ggml_iq2s_grid[qs[2] | ((qh[ib] << 4) & 0x300)], signs[2]),
			iq_flip8_u64(ggml_iq2s_grid[qs[3] | ((qh[ib] << 2) & 0x300)], signs[3]));
		__m256i wv = iq_pack16(w01, w23);
		tot += iq_dot32_dual16_avx2((const int8_t *)&wv, q8 + (ib * 32), 1 + 2 * (sb & 0xf),
									1 + 2 * (sb >> 4));
		qs += 4;
		signs += 4;
	}
	return tot;
}

static int32_t iq4_xs_block_tot(const void *b_, const int8_t *restrict q8) {
	const iq4_xs_block *b	= b_;
	const __m128i		lut = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
	const __m128i		nib = _mm_set1_epi8(0x0f);
	const uint8_t	   *qs	= b->qs;
	int32_t				tot = 0;
	for (int ib = 0; ib < 8; ib++) {
		int		sl = (b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
		int		sh = (b->scales_h >> (2 * ib)) & 3;
		int		dl = (sl | (sh << 4)) - 32;
		__m128i q  = _mm_loadu_si128((const __m128i *)(qs + (ib * 16)));
		__m128i w0 = _mm_shuffle_epi8(lut, _mm_and_si128(q, nib));
		__m128i w1 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), nib));
		__m256i wv = iq_pack16(w0, w1);
		tot += dl * iq_dot32_avx2((const int8_t *)&wv, q8 + (ib * 32));
	}
	return tot;
}

#define MK_X86_ROW_DRIVER(fn, wblk_t, xqblk_t, elems, acc_fn)                                      \
	static void fn(const void *w, const xqblk_t *restrict xq, float *restrict y, int n, int k) {   \
		const int	   bpr		  = k / (elems);                                                   \
		const size_t   row_stride = (size_t)bpr * sizeof(wblk_t);                                  \
		const uint8_t *wb		  = w;                                                             \
		for (int i = 0; i < n; i++) {                                                              \
			const uint8_t *row = wb + ((size_t)i * row_stride);                                    \
			float		   acc = 0.0f;                                                             \
			for (int bi = 0; bi < bpr; bi++)                                                       \
				acc = acc_fn(row + ((size_t)bi * sizeof(wblk_t)), &xq[bi], acc);                   \
			y[i] = acc;                                                                            \
		}                                                                                          \
	}

#endif
