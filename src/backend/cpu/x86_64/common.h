#ifndef CPU_X86_64_COMMON_H
#define CPU_X86_64_COMMON_H

#include "numeric.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float f16_to_f32_fast(uint16_t h) {
	__m128 v = _mm_cvtph_ps(_mm_set1_epi16((short)h));
	return _mm_cvtss_f32(v);
}

static inline __m256 loadu_f16x8_to_ps(const uint16_t *p) {
	__m128i h = _mm_loadu_si128((const __m128i *)(p));
	return _mm256_cvtph_ps(h);
}

static inline __m256 loadu_bf16x8_to_ps(const uint16_t *p) {
	__m128i h = _mm_loadu_si128((const __m128i *)p);
	return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

static inline __m256 vround_away_ps(__m256 x) {
	const __m256 sign_mask = _mm256_set1_ps(-0.0f);
	__m256		 t		   = _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
	__m256		 frac	   = _mm256_andnot_ps(sign_mask, _mm256_sub_ps(x, t));
	__m256		 ge_half   = _mm256_cmp_ps(frac, _mm256_set1_ps(0.5f), _CMP_GE_OQ);
	__m256		 step	   = _mm256_or_ps(_mm256_set1_ps(1.0f), _mm256_and_ps(x, sign_mask));
	return _mm256_add_ps(t, _mm256_and_ps(ge_half, step));
}

static inline __m256 vexp_split_ps(__m256 x, __m256 *pow2n) {
	x		 = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-NUM_EXP_LIMIT)),
							 _mm256_set1_ps(NUM_EXP_LIMIT));
	__m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(NUM_INV_LN2)),
							   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(NUM_LN2_HI), x);
	r		 = _mm256_fnmadd_ps(n, _mm256_set1_ps(NUM_LN2_LO), r);

	__m256 t = _mm256_set1_ps(1.0f / 5040.0f);
	t		 = _mm256_fmadd_ps(r, t, _mm256_set1_ps(1.0f / 720.0f));
	t		 = _mm256_fmadd_ps(r, t, _mm256_set1_ps(1.0f / 120.0f));
	t		 = _mm256_fmadd_ps(r, t, _mm256_set1_ps(1.0f / 24.0f));
	t		 = _mm256_fmadd_ps(r, t, _mm256_set1_ps(1.0f / 6.0f));
	t		 = _mm256_fmadd_ps(r, t, _mm256_set1_ps(0.5f));

	__m256i e = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127));
	*pow2n	  = _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));
	return _mm256_fmadd_ps(_mm256_mul_ps(r, r), t, r);
}

static inline __m256 vexp_ps(__m256 x) {
	__m256 pow2n;
	__m256 em1 = vexp_split_ps(x, &pow2n);
	return _mm256_fmadd_ps(pow2n, em1, pow2n);
}

static inline __m256 vexpm1_ps(__m256 x) {
	__m256 pow2n;
	__m256 em1 = vexp_split_ps(x, &pow2n);
	return _mm256_fmadd_ps(pow2n, em1, _mm256_sub_ps(pow2n, _mm256_set1_ps(1.0f)));
}

static inline __m256 vtanh_ps(__m256 x) {
	const __m256 sign_mask = _mm256_set1_ps(-0.0f);
	__m256		 a = _mm256_min_ps(_mm256_andnot_ps(sign_mask, x), _mm256_set1_ps(NUM_TANH_LIMIT));
	__m256		 em1 = vexpm1_ps(_mm256_add_ps(a, a));
	__m256		 t	 = _mm256_div_ps(em1, _mm256_add_ps(em1, _mm256_set1_ps(2.0f)));
	return _mm256_or_ps(t, _mm256_and_ps(x, sign_mask));
}

static inline __m256 vsigmoid_ps(__m256 x) {
	const __m256 sign_mask = _mm256_set1_ps(-0.0f);
	const __m256 one	   = _mm256_set1_ps(1.0f);
	__m256		 e		   = vexp_ps(_mm256_or_ps(x, sign_mask));
	__m256		 nonneg	   = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GE_OQ);
	return _mm256_div_ps(_mm256_blendv_ps(e, one, nonneg), _mm256_add_ps(one, e));
}

static inline __m256 vsilu_ps(__m256 x) {
	__m256 e = vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
	return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

static inline __m256 vgelu_ps(__m256 x) {
	__m256 x2	 = _mm256_mul_ps(x, x);
	__m256 inner = _mm256_mul_ps(
		_mm256_mul_ps(x, _mm256_fmadd_ps(x2, _mm256_set1_ps(0.044715f), _mm256_set1_ps(1.0f))),
		_mm256_set1_ps(0.7978845608028654f));
	__m256 t = _mm256_add_ps(_mm256_set1_ps(1.0f), vtanh_ps(inner));
	return _mm256_mul_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.5f)), t);
}

static inline float vreduce_add_ps(__m256 v) {
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	__m128 s  = _mm_add_ps(lo, hi);
	s		  = _mm_hadd_ps(s, s);
	s		  = _mm_hadd_ps(s, s);
	return _mm_cvtss_f32(s);
}

static inline int32_t vreduce_add_epi32(__m256i v) {
	__m128i lo = _mm256_castsi256_si128(v);
	__m128i hi = _mm256_extracti128_si256(v, 1);
	__m128i s  = _mm_add_epi32(lo, hi);
	s		   = _mm_hadd_epi32(s, s);
	s		   = _mm_hadd_epi32(s, s);
	return _mm_cvtsi128_si32(s);
}

static inline __m128i vreduce4_add_epi32(__m256i a, __m256i b, __m256i c, __m256i d) {
	__m128i a4 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
	__m128i b4 = _mm_add_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1));
	__m128i c4 = _mm_add_epi32(_mm256_castsi256_si128(c), _mm256_extracti128_si256(c, 1));
	__m128i d4 = _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
	return _mm_hadd_epi32(_mm_hadd_epi32(a4, b4), _mm_hadd_epi32(c4, d4));
}

static inline float vreduce_max_ps(__m256 v) {
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	__m128 s  = _mm_max_ps(lo, hi);
	s		  = _mm_max_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1)));
	s		  = _mm_max_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 0, 3, 2)));
	return _mm_cvtss_f32(s);
}

static inline __m256i dotprod_u8_s8_i32(__m256i a_u, __m256i b_s) {
	__m256i prod = _mm256_maddubs_epi16(a_u, b_s);
	return _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
}

static inline __m256i dotprod_s8_s8_i32(__m256i a, __m256i b) {
	const __m256i lo_clamp = _mm256_set1_epi8(-127);
	const __m256i ac	   = _mm256_max_epi8(a, lo_clamp);
	const __m256i bc	   = _mm256_max_epi8(b, lo_clamp);
	const __m256i a_abs	   = _mm256_abs_epi8(ac);
	const __m256i b_signed = _mm256_sign_epi8(bc, ac);
	return _mm256_madd_epi16(_mm256_maddubs_epi16(a_abs, b_signed), _mm256_set1_epi16(1));
}

static inline __m256i dotprod_s8_s8_i32_noclamp(__m256i a, __m256i b) {
	const __m256i a_abs	   = _mm256_abs_epi8(a);
	const __m256i b_signed = _mm256_sign_epi8(b, a);
	return _mm256_madd_epi16(_mm256_maddubs_epi16(a_abs, b_signed), _mm256_set1_epi16(1));
}

static inline void vld16_s8_to_ps(const int8_t *p, __m256 *o0, __m256 *o1) {
	__m128i v = _mm_loadu_si128((const __m128i *)(p));
	*o0		  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
	*o1		  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
}

static inline __m256 vld8_s8_to_ps(const int8_t *p) {
	__m128i v = _mm_loadl_epi64((const __m128i *)(p));
	return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
}

#endif
