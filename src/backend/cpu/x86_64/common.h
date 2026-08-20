#ifndef CPU_X86_64_COMMON_H
#define CPU_X86_64_COMMON_H

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

static inline __m256 vexp_ps(__m256 x) {
	x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.0f)), _mm256_set1_ps(88.0f));

	const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
	const __m256 ln2_hi	 = _mm256_set1_ps(6.9314718055994529e-01f);
	const __m256 ln2_lo	 = _mm256_set1_ps(1.9082149292705877e-10f);

	__m256 n_f =
		_mm256_round_ps(_mm256_mul_ps(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m256i n_i = _mm256_cvtps_epi32(n_f);

	__m256 r = _mm256_fnmadd_ps(n_f, ln2_hi, x);
	r		 = _mm256_fnmadd_ps(n_f, ln2_lo, r);

	__m256 p = _mm256_set1_ps(1.0f / 120.0f);
	p		 = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0f / 24.0f));
	p		 = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0f / 6.0f));
	p		 = _mm256_fmadd_ps(r, p, _mm256_set1_ps(0.5f));
	p		 = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0f));
	p		 = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0f));

	__m256i e	  = _mm256_add_epi32(n_i, _mm256_set1_epi32(127));
	__m256i e_shl = _mm256_slli_epi32(e, 23);
	__m256	pow2n = _mm256_castsi256_ps(e_shl);

	return _mm256_mul_ps(p, pow2n);
}

static inline __m256 vtanh_ps(__m256 x) {
	x		   = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-20.0f)), _mm256_set1_ps(20.0f));
	__m256 e2x = vexp_ps(_mm256_mul_ps(x, _mm256_set1_ps(2.0f)));
	__m256 num = _mm256_sub_ps(e2x, _mm256_set1_ps(1.0f));
	__m256 den = _mm256_add_ps(e2x, _mm256_set1_ps(1.0f));
	return _mm256_div_ps(num, den);
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

static inline __m256i maddubs_scale_i32(__m256i a_u, __m256i b_s, int scale) {
	return _mm256_madd_epi16(_mm256_maddubs_epi16(a_u, b_s), _mm256_set1_epi16((int16_t)scale));
}

static inline __m256i dotprod_s8_s8_i32(__m256i a, __m256i b) {
	/* IQ/Q8 quantizers clamp to [-127, 127], so abs(a) is representable as u8.
	 * Move a's sign into b and use the AVX2 u8*s8 pairwise dot product.  Each
	 * adjacent pair is bounded by 2*127*127, avoiding maddubs saturation. */
	const __m256i a_abs = _mm256_abs_epi8(a);
	const __m256i b_signed = _mm256_sign_epi8(b, a);
	return _mm256_madd_epi16(_mm256_maddubs_epi16(a_abs, b_signed),
							 _mm256_set1_epi16(1));
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
