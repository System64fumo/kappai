#ifndef NUMERIC_H
#define NUMERIC_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#define NUM_EXP_LIMIT 88.0f
#define NUM_TANH_LIMIT 10.0f
#define NUM_INV_LN2 1.4426950408889634f
#define NUM_LN2_HI 6.9314718055994529e-01f
#define NUM_LN2_LO (-1.9046542121259336e-09f)
#define NUM_RMS_LANES 256
#define NUM_DOT_LANES 16
#define NUM_ATTN_BLOCK 4

static inline uint32_t num_f32_bits(float f) {
	uint32_t u;
	memcpy(&u, &f, sizeof(u));
	return u;
}

static inline float num_bits_f32(uint32_t u) {
	float f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

static inline float num_exp_poly(float r) {
	float t	 = 1.0f / 5040.0f;
	t		 = fmaf(r, t, 1.0f / 720.0f);
	t		 = fmaf(r, t, 1.0f / 120.0f);
	t		 = fmaf(r, t, 1.0f / 24.0f);
	t		 = fmaf(r, t, 1.0f / 6.0f);
	t		 = fmaf(r, t, 0.5f);
	float r2 = r * r;
	return fmaf(r2, t, r);
}

static inline float num_exp_split(float x, float *pow2n) {
	float xc = fminf(fmaxf(x, -NUM_EXP_LIMIT), NUM_EXP_LIMIT);
	float n	 = rintf(xc * NUM_INV_LN2);
	float r	 = fmaf(-n, NUM_LN2_HI, xc);
	r		 = fmaf(-n, NUM_LN2_LO, r);
	*pow2n	 = num_bits_f32((uint32_t)((int32_t)n + 127) << 23);
	return num_exp_poly(r);
}

static inline float num_expf(float x) {
	float pow2n;
	float em1 = num_exp_split(x, &pow2n);
	return fmaf(pow2n, em1, pow2n);
}

static inline float num_expm1f(float x) {
	float pow2n;
	float em1 = num_exp_split(x, &pow2n);
	return fmaf(pow2n, em1, pow2n - 1.0f);
}

static inline float num_tanhf(float x) {
	float a	  = fminf(fabsf(x), NUM_TANH_LIMIT);
	float em1 = num_expm1f(a + a);
	float t	  = em1 / (em1 + 2.0f);
	return copysignf(t, x);
}

static inline float num_sigmoidf(float x) {
	float e = num_expf(-fabsf(x));
	return (x >= 0.0f ? 1.0f : e) / (1.0f + e);
}

static inline float num_silu(float x) {
	return x / (1.0f + num_expf(-x));
}

static inline float num_gelu_tanh(float x) {
	float x2	= x * x;
	float inner = (x * fmaf(x2, 0.044715f, 1.0f)) * 0.7978845608028654f;
	return (x * 0.5f) * (1.0f + num_tanhf(inner));
}

static inline float num_rms_scale(float sum_sq, int n, float eps) {
	float v = sum_sq / (float)n + eps;
	return v > 0.0f ? 1.0f / sqrtf(v) : 0.0f;
}

static inline uint16_t num_f32_to_f16(float f) {
	uint32_t bits = num_f32_bits(f);
	uint32_t sign = (bits >> 16) & 0x8000u;
	uint32_t a	  = bits & 0x7FFFFFFFu;
	if (a >= 0x7F800000u)
		return (uint16_t)(sign | (a > 0x7F800000u ? 0x7E00u : 0x7C00u));
	if (a >= 0x477FF000u)
		return (uint16_t)(sign | 0x7C00u);
	if (a <= 0x33000000u)
		return (uint16_t)sign;
	if (a < 0x38800000u) {
		uint32_t shift = 126u - (a >> 23);
		uint32_t m	   = (a & 0x7FFFFFu) | 0x800000u;
		uint32_t h	   = m >> shift;
		uint32_t rem   = m & ((1u << shift) - 1u);
		uint32_t half  = 1u << (shift - 1u);
		if (rem > half || (rem == half && (h & 1u)))
			h++;
		return (uint16_t)(sign | h);
	}
	uint32_t r = a + 0xFFFu + ((a >> 13) & 1u);
	return (uint16_t)(sign | ((r - 0x38000000u) >> 13));
}

#endif
