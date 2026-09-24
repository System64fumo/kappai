#ifndef CPU_AARCH64_COMMON_H
#define CPU_AARCH64_COMMON_H

#include <arm_neon.h>
#include <string.h>

#ifndef CACHE_LINE_BYTES
#define CACHE_LINE_BYTES 64
#endif

#ifndef L1D_SIZE_BYTES
#define L1D_SIZE_BYTES (32 * 1024)
#endif

#ifndef L2_SIZE_BYTES
#define L2_SIZE_BYTES (512 * 1024)
#endif

#ifndef PREFETCH_LOCALITY
#define PREFETCH_LOCALITY 1
#endif

#define PREFETCH(addr) __builtin_prefetch((addr), 0, PREFETCH_LOCALITY)

#define PF_ELEMS(base_elems_64b_line) (((base_elems_64b_line) * (CACHE_LINE_BYTES)) / 64)

static inline float f16_to_f32_fast(uint16_t h) {
	__fp16 v;
	memcpy(&v, &h, sizeof(v));
	return (float)v;
}

static inline float32x4_t vexpq_f32(float32x4_t x) {
	x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.0f)), vdupq_n_f32(88.0f));

	const float32x4_t inv_ln2 = vdupq_n_f32(1.4426950408889634f);
	const float32x4_t ln2_hi  = vdupq_n_f32(6.9314718055994529e-01f);
	const float32x4_t ln2_lo  = vdupq_n_f32(1.9082149292705877e-10f);

	float32x4_t n_f = vrndnq_f32(vmulq_f32(x, inv_ln2));
	int32x4_t	n_i = vcvtq_s32_f32(n_f);

	float32x4_t r = vfmsq_f32(vfmsq_f32(x, n_f, ln2_hi), n_f, ln2_lo);

	float32x4_t p = vdupq_n_f32(1.0f / 120.0f);
	p			  = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), r, p);
	p			  = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), r, p);
	p			  = vfmaq_f32(vdupq_n_f32(0.5f), r, p);
	p			  = vfmaq_f32(vdupq_n_f32(1.0f), r, p);
	p			  = vfmaq_f32(vdupq_n_f32(1.0f), r, p);

	int32x4_t	e	  = vaddq_s32(n_i, vdupq_n_s32(127));
	int32x4_t	e_shl = vshlq_n_s32(e, 23);
	float32x4_t pow2n = vreinterpretq_f32_s32(e_shl);

	return vmulq_f32(p, pow2n);
}

static inline float32x4_t vtanhq_f32(float32x4_t x) {
	x				= vminq_f32(vmaxq_f32(x, vdupq_n_f32(-20.0f)), vdupq_n_f32(20.0f));
	float32x4_t e2x = vexpq_f32(vmulq_n_f32(x, 2.0f));
	float32x4_t num = vsubq_f32(e2x, vdupq_n_f32(1.0f));
	float32x4_t den = vaddq_f32(e2x, vdupq_n_f32(1.0f));

	float32x4_t r = vrecpeq_f32(den);
	r			  = vmulq_f32(vrecpsq_f32(den, r), r);
	r			  = vmulq_f32(vrecpsq_f32(den, r), r);

	return vmulq_f32(num, r);
}

static inline float fast_expf(float x) {
	if (x < -88.0f)
		x = -88.0f;
	if (x > 88.0f)
		x = 88.0f;
	const float inv_ln2 = 1.4426950408889634f;
	const float ln2_hi	= 6.9314718055994529e-01f;
	const float ln2_lo	= 1.9082149292705877e-10f;
	float		n		= __builtin_rintf(x * inv_ln2);
	float		r		= __builtin_fmaf(-n, ln2_hi, x);
	r					= __builtin_fmaf(-n, ln2_lo, r);
	float p				= 1.0f / 120.0f;
	p					= __builtin_fmaf(r, p, 1.0f / 24.0f);
	p					= __builtin_fmaf(r, p, 1.0f / 6.0f);
	p					= __builtin_fmaf(r, p, 0.5f);
	p					= __builtin_fmaf(r, p, 1.0f);
	p					= __builtin_fmaf(r, p, 1.0f);
	int32_t e			= (int32_t)n + 127;
	e <<= 23;
	float pow2n;
	memcpy(&pow2n, &e, sizeof(pow2n));
	return p * pow2n;
}

static inline float32x4_t silu_mul_vec4(float32x4_t g, float32x4_t u) {
	float32x4_t e = vexpq_f32(vnegq_f32(g));
	float32x4_t d = vaddq_f32(vdupq_n_f32(1.0f), e);
	float32x4_t r = vrecpeq_f32(d);
	r			  = vmulq_f32(vrecpsq_f32(d, r), r);
	r			  = vmulq_f32(vrecpsq_f32(d, r), r);
	return vmulq_f32(vmulq_f32(g, r), u);
}

static inline float32x4_t gelu_mul_vec4(float32x4_t g, float32x4_t u) {
	const float c_fit = 0.7978845608028654f;
	const float c_x3  = 0.044715f;
	float32x4_t gsq	  = vmulq_f32(g, g);
	float32x4_t gc	  = vmulq_f32(g, vfmaq_n_f32(vdupq_n_f32(1.0f), gsq, c_x3));
	float32x4_t in	  = vmulq_n_f32(gc, c_fit);
	float32x4_t t	  = vtanhq_f32(in);
	return vmulq_f32(vmulq_f32(vmulq_n_f32(g, 0.5f), vaddq_f32(vdupq_n_f32(1.0f), t)), u);
}

#endif