#ifndef NUMERIC_NEON_H
#define NUMERIC_NEON_H

#include "numeric.h"

#include <arm_neon.h>

static inline float32x4_t vexp_splitq_f32(float32x4_t x, float32x4_t *pow2n) {
	x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-NUM_EXP_LIMIT)), vdupq_n_f32(NUM_EXP_LIMIT));
	float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(NUM_INV_LN2)));
	float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(NUM_LN2_HI));
	r			  = vfmsq_f32(r, n, vdupq_n_f32(NUM_LN2_LO));

	float32x4_t t = vdupq_n_f32(1.0f / 5040.0f);
	t			  = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), r, t);
	t			  = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), r, t);
	t			  = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), r, t);
	t			  = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), r, t);
	t			  = vfmaq_f32(vdupq_n_f32(0.5f), r, t);

	int32x4_t e = vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127));
	*pow2n		= vreinterpretq_f32_s32(vshlq_n_s32(e, 23));
	return vfmaq_f32(r, vmulq_f32(r, r), t);
}

static inline float32x4_t vexpq_f32(float32x4_t x) {
	float32x4_t pow2n;
	float32x4_t em1 = vexp_splitq_f32(x, &pow2n);
	return vfmaq_f32(pow2n, pow2n, em1);
}

static inline float32x4_t vexpm1q_f32(float32x4_t x) {
	float32x4_t pow2n;
	float32x4_t em1 = vexp_splitq_f32(x, &pow2n);
	return vfmaq_f32(vsubq_f32(pow2n, vdupq_n_f32(1.0f)), pow2n, em1);
}

static inline float32x4_t vtanhq_f32(float32x4_t x) {
	float32x4_t a	 = vminq_f32(vabsq_f32(x), vdupq_n_f32(NUM_TANH_LIMIT));
	float32x4_t em1	 = vexpm1q_f32(vaddq_f32(a, a));
	float32x4_t t	 = vdivq_f32(em1, vaddq_f32(em1, vdupq_n_f32(2.0f)));
	uint32x4_t	sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000u));
	return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(t), sign));
}

static inline float32x4_t vsigmoidq_f32(float32x4_t x) {
	float32x4_t one	   = vdupq_n_f32(1.0f);
	float32x4_t e	   = vexpq_f32(vnegq_f32(vabsq_f32(x)));
	uint32x4_t	nonneg = vcgeq_f32(x, vdupq_n_f32(0.0f));
	return vdivq_f32(vbslq_f32(nonneg, one, e), vaddq_f32(one, e));
}

static inline float32x4_t vsiluq_f32(float32x4_t x) {
	float32x4_t e = vexpq_f32(vnegq_f32(x));
	return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0f), e));
}

static inline float32x4_t vgeluq_f32(float32x4_t x) {
	float32x4_t x2 = vmulq_f32(x, x);
	float32x4_t inner =
		vmulq_f32(vmulq_f32(x, vfmaq_f32(vdupq_n_f32(1.0f), x2, vdupq_n_f32(0.044715f))),
				  vdupq_n_f32(0.7978845608028654f));
	float32x4_t t = vaddq_f32(vdupq_n_f32(1.0f), vtanhq_f32(inner));
	return vmulq_f32(vmulq_f32(x, vdupq_n_f32(0.5f)), t);
}

#endif
