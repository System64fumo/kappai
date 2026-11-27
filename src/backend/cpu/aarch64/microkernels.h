#ifndef CPU_AARCH64_MICROKERNELS_H
#define CPU_AARCH64_MICROKERNELS_H

#include "backend/cpu/quants.h"
#include <string.h>

static inline int32x4_t dotprod2_s8(int8x16_t a_lo, int8x16_t b_lo, int8x16_t a_hi,
									int8x16_t b_hi) {
#if defined(__ARM_FEATURE_DOTPROD)
	return vdotq_s32(vdotq_s32(vdupq_n_s32(0), a_lo, b_lo), a_hi, b_hi);
#else
	int16x8_t p0  = vmull_s8(vget_low_s8(a_lo), vget_low_s8(b_lo));
	int16x8_t p1  = vmull_s8(vget_high_s8(a_lo), vget_high_s8(b_lo));
	int16x8_t p2  = vmull_s8(vget_low_s8(a_hi), vget_low_s8(b_hi));
	int16x8_t p3  = vmull_s8(vget_high_s8(a_hi), vget_high_s8(b_hi));
	int32x4_t dot = vpaddlq_s16(p0);
	dot			  = vpadalq_s16(dot, p1);
	dot			  = vpadalq_s16(dot, p2);
	dot			  = vpadalq_s16(dot, p3);
	return dot;
#endif
}

static inline int32x4_t q5_dot(int8x16_t lo, int8x16_t hi, int8x16_t xlo, int8x16_t xhi) {
#if defined(__ARM_FEATURE_DOTPROD)
	return vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, xlo), hi, xhi);
#else
	int16x8_t p0  = vmull_s8(vget_low_s8(lo), vget_low_s8(xlo));
	int16x8_t p1  = vmull_s8(vget_high_s8(lo), vget_high_s8(xlo));
	int16x8_t p2  = vmull_s8(vget_low_s8(hi), vget_low_s8(xhi));
	int16x8_t p3  = vmull_s8(vget_high_s8(hi), vget_high_s8(xhi));
	int32x4_t acc = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
	acc			  = vaddq_s32(acc, vpaddlq_s16(p2));
	acc			  = vaddq_s32(acc, vpaddlq_s16(p3));
	return acc;
#endif
}

static inline void q4_0_unpack(const q4_0_block *b, int8x16_t *lo, int8x16_t *hi) {
	const uint8x16_t q = vld1q_u8(b->qs);
	const int8x16_t	 lo8 =
		vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
	const int8x16_t hi8 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), vdupq_n_s8(8));
	*lo					= lo8;
	*hi					= hi8;
}

static inline void q5_0_unpack(const q5_0_block *b, int8x16_t *lo, int8x16_t *hi) {
	uint32_t qh;
	memcpy(&qh, b->qh, 4);

	const uint8x16_t qs			 = vld1q_u8(b->qs);
	const uint8x16_t nib_lo		 = vandq_u8(qs, vdupq_n_u8(0x0F));
	const uint8x16_t nib_hi		 = vshrq_n_u8(qs, 4);
	const uint8x16_t qh_lo_bytes = vreinterpretq_u8_u32(vdupq_n_u32(qh));
	const uint8x16_t qh_hi_bytes = vreinterpretq_u8_u32(vdupq_n_u32(qh >> 16));
	const uint8x16_t byte_idx	 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
	const uint8x16_t bit_in_byte = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
	const uint8x16_t qh_lo_sel	 = vqtbl1q_u8(qh_lo_bytes, byte_idx);
	const uint8x16_t qh_hi_sel	 = vqtbl1q_u8(qh_hi_bytes, byte_idx);
	const int8x16_t	 lshift4	 = vsubq_s8(vdupq_n_s8(4), vreinterpretq_s8_u8(bit_in_byte));
	const uint8x16_t hi_lo_v	 = vandq_u8(vshlq_u8(qh_lo_sel, lshift4), vdupq_n_u8(0x10));
	const uint8x16_t hi_hi_v	 = vandq_u8(vshlq_u8(qh_hi_sel, lshift4), vdupq_n_u8(0x10));
	const uint8x16_t v16		 = vdupq_n_u8(16);
	*lo							 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(nib_lo, hi_lo_v), v16));
	*hi							 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(nib_hi, hi_hi_v), v16));
}

static inline void q5_1_unpack(const q5_1_block *b, int8x16_t *lo, int8x16_t *hi) {
	uint32_t qh;
	memcpy(&qh, b->qh, 4);

	const uint8x16_t qs			 = vld1q_u8(b->qs);
	const uint8x16_t nib_lo		 = vandq_u8(qs, vdupq_n_u8(0x0F));
	const uint8x16_t nib_hi		 = vshrq_n_u8(qs, 4);
	const uint8x16_t qh_lo_bytes = vreinterpretq_u8_u32(vdupq_n_u32(qh));
	const uint8x16_t qh_hi_bytes = vreinterpretq_u8_u32(vdupq_n_u32(qh >> 16));
	const uint8x16_t byte_idx	 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
	const uint8x16_t bit_in_byte = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
	const uint8x16_t qh_lo_sel	 = vqtbl1q_u8(qh_lo_bytes, byte_idx);
	const uint8x16_t qh_hi_sel	 = vqtbl1q_u8(qh_hi_bytes, byte_idx);
	const int8x16_t	 lshift4	 = vsubq_s8(vdupq_n_s8(4), vreinterpretq_s8_u8(bit_in_byte));
	const uint8x16_t hi_lo_v	 = vandq_u8(vshlq_u8(qh_lo_sel, lshift4), vdupq_n_u8(0x10));
	const uint8x16_t hi_hi_v	 = vandq_u8(vshlq_u8(qh_hi_sel, lshift4), vdupq_n_u8(0x10));
	*lo							 = vreinterpretq_s8_u8(vorrq_u8(nib_lo, hi_lo_v));
	*hi							 = vreinterpretq_s8_u8(vorrq_u8(nib_hi, hi_hi_v));
}

static inline float mk_a64_q4_0_acc(const void *wb, const void *xb, float acc) {
	const q4_0_block *b = wb;
	const q8_0_block *x = xb;
	int8x16_t		  lo, hi;
	q4_0_unpack(b, &lo, &hi);
	const int32x4_t dot = dotprod2_s8(lo, vld1q_s8(x->qs), hi, vld1q_s8(x->qs + 16));
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)vaddvq_s32(dot), acc);
}

static inline float mk_a64_q4_1_acc(const void *wb, const void *xb, float acc) {
	const q4_1_block *b	  = wb;
	const q8_1_block *x	  = xb;
	const uint8x16_t  q	  = vld1q_u8(b->qs);
	const int8x16_t	  lo  = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F)));
	const int8x16_t	  hi  = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
	const int32x4_t	  dot = q5_dot(lo, hi, vld1q_s8(x->qs), vld1q_s8(x->qs + 16));
	acc					  = fmaf(f16_to_f32_fast(b->m), f16_to_f32_fast(x->s), acc);
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)vaddvq_s32(dot), acc);
}

static inline float mk_a64_q5_0_acc(const void *wb, const void *xb, float acc) {
	const q5_0_block *b = wb;
	const q8_0_block *x = xb;
	int8x16_t		  lo, hi;
	q5_0_unpack(b, &lo, &hi);
	const int32x4_t dot = q5_dot(lo, hi, vld1q_s8(x->qs), vld1q_s8(x->qs + 16));
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)vaddvq_s32(dot), acc);
}

static inline float mk_a64_q5_1_acc(const void *wb, const void *xb, float acc) {
	const q5_1_block *b = wb;
	const q8_1_block *x = xb;
	int8x16_t		  lo, hi;
	q5_1_unpack(b, &lo, &hi);
	const int32x4_t dot = q5_dot(lo, hi, vld1q_s8(x->qs), vld1q_s8(x->qs + 16));
	acc					= fmaf(f16_to_f32_fast(b->m), f16_to_f32_fast(x->s), acc);
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)vaddvq_s32(dot), acc);
}

static inline float mk_a64_q8_0_acc(const void *wb, const void *xb, float acc) {
	const q8_0_block *b = wb;
	const q8_0_block *x = xb;
	const int32x4_t	  dot =
		dotprod2_s8(vld1q_s8(b->qs), vld1q_s8(x->qs), vld1q_s8(b->qs + 16), vld1q_s8(x->qs + 16));
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)vaddvq_s32(dot), acc);
}

static inline float mk_a64_iq4_nl_acc(const void *wb, const void *xb, float acc) {
	const iq4_nl_block *b		  = wb;
	const q8_0_block   *x		  = xb;
	const uint8x16_t	kvalues_u = vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t	q		  = vld1q_u8(b->qs);
	const int8x16_t lo	= vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vandq_u8(q, vdupq_n_u8(0x0F))));
	const int8x16_t hi	= vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vshrq_n_u8(q, 4)));
	const int32x4_t dot = q5_dot(lo, hi, vld1q_s8(x->qs), vld1q_s8(x->qs + 16));
	return fmaf(f16_to_f32_fast(b->d) * f16_to_f32_fast(x->d), (float)vaddvq_s32(dot), acc);
}

static inline float mk_a64_q6_k_acc(const void *wb, const void *xb, float acc) {
	const q6_k_block *b = wb;
	const q8_k_block *x = xb;
	int8_t			  a[256];
	const uint8_t	 *ql = b->ql;
	const uint8_t	 *qh = b->qh;
	int8_t			 *p	 = a;
	for (int n_iter = 0; n_iter < 2; n_iter++) {
		for (int l = 0; l < 32; l++) {
			p[l]	  = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
			p[l + 32] = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
			p[l + 64] = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
			p[l + 96] = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
		}
		p += 128;
		ql += 64;
		qh += 32;
	}
	const int8_t *sc   = b->scales;
	const int8_t *q8   = x->qs;
	int32_t		  acc0 = 0;
	int32_t		  acc1 = 0;
	for (int j = 0; j < 16; j++) {
		const int s	 = sc[j];
		int32_t	  l0 = 0;
		int32_t	  l1 = 0;
		for (int l = 0; l < 8; l++) {
			l0 += (int)q8[l] * (int)a[(j * 16) + l];
			l1 += (int)q8[l + 8] * (int)a[(j * 16) + l + 8];
		}
		acc0 += s * l0;
		acc1 += s * l1;
		q8 += 16;
	}
	return fmaf(f16_to_f32_fast(b->d) * x->d, (float)(acc0 + acc1), acc);
}

static inline float mk_a64_q4_k_acc(const void *wb, const void *xb, float acc) {
	const q4_k_block *b		 = wb;
	const q8_k_block *x		 = xb;
	const float		  d		 = f16_to_f32_fast(b->d);
	const float		  dmin	 = f16_to_f32_fast(b->dmin);
	const uint8_t	 *qbytes = b->qs;
	const uint8_t	 *sc	 = b->scales;
	const int8_t	 *xq8	 = x->qs;
	const int16_t	 *bs	 = x->bsums;
	int32_t			  sumi	 = 0;
	int32_t			  summ	 = 0;
	int				  is	 = 0;
	int				  ib	 = 0;
	uint8_t			  scu8;
	uint8_t			  mu8;
	for (int g = 0; g < 4; g++) {
		get_scale_min_k4(is + 0, sc, &scu8, &mu8);
		const int s0 = scu8;
		const int m0 = mu8;
		get_scale_min_k4(is + 1, sc, &scu8, &mu8);
		const int	   s1	= scu8;
		const int	   m1	= mu8;
		const uint8_t *qg	= qbytes + (ptrdiff_t)(g * 32);
		const int8_t  *xq0	= xq8 + (ptrdiff_t)(g * 64);
		const int8_t  *xq1	= xq8 + (ptrdiff_t)(g * 64) + 32;
		int32_t		   dot0 = 0;
		int32_t		   dot1 = 0;
		for (int l = 0; l < 32; l++) {
			const uint8_t byte = qg[l];
			dot0 += (int32_t)(byte & 0xF) * (int32_t)xq0[l];
			dot1 += (int32_t)(byte >> 4) * (int32_t)xq1[l];
		}
		sumi += (s0 * dot0) + (s1 * dot1);
		summ += m0 * (int)(bs[ib] + bs[ib + 1]);
		ib += 2;
		summ += m1 * (int)(bs[ib] + bs[ib + 1]);
		ib += 2;
		is += 2;
	}
	return fmaf(x->d, fmaf(d, (float)sumi, -(dmin * (float)summ)), acc);
}

static inline float mk_a64_q5_k_acc(const void *wb, const void *xb, float acc) {
	const q5_k_block *b		 = wb;
	const q8_k_block *x		 = xb;
	const float		  d		 = f16_to_f32_fast(b->d);
	const float		  dmin	 = f16_to_f32_fast(b->dmin);
	const uint8_t	 *qbytes = b->qs;
	const uint8_t	 *qh	 = b->qh;
	const uint8_t	 *sc	 = b->scales;
	const int8_t	 *xq8	 = x->qs;
	const int16_t	 *bs	 = x->bsums;
	int32_t			  sumi	 = 0;
	int32_t			  summ	 = 0;
	int				  is	 = 0;
	int				  ib	 = 0;
	uint8_t			  scu8;
	uint8_t			  mu8;
	uint8_t			  u1 = 1;
	uint8_t			  u2 = 2;
	for (int g = 0; g < 4; g++) {
		get_scale_min_k4(is + 0, sc, &scu8, &mu8);
		const int s0 = scu8;
		const int m0 = mu8;
		get_scale_min_k4(is + 1, sc, &scu8, &mu8);
		const int	   s1	= scu8;
		const int	   m1	= mu8;
		const uint8_t *qsg	= qbytes + (ptrdiff_t)(g * 32);
		const int8_t  *xq0	= xq8 + (ptrdiff_t)(g * 64);
		const int8_t  *xq1	= xq8 + (ptrdiff_t)(g * 64) + 32;
		int32_t		   dot0 = 0;
		int32_t		   dot1 = 0;
		for (int l = 0; l < 32; l++) {
			const uint8_t byte = qsg[l];
			const int	  hi0  = (qh[l] & u1) ? 16 : 0;
			const int	  hi1  = (qh[l] & u2) ? 16 : 0;
			dot0 += (int32_t)((byte & 0xF) + hi0) * (int32_t)xq0[l];
			dot1 += (int32_t)((byte >> 4) + hi1) * (int32_t)xq1[l];
		}
		sumi += (s0 * dot0) + (s1 * dot1);
		summ += m0 * (int)(bs[ib] + bs[ib + 1]);
		ib += 2;
		summ += m1 * (int)(bs[ib] + bs[ib + 1]);
		ib += 2;
		is += 2;
		u1 <<= 2;
		u2 <<= 2;
	}
	return fmaf(x->d, fmaf(d, (float)sumi, -(dmin * (float)summ)), acc);
}

static inline int32_t iq3s_block_dot(const iq3s_block *b, const int8_t *restrict q8);

static inline float mk_a64_iq3_s_acc(const void *wb, const void *xb, float acc) {
	const iq3s_block *b = wb;
	const q8_k_block *x = xb;
	return fmaf(f16_to_f32_fast(b->d) * x->d, (float)iq3s_block_dot(b, x->qs), acc);
}

extern uint8_t iq3s_sign_pattern[256][8];

static inline int8x8_t iq_apply_signs8(uint8x8_t grid, uint8_t signs) {
	uint8x8_t m	  = vdup_n_u8(signs);
	uint8x8_t km  = vld1_u8(kmask_iq2xs);
	uint8x8_t hit = vtst_u8(m, km);
	int8x8_t  g	  = vreinterpret_s8_u8(grid);
	return vbsl_s8(hit, vneg_s8(g), g);
}
static inline int32_t neon_dot16(const int8_t *a, const int8_t *b) {
	int8x16_t av = vld1q_s8(a);
	int8x16_t bv = vld1q_s8(b);
	return vaddvq_s32(dotprod2_s8(av, bv, vdupq_n_s8(0), vdupq_n_s8(0)));
}
static inline int32_t neon_dot32(const int8_t *a, const int8_t *b) {
	int8x16_t a0 = vld1q_s8(a);
	int8x16_t a1 = vld1q_s8(a + 16);
	int8x16_t b0 = vld1q_s8(b);
	int8x16_t b1 = vld1q_s8(b + 16);
	return vaddvq_s32(dotprod2_s8(a0, b0, a1, b1));
}
static inline int32_t neon_dot8v(int8x8_t a, int8x8_t b) {
	return vaddvq_s32(vpaddlq_s16(vmull_s8(a, b)));
}
static inline void neon_store_scaled8(const int8_t *v, float *dst, float scale) {
	int8x8_t	s  = vld1_s8(v);
	int16x8_t	e  = vmovl_s8(s);
	float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(e)));
	float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(e)));
	float32x4_t sc = vdupq_n_f32(scale);
	vst1q_f32(dst, vmulq_f32(f0, sc));
	vst1q_f32(dst + 4, vmulq_f32(f1, sc));
}
static inline void iq2_s_unpack(const iq2_s_block *b, int8_t *vals, float *scales) {
	const uint8_t *qs	 = b->qs;
	const uint8_t *qh	 = b->qh;
	const uint8_t *signs = qs + 32;
	for (int ib = 0; ib < 8; ib++) {
		scales[2 * ib + 0] = (float)(1 + 2 * (b->scales[ib] & 0xf));
		scales[2 * ib + 1] = (float)(1 + 2 * (b->scales[ib] >> 4));
		for (int l = 0; l < 4; l++) {
			uint8x8_t grid = vld1_u8(
				(const uint8_t *)&ggml_iq2s_grid[qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300)]);
			vst1_s8(vals + ib * 32 + l * 8, iq_apply_signs8(grid, signs[l]));
		}
		qs += 4;
		signs += 4;
	}
}
static inline void iq2_xxs_unpack(const iq2_xxs_block *b, int8_t *vals, float *scales) {
	uint32_t	   aux[2];
	const uint8_t *a8 = (const uint8_t *)aux;
	for (int ib = 0; ib < 8; ib++) {
		memcpy(aux, b->qs + 4 * ib, 8);
		scales[ib] = (float)(1 + 2 * (aux[1] >> 28));
		for (int l = 0; l < 4; l++) {
			uint8x8_t grid	= vld1_u8((const uint8_t *)&ggml_iq2xxs_grid[a8[l]]);
			uint8_t	  signs = ggml_ksigns_iq2xs[(aux[1] >> (7 * l)) & 127];
			vst1_s8(vals + ib * 32 + l * 8, iq_apply_signs8(grid, signs));
		}
	}
}
static inline void iq3_xxs_unpack(const iq3_xxs_block *b, int8_t *vals, float *scales) {
	uint32_t	   aux;
	const uint8_t *qs = b->qs;
	const uint8_t *ss = qs + 64;
	for (int ib = 0; ib < 8; ib++) {
		memcpy(&aux, ss + 4 * ib, 4);
		scales[ib] = (float)(1 + 2 * (aux >> 28));
		for (int l = 0; l < 4; l++) {
			uint8_t signs = ggml_ksigns_iq2xs[(aux >> (7 * l)) & 127];
			uint8_t tmp[8];
			memcpy(tmp, &ggml_iq3xxs_grid[qs[2 * l + 0]], 4);
			memcpy(tmp + 4, &ggml_iq3xxs_grid[qs[2 * l + 1]], 4);
			int8x8_t w = iq_apply_signs8(vld1_u8(tmp), signs);
			vst1_s8(vals + ib * 32 + l * 8, w);
		}
		qs += 8;
	}
}
static inline void iq4_xs_unpack(const iq4_xs_block *b, int8_t *vals, float *scales) {
	const uint8_t *qs = b->qs;
	for (int ib = 0; ib < 8; ib++) {
		int sl	   = (b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
		int sh	   = (b->scales_h >> (2 * ib)) & 3;
		scales[ib] = (float)((sl | (sh << 4)) - 32);
		for (int j = 0; j < 16; j++) {
			vals[ib * 32 + j]	   = kvalues_iq4nl[qs[j] & 0xf];
			vals[ib * 32 + j + 16] = kvalues_iq4nl[qs[j] >> 4];
		}
		qs += 16;
	}
}
static inline int8x8_t iq3s_flip8(uint8_t q0, uint8_t q1, uint8_t qh, uint8_t sign_byte, int sh0,
								  int sh1) {
	uint32_t   idx0 = q0 | ((qh << sh0) & 256);
	uint32_t   idx1 = q1 | ((qh << sh1) & 256);
	uint32x2_t pk	= {ggml_iq3s_grid[idx0], ggml_iq3s_grid[idx1]};
	uint8x8_t  gr	= vreinterpret_u8_u32(pk);
	uint8x8_t  nm	= vld1_u8(iq3s_sign_pattern[sign_byte]);
	return vreinterpret_s8_u8(vsub_u8(veor_u8(gr, nm), nm));
}
static inline void iq3s_block_decode(const iq3s_block *restrict b, int8x16_t decoded[16]) {
	const uint8_t *restrict qs_p	= b->qs;
	const uint8_t *restrict qh_p	= b->qh;
	const uint8_t *restrict signs_p = b->signs;

	for (int ib32 = 0; ib32 < 8; ib32 += 2) {
		const uint8_t qh0 = qh_p[0];
		const uint8_t qh1 = qh_p[1];
		const int	  vi  = ib32 * 2;

		const int8x8_t f00 = iq3s_flip8(qs_p[0], qs_p[1], qh0, signs_p[0], 8, 7);
		const int8x8_t f01 = iq3s_flip8(qs_p[2], qs_p[3], qh0, signs_p[1], 6, 5);
		const int8x8_t f02 = iq3s_flip8(qs_p[4], qs_p[5], qh0, signs_p[2], 4, 3);
		const int8x8_t f03 = iq3s_flip8(qs_p[6], qs_p[7], qh0, signs_p[3], 2, 1);
		const int8x8_t f10 = iq3s_flip8(qs_p[8], qs_p[9], qh1, signs_p[4], 8, 7);
		const int8x8_t f11 = iq3s_flip8(qs_p[10], qs_p[11], qh1, signs_p[5], 6, 5);
		const int8x8_t f12 = iq3s_flip8(qs_p[12], qs_p[13], qh1, signs_p[6], 4, 3);
		const int8x8_t f13 = iq3s_flip8(qs_p[14], qs_p[15], qh1, signs_p[7], 2, 1);

		decoded[vi + 0] = vcombine_s8(f00, f01);
		decoded[vi + 1] = vcombine_s8(f02, f03);
		decoded[vi + 2] = vcombine_s8(f10, f11);
		decoded[vi + 3] = vcombine_s8(f12, f13);

		qh_p += 2;
		qs_p += 16;
		signs_p += 8;
	}
}
static inline int32_t iq3s_block_dot_decoded(const iq3s_block *restrict b,
											 const int8x16_t decoded[16],
											 const int8_t *restrict q8) {
	int32x4_t acc = vdupq_n_s32(0);
	for (int ib32 = 0; ib32 < 8; ib32 += 2) {
		const int	  g0  = ib32;
		const uint8_t sb  = b->scales[ib32 / 2];
		const int	  sc0 = 1 + (2 * (sb & 0xf));
		const int	  sc1 = 1 + (2 * (sb >> 4));
		const int	  vi  = ib32 * 2;

		const int8x16_t c00 = decoded[vi + 0];
		const int8x16_t c01 = decoded[vi + 1];
		const int8x16_t c10 = decoded[vi + 2];
		const int8x16_t c11 = decoded[vi + 3];

		const int8_t *restrict qg0 = q8 + (g0 * 32);
		const int8_t *restrict qg1 = q8 + ((g0 + 1) * 32);

#if defined(__ARM_FEATURE_DOTPROD)
		int32x4_t a0 =
			vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0)), c01, vld1q_s8(qg0 + 16));
		int32x4_t a1 =
			vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1)), c11, vld1q_s8(qg1 + 16));
#else
		int16x8_t p00 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0)));
		int16x8_t p01 = vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0)));
		int16x8_t p02 = vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0 + 16)));
		int16x8_t p03 = vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0 + 16)));
		int32x4_t a0  = vpaddlq_s16(p00);
		a0			  = vpadalq_s16(a0, p01);
		a0			  = vpadalq_s16(a0, p02);
		a0			  = vpadalq_s16(a0, p03);
		int16x8_t p10 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1)));
		int16x8_t p11 = vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1)));
		int16x8_t p12 = vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1 + 16)));
		int16x8_t p13 = vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1 + 16)));
		int32x4_t a1  = vpaddlq_s16(p10);
		a1			  = vpadalq_s16(a1, p11);
		a1			  = vpadalq_s16(a1, p12);
		a1			  = vpadalq_s16(a1, p13);
#endif
		acc = vmlaq_s32(acc, a0, vdupq_n_s32(sc0));
		acc = vmlaq_s32(acc, a1, vdupq_n_s32(sc1));
	}
	return vaddvq_s32(acc);
}
static inline int32_t iq3s_block_dot(const iq3s_block *restrict b, const int8_t *restrict q8) {
	int8x16_t decoded[16];
	iq3s_block_decode(b, decoded);
	return iq3s_block_dot_decoded(b, decoded, q8);
}
static inline void mk_a64_repacked_sumi(const int8x16_t lo[8], const int8x16_t hi[8], int8x16_t xl,
										int8x16_t xh, int32x4_t corr, int32x4_t *sumi_lo,
										int32x4_t *sumi_hi) {
	int32x4_t acc[8];
	for (int r = 0; r < 8; r++) {
#if defined(__ARM_FEATURE_DOTPROD)
		int32x4_t a = vdotq_s32(vdupq_n_s32(0), lo[r], xl);
		a			= vdotq_s32(a, hi[r], xh);
#else
		int16x8_t p0 = vmull_s8(vget_low_s8(lo[r]), vget_low_s8(xl));
		int16x8_t p1 = vmull_s8(vget_high_s8(lo[r]), vget_high_s8(xl));
		int16x8_t p2 = vmull_s8(vget_low_s8(hi[r]), vget_low_s8(xh));
		int16x8_t p3 = vmull_s8(vget_high_s8(hi[r]), vget_high_s8(xh));
		int32x4_t a	 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
		a			 = vaddq_s32(a, vpaddlq_s16(p2));
		a			 = vaddq_s32(a, vpaddlq_s16(p3));
#endif
		acc[r] = a;
	}
	const int32x4_t sum01 = vpaddq_s32(acc[0], acc[1]);
	const int32x4_t sum23 = vpaddq_s32(acc[2], acc[3]);
	const int32x4_t sum45 = vpaddq_s32(acc[4], acc[5]);
	const int32x4_t sum67 = vpaddq_s32(acc[6], acc[7]);
	*sumi_lo			  = vsubq_s32(vpaddq_s32(sum01, sum23), corr);
	*sumi_hi			  = vsubq_s32(vpaddq_s32(sum45, sum67), corr);
}

static inline void mk_a64_repacked_compute(const int8x16_t lo[8], const int8x16_t hi[8],
										   int8x16_t xl, int8x16_t xh, int32x4_t corr,
										   const uint16_t *d_ptr, float d_xq, float32x4_t *acc_lo,
										   float32x4_t *acc_hi) {
	int32x4_t sumi_lo, sumi_hi;
	mk_a64_repacked_sumi(lo, hi, xl, xh, corr, &sumi_lo, &sumi_hi);

	const uint16x8_t d_w_u16 = vld1q_u16(d_ptr);
	float32x4_t		 d_w0	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_w_u16)));
	float32x4_t		 d_w1	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_w_u16)));
	float32x4_t		 d_xq_v	 = vdupq_n_f32(d_xq);

	*acc_lo = vfmaq_f32(*acc_lo, vmulq_f32(d_w0, d_xq_v), vcvtq_f32_s32(sumi_lo));
	*acc_hi = vfmaq_f32(*acc_hi, vmulq_f32(d_w1, d_xq_v), vcvtq_f32_s32(sumi_hi));
}

static inline void q4_0_repacked_block_accum(const uint8_t *blk, float d_xq, const int8_t *xq_qs,
											 float32x4_t *acc_lo, float32x4_t *acc_hi) {
	const int8x16_t xl	   = vld1q_s8(xq_qs);
	const int8x16_t xh	   = vld1q_s8(xq_qs + 16);
	const uint8_t  *qs_ptr = blk + Q4_0_REPACKED_ROWS * sizeof(uint16_t);
	int8x16_t		lo[8], hi[8];
	for (int r = 0; r < 8; r++) {
		const uint8x16_t q = vld1q_u8(qs_ptr + (size_t)r * 16);
		lo[r]			   = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F)));
		hi[r]			   = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
	}
	const int32_t xsum = (int32_t)vaddlvq_s8(xl) + (int32_t)vaddlvq_s8(xh);
	mk_a64_repacked_compute(lo, hi, xl, xh, vdupq_n_s32(xsum * 8), (const uint16_t *)blk, d_xq,
							acc_lo, acc_hi);
}

static inline void iq4_nl_repacked_block_accum(const uint8_t *blk, const uint8x16_t kvalues_u,
											   const uint8x16_t lo_mask, float d_xq,
											   const int8_t *xq_qs, float32x4_t *acc_lo,
											   float32x4_t *acc_hi) {
	const int8x16_t xl	   = vld1q_s8(xq_qs);
	const int8x16_t xh	   = vld1q_s8(xq_qs + 16);
	const uint8_t  *qs_ptr = blk + IQ4_NL_REPACKED_ROWS * sizeof(uint16_t);
	int8x16_t		lo[8], hi[8];
	for (int r = 0; r < 8; r++) {
		const uint8x16_t q = vld1q_u8(qs_ptr + (size_t)r * 16);
		lo[r]			   = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vandq_u8(q, lo_mask)));
		hi[r]			   = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vshrq_n_u8(q, 4)));
	}
	mk_a64_repacked_compute(lo, hi, xl, xh, vdupq_n_s32(0), (const uint16_t *)blk, d_xq, acc_lo,
							acc_hi);
}
static inline void iq3_s_repacked_unpack_group(const uint8_t *idx_ptr, uint8x16_t tbl,
											   int8x16_t *out0, int8x16_t *out1) {
	uint8x16_t packed = vld1q_u8(idx_ptr);
	uint8x16_t lo	  = vandq_u8(packed, vdupq_n_u8(0x0F));
	uint8x16_t hi	  = vshrq_n_u8(packed, 4);

	*out0 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, lo));
	*out1 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, hi));
}

static inline void mk_a64_q4_0_rp_decode(const uint8_t *blk, int p, int8x16_t avec[4], float *dwa,
										 float *dwb) {
	const uint16_t *d_ptr  = (const uint16_t *)blk;
	const uint8_t  *qs_ptr = blk + Q4_0_REPACKED_ROWS * sizeof(uint16_t);
	*dwa				   = f16_to_f32_fast(d_ptr[2 * p]);
	*dwb				   = f16_to_f32_fast(d_ptr[(2 * p) + 1]);
	const uint8x16_t q0	   = vld1q_u8(qs_ptr + (size_t)(2 * p) * 16);
	const uint8x16_t q1	   = vld1q_u8(qs_ptr + (size_t)((2 * p) + 1) * 16);
	const uint8x16_t l0	   = vandq_u8(q0, vdupq_n_u8(0x0F));
	const uint8x16_t h0	   = vshrq_n_u8(q0, 4);
	const uint8x16_t l1	   = vandq_u8(q1, vdupq_n_u8(0x0F));
	const uint8x16_t h1	   = vshrq_n_u8(q1, 4);
	avec[0] =
		vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(l0), vget_low_u8(l1))), vdupq_n_s8(8));
	avec[1] = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(l0), vget_high_u8(l1))),
					   vdupq_n_s8(8));
	avec[2] =
		vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(h0), vget_low_u8(h1))), vdupq_n_s8(8));
	avec[3] = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(h0), vget_high_u8(h1))),
					   vdupq_n_s8(8));
}

static inline void mk_a64_iq4_nl_rp_decode(const uint8_t *blk, int p, int8x16_t avec[4], float *dwa,
										   float *dwb) {
	const uint16_t	*d_ptr	   = (const uint16_t *)blk;
	const uint8_t	*qs_ptr	   = blk + IQ4_NL_REPACKED_ROWS * sizeof(uint16_t);
	const uint8x16_t kvalues_u = vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t lo_mask   = vdupq_n_u8(0x0F);
	*dwa					   = f16_to_f32_fast(d_ptr[2 * p]);
	*dwb					   = f16_to_f32_fast(d_ptr[(2 * p) + 1]);
	const uint8x16_t q0		   = vld1q_u8(qs_ptr + (size_t)(2 * p) * 16);
	const uint8x16_t q1		   = vld1q_u8(qs_ptr + (size_t)((2 * p) + 1) * 16);
	const uint8x16_t l0		   = vandq_u8(q0, lo_mask);
	const uint8x16_t h0		   = vshrq_n_u8(q0, 4);
	const uint8x16_t l1		   = vandq_u8(q1, lo_mask);
	const uint8x16_t h1		   = vshrq_n_u8(q1, 4);
	avec[0] =
		vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vcombine_u8(vget_low_u8(l0), vget_low_u8(l1))));
	avec[1] =
		vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vcombine_u8(vget_high_u8(l0), vget_high_u8(l1))));
	avec[2] =
		vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vcombine_u8(vget_low_u8(h0), vget_low_u8(h1))));
	avec[3] =
		vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vcombine_u8(vget_high_u8(h0), vget_high_u8(h1))));
}

static inline float mk_a64_quantize_q8_32(const float *x, int8_t *qs, int32_t *sum_out,
										  uint16_t *d16_out) {
	float32x4_t v0	 = vld1q_f32(x);
	float32x4_t v1	 = vld1q_f32(x + 4);
	float32x4_t v2	 = vld1q_f32(x + 8);
	float32x4_t v3	 = vld1q_f32(x + 12);
	float32x4_t v4	 = vld1q_f32(x + 16);
	float32x4_t v5	 = vld1q_f32(x + 20);
	float32x4_t v6	 = vld1q_f32(x + 24);
	float32x4_t v7	 = vld1q_f32(x + 28);
	float32x4_t a0	 = vabsq_f32(v0);
	float32x4_t a1	 = vabsq_f32(v1);
	float32x4_t a2	 = vabsq_f32(v2);
	float32x4_t a3	 = vabsq_f32(v3);
	float32x4_t a4	 = vabsq_f32(v4);
	float32x4_t a5	 = vabsq_f32(v5);
	float32x4_t a6	 = vabsq_f32(v6);
	float32x4_t a7	 = vabsq_f32(v7);
	float32x4_t m0	 = vmaxq_f32(a0, a1);
	float32x4_t m1	 = vmaxq_f32(a2, a3);
	float32x4_t m2	 = vmaxq_f32(a4, a5);
	float32x4_t m3	 = vmaxq_f32(a6, a7);
	m0				 = vmaxq_f32(m0, m1);
	m2				 = vmaxq_f32(m2, m3);
	m0				 = vmaxq_f32(m0, m2);
	const float amax = vmaxvq_f32(m0);

	const float d  = amax / 127.0f;
	const float id = d > 0 ? 1.0f / d : 0.0f;
	*d16_out	   = f32_to_f16(d);

	const float32x4_t id_v = vdupq_n_f32(id);
	int32x4_t		  q0   = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v0, id_v)));
	int32x4_t		  q1   = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v1, id_v)));
	int32x4_t		  q2   = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v2, id_v)));
	int32x4_t		  q3   = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v3, id_v)));
	int32x4_t		  q4   = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v4, id_v)));
	int32x4_t		  q5   = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v5, id_v)));
	int32x4_t		  q6   = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v6, id_v)));
	int32x4_t		  q7   = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v7, id_v)));
	int32x4_t		  s0   = vaddq_s32(q0, q1);
	int32x4_t		  s1   = vaddq_s32(q2, q3);
	int32x4_t		  s2   = vaddq_s32(q4, q5);
	int32x4_t		  s3   = vaddq_s32(q6, q7);
	s0					   = vaddq_s32(s0, s1);
	s2					   = vaddq_s32(s2, s3);
	s0					   = vaddq_s32(s0, s2);
	*sum_out			   = vaddvq_s32(s0);

	int16x4_t n0 = vqmovn_s32(q0);
	int16x4_t n1 = vqmovn_s32(q1);
	int16x4_t n2 = vqmovn_s32(q2);
	int16x4_t n3 = vqmovn_s32(q3);
	int16x4_t n4 = vqmovn_s32(q4);
	int16x4_t n5 = vqmovn_s32(q5);
	int16x4_t n6 = vqmovn_s32(q6);
	int16x4_t n7 = vqmovn_s32(q7);
	int16x8_t h0 = vcombine_s16(n0, n1);
	int16x8_t h1 = vcombine_s16(n2, n3);
	int16x8_t h2 = vcombine_s16(n4, n5);
	int16x8_t h3 = vcombine_s16(n6, n7);
	int8x8_t  b0 = vqmovn_s16(h0);
	int8x8_t  b1 = vqmovn_s16(h1);
	int8x8_t  b2 = vqmovn_s16(h2);
	int8x8_t  b3 = vqmovn_s16(h3);
	vst1_s8(qs, b0);
	vst1_s8(qs + 8, b1);
	vst1_s8(qs + 16, b2);
	vst1_s8(qs + 24, b3);
	return d;
}

#define MK_A64_MATMUL_F16(fn, load4_fn, finish_fn, dot_fn)                                         \
	void fn(const void *restrict w, const float *restrict x, float *restrict y, int n, int k) {    \
		const uint16_t *Wb = w;                                                                    \
		const int		mr = 4;                                                                    \
		int				i  = 0;                                                                    \
		for (; i + mr <= n; i += mr) {                                                             \
			const uint16_t *rows[4];                                                               \
			float32x4_t		acc[4][4];                                                             \
			for (int r = 0; r < mr; r++) {                                                         \
				rows[r] = Wb + ((size_t)(i + r) * k);                                              \
				for (int q = 0; q < 4; q++)                                                        \
					acc[r][q] = vdupq_n_f32(0.0f);                                                 \
			}                                                                                      \
			int j = 0;                                                                             \
			for (; j + 16 <= k; j += 16) {                                                         \
				PREFETCH(x + j + PF_ELEMS(64));                                                    \
				float32x4_t xv[4];                                                                 \
				for (int q = 0; q < 4; q++)                                                        \
					xv[q] = vld1q_f32(x + j + (4 * q));                                            \
				for (int r = 0; r < mr; r++) {                                                     \
					PREFETCH(rows[r] + j + PF_ELEMS(64));                                          \
					for (int q = 0; q < 4; q++)                                                    \
						acc[r][q] = vfmaq_f32(acc[r][q], xv[q], load4_fn(rows[r] + j + (4 * q)));  \
				}                                                                                  \
			}                                                                                      \
			for (int r = 0; r < mr; r++)                                                           \
				y[i + r] =                                                                         \
					finish_fn(acc[r][0], acc[r][1], acc[r][2], acc[r][3], x, rows[r], j, k);       \
		}                                                                                          \
		for (; i < n; i++)                                                                         \
			y[i] = dot_fn(x, Wb + ((size_t)i * k), k);                                             \
	}

#define MK_A64_REPACKED_I8MM(fn, row_fn, group_bytes, decode_fn)                                    \
	static void fn(const void *w, const q8_0_block *restrict xq, size_t xq_row_stride_blocks,       \
				   float *restrict y, int y_row_stride, int n, int k, int m) {                      \
		const int	   blocks_per_row = k / 32;                                                     \
		const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(q4_0_block);                \
		const uint8_t *Wb			  = w;                                                          \
		int			   i			  = 0;                                                          \
		for (; i + MR <= n; i += MR) {                                                              \
			const uint8_t *group = Wb + ((size_t)i * row_stride);                                   \
			int			   t	 = 0;                                                               \
			for (; t + I8MM_NR <= m; t += I8MM_NR) {                                                \
				float32x4_t facc[MR / 2][I8MM_NR / 2];                                              \
				for (int p = 0; p < MR / 2; p++)                                                    \
					for (int c = 0; c < I8MM_NR / 2; c++)                                           \
						facc[p][c] = vdupq_n_f32(0.0f);                                             \
				const q8_0_block *xrow[I8MM_NR];                                                    \
				for (int c = 0; c < I8MM_NR; c++)                                                   \
					xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);                        \
				for (int bi = 0; bi < blocks_per_row; bi++) {                                       \
					const uint8_t *blk = group + (size_t)bi * (group_bytes);                        \
					PREFETCH(blk + (group_bytes));                                                  \
					int8x16_t xq_lo[I8MM_NR];                                                       \
					int8x16_t xq_hi[I8MM_NR];                                                       \
					float	  xd[I8MM_NR];                                                          \
					for (int c = 0; c < I8MM_NR; c++) {                                             \
						xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);                                  \
						xq_lo[c] = vld1q_s8(xrow[c][bi].qs);                                        \
						xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);                                   \
					}                                                                               \
					int8x16_t bvec[I8MM_NR / 2][4];                                                 \
					for (int cp = 0; cp < I8MM_NR / 2; cp++) {                                      \
						const int ca = 2 * cp;                                                      \
						const int cb = 2 * cp + 1;                                                  \
						bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb])); \
						bvec[cp][1] =                                                               \
							vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));          \
						bvec[cp][2] = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));  \
						bvec[cp][3] =                                                               \
							vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));          \
					}                                                                               \
					for (int p = 0; p < MR / 2; p++) {                                              \
						int8x16_t avec[4];                                                          \
						float	  dwa, dwb;                                                         \
						decode_fn(blk, p, avec, &dwa, &dwb);                                        \
						const float32x4_t srow = vcombine_f32(vdup_n_f32(dwa), vdup_n_f32(dwb));    \
						for (int cp = 0; cp < I8MM_NR / 2; cp++) {                                  \
							const float32x4_t dcol = vzip1q_f32(vdupq_n_f32(xd[2 * cp]),            \
																vdupq_n_f32(xd[(2 * cp) + 1]));     \
							int32x4_t		  s	   = vdupq_n_s32(0);                                \
							s					   = vmmlaq_s32(s, avec[0], bvec[cp][0]);           \
							s					   = vmmlaq_s32(s, avec[1], bvec[cp][1]);           \
							s					   = vmmlaq_s32(s, avec[2], bvec[cp][2]);           \
							s					   = vmmlaq_s32(s, avec[3], bvec[cp][3]);           \
							facc[p][cp] =                                                           \
								vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));    \
						}                                                                           \
					}                                                                               \
				}                                                                                   \
				for (int p = 0; p < MR / 2; p++) {                                                  \
					for (int cp = 0; cp < I8MM_NR / 2; cp++) {                                      \
						float tmp[4];                                                               \
						vst1q_f32(tmp, facc[p][cp]);                                                \
						y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];    \
						y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];    \
						y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];    \
						y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];    \
					}                                                                               \
				}                                                                                   \
			}                                                                                       \
			for (; t < m; t++)                                                                      \
				row_fn(group, xq + ((size_t)t * xq_row_stride_blocks),                              \
					   y + ((size_t)t * y_row_stride) + i, MR, k);                                  \
		}                                                                                           \
	}

#define MK_A64_ROW_DRIVER(fn, wblk_t, xqblk_t, elems, acc_fn)                                      \
	static void fn(const void *w, const xqblk_t *restrict xq, float *restrict y, int n, int k) {   \
		const int	   bpr		  = k / (elems);                                                   \
		const size_t   row_stride = (size_t)bpr * sizeof(wblk_t);                                  \
		const uint8_t *wb		  = w;                                                             \
		int			   i		  = 0;                                                             \
		for (; i + MR <= n; i += MR) {                                                             \
			float acc[MR];                                                                         \
			for (int r = 0; r < MR; r++)                                                           \
				acc[r] = 0.0f;                                                                     \
			for (int bi = 0; bi < bpr; bi++) {                                                     \
				for (int r = 0; r < MR; r++) {                                                     \
					const void *row =                                                              \
						wb + ((size_t)(i + r) * row_stride) + ((size_t)bi * sizeof(wblk_t));       \
					acc[r] = acc_fn(row, &xq[bi], acc[r]);                                         \
				}                                                                                  \
			}                                                                                      \
			for (int r = 0; r < MR; r++)                                                           \
				y[i + r] = acc[r];                                                                 \
		}                                                                                          \
		for (; i < n; i++) {                                                                       \
			float acc = 0.0f;                                                                      \
			for (int bi = 0; bi < bpr; bi++) {                                                     \
				const void *row = wb + ((size_t)i * row_stride) + ((size_t)bi * sizeof(wblk_t));   \
				acc				= acc_fn(row, &xq[bi], acc);                                       \
			}                                                                                      \
			y[i] = acc;                                                                            \
		}                                                                                          \
	}

#endif
