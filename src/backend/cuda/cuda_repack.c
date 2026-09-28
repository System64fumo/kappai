/* CUDA backend host-side weight relayouts (see cuda_repack.h).
 *
 * Deliberately a plain C translation unit: no CUDA runtime headers, so it is
 * compiled by the ordinary host compiler rule and can be called from shared
 * code. Keeps CPU/Vulkan backends free of QM knowledge. */
#include "backend/cuda/cuda_repack.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "common.h"
#include "backend/cpu/scalar/quants.h"

#define CUDA_QM_GROUP 32

/* f16 <-> f32 helpers.
 *
 * This translation unit is linked into the engine (model.c calls it), while
 * the scalar backend's f16_to_f32/f32_to_f16 now live in a dlopen'd
 * libkappai_cpu_scalar.so, so the engine cannot link against them. Keep
 * bit-exact local copies instead: f32_to_f16 mirrors the scalar reference
 * algorithm verbatim, and qm_f16_to_f32 is the exact IEEE half expansion
 * (equal to the reference lookup table for all 65536 encodings). */
static inline uint32_t qm_f32_to_bits(float f) {
	union {
		float	 f;
		uint32_t i;
	} v;
	v.f = f;
	return v.i;
}

static inline float qm_f32_from_bits(uint32_t b) {
	union {
		uint32_t i;
		float	 f;
	} v;
	v.i = b;
	return v.f;
}

static inline float qm_f16_to_f32(uint16_t h) {
	const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
	uint32_t	   exp	 = ((uint32_t)h >> 10) & 0x1Fu;
	uint32_t	   mant = (uint32_t)h & 0x03FFu;
	uint32_t	   u;
	if (exp == 0) {
		if (mant == 0) {
			u = sign;
		} else {
			/* subnormal half: renormalize into the float normal range */
			exp = 1;
			while ((mant & 0x0400u) == 0) {
				mant <<= 1;
				exp--;
			}
			mant &= 0x03FFu;
			u = sign | ((exp + 112) << 23) | (mant << 13);
		}
	} else if (exp == 0x1F) {
		u = sign | 0x7F800000u | (mant << 13);
	} else {
		u = sign | ((exp + 112) << 23) | (mant << 13);
	}
	return qm_f32_from_bits(u);
}

static inline uint16_t qm_f32_to_f16(float f) {
	const float	base	= (fabsf(f) * 0x1.0p+112f) * 0x1.0p-110f;
	const uint32_t w		= qm_f32_to_bits(f);
	const uint32_t shl1_w = w + w;
	const uint32_t sign	= w & 0x80000000u;
	uint32_t		bias	= shl1_w & 0xFF000000u;
	if (bias < 0x71000000u)
		bias = 0x71000000u;
	const float		rounded = qm_f32_from_bits((bias >> 1) + 0x07800000u) + base;
	const uint32_t bits		= qm_f32_to_bits(rounded);
	const uint32_t exp_bits  = (bits >> 13) & 0x7C00u;
	const uint32_t mant_bits = bits & 0x0FFFu;
	const uint32_t nonsign   = exp_bits + mant_bits;
	return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

/* ---- Quad-major Q8_0 (34B/32e blocks) ------------------------------- */
static void repack_q8_0_qm_group(const q8_0_block *s, uint8_t *d, int G) {
	uint8_t *scales = d;
	uint8_t *quads	= d + (size_t)2 * G;
	for (int r = 0; r < G; r++) {
		uint16_t dv;
		memcpy(&dv, &s[r].d, 2);
		memcpy(scales + (size_t)2 * r, &dv, 2);
	}
	for (int t = 0; t < 8; t++)
		for (int r = 0; r < G; r++)
			memcpy(quads + ((size_t)t * G + r) * 4, s[r].qs + (size_t)t * 4, 4);
}

void cuda_repack_q8_0_qm_rows(const void *src, void *dst, int row_begin, int row_end, int k) {
	const int	   nb	  = k / 32;
	const size_t   stride = (size_t)nb * 34;
	const uint8_t *sp	  = src;
	uint8_t		  *dp	  = dst;
	for (int r = row_begin; r < row_end; r++) {
		const q8_0_block *srow = (const q8_0_block *)(sp + (size_t)r * stride);
		uint8_t			 *drow = dp + (size_t)r * stride;
		int b = 0;
		for (; b + CUDA_QM_GROUP <= nb; b += CUDA_QM_GROUP)
			repack_q8_0_qm_group(srow + b, drow + (size_t)b * 34, CUDA_QM_GROUP);
		if (b < nb)
			repack_q8_0_qm_group(srow + b, drow + (size_t)b * 34, nb - b);
	}
}

/* ---- Quad-major Q4_0 (18B/32e blocks) ------------------------------- */
static void repack_q4_0_qm_group(const q4_0_block *s, uint8_t *d, int G) {
	uint8_t *scales = d;
	uint8_t *planes = d + (size_t)2 * G;
	for (int r = 0; r < G; r++) {
		uint16_t dv;
		memcpy(&dv, &s[r].d, 2);
		memcpy(scales + (size_t)2 * r, &dv, 2);
	}
	for (int j = 0; j < 16; j++)
		for (int r = 0; r < G; r++)
			planes[(size_t)j * G + r] = s[r].qs[j];
}

void cuda_repack_q4_0_qm_rows(const void *src, void *dst, int row_begin, int row_end, int k) {
	const int	   nb	  = k / 32;
	const size_t   stride = (size_t)nb * 18;
	const uint8_t *sp	  = src;
	uint8_t		  *dp	  = dst;
	for (int r = row_begin; r < row_end; r++) {
		const q4_0_block *srow = (const q4_0_block *)(sp + (size_t)r * stride);
		uint8_t			 *drow = dp + (size_t)r * stride;
		int b = 0;
		for (; b + CUDA_QM_GROUP <= nb; b += CUDA_QM_GROUP)
			repack_q4_0_qm_group(srow + b, drow + (size_t)b * 18, CUDA_QM_GROUP);
		if (b < nb)
			repack_q4_0_qm_group(srow + b, drow + (size_t)b * 18, nb - b);
	}
}

/* ---- Lossless Q4_0 -> Q8_0 promotion -------------------------------- */
void *cuda_convert_q4_0_to_q8_0(const void *src, int n_rows, int k) {
	const int	 nb	 = k / 32;
	q8_0_block	*dst = xmalloc_aligned((size_t)n_rows * nb * sizeof(q8_0_block), 64);
	if (!dst)
		return NULL;
	const q4_0_block *s = src;
	for (int r = 0; r < n_rows; r++) {
		for (int b = 0; b < nb; b++) {
			const q4_0_block *sb = s + (size_t)r * nb + b;
			q8_0_block		 *db = dst + (size_t)r * nb + b;
			float			  d	 = qm_f16_to_f32(sb->d);
			db->d				 = qm_f32_to_f16(d * 0.5f);
			/* Nibble order: low -> elems 0-15, high -> elems 16-31. */
			for (int j = 0; j < 16; j++) {
				uint8_t qb	   = sb->qs[j];
				db->qs[j]	   = (int8_t)(((qb & 0xF) << 1) - 16);
				db->qs[j + 16] = (int8_t)(((qb >> 4) << 1) - 16);
			}
		}
	}
	return dst;
}
