/* CUDA backend host-side weight relayouts (see cuda_repack.h).
 *
 * Deliberately a plain C translation unit: no CUDA runtime headers, so it is
 * compiled by the ordinary host compiler rule and can be called from shared
 * code. Keeps CPU/Vulkan backends free of QM knowledge. */
#include "backend/cuda/cuda_repack.h"

#include <stddef.h>
#include <string.h>

#include "common.h"
#include "backend/cpu/scalar/quants.h"

#define CUDA_QM_GROUP 32

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
			float			  d	 = f16_to_f32(sb->d);
			db->d				 = f32_to_f16(d * 0.5f);
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
