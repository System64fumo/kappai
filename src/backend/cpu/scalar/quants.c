#include "backend/cpu/quants.h"
#include "threadpool.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MATMUL_MR 8

static inline float q4_0_dot(const q4_0_block *blk, const int8_t *xq8, float d_xq, float acc) {
	const uint8_t *qs	 = blk->qs;
	int			   sumi0 = 0;
	int			   sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		sumi0 += ((int)(qs[j] & 0xF) - 8) * (int)xq8[j];
		sumi1 += ((int)(qs[j] >> 4) - 8) * (int)xq8[j + 16];
	}
	return fmaf(f16_to_f32(blk->d) * d_xq, (float)(sumi0 + sumi1), acc);
}

static inline float q8_0_dot(const q8_0_block *blk, const int8_t *xq8, float d_xq, float acc) {
	const int8_t *qs   = blk->qs;
	int			  sumi = 0;
	for (int j = 0; j < 32; j++)
		sumi += (int)qs[j] * (int)xq8[j];
	return fmaf(f16_to_f32(blk->d) * d_xq, (float)sumi, acc);
}

static inline float iq4_nl_dot(const iq4_nl_block *blk, const int8_t *xq8, float d_xq, float acc) {
	const uint8_t *qs	 = blk->qs;
	int			   sumi0 = 0;
	int			   sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		sumi0 += (int)xq8[j] * (int)kvalues_iq4nl[qs[j] & 0xF];
		sumi1 += (int)xq8[j + 16] * (int)kvalues_iq4nl[qs[j] >> 4];
	}
	const float scaled = f16_to_f32(blk->d) * d_xq;
	return fmaf(scaled, (float)(sumi0 + sumi1), acc);
}

static inline float q4_1_dot(const q4_1_block *blk, const int8_t *xq8, float d_xq, float s_xq,
							 float acc) {
	const uint8_t *qs	 = blk->qs;
	int			   sumi0 = 0;
	int			   sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		sumi0 += (int)(qs[j] & 0xF) * (int)xq8[j];
		sumi1 += (int)(qs[j] >> 4) * (int)xq8[j + 16];
	}
	return fmaf(f16_to_f32(blk->d) * d_xq, (float)(sumi0 + sumi1), f16_to_f32(blk->m) * s_xq + acc);
}

static inline float q5_0_dot(const q5_0_block *blk, const int8_t *xq8, float d_xq, float acc) {
	const uint8_t *qs = blk->qs;
	uint32_t	   qh;
	memcpy(&qh, blk->qh, 4);
	int sumi0 = 0;
	int sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		int lo = (int)(qs[j] & 0xF) | (int)((qh >> j) & 1) << 4;
		int hi = (int)(qs[j] >> 4) | (int)((qh >> (j + 16)) & 1) << 4;
		sumi0 += (lo - 16) * (int)xq8[j];
		sumi1 += (hi - 16) * (int)xq8[j + 16];
	}
	return fmaf(f16_to_f32(blk->d) * d_xq, (float)(sumi0 + sumi1), acc);
}

static inline float q5_1_dot(const q5_1_block *blk, const int8_t *xq8, float d_xq, float s_xq,
							 float acc) {
	const uint8_t *qs = blk->qs;
	uint32_t	   qh;
	memcpy(&qh, blk->qh, 4);
	int sumi0 = 0;
	int sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		int lo = (int)(qs[j] & 0xF) | (int)((qh >> j) & 1) << 4;
		int hi = (int)(qs[j] >> 4) | (int)((qh >> (j + 16)) & 1) << 4;
		sumi0 += lo * (int)xq8[j];
		sumi1 += hi * (int)xq8[j + 16];
	}
	return fmaf(f16_to_f32(blk->d) * d_xq, (float)(sumi0 + sumi1), f16_to_f32(blk->m) * s_xq + acc);
}

static inline float q6_k_dot(const q6_k_block *blk, const int8_t *xq8, float d_xq, int8_t *q_unpack,
							 float acc) {
	const uint8_t *ql = blk->ql;
	const uint8_t *qh = blk->qh;
	int8_t		  *a  = q_unpack;
	for (int n_iter = 0; n_iter < 2; n_iter++) {
		for (int l = 0; l < 32; l++) {
			a[l]	  = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
			a[l + 32] = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
			a[l + 64] = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
			a[l + 96] = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
		}
		a += 128;
		ql += 64;
		qh += 32;
	}
	a				   = q_unpack;
	const int8_t *q8   = xq8;
	const int8_t *sc   = blk->scales;
	int32_t		  acc0 = 0;
	int32_t		  acc1 = 0;
	for (int j = 0; j < 16; j++) {
		int		s  = (int)sc[j];
		int32_t l0 = 0;
		int32_t l1 = 0;
		for (int l = 0; l < 8; l++) {
			l0 += (int)q8[l] * (int)a[l];
			l1 += (int)q8[l + 8] * (int)a[l + 8];
		}
		acc0 += s * l0;
		acc1 += s * l1;
		q8 += 16;
		a += 16;
	}
	return fmaf(f16_to_f32(blk->d) * d_xq, (float)(acc0 + acc1), acc);
}

static inline float q4_k_dot(const q4_k_block *blk, const int8_t *xq8, const int16_t *bs, float xd,
							 float acc) {
	const float	   d	  = f16_to_f32(blk->d);
	const float	   dmin	  = f16_to_f32(blk->dmin);
	const uint8_t *qbytes = blk->qs;
	const uint8_t *sc	  = blk->scales;
	int32_t		   sumi	  = 0;
	int32_t		   summ	  = 0;
	int			   is	  = 0;
	int			   ib	  = 0;
	uint8_t		   scu8;
	uint8_t		   mu8;
	for (int g = 0; g < 4; g++) {
		get_scale_min_k4(is + 0, sc, &scu8, &mu8);
		int s0 = scu8;
		int m0 = mu8;
		get_scale_min_k4(is + 1, sc, &scu8, &mu8);
		int			   s1	= scu8;
		int			   m1	= mu8;
		const uint8_t *qg	= qbytes + (ptrdiff_t)(g * 32);
		const int8_t  *xq0	= xq8 + (ptrdiff_t)(g * 64);
		const int8_t  *xq1	= xq8 + (ptrdiff_t)(g * 64) + 32;
		int32_t		   dot0 = 0;
		int32_t		   dot1 = 0;
		for (int l = 0; l < 32; l++) {
			uint8_t byte = qg[l];
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
	return fmaf(xd, (d * (float)sumi) - (dmin * (float)summ), acc);
}

static inline float q5_k_dot(const q5_k_block *blk, const int8_t *xq8, const int16_t *bs, float xd,
							 float acc) {
	const float	   d	  = f16_to_f32(blk->d);
	const float	   dmin	  = f16_to_f32(blk->dmin);
	const uint8_t *qbytes = blk->qs;
	const uint8_t *qh	  = blk->qh;
	const uint8_t *sc	  = blk->scales;
	int32_t		   sumi	  = 0;
	int32_t		   summ	  = 0;
	int			   is	  = 0;
	int			   ib	  = 0;
	uint8_t		   scu8;
	uint8_t		   mu8;
	uint8_t		   u1 = 1;
	uint8_t		   u2 = 2;
	for (int g = 0; g < 4; g++) {
		get_scale_min_k4(is + 0, sc, &scu8, &mu8);
		int s0 = scu8;
		int m0 = mu8;
		get_scale_min_k4(is + 1, sc, &scu8, &mu8);
		int			   s1	= scu8;
		int			   m1	= mu8;
		const uint8_t *qsg	= qbytes + (ptrdiff_t)(g * 32);
		const int8_t  *xq0	= xq8 + (ptrdiff_t)(g * 64);
		const int8_t  *xq1	= xq8 + (ptrdiff_t)(g * 64) + 32;
		int32_t		   dot0 = 0;
		int32_t		   dot1 = 0;
		for (int l = 0; l < 32; l++) {
			uint8_t byte = qsg[l];
			int		hi0	 = (qh[l] & u1) ? 16 : 0;
			int		hi1	 = (qh[l] & u2) ? 16 : 0;
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
	return fmaf(xd, (d * (float)sumi) - (dmin * (float)summ), acc);
}

#define MATMUL_K_BLOCK 64

void dequant_q4_0_row(const void *blocks, size_t n_blocks, float *dst) {
	const q4_0_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float d = f16_to_f32(b[bi].d);
		for (int j = 0; j < 16; j++) {
			uint8_t byte			= b[bi].qs[j];
			dst[(bi * 32) + j]		= (float)((int8_t)(byte & 0xF) - 8) * d;
			dst[(bi * 32) + j + 16] = (float)((int8_t)(byte >> 4) - 8) * d;
		}
	}
}

void dequant_q4_1_row(const void *blocks, size_t n_blocks, float *dst) {
	const q4_1_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float d = f16_to_f32(b[bi].d);
		float m = f16_to_f32(b[bi].m);
		for (int j = 0; j < 16; j++) {
			uint8_t byte			= b[bi].qs[j];
			dst[(bi * 32) + j]		= ((float)(byte & 0xF) * d) + m;
			dst[(bi * 32) + j + 16] = ((float)(byte >> 4) * d) + m;
		}
	}
}

void dequant_q5_0_row(const void *blocks, size_t n_blocks, float *dst) {
	const q5_0_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float	 d = f16_to_f32(b[bi].d);
		uint32_t qh;
		memcpy(&qh, b[bi].qh, 4);
		for (int j = 0; j < 16; j++) {
			uint8_t byte			= b[bi].qs[j];
			int		lo				= (byte & 0xF) | ((int)((qh >> j) & 1) << 4);
			int		hi				= (byte >> 4) | ((int)((qh >> (j + 16)) & 1) << 4);
			dst[(bi * 32) + j]		= (float)(lo - 16) * d;
			dst[(bi * 32) + j + 16] = (float)(hi - 16) * d;
		}
	}
}

void dequant_q5_1_row(const void *blocks, size_t n_blocks, float *dst) {
	const q5_1_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float	 d = f16_to_f32(b[bi].d);
		float	 m = f16_to_f32(b[bi].m);
		uint32_t qh;
		memcpy(&qh, b[bi].qh, 4);
		for (int j = 0; j < 16; j++) {
			uint8_t byte			= b[bi].qs[j];
			int		lo				= (byte & 0xF) | ((int)((qh >> j) & 1) << 4);
			int		hi				= (byte >> 4) | ((int)((qh >> (j + 16)) & 1) << 4);
			dst[(bi * 32) + j]		= ((float)lo * d) + m;
			dst[(bi * 32) + j + 16] = ((float)hi * d) + m;
		}
	}
}

void dequant_q8_0_row(const void *blocks, size_t n_blocks, float *dst) {
	const q8_0_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float d = f16_to_f32(b[bi].d);
		for (int j = 0; j < 32; j++)
			dst[(bi * 32) + j] = (float)b[bi].qs[j] * d;
	}
}

void dequant_iq3_s_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq3s_block *b = blocks;

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const float	   d	 = f16_to_f32(b[bi].d);
		const uint8_t *qs	 = b[bi].qs;
		const uint8_t *qh	 = b[bi].qh;
		const uint8_t *signs = b[bi].signs;
		float		  *y	 = dst + (bi * 256);

		for (int ib32 = 0; ib32 < 8; ib32 += 2) {
			const uint8_t scales_byte = b[bi].scales[ib32 / 2];
			const float	  db1		  = d * (1.0f + (2.0f * (scales_byte & 0xf)));
			const float	  db2		  = d * (1.0f + (2.0f * (scales_byte >> 4)));

			for (int l = 0; l < 4; l++) {
				const uint8_t *grid1 =
					(const uint8_t
						 *)(&ggml_iq3s_grid[qs[(2 * l) + 0] | ((qh[0] << (8 - (2 * l))) & 256)]);
				const uint8_t *grid2 =
					(const uint8_t
						 *)(&ggml_iq3s_grid[qs[(2 * l) + 1] | ((qh[0] << (7 - (2 * l))) & 256)]);

				const uint8_t sign_byte = signs[l];
				const float	  s0		= (sign_byte & kmask_iq2xs[0]) ? -db1 : db1;
				const float	  s1		= (sign_byte & kmask_iq2xs[1]) ? -db1 : db1;
				const float	  s2		= (sign_byte & kmask_iq2xs[2]) ? -db1 : db1;
				const float	  s3		= (sign_byte & kmask_iq2xs[3]) ? -db1 : db1;
				const float	  s4		= (sign_byte & kmask_iq2xs[4]) ? -db1 : db1;
				const float	  s5		= (sign_byte & kmask_iq2xs[5]) ? -db1 : db1;
				const float	  s6		= (sign_byte & kmask_iq2xs[6]) ? -db1 : db1;
				const float	  s7		= (sign_byte & kmask_iq2xs[7]) ? -db1 : db1;

				y[0] = s0 * grid1[0];
				y[1] = s1 * grid1[1];
				y[2] = s2 * grid1[2];
				y[3] = s3 * grid1[3];
				y[4] = s4 * grid2[0];
				y[5] = s5 * grid2[1];
				y[6] = s6 * grid2[2];
				y[7] = s7 * grid2[3];

				y += 8;
			}
			qs += 8;
			signs += 4;

			for (int l = 0; l < 4; l++) {
				const uint8_t *grid1 =
					(const uint8_t
						 *)(&ggml_iq3s_grid[qs[(2 * l) + 0] | ((qh[1] << (8 - (2 * l))) & 256)]);
				const uint8_t *grid2 =
					(const uint8_t
						 *)(&ggml_iq3s_grid[qs[(2 * l) + 1] | ((qh[1] << (7 - (2 * l))) & 256)]);

				const uint8_t sign_byte = signs[l];
				const float	  s0		= (sign_byte & kmask_iq2xs[0]) ? -db2 : db2;
				const float	  s1		= (sign_byte & kmask_iq2xs[1]) ? -db2 : db2;
				const float	  s2		= (sign_byte & kmask_iq2xs[2]) ? -db2 : db2;
				const float	  s3		= (sign_byte & kmask_iq2xs[3]) ? -db2 : db2;
				const float	  s4		= (sign_byte & kmask_iq2xs[4]) ? -db2 : db2;
				const float	  s5		= (sign_byte & kmask_iq2xs[5]) ? -db2 : db2;
				const float	  s6		= (sign_byte & kmask_iq2xs[6]) ? -db2 : db2;
				const float	  s7		= (sign_byte & kmask_iq2xs[7]) ? -db2 : db2;

				y[0] = s0 * grid1[0];
				y[1] = s1 * grid1[1];
				y[2] = s2 * grid1[2];
				y[3] = s3 * grid1[3];
				y[4] = s4 * grid2[0];
				y[5] = s5 * grid2[1];
				y[6] = s6 * grid2[2];
				y[7] = s7 * grid2[3];

				y += 8;
			}
			qh += 2;
			qs += 8;
			signs += 4;
		}
	}
}

void dequant_iq2_xxs_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq2_xxs_block *b = blocks;

	uint32_t	   aux32[2];
	const uint8_t *aux8 = (const uint8_t *)aux32;

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const float d = f16_to_f32(b[bi].d);
		float	   *y = dst + (bi * 256);

		for (int ib32 = 0; ib32 < 8; ++ib32) {
			memcpy(aux32, b[bi].qs + 4 * ib32, 8);
			const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
			for (int l = 0; l < 4; ++l) {
				const uint8_t *grid	 = (const uint8_t *)&ggml_iq2xxs_grid[aux8[l]];
				const uint8_t  signs = ggml_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
				for (int j = 0; j < 8; ++j)
					y[j] = db * (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
				y += 8;
			}
		}
	}
}

void dequant_iq3_xxs_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq3_xxs_block *b = blocks;

	uint32_t aux32;

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const float	   d				= f16_to_f32(b[bi].d);
		const uint8_t *qs				= b[bi].qs;
		const uint8_t *scales_and_signs = qs + 64;
		float		  *y				= dst + (bi * 256);

		for (int ib32 = 0; ib32 < 8; ++ib32) {
			memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
			const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
			for (int l = 0; l < 4; ++l) {
				const uint8_t  signs = ggml_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
				const uint8_t *grid1 = (const uint8_t *)&ggml_iq3xxs_grid[qs[2 * l + 0]];
				const uint8_t *grid2 = (const uint8_t *)&ggml_iq3xxs_grid[qs[2 * l + 1]];
				for (int j = 0; j < 4; ++j) {
					y[j + 0] = db * (float)grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
					y[j + 4] = db * (float)grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
				}
				y += 8;
			}
			qs += 8;
		}
	}
}

void dequant_iq2_s_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq2_s_block *b = blocks;

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const float	   d	 = f16_to_f32(b[bi].d);
		const uint8_t *qs	 = b[bi].qs;
		const uint8_t *qh	 = b[bi].qh;
		const uint8_t *signs = qs + 32;
		float		  *y	 = dst + (bi * 256);

		float db[2];
		for (int ib32 = 0; ib32 < 8; ++ib32) {
			db[0] = d * (0.5f + (float)(b[bi].scales[ib32] & 0xf)) * 0.25f;
			db[1] = d * (0.5f + (float)(b[bi].scales[ib32] >> 4)) * 0.25f;
			for (int l = 0; l < 4; ++l) {
				const float	   dl = db[l / 2];
				const uint8_t *grid =
					(const uint8_t *)&ggml_iq2s_grid[qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)];
				for (int j = 0; j < 8; ++j)
					y[j] = dl * (float)grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
				y += 8;
			}
			qs += 4;
			signs += 4;
		}
	}
}

void dequant_iq4_xs_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq4_xs_block *b = blocks;

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const float	   d  = f16_to_f32(b[bi].d);
		const uint8_t *qs = b[bi].qs;
		float		  *y  = dst + (bi * 256);

		for (int ib = 0; ib < 8; ++ib) {
			const int	sl = (b[bi].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf;
			const int	sh = (b[bi].scales_h >> 2 * ib) & 3;
			const float dl = d * ((sl | (sh << 4)) - 32);
			for (int j = 0; j < 16; ++j) {
				y[j]	  = dl * kvalues_iq4nl[qs[j] & 0xf];
				y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
			}
			y += 32;
			qs += 16;
		}
	}
}

static inline void iq3s_unpack_i8_generic(const iq3s_block *b, int8_t *a) {
	const uint8_t *qs	 = b->qs;
	const uint8_t *qh	 = b->qh;
	const uint8_t *signs = b->signs;
	int8_t		  *ap	 = a;

	for (int ib32 = 0; ib32 < 8; ib32 += 2) {
		for (int l = 0; l < 4; l++) {
			const uint8_t *grid1 =
				(const uint8_t
					 *)(&ggml_iq3s_grid[qs[(ptrdiff_t)(2 * l)] | ((qh[0] << (8 - (2 * l))) & 256)]);
			const uint8_t *grid2 =
				(const uint8_t
					 *)(&ggml_iq3s_grid[qs[(2 * l) + 1] | ((qh[0] << (7 - (2 * l))) & 256)]);
			uint8_t sign_byte = signs[l];
			for (int j = 0; j < 4; j++) {
				ap[j]	  = (sign_byte & kmask_iq2xs[j]) ? -(int8_t)grid1[j] : (int8_t)grid1[j];
				ap[j + 4] = (sign_byte & kmask_iq2xs[j + 4]) ? -(int8_t)grid2[j] : (int8_t)grid2[j];
			}
			ap += 8;
		}
		qs += 8;
		signs += 4;

		for (int l = 0; l < 4; l++) {
			const uint8_t *grid1 =
				(const uint8_t
					 *)(&ggml_iq3s_grid[qs[(ptrdiff_t)(2 * l)] | ((qh[1] << (8 - (2 * l))) & 256)]);
			const uint8_t *grid2 =
				(const uint8_t
					 *)(&ggml_iq3s_grid[qs[(2 * l) + 1] | ((qh[1] << (7 - (2 * l))) & 256)]);
			uint8_t sign_byte = signs[l];
			for (int j = 0; j < 4; j++) {
				ap[j]	  = (sign_byte & kmask_iq2xs[j]) ? -(int8_t)grid1[j] : (int8_t)grid1[j];
				ap[j + 4] = (sign_byte & kmask_iq2xs[j + 4]) ? -(int8_t)grid2[j] : (int8_t)grid2[j];
			}
			ap += 8;
		}
		qh += 2;
		qs += 8;
		signs += 4;
	}
}

static void matmul_iq3_s_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
											float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(iq3s_block);
	const iq3s_block *wb			 = w;
	int8_t			  a[256];

	for (int i = 0; i < n; i++) {
		const iq3s_block *bx = (const iq3s_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float			  sumf = 0.0f;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const iq3s_block *b	 = &bx[bi];
			const q8_k_block *yb = &xq[bi];
			const float		  d	 = f16_to_f32(b->d) * yb->d;

			iq3s_unpack_i8_generic(b, a);

			const int8_t *q8	= yb->qs;
			int32_t		  total = 0;
			for (int g = 0; g < 8; g++) {
				uint8_t scale_byte = b->scales[g / 2];
				int sc = (g & 1) ? (1 + (2 * (scale_byte >> 4))) : (1 + (2 * (scale_byte & 0xf)));
				int32_t group_sum = 0;
				for (int j = 0; j < 32; j++)
					group_sum += (int32_t)a[(g * 32) + j] * (int32_t)q8[(g * 32) + j];
				total += sc * group_sum;
			}
			sumf = fmaf(d, (float)total, sumf);
		}
		y[i] = sumf;
	}
}

typedef void (*matmul_row_fn)(const void *w, const float *restrict x, float *restrict y, int n,
							  int k);

#define MATMUL_QONLY_PANEL_BYTES ((size_t)128 << 10)

typedef void (*matmul_qonly_row_fn)(const void *w, const void *xq, float *restrict y, int n, int k);

static void matmul_qonly_dispatch(const void *w, const void *xq, size_t xq_elem,
								  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								  int n, int k, int m, size_t w_row_bytes, int row_align,
								  matmul_qonly_row_fn row) {
	size_t w_row = w_row_bytes;
	if (m <= 1 || w_row == 0) {
		for (int t = 0; t < m; t++)
			row(w, (const char *)xq + ((size_t)t * xq_row_stride_blocks) * xq_elem,
				y + ((size_t)t * y_row_stride), n, k);
		return;
	}
	int nb = (int)(MATMUL_QONLY_PANEL_BYTES / w_row);
	if (nb < row_align)
		nb = row_align;
	nb -= nb % row_align;
	if (nb < row_align)
		nb = row_align;
	for (int i0 = 0; i0 < n; i0 += nb) {
		int			   nn = (n - i0) < nb ? (n - i0) : nb;
		const uint8_t *wp = (const uint8_t *)w + ((size_t)i0 * w_row);
		for (int t = 0; t < m; t++)
			row(wp, (const char *)xq + ((size_t)t * xq_row_stride_blocks) * xq_elem,
				y + ((size_t)t * y_row_stride) + i0, nn, k);
	}
}

void matmul_iq3_s_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * sizeof(iq3s_block)), (1),
						  (matmul_qonly_row_fn)matmul_iq3_s_q8_k_qonly_f32_row);
}

static void matmul_iq3_s_repacked_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
													 float *restrict y, int n, int k) {
	matmul_iq3_s_repacked_q8_k_qonly_f32_rows_range(w, xq, y, 0, n, k);
}

void matmul_iq3_s_repacked_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
										  size_t	  xq_row_stride_blocks, float *restrict y,
										  int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * IQ3_S_REPACKED_BLOCK_BYTES), (IQ3_S_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_iq3_s_repacked_q8_k_qonly_f32_row);
}

void dequant_iq4_nl_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq4_nl_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float d = f16_to_f32(b[bi].d);
		for (int j = 0; j < 16; j++) {
			uint8_t byte			= b[bi].qs[j];
			dst[(bi * 32) + j]		= d * kvalues_iq4nl[byte & 0xF];
			dst[(bi * 32) + j + 16] = d * kvalues_iq4nl[byte >> 4];
		}
	}
}

static void matmul_iq4_nl_repacked_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
													float *restrict y, int n, int k) {
	matmul_iq4_nl_repacked_q8_qonly_f32_rows_range(w, xq, y, 0, n, k);
}

void matmul_iq4_nl_repacked_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
										 size_t		 xq_row_stride_blocks, float *restrict y,
										 int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_0_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(iq4_nl_block)), (IQ4_NL_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_iq4_nl_repacked_q8_qonly_f32_row);
}

static void matmul_q8_0_repacked_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
												  float *restrict y, int n, int k) {
	matmul_q8_0_repacked_q8_qonly_f32_rows_range(w, xq, y, 0, n, k);
}

void matmul_q8_0_repacked_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
									   size_t	   xq_row_stride_blocks, float *restrict y,
									   int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_0_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q8_0_block)), (Q8_0_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_q8_0_repacked_q8_qonly_f32_row);
}

static void matmul_q4_k_repacked_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
													float *restrict y, int n, int k) {
	matmul_q4_k_repacked_q8_k_qonly_f32_rows_range(w, xq, y, 0, n, k);
}

void matmul_q4_k_repacked_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
										 size_t		 xq_row_stride_blocks, float *restrict y,
										 int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * Q4_K_REPACKED_ROW_BYTES), (Q4_K_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_q4_k_repacked_q8_k_qonly_f32_row);
}

static void matmul_q5_k_repacked_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
													float *restrict y, int n, int k) {
	matmul_q5_k_repacked_q8_k_qonly_f32_rows_range(w, xq, y, 0, n, k);
}

void matmul_q5_k_repacked_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
										 size_t		 xq_row_stride_blocks, float *restrict y,
										 int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * Q5_K_REPACKED_ROW_BYTES), (Q5_K_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_q5_k_repacked_q8_k_qonly_f32_row);
}

static void matmul_q6_k_repacked_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
													float *restrict y, int n, int k) {
	matmul_q6_k_repacked_q8_k_qonly_f32_rows_range(w, xq, y, 0, n, k);
}

void matmul_q6_k_repacked_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
										 size_t		 xq_row_stride_blocks, float *restrict y,
										 int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * (Q6_K_REPACKED_GROUP_BYTES / Q6_K_REPACKED_ROWS)),
						  (Q6_K_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_q6_k_repacked_q8_k_qonly_f32_row);
}

static void matmul_q4_0_repacked_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
												  float *restrict y, int n, int k) {
	matmul_q4_0_repacked_q8_qonly_f32_rows_range(w, xq, y, 0, n, k);
}

void matmul_q4_0_repacked_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
									   size_t	   xq_row_stride_blocks, float *restrict y,
									   int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_0_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q4_0_block)), (Q4_0_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_q4_0_repacked_q8_qonly_f32_row);
}

void dequant_f16_row(const void *src, int n, float *dst) {
	const uint16_t *s = src;
	for (int i = 0; i < n; i++)
		dst[i] = f16_to_f32(s[i]);
}

void dequant_bf16_row(const void *src, int n, float *dst) {
	const uint16_t *s = src;
	for (int i = 0; i < n; i++)
		dst[i] = bf16_to_f32(s[i]);
}

void matmul_generic_f32(const void *w, uint32_t w_type, const float *x, float *y, int n, int k) {
	switch (w_type) {
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_0_REPACKED:
	case GGML_TYPE_Q4_0_REPACKED:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ4_NL_REPACKED:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ3_S_REPACKED: {
		static _Thread_local quant_scratch qs = {NULL, 0};
		if (!qs.q8_buf)
			tlocal_register((void **)&qs.q8_buf);
		switch (w_type) {
		case GGML_TYPE_Q4_0:
			matmul_q4_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q8_0:
			matmul_q8_0_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q8_0_REPACKED:
			matmul_q8_0_repacked_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q4_0_REPACKED:
			matmul_q4_0_repacked_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_IQ4_NL_REPACKED:
			matmul_iq4_nl_repacked_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q4_1:
			matmul_q4_1_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q5_0:
			matmul_q5_0_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q5_1:
			matmul_q5_1_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_IQ4_NL:
			matmul_iq4_nl_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q6_K:
			matmul_q6_k_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q4_K:
			matmul_q4_k_q8_k_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q5_K:
			matmul_q5_k_q8_k_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_IQ3_S:
			matmul_iq3_s_q8_k_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_IQ3_S_REPACKED:
			matmul_iq3_s_repacked_q8_k_f32(w, x, y, n, k, &qs);
			break;
		default:
			break;
		}
		return;
	}
	case GGML_TYPE_F32:
		matmul_f32_f32(w, x, y, n, k);
		return;
	case GGML_TYPE_F16:
		matmul_f16_f32(w, x, y, n, k);
		return;
	case GGML_TYPE_BF16:
		matmul_bf16_f32(w, x, y, n, k);
		return;
	default:
		break;
	}

	size_t row_stride = ggml_row_size(w_type, k);
	float *row_buf	  = xmalloc((size_t)k * sizeof(float));
	memset(row_buf, 0, (size_t)k * sizeof(float));

	for (int i = 0; i < n; i++) {
		const uint8_t *row = (const uint8_t *)w + ((size_t)i * row_stride);
		dequant_row_dispatch(w_type, row, k, row_buf);
		y[i] = dot_f32(row_buf, x, k);
	}
	free(row_buf);
}

static void quantize_q8_32block(const float *x, uint16_t *d_out, int8_t *qs_out, int32_t *sum_out,
								uint16_t *s_out) {
	float amax = 0;
	for (int j = 0; j < 32; j++) {
		float ax = fabsf(x[j]);
		if (ax > amax)
			amax = ax;
	}
	float d		= amax / 127.0f;
	float id	= d > 0 ? 1.0f / d : 0.0f;
	*d_out		= f32_to_f16(d);
	int32_t sum = 0;
	for (int j = 0; j < 32; j++) {
		int v = (int)roundf(x[j] * id);
		if (v > 127)
			v = 127;
		if (v < -127)
			v = -127;
		qs_out[j] = (int8_t)v;
		sum += v;
	}
	if (sum_out)
		*sum_out = sum;
	if (s_out)
		*s_out = f32_to_f16(d * (float)sum);
}

void quantize_q8_0(const float *x, q8_0_block *dst, int n) {
	int nb = n / 32;
	for (int i = 0; i < nb; i++) {
		quantize_q8_32block(x + (size_t)i * 32, &dst[i].d, dst[i].qs, NULL, NULL);
	}
}

void quantize_q8_1(const float *x, void *dst, int n) {
	int			nb = n / 32;
	q8_1_block *y  = dst;
	for (int i = 0; i < nb; i++) {
		quantize_q8_32block(x + (size_t)i * 32, &y[i].d, y[i].qs, NULL, &y[i].s);
	}
}

void quantize_q8_k(const float *x, q8_k_block *y, int n) {
	int nb = n / 256;
	for (int i = 0; i < nb; i++) {
		float max  = 0;
		float amax = 0;
		for (int j = 0; j < 256; j++) {
			float ax = fabsf(x[j]);
			if (ax > amax) {
				amax = ax;
				max	 = x[j];
			}
		}
		if (!amax) {
			y[i].d = 0;
			memset(y[i].qs, 0, 256);
			memset(y[i].bsums, 0, 16 * sizeof(int16_t));
			x += 256;
			continue;
		}
		float iscale = -127.0f / max;
		for (int j = 0; j < 256; j++) {
			int v = (int)roundf(iscale * x[j]);
			if (v > 127)
				v = 127;
			if (v < -127)
				v = -127;
			y[i].qs[j] = (int8_t)v;
		}
		for (int j = 0; j < 16; j++) {
			int sum = 0;
			for (int ii = 0; ii < 16; ii++)
				sum += y[i].qs[(j * 16) + ii];
			y[i].bsums[j] = (int16_t)sum;
		}
		y[i].d = 1.0f / iscale;
		x += 256;
	}
}

static void matmul_q4_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
									   float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_0_block);
	const q4_0_block *wb			 = w;
	const int		  MR			 = MATMUL_MR;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float sumf[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			sumf[r] = 0.0f;

		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)wb + ((size_t)(i + r) * row_stride);

		for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MATMUL_K_BLOCK) {
			const int kb_end =
				kb0 + MATMUL_K_BLOCK < blocks_per_row ? kb0 + MATMUL_K_BLOCK : blocks_per_row;
			for (int bi = kb0; bi < kb_end; bi++) {
				const int8_t *restrict xq8 = xq[bi].qs;
				const float d_xq		   = f16_to_f32(xq[bi].d);
				for (int r = 0; r < MATMUL_MR; r++) {
					const q4_0_block *row =
						(const q4_0_block *)(row_base[r] + ((size_t)bi * sizeof(q4_0_block)));
					sumf[r] = q4_0_dot(row, xq8, d_xq, sumf[r]);
				}
			}
		}
		for (int r = 0; r < MATMUL_MR; r++)
			y[i + r] = sumf[r];
	}

	for (; i < n; i++) {
		const q4_0_block *row =
			(const q4_0_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);
			sumf					   = q4_0_dot(&row[bi], xq8, d_xq, sumf);
		}
		y[i] = sumf;
	}
}

static void matmul_q8_0_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
										 float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q8_0_block);
	const q8_0_block *wb			 = w;
	const int		  MR			 = MATMUL_MR;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float sumf[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			sumf[r] = 0.0f;

		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)wb + ((size_t)(i + r) * row_stride);

		for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MATMUL_K_BLOCK) {
			const int kb_end =
				kb0 + MATMUL_K_BLOCK < blocks_per_row ? kb0 + MATMUL_K_BLOCK : blocks_per_row;
			for (int bi = kb0; bi < kb_end; bi++) {
				const int8_t *restrict xq8 = xq[bi].qs;
				const float d_xq		   = f16_to_f32(xq[bi].d);
				for (int r = 0; r < MATMUL_MR; r++) {
					const q8_0_block *row =
						(const q8_0_block *)(row_base[r] + ((size_t)bi * sizeof(q8_0_block)));
					sumf[r] = q8_0_dot(row, xq8, d_xq, sumf[r]);
				}
			}
		}
		for (int r = 0; r < MATMUL_MR; r++)
			y[i + r] = sumf[r];
	}

	for (; i < n; i++) {
		const q8_0_block *row =
			(const q8_0_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);
			sumf					   = q8_0_dot(&row[bi], xq8, d_xq, sumf);
		}
		y[i] = sumf;
	}
}

void matmul_q4_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							size_t xq_row_stride_blocks, float *restrict y, int y_row_stride, int n,
							int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_0_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q4_0_block)), (1),
						  (matmul_qonly_row_fn)matmul_q4_q8_qonly_f32_row);
}

void matmul_q8_0_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_0_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q8_0_block)), (1),
						  (matmul_qonly_row_fn)matmul_q8_0_q8_qonly_f32_row);
}

static void matmul_iq4_nl_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
										   float *restrict y, int n, int k) {
	const int			blocks_per_row = k / 32;
	const size_t		row_stride	   = (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const iq4_nl_block *wb			   = w;
	const int			MR			   = MATMUL_MR;
	int					i			   = 0;

	for (; i + MR <= n; i += MR) {
		float sumf[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			sumf[r] = 0.0f;

		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)wb + ((size_t)(i + r) * row_stride);

		for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MATMUL_K_BLOCK) {
			const int kb_end =
				kb0 + MATMUL_K_BLOCK < blocks_per_row ? kb0 + MATMUL_K_BLOCK : blocks_per_row;
			for (int bi = kb0; bi < kb_end; bi++) {
				const int8_t *restrict xq8 = xq[bi].qs;
				const float d_xq		   = f16_to_f32(xq[bi].d);
				for (int r = 0; r < MATMUL_MR; r++) {
					const iq4_nl_block *row =
						(const iq4_nl_block *)(row_base[r] + ((size_t)bi * sizeof(iq4_nl_block)));
					sumf[r] = iq4_nl_dot(row, xq8, d_xq, sumf[r]);
				}
			}
		}
		for (int r = 0; r < MATMUL_MR; r++)
			y[i + r] = sumf[r];
	}

	for (; i < n; i++) {
		const iq4_nl_block *row =
			(const iq4_nl_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);
			sumf					   = iq4_nl_dot(&row[bi], xq8, d_xq, sumf);
		}
		y[i] = sumf;
	}
}

void matmul_iq4_nl_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_0_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(iq4_nl_block)), (1),
						  (matmul_qonly_row_fn)matmul_iq4_nl_q8_qonly_f32_row);
}

static void matmul_q4_1_q8_qonly_f32_row(const void *w, const q8_1_block *restrict xq,
										 float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_1_block);
	const q4_1_block *wb			 = w;
	const int		  MR			 = MATMUL_MR;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float sumf[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			sumf[r] = 0.0f;

		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)wb + ((size_t)(i + r) * row_stride);

		for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MATMUL_K_BLOCK) {
			const int kb_end =
				kb0 + MATMUL_K_BLOCK < blocks_per_row ? kb0 + MATMUL_K_BLOCK : blocks_per_row;
			for (int bi = kb0; bi < kb_end; bi++) {
				const int8_t *restrict xq8 = xq[bi].qs;
				const float d_xq		   = f16_to_f32(xq[bi].d);
				const float s_xq		   = f16_to_f32(xq[bi].s);
				for (int r = 0; r < MATMUL_MR; r++) {
					const q4_1_block *row =
						(const q4_1_block *)(row_base[r] + ((size_t)bi * sizeof(q4_1_block)));
					sumf[r] = q4_1_dot(row, xq8, d_xq, s_xq, sumf[r]);
				}
			}
		}
		for (int r = 0; r < MATMUL_MR; r++)
			y[i + r] = sumf[r];
	}

	for (; i < n; i++) {
		const q4_1_block *row =
			(const q4_1_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);
			const float s_xq		   = f16_to_f32(xq[bi].s);
			sumf					   = q4_1_dot(&row[bi], xq8, d_xq, s_xq, sumf);
		}
		y[i] = sumf;
	}
}

void matmul_q4_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_1_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q4_1_block)), (1),
						  (matmul_qonly_row_fn)matmul_q4_1_q8_qonly_f32_row);
}

static void matmul_q5_0_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
										 float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_0_block);
	const q5_0_block *wb			 = w;
	const int		  MR			 = MATMUL_MR;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float sumf[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			sumf[r] = 0.0f;

		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)wb + ((size_t)(i + r) * row_stride);

		for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MATMUL_K_BLOCK) {
			const int kb_end =
				kb0 + MATMUL_K_BLOCK < blocks_per_row ? kb0 + MATMUL_K_BLOCK : blocks_per_row;
			for (int bi = kb0; bi < kb_end; bi++) {
				const int8_t *restrict xq8 = xq[bi].qs;
				const float d_xq		   = f16_to_f32(xq[bi].d);
				for (int r = 0; r < MATMUL_MR; r++) {
					const q5_0_block *row =
						(const q5_0_block *)(row_base[r] + ((size_t)bi * sizeof(q5_0_block)));
					sumf[r] = q5_0_dot(row, xq8, d_xq, sumf[r]);
				}
			}
		}
		for (int r = 0; r < MATMUL_MR; r++)
			y[i + r] = sumf[r];
	}

	for (; i < n; i++) {
		const q5_0_block *row =
			(const q5_0_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);
			sumf					   = q5_0_dot(&row[bi], xq8, d_xq, sumf);
		}
		y[i] = sumf;
	}
}

MATMUL_Q8_F32(q5_0_q8, q8_0_block, 32, quantize_q8_0, matmul_q5_0_q8_qonly_f32)

void matmul_q5_0_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_0_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q5_0_block)), (1),
						  (matmul_qonly_row_fn)matmul_q5_0_q8_qonly_f32_row);
}

static void matmul_q5_1_q8_qonly_f32_row(const void *w, const q8_1_block *restrict xq,
										 float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_1_block);
	const q5_1_block *wb			 = w;
	const int		  MR			 = MATMUL_MR;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float sumf[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			sumf[r] = 0.0f;

		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)wb + ((size_t)(i + r) * row_stride);

		for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MATMUL_K_BLOCK) {
			const int kb_end =
				kb0 + MATMUL_K_BLOCK < blocks_per_row ? kb0 + MATMUL_K_BLOCK : blocks_per_row;
			for (int bi = kb0; bi < kb_end; bi++) {
				const int8_t *restrict xq8 = xq[bi].qs;
				const float d_xq		   = f16_to_f32(xq[bi].d);
				const float s_xq		   = f16_to_f32(xq[bi].s);
				for (int r = 0; r < MATMUL_MR; r++) {
					const q5_1_block *row =
						(const q5_1_block *)(row_base[r] + ((size_t)bi * sizeof(q5_1_block)));
					sumf[r] = q5_1_dot(row, xq8, d_xq, s_xq, sumf[r]);
				}
			}
		}
		for (int r = 0; r < MATMUL_MR; r++)
			y[i + r] = sumf[r];
	}

	for (; i < n; i++) {
		const q5_1_block *row =
			(const q5_1_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);
			const float s_xq		   = f16_to_f32(xq[bi].s);
			sumf					   = q5_1_dot(&row[bi], xq8, d_xq, s_xq, sumf);
		}
		y[i] = sumf;
	}
}

MATMUL_Q8_F32(q5_1_q8, q8_1_block, 32, quantize_q8_1, matmul_q5_1_q8_qonly_f32)

void matmul_q5_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_1_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q5_1_block)), (1),
						  (matmul_qonly_row_fn)matmul_q5_1_q8_qonly_f32_row);
}

static void matmul_q6_k_q8_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
										 float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q6_k_block);
	const q6_k_block *wb			 = w;
	const int		  MR			 = MATMUL_MR;
	int				  i				 = 0;
	int8_t			  q_unpack[256];

	for (; i + MR <= n; i += MR) {
		float sumf[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			sumf[r] = 0.0f;

		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float d_xq			  = yb->d;
			for (int r = 0; r < MATMUL_MR; r++) {
				const q6_k_block *restrict b =
					(const q6_k_block *)(row_base[r] + ((size_t)bi * sizeof(q6_k_block)));
				sumf[r] = q6_k_dot(b, yb->qs, d_xq, q_unpack, sumf[r]);
			}
		}
		for (int r = 0; r < MATMUL_MR; r++)
			y[i + r] = sumf[r];
	}

	for (; i < n; i++) {
		const q6_k_block *bx = (const q6_k_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float			  sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float d_xq			  = yb->d;
			sumf						  = q6_k_dot(&bx[bi], yb->qs, d_xq, q_unpack, sumf);
		}
		y[i] = sumf;
	}
}

void matmul_q6_k_q8_qonly_f32(const void *w, const q8_k_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * sizeof(q6_k_block)), (1),
						  (matmul_qonly_row_fn)matmul_q6_k_q8_qonly_f32_row);
}

static void matmul_q4_k_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
										   float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q4_k_block);
	const q4_k_block *wb			 = w;
	const int		  MR			 = MATMUL_MR;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float sumf[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			sumf[r] = 0.0f;

		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict xb = &xq[bi];
			const float xd				  = xb->d;
			const int8_t *restrict xq8	  = xb->qs;
			const int16_t *restrict bs	  = xb->bsums;
			for (int r = 0; r < MATMUL_MR; r++) {
				const q4_k_block *restrict b =
					(const q4_k_block *)(row_base[r] + ((size_t)bi * sizeof(q4_k_block)));
				sumf[r] = q4_k_dot(b, xq8, bs, xd, sumf[r]);
			}
		}
		for (int r = 0; r < MATMUL_MR; r++)
			y[i + r] = sumf[r];
	}

	for (; i < n; i++) {
		const q4_k_block *row =
			(const q4_k_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict xb = &xq[bi];
			const float xd				  = xb->d;
			const int8_t *restrict xq8	  = xb->qs;
			const int16_t *restrict bs	  = xb->bsums;
			sumf						  = q4_k_dot(&row[bi], xq8, bs, xd, sumf);
		}
		y[i] = sumf;
	}
}

void matmul_q4_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * sizeof(q4_k_block)), (1),
						  (matmul_qonly_row_fn)matmul_q4_k_q8_k_qonly_f32_row);
}

static void matmul_q5_k_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
										   float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q5_k_block);
	const q5_k_block *wb			 = w;
	const int		  MR			 = MATMUL_MR;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float sumf[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			sumf[r] = 0.0f;

		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict xb = &xq[bi];
			const float xd				  = xb->d;
			const int8_t *restrict xq8	  = xb->qs;
			const int16_t *restrict bs	  = xb->bsums;
			for (int r = 0; r < MATMUL_MR; r++) {
				const q5_k_block *restrict b =
					(const q5_k_block *)(row_base[r] + ((size_t)bi * sizeof(q5_k_block)));
				sumf[r] = q5_k_dot(b, xq8, bs, xd, sumf[r]);
			}
		}
		for (int r = 0; r < MATMUL_MR; r++)
			y[i + r] = sumf[r];
	}

	for (; i < n; i++) {
		const q5_k_block *row =
			(const q5_k_block *)((const uint8_t *)wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict xb = &xq[bi];
			const float xd				  = xb->d;
			const int8_t *restrict xq8	  = xb->qs;
			const int16_t *restrict bs	  = xb->bsums;
			sumf						  = q5_k_dot(&row[bi], xq8, bs, xd, sumf);
		}
		y[i] = sumf;
	}
}

void matmul_q5_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * sizeof(q5_k_block)), (1),
						  (matmul_qonly_row_fn)matmul_q5_k_q8_k_qonly_f32_row);
}

void matmul_f32_f32(const float *restrict w, const float *restrict x, float *restrict y, int n,
					int k) {
	const float *xp = x;
	float		*yp = y;

	const int MR = 8;
	int		  i	 = 0;
	for (; i + MR <= n; i += MR) {
		float acc[MR];
		for (int r = 0; r < MR; r++)
			acc[r] = 0.0f;
		const float *rows[MR];
		for (int r = 0; r < MR; r++)
			rows[r] = w + ((size_t)(i + r) * k);
		for (int j = 0; j < k; j++) {
			float xv = xp[j];
			for (int r = 0; r < MR; r++)
				acc[r] += rows[r][j] * xv;
		}
		for (int r = 0; r < MR; r++)
			yp[i + r] = acc[r];
	}
	for (; i < n; i++) {
		const float *restrict wr = w + ((size_t)i * k);
		float acc				 = 0.0f;
		for (int j = 0; j < k; j++)
			acc += wr[j] * xp[j];
		yp[i] = acc;
	}
}

void matmul_bf16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					 int k) {
	const uint16_t *wb = w;
	for (int i = 0; i < n; i++) {
		const uint16_t *restrict wr = wb + ((size_t)i * k);
		float acc					= 0.0f;
		for (int j = 0; j < k; j++) {
			union {
				uint32_t u;
				float	 f;
			} v;
			v.u = ((uint32_t)wr[j]) << 16;
			acc += v.f * x[j];
		}
		y[i] = acc;
	}
}

static void matmul_f32_row_thunk(const void *w, const float *restrict x, float *restrict y, int n,
								 int k) {
	matmul_f32_f32((const float *)w, x, y, n, k);
}

static void matmul_f16_row_thunk(const void *w, const float *restrict x, float *restrict y, int n,
								 int k) {
	matmul_f16_f32(w, x, y, n, k);
}

static void matmul_bf16_row_thunk(const void *w, const float *restrict x, float *restrict y, int n,
								  int k) {
	matmul_bf16_f32(w, x, y, n, k);
}

static void matmul_panel_batch(const void *restrict w, const float *restrict x, float *restrict y,
							   int n, int k, int m, int x_row_stride, int y_row_stride,
							   size_t row_bytes, matmul_row_fn row_fn) {
	if (m <= 1) {
		for (int t = 0; t < m; t++)
			row_fn(w, x + (size_t)t * x_row_stride, y + (size_t)t * y_row_stride, n, k);
		return;
	}
	int nb = (int)(MATMUL_QONLY_PANEL_BYTES / row_bytes);
	if (nb < 1)
		nb = 1;
	for (int i0 = 0; i0 < n; i0 += nb) {
		int			nn = (n - i0) < nb ? (n - i0) : nb;
		const void *wp = (const uint8_t *)w + ((size_t)i0 * row_bytes);
		for (int t = 0; t < m; t++)
			row_fn(wp, x + (size_t)t * x_row_stride, y + (size_t)t * y_row_stride + i0, nn, k);
	}
}

void matmul_bf16_f32_batch(const void *restrict w, const float *restrict x, float *restrict y,
						   int n, int k, int m, int x_row_stride, int y_row_stride) {
	matmul_panel_batch(w, x, y, n, k, m, x_row_stride, y_row_stride, (size_t)k * sizeof(uint16_t),
					   matmul_bf16_row_thunk);
}

void matmul_f32_f32_batch(const float *restrict w, const float *restrict x, float *restrict y,
						  int n, int k, int m, int x_row_stride, int y_row_stride) {
	matmul_panel_batch(w, x, y, n, k, m, x_row_stride, y_row_stride, (size_t)k * sizeof(float),
					   matmul_f32_row_thunk);
}

void matmul_f16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					int k) {
	const uint16_t *wb = w;
	for (int i = 0; i < n; i++) {
		const uint16_t *restrict wr = wb + ((size_t)i * k);
		float acc					= 0.0f;
		for (int j = 0; j < k; j++)
			acc += f16_to_f32(wr[j]) * x[j];
		y[i] = acc;
	}
}

void matmul_f16_f32_batch(const void *restrict w, const float *restrict x, float *restrict y, int n,
						  int k, int m, int x_row_stride, int y_row_stride) {
	matmul_panel_batch(w, x, y, n, k, m, x_row_stride, y_row_stride, (size_t)k * sizeof(uint16_t),
					   matmul_f16_row_thunk);
}

float dot_f32(const float *restrict a, const float *restrict b, int n) {
	if (n <= 0)
		return 0.0f;
	float acc = 0.0f;
	for (int i = 0; i < n; i++)
		acc += a[i] * b[i];
	return acc;
}

float rmsnorm_sum_sq(const float *x, int n) {
	float ss0	 = 0;
	float ss1	 = 0;
	float ss2	 = 0;
	float ss3	 = 0;
	int	  i		 = 0;
	int	  n_main = n - (n % 4);
	for (; i < n_main; i += 4) {
		ss0 += x[i] * x[i];
		ss1 += x[i + 1] * x[i + 1];
		ss2 += x[i + 2] * x[i + 2];
		ss3 += x[i + 3] * x[i + 3];
	}
	for (; i < n; i++)
		ss0 += x[i] * x[i];
	return (ss0 + ss1) + (ss2 + ss3);
}

void rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
	float ss	= rmsnorm_sum_sq(x, n);
	float scale = 1.0f / sqrtf((ss / (float)n) + eps);
	for (int i = 0; i < n; i++)
		y[i] = x[i] * scale * w[i];
}

void rmsnorm_noweight(const float *x, float *y, int n, float eps) {
	float ss	= rmsnorm_sum_sq(x, n);
	float scale = 1.0f / sqrtf((ss / (float)n) + eps);
	for (int i = 0; i < n; i++)
		y[i] = x[i] * scale;
}

void rmsnorm_per_head(const float *x, const float *w, float *y, int n_heads, int head_dim,
					  float eps) {
	for (int h = 0; h < n_heads; h++) {
		const float *xh	   = x + ((size_t)h * head_dim);
		float		*yh	   = y + ((size_t)h * head_dim);
		float		 ss	   = rmsnorm_sum_sq(xh, head_dim);
		float		 scale = 1.0f / sqrtf((ss / (float)head_dim) + eps);
		for (int j = 0; j < head_dim; j++)
			yh[j] = xh[j] * scale * w[j];
	}
}

void softmax_masked(float *restrict scores, int n_valid) {
	float mx = scores[0];
	for (int i = 1; i < n_valid; i++)
		if (scores[i] > mx)
			mx = scores[i];
	float sum = 0.0f;
	for (int i = 0; i < n_valid; i++) {
		scores[i] = expf(scores[i] - mx);
		sum += scores[i];
	}
	float inv = 1.0f / sum;
	for (int i = 0; i < n_valid; i++)
		scores[i] *= inv;
}

void moe_activate_silu(float *restrict act, const float *restrict gate, const float *restrict up,
					   int n, float gs, float us) {
	for (int i = 0; i < n; i++)
		act[i] = silu(gate[i] * gs) * (up[i] * us);
}

void moe_activate_gelu(float *restrict act, const float *restrict gate, const float *restrict up,
					   int n, float gs, float us) {
	for (int i = 0; i < n; i++)
		act[i] = gelu_tanh(gate[i] * gs) * (up[i] * us);
}

void iq_qonly_via_dequant(const void *w, uint32_t wtype, const q8_k_block *restrict xq, size_t xs,
						  float *restrict y, int ys, int n, int k, int m) {
	dequant_blocks_fn dq  = NULL;
	int				  div = 256;
	dequant_row_plan(wtype, &dq, &div);
	size_t wb  = ggml_row_size(wtype, k);
	int	   bpr = k / 256;
	float *xf  = malloc((size_t)k * sizeof(float));
	float *wf  = malloc((size_t)k * sizeof(float));
	for (int t = 0; t < m; t++) {
		const q8_k_block *xb = xq + (size_t)t * xs;
		for (int bi = 0; bi < bpr; bi++) {
			float	d  = xb[bi].d;
			int8_t *qs = (int8_t *)xb[bi].qs;
			for (int j = 0; j < 256; j++)
				xf[bi * 256 + j] = d * (float)qs[j];
		}
		for (int i = 0; i < n; i++) {
			const uint8_t *row = (const uint8_t *)w + (size_t)i * wb;
			dq(row, (size_t)(k / div), wf);
			double acc = 0;
			for (int j = 0; j < k; j++)
				acc += (double)wf[j] * (double)xf[j];
			y[(size_t)t * ys + i] = (float)acc;
		}
	}
	free(xf);
	free(wf);
}
void matmul_iq2_s_q8_k_qonly_f32(const void *w, const q8_k_block *x, size_t xs, float *y, int ys,
								 int n, int k, int m) {
	iq_qonly_via_dequant(w, GGML_TYPE_IQ2_S, x, xs ? xs : (size_t)(k / 256), y, ys, n, k, m);
}
void matmul_iq2_xxs_q8_k_qonly_f32(const void *w, const q8_k_block *x, size_t xs, float *y, int ys,
								   int n, int k, int m) {
	iq_qonly_via_dequant(w, GGML_TYPE_IQ2_XXS, x, xs ? xs : (size_t)(k / 256), y, ys, n, k, m);
}
void matmul_iq3_xxs_q8_k_qonly_f32(const void *w, const q8_k_block *x, size_t xs, float *y, int ys,
								   int n, int k, int m) {
	iq_qonly_via_dequant(w, GGML_TYPE_IQ3_XXS, x, xs ? xs : (size_t)(k / 256), y, ys, n, k, m);
}
void matmul_iq4_xs_q8_k_qonly_f32(const void *w, const q8_k_block *x, size_t xs, float *y, int ys,
								  int n, int k, int m) {
	iq_qonly_via_dequant(w, GGML_TYPE_IQ4_XS, x, xs ? xs : (size_t)(k / 256), y, ys, n, k, m);
}

void repack_iq4_nl_to_q8_0_rows(const void *src, void *dst, int row_begin, int row_end, int k) {
	const int	 blocks_per_row = k / 32;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const size_t dst_stride		= (size_t)blocks_per_row * sizeof(q8_0_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	for (int r = row_begin; r < row_end; r++) {
		const iq4_nl_block *srow = (const iq4_nl_block *)(sp + (size_t)r * src_stride);
		q8_0_block		   *drow = (q8_0_block *)(dp + (size_t)r * dst_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			drow[bi].d = srow[bi].d;
			for (int j = 0; j < 16; j++) {
				uint8_t byte		= srow[bi].qs[j];
				drow[bi].qs[j]		= kvalues_iq4nl[byte & 0xF];
				drow[bi].qs[j + 16] = kvalues_iq4nl[byte >> 4];
			}
		}
	}
}

static float f16_to_f32_table[65536];

static inline uint32_t f32_to_bits(float f) {
	union {
		float	 v;
		uint32_t b;
	} u = {.v = f};
	return u.b;
}

static inline float f32_from_bits(uint32_t b) {
	union {
		uint32_t b;
		float	 v;
	} u = {.b = b};
	return u.v;
}

static float f16_to_f32_compute(uint16_t h) {
	uint32_t w			  = (uint32_t)h << 16;
	uint32_t sign		  = w & 0x80000000u;
	uint32_t two_w		  = w + w;
	uint32_t exp_offset	  = 0xE0u << 23;
	float	 exp_scale	  = 0x1.0p-112f;
	float	 normalized	  = f32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
	uint32_t magic_mask	  = 126u << 23;
	float	 denormalized = f32_from_bits((two_w >> 17) | magic_mask) - 0.5f;
	uint32_t cutoff		  = 1u << 27;
	uint32_t bits		  = two_w < cutoff ? f32_to_bits(denormalized) : f32_to_bits(normalized);
	return f32_from_bits(sign | bits);
}

__attribute__((constructor)) static void f16_to_f32_table_init(void) {
	for (uint32_t h = 0; h < 65536; h++)
		f16_to_f32_table[h] = f16_to_f32_compute((uint16_t)h);
}

float f16_to_f32(uint16_t h) {
	return f16_to_f32_table[h];
}

uint16_t f32_to_f16(float f) {
	float	 base	= (fabsf(f) * 0x1.0p+112f) * 0x1.0p-110f;
	uint32_t w		= f32_to_bits(f);
	uint32_t shl1_w = w + w;
	uint32_t sign	= w & 0x80000000u;
	uint32_t bias	= shl1_w & 0xFF000000u;
	if (bias < 0x71000000u)
		bias = 0x71000000u;
	base			   = f32_from_bits((bias >> 1) + 0x07800000u) + base;
	uint32_t bits	   = f32_to_bits(base);
	uint32_t exp_bits  = (bits >> 13) & 0x7C00u;
	uint32_t mant_bits = bits & 0x0FFFu;
	uint32_t nonsign   = exp_bits + mant_bits;
	return (sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign);
}

float bf16_to_f32(uint16_t h) {
	union {
		uint32_t u;
		float	 f;
	} v;
	v.u = ((uint32_t)h) << 16;
	return v.f;
}

void f32_to_f16_row(const float *src, uint16_t *dst, int n) {
	for (int i = 0; i < n; i++)
		dst[i] = f32_to_f16(src[i]);
}

void scale_f32(float *x, float s, int n) {
	for (int i = 0; i < n; i++)
		x[i] *= s;
}

void add_scale_f32(float *x, const float *y, float s, int n) {
	for (int i = 0; i < n; i++)
		x[i] = (x[i] + y[i]) * s;
}

void gdn_decay_accum_f32(float *restrict row, float *restrict mem, float decay, float ks, int n) {
	for (int i = 0; i < n; i++) {
		row[i] *= decay;
		mem[i] += row[i] * ks;
	}
}

void gdn_update_accum_f32(float *restrict row, float *restrict y, const float *restrict delta,
						  float ks, float qs, int n) {
	for (int i = 0; i < n; i++) {
		row[i] += ks * delta[i];
		y[i] += row[i] * qs;
	}
}

void gdn_out_norm_f32(float *restrict y, const float *restrict norm_w, const float *restrict z,
					  float inv_rms, int n) {
	for (int i = 0; i < n; i++)
		y[i] = y[i] * inv_rms * norm_w[i] * silu(z[i]);
}
