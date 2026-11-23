#ifndef CPU_SCALAR_MICROKERNELS_H
#define CPU_SCALAR_MICROKERNELS_H

#include "backend/cpu/quants.h"
#include <math.h>
#include <string.h>

#define MATMUL_MR 8
#define MATMUL_K_BLOCK 64

static inline float scalar_q4_0_dot(const void *wb, const void *xb, int8_t *scratch, float acc) {
	(void)scratch;
	const q4_0_block *blk	= wb;
	const q8_0_block *xblk	= xb;
	const uint8_t	 *qs	= blk->qs;
	const int8_t	 *xq8	= xblk->qs;
	int				  sumi0 = 0;
	int				  sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		sumi0 += ((int)(qs[j] & 0xF) - 8) * (int)xq8[j];
		sumi1 += ((int)(qs[j] >> 4) - 8) * (int)xq8[j + 16];
	}
	return fmaf(f16_to_f32(blk->d) * f16_to_f32(xblk->d), (float)(sumi0 + sumi1), acc);
}

static inline float scalar_q8_0_dot(const void *wb, const void *xb, int8_t *scratch, float acc) {
	(void)scratch;
	const q8_0_block *blk  = wb;
	const q8_0_block *xblk = xb;
	const int8_t	 *qs   = blk->qs;
	const int8_t	 *xq8  = xblk->qs;
	int				  sumi = 0;
	for (int j = 0; j < 32; j++)
		sumi += (int)qs[j] * (int)xq8[j];
	return fmaf(f16_to_f32(blk->d) * f16_to_f32(xblk->d), (float)sumi, acc);
}

static inline float scalar_iq4_nl_dot(const void *wb, const void *xb, int8_t *scratch, float acc) {
	(void)scratch;
	const iq4_nl_block *blk	  = wb;
	const q8_0_block   *xblk  = xb;
	const uint8_t	   *qs	  = blk->qs;
	const int8_t	   *xq8	  = xblk->qs;
	int					sumi0 = 0;
	int					sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		sumi0 += (int)xq8[j] * (int)kvalues_iq4nl[qs[j] & 0xF];
		sumi1 += (int)xq8[j + 16] * (int)kvalues_iq4nl[qs[j] >> 4];
	}
	return fmaf(f16_to_f32(blk->d) * f16_to_f32(xblk->d), (float)(sumi0 + sumi1), acc);
}

static inline float scalar_q4_1_dot(const void *wb, const void *xb, int8_t *scratch, float acc) {
	(void)scratch;
	const q4_1_block *blk	= wb;
	const q8_1_block *xblk	= xb;
	const uint8_t	 *qs	= blk->qs;
	const int8_t	 *xq8	= xblk->qs;
	int				  sumi0 = 0;
	int				  sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		sumi0 += (int)(qs[j] & 0xF) * (int)xq8[j];
		sumi1 += (int)(qs[j] >> 4) * (int)xq8[j + 16];
	}
	return fmaf(f16_to_f32(blk->d) * f16_to_f32(xblk->d), (float)(sumi0 + sumi1),
				fmaf(f16_to_f32(blk->m), f16_to_f32(xblk->s), acc));
}

static inline float scalar_q5_0_dot(const void *wb, const void *xb, int8_t *scratch, float acc) {
	(void)scratch;
	const q5_0_block *blk  = wb;
	const q8_0_block *xblk = xb;
	const uint8_t	 *qs   = blk->qs;
	const int8_t	 *xq8  = xblk->qs;
	uint32_t		  qh;
	memcpy(&qh, blk->qh, 4);
	int sumi0 = 0;
	int sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		int lo = (int)(qs[j] & 0xF) | (int)((qh >> j) & 1) << 4;
		int hi = (int)(qs[j] >> 4) | (int)((qh >> (j + 16)) & 1) << 4;
		sumi0 += (lo - 16) * (int)xq8[j];
		sumi1 += (hi - 16) * (int)xq8[j + 16];
	}
	return fmaf(f16_to_f32(blk->d) * f16_to_f32(xblk->d), (float)(sumi0 + sumi1), acc);
}

static inline float scalar_q5_1_dot(const void *wb, const void *xb, int8_t *scratch, float acc) {
	(void)scratch;
	const q5_1_block *blk  = wb;
	const q8_1_block *xblk = xb;
	const uint8_t	 *qs   = blk->qs;
	const int8_t	 *xq8  = xblk->qs;
	uint32_t		  qh;
	memcpy(&qh, blk->qh, 4);
	int sumi0 = 0;
	int sumi1 = 0;
	for (int j = 0; j < 16; j++) {
		int lo = (int)(qs[j] & 0xF) | (int)((qh >> j) & 1) << 4;
		int hi = (int)(qs[j] >> 4) | (int)((qh >> (j + 16)) & 1) << 4;
		sumi0 += lo * (int)xq8[j];
		sumi1 += hi * (int)xq8[j + 16];
	}
	return fmaf(f16_to_f32(blk->d) * f16_to_f32(xblk->d), (float)(sumi0 + sumi1),
				fmaf(f16_to_f32(blk->m), f16_to_f32(xblk->s), acc));
}

static inline float scalar_q6_k_dot(const void *wb, const void *xb, int8_t *scratch, float acc) {
	const q6_k_block *blk  = wb;
	const q8_k_block *xblk = xb;
	const uint8_t	 *ql   = blk->ql;
	const uint8_t	 *qh   = blk->qh;
	int8_t			 *a	   = scratch;
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
	a				   = scratch;
	const int8_t *q8   = xblk->qs;
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
	return fmaf(f16_to_f32(blk->d) * xblk->d, (float)(acc0 + acc1), acc);
}

static inline float scalar_q4_k_dot(const void *wb, const void *xb, int8_t *scratch, float acc) {
	(void)scratch;
	const q4_k_block *blk	 = wb;
	const q8_k_block *xblk	 = xb;
	const float		  d		 = f16_to_f32(blk->d);
	const float		  dmin	 = f16_to_f32(blk->dmin);
	const uint8_t	 *qbytes = blk->qs;
	const uint8_t	 *sc	 = blk->scales;
	const int8_t	 *xq8	 = xblk->qs;
	const int16_t	 *bs	 = xblk->bsums;
	int32_t			  sumi	 = 0;
	int32_t			  summ	 = 0;
	int				  is	 = 0;
	int				  ib	 = 0;
	uint8_t			  scu8;
	uint8_t			  mu8;
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
	return fmaf(xblk->d, fmaf(d, (float)sumi, -(dmin * (float)summ)), acc);
}

static inline float scalar_q5_k_dot(const void *wb, const void *xb, int8_t *scratch, float acc) {
	(void)scratch;
	const q5_k_block *blk	 = wb;
	const q8_k_block *xblk	 = xb;
	const float		  d		 = f16_to_f32(blk->d);
	const float		  dmin	 = f16_to_f32(blk->dmin);
	const uint8_t	 *qbytes = blk->qs;
	const uint8_t	 *qh	 = blk->qh;
	const uint8_t	 *sc	 = blk->scales;
	const int8_t	 *xq8	 = xblk->qs;
	const int16_t	 *bs	 = xblk->bsums;
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
	return fmaf(xblk->d, fmaf(d, (float)sumi, -(dmin * (float)summ)), acc);
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

typedef int32_t (*iq_block_tot_fn)(const void *b, const int8_t *q8);

static int32_t iq_dot8_signed(uint64_t grid, uint8_t signs, const int8_t *q8) {
	int32_t sum = 0;
	for (int j = 0; j < 8; j++) {
		int v = (int)((grid >> (8 * j)) & 0xff);
		sum += ((signs >> j) & 1 ? -v : v) * (int)q8[j];
	}
	return sum;
}

static int32_t iq2_xxs_block_tot(const void *b_, const int8_t *q8) {
	const iq2_xxs_block *b	 = b_;
	const uint8_t		*qs	 = (const uint8_t *)b->qs;
	int32_t				 tot = 0;
	for (int ib = 0; ib < 8; ib++) {
		uint32_t aux[2];
		memcpy(aux, qs + (8 * ib), 8);
		const uint8_t *a   = (const uint8_t *)aux;
		uint32_t	   s   = aux[1];
		int32_t		   sum = 0;
		for (int l = 0; l < 4; l++)
			sum += iq_dot8_signed(ggml_iq2xxs_grid[a[l]], ggml_ksigns_iq2xs[(s >> (7 * l)) & 127],
								  q8 + (ib * 32) + (l * 8));
		tot += (1 + 2 * (int)(s >> 28)) * sum;
	}
	return tot;
}

static int32_t iq3_xxs_block_tot(const void *b_, const int8_t *q8) {
	const iq3_xxs_block *b	 = b_;
	const uint8_t		*qs	 = b->qs;
	const uint8_t		*ss	 = b->qs + 64;
	int32_t				 tot = 0;
	for (int ib = 0; ib < 8; ib++) {
		uint32_t saux;
		memcpy(&saux, ss + (4 * ib), 4);
		const uint8_t *q   = qs + (8 * ib);
		int32_t		   sum = 0;
		for (int l = 0; l < 4; l++) {
			uint64_t grid = (uint64_t)ggml_iq3xxs_grid[q[2 * l]] |
							((uint64_t)ggml_iq3xxs_grid[q[(2 * l) + 1]] << 32);
			sum += iq_dot8_signed(grid, ggml_ksigns_iq2xs[(saux >> (7 * l)) & 127],
								  q8 + (ib * 32) + (l * 8));
		}
		tot += (1 + 2 * (int)(saux >> 28)) * sum;
	}
	return tot;
}

static int32_t iq2_s_block_tot(const void *b_, const int8_t *q8) {
	const iq2_s_block *b	 = b_;
	const uint8_t	  *qs	 = b->qs;
	const uint8_t	  *signs = b->qs + 32;
	int32_t			   tot	 = 0;
	for (int ib = 0; ib < 8; ib++) {
		uint8_t sb	 = b->scales[ib];
		int32_t sum0 = 0;
		int32_t sum1 = 0;
		for (int l = 0; l < 4; l++) {
			uint64_t grid = ggml_iq2s_grid[qs[l] | ((b->qh[ib] << (8 - (2 * l))) & 0x300)];
			int32_t	 part = iq_dot8_signed(grid, signs[l], q8 + (ib * 32) + (l * 8));
			if (l < 2)
				sum0 += part;
			else
				sum1 += part;
		}
		tot += ((1 + (2 * (sb & 0xf))) * sum0) + ((1 + (2 * (sb >> 4))) * sum1);
		qs += 4;
		signs += 4;
	}
	return tot;
}

static int32_t iq4_xs_block_tot(const void *b_, const int8_t *q8) {
	const iq4_xs_block *b	= b_;
	const uint8_t	   *qs	= b->qs;
	int32_t				tot = 0;
	for (int ib = 0; ib < 8; ib++) {
		int		sl	= (b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
		int		sh	= (b->scales_h >> (2 * ib)) & 3;
		int		dl	= (sl | (sh << 4)) - 32;
		int32_t sum = 0;
		for (int j = 0; j < 16; j++) {
			sum += (int)q8[(ib * 32) + j] * (int)kvalues_iq4nl[qs[(ib * 16) + j] & 0xF];
			sum += (int)q8[(ib * 32) + j + 16] * (int)kvalues_iq4nl[qs[(ib * 16) + j] >> 4];
		}
		tot += dl * sum;
	}
	return tot;
}

static void iq_qonly_scalar(const void *w, size_t block_size, iq_block_tot_fn tot_fn, float factor,
							const q8_k_block *xq, size_t xs, float *y, int ys, int n, int k,
							int m) {
	int			   bpr		  = k / 256;
	size_t		   row_stride = (size_t)bpr * block_size;
	const uint8_t *wb		  = w;
	for (int t = 0; t < m; t++) {
		const q8_k_block *xb = xq + ((size_t)t * xs);
		for (int i = 0; i < n; i++) {
			const uint8_t *row	= wb + ((size_t)i * row_stride);
			float		   sumf = 0.0f;
			for (int bi = 0; bi < bpr; bi++) {
				const void *b	= row + ((size_t)bi * block_size);
				int32_t		tot = tot_fn(b, xb[bi].qs);
				sumf = fmaf(factor * f16_to_f32(*(const uint16_t *)b) * xb[bi].d, (float)tot, sumf);
			}
			y[((size_t)t * ys) + i] = sumf;
		}
	}
}

#define MK_SCALAR_ROW_DRIVER(fn, wblk_t, xqblk_t, elems, dot_fn)                                   \
	static void fn(const void *w, const xqblk_t *restrict xq, float *restrict y, int n, int k) {   \
		const int	   bpr		  = k / (elems);                                                   \
		const size_t   row_stride = (size_t)bpr * sizeof(wblk_t);                                  \
		const uint8_t *wb		  = w;                                                             \
		int8_t		   scratch[256];                                                               \
		int			   i = 0;                                                                      \
		for (; i + MATMUL_MR <= n; i += MATMUL_MR) {                                               \
			float sumf[MATMUL_MR];                                                                 \
			for (int r = 0; r < MATMUL_MR; r++)                                                    \
				sumf[r] = 0.0f;                                                                    \
			for (int kb0 = 0; kb0 < bpr; kb0 += MATMUL_K_BLOCK) {                                  \
				const int kb_end = kb0 + MATMUL_K_BLOCK < bpr ? kb0 + MATMUL_K_BLOCK : bpr;        \
				for (int bi = kb0; bi < kb_end; bi++) {                                            \
					for (int r = 0; r < MATMUL_MR; r++) {                                          \
						const void *row =                                                          \
							wb + ((size_t)(i + r) * row_stride) + ((size_t)bi * sizeof(wblk_t));   \
						sumf[r] = dot_fn(row, &xq[bi], scratch, sumf[r]);                          \
					}                                                                              \
				}                                                                                  \
			}                                                                                      \
			for (int r = 0; r < MATMUL_MR; r++)                                                    \
				y[i + r] = sumf[r];                                                                \
		}                                                                                          \
		for (; i < n; i++) {                                                                       \
			float sumf = 0.0f;                                                                     \
			for (int bi = 0; bi < bpr; bi++) {                                                     \
				const void *row = wb + ((size_t)i * row_stride) + ((size_t)bi * sizeof(wblk_t));   \
				sumf			= dot_fn(row, &xq[bi], scratch, sumf);                             \
			}                                                                                      \
			y[i] = sumf;                                                                           \
		}                                                                                          \
	}

#endif
