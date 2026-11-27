#include "backend/cpu/quants.h"
#include "microkernels.h"
#include "threadpool.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

static void iq4b_rp_rows_range(const void *w, const q8_k_block *restrict xq, float *restrict y,
							   int row_begin, int row_end, int k, const int8_t *cb, float factor,
							   size_t row_bytes, size_t off_scales, size_t off_qs, int nscales) {
	const int	   bpr	 = k / 256;
	const size_t   group = row_bytes * IQ4_XS_REPACKED_ROWS;
	const uint8_t *wb	 = w;
	for (int g = row_begin - (row_begin % IQ4_XS_REPACKED_ROWS); g < row_end;
		 g += IQ4_XS_REPACKED_ROWS) {
		const uint8_t *dgroup = wb + (size_t)(g / IQ4_XS_REPACKED_ROWS) * bpr * group;
		float		   acc[IQ4_XS_REPACKED_ROWS] = {0};
		for (int bi = 0; bi < bpr; bi++) {
			const uint8_t	 *blk	 = dgroup + (size_t)bi * group;
			const uint16_t	 *dd	 = (const uint16_t *)(blk + IQ4_XS_REPACKED_OFF_D);
			const int8_t	 *scales = (const int8_t *)(blk + off_scales);
			const uint8_t	 *qs_all = blk + off_qs;
			const q8_k_block *xb	 = &xq[bi];
			const float		  dx	 = xb->d;
			for (int r = 0; r < IQ4_XS_REPACKED_ROWS; r++) {
				const uint8_t *qrow = qs_all + (size_t)r * 128;
				int32_t		   tot	= 0;
				for (int ib = 0; ib < 8; ib++) {
					const uint8_t *q  = qrow + (size_t)ib * 16;
					const int8_t  *x  = xb->qs + ib * 32;
					int32_t		   s0 = 0, s1 = 0;
					for (int j = 0; j < 16; j++) {
						s0 += cb[q[j] & 0xF] * (int32_t)x[j];
						s1 += cb[q[j] >> 4] * (int32_t)x[j + 16];
					}
					const int8_t *sc = scales + r * nscales;
					if (nscales == 16)
						tot += sc[2 * ib + 0] * s0 + sc[2 * ib + 1] * s1;
					else
						tot += sc[ib] * (s0 + s1);
				}
				acc[r] += f16_to_f32(dd[r]) * dx * factor * (float)tot;
			}
		}
		for (int r = 0; r < IQ4_XS_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = acc[r];
		}
	}
}

static void matmul_iq4_xs_repacked_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
													  float *restrict y, int n, int k) {
	iq4b_rp_rows_range(w, xq, y, 0, n, k, kvalues_iq4nl, 1.0f, IQ4_XS_REPACKED_ROW_BYTES,
					   IQ4_XS_REPACKED_OFF_SCALES, IQ4_XS_REPACKED_OFF_QS, 8);
}

void matmul_iq4_xs_repacked_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
										   size_t	   xq_row_stride_blocks, float *restrict y,
										   int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * IQ4_XS_REPACKED_ROW_BYTES), (IQ4_XS_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_iq4_xs_repacked_q8_k_qonly_f32_row);
}

static void matmul_iq3_xxs_repacked_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
													   float *restrict y, int n, int k) {
	iq4b_rp_rows_range(w, xq, y, 0, n, k, iq3_xxs_repacked_decode, 0.25f,
					   IQ3_XXS_REPACKED_ROW_BYTES, IQ3_XXS_REPACKED_OFF_SCALES,
					   IQ3_XXS_REPACKED_OFF_QS, 8);
}

void matmul_iq3_xxs_repacked_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
											size_t		xq_row_stride_blocks, float *restrict y,
											int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * IQ3_XXS_REPACKED_ROW_BYTES), (IQ3_XXS_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_iq3_xxs_repacked_q8_k_qonly_f32_row);
}

static void matmul_iq2_xxs_repacked_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
													   float *restrict y, int n, int k) {
	iq4b_rp_rows_range(w, xq, y, 0, n, k, iq3_xxs_repacked_decode, 0.125f,
					   IQ2_XXS_REPACKED_ROW_BYTES, IQ2_XXS_REPACKED_OFF_SCALES,
					   IQ2_XXS_REPACKED_OFF_QS, 8);
}

void matmul_iq2_xxs_repacked_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
											size_t		xq_row_stride_blocks, float *restrict y,
											int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * IQ2_XXS_REPACKED_ROW_BYTES), (IQ2_XXS_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_iq2_xxs_repacked_q8_k_qonly_f32_row);
}

static void matmul_iq2_s_repacked_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
													 float *restrict y, int n, int k) {
	iq4b_rp_rows_range(w, xq, y, 0, n, k, iq3_xxs_repacked_decode, 0.125f, IQ2_S_REPACKED_ROW_BYTES,
					   IQ2_S_REPACKED_OFF_SCALES, IQ2_S_REPACKED_OFF_QS, 16);
}

void matmul_iq2_s_repacked_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
										  size_t	  xq_row_stride_blocks, float *restrict y,
										  int y_row_stride, int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * IQ2_S_REPACKED_ROW_BYTES), (IQ2_S_REPACKED_ROWS),
						  (matmul_qonly_row_fn)matmul_iq2_s_repacked_q8_k_qonly_f32_row);
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

void matmul_q4_1_repacked_q8_qonly_f32_rows_range(const void *w, const q8_1_block *restrict xq,
												  float *restrict y, int row_begin, int row_end,
												  int k) {
	const int	   bpr	 = k / 32;
	const size_t   group = Q4_1_REPACKED_GROUP_BYTES;
	const uint8_t *wb	 = w;
	for (int g = row_begin - (row_begin % Q4_1_REPACKED_ROWS); g < row_end;
		 g += Q4_1_REPACKED_ROWS) {
		const uint8_t *dgroup = wb + (size_t)(g / Q4_1_REPACKED_ROWS) * bpr * group;
		float		   sumf[Q4_1_REPACKED_ROWS] = {0};
		for (int bi = 0; bi < bpr; bi++) {
			const uint8_t  *dblk = dgroup + (size_t)bi * group;
			const uint16_t *dd	 = (const uint16_t *)(dblk + Q4_1_REPACKED_OFF_D);
			const uint16_t *dm	 = (const uint16_t *)(dblk + Q4_1_REPACKED_OFF_M);
			const uint8_t  *dq	 = dblk + Q4_1_REPACKED_OFF_QS;
			const int8_t   *xq8	 = xq[bi].qs;
			const float		dx	 = f16_to_f32(xq[bi].d);
			const float		sx	 = f16_to_f32(xq[bi].s);
			for (int r = 0; r < Q4_1_REPACKED_ROWS; r++) {
				const uint8_t *qs	 = dq + (size_t)r * 16;
				int32_t		   sumi0 = 0, sumi1 = 0;
				for (int j = 0; j < 16; j++) {
					sumi0 += (int32_t)(qs[j] & 0xF) * (int32_t)xq8[j];
					sumi1 += (int32_t)(qs[j] >> 4) * (int32_t)xq8[j + 16];
				}
				sumf[r] = fmaf(f16_to_f32(dd[r]) * dx, (float)(sumi0 + sumi1),
							   fmaf(f16_to_f32(dm[r]), sx, sumf[r]));
			}
		}
		for (int r = 0; r < Q4_1_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

void matmul_q5_0_repacked_q8_qonly_f32_rows_range(const void *w, const q8_0_block *restrict xq,
												  float *restrict y, int row_begin, int row_end,
												  int k) {
	const int	   bpr	 = k / 32;
	const size_t   group = Q5_0_REPACKED_GROUP_BYTES;
	const uint8_t *wb	 = w;
	for (int g = row_begin - (row_begin % Q5_0_REPACKED_ROWS); g < row_end;
		 g += Q5_0_REPACKED_ROWS) {
		const uint8_t *dgroup = wb + (size_t)(g / Q5_0_REPACKED_ROWS) * bpr * group;
		float		   sumf[Q5_0_REPACKED_ROWS] = {0};
		for (int bi = 0; bi < bpr; bi++) {
			const uint8_t  *dblk = dgroup + (size_t)bi * group;
			const uint16_t *dd	 = (const uint16_t *)(dblk + Q5_0_REPACKED_OFF_D);
			const uint8_t  *dqh	 = dblk + Q5_0_REPACKED_OFF_QH;
			const uint8_t  *dq	 = dblk + Q5_0_REPACKED_OFF_QS;
			const int8_t   *xq8	 = xq[bi].qs;
			const float		dx	 = f16_to_f32(xq[bi].d);
			for (int r = 0; r < Q5_0_REPACKED_ROWS; r++) {
				uint32_t qh;
				memcpy(&qh, dqh + (size_t)r * 4, 4);
				const uint8_t *qs	 = dq + (size_t)r * 16;
				int32_t		   sumi0 = 0, sumi1 = 0;
				for (int j = 0; j < 16; j++) {
					int lo = (qs[j] & 0xF) | ((int)((qh >> j) & 1) << 4);
					int hi = (qs[j] >> 4) | ((int)((qh >> (j + 16)) & 1) << 4);
					sumi0 += (lo - 16) * (int32_t)xq8[j];
					sumi1 += (hi - 16) * (int32_t)xq8[j + 16];
				}
				sumf[r] = fmaf(f16_to_f32(dd[r]) * dx, (float)(sumi0 + sumi1), sumf[r]);
			}
		}
		for (int r = 0; r < Q5_0_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

void matmul_q5_1_repacked_q8_qonly_f32_rows_range(const void *w, const q8_1_block *restrict xq,
												  float *restrict y, int row_begin, int row_end,
												  int k) {
	const int	   bpr	 = k / 32;
	const size_t   group = Q5_1_REPACKED_GROUP_BYTES;
	const uint8_t *wb	 = w;
	for (int g = row_begin - (row_begin % Q5_1_REPACKED_ROWS); g < row_end;
		 g += Q5_1_REPACKED_ROWS) {
		const uint8_t *dgroup = wb + (size_t)(g / Q5_1_REPACKED_ROWS) * bpr * group;
		float		   sumf[Q5_1_REPACKED_ROWS] = {0};
		for (int bi = 0; bi < bpr; bi++) {
			const uint8_t  *dblk = dgroup + (size_t)bi * group;
			const uint16_t *dd	 = (const uint16_t *)(dblk + Q5_1_REPACKED_OFF_D);
			const uint16_t *dm	 = (const uint16_t *)(dblk + Q5_1_REPACKED_OFF_M);
			const uint8_t  *dqh	 = dblk + Q5_1_REPACKED_OFF_QH;
			const uint8_t  *dq	 = dblk + Q5_1_REPACKED_OFF_QS;
			const int8_t   *xq8	 = xq[bi].qs;
			const float		dx	 = f16_to_f32(xq[bi].d);
			const float		sx	 = f16_to_f32(xq[bi].s);
			for (int r = 0; r < Q5_1_REPACKED_ROWS; r++) {
				uint32_t qh;
				memcpy(&qh, dqh + (size_t)r * 4, 4);
				const uint8_t *qs	 = dq + (size_t)r * 16;
				int32_t		   sumi0 = 0, sumi1 = 0;
				for (int j = 0; j < 16; j++) {
					int lo = (qs[j] & 0xF) | ((int)((qh >> j) & 1) << 4);
					int hi = (qs[j] >> 4) | ((int)((qh >> (j + 16)) & 1) << 4);
					sumi0 += lo * (int32_t)xq8[j];
					sumi1 += hi * (int32_t)xq8[j + 16];
				}
				sumf[r] = fmaf(f16_to_f32(dd[r]) * dx, (float)(sumi0 + sumi1),
							   fmaf(f16_to_f32(dm[r]), sx, sumf[r]));
			}
		}
		for (int r = 0; r < Q5_1_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

void matmul_q4_1_repacked_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
									   size_t	   xq_row_stride_blocks, float *restrict y,
									   int y_row_stride, int n, int k, int m) {
	for (int t = 0; t < m; t++)
		matmul_q4_1_repacked_q8_qonly_f32_rows_range(w, xq + (size_t)t * xq_row_stride_blocks,
													 y + (size_t)t * y_row_stride, 0, n, k);
}

void matmul_q5_0_repacked_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
									   size_t	   xq_row_stride_blocks, float *restrict y,
									   int y_row_stride, int n, int k, int m) {
	for (int t = 0; t < m; t++)
		matmul_q5_0_repacked_q8_qonly_f32_rows_range(w, xq + (size_t)t * xq_row_stride_blocks,
													 y + (size_t)t * y_row_stride, 0, n, k);
}

void matmul_q5_1_repacked_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
									   size_t	   xq_row_stride_blocks, float *restrict y,
									   int y_row_stride, int n, int k, int m) {
	for (int t = 0; t < m; t++)
		matmul_q5_1_repacked_q8_qonly_f32_rows_range(w, xq + (size_t)t * xq_row_stride_blocks,
													 y + (size_t)t * y_row_stride, 0, n, k);
}

MATMUL_Q8_F32(q4_1_repacked_q8, q8_1_block, 32, quantize_q8_1, matmul_q4_1_repacked_q8_qonly_f32)
MATMUL_Q8_F32(q5_0_repacked_q8, q8_0_block, 32, quantize_q8_0, matmul_q5_0_repacked_q8_qonly_f32)
MATMUL_Q8_F32(q5_1_repacked_q8, q8_1_block, 32, quantize_q8_1, matmul_q5_1_repacked_q8_qonly_f32)

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
	case GGML_TYPE_Q4_1_REPACKED:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_0_REPACKED:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q5_1_REPACKED:
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
		case GGML_TYPE_Q4_1_REPACKED:
			matmul_q4_1_repacked_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q5_0:
			matmul_q5_0_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q5_0_REPACKED:
			matmul_q5_0_repacked_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q5_1:
			matmul_q5_1_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q5_1_REPACKED:
			matmul_q5_1_repacked_q8_f32(w, x, y, n, k, &qs);
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

MK_SCALAR_ROW_DRIVER(matmul_q4_q8_qonly_f32_row, q4_0_block, q8_0_block, 32, scalar_q4_0_dot)

MK_SCALAR_ROW_DRIVER(matmul_q8_0_q8_qonly_f32_row, q8_0_block, q8_0_block, 32, scalar_q8_0_dot)

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

MK_SCALAR_ROW_DRIVER(matmul_iq4_nl_q8_qonly_f32_row, iq4_nl_block, q8_0_block, 32,
					 scalar_iq4_nl_dot)

void matmul_iq4_nl_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_0_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(iq4_nl_block)), (1),
						  (matmul_qonly_row_fn)matmul_iq4_nl_q8_qonly_f32_row);
}

MK_SCALAR_ROW_DRIVER(matmul_q4_1_q8_qonly_f32_row, q4_1_block, q8_1_block, 32, scalar_q4_1_dot)

void matmul_q4_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_1_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q4_1_block)), (1),
						  (matmul_qonly_row_fn)matmul_q4_1_q8_qonly_f32_row);
}

MK_SCALAR_ROW_DRIVER(matmul_q5_0_q8_qonly_f32_row, q5_0_block, q8_0_block, 32, scalar_q5_0_dot)

MATMUL_Q8_F32(q5_0_q8, q8_0_block, 32, quantize_q8_0, matmul_q5_0_q8_qonly_f32)

void matmul_q5_0_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_0_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q5_0_block)), (1),
						  (matmul_qonly_row_fn)matmul_q5_0_q8_qonly_f32_row);
}

MK_SCALAR_ROW_DRIVER(matmul_q5_1_q8_qonly_f32_row, q5_1_block, q8_1_block, 32, scalar_q5_1_dot)

MATMUL_Q8_F32(q5_1_q8, q8_1_block, 32, quantize_q8_1, matmul_q5_1_q8_qonly_f32)

void matmul_q5_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_1_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 32) * sizeof(q5_1_block)), (1),
						  (matmul_qonly_row_fn)matmul_q5_1_q8_qonly_f32_row);
}

MK_SCALAR_ROW_DRIVER(matmul_q6_k_q8_qonly_f32_row, q6_k_block, q8_k_block, 256, scalar_q6_k_dot)

void matmul_q6_k_q8_qonly_f32(const void *w, const q8_k_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * sizeof(q6_k_block)), (1),
						  (matmul_qonly_row_fn)matmul_q6_k_q8_qonly_f32_row);
}

MK_SCALAR_ROW_DRIVER(matmul_q4_k_q8_k_qonly_f32_row, q4_k_block, q8_k_block, 256, scalar_q4_k_dot)

void matmul_q4_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * sizeof(q4_k_block)), (1),
						  (matmul_qonly_row_fn)matmul_q4_k_q8_k_qonly_f32_row);
}

MK_SCALAR_ROW_DRIVER(matmul_q5_k_q8_k_qonly_f32_row, q5_k_block, q8_k_block, 256, scalar_q5_k_dot)

void matmul_q5_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	matmul_qonly_dispatch(w, xq, sizeof(q8_k_block), xq_row_stride_blocks, y, y_row_stride, n, k, m,
						  (((size_t)k / 256) * sizeof(q5_k_block)), (1),
						  (matmul_qonly_row_fn)matmul_q5_k_q8_k_qonly_f32_row);
}

void matmul_f32_f32(const float *restrict w, const float *restrict x, float *restrict y, int n,
					int k) {
	for (int i = 0; i < n; i++)
		y[i] = num_dot_f32(w + ((size_t)i * k), x, k);
}

void matmul_bf16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					 int k) {
	const uint16_t *wb = w;
	for (int i = 0; i < n; i++)
		y[i] = num_dot_f32_bf16(x, wb + ((size_t)i * k), k);
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
	for (int t = 0; t < m; t++)
		matmul_bf16_f32(w, x + (size_t)t * x_row_stride, y + (size_t)t * y_row_stride, n, k);
}

void matmul_f32_f32_batch(const float *restrict w, const float *restrict x, float *restrict y,
						  int n, int k, int m, int x_row_stride, int y_row_stride) {
	matmul_panel_batch(w, x, y, n, k, m, x_row_stride, y_row_stride, (size_t)k * sizeof(float),
					   matmul_f32_row_thunk);
}

void matmul_f16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					int k) {
	const uint16_t *wb = w;
	for (int i = 0; i < n; i++)
		y[i] = num_dot_f32_f16(x, wb + ((size_t)i * k), k);
}

void matmul_f16_f32_batch(const void *restrict w, const float *restrict x, float *restrict y, int n,
						  int k, int m, int x_row_stride, int y_row_stride) {
	for (int t = 0; t < m; t++)
		matmul_f16_f32(w, x + (size_t)t * x_row_stride, y + (size_t)t * y_row_stride, n, k);
}

float dot_f32(const float *restrict a, const float *restrict b, int n) {
	return num_dot_f32(a, b, n);
}

void rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
	float ss	= rmsnorm_sum_sq(x, n);
	float scale = num_rms_scale(ss, n, eps);
	for (int i = 0; i < n; i++)
		y[i] = x[i] * scale * w[i];
}

void rmsnorm_noweight(const float *x, float *y, int n, float eps) {
	float ss	= rmsnorm_sum_sq(x, n);
	float scale = num_rms_scale(ss, n, eps);
	for (int i = 0; i < n; i++)
		y[i] = x[i] * scale;
}

void rmsnorm_per_head(const float *x, const float *w, float *y, int n_heads, int head_dim,
					  float eps) {
	for (int h = 0; h < n_heads; h++) {
		const float *xh	   = x + ((size_t)h * head_dim);
		float		*yh	   = y + ((size_t)h * head_dim);
		float		 ss	   = rmsnorm_sum_sq(xh, head_dim);
		float		 scale = num_rms_scale(ss, head_dim, eps);
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
		scores[i] = num_expf(scores[i] - mx);
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

void matmul_iq2_s_q8_k_qonly_f32(const void *w, const q8_k_block *x, size_t xs, float *y, int ys,
								 int n, int k, int m) {
	iq_qonly_scalar(w, sizeof(iq2_s_block), iq2_s_block_tot, 0.125f, x, xs ? xs : (size_t)(k / 256),
					y, ys, n, k, m);
}
void matmul_iq2_xxs_q8_k_qonly_f32(const void *w, const q8_k_block *x, size_t xs, float *y, int ys,
								   int n, int k, int m) {
	iq_qonly_scalar(w, sizeof(iq2_xxs_block), iq2_xxs_block_tot, 0.125f, x,
					xs ? xs : (size_t)(k / 256), y, ys, n, k, m);
}
void matmul_iq3_xxs_q8_k_qonly_f32(const void *w, const q8_k_block *x, size_t xs, float *y, int ys,
								   int n, int k, int m) {
	iq_qonly_scalar(w, sizeof(iq3_xxs_block), iq3_xxs_block_tot, 0.25f, x,
					xs ? xs : (size_t)(k / 256), y, ys, n, k, m);
}
void matmul_iq4_xs_q8_k_qonly_f32(const void *w, const q8_k_block *x, size_t xs, float *y, int ys,
								  int n, int k, int m) {
	iq_qonly_scalar(w, sizeof(iq4_xs_block), iq4_xs_block_tot, 1.0f, x, xs ? xs : (size_t)(k / 256),
					y, ys, n, k, m);
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
		mem[i] = fmaf(row[i], ks, mem[i]);
	}
}

void gdn_update_accum_f32(float *restrict row, float *restrict y, const float *restrict delta,
						  float ks, float qs, int n) {
	for (int i = 0; i < n; i++) {
		row[i] = fmaf(ks, delta[i], row[i]);
		y[i]   = fmaf(row[i], qs, y[i]);
	}
}

void gdn_out_norm_f32(float *restrict y, const float *restrict norm_w, const float *restrict z,
					  float inv_rms, int n) {
	for (int i = 0; i < n; i++)
		y[i] = y[i] * inv_rms * norm_w[i] * silu(z[i]);
}
