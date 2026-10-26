#include "backend/cpu/common.h"
#include "backend/backend.h"
#include "memconfig.h"
#include "model.h"
#include "moe/moe_stream.h"
#include "threadpool.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void dequant_q3_k_row(const void *blocks, size_t n_blocks, float *dst) {
	const q3_k_block *b		 = blocks;
	const uint32_t	  kmask1 = 0x03030303;
	const uint32_t	  kmask2 = 0x0f0f0f0f;

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const float d_all = f16_to_f32(b[bi].d);

		const uint8_t *q  = b[bi].qs;
		const uint8_t *hm = b[bi].hmask;
		uint8_t		   m  = 1;

		uint32_t	  aux[4];
		const int8_t *scales = (const int8_t *)aux;
		memcpy(aux, b[bi].scales, 12);
		uint32_t tmp = aux[2];
		aux[2]		 = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
		aux[3]		 = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
		aux[0]		 = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
		aux[1]		 = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

		float *y  = dst + (bi * 256);
		int	   is = 0;
		float  dl;
		for (int n = 0; n < 256; n += 128) {
			int shift = 0;
			for (int j = 0; j < 4; j++) {
				dl = d_all * (scales[is++] - 32);
				for (int l = 0; l < 16; l++)
					*y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));

				dl = d_all * (scales[is++] - 32);
				for (int l = 0; l < 16; l++)
					*y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));

				shift += 2;
				m <<= 1;
			}
			q += 32;
		}
	}
}

void dequant_q4_k_row(const void *blocks, size_t n_blocks, float *dst) {
	const q4_k_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float		   d   = f16_to_f32(b[bi].d);
		float		   min = f16_to_f32(b[bi].dmin);
		const uint8_t *q   = b[bi].qs;
		const uint8_t *s   = b[bi].scales;
		float		  *y   = dst + (bi * 256);
		int			   is  = 0;
		uint8_t		   sc;
		uint8_t		   m;
		for (int j = 0; j < 256; j += 64) {
			get_scale_min_k4(is + 0, s, &sc, &m);
			float d1 = d * sc;
			float m1 = min * m;
			get_scale_min_k4(is + 1, s, &sc, &m);
			float d2 = d * sc;
			float m2 = min * m;
			for (int l = 0; l < 32; l++) {
				y[j + l]	  = (d1 * (q[l] & 0xF)) - m1;
				y[j + l + 32] = (d2 * (q[l] >> 4)) - m2;
			}
			q += 32;
			is += 2;
		}
	}
}

void dequant_q5_k_row(const void *blocks, size_t n_blocks, float *dst) {
	const q5_k_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float		   d   = f16_to_f32(b[bi].d);
		float		   min = f16_to_f32(b[bi].dmin);
		const uint8_t *ql  = b[bi].qs;
		const uint8_t *qh  = b[bi].qh;
		const uint8_t *s   = b[bi].scales;
		float		  *y   = dst + (bi * 256);
		int			   is  = 0;
		uint8_t		   sc;
		uint8_t		   m;
		uint8_t		   u1 = 1;
		uint8_t		   u2 = 2;
		for (int j = 0; j < 256; j += 64) {
			get_scale_min_k4(is + 0, s, &sc, &m);
			float d1 = d * sc;
			float m1 = min * m;
			get_scale_min_k4(is + 1, s, &sc, &m);
			float d2 = d * sc;
			float m2 = min * m;
			for (int l = 0; l < 32; l++) {
				y[j + l]	  = (d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0))) - m1;
				y[j + l + 32] = (d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0))) - m2;
			}
			ql += 32;
			is += 2;
			u1 <<= 2;
			u2 <<= 2;
		}
	}
}

void dequant_q6_k_row(const void *blocks, size_t n_blocks, float *dst) {
	const q6_k_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float		   d  = f16_to_f32(b[bi].d);
		const uint8_t *ql = b[bi].ql;
		const uint8_t *qh = b[bi].qh;
		const int8_t  *sc = (const int8_t *)b[bi].scales;
		float		  *y  = dst + (bi * 256);
		for (int n_iter = 0; n_iter < 2; n_iter++) {
			for (int l = 0; l < 32; l++) {
				int	   is = l / 16;
				int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
				int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
				int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				y[l]	  = d * sc[is + 0] * q1;
				y[l + 32] = d * sc[is + 2] * q2;
				y[l + 64] = d * sc[is + 4] * q3;
				y[l + 96] = d * sc[is + 6] * q4;
			}
			y += 128;
			ql += 64;
			qh += 32;
			sc += 8;
		}
	}
}

MATMUL_Q8_F32(iq3_s_q8_k, q8_k_block, 256, quantize_q8_k, matmul_iq3_s_q8_k_qonly_f32)

static inline uint8_t iq3s_encode_val(uint8_t magnitude, int sign) {

	uint8_t idx = (magnitude - 1) / 2;
	if (sign)
		idx |= 8;
	return idx;
}

static uint16_t iq3s_enc4[512][16];

static void __attribute__((constructor)) iq3s_enc4_init(void) {
	for (int g = 0; g < 512; g++) {
		const uint8_t *mag = (const uint8_t *)&ggml_iq3s_grid[g];
		for (int sig = 0; sig < 16; sig++) {
			uint16_t v = 0;
			for (int j = 0; j < 4; j++) {
				uint8_t c = iq3s_encode_val(mag[j], (sig >> j) & 1);
				v |= (uint16_t)c << (4 * j);
			}
			iq3s_enc4[g][sig] = v;
		}
	}
}

static inline void iq3s_encode_half(const uint8_t *qs, uint8_t qh, const uint8_t *signs,
									uint8_t *idx_out) {
	const uint16_t *restrict e = &iq3s_enc4[0][0];

	const uint16_t g1c0 = e[(qs[0] | (((qh >> 0) & 1) << 8)) * 16 + (signs[0] & 0xF)];
	const uint16_t g2c0 = e[(qs[1] | (((qh >> 1) & 1) << 8)) * 16 + (signs[0] >> 4)];
	const uint16_t g1c1 = e[(qs[2] | (((qh >> 2) & 1) << 8)) * 16 + (signs[1] & 0xF)];
	const uint16_t g2c1 = e[(qs[3] | (((qh >> 3) & 1) << 8)) * 16 + (signs[1] >> 4)];
	const uint16_t g1c2 = e[(qs[4] | (((qh >> 4) & 1) << 8)) * 16 + (signs[2] & 0xF)];
	const uint16_t g2c2 = e[(qs[5] | (((qh >> 5) & 1) << 8)) * 16 + (signs[2] >> 4)];
	const uint16_t g1c3 = e[(qs[6] | (((qh >> 6) & 1) << 8)) * 16 + (signs[3] & 0xF)];
	const uint16_t g2c3 = e[(qs[7] | (((qh >> 7) & 1) << 8)) * 16 + (signs[3] >> 4)];

	idx_out[0]	= (g1c0 & 0xF) | ((g1c2 & 0xF) << 4);
	idx_out[1]	= ((g1c0 >> 4) & 0xF) | (((g1c2 >> 4) & 0xF) << 4);
	idx_out[2]	= ((g1c0 >> 8) & 0xF) | (((g1c2 >> 8) & 0xF) << 4);
	idx_out[3]	= ((g1c0 >> 12) & 0xF) | (((g1c2 >> 12) & 0xF) << 4);
	idx_out[4]	= (g2c0 & 0xF) | ((g2c2 & 0xF) << 4);
	idx_out[5]	= ((g2c0 >> 4) & 0xF) | (((g2c2 >> 4) & 0xF) << 4);
	idx_out[6]	= ((g2c0 >> 8) & 0xF) | (((g2c2 >> 8) & 0xF) << 4);
	idx_out[7]	= ((g2c0 >> 12) & 0xF) | (((g2c2 >> 12) & 0xF) << 4);
	idx_out[8]	= (g1c1 & 0xF) | ((g1c3 & 0xF) << 4);
	idx_out[9]	= ((g1c1 >> 4) & 0xF) | (((g1c3 >> 4) & 0xF) << 4);
	idx_out[10] = ((g1c1 >> 8) & 0xF) | (((g1c3 >> 8) & 0xF) << 4);
	idx_out[11] = ((g1c1 >> 12) & 0xF) | (((g1c3 >> 12) & 0xF) << 4);
	idx_out[12] = (g2c1 & 0xF) | ((g2c3 & 0xF) << 4);
	idx_out[13] = ((g2c1 >> 4) & 0xF) | (((g2c3 >> 4) & 0xF) << 4);
	idx_out[14] = ((g2c1 >> 8) & 0xF) | (((g2c3 >> 8) & 0xF) << 4);
	idx_out[15] = ((g2c1 >> 12) & 0xF) | (((g2c3 >> 12) & 0xF) << 4);
}

static inline void iq3s_encode_block(const iq3_s_block *b, uint16_t *d_out, uint8_t *scales_out,
									 uint8_t *idx_out) {
	*d_out = b->d;
	memcpy(scales_out, b->scales, 4);

	uint8_t		  *idx_ptr = idx_out;
	const uint8_t *qs	   = b->qs;
	const uint8_t *qh	   = b->qh;
	const uint8_t *signs   = b->signs;

	for (int ib32 = 0; ib32 < 8; ib32 += 2) {
		iq3s_encode_half(qs, qh[0], signs, idx_ptr);
		idx_ptr += 16;
		iq3s_encode_half(qs + 8, qh[1], signs + 4, idx_ptr);
		idx_ptr += 16;

		qh += 2;
		qs += 16;
		signs += 8;
	}
}

void repack_iq3_s_to_iq3_s_repacked_rows(const void *src, void *dst, int row_begin, int row_end,
										 int k) {
	const int	 blocks_per_row = k / 256;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(iq3_s_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % IQ3_S_REPACKED_ROWS);
	for (int g = group_begin; g < row_end; g += IQ3_S_REPACKED_ROWS) {
		const iq3_s_block *srow[IQ3_S_REPACKED_ROWS];
		for (int r = 0; r < IQ3_S_REPACKED_ROWS; r++)
			srow[r] = (const iq3_s_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * IQ3_S_REPACKED_BLOCK_BYTES;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * IQ3_S_REPACKED_GROUP_BYTES;

			uint16_t *dst_d		 = (uint16_t *)dblk;
			uint8_t	 *dst_scales = dblk + IQ3_S_REPACKED_ROWS * sizeof(uint16_t);
			uint8_t	 *dst_idx	 = dst_scales + IQ3_S_REPACKED_ROWS * 4;

			iq3s_encode_block(&srow[0][bi], dst_d + 0, dst_scales + 0, dst_idx + 0);
			iq3s_encode_block(&srow[1][bi], dst_d + 1, dst_scales + 4, dst_idx + 128);
			iq3s_encode_block(&srow[2][bi], dst_d + 2, dst_scales + 8, dst_idx + 256);
			iq3s_encode_block(&srow[3][bi], dst_d + 3, dst_scales + 12, dst_idx + 384);
			iq3s_encode_block(&srow[4][bi], dst_d + 4, dst_scales + 16, dst_idx + 512);
			iq3s_encode_block(&srow[5][bi], dst_d + 5, dst_scales + 20, dst_idx + 640);
			iq3s_encode_block(&srow[6][bi], dst_d + 6, dst_scales + 24, dst_idx + 768);
			iq3s_encode_block(&srow[7][bi], dst_d + 7, dst_scales + 28, dst_idx + 896);
		}
	}
}

void matmul_iq3_s_repacked_q8_k_qonly_f32_rows_range(const void *w, const q8_k_block *restrict xq,
													 float *restrict y, int row_begin, int row_end,
													 int k) {
	const int	   blocks_per_row = k / 256;
	const uint8_t *wb			  = w;

	for (int g = row_begin - (row_begin % IQ3_S_REPACKED_ROWS); g < row_end;
		 g += IQ3_S_REPACKED_ROWS) {
		const uint8_t *dgroup =
			wb + ((size_t)g / IQ3_S_REPACKED_ROWS) * blocks_per_row * IQ3_S_REPACKED_GROUP_BYTES;
		float sumf[IQ3_S_REPACKED_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk	   = dgroup + (size_t)bi * IQ3_S_REPACKED_GROUP_BYTES;
			const uint16_t *dst_d	   = (const uint16_t *)dblk;
			const uint8_t  *dst_scales = dblk + IQ3_S_REPACKED_ROWS * sizeof(uint16_t);
			const uint8_t  *dst_idx	   = dst_scales + IQ3_S_REPACKED_ROWS * 4;

			const q8_k_block *yb = &xq[bi];
			const int8_t	 *q8 = yb->qs;

			for (int r = 0; r < IQ3_S_REPACKED_ROWS; r++) {
				const uint8_t *scales = dst_scales + (size_t)r * 4;
				const uint8_t *idx	  = dst_idx + (size_t)r * 128;

				int32_t total = 0;
				for (int gr = 0; gr < 8; gr++) {
					uint8_t sb = scales[gr / 2];
					int		sc = (gr & 1) ? (1 + 2 * (sb >> 4)) : (1 + 2 * (sb & 0xf));

					const uint8_t *idx_g	 = idx + gr * 16;
					int32_t		   group_sum = 0;
					for (int j = 0; j < 32; j++) {
						uint8_t code = (j < 16) ? (idx_g[j] & 0xF) : (idx_g[j - 16] >> 4);
						int8_t	val	 = iq3_s_repacked_decode[code];
						group_sum += (int32_t)val * (int32_t)q8[gr * 32 + j];
					}
					total += sc * group_sum;
				}
				sumf[r] = fmaf(f16_to_f32(dst_d[r]) * yb->d, (float)total, sumf[r]);
			}
		}
		for (int r = 0; r < IQ3_S_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

MATMUL_Q8_F32(iq3_s_repacked_q8_k, q8_k_block, 256, quantize_q8_k,
			  matmul_iq3_s_repacked_q8_k_qonly_f32)

void dequant_q8_0_repacked_row(const void *repacked_buf, size_t n_blocks, float *dst) {
	const uint8_t *blk		   = repacked_buf;
	const size_t   group_bytes = (size_t)Q8_0_REPACKED_ROWS * sizeof(q8_0_block);

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const uint8_t  *b	   = blk + bi * group_bytes;
		const uint16_t *d8	   = (const uint16_t *)b;
		const int8_t   *qs_all = (const int8_t *)(b + Q8_0_REPACKED_ROWS * sizeof(uint16_t));

		for (int r = 0; r < Q8_0_REPACKED_ROWS; r++) {
			float d = f16_to_f32(d8[r]);
			for (int j = 0; j < 32; j++)
				dst[(r * 32) + j] = d * (float)qs_all[(size_t)r * 32 + j];
		}
	}
}

void repack_iq4_nl_to_iq4_nl_repacked_rows(const void *src, void *dst, int row_begin, int row_end,
										   int k) {
	const int	 blocks_per_row = k / 32;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(iq4_nl_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % IQ4_NL_REPACKED_ROWS);
	for (int g = group_begin; g < row_end; g += IQ4_NL_REPACKED_ROWS) {
		const iq4_nl_block *srow[IQ4_NL_REPACKED_ROWS];
		for (int r = 0; r < IQ4_NL_REPACKED_ROWS; r++)
			srow[r] = (const iq4_nl_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * sizeof(iq4_nl_block);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * IQ4_NL_REPACKED_ROWS * sizeof(iq4_nl_block);

			uint16_t *dst_d = (uint16_t *)dblk;
			for (int r = 0; r < IQ4_NL_REPACKED_ROWS; r++)
				dst_d[r] = srow[r][bi].d;

			uint8_t *dst_qs = dblk + IQ4_NL_REPACKED_ROWS * sizeof(uint16_t);
			for (int r = 0; r < IQ4_NL_REPACKED_ROWS; r++)
				memcpy(dst_qs + (size_t)r * 16, srow[r][bi].qs, 16);
		}
	}
}

void matmul_iq4_nl_repacked_q8_qonly_f32_rows_range(const void *w, const q8_0_block *restrict xq,
													float *restrict y, int row_begin, int row_end,
													int k) {
	const int	   blocks_per_row = k / 32;
	const size_t   group_bytes	  = (size_t)IQ4_NL_REPACKED_ROWS * sizeof(iq4_nl_block);
	const uint8_t *wb			  = w;

	for (int g = row_begin - (row_begin % IQ4_NL_REPACKED_ROWS); g < row_end;
		 g += IQ4_NL_REPACKED_ROWS) {
		const uint8_t *dgroup = wb + (size_t)g * blocks_per_row * sizeof(iq4_nl_block);
		float		   sumf[IQ4_NL_REPACKED_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk	   = dgroup + (size_t)bi * group_bytes;
			const uint16_t *d_w		   = (const uint16_t *)dblk;
			const uint8_t  *qs_w	   = dblk + IQ4_NL_REPACKED_ROWS * sizeof(uint16_t);
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);

			for (int r = 0; r < IQ4_NL_REPACKED_ROWS; r++) {
				const uint8_t *row_qs = qs_w + (size_t)r * 16;
				int32_t		   sumi	  = 0;
				for (int j = 0; j < 16; j++) {
					sumi += (int32_t)kvalues_iq4nl[row_qs[j] & 0xF] * (int32_t)xq8[j];
					sumi += (int32_t)kvalues_iq4nl[row_qs[j] >> 4] * (int32_t)xq8[j + 16];
				}
				sumf[r] = fmaf(f16_to_f32(d_w[r]) * d_xq, (float)sumi, sumf[r]);
			}
		}
		for (int r = 0; r < IQ4_NL_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

MATMUL_Q8_F32(iq4_nl_repacked_q8, q8_0_block, 32, quantize_q8_0,
			  matmul_iq4_nl_repacked_q8_qonly_f32)

void repack_q8_0_to_q8_0_repacked_rows(const void *src, void *dst, int row_begin, int row_end,
									   int k) {
	const int	 blocks_per_row = k / 32;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(q8_0_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % Q8_0_REPACKED_ROWS);
	for (int g = group_begin; g < row_end; g += Q8_0_REPACKED_ROWS) {
		const q8_0_block *srow[Q8_0_REPACKED_ROWS];
		for (int r = 0; r < Q8_0_REPACKED_ROWS; r++)
			srow[r] = (const q8_0_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * sizeof(q8_0_block);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * Q8_0_REPACKED_ROWS * sizeof(q8_0_block);

			uint16_t *dst_d = (uint16_t *)dblk;
			for (int r = 0; r < Q8_0_REPACKED_ROWS; r++)
				dst_d[r] = srow[r][bi].d;

			int8_t *dst_qs = (int8_t *)(dblk + Q8_0_REPACKED_ROWS * sizeof(uint16_t));
			for (int r = 0; r < Q8_0_REPACKED_ROWS; r++)
				memcpy(dst_qs + (size_t)r * 32, srow[r][bi].qs, 32);
		}
	}
}

void matmul_q8_0_repacked_q8_qonly_f32_rows_range(const void *w, const q8_0_block *restrict xq,
												  float *restrict y, int row_begin, int row_end,
												  int k) {
	const int	   blocks_per_row = k / 32;
	const size_t   group_bytes	  = (size_t)Q8_0_REPACKED_ROWS * sizeof(q8_0_block);
	const uint8_t *wb			  = w;

	for (int g = row_begin - (row_begin % Q8_0_REPACKED_ROWS); g < row_end;
		 g += Q8_0_REPACKED_ROWS) {
		const uint8_t *dgroup = wb + (size_t)g * blocks_per_row * sizeof(q8_0_block);
		float		   sumf[Q8_0_REPACKED_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk = dgroup + (size_t)bi * group_bytes;
			const uint16_t *d_w	 = (const uint16_t *)dblk;
			const int8_t   *qs_w = (const int8_t *)(dblk + Q8_0_REPACKED_ROWS * sizeof(uint16_t));
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);

			for (int r = 0; r < Q8_0_REPACKED_ROWS; r++) {
				const int8_t *row_qs = qs_w + (size_t)r * 32;
				int32_t		  sumi	 = 0;
				for (int j = 0; j < 32; j++)
					sumi += (int32_t)row_qs[j] * (int32_t)xq8[j];
				sumf[r] = fmaf(f16_to_f32(d_w[r]) * d_xq, (float)sumi, sumf[r]);
			}
		}
		for (int r = 0; r < Q8_0_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

MATMUL_Q8_F32(q8_0_repacked_q8, q8_0_block, 32, quantize_q8_0, matmul_q8_0_repacked_q8_qonly_f32)

void repack_q4_0_to_q4_0_repacked_rows(const void *src, void *dst, int row_begin, int row_end,
									   int k) {
	const int	 blocks_per_row = k / 32;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(q4_0_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % Q4_0_REPACKED_ROWS);
	for (int g = group_begin; g < row_end; g += Q4_0_REPACKED_ROWS) {
		const q4_0_block *srow[Q4_0_REPACKED_ROWS];
		for (int r = 0; r < Q4_0_REPACKED_ROWS; r++)
			srow[r] = (const q4_0_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * sizeof(q4_0_block);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * Q4_0_REPACKED_ROWS * sizeof(q4_0_block);

			uint16_t *dst_d = (uint16_t *)dblk;
			for (int r = 0; r < Q4_0_REPACKED_ROWS; r++)
				dst_d[r] = srow[r][bi].d;

			uint8_t *dst_qs = dblk + Q4_0_REPACKED_ROWS * sizeof(uint16_t);
			for (int r = 0; r < Q4_0_REPACKED_ROWS; r++)
				memcpy(dst_qs + (size_t)r * 16, srow[r][bi].qs, 16);
		}
	}
}

static void q4_k_expand_scales(const uint8_t *restrict sc, uint8_t *restrict se) {
	for (int is = 0; is < 8; is++) {
		uint8_t s, m;
		get_scale_min_k4(is, sc, &s, &m);
		se[is * 2 + 0] = s;
		se[is * 2 + 1] = m;
	}
}

void repack_q4_k_to_q4_k_repacked_rows(const void *src, void *dst, int row_begin, int row_end,
									   int k) {
	const int	 blocks_per_row = k / 256;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(q4_k_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % Q4_K_REPACKED_ROWS);
	for (int g = group_begin; g < row_end; g += Q4_K_REPACKED_ROWS) {
		const q4_k_block *srow[Q4_K_REPACKED_ROWS];
		for (int r = 0; r < Q4_K_REPACKED_ROWS; r++)
			srow[r] = (const q4_k_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * Q4_K_REPACKED_ROW_BYTES;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * Q4_K_REPACKED_GROUP_BYTES;

			uint16_t *dst_d	 = (uint16_t *)(dblk + Q4_K_REPACKED_OFF_D);
			uint16_t *dst_m	 = (uint16_t *)(dblk + Q4_K_REPACKED_OFF_DMIN);
			uint8_t	 *dst_se = dblk + Q4_K_REPACKED_OFF_SE;
			uint8_t	 *dst_q	 = dblk + Q4_K_REPACKED_OFF_QS;
			for (int r = 0; r < Q4_K_REPACKED_ROWS; r++) {
				dst_d[r] = srow[r][bi].d;
				dst_m[r] = srow[r][bi].dmin;
				q4_k_expand_scales(srow[r][bi].scales, dst_se + (size_t)r * 16);
				memcpy(dst_q + (size_t)r * 128, srow[r][bi].qs, 128);
			}
		}
	}
}

void repack_q5_k_to_q5_k_repacked_rows(const void *src, void *dst, int row_begin, int row_end,
									   int k) {
	const int	 blocks_per_row = k / 256;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(q5_k_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % Q5_K_REPACKED_ROWS);
	for (int g = group_begin; g < row_end; g += Q5_K_REPACKED_ROWS) {
		const q5_k_block *srow[Q5_K_REPACKED_ROWS];
		for (int r = 0; r < Q5_K_REPACKED_ROWS; r++)
			srow[r] = (const q5_k_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * Q5_K_REPACKED_ROW_BYTES;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * Q5_K_REPACKED_GROUP_BYTES;

			uint16_t *dst_d	 = (uint16_t *)(dblk + Q5_K_REPACKED_OFF_D);
			uint16_t *dst_m	 = (uint16_t *)(dblk + Q5_K_REPACKED_OFF_DMIN);
			uint8_t	 *dst_se = dblk + Q5_K_REPACKED_OFF_SE;
			uint8_t	 *dst_qh = dblk + Q5_K_REPACKED_OFF_QH;
			uint8_t	 *dst_qs = dblk + Q5_K_REPACKED_OFF_QS;
			for (int r = 0; r < Q5_K_REPACKED_ROWS; r++) {
				dst_d[r] = srow[r][bi].d;
				dst_m[r] = srow[r][bi].dmin;
				q4_k_expand_scales(srow[r][bi].scales, dst_se + (size_t)r * 16);
				memcpy(dst_qh + (size_t)r * 32, srow[r][bi].qh, 32);
				memcpy(dst_qs + (size_t)r * 128, srow[r][bi].qs, 128);
			}
		}
	}
}

void repack_q6_k_to_q6_k_repacked_rows(const void *src, void *dst, int row_begin, int row_end,
									   int k) {
	const int	 blocks_per_row = k / 256;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(q6_k_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % Q6_K_REPACKED_ROWS);
	for (int g = group_begin; g < row_end; g += Q6_K_REPACKED_ROWS) {
		const q6_k_block *srow[Q6_K_REPACKED_ROWS];
		for (int r = 0; r < Q6_K_REPACKED_ROWS; r++)
			srow[r] = (const q6_k_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * sizeof(q6_k_block);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * Q6_K_REPACKED_GROUP_BYTES;

			uint8_t	 *dst_ql = dblk + Q6_K_REPACKED_OFF_QL;
			uint8_t	 *dst_qh = dblk + Q6_K_REPACKED_OFF_QH;
			int8_t	 *dst_sc = (int8_t *)(dblk + Q6_K_REPACKED_OFF_SCALES);
			uint16_t *dst_d	 = (uint16_t *)(dblk + Q6_K_REPACKED_OFF_D);
			for (int r = 0; r < Q6_K_REPACKED_ROWS; r++) {
				memcpy(dst_ql + (size_t)r * 128, srow[r][bi].ql, 128);
				memcpy(dst_qh + (size_t)r * 64, srow[r][bi].qh, 64);
				memcpy(dst_sc + (size_t)r * 16, srow[r][bi].scales, 16);
				dst_d[r] = srow[r][bi].d;
			}
		}
	}
}

void matmul_q4_k_repacked_q8_k_qonly_f32_rows_range(const void *w, const q8_k_block *restrict xq,
													float *restrict y, int row_begin, int row_end,
													int k) {
	const int	   blocks_per_row = k / 256;
	const uint8_t *wb			  = w;

	for (int g = row_begin - (row_begin % Q4_K_REPACKED_ROWS); g < row_end;
		 g += Q4_K_REPACKED_ROWS) {
		const uint8_t *dgroup =
			wb + ((size_t)g / Q4_K_REPACKED_ROWS) * blocks_per_row * Q4_K_REPACKED_GROUP_BYTES;
		float sumf[Q4_K_REPACKED_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk = dgroup + (size_t)bi * Q4_K_REPACKED_GROUP_BYTES;
			const uint16_t *ds	 = (const uint16_t *)(dblk + Q4_K_REPACKED_OFF_D);
			const uint16_t *ms	 = (const uint16_t *)(dblk + Q4_K_REPACKED_OFF_DMIN);
			const uint8_t  *qss	 = dblk + Q4_K_REPACKED_OFF_QS;

			const q8_k_block *restrict xb = &xq[bi];
			for (int r = 0; r < Q4_K_REPACKED_ROWS; r++) {
				const uint8_t *se	= dblk + Q4_K_REPACKED_OFF_SE + (size_t)r * 16;
				const uint8_t *qb	= qss + (size_t)r * 128;
				float		   d	= f16_to_f32(ds[r]);
				float		   dmin = f16_to_f32(ms[r]);
				int32_t		   sumi = 0, summ = 0;
				int			   is = 0, ib = 0;
				for (int g = 0; g < 4; g++) {
					int s0 = se[is * 2 + 0], m0 = se[is * 2 + 1];
					int s1 = se[is * 2 + 2], m1 = se[is * 2 + 3];
					const uint8_t *restrict qg = qb + (g * 32);
					int32_t dot0 = 0, dot1 = 0;
					for (int l = 0; l < 32; l++) {
						dot0 += (int32_t)(qg[l] & 0xF) * (int32_t)xb->qs[g * 64 + l];
						dot1 += (int32_t)(qg[l] >> 4) * (int32_t)xb->qs[g * 64 + 32 + l];
					}
					sumi += (s0 * dot0) + (s1 * dot1);
					summ += m0 * (int)(xb->bsums[ib] + xb->bsums[ib + 1]);
					ib += 2;
					summ += m1 * (int)(xb->bsums[ib] + xb->bsums[ib + 1]);
					ib += 2;
					is += 2;
				}
				sumf[r] += xb->d * ((d * (float)sumi) - (dmin * (float)summ));
			}
		}
		for (int r = 0; r < Q4_K_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

void matmul_q5_k_repacked_q8_k_qonly_f32_rows_range(const void *w, const q8_k_block *restrict xq,
													float *restrict y, int row_begin, int row_end,
													int k) {
	const int	   blocks_per_row = k / 256;
	const uint8_t *wb			  = w;

	for (int g = row_begin - (row_begin % Q5_K_REPACKED_ROWS); g < row_end;
		 g += Q5_K_REPACKED_ROWS) {
		const uint8_t *dgroup =
			wb + ((size_t)g / Q5_K_REPACKED_ROWS) * blocks_per_row * Q5_K_REPACKED_GROUP_BYTES;
		float sumf[Q5_K_REPACKED_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk = dgroup + (size_t)bi * Q5_K_REPACKED_GROUP_BYTES;
			const uint16_t *ds	 = (const uint16_t *)(dblk + Q5_K_REPACKED_OFF_D);
			const uint16_t *ms	 = (const uint16_t *)(dblk + Q5_K_REPACKED_OFF_DMIN);
			const uint8_t  *qhs	 = dblk + Q5_K_REPACKED_OFF_QH;
			const uint8_t  *qss	 = dblk + Q5_K_REPACKED_OFF_QS;

			const q8_k_block *restrict xb = &xq[bi];
			for (int r = 0; r < Q5_K_REPACKED_ROWS; r++) {
				const uint8_t *se	= dblk + Q5_K_REPACKED_OFF_SE + (size_t)r * 16;
				const uint8_t *qh	= qhs + (size_t)r * 32;
				const uint8_t *qb	= qss + (size_t)r * 128;
				float		   d	= f16_to_f32(ds[r]);
				float		   dmin = f16_to_f32(ms[r]);
				int32_t		   sumi = 0, summ = 0;
				int			   is = 0, ib = 0;
				uint8_t		   u1 = 1, u2 = 2;
				for (int g = 0; g < 4; g++) {
					int s0 = se[is * 2 + 0], m0 = se[is * 2 + 1];
					int s1 = se[is * 2 + 2], m1 = se[is * 2 + 3];
					const uint8_t *restrict qg = qb + (g * 32);
					int32_t dot0 = 0, dot1 = 0;
					for (int l = 0; l < 32; l++) {
						int hi0 = (qh[l] & u1) ? 16 : 0;
						int hi1 = (qh[l] & u2) ? 16 : 0;
						dot0 += (int32_t)((qg[l] & 0xF) + hi0) * (int32_t)xb->qs[g * 64 + l];
						dot1 += (int32_t)((qg[l] >> 4) + hi1) * (int32_t)xb->qs[g * 64 + 32 + l];
					}
					sumi += (s0 * dot0) + (s1 * dot1);
					summ += m0 * (int)(xb->bsums[ib] + xb->bsums[ib + 1]);
					ib += 2;
					summ += m1 * (int)(xb->bsums[ib] + xb->bsums[ib + 1]);
					ib += 2;
					is += 2;
					u1 <<= 2;
					u2 <<= 2;
				}
				sumf[r] += xb->d * ((d * (float)sumi) - (dmin * (float)summ));
			}
		}
		for (int r = 0; r < Q5_K_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

static float q6_k_repacked_dot(const uint8_t *ql, const uint8_t *qh, const int8_t *sc,
							   uint16_t d_raw, const int8_t *xq8, float xd, float acc) {
	float  d = f16_to_f32(d_raw) * xd;
	int8_t a[256];
	int8_t *restrict ap = a;
	for (int n_iter = 0; n_iter < 2; n_iter++) {
		for (int l = 0; l < 32; l++) {
			ap[l]	   = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
			ap[l + 32] = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
			ap[l + 64] = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
			ap[l + 96] = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
		}
		ap += 128;
		ql += 64;
		qh += 32;
	}

	ap = a;

	int32_t total = 0;
	for (int j = 0; j < 16; j++) {
		int32_t l0 = 0;
		int32_t l1 = 0;
		for (int l = 0; l < 8; l++) {
			l0 += (int)xq8[l] * (int)ap[l];
			l1 += (int)xq8[l + 8] * (int)ap[l + 8];
		}
		total += sc[j] * (l0 + l1);
		xq8 += 16;
		ap += 16;
	}
	return acc + d * (float)total;
}

void matmul_q6_k_repacked_q8_k_qonly_f32_rows_range(const void *w, const q8_k_block *restrict xq,
													float *restrict y, int row_begin, int row_end,
													int k) {
	const int	   blocks_per_row = k / 256;
	const uint8_t *wb			  = w;

	for (int g = row_begin - (row_begin % Q6_K_REPACKED_ROWS); g < row_end;
		 g += Q6_K_REPACKED_ROWS) {
		const uint8_t *dgroup =
			wb + ((size_t)g / Q6_K_REPACKED_ROWS) * blocks_per_row * Q6_K_REPACKED_GROUP_BYTES;
		float sumf[Q6_K_REPACKED_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk = dgroup + (size_t)bi * Q6_K_REPACKED_GROUP_BYTES;
			const uint8_t  *qls	 = dblk + Q6_K_REPACKED_OFF_QL;
			const uint8_t  *qhs	 = dblk + Q6_K_REPACKED_OFF_QH;
			const int8_t   *scs	 = (const int8_t *)(dblk + Q6_K_REPACKED_OFF_SCALES);
			const uint16_t *ds	 = (const uint16_t *)(dblk + Q6_K_REPACKED_OFF_D);

			const q8_k_block *restrict xb = &xq[bi];
			for (int r = 0; r < Q6_K_REPACKED_ROWS; r++) {
				sumf[r] = q6_k_repacked_dot(qls + (size_t)r * 128, qhs + (size_t)r * 64,
											scs + (size_t)r * 16, ds[r], xb->qs, xb->d, sumf[r]);
			}
		}
		for (int r = 0; r < Q6_K_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

void matmul_q4_0_repacked_q8_qonly_f32_rows_range(const void *w, const q8_0_block *restrict xq,
												  float *restrict y, int row_begin, int row_end,
												  int k) {
	const int	   blocks_per_row = k / 32;
	const size_t   group_bytes	  = (size_t)Q4_0_REPACKED_ROWS * sizeof(q4_0_block);
	const uint8_t *wb			  = w;

	for (int g = row_begin - (row_begin % Q4_0_REPACKED_ROWS); g < row_end;
		 g += Q4_0_REPACKED_ROWS) {
		const uint8_t *dgroup = wb + (size_t)g * blocks_per_row * sizeof(q4_0_block);
		float		   sumf[Q4_0_REPACKED_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk	   = dgroup + (size_t)bi * group_bytes;
			const uint16_t *d_w		   = (const uint16_t *)dblk;
			const uint8_t  *qs_w	   = dblk + Q4_0_REPACKED_ROWS * sizeof(uint16_t);
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);

			for (int r = 0; r < Q4_0_REPACKED_ROWS; r++) {
				const uint8_t *row_qs = qs_w + (size_t)r * 16;
				int32_t		   sumi	  = 0;
				for (int j = 0; j < 16; j++) {
					sumi += ((int32_t)(row_qs[j] & 0xF) - 8) * (int32_t)xq8[j];
					sumi += ((int32_t)(row_qs[j] >> 4) - 8) * (int32_t)xq8[j + 16];
				}
				sumf[r] = fmaf(f16_to_f32(d_w[r]) * d_xq, (float)sumi, sumf[r]);
			}
		}
		for (int r = 0; r < Q4_0_REPACKED_ROWS; r++) {
			int row = g + r;
			if (row >= row_begin && row < row_end)
				y[row] = sumf[r];
		}
	}
}

MATMUL_Q8_F32(q4_0_repacked_q8, q8_0_block, 32, quantize_q8_0, matmul_q4_0_repacked_q8_qonly_f32)

static void dequant_f32_cells(const void *src, size_t n, float *dst) {
	memcpy(dst, src, n * sizeof(float));
}

static void dequant_f16_cells(const void *src, size_t n, float *dst) {
	dequant_f16_row(src, (int)n, dst);
}

static void dequant_bf16_cells(const void *src, size_t n, float *dst) {
	dequant_bf16_row(src, (int)n, dst);
}

static void dequant_zero_cells(const void *src, size_t n, float *dst) {
	(void)src;
	memset(dst, 0, n * sizeof(float));
}

void dequant_row_plan(uint32_t type, dequant_blocks_fn *fn_out, int *divisor_out) {
	switch (type) {
	case GGML_TYPE_F32:
		*fn_out		 = dequant_f32_cells;
		*divisor_out = 1;
		return;
	case GGML_TYPE_F16:
		*fn_out		 = dequant_f16_cells;
		*divisor_out = 1;
		return;
	case GGML_TYPE_BF16:
		*fn_out		 = dequant_bf16_cells;
		*divisor_out = 1;
		return;
	case GGML_TYPE_Q4_0:
		*fn_out		 = dequant_q4_0_row;
		*divisor_out = 32;
		return;
	case GGML_TYPE_Q4_1:
		*fn_out		 = dequant_q4_1_row;
		*divisor_out = 32;
		return;
	case GGML_TYPE_Q5_0:
		*fn_out		 = dequant_q5_0_row;
		*divisor_out = 32;
		return;
	case GGML_TYPE_Q5_1:
		*fn_out		 = dequant_q5_1_row;
		*divisor_out = 32;
		return;
	case GGML_TYPE_Q8_0:
		*fn_out		 = dequant_q8_0_row;
		*divisor_out = 32;
		return;
	case GGML_TYPE_Q8_0_REPACKED:
		*fn_out		 = dequant_q8_0_repacked_row;
		*divisor_out = 32;
		return;
	case GGML_TYPE_Q3_K:
		*fn_out		 = dequant_q3_k_row;
		*divisor_out = 256;
		return;
	case GGML_TYPE_Q4_K:
		*fn_out		 = dequant_q4_k_row;
		*divisor_out = 256;
		return;
	case GGML_TYPE_Q5_K:
		*fn_out		 = dequant_q5_k_row;
		*divisor_out = 256;
		return;
	case GGML_TYPE_Q6_K:
		*fn_out		 = dequant_q6_k_row;
		*divisor_out = 256;
		return;
	case GGML_TYPE_IQ4_NL:
		*fn_out		 = dequant_iq4_nl_row;
		*divisor_out = 32;
		return;
	case GGML_TYPE_IQ3_S:
		*fn_out		 = dequant_iq3_s_row;
		*divisor_out = 256;
		return;
	case GGML_TYPE_IQ2_XXS:
		*fn_out		 = dequant_iq2_xxs_row;
		*divisor_out = 256;
		return;
	case GGML_TYPE_IQ3_XXS:
		*fn_out		 = dequant_iq3_xxs_row;
		*divisor_out = 256;
		return;
	case GGML_TYPE_IQ2_S:
		*fn_out		 = dequant_iq2_s_row;
		*divisor_out = 256;
		return;
	case GGML_TYPE_IQ4_XS:
		*fn_out		 = dequant_iq4_xs_row;
		*divisor_out = 256;
		return;
	default:
		*fn_out		 = dequant_zero_cells;
		*divisor_out = 1;
		return;
	}
}

void dequant_row_dispatch(uint32_t type, const void *src, int n_elems, float *dst) {
	dequant_blocks_fn fn	  = NULL;
	int				  divisor = 1;
	dequant_row_plan(type, &fn, &divisor);
	fn(src, (size_t)(n_elems / divisor), dst);
}

void quant_scratch_ensure(quant_scratch *qs, size_t need) {
	if (qs->q8_buf_elems < need) {
		free(qs->q8_buf);
		qs->q8_buf		 = xmalloc_aligned(need, 64);
		qs->q8_buf_elems = need;
	}
}

MATMUL_Q8_F32(q4_q8, q8_0_block, 32, quantize_q8_0, matmul_q4_q8_qonly_f32)

MATMUL_Q8_F32(q8_0_q8, q8_0_block, 32, quantize_q8_0, matmul_q8_0_q8_qonly_f32)

MATMUL_Q8_F32(iq4_nl_q8, q8_0_block, 32, quantize_q8_0, matmul_iq4_nl_q8_qonly_f32)

MATMUL_Q8_F32(q4_1_q8, q8_1_block, 32, quantize_q8_1, matmul_q4_1_q8_qonly_f32)

MATMUL_Q8_F32(q6_k_q8, q8_k_block, 256, quantize_q8_k, matmul_q6_k_q8_qonly_f32)

MATMUL_Q8_F32(q4_k_q8_k, q8_k_block, 256, quantize_q8_k, matmul_q4_k_q8_k_qonly_f32)

MATMUL_Q8_F32(q5_k_q8_k, q8_k_block, 256, quantize_q8_k, matmul_q5_k_q8_k_qonly_f32)

status_code cpu_probe(void) {
	return OK;
}

size_t cpu_mem_available(backend *self) {
	(void)self;
	return get_available_memory();
}

size_t cpu_mem_total(backend *self) {
	(void)self;
	return get_total_memory();
}

status_code cpu_init(backend *self, int device_index) {
	(void)device_index;
	cpu_priv *p = xcalloc(1, sizeof(cpu_priv));
	self->priv	= p;

	int n_threads = tpool_default_n_threads();
	if (n_threads > 1) {
		p->pool		 = tpool_create(n_threads);
		p->n_threads = tpool_n_threads(p->pool);
		tpool_set_worker_cleanup(p->pool, moe_stream_thread_cleanup);

		p->thread_scratch = xcalloc((size_t)p->n_threads, sizeof(cpu_thread_scratch));
	}

	char cpu_desc[128] = "";
	detect_features(cpu_desc, sizeof(cpu_desc));
	log_tag("CPU", "%d threads, %s", p->n_threads > 0 ? p->n_threads : 1, cpu_desc);

	host_kernels_register(self->priority, matmul_generic_f32);

	return OK;
}

void feat_add(char *buf, size_t cap, const char *name) {
	size_t used = strlen(buf);
	int	   n	= snprintf(buf + used, cap - used, used ? ", %s" : "%s", name);
	if (n < 0 || (size_t)n >= cap - used)
		buf[used] = '\0';
}

void cpu_free(backend *self) {
	cpu_priv *p = self->priv;
	if (!p)
		return;
	if (p->pool)
		tpool_destroy(p->pool);
	tlocal_free_all();
	if (p->thread_scratch) {
		for (int i = 0; i < p->n_threads; i++) {
			free(p->thread_scratch[i].qscratch.q8_buf);
			free(p->thread_scratch[i].scores);
		}
		free(p->thread_scratch);
	}
	free(p->qscratch.q8_buf);
	free(p->xq8_buf);
	free(p->scores);

	for (int i = 0; i < 4; i++)
		free(p->matmul_multi_scratch[i].q8_buf);

	free(p->mla_krot.buf);
	free(p->residual_tmp);
	free(p->bitrev_perm_cache);
	free(p->rope_cs.cs);
	free(p->kv_layer_off);
	free(p->kv_pos_cap);
	free(p->kv_base_pos);
	free(p);
	self->priv = NULL;
}

status_code cpu_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
	out->handle	  = (void *)desc->host_data;
	out->size	  = 0;
	out->offset	  = 0;
	out->host_ptr = desc->host_data;
	out->owner	  = self;
	return OK;
}

status_code cpu_buffer_alloc_scratch(backend *self, size_t size, buffer *out) {
	out->handle	  = xmalloc(size);
	out->size	  = size;
	out->offset	  = 0;
	out->host_ptr = NULL;
	out->owner	  = self;
	return OK;
}

void cpu_buffer_free(backend *self, buffer *buf) {
	(void)self;

	if (!buf || !buf->handle)
		return;
	if (buf->host_ptr)
		return;
	free(buf->handle);
	buf->handle = NULL;
	buf->size	= 0;
	buf->offset = 0;
}

status_code cpu_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n) {
	(void)self;
	memcpy(host_dst, cpu_ptr(buf), (size_t)n * sizeof(float));
	return OK;
}

status_code cpu_buffer_write_f32(backend *self, buffer *buf, const float *host_src, int n) {
	(void)self;
	memcpy(cpu_ptr(buf), host_src, (size_t)n * sizeof(float));
	return OK;
}

status_code cpu_copy_buffer(backend *self, const buffer *src, buffer *dst, int n) {
	(void)self;

	memcpy(cpu_ptr(dst), cpu_ptr(src), (size_t)n * sizeof(float));
	return OK;
}

status_code cpu_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out) {
	cpu_priv *p		   = self->priv;
	p->kv_head_dim_max = desc->head_dim;
	p->kv_quant		   = desc->kv_quant;

	if (k_out->handle) {
		free(k_out->handle);
		k_out->handle = NULL;
	}
	if (v_out->handle) {
		free(v_out->handle);
		v_out->handle = NULL;
	}
	free(p->kv_layer_off);
	p->kv_layer_off = NULL;
	free(p->kv_pos_cap);
	p->kv_pos_cap = NULL;
	free(p->kv_base_pos);
	p->kv_base_pos	   = NULL;
	int has_layer_dims = desc->layer_head_dim && desc->layer_n_kv_heads && desc->n_kv_layers > 0;
	if (has_layer_dims) {
		int n			= desc->n_kv_layers;
		p->kv_layer_off = xcalloc((size_t)n + 1, sizeof(size_t));
	}
	{
		int n = desc->n_kv_layers > 0 ? desc->n_kv_layers : 1;
		free(p->kv_pos_cap);
		free(p->kv_base_pos);
		p->kv_pos_cap  = xcalloc((size_t)n, sizeof(size_t));
		p->kv_base_pos = xcalloc((size_t)n, sizeof(size_t));
		for (int i = 0; i < n; i++)
			p->kv_pos_cap[i] = (desc->layer_pos_cap && desc->layer_pos_cap[i] > 0)
								   ? (size_t)desc->layer_pos_cap[i]
								   : (size_t)desc->n_ctx;
		p->kv_n_layers = n;
	}

	size_t total_bytes;
	if (desc->kv_quant == KV_QUANT_Q8_0) {
		size_t per_layer_uniform = 0;
		if (!has_layer_dims) {
			size_t n_blocks	  = ((size_t)desc->head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
			per_layer_uniform = (size_t)desc->n_kv_heads * cpu_kv_uniform_rows(p, desc) * n_blocks *
								KV_Q8_0_BLOCK_BYTES;
		}
		total_bytes = 0;
		for (int i = 0; i < (has_layer_dims ? desc->n_kv_layers : 1); i++) {
			size_t per_layer = per_layer_uniform;
			if (has_layer_dims) {
				size_t hd		= (size_t)desc->layer_head_dim[i];
				size_t n_blocks = (hd + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
				per_layer		= (size_t)desc->layer_n_kv_heads[i] * p->kv_pos_cap[i] * n_blocks *
								  KV_Q8_0_BLOCK_BYTES;
				p->kv_layer_off[i] = total_bytes;
			}
			total_bytes += per_layer;
		}
		if (!has_layer_dims)
			total_bytes =
				per_layer_uniform * (size_t)(desc->n_kv_layers > 0 ? desc->n_kv_layers : 1);
		if (has_layer_dims)
			p->kv_layer_off[desc->n_kv_layers] = total_bytes;
	} else {
		size_t per_layer_uniform = 0;
		if (!has_layer_dims) {
			per_layer_uniform =
				(size_t)desc->n_kv_heads * cpu_kv_uniform_rows(p, desc) * desc->head_dim;
		}
		total_bytes = 0;
		for (int i = 0; i < (has_layer_dims ? desc->n_kv_layers : 1); i++) {
			size_t per_layer = per_layer_uniform;
			if (has_layer_dims) {
				per_layer		   = (size_t)desc->layer_n_kv_heads[i] * p->kv_pos_cap[i] *
									 (size_t)desc->layer_head_dim[i];
				p->kv_layer_off[i] = total_bytes;
			}
			total_bytes += per_layer;
		}
		if (!has_layer_dims)
			total_bytes =
				per_layer_uniform * (size_t)(desc->n_kv_layers > 0 ? desc->n_kv_layers : 1);
		if (has_layer_dims)
			p->kv_layer_off[desc->n_kv_layers] = total_bytes;
		total_bytes *= sizeof(uint16_t);
		if (has_layer_dims) {
			for (int i = 0; i <= desc->n_kv_layers; i++)
				p->kv_layer_off[i] *= sizeof(uint16_t);
		}
	}

	p->kv_k_bytes	= total_bytes;
	k_out->handle	= xmalloc_aligned(total_bytes, 64);
	k_out->size		= total_bytes;
	k_out->host_ptr = NULL;
	k_out->owner	= self;
	v_out->handle	= xmalloc_aligned(total_bytes, 64);
	v_out->size		= total_bytes;
	v_out->host_ptr = NULL;
	v_out->owner	= self;

	madvise_hugepage(k_out->handle, total_bytes);
	madvise_hugepage(v_out->handle, total_bytes);
	prefault(k_out->handle, total_bytes);
	prefault(v_out->handle, total_bytes);

	if (desc->kv_quant == KV_QUANT_Q8_0) {
		size_t n_blocks	   = ((size_t)desc->head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		p->kv_block_stride = n_blocks * KV_Q8_0_BLOCK_BYTES;
	} else {
		p->kv_block_stride = (size_t)desc->head_dim * sizeof(uint16_t);
	}
	p->kv_layer_stride = (size_t)desc->n_kv_heads * desc->n_ctx * p->kv_block_stride;
	p->kv_kvh_stride   = (size_t)desc->n_ctx * p->kv_block_stride;

	p->scores_want = (size_t)desc->n_ctx;

	return OK;
}

typedef struct {
	const float *kf_base;
	const float *vf_base;
	uint8_t		*kd_u8;
	uint8_t		*vd_u8;
	uint16_t	*kd_u16;
	uint16_t	*vd_u16;
	size_t		 layer_base;
	size_t		 blk_stride;
	size_t		 kvh_stride;
	size_t		 slot_start;
	int			 in_row_stride;
	int			 head_dim;
	int			 n_active;
	int			 is_q80;
} cpu_kv_put_batch_job;

static void cpu_kv_put_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_kv_put_batch_job *j = ctx;

	for (int idx = begin; idx < end; idx++) {
		int			 row = idx / j->n_active;
		int			 kvh = idx % j->n_active;
		const float *kf	 = j->kf_base + (size_t)row * j->in_row_stride + (size_t)kvh * j->head_dim;
		const float *vf	 = j->vf_base + (size_t)row * j->in_row_stride + (size_t)kvh * j->head_dim;
		if (j->is_q80) {
			size_t off = j->layer_base + (size_t)kvh * j->kvh_stride +
						 (j->slot_start + (size_t)row) * j->blk_stride;
			cpu_kv_put_q8_0_head(j->kd_u8 + off, j->vd_u8 + off, kf, vf, j->head_dim);
		} else {
			size_t off = j->layer_base + (size_t)kvh * j->kvh_stride +
						 (j->slot_start + (size_t)row) * (size_t)j->head_dim;
			cpu_kv_put_f16_head(j->kd_u16 + off, j->vd_u16 + off, kf, vf, j->head_dim);
		}
	}
}

status_code cpu_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
							 const buffer *k_in, const buffer *v_in, int in_row_stride,
							 int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
							 int m) {
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	int		  total	   = m * n_active;
	if (total <= 0)
		return OK;

	int	   is_q80 = (p->kv_quant == KV_QUANT_Q8_0);
	size_t rows	  = cpu_kv_layer_rows(p, layer, n_ctx);
	size_t blk_stride =
		is_q80 ? (((size_t)head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK * KV_Q8_0_BLOCK_BYTES)
			   : ((size_t)head_dim * sizeof(uint16_t));
	size_t layer_base = cpu_kv_layer_base(
		p, layer, (size_t)n_kv_heads * rows * (is_q80 ? blk_stride : (size_t)head_dim),
		is_q80 ? 1 : sizeof(uint16_t));

	size_t slot_start = (size_t)pos_start;
	if (m > 0) {
		cpu_kv_put_slot(p, layer, pos_start + m - 1, n_ctx,
						is_q80 ? (void *)k->handle : (void *)cpu_ptr(k),
						is_q80 ? (void *)v->handle : (void *)cpu_ptr(v),
						is_q80 ? layer_base : layer_base * sizeof(uint16_t),
						is_q80 ? rows * blk_stride : rows * (size_t)head_dim * sizeof(uint16_t),
						blk_stride, n_active);
		size_t b   = p->kv_base_pos ? p->kv_base_pos[layer] : 0;
		slot_start = b ? (size_t)pos_start - b : (size_t)pos_start;
	}

	cpu_kv_put_batch_job job = {.kf_base	= (const float *)cpu_ptr(k_in),
								.vf_base	= (const float *)cpu_ptr(v_in),
								.kd_u8		= (uint8_t *)k->handle,
								.vd_u8		= (uint8_t *)v->handle,
								.kd_u16		= (uint16_t *)cpu_ptr(k),
								.vd_u16		= (uint16_t *)cpu_ptr(v),
								.layer_base = layer_base,
								.blk_stride = blk_stride,
								.kvh_stride = is_q80 ? rows * blk_stride : rows * (size_t)head_dim,
								.slot_start = slot_start,
								.in_row_stride = in_row_stride,
								.head_dim	   = head_dim,
								.n_active	   = n_active,
								.is_q80		   = is_q80};

	if (p->pool && total >= 2) {
		tpool_parallel_for(p->pool, total, 1, cpu_kv_put_batch_chunk, &job);
		return OK;
	}
	cpu_kv_put_batch_chunk(0, total, -1, &job);
	return OK;
}

void cpu_synchronize(backend *self) {
	(void)self;
}

status_code cpu_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
							int token, int dim, buffer *x_out) {
	(void)self;
	size_t		   row_stride = ggml_row_size(tok_embd_type, dim);
	const uint8_t *embd		  = (const uint8_t *)cpu_ptr(tok_embd) + ((size_t)token * row_stride);
	float		  *dst		  = (float *)cpu_ptr(x_out);
	dequant_row_dispatch(tok_embd_type, embd, dim, dst);
	return OK;
}

static void cpu_kv_load_head(const uint8_t *src, int q8, int head_dim, float *dst) {
	if (!q8) {
		const uint16_t *h = (const uint16_t *)src;
		for (int i = 0; i < head_dim; i++)
			dst[i] = f16_to_f32(h[i]);
		return;
	}
	int n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	for (int b = 0; b < n_blocks; b++) {
		const uint8_t *blk = src + (size_t)b * KV_Q8_0_BLOCK_BYTES;
		uint16_t	   d16;
		memcpy(&d16, blk, sizeof(d16));
		float d	   = f16_to_f32(d16);
		int	  base = b * KV_Q8_0_BLOCK;
		int	  n	   = head_dim - base < KV_Q8_0_BLOCK ? head_dim - base : KV_Q8_0_BLOCK;
		for (int j = 0; j < n; j++)
			dst[base + j] = d * (float)(int8_t)blk[2 + j];
	}
}

static void cpu_kv_rotate_head(float *head, int head_dim, int neox, const float *cos_row,
							   const float *sin_row) {
	int half = head_dim / 2;
	for (int j = 0; j < half; j++) {
		int	  i0 = neox ? j : 2 * j;
		int	  i1 = neox ? j + half : 2 * j + 1;
		float a	 = head[i0];
		float b	 = head[i1];
		head[i0] = a * cos_row[j] - b * sin_row[j];
		head[i1] = a * sin_row[j] + b * cos_row[j];
	}
}

status_code cpu_kv_shift(backend *self, buffer *k, buffer *v, int layer,
						 const kv_shift_chunk *chunks, int n_chunks, const float *cos_tbl,
						 const float *sin_tbl, int n_kv_heads, int head_dim, int n_ctx,
						 int n_kv_heads_active) {
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	int		  half	   = head_dim / 2;
	int		  q8	   = p->kv_quant == KV_QUANT_Q8_0;
	if (n_chunks <= 0 || n_active <= 0 || half <= 0)
		return OK;

	size_t row_bytes =
		q8 ? (((size_t)head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK) * KV_Q8_0_BLOCK_BYTES
		   : (size_t)head_dim * sizeof(uint16_t);
	size_t rows		  = cpu_kv_layer_rows(p, layer, n_ctx);
	size_t kvh_stride = rows * row_bytes;
	size_t layer_base = cpu_kv_layer_base(p, layer, (size_t)n_kv_heads * kvh_stride, 1);
	size_t base0	  = p->kv_base_pos ? p->kv_base_pos[layer] : 0;
	size_t total_rows = 0;
	for (int i = 0; i < n_chunks; i++) {
		if ((size_t)chunks[i].src_pos < base0 && (size_t)chunks[i].dst_pos >= base0)
			return ERR_UNSUPPORTED;
		total_rows += (size_t)chunks[i].n_rows;
	}

	size_t row_floats = (size_t)n_active * (size_t)head_dim;
	float *kt		  = xmalloc(total_rows * row_floats * sizeof(float));
	float *vt		  = xmalloc(total_rows * row_floats * sizeof(float));
	size_t r		  = 0;
	for (int i = 0; i < n_chunks; i++) {
		for (int t = 0; t < chunks[i].n_rows; t++, r++) {
			size_t src = (size_t)(chunks[i].src_pos + t);
			if (src < base0 || src - base0 >= rows)
				continue;
			size_t pos_off = (src - base0) * row_bytes;
			for (int kvh = 0; kvh < n_active; kvh++) {
				size_t off = layer_base + (size_t)kvh * kvh_stride + pos_off;
				float *kh  = kt + r * row_floats + (size_t)kvh * head_dim;
				float *vh  = vt + r * row_floats + (size_t)kvh * head_dim;
				cpu_kv_load_head((const uint8_t *)k->handle + off, q8, head_dim, kh);
				cpu_kv_load_head((const uint8_t *)v->handle + off, q8, head_dim, vh);
				cpu_kv_rotate_head(kh, head_dim, self->rope_neox, cos_tbl + (size_t)i * half,
								   sin_tbl + (size_t)i * half);
			}
		}
	}

	status_code st = OK;
	r			   = 0;
	for (int i = 0; i < n_chunks && st == OK; i++) {
		for (int t = 0; t < chunks[i].n_rows && st == OK; t++, r++) {
			size_t dst = (size_t)(chunks[i].dst_pos + t);
			if (p->kv_base_pos && dst < p->kv_base_pos[layer])
				continue;
			buffer k_row = buffer_host_view(self, kt + r * row_floats, row_floats * sizeof(float));
			buffer v_row = buffer_host_view(self, vt + r * row_floats, row_floats * sizeof(float));
			st = self->kv_put(self, k, v, layer, (int)dst, &k_row, &v_row, n_kv_heads, head_dim,
							  n_ctx, n_kv_heads_active);
		}
	}
	free(kt);
	free(vt);
	return st;
}

status_code cpu_kv_alloc_mla(backend *self, int n_layers, int n_ctx, int kv_lora, int qk_rope,
							 buffer *kv_out) {
	cpu_priv *p			= self->priv;
	size_t	  per_layer = (size_t)(kv_lora + qk_rope) * (size_t)n_ctx * sizeof(float);
	size_t	  total		= per_layer * (size_t)n_layers;
	kv_out->handle		= xmalloc_aligned(total, 64);
	kv_out->size		= total;
	kv_out->host_ptr	= NULL;
	kv_out->owner		= self;
	madvise_hugepage(kv_out->handle, total);
	prefault(kv_out->handle, total);

	if (p) {
		size_t		krot_need = (size_t)n_ctx * (size_t)qk_rope * sizeof(float);
		status_code grow_st =
			cpu_buf_grow((void **)&p->mla_krot.buf, &p->mla_krot.cap, krot_need, 1);
		if (grow_st != OK)
			return grow_st;
	}

	return OK;
}

status_code cpu_kv_put_mla(backend *self, buffer *kv_cache, int layer, int pos,
						   const buffer *kv_a_in, const buffer *kv_a_norm_w, int kv_lora,
						   int qk_rope, int n_ctx, float eps) {
	(void)self;
	int			 total_dim	  = kv_lora + qk_rope;
	const float *kv_a		  = (const float *)cpu_ptr(kv_a_in);
	const float *norm_w		  = (const float *)cpu_ptr(kv_a_norm_w);
	float		*cache_base	  = (float *)cpu_ptr(kv_cache);
	size_t		 layer_stride = (size_t)total_dim * (size_t)n_ctx;
	float		*slot = cache_base + ((size_t)layer * layer_stride) + ((size_t)pos * total_dim);
	float		 ss	  = 0.0f;
	for (int i = 0; i < kv_lora; i++)
		ss += kv_a[i] * kv_a[i];
	ss = 1.0f / sqrtf((ss / (float)kv_lora) + eps);
	for (int i = 0; i < kv_lora; i++)
		slot[i] = kv_a[i] * ss * norm_w[i];
	for (int i = 0; i < qk_rope; i++)
		slot[kv_lora + i] = kv_a[kv_lora + i];
	return OK;
}

tpool *cpu_get_pool(backend *self) {
	cpu_priv *p = self->priv;
	return p ? p->pool : NULL;
}

void cpu_kv_free(backend *self, buffer *k, buffer *v) {
	cpu_buffer_free(self, k);
	cpu_buffer_free(self, v);
}

typedef struct {
	const void *src;
	void	   *dst;
	int			k;
	int			rows_per_group;
	void (*repack_fn)(const void *src, void *dst, int row_begin, int row_end, int k);
} cpu_repack_job;

static void cpu_repack_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_repack_job *job		  = ctx;
	int				row_begin = begin * job->rows_per_group;
	int				row_end	  = end * job->rows_per_group;
	job->repack_fn(job->src, job->dst, row_begin, row_end, job->k);
}

status_code cpu_repack_plan(backend *self, uint32_t type, uint64_t d0, uint64_t d1,
							uint32_t *re_type_out) {
	(void)self;
	uint32_t re_type = 0;
	if (type == GGML_TYPE_IQ3_S && (d0 % 256) == 0 && (d1 % IQ3_S_REPACKED_ROWS) == 0) {
		re_type = GGML_TYPE_IQ3_S_REPACKED;
	} else if (type == GGML_TYPE_IQ4_NL && (d0 % 32) == 0 && (d1 % IQ4_NL_REPACKED_ROWS) == 0) {
		re_type = GGML_TYPE_IQ4_NL_REPACKED;
	} else if (type == GGML_TYPE_Q8_0 && (d0 % 32) == 0 && (d1 % Q8_0_REPACKED_ROWS) == 0) {
		re_type = GGML_TYPE_Q8_0_REPACKED;
	} else if (type == GGML_TYPE_Q4_0 && (d0 % 32) == 0 && (d1 % Q4_0_REPACKED_ROWS) == 0) {
		re_type = GGML_TYPE_Q4_0_REPACKED;
	} else if (type == GGML_TYPE_Q4_K && (d0 % 256) == 0 && (d1 % Q4_K_REPACKED_ROWS) == 0) {
		re_type = GGML_TYPE_Q4_K_REPACKED;
	} else if (type == GGML_TYPE_Q5_K && (d0 % 256) == 0 && (d1 % Q5_K_REPACKED_ROWS) == 0) {
		re_type = GGML_TYPE_Q5_K_REPACKED;
	} else if (type == GGML_TYPE_Q6_K && (d0 % 256) == 0 && (d1 % Q6_K_REPACKED_ROWS) == 0) {
		re_type = GGML_TYPE_Q6_K_REPACKED;
	}
	if (!re_type)
		return ERR_UNSUPPORTED;
	*re_type_out = re_type;
	return OK;
}

status_code cpu_repack_weight(backend *self, uint32_t type, const void *src, void *dst, int n_rows,
							  int k) {
	cpu_repack_job job = {.src = src, .dst = dst, .k = k};
	if (type == GGML_TYPE_Q8_0) {
		job.rows_per_group = Q8_0_REPACKED_ROWS;
		job.repack_fn	   = repack_q8_0_to_q8_0_repacked_rows;
	} else if (type == GGML_TYPE_Q4_0) {
		job.rows_per_group = Q4_0_REPACKED_ROWS;
		job.repack_fn	   = repack_q4_0_to_q4_0_repacked_rows;
	} else if (type == GGML_TYPE_IQ3_S) {
		job.rows_per_group = IQ3_S_REPACKED_ROWS;
		job.repack_fn	   = repack_iq3_s_to_iq3_s_repacked_rows;
	} else if (type == GGML_TYPE_IQ4_NL) {
		job.rows_per_group = IQ4_NL_REPACKED_ROWS;
		job.repack_fn	   = repack_iq4_nl_to_iq4_nl_repacked_rows;
	} else if (type == GGML_TYPE_Q4_K) {
		job.rows_per_group = Q4_K_REPACKED_ROWS;
		job.repack_fn	   = repack_q4_k_to_q4_k_repacked_rows;
	} else if (type == GGML_TYPE_Q5_K) {
		job.rows_per_group = Q5_K_REPACKED_ROWS;
		job.repack_fn	   = repack_q5_k_to_q5_k_repacked_rows;
	} else if (type == GGML_TYPE_Q6_K) {
		job.rows_per_group = Q6_K_REPACKED_ROWS;
		job.repack_fn	   = repack_q6_k_to_q6_k_repacked_rows;
	} else {
		job.rows_per_group = 1;
		job.repack_fn	   = repack_iq4_nl_to_q8_0_rows;
	}
	tpool *pool		= (self && self->get_pool) ? self->get_pool(self) : NULL;
	int	   n_groups = n_rows / job.rows_per_group;
	tpool_parallel_for(pool, n_groups, 1, cpu_repack_chunk, &job);
	return OK;
}

status_code cpu_dequant_row(backend *self, uint32_t type, const void *src, int n_elems,
							float *dst) {
	(void)self;
	dequant_row_dispatch(type, src, n_elems, dst);
	return OK;
}

status_code cpu_moe_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
							 int n, float gate_scale, float up_scale, int use_gelu) {
	(void)self;
	if (use_gelu)
		moe_activate_gelu(cpu_ptr(out), cpu_ptr(gate), cpu_ptr(up), n, gate_scale, up_scale);
	else
		moe_activate_silu(cpu_ptr(out), cpu_ptr(gate), cpu_ptr(up), n, gate_scale, up_scale);
	return OK;
}

static status_code cpu_hc_init_buf(backend *self, const hc_init_desc *d) {
	hc_init_host_desc h = {d->dim, d->n_hc, buffer_f32(d->x), buffer_f32(d->hc)};
	return cpu_hc_init(self, &h);
}

static status_code cpu_hc_final_buf(backend *self, const hc_final_desc *d) {
	hc_final_host_desc h = {d->dim, d->n_hc, buffer_f32(d->hc), buffer_f32(d->x)};
	return cpu_hc_final(self, &h);
}

static status_code cpu_hc_pre_buf(backend *self, const hc_pre_desc *d) {
	float			*ws = d->ws ? d->ws : xmalloc(hc_pre_ws_floats(d) * sizeof(float));
	hc_pre_host_desc h	= {.dim			   = d->dim,
						   .n_hc		   = d->n_hc,
						   .sinkhorn_iters = d->sinkhorn_iters,
						   .norm_eps	   = d->norm_eps,
						   .eps			   = d->eps,
						   .stream		   = buffer_f32(d->stream),
						   .fn			   = d->fn,
						   .base		   = buffer_f32(d->base),
						   .scale		   = buffer_f32(d->scale),
						   .x			   = buffer_f32(d->x),
						   .coef		   = buffer_f32(d->coef),
						   .ws			   = ws};
	status_code		 st = cpu_hc_pre(self, &h);
	if (!d->ws)
		free(ws);
	return st;
}

static status_code cpu_hc_post_buf(backend *self, const hc_post_desc *d) {
	const size_t	  n	   = (size_t)d->n_hc * d->dim;
	float			 *cur  = buffer_f32(d->hc);
	float			 *next = xmalloc(n * sizeof(float));
	hc_post_host_desc h	   = {d->dim, d->n_hc, buffer_f32(d->sub), buffer_f32(d->coef), cur, next};
	status_code		  st   = cpu_hc_post(self, &h);
	if (st == OK)
		memcpy(cur, next, n * sizeof(float));
	free(next);
	return st;
}

static status_code cpu_kda_buf(backend *self, const kda_desc *d) {
	const size_t D	   = (size_t)d->n_heads * d->head_dim;
	float		*conv  = buffer_f32(d->conv_state);
	float		*state = buffer_f32(d->state);
	if (d->reset) {
		memset(conv, 0, 3 * D * (size_t)(d->conv_kernel - 1) * sizeof(float));
		memset(state, 0, (size_t)d->n_heads * d->head_dim * d->head_dim * sizeof(float));
	}
	float		 *ws = d->ws ? d->ws : xmalloc(kda_ws_floats(d) * sizeof(float));
	kda_host_desc h	 = {.dim			  = d->dim,
						.n_heads		  = d->n_heads,
						.head_dim		  = d->head_dim,
						.conv_kernel	  = d->conv_kernel,
						.gate_lower_bound = d->gate_lower_bound,
						.norm_eps		  = d->norm_eps,
						.x				  = buffer_f32(d->x),
						.wq				  = d->wq,
						.wk				  = d->wk,
						.wv				  = d->wv,
						.wo				  = d->wo,
						.f_a			  = d->f_a,
						.f_b			  = d->f_b,
						.g_a			  = d->g_a,
						.g_b			  = d->g_b,
						.beta			  = d->beta,
						.conv_q			  = buffer_f32(d->conv_q),
						.conv_k			  = buffer_f32(d->conv_k),
						.conv_v			  = buffer_f32(d->conv_v),
						.dt_bias		  = buffer_f32(d->dt_bias),
						.ssm_a			  = buffer_f32(d->ssm_a),
						.out_norm		  = buffer_f32(d->out_norm),
						.conv_state		  = conv,
						.state			  = state,
						.out			  = buffer_f32(d->out),
						.ws				  = ws};
	status_code	  st = cpu_kda(self, &h);
	if (!d->ws)
		free(ws);
	return st;
}

static status_code cpu_mla_absorbed_buf(backend *self, const mla_absorbed_desc *d) {
	float				  *ws = d->ws ? d->ws : xmalloc(mla_absorbed_ws_floats(d) * sizeof(float));
	mla_absorbed_host_desc h  = {.dim		   = d->dim,
								 .n_heads	   = d->n_heads,
								 .kv_lora	   = d->kv_lora,
								 .qk_head	   = d->qk_head,
								 .v_head	   = d->v_head,
								 .q_lora	   = d->q_lora,
								 .pos		   = d->pos,
								 .n_threads	   = d->n_threads,
								 .norm_eps	   = d->norm_eps,
								 .x			   = buffer_f32(d->x),
								 .q_a		   = d->q_a,
								 .q_b		   = d->q_b,
								 .kv_a		   = d->kv_a,
								 .wo		   = d->wo,
								 .q_a_norm	   = buffer_f32(d->q_a_norm),
								 .kv_a_norm	   = buffer_f32(d->kv_a_norm),
								 .k_b		   = buffer_f32(d->k_b),
								 .v_b		   = buffer_f32(d->v_b),
								 .latent_cache = buffer_f32(d->latent_cache),
								 .out		   = buffer_f32(d->out),
								 .ws		   = ws};
	status_code			   st = cpu_mla_absorbed(self, &h);
	if (!d->ws)
		free(ws);
	return st;
}

status_code cpu_backend_fill(backend *out) {
	out->probe							 = cpu_probe;
	out->init							 = cpu_init;
	out->free							 = cpu_free;
	out->buffer_alloc_weight			 = cpu_buffer_alloc_weight;
	out->buffer_alloc_scratch			 = cpu_buffer_alloc_scratch;
	out->buffer_free					 = cpu_buffer_free;
	out->copy_buffer					 = cpu_copy_buffer;
	out->buffer_read_f32				 = cpu_buffer_read_f32;
	out->buffer_write_f32				 = cpu_buffer_write_f32;
	out->kv_alloc						 = cpu_kv_alloc;
	out->kv_put							 = cpu_kv_put;
	out->kv_put_batch					 = cpu_kv_put_batch;
	out->kv_shift						 = cpu_kv_shift;
	out->embd_lookup					 = cpu_embd_lookup;
	out->rmsnorm						 = cpu_rmsnorm;
	out->matmul							 = cpu_matmul;
	out->matmul_residual				 = cpu_matmul_residual;
	out->matmul_batch					 = cpu_matmul_batch;
	out->matmul_multi					 = cpu_matmul_multi;
	out->matmul_multi_batch				 = cpu_matmul_multi_batch;
	out->matmul_qonly					 = cpu_matmul_qonly;
	out->prequantize_x					 = cpu_prequantize_x;
	out->rope							 = cpu_rope;
	out->rope_qk						 = cpu_rope_qk;
	out->rope_ext						 = cpu_rope_ext;
	out->rope_ext_batch					 = cpu_rope_ext_batch;
	out->attention						 = cpu_attention;
	out->attention_swa					 = cpu_attention_swa;
	out->add_inplace					 = cpu_add_inplace;
	out->ffn_activate					 = cpu_ffn_activate;
	out->ffn_activate_ex				 = cpu_ffn_activate_ex;
	out->rmsnorm_per_head				 = cpu_rmsnorm_per_head;
	out->rmsnorm_per_head_batch			 = cpu_rmsnorm_per_head_batch;
	out->rmsnorm_noweight_batch			 = cpu_rmsnorm_noweight_batch;
	out->rmsnorm_noweight_per_head_batch = cpu_rmsnorm_noweight_per_head_batch;
	out->rmsnorm_noweight				 = cpu_rmsnorm_noweight;
	out->rmsnorm_noweight_per_head		 = cpu_rmsnorm_noweight_per_head;
	out->argmax							 = cpu_argmax;
	out->gated_delta_net				 = cpu_gated_delta_net;
	out->repack_plan					 = cpu_repack_plan;
	out->repack_weight					 = cpu_repack_weight;
	out->dequant_row					 = cpu_dequant_row;
	out->moe_activate					 = cpu_moe_activate;
	out->synchronize					 = cpu_synchronize;
	out->attention_mla					 = cpu_attention_mla;
	out->kv_alloc_mla					 = cpu_kv_alloc_mla;
	out->kv_put_mla						 = cpu_kv_put_mla;
	out->get_pool						 = cpu_get_pool;
	out->matmul_thread_local			 = cpu_matmul_thread_local;
	out->mem_available					 = cpu_mem_available;
	out->mem_total						 = cpu_mem_total;
	out->rmsnorm_batch					 = cpu_rmsnorm_batch;
	out->add_batch						 = cpu_add_batch;
	out->ffn_activate_batch				 = cpu_ffn_activate_batch;
	out->rope_batch						 = cpu_rope_batch;
	out->rope_qk_batch					 = cpu_rope_qk_batch;
	out->attention_batch				 = cpu_attention_batch;
	out->attention_swa_batch			 = cpu_attention_swa_batch;
	out->rmsnorm_add					 = cpu_rmsnorm_add;
	out->rmsnorm_add_batch				 = cpu_rmsnorm_add_batch;
	out->scale_inplace					 = cpu_scale_inplace;
	out->softcap						 = cpu_softcap;
	out->split_qgate					 = cpu_split_qgate;
	out->attn_output_gate				 = cpu_attn_output_gate;
	out->partial_rope_qk				 = cpu_partial_rope_qk;
	out->ple_combine					 = cpu_ple_combine;
	out->matmul_ffn_down				 = cpu_matmul_ffn_down;
	out->matmul_ffn_down_batch			 = cpu_matmul_ffn_down_batch;
	out->kv_free						 = cpu_kv_free;
	out->hc_init						 = cpu_hc_init_buf;
	out->hc_pre							 = cpu_hc_pre_buf;
	out->hc_post						 = cpu_hc_post_buf;
	out->hc_final						 = cpu_hc_final_buf;
	out->kda							 = cpu_kda_buf;
	out->mla_absorbed					 = cpu_mla_absorbed_buf;
	return OK;
}

typedef struct {
	const float *x, *w;
	float		*y;
	int			 n;
	float		 eps;
} cpu_rmsnorm_batch_job;

status_code cpu_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
						float eps) {
	return cpu_rmsnorm_batch(self, x, w, y, n, eps, 1);
}

static void cpu_rmsnorm_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rmsnorm_batch_job *j = ctx;
	for (int row = begin; row < end; row++)
		rmsnorm(j->x + ((size_t)row * j->n), j->w, j->y + ((size_t)row * j->n), j->n, j->eps);
}

status_code cpu_rmsnorm_batch(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
							  float eps, int m) {
	cpu_priv			 *p	  = self->priv;
	cpu_rmsnorm_batch_job job = {
		.x = cpu_ptr(x), .w = cpu_ptr(w), .y = cpu_ptr(y), .n = n, .eps = eps};
	cpu_run_batch(p->pool, m, cpu_rmsnorm_batch_chunk, &job);
	return OK;
}

status_code cpu_rmsnorm_per_head(backend *self, const buffer *x, const buffer *w, buffer *y,
								 int n_heads, int head_dim, float eps) {
	(void)self;
	const float *xf = cpu_ptr(x);
	const float *wf = cpu_ptr(w);
	float		*yf = cpu_ptr(y);
	rmsnorm_per_head(xf, wf, yf, n_heads, head_dim, eps);
	return OK;
}

status_code cpu_rmsnorm_noweight(backend *self, const buffer *x, buffer *y, int n, float eps) {
	(void)self;
	const float *xf = cpu_ptr(x);
	float		*yf = cpu_ptr(y);
	rmsnorm_noweight(xf, yf, n, eps);
	return OK;
}

status_code cpu_rmsnorm_noweight_per_head(backend *self, const buffer *x, buffer *y, int n_heads,
										  int head_dim, float eps) {
	(void)self;
	const float *xf = cpu_ptr(x);
	float		*yf = cpu_ptr(y);
	for (int h = 0; h < n_heads; h++) {
		rmsnorm_noweight(xf + ((size_t)h * head_dim), yf + ((size_t)h * head_dim), head_dim, eps);
	}
	return OK;
}

typedef struct {
	const buffer *x;
	const buffer *w;
	buffer		 *y;
	int			  n_heads, head_dim, row_stride;
	float		  eps;
	int			  noweight;
} cpu_rmsnorm_row_batch_job;

static void cpu_rmsnorm_row_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rmsnorm_row_batch_job *j	  = ctx;
	size_t					   stride = (size_t)j->row_stride;
	for (int r = begin; r < end; r++) {
		const float *xr = (const float *)cpu_ptr(j->x) + (size_t)r * stride;
		float		*yr = (float *)cpu_ptr(j->y) + (size_t)r * stride;
		if (j->noweight) {
			if (j->n_heads > 0) {
				for (int h = 0; h < j->n_heads; h++)
					rmsnorm_noweight(xr + ((size_t)h * j->head_dim), yr + ((size_t)h * j->head_dim),
									 j->head_dim, j->eps);
			} else {
				rmsnorm_noweight(xr, yr, j->row_stride, j->eps);
			}
		} else {
			rmsnorm_per_head(xr, cpu_ptr(j->w), yr, j->n_heads, j->head_dim, j->eps);
		}
	}
}

static status_code cpu_rmsnorm_row_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
										 int n_heads, int head_dim, int row_stride, float eps,
										 int m, int noweight) {
	cpu_priv				 *p	  = self->priv;
	cpu_rmsnorm_row_batch_job job = {.x			 = x,
									 .w			 = w,
									 .y			 = y,
									 .n_heads	 = n_heads,
									 .head_dim	 = head_dim,
									 .row_stride = row_stride,
									 .eps		 = eps,
									 .noweight	 = noweight};
	cpu_run_batch(p->pool, m, cpu_rmsnorm_row_batch_chunk, &job);
	return OK;
}

status_code cpu_rmsnorm_per_head_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
									   int n_heads, int head_dim, float eps, int m) {
	int row_stride = n_heads * head_dim;
	return cpu_rmsnorm_row_batch(self, x, w, y, n_heads, head_dim, row_stride, eps, m, 0);
}

status_code cpu_rmsnorm_noweight_batch(backend *self, const buffer *x, buffer *y, int n, float eps,
									   int m) {
	return cpu_rmsnorm_row_batch(self, x, NULL, y, 0, 0, n, eps, m, 1);
}

status_code cpu_rmsnorm_noweight_per_head_batch(backend *self, const buffer *x, buffer *y,
												int n_heads, int head_dim, float eps, int m) {
	return cpu_rmsnorm_row_batch(self, x, NULL, y, n_heads, head_dim, n_heads * head_dim, eps, m,
								 1);
}

status_code cpu_rmsnorm_add(backend *self, const buffer *x, const buffer *w, const buffer *residual,
							buffer *y, int n, float eps, float out_scale) {
	return cpu_rmsnorm_add_batch(self, x, w, residual, y, n, eps, out_scale, 1);
}

status_code cpu_scale_inplace(backend *self, buffer *x, float scale, int n) {
	(void)self;
	scale_f32(cpu_ptr(x), scale, n);
	return OK;
}

status_code cpu_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
							float combine_scale) {
	(void)self;
	add_scale_f32(cpu_ptr(ple), cpu_ptr(proj), combine_scale, n);
	return OK;
}

typedef struct {
	float		*conv_out, *conv_state;
	const float *mixed, *conv_w;
	int			 conv_dim, conv_kernel, n_tokens, mixed_stride, history;
} gdn_conv_job;

static void gdn_conv_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	gdn_conv_job *j		  = ctx;
	int			  history = j->history;
	for (int c = begin; c < end; c++) {
		const float *w	   = j->conv_w + (size_t)c * j->conv_kernel;
		float		*hist  = j->conv_state + (size_t)c * history;
		const float *mix_c = j->mixed + c;
		if (history == 3) {
			float h0 = hist[0], h1 = hist[1], h2 = hist[2];
			for (int t = 0; t < j->n_tokens; t++) {
				float m	  = mix_c[(size_t)t * j->mixed_stride];
				float sum = h0 * w[0] + h1 * w[1] + h2 * w[2] + m * w[3];
				h0		  = h1;
				h1		  = h2;
				h2		  = m;
				j->conv_out[(size_t)t * j->conv_dim + c] = silu(sum);
			}
			hist[0] = h0;
			hist[1] = h1;
			hist[2] = h2;
		} else {
			for (int t = 0; t < j->n_tokens; t++) {
				const float *mix = j->mixed + (size_t)t * j->mixed_stride;
				float		*oc	 = j->conv_out + (size_t)t * j->conv_dim;
				float		 sum = mix[c] * w[history];
				if (history > 0) {
					for (int jj = 0; jj < history; jj++)
						sum += hist[jj] * w[jj];
					if (history > 1)
						memmove(hist, hist + 1, (size_t)(history - 1) * sizeof(float));
					hist[history - 1] = mix[c];
				}
				oc[c] = silu(sum);
			}
		}
	}
}

static void gdn_conv_tokens(tpool *pool, float *conv_out, float *conv_state, const float *mixed,
							const float *conv_w, int conv_dim, int conv_kernel, int n_tokens,
							int mixed_stride) {
	gdn_conv_job job = {.conv_out	  = conv_out,
						.conv_state	  = conv_state,
						.mixed		  = mixed,
						.conv_w		  = conv_w,
						.conv_dim	  = conv_dim,
						.conv_kernel  = conv_kernel,
						.n_tokens	  = n_tokens,
						.mixed_stride = mixed_stride,
						.history	  = conv_kernel - 1};
	if (pool && conv_dim > 8) {
		tpool_parallel_for(pool, conv_dim, 8, gdn_conv_chunk, &job);
		return;
	}
	gdn_conv_chunk(0, conv_dim, -1, &job);
}

typedef struct {
	float		*state;
	const float *conv;
	const float *z;
	const float *alpha;
	const float *beta;
	float		*out;
	const float *dt;
	const float *a;
	const float *norm_w;
	float		*scratch;
	int			 n_tokens;
	int			 conv_stride;
	int			 z_stride;
	int			 alpha_stride;
	int			 beta_stride;
	int			 out_stride;
	int			 nkh;
	int			 kd;
	int			 vd;
	int			 key_dim;
	int			 scratch_stride;
	float		 eps;
} gdn_job;

static void gdn_heads(int vh0, int vh1, const gdn_job *j, float *scratch) {
	int	   kd		  = j->kd;
	int	   vd		  = j->vd;
	float  q_scale	  = 1.0f / sqrtf((float)kd);
	float *k_s		  = scratch;
	float *q_s		  = k_s + kd;
	float *mem		  = q_s + kd;
	float *delta	  = mem + vd;
	int	   state_head = kd * vd;

	for (int vh = vh0; vh < vh1; vh++) {
		int	   kh	 = vh % j->nkh;
		float *shead = j->state + (size_t)vh * state_head;
		for (int t = 0; t < j->n_tokens; t++) {
			const float *conv_t	 = j->conv + (size_t)t * j->conv_stride;
			const float *q		 = conv_t + (size_t)kh * kd;
			const float *k		 = conv_t + j->key_dim + (size_t)kh * kd;
			const float *v		 = conv_t + 2 * j->key_dim + (size_t)vh * vd;
			const float *z_t	 = j->z + (size_t)t * j->z_stride + (size_t)vh * vd;
			float		*y		 = j->out + (size_t)t * j->out_stride + (size_t)vh * vd;
			float		 alpha_t = j->alpha[(size_t)t * j->alpha_stride + vh];
			float		 beta_t	 = j->beta[(size_t)t * j->beta_stride + vh];

			float decay = expf(j->a[vh] * softplusf(alpha_t + j->dt[vh]));
			float b		= sigmoidf(beta_t);

			float qn = q_scale / sqrtf(j->eps + rmsnorm_sum_sq(q, kd));
			float kn = 1.0f / sqrtf(j->eps + rmsnorm_sum_sq(k, kd));
			memcpy(q_s, q, (size_t)kd * sizeof(float));
			memcpy(k_s, k, (size_t)kd * sizeof(float));
			scale_f32(q_s, qn, kd);
			scale_f32(k_s, kn, kd);

			memset(mem, 0, (size_t)vd * sizeof(float));
			for (int d = 0; d < kd; d++)
				gdn_decay_accum_f32(shead + (size_t)d * vd, mem, decay, k_s[d], vd);
			for (int jj = 0; jj < vd; jj++)
				delta[jj] = (v[jj] - mem[jj]) * b;
			memset(y, 0, (size_t)vd * sizeof(float));
			for (int d = 0; d < kd; d++)
				gdn_update_accum_f32(shead + (size_t)d * vd, y, delta, k_s[d], q_s[d], vd);

			float inv_rms = 1.0f / sqrtf(j->eps + rmsnorm_sum_sq(y, vd) / (float)vd);
			gdn_out_norm_f32(y, j->norm_w, z_t, inv_rms, vd);
		}
	}
}

static void gdn_chunk(int begin, int end, int tid, void *ctx) {
	gdn_job *j = (gdn_job *)ctx;
	gdn_heads(begin, end, j, j->scratch + (size_t)tid * j->scratch_stride);
}

status_code cpu_gated_delta_net(backend *self, const gdn_desc *d) {
	tpool *pool			  = (self && self->get_pool) ? self->get_pool(self) : NULL;
	int	   scratch_stride = 2 * d->state_size + 2 * d->value_head_dim;
	float *conv			  = d->ws;
	float *scratch		  = d->ws + (size_t)d->n_tokens * d->conv_dim;

	gdn_conv_tokens(pool, conv, d->conv_state, d->mixed, d->conv_w, d->conv_dim, d->conv_kernel,
					d->n_tokens, d->conv_dim);

	gdn_job job = {
		.state			= d->state,
		.conv			= conv,
		.z				= d->z,
		.alpha			= d->alpha,
		.beta			= d->beta,
		.out			= d->out,
		.dt				= d->dt,
		.a				= d->a_vec,
		.norm_w			= d->norm_w,
		.scratch		= scratch,
		.n_tokens		= d->n_tokens,
		.conv_stride	= d->conv_dim,
		.z_stride		= d->value_dim,
		.alpha_stride	= d->n_value_heads,
		.beta_stride	= d->n_value_heads,
		.out_stride		= d->value_dim,
		.nkh			= d->n_key_heads,
		.kd				= d->state_size,
		.vd				= d->value_head_dim,
		.key_dim		= d->key_dim,
		.scratch_stride = scratch_stride,
		.eps			= d->eps,
	};
	if (pool && d->n_value_heads > 1)
		tpool_parallel_for(pool, d->n_value_heads, 1, gdn_chunk, &job);
	else
		gdn_heads(0, d->n_value_heads, &job, scratch);
	return OK;
}

void cpu_rope_cs_fill(float *cs, int half, int pos, double theta_scale, const float *freq_factors) {
	double base_freq = 1.0;
	for (int j = 0; j < half; j++) {
		double ff = 1.0;
		if (freq_factors) {
			float fv = freq_factors[j];
			ff		 = (fv >= 1e10f || fv == 0.0f) ? 0.0 : (double)fv;
		}
		if (ff > 0.0) {
			double angle	= base_freq * ff * (double)pos;
			cs[2 * j]		= (float)cos(angle);
			cs[(2 * j) + 1] = (float)sin(angle);
		} else {
			cs[2 * j]		= 1.0f;
			cs[(2 * j) + 1] = 0.0f;
		}
		base_freq *= theta_scale;
	}
}

status_code cpu_rope_cs_cached(cpu_priv *p, int head_dim, int pos, float theta,
							   const float *freq_factors, float **cs_out) {
	int half = head_dim / 2;

	double theta_scale;
	if (p->rope_cs.theta_cached == theta && p->rope_cs.head_dim_cached == head_dim) {
		theta_scale = p->rope_cs.theta_scale;
	} else {
		theta_scale				   = pow((double)theta, -2.0 / (double)head_dim);
		p->rope_cs.theta_scale	   = theta_scale;
		p->rope_cs.theta_cached	   = theta;
		p->rope_cs.head_dim_cached = head_dim;
	}

	int cache_valid =
		p->rope_cs.cs && p->rope_cs.pos_cached == pos && p->rope_cs.theta_cs_cached == theta &&
		p->rope_cs.head_dim_cs_cached == head_dim && p->rope_cs.freq_factors_cached == freq_factors;

	if ((size_t)p->rope_cs.cs_cap < (size_t)half * 2 * sizeof(float)) {
		status_code grow_st = cpu_buf_grow((void **)&p->rope_cs.cs, &p->rope_cs.cs_cap,
										   (size_t)half * 2 * sizeof(float), 1);
		if (grow_st != OK)
			return grow_st;
		cache_valid = 0;
	}

	if (!cache_valid) {
		cpu_rope_cs_fill(p->rope_cs.cs, half, pos, theta_scale, freq_factors);
		p->rope_cs.pos_cached		   = pos;
		p->rope_cs.theta_cs_cached	   = theta;
		p->rope_cs.head_dim_cs_cached  = head_dim;
		p->rope_cs.freq_factors_cached = freq_factors;
	}

	*cs_out = p->rope_cs.cs;
	return OK;
}

status_code cpu_softcap(backend *self, buffer *x, float cap, int n) {
	if (cap <= 0.0f || n <= 0)
		return OK;
	cpu_priv	   *p		= self->priv;
	float		   *xf		= cpu_ptr(x);
	float			inv_cap = 1.0f / cap;
	cpu_softcap_job job		= {.x = xf, .inv_cap = inv_cap, .cap = cap};
	tpool		   *pool	= p ? p->pool : NULL;
	int				cur_tid = tpool_current_tid();
	if (pool && tpool_n_threads(pool) > 1 && n >= 2 * CPU_ELEMWISE_MIN_PER_THREAD) {
		tpool_parallel_for(pool, n, CPU_ELEMWISE_MIN_PER_THREAD, cpu_softcap_chunk, &job);
	} else {
		cpu_softcap_chunk(0, n, cur_tid, &job);
	}
	return OK;
}

status_code cpu_split_qgate(backend *self, const buffer *mixed, buffer *q, buffer *gate,
							int n_heads, int head_dim, int n_rows) {
	(void)self;
	if (n_rows <= 0 || n_heads <= 0 || head_dim <= 0)
		return OK;
	cpu_split_qgate_job job = {.mixed	 = cpu_ptr(mixed),
							   .q		 = cpu_ptr(q),
							   .gate	 = cpu_ptr(gate),
							   .n_heads	 = n_heads,
							   .head_dim = head_dim};
	cpu_priv		   *p	= self->priv;
	cpu_run_batch(p ? p->pool : NULL, n_rows, cpu_split_qgate_chunk, &job);
	return OK;
}

status_code cpu_attn_output_gate(backend *self, buffer *out, const buffer *gate, int n,
								 int n_rows) {
	(void)self;
	if (n <= 0 || n_rows <= 0)
		return OK;
	cpu_attn_output_gate_job job = {.out = cpu_ptr(out), .gate = cpu_ptr(gate), .n = n};
	cpu_priv				*p	 = self->priv;
	cpu_run_batch(p ? p->pool : NULL, n_rows, cpu_attn_output_gate_chunk, &job);
	return OK;
}

status_code cpu_partial_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
								int head_dim, int rope_dim, int pos_start,
								const float *rope_cos_base, const float *rope_sin_base,
								int n_rows) {
	(void)self;
	if (n_rows <= 0 || rope_dim <= 0)
		return OK;
	int						qn	 = n_heads * head_dim;
	int						kn	 = n_kv_heads * head_dim;
	int						half = rope_dim / 2;
	cpu_partial_rope_qk_job job	 = {.q			= cpu_ptr(q),
									.k			= cpu_ptr(k),
									.cos_base	= rope_cos_base,
									.sin_base	= rope_sin_base,
									.qn			= qn,
									.kn			= kn,
									.half		= half,
									.rope_dim	= rope_dim,
									.n_heads	= n_heads,
									.n_kv_heads = n_kv_heads,
									.head_dim	= head_dim,
									.pos0		= pos_start};
	cpu_priv			   *p	 = self->priv;
	cpu_run_batch(p ? p->pool : NULL, n_rows, cpu_partial_rope_qk_chunk, &job);
	return OK;
}

#define CPU_MATMUL_MIN_ROWS_PER_THREAD 32
#define CPU_MATMUL_MULTI_MAX 8
#define CPU_QUANTIZE_MIN_ROWS_PER_THREAD 8

typedef void (*qonly_fn)(const void *, const void *, size_t, float *, int, int, int, int);

typedef struct {
	qonly_fn qonly;
	uint32_t w_type;
	int		 q8_class;
} matmul_kernel;

typedef struct {
	const void			*W;
	uint32_t			 w_type;
	const void			*xq;
	const float			*xf;
	float				*y;
	int					 n, k, m;
	size_t				 row_stride;
	size_t				 xq_row_stride;
	const float			*bias;
	const float			*residual;
	int					 group_rows;
	const matmul_kernel *kernel;
	size_t				 xq_row_stride_blocks;
} cpu_matmul_job;

typedef struct {
	uint32_t	   w_type;
	int			   group_rows;
	tpool_chunk_fn chunk_fn;
} grouped_matmul_kernel;

typedef struct {
	int			 q8_class;
	const float *xf;
	int			 k;
	size_t		 row_stride;
	uint8_t		*xq;
} cpu_quantize_x_rows_job;

typedef struct {
	cpu_matmul_job jobs[CPU_MATMUL_MULTI_MAX];
	int			   row_offset[CPU_MATMUL_MULTI_MAX + 1];
	int			   n_matmuls;
	int			   group_offset[CPU_MATMUL_MULTI_MAX + 1];
	int			   group_rows;
} cpu_matmul_multi_job;

#define MATMUL_KERNEL(wt, fn, q8) [(wt)] = {(qonly_fn)fn, wt, q8}

static const matmul_kernel k_kernel_by_type[GGML_TYPE_Q6_K_REPACKED + 1] = {
	MATMUL_KERNEL(GGML_TYPE_Q4_0, matmul_q4_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_IQ4_NL, matmul_iq4_nl_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_Q8_0, matmul_q8_0_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_Q4_1, matmul_q4_1_q8_qonly_f32, 2),
	MATMUL_KERNEL(GGML_TYPE_Q5_0, matmul_q5_0_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_Q5_1, matmul_q5_1_q8_qonly_f32, 2),
	MATMUL_KERNEL(GGML_TYPE_Q6_K, matmul_q6_k_q8_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_Q4_K, matmul_q4_k_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_Q5_K, matmul_q5_k_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_IQ3_S, matmul_iq3_s_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_IQ3_S_REPACKED, matmul_iq3_s_repacked_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_IQ2_S, matmul_iq2_s_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_IQ2_XXS, matmul_iq2_xxs_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_IQ3_XXS, matmul_iq3_xxs_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_IQ4_XS, matmul_iq4_xs_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_Q8_0_REPACKED, matmul_q8_0_repacked_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_Q4_0_REPACKED, matmul_q4_0_repacked_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_IQ4_NL_REPACKED, matmul_iq4_nl_repacked_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_Q4_K_REPACKED, matmul_q4_k_repacked_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_Q5_K_REPACKED, matmul_q5_k_repacked_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_Q6_K_REPACKED, matmul_q6_k_repacked_q8_k_qonly_f32, 3),
};

static inline void cpu_matmul_add_bias_residual(float *restrict y, const float *restrict bias,
												const float *restrict residual, int n) {
	if (bias) {
		if (residual) {
			for (int i = 0; i < n; i++)
				y[i] += bias[i] + residual[i];
		} else {
			for (int i = 0; i < n; i++)
				y[i] += bias[i];
		}
	} else if (residual) {
		for (int i = 0; i < n; i++)
			y[i] += residual[i];
	}
}

static const matmul_kernel *matmul_kernel_lookup(uint32_t w_type) {
	if (w_type >= ARRAY_LEN(k_kernel_by_type))
		return NULL;
	const matmul_kernel *k = &k_kernel_by_type[w_type];
	return k->qonly ? k : NULL;
}

static size_t cpu_matmul_q8_block_size(int q8_class) {
	switch (q8_class) {
	case 1:
		return sizeof(q8_0_block);
	case 2:
		return sizeof(q8_1_block);
	case 3:
		return sizeof(q8_k_block);
	default:
		return 0;
	}
}

static inline void cpu_matmul_job_prepare_kernel(cpu_matmul_job *j) {
	j->kernel				= matmul_kernel_lookup(j->w_type);
	j->xq_row_stride_blocks = 0;
	if (j->kernel)
		j->xq_row_stride_blocks =
			j->kernel->q8_class ? j->xq_row_stride / cpu_matmul_q8_block_size(j->kernel->q8_class)
								: 0;
}

static uint32_t cpu_matmul_q8_expected_type(int q8_class) {
	switch (q8_class) {
	case 1:
		return GGML_TYPE_Q8_0;
	case 2:
		return GGML_TYPE_Q8_1;
	case 3:
		return GGML_TYPE_Q8_K;
	default:
		return 0;
	}
}

static void cpu_matmul_generic_block(const cpu_matmul_job *j, const uint8_t *restrict W_sub,
									 float *restrict y_sub, int n_sub) {
	static _Thread_local float *row_buf;
	static _Thread_local size_t row_buf_cap;
	if (row_buf_cap < (size_t)j->k) {
		row_buf = xrealloc(row_buf, (size_t)j->k * sizeof(float));
		if (!row_buf_cap)
			tlocal_register((void **)&row_buf);
		row_buf_cap = (size_t)j->k;
	}
	dequant_blocks_fn dequant_fn;
	int				  divisor;
	dequant_row_plan(j->w_type, &dequant_fn, &divisor);
	size_t n_blocks = (size_t)(j->k / divisor);
	for (int r = 0; r < n_sub; r++) {
		dequant_fn(W_sub + ((size_t)r * j->row_stride), n_blocks, row_buf);
		for (int t = 0; t < j->m; t++)
			y_sub[((size_t)t * j->n) + r] = dot_f32(row_buf, j->xf + ((size_t)t * j->k), j->k);
	}
}

static void cpu_matmul_rows_worker(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_matmul_job *j			  = ctx;
	int				n_sub		  = end - begin;
	const uint8_t *restrict W_sub = (const uint8_t *)j->W + ((size_t)begin * j->row_stride);
	float *restrict y_sub		  = j->y + begin;

	const float *res_ptr = j->residual;
	int			 res_off = begin;

	const matmul_kernel *entry = j->kernel;
	if (entry) {
		entry->qonly(W_sub, j->xq, j->xq_row_stride_blocks, y_sub, j->n, n_sub, j->k, j->m);
	} else if (j->w_type == GGML_TYPE_F32) {
		matmul_f32_f32_batch((const float *)W_sub, j->xf, y_sub, n_sub, j->k, j->m, j->k, j->n);
	} else if (j->w_type == GGML_TYPE_F16) {
		matmul_f16_f32_batch(W_sub, j->xf, y_sub, n_sub, j->k, j->m, j->k, j->n);
	} else if (j->w_type == GGML_TYPE_BF16) {
		matmul_bf16_f32_batch(W_sub, j->xf, y_sub, n_sub, j->k, j->m, j->k, j->n);
	} else {
		cpu_matmul_generic_block(j, W_sub, y_sub, n_sub);
	}

	if (j->m == 1)
		cpu_matmul_add_bias_residual(y_sub, j->bias ? j->bias + begin : NULL,
									 j->residual ? res_ptr + res_off : NULL, n_sub);
}

static void cpu_matmul_groups_worker(int begin, int end, int tid, void *ctx) {
	cpu_matmul_job *j		  = ctx;
	int				gr		  = j->group_rows;
	int				row_begin = begin * gr;
	int				row_end	  = end * gr;
	if (row_begin < 0)
		row_begin = 0;
	if (row_end > j->n)
		row_end = j->n;
	if (row_begin >= row_end)
		return;
	cpu_matmul_rows_worker(row_begin, row_end, tid, ctx);
}

#define GROUPED_KERNEL(wt, rows, fn) [(wt) - GGML_TYPE_Q8_0_REPACKED] = {wt, rows, fn}

static const grouped_matmul_kernel k_grouped_by_type[] = {
	GROUPED_KERNEL(GGML_TYPE_Q8_0_REPACKED, Q8_0_REPACKED_ROWS, cpu_matmul_groups_worker),
	GROUPED_KERNEL(GGML_TYPE_Q4_0_REPACKED, Q4_0_REPACKED_ROWS, cpu_matmul_groups_worker),
	GROUPED_KERNEL(GGML_TYPE_IQ3_S_REPACKED, IQ3_S_REPACKED_ROWS, cpu_matmul_groups_worker),
	GROUPED_KERNEL(GGML_TYPE_IQ4_NL_REPACKED, IQ4_NL_REPACKED_ROWS, cpu_matmul_groups_worker),
	GROUPED_KERNEL(GGML_TYPE_Q4_K_REPACKED, Q4_K_REPACKED_ROWS, cpu_matmul_groups_worker),
	GROUPED_KERNEL(GGML_TYPE_Q5_K_REPACKED, Q5_K_REPACKED_ROWS, cpu_matmul_groups_worker),
	GROUPED_KERNEL(GGML_TYPE_Q6_K_REPACKED, Q6_K_REPACKED_ROWS, cpu_matmul_groups_worker),
};

static const grouped_matmul_kernel *grouped_matmul_kernel_lookup(uint32_t w_type) {
	if (w_type < GGML_TYPE_Q8_0_REPACKED)
		return NULL;
	size_t idx = (size_t)w_type - GGML_TYPE_Q8_0_REPACKED;
	if (idx >= ARRAY_LEN(k_grouped_by_type))
		return NULL;
	return &k_grouped_by_type[idx];
}

static void cpu_matmul_dispatch_rows(tpool *pool, uint32_t w_type, int n_rows, void *job,
									 tpool_chunk_fn plain_chunk_fn) {
	const grouped_matmul_kernel *gk = grouped_matmul_kernel_lookup(w_type);
	if (!gk) {
		tpool_parallel_for(pool, n_rows, CPU_MATMUL_MIN_ROWS_PER_THREAD, plain_chunk_fn, job);
		return;
	}
	((cpu_matmul_job *)job)->group_rows = gk->group_rows;
	int n_groups						= (n_rows + gk->group_rows - 1) / gk->group_rows;
	int min_groups_per_thr				= CPU_MATMUL_MIN_ROWS_PER_THREAD / gk->group_rows;
	if (min_groups_per_thr < 1)
		min_groups_per_thr = 1;
	tpool_parallel_for(pool, n_groups, min_groups_per_thr, gk->chunk_fn, job);
}

static size_t cpu_matmul_w_row_stride(uint32_t w_type, int k) {
	switch (w_type) {
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_0_REPACKED:
		return (size_t)(k / 32) * sizeof(q4_0_block);
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ4_NL_REPACKED:
		return (size_t)(k / 32) * sizeof(iq4_nl_block);
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_0_REPACKED:
		return (size_t)(k / 32) * sizeof(q8_0_block);
	case GGML_TYPE_Q4_1:
		return (size_t)(k / 32) * sizeof(q4_1_block);
	case GGML_TYPE_Q5_0:
		return (size_t)(k / 32) * sizeof(q5_0_block);
	case GGML_TYPE_Q5_1:
		return (size_t)(k / 32) * sizeof(q5_1_block);
	case GGML_TYPE_F32:
		return (size_t)k * sizeof(float);
	case GGML_TYPE_F16:
		return (size_t)k * sizeof(uint16_t);
	case GGML_TYPE_BF16:
		return (size_t)k * sizeof(uint16_t);
	default:

		return ggml_row_size(w_type, (size_t)k);
	}
}

static size_t cpu_matmul_xq_row_stride(int q8_class, int k) {
	switch (q8_class) {
	case 1:
		return (size_t)(k / 32) * sizeof(q8_0_block);
	case 2:
		return (size_t)(k / 32) * sizeof(q8_1_block);
	case 3:
		return (size_t)(k / 256) * sizeof(q8_k_block);
	default:
		return 0;
	}
}

static void cpu_quantize_class(const float *restrict xf, void *restrict dst, int k, int q8_class) {
	switch (q8_class) {
	case 1:
		quantize_q8_0(xf, dst, k);
		break;
	case 2:
		quantize_q8_1(xf, dst, k);
		break;
	case 3:
		quantize_q8_k(xf, dst, k);
		break;
	default:
		break;
	}
}

static void cpu_quantize_x_rows_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_quantize_x_rows_job *j = ctx;
	for (int i = begin; i < end; i++) {
		const float *xi	 = j->xf + ((size_t)i * j->k);
		void		*dst = j->xq + ((size_t)i * j->row_stride);
		cpu_quantize_class(xi, dst, j->k, j->q8_class);
	}
}

static void *cpu_matmul_quantize_x_rows(backend *self, quant_scratch *qs, int q8_class,
										const float *xf, int k, int m, size_t row_stride) {
	quant_scratch_ensure(qs, (size_t)m * row_stride);
	uint8_t *xq = qs->q8_buf;

	cpu_priv *p = self->priv;

	if (p->pool && m >= 2 * CPU_QUANTIZE_MIN_ROWS_PER_THREAD) {
		cpu_quantize_x_rows_job job = {
			.q8_class = q8_class, .xf = xf, .k = k, .row_stride = row_stride, .xq = xq};
		tpool_parallel_for(p->pool, m, CPU_QUANTIZE_MIN_ROWS_PER_THREAD, cpu_quantize_x_rows_chunk,
						   &job);
		return xq;
	}

	for (int i = 0; i < m; i++) {
		const float *xi	 = xf + ((size_t)i * k);
		void		*dst = xq + ((size_t)i * row_stride);
		cpu_quantize_class(xi, dst, k, q8_class);
	}
	return xq;
}

static void *cpu_matmul_quantize_x(quant_scratch *qs, int q8_class, const float *x, int k) {
	if (q8_class < 1 || q8_class > 3)
		return NULL;
	quant_scratch_ensure(qs, cpu_matmul_xq_row_stride(q8_class, k));
	cpu_quantize_class(x, qs->q8_buf, k, q8_class);
	return qs->q8_buf;
}

static void cpu_matmul_one(const void *restrict W, uint32_t w_type, const float *restrict x,
						   float *restrict y, int n, int k, quant_scratch *qs) {
	const matmul_kernel *entry = matmul_kernel_lookup(w_type);
	if (entry && entry->q8_class) {
		void *xq = cpu_matmul_quantize_x(qs, entry->q8_class, x, k);
		if (xq) {
			entry->qonly(W, xq, 0, y, n, n, k, 1);
			return;
		}
	}
	switch (w_type) {
	case GGML_TYPE_F32:
		matmul_f32_f32(W, x, y, n, k);
		break;
	case GGML_TYPE_F16:
		matmul_f16_f32(W, x, y, n, k);
		break;
	case GGML_TYPE_BF16:
		matmul_bf16_f32(W, x, y, n, k);
		break;
	default:
		matmul_generic_f32(W, w_type, x, y, n, k);
		break;
	}
}

static void cpu_matmul_threaded_bias_residual(backend *self, const void *restrict W,
											  uint32_t w_type, const float *restrict x,
											  float *restrict y, int n, int k,
											  const float *restrict bias,
											  const float *restrict residual) {
	cpu_priv *p = self->priv;

	int cur_tid			 = tpool_current_tid();
	int aliases_residual = (residual != NULL && residual == y);

	if (!p->pool || n < 2 * CPU_MATMUL_MIN_ROWS_PER_THREAD) {
		const float *res = residual;
		if (aliases_residual) {
			status_code grow_st = cpu_buf_grow((void **)&p->residual_tmp, &p->residual_tmp_cap,
											   (size_t)n * sizeof(float), 64);
			if (grow_st != OK)
				return;
			memcpy(p->residual_tmp, residual, (size_t)n * sizeof(float));
			res = p->residual_tmp;
		}
		quant_scratch *qs = cpu_scratch_for_tid(p, cur_tid);
		cpu_matmul_one(W, w_type, x, y, n, k, qs);
		cpu_matmul_add_bias_residual(y, bias, res, n);
		return;
	}

	cpu_matmul_job job = {.W		= W,
						  .w_type	= w_type,
						  .n		= n,
						  .k		= k,
						  .m		= 1,
						  .y		= y,
						  .xf		= x,
						  .bias		= bias,
						  .residual = residual};
	cpu_matmul_job_prepare_kernel(&job);

	int q8_class = job.kernel ? job.kernel->q8_class : 0;
	if (q8_class) {
		job.xq = cpu_matmul_quantize_x(&p->qscratch, q8_class, x, k);
	}
	job.row_stride = cpu_matmul_w_row_stride(w_type, k);

	if (aliases_residual && residual) {
		status_code grow_st = cpu_buf_grow((void **)&p->residual_tmp, &p->residual_tmp_cap,
										   (size_t)n * sizeof(float), 64);
		if (grow_st != OK)
			return;
		memcpy(p->residual_tmp, residual, (size_t)n * sizeof(float));
		job.residual = p->residual_tmp;
	}

	cpu_matmul_dispatch_rows(p->pool, w_type, n, &job, cpu_matmul_rows_worker);
}

static void cpu_matmul_multi_worker(int begin, int end, int tid, void *ctx) {
	cpu_matmul_multi_job *mj = ctx;

	for (int i = 0; i < mj->n_matmuls; i++) {
		int lo = mj->row_offset[i];
		int hi = mj->row_offset[i + 1];
		if (end <= lo)
			break;
		int b = begin > lo ? begin : lo;
		int e = end < hi ? end : hi;
		if (b >= e)
			continue;
		cpu_matmul_rows_worker(b - lo, e - lo, tid, &mj->jobs[i]);
	}
}

static void cpu_matmul_multi_groups_worker(int begin, int end, int tid, void *ctx) {
	cpu_matmul_multi_job *mj = ctx;
	int					  gr = mj->group_rows;

	for (int i = 0; i < mj->n_matmuls; i++) {
		int lo = mj->group_offset[i];
		int hi = mj->group_offset[i + 1];
		if (end <= lo)
			break;
		int b = begin > lo ? begin : lo;
		int e = end < hi ? end : hi;
		if (b >= e)
			continue;
		int row_begin = (b - lo) * gr;
		int row_end	  = (e - lo) * gr;
		if (row_end > mj->jobs[i].n)
			row_end = mj->jobs[i].n;
		if (row_begin >= row_end)
			continue;
		cpu_matmul_rows_worker(row_begin, row_end, tid, &mj->jobs[i]);
	}
}

status_code cpu_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
							 const buffer *x, buffer **y, const int *n_list, int k, int n_matmuls) {
	cpu_priv	*p	= self->priv;
	const float *xf = cpu_ptr(x);

	int total_rows = 0;
	for (int i = 0; i < n_matmuls; i++)
		total_rows += n_list[i];

	int cur_tid = tpool_current_tid();

	int has_grouped = 0;
	for (int i = 0; i < n_matmuls; i++)
		if (grouped_matmul_kernel_lookup(w_types[i]))
			has_grouped = 1;

	if (!p->pool || n_matmuls > CPU_MATMUL_MULTI_MAX ||
		total_rows < 2 * CPU_MATMUL_MIN_ROWS_PER_THREAD) {

		quant_scratch *qs = cpu_scratch_for_tid(p, cur_tid);

		void *xq_by_class[4] = {NULL, NULL, NULL, NULL};

		for (int i = 0; i < n_matmuls; i++) {
			float				*yf	   = cpu_ptr(y[i]);
			int					 n	   = n_list[i];
			const void			*W	   = cpu_ptr(w[i]);
			uint32_t			 w_t   = w_types[i];
			const matmul_kernel *entry = matmul_kernel_lookup(w_t);
			if (entry) {
				int q8_class = entry->q8_class;
				if (!xq_by_class[q8_class])
					xq_by_class[q8_class] = cpu_matmul_quantize_x(qs, q8_class, xf, k);
				entry->qonly(W, xq_by_class[q8_class], 0, yf, n, n, k, 1);
			} else {
				cpu_matmul_one(W, w_t, xf, yf, n, k, qs);
			}
		}
		return OK;
	}

	quant_scratch		*local_scratch	= p->matmul_multi_scratch;
	void				*xq_by_class[4] = {NULL, NULL, NULL, NULL};
	cpu_matmul_multi_job mj				= {.n_matmuls = n_matmuls};
	mj.row_offset[0]					= 0;
	for (int i = 0; i < n_matmuls; i++) {
		mj.row_offset[i + 1] = mj.row_offset[i] + n_list[i];

		cpu_matmul_job *j = &mj.jobs[i];
		j->W			  = cpu_ptr(w[i]);
		j->w_type		  = w_types[i];
		j->y			  = cpu_ptr(y[i]);
		j->k			  = k;
		j->n			  = n_list[i];
		j->m			  = 1;
		j->xf			  = xf;

		cpu_matmul_job_prepare_kernel(j);
		int q8_class = j->kernel ? j->kernel->q8_class : 0;
		if (q8_class) {
			if (!xq_by_class[q8_class])
				xq_by_class[q8_class] =
					cpu_matmul_quantize_x(&local_scratch[q8_class], q8_class, xf, k);
			j->xq = xq_by_class[q8_class];
		}
		j->row_stride = cpu_matmul_w_row_stride(w_types[i], k);
	}

	if (has_grouped) {
		int all_grouped = 1;
		int gr			= 0;
		for (int i = 0; i < n_matmuls; i++) {
			const grouped_matmul_kernel *gk = grouped_matmul_kernel_lookup(w_types[i]);
			if (!gk) {
				all_grouped = 0;
				break;
			}
			if (gr == 0)
				gr = gk->group_rows;
			else if (gk->group_rows != gr) {
				all_grouped = 0;
				break;
			}
		}

		if (all_grouped && gr > 0) {
			mj.group_rows	   = gr;
			mj.group_offset[0] = 0;
			int total_groups   = 0;
			for (int i = 0; i < n_matmuls; i++) {
				int n_groups_i = (n_list[i] + gr - 1) / gr;
				total_groups += n_groups_i;
				mj.group_offset[i + 1] = total_groups;
				mj.jobs[i].group_rows  = gr;
			}
			int min_groups_per_thr = CPU_MATMUL_MIN_ROWS_PER_THREAD / gr;
			if (min_groups_per_thr < 1)
				min_groups_per_thr = 1;
			tpool_parallel_for(p->pool, total_groups, min_groups_per_thr,
							   cpu_matmul_multi_groups_worker, &mj);
			return OK;
		}

		for (int i = 0; i < n_matmuls; i++)
			cpu_matmul_dispatch_rows(p->pool, w_types[i], n_list[i], &mj.jobs[i],
									 cpu_matmul_rows_worker);
		return OK;
	}

	tpool_parallel_for(p->pool, total_rows, CPU_MATMUL_MIN_ROWS_PER_THREAD, cpu_matmul_multi_worker,
					   &mj);

	return OK;
}

status_code cpu_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
								   const buffer *x, buffer **y, const int *n_list, int k,
								   int n_matmuls, int m) {
	if (m <= 1) {
		buffer *y1[CPU_MATMUL_MULTI_MAX];
		for (int i = 0; i < n_matmuls; i++)
			y1[i] = y[i];
		return cpu_matmul_multi(self, w, w_types, x, y1, n_list, k, n_matmuls);
	}

	cpu_priv	*p	= self->priv;
	const float *xf = cpu_ptr(x);

	if (n_matmuls > CPU_MATMUL_MULTI_MAX)
		return ERR_INVALID_ARG;

	quant_scratch *local_scratch  = p->matmul_multi_scratch;
	void		  *xq_by_class[4] = {NULL, NULL, NULL, NULL};

	cpu_matmul_multi_job mj = {.n_matmuls = n_matmuls};
	mj.row_offset[0]		= 0;
	int has_grouped			= 0;
	for (int i = 0; i < n_matmuls; i++) {
		cpu_matmul_job *j = &mj.jobs[i];
		j->W			  = cpu_ptr(w[i]);
		j->w_type		  = w_types[i];
		j->y			  = cpu_ptr(y[i]);
		j->k			  = k;
		j->xf			  = xf;

		cpu_matmul_job_prepare_kernel(j);
		int q8_class = j->kernel ? j->kernel->q8_class : 0;
		if (q8_class) {
			if (!xq_by_class[q8_class]) {
				size_t row_stride	  = cpu_matmul_xq_row_stride(q8_class, k);
				xq_by_class[q8_class] = cpu_matmul_quantize_x_rows(self, &local_scratch[q8_class],
																   q8_class, xf, k, m, row_stride);
			}
			j->xq					= xq_by_class[q8_class];
			j->xq_row_stride		= cpu_matmul_xq_row_stride(q8_class, k);
			j->xq_row_stride_blocks = j->xq_row_stride / cpu_matmul_q8_block_size(q8_class);
		}
		j->n				 = n_list[i];
		j->m				 = m;
		mj.row_offset[i + 1] = mj.row_offset[i] + n_list[i];
		j->row_stride		 = cpu_matmul_w_row_stride(w_types[i], k);
		if (grouped_matmul_kernel_lookup(w_types[i]))
			has_grouped = 1;
	}

	int total_rows = mj.row_offset[n_matmuls];

	if (!p->pool || (size_t)m * (size_t)total_rows < 2 * CPU_MATMUL_MIN_ROWS_PER_THREAD) {
		for (int i = 0; i < n_matmuls; i++) {
			cpu_matmul_job *j = &mj.jobs[i];
			if (j->n == 0)
				continue;
			cpu_matmul_rows_worker(0, j->n, -1, j);
		}
		return OK;
	}

	if (has_grouped) {
		int all_grouped = 1;
		int gr			= 0;
		for (int i = 0; i < n_matmuls; i++) {
			const grouped_matmul_kernel *gk = grouped_matmul_kernel_lookup(w_types[i]);
			if (!gk) {
				all_grouped = 0;
				break;
			}
			if (gr == 0)
				gr = gk->group_rows;
			else if (gk->group_rows != gr) {
				all_grouped = 0;
				break;
			}
		}

		if (all_grouped && gr > 0) {
			mj.group_rows	   = gr;
			mj.group_offset[0] = 0;
			int total_groups   = 0;
			for (int i = 0; i < n_matmuls; i++) {
				int n_groups_i = (n_list[i] + gr - 1) / gr;
				total_groups += n_groups_i;
				mj.group_offset[i + 1] = total_groups;
				mj.jobs[i].group_rows  = gr;
			}
			int min_groups_per_thr = CPU_MATMUL_MIN_ROWS_PER_THREAD / gr;
			if (min_groups_per_thr < 1)
				min_groups_per_thr = 1;
			tpool_parallel_for(p->pool, total_groups, min_groups_per_thr,
							   cpu_matmul_multi_groups_worker, &mj);
			return OK;
		}

		for (int i = 0; i < n_matmuls; i++) {
			cpu_matmul_job *j = &mj.jobs[i];
			if (j->n == 0)
				continue;
			cpu_matmul_dispatch_rows(p->pool, w_types[i], n_list[i], j, cpu_matmul_rows_worker);
		}
		return OK;
	}

	tpool_parallel_for(p->pool, total_rows, CPU_MATMUL_MIN_ROWS_PER_THREAD, cpu_matmul_multi_worker,
					   &mj);

	return OK;
}

status_code cpu_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x, buffer *y,
					   int n, int k) {
	return cpu_matmul_batch(self, w, w_type, x, y, n, k, 1);
}
status_code cpu_matmul_residual(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								const buffer *residual, buffer *y, int n, int k) {
	const void	*W	= cpu_ptr(w);
	const float *xf = cpu_ptr(x);
	float		*yf = cpu_ptr(y);
	const float *rf = cpu_ptr(residual);

	cpu_matmul_threaded_bias_residual(self, W, w_type, xf, yf, n, k, NULL, rf);
	return OK;
}

static status_code cpu_matmul_batch_prequant(backend *self, const void *W, uint32_t w_type,
											 const void *xq, size_t xq_row_stride, float *yf, int n,
											 int k, int m) {
	cpu_priv	   *p = self->priv;
	cpu_matmul_job	job_storage;
	cpu_matmul_job *job = &job_storage;
	memset(job, 0, sizeof(*job));
	job->W		= W;
	job->w_type = w_type;
	job->y		= yf;
	job->n		= n;
	job->k		= k;
	job->m		= m;
	cpu_matmul_job_prepare_kernel(job);
	int q8_class = job->kernel ? job->kernel->q8_class : 0;
	if (!q8_class)
		return ERR_UNSUPPORTED;
	job->xq					  = (void *)xq;
	job->xq_row_stride		  = xq_row_stride;
	job->xq_row_stride_blocks = xq_row_stride / cpu_matmul_q8_block_size(q8_class);
	job->row_stride			  = cpu_matmul_w_row_stride(w_type, k);
	if (!p->pool || !p->thread_scratch ||
		(size_t)m * (size_t)n < 2 * CPU_MATMUL_MIN_ROWS_PER_THREAD) {
		cpu_matmul_rows_worker(0, n, 0, job);
		return OK;
	}
	cpu_matmul_dispatch_rows(p->pool, w_type, n, job, cpu_matmul_rows_worker);
	return OK;
}

status_code cpu_matmul_batch(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							 buffer *y, int n, int k, int m) {
	cpu_priv	*p	= self->priv;
	const float *xf = cpu_ptr(x);
	float		*yf = cpu_ptr(y);
	const void	*W	= cpu_ptr(w);

	int cur_tid = tpool_current_tid();

	if (!p->pool || !p->thread_scratch ||
		(size_t)m * (size_t)n < 2 * CPU_MATMUL_MIN_ROWS_PER_THREAD) {
		quant_scratch *qs = cpu_scratch_for_tid(p, cur_tid);
		for (int i = 0; i < m; i++)
			cpu_matmul_one(W, w_type, xf + ((size_t)i * k), yf + ((size_t)i * n), n, k, qs);
		return OK;
	}

	cpu_matmul_job	job_storage;
	cpu_matmul_job *job = &job_storage;
	memset(job, 0, sizeof(*job));
	job->W		= W;
	job->w_type = w_type;
	job->xf		= xf;
	job->y		= yf;
	job->n		= n;
	job->k		= k;
	job->m		= m;
	cpu_matmul_job_prepare_kernel(job);

	int q8_class = job->kernel ? job->kernel->q8_class : 0;
	if (q8_class) {
		size_t stride = cpu_matmul_xq_row_stride(q8_class, k);
		void  *xq	  = cpu_matmul_quantize_x_rows(self, &p->qscratch, q8_class, xf, k, m, stride);
		return cpu_matmul_batch_prequant(self, W, w_type, xq, stride, yf, n, k, m);
	}
	job->row_stride = cpu_matmul_w_row_stride(w_type, k);

	cpu_matmul_dispatch_rows(p->pool, w_type, n, job, cpu_matmul_rows_worker);
	return OK;
}

status_code cpu_prequantize_x(backend *self, const buffer *x, int k, uint32_t q8_type,
							  buffer *x_q8_out) {
	cpu_priv *p = self->priv;

	int q8_class;
	switch (q8_type) {
	case GGML_TYPE_Q8_0:
		q8_class = 1;
		break;
	case GGML_TYPE_Q8_1:
		q8_class = 2;
		break;
	case GGML_TYPE_Q8_K:
		q8_class = 3;
		break;
	default:
		return ERR_UNSUPPORTED;
	}

	size_t		need	= cpu_matmul_xq_row_stride(q8_class, k);
	status_code grow_st = cpu_buf_grow(&p->xq8_buf, &p->xq8_buf_cap, need, 1);
	if (grow_st != OK)
		return grow_st;
	cpu_quantize_class(cpu_ptr(x), p->xq8_buf, k, q8_class);
	x_q8_out->handle   = p->xq8_buf;
	x_q8_out->size	   = need;
	x_q8_out->host_ptr = NULL;
	x_q8_out->owner	   = NULL;
	return OK;
}

status_code cpu_matmul_qonly(backend *self, const buffer *w, uint32_t w_type, const buffer *x_q8,
							 uint32_t q8_type, buffer *y, int n, int k, int m) {
	(void)self;
	const matmul_kernel *entry = matmul_kernel_lookup(w_type);
	if (!entry)
		return ERR_UNSUPPORTED;
	uint32_t want_q8 = cpu_matmul_q8_expected_type(entry->q8_class);
	if (!want_q8 || q8_type != want_q8)
		return ERR_INVALID_ARG;
	if (m <= 0)
		m = 1;
	size_t xq_row_stride_blocks = (size_t)(k / 32);
	int	   y_row_stride			= n;
	entry->qonly(cpu_ptr(w), x_q8->handle, xq_row_stride_blocks, cpu_ptr(y), y_row_stride, n, k, m);
	return OK;
}

status_code cpu_matmul_thread_local(backend *self, const void *W, uint32_t w_type, const float *x,
									float *y, int n, int k, int tid) {
	cpu_priv *p = self->priv;
	if (!p)
		return ERR_INVALID_ARG;
	quant_scratch *qs = cpu_scratch_for_tid(p, tid);
	cpu_matmul_one(W, w_type, x, y, n, k, qs);
	return OK;
}

status_code cpu_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type, const buffer *gate,
								const buffer *up, buffer *y, int n, int k, int activation) {
	cpu_priv	*p		= self->priv;
	const void	*W		= cpu_ptr(w);
	const float *gate_f = cpu_ptr(gate);
	const float *up_f	= cpu_ptr(up);
	float		*yf		= cpu_ptr(y);

	status_code grow_st = cpu_buf_grow((void **)&p->residual_tmp, &p->residual_tmp_cap,
									   (size_t)k * sizeof(float), 64);
	if (grow_st != OK)
		return grow_st;
	float *act = p->residual_tmp;

	cpu_ffn_down_act_args a		  = {.g = gate_f, .u = up_f, .o = act, .activation = activation};
	int					  cur_tid = tpool_current_tid();
	if (p->pool && k >= 2 * CPU_ELEMWISE_MIN_PER_THREAD) {
		tpool_parallel_for(p->pool, k, CPU_ELEMWISE_MIN_PER_THREAD, cpu_ffn_down_act_chunk, &a);
	} else {
		cpu_ffn_down_act_chunk(0, k, cur_tid, &a);
	}

	cpu_matmul_threaded_bias_residual(self, W, w_type, act, yf, n, k, NULL, NULL);
	return OK;
}

enum { FFN_ACTQ_TILE = 256 };

typedef struct {
	const float *g, *u;
	uint8_t		*xq;
	int			 k;
	size_t		 stride;
	int			 activation;
	int			 q8_class;
} cpu_ffn_actq_job;

static void ffn_actq_row(const float *g, const float *u, uint8_t *xq, int k, int activation,
						 int q8_class) {
	float tmp[FFN_ACTQ_TILE];
	for (int b = 0; b < k; b += FFN_ACTQ_TILE) {
		int					  n = k - b < FFN_ACTQ_TILE ? k - b : FFN_ACTQ_TILE;
		cpu_ffn_down_act_args a = {.g = g + b, .u = u + b, .o = tmp, .activation = activation};
		cpu_ffn_down_act_chunk(0, n, 0, &a);
		if (q8_class == 3) {
			quantize_q8_k(tmp, (q8_k_block *)(xq + (size_t)(b / 256) * sizeof(q8_k_block)), 256);
			continue;
		}
		size_t esz = (q8_class == 1) ? sizeof(q8_0_block) : sizeof(q8_1_block);
		for (int j = 0; j < n; j += 32) {
			void *dst = xq + (size_t)((b + j) / 32) * esz;
			if (q8_class == 1)
				quantize_q8_0(tmp + j, (q8_0_block *)dst, 32);
			else
				quantize_q8_1(tmp + j, dst, 32);
		}
	}
}

static void cpu_ffn_actq_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_actq_job *j = ctx;
	for (int i = begin; i < end; i++)
		ffn_actq_row(j->g + (size_t)i * j->k, j->u + (size_t)i * j->k,
					 j->xq + (size_t)i * j->stride, j->k, j->activation, j->q8_class);
}

status_code cpu_matmul_ffn_down_batch(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *gate, const buffer *up, buffer *y, int n, int k,
									  int activation, int m) {
	cpu_priv *p = self->priv;
	if (m <= 0)
		return ERR_INVALID_ARG;
	const matmul_kernel *entry	  = matmul_kernel_lookup(w_type);
	int					 q8_class = entry ? entry->q8_class : 0;
	if (m == 1 && !q8_class)
		return cpu_matmul_ffn_down(self, w, w_type, gate, up, y, n, k, activation);
	const float *gate_f = cpu_ptr(gate);
	const float *up_f	= cpu_ptr(up);
	const void	*W		= cpu_ptr(w);
	float		*yf		= cpu_ptr(y);

	if (q8_class) {
		size_t stride = cpu_matmul_xq_row_stride(q8_class, k);
		quant_scratch_ensure(&p->qscratch, (size_t)m * stride);
		uint8_t *xq		 = p->qscratch.q8_buf;
		int		 cur_tid = tpool_current_tid();
		if (p->pool && m >= 2 * CPU_QUANTIZE_MIN_ROWS_PER_THREAD) {
			cpu_ffn_actq_job job = {.g			= gate_f,
									.u			= up_f,
									.xq			= xq,
									.k			= k,
									.stride		= stride,
									.activation = activation,
									.q8_class	= q8_class};
			tpool_parallel_for(p->pool, m, CPU_QUANTIZE_MIN_ROWS_PER_THREAD, cpu_ffn_actq_chunk,
							   &job);
		} else {
			for (int i = 0; i < m; i++)
				ffn_actq_row(gate_f + (size_t)i * k, up_f + (size_t)i * k, xq + (size_t)i * stride,
							 k, activation, q8_class);
		}
		(void)cur_tid;
		return cpu_matmul_batch_prequant(self, W, w_type, xq, stride, yf, n, k, m);
	}

	status_code grow_st = cpu_buf_grow((void **)&p->residual_tmp, &p->residual_tmp_cap,
									   (size_t)m * (size_t)k * sizeof(float), 64);
	if (grow_st != OK)
		return grow_st;
	float *act = p->residual_tmp;

	cpu_ffn_down_act_args a		  = {.g = gate_f, .u = up_f, .o = act, .activation = activation};
	int					  cur_tid = tpool_current_tid();
	int					  total	  = m * k;
	if (p->pool && total >= 2 * CPU_ELEMWISE_MIN_PER_THREAD) {
		tpool_parallel_for(p->pool, total, CPU_ELEMWISE_MIN_PER_THREAD, cpu_ffn_down_act_chunk, &a);
	} else {
		cpu_ffn_down_act_chunk(0, total, cur_tid, &a);
	}

	buffer act_b = {0};
	act_b.handle = act;
	act_b.owner	 = self;
	act_b.size	 = (size_t)m * (size_t)k * sizeof(float);
	return cpu_matmul_batch(self, w, w_type, &act_b, y, n, k, m);
}
