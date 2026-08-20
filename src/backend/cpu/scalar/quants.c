#include "backend/cpu/scalar/quants.h"
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

#define IQ3_S_RE_OFF_D 0
#define IQ3_S_RE_OFF_SCALES 2
#define IQ3_S_RE_OFF_IDX 6

#define MATMUL_K_BLOCK 64

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

static float f16_to_f32_table[65536];

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

__attribute__((weak)) float f16_to_f32(uint16_t h) {
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

__attribute__((weak)) void dequant_q4_0_row(const void *blocks, size_t n_blocks, float *dst) {
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

__attribute__((weak)) void dequant_q4_1_row(const void *blocks, size_t n_blocks, float *dst) {
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

__attribute__((weak)) void dequant_q5_0_row(const void *blocks, size_t n_blocks, float *dst) {
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

__attribute__((weak)) void dequant_q5_1_row(const void *blocks, size_t n_blocks, float *dst) {
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

__attribute__((weak)) void dequant_q8_0_row(const void *blocks, size_t n_blocks, float *dst) {
	const q8_0_block *b = blocks;
	for (size_t bi = 0; bi < n_blocks; bi++) {
		float d = f16_to_f32(b[bi].d);
		for (int j = 0; j < 32; j++)
			dst[(bi * 32) + j] = (float)b[bi].qs[j] * d;
	}
}

__attribute__((weak)) void dequant_q3_k_row(const void *blocks, size_t n_blocks, float *dst) {
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

__attribute__((weak)) void dequant_q4_k_row(const void *blocks, size_t n_blocks, float *dst) {
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

__attribute__((weak)) void dequant_q5_k_row(const void *blocks, size_t n_blocks, float *dst) {
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

__attribute__((weak)) void dequant_q6_k_row(const void *blocks, size_t n_blocks, float *dst) {
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

const uint32_t iq3s_grid[512] = {
	0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
	0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
	0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
	0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
	0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
	0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
	0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
	0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
	0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
	0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
	0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
	0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
	0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
	0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
	0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
	0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
	0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
	0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
	0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
	0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
	0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
	0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
	0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
	0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
	0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
	0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
	0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
	0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
	0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
	0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
	0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
	0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
	0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
	0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
	0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
	0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
	0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
	0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
	0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
	0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
	0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
	0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
	0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
	0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
	0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
	0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
	0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
	0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
	0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
	0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
	0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
	0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
	0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
	0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
	0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
	0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
	0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
	0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
	0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
	0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
	0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
	0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
	0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
	0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

typedef iq3_s_block iq3s_block;

__attribute__((weak)) void dequant_iq3_s_row(const void *blocks, size_t n_blocks, float *dst) {
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
						 *)(&iq3s_grid[qs[(2 * l) + 0] | ((qh[0] << (8 - (2 * l))) & 256)]);
				const uint8_t *grid2 =
					(const uint8_t
						 *)(&iq3s_grid[qs[(2 * l) + 1] | ((qh[0] << (7 - (2 * l))) & 256)]);

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
						 *)(&iq3s_grid[qs[(2 * l) + 0] | ((qh[1] << (8 - (2 * l))) & 256)]);
				const uint8_t *grid2 =
					(const uint8_t
						 *)(&iq3s_grid[qs[(2 * l) + 1] | ((qh[1] << (7 - (2 * l))) & 256)]);

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

static inline void iq3s_unpack_i8_generic(const iq3s_block *b, int8_t *a) {
	const uint8_t *qs	 = b->qs;
	const uint8_t *qh	 = b->qh;
	const uint8_t *signs = b->signs;
	int8_t		  *ap	 = a;

	for (int ib32 = 0; ib32 < 8; ib32 += 2) {
		for (int l = 0; l < 4; l++) {
			const uint8_t *grid1 = (const uint8_t *)(&iq3s_grid[qs[(ptrdiff_t)(2 * l)] |
																((qh[0] << (8 - (2 * l))) & 256)]);
			const uint8_t *grid2 =
				(const uint8_t *)(&iq3s_grid[qs[(2 * l) + 1] | ((qh[0] << (7 - (2 * l))) & 256)]);
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
			const uint8_t *grid1 = (const uint8_t *)(&iq3s_grid[qs[(ptrdiff_t)(2 * l)] |
																((qh[1] << (8 - (2 * l))) & 256)]);
			const uint8_t *grid2 =
				(const uint8_t *)(&iq3s_grid[qs[(2 * l) + 1] | ((qh[1] << (7 - (2 * l))) & 256)]);
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

#define MATMUL_Q8_F32(name, xq_t, qk, quantize_fn, qonly_fn)                                       \
	__attribute__((weak)) void matmul_##name##_f32(const void *w, const float *restrict x,         \
												   float *restrict y, int n, int k,                \
												   quant_scratch *qs) {                            \
		int blocks_per_row = k / (qk);                                                             \
		quant_scratch_ensure(qs, (size_t)blocks_per_row * sizeof(xq_t));                           \
		xq_t *xq = qs->q8_buf;                                                                     \
		quantize_fn(x, xq, k);                                                                     \
		qonly_fn(w, xq, 0, y, n, n, k, 1);                                                         \
	}

MATMUL_Q8_F32(iq3_s_q8_k, q8_k_block, 256, quantize_q8_k, matmul_iq3_s_q8_k_qonly_f32)

#define MATMUL_QONLY_DISPATCH(name, xq_t)                                                          \
	__attribute__((weak)) void matmul_##name##_qonly_f32(                                          \
		const void *w, const xq_t *restrict xq, size_t xq_row_stride_blocks, float *restrict y,    \
		int y_row_stride, int n, int k, int m) {                                                   \
		for (int t = 0; t < m; t++)                                                                \
			matmul_##name##_qonly_f32_row(w, xq + ((size_t)t * xq_row_stride_blocks),              \
										  y + ((size_t)t * y_row_stride), n, k);                   \
	}

MATMUL_QONLY_DISPATCH(iq3_s_q8_k, q8_k_block)

static const int8_t iq3s_re_decode[16] = {
	1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15,
};

static inline uint8_t iq3s_encode_val(uint8_t magnitude, int sign) {

	uint8_t idx = (magnitude - 1) / 2;
	if (sign)
		idx |= 8;
	return idx;
}

static uint16_t iq3s_enc4[512][16];

static void __attribute__((constructor)) iq3s_enc4_init(void) {
	for (int g = 0; g < 512; g++) {
		const uint8_t *mag = (const uint8_t *)&iq3s_grid[g];
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

void repack_iq3_s_rows(const void *src, void *dst, int row_begin, int row_end, int k) {
	const int	 blocks_per_row = k / 256;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(iq3_s_block);
	const size_t dst_stride		= (size_t)blocks_per_row * IQ3_S_RE_BLOCK_BYTES;

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	for (int r = row_begin; r < row_end; r++) {
		const iq3_s_block *srow = (const iq3_s_block *)(sp + (size_t)r * src_stride);
		uint8_t			  *drow = dp + (size_t)r * dst_stride;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *blk = drow + (size_t)bi * IQ3_S_RE_BLOCK_BYTES;
			iq3s_encode_block(&srow[bi], (uint16_t *)(blk + IQ3_S_RE_OFF_D),
							  blk + IQ3_S_RE_OFF_SCALES, blk + IQ3_S_RE_OFF_IDX);
		}
	}
}

void repack_iq3_s_to_iq3_s_re8_rows(const void *src, void *dst, int row_begin, int row_end, int k) {
	const int	 blocks_per_row = k / 256;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(iq3_s_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % IQ3_S_RE8_ROWS);
	for (int g = group_begin; g < row_end; g += IQ3_S_RE8_ROWS) {
		const iq3_s_block *srow[IQ3_S_RE8_ROWS];
		for (int r = 0; r < IQ3_S_RE8_ROWS; r++)
			srow[r] = (const iq3_s_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * IQ3_S_RE_BLOCK_BYTES;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * IQ3_S_RE8_GROUP_BYTES;

			uint16_t *dst_d		 = (uint16_t *)dblk;
			uint8_t	 *dst_scales = dblk + IQ3_S_RE8_ROWS * sizeof(uint16_t);
			uint8_t	 *dst_idx	 = dst_scales + IQ3_S_RE8_ROWS * 4;

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

void repack_iq3_s_to_iq3_s_re8(const void *src, void *dst, int n_rows, int k) {
	repack_iq3_s_to_iq3_s_re8_rows(src, dst, 0, n_rows, k);
}

static void matmul_iq3_s_re8_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
												float *restrict y, int n, int k) {
	const int	   blocks_per_row = k / 256;
	const uint8_t *wb			  = w;

	for (int g = 0; g < n; g += IQ3_S_RE8_ROWS) {
		const uint8_t *dgroup =
			wb + ((size_t)g / IQ3_S_RE8_ROWS) * blocks_per_row * IQ3_S_RE8_GROUP_BYTES;
		float sumf[IQ3_S_RE8_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk	   = dgroup + (size_t)bi * IQ3_S_RE8_GROUP_BYTES;
			const uint16_t *dst_d	   = (const uint16_t *)dblk;
			const uint8_t  *dst_scales = dblk + IQ3_S_RE8_ROWS * sizeof(uint16_t);
			const uint8_t  *dst_idx	   = dst_scales + IQ3_S_RE8_ROWS * 4;

			const q8_k_block *yb = &xq[bi];
			const int8_t	 *q8 = yb->qs;

			for (int r = 0; r < IQ3_S_RE8_ROWS; r++) {
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
						int8_t	val	 = iq3s_re_decode[code];
						group_sum += (int32_t)val * (int32_t)q8[gr * 32 + j];
					}
					total += sc * group_sum;
				}
				sumf[r] = fmaf(f16_to_f32(dst_d[r]) * yb->d, (float)total, sumf[r]);
			}
		}
		int rows_out = n - g;
		if (rows_out > IQ3_S_RE8_ROWS)
			rows_out = IQ3_S_RE8_ROWS;
		for (int r = 0; r < rows_out; r++)
			y[g + r] = sumf[r];
	}
}

MATMUL_QONLY_DISPATCH(iq3_s_re8_q8_k, q8_k_block)

MATMUL_Q8_F32(iq3_s_re8_q8_k, q8_k_block, 256, quantize_q8_k, matmul_iq3_s_re8_q8_k_qonly_f32)

void repack_iq3_s(const void *src, void *dst, int n_rows, int k) {
	repack_iq3_s_rows(src, dst, 0, n_rows, k);
}

void dequant_iq3_s_re_row(const void *repacked_buf, size_t n_blocks, float *dst) {
	const uint8_t *blk = repacked_buf;

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const uint8_t *b	  = blk + bi * IQ3_S_RE_BLOCK_BYTES;
		float		   d	  = f16_to_f32(*(const uint16_t *)(b + IQ3_S_RE_OFF_D));
		const uint8_t *scales = b + IQ3_S_RE_OFF_SCALES;
		const uint8_t *idx	  = b + IQ3_S_RE_OFF_IDX;
		float		  *y	  = dst + (bi * 256);

		for (int g = 0; g < 8; g++) {
			uint8_t		   sb	 = scales[g / 2];
			int			   sc	 = (g & 1) ? (1 + 2 * (sb >> 4)) : (1 + 2 * (sb & 0xf));
			const uint8_t *idx_g = idx + g * 16;
			for (int j = 0; j < 32; j++) {
				uint8_t code	= (j < 16) ? (idx_g[j] & 0xF) : (idx_g[j - 16] >> 4);
				y[(g * 32) + j] = d * (float)sc * (float)iq3s_re_decode[code];
			}
		}
	}
}

void dequant_q8_0_r8_row(const void *repacked_buf, size_t n_blocks, float *dst) {
	const uint8_t *blk		   = repacked_buf;
	const size_t   group_bytes = (size_t)Q8_0_R8_ROWS * sizeof(q8_0_block);

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const uint8_t  *b	   = blk + bi * group_bytes;
		const uint16_t *d8	   = (const uint16_t *)b;
		const int8_t   *qs_all = (const int8_t *)(b + Q8_0_R8_ROWS * sizeof(uint16_t));

		for (int r = 0; r < Q8_0_R8_ROWS; r++) {
			float d = f16_to_f32(d8[r]);
			for (int j = 0; j < 32; j++)
				dst[(r * 32) + j] = d * (float)qs_all[(size_t)r * 32 + j];
		}
	}
}

__attribute__((weak)) void matmul_iq3_s_re_q8_k_qonly_f32(const void *w,
														  const q8_k_block *restrict xq,
														  size_t xq_row_stride_blocks,
														  float *restrict y, int y_row_stride,
														  int n, int k, int m) {
	const int	   blocks_per_row = k / 256;
	const size_t   row_stride	  = (size_t)blocks_per_row * IQ3_S_RE_BLOCK_BYTES;
	const uint8_t *wb			  = w;

	for (int i = 0; i < n; i++) {
		const uint8_t *row = wb + (size_t)i * row_stride;

		for (int t = 0; t < m; t++) {
			const q8_k_block *restrict xb = xq + ((size_t)t * xq_row_stride_blocks);
			float sumf					  = 0.0f;

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk		  = row + (size_t)bi * IQ3_S_RE_BLOCK_BYTES;
				float		   d_w		  = f16_to_f32(*(const uint16_t *)(blk + IQ3_S_RE_OFF_D));
				const uint8_t *scales	  = blk + IQ3_S_RE_OFF_SCALES;
				const uint8_t *idx		  = blk + IQ3_S_RE_OFF_IDX;
				const int8_t *restrict q8 = xb[bi].qs;

				int32_t total = 0;
				for (int g = 0; g < 8; g++) {
					uint8_t		   sb		 = scales[g / 2];
					int			   sc		 = (g & 1) ? (1 + 2 * (sb >> 4)) : (1 + 2 * (sb & 0xf));
					const uint8_t *idx_g	 = idx + g * 16;
					int32_t		   group_sum = 0;
					for (int j = 0; j < 32; j++) {
						uint8_t code = (j < 16) ? (idx_g[j] & 0xF) : (idx_g[j - 16] >> 4);
						int8_t	val	 = iq3s_re_decode[code];
						group_sum += (int32_t)val * (int32_t)q8[g * 32 + j];
					}
					total += sc * group_sum;
				}
				sumf += d_w * xb[bi].d * (float)total;
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

MATMUL_Q8_F32(iq3_s_re_q8_k, q8_k_block, 256, quantize_q8_k, matmul_iq3_s_re_q8_k_qonly_f32)

__attribute__((weak)) void dequant_iq4_nl_row(const void *blocks, size_t n_blocks, float *dst) {
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

__attribute__((weak)) void repack_iq4_nl_to_q8_0_rows(const void *src, void *dst, int row_begin,
													  int row_end, int k) {
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

void repack_iq4_nl_to_q8_0(const void *src, void *dst, int n_rows, int k) {
	repack_iq4_nl_to_q8_0_rows(src, dst, 0, n_rows, k);
}

void repack_iq4_nl_to_iq4_nl_r8_rows(const void *src, void *dst, int row_begin, int row_end,
									 int k) {
	const int	 blocks_per_row = k / 32;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(iq4_nl_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % IQ4_NL_R8_ROWS);
	for (int g = group_begin; g < row_end; g += IQ4_NL_R8_ROWS) {
		const iq4_nl_block *srow[IQ4_NL_R8_ROWS];
		for (int r = 0; r < IQ4_NL_R8_ROWS; r++)
			srow[r] = (const iq4_nl_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * sizeof(iq4_nl_block);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * IQ4_NL_R8_ROWS * sizeof(iq4_nl_block);

			uint16_t *dst_d = (uint16_t *)dblk;
			for (int r = 0; r < IQ4_NL_R8_ROWS; r++)
				dst_d[r] = srow[r][bi].d;

			uint8_t *dst_qs = dblk + IQ4_NL_R8_ROWS * sizeof(uint16_t);
			for (int r = 0; r < IQ4_NL_R8_ROWS; r++)
				memcpy(dst_qs + (size_t)r * 16, srow[r][bi].qs, 16);
		}
	}
}

void repack_iq4_nl_to_iq4_nl_r8(const void *src, void *dst, int n_rows, int k) {
	repack_iq4_nl_to_iq4_nl_r8_rows(src, dst, 0, n_rows, k);
}

static void matmul_iq4_nl_r8_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
											  float *restrict y, int n, int k) {
	const int	   blocks_per_row = k / 32;
	const size_t   group_bytes	  = (size_t)IQ4_NL_R8_ROWS * sizeof(iq4_nl_block);
	const uint8_t *wb			  = w;

	for (int g = 0; g < n; g += IQ4_NL_R8_ROWS) {
		const uint8_t *dgroup = wb + (size_t)g * blocks_per_row * sizeof(iq4_nl_block);
		float		   sumf[IQ4_NL_R8_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk	   = dgroup + (size_t)bi * group_bytes;
			const uint16_t *d_w		   = (const uint16_t *)dblk;
			const uint8_t  *qs_w	   = dblk + IQ4_NL_R8_ROWS * sizeof(uint16_t);
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);

			for (int r = 0; r < IQ4_NL_R8_ROWS; r++) {
				const uint8_t *row_qs = qs_w + (size_t)r * 16;
				int32_t		   sumi	  = 0;
				for (int j = 0; j < 16; j++) {
					sumi += (int32_t)kvalues_iq4nl[row_qs[j] & 0xF] * (int32_t)xq8[j];
					sumi += (int32_t)kvalues_iq4nl[row_qs[j] >> 4] * (int32_t)xq8[j + 16];
				}
				sumf[r] = fmaf(f16_to_f32(d_w[r]) * d_xq, (float)sumi, sumf[r]);
			}
		}
		int rows_out = n - g;
		if (rows_out > IQ4_NL_R8_ROWS)
			rows_out = IQ4_NL_R8_ROWS;
		for (int r = 0; r < rows_out; r++)
			y[g + r] = sumf[r];
	}
}

MATMUL_QONLY_DISPATCH(iq4_nl_r8_q8, q8_0_block)

MATMUL_Q8_F32(iq4_nl_r8_q8, q8_0_block, 32, quantize_q8_0, matmul_iq4_nl_r8_q8_qonly_f32)

void repack_q8_0_to_q8_0_r8_rows(const void *src, void *dst, int row_begin, int row_end, int k) {
	const int	 blocks_per_row = k / 32;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(q8_0_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % Q8_0_R8_ROWS);
	for (int g = group_begin; g < row_end; g += Q8_0_R8_ROWS) {
		const q8_0_block *srow[Q8_0_R8_ROWS];
		for (int r = 0; r < Q8_0_R8_ROWS; r++)
			srow[r] = (const q8_0_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * sizeof(q8_0_block);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * Q8_0_R8_ROWS * sizeof(q8_0_block);

			uint16_t *dst_d = (uint16_t *)dblk;
			for (int r = 0; r < Q8_0_R8_ROWS; r++)
				dst_d[r] = srow[r][bi].d;

			int8_t *dst_qs = (int8_t *)(dblk + Q8_0_R8_ROWS * sizeof(uint16_t));
			for (int r = 0; r < Q8_0_R8_ROWS; r++)
				memcpy(dst_qs + (size_t)r * 32, srow[r][bi].qs, 32);
		}
	}
}

void repack_q8_0_to_q8_0_r8(const void *src, void *dst, int n_rows, int k) {
	repack_q8_0_to_q8_0_r8_rows(src, dst, 0, n_rows, k);
}

static void matmul_q8_0_r8_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
											float *restrict y, int n, int k) {
	const int	   blocks_per_row = k / 32;
	const size_t   group_bytes	  = (size_t)Q8_0_R8_ROWS * sizeof(q8_0_block);
	const uint8_t *wb			  = w;

	for (int g = 0; g < n; g += Q8_0_R8_ROWS) {
		const uint8_t *dgroup			  = wb + (size_t)g * blocks_per_row * sizeof(q8_0_block);
		float		   sumf[Q8_0_R8_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk	   = dgroup + (size_t)bi * group_bytes;
			const uint16_t *d_w		   = (const uint16_t *)dblk;
			const int8_t   *qs_w	   = (const int8_t *)(dblk + Q8_0_R8_ROWS * sizeof(uint16_t));
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);

			for (int r = 0; r < Q8_0_R8_ROWS; r++) {
				const int8_t *row_qs = qs_w + (size_t)r * 32;
				int32_t		  sumi	 = 0;
				for (int j = 0; j < 32; j++)
					sumi += (int32_t)row_qs[j] * (int32_t)xq8[j];
				sumf[r] = fmaf(f16_to_f32(d_w[r]) * d_xq, (float)sumi, sumf[r]);
			}
		}
		int rows_out = n - g;
		if (rows_out > Q8_0_R8_ROWS)
			rows_out = Q8_0_R8_ROWS;
		for (int r = 0; r < rows_out; r++)
			y[g + r] = sumf[r];
	}
}

MATMUL_QONLY_DISPATCH(q8_0_r8_q8, q8_0_block)

MATMUL_Q8_F32(q8_0_r8_q8, q8_0_block, 32, quantize_q8_0, matmul_q8_0_r8_q8_qonly_f32)

void repack_q4_0_to_q4_0_r8_rows(const void *src, void *dst, int row_begin, int row_end, int k) {
	const int	 blocks_per_row = k / 32;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(q4_0_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	int group_begin = row_begin - (row_begin % Q4_0_R8_ROWS);
	for (int g = group_begin; g < row_end; g += Q4_0_R8_ROWS) {
		const q4_0_block *srow[Q4_0_R8_ROWS];
		for (int r = 0; r < Q4_0_R8_ROWS; r++)
			srow[r] = (const q4_0_block *)(sp + (size_t)(g + r) * src_stride);

		uint8_t *dgroup = dp + (size_t)g * blocks_per_row * sizeof(q4_0_block);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			uint8_t *dblk = dgroup + (size_t)bi * Q4_0_R8_ROWS * sizeof(q4_0_block);

			uint16_t *dst_d = (uint16_t *)dblk;
			for (int r = 0; r < Q4_0_R8_ROWS; r++)
				dst_d[r] = srow[r][bi].d;

			uint8_t *dst_qs = dblk + Q4_0_R8_ROWS * sizeof(uint16_t);
			for (int r = 0; r < Q4_0_R8_ROWS; r++)
				memcpy(dst_qs + (size_t)r * 16, srow[r][bi].qs, 16);
		}
	}
}

void repack_q4_0_to_q4_0_r8(const void *src, void *dst, int n_rows, int k) {
	repack_q4_0_to_q4_0_r8_rows(src, dst, 0, n_rows, k);
}

static void matmul_q4_0_r8_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
											float *restrict y, int n, int k) {
	const int	   blocks_per_row = k / 32;
	const size_t   group_bytes	  = (size_t)Q4_0_R8_ROWS * sizeof(q4_0_block);
	const uint8_t *wb			  = w;

	for (int g = 0; g < n; g += Q4_0_R8_ROWS) {
		const uint8_t *dgroup			  = wb + (size_t)g * blocks_per_row * sizeof(q4_0_block);
		float		   sumf[Q4_0_R8_ROWS] = {0};

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t  *dblk	   = dgroup + (size_t)bi * group_bytes;
			const uint16_t *d_w		   = (const uint16_t *)dblk;
			const uint8_t  *qs_w	   = dblk + Q4_0_R8_ROWS * sizeof(uint16_t);
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32(xq[bi].d);

			for (int r = 0; r < Q4_0_R8_ROWS; r++) {
				const uint8_t *row_qs = qs_w + (size_t)r * 16;
				int32_t		   sumi	  = 0;
				for (int j = 0; j < 16; j++) {
					sumi += ((int32_t)(row_qs[j] & 0xF) - 8) * (int32_t)xq8[j];
					sumi += ((int32_t)(row_qs[j] >> 4) - 8) * (int32_t)xq8[j + 16];
				}
				sumf[r] = fmaf(f16_to_f32(d_w[r]) * d_xq, (float)sumi, sumf[r]);
			}
		}
		int rows_out = n - g;
		if (rows_out > Q4_0_R8_ROWS)
			rows_out = Q4_0_R8_ROWS;
		for (int r = 0; r < rows_out; r++)
			y[g + r] = sumf[r];
	}
}

MATMUL_QONLY_DISPATCH(q4_0_r8_q8, q8_0_block)

MATMUL_Q8_F32(q4_0_r8_q8, q8_0_block, 32, quantize_q8_0, matmul_q4_0_r8_q8_qonly_f32)

__attribute__((weak)) void dequant_f16_row(const void *src, int n, float *dst) {
	const uint16_t *s = src;
	for (int i = 0; i < n; i++)
		dst[i] = f16_to_f32(s[i]);
}

__attribute__((weak)) float bf16_to_f32(uint16_t h) {
	union {
		uint32_t u;
		float	 f;
	} v;
	v.u = ((uint32_t)h) << 16;
	return v.f;
}

__attribute__((weak)) void dequant_bf16_row(const void *src, int n, float *dst) {
	const uint16_t *s = src;
	for (int i = 0; i < n; i++)
		dst[i] = bf16_to_f32(s[i]);
}

void dequant_row_dispatch(uint32_t type, const void *src, int n_elems, float *dst) {
	switch (type) {
	case GGML_TYPE_F32:
		memcpy(dst, src, (size_t)n_elems * sizeof(float));
		break;
	case GGML_TYPE_F16:
		dequant_f16_row(src, n_elems, dst);
		break;
	case GGML_TYPE_BF16:
		dequant_bf16_row(src, n_elems, dst);
		break;
	case GGML_TYPE_Q4_0:
		dequant_q4_0_row(src, n_elems / 32, dst);
		break;
	case GGML_TYPE_Q4_1:
		dequant_q4_1_row(src, n_elems / 32, dst);
		break;
	case GGML_TYPE_Q5_0:
		dequant_q5_0_row(src, n_elems / 32, dst);
		break;
	case GGML_TYPE_Q5_1:
		dequant_q5_1_row(src, n_elems / 32, dst);
		break;
	case GGML_TYPE_Q8_0:
		dequant_q8_0_row(src, n_elems / 32, dst);
		break;
	case GGML_TYPE_Q8_0_R8:
		dequant_q8_0_r8_row(src, n_elems / 32, dst);
		break;
	case GGML_TYPE_Q3_K:
		dequant_q3_k_row(src, n_elems / 256, dst);
		break;
	case GGML_TYPE_Q4_K:
		dequant_q4_k_row(src, n_elems / 256, dst);
		break;
	case GGML_TYPE_Q5_K:
		dequant_q5_k_row(src, n_elems / 256, dst);
		break;
	case GGML_TYPE_Q6_K:
		dequant_q6_k_row(src, n_elems / 256, dst);
		break;
	case GGML_TYPE_IQ4_NL:
		dequant_iq4_nl_row(src, n_elems / 32, dst);
		break;
	case GGML_TYPE_IQ3_S:
		dequant_iq3_s_row(src, n_elems / 256, dst);
		break;
	case GGML_TYPE_IQ3_S_RE:
		dequant_iq3_s_re_row(src, (size_t)n_elems / 256, dst);
		break;
	default:
		memset(dst, 0, (size_t)n_elems * sizeof(float));
		break;
	}
}

__attribute__((weak)) void matmul_generic_f32(const void *w, uint32_t w_type, const float *x,
											  float *y, int n, int k) {
	switch (w_type) {
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_0_R8:
	case GGML_TYPE_Q4_0_R8:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ3_S_RE: {
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
		case GGML_TYPE_Q8_0_R8:
			matmul_q8_0_r8_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q4_0_R8:
			matmul_q4_0_r8_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_IQ4_NL_R8:
			matmul_iq4_nl_r8_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q4_1:
			matmul_q4_1_q8_f32(w, x, y, n, k, &qs);
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
		case GGML_TYPE_IQ3_S_RE:
			matmul_iq3_s_re_q8_k_f32(w, x, y, n, k, &qs);
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

__attribute__((weak)) void quantize_q8_0(const float *x, q8_0_block *dst, int n) {
	int nb = n / 32;
	for (int i = 0; i < nb; i++) {
		float amax = 0;
		for (int j = 0; j < 32; j++) {
			float ax = fabsf(x[j]);
			if (ax > amax)
				amax = ax;
		}
		float d	 = amax / 127.0f;
		float id = d > 0 ? 1.0f / d : 0.0f;
		dst[i].d = f32_to_f16(d);

		for (int j = 0; j < 32; j++) {
			int v = (int)roundf(x[j] * id);
			if (v > 127)
				v = 127;
			if (v < -127)
				v = -127;
			dst[i].qs[j] = (int8_t)v;
		}
		x += 32;
	}
}

__attribute__((weak)) void quantize_q8_1(const float *x, void *dst, int n) {
	int			nb = n / 32;
	q8_1_block *y  = dst;
	for (int i = 0; i < nb; i++) {
		float amax = 0;
		for (int j = 0; j < 32; j++) {
			float ax = fabsf(x[j]);
			if (ax > amax)
				amax = ax;
		}
		float d		= amax / 127.0f;
		float id	= d > 0 ? 1.0f / d : 0.0f;
		y[i].d		= f32_to_f16(d);
		int32_t sum = 0;
		for (int j = 0; j < 32; j++) {
			int32_t q = (int32_t)roundf(x[j] * id);
			if (q > 127)
				q = 127;
			if (q < -127)
				q = -127;
			y[i].qs[j] = (int8_t)q;
			sum += q;
		}
		y[i].s = f32_to_f16(d * (float)sum);
		x += 32;
	}
}

__attribute__((weak)) void quantize_q8_k(const float *x, q8_k_block *y, int n) {
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

void quant_scratch_ensure(quant_scratch *qs, size_t need) {
	if (qs->q8_buf_elems < need) {
		free(qs->q8_buf);
		qs->q8_buf		 = xmalloc(need);
		qs->q8_buf_elems = need;
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

MATMUL_Q8_F32(q4_q8, q8_0_block, 32, quantize_q8_0, matmul_q4_q8_qonly_f32)

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

MATMUL_QONLY_DISPATCH(q4_q8, q8_0_block)

MATMUL_Q8_F32(q8_0_q8, q8_0_block, 32, quantize_q8_0, matmul_q8_0_q8_qonly_f32)

MATMUL_QONLY_DISPATCH(q8_0_q8, q8_0_block)

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

MATMUL_Q8_F32(iq4_nl_q8, q8_0_block, 32, quantize_q8_0, matmul_iq4_nl_q8_qonly_f32)

MATMUL_QONLY_DISPATCH(iq4_nl_q8, q8_0_block)

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

MATMUL_Q8_F32(q4_1_q8, q8_1_block, 32, quantize_q8_1, matmul_q4_1_q8_qonly_f32)

MATMUL_QONLY_DISPATCH(q4_1_q8, q8_1_block)

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

MATMUL_QONLY_DISPATCH(q5_0_q8, q8_0_block)

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

MATMUL_QONLY_DISPATCH(q5_1_q8, q8_1_block)

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

MATMUL_Q8_F32(q6_k_q8, q8_k_block, 256, quantize_q8_k, matmul_q6_k_q8_qonly_f32)

MATMUL_QONLY_DISPATCH(q6_k_q8, q8_k_block)

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

MATMUL_Q8_F32(q4_k_q8_k, q8_k_block, 256, quantize_q8_k, matmul_q4_k_q8_k_qonly_f32)

MATMUL_QONLY_DISPATCH(q4_k_q8_k, q8_k_block)

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

MATMUL_Q8_F32(q5_k_q8_k, q8_k_block, 256, quantize_q8_k, matmul_q5_k_q8_k_qonly_f32)
#undef MATMUL_Q8_F32

MATMUL_QONLY_DISPATCH(q5_k_q8_k, q8_k_block)
#undef MATMUL_QONLY_DISPATCH

__attribute__((weak)) void matmul_f32_f32(const float *restrict w, const float *restrict x,
										  float *restrict y, int n, int k) {
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

__attribute__((weak)) void matmul_bf16_f32(const void *restrict w, const float *restrict x,
										   float *restrict y, int n, int k) {
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

__attribute__((weak)) void matmul_bf16_f32_batch(const void *restrict w, const float *restrict x,
												 float *restrict y, int n, int k, int m,
												 int x_row_stride, int y_row_stride) {
	for (int row = 0; row < m; row++)
		matmul_bf16_f32(w, x + (size_t)row * x_row_stride, y + (size_t)row * y_row_stride, n, k);
}

__attribute__((weak)) void matmul_f32_f32_batch(const float *restrict w, const float *restrict x,
												float *restrict y, int n, int k, int m,
												int x_row_stride, int y_row_stride) {
	for (int row = 0; row < m; row++)
		matmul_f32_f32(w, x + (size_t)row * x_row_stride, y + (size_t)row * y_row_stride, n, k);
}

__attribute__((weak)) void matmul_f16_f32(const void *restrict w, const float *restrict x,
										  float *restrict y, int n, int k) {
	const uint16_t *wb = w;
	for (int i = 0; i < n; i++) {
		const uint16_t *restrict wr = wb + ((size_t)i * k);
		float acc					= 0.0f;
		for (int j = 0; j < k; j++)
			acc += f16_to_f32(wr[j]) * x[j];
		y[i] = acc;
	}
}

__attribute__((weak)) void matmul_f16_f32_batch(const void *restrict w, const float *restrict x,
												float *restrict y, int n, int k, int m,
												int x_row_stride, int y_row_stride) {
	for (int row = 0; row < m; row++)
		matmul_f16_f32(w, x + (size_t)row * x_row_stride, y + (size_t)row * y_row_stride, n, k);
}

__attribute__((weak)) float dot_f32(const float *restrict a, const float *restrict b, int n) {
	if (n <= 0)
		return 0.0f;
	float acc = 0.0f;
	for (int i = 0; i < n; i++)
		acc += a[i] * b[i];
	return acc;
}

static inline float rmsnorm_sum_sq(const float *x, int n) {
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

__attribute__((weak)) void rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
	float ss	= rmsnorm_sum_sq(x, n);
	float scale = 1.0f / sqrtf((ss / (float)n) + eps);
	for (int i = 0; i < n; i++)
		y[i] = x[i] * scale * w[i];
}

__attribute__((weak)) void rmsnorm_noweight(const float *x, float *y, int n, float eps) {
	float ss	= rmsnorm_sum_sq(x, n);
	float scale = 1.0f / sqrtf((ss / (float)n) + eps);
	for (int i = 0; i < n; i++)
		y[i] = x[i] * scale;
}

__attribute__((weak)) void rmsnorm_per_head(const float *x, const float *w, float *y, int n_heads,
											int head_dim, float eps) {
	for (int h = 0; h < n_heads; h++) {
		const float *xh	   = x + ((size_t)h * head_dim);
		float		*yh	   = y + ((size_t)h * head_dim);
		float		 ss	   = rmsnorm_sum_sq(xh, head_dim);
		float		 scale = 1.0f / sqrtf((ss / (float)head_dim) + eps);
		for (int j = 0; j < head_dim; j++)
			yh[j] = xh[j] * scale * w[j];
	}
}

__attribute__((weak)) void softmax_masked(float *restrict scores, int n_valid) {
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

__attribute__((weak)) void moe_activate_silu(float *restrict act, const float *restrict gate,
											 const float *restrict up, int n, float gs, float us) {
	for (int i = 0; i < n; i++)
		act[i] = silu(gate[i] * gs) * (up[i] * us);
}

__attribute__((weak)) void moe_activate_gelu(float *restrict act, const float *restrict gate,
											 const float *restrict up, int n, float gs, float us) {
	for (int i = 0; i < n; i++)
		act[i] = gelu_tanh(gate[i] * gs) * (up[i] * us);
}