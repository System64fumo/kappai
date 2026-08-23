#include "backend/cpu/scalar/quants.h"
#include "common.h"
#include "threadpool.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#define MR 8
_Static_assert(MR % 4 == 0, "matmul_iq4_nl_q8_qonly_f32 assumes MR is a multiple of 4");

typedef struct {
	uint16_t d;
	uint8_t	 qs[64];
	uint8_t	 qh[8];
	uint8_t	 signs[32];
	uint8_t	 scales[4];
} iq3s_block;

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

#if defined(__ARM_FEATURE_MATMUL_INT8)
#define I8MM_NR 4
#endif

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

static inline int32x4_t q4_dot_x4_sumi4(int8x16_t lo, int8x16_t hi, const int8x16_t xq_lo[4],
										const int8x16_t xq_hi[4]) {
#if defined(__ARM_FEATURE_DOTPROD)
	int32x4_t acc0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, xq_lo[0]), hi, xq_hi[0]);
	int32x4_t acc1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, xq_lo[1]), hi, xq_hi[1]);
	int32x4_t acc2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, xq_lo[2]), hi, xq_hi[2]);
	int32x4_t acc3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, xq_lo[3]), hi, xq_hi[3]);

	const int32x4_t sum01 = vpaddq_s32(acc0, acc1);
	const int32x4_t sum23 = vpaddq_s32(acc2, acc3);
	return vpaddq_s32(sum01, sum23);
#else
	int32x4_t accs[4];
	for (int c = 0; c < 4; c++) {
		int16x8_t p0 = vmull_s8(vget_low_s8(lo), vget_low_s8(xq_lo[c]));
		int16x8_t p1 = vmull_s8(vget_high_s8(lo), vget_high_s8(xq_lo[c]));
		int16x8_t p2 = vmull_s8(vget_low_s8(hi), vget_low_s8(xq_hi[c]));
		int16x8_t p3 = vmull_s8(vget_high_s8(hi), vget_high_s8(xq_hi[c]));
		int32x4_t s	 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
		s			 = vaddq_s32(s, vpaddlq_s16(p2));
		s			 = vaddq_s32(s, vpaddlq_s16(p3));
		accs[c]		 = s;
	}
	const int32x4_t sum01 = vpaddq_s32(accs[0], accs[1]);
	const int32x4_t sum23 = vpaddq_s32(accs[2], accs[3]);
	return vpaddq_s32(sum01, sum23);
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

static inline void iq3s_pair_dot(int8x16_t d00, int8x16_t d01, int8x16_t d10, int8x16_t d11,
								 const int8_t *restrict qg0, const int8_t *restrict qg1,
								 int32x4_t *a0_out, int32x4_t *a1_out) {
#if defined(__ARM_FEATURE_DOTPROD)
	int32x4_t a0 =
		vdotq_s32(vdotq_s32(vdupq_n_s32(0), d00, vld1q_s8(qg0)), d01, vld1q_s8(qg0 + 16));
	int32x4_t a1 =
		vdotq_s32(vdotq_s32(vdupq_n_s32(0), d10, vld1q_s8(qg1)), d11, vld1q_s8(qg1 + 16));
#else
	int16x8_t p00 = vmull_s8(vget_low_s8(d00), vget_low_s8(vld1q_s8(qg0)));
	int16x8_t p01 = vmull_s8(vget_high_s8(d00), vget_high_s8(vld1q_s8(qg0)));
	int16x8_t p02 = vmull_s8(vget_low_s8(d01), vget_low_s8(vld1q_s8(qg0 + 16)));
	int16x8_t p03 = vmull_s8(vget_high_s8(d01), vget_high_s8(vld1q_s8(qg0 + 16)));
	int32x4_t a0  = vpaddlq_s16(p00);
	a0			  = vpadalq_s16(a0, p01);
	a0			  = vpadalq_s16(a0, p02);
	a0			  = vpadalq_s16(a0, p03);
	int16x8_t p10 = vmull_s8(vget_low_s8(d10), vget_low_s8(vld1q_s8(qg1)));
	int16x8_t p11 = vmull_s8(vget_high_s8(d10), vget_high_s8(vld1q_s8(qg1)));
	int16x8_t p12 = vmull_s8(vget_low_s8(d11), vget_low_s8(vld1q_s8(qg1 + 16)));
	int16x8_t p13 = vmull_s8(vget_high_s8(d11), vget_high_s8(vld1q_s8(qg1 + 16)));
	int32x4_t a1  = vpaddlq_s16(p10);
	a1			  = vpadalq_s16(a1, p11);
	a1			  = vpadalq_s16(a1, p12);
	a1			  = vpadalq_s16(a1, p13);
#endif
	*a0_out = a0;
	*a1_out = a1;
}

static inline void store_acc_row_mr_nr(const float32x4_t acc_row[MR], float *restrict y,
									   int y_row_stride, size_t i, size_t t, int nr) {
	for (int r = 0; r < MR; r++) {
		float tmp[4];
		vst1q_f32(tmp, acc_row[r]);
		for (int c = 0; c < nr; c++)
			y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
	}
}

static inline void store_acc8(float32x4_t acc0, float32x4_t acc1, float *restrict y, int i) {
	float tmp0[4], tmp1[4];
	vst1q_f32(tmp0, acc0);
	vst1q_f32(tmp1, acc1);
	for (int r = 0; r < 4; r++) {
		y[i + r]	 = tmp0[r];
		y[i + 4 + r] = tmp1[r];
	}
}

static inline int32_t q8_0_block_sum(const int8_t *restrict p) {
	const int8x16_t a = vld1q_s8(p);
	const int8x16_t b = vld1q_s8(p + 16);
	int16x8_t		l = vmovl_s8(vget_low_s8(a));
	int16x8_t		h = vmovl_s8(vget_high_s8(a));
	int16x8_t		m = vmovl_s8(vget_low_s8(b));
	int16x8_t		n = vmovl_s8(vget_high_s8(b));
	int32x4_t		s = vpaddlq_s16(l);
	s				  = vpadalq_s16(s, h);
	s				  = vpadalq_s16(s, m);
	s				  = vpadalq_s16(s, n);
	return vaddvq_s32(s);
}

#if defined(__ARM_FEATURE_SVE)

static inline void q5_0_unpack(const q5_0_block *b, int8x16_t *lo, int8x16_t *hi);

static inline void q5_1_unpack(const q5_1_block *b, int8x16_t *lo, int8x16_t *hi);

#endif

#if defined(__ARM_FEATURE_SVE)

static void matmul_q4_q8_qonly_f32_sve(const void *w, const q8_0_block *restrict xq,
									   size_t	   xq_row_stride_blocks, float *restrict y,
									   int y_row_stride, int n, int k, int m);

static void matmul_q8_0_q8_qonly_f32_sve(const void *w, const q8_0_block *restrict xq,
										 size_t		 xq_row_stride_blocks, float *restrict y,
										 int y_row_stride, int n, int k, int m);

#endif

#if defined(__ARM_FEATURE_SVE)

static int32_t sve_sum_i8(const int8_t *restrict p, int n) {
	const uint64_t vbytes = svcntb();
	int32_t		   sum	  = 0;
	for (int off = 0; off < n; off += (int)vbytes) {
		svbool_t  pg = svwhilelt_b8_u64((uint64_t)off, (uint64_t)n);
		svint8_t  xv = svld1_s8(pg, p + off);
		svint16_t lo = svunpklo_s16(xv);
		svint16_t hi = svunpkhi_s16(xv);
		sum += (int32_t)svaddv_s16(svptrue_b16(), lo);
		sum += (int32_t)svaddv_s16(svptrue_b16(), hi);
	}
	return sum;
}

static inline int32_t sve_q8_0_block_dot(const q8_0_block *restrict w,
										 const q8_0_block *restrict x) {
	const uint64_t vbytes = svcntb();
	svint32_t	   part	  = svdup_n_s32(0);
	for (int off = 0; off < 32; off += (int)vbytes) {
		svbool_t pg = svwhilelt_b8_u64((uint64_t)off, 32);
		part		= svdot_s32(part, svld1_s8(pg, w->qs + off), svld1_s8(pg, x->qs + off));
	}
	return svaddv_s32(svptrue_b32(), part);
}

static inline int32_t sve_q4_q8_block_dot_raw(const q4_0_block *restrict w,
											  const q8_0_block *restrict x) {
	const uint64_t vbytes = svcntb();
	svint32_t	   part	  = svdup_n_s32(0);

	for (int off = 0; off < 16; off += (int)vbytes) {
		svbool_t  pg   = svwhilelt_b8_u64((uint64_t)off, 16);
		svuint8_t qb   = svld1_u8(pg, w->qs + off);
		svuint8_t lo   = svand_u8_z(pg, qb, svdup_n_u8(0x0F));
		svuint8_t hi   = svlsr_n_u8_z(pg, qb, 4);
		svbool_t  pglo = svwhilelt_b8_u64((uint64_t)off, 16);
		svbool_t  pghi = svwhilelt_b8_u64((uint64_t)(16 + off), 32);
		svint8_t  xlo  = svld1_s8(pglo, x->qs + off);
		svint8_t  xhi  = svld1_s8(pghi, x->qs + 16 + off);
		part		   = svusdot_s32(part, lo, xlo);
		part		   = svusdot_s32(part, hi, xhi);
	}
	return svaddv_s32(svptrue_b32(), part);
}

static inline int32_t sve_q4_q8_block_dot(const q4_0_block *restrict w,
										  const q8_0_block *restrict x) {
	return sve_q4_q8_block_dot_raw(w, x) - 8 * sve_sum_i8(x->qs, 32);
}

#endif

void dequant_f16_row(const void *src, int n, float *dst) {
	const uint16_t *s = src;
	int				i = 0;
	for (; i + 8 <= n; i += 8) {
		const float16_t *sp = (const float16_t *)(s + i);
		float32x4_t		 lo = vcvt_f32_f16(vld1_f16(sp));
		float32x4_t		 hi = vcvt_f32_f16(vld1_f16(sp + 4));
		vst1q_f32(dst + i, lo);
		vst1q_f32(dst + i + 4, hi);
	}
	for (; i < n; i++)
		dst[i] = f16_to_f32_fast(s[i]);
}

void dequant_bf16_row(const void *src, int n, float *dst) {
	const uint16_t *s = src;
	int				i = 0;
	for (; i + 8 <= n; i += 8) {
		uint16x8_t v  = vld1q_u16(s + i);
		uint32x4_t lo = vshll_n_u16(vget_low_u16(v), 16);
		uint32x4_t hi = vshll_n_u16(vget_high_u16(v), 16);
		vst1q_f32(dst + i, vreinterpretq_f32_u32(lo));
		vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(hi));
	}
	for (; i < n; i++)
		dst[i] = bf16_to_f32(s[i]);
}

static void matmul_q5_0_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
										 float *restrict y, int n, int k);
static void matmul_q5_1_q8_qonly_f32_row(const void *w, const q8_1_block *restrict xq,
										 float *restrict y, int n, int k);

void matmul_q5_0_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						quant_scratch *qs) {
	quant_scratch_ensure(qs, (size_t)(k / 32) * sizeof(q8_0_block));
	quantize_q8_0(x, (q8_0_block *)qs->q8_buf, k);
	matmul_q5_0_q8_qonly_f32_row(w, (const q8_0_block *)qs->q8_buf, y, n, k);
}

static void matmul_q5_1_q8_qonly_f32_row(const void *w, const q8_1_block *restrict xq,
										 float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_1_block);
	const q5_1_block *Wb			 = w;

	for (int i = 0; i < n; i++) {
		const q5_1_block *row =
			(const q5_1_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			int8x16_t lo, hi;
			q5_1_unpack(&row[bi], &lo, &hi);
			const int8x16_t xlo	 = vld1q_s8(xq[bi].qs);
			const int8x16_t xhi	 = vld1q_s8(xq[bi].qs + 16);
			const int32_t	sumi = vaddvq_s32(q5_dot(lo, hi, xlo, xhi));
			const float		d	 = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xq[bi].d);
			const float		m	 = f16_to_f32_fast(row[bi].m) * f16_to_f32_fast(xq[bi].s);
			sumf += (d * (float)sumi) + m;
		}
		y[i] = sumf;
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_q5_1_q8_qonly_f32_i8mm(const void *w, const q8_1_block *restrict xq,
										  size_t	  xq_row_stride_blocks, float *restrict y,
										  int y_row_stride, int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_1_block);
	const q5_1_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / I8MM_NR) > 0 ? blocks_per_row : 0;

		static _Thread_local int8x16_t(*lo_cache)[MR] = NULL;
		static _Thread_local int8x16_t(*hi_cache)[MR] = NULL;
		static _Thread_local float (*d_w_cache)[MR]	  = NULL;
		static _Thread_local float (*m_w_cache)[MR]	  = NULL;
		static _Thread_local int cache_cap			  = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				lo_cache  = realloc(lo_cache, sizeof(*lo_cache) * n_bi_tiles);
				hi_cache  = realloc(hi_cache, sizeof(*hi_cache) * n_bi_tiles);
				d_w_cache = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				m_w_cache = realloc(m_w_cache, sizeof(*m_w_cache) * n_bi_tiles);
				cache_cap = n_bi_tiles;
				tlocal_register((void **)&lo_cache);
				tlocal_register((void **)&hi_cache);
				tlocal_register((void **)&d_w_cache);
				tlocal_register((void **)&m_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q5_1_block)), 0,
										   1);
				}
				for (int r = 0; r < MR; r++) {
					const q5_1_block *b =
						(const q5_1_block *)(row_base[r] + ((size_t)bi * sizeof(q5_1_block)));
					q5_1_unpack(b, &lo_cache[bi][r], &hi_cache[bi][r]);
					d_w_cache[bi][r] = f16_to_f32_fast(b->d);
					m_w_cache[bi][r] = f16_to_f32_fast(b->m);
				}
			}
		}

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			float32x4_t oacc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++) {
					facc[p][c] = vdupq_n_f32(0.0f);
					oacc[p][c] = vdupq_n_f32(0.0f);
				}

			const q8_1_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				int8x16_t xq_lo[I8MM_NR];
				int8x16_t xq_hi[I8MM_NR];
				float	  xd[I8MM_NR];
				float	  xs[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xs[c]	 = f16_to_f32_fast(xrow[c][bi].s);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}

				int8x16_t bvec[I8MM_NR / 2][4];
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					const int ca = 2 * cp;
					const int cb = 2 * cp + 1;
					bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb]));
					bvec[cp][1]	 = vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));
					bvec[cp][2]	 = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));
					bvec[cp][3]	 = vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));
				}

				for (int p = 0; p < MR / 2; p++) {
					const int8x16_t l0		= lo_cache[bi][2 * p];
					const int8x16_t l1		= lo_cache[bi][2 * p + 1];
					const int8x16_t h0		= hi_cache[bi][2 * p];
					const int8x16_t h1		= hi_cache[bi][2 * p + 1];
					const int8x16_t avec[4] = {
						vcombine_s8(vget_low_s8(l0), vget_low_s8(l1)),
						vcombine_s8(vget_high_s8(l0), vget_high_s8(l1)),
						vcombine_s8(vget_low_s8(h0), vget_high_s8(h1)),
						vcombine_s8(vget_high_s8(h0), vget_high_s8(h1)),
					};

					const float32x4_t srow = vcombine_f32(vdup_n_f32(d_w_cache[bi][2 * p]),
														  vdup_n_f32(d_w_cache[bi][2 * p + 1]));
					const float32x4_t smin = vcombine_f32(vdup_n_f32(m_w_cache[bi][2 * p]),
														  vdup_n_f32(m_w_cache[bi][2 * p + 1]));

					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol =
							vzip1q_f32(vdupq_n_f32(xd[2 * cp]), vdupq_n_f32(xd[2 * cp + 1]));
						const float32x4_t scol =
							vzip1q_f32(vdupq_n_f32(xs[2 * cp]), vdupq_n_f32(xs[2 * cp + 1]));
						int32x4_t s = vdupq_n_s32(0);
						s			= vmmlaq_s32(s, avec[0], bvec[cp][0]);
						s			= vmmlaq_s32(s, avec[1], bvec[cp][1]);
						s			= vmmlaq_s32(s, avec[2], bvec[cp][2]);
						s			= vmmlaq_s32(s, avec[3], bvec[cp][3]);
						facc[p][cp] =
							vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));
						oacc[p][cp] = vfmaq_f32(oacc[p][cp], scol, smin);
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, vaddq_f32(facc[p][cp], oacc[p][cp]));
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_q5_1_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q5_1_block *row =
			(const q5_1_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			matmul_q5_1_q8_qonly_f32_row(row, xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}
}
#endif

#define NR 4

void matmul_q5_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_q5_1_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_1_block);
	const q5_1_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / NR) > 0 ? blocks_per_row : 0;

		static _Thread_local int8x16_t(*lo_cache)[MR] = NULL;
		static _Thread_local int8x16_t(*hi_cache)[MR] = NULL;
		static _Thread_local float (*d_w_cache)[MR]	  = NULL;
		static _Thread_local float (*m_w_cache)[MR]	  = NULL;
		static _Thread_local int cache_cap			  = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				lo_cache  = realloc(lo_cache, sizeof(*lo_cache) * n_bi_tiles);
				hi_cache  = realloc(hi_cache, sizeof(*hi_cache) * n_bi_tiles);
				d_w_cache = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				m_w_cache = realloc(m_w_cache, sizeof(*m_w_cache) * n_bi_tiles);
				cache_cap = n_bi_tiles;
				tlocal_register((void **)&lo_cache);
				tlocal_register((void **)&hi_cache);
				tlocal_register((void **)&d_w_cache);
				tlocal_register((void **)&m_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q5_1_block)), 0,
										   1);
				}
				for (int r = 0; r < MR; r++) {
					const q5_1_block *b =
						(const q5_1_block *)(row_base[r] + ((size_t)bi * sizeof(q5_1_block)));
					q5_1_unpack(b, &lo_cache[bi][r], &hi_cache[bi][r]);
					d_w_cache[bi][r] = f16_to_f32_fast(b->d);
					m_w_cache[bi][r] = f16_to_f32_fast(b->m);
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			float32x4_t off_row[MR];
			for (int r = 0; r < MR; r++) {
				acc_row[r] = vdupq_n_f32(0.0f);
				off_row[r] = vdupq_n_f32(0.0f);
			}

			const q8_1_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	  xd[NR];
				float	  xs[NR];
				int8x16_t xq_lo[NR];
				int8x16_t xq_hi[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xs[c]	 = f16_to_f32_fast(xrow[c][bi].s);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}
				const float32x4_t xd_vec = vld1q_f32(xd);
				const float32x4_t xs_vec = vld1q_f32(xs);

				for (int r = 0; r < MR; r++) {
					const int8x16_t lo	= lo_cache[bi][r];
					const int8x16_t hi	= hi_cache[bi][r];
					const float		d_w = d_w_cache[bi][r];
					const float		m_w = m_w_cache[bi][r];

					const int32x4_t sumi4 = q4_dot_x4_sumi4(lo, hi, xq_lo, xq_hi);

					const float32x4_t sumi_f = vcvtq_f32_s32(sumi4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(sumi_f, d_w));
					off_row[r] = vfmaq_n_f32(off_row[r], xs_vec, m_w);
				}
			}

			for (int r = 0; r < MR; r++) {
				float32x4_t total = vaddq_f32(acc_row[r], off_row[r]);
				float		tmp[4];
				vst1q_f32(tmp, total);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
		}

		for (; t < m; t++) {
			matmul_q5_1_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q5_1_block *row =
			(const q5_1_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			const q8_1_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			int8x16_t		  lo, hi;
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				q5_1_unpack(&row[bi], &lo, &hi);
				const int8x16_t xlo	 = vld1q_s8(xrow[bi].qs);
				const int8x16_t xhi	 = vld1q_s8(xrow[bi].qs + 16);
				const int32_t	sumi = vaddvq_s32(q5_dot(lo, hi, xlo, xhi));
				const float		d	 = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				const float		m	 = f16_to_f32_fast(row[bi].m) * f16_to_f32_fast(xrow[bi].s);
				sumf += (d * (float)sumi) + m;
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef NR

void matmul_q5_1_q8_f32(const void *w, const float *restrict x, float *restrict y, int n, int k,
						quant_scratch *qs) {
	quant_scratch_ensure(qs, (size_t)(k / 32) * sizeof(q8_1_block));
	quantize_q8_1(x, qs->q8_buf, k);
	matmul_q5_1_q8_qonly_f32_row(w, (const q8_1_block *)qs->q8_buf, y, n, k);
}

static void matmul_q5_0_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
										 float *restrict y, int n, int k);

void matmul_generic_f32(const void *w, uint32_t w_type, const float *x, float *y, int n, int k) {
	switch (w_type) {
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_0_R8:
	case GGML_TYPE_Q4_0_R8:
	case GGML_TYPE_IQ4_NL_R8:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_IQ3_S: {
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
		case GGML_TYPE_Q5_0:
			matmul_q5_0_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q5_1:
			matmul_q5_1_q8_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_IQ3_S:
			matmul_iq3_s_q8_k_f32(w, x, y, n, k, &qs);
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

	size_t						row_stride	= ggml_row_size(w_type, k);
	static _Thread_local float *row_buf		= NULL;
	static _Thread_local size_t row_buf_cap = 0;
	if ((size_t)k > row_buf_cap) {
		row_buf		= xrealloc(row_buf, (size_t)k * sizeof(float));
		row_buf_cap = (size_t)k;
		tlocal_register((void **)&row_buf);
	}

	if (k >= 32) {
		for (int i = 0; i < n; i++) {
			const uint8_t *row = (const uint8_t *)w + ((size_t)i * row_stride);
			dequant_row_dispatch(w_type, row, k, row_buf);

			float32x4_t acc0 = vdupq_n_f32(0.0f);
			float32x4_t acc1 = vdupq_n_f32(0.0f);
			float32x4_t acc2 = vdupq_n_f32(0.0f);
			float32x4_t acc3 = vdupq_n_f32(0.0f);
			int			d	 = 0;

			for (; d + 32 <= k; d += 32) {
				float32x4_t x0 = vld1q_f32(x + d);
				float32x4_t x1 = vld1q_f32(x + d + 4);
				float32x4_t x2 = vld1q_f32(x + d + 8);
				float32x4_t x3 = vld1q_f32(x + d + 12);
				float32x4_t x4 = vld1q_f32(x + d + 16);
				float32x4_t x5 = vld1q_f32(x + d + 20);
				float32x4_t x6 = vld1q_f32(x + d + 24);
				float32x4_t x7 = vld1q_f32(x + d + 28);

				float32x4_t r0 = vld1q_f32(row_buf + d);
				float32x4_t r1 = vld1q_f32(row_buf + d + 4);
				float32x4_t r2 = vld1q_f32(row_buf + d + 8);
				float32x4_t r3 = vld1q_f32(row_buf + d + 12);
				float32x4_t r4 = vld1q_f32(row_buf + d + 16);
				float32x4_t r5 = vld1q_f32(row_buf + d + 20);
				float32x4_t r6 = vld1q_f32(row_buf + d + 24);
				float32x4_t r7 = vld1q_f32(row_buf + d + 28);

				acc0 = vfmaq_f32(acc0, x0, r0);
				acc1 = vfmaq_f32(acc1, x1, r1);
				acc2 = vfmaq_f32(acc2, x2, r2);
				acc3 = vfmaq_f32(acc3, x3, r3);
				acc0 = vfmaq_f32(acc0, x4, r4);
				acc1 = vfmaq_f32(acc1, x5, r5);
				acc2 = vfmaq_f32(acc2, x6, r6);
				acc3 = vfmaq_f32(acc3, x7, r7);
			}

			for (; d + 16 <= k; d += 16) {
				float32x4_t x0 = vld1q_f32(x + d);
				float32x4_t x1 = vld1q_f32(x + d + 4);
				float32x4_t x2 = vld1q_f32(x + d + 8);
				float32x4_t x3 = vld1q_f32(x + d + 12);

				acc0 = vfmaq_f32(acc0, x0, vld1q_f32(row_buf + d));
				acc1 = vfmaq_f32(acc1, x1, vld1q_f32(row_buf + d + 4));
				acc2 = vfmaq_f32(acc2, x2, vld1q_f32(row_buf + d + 8));
				acc3 = vfmaq_f32(acc3, x3, vld1q_f32(row_buf + d + 12));
			}

			for (; d + 4 <= k; d += 4) {
				acc0 = vfmaq_f32(acc0, vld1q_f32(x + d), vld1q_f32(row_buf + d));
			}

			float32x4_t acc01 = vaddq_f32(acc0, acc1);
			float32x4_t acc23 = vaddq_f32(acc2, acc3);
			float32x4_t acc	  = vaddq_f32(acc01, acc23);
			float		sum	  = vaddvq_f32(acc);

			for (; d < k; d++) {
				sum += x[d] * row_buf[d];
			}

			y[i] = sum;
		}
	} else if (k >= 16) {
		for (int i = 0; i < n; i++) {
			const uint8_t *row = (const uint8_t *)w + ((size_t)i * row_stride);
			dequant_row_dispatch(w_type, row, k, row_buf);

			float32x4_t acc0 = vdupq_n_f32(0.0f);
			float32x4_t acc1 = vdupq_n_f32(0.0f);
			int			d	 = 0;

			for (; d + 16 <= k; d += 16) {
				float32x4_t x0 = vld1q_f32(x + d);
				float32x4_t x1 = vld1q_f32(x + d + 4);
				float32x4_t x2 = vld1q_f32(x + d + 8);
				float32x4_t x3 = vld1q_f32(x + d + 12);

				acc0 = vfmaq_f32(acc0, x0, vld1q_f32(row_buf + d));
				acc0 = vfmaq_f32(acc0, x1, vld1q_f32(row_buf + d + 4));
				acc1 = vfmaq_f32(acc1, x2, vld1q_f32(row_buf + d + 8));
				acc1 = vfmaq_f32(acc1, x3, vld1q_f32(row_buf + d + 12));
			}

			for (; d + 4 <= k; d += 4) {
				acc0 = vfmaq_f32(acc0, vld1q_f32(x + d), vld1q_f32(row_buf + d));
			}

			float32x4_t acc = vaddq_f32(acc0, acc1);
			float		sum = vaddvq_f32(acc);

			for (; d < k; d++) {
				sum += x[d] * row_buf[d];
			}

			y[i] = sum;
		}
	} else {
		for (int i = 0; i < n; i++) {
			const uint8_t *row = (const uint8_t *)w + ((size_t)i * row_stride);
			dequant_row_dispatch(w_type, row, k, row_buf);

			float32x4_t acc = vdupq_n_f32(0.0f);
			int			d	= 0;

			for (; d + 4 <= k; d += 4) {
				acc = vfmaq_f32(acc, vld1q_f32(x + d), vld1q_f32(row_buf + d));
			}

			float sum = vaddvq_f32(acc);

			for (; d < k; d++) {
				sum += x[d] * row_buf[d];
			}

			y[i] = sum;
		}
	}
}

void quantize_q8_0(const float *x, q8_0_block *dst, int n) {
	int nb = n / 32;
	for (int i = 0; i < nb; i++) {
		float32x4_t v0 = vld1q_f32(x);
		float32x4_t v1 = vld1q_f32(x + 4);
		float32x4_t v2 = vld1q_f32(x + 8);
		float32x4_t v3 = vld1q_f32(x + 12);
		float32x4_t v4 = vld1q_f32(x + 16);
		float32x4_t v5 = vld1q_f32(x + 20);
		float32x4_t v6 = vld1q_f32(x + 24);
		float32x4_t v7 = vld1q_f32(x + 28);
		float32x4_t a0 = vabsq_f32(v0);
		float32x4_t a1 = vabsq_f32(v1);
		float32x4_t a2 = vabsq_f32(v2);
		float32x4_t a3 = vabsq_f32(v3);
		float32x4_t a4 = vabsq_f32(v4);
		float32x4_t a5 = vabsq_f32(v5);
		float32x4_t a6 = vabsq_f32(v6);
		float32x4_t a7 = vabsq_f32(v7);
		float32x4_t m0 = vmaxq_f32(a0, a1);
		float32x4_t m1 = vmaxq_f32(a2, a3);
		float32x4_t m2 = vmaxq_f32(a4, a5);
		float32x4_t m3 = vmaxq_f32(a6, a7);
		m0			   = vmaxq_f32(m0, m1);
		m2			   = vmaxq_f32(m2, m3);
		m0			   = vmaxq_f32(m0, m2);
		float amax	   = vmaxvq_f32(m0);

		float d	 = amax / 127.0f;
		float id = d > 0 ? 1.0f / d : 0.0f;
		dst[i].d = f32_to_f16(d);

		float32x4_t id_v = vdupq_n_f32(id);
		int32x4_t	q0	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v0, id_v)));
		int32x4_t	q1	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v1, id_v)));
		int32x4_t	q2	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v2, id_v)));
		int32x4_t	q3	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v3, id_v)));
		int32x4_t	q4	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v4, id_v)));
		int32x4_t	q5	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v5, id_v)));
		int32x4_t	q6	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v6, id_v)));
		int32x4_t	q7	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v7, id_v)));
		int16x4_t	n0	 = vqmovn_s32(q0);
		int16x4_t	n1	 = vqmovn_s32(q1);
		int16x4_t	n2	 = vqmovn_s32(q2);
		int16x4_t	n3	 = vqmovn_s32(q3);
		int16x4_t	n4	 = vqmovn_s32(q4);
		int16x4_t	n5	 = vqmovn_s32(q5);
		int16x4_t	n6	 = vqmovn_s32(q6);
		int16x4_t	n7	 = vqmovn_s32(q7);
		int16x8_t	h0	 = vcombine_s16(n0, n1);
		int16x8_t	h1	 = vcombine_s16(n2, n3);
		int16x8_t	h2	 = vcombine_s16(n4, n5);
		int16x8_t	h3	 = vcombine_s16(n6, n7);
		int8x8_t	b0	 = vqmovn_s16(h0);
		int8x8_t	b1	 = vqmovn_s16(h1);
		int8x8_t	b2	 = vqmovn_s16(h2);
		int8x8_t	b3	 = vqmovn_s16(h3);
		vst1_s8(dst[i].qs, b0);
		vst1_s8(dst[i].qs + 8, b1);
		vst1_s8(dst[i].qs + 16, b2);
		vst1_s8(dst[i].qs + 24, b3);

		x += 32;
	}
}

void quantize_q8_1(const float *x, void *dst, int n) {
	int			nb = n / 32;
	q8_1_block *y  = dst;
	for (int i = 0; i < nb; i++) {
		float32x4_t v0 = vld1q_f32(x);
		float32x4_t v1 = vld1q_f32(x + 4);
		float32x4_t v2 = vld1q_f32(x + 8);
		float32x4_t v3 = vld1q_f32(x + 12);
		float32x4_t v4 = vld1q_f32(x + 16);
		float32x4_t v5 = vld1q_f32(x + 20);
		float32x4_t v6 = vld1q_f32(x + 24);
		float32x4_t v7 = vld1q_f32(x + 28);
		float32x4_t a0 = vabsq_f32(v0);
		float32x4_t a1 = vabsq_f32(v1);
		float32x4_t a2 = vabsq_f32(v2);
		float32x4_t a3 = vabsq_f32(v3);
		float32x4_t a4 = vabsq_f32(v4);
		float32x4_t a5 = vabsq_f32(v5);
		float32x4_t a6 = vabsq_f32(v6);
		float32x4_t a7 = vabsq_f32(v7);
		float32x4_t m0 = vmaxq_f32(a0, a1);
		float32x4_t m1 = vmaxq_f32(a2, a3);
		float32x4_t m2 = vmaxq_f32(a4, a5);
		float32x4_t m3 = vmaxq_f32(a6, a7);
		m0			   = vmaxq_f32(m0, m1);
		m2			   = vmaxq_f32(m2, m3);
		m0			   = vmaxq_f32(m0, m2);
		float amax	   = vmaxvq_f32(m0);

		float d	 = amax / 127.0f;
		float id = d > 0 ? 1.0f / d : 0.0f;
		y[i].d	 = f32_to_f16(d);

		float32x4_t id_v = vdupq_n_f32(id);
		int32x4_t	q0	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v0, id_v)));
		int32x4_t	q1	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v1, id_v)));
		int32x4_t	q2	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v2, id_v)));
		int32x4_t	q3	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v3, id_v)));
		int32x4_t	q4	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v4, id_v)));
		int32x4_t	q5	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v5, id_v)));
		int32x4_t	q6	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v6, id_v)));
		int32x4_t	q7	 = vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v7, id_v)));
		int32x4_t	s0	 = vaddq_s32(q0, q1);
		int32x4_t	s1	 = vaddq_s32(q2, q3);
		int32x4_t	s2	 = vaddq_s32(q4, q5);
		int32x4_t	s3	 = vaddq_s32(q6, q7);
		s0				 = vaddq_s32(s0, s1);
		s2				 = vaddq_s32(s2, s3);
		s0				 = vaddq_s32(s0, s2);
		int32_t	  sum	 = vaddvq_s32(s0);
		int16x4_t n0	 = vqmovn_s32(q0);
		int16x4_t n1	 = vqmovn_s32(q1);
		int16x4_t n2	 = vqmovn_s32(q2);
		int16x4_t n3	 = vqmovn_s32(q3);
		int16x4_t n4	 = vqmovn_s32(q4);
		int16x4_t n5	 = vqmovn_s32(q5);
		int16x4_t n6	 = vqmovn_s32(q6);
		int16x4_t n7	 = vqmovn_s32(q7);
		int16x8_t h0	 = vcombine_s16(n0, n1);
		int16x8_t h1	 = vcombine_s16(n2, n3);
		int16x8_t h2	 = vcombine_s16(n4, n5);
		int16x8_t h3	 = vcombine_s16(n6, n7);
		int8x8_t  b0	 = vqmovn_s16(h0);
		int8x8_t  b1	 = vqmovn_s16(h1);
		int8x8_t  b2	 = vqmovn_s16(h2);
		int8x8_t  b3	 = vqmovn_s16(h3);
		vst1_s8(y[i].qs, b0);
		vst1_s8(y[i].qs + 8, b1);
		vst1_s8(y[i].qs + 16, b2);
		vst1_s8(y[i].qs + 24, b3);

		if (sum > 127 * 32)
			sum = 127 * 32;
		if (sum < -127 * 32)
			sum = -127 * 32;
		y[i].s = f32_to_f16(d * (float)sum);

		x += 32;
	}
}

void quantize_q8_k(const float *x, q8_k_block *y, int n) {
	int nb = n / 256;
	for (int i = 0; i < nb; i++) {
		float		max		= 0;
		float32x4_t amax_v0 = vdupq_n_f32(0.0f);
		float32x4_t amax_v1 = vdupq_n_f32(0.0f);
		float32x4_t max_v0	= vdupq_n_f32(0.0f);
		float32x4_t max_v1	= vdupq_n_f32(0.0f);
		for (int j = 0; j < 256; j += 8) {
			float32x4_t v0 = vld1q_f32(x + j);
			float32x4_t v1 = vld1q_f32(x + j + 4);
			float32x4_t a0 = vabsq_f32(v0);
			float32x4_t a1 = vabsq_f32(v1);
			uint32x4_t	m0 = vcgtq_f32(a0, amax_v0);
			uint32x4_t	m1 = vcgtq_f32(a1, amax_v1);
			amax_v0		   = vmaxq_f32(amax_v0, a0);
			amax_v1		   = vmaxq_f32(amax_v1, a1);
			max_v0		   = vbslq_f32(m0, v0, max_v0);
			max_v1		   = vbslq_f32(m1, v1, max_v1);
		}
		float		amax_lanes[4] __attribute__((aligned(16)));
		float		max_lanes[4] __attribute__((aligned(16)));
		uint32x4_t	combined_mask = vcgtq_f32(amax_v1, amax_v0);
		float32x4_t amax_v		  = vmaxq_f32(amax_v0, amax_v1);
		float32x4_t max_v		  = vbslq_f32(combined_mask, max_v1, max_v0);
		vst1q_f32(amax_lanes, amax_v);
		vst1q_f32(max_lanes, max_v);
		float amax = amax_lanes[0];
		max		   = max_lanes[0];
		for (int lane = 1; lane < 4; lane++) {
			if (amax_lanes[lane] > amax) {
				amax = amax_lanes[lane];
				max	 = max_lanes[lane];
			}
		}
		if (!amax) {
			y[i].d = 0;
			memset(y[i].qs, 0, 256);
			memset(y[i].bsums, 0, 16 * sizeof(int16_t));
			x += 256;
			continue;
		}
		float		 iscale = -127.0f / max;
		float32x4_t	 is_v	= vdupq_n_f32(iscale);
		int8_t		*qsp	= y[i].qs;
		const float *xp		= x;
		for (int outer = 0; outer < 4; outer++) {
			for (int sub = 0; sub < 4; sub++) {
				float32x4_t v0	= vld1q_f32(xp);
				float32x4_t v1	= vld1q_f32(xp + 4);
				float32x4_t v2	= vld1q_f32(xp + 8);
				float32x4_t v3	= vld1q_f32(xp + 12);
				int32x4_t	q0	= vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v0, is_v)));
				int32x4_t	q1	= vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v1, is_v)));
				int32x4_t	q2	= vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v2, is_v)));
				int32x4_t	q3	= vcvtq_s32_f32(vrndaq_f32(vmulq_f32(v3, is_v)));
				int32x4_t	lo	= vdupq_n_s32(-127);
				int32x4_t	hi	= vdupq_n_s32(127);
				q0				= vmaxq_s32(vminq_s32(q0, hi), lo);
				q1				= vmaxq_s32(vminq_s32(q1, hi), lo);
				q2				= vmaxq_s32(vminq_s32(q2, hi), lo);
				q3				= vmaxq_s32(vminq_s32(q3, hi), lo);
				int32x4_t s01	= vaddq_s32(q0, q1);
				int32x4_t s23	= vaddq_s32(q2, q3);
				int32x4_t s_all = vaddq_s32(s01, s23);
				int32_t	  sum	= vaddvq_s32(s_all);
				int16x4_t n0	= vqmovn_s32(q0);
				int16x4_t n1	= vqmovn_s32(q1);
				int16x4_t n2	= vqmovn_s32(q2);
				int16x4_t n3	= vqmovn_s32(q3);
				int16x8_t h0	= vcombine_s16(n0, n1);
				int16x8_t h1	= vcombine_s16(n2, n3);
				int8x8_t  b0	= vqmovn_s16(h0);
				int8x8_t  b1	= vqmovn_s16(h1);
				vst1_s8(qsp, b0);
				vst1_s8(qsp + 8, b1);
				y[i].bsums[(outer * 4) + sub] = (int16_t)sum;
				qsp += 16;
				xp += 16;
			}
		}
		y[i].d = 1.0f / iscale;
		x += 256;
	}
}

static void matmul_q4_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
									   float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_0_block);
	const q4_0_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float32x4_t acc0 = vdupq_n_f32(0.0f);
		float32x4_t acc1 = vdupq_n_f32(0.0f);

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32_fast(xq[bi].d);

			const int8x16_t xq_lo = vld1q_s8(xq8);
			const int8x16_t xq_hi = vld1q_s8(xq8 + 16);

			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_0_block)), 0, 1);
			}

			int32_t	 sumi_lane[8];
			uint16_t d_w_raw[8];
#if defined(__ARM_FEATURE_DOTPROD)
			for (int r = 0; r < 8; r++) {
				const q4_0_block *row =
					(const q4_0_block *)(row_base[r] + (size_t)bi * sizeof(q4_0_block));
				const uint8_t *restrict qs = row->qs;
				const uint8x16_t q		   = vld1q_u8(qs);
				const int8x16_t	 lo =
					vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
				const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), vdupq_n_s8(8));

				int32x4_t acc = vdotq_s32(vdupq_n_s32(0), lo, xq_lo);
				acc			  = vdotq_s32(acc, hi, xq_hi);
				sumi_lane[r]  = vaddvq_s32(acc);
				d_w_raw[r]	  = row->d;
			}
#else
			for (int r = 0; r < 8; r++) {
				const q4_0_block *row =
					(const q4_0_block *)(row_base[r] + ((size_t)bi * sizeof(q4_0_block)));
				const uint8_t *restrict qs = row->qs;
				const uint8x16_t q		   = vld1q_u8(qs);
				const int8x16_t	 lo =
					vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
				const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), vdupq_n_s8(8));

				int16x8_t p0  = vmull_s8(vget_low_s8(lo), vget_low_s8(xq_lo));
				int16x8_t p1  = vmull_s8(vget_high_s8(lo), vget_high_s8(xq_lo));
				int16x8_t p2  = vmull_s8(vget_low_s8(hi), vget_low_s8(xq_hi));
				int16x8_t p3  = vmull_s8(vget_high_s8(hi), vget_high_s8(xq_hi));
				int32x4_t acc = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
				acc			  = vaddq_s32(acc, vpaddlq_s16(p2));
				acc			  = vaddq_s32(acc, vpaddlq_s16(p3));
				sumi_lane[r]  = vaddvq_s32(acc);
				d_w_raw[r]	  = row->d;
			}
#endif
			const uint16x8_t d_w_u16 = vld1q_u16(d_w_raw);
			float32x4_t		 d_w0	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_w_u16)));
			float32x4_t		 d_w1	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_w_u16)));
			float32x4_t		 d_xq_v	 = vdupq_n_f32(d_xq);

			int32x4_t sumi0 = vld1q_s32(sumi_lane);
			int32x4_t sumi1 = vld1q_s32(sumi_lane + 4);
			acc0			= vfmaq_f32(acc0, vmulq_f32(d_w0, d_xq_v), vcvtq_f32_s32(sumi0));
			acc1			= vfmaq_f32(acc1, vmulq_f32(d_w1, d_xq_v), vcvtq_f32_s32(sumi1));
		}

		store_acc8(acc0, acc1, y, i);
	}

	for (; i < n; i++) {
		const q4_0_block *row =
			(const q4_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t *restrict qs = row[bi].qs;
			const int8_t *restrict xq8 = xq[bi].qs;
			int sumi0				   = 0;
			int sumi1				   = 0;
			for (int j = 0; j < 16; j++) {
				sumi0 += ((int)(qs[j] & 0xF) - 8) * (int)xq8[j];
				sumi1 += ((int)(qs[j] >> 4) - 8) * (int)xq8[j + 16];
			}
			float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xq[bi].d);
			sumf += d * (float)(sumi0 + sumi1);
		}
		y[i] = sumf;
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_q4_q8_qonly_f32_i8mm(const void *w, const q8_0_block *restrict xq,
										size_t		xq_row_stride_blocks, float *restrict y,
										int y_row_stride, int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_0_block);
	const q4_0_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++)
					facc[p][c] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_0_block)), 0,
										   1);
				}

				int8x16_t xq_lo[I8MM_NR];
				int8x16_t xq_hi[I8MM_NR];
				float	  xd[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}

				int8x16_t bvec[I8MM_NR / 2][4];
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					const int ca = 2 * cp;
					const int cb = 2 * cp + 1;
					bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb]));
					bvec[cp][1]	 = vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));
					bvec[cp][2]	 = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));
					bvec[cp][3]	 = vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));
				}

				for (int p = 0; p < MR / 2; p++) {
					const q4_0_block *b0 =
						(const q4_0_block *)(row_base[2 * p] + ((size_t)bi * sizeof(q4_0_block)));
					const q4_0_block *b1  = (const q4_0_block *)(row_base[2 * p + 1] +
																 ((size_t)bi * sizeof(q4_0_block)));
					const float		  dwa = f16_to_f32_fast(b0->d);
					const float		  dwb = f16_to_f32_fast(b1->d);

					const uint8x16_t q0		 = vld1q_u8(b0->qs);
					const uint8x16_t q1		 = vld1q_u8(b1->qs);
					const uint8x16_t l0		 = vandq_u8(q0, vdupq_n_u8(0x0F));
					const uint8x16_t h0		 = vshrq_n_u8(q0, 4);
					const uint8x16_t l1		 = vandq_u8(q1, vdupq_n_u8(0x0F));
					const uint8x16_t h1		 = vshrq_n_u8(q1, 4);
					const int8x16_t	 avec[4] = {
						vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(l0), vget_low_u8(l1))),
								 vdupq_n_s8(8)),
						vsubq_s8(
							vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(l0), vget_high_u8(l1))),
							vdupq_n_s8(8)),
						vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(h0), vget_low_u8(h1))),
								 vdupq_n_s8(8)),
						vsubq_s8(
							vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(h0), vget_high_u8(h1))),
							vdupq_n_s8(8)),
					};

					const float32x4_t srow = vcombine_f32(vdup_n_f32(dwa), vdup_n_f32(dwb));

					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol =
							vzip1q_f32(vdupq_n_f32(xd[2 * cp]), vdupq_n_f32(xd[2 * cp + 1]));
						int32x4_t s = vdupq_n_s32(0);
						s			= vmmlaq_s32(s, avec[0], bvec[cp][0]);
						s			= vmmlaq_s32(s, avec[1], bvec[cp][1]);
						s			= vmmlaq_s32(s, avec[2], bvec[cp][2]);
						s			= vmmlaq_s32(s, avec[3], bvec[cp][3]);
						facc[p][cp] =
							vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, facc[p][cp]);
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_q4_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
									   y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q4_0_block *row =
			(const q4_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			matmul_q4_q8_qonly_f32_row(row, xq + ((size_t)t * xq_row_stride_blocks),
									   y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}
}
#endif

#define NR 4

void matmul_q4_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							size_t xq_row_stride_blocks, float *restrict y, int y_row_stride, int n,
							int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_q4_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
#if defined(__ARM_FEATURE_SVE)
	matmul_q4_q8_qonly_f32_sve(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_0_block);
	const q4_0_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	  xd[NR];
				int8x16_t xq_lo[NR];
				int8x16_t xq_hi[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_0_block)), 0,
										   1);
				}

				for (int r = 0; r < MR; r++) {
					const q4_0_block *b =
						(const q4_0_block *)(row_base[r] + ((size_t)bi * sizeof(q4_0_block)));
					const float		 d_w  = f16_to_f32_fast(b->d);
					const uint8x16_t q	  = vld1q_u8(b->qs);
					const uint8x16_t lo_u = vandq_u8(q, vdupq_n_u8(0x0F));
					const uint8x16_t hi_u = vshrq_n_u8(q, 4);

					const int8x16_t	  lo	 = vsubq_s8(vreinterpretq_s8_u8(lo_u), vdupq_n_s8(8));
					const int8x16_t	  hi	 = vsubq_s8(vreinterpretq_s8_u8(hi_u), vdupq_n_s8(8));
					const int32x4_t	  sumi4	 = q4_dot_x4_sumi4(lo, hi, xq_lo, xq_hi);
					const float32x4_t sumi_f = vcvtq_f32_s32(sumi4);
					const float32x4_t scaled = vmulq_n_f32(vmulq_f32(xd_vec, sumi_f), d_w);

					acc_row[r] = vaddq_f32(acc_row[r], scaled);
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_q4_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
									   y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q4_0_block *row =
			(const q4_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			const q8_0_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *restrict qs = row[bi].qs;
				const int8_t *restrict xq8 = xrow[bi].qs;
				int sumi0				   = 0;
				int sumi1				   = 0;
				for (int j = 0; j < 16; j++) {
					sumi0 += ((int)(qs[j] & 0xF) - 8) * (int)xq8[j];
					sumi1 += ((int)(qs[j] >> 4) - 8) * (int)xq8[j + 16];
				}
				float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				sumf += d * (float)(sumi0 + sumi1);
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef NR

static void matmul_q4_1_q8_qonly_f32_row(const void *w, const q8_1_block *restrict xq,
										 float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_1_block);
	const q4_1_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float32x4_t acc0 = vdupq_n_f32(0.0f);
		float32x4_t acc1 = vdupq_n_f32(0.0f);
		float32x4_t off0 = vdupq_n_f32(0.0f);
		float32x4_t off1 = vdupq_n_f32(0.0f);

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32_fast(xq[bi].d);
			const float s_xq		   = f16_to_f32_fast(xq[bi].s);

			const int8x16_t xq_lo = vld1q_s8(xq8);
			const int8x16_t xq_hi = vld1q_s8(xq8 + 16);

			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_1_block)), 0, 1);
			}

			int32_t	 sumi_lane[8];
			uint16_t d_w_raw[8];
			uint16_t m_w_raw[8];

#if defined(__ARM_FEATURE_DOTPROD)
			for (int r = 0; r < 8; r++) {
				const q4_1_block *row =
					(const q4_1_block *)(row_base[r] + (size_t)bi * sizeof(q4_1_block));
				d_w_raw[r]				   = row->d;
				m_w_raw[r]				   = row->m;
				const uint8_t *restrict qs = row->qs;
				const uint8x16_t q		   = vld1q_u8(qs);
				const int8x16_t	 lo		   = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F)));
				const int8x16_t	 hi		   = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));

				int32x4_t acc = vdotq_s32(vdupq_n_s32(0), lo, xq_lo);
				acc			  = vdotq_s32(acc, hi, xq_hi);
				sumi_lane[r]  = vaddvq_s32(acc);
			}
#else
			for (int r = 0; r < 8; r++) {
				const q4_1_block *row =
					(const q4_1_block *)(row_base[r] + ((size_t)bi * sizeof(q4_1_block)));
				d_w_raw[r]				   = row->d;
				m_w_raw[r]				   = row->m;
				const uint8_t *restrict qs = row->qs;
				const uint8x16_t q		   = vld1q_u8(qs);
				const int8x16_t	 lo		   = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F)));
				const int8x16_t	 hi		   = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));

				int16x8_t p0  = vmull_s8(vget_low_s8(lo), vget_low_s8(xq_lo));
				int16x8_t p1  = vmull_s8(vget_high_s8(lo), vget_high_s8(xq_lo));
				int16x8_t p2  = vmull_s8(vget_low_s8(hi), vget_low_s8(xq_hi));
				int16x8_t p3  = vmull_s8(vget_high_s8(hi), vget_high_s8(xq_hi));
				int32x4_t acc = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
				acc			  = vaddq_s32(acc, vpaddlq_s16(p2));
				acc			  = vaddq_s32(acc, vpaddlq_s16(p3));
				sumi_lane[r]  = vaddvq_s32(acc);
			}
#endif
			int32x4_t		 sumi0	 = vld1q_s32(sumi_lane);
			int32x4_t		 sumi1	 = vld1q_s32(sumi_lane + 4);
			float32x4_t		 sumi0f	 = vcvtq_f32_s32(sumi0);
			float32x4_t		 sumi1f	 = vcvtq_f32_s32(sumi1);
			const uint16x8_t d_w_u16 = vld1q_u16(d_w_raw);
			const uint16x8_t m_w_u16 = vld1q_u16(m_w_raw);
			float32x4_t		 d_w0	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_w_u16)));
			float32x4_t		 d_w1	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_w_u16)));
			float32x4_t		 m_w0	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(m_w_u16)));
			float32x4_t		 m_w1	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(m_w_u16)));
			float32x4_t		 d_xq_v	 = vdupq_n_f32(d_xq);
			float32x4_t		 s_xq_v	 = vdupq_n_f32(s_xq);

			acc0 = vfmaq_f32(acc0, vmulq_f32(d_w0, d_xq_v), sumi0f);
			acc1 = vfmaq_f32(acc1, vmulq_f32(d_w1, d_xq_v), sumi1f);
			off0 = vfmaq_f32(off0, m_w0, s_xq_v);
			off1 = vfmaq_f32(off1, m_w1, s_xq_v);
		}

		float32x4_t final0 = vaddq_f32(acc0, off0);
		float32x4_t final1 = vaddq_f32(acc1, off1);
		float		tmp0[4], tmp1[4];
		vst1q_f32(tmp0, final0);
		vst1q_f32(tmp1, final1);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
	}

	for (; i < n; i++) {
		const q4_1_block *row =
			(const q4_1_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t *restrict qs = row[bi].qs;
			const int8_t *restrict xq8 = xq[bi].qs;
			int sumi0				   = 0;
			int sumi1				   = 0;
			for (int j = 0; j < 16; j++) {
				sumi0 += (int)(qs[j] & 0xF) * (int)xq8[j];
				sumi1 += (int)(qs[j] >> 4) * (int)xq8[j + 16];
			}
			float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xq[bi].d);
			float m = f16_to_f32_fast(row[bi].m) * f16_to_f32_fast(xq[bi].s);
			sumf += (d * (float)(sumi0 + sumi1)) + m;
		}
		y[i] = sumf;
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_q4_1_q8_qonly_f32_i8mm(const void *w, const q8_1_block *restrict xq,
										  size_t	  xq_row_stride_blocks, float *restrict y,
										  int y_row_stride, int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_1_block);
	const q4_1_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			float32x4_t oacc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++) {
					facc[p][c] = vdupq_n_f32(0.0f);
					oacc[p][c] = vdupq_n_f32(0.0f);
				}

			const q8_1_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_1_block)), 0,
										   1);
				}

				int8x16_t xq_lo[I8MM_NR];
				int8x16_t xq_hi[I8MM_NR];
				float	  xd[I8MM_NR];
				float	  xs[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xs[c]	 = f16_to_f32_fast(xrow[c][bi].s);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}

				int8x16_t bvec[I8MM_NR / 2][4];
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					const int ca = 2 * cp;
					const int cb = 2 * cp + 1;
					bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb]));
					bvec[cp][1]	 = vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));
					bvec[cp][2]	 = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));
					bvec[cp][3]	 = vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));
				}

				for (int p = 0; p < MR / 2; p++) {
					const q4_1_block *b0 =
						(const q4_1_block *)(row_base[2 * p] + ((size_t)bi * sizeof(q4_1_block)));
					const q4_1_block *b1 = (const q4_1_block *)(row_base[2 * p + 1] +
																((size_t)bi * sizeof(q4_1_block)));

					const uint8x16_t q0		 = vld1q_u8(b0->qs);
					const uint8x16_t q1		 = vld1q_u8(b1->qs);
					const uint8x16_t l0		 = vandq_u8(q0, vdupq_n_u8(0x0F));
					const uint8x16_t h0		 = vshrq_n_u8(q0, 4);
					const uint8x16_t l1		 = vandq_u8(q1, vdupq_n_u8(0x0F));
					const uint8x16_t h1		 = vshrq_n_u8(q1, 4);
					const int8x16_t	 avec[4] = {
						vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(l0), vget_low_u8(l1))),
						vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(l0), vget_high_u8(l1))),
						vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(h0), vget_low_u8(h1))),
						vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(h0), vget_high_u8(h1))),
					};

					const float32x4_t srow = vcombine_f32(vdup_n_f32(f16_to_f32_fast(b0->d)),
														  vdup_n_f32(f16_to_f32_fast(b1->d)));
					const float32x4_t smin = vcombine_f32(vdup_n_f32(f16_to_f32_fast(b0->m)),
														  vdup_n_f32(f16_to_f32_fast(b1->m)));

					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol =
							vzip1q_f32(vdupq_n_f32(xd[2 * cp]), vdupq_n_f32(xd[2 * cp + 1]));
						const float32x4_t scol =
							vzip1q_f32(vdupq_n_f32(xs[2 * cp]), vdupq_n_f32(xs[2 * cp + 1]));
						int32x4_t s = vdupq_n_s32(0);
						s			= vmmlaq_s32(s, avec[0], bvec[cp][0]);
						s			= vmmlaq_s32(s, avec[1], bvec[cp][1]);
						s			= vmmlaq_s32(s, avec[2], bvec[cp][2]);
						s			= vmmlaq_s32(s, avec[3], bvec[cp][3]);
						facc[p][cp] =
							vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));
						oacc[p][cp] = vfmaq_f32(oacc[p][cp], scol, smin);
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, vaddq_f32(facc[p][cp], oacc[p][cp]));
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_q4_1_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q4_1_block *row =
			(const q4_1_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			matmul_q4_1_q8_qonly_f32_row(row, xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}
}
#endif

#define NR 4

void matmul_q4_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_q4_1_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_1_block);
	const q4_1_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_dot[MR];
			float32x4_t acc_off[MR];
			for (int r = 0; r < MR; r++) {
				acc_dot[r] = vdupq_n_f32(0.0f);
				acc_off[r] = vdupq_n_f32(0.0f);
			}

			const q8_1_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	  xd[NR];
				float	  xs[NR];
				int8x16_t xq_lo[NR];
				int8x16_t xq_hi[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xs[c]	 = f16_to_f32_fast(xrow[c][bi].s);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}
				const float32x4_t xd_vec = vld1q_f32(xd);
				const float32x4_t xs_vec = vld1q_f32(xs);

				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_1_block)), 0,
										   1);
				}

				for (int r = 0; r < MR; r++) {
					const q4_1_block *b =
						(const q4_1_block *)(row_base[r] + ((size_t)bi * sizeof(q4_1_block)));
					const uint8x16_t q	  = vld1q_u8(b->qs);
					const uint8x16_t lo_u = vandq_u8(q, vdupq_n_u8(0x0F));
					const uint8x16_t hi_u = vshrq_n_u8(q, 4);
					const int8x16_t	 lo	  = vreinterpretq_s8_u8(lo_u);
					const int8x16_t	 hi	  = vreinterpretq_s8_u8(hi_u);
					const float		 d_w  = f16_to_f32_fast(b->d);
					const float		 m_w  = f16_to_f32_fast(b->m);

					const int32x4_t	  sumi4	 = q4_dot_x4_sumi4(lo, hi, xq_lo, xq_hi);
					const float32x4_t sumi_f = vcvtq_f32_s32(sumi4);
					acc_dot[r] = vfmaq_f32(acc_dot[r], xd_vec, vmulq_n_f32(sumi_f, d_w));
					acc_off[r] = vfmaq_f32(acc_off[r], xs_vec, vdupq_n_f32(m_w));
				}
			}

			for (int r = 0; r < MR; r++) {
				float32x4_t total = vaddq_f32(acc_dot[r], acc_off[r]);
				float		tmp[4];
				vst1q_f32(tmp, total);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
		}

		for (; t < m; t++) {
			matmul_q4_1_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q4_1_block *row =
			(const q4_1_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			const q8_1_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *restrict qs = row[bi].qs;
				const int8_t *restrict xq8 = xrow[bi].qs;
				int sumi0 = 0, sumi1 = 0;
				for (int j = 0; j < 16; j++) {
					sumi0 += (int)(qs[j] & 0xF) * (int)xq8[j];
					sumi1 += (int)(qs[j] >> 4) * (int)xq8[j + 16];
				}
				float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				float m = f16_to_f32_fast(row[bi].m) * f16_to_f32_fast(xrow[bi].s);
				sumf += (d * (float)(sumi0 + sumi1)) + m;
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef NR

static inline void q5_0_unpack(const q5_0_block *b, int8x16_t *lo, int8x16_t *hi) {
	uint32_t qh;
	memcpy(&qh, b->qh, 4);

	const uint8x16_t qs		= vld1q_u8(b->qs);
	const uint8x16_t nib_lo = vandq_u8(qs, vdupq_n_u8(0x0F));
	const uint8x16_t nib_hi = vshrq_n_u8(qs, 4);

	const uint8x16_t qh_lo_bytes = vreinterpretq_u8_u32(vdupq_n_u32(qh));
	const uint8x16_t qh_hi_bytes = vreinterpretq_u8_u32(vdupq_n_u32(qh >> 16));
	const uint8x16_t byte_idx	 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
	const uint8x16_t bit_in_byte = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
	const uint8x16_t qh_lo_sel	 = vqtbl1q_u8(qh_lo_bytes, byte_idx);
	const uint8x16_t qh_hi_sel	 = vqtbl1q_u8(qh_hi_bytes, byte_idx);
	const int8x16_t	 lshift4	 = vsubq_s8(vdupq_n_s8(4), vreinterpretq_s8_u8(bit_in_byte));
	const uint8x16_t hi_lo_v	 = vandq_u8(vshlq_u8(qh_lo_sel, lshift4), vdupq_n_u8(0x10));
	const uint8x16_t hi_hi_v	 = vandq_u8(vshlq_u8(qh_hi_sel, lshift4), vdupq_n_u8(0x10));

	const uint8x16_t v16 = vdupq_n_u8(16);
	*lo					 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(nib_lo, hi_lo_v), v16));
	*hi					 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(nib_hi, hi_hi_v), v16));
}

static inline void q5_1_unpack(const q5_1_block *b, int8x16_t *lo, int8x16_t *hi) {
	uint32_t qh;
	memcpy(&qh, b->qh, 4);

	const uint8x16_t qs		= vld1q_u8(b->qs);
	const uint8x16_t nib_lo = vandq_u8(qs, vdupq_n_u8(0x0F));
	const uint8x16_t nib_hi = vshrq_n_u8(qs, 4);

	const uint8x16_t qh_lo_bytes = vreinterpretq_u8_u32(vdupq_n_u32(qh));
	const uint8x16_t qh_hi_bytes = vreinterpretq_u8_u32(vdupq_n_u32(qh >> 16));
	const uint8x16_t byte_idx	 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
	const uint8x16_t bit_in_byte = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
	const uint8x16_t qh_lo_sel	 = vqtbl1q_u8(qh_lo_bytes, byte_idx);
	const uint8x16_t qh_hi_sel	 = vqtbl1q_u8(qh_hi_bytes, byte_idx);
	const int8x16_t	 lshift4	 = vsubq_s8(vdupq_n_s8(4), vreinterpretq_s8_u8(bit_in_byte));
	const uint8x16_t hi_lo_v	 = vandq_u8(vshlq_u8(qh_lo_sel, lshift4), vdupq_n_u8(0x10));
	const uint8x16_t hi_hi_v	 = vandq_u8(vshlq_u8(qh_hi_sel, lshift4), vdupq_n_u8(0x10));

	*lo = vreinterpretq_s8_u8(vorrq_u8(nib_lo, hi_lo_v));
	*hi = vreinterpretq_s8_u8(vorrq_u8(nib_hi, hi_hi_v));
}

static void matmul_q5_0_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
										 float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_0_block);
	const q5_0_block *Wb			 = w;

	for (int i = 0; i < n; i++) {
		const q5_0_block *row =
			(const q5_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			int8x16_t lo, hi;
			q5_0_unpack(&row[bi], &lo, &hi);
			const int8x16_t xlo	 = vld1q_s8(xq[bi].qs);
			const int8x16_t xhi	 = vld1q_s8(xq[bi].qs + 16);
			const int32_t	sumi = vaddvq_s32(q5_dot(lo, hi, xlo, xhi));
			const float		d	 = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xq[bi].d);
			sumf += d * (float)sumi;
		}
		y[i] = sumf;
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_q5_0_q8_qonly_f32_i8mm(const void *w, const q8_0_block *restrict xq,
										  size_t	  xq_row_stride_blocks, float *restrict y,
										  int y_row_stride, int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_0_block);
	const q5_0_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / I8MM_NR) > 0 ? blocks_per_row : 0;

		static _Thread_local int8x16_t(*lo_cache)[MR] = NULL;
		static _Thread_local int8x16_t(*hi_cache)[MR] = NULL;
		static _Thread_local float (*d_w_cache)[MR]	  = NULL;
		static _Thread_local int cache_cap			  = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				lo_cache  = realloc(lo_cache, sizeof(*lo_cache) * n_bi_tiles);
				hi_cache  = realloc(hi_cache, sizeof(*hi_cache) * n_bi_tiles);
				d_w_cache = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				cache_cap = n_bi_tiles;
				tlocal_register((void **)&lo_cache);
				tlocal_register((void **)&hi_cache);
				tlocal_register((void **)&d_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q5_0_block)), 0,
										   1);
				}
				for (int r = 0; r < MR; r++) {
					const q5_0_block *b =
						(const q5_0_block *)(row_base[r] + ((size_t)bi * sizeof(q5_0_block)));
					q5_0_unpack(b, &lo_cache[bi][r], &hi_cache[bi][r]);
					d_w_cache[bi][r] = f16_to_f32_fast(b->d);
				}
			}
		}

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++)
					facc[p][c] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				int8x16_t xq_lo[I8MM_NR];
				int8x16_t xq_hi[I8MM_NR];
				float	  xd[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}

				int8x16_t bvec[I8MM_NR / 2][4];
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					const int ca = 2 * cp;
					const int cb = 2 * cp + 1;
					bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb]));
					bvec[cp][1]	 = vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));
					bvec[cp][2]	 = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));
					bvec[cp][3]	 = vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));
				}

				for (int p = 0; p < MR / 2; p++) {
					const int8x16_t l0		= lo_cache[bi][2 * p];
					const int8x16_t l1		= lo_cache[bi][2 * p + 1];
					const int8x16_t h0		= hi_cache[bi][2 * p];
					const int8x16_t h1		= hi_cache[bi][2 * p + 1];
					const int8x16_t avec[4] = {
						vcombine_s8(vget_low_s8(l0), vget_low_s8(l1)),
						vcombine_s8(vget_high_s8(l0), vget_high_s8(l1)),
						vcombine_s8(vget_low_s8(h0), vget_low_s8(h1)),
						vcombine_s8(vget_high_s8(h0), vget_high_s8(h1)),
					};

					const float32x4_t srow = vcombine_f32(vdup_n_f32(d_w_cache[bi][2 * p]),
														  vdup_n_f32(d_w_cache[bi][2 * p + 1]));

					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol =
							vzip1q_f32(vdupq_n_f32(xd[2 * cp]), vdupq_n_f32(xd[2 * cp + 1]));
						int32x4_t s = vdupq_n_s32(0);
						s			= vmmlaq_s32(s, avec[0], bvec[cp][0]);
						s			= vmmlaq_s32(s, avec[1], bvec[cp][1]);
						s			= vmmlaq_s32(s, avec[2], bvec[cp][2]);
						s			= vmmlaq_s32(s, avec[3], bvec[cp][3]);
						facc[p][cp] =
							vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, facc[p][cp]);
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_q5_0_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q5_0_block *row =
			(const q5_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			matmul_q5_0_q8_qonly_f32_row(row, xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}
}
#endif

#define NR 4

void matmul_q5_0_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_q5_0_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_0_block);
	const q5_0_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / NR) > 0 ? blocks_per_row : 0;

		static _Thread_local int8x16_t(*lo_cache)[MR] = NULL;
		static _Thread_local int8x16_t(*hi_cache)[MR] = NULL;
		static _Thread_local float (*d_w_cache)[MR]	  = NULL;
		static _Thread_local int cache_cap			  = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				lo_cache  = realloc(lo_cache, sizeof(*lo_cache) * n_bi_tiles);
				hi_cache  = realloc(hi_cache, sizeof(*hi_cache) * n_bi_tiles);
				d_w_cache = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				cache_cap = n_bi_tiles;
				tlocal_register((void **)&lo_cache);
				tlocal_register((void **)&hi_cache);
				tlocal_register((void **)&d_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q5_0_block)), 0,
										   1);
				}
				for (int r = 0; r < MR; r++) {
					const q5_0_block *b =
						(const q5_0_block *)(row_base[r] + ((size_t)bi * sizeof(q5_0_block)));
					q5_0_unpack(b, &lo_cache[bi][r], &hi_cache[bi][r]);
					d_w_cache[bi][r] = f16_to_f32_fast(b->d);
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				int8x16_t xq_lo[NR];
				int8x16_t xq_hi[NR];
				float	  xd[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				for (int r = 0; r < MR; r++) {
					const int8x16_t lo	= lo_cache[bi][r];
					const int8x16_t hi	= hi_cache[bi][r];
					const float		d_w = d_w_cache[bi][r];

					const int32x4_t sumi4 = q4_dot_x4_sumi4(lo, hi, xq_lo, xq_hi);

					const float32x4_t sumi_f = vcvtq_f32_s32(sumi4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(sumi_f, d_w));
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_q5_0_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q5_0_block *row =
			(const q5_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			const q8_0_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			int8x16_t		  lo, hi;
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				q5_0_unpack(&row[bi], &lo, &hi);
				const int8x16_t xlo	 = vld1q_s8(xrow[bi].qs);
				const int8x16_t xhi	 = vld1q_s8(xrow[bi].qs + 16);
				const int32_t	sumi = vaddvq_s32(q5_dot(lo, hi, xlo, xhi));
				const float		d	 = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				sumf += d * (float)sumi;
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef NR

static void matmul_q8_0_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
										 float *restrict y, int n, int k) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q8_0_block);
	const q8_0_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float32x4_t acc_lo = vdupq_n_f32(0.0f);
		float32x4_t acc_hi = vdupq_n_f32(0.0f);

		const q8_0_block *r0 =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)(i + 0) * row_stride));
		const q8_0_block *r1 =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)(i + 1) * row_stride));
		const q8_0_block *r2 =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)(i + 2) * row_stride));
		const q8_0_block *r3 =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)(i + 3) * row_stride));
		const q8_0_block *r4 =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)(i + 4) * row_stride));
		const q8_0_block *r5 =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)(i + 5) * row_stride));
		const q8_0_block *r6 =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)(i + 6) * row_stride));
		const q8_0_block *r7 =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)(i + 7) * row_stride));

		int bi = 0;
		for (; bi + 1 < blocks_per_row; bi += 2) {
			if (bi + 2 < blocks_per_row) {
				__builtin_prefetch(&r0[bi + 2], 0, 1);
				__builtin_prefetch(&r1[bi + 2], 0, 1);
				__builtin_prefetch(&r2[bi + 2], 0, 1);
				__builtin_prefetch(&r3[bi + 2], 0, 1);
				__builtin_prefetch(&r4[bi + 2], 0, 1);
				__builtin_prefetch(&r5[bi + 2], 0, 1);
				__builtin_prefetch(&r6[bi + 2], 0, 1);
				__builtin_prefetch(&r7[bi + 2], 0, 1);
			}

			const float		d_xq0 = f16_to_f32_fast(xq[bi].d);
			const float		d_xq1 = f16_to_f32_fast(xq[bi + 1].d);
			const int8x16_t xl0	  = vld1q_s8(xq[bi].qs);
			const int8x16_t xh0	  = vld1q_s8(xq[bi].qs + 16);
			const int8x16_t xl1	  = vld1q_s8(xq[bi + 1].qs);
			const int8x16_t xh1	  = vld1q_s8(xq[bi + 1].qs + 16);

			int32x4_t p0a = dotprod2_s8(vld1q_s8(r0[bi].qs), xl0, vld1q_s8(r0[bi].qs + 16), xh0);
			int32x4_t p0b =
				dotprod2_s8(vld1q_s8(r0[bi + 1].qs), xl1, vld1q_s8(r0[bi + 1].qs + 16), xh1);

			int32x4_t p1a = dotprod2_s8(vld1q_s8(r1[bi].qs), xl0, vld1q_s8(r1[bi].qs + 16), xh0);
			int32x4_t p1b =
				dotprod2_s8(vld1q_s8(r1[bi + 1].qs), xl1, vld1q_s8(r1[bi + 1].qs + 16), xh1);

			int32x4_t p2a = dotprod2_s8(vld1q_s8(r2[bi].qs), xl0, vld1q_s8(r2[bi].qs + 16), xh0);
			int32x4_t p2b =
				dotprod2_s8(vld1q_s8(r2[bi + 1].qs), xl1, vld1q_s8(r2[bi + 1].qs + 16), xh1);

			int32x4_t p3a = dotprod2_s8(vld1q_s8(r3[bi].qs), xl0, vld1q_s8(r3[bi].qs + 16), xh0);
			int32x4_t p3b =
				dotprod2_s8(vld1q_s8(r3[bi + 1].qs), xl1, vld1q_s8(r3[bi + 1].qs + 16), xh1);

			int32x4_t p4a = dotprod2_s8(vld1q_s8(r4[bi].qs), xl0, vld1q_s8(r4[bi].qs + 16), xh0);
			int32x4_t p4b =
				dotprod2_s8(vld1q_s8(r4[bi + 1].qs), xl1, vld1q_s8(r4[bi + 1].qs + 16), xh1);

			int32x4_t p5a = dotprod2_s8(vld1q_s8(r5[bi].qs), xl0, vld1q_s8(r5[bi].qs + 16), xh0);
			int32x4_t p5b =
				dotprod2_s8(vld1q_s8(r5[bi + 1].qs), xl1, vld1q_s8(r5[bi + 1].qs + 16), xh1);

			int32x4_t p6a = dotprod2_s8(vld1q_s8(r6[bi].qs), xl0, vld1q_s8(r6[bi].qs + 16), xh0);
			int32x4_t p6b =
				dotprod2_s8(vld1q_s8(r6[bi + 1].qs), xl1, vld1q_s8(r6[bi + 1].qs + 16), xh1);

			int32x4_t p7a = dotprod2_s8(vld1q_s8(r7[bi].qs), xl0, vld1q_s8(r7[bi].qs + 16), xh0);
			int32x4_t p7b =
				dotprod2_s8(vld1q_s8(r7[bi + 1].qs), xl1, vld1q_s8(r7[bi + 1].qs + 16), xh1);

			const int32x4_t sum01a	 = vpaddq_s32(p0a, p1a);
			const int32x4_t sum23a	 = vpaddq_s32(p2a, p3a);
			const int32x4_t sumi_loa = vpaddq_s32(sum01a, sum23a);
			const int32x4_t sum45a	 = vpaddq_s32(p4a, p5a);
			const int32x4_t sum67a	 = vpaddq_s32(p6a, p7a);
			const int32x4_t sumi_hia = vpaddq_s32(sum45a, sum67a);

			const int32x4_t sum01b	 = vpaddq_s32(p0b, p1b);
			const int32x4_t sum23b	 = vpaddq_s32(p2b, p3b);
			const int32x4_t sumi_lob = vpaddq_s32(sum01b, sum23b);
			const int32x4_t sum45b	 = vpaddq_s32(p4b, p5b);
			const int32x4_t sum67b	 = vpaddq_s32(p6b, p7b);
			const int32x4_t sumi_hib = vpaddq_s32(sum45b, sum67b);

			uint16x4_t d_u16_loa = {(uint16_t)r0[bi].d, (uint16_t)r1[bi].d, (uint16_t)r2[bi].d,
									(uint16_t)r3[bi].d};
			uint16x4_t d_u16_hia = {(uint16_t)r4[bi].d, (uint16_t)r5[bi].d, (uint16_t)r6[bi].d,
									(uint16_t)r7[bi].d};
			uint16x4_t d_u16_lob = {(uint16_t)r0[bi + 1].d, (uint16_t)r1[bi + 1].d,
									(uint16_t)r2[bi + 1].d, (uint16_t)r3[bi + 1].d};
			uint16x4_t d_u16_hib = {(uint16_t)r4[bi + 1].d, (uint16_t)r5[bi + 1].d,
									(uint16_t)r6[bi + 1].d, (uint16_t)r7[bi + 1].d};

			float32x4_t d_f32_loa = vcvt_f32_f16(vreinterpret_f16_u16(d_u16_loa));
			float32x4_t d_f32_hia = vcvt_f32_f16(vreinterpret_f16_u16(d_u16_hia));
			float32x4_t d_f32_lob = vcvt_f32_f16(vreinterpret_f16_u16(d_u16_lob));
			float32x4_t d_f32_hib = vcvt_f32_f16(vreinterpret_f16_u16(d_u16_hib));

			float32x4_t d_xq_v0 = vdupq_n_f32(d_xq0);
			float32x4_t d_xq_v1 = vdupq_n_f32(d_xq1);
			float32x4_t dw_loa	= vmulq_f32(d_f32_loa, d_xq_v0);
			float32x4_t dw_hia	= vmulq_f32(d_f32_hia, d_xq_v0);
			float32x4_t dw_lob	= vmulq_f32(d_f32_lob, d_xq_v1);
			float32x4_t dw_hib	= vmulq_f32(d_f32_hib, d_xq_v1);

			acc_lo = vfmaq_f32(acc_lo, vcvtq_f32_s32(sumi_loa), dw_loa);
			acc_hi = vfmaq_f32(acc_hi, vcvtq_f32_s32(sumi_hia), dw_hia);
			acc_lo = vfmaq_f32(acc_lo, vcvtq_f32_s32(sumi_lob), dw_lob);
			acc_hi = vfmaq_f32(acc_hi, vcvtq_f32_s32(sumi_hib), dw_hib);
		}

		for (; bi < blocks_per_row; bi++) {
			const float		d_xq = f16_to_f32_fast(xq[bi].d);
			const int8x16_t xl	 = vld1q_s8(xq[bi].qs);
			const int8x16_t xh	 = vld1q_s8(xq[bi].qs + 16);

			if (bi + 1 < blocks_per_row) {
				__builtin_prefetch(&r0[bi + 1], 0, 1);
				__builtin_prefetch(&r1[bi + 1], 0, 1);
				__builtin_prefetch(&r2[bi + 1], 0, 1);
				__builtin_prefetch(&r3[bi + 1], 0, 1);
				__builtin_prefetch(&r4[bi + 1], 0, 1);
				__builtin_prefetch(&r5[bi + 1], 0, 1);
				__builtin_prefetch(&r6[bi + 1], 0, 1);
				__builtin_prefetch(&r7[bi + 1], 0, 1);
			}

			int32x4_t p0 = dotprod2_s8(vld1q_s8(r0[bi].qs), xl, vld1q_s8(r0[bi].qs + 16), xh);
			int32x4_t p1 = dotprod2_s8(vld1q_s8(r1[bi].qs), xl, vld1q_s8(r1[bi].qs + 16), xh);
			int32x4_t p2 = dotprod2_s8(vld1q_s8(r2[bi].qs), xl, vld1q_s8(r2[bi].qs + 16), xh);
			int32x4_t p3 = dotprod2_s8(vld1q_s8(r3[bi].qs), xl, vld1q_s8(r3[bi].qs + 16), xh);
			int32x4_t p4 = dotprod2_s8(vld1q_s8(r4[bi].qs), xl, vld1q_s8(r4[bi].qs + 16), xh);
			int32x4_t p5 = dotprod2_s8(vld1q_s8(r5[bi].qs), xl, vld1q_s8(r5[bi].qs + 16), xh);
			int32x4_t p6 = dotprod2_s8(vld1q_s8(r6[bi].qs), xl, vld1q_s8(r6[bi].qs + 16), xh);
			int32x4_t p7 = dotprod2_s8(vld1q_s8(r7[bi].qs), xl, vld1q_s8(r7[bi].qs + 16), xh);

			const int32x4_t sum01	= vpaddq_s32(p0, p1);
			const int32x4_t sum23	= vpaddq_s32(p2, p3);
			const int32x4_t sumi_lo = vpaddq_s32(sum01, sum23);
			const int32x4_t sum45	= vpaddq_s32(p4, p5);
			const int32x4_t sum67	= vpaddq_s32(p6, p7);
			const int32x4_t sumi_hi = vpaddq_s32(sum45, sum67);

			uint16x4_t d_u16_lo = {(uint16_t)r0[bi].d, (uint16_t)r1[bi].d, (uint16_t)r2[bi].d,
								   (uint16_t)r3[bi].d};
			uint16x4_t d_u16_hi = {(uint16_t)r4[bi].d, (uint16_t)r5[bi].d, (uint16_t)r6[bi].d,
								   (uint16_t)r7[bi].d};

			float32x4_t d_f32_lo = vcvt_f32_f16(vreinterpret_f16_u16(d_u16_lo));
			float32x4_t d_f32_hi = vcvt_f32_f16(vreinterpret_f16_u16(d_u16_hi));

			float32x4_t d_xq_v = vdupq_n_f32(d_xq);
			float32x4_t dw_lo  = vmulq_f32(d_f32_lo, d_xq_v);
			float32x4_t dw_hi  = vmulq_f32(d_f32_hi, d_xq_v);

			acc_lo = vfmaq_f32(acc_lo, vcvtq_f32_s32(sumi_lo), dw_lo);
			acc_hi = vfmaq_f32(acc_hi, vcvtq_f32_s32(sumi_hi), dw_hi);
		}

		y[i + 0] = vgetq_lane_f32(acc_lo, 0);
		y[i + 1] = vgetq_lane_f32(acc_lo, 1);
		y[i + 2] = vgetq_lane_f32(acc_lo, 2);
		y[i + 3] = vgetq_lane_f32(acc_lo, 3);
		y[i + 4] = vgetq_lane_f32(acc_hi, 0);
		y[i + 5] = vgetq_lane_f32(acc_hi, 1);
		y[i + 6] = vgetq_lane_f32(acc_hi, 2);
		y[i + 7] = vgetq_lane_f32(acc_hi, 3);
	}

	for (; i < n; i++) {
		const q8_0_block *row =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict qs  = row[bi].qs;
			const int8_t *restrict xq8 = xq[bi].qs;
			const int8x16_t qs_lo	   = vld1q_s8(qs);
			const int8x16_t qs_hi	   = vld1q_s8(qs + 16);
			const int8x16_t x_lo	   = vld1q_s8(xq8);
			const int8x16_t x_hi	   = vld1q_s8(xq8 + 16);

#if defined(__ARM_FEATURE_DOTPROD)
			int32x4_t dot  = vdotq_s32(vdotq_s32(vdupq_n_s32(0), qs_lo, x_lo), qs_hi, x_hi);
			int32_t	  sumi = vaddvq_s32(dot);
#else
			int16x8_t p0  = vmull_s8(vget_low_s8(qs_lo), vget_low_s8(x_lo));
			int16x8_t p1  = vmull_s8(vget_high_s8(qs_lo), vget_high_s8(x_lo));
			int16x8_t p2  = vmull_s8(vget_low_s8(qs_hi), vget_low_s8(x_hi));
			int16x8_t p3  = vmull_s8(vget_high_s8(qs_hi), vget_high_s8(x_hi));
			int32x4_t dot = vpaddlq_s16(p0);
			dot			  = vpadalq_s16(dot, p1);
			dot			  = vpadalq_s16(dot, p2);
			dot			  = vpadalq_s16(dot, p3);
			int32_t sumi  = vaddvq_s32(dot);
#endif
			const float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xq[bi].d);
			sumf += d * (float)sumi;
		}
		y[i] = sumf;
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_q8_0_q8_qonly_f32_i8mm(const void *w, const q8_0_block *restrict xq,
										  size_t	  xq_row_stride_blocks, float *restrict y,
										  int y_row_stride, int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q8_0_block);
	const q8_0_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++)
					facc[p][c] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q8_0_block)), 0,
										   1);
				}

				int8x16_t xq_lo[I8MM_NR];
				int8x16_t xq_hi[I8MM_NR];
				float	  xd[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}

				int8x16_t bvec[I8MM_NR / 2][4];
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					const int ca = 2 * cp;
					const int cb = 2 * cp + 1;
					bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb]));
					bvec[cp][1]	 = vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));
					bvec[cp][2]	 = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));
					bvec[cp][3]	 = vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));
				}

				for (int p = 0; p < MR / 2; p++) {
					const q8_0_block *b0 =
						(const q8_0_block *)(row_base[2 * p] + ((size_t)bi * sizeof(q8_0_block)));
					const q8_0_block *b1  = (const q8_0_block *)(row_base[2 * p + 1] +
																 ((size_t)bi * sizeof(q8_0_block)));
					const float		  dwa = f16_to_f32_fast(b0->d);
					const float		  dwb = f16_to_f32_fast(b1->d);

					const int8x16_t w0		= vld1q_s8(b0->qs);
					const int8x16_t w1		= vld1q_s8(b1->qs);
					const int8x16_t wh0		= vld1q_s8(b0->qs + 16);
					const int8x16_t wh1		= vld1q_s8(b1->qs + 16);
					const int8x16_t avec[4] = {
						vcombine_s8(vget_low_s8(w0), vget_low_s8(w1)),
						vcombine_s8(vget_high_s8(w0), vget_high_s8(w1)),
						vcombine_s8(vget_low_s8(wh0), vget_low_s8(wh1)),
						vcombine_s8(vget_high_s8(wh0), vget_high_s8(wh1)),
					};

					const float32x4_t srow = vcombine_f32(vdup_n_f32(dwa), vdup_n_f32(dwb));

					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol =
							vzip1q_f32(vdupq_n_f32(xd[2 * cp]), vdupq_n_f32(xd[2 * cp + 1]));
						int32x4_t s = vdupq_n_s32(0);
						s			= vmmlaq_s32(s, avec[0], bvec[cp][0]);
						s			= vmmlaq_s32(s, avec[1], bvec[cp][1]);
						s			= vmmlaq_s32(s, avec[2], bvec[cp][2]);
						s			= vmmlaq_s32(s, avec[3], bvec[cp][3]);
						facc[p][cp] =
							vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, facc[p][cp]);
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_q8_0_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q8_0_block *row =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			matmul_q8_0_q8_qonly_f32_row(row, xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}
}
#endif

#define NR 4

void matmul_q8_0_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_q8_0_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
#if defined(__ARM_FEATURE_SVE)
	matmul_q8_0_q8_qonly_f32_sve(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q8_0_block);
	const q8_0_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	  xd[NR];
				int8x16_t xq_lo[NR];
				int8x16_t xq_hi[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q8_0_block)), 0,
										   1);
				}

				for (int r = 0; r < MR; r++) {
					const q8_0_block *b =
						(const q8_0_block *)(row_base[r] + ((size_t)bi * sizeof(q8_0_block)));
					const float		d_w = f16_to_f32_fast(b->d);
					const int8x16_t lo	= vld1q_s8(b->qs);
					const int8x16_t hi	= vld1q_s8(b->qs + 16);

					const int32x4_t	  sumi4	 = q4_dot_x4_sumi4(lo, hi, xq_lo, xq_hi);
					const float32x4_t sumi_f = vcvtq_f32_s32(sumi4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(sumi_f, d_w));
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_q8_0_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q8_0_block *row =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			const q8_0_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const int8_t *restrict qs  = row[bi].qs;
				const int8_t *restrict xq8 = xrow[bi].qs;
				const int8x16_t qs_lo	   = vld1q_s8(qs);
				const int8x16_t qs_hi	   = vld1q_s8(qs + 16);
				const int8x16_t x_lo	   = vld1q_s8(xq8);
				const int8x16_t x_hi	   = vld1q_s8(xq8 + 16);

#if defined(__ARM_FEATURE_DOTPROD)
				int32x4_t dot  = vdotq_s32(vdotq_s32(vdupq_n_s32(0), qs_lo, x_lo), qs_hi, x_hi);
				int32_t	  sumi = vaddvq_s32(dot);
#else
				int16x8_t p0  = vmull_s8(vget_low_s8(qs_lo), vget_low_s8(x_lo));
				int16x8_t p1  = vmull_s8(vget_high_s8(qs_lo), vget_high_s8(x_lo));
				int16x8_t p2  = vmull_s8(vget_low_s8(qs_hi), vget_low_s8(x_hi));
				int16x8_t p3  = vmull_s8(vget_high_s8(qs_hi), vget_high_s8(x_hi));
				int32x4_t dot = vpaddlq_s16(p0);
				dot			  = vpadalq_s16(dot, p1);
				dot			  = vpadalq_s16(dot, p2);
				dot			  = vpadalq_s16(dot, p3);
				int32_t sumi  = vaddvq_s32(dot);
#endif
				float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				sumf += d * (float)sumi;
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef NR

static void matmul_q4_k_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
										   float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q4_k_block);
	const q4_k_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float32x4_t acc0 = vdupq_n_f32(0.0f);
		float32x4_t acc1 = vdupq_n_f32(0.0f);

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_k_block)), 0, 1);
			}
			const q8_k_block *restrict xb = &xq[bi];
			const float xd				  = xb->d;
			const int8_t *restrict xq8	  = xb->qs;
			const int16_t *restrict bs	  = xb->bsums;

			int32_t bsum_pre[8];
			for (int g = 0; g < 4; g++) {
				int ib				  = g * 4;
				bsum_pre[g * 2]		  = (int32_t)bs[ib] + (int32_t)bs[ib + 1];
				bsum_pre[(g * 2) + 1] = (int32_t)bs[ib + 2] + (int32_t)bs[ib + 3];
			}

			int8x16_t xq_cache[4][2][2];
			for (int g = 0; g < 4; g++) {
				const int8_t *xq0 = xq8 + (g * 64);
				const int8_t *xq1 = xq8 + (g * 64) + 32;
				for (int half = 0; half < 2; half++) {
					xq_cache[g][half][0] = vld1q_s8(xq0 + half * 16);
					xq_cache[g][half][1] = vld1q_s8(xq1 + half * 16);
				}
			}

			int32_t sumi_lane[8];
			int32_t summ_lane[8];
			float	d_w[8];
			float	dmin_w[8];

			for (int r = 0; r < MR; r++) {
				const q4_k_block *restrict b =
					(const q4_k_block *)(row_base[r] + ((size_t)bi * sizeof(q4_k_block)));
				d_w[r]						   = f16_to_f32_fast(b->d);
				dmin_w[r]					   = f16_to_f32_fast(b->dmin);
				const uint8_t *restrict qbytes = b->qs;
				const uint8_t *restrict sc	   = b->scales;

				int32x4_t sumi_vec0 = vdupq_n_s32(0);
				int32x4_t sumi_vec1 = vdupq_n_s32(0);
				int32_t	  summ		= 0;
				int		  is		= 0;

				for (int g = 0; g < 4; g++) {
					uint8_t scu8;
					uint8_t mu8;
					get_scale_min_k4(is + 0, sc, &scu8, &mu8);
					int32x4_t s0 = vdupq_n_s32((int32_t)scu8);
					int32_t	  m0 = (int32_t)mu8;
					get_scale_min_k4(is + 1, sc, &scu8, &mu8);
					int32x4_t s1 = vdupq_n_s32((int32_t)scu8);
					int32_t	  m1 = (int32_t)mu8;

					const uint8_t *restrict qg = qbytes + (g * 32);
					int32x4_t d0			   = vdupq_n_s32(0);
					int32x4_t d1			   = vdupq_n_s32(0);

					for (int half = 0; half < 2; half++) {
						uint8x16_t qg_v	 = vld1q_u8(qg + half * 16);
						uint8x16_t lo_u	 = vandq_u8(qg_v, vdupq_n_u8(0x0F));
						uint8x16_t hi_u	 = vshrq_n_u8(qg_v, 4);
						int8x16_t  xq0_v = xq_cache[g][half][0];
						int8x16_t  xq1_v = xq_cache[g][half][1];

#if defined(__ARM_FEATURE_DOTPROD)
						int8x16_t lo = vreinterpretq_s8_u8(lo_u);
						int8x16_t hi = vreinterpretq_s8_u8(hi_u);
						d0			 = vdotq_s32(d0, lo, xq0_v);
						d1			 = vdotq_s32(d1, hi, xq1_v);
#else
						int8x16_t lo	= vreinterpretq_s8_u8(lo_u);
						int8x16_t hi	= vreinterpretq_s8_u8(hi_u);
						int16x8_t p0_lo = vmull_s8(vget_low_s8(lo), vget_low_s8(xq0_v));
						int16x8_t p0_hi = vmull_s8(vget_high_s8(lo), vget_high_s8(xq0_v));
						int16x8_t p1_lo = vmull_s8(vget_low_s8(hi), vget_low_s8(xq1_v));
						int16x8_t p1_hi = vmull_s8(vget_high_s8(hi), vget_high_s8(xq1_v));
						d0				= vaddq_s32(d0, vpaddlq_s16(p0_lo));
						d0				= vaddq_s32(d0, vpaddlq_s16(p0_hi));
						d1				= vaddq_s32(d1, vpaddlq_s16(p1_lo));
						d1				= vaddq_s32(d1, vpaddlq_s16(p1_hi));
#endif
					}

					sumi_vec0 = vmlaq_s32(sumi_vec0, s0, d0);
					sumi_vec1 = vmlaq_s32(sumi_vec1, s1, d1);
					summ += m0 * bsum_pre[g * 2];
					summ += m1 * bsum_pre[(g * 2) + 1];
					is += 2;
				}

				int32x4_t sumi_total = vaddq_s32(sumi_vec0, sumi_vec1);
				sumi_lane[r]		 = vaddvq_s32(sumi_total);
				summ_lane[r]		 = summ;
			}

			int32x4_t	sumi0  = vld1q_s32(sumi_lane);
			int32x4_t	sumi1  = vld1q_s32(sumi_lane + 4);
			int32x4_t	summ0  = vld1q_s32(summ_lane);
			int32x4_t	summ1  = vld1q_s32(summ_lane + 4);
			float32x4_t sumi0f = vcvtq_f32_s32(sumi0);
			float32x4_t sumi1f = vcvtq_f32_s32(sumi1);
			float32x4_t summ0f = vcvtq_f32_s32(summ0);
			float32x4_t summ1f = vcvtq_f32_s32(summ1);
			float32x4_t d_w0   = vld1q_f32(d_w);
			float32x4_t d_w1   = vld1q_f32(d_w + 4);
			float32x4_t dm0	   = vld1q_f32(dmin_w);
			float32x4_t dm1	   = vld1q_f32(dmin_w + 4);
			float32x4_t xd_v   = vdupq_n_f32(xd);

			acc0 =
				vfmaq_f32(acc0, xd_v, vsubq_f32(vmulq_f32(d_w0, sumi0f), vmulq_f32(dm0, summ0f)));
			acc1 =
				vfmaq_f32(acc1, xd_v, vsubq_f32(vmulq_f32(d_w1, sumi1f), vmulq_f32(dm1, summ1f)));
		}

		store_acc8(acc0, acc1, y, i);
	}

	for (; i < n; i++) {
		const q4_k_block *row =
			(const q4_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q4_k_block *restrict b   = &row[bi];
			const q8_k_block *restrict xb  = &xq[bi];
			float d						   = f16_to_f32_fast(b->d);
			float dmin					   = f16_to_f32_fast(b->dmin);
			float xd					   = xb->d;
			const uint8_t *restrict qbytes = b->qs;
			const uint8_t *restrict sc	   = b->scales;
			const int8_t *restrict xq8	   = xb->qs;
			const int16_t *restrict bs	   = xb->bsums;

			int32_t sumi = 0;
			int32_t summ = 0;
			int		is	 = 0;
			int		ib	 = 0;
			uint8_t scu8;
			uint8_t mu8;
			for (int g = 0; g < 4; g++) {
				get_scale_min_k4(is + 0, sc, &scu8, &mu8);
				int s0 = scu8;
				int m0 = mu8;
				get_scale_min_k4(is + 1, sc, &scu8, &mu8);
				int s1					   = scu8;
				int m1					   = mu8;
				const uint8_t *restrict qg = qbytes + (g * 32);
				const int8_t *restrict xq0 = xq8 + (g * 64);
				const int8_t *restrict xq1 = xq8 + (g * 64) + 32;
				int32_t dot0			   = 0;
				int32_t dot1			   = 0;
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
			sumf += xd * ((d * (float)sumi) - (dmin * (float)summ));
		}
		y[i] = sumf;
	}
}

#define NR 4

void matmul_q4_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q4_k_block);
	const q4_k_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / NR) > 0 ? blocks_per_row : 0;

		static _Thread_local uint8x16_t(*wlo_cache)[MR][4][2] = NULL;
		static _Thread_local uint8x16_t(*whi_cache)[MR][4][2] = NULL;
		static _Thread_local int32x4_t(*s_lo_cache)[MR][4]	  = NULL;
		static _Thread_local int32x4_t(*s_hi_cache)[MR][4]	  = NULL;
		static _Thread_local int32_t (*m_lo_cache)[MR][4]	  = NULL;
		static _Thread_local int32_t (*m_hi_cache)[MR][4]	  = NULL;
		static _Thread_local float (*d_w_cache)[MR]			  = NULL;
		static _Thread_local float (*dmin_w_cache)[MR]		  = NULL;
		static _Thread_local int cache_cap					  = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				wlo_cache	 = realloc(wlo_cache, sizeof(*wlo_cache) * n_bi_tiles);
				whi_cache	 = realloc(whi_cache, sizeof(*whi_cache) * n_bi_tiles);
				s_lo_cache	 = realloc(s_lo_cache, sizeof(*s_lo_cache) * n_bi_tiles);
				s_hi_cache	 = realloc(s_hi_cache, sizeof(*s_hi_cache) * n_bi_tiles);
				m_lo_cache	 = realloc(m_lo_cache, sizeof(*m_lo_cache) * n_bi_tiles);
				m_hi_cache	 = realloc(m_hi_cache, sizeof(*m_hi_cache) * n_bi_tiles);
				d_w_cache	 = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				dmin_w_cache = realloc(dmin_w_cache, sizeof(*dmin_w_cache) * n_bi_tiles);
				cache_cap	 = n_bi_tiles;
				tlocal_register((void **)&wlo_cache);
				tlocal_register((void **)&whi_cache);
				tlocal_register((void **)&s_lo_cache);
				tlocal_register((void **)&s_hi_cache);
				tlocal_register((void **)&m_lo_cache);
				tlocal_register((void **)&m_hi_cache);
				tlocal_register((void **)&d_w_cache);
				tlocal_register((void **)&dmin_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_k_block)), 0,
										   1);
				}
				for (int r = 0; r < MR; r++) {
					const q4_k_block *restrict b =
						(const q4_k_block *)(row_base[r] + ((size_t)bi * sizeof(q4_k_block)));
					d_w_cache[bi][r]			   = f16_to_f32_fast(b->d);
					dmin_w_cache[bi][r]			   = f16_to_f32_fast(b->dmin);
					const uint8_t *restrict qbytes = b->qs;
					const uint8_t *restrict sc	   = b->scales;

					int is = 0;
					for (int g = 0; g < 4; g++) {
						uint8_t scu8;
						uint8_t mu8;
						get_scale_min_k4(is + 0, sc, &scu8, &mu8);
						s_lo_cache[bi][r][g] = vdupq_n_s32((int32_t)scu8);
						m_lo_cache[bi][r][g] = (int32_t)mu8;
						get_scale_min_k4(is + 1, sc, &scu8, &mu8);
						s_hi_cache[bi][r][g] = vdupq_n_s32((int32_t)scu8);
						m_hi_cache[bi][r][g] = (int32_t)mu8;
						is += 2;

						const uint8_t *restrict qg = qbytes + (g * 32);
						for (int half = 0; half < 2; half++) {
							uint8x16_t qg_v			  = vld1q_u8(qg + half * 16);
							wlo_cache[bi][r][g][half] = vandq_u8(qg_v, vdupq_n_u8(0x0F));
							whi_cache[bi][r][g][half] = vshrq_n_u8(qg_v, 4);
						}
					}
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_k_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				uint8x16_t(*wlo)[4][2] = wlo_cache[bi];
				uint8x16_t(*whi)[4][2] = whi_cache[bi];
				int32x4_t(*s_lo)[4]	   = s_lo_cache[bi];
				int32x4_t(*s_hi)[4]	   = s_hi_cache[bi];
				int32_t (*m_lo)[4]	   = m_lo_cache[bi];
				int32_t (*m_hi)[4]	   = m_hi_cache[bi];
				float *d_w			   = d_w_cache[bi];
				float *dmin_w		   = dmin_w_cache[bi];

				float	  xd_arr[NR];
				int32_t	  bsum_pre[NR][8];
				int8x16_t xq_cache[NR][4][2][2];

				for (int c = 0; c < NR; c++) {
					const q8_k_block *restrict xb = &xrow[c][bi];
					xd_arr[c]					  = xb->d;
					const int8_t *restrict xq8	  = xb->qs;
					const int16_t *restrict bs	  = xb->bsums;

					for (int g = 0; g < 4; g++) {
						int ib					 = g * 4;
						bsum_pre[c][g * 2]		 = (int32_t)bs[ib] + (int32_t)bs[ib + 1];
						bsum_pre[c][(g * 2) + 1] = (int32_t)bs[ib + 2] + (int32_t)bs[ib + 3];
					}

					for (int g = 0; g < 4; g++) {
						const int8_t *xq0 = xq8 + (g * 64);
						const int8_t *xq1 = xq8 + (g * 64) + 32;
						for (int half = 0; half < 2; half++) {
							xq_cache[c][g][half][0] = vld1q_s8(xq0 + half * 16);
							xq_cache[c][g][half][1] = vld1q_s8(xq1 + half * 16);
						}
					}
				}

				const float32x4_t xd_vec = vld1q_f32(xd_arr);

				for (int r = 0; r < MR; r++) {
					int32x4_t sumi_vec0 = vdupq_n_s32(0);
					int32x4_t sumi_vec1 = vdupq_n_s32(0);
					int32x4_t summ_vec	= vdupq_n_s32(0);

					for (int g = 0; g < 4; g++) {
						int32x4_t d0_c[NR];
						int32x4_t d1_c[NR];
						for (int c = 0; c < NR; c++) {
							d0_c[c] = vdupq_n_s32(0);
							d1_c[c] = vdupq_n_s32(0);
						}
						for (int half = 0; half < 2; half++) {
							int8x16_t wlo_s = vreinterpretq_s8_u8(wlo[r][g][half]);
							int8x16_t whi_s = vreinterpretq_s8_u8(whi[r][g][half]);
							for (int c = 0; c < NR; c++) {
#if defined(__ARM_FEATURE_DOTPROD)
								d0_c[c] = vdotq_s32(d0_c[c], wlo_s, xq_cache[c][g][half][0]);
								d1_c[c] = vdotq_s32(d1_c[c], whi_s, xq_cache[c][g][half][1]);
#else
								int8x16_t xq0_v = xq_cache[c][g][half][0];
								int8x16_t xq1_v = xq_cache[c][g][half][1];
								int16x8_t p0_lo = vmull_s8(vget_low_s8(wlo_s), vget_low_s8(xq0_v));
								int16x8_t p0_hi =
									vmull_s8(vget_high_s8(wlo_s), vget_high_s8(xq0_v));
								int16x8_t p1_lo = vmull_s8(vget_low_s8(whi_s), vget_low_s8(xq1_v));
								int16x8_t p1_hi =
									vmull_s8(vget_high_s8(whi_s), vget_high_s8(xq1_v));
								d0_c[c] = vaddq_s32(d0_c[c], vpaddlq_s16(p0_lo));
								d0_c[c] = vaddq_s32(d0_c[c], vpaddlq_s16(p0_hi));
								d1_c[c] = vaddq_s32(d1_c[c], vpaddlq_s16(p1_lo));
								d1_c[c] = vaddq_s32(d1_c[c], vpaddlq_s16(p1_hi));
#endif
							}
						}
						const int32x4_t d0_01	= vpaddq_s32(d0_c[0], d0_c[1]);
						const int32x4_t d0_23	= vpaddq_s32(d0_c[2], d0_c[3]);
						const int32x4_t d0_pack = vpaddq_s32(d0_01, d0_23);
						const int32x4_t d1_01	= vpaddq_s32(d1_c[0], d1_c[1]);
						const int32x4_t d1_23	= vpaddq_s32(d1_c[2], d1_c[3]);
						const int32x4_t d1_pack = vpaddq_s32(d1_01, d1_23);

						sumi_vec0 = vmlaq_n_s32(sumi_vec0, d0_pack, m_lo[r][g] * 0 + 1);
						sumi_vec0 = vmulq_n_s32(sumi_vec0, 1);
						sumi_vec0 = vaddq_s32(sumi_vec0, vmulq_n_s32(d0_pack, 0));
						sumi_vec0 = vaddq_s32(sumi_vec0, vshlq_n_s32(vdupq_n_s32(0), 0));
						sumi_vec0 = vmlaq_s32(vdupq_n_s32(0), d0_pack, vdupq_n_s32(1));
						(void)0;

						int32x4_t sc_lo_bcast = vdupq_n_s32(0);
						int32x4_t sc_hi_bcast = vdupq_n_s32(0);
						sc_lo_bcast = vsetq_lane_s32(vgetq_lane_s32(s_lo[r][g], 0), sc_lo_bcast, 0);
						(void)sc_lo_bcast;
						(void)sc_hi_bcast;

						int32_t s_lo_scalar = vgetq_lane_s32(s_lo[r][g], 0);
						int32_t s_hi_scalar = vgetq_lane_s32(s_hi[r][g], 0);

						sumi_vec0 = vmlaq_n_s32(vdupq_n_s32(0), d0_pack, s_lo_scalar);
						sumi_vec1 = vmlaq_n_s32(sumi_vec1, d1_pack, s_hi_scalar);

						int32_t summ_g = 0;
						summ_g += m_lo[r][g] * 0;
						(void)summ_g;

						int32x4_t bsum_lo_vec = {bsum_pre[0][g * 2], bsum_pre[1][g * 2],
												 bsum_pre[2][g * 2], bsum_pre[3][g * 2]};
						int32x4_t bsum_hi_vec = {bsum_pre[0][g * 2 + 1], bsum_pre[1][g * 2 + 1],
												 bsum_pre[2][g * 2 + 1], bsum_pre[3][g * 2 + 1]};
						summ_vec			  = vmlaq_n_s32(summ_vec, bsum_lo_vec, m_lo[r][g]);
						summ_vec			  = vmlaq_n_s32(summ_vec, bsum_hi_vec, m_hi[r][g]);
					}

					int32x4_t	sumi_total = vaddq_s32(sumi_vec0, sumi_vec1);
					float32x4_t sumi_f	   = vcvtq_f32_s32(sumi_total);
					float32x4_t summ_f	   = vcvtq_f32_s32(summ_vec);
					float32x4_t d_w_v	   = vdupq_n_f32(d_w[r]);
					float32x4_t dmin_w_v   = vdupq_n_f32(dmin_w[r]);
					float32x4_t val =
						vsubq_f32(vmulq_f32(d_w_v, sumi_f), vmulq_f32(dmin_w_v, summ_f));
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, val);
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_q4_k_q8_k_qonly_f32_row((const uint8_t *)Wb + ((size_t)i * row_stride),
										   xq + ((size_t)t * xq_row_stride_blocks),
										   y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q4_k_block *row =
			(const q4_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			const q8_k_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const q4_k_block *restrict b   = &row[bi];
				const q8_k_block *restrict xb  = &xrow[bi];
				float d						   = f16_to_f32_fast(b->d);
				float dmin					   = f16_to_f32_fast(b->dmin);
				float xd					   = xb->d;
				const uint8_t *restrict qbytes = b->qs;
				const uint8_t *restrict sc	   = b->scales;
				const int8_t *restrict xq8	   = xb->qs;
				const int16_t *restrict bs	   = xb->bsums;

				int32_t sumi = 0;
				int32_t summ = 0;
				int		is	 = 0;
				int		ib	 = 0;
				uint8_t scu8;
				uint8_t mu8;
				for (int g = 0; g < 4; g++) {
					get_scale_min_k4(is + 0, sc, &scu8, &mu8);
					int s0 = scu8;
					int m0 = mu8;
					get_scale_min_k4(is + 1, sc, &scu8, &mu8);
					int s1					   = scu8;
					int m1					   = mu8;
					const uint8_t *restrict qg = qbytes + (g * 32);
					const int8_t *restrict xq0 = xq8 + (g * 64);
					const int8_t *restrict xq1 = xq8 + (g * 64) + 32;
					int32_t dot0			   = 0;
					int32_t dot1			   = 0;
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
				sumf += xd * ((d * (float)sumi) - (dmin * (float)summ));
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef NR

static void matmul_q5_k_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
										   float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q5_k_block);
	const q5_k_block *Wb			 = w;
	int				  i				 = 0;

	const uint8x16_t mask_0F  = vdupq_n_u8(0x0F);
	const uint8x16_t mask_16  = vdupq_n_u8(16);
	const uint8x16_t zero_vec = vdupq_n_u8(0);

	for (; i + MR <= n; i += MR) {
		float32x4_t acc0 = vdupq_n_f32(0.0f);
		float32x4_t acc1 = vdupq_n_f32(0.0f);

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			int pf_bi = bi + 2;
			if (pf_bi < blocks_per_row) {
				for (int r = 0; r < MR; r++) {
					const uint8_t *pf_base = row_base[r] + ((size_t)pf_bi * sizeof(q5_k_block));
					__builtin_prefetch(pf_base, 0, 1);
					__builtin_prefetch(pf_base + 96, 0, 1);
					__builtin_prefetch(pf_base + sizeof(q5_k_block) - 16, 0, 1);
				}
			} else if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++) {
					const uint8_t *pf_base = row_base[r] + ((size_t)(bi + 1) * sizeof(q5_k_block));
					__builtin_prefetch(pf_base, 0, 1);
					__builtin_prefetch(pf_base + 96, 0, 1);
					__builtin_prefetch(pf_base + sizeof(q5_k_block) - 16, 0, 1);
				}
			}
			const q8_k_block *restrict xb = &xq[bi];
			const float xd				  = xb->d;
			const int8_t *restrict xq8	  = xb->qs;
			const int16_t *restrict bs	  = xb->bsums;

			int32_t sumi_lane[8];
			int32_t summ_lane[8];
			float	d_w[8];
			float	dmin_w[8];

			for (int r = 0; r < MR; r++) {
				const q5_k_block *restrict b =
					(const q5_k_block *)(row_base[r] + ((size_t)bi * sizeof(q5_k_block)));
				d_w[r]						   = f16_to_f32_fast(b->d);
				dmin_w[r]					   = f16_to_f32_fast(b->dmin);
				const uint8_t *restrict qbytes = b->qs;
				const uint8_t *restrict qh	   = b->qh;
				const uint8_t *restrict sc	   = b->scales;

				int32_t sumi = 0;
				int32_t summ = 0;
				int		is	 = 0;
				int		ib	 = 0;

				uint8x16_t u1_vec = vdupq_n_u8(1);
				uint8x16_t u2_vec = vdupq_n_u8(2);

				for (int g = 0; g < 4; g++) {
					uint8_t scu8;
					uint8_t mu8;
					get_scale_min_k4(is + 0, sc, &scu8, &mu8);
					int s0 = scu8;
					int m0 = mu8;
					get_scale_min_k4(is + 1, sc, &scu8, &mu8);
					int s1 = scu8;
					int m1 = mu8;

					const uint8_t *restrict qsg = qbytes + (g * 32);
					const int8_t *restrict xq0	= xq8 + (g * 64);
					const int8_t *restrict xq1	= xq8 + (g * 64) + 32;

					int32x4_t d0 = vdupq_n_s32(0);
					int32x4_t d1 = vdupq_n_s32(0);

					for (int half = 0; half < 2; half++) {
						uint8x16_t qg_v = vld1q_u8(qsg + half * 16);
						uint8x16_t qh_v = vld1q_u8(qh + half * 16);

						uint8x16_t lo_nibble = vandq_u8(qg_v, mask_0F);
						uint8x16_t hi_nibble = vshrq_n_u8(qg_v, 4);

						uint8x16_t bit0 = vandq_u8(qh_v, u1_vec);
						uint8x16_t bit1 = vandq_u8(qh_v, u2_vec);

						uint8x16_t nz0	= vceqq_u8(bit0, zero_vec);
						uint8x16_t nz1	= vceqq_u8(bit1, zero_vec);
						uint8x16_t add0 = vandq_u8(vmvnq_u8(nz0), mask_16);
						uint8x16_t add1 = vandq_u8(vmvnq_u8(nz1), mask_16);

						uint8x16_t lo_u	 = vaddq_u8(lo_nibble, add0);
						uint8x16_t hi_u	 = vaddq_u8(hi_nibble, add1);
						int8x16_t  xq0_v = vld1q_s8(xq0 + half * 16);
						int8x16_t  xq1_v = vld1q_s8(xq1 + half * 16);

#if defined(__ARM_FEATURE_DOTPROD)
						int8x16_t lo = vreinterpretq_s8_u8(lo_u);
						int8x16_t hi = vreinterpretq_s8_u8(hi_u);
						d0			 = vdotq_s32(d0, lo, xq0_v);
						d1			 = vdotq_s32(d1, hi, xq1_v);
#else
						int8x16_t lo	= vreinterpretq_s8_u8(lo_u);
						int8x16_t hi	= vreinterpretq_s8_u8(hi_u);
						int16x8_t p0_lo = vmull_s8(vget_low_s8(lo), vget_low_s8(xq0_v));
						int16x8_t p0_hi = vmull_s8(vget_high_s8(lo), vget_high_s8(xq0_v));
						d0				= vpadalq_s16(d0, p0_lo);
						d0				= vpadalq_s16(d0, p0_hi);

						int16x8_t p1_lo = vmull_s8(vget_low_s8(hi), vget_low_s8(xq1_v));
						int16x8_t p1_hi = vmull_s8(vget_high_s8(hi), vget_high_s8(xq1_v));
						d1				= vpadalq_s16(d1, p1_lo);
						d1				= vpadalq_s16(d1, p1_hi);
#endif
					}

					u1_vec = vshlq_n_u8(u1_vec, 2);
					u2_vec = vshlq_n_u8(u2_vec, 2);

					int32_t dot0 = vaddvq_s32(d0);
					int32_t dot1 = vaddvq_s32(d1);

					sumi += (s0 * dot0) + (s1 * dot1);

					int16x4_t bs_vec = vld1_s16(bs + ib);
					summ += m0 * (vget_lane_s16(bs_vec, 0) + vget_lane_s16(bs_vec, 1));
					summ += m1 * (vget_lane_s16(bs_vec, 2) + vget_lane_s16(bs_vec, 3));

					ib += 4;
					is += 2;
				}
				sumi_lane[r] = sumi;
				summ_lane[r] = summ;
			}

			int32x4_t	sumi0  = vld1q_s32(sumi_lane);
			int32x4_t	sumi1  = vld1q_s32(sumi_lane + 4);
			int32x4_t	summ0  = vld1q_s32(summ_lane);
			int32x4_t	summ1  = vld1q_s32(summ_lane + 4);
			float32x4_t sumi0f = vcvtq_f32_s32(sumi0);
			float32x4_t sumi1f = vcvtq_f32_s32(sumi1);
			float32x4_t summ0f = vcvtq_f32_s32(summ0);
			float32x4_t summ1f = vcvtq_f32_s32(summ1);
			float32x4_t d_w0   = vld1q_f32(d_w);
			float32x4_t d_w1   = vld1q_f32(d_w + 4);
			float32x4_t dm0	   = vld1q_f32(dmin_w);
			float32x4_t dm1	   = vld1q_f32(dmin_w + 4);
			float32x4_t xd_v   = vdupq_n_f32(xd);

			acc0 =
				vfmaq_f32(acc0, xd_v, vsubq_f32(vmulq_f32(d_w0, sumi0f), vmulq_f32(dm0, summ0f)));
			acc1 =
				vfmaq_f32(acc1, xd_v, vsubq_f32(vmulq_f32(d_w1, sumi1f), vmulq_f32(dm1, summ1f)));
		}

		store_acc8(acc0, acc1, y, i);
	}

	for (; i < n; i++) {
		const q5_k_block *row =
			(const q5_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q5_k_block *restrict b   = &row[bi];
			const q8_k_block *restrict xb  = &xq[bi];
			const float d				   = f16_to_f32_fast(b->d);
			const float dmin			   = f16_to_f32_fast(b->dmin);
			const float xd				   = xb->d;
			const uint8_t *restrict qbytes = b->qs;
			const uint8_t *restrict qh	   = b->qh;
			const uint8_t *restrict sc	   = b->scales;
			const int8_t *restrict xq8	   = xb->qs;
			const int16_t *restrict bs	   = xb->bsums;

			int32_t sumi = 0;
			int32_t summ = 0;
			int		is	 = 0;
			int		ib	 = 0;
			uint8_t u1	 = 1;
			uint8_t u2	 = 2;

			for (int g = 0; g < 4; g++) {
				uint8_t scu8;
				uint8_t mu8;
				get_scale_min_k4(is + 0, sc, &scu8, &mu8);
				int s0 = scu8;
				int m0 = mu8;
				get_scale_min_k4(is + 1, sc, &scu8, &mu8);
				int s1 = scu8;
				int m1 = mu8;

				const uint8_t *restrict qsg = qbytes + (g * 32);
				const int8_t *restrict xq0	= xq8 + (g * 64);
				const int8_t *restrict xq1	= xq8 + (g * 64) + 32;

				int32_t dot0 = 0;
				int32_t dot1 = 0;

				for (int l = 0; l < 32; l++) {
					uint8_t byte = qsg[l];
					uint8_t qh_l = qh[l];
					int		hi0	 = (qh_l & u1) ? 16 : 0;
					int		hi1	 = (qh_l & u2) ? 16 : 0;
					dot0 += (int32_t)((byte & 0xF) + hi0) * (int32_t)xq0[l];
					dot1 += (int32_t)((byte >> 4) + hi1) * (int32_t)xq1[l];
				}

				sumi += (s0 * dot0) + (s1 * dot1);
				summ += m0 * (int)(bs[ib] + bs[ib + 1]);
				summ += m1 * (int)(bs[ib + 2] + bs[ib + 3]);
				ib += 4;
				is += 2;
				u1 <<= 2;
				u2 <<= 2;
			}
			sumf += xd * ((d * (float)sumi) - (dmin * (float)summ));
		}
		y[i] = sumf;
	}
}

#define MATMUL_MR 8

#define NR 4

void matmul_q5_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q5_k_block);
	const q5_k_block *Wb			 = w;
	int				  i				 = 0;

	const uint8x16_t mask_0F  = vdupq_n_u8(0x0F);
	const uint8x16_t mask_16  = vdupq_n_u8(16);
	const uint8x16_t zero_vec = vdupq_n_u8(0);

	for (; i + MATMUL_MR <= n; i += MATMUL_MR) {
		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / NR) > 0 ? blocks_per_row : 0;

		static _Thread_local int8x16_t(*lo_cache)[MATMUL_MR][4][2] = NULL;
		static _Thread_local int8x16_t(*hi_cache)[MATMUL_MR][4][2] = NULL;
		static _Thread_local int32_t (*s0_cache)[MATMUL_MR][4]	   = NULL;
		static _Thread_local int32_t (*s1_cache)[MATMUL_MR][4]	   = NULL;
		static _Thread_local int32_t (*m0_cache)[MATMUL_MR][4]	   = NULL;
		static _Thread_local int32_t (*m1_cache)[MATMUL_MR][4]	   = NULL;
		static _Thread_local float (*d_cache)[MATMUL_MR]		   = NULL;
		static _Thread_local float (*dmin_cache)[MATMUL_MR]		   = NULL;
		static _Thread_local int cache_cap						   = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				lo_cache   = realloc(lo_cache, sizeof(*lo_cache) * n_bi_tiles);
				hi_cache   = realloc(hi_cache, sizeof(*hi_cache) * n_bi_tiles);
				s0_cache   = realloc(s0_cache, sizeof(*s0_cache) * n_bi_tiles);
				s1_cache   = realloc(s1_cache, sizeof(*s1_cache) * n_bi_tiles);
				m0_cache   = realloc(m0_cache, sizeof(*m0_cache) * n_bi_tiles);
				m1_cache   = realloc(m1_cache, sizeof(*m1_cache) * n_bi_tiles);
				d_cache	   = realloc(d_cache, sizeof(*d_cache) * n_bi_tiles);
				dmin_cache = realloc(dmin_cache, sizeof(*dmin_cache) * n_bi_tiles);
				cache_cap  = n_bi_tiles;
				tlocal_register((void **)&lo_cache);
				tlocal_register((void **)&hi_cache);
				tlocal_register((void **)&s0_cache);
				tlocal_register((void **)&s1_cache);
				tlocal_register((void **)&m0_cache);
				tlocal_register((void **)&m1_cache);
				tlocal_register((void **)&d_cache);
				tlocal_register((void **)&dmin_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MATMUL_MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q5_k_block)), 0,
										   1);
				}

				for (int r = 0; r < MATMUL_MR; r++) {
					const q5_k_block *restrict b =
						(const q5_k_block *)(row_base[r] + ((size_t)bi * sizeof(q5_k_block)));
					d_cache[bi][r]				   = f16_to_f32_fast(b->d);
					dmin_cache[bi][r]			   = f16_to_f32_fast(b->dmin);
					const uint8_t *restrict qbytes = b->qs;
					const uint8_t *restrict qh	   = b->qh;
					const uint8_t *restrict sc	   = b->scales;

					int		   is	  = 0;
					uint8x16_t u1_vec = vdupq_n_u8(1);
					uint8x16_t u2_vec = vdupq_n_u8(2);

					for (int g = 0; g < 4; g++) {
						uint8_t scu8;
						uint8_t mu8;
						get_scale_min_k4(is + 0, sc, &scu8, &mu8);
						s0_cache[bi][r][g] = (int32_t)scu8;
						m0_cache[bi][r][g] = (int32_t)mu8;
						get_scale_min_k4(is + 1, sc, &scu8, &mu8);
						s1_cache[bi][r][g] = (int32_t)scu8;
						m1_cache[bi][r][g] = (int32_t)mu8;
						is += 2;

						const uint8_t *restrict qsg = qbytes + (g * 32);

						for (int half = 0; half < 2; half++) {
							uint8x16_t qg_v = vld1q_u8(qsg + half * 16);
							uint8x16_t qh_v = vld1q_u8(qh + half * 16);

							uint8x16_t lo_nibble = vandq_u8(qg_v, mask_0F);
							uint8x16_t hi_nibble = vshrq_n_u8(qg_v, 4);

							uint8x16_t bit0 = vandq_u8(qh_v, u1_vec);
							uint8x16_t bit1 = vandq_u8(qh_v, u2_vec);

							uint8x16_t nz0	= vceqq_u8(bit0, zero_vec);
							uint8x16_t nz1	= vceqq_u8(bit1, zero_vec);
							uint8x16_t add0 = vandq_u8(vmvnq_u8(nz0), mask_16);
							uint8x16_t add1 = vandq_u8(vmvnq_u8(nz1), mask_16);

							uint8x16_t lo_u = vaddq_u8(lo_nibble, add0);
							uint8x16_t hi_u = vaddq_u8(hi_nibble, add1);

							lo_cache[bi][r][g][half] = vreinterpretq_s8_u8(lo_u);
							hi_cache[bi][r][g][half] = vreinterpretq_s8_u8(hi_u);
						}

						u1_vec = vshlq_n_u8(u1_vec, 2);
						u2_vec = vshlq_n_u8(u2_vec, 2);
					}
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MATMUL_MR];
			for (int r = 0; r < MATMUL_MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_k_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				int8x16_t(*lo)[4][2] = lo_cache[bi];
				int8x16_t(*hi)[4][2] = hi_cache[bi];
				int32_t (*s0)[4]	 = s0_cache[bi];
				int32_t (*s1)[4]	 = s1_cache[bi];
				int32_t (*m0)[4]	 = m0_cache[bi];
				int32_t (*m1)[4]	 = m1_cache[bi];
				float *d			 = d_cache[bi];
				float *dmin			 = dmin_cache[bi];

				float xd_arr[NR];

				int32_t	  bsum_lo_pre[NR][4];
				int32_t	  bsum_hi_pre[NR][4];
				int8x16_t xq_cache[NR][4][2][2];

				for (int c = 0; c < NR; c++) {
					const q8_k_block *restrict xb = &xrow[c][bi];
					xd_arr[c]					  = xb->d;
					const int8_t *restrict xq8	  = xb->qs;
					const int16_t *restrict bs	  = xb->bsums;

					for (int g = 0; g < 4; g++) {
						int ib			  = g * 4;
						bsum_lo_pre[c][g] = (int32_t)bs[ib] + (int32_t)bs[ib + 1];
						bsum_hi_pre[c][g] = (int32_t)bs[ib + 2] + (int32_t)bs[ib + 3];
						const int8_t *xq0 = xq8 + (g * 64);
						const int8_t *xq1 = xq8 + (g * 64) + 32;
						for (int half = 0; half < 2; half++) {
							xq_cache[c][g][half][0] = vld1q_s8(xq0 + half * 16);
							xq_cache[c][g][half][1] = vld1q_s8(xq1 + half * 16);
						}
					}
				}

				const float32x4_t xd_vec = vld1q_f32(xd_arr);

				for (int r = 0; r < MATMUL_MR; r++) {
					int32x4_t sumi_vec0 = vdupq_n_s32(0);
					int32x4_t sumi_vec1 = vdupq_n_s32(0);
					int32x4_t summ_vec	= vdupq_n_s32(0);

					for (int g = 0; g < 4; g++) {
						int32x4_t d0_c[NR];
						int32x4_t d1_c[NR];
						for (int c = 0; c < NR; c++) {
							d0_c[c] = vdupq_n_s32(0);
							d1_c[c] = vdupq_n_s32(0);
						}

						for (int half = 0; half < 2; half++) {
							int8x16_t lo_v = lo[r][g][half];
							int8x16_t hi_v = hi[r][g][half];
							for (int c = 0; c < NR; c++) {
								int8x16_t xq0_v = xq_cache[c][g][half][0];
								int8x16_t xq1_v = xq_cache[c][g][half][1];
#if defined(__ARM_FEATURE_DOTPROD)
								d0_c[c] = vdotq_s32(d0_c[c], lo_v, xq0_v);
								d1_c[c] = vdotq_s32(d1_c[c], hi_v, xq1_v);
#else
								int16x8_t p0_lo = vmull_s8(vget_low_s8(lo_v), vget_low_s8(xq0_v));
								int16x8_t p0_hi = vmull_s8(vget_high_s8(lo_v), vget_high_s8(xq0_v));
								d0_c[c]			= vaddq_s32(d0_c[c], vpaddlq_s16(p0_lo));
								d0_c[c]			= vaddq_s32(d0_c[c], vpaddlq_s16(p0_hi));

								int16x8_t p1_lo = vmull_s8(vget_low_s8(hi_v), vget_low_s8(xq1_v));
								int16x8_t p1_hi = vmull_s8(vget_high_s8(hi_v), vget_high_s8(xq1_v));
								d1_c[c]			= vaddq_s32(d1_c[c], vpaddlq_s16(p1_lo));
								d1_c[c]			= vaddq_s32(d1_c[c], vpaddlq_s16(p1_hi));
#endif
							}
						}

						const int32x4_t d0_01	 = vpaddq_s32(d0_c[0], d0_c[1]);
						const int32x4_t d0_23	 = vpaddq_s32(d0_c[2], d0_c[3]);
						const int32x4_t dot0_vec = vpaddq_s32(d0_01, d0_23);
						const int32x4_t d1_01	 = vpaddq_s32(d1_c[0], d1_c[1]);
						const int32x4_t d1_23	 = vpaddq_s32(d1_c[2], d1_c[3]);
						const int32x4_t dot1_vec = vpaddq_s32(d1_01, d1_23);

						sumi_vec0 = vmlaq_n_s32(sumi_vec0, dot0_vec, s0[r][g]);
						sumi_vec1 = vmlaq_n_s32(sumi_vec1, dot1_vec, s1[r][g]);

						int32x4_t bsum_lo_vec = {bsum_lo_pre[0][g], bsum_lo_pre[1][g],
												 bsum_lo_pre[2][g], bsum_lo_pre[3][g]};
						int32x4_t bsum_hi_vec = {bsum_hi_pre[0][g], bsum_hi_pre[1][g],
												 bsum_hi_pre[2][g], bsum_hi_pre[3][g]};
						summ_vec			  = vmlaq_n_s32(summ_vec, bsum_lo_vec, m0[r][g]);
						summ_vec			  = vmlaq_n_s32(summ_vec, bsum_hi_vec, m1[r][g]);
					}

					int32x4_t	sumi_total = vaddq_s32(sumi_vec0, sumi_vec1);
					float32x4_t sumi_f	   = vcvtq_f32_s32(sumi_total);
					float32x4_t summ_f	   = vcvtq_f32_s32(summ_vec);
					float32x4_t d_v		   = vdupq_n_f32(d[r]);
					float32x4_t dmin_v	   = vdupq_n_f32(dmin[r]);
					float32x4_t val = vsubq_f32(vmulq_f32(d_v, sumi_f), vmulq_f32(dmin_v, summ_f));
					acc_row[r]		= vfmaq_f32(acc_row[r], xd_vec, val);
				}
			}

			for (int r = 0; r < MATMUL_MR; r++) {
				float tmp[4];
				vst1q_f32(tmp, acc_row[r]);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
		}

		for (; t < m; t++) {
			matmul_q5_k_q8_k_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										   y + ((size_t)t * y_row_stride) + i, MATMUL_MR, k);
		}
	}

	for (; i < n; i++) {
		const q5_k_block *row =
			(const q5_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			matmul_q5_k_q8_k_qonly_f32_row(row, xq + ((size_t)t * xq_row_stride_blocks),
										   y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}
}

#undef NR

#undef MATMUL_MR

static void matmul_q6_k_q8_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
										 float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q6_k_block);
	const q6_k_block *Wb			 = w;
	int				  i				 = 0;

	const uint8x16_t mask_0F = vdupq_n_u8(0x0F);
	const uint8x16_t mask_03 = vdupq_n_u8(0x03);
	const uint8x16_t mask_0C = vdupq_n_u8(0x0C);
	const uint8x16_t mask_30 = vdupq_n_u8(0x30);
	const uint8x16_t mask_C0 = vdupq_n_u8(0xC0);
	const int8x16_t	 bias_32 = vdupq_n_s8(32);

	for (; i + MR <= n; i += MR) {
		float32x4_t acc0 = vdupq_n_f32(0.0f);
		float32x4_t acc1 = vdupq_n_f32(0.0f);

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			int pf_bi = bi + 2;
			if (pf_bi < blocks_per_row) {
				for (int r = 0; r < MR; r++) {
					const uint8_t *pf_base = row_base[r] + ((size_t)pf_bi * sizeof(q6_k_block));
					__builtin_prefetch(pf_base, 0, 1);
					__builtin_prefetch(pf_base + 128, 0, 1);
					__builtin_prefetch(pf_base + sizeof(q6_k_block) - 16, 0, 1);
				}
			} else if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++) {
					const uint8_t *pf_base = row_base[r] + ((size_t)(bi + 1) * sizeof(q6_k_block));
					__builtin_prefetch(pf_base, 0, 1);
					__builtin_prefetch(pf_base + 128, 0, 1);
					__builtin_prefetch(pf_base + sizeof(q6_k_block) - 16, 0, 1);
				}
			}
			const q8_k_block *restrict yb = &xq[bi];
			const float d_xq			  = yb->d;

			int32_t	 sumi_lane[8];
			uint16_t d_w_raw[8];

			for (int r = 0; r < MR; r++) {
				const q6_k_block *restrict b =
					(const q6_k_block *)(row_base[r] + ((size_t)bi * sizeof(q6_k_block)));
				d_w_raw[r] = b->d;

				const uint8_t *restrict ql = b->ql;
				const uint8_t *restrict qh = b->qh;
				const int8_t *restrict sc  = b->scales;
				const int8_t *restrict q8p = yb->qs;

				int8x16_t  wvec[16];
				int8x16_t *wp = wvec;
				for (int n_iter = 0; n_iter < 2; n_iter++) {
					uint8x16_t ql_v0 = vld1q_u8(ql);
					uint8x16_t ql_v1 = vld1q_u8(ql + 16);
					uint8x16_t ql_v2 = vld1q_u8(ql + 32);
					uint8x16_t ql_v3 = vld1q_u8(ql + 48);
					uint8x16_t qh_v0 = vld1q_u8(qh);
					uint8x16_t qh_v1 = vld1q_u8(qh + 16);

					uint8x16_t ql_lo0 = vandq_u8(ql_v0, mask_0F);
					uint8x16_t ql_lo1 = vandq_u8(ql_v1, mask_0F);
					uint8x16_t ql_lo2 = vandq_u8(ql_v2, mask_0F);
					uint8x16_t ql_lo3 = vandq_u8(ql_v3, mask_0F);
					uint8x16_t ql_hi0 = vshrq_n_u8(ql_v0, 4);
					uint8x16_t ql_hi1 = vshrq_n_u8(ql_v1, 4);
					uint8x16_t ql_hi2 = vshrq_n_u8(ql_v2, 4);
					uint8x16_t ql_hi3 = vshrq_n_u8(ql_v3, 4);

					uint8x16_t qh_sub0_0 = vshlq_n_u8(vandq_u8(qh_v0, mask_03), 4);
					uint8x16_t qh_sub0_1 = vshlq_n_u8(vandq_u8(qh_v1, mask_03), 4);
					uint8x16_t qh_sub1_0 = vshlq_n_u8(vandq_u8(qh_v0, mask_0C), 2);
					uint8x16_t qh_sub1_1 = vshlq_n_u8(vandq_u8(qh_v1, mask_0C), 2);
					uint8x16_t qh_sub2_0 = vandq_u8(qh_v0, mask_30);
					uint8x16_t qh_sub2_1 = vandq_u8(qh_v1, mask_30);
					uint8x16_t qh_sub3_0 = vshrq_n_u8(vandq_u8(qh_v0, mask_C0), 2);
					uint8x16_t qh_sub3_1 = vshrq_n_u8(vandq_u8(qh_v1, mask_C0), 2);

					wp[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(ql_lo0, qh_sub0_0)), bias_32);
					wp[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(ql_lo1, qh_sub0_1)), bias_32);
					wp[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(ql_lo2, qh_sub1_0)), bias_32);
					wp[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(ql_lo3, qh_sub1_1)), bias_32);
					wp[4] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(ql_hi0, qh_sub2_0)), bias_32);
					wp[5] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(ql_hi1, qh_sub2_1)), bias_32);
					wp[6] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(ql_hi2, qh_sub3_0)), bias_32);
					wp[7] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(ql_hi3, qh_sub3_1)), bias_32);
					wp += 8;
					ql += 64;
					qh += 32;
				}

#if defined(__ARM_FEATURE_DOTPROD)
				int32x4_t acc = vdupq_n_s32(0);
				for (int j = 0; j < 16; j++) {
					int32x4_t s_v  = vdupq_n_s32((int32_t)sc[j]);
					int8x16_t q8_v = vld1q_s8(q8p);
					int8x16_t a_v  = wvec[j];
					int32x4_t d	   = vdotq_s32(vdupq_n_s32(0), q8_v, a_v);
					acc			   = vmlaq_s32(acc, s_v, d);
					q8p += 16;
				}
				int32_t total = vaddvq_s32(acc);
#else
				int32x4_t acc0q = vdupq_n_s32(0);
				int32x4_t acc1q = vdupq_n_s32(0);
				for (int j = 0; j < 16; j++) {
					int32x4_t s_v	= vdupq_n_s32((int32_t)sc[j]);
					int8x16_t q8_v	= vld1q_s8(q8p);
					int8x16_t a_v	= wvec[j];
					int16x8_t p0	= vmull_s8(vget_low_s8(q8_v), vget_low_s8(a_v));
					int32x4_t p0_32 = vpaddlq_s16(p0);
					int16x8_t p1	= vmull_s8(vget_high_s8(q8_v), vget_high_s8(a_v));
					int32x4_t p1_32 = vpaddlq_s16(p1);
					acc0q			= vmlaq_s32(acc0q, s_v, p0_32);
					acc1q			= vmlaq_s32(acc1q, s_v, p1_32);
					q8p += 16;
				}
				int32x4_t acc	= vaddq_s32(acc0q, acc1q);
				int32_t	  total = vaddvq_s32(acc);
#endif
				sumi_lane[r] = total;
			}

			int32x4_t		 sumi0	 = vld1q_s32(sumi_lane);
			int32x4_t		 sumi1	 = vld1q_s32(sumi_lane + 4);
			float32x4_t		 sumi0f	 = vcvtq_f32_s32(sumi0);
			float32x4_t		 sumi1f	 = vcvtq_f32_s32(sumi1);
			const uint16x8_t d_w_u16 = vld1q_u16(d_w_raw);
			float32x4_t		 d_w0	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_w_u16)));
			float32x4_t		 d_w1	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_w_u16)));
			float32x4_t		 d_xq_v	 = vdupq_n_f32(d_xq);

			acc0 = vfmaq_f32(acc0, vmulq_f32(d_w0, d_xq_v), sumi0f);
			acc1 = vfmaq_f32(acc1, vmulq_f32(d_w1, d_xq_v), sumi1f);
		}

		store_acc8(acc0, acc1, y, i);
	}

	for (; i < n; i++) {
		const q6_k_block *bx = (const q6_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float			  sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q6_k_block *restrict b  = &bx[bi];
			const q8_k_block *restrict yb = &xq[bi];
			float d						  = f16_to_f32_fast(b->d) * yb->d;

			const uint8_t *restrict ql = b->ql;
			const uint8_t *restrict qh = b->qh;
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

			ap						  = a;
			const int8_t *restrict q8 = yb->qs;
			const int8_t *restrict sc = b->scales;
			int32_t acc0			  = 0;
			int32_t acc1			  = 0;
			for (int j = 0; j < 16; j++) {
				int		s  = sc[j];
				int32_t l0 = 0;
				int32_t l1 = 0;
				for (int l = 0; l < 8; l++) {
					l0 += (int)q8[l] * (int)ap[l];
					l1 += (int)q8[l + 8] * (int)ap[l + 8];
				}
				acc0 += s * l0;
				acc1 += s * l1;
				q8 += 16;
				ap += 16;
			}
			sumf += d * (float)(acc0 + acc1);
		}
		y[i] = sumf;
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_q6_k_q8_qonly_f32_i8mm(const void *w, const q8_k_block *restrict xq,
										  size_t	  xq_row_stride_blocks, float *restrict y,
										  int y_row_stride, int n, int k, int m) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q6_k_block);
	const q6_k_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / I8MM_NR) > 0 ? blocks_per_row : 0;

		static _Thread_local int8_t (*q_unpack_cache)[MR][256] = NULL;
		static _Thread_local float (*d_w_cache)[MR]			   = NULL;
		static _Thread_local int cache_cap					   = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				q_unpack_cache = realloc(q_unpack_cache, sizeof(*q_unpack_cache) * n_bi_tiles);
				d_w_cache	   = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				cache_cap	   = n_bi_tiles;
				tlocal_register((void **)&q_unpack_cache);
				tlocal_register((void **)&d_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q6_k_block)), 0,
										   1);
				}
				for (int r = 0; r < MR; r++) {
					const q6_k_block *restrict b =
						(const q6_k_block *)(row_base[r] + ((size_t)bi * sizeof(q6_k_block)));
					d_w_cache[bi][r] = f16_to_f32_fast(b->d);

					const uint8_t *restrict ql = b->ql;
					const uint8_t *restrict qh = b->qh;
					int8_t *restrict a		   = q_unpack_cache[bi][r];
					for (int n_iter = 0; n_iter < 2; n_iter++) {
						for (int l = 0; l < 32; l++) {
							a[l] = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
							a[l + 32] =
								(int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
							a[l + 64] = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
							a[l + 96] =
								(int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
						}
						a += 128;
						ql += 64;
						qh += 32;
					}
				}
			}
		}

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++)
					facc[p][c] = vdupq_n_f32(0.0f);

			const q8_k_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float d_xq_arr[I8MM_NR];
				const int8_t *restrict q8p[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					d_xq_arr[c] = xrow[c][bi].d;
					q8p[c]		= xrow[c][bi].qs;
				}

				int32x4_t iacc[MR / 2][I8MM_NR / 2];
				for (int p = 0; p < MR / 2; p++)
					for (int c = 0; c < I8MM_NR / 2; c++)
						iacc[p][c] = vdupq_n_s32(0);

				for (int j = 0; j < 16; j++) {
					int8x16_t bvec[I8MM_NR / 2][2];
					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const int8_t   *xca = q8p[2 * cp] + 16 * j;
						const int8_t   *xcb = q8p[2 * cp + 1] + 16 * j;
						const int8x16_t xl	= vld1q_s8(xca);
						const int8x16_t xh	= vld1q_s8(xcb);
						bvec[cp][0]			= vcombine_s8(vget_low_s8(xl), vget_low_s8(xh));
						bvec[cp][1]			= vcombine_s8(vget_high_s8(xl), vget_high_s8(xh));
					}

					for (int p = 0; p < MR / 2; p++) {
						const int8_t   *wa = q_unpack_cache[bi][2 * p] + 16 * j;
						const int8_t   *wb = q_unpack_cache[bi][2 * p + 1] + 16 * j;
						const int8x16_t wl = vld1q_s8(wa);
						const int8x16_t wh = vld1q_s8(wb);
						const int8x16_t a0 = vcombine_s8(vget_low_s8(wl), vget_low_s8(wh));
						const int8x16_t a1 = vcombine_s8(vget_high_s8(wl), vget_high_s8(wh));

						const q6_k_block *b0 =
							(const q6_k_block *)(row_base[2 * p] +
												 ((size_t)bi * sizeof(q6_k_block)));
						const q6_k_block *b1 =
							(const q6_k_block *)(row_base[2 * p + 1] +
												 ((size_t)bi * sizeof(q6_k_block)));
						const int32x4_t sv =
							vcombine_s32(vdup_n_s32(b0->scales[j]), vdup_n_s32(b1->scales[j]));

						for (int cp = 0; cp < I8MM_NR / 2; cp++) {
							int32x4_t tt = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), a0, bvec[cp][0]),
													  a1, bvec[cp][1]);
							iacc[p][cp]	 = vmlaq_s32(iacc[p][cp], tt, sv);
						}
					}
				}

				for (int p = 0; p < MR / 2; p++) {
					const float32x4_t srow = vcombine_f32(vdup_n_f32(d_w_cache[bi][2 * p]),
														  vdup_n_f32(d_w_cache[bi][2 * p + 1]));
					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol = vzip1q_f32(vdupq_n_f32(d_xq_arr[2 * cp]),
															vdupq_n_f32(d_xq_arr[2 * cp + 1]));
						facc[p][cp]			   = vfmaq_f32(facc[p][cp], vcvtq_f32_s32(iacc[p][cp]),
														   vmulq_f32(srow, dcol));
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, facc[p][cp]);
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_q6_k_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q6_k_block *bx = (const q6_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			matmul_q6_k_q8_qonly_f32_row(bx, xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}
}
#endif

#define NR 4

void matmul_q6_k_q8_qonly_f32(const void *w, const q8_k_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_q6_k_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q6_k_block);
	const q6_k_block *Wb			 = w;
	int				  i				 = 0;
	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / NR) > 0 ? blocks_per_row : 0;

		static _Thread_local int8_t (*q_unpack_cache)[MR][256] = NULL;
		static _Thread_local float (*d_w_cache)[MR]			   = NULL;
		static _Thread_local int cache_cap					   = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				q_unpack_cache = realloc(q_unpack_cache, sizeof(*q_unpack_cache) * n_bi_tiles);
				d_w_cache	   = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				cache_cap	   = n_bi_tiles;
				tlocal_register((void **)&q_unpack_cache);
				tlocal_register((void **)&d_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q6_k_block)), 0,
										   1);
				}
				for (int r = 0; r < MR; r++) {
					const q6_k_block *restrict b =
						(const q6_k_block *)(row_base[r] + ((size_t)bi * sizeof(q6_k_block)));
					d_w_cache[bi][r] = f16_to_f32_fast(b->d);

					const uint8_t *restrict ql = b->ql;
					const uint8_t *restrict qh = b->qh;
					int8_t *restrict a		   = q_unpack_cache[bi][r];
					for (int n_iter = 0; n_iter < 2; n_iter++) {
						for (int l = 0; l < 32; l++) {
							a[l] = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
							a[l + 32] =
								(int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
							a[l + 64] = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
							a[l + 96] =
								(int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
						}
						a += 128;
						ql += 64;
						qh += 32;
					}
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_k_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float *d_w = d_w_cache[bi];

				for (int r = 0; r < MR; r++) {
					const q6_k_block *restrict b =
						(const q6_k_block *)(row_base[r] + ((size_t)bi * sizeof(q6_k_block)));
					const int8_t *restrict sc = b->scales;
					const int8_t *restrict a  = q_unpack_cache[bi][r];

					float d_xq_arr[NR];
					const int8_t *restrict q8p[NR];
					for (int c = 0; c < NR; c++) {
						const q8_k_block *restrict yb = &xrow[c][bi];
						d_xq_arr[c]					  = yb->d;
						q8p[c]						  = yb->qs;
					}

#if defined(__ARM_FEATURE_DOTPROD)
					int32x4_t acc[NR];
					for (int c = 0; c < NR; c++)
						acc[c] = vdupq_n_s32(0);
					const int8_t *restrict aa = a;
					const int8_t *qq8[NR];
					for (int c = 0; c < NR; c++)
						qq8[c] = q8p[c];
					for (int j = 0; j < 16; j++) {
						int32x4_t s_v = vdupq_n_s32((int32_t)sc[j]);
						int8x16_t a_v = vld1q_s8(aa);
						for (int c = 0; c < NR; c++) {
							int8x16_t q8_v = vld1q_s8(qq8[c]);
							int32x4_t d	   = vdotq_s32(vdupq_n_s32(0), q8_v, a_v);
							acc[c]		   = vmlaq_s32(acc[c], s_v, d);
							qq8[c] += 16;
						}
						aa += 16;
					}
					const int32x4_t p01		  = vpaddq_s32(acc[0], acc[1]);
					const int32x4_t p23		  = vpaddq_s32(acc[2], acc[3]);
					const int32x4_t total_vec = vpaddq_s32(p01, p23);
#else
					int32x4_t acc0[NR];
					int32x4_t acc1[NR];
					for (int c = 0; c < NR; c++) {
						acc0[c] = vdupq_n_s32(0);
						acc1[c] = vdupq_n_s32(0);
					}
					const int8_t *restrict aa = a;
					const int8_t *qq8[NR];
					for (int c = 0; c < NR; c++)
						qq8[c] = q8p[c];
					for (int j = 0; j < 16; j++) {
						int32x4_t s_v = vdupq_n_s32((int32_t)sc[j]);
						int8x16_t a_v = vld1q_s8(aa);
						for (int c = 0; c < NR; c++) {
							int8x16_t q8_v	= vld1q_s8(qq8[c]);
							int16x8_t p0	= vmull_s8(vget_low_s8(q8_v), vget_low_s8(a_v));
							int32x4_t p0_32 = vpaddlq_s16(p0);
							int16x8_t p1	= vmull_s8(vget_high_s8(q8_v), vget_high_s8(a_v));
							int32x4_t p1_32 = vpaddlq_s16(p1);
							acc0[c]			= vmlaq_s32(acc0[c], s_v, p0_32);
							acc1[c]			= vmlaq_s32(acc1[c], s_v, p1_32);
							qq8[c] += 16;
						}
						aa += 16;
					}
					int32x4_t total_c[NR];
					for (int c = 0; c < NR; c++)
						total_c[c] = vaddq_s32(acc0[c], acc1[c]);
					const int32x4_t p01		  = vpaddq_s32(total_c[0], total_c[1]);
					const int32x4_t p23		  = vpaddq_s32(total_c[2], total_c[3]);
					const int32x4_t total_vec = vpaddq_s32(p01, p23);
#endif

					float32x4_t total_f = vcvtq_f32_s32(total_vec);
					float32x4_t xd_vec	= vld1q_f32(d_xq_arr);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(total_f, d_w[r]));
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_q6_k_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										 y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const q6_k_block *bx = (const q6_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			const q8_k_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const q6_k_block *restrict b  = &bx[bi];
				const q8_k_block *restrict yb = &xrow[bi];
				float d						  = f16_to_f32_fast(b->d) * yb->d;

				const uint8_t *restrict ql = b->ql;
				const uint8_t *restrict qh = b->qh;
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

				ap						  = a;
				const int8_t *restrict q8 = yb->qs;
				const int8_t *restrict sc = b->scales;
				int32_t acc0			  = 0;
				int32_t acc1			  = 0;
				for (int j = 0; j < 16; j++) {
					int		s  = sc[j];
					int32_t l0 = 0;
					int32_t l1 = 0;
					for (int l = 0; l < 8; l++) {
						l0 += (int)q8[l] * (int)ap[l];
						l1 += (int)q8[l + 8] * (int)ap[l + 8];
					}
					acc0 += s * l0;
					acc1 += s * l1;
					q8 += 16;
					ap += 16;
				}
				sumf += d * (float)(acc0 + acc1);
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef NR

void dequant_iq4_nl_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq4_nl_block *b		  = blocks;
	const uint8x16_t	kvalues_u = vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t	lo_mask	  = vdupq_n_u8(0x0F);

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const float		  d		 = f16_to_f32_fast(b[bi].d);
		const float32x4_t d_vec	 = vdupq_n_f32(d);
		const uint8x16_t  q		 = vld1q_u8(b[bi].qs);
		const uint8x16_t  lo_idx = vandq_u8(q, lo_mask);
		const uint8x16_t  hi_idx = vshrq_n_u8(q, 4);
		const int8x16_t	  lo	 = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, lo_idx));
		const int8x16_t	  hi	 = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, hi_idx));

		float *dst_lo = dst + (bi * 32);
		float *dst_hi = dst_lo + 16;

		int16x8_t lo_s16_0 = vmovl_s8(vget_low_s8(lo));
		int16x8_t lo_s16_1 = vmovl_s8(vget_high_s8(lo));
		int16x8_t hi_s16_0 = vmovl_s8(vget_low_s8(hi));
		int16x8_t hi_s16_1 = vmovl_s8(vget_high_s8(hi));

		vst1q_f32(dst_lo + 0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16_0))), d_vec));
		vst1q_f32(dst_lo + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_s16_0))), d_vec));
		vst1q_f32(dst_lo + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16_1))), d_vec));
		vst1q_f32(dst_lo + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_s16_1))), d_vec));

		vst1q_f32(dst_hi + 0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16_0))), d_vec));
		vst1q_f32(dst_hi + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_s16_0))), d_vec));
		vst1q_f32(dst_hi + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16_1))), d_vec));
		vst1q_f32(dst_hi + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_s16_1))), d_vec));
	}
}

static void matmul_iq4_nl_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
										   float *restrict y, int n, int k) {
	const int			blocks_per_row = k / 32;
	const size_t		row_stride	   = (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const iq4_nl_block *Wb			   = w;
	const uint8x16_t	kvalues_u	   = vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t	lo_mask		   = vdupq_n_u8(0x0F);

	float	   xq_d_stack[256];
	int8x16_t  xq_lo_stack[256];
	int8x16_t  xq_hi_stack[256];
	float	  *xq_d	   = xq_d_stack;
	int8x16_t *xq_lo   = xq_lo_stack;
	int8x16_t *xq_hi   = xq_hi_stack;
	void	  *xq_heap = NULL;

	if (blocks_per_row > 256) {
		xq_heap = xmalloc((size_t)blocks_per_row * (sizeof(float) + (2 * sizeof(int8x16_t))));
		xq_d	= (float *)xq_heap;
		xq_lo	= (int8x16_t *)(xq_d + blocks_per_row);
		xq_hi	= xq_lo + blocks_per_row;
	}

	for (int bi = 0; bi < blocks_per_row; bi++) {
		xq_d[bi]  = f16_to_f32_fast(xq[bi].d);
		xq_lo[bi] = vld1q_s8(xq[bi].qs);
		xq_hi[bi] = vld1q_s8(xq[bi].qs + 16);
	}

	int i = 0;

	for (; i + MR <= n; i += MR) {
		float32x4_t			acc_lo = vdupq_n_f32(0.0f);
		float32x4_t			acc_hi = vdupq_n_f32(0.0f);
		const iq4_nl_block *row_ptrs[MR];

		for (int r = 0; r < MR; r++)
			row_ptrs[r] =
				(const iq4_nl_block *)((const uint8_t *)Wb + ((size_t)(i + r) * row_stride));

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const float		d_xq	  = xq_d[bi];
			const int8x16_t xq_lo_vec = xq_lo[bi];
			const int8x16_t xq_hi_vec = xq_hi[bi];

			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(&row_ptrs[r][bi + 1], 0, 1);
			}

			int r = 0;
			for (; r + 3 < MR; r += 4) {
				const uint8x16_t q0 = vld1q_u8(row_ptrs[r + 0][bi].qs);
				const uint8x16_t q1 = vld1q_u8(row_ptrs[r + 1][bi].qs);
				const uint8x16_t q2 = vld1q_u8(row_ptrs[r + 2][bi].qs);
				const uint8x16_t q3 = vld1q_u8(row_ptrs[r + 3][bi].qs);

				const int8x16_t lo0 =
					vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vandq_u8(q0, lo_mask)));
				const int8x16_t hi0 = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vshrq_n_u8(q0, 4)));
				const int8x16_t lo1 =
					vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vandq_u8(q1, lo_mask)));
				const int8x16_t hi1 = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vshrq_n_u8(q1, 4)));
				const int8x16_t lo2 =
					vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vandq_u8(q2, lo_mask)));
				const int8x16_t hi2 = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vshrq_n_u8(q2, 4)));
				const int8x16_t lo3 =
					vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vandq_u8(q3, lo_mask)));
				const int8x16_t hi3 = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vshrq_n_u8(q3, 4)));

				const int32x4_t acc0  = q5_dot(lo0, hi0, xq_lo_vec, xq_hi_vec);
				const int32x4_t acc1  = q5_dot(lo1, hi1, xq_lo_vec, xq_hi_vec);
				const int32x4_t acc2  = q5_dot(lo2, hi2, xq_lo_vec, xq_hi_vec);
				const int32x4_t acc3  = q5_dot(lo3, hi3, xq_lo_vec, xq_hi_vec);
				const int32x4_t sum01 = vpaddq_s32(acc0, acc1);
				const int32x4_t sum23 = vpaddq_s32(acc2, acc3);
				const int32x4_t sumi4 = vpaddq_s32(sum01, sum23);

				const float d_w0 = f16_to_f32_fast(row_ptrs[r + 0][bi].d);
				const float d_w1 = f16_to_f32_fast(row_ptrs[r + 1][bi].d);
				const float d_w2 = f16_to_f32_fast(row_ptrs[r + 2][bi].d);
				const float d_w3 = f16_to_f32_fast(row_ptrs[r + 3][bi].d);

				const float32x4_t d_w_vec = {d_w0, d_w1, d_w2, d_w3};
				const float32x4_t sumi_f  = vcvtq_f32_s32(sumi4);
				const float32x4_t scaled  = vmulq_n_f32(d_w_vec, d_xq);

				if (r == 0)
					acc_lo = vfmaq_f32(acc_lo, sumi_f, scaled);
				else
					acc_hi = vfmaq_f32(acc_hi, sumi_f, scaled);
			}
		}

		float tmp_lo[4];
		float tmp_hi[4];
		vst1q_f32(tmp_lo, acc_lo);
		vst1q_f32(tmp_hi, acc_hi);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp_lo[r];
			y[i + 4 + r] = tmp_hi[r];
		}
	}

	for (; i < n; i++) {
		const iq4_nl_block *row =
			(const iq4_nl_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const float		 d_xq	   = xq_d[bi];
			const int8x16_t	 xq_lo_vec = xq_lo[bi];
			const int8x16_t	 xq_hi_vec = xq_hi[bi];
			const uint8x16_t q		   = vld1q_u8(row[bi].qs);
			const uint8x16_t lo_idx	   = vandq_u8(q, lo_mask);
			const uint8x16_t hi_idx	   = vshrq_n_u8(q, 4);
			const int8x16_t	 lo		   = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, lo_idx));
			const int8x16_t	 hi		   = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, hi_idx));

#if defined(__ARM_FEATURE_DOTPROD)
			int32x4_t acc	   = vdotq_s32(vdupq_n_s32(0), lo, xq_lo_vec);
			acc				   = vdotq_s32(acc, hi, xq_hi_vec);
			const int32_t sumi = vaddvq_s32(acc);
#else
			int16x8_t p0	   = vmull_s8(vget_low_s8(lo), vget_low_s8(xq_lo_vec));
			int16x8_t p1	   = vmull_s8(vget_high_s8(lo), vget_high_s8(xq_lo_vec));
			int16x8_t p2	   = vmull_s8(vget_low_s8(hi), vget_low_s8(xq_hi_vec));
			int16x8_t p3	   = vmull_s8(vget_high_s8(hi), vget_high_s8(xq_hi_vec));
			int32x4_t acc	   = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
			acc				   = vaddq_s32(acc, vpaddlq_s16(p2));
			acc				   = vaddq_s32(acc, vpaddlq_s16(p3));
			const int32_t sumi = vaddvq_s32(acc);
#endif
			const float d = f16_to_f32_fast(row[bi].d) * d_xq;
			sumf += d * (float)sumi;
		}
		y[i] = sumf;
	}

	free(xq_heap);
}

#define NR 4

#if defined(__ARM_FEATURE_SVE)

#define SVE_MAX_B 16

#define Q4_PROC_SINGLE_8(XROW, ACC_LO, ACC_HI)                                                     \
	do {                                                                                           \
		const q8_0_block *xb	= &XROW[bi];                                                       \
		const int32_t	  x_sum = sve_sum_i8(xb->qs, 32);                                          \
		const int32_t	  off	= 8 * x_sum;                                                       \
		const float		  d_x	= f16_to_f32_fast(xb->d);                                          \
		int32_t			  dots_lo[4], dots_hi[4];                                                  \
		for (int r = 0; r < 4; r++) {                                                              \
			const q4_0_block *wb =                                                                 \
				(const q4_0_block *)(row_base[r] + ((size_t)bi * sizeof(q4_0_block)));             \
			dots_lo[r] = sve_q4_q8_block_dot_raw(wb, xb) - off;                                    \
		}                                                                                          \
		for (int r = 0; r < 4; r++) {                                                              \
			const q4_0_block *wb =                                                                 \
				(const q4_0_block *)(row_base[r + 4] + ((size_t)bi * sizeof(q4_0_block)));         \
			dots_hi[r] = sve_q4_q8_block_dot_raw(wb, xb) - off;                                    \
		}                                                                                          \
		svfloat32_t dotf_lo = svcvt_f32_s32_x(pg, svld1_s32(pg, dots_lo));                         \
		svfloat32_t dotf_hi = svcvt_f32_s32_x(pg, svld1_s32(pg, dots_hi));                         \
		svfloat32_t dxdw_lo = svmul_f32_x(pg, dw_v, svdup_n_f32(d_x));                             \
		svfloat32_t dxdw_hi = svmul_f32_x(pg, dw_v_hi, svdup_n_f32(d_x));                          \
		ACC_LO				= svmla_f32_x(pg, ACC_LO, dotf_lo, dxdw_lo);                           \
		ACC_HI				= svmla_f32_x(pg, ACC_HI, dotf_hi, dxdw_hi);                           \
	} while (0)

static void matmul_q4_q8_qonly_f32_sve(const void *w, const q8_0_block *restrict xq,
									   size_t	   xq_row_stride_blocks, float *restrict y,
									   int y_row_stride, int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_0_block);
	const q4_0_block *Wb			 = w;
	svbool_t		  pg			 = svptrue_b32();
	int				  i				 = 0;

	for (; i + 8 <= n; i += 8) {
		const uint8_t *row_base[8];
		for (int r = 0; r < 8; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + 4 <= m; t += 4) {
			const q8_0_block *xrow0 = xq + ((size_t)(t + 0) * xq_row_stride_blocks);
			const q8_0_block *xrow1 = xq + ((size_t)(t + 1) * xq_row_stride_blocks);
			const q8_0_block *xrow2 = xq + ((size_t)(t + 2) * xq_row_stride_blocks);
			const q8_0_block *xrow3 = xq + ((size_t)(t + 3) * xq_row_stride_blocks);

			svfloat32_t acc0_lo = svdup_n_f32(0.0f), acc0_hi = svdup_n_f32(0.0f);
			svfloat32_t acc1_lo = svdup_n_f32(0.0f), acc1_hi = svdup_n_f32(0.0f);
			svfloat32_t acc2_lo = svdup_n_f32(0.0f), acc2_hi = svdup_n_f32(0.0f);
			svfloat32_t acc3_lo = svdup_n_f32(0.0f), acc3_hi = svdup_n_f32(0.0f);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				if (bi + 1 < blocks_per_row) {
					__builtin_prefetch(row_base[0] + ((size_t)(bi + 1) * sizeof(q4_0_block)), 0, 1);
					__builtin_prefetch(row_base[7] + ((size_t)(bi + 1) * sizeof(q4_0_block)), 0, 1);
				}
				float dw_data[4];
				for (int r = 0; r < 4; r++) {
					const q4_0_block *wb =
						(const q4_0_block *)(row_base[r] + ((size_t)bi * sizeof(q4_0_block)));
					dw_data[r] = f16_to_f32_fast(wb->d);
				}
				svfloat32_t dw_v = svld1_f32(pg, dw_data);
				float		dw_data_hi[4];
				for (int r = 0; r < 4; r++) {
					const q4_0_block *wb =
						(const q4_0_block *)(row_base[r + 4] + ((size_t)bi * sizeof(q4_0_block)));
					dw_data_hi[r] = f16_to_f32_fast(wb->d);
				}
				svfloat32_t dw_v_hi = svld1_f32(pg, dw_data_hi);
				Q4_PROC_SINGLE_8(xrow0, acc0_lo, acc0_hi);
				Q4_PROC_SINGLE_8(xrow1, acc1_lo, acc1_hi);
				Q4_PROC_SINGLE_8(xrow2, acc2_lo, acc2_hi);
				Q4_PROC_SINGLE_8(xrow3, acc3_lo, acc3_hi);
			}
			svst1_f32(pg, y + ((size_t)(t + 0) * y_row_stride) + i, acc0_lo);
			svst1_f32(pg, y + ((size_t)(t + 0) * y_row_stride) + i + 4, acc0_hi);
			svst1_f32(pg, y + ((size_t)(t + 1) * y_row_stride) + i, acc1_lo);
			svst1_f32(pg, y + ((size_t)(t + 1) * y_row_stride) + i + 4, acc1_hi);
			svst1_f32(pg, y + ((size_t)(t + 2) * y_row_stride) + i, acc2_lo);
			svst1_f32(pg, y + ((size_t)(t + 2) * y_row_stride) + i + 4, acc2_hi);
			svst1_f32(pg, y + ((size_t)(t + 3) * y_row_stride) + i, acc3_lo);
			svst1_f32(pg, y + ((size_t)(t + 3) * y_row_stride) + i + 4, acc3_hi);
		}
		for (; t < m; t++) {
			const q8_0_block *restrict xrow = xq + ((size_t)t * xq_row_stride_blocks);
			svfloat32_t acc_lo = svdup_n_f32(0.0f), acc_hi = svdup_n_f32(0.0f);
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const q8_0_block *restrict xb = &xrow[bi];
				const int32_t x_sum			  = sve_sum_i8(xb->qs, 32);
				const int32_t off			  = 8 * x_sum;
				const float	  d_x			  = f16_to_f32_fast(xb->d);
				float		  dw_data[4];
				int32_t		  dots_lo[4];
				for (int r = 0; r < 4; r++) {
					const q4_0_block *restrict wb =
						(const q4_0_block *)(row_base[r] + ((size_t)bi * sizeof(q4_0_block)));
					dw_data[r] = f16_to_f32_fast(wb->d);
					dots_lo[r] = sve_q4_q8_block_dot_raw(wb, xb) - off;
				}
				float	dw_data_hi[4];
				int32_t dots_hi[4];
				for (int r = 0; r < 4; r++) {
					const q4_0_block *restrict wb =
						(const q4_0_block *)(row_base[r + 4] + ((size_t)bi * sizeof(q4_0_block)));
					dw_data_hi[r] = f16_to_f32_fast(wb->d);
					dots_hi[r]	  = sve_q4_q8_block_dot_raw(wb, xb) - off;
				}
				svfloat32_t dotf_lo = svcvt_f32_s32_x(pg, svld1_s32(pg, dots_lo));
				svfloat32_t dotf_hi = svcvt_f32_s32_x(pg, svld1_s32(pg, dots_hi));
				svfloat32_t dwv_lo	= svld1_f32(pg, dw_data);
				svfloat32_t dwv_hi	= svld1_f32(pg, dw_data_hi);
				svfloat32_t dxdw_lo = svmul_f32_x(pg, dwv_lo, svdup_n_f32(d_x));
				svfloat32_t dxdw_hi = svmul_f32_x(pg, dwv_hi, svdup_n_f32(d_x));
				acc_lo				= svmla_f32_x(pg, acc_lo, dotf_lo, dxdw_lo);
				acc_hi				= svmla_f32_x(pg, acc_hi, dotf_hi, dxdw_hi);
			}
			svst1_f32(pg, y + ((size_t)t * y_row_stride) + i, acc_lo);
			svst1_f32(pg, y + ((size_t)t * y_row_stride) + i + 4, acc_hi);
		}
	}
	for (; i < n; i++) {
		const q4_0_block *row =
			(const q4_0_block *)(((const uint8_t *)Wb) + (size_t)i * row_stride);
		for (int t = 0; t < m; t++) {
			const q8_0_block *restrict xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float sumf						= 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				sumf += d * (float)sve_q4_q8_block_dot(&row[bi], &xrow[bi]);
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef Q4_PROC_SINGLE_8

#define Q8_0_PROC_SINGLE_8(XROW, ACC_LO, ACC_HI)                                                   \
	do {                                                                                           \
		const q8_0_block *xb  = &XROW[bi];                                                         \
		const float		  d_x = f16_to_f32_fast(xb->d);                                            \
		int32_t			  dots_lo[4], dots_hi[4];                                                  \
		float			  dw_lo[4], dw_hi[4];                                                      \
		for (int r = 0; r < 4; r++) {                                                              \
			const q8_0_block *wb =                                                                 \
				(const q8_0_block *)(row_base[r] + ((size_t)bi * sizeof(q8_0_block)));             \
			dw_lo[r]   = f16_to_f32_fast(wb->d);                                                   \
			dots_lo[r] = sve_q8_0_block_dot(wb, xb);                                               \
		}                                                                                          \
		for (int r = 0; r < 4; r++) {                                                              \
			const q8_0_block *wb =                                                                 \
				(const q8_0_block *)(row_base[r + 4] + ((size_t)bi * sizeof(q8_0_block)));         \
			dw_hi[r]   = f16_to_f32_fast(wb->d);                                                   \
			dots_hi[r] = sve_q8_0_block_dot(wb, xb);                                               \
		}                                                                                          \
		svfloat32_t dotf_lo = svcvt_f32_s32_x(pg, svld1_s32(pg, dots_lo));                         \
		svfloat32_t dotf_hi = svcvt_f32_s32_x(pg, svld1_s32(pg, dots_hi));                         \
		svfloat32_t dwv_lo	= svld1_f32(pg, dw_lo);                                                \
		svfloat32_t dwv_hi	= svld1_f32(pg, dw_hi);                                                \
		svfloat32_t dxdw_lo = svmul_f32_x(pg, dwv_lo, svdup_n_f32(d_x));                           \
		svfloat32_t dxdw_hi = svmul_f32_x(pg, dwv_hi, svdup_n_f32(d_x));                           \
		ACC_LO				= svmla_f32_x(pg, ACC_LO, dotf_lo, dxdw_lo);                           \
		ACC_HI				= svmla_f32_x(pg, ACC_HI, dotf_hi, dxdw_hi);                           \
	} while (0)

static void matmul_q8_0_q8_qonly_f32_sve(const void *w, const q8_0_block *restrict xq,
										 size_t		 xq_row_stride_blocks, float *restrict y,
										 int y_row_stride, int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q8_0_block);
	const q8_0_block *Wb			 = w;
	svbool_t		  pg			 = svptrue_b32();
	int				  i				 = 0;

	for (; i + 8 <= n; i += 8) {
		const uint8_t *row_base[8];
		for (int r = 0; r < 8; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + 4 <= m; t += 4) {
			const q8_0_block *xrow0 = xq + ((size_t)(t + 0) * xq_row_stride_blocks);
			const q8_0_block *xrow1 = xq + ((size_t)(t + 1) * xq_row_stride_blocks);
			const q8_0_block *xrow2 = xq + ((size_t)(t + 2) * xq_row_stride_blocks);
			const q8_0_block *xrow3 = xq + ((size_t)(t + 3) * xq_row_stride_blocks);

			svfloat32_t acc0_lo = svdup_n_f32(0.0f), acc0_hi = svdup_n_f32(0.0f);
			svfloat32_t acc1_lo = svdup_n_f32(0.0f), acc1_hi = svdup_n_f32(0.0f);
			svfloat32_t acc2_lo = svdup_n_f32(0.0f), acc2_hi = svdup_n_f32(0.0f);
			svfloat32_t acc3_lo = svdup_n_f32(0.0f), acc3_hi = svdup_n_f32(0.0f);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				if (bi + 1 < blocks_per_row) {
					__builtin_prefetch(row_base[0] + ((size_t)(bi + 1) * sizeof(q8_0_block)), 0, 1);
					__builtin_prefetch(row_base[7] + ((size_t)(bi + 1) * sizeof(q8_0_block)), 0, 1);
				}
				Q8_0_PROC_SINGLE_8(xrow0, acc0_lo, acc0_hi);
				Q8_0_PROC_SINGLE_8(xrow1, acc1_lo, acc1_hi);
				Q8_0_PROC_SINGLE_8(xrow2, acc2_lo, acc2_hi);
				Q8_0_PROC_SINGLE_8(xrow3, acc3_lo, acc3_hi);
			}
			svst1_f32(pg, y + ((size_t)(t + 0) * y_row_stride) + i, acc0_lo);
			svst1_f32(pg, y + ((size_t)(t + 0) * y_row_stride) + i + 4, acc0_hi);
			svst1_f32(pg, y + ((size_t)(t + 1) * y_row_stride) + i, acc1_lo);
			svst1_f32(pg, y + ((size_t)(t + 1) * y_row_stride) + i + 4, acc1_hi);
			svst1_f32(pg, y + ((size_t)(t + 2) * y_row_stride) + i, acc2_lo);
			svst1_f32(pg, y + ((size_t)(t + 2) * y_row_stride) + i + 4, acc2_hi);
			svst1_f32(pg, y + ((size_t)(t + 3) * y_row_stride) + i, acc3_lo);
			svst1_f32(pg, y + ((size_t)(t + 3) * y_row_stride) + i + 4, acc3_hi);
		}
		for (; t < m; t++) {
			const q8_0_block *restrict xrow = xq + ((size_t)t * xq_row_stride_blocks);
			svfloat32_t acc_lo = svdup_n_f32(0.0f), acc_hi = svdup_n_f32(0.0f);
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const q8_0_block *xrow_single = xrow;
				Q8_0_PROC_SINGLE_8(xrow_single, acc_lo, acc_hi);
			}
			svst1_f32(pg, y + ((size_t)t * y_row_stride) + i, acc_lo);
			svst1_f32(pg, y + ((size_t)t * y_row_stride) + i + 4, acc_hi);
		}
	}
	for (; i < n; i++) {
		const q8_0_block *row =
			(const q8_0_block *)(((const uint8_t *)Wb) + (size_t)i * row_stride);
		for (int t = 0; t < m; t++) {
			const q8_0_block *restrict xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float sumf						= 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				sumf += d * (float)sve_q8_0_block_dot(&row[bi], &xrow[bi]);
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef Q8_0_PROC_SINGLE_8

#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_iq4_nl_q8_qonly_f32_i8mm(const void *w, const q8_0_block *restrict xq,
											size_t		xq_row_stride_blocks, float *restrict y,
											int y_row_stride, int n, int k, int m) {
	const int			blocks_per_row = k / 32;
	const size_t		row_stride	   = (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const iq4_nl_block *Wb			   = w;
	const uint8x16_t	kvalues_u	   = vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t	lo_mask		   = vdupq_n_u8(0x0F);
	int					i			   = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / I8MM_NR) > 0 ? blocks_per_row : 0;

		static _Thread_local int8x16_t(*lo_cache)[MR] = NULL;
		static _Thread_local int8x16_t(*hi_cache)[MR] = NULL;
		static _Thread_local float (*d_w_cache)[MR]	  = NULL;
		static _Thread_local int cache_cap			  = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				lo_cache  = realloc(lo_cache, sizeof(*lo_cache) * n_bi_tiles);
				hi_cache  = realloc(hi_cache, sizeof(*hi_cache) * n_bi_tiles);
				d_w_cache = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				cache_cap = n_bi_tiles;
				tlocal_register((void **)&lo_cache);
				tlocal_register((void **)&hi_cache);
				tlocal_register((void **)&d_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(iq4_nl_block)),
										   0, 1);
				}
				for (int r = 0; r < MR; r++) {
					const iq4_nl_block *b =
						(const iq4_nl_block *)(row_base[r] + ((size_t)bi * sizeof(iq4_nl_block)));
					d_w_cache[bi][r]		= f16_to_f32_fast(b->d);
					const uint8x16_t q		= vld1q_u8(b->qs);
					const uint8x16_t lo_idx = vandq_u8(q, lo_mask);
					const uint8x16_t hi_idx = vshrq_n_u8(q, 4);
					lo_cache[bi][r]			= vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, lo_idx));
					hi_cache[bi][r]			= vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, hi_idx));
				}
			}
		}

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++)
					facc[p][c] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				int8x16_t xq_lo[I8MM_NR];
				int8x16_t xq_hi[I8MM_NR];
				float	  xd[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}

				int8x16_t bvec[I8MM_NR / 2][4];
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					const int ca = 2 * cp;
					const int cb = 2 * cp + 1;
					bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb]));
					bvec[cp][1]	 = vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));
					bvec[cp][2]	 = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));
					bvec[cp][3]	 = vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));
				}

				for (int p = 0; p < MR / 2; p++) {
					const int8x16_t l0		= lo_cache[bi][2 * p];
					const int8x16_t l1		= lo_cache[bi][2 * p + 1];
					const int8x16_t h0		= hi_cache[bi][2 * p];
					const int8x16_t h1		= hi_cache[bi][2 * p + 1];
					const int8x16_t avec[4] = {
						vcombine_s8(vget_low_s8(l0), vget_low_s8(l1)),
						vcombine_s8(vget_high_s8(l0), vget_high_s8(l1)),
						vcombine_s8(vget_low_s8(h0), vget_low_s8(h1)),
						vcombine_s8(vget_high_s8(h0), vget_high_s8(h1)),
					};

					const float32x4_t srow = vcombine_f32(vdup_n_f32(d_w_cache[bi][2 * p]),
														  vdup_n_f32(d_w_cache[bi][2 * p + 1]));

					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol =
							vzip1q_f32(vdupq_n_f32(xd[2 * cp]), vdupq_n_f32(xd[2 * cp + 1]));
						int32x4_t s = vdupq_n_s32(0);
						s			= vmmlaq_s32(s, avec[0], bvec[cp][0]);
						s			= vmmlaq_s32(s, avec[1], bvec[cp][1]);
						s			= vmmlaq_s32(s, avec[2], bvec[cp][2]);
						s			= vmmlaq_s32(s, avec[3], bvec[cp][3]);
						facc[p][cp] =
							vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, facc[p][cp]);
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_iq4_nl_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										   y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const iq4_nl_block *row =
			(const iq4_nl_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			matmul_iq4_nl_q8_qonly_f32_row(row, xq + ((size_t)t * xq_row_stride_blocks),
										   y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}
}
#endif

void matmul_iq4_nl_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_iq4_nl_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int			blocks_per_row = k / 32;
	const size_t		row_stride	   = (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const iq4_nl_block *Wb			   = w;
	const uint8x16_t	kvalues_u	   = vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t	lo_mask		   = vdupq_n_u8(0x0F);
	int					i			   = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / NR) > 0 ? blocks_per_row : 0;

		static _Thread_local int8x16_t(*lo_cache)[MR] = NULL;
		static _Thread_local int8x16_t(*hi_cache)[MR] = NULL;
		static _Thread_local float (*d_w_cache)[MR]	  = NULL;
		static _Thread_local int cache_cap			  = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				lo_cache  = realloc(lo_cache, sizeof(*lo_cache) * n_bi_tiles);
				hi_cache  = realloc(hi_cache, sizeof(*hi_cache) * n_bi_tiles);
				d_w_cache = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				cache_cap = n_bi_tiles;
				tlocal_register((void **)&lo_cache);
				tlocal_register((void **)&hi_cache);
				tlocal_register((void **)&d_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(iq4_nl_block)),
										   0, 1);
				}
				for (int r = 0; r < MR; r++) {
					const iq4_nl_block *b =
						(const iq4_nl_block *)(row_base[r] + ((size_t)bi * sizeof(iq4_nl_block)));
					d_w_cache[bi][r]		= f16_to_f32_fast(b->d);
					const uint8x16_t q		= vld1q_u8(b->qs);
					const uint8x16_t lo_idx = vandq_u8(q, lo_mask);
					const uint8x16_t hi_idx = vshrq_n_u8(q, 4);
					lo_cache[bi][r]			= vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, lo_idx));
					hi_cache[bi][r]			= vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, hi_idx));
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	  xd[NR];
				int8x16_t xq_lo[NR];
				int8x16_t xq_hi[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				for (int r = 0; r < MR; r++) {
					const float		d_w = d_w_cache[bi][r];
					const int8x16_t lo	= lo_cache[bi][r];
					const int8x16_t hi	= hi_cache[bi][r];

					const int32x4_t sumi4 = q4_dot_x4_sumi4(lo, hi, xq_lo, xq_hi);

					const float32x4_t sumi_f = vcvtq_f32_s32(sumi4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(sumi_f, d_w));
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_iq4_nl_q8_qonly_f32_row(row_base[0], xq + ((size_t)t * xq_row_stride_blocks),
										   y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const iq4_nl_block *row =
			(const iq4_nl_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			const q8_0_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8x16_t q		   = vld1q_u8(row[bi].qs);
				const uint8x16_t lo_idx	   = vandq_u8(q, lo_mask);
				const uint8x16_t hi_idx	   = vshrq_n_u8(q, 4);
				const int8x16_t	 lo		   = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, lo_idx));
				const int8x16_t	 hi		   = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, hi_idx));
				const int8x16_t	 xq_lo_vec = vld1q_s8(xrow[bi].qs);
				const int8x16_t	 xq_hi_vec = vld1q_s8(xrow[bi].qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
				int32x4_t acc	   = vdotq_s32(vdupq_n_s32(0), lo, xq_lo_vec);
				acc				   = vdotq_s32(acc, hi, xq_hi_vec);
				const int32_t sumi = vaddvq_s32(acc);
#else
				int16x8_t p0	   = vmull_s8(vget_low_s8(lo), vget_low_s8(xq_lo_vec));
				int16x8_t p1	   = vmull_s8(vget_high_s8(lo), vget_high_s8(xq_lo_vec));
				int16x8_t p2	   = vmull_s8(vget_low_s8(hi), vget_low_s8(xq_hi_vec));
				int16x8_t p3	   = vmull_s8(vget_high_s8(hi), vget_high_s8(xq_hi_vec));
				int32x4_t acc	   = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
				acc				   = vaddq_s32(acc, vpaddlq_s16(p2));
				acc				   = vaddq_s32(acc, vpaddlq_s16(p3));
				const int32_t sumi = vaddvq_s32(acc);
#endif
				const float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				sumf += d * (float)sumi;
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}

#undef NR

void dequant_iq3_s_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq3s_block *b = blocks;

	const int32x4_t	 mask_vals_low	= {kmask_iq2xs[0], kmask_iq2xs[1], kmask_iq2xs[2],
									   kmask_iq2xs[3]};
	const int32x4_t	 mask_vals_high = {kmask_iq2xs[4], kmask_iq2xs[5], kmask_iq2xs[6],
									   kmask_iq2xs[7]};
	const uint32x4_t zero			= vdupq_n_u32(0);

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const float	   d	 = f16_to_f32_fast(b[bi].d);
		const uint8_t *qs	 = b[bi].qs;
		const uint8_t *qh	 = b[bi].qh;
		const uint8_t *signs = b[bi].signs;
		float		  *y	 = dst + (bi * 256);

		for (int ib32 = 0; ib32 < 8; ib32 += 2) {
			const uint8_t scales_byte = b[bi].scales[ib32 / 2];
			const float	  db1		  = d * (1.0f + (2.0f * (scales_byte & 0xf)));
			const float	  db2		  = d * (1.0f + (2.0f * (scales_byte >> 4)));

			float32x4_t db1_vec = vdupq_n_f32(db1);
			float32x4_t db2_vec = vdupq_n_f32(db2);
			float32x4_t neg_db1 = vnegq_f32(db1_vec);
			float32x4_t neg_db2 = vnegq_f32(db2_vec);

			uint8x8_t qh_vals = vld1_u8(qh);
			uint8_t	  qh0	  = vget_lane_u8(qh_vals, 0);
			uint8_t	  qh1	  = vget_lane_u8(qh_vals, 1);

			for (int l = 0; l < 4; l++) {
				uint8_t	 q0	  = qs[2 * l];
				uint8_t	 q1	  = qs[(2 * l) + 1];
				uint32_t idx0 = q0 | ((qh0 << (8 - (2 * l))) & 256);
				uint32_t idx1 = q1 | ((qh0 << (7 - (2 * l))) & 256);

				uint32x2_t packed = {iq3s_grid[idx0], iq3s_grid[idx1]};
				uint8x8_t  bytes  = vreinterpret_u8_u32(packed);
				uint16x8_t u16	  = vmovl_u8(bytes);
				uint32x4_t u32_0  = vmovl_u16(vget_low_u16(u16));
				uint32x4_t u32_1  = vmovl_u16(vget_high_u16(u16));

				float32x4_t grid_0 = vcvtq_f32_u32(u32_0);
				float32x4_t grid_1 = vcvtq_f32_u32(u32_1);

				int32x4_t sign_vec = vdupq_n_s32(signs[l]);

				uint32x4_t test_low	 = vandq_u32(vreinterpretq_u32_s32(sign_vec),
												 vreinterpretq_u32_s32(mask_vals_low));
				uint32x4_t test_high = vandq_u32(vreinterpretq_u32_s32(sign_vec),
												 vreinterpretq_u32_s32(mask_vals_high));
				uint32x4_t mask_low	 = vcgtq_u32(test_low, zero);
				uint32x4_t mask_high = vcgtq_u32(test_high, zero);

				vst1q_f32(y, vmulq_f32(vbslq_f32(mask_low, neg_db1, db1_vec), grid_0));
				vst1q_f32(y + 4, vmulq_f32(vbslq_f32(mask_high, neg_db1, db1_vec), grid_1));
				y += 8;
			}
			qs += 8;
			signs += 4;

			for (int l = 0; l < 4; l++) {
				uint8_t	 q0	  = qs[2 * l];
				uint8_t	 q1	  = qs[(2 * l) + 1];
				uint32_t idx0 = q0 | ((qh1 << (8 - (2 * l))) & 256);
				uint32_t idx1 = q1 | ((qh1 << (7 - (2 * l))) & 256);

				uint32x2_t packed = {iq3s_grid[idx0], iq3s_grid[idx1]};
				uint8x8_t  bytes  = vreinterpret_u8_u32(packed);
				uint16x8_t u16	  = vmovl_u8(bytes);
				uint32x4_t u32_0  = vmovl_u16(vget_low_u16(u16));
				uint32x4_t u32_1  = vmovl_u16(vget_high_u16(u16));

				float32x4_t grid_0 = vcvtq_f32_u32(u32_0);
				float32x4_t grid_1 = vcvtq_f32_u32(u32_1);

				int32x4_t sign_vec = vdupq_n_s32(signs[l]);

				uint32x4_t test_low	 = vandq_u32(vreinterpretq_u32_s32(sign_vec),
												 vreinterpretq_u32_s32(mask_vals_low));
				uint32x4_t test_high = vandq_u32(vreinterpretq_u32_s32(sign_vec),
												 vreinterpretq_u32_s32(mask_vals_high));
				uint32x4_t mask_low	 = vcgtq_u32(test_low, zero);
				uint32x4_t mask_high = vcgtq_u32(test_high, zero);

				vst1q_f32(y, vmulq_f32(vbslq_f32(mask_low, neg_db2, db2_vec), grid_0));
				vst1q_f32(y + 4, vmulq_f32(vbslq_f32(mask_high, neg_db2, db2_vec), grid_1));
				y += 8;
			}
			qh += 2;
			qs += 8;
			signs += 4;
		}
	}
}

static uint8_t iq3s_sign_pattern[256][8];

static void __attribute__((constructor)) iq3s_sign_pattern_init(void) {
	for (int byte = 0; byte < 256; byte++) {
		for (int j = 0; j < 8; j++) {
			iq3s_sign_pattern[byte][j] = (byte & (1 << j)) ? 0xFF : 0x00;
		}
	}
}

static inline int8x8_t iq3s_flip8(uint8_t q0, uint8_t q1, uint8_t qh, uint8_t sign_byte, int sh0,
								  int sh1) {
	uint32_t   idx0 = q0 | ((qh << sh0) & 256);
	uint32_t   idx1 = q1 | ((qh << sh1) & 256);
	uint32x2_t pk	= {iq3s_grid[idx0], iq3s_grid[idx1]};
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

static void matmul_iq3_s_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
											float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(iq3s_block);
	const iq3s_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		float32x4_t acc0 = vdupq_n_f32(0.0f);
		float32x4_t acc1 = vdupq_n_f32(0.0f);

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float d_xq			  = yb->d;

			int pf_bi = bi + 2;
			if (pf_bi < blocks_per_row) {
				for (int r = 0; r < MR; r++) {
					const uint8_t *pf_base = row_base[r] + ((size_t)pf_bi * sizeof(iq3s_block));
					__builtin_prefetch(pf_base, 0, 1);
					__builtin_prefetch(pf_base + sizeof(iq3s_block) - 16, 0, 1);
				}
			} else if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++) {
					const uint8_t *pf_base = row_base[r] + ((size_t)(bi + 1) * sizeof(iq3s_block));
					__builtin_prefetch(pf_base, 0, 1);
					__builtin_prefetch(pf_base + sizeof(iq3s_block) - 16, 0, 1);
				}
			}

			int32_t	 sumi_lane[8];
			uint16_t d_w_raw[8];

			for (int r = 0; r < MR; r++) {
				const iq3s_block *restrict b =
					(const iq3s_block *)(row_base[r] + ((size_t)bi * sizeof(iq3s_block)));
				d_w_raw[r]	  = b->d;
				int32_t total = iq3s_block_dot(b, yb->qs);
				sumi_lane[r]  = total;
			}

			int32x4_t		 sumi0	 = vld1q_s32(sumi_lane);
			int32x4_t		 sumi1	 = vld1q_s32(sumi_lane + 4);
			float32x4_t		 sumi0f	 = vcvtq_f32_s32(sumi0);
			float32x4_t		 sumi1f	 = vcvtq_f32_s32(sumi1);
			const uint16x8_t d_w_u16 = vld1q_u16(d_w_raw);
			float32x4_t		 d_w0	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_w_u16)));
			float32x4_t		 d_w1	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_w_u16)));
			float32x4_t		 d_xq_v	 = vdupq_n_f32(d_xq);

			acc0 = vfmaq_f32(acc0, vmulq_f32(d_w0, d_xq_v), sumi0f);
			acc1 = vfmaq_f32(acc1, vmulq_f32(d_w1, d_xq_v), sumi1f);
		}

		store_acc8(acc0, acc1, y, i);
	}

	for (; i < n; i++) {
		const iq3s_block *bx = (const iq3s_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float			  sumf = 0.0f;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const iq3s_block *restrict b  = &bx[bi];
			const q8_k_block *restrict yb = &xq[bi];
			const float d				  = f16_to_f32_fast(b->d) * yb->d;

			int32_t total = iq3s_block_dot(b, yb->qs);
			sumf += d * (float)total;
		}
		y[i] = sumf;
	}
}

#define NR 4

void matmul_iq3_s_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(iq3s_block);
	const iq3s_block *Wb			 = w;
	int				  i				 = 0;

	int8x16_t *decoded_cache = NULL;
	int32_t	  *sc0_cache	 = NULL;
	int32_t	  *sc1_cache	 = NULL;
	float	  *d_w_cache	 = NULL;
	if (m >= 2 * NR && blocks_per_row > 0) {
		decoded_cache = malloc(sizeof(int8x16_t) * (size_t)blocks_per_row * MR * 16);
		sc0_cache	  = malloc(sizeof(int32_t) * (size_t)blocks_per_row * MR * 4);
		sc1_cache	  = malloc(sizeof(int32_t) * (size_t)blocks_per_row * MR * 4);
		d_w_cache	  = malloc(sizeof(float) * (size_t)blocks_per_row * MR);
		if (!decoded_cache || !sc0_cache || !sc1_cache || !d_w_cache) {
			free(decoded_cache);
			free(sc0_cache);
			free(sc1_cache);
			free(d_w_cache);
			decoded_cache = NULL;
			sc0_cache = sc1_cache = NULL;
			d_w_cache			  = NULL;
		}
	}

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		if (decoded_cache) {
			for (int bi = 0; bi < blocks_per_row; bi++) {
				int8x16_t *dcache = decoded_cache + (size_t)bi * MR * 16;
				int32_t	  *sc0c	  = sc0_cache + (size_t)bi * MR * 4;
				int32_t	  *sc1c	  = sc1_cache + (size_t)bi * MR * 4;
				float	  *dwc	  = d_w_cache + (size_t)bi * MR;

				for (int r = 0; r < MR; r++) {
					const iq3s_block *restrict b =
						(const iq3s_block *)(row_base[r] + ((size_t)bi * sizeof(iq3s_block)));
					dwc[r] = f16_to_f32_fast(b->d);

					const uint8_t *restrict qs	  = b->qs;
					const uint8_t *restrict qh	  = b->qh;
					const uint8_t *restrict signs = b->signs;
					int8x16_t *restrict decoded_r = dcache + (size_t)r * 16;

					for (int ib32 = 0; ib32 < 8; ib32 += 2) {
						const uint8_t sb	   = b->scales[ib32 / 2];
						sc0c[r * 4 + ib32 / 2] = 1 + (2 * (sb & 0xf));
						sc1c[r * 4 + ib32 / 2] = 1 + (2 * (sb >> 4));

						const uint8_t qh0 = qh[0];
						const uint8_t qh1 = qh[1];
						const int	  vi  = ib32 * 2;

						const int8x8_t f00 = iq3s_flip8(qs[0], qs[1], qh0, signs[0], 8, 7);
						const int8x8_t f01 = iq3s_flip8(qs[2], qs[3], qh0, signs[1], 6, 5);
						const int8x8_t f02 = iq3s_flip8(qs[4], qs[5], qh0, signs[2], 4, 3);
						const int8x8_t f03 = iq3s_flip8(qs[6], qs[7], qh0, signs[3], 2, 1);
						const int8x8_t f10 = iq3s_flip8(qs[8], qs[9], qh1, signs[4], 8, 7);
						const int8x8_t f11 = iq3s_flip8(qs[10], qs[11], qh1, signs[5], 6, 5);
						const int8x8_t f12 = iq3s_flip8(qs[12], qs[13], qh1, signs[6], 4, 3);
						const int8x8_t f13 = iq3s_flip8(qs[14], qs[15], qh1, signs[7], 2, 1);

						decoded_r[vi + 0] = vcombine_s8(f00, f01);
						decoded_r[vi + 1] = vcombine_s8(f02, f03);
						decoded_r[vi + 2] = vcombine_s8(f10, f11);
						decoded_r[vi + 3] = vcombine_s8(f12, f13);

						qh += 2;
						qs += 16;
						signs += 8;
					}
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_k_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				int8x16_t local_decoded[MR][16];
				int32_t	  local_sc0[MR][4];
				int32_t	  local_sc1[MR][4];
				float	  local_d_w[MR];

				int8x16_t(*decoded)[16];
				int32_t (*sc0)[4];
				int32_t (*sc1)[4];
				float *d_w;

				if (decoded_cache) {
					decoded = (int8x16_t(*)[16])(decoded_cache + (size_t)bi * MR * 16);
					sc0		= (int32_t (*)[4])(sc0_cache + (size_t)bi * MR * 4);
					sc1		= (int32_t (*)[4])(sc1_cache + (size_t)bi * MR * 4);
					d_w		= d_w_cache + (size_t)bi * MR;
				} else {
					decoded = local_decoded;
					sc0		= local_sc0;
					sc1		= local_sc1;
					d_w		= local_d_w;

					for (int r = 0; r < MR; r++) {
						const iq3s_block *restrict b =
							(const iq3s_block *)(row_base[r] + ((size_t)bi * sizeof(iq3s_block)));
						d_w[r] = f16_to_f32_fast(b->d);

						const uint8_t *restrict qs	  = b->qs;
						const uint8_t *restrict qh	  = b->qh;
						const uint8_t *restrict signs = b->signs;

						for (int ib32 = 0; ib32 < 8; ib32 += 2) {
							const uint8_t sb = b->scales[ib32 / 2];
							sc0[r][ib32 / 2] = 1 + (2 * (sb & 0xf));
							sc1[r][ib32 / 2] = 1 + (2 * (sb >> 4));

							const uint8_t qh0 = qh[0];
							const uint8_t qh1 = qh[1];
							const int	  vi  = ib32 * 2;

							const int8x8_t f00 = iq3s_flip8(qs[0], qs[1], qh0, signs[0], 8, 7);
							const int8x8_t f01 = iq3s_flip8(qs[2], qs[3], qh0, signs[1], 6, 5);
							const int8x8_t f02 = iq3s_flip8(qs[4], qs[5], qh0, signs[2], 4, 3);
							const int8x8_t f03 = iq3s_flip8(qs[6], qs[7], qh0, signs[3], 2, 1);
							const int8x8_t f10 = iq3s_flip8(qs[8], qs[9], qh1, signs[4], 8, 7);
							const int8x8_t f11 = iq3s_flip8(qs[10], qs[11], qh1, signs[5], 6, 5);
							const int8x8_t f12 = iq3s_flip8(qs[12], qs[13], qh1, signs[6], 4, 3);
							const int8x8_t f13 = iq3s_flip8(qs[14], qs[15], qh1, signs[7], 2, 1);

							decoded[r][vi + 0] = vcombine_s8(f00, f01);
							decoded[r][vi + 1] = vcombine_s8(f02, f03);
							decoded[r][vi + 2] = vcombine_s8(f10, f11);
							decoded[r][vi + 3] = vcombine_s8(f12, f13);

							qh += 2;
							qs += 16;
							signs += 8;
						}
					}
				}

				float xd[NR];
				const int8_t *restrict q8p[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]  = xrow[c][bi].d;
					q8p[c] = xrow[c][bi].qs;
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				for (int r = 0; r < MR; r++) {
					int32x4_t total4 = vdupq_n_s32(0);

					for (int ib32 = 0; ib32 < 8; ib32 += 2) {
						const int8x16_t c00 = decoded[r][(ib32 * 2) + 0];
						const int8x16_t c01 = decoded[r][(ib32 * 2) + 1];
						const int8x16_t c10 = decoded[r][(ib32 * 2) + 2];
						const int8x16_t c11 = decoded[r][(ib32 * 2) + 3];

						const int8_t *restrict qg0_0 = q8p[0] + (ib32 * 32);
						const int8_t *restrict qg0_1 = q8p[1] + (ib32 * 32);
						const int8_t *restrict qg0_2 = q8p[2] + (ib32 * 32);
						const int8_t *restrict qg0_3 = q8p[3] + (ib32 * 32);

						const int8_t *restrict qg1_0 = q8p[0] + ((ib32 + 1) * 32);
						const int8_t *restrict qg1_1 = q8p[1] + ((ib32 + 1) * 32);
						const int8_t *restrict qg1_2 = q8p[2] + ((ib32 + 1) * 32);
						const int8_t *restrict qg1_3 = q8p[3] + ((ib32 + 1) * 32);

#if defined(__ARM_FEATURE_DOTPROD)
						int32x4_t a0_c0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_0)),
													c01, vld1q_s8(qg0_0 + 16));
						int32x4_t a0_c1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_1)),
													c01, vld1q_s8(qg0_1 + 16));
						int32x4_t a0_c2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_2)),
													c01, vld1q_s8(qg0_2 + 16));
						int32x4_t a0_c3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_3)),
													c01, vld1q_s8(qg0_3 + 16));

						int32x4_t a1_c0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_0)),
													c11, vld1q_s8(qg1_0 + 16));
						int32x4_t a1_c1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_1)),
													c11, vld1q_s8(qg1_1 + 16));
						int32x4_t a1_c2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_2)),
													c11, vld1q_s8(qg1_2 + 16));
						int32x4_t a1_c3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_3)),
													c11, vld1q_s8(qg1_3 + 16));
#else
						int16x8_t p00_0 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_0)));
						int16x8_t p01_0 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_0)));
						int16x8_t p02_0 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_0 + 16)));
						int16x8_t p03_0 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_0 + 16)));
						int32x4_t a0_c0 = vpaddlq_s16(p00_0);
						a0_c0			= vpadalq_s16(a0_c0, p01_0);
						a0_c0			= vpadalq_s16(a0_c0, p02_0);
						a0_c0			= vpadalq_s16(a0_c0, p03_0);

						int16x8_t p00_1 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_1)));
						int16x8_t p01_1 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_1)));
						int16x8_t p02_1 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_1 + 16)));
						int16x8_t p03_1 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_1 + 16)));
						int32x4_t a0_c1 = vpaddlq_s16(p00_1);
						a0_c1			= vpadalq_s16(a0_c1, p01_1);
						a0_c1			= vpadalq_s16(a0_c1, p02_1);
						a0_c1			= vpadalq_s16(a0_c1, p03_1);

						int16x8_t p00_2 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_2)));
						int16x8_t p01_2 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_2)));
						int16x8_t p02_2 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_2 + 16)));
						int16x8_t p03_2 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_2 + 16)));
						int32x4_t a0_c2 = vpaddlq_s16(p00_2);
						a0_c2			= vpadalq_s16(a0_c2, p01_2);
						a0_c2			= vpadalq_s16(a0_c2, p02_2);
						a0_c2			= vpadalq_s16(a0_c2, p03_2);

						int16x8_t p00_3 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_3)));
						int16x8_t p01_3 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_3)));
						int16x8_t p02_3 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_3 + 16)));
						int16x8_t p03_3 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_3 + 16)));
						int32x4_t a0_c3 = vpaddlq_s16(p00_3);
						a0_c3			= vpadalq_s16(a0_c3, p01_3);
						a0_c3			= vpadalq_s16(a0_c3, p02_3);
						a0_c3			= vpadalq_s16(a0_c3, p03_3);

						int16x8_t p10_0 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_0)));
						int16x8_t p11_0 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_0)));
						int16x8_t p12_0 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_0 + 16)));
						int16x8_t p13_0 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_0 + 16)));
						int32x4_t a1_c0 = vpaddlq_s16(p10_0);
						a1_c0			= vpadalq_s16(a1_c0, p11_0);
						a1_c0			= vpadalq_s16(a1_c0, p12_0);
						a1_c0			= vpadalq_s16(a1_c0, p13_0);

						int16x8_t p10_1 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_1)));
						int16x8_t p11_1 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_1)));
						int16x8_t p12_1 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_1 + 16)));
						int16x8_t p13_1 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_1 + 16)));
						int32x4_t a1_c1 = vpaddlq_s16(p10_1);
						a1_c1			= vpadalq_s16(a1_c1, p11_1);
						a1_c1			= vpadalq_s16(a1_c1, p12_1);
						a1_c1			= vpadalq_s16(a1_c1, p13_1);

						int16x8_t p10_2 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_2)));
						int16x8_t p11_2 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_2)));
						int16x8_t p12_2 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_2 + 16)));
						int16x8_t p13_2 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_2 + 16)));
						int32x4_t a1_c2 = vpaddlq_s16(p10_2);
						a1_c2			= vpadalq_s16(a1_c2, p11_2);
						a1_c2			= vpadalq_s16(a1_c2, p12_2);
						a1_c2			= vpadalq_s16(a1_c2, p13_2);

						int16x8_t p10_3 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_3)));
						int16x8_t p11_3 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_3)));
						int16x8_t p12_3 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_3 + 16)));
						int16x8_t p13_3 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_3 + 16)));
						int32x4_t a1_c3 = vpaddlq_s16(p10_3);
						a1_c3			= vpadalq_s16(a1_c3, p11_3);
						a1_c3			= vpadalq_s16(a1_c3, p12_3);
						a1_c3			= vpadalq_s16(a1_c3, p13_3);
#endif
						int32x4_t a0_s01	= vpaddq_s32(a0_c0, a0_c1);
						int32x4_t a0_s23	= vpaddq_s32(a0_c2, a0_c3);
						int32x4_t a0_packed = vpaddq_s32(a0_s01, a0_s23);

						int32x4_t a1_s01	= vpaddq_s32(a1_c0, a1_c1);
						int32x4_t a1_s23	= vpaddq_s32(a1_c2, a1_c3);
						int32x4_t a1_packed = vpaddq_s32(a1_s01, a1_s23);

						total4 = vmlaq_n_s32(total4, a0_packed, sc0[r][ib32 / 2]);
						total4 = vmlaq_n_s32(total4, a1_packed, sc1[r][ib32 / 2]);
					}

					float32x4_t total4f = vcvtq_f32_s32(total4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(total4f, d_w[r]));
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_iq3_s_q8_k_qonly_f32_row((const uint8_t *)Wb + ((size_t)i * row_stride),
											xq + ((size_t)t * xq_row_stride_blocks),
											y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const iq3s_block *row =
			(const iq3s_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		for (int t = 0; t < m; t++) {
			const q8_k_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const iq3s_block *restrict b  = &row[bi];
				const q8_k_block *restrict xb = &xrow[bi];
				const float d				  = f16_to_f32_fast(b->d) * xb->d;
				int32_t		total			  = iq3s_block_dot(b, xb->qs);
				sumf += d * (float)total;
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}

	free(decoded_cache);
	free(sc0_cache);
	free(sc1_cache);
	free(d_w_cache);
}

#undef NR

#define Q4_0_R8_GROUP_BYTES (Q4_0_R8_ROWS * sizeof(uint16_t) + Q4_0_R8_ROWS * 16)

static void matmul_q4_0_r8_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
											float *restrict y, int n, int k) {
	const int	   blocks_per_row = k / 32;
	const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(q4_0_block);
	const uint8_t *Wb			  = w;
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		float32x4_t acc_lo = vdupq_n_f32(0.0f);
		float32x4_t acc_hi = vdupq_n_f32(0.0f);

		const uint8_t *group = Wb + ((size_t)i * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t *blk = group + (size_t)bi * Q4_0_R8_GROUP_BYTES;

			if (bi + 1 < blocks_per_row)
				__builtin_prefetch(blk + Q4_0_R8_GROUP_BYTES, 0, 1);

			const float		d_xq = f16_to_f32_fast(xq[bi].d);
			const int8x16_t xl	 = vld1q_s8(xq[bi].qs);
			const int8x16_t xh	 = vld1q_s8(xq[bi].qs + 16);

			const uint16_t *d_ptr  = (const uint16_t *)blk;
			const uint8_t  *qs_ptr = blk + Q4_0_R8_ROWS * sizeof(uint16_t);

			int32x4_t acc[8];
			for (int r = 0; r < 8; r++) {
				const uint8x16_t q = vld1q_u8(qs_ptr + (size_t)r * 16);
				const int8x16_t	 lo =
					vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
				const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), vdupq_n_s8(8));

#if defined(__ARM_FEATURE_DOTPROD)
				int32x4_t a = vdotq_s32(vdupq_n_s32(0), lo, xl);
				a			= vdotq_s32(a, hi, xh);
#else
				int16x8_t p0 = vmull_s8(vget_low_s8(lo), vget_low_s8(xl));
				int16x8_t p1 = vmull_s8(vget_high_s8(lo), vget_high_s8(xl));
				int16x8_t p2 = vmull_s8(vget_low_s8(hi), vget_low_s8(xh));
				int16x8_t p3 = vmull_s8(vget_high_s8(hi), vget_high_s8(xh));
				int32x4_t a	 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
				a			 = vaddq_s32(a, vpaddlq_s16(p2));
				a			 = vaddq_s32(a, vpaddlq_s16(p3));
#endif
				acc[r] = a;
			}

			const int32x4_t sum01	= vpaddq_s32(acc[0], acc[1]);
			const int32x4_t sum23	= vpaddq_s32(acc[2], acc[3]);
			const int32x4_t sumi_lo = vpaddq_s32(sum01, sum23);

			const int32x4_t sum45	= vpaddq_s32(acc[4], acc[5]);
			const int32x4_t sum67	= vpaddq_s32(acc[6], acc[7]);
			const int32x4_t sumi_hi = vpaddq_s32(sum45, sum67);

			const uint16x8_t d_w_u16 = vld1q_u16(d_ptr);
			float32x4_t		 d_w0	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_w_u16)));
			float32x4_t		 d_w1	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_w_u16)));
			float32x4_t		 d_xq_v	 = vdupq_n_f32(d_xq);

			acc_lo = vfmaq_f32(acc_lo, vmulq_f32(d_w0, d_xq_v), vcvtq_f32_s32(sumi_lo));
			acc_hi = vfmaq_f32(acc_hi, vmulq_f32(d_w1, d_xq_v), vcvtq_f32_s32(sumi_hi));
		}

		float tmp0[4], tmp1[4];
		vst1q_f32(tmp0, acc_lo);
		vst1q_f32(tmp1, acc_hi);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_q4_0_r8_q8_qonly_f32_i8mm(const void *w, const q8_0_block *restrict xq,
											 size_t		 xq_row_stride_blocks, float *restrict y,
											 int y_row_stride, int n, int k, int m) {
	const int	   blocks_per_row = k / 32;
	const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(q4_0_block);
	const uint8_t *Wb			  = w;
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++)
					facc[p][c] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * Q4_0_R8_GROUP_BYTES;

				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + Q4_0_R8_GROUP_BYTES, 0, 1);

				int8x16_t xq_lo[I8MM_NR];
				int8x16_t xq_hi[I8MM_NR];
				float	  xd[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}

				int8x16_t bvec[I8MM_NR / 2][4];
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					const int ca = 2 * cp;
					const int cb = 2 * cp + 1;
					bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb]));
					bvec[cp][1]	 = vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));
					bvec[cp][2]	 = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));
					bvec[cp][3]	 = vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));
				}

				const uint16_t *d_ptr  = (const uint16_t *)blk;
				const uint8_t  *qs_ptr = blk + Q4_0_R8_ROWS * sizeof(uint16_t);

				for (int p = 0; p < MR / 2; p++) {
					const float dwa = f16_to_f32_fast(d_ptr[2 * p]);
					const float dwb = f16_to_f32_fast(d_ptr[2 * p + 1]);

					const uint8x16_t q0		 = vld1q_u8(qs_ptr + (size_t)(2 * p) * 16);
					const uint8x16_t q1		 = vld1q_u8(qs_ptr + (size_t)(2 * p + 1) * 16);
					const uint8x16_t l0		 = vandq_u8(q0, vdupq_n_u8(0x0F));
					const uint8x16_t h0		 = vshrq_n_u8(q0, 4);
					const uint8x16_t l1		 = vandq_u8(q1, vdupq_n_u8(0x0F));
					const uint8x16_t h1		 = vshrq_n_u8(q1, 4);
					const int8x16_t	 avec[4] = {
						vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(l0), vget_low_u8(l1))),
								 vdupq_n_s8(8)),
						vsubq_s8(
							vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(l0), vget_high_u8(l1))),
							vdupq_n_s8(8)),
						vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(h0), vget_low_u8(h1))),
								 vdupq_n_s8(8)),
						vsubq_s8(
							vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(h0), vget_high_u8(h1))),
							vdupq_n_s8(8)),
					};

					const float32x4_t srow = vcombine_f32(vdup_n_f32(dwa), vdup_n_f32(dwb));

					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol =
							vzip1q_f32(vdupq_n_f32(xd[2 * cp]), vdupq_n_f32(xd[2 * cp + 1]));
						int32x4_t s = vdupq_n_s32(0);
						s			= vmmlaq_s32(s, avec[0], bvec[cp][0]);
						s			= vmmlaq_s32(s, avec[1], bvec[cp][1]);
						s			= vmmlaq_s32(s, avec[2], bvec[cp][2]);
						s			= vmmlaq_s32(s, avec[3], bvec[cp][3]);
						facc[p][cp] =
							vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, facc[p][cp]);
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_q4_0_r8_q8_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
											y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}
}
#endif

#define NR 4

void matmul_q4_0_r8_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_q4_0_r8_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int	   blocks_per_row = k / 32;
	const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(q4_0_block);
	const uint8_t *Wb			  = w;
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * Q4_0_R8_GROUP_BYTES;

				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + Q4_0_R8_GROUP_BYTES, 0, 1);

				float	  xd[NR];
				int8x16_t xq_lo[NR];
				int8x16_t xq_hi[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				const uint16_t *d_ptr  = (const uint16_t *)blk;
				const uint8_t  *qs_ptr = blk + Q4_0_R8_ROWS * sizeof(uint16_t);

				for (int r = 0; r < MR; r++) {
					const float		 d_w = f16_to_f32_fast(d_ptr[r]);
					const uint8x16_t q	 = vld1q_u8(qs_ptr + (size_t)r * 16);
					const int8x16_t	 lo =
						vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
					const int8x16_t hi =
						vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), vdupq_n_s8(8));

					const int32x4_t	  sumi4	 = q4_dot_x4_sumi4(lo, hi, xq_lo, xq_hi);
					const float32x4_t sumi_f = vcvtq_f32_s32(sumi4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(sumi_f, d_w));
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_q4_0_r8_q8_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
											y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}
}

#undef NR

#define Q8_0_R8_GROUP_BYTES (Q8_0_R8_ROWS * sizeof(uint16_t) + Q8_0_R8_ROWS * 32)

static void matmul_q8_0_r8_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
											float *restrict y, int n, int k) {
	const int	   blocks_per_row = k / 32;
	const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(q8_0_block);
	const uint8_t *Wb			  = w;
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		float32x4_t acc_lo = vdupq_n_f32(0.0f);
		float32x4_t acc_hi = vdupq_n_f32(0.0f);

		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int bi = 0;
		for (; bi + 1 < blocks_per_row; bi += 2) {
			const uint8_t *blk0 = group + (size_t)bi * Q8_0_R8_GROUP_BYTES;
			const uint8_t *blk1 = group + (size_t)(bi + 1) * Q8_0_R8_GROUP_BYTES;

			if (bi + 2 < blocks_per_row)
				__builtin_prefetch(group + (size_t)(bi + 2) * Q8_0_R8_GROUP_BYTES, 0, 1);

			const float		d_xq0 = f16_to_f32_fast(xq[bi].d);
			const float		d_xq1 = f16_to_f32_fast(xq[bi + 1].d);
			const int8x16_t xl0	  = vld1q_s8(xq[bi].qs);
			const int8x16_t xh0	  = vld1q_s8(xq[bi].qs + 16);
			const int8x16_t xl1	  = vld1q_s8(xq[bi + 1].qs);
			const int8x16_t xh1	  = vld1q_s8(xq[bi + 1].qs + 16);

			const uint16_t *d_ptr0 = (const uint16_t *)blk0;
			const uint16_t *d_ptr1 = (const uint16_t *)blk1;
			const int8_t   *qs0	   = (const int8_t *)(blk0 + Q8_0_R8_ROWS * sizeof(uint16_t));
			const int8_t   *qs1	   = (const int8_t *)(blk1 + Q8_0_R8_ROWS * sizeof(uint16_t));

			int32x4_t p0a =
				dotprod2_s8(vld1q_s8(qs0 + 0 * 32), xl0, vld1q_s8(qs0 + 0 * 32 + 16), xh0);

			int32x4_t p0b =
				dotprod2_s8(vld1q_s8(qs1 + 0 * 32), xl1, vld1q_s8(qs1 + 0 * 32 + 16), xh1);

			int32x4_t p1a =
				dotprod2_s8(vld1q_s8(qs0 + 1 * 32), xl0, vld1q_s8(qs0 + 1 * 32 + 16), xh0);
			int32x4_t p1b =
				dotprod2_s8(vld1q_s8(qs1 + 1 * 32), xl1, vld1q_s8(qs1 + 1 * 32 + 16), xh1);

			int32x4_t p2a =
				dotprod2_s8(vld1q_s8(qs0 + 2 * 32), xl0, vld1q_s8(qs0 + 2 * 32 + 16), xh0);
			int32x4_t p2b =
				dotprod2_s8(vld1q_s8(qs1 + 2 * 32), xl1, vld1q_s8(qs1 + 2 * 32 + 16), xh1);

			int32x4_t p3a =
				dotprod2_s8(vld1q_s8(qs0 + 3 * 32), xl0, vld1q_s8(qs0 + 3 * 32 + 16), xh0);
			int32x4_t p3b =
				dotprod2_s8(vld1q_s8(qs1 + 3 * 32), xl1, vld1q_s8(qs1 + 3 * 32 + 16), xh1);

			int32x4_t p4a =
				dotprod2_s8(vld1q_s8(qs0 + 4 * 32), xl0, vld1q_s8(qs0 + 4 * 32 + 16), xh0);
			int32x4_t p4b =
				dotprod2_s8(vld1q_s8(qs1 + 4 * 32), xl1, vld1q_s8(qs1 + 4 * 32 + 16), xh1);

			int32x4_t p5a =
				dotprod2_s8(vld1q_s8(qs0 + 5 * 32), xl0, vld1q_s8(qs0 + 5 * 32 + 16), xh0);
			int32x4_t p5b =
				dotprod2_s8(vld1q_s8(qs1 + 5 * 32), xl1, vld1q_s8(qs1 + 5 * 32 + 16), xh1);

			int32x4_t p6a =
				dotprod2_s8(vld1q_s8(qs0 + 6 * 32), xl0, vld1q_s8(qs0 + 6 * 32 + 16), xh0);
			int32x4_t p6b =
				dotprod2_s8(vld1q_s8(qs1 + 6 * 32), xl1, vld1q_s8(qs1 + 6 * 32 + 16), xh1);

			int32x4_t p7a =
				dotprod2_s8(vld1q_s8(qs0 + 7 * 32), xl0, vld1q_s8(qs0 + 7 * 32 + 16), xh0);
			int32x4_t p7b =
				dotprod2_s8(vld1q_s8(qs1 + 7 * 32), xl1, vld1q_s8(qs1 + 7 * 32 + 16), xh1);

			const int32x4_t sum01a	 = vpaddq_s32(p0a, p1a);
			const int32x4_t sum23a	 = vpaddq_s32(p2a, p3a);
			const int32x4_t sumi_loa = vpaddq_s32(sum01a, sum23a);
			const int32x4_t sum45a	 = vpaddq_s32(p4a, p5a);
			const int32x4_t sum67a	 = vpaddq_s32(p6a, p7a);
			const int32x4_t sumi_hia = vpaddq_s32(sum45a, sum67a);

			const int32x4_t sum01b	 = vpaddq_s32(p0b, p1b);
			const int32x4_t sum23b	 = vpaddq_s32(p2b, p3b);
			const int32x4_t sumi_lob = vpaddq_s32(sum01b, sum23b);
			const int32x4_t sum45b	 = vpaddq_s32(p4b, p5b);
			const int32x4_t sum67b	 = vpaddq_s32(p6b, p7b);
			const int32x4_t sumi_hib = vpaddq_s32(sum45b, sum67b);

			const uint16x8_t d_u16_0   = vld1q_u16(d_ptr0);
			const uint16x8_t d_u16_1   = vld1q_u16(d_ptr1);
			float32x4_t		 d_f32_lo0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_u16_0)));
			float32x4_t		 d_f32_hi0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_u16_0)));
			float32x4_t		 d_f32_lo1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_u16_1)));
			float32x4_t		 d_f32_hi1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_u16_1)));

			float32x4_t d_xq_v0 = vdupq_n_f32(d_xq0);
			float32x4_t d_xq_v1 = vdupq_n_f32(d_xq1);
			float32x4_t dw_lo0	= vmulq_f32(d_f32_lo0, d_xq_v0);
			float32x4_t dw_hi0	= vmulq_f32(d_f32_hi0, d_xq_v0);
			float32x4_t dw_lo1	= vmulq_f32(d_f32_lo1, d_xq_v1);
			float32x4_t dw_hi1	= vmulq_f32(d_f32_hi1, d_xq_v1);

			acc_lo = vfmaq_f32(acc_lo, vcvtq_f32_s32(sumi_loa), dw_lo0);
			acc_hi = vfmaq_f32(acc_hi, vcvtq_f32_s32(sumi_hia), dw_hi0);
			acc_lo = vfmaq_f32(acc_lo, vcvtq_f32_s32(sumi_lob), dw_lo1);
			acc_hi = vfmaq_f32(acc_hi, vcvtq_f32_s32(sumi_hib), dw_hi1);
		}

		for (; bi < blocks_per_row; bi++) {
			const uint8_t *blk = group + (size_t)bi * Q8_0_R8_GROUP_BYTES;

			const float		d_xq = f16_to_f32_fast(xq[bi].d);
			const int8x16_t xl	 = vld1q_s8(xq[bi].qs);
			const int8x16_t xh	 = vld1q_s8(xq[bi].qs + 16);

			const uint16_t *d_ptr  = (const uint16_t *)blk;
			const int8_t   *qs_ptr = (const int8_t *)(blk + Q8_0_R8_ROWS * sizeof(uint16_t));

			int32x4_t p0 =
				dotprod2_s8(vld1q_s8(qs_ptr + 0 * 32), xl, vld1q_s8(qs_ptr + 0 * 32 + 16), xh);
			int32x4_t p1 =
				dotprod2_s8(vld1q_s8(qs_ptr + 1 * 32), xl, vld1q_s8(qs_ptr + 1 * 32 + 16), xh);
			int32x4_t p2 =
				dotprod2_s8(vld1q_s8(qs_ptr + 2 * 32), xl, vld1q_s8(qs_ptr + 2 * 32 + 16), xh);
			int32x4_t p3 =
				dotprod2_s8(vld1q_s8(qs_ptr + 3 * 32), xl, vld1q_s8(qs_ptr + 3 * 32 + 16), xh);
			int32x4_t p4 =
				dotprod2_s8(vld1q_s8(qs_ptr + 4 * 32), xl, vld1q_s8(qs_ptr + 4 * 32 + 16), xh);
			int32x4_t p5 =
				dotprod2_s8(vld1q_s8(qs_ptr + 5 * 32), xl, vld1q_s8(qs_ptr + 5 * 32 + 16), xh);
			int32x4_t p6 =
				dotprod2_s8(vld1q_s8(qs_ptr + 6 * 32), xl, vld1q_s8(qs_ptr + 6 * 32 + 16), xh);
			int32x4_t p7 =
				dotprod2_s8(vld1q_s8(qs_ptr + 7 * 32), xl, vld1q_s8(qs_ptr + 7 * 32 + 16), xh);

			const int32x4_t sum01	= vpaddq_s32(p0, p1);
			const int32x4_t sum23	= vpaddq_s32(p2, p3);
			const int32x4_t sumi_lo = vpaddq_s32(sum01, sum23);
			const int32x4_t sum45	= vpaddq_s32(p4, p5);
			const int32x4_t sum67	= vpaddq_s32(p6, p7);
			const int32x4_t sumi_hi = vpaddq_s32(sum45, sum67);

			const uint16x8_t d_u16	  = vld1q_u16(d_ptr);
			float32x4_t		 d_f32_lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_u16)));
			float32x4_t		 d_f32_hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_u16)));

			float32x4_t d_xq_v = vdupq_n_f32(d_xq);
			float32x4_t dw_lo  = vmulq_f32(d_f32_lo, d_xq_v);
			float32x4_t dw_hi  = vmulq_f32(d_f32_hi, d_xq_v);

			acc_lo = vfmaq_f32(acc_lo, vcvtq_f32_s32(sumi_lo), dw_lo);
			acc_hi = vfmaq_f32(acc_hi, vcvtq_f32_s32(sumi_hi), dw_hi);
		}

		y[i + 0] = vgetq_lane_f32(acc_lo, 0);
		y[i + 1] = vgetq_lane_f32(acc_lo, 1);
		y[i + 2] = vgetq_lane_f32(acc_lo, 2);
		y[i + 3] = vgetq_lane_f32(acc_lo, 3);
		y[i + 4] = vgetq_lane_f32(acc_hi, 0);
		y[i + 5] = vgetq_lane_f32(acc_hi, 1);
		y[i + 6] = vgetq_lane_f32(acc_hi, 2);
		y[i + 7] = vgetq_lane_f32(acc_hi, 3);
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_q8_0_r8_q8_qonly_f32_i8mm(const void *w, const q8_0_block *restrict xq,
											 size_t		 xq_row_stride_blocks, float *restrict y,
											 int y_row_stride, int n, int k, int m) {
	const int	   blocks_per_row = k / 32;
	const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(q8_0_block);
	const uint8_t *Wb			  = w;
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++)
					facc[p][c] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * Q8_0_R8_GROUP_BYTES;

				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + Q8_0_R8_GROUP_BYTES, 0, 1);

				int8x16_t xq_lo[I8MM_NR];
				int8x16_t xq_hi[I8MM_NR];
				float	  xd[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}

				int8x16_t bvec[I8MM_NR / 2][4];
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					const int ca = 2 * cp;
					const int cb = 2 * cp + 1;
					bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb]));
					bvec[cp][1]	 = vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));
					bvec[cp][2]	 = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));
					bvec[cp][3]	 = vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));
				}

				const uint16_t *d_ptr  = (const uint16_t *)blk;
				const int8_t   *qs_ptr = (const int8_t *)(blk + Q8_0_R8_ROWS * sizeof(uint16_t));

				for (int p = 0; p < MR / 2; p++) {
					const float dwa = f16_to_f32_fast(d_ptr[2 * p]);
					const float dwb = f16_to_f32_fast(d_ptr[2 * p + 1]);

					const int8x16_t w0		= vld1q_s8(qs_ptr + (size_t)(2 * p) * 32);
					const int8x16_t wh0		= vld1q_s8(qs_ptr + (size_t)(2 * p) * 32 + 16);
					const int8x16_t w1		= vld1q_s8(qs_ptr + (size_t)(2 * p + 1) * 32);
					const int8x16_t wh1		= vld1q_s8(qs_ptr + (size_t)(2 * p + 1) * 32 + 16);
					const int8x16_t avec[4] = {
						vcombine_s8(vget_low_s8(w0), vget_low_s8(w1)),
						vcombine_s8(vget_high_s8(w0), vget_high_s8(w1)),
						vcombine_s8(vget_low_s8(wh0), vget_low_s8(wh1)),
						vcombine_s8(vget_high_s8(wh0), vget_high_s8(wh1)),
					};

					const float32x4_t srow = vcombine_f32(vdup_n_f32(dwa), vdup_n_f32(dwb));

					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol =
							vzip1q_f32(vdupq_n_f32(xd[2 * cp]), vdupq_n_f32(xd[2 * cp + 1]));
						int32x4_t s = vdupq_n_s32(0);
						s			= vmmlaq_s32(s, avec[0], bvec[cp][0]);
						s			= vmmlaq_s32(s, avec[1], bvec[cp][1]);
						s			= vmmlaq_s32(s, avec[2], bvec[cp][2]);
						s			= vmmlaq_s32(s, avec[3], bvec[cp][3]);
						facc[p][cp] =
							vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, facc[p][cp]);
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_q8_0_r8_q8_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
											y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}
}
#endif

#define NR 4

void matmul_q8_0_r8_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_q8_0_r8_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int	   blocks_per_row = k / 32;
	const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(q8_0_block);
	const uint8_t *Wb			  = w;
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * Q8_0_R8_GROUP_BYTES;

				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + Q8_0_R8_GROUP_BYTES, 0, 1);

				float	  xd[NR];
				int8x16_t xq_lo[NR];
				int8x16_t xq_hi[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				const uint16_t *d_ptr  = (const uint16_t *)blk;
				const int8_t   *qs_ptr = (const int8_t *)(blk + Q8_0_R8_ROWS * sizeof(uint16_t));

				for (int r = 0; r < MR; r++) {
					const float		d_w = f16_to_f32_fast(d_ptr[r]);
					const int8x16_t lo	= vld1q_s8(qs_ptr + (size_t)r * 32);
					const int8x16_t hi	= vld1q_s8(qs_ptr + (size_t)r * 32 + 16);

					const int32x4_t	  sumi4	 = q4_dot_x4_sumi4(lo, hi, xq_lo, xq_hi);
					const float32x4_t sumi_f = vcvtq_f32_s32(sumi4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(sumi_f, d_w));
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_q8_0_r8_q8_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
											y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}
}

#undef NR

#define IQ4_NL_R8_GROUP_BYTES (IQ4_NL_R8_ROWS * sizeof(uint16_t) + IQ4_NL_R8_ROWS * 16)

static void matmul_iq4_nl_r8_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
											  float *restrict y, int n, int k) {
	const int		 blocks_per_row = k / 32;
	const size_t	 row_stride		= (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const uint8_t	*Wb				= w;
	const uint8x16_t kvalues_u		= vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t lo_mask		= vdupq_n_u8(0x0F);
	int				 i				= 0;

	for (; i + MR <= n; i += MR) {
		float32x4_t acc_lo = vdupq_n_f32(0.0f);
		float32x4_t acc_hi = vdupq_n_f32(0.0f);

		const uint8_t *group = Wb + ((size_t)i * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t *blk = group + (size_t)bi * IQ4_NL_R8_GROUP_BYTES;

			if (bi + 1 < blocks_per_row)
				__builtin_prefetch(blk + IQ4_NL_R8_GROUP_BYTES, 0, 2);

			const float		d_xq = f16_to_f32_fast(xq[bi].d);
			const int8x16_t xl	 = vld1q_s8(xq[bi].qs);
			const int8x16_t xh	 = vld1q_s8(xq[bi].qs + 16);

			const uint16_t *d_ptr  = (const uint16_t *)blk;
			const uint8_t  *qs_ptr = blk + IQ4_NL_R8_ROWS * sizeof(uint16_t);

			int32x4_t acc[8];
			for (int r = 0; r < 8; r++) {
				const uint8x16_t q = vld1q_u8(qs_ptr + (size_t)r * 16);
				const int8x16_t	 lo =
					vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vandq_u8(q, lo_mask)));
				const int8x16_t hi = vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vshrq_n_u8(q, 4)));

#if defined(__ARM_FEATURE_DOTPROD)
				int32x4_t a = vdotq_s32(vdupq_n_s32(0), lo, xl);
				a			= vdotq_s32(a, hi, xh);
#else
				int16x8_t p0 = vmull_s8(vget_low_s8(lo), vget_low_s8(xl));
				int16x8_t p1 = vmull_s8(vget_high_s8(lo), vget_high_s8(xl));
				int16x8_t p2 = vmull_s8(vget_low_s8(hi), vget_low_s8(xh));
				int16x8_t p3 = vmull_s8(vget_high_s8(hi), vget_high_s8(xh));
				int32x4_t a	 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
				a			 = vaddq_s32(a, vpaddlq_s16(p2));
				a			 = vaddq_s32(a, vpaddlq_s16(p3));
#endif
				acc[r] = a;
			}

			const int32x4_t sum01	= vpaddq_s32(acc[0], acc[1]);
			const int32x4_t sum23	= vpaddq_s32(acc[2], acc[3]);
			const int32x4_t sumi_lo = vpaddq_s32(sum01, sum23);

			const int32x4_t sum45	= vpaddq_s32(acc[4], acc[5]);
			const int32x4_t sum67	= vpaddq_s32(acc[6], acc[7]);
			const int32x4_t sumi_hi = vpaddq_s32(sum45, sum67);

			const uint16x8_t d_w_u16 = vld1q_u16(d_ptr);
			float32x4_t		 d_w0	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(d_w_u16)));
			float32x4_t		 d_w1	 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(d_w_u16)));
			float32x4_t		 d_xq_v	 = vdupq_n_f32(d_xq);

			acc_lo = vfmaq_f32(acc_lo, vmulq_f32(d_w0, d_xq_v), vcvtq_f32_s32(sumi_lo));
			acc_hi = vfmaq_f32(acc_hi, vmulq_f32(d_w1, d_xq_v), vcvtq_f32_s32(sumi_hi));
		}

		float tmp0[4], tmp1[4];
		vst1q_f32(tmp0, acc_lo);
		vst1q_f32(tmp1, acc_hi);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_iq4_nl_r8_q8_qonly_f32_i8mm(const void *w, const q8_0_block *restrict xq,
											   size_t	   xq_row_stride_blocks, float *restrict y,
											   int y_row_stride, int n, int k, int m) {
	const int		 blocks_per_row = k / 32;
	const size_t	 row_stride		= (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const uint8_t	*Wb				= w;
	const uint8x16_t kvalues_u		= vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t lo_mask		= vdupq_n_u8(0x0F);
	int				 i				= 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int t = 0;
		for (; t + I8MM_NR <= m; t += I8MM_NR) {
			float32x4_t facc[MR / 2][I8MM_NR / 2];
			for (int p = 0; p < MR / 2; p++)
				for (int c = 0; c < I8MM_NR / 2; c++)
					facc[p][c] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[I8MM_NR];
			for (int c = 0; c < I8MM_NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * IQ4_NL_R8_GROUP_BYTES;

				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + IQ4_NL_R8_GROUP_BYTES, 0, 2);

				int8x16_t xq_lo[I8MM_NR];
				int8x16_t xq_hi[I8MM_NR];
				float	  xd[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}

				int8x16_t bvec[I8MM_NR / 2][4];
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					const int ca = 2 * cp;
					const int cb = 2 * cp + 1;
					bvec[cp][0]	 = vcombine_s8(vget_low_s8(xq_lo[ca]), vget_low_s8(xq_lo[cb]));
					bvec[cp][1]	 = vcombine_s8(vget_high_s8(xq_lo[ca]), vget_high_s8(xq_lo[cb]));
					bvec[cp][2]	 = vcombine_s8(vget_low_s8(xq_hi[ca]), vget_low_s8(xq_hi[cb]));
					bvec[cp][3]	 = vcombine_s8(vget_high_s8(xq_hi[ca]), vget_high_s8(xq_hi[cb]));
				}

				const uint16_t *d_ptr  = (const uint16_t *)blk;
				const uint8_t  *qs_ptr = blk + IQ4_NL_R8_ROWS * sizeof(uint16_t);

				for (int p = 0; p < MR / 2; p++) {
					const float dwa = f16_to_f32_fast(d_ptr[2 * p]);
					const float dwb = f16_to_f32_fast(d_ptr[2 * p + 1]);

					const uint8x16_t q0		 = vld1q_u8(qs_ptr + (size_t)(2 * p) * 16);
					const uint8x16_t q1		 = vld1q_u8(qs_ptr + (size_t)(2 * p + 1) * 16);
					const uint8x16_t l0		 = vandq_u8(q0, lo_mask);
					const uint8x16_t h0		 = vshrq_n_u8(q0, 4);
					const uint8x16_t l1		 = vandq_u8(q1, lo_mask);
					const uint8x16_t h1		 = vshrq_n_u8(q1, 4);
					const int8x16_t	 avec[4] = {
						vreinterpretq_s8_u8(
							vqtbl1q_u8(kvalues_u, vcombine_u8(vget_low_u8(l0), vget_low_u8(l1)))),
						vreinterpretq_s8_u8(
							vqtbl1q_u8(kvalues_u, vcombine_u8(vget_high_u8(l0), vget_high_u8(l1)))),
						vreinterpretq_s8_u8(
							vqtbl1q_u8(kvalues_u, vcombine_u8(vget_low_u8(h0), vget_low_u8(h1)))),
						vreinterpretq_s8_u8(
							vqtbl1q_u8(kvalues_u, vcombine_u8(vget_high_u8(h0), vget_high_u8(h1)))),
					};

					const float32x4_t srow = vcombine_f32(vdup_n_f32(dwa), vdup_n_f32(dwb));

					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						const float32x4_t dcol =
							vzip1q_f32(vdupq_n_f32(xd[2 * cp]), vdupq_n_f32(xd[2 * cp + 1]));
						int32x4_t s = vdupq_n_s32(0);
						s			= vmmlaq_s32(s, avec[0], bvec[cp][0]);
						s			= vmmlaq_s32(s, avec[1], bvec[cp][1]);
						s			= vmmlaq_s32(s, avec[2], bvec[cp][2]);
						s			= vmmlaq_s32(s, avec[3], bvec[cp][3]);
						facc[p][cp] =
							vfmaq_f32(facc[p][cp], vcvtq_f32_s32(s), vmulq_f32(srow, dcol));
					}
				}
			}

			for (int p = 0; p < MR / 2; p++) {
				for (int cp = 0; cp < I8MM_NR / 2; cp++) {
					float tmp[4];
					vst1q_f32(tmp, facc[p][cp]);
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
					y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
					y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
				}
			}
		}

		for (; t < m; t++) {
			matmul_iq4_nl_r8_q8_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
											  y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}
}
#endif

#define NR 4

void matmul_iq4_nl_r8_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								   size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								   int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_iq4_nl_r8_q8_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int		 blocks_per_row = k / 32;
	const size_t	 row_stride		= (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const uint8_t	*Wb				= w;
	const uint8x16_t kvalues_u		= vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t lo_mask		= vdupq_n_u8(0x0F);
	int				 i				= 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * IQ4_NL_R8_GROUP_BYTES;

				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + IQ4_NL_R8_GROUP_BYTES, 0, 2);

				float	  xd[NR];
				int8x16_t xq_lo[NR];
				int8x16_t xq_hi[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq_lo[c] = vld1q_s8(xrow[c][bi].qs);
					xq_hi[c] = vld1q_s8(xrow[c][bi].qs + 16);
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				const uint16_t *d_ptr  = (const uint16_t *)blk;
				const uint8_t  *qs_ptr = blk + IQ4_NL_R8_ROWS * sizeof(uint16_t);

				for (int r = 0; r < MR; r++) {
					const float		 d_w = f16_to_f32_fast(d_ptr[r]);
					const uint8x16_t q	 = vld1q_u8(qs_ptr + (size_t)r * 16);
					const int8x16_t	 lo =
						vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vandq_u8(q, lo_mask)));
					const int8x16_t hi =
						vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, vshrq_n_u8(q, 4)));

					const int32x4_t	  sumi4	 = q4_dot_x4_sumi4(lo, hi, xq_lo, xq_hi);
					const float32x4_t sumi_f = vcvtq_f32_s32(sumi4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(sumi_f, d_w));
				}
			}

			store_acc_row_mr_nr(acc_row, y, y_row_stride, i, t, NR);
		}

		for (; t < m; t++) {
			matmul_iq4_nl_r8_q8_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
											  y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}
}

#undef NR

#define IQ3S_RE_OFF_D 0

#define IQ3S_RE_OFF_SCALES 2

#define IQ3S_RE_OFF_IDX 6

static const int8_t iq3s_re_decode_tbl[16] = {
	1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15,
};

static inline void iq3s_re_unpack_group(const uint8_t *idx_ptr, uint8x16_t tbl, int8x16_t *out0,
										int8x16_t *out1) {
	uint8x16_t packed = vld1q_u8(idx_ptr);
	uint8x16_t lo	  = vandq_u8(packed, vdupq_n_u8(0x0F));
	uint8x16_t hi	  = vshrq_n_u8(packed, 4);

	*out0 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, lo));
	*out1 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, hi));
}

static void matmul_iq3_s_re_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
											   float *restrict y, int n, int k) {
	const int	   blocks_per_row = k / 256;
	const size_t   row_stride	  = (size_t)blocks_per_row * IQ3_S_RE_BLOCK_BYTES;
	const uint8_t *Wb			  = w;

	uint8x16_t tbl = vld1q_u8((const uint8_t *)iq3s_re_decode_tbl);
	int		   i   = 0;

	for (; i + 8 <= n; i += 8) {
		float32x4_t acc0 = vdupq_n_f32(0.0f);
		float32x4_t acc1 = vdupq_n_f32(0.0f);

		const uint8_t *row_base[8];
		for (int r = 0; r < 8; r++)
			row_base[r] = Wb + (size_t)(i + r) * row_stride;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float d_xq			  = yb->d;
			const int8_t *restrict q8	  = yb->qs;

			if (bi + 2 < blocks_per_row) {
				for (int r = 0; r < 8; r++)
					__builtin_prefetch(row_base[r] + (size_t)(bi + 2) * IQ3_S_RE_BLOCK_BYTES, 0, 1);
			}

			int32_t sumi_lane[8];
			float	d_w[8];

			for (int r = 0; r < 8; r++) {
				const uint8_t *blk	  = row_base[r] + (size_t)bi * IQ3_S_RE_BLOCK_BYTES;
				d_w[r]				  = f16_to_f32_fast(*(const uint16_t *)(blk + IQ3S_RE_OFF_D));
				const uint8_t *scales = blk + IQ3S_RE_OFF_SCALES;
				const uint8_t *idx	  = blk + IQ3S_RE_OFF_IDX;

				int32x4_t acc = vdupq_n_s32(0);
				for (int g = 0; g < 8; g += 2) {
					uint8_t sb	= scales[g / 2];
					int		sc0 = 1 + 2 * (sb & 0xf);
					int		sc1 = 1 + 2 * (sb >> 4);

					int8x16_t d00, d01, d10, d11;
					iq3s_re_unpack_group(idx + g * 16, tbl, &d00, &d01);
					iq3s_re_unpack_group(idx + (g + 1) * 16, tbl, &d10, &d11);

					const int8_t *restrict qg0 = q8 + g * 32;
					const int8_t *restrict qg1 = q8 + (g + 1) * 32;

					int32x4_t a0, a1;
					iq3s_pair_dot(d00, d01, d10, d11, qg0, qg1, &a0, &a1);
					acc = vmlaq_s32(acc, a0, vdupq_n_s32(sc0));
					acc = vmlaq_s32(acc, a1, vdupq_n_s32(sc1));
				}
				sumi_lane[r] = vaddvq_s32(acc);
			}

			int32x4_t	sumi0  = vld1q_s32(sumi_lane);
			int32x4_t	sumi1  = vld1q_s32(sumi_lane + 4);
			float32x4_t sumi0f = vcvtq_f32_s32(sumi0);
			float32x4_t sumi1f = vcvtq_f32_s32(sumi1);
			float32x4_t d_w0   = vld1q_f32(d_w);
			float32x4_t d_w1   = vld1q_f32(d_w + 4);
			float32x4_t d_xq_v = vdupq_n_f32(d_xq);

			acc0 = vfmaq_f32(acc0, vmulq_f32(d_w0, d_xq_v), sumi0f);
			acc1 = vfmaq_f32(acc1, vmulq_f32(d_w1, d_xq_v), sumi1f);
		}

		store_acc8(acc0, acc1, y, i);
	}

	for (; i < n; i++) {
		const uint8_t *row	= Wb + (size_t)i * row_stride;
		float		   sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t *blk	  = row + (size_t)bi * IQ3_S_RE_BLOCK_BYTES;
			float		   d_w	  = f16_to_f32_fast(*(const uint16_t *)(blk + IQ3S_RE_OFF_D));
			const uint8_t *scales = blk + IQ3S_RE_OFF_SCALES;
			const uint8_t *idx	  = blk + IQ3S_RE_OFF_IDX;

			const q8_k_block *restrict yb = &xq[bi];
			const int8_t *restrict q8	  = yb->qs;
			int32x4_t acc				  = vdupq_n_s32(0);

			for (int g = 0; g < 8; g += 2) {
				uint8_t sb	= scales[g / 2];
				int		sc0 = 1 + 2 * (sb & 0xf);
				int		sc1 = 1 + 2 * (sb >> 4);

				int8x16_t d00, d01, d10, d11;
				iq3s_re_unpack_group(idx + g * 16, tbl, &d00, &d01);
				iq3s_re_unpack_group(idx + (g + 1) * 16, tbl, &d10, &d11);

				const int8_t *restrict qg0 = q8 + g * 32;
				const int8_t *restrict qg1 = q8 + (g + 1) * 32;

				int32x4_t a0, a1;
				iq3s_pair_dot(d00, d01, d10, d11, qg0, qg1, &a0, &a1);
				acc = vmlaq_s32(acc, a0, vdupq_n_s32(sc0));
				acc = vmlaq_s32(acc, a1, vdupq_n_s32(sc1));
			}
			sumf += d_w * yb->d * (float)vaddvq_s32(acc);
		}
		y[i] = sumf;
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_iq3_s_re_q8_k_qonly_f32_i8mm(const void *w, const q8_k_block *restrict xq,
												size_t		xq_row_stride_blocks, float *restrict y,
												int y_row_stride, int n, int k, int m) {
	const int	   blocks_per_row = k / 256;
	const size_t   row_stride	  = (size_t)blocks_per_row * IQ3_S_RE_BLOCK_BYTES;
	const uint8_t *Wb			  = w;
	uint8x16_t	   tbl			  = vld1q_u8((const uint8_t *)iq3s_re_decode_tbl);
	int			   i			  = 0;

	int8x16_t(*decoded_cache)[MR][16]	 = NULL;
	int8x16_t(*pdec_cache)[MR / 2][8][4] = NULL;
	int32_t (*sc0_cache)[MR][4]			 = NULL;
	int32_t (*sc1_cache)[MR][4]			 = NULL;
	int32x4_t(*sv_cache)[MR / 2][4][2]	 = NULL;
	float (*d_w_cache)[MR]				 = NULL;

	if (blocks_per_row > 0 && m >= I8MM_NR) {
		decoded_cache = malloc(sizeof(*decoded_cache) * blocks_per_row);
		pdec_cache	  = malloc(sizeof(*pdec_cache) * blocks_per_row);
		sc0_cache	  = malloc(sizeof(*sc0_cache) * blocks_per_row);
		sc1_cache	  = malloc(sizeof(*sc1_cache) * blocks_per_row);
		sv_cache	  = malloc(sizeof(*sv_cache) * blocks_per_row);
		d_w_cache	  = malloc(sizeof(*d_w_cache) * blocks_per_row);
		if (!decoded_cache || !pdec_cache || !sc0_cache || !sc1_cache || !sv_cache || !d_w_cache) {
			free(decoded_cache);
			free(pdec_cache);
			free(sc0_cache);
			free(sc1_cache);
			free(sv_cache);
			free(d_w_cache);
			decoded_cache = NULL;
		}
	}

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = Wb + (size_t)(i + r) * row_stride;

		const int n_bi_tiles = decoded_cache ? blocks_per_row : 0;

		if (decoded_cache) {
			for (int bi = 0; bi < n_bi_tiles; bi++) {
				for (int r = 0; r < MR; r++) {
					const uint8_t *blk = row_base[r] + (size_t)bi * IQ3_S_RE_BLOCK_BYTES;
					d_w_cache[bi][r]   = f16_to_f32_fast(*(const uint16_t *)(blk + IQ3S_RE_OFF_D));
					const uint8_t *scales = blk + IQ3S_RE_OFF_SCALES;
					const uint8_t *idx	  = blk + IQ3S_RE_OFF_IDX;

					for (int g = 0; g < 8; g += 2) {
						uint8_t sb				= scales[g / 2];
						sc0_cache[bi][r][g / 2] = 1 + 2 * (sb & 0xf);
						sc1_cache[bi][r][g / 2] = 1 + 2 * (sb >> 4);

						const int vi = g * 2;
						iq3s_re_unpack_group(idx + g * 16, tbl, &decoded_cache[bi][r][vi],
											 &decoded_cache[bi][r][vi + 1]);
						iq3s_re_unpack_group(idx + (g + 1) * 16, tbl, &decoded_cache[bi][r][vi + 2],
											 &decoded_cache[bi][r][vi + 3]);
					}
				}

				for (int p = 0; p < MR / 2; p++) {
					const int8x16_t *da = decoded_cache[bi][2 * p];
					const int8x16_t *db = decoded_cache[bi][2 * p + 1];
					for (int s = 0; s < 8; s++) {
						pdec_cache[bi][p][s][0] =
							vcombine_s8(vget_low_s8(da[2 * s]), vget_low_s8(db[2 * s]));
						pdec_cache[bi][p][s][1] =
							vcombine_s8(vget_high_s8(da[2 * s]), vget_high_s8(db[2 * s]));
						pdec_cache[bi][p][s][2] =
							vcombine_s8(vget_low_s8(da[2 * s + 1]), vget_low_s8(db[2 * s + 1]));
						pdec_cache[bi][p][s][3] =
							vcombine_s8(vget_high_s8(da[2 * s + 1]), vget_high_s8(db[2 * s + 1]));
					}
					for (int gi = 0; gi < 4; gi++) {
						sv_cache[bi][p][gi][0] =
							vcombine_s32(vdup_n_s32(sc0_cache[bi][2 * p][gi]),
										 vdup_n_s32(sc0_cache[bi][2 * p + 1][gi]));
						sv_cache[bi][p][gi][1] =
							vcombine_s32(vdup_n_s32(sc1_cache[bi][2 * p][gi]),
										 vdup_n_s32(sc1_cache[bi][2 * p + 1][gi]));
					}
				}
			}
		}

		int t = 0;
		if (decoded_cache) {
			for (; t + I8MM_NR <= m; t += I8MM_NR) {
				float32x4_t facc[MR / 2][I8MM_NR / 2];
				for (int p = 0; p < MR / 2; p++)
					for (int c = 0; c < I8MM_NR / 2; c++)
						facc[p][c] = vdupq_n_f32(0.0f);

				const q8_k_block *xrow[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++)
					xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

				for (int bi = 0; bi < blocks_per_row; bi++) {
					const int8_t *restrict q8p[I8MM_NR];
					for (int c = 0; c < I8MM_NR; c++)
						q8p[c] = xrow[c][bi].qs;

					float32x4_t dcol[I8MM_NR / 2];
					float32x4_t drowv[MR / 2];
					for (int cp = 0; cp < I8MM_NR / 2; cp++)
						dcol[cp] = vzip1q_f32(vdupq_n_f32(xrow[2 * cp][bi].d),
											  vdupq_n_f32(xrow[2 * cp + 1][bi].d));
					for (int p = 0; p < MR / 2; p++)
						drowv[p] = vcombine_f32(vdup_n_f32(d_w_cache[bi][2 * p]),
												vdup_n_f32(d_w_cache[bi][2 * p + 1]));

					for (int gi = 0; gi < 4; gi++) {
						for (int cp = 0; cp < I8MM_NR / 2; cp++) {
							const int8_t *ea = q8p[2 * cp] + (2 * gi) * 32;
							const int8_t *eb = q8p[2 * cp + 1] + (2 * gi) * 32;
							const int8_t *oa = ea + 32;
							const int8_t *ob = eb + 32;

							const int8x16_t eal = vld1q_s8(ea);
							const int8x16_t eah = vld1q_s8(ea + 16);
							const int8x16_t ebl = vld1q_s8(eb);
							const int8x16_t ebh = vld1q_s8(eb + 16);
							int8x16_t		BE[4];
							BE[0] = vcombine_s8(vget_low_s8(eal), vget_low_s8(ebl));
							BE[1] = vcombine_s8(vget_high_s8(eal), vget_high_s8(ebl));
							BE[2] = vcombine_s8(vget_low_s8(eah), vget_low_s8(ebh));
							BE[3] = vcombine_s8(vget_high_s8(eah), vget_high_s8(ebh));

							const int8x16_t oal = vld1q_s8(oa);
							const int8x16_t oah = vld1q_s8(oa + 16);
							const int8x16_t obl = vld1q_s8(ob);
							const int8x16_t obh = vld1q_s8(ob + 16);
							int8x16_t		BO[4];
							BO[0] = vcombine_s8(vget_low_s8(oal), vget_low_s8(obl));
							BO[1] = vcombine_s8(vget_high_s8(oal), vget_high_s8(obl));
							BO[2] = vcombine_s8(vget_low_s8(oah), vget_low_s8(obh));
							BO[3] = vcombine_s8(vget_high_s8(oah), vget_high_s8(obh));

							for (int p = 0; p < MR / 2; p++) {
								const int8x16_t *pe = pdec_cache[bi][p][2 * gi];
								const int8x16_t *po = pdec_cache[bi][p][2 * gi + 1];

								int32x4_t tE = vmmlaq_s32(
									vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), pe[0], BE[0]),
														  pe[1], BE[1]),
											   pe[2], BE[2]),
									pe[3], BE[3]);
								int32x4_t tO = vmmlaq_s32(
									vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), po[0], BO[0]),
														  po[1], BO[1]),
											   po[2], BO[2]),
									po[3], BO[3]);

								const int32x4_t sumi =
									vmlaq_s32(vmulq_s32(tE, sv_cache[bi][p][gi][0]), tO,
											  sv_cache[bi][p][gi][1]);

								facc[p][cp] = vfmaq_f32(facc[p][cp], vcvtq_f32_s32(sumi),
														vmulq_f32(drowv[p], dcol[cp]));
							}
						}
					}
				}

				for (int p = 0; p < MR / 2; p++) {
					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						float tmp[4];
						vst1q_f32(tmp, facc[p][cp]);
						y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
						y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
						y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
						y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
					}
				}
			}
		}

		for (; t < m; t++) {
			matmul_iq3_s_re_q8_k_qonly_f32_row(Wb + (size_t)i * row_stride,
											   xq + ((size_t)t * xq_row_stride_blocks),
											   y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		for (int t = 0; t < m; t++) {
			matmul_iq3_s_re_q8_k_qonly_f32_row(Wb + (size_t)i * row_stride,
											   xq + ((size_t)t * xq_row_stride_blocks),
											   y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}

	free(decoded_cache);
	free(pdec_cache);
	free(sc0_cache);
	free(sc1_cache);
	free(sv_cache);
	free(d_w_cache);
}
#endif

#define NR_IQ3S_RE 4

void matmul_iq3_s_re_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
									size_t		xq_row_stride_blocks, float *restrict y,
									int y_row_stride, int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_iq3_s_re_q8_k_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int	   blocks_per_row = k / 256;
	const size_t   row_stride	  = (size_t)blocks_per_row * IQ3_S_RE_BLOCK_BYTES;
	const uint8_t *Wb			  = w;
	uint8x16_t	   tbl			  = vld1q_u8((const uint8_t *)iq3s_re_decode_tbl);
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = Wb + (size_t)(i + r) * row_stride;

		const int n_bi_tiles = (m / NR_IQ3S_RE) > 0 ? blocks_per_row : 0;

		static _Thread_local int8x16_t(*decoded_cache)[MR][16] = NULL;
		static _Thread_local int32_t (*sc0_cache)[MR][4]	   = NULL;
		static _Thread_local int32_t (*sc1_cache)[MR][4]	   = NULL;
		static _Thread_local float (*d_w_cache)[MR]			   = NULL;
		static _Thread_local int cache_cap					   = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				decoded_cache = realloc(decoded_cache, sizeof(*decoded_cache) * n_bi_tiles);
				sc0_cache	  = realloc(sc0_cache, sizeof(*sc0_cache) * n_bi_tiles);
				sc1_cache	  = realloc(sc1_cache, sizeof(*sc1_cache) * n_bi_tiles);
				d_w_cache	  = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				cache_cap	  = n_bi_tiles;
				tlocal_register((void **)&decoded_cache);
				tlocal_register((void **)&sc0_cache);
				tlocal_register((void **)&sc1_cache);
				tlocal_register((void **)&d_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				for (int r = 0; r < MR; r++) {
					const uint8_t *blk = row_base[r] + (size_t)bi * IQ3_S_RE_BLOCK_BYTES;
					d_w_cache[bi][r]   = f16_to_f32_fast(*(const uint16_t *)(blk + IQ3S_RE_OFF_D));
					const uint8_t *scales = blk + IQ3S_RE_OFF_SCALES;
					const uint8_t *idx	  = blk + IQ3S_RE_OFF_IDX;

					for (int g = 0; g < 8; g += 2) {
						uint8_t sb				= scales[g / 2];
						sc0_cache[bi][r][g / 2] = 1 + 2 * (sb & 0xf);
						sc1_cache[bi][r][g / 2] = 1 + 2 * (sb >> 4);

						const int vi = g * 2;
						iq3s_re_unpack_group(idx + g * 16, tbl, &decoded_cache[bi][r][vi + 0],
											 &decoded_cache[bi][r][vi + 1]);
						iq3s_re_unpack_group(idx + (g + 1) * 16, tbl, &decoded_cache[bi][r][vi + 2],
											 &decoded_cache[bi][r][vi + 3]);
					}
				}
			}
		}

		int t = 0;
		for (; t + NR_IQ3S_RE <= m; t += NR_IQ3S_RE) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_k_block *xrow[NR_IQ3S_RE];
			for (int c = 0; c < NR_IQ3S_RE; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {

				float(*d_w)				= d_w_cache[bi];
				int8x16_t(*decoded)[16] = decoded_cache[bi];
				int32_t (*sc0)[4]		= sc0_cache[bi];
				int32_t (*sc1)[4]		= sc1_cache[bi];

				float xd[NR_IQ3S_RE];
				const int8_t *restrict q8p[NR_IQ3S_RE];
				for (int c = 0; c < NR_IQ3S_RE; c++) {
					xd[c]  = xrow[c][bi].d;
					q8p[c] = xrow[c][bi].qs;
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				for (int r = 0; r < MR; r++) {
					int32x4_t total4 = vdupq_n_s32(0);

					for (int g = 0; g < 8; g += 2) {
						const int		vi	= g * 2;
						const int8x16_t c00 = decoded[r][vi + 0];
						const int8x16_t c01 = decoded[r][vi + 1];
						const int8x16_t c10 = decoded[r][vi + 2];
						const int8x16_t c11 = decoded[r][vi + 3];

						const int8_t *restrict qg0_0 = q8p[0] + g * 32;
						const int8_t *restrict qg0_1 = q8p[1] + g * 32;
						const int8_t *restrict qg0_2 = q8p[2] + g * 32;
						const int8_t *restrict qg0_3 = q8p[3] + g * 32;

						const int8_t *restrict qg1_0 = q8p[0] + (g + 1) * 32;
						const int8_t *restrict qg1_1 = q8p[1] + (g + 1) * 32;
						const int8_t *restrict qg1_2 = q8p[2] + (g + 1) * 32;
						const int8_t *restrict qg1_3 = q8p[3] + (g + 1) * 32;

#if defined(__ARM_FEATURE_DOTPROD)
						int32x4_t a0_c0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_0)),
													c01, vld1q_s8(qg0_0 + 16));
						int32x4_t a0_c1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_1)),
													c01, vld1q_s8(qg0_1 + 16));
						int32x4_t a0_c2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_2)),
													c01, vld1q_s8(qg0_2 + 16));
						int32x4_t a0_c3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_3)),
													c01, vld1q_s8(qg0_3 + 16));

						int32x4_t a1_c0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_0)),
													c11, vld1q_s8(qg1_0 + 16));
						int32x4_t a1_c1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_1)),
													c11, vld1q_s8(qg1_1 + 16));
						int32x4_t a1_c2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_2)),
													c11, vld1q_s8(qg1_2 + 16));
						int32x4_t a1_c3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_3)),
													c11, vld1q_s8(qg1_3 + 16));
#else
						int16x8_t p00_0 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_0)));
						int16x8_t p01_0 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_0)));
						int16x8_t p02_0 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_0 + 16)));
						int16x8_t p03_0 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_0 + 16)));
						int32x4_t a0_c0 = vpaddlq_s16(p00_0);
						a0_c0			= vpadalq_s16(a0_c0, p01_0);
						a0_c0			= vpadalq_s16(a0_c0, p02_0);
						a0_c0			= vpadalq_s16(a0_c0, p03_0);

						int16x8_t p00_1 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_1)));
						int16x8_t p01_1 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_1)));
						int16x8_t p02_1 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_1 + 16)));
						int16x8_t p03_1 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_1 + 16)));
						int32x4_t a0_c1 = vpaddlq_s16(p00_1);
						a0_c1			= vpadalq_s16(a0_c1, p01_1);
						a0_c1			= vpadalq_s16(a0_c1, p02_1);
						a0_c1			= vpadalq_s16(a0_c1, p03_1);

						int16x8_t p00_2 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_2)));
						int16x8_t p01_2 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_2)));
						int16x8_t p02_2 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_2 + 16)));
						int16x8_t p03_2 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_2 + 16)));
						int32x4_t a0_c2 = vpaddlq_s16(p00_2);
						a0_c2			= vpadalq_s16(a0_c2, p01_2);
						a0_c2			= vpadalq_s16(a0_c2, p02_2);
						a0_c2			= vpadalq_s16(a0_c2, p03_2);

						int16x8_t p00_3 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_3)));
						int16x8_t p01_3 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_3)));
						int16x8_t p02_3 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_3 + 16)));
						int16x8_t p03_3 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_3 + 16)));
						int32x4_t a0_c3 = vpaddlq_s16(p00_3);
						a0_c3			= vpadalq_s16(a0_c3, p01_3);
						a0_c3			= vpadalq_s16(a0_c3, p02_3);
						a0_c3			= vpadalq_s16(a0_c3, p03_3);

						int16x8_t p10_0 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_0)));
						int16x8_t p11_0 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_0)));
						int16x8_t p12_0 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_0 + 16)));
						int16x8_t p13_0 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_0 + 16)));
						int32x4_t a1_c0 = vpaddlq_s16(p10_0);
						a1_c0			= vpadalq_s16(a1_c0, p11_0);
						a1_c0			= vpadalq_s16(a1_c0, p12_0);
						a1_c0			= vpadalq_s16(a1_c0, p13_0);

						int16x8_t p10_1 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_1)));
						int16x8_t p11_1 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_1)));
						int16x8_t p12_1 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_1 + 16)));
						int16x8_t p13_1 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_1 + 16)));
						int32x4_t a1_c1 = vpaddlq_s16(p10_1);
						a1_c1			= vpadalq_s16(a1_c1, p11_1);
						a1_c1			= vpadalq_s16(a1_c1, p12_1);
						a1_c1			= vpadalq_s16(a1_c1, p13_1);

						int16x8_t p10_2 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_2)));
						int16x8_t p11_2 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_2)));
						int16x8_t p12_2 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_2 + 16)));
						int16x8_t p13_2 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_2 + 16)));
						int32x4_t a1_c2 = vpaddlq_s16(p10_2);
						a1_c2			= vpadalq_s16(a1_c2, p11_2);
						a1_c2			= vpadalq_s16(a1_c2, p12_2);
						a1_c2			= vpadalq_s16(a1_c2, p13_2);

						int16x8_t p10_3 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_3)));
						int16x8_t p11_3 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_3)));
						int16x8_t p12_3 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_3 + 16)));
						int16x8_t p13_3 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_3 + 16)));
						int32x4_t a1_c3 = vpaddlq_s16(p10_3);
						a1_c3			= vpadalq_s16(a1_c3, p11_3);
						a1_c3			= vpadalq_s16(a1_c3, p12_3);
						a1_c3			= vpadalq_s16(a1_c3, p13_3);
#endif
						int32x4_t a0_s01	= vpaddq_s32(a0_c0, a0_c1);
						int32x4_t a0_s23	= vpaddq_s32(a0_c2, a0_c3);
						int32x4_t a0_packed = vpaddq_s32(a0_s01, a0_s23);

						int32x4_t a1_s01	= vpaddq_s32(a1_c0, a1_c1);
						int32x4_t a1_s23	= vpaddq_s32(a1_c2, a1_c3);
						int32x4_t a1_packed = vpaddq_s32(a1_s01, a1_s23);

						total4 = vmlaq_n_s32(total4, a0_packed, sc0[r][g / 2]);
						total4 = vmlaq_n_s32(total4, a1_packed, sc1[r][g / 2]);
					}

					float32x4_t total4f = vcvtq_f32_s32(total4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(total4f, d_w[r]));
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				vst1q_f32(tmp, acc_row[r]);
				for (int c = 0; c < NR_IQ3S_RE; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
		}

		for (; t < m; t++) {
			const q8_k_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			 *yrow = y + ((size_t)t * y_row_stride) + i;
			matmul_iq3_s_re_q8_k_qonly_f32_row(Wb + (size_t)i * row_stride, xrow, yrow, MR, k);
		}
	}

	for (; i < n; i++) {
		for (int t = 0; t < m; t++) {
			const q8_k_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			 *yrow = y + ((size_t)t * y_row_stride) + i;
			matmul_iq3_s_re_q8_k_qonly_f32_row(Wb + (size_t)i * row_stride, xrow, yrow, 1, k);
		}
	}
}

#undef NR_IQ3S_RE

#define IQ3S_RE8_OFF_D 0

#define IQ3S_RE8_OFF_SCALES (IQ3_S_RE8_ROWS * sizeof(uint16_t))

#define IQ3S_RE8_OFF_IDX (IQ3S_RE8_OFF_SCALES + IQ3_S_RE8_ROWS * 4)

static void matmul_iq3_s_re8_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
												float *restrict y, int n, int k) {
	const int	   blocks_per_row = k / 256;
	const size_t   row_stride	  = (size_t)blocks_per_row * IQ3_S_RE8_GROUP_BYTES;
	const uint8_t *Wb			  = w;
	uint8x16_t	   tbl			  = vld1q_u8((const uint8_t *)iq3s_re_decode_tbl);
	int			   i			  = 0;

	const uint8x16_t mask_0F = vdupq_n_u8(0x0F);

	for (; i + 8 <= n; i += 8) {
		float32x4_t acc0 = vdupq_n_f32(0.0f);
		float32x4_t acc1 = vdupq_n_f32(0.0f);

		const uint8_t *group = Wb + (size_t)(i / 8) * row_stride;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float d_xq			  = yb->d;
			const int8_t *restrict q8	  = yb->qs;

			const uint8_t *blk = group + (size_t)bi * IQ3_S_RE8_GROUP_BYTES;

			if (bi + 2 < blocks_per_row)
				__builtin_prefetch(blk + IQ3_S_RE8_GROUP_BYTES, 0, 1);

			const uint16_t *d_ptr	   = (const uint16_t *)(blk + IQ3S_RE8_OFF_D);
			const uint8_t  *scales_all = blk + IQ3S_RE8_OFF_SCALES;
			const uint8_t  *idx_all	   = blk + IQ3S_RE8_OFF_IDX;

			int32_t sumi_lane[8];
			float	d_w[8];

			for (int r = 0; r < 8; r++) {
				d_w[r]				  = f16_to_f32_fast(d_ptr[r]);
				const uint8_t *scales = scales_all + (size_t)r * 4;
				const uint8_t *idx	  = idx_all + (size_t)r * 128;

				int32x4_t lane0 = vdupq_n_s32(0);
				int32x4_t lane1 = vdupq_n_s32(0);

				for (int g = 0; g < 8; g += 2) {
					uint8_t	  sb   = scales[g / 2];
					int32_t	  sc0  = 1 + 2 * (sb & 0xf);
					int32_t	  sc1  = 1 + 2 * (sb >> 4);
					int32x4_t sc0v = vdupq_n_s32(sc0);
					int32x4_t sc1v = vdupq_n_s32(sc1);

					uint8x16_t packed0 = vld1q_u8(idx + g * 16);
					uint8x16_t packed1 = vld1q_u8(idx + (g + 1) * 16);

					uint8x16_t lo0 = vandq_u8(packed0, mask_0F);
					uint8x16_t hi0 = vshrq_n_u8(packed0, 4);
					uint8x16_t lo1 = vandq_u8(packed1, mask_0F);
					uint8x16_t hi1 = vshrq_n_u8(packed1, 4);

					int8x16_t d00 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, lo0));
					int8x16_t d01 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, hi0));
					int8x16_t d10 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, lo1));
					int8x16_t d11 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, hi1));

					const int8_t *restrict qg0 = q8 + g * 32;
					const int8_t *restrict qg1 = q8 + (g + 1) * 32;

					int32x4_t a0, a1;
					iq3s_pair_dot(d00, d01, d10, d11, qg0, qg1, &a0, &a1);
					lane0 = vmlaq_s32(lane0, sc0v, a0);
					lane1 = vmlaq_s32(lane1, sc1v, a1);
				}
				sumi_lane[r] = vaddvq_s32(lane0) + vaddvq_s32(lane1);
			}

			int32x4_t	sumi0  = vld1q_s32(sumi_lane);
			int32x4_t	sumi1  = vld1q_s32(sumi_lane + 4);
			float32x4_t sumi0f = vcvtq_f32_s32(sumi0);
			float32x4_t sumi1f = vcvtq_f32_s32(sumi1);
			float32x4_t d_w0   = vld1q_f32(d_w);
			float32x4_t d_w1   = vld1q_f32(d_w + 4);
			float32x4_t d_xq_v = vdupq_n_f32(d_xq);

			acc0 = vfmaq_f32(acc0, vmulq_f32(d_w0, d_xq_v), sumi0f);
			acc1 = vfmaq_f32(acc1, vmulq_f32(d_w1, d_xq_v), sumi1f);
		}

		store_acc8(acc0, acc1, y, i);
	}

	for (; i < n; i++) {
		const uint8_t *group = Wb + (size_t)(i / 8) * row_stride;
		float		   sumf	 = 0.0f;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float d_xq			  = yb->d;
			const int8_t *restrict q8	  = yb->qs;

			const uint8_t  *blk		   = group + (size_t)bi * IQ3_S_RE8_GROUP_BYTES;
			const uint16_t *d_ptr	   = (const uint16_t *)(blk + IQ3S_RE8_OFF_D);
			const uint8_t  *scales_all = blk + IQ3S_RE8_OFF_SCALES;
			const uint8_t  *idx_all	   = blk + IQ3S_RE8_OFF_IDX;

			float		   d_w	  = f16_to_f32_fast(d_ptr[i % 8]);
			const uint8_t *scales = scales_all + (size_t)(i % 8) * 4;
			const uint8_t *idx	  = idx_all + (size_t)(i % 8) * 128;

			int32x4_t lane0 = vdupq_n_s32(0);
			int32x4_t lane1 = vdupq_n_s32(0);

			for (int g = 0; g < 8; g += 2) {
				uint8_t	  sb   = scales[g / 2];
				int32_t	  sc0  = 1 + 2 * (sb & 0xf);
				int32_t	  sc1  = 1 + 2 * (sb >> 4);
				int32x4_t sc0v = vdupq_n_s32(sc0);
				int32x4_t sc1v = vdupq_n_s32(sc1);

				uint8x16_t packed0 = vld1q_u8(idx + g * 16);
				uint8x16_t packed1 = vld1q_u8(idx + (g + 1) * 16);

				uint8x16_t lo0 = vandq_u8(packed0, mask_0F);
				uint8x16_t hi0 = vshrq_n_u8(packed0, 4);
				uint8x16_t lo1 = vandq_u8(packed1, mask_0F);
				uint8x16_t hi1 = vshrq_n_u8(packed1, 4);

				int8x16_t d00 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, lo0));
				int8x16_t d01 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, hi0));
				int8x16_t d10 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, lo1));
				int8x16_t d11 = vreinterpretq_s8_u8(vqtbl1q_u8(tbl, hi1));

				const int8_t *restrict qg0 = q8 + g * 32;
				const int8_t *restrict qg1 = q8 + (g + 1) * 32;

				int32x4_t a0, a1;
				iq3s_pair_dot(d00, d01, d10, d11, qg0, qg1, &a0, &a1);
				lane0 = vmlaq_s32(lane0, sc0v, a0);
				lane1 = vmlaq_s32(lane1, sc1v, a1);
			}

			int32_t total = vaddvq_s32(lane0) + vaddvq_s32(lane1);
			sumf += d_w * d_xq * (float)total;
		}
		y[i] = sumf;
	}
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_iq3_s_re8_q8_k_qonly_f32_i8mm(const void *w, const q8_k_block *restrict xq,
												 size_t xq_row_stride_blocks, float *restrict y,
												 int y_row_stride, int n, int k, int m) {
	const int	   blocks_per_row = k / 256;
	const size_t   row_stride	  = (size_t)blocks_per_row * IQ3_S_RE8_GROUP_BYTES;
	const uint8_t *Wb			  = w;
	uint8x16_t	   tbl			  = vld1q_u8((const uint8_t *)iq3s_re_decode_tbl);
	int			   i			  = 0;

	int8x16_t(*decoded_cache)[MR][16]	 = NULL;
	int8x16_t(*pdec_cache)[MR / 2][8][4] = NULL;
	int32_t (*sc0_cache)[MR][4]			 = NULL;
	int32_t (*sc1_cache)[MR][4]			 = NULL;
	int32x4_t(*sv_cache)[MR / 2][4][2]	 = NULL;
	float (*d_w_cache)[MR]				 = NULL;

	if (blocks_per_row > 0 && m >= I8MM_NR) {
		decoded_cache = malloc(sizeof(*decoded_cache) * blocks_per_row);
		pdec_cache	  = malloc(sizeof(*pdec_cache) * blocks_per_row);
		sc0_cache	  = malloc(sizeof(*sc0_cache) * blocks_per_row);
		sc1_cache	  = malloc(sizeof(*sc1_cache) * blocks_per_row);
		sv_cache	  = malloc(sizeof(*sv_cache) * blocks_per_row);
		d_w_cache	  = malloc(sizeof(*d_w_cache) * blocks_per_row);
		if (!decoded_cache || !pdec_cache || !sc0_cache || !sc1_cache || !sv_cache || !d_w_cache) {
			free(decoded_cache);
			free(pdec_cache);
			free(sc0_cache);
			free(sc1_cache);
			free(sv_cache);
			free(d_w_cache);
			decoded_cache = NULL;
		}
	}

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + (size_t)(i / 8) * row_stride;

		const int n_bi_tiles = decoded_cache ? blocks_per_row : 0;

		if (decoded_cache) {
			for (int bi = 0; bi < n_bi_tiles; bi++) {
				const uint8_t *blk = group + (size_t)bi * IQ3_S_RE8_GROUP_BYTES;
				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + IQ3_S_RE8_GROUP_BYTES, 0, 1);

				const uint16_t *d_ptr	   = (const uint16_t *)(blk + IQ3S_RE8_OFF_D);
				const uint8_t  *scales_all = blk + IQ3S_RE8_OFF_SCALES;
				const uint8_t  *idx_all	   = blk + IQ3S_RE8_OFF_IDX;

				for (int r = 0; r < MR; r++) {
					d_w_cache[bi][r]	  = f16_to_f32_fast(d_ptr[r]);
					const uint8_t *scales = scales_all + (size_t)r * 4;
					const uint8_t *idx	  = idx_all + (size_t)r * 128;

					for (int g = 0; g < 8; g += 2) {
						uint8_t sb				= scales[g / 2];
						sc0_cache[bi][r][g / 2] = 1 + 2 * (sb & 0xf);
						sc1_cache[bi][r][g / 2] = 1 + 2 * (sb >> 4);

						const int vi = g * 2;
						iq3s_re_unpack_group(idx + g * 16, tbl, &decoded_cache[bi][r][vi],
											 &decoded_cache[bi][r][vi + 1]);
						iq3s_re_unpack_group(idx + (g + 1) * 16, tbl, &decoded_cache[bi][r][vi + 2],
											 &decoded_cache[bi][r][vi + 3]);
					}
				}

				for (int p = 0; p < MR / 2; p++) {
					const int8x16_t *da = decoded_cache[bi][2 * p];
					const int8x16_t *db = decoded_cache[bi][2 * p + 1];
					for (int s = 0; s < 8; s++) {
						pdec_cache[bi][p][s][0] =
							vcombine_s8(vget_low_s8(da[2 * s]), vget_low_s8(db[2 * s]));
						pdec_cache[bi][p][s][1] =
							vcombine_s8(vget_high_s8(da[2 * s]), vget_high_s8(db[2 * s]));
						pdec_cache[bi][p][s][2] =
							vcombine_s8(vget_low_s8(da[2 * s + 1]), vget_low_s8(db[2 * s + 1]));
						pdec_cache[bi][p][s][3] =
							vcombine_s8(vget_high_s8(da[2 * s + 1]), vget_high_s8(db[2 * s + 1]));
					}
					for (int gi = 0; gi < 4; gi++) {
						sv_cache[bi][p][gi][0] =
							vcombine_s32(vdup_n_s32(sc0_cache[bi][2 * p][gi]),
										 vdup_n_s32(sc0_cache[bi][2 * p + 1][gi]));
						sv_cache[bi][p][gi][1] =
							vcombine_s32(vdup_n_s32(sc1_cache[bi][2 * p][gi]),
										 vdup_n_s32(sc1_cache[bi][2 * p + 1][gi]));
					}
				}
			}
		}

		int t = 0;
		if (decoded_cache) {
			for (; t + I8MM_NR <= m; t += I8MM_NR) {
				float32x4_t facc[MR / 2][I8MM_NR / 2];
				for (int p = 0; p < MR / 2; p++)
					for (int c = 0; c < I8MM_NR / 2; c++)
						facc[p][c] = vdupq_n_f32(0.0f);

				const q8_k_block *xrow[I8MM_NR];
				for (int c = 0; c < I8MM_NR; c++)
					xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

				for (int bi = 0; bi < blocks_per_row; bi++) {
					const int8_t *restrict q8p[I8MM_NR];
					for (int c = 0; c < I8MM_NR; c++)
						q8p[c] = xrow[c][bi].qs;

					float32x4_t dcol[I8MM_NR / 2];
					float32x4_t drowv[MR / 2];
					for (int cp = 0; cp < I8MM_NR / 2; cp++)
						dcol[cp] = vzip1q_f32(vdupq_n_f32(xrow[2 * cp][bi].d),
											  vdupq_n_f32(xrow[2 * cp + 1][bi].d));
					for (int p = 0; p < MR / 2; p++)
						drowv[p] = vcombine_f32(vdup_n_f32(d_w_cache[bi][2 * p]),
												vdup_n_f32(d_w_cache[bi][2 * p + 1]));

					for (int gi = 0; gi < 4; gi++) {
						for (int cp = 0; cp < I8MM_NR / 2; cp++) {
							const int8_t *ea = q8p[2 * cp] + (2 * gi) * 32;
							const int8_t *eb = q8p[2 * cp + 1] + (2 * gi) * 32;
							const int8_t *oa = ea + 32;
							const int8_t *ob = eb + 32;

							const int8x16_t eal = vld1q_s8(ea);
							const int8x16_t eah = vld1q_s8(ea + 16);
							const int8x16_t ebl = vld1q_s8(eb);
							const int8x16_t ebh = vld1q_s8(eb + 16);
							int8x16_t		BE[4];
							BE[0] = vcombine_s8(vget_low_s8(eal), vget_low_s8(ebl));
							BE[1] = vcombine_s8(vget_high_s8(eal), vget_high_s8(ebl));
							BE[2] = vcombine_s8(vget_low_s8(eah), vget_low_s8(ebh));
							BE[3] = vcombine_s8(vget_high_s8(eah), vget_high_s8(ebh));

							const int8x16_t oal = vld1q_s8(oa);
							const int8x16_t oah = vld1q_s8(oa + 16);
							const int8x16_t obl = vld1q_s8(ob);
							const int8x16_t obh = vld1q_s8(ob + 16);
							int8x16_t		BO[4];
							BO[0] = vcombine_s8(vget_low_s8(oal), vget_low_s8(obl));
							BO[1] = vcombine_s8(vget_high_s8(oal), vget_high_s8(obl));
							BO[2] = vcombine_s8(vget_low_s8(oah), vget_low_s8(obh));
							BO[3] = vcombine_s8(vget_high_s8(oah), vget_high_s8(obh));

							for (int p = 0; p < MR / 2; p++) {
								const int8x16_t *pe = pdec_cache[bi][p][2 * gi];
								const int8x16_t *po = pdec_cache[bi][p][2 * gi + 1];

								int32x4_t tE = vmmlaq_s32(
									vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), pe[0], BE[0]),
														  pe[1], BE[1]),
											   pe[2], BE[2]),
									pe[3], BE[3]);
								int32x4_t tO = vmmlaq_s32(
									vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), po[0], BO[0]),
														  po[1], BO[1]),
											   po[2], BO[2]),
									po[3], BO[3]);

								const int32x4_t sumi =
									vmlaq_s32(vmulq_s32(tE, sv_cache[bi][p][gi][0]), tO,
											  sv_cache[bi][p][gi][1]);
								facc[p][cp] = vfmaq_f32(facc[p][cp], vcvtq_f32_s32(sumi),
														vmulq_f32(drowv[p], dcol[cp]));
							}
						}
					}
				}

				for (int p = 0; p < MR / 2; p++) {
					for (int cp = 0; cp < I8MM_NR / 2; cp++) {
						float tmp[4];
						vst1q_f32(tmp, facc[p][cp]);
						y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 0)] = tmp[0];
						y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 0)] = tmp[1];
						y[((size_t)(t + 2 * cp + 0) * y_row_stride) + (i + 2 * p + 1)] = tmp[2];
						y[((size_t)(t + 2 * cp + 1) * y_row_stride) + (i + 2 * p + 1)] = tmp[3];
					}
				}
			}
		}

		for (; t < m; t++) {
			matmul_iq3_s_re8_q8_k_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
												y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}

	for (; i < n; i++) {
		const uint8_t *igroup = Wb + (size_t)(i / 8) * row_stride;
		for (int t = 0; t < m; t++) {
			matmul_iq3_s_re8_q8_k_qonly_f32_row(igroup, xq + ((size_t)t * xq_row_stride_blocks),
												y + ((size_t)t * y_row_stride) + i, 1, k);
		}
	}

	free(decoded_cache);
	free(pdec_cache);
	free(sc0_cache);
	free(sc1_cache);
	free(sv_cache);
	free(d_w_cache);
}
#endif

#define NR_IQ3S_RE8 4

void matmul_iq3_s_re8_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
									 size_t		 xq_row_stride_blocks, float *restrict y,
									 int y_row_stride, int n, int k, int m) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
	matmul_iq3_s_re8_q8_k_qonly_f32_i8mm(w, xq, xq_row_stride_blocks, y, y_row_stride, n, k, m);
	return;
#endif
	const int	   blocks_per_row = k / 256;
	const size_t   row_stride	  = (size_t)blocks_per_row * IQ3_S_RE8_GROUP_BYTES;
	const uint8_t *Wb			  = w;
	uint8x16_t	   tbl			  = vld1q_u8((const uint8_t *)iq3s_re_decode_tbl);
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + (size_t)(i / 8) * row_stride;
		int			   t	 = 0;

		const int n_bi_tiles = (m / NR_IQ3S_RE8) > 0 ? blocks_per_row : 0;

		static _Thread_local int8x16_t(*decoded_cache)[MR][16] = NULL;
		static _Thread_local int32_t (*sc0_cache)[MR][4]	   = NULL;
		static _Thread_local int32_t (*sc1_cache)[MR][4]	   = NULL;
		static _Thread_local float (*d_w_cache)[MR]			   = NULL;
		static _Thread_local int cache_cap					   = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				decoded_cache = realloc(decoded_cache, sizeof(*decoded_cache) * n_bi_tiles);
				sc0_cache	  = realloc(sc0_cache, sizeof(*sc0_cache) * n_bi_tiles);
				sc1_cache	  = realloc(sc1_cache, sizeof(*sc1_cache) * n_bi_tiles);
				d_w_cache	  = realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				cache_cap	  = n_bi_tiles;
				tlocal_register((void **)&decoded_cache);
				tlocal_register((void **)&sc0_cache);
				tlocal_register((void **)&sc1_cache);
				tlocal_register((void **)&d_w_cache);
			}

			for (int bi = 0; bi < n_bi_tiles; bi++) {
				const uint8_t *blk = group + (size_t)bi * IQ3_S_RE8_GROUP_BYTES;
				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + IQ3_S_RE8_GROUP_BYTES, 0, 1);

				const uint16_t *d_ptr	   = (const uint16_t *)(blk + IQ3S_RE8_OFF_D);
				const uint8_t  *scales_all = blk + IQ3S_RE8_OFF_SCALES;
				const uint8_t  *idx_all	   = blk + IQ3S_RE8_OFF_IDX;

				for (int r = 0; r < MR; r++) {
					d_w_cache[bi][r]	  = f16_to_f32_fast(d_ptr[r]);
					const uint8_t *scales = scales_all + (size_t)r * 4;
					const uint8_t *idx	  = idx_all + (size_t)r * 128;

					for (int g = 0; g < 8; g += 2) {
						uint8_t sb				= scales[g / 2];
						sc0_cache[bi][r][g / 2] = 1 + 2 * (sb & 0xf);
						sc1_cache[bi][r][g / 2] = 1 + 2 * (sb >> 4);
						const int vi			= g * 2;
						iq3s_re_unpack_group(idx + g * 16, tbl, &decoded_cache[bi][r][vi],
											 &decoded_cache[bi][r][vi + 1]);
						iq3s_re_unpack_group(idx + (g + 1) * 16, tbl, &decoded_cache[bi][r][vi + 2],
											 &decoded_cache[bi][r][vi + 3]);
					}
				}
			}
		}

		for (; t + NR_IQ3S_RE8 <= m; t += NR_IQ3S_RE8) {
			float32x4_t acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = vdupq_n_f32(0.0f);

			const q8_k_block *xrow[NR_IQ3S_RE8];

			for (int c = 0; c < NR_IQ3S_RE8; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float(*d_w)				= d_w_cache[bi];
				int8x16_t(*decoded)[16] = decoded_cache[bi];
				int32_t (*sc0)[4]		= sc0_cache[bi];
				int32_t (*sc1)[4]		= sc1_cache[bi];

				float xd[NR_IQ3S_RE8];
				const int8_t *restrict q8p[NR_IQ3S_RE8];
				for (int c = 0; c < NR_IQ3S_RE8; c++) {
					xd[c] = xrow[c][bi].d;
				}
				const float32x4_t xd_vec = vld1q_f32(xd);

				for (int r = 0; r < MR; r++) {
					int32x4_t total4 = vdupq_n_s32(0);

					for (int g = 0; g < 8; g += 2) {
						const int		vi	= g * 2;
						const int8x16_t c00 = decoded[r][vi];
						const int8x16_t c01 = decoded[r][vi + 1];
						const int8x16_t c10 = decoded[r][vi + 2];
						const int8x16_t c11 = decoded[r][vi + 3];

						const int8_t *restrict qg0_0 = q8p[0] + g * 32;
						const int8_t *restrict qg0_1 = q8p[1] + g * 32;
						const int8_t *restrict qg0_2 = q8p[2] + g * 32;
						const int8_t *restrict qg0_3 = q8p[3] + g * 32;

						const int8_t *restrict qg1_0 = q8p[0] + (g + 1) * 32;
						const int8_t *restrict qg1_1 = q8p[1] + (g + 1) * 32;
						const int8_t *restrict qg1_2 = q8p[2] + (g + 1) * 32;
						const int8_t *restrict qg1_3 = q8p[3] + (g + 1) * 32;

#if defined(__ARM_FEATURE_DOTPROD)
						int32x4_t a0_c0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_0)),
													c01, vld1q_s8(qg0_0 + 16));
						int32x4_t a0_c1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_1)),
													c01, vld1q_s8(qg0_1 + 16));
						int32x4_t a0_c2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_2)),
													c01, vld1q_s8(qg0_2 + 16));
						int32x4_t a0_c3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c00, vld1q_s8(qg0_3)),
													c01, vld1q_s8(qg0_3 + 16));

						int32x4_t a1_c0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_0)),
													c11, vld1q_s8(qg1_0 + 16));
						int32x4_t a1_c1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_1)),
													c11, vld1q_s8(qg1_1 + 16));
						int32x4_t a1_c2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_2)),
													c11, vld1q_s8(qg1_2 + 16));
						int32x4_t a1_c3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), c10, vld1q_s8(qg1_3)),
													c11, vld1q_s8(qg1_3 + 16));
#else
						int16x8_t p00_0 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_0)));
						int16x8_t p01_0 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_0)));
						int16x8_t p02_0 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_0 + 16)));
						int16x8_t p03_0 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_0 + 16)));
						int32x4_t a0_c0 = vpaddlq_s16(p00_0);
						a0_c0			= vpadalq_s16(a0_c0, p01_0);
						a0_c0			= vpadalq_s16(a0_c0, p02_0);
						a0_c0			= vpadalq_s16(a0_c0, p03_0);

						int16x8_t p00_1 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_1)));
						int16x8_t p01_1 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_1)));
						int16x8_t p02_1 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_1 + 16)));
						int16x8_t p03_1 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_1 + 16)));
						int32x4_t a0_c1 = vpaddlq_s16(p00_1);
						a0_c1			= vpadalq_s16(a0_c1, p01_1);
						a0_c1			= vpadalq_s16(a0_c1, p02_1);
						a0_c1			= vpadalq_s16(a0_c1, p03_1);

						int16x8_t p00_2 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_2)));
						int16x8_t p01_2 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_2)));
						int16x8_t p02_2 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_2 + 16)));
						int16x8_t p03_2 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_2 + 16)));
						int32x4_t a0_c2 = vpaddlq_s16(p00_2);
						a0_c2			= vpadalq_s16(a0_c2, p01_2);
						a0_c2			= vpadalq_s16(a0_c2, p02_2);
						a0_c2			= vpadalq_s16(a0_c2, p03_2);

						int16x8_t p00_3 = vmull_s8(vget_low_s8(c00), vget_low_s8(vld1q_s8(qg0_3)));
						int16x8_t p01_3 =
							vmull_s8(vget_high_s8(c00), vget_high_s8(vld1q_s8(qg0_3)));
						int16x8_t p02_3 =
							vmull_s8(vget_low_s8(c01), vget_low_s8(vld1q_s8(qg0_3 + 16)));
						int16x8_t p03_3 =
							vmull_s8(vget_high_s8(c01), vget_high_s8(vld1q_s8(qg0_3 + 16)));
						int32x4_t a0_c3 = vpaddlq_s16(p00_3);
						a0_c3			= vpadalq_s16(a0_c3, p01_3);
						a0_c3			= vpadalq_s16(a0_c3, p02_3);
						a0_c3			= vpadalq_s16(a0_c3, p03_3);

						int16x8_t p10_0 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_0)));
						int16x8_t p11_0 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_0)));
						int16x8_t p12_0 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_0 + 16)));
						int16x8_t p13_0 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_0 + 16)));
						int32x4_t a1_c0 = vpaddlq_s16(p10_0);
						a1_c0			= vpadalq_s16(a1_c0, p11_0);
						a1_c0			= vpadalq_s16(a1_c0, p12_0);
						a1_c0			= vpadalq_s16(a1_c0, p13_0);

						int16x8_t p10_1 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_1)));
						int16x8_t p11_1 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_1)));
						int16x8_t p12_1 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_1 + 16)));
						int16x8_t p13_1 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_1 + 16)));
						int32x4_t a1_c1 = vpaddlq_s16(p10_1);
						a1_c1			= vpadalq_s16(a1_c1, p11_1);
						a1_c1			= vpadalq_s16(a1_c1, p12_1);
						a1_c1			= vpadalq_s16(a1_c1, p13_1);

						int16x8_t p10_2 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_2)));
						int16x8_t p11_2 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_2)));
						int16x8_t p12_2 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_2 + 16)));
						int16x8_t p13_2 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_2 + 16)));
						int32x4_t a1_c2 = vpaddlq_s16(p10_2);
						a1_c2			= vpadalq_s16(a1_c2, p11_2);
						a1_c2			= vpadalq_s16(a1_c2, p12_2);
						a1_c2			= vpadalq_s16(a1_c2, p13_2);

						int16x8_t p10_3 = vmull_s8(vget_low_s8(c10), vget_low_s8(vld1q_s8(qg1_3)));
						int16x8_t p11_3 =
							vmull_s8(vget_high_s8(c10), vget_high_s8(vld1q_s8(qg1_3)));
						int16x8_t p12_3 =
							vmull_s8(vget_low_s8(c11), vget_low_s8(vld1q_s8(qg1_3 + 16)));
						int16x8_t p13_3 =
							vmull_s8(vget_high_s8(c11), vget_high_s8(vld1q_s8(qg1_3 + 16)));
						int32x4_t a1_c3 = vpaddlq_s16(p10_3);
						a1_c3			= vpadalq_s16(a1_c3, p11_3);
						a1_c3			= vpadalq_s16(a1_c3, p12_3);
						a1_c3			= vpadalq_s16(a1_c3, p13_3);
#endif
						int32x4_t a0_s01	= vpaddq_s32(a0_c0, a0_c1);
						int32x4_t a0_s23	= vpaddq_s32(a0_c2, a0_c3);
						int32x4_t a0_packed = vpaddq_s32(a0_s01, a0_s23);

						int32x4_t a1_s01	= vpaddq_s32(a1_c0, a1_c1);
						int32x4_t a1_s23	= vpaddq_s32(a1_c2, a1_c3);
						int32x4_t a1_packed = vpaddq_s32(a1_s01, a1_s23);

						total4 = vmlaq_n_s32(total4, a0_packed, sc0[r][g / 2]);
						total4 = vmlaq_n_s32(total4, a1_packed, sc1[r][g / 2]);
					}

					float32x4_t total4f = vcvtq_f32_s32(total4);
					acc_row[r] = vfmaq_f32(acc_row[r], xd_vec, vmulq_n_f32(total4f, d_w[r]));
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				vst1q_f32(tmp, acc_row[r]);
				for (int c = 0; c < NR_IQ3S_RE8; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
		}

		for (; t < m; t++) {
			const q8_k_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			 *yrow = y + ((size_t)t * y_row_stride) + i;
			matmul_iq3_s_re8_q8_k_qonly_f32_row(group, xrow, yrow, MR, k);
		}
	}

	for (; i < n; i++) {
		const uint8_t *group = Wb + (size_t)(i / 8) * row_stride;
		for (int t = 0; t < m; t++) {
			const q8_k_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			 *yrow = y + ((size_t)t * y_row_stride) + i;
			matmul_iq3_s_re8_q8_k_qonly_f32_row(group, xrow, yrow, 1, k);
		}
	}
}

#undef NR_IQ3S_RE8

void matmul_f16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					int k) {
	const uint16_t *Wb = w;
	const int		mr = 8;
	int				i  = 0;

	for (; i + mr <= n; i += mr) {
		float32x4_t acc[8];
		for (int r = 0; r < 8; r++)
			acc[r] = vdupq_n_f32(0.0f);
		const uint16_t *rows[8];
		for (int r = 0; r < 8; r++)
			rows[r] = Wb + ((size_t)(i + r) * k);

		int j = 0;
		for (; j + 8 <= k; j += 8) {
			__builtin_prefetch(x + j + 32, 0, 1);
			for (int r = 0; r < 8; r++)
				__builtin_prefetch(rows[r] + j + 32, 0, 1);
			float32x4_t x0 = vld1q_f32(x + j);
			float32x4_t x1 = vld1q_f32(x + j + 4);
			for (int r = 0; r < 8; r++) {
				float16x8_t wh = vld1q_f16((const float16_t *)(rows[r] + j));
				float32x4_t w0 = vcvt_f32_f16(vget_low_f16(wh));
				float32x4_t w1 = vcvt_f32_f16(vget_high_f16(wh));
				acc[r]		   = vfmaq_f32(acc[r], x0, w0);
				acc[r]		   = vfmaq_f32(acc[r], x1, w1);
			}
		}
		for (int r = 0; r < 8; r++) {
			float s = vaddvq_f32(acc[r]);
			for (int j2 = j; j2 < k; j2++)
				s += f16_to_f32_fast(rows[r][j2]) * x[j2];
			y[i + r] = s;
		}
	}

	for (; i < n; i++) {
		const uint16_t *restrict wr = Wb + (size_t)i * k;
		float32x4_t acc0			= vdupq_n_f32(0.0f);
		float32x4_t acc1			= vdupq_n_f32(0.0f);
		int			j				= 0;
		for (; j + 16 <= k; j += 16) {
			__builtin_prefetch(wr + j + 32, 0, 1);
			float16x8_t wh0 = vld1q_f16((const float16_t *)(wr + j));
			float16x8_t wh1 = vld1q_f16((const float16_t *)(wr + j + 8));
			float32x4_t x0	= vld1q_f32(x + j);
			float32x4_t x1	= vld1q_f32(x + j + 4);
			float32x4_t x2	= vld1q_f32(x + j + 8);
			float32x4_t x3	= vld1q_f32(x + j + 12);
			acc0			= vfmaq_f32(acc0, x0, vcvt_f32_f16(vget_low_f16(wh0)));
			acc1			= vfmaq_f32(acc1, x1, vcvt_f32_f16(vget_high_f16(wh0)));
			acc0			= vfmaq_f32(acc0, x2, vcvt_f32_f16(vget_low_f16(wh1)));
			acc1			= vfmaq_f32(acc1, x3, vcvt_f32_f16(vget_high_f16(wh1)));
		}
		for (; j + 8 <= k; j += 8) {
			float16x8_t wh = vld1q_f16((const float16_t *)(wr + j));
			float32x4_t x0 = vld1q_f32(x + j);
			float32x4_t x1 = vld1q_f32(x + j + 4);
			acc0		   = vfmaq_f32(acc0, x0, vcvt_f32_f16(vget_low_f16(wh)));
			acc1		   = vfmaq_f32(acc1, x1, vcvt_f32_f16(vget_high_f16(wh)));
		}
		float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
		for (; j < k; j++)
			s += f16_to_f32_fast(wr[j]) * x[j];
		y[i] = s;
	}
}

void matmul_f16_f32_batch(const void *restrict w, const float *restrict x, float *restrict y, int n,
						  int k, int m, int x_row_stride, int y_row_stride) {
	const uint16_t *Wb = w;
	const int		MB = 4;
	int				mb = 0;
	for (; mb + MB <= m; mb += MB) {
		const float *xb[4];
		for (int t = 0; t < 4; t++)
			xb[t] = x + (size_t)(mb + t) * x_row_stride;

		const int NR = 4;
		int		  i	 = 0;
		for (; i + NR <= n; i += NR) {
			const uint16_t *rows[4];
			for (int r = 0; r < 4; r++)
				rows[r] = Wb + (size_t)(i + r) * k;

			float32x4_t acc[4][4];
			for (int t = 0; t < 4; t++)
				for (int r = 0; r < 4; r++)
					acc[t][r] = vdupq_n_f32(0.0f);

			int j = 0;
			for (; j + 8 <= k; j += 8) {
				__builtin_prefetch(xb[0] + j + 32, 0, 1);
				for (int r = 0; r < 4; r++)
					__builtin_prefetch(rows[r] + j + 32, 0, 1);
				float16x8_t wh0	  = vld1q_f16((const float16_t *)(rows[0] + j));
				float16x8_t wh1	  = vld1q_f16((const float16_t *)(rows[1] + j));
				float16x8_t wh2	  = vld1q_f16((const float16_t *)(rows[2] + j));
				float16x8_t wh3	  = vld1q_f16((const float16_t *)(rows[3] + j));
				float32x4_t w0_lo = vcvt_f32_f16(vget_low_f16(wh0));
				float32x4_t w0_hi = vcvt_f32_f16(vget_high_f16(wh0));
				float32x4_t w1_lo = vcvt_f32_f16(vget_low_f16(wh1));
				float32x4_t w1_hi = vcvt_f32_f16(vget_high_f16(wh1));
				float32x4_t w2_lo = vcvt_f32_f16(vget_low_f16(wh2));
				float32x4_t w2_hi = vcvt_f32_f16(vget_high_f16(wh2));
				float32x4_t w3_lo = vcvt_f32_f16(vget_low_f16(wh3));
				float32x4_t w3_hi = vcvt_f32_f16(vget_high_f16(wh3));
				for (int t = 0; t < 4; t++) {
					float32x4_t xv_lo = vld1q_f32(xb[t] + j);
					float32x4_t xv_hi = vld1q_f32(xb[t] + j + 4);
					acc[t][0]		  = vfmaq_f32(acc[t][0], xv_lo, w0_lo);
					acc[t][0]		  = vfmaq_f32(acc[t][0], xv_hi, w0_hi);
					acc[t][1]		  = vfmaq_f32(acc[t][1], xv_lo, w1_lo);
					acc[t][1]		  = vfmaq_f32(acc[t][1], xv_hi, w1_hi);
					acc[t][2]		  = vfmaq_f32(acc[t][2], xv_lo, w2_lo);
					acc[t][2]		  = vfmaq_f32(acc[t][2], xv_hi, w2_hi);
					acc[t][3]		  = vfmaq_f32(acc[t][3], xv_lo, w3_lo);
					acc[t][3]		  = vfmaq_f32(acc[t][3], xv_hi, w3_hi);
				}
			}
			for (; j + 4 <= k; j += 4) {
				float16x4_t wh0 = vld1_f16((const float16_t *)(rows[0] + j));
				float16x4_t wh1 = vld1_f16((const float16_t *)(rows[1] + j));
				float16x4_t wh2 = vld1_f16((const float16_t *)(rows[2] + j));
				float16x4_t wh3 = vld1_f16((const float16_t *)(rows[3] + j));
				float32x4_t w0	= vcvt_f32_f16(wh0);
				float32x4_t w1	= vcvt_f32_f16(wh1);
				float32x4_t w2	= vcvt_f32_f16(wh2);
				float32x4_t w3	= vcvt_f32_f16(wh3);
				for (int t = 0; t < 4; t++) {
					float32x4_t xv = vld1q_f32(xb[t] + j);
					acc[t][0]	   = vfmaq_f32(acc[t][0], xv, w0);
					acc[t][1]	   = vfmaq_f32(acc[t][1], xv, w1);
					acc[t][2]	   = vfmaq_f32(acc[t][2], xv, w2);
					acc[t][3]	   = vfmaq_f32(acc[t][3], xv, w3);
				}
			}
			for (int t = 0; t < 4; t++) {
				float s[4];
				for (int r = 0; r < 4; r++)
					s[r] = vaddvq_f32(acc[t][r]);
				for (int j2 = j; j2 < k; j2++) {
					float xv = xb[t][j2];
					for (int r = 0; r < 4; r++)
						s[r] += f16_to_f32_fast(rows[r][j2]) * xv;
				}
				for (int r = 0; r < 4; r++)
					y[(size_t)(mb + t) * y_row_stride + i + r] = s[r];
			}
		}
		for (; i < n; i++) {
			const uint16_t *wr = Wb + (size_t)i * k;
			for (int t = 0; t < 4; t++) {
				float32x4_t acc = vdupq_n_f32(0.0f);
				int			j	= 0;
				for (; j + 4 <= k; j += 4) {
					float16x4_t wh = vld1_f16((const float16_t *)(wr + j));
					acc			   = vfmaq_f32(acc, vld1q_f32(xb[t] + j), vcvt_f32_f16(wh));
				}
				float s = vaddvq_f32(acc);
				for (; j < k; j++)
					s += f16_to_f32_fast(wr[j]) * xb[t][j];
				y[(size_t)(mb + t) * y_row_stride + i] = s;
			}
		}
	}
	for (; mb < m; mb++) {
		matmul_f16_f32(w, x + (size_t)mb * x_row_stride, y + (size_t)mb * y_row_stride, n, k);
	}
}

void matmul_bf16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					 int k) {
	const uint16_t *Wb = w;
	const int		mr = 4;
	int				i  = 0;

#if defined(__ARM_FEATURE_BF16)
#define F32x2_TO_BF16x8(lo, hi)                                                                    \
	vreinterpretq_bf16_u16(vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(lo), 16),                \
										vshrn_n_u32(vreinterpretq_u32_f32(hi), 16)))

	for (; i + mr <= n; i += mr) {
		float32x4_t acc[4];
		for (int r = 0; r < 4; r++)
			acc[r] = vdupq_n_f32(0.0f);

		const uint16_t *rows[4];
		for (int r = 0; r < 4; r++)
			rows[r] = Wb + (size_t)(i + r) * k;

		int j = 0;
		for (; j + 16 <= k; j += 16) {
			__builtin_prefetch(x + j + 32, 0, 1);
			for (int r = 0; r < 4; r++)
				__builtin_prefetch(rows[r] + j + 32, 0, 1);
			float32x4_t	 x0	 = vld1q_f32(x + j);
			float32x4_t	 x1	 = vld1q_f32(x + j + 4);
			float32x4_t	 x2	 = vld1q_f32(x + j + 8);
			float32x4_t	 x3	 = vld1q_f32(x + j + 12);
			bfloat16x8_t xb0 = F32x2_TO_BF16x8(x0, x1);
			bfloat16x8_t xb1 = F32x2_TO_BF16x8(x2, x3);
			for (int r = 0; r < 4; r++) {
				bfloat16x8_t wb0 = vreinterpretq_bf16_u16(vld1q_u16(rows[r] + j));
				bfloat16x8_t wb1 = vreinterpretq_bf16_u16(vld1q_u16(rows[r] + j + 8));
				acc[r]			 = vbfdotq_f32(acc[r], wb0, xb0);
				acc[r]			 = vbfdotq_f32(acc[r], wb1, xb1);
			}
		}
		for (; j + 8 <= k; j += 8) {
			float32x4_t	 x0 = vld1q_f32(x + j);
			float32x4_t	 x1 = vld1q_f32(x + j + 4);
			bfloat16x8_t xb = F32x2_TO_BF16x8(x0, x1);
			for (int r = 0; r < 4; r++) {
				bfloat16x8_t wb = vreinterpretq_bf16_u16(vld1q_u16(rows[r] + j));
				acc[r]			= vbfdotq_f32(acc[r], wb, xb);
			}
		}
		for (int r = 0; r < 4; r++) {
			float s = vaddvq_f32(acc[r]);
			for (int j2 = j; j2 < k; j2++) {
				union {
					uint32_t u;
					float	 f;
				} v;
				v.u = ((uint32_t)rows[r][j2]) << 16;
				s += v.f * x[j2];
			}
			y[i + r] = s;
		}
	}
	for (; i < n; i++) {
		const uint16_t *restrict wr = Wb + (size_t)i * k;
		float32x4_t acc0			= vdupq_n_f32(0.0f);
		float32x4_t acc1			= vdupq_n_f32(0.0f);
		int			j				= 0;
		for (; j + 16 <= k; j += 16) {
			__builtin_prefetch(wr + j + 32, 0, 1);
			float32x4_t	 x0	 = vld1q_f32(x + j);
			float32x4_t	 x1	 = vld1q_f32(x + j + 4);
			float32x4_t	 x2	 = vld1q_f32(x + j + 8);
			float32x4_t	 x3	 = vld1q_f32(x + j + 12);
			bfloat16x8_t xb0 = F32x2_TO_BF16x8(x0, x1);
			bfloat16x8_t xb1 = F32x2_TO_BF16x8(x2, x3);
			bfloat16x8_t wb0 = vreinterpretq_bf16_u16(vld1q_u16(wr + j));
			bfloat16x8_t wb1 = vreinterpretq_bf16_u16(vld1q_u16(wr + j + 8));
			acc0			 = vbfdotq_f32(acc0, wb0, xb0);
			acc1			 = vbfdotq_f32(acc1, wb1, xb1);
		}
		for (; j + 8 <= k; j += 8) {
			float32x4_t	 x0 = vld1q_f32(x + j);
			float32x4_t	 x1 = vld1q_f32(x + j + 4);
			bfloat16x8_t xb = F32x2_TO_BF16x8(x0, x1);
			bfloat16x8_t wb = vreinterpretq_bf16_u16(vld1q_u16(wr + j));
			acc0			= vbfdotq_f32(acc0, wb, xb);
		}
		float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
		for (; j < k; j++) {
			union {
				uint32_t u;
				float	 f;
			} v;
			v.u = ((uint32_t)wr[j]) << 16;
			s += v.f * x[j];
		}
		y[i] = s;
	}
#undef F32x2_TO_BF16x8
#else
	for (; i + mr <= n; i += mr) {
		float32x4_t acc[4];
		for (int r = 0; r < 4; r++)
			acc[r] = vdupq_n_f32(0.0f);

		const uint16_t *rows[4];
		for (int r = 0; r < 4; r++)
			rows[r] = Wb + ((size_t)(i + r) * k);

		int j = 0;
		for (; j + 16 <= k; j += 16) {
			__builtin_prefetch(x + j + 32, 0, 1);
			for (int r = 0; r < 4; r++)
				__builtin_prefetch(rows[r] + j + 32, 0, 1);
			float32x4_t x0 = vld1q_f32(x + j);
			float32x4_t x1 = vld1q_f32(x + j + 4);
			float32x4_t x2 = vld1q_f32(x + j + 8);
			float32x4_t x3 = vld1q_f32(x + j + 12);
			for (int r = 0; r < 4; r++) {
				uint16x8_t	bw0	 = vld1q_u16(rows[r] + j);
				uint16x8_t	bw1	 = vld1q_u16(rows[r] + j + 8);
				uint32x4_t	lo32 = vshll_n_u16(vget_low_u16(bw0), 16);
				uint32x4_t	hi32 = vshll_n_u16(vget_high_u16(bw0), 16);
				float32x4_t w0	 = vreinterpretq_f32_u32(lo32);
				float32x4_t w1	 = vreinterpretq_f32_u32(hi32);
				acc[r]			 = vfmaq_f32(acc[r], x0, w0);
				acc[r]			 = vfmaq_f32(acc[r], x1, w1);
				lo32			 = vshll_n_u16(vget_low_u16(bw1), 16);
				hi32			 = vshll_n_u16(vget_high_u16(bw1), 16);
				w0				 = vreinterpretq_f32_u32(lo32);
				w1				 = vreinterpretq_f32_u32(hi32);
				acc[r]			 = vfmaq_f32(acc[r], x2, w0);
				acc[r]			 = vfmaq_f32(acc[r], x3, w1);
			}
		}
		for (; j + 8 <= k; j += 8) {
			float32x4_t x0 = vld1q_f32(x + j);
			float32x4_t x1 = vld1q_f32(x + j + 4);
			for (int r = 0; r < 4; r++) {
				uint16x8_t	bw	 = vld1q_u16(rows[r] + j);
				uint32x4_t	lo32 = vshll_n_u16(vget_low_u16(bw), 16);
				uint32x4_t	hi32 = vshll_n_u16(vget_high_u16(bw), 16);
				float32x4_t w0	 = vreinterpretq_f32_u32(lo32);
				float32x4_t w1	 = vreinterpretq_f32_u32(hi32);
				acc[r]			 = vfmaq_f32(acc[r], x0, w0);
				acc[r]			 = vfmaq_f32(acc[r], x1, w1);
			}
		}
		for (int r = 0; r < 4; r++) {
			float s = vaddvq_f32(acc[r]);
			for (int j2 = j; j2 < k; j2++) {
				union {
					uint32_t u;
					float	 f;
				} v;
				v.u = ((uint32_t)rows[r][j2]) << 16;
				s += v.f * x[j2];
			}
			y[i + r] = s;
		}
	}
	for (; i < n; i++) {
		const uint16_t *restrict wr = Wb + ((size_t)i * k);
		float32x4_t acc0			= vdupq_n_f32(0.0f);
		float32x4_t acc1			= vdupq_n_f32(0.0f);
		int			j				= 0;
		for (; j + 16 <= k; j += 16) {
			__builtin_prefetch(wr + j + 32, 0, 1);
			float32x4_t x0	 = vld1q_f32(x + j);
			float32x4_t x1	 = vld1q_f32(x + j + 4);
			float32x4_t x2	 = vld1q_f32(x + j + 8);
			float32x4_t x3	 = vld1q_f32(x + j + 12);
			uint16x8_t	bw0	 = vld1q_u16(wr + j);
			uint16x8_t	bw1	 = vld1q_u16(wr + j + 8);
			uint32x4_t	lo32 = vshll_n_u16(vget_low_u16(bw0), 16);
			uint32x4_t	hi32 = vshll_n_u16(vget_high_u16(bw0), 16);
			acc0			 = vfmaq_f32(acc0, x0, vreinterpretq_f32_u32(lo32));
			acc0			 = vfmaq_f32(acc0, x1, vreinterpretq_f32_u32(hi32));
			lo32			 = vshll_n_u16(vget_low_u16(bw1), 16);
			hi32			 = vshll_n_u16(vget_high_u16(bw1), 16);
			acc1			 = vfmaq_f32(acc1, x2, vreinterpretq_f32_u32(lo32));
			acc1			 = vfmaq_f32(acc1, x3, vreinterpretq_f32_u32(hi32));
		}
		for (; j + 8 <= k; j += 8) {
			float32x4_t x0	 = vld1q_f32(x + j);
			float32x4_t x1	 = vld1q_f32(x + j + 4);
			uint16x8_t	bw	 = vld1q_u16(wr + j);
			uint32x4_t	lo32 = vshll_n_u16(vget_low_u16(bw), 16);
			uint32x4_t	hi32 = vshll_n_u16(vget_high_u16(bw), 16);
			acc0			 = vfmaq_f32(acc0, x0, vreinterpretq_f32_u32(lo32));
			acc0			 = vfmaq_f32(acc0, x1, vreinterpretq_f32_u32(hi32));
		}
		float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
		for (; j < k; j++) {
			union {
				uint32_t u;
				float	 f;
			} v;
			v.u = ((uint32_t)wr[j]) << 16;
			s += v.f * x[j];
		}
		y[i] = s;
	}
#endif
}

void matmul_bf16_f32_batch(const void *restrict w, const float *restrict x, float *restrict y,
						   int n, int k, int m, int x_row_stride, int y_row_stride) {
	if (m <= 1) {
		if (m == 1)
			matmul_bf16_f32(w, x, y, n, k);
		return;
	}

	const uint16_t *Wb = w;
	const int		mr = 4;
	int				i  = 0;

#if defined(__ARM_FEATURE_BF16)
#define F32x2_TO_BF16x8(lo, hi)                                                                    \
	vreinterpretq_bf16_u16(vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(lo), 16),                \
										vshrn_n_u32(vreinterpretq_u32_f32(hi), 16)))

	for (; i + mr <= n; i += mr) {
		const uint16_t *rows[4];
		for (int r = 0; r < 4; r++)
			rows[r] = Wb + (size_t)(i + r) * k;

		for (int t = 0; t < m; t++) {
			const float *xt = x + (size_t)t * x_row_stride;
			float32x4_t	 acc[4];
			for (int r = 0; r < 4; r++)
				acc[r] = vdupq_n_f32(0.0f);

			int j = 0;
			for (; j + 16 <= k; j += 16) {
				__builtin_prefetch(xt + j + 32, 0, 1);
				for (int r = 0; r < 4; r++)
					__builtin_prefetch(rows[r] + j + 32, 0, 1);
				float32x4_t	 x0	 = vld1q_f32(xt + j);
				float32x4_t	 x1	 = vld1q_f32(xt + j + 4);
				float32x4_t	 x2	 = vld1q_f32(xt + j + 8);
				float32x4_t	 x3	 = vld1q_f32(xt + j + 12);
				bfloat16x8_t xb0 = F32x2_TO_BF16x8(x0, x1);
				bfloat16x8_t xb1 = F32x2_TO_BF16x8(x2, x3);
				for (int r = 0; r < 4; r++) {
					bfloat16x8_t wb0 = vreinterpretq_bf16_u16(vld1q_u16(rows[r] + j));
					bfloat16x8_t wb1 = vreinterpretq_bf16_u16(vld1q_u16(rows[r] + j + 8));
					acc[r]			 = vbfdotq_f32(acc[r], wb0, xb0);
					acc[r]			 = vbfdotq_f32(acc[r], wb1, xb1);
				}
			}
			for (; j + 8 <= k; j += 8) {
				float32x4_t	 x0 = vld1q_f32(xt + j);
				float32x4_t	 x1 = vld1q_f32(xt + j + 4);
				bfloat16x8_t xb = F32x2_TO_BF16x8(x0, x1);
				for (int r = 0; r < 4; r++) {
					bfloat16x8_t wb = vreinterpretq_bf16_u16(vld1q_u16(rows[r] + j));
					acc[r]			= vbfdotq_f32(acc[r], wb, xb);
				}
			}
			float *yt = y + (size_t)t * y_row_stride;
			for (int r = 0; r < 4; r++) {
				float s = vaddvq_f32(acc[r]);
				for (int j2 = j; j2 < k; j2++) {
					union {
						uint32_t u;
						float	 f;
					} v;
					v.u = ((uint32_t)rows[r][j2]) << 16;
					s += v.f * xt[j2];
				}
				yt[i + r] = s;
			}
		}
	}
	for (; i < n; i++) {
		const uint16_t *restrict wr = Wb + (size_t)i * k;
		for (int t = 0; t < m; t++) {
			const float *xt	  = x + (size_t)t * x_row_stride;
			float32x4_t	 acc0 = vdupq_n_f32(0.0f);
			float32x4_t	 acc1 = vdupq_n_f32(0.0f);
			int			 j	  = 0;
			for (; j + 16 <= k; j += 16) {
				__builtin_prefetch(wr + j + 32, 0, 1);
				float32x4_t	 x0	 = vld1q_f32(xt + j);
				float32x4_t	 x1	 = vld1q_f32(xt + j + 4);
				float32x4_t	 x2	 = vld1q_f32(xt + j + 8);
				float32x4_t	 x3	 = vld1q_f32(xt + j + 12);
				bfloat16x8_t xb0 = F32x2_TO_BF16x8(x0, x1);
				bfloat16x8_t xb1 = F32x2_TO_BF16x8(x2, x3);
				bfloat16x8_t wb0 = vreinterpretq_bf16_u16(vld1q_u16(wr + j));
				bfloat16x8_t wb1 = vreinterpretq_bf16_u16(vld1q_u16(wr + j + 8));
				acc0			 = vbfdotq_f32(acc0, wb0, xb0);
				acc1			 = vbfdotq_f32(acc1, wb1, xb1);
			}
			for (; j + 8 <= k; j += 8) {
				float32x4_t	 x0 = vld1q_f32(xt + j);
				float32x4_t	 x1 = vld1q_f32(xt + j + 4);
				bfloat16x8_t xb = F32x2_TO_BF16x8(x0, x1);
				bfloat16x8_t wb = vreinterpretq_bf16_u16(vld1q_u16(wr + j));
				acc0			= vbfdotq_f32(acc0, wb, xb);
			}
			float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
			for (; j < k; j++) {
				union {
					uint32_t u;
					float	 f;
				} v;
				v.u = ((uint32_t)wr[j]) << 16;
				s += v.f * xt[j];
			}
			y[(size_t)t * y_row_stride + i] = s;
		}
	}
#undef F32x2_TO_BF16x8
#else
	for (; i + mr <= n; i += mr) {
		const uint16_t *rows[4];
		for (int r = 0; r < 4; r++)
			rows[r] = Wb + ((size_t)(i + r) * k);

		for (int t = 0; t < m; t++) {
			const float *xt = x + (size_t)t * x_row_stride;
			float32x4_t	 acc[4];
			for (int r = 0; r < 4; r++)
				acc[r] = vdupq_n_f32(0.0f);

			int j = 0;
			for (; j + 16 <= k; j += 16) {
				__builtin_prefetch(xt + j + 32, 0, 1);
				for (int r = 0; r < 4; r++)
					__builtin_prefetch(rows[r] + j + 32, 0, 1);
				float32x4_t x0 = vld1q_f32(xt + j);
				float32x4_t x1 = vld1q_f32(xt + j + 4);
				float32x4_t x2 = vld1q_f32(xt + j + 8);
				float32x4_t x3 = vld1q_f32(xt + j + 12);
				for (int r = 0; r < 4; r++) {
					uint16x8_t	bw0	 = vld1q_u16(rows[r] + j);
					uint16x8_t	bw1	 = vld1q_u16(rows[r] + j + 8);
					uint32x4_t	lo32 = vshll_n_u16(vget_low_u16(bw0), 16);
					uint32x4_t	hi32 = vshll_n_u16(vget_high_u16(bw0), 16);
					float32x4_t w0	 = vreinterpretq_f32_u32(lo32);
					float32x4_t w1	 = vreinterpretq_f32_u32(hi32);
					acc[r]			 = vfmaq_f32(acc[r], x0, w0);
					acc[r]			 = vfmaq_f32(acc[r], x1, w1);
					lo32			 = vshll_n_u16(vget_low_u16(bw1), 16);
					hi32			 = vshll_n_u16(vget_high_u16(bw1), 16);
					w0				 = vreinterpretq_f32_u32(lo32);
					w1				 = vreinterpretq_f32_u32(hi32);
					acc[r]			 = vfmaq_f32(acc[r], x2, w0);
					acc[r]			 = vfmaq_f32(acc[r], x3, w1);
				}
			}
			for (; j + 8 <= k; j += 8) {
				float32x4_t x0 = vld1q_f32(xt + j);
				float32x4_t x1 = vld1q_f32(xt + j + 4);
				for (int r = 0; r < 4; r++) {
					uint16x8_t	bw	 = vld1q_u16(rows[r] + j);
					uint32x4_t	lo32 = vshll_n_u16(vget_low_u16(bw), 16);
					uint32x4_t	hi32 = vshll_n_u16(vget_high_u16(bw), 16);
					float32x4_t w0	 = vreinterpretq_f32_u32(lo32);
					float32x4_t w1	 = vreinterpretq_f32_u32(hi32);
					acc[r]			 = vfmaq_f32(acc[r], x0, w0);
					acc[r]			 = vfmaq_f32(acc[r], x1, w1);
				}
			}
			float *yt = y + (size_t)t * y_row_stride;
			for (int r = 0; r < 4; r++) {
				float s = vaddvq_f32(acc[r]);
				for (int j2 = j; j2 < k; j2++) {
					union {
						uint32_t u;
						float	 f;
					} v;
					v.u = ((uint32_t)rows[r][j2]) << 16;
					s += v.f * xt[j2];
				}
				yt[i + r] = s;
			}
		}
	}
	for (; i < n; i++) {
		const uint16_t *restrict wr = Wb + ((size_t)i * k);
		for (int t = 0; t < m; t++) {
			const float *xt	  = x + (size_t)t * x_row_stride;
			float32x4_t	 acc0 = vdupq_n_f32(0.0f);
			float32x4_t	 acc1 = vdupq_n_f32(0.0f);
			int			 j	  = 0;
			for (; j + 16 <= k; j += 16) {
				__builtin_prefetch(wr + j + 32, 0, 1);
				float32x4_t x0	 = vld1q_f32(xt + j);
				float32x4_t x1	 = vld1q_f32(xt + j + 4);
				float32x4_t x2	 = vld1q_f32(xt + j + 8);
				float32x4_t x3	 = vld1q_f32(xt + j + 12);
				uint16x8_t	bw0	 = vld1q_u16(wr + j);
				uint16x8_t	bw1	 = vld1q_u16(wr + j + 8);
				uint32x4_t	lo32 = vshll_n_u16(vget_low_u16(bw0), 16);
				uint32x4_t	hi32 = vshll_n_u16(vget_high_u16(bw0), 16);
				acc0			 = vfmaq_f32(acc0, x0, vreinterpretq_f32_u32(lo32));
				acc0			 = vfmaq_f32(acc0, x1, vreinterpretq_f32_u32(hi32));
				lo32			 = vshll_n_u16(vget_low_u16(bw1), 16);
				hi32			 = vshll_n_u16(vget_high_u16(bw1), 16);
				acc1			 = vfmaq_f32(acc1, x2, vreinterpretq_f32_u32(lo32));
				acc1			 = vfmaq_f32(acc1, x3, vreinterpretq_f32_u32(hi32));
			}
			for (; j + 8 <= k; j += 8) {
				float32x4_t x0	 = vld1q_f32(xt + j);
				float32x4_t x1	 = vld1q_f32(xt + j + 4);
				uint16x8_t	bw	 = vld1q_u16(wr + j);
				uint32x4_t	lo32 = vshll_n_u16(vget_low_u16(bw), 16);
				uint32x4_t	hi32 = vshll_n_u16(vget_high_u16(bw), 16);
				acc0			 = vfmaq_f32(acc0, x0, vreinterpretq_f32_u32(lo32));
				acc0			 = vfmaq_f32(acc0, x1, vreinterpretq_f32_u32(hi32));
			}
			float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
			for (; j < k; j++) {
				union {
					uint32_t u;
					float	 f;
				} v;
				v.u = ((uint32_t)wr[j]) << 16;
				s += v.f * xt[j];
			}
			y[(size_t)t * y_row_stride + i] = s;
		}
	}
#endif
}

void matmul_f32_f32(const float *restrict w, const float *restrict x, float *restrict y, int n,
					int k) {
	const float *xp = x;
	float		*yp = y;
	const int	 mr = 8;
	int			 i	= 0;
	for (; i + mr <= n; i += mr) {
		float32x4_t acc[8];
		for (int r = 0; r < 8; r++)
			acc[r] = vdupq_n_f32(0.0f);
		const float *rows[8];
		for (int r = 0; r < 8; r++)
			rows[r] = w + ((size_t)(i + r) * k);
		int j = 0;
		for (; j + 32 <= k; j += 32) {
			__builtin_prefetch(xp + j + 64, 0, 1);
			for (int r = 0; r < 8; r++)
				__builtin_prefetch(rows[r] + j + 64, 0, 1);
			float32x4_t x0 = vld1q_f32(xp + j);
			float32x4_t x1 = vld1q_f32(xp + j + 4);
			float32x4_t x2 = vld1q_f32(xp + j + 8);
			float32x4_t x3 = vld1q_f32(xp + j + 12);
			float32x4_t x4 = vld1q_f32(xp + j + 16);
			float32x4_t x5 = vld1q_f32(xp + j + 20);
			float32x4_t x6 = vld1q_f32(xp + j + 24);
			float32x4_t x7 = vld1q_f32(xp + j + 28);
			for (int r = 0; r < 8; r++) {
				acc[r] = vfmaq_f32(acc[r], x0, vld1q_f32(rows[r] + j));
				acc[r] = vfmaq_f32(acc[r], x1, vld1q_f32(rows[r] + j + 4));
				acc[r] = vfmaq_f32(acc[r], x2, vld1q_f32(rows[r] + j + 8));
				acc[r] = vfmaq_f32(acc[r], x3, vld1q_f32(rows[r] + j + 12));
				acc[r] = vfmaq_f32(acc[r], x4, vld1q_f32(rows[r] + j + 16));
				acc[r] = vfmaq_f32(acc[r], x5, vld1q_f32(rows[r] + j + 20));
				acc[r] = vfmaq_f32(acc[r], x6, vld1q_f32(rows[r] + j + 24));
				acc[r] = vfmaq_f32(acc[r], x7, vld1q_f32(rows[r] + j + 28));
			}
		}
		for (; j + 16 <= k; j += 16) {
			float32x4_t x0 = vld1q_f32(xp + j);
			float32x4_t x1 = vld1q_f32(xp + j + 4);
			float32x4_t x2 = vld1q_f32(xp + j + 8);
			float32x4_t x3 = vld1q_f32(xp + j + 12);
			for (int r = 0; r < 8; r++) {
				acc[r] = vfmaq_f32(acc[r], x0, vld1q_f32(rows[r] + j));
				acc[r] = vfmaq_f32(acc[r], x1, vld1q_f32(rows[r] + j + 4));
				acc[r] = vfmaq_f32(acc[r], x2, vld1q_f32(rows[r] + j + 8));
				acc[r] = vfmaq_f32(acc[r], x3, vld1q_f32(rows[r] + j + 12));
			}
		}
		for (int r = 0; r < 8; r++) {
			float32x4_t rem_acc = vdupq_n_f32(0.0f);
			int			j2		= j;
			for (; j2 + 4 <= k; j2 += 4)
				rem_acc = vfmaq_f32(rem_acc, vld1q_f32(xp + j2), vld1q_f32(rows[r] + j2));
			float s = vaddvq_f32(acc[r]) + vaddvq_f32(rem_acc);
			for (; j2 < k; j2++)
				s += rows[r][j2] * xp[j2];
			yp[i + r] = s;
		}
	}

	for (; i < n; i++) {
		const float *restrict wr = w + ((size_t)i * k);
		float32x4_t acc0		 = vdupq_n_f32(0.0f);
		float32x4_t acc1		 = vdupq_n_f32(0.0f);
		float32x4_t acc2		 = vdupq_n_f32(0.0f);
		float32x4_t acc3		 = vdupq_n_f32(0.0f);
		int			j			 = 0;
		for (; j + 32 <= k; j += 32) {
			__builtin_prefetch(wr + j + 64, 0, 1);
			__builtin_prefetch(xp + j + 64, 0, 1);
			acc0 = vfmaq_f32(acc0, vld1q_f32(xp + j), vld1q_f32(wr + j));
			acc1 = vfmaq_f32(acc1, vld1q_f32(xp + j + 4), vld1q_f32(wr + j + 4));
			acc2 = vfmaq_f32(acc2, vld1q_f32(xp + j + 8), vld1q_f32(wr + j + 8));
			acc3 = vfmaq_f32(acc3, vld1q_f32(xp + j + 12), vld1q_f32(wr + j + 12));
			acc0 = vfmaq_f32(acc0, vld1q_f32(xp + j + 16), vld1q_f32(wr + j + 16));
			acc1 = vfmaq_f32(acc1, vld1q_f32(xp + j + 20), vld1q_f32(wr + j + 20));
			acc2 = vfmaq_f32(acc2, vld1q_f32(xp + j + 24), vld1q_f32(wr + j + 24));
			acc3 = vfmaq_f32(acc3, vld1q_f32(xp + j + 28), vld1q_f32(wr + j + 28));
		}
		for (; j + 16 <= k; j += 16) {
			acc0 = vfmaq_f32(acc0, vld1q_f32(xp + j), vld1q_f32(wr + j));
			acc1 = vfmaq_f32(acc1, vld1q_f32(xp + j + 4), vld1q_f32(wr + j + 4));
			acc2 = vfmaq_f32(acc2, vld1q_f32(xp + j + 8), vld1q_f32(wr + j + 8));
			acc3 = vfmaq_f32(acc3, vld1q_f32(xp + j + 12), vld1q_f32(wr + j + 12));
		}
		for (; j + 4 <= k; j += 4)
			acc0 = vfmaq_f32(acc0, vld1q_f32(xp + j), vld1q_f32(wr + j));
		float s = vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);
		for (; j < k; j++)
			s += wr[j] * xp[j];
		yp[i] = s;
	}
}

void matmul_f32_f32_batch(const float *restrict w, const float *restrict x, float *restrict y,
						  int n, int k, int m, int x_row_stride, int y_row_stride) {
	const int MB = 4;
	int		  mb = 0;
	for (; mb + MB <= m; mb += MB) {
		const float *xb[4];
		for (int t = 0; t < 4; t++)
			xb[t] = x + (size_t)(mb + t) * x_row_stride;

		const int NR = 4;
		int		  i	 = 0;
		for (; i + NR <= n; i += NR) {
			const float *rows[4];
			for (int r = 0; r < 4; r++)
				rows[r] = w + (size_t)(i + r) * k;

			float32x4_t acc[4][4];
			for (int t = 0; t < 4; t++)
				for (int r = 0; r < 4; r++)
					acc[t][r] = vdupq_n_f32(0.0f);

			int j = 0;
			for (; j + 4 <= k; j += 4) {
				if (j + 32 < k) {
					__builtin_prefetch(xb[0] + j + 32, 0, 1);
					for (int r = 0; r < 4; r++)
						__builtin_prefetch(rows[r] + j + 32, 0, 1);
				}
				float32x4_t w0 = vld1q_f32(rows[0] + j);
				float32x4_t w1 = vld1q_f32(rows[1] + j);
				float32x4_t w2 = vld1q_f32(rows[2] + j);
				float32x4_t w3 = vld1q_f32(rows[3] + j);
				for (int t = 0; t < 4; t++) {
					float32x4_t xv = vld1q_f32(xb[t] + j);
					acc[t][0]	   = vfmaq_f32(acc[t][0], xv, w0);
					acc[t][1]	   = vfmaq_f32(acc[t][1], xv, w1);
					acc[t][2]	   = vfmaq_f32(acc[t][2], xv, w2);
					acc[t][3]	   = vfmaq_f32(acc[t][3], xv, w3);
				}
			}
			for (int t = 0; t < 4; t++) {
				float s[4];
				for (int r = 0; r < 4; r++)
					s[r] = vaddvq_f32(acc[t][r]);
				for (int j2 = j; j2 < k; j2++) {
					float xv = xb[t][j2];
					for (int r = 0; r < 4; r++)
						s[r] += rows[r][j2] * xv;
				}
				for (int r = 0; r < 4; r++)
					y[(size_t)(mb + t) * y_row_stride + i + r] = s[r];
			}
		}
		for (; i < n; i++) {
			const float *wr = w + (size_t)i * k;
			for (int t = 0; t < 4; t++) {
				float32x4_t acc = vdupq_n_f32(0.0f);
				int			j	= 0;
				for (; j + 4 <= k; j += 4)
					acc = vfmaq_f32(acc, vld1q_f32(xb[t] + j), vld1q_f32(wr + j));
				float s = vaddvq_f32(acc);
				for (; j < k; j++)
					s += wr[j] * xb[t][j];
				y[(size_t)(mb + t) * y_row_stride + i] = s;
			}
		}
	}
	for (; mb < m; mb++) {
		matmul_f32_f32(w, x + (size_t)mb * x_row_stride, y + (size_t)mb * y_row_stride, n, k);
	}
}

float dot_f32(const float *restrict a, const float *restrict b, int n) {
	float32x4_t acc0 = vdupq_n_f32(0.0f);
	float32x4_t acc1 = vdupq_n_f32(0.0f);
	float32x4_t acc2 = vdupq_n_f32(0.0f);
	float32x4_t acc3 = vdupq_n_f32(0.0f);
	int			i	 = 0;
	for (; i + 32 <= n; i += 32) {
		__builtin_prefetch(a + i + 64, 0, 1);
		__builtin_prefetch(b + i + 64, 0, 1);
		acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
		acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
		acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
		acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
		acc0 = vfmaq_f32(acc0, vld1q_f32(a + i + 16), vld1q_f32(b + i + 16));
		acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 20), vld1q_f32(b + i + 20));
		acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 24), vld1q_f32(b + i + 24));
		acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 28), vld1q_f32(b + i + 28));
	}
	for (; i + 16 <= n; i += 16) {
		acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
		acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
		acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
		acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
	}
	for (; i + 4 <= n; i += 4)
		acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
	float s = vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);
	for (; i < n; i++)
		s += a[i] * b[i];
	return s;
}

static inline float rmsnorm_sum_sq_neon(const float *x, int n) {
	float32x4_t ss_v = vdupq_n_f32(0.0f);
	int			i	 = 0;
	for (; i + 16 <= n; i += 16) {
		float32x4_t x0 = vld1q_f32(x + i);
		float32x4_t x1 = vld1q_f32(x + i + 4);
		float32x4_t x2 = vld1q_f32(x + i + 8);
		float32x4_t x3 = vld1q_f32(x + i + 12);
		ss_v		   = vfmaq_f32(ss_v, x0, x0);
		ss_v		   = vfmaq_f32(ss_v, x1, x1);
		ss_v		   = vfmaq_f32(ss_v, x2, x2);
		ss_v		   = vfmaq_f32(ss_v, x3, x3);
	}
	for (; i + 4 <= n; i += 4) {
		float32x4_t x0 = vld1q_f32(x + i);
		ss_v		   = vfmaq_f32(ss_v, x0, x0);
	}
	float ss = vaddvq_f32(ss_v);
	for (; i < n; i++)
		ss += x[i] * x[i];
	return ss;
}

void rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
	float		ss		= rmsnorm_sum_sq_neon(x, n);
	float		scale	= 1.0f / sqrtf((ss / (float)n) + eps);
	float32x4_t scale_v = vdupq_n_f32(scale);
	int			i		= 0;
	for (; i + 16 <= n; i += 16) {
		float32x4_t x0 = vld1q_f32(x + i);
		float32x4_t x1 = vld1q_f32(x + i + 4);
		float32x4_t x2 = vld1q_f32(x + i + 8);
		float32x4_t x3 = vld1q_f32(x + i + 12);
		vst1q_f32(y + i, vmulq_f32(vmulq_f32(x0, scale_v), vld1q_f32(w + i)));
		vst1q_f32(y + i + 4, vmulq_f32(vmulq_f32(x1, scale_v), vld1q_f32(w + i + 4)));
		vst1q_f32(y + i + 8, vmulq_f32(vmulq_f32(x2, scale_v), vld1q_f32(w + i + 8)));
		vst1q_f32(y + i + 12, vmulq_f32(vmulq_f32(x3, scale_v), vld1q_f32(w + i + 12)));
	}
	for (; i + 4 <= n; i += 4) {
		float32x4_t x0 = vld1q_f32(x + i);
		vst1q_f32(y + i, vmulq_f32(vmulq_f32(x0, scale_v), vld1q_f32(w + i)));
	}
	for (; i < n; i++)
		y[i] = x[i] * scale * w[i];
}

void rmsnorm_noweight(const float *x, float *y, int n, float eps) {
	float		ss		= rmsnorm_sum_sq_neon(x, n);
	float		scale	= 1.0f / sqrtf((ss / (float)n) + eps);
	float32x4_t scale_v = vdupq_n_f32(scale);
	int			i		= 0;
	for (; i + 16 <= n; i += 16) {
		vst1q_f32(y + i, vmulq_f32(vld1q_f32(x + i), scale_v));
		vst1q_f32(y + i + 4, vmulq_f32(vld1q_f32(x + i + 4), scale_v));
		vst1q_f32(y + i + 8, vmulq_f32(vld1q_f32(x + i + 8), scale_v));
		vst1q_f32(y + i + 12, vmulq_f32(vld1q_f32(x + i + 12), scale_v));
	}
	for (; i + 4 <= n; i += 4)
		vst1q_f32(y + i, vmulq_f32(vld1q_f32(x + i), scale_v));
	for (; i < n; i++)
		y[i] = x[i] * scale;
}

void rmsnorm_per_head(const float *x, const float *w, float *y, int n_heads, int head_dim,
					  float eps) {
	for (int h = 0; h < n_heads; h++) {
		const float *xh		 = x + ((size_t)h * head_dim);
		float		*yh		 = y + ((size_t)h * head_dim);
		float		 ss		 = rmsnorm_sum_sq_neon(xh, head_dim);
		float		 scale	 = 1.0f / sqrtf((ss / (float)head_dim) + eps);
		float32x4_t	 scale_v = vdupq_n_f32(scale);
		int			 j		 = 0;
		for (; j + 16 <= head_dim; j += 16) {
			float32x4_t x0 = vld1q_f32(xh + j);
			float32x4_t x1 = vld1q_f32(xh + j + 4);
			float32x4_t x2 = vld1q_f32(xh + j + 8);
			float32x4_t x3 = vld1q_f32(xh + j + 12);
			vst1q_f32(yh + j, vmulq_f32(vmulq_f32(x0, scale_v), vld1q_f32(w + j)));
			vst1q_f32(yh + j + 4, vmulq_f32(vmulq_f32(x1, scale_v), vld1q_f32(w + j + 4)));
			vst1q_f32(yh + j + 8, vmulq_f32(vmulq_f32(x2, scale_v), vld1q_f32(w + j + 8)));
			vst1q_f32(yh + j + 12, vmulq_f32(vmulq_f32(x3, scale_v), vld1q_f32(w + j + 12)));
		}
		for (; j + 4 <= head_dim; j += 4) {
			float32x4_t x0 = vld1q_f32(xh + j);
			vst1q_f32(yh + j, vmulq_f32(vmulq_f32(x0, scale_v), vld1q_f32(w + j)));
		}
		for (; j < head_dim; j++)
			yh[j] = xh[j] * scale * w[j];
	}
}

void softmax_masked(float *restrict scores, int n_valid) {
	float32x4_t mx_v = vdupq_n_f32(-INFINITY);
	int			i	 = 0;
	for (; i + 16 <= n_valid; i += 16) {
		mx_v = vmaxq_f32(mx_v, vld1q_f32(scores + i));
		mx_v = vmaxq_f32(mx_v, vld1q_f32(scores + i + 4));
		mx_v = vmaxq_f32(mx_v, vld1q_f32(scores + i + 8));
		mx_v = vmaxq_f32(mx_v, vld1q_f32(scores + i + 12));
	}
	for (; i + 4 <= n_valid; i += 4)
		mx_v = vmaxq_f32(mx_v, vld1q_f32(scores + i));
	float mx = vmaxvq_f32(mx_v);
	for (; i < n_valid; i++)
		if (scores[i] > mx)
			mx = scores[i];

	float32x4_t mx_vec = vdupq_n_f32(mx);
	float32x4_t sum_v  = vdupq_n_f32(0.0f);
	i				   = 0;
	for (; i + 16 <= n_valid; i += 16) {
		float32x4_t s0 = vsubq_f32(vld1q_f32(scores + i), mx_vec);
		float32x4_t s1 = vsubq_f32(vld1q_f32(scores + i + 4), mx_vec);
		float32x4_t s2 = vsubq_f32(vld1q_f32(scores + i + 8), mx_vec);
		float32x4_t s3 = vsubq_f32(vld1q_f32(scores + i + 12), mx_vec);
		float32x4_t v0 = vexpq_f32(s0);
		float32x4_t v1 = vexpq_f32(s1);
		float32x4_t v2 = vexpq_f32(s2);
		float32x4_t v3 = vexpq_f32(s3);
		vst1q_f32(scores + i, v0);
		vst1q_f32(scores + i + 4, v1);
		vst1q_f32(scores + i + 8, v2);
		vst1q_f32(scores + i + 12, v3);
		sum_v = vaddq_f32(sum_v, v0);
		sum_v = vaddq_f32(sum_v, v1);
		sum_v = vaddq_f32(sum_v, v2);
		sum_v = vaddq_f32(sum_v, v3);
	}
	for (; i + 4 <= n_valid; i += 4) {
		float32x4_t s = vsubq_f32(vld1q_f32(scores + i), mx_vec);
		float32x4_t v = vexpq_f32(s);
		vst1q_f32(scores + i, v);
		sum_v = vaddq_f32(sum_v, v);
	}
	float sum = vaddvq_f32(sum_v);
	for (; i < n_valid; i++) {
		scores[i] = expf(scores[i] - mx);
		sum += scores[i];
	}
	float		inv	  = 1.0f / sum;
	float32x4_t inv_v = vdupq_n_f32(inv);
	i				  = 0;
	for (; i + 16 <= n_valid; i += 16) {
		vst1q_f32(scores + i, vmulq_f32(vld1q_f32(scores + i), inv_v));
		vst1q_f32(scores + i + 4, vmulq_f32(vld1q_f32(scores + i + 4), inv_v));
		vst1q_f32(scores + i + 8, vmulq_f32(vld1q_f32(scores + i + 8), inv_v));
		vst1q_f32(scores + i + 12, vmulq_f32(vld1q_f32(scores + i + 12), inv_v));
	}
	for (; i + 4 <= n_valid; i += 4)
		vst1q_f32(scores + i, vmulq_f32(vld1q_f32(scores + i), inv_v));
	for (; i < n_valid; i++)
		scores[i] *= inv;
}

void moe_activate_silu(float *restrict act, const float *restrict gate, const float *restrict up,
					   int n, float gs, float us) {
	float32x4_t gs_v  = vdupq_n_f32(gs);
	float32x4_t us_v  = vdupq_n_f32(us);
	float32x4_t one_v = vdupq_n_f32(1.0f);
	int			i	  = 0;
	for (; i + 16 <= n; i += 16) {
		float32x4_t g0 = vmulq_f32(vld1q_f32(gate + i), gs_v);
		float32x4_t g1 = vmulq_f32(vld1q_f32(gate + i + 4), gs_v);
		float32x4_t g2 = vmulq_f32(vld1q_f32(gate + i + 8), gs_v);
		float32x4_t g3 = vmulq_f32(vld1q_f32(gate + i + 12), gs_v);
		float32x4_t s0 = vmulq_f32(g0, vrecpeq_f32(vaddq_f32(one_v, vexpq_f32(vnegq_f32(g0)))));
		float32x4_t s1 = vmulq_f32(g1, vrecpeq_f32(vaddq_f32(one_v, vexpq_f32(vnegq_f32(g1)))));
		float32x4_t s2 = vmulq_f32(g2, vrecpeq_f32(vaddq_f32(one_v, vexpq_f32(vnegq_f32(g2)))));
		float32x4_t s3 = vmulq_f32(g3, vrecpeq_f32(vaddq_f32(one_v, vexpq_f32(vnegq_f32(g3)))));
		float32x4_t d0 = vaddq_f32(one_v, vexpq_f32(vnegq_f32(g0)));
		float32x4_t d1 = vaddq_f32(one_v, vexpq_f32(vnegq_f32(g1)));
		float32x4_t d2 = vaddq_f32(one_v, vexpq_f32(vnegq_f32(g2)));
		float32x4_t d3 = vaddq_f32(one_v, vexpq_f32(vnegq_f32(g3)));
		s0			   = vmulq_f32(s0, vmulq_f32(vrecpsq_f32(d0, s0), s0));
		s1			   = vmulq_f32(s1, vmulq_f32(vrecpsq_f32(d1, s1), s1));
		s2			   = vmulq_f32(s2, vmulq_f32(vrecpsq_f32(d2, s2), s2));
		s3			   = vmulq_f32(s3, vmulq_f32(vrecpsq_f32(d3, s3), s3));
		vst1q_f32(act + i, vmulq_f32(s0, vmulq_f32(vld1q_f32(up + i), us_v)));
		vst1q_f32(act + i + 4, vmulq_f32(s1, vmulq_f32(vld1q_f32(up + i + 4), us_v)));
		vst1q_f32(act + i + 8, vmulq_f32(s2, vmulq_f32(vld1q_f32(up + i + 8), us_v)));
		vst1q_f32(act + i + 12, vmulq_f32(s3, vmulq_f32(vld1q_f32(up + i + 12), us_v)));
	}
	for (; i + 4 <= n; i += 4) {
		float32x4_t g = vmulq_f32(vld1q_f32(gate + i), gs_v);
		float32x4_t d = vaddq_f32(one_v, vexpq_f32(vnegq_f32(g)));
		float32x4_t r = vrecpeq_f32(d);
		r			  = vmulq_f32(r, vmulq_f32(vrecpsq_f32(d, r), r));
		float32x4_t s = vmulq_f32(g, r);
		vst1q_f32(act + i, vmulq_f32(s, vmulq_f32(vld1q_f32(up + i), us_v)));
	}
	for (; i < n; i++)
		act[i] = silu(gate[i] * gs) * (up[i] * us);
}

void moe_activate_gelu(float *restrict act, const float *restrict gate, const float *restrict up,
					   int n, float gs, float us) {
	float32x4_t gs_v   = vdupq_n_f32(gs);
	float32x4_t us_v   = vdupq_n_f32(us);
	float32x4_t half_v = vdupq_n_f32(0.5f);
	float32x4_t one_v  = vdupq_n_f32(1.0f);
	float32x4_t c_v	   = vdupq_n_f32(0.7978845608028654f);
	float32x4_t k_v	   = vdupq_n_f32(0.044715f);
	int			i	   = 0;
	for (; i + 16 <= n; i += 16) {
		float32x4_t g0	 = vmulq_f32(vld1q_f32(gate + i), gs_v);
		float32x4_t g1	 = vmulq_f32(vld1q_f32(gate + i + 4), gs_v);
		float32x4_t g2	 = vmulq_f32(vld1q_f32(gate + i + 8), gs_v);
		float32x4_t g3	 = vmulq_f32(vld1q_f32(gate + i + 12), gs_v);
		float32x4_t x3_0 = vmulq_f32(g0, vmulq_f32(g0, g0));
		float32x4_t x3_1 = vmulq_f32(g1, vmulq_f32(g1, g1));
		float32x4_t x3_2 = vmulq_f32(g2, vmulq_f32(g2, g2));
		float32x4_t x3_3 = vmulq_f32(g3, vmulq_f32(g3, g3));
		float32x4_t in0	 = vmulq_f32(c_v, vaddq_f32(g0, vmulq_f32(k_v, x3_0)));
		float32x4_t in1	 = vmulq_f32(c_v, vaddq_f32(g1, vmulq_f32(k_v, x3_1)));
		float32x4_t in2	 = vmulq_f32(c_v, vaddq_f32(g2, vmulq_f32(k_v, x3_2)));
		float32x4_t in3	 = vmulq_f32(c_v, vaddq_f32(g3, vmulq_f32(k_v, x3_3)));
		float32x4_t t0	 = vtanhq_f32(in0);
		float32x4_t t1	 = vtanhq_f32(in1);
		float32x4_t t2	 = vtanhq_f32(in2);
		float32x4_t t3	 = vtanhq_f32(in3);
		float32x4_t s0	 = vmulq_f32(half_v, vmulq_f32(g0, vaddq_f32(one_v, t0)));
		float32x4_t s1	 = vmulq_f32(half_v, vmulq_f32(g1, vaddq_f32(one_v, t1)));
		float32x4_t s2	 = vmulq_f32(half_v, vmulq_f32(g2, vaddq_f32(one_v, t2)));
		float32x4_t s3	 = vmulq_f32(half_v, vmulq_f32(g3, vaddq_f32(one_v, t3)));
		vst1q_f32(act + i, vmulq_f32(s0, vmulq_f32(vld1q_f32(up + i), us_v)));
		vst1q_f32(act + i + 4, vmulq_f32(s1, vmulq_f32(vld1q_f32(up + i + 4), us_v)));
		vst1q_f32(act + i + 8, vmulq_f32(s2, vmulq_f32(vld1q_f32(up + i + 8), us_v)));
		vst1q_f32(act + i + 12, vmulq_f32(s3, vmulq_f32(vld1q_f32(up + i + 12), us_v)));
	}
	for (; i + 4 <= n; i += 4) {
		float32x4_t g  = vmulq_f32(vld1q_f32(gate + i), gs_v);
		float32x4_t x3 = vmulq_f32(g, vmulq_f32(g, g));
		float32x4_t in = vmulq_f32(c_v, vaddq_f32(g, vmulq_f32(k_v, x3)));
		float32x4_t t  = vtanhq_f32(in);
		float32x4_t s  = vmulq_f32(half_v, vmulq_f32(g, vaddq_f32(one_v, t)));
		vst1q_f32(act + i, vmulq_f32(s, vmulq_f32(vld1q_f32(up + i), us_v)));
	}
	for (; i < n; i++)
		act[i] = gelu_tanh(gate[i] * gs) * (up[i] * us);
}

void repack_iq4_nl_to_q8_0_rows(const void *src, void *dst, int row_begin, int row_end, int k) {
	const int	 blocks_per_row = k / 32;
	const size_t src_stride		= (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const size_t dst_stride		= (size_t)blocks_per_row * sizeof(q8_0_block);

	const uint8_t *sp = src;
	uint8_t		  *dp = dst;

	const uint8x16_t kvalues_u = vreinterpretq_u8_s8(vld1q_s8(kvalues_iq4nl));
	const uint8x16_t lo_mask   = vdupq_n_u8(0x0F);

	for (int r = row_begin; r < row_end; r++) {
		const iq4_nl_block *srow = (const iq4_nl_block *)(sp + (size_t)r * src_stride);
		q8_0_block		   *drow = (q8_0_block *)(dp + (size_t)r * dst_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			drow[bi].d				= srow[bi].d;
			const uint8x16_t qs		= vld1q_u8(srow[bi].qs);
			const uint8x16_t lo_idx = vandq_u8(qs, lo_mask);
			const uint8x16_t hi_idx = vshrq_n_u8(qs, 4);
			vst1q_s8(drow[bi].qs, vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, lo_idx)));
			vst1q_s8(drow[bi].qs + 16, vreinterpretq_s8_u8(vqtbl1q_u8(kvalues_u, hi_idx)));
		}
	}
}