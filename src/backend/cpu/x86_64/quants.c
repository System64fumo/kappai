#include "backend/cpu/scalar/quants.h"
#include "backend/cpu/x86_64/common.h"
#include "common.h"
#include "threadpool.h"

#include <stdint.h>
#include <string.h>

#define MR 8
_Static_assert(MR % 4 == 0, "matmul_iq4_nl_q8_qonly_f32 assumes MR is a multiple of 4");

/* Caches holding __m256i lanes must be at least 32-byte (we use 64) aligned:
 * the compiler emits aligned vmovdqa stores for __m256i-typed objects, and
 * malloc/realloc only guarantee 16-byte alignment. */
static inline void *ymm_cache_alloc(size_t bytes) {
	return xmalloc_aligned(bytes, 64);
}

typedef struct {
	uint16_t d;
	uint8_t	 qs[64];
	uint8_t	 qh[8];
	uint8_t	 signs[32];
	uint8_t	 scales[4];
} iq3s_block;

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static inline __m128 loadu_f16x4_to_ps_128(const uint16_t *p) {
	__m128i h = _mm_loadl_epi64((const __m128i *)(p));
	return _mm_cvtph_ps(h);
}

static inline __m256 loadu_f16x8_to_ps_256(const uint16_t *p) {
	return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(p)));
}

static inline float vreduce_add_ps_128(__m128 v) {
	v = _mm_add_ps(v, _mm_movehl_ps(v, v));
	v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
	return _mm_cvtss_f32(v);
}

void dequant_iq3_s_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq3s_block *b = blocks;

	const __m256i mask_vals_all =
		_mm256_setr_epi32(kmask_iq2xs[0], kmask_iq2xs[1], kmask_iq2xs[2], kmask_iq2xs[3],
						  kmask_iq2xs[4], kmask_iq2xs[5], kmask_iq2xs[6], kmask_iq2xs[7]);
	const __m256i zero = _mm256_setzero_si256();

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

			__m256 db1_vec = _mm256_set1_ps(db1);
			__m256 db2_vec = _mm256_set1_ps(db2);
			__m256 neg_db1 = _mm256_sub_ps(_mm256_setzero_ps(), db1_vec);
			__m256 neg_db2 = _mm256_sub_ps(_mm256_setzero_ps(), db2_vec);

			uint8_t qh0 = qh[0];
			uint8_t qh1 = qh[1];

			for (int l = 0; l < 4; l++) {
				uint8_t	 q0  = qs[2 * l];
				uint8_t	 q1  = qs[(2 * l) + 1];
				uint32_t idx0 = q0 | ((qh0 << (8 - (2 * l))) & 256);
				uint32_t idx1 = q1 | ((qh0 << (7 - (2 * l))) & 256);

				uint32_t g0 = iq3s_grid[idx0];
				uint32_t g1 = iq3s_grid[idx1];

				uint8_t bytes[8];
				memcpy(bytes, &g0, 4);
				memcpy(bytes + 4, &g1, 4);
				__m256i grid_i = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(bytes)));
				__m256	grid_f = _mm256_cvtepi32_ps(grid_i);

				__m256i sign_vec = _mm256_set1_epi32(signs[l]);
				__m256i test	 = _mm256_and_si256(sign_vec, mask_vals_all);
				__m256i msk		 = _mm256_cmpgt_epi32(test, zero);
				__m256	signed_db = _mm256_blendv_ps(db1_vec, neg_db1, _mm256_castsi256_ps(msk));

				_mm256_storeu_ps(y, _mm256_mul_ps(signed_db, grid_f));
				y += 8;
			}
			qs += 8;
			signs += 4;

			for (int l = 0; l < 4; l++) {
				uint8_t	 q0  = qs[2 * l];
				uint8_t	 q1  = qs[(2 * l) + 1];
				uint32_t idx0 = q0 | ((qh1 << (8 - (2 * l))) & 256);
				uint32_t idx1 = q1 | ((qh1 << (7 - (2 * l))) & 256);

				uint32_t g0 = iq3s_grid[idx0];
				uint32_t g1 = iq3s_grid[idx1];

				uint8_t bytes[8];
				memcpy(bytes, &g0, 4);
				memcpy(bytes + 4, &g1, 4);
				__m256i grid_i = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(bytes)));
				__m256	grid_f = _mm256_cvtepi32_ps(grid_i);

				__m256i sign_vec = _mm256_set1_epi32(signs[l]);
				__m256i test	 = _mm256_and_si256(sign_vec, mask_vals_all);
				__m256i msk		 = _mm256_cmpgt_epi32(test, zero);
				__m256	signed_db = _mm256_blendv_ps(db2_vec, neg_db2, _mm256_castsi256_ps(msk));

				_mm256_storeu_ps(y, _mm256_mul_ps(signed_db, grid_f));
				y += 8;
			}
			qh += 2;
			qs += 8;
			signs += 4;
		}
	}
}

#define IQ3S_RE_OFF_D 0
#define IQ3S_RE_OFF_SCALES 2
#define IQ3S_RE_OFF_IDX 6

static const int8_t iq3s_re_decode_tbl[16] = {
	1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15,
};

static inline __m256i iq3s_re_unpack_group(const uint8_t *idx_ptr, __m256i tbl) {
	__m128i packed = _mm_loadu_si128((const __m128i *)(idx_ptr));
	__m128i lo	 = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
	__m128i hi	 = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
	__m256i nib	 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
	return _mm256_shuffle_epi8(tbl, nib);
}

static void matmul_iq3_s_re_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
											   float *restrict y, int n, int k) {
	const int	 blocks_per_row = k / 256;
	const size_t row_stride		= (size_t)blocks_per_row * IQ3_S_RE_BLOCK_BYTES;
	const uint8_t *Wb			= w;

	__m256i tbl =
		_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq3s_re_decode_tbl)));
	int i = 0;

	for (; i + 8 <= n; i += 8) {
		__m128 acc0 = _mm_setzero_ps();
		__m128 acc1 = _mm_setzero_ps();

		const uint8_t *row_base[8];
		for (int r = 0; r < 8; r++)
			row_base[r] = Wb + (size_t)(i + r) * row_stride;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float		  d_xq		  = yb->d;
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

				int32_t total = 0;
				for (int g = 0; g < 8; g += 2) {
					uint8_t sb	= scales[g / 2];
					int		sc0 = 1 + 2 * (sb & 0xf);
					int		sc1 = 1 + 2 * (sb >> 4);

					__m256i d0 = iq3s_re_unpack_group(idx + g * 16, tbl);
					__m256i d1 = iq3s_re_unpack_group(idx + (g + 1) * 16, tbl);

					const int8_t *restrict qg0 = q8 + g * 32;
					const int8_t *restrict qg1 = q8 + (g + 1) * 32;

					int32_t a0 =
						vreduce_add_epi32(dotprod_s8_s8_i32(d0, _mm256_loadu_si256(
															  (const __m256i *)(qg0))));
					int32_t a1 =
						vreduce_add_epi32(dotprod_s8_s8_i32(d1, _mm256_loadu_si256(
															  (const __m256i *)(qg1))));

					total += sc0 * a0;
					total += sc1 * a1;
				}
				sumi_lane[r] = total;
			}

			__m128 sumi0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane)));
			__m128 sumi1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane + 4)));
			__m128 d_w0	 = _mm_loadu_ps(d_w);
			__m128 d_w1	 = _mm_loadu_ps(d_w + 4);
			__m128 d_xq_v = _mm_set1_ps(d_xq);

			acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_mul_ps(d_w0, d_xq_v), sumi0));
			acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_mul_ps(d_w1, d_xq_v), sumi1));
		}

		float tmp0[4], tmp1[4];
		_mm_storeu_ps(tmp0, acc0);
		_mm_storeu_ps(tmp1, acc1);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
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
			int32_t total				  = 0;

			for (int g = 0; g < 8; g += 2) {
				uint8_t sb	= scales[g / 2];
				int		sc0 = 1 + 2 * (sb & 0xf);
				int		sc1 = 1 + 2 * (sb >> 4);

				__m256i d0 = iq3s_re_unpack_group(idx + g * 16, tbl);
				__m256i d1 = iq3s_re_unpack_group(idx + (g + 1) * 16, tbl);

				const int8_t *restrict qg0 = q8 + g * 32;
				const int8_t *restrict qg1 = q8 + (g + 1) * 32;

				int32_t a0 =
					vreduce_add_epi32(dotprod_s8_s8_i32(d0, _mm256_loadu_si256(
														  (const __m256i *)(qg0))));
				int32_t a1 =
					vreduce_add_epi32(dotprod_s8_s8_i32(d1, _mm256_loadu_si256(
														  (const __m256i *)(qg1))));

				total += sc0 * a0;
				total += sc1 * a1;
			}
			sumf += d_w * yb->d * (float)total;
		}
		y[i] = sumf;
	}
}

#define NR_IQ3S_RE 4
void matmul_iq3_s_re_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
									size_t		xq_row_stride_blocks, float *restrict y,
									int y_row_stride, int n, int k, int m) {
	const int	 blocks_per_row = k / 256;
	const size_t row_stride		= (size_t)blocks_per_row * IQ3_S_RE_BLOCK_BYTES;
	const uint8_t *Wb			= w;
	__m256i tbl =
		_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq3s_re_decode_tbl)));
	int i = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = Wb + (size_t)(i + r) * row_stride;

		int t = 0;
		for (; t + NR_IQ3S_RE <= m; t += NR_IQ3S_RE) {
			__m128 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_k_block *xrow[NR_IQ3S_RE];
			for (int c = 0; c < NR_IQ3S_RE; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	d_w[MR];
				__m256i decoded[MR][8];
				int32_t sc0[MR][4], sc1[MR][4];

				for (int r = 0; r < MR; r++) {
					const uint8_t *blk	  = row_base[r] + (size_t)bi * IQ3_S_RE_BLOCK_BYTES;
					d_w[r]				  = f16_to_f32_fast(*(const uint16_t *)(blk + IQ3S_RE_OFF_D));
					const uint8_t *scales = blk + IQ3S_RE_OFF_SCALES;
					const uint8_t *idx	  = blk + IQ3S_RE_OFF_IDX;

					for (int g = 0; g < 8; g += 2) {
						uint8_t sb	  = scales[g / 2];
						sc0[r][g / 2] = 1 + 2 * (sb & 0xf);
						sc1[r][g / 2] = 1 + 2 * (sb >> 4);

						decoded[r][g]	  = iq3s_re_unpack_group(idx + g * 16, tbl);
						decoded[r][g + 1] = iq3s_re_unpack_group(idx + (g + 1) * 16, tbl);
					}
				}

				float xd[NR_IQ3S_RE];
				const int8_t *restrict q8p[NR_IQ3S_RE];
				for (int c = 0; c < NR_IQ3S_RE; c++) {
					xd[c]  = xrow[c][bi].d;
					q8p[c] = xrow[c][bi].qs;
				}
				__m128 xd_vec = _mm_loadu_ps(xd);

				for (int r = 0; r < MR; r++) {
					int32_t total_arr[4] = {0, 0, 0, 0};

					for (int g = 0; g < 8; g += 2) {
						const __m256i c0 = decoded[r][g];
						const __m256i c1 = decoded[r][g + 1];
						const int	  sc0v = sc0[r][g / 2];
						const int	  sc1v = sc1[r][g / 2];

						for (int c = 0; c < 4; c++) {
							const int8_t *restrict qg0 = q8p[c] + g * 32;
							const int8_t *restrict qg1 = q8p[c] + (g + 1) * 32;
							int32_t a0 =
								vreduce_add_epi32(dotprod_s8_s8_i32(
									c0, _mm256_loadu_si256((const __m256i *)(qg0))));
							int32_t a1 =
								vreduce_add_epi32(dotprod_s8_s8_i32(
									c1, _mm256_loadu_si256((const __m256i *)(qg1))));
							total_arr[c] += sc0v * a0 + sc1v * a1;
						}
					}

					const __m128i total4 = _mm_loadu_si128((const __m128i *)total_arr);
					__m128 total4f = _mm_cvtepi32_ps(total4);
					__m128 dw		 = _mm_set1_ps(d_w[r]);
					acc_row[r] = _mm_add_ps(acc_row[r], _mm_mul_ps(_mm_mul_ps(dw, total4f), xd_vec));
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
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
	const int	 blocks_per_row = k / 256;
	const size_t row_stride		= (size_t)blocks_per_row * IQ3_S_RE8_GROUP_BYTES;
	const uint8_t *Wb			= w;
	__m256i tbl =
		_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq3s_re_decode_tbl)));
	int i = 0;

	for (; i + 8 <= n; i += 8) {
		__m128 acc0 = _mm_setzero_ps();
		__m128 acc1 = _mm_setzero_ps();

		const uint8_t *group = Wb + (size_t)(i / 8) * row_stride;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float		  d_xq		  = yb->d;
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

				__m128i totalv = _mm_setzero_si128();

				for (int g = 0; g < 8; g += 2) {
					uint8_t sb	= scales[g / 2];
					int32_t sc0 = 1 + 2 * (sb & 0xf);
					int32_t sc1 = 1 + 2 * (sb >> 4);

					__m256i d0 = iq3s_re_unpack_group(idx + g * 16, tbl);
					__m256i d1 = iq3s_re_unpack_group(idx + (g + 1) * 16, tbl);

					const int8_t *restrict qg0 = q8 + g * 32;
					const int8_t *restrict qg1 = q8 + (g + 1) * 32;

					int32_t a0 =
						vreduce_add_epi32(dotprod_s8_s8_i32(d0, _mm256_loadu_si256(
															  (const __m256i *)(qg0))));
					int32_t a1 =
						vreduce_add_epi32(dotprod_s8_s8_i32(d1, _mm256_loadu_si256(
															  (const __m256i *)(qg1))));

					__m128i sc0v = _mm_set1_epi32(sc0);
					__m128i sc1v = _mm_set1_epi32(sc1);
					totalv		 = _mm_add_epi32(totalv, _mm_mullo_epi32(sc0v, _mm_set1_epi32(a0)));
					totalv		 = _mm_add_epi32(totalv, _mm_mullo_epi32(sc1v, _mm_set1_epi32(a1)));
				}
				sumi_lane[r] = _mm_cvtsi128_si32(totalv);
			}

			__m128 sumi0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane)));
			__m128 sumi1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane + 4)));
			__m128 d_w0	 = _mm_loadu_ps(d_w);
			__m128 d_w1	 = _mm_loadu_ps(d_w + 4);
			__m128 d_xq_v = _mm_set1_ps(d_xq);

			acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_mul_ps(d_w0, d_xq_v), sumi0));
			acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_mul_ps(d_w1, d_xq_v), sumi1));
		}

		float tmp0[4], tmp1[4];
		_mm_storeu_ps(tmp0, acc0);
		_mm_storeu_ps(tmp1, acc1);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
	}

	for (; i < n; i++) {
		const uint8_t *group = Wb + (size_t)(i / 8) * row_stride;
		float		   sumf	 = 0.0f;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float		  d_xq		  = yb->d;
			const int8_t *restrict q8	  = yb->qs;

			const uint8_t  *blk		   = group + (size_t)bi * IQ3_S_RE8_GROUP_BYTES;
			const uint16_t *d_ptr	   = (const uint16_t *)(blk + IQ3S_RE8_OFF_D);
			const uint8_t  *scales_all = blk + IQ3S_RE8_OFF_SCALES;
			const uint8_t  *idx_all	   = blk + IQ3S_RE8_OFF_IDX;

			float		   d_w	  = f16_to_f32_fast(d_ptr[i % 8]);
			const uint8_t *scales = scales_all + (size_t)(i % 8) * 4;
			const uint8_t *idx	  = idx_all + (size_t)(i % 8) * 128;

			__m128i totalv = _mm_setzero_si128();

			for (int g = 0; g < 8; g += 2) {
				uint8_t sb	= scales[g / 2];
				int32_t sc0 = 1 + 2 * (sb & 0xf);
				int32_t sc1 = 1 + 2 * (sb >> 4);

				__m256i d0 = iq3s_re_unpack_group(idx + g * 16, tbl);
				__m256i d1 = iq3s_re_unpack_group(idx + (g + 1) * 16, tbl);

				const int8_t *restrict qg0 = q8 + g * 32;
				const int8_t *restrict qg1 = q8 + (g + 1) * 32;

				int32_t a0 =
					vreduce_add_epi32(dotprod_s8_s8_i32(d0, _mm256_loadu_si256(
														  (const __m256i *)(qg0))));
				int32_t a1 =
					vreduce_add_epi32(dotprod_s8_s8_i32(d1, _mm256_loadu_si256(
														  (const __m256i *)(qg1))));

				__m128i sc0v = _mm_set1_epi32(sc0);
				__m128i sc1v = _mm_set1_epi32(sc1);
				totalv		 = _mm_add_epi32(totalv, _mm_mullo_epi32(sc0v, _mm_set1_epi32(a0)));
				totalv		 = _mm_add_epi32(totalv, _mm_mullo_epi32(sc1v, _mm_set1_epi32(a1)));
			}

			int32_t total = _mm_cvtsi128_si32(totalv);
			sumf += d_w * d_xq * (float)total;
		}
		y[i] = sumf;
	}
}

#define NR_IQ3S_RE8 4
void matmul_iq3_s_re8_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
									 size_t		 xq_row_stride_blocks, float *restrict y,
									 int y_row_stride, int n, int k, int m) {
	const int	 blocks_per_row = k / 256;
	const size_t row_stride		= (size_t)blocks_per_row * IQ3_S_RE8_GROUP_BYTES;
	const uint8_t *Wb			= w;
	__m256i tbl =
		_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq3s_re_decode_tbl)));
	int i = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + (size_t)(i / 8) * row_stride;
		int			   t	 = 0;

		for (; t + NR_IQ3S_RE8 <= m; t += NR_IQ3S_RE8) {
			__m128 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_k_block *xrow[NR_IQ3S_RE8];

			for (int c = 0; c < NR_IQ3S_RE8; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * IQ3_S_RE8_GROUP_BYTES;
				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + IQ3_S_RE8_GROUP_BYTES, 0, 1);

				const uint16_t *d_ptr	   = (const uint16_t *)(blk + IQ3S_RE8_OFF_D);
				const uint8_t  *scales_all = blk + IQ3S_RE8_OFF_SCALES;
				const uint8_t  *idx_all	   = blk + IQ3S_RE8_OFF_IDX;

				float	d_w[MR];
				__m256i decoded[MR][8];
				int32_t sc0[MR][4], sc1[MR][4];

				for (int r = 0; r < MR; r++) {
					d_w[r]				  = f16_to_f32_fast(d_ptr[r]);
					const uint8_t *scales = scales_all + (size_t)r * 4;
					const uint8_t *idx	  = idx_all + (size_t)r * 128;

					for (int g = 0; g < 8; g += 2) {
						uint8_t sb	  = scales[g / 2];
						sc0[r][g / 2] = 1 + 2 * (sb & 0xf);
						sc1[r][g / 2] = 1 + 2 * (sb >> 4);

						decoded[r][g]	  = iq3s_re_unpack_group(idx + g * 16, tbl);
						decoded[r][g + 1] = iq3s_re_unpack_group(idx + (g + 1) * 16, tbl);
					}
				}

				float xd[NR_IQ3S_RE8];
				const int8_t *restrict q8p[NR_IQ3S_RE8];
				for (int c = 0; c < NR_IQ3S_RE8; c++) {
					xd[c]  = xrow[c][bi].d;
					q8p[c] = xrow[c][bi].qs;
				}
				__m128 xd_vec = _mm_loadu_ps(xd);

				for (int r = 0; r < MR; r++) {
					int32_t total_arr[4] = {0, 0, 0, 0};

					for (int g = 0; g < 8; g += 2) {
						const __m256i c0 = decoded[r][g];
						const __m256i c1 = decoded[r][g + 1];
						const int	  sc0v = sc0[r][g / 2];
						const int	  sc1v = sc1[r][g / 2];

						for (int c = 0; c < 4; c++) {
							const int8_t *restrict qg0 = q8p[c] + g * 32;
							const int8_t *restrict qg1 = q8p[c] + (g + 1) * 32;
							int32_t a0 =
								vreduce_add_epi32(dotprod_s8_s8_i32(
									c0, _mm256_loadu_si256((const __m256i *)(qg0))));
							int32_t a1 =
								vreduce_add_epi32(dotprod_s8_s8_i32(
									c1, _mm256_loadu_si256((const __m256i *)(qg1))));
							total_arr[c] += sc0v * a0 + sc1v * a1;
						}
					}

					const __m128i total4 = _mm_loadu_si128((const __m128i *)total_arr);
					__m128 total4f = _mm_cvtepi32_ps(total4);
					__m128 dw		 = _mm_set1_ps(d_w[r]);
					acc_row[r] = _mm_add_ps(acc_row[r], _mm_mul_ps(_mm_mul_ps(dw, total4f), xd_vec));
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
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

static uint8_t iq3s_sign_pattern[256][8];

static void __attribute__((constructor)) iq3s_sign_pattern_init(void) {
	for (int byte = 0; byte < 256; byte++) {
		for (int j = 0; j < 8; j++) {
			iq3s_sign_pattern[byte][j] = (byte & (1 << j)) ? 0xFF : 0x00;
		}
	}
}

static inline __m128i iq3s_flip8(uint8_t q0, uint8_t q1, uint8_t qh, uint8_t sign_byte, int sh0,
								 int sh1) {
	uint32_t idx0 = q0 | ((qh << sh0) & 256);
	uint32_t idx1 = q1 | ((qh << sh1) & 256);

	uint32_t g0 = iq3s_grid[idx0];
	uint32_t g1 = iq3s_grid[idx1];

	uint8_t gr[8];
	memcpy(gr, &g0, 4);
	memcpy(gr + 4, &g1, 4);

	const uint8_t *nm = iq3s_sign_pattern[sign_byte];
	uint8_t		   out[8];
	for (int j = 0; j < 8; j++)
		out[j] = (uint8_t)((gr[j] ^ nm[j]) - nm[j]);

	return _mm_loadl_epi64((const __m128i *)(out));
}

static inline int32_t iq3s_block_dot(const iq3s_block *restrict b, const int8_t *restrict q8) {
	const uint8_t *restrict qs	  = b->qs;
	const uint8_t *restrict qh	  = b->qh;
	const uint8_t *restrict signs = b->signs;

	__m128i decoded[16];

	{
		const uint8_t *restrict qs_p	= qs;
		const uint8_t *restrict qh_p	= qh;
		const uint8_t *restrict signs_p = signs;

		for (int ib32 = 0; ib32 < 8; ib32 += 2) {
			const uint8_t qh0 = qh_p[0];
			const uint8_t qh1 = qh_p[1];
			const int	  vi  = ib32 * 2;

			const __m128i f00 = iq3s_flip8(qs_p[0], qs_p[1], qh0, signs_p[0], 8, 7);
			const __m128i f01 = iq3s_flip8(qs_p[2], qs_p[3], qh0, signs_p[1], 6, 5);
			const __m128i f02 = iq3s_flip8(qs_p[4], qs_p[5], qh0, signs_p[2], 4, 3);
			const __m128i f03 = iq3s_flip8(qs_p[6], qs_p[7], qh0, signs_p[3], 2, 1);
			const __m128i f10 = iq3s_flip8(qs_p[8], qs_p[9], qh1, signs_p[4], 8, 7);
			const __m128i f11 = iq3s_flip8(qs_p[10], qs_p[11], qh1, signs_p[5], 6, 5);
			const __m128i f12 = iq3s_flip8(qs_p[12], qs_p[13], qh1, signs_p[6], 4, 3);
			const __m128i f13 = iq3s_flip8(qs_p[14], qs_p[15], qh1, signs_p[7], 2, 1);

			decoded[vi + 0] = _mm_unpacklo_epi64(f00, f01);
			decoded[vi + 1] = _mm_unpacklo_epi64(f02, f03);
			decoded[vi + 2] = _mm_unpacklo_epi64(f10, f11);
			decoded[vi + 3] = _mm_unpacklo_epi64(f12, f13);

			qh_p += 2;
			qs_p += 16;
			signs_p += 8;
		}
	}

	int32_t total = 0;
	for (int ib32 = 0; ib32 < 8; ib32 += 2) {
		const int	  g0  = ib32;
		const uint8_t sb  = b->scales[ib32 / 2];
		const int	  sc0 = 1 + (2 * (sb & 0xf));
		const int	  sc1 = 1 + (2 * (sb >> 4));
		const int	  vi  = ib32 * 2;

		const __m128i c00 = decoded[vi + 0];
		const __m128i c01 = decoded[vi + 1];
		const __m128i c10 = decoded[vi + 2];
		const __m128i c11 = decoded[vi + 3];

		__m256i dec0 = _mm256_inserti128_si256(_mm256_castsi128_si256(c00), c01, 1);
		__m256i dec1 = _mm256_inserti128_si256(_mm256_castsi128_si256(c10), c11, 1);

		const int8_t *restrict qg0 = q8 + (g0 * 32);
		const int8_t *restrict qg1 = q8 + ((g0 + 1) * 32);

		int32_t a0 =
			vreduce_add_epi32(dotprod_s8_s8_i32(dec0, _mm256_loadu_si256((const __m256i *)(qg0))));
		int32_t a1 =
			vreduce_add_epi32(dotprod_s8_s8_i32(dec1, _mm256_loadu_si256((const __m256i *)(qg1))));

		total += sc0 * a0;
		total += sc1 * a1;
	}
	return total;
}

static void matmul_iq3_s_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
											float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(iq3s_block);
	const iq3s_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		__m128 acc0 = _mm_setzero_ps();
		__m128 acc1 = _mm_setzero_ps();

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q8_k_block *restrict yb = &xq[bi];
			const float		  d_xq		  = yb->d;

			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(iq3s_block)), 0, 1);
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

			__m128 sumi0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane)));
			__m128 sumi1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane + 4)));
			__m128 d_w0	 = loadu_f16x4_to_ps_128(d_w_raw);
			__m128 d_w1	 = loadu_f16x4_to_ps_128(d_w_raw + 4);
			__m128 d_xq_v = _mm_set1_ps(d_xq);

			acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_mul_ps(d_w0, d_xq_v), sumi0));
			acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_mul_ps(d_w1, d_xq_v), sumi1));
		}

		float tmp0[4], tmp1[4];
		_mm_storeu_ps(tmp0, acc0);
		_mm_storeu_ps(tmp1, acc1);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
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

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m128 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_k_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				__m128i decoded[MR][16];
				int32_t sc0[MR][4];
				int32_t sc1[MR][4];
				float	d_w[MR];

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

						const __m128i f00 = iq3s_flip8(qs[0], qs[1], qh0, signs[0], 8, 7);
						const __m128i f01 = iq3s_flip8(qs[2], qs[3], qh0, signs[1], 6, 5);
						const __m128i f02 = iq3s_flip8(qs[4], qs[5], qh0, signs[2], 4, 3);
						const __m128i f03 = iq3s_flip8(qs[6], qs[7], qh0, signs[3], 2, 1);
						const __m128i f10 = iq3s_flip8(qs[8], qs[9], qh1, signs[4], 8, 7);
						const __m128i f11 = iq3s_flip8(qs[10], qs[11], qh1, signs[5], 6, 5);
						const __m128i f12 = iq3s_flip8(qs[12], qs[13], qh1, signs[6], 4, 3);
						const __m128i f13 = iq3s_flip8(qs[14], qs[15], qh1, signs[7], 2, 1);

						decoded[r][vi + 0] = _mm_unpacklo_epi64(f00, f01);
						decoded[r][vi + 1] = _mm_unpacklo_epi64(f02, f03);
						decoded[r][vi + 2] = _mm_unpacklo_epi64(f10, f11);
						decoded[r][vi + 3] = _mm_unpacklo_epi64(f12, f13);

						qh += 2;
						qs += 16;
						signs += 8;
					}
				}

				float xd[NR];
				const int8_t *restrict q8p[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]  = xrow[c][bi].d;
					q8p[c] = xrow[c][bi].qs;
				}
				__m128 xd_vec = _mm_loadu_ps(xd);

				for (int r = 0; r < MR; r++) {
					int32_t total_arr[4] = {0, 0, 0, 0};

					for (int ib32 = 0; ib32 < 8; ib32 += 2) {
						const __m128i c00 = decoded[r][(ib32 * 2) + 0];
						const __m128i c01 = decoded[r][(ib32 * 2) + 1];
						const __m128i c10 = decoded[r][(ib32 * 2) + 2];
						const __m128i c11 = decoded[r][(ib32 * 2) + 3];

						__m256i dec0 =
							_mm256_inserti128_si256(_mm256_castsi128_si256(c00), c01, 1);
						__m256i dec1 =
							_mm256_inserti128_si256(_mm256_castsi128_si256(c10), c11, 1);

						for (int c = 0; c < 4; c++) {
							const int8_t *restrict qg0 = q8p[c] + (ib32 * 32);
							const int8_t *restrict qg1 = q8p[c] + ((ib32 + 1) * 32);
							int32_t a0 =
								vreduce_add_epi32(dotprod_s8_s8_i32(
									dec0, _mm256_loadu_si256((const __m256i *)(qg0))));
							int32_t a1 =
								vreduce_add_epi32(dotprod_s8_s8_i32(
									dec1, _mm256_loadu_si256((const __m256i *)(qg1))));
							total_arr[c] += sc0[r][ib32 / 2] * a0 + sc1[r][ib32 / 2] * a1;
						}
					}

					const __m128i total4 = _mm_loadu_si128((const __m128i *)total_arr);
					__m128 total4f = _mm_cvtepi32_ps(total4);
					__m128 dw		 = _mm_set1_ps(d_w[r]);
					acc_row[r] = _mm_add_ps(acc_row[r], _mm_mul_ps(_mm_mul_ps(dw, total4f), xd_vec));
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
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
}
#undef NR

void dequant_f16_row(const void *src, int n, float *dst) {
	const uint16_t *s = src;
	int				i = 0;
	for (; i + 8 <= n; i += 8) {
		__m256 v = loadu_f16x8_to_ps(s + i);
		_mm256_storeu_ps(dst + i, v);
	}
	for (; i < n; i++)
		dst[i] = f16_to_f32_fast(s[i]);
}

void dequant_bf16_row(const void *src, int n, float *dst) {
	const uint16_t *s = src;
	int				i = 0;
	for (; i + 8 <= n; i += 8) {
		__m128i v  = _mm_loadu_si128((const __m128i *)(s + i));
		__m256i lo = _mm256_cvtepu16_epi32(v);
		__m256i hi = _mm256_slli_epi32(lo, 16);
		_mm256_storeu_ps(dst + i, _mm256_castsi256_ps(hi));
	}
	for (; i < n; i++)
		dst[i] = bf16_to_f32(s[i]);
}

void matmul_generic_f32(const void *w, uint32_t w_type, const float *x, float *y, int n, int k) {
	switch (w_type) {
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_0_R8:
	case GGML_TYPE_Q4_0_R8:
	case GGML_TYPE_IQ4_NL_R8:
	case GGML_TYPE_Q4_K_R8:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_Q4_K:
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
		case GGML_TYPE_Q4_K_R8:
			matmul_q4_k_r8_q8_k_f32(w, x, y, n, k, &qs);
			break;
		case GGML_TYPE_Q5_K:
			matmul_q5_k_q8_k_f32(w, x, y, n, k, &qs);
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

			__m256 acc0 = _mm256_setzero_ps();
			__m256 acc1 = _mm256_setzero_ps();
			__m256 acc2 = _mm256_setzero_ps();
			__m256 acc3 = _mm256_setzero_ps();
			int		d	 = 0;

			for (; d + 32 <= k; d += 32) {
				__m256 x0 = _mm256_loadu_ps(x + d);
				__m256 x1 = _mm256_loadu_ps(x + d + 8);
				__m256 x2 = _mm256_loadu_ps(x + d + 16);
				__m256 x3 = _mm256_loadu_ps(x + d + 24);

				__m256 r0 = _mm256_loadu_ps(row_buf + d);
				__m256 r1 = _mm256_loadu_ps(row_buf + d + 8);
				__m256 r2 = _mm256_loadu_ps(row_buf + d + 16);
				__m256 r3 = _mm256_loadu_ps(row_buf + d + 24);

				acc0 = _mm256_fmadd_ps(x0, r0, acc0);
				acc1 = _mm256_fmadd_ps(x1, r1, acc1);
				acc2 = _mm256_fmadd_ps(x2, r2, acc2);
				acc3 = _mm256_fmadd_ps(x3, r3, acc3);
			}

			for (; d + 16 <= k; d += 16) {
				__m256 x0 = _mm256_loadu_ps(x + d);
				__m256 x1 = _mm256_loadu_ps(x + d + 8);

				acc0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(row_buf + d), acc0);
				acc1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(row_buf + d + 8), acc1);
			}

			for (; d + 8 <= k; d += 8) {
				acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(row_buf + d), acc0);
			}

			for (; d + 4 <= k; d += 4) {
				__m128 x0 = _mm_loadu_ps(x + d);
				__m128 r0 = _mm_loadu_ps(row_buf + d);
				__m128 a0 = _mm256_castps256_ps128(acc0);
				acc0	 = _mm256_insertf128_ps(acc0, _mm_add_ps(a0, _mm_mul_ps(x0, r0)), 0);
			}

			__m256 acc01 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
			float sum	  = vreduce_add_ps(acc01);

			for (; d < k; d++) {
				sum += x[d] * row_buf[d];
			}

			y[i] = sum;
		}
	} else if (k >= 16) {
		for (int i = 0; i < n; i++) {
			const uint8_t *row = (const uint8_t *)w + ((size_t)i * row_stride);
			dequant_row_dispatch(w_type, row, k, row_buf);

			__m128 acc0 = _mm_setzero_ps();
			__m128 acc1 = _mm_setzero_ps();
			int		d	 = 0;

			for (; d + 16 <= k; d += 16) {
				__m128 x0 = _mm_loadu_ps(x + d);
				__m128 x1 = _mm_loadu_ps(x + d + 4);
				__m128 x2 = _mm_loadu_ps(x + d + 8);
				__m128 x3 = _mm_loadu_ps(x + d + 12);

				acc0 = _mm_fmadd_ps(x0, _mm_loadu_ps(row_buf + d), acc0);
				acc0 = _mm_fmadd_ps(x1, _mm_loadu_ps(row_buf + d + 4), acc0);
				acc1 = _mm_fmadd_ps(x2, _mm_loadu_ps(row_buf + d + 8), acc1);
				acc1 = _mm_fmadd_ps(x3, _mm_loadu_ps(row_buf + d + 12), acc1);
			}

			for (; d + 4 <= k; d += 4) {
				acc0 = _mm_fmadd_ps(_mm_loadu_ps(x + d), _mm_loadu_ps(row_buf + d), acc0);
			}

			__m128 acc = _mm_add_ps(acc0, acc1);
			float	sum = vreduce_add_ps_128(acc);

			for (; d < k; d++) {
				sum += x[d] * row_buf[d];
			}

			y[i] = sum;
		}
	} else {
		for (int i = 0; i < n; i++) {
			const uint8_t *row = (const uint8_t *)w + ((size_t)i * row_stride);
			dequant_row_dispatch(w_type, row, k, row_buf);

			__m128 acc = _mm_setzero_ps();
			int		d	= 0;

			for (; d + 4 <= k; d += 4) {
				acc = _mm_fmadd_ps(_mm_loadu_ps(x + d), _mm_loadu_ps(row_buf + d), acc);
			}

			float sum = vreduce_add_ps_128(acc);

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
		__m256 v0 = _mm256_loadu_ps(x);
		__m256 v1 = _mm256_loadu_ps(x + 8);
		__m256 v2 = _mm256_loadu_ps(x + 16);
		__m256 v3 = _mm256_loadu_ps(x + 24);
		__m256 a0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v0);
		__m256 a1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v1);
		__m256 a2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v2);
		__m256 a3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v3);
		__m256 m	= _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
		float amax = vreduce_max_ps(m);

		float d	 = amax / 127.0f;
		float id = d > 0 ? 1.0f / d : 0.0f;
		dst[i].d = f32_to_f16(d);

		__m256 id_v = _mm256_set1_ps(id);
		__m256i q0	= _mm256_cvtps_epi32(
			_mm256_round_ps(_mm256_mul_ps(v0, id_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
		__m256i q1	= _mm256_cvtps_epi32(
			_mm256_round_ps(_mm256_mul_ps(v1, id_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
		__m256i q2	= _mm256_cvtps_epi32(
			_mm256_round_ps(_mm256_mul_ps(v2, id_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
		__m256i q3	= _mm256_cvtps_epi32(
			_mm256_round_ps(_mm256_mul_ps(v3, id_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

		__m128i n0 = _mm_packs_epi32(_mm256_castsi256_si128(q0), _mm256_extracti128_si256(q0, 1));
		__m128i n1 = _mm_packs_epi32(_mm256_castsi256_si128(q1), _mm256_extracti128_si256(q1, 1));
		__m128i n2 = _mm_packs_epi32(_mm256_castsi256_si128(q2), _mm256_extracti128_si256(q2, 1));
		__m128i n3 = _mm_packs_epi32(_mm256_castsi256_si128(q3), _mm256_extracti128_si256(q3, 1));
		__m128i b0 = _mm_packs_epi16(n0, n1);
		__m128i b1 = _mm_packs_epi16(n2, n3);
		_mm_storeu_si128((__m128i *)(dst[i].qs), b0);
		_mm_storeu_si128((__m128i *)(dst[i].qs + 16), b1);

		x += 32;
	}
}

void quantize_q8_1(const float *x, void *dst, int n) {
	int			nb = n / 32;
	q8_1_block *y  = dst;
	for (int i = 0; i < nb; i++) {
		__m256 v0 = _mm256_loadu_ps(x);
		__m256 v1 = _mm256_loadu_ps(x + 8);
		__m256 v2 = _mm256_loadu_ps(x + 16);
		__m256 v3 = _mm256_loadu_ps(x + 24);
		__m256 a0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v0);
		__m256 a1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v1);
		__m256 a2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v2);
		__m256 a3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v3);
		__m256 m	= _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
		float amax = vreduce_max_ps(m);

		float d	 = amax / 127.0f;
		float id = d > 0 ? 1.0f / d : 0.0f;
		y[i].d	 = f32_to_f16(d);

		__m256 id_v = _mm256_set1_ps(id);
		__m256i q0	= _mm256_cvtps_epi32(
			_mm256_round_ps(_mm256_mul_ps(v0, id_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
		__m256i q1	= _mm256_cvtps_epi32(
			_mm256_round_ps(_mm256_mul_ps(v1, id_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
		__m256i q2	= _mm256_cvtps_epi32(
			_mm256_round_ps(_mm256_mul_ps(v2, id_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
		__m256i q3	= _mm256_cvtps_epi32(
			_mm256_round_ps(_mm256_mul_ps(v3, id_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

		__m256i s01 = _mm256_add_epi32(q0, q1);
		__m256i s23 = _mm256_add_epi32(q2, q3);
		__m256i s_all = _mm256_add_epi32(s01, s23);
		int32_t sum	= vreduce_add_epi32(s_all);

		__m128i n0 = _mm_packs_epi32(_mm256_castsi256_si128(q0), _mm256_extracti128_si256(q0, 1));
		__m128i n1 = _mm_packs_epi32(_mm256_castsi256_si128(q1), _mm256_extracti128_si256(q1, 1));
		__m128i n2 = _mm_packs_epi32(_mm256_castsi256_si128(q2), _mm256_extracti128_si256(q2, 1));
		__m128i n3 = _mm_packs_epi32(_mm256_castsi256_si128(q3), _mm256_extracti128_si256(q3, 1));
		__m128i b0 = _mm_packs_epi16(n0, n1);
		__m128i b1 = _mm_packs_epi16(n2, n3);
		_mm_storeu_si128((__m128i *)(y[i].qs), b0);
		_mm_storeu_si128((__m128i *)(y[i].qs + 16), b1);

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
		__m256		amax_v0 = _mm256_setzero_ps();
		__m256		max_v0	= _mm256_setzero_ps();
		for (int j = 0; j < 256; j += 8) {
			__m256 v0 = _mm256_loadu_ps(x + j);
			__m256 a0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v0);
			__m256 m0 = _mm256_cmp_ps(a0, amax_v0, _CMP_GT_OQ);
			amax_v0	  = _mm256_max_ps(amax_v0, a0);
			max_v0	  = _mm256_blendv_ps(max_v0, v0, m0);
		}
		float		amax_lanes[8] __attribute__((aligned(32)));
		float		max_lanes[8] __attribute__((aligned(32)));
		_mm256_store_ps(amax_lanes, amax_v0);
		_mm256_store_ps(max_lanes, max_v0);
		float amax = amax_lanes[0];
		max		   = max_lanes[0];
		for (int lane = 1; lane < 8; lane++) {
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
		__m256		 is_v	= _mm256_set1_ps(iscale);
		int8_t		*qsp	= y[i].qs;
		const float *xp		= x;
		for (int outer = 0; outer < 4; outer++) {
			for (int sub = 0; sub < 4; sub++) {
				__m256 v0 = _mm256_loadu_ps(xp);
				__m256 v1 = _mm256_loadu_ps(xp + 8);
				__m256i q0	= _mm256_cvtps_epi32(_mm256_round_ps(
					_mm256_mul_ps(v0, is_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
				__m256i q1	= _mm256_cvtps_epi32(_mm256_round_ps(
					_mm256_mul_ps(v1, is_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
				__m256i lo	= _mm256_set1_epi32(-127);
				__m256i hi	= _mm256_set1_epi32(127);
				q0			= _mm256_max_epi32(_mm256_min_epi32(q0, hi), lo);
				q1			= _mm256_max_epi32(_mm256_min_epi32(q1, hi), lo);
				__m256i s_all = _mm256_add_epi32(q0, q1);
				int32_t sum	= vreduce_add_epi32(s_all);
				__m128i n0 = _mm_packs_epi32(_mm256_castsi256_si128(q0), _mm256_extracti128_si256(q0, 1));
				__m128i n1 = _mm_packs_epi32(_mm256_castsi256_si128(q1), _mm256_extracti128_si256(q1, 1));
				__m128i b0 = _mm_packs_epi16(n0, n1);
				_mm_storeu_si128((__m128i *)(qsp), b0);
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
		__m128 acc0 = _mm_setzero_ps();
		__m128 acc1 = _mm_setzero_ps();

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32_fast(xq[bi].d);

			const __m256i xq32 = _mm256_loadu_si256((const __m256i *)(xq8));

			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_0_block)), 0, 1);
			}

			int32_t sumi_lane[8];
			for (int r = 0; r < 8; r++) {
				const q4_0_block *row =
					(const q4_0_block *)(row_base[r] + (size_t)bi * sizeof(q4_0_block));
				const __m128i q	 = _mm_loadu_si128((const __m128i *)(row->qs));
				const __m128i lo_u = _mm_and_si128(q, _mm_set1_epi8(0x0F));
				const __m128i hi_u =
					_mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0F));
				const __m128i lo = _mm_sub_epi8(lo_u, _mm_set1_epi8(8));
				const __m128i hi = _mm_sub_epi8(hi_u, _mm_set1_epi8(8));
				__m256i		  lo32 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);

				__m256i acc	 = dotprod_s8_s8_i32(lo32, xq32);
				sumi_lane[r] = vreduce_add_epi32(acc);
			}
			uint16_t d_w_raw[8];
			for (int r = 0; r < 8; r++)
				d_w_raw[r] =
					((const q4_0_block *)(row_base[r] + (size_t)bi * sizeof(q4_0_block)))->d;
			__m128 d_w0	 = loadu_f16x4_to_ps_128(d_w_raw);
			__m128 d_w1	 = loadu_f16x4_to_ps_128(d_w_raw + 4);
			__m128 d_xq_v = _mm_set1_ps(d_xq);

			__m128 sumi0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane)));
			__m128 sumi1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane + 4)));
			acc0		 = _mm_add_ps(acc0, _mm_mul_ps(_mm_mul_ps(d_w0, d_xq_v), sumi0));
			acc1		 = _mm_add_ps(acc1, _mm_mul_ps(_mm_mul_ps(d_w1, d_xq_v), sumi1));
		}

		float tmp0[4], tmp1[4];
		_mm_storeu_ps(tmp0, acc0);
		_mm_storeu_ps(tmp1, acc1);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
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

#define NR 8
#define RT 32
void matmul_q4_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							size_t xq_row_stride_blocks, float *restrict y, int y_row_stride, int n,
							int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q4_0_block);
	const q4_0_block *Wb			 = w;

	int t = 0;
	for (; t + NR <= m; t += NR) {
		const q8_0_block *xrow[NR];
		for (int c = 0; c < NR; c++)
			xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

		int16_t _Alignas(64) xq16[NR][blocks_per_row][32];
		float				d_xq[NR][blocks_per_row];
		for (int c = 0; c < NR; c++) {
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const int8_t *restrict qs = xrow[c][bi].qs;
				for (int j = 0; j < 32; j++)
					xq16[c][bi][j] = qs[j];
				d_xq[c][bi]			   = f16_to_f32_fast(xrow[c][bi].d);
			}
		}

		int i = 0;
		for (; i < n; i += RT) {
			const int			 rmax = RT < n - i ? RT : n - i;
			const q4_0_block *rowbase =
				(const q4_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));

			for (int r = 0; r < rmax; r++) {
				const q4_0_block *row = rowbase + ((size_t)r * blocks_per_row);
				__m256			  acc_f[NR];
				for (int c = 0; c < NR; c++)
					acc_f[c] = _mm256_setzero_ps();

				for (int bi = 0; bi < blocks_per_row; bi++) {
					if (bi + 1 < blocks_per_row)
						__builtin_prefetch(&row[bi + 1], 0, 1);

					const q4_0_block *b		= &row[bi];
					const float		   d_w  = f16_to_f32_fast(b->d);
					const __m128i	   q	= _mm_loadu_si128((const __m128i *)(b->qs));
					const __m128i	   lo_u = _mm_and_si128(q, _mm_set1_epi8(0x0F));
					const __m128i	   hi_u = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0F));
					const __m128i	   lo	= _mm_sub_epi8(lo_u, _mm_set1_epi8(8));
					const __m128i	   hi	= _mm_sub_epi8(hi_u, _mm_set1_epi8(8));
					const __m256i q32_lo = _mm256_cvtepi8_epi16(lo);
					const __m256i q32_hi = _mm256_cvtepi8_epi16(hi);

					for (int c = 0; c < NR; c++) {
						const __m256i x_lo =
							_mm256_loadu_si256((const __m256i *)(&xq16[c][bi][0]));
						const __m256i x_hi =
							_mm256_loadu_si256((const __m256i *)(&xq16[c][bi][16]));
						__m256i dot = _mm256_madd_epi16(q32_lo, x_lo);
						dot			= _mm256_add_epi32(dot, _mm256_madd_epi16(q32_hi, x_hi));
						acc_f[c]	= _mm256_fmadd_ps(
							_mm256_cvtepi32_ps(dot), _mm256_set1_ps(d_w * d_xq[c][bi]), acc_f[c]);
					}
				}

				for (int c = 0; c < NR; c++) {
					const __m128 lo128 = _mm256_castps256_ps128(acc_f[c]);
					const __m128 hi128 = _mm256_extractf128_ps(acc_f[c], 1);
					y[((size_t)(t + c) * y_row_stride) + (i + r)] =
						vreduce_add_ps_128(_mm_add_ps(lo128, hi128));
				}
			}
		}
	}

	for (; t < m; t++) {
		const q8_0_block *restrict xrow = xq + ((size_t)t * xq_row_stride_blocks);
		float			  *restrict yrow = y + ((size_t)t * y_row_stride);
		for (int i = 0; i < n; i += MR) {
			matmul_q4_q8_qonly_f32_row((const uint8_t *)Wb + ((size_t)i * row_stride), xrow,
									   yrow + i, n - i < MR ? n - i : MR, k);
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
		__m128 acc_lo = _mm_setzero_ps();
		__m128 acc_hi = _mm_setzero_ps();

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
				__builtin_prefetch(&r7[bi + 2], 0, 1);
			}

			const float		d_xq0 = f16_to_f32_fast(xq[bi].d);
			const float		d_xq1 = f16_to_f32_fast(xq[bi + 1].d);
			const __m256i xq0	= _mm256_loadu_si256((const __m256i *)(xq[bi].qs));
			const __m256i xq1	= _mm256_loadu_si256((const __m256i *)(xq[bi + 1].qs));

			int32_t dots_a[8], dots_b[8];
			const q8_0_block *rows_a[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
			for (int r = 0; r < 8; r++) {
				dots_a[r] = vreduce_add_epi32(dotprod_s8_s8_i32(
					_mm256_loadu_si256((const __m256i *)(rows_a[r][bi].qs)), xq0));
				dots_b[r] = vreduce_add_epi32(dotprod_s8_s8_i32(
					_mm256_loadu_si256((const __m256i *)(rows_a[r][bi + 1].qs)), xq1));
			}

			__m128i sumi_loa = _mm_setr_epi32(dots_a[0], dots_a[1], dots_a[2], dots_a[3]);
			__m128i sumi_hia = _mm_setr_epi32(dots_a[4], dots_a[5], dots_a[6], dots_a[7]);
			__m128i sumi_lob = _mm_setr_epi32(dots_b[0], dots_b[1], dots_b[2], dots_b[3]);
			__m128i sumi_hib = _mm_setr_epi32(dots_b[4], dots_b[5], dots_b[6], dots_b[7]);

			uint16_t d_loa[4] = {r0[bi].d, r1[bi].d, r2[bi].d, r3[bi].d};
			uint16_t d_hia[4] = {r4[bi].d, r5[bi].d, r6[bi].d, r7[bi].d};
			uint16_t d_lob[4] = {r0[bi + 1].d, r1[bi + 1].d, r2[bi + 1].d, r3[bi + 1].d};
			uint16_t d_hib[4] = {r4[bi + 1].d, r5[bi + 1].d, r6[bi + 1].d, r7[bi + 1].d};

			__m128 d_f32_loa = loadu_f16x4_to_ps_128(d_loa);
			__m128 d_f32_hia = loadu_f16x4_to_ps_128(d_hia);
			__m128 d_f32_lob = loadu_f16x4_to_ps_128(d_lob);
			__m128 d_f32_hib = loadu_f16x4_to_ps_128(d_hib);

			__m128 d_xq_v0 = _mm_set1_ps(d_xq0);
			__m128 d_xq_v1 = _mm_set1_ps(d_xq1);
			__m128 dw_loa	= _mm_mul_ps(d_f32_loa, d_xq_v0);
			__m128 dw_hia	= _mm_mul_ps(d_f32_hia, d_xq_v0);
			__m128 dw_lob	= _mm_mul_ps(d_f32_lob, d_xq_v1);
			__m128 dw_hib	= _mm_mul_ps(d_f32_hib, d_xq_v1);

			acc_lo = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_loa), dw_loa, acc_lo);
			acc_hi = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_hia), dw_hia, acc_hi);
			acc_lo = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_lob), dw_lob, acc_lo);
			acc_hi = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_hib), dw_hib, acc_hi);
		}

		for (; bi < blocks_per_row; bi++) {
			const float		d_xq = f16_to_f32_fast(xq[bi].d);
			const __m256i	xq8	 = _mm256_loadu_si256((const __m256i *)(xq[bi].qs));

			if (bi + 1 < blocks_per_row) {
				__builtin_prefetch(&r0[bi + 1], 0, 1);
				__builtin_prefetch(&r7[bi + 1], 0, 1);
			}

			int32_t dots[8];
			const q8_0_block *rows[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
			for (int r = 0; r < 8; r++)
				dots[r] = vreduce_add_epi32(dotprod_s8_s8_i32(
					_mm256_loadu_si256((const __m256i *)(rows[r][bi].qs)), xq8));

			__m128i sumi_lo = _mm_setr_epi32(dots[0], dots[1], dots[2], dots[3]);
			__m128i sumi_hi = _mm_setr_epi32(dots[4], dots[5], dots[6], dots[7]);

			uint16_t d_lo[4] = {r0[bi].d, r1[bi].d, r2[bi].d, r3[bi].d};
			uint16_t d_hi[4] = {r4[bi].d, r5[bi].d, r6[bi].d, r7[bi].d};

			__m128 d_f32_lo = loadu_f16x4_to_ps_128(d_lo);
			__m128 d_f32_hi = loadu_f16x4_to_ps_128(d_hi);

			__m128 d_xq_v = _mm_set1_ps(d_xq);
			__m128 dw_lo	= _mm_mul_ps(d_f32_lo, d_xq_v);
			__m128 dw_hi	= _mm_mul_ps(d_f32_hi, d_xq_v);

			acc_lo = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_lo), dw_lo, acc_lo);
			acc_hi = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_hi), dw_hi, acc_hi);
		}

		float lo[4], hi[4];
		_mm_storeu_ps(lo, acc_lo);
		_mm_storeu_ps(hi, acc_hi);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = lo[r];
			y[i + 4 + r] = hi[r];
		}
	}

	for (; i < n; i++) {
		const q8_0_block *row =
			(const q8_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const __m256i qs	= _mm256_loadu_si256((const __m256i *)(row[bi].qs));
			const __m256i xq8	= _mm256_loadu_si256((const __m256i *)(xq[bi].qs));
			int32_t		sumi	= vreduce_add_epi32(dotprod_s8_s8_i32(qs, xq8));
			const float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xq[bi].d);
			sumf = fmaf(d, (float)sumi, sumf);
		}
		y[i] = sumf;
	}
}

#define NR 8
#define RT 32
void matmul_q8_0_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q8_0_block);
	const q8_0_block *Wb			 = w;

	int t = 0;
	for (; t + NR <= m; t += NR) {
		const q8_0_block *xrow[NR];
		for (int c = 0; c < NR; c++)
			xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

		int16_t _Alignas(64) xq16[NR][blocks_per_row][32];
		float				d_xq[NR][blocks_per_row];
		for (int c = 0; c < NR; c++) {
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const int8_t *restrict qs = xrow[c][bi].qs;
				for (int j = 0; j < 32; j++)
					xq16[c][bi][j] = qs[j];
				d_xq[c][bi]			   = f16_to_f32_fast(xrow[c][bi].d);
			}
		}

		int i = 0;
		for (; i < n; i += RT) {
			const int			 rmax = RT < n - i ? RT : n - i;
			const q8_0_block *rowbase =
				(const q8_0_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));

			for (int r = 0; r < rmax; r++) {
				const q8_0_block *row = rowbase + ((size_t)r * blocks_per_row);
				__m256			  acc_f[NR];
				for (int c = 0; c < NR; c++)
					acc_f[c] = _mm256_setzero_ps();

				for (int bi = 0; bi < blocks_per_row; bi++) {
					if (bi + 1 < blocks_per_row)
						__builtin_prefetch(&row[bi + 1], 0, 1);

					const q8_0_block *b		= &row[bi];
					const float		   d_w  = f16_to_f32_fast(b->d);
					const __m256i	   q8	= _mm256_loadu_si256((const __m256i *)(b->qs));
					const __m256i q32_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8));
					const __m256i q32_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8, 1));

					for (int c = 0; c < NR; c++) {
						const __m256i x_lo =
							_mm256_loadu_si256((const __m256i *)(&xq16[c][bi][0]));
						const __m256i x_hi =
							_mm256_loadu_si256((const __m256i *)(&xq16[c][bi][16]));
						__m256i dot = _mm256_madd_epi16(q32_lo, x_lo);
						dot			= _mm256_add_epi32(dot, _mm256_madd_epi16(q32_hi, x_hi));
						acc_f[c]	= _mm256_fmadd_ps(
							_mm256_cvtepi32_ps(dot), _mm256_set1_ps(d_w * d_xq[c][bi]), acc_f[c]);
					}
				}

				for (int c = 0; c < NR; c++) {
					const __m128 lo128 = _mm256_castps256_ps128(acc_f[c]);
					const __m128 hi128 = _mm256_extractf128_ps(acc_f[c], 1);
					y[((size_t)(t + c) * y_row_stride) + (i + r)] =
						vreduce_add_ps_128(_mm_add_ps(lo128, hi128));
				}
			}
		}
	}

	for (; t < m; t++) {
		const q8_0_block *restrict xrow = xq + ((size_t)t * xq_row_stride_blocks);
		float			  *restrict yrow = y + ((size_t)t * y_row_stride);
		for (int i = 0; i < n; i += MR) {
			matmul_q8_0_q8_qonly_f32_row((const uint8_t *)Wb + ((size_t)i * row_stride), xrow,
										 yrow + i, n - i < MR ? n - i : MR, k);
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
		__m128 acc_lo = _mm_setzero_ps();
		__m128 acc_hi = _mm_setzero_ps();

		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int bi = 0;
		for (; bi + 1 < blocks_per_row; bi += 2) {
			const uint8_t *blk0 = group + (size_t)bi * Q8_0_R8_GROUP_BYTES;
			const uint8_t *blk1 = group + (size_t)(bi + 1) * Q8_0_R8_GROUP_BYTES;

			if (bi + 2 < blocks_per_row)
				__builtin_prefetch(group + (size_t)(bi + 2) * Q8_0_R8_GROUP_BYTES, 0, 1);

			const float		d_xq0 = f16_to_f32_fast(xq[bi].d);
			const float		d_xq1 = f16_to_f32_fast(xq[bi + 1].d);
			const __m256i xq0	= _mm256_loadu_si256((const __m256i *)(xq[bi].qs));
			const __m256i xq1	= _mm256_loadu_si256((const __m256i *)(xq[bi + 1].qs));

			const uint16_t *d_ptr0 = (const uint16_t *)blk0;
			const uint16_t *d_ptr1 = (const uint16_t *)blk1;
			const int8_t   *qs0	 = (const int8_t *)(blk0 + Q8_0_R8_ROWS * sizeof(uint16_t));
			const int8_t   *qs1	 = (const int8_t *)(blk1 + Q8_0_R8_ROWS * sizeof(uint16_t));

			int32_t dots_a[8], dots_b[8];
			for (int r = 0; r < 8; r++) {
				dots_a[r] = vreduce_add_epi32(dotprod_s8_s8_i32(
					_mm256_loadu_si256((const __m256i *)(qs0 + (size_t)r * 32)), xq0));
				dots_b[r] = vreduce_add_epi32(dotprod_s8_s8_i32(
					_mm256_loadu_si256((const __m256i *)(qs1 + (size_t)r * 32)), xq1));
			}

			__m128i sumi_loa = _mm_setr_epi32(dots_a[0], dots_a[1], dots_a[2], dots_a[3]);
			__m128i sumi_hia = _mm_setr_epi32(dots_a[4], dots_a[5], dots_a[6], dots_a[7]);
			__m128i sumi_lob = _mm_setr_epi32(dots_b[0], dots_b[1], dots_b[2], dots_b[3]);
			__m128i sumi_hib = _mm_setr_epi32(dots_b[4], dots_b[5], dots_b[6], dots_b[7]);

			__m128 d_f32_lo0 = loadu_f16x4_to_ps_128(d_ptr0);
			__m128 d_f32_hi0 = loadu_f16x4_to_ps_128(d_ptr0 + 4);
			__m128 d_f32_lo1 = loadu_f16x4_to_ps_128(d_ptr1);
			__m128 d_f32_hi1 = loadu_f16x4_to_ps_128(d_ptr1 + 4);

			__m128 d_xq_v0 = _mm_set1_ps(d_xq0);
			__m128 d_xq_v1 = _mm_set1_ps(d_xq1);
			__m128 dw_lo0	= _mm_mul_ps(d_f32_lo0, d_xq_v0);
			__m128 dw_hi0	= _mm_mul_ps(d_f32_hi0, d_xq_v0);
			__m128 dw_lo1	= _mm_mul_ps(d_f32_lo1, d_xq_v1);
			__m128 dw_hi1	= _mm_mul_ps(d_f32_hi1, d_xq_v1);

			acc_lo = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_loa), dw_lo0, acc_lo);
			acc_hi = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_hia), dw_hi0, acc_hi);
			acc_lo = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_lob), dw_lo1, acc_lo);
			acc_hi = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_hib), dw_hi1, acc_hi);
		}

		for (; bi < blocks_per_row; bi++) {
			const uint8_t *blk = group + (size_t)bi * Q8_0_R8_GROUP_BYTES;

			const float		d_xq = f16_to_f32_fast(xq[bi].d);
			const __m256i xq8	= _mm256_loadu_si256((const __m256i *)(xq[bi].qs));

			const uint16_t *d_ptr  = (const uint16_t *)blk;
			const int8_t   *qs_ptr = (const int8_t *)(blk + Q8_0_R8_ROWS * sizeof(uint16_t));

			int32_t dots[8];
			for (int r = 0; r < 8; r++)
				dots[r] = vreduce_add_epi32(dotprod_s8_s8_i32(
					_mm256_loadu_si256((const __m256i *)(qs_ptr + (size_t)r * 32)), xq8));

			__m128i sumi_lo = _mm_setr_epi32(dots[0], dots[1], dots[2], dots[3]);
			__m128i sumi_hi = _mm_setr_epi32(dots[4], dots[5], dots[6], dots[7]);

			__m128 d_f32_lo = loadu_f16x4_to_ps_128(d_ptr);
			__m128 d_f32_hi = loadu_f16x4_to_ps_128(d_ptr + 4);

			__m128 d_xq_v = _mm_set1_ps(d_xq);
			__m128 dw_lo	= _mm_mul_ps(d_f32_lo, d_xq_v);
			__m128 dw_hi	= _mm_mul_ps(d_f32_hi, d_xq_v);

			acc_lo = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_lo), dw_lo, acc_lo);
			acc_hi = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_hi), dw_hi, acc_hi);
		}

		float lo[4], hi[4];
		_mm_storeu_ps(lo, acc_lo);
		_mm_storeu_ps(hi, acc_hi);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = lo[r];
			y[i + 4 + r] = hi[r];
		}
	}
}

#define NR 4
void matmul_q8_0_r8_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m) {
	const int	   blocks_per_row = k / 32;
	const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(q8_0_block);
	const uint8_t *Wb			  = w;
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m128 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * Q8_0_R8_GROUP_BYTES;

				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + Q8_0_R8_GROUP_BYTES, 0, 1);

				float	xd[NR];
				__m256i xq32[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq32[c]	 = _mm256_loadu_si256((const __m256i *)(xrow[c][bi].qs));
				}
				const __m128 xd_vec = _mm_loadu_ps(xd);

				const uint16_t *d_ptr  = (const uint16_t *)blk;
				const int8_t   *qs_ptr = (const int8_t *)(blk + Q8_0_R8_ROWS * sizeof(uint16_t));

				for (int r = 0; r < MR; r++) {
					const float	d_w = f16_to_f32_fast(d_ptr[r]);
					const __m256i qs8 =
						_mm256_loadu_si256((const __m256i *)(qs_ptr + (size_t)r * 32));

					int32_t sumi_arr[NR];
					for (int c = 0; c < NR; c++)
						sumi_arr[c] = vreduce_add_epi32(dotprod_s8_s8_i32(qs8, xq32[c]));
					const __m128i sumi4 = _mm_loadu_si128((const __m128i *)sumi_arr);

					const __m128 sumi_f = _mm_cvtepi32_ps(sumi4);
					const __m128 scaled = _mm_mul_ps(_mm_mul_ps(xd_vec, sumi_f), _mm_set1_ps(d_w));

					acc_row[r] = _mm_add_ps(acc_row[r], scaled);
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
		}

		for (; t < m; t++) {
			matmul_q8_0_r8_q8_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
											y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}
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
		__m128 acc_lo = _mm_setzero_ps();
		__m128 acc_hi = _mm_setzero_ps();

		const uint8_t *group = Wb + ((size_t)i * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t *blk = group + (size_t)bi * Q4_0_R8_GROUP_BYTES;

			if (bi + 1 < blocks_per_row)
				__builtin_prefetch(blk + Q4_0_R8_GROUP_BYTES, 0, 1);

			const float		d_xq = f16_to_f32_fast(xq[bi].d);
			const __m256i xq8	= _mm256_loadu_si256((const __m256i *)(xq[bi].qs));

			const uint16_t *d_ptr  = (const uint16_t *)blk;
			const uint8_t  *qs_ptr = blk + Q4_0_R8_ROWS * sizeof(uint16_t);

			__m256i dots[8];
			for (int r = 0; r < 8; r++) {
				const __m128i q	 = _mm_loadu_si128((const __m128i *)(qs_ptr + (size_t)r * 16));
				const __m128i lo_u = _mm_and_si128(q, _mm_set1_epi8(0x0F));
				const __m128i hi_u =
					_mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0F));
				const __m128i lo = _mm_sub_epi8(lo_u, _mm_set1_epi8(8));
				const __m128i hi = _mm_sub_epi8(hi_u, _mm_set1_epi8(8));
				__m256i		  q32 =
					_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
				dots[r] = dotprod_s8_s8_i32(q32, xq8);
			}

			__m128i sumi_lo = vreduce4_add_epi32(dots[0], dots[1], dots[2], dots[3]);
			__m128i sumi_hi = vreduce4_add_epi32(dots[4], dots[5], dots[6], dots[7]);

			__m128 d_w0	 = loadu_f16x4_to_ps_128(d_ptr);
			__m128 d_w1	 = loadu_f16x4_to_ps_128(d_ptr + 4);
			__m128 d_xq_v = _mm_set1_ps(d_xq);

			acc_lo = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_lo), _mm_mul_ps(d_w0, d_xq_v), acc_lo);
			acc_hi = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_hi), _mm_mul_ps(d_w1, d_xq_v), acc_hi);
		}

		float tmp0[4], tmp1[4];
		_mm_storeu_ps(tmp0, acc_lo);
		_mm_storeu_ps(tmp1, acc_hi);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
	}
}

#define NR 4
void matmul_q4_0_r8_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								 size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								 int n, int k, int m) {
	const int	   blocks_per_row = k / 32;
	const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(q4_0_block);
	const uint8_t *Wb			  = w;
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m128 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * Q4_0_R8_GROUP_BYTES;

				if (bi + 1 < blocks_per_row)
					__builtin_prefetch(blk + Q4_0_R8_GROUP_BYTES, 0, 1);

				float	xd[NR];
				__m256i xq32[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq32[c]	 = _mm256_loadu_si256((const __m256i *)(xrow[c][bi].qs));
				}
				const __m128 xd_vec = _mm_loadu_ps(xd);

				const uint16_t *d_ptr  = (const uint16_t *)blk;
				const uint8_t  *qs_ptr = blk + Q4_0_R8_ROWS * sizeof(uint16_t);

				for (int r = 0; r < MR; r++) {
					const float		d_w = f16_to_f32_fast(d_ptr[r]);
					const __m128i	q	 = _mm_loadu_si128((const __m128i *)(qs_ptr + (size_t)r * 16));
					const __m128i	lo_u = _mm_and_si128(q, _mm_set1_epi8(0x0F));
					const __m128i	hi_u = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0F));
					const __m128i	lo	 = _mm_sub_epi8(lo_u, _mm_set1_epi8(8));
					const __m128i	hi	 = _mm_sub_epi8(hi_u, _mm_set1_epi8(8));
					const __m256i	q32 =
						_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);

					int32_t sumi_arr[NR];
					for (int c = 0; c < NR; c++)
						sumi_arr[c] = vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq32[c]));
					const __m128i sumi4 = _mm_loadu_si128((const __m128i *)sumi_arr);

					const __m128 sumi_f = _mm_cvtepi32_ps(sumi4);
					const __m128 scaled = _mm_mul_ps(_mm_mul_ps(xd_vec, sumi_f), _mm_set1_ps(d_w));

					acc_row[r] = _mm_add_ps(acc_row[r], scaled);
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
		}

		for (; t < m; t++) {
			matmul_q4_0_r8_q8_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
											y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}
}
#undef NR

#define IQ4_NL_R8_GROUP_BYTES (IQ4_NL_R8_ROWS * sizeof(uint16_t) + IQ4_NL_R8_ROWS * 16)

static void matmul_iq4_nl_r8_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
											  float *restrict y, int n, int k) {
	const int		blocks_per_row = k / 32;
	const size_t	row_stride	   = (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const uint8_t  *Wb			   = w;
	const __m128i	kvalues_u	   = _mm_loadu_si128((const __m128i *)(kvalues_iq4nl));
	const __m128i	lo_mask		   = _mm_set1_epi8(0x0F);
	int				i			   = 0;

	for (; i + MR <= n; i += MR) {
		__m128 acc_lo = _mm_setzero_ps();
		__m128 acc_hi = _mm_setzero_ps();

		const uint8_t *group = Wb + ((size_t)i * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t *blk = group + (size_t)bi * IQ4_NL_R8_GROUP_BYTES;

			if (bi + 8 < blocks_per_row)
				__builtin_prefetch(blk + 8 * IQ4_NL_R8_GROUP_BYTES, 0, 1);

			const float	  d_xq = f16_to_f32_fast(xq[bi].d);
			const __m256i xq8  = _mm256_loadu_si256((const __m256i *)(xq[bi].qs));

			const uint16_t *d_ptr  = (const uint16_t *)blk;
			const uint8_t  *qs_ptr = blk + IQ4_NL_R8_ROWS * sizeof(uint16_t);

			__m256i dots[8];
			for (int r = 0; r < 8; r++) {
				const __m128i q		 = _mm_loadu_si128((const __m128i *)(qs_ptr + (size_t)r * 16));
				const __m128i lo_idx = _mm_and_si128(q, lo_mask);
				const __m128i hi_idx = _mm_and_si128(_mm_srli_epi16(q, 4), lo_mask);
				const __m128i lo	 = _mm_shuffle_epi8(kvalues_u, lo_idx);
				const __m128i hi	 = _mm_shuffle_epi8(kvalues_u, hi_idx);
				const __m256i q32 =
					_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
				dots[r] = dotprod_s8_s8_i32(q32, xq8);
			}

			__m128i sumi_lo = vreduce4_add_epi32(dots[0], dots[1], dots[2], dots[3]);
			__m128i sumi_hi = vreduce4_add_epi32(dots[4], dots[5], dots[6], dots[7]);

			__m128 d_w0	 = loadu_f16x4_to_ps_128(d_ptr);
			__m128 d_w1	 = loadu_f16x4_to_ps_128(d_ptr + 4);
			__m128 d_xq_v = _mm_set1_ps(d_xq);

			acc_lo = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_lo), _mm_mul_ps(d_w0, d_xq_v), acc_lo);
			acc_hi = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_hi), _mm_mul_ps(d_w1, d_xq_v), acc_hi);
		}

		float tmp0[4], tmp1[4];
		_mm_storeu_ps(tmp0, acc_lo);
		_mm_storeu_ps(tmp1, acc_hi);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
	}
}

#define NR 4
void matmul_iq4_nl_r8_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								   size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								   int n, int k, int m) {
	const int	   blocks_per_row = k / 32;
	const size_t   row_stride	  = (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const uint8_t *Wb			  = w;
	const __m128i  kvalues_u	  = _mm_loadu_si128((const __m128i *)(kvalues_iq4nl));
	const __m128i  lo_mask		  = _mm_set1_epi8(0x0F);
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *group = Wb + ((size_t)i * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m128 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				const uint8_t *blk = group + (size_t)bi * IQ4_NL_R8_GROUP_BYTES;

				if (bi + 8 < blocks_per_row)
					__builtin_prefetch(blk + 8 * IQ4_NL_R8_GROUP_BYTES, 0, 1);

				float	xd[NR];
				__m256i xq32[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	= f16_to_f32_fast(xrow[c][bi].d);
					xq32[c] = _mm256_loadu_si256((const __m256i *)(xrow[c][bi].qs));
				}
				const __m128 xd_vec = _mm_loadu_ps(xd);

				const uint16_t *d_ptr  = (const uint16_t *)blk;
				const uint8_t  *qs_ptr = blk + IQ4_NL_R8_ROWS * sizeof(uint16_t);

				for (int r = 0; r < MR; r++) {
					const float	  d_w	 = f16_to_f32_fast(d_ptr[r]);
					const __m128i q		 = _mm_loadu_si128((const __m128i *)(qs_ptr + (size_t)r * 16));
					const __m128i lo_idx = _mm_and_si128(q, lo_mask);
					const __m128i hi_idx = _mm_and_si128(_mm_srli_epi16(q, 4), lo_mask);
					const __m128i lo	 = _mm_shuffle_epi8(kvalues_u, lo_idx);
					const __m128i hi	 = _mm_shuffle_epi8(kvalues_u, hi_idx);
					const __m256i q32 =
						_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);

					int32_t sumi_arr[NR];
					for (int c = 0; c < NR; c++)
						sumi_arr[c] = vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq32[c]));
					const __m128i sumi4 = _mm_loadu_si128((const __m128i *)sumi_arr);

					const __m128 sumi_f = _mm_cvtepi32_ps(sumi4);
					acc_row[r] =
						_mm_fmadd_ps(sumi_f, _mm_mul_ps(xd_vec, _mm_set1_ps(d_w)), acc_row[r]);
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
		}

		for (; t < m; t++) {
			matmul_iq4_nl_r8_q8_qonly_f32_row(group, xq + ((size_t)t * xq_row_stride_blocks),
											  y + ((size_t)t * y_row_stride) + i, MR, k);
		}
	}
}
#undef NR

void dequant_iq4_nl_row(const void *blocks, size_t n_blocks, float *dst) {
	const iq4_nl_block *b		  = blocks;
	const __m128i		kvalues_u = _mm_loadu_si128((const __m128i *)(kvalues_iq4nl));
	const __m128i		lo_mask	  = _mm_set1_epi8(0x0F);

	for (size_t bi = 0; bi < n_blocks; bi++) {
		const float		d		= f16_to_f32_fast(b[bi].d);
		const __m256	d_vec	= _mm256_set1_ps(d);
		const __m128i	q		= _mm_loadu_si128((const __m128i *)(b[bi].qs));
		const __m128i	lo_idx	= _mm_and_si128(q, lo_mask);
		const __m128i	hi_idx	= _mm_and_si128(_mm_srli_epi16(q, 4), lo_mask);
		const __m128i	lo		= _mm_shuffle_epi8(kvalues_u, lo_idx);
		const __m128i	hi		= _mm_shuffle_epi8(kvalues_u, hi_idx);

		float *dst_lo = dst + (bi * 32);
		float *dst_hi = dst_lo + 16;

		__m256 lo_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
		__m256 lo_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
		__m256 hi_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
		__m256 hi_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));

		_mm256_storeu_ps(dst_lo + 0, _mm256_mul_ps(lo_f0, d_vec));
		_mm256_storeu_ps(dst_lo + 8, _mm256_mul_ps(lo_f1, d_vec));
		_mm256_storeu_ps(dst_hi + 0, _mm256_mul_ps(hi_f0, d_vec));
		_mm256_storeu_ps(dst_hi + 8, _mm256_mul_ps(hi_f1, d_vec));
	}
}

static void matmul_iq4_nl_q8_qonly_f32_row(const void *w, const q8_0_block *restrict xq,
										   float *restrict y, int n, int k) {
	const int			blocks_per_row = k / 32;
	const size_t		row_stride	   = (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const iq4_nl_block *Wb			   = w;
	const __m128i		kvalues_u	   = _mm_loadu_si128((const __m128i *)(kvalues_iq4nl));
	const __m128i		lo_mask		   = _mm_set1_epi8(0x0F);

	float	   xq_d_stack[256];
	__m256i	   xq32_stack[256];
	float	  *xq_d	 = xq_d_stack;
	__m256i	  *xq32	 = xq32_stack;
	void	  *xq_heap = NULL;

	if (blocks_per_row > 256) {
		xq_heap = xmalloc((size_t)blocks_per_row * (sizeof(float) + sizeof(__m256i)));
		xq_d	= (float *)xq_heap;
		xq32	= (__m256i *)(xq_d + blocks_per_row);
	}

	for (int bi = 0; bi < blocks_per_row; bi++) {
		xq_d[bi]  = f16_to_f32_fast(xq[bi].d);
		xq32[bi]  = _mm256_loadu_si256((const __m256i *)(xq[bi].qs));
	}

	int i = 0;

	for (; i + MR <= n; i += MR) {
		__m128					acc_lo = _mm_setzero_ps();
		__m128					acc_hi = _mm_setzero_ps();
		const iq4_nl_block *row_ptrs[MR];

		for (int r = 0; r < MR; r++)
			row_ptrs[r] =
				(const iq4_nl_block *)((const uint8_t *)Wb + ((size_t)(i + r) * row_stride));

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const float		d_xq	  = xq_d[bi];
			const __m256i	xq8		  = xq32[bi];

			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(&row_ptrs[r][bi + 1], 0, 1);
			}

			int32_t dots[8];
			for (int r = 0; r < 8; r++) {
				const __m128i q	 = _mm_loadu_si128((const __m128i *)(row_ptrs[r][bi].qs));
				const __m128i lo_idx = _mm_and_si128(q, lo_mask);
				const __m128i hi_idx = _mm_and_si128(_mm_srli_epi16(q, 4), lo_mask);
				const __m128i lo	 = _mm_shuffle_epi8(kvalues_u, lo_idx);
				const __m128i hi	 = _mm_shuffle_epi8(kvalues_u, hi_idx);
				const __m256i q32 =
					_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
				dots[r] = vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq8));
			}

			uint16_t d_w_raw[8];
			for (int r = 0; r < 8; r++)
				d_w_raw[r] = row_ptrs[r][bi].d;
			__m128 d_w0	 = loadu_f16x4_to_ps_128(d_w_raw);
			__m128 d_w1	 = loadu_f16x4_to_ps_128(d_w_raw + 4);
			__m128 d_xq_v = _mm_set1_ps(d_xq);

			__m128i sumi_lo = _mm_setr_epi32(dots[0], dots[1], dots[2], dots[3]);
			__m128i sumi_hi = _mm_setr_epi32(dots[4], dots[5], dots[6], dots[7]);

			acc_lo = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_lo), _mm_mul_ps(d_w0, d_xq_v), acc_lo);
			acc_hi = _mm_fmadd_ps(_mm_cvtepi32_ps(sumi_hi), _mm_mul_ps(d_w1, d_xq_v), acc_hi);
		}

		float tmp_lo[4];
		float tmp_hi[4];
		_mm_storeu_ps(tmp_lo, acc_lo);
		_mm_storeu_ps(tmp_hi, acc_hi);
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
			const float		 d_xq	 = xq_d[bi];
			const __m256i	 xq8	 = xq32[bi];
			const __m128i	 q		 = _mm_loadu_si128((const __m128i *)(row[bi].qs));
			const __m128i	 lo_idx	 = _mm_and_si128(q, lo_mask);
			const __m128i	 hi_idx	 = _mm_and_si128(_mm_srli_epi16(q, 4), lo_mask);
			const __m128i	 lo		 = _mm_shuffle_epi8(kvalues_u, lo_idx);
			const __m128i	 hi		 = _mm_shuffle_epi8(kvalues_u, hi_idx);
			const __m256i	 q32	 =
				_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
			const int32_t sumi = vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq8));
			const float d = f16_to_f32_fast(row[bi].d) * d_xq;
			sumf += d * (float)sumi;
		}
		y[i] = sumf;
	}

	free(xq_heap);
}

#define NR 4
void matmul_iq4_nl_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	const int			blocks_per_row = k / 32;
	const size_t		row_stride	   = (size_t)blocks_per_row * sizeof(iq4_nl_block);
	const iq4_nl_block *Wb			   = w;
	const __m128i		kvalues_u	   = _mm_loadu_si128((const __m128i *)(kvalues_iq4nl));
	const __m128i		lo_mask		   = _mm_set1_epi8(0x0F);
	int					i			   = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m128 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	xd[NR];
				__m256i xq32[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq32[c]	 = _mm256_loadu_si256((const __m256i *)(xrow[c][bi].qs));
				}
				const __m128 xd_vec = _mm_loadu_ps(xd);

				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(iq4_nl_block)),
										   0, 1);
				}

				for (int r = 0; r < MR; r++) {
					const iq4_nl_block *b =
						(const iq4_nl_block *)(row_base[r] + ((size_t)bi * sizeof(iq4_nl_block)));
					const float		d_w	= f16_to_f32_fast(b->d);
					const __m128i	q		= _mm_loadu_si128((const __m128i *)(b->qs));
					const __m128i	lo_idx = _mm_and_si128(q, lo_mask);
					const __m128i	hi_idx = _mm_and_si128(_mm_srli_epi16(q, 4), lo_mask);
					const __m128i	lo		= _mm_shuffle_epi8(kvalues_u, lo_idx);
					const __m128i	hi		= _mm_shuffle_epi8(kvalues_u, hi_idx);
					const __m256i	q32	=
						_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);

int32_t sumi_arr[NR];
					for (int c = 0; c < NR; c++)
						sumi_arr[c] = vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq32[c]));
					const __m128i sumi4 = _mm_loadu_si128((const __m128i *)sumi_arr);

					const __m128 sumi_f = _mm_cvtepi32_ps(sumi4);
					acc_row[r] = _mm_fmadd_ps(sumi_f, _mm_mul_ps(xd_vec, _mm_set1_ps(d_w)),
											  acc_row[r]);
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
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
			const q8_0_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const __m128i q	 = _mm_loadu_si128((const __m128i *)(row[bi].qs));
				const __m128i lo_idx = _mm_and_si128(q, lo_mask);
				const __m128i hi_idx = _mm_and_si128(_mm_srli_epi16(q, 4), lo_mask);
				const __m128i lo	 = _mm_shuffle_epi8(kvalues_u, lo_idx);
				const __m128i hi	 = _mm_shuffle_epi8(kvalues_u, hi_idx);
				const __m256i q32 =
					_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
				const __m256i xq8	= _mm256_loadu_si256((const __m256i *)(xrow[bi].qs));
				const int32_t sumi = vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq8));
				const float d = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				sumf += d * (float)sumi;
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
		__m128 acc0 = _mm_setzero_ps();
		__m128 acc1 = _mm_setzero_ps();
		__m128 off0 = _mm_setzero_ps();
		__m128 off1 = _mm_setzero_ps();

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const int8_t *restrict xq8 = xq[bi].qs;
			const float d_xq		   = f16_to_f32_fast(xq[bi].d);
			const float s_xq		   = f16_to_f32_fast(xq[bi].s);

			const __m256i xq32 = _mm256_loadu_si256((const __m256i *)(xq8));

			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_1_block)), 0, 1);
			}

			int32_t	 sumi_lane[8];
			uint16_t d_w_raw[8];
			uint16_t m_w_raw[8];

			for (int r = 0; r < 8; r++) {
				const q4_1_block *row =
					(const q4_1_block *)(row_base[r] + (size_t)bi * sizeof(q4_1_block));
				d_w_raw[r]				   = row->d;
				m_w_raw[r]				   = row->m;
				const __m128i q		   = _mm_loadu_si128((const __m128i *)(row->qs));
				const __m128i lo	   = _mm_and_si128(q, _mm_set1_epi8(0x0F));
				const __m128i hi	   = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0F));
				__m256i		  q32	   = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
				sumi_lane[r]		   = vreduce_add_epi32(dotprod_u8_s8_i32(q32, xq32));
			}
			__m128 sumi0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane)));
			__m128 sumi1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane + 4)));
			__m128 d_w0	 = loadu_f16x4_to_ps_128(d_w_raw);
			__m128 d_w1	 = loadu_f16x4_to_ps_128(d_w_raw + 4);
			__m128 m_w0	 = loadu_f16x4_to_ps_128(m_w_raw);
			__m128 m_w1	 = loadu_f16x4_to_ps_128(m_w_raw + 4);
			__m128 d_xq_v = _mm_set1_ps(d_xq);
			__m128 s_xq_v = _mm_set1_ps(s_xq);

			acc0 = _mm_fmadd_ps(sumi0, _mm_mul_ps(d_w0, d_xq_v), acc0);
			acc1 = _mm_fmadd_ps(sumi1, _mm_mul_ps(d_w1, d_xq_v), acc1);
			off0 = _mm_fmadd_ps(m_w0, s_xq_v, off0);
			off1 = _mm_fmadd_ps(m_w1, s_xq_v, off1);
		}

		__m128 final0 = _mm_add_ps(acc0, off0);
		__m128 final1 = _mm_add_ps(acc1, off1);
		float	tmp0[4], tmp1[4];
		_mm_storeu_ps(tmp0, final0);
		_mm_storeu_ps(tmp1, final1);
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

#define NR 4
void matmul_q4_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
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
			__m128 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_1_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	xd[NR];
				float	xs[NR];
				__m256i xq32[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xs[c]	 = f16_to_f32_fast(xrow[c][bi].s);
					xq32[c]	 = _mm256_loadu_si256((const __m256i *)(xrow[c][bi].qs));
				}
				const __m128 xd_vec = _mm_loadu_ps(xd);
				const __m128 xs_vec = _mm_loadu_ps(xs);

				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q4_1_block)), 0,
										   1);
				}

				for (int r = 0; r < MR; r++) {
					const q4_1_block *b =
						(const q4_1_block *)(row_base[r] + ((size_t)bi * sizeof(q4_1_block)));
					const __m128i q	 = _mm_loadu_si128((const __m128i *)(b->qs));
					const __m128i lo	 = _mm_and_si128(q, _mm_set1_epi8(0x0F));
					const __m128i hi	 = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0F));
					const __m256i q32 =
						_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
					const float d_w = f16_to_f32_fast(b->d);
					const float m_w = f16_to_f32_fast(b->m);

					int32_t sumi_arr[NR];
					for (int c = 0; c < NR; c++)
						sumi_arr[c] = vreduce_add_epi32(dotprod_u8_s8_i32(q32, xq32[c]));
					const __m128i sumi4 = _mm_loadu_si128((const __m128i *)sumi_arr);
					const __m128 sumi_f = _mm_cvtepi32_ps(sumi4);

					__m128 term = _mm_fmadd_ps(_mm_mul_ps(_mm_set1_ps(d_w), xd_vec), sumi_f,
											   _mm_mul_ps(_mm_set1_ps(m_w), xs_vec));
					acc_row[r]	= _mm_add_ps(acc_row[r], term);
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
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
				int sumi0				   = 0;
				int sumi1				   = 0;
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

static inline __m128i spread_byte_bits(uint8_t b) {
	__m128i v = _mm_set1_epi8((char)b);
	__m128i m = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
	return _mm_min_epu8(_mm_and_si128(v, m), _mm_set1_epi8(1));
}

static inline __m128i q5_bitmask_lo(uint32_t qh) {
	const __m128i lo_mask =
		_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);
	__m128i b0 = spread_byte_bits(qh & 0xFF);
	__m128i b1 = spread_byte_bits((qh >> 8) & 0xFF);
	return _mm_or_si128(_mm_and_si128(b0, lo_mask), _mm_andnot_si128(lo_mask, b1));
}

static inline __m128i q5_bitmask_hi(uint32_t qh) {
	const __m128i lo_mask =
		_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);
	__m128i b0 = spread_byte_bits((qh >> 16) & 0xFF);
	__m128i b1 = spread_byte_bits((qh >> 24) & 0xFF);
	return _mm_or_si128(_mm_and_si128(b0, lo_mask), _mm_andnot_si128(lo_mask, b1));
}

static inline void q5_0_unpack(const q5_0_block *b, __m128i *lo, __m128i *hi) {
	uint32_t qh;
	memcpy(&qh, b->qh, 4);

	const __m128i qs	  = _mm_loadu_si128((const __m128i *)(b->qs));
	const __m128i nib_lo  = _mm_and_si128(qs, _mm_set1_epi8(0x0F));
	const __m128i nib_hi  = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm_set1_epi8(0x0F));

	const __m128i qh_lo = _mm_slli_epi16(q5_bitmask_lo(qh), 4);
	const __m128i qh_hi = _mm_slli_epi16(q5_bitmask_hi(qh), 4);

	*lo = _mm_sub_epi8(_mm_or_si128(nib_lo, qh_lo), _mm_set1_epi8(16));
	*hi = _mm_sub_epi8(_mm_or_si128(nib_hi, qh_hi), _mm_set1_epi8(16));
}

static inline void q5_1_unpack(const q5_1_block *b, __m128i *lo, __m128i *hi) {
	uint32_t qh;
	memcpy(&qh, b->qh, 4);

	const __m128i qs	  = _mm_loadu_si128((const __m128i *)(b->qs));
	const __m128i nib_lo  = _mm_and_si128(qs, _mm_set1_epi8(0x0F));
	const __m128i nib_hi  = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm_set1_epi8(0x0F));

	const __m128i qh_lo = _mm_slli_epi16(q5_bitmask_lo(qh), 4);
	const __m128i qh_hi = _mm_slli_epi16(q5_bitmask_hi(qh), 4);

	*lo = _mm_or_si128(nib_lo, qh_lo);
	*hi = _mm_or_si128(nib_hi, qh_hi);
}

static inline int32_t q5_dot(__m128i lo, __m128i hi, __m256i xq) {
	const __m256i q32 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
	return vreduce_add_epi32(dotprod_s8_s8_i32(q32, xq));
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
			__m128i lo, hi;
			q5_0_unpack(&row[bi], &lo, &hi);
			const __m256i xq8 = _mm256_loadu_si256((const __m256i *)(xq[bi].qs));
			const int32_t sumi = q5_dot(lo, hi, xq8);
			const float	 d	  = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xq[bi].d);
			sumf += d * (float)sumi;
		}
		y[i] = sumf;
	}
}

#define NR 4
void matmul_q5_0_q8_qonly_f32(const void *w, const q8_0_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_0_block);
	const q5_0_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m128 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_0_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	xd[NR];
				__m256i xq32[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xq32[c]	 = _mm256_loadu_si256((const __m256i *)(xrow[c][bi].qs));
				}
				const __m128 xd_vec = _mm_loadu_ps(xd);

				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q5_0_block)), 0,
										   1);
				}

				for (int r = 0; r < MR; r++) {
					const q5_0_block *b =
						(const q5_0_block *)(row_base[r] + ((size_t)bi * sizeof(q5_0_block)));
					__m128i lo, hi;
					q5_0_unpack(b, &lo, &hi);
					const float d_w = f16_to_f32_fast(b->d);

					int32_t sumi_arr[NR];
					for (int c = 0; c < NR; c++)
						sumi_arr[c] = q5_dot(lo, hi, xq32[c]);
					const __m128i sumi4 = _mm_loadu_si128((const __m128i *)sumi_arr);

					const __m128 sumi_f = _mm_cvtepi32_ps(sumi4);
					acc_row[r] = _mm_fmadd_ps(sumi_f, _mm_mul_ps(xd_vec, _mm_set1_ps(d_w)),
											  acc_row[r]);
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
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
			const q8_0_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			__m128i			  lo, hi;
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				q5_0_unpack(&row[bi], &lo, &hi);
				const __m256i xq8	 = _mm256_loadu_si256((const __m256i *)(xrow[bi].qs));
				const int32_t sumi	 = q5_dot(lo, hi, xq8);
				const float		d	 = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xrow[bi].d);
				sumf += d * (float)sumi;
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}
#undef NR

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
			__m128i lo, hi;
			q5_1_unpack(&row[bi], &lo, &hi);
			const __m256i xq8 = _mm256_loadu_si256((const __m256i *)(xq[bi].qs));
			const int32_t sumi = q5_dot(lo, hi, xq8);
			const float	 d	  = f16_to_f32_fast(row[bi].d) * f16_to_f32_fast(xq[bi].d);
			const float	 m	  = f16_to_f32_fast(row[bi].m) * f16_to_f32_fast(xq[bi].s);
			sumf += (d * (float)sumi) + m;
		}
		y[i] = sumf;
	}
}

#define NR 4
void matmul_q5_1_q8_qonly_f32(const void *w, const q8_1_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	const int		  blocks_per_row = k / 32;
	const size_t	  row_stride	 = (size_t)blocks_per_row * sizeof(q5_1_block);
	const q5_1_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m128 acc_row[MR];
			__m128 off_row[MR];
			for (int r = 0; r < MR; r++) {
				acc_row[r] = _mm_setzero_ps();
				off_row[r] = _mm_setzero_ps();
			}

			const q8_1_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float	xd[NR];
				float	xs[NR];
				__m256i xq32[NR];
				for (int c = 0; c < NR; c++) {
					xd[c]	 = f16_to_f32_fast(xrow[c][bi].d);
					xs[c]	 = f16_to_f32_fast(xrow[c][bi].s);
					xq32[c]	 = _mm256_loadu_si256((const __m256i *)(xrow[c][bi].qs));
				}
				const __m128 xd_vec = _mm_loadu_ps(xd);
				const __m128 xs_vec = _mm_loadu_ps(xs);

				if (bi + 1 < blocks_per_row) {
					for (int r = 0; r < MR; r++)
						__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q5_1_block)), 0,
										   1);
				}

				for (int r = 0; r < MR; r++) {
					const q5_1_block *b =
						(const q5_1_block *)(row_base[r] + ((size_t)bi * sizeof(q5_1_block)));
					__m128i lo, hi;
					q5_1_unpack(b, &lo, &hi);
					const float d_w = f16_to_f32_fast(b->d);
					const float m_w = f16_to_f32_fast(b->m);

					int32_t sumi_arr[NR];
					for (int c = 0; c < NR; c++)
						sumi_arr[c] = q5_dot(lo, hi, xq32[c]);
					const __m128i sumi4 = _mm_loadu_si128((const __m128i *)sumi_arr);

					const __m128 sumi_f = _mm_cvtepi32_ps(sumi4);
					acc_row[r] = _mm_fmadd_ps(sumi_f, _mm_mul_ps(xd_vec, _mm_set1_ps(d_w)),
											  acc_row[r]);
					off_row[r] = _mm_fmadd_ps(xs_vec, _mm_set1_ps(m_w), off_row[r]);
				}
			}

			for (int r = 0; r < MR; r++) {
				__m128 total = _mm_add_ps(acc_row[r], off_row[r]);
				float tmp[4];
				_mm_storeu_ps(tmp, total);
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
			__m128i			  lo, hi;
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				q5_1_unpack(&row[bi], &lo, &hi);
				const __m256i xq8	 = _mm256_loadu_si256((const __m256i *)(xrow[bi].qs));
				const int32_t sumi	 = q5_dot(lo, hi, xq8);
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

static inline int32_t dot16_s8_s8(__m128i a, __m128i b) {
	__m256i a32 = _mm256_zextsi128_si256(a);
	__m256i b32 = _mm256_zextsi128_si256(b);
	return vreduce_add_epi32(dotprod_s8_s8_i32(a32, b32));
}

static inline int32_t dot16_u8_s8(__m128i a, __m128i b) {
	__m256i a32 = _mm256_zextsi128_si256(a);
	__m256i b32 = _mm256_zextsi128_si256(b);
	return vreduce_add_epi32(dotprod_u8_s8_i32(a32, b32));
}

static inline __m256i q6_pack32(__m128i a, __m128i b, __m128i sub_32) {
	return _mm256_set_m128i(_mm_sub_epi8(b, sub_32), _mm_sub_epi8(a, sub_32));
}

static inline void q6_k_unpack_ymm(const q6_k_block *b, __m256i out[8]) {
	const uint8_t *restrict ql = b->ql;
	const uint8_t *restrict qh = b->qh;
	const __m128i mask_0F	   = _mm_set1_epi8(0x0F);
	const __m128i mask_03	   = _mm_set1_epi8(0x03);
	const __m128i mask_F0	   = _mm_set1_epi8(0xF0);
	const __m128i sub_32	   = _mm_set1_epi8(32);

	for (int n_iter = 0; n_iter < 2; n_iter++) {
		__m128i ql0 = _mm_loadu_si128((const __m128i *)(ql));
		__m128i ql1 = _mm_loadu_si128((const __m128i *)(ql + 16));
		__m128i ql2 = _mm_loadu_si128((const __m128i *)(ql + 32));
		__m128i ql3 = _mm_loadu_si128((const __m128i *)(ql + 48));
		__m128i qh0 = _mm_loadu_si128((const __m128i *)(qh));
		__m128i qh1 = _mm_loadu_si128((const __m128i *)(qh + 16));

		__m128i s0a = _mm_and_si128(qh0, mask_03);
		__m128i s1a = _mm_and_si128(_mm_srli_epi16(qh0, 2), mask_03);
		__m128i s2a = _mm_and_si128(_mm_srli_epi16(qh0, 4), mask_03);
		__m128i s3a = _mm_and_si128(_mm_srli_epi16(qh0, 6), mask_03);
		__m128i s0b = _mm_and_si128(qh1, mask_03);
		__m128i s1b = _mm_and_si128(_mm_srli_epi16(qh1, 2), mask_03);
		__m128i s2b = _mm_and_si128(_mm_srli_epi16(qh1, 4), mask_03);
		__m128i s3b = _mm_and_si128(_mm_srli_epi16(qh1, 6), mask_03);

		__m128i v0 = _mm_or_si128(_mm_and_si128(ql0, mask_0F),
								  _mm_and_si128(_mm_slli_epi16(s0a, 4), mask_F0));
		__m128i v1 = _mm_or_si128(_mm_and_si128(ql1, mask_0F),
								  _mm_and_si128(_mm_slli_epi16(s0b, 4), mask_F0));
		__m128i v2 = _mm_or_si128(_mm_and_si128(ql2, mask_0F),
								  _mm_and_si128(_mm_slli_epi16(s1a, 4), mask_F0));
		__m128i v3 = _mm_or_si128(_mm_and_si128(ql3, mask_0F),
								  _mm_and_si128(_mm_slli_epi16(s1b, 4), mask_F0));
		__m128i v4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mask_0F),
								  _mm_and_si128(_mm_slli_epi16(s2a, 4), mask_F0));
		__m128i v5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mask_0F),
								  _mm_and_si128(_mm_slli_epi16(s2b, 4), mask_F0));
		__m128i v6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql2, 4), mask_0F),
								  _mm_and_si128(_mm_slli_epi16(s3a, 4), mask_F0));
		__m128i v7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql3, 4), mask_0F),
								  _mm_and_si128(_mm_slli_epi16(s3b, 4), mask_F0));

		__m256i *dst = out + n_iter * 4;
		dst[0]		 = q6_pack32(v0, v1, sub_32);
		dst[1]		 = q6_pack32(v2, v3, sub_32);
		dst[2]		 = q6_pack32(v4, v5, sub_32);
		dst[3]		 = q6_pack32(v6, v7, sub_32);
		ql += 64;
		qh += 32;
	}
}

static inline int32_t q6k_dot_ymm(const __m256i a[8], const int8_t *q8, const int8_t *sc) {
	__m256i acc = _mm256_setzero_si256();
	for (int j = 0; j < 8; j++) {
		__m256i d = dotprod_s8_s8_i32(_mm256_loadu_si256((const __m256i *)(q8 + j * 32)), a[j]);
		__m256i scv = _mm256_set_m128i(_mm_set1_epi32((int32_t)sc[2 * j + 1]),
									   _mm_set1_epi32((int32_t)sc[2 * j]));
		acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(d, scv));
	}
	return vreduce_add_epi32(acc);
}

static void matmul_q6_k_q8_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
										 float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q6_k_block);
	const q6_k_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		__m128 acc0 = _mm_setzero_ps();
		__m128 acc1 = _mm_setzero_ps();

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q6_k_block)), 0, 1);
			}
			const q8_k_block *restrict yb = &xq[bi];
			const float d_xq			  = yb->d;

			int32_t	 sumi_lane[8];
			uint16_t d_w_raw[8];

			for (int r = 0; r < MR; r++) {
				const q6_k_block *restrict b =
					(const q6_k_block *)(row_base[r] + ((size_t)bi * sizeof(q6_k_block)));
				d_w_raw[r] = b->d;
				__m256i ay[8];
				q6_k_unpack_ymm(b, ay);
				sumi_lane[r] = q6k_dot_ymm(ay, yb->qs, b->scales);
			}
			__m128 sumi0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane)));
			__m128 sumi1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane + 4)));
			__m128 d_w0	 = loadu_f16x4_to_ps_128(d_w_raw);
			__m128 d_w1	 = loadu_f16x4_to_ps_128(d_w_raw + 4);
			__m128 d_xq_v = _mm_set1_ps(d_xq);

			acc0 = _mm_fmadd_ps(sumi0, _mm_mul_ps(d_w0, d_xq_v), acc0);
			acc1 = _mm_fmadd_ps(sumi1, _mm_mul_ps(d_w1, d_xq_v), acc1);
		}

		float tmp0[4], tmp1[4];
		_mm_storeu_ps(tmp0, acc0);
		_mm_storeu_ps(tmp1, acc1);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
	}

	for (; i < n; i++) {
		const q6_k_block *bx = (const q6_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float			  sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q6_k_block *restrict b  = &bx[bi];
			const q8_k_block *restrict yb = &xq[bi];
			float d						  = f16_to_f32_fast(b->d) * yb->d;
			__m256i ay[8];
			q6_k_unpack_ymm(b, ay);
			sumf += d * (float)q6k_dot_ymm(ay, yb->qs, b->scales);
		}
		y[i] = sumf;
	}
}

#define NR 8
void matmul_q6_k_q8_qonly_f32(const void *w, const q8_k_block *restrict xq,
							  size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
							  int n, int k, int m) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q6_k_block);
	const q6_k_block *Wb			 = w;
	int				  i				 = 0;
	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / NR) > 0 ? blocks_per_row : 0;

		static _Thread_local __m256i (*q_ymm_cache)[MR][8] = NULL;
		static _Thread_local int8_t (*sc_cache)[MR][16]	   = NULL;
		static _Thread_local float (*d_w_cache)[MR]		   = NULL;
		static _Thread_local int cache_cap				   = 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				q_ymm_cache = ymm_cache_alloc(sizeof(*q_ymm_cache) * n_bi_tiles);
				sc_cache	= realloc(sc_cache, sizeof(*sc_cache) * n_bi_tiles);
				d_w_cache	= realloc(d_w_cache, sizeof(*d_w_cache) * n_bi_tiles);
				cache_cap	= n_bi_tiles;
				tlocal_register((void **)&q_ymm_cache);
				tlocal_register((void **)&sc_cache);
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
					memcpy(sc_cache[bi][r], b->scales, 16);
					q6_k_unpack_ymm(b, q_ymm_cache[bi][r]);
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m256 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm256_setzero_ps();

			const q8_k_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				float *d_w = d_w_cache[bi];

				for (int r = 0; r < MR; r++) {
					const int8_t *restrict sc = sc_cache[bi][r];
					const __m256i *restrict a = q_ymm_cache[bi][r];

					int32_t total_arr[NR];
					float	d_xq_arr[NR];

					for (int c = 0; c < NR; c++) {
						const q8_k_block *restrict yb = &xrow[c][bi];
						d_xq_arr[c]					  = yb->d;
						total_arr[c]				  = q6k_dot_ymm(a, yb->qs, sc);
					}

					__m256 total_f =
						_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(total_arr)));
					__m256 xd_vec = _mm256_loadu_ps(d_xq_arr);
					acc_row[r]	  = _mm256_fmadd_ps(
						 xd_vec, _mm256_mul_ps(total_f, _mm256_set1_ps(d_w[r])), acc_row[r]);
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[8];
				_mm256_storeu_ps(tmp, acc_row[r]);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
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
			const q8_k_block *xrow = xq + ((size_t)t * xq_row_stride_blocks);
			float			  sumf = 0.0f;
			for (int bi = 0; bi < blocks_per_row; bi++) {
				const q6_k_block *restrict b  = &bx[bi];
				const q8_k_block *restrict yb = &xrow[bi];
				float d						  = f16_to_f32_fast(b->d) * yb->d;
				__m256i ay[8];
				q6_k_unpack_ymm(b, ay);
				sumf += d * (float)q6k_dot_ymm(ay, yb->qs, b->scales);
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}
#undef NR

static inline void q4k_block_dot(const q4_k_block *b, const q8_k_block *xb, int32_t *sumi_out,
								 int32_t *summ_out) {
	const uint8_t *restrict qbytes = b->qs;
	const uint8_t *restrict sc	   = b->scales;
	const int8_t *restrict xq8	   = xb->qs;
	const int16_t *restrict bs	   = xb->bsums;
	__m256i sumi_v				   = _mm256_setzero_si256();
	int32_t summ					   = 0;
	int		is					   = 0;
	int		ib					   = 0;
	const __m256i lo_mask		   = _mm256_set1_epi8(0x0F);
	for (int g = 0; g < 4; g++) {
		uint8_t scu8;
		uint8_t mu8;
		get_scale_min_k4(is + 0, sc, &scu8, &mu8);
		int s0 = scu8;
		int m0 = mu8;
		get_scale_min_k4(is + 1, sc, &scu8, &mu8);
		int s1				   = scu8;
		int m1				   = mu8;
		const uint8_t *restrict qg = qbytes + (g * 32);
		const int8_t *restrict xq0 = xq8 + (g * 64);
		const int8_t *restrict xq1 = xq8 + (g * 64) + 32;
		const __m256i qg_v = _mm256_loadu_si256((const __m256i *)qg);
		const __m256i lo_u = _mm256_and_si256(qg_v, lo_mask);
		const __m256i hi_u = _mm256_and_si256(_mm256_srli_epi16(qg_v, 4), lo_mask);
		const __m256i xq0v = _mm256_loadu_si256((const __m256i *)xq0);
		const __m256i xq1v = _mm256_loadu_si256((const __m256i *)xq1);
		const __m256i d0 = dotprod_u8_s8_i32(lo_u, xq0v);
		const __m256i d1 = dotprod_u8_s8_i32(hi_u, xq1v);
		sumi_v = _mm256_add_epi32(
			sumi_v,
			_mm256_add_epi32(_mm256_mullo_epi32(d0, _mm256_set1_epi32(s0)),
							 _mm256_mullo_epi32(d1, _mm256_set1_epi32(s1))));
		summ += m0 * (int32_t)((int32_t)bs[ib] + (int32_t)bs[ib + 1]);
		ib += 2;
		summ += m1 * (int32_t)((int32_t)bs[ib] + (int32_t)bs[ib + 1]);
		ib += 2;
		is += 2;
	}
	*sumi_out = vreduce_add_epi32(sumi_v);
	*summ_out = summ;
}

static void matmul_q4_k_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
										   float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q4_k_block);
	const q4_k_block *Wb			 = w;

	for (int i = 0; i < n; i++) {
		const q4_k_block *row =
			(const q4_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q4_k_block *restrict b  = &row[bi];
			const q8_k_block *restrict xb = &xq[bi];
			float d						  = f16_to_f32_fast(b->d);
			float dmin					  = f16_to_f32_fast(b->dmin);
			float xd					  = xb->d;
			int32_t sumi;
			int32_t summ;
			q4k_block_dot(b, xb, &sumi, &summ);
			sumf += xd * ((d * (float)sumi) - (dmin * (float)summ));
		}
		y[i] = sumf;
	}
}

#define NR 8
void matmul_q4_k_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								int n, int k, int m) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q4_k_block);
	const q4_k_block *Wb			 = w;
	int				  i				 = 0;
	const __m256i	  lo_mask		 = _mm256_set1_epi8(0x0F);

	for (; i + MR <= n; i += MR) {
		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / NR) > 0 ? blocks_per_row : 0;

		static _Thread_local __m256i (*wlo_cache)[MR][4] = NULL;
		static _Thread_local __m256i (*whi_cache)[MR][4] = NULL;
		static _Thread_local int32_t (*s_lo_cache)[MR][4]	= NULL;
		static _Thread_local int32_t (*s_hi_cache)[MR][4]	= NULL;
		static _Thread_local int32_t (*m_lo_cache)[MR][4]	= NULL;
		static _Thread_local int32_t (*m_hi_cache)[MR][4]	= NULL;
		static _Thread_local float (*d_w_cache)[MR]			= NULL;
		static _Thread_local float (*dmin_w_cache)[MR]		= NULL;
		static _Thread_local int cache_cap					= 0;

		if (n_bi_tiles > 0) {
			if (cache_cap < n_bi_tiles) {
				wlo_cache	 = ymm_cache_alloc(sizeof(*wlo_cache) * n_bi_tiles);
				whi_cache	 = ymm_cache_alloc(sizeof(*whi_cache) * n_bi_tiles);
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
					d_w_cache[bi][r]			 = f16_to_f32_fast(b->d);
					dmin_w_cache[bi][r]			 = f16_to_f32_fast(b->dmin);
					const uint8_t *restrict qbytes = b->qs;
					const uint8_t *restrict sc	 = b->scales;

					int is = 0;
					for (int g = 0; g < 4; g++) {
						uint8_t scu8;
						uint8_t mu8;
						get_scale_min_k4(is + 0, sc, &scu8, &mu8);
						s_lo_cache[bi][r][g] = (int32_t)scu8;
						m_lo_cache[bi][r][g] = (int32_t)mu8;
						get_scale_min_k4(is + 1, sc, &scu8, &mu8);
						s_hi_cache[bi][r][g] = (int32_t)scu8;
						m_hi_cache[bi][r][g] = (int32_t)mu8;
						is += 2;

						const __m256i qg_v =
							_mm256_loadu_si256((const __m256i *)(qbytes + g * 32));
						wlo_cache[bi][r][g] = _mm256_and_si256(qg_v, lo_mask);
						whi_cache[bi][r][g] =
							_mm256_and_si256(_mm256_srli_epi16(qg_v, 4), lo_mask);
					}
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m256 acc_row[MR];
			for (int r = 0; r < MR; r++)
				acc_row[r] = _mm256_setzero_ps();

			const q8_k_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				__m256i (*wlo)[4]	 = wlo_cache[bi];
				__m256i (*whi)[4]	 = whi_cache[bi];
				int32_t (*s_lo)[4]	 = s_lo_cache[bi];
				int32_t (*s_hi)[4]	 = s_hi_cache[bi];
				int32_t (*m_lo)[4]	 = m_lo_cache[bi];
				int32_t (*m_hi)[4]	 = m_hi_cache[bi];
				float *d_w			 = d_w_cache[bi];
				float *dmin_w		 = dmin_w_cache[bi];

				int32_t sumi_arr[MR][NR];
				int32_t summ_arr[MR][NR];
				float	xd_arr[NR];

				for (int c = 0; c < NR; c++) {
					const q8_k_block *restrict xb = &xrow[c][bi];
					xd_arr[c]					  = xb->d;
					const int8_t *restrict xq8	  = xb->qs;
					const int16_t *restrict bs	  = xb->bsums;

					int32_t bsum_pre[8];
					for (int g = 0; g < 4; g++) {
						int ib				  = g * 4;
						bsum_pre[g * 2]		  = (int32_t)bs[ib] + (int32_t)bs[ib + 1];
						bsum_pre[(g * 2) + 1] = (int32_t)bs[ib + 2] + (int32_t)bs[ib + 3];
					}

					__m256i xq0_v[4];
					__m256i xq1_v[4];
					for (int g = 0; g < 4; g++) {
						xq0_v[g] = _mm256_loadu_si256((const __m256i *)(xq8 + g * 64));
						xq1_v[g] = _mm256_loadu_si256((const __m256i *)(xq8 + g * 64 + 32));
					}

					for (int r = 0; r < MR; r++) {
						__m256i acc = _mm256_setzero_si256();
						int32_t summ = 0;
						for (int g = 0; g < 4; g++) {
							__m256i d0 = dotprod_u8_s8_i32(wlo[r][g], xq0_v[g]);
							__m256i d1 = dotprod_u8_s8_i32(whi[r][g], xq1_v[g]);
							acc = _mm256_add_epi32(
								acc, _mm256_add_epi32(
										 _mm256_mullo_epi32(d0, _mm256_set1_epi32(s_lo[r][g])),
										 _mm256_mullo_epi32(d1, _mm256_set1_epi32(s_hi[r][g]))));
							summ += m_lo[r][g] * bsum_pre[g * 2];
							summ += m_hi[r][g] * bsum_pre[(g * 2) + 1];
						}
						sumi_arr[r][c] = vreduce_add_epi32(acc);
						summ_arr[r][c] = summ;
					}
				}

				const __m256 xd_vec = _mm256_loadu_ps(xd_arr);
				for (int r = 0; r < MR; r++) {
					__m256 sumi_f =
						_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(sumi_arr[r])));
					__m256 summ_f =
						_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(summ_arr[r])));
					__m256 d_w_v	 = _mm256_set1_ps(d_w[r]);
					__m256 dmin_w_v = _mm256_set1_ps(dmin_w[r]);
					__m256 val =
						_mm256_sub_ps(_mm256_mul_ps(d_w_v, sumi_f), _mm256_mul_ps(dmin_w_v, summ_f));
					acc_row[r] = _mm256_fmadd_ps(xd_vec, val, acc_row[r]);
				}
			}

			for (int r = 0; r < MR; r++) {
				float tmp[8];
				_mm256_storeu_ps(tmp, acc_row[r]);
				for (int c = 0; c < NR; c++)
					y[((size_t)(t + c) * y_row_stride) + (i + r)] = tmp[c];
			}
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
				const q4_k_block *restrict b  = &row[bi];
				const q8_k_block *restrict xb = &xrow[bi];
				float d						  = f16_to_f32_fast(b->d);
				float dmin					  = f16_to_f32_fast(b->dmin);
				float xd					  = xb->d;
				int32_t sumi;
				int32_t summ;
				q4k_block_dot(b, xb, &sumi, &summ);
				sumf += xd * ((d * (float)sumi) - (dmin * (float)summ));
			}
			y[((size_t)t * y_row_stride) + i] = sumf;
		}
	}
}
#undef NR

static inline void q5k_block_dot(const q5_k_block *b, const q8_k_block *xb, int32_t *sumi_out,
								 int32_t *summ_out) {
	const uint8_t *restrict qbytes = b->qs;
	const uint8_t *restrict qh	   = b->qh;
	const uint8_t *restrict sc	   = b->scales;
	const int8_t *restrict xq8	   = xb->qs;
	const int16_t *restrict bs	   = xb->bsums;
	int32_t sumi					   = 0;
	int32_t summ					   = 0;
	int		is					   = 0;
	int		ib					   = 0;
	uint8_t u1					   = 1;
	uint8_t u2					   = 2;
	const __m128i zero			   = _mm_setzero_si128();
	const __m128i mask_0F		   = _mm_set1_epi8(0x0F);
	const __m128i mask_16		   = _mm_set1_epi8(16);

	for (int g = 0; g < 4; g++) {
		uint8_t scu8;
		uint8_t mu8;
		get_scale_min_k4(is + 0, sc, &scu8, &mu8);
		int s0 = scu8;
		int m0 = mu8;
		get_scale_min_k4(is + 1, sc, &scu8, &mu8);
		int s1				   = scu8;
		int m1				   = mu8;
		const uint8_t *restrict qsg = qbytes + (g * 32);
		const int8_t *restrict xq0	= xq8 + (g * 64);
		const int8_t *restrict xq1	= xq8 + (g * 64) + 32;
		int32_t d0				   = 0;
		int32_t d1				   = 0;
		__m128i u1_vec			   = _mm_set1_epi8((char)u1);
		__m128i u2_vec			   = _mm_set1_epi8((char)u2);

		for (int half = 0; half < 2; half++) {
			__m128i qg_v = _mm_loadu_si128((const __m128i *)(qsg + half * 16));
			__m128i qh_v = _mm_loadu_si128((const __m128i *)(qh + half * 16));
			__m128i lo_nibble = _mm_and_si128(qg_v, mask_0F);
			__m128i hi_nibble = _mm_and_si128(_mm_srli_epi16(qg_v, 4), mask_0F);
			__m128i bit0	  = _mm_and_si128(qh_v, u1_vec);
			__m128i bit1	  = _mm_and_si128(qh_v, u2_vec);
			__m128i add0	  = _mm_andnot_si128(_mm_cmpeq_epi8(bit0, zero), mask_16);
			__m128i add1	  = _mm_andnot_si128(_mm_cmpeq_epi8(bit1, zero), mask_16);
			__m128i lo_u	  = _mm_add_epi8(lo_nibble, add0);
			__m128i hi_u	  = _mm_add_epi8(hi_nibble, add1);
			__m128i xq0v	  = _mm_loadu_si128((const __m128i *)(xq0 + half * 16));
			__m128i xq1v	  = _mm_loadu_si128((const __m128i *)(xq1 + half * 16));
			d0 += dot16_u8_s8(lo_u, xq0v);
			d1 += dot16_u8_s8(hi_u, xq1v);
		}

		sumi += (s0 * d0) + (s1 * d1);
		summ += m0 * (int32_t)((int32_t)bs[ib] + (int32_t)bs[ib + 1]);
		summ += m1 * (int32_t)((int32_t)bs[ib + 2] + (int32_t)bs[ib + 3]);
		ib += 4;
		is += 2;
		u1 <<= 2;
		u2 <<= 2;
	}
	*sumi_out = sumi;
	*summ_out = summ;
}

static void matmul_q5_k_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
										   float *restrict y, int n, int k) {
	int				  blocks_per_row = k / 256;
	size_t			  row_stride	 = (size_t)blocks_per_row * sizeof(q5_k_block);
	const q5_k_block *Wb			 = w;
	int				  i				 = 0;

	for (; i + MR <= n; i += MR) {
		__m128 acc0 = _mm_setzero_ps();
		__m128 acc1 = _mm_setzero_ps();

		const uint8_t *row_base[MR];
		for (int r = 0; r < MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		for (int bi = 0; bi < blocks_per_row; bi++) {
			if (bi + 1 < blocks_per_row) {
				for (int r = 0; r < MR; r++)
					__builtin_prefetch(row_base[r] + ((size_t)(bi + 1) * sizeof(q5_k_block)), 0, 1);
			}
			const q8_k_block *restrict xb = &xq[bi];
			const float xd				  = xb->d;

			int32_t sumi_lane[8];
			int32_t summ_lane[8];
			float	d_w[8];
			float	dmin_w[8];

			for (int r = 0; r < MR; r++) {
				const q5_k_block *restrict b =
					(const q5_k_block *)(row_base[r] + ((size_t)bi * sizeof(q5_k_block)));
				d_w[r]						   = f16_to_f32_fast(b->d);
				dmin_w[r]					   = f16_to_f32_fast(b->dmin);
				q5k_block_dot(b, xb, &sumi_lane[r], &summ_lane[r]);
			}

			__m128 sumi0f = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane)));
			__m128 sumi1f = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_lane + 4)));
			__m128 summ0f = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(summ_lane)));
			__m128 summ1f = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(summ_lane + 4)));
			__m128 d_w0	 = _mm_loadu_ps(d_w);
			__m128 d_w1	 = _mm_loadu_ps(d_w + 4);
			__m128 dm0	 = _mm_loadu_ps(dmin_w);
			__m128 dm1	 = _mm_loadu_ps(dmin_w + 4);
			__m128 xd_v	 = _mm_set1_ps(xd);

			acc0 = _mm_fmadd_ps(xd_v, _mm_sub_ps(_mm_mul_ps(d_w0, sumi0f), _mm_mul_ps(dm0, summ0f)),
								acc0);
			acc1 = _mm_fmadd_ps(xd_v, _mm_sub_ps(_mm_mul_ps(d_w1, sumi1f), _mm_mul_ps(dm1, summ1f)),
								acc1);
		}

		float tmp0[4], tmp1[4];
		_mm_storeu_ps(tmp0, acc0);
		_mm_storeu_ps(tmp1, acc1);
		for (int r = 0; r < 4; r++) {
			y[i + r]	 = tmp0[r];
			y[i + 4 + r] = tmp1[r];
		}
	}

	for (; i < n; i++) {
		const q5_k_block *row =
			(const q5_k_block *)((const uint8_t *)Wb + ((size_t)i * row_stride));
		float sumf = 0.0f;
		for (int bi = 0; bi < blocks_per_row; bi++) {
			const q5_k_block *restrict b  = &row[bi];
			const q8_k_block *restrict xb = &xq[bi];
			float d						  = f16_to_f32_fast(b->d);
			float dmin					  = f16_to_f32_fast(b->dmin);
			float xd					  = xb->d;
			int32_t sumi;
			int32_t summ;
			q5k_block_dot(b, xb, &sumi, &summ);
			sumf += xd * ((d * (float)sumi) - (dmin * (float)summ));
		}
		y[i] = sumf;
	}
}

static void matmul_q4_k_r8_q8_k_qonly_f32_row(const void *w, const q8_k_block *restrict xq,
											  float *restrict y, int n, int k) {
	const int	   blocks_per_row = k / 256;
	const uint8_t *Wb			  = w;
	const __m256i  lo_mask		  = _mm256_set1_epi8(0x0F);
	int			   i			  = 0;

	for (; i + MR <= n; i += MR) {
		__m256 acc	  = _mm256_setzero_ps();
		const uint8_t *group =
			Wb + ((size_t)(i / Q4_K_R8_ROWS)) * blocks_per_row * Q4_K_R8_GROUP_BYTES;

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t *blk = group + ((size_t)bi * Q4_K_R8_GROUP_BYTES);
			if (bi + 1 < blocks_per_row)
				__builtin_prefetch(blk + Q4_K_R8_GROUP_BYTES, 0, 1);

			const q8_k_block *restrict xb = &xq[bi];
			const float xd				  = xb->d;
			const int8_t *restrict xq8	  = xb->qs;
			const int16_t *restrict bs	  = xb->bsums;
			__m256i xq0_v[4], xq1_v[4];
			for (int g = 0; g < 4; g++) {
				xq0_v[g] = _mm256_loadu_si256((const __m256i *)(xq8 + (g * 64)));
				xq1_v[g] = _mm256_loadu_si256((const __m256i *)(xq8 + (g * 64) + 32));
			}
			int32_t bsum_pre[8];
			for (int g = 0; g < 4; g++) {
				int ib				  = g * 4;
				bsum_pre[g * 2]		  = (int32_t)bs[ib] + (int32_t)bs[ib + 1];
				bsum_pre[(g * 2) + 1] = (int32_t)bs[ib + 2] + (int32_t)bs[ib + 3];
			}

			const uint16_t *d_ptr	 = (const uint16_t *)blk;
			const uint16_t *dmin_ptr = d_ptr + Q4_K_R8_ROWS;
			const uint8_t  *sc_base	 = blk + Q4_K_R8_OFF_SE;
			const uint8_t  *qs_base	 = blk + Q4_K_R8_OFF_QS;
			int32_t			sumi_lane[8];
			int32_t			summ_lane[8];
			for (int r = 0; r < MR; r++) {
				const uint8_t *restrict sc	   = sc_base + ((size_t)r * 16);
				const uint8_t *restrict qbytes = qs_base + ((size_t)r * 128);
				__m256i sumi_v				   = _mm256_setzero_si256();
				int32_t summ				   = 0;
				for (int g = 0; g < 4; g++) {
					int s0 = sc[g * 4 + 0], m0 = sc[g * 4 + 1];
					int s1 = sc[g * 4 + 2], m1 = sc[g * 4 + 3];
					const __m256i qg   = _mm256_loadu_si256((const __m256i *)(qbytes + (g * 32)));
					const __m256i lo_u = _mm256_and_si256(qg, lo_mask);
					const __m256i hi_u = _mm256_and_si256(_mm256_srli_epi16(qg, 4), lo_mask);
					sumi_v			   = _mm256_add_epi32(
						  sumi_v, _mm256_add_epi32(maddubs_scale_i32(lo_u, xq0_v[g], s0),
												   maddubs_scale_i32(hi_u, xq1_v[g], s1)));
					summ += m0 * bsum_pre[g * 2];
					summ += m1 * bsum_pre[(g * 2) + 1];
				}
				sumi_lane[r] = vreduce_add_epi32(sumi_v);
				summ_lane[r] = summ;
			}
			__m256 sumi_f = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)sumi_lane));
			__m256 summ_f = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)summ_lane));
			__m256 d_w	  = loadu_f16x8_to_ps(d_ptr);
			__m256 dmin_w = loadu_f16x8_to_ps(dmin_ptr);
			__m256 val =
				_mm256_sub_ps(_mm256_mul_ps(d_w, sumi_f), _mm256_mul_ps(dmin_w, summ_f));
			acc = _mm256_fmadd_ps(_mm256_set1_ps(xd), val, acc);
		}
		_mm256_storeu_ps(y + i, acc);
	}
}

void matmul_q4_k_r8_q8_k_qonly_f32(const void *w, const q8_k_block *restrict xq,
								   size_t xq_row_stride_blocks, float *restrict y, int y_row_stride,
								   int n, int k, int m) {
	if (m == 1) {
		matmul_q4_k_r8_q8_k_qonly_f32_row(w, xq, y, n, k);
		return;
	}

	const int	   blocks_per_row = k / 256;
	const uint8_t *Wb			  = w;
	const __m256i  lo_mask		  = _mm256_set1_epi8(0x0F);

	static _Thread_local float *acc_buf = NULL;
	static _Thread_local int	acc_cap = 0;
	const int					need	= MR * m;
	if (acc_cap < need) {
		float *nb = realloc(acc_buf, (size_t)need * sizeof(float));
		if (!nb) {
			for (int t = 0; t < m; t++)
				matmul_q4_k_r8_q8_k_qonly_f32_row(w, xq + ((size_t)t * xq_row_stride_blocks),
												  y + ((size_t)t * y_row_stride), n, k);
			return;
		}
		acc_buf = nb;
		acc_cap = need;
		tlocal_register((void **)&acc_buf);
	}

	int i = 0;
	for (; i + MR <= n; i += MR) {
		const uint8_t *group =
			Wb + ((size_t)(i / Q4_K_R8_ROWS)) * blocks_per_row * Q4_K_R8_GROUP_BYTES;
		for (int t = 0; t < m; t++)
			_mm256_storeu_ps(acc_buf + (size_t)t * MR, _mm256_setzero_ps());

		for (int bi = 0; bi < blocks_per_row; bi++) {
			const uint8_t *blk = group + ((size_t)bi * Q4_K_R8_GROUP_BYTES);
			if (bi + 1 < blocks_per_row)
				__builtin_prefetch(blk + Q4_K_R8_GROUP_BYTES, 0, 1);

			const uint16_t *d_ptr	 = (const uint16_t *)blk;
			const uint16_t *dmin_ptr = d_ptr + Q4_K_R8_ROWS;
			const uint8_t  *sc_base	 = blk + Q4_K_R8_OFF_SE;
			const uint8_t  *qs_base	 = blk + Q4_K_R8_OFF_QS;
			__m256			d_w		 = loadu_f16x8_to_ps(d_ptr);
			__m256			dmin_w	 = loadu_f16x8_to_ps(dmin_ptr);

			__m256i wlo[MR][4], whi[MR][4];
			int32_t s_lo[MR][4], s_hi[MR][4], m_lo[MR][4], m_hi[MR][4];
			for (int r = 0; r < MR; r++) {
				const uint8_t *restrict sc = sc_base + ((size_t)r * 16);
				const uint8_t *qbytes	   = qs_base + ((size_t)r * 128);
				for (int g = 0; g < 4; g++) {
					s_lo[r][g] = (int32_t)sc[g * 4 + 0];
					m_lo[r][g] = (int32_t)sc[g * 4 + 1];
					s_hi[r][g] = (int32_t)sc[g * 4 + 2];
					m_hi[r][g] = (int32_t)sc[g * 4 + 3];
					const __m256i qg = _mm256_loadu_si256((const __m256i *)(qbytes + (g * 32)));
					wlo[r][g]		 = _mm256_and_si256(qg, lo_mask);
					whi[r][g]		 = _mm256_and_si256(_mm256_srli_epi16(qg, 4), lo_mask);
				}
			}

			for (int t = 0; t < m; t++) {
				const q8_k_block *restrict xb = xq + ((size_t)t * xq_row_stride_blocks) + bi;
				const float xd				  = xb->d;
				const int8_t *restrict xq8	  = xb->qs;
				const int16_t *restrict bs	  = xb->bsums;
				int32_t bsum_pre[8];
				for (int g = 0; g < 4; g++) {
					int ib				  = g * 4;
					bsum_pre[g * 2]		  = (int32_t)bs[ib] + (int32_t)bs[ib + 1];
					bsum_pre[(g * 2) + 1] = (int32_t)bs[ib + 2] + (int32_t)bs[ib + 3];
				}
				__m256i xq0_v[4], xq1_v[4];
				for (int g = 0; g < 4; g++) {
					xq0_v[g] = _mm256_loadu_si256((const __m256i *)(xq8 + (g * 64)));
					xq1_v[g] = _mm256_loadu_si256((const __m256i *)(xq8 + (g * 64) + 32));
				}
				int32_t sumi_lane[8];
				int32_t summ_lane[8];
				for (int r = 0; r < MR; r++) {
					__m256i acc	 = _mm256_setzero_si256();
					int32_t summ = 0;
					for (int g = 0; g < 4; g++) {
						acc = _mm256_add_epi32(
							acc, _mm256_add_epi32(maddubs_scale_i32(wlo[r][g], xq0_v[g], s_lo[r][g]),
												  maddubs_scale_i32(whi[r][g], xq1_v[g], s_hi[r][g])));
						summ += m_lo[r][g] * bsum_pre[g * 2];
						summ += m_hi[r][g] * bsum_pre[(g * 2) + 1];
					}
					sumi_lane[r] = vreduce_add_epi32(acc);
					summ_lane[r] = summ;
				}
				__m256 sumi_f = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)sumi_lane));
				__m256 summ_f = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)summ_lane));
				__m256 val =
					_mm256_sub_ps(_mm256_mul_ps(d_w, sumi_f), _mm256_mul_ps(dmin_w, summ_f));
				float *ap = acc_buf + (size_t)t * MR;
				_mm256_storeu_ps(ap, _mm256_fmadd_ps(_mm256_set1_ps(xd), val, _mm256_loadu_ps(ap)));
			}
		}

		for (int t = 0; t < m; t++)
			_mm256_storeu_ps(y + ((size_t)t * y_row_stride) + i,
							 _mm256_loadu_ps(acc_buf + (size_t)t * MR));
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

	const __m128i zero			= _mm_setzero_si128();
	const __m128i mask_0F		= _mm_set1_epi8(0x0F);
	const __m128i mask_16		= _mm_set1_epi8(16);

	for (; i + MATMUL_MR <= n; i += MATMUL_MR) {
		const uint8_t *row_base[MATMUL_MR];
		for (int r = 0; r < MATMUL_MR; r++)
			row_base[r] = (const uint8_t *)Wb + ((size_t)(i + r) * row_stride);

		const int n_bi_tiles = (m / NR) > 0 ? blocks_per_row : 0;

		static _Thread_local __m128i (*lo_cache)[MATMUL_MR][4][2] = NULL;
		static _Thread_local __m128i (*hi_cache)[MATMUL_MR][4][2] = NULL;
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
					d_cache[bi][r]			   = f16_to_f32_fast(b->d);
					dmin_cache[bi][r]		   = f16_to_f32_fast(b->dmin);
					const uint8_t *restrict qbytes = b->qs;
					const uint8_t *restrict qh = b->qh;
					const uint8_t *restrict sc = b->scales;

					int		is	 = 0;
					uint8_t u1	 = 1;
					uint8_t u2	 = 2;

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
						__m128i u1_vec				= _mm_set1_epi8((char)u1);
						__m128i u2_vec				= _mm_set1_epi8((char)u2);

						for (int half = 0; half < 2; half++) {
							__m128i qg_v = _mm_loadu_si128((const __m128i *)(qsg + half * 16));
							__m128i qh_v = _mm_loadu_si128((const __m128i *)(qh + half * 16));
							__m128i lo_nibble = _mm_and_si128(qg_v, mask_0F);
							__m128i hi_nibble = _mm_and_si128(_mm_srli_epi16(qg_v, 4), mask_0F);
							__m128i bit0	  = _mm_and_si128(qh_v, u1_vec);
							__m128i bit1	  = _mm_and_si128(qh_v, u2_vec);
							__m128i add0	  = _mm_andnot_si128(_mm_cmpeq_epi8(bit0, zero), mask_16);
							__m128i add1	  = _mm_andnot_si128(_mm_cmpeq_epi8(bit1, zero), mask_16);
							__m128i lo_u	  = _mm_add_epi8(lo_nibble, add0);
							__m128i hi_u	  = _mm_add_epi8(hi_nibble, add1);
							lo_cache[bi][r][g][half] = lo_u;
							hi_cache[bi][r][g][half] = hi_u;
						}

						u1 <<= 2;
						u2 <<= 2;
					}
				}
			}
		}

		int t = 0;
		for (; t + NR <= m; t += NR) {
			__m128 acc_row[MATMUL_MR];
			for (int r = 0; r < MATMUL_MR; r++)
				acc_row[r] = _mm_setzero_ps();

			const q8_k_block *xrow[NR];
			for (int c = 0; c < NR; c++)
				xrow[c] = xq + ((size_t)(t + c) * xq_row_stride_blocks);

			for (int bi = 0; bi < blocks_per_row; bi++) {
				__m128i (*lo)[4][2] = lo_cache[bi];
				__m128i (*hi)[4][2] = hi_cache[bi];
				int32_t (*s0)[4]	= s0_cache[bi];
				int32_t (*s1)[4]	= s1_cache[bi];
				int32_t (*m0)[4]	= m0_cache[bi];
				int32_t (*m1)[4]	= m1_cache[bi];
				float *d			= d_cache[bi];
				float *dmin			= dmin_cache[bi];

				int32_t sumi_arr[MATMUL_MR][NR];
				int32_t summ_arr[MATMUL_MR][NR];
				float	xd_arr[NR];

				for (int c = 0; c < NR; c++) {
					const q8_k_block *restrict xb = &xrow[c][bi];
					xd_arr[c]					  = xb->d;
					const int8_t *restrict xq8	  = xb->qs;
					const int16_t *restrict bs	  = xb->bsums;

					__m128i xq_cache[4][2][2];
					for (int g = 0; g < 4; g++) {
						const int8_t *xq0 = xq8 + (g * 64);
						const int8_t *xq1 = xq8 + (g * 64) + 32;
						for (int half = 0; half < 2; half++) {
							xq_cache[g][half][0] = _mm_loadu_si128((const __m128i *)(xq0 + half * 16));
							xq_cache[g][half][1] = _mm_loadu_si128((const __m128i *)(xq1 + half * 16));
						}
					}

					for (int r = 0; r < MATMUL_MR; r++) {
						int32_t sumi = 0;
						int32_t summ = 0;
						int		ib	 = 0;

						for (int g = 0; g < 4; g++) {
							int32_t d0 = 0;
							int32_t d1 = 0;
							for (int half = 0; half < 2; half++) {
								d0 += dot16_u8_s8(lo[r][g][half], xq_cache[g][half][0]);
								d1 += dot16_u8_s8(hi[r][g][half], xq_cache[g][half][1]);
							}

							sumi += (s0[r][g] * d0) + (s1[r][g] * d1);
							summ += m0[r][g] * (int32_t)((int32_t)bs[ib] + (int32_t)bs[ib + 1]);
							summ += m1[r][g] * (int32_t)((int32_t)bs[ib + 2] + (int32_t)bs[ib + 3]);
							ib += 4;
						}
						sumi_arr[r][c] = sumi;
						summ_arr[r][c] = summ;
					}
				}

				const __m128 xd_vec = _mm_loadu_ps(xd_arr);
				for (int r = 0; r < MATMUL_MR; r++) {
					__m128 sumi_f = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(sumi_arr[r])));
					__m128 summ_f = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(summ_arr[r])));
					__m128 d_v		= _mm_set1_ps(d[r]);
					__m128 dmin_v	= _mm_set1_ps(dmin[r]);
					__m128 val	= _mm_sub_ps(_mm_mul_ps(d_v, sumi_f), _mm_mul_ps(dmin_v, summ_f));
					acc_row[r]	= _mm_fmadd_ps(xd_vec, val, acc_row[r]);
				}
			}

			for (int r = 0; r < MATMUL_MR; r++) {
				float tmp[4];
				_mm_storeu_ps(tmp, acc_row[r]);
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

void matmul_f32_f32(const float *restrict w, const float *restrict x, float *restrict y, int n,
					int k) {
	const float *xp = x;
	float		*yp = y;
	const int	 mr = 8;
	int			 i	= 0;
	for (; i + mr <= n; i += mr) {
		__m256 acc[8];
		for (int r = 0; r < 8; r++)
			acc[r] = _mm256_setzero_ps();
		const float *rows[8];
		for (int r = 0; r < 8; r++)
			rows[r] = w + ((size_t)(i + r) * k);
		int j = 0;
		for (; j + 32 <= k; j += 32) {
			__builtin_prefetch(xp + j + 64, 0, 1);
			for (int r = 0; r < 8; r++)
				__builtin_prefetch(rows[r] + j + 64, 0, 1);
			__m256 x0 = _mm256_loadu_ps(xp + j);
			__m256 x1 = _mm256_loadu_ps(xp + j + 8);
			__m256 x2 = _mm256_loadu_ps(xp + j + 16);
			__m256 x3 = _mm256_loadu_ps(xp + j + 24);
			for (int r = 0; r < 8; r++) {
				acc[r] = _mm256_fmadd_ps(x0, _mm256_loadu_ps(rows[r] + j), acc[r]);
				acc[r] = _mm256_fmadd_ps(x1, _mm256_loadu_ps(rows[r] + j + 8), acc[r]);
				acc[r] = _mm256_fmadd_ps(x2, _mm256_loadu_ps(rows[r] + j + 16), acc[r]);
				acc[r] = _mm256_fmadd_ps(x3, _mm256_loadu_ps(rows[r] + j + 24), acc[r]);
			}
		}
		for (; j + 16 <= k; j += 16) {
			__m256 x0 = _mm256_loadu_ps(xp + j);
			__m256 x1 = _mm256_loadu_ps(xp + j + 8);
			for (int r = 0; r < 8; r++) {
				acc[r] = _mm256_fmadd_ps(x0, _mm256_loadu_ps(rows[r] + j), acc[r]);
				acc[r] = _mm256_fmadd_ps(x1, _mm256_loadu_ps(rows[r] + j + 8), acc[r]);
			}
		}
		for (int r = 0; r < 8; r++) {
			__m128 rem_acc = _mm_setzero_ps();
			int		j2	   = j;
			for (; j2 + 4 <= k; j2 += 4)
				rem_acc =
					_mm_fmadd_ps(_mm_loadu_ps(xp + j2), _mm_loadu_ps(rows[r] + j2), rem_acc);
			float s = vreduce_add_ps(acc[r]) + vreduce_add_ps_128(rem_acc);
			for (; j2 < k; j2++)
				s += rows[r][j2] * xp[j2];
			yp[i + r] = s;
		}
	}

	for (; i < n; i++) {
		const float *restrict wr = w + ((size_t)i * k);
		__m128 acc0				= _mm_setzero_ps();
		__m128 acc1				= _mm_setzero_ps();
		__m128 acc2				= _mm_setzero_ps();
		__m128 acc3				= _mm_setzero_ps();
		int		j				= 0;
		for (; j + 32 <= k; j += 32) {
			__builtin_prefetch(wr + j + 64, 0, 1);
			__builtin_prefetch(xp + j + 64, 0, 1);
			acc0 = _mm_fmadd_ps(_mm_loadu_ps(xp + j), _mm_loadu_ps(wr + j), acc0);
			acc1 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 4), _mm_loadu_ps(wr + j + 4), acc1);
			acc2 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 8), _mm_loadu_ps(wr + j + 8), acc2);
			acc3 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 12), _mm_loadu_ps(wr + j + 12), acc3);
			acc0 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 16), _mm_loadu_ps(wr + j + 16), acc0);
			acc1 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 20), _mm_loadu_ps(wr + j + 20), acc1);
			acc2 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 24), _mm_loadu_ps(wr + j + 24), acc2);
			acc3 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 28), _mm_loadu_ps(wr + j + 28), acc3);
		}
		for (; j + 16 <= k; j += 16) {
			acc0 = _mm_fmadd_ps(_mm_loadu_ps(xp + j), _mm_loadu_ps(wr + j), acc0);
			acc1 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 4), _mm_loadu_ps(wr + j + 4), acc1);
			acc2 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 8), _mm_loadu_ps(wr + j + 8), acc2);
			acc3 = _mm_fmadd_ps(_mm_loadu_ps(xp + j + 12), _mm_loadu_ps(wr + j + 12), acc3);
		}
		for (; j + 4 <= k; j += 4)
			acc0 = _mm_fmadd_ps(_mm_loadu_ps(xp + j), _mm_loadu_ps(wr + j), acc0);
		float s = vreduce_add_ps_128(acc0) + vreduce_add_ps_128(acc1) + vreduce_add_ps_128(acc2) +
				  vreduce_add_ps_128(acc3);
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

		const int NR = 2;
		int		  i	 = 0;
		for (; i + NR <= n; i += NR) {
			const float *rows[2];
			for (int r = 0; r < 2; r++)
				rows[r] = w + (size_t)(i + r) * k;

			__m256 acc[4][2];
			for (int t = 0; t < 4; t++)
				for (int r = 0; r < 2; r++)
					acc[t][r] = _mm256_setzero_ps();

			int j = 0;
			for (; j + 8 <= k; j += 8) {
				if (j + 32 < k) {
					__builtin_prefetch(xb[0] + j + 32, 0, 1);
					for (int r = 0; r < 2; r++)
						__builtin_prefetch(rows[r] + j + 32, 0, 1);
				}
				__m256 w0 = _mm256_loadu_ps(rows[0] + j);
				__m256 w1 = _mm256_loadu_ps(rows[1] + j);
				for (int t = 0; t < 4; t++) {
					__m256 xv = _mm256_loadu_ps(xb[t] + j);
					acc[t][0] = _mm256_fmadd_ps(xv, w0, acc[t][0]);
					acc[t][1] = _mm256_fmadd_ps(xv, w1, acc[t][1]);
				}
			}
			for (int t = 0; t < 4; t++) {
				float s[2];
				for (int r = 0; r < 2; r++)
					s[r] = vreduce_add_ps(acc[t][r]);
				for (int j2 = j; j2 < k; j2++) {
					float xv = xb[t][j2];
					for (int r = 0; r < 2; r++)
						s[r] += rows[r][j2] * xv;
				}
				for (int r = 0; r < 2; r++)
					y[(size_t)(mb + t) * y_row_stride + i + r] = s[r];
			}
		}
		for (; i < n; i++) {
			const float *wr = w + (size_t)i * k;
			for (int t = 0; t < 4; t++) {
				__m256 acc = _mm256_setzero_ps();
				int		j  = 0;
				for (; j + 8 <= k; j += 8)
					acc = _mm256_fmadd_ps(_mm256_loadu_ps(xb[t] + j), _mm256_loadu_ps(wr + j),
										  acc);
				float s = vreduce_add_ps(acc);
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

void matmul_bf16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					 int k) {
	const uint16_t *Wb = w;
	const int		mr = 8;
	int				i  = 0;

	for (; i + mr <= n; i += mr) {
		__m256 acc[8];
		for (int r = 0; r < 8; r++)
			acc[r] = _mm256_setzero_ps();

		const uint16_t *rows[8];
		for (int r = 0; r < 8; r++)
			rows[r] = Wb + (size_t)(i + r) * k;

		int j = 0;
		for (; j + 32 <= k; j += 32) {
			__builtin_prefetch(x + j + 64, 0, 1);
			for (int r = 0; r < 8; r++)
				__builtin_prefetch(rows[r] + j + 64, 0, 1);
			__m256 x0 = _mm256_loadu_ps(x + j);
			__m256 x1 = _mm256_loadu_ps(x + j + 8);
			__m256 x2 = _mm256_loadu_ps(x + j + 16);
			__m256 x3 = _mm256_loadu_ps(x + j + 24);
			for (int r = 0; r < 8; r++) {
				acc[r] = _mm256_fmadd_ps(x0, loadu_bf16x8_to_ps(rows[r] + j), acc[r]);
				acc[r] = _mm256_fmadd_ps(x1, loadu_bf16x8_to_ps(rows[r] + j + 8), acc[r]);
				acc[r] = _mm256_fmadd_ps(x2, loadu_bf16x8_to_ps(rows[r] + j + 16), acc[r]);
				acc[r] = _mm256_fmadd_ps(x3, loadu_bf16x8_to_ps(rows[r] + j + 24), acc[r]);
			}
		}
		for (; j + 16 <= k; j += 16) {
			__m256 x0 = _mm256_loadu_ps(x + j);
			__m256 x1 = _mm256_loadu_ps(x + j + 8);
			for (int r = 0; r < 8; r++) {
				acc[r] = _mm256_fmadd_ps(x0, loadu_bf16x8_to_ps(rows[r] + j), acc[r]);
				acc[r] = _mm256_fmadd_ps(x1, loadu_bf16x8_to_ps(rows[r] + j + 8), acc[r]);
			}
		}
		for (int r = 0; r < 8; r++) {
			int	  j2 = j;
			float s  = vreduce_add_ps(acc[r]);
			for (; j2 < k; j2++) {
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
		__m256 acc0				= _mm256_setzero_ps();
		__m256 acc1				= _mm256_setzero_ps();
		__m256 acc2				= _mm256_setzero_ps();
		__m256 acc3				= _mm256_setzero_ps();
		int		j				= 0;
		for (; j + 32 <= k; j += 32) {
			__builtin_prefetch(wr + j + 64, 0, 1);
			acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), loadu_bf16x8_to_ps(wr + j), acc0);
			acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j + 8),
								  loadu_bf16x8_to_ps(wr + j + 8), acc1);
			acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j + 16),
								  loadu_bf16x8_to_ps(wr + j + 16), acc2);
			acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j + 24),
								  loadu_bf16x8_to_ps(wr + j + 24), acc3);
		}
		for (; j + 8 <= k; j += 8)
			acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), loadu_bf16x8_to_ps(wr + j), acc0);
		float s = vreduce_add_ps(acc0) + vreduce_add_ps(acc1) + vreduce_add_ps(acc2) +
				  vreduce_add_ps(acc3);
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
}

void matmul_bf16_f32_batch(const void *restrict w, const float *restrict x, float *restrict y,
						   int n, int k, int m, int x_row_stride, int y_row_stride) {
	const uint16_t *Wb = w;
	const int		MB = 4;
	int				mb = 0;
	for (; mb + MB <= m; mb += MB) {
		const float *xb[4];
		for (int t = 0; t < MB; t++)
			xb[t] = x + (size_t)(mb + t) * x_row_stride;

		const int NR = 2;
		int		  i  = 0;
		for (; i + NR <= n; i += NR) {
			const uint16_t *rows[2];
			for (int r = 0; r < NR; r++)
				rows[r] = Wb + (size_t)(i + r) * k;

			__m256 acc[4][2];
			for (int t = 0; t < MB; t++)
				for (int r = 0; r < NR; r++)
					acc[t][r] = _mm256_setzero_ps();

			int j = 0;
			for (; j + 8 <= k; j += 8) {
				if (j + 32 < k) {
					__builtin_prefetch(xb[0] + j + 32, 0, 1);
					for (int r = 0; r < NR; r++)
						__builtin_prefetch(rows[r] + j + 32, 0, 1);
				}
				__m256 w0 = loadu_bf16x8_to_ps(rows[0] + j);
				__m256 w1 = loadu_bf16x8_to_ps(rows[1] + j);
				for (int t = 0; t < MB; t++) {
					__m256 xv = _mm256_loadu_ps(xb[t] + j);
					acc[t][0] = _mm256_fmadd_ps(xv, w0, acc[t][0]);
					acc[t][1] = _mm256_fmadd_ps(xv, w1, acc[t][1]);
				}
			}
			for (int t = 0; t < MB; t++) {
				float sums[2];
				for (int r = 0; r < NR; r++)
					sums[r] = vreduce_add_ps(acc[t][r]);
				for (int j2 = j; j2 < k; j2++) {
					float xv = xb[t][j2];
					for (int r = 0; r < NR; r++) {
						union {
							uint32_t u;
							float	 f;
						} v;
						v.u = (uint32_t)rows[r][j2] << 16;
						sums[r] += v.f * xv;
					}
				}
				for (int r = 0; r < NR; r++)
					y[(size_t)(mb + t) * y_row_stride + i + r] = sums[r];
			}
		}
		for (; i < n; i++) {
			const uint16_t *wr = Wb + (size_t)i * k;
			for (int t = 0; t < MB; t++) {
				__m256 acc = _mm256_setzero_ps();
				int		j	= 0;
				for (; j + 8 <= k; j += 8)
					acc = _mm256_fmadd_ps(_mm256_loadu_ps(xb[t] + j),
										 loadu_bf16x8_to_ps(wr + j), acc);
				float sum = vreduce_add_ps(acc);
				for (; j < k; j++) {
					union {
						uint32_t u;
						float	 f;
					} v;
					v.u = (uint32_t)wr[j] << 16;
					sum += v.f * xb[t][j];
				}
				y[(size_t)(mb + t) * y_row_stride + i] = sum;
			}
		}
	}
	for (; mb < m; mb++)
		matmul_bf16_f32(Wb, x + (size_t)mb * x_row_stride,
						 y + (size_t)mb * y_row_stride, n, k);
}

void matmul_f16_f32(const void *restrict w, const float *restrict x, float *restrict y, int n,
					int k) {
	const uint16_t *Wb = w;
	const int		mr = 8;
	int				i  = 0;
	for (; i + mr <= n; i += mr) {
		__m256 acc[8];
		for (int r = 0; r < 8; r++)
			acc[r] = _mm256_setzero_ps();

		const uint16_t *rows[8];
		for (int r = 0; r < 8; r++)
			rows[r] = Wb + (size_t)(i + r) * k;

		int j = 0;
		for (; j + 32 <= k; j += 32) {
			__builtin_prefetch(x + j + 64, 0, 1);
			for (int r = 0; r < 8; r++)
				__builtin_prefetch(rows[r] + j + 64, 0, 1);
			__m256 x0 = _mm256_loadu_ps(x + j);
			__m256 x1 = _mm256_loadu_ps(x + j + 8);
			__m256 x2 = _mm256_loadu_ps(x + j + 16);
			__m256 x3 = _mm256_loadu_ps(x + j + 24);
			for (int r = 0; r < 8; r++) {
				acc[r] = _mm256_fmadd_ps(x0, loadu_f16x8_to_ps_256(rows[r] + j), acc[r]);
				acc[r] = _mm256_fmadd_ps(x1, loadu_f16x8_to_ps_256(rows[r] + j + 8), acc[r]);
				acc[r] = _mm256_fmadd_ps(x2, loadu_f16x8_to_ps_256(rows[r] + j + 16), acc[r]);
				acc[r] = _mm256_fmadd_ps(x3, loadu_f16x8_to_ps_256(rows[r] + j + 24), acc[r]);
			}
		}
		for (; j + 16 <= k; j += 16) {
			__m256 x0 = _mm256_loadu_ps(x + j);
			__m256 x1 = _mm256_loadu_ps(x + j + 8);
			for (int r = 0; r < 8; r++) {
				acc[r] = _mm256_fmadd_ps(x0, loadu_f16x8_to_ps_256(rows[r] + j), acc[r]);
				acc[r] = _mm256_fmadd_ps(x1, loadu_f16x8_to_ps_256(rows[r] + j + 8), acc[r]);
			}
		}
		for (int r = 0; r < 8; r++) {
			__m128 rem_acc = _mm_setzero_ps();
			int		j2	   = j;
			for (; j2 + 4 <= k; j2 += 4)
				rem_acc = _mm_fmadd_ps(_mm_loadu_ps(x + j2),
									   loadu_f16x4_to_ps_128(rows[r] + j2), rem_acc);
			float s = vreduce_add_ps(acc[r]) + vreduce_add_ps_128(rem_acc);
			for (; j2 < k; j2++)
				s += f16_to_f32_fast(rows[r][j2]) * x[j2];
			y[i + r] = s;
		}
	}
	for (; i < n; i++) {
		const uint16_t *restrict wr = Wb + (size_t)i * k;
		__m256 acc0				= _mm256_setzero_ps();
		__m256 acc1				= _mm256_setzero_ps();
		__m256 acc2				= _mm256_setzero_ps();
		__m256 acc3				= _mm256_setzero_ps();
		int		j				= 0;
		for (; j + 32 <= k; j += 32) {
			__builtin_prefetch(wr + j + 64, 0, 1);
			__builtin_prefetch(x + j + 64, 0, 1);
			acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), loadu_f16x8_to_ps_256(wr + j), acc0);
			acc1 =
				_mm256_fmadd_ps(_mm256_loadu_ps(x + j + 8), loadu_f16x8_to_ps_256(wr + j + 8),
							   acc1);
			acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j + 16),
								   loadu_f16x8_to_ps_256(wr + j + 16), acc2);
			acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j + 24),
								   loadu_f16x8_to_ps_256(wr + j + 24), acc3);
		}
		for (; j + 16 <= k; j += 16) {
			acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), loadu_f16x8_to_ps_256(wr + j), acc0);
			acc1 =
				_mm256_fmadd_ps(_mm256_loadu_ps(x + j + 8), loadu_f16x8_to_ps_256(wr + j + 8),
							   acc1);
		}
		for (; j + 8 <= k; j += 8)
			acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), loadu_f16x8_to_ps_256(wr + j), acc0);
		float s = vreduce_add_ps(acc0) + vreduce_add_ps(acc1) + vreduce_add_ps(acc2) +
				  vreduce_add_ps(acc3);
		for (; j < k; j++)
			s += f16_to_f32_fast(wr[j]) * x[j];
		y[i] = s;
	}
}

void matmul_f16_f32_batch(const void *restrict w, const float *restrict x, float *restrict y,
						  int n, int k, int m, int x_row_stride, int y_row_stride) {
	const uint16_t *Wb = w;
	const int		MB = 4;
	int				mb = 0;
	for (; mb + MB <= m; mb += MB) {
		const float *xb[4];
		for (int t = 0; t < 4; t++)
			xb[t] = x + (size_t)(mb + t) * x_row_stride;

		const int NR = 2;
		int		  i	 = 0;
		for (; i + NR <= n; i += NR) {
			const uint16_t *rows[2];
			for (int r = 0; r < 2; r++)
				rows[r] = Wb + (size_t)(i + r) * k;

			__m256 acc[4][2];
			for (int t = 0; t < 4; t++)
				for (int r = 0; r < 2; r++)
					acc[t][r] = _mm256_setzero_ps();

			int j = 0;
			for (; j + 8 <= k; j += 8) {
				__builtin_prefetch(xb[0] + j + 32, 0, 1);
				for (int r = 0; r < 2; r++)
					__builtin_prefetch(rows[r] + j + 32, 0, 1);
				__m256 w0 = loadu_f16x8_to_ps_256(rows[0] + j);
				__m256 w1 = loadu_f16x8_to_ps_256(rows[1] + j);
				for (int t = 0; t < 4; t++) {
					__m256 xv = _mm256_loadu_ps(xb[t] + j);
					acc[t][0] = _mm256_fmadd_ps(xv, w0, acc[t][0]);
					acc[t][1] = _mm256_fmadd_ps(xv, w1, acc[t][1]);
				}
			}
			for (int t = 0; t < 4; t++) {
				float s[2];
				for (int r = 0; r < 2; r++)
					s[r] = vreduce_add_ps(acc[t][r]);
				for (int j2 = j; j2 < k; j2++) {
					float xv = xb[t][j2];
					for (int r = 0; r < 2; r++)
						s[r] += f16_to_f32_fast(rows[r][j2]) * xv;
				}
				for (int r = 0; r < 2; r++)
					y[(size_t)(mb + t) * y_row_stride + i + r] = s[r];
			}
		}
		for (; i < n; i++) {
			const uint16_t *wr = Wb + (size_t)i * k;
			for (int t = 0; t < 4; t++) {
				__m256 acc = _mm256_setzero_ps();
				int		j  = 0;
				for (; j + 8 <= k; j += 8)
					acc = _mm256_fmadd_ps(_mm256_loadu_ps(xb[t] + j),
										  loadu_f16x8_to_ps_256(wr + j), acc);
				float s = vreduce_add_ps(acc);
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

float dot_f32(const float *restrict a, const float *restrict b, int n) {
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();
	int		i	 = 0;
	for (; i + 32 <= n; i += 32) {
		__builtin_prefetch(a + i + 64, 0, 1);
		__builtin_prefetch(b + i + 64, 0, 1);
		acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
		acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
		acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
		acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
	}
	for (; i + 16 <= n; i += 16) {
		acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
		acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
	}
	__m128 acc4 = _mm_setzero_ps();
	for (; i + 4 <= n; i += 4)
		acc4 = _mm_fmadd_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i), acc4);
	float s = vreduce_add_ps(acc0) + vreduce_add_ps(acc1) + vreduce_add_ps(acc2) +
			  vreduce_add_ps(acc3) + vreduce_add_ps_128(acc4);
	for (; i < n; i++)
		s += a[i] * b[i];
	return s;
}

static inline float rmsnorm_sum_sq_avx(const float *x, int n) {
	__m128 ss_v = _mm_setzero_ps();
	int		i	 = 0;
	for (; i + 16 <= n; i += 16) {
		__m128 x0 = _mm_loadu_ps(x + i);
		__m128 x1 = _mm_loadu_ps(x + i + 4);
		__m128 x2 = _mm_loadu_ps(x + i + 8);
		__m128 x3 = _mm_loadu_ps(x + i + 12);
		ss_v	   = _mm_fmadd_ps(x0, x0, ss_v);
		ss_v	   = _mm_fmadd_ps(x1, x1, ss_v);
		ss_v	   = _mm_fmadd_ps(x2, x2, ss_v);
		ss_v	   = _mm_fmadd_ps(x3, x3, ss_v);
	}
	for (; i + 4 <= n; i += 4) {
		__m128 x0 = _mm_loadu_ps(x + i);
		ss_v	   = _mm_fmadd_ps(x0, x0, ss_v);
	}
	float ss = vreduce_add_ps_128(ss_v);
	for (; i < n; i++)
		ss += x[i] * x[i];
	return ss;
}

void rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
	float		ss		= rmsnorm_sum_sq_avx(x, n);
	float		scale	= 1.0f / sqrtf((ss / (float)n) + eps);
	__m128		scale_v = _mm_set1_ps(scale);
	int			i		= 0;
	for (; i + 16 <= n; i += 16) {
		_mm_storeu_ps(y + i, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x + i), scale_v),
										_mm_loadu_ps(w + i)));
		_mm_storeu_ps(y + i + 4, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x + i + 4), scale_v),
											_mm_loadu_ps(w + i + 4)));
		_mm_storeu_ps(y + i + 8, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x + i + 8), scale_v),
											_mm_loadu_ps(w + i + 8)));
		_mm_storeu_ps(y + i + 12, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x + i + 12), scale_v),
											 _mm_loadu_ps(w + i + 12)));
	}
	for (; i + 4 <= n; i += 4)
		_mm_storeu_ps(y + i,
					  _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x + i), scale_v), _mm_loadu_ps(w + i)));
	for (; i < n; i++)
		y[i] = x[i] * scale * w[i];
}

void rmsnorm_noweight(const float *x, float *y, int n, float eps) {
	float		ss		= rmsnorm_sum_sq_avx(x, n);
	float		scale	= 1.0f / sqrtf((ss / (float)n) + eps);
	__m128		scale_v = _mm_set1_ps(scale);
	int			i		= 0;
	for (; i + 16 <= n; i += 16) {
		_mm_storeu_ps(y + i, _mm_mul_ps(_mm_loadu_ps(x + i), scale_v));
		_mm_storeu_ps(y + i + 4, _mm_mul_ps(_mm_loadu_ps(x + i + 4), scale_v));
		_mm_storeu_ps(y + i + 8, _mm_mul_ps(_mm_loadu_ps(x + i + 8), scale_v));
		_mm_storeu_ps(y + i + 12, _mm_mul_ps(_mm_loadu_ps(x + i + 12), scale_v));
	}
	for (; i + 4 <= n; i += 4)
		_mm_storeu_ps(y + i, _mm_mul_ps(_mm_loadu_ps(x + i), scale_v));
	for (; i < n; i++)
		y[i] = x[i] * scale;
}

void rmsnorm_per_head(const float *x, const float *w, float *y, int n_heads, int head_dim,
					  float eps) {
	for (int h = 0; h < n_heads; h++) {
		const float *xh	 = x + ((size_t)h * head_dim);
		float		*yh	 = y + ((size_t)h * head_dim);
		float		ss	 = rmsnorm_sum_sq_avx(xh, head_dim);
		float		scale = 1.0f / sqrtf((ss / (float)head_dim) + eps);
		__m128		scale_v = _mm_set1_ps(scale);
		int			j	 = 0;
		for (; j + 16 <= head_dim; j += 16) {
			_mm_storeu_ps(yh + j, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(xh + j), scale_v),
											 _mm_loadu_ps(w + j)));
			_mm_storeu_ps(yh + j + 4, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(xh + j + 4), scale_v),
												 _mm_loadu_ps(w + j + 4)));
			_mm_storeu_ps(yh + j + 8, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(xh + j + 8), scale_v),
												 _mm_loadu_ps(w + j + 8)));
			_mm_storeu_ps(yh + j + 12, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(xh + j + 12), scale_v),
												  _mm_loadu_ps(w + j + 12)));
		}
		for (; j + 4 <= head_dim; j += 4)
			_mm_storeu_ps(yh + j, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(xh + j), scale_v),
											 _mm_loadu_ps(w + j)));
		for (; j < head_dim; j++)
			yh[j] = xh[j] * scale * w[j];
	}
}

void softmax_masked(float *restrict scores, int n_valid) {
	__m256 mx_v = _mm256_set1_ps(-INFINITY);
	int		i	= 0;
	for (; i + 32 <= n_valid; i += 32) {
		mx_v = _mm256_max_ps(mx_v, _mm256_loadu_ps(scores + i));
		mx_v = _mm256_max_ps(mx_v, _mm256_loadu_ps(scores + i + 8));
		mx_v = _mm256_max_ps(mx_v, _mm256_loadu_ps(scores + i + 16));
		mx_v = _mm256_max_ps(mx_v, _mm256_loadu_ps(scores + i + 24));
	}
	for (; i + 8 <= n_valid; i += 8)
		mx_v = _mm256_max_ps(mx_v, _mm256_loadu_ps(scores + i));
	float mx = vreduce_max_ps(mx_v);
	for (; i < n_valid; i++)
		if (scores[i] > mx)
			mx = scores[i];

	__m256 mx_vec = _mm256_set1_ps(mx);
	__m256 sum_v	= _mm256_setzero_ps();
	i				= 0;
	for (; i + 32 <= n_valid; i += 32) {
		__m256 s0 = _mm256_sub_ps(_mm256_loadu_ps(scores + i), mx_vec);
		__m256 s1 = _mm256_sub_ps(_mm256_loadu_ps(scores + i + 8), mx_vec);
		__m256 s2 = _mm256_sub_ps(_mm256_loadu_ps(scores + i + 16), mx_vec);
		__m256 s3 = _mm256_sub_ps(_mm256_loadu_ps(scores + i + 24), mx_vec);
		__m256 v0 = vexp_ps(s0);
		__m256 v1 = vexp_ps(s1);
		__m256 v2 = vexp_ps(s2);
		__m256 v3 = vexp_ps(s3);
		_mm256_storeu_ps(scores + i, v0);
		_mm256_storeu_ps(scores + i + 8, v1);
		_mm256_storeu_ps(scores + i + 16, v2);
		_mm256_storeu_ps(scores + i + 24, v3);
		sum_v = _mm256_add_ps(sum_v, v0);
		sum_v = _mm256_add_ps(sum_v, v1);
		sum_v = _mm256_add_ps(sum_v, v2);
		sum_v = _mm256_add_ps(sum_v, v3);
	}
	for (; i + 8 <= n_valid; i += 8) {
		__m256 s = _mm256_sub_ps(_mm256_loadu_ps(scores + i), mx_vec);
		__m256 v = vexp_ps(s);
		_mm256_storeu_ps(scores + i, v);
		sum_v = _mm256_add_ps(sum_v, v);
	}
	float sum = vreduce_add_ps(sum_v);
	for (; i < n_valid; i++) {
		scores[i] = expf(scores[i] - mx);
		sum += scores[i];
	}
	float		inv	  = 1.0f / sum;
	__m256		inv_v = _mm256_set1_ps(inv);
	i				  = 0;
	for (; i + 32 <= n_valid; i += 32) {
		_mm256_storeu_ps(scores + i, _mm256_mul_ps(_mm256_loadu_ps(scores + i), inv_v));
		_mm256_storeu_ps(scores + i + 8, _mm256_mul_ps(_mm256_loadu_ps(scores + i + 8), inv_v));
		_mm256_storeu_ps(scores + i + 16, _mm256_mul_ps(_mm256_loadu_ps(scores + i + 16), inv_v));
		_mm256_storeu_ps(scores + i + 24, _mm256_mul_ps(_mm256_loadu_ps(scores + i + 24), inv_v));
	}
	for (; i + 8 <= n_valid; i += 8)
		_mm256_storeu_ps(scores + i, _mm256_mul_ps(_mm256_loadu_ps(scores + i), inv_v));
	for (; i < n_valid; i++)
		scores[i] *= inv;
}

void moe_activate_silu(float *restrict act, const float *restrict gate, const float *restrict up,
					   int n, float gs, float us) {
	__m256 gs_v  = _mm256_set1_ps(gs);
	__m256 us_v  = _mm256_set1_ps(us);
	__m256 one_v = _mm256_set1_ps(1.0f);
	__m256 zero  = _mm256_setzero_ps();
	int		i	  = 0;
	for (; i + 32 <= n; i += 32) {
		__m256 g0 = _mm256_mul_ps(_mm256_loadu_ps(gate + i), gs_v);
		__m256 g1 = _mm256_mul_ps(_mm256_loadu_ps(gate + i + 8), gs_v);
		__m256 g2 = _mm256_mul_ps(_mm256_loadu_ps(gate + i + 16), gs_v);
		__m256 g3 = _mm256_mul_ps(_mm256_loadu_ps(gate + i + 24), gs_v);
		__m256 s0 = _mm256_div_ps(g0, _mm256_add_ps(one_v, vexp_ps(_mm256_sub_ps(zero, g0))));
		__m256 s1 = _mm256_div_ps(g1, _mm256_add_ps(one_v, vexp_ps(_mm256_sub_ps(zero, g1))));
		__m256 s2 = _mm256_div_ps(g2, _mm256_add_ps(one_v, vexp_ps(_mm256_sub_ps(zero, g2))));
		__m256 s3 = _mm256_div_ps(g3, _mm256_add_ps(one_v, vexp_ps(_mm256_sub_ps(zero, g3))));
		_mm256_storeu_ps(act + i, _mm256_mul_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(up + i), us_v)));
		_mm256_storeu_ps(act + i + 8,
						 _mm256_mul_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(up + i + 8), us_v)));
		_mm256_storeu_ps(act + i + 16,
						 _mm256_mul_ps(s2, _mm256_mul_ps(_mm256_loadu_ps(up + i + 16), us_v)));
		_mm256_storeu_ps(act + i + 24,
						 _mm256_mul_ps(s3, _mm256_mul_ps(_mm256_loadu_ps(up + i + 24), us_v)));
	}
	for (; i + 8 <= n; i += 8) {
		__m256 g = _mm256_mul_ps(_mm256_loadu_ps(gate + i), gs_v);
		__m256 s = _mm256_div_ps(g, _mm256_add_ps(one_v, vexp_ps(_mm256_sub_ps(zero, g))));
		_mm256_storeu_ps(act + i, _mm256_mul_ps(s, _mm256_mul_ps(_mm256_loadu_ps(up + i), us_v)));
	}
	for (; i < n; i++)
		act[i] = silu(gate[i] * gs) * (up[i] * us);
}

void moe_activate_gelu(float *restrict act, const float *restrict gate, const float *restrict up,
					   int n, float gs, float us) {
	__m256 gs_v   = _mm256_set1_ps(gs);
	__m256 us_v   = _mm256_set1_ps(us);
	__m256 half_v = _mm256_set1_ps(0.5f);
	__m256 one_v  = _mm256_set1_ps(1.0f);
	__m256 c_v	  = _mm256_set1_ps(0.7978845608028654f);
	__m256 k_v	  = _mm256_set1_ps(0.044715f);
	int		i	   = 0;
	for (; i + 32 <= n; i += 32) {
		__m256 g0	= _mm256_mul_ps(_mm256_loadu_ps(gate + i), gs_v);
		__m256 g1	= _mm256_mul_ps(_mm256_loadu_ps(gate + i + 8), gs_v);
		__m256 g2	= _mm256_mul_ps(_mm256_loadu_ps(gate + i + 16), gs_v);
		__m256 g3	= _mm256_mul_ps(_mm256_loadu_ps(gate + i + 24), gs_v);
		__m256 x3_0 = _mm256_mul_ps(g0, _mm256_mul_ps(g0, g0));
		__m256 x3_1 = _mm256_mul_ps(g1, _mm256_mul_ps(g1, g1));
		__m256 x3_2 = _mm256_mul_ps(g2, _mm256_mul_ps(g2, g2));
		__m256 x3_3 = _mm256_mul_ps(g3, _mm256_mul_ps(g3, g3));
		__m256 in0	= _mm256_mul_ps(c_v, _mm256_add_ps(g0, _mm256_mul_ps(k_v, x3_0)));
		__m256 in1	= _mm256_mul_ps(c_v, _mm256_add_ps(g1, _mm256_mul_ps(k_v, x3_1)));
		__m256 in2	= _mm256_mul_ps(c_v, _mm256_add_ps(g2, _mm256_mul_ps(k_v, x3_2)));
		__m256 in3	= _mm256_mul_ps(c_v, _mm256_add_ps(g3, _mm256_mul_ps(k_v, x3_3)));
		__m256 t0	= vtanh_ps(in0);
		__m256 t1	= vtanh_ps(in1);
		__m256 t2	= vtanh_ps(in2);
		__m256 t3	= vtanh_ps(in3);
		__m256 s0	= _mm256_mul_ps(half_v, _mm256_mul_ps(g0, _mm256_add_ps(one_v, t0)));
		__m256 s1	= _mm256_mul_ps(half_v, _mm256_mul_ps(g1, _mm256_add_ps(one_v, t1)));
		__m256 s2	= _mm256_mul_ps(half_v, _mm256_mul_ps(g2, _mm256_add_ps(one_v, t2)));
		__m256 s3	= _mm256_mul_ps(half_v, _mm256_mul_ps(g3, _mm256_add_ps(one_v, t3)));
		_mm256_storeu_ps(act + i, _mm256_mul_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(up + i), us_v)));
		_mm256_storeu_ps(act + i + 8,
						 _mm256_mul_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(up + i + 8), us_v)));
		_mm256_storeu_ps(act + i + 16,
						 _mm256_mul_ps(s2, _mm256_mul_ps(_mm256_loadu_ps(up + i + 16), us_v)));
		_mm256_storeu_ps(act + i + 24,
						 _mm256_mul_ps(s3, _mm256_mul_ps(_mm256_loadu_ps(up + i + 24), us_v)));
	}
	for (; i + 8 <= n; i += 8) {
		__m256 g	= _mm256_mul_ps(_mm256_loadu_ps(gate + i), gs_v);
		__m256 x3	= _mm256_mul_ps(g, _mm256_mul_ps(g, g));
		__m256 in	= _mm256_mul_ps(c_v, _mm256_add_ps(g, _mm256_mul_ps(k_v, x3)));
		__m256 t	= vtanh_ps(in);
		__m256 s	= _mm256_mul_ps(half_v, _mm256_mul_ps(g, _mm256_add_ps(one_v, t)));
		_mm256_storeu_ps(act + i, _mm256_mul_ps(s, _mm256_mul_ps(_mm256_loadu_ps(up + i), us_v)));
	}
	for (; i < n; i++)
		act[i] = gelu_tanh(gate[i] * gs) * (up[i] * us);
}
