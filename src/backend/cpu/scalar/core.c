#include "backend/backend.h"
#include "backend/cpu/common.h"
#include "backend/cpu/quants.h"
#include "model.h"
#include "moe/moe_stream.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { HC_SUPPORTED = 4 };

typedef struct {
	const float *qf;
	float		*outf;
	const float *k_b;
	const float *v_b;
	const float *layer_c;
	const float *k_pe_rot_all;
	const float *rope_cos_base;
	const float *rope_sin_base;
	int			 qk_head;
	int			 qk_rope;
	int			 qk_nope;
	int			 v_head;
	int			 kv_lora;
	int			 total_dim;
	int			 pos;
	float		 scale;
} cpu_attn_mla_job;

typedef struct {
	float		*base;
	int			 n_heads, head_dim, row_stride, pos_start;
	float		 theta;
	int			 neox;
	const float *freq_factors;
	const float *cos_base, *sin_base;
} cpu_rope_ext_batch_job;

typedef struct {
	float		*state;
	const float *q, *k, *v, *g, *beta;
	float		*out;
	int			 head_dim;
} kda_job;

typedef struct {
	const float *q_abs;
	const float *cache;
	const float *v_b;
	float		*out;
	float		*scratch;
	int			 n_pos, kv_lora, v_head, scratch_stride;
	float		 scale;
} mla_job;

static const float KDA_L2_EPS = 1e-6f;

void detect_features(char *buf, size_t cap) {
	snprintf(buf, cap, "generic");
}

static void cpu_rope_one(float *v, int n_heads, int head_dim, const float *rope_cos,
						 const float *rope_sin, int neox) {
	int half = head_dim / 2;
	if (neox) {
		rope_rotate_neox(v, n_heads, head_dim, head_dim, rope_cos, rope_sin);
		return;
	}
	for (int h = 0; h < n_heads; h++) {
		float *vh = v + ((size_t)h * head_dim);
		for (int j = 0; j < half; j++) {
			float c			= rope_cos[j];
			float s			= rope_sin[j];
			float v0		= vh[2 * j];
			float v1		= vh[(2 * j) + 1];
			vh[2 * j]		= (v0 * c) - (v1 * s);
			vh[(2 * j) + 1] = (v0 * s) + (v1 * c);
		}
	}
}

static status_code cpu_rope_one_factored(cpu_priv *p, float *v, int n_heads, int head_dim, int pos,
										 float theta, const float *freq_factors, int neox) {
	float	   *cs;
	status_code st = cpu_rope_cs_cached(p, head_dim, pos, theta, freq_factors, &cs);
	if (st != OK)
		return st;
	rope_rotate_rows(v, n_heads, head_dim, head_dim / 2, cs, neox);
	return OK;
}

status_code cpu_rope(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
					 const float *rope_cos_base, const float *rope_sin_base) {
	return cpu_rope_batch(self, vec, n_heads, head_dim, pos, rope_cos_base, rope_sin_base, 1);
}

status_code cpu_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
						int head_dim, int pos, const float *rope_cos_base,
						const float *rope_sin_base) {
	return cpu_rope_qk_batch(self, q, k, n_heads, n_kv_heads, head_dim, pos, rope_cos_base,
							 rope_sin_base, 1);
}

static void cpu_rope_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rope_batch_job *j	 = ctx;
	int					half = j->head_dim / 2;
	for (int row = begin; row < end; row++) {
		int			 pos = j->pos_start + row;
		const float *rc	 = j->rope_cos_base + ((size_t)pos * half);
		const float *rs	 = j->rope_sin_base + ((size_t)pos * half);
		cpu_rope_one(j->vec + ((size_t)row * j->n_heads * j->head_dim), j->n_heads, j->head_dim, rc,
					 rs, j->rope_neox);
	}
}

status_code cpu_rope_batch(backend *self, buffer *vec, int n_heads, int head_dim, int pos_start,
						   const float *rope_cos_base, const float *rope_sin_base, int m) {
	cpu_priv		  *p   = self->priv;
	cpu_rope_batch_job job = {.vec			 = (float *)cpu_ptr(vec),
							  .n_heads		 = n_heads,
							  .head_dim		 = head_dim,
							  .pos_start	 = pos_start,
							  .rope_cos_base = rope_cos_base,
							  .rope_sin_base = rope_sin_base,
							  .rope_neox	 = self->rope_neox};
	cpu_run_batch(p->pool, m, cpu_rope_batch_chunk, &job);
	return OK;
}

static void cpu_rope_qk_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rope_qk_batch_job *j	= ctx;
	int					   half = j->head_dim / 2;
	for (int row = begin; row < end; row++) {
		int			 pos = j->pos_start + row;
		const float *rc	 = j->rope_cos_base + ((size_t)pos * half);
		const float *rs	 = j->rope_sin_base + ((size_t)pos * half);
		cpu_rope_one(j->q + ((size_t)row * j->n_heads * j->head_dim), j->n_heads, j->head_dim, rc,
					 rs, j->rope_neox);
		cpu_rope_one(j->k + ((size_t)row * j->n_kv_heads * j->head_dim), j->n_kv_heads, j->head_dim,
					 rc, rs, j->rope_neox);
	}
}

status_code cpu_rope_qk_batch(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
							  int head_dim, int pos_start, const float *rope_cos_base,
							  const float *rope_sin_base, int m) {
	cpu_priv			 *p	  = self->priv;
	cpu_rope_qk_batch_job job = {.q				= (float *)cpu_ptr(q),
								 .k				= (float *)cpu_ptr(k),
								 .n_heads		= n_heads,
								 .n_kv_heads	= n_kv_heads,
								 .head_dim		= head_dim,
								 .pos_start		= pos_start,
								 .rope_cos_base = rope_cos_base,
								 .rope_sin_base = rope_sin_base,
								 .rope_neox		= self->rope_neox};
	cpu_run_batch(p->pool, m, cpu_rope_qk_batch_chunk, &job);
	return OK;
}

status_code cpu_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
						 const float *rope_cos_base, const float *rope_sin_base,
						 const float *freq_factors) {
	cpu_priv *p = self->priv;
	if (freq_factors) {
		status_code err = cpu_rope_one_factored(p, (float *)cpu_ptr(vec), n_heads, head_dim, pos,
												self->rope_theta, freq_factors, self->rope_neox);
		if (err)
			return err;
	} else {
		int			 half	  = head_dim / 2;
		const float *rope_cos = rope_cos_base + ((size_t)pos * half);
		const float *rope_sin = rope_sin_base + ((size_t)pos * half);
		cpu_rope_one((float *)cpu_ptr(vec), n_heads, head_dim, rope_cos, rope_sin, self->rope_neox);
	}
	return OK;
}

static void cpu_rope_ext_batch_row(float *v, int n_heads, int half, const float *cs, int neox) {
	rope_rotate_rows(v, n_heads, half * 2, half, cs, neox);
}

static void cpu_rope_ext_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rope_ext_batch_job *j			= ctx;
	int						half		= j->head_dim / 2;
	double					theta_scale = pow((double)j->theta, -2.0 / (double)j->head_dim);

	for (int r = begin; r < end; r++) {
		int	   pos = j->pos_start + r;
		float  cs[512];
		float *csp = cs;
		if (half > 256)
			csp = xmalloc((size_t)half * 2 * sizeof(float));
		if (j->freq_factors)
			cpu_rope_cs_fill(csp, half, pos, theta_scale, j->freq_factors);
		float *v = j->base + (size_t)r * j->row_stride;
		if (j->freq_factors) {
			cpu_rope_ext_batch_row(v, j->n_heads, half, csp, j->neox);
		} else {
			const float *rc = j->cos_base + ((size_t)pos * half);
			const float *rs = j->sin_base + ((size_t)pos * half);
			for (int h = 0; h < j->n_heads; h++) {
				float *vh = v + ((size_t)h * j->head_dim);
				for (int jj = 0; jj < half; jj++) {
					float c = rc[jj];
					float s = rs[jj];
					if (j->neox) {
						float v0	  = vh[jj];
						float v1	  = vh[jj + half];
						vh[jj]		  = (v0 * c) - (v1 * s);
						vh[jj + half] = (v0 * s) + (v1 * c);
					} else {
						float v0		 = vh[2 * jj];
						float v1		 = vh[(2 * jj) + 1];
						vh[2 * jj]		 = (v0 * c) - (v1 * s);
						vh[(2 * jj) + 1] = (v0 * s) + (v1 * c);
					}
				}
			}
		}
		if (csp != cs)
			free(csp);
	}
}

status_code cpu_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim, int pos_start,
							   const float *rope_cos_base, const float *rope_sin_base,
							   const float *freq_factors, int m) {
	cpu_priv *p = self->priv;
	if (m <= 1)
		return cpu_rope_ext(self, vec, n_heads, head_dim, pos_start, rope_cos_base, rope_sin_base,
							freq_factors);

	cpu_rope_ext_batch_job job = {.base			= (float *)cpu_ptr(vec),
								  .n_heads		= n_heads,
								  .head_dim		= head_dim,
								  .row_stride	= n_heads * head_dim,
								  .pos_start	= pos_start,
								  .theta		= self->rope_theta,
								  .neox			= self->rope_neox,
								  .freq_factors = freq_factors,
								  .cos_base		= rope_cos_base,
								  .sin_base		= rope_sin_base};

	if (p->pool) {
		tpool_parallel_for(p->pool, m, 1, cpu_rope_ext_batch_chunk, &job);
		return OK;
	}
	cpu_rope_ext_batch_chunk(0, m, -1, &job);
	return OK;
}

float dot8(const float *restrict a, const float *restrict b, int head_dim) {
	float a0	  = 0;
	float a1	  = 0;
	float a2	  = 0;
	float a3	  = 0;
	int	  d		  = 0;
	int	  hd_main = head_dim - (head_dim % 8);
	for (; d < hd_main; d += 8) {
		a0 += (a[d] * b[d]) + (a[d + 4] * b[d + 4]);
		a1 += (a[d + 1] * b[d + 1]) + (a[d + 5] * b[d + 5]);
		a2 += (a[d + 2] * b[d + 2]) + (a[d + 6] * b[d + 6]);
		a3 += (a[d + 3] * b[d + 3]) + (a[d + 7] * b[d + 7]);
	}
	for (; d < head_dim; d++)
		a0 += a[d] * b[d];
	return (a0 + a1) + (a2 + a3);
}

float dot8_f16(const float *restrict a, const uint16_t *restrict b, int head_dim) {
	float a0	  = 0;
	float a1	  = 0;
	float a2	  = 0;
	float a3	  = 0;
	int	  d		  = 0;
	int	  hd_main = head_dim - (head_dim % 8);
	for (; d < hd_main; d += 8) {
		float b0 = f16_to_f32(b[d]);
		float b1 = f16_to_f32(b[d + 1]);
		float b2 = f16_to_f32(b[d + 2]);
		float b3 = f16_to_f32(b[d + 3]);
		float b4 = f16_to_f32(b[d + 4]);
		float b5 = f16_to_f32(b[d + 5]);
		float b6 = f16_to_f32(b[d + 6]);
		float b7 = f16_to_f32(b[d + 7]);
		a0 += (a[d] * b0) + (a[d + 4] * b4);
		a1 += (a[d + 1] * b1) + (a[d + 5] * b5);
		a2 += (a[d + 2] * b2) + (a[d + 6] * b6);
		a3 += (a[d + 3] * b3) + (a[d + 7] * b7);
	}
	for (; d < head_dim; d++)
		a0 += a[d] * f16_to_f32(b[d]);
	return (a0 + a1) + (a2 + a3);
}

static inline float dot8_q8_0(const float *restrict a, const uint8_t *restrict block_ptr,
							  int head_dim) {
	int	  n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	float sum	   = 0.0f;
	for (int b = 0; b < n_blocks; b++) {
		const q8_0_block *blk = (const q8_0_block *)(block_ptr + ((size_t)b * KV_Q8_0_BLOCK_BYTES));
		float			  d	  = f16_to_f32(blk->d);
		int				  base = b * KV_Q8_0_BLOCK;
		int				  n	   = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;
		float partial = 0.0f;
		for (int j = 0; j < n; j++)
			partial += a[base + j] * (float)blk->qs[j];
		sum += partial * d;
	}
	return sum;
}

static inline void accum_v_q8_0(float *restrict out_h, const uint8_t *restrict block_ptr,
								float weight, int head_dim) {
	int n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	for (int b = 0; b < n_blocks; b++) {
		const q8_0_block *blk = (const q8_0_block *)(block_ptr + ((size_t)b * KV_Q8_0_BLOCK_BYTES));
		float			  d	  = f16_to_f32(blk->d) * weight;
		int				  base = b * KV_Q8_0_BLOCK;
		int				  n	   = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;
		for (int j = 0; j < n; j++)
			out_h[base + j] = fmaf((float)blk->qs[j], d, out_h[base + j]);
	}
}

static void cpu_attention_inner_q8_0(const uint8_t *k_slice, const uint8_t *v_slice,
									 size_t kv_stride_bytes, const float *qh, float *out_h,
									 int head_dim, int n_pos, float scale, int flash_attn,
									 float *scores) {
	if (flash_attn) {
		float M					= -INFINITY;
		float S					= 0.0f;
		float VKQ[HEAD_DIM_MAX] = {0};
		for (int t = 0; t < n_pos; t++) {
			const uint8_t *kt = k_slice + ((size_t)t * kv_stride_bytes);
			float		   ss = dot8_q8_0(qh, kt, head_dim) * scale;
			float		   ms = 1.0f;
			float		   vs = 1.0f;
			if (ss > M) {
				float Mold = M;
				M		   = ss;
				ms		   = expf(Mold - M);
				for (int d = 0; d < head_dim; d++)
					VKQ[d] *= ms;
			} else {
				vs = expf(ss - M);
			}
			const uint8_t *vt = v_slice + ((size_t)t * kv_stride_bytes);
			accum_v_q8_0(VKQ, vt, vs, head_dim);
			S = (S * ms) + vs;
		}
		float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
		for (int d = 0; d < head_dim; d++)
			out_h[d] = VKQ[d] * S_inv;
		return;
	}

	for (int t = 0; t < n_pos; t++) {
		const uint8_t *kt = k_slice + ((size_t)t * kv_stride_bytes);
		scores[t]		  = dot8_q8_0(qh, kt, head_dim) * scale;
	}
	softmax_masked(scores, n_pos);

	memset(out_h, 0, (size_t)head_dim * sizeof(float));
	for (int t = 0; t < n_pos; t++) {
		const uint8_t *vt = v_slice + ((size_t)t * kv_stride_bytes);
		accum_v_q8_0(out_h, vt, scores[t], head_dim);
	}
}

static void cpu_attention_inner(const uint16_t *k_slice, const uint16_t *v_slice, int kv_stride,
								const float *qh, float *out_h, int head_dim, int n_pos, float scale,
								int flash_attn, float *scores) {
	if (flash_attn) {
		float M					= -INFINITY;
		float S					= 0.0f;
		float VKQ[HEAD_DIM_MAX] = {0};
		for (int t = 0; t < n_pos; t++) {
			const uint16_t *kt = k_slice + ((size_t)t * kv_stride);
			float			ss = dot8_f16(qh, kt, head_dim) * scale;
			float			ms = 1.0f;
			float			vs = 1.0f;
			if (ss > M) {
				float Mold = M;
				M		   = ss;
				ms		   = expf(Mold - M);
				for (int d = 0; d < head_dim; d++)
					VKQ[d] *= ms;
			} else {
				vs = expf(ss - M);
			}
			const uint16_t *vt = v_slice + ((size_t)t * kv_stride);
			for (int d = 0; d < head_dim; d++)
				VKQ[d] = fmaf(f16_to_f32(vt[d]), vs, VKQ[d]);
			S = (S * ms) + vs;
		}
		float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
		for (int d = 0; d < head_dim; d++)
			out_h[d] = VKQ[d] * S_inv;
		return;
	}

	for (int t = 0; t < n_pos; t++) {
		const uint16_t *kt = k_slice + ((size_t)t * kv_stride);
		scores[t]		   = dot8_f16(qh, kt, head_dim) * scale;
	}
	softmax_masked(scores, n_pos);

	if (n_pos > 0) {
		float			sv0 = scores[0];
		const uint16_t *vt0 = v_slice;
		for (int d = 0; d < head_dim; d++)
			out_h[d] = sv0 * f16_to_f32(vt0[d]);
	}
	for (int t = 1; t < n_pos; t++) {
		float			sv = scores[t];
		const uint16_t *vt = v_slice + ((size_t)t * kv_stride);
		for (int d = 0; d < head_dim; d++)
			out_h[d] += sv * f16_to_f32(vt[d]);
	}
}

static void cpu_attn_head_chunk(int begin, int end, int tid, void *ctx) {
	cpu_attn_job *j		 = ctx;
	float		 *scores = j->flash_attn ? NULL : cpu_grow_scores(j->p, tid, j->n_pos);
	for (int h = begin; h < end; h++) {
		int			 kvh   = h / j->n_groups;
		const float *qh	   = j->qf + ((size_t)h * j->head_dim);
		float		*out_h = j->outf + ((size_t)h * j->head_dim);
		if (j->kv_quant == KV_QUANT_Q8_0) {
			const uint8_t *k_slice = (const uint8_t *)j->kl_base + ((size_t)kvh * j->kvh_stride);
			const uint8_t *v_slice = (const uint8_t *)j->vl_base + ((size_t)kvh * j->kvh_stride);
			cpu_attention_inner_q8_0(k_slice, v_slice, (size_t)j->hd_stride, qh, out_h, j->head_dim,
									 j->n_pos, j->scale, j->flash_attn, scores);
		} else {
			const uint16_t *k_slice = j->kl_base + ((size_t)kvh * j->kvh_stride);
			const uint16_t *v_slice = j->vl_base + ((size_t)kvh * j->kvh_stride);
			cpu_attention_inner(k_slice, v_slice, j->hd_stride, qh, out_h, j->head_dim, j->n_pos,
								j->scale, j->flash_attn, scores);
		}
	}
}

static void cpu_attn_kvh_chunk(int begin, int end, int tid, void *ctx) {
	cpu_attn_job *j		 = ctx;
	float		 *scores = j->flash_attn ? NULL : cpu_grow_scores(j->p, tid, j->n_pos);
	for (int kvh = begin; kvh < end; kvh++) {
		int h_start = kvh * j->n_groups;
		int h_end	= h_start + j->n_groups;
		if (h_end > j->n_heads)
			h_end = j->n_heads;
		const void *k_slice;
		const void *v_slice;
		if (j->kv_quant == KV_QUANT_Q8_0) {
			k_slice = (const uint8_t *)j->kl_base + ((size_t)kvh * j->kvh_stride);
			v_slice = (const uint8_t *)j->vl_base + ((size_t)kvh * j->kvh_stride);
		} else {
			k_slice = j->kl_base + ((size_t)kvh * j->kvh_stride);
			v_slice = j->vl_base + ((size_t)kvh * j->kvh_stride);
		}
		for (int h = h_start; h < h_end; h++) {
			const float *qh	   = j->qf + ((size_t)h * j->head_dim);
			float		*out_h = j->outf + ((size_t)h * j->head_dim);
			if (j->kv_quant == KV_QUANT_Q8_0) {
				cpu_attention_inner_q8_0(k_slice, v_slice, (size_t)j->hd_stride, qh, out_h,
										 j->head_dim, j->n_pos, j->scale, j->flash_attn, scores);
			} else {
				cpu_attention_inner(k_slice, v_slice, j->hd_stride, qh, out_h, j->head_dim,
									j->n_pos, j->scale, j->flash_attn, scores);
			}
		}
	}
}

status_code cpu_attention_impl(backend *self, const buffer *q, const buffer *k_cache,
							   const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
							   int n_kv_heads, int head_dim, int n_ctx, int flash_attn, float scale,
							   int sliding_window, int n_kv_heads_active) {
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

	int			 n_groups = (n_heads + n_active - 1) / n_active;
	const float *qf		  = (const float *)cpu_ptr(q);
	float		*outf	  = (float *)cpu_ptr(out);

	int hd_stride_elems = head_dim;

	int attn_start = 0;
	int n_pos	   = pos + 1;
	if (sliding_window > 0 && n_pos > sliding_window) {
		attn_start = n_pos - sliding_window;
		n_pos	   = sliding_window;
	}
	cpu_kv_window_to_slots(p, layer, &attn_start, &n_pos);

	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);

	if (p->kv_quant == KV_QUANT_Q8_0) {
		size_t n_blocks	   = ((size_t)hd_stride_elems + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		size_t elem_stride = n_blocks * KV_Q8_0_BLOCK_BYTES;
		size_t kvh_stride  = cpu_kv_layer_rows(p, layer, n_ctx) * elem_stride;

		size_t		   layer_stride = (size_t)n_kv_heads * kvh_stride;
		size_t		   layer_base	= cpu_kv_layer_base(p, layer, layer_stride, 1);
		const uint8_t *kl_base =
			(const uint8_t *)k_cache->handle + layer_base + ((size_t)attn_start * elem_stride);
		const uint8_t *vl_base =
			(const uint8_t *)v_cache->handle + layer_base + ((size_t)attn_start * elem_stride);

		if (can_recurse && p->pool && p->thread_scratch && n_heads > 1 &&
			(size_t)n_heads * (size_t)n_pos * (size_t)head_dim >= 4096) {
			cpu_attn_job job = {.kl_base	= (const uint16_t *)kl_base,
								.vl_base	= (const uint16_t *)vl_base,
								.qf			= qf,
								.outf		= outf,
								.n_groups	= n_groups,
								.head_dim	= head_dim,
								.hd_stride	= (int)elem_stride,
								.n_pos		= n_pos,
								.flash_attn = flash_attn,
								.n_heads	= n_heads,
								.scale		= scale,
								.kvh_stride = kvh_stride,
								.p			= p,
								.kv_quant	= KV_QUANT_Q8_0};
			if (n_groups > 1)
				tpool_parallel_for(p->pool, n_active, 1, cpu_attn_kvh_chunk, &job);
			else
				tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_head_chunk, &job);
			return OK;
		}

		float *scores = flash_attn ? NULL : cpu_grow_scores(p, cur_tid, n_pos);
		for (int h = 0; h < n_heads; h++) {
			int			   kvh	   = h / n_groups;
			const uint8_t *k_slice = kl_base + ((size_t)kvh * kvh_stride);
			const uint8_t *v_slice = vl_base + ((size_t)kvh * kvh_stride);
			const float	  *qh	   = qf + ((size_t)h * head_dim);
			float		  *out_h   = outf + ((size_t)h * head_dim);
			cpu_attention_inner_q8_0(k_slice, v_slice, elem_stride, qh, out_h, head_dim, n_pos,
									 scale, flash_attn, scores);
		}
		return OK;
	}

	int	   hd_stride	= hd_stride_elems;
	size_t kvh_stride	= cpu_kv_layer_rows(p, layer, n_ctx) * hd_stride;
	size_t layer_stride = (size_t)n_kv_heads * kvh_stride;
	size_t layer_base	= cpu_kv_layer_base(p, layer, layer_stride, sizeof(uint16_t));

	uint16_t *kl_base = (uint16_t *)k_cache->handle + layer_base;
	uint16_t *vl_base = (uint16_t *)v_cache->handle + layer_base;
	kl_base += (size_t)attn_start * hd_stride;
	vl_base += (size_t)attn_start * hd_stride;

	if (can_recurse && p->pool && p->thread_scratch && n_heads > 1 &&
		(size_t)n_heads * (size_t)n_pos * (size_t)head_dim >= 4096) {
		cpu_attn_job job = {.kl_base	= kl_base,
							.vl_base	= vl_base,
							.qf			= qf,
							.outf		= outf,
							.n_groups	= n_groups,
							.head_dim	= head_dim,
							.hd_stride	= hd_stride,
							.n_pos		= n_pos,
							.flash_attn = flash_attn,
							.n_heads	= n_heads,
							.scale		= scale,
							.kvh_stride = kvh_stride,
							.p			= p,
							.kv_quant	= KV_QUANT_F16};
		if (n_groups > 1)
			tpool_parallel_for(p->pool, n_active, 1, cpu_attn_kvh_chunk, &job);
		else
			tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_head_chunk, &job);
		return OK;
	}

	float *scores = flash_attn ? NULL : cpu_grow_scores(p, cur_tid, n_pos);
	for (int h = 0; h < n_heads; h++) {
		int		  kvh	  = h / n_groups;
		uint16_t *k_slice = kl_base + ((size_t)kvh * kvh_stride);
		uint16_t *v_slice = vl_base + ((size_t)kvh * kvh_stride);

		const float *qh	   = qf + ((size_t)h * head_dim);
		float		*out_h = outf + ((size_t)h * head_dim);
		cpu_attention_inner(k_slice, v_slice, hd_stride, qh, out_h, head_dim, n_pos, scale,
							flash_attn, scores);
	}
	return OK;
}

status_code cpu_attention(backend *self, const buffer *q, const buffer *k_cache,
						  const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
						  int n_kv_heads, int head_dim, int n_ctx, int flash_attn, float scale,
						  int n_kv_heads_active) {
	return cpu_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							  head_dim, n_ctx, flash_attn, scale, 0, n_kv_heads_active);
}

status_code cpu_attention_swa(backend *self, const buffer *q, const buffer *k_cache,
							  const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
							  int n_kv_heads, int head_dim, int n_ctx, int flash_attn, float scale,
							  int sliding_window, int n_kv_heads_active) {
	return cpu_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							  head_dim, n_ctx, flash_attn, scale, sliding_window,
							  n_kv_heads_active);
}

static void cpu_attn_batch_chunk(int begin, int end, int tid, void *ctx) {
	cpu_attn_batch_job *j			= ctx;
	int					need_scores = j->pos_start + j->m;
	if (j->kv_rows && (size_t)need_scores > j->kv_rows)
		need_scores = (int)j->kv_rows;
	float *scores = j->flash_attn ? NULL : cpu_grow_scores(j->p, tid, need_scores);

	for (int idx = begin; idx < end; idx++) {
		int dispatch_row = idx / j->n_heads;
		int h			 = idx % j->n_heads;

		int row = j->bitrev_perm ? j->bitrev_perm[dispatch_row] : dispatch_row;
		if (row >= j->m)
			continue;

		int pos		   = j->pos_start + row;
		int n_pos	   = j->n_keys > 0 ? j->n_keys : pos + 1;
		int attn_start = 0;
		if (j->sliding_window > 0 && n_pos > j->sliding_window) {
			attn_start = n_pos - j->sliding_window;
			n_pos	   = j->sliding_window;
		}
		cpu_kv_window_to_slots_base(j->kv_base, &attn_start, &n_pos);

		int			 kvh   = h / j->n_groups;
		const float *qh	   = j->qf + ((((size_t)row * j->n_heads) + h) * j->head_dim);
		float		*out_h = j->outf + ((((size_t)row * j->n_heads) + h) * j->head_dim);
		if (j->kv_quant == KV_QUANT_Q8_0) {
			const uint8_t *k_slice = (const uint8_t *)j->kl_base + ((size_t)kvh * j->kvh_stride) +
									 ((size_t)attn_start * (size_t)j->hd_stride);
			const uint8_t *v_slice = (const uint8_t *)j->vl_base + ((size_t)kvh * j->kvh_stride) +
									 ((size_t)attn_start * (size_t)j->hd_stride);
			cpu_attention_inner_q8_0(k_slice, v_slice, (size_t)j->hd_stride, qh, out_h, j->head_dim,
									 n_pos, j->scale, j->flash_attn, scores);
		} else {
			const uint16_t *k_slice =
				j->kl_base + ((size_t)kvh * j->kvh_stride) + ((size_t)attn_start * j->hd_stride);
			const uint16_t *v_slice =
				j->vl_base + ((size_t)kvh * j->kvh_stride) + ((size_t)attn_start * j->hd_stride);
			cpu_attention_inner(k_slice, v_slice, j->hd_stride, qh, out_h, j->head_dim, n_pos,
								j->scale, j->flash_attn, scores);
		}
	}
}

static status_code cpu_attention_batch_impl(backend *self, const buffer *q, const buffer *k_cache,
											const buffer *v_cache, buffer *out, int layer,
											int pos_start, int n_heads, int n_kv_heads,
											int head_dim, int n_ctx, int flash_attn, float scale,
											int sliding_window, int n_kv_heads_active, int m) {
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	int		  n_groups = (n_heads + n_active - 1) / n_active;

	size_t			  rows = cpu_kv_layer_rows(p, layer, n_ctx);
	size_t			  base = p->kv_base_pos ? p->kv_base_pos[layer] : 0;
	cpu_kv_layer_view view =
		cpu_kv_layer_view_get(p, k_cache, v_cache, layer, n_kv_heads, head_dim, rows);

	int use_bitrev = (m >= ATTN_BITREV_MIN_M);
	int m_pow2	   = 1;
	while (m_pow2 < m)
		m_pow2 <<= 1;

	int *bitrev_perm = NULL;
	if (use_bitrev)
		bitrev_perm = cpu_bitrev_perm_get(p, m_pow2);

	cpu_attn_batch_job job = {.kl_base		  = view.k_base,
							  .vl_base		  = view.v_base,
							  .qf			  = (const float *)cpu_ptr(q),
							  .outf			  = (float *)cpu_ptr(out),
							  .n_groups		  = n_groups,
							  .head_dim		  = head_dim,
							  .hd_stride	  = view.hd_stride,
							  .n_heads		  = n_heads,
							  .pos_start	  = pos_start,
							  .m			  = m,
							  .sliding_window = sliding_window,
							  .flash_attn	  = flash_attn,
							  .scale		  = scale,
							  .kvh_stride	  = view.kvh_stride,
							  .p			  = p,
							  .bitrev_perm	  = bitrev_perm,
							  .kv_quant		  = p->kv_quant,
							  .kv_rows		  = rows,
							  .kv_base		  = base};

	int total		= use_bitrev ? (n_heads * m_pow2) : (n_heads * m);
	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);
	if (can_recurse && p->pool && p->thread_scratch && total >= 2) {
		tpool_parallel_for(p->pool, total, 1, cpu_attn_batch_chunk, &job);
	} else {
		cpu_attn_batch_chunk(0, total, 0, &job);
	}

	return OK;
}

status_code cpu_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
								const buffer *v_cache, buffer *out, int layer, int pos_start,
								int n_heads, int n_kv_heads, int head_dim, int n_ctx,
								int flash_attn, float scale, int n_kv_heads_active, int m) {
	return cpu_attention_batch_impl(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
									n_kv_heads, head_dim, n_ctx, flash_attn, scale, 0,
									n_kv_heads_active, m);
}

status_code cpu_attention_swa_batch(backend *self, const buffer *q, const buffer *k_cache,
									const buffer *v_cache, buffer *out, int layer, int pos_start,
									int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									int flash_attn, float scale, int sliding_window,
									int n_kv_heads_active, int m) {
	return cpu_attention_batch_impl(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
									n_kv_heads, head_dim, n_ctx, flash_attn, scale, sliding_window,
									n_kv_heads_active, m);
}

status_code cpu_add_inplace(backend *self, buffer *x, const buffer *y, int n) {
	return cpu_add_batch(self, x, y, n, 1);
}

status_code cpu_ffn_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
							 int n) {
	return cpu_ffn_activate_batch(self, gate, up, out, n, 0, 1);
}

status_code cpu_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up, buffer *out,
								int n, int activation) {
	return cpu_ffn_activate_batch(self, gate, up, out, n, activation, 1);
}

static void cpu_add_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_add_batch_job *j = ctx;
	for (int row = begin; row < end; row++) {
		float		*xr = j->x + ((size_t)row * j->n);
		const float *yr = j->y + ((size_t)row * j->n);
		for (int i = 0; i < j->n; i++)
			xr[i] += yr[i];
	}
}

status_code cpu_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
	cpu_priv		 *p	  = self->priv;
	cpu_add_batch_job job = {.x = cpu_ptr(x), .y = cpu_ptr(y), .n = n};
	cpu_run_batch(p->pool, m, cpu_add_batch_chunk, &job);
	return OK;
}

static void cpu_ffn_act_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_act_batch_job *j = ctx;
	for (int row = begin; row < end; row++) {
		const float *g = j->g + ((size_t)row * j->n);
		const float *u = j->u + ((size_t)row * j->n);
		float		*o = j->o + ((size_t)row * j->n);
		if (j->activation == 1) {
			for (int i = 0; i < j->n; i++)
				o[i] = gelu_tanh(g[i]) * u[i];
		} else {
			for (int i = 0; i < j->n; i++) {
				float gv = g[i];
				o[i]	 = gv / (1.0f + expf(-gv)) * u[i];
			}
		}
	}
}

status_code cpu_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n, int activation, int m) {
	cpu_priv			 *p	  = self->priv;
	cpu_ffn_act_batch_job job = {
		.g = cpu_ptr(gate), .u = cpu_ptr(up), .o = cpu_ptr(out), .n = n, .activation = activation};
	cpu_run_batch(p->pool, m, cpu_ffn_act_batch_chunk, &job);
	return OK;
}

status_code cpu_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx) {
	(void)self;
	*out_idx = cpu_argmax_f32(cpu_ptr(logits), n);
	return OK;
}

static void cpu_attn_mla_head_compute(const cpu_attn_mla_job *j, int h) {
	const int	half_rope = j->qk_rope / 2;
	const int	n_pos	  = j->pos + 1;
	const float scale	  = j->scale;

	float q_h[HEAD_DIM_MAX];
	float q_absorbed[HEAD_DIM_MAX];
	float VKQ_latent[HEAD_DIM_MAX];

	const float *q_src = j->qf + ((size_t)h * j->qk_head);
	for (int d = 0; d < j->qk_head; d++)
		q_h[d] = q_src[d];
	const float *q_rope_cos = j->rope_cos_base + ((size_t)j->pos * half_rope);
	const float *q_rope_sin = j->rope_sin_base + ((size_t)j->pos * half_rope);
	for (int k = 0; k < half_rope; k++) {
		float c						  = q_rope_cos[k];
		float s						  = q_rope_sin[k];
		float a						  = q_h[j->qk_nope + (2 * k)];
		float b						  = q_h[j->qk_nope + (2 * k) + 1];
		q_h[j->qk_nope + (2 * k)]	  = (a * c) - (b * s);
		q_h[j->qk_nope + (2 * k) + 1] = (a * s) + (b * c);
	}

	const float *k_b_h = j->k_b + ((size_t)h * j->qk_nope * j->kv_lora);
	const float *v_b_h = j->v_b + ((size_t)h * j->kv_lora * j->v_head);

	for (int i = 0; i < j->kv_lora; i++) {
		float		 acc = 0.0f;
		const float *row = k_b_h + ((size_t)i * j->qk_nope);
		for (int d = 0; d < j->qk_nope; d++)
			acc += row[d] * q_h[d];
		q_absorbed[i] = acc;
	}
	const float *q_rope_part = q_h + j->qk_nope;

	float M = -INFINITY;
	float S = 0.0f;
	memset(VKQ_latent, 0, (size_t)j->kv_lora * sizeof(float));
	for (int t = 0; t < n_pos; t++) {
		const float *slot	  = j->layer_c + ((size_t)t * j->total_dim);
		const float *latent	  = slot;
		const float *k_pe_rot = j->k_pe_rot_all + ((size_t)t * j->qk_rope);

		float score = 0.0f;
		for (int i = 0; i < j->kv_lora; i++)
			score += latent[i] * q_absorbed[i];
		for (int d = 0; d < j->qk_rope; d++)
			score += q_rope_part[d] * k_pe_rot[d];
		score *= scale;

		float ms = 1.0f;
		float vs = 1.0f;
		if (score > M) {
			float Mold = M;
			M		   = score;
			ms		   = expf(Mold - M);
			for (int i = 0; i < j->kv_lora; i++)
				VKQ_latent[i] *= ms;
		} else {
			vs = expf(score - M);
		}
		for (int i = 0; i < j->kv_lora; i++)
			VKQ_latent[i] = fmaf(latent[i], vs, VKQ_latent[i]);
		S = (S * ms) + vs;
	}
	float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
	float VKQ[HEAD_DIM_MAX];
	for (int d = 0; d < j->v_head; d++) {
		float		 acc = 0.0f;
		const float *row = v_b_h + ((size_t)d * j->kv_lora);
		for (int i = 0; i < j->kv_lora; i++)
			acc += VKQ_latent[i] * row[i];
		VKQ[d] = acc * S_inv;
	}
	for (int d = 0; d < j->v_head; d++)
		j->outf[((size_t)h * j->v_head) + d] = VKQ[d];
}

static void cpu_attn_mla_head_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_attn_mla_job *j = ctx;
	for (int h = begin; h < end; h++)
		cpu_attn_mla_head_compute(j, h);
}

status_code cpu_attention_mla(backend *self, const buffer *q, const buffer *kv_cache,
							  const buffer *k_b_w, const buffer *v_b_w, buffer *out, int layer,
							  int pos, int n_heads, int qk_head, int qk_rope, int qk_nope,
							  int v_head, int kv_lora, int n_ctx, const float *rope_cos_base,
							  const float *rope_sin_base, uint32_t k_b_type, uint32_t v_b_type,
							  float scale) {
	(void)k_b_type;
	(void)v_b_type;
	cpu_priv	*p			  = self->priv;
	int			 total_dim	  = kv_lora + qk_rope;
	const float *qf			  = (const float *)cpu_ptr(q);
	float		*outf		  = (float *)cpu_ptr(out);
	const float *cache		  = (const float *)cpu_ptr(kv_cache);
	const float *k_b		  = (const float *)cpu_ptr(k_b_w);
	const float *v_b		  = (const float *)cpu_ptr(v_b_w);
	size_t		 layer_stride = (size_t)total_dim * (size_t)n_ctx;
	const float *layer_c	  = cache + ((size_t)layer * layer_stride);

	int n_pos	  = pos + 1;
	int half_rope = qk_rope / 2;

	size_t		krot_need = (size_t)n_pos * qk_rope * sizeof(float);
	status_code grow_st	  = cpu_buf_grow((void **)&p->mla_krot.buf, &p->mla_krot.cap, krot_need, 1);
	if (grow_st != OK)
		return grow_st;
	float *k_pe_rot_all = p->mla_krot.buf;

	int cached_n_pos = p->mla_krot.n_pos_cached;
	int can_incr =
		(cached_n_pos == n_pos - 1) && (p->mla_krot.layer_cached == layer) &&
		(p->mla_krot.layer_c_cached == layer_c) && (p->mla_krot.total_dim_cached == total_dim) &&
		(p->mla_krot.kv_lora_cached == kv_lora) && (p->mla_krot.qk_rope_cached == qk_rope) &&
		(p->mla_krot.half_rope_cached == half_rope);

	if (can_incr && n_pos >= 1) {
		int			 t			= n_pos - 1;
		const float *slot		= layer_c + ((size_t)t * total_dim);
		const float *k_pe		= slot + kv_lora;
		const float *t_rope_cos = rope_cos_base + ((size_t)t * half_rope);
		const float *t_rope_sin = rope_sin_base + ((size_t)t * half_rope);
		float		*dst		= k_pe_rot_all + ((size_t)t * qk_rope);
		for (int j = 0; j < half_rope; j++) {
			float c			 = t_rope_cos[j];
			float s			 = t_rope_sin[j];
			float a			 = k_pe[2 * j];
			float b			 = k_pe[(2 * j) + 1];
			dst[2 * j]		 = (a * c) - (b * s);
			dst[(2 * j) + 1] = (a * s) + (b * c);
		}
		p->mla_krot.n_pos_cached = n_pos;
	} else {
		for (int t = 0; t < n_pos; t++) {
			const float *slot		= layer_c + ((size_t)t * total_dim);
			const float *k_pe		= slot + kv_lora;
			const float *t_rope_cos = rope_cos_base + ((size_t)t * half_rope);
			const float *t_rope_sin = rope_sin_base + ((size_t)t * half_rope);
			float		*dst		= k_pe_rot_all + ((size_t)t * qk_rope);
			for (int j = 0; j < half_rope; j++) {
				float c			 = t_rope_cos[j];
				float s			 = t_rope_sin[j];
				float a			 = k_pe[2 * j];
				float b			 = k_pe[(2 * j) + 1];
				dst[2 * j]		 = (a * c) - (b * s);
				dst[(2 * j) + 1] = (a * s) + (b * c);
			}
		}
		p->mla_krot.layer_c_cached	 = layer_c;
		p->mla_krot.layer_cached	 = layer;
		p->mla_krot.total_dim_cached = total_dim;
		p->mla_krot.kv_lora_cached	 = kv_lora;
		p->mla_krot.qk_rope_cached	 = qk_rope;
		p->mla_krot.half_rope_cached = half_rope;
		p->mla_krot.n_pos_cached	 = n_pos;
	}

	for (int i = 0; i < n_heads * v_head; i++)
		outf[i] = 0.0f;

	cpu_attn_mla_job job = {
		.qf			   = qf,
		.outf		   = outf,
		.k_b		   = k_b,
		.v_b		   = v_b,
		.layer_c	   = layer_c,
		.k_pe_rot_all  = k_pe_rot_all,
		.rope_cos_base = rope_cos_base,
		.rope_sin_base = rope_sin_base,
		.qk_head	   = qk_head,
		.qk_rope	   = qk_rope,
		.qk_nope	   = qk_nope,
		.v_head		   = v_head,
		.kv_lora	   = kv_lora,
		.total_dim	   = total_dim,
		.pos		   = pos,
		.scale		   = scale,
	};

	if (p->pool && p->thread_scratch && n_heads >= 2) {
		tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_mla_head_chunk, &job);
		return OK;
	}

	for (int h = 0; h < n_heads; h++)
		cpu_attn_mla_head_compute(&job, h);

	return OK;
}

static void cpu_rmsnorm_add_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rmsnorm_add_batch_job *j = ctx;
	for (int row = begin; row < end; row++) {
		const float *xr	   = j->x + ((size_t)row * j->n);
		const float *rr	   = j->residual + ((size_t)row * j->n);
		float		*yr	   = j->y + ((size_t)row * j->n);
		float		 ss	   = rmsnorm_sum_sq(xr, j->n);
		float		 scale = 1.0f / sqrtf((ss / (float)j->n) + j->eps);
		float		 t;
		for (int i = 0; i < j->n; i++) {
			t	  = xr[i] * scale * j->w[i] + rr[i];
			yr[i] = t * j->out_scale;
		}
	}
}

status_code cpu_rmsnorm_add_batch(backend *self, const buffer *x, const buffer *w,
								  const buffer *residual, buffer *y, int n, float eps,
								  float out_scale, int m) {
	cpu_priv				 *p	  = self->priv;
	cpu_rmsnorm_add_batch_job job = {.x			= cpu_ptr(x),
									 .w			= cpu_ptr(w),
									 .residual	= cpu_ptr(residual),
									 .y			= cpu_ptr(y),
									 .n			= n,
									 .eps		= eps,
									 .out_scale = out_scale};
	cpu_run_batch(p->pool, m, cpu_rmsnorm_add_batch_chunk, &job);
	return OK;
}

void cpu_softcap_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_softcap_job *j		 = ctx;
	float			*x		 = j->x;
	float			 inv_cap = j->inv_cap;
	float			 cap	 = j->cap;
	for (int i = begin; i < end; i++)
		x[i] = cap * tanhf(x[i] * inv_cap);
}

void cpu_split_qgate_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_split_qgate_job *j			  = ctx;
	int					 q_out		  = j->n_heads * j->head_dim;
	int					 mixed_stride = 2 * q_out;
	for (int row = begin; row < end; row++) {
		const float *src = j->mixed + (size_t)row * mixed_stride;
		float		*qd	 = j->q + (size_t)row * q_out;
		float		*gd	 = j->gate + (size_t)row * q_out;
		for (int h = 0; h < j->n_heads; h++, src += 2 * j->head_dim) {
			for (int jj = 0; jj < j->head_dim; jj++) {
				qd[jj] = src[jj];
				gd[jj] = src[jj + j->head_dim];
			}
			qd += j->head_dim;
			gd += j->head_dim;
		}
	}
}

void cpu_attn_output_gate_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_attn_output_gate_job *j = ctx;
	int						  n = j->n;
	for (int row = begin; row < end; row++) {
		float		*o = j->out + (size_t)row * n;
		const float *g = j->gate + (size_t)row * n;
		for (int i = 0; i < n; i++)
			o[i] *= sigmoidf(g[i]);
	}
}

void cpu_partial_rope_qk_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_partial_rope_qk_job *j = ctx;
	for (int row = begin; row < end; row++) {
		const float *cosv = j->cos_base + (size_t)(j->pos0 + row) * j->half;
		const float *sinv = j->sin_base + (size_t)(j->pos0 + row) * j->half;
		rope_rotate_neox(j->q + (size_t)row * j->qn, j->n_heads, j->head_dim, j->rope_dim, cosv,
						 sinv);
		rope_rotate_neox(j->k + (size_t)row * j->kn, j->n_kv_heads, j->head_dim, j->rope_dim, cosv,
						 sinv);
	}
}

void cpu_ffn_down_act_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_down_act_args *a = ctx;
	const float *restrict g	 = a->g;
	const float *restrict u	 = a->u;
	float *restrict o		 = a->o;
	if (a->activation == 1) {
		for (int i = begin; i < end; i++)
			o[i] = gelu_tanh(g[i]) * u[i];
	} else {
		for (int i = begin; i < end; i++) {
			float gv = g[i];
			o[i]	 = gv / (1.0f + expf(-gv)) * u[i];
		}
	}
}

int32_t cpu_argmax_f32(const float *logits, int vocab) {
	if (vocab <= 0)
		return 0;
	int32_t best  = 0;
	float	bestv = logits[0];
	for (int i = 1; i < vocab; i++)
		if (logits[i] > bestv) {
			bestv = logits[i];
			best  = i;
		}
	return best;
}

static status_code matmul_vec(backend *self, const backend_weight *w, const float *x, float *y,
							  int n, int k) {
	buffer		xb = buffer_host_view(self, (void *)x, (size_t)k * sizeof(float));
	buffer		yb = buffer_host_view(self, y, (size_t)n * sizeof(float));
	status_code st = self->matmul(self, w->buf, w->type, &xb, &yb, n, k);
	ensure_sync(self);
	return st;
}

static void rmsnorm_inplace(float *x, const float *w, int n, float eps) {
	float ss = 0.0f;
	for (int i = 0; i < n; i++)
		ss += x[i] * x[i];
	float inv = 1.0f / sqrtf(ss / (float)n + eps);
	for (int i = 0; i < n; i++)
		x[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

status_code cpu_hc_init(backend *self, const hc_init_host_desc *d) {
	(void)self;
	for (int h = 0; h < d->n_hc; h++)
		memcpy(d->hc + (size_t)h * d->dim, d->x, (size_t)d->dim * sizeof(float));
	return OK;
}

status_code cpu_hc_final(backend *self, const hc_final_host_desc *d) {
	(void)self;
	for (int i = 0; i < d->dim; i++) {
		float acc = 0.0f;
		for (int h = 0; h < d->n_hc; h++)
			acc += d->hc[(size_t)h * d->dim + i];
		d->x[i] = acc / (float)d->n_hc;
	}
	return OK;
}

static void hc_sinkhorn(float comb[HC_SUPPORTED * HC_SUPPORTED], int iters, float eps) {
	enum { HC = HC_SUPPORTED };
	for (int src = 0; src < HC; src++) {
		float *col = comb + src * HC;
		float  mx  = col[0];
		for (int d = 1; d < HC; d++)
			mx = fmaxf(mx, col[d]);
		float sum = 0.0f;
		for (int d = 0; d < HC; d++) {
			col[d] = expf(col[d] - mx);
			sum += col[d];
		}
		for (int d = 0; d < HC; d++)
			col[d] = col[d] / sum + eps;
	}
	for (int it = 0; it < iters; it++) {
		if (it > 0) {
			for (int src = 0; src < HC; src++) {
				float sum = eps;
				for (int d = 0; d < HC; d++)
					sum += comb[src * HC + d];
				for (int d = 0; d < HC; d++)
					comb[src * HC + d] /= sum;
			}
		}
		for (int d = 0; d < HC; d++) {
			float sum = eps;
			for (int src = 0; src < HC; src++)
				sum += comb[src * HC + d];
			for (int src = 0; src < HC; src++)
				comb[src * HC + d] /= sum;
		}
	}
}

status_code cpu_hc_pre(backend *self, const hc_pre_host_desc *d) {
	const int hc = d->n_hc, dim = d->dim, flat = hc * dim;
	if (hc != HC_SUPPORTED)
		return ERR_UNSUPPORTED;

	float *normed = d->ws;
	float *mixes  = normed + flat;
	memcpy(normed, d->stream, (size_t)flat * sizeof(float));
	rmsnorm_inplace(normed, NULL, flat, d->norm_eps);

	status_code st = matmul_vec(self, &d->fn, normed, mixes, hc_mix_dim(hc), flat);
	if (st != OK)
		return st;

	float pre[HC_SUPPORTED];
	for (int h = 0; h < hc; h++) {
		pre[h]	   = sigmoidf(mixes[h] * d->scale[0] + d->base[h]) + d->eps;
		d->coef[h] = 2.0f * sigmoidf(mixes[hc + h] * d->scale[1] + d->base[hc + h]);
	}
	float *comb = d->coef + hc;
	for (int j = 0; j < hc * hc; j++)
		comb[j] = mixes[2 * hc + j] * d->scale[2] + d->base[2 * hc + j];
	hc_sinkhorn(comb, d->sinkhorn_iters, d->eps);

	for (int i = 0; i < dim; i++) {
		float acc = 0.0f;
		for (int h = 0; h < hc; h++)
			acc += pre[h] * d->stream[(size_t)h * dim + i];
		d->x[i] = acc;
	}
	return OK;
}

status_code cpu_hc_post(backend *self, const hc_post_host_desc *d) {
	(void)self;
	const int	 hc	  = d->n_hc;
	const int	 dim  = d->dim;
	const float *post = d->coef;
	const float *comb = d->coef + hc;
	for (int dst_h = 0; dst_h < hc; dst_h++) {
		float *dst = d->hc_next + (size_t)dst_h * dim;
		for (int i = 0; i < dim; i++) {
			float acc = d->sub[i] * post[dst_h];
			for (int src_h = 0; src_h < hc; src_h++)
				acc += d->hc_cur[(size_t)src_h * dim + i] * comb[dst_h + src_h * hc];
			dst[i] = acc;
		}
	}
	return OK;
}

static void l2_normalize_heads(float *x, int n_heads, int head_dim) {
	for (int h = 0; h < n_heads; h++) {
		float *v  = x + (size_t)h * head_dim;
		float  ss = 0.0f;
		for (int i = 0; i < head_dim; i++)
			ss += v[i] * v[i];
		float inv = 1.0f / sqrtf(ss + KDA_L2_EPS);
		for (int i = 0; i < head_dim; i++)
			v[i] *= inv;
	}
}

static void causal_conv_silu(float *state, const float *w, float *x, int channels, int kernel) {
	const int hist = kernel - 1;
	for (int c = 0; c < channels; c++) {
		float		*h	 = state + (size_t)c * hist;
		const float *wc	 = w + (size_t)c * kernel;
		float		 sum = x[c] * wc[kernel - 1];
		for (int k = 0; k < hist; k++)
			sum += h[k] * wc[k];
		memmove(h, h + 1, (size_t)(hist - 1) * sizeof(float));
		h[hist - 1] = x[c];
		x[c]		= silu(sum);
	}
}

static void kda_heads(int begin, int end, int tid, void *arg) {
	(void)tid;
	kda_job	   *j	  = arg;
	const int	S	  = j->head_dim;
	const float scale = 1.0f / sqrtf((float)S);
	float		delta[512];
	for (int h = begin; h < end; h++) {
		float		*M = j->state + (size_t)h * S * S;
		const float *q = j->q + (size_t)h * S;
		const float *k = j->k + (size_t)h * S;
		const float *v = j->v + (size_t)h * S;
		const float *g = j->g + (size_t)h * S;
		float		*o = j->out + (size_t)h * S;
		float		 b = j->beta[h];
		float		 decay[512];
		for (int i = 0; i < S; i++)
			decay[i] = expf(g[i]);
		for (int r = 0; r < S; r++) {
			float *row = M + (size_t)r * S;
			float  dot = 0.0f;
			for (int i = 0; i < S; i++) {
				row[i] *= decay[i];
				dot += row[i] * k[i];
			}
			delta[r] = (v[r] - dot) * b;
		}
		for (int r = 0; r < S; r++) {
			float *row = M + (size_t)r * S;
			float  dot = 0.0f;
			for (int i = 0; i < S; i++) {
				row[i] += delta[r] * k[i];
				dot += row[i] * q[i];
			}
			o[r] = dot * scale;
		}
	}
}

status_code cpu_kda(backend *self, const kda_host_desc *d) {
	const int dim = d->dim, H = d->n_heads, S = d->head_dim, D = H * S;
	const int K = d->conv_kernel, hist = K - 1;

	float *q	= d->ws;
	float *k	= q + D;
	float *v	= k + D;
	float *g	= v + D;
	float *o	= g + D;
	float *g2	= o + D;
	float *tmp	= g2 + D;
	float *beta = tmp + S;

	status_code st = matmul_vec(self, &d->wq, d->x, q, D, dim);
	if (st == OK)
		st = matmul_vec(self, &d->wk, d->x, k, D, dim);
	if (st == OK)
		st = matmul_vec(self, &d->wv, d->x, v, D, dim);
	if (st != OK)
		return st;
	causal_conv_silu(d->conv_state, d->conv_q, q, D, K);
	causal_conv_silu(d->conv_state + (size_t)D * hist, d->conv_k, k, D, K);
	causal_conv_silu(d->conv_state + (size_t)2 * D * hist, d->conv_v, v, D, K);
	l2_normalize_heads(q, H, S);
	l2_normalize_heads(k, H, S);

	st = matmul_vec(self, &d->f_a, d->x, tmp, S, dim);
	if (st == OK)
		st = matmul_vec(self, &d->f_b, tmp, g, D, S);
	if (st == OK)
		st = matmul_vec(self, &d->beta, d->x, beta, H, dim);
	if (st != OK)
		return st;
	const float lb = d->gate_lower_bound;
	for (int h = 0; h < H; h++) {
		for (int i = 0; i < S; i++) {
			int	  idx = h * S + i;
			float t	  = g[idx] + d->dt_bias[idx];
			if (lb > -INFINITY) {
				g[idx] = lb * sigmoidf(-(t * d->ssm_a[h]));
			} else {
				float sp = t > 20.0f ? t : log1pf(expf(t));
				g[idx]	 = sp * d->ssm_a[h];
			}
		}
		beta[h] = sigmoidf(beta[h]);
	}

	kda_job job	 = {d->state, q, k, v, g, beta, o, S};
	tpool  *pool = self->get_pool ? self->get_pool(self) : NULL;
	if (pool && tpool_current_tid() < 0)
		tpool_parallel_for(pool, H, 1, kda_heads, &job);
	else
		kda_heads(0, H, -1, &job);

	st = matmul_vec(self, &d->g_a, d->x, tmp, S, dim);
	if (st == OK)
		st = matmul_vec(self, &d->g_b, tmp, g2, D, S);
	if (st != OK)
		return st;
	for (int h = 0; h < H; h++) {
		float *oh = o + (size_t)h * S;
		rmsnorm_inplace(oh, d->out_norm, S, d->norm_eps);
		for (int i = 0; i < S; i++)
			oh[i] *= sigmoidf(g2[h * S + i]);
	}
	return matmul_vec(self, &d->wo, o, d->out, dim, D);
}

static void mla_heads(int begin, int end, int tid, void *arg) {
	mla_job	 *j	  = arg;
	float	 *sc  = j->scratch + (size_t)(tid < 0 ? 0 : tid) * j->scratch_stride;
	float	 *lat = sc + j->n_pos;
	const int R	  = j->kv_lora;
	for (int h = begin; h < end; h++) {
		const float *qa = j->q_abs + (size_t)h * R;
		float		 mx = -INFINITY;
		for (int t = 0; t < j->n_pos; t++) {
			const float *kv = j->cache + (size_t)t * R;
			float		 s	= 0.0f;
			for (int r = 0; r < R; r++)
				s += qa[r] * kv[r];
			sc[t] = s * j->scale;
			mx	  = fmaxf(mx, sc[t]);
		}
		float sum = 0.0f;
		for (int t = 0; t < j->n_pos; t++) {
			sc[t] = expf(sc[t] - mx);
			sum += sc[t];
		}
		memset(lat, 0, (size_t)R * sizeof(float));
		for (int t = 0; t < j->n_pos; t++) {
			const float *kv = j->cache + (size_t)t * R;
			float		 p	= sc[t] / sum;
			for (int r = 0; r < R; r++)
				lat[r] += p * kv[r];
		}
		const float *vb = j->v_b + (size_t)h * j->v_head * R;
		float		*o	= j->out + (size_t)h * j->v_head;
		for (int d = 0; d < j->v_head; d++) {
			const float *row = vb + (size_t)d * R;
			float		 s	 = 0.0f;
			for (int r = 0; r < R; r++)
				s += row[r] * lat[r];
			o[d] = s;
		}
	}
}

status_code cpu_mla_absorbed(backend *self, const mla_absorbed_host_desc *d) {
	const int dim = d->dim, H = d->n_heads, R = d->kv_lora;
	const int qk = d->qk_head, vh = d->v_head, ql = d->q_lora;
	const int n_pos			 = d->pos + 1;
	const int scratch_stride = n_pos + R;

	float *qa	 = d->ws;
	float *q	 = qa + ql;
	float *q_abs = q + (size_t)H * qk;
	float *kv	 = q_abs + (size_t)H * R;
	float *ho	 = kv + R;
	float *scr	 = ho + (size_t)H * vh;

	status_code st = matmul_vec(self, &d->q_a, d->x, qa, ql, dim);
	if (st != OK)
		return st;
	rmsnorm_inplace(qa, d->q_a_norm, ql, d->norm_eps);
	st = matmul_vec(self, &d->q_b, qa, q, H * qk, ql);
	if (st == OK)
		st = matmul_vec(self, &d->kv_a, d->x, kv, R, dim);
	if (st != OK)
		return st;
	rmsnorm_inplace(kv, d->kv_a_norm, R, d->norm_eps);
	memcpy(d->latent_cache + (size_t)d->pos * R, kv, (size_t)R * sizeof(float));

	for (int h = 0; h < H; h++) {
		const float *qh = q + (size_t)h * qk;
		const float *kh = d->k_b + (size_t)h * R * qk;
		for (int r = 0; r < R; r++) {
			const float *row = kh + (size_t)r * qk;
			float		 s	 = 0.0f;
			for (int i = 0; i < qk; i++)
				s += row[i] * qh[i];
			q_abs[(size_t)h * R + r] = s;
		}
	}

	mla_job job	 = {q_abs,			d->latent_cache,		d->v_b, ho, scr, n_pos, R, vh,
					scratch_stride, 1.0f / sqrtf((float)qk)};
	tpool  *pool = self->get_pool ? self->get_pool(self) : NULL;
	if (pool && tpool_current_tid() < 0 && tpool_n_threads(pool) <= d->n_threads + 1)
		tpool_parallel_for(pool, H, 1, mla_heads, &job);
	else
		mla_heads(0, H, -1, &job);

	return matmul_vec(self, &d->wo, ho, d->out, dim, H * vh);
}

CPU_BACKEND_REGISTER("cpu_scalar", cpu_scalar_ctor, 0, NULL)

status_code cpu_attention_full_batch(backend *self, const buffer *q, const buffer *k,
									 const buffer *v, buffer *out, int n_heads, int n_kv_heads,
									 int head_dim, int n_rows, int flash_attn, float scale) {
	if (n_rows < 1 || n_heads < 1 || n_kv_heads < 1 || n_heads % n_kv_heads != 0 || head_dim < 1 ||
		head_dim > HEAD_DIM_MAX)
		return ERR_INVALID_ARG;
	cpu_attn_batch_job job	 = cpu_full_attn_job(self, q, k, v, out, n_heads, n_kv_heads, head_dim,
												 n_rows, flash_attn, scale);
	cpu_priv		  *p	 = self->priv;
	int				   total = n_rows * n_heads;
	if (p->pool && p->thread_scratch && tpool_current_tid() < 0 && total >= 2)
		tpool_parallel_for(p->pool, total, 1, cpu_attn_batch_chunk, &job);
	else
		cpu_attn_batch_chunk(0, total, 0, &job);
	return OK;
}
