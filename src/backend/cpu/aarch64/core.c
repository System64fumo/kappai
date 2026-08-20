#include "backend/backend.h"
#include "backend/cpu/common.h"
#include "backend/cpu/scalar/quants.h"
#include "common.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define CPU_ELEMWISE_MIN_PER_THREAD 4096
#define MLA_KROT_PARALLEL_MIN_POS 64
#define ATTN_BITREV_MIN_M 8
#define ATTN_BITREV_STACK_MAX 256

static inline void neon_zero_f32(float *dst, int n) {
	float32x4_t zero = vdupq_n_f32(0.0f);
	int			d	 = 0;
	for (; d + 32 <= n; d += 32) {
		vst1q_f32(dst + d, zero);
		vst1q_f32(dst + d + 4, zero);
		vst1q_f32(dst + d + 8, zero);
		vst1q_f32(dst + d + 12, zero);
		vst1q_f32(dst + d + 16, zero);
		vst1q_f32(dst + d + 20, zero);
		vst1q_f32(dst + d + 24, zero);
		vst1q_f32(dst + d + 28, zero);
	}
	for (; d + 16 <= n; d += 16) {
		vst1q_f32(dst + d, zero);
		vst1q_f32(dst + d + 4, zero);
		vst1q_f32(dst + d + 8, zero);
		vst1q_f32(dst + d + 12, zero);
	}
	for (; d + 4 <= n; d += 4)
		vst1q_f32(dst + d, zero);
	for (; d < n; d++)
		dst[d] = 0.0f;
}

#define cpu_attn_job_neon cpu_attn_job
#define cpu_attn_batch_job_neon cpu_attn_batch_job

typedef struct {
	const float *g, *u;
	float		*o;
} cpu_ffn_act_args;

typedef struct {
	const float *qf;
	float		*outf;
	const float *layer_c;
	const float *k_b;
	const float *v_b;
	const float *k_pe_rot_all;
	const float *rope_cos_base;
	const float *rope_sin_base;
	int			 qk_head, qk_rope, qk_nope, v_head, kv_lora, n_pos, half_rope, pos;
	float		 scale;
} cpu_mla_job_neon;

typedef struct {
	const float *layer_c;
	float		*k_pe_rot_all;
	const float *rope_cos_base;
	const float *rope_sin_base;
	int			 total_dim;
	int			 kv_lora;
	int			 qk_rope;
	int			 half_rope;
} cpu_mla_krot_job;

status_code cpu_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos, const buffer *k_in,
					   const buffer *v_in, int n_kv_heads, int head_dim, int n_ctx,
					   int n_kv_heads_active) {
	cpu_priv	*p		  = self->priv;
	const float *kf		  = (const float *)cpu_ptr(k_in);
	const float *vf		  = (const float *)cpu_ptr(v_in);
	int			 n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

	if (p->kv_quant == KV_QUANT_Q8_0) {
		int		 hd_stride	  = head_dim;
		size_t	 n_blocks	  = ((size_t)hd_stride + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		size_t	 block_stride = n_blocks * KV_Q8_0_BLOCK_BYTES;
		size_t	 layer_stride = (size_t)n_kv_heads * n_ctx * block_stride;
		size_t	 kvh_stride	  = (size_t)n_ctx * block_stride;
		size_t	 pos_off	  = (size_t)pos * block_stride;
		uint8_t *kd_base	  = (uint8_t *)k->handle;
		uint8_t *vd_base	  = (uint8_t *)v->handle;
		for (int kvh = 0; kvh < n_active; kvh++) {
			size_t layer_base =
				p->kv_layer_off ? p->kv_layer_off[layer] : ((size_t)layer * layer_stride);
			size_t off = layer_base + ((size_t)kvh * kvh_stride) + pos_off;
			cpu_kv_put_q8_0_head(kd_base + off, vd_base + off, kf + ((size_t)kvh * head_dim),
								 vf + ((size_t)kvh * head_dim), head_dim);
		}
		return OK;
	}
	uint16_t *kd_base	   = (uint16_t *)cpu_ptr(k);
	uint16_t *vd_base	   = (uint16_t *)cpu_ptr(v);
	int		  hd_stride	   = head_dim;
	size_t	  layer_stride = (size_t)n_kv_heads * n_ctx * hd_stride;
	size_t	  kvh_stride   = (size_t)n_ctx * hd_stride;
	size_t	  pos_off	   = (size_t)pos * hd_stride;

	for (int kvh = 0; kvh < n_active; kvh++) {
		size_t		 layer_base = p->kv_layer_off ? p->kv_layer_off[layer] / sizeof(uint16_t)
												  : ((size_t)layer * layer_stride);
		size_t		 off		= layer_base + ((size_t)kvh * kvh_stride) + pos_off;
		uint16_t	*kd			= kd_base + off;
		uint16_t	*vd			= vd_base + off;
		const float *kfh		= kf + ((size_t)kvh * head_dim);
		const float *vfh		= vf + ((size_t)kvh * head_dim);
		int			 i			= 0;
		for (; i + 32 <= head_dim; i += 32) {
			float32x4_t k0	= vld1q_f32(kfh + i);
			float32x4_t k1	= vld1q_f32(kfh + i + 4);
			float32x4_t k2	= vld1q_f32(kfh + i + 8);
			float32x4_t k3	= vld1q_f32(kfh + i + 12);
			float32x4_t k4	= vld1q_f32(kfh + i + 16);
			float32x4_t k5	= vld1q_f32(kfh + i + 20);
			float32x4_t k6	= vld1q_f32(kfh + i + 24);
			float32x4_t k7	= vld1q_f32(kfh + i + 28);
			float16x4_t kh0 = vcvt_f16_f32(k0);
			float16x4_t kh1 = vcvt_f16_f32(k1);
			float16x4_t kh2 = vcvt_f16_f32(k2);
			float16x4_t kh3 = vcvt_f16_f32(k3);
			float16x4_t kh4 = vcvt_f16_f32(k4);
			float16x4_t kh5 = vcvt_f16_f32(k5);
			float16x4_t kh6 = vcvt_f16_f32(k6);
			float16x4_t kh7 = vcvt_f16_f32(k7);
			vst1q_u16(kd + i, vreinterpretq_u16_f16(vcombine_f16(kh0, kh1)));
			vst1q_u16(kd + i + 8, vreinterpretq_u16_f16(vcombine_f16(kh2, kh3)));
			vst1q_u16(kd + i + 16, vreinterpretq_u16_f16(vcombine_f16(kh4, kh5)));
			vst1q_u16(kd + i + 24, vreinterpretq_u16_f16(vcombine_f16(kh6, kh7)));

			float32x4_t v0	= vld1q_f32(vfh + i);
			float32x4_t v1	= vld1q_f32(vfh + i + 4);
			float32x4_t v2	= vld1q_f32(vfh + i + 8);
			float32x4_t v3	= vld1q_f32(vfh + i + 12);
			float32x4_t v4	= vld1q_f32(vfh + i + 16);
			float32x4_t v5	= vld1q_f32(vfh + i + 20);
			float32x4_t v6	= vld1q_f32(vfh + i + 24);
			float32x4_t v7	= vld1q_f32(vfh + i + 28);
			float16x4_t vh0 = vcvt_f16_f32(v0);
			float16x4_t vh1 = vcvt_f16_f32(v1);
			float16x4_t vh2 = vcvt_f16_f32(v2);
			float16x4_t vh3 = vcvt_f16_f32(v3);
			float16x4_t vh4 = vcvt_f16_f32(v4);
			float16x4_t vh5 = vcvt_f16_f32(v5);
			float16x4_t vh6 = vcvt_f16_f32(v6);
			float16x4_t vh7 = vcvt_f16_f32(v7);
			vst1q_u16(vd + i, vreinterpretq_u16_f16(vcombine_f16(vh0, vh1)));
			vst1q_u16(vd + i + 8, vreinterpretq_u16_f16(vcombine_f16(vh2, vh3)));
			vst1q_u16(vd + i + 16, vreinterpretq_u16_f16(vcombine_f16(vh4, vh5)));
			vst1q_u16(vd + i + 24, vreinterpretq_u16_f16(vcombine_f16(vh6, vh7)));
		}
		for (; i + 16 <= head_dim; i += 16) {
			float32x4_t k0	= vld1q_f32(kfh + i);
			float32x4_t k1	= vld1q_f32(kfh + i + 4);
			float32x4_t k2	= vld1q_f32(kfh + i + 8);
			float32x4_t k3	= vld1q_f32(kfh + i + 12);
			float16x4_t kh0 = vcvt_f16_f32(k0);
			float16x4_t kh1 = vcvt_f16_f32(k1);
			float16x4_t kh2 = vcvt_f16_f32(k2);
			float16x4_t kh3 = vcvt_f16_f32(k3);
			vst1q_u16(kd + i, vreinterpretq_u16_f16(vcombine_f16(kh0, kh1)));
			vst1q_u16(kd + i + 8, vreinterpretq_u16_f16(vcombine_f16(kh2, kh3)));

			float32x4_t v0	= vld1q_f32(vfh + i);
			float32x4_t v1	= vld1q_f32(vfh + i + 4);
			float32x4_t v2	= vld1q_f32(vfh + i + 8);
			float32x4_t v3	= vld1q_f32(vfh + i + 12);
			float16x4_t vh0 = vcvt_f16_f32(v0);
			float16x4_t vh1 = vcvt_f16_f32(v1);
			float16x4_t vh2 = vcvt_f16_f32(v2);
			float16x4_t vh3 = vcvt_f16_f32(v3);
			vst1q_u16(vd + i, vreinterpretq_u16_f16(vcombine_f16(vh0, vh1)));
			vst1q_u16(vd + i + 8, vreinterpretq_u16_f16(vcombine_f16(vh2, vh3)));
		}
		for (; i + 8 <= head_dim; i += 8) {
			float32x4_t k0	= vld1q_f32(kfh + i);
			float32x4_t k1	= vld1q_f32(kfh + i + 4);
			float16x4_t kh0 = vcvt_f16_f32(k0);
			float16x4_t kh1 = vcvt_f16_f32(k1);
			vst1q_u16(kd + i, vreinterpretq_u16_f16(vcombine_f16(kh0, kh1)));

			float32x4_t v0	= vld1q_f32(vfh + i);
			float32x4_t v1	= vld1q_f32(vfh + i + 4);
			float16x4_t vh0 = vcvt_f16_f32(v0);
			float16x4_t vh1 = vcvt_f16_f32(v1);
			vst1q_u16(vd + i, vreinterpretq_u16_f16(vcombine_f16(vh0, vh1)));
		}
		for (; i + 4 <= head_dim; i += 4) {
			float32x4_t k0 = vld1q_f32(kfh + i);
			float32x4_t v0 = vld1q_f32(vfh + i);
			vst1_u16(kd + i, vreinterpret_u16_f16(vcvt_f16_f32(k0)));
			vst1_u16(vd + i, vreinterpret_u16_f16(vcvt_f16_f32(v0)));
		}
		for (; i < head_dim; i++) {
			kd[i] = f32_to_f16(kfh[i]);
			vd[i] = f32_to_f16(vfh[i]);
		}
	}
	return OK;
}

static void cpu_rope_one_neon(float *v, int n_heads, int head_dim, const float *rope_cos,
							  const float *rope_sin, int neox) {
	int half = head_dim / 2;
	if (neox) {
		int j = 0;
		for (; j + 8 <= half; j += 8) {
			const float32x4_t c0 = vld1q_f32(rope_cos + j);
			const float32x4_t c1 = vld1q_f32(rope_cos + j + 4);
			const float32x4_t s0 = vld1q_f32(rope_sin + j);
			const float32x4_t s1 = vld1q_f32(rope_sin + j + 4);
			for (int h = 0; h < n_heads; h++) {
				float	   *vh	= v + ((size_t)h * head_dim);
				float32x4_t v0a = vld1q_f32(vh + j);
				float32x4_t v0b = vld1q_f32(vh + j + 4);
				float32x4_t v1a = vld1q_f32(vh + j + half);
				float32x4_t v1b = vld1q_f32(vh + j + half + 4);
				vst1q_f32(vh + j, vmlsq_f32(vmulq_f32(v0a, c0), v1a, s0));
				vst1q_f32(vh + j + 4, vmlsq_f32(vmulq_f32(v0b, c1), v1b, s1));
				vst1q_f32(vh + j + half, vmlaq_f32(vmulq_f32(v0a, s0), v1a, c0));
				vst1q_f32(vh + j + half + 4, vmlaq_f32(vmulq_f32(v0b, s1), v1b, c1));
			}
		}
		for (; j + 4 <= half; j += 4) {
			const float32x4_t c0 = vld1q_f32(rope_cos + j);
			const float32x4_t s0 = vld1q_f32(rope_sin + j);
			for (int h = 0; h < n_heads; h++) {
				float	   *vh	= v + ((size_t)h * head_dim);
				float32x4_t v0a = vld1q_f32(vh + j);
				float32x4_t v1a = vld1q_f32(vh + j + half);
				vst1q_f32(vh + j, vmlsq_f32(vmulq_f32(v0a, c0), v1a, s0));
				vst1q_f32(vh + j + half, vmlaq_f32(vmulq_f32(v0a, s0), v1a, c0));
			}
		}
		for (; j < half; j++) {
			const float c = rope_cos[j];
			const float s = rope_sin[j];
			for (int h = 0; h < n_heads; h++) {
				float *vh	 = v + ((size_t)h * head_dim);
				float  v0	 = vh[j];
				float  v1	 = vh[j + half];
				vh[j]		 = (v0 * c) - (v1 * s);
				vh[j + half] = (v0 * s) + (v1 * c);
			}
		}
	} else {
		int j = 0;
		for (; j + 8 <= half; j += 8) {
			const float32x4_t c0 = vld1q_f32(rope_cos + j);
			const float32x4_t c1 = vld1q_f32(rope_cos + j + 4);
			const float32x4_t s0 = vld1q_f32(rope_sin + j);
			const float32x4_t s1 = vld1q_f32(rope_sin + j + 4);
			for (int h = 0; h < n_heads; h++) {
				float		 *vh  = v + ((size_t)h * head_dim);
				float32x4x2_t pa0 = vld2q_f32(vh + 2 * j);
				float32x4x2_t pa1 = vld2q_f32(vh + 2 * j + 8);
				float32x4_t	  v0a = pa0.val[0];
				float32x4_t	  v1a = pa0.val[1];
				float32x4_t	  v0b = pa1.val[0];
				float32x4_t	  v1b = pa1.val[1];
				float32x4_t	  ra0 = vmlsq_f32(vmulq_f32(v0a, c0), v1a, s0);
				float32x4_t	  ra1 = vmlsq_f32(vmulq_f32(v0b, c1), v1b, s1);
				float32x4_t	  rb0 = vmlaq_f32(vmulq_f32(v0a, s0), v1a, c0);
				float32x4_t	  rb1 = vmlaq_f32(vmulq_f32(v0b, s1), v1b, c1);
				float32x4x2_t ro0 = {{ra0, rb0}};
				float32x4x2_t ro1 = {{ra1, rb1}};
				vst2q_f32(vh + 2 * j, ro0);
				vst2q_f32(vh + 2 * j + 8, ro1);
			}
		}
		for (; j < half; j++) {
			const float c = rope_cos[j];
			const float s = rope_sin[j];
			for (int h = 0; h < n_heads; h++) {
				float *vh		= v + ((size_t)h * head_dim);
				float  v0		= vh[2 * j];
				float  v1		= vh[(2 * j) + 1];
				vh[2 * j]		= (v0 * c) - (v1 * s);
				vh[(2 * j) + 1] = (v0 * s) + (v1 * c);
			}
		}
	}
}

status_code cpu_rope(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
					 const float *rope_cos_base, const float *rope_sin_base) {
	int			 half	  = head_dim / 2;
	const float *rope_cos = rope_cos_base + ((size_t)pos * half);
	const float *rope_sin = rope_sin_base + ((size_t)pos * half);
	cpu_rope_one_neon((float *)cpu_ptr(vec), n_heads, head_dim, rope_cos, rope_sin,
					  self->rope_neox);
	return OK;
}

status_code cpu_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
						int head_dim, int pos, const float *rope_cos_base,
						const float *rope_sin_base) {
	int			 half	  = head_dim / 2;
	const float *rope_cos = rope_cos_base + ((size_t)pos * half);
	const float *rope_sin = rope_sin_base + ((size_t)pos * half);
	int			 neox	  = self->rope_neox;
	cpu_rope_one_neon((float *)cpu_ptr(q), n_heads, head_dim, rope_cos, rope_sin, neox);
	cpu_rope_one_neon((float *)cpu_ptr(k), n_kv_heads, head_dim, rope_cos, rope_sin, neox);
	return OK;
}

static void cpu_rope_batch_chunk_neon(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rope_batch_job *j	 = ctx;
	int					half = j->head_dim / 2;
	int					neox = j->rope_neox;
	for (int row = begin; row < end; row++) {
		int			 pos = j->pos_start + row;
		const float *rc	 = j->rope_cos_base + ((size_t)pos * half);
		const float *rs	 = j->rope_sin_base + ((size_t)pos * half);
		cpu_rope_one_neon(j->vec + ((size_t)row * j->n_heads * j->head_dim), j->n_heads,
						  j->head_dim, rc, rs, neox);
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
	cpu_run_batch(p->pool, m, cpu_rope_batch_chunk_neon, &job);
	return OK;
}

static void cpu_rope_qk_batch_chunk_neon(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rope_qk_batch_job *j	= ctx;
	int					   half = j->head_dim / 2;
	int					   neox = j->rope_neox;
	for (int row = begin; row < end; row++) {
		int			 pos = j->pos_start + row;
		const float *rc	 = j->rope_cos_base + ((size_t)pos * half);
		const float *rs	 = j->rope_sin_base + ((size_t)pos * half);
		cpu_rope_one_neon(j->q + ((size_t)row * j->n_heads * j->head_dim), j->n_heads, j->head_dim,
						  rc, rs, neox);
		cpu_rope_one_neon(j->k + ((size_t)row * j->n_kv_heads * j->head_dim), j->n_kv_heads,
						  j->head_dim, rc, rs, neox);
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
	cpu_run_batch(p->pool, m, cpu_rope_qk_batch_chunk_neon, &job);
	return OK;
}

float dot8(const float *restrict a, const float *restrict b, int head_dim) {
	float32x4_t acc0 = vdupq_n_f32(0.0f);
	float32x4_t acc1 = vdupq_n_f32(0.0f);
	float32x4_t acc2 = vdupq_n_f32(0.0f);
	float32x4_t acc3 = vdupq_n_f32(0.0f);
	int			d	 = 0;

	for (; d + 32 <= head_dim; d += 32) {

		float32x4_t a0 = vld1q_f32(a + d);
		float32x4_t a1 = vld1q_f32(a + d + 4);
		float32x4_t a2 = vld1q_f32(a + d + 8);
		float32x4_t a3 = vld1q_f32(a + d + 12);
		float32x4_t a4 = vld1q_f32(a + d + 16);
		float32x4_t a5 = vld1q_f32(a + d + 20);
		float32x4_t a6 = vld1q_f32(a + d + 24);
		float32x4_t a7 = vld1q_f32(a + d + 28);

		acc0 = vfmaq_f32(acc0, a0, vld1q_f32(b + d));
		acc1 = vfmaq_f32(acc1, a1, vld1q_f32(b + d + 4));
		acc2 = vfmaq_f32(acc2, a2, vld1q_f32(b + d + 8));
		acc3 = vfmaq_f32(acc3, a3, vld1q_f32(b + d + 12));
		acc0 = vfmaq_f32(acc0, a4, vld1q_f32(b + d + 16));
		acc1 = vfmaq_f32(acc1, a5, vld1q_f32(b + d + 20));
		acc2 = vfmaq_f32(acc2, a6, vld1q_f32(b + d + 24));
		acc3 = vfmaq_f32(acc3, a7, vld1q_f32(b + d + 28));
	}

	for (; d + 16 <= head_dim; d += 16) {
		float32x4_t a0 = vld1q_f32(a + d);
		float32x4_t a1 = vld1q_f32(a + d + 4);
		float32x4_t a2 = vld1q_f32(a + d + 8);
		float32x4_t a3 = vld1q_f32(a + d + 12);

		acc0 = vfmaq_f32(acc0, a0, vld1q_f32(b + d));
		acc1 = vfmaq_f32(acc1, a1, vld1q_f32(b + d + 4));
		acc2 = vfmaq_f32(acc2, a2, vld1q_f32(b + d + 8));
		acc3 = vfmaq_f32(acc3, a3, vld1q_f32(b + d + 12));
	}

	for (; d + 4 <= head_dim; d += 4) {
		acc0 = vfmaq_f32(acc0, vld1q_f32(a + d), vld1q_f32(b + d));
	}

	float32x4_t acc01 = vaddq_f32(acc0, acc1);
	float32x4_t acc23 = vaddq_f32(acc2, acc3);
	float32x4_t acc	  = vaddq_f32(acc01, acc23);
	float		s	  = vaddvq_f32(acc);

	for (; d < head_dim; d++) {
		s += a[d] * b[d];
	}

	return s;
}

float dot8_f16(const float *restrict a, const uint16_t *restrict b, int head_dim) {
	float32x4_t acc0 = vdupq_n_f32(0.0f);
	float32x4_t acc1 = vdupq_n_f32(0.0f);
	float32x4_t acc2 = vdupq_n_f32(0.0f);
	float32x4_t acc3 = vdupq_n_f32(0.0f);
	int			d	 = 0;

	for (; d + 32 <= head_dim; d += 32) {

		float32x4_t a0 = vld1q_f32(a + d);
		float32x4_t a1 = vld1q_f32(a + d + 4);
		float32x4_t a2 = vld1q_f32(a + d + 8);
		float32x4_t a3 = vld1q_f32(a + d + 12);
		float32x4_t a4 = vld1q_f32(a + d + 16);
		float32x4_t a5 = vld1q_f32(a + d + 20);
		float32x4_t a6 = vld1q_f32(a + d + 24);
		float32x4_t a7 = vld1q_f32(a + d + 28);

		float16x8_t bh0 = vld1q_f16((const float16_t *)(b + d));
		float16x8_t bh1 = vld1q_f16((const float16_t *)(b + d + 8));
		float16x8_t bh2 = vld1q_f16((const float16_t *)(b + d + 16));
		float16x8_t bh3 = vld1q_f16((const float16_t *)(b + d + 24));
		float32x4_t b0	= vcvt_f32_f16(vget_low_f16(bh0));
		float32x4_t b1	= vcvt_f32_f16(vget_high_f16(bh0));
		float32x4_t b2	= vcvt_f32_f16(vget_low_f16(bh1));
		float32x4_t b3	= vcvt_f32_f16(vget_high_f16(bh1));
		float32x4_t b4	= vcvt_f32_f16(vget_low_f16(bh2));
		float32x4_t b5	= vcvt_f32_f16(vget_high_f16(bh2));
		float32x4_t b6	= vcvt_f32_f16(vget_low_f16(bh3));
		float32x4_t b7	= vcvt_f32_f16(vget_high_f16(bh3));

		acc0 = vfmaq_f32(acc0, a0, b0);
		acc1 = vfmaq_f32(acc1, a1, b1);
		acc2 = vfmaq_f32(acc2, a2, b2);
		acc3 = vfmaq_f32(acc3, a3, b3);
		acc0 = vfmaq_f32(acc0, a4, b4);
		acc1 = vfmaq_f32(acc1, a5, b5);
		acc2 = vfmaq_f32(acc2, a6, b6);
		acc3 = vfmaq_f32(acc3, a7, b7);
	}

	for (; d + 16 <= head_dim; d += 16) {
		float32x4_t a0 = vld1q_f32(a + d);
		float32x4_t a1 = vld1q_f32(a + d + 4);
		float32x4_t a2 = vld1q_f32(a + d + 8);
		float32x4_t a3 = vld1q_f32(a + d + 12);

		float16x8_t bh0 = vld1q_f16((const float16_t *)(b + d));
		float16x8_t bh1 = vld1q_f16((const float16_t *)(b + d + 8));
		float32x4_t b0	= vcvt_f32_f16(vget_low_f16(bh0));
		float32x4_t b1	= vcvt_f32_f16(vget_high_f16(bh0));
		float32x4_t b2	= vcvt_f32_f16(vget_low_f16(bh1));
		float32x4_t b3	= vcvt_f32_f16(vget_high_f16(bh1));

		acc0 = vfmaq_f32(acc0, a0, b0);
		acc1 = vfmaq_f32(acc1, a1, b1);
		acc2 = vfmaq_f32(acc2, a2, b2);
		acc3 = vfmaq_f32(acc3, a3, b3);
	}

	for (; d + 8 <= head_dim; d += 8) {
		float32x4_t a0	= vld1q_f32(a + d);
		float32x4_t a1	= vld1q_f32(a + d + 4);
		float16x8_t bh0 = vld1q_f16((const float16_t *)(b + d));
		float32x4_t b0	= vcvt_f32_f16(vget_low_f16(bh0));
		float32x4_t b1	= vcvt_f32_f16(vget_high_f16(bh0));
		acc0			= vfmaq_f32(acc0, a0, b0);
		acc1			= vfmaq_f32(acc1, a1, b1);
	}

	for (; d + 4 <= head_dim; d += 4) {
		float32x4_t a0	= vld1q_f32(a + d);
		float16x4_t bh0 = vld1_f16((const float16_t *)(b + d));
		float32x4_t b0	= vcvt_f32_f16(bh0);
		acc0			= vfmaq_f32(acc0, a0, b0);
	}

	float32x4_t acc01 = vaddq_f32(acc0, acc1);
	float32x4_t acc23 = vaddq_f32(acc2, acc3);
	float32x4_t acc	  = vaddq_f32(acc01, acc23);
	float		s	  = vaddvq_f32(acc);

	for (; d < head_dim; d++) {
		s += a[d] * f16_to_f32_fast(b[d]);
	}

	return s;
}

static inline float32x4_t vld1q_f16_to_f32(const uint16_t *ptr) {
	float16x4_t h = vld1_f16((const float16_t *)ptr);
	return vcvt_f32_f16(h);
}

static inline void vld1q_f16x8_to_f32x2(const uint16_t *ptr, float32x4_t *lo, float32x4_t *hi) {
	float16x8_t h = vld1q_f16((const float16_t *)ptr);
	*lo			  = vcvt_f32_f16(vget_low_f16(h));
	*hi			  = vcvt_f32_f16(vget_high_f16(h));
}

static void cpu_attention_inner(uint16_t *restrict k_slice, uint16_t *restrict v_slice,
								int kv_stride, const float *qh, float *out_h, int head_dim,
								int n_pos, float scale, int flash_attn, float *restrict scores) {
	if (flash_attn) {
		float M = -INFINITY;
		float S = 0.0f;
		float VKQ[HEAD_DIM_MAX] __attribute__((aligned(64)));
		neon_zero_f32(VKQ, head_dim);

		for (int t = 0; t < n_pos; t++) {
			const uint16_t *kt = k_slice + ((size_t)t * kv_stride);
			const uint16_t *vt = v_slice + ((size_t)t * kv_stride);
			float			ss = dot8_f16(qh, kt, head_dim) * scale;

			float vs;
			float ms = 1.0f;
			if (ss > M) {
				float M_old = M;
				M			= ss;
				if (S > 0.0f) {
					ms				 = fast_expf(M_old - M);
					float32x4_t ms_v = vdupq_n_f32(ms);
					int			d	 = 0;
					for (; d + 32 <= head_dim; d += 32) {
						vst1q_f32(VKQ + d, vmulq_f32(vld1q_f32(VKQ + d), ms_v));
						vst1q_f32(VKQ + d + 4, vmulq_f32(vld1q_f32(VKQ + d + 4), ms_v));
						vst1q_f32(VKQ + d + 8, vmulq_f32(vld1q_f32(VKQ + d + 8), ms_v));
						vst1q_f32(VKQ + d + 12, vmulq_f32(vld1q_f32(VKQ + d + 12), ms_v));
						vst1q_f32(VKQ + d + 16, vmulq_f32(vld1q_f32(VKQ + d + 16), ms_v));
						vst1q_f32(VKQ + d + 20, vmulq_f32(vld1q_f32(VKQ + d + 20), ms_v));
						vst1q_f32(VKQ + d + 24, vmulq_f32(vld1q_f32(VKQ + d + 24), ms_v));
						vst1q_f32(VKQ + d + 28, vmulq_f32(vld1q_f32(VKQ + d + 28), ms_v));
					}
					for (; d + 16 <= head_dim; d += 16) {
						vst1q_f32(VKQ + d, vmulq_f32(vld1q_f32(VKQ + d), ms_v));
						vst1q_f32(VKQ + d + 4, vmulq_f32(vld1q_f32(VKQ + d + 4), ms_v));
						vst1q_f32(VKQ + d + 8, vmulq_f32(vld1q_f32(VKQ + d + 8), ms_v));
						vst1q_f32(VKQ + d + 12, vmulq_f32(vld1q_f32(VKQ + d + 12), ms_v));
					}
					for (; d + 4 <= head_dim; d += 4)
						vst1q_f32(VKQ + d, vmulq_f32(vld1q_f32(VKQ + d), ms_v));
					for (; d < head_dim; d++)
						VKQ[d] *= ms;
				}
				vs = 1.0f;
			} else {
				vs = fast_expf(ss - M);
			}

			float32x4_t vs_v = vdupq_n_f32(vs);
			int			d	 = 0;
			for (; d + 32 <= head_dim; d += 32) {
				float32x4_t vt0, vt1, vt2, vt3, vt4, vt5, vt6, vt7;
				vld1q_f16x8_to_f32x2(vt + d, &vt0, &vt1);
				vld1q_f16x8_to_f32x2(vt + d + 8, &vt2, &vt3);
				vld1q_f16x8_to_f32x2(vt + d + 16, &vt4, &vt5);
				vld1q_f16x8_to_f32x2(vt + d + 24, &vt6, &vt7);
				vst1q_f32(VKQ + d, vfmaq_f32(vld1q_f32(VKQ + d), vs_v, vt0));
				vst1q_f32(VKQ + d + 4, vfmaq_f32(vld1q_f32(VKQ + d + 4), vs_v, vt1));
				vst1q_f32(VKQ + d + 8, vfmaq_f32(vld1q_f32(VKQ + d + 8), vs_v, vt2));
				vst1q_f32(VKQ + d + 12, vfmaq_f32(vld1q_f32(VKQ + d + 12), vs_v, vt3));
				vst1q_f32(VKQ + d + 16, vfmaq_f32(vld1q_f32(VKQ + d + 16), vs_v, vt4));
				vst1q_f32(VKQ + d + 20, vfmaq_f32(vld1q_f32(VKQ + d + 20), vs_v, vt5));
				vst1q_f32(VKQ + d + 24, vfmaq_f32(vld1q_f32(VKQ + d + 24), vs_v, vt6));
				vst1q_f32(VKQ + d + 28, vfmaq_f32(vld1q_f32(VKQ + d + 28), vs_v, vt7));
			}
			for (; d + 16 <= head_dim; d += 16) {
				float32x4_t vt0, vt1, vt2, vt3;
				vld1q_f16x8_to_f32x2(vt + d, &vt0, &vt1);
				vld1q_f16x8_to_f32x2(vt + d + 8, &vt2, &vt3);
				vst1q_f32(VKQ + d, vfmaq_f32(vld1q_f32(VKQ + d), vs_v, vt0));
				vst1q_f32(VKQ + d + 4, vfmaq_f32(vld1q_f32(VKQ + d + 4), vs_v, vt1));
				vst1q_f32(VKQ + d + 8, vfmaq_f32(vld1q_f32(VKQ + d + 8), vs_v, vt2));
				vst1q_f32(VKQ + d + 12, vfmaq_f32(vld1q_f32(VKQ + d + 12), vs_v, vt3));
			}
			for (; d + 8 <= head_dim; d += 8) {
				float32x4_t vt0, vt1;
				vld1q_f16x8_to_f32x2(vt + d, &vt0, &vt1);
				vst1q_f32(VKQ + d, vfmaq_f32(vld1q_f32(VKQ + d), vs_v, vt0));
				vst1q_f32(VKQ + d + 4, vfmaq_f32(vld1q_f32(VKQ + d + 4), vs_v, vt1));
			}
			for (; d + 4 <= head_dim; d += 4) {
				float32x4_t vt0 = vld1q_f16_to_f32(vt + d);
				vst1q_f32(VKQ + d, vfmaq_f32(vld1q_f32(VKQ + d), vs_v, vt0));
			}
			for (; d < head_dim; d++)
				VKQ[d] = fmaf(vs, f16_to_f32_fast(vt[d]), VKQ[d]);
			S = (S * ms) + vs;
		}

		float		S_inv = (S == 0.0f) ? 0.0f : 1.0f / S;
		float32x4_t inv_v = vdupq_n_f32(S_inv);
		int			d	  = 0;
		for (; d + 32 <= head_dim; d += 32) {
			vst1q_f32(out_h + d, vmulq_f32(vld1q_f32(VKQ + d), inv_v));
			vst1q_f32(out_h + d + 4, vmulq_f32(vld1q_f32(VKQ + d + 4), inv_v));
			vst1q_f32(out_h + d + 8, vmulq_f32(vld1q_f32(VKQ + d + 8), inv_v));
			vst1q_f32(out_h + d + 12, vmulq_f32(vld1q_f32(VKQ + d + 12), inv_v));
			vst1q_f32(out_h + d + 16, vmulq_f32(vld1q_f32(VKQ + d + 16), inv_v));
			vst1q_f32(out_h + d + 20, vmulq_f32(vld1q_f32(VKQ + d + 20), inv_v));
			vst1q_f32(out_h + d + 24, vmulq_f32(vld1q_f32(VKQ + d + 24), inv_v));
			vst1q_f32(out_h + d + 28, vmulq_f32(vld1q_f32(VKQ + d + 28), inv_v));
		}
		for (; d + 16 <= head_dim; d += 16) {
			vst1q_f32(out_h + d, vmulq_f32(vld1q_f32(VKQ + d), inv_v));
			vst1q_f32(out_h + d + 4, vmulq_f32(vld1q_f32(VKQ + d + 4), inv_v));
			vst1q_f32(out_h + d + 8, vmulq_f32(vld1q_f32(VKQ + d + 8), inv_v));
			vst1q_f32(out_h + d + 12, vmulq_f32(vld1q_f32(VKQ + d + 12), inv_v));
		}
		for (; d + 4 <= head_dim; d += 4)
			vst1q_f32(out_h + d, vmulq_f32(vld1q_f32(VKQ + d), inv_v));
		for (; d < head_dim; d++)
			out_h[d] = VKQ[d] * S_inv;
		return;
	}

	for (int t = 0; t < n_pos; t++) {
		const uint16_t *kt = k_slice + ((size_t)t * kv_stride);
		scores[t]		   = dot8_f16(qh, kt, head_dim) * scale;
	}
	softmax_masked(scores, n_pos);

	if (n_pos == 0) {
		float32x4_t zero = vdupq_n_f32(0.0f);
		int			d	 = 0;
		for (; d + 32 <= head_dim; d += 32) {
			vst1q_f32(out_h + d, zero);
			vst1q_f32(out_h + d + 4, zero);
			vst1q_f32(out_h + d + 8, zero);
			vst1q_f32(out_h + d + 12, zero);
			vst1q_f32(out_h + d + 16, zero);
			vst1q_f32(out_h + d + 20, zero);
			vst1q_f32(out_h + d + 24, zero);
			vst1q_f32(out_h + d + 28, zero);
		}
		for (; d + 16 <= head_dim; d += 16) {
			vst1q_f32(out_h + d, zero);
			vst1q_f32(out_h + d + 4, zero);
			vst1q_f32(out_h + d + 8, zero);
			vst1q_f32(out_h + d + 12, zero);
		}
		for (; d + 4 <= head_dim; d += 4)
			vst1q_f32(out_h + d, zero);
		for (; d < head_dim; d++)
			out_h[d] = 0.0f;
		return;
	}

	{
		float			sv	 = scores[0];
		const uint16_t *vt	 = v_slice;
		float32x4_t		sv_v = vdupq_n_f32(sv);
		int				d	 = 0;
		for (; d + 32 <= head_dim; d += 32) {
			float32x4_t vt0;
			float32x4_t vt1;
			float32x4_t vt2;
			float32x4_t vt3;
			float32x4_t vt4;
			float32x4_t vt5;
			float32x4_t vt6;
			float32x4_t vt7;
			vld1q_f16x8_to_f32x2(vt + d, &vt0, &vt1);
			vld1q_f16x8_to_f32x2(vt + d + 8, &vt2, &vt3);
			vld1q_f16x8_to_f32x2(vt + d + 16, &vt4, &vt5);
			vld1q_f16x8_to_f32x2(vt + d + 24, &vt6, &vt7);
			vst1q_f32(out_h + d, vmulq_f32(sv_v, vt0));
			vst1q_f32(out_h + d + 4, vmulq_f32(sv_v, vt1));
			vst1q_f32(out_h + d + 8, vmulq_f32(sv_v, vt2));
			vst1q_f32(out_h + d + 12, vmulq_f32(sv_v, vt3));
			vst1q_f32(out_h + d + 16, vmulq_f32(sv_v, vt4));
			vst1q_f32(out_h + d + 20, vmulq_f32(sv_v, vt5));
			vst1q_f32(out_h + d + 24, vmulq_f32(sv_v, vt6));
			vst1q_f32(out_h + d + 28, vmulq_f32(sv_v, vt7));
		}
		for (; d + 16 <= head_dim; d += 16) {
			float32x4_t vt0;
			float32x4_t vt1;
			float32x4_t vt2;
			float32x4_t vt3;
			vld1q_f16x8_to_f32x2(vt + d, &vt0, &vt1);
			vld1q_f16x8_to_f32x2(vt + d + 8, &vt2, &vt3);
			vst1q_f32(out_h + d, vmulq_f32(sv_v, vt0));
			vst1q_f32(out_h + d + 4, vmulq_f32(sv_v, vt1));
			vst1q_f32(out_h + d + 8, vmulq_f32(sv_v, vt2));
			vst1q_f32(out_h + d + 12, vmulq_f32(sv_v, vt3));
		}
		for (; d + 8 <= head_dim; d += 8) {
			float32x4_t vt0;
			float32x4_t vt1;
			vld1q_f16x8_to_f32x2(vt + d, &vt0, &vt1);
			vst1q_f32(out_h + d, vmulq_f32(sv_v, vt0));
			vst1q_f32(out_h + d + 4, vmulq_f32(sv_v, vt1));
		}
		for (; d + 4 <= head_dim; d += 4) {
			float32x4_t vt0 = vld1q_f16_to_f32(vt + d);
			vst1q_f32(out_h + d, vmulq_f32(sv_v, vt0));
		}
		for (; d < head_dim; d++)
			out_h[d] = sv * f16_to_f32_fast(vt[d]);
	}

	for (int t = 1; t < n_pos; t++) {
		float			sv	 = scores[t];
		const uint16_t *vt	 = v_slice + ((size_t)t * kv_stride);
		float32x4_t		sv_v = vdupq_n_f32(sv);
		int				d	 = 0;
		for (; d + 32 <= head_dim; d += 32) {
			float32x4_t vt0;
			float32x4_t vt1;
			float32x4_t vt2;
			float32x4_t vt3;
			float32x4_t vt4;
			float32x4_t vt5;
			float32x4_t vt6;
			float32x4_t vt7;
			vld1q_f16x8_to_f32x2(vt + d, &vt0, &vt1);
			vld1q_f16x8_to_f32x2(vt + d + 8, &vt2, &vt3);
			vld1q_f16x8_to_f32x2(vt + d + 16, &vt4, &vt5);
			vld1q_f16x8_to_f32x2(vt + d + 24, &vt6, &vt7);
			float32x4_t oh0 = vld1q_f32(out_h + d);
			float32x4_t oh1 = vld1q_f32(out_h + d + 4);
			float32x4_t oh2 = vld1q_f32(out_h + d + 8);
			float32x4_t oh3 = vld1q_f32(out_h + d + 12);
			float32x4_t oh4 = vld1q_f32(out_h + d + 16);
			float32x4_t oh5 = vld1q_f32(out_h + d + 20);
			float32x4_t oh6 = vld1q_f32(out_h + d + 24);
			float32x4_t oh7 = vld1q_f32(out_h + d + 28);
			vst1q_f32(out_h + d, vfmaq_f32(oh0, sv_v, vt0));
			vst1q_f32(out_h + d + 4, vfmaq_f32(oh1, sv_v, vt1));
			vst1q_f32(out_h + d + 8, vfmaq_f32(oh2, sv_v, vt2));
			vst1q_f32(out_h + d + 12, vfmaq_f32(oh3, sv_v, vt3));
			vst1q_f32(out_h + d + 16, vfmaq_f32(oh4, sv_v, vt4));
			vst1q_f32(out_h + d + 20, vfmaq_f32(oh5, sv_v, vt5));
			vst1q_f32(out_h + d + 24, vfmaq_f32(oh6, sv_v, vt6));
			vst1q_f32(out_h + d + 28, vfmaq_f32(oh7, sv_v, vt7));
		}
		for (; d + 16 <= head_dim; d += 16) {
			float32x4_t vt0;
			float32x4_t vt1;
			float32x4_t vt2;
			float32x4_t vt3;
			vld1q_f16x8_to_f32x2(vt + d, &vt0, &vt1);
			vld1q_f16x8_to_f32x2(vt + d + 8, &vt2, &vt3);
			vst1q_f32(out_h + d, vfmaq_f32(vld1q_f32(out_h + d), sv_v, vt0));
			vst1q_f32(out_h + d + 4, vfmaq_f32(vld1q_f32(out_h + d + 4), sv_v, vt1));
			vst1q_f32(out_h + d + 8, vfmaq_f32(vld1q_f32(out_h + d + 8), sv_v, vt2));
			vst1q_f32(out_h + d + 12, vfmaq_f32(vld1q_f32(out_h + d + 12), sv_v, vt3));
		}
		for (; d + 8 <= head_dim; d += 8) {
			float32x4_t vt0;
			float32x4_t vt1;
			vld1q_f16x8_to_f32x2(vt + d, &vt0, &vt1);
			vst1q_f32(out_h + d, vfmaq_f32(vld1q_f32(out_h + d), sv_v, vt0));
			vst1q_f32(out_h + d + 4, vfmaq_f32(vld1q_f32(out_h + d + 4), sv_v, vt1));
		}
		for (; d + 4 <= head_dim; d += 4) {
			float32x4_t vt0 = vld1q_f16_to_f32(vt + d);
			vst1q_f32(out_h + d, vfmaq_f32(vld1q_f32(out_h + d), sv_v, vt0));
		}
		for (; d < head_dim; d++)
			out_h[d] = fmaf(sv, f16_to_f32_fast(vt[d]), out_h[d]);
	}
}

static inline void vld1q_s8_to_f32x4(const int8_t *p, float32x4_t *o0, float32x4_t *o1,
									 float32x4_t *o2, float32x4_t *o3) {
	int8x16_t v	 = vld1q_s8(p);
	int16x8_t lo = vmovl_s8(vget_low_s8(v));
	int16x8_t hi = vmovl_s8(vget_high_s8(v));
	*o0			 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
	*o1			 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
	*o2			 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
	*o3			 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
}

static inline void vld1_s8x8_to_f32x2(const int8_t *p, float32x4_t *o0, float32x4_t *o1) {
	int16x8_t s16 = vmovl_s8(vld1_s8(p));
	*o0			  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16)));
	*o1			  = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16)));
}

static inline float32x4_t vld1_s8x4_to_f32(const int8_t *p) {
	int8x8_t v	  = vdup_n_s8(0);
	v			  = vld1_lane_s8(p, v, 0);
	v			  = vld1_lane_s8(p + 1, v, 1);
	v			  = vld1_lane_s8(p + 2, v, 2);
	v			  = vld1_lane_s8(p + 3, v, 3);
	int16x8_t s16 = vmovl_s8(v);
	return vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16)));
}

static float dot8_q8_0(const float *restrict a, const uint8_t *restrict block_ptr, int head_dim) {
	int	  n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	float sum	   = 0.0f;
	for (int b = 0; b < n_blocks; b++) {
		const q8_0_block *blk = (const q8_0_block *)(block_ptr + ((size_t)b * KV_Q8_0_BLOCK_BYTES));
		float			  d	  = f16_to_f32(blk->d);
		int				  base = b * KV_Q8_0_BLOCK;
		int				  n	   = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;

		const int8_t *qs   = blk->qs;
		const float	 *av   = a + base;
		int			  j	   = 0;
		float32x4_t	  acc0 = vdupq_n_f32(0.0f);
		float32x4_t	  acc1 = vdupq_n_f32(0.0f);
		float32x4_t	  acc2 = vdupq_n_f32(0.0f);
		float32x4_t	  acc3 = vdupq_n_f32(0.0f);
		for (; j + 32 <= n; j += 32) {
			float32x4_t a0 = vld1q_f32(av + j);
			float32x4_t a1 = vld1q_f32(av + j + 4);
			float32x4_t a2 = vld1q_f32(av + j + 8);
			float32x4_t a3 = vld1q_f32(av + j + 12);
			float32x4_t a4 = vld1q_f32(av + j + 16);
			float32x4_t a5 = vld1q_f32(av + j + 20);
			float32x4_t a6 = vld1q_f32(av + j + 24);
			float32x4_t a7 = vld1q_f32(av + j + 28);
			float32x4_t q0, q1, q2, q3, q4, q5, q6, q7;
			vld1q_s8_to_f32x4(qs + j, &q0, &q1, &q2, &q3);
			vld1q_s8_to_f32x4(qs + j + 16, &q4, &q5, &q6, &q7);
			acc0 = vfmaq_f32(acc0, a0, q0);
			acc1 = vfmaq_f32(acc1, a1, q1);
			acc2 = vfmaq_f32(acc2, a2, q2);
			acc3 = vfmaq_f32(acc3, a3, q3);
			acc0 = vfmaq_f32(acc0, a4, q4);
			acc1 = vfmaq_f32(acc1, a5, q5);
			acc2 = vfmaq_f32(acc2, a6, q6);
			acc3 = vfmaq_f32(acc3, a7, q7);
		}
		for (; j + 16 <= n; j += 16) {
			float32x4_t a0 = vld1q_f32(av + j);
			float32x4_t a1 = vld1q_f32(av + j + 4);
			float32x4_t a2 = vld1q_f32(av + j + 8);
			float32x4_t a3 = vld1q_f32(av + j + 12);
			float32x4_t q0, q1, q2, q3;
			vld1q_s8_to_f32x4(qs + j, &q0, &q1, &q2, &q3);
			acc0 = vfmaq_f32(acc0, a0, q0);
			acc1 = vfmaq_f32(acc1, a1, q1);
			acc2 = vfmaq_f32(acc2, a2, q2);
			acc3 = vfmaq_f32(acc3, a3, q3);
		}
		for (; j + 8 <= n; j += 8) {
			float32x4_t a0 = vld1q_f32(av + j);
			float32x4_t a1 = vld1q_f32(av + j + 4);
			float32x4_t q0, q1;
			vld1_s8x8_to_f32x2(qs + j, &q0, &q1);
			acc0 = vfmaq_f32(acc0, a0, q0);
			acc1 = vfmaq_f32(acc1, a1, q1);
		}
		for (; j + 4 <= n; j += 4) {
			float32x4_t a0 = vld1q_f32(av + j);
			float32x4_t q0 = vld1_s8x4_to_f32(qs + j);
			acc0		   = vfmaq_f32(acc0, a0, q0);
		}
		float32x4_t acc01	= vaddq_f32(acc0, acc1);
		float32x4_t acc23	= vaddq_f32(acc2, acc3);
		float32x4_t acc		= vaddq_f32(acc01, acc23);
		float		partial = vaddvq_f32(acc);
		for (; j < n; j++)
			partial += av[j] * (float)qs[j];
		sum += partial * d;
	}
	return sum;
}

static void accum_v_q8_0(float *restrict out_h, const uint8_t *restrict block_ptr, float weight,
						 int head_dim) {
	int n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	for (int b = 0; b < n_blocks; b++) {
		const q8_0_block *blk = (const q8_0_block *)(block_ptr + ((size_t)b * KV_Q8_0_BLOCK_BYTES));
		float			  d	  = f16_to_f32(blk->d) * weight;
		int				  base = b * KV_Q8_0_BLOCK;
		int				  n	   = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;

		const int8_t *qs  = blk->qs;
		float		 *oh  = out_h + base;
		float32x4_t	  d_v = vdupq_n_f32(d);
		int			  j	  = 0;
		for (; j + 32 <= n; j += 32) {
			float32x4_t q0, q1, q2, q3, q4, q5, q6, q7;
			vld1q_s8_to_f32x4(qs + j, &q0, &q1, &q2, &q3);
			vld1q_s8_to_f32x4(qs + j + 16, &q4, &q5, &q6, &q7);
			vst1q_f32(oh + j, vfmaq_f32(vld1q_f32(oh + j), d_v, q0));
			vst1q_f32(oh + j + 4, vfmaq_f32(vld1q_f32(oh + j + 4), d_v, q1));
			vst1q_f32(oh + j + 8, vfmaq_f32(vld1q_f32(oh + j + 8), d_v, q2));
			vst1q_f32(oh + j + 12, vfmaq_f32(vld1q_f32(oh + j + 12), d_v, q3));
			vst1q_f32(oh + j + 16, vfmaq_f32(vld1q_f32(oh + j + 16), d_v, q4));
			vst1q_f32(oh + j + 20, vfmaq_f32(vld1q_f32(oh + j + 20), d_v, q5));
			vst1q_f32(oh + j + 24, vfmaq_f32(vld1q_f32(oh + j + 24), d_v, q6));
			vst1q_f32(oh + j + 28, vfmaq_f32(vld1q_f32(oh + j + 28), d_v, q7));
		}
		for (; j + 16 <= n; j += 16) {
			float32x4_t q0, q1, q2, q3;
			vld1q_s8_to_f32x4(qs + j, &q0, &q1, &q2, &q3);
			vst1q_f32(oh + j, vfmaq_f32(vld1q_f32(oh + j), d_v, q0));
			vst1q_f32(oh + j + 4, vfmaq_f32(vld1q_f32(oh + j + 4), d_v, q1));
			vst1q_f32(oh + j + 8, vfmaq_f32(vld1q_f32(oh + j + 8), d_v, q2));
			vst1q_f32(oh + j + 12, vfmaq_f32(vld1q_f32(oh + j + 12), d_v, q3));
		}
		for (; j + 8 <= n; j += 8) {
			float32x4_t q0, q1;
			vld1_s8x8_to_f32x2(qs + j, &q0, &q1);
			vst1q_f32(oh + j, vfmaq_f32(vld1q_f32(oh + j), d_v, q0));
			vst1q_f32(oh + j + 4, vfmaq_f32(vld1q_f32(oh + j + 4), d_v, q1));
		}
		for (; j + 4 <= n; j += 4) {
			float32x4_t q0 = vld1_s8x4_to_f32(qs + j);
			vst1q_f32(oh + j, vfmaq_f32(vld1q_f32(oh + j), d_v, q0));
		}
		for (; j < n; j++)
			oh[j] = fmaf((float)qs[j], d, oh[j]);
	}
}

static void cpu_attention_inner_q8_0(const uint8_t *restrict k_slice,
									 const uint8_t *restrict v_slice, int kv_stride,
									 const float *qh, float *out_h, int head_dim, int n_pos,
									 float scale, int flash_attn, float *restrict scores) {
	if (flash_attn) {
		float M = -INFINITY;
		float S = 0.0f;
		float VKQ[HEAD_DIM_MAX] __attribute__((aligned(64)));
		neon_zero_f32(VKQ, head_dim);
		for (int t = 0; t < n_pos; t++) {
			const uint8_t *kt = k_slice + ((size_t)t * kv_stride);
			float		   ss = dot8_q8_0(qh, kt, head_dim) * scale;

			float vs;
			float ms = 1.0f;
			if (ss > M) {
				float M_old = M;
				M			= ss;
				if (S > 0.0f) {
					ms				 = fast_expf(M_old - M);
					float32x4_t ms_v = vdupq_n_f32(ms);
					int			d	 = 0;
					for (; d + 32 <= head_dim; d += 32) {
						vst1q_f32(VKQ + d, vmulq_f32(vld1q_f32(VKQ + d), ms_v));
						vst1q_f32(VKQ + d + 4, vmulq_f32(vld1q_f32(VKQ + d + 4), ms_v));
						vst1q_f32(VKQ + d + 8, vmulq_f32(vld1q_f32(VKQ + d + 8), ms_v));
						vst1q_f32(VKQ + d + 12, vmulq_f32(vld1q_f32(VKQ + d + 12), ms_v));
						vst1q_f32(VKQ + d + 16, vmulq_f32(vld1q_f32(VKQ + d + 16), ms_v));
						vst1q_f32(VKQ + d + 20, vmulq_f32(vld1q_f32(VKQ + d + 20), ms_v));
						vst1q_f32(VKQ + d + 24, vmulq_f32(vld1q_f32(VKQ + d + 24), ms_v));
						vst1q_f32(VKQ + d + 28, vmulq_f32(vld1q_f32(VKQ + d + 28), ms_v));
					}
					for (; d + 16 <= head_dim; d += 16) {
						vst1q_f32(VKQ + d, vmulq_f32(vld1q_f32(VKQ + d), ms_v));
						vst1q_f32(VKQ + d + 4, vmulq_f32(vld1q_f32(VKQ + d + 4), ms_v));
						vst1q_f32(VKQ + d + 8, vmulq_f32(vld1q_f32(VKQ + d + 8), ms_v));
						vst1q_f32(VKQ + d + 12, vmulq_f32(vld1q_f32(VKQ + d + 12), ms_v));
					}
					for (; d + 4 <= head_dim; d += 4)
						vst1q_f32(VKQ + d, vmulq_f32(vld1q_f32(VKQ + d), ms_v));
					for (; d < head_dim; d++)
						VKQ[d] *= ms;
				}
				vs = 1.0f;
			} else {
				vs = fast_expf(ss - M);
			}

			const uint8_t *vt = v_slice + ((size_t)t * kv_stride);
			accum_v_q8_0(VKQ, vt, vs, head_dim);
			S = (S * ms) + vs;
		}

		float		S_inv = (S == 0.0f) ? 0.0f : 1.0f / S;
		float32x4_t inv_v = vdupq_n_f32(S_inv);
		int			d	  = 0;
		for (; d + 32 <= head_dim; d += 32) {
			vst1q_f32(out_h + d, vmulq_f32(vld1q_f32(VKQ + d), inv_v));
			vst1q_f32(out_h + d + 4, vmulq_f32(vld1q_f32(VKQ + d + 4), inv_v));
			vst1q_f32(out_h + d + 8, vmulq_f32(vld1q_f32(VKQ + d + 8), inv_v));
			vst1q_f32(out_h + d + 12, vmulq_f32(vld1q_f32(VKQ + d + 12), inv_v));
			vst1q_f32(out_h + d + 16, vmulq_f32(vld1q_f32(VKQ + d + 16), inv_v));
			vst1q_f32(out_h + d + 20, vmulq_f32(vld1q_f32(VKQ + d + 20), inv_v));
			vst1q_f32(out_h + d + 24, vmulq_f32(vld1q_f32(VKQ + d + 24), inv_v));
			vst1q_f32(out_h + d + 28, vmulq_f32(vld1q_f32(VKQ + d + 28), inv_v));
		}
		for (; d + 16 <= head_dim; d += 16) {
			vst1q_f32(out_h + d, vmulq_f32(vld1q_f32(VKQ + d), inv_v));
			vst1q_f32(out_h + d + 4, vmulq_f32(vld1q_f32(VKQ + d + 4), inv_v));
			vst1q_f32(out_h + d + 8, vmulq_f32(vld1q_f32(VKQ + d + 8), inv_v));
			vst1q_f32(out_h + d + 12, vmulq_f32(vld1q_f32(VKQ + d + 12), inv_v));
		}
		for (; d + 4 <= head_dim; d += 4)
			vst1q_f32(out_h + d, vmulq_f32(vld1q_f32(VKQ + d), inv_v));
		for (; d < head_dim; d++)
			out_h[d] = VKQ[d] * S_inv;
		return;
	}

	for (int t = 0; t < n_pos; t++) {
		const uint8_t *kt = k_slice + ((size_t)t * kv_stride);
		scores[t]		  = dot8_q8_0(qh, kt, head_dim) * scale;
	}
	softmax_masked(scores, n_pos);

	if (n_pos == 0) {
		float32x4_t zero = vdupq_n_f32(0.0f);
		int			d	 = 0;
		for (; d + 32 <= head_dim; d += 32) {
			vst1q_f32(out_h + d, zero);
			vst1q_f32(out_h + d + 4, zero);
			vst1q_f32(out_h + d + 8, zero);
			vst1q_f32(out_h + d + 12, zero);
			vst1q_f32(out_h + d + 16, zero);
			vst1q_f32(out_h + d + 20, zero);
			vst1q_f32(out_h + d + 24, zero);
			vst1q_f32(out_h + d + 28, zero);
		}
		for (; d + 16 <= head_dim; d += 16) {
			vst1q_f32(out_h + d, zero);
			vst1q_f32(out_h + d + 4, zero);
			vst1q_f32(out_h + d + 8, zero);
			vst1q_f32(out_h + d + 12, zero);
		}
		for (; d + 4 <= head_dim; d += 4)
			vst1q_f32(out_h + d, zero);
		for (; d < head_dim; d++)
			out_h[d] = 0.0f;
		return;
	}

	memset(out_h, 0, (size_t)head_dim * sizeof(float));
	for (int t = 0; t < n_pos; t++) {
		const uint8_t *vt = v_slice + ((size_t)t * kv_stride);
		accum_v_q8_0(out_h, vt, scores[t], head_dim);
	}
}

static void cpu_attn_head_chunk_neon(int begin, int end, int tid, void *ctx) {
	cpu_attn_job_neon *j = ctx;
	float			  *scores;
	if (tid == 0) {
		scores = j->p->scores;
	} else {
		cpu_thread_scratch *ts = &j->p->thread_scratch[tid];
		if (ts->scores_cap < j->n_pos) {
			free(ts->scores);
			ts->scores	   = xmalloc((size_t)j->n_pos * sizeof(float));
			ts->scores_cap = j->n_pos;
		}
		scores = ts->scores;
	}
	for (int h = begin; h < end; h++) {
		int			 kvh   = h / j->n_groups;
		const float *qh	   = j->qf + ((size_t)h * j->head_dim);
		float		*out_h = j->outf + ((size_t)h * j->head_dim);
		if (j->kv_quant == KV_QUANT_Q8_0) {
			const uint8_t *k_slice = (const uint8_t *)j->kl_base + ((size_t)kvh * j->kvh_stride);
			const uint8_t *v_slice = (const uint8_t *)j->vl_base + ((size_t)kvh * j->kvh_stride);
			cpu_attention_inner_q8_0(k_slice, v_slice, j->hd_stride, qh, out_h, j->head_dim,
									 j->n_pos, j->scale, j->flash_attn, scores);
		} else {
			const uint16_t *k_slice = j->kl_base + ((size_t)kvh * j->kvh_stride);
			const uint16_t *v_slice = j->vl_base + ((size_t)kvh * j->kvh_stride);
			cpu_attention_inner((uint16_t *)k_slice, (uint16_t *)v_slice, j->hd_stride, qh, out_h,
								j->head_dim, j->n_pos, j->scale, j->flash_attn, scores);
		}
	}
}

status_code cpu_attention_impl(backend *self, const buffer *q, const buffer *k_cache,
							   const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
							   int n_kv_heads, int head_dim, int n_ctx, int flash_attn, float scale,
							   int sliding_window, int n_kv_heads_active) {
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	if (n_active <= 0)
		return ERR_INVALID_ARG;

	int			 n_groups = (n_heads + n_active - 1) / n_active;
	const float *qf		  = (const float *)cpu_ptr(q);
	float		*outf	  = (float *)cpu_ptr(out);
	(void)k_cache;
	(void)v_cache;

	int hd_stride_elems = head_dim;

	int attn_start = 0;
	int n_pos	   = pos + 1;
	if (sliding_window > 0 && n_pos > sliding_window) {
		attn_start = n_pos - sliding_window;
		n_pos	   = sliding_window;
	}

	if (p->kv_quant == KV_QUANT_Q8_0) {
		size_t n_blocks		= ((size_t)hd_stride_elems + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		size_t elem_stride	= n_blocks * KV_Q8_0_BLOCK_BYTES;
		size_t layer_stride = (size_t)n_kv_heads * n_ctx * elem_stride;
		size_t kvh_stride	= (size_t)n_ctx * elem_stride;
		size_t layer_base =
			p->kv_layer_off ? p->kv_layer_off[layer] : ((size_t)layer * layer_stride);
		const uint8_t *kl_base =
			(const uint8_t *)p->kv_k + layer_base + ((size_t)attn_start * elem_stride);
		const uint8_t *vl_base =
			(const uint8_t *)p->kv_v + layer_base + ((size_t)attn_start * elem_stride);

		if (p->pool && p->thread_scratch && n_heads > 1 &&
			(size_t)n_heads * (size_t)n_pos * (size_t)head_dim >= 4096) {
			cpu_attn_job_neon job = {.kl_base	 = (const uint16_t *)kl_base,
									 .vl_base	 = (const uint16_t *)vl_base,
									 .qf		 = qf,
									 .outf		 = outf,
									 .n_groups	 = n_groups,
									 .head_dim	 = head_dim,
									 .hd_stride	 = (int)elem_stride,
									 .n_pos		 = n_pos,
									 .flash_attn = flash_attn,
									 .scale		 = scale,
									 .kvh_stride = kvh_stride,
									 .p			 = p,
									 .kv_quant	 = KV_QUANT_Q8_0};
			tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_head_chunk_neon, &job);
			return OK;
		}

		float *scores = p->scores;
		for (int kvh = 0; kvh < n_active; kvh++) {
			const uint8_t *k_slice = kl_base + ((size_t)kvh * kvh_stride);
			const uint8_t *v_slice = vl_base + ((size_t)kvh * kvh_stride);
			for (int hg = 0; hg < n_groups; hg++) {
				int			 h	   = (kvh * n_groups) + hg;
				const float *qh	   = qf + ((size_t)h * head_dim);
				float		*out_h = outf + ((size_t)h * head_dim);
				cpu_attention_inner_q8_0(k_slice, v_slice, (int)elem_stride, qh, out_h, head_dim,
										 n_pos, scale, flash_attn, scores);
			}
		}
		return OK;
	}

	int	   hd_stride	= hd_stride_elems;
	size_t layer_stride = (size_t)n_kv_heads * n_ctx * hd_stride;
	size_t kvh_stride	= (size_t)n_ctx * hd_stride;
	size_t layer_base	= p->kv_layer_off ? p->kv_layer_off[layer] / sizeof(uint16_t)
										  : ((size_t)layer * layer_stride);

	uint16_t *kl_base = p->kv_k + layer_base;
	uint16_t *vl_base = p->kv_v + layer_base;
	kl_base += (size_t)attn_start * hd_stride;
	vl_base += (size_t)attn_start * hd_stride;

	if (p->pool && p->thread_scratch && n_heads > 1 &&
		(size_t)n_heads * (size_t)n_pos * (size_t)head_dim >= 4096) {
		cpu_attn_job_neon job = {.kl_base	 = kl_base,
								 .vl_base	 = vl_base,
								 .qf		 = qf,
								 .outf		 = outf,
								 .n_groups	 = n_groups,
								 .head_dim	 = head_dim,
								 .hd_stride	 = hd_stride,
								 .n_pos		 = n_pos,
								 .flash_attn = flash_attn,
								 .scale		 = scale,
								 .kvh_stride = kvh_stride,
								 .p			 = p,
								 .kv_quant	 = KV_QUANT_F16};
		tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_head_chunk_neon, &job);
		return OK;
	}

	float *scores = p->scores;
	for (int kvh = 0; kvh < n_active; kvh++) {
		uint16_t *k_slice = kl_base + ((size_t)kvh * kvh_stride);
		uint16_t *v_slice = vl_base + ((size_t)kvh * kvh_stride);

		for (int hg = 0; hg < n_groups; hg++) {
			int			 h	   = (kvh * n_groups) + hg;
			const float *qh	   = qf + ((size_t)h * head_dim);
			float		*out_h = outf + ((size_t)h * head_dim);
			cpu_attention_inner(k_slice, v_slice, hd_stride, qh, out_h, head_dim, n_pos, scale,
								flash_attn, scores);
		}
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

static void cpu_attn_batch_chunk_neon(int begin, int end, int tid, void *ctx) {
	cpu_attn_batch_job_neon *j = ctx;
	float					*scores;
	if (tid == 0) {
		scores = j->p->scores;
	} else {
		cpu_thread_scratch *ts	 = &j->p->thread_scratch[tid];
		int					need = j->pos_start + j->m;
		if (ts->scores_cap < need) {
			free(ts->scores);
			ts->scores	   = xmalloc((size_t)need * sizeof(float));
			ts->scores_cap = need;
		}
		scores = ts->scores;
	}

	for (int idx = begin; idx < end; idx++) {
		int dispatch_row = idx / j->n_heads;
		int h			 = idx % j->n_heads;

		int row = j->bitrev_perm ? j->bitrev_perm[dispatch_row] : dispatch_row;
		if (row >= j->m)
			continue;

		int pos		   = j->pos_start + row;
		int n_pos	   = pos + 1;
		int attn_start = 0;
		if (j->sliding_window > 0 && n_pos > j->sliding_window) {
			attn_start = n_pos - j->sliding_window;
			n_pos	   = j->sliding_window;
		}

		int			 kvh   = h / j->n_groups;
		const float *qh	   = j->qf + ((((size_t)row * j->n_heads) + h) * j->head_dim);
		float		*out_h = j->outf + ((((size_t)row * j->n_heads) + h) * j->head_dim);
		if (j->kv_quant == KV_QUANT_Q8_0) {
			const uint8_t *k_slice = (const uint8_t *)j->kl_base + ((size_t)kvh * j->kvh_stride) +
									 ((size_t)attn_start * (size_t)j->hd_stride);
			const uint8_t *v_slice = (const uint8_t *)j->vl_base + ((size_t)kvh * j->kvh_stride) +
									 ((size_t)attn_start * (size_t)j->hd_stride);
			cpu_attention_inner_q8_0(k_slice, v_slice, j->hd_stride, qh, out_h, j->head_dim, n_pos,
									 j->scale, j->flash_attn, scores);
		} else {
			const uint16_t *k_slice =
				j->kl_base + ((size_t)kvh * j->kvh_stride) + ((size_t)attn_start * j->hd_stride);
			const uint16_t *v_slice =
				j->vl_base + ((size_t)kvh * j->kvh_stride) + ((size_t)attn_start * j->hd_stride);
			cpu_attention_inner((uint16_t *)k_slice, (uint16_t *)v_slice, j->hd_stride, qh, out_h,
								j->head_dim, n_pos, j->scale, j->flash_attn, scores);
		}
	}
}

status_code cpu_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
								const buffer *v_cache, buffer *out, int layer, int pos_start,
								int n_heads, int n_kv_heads, int head_dim, int n_ctx,
								int flash_attn, float scale, int n_kv_heads_active, int m) {
	(void)k_cache;
	(void)v_cache;
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	if (n_active <= 0)
		return ERR_INVALID_ARG;
	int n_groups = (n_heads + n_active - 1) / n_active;

	int			hd_stride;
	size_t		layer_stride;
	size_t		kvh_stride;
	const void *kl_base_raw;
	const void *vl_base_raw;
	if (p->kv_quant == KV_QUANT_Q8_0) {
		size_t n_blocks = ((size_t)head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		hd_stride		= (int)(n_blocks * KV_Q8_0_BLOCK_BYTES);
		layer_stride	= (size_t)n_kv_heads * n_ctx * (size_t)hd_stride;
		kvh_stride		= (size_t)n_ctx * (size_t)hd_stride;
		size_t layer_base =
			p->kv_layer_off ? p->kv_layer_off[layer] : ((size_t)layer * layer_stride);
		kl_base_raw = (const uint8_t *)p->kv_k + layer_base;
		vl_base_raw = (const uint8_t *)p->kv_v + layer_base;
	} else {
		hd_stride		  = head_dim;
		layer_stride	  = (size_t)n_kv_heads * n_ctx * hd_stride;
		kvh_stride		  = (size_t)n_ctx * hd_stride;
		size_t layer_base = p->kv_layer_off ? p->kv_layer_off[layer] / sizeof(uint16_t)
											: ((size_t)layer * layer_stride);
		kl_base_raw		  = p->kv_k + layer_base;
		vl_base_raw		  = p->kv_v + layer_base;
	}

	int use_bitrev = (m >= ATTN_BITREV_MIN_M);
	int m_pow2	   = 1;
	while (m_pow2 < m)
		m_pow2 <<= 1;

	int	 bitrev_stack[ATTN_BITREV_STACK_MAX];
	int *bitrev_perm = NULL;
	if (use_bitrev) {
		if (m_pow2 <= ATTN_BITREV_STACK_MAX) {
			bitrev_perm = bitrev_stack;
		} else {
			bitrev_perm = xmalloc((size_t)m_pow2 * sizeof(int));
		}
		unsigned bits = 0;
		while ((1u << bits) < (unsigned)m_pow2)
			bits++;
		for (int r = 0; r < m_pow2; r++) {
			unsigned rev = 0, tmp = (unsigned)r;
			for (unsigned b = 0; b < bits; b++) {
				rev = (rev << 1) | (tmp & 1u);
				tmp >>= 1;
			}
			bitrev_perm[r] = (int)rev;
		}
	}

	cpu_attn_batch_job_neon job = {.kl_base		= kl_base_raw,
								   .vl_base		= vl_base_raw,
								   .qf			= (const float *)cpu_ptr(q),
								   .outf		= (float *)cpu_ptr(out),
								   .n_groups	= n_groups,
								   .head_dim	= head_dim,
								   .hd_stride	= hd_stride,
								   .n_heads		= n_heads,
								   .pos_start	= pos_start,
								   .m			= m,
								   .flash_attn	= flash_attn,
								   .scale		= scale,
								   .kvh_stride	= kvh_stride,
								   .p			= p,
								   .bitrev_perm = bitrev_perm,
								   .kv_quant	= p->kv_quant};

	int total		= use_bitrev ? (n_heads * m_pow2) : (n_heads * m);
	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);
	if (can_recurse && p->pool && p->thread_scratch && total >= 2) {
		tpool_parallel_for(p->pool, total, 1, cpu_attn_batch_chunk_neon, &job);
	} else {
		cpu_attn_batch_chunk_neon(0, total, 0, &job);
	}

	if (bitrev_perm != bitrev_stack && bitrev_perm != NULL)
		free(bitrev_perm);

	return OK;
}

static void cpu_add_inplace_chunk_neon(int begin, int end, int tid, void *ctx) {
	(void)tid;
	float **pp				 = ctx;
	float *restrict xf		 = pp[0];
	const float *restrict yf = pp[1];
	int i					 = begin;
	int n					 = end;
	for (; i + 16 <= n; i += 16) {
		vst1q_f32(xf + i, vaddq_f32(vld1q_f32(xf + i), vld1q_f32(yf + i)));
		vst1q_f32(xf + i + 4, vaddq_f32(vld1q_f32(xf + i + 4), vld1q_f32(yf + i + 4)));
		vst1q_f32(xf + i + 8, vaddq_f32(vld1q_f32(xf + i + 8), vld1q_f32(yf + i + 8)));
		vst1q_f32(xf + i + 12, vaddq_f32(vld1q_f32(xf + i + 12), vld1q_f32(yf + i + 12)));
	}
	for (; i + 4 <= n; i += 4)
		vst1q_f32(xf + i, vaddq_f32(vld1q_f32(xf + i), vld1q_f32(yf + i)));
	for (; i < n; i++)
		xf[i] += yf[i];
}

status_code cpu_add_inplace(backend *self, buffer *x, const buffer *y, int n) {
	cpu_priv *p		  = self->priv;
	float	 *args[2] = {cpu_ptr(x), (float *)cpu_ptr(y)};
	if (p->pool && n >= 2 * CPU_ELEMWISE_MIN_PER_THREAD) {
		tpool_parallel_for(p->pool, n, CPU_ELEMWISE_MIN_PER_THREAD, cpu_add_inplace_chunk_neon,
						   args);
	} else {
		cpu_add_inplace_chunk_neon(0, n, 0, args);
	}
	return OK;
}

static void cpu_add_batch_chunk_neon(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_add_batch_job *j = ctx;
	int				   n = j->n;
	for (int row = begin; row < end; row++) {
		float		*xr		 = j->x + ((size_t)row * n);
		const float *yr		 = j->y + ((size_t)row * n);
		float		*args[2] = {xr, (float *)yr};
		cpu_add_inplace_chunk_neon(0, n, 0, args);
	}
}

status_code cpu_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
	cpu_priv		 *p	  = self->priv;
	cpu_add_batch_job job = {.x = cpu_ptr(x), .y = cpu_ptr(y), .n = n};
	cpu_run_batch(p->pool, m, cpu_add_batch_chunk_neon, &job);
	return OK;
}

static void cpu_ffn_silu_chunk_neon(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_act_args *a		= ctx;
	const float *restrict g = a->g;
	const float *restrict u = a->u;
	float *restrict o		= a->o;
	int			i			= begin;
	int			n			= end;
	float32x4_t one			= vdupq_n_f32(1.0f);
	for (; i + 16 <= n; i += 16) {
		float32x4_t g0 = vld1q_f32(g + i);
		float32x4_t g1 = vld1q_f32(g + i + 4);
		float32x4_t g2 = vld1q_f32(g + i + 8);
		float32x4_t g3 = vld1q_f32(g + i + 12);
		float32x4_t e0 = vexpq_f32(vnegq_f32(g0));
		float32x4_t e1 = vexpq_f32(vnegq_f32(g1));
		float32x4_t e2 = vexpq_f32(vnegq_f32(g2));
		float32x4_t e3 = vexpq_f32(vnegq_f32(g3));
		float32x4_t d0 = vaddq_f32(one, e0);
		float32x4_t d1 = vaddq_f32(one, e1);
		float32x4_t d2 = vaddq_f32(one, e2);
		float32x4_t d3 = vaddq_f32(one, e3);
		float32x4_t s0 = vrecpeq_f32(d0);
		float32x4_t s1 = vrecpeq_f32(d1);
		float32x4_t s2 = vrecpeq_f32(d2);
		float32x4_t s3 = vrecpeq_f32(d3);
		s0			   = vmulq_f32(vrecpsq_f32(d0, s0), s0);
		s1			   = vmulq_f32(vrecpsq_f32(d1, s1), s1);
		s2			   = vmulq_f32(vrecpsq_f32(d2, s2), s2);
		s3			   = vmulq_f32(vrecpsq_f32(d3, s3), s3);
		s0			   = vmulq_f32(vrecpsq_f32(d0, s0), s0);
		s1			   = vmulq_f32(vrecpsq_f32(d1, s1), s1);
		s2			   = vmulq_f32(vrecpsq_f32(d2, s2), s2);
		s3			   = vmulq_f32(vrecpsq_f32(d3, s3), s3);
		vst1q_f32(o + i, vmulq_f32(vmulq_f32(g0, s0), vld1q_f32(u + i)));
		vst1q_f32(o + i + 4, vmulq_f32(vmulq_f32(g1, s1), vld1q_f32(u + i + 4)));
		vst1q_f32(o + i + 8, vmulq_f32(vmulq_f32(g2, s2), vld1q_f32(u + i + 8)));
		vst1q_f32(o + i + 12, vmulq_f32(vmulq_f32(g3, s3), vld1q_f32(u + i + 12)));
	}
	for (; i + 4 <= n; i += 4) {
		float32x4_t g0 = vld1q_f32(g + i);
		float32x4_t e0 = vexpq_f32(vnegq_f32(g0));
		float32x4_t d0 = vaddq_f32(one, e0);
		float32x4_t s0 = vrecpeq_f32(d0);
		s0			   = vmulq_f32(vrecpsq_f32(d0, s0), s0);
		s0			   = vmulq_f32(vrecpsq_f32(d0, s0), s0);
		vst1q_f32(o + i, vmulq_f32(vmulq_f32(g0, s0), vld1q_f32(u + i)));
	}
	for (; i < n; i++) {
		float gv = g[i];
		o[i]	 = gv / (1.0f + expf(-gv)) * u[i];
	}
}

status_code cpu_ffn_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
							 int n) {
	cpu_priv		*p = self->priv;
	cpu_ffn_act_args a = {.g = cpu_ptr(gate), .u = cpu_ptr(up), .o = cpu_ptr(out)};
	if (p->pool && n >= 2 * CPU_ELEMWISE_MIN_PER_THREAD) {
		tpool_parallel_for(p->pool, n, CPU_ELEMWISE_MIN_PER_THREAD, cpu_ffn_silu_chunk_neon, &a);
	} else {
		cpu_ffn_silu_chunk_neon(0, n, 0, &a);
	}
	return OK;
}

static void cpu_ffn_gelu_chunk_neon(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_act_args *a		= ctx;
	const float *restrict g = a->g;
	const float *restrict u = a->u;
	float *restrict o		= a->o;
	const float c_fit		= 0.7978845608028654f;
	const float c_x3		= 0.044715f;
	int			i			= begin;
	int			n			= end;
	float32x4_t half		= vdupq_n_f32(0.5f);
	float32x4_t one			= vdupq_n_f32(1.0f);
	float32x4_t c_v			= vdupq_n_f32(c_fit);
	for (; i + 16 <= n; i += 16) {
		float32x4_t g0	 = vld1q_f32(g + i);
		float32x4_t g1	 = vld1q_f32(g + i + 4);
		float32x4_t g2	 = vld1q_f32(g + i + 8);
		float32x4_t g3	 = vld1q_f32(g + i + 12);
		float32x4_t g0sq = vmulq_f32(g0, g0);
		float32x4_t g1sq = vmulq_f32(g1, g1);
		float32x4_t g2sq = vmulq_f32(g2, g2);
		float32x4_t g3sq = vmulq_f32(g3, g3);
		float32x4_t g0c	 = vmulq_f32(g0, vfmaq_n_f32(one, g0sq, c_x3));
		float32x4_t g1c	 = vmulq_f32(g1, vfmaq_n_f32(one, g1sq, c_x3));
		float32x4_t g2c	 = vmulq_f32(g2, vfmaq_n_f32(one, g2sq, c_x3));
		float32x4_t g3c	 = vmulq_f32(g3, vfmaq_n_f32(one, g3sq, c_x3));
		float32x4_t in0	 = vmulq_f32(c_v, g0c);
		float32x4_t in1	 = vmulq_f32(c_v, g1c);
		float32x4_t in2	 = vmulq_f32(c_v, g2c);
		float32x4_t in3	 = vmulq_f32(c_v, g3c);
		float32x4_t t0	 = vaddq_f32(one, vtanhq_f32(in0));
		float32x4_t t1	 = vaddq_f32(one, vtanhq_f32(in1));
		float32x4_t t2	 = vaddq_f32(one, vtanhq_f32(in2));
		float32x4_t t3	 = vaddq_f32(one, vtanhq_f32(in3));
		vst1q_f32(o + i, vmulq_f32(vmulq_f32(vmulq_f32(half, g0), t0), vld1q_f32(u + i)));
		vst1q_f32(o + i + 4, vmulq_f32(vmulq_f32(vmulq_f32(half, g1), t1), vld1q_f32(u + i + 4)));
		vst1q_f32(o + i + 8, vmulq_f32(vmulq_f32(vmulq_f32(half, g2), t2), vld1q_f32(u + i + 8)));
		vst1q_f32(o + i + 12, vmulq_f32(vmulq_f32(vmulq_f32(half, g3), t3), vld1q_f32(u + i + 12)));
	}
	for (; i < n; i++) {
		float x		= g[i];
		float x3	= x * x * x;
		float inner = c_fit * (x + (c_x3 * x3));
		float t		= tanhf(inner);
		o[i]		= 0.5f * x * (1.0f + t) * u[i];
	}
}

void cpu_ffn_down_act_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_down_act_args *a = ctx;
	const float *restrict g	 = a->g;
	const float *restrict u	 = a->u;
	float *restrict o		 = a->o;
	int			i			 = begin;
	int			n			 = end;
	float32x4_t one			 = vdupq_n_f32(1.0f);
	if (a->activation == 1) {
		const float c_fit = 0.7978845608028654f;
		const float c_x3  = 0.044715f;
		float32x4_t half  = vdupq_n_f32(0.5f);
		float32x4_t c_v	  = vdupq_n_f32(c_fit);
		for (; i + 16 <= n; i += 16) {
			float32x4_t g0	 = vld1q_f32(g + i);
			float32x4_t g1	 = vld1q_f32(g + i + 4);
			float32x4_t g2	 = vld1q_f32(g + i + 8);
			float32x4_t g3	 = vld1q_f32(g + i + 12);
			float32x4_t g0sq = vmulq_f32(g0, g0);
			float32x4_t g1sq = vmulq_f32(g1, g1);
			float32x4_t g2sq = vmulq_f32(g2, g2);
			float32x4_t g3sq = vmulq_f32(g3, g3);
			float32x4_t g0c	 = vmulq_f32(g0, vfmaq_n_f32(one, g0sq, c_x3));
			float32x4_t g1c	 = vmulq_f32(g1, vfmaq_n_f32(one, g1sq, c_x3));
			float32x4_t g2c	 = vmulq_f32(g2, vfmaq_n_f32(one, g2sq, c_x3));
			float32x4_t g3c	 = vmulq_f32(g3, vfmaq_n_f32(one, g3sq, c_x3));
			float32x4_t in0	 = vmulq_f32(c_v, g0c);
			float32x4_t in1	 = vmulq_f32(c_v, g1c);
			float32x4_t in2	 = vmulq_f32(c_v, g2c);
			float32x4_t in3	 = vmulq_f32(c_v, g3c);
			float32x4_t t0	 = vaddq_f32(one, vtanhq_f32(in0));
			float32x4_t t1	 = vaddq_f32(one, vtanhq_f32(in1));
			float32x4_t t2	 = vaddq_f32(one, vtanhq_f32(in2));
			float32x4_t t3	 = vaddq_f32(one, vtanhq_f32(in3));
			vst1q_f32(o + i, vmulq_f32(vmulq_f32(vmulq_f32(half, g0), t0), vld1q_f32(u + i)));
			vst1q_f32(o + i + 4,
					  vmulq_f32(vmulq_f32(vmulq_f32(half, g1), t1), vld1q_f32(u + i + 4)));
			vst1q_f32(o + i + 8,
					  vmulq_f32(vmulq_f32(vmulq_f32(half, g2), t2), vld1q_f32(u + i + 8)));
			vst1q_f32(o + i + 12,
					  vmulq_f32(vmulq_f32(vmulq_f32(half, g3), t3), vld1q_f32(u + i + 12)));
		}
		for (; i < n; i++) {
			float x		= g[i];
			float x3	= x * x * x;
			float inner = c_fit * (x + (c_x3 * x3));
			float t		= tanhf(inner);
			o[i]		= 0.5f * x * (1.0f + t) * u[i];
		}
		return;
	}
	for (; i + 16 <= n; i += 16) {
		float32x4_t g0 = vld1q_f32(g + i);
		float32x4_t g1 = vld1q_f32(g + i + 4);
		float32x4_t g2 = vld1q_f32(g + i + 8);
		float32x4_t g3 = vld1q_f32(g + i + 12);
		float32x4_t e0 = vexpq_f32(vnegq_f32(g0));
		float32x4_t e1 = vexpq_f32(vnegq_f32(g1));
		float32x4_t e2 = vexpq_f32(vnegq_f32(g2));
		float32x4_t e3 = vexpq_f32(vnegq_f32(g3));
		float32x4_t d0 = vaddq_f32(one, e0);
		float32x4_t d1 = vaddq_f32(one, e1);
		float32x4_t d2 = vaddq_f32(one, e2);
		float32x4_t d3 = vaddq_f32(one, e3);
		float32x4_t s0 = vrecpeq_f32(d0);
		float32x4_t s1 = vrecpeq_f32(d1);
		float32x4_t s2 = vrecpeq_f32(d2);
		float32x4_t s3 = vrecpeq_f32(d3);
		s0			   = vmulq_f32(vrecpsq_f32(d0, s0), s0);
		s1			   = vmulq_f32(vrecpsq_f32(d1, s1), s1);
		s2			   = vmulq_f32(vrecpsq_f32(d2, s2), s2);
		s3			   = vmulq_f32(vrecpsq_f32(d3, s3), s3);
		s0			   = vmulq_f32(vrecpsq_f32(d0, s0), s0);
		s1			   = vmulq_f32(vrecpsq_f32(d1, s1), s1);
		s2			   = vmulq_f32(vrecpsq_f32(d2, s2), s2);
		s3			   = vmulq_f32(vrecpsq_f32(d3, s3), s3);
		vst1q_f32(o + i, vmulq_f32(vmulq_f32(g0, s0), vld1q_f32(u + i)));
		vst1q_f32(o + i + 4, vmulq_f32(vmulq_f32(g1, s1), vld1q_f32(u + i + 4)));
		vst1q_f32(o + i + 8, vmulq_f32(vmulq_f32(g2, s2), vld1q_f32(u + i + 8)));
		vst1q_f32(o + i + 12, vmulq_f32(vmulq_f32(g3, s3), vld1q_f32(u + i + 12)));
	}
	for (; i + 4 <= n; i += 4) {
		float32x4_t g0 = vld1q_f32(g + i);
		float32x4_t e0 = vexpq_f32(vnegq_f32(g0));
		float32x4_t d0 = vaddq_f32(one, e0);
		float32x4_t s0 = vrecpeq_f32(d0);
		s0			   = vmulq_f32(vrecpsq_f32(d0, s0), s0);
		s0			   = vmulq_f32(vrecpsq_f32(d0, s0), s0);
		vst1q_f32(o + i, vmulq_f32(vmulq_f32(g0, s0), vld1q_f32(u + i)));
	}
	for (; i < n; i++) {
		float gv = g[i];
		o[i]	 = gv / (1.0f + expf(-gv)) * u[i];
	}
}

status_code cpu_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up, buffer *out,
								int n, int activation) {
	cpu_priv		*p	= self->priv;
	cpu_ffn_act_args a	= {.g = cpu_ptr(gate), .u = cpu_ptr(up), .o = cpu_ptr(out)};
	tpool_chunk_fn	 fn = activation == 1 ? cpu_ffn_gelu_chunk_neon : cpu_ffn_silu_chunk_neon;
	if (p->pool && n >= 2 * CPU_ELEMWISE_MIN_PER_THREAD) {
		tpool_parallel_for(p->pool, n, CPU_ELEMWISE_MIN_PER_THREAD, fn, &a);
	} else {
		fn(0, n, 0, &a);
	}
	return OK;
}

static void cpu_ffn_act_batch_chunk_neon(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_act_batch_job *j	  = ctx;
	int					   n	  = j->n;
	size_t				   stride = j->fused_stride ? (size_t)j->fused_stride : (size_t)n;
	tpool_chunk_fn fn = (j->activation == 1) ? cpu_ffn_gelu_chunk_neon : cpu_ffn_silu_chunk_neon;
	for (int row = begin; row < end; row++) {
		cpu_ffn_act_args a = {.g = j->g + ((size_t)row * stride),
							  .u = j->u + ((size_t)row * stride),
							  .o = j->o + ((size_t)row * n)};
		fn(0, n, 0, &a);
	}
}

status_code cpu_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n, int activation, int m) {
	cpu_priv			 *p	  = self->priv;
	cpu_ffn_act_batch_job job = {
		.g = cpu_ptr(gate), .u = cpu_ptr(up), .o = cpu_ptr(out), .n = n, .activation = activation};
	cpu_run_batch(p->pool, m, cpu_ffn_act_batch_chunk_neon, &job);
	return OK;
}

status_code cpu_ffn_activate_fused_batch(backend *self, const buffer *fused, buffer *out, int n,
										 int activation, int m) {
	cpu_priv			 *p	 = self->priv;
	const float			 *fp = cpu_ptr(fused);
	cpu_ffn_act_batch_job job = {.g			   = fp,
								 .u			   = fp + n,
								 .o			   = cpu_ptr(out),
								 .n			   = n,
								 .activation   = activation,
								 .fused_stride = 2 * n};
	cpu_run_batch(p->pool, m, cpu_ffn_act_batch_chunk_neon, &job);
	return OK;
}

status_code cpu_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx) {
	(void)self;
	const float	   *lp		= cpu_ptr(logits);
	float32x4_t		best_v0 = vdupq_n_f32(-INFINITY);
	float32x4_t		best_v1 = vdupq_n_f32(-INFINITY);
	float32x4_t		best_v2 = vdupq_n_f32(-INFINITY);
	float32x4_t		best_v3 = vdupq_n_f32(-INFINITY);
	int32x4_t		best_i0 = vdupq_n_s32(0);
	int32x4_t		best_i1 = vdupq_n_s32(0);
	int32x4_t		best_i2 = vdupq_n_s32(0);
	int32x4_t		best_i3 = vdupq_n_s32(0);
	int32x4_t		idx0	= {0, 1, 2, 3};
	int32x4_t		idx1	= {4, 5, 6, 7};
	int32x4_t		idx2	= {8, 9, 10, 11};
	int32x4_t		idx3	= {12, 13, 14, 15};
	const int32x4_t stride	= vdupq_n_s32(16);
	int				i		= 0;
	for (; i + 16 <= n; i += 16) {
		float32x4_t v0 = vld1q_f32(lp + i);
		float32x4_t v1 = vld1q_f32(lp + i + 4);
		float32x4_t v2 = vld1q_f32(lp + i + 8);
		float32x4_t v3 = vld1q_f32(lp + i + 12);
		uint32x4_t	m0 = vcgtq_f32(v0, best_v0);
		uint32x4_t	m1 = vcgtq_f32(v1, best_v1);
		uint32x4_t	m2 = vcgtq_f32(v2, best_v2);
		uint32x4_t	m3 = vcgtq_f32(v3, best_v3);
		best_v0		   = vbslq_f32(m0, v0, best_v0);
		best_v1		   = vbslq_f32(m1, v1, best_v1);
		best_v2		   = vbslq_f32(m2, v2, best_v2);
		best_v3		   = vbslq_f32(m3, v3, best_v3);
		best_i0		   = vbslq_s32(m0, idx0, best_i0);
		best_i1		   = vbslq_s32(m1, idx1, best_i1);
		best_i2		   = vbslq_s32(m2, idx2, best_i2);
		best_i3		   = vbslq_s32(m3, idx3, best_i3);
		idx0		   = vaddq_s32(idx0, stride);
		idx1		   = vaddq_s32(idx1, stride);
		idx2		   = vaddq_s32(idx2, stride);
		idx3		   = vaddq_s32(idx3, stride);
	}
	uint32x4_t	m01	  = vcgtq_f32(best_v1, best_v0);
	float32x4_t bv01  = vbslq_f32(m01, best_v1, best_v0);
	int32x4_t	bi01  = vbslq_s32(m01, best_i1, best_i0);
	uint32x4_t	m23	  = vcgtq_f32(best_v3, best_v2);
	float32x4_t bv23  = vbslq_f32(m23, best_v3, best_v2);
	int32x4_t	bi23  = vbslq_s32(m23, best_i3, best_i2);
	uint32x4_t	m0123 = vcgtq_f32(bv23, bv01);
	float32x4_t bv	  = vbslq_f32(m0123, bv23, bv01);
	int32x4_t	bi	  = vbslq_s32(m0123, bi23, bi01);

	float	vals[4];
	int32_t idxs[4];
	vst1q_f32(vals, bv);
	vst1q_s32(idxs, bi);
	float bestv = vals[0];
	int	  best	= idxs[0];
	if (vals[1] > bestv) {
		bestv = vals[1];
		best  = idxs[1];
	}
	if (vals[2] > bestv) {
		bestv = vals[2];
		best  = idxs[2];
	}
	if (vals[3] > bestv) {
		bestv = vals[3];
		best  = idxs[3];
	}
	for (; i < n; i++) {
		if (lp[i] > bestv) {
			bestv = lp[i];
			best  = i;
		}
	}
	*out_idx = best;
	return OK;
}

int32_t cpu_argmax_f32(const float *restrict logits, int vocab) {
	if (vocab <= 0)
		return 0;
	float32x4_t		best_v0 = vdupq_n_f32(-INFINITY);
	float32x4_t		best_v1 = vdupq_n_f32(-INFINITY);
	int32x4_t		best_i0 = vdupq_n_s32(0);
	int32x4_t		best_i1 = vdupq_n_s32(0);
	int32x4_t		idx0	= {0, 1, 2, 3};
	int32x4_t		idx1	= {4, 5, 6, 7};
	const int32x4_t stride	= vdupq_n_s32(8);
	int32_t			i		= 0;
	for (; i + 8 <= vocab; i += 8) {
		float32x4_t v0 = vld1q_f32(logits + i);
		float32x4_t v1 = vld1q_f32(logits + i + 4);
		uint32x4_t	m0 = vcgtq_f32(v0, best_v0);
		uint32x4_t	m1 = vcgtq_f32(v1, best_v1);
		best_v0		   = vbslq_f32(m0, v0, best_v0);
		best_v1		   = vbslq_f32(m1, v1, best_v1);
		best_i0		   = vbslq_s32(m0, idx0, best_i0);
		best_i1		   = vbslq_s32(m1, idx1, best_i1);
		idx0		   = vaddq_s32(idx0, stride);
		idx1		   = vaddq_s32(idx1, stride);
	}
	uint32x4_t	msel = vcgtq_f32(best_v1, best_v0);
	float32x4_t bv	 = vbslq_f32(msel, best_v1, best_v0);
	int32x4_t	bi	 = vbslq_s32(msel, best_i1, best_i0);
	float		vals[4];
	int32_t		idxs[4];
	vst1q_f32(vals, bv);
	vst1q_s32(idxs, bi);
	int	  best	= idxs[0];
	float bestv = vals[0];
	for (int k = 1; k < 4; k++) {
		if (vals[k] > bestv) {
			bestv = vals[k];
			best  = idxs[k];
		}
	}
	for (; i < vocab; i++) {
		if (logits[i] > bestv) {
			bestv = logits[i];
			best  = i;
		}
	}
	return best;
}

static void cpu_attention_mla_head_neon(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_mla_job_neon *j = ctx;

	float q_h[HEAD_DIM_MAX];
	float q_absorbed[HEAD_DIM_MAX];
	float VKQ_latent[HEAD_DIM_MAX];

	const float *q_rope_cos = j->rope_cos_base + ((size_t)j->pos * j->half_rope);
	const float *q_rope_sin = j->rope_sin_base + ((size_t)j->pos * j->half_rope);

	for (int h = begin; h < end; h++) {
		const float *q_src = j->qf + ((size_t)h * j->qk_head);
		for (int d = 0; d < j->qk_head; d++)
			q_h[d] = q_src[d];
		for (int jj = 0; jj < j->half_rope; jj++) {
			float c						   = q_rope_cos[jj];
			float s						   = q_rope_sin[jj];
			float a						   = q_h[j->qk_nope + (2 * jj)];
			float b						   = q_h[j->qk_nope + (2 * jj) + 1];
			q_h[j->qk_nope + (2 * jj)]	   = (a * c) - (b * s);
			q_h[j->qk_nope + (2 * jj) + 1] = (a * s) + (b * c);
		}

		const float *k_b_h = j->k_b + ((size_t)h * j->qk_nope * j->kv_lora);
		const float *v_b_h = j->v_b + ((size_t)h * j->kv_lora * j->v_head);

		for (int i = 0; i < j->kv_lora; i++) {
			const float *row = k_b_h + ((size_t)i * j->qk_nope);
			float32x4_t	 acc = vdupq_n_f32(0.0f);
			int			 d	 = 0;
			for (; d + 4 <= j->qk_nope; d += 4)
				acc = vfmaq_f32(acc, vld1q_f32(row + d), vld1q_f32(q_h + d));
			float sum = vaddvq_f32(acc);
			for (; d < j->qk_nope; d++)
				sum += row[d] * q_h[d];
			q_absorbed[i] = sum;
		}
		const float *q_rope_part = q_h + j->qk_nope;

		float M = -INFINITY;
		float S = 0.0f;
		neon_zero_f32(VKQ_latent, j->kv_lora);
		for (int t = 0; t < j->n_pos; t++) {
			const float *slot	  = j->layer_c + ((size_t)t * (j->kv_lora + j->qk_rope));
			const float *latent	  = slot;
			const float *k_pe_rot = j->k_pe_rot_all + ((size_t)t * j->qk_rope);

			float32x4_t acc0 = vdupq_n_f32(0.0f);
			int			i	 = 0;
			for (; i + 4 <= j->kv_lora; i += 4)
				acc0 = vfmaq_f32(acc0, vld1q_f32(latent + i), vld1q_f32(q_absorbed + i));
			float score = vaddvq_f32(acc0);
			for (; i < j->kv_lora; i++)
				score += latent[i] * q_absorbed[i];

			float32x4_t acc1 = vdupq_n_f32(0.0f);
			int			d	 = 0;
			for (; d + 4 <= j->qk_rope; d += 4)
				acc1 = vfmaq_f32(acc1, vld1q_f32(q_rope_part + d), vld1q_f32(k_pe_rot + d));
			score += vaddvq_f32(acc1);
			for (; d < j->qk_rope; d++)
				score += q_rope_part[d] * k_pe_rot[d];
			score *= j->scale;

			float ms = 1.0f;
			float vs = 1.0f;
			if (score > M) {
				float Mold		= M;
				M				= score;
				ms				= fast_expf(Mold - M);
				float32x4_t msv = vdupq_n_f32(ms);
				int			i2	= 0;
				for (; i2 + 4 <= j->kv_lora; i2 += 4)
					vst1q_f32(VKQ_latent + i2, vmulq_f32(vld1q_f32(VKQ_latent + i2), msv));
				for (; i2 < j->kv_lora; i2++)
					VKQ_latent[i2] *= ms;
			} else {
				vs = fast_expf(score - M);
			}
			float32x4_t vsv = vdupq_n_f32(vs);
			int			i3	= 0;
			for (; i3 + 4 <= j->kv_lora; i3 += 4)
				vst1q_f32(VKQ_latent + i3,
						  vfmaq_f32(vld1q_f32(VKQ_latent + i3), vld1q_f32(latent + i3), vsv));
			for (; i3 < j->kv_lora; i3++)
				VKQ_latent[i3] = fmaf(latent[i3], vs, VKQ_latent[i3]);
			S = (S * ms) + vs;
		}
		float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
		float VKQ[HEAD_DIM_MAX];
		for (int d = 0; d < j->v_head; d++) {
			const float *row = v_b_h + ((size_t)d * j->kv_lora);
			float32x4_t	 acc = vdupq_n_f32(0.0f);
			int			 i	 = 0;
			for (; i + 4 <= j->kv_lora; i += 4)
				acc = vfmaq_f32(acc, vld1q_f32(VKQ_latent + i), vld1q_f32(row + i));
			float sum = vaddvq_f32(acc);
			for (; i < j->kv_lora; i++)
				sum += VKQ_latent[i] * row[i];
			VKQ[d] = sum * S_inv;
		}
		for (int d = 0; d < j->v_head; d++)
			j->outf[((size_t)h * j->v_head) + d] = VKQ[d];
	}
}

static void cpu_mla_krot_one(const cpu_mla_krot_job *j, int t) {
	const float *slot		= j->layer_c + ((size_t)t * j->total_dim);
	const float *k_pe		= slot + j->kv_lora;
	const float *t_rope_cos = j->rope_cos_base + ((size_t)t * j->half_rope);
	const float *t_rope_sin = j->rope_sin_base + ((size_t)t * j->half_rope);
	float		*dst		= j->k_pe_rot_all + ((size_t)t * j->qk_rope);

	int jj = 0;
	for (; jj + 4 <= j->half_rope; jj += 4) {
		float32x4x2_t pa  = vld2q_f32(k_pe + 2 * jj);
		float32x4_t	  c	  = vld1q_f32(t_rope_cos + jj);
		float32x4_t	  s	  = vld1q_f32(t_rope_sin + jj);
		float32x4_t	  re  = vmlsq_f32(vmulq_f32(pa.val[0], c), pa.val[1], s);
		float32x4_t	  im  = vmlaq_f32(vmulq_f32(pa.val[0], s), pa.val[1], c);
		float32x4x2_t out = {.val = {re, im}};
		vst2q_f32(dst + 2 * jj, out);
	}
	for (; jj < j->half_rope; jj++) {
		float c			  = t_rope_cos[jj];
		float s			  = t_rope_sin[jj];
		float a			  = k_pe[2 * jj];
		float b			  = k_pe[(2 * jj) + 1];
		dst[2 * jj]		  = (a * c) - (b * s);
		dst[(2 * jj) + 1] = (a * s) + (b * c);
	}
}

static void cpu_mla_krot_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_mla_krot_job *j = ctx;
	for (int t = begin; t < end; t++)
		cpu_mla_krot_one(j, t);
}

static void cpu_mla_precompute_k_pe_rot(cpu_priv *p, const float *layer_c, float *k_pe_rot_all,
										const float *rope_cos_base, const float *rope_sin_base,
										int total_dim, int kv_lora, int qk_rope, int half_rope,
										int n_pos, int layer) {
	cpu_mla_krot_job job = {.layer_c	   = layer_c,
							.k_pe_rot_all  = k_pe_rot_all,
							.rope_cos_base = rope_cos_base,
							.rope_sin_base = rope_sin_base,
							.total_dim	   = total_dim,
							.kv_lora	   = kv_lora,
							.qk_rope	   = qk_rope,
							.half_rope	   = half_rope};

	int cached_n_pos = p->mla_krot.n_pos_cached;
	int can_incr =
		(cached_n_pos == n_pos - 1) && (p->mla_krot.layer_cached == layer) &&
		(p->mla_krot.layer_c_cached == layer_c) && (p->mla_krot.total_dim_cached == total_dim) &&
		(p->mla_krot.kv_lora_cached == kv_lora) && (p->mla_krot.qk_rope_cached == qk_rope) &&
		(p->mla_krot.half_rope_cached == half_rope);
	if (can_incr && n_pos >= 1) {
		cpu_mla_krot_one(&job, n_pos - 1);
		p->mla_krot.n_pos_cached = n_pos;
		return;
	}

	if (p->pool && n_pos >= MLA_KROT_PARALLEL_MIN_POS) {
		tpool_parallel_for(p->pool, n_pos, 1, cpu_mla_krot_chunk, &job);
	} else {
		cpu_mla_krot_chunk(0, n_pos, 0, &job);
	}
	p->mla_krot.layer_c_cached	 = layer_c;
	p->mla_krot.layer_cached	 = layer;
	p->mla_krot.total_dim_cached = total_dim;
	p->mla_krot.kv_lora_cached	 = kv_lora;
	p->mla_krot.qk_rope_cached	 = qk_rope;
	p->mla_krot.half_rope_cached = half_rope;
	p->mla_krot.n_pos_cached	 = n_pos;
}

status_code cpu_attention_mla(backend *self, const buffer *q, const buffer *kv_cache,
							  const buffer *k_b_w, const buffer *v_b_w, buffer *out, int layer,
							  int pos, int n_heads, int qk_head, int qk_rope, int qk_nope,
							  int v_head, int kv_lora, int n_ctx, const float *rope_cos_base,
							  const float *rope_sin_base, float scale) {
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
	status_code grow_st = cpu_scratch_grow((void **)&p->mla_krot.buf, &p->mla_krot.cap, krot_need);
	if (grow_st != OK)
		return grow_st;
	float *k_pe_rot_all = p->mla_krot.buf;
	cpu_mla_precompute_k_pe_rot(p, layer_c, k_pe_rot_all, rope_cos_base, rope_sin_base, total_dim,
								kv_lora, qk_rope, half_rope, n_pos, layer);

	for (int i = 0; i < n_heads * v_head; i++)
		outf[i] = 0.0f;

	cpu_mla_job_neon job = {.qf			   = qf,
							.outf		   = outf,
							.layer_c	   = layer_c,
							.k_b		   = k_b,
							.v_b		   = v_b,
							.k_pe_rot_all  = k_pe_rot_all,
							.rope_cos_base = rope_cos_base,
							.rope_sin_base = rope_sin_base,
							.qk_head	   = qk_head,
							.qk_rope	   = qk_rope,
							.qk_nope	   = qk_nope,
							.v_head		   = v_head,
							.kv_lora	   = kv_lora,
							.n_pos		   = n_pos,
							.half_rope	   = half_rope,
							.pos		   = pos,
							.scale		   = scale};

	if (p->pool && n_heads > 1 &&
		(size_t)n_heads * (size_t)n_pos * (size_t)(qk_head + v_head) >= 4096) {
		tpool_parallel_for(p->pool, n_heads, 1, cpu_attention_mla_head_neon, &job);
	} else {
		cpu_attention_mla_head_neon(0, n_heads, 0, &job);
	}

	return OK;
}

void detect_features(char *buf, size_t cap) {
	buf[0] = '\0';
	feat_add(buf, cap, "aarch64");
#if defined(__ARM_FEATURE_FMA)
	feat_add(buf, cap, "fma");
#endif
#if defined(__ARM_NEON)
	feat_add(buf, cap, "neon");
#endif
#if defined(__ARM_FEATURE_DOTPROD)
	feat_add(buf, cap, "dotprod");
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
	feat_add(buf, cap, "fp16");
#endif
#if defined(__ARM_FEATURE_SVE)
	feat_add(buf, cap, "sve");
#endif
}