#include "backend/backend.h"
#include "backend/cpu/common.h"
#include "backend/cpu/scalar/quants.h"
#include "common.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	const float *g, *u;
	float		*o;
	int			 n;
	int			 activation;
} cpu_ffn_act_batch_args;

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
		size_t	 kvh_stride	  = cpu_kv_layer_rows(p, layer, n_ctx) * block_stride;
		size_t	 layer_stride = (size_t)n_kv_heads * kvh_stride;
		uint8_t *kd_base	  = (uint8_t *)k->handle;
		uint8_t *vd_base	  = (uint8_t *)v->handle;
		for (int kvh = 0; kvh < n_active; kvh++) {
			size_t layer_base = cpu_kv_layer_base(p, layer, layer_stride, 1);
			size_t pos_off	  = cpu_kv_put_slot(p, layer, pos, n_ctx, kd_base, vd_base, layer_base,
												kvh_stride, block_stride, n_active) *
								block_stride;
			size_t off		  = layer_base + ((size_t)kvh * kvh_stride) + pos_off;
			cpu_kv_put_q8_0_head(kd_base + off, vd_base + off, kf + ((size_t)kvh * head_dim),
								 vf + ((size_t)kvh * head_dim), head_dim);
		}
		return OK;
	}
	uint16_t *kd_base	   = (uint16_t *)cpu_ptr(k);
	uint16_t *vd_base	   = (uint16_t *)cpu_ptr(v);
	int		  hd_stride	   = head_dim;
	size_t	  kvh_stride   = cpu_kv_layer_rows(p, layer, n_ctx) * hd_stride;
	size_t	  layer_stride = (size_t)n_kv_heads * kvh_stride;

	for (int kvh = 0; kvh < n_active; kvh++) {
		size_t layer_base = cpu_kv_layer_base(p, layer, layer_stride, sizeof(uint16_t));
		size_t pos_off =
			cpu_kv_put_slot(p, layer, pos, n_ctx, kd_base, vd_base, layer_base * sizeof(uint16_t),
							kvh_stride * sizeof(uint16_t), (size_t)hd_stride * sizeof(uint16_t),
							n_active) *
			hd_stride;
		size_t		 off = layer_base + ((size_t)kvh * kvh_stride) + pos_off;
		uint16_t	*kd	 = kd_base + off;
		uint16_t	*vd	 = vd_base + off;
		const float *kfh = kf + ((size_t)kvh * head_dim);
		const float *vfh = vf + ((size_t)kvh * head_dim);
		int			 i	 = 0;
		for (; i + 32 <= head_dim; i += 32) {
			__m256 k0 = _mm256_loadu_ps(kfh + i);
			__m256 k1 = _mm256_loadu_ps(kfh + i + 8);
			__m256 k2 = _mm256_loadu_ps(kfh + i + 16);
			__m256 k3 = _mm256_loadu_ps(kfh + i + 24);
			_mm_storeu_si128((__m128i *)(kd + i), _mm256_cvtps_ph(k0, _MM_FROUND_TO_NEAREST_INT));
			_mm_storeu_si128((__m128i *)(kd + i + 8),
							 _mm256_cvtps_ph(k1, _MM_FROUND_TO_NEAREST_INT));
			_mm_storeu_si128((__m128i *)(kd + i + 16),
							 _mm256_cvtps_ph(k2, _MM_FROUND_TO_NEAREST_INT));
			_mm_storeu_si128((__m128i *)(kd + i + 24),
							 _mm256_cvtps_ph(k3, _MM_FROUND_TO_NEAREST_INT));

			__m256 v0 = _mm256_loadu_ps(vfh + i);
			__m256 v1 = _mm256_loadu_ps(vfh + i + 8);
			__m256 v2 = _mm256_loadu_ps(vfh + i + 16);
			__m256 v3 = _mm256_loadu_ps(vfh + i + 24);
			_mm_storeu_si128((__m128i *)(vd + i), _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT));
			_mm_storeu_si128((__m128i *)(vd + i + 8),
							 _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT));
			_mm_storeu_si128((__m128i *)(vd + i + 16),
							 _mm256_cvtps_ph(v2, _MM_FROUND_TO_NEAREST_INT));
			_mm_storeu_si128((__m128i *)(vd + i + 24),
							 _mm256_cvtps_ph(v3, _MM_FROUND_TO_NEAREST_INT));
		}
		for (; i + 16 <= head_dim; i += 16) {
			__m256 k0 = _mm256_loadu_ps(kfh + i);
			__m256 k1 = _mm256_loadu_ps(kfh + i + 8);
			_mm_storeu_si128((__m128i *)(kd + i), _mm256_cvtps_ph(k0, _MM_FROUND_TO_NEAREST_INT));
			_mm_storeu_si128((__m128i *)(kd + i + 8),
							 _mm256_cvtps_ph(k1, _MM_FROUND_TO_NEAREST_INT));

			__m256 v0 = _mm256_loadu_ps(vfh + i);
			__m256 v1 = _mm256_loadu_ps(vfh + i + 8);
			_mm_storeu_si128((__m128i *)(vd + i), _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT));
			_mm_storeu_si128((__m128i *)(vd + i + 8),
							 _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT));
		}
		for (; i + 8 <= head_dim; i += 8) {
			__m256 k0 = _mm256_loadu_ps(kfh + i);
			_mm_storeu_si128((__m128i *)(kd + i), _mm256_cvtps_ph(k0, _MM_FROUND_TO_NEAREST_INT));

			__m256 v0 = _mm256_loadu_ps(vfh + i);
			_mm_storeu_si128((__m128i *)(vd + i), _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT));
		}
		for (; i + 4 <= head_dim; i += 4) {
			__m128 k0 = _mm_loadu_ps(kfh + i);
			__m128 v0 = _mm_loadu_ps(vfh + i);
			_mm_storel_epi64((__m128i *)(kd + i), _mm_cvtps_ph(k0, _MM_FROUND_TO_NEAREST_INT));
			_mm_storel_epi64((__m128i *)(vd + i), _mm_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT));
		}
		for (; i < head_dim; i++) {
			kd[i] = f32_to_f16(kfh[i]);
			vd[i] = f32_to_f16(vfh[i]);
		}
	}
	return OK;
}

static void cpu_rope_neox_one_avx(float *vh, int half, const float *rope_cos,
								  const float *rope_sin) {
	int j = 0;
	for (; j + 8 <= half; j += 8) {
		__m256 c0 = _mm256_loadu_ps(rope_cos + j);
		__m256 s0 = _mm256_loadu_ps(rope_sin + j);
		__m256 v0 = _mm256_loadu_ps(vh + j);
		__m256 v1 = _mm256_loadu_ps(vh + j + half);
		_mm256_storeu_ps(vh + j, _mm256_fnmadd_ps(v1, s0, _mm256_mul_ps(v0, c0)));
		_mm256_storeu_ps(vh + j + half, _mm256_fmadd_ps(v1, c0, _mm256_mul_ps(v0, s0)));
	}
	for (; j < half; j++) {
		float c		 = rope_cos[j];
		float s		 = rope_sin[j];
		float v0	 = vh[j];
		float v1	 = vh[j + half];
		vh[j]		 = (v0 * c) - (v1 * s);
		vh[j + half] = (v0 * s) + (v1 * c);
	}
}

static void cpu_rope_interleave_one_avx(float *vh, int half, const float *rope_cos,
										const float *rope_sin) {
	int j = 0;
	for (; j + 2 <= half; j += 2) {
		float *base = vh + 2 * j;
		__m128 p	= _mm_loadu_ps(base);
		__m128 c	= _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)(rope_cos + j)));
		__m128 s	= _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)(rope_sin + j)));
		__m128 ev	= _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 0, 2, 0));
		__m128 od	= _mm_shuffle_ps(p, p, _MM_SHUFFLE(3, 1, 3, 1));
		__m128 re	= _mm_sub_ps(_mm_mul_ps(ev, c), _mm_mul_ps(od, s));
		__m128 im	= _mm_add_ps(_mm_mul_ps(ev, s), _mm_mul_ps(od, c));
		_mm_storeu_ps(base, _mm_unpacklo_ps(re, im));
	}
	for (; j < half; j++) {
		float c			= rope_cos[j];
		float s			= rope_sin[j];
		float v0		= vh[2 * j];
		float v1		= vh[(2 * j) + 1];
		vh[2 * j]		= (v0 * c) - (v1 * s);
		vh[(2 * j) + 1] = (v0 * s) + (v1 * c);
	}
}

static void cpu_rope_cs_one_avx(float *vh, int half, const float *cs, int neox) {
	int j = 0;
	if (neox) {
		const __m256i idx_lo = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
		for (; j + 8 <= half; j += 8) {
			__m256 v0  = _mm256_loadu_ps(vh + j);
			__m256 v1  = _mm256_loadu_ps(vh + j + half);
			__m256 c_a = _mm256_loadu_ps(cs + (2 * j));
			__m256 c_b = _mm256_loadu_ps(cs + (2 * j) + 8);
			__m256 c   = _mm256_permutevar8x32_ps(
				_mm256_shuffle_ps(c_a, c_b, _MM_SHUFFLE(2, 0, 2, 0)), idx_lo);
			__m256 s = _mm256_permutevar8x32_ps(
				_mm256_shuffle_ps(c_a, c_b, _MM_SHUFFLE(3, 1, 3, 1)), idx_lo);
			_mm256_storeu_ps(vh + j, _mm256_fnmadd_ps(v1, s, _mm256_mul_ps(v0, c)));
			_mm256_storeu_ps(vh + j + half, _mm256_fmadd_ps(v1, c, _mm256_mul_ps(v0, s)));
		}
		for (; j < half; j++) {
			float c		 = cs[2 * j];
			float s		 = cs[(2 * j) + 1];
			float v0	 = vh[j];
			float v1	 = vh[j + half];
			vh[j]		 = (v0 * c) - (v1 * s);
			vh[j + half] = (v0 * s) + (v1 * c);
		}
		return;
	}
	for (; j + 2 <= half; j += 2) {
		float *base = vh + 2 * j;
		__m128 p	= _mm_loadu_ps(base);
		__m128 cs_v = _mm_loadu_ps(cs + (2 * j));
		__m128 ev	= _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 0, 2, 0));
		__m128 od	= _mm_shuffle_ps(p, p, _MM_SHUFFLE(3, 1, 3, 1));
		__m128 c	= _mm_shuffle_ps(cs_v, cs_v, _MM_SHUFFLE(2, 0, 2, 0));
		__m128 s	= _mm_shuffle_ps(cs_v, cs_v, _MM_SHUFFLE(3, 1, 3, 1));
		__m128 re	= _mm_sub_ps(_mm_mul_ps(ev, c), _mm_mul_ps(od, s));
		__m128 im	= _mm_add_ps(_mm_mul_ps(ev, s), _mm_mul_ps(od, c));
		_mm_storeu_ps(base, _mm_unpacklo_ps(re, im));
	}
	for (; j < half; j++) {
		float c			= cs[2 * j];
		float s			= cs[(2 * j) + 1];
		float v0		= vh[2 * j];
		float v1		= vh[(2 * j) + 1];
		vh[2 * j]		= (v0 * c) - (v1 * s);
		vh[(2 * j) + 1] = (v0 * s) + (v1 * c);
	}
}

static void cpu_rope_one_avx(float *v, int n_heads, int head_dim, const float *rope_cos,
							 const float *rope_sin, int neox) {
	for (int h = 0; h < n_heads; h++) {
		float *vh = v + ((size_t)h * head_dim);
		if (neox)
			cpu_rope_neox_one_avx(vh, head_dim / 2, rope_cos, rope_sin);
		else
			cpu_rope_interleave_one_avx(vh, head_dim / 2, rope_cos, rope_sin);
	}
}

status_code cpu_rope(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
					 const float *rope_cos_base, const float *rope_sin_base) {
	int			 half	  = head_dim / 2;
	const float *rope_cos = rope_cos_base + ((size_t)pos * half);
	const float *rope_sin = rope_sin_base + ((size_t)pos * half);
	cpu_rope_one_avx((float *)cpu_ptr(vec), n_heads, head_dim, rope_cos, rope_sin, self->rope_neox);
	return OK;
}

status_code cpu_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
						int head_dim, int pos, const float *rope_cos_base,
						const float *rope_sin_base) {
	int			 half	  = head_dim / 2;
	const float *rope_cos = rope_cos_base + ((size_t)pos * half);
	const float *rope_sin = rope_sin_base + ((size_t)pos * half);
	int			 neox	  = self->rope_neox;
	cpu_rope_one_avx((float *)cpu_ptr(q), n_heads, head_dim, rope_cos, rope_sin, neox);
	cpu_rope_one_avx((float *)cpu_ptr(k), n_kv_heads, head_dim, rope_cos, rope_sin, neox);
	return OK;
}

static void cpu_rope_batch_chunk_avx(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rope_batch_job *j	 = ctx;
	int					half = j->head_dim / 2;
	for (int row = begin; row < end; row++) {
		int			 pos = j->pos_start + row;
		const float *rc	 = j->rope_cos_base + ((size_t)pos * half);
		const float *rs	 = j->rope_sin_base + ((size_t)pos * half);
		cpu_rope_one_avx(j->vec + ((size_t)row * j->n_heads * j->head_dim), j->n_heads, j->head_dim,
						 rc, rs, j->rope_neox);
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
	cpu_run_batch(p->pool, m, cpu_rope_batch_chunk_avx, &job);
	return OK;
}

static void cpu_rope_qk_batch_chunk_avx(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rope_qk_batch_job *j	= ctx;
	int					   half = j->head_dim / 2;
	for (int row = begin; row < end; row++) {
		int			 pos = j->pos_start + row;
		const float *rc	 = j->rope_cos_base + ((size_t)pos * half);
		const float *rs	 = j->rope_sin_base + ((size_t)pos * half);
		cpu_rope_one_avx(j->q + ((size_t)row * j->n_heads * j->head_dim), j->n_heads, j->head_dim,
						 rc, rs, j->rope_neox);
		cpu_rope_one_avx(j->k + ((size_t)row * j->n_kv_heads * j->head_dim), j->n_kv_heads,
						 j->head_dim, rc, rs, j->rope_neox);
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
	cpu_run_batch(p->pool, m, cpu_rope_qk_batch_chunk_avx, &job);
	return OK;
}

typedef struct {
	float		*base;
	int			 n_heads, head_dim, row_stride, pos_start;
	float		 theta;
	int			 neox;
	const float *freq_factors;
	const float *cos_base, *sin_base;
} x86_rope_ext_batch_job;

static void x86_rope_cs_fill(float *cs, int half, int pos, double theta_scale,
							 const float *freq_factors) {
	double base_freq = 1.0;
	for (int j = 0; j < half; j++) {
		double ff;
		if (freq_factors) {
			float fv = freq_factors[j];
			ff		 = (fv >= 1e10f || fv == 0.0f) ? 0.0 : (double)fv;
		} else {
			ff = 1.0;
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

static status_code x86_rope_cs_cached(cpu_priv *p, int head_dim, int pos, float theta,
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
		x86_rope_cs_fill(p->rope_cs.cs, half, pos, theta_scale, freq_factors);
		p->rope_cs.pos_cached		   = pos;
		p->rope_cs.theta_cs_cached	   = theta;
		p->rope_cs.head_dim_cs_cached  = head_dim;
		p->rope_cs.freq_factors_cached = freq_factors;
	}

	*cs_out = p->rope_cs.cs;
	return OK;
}

status_code cpu_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
						 const float *rope_cos_base, const float *rope_sin_base,
						 const float *freq_factors) {
	float *v = cpu_ptr(vec);
	if (freq_factors) {
		float	   *cs;
		status_code st =
			x86_rope_cs_cached(self->priv, head_dim, pos, self->rope_theta, freq_factors, &cs);
		if (st != OK)
			return st;
		for (int h = 0; h < n_heads; h++)
			cpu_rope_cs_one_avx(v + ((size_t)h * head_dim), head_dim / 2, cs, self->rope_neox);
	} else {
		int			 half	  = head_dim / 2;
		const float *rope_cos = rope_cos_base + ((size_t)pos * half);
		const float *rope_sin = rope_sin_base + ((size_t)pos * half);
		cpu_rope_one_avx(v, n_heads, head_dim, rope_cos, rope_sin, self->rope_neox);
	}
	return OK;
}

static void cpu_rope_ext_batch_chunk_avx(int begin, int end, int tid, void *ctx) {
	(void)tid;
	x86_rope_ext_batch_job *j			= ctx;
	int						half		= j->head_dim / 2;
	double					theta_scale = pow((double)j->theta, -2.0 / (double)j->head_dim);

	for (int r = begin; r < end; r++) {
		int	   pos = j->pos_start + r;
		float  cs_stack[512];
		float *csp = cs_stack;
		if (j->freq_factors) {
			if (half > 256)
				csp = xmalloc((size_t)half * 2 * sizeof(float));
			x86_rope_cs_fill(csp, half, pos, theta_scale, j->freq_factors);
			float *v = j->base + (size_t)r * j->row_stride;
			for (int h = 0; h < j->n_heads; h++)
				cpu_rope_cs_one_avx(v + ((size_t)h * j->head_dim), half, csp, j->neox);
			if (csp != cs_stack)
				free(csp);
		} else {
			const float *rc = j->cos_base + ((size_t)pos * half);
			const float *rs = j->sin_base + ((size_t)pos * half);
			cpu_rope_one_avx(j->base + ((size_t)r * j->row_stride), j->n_heads, j->head_dim, rc, rs,
							 j->neox);
		}
	}
}

status_code cpu_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim, int pos_start,
							   const float *rope_cos_base, const float *rope_sin_base,
							   const float *freq_factors, int m) {
	cpu_priv *p = self->priv;
	if (m <= 1)
		return cpu_rope_ext(self, vec, n_heads, head_dim, pos_start, rope_cos_base, rope_sin_base,
							freq_factors);

	x86_rope_ext_batch_job job = {.base			= (float *)cpu_ptr(vec),
								  .n_heads		= n_heads,
								  .head_dim		= head_dim,
								  .row_stride	= n_heads * head_dim,
								  .pos_start	= pos_start,
								  .theta		= self->rope_theta,
								  .neox			= self->rope_neox,
								  .freq_factors = freq_factors,
								  .cos_base		= rope_cos_base,
								  .sin_base		= rope_sin_base};
	cpu_run_batch(p->pool, m, cpu_rope_ext_batch_chunk_avx, &job);
	return OK;
}

void cpu_partial_rope_qk_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_partial_rope_qk_job *j = ctx;
	for (int row = begin; row < end; row++) {
		const float *cosv = j->cos_base + (size_t)(j->pos0 + row) * j->half;
		const float *sinv = j->sin_base + (size_t)(j->pos0 + row) * j->half;
		float		*qr	  = j->q + (size_t)row * j->qn;
		float		*kr	  = j->k + (size_t)row * j->kn;
		for (int h = 0; h < j->n_heads; h++)
			cpu_rope_neox_one_avx(qr + ((size_t)h * j->head_dim), j->half, cosv, sinv);
		for (int h = 0; h < j->n_kv_heads; h++)
			cpu_rope_neox_one_avx(kr + ((size_t)h * j->head_dim), j->half, cosv, sinv);
	}
}

float dot8(const float *restrict a, const float *restrict b, int head_dim) {
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();
	int	   d	= 0;

	for (; d + 64 <= head_dim; d += 64) {
		__m256 a0 = _mm256_loadu_ps(a + d);
		__m256 a1 = _mm256_loadu_ps(a + d + 8);
		__m256 a2 = _mm256_loadu_ps(a + d + 16);
		__m256 a3 = _mm256_loadu_ps(a + d + 24);
		__m256 a4 = _mm256_loadu_ps(a + d + 32);
		__m256 a5 = _mm256_loadu_ps(a + d + 40);
		__m256 a6 = _mm256_loadu_ps(a + d + 48);
		__m256 a7 = _mm256_loadu_ps(a + d + 56);

		acc0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b + d), acc0);
		acc1 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b + d + 8), acc1);
		acc2 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b + d + 16), acc2);
		acc3 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b + d + 24), acc3);
		acc0 = _mm256_fmadd_ps(a4, _mm256_loadu_ps(b + d + 32), acc0);
		acc1 = _mm256_fmadd_ps(a5, _mm256_loadu_ps(b + d + 40), acc1);
		acc2 = _mm256_fmadd_ps(a6, _mm256_loadu_ps(b + d + 48), acc2);
		acc3 = _mm256_fmadd_ps(a7, _mm256_loadu_ps(b + d + 56), acc3);
	}

	for (; d + 32 <= head_dim; d += 32) {
		__m256 a0 = _mm256_loadu_ps(a + d);
		__m256 a1 = _mm256_loadu_ps(a + d + 8);
		__m256 a2 = _mm256_loadu_ps(a + d + 16);
		__m256 a3 = _mm256_loadu_ps(a + d + 24);

		acc0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b + d), acc0);
		acc1 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b + d + 8), acc1);
		acc2 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b + d + 16), acc2);
		acc3 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b + d + 24), acc3);
	}

	for (; d + 8 <= head_dim; d += 8) {
		acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc0);
	}

	__m256 acc01 = _mm256_add_ps(acc0, acc1);
	__m256 acc23 = _mm256_add_ps(acc2, acc3);
	__m256 acc	 = _mm256_add_ps(acc01, acc23);
	float  s	 = vreduce_add_ps(acc);

	for (; d < head_dim; d++) {
		s += a[d] * b[d];
	}

	return s;
}

float dot8_f16(const float *restrict a, const uint16_t *restrict b, int head_dim) {
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();
	int	   d	= 0;

	for (; d + 64 <= head_dim; d += 64) {
		__m256 a0 = _mm256_loadu_ps(a + d);
		__m256 a1 = _mm256_loadu_ps(a + d + 8);
		__m256 a2 = _mm256_loadu_ps(a + d + 16);
		__m256 a3 = _mm256_loadu_ps(a + d + 24);
		__m256 a4 = _mm256_loadu_ps(a + d + 32);
		__m256 a5 = _mm256_loadu_ps(a + d + 40);
		__m256 a6 = _mm256_loadu_ps(a + d + 48);
		__m256 a7 = _mm256_loadu_ps(a + d + 56);

		__m256 b0 = loadu_f16x8_to_ps(b + d);
		__m256 b1 = loadu_f16x8_to_ps(b + d + 8);
		__m256 b2 = loadu_f16x8_to_ps(b + d + 16);
		__m256 b3 = loadu_f16x8_to_ps(b + d + 24);
		__m256 b4 = loadu_f16x8_to_ps(b + d + 32);
		__m256 b5 = loadu_f16x8_to_ps(b + d + 40);
		__m256 b6 = loadu_f16x8_to_ps(b + d + 48);
		__m256 b7 = loadu_f16x8_to_ps(b + d + 56);

		acc0 = _mm256_fmadd_ps(a0, b0, acc0);
		acc1 = _mm256_fmadd_ps(a1, b1, acc1);
		acc2 = _mm256_fmadd_ps(a2, b2, acc2);
		acc3 = _mm256_fmadd_ps(a3, b3, acc3);
		acc0 = _mm256_fmadd_ps(a4, b4, acc0);
		acc1 = _mm256_fmadd_ps(a5, b5, acc1);
		acc2 = _mm256_fmadd_ps(a6, b6, acc2);
		acc3 = _mm256_fmadd_ps(a7, b7, acc3);
	}

	for (; d + 32 <= head_dim; d += 32) {
		__m256 a0 = _mm256_loadu_ps(a + d);
		__m256 a1 = _mm256_loadu_ps(a + d + 8);
		__m256 a2 = _mm256_loadu_ps(a + d + 16);
		__m256 a3 = _mm256_loadu_ps(a + d + 24);

		__m256 b0 = loadu_f16x8_to_ps(b + d);
		__m256 b1 = loadu_f16x8_to_ps(b + d + 8);
		__m256 b2 = loadu_f16x8_to_ps(b + d + 16);
		__m256 b3 = loadu_f16x8_to_ps(b + d + 24);

		acc0 = _mm256_fmadd_ps(a0, b0, acc0);
		acc1 = _mm256_fmadd_ps(a1, b1, acc1);
		acc2 = _mm256_fmadd_ps(a2, b2, acc2);
		acc3 = _mm256_fmadd_ps(a3, b3, acc3);
	}

	for (; d + 16 <= head_dim; d += 16) {
		__m256 a0 = _mm256_loadu_ps(a + d);
		__m256 a1 = _mm256_loadu_ps(a + d + 8);

		__m256 b0 = loadu_f16x8_to_ps(b + d);
		__m256 b1 = loadu_f16x8_to_ps(b + d + 8);

		acc0 = _mm256_fmadd_ps(a0, b0, acc0);
		acc1 = _mm256_fmadd_ps(a1, b1, acc1);
	}

	for (; d + 8 <= head_dim; d += 8) {
		__m256 a0 = _mm256_loadu_ps(a + d);
		__m256 b0 = loadu_f16x8_to_ps(b + d);
		acc0	  = _mm256_fmadd_ps(a0, b0, acc0);
	}

	__m256 acc01 = _mm256_add_ps(acc0, acc1);
	__m256 acc23 = _mm256_add_ps(acc2, acc3);
	__m256 acc	 = _mm256_add_ps(acc01, acc23);
	float  s	 = vreduce_add_ps(acc);

	for (; d < head_dim; d++) {
		s += a[d] * f16_to_f32_fast(b[d]);
	}

	return s;
}

static void cpu_attention_inner(uint16_t *restrict k_slice, uint16_t *restrict v_slice,
								int kv_stride, const float *qh, float *out_h, int head_dim,
								int n_pos, float scale, int flash_attn, float *restrict scores) {
	if (flash_attn || scores == NULL) {
		float M = -INFINITY;
		float S = 0.0f;
		float VKQ[HEAD_DIM_MAX] __attribute__((aligned(64)));
		memset(VKQ, 0, (size_t)head_dim * sizeof(float));

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
					ms			= expf(M_old - M);
					__m256 ms_v = _mm256_set1_ps(ms);
					int	   d	= 0;
					for (; d + 64 <= head_dim; d += 64) {
						_mm256_storeu_ps(VKQ + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), ms_v));
						_mm256_storeu_ps(VKQ + d + 8,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 8), ms_v));
						_mm256_storeu_ps(VKQ + d + 16,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 16), ms_v));
						_mm256_storeu_ps(VKQ + d + 24,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 24), ms_v));
						_mm256_storeu_ps(VKQ + d + 32,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 32), ms_v));
						_mm256_storeu_ps(VKQ + d + 40,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 40), ms_v));
						_mm256_storeu_ps(VKQ + d + 48,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 48), ms_v));
						_mm256_storeu_ps(VKQ + d + 56,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 56), ms_v));
					}
					for (; d + 32 <= head_dim; d += 32) {
						_mm256_storeu_ps(VKQ + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), ms_v));
						_mm256_storeu_ps(VKQ + d + 8,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 8), ms_v));
						_mm256_storeu_ps(VKQ + d + 16,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 16), ms_v));
						_mm256_storeu_ps(VKQ + d + 24,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 24), ms_v));
					}
					for (; d + 8 <= head_dim; d += 8)
						_mm256_storeu_ps(VKQ + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), ms_v));
					for (; d < head_dim; d++)
						VKQ[d] *= ms;
				}
				vs = 1.0f;
			} else {
				vs = expf(ss - M);
			}

			__m256 vs_v = _mm256_set1_ps(vs);
			int	   d	= 0;
			for (; d + 64 <= head_dim; d += 64) {
				__m256 vt0 = loadu_f16x8_to_ps(vt + d);
				__m256 vt1 = loadu_f16x8_to_ps(vt + d + 8);
				__m256 vt2 = loadu_f16x8_to_ps(vt + d + 16);
				__m256 vt3 = loadu_f16x8_to_ps(vt + d + 24);
				__m256 vt4 = loadu_f16x8_to_ps(vt + d + 32);
				__m256 vt5 = loadu_f16x8_to_ps(vt + d + 40);
				__m256 vt6 = loadu_f16x8_to_ps(vt + d + 48);
				__m256 vt7 = loadu_f16x8_to_ps(vt + d + 56);
				_mm256_storeu_ps(VKQ + d, _mm256_fmadd_ps(vt0, vs_v, _mm256_loadu_ps(VKQ + d)));
				_mm256_storeu_ps(VKQ + d + 8,
								 _mm256_fmadd_ps(vt1, vs_v, _mm256_loadu_ps(VKQ + d + 8)));
				_mm256_storeu_ps(VKQ + d + 16,
								 _mm256_fmadd_ps(vt2, vs_v, _mm256_loadu_ps(VKQ + d + 16)));
				_mm256_storeu_ps(VKQ + d + 24,
								 _mm256_fmadd_ps(vt3, vs_v, _mm256_loadu_ps(VKQ + d + 24)));
				_mm256_storeu_ps(VKQ + d + 32,
								 _mm256_fmadd_ps(vt4, vs_v, _mm256_loadu_ps(VKQ + d + 32)));
				_mm256_storeu_ps(VKQ + d + 40,
								 _mm256_fmadd_ps(vt5, vs_v, _mm256_loadu_ps(VKQ + d + 40)));
				_mm256_storeu_ps(VKQ + d + 48,
								 _mm256_fmadd_ps(vt6, vs_v, _mm256_loadu_ps(VKQ + d + 48)));
				_mm256_storeu_ps(VKQ + d + 56,
								 _mm256_fmadd_ps(vt7, vs_v, _mm256_loadu_ps(VKQ + d + 56)));
			}
			for (; d + 32 <= head_dim; d += 32) {
				__m256 vt0 = loadu_f16x8_to_ps(vt + d);
				__m256 vt1 = loadu_f16x8_to_ps(vt + d + 8);
				__m256 vt2 = loadu_f16x8_to_ps(vt + d + 16);
				__m256 vt3 = loadu_f16x8_to_ps(vt + d + 24);
				_mm256_storeu_ps(VKQ + d, _mm256_fmadd_ps(vt0, vs_v, _mm256_loadu_ps(VKQ + d)));
				_mm256_storeu_ps(VKQ + d + 8,
								 _mm256_fmadd_ps(vt1, vs_v, _mm256_loadu_ps(VKQ + d + 8)));
				_mm256_storeu_ps(VKQ + d + 16,
								 _mm256_fmadd_ps(vt2, vs_v, _mm256_loadu_ps(VKQ + d + 16)));
				_mm256_storeu_ps(VKQ + d + 24,
								 _mm256_fmadd_ps(vt3, vs_v, _mm256_loadu_ps(VKQ + d + 24)));
			}
			for (; d + 16 <= head_dim; d += 16) {
				__m256 vt0 = loadu_f16x8_to_ps(vt + d);
				__m256 vt1 = loadu_f16x8_to_ps(vt + d + 8);
				_mm256_storeu_ps(VKQ + d, _mm256_fmadd_ps(vt0, vs_v, _mm256_loadu_ps(VKQ + d)));
				_mm256_storeu_ps(VKQ + d + 8,
								 _mm256_fmadd_ps(vt1, vs_v, _mm256_loadu_ps(VKQ + d + 8)));
			}
			for (; d + 8 <= head_dim; d += 8) {
				__m256 vt0 = loadu_f16x8_to_ps(vt + d);
				_mm256_storeu_ps(VKQ + d, _mm256_fmadd_ps(vt0, vs_v, _mm256_loadu_ps(VKQ + d)));
			}
			for (; d < head_dim; d++)
				VKQ[d] = fmaf(vs, f16_to_f32_fast(vt[d]), VKQ[d]);
			S = (S * ms) + vs;
		}

		float  S_inv = (S == 0.0f) ? 0.0f : 1.0f / S;
		__m256 inv_v = _mm256_set1_ps(S_inv);
		int	   d	 = 0;
		for (; d + 64 <= head_dim; d += 64) {
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), inv_v));
			_mm256_storeu_ps(out_h + d + 8, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 8), inv_v));
			_mm256_storeu_ps(out_h + d + 16, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 16), inv_v));
			_mm256_storeu_ps(out_h + d + 24, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 24), inv_v));
			_mm256_storeu_ps(out_h + d + 32, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 32), inv_v));
			_mm256_storeu_ps(out_h + d + 40, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 40), inv_v));
			_mm256_storeu_ps(out_h + d + 48, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 48), inv_v));
			_mm256_storeu_ps(out_h + d + 56, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 56), inv_v));
		}
		for (; d + 32 <= head_dim; d += 32) {
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), inv_v));
			_mm256_storeu_ps(out_h + d + 8, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 8), inv_v));
			_mm256_storeu_ps(out_h + d + 16, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 16), inv_v));
			_mm256_storeu_ps(out_h + d + 24, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 24), inv_v));
		}
		for (; d + 8 <= head_dim; d += 8)
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), inv_v));
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
		__m256 zero = _mm256_setzero_ps();
		int	   d	= 0;
		for (; d + 64 <= head_dim; d += 64) {
			_mm256_storeu_ps(out_h + d, zero);
			_mm256_storeu_ps(out_h + d + 8, zero);
			_mm256_storeu_ps(out_h + d + 16, zero);
			_mm256_storeu_ps(out_h + d + 24, zero);
			_mm256_storeu_ps(out_h + d + 32, zero);
			_mm256_storeu_ps(out_h + d + 40, zero);
			_mm256_storeu_ps(out_h + d + 48, zero);
			_mm256_storeu_ps(out_h + d + 56, zero);
		}
		for (; d + 32 <= head_dim; d += 32) {
			_mm256_storeu_ps(out_h + d, zero);
			_mm256_storeu_ps(out_h + d + 8, zero);
			_mm256_storeu_ps(out_h + d + 16, zero);
			_mm256_storeu_ps(out_h + d + 24, zero);
		}
		for (; d + 8 <= head_dim; d += 8)
			_mm256_storeu_ps(out_h + d, zero);
		for (; d < head_dim; d++)
			out_h[d] = 0.0f;
		return;
	}

	{
		float			sv	 = scores[0];
		const uint16_t *vt	 = v_slice;
		__m256			sv_v = _mm256_set1_ps(sv);
		int				d	 = 0;
		for (; d + 64 <= head_dim; d += 64) {
			__m256 vt0 = loadu_f16x8_to_ps(vt + d);
			__m256 vt1 = loadu_f16x8_to_ps(vt + d + 8);
			__m256 vt2 = loadu_f16x8_to_ps(vt + d + 16);
			__m256 vt3 = loadu_f16x8_to_ps(vt + d + 24);
			__m256 vt4 = loadu_f16x8_to_ps(vt + d + 32);
			__m256 vt5 = loadu_f16x8_to_ps(vt + d + 40);
			__m256 vt6 = loadu_f16x8_to_ps(vt + d + 48);
			__m256 vt7 = loadu_f16x8_to_ps(vt + d + 56);
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(sv_v, vt0));
			_mm256_storeu_ps(out_h + d + 8, _mm256_mul_ps(sv_v, vt1));
			_mm256_storeu_ps(out_h + d + 16, _mm256_mul_ps(sv_v, vt2));
			_mm256_storeu_ps(out_h + d + 24, _mm256_mul_ps(sv_v, vt3));
			_mm256_storeu_ps(out_h + d + 32, _mm256_mul_ps(sv_v, vt4));
			_mm256_storeu_ps(out_h + d + 40, _mm256_mul_ps(sv_v, vt5));
			_mm256_storeu_ps(out_h + d + 48, _mm256_mul_ps(sv_v, vt6));
			_mm256_storeu_ps(out_h + d + 56, _mm256_mul_ps(sv_v, vt7));
		}
		for (; d + 32 <= head_dim; d += 32) {
			__m256 vt0 = loadu_f16x8_to_ps(vt + d);
			__m256 vt1 = loadu_f16x8_to_ps(vt + d + 8);
			__m256 vt2 = loadu_f16x8_to_ps(vt + d + 16);
			__m256 vt3 = loadu_f16x8_to_ps(vt + d + 24);
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(sv_v, vt0));
			_mm256_storeu_ps(out_h + d + 8, _mm256_mul_ps(sv_v, vt1));
			_mm256_storeu_ps(out_h + d + 16, _mm256_mul_ps(sv_v, vt2));
			_mm256_storeu_ps(out_h + d + 24, _mm256_mul_ps(sv_v, vt3));
		}
		for (; d + 16 <= head_dim; d += 16) {
			__m256 vt0 = loadu_f16x8_to_ps(vt + d);
			__m256 vt1 = loadu_f16x8_to_ps(vt + d + 8);
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(sv_v, vt0));
			_mm256_storeu_ps(out_h + d + 8, _mm256_mul_ps(sv_v, vt1));
		}
		for (; d + 8 <= head_dim; d += 8) {
			__m256 vt0 = loadu_f16x8_to_ps(vt + d);
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(sv_v, vt0));
		}
		for (; d < head_dim; d++)
			out_h[d] = sv * f16_to_f32_fast(vt[d]);
	}

	for (int t = 1; t < n_pos; t++) {
		float			sv	 = scores[t];
		const uint16_t *vt	 = v_slice + ((size_t)t * kv_stride);
		__m256			sv_v = _mm256_set1_ps(sv);
		int				d	 = 0;
		for (; d + 64 <= head_dim; d += 64) {
			__m256 vt0 = loadu_f16x8_to_ps(vt + d);
			__m256 vt1 = loadu_f16x8_to_ps(vt + d + 8);
			__m256 vt2 = loadu_f16x8_to_ps(vt + d + 16);
			__m256 vt3 = loadu_f16x8_to_ps(vt + d + 24);
			__m256 vt4 = loadu_f16x8_to_ps(vt + d + 32);
			__m256 vt5 = loadu_f16x8_to_ps(vt + d + 40);
			__m256 vt6 = loadu_f16x8_to_ps(vt + d + 48);
			__m256 vt7 = loadu_f16x8_to_ps(vt + d + 56);
			__m256 oh0 = _mm256_loadu_ps(out_h + d);
			__m256 oh1 = _mm256_loadu_ps(out_h + d + 8);
			__m256 oh2 = _mm256_loadu_ps(out_h + d + 16);
			__m256 oh3 = _mm256_loadu_ps(out_h + d + 24);
			__m256 oh4 = _mm256_loadu_ps(out_h + d + 32);
			__m256 oh5 = _mm256_loadu_ps(out_h + d + 40);
			__m256 oh6 = _mm256_loadu_ps(out_h + d + 48);
			__m256 oh7 = _mm256_loadu_ps(out_h + d + 56);
			_mm256_storeu_ps(out_h + d, _mm256_fmadd_ps(vt0, sv_v, oh0));
			_mm256_storeu_ps(out_h + d + 8, _mm256_fmadd_ps(vt1, sv_v, oh1));
			_mm256_storeu_ps(out_h + d + 16, _mm256_fmadd_ps(vt2, sv_v, oh2));
			_mm256_storeu_ps(out_h + d + 24, _mm256_fmadd_ps(vt3, sv_v, oh3));
			_mm256_storeu_ps(out_h + d + 32, _mm256_fmadd_ps(vt4, sv_v, oh4));
			_mm256_storeu_ps(out_h + d + 40, _mm256_fmadd_ps(vt5, sv_v, oh5));
			_mm256_storeu_ps(out_h + d + 48, _mm256_fmadd_ps(vt6, sv_v, oh6));
			_mm256_storeu_ps(out_h + d + 56, _mm256_fmadd_ps(vt7, sv_v, oh7));
		}
		for (; d + 32 <= head_dim; d += 32) {
			__m256 vt0 = loadu_f16x8_to_ps(vt + d);
			__m256 vt1 = loadu_f16x8_to_ps(vt + d + 8);
			__m256 vt2 = loadu_f16x8_to_ps(vt + d + 16);
			__m256 vt3 = loadu_f16x8_to_ps(vt + d + 24);
			_mm256_storeu_ps(out_h + d, _mm256_fmadd_ps(vt0, sv_v, _mm256_loadu_ps(out_h + d)));
			_mm256_storeu_ps(out_h + d + 8,
							 _mm256_fmadd_ps(vt1, sv_v, _mm256_loadu_ps(out_h + d + 8)));
			_mm256_storeu_ps(out_h + d + 16,
							 _mm256_fmadd_ps(vt2, sv_v, _mm256_loadu_ps(out_h + d + 16)));
			_mm256_storeu_ps(out_h + d + 24,
							 _mm256_fmadd_ps(vt3, sv_v, _mm256_loadu_ps(out_h + d + 24)));
		}
		for (; d + 16 <= head_dim; d += 16) {
			__m256 vt0 = loadu_f16x8_to_ps(vt + d);
			__m256 vt1 = loadu_f16x8_to_ps(vt + d + 8);
			_mm256_storeu_ps(out_h + d, _mm256_fmadd_ps(vt0, sv_v, _mm256_loadu_ps(out_h + d)));
			_mm256_storeu_ps(out_h + d + 8,
							 _mm256_fmadd_ps(vt1, sv_v, _mm256_loadu_ps(out_h + d + 8)));
		}
		for (; d + 8 <= head_dim; d += 8) {
			__m256 vt0 = loadu_f16x8_to_ps(vt + d);
			_mm256_storeu_ps(out_h + d, _mm256_fmadd_ps(vt0, sv_v, _mm256_loadu_ps(out_h + d)));
		}
		for (; d < head_dim; d++)
			out_h[d] = fmaf(sv, f16_to_f32_fast(vt[d]), out_h[d]);
	}
}

static float dot8_q8_0(const float *restrict a, const uint8_t *restrict block_ptr, int head_dim) {
	int	  n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	float sum	   = 0.0f;
	for (int b = 0; b < n_blocks; b++) {
		const q8_0_block *blk = (const q8_0_block *)(block_ptr + ((size_t)b * KV_Q8_0_BLOCK_BYTES));
		float			  d	  = f16_to_f32_fast(blk->d);
		int				  base = b * KV_Q8_0_BLOCK;
		int				  n	   = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;

		const int8_t *qs   = blk->qs;
		const float	 *av   = a + base;
		int			  j	   = 0;
		__m256		  acc0 = _mm256_setzero_ps();
		__m256		  acc1 = _mm256_setzero_ps();
		__m256		  acc2 = _mm256_setzero_ps();
		__m256		  acc3 = _mm256_setzero_ps();
		for (; j + 32 <= n; j += 32) {
			__m256 a0 = _mm256_loadu_ps(av + j);
			__m256 a1 = _mm256_loadu_ps(av + j + 8);
			__m256 a2 = _mm256_loadu_ps(av + j + 16);
			__m256 a3 = _mm256_loadu_ps(av + j + 24);
			__m256 q0, q1, q2, q3;
			vld16_s8_to_ps(qs + j, &q0, &q1);
			vld16_s8_to_ps(qs + j + 16, &q2, &q3);
			acc0 = _mm256_fmadd_ps(a0, q0, acc0);
			acc1 = _mm256_fmadd_ps(a1, q1, acc1);
			acc2 = _mm256_fmadd_ps(a2, q2, acc2);
			acc3 = _mm256_fmadd_ps(a3, q3, acc3);
		}
		for (; j + 16 <= n; j += 16) {
			__m256 a0 = _mm256_loadu_ps(av + j);
			__m256 a1 = _mm256_loadu_ps(av + j + 8);
			__m256 q0, q1;
			vld16_s8_to_ps(qs + j, &q0, &q1);
			acc0 = _mm256_fmadd_ps(a0, q0, acc0);
			acc1 = _mm256_fmadd_ps(a1, q1, acc1);
		}
		for (; j + 8 <= n; j += 8) {
			__m256 a0 = _mm256_loadu_ps(av + j);
			__m256 q0 = vld8_s8_to_ps(qs + j);
			acc0	  = _mm256_fmadd_ps(a0, q0, acc0);
		}
		__m256 acc01   = _mm256_add_ps(acc0, acc1);
		__m256 acc23   = _mm256_add_ps(acc2, acc3);
		__m256 acc	   = _mm256_add_ps(acc01, acc23);
		float  partial = vreduce_add_ps(acc);
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
		float			  d	  = f16_to_f32_fast(blk->d) * weight;
		int				  base = b * KV_Q8_0_BLOCK;
		int				  n	   = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;

		const int8_t *qs  = blk->qs;
		float		 *oh  = out_h + base;
		__m256		  d_v = _mm256_set1_ps(d);
		int			  j	  = 0;
		for (; j + 64 <= n; j += 64) {
			__m256 q0, q1, q2, q3, q4, q5, q6, q7;
			vld16_s8_to_ps(qs + j, &q0, &q1);
			vld16_s8_to_ps(qs + j + 16, &q2, &q3);
			vld16_s8_to_ps(qs + j + 32, &q4, &q5);
			vld16_s8_to_ps(qs + j + 48, &q6, &q7);
			_mm256_storeu_ps(oh + j, _mm256_fmadd_ps(d_v, q0, _mm256_loadu_ps(oh + j)));
			_mm256_storeu_ps(oh + j + 8, _mm256_fmadd_ps(d_v, q1, _mm256_loadu_ps(oh + j + 8)));
			_mm256_storeu_ps(oh + j + 16, _mm256_fmadd_ps(d_v, q2, _mm256_loadu_ps(oh + j + 16)));
			_mm256_storeu_ps(oh + j + 24, _mm256_fmadd_ps(d_v, q3, _mm256_loadu_ps(oh + j + 24)));
			_mm256_storeu_ps(oh + j + 32, _mm256_fmadd_ps(d_v, q4, _mm256_loadu_ps(oh + j + 32)));
			_mm256_storeu_ps(oh + j + 40, _mm256_fmadd_ps(d_v, q5, _mm256_loadu_ps(oh + j + 40)));
			_mm256_storeu_ps(oh + j + 48, _mm256_fmadd_ps(d_v, q6, _mm256_loadu_ps(oh + j + 48)));
			_mm256_storeu_ps(oh + j + 56, _mm256_fmadd_ps(d_v, q7, _mm256_loadu_ps(oh + j + 56)));
		}
		for (; j + 16 <= n; j += 16) {
			__m256 q0, q1;
			vld16_s8_to_ps(qs + j, &q0, &q1);
			_mm256_storeu_ps(oh + j, _mm256_fmadd_ps(d_v, q0, _mm256_loadu_ps(oh + j)));
			_mm256_storeu_ps(oh + j + 8, _mm256_fmadd_ps(d_v, q1, _mm256_loadu_ps(oh + j + 8)));
		}
		for (; j + 8 <= n; j += 8) {
			__m256 q0 = vld8_s8_to_ps(qs + j);
			_mm256_storeu_ps(oh + j, _mm256_fmadd_ps(d_v, q0, _mm256_loadu_ps(oh + j)));
		}
		for (; j < n; j++)
			oh[j] = fmaf((float)qs[j], d, oh[j]);
	}
}

static void cpu_attention_inner_q8_0(const uint8_t *restrict k_slice,
									 const uint8_t *restrict v_slice, int kv_stride,
									 const float *qh, float *out_h, int head_dim, int n_pos,
									 float scale, int flash_attn, float *restrict scores) {
	if (flash_attn || scores == NULL) {
		float M = -INFINITY;
		float S = 0.0f;
		float VKQ[HEAD_DIM_MAX] __attribute__((aligned(64)));
		memset(VKQ, 0, (size_t)head_dim * sizeof(float));
		for (int t = 0; t < n_pos; t++) {
			const uint8_t *kt = k_slice + ((size_t)t * kv_stride);
			float		   ss = dot8_q8_0(qh, kt, head_dim) * scale;

			float vs;
			float ms = 1.0f;
			if (ss > M) {
				float M_old = M;
				M			= ss;
				if (S > 0.0f) {
					ms			= expf(M_old - M);
					__m256 ms_v = _mm256_set1_ps(ms);
					int	   d	= 0;
					for (; d + 64 <= head_dim; d += 64) {
						_mm256_storeu_ps(VKQ + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), ms_v));
						_mm256_storeu_ps(VKQ + d + 8,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 8), ms_v));
						_mm256_storeu_ps(VKQ + d + 16,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 16), ms_v));
						_mm256_storeu_ps(VKQ + d + 24,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 24), ms_v));
						_mm256_storeu_ps(VKQ + d + 32,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 32), ms_v));
						_mm256_storeu_ps(VKQ + d + 40,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 40), ms_v));
						_mm256_storeu_ps(VKQ + d + 48,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 48), ms_v));
						_mm256_storeu_ps(VKQ + d + 56,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 56), ms_v));
					}
					for (; d + 32 <= head_dim; d += 32) {
						_mm256_storeu_ps(VKQ + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), ms_v));
						_mm256_storeu_ps(VKQ + d + 8,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 8), ms_v));
						_mm256_storeu_ps(VKQ + d + 16,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 16), ms_v));
						_mm256_storeu_ps(VKQ + d + 24,
										 _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 24), ms_v));
					}
					for (; d + 8 <= head_dim; d += 8)
						_mm256_storeu_ps(VKQ + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), ms_v));
					for (; d < head_dim; d++)
						VKQ[d] *= ms;
				}
				vs = 1.0f;
			} else {
				vs = expf(ss - M);
			}

			const uint8_t *vt = v_slice + ((size_t)t * kv_stride);
			accum_v_q8_0(VKQ, vt, vs, head_dim);
			S = (S * ms) + vs;
		}

		float  S_inv = (S == 0.0f) ? 0.0f : 1.0f / S;
		__m256 inv_v = _mm256_set1_ps(S_inv);
		int	   d	 = 0;
		for (; d + 64 <= head_dim; d += 64) {
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), inv_v));
			_mm256_storeu_ps(out_h + d + 8, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 8), inv_v));
			_mm256_storeu_ps(out_h + d + 16, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 16), inv_v));
			_mm256_storeu_ps(out_h + d + 24, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 24), inv_v));
			_mm256_storeu_ps(out_h + d + 32, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 32), inv_v));
			_mm256_storeu_ps(out_h + d + 40, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 40), inv_v));
			_mm256_storeu_ps(out_h + d + 48, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 48), inv_v));
			_mm256_storeu_ps(out_h + d + 56, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 56), inv_v));
		}
		for (; d + 32 <= head_dim; d += 32) {
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), inv_v));
			_mm256_storeu_ps(out_h + d + 8, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 8), inv_v));
			_mm256_storeu_ps(out_h + d + 16, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 16), inv_v));
			_mm256_storeu_ps(out_h + d + 24, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d + 24), inv_v));
		}
		for (; d + 8 <= head_dim; d += 8)
			_mm256_storeu_ps(out_h + d, _mm256_mul_ps(_mm256_loadu_ps(VKQ + d), inv_v));
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
		__m256 zero = _mm256_setzero_ps();
		int	   d	= 0;
		for (; d + 64 <= head_dim; d += 64) {
			_mm256_storeu_ps(out_h + d, zero);
			_mm256_storeu_ps(out_h + d + 8, zero);
			_mm256_storeu_ps(out_h + d + 16, zero);
			_mm256_storeu_ps(out_h + d + 24, zero);
			_mm256_storeu_ps(out_h + d + 32, zero);
			_mm256_storeu_ps(out_h + d + 40, zero);
			_mm256_storeu_ps(out_h + d + 48, zero);
			_mm256_storeu_ps(out_h + d + 56, zero);
		}
		for (; d + 32 <= head_dim; d += 32) {
			_mm256_storeu_ps(out_h + d, zero);
			_mm256_storeu_ps(out_h + d + 8, zero);
			_mm256_storeu_ps(out_h + d + 16, zero);
			_mm256_storeu_ps(out_h + d + 24, zero);
		}
		for (; d + 8 <= head_dim; d += 8)
			_mm256_storeu_ps(out_h + d, zero);
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

static void cpu_attn_head_chunk_avx(int begin, int end, int tid, void *ctx) {
	cpu_attn_job *j		 = ctx;
	float		 *scores = j->flash_attn ? NULL : cpu_grow_scores(j->p, tid, j->n_pos);
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

static void cpu_attn_kvh_chunk_avx(int begin, int end, int tid, void *ctx) {
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
				cpu_attention_inner_q8_0(k_slice, v_slice, j->hd_stride, qh, out_h, j->head_dim,
										 j->n_pos, j->scale, j->flash_attn, scores);
			} else {
				cpu_attention_inner((uint16_t *)k_slice, (uint16_t *)v_slice, j->hd_stride, qh,
									out_h, j->head_dim, j->n_pos, j->scale, j->flash_attn, scores);
			}
		}
	}
}

static status_code cpu_attention_impl(backend *self, const buffer *q, const buffer *k_cache,
									  const buffer *v_cache, buffer *out, int layer, int pos,
									  int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									  int flash_attn, float scale, int sliding_window,
									  int n_kv_heads_active) {
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	if (n_active <= 0)
		return ERR_INVALID_ARG;

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

	int can_recurse = (tpool_current_tid() < 0);

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
				tpool_parallel_for(p->pool, n_active, 1, cpu_attn_kvh_chunk_avx, &job);
			else
				tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_head_chunk_avx, &job);
			return OK;
		}

		float *scores = flash_attn ? NULL : cpu_grow_scores(p, -1, n_pos);
		for (int h = 0; h < n_heads; h++) {
			int			   kvh	   = h / n_groups;
			const uint8_t *k_slice = kl_base + ((size_t)kvh * kvh_stride);
			const uint8_t *v_slice = vl_base + ((size_t)kvh * kvh_stride);
			const float	  *qh	   = qf + ((size_t)h * head_dim);
			float		  *out_h   = outf + ((size_t)h * head_dim);
			cpu_attention_inner_q8_0(k_slice, v_slice, (int)elem_stride, qh, out_h, head_dim, n_pos,
									 scale, flash_attn, scores);
		}
		return OK;
	}

	int	   hd_stride  = hd_stride_elems;
	size_t kvh_stride = cpu_kv_layer_rows(p, layer, n_ctx) * hd_stride;

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
			tpool_parallel_for(p->pool, n_active, 1, cpu_attn_kvh_chunk_avx, &job);
		else
			tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_head_chunk_avx, &job);
		return OK;
	}

	float *scores = flash_attn ? NULL : cpu_grow_scores(p, -1, n_pos);
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

static void cpu_attn_batch_chunk_avx(int begin, int end, int tid, void *ctx) {
	cpu_attn_batch_job *j			= ctx;
	int					need_scores = j->pos_start + j->m;
	if (j->kv_rows && (size_t)need_scores > j->kv_rows)
		need_scores = (int)j->kv_rows;
	float *scores = j->flash_attn ? NULL : cpu_grow_scores(j->p, tid, need_scores);

	for (int idx = begin; idx < end; idx++) {
		int row = idx / j->n_heads;
		int h	= idx % j->n_heads;

		int pos		   = j->pos_start + row;
		int n_pos	   = pos + 1;
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

static status_code cpu_attention_batch_impl(backend *self, const buffer *q, const buffer *k_cache,
											const buffer *v_cache, buffer *out, int layer,
											int pos_start, int n_heads, int n_kv_heads,
											int head_dim, int n_ctx, int flash_attn, float scale,
											int sliding_window, int n_kv_heads_active, int m) {
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	if (n_active <= 0)
		return ERR_INVALID_ARG;
	int n_groups = (n_heads + n_active - 1) / n_active;

	size_t			  rows = cpu_kv_layer_rows(p, layer, n_ctx);
	size_t			  base = p->kv_base_pos ? p->kv_base_pos[layer] : 0;
	cpu_kv_layer_view view =
		cpu_kv_layer_view_get(p, k_cache, v_cache, layer, n_kv_heads, head_dim, rows);

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
							  .kv_quant		  = p->kv_quant,
							  .kv_rows		  = rows,
							  .kv_base		  = base};

	int total		= n_heads * m;
	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);
	if (can_recurse && p->pool && p->thread_scratch && total >= 2) {
		tpool_parallel_for(p->pool, total, 1, cpu_attn_batch_chunk_avx, &job);
	} else {
		cpu_attn_batch_chunk_avx(0, total, 0, &job);
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

static void cpu_add_inplace_chunk_avx(int begin, int end, int tid, void *ctx) {
	(void)tid;
	float **pp				 = ctx;
	float *restrict xf		 = pp[0];
	const float *restrict yf = pp[1];
	int i					 = begin;
	int n					 = end;
	for (; i + 32 <= n; i += 32) {
		_mm256_storeu_ps(xf + i, _mm256_add_ps(_mm256_loadu_ps(xf + i), _mm256_loadu_ps(yf + i)));
		_mm256_storeu_ps(xf + i + 8,
						 _mm256_add_ps(_mm256_loadu_ps(xf + i + 8), _mm256_loadu_ps(yf + i + 8)));
		_mm256_storeu_ps(xf + i + 16,
						 _mm256_add_ps(_mm256_loadu_ps(xf + i + 16), _mm256_loadu_ps(yf + i + 16)));
		_mm256_storeu_ps(xf + i + 24,
						 _mm256_add_ps(_mm256_loadu_ps(xf + i + 24), _mm256_loadu_ps(yf + i + 24)));
	}
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(xf + i, _mm256_add_ps(_mm256_loadu_ps(xf + i), _mm256_loadu_ps(yf + i)));
	for (; i < n; i++)
		xf[i] += yf[i];
}

status_code cpu_add_inplace(backend *self, buffer *x, const buffer *y, int n) {
	cpu_priv *p		  = self->priv;
	float	 *args[2] = {cpu_ptr(x), (float *)cpu_ptr(y)};
	cpu_run_batch_full(p->pool, n, CPU_ELEMWISE_MIN_PER_THREAD, 2 * CPU_ELEMWISE_MIN_PER_THREAD,
					   cpu_add_inplace_chunk_avx, args);
	return OK;
}

static void cpu_add_batch_chunk_avx(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_add_batch_job *j = ctx;
	int				   n = j->n;
	for (int row = begin; row < end; row++) {
		float		*xr		 = j->x + ((size_t)row * n);
		const float *yr		 = j->y + ((size_t)row * n);
		float		*args[2] = {xr, (float *)yr};
		cpu_add_inplace_chunk_avx(0, n, 0, args);
	}
}

status_code cpu_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
	cpu_priv		 *p	  = self->priv;
	cpu_add_batch_job job = {.x = cpu_ptr(x), .y = cpu_ptr(y), .n = n};
	cpu_run_batch(p->pool, m, cpu_add_batch_chunk_avx, &job);
	return OK;
}

static void cpu_ffn_silu_chunk_avx(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_act_args *a		= ctx;
	const float *restrict g = a->g;
	const float *restrict u = a->u;
	float *restrict o		= a->o;
	int i					= begin;
	int n					= end;
	for (; i + 32 <= n; i += 32) {
		__m256 g0 = _mm256_loadu_ps(g + i);
		__m256 g1 = _mm256_loadu_ps(g + i + 8);
		__m256 g2 = _mm256_loadu_ps(g + i + 16);
		__m256 g3 = _mm256_loadu_ps(g + i + 24);
		__m256 s0 =
			_mm256_div_ps(g0, _mm256_add_ps(_mm256_set1_ps(1.0f),
											vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), g0))));
		__m256 s1 =
			_mm256_div_ps(g1, _mm256_add_ps(_mm256_set1_ps(1.0f),
											vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), g1))));
		__m256 s2 =
			_mm256_div_ps(g2, _mm256_add_ps(_mm256_set1_ps(1.0f),
											vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), g2))));
		__m256 s3 =
			_mm256_div_ps(g3, _mm256_add_ps(_mm256_set1_ps(1.0f),
											vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), g3))));
		_mm256_storeu_ps(o + i, _mm256_mul_ps(s0, _mm256_loadu_ps(u + i)));
		_mm256_storeu_ps(o + i + 8, _mm256_mul_ps(s1, _mm256_loadu_ps(u + i + 8)));
		_mm256_storeu_ps(o + i + 16, _mm256_mul_ps(s2, _mm256_loadu_ps(u + i + 16)));
		_mm256_storeu_ps(o + i + 24, _mm256_mul_ps(s3, _mm256_loadu_ps(u + i + 24)));
	}
	for (; i + 8 <= n; i += 8) {
		__m256 g0 = _mm256_loadu_ps(g + i);
		__m256 s0 =
			_mm256_div_ps(g0, _mm256_add_ps(_mm256_set1_ps(1.0f),
											vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), g0))));
		_mm256_storeu_ps(o + i, _mm256_mul_ps(s0, _mm256_loadu_ps(u + i)));
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
	cpu_run_batch_full(p->pool, n, CPU_ELEMWISE_MIN_PER_THREAD, 2 * CPU_ELEMWISE_MIN_PER_THREAD,
					   cpu_ffn_silu_chunk_avx, &a);
	return OK;
}

static void cpu_ffn_gelu_chunk_avx(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_act_args *a		= ctx;
	const float *restrict g = a->g;
	const float *restrict u = a->u;
	float *restrict o		= a->o;
	const float c_fit		= 0.7978845608028654f;
	const float c_x3		= 0.044715f;
	int			i			= begin;
	int			n			= end;
	for (; i + 32 <= n; i += 32) {
		__m256 g0	= _mm256_loadu_ps(g + i);
		__m256 g1	= _mm256_loadu_ps(g + i + 8);
		__m256 g2	= _mm256_loadu_ps(g + i + 16);
		__m256 g3	= _mm256_loadu_ps(g + i + 24);
		__m256 g0sq = _mm256_mul_ps(g0, g0);
		__m256 g1sq = _mm256_mul_ps(g1, g1);
		__m256 g2sq = _mm256_mul_ps(g2, g2);
		__m256 g3sq = _mm256_mul_ps(g3, g3);
		__m256 in0	= _mm256_mul_ps(
			_mm256_set1_ps(c_fit),
			_mm256_mul_ps(g0, _mm256_add_ps(_mm256_set1_ps(1.0f),
											_mm256_mul_ps(_mm256_set1_ps(c_x3), g0sq))));
		__m256 in1 = _mm256_mul_ps(
			_mm256_set1_ps(c_fit),
			_mm256_mul_ps(g1, _mm256_add_ps(_mm256_set1_ps(1.0f),
											_mm256_mul_ps(_mm256_set1_ps(c_x3), g1sq))));
		__m256 in2 = _mm256_mul_ps(
			_mm256_set1_ps(c_fit),
			_mm256_mul_ps(g2, _mm256_add_ps(_mm256_set1_ps(1.0f),
											_mm256_mul_ps(_mm256_set1_ps(c_x3), g2sq))));
		__m256 in3 = _mm256_mul_ps(
			_mm256_set1_ps(c_fit),
			_mm256_mul_ps(g3, _mm256_add_ps(_mm256_set1_ps(1.0f),
											_mm256_mul_ps(_mm256_set1_ps(c_x3), g3sq))));
		__m256 t0 = _mm256_add_ps(_mm256_set1_ps(1.0f), vtanh_ps(in0));
		__m256 t1 = _mm256_add_ps(_mm256_set1_ps(1.0f), vtanh_ps(in1));
		__m256 t2 = _mm256_add_ps(_mm256_set1_ps(1.0f), vtanh_ps(in2));
		__m256 t3 = _mm256_add_ps(_mm256_set1_ps(1.0f), vtanh_ps(in3));
		_mm256_storeu_ps(o + i,
						 _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), g0), t0),
									   _mm256_loadu_ps(u + i)));
		_mm256_storeu_ps(o + i + 8,
						 _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), g1), t1),
									   _mm256_loadu_ps(u + i + 8)));
		_mm256_storeu_ps(o + i + 16,
						 _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), g2), t2),
									   _mm256_loadu_ps(u + i + 16)));
		_mm256_storeu_ps(o + i + 24,
						 _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), g3), t3),
									   _mm256_loadu_ps(u + i + 24)));
	}
	for (; i < n; i++) {
		float x		= g[i];
		float x3	= x * x * x;
		float inner = c_fit * (x + (c_x3 * x3));
		float t		= tanhf(inner);
		o[i]		= 0.5f * x * (1.0f + t) * u[i];
	}
}

status_code cpu_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up, buffer *out,
								int n, int activation) {
	cpu_priv		*p	= self->priv;
	cpu_ffn_act_args a	= {.g = cpu_ptr(gate), .u = cpu_ptr(up), .o = cpu_ptr(out)};
	tpool_chunk_fn	 fn = activation == 1 ? cpu_ffn_gelu_chunk_avx : cpu_ffn_silu_chunk_avx;
	cpu_run_batch_full(p->pool, n, CPU_ELEMWISE_MIN_PER_THREAD, 2 * CPU_ELEMWISE_MIN_PER_THREAD, fn,
					   &a);
	return OK;
}

static void cpu_ffn_act_batch_chunk_avx(int begin, int end, int tid, void *ctx) {
	cpu_ffn_act_batch_args *j = ctx;
	tpool_chunk_fn fn = j->activation == 1 ? cpu_ffn_gelu_chunk_avx : cpu_ffn_silu_chunk_avx;
	for (int row = begin; row < end; row++) {
		cpu_ffn_act_args a = {
			.g = j->g + ((size_t)row * j->n),
			.u = j->u + ((size_t)row * j->n),
			.o = j->o + ((size_t)row * j->n),
		};
		fn(0, j->n, tid, &a);
	}
}

status_code cpu_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n, int activation, int m) {
	cpu_priv			  *p   = self->priv;
	cpu_ffn_act_batch_args job = {
		.g = cpu_ptr(gate), .u = cpu_ptr(up), .o = cpu_ptr(out), .n = n, .activation = activation};
	cpu_run_batch(p->pool, m, cpu_ffn_act_batch_chunk_avx, &job);
	return OK;
}

void cpu_ffn_down_act_chunk(int begin, int end, int tid, void *ctx) {
	cpu_ffn_down_act_args *a   = ctx;
	cpu_ffn_act_args	   sub = {.g = a->g, .u = a->u, .o = a->o};
	if (a->activation == 1)
		cpu_ffn_gelu_chunk_avx(begin, end, tid, &sub);
	else
		cpu_ffn_silu_chunk_avx(begin, end, tid, &sub);
}

void cpu_softcap_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_softcap_job *j		 = ctx;
	float			 inv_cap = j->inv_cap;
	float			 cap	 = j->cap;
	__m256			 vinv	 = _mm256_set1_ps(inv_cap);
	__m256			 vcap	 = _mm256_set1_ps(cap);
	int				 i		 = begin;
	for (; i + 8 <= end; i += 8) {
		__m256 x = _mm256_mul_ps(_mm256_loadu_ps(j->x + i), vinv);
		_mm256_storeu_ps(j->x + i, _mm256_mul_ps(vtanh_ps(x), vcap));
	}
	for (; i < end; i++)
		j->x[i] = cap * tanhf(j->x[i] * inv_cap);
}

void cpu_split_qgate_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_split_qgate_job *j			  = ctx;
	const int			 hd			  = j->head_dim;
	int					 q_out		  = j->n_heads * hd;
	int					 mixed_stride = 2 * q_out;
	for (int row = begin; row < end; row++) {
		const float *src = j->mixed + (size_t)row * mixed_stride;
		float		*qd	 = j->q + (size_t)row * q_out;
		float		*gd	 = j->gate + (size_t)row * q_out;
		for (int h = 0; h < j->n_heads; h++, src += 2 * hd) {
			int jj = 0;
			for (; jj + 8 <= hd; jj += 8) {
				_mm256_storeu_ps(qd + jj, _mm256_loadu_ps(src + jj));
				_mm256_storeu_ps(gd + jj, _mm256_loadu_ps(src + hd + jj));
			}
			for (; jj + 4 <= hd; jj += 4) {
				_mm_storeu_ps(qd + jj, _mm_loadu_ps(src + jj));
				_mm_storeu_ps(gd + jj, _mm_loadu_ps(src + hd + jj));
			}
			for (; jj < hd; jj++) {
				qd[jj] = src[jj];
				gd[jj] = src[hd + jj];
			}
			qd += hd;
			gd += hd;
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
		int			 i = 0;
		for (; i + 8 <= n; i += 8) {
			__m256 gv = _mm256_loadu_ps(g + i);
			__m256 sig =
				_mm256_div_ps(_mm256_set1_ps(1.0f),
							  _mm256_add_ps(_mm256_set1_ps(1.0f),
											vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), gv))));
			_mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_loadu_ps(o + i), sig));
		}
		for (; i < n; i++)
			o[i] *= sigmoidf(g[i]);
	}
}

static void cpu_rmsnorm_add_batch_chunk_avx(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rmsnorm_add_batch_job *j	  = ctx;
	const int				   n	  = j->n;
	const float				  *w	  = j->w;
	const float				   oscale = j->out_scale;
	for (int row = begin; row < end; row++) {
		const float *xr		  = j->x + ((size_t)row * n);
		const float *rr		  = j->residual + ((size_t)row * n);
		float		*yr		  = j->y + ((size_t)row * n);
		float		 ss		  = rmsnorm_sum_sq(xr, n);
		float		 scale	  = 1.0f / sqrtf((ss / (float)n) + j->eps);
		__m256		 scale_v  = _mm256_set1_ps(scale);
		__m256		 oscale_v = _mm256_set1_ps(oscale);
		int			 i		  = 0;
		for (; i + 32 <= n; i += 32) {
			__m256 t0 = _mm256_mul_ps(_mm256_loadu_ps(xr + i), scale_v);
			__m256 t1 = _mm256_mul_ps(_mm256_loadu_ps(xr + i + 8), scale_v);
			__m256 t2 = _mm256_mul_ps(_mm256_loadu_ps(xr + i + 16), scale_v);
			__m256 t3 = _mm256_mul_ps(_mm256_loadu_ps(xr + i + 24), scale_v);
			_mm256_storeu_ps(yr + i, _mm256_mul_ps(_mm256_fmadd_ps(t0, _mm256_loadu_ps(w + i),
																   _mm256_loadu_ps(rr + i)),
												   oscale_v));
			_mm256_storeu_ps(yr + i + 8,
							 _mm256_mul_ps(_mm256_fmadd_ps(t1, _mm256_loadu_ps(w + i + 8),
														   _mm256_loadu_ps(rr + i + 8)),
										   oscale_v));
			_mm256_storeu_ps(yr + i + 16,
							 _mm256_mul_ps(_mm256_fmadd_ps(t2, _mm256_loadu_ps(w + i + 16),
														   _mm256_loadu_ps(rr + i + 16)),
										   oscale_v));
			_mm256_storeu_ps(yr + i + 24,
							 _mm256_mul_ps(_mm256_fmadd_ps(t3, _mm256_loadu_ps(w + i + 24),
														   _mm256_loadu_ps(rr + i + 24)),
										   oscale_v));
		}
		for (; i + 8 <= n; i += 8) {
			__m256 t = _mm256_mul_ps(_mm256_loadu_ps(xr + i), scale_v);
			_mm256_storeu_ps(yr + i, _mm256_mul_ps(_mm256_fmadd_ps(t, _mm256_loadu_ps(w + i),
																   _mm256_loadu_ps(rr + i)),
												   oscale_v));
		}
		for (; i < n; i++)
			yr[i] = (xr[i] * scale * w[i] + rr[i]) * oscale;
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
	cpu_run_batch(p->pool, m, cpu_rmsnorm_add_batch_chunk_avx, &job);
	return OK;
}

status_code cpu_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx) {
	(void)self;
	*out_idx = cpu_argmax_f32(cpu_ptr(logits), n);
	return OK;
}

int32_t cpu_argmax_f32(const float *restrict logits, int vocab) {
	if (vocab <= 0)
		return 0;
	__m256		  best_v = _mm256_set1_ps(-INFINITY);
	__m256i		  best_i = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i		  idx	 = best_i;
	const __m256i stride = _mm256_set1_epi32(8);
	int32_t		  i		 = 0;
	for (; i + 8 <= vocab; i += 8) {
		__m256 v	= _mm256_loadu_ps(logits + i);
		__m256 mask = _mm256_cmp_ps(v, best_v, _CMP_GT_OQ);
		best_v		= _mm256_blendv_ps(best_v, v, mask);
		best_i		= _mm256_blendv_epi8(best_i, idx, _mm256_castps_si256(mask));
		idx			= _mm256_add_epi32(idx, stride);
	}

	float	vals[8];
	int32_t idxs[8];
	_mm256_storeu_ps(vals, best_v);
	_mm256_storeu_si256((__m256i *)idxs, best_i);
	int	  best	= idxs[0];
	float bestv = vals[0];
	for (int k = 1; k < 8; k++) {
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

static void cpu_attention_mla_head_avx(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_mla_job *j = ctx;

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
			__m256		 acc = _mm256_setzero_ps();
			int			 d	 = 0;
			for (; d + 8 <= j->qk_nope; d += 8)
				acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + d), _mm256_loadu_ps(q_h + d), acc);
			float sum = vreduce_add_ps(acc);
			for (; d < j->qk_nope; d++)
				sum += row[d] * q_h[d];
			q_absorbed[i] = sum;
		}
		const float *q_rope_part = q_h + j->qk_nope;

		float M = -INFINITY;
		float S = 0.0f;
		memset(VKQ_latent, 0, (size_t)j->kv_lora * sizeof(float));
		for (int t = 0; t < j->n_pos; t++) {
			const float *slot	  = j->layer_c + ((size_t)t * (j->kv_lora + j->qk_rope));
			const float *latent	  = slot;
			const float *k_pe_rot = j->k_pe_rot_all + ((size_t)t * j->qk_rope);

			__m256 acc0 = _mm256_setzero_ps();
			int	   i	= 0;
			for (; i + 8 <= j->kv_lora; i += 8)
				acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(latent + i), _mm256_loadu_ps(q_absorbed + i),
									   acc0);
			float score = vreduce_add_ps(acc0);
			for (; i < j->kv_lora; i++)
				score += latent[i] * q_absorbed[i];

			__m256 acc1 = _mm256_setzero_ps();
			int	   d	= 0;
			for (; d + 8 <= j->qk_rope; d += 8)
				acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q_rope_part + d),
									   _mm256_loadu_ps(k_pe_rot + d), acc1);
			score += vreduce_add_ps(acc1);
			for (; d < j->qk_rope; d++)
				score += q_rope_part[d] * k_pe_rot[d];
			score *= j->scale;

			float ms = 1.0f;
			float vs = 1.0f;
			if (score > M) {
				float Mold = M;
				M		   = score;
				ms		   = expf(Mold - M);
				__m256 msv = _mm256_set1_ps(ms);
				int	   i2  = 0;
				for (; i2 + 8 <= j->kv_lora; i2 += 8)
					_mm256_storeu_ps(VKQ_latent + i2,
									 _mm256_mul_ps(_mm256_loadu_ps(VKQ_latent + i2), msv));
				for (; i2 < j->kv_lora; i2++)
					VKQ_latent[i2] *= ms;
			} else {
				vs = expf(score - M);
			}
			__m256 vsv = _mm256_set1_ps(vs);
			int	   i3  = 0;
			for (; i3 + 8 <= j->kv_lora; i3 += 8)
				_mm256_storeu_ps(VKQ_latent + i3,
								 _mm256_fmadd_ps(_mm256_loadu_ps(latent + i3), vsv,
												 _mm256_loadu_ps(VKQ_latent + i3)));
			for (; i3 < j->kv_lora; i3++)
				VKQ_latent[i3] = fmaf(latent[i3], vs, VKQ_latent[i3]);
			S = (S * ms) + vs;
		}
		float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
		float VKQ[HEAD_DIM_MAX];
		for (int d = 0; d < j->v_head; d++) {
			const float *row = v_b_h + ((size_t)d * j->kv_lora);
			__m256		 acc = _mm256_setzero_ps();
			int			 i	 = 0;
			for (; i + 8 <= j->kv_lora; i += 8)
				acc =
					_mm256_fmadd_ps(_mm256_loadu_ps(VKQ_latent + i), _mm256_loadu_ps(row + i), acc);
			float sum = vreduce_add_ps(acc);
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
	for (; jj + 2 <= j->half_rope; jj += 2) {
		const float *base = k_pe + 2 * jj;
		__m128		 p	  = _mm_loadu_ps(base);
		__m128		 c	  = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)(t_rope_cos + jj)));
		__m128		 s	  = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)(t_rope_sin + jj)));
		__m128		 ev	  = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 0, 2, 0));
		__m128		 od	  = _mm_shuffle_ps(p, p, _MM_SHUFFLE(3, 1, 3, 1));
		__m128		 re	  = _mm_sub_ps(_mm_mul_ps(ev, c), _mm_mul_ps(od, s));
		__m128		 im	  = _mm_add_ps(_mm_mul_ps(ev, s), _mm_mul_ps(od, c));
		_mm_storeu_ps(dst + 2 * jj, _mm_unpacklo_ps(re, im));
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
	cpu_mla_precompute_k_pe_rot(p, layer_c, k_pe_rot_all, rope_cos_base, rope_sin_base, total_dim,
								kv_lora, qk_rope, half_rope, n_pos, layer);

	for (int i = 0; i < n_heads * v_head; i++)
		outf[i] = 0.0f;

	cpu_mla_job job = {.qf			  = qf,
					   .outf		  = outf,
					   .layer_c		  = layer_c,
					   .k_b			  = k_b,
					   .v_b			  = v_b,
					   .k_pe_rot_all  = k_pe_rot_all,
					   .rope_cos_base = rope_cos_base,
					   .rope_sin_base = rope_sin_base,
					   .qk_head		  = qk_head,
					   .qk_rope		  = qk_rope,
					   .qk_nope		  = qk_nope,
					   .v_head		  = v_head,
					   .kv_lora		  = kv_lora,
					   .n_pos		  = n_pos,
					   .half_rope	  = half_rope,
					   .pos			  = pos,
					   .scale		  = scale};

	if (p->pool && n_heads > 1 &&
		(size_t)n_heads * (size_t)n_pos * (size_t)(qk_head + v_head) >= 4096) {
		tpool_parallel_for(p->pool, n_heads, 1, cpu_attention_mla_head_avx, &job);
	} else {
		cpu_attention_mla_head_avx(0, n_heads, 0, &job);
	}

	return OK;
}

typedef void (*x86_qonly_fn)(const void *, const void *, size_t, float *, int, int, int, int);

typedef struct {
	x86_qonly_fn qonly;
	uint32_t	 expected_q8;
} x86_matmul_kernel;

static const x86_matmul_kernel x86_kernel_by_type[GGML_TYPE_Q6_K_REPACKED + 1] = {
	[GGML_TYPE_Q4_0]			= {(x86_qonly_fn)matmul_q4_q8_qonly_f32, GGML_TYPE_Q8_0},
	[GGML_TYPE_IQ4_NL]			= {(x86_qonly_fn)matmul_iq4_nl_q8_qonly_f32, GGML_TYPE_Q8_0},
	[GGML_TYPE_Q8_0]			= {(x86_qonly_fn)matmul_q8_0_q8_qonly_f32, GGML_TYPE_Q8_0},
	[GGML_TYPE_Q4_1]			= {(x86_qonly_fn)matmul_q4_1_q8_qonly_f32, GGML_TYPE_Q8_1},
	[GGML_TYPE_Q5_0]			= {(x86_qonly_fn)matmul_q5_0_q8_qonly_f32, GGML_TYPE_Q8_0},
	[GGML_TYPE_Q5_1]			= {(x86_qonly_fn)matmul_q5_1_q8_qonly_f32, GGML_TYPE_Q8_1},
	[GGML_TYPE_Q6_K]			= {(x86_qonly_fn)matmul_q6_k_q8_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_Q4_K]			= {(x86_qonly_fn)matmul_q4_k_q8_k_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_Q5_K]			= {(x86_qonly_fn)matmul_q5_k_q8_k_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_IQ3_S]			= {(x86_qonly_fn)matmul_iq3_s_q8_k_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_IQ3_S_REPACKED]	= {(x86_qonly_fn)matmul_iq3_s_repacked_q8_k_qonly_f32,
								   GGML_TYPE_Q8_K},
	[GGML_TYPE_IQ2_S]			= {(x86_qonly_fn)matmul_iq2_s_q8_k_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_IQ2_XXS]			= {(x86_qonly_fn)matmul_iq2_xxs_q8_k_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_IQ3_XXS]			= {(x86_qonly_fn)matmul_iq3_xxs_q8_k_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_IQ4_XS]			= {(x86_qonly_fn)matmul_iq4_xs_q8_k_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_Q8_0_REPACKED]	= {(x86_qonly_fn)matmul_q8_0_repacked_q8_qonly_f32, GGML_TYPE_Q8_0},
	[GGML_TYPE_Q4_0_REPACKED]	= {(x86_qonly_fn)matmul_q4_0_repacked_q8_qonly_f32, GGML_TYPE_Q8_0},
	[GGML_TYPE_IQ4_NL_REPACKED] = {(x86_qonly_fn)matmul_iq4_nl_repacked_q8_qonly_f32,
								   GGML_TYPE_Q8_0},
	[GGML_TYPE_Q4_K_REPACKED] = {(x86_qonly_fn)matmul_q4_k_repacked_q8_k_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_Q5_K_REPACKED] = {(x86_qonly_fn)matmul_q5_k_repacked_q8_k_qonly_f32, GGML_TYPE_Q8_K},
	[GGML_TYPE_Q6_K_REPACKED] = {(x86_qonly_fn)matmul_q6_k_repacked_q8_k_qonly_f32, GGML_TYPE_Q8_K},
};

status_code cpu_matmul_qonly(backend *self, const buffer *w, uint32_t w_type, const buffer *x_q8,
							 uint32_t q8_type, buffer *y, int n, int k, int m) {
	(void)self;
	if (w_type > GGML_TYPE_Q6_K_REPACKED || !x86_kernel_by_type[w_type].qonly)
		return ERR_UNSUPPORTED;
	const x86_matmul_kernel *entry = &x86_kernel_by_type[w_type];
	if (q8_type != entry->expected_q8)
		return ERR_INVALID_ARG;
	if (m <= 0)
		m = 1;
	entry->qonly(cpu_ptr(w), x_q8->handle, (size_t)(k / 32), cpu_ptr(y), n, n, k, m);
	return OK;
}

void detect_features(char *buf, size_t cap) {
	buf[0] = '\0';
	feat_add(buf, cap, "x86_64");
#if defined(__AVX2__)
	feat_add(buf, cap, "avx2");
#endif
#if defined(__FMA__)
	feat_add(buf, cap, "fma");
#endif
#if defined(__F16C__)
	feat_add(buf, cap, "f16c");
#endif
}
CPU_BACKEND_REGISTER("cpu_x86_64", cpu_arch_ctor, 10, "host x86_64 SIMD optimized")
