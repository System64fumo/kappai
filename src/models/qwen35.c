#include "backend/backend.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "recipe.h"
#include "threadpool.h"

#include <math.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

static float gdn_sigmoid(float x) {
	if (x >= 0.0f) {
		float z = expf(-x);
		return 1.0f / (1.0f + z);
	}
	float z = expf(x);
	return z / (1.0f + z);
}

static float gdn_silu(float x) {
	return x * gdn_sigmoid(x);
}

static float gdn_softplus(float x) {
	if (x > 20.0f)
		return x;
	if (x < -20.0f)
		return expf(x);
	return log1pf(expf(x));
}

static void split_qgate_rows(const float *mixed, float *q, float *gate, int n_heads, int head_dim,
							 int n_rows) {
	int q_out		 = n_heads * head_dim;
	int mixed_stride = 2 * q_out;
	for (int row = 0; row < n_rows; row++) {
		const float *src = mixed + (size_t)row * mixed_stride;
		float		*qd	 = q + (size_t)row * q_out;
		float		*gd	 = gate + (size_t)row * q_out;
		for (int h = 0; h < n_heads; h++, src += 2 * head_dim) {
			for (int j = 0; j < head_dim; j++) {
				qd[j] = src[j];
				gd[j] = src[j + head_dim];
			}
			qd += head_dim;
			gd += head_dim;
		}
	}
}

status_code op_split_qgate(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	model		*m		  = ctx->m;
	int			 head_dim = m->head_dim;
	int			 n_heads  = m->n_heads;
	const float *mixed	  = recipe_slot_f32(ctx, RECIPE_SLOT_HYB_PROJ);
	float		*q		  = recipe_slot_f32(ctx, RECIPE_SLOT_Q);
	float		*gate	  = recipe_slot_f32(ctx, RECIPE_SLOT_HYB_GATE);
	if (!mixed || !q || !gate)
		return ERR_INVALID_ARG;
	split_qgate_rows(mixed, q, gate, n_heads, head_dim,
					 recipe_exec_is_batch(ctx) ? ctx->n_rows : 1);
	return OK;
}

static void partial_rope_one(float *x, int n_heads, int head_dim, int rope_dim, const float *cosv,
							 const float *sinv) {
	int half = rope_dim / 2;
	for (int h = 0; h < n_heads; h++) {
		float *v = x + (size_t)h * head_dim;
		for (int j = 0; j < half; j++) {
			float a		= v[j];
			float b		= v[j + half];
			v[j]		= a * cosv[j] - b * sinv[j];
			v[j + half] = a * sinv[j] + b * cosv[j];
		}
	}
}

status_code op_partial_rope_qk(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s || ctx->pos < 0)
		return ERR_INVALID_ARG;
	model *m	= ctx->m;
	int	   qn	= m->n_heads * m->head_dim;
	int	   kn	= m->n_kv_heads * m->head_dim;
	int	   half = m->rope_dim / 2;
	int	   rows = recipe_exec_is_batch(ctx) ? ctx->n_rows : 1;
	int	   pos0 = recipe_exec_is_batch(ctx) ? ctx->pos_start : ctx->pos;
	float *q	= recipe_slot_f32(ctx, RECIPE_SLOT_Q);
	float *k	= recipe_slot_f32(ctx, RECIPE_SLOT_K);
	if (!q || !k)
		return ERR_INVALID_ARG;
	for (int row = 0; row < rows; row++) {
		const float *cosv = ctx->s->rope_cos + (size_t)(pos0 + row) * half;
		const float *sinv = ctx->s->rope_sin + (size_t)(pos0 + row) * half;
		partial_rope_one(q + (size_t)row * qn, m->n_heads, m->head_dim, m->rope_dim, cosv, sinv);
		partial_rope_one(k + (size_t)row * kn, m->n_kv_heads, m->head_dim, m->rope_dim, cosv, sinv);
	}
	return OK;
}

status_code op_attn_output_gate(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	int			 n	  = ctx->m->n_heads * ctx->m->head_dim;
	int			 rows = recipe_exec_is_batch(ctx) ? ctx->n_rows : 1;
	float		*out  = recipe_slot_f32(ctx, ctx->op->in[0]);
	const float *gate = recipe_slot_f32(ctx, RECIPE_SLOT_HYB_GATE);
	if (!out || !gate)
		return ERR_INVALID_ARG;
	for (int row = 0; row < rows; row++) {
		float		*o = out + (size_t)row * n;
		const float *g = gate + (size_t)row * n;
		for (int i = 0; i < n; i++)
			o[i] *= gdn_sigmoid(g[i]);
	}
	return OK;
}

static void gdn_conv_tokens(float *conv_out, float *conv_state, const float *mixed,
							const float *conv_w, int conv_dim, int conv_kernel, int n_tokens,
							int mixed_stride) {
	int history = conv_kernel - 1;
	if (history == 3) {
		for (int t = 0; t < n_tokens; t++) {
			const float *mix = mixed + (size_t)t * mixed_stride;
			float		*out = conv_out + (size_t)t * conv_dim;
			for (int c = 0; c < conv_dim; c++) {
				const float *w	  = conv_w + (size_t)c * 4;
				float		*hist = conv_state + (size_t)c * 3;
				float		 sum = hist[0] * w[0] + hist[1] * w[1] + hist[2] * w[2] + mix[c] * w[3];
				hist[0]			 = hist[1];
				hist[1]			 = hist[2];
				hist[2]			 = mix[c];
				out[c]			 = gdn_silu(sum);
			}
		}
		return;
	}
	for (int t = 0; t < n_tokens; t++) {
		const float *mix = mixed + (size_t)t * mixed_stride;
		float		*out = conv_out + (size_t)t * conv_dim;
		for (int c = 0; c < conv_dim; c++) {
			const float *w	 = conv_w + (size_t)c * conv_kernel;
			float		 sum = mix[c] * w[history];
			if (history > 0) {
				float *hist = conv_state + (size_t)c * history;
				for (int j = 0; j < history; j++)
					sum += hist[j] * w[j];
				if (history > 1)
					memmove(hist, hist + 1, (size_t)(history - 1) * sizeof(float));
				hist[history - 1] = mix[c];
			}
			out[c] = gdn_silu(sum);
		}
	}
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

#if defined(__AVX2__)
static inline float qwen_dot_f32(const float *a, const float *b, int n) {
	__m256 acc = _mm256_setzero_ps();
	int	   i   = 0;
	for (; i + 8 <= n; i += 8)
		acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
	__m128 lo = _mm256_castps256_ps128(acc);
	__m128 hi = _mm256_extractf128_ps(acc, 1);
	__m128 s  = _mm_add_ps(lo, hi);
	s		  = _mm_add_ps(s, _mm_movehdup_ps(s));
	s		  = _mm_add_ss(s, _mm_movehl_ps(s, s));
	float sum = _mm_cvtss_f32(s);
	for (; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

static inline void qwen_scale_f32(float *dst, const float *src, float scale, int n) {
	__m256 vs = _mm256_set1_ps(scale);
	int	   i  = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vs));
	for (; i < n; i++)
		dst[i] = src[i] * scale;
}

static inline void qwen_state_decay_mem(float *row, float *mem, float decay, float ks, int n) {
	__m256 vd = _mm256_set1_ps(decay);
	__m256 vk = _mm256_set1_ps(ks);
	int	   i  = 0;
	for (; i + 8 <= n; i += 8) {
		__m256 r = _mm256_mul_ps(_mm256_loadu_ps(row + i), vd);
		_mm256_storeu_ps(row + i, r);
		_mm256_storeu_ps(mem + i, _mm256_fmadd_ps(r, vk, _mm256_loadu_ps(mem + i)));
	}
	for (; i < n; i++) {
		row[i] *= decay;
		mem[i] += row[i] * ks;
	}
}

static inline void qwen_state_update_out(float *row, float *y, const float *delta, float ks,
										 float qs, int n) {
	__m256 vk = _mm256_set1_ps(ks);
	__m256 vq = _mm256_set1_ps(qs);
	int	   i  = 0;
	for (; i + 8 <= n; i += 8) {
		__m256 r = _mm256_fmadd_ps(vk, _mm256_loadu_ps(delta + i), _mm256_loadu_ps(row + i));
		_mm256_storeu_ps(row + i, r);
		_mm256_storeu_ps(y + i, _mm256_fmadd_ps(r, vq, _mm256_loadu_ps(y + i)));
	}
	for (; i < n; i++) {
		row[i] += ks * delta[i];
		y[i] += row[i] * qs;
	}
}

static inline void qwen_axpy_sub_scale(float *delta, const float *v, const float *mem, float b,
									   int n) {
	__m256 vb = _mm256_set1_ps(b);
	int	   i  = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(
			delta + i,
			_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + i), _mm256_loadu_ps(mem + i)), vb));
	for (; i < n; i++)
		delta[i] = (v[i] - mem[i]) * b;
}
#endif

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

			float decay = expf(j->a[vh] * gdn_softplus(alpha_t + j->dt[vh]));
			float b		= gdn_sigmoid(beta_t);
#if defined(__AVX2__)
			float qn = q_scale / sqrtf(j->eps + qwen_dot_f32(q, q, kd));
			float kn = 1.0f / sqrtf(j->eps + qwen_dot_f32(k, k, kd));
			qwen_scale_f32(q_s, q, qn, kd);
			qwen_scale_f32(k_s, k, kn, kd);
			memset(mem, 0, (size_t)vd * sizeof(float));
			for (int d = 0; d < kd; d++)
				qwen_state_decay_mem(shead + (size_t)d * vd, mem, decay, k_s[d], vd);
			qwen_axpy_sub_scale(delta, v, mem, b, vd);
			memset(y, 0, (size_t)vd * sizeof(float));
			for (int d = 0; d < kd; d++)
				qwen_state_update_out(shead + (size_t)d * vd, y, delta, k_s[d], q_s[d], vd);
			float inv_rms = 1.0f / sqrtf(j->eps + qwen_dot_f32(y, y, vd) / (float)vd);
			for (int jj = 0; jj < vd; jj++)
				y[jj] = y[jj] * inv_rms * j->norm_w[jj] * gdn_silu(z_t[jj]);
#else
			float qn = j->eps;
			float kn = j->eps;
			for (int d = 0; d < kd; d++) {
				qn += q[d] * q[d];
				kn += k[d] * k[d];
			}
			qn = q_scale / sqrtf(qn);
			kn = 1.0f / sqrtf(kn);
			for (int d = 0; d < kd; d++) {
				q_s[d] = q[d] * qn;
				k_s[d] = k[d] * kn;
			}

			memset(mem, 0, (size_t)vd * sizeof(float));
			for (int d = 0; d < kd; d++) {
				float *row = shead + (size_t)d * vd;
				float  ks  = k_s[d];
				for (int jj = 0; jj < vd; jj++) {
					row[jj] *= decay;
					mem[jj] += row[jj] * ks;
				}
			}
			for (int jj = 0; jj < vd; jj++)
				delta[jj] = (v[jj] - mem[jj]) * b;
			memset(y, 0, (size_t)vd * sizeof(float));
			for (int d = 0; d < kd; d++) {
				float *row = shead + (size_t)d * vd;
				float  ks  = k_s[d];
				float  qs  = q_s[d];
				for (int jj = 0; jj < vd; jj++) {
					row[jj] += ks * delta[jj];
					y[jj] += row[jj] * qs;
				}
			}

			float mean_sq = j->eps;
			for (int jj = 0; jj < vd; jj++)
				mean_sq += y[jj] * y[jj] / (float)vd;
			float inv_rms = 1.0f / sqrtf(mean_sq);
			for (int jj = 0; jj < vd; jj++)
				y[jj] = y[jj] * inv_rms * j->norm_w[jj] * gdn_silu(z_t[jj]);
#endif
		}
	}
}

static void gdn_chunk(int begin, int end, int tid, void *ctx) {
	gdn_job *j = (gdn_job *)ctx;
	gdn_heads(begin, end, j, j->scratch + (size_t)tid * j->scratch_stride);
}

static status_code gdn_run(exec_ctx *ctx, const float *mixed, const float *z, const float *alpha,
						   const float *beta, float *out, float *ws, tpool *pool) {
	model					  *m	  = ctx->m;
	const model_hybrid_params *p	  = &m->hybrid;
	layer_weights			  *L	  = &m->layers[ctx->li];
	const float				  *conv_w = (const float *)L->ssm_conv1d_w.host_ptr;
	const float				  *dt	  = (const float *)L->ssm_dt_b.host_ptr;
	const float				  *a	  = (const float *)L->ssm_a.host_ptr;
	const float				  *norm_w = (const float *)L->ssm_norm_w.host_ptr;
	if (!conv_w || !dt || !a || !norm_w)
		return ERR_FORMAT;

	kvcache_hybrid *cache	   = ctx->cache->hybrid;
	float		   *conv_state = cache->conv_state + (size_t)ctx->li * cache->conv_stride;
	float		   *state	   = cache->recurrent_state + (size_t)ctx->li * cache->recurrent_stride;
	if (ctx->pos_start == 0) {
		memset(conv_state, 0, cache->conv_stride * sizeof(float));
		memset(state, 0, cache->recurrent_stride * sizeof(float));
	}

	int	   n_tokens		  = ctx->n_rows > 0 ? ctx->n_rows : 1;
	int	   scratch_stride = 2 * p->state_size + 2 * p->value_head_dim;
	float *conv			  = ws;
	float *scratch		  = ws + (size_t)n_tokens * p->conv_dim;

	gdn_conv_tokens(conv, conv_state, mixed, conv_w, p->conv_dim, p->conv_kernel, n_tokens,
					p->conv_dim);

	gdn_job job = {
		.state			= state,
		.conv			= conv,
		.z				= z,
		.alpha			= alpha,
		.beta			= beta,
		.out			= out,
		.dt				= dt,
		.a				= a,
		.norm_w			= norm_w,
		.scratch		= scratch,
		.n_tokens		= n_tokens,
		.conv_stride	= p->conv_dim,
		.z_stride		= p->value_dim,
		.alpha_stride	= p->n_value_heads,
		.beta_stride	= p->n_value_heads,
		.out_stride		= p->value_dim,
		.nkh			= p->n_key_heads,
		.kd				= p->state_size,
		.vd				= p->value_head_dim,
		.key_dim		= p->key_dim,
		.scratch_stride = scratch_stride,
		.eps			= m->norm_eps,
	};
	if (pool && p->n_value_heads > 1)
		tpool_parallel_for(pool, p->n_value_heads, 1, gdn_chunk, &job);
	else
		gdn_heads(0, p->n_value_heads, &job, scratch);
	return OK;
}

status_code op_gated_delta_net(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->cache || !ctx->s || !ctx->cache->hybrid)
		return ERR_INVALID_ARG;
	profile_scope			   ps		= profile_begin(&ctx->s->prof, ctx->op->stage);
	const model_hybrid_params *p		= &ctx->m->hybrid;
	int						   n_tokens = ctx->n_rows > 0 ? ctx->n_rows : 1;

	tpool *pool = NULL;
	if (ctx->m->backend && ctx->m->backend->get_pool)
		pool = ctx->m->backend->get_pool(ctx->m->backend);
	int n_threads = pool ? tpool_n_threads(pool) : 1;
	if (n_threads < 1)
		n_threads = 1;

	size_t conv_need	= (size_t)n_tokens * p->conv_dim;
	size_t scratch_need = (size_t)n_threads * (2 * p->state_size + 2 * p->value_head_dim);

	const float *mixed = recipe_slot_f32(ctx, RECIPE_SLOT_HYB_PROJ);
	const float *z	   = recipe_slot_f32(ctx, RECIPE_SLOT_HYB_GATE);
	const float *alpha = recipe_slot_f32(ctx, RECIPE_SLOT_HYB_ALPHA);
	const float *beta  = recipe_slot_f32(ctx, RECIPE_SLOT_HYB_BETA);
	float		*out   = recipe_slot_f32(ctx, ctx->op->out);
	if (!mixed || !z || !alpha || !beta || !out)
		return ERR_INVALID_ARG;

	float	   *ws = float_buf_ensure(&ctx->s->hybrid_host, conv_need + scratch_need);
	status_code st = gdn_run(ctx, mixed, z, alpha, beta, out, ws, pool);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static recipe_op qwen_matmul(uint8_t in, uint8_t out, uint8_t weight, int n, int k) {
	return mk_matmul(in, out, weight, n, k, STAGE_MATMUL);
}

static int qwen_append_ffn(recipe_op *ops, int i, const model *m) {
	int	  dim			   = m->dim;
	int	  inter			   = m->intermediate;
	float eps			   = m->norm_eps;
	int	  has_matmul_multi = backend_has_cap(m->backend, BCAP_MULTI_MATMUL);
	ops[i++]			   = mk_add(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++]			   = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD);
	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_POST_ATTN_NORM, eps, STAGE_RMSNORM);
	if (m->layers[0].gate_up_fused) {
		ops[i++] =
			qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP, 2 * inter, dim);
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE_FUSED,
			.in		   = {RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	} else if (has_matmul_multi) {
		ops[i++] = (recipe_op){
			.kind			= OP_MATMUL_MULTI,
			.in				= {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out			= RECIPE_SLOT_FFN_GATE,
			.w_idx			= WIDX_GATE,
			.stage			= STAGE_MATMUL,
			.u.matmul_multi = {.n = 2, .k = dim, .n_out = {inter, inter}},
		};
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE,
			.in		   = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	} else {
		ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter, dim);
		ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim);
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE,
			.in		   = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	}
	ops[i++] = qwen_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim, inter);
	ops[i++] = mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD);
	return i;
}

static int qwen_append_recurrent(recipe_op *ops, const model *m) {
	int						   i   = 0;
	int						   dim = m->dim;
	const model_hybrid_params *p   = &m->hybrid;
	ops[i++] =
		mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_ATTN_QKV, p->conv_dim, dim);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_GATE, WIDX_ATTN_GATE, p->value_dim, dim);
	ops[i++] =
		qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_ALPHA, WIDX_SSM_ALPHA, p->n_value_heads, dim);
	ops[i++] =
		qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_BETA, WIDX_SSM_BETA, p->n_value_heads, dim);
	ops[i++] = (recipe_op){
		.kind  = OP_GATED_DELTA_NET,
		.in	   = {RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_HYB_GATE, RECIPE_SLOT_HYB_ALPHA},
		.out   = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
	};
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_SSM_OUT, dim, p->value_dim);
	return qwen_append_ffn(ops, i, m);
}

static int qwen_append_full_attention(recipe_op *ops, const model *m) {
	int i	   = 0;
	int dim	   = m->dim;
	int q_out  = m->n_heads * m->head_dim;
	int kv_out = m->n_kv_heads * m->head_dim;
	ops[i++] =
		mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_WQ, 2 * q_out, dim);
	ops[i++] = (recipe_op){
		.kind  = OP_SPLIT_QGATE,
		.in	   = {RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_Q,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_MATMUL,
	};
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim);
	ops[i++] = (recipe_op){
		.kind	   = OP_RMSNORM_PER_HEAD,
		.in		   = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	   = RECIPE_SLOT_Q,
		.w_idx	   = WIDX_ATTN_Q_NORM,
		.stage	   = STAGE_RMSNORM,
		.u.rmsnorm = {.eps = m->norm_eps, .n_heads = m->n_heads},
	};
	ops[i++] = (recipe_op){
		.kind	   = OP_RMSNORM_PER_HEAD,
		.in		   = {RECIPE_SLOT_K, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	   = RECIPE_SLOT_K,
		.w_idx	   = WIDX_ATTN_K_NORM,
		.stage	   = STAGE_RMSNORM,
		.u.rmsnorm = {.eps = m->norm_eps, .n_heads = m->n_kv_heads},
	};
	ops[i++] = (recipe_op){
		.kind  = OP_PARTIAL_ROPE_QK,
		.in	   = {RECIPE_SLOT_Q, RECIPE_SLOT_K, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ROPE,
	};
	ops[i++] = (recipe_op){
		.kind  = OP_KV_PUT,
		.in	   = {RECIPE_SLOT_K, RECIPE_SLOT_V, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_KVPUT,
	};
	ops[i++] = (recipe_op){
		.kind  = OP_ATTENTION,
		.in	   = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
		.u.attention =
			{
				.n_heads		   = m->n_heads,
				.n_kv_heads		   = m->n_kv_heads,
				.head_dim		   = m->head_dim,
				.n_ctx			   = m->n_ctx,
				.scale			   = 1.0f / sqrtf((float)m->head_dim),
				.n_kv_heads_active = m->n_kv_heads,
			},
	};
	ops[i++] = (recipe_op){
		.kind  = OP_ATTN_OUTPUT_GATE,
		.in	   = {RECIPE_SLOT_XB2, RECIPE_SLOT_HYB_GATE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
	};
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out);
	return qwen_append_ffn(ops, i, m);
}

enum { QWEN35_MAX_OPS_PER_LAYER = 24 };

static model_recipe *build_qwen35_recipe(const model *m) {
	model_recipe *r		= xcalloc(1, sizeof(*r));
	r->max_intermediate = m->intermediate;
	r->max_head_dim		= m->head_dim;
	r->max_kv_heads		= m->n_kv_heads;
	{
		r->pre_ops	  = xcalloc(1, sizeof(recipe_op));
		r->pre_ops[0] = (recipe_op){
			.kind  = OP_EMBD_LOOKUP,
			.in	   = {RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_X,
			.w_idx = WIDX_TOK_EMBD,
			.stage = STAGE_EMBD,
		};
		r->n_pre_ops = 1;
	}

	r->layer.ops	 = xcalloc(QWEN35_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	r->layer.n_ops	 = QWEN35_MAX_OPS_PER_LAYER;
	r->per_layer_ops = xcalloc((size_t)m->n_layers * QWEN35_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	for (int li = 0; li < m->n_layers; li++) {
		recipe_op *ops = r->per_layer_ops + (size_t)li * QWEN35_MAX_OPS_PER_LAYER;
		int		   n   = model_layer_is_recurrent(m, li) ? qwen_append_recurrent(ops, m)
														 : qwen_append_full_attention(ops, m);
		if (n > QWEN35_MAX_OPS_PER_LAYER) {
			ERROR("qwen35: layer %d produced %d recipe ops but capacity is %d -- "
				  "raise QWEN35_MAX_OPS_PER_LAYER",
				  li, n, QWEN35_MAX_OPS_PER_LAYER);
			free(r->pre_ops);
			free(r->layer.ops);
			free(r->per_layer_ops);
			free(r);
			return NULL;
		}
	}
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(qwen35, "qwen35", build_qwen35_recipe)
