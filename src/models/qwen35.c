#include "backend/backend.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "model.h"
#include "recipe.h"
#include "threadpool.h"

#include <math.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

static float *qwen_host_f32(buffer *b) {
	if (!b || !b->host_ptr)
		return NULL;
	return (float *)((char *)b->host_ptr + b->offset);
}

static status_code qwen_read(const buffer *b, float *dst, int n) {
	if (!b || !b->owner || !b->owner->buffer_read_f32)
		return ERR_UNSUPPORTED;
	if (b->owner->synchronize)
		b->owner->synchronize(b->owner);
	return b->owner->buffer_read_f32(b->owner, b, dst, n);
}

static status_code qwen_write(buffer *b, const float *src, int n) {
	if (!b || !b->owner || !b->owner->buffer_write_f32)
		return ERR_UNSUPPORTED;
	return b->owner->buffer_write_f32(b->owner, b, src, n);
}

static float qwen_sigmoid(float x) {
	if (x >= 0.0f) {
		float z = expf(-x);
		return 1.0f / (1.0f + z);
	}
	float z = expf(x);
	return z / (1.0f + z);
}

static float qwen_silu(float x) {
	return x * qwen_sigmoid(x);
}

static float qwen_softplus(float x) {
	if (x > 20.0f)
		return x;
	if (x < -20.0f)
		return expf(x);
	return log1pf(expf(x));
}

static void qwen_split_qgate_rows(const float *mixed, float *q, float *gate, int n_heads,
								  int head_dim, int n_rows) {
	int q_out = n_heads * head_dim;
	int mixed_stride = 2 * q_out;
	for (int row = 0; row < n_rows; row++) {
		const float *src_row = mixed + (size_t)row * mixed_stride;
		float *q_row = q + (size_t)row * q_out;
		float *g_row = gate + (size_t)row * q_out;
		for (int h = 0; h < n_heads; h++) {
			const float *src = src_row + (size_t)h * 2 * head_dim;
			memcpy(q_row + (size_t)h * head_dim, src, (size_t)head_dim * sizeof(float));
			memcpy(g_row + (size_t)h * head_dim, src + head_dim,
				   (size_t)head_dim * sizeof(float));
		}
	}
}

status_code op_qwen_split_qgate(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	model *m = ctx->m;
	int head_dim = m->head_dim;
	int n_heads = m->n_heads;
	int q_out = n_heads * head_dim;
	if (recipe_exec_is_batch(ctx)) {
		const float *mixed = recipe_slot_f32(ctx, RECIPE_SLOT_QWEN_PROJ);
		float *q = recipe_slot_f32(ctx, RECIPE_SLOT_Q);
		float *gate = recipe_slot_f32(ctx, RECIPE_SLOT_QWEN_GATE);
		if (!mixed || !q || !gate)
			return ERR_INVALID_ARG;
		qwen_split_qgate_rows(mixed, q, gate, n_heads, head_dim, ctx->n_rows);
		return OK;
	}
	buffer *slots = compute_slots_array(ctx->s);
	float *mixed_h = qwen_host_f32(&slots[RECIPE_SLOT_QWEN_PROJ]);
	float *q_h = qwen_host_f32(&slots[RECIPE_SLOT_Q]);
	float *gate_h = qwen_host_f32(&slots[RECIPE_SLOT_QWEN_GATE]);
	if (mixed_h && q_h && gate_h) {
		qwen_split_qgate_rows(mixed_h, q_h, gate_h, n_heads, head_dim, 1);
		return OK;
	}
	float *tmp = float_buf_ensure(&ctx->s->qwen_host, (size_t)4 * q_out);
	float *mixed = tmp;
	float *q = mixed + 2 * q_out;
	float *gate = q + q_out;
	status_code st = qwen_read(&slots[RECIPE_SLOT_QWEN_PROJ], mixed, 2 * q_out);
	if (st != OK)
		return st;
	qwen_split_qgate_rows(mixed, q, gate, n_heads, head_dim, 1);
	st = qwen_write(&slots[RECIPE_SLOT_Q], q, q_out);
	if (st == OK)
		st = qwen_write(&slots[RECIPE_SLOT_QWEN_GATE], gate, q_out);
	return st;
}

static void qwen_partial_rope_one(float *x, int n_heads, int head_dim, int rope_dim,
								  const float *cosv, const float *sinv) {
	int half = rope_dim / 2;
	for (int h = 0; h < n_heads; h++) {
		float *v = x + (size_t)h * head_dim;
		for (int j = 0; j < half; j++) {
			float a = v[j];
			float b = v[j + half];
			v[j] = a * cosv[j] - b * sinv[j];
			v[j + half] = a * sinv[j] + b * cosv[j];
		}
	}
}

status_code op_qwen_partial_rope_qk(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s || ctx->pos < 0)
		return ERR_INVALID_ARG;
	model *m = ctx->m;
	int qn = m->n_heads * m->head_dim;
	int kn = m->n_kv_heads * m->head_dim;
	int half = m->rope_dim / 2;
	if (recipe_exec_is_batch(ctx)) {
		float *q = recipe_slot_f32(ctx, RECIPE_SLOT_Q);
		float *k = recipe_slot_f32(ctx, RECIPE_SLOT_K);
		if (!q || !k)
			return ERR_INVALID_ARG;
		for (int row = 0; row < ctx->n_rows; row++) {
			int pos = ctx->pos_start + row;
			const float *cosv = ctx->s->rope_cos + (size_t)pos * half;
			const float *sinv = ctx->s->rope_sin + (size_t)pos * half;
			qwen_partial_rope_one(q + (size_t)row * qn, m->n_heads, m->head_dim, m->rope_dim,
								  cosv, sinv);
			qwen_partial_rope_one(k + (size_t)row * kn, m->n_kv_heads, m->head_dim, m->rope_dim,
								  cosv, sinv);
		}
		return OK;
	}
	buffer *slots = compute_slots_array(ctx->s);
	float *q_h = qwen_host_f32(&slots[RECIPE_SLOT_Q]);
	float *k_h = qwen_host_f32(&slots[RECIPE_SLOT_K]);
	if (q_h && k_h) {
		const float *cosv = ctx->s->rope_cos + (size_t)ctx->pos * half;
		const float *sinv = ctx->s->rope_sin + (size_t)ctx->pos * half;
		qwen_partial_rope_one(q_h, m->n_heads, m->head_dim, m->rope_dim, cosv, sinv);
		qwen_partial_rope_one(k_h, m->n_kv_heads, m->head_dim, m->rope_dim, cosv, sinv);
		return OK;
	}
	float *tmp = float_buf_ensure(&ctx->s->qwen_host, (size_t)qn + kn);
	float *q = tmp;
	float *k = q + qn;
	status_code st = qwen_read(&slots[RECIPE_SLOT_Q], q, qn);
	if (st != OK)
		return st;
	st = qwen_read(&slots[RECIPE_SLOT_K], k, kn);
	if (st != OK)
		return st;
	const float *cosv = ctx->s->rope_cos + (size_t)ctx->pos * half;
	const float *sinv = ctx->s->rope_sin + (size_t)ctx->pos * half;
	qwen_partial_rope_one(q, m->n_heads, m->head_dim, m->rope_dim, cosv, sinv);
	qwen_partial_rope_one(k, m->n_kv_heads, m->head_dim, m->rope_dim, cosv, sinv);
	st = qwen_write(&slots[RECIPE_SLOT_Q], q, qn);
	if (st == OK)
		st = qwen_write(&slots[RECIPE_SLOT_K], k, kn);
	return st;
}

status_code op_qwen_attn_gate(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	int n = ctx->m->n_heads * ctx->m->head_dim;
	if (recipe_exec_is_batch(ctx)) {
		float *out = recipe_slot_f32(ctx, ctx->op->in[0]);
		const float *gate = recipe_slot_f32(ctx, RECIPE_SLOT_QWEN_GATE);
		if (!out || !gate)
			return ERR_INVALID_ARG;
		for (int row = 0; row < ctx->n_rows; row++) {
			float *o = out + (size_t)row * n;
			const float *g = gate + (size_t)row * n;
			for (int i = 0; i < n; i++)
				o[i] *= qwen_sigmoid(g[i]);
		}
		return OK;
	}
	buffer *slots = compute_slots_array(ctx->s);
	float *out_h = qwen_host_f32(&slots[ctx->op->in[0]]);
	float *gate_h = qwen_host_f32(&slots[RECIPE_SLOT_QWEN_GATE]);
	if (out_h && gate_h) {
		for (int i = 0; i < n; i++)
			out_h[i] *= qwen_sigmoid(gate_h[i]);
		return OK;
	}
	float *tmp = float_buf_ensure(&ctx->s->qwen_host, (size_t)2 * n);
	float *out = tmp;
	float *gate = out + n;
	status_code st = qwen_read(&slots[ctx->op->in[0]], out, n);
	if (st != OK)
		return st;
	st = qwen_read(&slots[RECIPE_SLOT_QWEN_GATE], gate, n);
	if (st != OK)
		return st;
	for (int i = 0; i < n; i++)
		out[i] *= qwen_sigmoid(gate[i]);
	return qwen_write(&slots[ctx->op->in[0]], out, n);
}

static void qwen_gdn_conv_tokens(float *conv_out, float *conv_state, const float *mixed,
								 const float *conv_w, int conv_dim, int conv_kernel, int n_tokens,
								 int mixed_stride) {
	int history = conv_kernel - 1;
	if (history == 3) {
		for (int t = 0; t < n_tokens; t++) {
			const float *mix = mixed + (size_t)t * mixed_stride;
			float *out = conv_out + (size_t)t * conv_dim;
			for (int c = 0; c < conv_dim; c++) {
				const float *w = conv_w + (size_t)c * 4;
				float *hist = conv_state + (size_t)c * 3;
				float sum = hist[0] * w[0] + hist[1] * w[1] + hist[2] * w[2] + mix[c] * w[3];
				hist[0] = hist[1];
				hist[1] = hist[2];
				hist[2] = mix[c];
				out[c] = qwen_silu(sum);
			}
		}
		return;
	}
	for (int t = 0; t < n_tokens; t++) {
		const float *mix = mixed + (size_t)t * mixed_stride;
		float *out = conv_out + (size_t)t * conv_dim;
		for (int c = 0; c < conv_dim; c++) {
			const float *w = conv_w + (size_t)c * conv_kernel;
			float sum = mix[c] * w[history];
			if (history > 0) {
				float *hist = conv_state + (size_t)c * history;
				for (int j = 0; j < history; j++)
					sum += hist[j] * w[j];
				if (history > 1)
					memmove(hist, hist + 1, (size_t)(history - 1) * sizeof(float));
				hist[history - 1] = mix[c];
			}
			out[c] = qwen_silu(sum);
		}
	}
}

typedef struct {
	float *state;
	const float *conv;
	const float *z;
	const float *alpha;
	const float *beta;
	float *out;
	const float *dt;
	const float *a;
	const float *norm_w;
	float *scratch;
	int n_tokens;
	int conv_stride;
	int z_stride;
	int alpha_stride;
	int beta_stride;
	int out_stride;
	int nkh;
	int kd;
	int vd;
	int key_dim;
	int scratch_stride;
	float eps;
} qwen_gdn_job;

#if defined(__AVX2__)
static inline float qwen_dot_f32(const float *a, const float *b, int n) {
	__m256 acc = _mm256_setzero_ps();
	int i = 0;
	for (; i + 8 <= n; i += 8)
		acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
	__m128 lo = _mm256_castps256_ps128(acc);
	__m128 hi = _mm256_extractf128_ps(acc, 1);
	__m128 s = _mm_add_ps(lo, hi);
	s = _mm_add_ps(s, _mm_movehdup_ps(s));
	s = _mm_add_ss(s, _mm_movehl_ps(s, s));
	float sum = _mm_cvtss_f32(s);
	for (; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

static inline void qwen_scale_f32(float *dst, const float *src, float scale, int n) {
	__m256 vs = _mm256_set1_ps(scale);
	int i = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vs));
	for (; i < n; i++)
		dst[i] = src[i] * scale;
}

static inline void qwen_state_decay_mem(float *row, float *mem, float decay, float ks, int n) {
	__m256 vd = _mm256_set1_ps(decay);
	__m256 vk = _mm256_set1_ps(ks);
	int i = 0;
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
	int i = 0;
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
	int i = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(delta + i, _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + i),
																 _mm256_loadu_ps(mem + i)),
												  vb));
	for (; i < n; i++)
		delta[i] = (v[i] - mem[i]) * b;
}
#endif

static void qwen_gdn_heads(int vh0, int vh1, const qwen_gdn_job *j, float *scratch) {
	int kd = j->kd;
	int vd = j->vd;
	float q_scale = 1.0f / sqrtf((float)kd);
	float *k_s = scratch;
	float *q_s = k_s + kd;
	float *mem = q_s + kd;
	float *delta = mem + vd;
	int state_head = kd * vd;

	for (int vh = vh0; vh < vh1; vh++) {
		int kh = vh % j->nkh;
		float *shead = j->state + (size_t)vh * state_head;
		for (int t = 0; t < j->n_tokens; t++) {
			const float *conv_t = j->conv + (size_t)t * j->conv_stride;
			const float *q = conv_t + (size_t)kh * kd;
			const float *k = conv_t + j->key_dim + (size_t)kh * kd;
			const float *v = conv_t + 2 * j->key_dim + (size_t)vh * vd;
			const float *z_t = j->z + (size_t)t * j->z_stride + (size_t)vh * vd;
			float *y = j->out + (size_t)t * j->out_stride + (size_t)vh * vd;
			float alpha_t = j->alpha[(size_t)t * j->alpha_stride + vh];
			float beta_t = j->beta[(size_t)t * j->beta_stride + vh];

			float decay = expf(j->a[vh] * qwen_softplus(alpha_t + j->dt[vh]));
			float b = qwen_sigmoid(beta_t);
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
				y[jj] = y[jj] * inv_rms * j->norm_w[jj] * qwen_silu(z_t[jj]);
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
				float ks = k_s[d];
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
				float ks = k_s[d];
				float qs = q_s[d];
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
				y[jj] = y[jj] * inv_rms * j->norm_w[jj] * qwen_silu(z_t[jj]);
#endif
		}
	}
}

static void qwen_gdn_chunk(int begin, int end, int tid, void *ctx) {
	qwen_gdn_job *j = (qwen_gdn_job *)ctx;
	qwen_gdn_heads(begin, end, j, j->scratch + (size_t)tid * j->scratch_stride);
}

static status_code qwen_gdn_run(exec_ctx *ctx, const float *mixed, const float *z,
								const float *alpha, const float *beta, float *out, int n_tokens,
								int mixed_stride, int z_stride, int alpha_stride, int beta_stride,
								int out_stride, float *conv, float *scratch, int scratch_stride,
								tpool *pool) {
	model *m = ctx->m;
	const model_qwen35_params *p = &m->qwen35;
	layer_weights *L = &m->layers[ctx->li];
	const float *conv_w = (const float *)L->ssm_conv1d_w.host_ptr;
	const float *dt = (const float *)L->ssm_dt_b.host_ptr;
	const float *a = (const float *)L->ssm_a.host_ptr;
	const float *norm_w = (const float *)L->ssm_norm_w.host_ptr;
	if (!conv_w || !dt || !a || !norm_w)
		return ERR_FORMAT;

	kvcache_qwen35 *cache = ctx->cache->qwen35;
	float *conv_state = cache->conv_state + (size_t)ctx->li * cache->conv_stride;
	float *state = cache->recurrent_state + (size_t)ctx->li * cache->recurrent_stride;
	if (ctx->pos_start == 0) {
		memset(conv_state, 0, cache->conv_stride * sizeof(float));
		memset(state, 0, cache->recurrent_stride * sizeof(float));
	}

	qwen_gdn_conv_tokens(conv, conv_state, mixed, conv_w, p->conv_dim, p->conv_kernel, n_tokens,
						 mixed_stride);

	qwen_gdn_job job = {
		.state = state,
		.conv = conv,
		.z = z,
		.alpha = alpha,
		.beta = beta,
		.out = out,
		.dt = dt,
		.a = a,
		.norm_w = norm_w,
		.scratch = scratch,
		.n_tokens = n_tokens,
		.conv_stride = p->conv_dim,
		.z_stride = z_stride,
		.alpha_stride = alpha_stride,
		.beta_stride = beta_stride,
		.out_stride = out_stride,
		.nkh = p->n_key_heads,
		.kd = p->state_size,
		.vd = p->value_head_dim,
		.key_dim = p->key_dim,
		.scratch_stride = scratch_stride,
		.eps = m->norm_eps,
	};
	if (pool && p->n_value_heads > 1)
		tpool_parallel_for(pool, p->n_value_heads, 1, qwen_gdn_chunk, &job);
	else
		qwen_gdn_heads(0, p->n_value_heads, &job, scratch);
	return OK;
}

status_code op_qwen_gated_delta_net(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->cache || !ctx->s || !ctx->cache->qwen35)
		return ERR_INVALID_ARG;
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	const model_qwen35_params *p = &ctx->m->qwen35;
	int conv_dim = p->conv_dim;
	int value_dim = p->value_dim;
	int nvh = p->n_value_heads;
	int n_tokens = ctx->n_rows > 0 ? ctx->n_rows : 1;
	int kd = p->state_size;
	int vd = p->value_head_dim;
	int scratch_stride = 2 * kd + 2 * vd;

	tpool *pool = NULL;
	if (ctx->m->backend && ctx->m->backend->get_pool)
		pool = ctx->m->backend->get_pool(ctx->m->backend);
	int n_threads = pool ? tpool_n_threads(pool) : 1;
	if (n_threads < 1)
		n_threads = 1;

	size_t conv_need = (size_t)n_tokens * conv_dim;
	size_t scratch_need = (size_t)n_threads * scratch_stride;

	status_code st = OK;
	if (recipe_exec_is_batch(ctx)) {
		const float *mixed = recipe_slot_f32(ctx, RECIPE_SLOT_QWEN_PROJ);
		const float *z = recipe_slot_f32(ctx, RECIPE_SLOT_QWEN_GATE);
		const float *alpha = recipe_slot_f32(ctx, RECIPE_SLOT_QWEN_ALPHA);
		const float *beta = recipe_slot_f32(ctx, RECIPE_SLOT_QWEN_BETA);
		float *out = recipe_slot_f32(ctx, ctx->op->out);
		if (!mixed || !z || !alpha || !beta || !out)
			st = ERR_INVALID_ARG;
		else {
			float *ws = float_buf_ensure(&ctx->s->qwen_host, conv_need + scratch_need);
			st = qwen_gdn_run(ctx, mixed, z, alpha, beta, out, n_tokens, conv_dim, value_dim, nvh,
							  nvh, value_dim, ws, ws + conv_need, scratch_stride, pool);
		}
	} else {
		buffer *slots = compute_slots_array(ctx->s);
		float *mixed_h = qwen_host_f32(&slots[RECIPE_SLOT_QWEN_PROJ]);
		float *z_h = qwen_host_f32(&slots[RECIPE_SLOT_QWEN_GATE]);
		float *alpha_h = qwen_host_f32(&slots[RECIPE_SLOT_QWEN_ALPHA]);
		float *beta_h = qwen_host_f32(&slots[RECIPE_SLOT_QWEN_BETA]);
		float *out_h = qwen_host_f32(&slots[ctx->op->out]);
		if (mixed_h && z_h && alpha_h && beta_h && out_h) {
			float *ws = float_buf_ensure(&ctx->s->qwen_host, conv_need + scratch_need);
			st = qwen_gdn_run(ctx, mixed_h, z_h, alpha_h, beta_h, out_h, 1, conv_dim, value_dim, nvh,
							  nvh, value_dim, ws, ws + conv_need, scratch_stride, pool);
		} else {
			size_t in_need = (size_t)conv_dim + (size_t)value_dim * 2 + (size_t)nvh * 2;
			float *tmp = float_buf_ensure(&ctx->s->qwen_host, in_need + conv_need + scratch_need);
			float *mixed = tmp;
			float *z = mixed + conv_dim;
			float *out = z + value_dim;
			float *alpha = out + value_dim;
			float *beta = alpha + nvh;
			float *conv = beta + nvh;
			float *scratch = conv + conv_need;

			st = qwen_read(&slots[RECIPE_SLOT_QWEN_PROJ], mixed, conv_dim);
			if (st == OK)
				st = qwen_read(&slots[RECIPE_SLOT_QWEN_GATE], z, value_dim);
			if (st == OK)
				st = qwen_read(&slots[RECIPE_SLOT_QWEN_ALPHA], alpha, nvh);
			if (st == OK)
				st = qwen_read(&slots[RECIPE_SLOT_QWEN_BETA], beta, nvh);
			if (st == OK)
				st = qwen_gdn_run(ctx, mixed, z, alpha, beta, out, 1, conv_dim, value_dim, nvh, nvh,
								  value_dim, conv, scratch, scratch_stride, pool);
			if (st == OK)
				st = qwen_write(&slots[ctx->op->out], out, value_dim);
		}
	}
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static recipe_op qwen_matmul(uint8_t in, uint8_t out, uint8_t weight, int n, int k) {
	return mk_matmul(in, out, weight, n, k, STAGE_MATMUL);
}

static int qwen_append_ffn(recipe_op *ops, int i, const model *m, int li) {
	int dim = m->dim;
	int inter = m->intermediate;
	float eps = m->norm_eps;
	int has_matmul_multi = backend_has_cap(m->backend, BCAP_MULTI_MATMUL);
	ops[i++] = mk_add(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD);
	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_POST_ATTN_NORM, eps, STAGE_RMSNORM);
	if (m->layers[li].gate_up_fused) {
		ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP,
						  2 * inter, dim);
		ops[i++] = (recipe_op){
			.kind = OP_FFN_ACTIVATE_FUSED,
			.in = {RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out = RECIPE_SLOT_FFN_ACT,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	} else if (has_matmul_multi) {
		ops[i++] = (recipe_op){
			.kind = OP_MATMUL_MULTI,
			.in = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out = RECIPE_SLOT_FFN_GATE,
			.w_idx = WIDX_GATE,
			.stage = STAGE_MATMUL,
			.u.matmul_multi = {.n = 2, .k = dim, .n_out = {inter, inter}},
		};
		ops[i++] = (recipe_op){
			.kind = OP_FFN_ACTIVATE,
			.in = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
			.out = RECIPE_SLOT_FFN_ACT,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	} else {
		ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter, dim);
		ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim);
		ops[i++] = (recipe_op){
			.kind = OP_FFN_ACTIVATE,
			.in = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
			.out = RECIPE_SLOT_FFN_ACT,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	}
	ops[i++] = qwen_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim, inter);
	ops[i++] = mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD);
	return i;
}

static int qwen_append_recurrent(recipe_op *ops, const model *m, int li) {
	int i = 0;
	int dim = m->dim;
	const model_qwen35_params *p = &m->qwen35;
	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps,
					  STAGE_RMSNORM);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_QWEN_PROJ, WIDX_ATTN_QKV, p->conv_dim,
						  dim);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_QWEN_GATE, WIDX_ATTN_GATE,
						  p->value_dim, dim);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_QWEN_ALPHA, WIDX_SSM_ALPHA,
						  p->n_value_heads, dim);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_QWEN_BETA, WIDX_SSM_BETA,
						  p->n_value_heads, dim);
	ops[i++] = (recipe_op){
		.kind = OP_QWEN_GATED_DELTA_NET,
		.in = {RECIPE_SLOT_QWEN_PROJ, RECIPE_SLOT_QWEN_GATE, RECIPE_SLOT_QWEN_ALPHA},
		.out = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
	};
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_SSM_OUT, dim,
						  p->value_dim);
	return qwen_append_ffn(ops, i, m, li);
}

static int qwen_append_full_attention(recipe_op *ops, const model *m, int li) {
	int i = 0;
	int dim = m->dim;
	int q_out = m->n_heads * m->head_dim;
	int kv_out = m->n_kv_heads * m->head_dim;
	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps,
					  STAGE_RMSNORM);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_QWEN_PROJ, WIDX_WQ, 2 * q_out, dim);
	ops[i++] = (recipe_op){
		.kind = OP_QWEN_SPLIT_QGATE,
		.in = {RECIPE_SLOT_QWEN_PROJ, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out = RECIPE_SLOT_Q,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_MATMUL,
	};
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim);
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim);
	ops[i++] = (recipe_op){
		.kind = OP_RMSNORM_PER_HEAD,
		.in = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out = RECIPE_SLOT_Q,
		.w_idx = WIDX_ATTN_Q_NORM,
		.stage = STAGE_RMSNORM,
		.u.rmsnorm = {.eps = m->norm_eps, .n_heads = m->n_heads},
	};
	ops[i++] = (recipe_op){
		.kind = OP_RMSNORM_PER_HEAD,
		.in = {RECIPE_SLOT_K, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out = RECIPE_SLOT_K,
		.w_idx = WIDX_ATTN_K_NORM,
		.stage = STAGE_RMSNORM,
		.u.rmsnorm = {.eps = m->norm_eps, .n_heads = m->n_kv_heads},
	};
	ops[i++] = (recipe_op){
		.kind = OP_QWEN_PARTIAL_ROPE_QK,
		.in = {RECIPE_SLOT_Q, RECIPE_SLOT_K, RECIPE_SLOT_NONE},
		.out = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ROPE,
	};
	ops[i++] = (recipe_op){
		.kind = OP_KV_PUT,
		.in = {RECIPE_SLOT_K, RECIPE_SLOT_V, RECIPE_SLOT_NONE},
		.out = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_KVPUT,
	};
	ops[i++] = (recipe_op){
		.kind = OP_ATTENTION,
		.in = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
		.u.attention = {
			.n_heads = m->n_heads,
			.n_kv_heads = m->n_kv_heads,
			.head_dim = m->head_dim,
			.n_ctx = m->n_ctx,
			.scale = 1.0f / sqrtf((float)m->head_dim),
			.n_kv_heads_active = m->n_kv_heads,
		},
	};
	ops[i++] = (recipe_op){
		.kind = OP_QWEN_ATTN_GATE,
		.in = {RECIPE_SLOT_XB2, RECIPE_SLOT_QWEN_GATE, RECIPE_SLOT_NONE},
		.out = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
	};
	ops[i++] = qwen_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out);
	return qwen_append_ffn(ops, i, m, li);
}

static model_recipe *build_qwen35_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(*r));
	r->max_intermediate = m->intermediate;
	r->max_head_dim = m->head_dim;
	r->max_kv_heads = m->n_kv_heads;
	{
		r->pre_ops = xcalloc(1, sizeof(recipe_op));
		r->pre_ops[0] = (recipe_op){
			.kind = OP_EMBD_LOOKUP,
			.in = {RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out = RECIPE_SLOT_X,
			.w_idx = WIDX_TOK_EMBD,
			.stage = STAGE_EMBD,
		};
		r->n_pre_ops = 1;
	}

	const int ops_per_layer = 24;
	r->layer.ops = xcalloc((size_t)ops_per_layer, sizeof(recipe_op));
	r->layer.n_ops = ops_per_layer;
	r->per_layer_ops = xcalloc((size_t)m->n_layers * ops_per_layer, sizeof(recipe_op));
	for (int li = 0; li < m->n_layers; li++) {
		recipe_op *ops = r->per_layer_ops + (size_t)li * ops_per_layer;
		int n = model_layer_is_recurrent(m, li) ? qwen_append_recurrent(ops, m, li)
											 : qwen_append_full_attention(ops, m, li);
		if (n > ops_per_layer)
			return NULL;
	}
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(qwen35, "qwen35", build_qwen35_recipe)
