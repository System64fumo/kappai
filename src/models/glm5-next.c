#include "backend/backend.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "model.h"
#include "moe/moe_stream.h"
#include "recipe.h"
#include "threadpool.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum {
	GLM5_NEXT_MAX_OPS_PER_LAYER = 24,
	GLM5_NEXT_HC_MIX			= 24,
};

static const float GLM5_NEXT_L2_EPS = 1e-6f;

static float sigmoidf(float x) {
	return 1.0f / (1.0f + expf(-x));
}

static float siluf(float x) {
	return x * sigmoidf(x);
}

static float *workspace(exec_ctx *ctx, size_t n_floats) {
	return float_buf_ensure_nocopy(&ctx->s->glm_ws, n_floats, 64);
}

static status_code host_matmul(backend *a, const weight_ref *w, const float *x, float *y, int n,
							   int k) {
	buffer		xb = buffer_host_view(a, (void *)x, (size_t)k * sizeof(float));
	buffer		yb = buffer_host_view(a, y, (size_t)n * sizeof(float));
	status_code st = a->matmul(a, &w->buf, w->type, &xb, &yb, n, k);
	ensure_sync(a);
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

status_code glm5_next_op_hc_init(exec_ctx *ctx) {
	const int dim = ctx->m->dim;
	float	 *x	  = recipe_exec_slot_host(ctx, ctx->op->in[0]);
	float	 *hc  = recipe_exec_slot_host(ctx, ctx->op->out);
	if (!x || !hc)
		return ERR_UNSUPPORTED;
	for (int h = 0; h < ctx->m->kda.hc_count; h++)
		memcpy(hc + (size_t)h * dim, x, (size_t)dim * sizeof(float));
	return OK;
}

status_code glm5_next_op_hc_final(exec_ctx *ctx) {
	const int dim = ctx->m->dim;
	const int n	  = ctx->m->kda.hc_count;
	float	 *hc  = recipe_exec_slot_host(ctx, ctx->op->in[0]);
	float	 *x	  = recipe_exec_slot_host(ctx, ctx->op->out);
	if (!x || !hc)
		return ERR_UNSUPPORTED;
	for (int i = 0; i < dim; i++) {
		float acc = 0.0f;
		for (int h = 0; h < n; h++)
			acc += hc[(size_t)h * dim + i];
		x[i] = acc / (float)n;
	}
	return OK;
}

static void hc_sinkhorn(float comb[16], int iters, float eps) {
	enum { HC = 4 };
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

status_code glm5_next_op_hc_pre(exec_ctx *ctx) {
	model				*m	  = ctx->m;
	const layer_weights *L	  = &m->layers[ctx->li];
	const int			 attn = ctx->op->w_idx == WIDX_HC_ATTN_FN;
	const weight_ref	*fn	  = attn ? &L->hc_attn_fn_w : &L->hc_ffn_fn_w;
	const float			*base =
		(const float *)(attn ? L->hc_attn_base_w.host_ptr : L->hc_ffn_base_w.host_ptr);
	const float *scale =
		(const float *)(attn ? L->hc_attn_scale_w.host_ptr : L->hc_ffn_scale_w.host_ptr);
	const int hc	 = m->kda.hc_count;
	const int dim	 = m->dim;
	const int flat	 = hc * dim;
	float	 *stream = recipe_exec_slot_host(ctx, ctx->op->in[0]);
	float	 *x		 = recipe_exec_slot_host(ctx, ctx->op->out);
	float	 *coef	 = recipe_exec_slot_host(ctx, RECIPE_SLOT_HC_COEF);
	backend	 *a		 = model_layer_backend(ctx->m, ctx->li);
	if (!stream || !x || !coef || !base || !scale || !a)
		return ERR_UNSUPPORTED;

	float *normed = workspace(ctx, (size_t)flat + GLM5_NEXT_HC_MIX);
	float *mixes  = normed + flat;
	memcpy(normed, stream, (size_t)flat * sizeof(float));
	rmsnorm_inplace(normed, NULL, flat, m->norm_eps);

	status_code st = host_matmul(a, fn, normed, mixes, GLM5_NEXT_HC_MIX, flat);
	if (st != OK)
		return st;

	float pre[4];
	for (int h = 0; h < hc; h++) {
		pre[h]	= sigmoidf(mixes[h] * scale[0] + base[h]) + m->kda.hc_eps;
		coef[h] = 2.0f * sigmoidf(mixes[hc + h] * scale[1] + base[hc + h]);
	}
	float *comb = coef + hc;
	for (int j = 0; j < hc * hc; j++)
		comb[j] = mixes[2 * hc + j] * scale[2] + base[2 * hc + j];
	hc_sinkhorn(comb, m->kda.hc_sinkhorn_iters, m->kda.hc_eps);

	for (int i = 0; i < dim; i++) {
		float acc = 0.0f;
		for (int h = 0; h < hc; h++)
			acc += pre[h] * stream[(size_t)h * dim + i];
		x[i] = acc;
	}
	return OK;
}

status_code glm5_next_op_hc_post(exec_ctx *ctx) {
	const model *m	  = ctx->m;
	const int	 hc	  = m->kda.hc_count;
	const int	 dim  = m->dim;
	const float *sub  = recipe_exec_slot_host(ctx, ctx->op->in[0]);
	float		*hs	  = recipe_exec_slot_host(ctx, ctx->op->out);
	const float *coef = recipe_exec_slot_host(ctx, RECIPE_SLOT_HC_COEF);
	if (!sub || !hs || !coef)
		return ERR_UNSUPPORTED;
	const float *post = coef;
	const float *comb = coef + hc;

	float *next = workspace(ctx, (size_t)hc * dim);
	for (int d = 0; d < hc; d++) {
		float *dst = next + (size_t)d * dim;
		for (int i = 0; i < dim; i++) {
			float acc = sub[i] * post[d];
			for (int s = 0; s < hc; s++)
				acc += hs[(size_t)s * dim + i] * comb[d + s * hc];
			dst[i] = acc;
		}
	}
	memcpy(hs, next, (size_t)hc * dim * sizeof(float));
	return OK;
}

static void l2_normalize_heads(float *x, int n_heads, int head_dim) {
	for (int h = 0; h < n_heads; h++) {
		float *v  = x + (size_t)h * head_dim;
		float  ss = 0.0f;
		for (int i = 0; i < head_dim; i++)
			ss += v[i] * v[i];
		float inv = 1.0f / sqrtf(ss + GLM5_NEXT_L2_EPS);
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
		x[c]		= siluf(sum);
	}
}

typedef struct {
	float		*state;
	const float *q, *k, *v, *g, *beta;
	float		*out;
	int			 head_dim;
} kda_job;

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

status_code glm5_next_op_kda(exec_ctx *ctx) {
	model				*m	 = ctx->m;
	const layer_weights *L	 = &m->layers[ctx->li];
	kvcache_hybrid		*c	 = ctx->cache->hybrid;
	backend				*a	 = model_layer_backend(ctx->m, ctx->li);
	const int			 dim = m->dim, H = m->n_heads, S = m->kda.head_dim, D = H * S;
	const int			 K = m->hybrid.conv_kernel, hist = K - 1;
	const float			*x	 = recipe_exec_slot_host(ctx, ctx->op->in[0]);
	float				*out = recipe_exec_slot_host(ctx, ctx->op->out);
	if (!x || !out || !c || !a)
		return ERR_UNSUPPORTED;

	float *conv_state = c->conv_state + (size_t)ctx->li * c->conv_stride;
	float *state	  = c->recurrent_state + (size_t)ctx->li * c->recurrent_stride;
	if (ctx->pos_start == 0) {
		memset(conv_state, 0, c->conv_stride * sizeof(float));
		memset(state, 0, c->recurrent_stride * sizeof(float));
	}

	float *q	= workspace(ctx, (size_t)6 * D + 2 * S + 2 * H);
	float *k	= q + D;
	float *v	= k + D;
	float *g	= v + D;
	float *o	= g + D;
	float *g2	= o + D;
	float *tmp	= g2 + D;
	float *beta = tmp + S;
	float *tmp2 = beta + H;

	status_code st = host_matmul(a, &L->wq, x, q, D, dim);
	if (st == OK)
		st = host_matmul(a, &L->wk, x, k, D, dim);
	if (st == OK)
		st = host_matmul(a, &L->wv, x, v, D, dim);
	if (st != OK)
		return st;
	causal_conv_silu(conv_state, (const float *)L->kda_conv_q_w.host_ptr, q, D, K);
	causal_conv_silu(conv_state + (size_t)D * hist, (const float *)L->kda_conv_k_w.host_ptr, k, D,
					 K);
	causal_conv_silu(conv_state + (size_t)2 * D * hist, (const float *)L->kda_conv_v_w.host_ptr, v,
					 D, K);
	l2_normalize_heads(q, H, S);
	l2_normalize_heads(k, H, S);

	st = host_matmul(a, &L->kda_f_a_w, x, tmp, S, dim);
	if (st == OK)
		st = host_matmul(a, &L->kda_f_b_w, tmp, g, D, S);
	if (st == OK)
		st = host_matmul(a, &L->ssm_beta_w, x, beta, H, dim);
	if (st != OK)
		return st;
	const float *dt	  = (const float *)L->ssm_dt_b.host_ptr;
	const float *ssma = (const float *)L->ssm_a.host_ptr;
	const float	 lb	  = m->kda.gate_lower_bound;
	for (int h = 0; h < H; h++) {
		for (int i = 0; i < S; i++) {
			int	  idx = h * S + i;
			float t	  = (g[idx] + dt[idx]);
			if (lb > -INFINITY) {
				g[idx] = lb * sigmoidf(-(t * ssma[h]));
			} else {
				float sp = t > 20.0f ? t : log1pf(expf(t));
				g[idx]	 = sp * ssma[h];
			}
		}
		beta[h] = sigmoidf(beta[h]);
	}

	kda_job job	 = {state, q, k, v, g, beta, o, S};
	tpool  *pool = model_get_pool(m);
	if (pool && tpool_current_tid() < 0)
		tpool_parallel_for(pool, H, 1, kda_heads, &job);
	else
		kda_heads(0, H, -1, &job);

	st = host_matmul(a, &L->kda_g_a_w, x, tmp, S, dim);
	if (st == OK)
		st = host_matmul(a, &L->kda_g_b_w, tmp, g2, D, S);
	if (st != OK)
		return st;
	const float *onorm = (const float *)L->ssm_norm_w.host_ptr;
	for (int h = 0; h < H; h++) {
		float *oh = o + (size_t)h * S;
		rmsnorm_inplace(oh, onorm, S, m->norm_eps);
		for (int i = 0; i < S; i++)
			oh[i] *= sigmoidf(g2[h * S + i]);
	}
	(void)tmp2;
	return host_matmul(a, &L->wo, o, out, dim, D);
}

typedef struct {
	const float *q_abs;
	const float *cache;
	const float *v_b;
	float		*out;
	float		*scratch;
	int			 n_pos, kv_lora, v_head, scratch_stride;
	float		 scale;
} mla_job;

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

status_code glm5_next_op_mla(exec_ctx *ctx) {
	model				*m	 = ctx->m;
	const layer_weights *L	 = &m->layers[ctx->li];
	kvcache_hybrid		*c	 = ctx->cache->hybrid;
	backend				*a	 = model_layer_backend(ctx->m, ctx->li);
	const int			 dim = m->dim, H = m->n_heads, R = m->mla.kv_lora;
	const int			 qk = m->mla.qk_head, vh = m->mla.v_head, ql = m->mla.q_lora;
	const float			*x	 = recipe_exec_slot_host(ctx, ctx->op->in[0]);
	float				*out = recipe_exec_slot_host(ctx, ctx->op->out);
	if (!x || !out || !c || !a || L->k_b_w.type != GGML_TYPE_F32 || L->v_b_w.type != GGML_TYPE_F32)
		return ERR_UNSUPPORTED;

	const int n_pos = ctx->pos + 1;
	float	 *cache = c->mla_latent + (size_t)ctx->li * c->mla_latent_stride;
	tpool	 *pool	= model_get_pool(m);
	int		  n_thr = pool ? tpool_n_threads(pool) : 1;
	if (n_thr < 1)
		n_thr = 1;
	const int scratch_stride = n_pos + R;

	float *qa	 = workspace(ctx, (size_t)ql + (size_t)H * qk + (size_t)H * R + R + (size_t)H * vh +
									  (size_t)(n_thr + 1) * scratch_stride);
	float *q	 = qa + ql;
	float *q_abs = q + (size_t)H * qk;
	float *kv	 = q_abs + (size_t)H * R;
	float *ho	 = kv + R;
	float *scr	 = ho + (size_t)H * vh;

	status_code st = host_matmul(a, &L->q_a_w, x, qa, ql, dim);
	if (st != OK)
		return st;
	rmsnorm_inplace(qa, (const float *)L->q_a_norm_w.host_ptr, ql, m->norm_eps);
	st = host_matmul(a, &L->q_b_w, qa, q, H * qk, ql);
	if (st == OK)
		st = host_matmul(a, &L->kv_a_w, x, kv, R, dim);
	if (st != OK)
		return st;
	rmsnorm_inplace(kv, (const float *)L->kv_a_norm_w.host_ptr, R, m->norm_eps);
	memcpy(cache + (size_t)ctx->pos * R, kv, (size_t)R * sizeof(float));

	const float *kb = (const float *)L->k_b_w.host_ptr;
	for (int h = 0; h < H; h++) {
		const float *qh = q + (size_t)h * qk;
		const float *kh = kb + (size_t)h * R * qk;
		for (int r = 0; r < R; r++) {
			const float *row = kh + (size_t)r * qk;
			float		 s	 = 0.0f;
			for (int d = 0; d < qk; d++)
				s += row[d] * qh[d];
			q_abs[(size_t)h * R + r] = s;
		}
	}

	mla_job job = {q_abs, cache,		  (const float *)L->v_b_w.host_ptr, ho, scr, n_pos, R,
				   vh,	  scratch_stride, 1.0f / sqrtf((float)qk)};
	if (pool && tpool_current_tid() < 0)
		tpool_parallel_for(pool, H, 1, mla_heads, &job);
	else
		mla_heads(0, H, -1, &job);

	return host_matmul(a, &L->wo, ho, out, dim, H * vh);
}

static void append_dense_ffn(op_emitter *e, const model *m, int li) {
	const int dim = m->dim, inter = m->intermediate;
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, m->norm_eps, STAGE_RMSNORM));
	if (m->layers[li].gate_up_fused) {
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP, 2 * inter, dim,
							 STAGE_MATMUL));
		OP_EMIT(e, mk_ffn_activate_fused(RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_FFN_ACT, inter,
										 ACTIVATION_SILU));
	} else {
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter, dim,
							 STAGE_MATMUL));
		OP_EMIT(e,
				mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim, STAGE_MATMUL));
		OP_EMIT(e, mk_ffn_activate(RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_FFN_ACT,
								   inter, ACTIVATION_SILU));
	}
	OP_EMIT(e,
			mk_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim, inter, STAGE_MATMUL));
}

static void append_moe_ffn(op_emitter *e, const model *m) {
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, m->norm_eps, STAGE_RMSNORM));
	e->count =
		recipe_append_moe_ffn(e->ops, e->count, m, RECIPE_SLOT_XB, RECIPE_SLOT_XB, RECIPE_SLOT_XB2);
	OP_EMIT(e, mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_FFN_ACT, STAGE_ADD));
}

static void append_layer(op_emitter *e, const model *m, int li) {
	OP_EMIT(e,
			mk_host_op(OP_GLM_HC_PRE, RECIPE_SLOT_HC, RECIPE_SLOT_X, WIDX_HC_ATTN_FN, STAGE_ATTN));
	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM));
	OP_EMIT(e, mk_host_op(model_layer_is_recurrent(m, li) ? OP_GLM_KDA : OP_GLM_MLA, RECIPE_SLOT_XB,
						  RECIPE_SLOT_ATTN_OUT, WIDX_NONE, STAGE_ATTN));
	OP_EMIT(e,
			mk_host_op(OP_GLM_HC_POST, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_HC, WIDX_NONE, STAGE_ADD));
	OP_EMIT(e,
			mk_host_op(OP_GLM_HC_PRE, RECIPE_SLOT_HC, RECIPE_SLOT_X, WIDX_HC_FFN_FN, STAGE_ATTN));
	if (model_layer_is_moe(m, li))
		append_moe_ffn(e, m);
	else
		append_dense_ffn(e, m, li);
	OP_EMIT(e, mk_host_op(OP_GLM_HC_POST, RECIPE_SLOT_XB2, RECIPE_SLOT_HC, WIDX_NONE, STAGE_ADD));
}

static model_recipe *build_glm5_next_recipe(const model *m) {
	if (!backend_has_cap(m->backend, BCAP_IS_HOST) || m->mixed_backend_mode) {
		ERROR("glm5-next: only the CPU backend is supported (host-side mHC/KDA/MLA ops)");
		return NULL;
	}
	for (int li = 0; li < m->n_layers; li++) {
		if (!model_layer_is_recurrent(m, li) && (m->layers[li].k_b_w.type != GGML_TYPE_F32 ||
												 m->layers[li].v_b_w.type != GGML_TYPE_F32)) {
			ERROR("glm5-next: layer %d MLA k_b/v_b could not be dequantized to f32", li);
			return NULL;
		}
	}

	model_recipe *r = xcalloc(1, sizeof(*r));
	recipe_build_pre_ops(r, m);
	r->pre_ops = xrealloc(r->pre_ops, (size_t)(r->n_pre_ops + 1) * sizeof(recipe_op));
	r->pre_ops[r->n_pre_ops++] =
		mk_host_op(OP_GLM_HC_INIT, RECIPE_SLOT_X, RECIPE_SLOT_HC, WIDX_NONE, STAGE_ADD);

	r->layer.ops   = xcalloc(GLM5_NEXT_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	r->layer.n_ops = GLM5_NEXT_MAX_OPS_PER_LAYER;
	r->per_layer_ops =
		xcalloc((size_t)m->n_layers * GLM5_NEXT_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	for (int li = 0; li < m->n_layers; li++) {
		recipe_op *ops = r->per_layer_ops + (size_t)li * GLM5_NEXT_MAX_OPS_PER_LAYER;
		op_emitter e   = op_emitter_make(ops, GLM5_NEXT_MAX_OPS_PER_LAYER, "glm5-next");
		append_layer(&e, m, li);
	}

	recipe_build_post_ops(r, m);
	r->post_ops = xrealloc(r->post_ops, (size_t)(r->n_post_ops + 1) * sizeof(recipe_op));
	memmove(r->post_ops + 1, r->post_ops, (size_t)r->n_post_ops * sizeof(recipe_op));
	r->post_ops[0] =
		mk_host_op(OP_GLM_HC_FINAL, RECIPE_SLOT_HC, RECIPE_SLOT_X, WIDX_NONE, STAGE_ADD);
	r->n_post_ops++;

	moe_stream_cache_init((struct model *)m);
	return r;
}

RECIPE_REGISTER(glm5_next, "glm5-next", build_glm5_next_recipe)
