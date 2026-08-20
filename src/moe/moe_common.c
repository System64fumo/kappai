#include "moe/moe_common.h"
#include "backend/backend.h"
#include "backend/cpu/scalar/quants.h"
#include "common.h"
#include "compute.h"
#include "config.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "moe/moe_stream.h"
#include "monitor.h"
#include "threadpool.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOE_MAX_DIM_STACK 8192

typedef struct {
	model				  *m;
	int					   li;
	int					   top_k, inter, dim;
	int					   any_fused, use_gelu;
	int					   xb_q8_gate_ok;
	uint32_t			   gate_q8_type;
	const moe_expert_slot *slot_buf;
	const int			  *expert_ids;
	const float			  *weights;
	const float			  *xb_f;
	const buffer		  *xb_q8_gate;
	backend				  *a;
	float				  *all_scratch;
	size_t				   per_thread_scratch;
	float				  *all_outs;
	size_t				   per_thread_out;
	_Atomic int			   first_err;
	_Atomic int			   n_err;
} moe_par_job;

static inline void topk_heap_sift_down(float *score, int *idx, int n, int pos) {
	for (;;) {
		int l = 2 * pos + 1, r = 2 * pos + 2, smallest = pos;
		if (l < n && score[l] < score[smallest])
			smallest = l;
		if (r < n && score[r] < score[smallest])
			smallest = r;
		if (smallest == pos)
			return;
		float ts		= score[pos];
		score[pos]		= score[smallest];
		score[smallest] = ts;
		int ti			= idx[pos];
		idx[pos]		= idx[smallest];
		idx[smallest]	= ti;
		pos				= smallest;
	}
}

int moe_topk_select(const float *scores, int n_experts, int top_k, int *top_idx, float *top_score) {
	if (top_k <= 0)
		return 0;
	if (top_k > MOE_MAX_TOPK)
		top_k = MOE_MAX_TOPK;

	int heap_n = 0;
	for (int e = 0; e < n_experts && heap_n < top_k; e++) {
		top_idx[heap_n]	  = e;
		top_score[heap_n] = scores[e];
		int pos			  = heap_n++;
		while (pos > 0) {
			int parent = (pos - 1) / 2;
			if (top_score[parent] <= top_score[pos])
				break;
			float ts		  = top_score[pos];
			top_score[pos]	  = top_score[parent];
			top_score[parent] = ts;
			int ti			  = top_idx[pos];
			top_idx[pos]	  = top_idx[parent];
			top_idx[parent]	  = ti;
			pos				  = parent;
		}
	}
	for (int e = heap_n; e < n_experts; e++) {
		if (scores[e] > top_score[0]) {
			top_score[0] = scores[e];
			top_idx[0]	 = e;
			topk_heap_sift_down(top_score, top_idx, heap_n, 0);
		}
	}
	for (int k = heap_n; k < top_k; k++) {
		top_idx[k]	 = -1;
		top_score[k] = -1e30f;
	}
	return top_k;
}

void moe_apply_weights(float *weight, int n_k, int norm_topk, float routed_scale) {
	if (norm_topk) {
		float sum = 0.0f;
		for (int k = 0; k < n_k; k++)
			sum += weight[k];
		if (sum > 1e-20f) {
			float inv = routed_scale / sum;
			for (int k = 0; k < n_k; k++)
				weight[k] *= inv;
			return;
		}
	}
	for (int k = 0; k < n_k; k++)
		weight[k] *= routed_scale;
}

void moe_activate(float *act, const float *gate, const float *up, int n_i, float gs, float us,
				  int use_gelu) {
	if (use_gelu) {
		for (int i = 0; i < n_i; i++)
			act[i] = gelu_tanh(gate[i] * gs) * (up[i] * us);
	} else {
		for (int i = 0; i < n_i; i++)
			act[i] = silu(gate[i] * gs) * (up[i] * us);
	}
}

void moe_router_normalize_input(const struct model *m, const struct layer_weights *L, int dim,
								float *router_input) {
	float ss = 0.0f;
	for (int i = 0; i < dim; i++)
		ss += router_input[i] * router_input[i];
	float inv		 = 1.0f / sqrtf((ss / (float)dim) + m->norm_eps);
	float base_scale = inv * m->moe.router_dim_scale;
	if (L->router_scale_w.host_ptr) {
		const float *scale_w = (const float *)L->router_scale_w.host_ptr;
		for (int i = 0; i < dim; i++)
			router_input[i] *= base_scale * scale_w[i];
	} else {
		for (int i = 0; i < dim; i++)
			router_input[i] *= base_scale;
	}
}

int moe_router_emit(int E, int K, int use_softmax, int norm_topk, float routed_scale, float *logits,
					const float *bias, float *scores_scratch, int *ids_out, float *w_out) {
	int	  top_idx[MOE_MAX_TOPK];
	float top_score[MOE_MAX_TOPK];
	float weight[MOE_MAX_TOPK];

	if (use_softmax) {
		float max_logit = logits[0];
		for (int e = 1; e < E; e++)
			if (logits[e] > max_logit)
				max_logit = logits[e];
		float sum_exp = 0.0f;
		for (int e = 0; e < E; e++) {
			logits[e] = expf(logits[e] - max_logit);
			sum_exp += logits[e];
		}
		for (int e = 0; e < E; e++)
			logits[e] /= sum_exp;
	} else {
		float *scores = scores_scratch;
		float  scores_stack[8192];
		if (!scores)
			scores = (E <= (int)ARRAY_LEN(scores_stack)) ? scores_stack
														 : xmalloc((size_t)E * sizeof(float));
		for (int e = 0; e < E; e++)
			scores[e] = (1.0f / (1.0f + expf(-logits[e]))) + (bias ? bias[e] : 0.0f);
		K = moe_topk_select(scores, E, K, top_idx, top_score);
		if (scores != scores_scratch && scores != scores_stack)
			free(scores);
		for (int k = 0; k < K; k++)
			weight[k] = (top_idx[k] >= 0) ? top_score[k] - (bias ? bias[top_idx[k]] : 0.0f) : 0.0f;
		moe_apply_weights(weight, K, norm_topk, routed_scale);
		for (int k = 0; k < K; k++) {
			ids_out[k] = top_idx[k];
			w_out[k]   = weight[k];
		}
		return K;
	}

	K = moe_topk_select(logits, E, K, top_idx, top_score);
	for (int k = 0; k < K; k++)
		weight[k] = top_score[k];
	moe_apply_weights(weight, K, norm_topk, routed_scale);
	for (int k = 0; k < K; k++) {
		ids_out[k] = top_idx[k];
		w_out[k]   = weight[k];
	}
	return K;
}

status_code moe_expert_exec(moe_expert_ctx *cx, const moe_expert_slot *es, float weight, int tid) {
	moe_stream_wait_slot(es);
	if (es->eid < 0 || !es->gate_w)
		return OK;

	float *gate_h = cx->scratch;
	float *up_h	  = gate_h + cx->inter;
	float *act_h  = up_h + cx->inter;
	float *y_h	  = act_h + cx->inter;

	status_code st = OK;

	if (es->gate_up_fused) {
		float *gu_h = y_h + cx->dim;
		buffer bw	= {0};
		bw.handle	= (void *)es->gate_w;
		bw.host_ptr = (void *)es->gate_w;
		bw.owner	= NULL;

		int qonly_done = 0;
		if (cx->q8_gate_ok && wtype_to_q8type(es->gate_type) == cx->q8_gate_type) {
			buffer gu_buf = {0};
			gu_buf.handle = gu_h;
			gu_buf.size	  = (size_t)cx->inter * 2 * sizeof(float);
			status_code s2 =
				cx->a->matmul_qonly(cx->a, &bw, es->gate_type, cx->xb_q8_gate, cx->q8_gate_type,
									&gu_buf, cx->inter * 2, cx->dim, 1);
			if (s2 == OK) {
				qonly_done = 1;
			} else if (s2 != ERR_UNSUPPORTED && s2 != ERR_INVALID_ARG) {
				return s2;
			}
		}
		if (!qonly_done) {
			st = cx->a->matmul_thread_local(cx->a, es->gate_w, es->gate_type, cx->xb_f, gu_h,
											cx->inter * 2, cx->dim, tid);
			if (st != OK)
				return st;
		}
		moe_activate(act_h, gu_h, gu_h + cx->inter, cx->inter, es->gate_scale, es->up_scale,
					 cx->use_gelu);
	} else {
		buffer bg	= {0};
		buffer bu	= {0};
		bg.handle	= (void *)es->gate_w;
		bg.host_ptr = (void *)es->gate_w;
		bg.owner	= NULL;
		bu.handle	= (void *)es->up_w;
		bu.host_ptr = (void *)es->up_w;
		bu.owner	= NULL;

		int g_qonly = 0;
		if (cx->q8_gate_ok && wtype_to_q8type(es->gate_type) == cx->q8_gate_type) {
			buffer gbuf	   = {0};
			gbuf.handle	   = gate_h;
			gbuf.size	   = (size_t)cx->inter * sizeof(float);
			status_code s2 = cx->a->matmul_qonly(cx->a, &bg, es->gate_type, cx->xb_q8_gate,
												 cx->q8_gate_type, &gbuf, cx->inter, cx->dim, 1);
			if (s2 == OK) {
				g_qonly = 1;
			} else if (s2 != ERR_UNSUPPORTED && s2 != ERR_INVALID_ARG) {
				return s2;
			}
		}
		if (!g_qonly) {
			st = cx->a->matmul_thread_local(cx->a, es->gate_w, es->gate_type, cx->xb_f, gate_h,
											cx->inter, cx->dim, tid);
			if (st != OK)
				return st;
		}

		int u_qonly = 0;
		if (cx->q8_gate_ok && wtype_to_q8type(es->up_type) == cx->q8_gate_type) {
			buffer ubuf	   = {0};
			ubuf.handle	   = up_h;
			ubuf.size	   = (size_t)cx->inter * sizeof(float);
			status_code s2 = cx->a->matmul_qonly(cx->a, &bu, es->up_type, cx->xb_q8_gate,
												 cx->q8_gate_type, &ubuf, cx->inter, cx->dim, 1);
			if (s2 == OK) {
				u_qonly = 1;
			} else if (s2 != ERR_UNSUPPORTED && s2 != ERR_INVALID_ARG) {
				return s2;
			}
		}
		if (!u_qonly) {
			st = cx->a->matmul_thread_local(cx->a, es->up_w, es->up_type, cx->xb_f, up_h, cx->inter,
											cx->dim, tid);
			if (st != OK)
				return st;
		}
		moe_activate(act_h, gate_h, up_h, cx->inter, es->gate_scale, es->up_scale, cx->use_gelu);
	}

	st = cx->a->matmul_thread_local(cx->a, es->down_w, es->down_type, act_h, y_h, cx->dim,
									cx->inter, tid);
	if (st != OK)
		return st;

	if (es->down_scale != 1.0f) {
		float ds = es->down_scale;
		for (int d = 0; d < cx->dim; d++)
			y_h[d] *= ds;
	}
	for (int d = 0; d < cx->dim; d++)
		cx->out[d] += weight * y_h[d];
	return OK;
}

static status_code op_moe_router_softmax(backend *a, layer_weights *L, int E, int K, int dim,
										 int norm_topk, float routed_scale, float *router_input,
										 struct compute_scratch *s, buffer *slots) {
	float  logits_fallback[8192];
	float *logits = logits_fallback;
	if (E > (int)(sizeof(logits_fallback) / sizeof(float)))
		logits = xmalloc((size_t)E * sizeof(float));

	buffer *inp_buf = &s->router_softmax_inp_gpu;
	if (inp_buf->size < (size_t)dim * sizeof(float) || inp_buf->owner != a) {
		if (inp_buf->owner)
			inp_buf->owner->buffer_free(inp_buf->owner, inp_buf);
		memset(inp_buf, 0, sizeof(*inp_buf));
		status_code ast = a->buffer_alloc_scratch(a, (size_t)dim * sizeof(float), inp_buf);
		if (ast != OK) {
			if (logits != logits_fallback)
				free(logits);
			return ast;
		}
	}
	status_code st = a->buffer_write_f32(a, inp_buf, router_input, dim);
	if (st != OK) {
		if (logits != logits_fallback)
			free(logits);
		return st;
	}

	buffer *logits_gpu_buf = &s->router_logits_gpu;
	if (logits_gpu_buf->size < (size_t)E * sizeof(float) || logits_gpu_buf->owner != a) {
		if (logits_gpu_buf->owner)
			logits_gpu_buf->owner->buffer_free(logits_gpu_buf->owner, logits_gpu_buf);
		memset(logits_gpu_buf, 0, sizeof(*logits_gpu_buf));
		status_code ast = a->buffer_alloc_scratch(a, (size_t)E * sizeof(float), logits_gpu_buf);
		if (ast != OK) {
			if (logits != logits_fallback)
				free(logits);
			return ast;
		}
	}

	st = a->matmul(a, &L->router_w.buf, L->router_w.type, inp_buf, logits_gpu_buf, E, dim);
	if (st == OK)
		st = a->buffer_read_f32(a, logits_gpu_buf, logits, E);
	if (st != OK) {
		if (logits != logits_fallback)
			free(logits);
		return st;
	}

	int	  *idx_out = (int *)slots[RECIPE_SLOT_ROUTER_IDS].handle;
	float *w_out   = (float *)slots[RECIPE_SLOT_ROUTER_W].handle;
	moe_router_emit(E, K, 1, norm_topk, routed_scale, logits, NULL, NULL, idx_out, w_out);
	if (logits != logits_fallback)
		free(logits);
	return OK;
}

static status_code op_moe_router_direct(backend *a, layer_weights *L, int E, int K, int dim,
										int norm_topk, float routed_scale, buffer *inp,
										struct compute_scratch *s, buffer *slots) {
	float  logits_fallback[8192];
	float *logits = logits_fallback;
	if (E > (int)(sizeof(logits_fallback) / sizeof(float)))
		logits = xmalloc((size_t)E * sizeof(float));

	status_code st;
	if (backend_has_cap(a, BCAP_IS_HOST)) {
		buffer logits_buf	= {0};
		logits_buf.handle	= logits;
		logits_buf.host_ptr = logits;
		logits_buf.size		= (size_t)E * sizeof(float);
		st = a->matmul(a, &L->router_w.buf, L->router_w.type, inp, &logits_buf, E, dim);
	} else {
		buffer *logits_gpu_buf = &s->router_logits_gpu;
		if (logits_gpu_buf->size < (size_t)E * sizeof(float) || logits_gpu_buf->owner != a) {
			if (logits_gpu_buf->owner)
				logits_gpu_buf->owner->buffer_free(logits_gpu_buf->owner, logits_gpu_buf);
			memset(logits_gpu_buf, 0, sizeof(*logits_gpu_buf));
			status_code ast = a->buffer_alloc_scratch(a, (size_t)E * sizeof(float), logits_gpu_buf);
			if (ast != OK) {
				if (logits != logits_fallback)
					free(logits);
				return ast;
			}
		}
		st = a->matmul(a, &L->router_w.buf, L->router_w.type, inp, logits_gpu_buf, E, dim);
		if (st == OK)
			st = a->buffer_read_f32(a, logits_gpu_buf, logits, E);
	}
	if (st != OK) {
		if (logits != logits_fallback)
			free(logits);
		return st;
	}

	const float *bias	 = (const float *)L->router_bias.host_ptr;
	int			*idx_out = (int *)slots[RECIPE_SLOT_ROUTER_IDS].handle;
	float		*w_out	 = (float *)slots[RECIPE_SLOT_ROUTER_W].handle;
	moe_router_emit(E, K, 0, norm_topk, routed_scale, logits, bias, NULL, idx_out, w_out);
	if (logits != logits_fallback)
		free(logits);
	return OK;
}

status_code op_moe_router(exec_ctx *ctx) {
	const recipe_op		   *op	  = ctx->op;
	struct model		   *m	  = ctx->m;
	struct compute_scratch *s	  = ctx->s;
	int						li	  = ctx->li;
	buffer				   *slots = compute_slots_array(ctx->s);
	if (!model_layer_is_moe(m, li))
		return OK;

	backend		  *a			= model_layer_backend(ctx->m, ctx->li);
	layer_weights *L			= &m->layers[li];
	int			   E			= m->moe.n_experts;
	int			   K			= m->moe.n_experts_used;
	int			   dim			= op->u.matmul.k;
	float		   routed_scale = m->moe.routed_scale;
	int			   norm_topk	= m->moe.norm_topk_prob;
	int			   uses_softmax = m->arch_info->uses_moe_softmax_router;

	buffer *inp = &slots[op->in[0]];

	if (uses_softmax) {
		float *router_input = float_buf_ensure(&s->moe_xb_f, (size_t)dim);
		a->synchronize(a);
		backend	   *ib = inp->owner ? inp->owner : a;
		status_code rs = ib->buffer_read_f32(ib, inp, router_input, dim);
		if (rs != OK)
			return rs;

		moe_router_normalize_input(m, L, dim, router_input);

		return op_moe_router_softmax(a, L, E, K, dim, norm_topk, routed_scale, router_input, s,
									 slots);
	}

	return op_moe_router_direct(a, L, E, K, dim, norm_topk, routed_scale, inp, s, slots);
}

static void moe_expert_chunk(int begin, int end, int tid, void *ctx) {
	moe_par_job *j	 = ctx;
	int			 dim = j->dim;

	moe_expert_ctx cx = {
		.a			  = j->a,
		.inter		  = j->inter,
		.dim		  = dim,
		.use_gelu	  = j->use_gelu,
		.q8_gate_ok	  = j->xb_q8_gate_ok,
		.q8_gate_type = j->gate_q8_type,
		.xb_q8_gate	  = j->xb_q8_gate,
		.xb_f		  = j->xb_f,
		.out		  = (float *)((char *)j->all_outs + ((size_t)tid * j->per_thread_out)),
	};
	cx.scratch = (float *)((char *)j->all_scratch + ((size_t)tid * j->per_thread_scratch));

	for (int k = begin; k < end; k++) {
		int eid = j->expert_ids[k];
		if (eid < 0 || eid >= j->m->moe.n_experts)
			continue;
		status_code local_st = moe_expert_exec(&cx, &j->slot_buf[k], j->weights[k], tid);
		if (local_st != OK) {
			if (atomic_fetch_add(&j->n_err, 1) == 0)
				atomic_store(&j->first_err, local_st);
		}
	}
}

typedef struct {
	moe_stream_op *op;
	moe_par_job	  *job;
} moe_fused_job;

static void moe_fused_chunk(int begin, int end, int tid, void *ctx) {
	moe_fused_job *fj = (moe_fused_job *)ctx;
	for (int i = begin; i < end; i++) {
		int k = moe_stream_op_compute_k(fj->op, i);
		if (k < 0)
			moe_stream_op_fill_run(fj->op, i, tid);
		else
			moe_expert_chunk(k, k + 1, tid, fj->job);
	}
}

static status_code moe_experts_run_parallel(model *m, int li, int K, int dim, backend *a,
											moe_expert_slot *slot_buf, const int *expert_ids,
											const float *weights, buffer *xb, compute_scratch *s,
											int I, int any_fused, int use_gelu, int xb_q8_gate_ok,
											uint32_t gate_q8_type, const buffer *xb_q8_gate,
											size_t scratch_need, tpool *pool, int interleave,
											moe_stream_op *moe_op, float *outf) {
	int n_threads = tpool_n_threads(pool);
	if (n_threads < 1)
		n_threads = 1;
	if (n_threads > K)
		n_threads = K;

	size_t per_thread_scratch = scratch_need;
	size_t total_scratch	  = (size_t)n_threads * per_thread_scratch;
	float *all_scratch = float_buf_ensure(&s->moe_all_scratch, total_scratch / sizeof(float));

	size_t per_thread_out = (size_t)dim * sizeof(float);
	size_t total_out	  = (size_t)n_threads * per_thread_out;
	float *all_outs		  = float_buf_ensure(&s->moe_all_outs, total_out / sizeof(float));
	for (int t = 0; t < n_threads; t++) {
		float *o = (float *)((char *)all_outs + ((size_t)t * per_thread_out));
		for (int d = 0; d < dim; d++)
			o[d] = 0.0f;
	}
	for (int d = 0; d < dim; d++)
		outf[d] = 0.0f;

	float *xb_f = float_buf_ensure(&s->moe_xb_f, (size_t)dim);
	{
		backend *xb_owner = xb->owner ? xb->owner : a;
		if (xb_owner->synchronize)
			xb_owner->synchronize(xb_owner);
		status_code rs = xb_owner->buffer_read_f32(xb_owner, xb, xb_f, dim);
		if (rs != OK)
			return rs;
	}

	moe_par_job job = {.m				   = m,
					   .li				   = li,
					   .top_k			   = K,
					   .inter			   = I,
					   .dim				   = dim,
					   .any_fused		   = any_fused,
					   .use_gelu		   = use_gelu,
					   .xb_q8_gate_ok	   = xb_q8_gate_ok,
					   .gate_q8_type	   = gate_q8_type,
					   .slot_buf		   = slot_buf,
					   .expert_ids		   = expert_ids,
					   .weights			   = weights,
					   .xb_f			   = xb_f,
					   .xb_q8_gate		   = xb_q8_gate,
					   .a				   = a,
					   .all_scratch		   = all_scratch,
					   .per_thread_scratch = per_thread_scratch,
					   .all_outs		   = all_outs,
					   .per_thread_out	   = per_thread_out};
	atomic_store(&job.first_err, OK);
	atomic_store(&job.n_err, 0);

	if (interleave && moe_op) {
		moe_stream_op_set_compute_hook(moe_op, moe_expert_chunk, &job);
		moe_fused_job fj = {.op = moe_op, .job = &job};
		tpool_parallel_for(pool, moe_stream_op_n_items(moe_op), 1, moe_fused_chunk, &fj);
	} else {
		tpool_parallel_for(pool, K, 1, moe_expert_chunk, &job);
	}

	for (int t = 0; t < n_threads; t++) {
		float *o = (float *)((char *)all_outs + ((size_t)t * per_thread_out));
		for (int d = 0; d < dim; d++)
			outf[d] += o[d];
	}

	if (atomic_load(&job.n_err) > 0) {
		status_code st = atomic_load(&job.first_err);
		if (st == OK)
			st = ERR_INTERNAL;
		return st;
	}
	return OK;
}

static status_code moe_experts_run_sequential(model *m, int li, int K, int dim, backend *a,
											  moe_expert_slot *slot_buf, const int *expert_ids,
											  const float *weights, buffer *xb, compute_scratch *s,
											  int I, int use_gelu, int xb_q8_gate_ok,
											  uint32_t gate_q8_type, const buffer *xb_q8_gate,
											  size_t scratch_need, float *outf) {
	(void)li;
	float *scratch = float_buf_ensure(&s->moe_scratch, scratch_need / sizeof(float));

	float	*xb_f	  = float_buf_ensure(&s->moe_xb_f, dim);
	backend *xb_owner = xb->owner ? xb->owner : a;
	if (xb_owner->synchronize)
		xb_owner->synchronize(xb_owner);
	status_code st = xb_owner->buffer_read_f32(xb_owner, xb, xb_f, dim);
	if (st != OK)
		return st;

	for (int d = 0; d < dim; d++)
		outf[d] = 0.0f;

	moe_expert_ctx cx = {
		.a			  = a,
		.inter		  = I,
		.dim		  = dim,
		.use_gelu	  = use_gelu,
		.q8_gate_ok	  = xb_q8_gate_ok,
		.q8_gate_type = gate_q8_type,
		.xb_q8_gate	  = xb_q8_gate,
		.xb_f		  = xb_f,
		.scratch	  = scratch,
		.out		  = outf,
	};

	for (int k = 0; k < K; k++) {
		int eid = expert_ids[k];
		if (eid < 0 || eid >= m->moe.n_experts)
			continue;
		st = moe_expert_exec(&cx, &slot_buf[k], weights[k], 0);
		if (st != OK)
			return st;
	}
	return OK;
}

status_code op_moe_experts(exec_ctx *ctx) {
	struct model		   *m		= ctx->m;
	struct compute_scratch *s		= ctx->s;
	int						li		= ctx->li;
	buffer				   *slots	= compute_slots_array(ctx->s);
	buffer				   *out_buf = &slots[RECIPE_SLOT_XB2];
	int						dim		= m->dim;

	int out_is_host =
		out_buf->host_ptr != NULL || !out_buf->owner || out_buf->owner == backend_host();
	float  local_outf[MOE_MAX_DIM_STACK];
	float *heap_outf = NULL;
	float *outf;
	if (out_is_host) {
		outf = (float *)(out_buf->host_ptr ? out_buf->host_ptr : out_buf->handle);
		if (!outf) {
			ERROR("op_moe_experts: output slot XB2 has no host-accessible pointer (backend=%s)",
				  m->backend && m->backend->name ? m->backend->name : "?");
			return ERR_UNSUPPORTED;
		}
	} else if (dim <= MOE_MAX_DIM_STACK) {
		outf = local_outf;
	} else {
		heap_outf = xmalloc((size_t)dim * sizeof(float));
		outf	  = heap_outf;
	}
	if (!model_layer_is_moe(m, li)) {
		for (int d = 0; d < dim; d++)
			outf[d] = 0.0f;
		if (!out_is_host) {
			status_code wst = out_buf->owner->buffer_write_f32(out_buf->owner, out_buf, outf, dim);
			free(heap_outf);
			return wst;
		}
		return OK;
	}

	backend *a		= model_layer_backend(ctx->m, ctx->li);
	backend *host_a = backend_host();
	int		 K		= m->moe.n_experts_used;
	int		 I		= m->moe.moe_intermediate;

	const int	*expert_ids = (const int *)slots[RECIPE_SLOT_ROUTER_IDS].handle;
	const float *weights	= (const float *)slots[RECIPE_SLOT_ROUTER_W].handle;
	if (!expert_ids || !weights) {
		ERROR("op_moe_experts: router slot handles are NULL (layer=%d)", li);
		for (int d = 0; d < dim; d++)
			outf[d] = 0.0f;
		if (!out_is_host) {
			out_buf->owner->buffer_write_f32(out_buf->owner, out_buf, outf, dim);
			free(heap_outf);
		}
		return ERR_INVALID_ARG;
	}

	moe_expert_slot *slot_buf = s->moe_slot_buf;
	if (K > 64)
		K = 64;

	monitor_emit_moe_experts(g_monitor, li, -1, expert_ids, weights, K);

	tpool *pool		  = (a->get_pool && a->matmul_thread_local) ? a->get_pool(a) : NULL;
	int	   par		  = (pool && a->matmul_thread_local && K >= 2);
	int	   interleave = par;

	profile_scope  io_ps  = profile_begin(&s->prof, STAGE_MOE_IO_WAIT);
	moe_stream_op *moe_op = NULL;
	if (interleave) {
		moe_op = moe_stream_resolve_prep(m, li, expert_ids, K, slot_buf);
	} else {
		status_code st_w = moe_stream_resolve(m, li, expert_ids, K, slot_buf);
		if (st_w != OK) {
			profile_end(&s->prof, &io_ps);
			ERROR("op_moe_experts: moe_stream_resolve failed (layer=%d, st=%d)", li, st_w);
			for (int d = 0; d < dim; d++)
				outf[d] = 0.0f;
			if (!out_is_host) {
				out_buf->owner->buffer_write_f32(out_buf->owner, out_buf, outf, dim);
				free(heap_outf);
			}
			return st_w;
		}
	}
	profile_end(&s->prof, &io_ps);

	profile_scope compute_ps = profile_begin(&s->prof, STAGE_MATMUL_FFN);

	int use_gelu = m->arch_info->uses_gelu_activation;

	size_t scratch_need = ((size_t)I * sizeof(float) * 3) + ((size_t)dim * sizeof(float));
	int	   any_fused	= m->layers[li].any_fused_experts;
	if (any_fused)
		scratch_need += (size_t)I * 2 * sizeof(float);

	buffer	   *xb = &slots[RECIPE_SLOT_XB];
	status_code st = OK;

	int	   has_qonly = backend_has_cap(a, BCAP_MATMUL_QONLY) && a->matmul_qonly && a->prequantize_x;
	buffer xb_q8_gate	  = {0};
	int	   xb_q8_gate_ok  = 0;
	uint32_t gate_q8_type = has_qonly ? m->layers[li].gate_q8_type : 0;
	if (has_qonly) {
		if (gate_q8_type) {
			st = a->prequantize_x(a, xb, dim, gate_q8_type, &xb_q8_gate);
			if (st == OK)
				xb_q8_gate_ok = 1;
			else if (st != ERR_UNSUPPORTED)
				goto cleanup;
		}
	}

	if (par) {
		st = moe_experts_run_parallel(m, li, K, dim, host_a, slot_buf, expert_ids, weights, xb, s,
									  I, any_fused, use_gelu, xb_q8_gate_ok, gate_q8_type,
									  &xb_q8_gate, scratch_need, pool, interleave, moe_op, outf);
	} else {
		st = moe_experts_run_sequential(m, li, K, dim, host_a, slot_buf, expert_ids, weights, xb, s,
										I, use_gelu, xb_q8_gate_ok, gate_q8_type, &xb_q8_gate,
										scratch_need, outf);
	}

	if (moe_op) {
		moe_stream_op_finish(moe_op);
		moe_stream_op_free(moe_op);
		moe_op = NULL;
	}

cleanup:
	profile_end(&s->prof, &compute_ps);
	for (int k = 0; k < K; k++) {
		if (slot_buf[k].owned && slot_buf[k].heap_buf) {
			free(slot_buf[k].heap_buf);
			slot_buf[k].heap_buf = NULL;
			slot_buf[k].owned	 = 0;
		}
	}
	moe_stream_release_slots(m, li, slot_buf, K);

	if (!out_is_host) {
		status_code wst = out_buf->owner->buffer_write_f32(out_buf->owner, out_buf, outf, dim);
		free(heap_outf);
		if (st == OK)
			st = wst;
	}

	return st;
}

status_code op_moe_shared(exec_ctx *ctx) {
	struct model		   *m	   = ctx->m;
	struct compute_scratch *s	   = ctx->s;
	int						li	   = ctx->li;
	buffer				   *slots  = compute_slots_array(ctx->s);
	backend				   *a	   = model_layer_backend(ctx->m, ctx->li);
	layer_weights		   *L	   = &m->layers[li];
	int						dim	   = m->dim;
	int						is_moe = model_layer_is_moe(m, li);

	int sh_inter = is_moe ? (m->moe.moe_intermediate * m->moe.n_shared_experts) : m->intermediate;
	if (is_moe && m->moe.n_shared_experts == 0)
		return OK;

	profile		 *prof = &s->prof;
	profile_scope ps   = profile_begin(prof, STAGE_MOE_SHARED);

	buffer *xb		 = &slots[RECIPE_SLOT_XB];
	buffer *gate_buf = &slots[RECIPE_SLOT_FFN_GATE];
	buffer *up_buf	 = &slots[RECIPE_SLOT_FFN_UP];
	buffer *act_buf	 = &slots[RECIPE_SLOT_FFN_ACT];

	uint32_t gate_type = is_moe ? L->shexp_gate_w.type : L->gate_w.type;
	uint32_t up_type   = is_moe ? L->shexp_up_w.type : L->up_w.type;
	int has_qonly	 = backend_has_cap(a, BCAP_MATMUL_QONLY) && a->matmul_qonly && a->prequantize_x;
	uint32_t gate_q8 = has_qonly ? wtype_to_q8type(gate_type) : 0;
	uint32_t up_q8	 = has_qonly ? wtype_to_q8type(up_type) : 0;
	int		 share_q8 = gate_q8 && gate_q8 == up_q8;

	buffer		xb_q8 = {0};
	status_code st;

	if (is_moe) {
		if (share_q8) {
			st = a->prequantize_x(a, xb, dim, gate_q8, &xb_q8);
			if (st != OK)
				share_q8 = 0;
		}
		if (share_q8) {
			st = a->matmul_qonly(a, &L->shexp_gate_w.buf, gate_type, &xb_q8, gate_q8, gate_buf,
								 sh_inter, dim, 1);
		} else {
			st = a->matmul(a, &L->shexp_gate_w.buf, L->shexp_gate_w.type, xb, gate_buf, sh_inter,
						   dim);
		}
		if (st != OK)
			goto done;
		buffer up_w_buf =
			buffer_slice(&L->shexp_up_w.buf, 0, L->shexp_up_w.buf.size - L->shexp_up_w.buf.offset);
		if (share_q8) {
			st = a->matmul_qonly(a, &up_w_buf, up_type, &xb_q8, up_q8, up_buf, sh_inter, dim, 1);
		} else {
			st = a->matmul(a, &up_w_buf, L->shexp_up_w.type, xb, up_buf, sh_inter, dim);
		}
		if (st != OK)
			goto done;
	} else {
		if (L->gate_up_fused) {
			st = a->matmul(a, &L->gate_up_w.buf, L->gate_up_w.type, xb,
						   &slots[RECIPE_SLOT_FFN_GATE_UP], 2 * sh_inter, dim);
			if (st != OK)
				goto done;
			*gate_buf =
				buffer_slice(&slots[RECIPE_SLOT_FFN_GATE_UP], 0, (size_t)sh_inter * sizeof(float));
			*up_buf =
				buffer_slice(&slots[RECIPE_SLOT_FFN_GATE_UP], (size_t)sh_inter * sizeof(float),
							 (size_t)sh_inter * sizeof(float));
		} else {
			if (share_q8) {
				st = a->prequantize_x(a, xb, dim, gate_q8, &xb_q8);
				if (st != OK)
					share_q8 = 0;
			}
			if (share_q8) {
				st = a->matmul_qonly(a, &L->gate_w.buf, gate_type, &xb_q8, gate_q8, gate_buf,
									 sh_inter, dim, 1);
			} else {
				st = a->matmul(a, &L->gate_w.buf, L->gate_w.type, xb, gate_buf, sh_inter, dim);
			}
			if (st != OK)
				goto done;
			if (share_q8) {
				st = a->matmul_qonly(a, &L->up_w.buf, up_type, &xb_q8, up_q8, up_buf, sh_inter, dim,
									 1);
			} else {
				st = a->matmul(a, &L->up_w.buf, L->up_w.type, xb, up_buf, sh_inter, dim);
			}
			if (st != OK)
				goto done;
		}
	}

	float *g = (float *)((char *)gate_buf->handle + gate_buf->offset);
	float *u = (float *)((char *)up_buf->handle + up_buf->offset);
	float *o = (float *)((char *)act_buf->handle + act_buf->offset);
	moe_activate(o, g, u, sh_inter, 1.0f, 1.0f, m->arch_info->uses_gelu_activation);

	float *yp	   = float_buf_ensure(&s->moe_shared_y, (size_t)dim);
	buffer y_buf   = {0};
	y_buf.handle   = yp;
	y_buf.host_ptr = yp;
	y_buf.owner	   = NULL;
	y_buf.size	   = (size_t)dim * sizeof(float);

	if (is_moe) {
		st = a->matmul(a, &L->shexp_down_w.buf, L->shexp_down_w.type, act_buf, &y_buf, dim,
					   sh_inter);
	} else {
		st = a->matmul(a, &L->down_w.buf, L->down_w.type, act_buf, &y_buf, dim, sh_inter);
	}
	if (st != OK)
		goto done;
	memcpy(act_buf->handle, yp, (size_t)dim * sizeof(float));
	st = OK;

done:
	profile_end(prof, &ps);
	return st;
}