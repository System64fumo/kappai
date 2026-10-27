#include "recipe/internal.h"

typedef struct {
	model			*m;
	backend			*a;
	batch_scratch	*bs;
	moe_expert_slot *slot_buf;
	int				 li, top_k, inter, dim, n_tokens, use_gelu;
	const float		*xb_base;
	float			*all_scratch;
	float			*all_outs;
	size_t			 per_thread_scratch;
	size_t			 per_expert_out;
} moe_batch_job;

typedef struct {
	batch_scratch *bs;
	layer_weights *L;
	int			   E, K, rows;
	int			   uses_softmax, norm_topk;
	float		   routed_scale;
	int			   n_group, topk_group;
} moe_router_emit_job;

typedef struct {
	int dim;
	int K;
	int I;
	int use_gelu;
} moe_exec_plan;

typedef struct {
	int			  is_moe;
	int			  sh_inter;
	int			  use_gelu;
	int			  gate_up_fused;
	const buffer *gate_w;
	uint32_t	  gate_type;
	const buffer *up_w;
	uint32_t	  up_type;
	const buffer *down_w;
	uint32_t	  down_type;
	const buffer *gate_up_w;
	uint32_t	  gate_up_type;
} moe_shared_plan;

typedef struct {
	moe_resident_expert *experts;
	int					*counts;
	int					*idx;
	int					*rows_packed;
	float				*weights_packed;
	backend			   **owners;
} moe_pack;

static int	moe_router_emit_ex(int E, int K, int use_softmax, int norm_topk, float routed_scale,
							   int n_group, int topk_group, float *logits, const float *bias,
							   float *scores_scratch, int *ids_out, float *w_out);
static void moe_batch_expert_chunk(int begin, int end, int tid, void *ctx);

static int moe_order_pair_cmp(const void *pa, const void *pb) {
	const moe_order_pair *a = pa, *b = pb;
	if (a->cnt != b->cnt)
		return b->cnt - a->cnt;
	return a->u - b->u;
}

static inline void batch_sync(const exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (a && a->synchronize && !backend_has_cap(a, BCAP_IS_HOST))
		a->synchronize(a);
}

int moe_topk_select(const float *scores, int n_experts, int top_k, int *top_idx, float *top_score) {
	if (top_k <= 0)
		return 0;
	if (top_k > MOE_MAX_TOPK)
		top_k = MOE_MAX_TOPK;
	int hn = topk_heap_select(scores, n_experts, top_k, top_score, top_idx);
	for (int k = hn; k < top_k; k++) {
		top_idx[k]	 = -1;
		top_score[k] = -1e30f;
	}
	return top_k;
}

static int moe_grouped_topk_select(const float *scores, int E, int K, int n_group, int topk_group,
								   int *top_idx, float *top_score) {
	if (n_group <= 1 || topk_group <= 0 || topk_group >= n_group || n_group > E)
		return moe_topk_select(scores, E, K, top_idx, top_score);

	enum { GROUP_STACK_CAP = 256, CAND_STACK_CAP = 2048 };
	float  group_sum_stack[GROUP_STACK_CAP];
	float  group_kept_stack[GROUP_STACK_CAP];
	int	   group_sel_stack[GROUP_STACK_CAP];
	float  cand_score_stack[CAND_STACK_CAP];
	int	   cand_eid_stack[CAND_STACK_CAP];
	float *group_sum  = group_sum_stack;
	float *group_kept = group_kept_stack;
	int	  *group_sel  = group_sel_stack;
	float *cand_score = cand_score_stack;
	int	  *cand_eid	  = cand_eid_stack;
	if (n_group > GROUP_STACK_CAP) {
		group_sum  = xmalloc((size_t)n_group * sizeof(float));
		group_kept = xmalloc((size_t)n_group * sizeof(float));
		group_sel  = xmalloc((size_t)n_group * sizeof(int));
	}
	if (E > CAND_STACK_CAP) {
		cand_score = xmalloc((size_t)E * sizeof(float));
		cand_eid   = xmalloc((size_t)E * sizeof(int));
	}

	int epg		= E / n_group;
	int rem		= E % n_group;
	int g_start = 0;
	for (int g = 0; g < n_group; g++) {
		int	  g_end	 = g_start + epg + (g < rem ? 1 : 0);
		float best	 = -1e30f;
		float second = -1e30f;
		for (int e = g_start; e < g_end; e++) {
			if (scores[e] > best) {
				second = best;
				best   = scores[e];
			} else if (scores[e] > second) {
				second = scores[e];
			}
		}
		group_sum[g] = best + second;
		g_start		 = g_end;
	}

	int n_kept = topk_heap_select(group_sum, n_group, topk_group, group_kept, group_sel);

	for (int i = 1; i < n_kept; i++) {
		int g = group_sel[i];
		int j = i - 1;
		while (j >= 0 && group_sel[j] > g) {
			group_sel[j + 1] = group_sel[j];
			j--;
		}
		group_sel[j + 1] = g;
	}

	int n_cand = 0;
	for (int i = 0; i < n_kept; i++) {
		int g  = group_sel[i];
		int gs = g * epg + (g < rem ? g : rem);
		int ge = gs + epg + (g < rem ? 1 : 0);
		for (int e = gs; e < ge; e++) {
			cand_score[n_cand] = scores[e];
			cand_eid[n_cand]   = e;
			n_cand++;
		}
	}

	int sel = moe_topk_select(cand_score, n_cand, K, top_idx, top_score);
	for (int k = 0; k < sel; k++)
		if (top_idx[k] >= 0)
			top_idx[k] = cand_eid[top_idx[k]];

	if (group_sum != group_sum_stack) {
		free(group_sum);
		free(group_kept);
		free(group_sel);
	}
	if (cand_score != cand_score_stack) {
		free(cand_score);
		free(cand_eid);
	}
	return sel;
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

status_code moe_activate(backend *a, float *act, const float *gate, const float *up, int n_i,
						 float gs, float us, int use_gelu) {
	if (!(a && a->moe_activate && backend_has_cap(a, BCAP_IS_HOST)))
		backend_report_host_fallback(a, "moe_activate", HFB_OP_NOT_NATIVE,
									 "moe activation executes on the host (cpu) fallback");
	backend *h = (a && a->moe_activate && backend_has_cap(a, BCAP_IS_HOST)) ? a : backend_host();
	if (!h || !h->moe_activate)
		return ERR_UNSUPPORTED;
	buffer gb = buffer_host_view(h, (void *)gate, (size_t)n_i * sizeof(float));
	buffer ub = buffer_host_view(h, (void *)up, (size_t)n_i * sizeof(float));
	buffer ob = buffer_host_view(h, act, (size_t)n_i * sizeof(float));
	return h->moe_activate(h, &gb, &ub, &ob, n_i, gs, us, use_gelu);
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

static int moe_router_logits_finite(const float *logits, int n) {
	for (int e = 0; e < n; e++)
		if (!isfinite(logits[e]))
			return 0;
	return 1;
}

int moe_router_emit(int E, int K, int use_softmax, int norm_topk, float routed_scale, float *logits,
					const float *bias, float *scores_scratch, int *ids_out, float *w_out) {
	return moe_router_emit_ex(E, K, use_softmax, norm_topk, routed_scale, 1, 0, logits, bias,
							  scores_scratch, ids_out, w_out);
}

int moe_router_emit_ex(int E, int K, int use_softmax, int norm_topk, float routed_scale,
					   int n_group, int topk_group, float *logits, const float *bias,
					   float *scores_scratch, int *ids_out, float *w_out) {
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

		K = moe_topk_select(logits, E, K, top_idx, top_score);
		for (int k = 0; k < K; k++)
			weight[k] = top_score[k];
	} else {
		float *scores = scores_scratch;
		float  scores_stack[8192];
		if (!scores)
			scores = (E <= (int)ARRAY_LEN(scores_stack)) ? scores_stack
														 : xmalloc((size_t)E * sizeof(float));
		for (int e = 0; e < E; e++)
			scores[e] = (1.0f / (1.0f + expf(-logits[e]))) + (bias ? bias[e] : 0.0f);
		K = moe_grouped_topk_select(scores, E, K, n_group, topk_group, top_idx, top_score);
		if (scores != scores_scratch && scores != scores_stack)
			free(scores);
		for (int k = 0; k < K; k++)
			weight[k] = (top_idx[k] >= 0) ? top_score[k] - (bias ? bias[top_idx[k]] : 0.0f) : 0.0f;
	}

	moe_apply_weights(weight, K, norm_topk, routed_scale);
	for (int k = 0; k < K; k++) {
		ids_out[k] = top_idx[k];
		w_out[k]   = weight[k];
	}
	return K;
}

static void moe_router_emit_chunk(int begin, int end, int tid, void *v) {
	(void)tid;
	moe_router_emit_job *j = v;
	for (int row = begin; row < end; row++) {
		float		*logits	 = j->bs->moe_router_logits.p + (size_t)row * j->E;
		int			*row_ids = j->bs->moe_router_ids + (size_t)row * j->K;
		float		*row_w	 = j->bs->moe_router_w + (size_t)row * j->K;
		const float *bias	 = (const float *)j->L->router_bias.host_ptr;
		int			 K_row	 = moe_router_emit_ex(
			j->E, j->K, j->uses_softmax, j->norm_topk, j->routed_scale, j->n_group, j->topk_group,
			logits, bias, j->bs->moe_router_logits.p + (size_t)j->rows * j->E + (size_t)row * j->E,
			row_ids, row_w);
		for (int k = K_row; k < j->K; k++) {
			row_ids[k] = -1;
			row_w[k]   = 0.0f;
		}
	}
}

static void moe_batch_ensure_rowslots(batch_scratch *bs, int rows, int K) {
	size_t need = (size_t)rows * (size_t)K;
	if ((size_t)bs->moe_rowslots_cap < need) {
		bs->moe_router_ids	  = xrealloc(bs->moe_router_ids, need * sizeof(int));
		bs->moe_router_w	  = xrealloc(bs->moe_router_w, need * sizeof(float));
		bs->moe_union_ids	  = xrealloc(bs->moe_union_ids, need * sizeof(int));
		bs->moe_union_rows	  = xrealloc(bs->moe_union_rows, need * sizeof(int));
		bs->moe_union_kidx	  = xrealloc(bs->moe_union_kidx, need * sizeof(int));
		bs->moe_union_offsets = xrealloc(bs->moe_union_offsets, (need + 1) * sizeof(int));
		bs->moe_union_cursor  = xrealloc(bs->moe_union_cursor, need * sizeof(int));
		bs->moe_union_count	  = xrealloc(bs->moe_union_count, need * sizeof(int));
		bs->moe_rowslots_cap  = (int)need;
	}
}

static void moe_batch_ensure_experts(batch_scratch *bs, int E) {
	if (bs->moe_expert_alloc < E) {
		bs->moe_expert_union_idx = xrealloc(bs->moe_expert_union_idx, (size_t)E * sizeof(int));
		bs->moe_expert_seen_gen	 = xrealloc(bs->moe_expert_seen_gen, (size_t)E * sizeof(int));
		memset(bs->moe_expert_seen_gen, 0, (size_t)E * sizeof(int));
		bs->moe_expert_gen	 = 0;
		bs->moe_expert_alloc = E;
	}
}

static void moe_union_slots_ensure(batch_scratch *bs, int n_union) {
	if (bs->moe_union_slots_cap < n_union) {
		moe_expert_slot *grown =
			xrealloc(bs->moe_union_slots, (size_t)n_union * sizeof(moe_expert_slot));
		size_t old_n = (size_t)bs->moe_union_slots_cap;
		memset(grown + old_n, 0, ((size_t)n_union - old_n) * sizeof(moe_expert_slot));
		bs->moe_union_slots		= grown;
		bs->moe_union_slots_cap = n_union;
	}
}

status_code moe_router_batch(exec_ctx *ctx) {

	backend *a = exec_layer_backend(ctx);
	if (!model_layer_is_moe(ctx->m, ctx->li))
		return OK;

	int			   E			= ctx->m->moe.n_experts;
	int			   K			= ctx->m->moe.n_experts_used;
	int			   router_dim	= ctx->op->u.matmul.k;
	float		   routed_scale = ctx->m->moe.routed_scale;
	int			   norm_topk	= ctx->m->moe.norm_topk_prob;
	int			   uses_softmax = ctx->m->arch_info->uses_moe_softmax_router;
	layer_weights *L			= &ctx->m->layers[ctx->li];

	moe_batch_ensure_rowslots(ctx->bs, ctx->rows.n_rows, K);
	moe_batch_ensure_experts(ctx->bs, E);
	float_buf_ensure(&ctx->bs->moe_router_logits, (size_t)ctx->rows.n_rows * E * 2);
	float_buf_ensure(&ctx->bs->moe_router_inp, (size_t)ctx->rows.n_rows * router_dim);
	float_buf_ensure(&ctx->bs->moe_xb_f, (size_t)ctx->rows.n_rows * router_dim);

	buffer *xb = batch_slot(ctx->bs, ctx->op->in[0]);

	int			n_rows	= ctx->rows.n_rows;
	float	   *xb_host = ctx->bs->moe_xb_f.p;
	status_code st		= a->buffer_read_f32(a, xb, xb_host, n_rows * router_dim);
	if (st != OK)
		return st;

	for (int row = 0; row < n_rows; row++) {
		const float *router_input_src = xb_host + (size_t)row * router_dim;
		float		*router_input	  = ctx->bs->moe_router_inp.p + (size_t)row * router_dim;
		memcpy(router_input, router_input_src, (size_t)router_dim * sizeof(float));

		if (uses_softmax)
			moe_router_normalize_input(ctx->m, L, router_dim, router_input);
	}

	buffer *router_inp_buf = &ctx->bs->moe_router_inp_dev;
	st = buffer_ensure_scratch(a, router_inp_buf,
							   (size_t)ctx->rows.n_rows * router_dim * sizeof(float));
	if (st != OK)
		return st;

	st = a->buffer_write_f32(a, router_inp_buf, ctx->bs->moe_router_inp.p,
							 ctx->rows.n_rows * router_dim);
	if (st != OK)
		return st;

	buffer *router_logits_buf = &ctx->bs->moe_router_logits_dev;
	st = buffer_ensure_scratch(a, router_logits_buf, (size_t)ctx->rows.n_rows * E * sizeof(float));
	if (st != OK)
		return st;

	st = a->matmul_batch(a, &L->router_w.buf, L->router_w.type, router_inp_buf, router_logits_buf,
						 E, router_dim, ctx->rows.n_rows);
	if (st != OK)
		return st;

	st = a->buffer_read_f32(a, router_logits_buf, ctx->bs->moe_router_logits.p,
							ctx->rows.n_rows * E);
	if (st != OK)
		return st;

	if (!moe_router_logits_finite(ctx->bs->moe_router_logits.p, (size_t)ctx->rows.n_rows * E)) {
		ERROR("moe_router_batch: non-finite router logit (layer=%d, rows=%d, E=%d)", ctx->li,
			  ctx->rows.n_rows, E);
		return ERR_FORMAT;
	}

	tpool			   *router_pool = model_get_pool(ctx->m);
	moe_router_emit_job emit_job	= {.bs			 = ctx->bs,
									   .L			 = L,
									   .E			 = E,
									   .K			 = K,
									   .rows		 = ctx->rows.n_rows,
									   .uses_softmax = uses_softmax,
									   .norm_topk	 = norm_topk,
									   .routed_scale = routed_scale,
									   .n_group		 = ctx->m->moe.n_group,
									   .topk_group	 = ctx->m->moe.topk_group};
	if (router_pool && ctx->rows.n_rows >= 8 && E >= 8 && tpool_current_tid() < 0) {
		tpool_parallel_for(router_pool, ctx->rows.n_rows, 1, moe_router_emit_chunk, &emit_job);
	} else {
		moe_router_emit_chunk(0, ctx->rows.n_rows, 0, &emit_job);
	}

	int gen = ++ctx->bs->moe_expert_gen;

	int n_union = 0;
	for (int row = 0; row < ctx->rows.n_rows; row++) {
		const int *row_ids = ctx->bs->moe_router_ids + (size_t)row * K;
		for (int k = 0; k < K; k++) {
			int eid = row_ids[k];
			if (eid < 0)
				continue;
			int union_idx;
			if (ctx->bs->moe_expert_seen_gen[eid] == gen) {
				union_idx = ctx->bs->moe_expert_union_idx[eid];
			} else {
				union_idx							= n_union++;
				ctx->bs->moe_expert_seen_gen[eid]	= gen;
				ctx->bs->moe_expert_union_idx[eid]	= union_idx;
				ctx->bs->moe_union_ids[union_idx]	= eid;
				ctx->bs->moe_union_count[union_idx] = 0;
			}
			ctx->bs->moe_union_count[union_idx]++;
		}
	}
	ctx->bs->moe_n_union = n_union;

	ctx->bs->moe_union_offsets[0] = 0;
	for (int u = 0; u < n_union; u++)
		ctx->bs->moe_union_offsets[u + 1] =
			ctx->bs->moe_union_offsets[u] + ctx->bs->moe_union_count[u];

	{
		int *cursor = ctx->bs->moe_union_cursor;
		for (int u = 0; u < n_union; u++)
			cursor[u] = ctx->bs->moe_union_offsets[u];
		for (int row = 0; row < ctx->rows.n_rows; row++) {
			const int *row_ids = ctx->bs->moe_router_ids + (size_t)row * K;
			for (int k = 0; k < K; k++) {
				if (row_ids[k] < 0)
					continue;
				int u						  = ctx->bs->moe_expert_union_idx[row_ids[k]];
				int slot					  = cursor[u]++;
				ctx->bs->moe_union_rows[slot] = row;
				ctx->bs->moe_union_kidx[slot] = k;
			}
		}
	}

	{
		ARR_ENSURE(ctx->bs->moe_union_order, n_union, ctx->bs->moe_union_order_cap);
		ARR_ENSURE(ctx->bs->moe_union_sorted_ids, n_union, ctx->bs->moe_union_sorted_ids_cap);
		ARR_ENSURE(ctx->bs->moe_order_pairs, n_union, ctx->bs->moe_order_pairs_cap);
		for (int u = 0; u < n_union; u++)
			ctx->bs->moe_order_pairs[u] =
				(moe_order_pair){.cnt = ctx->bs->moe_union_count[u], .u = u};
		qsort(ctx->bs->moe_order_pairs, (size_t)n_union, sizeof(moe_order_pair),
			  moe_order_pair_cmp);
		for (int i = 0; i < n_union; i++) {
			ctx->bs->moe_union_order[i] = ctx->bs->moe_order_pairs[i].u;
			ctx->bs->moe_union_sorted_ids[i] =
				ctx->bs->moe_union_ids[ctx->bs->moe_order_pairs[i].u];
		}
	}

	{
		int total_slots = ctx->rows.n_rows * K;
		ARR_ENSURE(ctx->bs->moe_per_token_slots, total_slots, ctx->bs->moe_per_token_slots_cap);
	}

	moe_union_slots_ensure(ctx->bs, n_union);

	status_code rst = moe_stream_resolve(ctx->m, ctx->li, ctx->bs->moe_union_sorted_ids, n_union,
										 ctx->bs->moe_union_slots);
	if (rst != OK) {
		ERROR("moe_router_batch: moe_stream_resolve failed (layer=%d, union=%d, st=%d)", ctx->li,
			  n_union, rst);
		ctx->bs->moe_n_union		 = 0;
		ctx->bs->moe_union_pending_n = 0;
		return ERR_INTERNAL;
	}
	ctx->bs->moe_union_pending_n = n_union;
	return OK;
}

static status_code moe_experts_grouped(exec_ctx *ctx, backend *a, int dim, int K, int I,
									   int use_gelu) {
	batch_scratch *bs	   = ctx->bs;
	int			   n_rows  = ctx->rows.n_rows;
	int			   n_union = bs->moe_n_union;

	buffer	   *xb_dev	= batch_slot(bs, ctx->op->in[0]);
	float	   *xb_host = float_buf_ensure(&bs->moe_xb_f, (size_t)n_rows * dim);
	status_code st		= a->buffer_read_f32(a, xb_dev, xb_host, n_rows * dim);
	if (st != OK)
		return st;
	const float *xb_base = xb_host;
	float		*out	 = bs->moe_out.p;

	float_buf_ensure(&bs->moe_gather_x, (size_t)n_rows * dim);
	float_buf_ensure(&bs->moe_exp_act, (size_t)n_rows * I);
	float_buf_ensure(&bs->moe_exp_y, (size_t)n_rows * dim);
	float *contrib = float_buf_ensure(&bs->moe_expert_out, (size_t)n_rows * K * dim);
	memset(contrib, 0, (size_t)n_rows * K * dim * sizeof(float));

	for (int i = 0; i < n_union; i++) {
		int				 u	  = bs->moe_union_order[i];
		int				 cnt  = bs->moe_union_count[u];
		const int		*rows = bs->moe_union_rows + bs->moe_union_offsets[u];
		const int		*kidx = bs->moe_union_kidx + bs->moe_union_offsets[u];
		moe_expert_slot *es	  = &bs->moe_union_slots[i];
		if (cnt <= 0 || es->eid < 0 || !es->gate_w)
			continue;
		moe_stream_wait_slot(es);

		float *X = bs->moe_gather_x.p;
		for (int c = 0; c < cnt; c++)
			memcpy(X + (size_t)c * dim, xb_base + (size_t)rows[c] * dim,
				   (size_t)dim * sizeof(float));
		float *A = bs->moe_exp_act.p;

		if (es->gate_up_fused) {
			float *GU = float_buf_ensure(&bs->moe_exp_gu, (size_t)cnt * I * 2);
			for (int c = 0; c < cnt; c++)
				host_matmul_generic(es->gate_w, es->gate_type, X + (size_t)c * dim,
									GU + (size_t)c * I * 2, I * 2, dim);
			for (int c = 0; c < cnt; c++) {
				st = moe_activate(backend_host(), A + (size_t)c * I, GU + (size_t)c * I * 2,
								  GU + (size_t)c * I * 2 + I, I, es->gate_scale, es->up_scale,
								  use_gelu);
				if (st != OK)
					return st;
			}
		} else {
			float *G = float_buf_ensure(&bs->moe_exp_gu, (size_t)cnt * I);
			float *U = float_buf_ensure(&bs->moe_up_scratch, (size_t)cnt * I);
			for (int c = 0; c < cnt; c++) {
				host_matmul_generic(es->gate_w, es->gate_type, X + (size_t)c * dim,
									G + (size_t)c * I, I, dim);
				host_matmul_generic(es->up_w, es->up_type, X + (size_t)c * dim, U + (size_t)c * I,
									I, dim);
			}
			for (int c = 0; c < cnt; c++) {
				st = moe_activate(backend_host(), A + (size_t)c * I, G + (size_t)c * I,
								  U + (size_t)c * I, I, es->gate_scale, es->up_scale, use_gelu);
				if (st != OK)
					return st;
			}
		}

		float *Y = bs->moe_exp_y.p;
		for (int c = 0; c < cnt; c++)
			host_matmul_generic(es->down_w, es->down_type, A + (size_t)c * I, Y + (size_t)c * dim,
								dim, I);

		float ds = es->down_scale;
		for (int c = 0; c < cnt; c++) {
			float		 w = bs->moe_router_w[(size_t)rows[c] * K + kidx[c]];
			const float *y = Y + (size_t)c * dim;
			float		*o = contrib + ((size_t)rows[c] * K + kidx[c]) * dim;
			if (ds == 1.0f) {
				for (int d = 0; d < dim; d++)
					o[d] = w * y[d];
			} else {
				for (int d = 0; d < dim; d++)
					o[d] = w * (ds * y[d]);
			}
		}
	}
	for (int row = 0; row < n_rows; row++) {
		float *o = out + (size_t)row * dim;
		for (int k = 0; k < K; k++) {
			const float *c = contrib + ((size_t)row * K + k) * dim;
			for (int d = 0; d < dim; d++)
				o[d] += c[d];
		}
	}
	return OK;
}

static size_t moe_pack_align(size_t n) {
	return (n + 15) & ~(size_t)15;
}

static moe_pack moe_pack_get(batch_scratch *bs, int n_union, int n_rows, int K) {
	size_t n_pack	= (size_t)n_rows * K;
	size_t o_counts = moe_pack_align((size_t)n_union * sizeof(moe_resident_expert));
	size_t o_idx	= o_counts + moe_pack_align((size_t)n_union * sizeof(int));
	size_t o_rows	= o_idx + moe_pack_align((size_t)n_union * sizeof(int));
	size_t o_w		= o_rows + moe_pack_align(n_pack * sizeof(int));
	size_t o_own	= o_w + moe_pack_align(n_pack * sizeof(float));
	size_t total	= o_own + (size_t)n_union * sizeof(backend *);
	if (bs->moe_pack_cap < total) {
		free(bs->moe_pack);
		bs->moe_pack	 = xmalloc_aligned(total, 16);
		bs->moe_pack_cap = total;
	}
	memset(bs->moe_pack, 0, total);
	return (moe_pack){.experts		  = (moe_resident_expert *)bs->moe_pack,
					  .counts		  = (int *)(bs->moe_pack + o_counts),
					  .idx			  = (int *)(bs->moe_pack + o_idx),
					  .rows_packed	  = (int *)(bs->moe_pack + o_rows),
					  .weights_packed = (float *)(bs->moe_pack + o_w),
					  .owners		  = (backend **)(bs->moe_pack + o_own)};
}

static status_code moe_experts_resident_grouped(exec_ctx *ctx, backend *a, int dim, int K, int I,
												int use_gelu) {
	batch_scratch *bs	   = ctx->bs;
	int			   n_rows  = ctx->rows.n_rows;
	int			   n_union = bs->moe_n_union;

	buffer *xb_dev = batch_slot(bs, ctx->op->in[0]);
	buffer *out_b  = batch_slot(bs, ctx->op->out);
	if (xb_dev->owner != a || out_b->owner != a)
		return ERR_UNSUPPORTED;

	int counts[MOE_MAX_TOPK * 8];
	if (n_union > (int)(sizeof(counts) / sizeof(counts[0])))
		return ERR_UNSUPPORTED;

	moe_pack			 pack			= moe_pack_get(bs, n_union, n_rows, K);
	moe_resident_expert *experts		= pack.experts;
	int					*rows_packed	= pack.rows_packed;
	float				*weights_packed = pack.weights_packed;
	int					 packed			= 0;
	for (int i = 0; i < n_union; i++) {
		int				 u	  = bs->moe_union_order[i];
		int				 cnt  = bs->moe_union_count[u];
		const int		*rows = bs->moe_union_rows + bs->moe_union_offsets[u];
		const int		*kidx = bs->moe_union_kidx + bs->moe_union_offsets[u];
		moe_expert_slot *es	  = &bs->moe_union_slots[i];
		if (cnt <= 0) {
			counts[i] = 0;
			continue;
		}
		if (es->eid < 0 || !es->dev_ready || es->dev_gate.owner != a) {
			return ERR_UNSUPPORTED;
		}
		counts[i]				 = cnt;
		experts[i].gate_w		 = &es->dev_gate;
		experts[i].up_w			 = &es->dev_up;
		experts[i].down_w		 = &es->dev_down;
		experts[i].gate_type	 = es->gate_type;
		experts[i].up_type		 = es->up_type;
		experts[i].down_type	 = es->down_type;
		experts[i].gate_up_fused = es->gate_up_fused;
		experts[i].use_gelu		 = use_gelu;
		experts[i].gate_scale	 = es->gate_scale;
		experts[i].up_scale		 = es->up_scale;
		experts[i].down_scale	 = es->down_scale;
		for (int c = 0; c < cnt; c++) {
			rows_packed[packed]	   = rows[c];
			weights_packed[packed] = ctx->bs->moe_router_w[(size_t)rows[c] * K + kidx[c]];
			packed++;
		}
	}

	status_code st = a->scale_inplace(a, out_b, 0.0f, n_rows * dim);
	if (st == OK)
		st = a->moe_experts_batch(a, xb_dev, out_b, n_rows, dim, I, use_gelu, n_union, experts,
								  counts, rows_packed, weights_packed);

	return st;
}

static void moe_experts_batch_split(exec_ctx *ctx, backend *a, int dim, int K, int I,
									int use_gelu) {
	batch_scratch *bs	   = ctx->bs;
	int			   n_rows  = ctx->rows.n_rows;
	int			   n_union = bs->moe_n_union;
	if (n_union <= 0)
		return;

	moe_pack  pack	 = moe_pack_get(bs, n_union, n_rows, K);
	backend **owners = pack.owners;
	int		  any	 = 0;
	for (int i = 0; i < n_union; i++) {
		int				 u	= bs->moe_union_order[i];
		moe_expert_slot *es = &bs->moe_union_slots[i];
		if (bs->moe_union_count[u] > 0 && es->eid >= 0 && es->dev_ready) {
			owners[i] = es->dev_gate.owner;
			any |= owners[i] != NULL;
		}
	}
	size_t total = (size_t)n_rows * dim;
	float *xh	 = float_buf_ensure(&bs->moe_xb_f, total);
	if (!any || a->buffer_read_f32(a, batch_slot(bs, ctx->op->in[0]), xh, (int)total) != OK) {
		return;
	}

	moe_resident_expert *experts		= pack.experts;
	int					*counts			= pack.counts;
	int					*idx			= pack.idx;
	int					*rows_packed	= pack.rows_packed;
	float				*weights_packed = pack.weights_packed;

	for (int i0 = 0; i0 < n_union; i0++) {
		backend *eb = owners[i0];
		if (!eb)
			continue;
		int n_e	   = 0;
		int packed = 0;
		for (int i = i0; i < n_union; i++) {
			if (owners[i] != eb)
				continue;
			int				 u		   = bs->moe_union_order[i];
			int				 cnt	   = bs->moe_union_count[u];
			const int		*rows	   = bs->moe_union_rows + bs->moe_union_offsets[u];
			const int		*kidx	   = bs->moe_union_kidx + bs->moe_union_offsets[u];
			moe_expert_slot *es		   = &bs->moe_union_slots[i];
			idx[n_e]				   = i;
			counts[n_e]				   = cnt;
			experts[n_e].gate_w		   = &es->dev_gate;
			experts[n_e].up_w		   = &es->dev_up;
			experts[n_e].down_w		   = &es->dev_down;
			experts[n_e].gate_type	   = es->gate_type;
			experts[n_e].up_type	   = es->up_type;
			experts[n_e].down_type	   = es->down_type;
			experts[n_e].gate_up_fused = es->gate_up_fused;
			experts[n_e].use_gelu	   = use_gelu;
			experts[n_e].gate_scale	   = es->gate_scale;
			experts[n_e].up_scale	   = es->up_scale;
			experts[n_e].down_scale	   = es->down_scale;
			for (int c = 0; c < cnt; c++) {
				rows_packed[packed]	   = rows[c];
				weights_packed[packed] = bs->moe_router_w[(size_t)rows[c] * K + kidx[c]];
				packed++;
			}
			n_e++;
		}

		status_code		  st = OK;
		expert_dev_entry *e	 = NULL;
		if (!eb->moe_experts_batch || n_e > MOE_MAX_TOPK * 8)
			st = ERR_UNSUPPORTED;
		else if (!(e = expert_dev_scratch_get(&ctx->s->moe.expert_dev, eb, total)))
			st = ERR_OUT_OF_MEMORY;
		if (st == OK)
			st = eb->buffer_write_f32(eb, &e->x, xh, (int)total);
		if (st == OK)
			st = eb->scale_inplace(eb, &e->out, 0.0f, (int)total);
		if (st == OK)
			st = eb->moe_experts_batch(eb, &e->x, &e->out, n_rows, dim, I, use_gelu, n_e, experts,
									   counts, rows_packed, weights_packed);
		if (st == OK) {
			float *tmp = float_buf_ensure(&ctx->s->moe.expert_dev.tmp, total);
			st		   = eb->buffer_read_f32(eb, &e->out, tmp, (int)total);
			if (st == OK)
				for (size_t j = 0; j < total; j++)
					bs->moe_out.p[j] += tmp[j];
		}
		if (st == OK) {
			for (int j = 0; j < n_e; j++)
				bs->moe_union_count[bs->moe_union_order[idx[j]]] = 0;
		} else {
			WARN("moe expert batch on '%s' failed (st=%d) -- falling back to cpu",
				 eb->name ? eb->name : "?", (int)st);
		}
		for (int i = i0; i < n_union; i++)
			if (owners[i] == eb)
				owners[i] = NULL;
	}
}

static void moe_release_batch_slots(model *m, int li, batch_scratch *bs, int n_rows, int K,
									int n_union) {
	for (int i = 0; i < n_union; i++) {
		moe_expert_slot *us = &bs->moe_union_slots[i];
		if (us->owned && us->heap_buf) {
			free(us->heap_buf);
			us->heap_buf = NULL;
			us->eid		 = -1;
		}
	}
	moe_stream_release_slots(m, li, bs->moe_union_slots, n_union);

	for (int i = 0; i < n_union; i++)
		bs->moe_union_slots[i].eid = -1;
	for (int i = 0; i < n_rows * K; i++) {
		moe_expert_slot *es = &bs->moe_per_token_slots[i];
		es->eid				= -1;
		es->owned			= 0;
		es->heap_buf		= NULL;
		es->gate_w			= NULL;
		es->up_w			= NULL;
		es->down_w			= NULL;
	}
}

static void moe_exec_plan_fill(moe_exec_plan *p, struct model *m) {
	p->dim		= m->dim;
	p->K		= m->moe.n_experts_used;
	p->I		= m->moe.moe_intermediate;
	p->use_gelu = m->arch_info->uses_gelu_activation;
}

status_code moe_experts_batch(exec_ctx *ctx) {

	backend	 *a	  = exec_layer_backend(ctx);
	const int dim = ctx->m->dim;
	if (!model_layer_is_moe(ctx->m, ctx->li)) {
		buffer *dst	  = batch_slot(ctx->bs, ctx->op->out);
		size_t	need  = (size_t)ctx->rows.n_rows * dim;
		size_t	old	  = ctx->bs->moe_zeros.cap;
		float  *zeros = float_buf_ensure(&ctx->bs->moe_zeros, need);
		if (need > old)
			memset(zeros + old, 0, (need - old) * sizeof(float));
		return a->buffer_write_f32(a, dst, zeros, ctx->rows.n_rows * dim);
	}

	moe_exec_plan plan;
	moe_exec_plan_fill(&plan, ctx->m);
	int K		 = plan.K;
	int I		 = plan.I;
	int use_gelu = plan.use_gelu;

	int n_union = ctx->bs->moe_union_pending_n;
	for (int i = 0; i < ctx->rows.n_rows * K; i++)
		ctx->bs->moe_per_token_slots[i].eid = -1;
	for (int i = 0; i < n_union; i++) {
		int		   u	= ctx->bs->moe_union_order[i];
		int		   cnt	= ctx->bs->moe_union_count[u];
		const int *rows = ctx->bs->moe_union_rows + ctx->bs->moe_union_offsets[u];
		const int *kidx = ctx->bs->moe_union_kidx + ctx->bs->moe_union_offsets[u];
		for (int c = 0; c < cnt; c++) {
			moe_expert_slot slot = ctx->bs->moe_union_slots[i];
			if (c > 0) {
				slot.owned	  = 0;
				slot.heap_buf = NULL;
			}
			ctx->bs->moe_per_token_slots[(size_t)rows[c] * K + kidx[c]] = slot;
		}
		if (cnt > 1)
			moe_stream_record_extra_hits(ctx->m, ctx->li, cnt - 1, cnt - 1,
										 ctx->bs->moe_union_slots[i].pinned);
	}

	for (int row = 0; row < ctx->rows.n_rows; row++) {
		const int	*row_ids = ctx->bs->moe_router_ids + (size_t)row * K;
		const float *row_w	 = ctx->bs->moe_router_w + (size_t)row * K;
		monitor_emit_moe_experts(g_monitor, ctx->li, -1, row_ids, row_w, K);
	}

	float_buf_ensure(&ctx->bs->moe_out, (size_t)ctx->rows.n_rows * dim);
	memset(ctx->bs->moe_out.p, 0, ctx->rows.n_rows * dim * sizeof(float));

	status_code st = OK;
	if (ctx->rows.n_rows > 1 && a->matmul_batch) {
		if (backend_has_cap(a, BCAP_MOE_EXPERT_RESIDENT) && a->moe_experts_batch &&
			ctx->m->layers[ctx->li].experts_resident) {
			st = moe_experts_resident_grouped(ctx, a, dim, K, I, use_gelu);
			if (st == OK) {
				moe_release_batch_slots(ctx->m, ctx->li, ctx->bs, ctx->rows.n_rows, K, n_union);
				return OK;
			}
			if (st != ERR_UNSUPPORTED)
				WARN("moe resident expert batch failed (st=%d) -- falling back to cpu", (int)st);
		}
		if (ctx->m->layers[ctx->li].experts_resident)
			moe_experts_batch_split(ctx, a, dim, K, I, use_gelu);
		moe_experts_grouped(ctx, a, dim, K, I, use_gelu);
		goto finish;
	}

	tpool *pool			  = model_get_pool(ctx->m);
	int	   total_experts  = ctx->rows.n_rows * K;
	size_t per_expert_out = (size_t)ctx->rows.n_rows * dim;

	if (pool && total_experts >= 2 &&
		(size_t)total_experts * per_expert_out * sizeof(float) <= (size_t)256 * 1024 * 1024) {
		int n_threads = tpool_n_threads(pool);
		if (n_threads < 1)
			n_threads = 1;
		if (n_threads > total_experts)
			n_threads = total_experts;

		size_t per_thread_scratch = (size_t)I * 3 + (size_t)dim + (size_t)I * 2;
		float *all_scratch =
			float_buf_ensure(&ctx->s->moe.all_scratch, (size_t)n_threads * per_thread_scratch);
		float *all_outs =
			float_buf_ensure(&ctx->s->moe.all_outs, (size_t)total_experts * per_expert_out);
		memset(all_outs, 0, (size_t)total_experts * per_expert_out * sizeof(float));

		moe_batch_job job;
		job.m		 = ctx->m;
		job.a		 = backend_host();
		job.bs		 = ctx->bs;
		job.slot_buf = ctx->bs->moe_per_token_slots;
		job.li		 = ctx->li;
		job.top_k	 = K;
		job.inter	 = I;
		job.dim		 = dim;
		job.n_tokens = ctx->rows.n_rows;
		job.use_gelu = use_gelu;
		{
			buffer *xb_b = batch_slot(ctx->bs, ctx->op->in[0]);
			float  *xb_h = float_buf_ensure(&ctx->bs->moe_xb_f, (size_t)ctx->rows.n_rows * dim);
			st			 = a->buffer_read_f32(a, xb_b, xb_h, ctx->rows.n_rows * dim);
			if (st != OK)
				return st;
			job.xb_base = xb_h;
		}
		job.all_scratch		   = all_scratch;
		job.all_outs		   = all_outs;
		job.per_thread_scratch = per_thread_scratch;
		job.per_expert_out	   = per_expert_out;

		tpool_parallel_for(pool, total_experts, 1, moe_batch_expert_chunk, &job);

		for (int idx = 0; idx < total_experts; idx++) {
			const float *o = all_outs + (size_t)idx * per_expert_out;
			for (int i = 0; i < (int)per_expert_out; i++)
				ctx->bs->moe_out.p[i] += o[i];
		}
	} else {
		float *scratch =
			float_buf_ensure(&ctx->s->moe.scratch, ((size_t)I * 3 + (size_t)dim + (size_t)I * 2));
		float *gate_h = scratch;
		float *up_h	  = gate_h + I;
		float *act_h  = up_h + I;
		float *y_h	  = act_h + I;
		float *gu_h	  = y_h + dim;

		buffer *xb_b	= batch_slot(ctx->bs, ctx->op->in[0]);
		float  *xb_host = float_buf_ensure(&ctx->bs->moe_xb_f, (size_t)ctx->rows.n_rows * dim);
		st				= a->buffer_read_f32(a, xb_b, xb_host, ctx->rows.n_rows * dim);
		if (st != OK)
			return st;

		for (int row = 0; row < ctx->rows.n_rows; row++) {
			const float *xb_row = xb_host + (size_t)row * dim;
			for (int k = 0; k < K; k++) {
				moe_expert_slot *es = &ctx->bs->moe_per_token_slots[(size_t)row * K + k];
				if (es->eid < 0 || !es->gate_w)
					continue;

				if (es->gate_up_fused) {
					host_matmul_generic(es->gate_w, es->gate_type, xb_row, gu_h, I * 2, dim);
					st = moe_activate(backend_host(), act_h, gu_h, gu_h + I, I, es->gate_scale,
									  es->up_scale, use_gelu);
				} else {
					host_matmul_generic(es->gate_w, es->gate_type, xb_row, gate_h, I, dim);
					host_matmul_generic(es->up_w, es->up_type, xb_row, up_h, I, dim);
					st = moe_activate(backend_host(), act_h, gate_h, up_h, I, es->gate_scale,
									  es->up_scale, use_gelu);
				}
				if (st != OK)
					return st;

				host_matmul_generic(es->down_w, es->down_type, act_h, y_h, dim, I);
				if (es->down_scale != 1.0f) {
					float ds = es->down_scale;
					for (int d = 0; d < dim; d++)
						y_h[d] *= ds;
				}

				float  w	   = ctx->bs->moe_router_w[(size_t)row * K + k];
				float *out_row = ctx->bs->moe_out.p + (size_t)row * dim;
				for (int d = 0; d < dim; d++)
					out_row[d] += w * y_h[d];
			}
		}
	}

finish:;
	buffer *xb2 = batch_slot(ctx->bs, ctx->op->out);
	st			= a->buffer_write_f32(a, xb2, ctx->bs->moe_out.p, ctx->rows.n_rows * dim);
	if (st != OK)
		return st;

	moe_release_batch_slots(ctx->m, ctx->li, ctx->bs, ctx->rows.n_rows, K, n_union);
	return st;
}

static int moe_shared_plan_fill(moe_shared_plan *p, struct model *m, int li) {
	int is_moe = model_layer_is_moe(m, li);
	if (is_moe && m->moe.n_shared_experts == 0)
		return 0;
	layer_weights *L = &m->layers[li];
	p->is_moe		 = is_moe;
	p->sh_inter = is_moe ? (m->moe.moe_intermediate * m->moe.n_shared_experts) : m->intermediate;
	p->use_gelu = m->arch_info->uses_gelu_activation;
	p->gate_up_fused = (!is_moe && L->gate_up_fused) ? 1 : 0;
	if (is_moe) {
		p->gate_w		= &L->shexp_gate_w.buf;
		p->gate_type	= L->shexp_gate_w.type;
		p->up_w			= &L->shexp_up_w.buf;
		p->up_type		= L->shexp_up_w.type;
		p->down_w		= &L->shexp_down_w.buf;
		p->down_type	= L->shexp_down_w.type;
		p->gate_up_w	= NULL;
		p->gate_up_type = 0;
	} else {
		p->gate_w		= &L->gate_w.buf;
		p->gate_type	= L->gate_w.type;
		p->up_w			= &L->up_w.buf;
		p->up_type		= L->up_w.type;
		p->down_w		= &L->down_w.buf;
		p->down_type	= L->down_w.type;
		p->gate_up_w	= &L->gate_up_w.buf;
		p->gate_up_type = L->gate_up_w.type;
	}
	return 1;
}

status_code moe_shared_batch(exec_ctx *ctx) {

	backend		   *a	 = exec_layer_backend(ctx);
	const uint32_t	caps = exec_caps(ctx, moe_shared_caps);
	const int		dim	 = ctx->m->dim;
	moe_shared_plan plan;
	if (!moe_shared_plan_fill(&plan, ctx->m, ctx->li))
		return OK;
	int sh_inter = plan.sh_inter;

	buffer *xb = batch_slot(ctx->bs, ctx->op->in[0]);

	const buffer *gate_w_buf = plan.gate_w;
	uint32_t	  gate_wt	 = plan.gate_type;
	const buffer *up_w_buf	 = plan.up_w;
	uint32_t	  up_wt		 = plan.up_type;
	const buffer *down_w_buf = plan.down_w;
	uint32_t	  down_wt	 = plan.down_type;

	if (plan.gate_up_fused) {
		bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE_UP,
					   (size_t)ctx->rows.n_rows * 2 * sh_inter);

		status_code st = a->matmul_batch(a, plan.gate_up_w, plan.gate_up_type, xb,
										 &ctx->bs->pair[RECIPE_SLOT_FFN_GATE_UP].b, 2 * sh_inter,
										 dim, ctx->rows.n_rows);
		if (st != OK)
			return st;

		int use_gelu = plan.use_gelu;
		int n_rows	 = ctx->rows.n_rows;
		if (caps & MOE_SH_FUSED_ACT) {
			bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_ACT, (size_t)n_rows * sh_inter);
			status_code ast = a->ffn_activate_fused_batch(
				a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE_UP].b, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b,
				sh_inter, use_gelu ? 1 : 0, n_rows);
			if (ast == OK)
				goto down_batch;
			if (ast != ERR_UNSUPPORTED)
				return ast;
		}
		float *gu_host	= float_buf_ensure(&ctx->bs->moe_exp_gu, (size_t)n_rows * 2 * sh_inter);
		float *act_host = float_buf_ensure(&ctx->bs->moe_exp_act, (size_t)n_rows * sh_inter);
		st				= a->buffer_read_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE_UP].b, gu_host,
											 n_rows * 2 * sh_inter);
		if (st != OK)
			return st;
		for (int row = 0; row < n_rows; row++) {
			const float *gu = gu_host + (size_t)row * 2 * sh_inter;
			float		*o	= act_host + (size_t)row * sh_inter;
			st				= moe_activate(a, o, gu, gu + sh_inter, sh_inter, 1.0f, 1.0f, use_gelu);
			if (st != OK)
				return st;
		}
		st = a->buffer_write_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, act_host,
								 n_rows * sh_inter);
		if (st != OK)
			return st;
	} else {
		bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE, (size_t)ctx->rows.n_rows * sh_inter);
		bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_UP, (size_t)ctx->rows.n_rows * sh_inter);

		status_code st;
		if (caps & MOE_SH_MULTI) {
			const buffer *ws[2]	   = {gate_w_buf, up_w_buf};
			uint32_t	  wts[2]   = {gate_wt, up_wt};
			buffer		 *ys[2]	   = {&ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b,
									  &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b};
			int			  n_out[2] = {sh_inter, sh_inter};
			st = a->matmul_multi_batch(a, ws, wts, xb, ys, n_out, dim, 2, ctx->rows.n_rows);
		} else {
			st = a->matmul_batch(a, gate_w_buf, gate_wt, xb, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b,
								 sh_inter, dim, ctx->rows.n_rows);
			if (st == OK)
				st = a->matmul_batch(a, up_w_buf, up_wt, xb, &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b,
									 sh_inter, dim, ctx->rows.n_rows);
		}
		if (st != OK)
			return st;

		if (caps & MOE_SH_ACT) {
			int n_rows2 = ctx->rows.n_rows;
			bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_ACT, (size_t)n_rows2 * sh_inter);
			status_code ast = a->ffn_activate_batch(
				a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b, &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b,
				&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, sh_inter, plan.use_gelu ? 1 : 0, n_rows2);
			if (ast == OK)
				goto down_batch;
			if (ast != ERR_UNSUPPORTED)
				return ast;
		}

		int	   use_gelu = plan.use_gelu;
		int	   n_rows2	= ctx->rows.n_rows;
		float *g_host	= float_buf_ensure(&ctx->bs->moe_exp_gu, (size_t)n_rows2 * sh_inter);
		float *u_host	= float_buf_ensure(&ctx->bs->moe_up_scratch, (size_t)n_rows2 * sh_inter);
		float *act_host = float_buf_ensure(&ctx->bs->moe_exp_act, (size_t)n_rows2 * sh_inter);
		st				= a->buffer_read_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b, g_host,
											 n_rows2 * sh_inter);
		if (st == OK)
			st = a->buffer_read_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b, u_host,
									n_rows2 * sh_inter);
		if (st != OK)
			return st;
		for (int row = 0; row < n_rows2; row++) {
			const float *g = g_host + (size_t)row * sh_inter;
			const float *u = u_host + (size_t)row * sh_inter;
			float		*o = act_host + (size_t)row * sh_inter;
			st			   = moe_activate(a, o, g, u, sh_inter, 1.0f, 1.0f, use_gelu);
			if (st != OK)
				return st;
		}
		st = a->buffer_write_f32(a, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, act_host,
								 n_rows2 * sh_inter);
		if (st != OK)
			return st;
	}

down_batch:
	ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b.size = (size_t)ctx->rows.n_rows * sh_inter * sizeof(float);

	status_code st =
		a->matmul_batch(a, down_w_buf, down_wt, &ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b,
						&ctx->bs->pair[RECIPE_SLOT_RESID_TMP].b, dim, sh_inter, ctx->rows.n_rows);
	if (st != OK)
		return st;

	if (caps & MOE_SH_COPY) {
		st = a->copy_buffer(a, &ctx->bs->pair[RECIPE_SLOT_RESID_TMP].b,
							&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b, ctx->rows.n_rows * dim);
		if (st != OK)
			return st;
	} else {
		batch_sync(ctx);
		memcpy(batch_buf_ptr(&ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b),
			   batch_buf_ptr(&ctx->bs->pair[RECIPE_SLOT_RESID_TMP].b),
			   (size_t)ctx->rows.n_rows * dim * sizeof(float));
	}
	ctx->bs->pair[RECIPE_SLOT_FFN_ACT].b.size = (size_t)ctx->rows.n_rows * dim * sizeof(float);
	return OK;
}

static void moe_batch_expert_chunk(int begin, int end, int tid, void *ctx) {
	moe_batch_job *j	   = ctx;
	float		  *scratch = j->all_scratch + (size_t)tid * j->per_thread_scratch;

	moe_expert_ctx cx = {
		.a		  = j->a,
		.inter	  = j->inter,
		.dim	  = j->dim,
		.use_gelu = j->use_gelu,
		.scratch  = scratch,
	};

	for (int idx = begin; idx < end; idx++) {
		int row = idx / j->top_k;
		int k	= idx % j->top_k;

		moe_expert_slot *es = &j->slot_buf[(size_t)row * j->top_k + k];
		if (es->eid < 0 || !es->gate_w)
			continue;

		const float *xb_row = j->xb_base + (size_t)row * j->dim;
		float		*w		= &j->bs->moe_router_w[(size_t)row * j->top_k + k];
		cx.out				= j->all_outs + (size_t)idx * j->per_expert_out + (size_t)row * j->dim;
		(void)moe_expert_exec(&cx, es, xb_row, *w, tid);
	}
}
