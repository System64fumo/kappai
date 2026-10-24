#include "recipe/internal.h"

_Atomic(backend *) cached_host_backend = NULL;

static status_code batch_copy_k_to_v(model *m, batch_scratch *bs, int li, int n_tokens,
									 uint8_t k_slot, uint8_t v_slot);

static status_code ffn_activate_fused_device_fallback(exec_ctx *ctx, backend *a, int n, int act,
													  int fused_stride, int n_rows);

const float *recipe_slot_read_f32(const exec_ctx *ctx, uint8_t idx, float_buf *stage, int n) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return NULL;
	if (buffer_is_host_resident(b))
		return (const float *)(b->handle ? (char *)b->handle + b->offset : b->host_ptr);
	float *dst = float_buf_ensure_nocopy(stage, (size_t)n, 64);
	if (!dst || b->owner->buffer_read_f32(b->owner, b, dst, n) != OK)
		return NULL;
	return dst;
}

float *recipe_slot_write_stage(const exec_ctx *ctx, uint8_t idx, float_buf *stage, int n) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return NULL;
	if (buffer_is_host_resident(b))
		return (float *)(b->handle ? (char *)b->handle + b->offset : b->host_ptr);
	return float_buf_ensure_nocopy(stage, (size_t)n, 64);
}

status_code recipe_slot_write_commit(const exec_ctx *ctx, uint8_t idx, const float *staged, int n) {
	buffer *b = exec_slot(ctx, idx);
	if (!b)
		return ERR_INVALID_ARG;
	if (buffer_is_host_resident(b)) {
		float *dst = (float *)(b->handle ? (char *)b->handle + b->offset : (char *)b->host_ptr);
		if (dst && (const float *)dst != staged)
			memcpy(dst, staged, (size_t)n * sizeof(float));
		return OK;
	}
	return b->owner->buffer_write_f32(b->owner, b, staged, n);
}

static inline const layer_ctx_entry *exec_layer_ctx(const exec_ctx *ctx) {
	return ctx->lc ? ctx->lc : &ctx->m->recipe->layer_ctx[ctx->li];
}

static inline weight_ref *exec_weight_ref(const exec_ctx *ctx, uint8_t w_idx) {
	if (w_idx == RECIPE_NO_WEIGHT || w_idx == WIDX_NONE)
		return (weight_ref *)&WEIGHT_REF_NONE;
	if (ctx->wtab_row) {
		weight_ref *w = ctx->wtab_row[w_idx];
		return w ? w : (weight_ref *)&WEIGHT_REF_NONE;
	}
	return resolve_weight_ref(ctx->m, ctx->li, w_idx);
}

static inline const buffer *exec_weight(const exec_ctx *ctx, uint8_t w_idx) {
	return &exec_weight_ref(ctx, w_idx)->buf;
}

static status_code copy_k_to_v_buffers(backend *a, buffer *kb, buffer *vb, int kv_out,
									   int kv_stride, int n_rows, float_buf *stage) {
	if (a->copy_buffer && kb->owner == vb->owner) {
		for (int row = 0; row < n_rows; row++) {
			buffer		krow = batch_row_view(kb, row, kv_stride);
			buffer		vrow = batch_row_view(vb, row, kv_stride);
			status_code st	 = a->copy_buffer(a, &krow, &vrow, kv_out);
			if (st != OK)
				return st;
		}
		return OK;
	}
	ensure_sync(a);
	if (!kb->host_ptr || !vb->host_ptr) {
		float_buf_ensure_nocopy(stage, (size_t)kv_out, 64);
		float	   *tmp = stage->p;
		status_code st	= OK;
		for (int row = 0; row < n_rows && st == OK; row++) {
			buffer krow = batch_row_view(kb, row, kv_stride);
			buffer vrow = batch_row_view(vb, row, kv_stride);
			st			= kb->owner->buffer_read_f32(kb->owner, &krow, tmp, kv_out);
			if (st == OK)
				st = vb->owner->buffer_write_f32(vb->owner, &vrow, tmp, kv_out);
		}
		return st;
	}
	for (int row = 0; row < n_rows; row++) {
		const float *k =
			(const float *)((const char *)kb->host_ptr + kb->offset) + (size_t)row * kv_stride;
		float *v = (float *)((char *)vb->host_ptr + vb->offset) + (size_t)row * kv_stride;
		memcpy(v, k, (size_t)kv_out * sizeof(float));
	}
	return OK;
}

void profile_weight_bytes(profile *prof, stage s, uint32_t w_type, int rows, int k) {
	if (rows <= 0 || k <= 0)
		return;
	size_t rb = ggml_row_size(w_type, (size_t)k);
	profile_add_bytes(prof, s, (uint64_t)rb * (uint64_t)rows);
}

static int exec_matmul_run_coalesced_impl(const recipe_op *ops, int n_ops, model *m,
										  compute_scratch *s, int li, buffer *xb, buffer *ys[],
										  int batched, int n_rows, status_code *out_status) {
	backend *a = m->backend;
	if ((batched ? !a->matmul_multi_batch : !a->matmul_multi) || n_ops < 2) {
		*out_status = OK;
		return 0;
	}

	int n = n_ops > RECIPE_COALESCE_MAX ? RECIPE_COALESCE_MAX : n_ops;
	for (int i = 0; i < n; i++) {
		if (ops[i].w_idx == WIDX_WV && li >= 0 && !model_layer_has_own_v(m, li)) {
			n = i;
			break;
		}
	}
	if (n < 2) {
		*out_status = OK;
		return 0;
	}
	const buffer *w_list[RECIPE_COALESCE_MAX];
	uint32_t	  w_types[RECIPE_COALESCE_MAX];
	buffer		 *y_list[RECIPE_COALESCE_MAX];
	int			  n_out[RECIPE_COALESCE_MAX];
	int			  k = ops[0].u.matmul.k;

	for (int i = 0; i < n; i++) {
		weight_ref *w = resolve_weight_ref(m, li, ops[i].w_idx);
		w_list[i]	  = &w->buf;
		w_types[i]	  = w->type;
		y_list[i]	  = ys[i];
		n_out[i]	  = ops[i].u.matmul.n;
	}

	profile		 *prof = &s->prof;
	profile_scope ps   = profile_begin(prof, ops[0].stage);
	for (int i = 0; i < n; i++)
		profile_weight_bytes(prof, ops[0].stage, w_types[i], n_out[i], k);
	status_code st =
		batched ? a->matmul_multi_batch(a, w_list, w_types, xb, y_list, n_out, k, n, n_rows)
				: a->matmul_multi(a, w_list, w_types, xb, y_list, n_out, k, n);
	profile_end(prof, &ps);

	*out_status = st;
	return n;
}

int exec_matmul_run_coalesced_batch(const recipe_op *ops, int n_ops, model *m, compute_scratch *s,
									int li, buffer *xb, buffer *ys[], int n_rows,
									status_code *out_status) {
	return exec_matmul_run_coalesced_impl(ops, n_ops, m, s, li, xb, ys, 1, n_rows, out_status);
}

int exec_matmul_run_qonly(const recipe_op *ops, int n_ops, model *m, buffer *xb, buffer **y_list,
						  int li, int n_rows, status_code *out_status) {
	backend *a	= m->backend;
	*out_status = OK;
	if (n_ops < 2 || !backend_has_cap(a, BCAP_MATMUL_QONLY) || !a->prequantize_x ||
		!a->matmul_qonly)
		return 0;

	int n = n_ops > RECIPE_COALESCE_MAX ? RECIPE_COALESCE_MAX : n_ops;
	for (int i = 0; i < n; i++) {
		if (ops[i].w_idx == WIDX_WV && li >= 0 && !model_layer_has_own_v(m, li)) {
			n = i;
			break;
		}
	}
	if (n < 2)
		return 0;

	int		 k	= ops[0].u.matmul.k;
	uint32_t q8 = wtype_to_q8type(resolve_weight_ref(m, li, ops[0].w_idx)->type);
	if (!q8)
		return 0;
	for (int i = 1; i < n; i++) {
		if (ops[i].u.matmul.k != k ||
			wtype_to_q8type(resolve_weight_ref(m, li, ops[i].w_idx)->type) != q8) {
			n = i;
			break;
		}
	}
	if (n < 2)
		return 0;

	buffer		xq = {0};
	status_code st = a->prequantize_x(a, xb, k, q8, &xq);
	if (st != OK) {
		if (st == ERR_UNSUPPORTED || st == ERR_INVALID_ARG) {
			*out_status = OK;
			return 0;
		}
		*out_status = st;
		return -1;
	}

	for (int i = 0; i < n; i++) {
		weight_ref *w = resolve_weight_ref(m, li, ops[i].w_idx);
		status_code ost =
			a->matmul_qonly(a, &w->buf, w->type, &xq, q8, y_list[i], ops[i].u.matmul.n, k, n_rows);
		if (ost != OK) {
			*out_status = ost;
			return -1;
		}
	}
	*out_status = OK;
	return n;
}

#define OP_FALLBACK_SEEN_MAX 64

typedef struct {
	const backend *be;
	const char	  *op;
} op_fallback_seen;

static op_fallback_seen g_op_fallback_seen[OP_FALLBACK_SEEN_MAX];

static _Atomic int g_op_fallback_n_seen;

static pthread_mutex_t g_op_fallback_mtx = PTHREAD_MUTEX_INITIALIZER;

static bool op_fallback_seen_before(const backend *a, const char *op_name, int n) {
	for (int i = 0; i < n && i < OP_FALLBACK_SEEN_MAX; i++)
		if (g_op_fallback_seen[i].be == a && strcmp(g_op_fallback_seen[i].op, op_name) == 0)
			return true;
	return false;
}

void op_fallback_note(backend *a, const char *op_name) {
	if (op_fallback_seen_before(a, op_name, atomic_load(&g_op_fallback_n_seen)))
		return;
	pthread_mutex_lock(&g_op_fallback_mtx);
	int n = atomic_load(&g_op_fallback_n_seen);
	if (n >= OP_FALLBACK_SEEN_MAX || op_fallback_seen_before(a, op_name, n)) {
		pthread_mutex_unlock(&g_op_fallback_mtx);
		return;
	}
	g_op_fallback_seen[n].be = a;
	g_op_fallback_seen[n].op = op_name;
	atomic_store(&g_op_fallback_n_seen, n + 1);
	pthread_mutex_unlock(&g_op_fallback_mtx);
	backend_report_host_fallback(a, op_name, HFB_OP_NOT_NATIVE,
								 "backend has no native '%s'; running on the host fallback",
								 op_name);
}

static inline void backend_sync_rope(backend *t, backend *a) {
	if (t != a) {
		t->rope_neox  = a->rope_neox;
		t->rope_theta = a->rope_theta;
	}
}

status_code op_ffn_activate_ex(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps;
	status_code	  st;
	int			  intermediate = ctx->op->u.matmul.n;
	int			  activation   = ctx->op->u.ffn_act.activation;
	ps						   = profile_begin(prof, ctx->op->stage);
	st = a->ffn_activate_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
							   exec_slot(ctx, ctx->op->out), intermediate, activation,
							   ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st;
}

typedef struct {
	const float *cos;
	const float *sin;
	const float *freq_factors;
} rope_tables;

static rope_tables exec_rope_setup(const exec_ctx *ctx, backend *a, int rope_neox,
								   int use_freq_factors) {
	const struct model	  *m  = ctx->m;
	const layer_ctx_entry *lc = exec_layer_ctx(ctx);
	a->rope_neox			  = rope_neox;
	a->rope_theta = lc->is_global ? m->layer_dims.rope_theta_global : m->layer_dims.rope_theta_swa;
	if (lc->is_global)
		return (rope_tables){ctx->s->model.rope_cos, ctx->s->model.rope_sin,
							 use_freq_factors ? m->rope_freqs : NULL};
	return (rope_tables){ctx->s->model.rope_cos_swa, ctx->s->model.rope_sin_swa, NULL};
}

typedef struct {
	int			n_heads;
	int			head_dim;
	int			row_stride;
	rope_tables rt;
} rope_ext_args;

static rope_ext_args rope_ext_args_of(exec_ctx *ctx, backend *a) {
	const recipe_op		  *op	= ctx->op;
	const layer_ctx_entry *lc	= exec_layer_ctx(ctx);
	int					   is_q = op->u.rope_ext.is_q;
	rope_ext_args		   g	= {.n_heads	   = op->u.rope_ext.n_heads,
								   .head_dim   = lc->head_dim,
								   .row_stride = is_q ? lc->q_row_stride : lc->kv_row_stride};
	if (g.n_heads == 0)
		g.n_heads = is_q ? ctx->m->n_heads : lc->n_kv_heads;
	g.rt = exec_rope_setup(ctx, a, op->u.rope_ext.rope_neox, op->u.rope_ext.use_freq_factors);
	return g;
}

static status_code rope_ext_batched(exec_ctx *ctx, backend *a) {
	if (ctx->li < 0)
		return ERR_INVALID_ARG;
	rope_ext_args g		= rope_ext_args_of(ctx, a);
	buffer		 *whole = batch_slot(ctx->bs, ctx->op->in[0]);
	exec_segments seg	= exec_segments_of(ctx);
	status_code	  st	= OK;
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer view = batch_row_view(whole, c * seg.rows, g.row_stride);
		st = a->rope_ext_batch(a, &view, g.n_heads, g.head_dim, exec_segment_pos(ctx, c), g.rt.cos,
							   g.rt.sin, g.rt.freq_factors, seg.rows);
	}
	return st;
}

static status_code rope_ext_rows(exec_ctx *ctx, backend *a) {
	if (ctx->li < 0)
		return ERR_INVALID_ARG;
	rope_ext_args g = rope_ext_args_of(ctx, a);
	backend		 *t = a->rope_ext ? a : OP_BACKEND(a, rope);
	backend_sync_rope(t, a);
	backend *t2 = NULL;
	if (!t->rope_ext) {
		t2 = OP_BACKEND(a, rope);
		backend_sync_rope(t2, a);
	}
	status_code st = OK;
	for (int r = 0; r < ctx->rows.n_rows && st == OK; r++) {
		int	   row_pos = exec_row_pos(ctx, r);
		buffer vec	   = batch_row_view(batch_slot(ctx->bs, ctx->op->in[0]), r, g.row_stride);
		st = t->rope_ext ? t->rope_ext(t, &vec, g.n_heads, g.head_dim, row_pos, g.rt.cos, g.rt.sin,
									   g.rt.freq_factors)
						 : t2->rope(t2, &vec, g.n_heads, g.head_dim, row_pos, g.rt.cos, g.rt.sin);
	}
	return st;
}

static status_code rope_ext_dynamic(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	return a->rope_ext_batch && a->rope_ext ? rope_ext_batched(ctx, a) : rope_ext_rows(ctx, a);
}

status_code op_rope_ext_batched(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rope_ext_batched(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_rope_ext(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rope_ext_dynamic(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code qk_norm_rope_unfused(exec_ctx *ctx, const layer_ctx_entry *lc) {
	const float eps		 = ctx->op->u.qk_norm_rope.eps;
	const int	neox	 = ctx->op->u.qk_norm_rope.rope_neox;
	recipe_op	steps[4] = {
		mk_rmsnorm_per_head(ctx->op->in[0], WIDX_ATTN_Q_NORM, eps, 0),
		mk_rmsnorm_per_head(ctx->op->in[1], WIDX_ATTN_K_NORM, eps, 0),
		mk_rope_ext(ctx->op->in[0], 1, neox),
		mk_rope_ext(ctx->op->in[1], 0, neox),
	};
	exec_ctx	sub = *ctx;
	status_code st	= OK;
	for (int i = 0; i < 4 && st == OK; i++) {
		if (!lc->has_kv && (i == 1 || i == 3))
			continue;
		sub.op = &steps[i];
		st	   = i < 2 ? op_rmsnorm_per_head(&sub) : op_rope_ext(&sub);
	}
	return st;
}

static status_code qk_norm_rope_fused(exec_ctx *ctx) {
	backend				  *a  = exec_layer_backend(ctx);
	const layer_ctx_entry *lc = exec_layer_ctx(ctx);
	profile_scope		   ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	rope_tables			   rt = exec_rope_setup(ctx, a, ctx->op->u.qk_norm_rope.rope_neox, 1);
	const int			   n_kv_heads = lc->has_kv ? lc->n_kv_heads : 0;
	exec_segments		   seg		  = exec_segments_of(ctx);
	buffer				  *q_all	  = exec_slot(ctx, ctx->op->in[0]);
	buffer				  *k_all	  = exec_slot(ctx, ctx->op->in[1]);
	status_code			   st		  = OK;
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer q_view = batch_row_view(q_all, c * seg.rows, lc->q_row_stride);
		buffer k_view = {0};
		if (n_kv_heads)
			k_view = batch_row_view(k_all, c * seg.rows, lc->kv_row_stride);
		st = a->qk_norm_rope(a, &q_view, n_kv_heads ? &k_view : NULL,
							 exec_weight(ctx, WIDX_ATTN_Q_NORM), exec_weight(ctx, WIDX_ATTN_K_NORM),
							 ctx->m->n_heads, n_kv_heads, lc->head_dim, ctx->op->u.qk_norm_rope.eps,
							 exec_segment_pos(ctx, c), rt.cos, rt.sin, rt.freq_factors, seg.rows);
		if (st == ERR_UNSUPPORTED && c == 0) {
			profile_end(&ctx->s->prof, &ps);
			return qk_norm_rope_unfused(ctx, lc);
		}
	}
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_qk_norm_rope_unfused(exec_ctx *ctx) {
	return qk_norm_rope_unfused(ctx, exec_layer_ctx(ctx));
}

status_code op_qk_norm_rope(exec_ctx *ctx) {
	return exec_layer_backend(ctx)->qk_norm_rope ? qk_norm_rope_fused(ctx)
												 : op_qk_norm_rope_unfused(ctx);
}

status_code scale_inplace_fallback(struct compute_scratch *s, backend *sync_be, buffer *xb, int n,
								   float scale) {
	ensure_sync(sync_be);
	compute_small_host_ensure(s, n, 0, 0);
	float	   *xf	  = s->run.xf_host;
	backend	   *owner = xb->owner;
	status_code st	  = owner->buffer_read_f32(owner, xb, xf, n);
	if (st == OK) {
		for (int i = 0; i < n; i++)
			xf[i] *= scale;
		st = owner->buffer_write_f32(owner, xb, xf, n);
	}
	return st;
}

status_code op_rmsnorm(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	const buffer *w	   = exec_weight(ctx, ctx->op->w_idx);
	status_code	  st;
	st = a->rmsnorm_batch(a, exec_slot(ctx, ctx->op->in[0]), w, exec_slot(ctx, ctx->op->out),
						  ctx->m->dim, ctx->op->u.rmsnorm.eps, ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st;
}

status_code op_matmul_plain(exec_ctx *ctx) {
	const recipe_op *op	  = ctx->op;
	weight_ref		*wref = exec_weight_ref(ctx, op->w_idx);
	int				 n = op->u.matmul.n, k = op->u.matmul.k;
	profile_scope	 ps = profile_begin(&ctx->s->prof, op->stage);
	profile_weight_bytes(&ctx->s->prof, op->stage, wref->type, n, k);
	backend	   *a  = exec_layer_backend(ctx);
	status_code st = a->matmul_batch(a, &wref->buf, wref->type, exec_slot(ctx, op->in[0]),
									 exec_slot(ctx, op->out), n, k, ctx->rows.n_rows);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_matmul(exec_ctx *ctx) {
	if (ctx->op->w_idx == WIDX_WV && ctx->li >= 0 && ctx->li < ctx->m->n_layers &&
		!exec_layer_ctx(ctx)->has_own_v)
		return batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->rows.n_rows, ctx->op->in[1],
								 ctx->op->out);
	return op_matmul_plain(ctx);
}

status_code op_matmul_residual(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	weight_ref	 *wref = exec_weight_ref(ctx, ctx->op->w_idx);
	const buffer *w	   = &wref->buf;
	uint32_t	  wt   = wref->type;
	int			  n	   = ctx->op->u.matmul.n;
	int			  k	   = ctx->op->u.matmul.k;
	profile_weight_bytes(prof, ctx->op->stage, wt, n, k);
	buffer *tmp = batch_slot(ctx->bs, RECIPE_SLOT_RESID_TMP);
	st = a->matmul_batch(a, w, wt, exec_slot(ctx, ctx->op->in[0]), tmp, n, k, ctx->rows.n_rows);
	if (st == OK)
		st = a->add_batch(a, exec_slot(ctx, ctx->op->out), tmp, n, ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st;
}

static status_code ffn_activate_ex_rows(backend *t, const float *fused, float *out, int n, int act,
										int n_rows, int fused_stride);

typedef struct {
	const buffer *w;
	uint32_t	  wt;
	int			  n, k, act;
} ffn_down_args;

static ffn_down_args ffn_down_args_of(exec_ctx *ctx) {
	weight_ref	 *wref = exec_weight_ref(ctx, ctx->op->w_idx);
	ffn_down_args g	   = {&wref->buf, wref->type, ctx->op->u.matmul_ffn_down.n,
						  ctx->op->u.matmul_ffn_down.k, ctx->op->u.matmul_ffn_down.activation};
	if (g.n == 0)
		g.n = ctx->m->dim;
	profile_weight_bytes(&ctx->s->prof, ctx->op->stage, g.wt, g.n, g.k);
	return g;
}

static status_code ffn_down_fused(exec_ctx *ctx, backend *a) {
	ffn_down_args g = ffn_down_args_of(ctx);
	return a->matmul_ffn_down_batch(a, g.w, g.wt, exec_slot(ctx, ctx->op->in[0]),
									exec_slot(ctx, ctx->op->in[1]), exec_slot(ctx, ctx->op->out),
									g.n, g.k, g.act, ctx->rows.n_rows);
}

static status_code ffn_down_composed(exec_ctx *ctx, backend *a) {
	ffn_down_args g	  = ffn_down_args_of(ctx);
	buffer		 *act = batch_slot(ctx->bs, RECIPE_SLOT_FFN_ACT);
	status_code	  st =
		a->ffn_activate_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
							  act, g.k, g.act, ctx->rows.n_rows);
	if (st != OK)
		return st;
	return a->matmul_batch(a, g.w, g.wt, act, exec_slot(ctx, ctx->op->out), g.n, g.k,
						   ctx->rows.n_rows);
}

status_code op_matmul_ffn_down(exec_ctx *ctx) {
	backend		 *a	 = exec_layer_backend(ctx);
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = ffn_down_native(a) ? ffn_down_fused(ctx, a) : ffn_down_composed(ctx, a);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_matmul_ffn_down_fused(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = ffn_down_fused(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_matmul_ffn_down_composed(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = ffn_down_composed(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_rope(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	int			  n_heads  = ctx->op->u.rope.n_heads;
	int			  head_dim = ctx->op->u.rope.head_dim;
	a->rope_neox		   = ctx->op->u.rope.rope_neox;
	exec_segments seg	   = exec_segments_of(ctx);
	buffer		 *whole	   = exec_slot(ctx, ctx->op->in[0]);
	st					   = OK;
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer view = batch_row_view(whole, c * seg.rows, n_heads * head_dim);
		st			= a->rope_batch(a, &view, n_heads, head_dim, exec_segment_pos(ctx, c),
									ctx->s->model.rope_cos, ctx->s->model.rope_sin, seg.rows);
	}
	profile_end(prof, &ps);
	return st;
}

status_code op_rope_qk_fused(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	int			  n_heads	 = ctx->op->u.rope.n_heads;
	int			  n_kv_heads = ctx->op->u.rope.n_kv_heads;
	int			  head_dim	 = ctx->op->u.rope.head_dim;
	a->rope_neox			 = ctx->op->u.rope.rope_neox;
	exec_segments seg		 = exec_segments_of(ctx);
	buffer		 *q_all		 = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *k_all		 = exec_slot(ctx, ctx->op->in[1]);
	st						 = OK;
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer q_view = batch_row_view(q_all, c * seg.rows, n_heads * head_dim);
		buffer k_view = batch_row_view(k_all, c * seg.rows, n_kv_heads * head_dim);
		st			  = a->rope_qk_batch(a, &q_view, &k_view, n_heads, n_kv_heads, head_dim,
										 exec_segment_pos(ctx, c), ctx->s->model.rope_cos,
										 ctx->s->model.rope_sin, seg.rows);
	}
	profile_end(prof, &ps);
	return st;
}

status_code op_kv_put(exec_ctx *ctx) {
	profile_scope ps			= profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st			= OK;
	int			  kv_row_stride = exec_layer_ctx(ctx)->kv_row_stride;
	exec_segments seg			= exec_segments_of(ctx);
	buffer		 *k_all			= batch_slot(ctx->bs, ctx->op->in[0]);
	buffer		 *v_all			= batch_slot(ctx->bs, ctx->op->in[1]);
	for (int c = 0; c < seg.n_calls; c++) {
		buffer k_view = batch_row_view(k_all, c * seg.rows, kv_row_stride);
		buffer v_view = batch_row_view(v_all, c * seg.rows, kv_row_stride);
		st = kvcache_put_batch(exec_segment_cache(ctx, c), ctx->m, ctx->li,
							   exec_segment_pos(ctx, c), &k_view, &v_view, kv_row_stride, seg.rows);
		if (st != OK)
			break;
	}
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code attention_run(exec_ctx *ctx, backend *a, int use_swa) {
	int	  n_heads			= ctx->op->u.attention.n_heads;
	int	  n_kv_heads		= ctx->op->u.attention.n_kv_heads;
	int	  head_dim			= ctx->op->u.attention.head_dim;
	int	  n_ctx				= ctx->rows.cache->n_ctx;
	float scale				= ctx->op->u.attention.scale;
	int	  sliding_window	= ctx->op->u.attention.sliding_window;
	int	  n_kv_heads_active = ctx->op->u.attention.n_kv_heads_active;
	int	  kv_layer =
		ctx->m->arch_info->has_variable_layer_dims ? ctx->op->u.attention.kv_layer : ctx->li;

	exec_segments seg	= exec_segments_of(ctx);
	buffer		 *q_all = batch_slot(ctx->bs, ctx->op->in[0]);
	buffer		 *o_all = batch_slot(ctx->bs, ctx->op->out);
	status_code	  st	= OK;
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer	q_view = batch_row_view(q_all, c * seg.rows, n_heads * head_dim);
		buffer	o_view = batch_row_view(o_all, c * seg.rows, n_heads * head_dim);
		buffer *kb;
		buffer *vb;
		int		kv_slot;
		kvcache_attn_view(exec_segment_cache(ctx, c), ctx->m, ctx->li, kv_layer, &kb, &vb,
						  &kv_slot);
		if (use_swa)
			st = a->attention_swa_batch(a, &q_view, kb, vb, &o_view, kv_slot,
										exec_segment_pos(ctx, c), n_heads, n_kv_heads, head_dim,
										n_ctx, ctx->flash_attn, scale, sliding_window,
										n_kv_heads_active, seg.rows);
		else
			st = a->attention_batch(a, &q_view, kb, vb, &o_view, kv_slot, exec_segment_pos(ctx, c),
									n_heads, n_kv_heads, head_dim, n_ctx, ctx->flash_attn, scale,
									n_kv_heads_active, seg.rows);
	}
	return st;
}

static status_code attention_dynamic(exec_ctx *ctx) {
	backend *a		 = exec_layer_backend(ctx);
	int		 use_swa = ctx->op->u.attention.sliding_window > 0 && ctx->li >= 0 &&
					   model_layer_is_sliding(ctx->m, ctx->li) && a->attention_swa_batch != NULL;
	return attention_run(ctx, a, use_swa);
}

status_code op_attention_global(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = attention_run(ctx, exec_layer_backend(ctx), 0);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_attention_swa(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st =
		attention_run(ctx, exec_layer_backend(ctx), model_layer_is_sliding(ctx->m, ctx->li));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_attention(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = attention_dynamic(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_add(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	const int	  dim = ctx->m->dim;
	st = a->add_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]), dim,
					  ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st;
}

status_code op_swap(exec_ctx *ctx) {
	buffer *s0	= exec_slot(ctx, ctx->op->in[0]);
	buffer *s1	= exec_slot(ctx, ctx->op->in[1]);
	buffer	tmp = *s0;
	*s0			= *s1;
	*s1			= tmp;
	return OK;
}

status_code op_ffn_activate(exec_ctx *ctx) {
	backend		 *a	   = exec_layer_backend(ctx);
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, ctx->op->stage);
	status_code	  st;
	int			  n	  = ctx->op->u.ffn_act.n;
	int			  act = ctx->op->u.ffn_act.activation;
	st = a->ffn_activate_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
							   exec_slot(ctx, ctx->op->out), n, act, ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st;
}

static status_code ffn_fused_device(exec_ctx *ctx, backend *a) {
	int n = ctx->op->u.ffn_act.n;
	backend_report_host_fallback(a, "ffn_activate_fused", HFB_BATCH_DESIGN,
								 "batch prefill stages gate/up activation through host memory");
	return ffn_activate_fused_device_fallback(ctx, a, n, ctx->op->u.ffn_act.activation, 2 * n,
											  ctx->rows.n_rows);
}

static status_code ffn_fused_host(exec_ctx *ctx) {
	int			 n	   = ctx->op->u.ffn_act.n;
	const float *fused = batch_buf_ptr(batch_slot(ctx->bs, ctx->op->in[0]));
	float		*out   = batch_buf_ptr(batch_slot(ctx->bs, ctx->op->out));
	backend		*t	   = op_host_backend();
	if (!t->ffn_activate_ex)
		return ERR_UNSUPPORTED;
	return ffn_activate_ex_rows(t, fused, out, n, ctx->op->u.ffn_act.activation, ctx->rows.n_rows,
								2 * n);
}

static status_code ffn_fused_dynamic(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (a->ffn_activate_fused_batch) {
		status_code st = a->ffn_activate_fused_batch(
			a, batch_slot(ctx->bs, ctx->op->in[0]), batch_slot(ctx->bs, ctx->op->out),
			ctx->op->u.ffn_act.n, ctx->op->u.ffn_act.activation, ctx->rows.n_rows);
		if (st != ERR_UNSUPPORTED)
			return st;
	}
	return backend_has_cap(a, BCAP_IS_HOST) ? ffn_fused_host(ctx) : ffn_fused_device(ctx, a);
}

status_code op_ffn_activate_fused_device(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = ffn_fused_device(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_ffn_activate_fused_host(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = ffn_fused_host(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_ffn_activate_fused(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = ffn_fused_dynamic(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code softcap_run(exec_ctx *ctx, backend *t) {
	float cap = ctx->op->u.softcap.cap;
	if (cap <= 0.0f)
		return OK;
	return t->softcap(t, exec_slot(ctx, RECIPE_SLOT_LOGITS), cap,
					  ctx->m->vocab_size * ctx->rows.n_rows);
}

status_code op_softcap_native(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = softcap_run(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_softcap(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = softcap_run(ctx, OP_BACKEND(exec_layer_backend(ctx), softcap));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_logits_readback(exec_ctx *ctx) {
	if (!ctx->logits_out)
		return OK;
	ensure_sync(ctx->m->backend);
	buffer *logits = exec_slot(ctx, RECIPE_SLOT_LOGITS);
	return logits->owner->buffer_read_f32(logits->owner, logits, ctx->logits_out,
										  ctx->rows.n_rows * ctx->m->vocab_size);
}

typedef status_code (*batch_row_fn)(buffer *row, void *ud);

static status_code batch_row_each(exec_ctx *ctx, uint8_t slot, int row_elems, batch_row_fn fn,
								  void *ud) {
	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer		rowb = batch_row_view(batch_slot(ctx->bs, slot), row, row_elems);
		status_code st	 = fn(&rowb, ud);
		if (st != OK)
			return st;
	}
	return OK;
}

typedef struct {
	backend		 *t;
	const buffer *w;
	int			  n_heads;
	int			  head_dim;
	float		  eps;
} rmsnorm_ph_row_ud;

static status_code rmsnorm_ph_row_fn(buffer *rowb, void *v) {
	rmsnorm_ph_row_ud *u = v;
	return u->t->rmsnorm_per_head(u->t, rowb, u->w, rowb, u->n_heads, u->head_dim, u->eps);
}

typedef struct {
	backend *t;
	float	 eps;
	int		 n;
} rmsnorm_nw_row_ud;

static status_code rmsnorm_nw_row_fn(buffer *rowb, void *v) {
	rmsnorm_nw_row_ud *u = v;
	return u->t->rmsnorm_noweight(u->t, rowb, rowb, u->n, u->eps);
}

static status_code rmsnorm_per_head_batched(exec_ctx *ctx, backend *t) {
	const layer_ctx_entry *lc = exec_layer_ctx(ctx);
	int n_heads = ctx->op->w_idx == WIDX_ATTN_Q_NORM ? ctx->m->n_heads : lc->n_kv_heads;
	return t->rmsnorm_per_head_batch(
		t, exec_slot(ctx, ctx->op->in[0]), exec_weight(ctx, ctx->op->w_idx),
		exec_slot(ctx, ctx->op->in[0]), n_heads, lc->head_dim, ctx->m->norm_eps, ctx->rows.n_rows);
}

static status_code rmsnorm_per_head_rows(exec_ctx *ctx, backend *t) {
	const layer_ctx_entry *lc	= exec_layer_ctx(ctx);
	int					   is_q = ctx->op->w_idx == WIDX_ATTN_Q_NORM;
	rmsnorm_ph_row_ud	   ud	= {t, exec_weight(ctx, ctx->op->w_idx),
								   is_q ? ctx->m->n_heads : lc->n_kv_heads, lc->head_dim,
								   ctx->m->norm_eps};
	return batch_row_each(ctx, ctx->op->in[0], is_q ? lc->q_row_stride : lc->kv_row_stride,
						  rmsnorm_ph_row_fn, &ud);
}

status_code op_rmsnorm_per_head_batch(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_per_head_batched(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_rmsnorm_per_head_rows(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_per_head_rows(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code rmsnorm_per_head_dynamic(exec_ctx *ctx) {
	backend *t = OP_BACKEND(exec_layer_backend(ctx), rmsnorm_per_head);
	return t->rmsnorm_per_head_batch ? rmsnorm_per_head_batched(ctx, t)
									 : rmsnorm_per_head_rows(ctx, t);
}

status_code op_rmsnorm_per_head(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_per_head_dynamic(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code rmsnorm_noweight_ph_batched(exec_ctx *ctx, backend *t) {
	const layer_ctx_entry *lc = exec_layer_ctx(ctx);
	return t->rmsnorm_noweight_per_head_batch(t, exec_slot(ctx, ctx->op->in[0]),
											  exec_slot(ctx, ctx->op->in[0]), lc->n_kv_heads,
											  lc->head_dim, ctx->m->norm_eps, ctx->rows.n_rows);
}

static status_code rmsnorm_noweight_batched(exec_ctx *ctx, backend *t) {
	const layer_ctx_entry *lc = exec_layer_ctx(ctx);
	return t->rmsnorm_noweight_batch(t, exec_slot(ctx, ctx->op->in[0]),
									 exec_slot(ctx, ctx->op->in[0]), lc->kv_row_stride,
									 ctx->m->norm_eps, ctx->rows.n_rows);
}

static status_code rmsnorm_noweight_ph_rows(exec_ctx *ctx, backend *t) {
	const layer_ctx_entry *lc = exec_layer_ctx(ctx);
	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer rowb = batch_row_view(batch_slot(ctx->bs, ctx->op->in[0]), row, lc->kv_row_stride);
		status_code st = t->rmsnorm_noweight_per_head(t, &rowb, &rowb, lc->n_kv_heads, lc->head_dim,
													  ctx->m->norm_eps);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code rmsnorm_noweight_rows(exec_ctx *ctx, backend *t) {
	const layer_ctx_entry *lc = exec_layer_ctx(ctx);
	rmsnorm_nw_row_ud	   ud = {t, ctx->m->norm_eps, lc->kv_row_stride};
	return batch_row_each(ctx, ctx->op->in[0], lc->kv_row_stride, rmsnorm_nw_row_fn, &ud);
}

status_code op_rmsnorm_noweight_ph_batch(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_noweight_ph_batched(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_rmsnorm_noweight_batch(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_noweight_batched(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_rmsnorm_noweight_ph_rows(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_noweight_ph_rows(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_rmsnorm_noweight_rows(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_noweight_rows(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code rmsnorm_noweight_dynamic(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	backend *t = OP_BACKEND(a, rmsnorm_noweight_per_head);
	if (t->rmsnorm_noweight_per_head_batch)
		return rmsnorm_noweight_ph_batched(ctx, t);
	if (t->rmsnorm_noweight_batch)
		return rmsnorm_noweight_batched(ctx, t);
	if (t->rmsnorm_noweight_per_head)
		return rmsnorm_noweight_ph_rows(ctx, t);
	return rmsnorm_noweight_rows(ctx, OP_BACKEND(a, rmsnorm_noweight));
}

status_code op_rmsnorm_noweight(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_noweight_dynamic(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code rmsnorm_add_fused(exec_ctx *ctx, backend *a) {
	return a->rmsnorm_add_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_weight(ctx, ctx->op->w_idx),
								exec_slot(ctx, ctx->op->in[1]), exec_slot(ctx, ctx->op->out),
								ctx->m->dim, ctx->op->u.rmsnorm.eps, ctx->op->u.rmsnorm.out_scale,
								ctx->rows.n_rows);
}

static status_code rmsnorm_add_rows(exec_ctx *ctx, backend *a) {
	const buffer *w = exec_weight(ctx, ctx->op->w_idx);
	int			  n = ctx->m->dim;
	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer		xrow   = batch_row_view(exec_slot(ctx, ctx->op->in[0]), row, n);
		buffer		resrow = batch_row_view(exec_slot(ctx, ctx->op->in[1]), row, n);
		buffer		outrow = batch_row_view(exec_slot(ctx, ctx->op->out), row, n);
		status_code st = a->rmsnorm_add(a, &xrow, w, &resrow, &outrow, n, ctx->op->u.rmsnorm.eps,
										ctx->op->u.rmsnorm.out_scale);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code rmsnorm_add_composed(exec_ctx *ctx, backend *a) {
	int			n		  = ctx->m->dim;
	float		out_scale = ctx->op->u.rmsnorm.out_scale;
	buffer	   *outb	  = exec_slot(ctx, ctx->op->out);
	status_code st =
		a->rmsnorm_batch(a, exec_slot(ctx, ctx->op->in[0]), exec_weight(ctx, ctx->op->w_idx), outb,
						 n, ctx->op->u.rmsnorm.eps, ctx->rows.n_rows);
	if (st == OK)
		st = a->add_batch(a, outb, exec_slot(ctx, ctx->op->in[1]), n, ctx->rows.n_rows);
	if (st != OK || out_scale == 1.0f)
		return st;
	if (a->scale_inplace)
		return a->scale_inplace(a, outb, out_scale, n * ctx->rows.n_rows);
	return scale_inplace_fallback(ctx->s, a, outb, n * ctx->rows.n_rows, out_scale);
}

status_code op_rmsnorm_add_fused(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_add_fused(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_rmsnorm_add_composed(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_add_composed(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code rmsnorm_add_dynamic(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (a->rmsnorm_add_batch)
		return rmsnorm_add_fused(ctx, a);
	if (a->rmsnorm_add && backend_has_cap(a, BCAP_RMSNORM_ADD) && ctx->rows.n_rows < 4)
		return rmsnorm_add_rows(ctx, a);
	return rmsnorm_add_composed(ctx, a);
}

status_code op_rmsnorm_add(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = rmsnorm_add_dynamic(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

int exec_rmsnorm_add_norm(exec_ctx *ctx, const recipe_op *add, const recipe_op *norm) {
	backend *a = exec_layer_backend(ctx);
	if (!a->rmsnorm_add_norm_batch || norm->kind != OP_RMSNORM || norm->in[0] != add->out ||
		norm->u.rmsnorm.eps != add->u.rmsnorm.eps)
		return 0;
	profile		 *prof = &ctx->s->prof;
	profile_scope ps   = profile_begin(prof, add->stage);
	status_code	  st   = a->rmsnorm_add_norm_batch(
		a, exec_slot(ctx, add->in[0]), exec_weight(ctx, add->w_idx), exec_slot(ctx, add->in[1]),
		exec_slot(ctx, add->out), exec_weight(ctx, norm->w_idx), exec_slot(ctx, norm->out),
		ctx->m->dim, add->u.rmsnorm.eps, add->u.rmsnorm.out_scale, ctx->rows.n_rows);
	profile_end(prof, &ps);
	return st == OK ? 2 : 0;
}

static status_code scale_run(exec_ctx *ctx, int native) {
	backend					   *a	 = exec_layer_backend(ctx);
	profile					   *prof = &ctx->s->prof;
	profile_scope				ps	 = profile_begin(prof, ctx->op->stage);
	status_code					st	 = OK;
	const struct layer_weights *L	 = &ctx->m->layers[ctx->li];
	float						sc	 = ctx->op->u.scale.scale;
	if (ctx->m->arch_info->has_layer_output_scale && L && L->layer_out_scale != 1.0f)
		sc = L->layer_out_scale;
	if (sc == 1.0f)
		return OK;
	int scale_dim = ctx->m->dim;
	if (native) {
		st = a->scale_inplace(a, exec_slot(ctx, ctx->op->in[0]), sc, scale_dim * ctx->rows.n_rows);
	} else {
		st = scale_inplace_fallback(ctx->s, a, exec_slot(ctx, ctx->op->in[0]),
									scale_dim * ctx->rows.n_rows, sc);
	}
	profile_end(prof, &ps);
	return st;
}

status_code op_scale(exec_ctx *ctx) {
	return scale_run(ctx, exec_layer_backend(ctx)->scale_inplace != NULL);
}

status_code op_scale_native(exec_ctx *ctx) {
	return scale_run(ctx, 1);
}

status_code op_scale_host(exec_ctx *ctx) {
	return scale_run(ctx, 0);
}

static status_code ffn_activate_ex_rows(backend *t, const float *fused, float *out, int n, int act,
										int n_rows, int fused_stride) {
	for (int row = 0; row < n_rows; row++) {
		const float *g	= fused + (size_t)row * fused_stride;
		const float *u	= g + n;
		float		*o	= out + (size_t)row * n;
		buffer		 gb = buffer_host_view(t, (void *)g, (size_t)n * sizeof(float));
		buffer		 ub = buffer_host_view(t, (void *)u, (size_t)n * sizeof(float));
		buffer		 ob = buffer_host_view(t, o, (size_t)n * sizeof(float));
		status_code	 st = t->ffn_activate_ex(t, &gb, &ub, &ob, n, act);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code ffn_activate_fused_device_fallback(exec_ctx *ctx, backend *a, int n, int act,
													  int fused_stride, int n_rows) {
	buffer *fused_buf = batch_slot(ctx->bs, ctx->op->in[0]);
	buffer *out_buf	  = batch_slot(ctx->bs, ctx->op->out);
	float  *fused_host =
		float_buf_ensure_nocopy(&ctx->bs->moe_fused_host, (size_t)n_rows * fused_stride, 64);
	float	   *out_host = float_buf_ensure_nocopy(&ctx->bs->moe_fact_out, (size_t)n_rows * n, 64);
	status_code st		 = a->buffer_read_f32(a, fused_buf, fused_host, n_rows * fused_stride);
	if (st != OK)
		return st;
	backend *t = op_host_backend();
	if (!t->ffn_activate_ex)
		return ERR_UNSUPPORTED;
	st = ffn_activate_ex_rows(t, fused_host, out_host, n, act, n_rows, fused_stride);
	if (st != OK)
		return st;
	return a->buffer_write_f32(a, out_buf, out_host, n_rows * n);
}

static status_code matmul_multi_run(exec_ctx *ctx, backend *a, int fused) {
	const layer_weights	  *lw = &ctx->m->layers[ctx->li];
	const layer_ctx_entry *lc = exec_layer_ctx(ctx);
	int					   k  = ctx->op->u.matmul_multi.k;
	if (ctx->op->w_idx == WIDX_WQ) {
		const int *n_out = ctx->op->u.matmul_multi.n_out;
		if (!lc->has_kv) {
			profile_weight_bytes(&ctx->s->prof, ctx->op->stage, lw->wq.type, n_out[0], k);
			return a->matmul_batch(a, &lw->wq.buf, lw->wq.type, batch_slot(ctx->bs, ctx->op->in[0]),
								   batch_slot(ctx->bs, ctx->op->out), n_out[0], k,
								   ctx->rows.n_rows);
		}
		int n_mats = lc->has_own_v ? 3 : 2;
		if (fused) {
			const buffer *ws[3]	 = {&lw->wq.buf, &lw->wk.buf, &lw->wv.buf};
			uint32_t	  wts[3] = {lw->wq.type, lw->wk.type, lw->wv.type};
			int			  nb	 = n_mats;
			for (int i = 0; i < nb; i++)
				profile_weight_bytes(&ctx->s->prof, ctx->op->stage, wts[i], n_out[i], k);
			buffer *ys[3]  = {batch_slot(ctx->bs, ctx->op->out), batch_slot(ctx->bs, ctx->op->out1),
							  batch_slot(ctx->bs, ctx->op->out2)};
			status_code st = a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]),
												   ys, n_out, k, n_mats, ctx->rows.n_rows);
			if (st == OK && !lc->has_own_v)
				st = batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->rows.n_rows, ctx->op->out1,
									   ctx->op->out2);
			return st;
		}
		status_code st =
			a->matmul_batch(a, &lw->wq.buf, lw->wq.type, batch_slot(ctx->bs, ctx->op->in[0]),
							batch_slot(ctx->bs, ctx->op->out), n_out[0], k, ctx->rows.n_rows);
		if (st == OK)
			st = a->matmul_batch(a, &lw->wk.buf, lw->wk.type, batch_slot(ctx->bs, ctx->op->in[0]),
								 batch_slot(ctx->bs, ctx->op->out1), n_out[1], k, ctx->rows.n_rows);
		if (st == OK && lc->has_own_v)
			st = a->matmul_batch(a, &lw->wv.buf, lw->wv.type, batch_slot(ctx->bs, ctx->op->in[0]),
								 batch_slot(ctx->bs, ctx->op->out2), n_out[2], k, ctx->rows.n_rows);
		if (st == OK && !lc->has_own_v)
			st = batch_copy_k_to_v(ctx->m, ctx->bs, ctx->li, ctx->rows.n_rows, ctx->op->out1,
								   ctx->op->out2);
		return st;
	}
	if (ctx->op->w_idx == WIDX_GATE) {
		const int *n_out = ctx->op->u.matmul_multi.n_out;
		if (fused) {
			const buffer *ws[2]	 = {&lw->gate_w.buf, &lw->up_w.buf};
			uint32_t	  wts[2] = {lw->gate_w.type, lw->up_w.type};
			profile_weight_bytes(&ctx->s->prof, ctx->op->stage, wts[0], n_out[0], k);
			profile_weight_bytes(&ctx->s->prof, ctx->op->stage, wts[1], n_out[1], k);
			buffer *ys[2] = {batch_slot(ctx->bs, ctx->op->out), batch_slot(ctx->bs, ctx->op->out1)};
			return a->matmul_multi_batch(a, ws, wts, batch_slot(ctx->bs, ctx->op->in[0]), ys, n_out,
										 k, 2, ctx->rows.n_rows);
		}
		status_code st = a->matmul_batch(
			a, &lw->gate_w.buf, lw->gate_w.type, batch_slot(ctx->bs, ctx->op->in[0]),
			batch_slot(ctx->bs, ctx->op->out), n_out[0], k, ctx->rows.n_rows);
		if (st == OK)
			st = a->matmul_batch(a, &lw->up_w.buf, lw->up_w.type,
								 batch_slot(ctx->bs, ctx->op->in[0]),
								 batch_slot(ctx->bs, ctx->op->out1), n_out[1], k, ctx->rows.n_rows);
		return st;
	}
	return ERR_INVALID_ARG;
}

static status_code matmul_multi_dynamic(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	return matmul_multi_run(ctx, a, a->matmul_multi_batch != NULL);
}

status_code op_matmul_multi_fused(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = matmul_multi_run(ctx, exec_layer_backend(ctx), 1);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_matmul_multi_split(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = matmul_multi_run(ctx, exec_layer_backend(ctx), 0);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code matmul_multi_batch_body(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = matmul_multi_dynamic(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code batch_copy_k_to_v(model *m, batch_scratch *bs, int li, int n_tokens,
									 uint8_t k_slot, uint8_t v_slot) {
	const layer_ctx_entry *lc			 = &m->recipe->layer_ctx[li];
	int					   kv_out		 = lc->kv_row_stride;
	backend				  *a			 = m->backend;
	buffer				  *kb			 = &bs->pair[k_slot].b;
	buffer				  *vb			 = &bs->pair[v_slot].b;
	int					   kv_row_stride = lc->kv_row_stride;

	return copy_k_to_v_buffers(a, kb, vb, kv_out, kv_row_stride, n_tokens, &bs->copy_tmp);
}
