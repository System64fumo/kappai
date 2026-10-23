#include "recipe/internal.h"

static buffer mla_stage_view(float_buf *fb, size_t n) {
	float_buf_ensure_nocopy(fb, n, 64);
	buffer buf	 = {0};
	buf.handle	 = fb->p;
	buf.size	 = n * sizeof(float);
	buf.host_ptr = fb->p;
	return buf;
}

static buffer *mla_kv_view(const exec_ctx *ctx, kvcache *cache, int *slot) {
	buffer *k, *v;
	kvcache_attn_view(cache, ctx->m, ctx->li, ctx->li, &k, &v, slot);
	return k;
}

static const buffer *mla_host_weight(const weight_ref *w, size_t bytes, backend *host,
									 buffer *view) {
	if (!w->buf.owner || backend_has_cap(w->buf.owner, BCAP_IS_HOST))
		return &w->buf;
	*view = buffer_host_view(host, (void *)w->host_ptr, bytes);
	return view;
}

static status_code mla_kv_put_row(exec_ctx *ctx, kvcache *cache, backend *a, const buffer *kv_a_row,
								  int pos) {
	const model	  *m	   = ctx->m;
	layer_weights *L	   = &m->layers[ctx->li];
	const int	   kv_lora = m->mla.kv_lora;
	const int	   qk_rope = m->mla.qk_rope;
	int			   slot;
	buffer		  *kv = mla_kv_view(ctx, cache, &slot);
	if (kv->owner == a)
		return a->kv_put_mla(a, kv, slot, pos, kv_a_row, &L->kv_a_norm_w.buf, kv_lora, qk_rope,
							 cache->n_ctx, m->norm_eps);

	backend		 *host = op_backend_fallback(a, "kv_put_mla");
	buffer		  norm_view;
	const buffer *norm_w =
		mla_host_weight(&L->kv_a_norm_w, (size_t)kv_lora * sizeof(float), host, &norm_view);
	buffer		staged = mla_stage_view(&ctx->s->run.stage_a, (size_t)kv_lora + qk_rope);
	status_code st	   = a->buffer_read_f32(a, kv_a_row, staged.handle, kv_lora + qk_rope);
	if (st == OK)
		st = host->kv_put_mla(host, kv, slot, pos, &staged, norm_w, kv_lora, qk_rope, cache->n_ctx,
							  m->norm_eps);
	return st;
}

static status_code mla_attention_row(exec_ctx *ctx, kvcache *cache, backend *a, const buffer *q,
									 buffer *out, int pos) {
	const model	  *m	   = ctx->m;
	layer_weights *L	   = &m->layers[ctx->li];
	const int	   n_heads = ctx->op->u.attention.n_heads;
	const float	   scale   = ctx->op->u.attention.scale;
	int			   slot;
	buffer		  *kv = mla_kv_view(ctx, cache, &slot);
	if (kv->owner == a)
		return a->attention_mla(a, q, kv, &L->k_b_w.buf, &L->v_b_w.buf, out, slot, pos, n_heads,
								m->mla.qk_head, m->mla.qk_rope, m->mla.qk_nope, m->mla.v_head,
								m->mla.kv_lora, cache->n_ctx, ctx->s->model.rope_cos,
								ctx->s->model.rope_sin, L->k_b_w.type, L->v_b_w.type, scale);

	backend		 *host = op_backend_fallback(a, "attention_mla");
	buffer		  k_b_view, v_b_view;
	const buffer *k_b_w = mla_host_weight(
		&L->k_b_w, (size_t)n_heads * m->mla.qk_nope * m->mla.kv_lora * sizeof(float), host,
		&k_b_view);
	const buffer *v_b_w =
		mla_host_weight(&L->v_b_w, (size_t)n_heads * m->mla.kv_lora * m->mla.v_head * sizeof(float),
						host, &v_b_view);
	int			q_n		 = n_heads * m->mla.qk_head;
	int			out_n	 = n_heads * m->mla.v_head;
	buffer		q_host	 = mla_stage_view(&ctx->s->run.stage_a, (size_t)q_n);
	buffer		out_host = mla_stage_view(&ctx->s->run.stage_b, (size_t)out_n);
	status_code st		 = a->buffer_read_f32(a, q, q_host.handle, q_n);
	if (st == OK)
		st = host->attention_mla(host, &q_host, kv, k_b_w, v_b_w, &out_host, slot, pos, n_heads,
								 m->mla.qk_head, m->mla.qk_rope, m->mla.qk_nope, m->mla.v_head,
								 m->mla.kv_lora, cache->n_ctx, ctx->s->model.rope_cos,
								 ctx->s->model.rope_sin, L->k_b_w.type, L->v_b_w.type, scale);
	if (st == OK)
		st = a->buffer_write_f32(a, out, out_host.handle, out_n);
	return st;
}

status_code op_mla_qkv_proj_fused(exec_ctx *ctx) {

	backend		  *a		 = exec_layer_backend(ctx);
	const int	   dim		 = ctx->m->dim;
	layer_weights *L		 = &ctx->m->layers[ctx->li];
	int			   q_lora	 = ctx->m->mla.q_lora;
	int			   kv_lora	 = ctx->m->mla.kv_lora;
	int			   qk_rope	 = ctx->m->mla.qk_rope;
	int			   q_b_rows	 = ctx->m->n_heads * ctx->m->mla.qk_head;
	int			   kv_a_rows = kv_lora + qk_rope;
	buffer		  *xb		 = batch_slot(ctx->bs, ctx->op->in[0]);

	bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE, (size_t)ctx->rows.n_rows * q_lora);
	bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_UP, (size_t)ctx->rows.n_rows * kv_a_rows);
	status_code st	   = OK;
	buffer	   *q_a_b  = &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b;
	buffer	   *kv_a_b = &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b;

	if (exec_caps(ctx, mla_qkv_caps) & MLA_QKV_MULTI) {
		const buffer *ws[2]	   = {&L->q_a_w.buf, &L->kv_a_w.buf};
		uint32_t	  wts[2]   = {L->q_a_w.type, L->kv_a_w.type};
		buffer		 *ys[2]	   = {q_a_b, kv_a_b};
		int			  n_out[2] = {q_lora, kv_a_rows};
		st = a->matmul_multi_batch(a, ws, wts, xb, ys, n_out, dim, 2, ctx->rows.n_rows);
	} else {
		st = a->matmul_batch(a, &L->q_a_w.buf, L->q_a_w.type, xb, q_a_b, q_lora, dim,
							 ctx->rows.n_rows);
		if (st == OK)
			st = a->matmul_batch(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, kv_a_b, kv_a_rows, dim,
								 ctx->rows.n_rows);
	}
	if (st != OK)
		goto mla_qkv_cleanup;

	if (q_lora > 0)
		st = a->rmsnorm_batch(a, q_a_b, &L->q_a_norm_w.buf, q_a_b, q_lora, ctx->m->norm_eps,
							  ctx->rows.n_rows);
	if (st != OK)
		goto mla_qkv_cleanup;

	buffer *q_buf = batch_slot(ctx->bs, RECIPE_SLOT_Q);
	st			  = a->matmul_batch(a, &L->q_b_w.buf, L->q_b_w.type, q_a_b, q_buf, q_b_rows, q_lora,
									ctx->rows.n_rows);
	if (st != OK)
		goto mla_qkv_cleanup;

	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer kv_a_row = batch_row_view(kv_a_b, row, kv_a_rows);
		st =
			mla_kv_put_row(ctx, exec_segment_cache(ctx, row), a, &kv_a_row, exec_row_pos(ctx, row));
		if (st != OK)
			goto mla_qkv_cleanup;
	}
mla_qkv_cleanup:
	return st;
}

status_code op_mla_q_proj(exec_ctx *ctx) {

	backend		  *a		= exec_layer_backend(ctx);
	const int	   dim		= ctx->m->dim;
	layer_weights *L		= &ctx->m->layers[ctx->li];
	int			   q_lora	= ctx->m->mla.q_lora;
	int			   q_b_rows = ctx->m->n_heads * ctx->m->mla.qk_head;
	buffer		  *xb		= batch_slot(ctx->bs, ctx->op->in[0]);

	bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_GATE, (size_t)ctx->rows.n_rows * q_lora);
	status_code st	  = OK;
	buffer	   *q_a_b = &ctx->bs->pair[RECIPE_SLOT_FFN_GATE].b;

	st = a->matmul_batch(a, &L->q_a_w.buf, L->q_a_w.type, xb, q_a_b, q_lora, dim, ctx->rows.n_rows);
	if (st != OK)
		goto mla_q_cleanup;
	if (q_lora > 0)
		st = a->rmsnorm_batch(a, q_a_b, &L->q_a_norm_w.buf, q_a_b, q_lora, ctx->m->norm_eps,
							  ctx->rows.n_rows);
	if (st != OK)
		goto mla_q_cleanup;
	buffer *q_buf = batch_slot(ctx->bs, RECIPE_SLOT_Q);
	st			  = a->matmul_batch(a, &L->q_b_w.buf, L->q_b_w.type, q_a_b, q_buf, q_b_rows, q_lora,
									ctx->rows.n_rows);
mla_q_cleanup:
	return st;
}

status_code op_mla_kv_proj(exec_ctx *ctx) {

	backend		  *a		 = exec_layer_backend(ctx);
	const int	   dim		 = ctx->m->dim;
	layer_weights *L		 = &ctx->m->layers[ctx->li];
	int			   kv_lora	 = ctx->m->mla.kv_lora;
	int			   qk_rope	 = ctx->m->mla.qk_rope;
	int			   kv_a_rows = kv_lora + qk_rope;
	buffer		  *xb		 = batch_slot(ctx->bs, ctx->op->in[0]);
	bs_ensure_slot(ctx->bs, a, RECIPE_SLOT_FFN_UP, (size_t)ctx->rows.n_rows * kv_a_rows);
	status_code st	   = OK;
	buffer	   *kv_a_b = &ctx->bs->pair[RECIPE_SLOT_FFN_UP].b;

	st = a->matmul_batch(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, kv_a_b, kv_a_rows, dim,
						 ctx->rows.n_rows);
	if (st != OK)
		goto mla_kv_cleanup;
	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer kv_a_row = batch_row_view(kv_a_b, row, kv_a_rows);
		st =
			mla_kv_put_row(ctx, exec_segment_cache(ctx, row), a, &kv_a_row, exec_row_pos(ctx, row));
		if (st != OK)
			goto mla_kv_cleanup;
	}
mla_kv_cleanup:
	return st;
}

status_code op_attention_mla(exec_ctx *ctx) {
	backend	 *a		   = exec_layer_backend(ctx);
	const int q_b_rows = ctx->m->n_heads * ctx->m->mla.qk_head;
	const int wo_in	   = ctx->op->u.attention.n_heads * ctx->m->mla.v_head;
	buffer	 *q_buf	   = batch_slot(ctx->bs, ctx->op->in[0]);
	buffer	 *out_buf  = batch_slot(ctx->bs, ctx->op->out);
	for (int row = 0; row < ctx->rows.n_rows; row++) {
		buffer		q_row	= batch_row_view(q_buf, row, q_b_rows);
		buffer		out_row = batch_row_view(out_buf, row, wo_in);
		status_code st = mla_attention_row(ctx, exec_segment_cache(ctx, row), a, &q_row, &out_row,
										   exec_row_pos(ctx, row));
		if (st != OK)
			return st;
	}
	return OK;
}

status_code op_mla_absorbed(exec_ctx *ctx) {
	backend		   *a = exec_layer_backend(ctx);
	kvcache_hybrid *c = ctx->rows.cache ? ctx->rows.cache->hybrid : NULL;
	if (!a->mla_absorbed || !c || !c->latent_buf)
		return ERR_UNSUPPORTED;
	const model			*m = ctx->m;
	const layer_weights *L = &m->layers[ctx->li];
	if (L->k_b_w.type != GGML_TYPE_F32 || L->v_b_w.type != GGML_TYPE_F32)
		return ERR_UNSUPPORTED;

	tpool *pool		 = backend_get_pool(a);
	int	   n_threads = pool ? tpool_n_threads(pool) : 1;

	mla_absorbed_desc d = {
		.dim	   = m->dim,
		.n_heads   = m->n_heads,
		.kv_lora   = m->mla.kv_lora,
		.qk_head   = m->mla.qk_head,
		.v_head	   = m->mla.v_head,
		.q_lora	   = m->mla.q_lora,
		.n_threads = n_threads < 1 ? 1 : n_threads,
		.norm_eps  = m->norm_eps,
		.q_a	   = weight_view(&L->q_a_w),
		.q_b	   = weight_view(&L->q_b_w),
		.kv_a	   = weight_view(&L->kv_a_w),
		.wo		   = weight_view(&L->wo),
		.q_a_norm  = &L->q_a_norm_w.buf,
		.kv_a_norm = &L->kv_a_norm_w.buf,
		.k_b	   = &L->k_b_w.buf,
		.v_b	   = &L->v_b_w.buf,
	};
	if (backend_has_cap(a, BCAP_IS_HOST))
		d.ws = float_buf_ensure_nocopy(&ctx->s->run.op_ws, mla_absorbed_ws_floats(&d), 64);

	exec_segments seg	  = exec_segments_of(ctx);
	buffer		 *x_all	  = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *out_all = exec_slot(ctx, ctx->op->out);
	status_code	  st	  = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		kvcache		   *rcache = exec_segment_cache(ctx, call);
		kvcache_hybrid *rc	   = rcache ? rcache->hybrid : c;
		if (!rc || !rc->latent_buf)
			return ERR_UNSUPPORTED;
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int	   row	   = call * seg.rows + t;
			buffer xrow	   = batch_row_view(x_all, row, m->dim);
			buffer orow	   = batch_row_view(out_all, row, m->dim);
			d.latent_cache = &rc->latent_buf[ctx->li];
			d.x			   = &xrow;
			d.out		   = &orow;
			d.pos		   = exec_row_pos(ctx, row);
			st			   = a->mla_absorbed(a, &d);
		}
	}
	return st;
}
