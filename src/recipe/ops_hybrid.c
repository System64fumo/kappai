#include "recipe/internal.h"

static status_code split_qgate_run(exec_ctx *ctx, backend *t) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	return t->split_qgate(t, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->out),
						  exec_slot(ctx, ctx->op->out1), ctx->m->n_heads, ctx->m->head_dim,
						  ctx->rows.n_rows);
}

status_code op_split_qgate_native(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = split_qgate_run(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_split_qgate(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = split_qgate_run(ctx, OP_BACKEND(exec_layer_backend(ctx), split_qgate));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code partial_rope_qk_run(exec_ctx *ctx, backend *t) {
	model		 *m			 = ctx->m;
	int			  n_heads	 = m->n_heads;
	int			  n_kv_heads = m->n_kv_heads;
	int			  head_dim	 = m->head_dim;
	buffer		 *q			 = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *k			 = exec_slot(ctx, ctx->op->in[1]);
	status_code	  st		 = OK;
	exec_segments seg		 = exec_segments_of(ctx);
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		buffer q_view = batch_row_view(q, exec_seg_row0(&seg, c), n_heads * head_dim);
		buffer k_view = batch_row_view(k, exec_seg_row0(&seg, c), n_kv_heads * head_dim);
		st = t->partial_rope_qk(t, &q_view, &k_view, n_heads, n_kv_heads, head_dim, m->rope_dim,
								exec_segment_pos(ctx, c), ctx->s->model.rope_cos,
								ctx->s->model.rope_sin, exec_seg_rows(&seg, c));
	}
	return st;
}

static status_code partial_rope_qk_dynamic(exec_ctx *ctx) {
	return partial_rope_qk_run(ctx, OP_BACKEND(exec_layer_backend(ctx), partial_rope_qk));
}

status_code op_partial_rope_qk_native(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = partial_rope_qk_run(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_partial_rope_qk(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = partial_rope_qk_dynamic(ctx);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static status_code attn_output_gate_run(exec_ctx *ctx, backend *t) {
	if (!ctx || !ctx->m || !ctx->s)
		return ERR_INVALID_ARG;
	return t->attn_output_gate(t, exec_slot(ctx, ctx->op->in[0]), exec_slot(ctx, ctx->op->in[1]),
							   ctx->m->n_heads * ctx->m->head_dim, ctx->rows.n_rows);
}

status_code op_attn_output_gate_native(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st = attn_output_gate_run(ctx, exec_layer_backend(ctx));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

status_code op_attn_output_gate(exec_ctx *ctx) {
	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);
	status_code	  st =
		attn_output_gate_run(ctx, OP_BACKEND(exec_layer_backend(ctx), attn_output_gate));
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static buffer gdn_rows(const buffer *whole, int row0, int row_elems, int n_rows) {
	return buffer_slice(whole, (size_t)row0 * (size_t)row_elems * sizeof(float),
						(size_t)n_rows * (size_t)row_elems * sizeof(float));
}

static buffer gdn_host_rows(backend *owner, const float *base, int row0, int row_elems,
							int n_rows) {
	return buffer_host_view(owner, (void *)(base + (size_t)row0 * (size_t)row_elems),
							(size_t)n_rows * (size_t)row_elems * sizeof(float));
}

status_code op_gated_delta_net(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->rows.cache || !ctx->s || !ctx->rows.cache->hybrid)
		return ERR_INVALID_ARG;
	profile_scope			   ps		 = profile_begin(&ctx->s->prof, ctx->op->stage);
	const model_hybrid_params *p		 = &ctx->m->hybrid;
	layer_weights			  *L		 = &ctx->m->layers[ctx->li];
	int						   n_tokens	 = ctx->rows.n_rows > 0 ? ctx->rows.n_rows : 1;
	backend					  *a		 = exec_layer_backend(ctx);
	kvcache_hybrid			  *hc0		 = ctx->rows.cache->hybrid;
	const int				   state_dev = hc0->conv_buf != NULL;
	backend					  *t;
	if (state_dev) {
		t = a;
	} else {
		const int native = (exec_caps(ctx, gdn_caps) & GDN_NATIVE) != 0;
		if (!native)
			backend_report_host_fallback(a, "gated_delta_net", HFB_OP_NOT_NATIVE,
										 "gated delta net executes on the host (cpu) fallback");
		t = native && backend_has_cap(a, BCAP_IS_HOST) ? a : op_host_backend();
	}
	if (!t->gated_delta_net) {
		profile_end(&ctx->s->prof, &ps);
		return ERR_UNSUPPORTED;
	}
	if (!L->ssm_conv1d_w.buf.owner || !L->ssm_dt_b.buf.owner || !L->ssm_a.buf.owner ||
		!L->ssm_norm_w.buf.owner) {
		profile_end(&ctx->s->prof, &ps);
		return ERR_FORMAT;
	}

	gdn_desc d = {
		.n_value_heads	= p->n_value_heads,
		.n_key_heads	= p->n_key_heads,
		.conv_dim		= p->conv_dim,
		.conv_kernel	= p->conv_kernel,
		.key_dim		= p->key_dim,
		.state_size		= p->state_size,
		.value_head_dim = p->value_head_dim,
		.value_dim		= p->value_dim,
		.eps			= ctx->m->norm_eps,
		.conv_w			= &L->ssm_conv1d_w.buf,
		.dt				= &L->ssm_dt_b.buf,
		.a_vec			= &L->ssm_a.buf,
		.norm_w			= &L->ssm_norm_w.buf,
	};

	if (backend_has_cap(t, BCAP_IS_HOST)) {
		tpool *bpool	 = backend_get_pool(t);
		int	   n_threads = bpool ? tpool_n_threads(bpool) : 1;
		if (n_threads < 1)
			n_threads = 1;
		size_t conv_need	= (size_t)n_tokens * p->conv_dim;
		size_t scratch_need = (size_t)n_threads * (2 * p->state_size + 2 * p->value_head_dim);
		d.ws				= float_buf_ensure(&ctx->s->hybrid.gdn_ws, conv_need + scratch_need);
		if (!d.ws) {
			profile_end(&ctx->s->prof, &ps);
			return ERR_INVALID_ARG;
		}
	}

	status_code st = OK;
	if (state_dev) {
		buffer *mixed_b = exec_slot(ctx, ctx->op->in[0]);
		buffer *z_b		= exec_slot(ctx, ctx->op->in[1]);
		buffer *alpha_b = exec_slot(ctx, ctx->op->in[2]);
		buffer *beta_b	= exec_slot(ctx, ctx->op->in[3]);
		buffer *out_b	= exec_slot(ctx, ctx->op->out);
		if (!mixed_b || !z_b || !alpha_b || !beta_b || !out_b) {
			profile_end(&ctx->s->prof, &ps);
			return ERR_INVALID_ARG;
		}
		exec_segments seg = exec_segments_of(ctx);
		for (int c = 0; c < seg.n_calls && st == OK; c++) {
			kvcache_hybrid *hc = exec_segment_cache(ctx, c)->hybrid;
			if (!hc || !hc->conv_buf || !hc->state_buf) {
				st = ERR_INVALID_ARG;
				break;
			}
			int	   row0	   = c * seg.rows;
			buffer mixed_v = gdn_rows(mixed_b, row0, p->conv_dim, seg.rows);
			buffer z_v	   = gdn_rows(z_b, row0, p->value_dim, seg.rows);
			buffer alpha_v = gdn_rows(alpha_b, row0, p->n_value_heads, seg.rows);
			buffer beta_v  = gdn_rows(beta_b, row0, p->n_value_heads, seg.rows);
			buffer out_v   = gdn_rows(out_b, row0, p->value_dim, seg.rows);
			d.n_tokens	   = seg.rows;
			d.reset		   = exec_segment_pos(ctx, c) == 0;
			d.mixed		   = &mixed_v;
			d.z			   = &z_v;
			d.alpha		   = &alpha_v;
			d.beta		   = &beta_v;
			d.out		   = &out_v;
			d.conv_state   = &hc->conv_buf[ctx->li];
			d.state		   = &hc->state_buf[ctx->li];
			st			   = t->gated_delta_net(t, &d);
		}
		profile_end(&ctx->s->prof, &ps);
		return st;
	}

	int mixed_n = n_tokens * p->conv_dim;
	int z_n		= n_tokens * p->value_dim;
	int alpha_n = n_tokens * p->n_value_heads;
	int beta_n	= n_tokens * p->n_value_heads;
	int out_n	= n_tokens * p->value_dim;

	const float *mixed = recipe_slot_read_f32(ctx, ctx->op->in[0], &ctx->s->hybrid.host, mixed_n);
	const float *z	   = recipe_slot_read_f32(ctx, ctx->op->in[1], &ctx->s->hybrid.gdn_z, z_n);
	const float *alpha =
		recipe_slot_read_f32(ctx, ctx->op->in[2], &ctx->s->hybrid.gdn_alpha, alpha_n);
	const float *beta = recipe_slot_read_f32(ctx, ctx->op->in[3], &ctx->s->hybrid.gdn_beta, beta_n);
	float		*out  = recipe_slot_write_stage(ctx, ctx->op->out, &ctx->s->hybrid.gdn_out, out_n);
	if (!mixed || !z || !alpha || !beta || !out) {
		profile_end(&ctx->s->prof, &ps);
		return ERR_INVALID_ARG;
	}

	const weight_ref *gdn_w[4]	= {&L->ssm_conv1d_w, &L->ssm_dt_b, &L->ssm_a, &L->ssm_norm_w};
	const int		  gdn_wn[4] = {p->conv_dim * p->conv_kernel, p->n_value_heads, p->n_value_heads,
								   p->value_head_dim};
	buffer			  w_v[4];
	for (int i = 0; i < 4; i++) {
		if (buffer_is_host_resident(&gdn_w[i]->buf)) {
			w_v[i] = gdn_w[i]->buf;
			continue;
		}
		float *stg = float_buf_ensure(&ctx->s->hybrid.gdn_w[i], gdn_wn[i]);
		if (!stg || !gdn_w[i]->buf.owner ||
			gdn_w[i]->buf.owner->buffer_read_f32(gdn_w[i]->buf.owner, &gdn_w[i]->buf, stg,
												 gdn_wn[i]) != OK) {
			profile_end(&ctx->s->prof, &ps);
			return ERR_INVALID_ARG;
		}
		w_v[i] = buffer_host_view(t, stg, (size_t)gdn_wn[i] * sizeof(float));
	}
	d.conv_w = &w_v[0];
	d.dt	 = &w_v[1];
	d.a_vec	 = &w_v[2];
	d.norm_w = &w_v[3];

	exec_segments seg = exec_segments_of(ctx);
	for (int c = 0; c < seg.n_calls && st == OK; c++) {
		kvcache_hybrid *hc = exec_segment_cache(ctx, c)->hybrid;
		if (!hc || !hc->conv_state || !hc->recurrent_state) {
			st = ERR_INVALID_ARG;
			break;
		}
		int	   row0	   = c * seg.rows;
		buffer mixed_v = gdn_host_rows(t, mixed, row0, p->conv_dim, seg.rows);
		buffer z_v	   = gdn_host_rows(t, z, row0, p->value_dim, seg.rows);
		buffer alpha_v = gdn_host_rows(t, alpha, row0, p->n_value_heads, seg.rows);
		buffer beta_v  = gdn_host_rows(t, beta, row0, p->n_value_heads, seg.rows);
		buffer out_v   = gdn_host_rows(t, out, row0, p->value_dim, seg.rows);
		buffer conv_v  = buffer_host_view(t, hc->conv_state + (size_t)ctx->li * hc->conv_stride,
										  hc->conv_stride * sizeof(float));
		buffer state_v =
			buffer_host_view(t, hc->recurrent_state + (size_t)ctx->li * hc->recurrent_stride,
							 hc->recurrent_stride * sizeof(float));
		d.n_tokens	 = seg.rows;
		d.reset		 = exec_segment_pos(ctx, c) == 0;
		d.mixed		 = &mixed_v;
		d.z			 = &z_v;
		d.alpha		 = &alpha_v;
		d.beta		 = &beta_v;
		d.out		 = &out_v;
		d.conv_state = &conv_v;
		d.state		 = &state_v;
		st			 = t->gated_delta_net(t, &d);
	}
	if (st == OK)
		st = recipe_slot_write_commit(ctx, ctx->op->out, out, out_n);
	profile_end(&ctx->s->prof, &ps);
	return st;
}

static int lfm_attn_buf(const model *m) {
	int buf	  = m->dim;
	int q_out = m->n_heads * m->head_dim;
	return buf > q_out ? buf : q_out;
}

status_code op_shortconv(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->s ||
		(!ctx->op->u.shortconv.centered && (!ctx->rows.cache || !ctx->rows.cache->hybrid)))
		return ERR_INVALID_ARG;
	model					  *m	  = ctx->m;
	layer_weights			  *L	  = &m->layers[ctx->li];
	const model_hybrid_params *p	  = &m->hybrid;
	const float				  *conv_w = (const float *)L->ssm_conv1d_w.host_ptr;
	if (!conv_w || p->conv_kernel < 2 || ctx->li < 0 || ctx->li >= m->n_layers ||
		(ctx->op->u.shortconv.centered && p->conv_kernel != 3))
		return ERR_FORMAT;

	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);

	int n_rows = ctx->rows.n_rows;

	int			 in_stride	= model_hybrid_proj_size(m);
	int			 out_stride = lfm_attn_buf(m);
	int			 mixed_n	= n_rows * in_stride;
	int			 out_n		= n_rows * out_stride;
	const float *mixed = recipe_slot_read_f32(ctx, ctx->op->in[0], &ctx->s->hybrid.host, mixed_n);
	float		*out   = recipe_slot_write_stage(ctx, ctx->op->out, &ctx->s->hybrid.host2, out_n);

	if (!out || !mixed)
		return ERR_INVALID_ARG;

	backend *a = OP_BACKEND(exec_layer_backend(ctx), shortconv);
	if (!a->shortconv)
		return ERR_UNSUPPORTED;
	exec_segments seg = exec_segments_of(ctx);
	for (int c = 0; c < seg.n_calls; c++) {
		kvcache_hybrid *hc =
			ctx->op->u.shortconv.centered ? NULL : exec_segment_cache(ctx, c)->hybrid;
		float *state = hc ? hc->conv_state + (size_t)ctx->li * hc->conv_stride : NULL;
		if (state && exec_segment_pos(ctx, c) == 0)
			memset(state, 0, (size_t)m->dim * (size_t)(p->conv_kernel - 1) * sizeof(float));
		size_t		   row0 = (size_t)exec_seg_row0(&seg, c);
		shortconv_desc desc = {
			.out		= out + row0 * out_stride,
			.state		= state,
			.mixed		= mixed + row0 * in_stride,
			.conv_w		= conv_w,
			.dim		= m->dim,
			.kernel		= p->conv_kernel,
			.in_stride	= in_stride,
			.out_stride = out_stride,
			.n_rows		= exec_seg_rows(&seg, c),
			.centered	= ctx->op->u.shortconv.centered,
		};
		status_code st = a->shortconv(a, &desc);
		if (st != OK) {
			profile_end(&ctx->s->prof, &ps);
			return st;
		}
	}

	status_code st = recipe_slot_write_commit(ctx, ctx->op->out, out, out_n);

	profile_end(&ctx->s->prof, &ps);
	return st;
}

backend_weight weight_view(const weight_ref *w) {
	return (backend_weight){&w->buf, w->type};
}

status_code op_hc_init(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (!a->hc_init)
		return ERR_UNSUPPORTED;
	const model	 *m		 = ctx->m;
	exec_segments seg	 = exec_segments_of(ctx);
	buffer		 *x_all	 = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *hc_all = exec_slot(ctx, ctx->op->out);
	status_code	  st	 = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int			 row  = call * seg.rows + t;
			buffer		 xrow = batch_row_view(x_all, row, m->dim);
			buffer		 hrow = batch_row_view(hc_all, row, m->kda.hc_count * m->dim);
			hc_init_desc d	  = {.dim = m->dim, .n_hc = m->kda.hc_count, .x = &xrow, .hc = &hrow};
			st				  = a->hc_init(a, &d);
		}
	}
	return st;
}

status_code op_hc_final(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (!a->hc_final)
		return ERR_UNSUPPORTED;
	const model	 *m		 = ctx->m;
	exec_segments seg	 = exec_segments_of(ctx);
	buffer		 *hc_all = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *x_all	 = exec_slot(ctx, ctx->op->out);
	status_code	  st	 = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int			  row  = call * seg.rows + t;
			buffer		  hrow = batch_row_view(hc_all, row, m->kda.hc_count * m->dim);
			buffer		  xrow = batch_row_view(x_all, row, m->dim);
			hc_final_desc d	   = {.dim = m->dim, .n_hc = m->kda.hc_count, .hc = &hrow, .x = &xrow};
			st				   = a->hc_final(a, &d);
		}
	}
	return st;
}

status_code op_hc_pre(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (!a->hc_pre)
		return ERR_UNSUPPORTED;
	const model			*m			= ctx->m;
	const layer_weights *L			= &m->layers[ctx->li];
	const bool			 attn		= ctx->op->w_idx == WIDX_HC_ATTN_FN;
	const weight_ref	*fn			= attn ? &L->hc_attn_fn_w : &L->hc_ffn_fn_w;
	const weight_ref	*base		= attn ? &L->hc_attn_base_w : &L->hc_ffn_base_w;
	const weight_ref	*scale		= attn ? &L->hc_attn_scale_w : &L->hc_ffn_scale_w;
	exec_segments		 seg		= exec_segments_of(ctx);
	buffer				*stream_all = exec_slot(ctx, ctx->op->in[0]);
	buffer				*x_all		= exec_slot(ctx, ctx->op->out);
	buffer				*coef_all	= exec_slot(ctx, ctx->op->out1);
	const int			 flat		= m->kda.hc_count * m->dim;
	const int			 coef_elems = m->kda.hc_count * (m->kda.hc_count + 1);
	hc_pre_desc			 d			= {
		.dim			= m->dim,
		.n_hc			= m->kda.hc_count,
		.sinkhorn_iters = m->kda.hc_sinkhorn_iters,
		.norm_eps		= m->norm_eps,
		.eps			= m->kda.hc_eps,
		.fn				= weight_view(fn),
		.base			= &base->buf,
		.scale			= &scale->buf,
	};
	if (backend_has_cap(a, BCAP_IS_HOST))
		d.ws = float_buf_ensure_nocopy(&ctx->s->run.op_ws, hc_pre_ws_floats(&d), 64);
	status_code st = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int	   row	= call * seg.rows + t;
			buffer srow = batch_row_view(stream_all, row, flat);
			buffer xrow = batch_row_view(x_all, row, m->dim);
			buffer crow = batch_row_view(coef_all, row, coef_elems);
			d.stream	= &srow;
			d.x			= &xrow;
			d.coef		= &crow;
			st			= a->hc_pre(a, &d);
		}
	}
	return st;
}

status_code op_hc_post(exec_ctx *ctx) {
	backend *a = exec_layer_backend(ctx);
	if (!a->hc_post)
		return ERR_UNSUPPORTED;
	const model	 *m			 = ctx->m;
	exec_segments seg		 = exec_segments_of(ctx);
	buffer		 *sub_all	 = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *hc_all	 = exec_slot(ctx, ctx->op->out);
	buffer		 *coef_all	 = exec_slot(ctx, ctx->op->in[1]);
	const int	  flat		 = m->kda.hc_count * m->dim;
	const int	  coef_elems = m->kda.hc_count * (m->kda.hc_count + 1);
	status_code	  st		 = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int			 row	= call * seg.rows + t;
			buffer		 subrow = batch_row_view(sub_all, row, m->dim);
			buffer		 crow	= batch_row_view(coef_all, row, coef_elems);
			buffer		 hrow	= batch_row_view(hc_all, row, flat);
			hc_post_desc d		= {
				.dim  = m->dim,
				.n_hc = m->kda.hc_count,
				.sub  = &subrow,
				.coef = &crow,
				.hc	  = &hrow,
			};
			st = a->hc_post(a, &d);
		}
	}
	return st;
}

status_code op_kda(exec_ctx *ctx) {
	backend		   *a = exec_layer_backend(ctx);
	kvcache_hybrid *c = ctx->rows.cache ? ctx->rows.cache->hybrid : NULL;
	if (!a->kda || !c || !c->conv_buf || !c->state_buf)
		return ERR_UNSUPPORTED;
	const model			*m = ctx->m;
	const layer_weights *L = &m->layers[ctx->li];

	kda_desc d = {
		.dim			  = m->dim,
		.n_heads		  = m->n_heads,
		.head_dim		  = m->kda.head_dim,
		.conv_kernel	  = m->hybrid.conv_kernel,
		.gate_lower_bound = m->kda.gate_lower_bound,
		.norm_eps		  = m->norm_eps,
		.wq				  = weight_view(&L->wq),
		.wk				  = weight_view(&L->wk),
		.wv				  = weight_view(&L->wv),
		.wo				  = weight_view(&L->wo),
		.f_a			  = weight_view(&L->kda_f_a_w),
		.f_b			  = weight_view(&L->kda_f_b_w),
		.g_a			  = weight_view(&L->kda_g_a_w),
		.g_b			  = weight_view(&L->kda_g_b_w),
		.beta			  = weight_view(&L->ssm_beta_w),
		.conv_q			  = &L->kda_conv_q_w.buf,
		.conv_k			  = &L->kda_conv_k_w.buf,
		.conv_v			  = &L->kda_conv_v_w.buf,
		.dt_bias		  = &L->ssm_dt_b.buf,
		.ssm_a			  = &L->ssm_a.buf,
		.out_norm		  = &L->ssm_norm_w.buf,
	};
	if (backend_has_cap(a, BCAP_IS_HOST))
		d.ws = float_buf_ensure_nocopy(&ctx->s->run.op_ws, kda_ws_floats(&d), 64);

	exec_segments seg	  = exec_segments_of(ctx);
	buffer		 *x_all	  = exec_slot(ctx, ctx->op->in[0]);
	buffer		 *out_all = exec_slot(ctx, ctx->op->out);
	status_code	  st	  = OK;
	for (int call = 0; call < seg.n_calls && st == OK; call++) {
		kvcache		   *rcache = exec_segment_cache(ctx, call);
		kvcache_hybrid *rc	   = rcache ? rcache->hybrid : c;
		if (!rc || !rc->conv_buf || !rc->state_buf)
			return ERR_UNSUPPORTED;
		for (int t = 0; t < seg.rows && st == OK; t++) {
			int	   row	 = call * seg.rows + t;
			buffer xrow	 = batch_row_view(x_all, row, m->dim);
			buffer orow	 = batch_row_view(out_all, row, m->dim);
			d.x			 = &xrow;
			d.out		 = &orow;
			d.conv_state = &rc->conv_buf[ctx->li];
			d.state		 = &rc->state_buf[ctx->li];
			d.reset		 = exec_row_pos(ctx, row) == 0;
			st			 = a->kda(a, &d);
		}
	}
	return st;
}
