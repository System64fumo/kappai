#include "recipe/internal.h"

typedef struct {
	struct model *m;
	backend		 *host;
	backend		 *dev;
	int			  total_ple;
	int			  dim;
	int			  n_embd_per_layer;
	int			  n_layers;
	float		  n_embd_sqrt;
	float		  inv_sqrt_ple;
	float		  eps;
} ple_build_plan;

typedef struct {
	struct model			   *m;
	const struct layer_weights *L;
	int							li;
	int							n_embd_per_layer;
	int							total_ple;
	int							inj_dim;
	float						eps;
	const buffer			   *gate_w;
	const buffer			   *proj_w;
	const buffer			   *post_norm_w;
	uint32_t					gate_type;
	uint32_t					proj_type;
} ple_inject_plan;

enum { PLE_BUILD_AUTO, PLE_BUILD_DEVICE, PLE_BUILD_HOST };

static status_code ple_ensure_norm_w(backend *a, compute_scratch *s, model *m,
									 int n_embd_per_layer) {
	if (s->model.ple_proj_norm_w_uploaded)
		return OK;
	status_code st = buffer_ensure_scratch(a, &s->model.ple_proj_norm_w,
										   (size_t)n_embd_per_layer * sizeof(float));
	if (st != OK)
		return st;
	st = a->buffer_write_f32(a, &s->model.ple_proj_norm_w,
							 m->layer_dims.per_layer_proj_norm_w.host_ptr, n_embd_per_layer);
	if (st != OK)
		return st;
	s->model.ple_proj_norm_w_uploaded = 1;
	return OK;
}

static void ple_build_plan_fill(ple_build_plan *p, struct model *m, backend *dev) {
	p->m				= m;
	p->host				= NULL;
	p->dev				= dev;
	p->n_embd_per_layer = m->layer_dims.n_embd_per_layer;
	p->n_layers			= m->n_layers;
	p->total_ple		= p->n_embd_per_layer * p->n_layers;
	p->dim				= m->dim;
	p->n_embd_sqrt		= sqrtf((float)p->n_embd_per_layer);
	p->inv_sqrt_ple		= m->dim_sqrt > 0 ? 1.0f / m->dim_sqrt : 0.0f;
	p->eps				= m->norm_eps;
}

static int ple_build_dev_caps_ok(backend *a, const struct model *m, int batched) {
	int has_proj = batched ? (a->matmul_batch != NULL) : (a->matmul != NULL);
	int has_norm = batched ? (a->rmsnorm_batch != NULL) : (a->rmsnorm != NULL);
	return a->embd_lookup && has_proj && a->scale_inplace && has_norm && a->ple_combine &&
		   matmul_type_ok(a, m->layer_dims.per_layer_model_proj.type);
}

int ple_build_dev_path_ok(backend *a, const struct model *m, int batched) {
	return ple_build_dev_caps_ok(a, m, batched) && m->layer_dims.per_layer_model_proj.buf.handle &&
		   m->layer_dims.per_layer_tok_embd.buf.handle;
}

int ple_device_path_possible(const struct model *m) {
	return !m->mixed_backend_mode && m->backend && ple_build_dev_caps_ok(m->backend, m, 0) &&
		   ple_build_dev_caps_ok(m->backend, m, 1);
}

static status_code ple_host_decode_row(const ple_build_plan *p, int token, float *ple_row) {
	size_t row_stride	= ggml_row_size(p->m->layer_dims.per_layer_tok_embd.type, p->total_ple);
	const uint8_t *embd = (const uint8_t *)p->m->layer_dims.per_layer_tok_embd.host_ptr +
						  ((size_t)token * row_stride);
	backend		  *dq	= (p->dev && p->dev->dequant_row) ? p->dev : NULL;
	if (!dq) {
		ple_build_plan *mp = (ple_build_plan *)p;
		if (!mp->host)
			mp->host = backend_host();
		dq = mp->host;
	}
	if (!dq || !dq->dequant_row)
		return ERR_UNSUPPORTED;
	status_code st =
		dq->dequant_row(dq, p->m->layer_dims.per_layer_tok_embd.type, embd, p->total_ple, ple_row);
	if (st != OK)
		return st;
	for (int i = 0; i < p->total_ple; i++)
		ple_row[i] *= p->n_embd_sqrt;
	return OK;
}

static status_code ple_norm_layers(backend *a, const buffer *norm_w, buffer *x, int n_layers,
								   int slice_elems, float eps) {
	if (a->rmsnorm_batch && n_layers > 1)
		return a->rmsnorm_batch(a, x, norm_w, x, slice_elems, eps, n_layers);
	for (int l = 0; l < n_layers; l++) {
		buffer xs	   = *x;
		xs.offset	   = x->offset + (size_t)l * slice_elems * sizeof(float);
		status_code st = a->rmsnorm(a, &xs, norm_w, &xs, slice_elems, eps);
		if (st != OK)
			return st;
	}
	return OK;
}

static void ple_host_project_row(const ple_build_plan *p, const float *inp_row, float *proj_row) {
	host_matmul_generic(p->m->layer_dims.per_layer_model_proj.host_ptr,
						p->m->layer_dims.per_layer_model_proj.type, inp_row, proj_row, p->total_ple,
						p->dim);
}

static void ple_host_finish(const ple_build_plan *p, float *ple, float *proj, int n_rows) {
	ple_build_plan *mp = (ple_build_plan *)p;
	if (!mp->host)
		mp->host = backend_host();
	backend *host	 = mp->host;
	buffer proj_view = buffer_host_view(host, proj, (size_t)n_rows * p->total_ple * sizeof(float));
	host->scale_inplace(host, &proj_view, p->inv_sqrt_ple, n_rows * p->total_ple);
	buffer w_view = buffer_host_view(host, (void *)p->m->layer_dims.per_layer_proj_norm_w.host_ptr,
									 (size_t)p->n_embd_per_layer * sizeof(float));
	for (int row = 0; row < n_rows; row++) {
		buffer row_buf = buffer_host_view(host, proj + (size_t)row * p->total_ple,
										  (size_t)p->total_ple * sizeof(float));
		ple_norm_layers(host, &w_view, &row_buf, p->n_layers, p->n_embd_per_layer, p->eps);
	}
	buffer ple_view	  = buffer_host_view(host, ple, (size_t)n_rows * p->total_ple * sizeof(float));
	buffer proj_view2 = buffer_host_view(host, proj, (size_t)n_rows * p->total_ple * sizeof(float));
	host->ple_combine(host, &ple_view, &proj_view2, n_rows * p->total_ple, 0.70710678118654752f);
}

static int ple_inject_plan_fill(ple_inject_plan *p, struct model *m, int li) {
	if (li < 0 || li >= m->n_layers)
		return 0;
	p->m				= m;
	p->L				= &m->layers[li];
	p->li				= li;
	p->n_embd_per_layer = m->layer_dims.n_embd_per_layer;
	p->total_ple		= p->n_embd_per_layer * m->n_layers;
	p->inj_dim			= m->dim;
	p->eps				= m->norm_eps;
	p->gate_w			= &p->L->ple_inp_gate_w.buf;
	p->proj_w			= &p->L->ple_proj_w.buf;
	p->post_norm_w		= &p->L->ple_post_norm_w.buf;
	p->gate_type		= p->L->ple_inp_gate_w.type;
	p->proj_type		= p->L->ple_proj_w.type;
	return 1;
}

static status_code ple_build_run(exec_ctx *ctx, int mode) {

	if (!ctx->m->has_per_layer_embeddings)
		return OK;
	struct model  *m = ctx->m;
	backend		  *a = exec_layer_backend(ctx);
	ple_build_plan plan;
	ple_build_plan_fill(&plan, m, a);
	const int	total_ple = plan.total_ple;
	const int	n_layers  = plan.n_layers;
	const int	dim		  = plan.dim;
	const int	n_rows	  = ctx->rows.n_rows;
	status_code st;

	st = buffer_ensure_scratch(a, &ctx->bs->ple_all, (size_t)n_rows * total_ple * sizeof(float));
	if (st != OK)
		return st;

	int dev_path_ok =
		mode == PLE_BUILD_AUTO ? ple_build_dev_path_cached(m, a, 1) : mode == PLE_BUILD_DEVICE;

	if (dev_path_ok) {
		st = embd_fill_rows(a, &m->layer_dims.per_layer_tok_embd.buf,
							m->layer_dims.per_layer_tok_embd.type, ctx->bs->tokens,
							&ctx->bs->ple_all, n_rows, total_ple, 1.0f);
		if (st != OK)
			return st;
		st = a->scale_inplace(a, &ctx->bs->ple_all, plan.n_embd_sqrt, n_rows * total_ple);
		if (st != OK)
			return st;

		buffer *xb = batch_slot(ctx->bs, ctx->op->in[0]);

		size_t n_proj	= (size_t)n_rows * total_ple;
		buffer proj_buf = {0};
		if (backend_has_cap(a, BCAP_IS_HOST)) {
			float_buf_ensure(&ctx->bs->ple_proj, n_proj);
			proj_buf = bs_cached_view(&ctx->bs->ple_proj, a, n_proj);
		} else {
			st = buffer_ensure_scratch(a, &ctx->bs->ple_proj_dev, n_proj * sizeof(float));
			if (st != OK)
				return st;
			proj_buf = ctx->bs->ple_proj_dev;
		}

		st = a->matmul_batch(a, &m->layer_dims.per_layer_model_proj.buf,
							 m->layer_dims.per_layer_model_proj.type, xb, &proj_buf, total_ple, dim,
							 n_rows);
		if (st != OK)
			return st;

		st = a->scale_inplace(a, &proj_buf, plan.inv_sqrt_ple, n_rows * total_ple);
		if (st != OK)
			return st;

		st = ple_ensure_norm_w(a, ctx->s, m, plan.n_embd_per_layer);
		if (st != OK)
			return st;

		for (int row = 0; row < n_rows; row++) {
			buffer row_view = proj_buf;
			row_view.offset = (size_t)row * total_ple * sizeof(float);
			st = ple_norm_layers(a, &ctx->s->model.ple_proj_norm_w, &row_view, n_layers,
								 plan.n_embd_per_layer, plan.eps);
			if (st != OK)
				return st;
		}

		return a->ple_combine(a, &ctx->bs->ple_all, &proj_buf, n_rows * total_ple,
							  0.70710678118654752f);
	}

	ensure_sync(a);

	float_buf_ensure(&ctx->bs->ple_buf, (size_t)n_rows * total_ple);
	float_buf_ensure(&ctx->bs->ple_proj, (size_t)n_rows * total_ple);
	float *ple = ctx->bs->ple_buf.p;
	backend_report_host_fallback(a, "ple_build", HFB_WEIGHT_TYPE,
								 "per-layer token embeddings are dequantized on the host (cpu) "
								 "before the projection runs");
	for (int row = 0; row < n_rows; row++) {
		int token = ctx->bs->tokens ? ctx->bs->tokens[row] : 0;
		st		  = ple_host_decode_row(&plan, token, ple + (size_t)row * total_ple);
		if (st != OK)
			return st;
	}

	float *ple_proj	 = float_buf_ensure(&ctx->s->ple.proj_host, (size_t)n_rows * total_ple);
	float *inpL_host = float_buf_ensure(&ctx->s->ple.inpL_host, (size_t)n_rows * dim);

	buffer	*xb		 = batch_slot(ctx->bs, ctx->op->in[0]);
	backend *x_owner = xb->owner;
	st				 = x_owner->buffer_read_f32(x_owner, xb, inpL_host, n_rows * dim);
	if (st != OK)
		return st;

	backend_report_host_fallback(a, "ple_proj_matmul", HFB_OP_NOT_NATIVE,
								 "device backend lacks a batched per-layer projection path; "
								 "projection matmul executes on host (cpu)");
	for (int row = 0; row < n_rows; row++)
		ple_host_project_row(&plan, inpL_host + (size_t)row * dim,
							 ple_proj + (size_t)row * total_ple);

	ple_host_finish(&plan, ple, ple_proj, n_rows);

	if (a->copy_buffer) {
		st = a->buffer_write_f32(a, &ctx->bs->ple_all, ple, n_rows * total_ple);
		if (st != OK)
			return st;
	}

	return OK;
}

status_code ple_build_batch(exec_ctx *ctx) {
	return ple_build_run(ctx, PLE_BUILD_AUTO);
}

status_code ple_build_device(exec_ctx *ctx) {
	return ple_build_run(ctx, PLE_BUILD_DEVICE);
}

status_code ple_build_host(exec_ctx *ctx) {
	return ple_build_run(ctx, PLE_BUILD_HOST);
}

status_code ple_proj_inject_batch(exec_ctx *ctx) {

	if (ctx->li < 0)
		return ERR_INVALID_ARG;
	if (!ctx->m->has_per_layer_embeddings)
		return OK;
	ple_inject_plan plan;
	if (!ple_inject_plan_fill(&plan, ctx->m, ctx->li))
		return ERR_INVALID_ARG;
	backend		  *a				= exec_layer_backend(ctx);
	const int	   n_embd_per_layer = plan.n_embd_per_layer;
	const int	   total_ple		= plan.total_ple;
	const int	   inj_dim			= plan.inj_dim;
	const int	   n_rows			= ctx->rows.n_rows;
	const float	   eps				= plan.eps;
	batch_scratch *ple_src_bs		= ctx->s->bs;

	float_buf_ensure(&ctx->bs->ple_inp, (size_t)n_rows * n_embd_per_layer);
	float_buf_ensure(&ctx->bs->ple_slice, (size_t)n_rows * n_embd_per_layer);

	status_code st;
	size_t		n_ple	  = (size_t)n_rows * n_embd_per_layer;
	buffer		ple_inp_b = {0}, ple_slice_b = {0};
	uint32_t	caps = exec_caps(ctx, ple_inject_caps);
	int			heap = (caps & PLE_CAP_HEAP) != 0;
	int			slice_in_place =
		heap && n_rows == 1 && ple_src_bs->ple_all.handle && ple_src_bs->ple_all.owner == a;
	if (heap) {
		st = buffer_ensure_scratch(a, &ctx->bs->ple_inp_dev, n_ple * sizeof(float));
		if (st != OK)
			return st;
		ple_inp_b = ctx->bs->ple_inp_dev;
		if (slice_in_place) {
			ple_slice_b = ple_src_bs->ple_all;
			ple_slice_b.offset += (size_t)ctx->li * n_embd_per_layer * sizeof(float);
		} else {
			st = buffer_ensure_scratch(a, &ctx->bs->ple_slice_dev, n_ple * sizeof(float));
			if (st != OK)
				return st;
			ple_slice_b = ctx->bs->ple_slice_dev;
		}
	} else {
		ple_inp_b	= bs_cached_view(&ctx->bs->ple_inp, a, n_ple);
		ple_slice_b = bs_cached_view(&ctx->bs->ple_slice, a, n_ple);
	}

	if (!slice_in_place && ple_src_bs->ple_all.handle) {
		for (int row = 0; row < n_rows; row++) {
			buffer ple_src = ple_src_bs->ple_all;
			ple_src.offset +=
				((size_t)row * total_ple + (size_t)ctx->li * n_embd_per_layer) * sizeof(float);
			buffer dst_row = batch_row_view(&ple_slice_b, row, n_embd_per_layer);
			st = compute_copy_buffer_cross(ctx->s, &ple_src, &dst_row, n_embd_per_layer);
			if (st != OK) {
				ERROR("ple_proj_inject_batch: copy_buffer_cross(ple_src->slice) failed (st=%d)",
					  (int)st);
				goto ple_cleanup;
			}
		}
	} else if (!slice_in_place) {
		for (int row = 0; row < n_rows; row++) {
			float *ple_row = ple_src_bs->ple_buf.p + (size_t)row * total_ple +
							 (size_t)ctx->li * n_embd_per_layer;
			buffer dst_row = batch_row_view(&ple_slice_b, row, n_embd_per_layer);
			st			   = a->buffer_write_f32(a, &dst_row, ple_row, n_embd_per_layer);
			if (st != OK)
				goto ple_cleanup;
		}
	}

	buffer		 *xb2b = batch_slot(ctx->bs, ctx->op->in[0]);
	profile_scope pps  = profile_begin(&ctx->s->prof, ctx->op->stage);
	profile_weight_bytes(&ctx->s->prof, ctx->op->stage, plan.gate_type, inj_dim, n_embd_per_layer);
	profile_weight_bytes(&ctx->s->prof, ctx->op->stage, plan.proj_type, n_embd_per_layer, inj_dim);
	st = a->matmul_batch(a, plan.gate_w, plan.gate_type, xb2b, &ple_inp_b, n_embd_per_layer,
						 inj_dim, n_rows);
	if (st != OK) {
		ERROR("ple_proj_inject_batch: matmul_batch(ple_inp_gate) failed (st=%d)", (int)st);
		goto ple_cleanup;
	}

	buffer *aob = batch_slot(ctx->bs, ctx->op->in[1]);
	if (caps & PLE_CAP_FUSED_DOWN) {
		st = a->matmul_ffn_down_batch(a, plan.proj_w, plan.proj_type, &ple_inp_b, &ple_slice_b, aob,
									  inj_dim, n_embd_per_layer, ACTIVATION_GELU, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: matmul_ffn_down_batch failed (st=%d)", (int)st);
			goto ple_cleanup;
		}
	} else {
		st = a->ffn_activate_batch(a, &ple_inp_b, &ple_slice_b, &ple_inp_b, n_embd_per_layer,
								   ACTIVATION_GELU, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: ffn_activate_batch failed (st=%d)", (int)st);
			goto ple_cleanup;
		}
		st = a->matmul_batch(a, plan.proj_w, plan.proj_type, &ple_inp_b, aob, inj_dim,
							 n_embd_per_layer, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: matmul_batch(ple_proj) failed (st=%d)", (int)st);
			goto ple_cleanup;
		}
	}
	profile_end(&ctx->s->prof, &pps);

	if (caps & PLE_CAP_FUSED_NORM) {
		float out_scale = 1.0f;
		if (ctx->m->arch_info->has_layer_output_scale && ctx->li >= 0 &&
			ctx->li < ctx->m->n_layers && ctx->m->layers[ctx->li].layer_out_scale != 1.0f)
			out_scale = ctx->m->layers[ctx->li].layer_out_scale;
		st = a->rmsnorm_add_batch(a, aob, plan.post_norm_w, xb2b, xb2b, inj_dim, eps, out_scale,
								  n_rows);
		if (st != OK)
			ERROR("ple_proj_inject_batch: rmsnorm_add_batch failed (st=%d)", (int)st);
	} else {
		st = a->rmsnorm_batch(a, aob, plan.post_norm_w, aob, inj_dim, eps, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: rmsnorm_batch failed (st=%d)", (int)st);
			goto ple_cleanup;
		}

		st = a->add_batch(a, xb2b, aob, inj_dim, n_rows);
		if (st != OK) {
			ERROR("ple_proj_inject_batch: add_batch failed (st=%d)", (int)st);
			goto ple_cleanup;
		}
		float out_scale = 1.0f;
		if (ctx->m->arch_info->has_layer_output_scale && ctx->li >= 0 &&
			ctx->li < ctx->m->n_layers && ctx->m->layers[ctx->li].layer_out_scale != 1.0f)
			out_scale = ctx->m->layers[ctx->li].layer_out_scale;
		if (out_scale != 1.0f) {
			if (a->scale_inplace)
				st = a->scale_inplace(a, xb2b, out_scale, inj_dim * n_rows);
			else
				st = scale_inplace_fallback(ctx->s, a, xb2b, inj_dim * n_rows, out_scale);
			if (st != OK)
				ERROR("ple_proj_inject_batch: scale fallback failed (st=%d)", (int)st);
		}
	}

ple_cleanup:
	return st;
}
