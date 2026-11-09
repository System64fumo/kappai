#include "rpc/server/session.h"

static status_code expert_body(backend *be, const moe_resident_expert *e, const buffer *x,
							   buffer *gate_v, buffer *up_v, buffer *act_v, buffer *y_v, int dim,
							   int inter, int m) {
	status_code s;
	if (e->gate_up_fused) {
		s = be->matmul_batch(be, e->gate_w, e->gate_type, x, gate_v, inter * 2, dim, m);
	} else {
		s = be->matmul_batch(be, e->gate_w, e->gate_type, x, gate_v, inter, dim, m);
		if (s == OK)
			s = be->matmul_batch(be, e->up_w, e->up_type, x, up_v, inter, dim, m);
	}
	if (s != OK)
		return s;

	float *gate = buffer_f32(gate_v);
	float *up	= buffer_f32(up_v);
	float *act	= buffer_f32(act_v);
	if (!e->gate_up_fused) {
		buffer g = buffer_host_view(be, gate, (size_t)m * inter * sizeof(float));
		buffer u = buffer_host_view(be, up, (size_t)m * inter * sizeof(float));
		buffer a = buffer_host_view(be, act, (size_t)m * inter * sizeof(float));
		s = be->moe_activate(be, &g, &u, &a, m * inter, e->gate_scale, e->up_scale, e->use_gelu);
	}
	for (int c = 0; e->gate_up_fused && c < m && s == OK; c++) {
		size_t row_bytes = (size_t)inter * sizeof(float);
		buffer g		 = buffer_host_view(be, gate + (size_t)c * 2 * inter, row_bytes);
		buffer u		 = buffer_host_view(be, gate + ((size_t)c * 2 + 1) * inter, row_bytes);
		buffer a		 = buffer_host_view(be, act + (size_t)c * inter, row_bytes);
		s = be->moe_activate(be, &g, &u, &a, inter, e->gate_scale, e->up_scale, e->use_gelu);
	}
	if (s != OK)
		return s;
	return be->matmul_batch(be, e->down_w, e->down_type, act_v, y_v, dim, inter, m);
}

static float *scratch_for(srv_session *s, size_t rows, int dim, int inter) {
	size_t per_row = (size_t)dim * 2 + (size_t)inter * 3;
	return float_buf_ensure(&s->moe_scratch, rows * per_row);
}

status_code resident_expert_ffn(srv_session *s, const buffer *x, buffer *out,
								const moe_resident_expert *e, int dim, int inter) {
	backend *be = s->be;
	if (!e->gate_w || !e->down_w || (!e->up_w && !e->gate_up_fused))
		return ERR_INVALID_ARG;
	float	   *base = scratch_for(s, 1, dim, inter);
	float	   *gate = base;
	float	   *up	 = base + inter;
	float	   *act	 = base + (size_t)inter * 2;
	float	   *y	 = base + (size_t)inter * 3;
	buffer		gv	 = buffer_host_view(be, gate, (size_t)inter * 2 * sizeof(float));
	buffer		uv = buffer_host_view(be, e->gate_up_fused ? gate + inter : up, (size_t)inter * 4);
	buffer		av = buffer_host_view(be, act, (size_t)inter * sizeof(float));
	buffer		yv = buffer_host_view(be, y, (size_t)dim * sizeof(float));
	status_code st = expert_body(be, e, x, &gv, &uv, &av, &yv, dim, inter, 1);
	if (st != OK)
		return st;
	st = be->scale_inplace(be, &yv, e->weight * e->down_scale, dim);
	return st == OK ? be->add_inplace(be, out, &yv, dim) : st;
}

status_code resident_experts_batch(srv_session *s, const buffer *xb, buffer *out, int n_rows,
								   int dim, int inter, int use_gelu, int n_experts,
								   const moe_resident_expert *experts, const int *counts,
								   const int *rows_packed, const float *weights_packed) {
	(void)n_rows;
	(void)use_gelu;
	backend *be		 = s->be;
	int		 max_cnt = 0;
	for (int i = 0; i < n_experts; i++)
		max_cnt = MAX(max_cnt, counts[i]);
	if (max_cnt == 0)
		return OK;

	size_t m_max = (size_t)max_cnt;
	float *base	 = scratch_for(s, m_max, dim, inter);
	float *xg	 = base;
	float *gate	 = xg + m_max * dim;
	float *act	 = gate + m_max * 2 * inter;
	float *y	 = act + m_max * inter;
	float *src	 = buffer_f32(xb);
	float *dst	 = buffer_f32(out);

	size_t packed = 0;
	for (int i = 0; i < n_experts; i++) {
		int cnt = counts[i];
		if (cnt <= 0)
			continue;
		const moe_resident_expert *e	= &experts[i];
		const int				  *rows = rows_packed + packed;
		const float				  *w	= weights_packed + packed;
		packed += (size_t)cnt;
		if (!e->gate_w || !e->down_w)
			return ERR_INVALID_ARG;
		for (int c = 0; c < cnt; c++)
			memcpy(xg + (size_t)c * dim, src + (size_t)rows[c] * dim, (size_t)dim * sizeof(float));

		buffer xv = buffer_host_view(be, xg, (size_t)cnt * dim * sizeof(float));
		buffer gv = buffer_host_view(be, gate, (size_t)cnt * 2 * inter * sizeof(float));
		buffer uv =
			buffer_host_view(be, gate + (size_t)cnt * inter, (size_t)cnt * inter * sizeof(float));
		buffer		av = buffer_host_view(be, act, (size_t)cnt * inter * sizeof(float));
		buffer		yv = buffer_host_view(be, y, (size_t)cnt * dim * sizeof(float));
		status_code st = expert_body(be, e, &xv, &gv, &uv, &av, &yv, dim, inter, cnt);
		if (st != OK)
			return st;
		for (int c = 0; c < cnt; c++) {
			float		 scale = w[c] * e->down_scale;
			float		*o	   = dst + (size_t)rows[c] * dim;
			const float *yr	   = y + (size_t)c * dim;
			for (int j = 0; j < dim; j++)
				o[j] += scale * yr[j];
		}
	}
	return OK;
}
