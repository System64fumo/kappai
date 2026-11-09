#include "rocket.h"

const void		*rkt_native_key;
static rkt_priv *native_owner;

void rkt_native_release(rkt_priv *p) {
	rkt_native_key = NULL;
	p->nat_f32	   = NULL;
}

void rkt_native_settle(void) {
	rkt_priv *p = native_owner;
	if (!rkt_native_key || !p || !p->nat_f32)
		return;
	float *dst = p->nat_f32;
	int	   n   = p->nat_n;
	int	   m   = p->nat_m;
	rkt_native_release(p);
	for (int r = 0; r < m; r++)
		rkt_convert_from_f16(dst + (size_t)r * n, p->nat_act + (size_t)r * n, (size_t)n);
}

int rkt_native_pending(const rkt_priv *p, const buffer *x, int k, int m) {
	return rkt_native_key && rkt_ptr_raw(x) == rkt_native_key && p == native_owner &&
		   p->nat_n == k && p->nat_m == m;
}

int rkt_native_matches(const rkt_priv *p, const buffer *x, const rkt_weight *w, int k, int m) {
	return rkt_native_pending(p, x, k, m) && w->k == k && w->fmt == RKT_FMT_F16 && !w->quant &&
		   !w->host_copy && w->n_kchunks > 0;
}

static void native_set_pending(rkt_priv *p, float *key, int n, int m) {
	p->nat_f32	   = key;
	p->nat_n	   = n;
	p->nat_m	   = m;
	native_owner   = p;
	rkt_native_key = key;
}

status_code rkt_native_ffn_activate(rkt_priv *p, rkt_lut_kind kind, const float *gate,
									int gate_stride, const float *up, int up_stride, float *out,
									int n, int m) {
	if (p->npu_failed || kind == RKT_LUT_NONE)
		return ERR_UNSUPPORTED;
	rkt_native_settle();
	if (!rkt_ensure_capacity((void **)&p->nat_act, &p->nat_act_cap, (size_t)m * n,
							 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	pthread_mutex_lock(&p->lock);
	status_code s = rkt_lut_activate_rows_f16(p, &p->lut, kind, gate, gate_stride, up, up_stride,
											  p->nat_act, n, n, m);
	pthread_mutex_unlock(&p->lock);
	if (s == OK)
		native_set_pending(p, out, n, m);
	return s;
}

void rkt_native_free(rkt_priv *p) {
	if (native_owner == p) {
		rkt_native_release(p);
		native_owner = NULL;
	}
	free(p->nat_gate);
	free(p->nat_up);
	free(p->nat_act);
}

static int native_staging_ready(rkt_priv *p, size_t staged, size_t act_elems) {
	return rkt_ensure_capacity((void **)&p->nat_gate, &p->nat_gate_cap, staged, sizeof(uint16_t)) &&
		   rkt_ensure_capacity((void **)&p->nat_up, &p->nat_up_cap, staged, sizeof(uint16_t)) &&
		   rkt_ensure_capacity((void **)&p->nat_act, &p->nat_act_cap, act_elems, sizeof(uint16_t));
}

static status_code native_gate_act_run(rkt_priv *p, const rkt_weight *gw, int grow0,
									   const rkt_weight *uw, int urow0, const float *x, int n,
									   int m, rkt_lut_kind kind, int stride) {
	status_code s = rkt_dense_matmul_enqueue_f16(p, gw, grow0, n, x, p->nat_gate, stride, m);
	if (s == OK)
		s = rkt_dense_matmul_enqueue_f16(p, uw, urow0, n, x, p->nat_up, stride, m);
	if (s != OK) {
		rkt_queue_abort(p);
		return s;
	}
	s = rkt_queue_flush(p);
	if (s != OK)
		return s;
	return rkt_lut_activate_native_mul(p, &p->lut, kind, p->nat_gate, p->nat_up, p->nat_act, n, n,
									   m);
}

status_code rkt_native_gate_act(rkt_priv *p, const rkt_weight *gw, int grow0, const rkt_weight *uw,
								int urow0, const float *x, float *act, int n, int m,
								rkt_lut_kind kind) {
	if (p->npu_failed || gw->fmt != RKT_FMT_F16 || uw->fmt != RKT_FMT_F16)
		return ERR_UNSUPPORTED;
	rkt_native_settle();
	int	   stride = (n + RKT_LUT_CHUNK - 1) / RKT_LUT_CHUNK * RKT_LUT_CHUNK;
	size_t staged = (size_t)m * stride;
	if (!native_staging_ready(p, staged, (size_t)m * n))
		return ERR_OUT_OF_MEMORY;
	for (int r = 0; r < m; r++)
		memset(p->nat_gate + (size_t)r * stride + n, 0, (size_t)(stride - n) * sizeof(uint16_t));

	pthread_mutex_lock(&p->lock);
	status_code s = native_gate_act_run(p, gw, grow0, uw, urow0, x, n, m, kind, stride);
	pthread_mutex_unlock(&p->lock);
	if (s != OK)
		return s;
	native_set_pending(p, act, n, m);
	return OK;
}
