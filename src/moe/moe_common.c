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

static status_code moe_matmul_maybe_qonly(moe_expert_ctx *cx, const float *w, uint32_t wt,
										  const float *xb, float *out, int n_out, int tid) {
	if (cx->q8_gate_ok && wtype_to_q8type(wt) == cx->q8_gate_type) {
		buffer bw	   = {0};
		bw.handle	   = (void *)w;
		bw.host_ptr	   = (void *)w;
		buffer obuf	   = {0};
		obuf.handle	   = out;
		obuf.size	   = (size_t)n_out * sizeof(float);
		status_code s2 = cx->a->matmul_qonly(cx->a, &bw, wt, cx->xb_q8_gate, cx->q8_gate_type,
											 &obuf, n_out, cx->dim, 1);
		if (s2 == OK)
			return OK;
		if (s2 != ERR_UNSUPPORTED && s2 != ERR_INVALID_ARG)
			return s2;
	}
	return cx->a->matmul_thread_local(cx->a, w, wt, xb, out, n_out, cx->dim, tid);
}

status_code moe_expert_exec(moe_expert_ctx *cx, const moe_expert_slot *es, const float *xb,
							float weight, int tid) {
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
		st = moe_matmul_maybe_qonly(cx, es->gate_w, es->gate_type, xb, gu_h, cx->inter * 2, tid);
		if (st != OK)
			return st;
		moe_activate(act_h, gu_h, gu_h + cx->inter, cx->inter, es->gate_scale, es->up_scale,
					 cx->use_gelu);
	} else {
		st = moe_matmul_maybe_qonly(cx, es->gate_w, es->gate_type, xb, gate_h, cx->inter, tid);
		if (st != OK)
			return st;
		st = moe_matmul_maybe_qonly(cx, es->up_w, es->up_type, xb, up_h, cx->inter, tid);
		if (st != OK)
			return st;
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
