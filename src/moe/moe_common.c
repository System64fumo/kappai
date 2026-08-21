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
			st = cx->a->matmul_thread_local(cx->a, es->gate_w, es->gate_type, xb, gu_h,
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
			st = cx->a->matmul_thread_local(cx->a, es->gate_w, es->gate_type, xb, gate_h, cx->inter,
											cx->dim, tid);
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
			st = cx->a->matmul_thread_local(cx->a, es->up_w, es->up_type, xb, up_h, cx->inter,
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
