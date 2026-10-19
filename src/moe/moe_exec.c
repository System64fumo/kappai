#include "moe/moe_exec.h"

#include "common.h"
#include "compute.h"
#include "log.h"
#include "model.h"
#include "moe/moe_common.h"
#include "profile.h"

#include <string.h>

static status_code device_submit(moe_exec_target *t, const moe_exec_request *rq, const int *ids) {
	struct model *m	 = rq->m;
	backend		 *eb = t->be;
	status_code	  st = moe_stream_resolve(m, rq->layer, ids, rq->K, t->slots);
	if (st != OK)
		return st;
	const buffer *x_dev = rq->xb_dev;
	buffer		 *o_dev = rq->out_dev;
	if (!t->direct) {
		expert_dev_entry *e = expert_dev_scratch_get(&rq->s->expert_dev, eb, (size_t)rq->dim);
		if (!e) {
			moe_stream_release_slots(m, rq->layer, t->slots, rq->K);
			return ERR_OUT_OF_MEMORY;
		}
		x_dev = &e->x;
		o_dev = &e->out;
		st	  = eb->buffer_write_f32(eb, &e->x, rq->x, rq->dim);
	}
	if (st == OK)
		st = eb->scale_inplace(eb, o_dev, 0.0f, rq->dim);
	for (int k = 0; st == OK && k < rq->K; k++) {
		const moe_expert_slot *sl = &t->slots[k];
		if (sl->eid < 0 || !moe_slot_present(sl))
			continue;
		if (!sl->dev_ready || sl->dev_gate.owner != eb) {
			backend_report_host_fallback(
				eb, "moe_expert_ffn", HFB_BUF_HOST_RESIDENT,
				"layer %d expert %d slot %s (dev_ready=%d)", rq->layer, sl->eid,
				sl->dev_ready ? "owned by another backend" : "not resident on device",
				(int)sl->dev_ready);
			st = ERR_UNSUPPORTED;
			break;
		}
		moe_resident_expert ge = {
			.gate_w		   = &sl->dev_gate,
			.up_w		   = &sl->dev_up,
			.down_w		   = &sl->dev_down,
			.gate_type	   = sl->gate_type,
			.up_type	   = sl->up_type,
			.down_type	   = sl->down_type,
			.gate_up_fused = sl->gate_up_fused,
			.use_gelu	   = rq->use_gelu,
			.gate_scale	   = sl->gate_scale,
			.up_scale	   = sl->up_scale,
			.down_scale	   = sl->down_scale,
			.weight		   = rq->weights[k],
		};
		st = eb->moe_expert_ffn(eb, x_dev, o_dev, &ge, rq->dim, rq->I);
	}
	if (st != OK) {
		if (t->direct)
			(void)eb->scale_inplace(eb, o_dev, 0.0f, rq->dim);
		moe_stream_release_slots(m, rq->layer, t->slots, rq->K);
		return st;
	}
	if (eb->submit)
		eb->submit(eb);
	return OK;
}

static status_code device_collect(moe_exec_target *t, const moe_exec_request *rq, float *out) {
	if (t->direct) {
		moe_stream_release_slots(rq->m, rq->layer, t->slots, rq->K);
		return OK;
	}
	expert_dev_entry *e	 = expert_dev_scratch_get(&rq->s->expert_dev, t->be, (size_t)rq->dim);
	status_code		  st = e ? OK : ERR_OUT_OF_MEMORY;
	if (st == OK)
		st = t->be->buffer_read_f32(t->be, &e->out, out, rq->dim);
	moe_stream_release_slots(rq->m, rq->layer, t->slots, rq->K);
	return st;
}

static const moe_exec_ops device_ops = {.submit = device_submit, .collect = device_collect};

static void collect_owners(struct model *m, int layer, const int *expert_ids, int K,
						   backend **owners, int *host_ids, int *any) {
	for (int k = 0; k < K; k++) {
		owners[k]	= moe_stream_expert_owner(m, layer, expert_ids[k]);
		host_ids[k] = owners[k] ? -1 : expert_ids[k];
		*any |= owners[k] != NULL;
	}
}

int moe_exec_direct_capable(const backend *a, const buffer *xb, const buffer *out) {
	return a && xb && out && backend_has_cap(a, BCAP_IN_ORDER_QUEUE) && a->moe_expert_ffn &&
		   a->scale_inplace && a->add_inplace && a->submit && xb->owner == a && out->owner == a;
}

int moe_exec_staged_capable(const backend *a, const buffer *out) {
	return a && out && backend_has_cap(a, BCAP_IN_ORDER_QUEUE) && a->copy_buffer &&
		   a->add_inplace && out->owner == a;
}

status_code moe_exec_upload_host(struct compute_scratch *s, backend *a, buffer *out,
								 const float *host, int dim, int accumulate) {
	buffer *stage = &s->moe_stage[s->moe_stage_next];
	size_t	bytes = (size_t)dim * sizeof(float);
	if (!stage->owner || stage->size < bytes) {
		if (stage->owner)
			stage->owner->buffer_free(stage->owner, stage);
		memset(stage, 0, sizeof(*stage));
		status_code as = a->buffer_alloc_scratch(a, bytes, stage);
		if (as != OK)
			return as;
	}
	s->moe_stage_next = (s->moe_stage_next + 1) % MOE_STAGE_RING;
	status_code st	  = a->buffer_write_f32(a, stage, host, dim);
	if (st != OK)
		return st;
	return accumulate ? a->add_inplace(a, out, stage, dim) : a->copy_buffer(a, stage, out, dim);
}

static void submit_target(moe_exec_request *rq, moe_exec_pending *pend, backend *eb, backend *local,
						  const int *expert_ids, backend **owners, int *host_ids, int *n_dev) {
	int ids[MOE_MAX_TOPK];
	for (int k = 0; k < rq->K; k++)
		ids[k] = owners[k] == eb ? expert_ids[k] : -1;

	status_code		 st	  = ERR_UNSUPPORTED;
	moe_exec_target *slot = NULL;
	if (pend->n < MOE_EXEC_MAX_TARGETS) {
		slot			 = &pend->targets[pend->n];
		slot->ops		 = &device_ops;
		slot->be		 = eb;
		slot->direct	 = eb == local && rq->out_dev && !pend->n_direct;
		slot->slots		 = rq->s->moe_slot_dev + (size_t)pend->n * MOE_MAX_TOPK;
		profile_scope ps = profile_begin(&rq->s->prof, STAGE_MOE_DEV_SUBMIT);
		st				 = slot->ops->submit(slot, rq, ids);
		profile_end(&rq->s->prof, &ps);
	}
	for (int k = 0; k < rq->K; k++) {
		if (owners[k] != eb)
			continue;
		owners[k] = NULL;
		if (st == OK)
			(*n_dev)++;
		else
			host_ids[k] = expert_ids[k];
	}
	if (st == OK) {
		pend->n++;
		if (slot->direct)
			pend->n_direct++;
		return;
	}
	WARN("moe expert ffn on '%s' layer %d failed (%s) -- falling back to cpu",
		 eb->name ? eb->name : "?", rq->layer, status_str(st));
}

int moe_exec_dispatch(moe_exec_request *rq, backend *local, const buffer *xb, const int *expert_ids,
					  int *host_ids, moe_exec_pending *pend) {
	struct compute_scratch *s = rq->s;
	backend				   *owners[MOE_MAX_TOPK];
	int						any = 0;

	profile_scope tier_ps = profile_begin(&s->prof, STAGE_MOE_TIER);
	moe_stream_tier_maintain(rq->m, rq->layer);
	profile_end(&s->prof, &tier_ps);

	collect_owners(rq->m, rq->layer, expert_ids, rq->K, owners, host_ids, &any);
	if (!any)
		return 0;

	if (!rq->x) {
		float		 *xh	  = float_buf_ensure(&s->moe_xb_f, (size_t)rq->dim);
		profile_scope sync_ps = profile_begin(&s->prof, STAGE_MOE_XB_SYNC);
		status_code	  rs	  = local->buffer_read_f32(local, xb, xh, rq->dim);
		profile_end(&s->prof, &sync_ps);
		if (rs != OK) {
			memcpy(host_ids, expert_ids, (size_t)rq->K * sizeof(int));
			return 0;
		}
		rq->x = xh;
	}
	if (!s->moe_slot_dev)
		s->moe_slot_dev =
			xcalloc((size_t)MOE_MAX_TOPK * MOE_EXEC_MAX_TARGETS, sizeof(moe_expert_slot));

	int n_dev = 0;
	for (int pass = 0; pass < 2; pass++)
		for (int k = 0; k < rq->K; k++)
			if (owners[k] && (owners[k] == local) == (pass == 1))
				submit_target(rq, pend, owners[k], local, expert_ids, owners, host_ids, &n_dev);
	return n_dev;
}

status_code moe_exec_gather(moe_exec_pending *pend, const moe_exec_request *rq, int discard) {
	struct compute_scratch *s	= rq->s;
	profile_scope			ps	= profile_begin(&s->prof, STAGE_MOE_DEV_COLLECT);
	float				   *acc = float_buf_ensure(&s->expert_dev.acc, (size_t)rq->dim);
	float				   *tmp = float_buf_ensure(&s->expert_dev.tmp, (size_t)rq->dim);
	status_code				st	= OK;
	memset(acc, 0, (size_t)rq->dim * sizeof(float));
	for (int g = 0; g < pend->n; g++) {
		moe_exec_target *t	= &pend->targets[g];
		status_code		 gs = t->ops->collect(t, rq, tmp);
		if (gs != OK) {
			ERROR("moe expert ffn readback on '%s' layer %d failed (%s)",
				  t->be->name ? t->be->name : "?", rq->layer, status_str(gs));
			if (st == OK)
				st = gs;
			continue;
		}
		if (!discard && !t->direct)
			for (int d = 0; d < rq->dim; d++)
				acc[d] += tmp[d];
	}
	pend->n		   = 0;
	pend->n_direct = 0;
	profile_end(&s->prof, &ps);
	return st;
}
