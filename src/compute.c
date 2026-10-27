#include "compute.h"
#include "log.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ROPE_TABLE_BLOCK 4096

typedef struct {
	float *cos_out;
	float *sin_out;
	int	   n_ctx;
	int	   half;
	double theta_scale;
} rope_table_job;

void compute_scratch_init(compute_scratch *s) {
	memset(s, 0, sizeof(*s));
}

static void rope_table_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	rope_table_job *j = ctx;
	for (int col = begin; col < end; col++) {
		double freq		= pow(j->theta_scale, (double)col);
		double step_cos = cos(freq);
		double step_sin = sin(freq);
		float *cos_col	= j->cos_out + col;
		float *sin_col	= j->sin_out + col;
		for (int base = 0; base < j->n_ctx; base += ROPE_TABLE_BLOCK) {
			int	   stop	 = base + ROPE_TABLE_BLOCK > j->n_ctx ? j->n_ctx : base + ROPE_TABLE_BLOCK;
			double angle = freq * (double)base;
			double c	 = cos(angle);
			double s	 = sin(angle);
			cos_col[(ptrdiff_t)base * j->half] = (float)c;
			sin_col[(ptrdiff_t)base * j->half] = (float)s;
			for (int pos = base + 1; pos < stop; pos++) {
				double nc						  = (c * step_cos) - (s * step_sin);
				s								  = (c * step_sin) + (s * step_cos);
				c								  = nc;
				cos_col[(ptrdiff_t)pos * j->half] = (float)c;
				sin_col[(ptrdiff_t)pos * j->half] = (float)s;
			}
		}
	}
}

static void build_rope_table(float *cos_out, float *sin_out, int n_ctx, int head_dim, float theta) {
	const int	   half		   = head_dim / 2;
	const double   theta_scale = pow(theta, -2.0 / (double)head_dim);
	rope_table_job job		   = {.cos_out	   = cos_out,
								  .sin_out	   = sin_out,
								  .n_ctx	   = n_ctx,
								  .half		   = half,
								  .theta_scale = theta_scale};

	tpool *pool = tpool_fetch_pool();
	if (pool && tpool_n_threads(pool) > 1 && half >= tpool_n_threads(pool) &&
		tpool_current_tid() < 0) {
		tpool_parallel_for(pool, half, 1, rope_table_chunk, &job);
		return;
	}
	rope_table_chunk(0, half, -1, &job);
}

static void free_buf(buffer *b) {
	if (!b->owner)
		return;
	b->owner->buffer_free(b->owner, b);
}

static void scratch_free_device_buffers(compute_scratch *s) {
	free_buf(&s->model.ple_proj_norm_w);
	for (int i = 0; i < MOE_STAGE_RING; i++)
		free_buf(&s->moe.stage[i]);

	for (int d = 0; d < s->n_mirrors; d++) {
		scratch_mirror *mir = &s->mirror[d];
		if (mir->bs) {
			batch_scratch_free(mir->bs);
			free(mir->bs);
			mir->bs = NULL;
		}
		mir->be = NULL;
	}
	s->n_mirrors	   = 0;
	s->mirrors_alloced = 0;
	free(s->run.transfer_buf);
	s->run.transfer_buf		= NULL;
	s->run.transfer_buf_cap = 0;
	free(s->logits_host);
	s->logits_host = NULL;
}

static void scratch_free_host_buffers(compute_scratch *s) {
	free(s->model.rope_cos);
	free(s->model.rope_sin);
	free(s->model.rope_cos_swa);
	free(s->model.rope_sin_swa);
	free(s->ple.proj_host.p);
	free(s->ple.inpL_host.p);
	s->ple.proj_host.p	 = NULL;
	s->ple.proj_host.cap = 0;
	s->ple.inpL_host.p	 = NULL;
	s->ple.inpL_host.cap = 0;
	free(s->moe.all_scratch.p);
	free(s->moe.all_outs.p);
	free(s->moe.scratch.p);
	free(s->moe.xb_f.p);
	s->moe.all_scratch.p = NULL;
	s->moe.all_outs.p	 = NULL;
	s->moe.scratch.p	 = NULL;
	s->moe.xb_f.p		 = NULL;
	free(s->moe.slot_buf);
	s->moe.slot_buf = NULL;
	free(s->moe.slot_dev);
	s->moe.slot_dev = NULL;
	expert_dev_scratch_release(&s->moe.expert_dev);

	float_buf *buffers[] = {
		&s->hybrid.host,	  &s->hybrid.host2,	   &s->hybrid.gdn_ws,  &s->run.op_ws,
		&s->run.stage_a,	  &s->run.stage_b,	   &s->run.scale_tmp,  &s->hybrid.gdn_z,
		&s->hybrid.gdn_alpha, &s->hybrid.gdn_beta, &s->hybrid.gdn_out,
	};
	for (size_t i = 0; i < ARRAY_LEN(buffers); i++) {
		free(buffers[i]->p);
		*buffers[i] = (float_buf){0};
	}

	if (s->bs) {
		batch_scratch_free(s->bs);
		free(s->bs);
		s->bs = NULL;
	}
	free(s->run.last_only);
	s->run.last_only	 = NULL;
	s->run.last_only_cap = 0;
}

static void expert_dev_entry_release(expert_dev_entry *e) {
	if (e->x.owner)
		e->x.owner->buffer_free(e->x.owner, &e->x);
	if (e->out.owner)
		e->out.owner->buffer_free(e->out.owner, &e->out);
	memset(e, 0, sizeof(*e));
}

void expert_dev_scratch_release(expert_dev_scratch *ds) {
	for (int i = 0; i < EXPERT_DEV_MAX; i++)
		expert_dev_entry_release(&ds->entries[i]);
	free(ds->acc.p);
	free(ds->tmp.p);
	memset(ds, 0, sizeof(*ds));
}

expert_dev_entry *expert_dev_scratch_get(expert_dev_scratch *ds, backend *eb, size_t floats) {
	expert_dev_entry *slot = NULL;
	for (int i = 0; i < EXPERT_DEV_MAX; i++) {
		if (ds->entries[i].x.owner == eb) {
			slot = &ds->entries[i];
			break;
		}
		if (!slot && !ds->entries[i].x.owner)
			slot = &ds->entries[i];
	}
	if (!slot)
		return NULL;
	if (slot->x.owner == eb && slot->cap >= floats)
		return slot;
	expert_dev_entry_release(slot);
	size_t bytes = floats * sizeof(float);
	if (eb->buffer_alloc_scratch(eb, bytes, &slot->x) != OK ||
		eb->buffer_alloc_scratch(eb, bytes, &slot->out) != OK) {
		expert_dev_entry_release(slot);
		return NULL;
	}
	slot->cap = floats;
	return slot;
}

void compute_scratch_free(compute_scratch *s) {
	scratch_free_device_buffers(s);
	scratch_free_host_buffers(s);
	memset(s, 0, sizeof(*s));
}

status_code compute_scratch_ensure(compute_scratch *s, const model *m, int n_ctx) {
	if (s->model.last_model == m && s->model.allocated_n_ctx >= n_ctx)
		return OK;

	scratch_free_device_buffers(s);
	scratch_free_host_buffers(s);

	s->model.backend = m->backend;

	s->logits_host = xmalloc((size_t)m->vocab_size * sizeof(float));

	if (m->arch_info->is_moe)
		s->moe.slot_buf = xcalloc(MOE_MAX_TOPK, sizeof(*s->moe.slot_buf));

	if (m->arch_info->has_variable_layer_dims) {
		const int half_swa	  = m->layer_dims.head_dim_swa / 2;
		s->model.rope_cos_swa = xmalloc((size_t)n_ctx * half_swa * sizeof(float));
		s->model.rope_sin_swa = xmalloc((size_t)n_ctx * half_swa * sizeof(float));
		build_rope_table(s->model.rope_cos_swa, s->model.rope_sin_swa, n_ctx,
						 m->layer_dims.head_dim_swa, m->layer_dims.rope_theta_swa);

		const int half_global = m->layer_dims.head_dim_global / 2;
		s->model.rope_cos	  = xmalloc((size_t)n_ctx * half_global * sizeof(float));
		s->model.rope_sin	  = xmalloc((size_t)n_ctx * half_global * sizeof(float));
		build_rope_table(s->model.rope_cos, s->model.rope_sin, n_ctx, m->layer_dims.head_dim_global,
						 m->layer_dims.rope_theta_global);

	} else {
		const int rope_head_dim =
			m->arch_info->is_mla ? m->mla.qk_rope
								 : (m->arch_info->is_hybrid_recurrent ? m->rope_dim : m->head_dim);
		const int half = rope_head_dim / 2;
		if (half > 0) {
			s->model.rope_cos = xmalloc((size_t)n_ctx * half * sizeof(float));
			s->model.rope_sin = xmalloc((size_t)n_ctx * half * sizeof(float));
			build_rope_table(s->model.rope_cos, s->model.rope_sin, n_ctx, rope_head_dim,
							 m->rope_theta);
		}
	}

	s->model.allocated_n_ctx = n_ctx;
	s->model.last_model		 = m;

	return OK;
}
status_code compute_forward(model *m, kvcache *cache, compute_scratch *s, int token, int pos,
							int flash_attn, float *logits_out) {
	status_code st = compute_scratch_ensure(s, m, cache->n_ctx);
	if (st != OK)
		return st;
	return compute_forward_recipe(m, cache, s, token, pos, flash_attn, logits_out);
}

status_code compute_forward_batch(model *m, kvcache *cache, compute_scratch *s,
								  const int32_t *tokens, int n_tokens, int pos_start,
								  int flash_attn, float *logits_out) {
	long w = m->sliding_window;
	if (w > 0) {
		long slack = w / 4 < 8 ? 8 : w / 4;
		long cap   = w + slack;
		if (pos_start + n_tokens > cap && n_tokens > slack + 1) {
			ERROR("batch of %d tokens exceeds the %ld-token sliding-window slack (window %ld, "
				  "capacity %ld): rows early in this batch would lose window history. "
				  "Reduce the prefill chunk size to %ld.",
				  n_tokens, slack, w, cap, slack + 1);
			return ERR_INVALID_ARG;
		}
	}
	status_code st = compute_scratch_ensure(s, m, cache->n_ctx);
	if (st != OK)
		return st;
	return compute_forward_batch_recipe(m, cache, s, tokens, n_tokens, pos_start, flash_attn,
										logits_out);
}

status_code compute_forward_decode_batch(model *m, kvcache **caches, compute_scratch *s,
										 const int32_t *tokens, const int *positions,
										 const uint8_t *needs_logits, int n_seqs, int flash_attn,
										 float *logits_out) {
	status_code st = compute_scratch_ensure(s, m, caches[0]->n_ctx);
	if (st != OK)
		return st;
	return compute_forward_decode_batch_recipe(m, caches, s, tokens, positions, needs_logits,
											   n_seqs, flash_attn, logits_out);
}

void compute_set_layer_progress_cb(compute_scratch *s, layer_progress_cb cb, void *ud) {
	s->layer_cb	   = cb;
	s->layer_cb_ud = ud;
}

static status_code ensure_transfer_buf(compute_scratch *s, size_t need_floats) {
	if (s->run.transfer_buf_cap >= need_floats)
		return OK;
	s->run.transfer_buf		= xrealloc(s->run.transfer_buf, need_floats * sizeof(float));
	s->run.transfer_buf_cap = need_floats;
	return OK;
}

int compute_device_index(const compute_scratch *s, const backend *b) {
	if (b == s->model.backend)
		return 0;
	for (int i = 0; i < s->n_mirrors; i++)
		if (s->mirror[i].be == b)
			return i + 1;
	return -1;
}

status_code compute_scratch_ensure_mirror(compute_scratch *s, const model *m) {
	if (s->mirrors_alloced)
		return OK;
	if (!m || !m->mixed_backend_mode)
		return OK;

	for (int i = 0; i < m->n_layers; i++) {
		backend *lb = model_layer_backend(m, i);
		if (!lb || compute_device_index(s, lb) >= 0)
			continue;
		if (s->n_mirrors >= SCRATCH_MAX_MIRRORS) {
			ERROR("compute: more than %d devices hold layers", SCRATCH_MAX_MIRRORS + 1);
			return ERR_UNSUPPORTED;
		}
		s->mirror[s->n_mirrors++].be = lb;
	}

	s->mirrors_alloced = 1;
	return OK;
}

status_code compute_copy_buffer_cross(compute_scratch *s, const buffer *src, buffer *dst, int n) {
	if (!src || !dst || n <= 0)
		return ERR_INVALID_ARG;

	backend *src_owner = src->owner;
	backend *dst_owner = dst->owner;

	if (src_owner == dst_owner) {
		if (src_owner && src_owner->copy_buffer)
			return src_owner->copy_buffer(src_owner, src, dst, n);
		if (src_owner && src_owner->buffer_read_f32 && src_owner->buffer_write_f32) {
			status_code st = ensure_transfer_buf(s, (size_t)n);
			if (st != OK)
				return st;
			st = src_owner->buffer_read_f32(src_owner, src, s->run.transfer_buf, n);
			if (st != OK)
				return st;
			return src_owner->buffer_write_f32(src_owner, dst, s->run.transfer_buf, n);
		}
		if (src->host_ptr && dst->host_ptr) {
			memcpy((void *)dst->host_ptr, src->host_ptr, (size_t)n * sizeof(float));
			return OK;
		}
		return ERR_UNSUPPORTED;
	}

	ensure_sync(src_owner);

	const int	 dst_direct = buffer_is_host_resident(dst);
	const float *from		= NULL;
	if (buffer_is_host_resident(src)) {
		from = buffer_f32(src);
	} else if (src_owner && src_owner->buffer_read_f32) {
		float *to = dst_direct ? buffer_f32(dst) : NULL;
		if (!to) {
			status_code st = ensure_transfer_buf(s, (size_t)n);
			if (st != OK)
				return st;
			to = s->run.transfer_buf;
		}
		status_code st = src_owner->buffer_read_f32(src_owner, src, to, n);
		if (st != OK || dst_direct)
			return st;
		from = to;
	} else if (src->host_ptr) {
		from = src->host_ptr;
	} else {
		return ERR_UNSUPPORTED;
	}

	if (dst_direct) {
		memcpy(buffer_f32(dst), from, (size_t)n * sizeof(float));
		return OK;
	}
	if (dst_owner && dst_owner->buffer_write_f32)
		return dst_owner->buffer_write_f32(dst_owner, dst, from, n);
	if (dst->host_ptr) {
		memcpy((void *)dst->host_ptr, from, (size_t)n * sizeof(float));
		return OK;
	}
	return ERR_UNSUPPORTED;
}
