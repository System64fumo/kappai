#include "recipe/internal.h"

typedef struct {
	backend		  *a;
	const buffer  *w;
	uint32_t	   w_type;
	const int32_t *tokens;
	buffer		  *xs;
	int			   dim;
	float		   scale;
	_Atomic int	  *first_err;
} embd_fill_job;

static int bs_ensure_detached(batch_scratch *bs, backend *owner, uint8_t slot, size_t n_elems) {
	bs_pair *p	   = &bs->pair[slot];
	size_t	 bytes = n_elems * sizeof(float);

	bs->own[slot] = 1;
	if (!backend_has_cap(owner, BCAP_IS_HOST)) {
		if (p->b.owner == owner && p->b.handle && p->b.size >= bytes)
			return 0;
		if (p->b.handle && p->b.owner && !backend_has_cap(p->b.owner, BCAP_IS_HOST))
			p->b.owner->buffer_free(p->b.owner, &p->b);
		memset(&p->b, 0, sizeof(p->b));
		owner->buffer_alloc_scratch(owner, bytes, &p->b);
		return 1;
	}
	if (p->b.handle && p->b.owner && p->b.owner != owner &&
		!backend_has_cap(p->b.owner, BCAP_IS_HOST)) {
		p->b.owner->buffer_free(p->b.owner, &p->b);
		memset(&p->b, 0, sizeof(p->b));
	}
	int	   existed = p->b.handle != NULL;
	size_t old_cap = p->fb.cap;
	float_buf_ensure_nocopy(&p->fb, n_elems, 64);
	int fresh	  = !existed || p->fb.cap > old_cap;
	p->b.handle	  = p->fb.p;
	p->b.host_ptr = p->fb.p;
	p->b.size	  = bytes;
	p->b.offset	  = 0;
	p->b.owner	  = owner;
	return fresh;
}

int bs_ensure_slot(batch_scratch *bs, backend *owner, uint8_t slot, size_t n_elems) {
	size_t bytes = n_elems * sizeof(float);
	size_t cap	 = bs->layout ? bs->layout->cap[slot] : 0;
	if (cap && !bs->own[slot]) {
		if (bytes <= cap)
			return 0;
		WARN(
			"batch scratch: slot %u needs %zu B but the plan reserves %zu B; allocating it outside "
			"the arena",
			(unsigned)slot, bytes, cap);
		memset(&bs->pair[slot].b, 0, sizeof(bs->pair[slot].b));
	}
	return bs_ensure_detached(bs, owner, slot, n_elems);
}

static void bs_release_slot(batch_scratch *bs, uint8_t slot) {
	bs_pair *p = &bs->pair[slot];
	if (bs->own[slot] && p->b.owner && !backend_has_cap(p->b.owner, BCAP_IS_HOST))
		p->b.owner->buffer_free(p->b.owner, &p->b);
	free(p->fb.p);
	memset(p, 0, sizeof(*p));
	bs->own[slot] = 0;
}

static void bs_arena_ensure(batch_scratch *bs, backend *owner, size_t bytes) {
	if (bs->arena_bytes >= bytes)
		return;
	if (backend_has_cap(owner, BCAP_IS_HOST)) {
		float_buf_ensure_nocopy(&bs->arena_host, bytes / sizeof(float), 64);
	} else {
		if (bs->arena.owner)
			bs->arena.owner->buffer_free(bs->arena.owner, &bs->arena);
		memset(&bs->arena, 0, sizeof(bs->arena));
		if (owner->buffer_alloc_scratch(owner, bytes, &bs->arena) != OK) {
			ERROR("batch scratch: arena allocation of %zu B failed", bytes);
			bs->arena_bytes = 0;
			return;
		}
	}
	bs->arena_bytes = bytes;
	DEBUG("batch scratch: arena %zu B (%zu MiB) for %d rows", bytes, bytes >> 20,
		  bs->layout->bucket);
}

static buffer bs_arena_view(const batch_scratch *bs, backend *owner, size_t offset, size_t bytes) {
	if (backend_has_cap(owner, BCAP_IS_HOST))
		return buffer_host_view(owner, (char *)bs->arena_host.p + offset, bytes);
	return buffer_slice(&bs->arena, offset, bytes);
}

static void bs_bind_views(batch_scratch *bs, backend *owner) {
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (!bs->layout->cap[slot])
			continue;
		if (bs->own[slot])
			bs_release_slot(bs, (uint8_t)slot);
		bs->pair[slot].b =
			bs_arena_view(bs, owner, bs->layout->offset[slot], bs->layout->cap[slot]);
	}
}

void batch_scratch_free(batch_scratch *bs) {
	if (!bs)
		return;
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++)
		bs_release_slot(bs, (uint8_t)slot);
	if (bs->arena.owner)
		bs->arena.owner->buffer_free(bs->arena.owner, &bs->arena);
	free(bs->arena_host.p);
	free(bs->moe_pack);
	free(bs->copy_tmp.p);
	free(bs->moe_router_ids);
	free(bs->moe_router_w);
	free(bs->moe_union_ids);
	free(bs->moe_union_rows);
	free(bs->moe_union_kidx);
	free(bs->moe_union_offsets);
	free(bs->moe_union_cursor);
	free(bs->moe_union_count);
	free(bs->moe_expert_union_idx);
	free(bs->moe_expert_seen_gen);
	free(bs->moe_union_order);
	free(bs->moe_union_sorted_ids);
	free(bs->moe_order_pairs);
	free(bs->moe_expert_out.p);
	free(bs->moe_out.p);
	free(bs->moe_router_logits.p);
	free(bs->moe_router_inp.p);
	if (bs->moe_router_logits_dev.owner)
		bs->moe_router_logits_dev.owner->buffer_free(bs->moe_router_logits_dev.owner,
													 &bs->moe_router_logits_dev);
	if (bs->moe_router_inp_dev.owner)
		bs->moe_router_inp_dev.owner->buffer_free(bs->moe_router_inp_dev.owner,
												  &bs->moe_router_inp_dev);
	free(bs->moe_xb_f.p);
	free(bs->moe_gather_x.p);
	free(bs->moe_exp_gu.p);
	free(bs->moe_up_scratch.p);
	free(bs->moe_exp_act.p);
	free(bs->moe_exp_y.p);
	free(bs->moe_fused_host.p);
	free(bs->moe_fact_out.p);
	free(bs->moe_zeros.p);
	free(bs->moe_per_token_slots);
	free(bs->moe_union_slots);
	free(bs->ple_buf.p);
	if (bs->ple_all.owner)
		bs->ple_all.owner->buffer_free(bs->ple_all.owner, &bs->ple_all);
	free(bs->ple_inp.p);
	free(bs->ple_proj.p);
	free(bs->ple_slice.p);
	if (bs->ple_inp_dev.owner)
		bs->ple_inp_dev.owner->buffer_free(bs->ple_inp_dev.owner, &bs->ple_inp_dev);
	if (bs->ple_slice_dev.owner)
		bs->ple_slice_dev.owner->buffer_free(bs->ple_slice_dev.owner, &bs->ple_slice_dev);
	if (bs->ple_proj_dev.owner)
		bs->ple_proj_dev.owner->buffer_free(bs->ple_proj_dev.owner, &bs->ple_proj_dev);
	memset(bs, 0, sizeof(*bs));
}

buffer *batch_slot(batch_scratch *bs, uint8_t slot) {
	if (!graph_slot_valid(slot))
		return NULL;
	return &bs->pair[slot].b;
}

buffer batch_row_view(const buffer *whole, int row, int row_elems) {
	buffer v	 = *whole;
	size_t delta = (size_t)row * row_elems * sizeof(float);
	v.offset	 = whole->offset + delta;
	if (whole->size >= delta)
		v.size = whole->size - delta;
	return v;
}

static void embd_fill_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	embd_fill_job *j = ctx;
	for (int row = begin; row < end; row++) {
		buffer		xrow = batch_row_view(j->xs, row, j->dim);
		int			tok	 = j->tokens ? j->tokens[row] : 0;
		status_code st	 = j->a->embd_lookup(j->a, j->w, j->w_type, tok, j->dim, &xrow);
		if (st != OK) {
			atomic_store_explicit(j->first_err, st, memory_order_relaxed);
			continue;
		}
		if (j->scale != 1.0f) {
			float *xf = batch_buf_ptr(&xrow);
			for (int i = 0; i < j->dim; i++)
				xf[i] *= j->scale;
		}
	}
}

status_code embd_fill_rows(backend *a, const buffer *w, uint32_t w_type, const int32_t *tokens,
						   buffer *xs, int n_rows, int dim, float scale) {
	tpool *pool = backend_get_pool(a);
	if (pool && n_rows >= 16 && backend_has_cap(a, BCAP_IS_HOST) && tpool_current_tid() < 0 &&
		tpool_n_threads(pool) > 1) {
		_Atomic int	  err = OK;
		embd_fill_job j	  = {.a			= a,
							 .w			= w,
							 .w_type	= w_type,
							 .tokens	= tokens,
							 .xs		= xs,
							 .dim		= dim,
							 .scale		= scale,
							 .first_err = &err};
		tpool_parallel_for(pool, n_rows, 4, embd_fill_chunk, &j);
		return atomic_load_explicit(&err, memory_order_relaxed);
	}

	for (int row = 0; row < n_rows; row++) {
		buffer		xrow = batch_row_view(xs, row, dim);
		int			tok	 = tokens ? tokens[row] : 0;
		status_code st	 = a->embd_lookup(a, w, w_type, tok, dim, &xrow);
		if (st != OK)
			return st;
	}
	if (scale == 1.0f)
		return OK;
	if (a->scale_inplace)
		return a->scale_inplace(a, xs, scale, n_rows * dim);
	ensure_sync(a);
	float *xf = batch_buf_ptr(xs);
	for (int i = 0; i < n_rows * dim; i++)
		xf[i] *= scale;
	return OK;
}

buffer bs_cached_view(float_buf *fb, backend *owner, size_t n_elems) {
	buffer b   = {0};
	b.handle   = fb->p;
	b.host_ptr = fb->p;
	b.size	   = n_elems * sizeof(float);
	b.owner	   = owner;
	return b;
}

static void batch_scratch_bind(model *m, batch_scratch *bs, backend *a, int n_rows) {
	bs->layout		= mem_layout_get(&bs->layouts, m->recipe->mem, n_rows);
	bs->logits_slot = m->recipe->logits_slot;
	bs_arena_ensure(bs, a, bs->layout->total);
	bs_bind_views(bs, a);
}

batch_scratch *batch_scratch_prepare(model *m, compute_scratch *s, int n_rows) {
	for (int d = 0; d < s->n_mirrors; d++) {
		batch_scratch **slot = &s->mirror[d].bs;
		if (!*slot)
			*slot = xcalloc(1, sizeof(batch_scratch));
		batch_scratch_bind(m, *slot, s->mirror[d].be, n_rows);
	}
	if (!s->bs)
		s->bs = xcalloc(1, sizeof(batch_scratch));
	batch_scratch_bind(m, s->bs, m->backend, n_rows);
	return s->bs;
}

buffer *compute_logits_device(compute_scratch *s) {
	return &s->bs->pair[s->bs->logits_slot].b;
}

void compute_scratch_reserve_rows(model *m, compute_scratch *s, int n_rows) {
	if (n_rows < 1 || !recipe_is_decode_batchable(m))
		return;
	batch_scratch_prepare(m, s, n_rows);
}
