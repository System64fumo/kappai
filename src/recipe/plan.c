#include "recipe/plan.h"
#include "log.h"
#include "recipe/capability.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define ALL_SLOTS ((1u << RECIPE_SLOT_MAX) - 1u)
#define HYBRID_SLOTS                                                                               \
	((1u << RECIPE_SLOT_HYB_PROJ) | (1u << RECIPE_SLOT_HYB_GATE) | (1u << RECIPE_SLOT_HYB_ALPHA) | \
	 (1u << RECIPE_SLOT_HYB_BETA))
#define REPORT_ROWS 512

typedef struct {
	uint8_t	 slot;
	cg_range range;
} hidden_use;

typedef struct {
	uint8_t conflict[RECIPE_SLOT_MAX][RECIPE_SLOT_MAX];
	uint8_t seen[RECIPE_SLOT_MAX];
	uint8_t pinned[RECIPE_SLOT_MAX];
} interference;

void mem_dims_init(mem_dims *d, const struct model *m) {
	int max_head_dim = m->head_dim;
	int max_kv_heads = m->n_kv_heads;
	int max_inter	 = m->intermediate;
	for (int li = 0; li < m->n_layers; li++) {
		max_head_dim = MAX(max_head_dim, model_layer_head_dim(m, li));
		max_kv_heads = MAX(max_kv_heads, model_layer_kv_heads(m, li));
		max_inter	 = MAX(max_inter, model_layer_intermediate(m, li));
	}
	if (m->arch_info->is_mla) {
		max_head_dim = MAX(max_head_dim, m->mla.qk_head);
		max_kv_heads = MAX(max_kv_heads, m->n_heads);
	}
	if (m->moe.n_experts > 0 || m->moe.moe_intermediate > 0) {
		max_inter = MAX(max_inter, m->moe.moe_intermediate);
		max_inter = MAX(max_inter, m->moe.moe_intermediate * m->moe.n_shared_experts);
	}

	*d				= (mem_dims){0};
	d->dim			= m->dim;
	d->max_head_dim = max_head_dim;
	d->max_kv_heads = max_kv_heads;
	d->q_out		= m->n_heads * max_head_dim;
	d->kv_out		= max_kv_heads * max_head_dim;
	d->max_inter	= max_inter;
	d->ffn_act		= MAX(max_inter, m->dim);
	d->gate_row		= max_inter;
	d->up_row		= max_inter;
	d->attn_buf		= MAX(m->dim, d->q_out);
	if (m->arch_info->is_mla) {
		d->attn_buf = MAX(d->attn_buf, m->n_heads * m->mla.v_head);
		d->gate_row = MAX(d->gate_row, m->mla.q_lora);
		d->up_row	= MAX(d->up_row, m->mla.kv_lora + m->mla.qk_rope);
	}
	if (m->arch_info->has_hyper_connections) {
		d->hc_flat = m->kda.hc_count * m->dim;
		d->hc_coef = m->kda.hc_count * (m->kda.hc_count + 1);
	}
	if (m->arch_info->is_hybrid_recurrent) {
		d->attn_buf	 = MAX(d->attn_buf, m->hybrid.value_dim);
		d->hyb_proj	 = model_hybrid_proj_size(m);
		d->hyb_gate	 = model_hybrid_gate_size(m);
		d->hyb_heads = m->hybrid.n_value_heads;
	}
}

static size_t slot_row_elems(const mem_dims *d, uint8_t slot) {
	switch (slot) {
	case RECIPE_SLOT_FFN_GATE:
		return (size_t)d->gate_row;
	case RECIPE_SLOT_FFN_UP:
		return (size_t)d->up_row;
	case RECIPE_SLOT_HYB_PROJ:
		return (size_t)d->hyb_proj;
	case RECIPE_SLOT_HYB_GATE:
		return (size_t)d->hyb_gate;
	case RECIPE_SLOT_HYB_ALPHA:
	case RECIPE_SLOT_HYB_BETA:
		return (size_t)d->hyb_heads;
	default:
		break;
	}
	switch (graph_slot_size_kind(slot)) {
	case BS_SIZE_ATTN_BUF:
		return (size_t)d->attn_buf;
	case BS_SIZE_Q_OUT:
		return (size_t)d->q_out;
	case BS_SIZE_KV_OUT:
		return (size_t)d->kv_out;
	case BS_SIZE_INTER:
		return (size_t)d->max_inter;
	case BS_SIZE_INTER_X2:
		return 2 * (size_t)d->max_inter;
	case BS_SIZE_ACT:
		return (size_t)d->ffn_act;
	case BS_SIZE_HC:
		return (size_t)d->hc_flat;
	case BS_SIZE_HC_COEF:
		return (size_t)d->hc_coef;
	default:
		return 0;
	}
}

static int ranges_overlap(const cg_range *a, const cg_range *b) {
	return a->start <= b->end && b->start <= a->end;
}

static void add_conflict(interference *x, uint8_t a, uint8_t b) {
	if (a == b)
		return;
	x->conflict[a][b] = 1;
	x->conflict[b][a] = 1;
}

static int has_storage(const cg_graph *g, const cg_range *ranges, int v) {
	return ranges[v].end >= 0 && g->values[v].view_of == CG_NONE;
}

static void collect_value_conflicts(interference *x, const cg_graph *g, const cg_range *ranges) {
	for (int v = 0; v < g->n_values; v++) {
		if (!has_storage(g, ranges, v))
			continue;
		x->seen[g->values[v].slot] = 1;
		for (int w = v + 1; w < g->n_values; w++)
			if (has_storage(g, ranges, w) && ranges_overlap(&ranges[v], &ranges[w]))
				add_conflict(x, g->values[v].slot, g->values[w].slot);
	}
}

static int node_local_scratch(op_kind kind) {
	return kind == OP_MATMUL || kind == OP_MATMUL_MULTI || kind == OP_MATMUL_RESIDUAL ||
		   kind == OP_MATMUL_FFN_DOWN;
}

static int collect_hidden_uses(const cg_graph *g, hidden_use *uses) {
	cg_range spans[RECIPE_SLOT_MAX][CG_SECTION_COUNT];
	int		 n_uses = 0;
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++)
		for (int section = 0; section < CG_SECTION_COUNT; section++)
			spans[slot][section] = (cg_range){.start = INT_MAX, .end = -1};
	for (int i = 0; i < g->n_nodes; i++) {
		const cg_node *node	  = &g->nodes[i];
		uint32_t	   hidden = recipe_op_batch_extra_mask(node->kind);
		for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
			if (!((hidden >> slot) & 1u))
				continue;
			if (node_local_scratch(node->kind)) {
				uses[n_uses++] =
					(hidden_use){.slot = (uint8_t)slot, .range = {.start = i, .end = i}};
				continue;
			}
			cg_range *span = &spans[slot][node->section];
			span->start	   = MIN(span->start, i);
			span->end	   = MAX(span->end, i);
		}
	}
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++)
		for (int section = 0; section < CG_SECTION_COUNT; section++)
			if (spans[slot][section].end >= 0)
				uses[n_uses++] = (hidden_use){.slot = (uint8_t)slot, .range = spans[slot][section]};
	return n_uses;
}

static void collect_use_conflicts(interference *x, const cg_graph *g, const cg_range *ranges,
								  const hidden_use *uses, int n_uses) {
	for (int u = 0; u < n_uses; u++) {
		x->seen[uses[u].slot] = 1;
		for (int v = 0; v < g->n_values; v++)
			if (has_storage(g, ranges, v) && ranges_overlap(&ranges[v], &uses[u].range))
				add_conflict(x, uses[u].slot, g->values[v].slot);
		for (int w = u + 1; w < n_uses; w++)
			if (ranges_overlap(&uses[u].range, &uses[w].range))
				add_conflict(x, uses[u].slot, uses[w].slot);
	}
}

static void pin_swapped_slots(interference *x, const cg_graph *g, const cg_node *n) {
	for (int j = 0; j < n->n_in + n->n_out; j++)
		x->pinned[g->values[j < n->n_in ? n->in[j] : n->out[j - n->n_in]].slot] = 1;
}

static void collect_graph(interference *x, const cg_graph *g) {
	size_t	  max_uses = (size_t)g->n_nodes * RECIPE_SLOT_MAX + RECIPE_SLOT_MAX * CG_SECTION_COUNT;
	cg_range *ranges   = graph_value_ranges(g);
	hidden_use *uses   = xmalloc((max_uses + 1) * sizeof(hidden_use));
	int			n_uses = collect_hidden_uses(g, uses);
	collect_value_conflicts(x, g, ranges);
	collect_use_conflicts(x, g, ranges, uses, n_uses);
	for (int i = 0; i < g->n_nodes; i++)
		if (g->nodes[i].kind == OP_SWAP)
			pin_swapped_slots(x, g, &g->nodes[i]);
	free(uses);
	free(ranges);
}

static int aliasable(const interference *x, uint8_t slot) {
	return x && x->seen[slot] && !x->pinned[slot];
}

static int can_join(const mem_plan *p, const interference *x, uint8_t slot, uint8_t rep) {
	if (!aliasable(x, rep))
		return 0;
	for (int k = 0; k < RECIPE_SLOT_MAX; k++)
		if (k != slot && p->rep[k] == rep && x->conflict[k][slot])
			return 0;
	return 1;
}

static int order_slots(const mem_plan *p, uint8_t *order) {
	int n = 0;
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (!p->planned[slot])
			continue;
		int j = n++;
		while (j > 0 && p->row_elems[order[j - 1]] < p->row_elems[slot]) {
			order[j] = order[j - 1];
			j--;
		}
		order[j] = (uint8_t)slot;
	}
	return n;
}

static void group_slots(mem_plan *p, const interference *x) {
	uint8_t order[RECIPE_SLOT_MAX];
	int		n = order_slots(p, order);
	for (int i = 0; i < n; i++) {
		if (!aliasable(x, order[i]))
			continue;
		for (int j = 0; j < i; j++) {
			if (p->rep[order[j]] != order[j] || !can_join(p, x, order[i], order[j]))
				continue;
			p->rep[order[i]] = order[j];
			break;
		}
	}
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (!p->planned[slot])
			continue;
		uint8_t rep				= p->rep[slot];
		p->group_row_elems[rep] = MAX(p->group_row_elems[rep], p->row_elems[slot]);
		p->n_slots++;
		p->n_buffers += rep == slot;
	}
}

static void report(const mem_plan *p, int aliasing) {
	size_t shared = 0, separate = 0;
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (!p->planned[slot])
			continue;
		separate += p->row_elems[slot] * sizeof(float);
		shared += p->rep[slot] == slot ? p->group_row_elems[slot] * sizeof(float) : 0;
	}
	DEBUG("memory plan: %d slots in %d buffers (aliasing %s), %zu B/row vs %zu B/row unaliased, "
		  "%zu MiB at %d rows",
		  p->n_slots, p->n_buffers, aliasing ? "on" : "off", shared, separate,
		  (shared * REPORT_ROWS) >> 20, REPORT_ROWS);
}

mem_plan *mem_plan_build(const cg_set *graphs, const struct model *m) {
	mem_plan *p	   = xcalloc(1, sizeof(*p));
	uint32_t  mask = graphs ? graph_set_batch_slot_mask(graphs) : ALL_SLOTS;
	if (!mask)
		mask = ALL_SLOTS;
	mem_dims_init(&p->dims, m);
	if (m->arch_info->is_hybrid_recurrent)
		mask |= HYBRID_SLOTS;
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		p->rep[slot]	   = (uint8_t)slot;
		p->row_elems[slot] = slot_row_elems(&p->dims, (uint8_t)slot);
		p->planned[slot]   = ((mask >> slot) & 1u) && p->row_elems[slot] > 0;
	}

	interference *x = NULL;
	if (graphs) {
		x = xcalloc(1, sizeof(*x));
		for (int g = 0; g < graphs->n_graphs; g++)
			collect_graph(x, graphs->graphs[g]);
	}
	group_slots(p, x);
	report(p, x != NULL);
	free(x);
	return p;
}

void mem_plan_free(mem_plan *p) {
	free(p);
}

int mem_rows_bucket(int n_rows) {
	if (n_rows > 64)
		return (n_rows + 63) & ~63;
	int bucket = 1;
	while (bucket < n_rows)
		bucket <<= 1;
	return bucket;
}

static size_t align_up(size_t v) {
	return (v + MEM_ARENA_ALIGN - 1) & ~(size_t)(MEM_ARENA_ALIGN - 1);
}

static void layout_compute(const mem_plan *p, int n_rows, mem_layout *out) {
	memset(out, 0, sizeof(*out));
	out->bucket = mem_rows_bucket(n_rows);
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (!p->planned[slot] || p->rep[slot] != slot)
			continue;
		out->offset[slot] = out->total;
		out->cap[slot] = align_up(p->group_row_elems[slot] * (size_t)out->bucket * sizeof(float));
		out->total += out->cap[slot];
	}
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (!p->planned[slot] || p->rep[slot] == slot)
			continue;
		out->offset[slot] = out->offset[p->rep[slot]];
		out->cap[slot]	  = out->cap[p->rep[slot]];
	}
}

const mem_layout *mem_layout_get(mem_layout_cache *cache, const mem_plan *p, int n_rows) {
	if (cache->plan != p) {
		memset(cache, 0, sizeof(*cache));
		cache->plan = p;
	}
	int bucket = mem_rows_bucket(n_rows);
	for (int i = 0; i < cache->n; i++)
		if (cache->entries[i].bucket == bucket)
			return &cache->entries[i];
	int at;
	if (cache->n < MEM_LAYOUT_CACHE) {
		at = cache->n++;
	} else {
		at			= cache->next;
		cache->next = (cache->next + 1) % MEM_LAYOUT_CACHE;
	}
	layout_compute(p, n_rows, &cache->entries[at]);
	return &cache->entries[at];
}

static backend *part_layer_backend(const model *m, int li) {
	return model_mixed_backend_mode(m) ? model_layer_backend(m, li) : m->backend;
}

static void part_push(part_plan *p, part_node node) {
	p->nodes[p->n_nodes++] = node;
}

static int part_in_carry(const part_plan *p, uint8_t slot) {
	for (int c = 0; c < p->n_carry; c++)
		if (p->carry[c].slot == slot)
			return 1;
	return 0;
}

part_plan *part_plan_build(const model *m) {
	if (!m || !m->backend)
		return NULL;
	part_plan *p		   = xcalloc(1, sizeof(part_plan));
	p->nodes			   = xcalloc((size_t)m->n_layers * 2 + 2, sizeof(part_node));
	p->carry[p->n_carry++] = (part_carry){.slot = RECIPE_SLOT_X, .row_elems = m->dim};
	if (m->arch_info->has_hyper_connections)
		p->carry[p->n_carry++] =
			(part_carry){.slot = RECIPE_SLOT_HC, .row_elems = m->kda.hc_count * m->dim};
	backend *cur = m->backend;
	for (int li = 0; li < m->n_layers; li++) {
		backend	  *be	= part_layer_backend(m, li);
		part_node *last = p->n_nodes ? &p->nodes[p->n_nodes - 1] : NULL;
		if (last && last->kind == PART_COMPUTE && last->be == be) {
			last->n_layers++;
			continue;
		}
		if (be != cur)
			part_push(p, (part_node){.kind = PART_TRANSFER, .be = be, .src = cur});
		part_push(p, (part_node){.kind = PART_COMPUTE, .be = be, .first_layer = li, .n_layers = 1});
		cur = be;
	}
	if (cur != m->backend)
		part_push(p, (part_node){.kind = PART_TRANSFER, .be = m->backend, .src = cur});
	part_push(p, (part_node){.kind = PART_GATHER, .be = m->backend});
	return p;
}

void part_plan_free(part_plan *p) {
	if (!p)
		return;
	free(p->nodes);
	free(p);
}

int part_plan_n_transfers(const part_plan *p) {
	int n = 0;
	for (int i = 0; p && i < p->n_nodes; i++)
		n += p->nodes[i].kind == PART_TRANSFER;
	return n;
}

int part_plan_check_carry(const part_plan *p, const cg_set *graphs) {
	if (!p || !graphs || !part_plan_n_transfers(p))
		return 0;
	int missing = 0;
	for (int v = 0; v < graphs->n_graphs; v++) {
		const cg_graph *g = graphs->graphs[v];
		for (int c = 0; c < g->n_carry; c++) {
			if (part_in_carry(p, g->carry[c].slot))
				continue;
			WARN("partition: layer variant %d carries slot %d across layers but device "
				 "boundaries do not transfer it",
				 v, g->carry[c].slot);
			missing++;
		}
	}
	return missing;
}

typedef struct {
	size_t begin, end;
} layout_span;

static const char *const VG_SECTION_LABELS[CG_SECTION_COUNT] = {"pre", "layer", "post"};

static void vrange_merge(cg_range *r, cg_range from) {
	r->start = MIN(r->start, from.start);
	r->end	 = MAX(r->end, from.end);
}

static cg_range *vrange_roots(const cg_graph *c) {
	cg_range *ranges = graph_value_ranges(c);
	cg_range *roots	 = xmalloc((size_t)(c->n_values + 1) * sizeof(cg_range));
	for (int v = 0; v < c->n_values; v++)
		roots[v] = (cg_range){.start = INT_MAX, .end = -1};
	for (int v = 0; v < c->n_values; v++)
		if (ranges[v].end >= 0)
			vrange_merge(&roots[cg_value_root(c, v)], ranges[v]);
	free(ranges);
	return roots;
}

static bool vrange_overlap(cg_range a, cg_range b) {
	return a.start <= b.end && b.start <= a.end;
}

static size_t aligned_floats(size_t floats) {
	return (floats + CG_ARENA_ALIGN_FLOATS - 1) / CG_ARENA_ALIGN_FLOATS * CG_ARENA_ALIGN_FLOATS;
}

static int layout_span_cmp(const void *a, const void *b) {
	size_t x = ((const layout_span *)a)->begin, y = ((const layout_span *)b)->begin;
	return x < y ? -1 : x > y;
}

static size_t layout_first_fit(layout_span *taken, int n, size_t floats) {
	qsort(taken, (size_t)n, sizeof(layout_span), layout_span_cmp);
	size_t cursor = 0;
	for (int i = 0; i < n; i++) {
		if (taken[i].begin >= cursor + floats)
			break;
		cursor = MAX(cursor, taken[i].end);
	}
	return cursor;
}

static const cg_graph *layout_sort_graph;

static int layout_by_size_desc(const void *a, const void *b) {
	size_t x = layout_sort_graph->values[*(const int *)a].elems;
	size_t y = layout_sort_graph->values[*(const int *)b].elems;
	return x < y ? 1 : x > y ? -1 : *(const int *)a - *(const int *)b;
}

static bool layout_needs_arena(const cg_graph *c, const cg_range *r, int v) {
	const cg_value *val = &c->values[v];
	return val->alias_of == CG_NONE && val->view_of == CG_NONE && val->external != CG_EXT_LAYER &&
		   r[v].end >= 0;
}

void cg_arena_layout_plan(cg_arena_layout *l, const cg_graph *c, const cg_range *r) {
	int			*order	= xmalloc((size_t)(c->n_values + 1) * sizeof(int));
	layout_span *taken	= xmalloc((size_t)(c->n_values + 1) * sizeof(layout_span));
	int			 n		= 0;
	l->arena_floats		= 0;
	l->unplanned_floats = 0;
	for (int v = 0; v < c->n_values; v++) {
		l->offset[v] = CG_NO_OFFSET;
		if (layout_needs_arena(c, r, v)) {
			order[n++] = v;
			l->unplanned_floats += aligned_floats(c->values[v].elems);
		}
	}
	layout_sort_graph = c;
	qsort(order, (size_t)n, sizeof(int), layout_by_size_desc);
	for (int i = 0; i < n; i++) {
		int	   v	   = order[i];
		size_t floats  = aligned_floats(c->values[v].elems);
		int	   n_taken = 0;
		for (int j = 0; j < i; j++) {
			int u = order[j];
			if (vrange_overlap(r[v], r[u]))
				taken[n_taken++] =
					(layout_span){l->offset[u], l->offset[u] + aligned_floats(c->values[u].elems)};
		}
		l->offset[v]	= layout_first_fit(taken, n_taken, floats);
		l->arena_floats = MAX(l->arena_floats, l->offset[v] + floats);
	}
	free(order);
	free(taken);
}

void cg_arena_layout_release(cg_arena_layout *l) {
	free(l->offset);
	l->offset = NULL;
}

status_code cg_arena_layout_build(cg_arena_layout *l, const cg_graph *c) {
	l->offset	= xmalloc((size_t)(c->n_values + 1) * sizeof(size_t));
	cg_range *r = vrange_roots(c);
	cg_arena_layout_plan(l, c, r);
	free(r);
	return OK;
}

static bool vg_param_present(const cg_node *n, const cg_param *p) {
	switch (n->kind) {
	case OP_DF_MATMUL:
		return p && p->mat;
	case OP_DF_CONV:
		return p && p->conv;
	case OP_DF_RMSNORM:
	case OP_DF_RMSNORM_PER_HEAD:
		return p && p->vec;
	default:
		return true;
	}
}

static status_code vg_check_params(const cg_builder *g) {
	for (int i = 0; i < g->g->n_nodes; i++) {
		const cg_node *n = &g->g->nodes[i];
		if (n->param.scope == CG_SCOPE_LAYER && !g->layer_params) {
			ERROR("graph: n%d %s uses layer params but none are bound", i, recipe_op_name(n->kind));
			return ERR_NOT_FOUND;
		}
		int layers = n->param.scope == CG_SCOPE_LAYER ? g->n_layers : 1;
		for (int l = 0; l < layers; l++)
			if (!vg_param_present(n, cg_builder_param(g, n->param, l))) {
				ERROR("graph: n%d %s has no weight (layer %d)", i, recipe_op_name(n->kind), l);
				return ERR_NOT_FOUND;
			}
	}
	return OK;
}

static status_code vg_resolve_steps(cg_plan *p, backend *be) {
	bool resolved[OP_KIND_COUNT] = {false};
	for (int i = 0; i < p->graph->g->n_nodes; i++) {
		op_kind kind = p->graph->g->nodes[i].kind;
		if (!resolved[kind]) {
			cap_resolution res = cap_resolve(be, NULL, kind, NULL, 1);
			if (res.impl != CAP_NATIVE || !cg_node_handler_for(kind)) {
				ERROR("graph: backend '%s' cannot run %s (missing %s)", be->name,
					  recipe_op_name(kind), res.fn ? res.fn : "handler");
				cap_report(be, &res);
				return ERR_UNSUPPORTED;
			}
			resolved[kind] = true;
		}
		p->steps[i].fn = cg_node_handler_for(kind);
	}
	return OK;
}

status_code cg_plan_build(cg_builder *g, backend *be, const char *label, cg_plan **out) {
	cg_builder_finish(g);
	cg_plan *p	  = xcalloc(1, sizeof(*p));
	p->graph	  = g;
	p->label	  = label;
	p->profiling  = log_get_level() <= LOG_DEBUG;
	p->steps	  = xcalloc((size_t)g->g->n_nodes + 1, sizeof(cg_node_step));
	status_code s = cg_builder_validate(g) == 0 ? vg_check_params(g) : ERR_FORMAT;
	if (s == OK)
		s = vg_resolve_steps(p, be);
	if (s != OK) {
		ERROR("%s graph rejected", label);
		cg_plan_free(p);
		return s;
	}
	status_code s2 = cg_arena_layout_build(&p->layout, g->g);
	if (s2 != OK) {
		cg_plan_free(p);
		return s2;
	}
	if (graph_dump_mode() == CG_DUMP_TEXT)
		graph_dump(g->g, stderr);
	else if (graph_dump_mode() == CG_DUMP_DOT)
		graph_dump_dot(g->g, stderr);
	DEBUG("%s graph: %d nodes, %d values, arena %.1f MiB (%.1f MiB without reuse)", label,
		  g->g->n_nodes, g->g->n_values,
		  (double)(p->layout.arena_floats * sizeof(float)) / (1 << 20),
		  (double)(p->layout.unplanned_floats * sizeof(float)) / (1 << 20));
	*out = p;
	return OK;
}

void cg_plan_report(cg_plan *p, FILE *fp) {
	bool printed = false;
	for (int i = 0; i < p->graph->g->n_nodes; i++) {
		cg_node_step *st = &p->steps[i];
		if (!st->calls)
			continue;
		if (!printed) {
			fprintf(fp, "[profile] %s  per graph node (host dispatch time)\n", p->label);
			printed = true;
		}
		const cg_node *n = &p->graph->g->nodes[i];
		fprintf(fp, "  %-5s n%-4d %-24s calls=%-6llu us=%-9llu us/call=%.1f\n",
				VG_SECTION_LABELS[n->section], i, recipe_op_name(n->kind),
				(unsigned long long)st->calls, (unsigned long long)st->us,
				(double)st->us / (double)st->calls);
		st->us	  = 0;
		st->calls = 0;
	}
}

void cg_plan_free(cg_plan *p) {
	if (!p)
		return;
	if (p->profiling)
		cg_plan_report(p, stderr);
	cg_builder_free(p->graph);
	free(p->steps);
	cg_arena_layout_release(&p->layout);
	free(p);
}

cg_plan *cg_plan_cache_get(const cg_plan_cache *c, uint64_t key) {
	for (int i = 0; i < c->count; i++)
		if (c->keys[i] == key)
			return c->plans[i];
	return NULL;
}

void cg_plan_cache_put(cg_plan_cache *c, uint64_t key, cg_plan *plan) {
	int slot;
	if (c->count < CG_PLAN_CACHE_SIZE)
		slot = c->count++;
	else {
		slot	= c->next;
		c->next = (c->next + 1) % CG_PLAN_CACHE_SIZE;
		cg_plan_free(c->plans[slot]);
	}
	c->plans[slot] = plan;
	c->keys[slot]  = key;
}

void cg_plan_cache_free(cg_plan_cache *c) {
	for (int i = 0; i < c->count; i++)
		cg_plan_free(c->plans[i]);
	memset(c, 0, sizeof(*c));
}
