#include "mem_plan.h"

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
