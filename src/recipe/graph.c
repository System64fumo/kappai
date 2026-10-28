#include "recipe/graph.h"

#include "recipe/capability.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifdef KAPPAI_DEBUG
#define GRAPH_DIAG ERROR
#else
#define GRAPH_DIAG WARN
#endif

typedef struct {
	cg_graph *g;
	int		  cur[RECIPE_SLOT_MAX];
	int		  entry[RECIPE_SLOT_MAX];
	uint8_t	  exposed[RECIPE_SLOT_MAX];
	int		  last_writer[CG_STATE_SLOTS];
	int		  tracking;
} builder;

static const char *SLOT_NAMES[RECIPE_SLOT_MAX] = {
	[RECIPE_SLOT_X]			  = "X",
	[RECIPE_SLOT_XB]		  = "XB",
	[RECIPE_SLOT_XB2]		  = "XB2",
	[RECIPE_SLOT_ATTN_OUT]	  = "ATTN_OUT",
	[RECIPE_SLOT_Q]			  = "Q",
	[RECIPE_SLOT_K]			  = "K",
	[RECIPE_SLOT_V]			  = "V",
	[RECIPE_SLOT_FFN_GATE]	  = "FFN_GATE",
	[RECIPE_SLOT_FFN_UP]	  = "FFN_UP",
	[RECIPE_SLOT_FFN_ACT]	  = "FFN_ACT",
	[RECIPE_SLOT_FFN_GATE_UP] = "FFN_GATE_UP",
	[RECIPE_SLOT_RESID_TMP]	  = "RESID_TMP",
	[RECIPE_SLOT_LOGITS]	  = "LOGITS",
	[RECIPE_SLOT_ROUTER_IDS]  = "ROUTER_IDS",
	[RECIPE_SLOT_ROUTER_W]	  = "ROUTER_W",
	[RECIPE_SLOT_HYB_PROJ]	  = "HYB_PROJ",
	[RECIPE_SLOT_HYB_GATE]	  = "HYB_GATE",
	[RECIPE_SLOT_HYB_ALPHA]	  = "HYB_ALPHA",
	[RECIPE_SLOT_HYB_BETA]	  = "HYB_BETA",
	[RECIPE_SLOT_HC]		  = "HC",
	[RECIPE_SLOT_HC_COEF]	  = "HC_COEF",
};

static const struct {
	cg_state	state;
	const char *name;
} STATE_NAMES[] = {
	{CG_STATE_KV, "kv"},   {CG_STATE_MLA, "mla"},		{CG_STATE_RECURRENT, "rec"},
	{CG_STATE_PLE, "ple"}, {CG_STATE_LOGITS, "logits"},
};

static const char *SECTION_NAMES[CG_SECTION_COUNT] = {"pre", "layer", "post"};

int graph_dump_mode(void) {
	static _Atomic int cached = -1;
	if (cached >= 0)
		return cached;
	const char *v = getenv("KAPPAI_GRAPH_DUMP");
	if (!v || !v[0] || strcmp(v, "0") == 0)
		cached = CG_DUMP_OFF;
	else
		cached = strcmp(v, "dot") == 0 ? CG_DUMP_DOT : CG_DUMP_TEXT;
	return cached;
}

static int slot_is_router(uint8_t slot) {
	return slot == RECIPE_SLOT_ROUTER_IDS || slot == RECIPE_SLOT_ROUTER_W;
}

static const uint8_t SLOT_SIZE_KIND[RECIPE_SLOT_MAX] = {
	[RECIPE_SLOT_X]			  = BS_SIZE_ATTN_BUF,
	[RECIPE_SLOT_XB]		  = BS_SIZE_ATTN_BUF,
	[RECIPE_SLOT_XB2]		  = BS_SIZE_ATTN_BUF,
	[RECIPE_SLOT_ATTN_OUT]	  = BS_SIZE_ATTN_BUF,
	[RECIPE_SLOT_Q]			  = BS_SIZE_Q_OUT,
	[RECIPE_SLOT_K]			  = BS_SIZE_KV_OUT,
	[RECIPE_SLOT_V]			  = BS_SIZE_KV_OUT,
	[RECIPE_SLOT_FFN_GATE]	  = BS_SIZE_INTER,
	[RECIPE_SLOT_FFN_UP]	  = BS_SIZE_INTER,
	[RECIPE_SLOT_FFN_ACT]	  = BS_SIZE_ACT,
	[RECIPE_SLOT_FFN_GATE_UP] = BS_SIZE_INTER_X2,
	[RECIPE_SLOT_RESID_TMP]	  = BS_SIZE_ATTN_BUF,
	[RECIPE_SLOT_HC]		  = BS_SIZE_HC,
	[RECIPE_SLOT_HC_COEF]	  = BS_SIZE_HC_COEF,
};

static const struct {
	uint8_t parent;
	uint8_t views[2];
} SLOT_VIEWS[] = {
	{RECIPE_SLOT_FFN_GATE_UP, {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP}},
};

int graph_slot_size_kind(uint8_t slot) {
	return slot < RECIPE_SLOT_MAX ? SLOT_SIZE_KIND[slot] : BS_SIZE_NONE;
}

static int slot_batch_backed(uint8_t slot) {
	return slot_is_router(slot) || graph_slot_valid(slot);
}

static int new_value(builder *b, uint8_t slot, int producer, uint8_t port, int host_only) {
	cg_graph *g	  = b->g;
	int		  id  = g->n_values++;
	g->values[id] = (cg_value){
		.producer  = producer,
		.alias_of  = CG_NONE,
		.view_of   = CG_NONE,
		.slot	   = slot,
		.port	   = port,
		.size_kind = (uint8_t)graph_slot_size_kind(slot),
		.rows	   = slot == RECIPE_SLOT_LOGITS ? CG_ROWS_OUTPUT : CG_ROWS_BATCH,
		.dtype	   = slot == RECIPE_SLOT_ROUTER_IDS ? CG_DTYPE_I32 : CG_DTYPE_F32,
		.residency = host_only || slot_is_router(slot) ? CG_RES_HOST : CG_RES_DEVICE,
	};
	return id;
}

static int read_slot(builder *b, uint8_t slot) {
	if (b->cur[slot] == CG_NONE) {
		b->cur[slot] = new_value(b, slot, CG_NONE, 0, 0);
		if (b->tracking)
			b->entry[slot] = b->cur[slot];
	}
	if (b->tracking && b->cur[slot] == b->entry[slot])
		b->exposed[slot] = 1;
	return b->cur[slot];
}

static int find_slot(const uint8_t *slots, int n, uint8_t slot) {
	for (int i = 0; i < n; i++)
		if (slots[i] == slot)
			return i;
	return CG_NONE;
}

static void add_slot_views(builder *b, cg_node *n, int node_id, int parent, int host_only) {
	cg_graph *g = b->g;
	for (size_t i = 0; i < ARRAY_LEN(SLOT_VIEWS); i++) {
		if (SLOT_VIEWS[i].parent != g->values[parent].slot)
			continue;
		for (size_t k = 0; k < ARRAY_LEN(SLOT_VIEWS[i].views) && n->n_out < OP_TOUCH_MAX; k++) {
			uint8_t slot		 = SLOT_VIEWS[i].views[k];
			int		v			 = new_value(b, slot, node_id, (uint8_t)n->n_out, host_only);
			g->values[v].view_of = parent;
			b->cur[slot]		 = v;
			n->out[n->n_out++]	 = v;
		}
	}
}

static void lower_op(builder *b, const recipe_op *op, cg_section section, int recipe_index) {
	uint8_t reads[OP_TOUCH_MAX];
	uint8_t writes[OP_TOUCH_MAX];
	int		n_reads	 = 0;
	int		n_writes = 0;
	if (!recipe_op_touch(op, reads, &n_reads, writes, &n_writes))
		return;

	cg_graph *g	 = b->g;
	int		  id = g->n_nodes++;
	cg_node	 *n	 = &g->nodes[id];
	*n			 = (cg_node){.kind			   = op->kind,
							 .section		   = section,
							 .stage			   = op->stage,
							 .recipe_index	   = recipe_index,
							 .w_idx			   = op->w_idx,
							 .coalesce_run_len = op->coalesce_run_len,
							 .fuse_next		   = op->fuse_next,
							 .state_reads	   = recipe_op_state_reads(op->kind),
							 .state_writes	   = recipe_op_state_writes(op->kind),
							 .params		   = op->u};

	for (int k = 0; k < CG_STATE_SLOTS; k++)
		n->state_dep[k] = (n->state_reads >> k) & 1 ? b->last_writer[k] : CG_NONE;

	for (int i = 0; i < n_reads; i++) {
		n->in[n->n_in] = read_slot(b, reads[i]);
		g->values[n->in[n->n_in]].n_uses++;
		n->n_in++;
	}
	for (int i = 0; i < n_writes; i++) {
		int host_only = cap_host_only_fn(op->kind) != NULL;
		int v		  = new_value(b, writes[i], id, (uint8_t)n->n_out, host_only);
		int r		  = find_slot(reads, n_reads, writes[i]);
		if (r != CG_NONE)
			g->values[v].alias_of = n->in[r];
		b->cur[writes[i]]  = v;
		n->out[n->n_out++] = v;
		add_slot_views(b, n, id, v, host_only);
	}
	for (int k = 0; k < CG_STATE_SLOTS; k++)
		if ((n->state_writes >> k) & 1)
			b->last_writer[k] = id;
}

static void lower_ops(builder *b, const recipe_op *ops, int n_ops, cg_section section) {
	for (int i = 0; i < n_ops; i++)
		if (ops[i].kind != OP_NONE)
			lower_op(b, &ops[i], section, i);
}

static void collect_carries(builder *b) {
	cg_graph *g = b->g;
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (!b->exposed[slot] || b->cur[slot] == b->entry[slot])
			continue;
		g->values[b->entry[slot]].loop_carried = 1;
		g->carry[g->n_carry++] =
			(cg_carry){.slot = (uint8_t)slot, .in = b->entry[slot], .out = b->cur[slot]};
	}
}

static int count_ops(const recipe_op *ops, int n_ops) {
	int n = 0;
	for (int i = 0; i < n_ops; i++)
		n += ops[i].kind != OP_NONE;
	return n;
}

cg_graph *graph_build_variant(const model_recipe *r, int variant) {
	if (!r)
		return NULL;
	const recipe_op *layer_ops = recipe_layer_ops(r, variant);
	int max_nodes = count_ops(r->pre_ops, r->n_pre_ops) + count_ops(layer_ops, r->layer.n_ops) +
					count_ops(r->post_ops, r->n_post_ops);

	cg_graph *g = xcalloc(1, sizeof(cg_graph));
	g->nodes	= xcalloc((size_t)max_nodes + 1, sizeof(cg_node));
	g->values	= xcalloc((size_t)max_nodes * 2 * OP_TOUCH_MAX + 1, sizeof(cg_value));

	builder b = {.g = g};
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++)
		b.cur[slot] = b.entry[slot] = CG_NONE;
	for (int k = 0; k < CG_STATE_SLOTS; k++)
		b.last_writer[k] = CG_NONE;

	g->section_begin[CG_SECTION_PRE] = 0;
	lower_ops(&b, r->pre_ops, r->n_pre_ops, CG_SECTION_PRE);

	g->section_begin[CG_SECTION_LAYER] = g->n_nodes;
	memcpy(b.entry, b.cur, sizeof(b.entry));
	b.tracking = 1;
	lower_ops(&b, layer_ops, r->layer.n_ops, CG_SECTION_LAYER);
	b.tracking = 0;
	collect_carries(&b);

	g->section_begin[CG_SECTION_POST] = g->n_nodes;
	lower_ops(&b, r->post_ops, r->n_post_ops, CG_SECTION_POST);
	g->section_begin[CG_SECTION_COUNT] = g->n_nodes;
	graph_index_consumers(g);
	return g;
}

cg_graph *graph_build(const model_recipe *r) {
	return graph_build_variant(r, 0);
}

void graph_free(cg_graph *g) {
	if (!g)
		return;
	free(g->nodes);
	free(g->values);
	free(g->consumer_begin);
	free(g->consumers);
	free(g);
}

void graph_index_consumers(cg_graph *g) {
	free(g->consumer_begin);
	free(g->consumers);
	g->consumer_begin = xcalloc((size_t)g->n_values + 2, sizeof(int));
	for (int i = 0; i < g->n_nodes; i++)
		for (int j = 0; j < g->nodes[i].n_in; j++)
			g->consumer_begin[g->nodes[i].in[j] + 1]++;
	for (int v = 0; v < g->n_values; v++)
		g->consumer_begin[v + 1] += g->consumer_begin[v];
	int *cursor = xmalloc((size_t)g->n_values * sizeof(int) + sizeof(int));
	memcpy(cursor, g->consumer_begin, (size_t)g->n_values * sizeof(int));
	g->consumers = xmalloc((size_t)g->consumer_begin[g->n_values] * sizeof(int) + sizeof(int));
	for (int i = 0; i < g->n_nodes; i++)
		for (int j = 0; j < g->nodes[i].n_in; j++)
			g->consumers[cursor[g->nodes[i].in[j]]++] = i;
	free(cursor);
}

int graph_consumers(const cg_graph *g, int value, const int **list) {
	*list = &g->consumers[g->consumer_begin[value]];
	return g->consumer_begin[value + 1] - g->consumer_begin[value];
}

void graph_freeze(cg_graph *g) {
	g->frozen = 1;
}

static const char *state_name(uint8_t mask) {
	for (size_t i = 0; i < ARRAY_LEN(STATE_NAMES); i++)
		if (mask & STATE_NAMES[i].state)
			return STATE_NAMES[i].name;
	return "?";
}

static int count_unbacked_values(const cg_graph *g, const cg_node *n, int index) {
	int n_errors = 0;
	for (int j = 0; j < n->n_in + n->n_out; j++) {
		int		value = j < n->n_in ? n->in[j] : n->out[j - n->n_in];
		uint8_t slot  = g->values[value].slot;
		if (slot_batch_backed(slot))
			continue;
		GRAPH_DIAG("graph: node %d (%s, %s) touches slot %u which has no batch buffer", index,
				   recipe_op_name(n->kind), SECTION_NAMES[n->section], slot);
		n_errors++;
	}
	return n_errors;
}

static uint8_t unordered_state_reads(const cg_node *n, int index, uint8_t written_in_graph) {
	uint8_t mask = 0;
	for (int k = 0; k < CG_STATE_SLOTS; k++) {
		uint8_t bit = (uint8_t)(1u << k);
		if (!(n->state_reads & bit))
			continue;
		int dep = n->state_dep[k];
		if (dep == CG_NONE ? (bit & written_in_graph) && bit != CG_STATE_RECURRENT : dep >= index)
			mask |= bit;
	}
	return mask;
}

static int count_add_size_mismatch(const cg_graph *g, const cg_node *n, int index) {
	if (n->kind != OP_ADD || n->n_in <= CG_ADD_ADDEND)
		return 0;
	uint8_t acc	   = g->values[n->in[CG_ADD_ACC]].size_kind;
	uint8_t addend = g->values[n->in[CG_ADD_ADDEND]].size_kind;
	if (!acc || !addend || acc == addend)
		return 0;
	if (acc == BS_SIZE_ACT || addend == BS_SIZE_ACT)
		return 0;
	GRAPH_DIAG("graph: node %d (add, %s) adds buffers of different sizes", index,
			   SECTION_NAMES[n->section]);
	return 1;
}

int graph_validate(const cg_graph *g) {
	int		n_errors		 = 0;
	uint8_t written_in_graph = 0;
	for (int i = 0; i < g->n_nodes; i++)
		written_in_graph |= g->nodes[i].state_writes;
	for (int i = 0; i < g->n_nodes; i++) {
		const cg_node *n		 = &g->nodes[i];
		uint8_t		   unwritten = unordered_state_reads(n, i, written_in_graph);
		if (unwritten) {
			GRAPH_DIAG("graph: node %d (%s, %s) reads %s state before any node writes it", i,
					   recipe_op_name(n->kind), SECTION_NAMES[n->section], state_name(unwritten));
			n_errors++;
		}
		n_errors += count_unbacked_values(g, n, i);
		n_errors += count_add_size_mismatch(g, n, i);
		for (int j = 0; j < n->n_in; j++) {
			const cg_value *v = &g->values[n->in[j]];
			if (v->producer != CG_NONE && v->producer < i)
				continue;
			GRAPH_DIAG("graph: node %d (%s, %s) reads %s before it is defined", i,
					   recipe_op_name(n->kind), SECTION_NAMES[n->section],
					   SLOT_NAMES[v->slot] ? SLOT_NAMES[v->slot] : "?");
			n_errors++;
		}
	}
	return n_errors;
}

static void range_extend(cg_range *r, int from, int to) {
	r->start = MIN(r->start, from);
	r->end	 = MAX(r->end, to);
}

static int range_overlaps(const cg_range *a, const cg_range *b) {
	return a->start <= b->end && b->start <= a->end;
}

cg_range *graph_value_ranges(const cg_graph *g) {
	cg_range *ranges	  = xmalloc((size_t)(g->n_values + 1) * sizeof(cg_range));
	int		  layer_begin = g->section_begin[CG_SECTION_LAYER];
	int		  post_begin  = g->section_begin[CG_SECTION_POST];
	for (int v = 0; v < g->n_values; v++)
		ranges[v] = (cg_range){.start = INT_MAX, .end = -1};
	for (int i = 0; i < g->n_nodes; i++) {
		const cg_node *n = &g->nodes[i];
		for (int j = 0; j < n->n_in + n->n_out; j++)
			range_extend(&ranges[j < n->n_in ? n->in[j] : n->out[j - n->n_in]], i, i);
	}
	for (int v = 0; v < g->n_values; v++) {
		if (ranges[v].end < 0)
			continue;
		int producer = g->values[v].producer;
		if (producer == CG_NONE)
			ranges[v].start = 0;
		int defined_before_layer = producer == CG_NONE || producer < layer_begin;
		if (defined_before_layer && ranges[v].end >= layer_begin && ranges[v].end < post_begin)
			ranges[v].end = post_begin - 1;
	}
	for (int c = 0; c < g->n_carry; c++) {
		range_extend(&ranges[g->carry[c].in], 0, g->n_nodes - 1);
		range_extend(&ranges[g->carry[c].out], 0, g->n_nodes - 1);
	}
	for (int v = 0; v < g->n_values; v++)
		if (g->values[v].view_of != CG_NONE && ranges[v].end >= 0)
			range_extend(&ranges[g->values[v].view_of], ranges[v].start, ranges[v].end);
	return ranges;
}

int graph_peak_live_buffers(const cg_graph *g) {
	cg_range *ranges = graph_value_ranges(g);
	int		 *root	 = xmalloc((size_t)(g->n_values + 1) * sizeof(int));
	cg_range *groups = xmalloc((size_t)(g->n_values + 1) * sizeof(cg_range));
	for (int v = 0; v < g->n_values; v++) {
		int parent =
			g->values[v].alias_of != CG_NONE ? g->values[v].alias_of : g->values[v].view_of;
		root[v]	  = parent == CG_NONE ? v : root[parent];
		groups[v] = (cg_range){.start = INT_MAX, .end = -1};
	}
	for (int v = 0; v < g->n_values; v++)
		if (ranges[v].end >= 0)
			range_extend(&groups[root[v]], ranges[v].start, ranges[v].end);
	int peak = 0;
	for (int t = 0; t < g->n_nodes; t++) {
		cg_range at	  = {.start = t, .end = t};
		int		 live = 0;
		for (int v = 0; v < g->n_values; v++)
			live += root[v] == v && groups[v].end >= 0 && range_overlaps(&groups[v], &at);
		peak = MAX(peak, live);
	}
	free(ranges);
	free(root);
	free(groups);
	return peak;
}

static const char *slot_label(uint8_t slot) {
	return slot < RECIPE_SLOT_MAX && SLOT_NAMES[slot] ? SLOT_NAMES[slot] : "-";
}

static void dump_value(FILE *out, const cg_graph *g, int id) {
	const cg_value *v = &g->values[id];
	fprintf(out, " v%d:%s%s%s%s", id, slot_label(v->slot), v->alias_of != CG_NONE ? "~" : "",
			v->view_of != CG_NONE ? "^" : "", v->loop_carried ? "*" : "");
}

static void dump_node(FILE *out, const cg_graph *g, int id) {
	const cg_node *n = &g->nodes[id];
	fprintf(out, "  n%-4d %-20s stage=%-2d w=%-2u op=%d in=[", id, recipe_op_name(n->kind),
			(int)n->stage, (unsigned)n->w_idx, n->recipe_index);
	for (int i = 0; i < n->n_in; i++)
		dump_value(out, g, n->in[i]);
	fprintf(out, " ] out=[");
	for (int i = 0; i < n->n_out; i++)
		dump_value(out, g, n->out[i]);
	fprintf(out, " ]");
	for (size_t i = 0; i < ARRAY_LEN(STATE_NAMES); i++) {
		if (n->state_reads & STATE_NAMES[i].state)
			fprintf(out, " r:%s", STATE_NAMES[i].name);
		if (n->state_writes & STATE_NAMES[i].state)
			fprintf(out, " w:%s", STATE_NAMES[i].name);
	}
	fprintf(out, "\n");
}

void graph_dump(const cg_graph *g, FILE *out) {
	if (!g)
		return;
	fprintf(out, "graph: %d nodes, %d values, %d carried, peak %d live buffers\n", g->n_nodes,
			g->n_values, g->n_carry, graph_peak_live_buffers(g));
	for (int s = 0; s < CG_SECTION_COUNT; s++) {
		fprintf(out, "[%s]\n", SECTION_NAMES[s]);
		for (int i = g->section_begin[s]; i < g->section_begin[s + 1]; i++)
			dump_node(out, g, i);
	}
	for (int i = 0; i < g->n_carry; i++)
		fprintf(out, "carry %s: v%d -> v%d\n", SLOT_NAMES[g->carry[i].slot], g->carry[i].in,
				g->carry[i].out);
}

void graph_dump_dot(const cg_graph *g, FILE *out) {
	if (!g)
		return;
	fprintf(out, "digraph compute_graph {\n  node [shape=box];\n");
	for (int i = 0; i < g->n_nodes; i++)
		fprintf(out, "  n%d [label=\"%d %s\\n%s\"];\n", i, i, recipe_op_name(g->nodes[i].kind),
				SECTION_NAMES[g->nodes[i].section]);
	for (int i = 0; i < g->n_nodes; i++) {
		const cg_node *n = &g->nodes[i];
		for (int j = 0; j < n->n_in; j++) {
			const cg_value *v = &g->values[n->in[j]];
			if (v->producer == CG_NONE)
				fprintf(out, "  v%d [shape=ellipse,label=\"%s\"];\n  v%d -> n%d;\n", n->in[j],
						slot_label(v->slot), n->in[j], i);
			else
				fprintf(out, "  n%d -> n%d [label=\"%s\"];\n", v->producer, i, slot_label(v->slot));
		}
	}
	for (int i = 0; i < g->n_carry; i++) {
		int from = g->values[g->carry[i].out].producer;
		int to	 = g->values[g->carry[i].in].producer;
		if (from != CG_NONE && to != CG_NONE)
			fprintf(out, "  n%d -> n%d [style=dashed,label=\"carry %s\"];\n", from, to,
					slot_label(g->carry[i].slot));
	}
	fprintf(out, "}\n");
}

static int layer_ops_equal(const model_recipe *r, int a, int b) {
	if (!r->per_layer_ops)
		return 1;
	return memcmp(recipe_layer_ops(r, a), recipe_layer_ops(r, b),
				  (size_t)r->layer.n_ops * sizeof(recipe_op)) == 0;
}

static int find_equal_graph(const cg_set *s, const model_recipe *r, int layer) {
	for (int g = 0; g < s->n_graphs; g++)
		if (layer_ops_equal(r, s->first_layer[g], layer))
			return g;
	return CG_NONE;
}

static int layer_node_count(const cg_graph *g) {
	return g->section_begin[CG_SECTION_POST] - g->section_begin[CG_SECTION_LAYER];
}

static void resolve_outer_weights(cg_graph *g, const struct model *m) {
	for (int i = 0; i < g->n_nodes; i++)
		if (g->nodes[i].section != CG_SECTION_LAYER)
			g->nodes[i].weight = recipe_resolve_weight_ref(m, -1, g->nodes[i].w_idx);
}

static void resolve_layer_weights(cg_set *s, const struct model *m, int count) {
	for (int g = 0; g < s->n_graphs; g++)
		if (layer_node_count(s->graphs[g]) > s->weight_stride)
			s->weight_stride = layer_node_count(s->graphs[g]);
	s->layer_weights =
		xcalloc((size_t)count * (size_t)(s->weight_stride + 1), sizeof(weight_ref *));
	for (int li = 0; li < count; li++) {
		const cg_graph *g = s->graphs[s->layer_graph[li]];
		for (int i = 0; i < layer_node_count(g); i++)
			s->layer_weights[(size_t)li * s->weight_stride + i] = recipe_resolve_weight_ref(
				m, li, g->nodes[g->section_begin[CG_SECTION_LAYER] + i].w_idx);
	}
}

const weight_ref *graph_set_weight(const cg_set *s, int layer, int node) {
	const cg_graph *g = s->graphs[s->layer_graph[layer]];
	if (g->nodes[node].section != CG_SECTION_LAYER)
		return g->nodes[node].weight;
	if (!s->layer_weights)
		return NULL;
	return s->layer_weights[(size_t)layer * s->weight_stride +
							(node - g->section_begin[CG_SECTION_LAYER])];
}

backend *graph_set_backend(const cg_set *s, int layer) {
	return s->layer_backend ? s->layer_backend[layer] : NULL;
}

static void resolve_layer_backends(cg_set *s, const struct model *m, int count) {
	s->layer_backend = xcalloc((size_t)count, sizeof(backend *));
	for (int li = 0; li < count; li++)
		s->layer_backend[li] = model_layer_backend(m, li);
}

static int fallback_seen(const cg_set *s, const backend *be, op_kind kind) {
	for (int i = 0; i < s->n_fallbacks; i++)
		if (s->fallbacks[i].be == be && s->fallbacks[i].kind == kind)
			return 1;
	return 0;
}

static void add_fallback(cg_set *s, backend *be, op_kind kind, const char *fn) {
	s->fallbacks = xrealloc(s->fallbacks, (size_t)(s->n_fallbacks + 1) * sizeof(cg_fallback));
	s->fallbacks[s->n_fallbacks++] = (cg_fallback){.be = be, .kind = kind, .fn = fn};
	INFO("graph: backend '%s' lacks %s; %s executes via fallback", be->name ? be->name : "?", fn,
		 recipe_op_name(kind));
}

static void resolve_node(cg_set *s, const struct model *m, backend *be, const cg_node *n,
						 const weight_ref *w, uint8_t *impl) {
	cap_resolution res = cap_resolve(be, m, n->kind, w, 1);
	*impl			   = (uint8_t)res.impl;
	if (res.impl == CAP_NATIVE || fallback_seen(s, be, n->kind))
		return;
	add_fallback(s, be, n->kind, res.fn);
	cap_report(be, &res);
}

static void resolve_impls(cg_set *s, const struct model *m, int count) {
	s->layer_impl = xcalloc((size_t)count * (size_t)(s->weight_stride + 1), sizeof(uint8_t));
	for (int li = 0; li < count; li++) {
		cg_graph *g		= s->graphs[s->layer_graph[li]];
		int		  begin = g->section_begin[CG_SECTION_LAYER];
		for (int i = begin; i < g->section_begin[CG_SECTION_POST]; i++)
			resolve_node(s, m, s->layer_backend[li], &g->nodes[i], graph_set_weight(s, li, i),
						 &s->layer_impl[(size_t)li * s->weight_stride + (i - begin)]);
	}
	for (int gi = 0; gi < s->n_graphs; gi++) {
		cg_graph *g = s->graphs[gi];
		for (int i = 0; i < g->n_nodes; i++)
			if (g->nodes[i].section != CG_SECTION_LAYER)
				resolve_node(s, m, m->backend, &g->nodes[i], g->nodes[i].weight, &g->nodes[i].impl);
	}
}

uint8_t graph_set_impl(const cg_set *s, int layer, int node) {
	const cg_graph *g = s->graphs[s->layer_graph[layer]];
	if (g->nodes[node].section != CG_SECTION_LAYER)
		return g->nodes[node].impl;
	if (!s->layer_impl)
		return CAP_NATIVE;
	return s->layer_impl[(size_t)layer * s->weight_stride +
						 (node - g->section_begin[CG_SECTION_LAYER])];
}

static int lower_outer_sections(cg_set *s, const model_recipe *r) {
	s->pre_ops	= xcalloc((size_t)r->n_pre_ops + 1, sizeof(recipe_op));
	s->post_ops = xcalloc((size_t)r->n_post_ops + 1, sizeof(recipe_op));
	if (graph_lower_section(s->graphs[0], CG_SECTION_PRE, r->pre_ops, r->n_pre_ops, s->pre_ops) ==
		CG_NONE)
		return CG_NONE;
	return graph_lower_section(s->graphs[0], CG_SECTION_POST, r->post_ops, r->n_post_ops,
							   s->post_ops);
}

static int lower_layers(cg_set *s, const model_recipe *r) {
	int n		   = r->layer.n_ops;
	s->n_layer_ops = n;
	s->layer_ops   = xcalloc((size_t)s->n_graphs * (size_t)n + 1, sizeof(recipe_op));
	for (int g = 0; g < s->n_graphs; g++)
		if (graph_lower_section(s->graphs[g], CG_SECTION_LAYER,
								recipe_layer_ops(r, s->first_layer[g]), n,
								&s->layer_ops[(size_t)g * n]) == CG_NONE)
			return CG_NONE;
	return OK;
}

static void lower_set(cg_set *s, const model_recipe *r) {
	if (lower_outer_sections(s, r) != CG_NONE && lower_layers(s, r) != CG_NONE)
		return;
	ERROR("graph: lowering failed");
	free(s->pre_ops);
	free(s->layer_ops);
	free(s->post_ops);
	s->pre_ops	 = NULL;
	s->layer_ops = NULL;
	s->post_ops	 = NULL;
}

static uint32_t node_batch_slot_mask(const cg_graph *g, const cg_node *n) {
	uint32_t mask = recipe_op_batch_extra_mask(n->kind);
	for (int j = 0; j < n->n_in + n->n_out; j++) {
		const cg_value *v = &g->values[j < n->n_in ? n->in[j] : n->out[j - n->n_in]];
		if (v->view_of != CG_NONE)
			continue;
		if (v->slot < RECIPE_SLOT_MAX)
			mask |= 1u << v->slot;
	}
	return mask;
}

uint32_t graph_set_batch_slot_mask(const cg_set *s) {
	uint32_t mask = 0;
	for (int k = 0; k < s->n_graphs; k++) {
		const cg_graph *g = s->graphs[k];
		for (int i = g->section_begin[CG_SECTION_PRE]; i < g->section_begin[CG_SECTION_POST]; i++)
			mask |= node_batch_slot_mask(g, &g->nodes[i]);
	}
	return mask;
}

int graph_set_executable(const cg_set *s) {
	return s && s->layer_ops;
}

const recipe_op *graph_set_pre_ops(const cg_set *s) {
	return s->pre_ops;
}

const recipe_op *graph_set_layer_ops(const cg_set *s, int layer) {
	return &s->layer_ops[(size_t)s->layer_graph[layer] * s->n_layer_ops];
}

const recipe_op *graph_set_post_ops(const cg_set *s) {
	return s->post_ops;
}

cg_set *graph_set_build(const model_recipe *r, const struct model *m, int n_layers) {
	if (!r)
		return NULL;
	int		count  = n_layers > 0 ? n_layers : 1;
	cg_set *s	   = xcalloc(1, sizeof(cg_set));
	s->n_layers	   = n_layers;
	s->graphs	   = xcalloc((size_t)count, sizeof(cg_graph *));
	s->first_layer = xcalloc((size_t)count, sizeof(int));
	s->layer_graph = xcalloc((size_t)count, sizeof(int));
	for (int li = 0; li < count; li++) {
		int g = find_equal_graph(s, r, li);
		if (g == CG_NONE) {
			g				  = s->n_graphs++;
			s->graphs[g]	  = graph_build_variant(r, li);
			s->first_layer[g] = li;
		}
		s->layer_graph[li] = g;
	}
	if (m) {
		for (int g = 0; g < s->n_graphs; g++)
			resolve_outer_weights(s->graphs[g], m);
		resolve_layer_weights(s, m, count);
		resolve_layer_backends(s, m, count);
		resolve_impls(s, m, count);
	}
	lower_set(s, r);
	for (int g = 0; g < s->n_graphs; g++)
		graph_freeze(s->graphs[g]);
	for (int g = 0; g < s->n_graphs; g++)
		DEBUG("graph: variant %d: %d nodes, %d values, peak %d live buffers", g,
			  s->graphs[g]->n_nodes, s->graphs[g]->n_values, graph_peak_live_buffers(s->graphs[g]));
	DEBUG("graph: %d distinct layer variants over %d layers", s->n_graphs, n_layers);
	return s;
}

void graph_set_free(cg_set *s) {
	if (!s)
		return;
	for (int g = 0; g < s->n_graphs; g++)
		graph_free(s->graphs[g]);
	free(s->graphs);
	free(s->first_layer);
	free(s->layer_graph);
	free(s->layer_weights);
	free(s->pre_ops);
	free(s->layer_ops);
	free(s->post_ops);
	free(s->layer_backend);
	free(s->layer_impl);
	free(s->fallbacks);
	free(s);
}

int graph_set_validate(const cg_set *s) {
	int n_errors = 0;
	for (int g = 0; g < s->n_graphs; g++)
		n_errors += graph_validate(s->graphs[g]);
	return n_errors;
}

static const weight_ref *executor_weight(const struct model *m, int layer, uint8_t w_idx) {
	if (m->wrefs_by_layer && m->n_layers > 0)
		return m->wrefs_by_layer[(size_t)layer * WIDX_COUNT + w_idx];
	return recipe_resolve_weight_ref(m, layer, w_idx);
}

int graph_set_check_weights(const cg_set *s, const struct model *m) {
	int n_errors = 0;
	for (int li = 0; li < s->n_layers; li++) {
		const cg_graph *g = s->graphs[s->layer_graph[li]];
		for (int i = g->section_begin[CG_SECTION_LAYER]; i < g->section_begin[CG_SECTION_POST];
			 i++) {
			uint8_t w_idx = g->nodes[i].w_idx;
			if (w_idx == RECIPE_NO_WEIGHT || w_idx == WIDX_NONE)
				continue;
			if (graph_set_weight(s, li, i) == executor_weight(m, li, w_idx))
				continue;
			WARN("graph: layer %d node %d (%s) weight %u differs from the executor's table", li, i,
				 recipe_op_name(g->nodes[i].kind), w_idx);
			n_errors++;
		}
	}
	return n_errors;
}

void graph_set_dump(const cg_set *s, FILE *out, int dot) {
	for (int g = 0; g < s->n_graphs; g++) {
		int n_layers = 0;
		for (int li = 0; li < s->n_layers; li++)
			n_layers += s->layer_graph[li] == g;
		fprintf(out, "%s variant %d: first layer %d, %d layers\n", dot ? "//" : "#", g,
				s->first_layer[g], n_layers);
		if (dot)
			graph_dump_dot(s->graphs[g], out);
		else
			graph_dump(s->graphs[g], out);
	}
}

cg_builder *cg_builder_new(void) {
	cg_builder *g = xcalloc(1, sizeof(*g));
	g->g		  = xcalloc(1, sizeof(cg_graph));
	for (int s = 0; s <= CG_SECTION_COUNT; s++)
		g->g->section_begin[s] = CG_NONE;
	g->g->section_begin[CG_SECTION_PRE] = 0;
	g->n_layers							= 1;
	return g;
}

void cg_builder_free(cg_builder *g) {
	if (!g)
		return;
	graph_free(g->g);
	free(g->globals);
	free(g);
}

void cg_builder_section(cg_builder *g, cg_section section) {
	g->current					 = section;
	g->g->section_begin[section] = g->g->n_nodes;
}

void cg_builder_layers(cg_builder *g, int n_layers, const cg_param *rows, int stride) {
	g->n_layers		= n_layers;
	g->layer_params = rows;
	g->layer_stride = stride;
}

void cg_builder_finish(cg_builder *g) {
	cg_graph *c						   = g->g;
	int		  end					   = c->n_nodes;
	c->section_begin[CG_SECTION_COUNT] = end;
	for (int s = CG_SECTION_COUNT - 1; s >= 0; s--) {
		if (c->section_begin[s] == CG_NONE)
			c->section_begin[s] = end;
		end = c->section_begin[s];
	}
	graph_index_consumers(c);
	graph_freeze(c);
}

static cg_param_ref builder_add_global(cg_builder *g, cg_param p) {
	ARR_RESERVE(g->globals, g->n_globals, g->cap_globals);
	g->globals[g->n_globals] = p;
	return (cg_param_ref){CG_SCOPE_GLOBAL, g->n_globals++};
}

cg_param_ref cg_param_mat(cg_builder *g, const weight *w) {
	return builder_add_global(g, (cg_param){.mat = w});
}

cg_param_ref cg_param_vec(cg_builder *g, const buffer *v) {
	return builder_add_global(g, (cg_param){.vec = v});
}

cg_param_ref cg_param_conv(cg_builder *g, const nn_conv *c) {
	return builder_add_global(g, (cg_param){.conv = c});
}

const cg_param *cg_builder_param(const cg_builder *g, cg_param_ref ref, int layer) {
	if (ref.scope == CG_SCOPE_GLOBAL)
		return &g->globals[ref.idx];
	if (ref.scope == CG_SCOPE_LAYER)
		return &g->layer_params[(size_t)layer * g->layer_stride + ref.idx];
	return NULL;
}

static cg_value builder_make_value(size_t elems) {
	return (cg_value){.producer	 = CG_NONE,
					  .alias_of	 = CG_NONE,
					  .view_of	 = CG_NONE,
					  .slot		 = RECIPE_SLOT_NONE,
					  .rows		 = CG_ROWS_BATCH,
					  .dtype	 = CG_DTYPE_F32,
					  .residency = CG_RES_DEVICE,
					  .bind		 = CG_NONE,
					  .elems	 = elems};
}

static int builder_add_value(cg_builder *g, cg_value v) {
	cg_graph *c = g->g;
	ARR_RESERVE(c->values, c->n_values, g->cap_values);
	c->values[c->n_values] = v;
	return c->n_values++;
}

int cg_input(cg_builder *g, size_t elems, int bind) {
	cg_value v = builder_make_value(elems);
	v.external = CG_EXT_HOST_INPUT;
	v.bind	   = bind;
	return builder_add_value(g, v);
}

int cg_layer_external(cg_builder *g, size_t elems, size_t layer_stride, int bind) {
	cg_value v	   = builder_make_value(elems);
	v.external	   = CG_EXT_LAYER;
	v.bind		   = bind;
	v.layer_stride = layer_stride;
	return builder_add_value(g, v);
}

int cg_view(cg_builder *g, int parent, size_t offset, size_t elems) {
	cg_value v	  = builder_make_value(elems);
	v.view_of	  = parent;
	v.view_offset = offset;
	return builder_add_value(g, v);
}

void cg_output_value(cg_builder *g, int value, int bind) {
	if (g->n_outputs >= CG_MAX_BINDS) {
		ERROR("graph: too many outputs");
		abort();
	}
	g->outputs[g->n_outputs++] = (cg_output){value, bind};
}

static int builder_node_out(cg_builder *g, int out, size_t elems) {
	if (out == CG_NEW)
		out = builder_add_value(g, builder_make_value(elems));
	g->g->values[out].producer = g->g->n_nodes;
	return out;
}

static int builder_node_alias(cg_builder *g, int x) {
	cg_value v = builder_make_value(g->g->values[x].elems);
	v.alias_of = x;
	v.producer = g->g->n_nodes;
	return builder_add_value(g, v);
}

static int builder_emit(cg_builder *g, op_kind kind, cg_param_ref param, int out, const int *in,
						int n_in, op_params p) {
	cg_graph *c = g->g;
	ARR_RESERVE(c->nodes, c->n_nodes, g->cap_nodes);
	cg_node *n = &c->nodes[c->n_nodes++];
	memset(n, 0, sizeof(*n));
	n->kind			= kind;
	n->section		= g->current;
	n->recipe_index = CG_NONE;
	n->params		= p;
	n->param		= param;
	n->n_in			= n_in;
	n->n_out		= 1;
	n->out[0]		= out;
	for (int k = 0; k < CG_STATE_SLOTS; k++)
		n->state_dep[k] = CG_NONE;
	for (int i = 0; i < n_in; i++) {
		n->in[i] = in[i];
		c->values[in[i]].n_uses++;
	}
	return out;
}

int cg_matmul(cg_builder *g, cg_param_ref w, int in, int out, int n, int k, int rows) {
	out = builder_node_out(g, out, (size_t)rows * n);
	return builder_emit(g, OP_DF_MATMUL, w, out, (int[]){in}, 1,
						(op_params){.df_matmul = {n, k, rows}});
}

int cg_conv(cg_builder *g, cg_param_ref conv, int in, int out, int height, int width, int shift) {
	const nn_conv *c = g->globals[conv.idx].conv;
	out				 = builder_node_out(g, out, (size_t)height * width * c->oc);
	return builder_emit(g, OP_DF_CONV, conv, out, (int[]){in}, 1,
						(op_params){.df_conv = {height, width, shift}});
}

int cg_rmsnorm(cg_builder *g, cg_param_ref gamma, int in, int out, int dim, float eps, int rows) {
	out = builder_node_out(g, out, (size_t)rows * dim);
	return builder_emit(g, OP_DF_RMSNORM, gamma, out, (int[]){in}, 1,
						(op_params){.df_norm = {dim, rows, eps}});
}

int cg_rmsnorm_per_head(cg_builder *g, cg_param_ref gamma, int in, int out, int heads, int head_dim,
						float eps, int rows) {
	out = builder_node_out(g, out, (size_t)rows * heads * head_dim);
	return builder_emit(g, OP_DF_RMSNORM_PER_HEAD, gamma, out, (int[]){in}, 1,
						(op_params){.df_head_norm = {heads, head_dim, rows, eps}});
}

int cg_layernorm_modulate(cg_builder *g, int in, int scale, int out, int dim, float eps, int rows) {
	out = builder_node_out(g, out, (size_t)rows * dim);
	return builder_emit(g, OP_DF_LAYERNORM_MODULATE, (cg_param_ref){0}, out, (int[]){in, scale}, 2,
						(op_params){.df_norm = {dim, rows, eps}});
}

int cg_add(cg_builder *g, int x, int y, int dim, int rows) {
	int out = builder_node_alias(g, x);
	return builder_emit(g, OP_DF_ADD, (cg_param_ref){0}, out, (int[]){x, y}, 2,
						(op_params){.df_add = {dim, rows}});
}

int cg_gated_add(cg_builder *g, int x, int y, int gate, int dim, int rows) {
	int out = builder_node_alias(g, x);
	return builder_emit(g, OP_DF_GATED_ADD, (cg_param_ref){0}, out, (int[]){x, y, gate}, 3,
						(op_params){.df_add = {dim, rows}});
}

int cg_activate(cg_builder *g, int x, int activation) {
	int out = builder_node_alias(g, x);
	return builder_emit(g, OP_DF_ACTIVATE, (cg_param_ref){0}, out, (int[]){x}, 1,
						(op_params){.df_activate = {(int)g->g->values[x].elems, activation}});
}

int cg_ffn_activate(cg_builder *g, int gate, int up, int out, int inter, int activation, int rows) {
	out = builder_node_out(g, out, (size_t)rows * inter);
	return builder_emit(g, OP_DF_FFN_ACTIVATE, (cg_param_ref){0}, out, (int[]){gate, up}, 2,
						(op_params){.df_ffn = {inter, activation, rows}});
}

int cg_rope_table(cg_builder *g, int x, int table, int rows, int heads, int head_dim) {
	int out = builder_node_alias(g, x);
	return builder_emit(g, OP_DF_ROPE_TABLE, (cg_param_ref){0}, out, (int[]){x, table}, 2,
						(op_params){.df_rope = {rows, heads, head_dim}});
}

int cg_attention(cg_builder *g, int q, int k0, int v0, int k1, int v1, int out,
				 cg_attention_dims dims) {
	out = builder_node_out(g, out, (size_t)dims.n_q * dims.n_heads * dims.head_dim);
	int in[CG_EMIT_MAX_IN] = {q, k0, v0, k1, v1};
	return builder_emit(g, OP_DF_ATTENTION, (cg_param_ref){0}, out, in, k1 == CG_NONE ? 3 : 5,
						(op_params){.df_attention = dims});
}

int cg_upsample_shortcut_add(cg_builder *g, int in, int acc, cg_upsample_dims dims) {
	int out = builder_node_alias(g, acc);
	return builder_emit(g, OP_DF_UPSAMPLE_SHORTCUT_ADD, (cg_param_ref){0}, out, (int[]){in, acc}, 2,
						(op_params){.df_upsample = dims});
}

int cg_value_root(const cg_graph *c, int value) {
	for (;;) {
		const cg_value *v	   = &c->values[value];
		int				parent = v->alias_of != CG_NONE ? v->alias_of : v->view_of;
		if (parent == CG_NONE)
			return value;
		value = parent;
	}
}

static bool builder_is_defined(const cg_graph *c, int value, int node) {
	const cg_value *v = &c->values[value];
	if (v->producer != CG_NONE)
		return v->producer < node;
	if (v->external != CG_EXT_NONE)
		return true;
	int parent = v->alias_of != CG_NONE ? v->alias_of : v->view_of;
	return parent != CG_NONE && builder_is_defined(c, parent, node);
}

typedef struct {
	size_t in[4], out;
} size_expect;

static size_expect builder_expected_sizes(const cg_node *n) {
	const op_params *p = &n->params;
	switch (n->kind) {
	case OP_DF_MATMUL:
		return (size_expect){.in  = {(size_t)p->df_matmul.rows * p->df_matmul.k},
							 .out = (size_t)p->df_matmul.rows * p->df_matmul.n};
	case OP_DF_RMSNORM: {
		size_t total = (size_t)p->df_norm.rows * p->df_norm.dim;
		return (size_expect){.in = {total}, .out = total};
	}
	case OP_DF_LAYERNORM_MODULATE: {
		size_t total = (size_t)p->df_norm.rows * p->df_norm.dim;
		return (size_expect){.in = {total, (size_t)p->df_norm.dim}, .out = total};
	}
	case OP_DF_RMSNORM_PER_HEAD: {
		size_t total =
			(size_t)p->df_head_norm.rows * p->df_head_norm.heads * p->df_head_norm.head_dim;
		return (size_expect){.in = {total}, .out = total};
	}
	case OP_DF_ADD: {
		size_t total = (size_t)p->df_add.rows * p->df_add.dim;
		return (size_expect){.in = {total, total}};
	}
	case OP_DF_GATED_ADD: {
		size_t total = (size_t)p->df_add.rows * p->df_add.dim;
		return (size_expect){.in = {total, total, (size_t)p->df_add.dim}};
	}
	case OP_DF_FFN_ACTIVATE: {
		size_t total = (size_t)p->df_ffn.rows * p->df_ffn.inter;
		return (size_expect){.in = {total, total}, .out = total};
	}
	case OP_DF_ROPE_TABLE:
		return (size_expect){
			.in = {(size_t)p->df_rope.rows * p->df_rope.heads * p->df_rope.head_dim,
				   (size_t)p->df_rope.rows * p->df_rope.head_dim}};
	case OP_DF_ATTENTION: {
		const df_attention_dims *a	 = &p->df_attention;
		size_t					 row = (size_t)a->n_heads * a->head_dim;
		return (size_expect){.in  = {a->n_q * row, a->n0 * row, a->n0 * row, a->n1 * row},
							 .out = a->n_q * row};
	}
	default:
		return (size_expect){0};
	}
}

static int builder_check_size(const cg_graph *c, int node, const char *what, int value,
							  size_t expect) {
	if (!expect || c->values[value].elems == expect)
		return 0;
	ERROR("graph: n%d %s %s has %zu floats, expected %zu", node,
		  recipe_op_name(c->nodes[node].kind), what, c->values[value].elems, expect);
	return 1;
}

static int builder_validate_node(const cg_graph *c, int index) {
	const cg_node *n	  = &c->nodes[index];
	int			   errors = 0;
	for (int i = 0; i < n->n_in; i++)
		if (!builder_is_defined(c, n->in[i], index)) {
			ERROR("graph: n%d %s reads v%d before it is defined", index, recipe_op_name(n->kind),
				  n->in[i]);
			errors++;
		}
	size_expect e = builder_expected_sizes(n);
	for (int i = 0; i < n->n_in && i < 4; i++)
		errors += builder_check_size(c, index, "input", n->in[i], e.in[i]);
	if (n->n_out > 0)
		errors += builder_check_size(c, index, "output", n->out[0], e.out);
	return errors;
}

int cg_builder_validate(const cg_builder *g) {
	const cg_graph *c	   = g->g;
	int				errors = 0;
	for (int i = 0; i < c->n_nodes; i++)
		errors += builder_validate_node(c, i);
	for (int o = 0; o < g->n_outputs; o++)
		if (g->outputs[o].bind < 0 || g->outputs[o].bind >= CG_MAX_BINDS)
			errors++;
	for (int v = 0; v < c->n_values; v++) {
		const cg_value *val = &c->values[v];
		if (val->external != CG_EXT_NONE && (val->bind < 0 || val->bind >= CG_MAX_BINDS)) {
			ERROR("graph: v%d has invalid bind %d", v, val->bind);
			errors++;
		}
	}
	return errors;
}
