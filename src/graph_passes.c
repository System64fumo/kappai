#include "graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void graph_compact(cg_graph *g) {
	int *remap = xmalloc((size_t)g->n_nodes * sizeof(int));
	int	 kept  = 0;
	int	 begin[CG_SECTION_COUNT + 1];
	for (int s = 0; s < CG_SECTION_COUNT; s++) {
		begin[s] = kept;
		for (int i = g->section_begin[s]; i < g->section_begin[s + 1]; i++) {
			if (g->nodes[i].kind == OP_NONE) {
				remap[i] = CG_NONE;
				continue;
			}
			remap[i]		 = kept;
			g->nodes[kept++] = g->nodes[i];
		}
	}
	begin[CG_SECTION_COUNT] = kept;
	memcpy(g->section_begin, begin, sizeof(begin));
	for (int v = 0; v < g->n_values; v++)
		if (g->values[v].producer != CG_NONE)
			g->values[v].producer = remap[g->values[v].producer];
	for (int i = 0; i < kept; i++)
		for (int k = 0; k < CG_STATE_SLOTS; k++)
			if (g->nodes[i].state_dep[k] != CG_NONE)
				g->nodes[i].state_dep[k] = remap[g->nodes[i].state_dep[k]];
	g->n_nodes = kept;
	graph_index_consumers(g);
	free(remap);
}

int graph_run_passes(cg_graph *g, const graph_pass *passes, int n_passes, const cg_pass_ctx *ctx,
					 int *per_pass) {
	if (g->frozen) {
		ERROR("graph: passes cannot run on a frozen graph");
		return 0;
	}
	int total = 0;
	for (int i = 0; i < n_passes; i++) {
		int changed = passes[i].run(g, ctx);
		if (changed == 0)
			continue;
		graph_compact(g);
		if (per_pass)
			per_pass[i] += changed;
#ifdef KAPPAI_DEBUG
		graph_validate(g);
#endif
		total += changed;
	}
	return total;
}

typedef int (*backend_pred)(const backend *be);

static int has_matmul_residual(const backend *be) {
	return backend_has_cap(be, BCAP_MATMUL_RESIDUAL);
}

static int has_multi_matmul(const backend *be) {
	return backend_has_cap(be, BCAP_MULTI_MATMUL);
}

static int supports_coalescing(const backend *be) {
	return be->matmul_multi_batch != NULL || backend_has_cap(be, BCAP_MATMUL_QONLY);
}

static int has_add_norm(const backend *be) {
	return be->rmsnorm_add_norm_batch != NULL;
}

static int pass_backends_ok(const cg_pass_ctx *ctx, cg_section section, backend_pred pred) {
	if (!ctx || !ctx->m)
		return 1;
	const struct model *m = ctx->m;
	if (section != CG_SECTION_LAYER || !m->mixed_backend_mode)
		return !m->backend || pred(m->backend);
	int first = ctx->layer >= 0 ? ctx->layer : 0;
	int last  = ctx->layer >= 0 ? ctx->layer + 1 : m->n_layers;
	for (int li = first; li < last; li++) {
		backend *be = model_layer_backend(m, li);
		if (be && !pred(be))
			return 0;
	}
	return 1;
}

static int value_is_carry_out(const cg_graph *g, int value) {
	for (int i = 0; i < g->n_carry; i++)
		if (g->carry[i].out == value)
			return 1;
	return 0;
}

static int in_coalesced_run(const cg_graph *g, int node) {
	const cg_node *n = &g->nodes[node];
	for (int i = g->section_begin[n->section]; i < g->section_begin[n->section + 1]; i++) {
		const cg_node *s = &g->nodes[i];
		if (s->kind != OP_MATMUL || s->coalesce_run_len <= 1)
			continue;
		if (n->recipe_index >= s->recipe_index &&
			n->recipe_index < s->recipe_index + s->coalesce_run_len)
			return 1;
	}
	return 0;
}

static int writes_slot_between(const cg_graph *g, int first, int last, uint8_t slot) {
	for (int k = first + 1; k < last; k++) {
		const cg_node *n = &g->nodes[k];
		if (n->kind == OP_NONE)
			continue;
		for (int o = 0; o < n->n_out; o++)
			if (g->values[n->out[o]].slot == slot)
				return 1;
	}
	return 0;
}

static int try_fuse_matmul_residual(cg_graph *g, int add_index) {
	cg_node *add = &g->nodes[add_index];
	if (add->kind != OP_ADD || add->section != CG_SECTION_LAYER || add->n_in != 2 ||
		add->n_out != 1)
		return 0;
	int		  y		   = add->in[CG_ADD_ADDEND];
	cg_value *yv	   = &g->values[y];
	int		  mm_index = yv->producer;
	if (mm_index == CG_NONE || yv->n_uses != 1 || yv->loop_carried || value_is_carry_out(g, y))
		return 0;
	cg_node *mm = &g->nodes[mm_index];
	if (mm->kind != OP_MATMUL || mm->section != add->section || mm->n_in != 1 || mm->n_out != 1)
		return 0;
	if (in_coalesced_run(g, mm_index))
		return 0;
	int input = mm->in[0];
	if (writes_slot_between(g, mm_index, add_index, g->values[input].slot))
		return 0;

	add->kind				= OP_MATMUL_RESIDUAL;
	add->stage				= mm->stage;
	add->w_idx				= mm->w_idx;
	add->params				= mm->params;
	add->in[CG_RESID_ACC]	= add->in[CG_ADD_ACC];
	add->in[CG_RESID_INPUT] = input;
	yv->n_uses				= 0;
	mm->kind				= OP_NONE;
	return 1;
}

int graph_fuse_matmul_residual(cg_graph *g, const cg_pass_ctx *ctx) {
	if (!pass_backends_ok(ctx, CG_SECTION_LAYER, has_matmul_residual))
		return 0;
	int changed = 0;
	for (int i = 0; i < g->n_nodes; i++)
		changed += try_fuse_matmul_residual(g, i);
	return changed;
}

int graph_refine_matmul_stages(cg_graph *g, const cg_pass_ctx *ctx) {
	(void)ctx;
	int changed = 0;
	for (int i = 0; i < g->n_nodes; i++) {
		cg_node *n = &g->nodes[i];
		if (!recipe_op_has_matmul_substage(n->kind) || n->stage != STAGE_MATMUL)
			continue;
		stage refined = recipe_matmul_substage(n->w_idx);
		if (refined == n->stage)
			continue;
		n->stage = refined;
		changed++;
	}
	return changed;
}

static int node_touches_slot(const cg_graph *g, const cg_node *n, uint8_t slot) {
	for (int j = 0; j < n->n_in + n->n_out; j++)
		if (g->values[j < n->n_in ? n->in[j] : n->out[j - n->n_in]].slot == slot)
			return 1;
	return 0;
}

static void replace_value(cg_graph *g, int from, int to) {
	for (int i = 0; i < g->n_nodes; i++)
		for (int j = 0; j < g->nodes[i].n_in; j++)
			if (g->nodes[i].in[j] == from)
				g->nodes[i].in[j] = to;
	for (int c = 0; c < g->n_carry; c++)
		if (g->carry[c].out == from)
			g->carry[c].out = to;
	for (int v = 0; v < g->n_values; v++) {
		if (g->values[v].alias_of == from)
			g->values[v].alias_of = to;
		if (g->values[v].view_of == from)
			g->values[v].view_of = to;
	}
	g->values[to].n_uses += g->values[from].n_uses - 1;
	g->values[from].n_uses = 0;
}

static void drop_scale_node(cg_graph *g, cg_node *scale) {
	replace_value(g, scale->out[0], scale->in[0]);
	scale->kind = OP_NONE;
}

static float layer_output_scale(const cg_node *scale, const cg_pass_ctx *ctx) {
	const struct model *m = ctx->m;
	if (m->arch_info->has_layer_output_scale && ctx->layer >= 0 && ctx->layer < m->n_layers &&
		m->layers[ctx->layer].layer_out_scale != 1.0f)
		return m->layers[ctx->layer].layer_out_scale;
	return scale->params.scale.scale;
}

static int fold_scale_node(cg_graph *g, const cg_pass_ctx *ctx, int index) {
	cg_node *scale = &g->nodes[index];
	uint8_t	 slot  = g->values[scale->in[0]].slot;
	float	 sc	   = layer_output_scale(scale, ctx);
	for (int k = index - 1; k >= g->section_begin[scale->section]; k--) {
		cg_node *prev = &g->nodes[k];
		if (prev->kind == OP_NONE)
			continue;
		if (prev->kind == OP_RMSNORM_ADD &&
			(g->values[prev->out[0]].slot == slot || g->values[prev->in[0]].slot == slot)) {
			prev->params.rmsnorm.out_scale *= sc;
			drop_scale_node(g, scale);
			return 1;
		}
		if (prev->kind == OP_PLE_PROJ_INJECT && g->values[prev->in[0]].slot == slot) {
			if (!ctx->m->has_per_layer_embeddings)
				continue;
			drop_scale_node(g, scale);
			return 1;
		}
		if (node_touches_slot(g, prev, slot))
			break;
	}
	return 0;
}

int graph_fuse_output_scale(cg_graph *g, const cg_pass_ctx *ctx) {
	if (!ctx || !ctx->m || ctx->layer < 0)
		return 0;
	int changed = 0;
	for (int i = 0; i < g->n_nodes; i++)
		if (g->nodes[i].kind == OP_SCALE && g->nodes[i].section == CG_SECTION_LAYER)
			changed += fold_scale_node(g, ctx, i);
	return changed;
}

static int can_merge_gate_up(const cg_graph *g, const cg_node *gate, const cg_node *up) {
	if (gate->kind != OP_MATMUL || up->kind != OP_MATMUL || gate->section != CG_SECTION_LAYER ||
		up->section != CG_SECTION_LAYER)
		return 0;
	if (gate->w_idx != WIDX_GATE || up->w_idx != WIDX_UP ||
		up->recipe_index != gate->recipe_index + 1)
		return 0;
	if (gate->n_in != 1 || up->n_in != 1 || gate->n_out != 1 || up->n_out != 1 ||
		gate->in[0] != up->in[0])
		return 0;
	return gate->params.matmul.k == up->params.matmul.k &&
		   g->values[up->out[0]].slot == g->values[gate->out[0]].slot + 1;
}

static void merge_gate_up(cg_graph *g, int gate_index) {
	cg_node	 *gate	 = &g->nodes[gate_index];
	cg_node	 *up	 = &g->nodes[gate_index + 1];
	int		  up_out = up->out[0];
	op_params params;
	memset(&params, 0, sizeof(params));
	params.matmul_multi.n		 = 2;
	params.matmul_multi.k		 = gate->params.matmul.k;
	params.matmul_multi.n_out[0] = gate->params.matmul.n;
	params.matmul_multi.n_out[1] = up->params.matmul.n;
	g->values[gate->in[0]].n_uses--;
	g->values[up_out].producer = gate_index;
	g->values[up_out].port	   = 1;
	gate->kind				   = OP_MATMUL_MULTI;
	gate->params			   = params;
	gate->n_out				   = 2;
	gate->out[1]			   = up_out;
	up->kind				   = OP_NONE;
}

int graph_fuse_gate_up_multi(cg_graph *g, const cg_pass_ctx *ctx) {
	if (!pass_backends_ok(ctx, CG_SECTION_LAYER, has_multi_matmul))
		return 0;
	int changed = 0;
	for (int i = 0; i + 1 < g->n_nodes; i++) {
		if (!can_merge_gate_up(g, &g->nodes[i], &g->nodes[i + 1]))
			continue;
		merge_gate_up(g, i);
		changed++;
	}
	return changed;
}

static int coalesce_run_at(const cg_graph *g, int index) {
	const cg_node *head = &g->nodes[index];
	int			   end	= g->section_begin[head->section + 1];
	int			   run	= 1;
	while (run < RECIPE_COALESCE_MAX && index + run < end) {
		const cg_node *next = &g->nodes[index + run];
		if (next->kind != OP_MATMUL || next->params.matmul.n == 0 || next->params.matmul.k == 0 ||
			next->recipe_index != head->recipe_index + run ||
			g->values[next->in[0]].slot != g->values[head->in[0]].slot ||
			next->params.matmul.k != head->params.matmul.k)
			break;
		run++;
	}
	return run;
}

static int coalescing_supported(const cg_pass_ctx *ctx) {
	if (!pass_backends_ok(ctx, CG_SECTION_LAYER, supports_coalescing))
		return 0;
	return !ctx || !ctx->m || !ctx->m->backend || supports_coalescing(ctx->m->backend);
}

int graph_coalesce_matmul_runs(cg_graph *g, const cg_pass_ctx *ctx) {
	int supported = coalescing_supported(ctx);
	int changed	  = 0;
	for (int i = 0; i < g->n_nodes; i++) {
		cg_node *n = &g->nodes[i];
		if (n->kind != OP_MATMUL || n->section != CG_SECTION_LAYER || n->params.matmul.n == 0 ||
			n->params.matmul.k == 0)
			continue;
		int want = supported ? coalesce_run_at(g, i) : 1;
		if (want != n->coalesce_run_len && (want > 1 || n->coalesce_run_len > 1))
			changed++;
		n->coalesce_run_len = (uint8_t)want;
	}
	return changed;
}

int graph_fuse_rmsnorm_add_norm(cg_graph *g, const cg_pass_ctx *ctx) {
	int backend_ok[CG_SECTION_COUNT];
	for (int s = 0; s < CG_SECTION_COUNT; s++)
		backend_ok[s] = -1;
	int changed = 0;
	for (int i = 0; i + 1 < g->n_nodes; i++) {
		cg_node *add  = &g->nodes[i];
		cg_node *norm = &g->nodes[i + 1];
		if (add->kind != OP_RMSNORM_ADD || norm->kind != OP_RMSNORM || add->fuse_next ||
			norm->section != add->section || norm->recipe_index != add->recipe_index + 1 ||
			norm->in[0] != add->out[0] || norm->params.rmsnorm.eps != add->params.rmsnorm.eps)
			continue;
		if (backend_ok[add->section] < 0)
			backend_ok[add->section] = pass_backends_ok(ctx, add->section, has_add_norm);
		if (!backend_ok[add->section])
			continue;
		add->fuse_next = 1;
		changed++;
	}
	return changed;
}

static uint8_t node_in_slot(const cg_graph *g, const cg_node *n, int index) {
	return g->values[n->in[index]].slot;
}

static recipe_op lower_fused_node(const cg_graph *g, const cg_node *n) {
	uint8_t residual = node_in_slot(g, n, CG_RESID_ACC);
	return mk_matmul_residual(node_in_slot(g, n, CG_RESID_INPUT), residual, residual, n->w_idx,
							  n->params.matmul.n, n->params.matmul.k);
}

static recipe_op lower_multi_node(const cg_graph *g, const cg_node *n) {
	return mk_matmul_multi2(node_in_slot(g, n, 0), g->values[n->out[0]].slot,
							g->values[n->out[1]].slot, n->w_idx, n->params.matmul_multi.n_out[0],
							n->params.matmul_multi.n_out[1], n->params.matmul_multi.k);
}

static int lowerable_rewrite(op_kind lowered, op_kind original) {
	return (lowered == OP_MATMUL_RESIDUAL && original == OP_ADD) ||
		   (lowered == OP_MATMUL_MULTI && original == OP_MATMUL);
}

static recipe_op lower_node(const cg_graph *g, const cg_node *n, const recipe_op *orig) {
	recipe_op op = *orig;
	if (n->kind != orig->kind) {
		op		= n->kind == OP_MATMUL_RESIDUAL ? lower_fused_node(g, n) : lower_multi_node(g, n);
		op.geom = orig->geom;
	}
	if (n->kind == orig->kind)
		op.stage = n->stage;
	op.coalesce_run_len = n->coalesce_run_len;
	op.fuse_next		= n->fuse_next;
	op.u				= n->params;
	return op;
}

int graph_lower_section(const cg_graph *g, cg_section section, const recipe_op *orig, int n_orig,
						recipe_op *out) {
	if (n_orig <= 0)
		return 0;
	memcpy(out, orig, (size_t)n_orig * sizeof(recipe_op));
	for (int i = 0; i < n_orig; i++)
		out[i].kind = OP_NONE;
	int n_lowered = 0;
	for (int i = g->section_begin[section]; i < g->section_begin[section + 1]; i++) {
		const cg_node *n = &g->nodes[i];
		if (n->recipe_index >= n_orig)
			return CG_NONE;
		const recipe_op *base = &orig[n->recipe_index];
		if (n->kind != base->kind && !lowerable_rewrite(n->kind, base->kind))
			return CG_NONE;
		n_lowered += n->kind != base->kind;
		out[n->recipe_index] = lower_node(g, n, base);
	}
	if (n_lowered)
		recipe_refine_matmul_stages(out, n_orig);
	return n_lowered;
}

#define N_PASSES ((int)ARRAY_LEN(OPTIMIZE_PASSES))

static const graph_pass OPTIMIZE_PASSES[] = {
	{"refine_matmul_stages", graph_refine_matmul_stages},
	{"fuse_output_scale", graph_fuse_output_scale},
	{"fuse_gate_up_multi", graph_fuse_gate_up_multi},
	{"coalesce_matmul_runs", graph_coalesce_matmul_runs},
	{"fuse_rmsnorm_add_norm", graph_fuse_rmsnorm_add_norm},
	{"fuse_matmul_residual", graph_fuse_matmul_residual},
};

static void lower_into(const cg_graph *g, cg_section section, recipe_op *ops, int n_ops) {
	if (n_ops <= 0)
		return;
	recipe_op *lowered = xcalloc((size_t)n_ops, sizeof(recipe_op));
	if (graph_lower_section(g, section, ops, n_ops, lowered) == CG_NONE)
		WARN("graph: could not lower the optimized graph; keeping the recipe ops unchanged");
	else
		memcpy(ops, lowered, (size_t)n_ops * sizeof(recipe_op));
	free(lowered);
}

static void log_pass_totals(const int *totals) {
	char   line[256] = "";
	size_t used		 = 0;
	for (int i = 0; i < N_PASSES && used < sizeof(line); i++) {
		if (!totals[i])
			continue;
		int n = snprintf(line + used, sizeof(line) - used, "%s%s=%d", used ? " " : "",
						 OPTIMIZE_PASSES[i].name, totals[i]);
		if (n < 0)
			break;
		used += (size_t)n;
	}
	DEBUG("graph: optimization passes changed: %s", used ? line : "nothing");
}

void graph_optimize_recipe(model_recipe *r, const struct model *m) {
	int totals[N_PASSES];
	memset(totals, 0, sizeof(totals));
	int n_variants = r->per_layer_ops ? MAX(m->n_layers, 1) : 1;
	for (int v = 0; v < n_variants; v++) {
		cg_graph   *g	= graph_build_variant(r, v);
		cg_pass_ctx ctx = {.m = m, .layer = r->per_layer_ops ? v : CG_NONE};
		if (graph_run_passes(g, OPTIMIZE_PASSES, N_PASSES, &ctx, totals) > 0) {
			recipe_op *ops =
				r->per_layer_ops ? &r->per_layer_ops[(size_t)v * r->layer.n_ops] : r->layer.ops;
			lower_into(g, CG_SECTION_LAYER, ops, r->layer.n_ops);
			if (v == 0) {
				lower_into(g, CG_SECTION_PRE, r->pre_ops, r->n_pre_ops);
				lower_into(g, CG_SECTION_POST, r->post_ops, r->n_post_ops);
			}
		}
		graph_free(g);
	}
	log_pass_totals(totals);
}
