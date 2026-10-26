#include "partition.h"
#include "log.h"

#include <stdlib.h>

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
	part_plan *p = xcalloc(1, sizeof(part_plan));
	p->nodes	 = xcalloc((size_t)m->n_layers * 2 + 2, sizeof(part_node));

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
