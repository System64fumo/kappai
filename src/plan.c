#include "plan.h"
#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PLAN_MAX_VARIANTS 512

typedef struct {
	op_kind kind;
	int		section;
	uint8_t reads[OP_TOUCH_MAX];
	uint8_t writes[OP_TOUCH_MAX];
	int		n_reads;
	int		n_writes;
} plan_node;

struct model_plan {
	plan_node *nodes;
	int		   n_total;
	int		   n_first_variant;
	int		   n_variants;
	int		   touched[RECIPE_SLOT_MAX];
	int		   written[RECIPE_SLOT_MAX];
	uint32_t   live_slot_mask;
	uint32_t   dead_slot_mask;
	int		   n_fallbacks;
};

static int slot_is_router_scratch(uint8_t slot) {
	return slot == RECIPE_SLOT_ROUTER_IDS || slot == RECIPE_SLOT_ROUTER_W;
}

static void plan_push_op(model_plan *p, const recipe_op *op, int section) {
	if (op->kind == OP_NONE)
		return;
	plan_node *n = &p->nodes[p->n_total++];
	memset(n, 0, sizeof(*n));
	n->kind	   = op->kind;
	n->section = section;
	recipe_op_touch(op, n->reads, &n->n_reads, n->writes, &n->n_writes);
	for (int w = 0; w < n->n_writes; w++) {
		p->touched[n->writes[w]] = 1;
		p->written[n->writes[w]] = 1;
	}
	for (int r = 0; r < n->n_reads; r++)
		p->touched[n->reads[r]] = 1;
}

static void plan_build_schedule(model_plan *p, const model_recipe *r, int variant) {
	const recipe_op *layer_ops = recipe_layer_ops(r, variant);
	for (int i = 0; i < r->n_pre_ops; i++)
		plan_push_op(p, &r->pre_ops[i], 0);
	for (int i = 0; i < r->layer.n_ops; i++)
		plan_push_op(p, &layer_ops[i], 1);
	for (int i = 0; i < r->n_post_ops; i++)
		plan_push_op(p, &r->post_ops[i], 2);
}

static void plan_slot_masks(model_plan *p) {
	for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++) {
		if (slot_is_router_scratch((uint8_t)slot))
			continue;
		if (p->touched[slot])
			p->live_slot_mask |= (1u << slot);
		else
			p->dead_slot_mask |= (1u << slot);
	}
}

static void plan_validate(const model_plan *p) {
	for (int i = 0; i < p->n_total; i++) {
		const plan_node *n = &p->nodes[i];
		for (int r = 0; r < n->n_reads; r++) {
			uint8_t slot = n->reads[r];
			if (p->written[slot] || slot_is_router_scratch(slot))
				continue;
			WARN("plan: node %d (section %d, kind %d) reads slot %u which no op in the recipe "
				 "writes",
				 i, n->section, n->kind, slot);
		}
	}
}

static void plan_fallback_scan(model_plan *p, backend *a, const struct model *m) {
	bool scanned[OP_KIND_COUNT] = {false};
	for (int i = 0; i < p->n_total; i++) {
		op_kind k = p->nodes[i].kind;
		if (k == OP_PLE_BUILD) {
			if (!scanned[k]) {
				scanned[k] = true;
				if (!ple_build_dev_path_ok(a, (struct model *)m, 1))
					backend_report_host_fallback(a, "ple_build", HFB_WEIGHT_TYPE, NULL);
			}
			continue;
		}
		if (k == OP_FFN_ACTIVATE_FUSED && !backend_has_cap(a, BCAP_IS_HOST) &&
			!a->ffn_activate_fused_batch) {
			if (!scanned[k])
				backend_report_host_fallback(
					a, "ffn_activate_fused", HFB_BATCH_DESIGN,
					"batch prefill stages gate/up activation through host memory");
			scanned[k] = true;
		}
		const char *missing = recipe_op_missing_fn(a, k);
		if (!missing || scanned[k])
			continue;
		scanned[k] = true;
		backend_report_host_fallback(a, missing, HFB_OP_NOT_NATIVE, NULL);
		p->n_fallbacks++;
	}
}

static void plan_log(const model_plan *p, const struct model *m) {
	static const char *SLOT_NAMES[RECIPE_SLOT_MAX] = {
		"X",		"XB",		 "XB2",		 "ATTN_OUT",   "Q",		   "K",
		"V",		"GATE",		 "UP",		 "ACT",		   "GATE_UP",  "RESID_TMP",
		"slot12",	"slot13",	 "LOGITS",	 "ROUTER_IDS", "ROUTER_W", "HYB_PROJ",
		"HYB_GATE", "HYB_ALPHA", "HYB_BETA", "HC",		   "HC_COEF",
	};
	char dead[256];
	dead[0]	   = 0;
	size_t off = 0;
	for (int s = 0; s < RECIPE_SLOT_MAX && off < sizeof(dead) - 1; s++) {
		if (p->dead_slot_mask & (1u << s))
			off += (size_t)snprintf(dead + off, sizeof(dead) - off, "%s,", SLOT_NAMES[s]);
	}
	if (off > 0)
		dead[off - 1] = 0;

	INFO("plan '%s': %d ops in schedule over %d layer variants, %d/%d scratch slots live",
		 m->arch_info->gguf_name, p->n_first_variant, p->n_variants,
		 __builtin_popcount(p->live_slot_mask), RECIPE_SLOT_MAX);
	if (off > 0)
		INFO("plan '%s': dead scratch slots: %s", m->arch_info->gguf_name, dead);
	INFO("plan '%s': %d host-fallback decisions pre-resolved", m->arch_info->gguf_name,
		 p->n_fallbacks);
}

static void plan_build_all(model_plan *p, const model_recipe *r, const struct model *m) {
	int n_variants = r->per_layer_ops ? m->n_layers : 1;
	if (n_variants > PLAN_MAX_VARIANTS)
		n_variants = PLAN_MAX_VARIANTS;
	p->n_variants = n_variants;
	size_t est	  = (size_t)n_variants * (r->n_pre_ops + r->layer.n_ops + r->n_post_ops);
	if (est == 0)
		return;
	p->nodes = xmalloc(est * sizeof(plan_node));
	for (int v = 0; v < n_variants; v++) {
		plan_build_schedule(p, r, v);
		if (v == 0)
			p->n_first_variant = p->n_total;
	}
	plan_slot_masks(p);
}

static int op_blocks_slot(const recipe_op *op, uint8_t slot, int write_only) {
	uint8_t reads[OP_TOUCH_MAX];
	uint8_t writes[OP_TOUCH_MAX];
	int		nr = 0, nw = 0;
	recipe_op_touch(op, reads, &nr, writes, &nw);
	if (!write_only)
		for (int r = 0; r < nr; r++)
			if (reads[r] == slot)
				return 1;
	for (int w = 0; w < nw; w++)
		if (writes[w] == slot)
			return 1;
	return 0;
}

int plan_fuse_residual_variant(recipe_op *ops, int n_ops) {
	int n_fused = 0;
	int run_end = 0;
	for (int i = 0; i < n_ops; i++) {
		if (ops[i].kind == OP_MATMUL && ops[i].coalesce_run_len > 1) {
			int end = i + ops[i].coalesce_run_len;
			if (end > run_end)
				run_end = end;
		}
		if (i < run_end)
			continue;
		if (ops[i].kind != OP_MATMUL)
			continue;
		uint8_t y = ops[i].out;
		if (y == RECIPE_SLOT_NONE)
			continue;
		int j = -1;
		for (int k = i + 1; k < n_ops && j < 0; k++) {
			if (ops[k].kind == OP_ADD && ops[k].in[1] == y)
				j = k;
		}
		if (j < 0)
			continue;
		int		ok	 = 1;
		uint8_t xin0 = ops[i].in[0];
		for (int k = i + 1; k < j && ok; k++)
			ok = !op_blocks_slot(&ops[k], xin0, 1) && !op_blocks_slot(&ops[k], y, 0);
		for (int k = j + 1; k < n_ops && ok; k++)
			ok = !op_blocks_slot(&ops[k], y, 0);
		if (!ok)
			continue;
		recipe_op fused = mk_matmul_residual(ops[i].in[0], ops[j].in[0], ops[j].in[0], ops[i].w_idx,
											 ops[i].u.matmul.n, ops[i].u.matmul.k);
		ops[i].kind		= OP_NONE;
		ops[j]			= fused;
		n_fused++;
	}
	if (n_fused)
		recipe_refine_matmul_stages(ops, n_ops);
	return n_fused;
}

int plan_fuse_residuals(model_recipe *r, const struct model *m) {
	if (!r || !m || r->layer.n_ops <= 0)
		return 0;
	int n_fused	   = 0;
	int n_variants = r->per_layer_ops ? m->n_layers : 1;
	if (n_variants > PLAN_MAX_VARIANTS)
		n_variants = PLAN_MAX_VARIANTS;
	for (int v = 0; v < n_variants; v++) {
		backend *bv = m->mixed_backend_mode ? model_layer_backend(m, v) : m->backend;
		if (!r->per_layer_ops && m->mixed_backend_mode)
			break;
		if (!backend_has_cap(bv, BCAP_MATMUL_RESIDUAL))
			continue;
		recipe_op *ops =
			r->per_layer_ops ? &r->per_layer_ops[(size_t)v * r->layer.n_ops] : r->layer.ops;
		n_fused += plan_fuse_residual_variant(ops, r->layer.n_ops);
	}
	if (n_fused > 0) {
		INFO("plan '%s': fused %d residual adds into matmul_residual", m->arch_info->gguf_name,
			 n_fused);
		recipe_batch_slot_mask(r, m);
	}
	return n_fused;
}

void plan_attach(model_recipe *r, const struct model *m) {
	if (!r || !m || !m->backend)
		return;
	model_plan *p = xcalloc(1, sizeof(model_plan));
	plan_build_all(p, r, m);
	plan_validate(p);
	plan_fallback_scan(p, m->backend, m);
	plan_log(p, m);
	r->plan = p;
}

void plan_free(model_plan *p) {
	if (!p)
		return;
	free(p->nodes);
	free(p);
}
