#include "recipe/internal.h"

typedef status_code (*op_handler)(exec_ctx *ctx);

typedef struct {
	const recipe_op *op;
	op_handler		 fn;
	uint32_t		 caps;
	int				 node;
	int				 op_index;
	_Atomic uint64_t us;
	_Atomic uint64_t calls;
} exec_step;

typedef struct {
	exec_step		*steps;
	int				 n_steps;
	const recipe_op *ops;
	int				 n_ops;
} exec_schedule;

struct exec_plan {
	exec_schedule  pre;
	exec_schedule  post;
	exec_schedule *layer;
	int			   n_variants;
	int			   n_layers;
	int			   post_reads_logits;
	weight_ref	 **wtab;
};

typedef struct {
	struct model		   *m;
	row_desc				rows;
	struct compute_scratch *s;
	batch_scratch		   *bs;
	int						li;
	int						flash_attn;
	const struct cg_set	   *graphs;
	const struct exec_plan *plan;
} layer_run;

static const op_handler g_op_handlers[OP_KIND_COUNT];

uint32_t ple_inject_caps(const backend *a) {
	return (backend_has_cap(a, BCAP_IS_HOST) ? 0 : PLE_CAP_HEAP) |
		   (a->matmul_ffn_down_batch ? PLE_CAP_FUSED_DOWN : 0) |
		   (a->rmsnorm_add_batch ? PLE_CAP_FUSED_NORM : 0);
}

uint32_t moe_shared_caps(const backend *a) {
	return (a->ffn_activate_fused_batch ? MOE_SH_FUSED_ACT : 0) |
		   (a->matmul_multi_batch ? MOE_SH_MULTI : 0) | (a->ffn_activate_batch ? MOE_SH_ACT : 0) |
		   (a->copy_buffer ? MOE_SH_COPY : 0);
}

uint32_t mla_qkv_caps(const backend *a) {
	return a->matmul_multi_batch ? MLA_QKV_MULTI : 0;
}

uint32_t gdn_caps(const backend *a) {
	return a && a->gated_delta_net && backend_has_cap(a, BCAP_IS_HOST) ? GDN_NATIVE : 0;
}

static uint32_t exec_pick_caps(op_kind kind, const backend *a) {
	switch (kind) {
	case OP_PLE_PROJ_INJECT:
		return ple_inject_caps(a) | EXEC_CAPS_RESOLVED;
	case OP_MOE_SHARED:
		return moe_shared_caps(a) | EXEC_CAPS_RESOLVED;
	case OP_MLA_QKV_PROJ_FUSED:
		return mla_qkv_caps(a) | EXEC_CAPS_RESOLVED;
	case OP_GATED_DELTA_NET:
		return gdn_caps(a) | EXEC_CAPS_RESOLVED;
	default:
		return 0;
	}
}

static inline int backend_native_or_host(const backend *a, int has_fn) {
	return has_fn || backend_has_cap(a, BCAP_IS_HOST);
}

static op_handler exec_pick_variant(const recipe_op *op, const backend *a) {
	switch (op->kind) {
	case OP_MATMUL_FFN_DOWN:
		return ffn_down_native(a) ? op_matmul_ffn_down_fused : op_matmul_ffn_down_composed;
	case OP_ROPE_EXT:
		return a->rope_ext_batch && a->rope_ext ? op_rope_ext_batched : NULL;
	case OP_QK_NORM_ROPE:
		return a->qk_norm_rope ? NULL : op_qk_norm_rope_unfused;
	case OP_PARTIAL_ROPE_QK:
		return backend_native_or_host(a, a->partial_rope_qk != NULL) ? op_partial_rope_qk_native
																	 : NULL;
	case OP_ATTENTION:
		return op->u.attention.sliding_window > 0 && a->attention_swa_batch ? op_attention_swa
																			: op_attention_global;
	case OP_FFN_ACTIVATE_FUSED:
		if (a->ffn_activate_fused_batch)
			return NULL;
		return backend_has_cap(a, BCAP_IS_HOST) ? op_ffn_activate_fused_host
												: op_ffn_activate_fused_device;
	case OP_SOFTCAP:
		return backend_native_or_host(a, a->softcap != NULL) ? op_softcap_native : NULL;
	case OP_SPLIT_QGATE:
		return backend_native_or_host(a, a->split_qgate != NULL) ? op_split_qgate_native : NULL;
	case OP_ATTN_OUTPUT_GATE:
		return backend_native_or_host(a, a->attn_output_gate != NULL) ? op_attn_output_gate_native
																	  : NULL;
	case OP_MATMUL_MULTI:
		return a->matmul_multi_batch ? op_matmul_multi_fused : op_matmul_multi_split;
	case OP_RMSNORM_PER_HEAD:
		if (!backend_native_or_host(a, a->rmsnorm_per_head != NULL))
			return NULL;
		return a->rmsnorm_per_head_batch ? op_rmsnorm_per_head_batch : op_rmsnorm_per_head_rows;
	case OP_RMSNORM_NOWEIGHT:
		if (!backend_native_or_host(a, a->rmsnorm_noweight_per_head != NULL))
			return NULL;
		if (a->rmsnorm_noweight_per_head_batch)
			return op_rmsnorm_noweight_ph_batch;
		if (a->rmsnorm_noweight_batch)
			return op_rmsnorm_noweight_batch;
		if (a->rmsnorm_noweight_per_head)
			return op_rmsnorm_noweight_ph_rows;
		return backend_native_or_host(a, a->rmsnorm_noweight != NULL) ? op_rmsnorm_noweight_rows
																	  : NULL;
	case OP_RMSNORM_ADD:
		if (a->rmsnorm_add_batch)
			return op_rmsnorm_add_fused;
		if (!(a->rmsnorm_add && backend_has_cap(a, BCAP_RMSNORM_ADD)))
			return op_rmsnorm_add_composed;
		return NULL;
	default:
		return NULL;
	}
}

static exec_ctx layer_run_ctx(layer_run *w) {
	model	*m		  = w->m;
	int		 li		  = w->li;
	backend *layer_be = NULL;
	if (m->mixed_backend_mode)
		layer_be = graph_set_backend(w->graphs, li) ? graph_set_backend(w->graphs, li)
													: model_layer_backend(m, li);
	else
		layer_be = m->backend;
	weight_ref **wtab_row =
		&w->plan->wtab[(size_t)((li >= 0 && li < w->plan->n_layers) ? li : 0) * WIDX_COUNT];
	return (exec_ctx){
		.op			= NULL,
		.m			= m,
		.rows		= w->rows,
		.s			= w->s,
		.bs			= w->bs,
		.li			= li,
		.flash_attn = w->flash_attn,
		.layer_be	= layer_be,
		.wtab_row	= wtab_row,
	};
}

static status_code layer_run_step(layer_run *w, exec_ctx *ctx, const recipe_op *lops, int n_ops,
								  int j, int node_id, op_handler fn, uint32_t caps, int *advance) {
	model			*m	 = w->m;
	int				 li	 = w->li;
	const recipe_op *rop = &lops[j];
	*advance			 = 1;

	int run_len = (int)rop->coalesce_run_len;
	if (run_len > 1) {
		status_code coalesced_status;
		buffer	   *ys[RECIPE_COALESCE_MAX];
		for (int q = 0; q < run_len; q++)
			ys[q] = batch_slot(w->bs, lops[j + q].out);
		int consumed = exec_matmul_run_coalesced_batch(&lops[j], run_len, m, w->s, li,
													   batch_slot(w->bs, rop->in[0]), ys,
													   w->rows.n_rows, &coalesced_status);
		if (consumed == 0 && coalesced_status == OK)
			consumed = exec_matmul_run_qonly(&lops[j], run_len, m, batch_slot(w->bs, rop->in[0]),
											 ys, li, w->rows.n_rows, &coalesced_status);
		if (consumed < 0)
			return coalesced_status;
		if (consumed > 0) {
			*advance = consumed;
			return coalesced_status;
		}
	}

	if (rop->kind == OP_RMSNORM_ADD && j + 1 < n_ops && rop->fuse_next) {
		int fused = exec_rmsnorm_add_norm(ctx, rop, &lops[j + 1]);
		if (fused > 0) {
			*advance = fused;
			return OK;
		}
	}

	if (!fn)
		return ERR_INVALID_ARG;
	ctx->op		   = rop;
	ctx->caps	   = caps;
	status_code st = fn(ctx);
	if (st != OK)
		ERROR("layer %d op[%d] node %d kind=%d (stage=%d) w_idx=%d failed: %s", li, j, node_id,
			  rop->kind, rop->stage, rop->w_idx, status_str(st));
	return st;
}

static status_code layer_run_finish(layer_run *w) {
	if (w->s->layer_cb)
		w->s->layer_cb(w->li + 1, w->m->n_layers, -1, 1, w->s->layer_cb_ud);
	if (w->s->interrupt && *w->s->interrupt)
		return ERR_INTERRUPTED;
	return OK;
}

static inline void exec_step_account(exec_step *step, uint64_t t0) {
	atomic_fetch_add_explicit(&step->us, (time_ns() - t0) / 1000, memory_order_relaxed);
	atomic_fetch_add_explicit(&step->calls, 1, memory_order_relaxed);
}

static status_code layer_run_schedule(layer_run *w) {
	const exec_schedule *sch	 = &w->plan->layer[w->graphs->layer_graph[w->li]];
	exec_ctx			 ctx	 = layer_run_ctx(w);
	const int			 prof_on = w->s->prof.enabled;
	int					 next	 = 0;
	for (int k = 0; k < sch->n_steps; k++) {
		exec_step *step = &sch->steps[k];
		if (step->op_index < next)
			continue;
		int			advance;
		uint64_t	t0 = prof_on ? time_ns() : 0;
		status_code st = layer_run_step(w, &ctx, sch->ops, sch->n_ops, step->op_index, step->node,
										step->fn, step->caps, &advance);
		if (prof_on)
			exec_step_account(step, t0);
		if (st != OK)
			return st;
		next = step->op_index + advance;
	}
	return layer_run_finish(w);
}

static status_code exec_outer_schedule(const struct exec_plan *plan, const exec_schedule *sch,
									   op_kind skip_a, op_kind skip_b, model *m, kvcache *cache,
									   compute_scratch *s, batch_scratch *bs, int pos_start,
									   int n_tokens, int flash_attn, float *logits_out) {
	const int prof_on = s->prof.enabled;
	for (int k = 0; k < sch->n_steps; k++) {
		exec_step *step = &sch->steps[k];
		if (step->op->kind == skip_a || step->op->kind == skip_b)
			continue;
		if (!step->fn)
			return ERR_INVALID_ARG;
		exec_ctx ctx = {
			.op			= step->op,
			.m			= m,
			.rows		= {.cache = cache, .pos_start = pos_start, .n_rows = n_tokens},
			.s			= s,
			.bs			= bs,
			.li			= -1,
			.flash_attn = flash_attn,
			.wtab_row	= plan->wtab,
			.logits_out = logits_out,
			.caps		= step->caps,
		};
		uint64_t	t0 = prof_on ? time_ns() : 0;
		status_code st = step->fn(&ctx);
		if (prof_on)
			exec_step_account(step, t0);
		if (st != OK) {
			ERROR("graph: node %d (%s) stage=%d failed: %s", step->node,
				  recipe_op_name(step->op->kind), step->op->stage, status_str(st));
			return st;
		}
	}
	return OK;
}

static int exec_schedule_build(exec_schedule *sch, const cg_graph *g, cg_section section,
							   const recipe_op *ops, int n_ops) {
	int begin	 = g->section_begin[section];
	sch->n_steps = g->section_begin[section + 1] - begin;
	sch->ops	 = ops;
	sch->n_ops	 = n_ops;
	sch->steps	 = xcalloc((size_t)sch->n_steps + 1, sizeof(exec_step));
	for (int k = 0; k < sch->n_steps; k++) {
		const cg_node	*n	= &g->nodes[begin + k];
		const recipe_op *op = &ops[n->recipe_index];
		if (op->kind < 0 || op->kind >= OP_KIND_COUNT)
			return CG_NONE;
		sch->steps[k].op	   = op;
		sch->steps[k].fn	   = g_op_handlers[op->kind];
		sch->steps[k].node	   = begin + k;
		sch->steps[k].op_index = n->recipe_index;
	}
	return OK;
}

static int exec_variant_impl(const cg_set *gs, int variant, int node) {
	if (variant < 0)
		return graph_set_impl(gs, 0, node);
	int impl = -1;
	for (int li = 0; li < gs->n_layers; li++) {
		if (gs->layer_graph[li] != variant)
			continue;
		int layer_impl = graph_set_impl(gs, li, node);
		if (impl >= 0 && impl != layer_impl)
			return -1;
		impl = layer_impl;
	}
	return impl;
}

static const backend *exec_variant_backend(const cg_set *gs, int variant) {
	const backend *be = NULL;
	for (int li = 0; li < gs->n_layers; li++) {
		if (gs->layer_graph[li] != variant)
			continue;
		const backend *layer_be = graph_set_backend(gs, li);
		if (be && be != layer_be)
			return NULL;
		be = layer_be;
	}
	return be;
}

static void exec_resolve_steps(exec_schedule *sch, const cg_set *gs, int variant,
							   const backend *outer_be) {
	const backend *variant_be = variant >= 0 ? exec_variant_backend(gs, variant) : outer_be;
	for (int k = 0; k < sch->n_steps; k++) {
		exec_step *step = &sch->steps[k];
		op_kind	   kind = step->op->kind;
		if (kind != OP_SCALE && kind != OP_PLE_BUILD) {
			op_handler fn = variant_be ? exec_pick_variant(step->op, variant_be) : NULL;
			if (fn)
				step->fn = fn;
			step->caps = variant_be ? exec_pick_caps(kind, variant_be) : 0;
			continue;
		}
		int impl = exec_variant_impl(gs, variant, step->node);
		if (impl < 0)
			continue;
		if (kind == OP_SCALE)
			step->fn = impl == CAP_HOST ? op_scale_host : op_scale_native;
		else
			step->fn = impl == CAP_HOST ? ple_build_host : ple_build_device;
	}
}

void exec_plan_free(struct exec_plan *p) {
	if (!p)
		return;
	free(p->pre.steps);
	free(p->post.steps);
	for (int v = 0; v < p->n_variants; v++)
		free(p->layer[v].steps);
	free(p->layer);
	free(p->wtab);
	free(p);
}

status_code exec_plan_build(model_recipe *r, const struct model *m) {
	const cg_set *gs = r->graphs;
	if (!graph_set_executable(gs)) {
		ERROR("graph: the optimized graph could not be lowered to an execution schedule");
		return ERR_UNSUPPORTED;
	}
	struct exec_plan *p = xcalloc(1, sizeof(*p));
	p->n_variants		= gs->n_graphs;
	p->n_layers			= m->n_layers;
	p->layer			= xcalloc((size_t)gs->n_graphs, sizeof(exec_schedule));
	int ok = exec_schedule_build(&p->pre, gs->graphs[0], CG_SECTION_PRE, graph_set_pre_ops(gs),
								 r->n_pre_ops) == OK &&
			 exec_schedule_build(&p->post, gs->graphs[0], CG_SECTION_POST, graph_set_post_ops(gs),
								 r->n_post_ops) == OK;
	for (int v = 0; ok && v < gs->n_graphs; v++)
		ok = exec_schedule_build(&p->layer[v], gs->graphs[v], CG_SECTION_LAYER,
								 graph_set_layer_ops(gs, gs->first_layer[v]), r->layer.n_ops) == OK;
	if (!ok) {
		ERROR("graph: execution plan failed");
		exec_plan_free(p);
		return ERR_UNSUPPORTED;
	}
	exec_resolve_steps(&p->pre, gs, -1, m->backend);
	exec_resolve_steps(&p->post, gs, -1, m->backend);
	for (int v = 0; v < gs->n_graphs; v++)
		exec_resolve_steps(&p->layer[v], gs, v, NULL);
	p->wtab = xcalloc((size_t)MAX(m->n_layers, 1) * WIDX_COUNT, sizeof(weight_ref *));
	for (int li = 0; li < m->n_layers; li++)
		for (int w = 0; w < WIDX_COUNT; w++)
			p->wtab[(size_t)li * WIDX_COUNT + w] = resolve_weight_ref(m, li, (uint8_t)w);
	for (int k = 0; k < p->post.n_steps; k++)
		p->post_reads_logits |= p->post.steps[k].op->kind == OP_LOGITS_READBACK;
	r->exec = p;
	return OK;
}

static int exec_schedule_report(exec_schedule *sch, const char *where, int variant, int *printed,
								const char *label, FILE *fp) {
	for (int k = 0; k < sch->n_steps; k++) {
		exec_step *step	 = &sch->steps[k];
		uint64_t   calls = atomic_exchange_explicit(&step->calls, 0, memory_order_relaxed);
		uint64_t   us	 = atomic_exchange_explicit(&step->us, 0, memory_order_relaxed);
		if (!calls)
			continue;
		if (!*printed) {
			fprintf(fp, "[profile] %s  per graph node (host dispatch time)\n", label);
			*printed = 1;
		}
		fprintf(fp, "  %-5s v%-2d n%-3d %-22s calls=%-6llu us=%-9llu us/call=%.1f\n", where,
				variant, step->node, recipe_op_name(step->op->kind), (unsigned long long)calls,
				(unsigned long long)us, (double)us / (double)calls);
	}
	return *printed;
}

void recipe_exec_profile_report(const model_recipe *r, const char *label, FILE *fp) {
	if (!r || !r->exec)
		return;
	int printed = 0;
	exec_schedule_report(&r->exec->pre, "pre", 0, &printed, label, fp);
	for (int v = 0; v < r->exec->n_variants; v++)
		exec_schedule_report(&r->exec->layer[v], "layer", v, &printed, label, fp);
	exec_schedule_report(&r->exec->post, "post", 0, &printed, label, fp);
}

status_code compute_forward_recipe(struct model *m, struct kvcache *cache,
								   struct compute_scratch *s, int token, int pos, int flash_attn,
								   float *logits_out) {
	const int32_t tok = token;
	return compute_forward_decode_batch_recipe(m, &cache, s, &tok, &pos, NULL, 1, flash_attn,
											   logits_out);
}

static const op_handler g_op_handlers[OP_KIND_COUNT] = {
	[OP_RMSNORM]			= op_rmsnorm,
	[OP_RMSNORM_PER_HEAD]	= op_rmsnorm_per_head,
	[OP_RMSNORM_NOWEIGHT]	= op_rmsnorm_noweight,
	[OP_RMSNORM_ADD]		= op_rmsnorm_add,
	[OP_MATMUL]				= op_matmul_plain,
	[OP_COPY_K_TO_V]		= op_copy_k_to_v,
	[OP_MATMUL_RESIDUAL]	= op_matmul_residual,
	[OP_MATMUL_MULTI]		= matmul_multi_batch_body,
	[OP_MATMUL_FFN_DOWN]	= op_matmul_ffn_down,
	[OP_ROPE]				= op_rope,
	[OP_ROPE_QK_FUSED]		= op_rope_qk_fused,
	[OP_ROPE_EXT]			= op_rope_ext,
	[OP_QK_NORM_ROPE]		= op_qk_norm_rope,
	[OP_KV_PUT]				= op_kv_put,
	[OP_ATTENTION]			= op_attention,
	[OP_ADD]				= op_add,
	[OP_SWAP]				= op_swap,
	[OP_FFN_ACTIVATE]		= op_ffn_activate,
	[OP_FFN_ACTIVATE_EX]	= op_ffn_activate_ex,
	[OP_FFN_ACTIVATE_FUSED] = op_ffn_activate_fused,
	[OP_SCALE]				= op_scale,
	[OP_SOFTCAP]			= op_softcap,
	[OP_LOGITS_READBACK]	= op_logits_readback,
	[OP_PLE_BUILD]			= ple_build_batch,
	[OP_PLE_PROJ_INJECT]	= ple_proj_inject_batch,
	[OP_MLA_Q_PROJ]			= op_mla_q_proj,
	[OP_MLA_KV_PROJ]		= op_mla_kv_proj,
	[OP_MLA_QKV_PROJ_FUSED] = op_mla_qkv_proj_fused,
	[OP_ATTENTION_MLA]		= op_attention_mla,
	[OP_MOE_ROUTER]			= moe_router_batch,
	[OP_MOE_EXPERTS]		= moe_experts_batch,
	[OP_MOE_SHARED]			= moe_shared_batch,
	[OP_SPLIT_QGATE]		= op_split_qgate,
	[OP_PARTIAL_ROPE_QK]	= op_partial_rope_qk,
	[OP_ATTN_OUTPUT_GATE]	= op_attn_output_gate,
	[OP_GATED_DELTA_NET]	= op_gated_delta_net,
	[OP_SHORTCONV]			= op_shortconv,
	[OP_HC_INIT]			= op_hc_init,
	[OP_HC_PRE]				= op_hc_pre,
	[OP_HC_POST]			= op_hc_post,
	[OP_HC_FINAL]			= op_hc_final,
	[OP_KDA]				= op_kda,
	[OP_MLA_ABSORBED]		= op_mla_absorbed,
};

static int backend_has_batch_ops(const model *m, const backend *a) {
	if (!(a->matmul_batch && a->rmsnorm_batch && a->add_batch && a->ffn_activate_batch &&
		  a->rope_batch && a->rope_qk_batch && a->attention_batch))
		return 0;
	int has_sliding = m->recipe ? m->recipe->has_sliding_layers : model_has_sliding_layers(m);
	if (has_sliding && !a->attention_swa_batch)
		return 0;
	return 1;
}

static int backend_supports_batch_ops(const model *m) {
	if (!model_mixed_backend_mode(m))
		return backend_has_batch_ops(m, m->backend);
	for (int li = 0; li < m->n_layers; li++)
		if (!backend_has_batch_ops(m, model_layer_backend(m, li)))
			return 0;
	return backend_has_batch_ops(m, m->backend);
}

static status_code batch_run_post_ops(model *m, kvcache *cache, compute_scratch *s,
									  batch_scratch *bs, int n_rows, int flash_attn,
									  float *logits_out) {
	backend			   *a	  = m->backend;
	const model_recipe *r	  = m->recipe;
	const int			total = n_rows * m->vocab_size;
	bs_ensure_slot(bs, a, r->logits_slot, (size_t)total);
	status_code st = exec_outer_schedule(r->exec, &r->exec->post, OP_NONE, OP_NONE, m, cache, s, bs,
										 0, n_rows, flash_attn, logits_out);
	if (st != OK || r->exec->post_reads_logits)
		return st;
	if (!logits_out)
		return OK;
	ensure_sync(a);
	buffer *logits = batch_slot(bs, r->logits_slot);
	return logits->owner->buffer_read_f32(logits->owner, logits, logits_out, total);
}

static int rows_n_logit_rows(const row_desc *rows) {
	if (!rows->needs_logits)
		return rows->n_rows;
	int n = 0;
	for (int i = 0; i < rows->n_rows; i++)
		n += rows->needs_logits[i] != 0;
	return n;
}

static batch_scratch *part_scratch(model *m, compute_scratch *s, const backend *be) {
	if (be == m->backend)
		return s->bs;
	int dev = compute_device_index(s, be);
	if (dev < 1) {
		ERROR("partition: backend '%s' is not a device of this model", be->name);
		return NULL;
	}
	return s->mirror[dev - 1].bs;
}

static status_code part_run_transfer(model *m, compute_scratch *s, const part_plan *p,
									 const part_node *node, int n_rows, const int32_t *tokens) {
	batch_scratch *from = part_scratch(m, s, node->src);
	batch_scratch *to	= part_scratch(m, s, node->be);
	if (!from || !to)
		return ERR_UNSUPPORTED;
	for (int c = 0; c < p->n_carry; c++) {
		status_code st = compute_copy_buffer_cross(s, batch_slot(from, p->carry[c].slot),
												   batch_slot(to, p->carry[c].slot),
												   n_rows * p->carry[c].row_elems);
		if (st != OK)
			return st;
	}
	to->tokens = tokens;
	return OK;
}

static status_code part_run_layers(model *m, compute_scratch *s, const row_desc *rows,
								   const part_node *node, const int32_t *tokens, int flash_attn) {
	batch_scratch *bs = part_scratch(m, s, node->be);
	if (!bs)
		return ERR_UNSUPPORTED;
	bs->tokens			  = tokens;
	backend			   *a = node->be;
	const model_recipe *r = m->recipe;

	if (a->begin_batch)
		a->begin_batch(a);
	status_code st = OK;
	for (int li = node->first_layer; st == OK && li < node->first_layer + node->n_layers; li++) {
		layer_run w = {.m		   = m,
					   .rows	   = *rows,
					   .s		   = s,
					   .bs		   = bs,
					   .li		   = li,
					   .flash_attn = flash_attn,
					   .graphs	   = r->graphs,
					   .plan	   = r->exec};
		st			= layer_run_schedule(&w);
	}
	if (a->end_batch)
		a->end_batch(a);
	return st;
}

static status_code part_run_gather(model *m, compute_scratch *s, const part_plan *p,
								   const part_node *node, const row_desc *rows) {
	if (!rows->needs_logits)
		return OK;
	batch_scratch *bs  = part_scratch(m, s, node->be);
	int			   dst = 0;
	for (int src = 0; src < rows->n_rows; src++) {
		if (!rows->needs_logits[src])
			continue;
		for (int c = 0; src != dst && c < p->n_carry; c++) {
			const buffer *whole = batch_slot(bs, p->carry[c].slot);
			int			  e		= p->carry[c].row_elems;
			buffer		  from	= batch_row_view(whole, src, e);
			buffer		  to	= batch_row_view(whole, dst, e);
			status_code	  st	= compute_copy_buffer_cross(s, &from, &to, e);
			if (st != OK)
				return st;
		}
		dst++;
	}
	return OK;
}

static status_code part_run(model *m, compute_scratch *s, const row_desc *rows,
							const int32_t *tokens, int flash_attn) {
	const part_plan *p = m->recipe->part;
	for (int i = 0; i < p->n_nodes; i++) {
		const part_node *node = &p->nodes[i];
		status_code		 st	  = OK;
		switch (node->kind) {
		case PART_TRANSFER:
			st = part_run_transfer(m, s, p, node, rows->n_rows, tokens);
			break;
		case PART_COMPUTE:
			st = part_run_layers(m, s, rows, node, tokens, flash_attn);
			break;
		case PART_GATHER:
			st = part_run_gather(m, s, p, node, rows);
			break;
		}
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code compute_forward_rows(struct model *m, const row_desc *rows,
										struct compute_scratch *s, const int32_t *tokens,
										int flash_attn, float *logits_out, float *hidden_out) {
	struct kvcache *cache	  = rows->cache;
	const int		n_tokens  = rows->n_rows;
	const int		pos_start = rows->pos_start;
	if (n_tokens < 1)
		return OK;
	if (!recipe_is_decode_batchable(m)) {
		ERROR("model cannot run on the graph executor: a backend holding layers lacks batch ops");
		return ERR_UNSUPPORTED;
	}

	backend			   *a	  = m->backend;
	const model_recipe *r	  = m->recipe;
	const int			dim	  = m->dim;
	const int			n_out = rows_n_logit_rows(rows);

	status_code st = OK;

	if (model_mixed_backend_mode(m)) {
		st = compute_scratch_ensure_mirror(s, m);
		if (st != OK)
			return st;
	}

	batch_scratch *bs = batch_scratch_prepare(m, s, n_tokens);
	bs->tokens		  = tokens;

	st = embd_fill_rows(a, &m->tok_embd.buf, m->tok_embd.type, tokens, &bs->pair[r->embd_slot].b,
						n_tokens, dim, m->arch_info->has_scale_embeddings ? m->dim_sqrt : 1.0f);
	if (st != OK)
		goto done;

	st = exec_outer_schedule(r->exec, &r->exec->pre, OP_EMBD_LOOKUP, OP_SCALE_EMBEDDINGS, m, cache,
							 s, bs, pos_start, n_tokens, flash_attn, NULL);
	if (st != OK)
		goto done;

	st = part_run(m, s, rows, tokens, flash_attn);
	if (st != OK)
		goto done;
	if (hidden_out) {
		ensure_sync(a);
		buffer *x = batch_slot(bs, RECIPE_SLOT_X);
		st		  = x->owner->buffer_read_f32(x->owner, x, hidden_out, n_tokens * dim);
		goto done;
	}
	st = batch_run_post_ops(m, cache, s, bs, n_out, flash_attn, logits_out);

done:
	if (st != OK && logits_out)
		memset(logits_out, 0, (size_t)n_out * m->vocab_size * sizeof(float));
	return st;
}

int recipe_is_decode_batchable(const model *m) {
	return m->recipe && backend_supports_batch_ops(m);
}

status_code compute_forward_decode_batch_recipe(struct model *m, struct kvcache **caches,
												struct compute_scratch *s, const int32_t *tokens,
												const int *positions, const uint8_t *needs_logits,
												int n_seqs, int flash_attn, float *logits_out) {
	row_desc rows = {.n_rows	   = n_seqs,
					 .pos_start	   = positions[0],
					 .pos		   = positions,
					 .cache		   = caches[0],
					 .caches	   = caches,
					 .needs_logits = needs_logits};
	return compute_forward_rows(m, &rows, s, tokens, flash_attn, logits_out, NULL);
}

status_code compute_forward_batch_recipe(struct model *m, struct kvcache *cache,
										 struct compute_scratch *s, const int32_t *tokens,
										 int n_tokens, int pos_start, int flash_attn,
										 float *logits_out) {
	if (n_tokens < 1)
		return OK;
	scratch_run *run = &s->run;
	if (run->last_only_cap < (size_t)n_tokens) {
		run->last_only	   = xrealloc(run->last_only, (size_t)n_tokens);
		run->last_only_cap = (size_t)n_tokens;
	}
	uint8_t *last_only = run->last_only;
	memset(last_only, 0, (size_t)n_tokens);
	last_only[n_tokens - 1] = 1;
	row_desc rows			= {
		.n_rows = n_tokens, .pos_start = pos_start, .cache = cache, .needs_logits = last_only};
	return compute_forward_rows(m, &rows, s, tokens, flash_attn, logits_out, NULL);
}

status_code compute_hidden_states_recipe(struct model *m, struct kvcache *cache,
										 struct compute_scratch *s, const int32_t *tokens,
										 int n_tokens, int flash_attn, float *hidden_out) {
	row_desc rows = {.n_rows = n_tokens, .pos_start = 0, .cache = cache};
	return compute_forward_rows(m, &rows, s, tokens, flash_attn, NULL, hidden_out);
}
