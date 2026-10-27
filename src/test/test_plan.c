#include "recipe/graph.h"
#include "recipe/internal.h"
#include "recipe/plan.h"
#include "test_core.h"

#define record_plan(label, ok, ...) record_resultf(OPFAM_PLAN, (label), (ok), __VA_ARGS__)

static void append_test_layer(op_emitter *e, const model *m, int li) {
	for (int i = 0; i <= li; i++)
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_WQ, li + 1, m->dim, STAGE_MATMUL));
}

static void test_recipe_build_layers(void) {
	model m = {.n_layers = 3, .dim = 64};
	for (int cap = 3; cap <= 5; cap += 2) {
		model_recipe *r = xcalloc(1, sizeof(*r));
		recipe_build_layers(r, &m, cap, "test", append_test_layer);
		int ok = r->layer.n_ops == 3;
		for (int li = 0; li < m.n_layers; li++) {
			for (int i = 0; i < r->layer.n_ops; i++) {
				const recipe_op *op = r->per_layer_ops + li * r->layer.n_ops + i;
				if (i <= li)
					ok &= op->kind == OP_MATMUL && op->u.matmul.n == li + 1 &&
						  op->u.matmul.k == m.dim;
				else
					ok &= op->kind == OP_NONE;
			}
			ok &= r->layer.ops[li].kind == OP_NONE;
		}
		record_plan("recipe.build_layers", ok, "cap=%d stride=%d", cap, r->layer.n_ops);
		recipe_free(r);
	}
}

static void test_recipe_ffn_activation(void) {
	for (int fused = 0; fused <= 1; fused++) {
		for (int multi = 0; multi <= 1; multi++) {
			for (int act = ACTIVATION_SILU; act <= ACTIVATION_GELU; act++) {
				recipe_op  ops[3] = {0};
				op_emitter e	  = op_emitter_make(ops, 3, "test");
				recipe_emit_ffn_activation(&e, 64, 128, fused, multi, act);
				int ok = e.count == (fused || multi ? 2 : 3);
				ok &= ops[0].in[0] == RECIPE_SLOT_XB;
				if (fused) {
					ok &= ops[0].kind == OP_MATMUL && ops[0].w_idx == WIDX_GATE_UP &&
						  ops[0].u.matmul.n == 256 && ops[0].u.matmul.k == 64 &&
						  ops[0].out == RECIPE_SLOT_FFN_GATE_UP;
				} else if (multi) {
					ok &= ops[0].kind == OP_MATMUL_MULTI && ops[0].w_idx == WIDX_GATE &&
						  ops[0].u.matmul_multi.n == 2 && ops[0].u.matmul_multi.k == 64 &&
						  ops[0].u.matmul_multi.n_out[0] == 128 &&
						  ops[0].u.matmul_multi.n_out[1] == 128 &&
						  ops[0].out == RECIPE_SLOT_FFN_GATE && ops[0].out1 == RECIPE_SLOT_FFN_UP;
				} else {
					ok &= ops[0].kind == OP_MATMUL && ops[0].w_idx == WIDX_GATE &&
						  ops[1].kind == OP_MATMUL && ops[1].w_idx == WIDX_UP &&
						  ops[0].u.matmul.n == 128 && ops[1].u.matmul.n == 128 &&
						  ops[0].u.matmul.k == 64 && ops[1].u.matmul.k == 64;
				}
				const recipe_op *activation = &ops[e.count - 1];
				ok &= activation->kind == (fused ? OP_FFN_ACTIVATE_FUSED : OP_FFN_ACTIVATE) &&
					  activation->out == RECIPE_SLOT_FFN_ACT &&
					  activation->u.ffn_act.activation == act;
				record_plan("recipe.ffn_activation", ok, "fused=%d multi=%d act=%d", fused, multi,
							act);
			}
		}
	}
}

static void test_touch_contents(void) {
	recipe_op op;
	memset(&op, 0, sizeof(op));
	op.in[0] = RECIPE_SLOT_X;
	op.in[1] = RECIPE_SLOT_XB;
	op.in[2] = RECIPE_SLOT_XB2;
	op.out	 = RECIPE_SLOT_ATTN_OUT;
	op.stage = STAGE_ATTN;

	uint8_t reads[OP_TOUCH_MAX];
	uint8_t writes[OP_TOUCH_MAX];
	int		nr, nw;
	int		ok = 1;

	op.kind = OP_EMBD_LOOKUP;
	nr = nw = 0;
	ok &= recipe_op_touch(&op, reads, &nr, writes, &nw) == 1;
	ok &= nr == 0 && nw == 1 && writes[0] == RECIPE_SLOT_ATTN_OUT;
	record_plan("touch.embd_lookup", ok, "nr=%d nw=%d", nr, nw);

	nr = nw = 0;
	op.kind = OP_KV_PUT;
	ok &= recipe_op_touch(&op, reads, &nr, writes, &nw) == 1;
	ok &= nr == 2 && nw == 0 && reads[0] == RECIPE_SLOT_X && reads[1] == RECIPE_SLOT_XB;
	record_plan("touch.kv_put_reads_only", ok, "nr=%d", nr);

	nr = nw = 0;
	op.kind = OP_ADD;
	ok &= recipe_op_touch(&op, reads, &nr, writes, &nw) == 1;
	ok &= nr == 2 && nw == 1 && writes[0] == RECIPE_SLOT_X;
	record_plan("touch.add_writes_in0", ok, "nr=%d nw=%d", nr, nw);

	ok = 1;
	for (int k = 1; k < OP_KIND_COUNT; k++) {
		op.kind = (op_kind)k;
		if (k == OP_PLE_BUILD)
			continue;
		nr = nw = 0;
		ok &= recipe_op_touch(&op, reads, &nr, writes, &nw) == 1;
		ok &= nr + nw > 0;
		for (int w = 0; w < nw; w++)
			ok &= writes[w] < RECIPE_SLOT_MAX;
	}
	record_plan("touch.table_complete", ok, "");

	op.kind	 = OP_SCALE;
	op.in[0] = RECIPE_SLOT_X;
	op.out	 = RECIPE_SLOT_X;
	nr = nw = 0;
	ok		= recipe_op_touch(&op, reads, &nr, writes, &nw) == 1;
	ok &= nr == 1 && reads[0] == RECIPE_SLOT_X && nw == 1 && writes[0] == RECIPE_SLOT_X;
	record_plan("touch.inplace_explicit", ok, "nr=%d nw=%d", nr, nw);

	op.kind	 = OP_MATMUL_MULTI;
	op.out	 = RECIPE_SLOT_Q;
	op.w_idx = WIDX_WQ;
	op.out1	 = RECIPE_SLOT_K;
	op.out2	 = RECIPE_SLOT_V;
	nr = nw = 0;
	ok		= recipe_op_touch(&op, reads, &nr, writes, &nw) == 1;
	ok &= nr == 1 && nw == 3 && writes[0] == RECIPE_SLOT_Q && writes[1] == RECIPE_SLOT_K &&
		  writes[2] == RECIPE_SLOT_V;
	op.out2 = RECIPE_SLOT_NONE;
	nw		= 0;
	(void)recipe_op_touch(&op, reads, &nr, writes, &nw);
	ok &= nw == 2;
	record_plan("touch.matmul_multi", ok, "nw=%d", nw);

	op.kind	 = OP_MLA_QKV_PROJ_FUSED;
	op.out	 = RECIPE_SLOT_XB;
	op.w_idx = WIDX_NONE;
	nr = nw = 0;
	(void)recipe_op_touch(&op, reads, &nr, writes, &nw);
	ok = nr == 1 && nw == 1 && writes[0] == RECIPE_SLOT_XB;
	record_plan("touch.mla_qkv_fused", ok, "nr=%d nw=%d", nr, nw);

	op.kind	 = OP_MATMUL;
	op.out	 = RECIPE_SLOT_FFN_GATE_UP;
	op.w_idx = WIDX_GATE_UP;
	op.in[1] = RECIPE_SLOT_NONE;
	nr = nw = 0;
	(void)recipe_op_touch(&op, reads, &nr, writes, &nw);
	ok = nr == 1 && nw == 1 && writes[0] == RECIPE_SLOT_FFN_GATE_UP;
	record_plan("touch.matmul_gate_up_primary", ok, "nr=%d nw=%d", nr, nw);

	op.kind	 = OP_COPY_K_TO_V;
	op.in[0] = RECIPE_SLOT_K;
	op.in[1] = RECIPE_SLOT_NONE;
	op.out	 = RECIPE_SLOT_V;
	nr = nw = 0;
	(void)recipe_op_touch(&op, reads, &nr, writes, &nw);
	ok = nr == 1 && nw == 1 && reads[0] == RECIPE_SLOT_K && writes[0] == RECIPE_SLOT_V;
	record_plan("touch.copy_k_to_v", ok, "nr=%d nw=%d", nr, nw);
}

static void reset_op(recipe_op *op) {
	memset(op, 0, sizeof(*op));
	for (size_t i = 0; i < sizeof(op->in) / sizeof(op->in[0]); i++)
		op->in[i] = RECIPE_SLOT_NONE;
	op->out1 = RECIPE_SLOT_NONE;
	op->out2 = RECIPE_SLOT_NONE;
}

static void set_mm(recipe_op *op, uint8_t out) {
	reset_op(op);
	op->kind  = OP_MATMUL;
	op->out	  = out;
	op->stage = STAGE_ATTN;
}

static void set_op(recipe_op *op, op_kind kind, uint8_t in0, uint8_t in1, uint8_t out) {
	reset_op(op);
	op->kind  = kind;
	op->in[0] = in0;
	op->in[1] = in1;
	op->out	  = out;
	op->stage = STAGE_ATTN;
}

static int fuse_residual_variant(recipe_op *ops, int n_ops) {
	model_recipe r;
	memset(&r, 0, sizeof(r));
	r.layer.ops				  = ops;
	r.layer.n_ops			  = n_ops;
	cg_graph		*g		  = graph_build(&r);
	const graph_pass passes[] = {{"fuse_matmul_residual", graph_fuse_matmul_residual}};
	int				 n_fused  = g ? graph_run_passes(g, passes, 1, NULL, NULL) : 0;
	recipe_op		 lowered[8];
	if (n_fused > 0 && n_ops <= 8 &&
		graph_lower_section(g, CG_SECTION_LAYER, ops, n_ops, lowered) != CG_NONE)
		memcpy(ops, lowered, (size_t)n_ops * sizeof(recipe_op));
	graph_free(g);
	return n_fused;
}

static void test_fuse_legality(void) {
	recipe_op ops[8];
	int		  ok = 1;

	set_mm(&ops[0], RECIPE_SLOT_ATTN_OUT);
	ops[0].in[0]	  = RECIPE_SLOT_XB;
	ops[0].u.matmul.n = 4;
	ops[0].u.matmul.k = 4;
	set_op(&ops[1], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	int nfused = fuse_residual_variant(ops, 2);
	ok &= nfused == 1 && ops[0].kind == OP_NONE && ops[1].kind == OP_MATMUL_RESIDUAL;
	record_plan("fuse.adjacent", ok, "nfused=%d", nfused);

	set_mm(&ops[0], RECIPE_SLOT_ATTN_OUT);
	ops[0].in[0]	  = RECIPE_SLOT_XB;
	ops[0].u.matmul.n = 4;
	ops[0].u.matmul.k = 4;
	set_op(&ops[1], OP_SCALE, RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_op(&ops[2], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	ok = fuse_residual_variant(ops, 3) == 0 && ops[0].kind == OP_MATMUL;
	record_plan("fuse.input_written_between", ok, "");

	ops[0] = (recipe_op){0};
	set_mm(&ops[0], RECIPE_SLOT_ATTN_OUT);
	ops[0].in[0]	  = RECIPE_SLOT_XB;
	ops[0].u.matmul.n = 4;
	ops[0].u.matmul.k = 4;
	set_op(&ops[1], OP_RMSNORM, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE, RECIPE_SLOT_XB2);
	set_op(&ops[2], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	ok = fuse_residual_variant(ops, 3) == 0 && ops[0].kind == OP_MATMUL;
	record_plan("fuse.y_read_between", ok, "");

	set_mm(&ops[0], RECIPE_SLOT_ATTN_OUT);
	ops[0].in[0]	  = RECIPE_SLOT_XB;
	ops[0].u.matmul.n = 4;
	ops[0].u.matmul.k = 4;
	set_op(&ops[1], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	set_op(&ops[2], OP_RMSNORM, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE, RECIPE_SLOT_XB2);
	ok = fuse_residual_variant(ops, 3) == 0 && ops[0].kind == OP_MATMUL;
	record_plan("fuse.y_read_after", ok, "");

	set_mm(&ops[0], RECIPE_SLOT_Q);
	ops[0].in[0]			= RECIPE_SLOT_XB;
	ops[0].coalesce_run_len = 2;
	set_mm(&ops[1], RECIPE_SLOT_K);
	ops[1].in[0] = RECIPE_SLOT_XB;
	set_op(&ops[2], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_K, RECIPE_SLOT_NONE);
	ok = fuse_residual_variant(ops, 3) == 0 && ops[1].kind == OP_MATMUL;
	record_plan("fuse.run_member", ok, "");
}

static void test_graph_build(void) {
	recipe_op pre[1];
	recipe_op layer[3];
	recipe_op post[1];
	set_op(&pre[0], OP_EMBD_LOOKUP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_X);
	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_mm(&layer[1], RECIPE_SLOT_ATTN_OUT);
	layer[1].in[0] = RECIPE_SLOT_XB;
	set_op(&layer[2], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	set_op(&post[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);

	model_recipe r;
	memset(&r, 0, sizeof(r));
	r.pre_ops	  = pre;
	r.n_pre_ops	  = 1;
	r.layer.ops	  = layer;
	r.layer.n_ops = 3;
	r.post_ops	  = post;
	r.n_post_ops  = 1;

	cg_graph *g	 = graph_build(&r);
	int		  ok = g && g->n_nodes == 5;
	ok &= g && g->section_begin[CG_SECTION_LAYER] == 1 && g->section_begin[CG_SECTION_POST] == 4;
	record_plan("graph.node_count_and_sections", ok, "n=%d", g ? g->n_nodes : -1);
	if (!g)
		return;

	const cg_node *add = &g->nodes[3];
	ok				   = add->kind == OP_ADD && add->n_in == 2 && add->n_out == 1;
	ok &= g->values[add->out[0]].alias_of == add->in[0];
	record_plan("graph.inplace_alias", ok, "");

	ok = g->n_carry == 1 && g->carry[0].slot == RECIPE_SLOT_X &&
		 g->carry[0].in == g->nodes[0].out[0] && g->carry[0].out == add->out[0] &&
		 g->values[g->carry[0].in].loop_carried;
	record_plan("graph.loop_carried_x", ok, "n_carry=%d", g->n_carry);

	ok = g->nodes[4].in[0] == add->out[0] && g->values[add->out[0]].n_uses == 1;
	record_plan("graph.post_reads_layer_output", ok, "");

	ok = graph_peak_live_buffers(g) == 3;
	record_plan("graph.peak_live_buffers", ok, "peak=%d", graph_peak_live_buffers(g));

	ok = graph_validate(g) == 0;
	record_plan("graph.validate_clean", ok, "");
	graph_free(g);

	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	g  = graph_build(&r);
	ok = g && graph_validate(g) == 1;
	record_plan("graph.validate_use_before_def", ok, "");
	graph_free(g);
}

static void fill_residual_recipe(model_recipe *r, recipe_op *pre, recipe_op *layer) {
	set_op(&pre[0], OP_EMBD_LOOKUP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_X);
	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_mm(&layer[1], RECIPE_SLOT_ATTN_OUT);
	layer[1].in[0]		= RECIPE_SLOT_XB;
	layer[1].u.matmul.n = 4;
	layer[1].u.matmul.k = 4;
	layer[1].w_idx		= WIDX_WO;
	set_op(&layer[2], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	memset(r, 0, sizeof(*r));
	r->pre_ops	   = pre;
	r->n_pre_ops   = 1;
	r->layer.ops   = layer;
	r->layer.n_ops = 3;
}

static void test_graph_fuse_residual(void) {
	recipe_op	 pre[1];
	recipe_op	 layer[4];
	model_recipe r;
	fill_residual_recipe(&r, pre, layer);

	cg_graph		*g		  = graph_build(&r);
	const graph_pass passes[] = {{"fuse_matmul_residual", graph_fuse_matmul_residual}};
	int				 ok = g && graph_run_passes(g, passes, 1, NULL, NULL) == 1 && g->n_nodes == 3;
	ok &= g && g->nodes[2].kind == OP_MATMUL_RESIDUAL && g->nodes[2].n_in == 2 &&
		  g->nodes[2].w_idx == WIDX_WO && g->nodes[2].params.matmul.n == 4;
	ok &= g && g->section_begin[CG_SECTION_POST] == 3 && graph_validate(g) == 0;
	record_plan("graph.fuse_residual_positive", ok, "n=%d", g ? g->n_nodes : -1);
	graph_free(g);

	fill_residual_recipe(&r, pre, layer);
	set_op(&layer[3], OP_RMSNORM, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE, RECIPE_SLOT_XB2);
	r.layer.n_ops = 4;
	g			  = graph_build(&r);
	ok			  = g && graph_run_passes(g, passes, 1, NULL, NULL) == 0 && g->n_nodes == 5;
	record_plan("graph.fuse_residual_y_read_after", ok, "");
	graph_free(g);

	fill_residual_recipe(&r, pre, layer);
	layer[3] = layer[2];
	set_op(&layer[2], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	r.layer.n_ops = 4;
	g			  = graph_build(&r);
	ok			  = g && graph_run_passes(g, passes, 1, NULL, NULL) == 0;
	record_plan("graph.fuse_residual_input_overwritten", ok, "");
	graph_free(g);

	fill_residual_recipe(&r, pre, layer);
	layer[1].coalesce_run_len = 2;
	g						  = graph_build(&r);
	ok						  = g && graph_run_passes(g, passes, 1, NULL, NULL) == 0;
	record_plan("graph.fuse_residual_coalesced_run", ok, "");
	graph_free(g);
}

static int lowered_op_equal(const recipe_op *a, const recipe_op *b) {
	return a->kind == b->kind && a->stage == b->stage && a->w_idx == b->w_idx && a->out == b->out &&
		   memcmp(a->in, b->in, sizeof(a->in)) == 0 && a->u.matmul.n == b->u.matmul.n &&
		   a->u.matmul.k == b->u.matmul.k;
}

static void test_graph_lower_layer(void) {
	recipe_op	 pre[1];
	recipe_op	 layer[4];
	recipe_op	 lowered[4];
	recipe_op	 expected[4];
	model_recipe r;
	fill_residual_recipe(&r, pre, layer);
	memcpy(expected, layer, sizeof(layer));
	expected[1].kind = OP_NONE;
	expected[2] = mk_matmul_residual(RECIPE_SLOT_XB, RECIPE_SLOT_X, RECIPE_SLOT_X, WIDX_WO, 4, 4);
	recipe_refine_matmul_stages(expected, r.layer.n_ops);
	int n_fused = 1;

	cg_graph		*g		  = graph_build(&r);
	const graph_pass passes[] = {{"fuse_matmul_residual", graph_fuse_matmul_residual}};
	int				 ok		  = g && graph_run_passes(g, passes, 1, NULL, NULL) == n_fused;
	ok &= g && graph_lower_section(g, CG_SECTION_LAYER, layer, r.layer.n_ops, lowered) == n_fused;
	for (int i = 0; ok && i < r.layer.n_ops; i++)
		ok = lowered_op_equal(&lowered[i], &expected[i]);
	record_plan("graph.lower_matches_expected_fusion", ok, "");

	int n_nodes = g ? g->section_begin[CG_SECTION_POST] - g->section_begin[CG_SECTION_LAYER] : -1;
	int n_live	= 0;
	for (int i = 0; i < r.layer.n_ops; i++)
		n_live += lowered[i].kind != OP_NONE;
	ok = g && n_nodes == n_live;
	for (int i = 0; ok && i < n_nodes; i++)
		ok = lowered[g->nodes[g->section_begin[CG_SECTION_LAYER] + i].recipe_index].kind != OP_NONE;
	record_plan("graph.lowered_ops_match_layer_nodes", ok, "nodes=%d live=%d", n_nodes, n_live);
	graph_free(g);

	g  = graph_build(&r);
	ok = g && graph_lower_section(g, CG_SECTION_LAYER, layer, r.layer.n_ops, lowered) == 0;
	for (int i = 0; ok && i < r.layer.n_ops; i++)
		ok = lowered_op_equal(&lowered[i], &layer[i]);
	record_plan("graph.lower_unchanged_identity", ok, "");
	graph_free(g);
}

static void fill_recipe(model_recipe *r, recipe_op *pre, recipe_op *layer, int n_layer,
						recipe_op *post, int n_post) {
	memset(r, 0, sizeof(*r));
	set_op(&pre[0], OP_EMBD_LOOKUP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_X);
	r->pre_ops	   = pre;
	r->n_pre_ops   = 1;
	r->layer.ops   = layer;
	r->layer.n_ops = n_layer;
	r->post_ops	   = post;
	r->n_post_ops  = n_post;
}

static void set_matmul(recipe_op *op, uint8_t in, uint8_t out, uint8_t w_idx, int n, int k) {
	set_mm(op, out);
	op->in[0]	   = in;
	op->w_idx	   = w_idx;
	op->stage	   = STAGE_MATMUL;
	op->u.matmul.n = n;
	op->u.matmul.k = k;
}

static void test_graph_pass_stages(void) {
	recipe_op	 pre[1], layer[2];
	model_recipe r;
	fill_recipe(&r, pre, layer, 2, NULL, 0);
	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_matmul(&layer[1], RECIPE_SLOT_XB, RECIPE_SLOT_ATTN_OUT, WIDX_WO, 4, 4);

	cg_graph		*g		  = graph_build(&r);
	const graph_pass passes[] = {{"refine_matmul_stages", graph_refine_matmul_stages}};
	int				 changed  = g ? graph_run_passes(g, passes, 1, NULL, NULL) : -1;
	int				 ok		  = changed == 1 && g->nodes[2].stage == STAGE_MATMUL_ATTN_OUT;
	ok &= g && graph_run_passes(g, passes, 1, NULL, NULL) == 0;
	record_plan("graph.pass_refine_stages", ok, "changed=%d", changed);
	graph_free(g);
}

static void test_graph_pass_coalesce(void) {
	recipe_op	 pre[1], layer[4];
	model_recipe r;
	fill_recipe(&r, pre, layer, 4, NULL, 0);
	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_matmul(&layer[1], RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, 4, 4);
	set_matmul(&layer[2], RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, 4, 4);
	set_matmul(&layer[3], RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, 4, 4);

	cg_graph		*g		  = graph_build(&r);
	const graph_pass passes[] = {{"coalesce_matmul_runs", graph_coalesce_matmul_runs}};
	int				 changed  = g ? graph_run_passes(g, passes, 1, NULL, NULL) : -1;
	int				 ok = changed == 2 && g->nodes[2].coalesce_run_len == 3 &&
						  g->nodes[3].coalesce_run_len == 2 && g->nodes[4].coalesce_run_len == 1;
	record_plan("graph.pass_coalesce_positive", ok, "changed=%d", changed);
	graph_free(g);

	layer[3].u.matmul.k = 8;
	g					= graph_build(&r);
	ok					= g && graph_run_passes(g, passes, 1, NULL, NULL) == 1 &&
						  g->nodes[2].coalesce_run_len == 2 && g->nodes[3].coalesce_run_len == 1;
	record_plan("graph.pass_coalesce_k_mismatch", ok, "");
	graph_free(g);
}

static void test_graph_pass_add_norm(void) {
	recipe_op	 pre[1], layer[4];
	model_recipe r;
	fill_recipe(&r, pre, layer, 4, NULL, 0);
	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_matmul(&layer[1], RECIPE_SLOT_XB, RECIPE_SLOT_ATTN_OUT, WIDX_WO, 4, 4);
	set_op(&layer[2], OP_RMSNORM_ADD, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, RECIPE_SLOT_XB2);
	set_op(&layer[3], OP_RMSNORM, RECIPE_SLOT_XB2, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);

	cg_graph		*g		  = graph_build(&r);
	const graph_pass passes[] = {{"fuse_rmsnorm_add_norm", graph_fuse_rmsnorm_add_norm}};
	int				 changed  = g ? graph_run_passes(g, passes, 1, NULL, NULL) : -1;
	int				 ok = changed == 1 && g->nodes[3].fuse_next == 1 && g->nodes[4].fuse_next == 0;
	record_plan("graph.pass_add_norm_positive", ok, "changed=%d", changed);
	graph_free(g);

	layer[3].u.rmsnorm.eps = 1.0f;
	g					   = graph_build(&r);
	ok					   = g && graph_run_passes(g, passes, 1, NULL, NULL) == 0;
	record_plan("graph.pass_add_norm_eps_mismatch", ok, "");
	graph_free(g);

	layer[3].u.rmsnorm.eps = 0.0f;
	layer[3].in[0]		   = RECIPE_SLOT_XB;
	g					   = graph_build(&r);
	ok					   = g && graph_run_passes(g, passes, 1, NULL, NULL) == 0;
	record_plan("graph.pass_add_norm_not_consumer", ok, "");
	graph_free(g);
}

static void test_graph_slot_views(void) {
	recipe_op	 pre[1], layer[3];
	model_recipe r;
	fill_recipe(&r, pre, layer, 3, NULL, 0);
	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_matmul(&layer[1], RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP, 16, 4);
	set_op(&layer[2], OP_FFN_ACTIVATE_FUSED, RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_NONE,
		   RECIPE_SLOT_FFN_ACT);

	cg_graph *g	 = graph_build(&r);
	int		  ok = g && g->n_nodes == 4 && g->nodes[2].n_out == 3;
	if (ok) {
		const cg_node  *mm	   = &g->nodes[2];
		const cg_value *parent = &g->values[mm->out[0]];
		const cg_value *gate   = &g->values[mm->out[1]];
		const cg_value *up	   = &g->values[mm->out[2]];
		ok = parent->view_of == CG_NONE && gate->view_of == mm->out[0] &&
			 up->view_of == mm->out[0] && gate->slot == RECIPE_SLOT_FFN_GATE &&
			 up->slot == RECIPE_SLOT_FFN_UP && gate->producer == 2 && up->producer == 2 &&
			 gate->port == 1 && up->port == 2;
	}
	record_plan("graph.slot_views_gate_up", ok, "");
	ok = g && graph_validate(g) == 0;
	record_plan("graph.slot_views_validate", ok, "");
	cg_range *ranges = g ? graph_value_ranges(g) : NULL;
	ok				 = ranges != NULL;
	if (ok) {
		const cg_node *mm = &g->nodes[2];
		for (int k = 1; k < mm->n_out; k++)
			ok &= ranges[mm->out[0]].start <= ranges[mm->out[k]].start &&
				  ranges[mm->out[0]].end >= ranges[mm->out[k]].end;
	}
	record_plan("graph.slot_views_range_folded", ok, "");
	free(ranges);
	graph_free(g);
}

static void test_graph_pass_gate_up(void) {
	recipe_op	 pre[1], layer[4], lowered[4];
	model_recipe r;
	fill_recipe(&r, pre, layer, 3, NULL, 0);
	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_matmul(&layer[1], RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, 8, 4);
	set_matmul(&layer[2], RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, 8, 4);

	cg_graph		*g		  = graph_build(&r);
	const graph_pass passes[] = {{"fuse_gate_up_multi", graph_fuse_gate_up_multi}};
	int				 changed  = g ? graph_run_passes(g, passes, 1, NULL, NULL) : -1;
	int				 ok = changed == 1 && g->n_nodes == 3 && g->nodes[2].kind == OP_MATMUL_MULTI &&
						  g->nodes[2].n_out == 2 && g->values[g->nodes[2].out[1]].producer == 2;
	ok &= g && graph_validate(g) == 0;
	record_plan("graph.pass_gate_up_positive", ok, "changed=%d", changed);
	if (g && graph_lower_section(g, CG_SECTION_LAYER, layer, 3, lowered) == 1)
		ok = lowered[1].kind == OP_MATMUL_MULTI && lowered[1].out == RECIPE_SLOT_FFN_GATE &&
			 lowered[1].u.matmul_multi.n_out[1] == 8 && lowered[2].kind == OP_NONE;
	else
		ok = 0;
	record_plan("graph.pass_gate_up_lowering", ok, "");
	graph_free(g);

	layer[2].out = RECIPE_SLOT_FFN_ACT;
	g			 = graph_build(&r);
	ok			 = g && graph_run_passes(g, passes, 1, NULL, NULL) == 0;
	record_plan("graph.pass_gate_up_slots_not_adjacent", ok, "");
	graph_free(g);
}

static void test_graph_pass_output_scale(void) {
	recipe_op	 pre[1], layer[4], post[1];
	model_recipe r;
	arch_info	 a;
	model		 m;
	memset(&a, 0, sizeof(a));
	memset(&m, 0, sizeof(m));
	m.arch_info = &a;
	m.n_layers	= 1;
	fill_recipe(&r, pre, layer, 4, post, 1);
	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_matmul(&layer[1], RECIPE_SLOT_XB, RECIPE_SLOT_ATTN_OUT, WIDX_WO, 4, 4);
	set_op(&layer[2], OP_RMSNORM_ADD, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, RECIPE_SLOT_X);
	layer[2].u.rmsnorm.out_scale = 2.0f;
	set_op(&layer[3], OP_SCALE, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_X);
	layer[3].u.scale.scale = 0.5f;
	set_op(&post[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);

	cg_graph		*g		  = graph_build(&r);
	const graph_pass passes[] = {{"fuse_output_scale", graph_fuse_output_scale}};
	cg_pass_ctx		 ctx	  = {.m = &m, .layer = -1};
	int				 ok		  = g && graph_run_passes(g, passes, 1, &ctx, NULL) == 0;
	record_plan("graph.pass_output_scale_shared_variant", ok, "");
	graph_free(g);

	g			= graph_build(&r);
	ctx.layer	= 0;
	int changed = g ? graph_run_passes(g, passes, 1, &ctx, NULL) : -1;
	ok			= changed == 1 && g->n_nodes == 5 && g->nodes[3].params.rmsnorm.out_scale == 1.0f;
	ok &= g && g->nodes[4].in[0] == g->nodes[3].out[0] && g->n_carry == 1 &&
		  g->carry[0].out == g->nodes[3].out[0] && graph_validate(g) == 0;
	record_plan("graph.pass_output_scale_folds", ok, "changed=%d", changed);
	graph_free(g);
}

static void test_graph_set_variants(void) {
	recipe_op pre[1];
	recipe_op per_layer[6];
	set_op(&pre[0], OP_EMBD_LOOKUP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_X);
	for (int li = 0; li < 3; li++) {
		set_op(&per_layer[li * 2], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
		set_op(&per_layer[li * 2 + 1], li == 2 ? OP_NONE : OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_XB,
			   RECIPE_SLOT_NONE);
	}

	model_recipe r;
	memset(&r, 0, sizeof(r));
	r.pre_ops		= pre;
	r.n_pre_ops		= 1;
	r.per_layer_ops = per_layer;
	r.layer.n_ops	= 2;

	cg_set *s  = graph_set_build(&r, NULL, 3);
	int		ok = s && s->n_graphs == 2 && s->layer_graph[0] == 0 && s->layer_graph[1] == 0 &&
				 s->layer_graph[2] == 1 && s->first_layer[1] == 2;
	ok &= s && s->graphs[0]->n_nodes == 3 && s->graphs[1]->n_nodes == 2;
	ok &= s && graph_set_validate(s) == 0;
	record_plan("graph.set_dedupes_identical_layers", ok, "n_graphs=%d", s ? s->n_graphs : -1);

	ok = graph_set_executable(s) && graph_set_layer_ops(s, 0)[1].kind == OP_ADD &&
		 graph_set_layer_ops(s, 1)[1].kind == OP_ADD &&
		 graph_set_layer_ops(s, 2)[1].kind == OP_NONE &&
		 graph_set_layer_ops(s, 2)[0].kind == OP_RMSNORM &&
		 graph_set_pre_ops(s)[0].kind == OP_EMBD_LOOKUP;
	record_plan("graph.set_lowers_executable_ops", ok, "");
	graph_set_free(s);
}

static int part_nodes_match(const part_plan *p, const part_node *want, int n) {
	if (!p || p->n_nodes != n)
		return 0;
	for (int i = 0; i < n; i++) {
		const part_node *a = &p->nodes[i];
		if (a->kind != want[i].kind || a->be != want[i].be || a->src != want[i].src ||
			a->first_layer != want[i].first_layer || a->n_layers != want[i].n_layers)
			return 0;
	}
	return 1;
}

static void test_partition_plan(void) {
	static backend	 be_main, be_other;
	static backend	*layer_backends[5];
	static arch_info info;
	static model	 m;
	memset(&m, 0, sizeof(m));
	memset(&info, 0, sizeof(info));
	m.arch_info		   = &info;
	m.backend		   = &be_main;
	m.dim			   = 8;
	m.n_layers		   = 5;
	m.layer_backends   = layer_backends;
	m.n_layer_backends = 5;
	for (int i = 0; i < 5; i++)
		layer_backends[i] = &be_main;

	part_plan *p		= part_plan_build(&m);
	part_node  single[] = {{PART_COMPUTE, &be_main, NULL, 0, 5},
						   {PART_GATHER, &be_main, NULL, 0, 0}};
	int		   ok		= part_nodes_match(p, single, 2) && part_plan_n_transfers(p) == 0;
	record_plan("partition.single_device_one_compute", ok, "n_nodes=%d", p ? p->n_nodes : -1);
	part_plan_free(p);

	m.mixed_backend_mode = 1;
	layer_backends[2]	 = &be_other;
	layer_backends[3]	 = &be_other;
	p					 = part_plan_build(&m);
	part_node split[]	 = {
		{PART_COMPUTE, &be_main, NULL, 0, 2},  {PART_TRANSFER, &be_other, &be_main, 0, 0},
		{PART_COMPUTE, &be_other, NULL, 2, 2}, {PART_TRANSFER, &be_main, &be_other, 0, 0},
		{PART_COMPUTE, &be_main, NULL, 4, 1},  {PART_GATHER, &be_main, NULL, 0, 0}};
	ok = part_nodes_match(p, split, 6) && part_plan_n_transfers(p) == 2;
	record_plan("partition.splits_at_device_boundaries", ok, "n_nodes=%d", p ? p->n_nodes : -1);
	part_plan_free(p);

	layer_backends[0] = &be_other;
	layer_backends[1] = &be_other;
	layer_backends[4] = &be_other;
	p				  = part_plan_build(&m);
	part_node tail[]  = {{PART_TRANSFER, &be_other, &be_main, 0, 0},
						 {PART_COMPUTE, &be_other, NULL, 0, 5},
						 {PART_TRANSFER, &be_main, &be_other, 0, 0},
						 {PART_GATHER, &be_main, NULL, 0, 0}};
	ok				  = part_nodes_match(p, tail, 4);
	record_plan("partition.transfers_in_and_out_of_main", ok, "n_nodes=%d", p ? p->n_nodes : -1);

	ok = p && p->n_carry == 1 && p->carry[0].slot == RECIPE_SLOT_X && p->carry[0].row_elems == 8;
	record_plan("partition.carry_is_x", ok, "n_carry=%d", p ? p->n_carry : -1);
	part_plan_free(p);

	info.has_hyper_connections = true;
	m.kda.hc_count			   = 4;
	p						   = part_plan_build(&m);
	ok = p && p->n_carry == 2 && p->carry[1].slot == RECIPE_SLOT_HC && p->carry[1].row_elems == 32;
	record_plan("partition.carry_adds_hc", ok, "n_carry=%d", p ? p->n_carry : -1);
	part_plan_free(p);
}

static void test_graph_set_weights(void) {
	recipe_op pre[1];
	recipe_op per_layer[4];
	set_op(&pre[0], OP_EMBD_LOOKUP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_X);
	pre[0].w_idx = WIDX_TOK_EMBD;
	for (int li = 0; li < 2; li++) {
		set_op(&per_layer[li * 2], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
		per_layer[li * 2].w_idx = WIDX_ATTN_NORM;
		set_op(&per_layer[li * 2 + 1], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_XB, RECIPE_SLOT_NONE);
	}

	static weight_ref  wrefs[2][WIDX_COUNT];
	static weight_ref *table[2 * WIDX_COUNT];
	for (int li = 0; li < 2; li++)
		for (int w = 0; w < WIDX_COUNT; w++)
			table[li * WIDX_COUNT + w] = &wrefs[li][w];

	static model m;
	memset(&m, 0, sizeof(m));
	m.n_layers		 = 2;
	m.wrefs_by_layer = table;
	static backend	be_default;
	static backend	be_pinned;
	static backend *layer_backends[2];
	layer_backends[0]  = &be_pinned;
	layer_backends[1]  = NULL;
	m.backend		   = &be_default;
	m.layer_backends   = layer_backends;
	m.n_layer_backends = 2;

	model_recipe r;
	memset(&r, 0, sizeof(r));
	r.pre_ops		= pre;
	r.n_pre_ops		= 1;
	r.per_layer_ops = per_layer;
	r.layer.n_ops	= 2;

	cg_set *s  = graph_set_build(&r, &m, 2);
	int		ok = s && s->n_graphs == 1;
	ok &= s && graph_set_weight(s, 0, 1) == &wrefs[0][WIDX_ATTN_NORM] &&
		  graph_set_weight(s, 1, 1) == &wrefs[1][WIDX_ATTN_NORM];
	ok &= s && graph_set_weight(s, 1, 0) == &wrefs[0][WIDX_TOK_EMBD];
	record_plan("graph.set_resolves_weights_per_layer", ok, "");

	ok = s && graph_set_backend(s, 0) == &be_pinned && graph_set_backend(s, 1) == &be_default;
	record_plan("graph.set_resolves_layer_backend", ok, "");

	ok = s && s->n_fallbacks == 5;
	record_plan("graph.set_scans_fallbacks_per_device", ok, "n=%d", s ? s->n_fallbacks : -1);
	graph_set_free(s);
}

static void test_graph_roundtrip(void) {
	recipe_op pre[3];
	recipe_op layer[3];
	recipe_op post[1];
	set_op(&pre[0], OP_EMBD_LOOKUP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_X);
	set_op(&pre[1], OP_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE);
	set_op(&pre[2], OP_SCALE_EMBEDDINGS, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE);
	set_op(&layer[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_op(&layer[1], OP_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE);
	set_op(&layer[2], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_XB, RECIPE_SLOT_NONE);
	set_op(&post[0], OP_RMSNORM, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);

	model_recipe r;
	memset(&r, 0, sizeof(r));
	r.pre_ops	  = pre;
	r.n_pre_ops	  = 3;
	r.layer.ops	  = layer;
	r.layer.n_ops = 3;
	r.post_ops	  = post;
	r.n_post_ops  = 1;

	static const op_kind kinds[]   = {OP_EMBD_LOOKUP, OP_SCALE_EMBEDDINGS, OP_RMSNORM, OP_ADD,
									  OP_RMSNORM};
	static const int	 indices[] = {0, 2, 0, 2, 0};

	cg_graph *g	 = graph_build(&r);
	int		  ok = g && g->n_nodes == 5 && graph_validate(g) == 0;
	for (int i = 0; ok && i < 5; i++)
		ok = g->nodes[i].kind == kinds[i] && g->nodes[i].recipe_index == indices[i];
	record_plan("graph.roundtrip_recipe_order", ok, "");
	graph_free(g);
}

static void test_graph_state(void) {
	recipe_op pre[1];
	recipe_op layer[5];
	set_op(&pre[0], OP_EMBD_LOOKUP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_X);
	set_mm(&layer[0], RECIPE_SLOT_Q);
	set_mm(&layer[1], RECIPE_SLOT_K);
	set_mm(&layer[2], RECIPE_SLOT_V);
	set_op(&layer[3], OP_KV_PUT, RECIPE_SLOT_K, RECIPE_SLOT_V, RECIPE_SLOT_NONE);
	set_op(&layer[4], OP_ATTENTION, RECIPE_SLOT_Q, RECIPE_SLOT_K, RECIPE_SLOT_XB2);
	layer[4].in[2] = RECIPE_SLOT_V;

	model_recipe r;
	memset(&r, 0, sizeof(r));
	r.pre_ops	  = pre;
	r.n_pre_ops	  = 1;
	r.layer.ops	  = layer;
	r.layer.n_ops = 5;

	cg_graph *g = graph_build(&r);
	int ok = g && g->nodes[4].state_writes == CG_STATE_KV && g->nodes[5].state_reads == CG_STATE_KV;
	ok &= g && graph_validate(g) == 0;
	record_plan("graph.state_kv_ordered", ok, "");
	graph_free(g);

	recipe_op swapped = layer[3];
	layer[3]		  = layer[4];
	layer[4]		  = swapped;
	g				  = graph_build(&r);
	ok				  = g && graph_validate(g) == 1;
	record_plan("graph.state_read_before_write", ok, "");
	graph_free(g);
}

#define PARITY_TOKENS 12
#define PARITY_CTX 128
#define PARITY_SEQS 2

typedef struct {
	backend		   *be;
	model			m;
	compute_scratch s;
	kvcache			kv[PARITY_SEQS];
	int				n_kv;
	int				has_scratch;
} parity_sess;

typedef float *(*parity_fn)(parity_sess *p, int *count);

static void parity_close(parity_sess *p) {
	if (p->has_scratch)
		compute_scratch_free(&p->s);
	for (int i = 0; i < p->n_kv; i++)
		kvcache_free(&p->kv[i]);
	model_free(&p->m);
	backend_destroy(p->be);
}

static int parity_open(parity_sess *p, const char *path) {
	memset(p, 0, sizeof(*p));
	if (backend_create("cpu", 0, &p->be) != OK)
		return 0;
	if (model_load_backend_ex_repack(&p->m, path, p->be, 0, NULL) != OK) {
		backend_destroy(p->be);
		return 0;
	}
	int ok = p->m.recipe && graph_set_executable(p->m.recipe->graphs);
	while (ok && p->n_kv < PARITY_SEQS) {
		ok = kvcache_init(&p->kv[p->n_kv], &p->m, PARITY_CTX, KV_QUANT_F16) == OK;
		p->n_kv += ok;
	}
	if (ok) {
		compute_scratch_init(&p->s);
		p->has_scratch = 1;
		ok			   = compute_scratch_ensure(&p->s, &p->m, PARITY_CTX) == OK;
	}
	if (!ok)
		parity_close(p);
	return ok;
}

static void parity_tokens(int32_t *toks) {
	for (int i = 0; i < PARITY_TOKENS; i++)
		toks[i] = (int32_t)((i * 7 + 3) % 32);
}

static float *parity_finish(float *out, int ok, int n_floats, int *count) {
	if (ok) {
		*count = n_floats;
		return out;
	}
	free(out);
	return NULL;
}

static float *parity_decode(parity_sess *p, int *count) {
	int32_t toks[PARITY_TOKENS];
	parity_tokens(toks);
	int	   vocab = p->m.vocab_size;
	float *out	 = xcalloc((size_t)PARITY_TOKENS * (size_t)vocab, sizeof(float));
	int	   ok	 = 1;
	for (int i = 0; ok && i < PARITY_TOKENS; i++)
		ok = compute_forward(&p->m, &p->kv[0], &p->s, toks[i], i, 0, out + (size_t)i * vocab) == OK;
	return parity_finish(out, ok, PARITY_TOKENS * vocab, count);
}

static float *parity_batch(parity_sess *p, int *count) {
	int32_t toks[PARITY_TOKENS];
	parity_tokens(toks);
	int	   vocab = p->m.vocab_size;
	float *out	 = xcalloc((size_t)vocab, sizeof(float));
	int ok = compute_forward_batch(&p->m, &p->kv[0], &p->s, toks, PARITY_TOKENS, 0, 0, out) == OK;
	return parity_finish(out, ok, vocab, count);
}

static float *parity_multiseq(parity_sess *p, int *count) {
	int32_t toks[PARITY_TOKENS];
	parity_tokens(toks);
	kvcache *caches[PARITY_SEQS] = {&p->kv[0], &p->kv[1]};
	int		 vocab				 = p->m.vocab_size;
	float	*out = xcalloc((size_t)PARITY_TOKENS * PARITY_SEQS * (size_t)vocab, sizeof(float));
	int		 ok	 = 1;
	for (int i = 0; ok && i < PARITY_TOKENS; i++) {
		int32_t step_tokens[PARITY_SEQS] = {toks[i], toks[(i + 5) % PARITY_TOKENS]};
		int		positions[PARITY_SEQS]	 = {i, i};
		ok = compute_forward_decode_batch(&p->m, caches, &p->s, step_tokens, positions, NULL,
										  PARITY_SEQS, 0,
										  out + (size_t)i * PARITY_SEQS * vocab) == OK;
		for (int q = 0; q < PARITY_SEQS; q++)
			caches[q]->n_pos = i + 1;
	}
	return parity_finish(out, ok, PARITY_TOKENS * PARITY_SEQS * vocab, count);
}

static float *parity_collect(const char *path, parity_fn run, int *count) {
	*count = -1;
	parity_sess p;
	if (!parity_open(&p, path))
		return NULL;
	float *out = run(&p, count);
	parity_close(&p);
	return out;
}

static void test_graph_executor_runs(const char *arch, const char *path) {
	static const struct {
		const char *name;
		parity_fn	run;
	} RUNS[] = {{"decode", parity_decode}, {"batch", parity_batch}, {"multiseq", parity_multiseq}};
	for (size_t i = 0; i < ARRAY_LEN(RUNS); i++) {
		int	   n_first, n_second;
		float *first  = parity_collect(path, RUNS[i].run, &n_first);
		float *second = parity_collect(path, RUNS[i].run, &n_second);
		int	   ok	  = first && second && n_first > 0 && n_first == n_second &&
						count_nonfinite(first, n_first) == 0 &&
						memcmp(first, second, (size_t)n_first * sizeof(float)) == 0;
		char   label[96];
		snprintf(label, sizeof(label), "graph.executor_%s_%s", arch, RUNS[i].name);
		record_plan(label, ok, "first=%d second=%d", n_first, n_second);
		free(first);
		free(second);
	}
}

static int mem_plan_aliases_live(const mem_plan *mp, const cg_graph *g) {
	cg_range *ranges = graph_value_ranges(g);
	int		  ok	 = 1;
	for (int v = 0; ok && v < g->n_values; v++) {
		if (ranges[v].end < 0)
			continue;
		for (int w = v + 1; ok && w < g->n_values; w++) {
			uint8_t sv = g->values[v].slot;
			uint8_t sw = g->values[w].slot;
			if (ranges[w].end < 0 || sv == sw || !mp->planned[sv] || !mp->planned[sw])
				continue;
			int overlap = ranges[v].start <= ranges[w].end && ranges[w].start <= ranges[v].end;
			ok			= !(overlap && mp->rep[sv] == mp->rep[sw]);
		}
	}
	free(ranges);
	return ok;
}

static int mem_plan_keeps_swaps_apart(const mem_plan *mp, const cg_graph *g) {
	for (int i = 0; i < g->n_nodes; i++) {
		const cg_node *n = &g->nodes[i];
		if (n->kind != OP_SWAP)
			continue;
		for (int j = 0; j < n->n_in + n->n_out; j++) {
			uint8_t slot = g->values[j < n->n_in ? n->in[j] : n->out[j - n->n_in]].slot;
			for (int k = 0; k < RECIPE_SLOT_MAX; k++)
				if (k != slot && mp->planned[k] && mp->rep[k] == slot)
					return 0;
			if (mp->rep[slot] != slot)
				return 0;
		}
	}
	return 1;
}

static void test_mem_plan_aliasing(const char *arch, const char *path) {
	parity_sess p;
	int			opened = parity_open(&p, path);
	int			ok	   = opened;
	mem_plan   *mp	   = ok ? mem_plan_build(p.m.recipe->graphs, &p.m) : NULL;
	ok				   = ok && mp;
	for (int v = 0; ok && v < p.m.recipe->graphs->n_graphs; v++)
		ok = mem_plan_aliases_live(mp, p.m.recipe->graphs->graphs[v]) &&
			 mem_plan_keeps_swaps_apart(mp, p.m.recipe->graphs->graphs[v]);
	char label[96];
	snprintf(label, sizeof(label), "mem_plan.no_live_aliasing_%s", arch);
	record_plan(label, ok, "buffers=%d slots=%d", mp ? mp->n_buffers : -1, mp ? mp->n_slots : -1);
	mem_plan_free(mp);
	if (opened)
		parity_close(&p);
}

static void test_scratch_storage(void) {
	parity_sess p;
	if (!parity_open(&p, synth_chat_model_path)) {
		record_plan("scratch.storage", 0, "fixture initialization failed");
		return;
	}
	record_plan("scratch.scale_lazy", p.s.run.scale_tmp.p == NULL && p.s.run.scale_tmp.cap == 0,
				"");
	buffer		b		 = {0};
	int			ok		 = p.be->buffer_alloc_scratch(p.be, 8 * sizeof(float), &b) == OK;
	const float input[8] = {1, -2, 3, -4, 5, -6, 7, -8};
	const int	sizes[]	 = {4, 8, 2};
	for (size_t i = 0; ok && i < ARRAY_LEN(sizes); i++) {
		float output[8];
		ok = p.be->buffer_write_f32(p.be, &b, input, sizes[i]) == OK &&
			 scale_inplace_fallback(&p.s, p.be, &b, sizes[i], 2.0f) == OK &&
			 p.be->buffer_read_f32(p.be, &b, output, sizes[i]) == OK;
		for (int j = 0; ok && j < sizes[i]; j++)
			ok &= output[j] == 2.0f * input[j];
	}
	record_plan("scratch.scale_growth_reuse", ok && p.s.run.scale_tmp.cap == 8, "");
	if (b.owner)
		b.owner->buffer_free(b.owner, &b);

	const int rows[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 32};
	ok				 = 1;
	for (size_t i = 0; i < ARRAY_LEN(rows); i++) {
		batch_scratch *bs	  = batch_scratch_prepare(&p.m, &p.s, rows[i]);
		int			   cached = 0;
		for (int j = 0; j < bs->layouts.n; j++)
			cached |= bs->layout == &bs->layouts.entries[j];
		ok &= cached && bs->layout->bucket == mem_rows_bucket(rows[i]);
		for (int slot = 0; slot < RECIPE_SLOT_MAX; slot++)
			if (bs->layout->cap[slot])
				ok &= bs->pair[slot].b.size == bs->layout->cap[slot];
	}
	record_plan("scratch.layout_cache_rebinding", ok, "");
	ok = compute_scratch_ensure(&p.s, &p.m, PARITY_CTX + 1) == OK && p.s.run.scale_tmp.p == NULL &&
		 p.s.run.scale_tmp.cap == 0;
	record_plan("scratch.scale_reset", ok, "");
	parity_close(&p);
}

void run_plan_tests(void) {
	test_recipe_build_layers();
	test_recipe_ffn_activation();
	test_touch_contents();
	test_fuse_legality();
	test_graph_build();
	test_graph_state();
	test_graph_roundtrip();
	test_graph_fuse_residual();
	test_graph_lower_layer();
	test_graph_pass_stages();
	test_graph_pass_coalesce();
	test_graph_pass_add_norm();
	test_graph_pass_gate_up();
	test_graph_slot_views();
	test_graph_pass_output_scale();
	test_graph_set_variants();
	test_graph_set_weights();
	test_partition_plan();
	test_scratch_storage();
	test_graph_executor_runs("llama", synth_chat_model_path);
	test_graph_executor_runs("lfm2", synth_lfm2_model_path);
	test_graph_executor_runs("dsa", synth_dsa_model_path);
	test_mem_plan_aliasing("llama", synth_chat_model_path);
	test_mem_plan_aliasing("lfm2", synth_lfm2_model_path);
	test_mem_plan_aliasing("dsa", synth_dsa_model_path);
}
