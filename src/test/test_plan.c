#include "plan.h"
#include "test_core.h"

#define record_plan(label, ok, ...) record_resultf(OPFAM_PLAN, (label), (ok), __VA_ARGS__)

static int find_slot(const uint8_t *list, int n, uint8_t slot) {
	for (int i = 0; i < n; i++)
		if (list[i] == slot)
			return 1;
	return 0;
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

	op.kind = OP_SCALE;
	op.out	= RECIPE_SLOT_NONE;
	nr = nw = 0;
	ok		= recipe_op_touch(&op, reads, &nr, writes, &nw) == 1;
	ok &= nr == 1 && reads[0] == RECIPE_SLOT_X && nw == 1 && writes[0] == RECIPE_SLOT_X;
	record_plan("touch.inplace_out_none", ok, "nr=%d nw=%d", nr, nw);

	op.kind	 = OP_MATMUL_MULTI;
	op.out	 = RECIPE_SLOT_Q;
	op.w_idx = WIDX_WQ;
	nr = nw = 0;
	ok		= recipe_op_touch(&op, reads, &nr, writes, &nw) == 1;
	ok &= nr == 1 && nw == 3 && writes[0] == RECIPE_SLOT_Q && writes[1] == RECIPE_SLOT_Q + 1 &&
		  writes[2] == RECIPE_SLOT_Q + 2;
	op.w_idx = WIDX_WV;
	nw		 = 0;
	(void)recipe_op_touch(&op, reads, &nr, writes, &nw);
	ok &= nw == 2;
	record_plan("touch.matmul_multi", ok, "nw=%d", nw);

	op.kind	 = OP_MLA_QKV_PROJ_FUSED;
	op.out	 = RECIPE_SLOT_XB;
	op.w_idx = WIDX_NONE;
	nr = nw = 0;
	(void)recipe_op_touch(&op, reads, &nr, writes, &nw);
	ok = nr == 3 && nw == 3 && writes[1] == RECIPE_SLOT_FFN_GATE && writes[2] == RECIPE_SLOT_FFN_UP;
	record_plan("touch.mla_qkv_fused", ok, "nr=%d nw=%d", nr, nw);

	op.kind	 = OP_MATMUL;
	op.out	 = RECIPE_SLOT_FFN_GATE_UP;
	op.w_idx = WIDX_GATE_UP;
	nr = nw = 0;
	(void)recipe_op_touch(&op, reads, &nr, writes, &nw);
	ok = nr == 1 && nw == 3 && writes[1] == RECIPE_SLOT_FFN_GATE && writes[2] == RECIPE_SLOT_FFN_UP;
	op.out = RECIPE_SLOT_ATTN_OUT;
	nr = nw = 0;
	(void)recipe_op_touch(&op, reads, &nr, writes, &nw);
	ok &= nr == 1 && nw == 1 && writes[0] == RECIPE_SLOT_ATTN_OUT;
	record_plan("touch.matmul_gate_up_slices", ok, "nr=%d nw=%d", nr, nw);
}

static void set_mm(recipe_op *op, uint8_t out) {
	memset(op, 0, sizeof(*op));
	op->kind  = OP_MATMUL;
	op->out	  = out;
	op->stage = STAGE_ATTN;
}

static void set_op(recipe_op *op, op_kind kind, uint8_t in0, uint8_t in1, uint8_t out) {
	memset(op, 0, sizeof(*op));
	op->kind  = kind;
	op->in[0] = in0;
	op->in[1] = in1;
	op->out	  = out;
	op->stage = STAGE_ATTN;
}

static void test_fuse_legality(void) {
	recipe_op ops[8];
	int		  ok = 1;

	set_mm(&ops[0], RECIPE_SLOT_ATTN_OUT);
	ops[0].in[0]	  = RECIPE_SLOT_XB;
	ops[0].u.matmul.n = 4;
	ops[0].u.matmul.k = 4;
	set_op(&ops[1], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	int nfused = plan_fuse_residual_variant(ops, 2);
	ok &= nfused == 1 && ops[0].kind == OP_NONE && ops[1].kind == OP_MATMUL_RESIDUAL;
	record_plan("fuse.adjacent", ok, "nfused=%d", nfused);

	set_mm(&ops[0], RECIPE_SLOT_ATTN_OUT);
	ops[0].in[0]	  = RECIPE_SLOT_XB;
	ops[0].u.matmul.n = 4;
	ops[0].u.matmul.k = 4;
	set_op(&ops[1], OP_SCALE, RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_XB);
	set_op(&ops[2], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	ok = plan_fuse_residual_variant(ops, 3) == 0 && ops[0].kind == OP_MATMUL;
	record_plan("fuse.input_written_between", ok, "");

	ops[0] = (recipe_op){0};
	set_mm(&ops[0], RECIPE_SLOT_ATTN_OUT);
	ops[0].in[0]	  = RECIPE_SLOT_XB;
	ops[0].u.matmul.n = 4;
	ops[0].u.matmul.k = 4;
	set_op(&ops[1], OP_RMSNORM, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE, RECIPE_SLOT_XB2);
	set_op(&ops[2], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	ok = plan_fuse_residual_variant(ops, 3) == 0 && ops[0].kind == OP_MATMUL;
	record_plan("fuse.y_read_between", ok, "");

	set_mm(&ops[0], RECIPE_SLOT_ATTN_OUT);
	ops[0].in[0]	  = RECIPE_SLOT_XB;
	ops[0].u.matmul.n = 4;
	ops[0].u.matmul.k = 4;
	set_op(&ops[1], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE);
	set_op(&ops[2], OP_RMSNORM, RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_NONE, RECIPE_SLOT_XB2);
	ok = plan_fuse_residual_variant(ops, 3) == 0 && ops[0].kind == OP_MATMUL;
	record_plan("fuse.y_read_after", ok, "");

	set_mm(&ops[0], RECIPE_SLOT_Q);
	ops[0].in[0]			= RECIPE_SLOT_XB;
	ops[0].coalesce_run_len = 2;
	set_mm(&ops[1], RECIPE_SLOT_K);
	ops[1].in[0] = RECIPE_SLOT_XB;
	set_op(&ops[2], OP_ADD, RECIPE_SLOT_X, RECIPE_SLOT_K, RECIPE_SLOT_NONE);
	ok = plan_fuse_residual_variant(ops, 3) == 0 && ops[1].kind == OP_MATMUL;
	record_plan("fuse.run_member", ok, "");
}

void run_plan_tests(void) {
	test_touch_contents();
	test_fuse_legality();
}
