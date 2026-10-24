#include "recipe/internal.h"

static stage matmul_substage(uint8_t w_idx);

typedef struct {
	const char *fn_name;
	size_t		slot0;
	size_t		slot1;
	uint32_t	batch_extra_mask;
	int8_t		reads[4];
	int8_t		writes[4];
	const char *name;
	uint8_t		state_reads;
	uint8_t		state_writes;
} op_requirement;

#define OP_NO_SLOT ((size_t)-1)

#define OP_FN_SLOT(fn) offsetof(backend, fn)

static const op_requirement OP_REQUIREMENTS[OP_KIND_COUNT] =
	{
		[OP_NONE] =
			{
				.name = "none",
			},
		[OP_PLE_BUILD] =
			{
				.name		  = "ple_build",
				.state_writes = CG_STATE_PLE,
			},
		[OP_EMBD_LOOKUP] = {.name	 = "embd_lookup",
							.fn_name = "embd_lookup",
							.slot0	 = OP_FN_SLOT(embd_lookup),
							.slot1	 = OP_NO_SLOT,
							.writes	 = {OP_TOUCH_OUT}},
		[OP_SCALE_EMBEDDINGS] =
			{
				.name	 = "scale_embeddings",
				.fn_name = "scale_inplace",
				.slot0	 = OP_FN_SLOT(scale_inplace),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_SCALE] =
			{
				.name	 = "scale",
				.fn_name = "scale_inplace",
				.slot0	 = OP_FN_SLOT(scale_inplace),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_RMSNORM] =
			{
				.name	 = "rmsnorm",
				.fn_name = "rmsnorm",
				.slot0	 = OP_FN_SLOT(rmsnorm),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_RMSNORM_PER_HEAD] =
			{
				.name	 = "rmsnorm_per_head",
				.fn_name = "rmsnorm_per_head",
				.slot0	 = OP_FN_SLOT(rmsnorm_per_head),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_RMSNORM_NOWEIGHT] =
			{
				.name	 = "rmsnorm_noweight",
				.fn_name = "rmsnorm_noweight",
				.slot0	 = OP_FN_SLOT(rmsnorm_noweight),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_RMSNORM_ADD] =
			{
				.name	 = "rmsnorm_add",
				.fn_name = "rmsnorm_add",
				.slot0	 = OP_FN_SLOT(rmsnorm_add),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_MATMUL] =
			{
				.name	 = "matmul",
				.fn_name = "matmul",
				.slot0	 = OP_FN_SLOT(matmul),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_MATMUL_MULTI] =
			{
				.name	 = "matmul_multi",
				.fn_name = "matmul",
				.slot0	 = OP_FN_SLOT(matmul),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT, OP_TOUCH_OUT1, OP_TOUCH_OUT2},
			},
		[OP_MATMUL_RESIDUAL] =
			{
				.name			  = "matmul_residual",
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_RESID_TMP),
				.reads			  = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes			  = {OP_TOUCH_OUT},
			},
		[OP_MATMUL_FFN_DOWN] =
			{
				.name			  = "matmul_ffn_down",
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_FFN_ACT),
				.reads			  = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes			  = {OP_TOUCH_OUT},
			},
		[OP_MLA_Q_PROJ] =
			{
				.name			  = "mla_q_proj",
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = 1u << RECIPE_SLOT_FFN_GATE,
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
			},
		[OP_MLA_KV_PROJ] =
			{
				.name			  = "mla_kv_proj",
				.state_writes	  = CG_STATE_MLA,
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = 1u << RECIPE_SLOT_FFN_UP,
				.reads			  = {OP_TOUCH_IN0},
			},
		[OP_MLA_QKV_PROJ_FUSED] =
			{
				.name			  = "mla_qkv_proj_fused",
				.state_writes	  = CG_STATE_MLA,
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_FFN_GATE) | (1u << RECIPE_SLOT_FFN_UP),
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
			},
		[OP_MOE_ROUTER] =
			{
				.name			  = "moe_router",
				.fn_name		  = "matmul",
				.slot0			  = OP_FN_SLOT(matmul),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_XB) | (1u << RECIPE_SLOT_XB2) |
									(1u << RECIPE_SLOT_FFN_ACT) | (1u << RECIPE_SLOT_RESID_TMP),
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT, OP_TOUCH_OUT1},
			},
		[OP_ROPE] =
			{
				.name	 = "rope",
				.fn_name = "rope",
				.slot0	 = OP_FN_SLOT(rope),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_ROPE_QK_FUSED] =
			{
				.name	 = "rope_qk_fused",
				.fn_name = "rope_qk",
				.slot0	 = OP_FN_SLOT(rope_qk),
				.slot1	 = OP_FN_SLOT(rope),
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT, OP_TOUCH_OUT1},
			},
		[OP_ROPE_EXT] =
			{
				.name	 = "rope_ext",
				.fn_name = "rope_ext",
				.slot0	 = OP_FN_SLOT(rope_ext),
				.slot1	 = OP_FN_SLOT(rope),
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_QK_NORM_ROPE] =
			{
				.name	 = "qk_norm_rope",
				.fn_name = "qk_norm_rope",
				.slot0	 = OP_FN_SLOT(qk_norm_rope),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT, OP_TOUCH_OUT1},
			},
		[OP_KV_PUT] =
			{
				.name		  = "kv_put",
				.state_writes = CG_STATE_KV,
				.fn_name	  = "kv_put",
				.slot0		  = OP_FN_SLOT(kv_put),
				.slot1		  = OP_NO_SLOT,
				.reads		  = {OP_TOUCH_IN0, OP_TOUCH_IN1},
			},
		[OP_ATTENTION] =
			{
				.name		 = "attention",
				.state_reads = CG_STATE_KV,
				.fn_name	 = "attention",
				.slot0		 = OP_FN_SLOT(attention),
				.slot1		 = OP_NO_SLOT,
				.reads		 = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2},
				.writes		 = {OP_TOUCH_OUT},
			},
		[OP_ATTENTION_MLA] =
			{
				.name		 = "attention_mla",
				.state_reads = CG_STATE_MLA,
				.fn_name	 = "attention",
				.slot0		 = OP_FN_SLOT(attention),
				.slot1		 = OP_NO_SLOT,
				.reads		 = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2},
				.writes		 = {OP_TOUCH_OUT},
			},
		[OP_MLA_ABSORBED] =
			{
				.name		 = "mla_absorbed",
				.state_reads = CG_STATE_MLA,
				.fn_name	 = "mla_absorbed",
				.slot0		 = OP_FN_SLOT(mla_absorbed),
				.slot1		 = OP_NO_SLOT,
				.reads		 = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2},
				.writes		 = {OP_TOUCH_OUT},
			},
		[OP_ADD] =
			{
				.name	 = "add",
				.fn_name = "add_inplace",
				.slot0	 = OP_FN_SLOT(add_inplace),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_IN0},
			},
		[OP_SWAP] =
			{
				.name	= "swap",
				.reads	= {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes = {OP_TOUCH_IN0, OP_TOUCH_IN1},
			},
		[OP_FFN_ACTIVATE] =
			{
				.name	 = "ffn_activate",
				.fn_name = "ffn_activate",
				.slot0	 = OP_FN_SLOT(ffn_activate),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_FFN_ACTIVATE_EX] =
			{
				.name	 = "ffn_activate_ex",
				.fn_name = "ffn_activate_ex",
				.slot0	 = OP_FN_SLOT(ffn_activate_ex),
				.slot1	 = OP_FN_SLOT(ffn_activate),
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_FFN_ACTIVATE_FUSED] =
			{
				.name	 = "ffn_activate_fused",
				.fn_name = "ffn_activate",
				.slot0	 = OP_FN_SLOT(ffn_activate),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_SOFTCAP] =
			{
				.name	 = "softcap",
				.fn_name = "softcap",
				.slot0	 = OP_FN_SLOT(softcap),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_SPLIT_QGATE] =
			{
				.name	 = "split_qgate",
				.fn_name = "split_qgate",
				.slot0	 = OP_FN_SLOT(split_qgate),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT, OP_TOUCH_OUT1},
			},
		[OP_PARTIAL_ROPE_QK] =
			{
				.name	 = "partial_rope_qk",
				.fn_name = "partial_rope_qk",
				.slot0	 = OP_FN_SLOT(partial_rope_qk),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT, OP_TOUCH_OUT1},
			},
		[OP_ATTN_OUTPUT_GATE] =
			{
				.name	 = "attn_output_gate",
				.fn_name = "attn_output_gate",
				.slot0	 = OP_FN_SLOT(attn_output_gate),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_GATED_DELTA_NET] =
			{
				.name		  = "gated_delta_net",
				.state_reads  = CG_STATE_RECURRENT,
				.state_writes = CG_STATE_RECURRENT,
				.fn_name	  = "gated_delta_net",
				.slot0		  = OP_FN_SLOT(gated_delta_net),
				.slot1		  = OP_NO_SLOT,
				.reads		  = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2, OP_TOUCH_IN3},
				.writes		  = {OP_TOUCH_OUT},
			},
		[OP_SHORTCONV] =
			{
				.name		  = "shortconv",
				.state_reads  = CG_STATE_RECURRENT,
				.state_writes = CG_STATE_RECURRENT,
				.reads		  = {OP_TOUCH_IN0},
				.writes		  = {OP_TOUCH_OUT},
			},
		[OP_HC_INIT] =
			{
				.name	 = "hc_init",
				.fn_name = "hc_init",
				.slot0	 = OP_FN_SLOT(hc_init),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_HC_PRE] =
			{
				.name	 = "hc_pre",
				.fn_name = "hc_pre",
				.slot0	 = OP_FN_SLOT(hc_pre),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT, OP_TOUCH_OUT1},
			},
		[OP_HC_POST] =
			{
				.name	 = "hc_post",
				.fn_name = "hc_post",
				.slot0	 = OP_FN_SLOT(hc_post),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_HC_FINAL] =
			{
				.name	 = "hc_final",
				.fn_name = "hc_final",
				.slot0	 = OP_FN_SLOT(hc_final),
				.slot1	 = OP_NO_SLOT,
				.reads	 = {OP_TOUCH_IN0},
				.writes	 = {OP_TOUCH_OUT},
			},
		[OP_KDA] =
			{
				.name		  = "kda",
				.state_reads  = CG_STATE_RECURRENT,
				.state_writes = CG_STATE_RECURRENT,
				.fn_name	  = "kda",
				.slot0		  = OP_FN_SLOT(kda),
				.slot1		  = OP_NO_SLOT,
				.reads		  = {OP_TOUCH_IN0},
				.writes		  = {OP_TOUCH_OUT},
			},
		[OP_LOGITS_READBACK] =
			{
				.name		  = "logits_readback",
				.state_writes = CG_STATE_LOGITS,
				.reads		  = {OP_TOUCH_IN0, OP_TOUCH_IN1},
			},
		[OP_PLE_PROJ_INJECT] =
			{
				.name		 = "ple_proj_inject",
				.state_reads = CG_STATE_PLE,
				.reads		 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
				.writes		 = {OP_TOUCH_IN0, OP_TOUCH_IN1},
			},
		[OP_MOE_SHARED] =
			{
				.name			  = "moe_shared",
				.fn_name		  = "moe_activate",
				.slot0			  = OP_FN_SLOT(moe_activate),
				.slot1			  = OP_NO_SLOT,
				.batch_extra_mask = (1u << RECIPE_SLOT_XB) | (1u << RECIPE_SLOT_XB2) |
									(1u << RECIPE_SLOT_FFN_ACT) | (1u << RECIPE_SLOT_RESID_TMP) |
									(1u << RECIPE_SLOT_FFN_GATE) | (1u << RECIPE_SLOT_FFN_UP) |
									(1u << RECIPE_SLOT_FFN_GATE_UP),
				.reads			  = {OP_TOUCH_IN0},
				.writes			  = {OP_TOUCH_OUT},
			},
		[OP_MOE_EXPERTS] =
			{
				.name			  = "moe_experts",
				.fn_name		  = "moe_experts",
				.slot0			  = OP_FN_SLOT(moe_experts_batch),
				.slot1			  = OP_FN_SLOT(matmul_thread_local),
				.batch_extra_mask = (1u << RECIPE_SLOT_XB) | (1u << RECIPE_SLOT_XB2) |
									(1u << RECIPE_SLOT_FFN_ACT) | (1u << RECIPE_SLOT_RESID_TMP),
				.reads			  = {OP_TOUCH_IN0, OP_TOUCH_IN1, OP_TOUCH_IN2},
				.writes			  = {OP_TOUCH_OUT},
			},
};

const char *recipe_op_name(op_kind k) {
	if ((int)k < 0 || k >= OP_KIND_COUNT || !OP_REQUIREMENTS[k].name)
		return "?";
	return OP_REQUIREMENTS[k].name;
}

uint8_t recipe_op_state_reads(op_kind k) {
	return ((int)k < 0 || k >= OP_KIND_COUNT) ? 0 : OP_REQUIREMENTS[k].state_reads;
}

uint8_t recipe_op_state_writes(op_kind k) {
	return ((int)k < 0 || k >= OP_KIND_COUNT) ? 0 : OP_REQUIREMENTS[k].state_writes;
}

const char *recipe_op_missing_fn(const backend *a, op_kind k) {
	if ((int)k < 0 || k >= OP_KIND_COUNT)
		return NULL;
	const op_requirement *req = &OP_REQUIREMENTS[k];
	if (!req->fn_name)
		return NULL;
	const char *base = (const char *)a;
	if (*(const void *const *)(base + req->slot0))
		return NULL;
	if (req->slot1 != OP_NO_SLOT && *(const void *const *)(base + req->slot1))
		return NULL;
	return req->fn_name;
}

static void touch_push(uint8_t *list, int *n, uint8_t slot) {
	if (slot == RECIPE_SLOT_NONE || *n >= OP_TOUCH_MAX)
		return;
	for (int i = 0; i < *n; i++)
		if (list[i] == slot)
			return;
	list[(*n)++] = slot;
}

static uint8_t touch_resolve(int8_t t, const recipe_op *op) {
	if (t == OP_TOUCH_IN3)
		return op->in[3];
	if (t <= OP_TOUCH_IN2)
		return op->in[t - 1];
	if (t == OP_TOUCH_OUT)
		return op->out;
	if (t == OP_TOUCH_OUT1)
		return op->out1;
	if (t == OP_TOUCH_OUT2)
		return op->out2;
	return RECIPE_SLOT_NONE;
}

int recipe_op_touch(const recipe_op *op, uint8_t *reads, int *n_reads, uint8_t *writes,
					int *n_writes) {
	if (op->kind < 0 || op->kind >= OP_KIND_COUNT)
		return 0;
	const op_requirement *req = &OP_REQUIREMENTS[op->kind];
	for (int i = 0; i < 4 && req->reads[i]; i++)
		touch_push(reads, n_reads, touch_resolve(req->reads[i], op));
	for (int i = 0; i < 4 && req->writes[i]; i++)
		touch_push(writes, n_writes, touch_resolve(req->writes[i], op));
	return 1;
}

static int matmul_op_uses_substage(op_kind k) {
	return k == OP_MATMUL || k == OP_MATMUL_MULTI || k == OP_MATMUL_RESIDUAL ||
		   k == OP_MATMUL_FFN_DOWN;
}

void recipe_refine_matmul_stages(recipe_op *ops, int n_ops) {
	for (int i = 0; i < n_ops; i++) {
		recipe_op *op = &ops[i];
		if (!matmul_op_uses_substage(op->kind) || op->stage != STAGE_MATMUL)
			continue;
		op->stage = matmul_substage(op->w_idx);
	}
}

stage recipe_matmul_substage(uint8_t w_idx) {
	return matmul_substage(w_idx);
}

int recipe_op_has_matmul_substage(op_kind k) {
	return matmul_op_uses_substage(k);
}

static stage matmul_substage(uint8_t w_idx) {
	switch ((weight_idx)w_idx) {
	case WIDX_WQ:
	case WIDX_WK:
	case WIDX_WV:
	case WIDX_MLA_Q_A:
	case WIDX_MLA_Q_B:
	case WIDX_MLA_KV_A:
	case WIDX_MLA_K_B:
	case WIDX_MLA_V_B:
		return STAGE_MATMUL_QKV;
	case WIDX_WO:
		return STAGE_MATMUL_ATTN_OUT;
	case WIDX_GATE:
	case WIDX_UP:
	case WIDX_GATE_UP:
	case WIDX_SHEXP_GATE:
	case WIDX_SHEXP_UP:
		return STAGE_MATMUL_FFN_GATE_UP;
	case WIDX_DOWN:
	case WIDX_SHEXP_DOWN:
		return STAGE_MATMUL_FFN_DOWN;
	case WIDX_OUTPUT_W:
		return STAGE_LOGITS_MATMUL;
	default:
		return STAGE_MATMUL;
	}
}

uint32_t recipe_op_batch_extra_mask(op_kind k) {
	return ((int)k < 0 || k >= OP_KIND_COUNT) ? 0 : OP_REQUIREMENTS[k].batch_extra_mask;
}
