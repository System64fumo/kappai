#include "recipe/internal.h"

static recipe_op mk_op_base(op_kind kind, uint8_t in0, uint8_t in1, uint8_t in2, uint8_t out,
							uint8_t widx, stage stg) {
	recipe_op op = {
		.kind  = kind,
		.in	   = {in0, in1, in2, RECIPE_SLOT_NONE},
		.out   = out,
		.out1  = RECIPE_SLOT_NONE,
		.out2  = RECIPE_SLOT_NONE,
		.w_idx = widx,
		.stage = stg,
	};
	return op;
}

recipe_op mk_backend_op(op_kind kind, uint8_t in, uint8_t out, uint8_t widx, stage stage) {
	return mk_op_base(kind, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, widx, stage);
}

recipe_op mk_rmsnorm(uint8_t in, uint8_t out, uint8_t widx, float eps, stage stage) {
	recipe_op op = mk_op_base(OP_RMSNORM, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, widx, stage);
	op.u.rmsnorm.eps = eps;
	return op;
}

recipe_op mk_rmsnorm_add(uint8_t in, uint8_t residual, uint8_t out, uint8_t widx, float eps,
						 stage stage) {
	recipe_op op	 = mk_op_base(OP_RMSNORM_ADD, in, residual, RECIPE_SLOT_NONE, out, widx, stage);
	op.u.rmsnorm.eps = eps;
	op.u.rmsnorm.out_scale = 1.0f;
	return op;
}

recipe_op mk_matmul(uint8_t in, uint8_t out, uint8_t widx, int n, int k, stage stage) {
	recipe_op op  = mk_op_base(OP_MATMUL, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, widx, stage);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

recipe_op mk_copy_k_to_v(uint8_t k_slot, uint8_t v_slot) {
	return mk_op_base(OP_COPY_K_TO_V, k_slot, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, v_slot,
					  RECIPE_NO_WEIGHT, STAGE_MATMUL);
}

void recipe_build_layers(model_recipe *r, const model *m, int cap, const char *arch_name,
						 void (*append_layer)(op_emitter *, const model *, int)) {
	r->layer.ops	 = xcalloc((size_t)cap, sizeof(recipe_op));
	r->layer.n_ops	 = cap;
	r->per_layer_ops = xcalloc((size_t)m->n_layers * cap, sizeof(recipe_op));
	int max_ops		 = 0;
	for (int li = 0; li < m->n_layers; li++) {
		op_emitter e = op_emitter_make(r->per_layer_ops + (size_t)li * cap, cap, arch_name);
		append_layer(&e, m, li);
		if (e.count > max_ops)
			max_ops = e.count;
	}
	if (max_ops <= 0 || max_ops >= cap)
		return;
	for (int li = 1; li < m->n_layers; li++) {
		recipe_op *src = r->per_layer_ops + (size_t)li * cap;
		recipe_op *dst = r->per_layer_ops + (size_t)li * (size_t)max_ops;
		memmove(dst, src, (size_t)max_ops * sizeof(recipe_op));
	}
	r->per_layer_ops =
		xrealloc(r->per_layer_ops, (size_t)m->n_layers * (size_t)max_ops * sizeof(recipe_op));
	r->layer.n_ops = max_ops;
}

recipe_op mk_matmul_multi2(uint8_t in, uint8_t out0, uint8_t out1, uint8_t widx, int n0, int n1,
						   int k) {
	recipe_op op = mk_op_base(OP_MATMUL_MULTI, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out0, widx,
							  STAGE_MATMUL);
	op.out1		 = out1;
	op.u.matmul_multi.n		   = 2;
	op.u.matmul_multi.k		   = k;
	op.u.matmul_multi.n_out[0] = n0;
	op.u.matmul_multi.n_out[1] = n1;
	return op;
}

recipe_op mk_matmul_multi3(uint8_t in, uint8_t out0, uint8_t out1, uint8_t out2, uint8_t widx,
						   int n0, int n1, int n2, int k) {
	recipe_op op = mk_op_base(OP_MATMUL_MULTI, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out0, widx,
							  STAGE_MATMUL);
	op.out1		 = out1;
	op.out2		 = out2;
	op.u.matmul_multi.n		   = 3;
	op.u.matmul_multi.k		   = k;
	op.u.matmul_multi.n_out[0] = n0;
	op.u.matmul_multi.n_out[1] = n1;
	op.u.matmul_multi.n_out[2] = n2;
	return op;
}

recipe_op mk_matmul_residual(uint8_t in, uint8_t residual, uint8_t out, uint8_t widx, int n,
							 int k) {
	recipe_op op =
		mk_op_base(OP_MATMUL_RESIDUAL, in, residual, RECIPE_SLOT_NONE, out, widx, STAGE_MATMUL);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

recipe_op mk_matmul_ffn_down(uint8_t gate_in, uint8_t up_in, uint8_t out, uint8_t widx, int n,
							 int k, int activation) {
	recipe_op op =
		mk_op_base(OP_MATMUL_FFN_DOWN, gate_in, up_in, RECIPE_SLOT_NONE, out, widx, STAGE_MATMUL);
	op.u.matmul_ffn_down.n			= n;
	op.u.matmul_ffn_down.k			= k;
	op.u.matmul_ffn_down.activation = activation;
	return op;
}

recipe_op mk_ffn_activate(uint8_t gate_in, uint8_t up_in, uint8_t out, int n, int activation) {
	recipe_op op			= mk_op_base(OP_FFN_ACTIVATE, gate_in, up_in, RECIPE_SLOT_NONE, out,
										 RECIPE_NO_WEIGHT, STAGE_FFN_ACT);
	op.u.ffn_act.n			= n;
	op.u.ffn_act.activation = activation;
	return op;
}

recipe_op mk_ffn_activate_fused(uint8_t in, uint8_t out, int n, int activation) {
	recipe_op op   = mk_op_base(OP_FFN_ACTIVATE_FUSED, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out,
								RECIPE_NO_WEIGHT, STAGE_FFN_ACT);
	op.u.ffn_act.n = n;
	op.u.ffn_act.activation = activation;
	return op;
}

recipe_op mk_rmsnorm_per_head(uint8_t io, uint8_t widx, float eps, int n_heads) {
	recipe_op op = mk_op_base(OP_RMSNORM_PER_HEAD, io, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, io, widx,
							  STAGE_RMSNORM);
	op.u.rmsnorm.eps	 = eps;
	op.u.rmsnorm.n_heads = n_heads;
	return op;
}

recipe_op mk_rmsnorm_noweight(uint8_t in) {
	return mk_op_base(OP_RMSNORM_NOWEIGHT, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, in,
					  RECIPE_NO_WEIGHT, STAGE_RMSNORM);
}

recipe_op mk_split_qgate(uint8_t proj, uint8_t q, uint8_t gate) {
	recipe_op op = mk_op_base(OP_SPLIT_QGATE, proj, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, q,
							  RECIPE_NO_WEIGHT, STAGE_MATMUL);
	op.out1		 = gate;
	return op;
}

recipe_op mk_attn_output_gate(uint8_t io, uint8_t gate) {
	return mk_op_base(OP_ATTN_OUTPUT_GATE, io, gate, RECIPE_SLOT_NONE, io, RECIPE_NO_WEIGHT,
					  STAGE_ATTN);
}

recipe_op mk_gated_delta_net(uint8_t proj_in, uint8_t gate_in, uint8_t alpha_in, uint8_t beta_in,
							 uint8_t out) {
	recipe_op op = mk_op_base(OP_GATED_DELTA_NET, proj_in, gate_in, alpha_in, out, RECIPE_NO_WEIGHT,
							  STAGE_ATTN);
	op.in[3]	 = beta_in;
	return op;
}

recipe_op mk_hc_pre(uint8_t stream, uint8_t x, uint8_t coef, uint8_t widx) {
	recipe_op op =
		mk_op_base(OP_HC_PRE, stream, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, x, widx, STAGE_ATTN);
	op.out1 = coef;
	return op;
}

recipe_op mk_hc_post(uint8_t sub, uint8_t coef, uint8_t hc) {
	return mk_op_base(OP_HC_POST, sub, coef, hc, hc, WIDX_NONE, STAGE_ADD);
}

recipe_op mk_shortconv(uint8_t in, uint8_t out) {
	return mk_op_base(OP_SHORTCONV, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out, RECIPE_NO_WEIGHT,
					  STAGE_ATTN);
}

recipe_op mk_mla_qkv_proj_fused(uint8_t in, uint8_t out, int n, int k) {
	recipe_op op  = mk_op_base(OP_MLA_QKV_PROJ_FUSED, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out,
							   WIDX_NONE, STAGE_MATMUL);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

recipe_op mk_attention_mla(uint8_t q_in, uint8_t out, int n_heads, int head_dim, int n_ctx,
						   float scale) {
	recipe_op op = mk_op_base(OP_ATTENTION_MLA, q_in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out,
							  WIDX_NONE, STAGE_ATTN);
	op.u.attention.n_heads			 = n_heads;
	op.u.attention.n_kv_heads		 = n_heads;
	op.u.attention.head_dim			 = head_dim;
	op.u.attention.n_ctx			 = n_ctx;
	op.u.attention.scale			 = scale;
	op.u.attention.sliding_window	 = 0;
	op.u.attention.n_kv_heads_active = n_heads;
	return op;
}

recipe_op mk_scale(uint8_t io, uint8_t widx, float scale) {
	recipe_op op =
		mk_op_base(OP_SCALE, io, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, io, widx, STAGE_ADD);
	op.u.scale.scale = scale;
	return op;
}

recipe_op mk_ple_proj_inject(uint8_t in, uint8_t residual, uint8_t widx) {
	return mk_op_base(OP_PLE_PROJ_INJECT, in, residual, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, widx,
					  STAGE_PLE);
}

recipe_op mk_add(uint8_t in0, uint8_t in1, stage stage) {
	return mk_op_base(OP_ADD, in0, in1, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT,
					  stage);
}

recipe_op mk_swap(uint8_t in0, uint8_t in1, stage stage) {
	return mk_op_base(OP_SWAP, in0, in1, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT,
					  stage);
}

recipe_op mk_kvput(uint8_t k_in, uint8_t v_in) {
	return mk_op_base(OP_KV_PUT, k_in, v_in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT,
					  STAGE_KVPUT);
}

recipe_op mk_attention(uint8_t q_in, uint8_t out, int n_heads, int n_kv_heads, int head_dim,
					   int n_ctx, float scale, int sliding_window) {
	recipe_op op		   = mk_op_base(OP_ATTENTION, q_in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, out,
										RECIPE_NO_WEIGHT, STAGE_ATTN);
	op.u.attention.n_heads = n_heads;
	op.u.attention.n_kv_heads		 = n_kv_heads;
	op.u.attention.head_dim			 = head_dim;
	op.u.attention.n_ctx			 = n_ctx;
	op.u.attention.scale			 = scale;
	op.u.attention.sliding_window	 = sliding_window;
	op.u.attention.n_kv_heads_active = n_kv_heads;
	return op;
}

recipe_op mk_attention_default_scale(uint8_t q_in, uint8_t out, int n_heads, int n_kv_heads,
									 int head_dim, int n_ctx, int sliding_window) {
	return mk_attention(q_in, out, n_heads, n_kv_heads, head_dim, n_ctx,
						1.0f / sqrtf((float)head_dim), sliding_window);
}

recipe_op mk_rope(uint8_t in, int n_heads, int head_dim, int rope_neox) {
	recipe_op op = mk_op_base(OP_ROPE, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, in, RECIPE_NO_WEIGHT,
							  STAGE_ROPE);
	op.u.rope.n_heads	= n_heads;
	op.u.rope.head_dim	= head_dim;
	op.u.rope.rope_neox = rope_neox;
	return op;
}

recipe_op mk_rope_qk_fused(uint8_t q, uint8_t k, int n_heads, int n_kv_heads, int head_dim,
						   int rope_neox) {
	recipe_op op =
		mk_op_base(OP_ROPE_QK_FUSED, q, k, RECIPE_SLOT_NONE, q, RECIPE_NO_WEIGHT, STAGE_ROPE);
	op.out1				 = k;
	op.u.rope.n_heads	 = n_heads;
	op.u.rope.n_kv_heads = n_kv_heads;
	op.u.rope.head_dim	 = head_dim;
	op.u.rope.rope_neox	 = rope_neox;
	return op;
}

recipe_op mk_rope_ext(uint8_t in, int is_q, int rope_neox) {
	recipe_op op		   = mk_op_base(OP_ROPE_EXT, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, in,
										RECIPE_NO_WEIGHT, STAGE_ROPE);
	op.u.rope_ext.n_heads  = 0;
	op.u.rope_ext.head_dim = 0;
	op.u.rope_ext.use_freq_factors = 1;
	op.u.rope_ext.rope_neox		   = rope_neox;
	op.u.rope_ext.is_q			   = is_q;
	return op;
}

recipe_op mk_qk_norm_rope(uint8_t q, uint8_t k, float eps, int rope_neox) {
	recipe_op op =
		mk_op_base(OP_QK_NORM_ROPE, q, k, RECIPE_SLOT_NONE, q, WIDX_ATTN_Q_NORM, STAGE_ROPE);
	op.out1						= k;
	op.u.qk_norm_rope.q_only	= k == RECIPE_SLOT_NONE;
	op.u.qk_norm_rope.eps		= eps;
	op.u.qk_norm_rope.rope_neox = rope_neox;
	return op;
}

recipe_op mk_partial_rope_qk(uint8_t q, uint8_t k) {
	recipe_op op =
		mk_op_base(OP_PARTIAL_ROPE_QK, q, k, RECIPE_SLOT_NONE, q, RECIPE_NO_WEIGHT, STAGE_ROPE);
	op.out1 = k;
	return op;
}

static recipe_op mk_embd_lookup(void) {
	return mk_op_base(OP_EMBD_LOOKUP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_SLOT_X, WIDX_TOK_EMBD, STAGE_EMBD);
}

static recipe_op mk_scale_embeddings(void) {
	return mk_op_base(OP_SCALE_EMBEDDINGS, RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_SLOT_X, RECIPE_NO_WEIGHT, STAGE_EMBD);
}

static recipe_op mk_ple_build(uint8_t x) {
	return mk_op_base(OP_PLE_BUILD, x, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_NO_WEIGHT, STAGE_EMBD);
}

static recipe_op mk_softcap(uint8_t in, float cap) {
	recipe_op op = mk_op_base(OP_SOFTCAP, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
							  RECIPE_NO_WEIGHT, STAGE_LOGITS_READBACK);
	op.u.softcap.cap = cap;
	return op;
}

static recipe_op mk_logits_readback(void) {
	return mk_op_base(OP_LOGITS_READBACK, RECIPE_SLOT_LOGITS, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
					  RECIPE_SLOT_NONE, RECIPE_NO_WEIGHT, STAGE_LOGITS_READBACK);
}

static recipe_op mk_moe_router(uint8_t in, uint8_t ids, uint8_t weights, int n_experts, int k) {
	recipe_op op  = mk_op_base(OP_MOE_ROUTER, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, ids,
							   WIDX_FFN_GATE_INP, STAGE_MATMUL);
	op.out1		  = weights;
	op.u.matmul.n = n_experts;
	op.u.matmul.k = k;
	return op;
}

static recipe_op mk_moe_experts(uint8_t in, uint8_t ids, uint8_t weights, uint8_t out, int n,
								int k) {
	recipe_op op  = mk_op_base(OP_MOE_EXPERTS, in, ids, weights, out, WIDX_NONE, STAGE_MATMUL);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

static recipe_op mk_moe_shared(uint8_t in, int n, int k) {
	recipe_op op  = mk_op_base(OP_MOE_SHARED, in, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE,
							   RECIPE_SLOT_FFN_ACT, WIDX_NONE, STAGE_MATMUL);
	op.u.matmul.n = n;
	op.u.matmul.k = k;
	return op;
}

void recipe_emit_ffn_activation(op_emitter *e, int dim, int inter, int gate_up_fused,
								int has_matmul_multi, int act) {
	if (gate_up_fused) {
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP, 2 * inter, dim,
							 STAGE_MATMUL));
		OP_EMIT(e, mk_ffn_activate_fused(RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_FFN_ACT, inter, act));
		return;
	}
	if (has_matmul_multi) {
		OP_EMIT(e, mk_matmul_multi2(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP,
									WIDX_GATE, inter, inter, dim));
	} else {
		OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter, dim,
							 STAGE_MATMUL));
		OP_EMIT(e,
				mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim, STAGE_MATMUL));
	}
	OP_EMIT(e, mk_ffn_activate(RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_FFN_ACT, inter,
							   act));
}

void recipe_emit_dense_ffn(op_emitter *e, const model *m, int li, uint8_t norm_widx) {
	int act =
		m->arch_info && m->arch_info->uses_gelu_activation ? ACTIVATION_GELU : ACTIVATION_SILU;
	OP_EMIT(e, mk_add(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD));
	OP_EMIT(e, mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, norm_widx, m->norm_eps, STAGE_RMSNORM));
	recipe_emit_ffn_activation(e, m->dim, m->intermediate, li >= 0 && m->layers[li].gate_up_fused,
							   backend_has_cap(m->backend, BCAP_MULTI_MATMUL), act);
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, m->dim, m->intermediate,
						 STAGE_MATMUL));
	OP_EMIT(e, mk_add(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD));
}

void recipe_emit_qk_norm_attention(op_emitter *e, const model *m, int li) {
	int dim	   = m->dim;
	int q_out  = m->n_heads * m->head_dim;
	int kv_out = m->n_kv_heads * m->head_dim;
	int neox   = m->arch_info->uses_neox_rope;

	OP_EMIT(e,
			mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, q_out, dim, STAGE_MATMUL));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim, STAGE_MATMUL));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim, STAGE_MATMUL));

	OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_Q, WIDX_ATTN_Q_NORM, m->norm_eps, m->n_heads));
	OP_EMIT(e, mk_rmsnorm_per_head(RECIPE_SLOT_K, WIDX_ATTN_K_NORM, m->norm_eps, m->n_kv_heads));
	OP_EMIT(e, mk_rope(RECIPE_SLOT_Q, m->n_heads, m->head_dim, neox));
	OP_EMIT(e, mk_rope(RECIPE_SLOT_K, m->n_kv_heads, m->head_dim, neox));
	OP_EMIT(e, mk_kvput(RECIPE_SLOT_K, RECIPE_SLOT_V));
	OP_EMIT(e, mk_attention_default_scale(RECIPE_SLOT_Q, RECIPE_SLOT_XB2, m->n_heads, m->n_kv_heads,
										  m->head_dim, m->n_ctx, m->sliding_window));
	OP_EMIT(e, mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out, STAGE_MATMUL));
	recipe_emit_dense_ffn(e, m, li, WIDX_FFN_NORM);
}

void recipe_emit_moe_ffn(op_emitter *e, const struct model *m, uint8_t router_in_slot,
						 uint8_t experts_in_slot, uint8_t out_slot) {
	int dim		  = m->dim;
	int moe_inter = m->moe.moe_intermediate;

	OP_EMIT(e, mk_moe_router(router_in_slot, RECIPE_SLOT_ROUTER_IDS, RECIPE_SLOT_ROUTER_W,
							 m->moe.n_experts, dim));

	if (m->moe.n_shared_experts > 0) {
		int sh_inter = moe_inter * m->moe.n_shared_experts;
		OP_EMIT(e, mk_moe_shared(experts_in_slot, sh_inter, dim));
	}

	OP_EMIT(e, mk_moe_experts(experts_in_slot, RECIPE_SLOT_ROUTER_IDS, RECIPE_SLOT_ROUTER_W,
							  out_slot, dim, moe_inter));
}

void recipe_build_pre_ops(model_recipe *r, const model *m) {
	int n = 1;
	if (m->arch_info->has_scale_embeddings)
		n++;
	if (m->has_per_layer_embeddings)
		n++;
	recipe_op *ops = xcalloc((size_t)n, sizeof(recipe_op));
	int		   i   = 0;

	ops[i++] = mk_embd_lookup();
	if (m->arch_info->has_scale_embeddings) {
		ops[i++] = mk_scale_embeddings();
	}
	if (m->has_per_layer_embeddings) {
		ops[i++] = mk_ple_build(RECIPE_SLOT_X);
	}
	r->pre_ops	 = ops;
	r->n_pre_ops = i;
}

void recipe_build_post_ops(model_recipe *r, const model *m) {
	const int	dim = m->dim;
	const float eps = m->norm_eps;
	int			cap = 5;
	recipe_op  *ops = xcalloc(cap, sizeof(recipe_op));
	int			i	= 0;

	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_OUTPUT_NORM, eps, STAGE_LOGITS_NORM);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_LOGITS, WIDX_OUTPUT_W, m->vocab_size, dim,
						 STAGE_LOGITS_MATMUL);

	if (m->final_logit_softcap > 0.0f) {
		ops[i++] = mk_softcap(RECIPE_SLOT_LOGITS, m->final_logit_softcap);
	}

	ops[i++] = mk_logits_readback();

	r->post_ops	  = ops;
	r->n_post_ops = i;
}
