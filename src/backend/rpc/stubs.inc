static status_code stub_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos,
							   const buffer *k_in, const buffer *v_in, int n_kv_heads, int head_dim,
							   int n_ctx, int n_kv_heads_active) {
	return SEND_VALS(self, RPC_OP_kv_put, RPC_V(k), RPC_V(v), RPC_V(layer), RPC_V(pos), RPC_V(k_in),
					 RPC_V(v_in), RPC_V(n_kv_heads), RPC_V(head_dim), RPC_V(n_ctx),
					 RPC_V(n_kv_heads_active));
}

static status_code stub_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
									 const buffer *k_in, const buffer *v_in, int in_row_stride,
									 int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
									 int m) {
	return SEND_VALS(self, RPC_OP_kv_put_batch, RPC_V(k), RPC_V(v), RPC_V(layer), RPC_V(pos_start),
					 RPC_V(k_in), RPC_V(v_in), RPC_V(in_row_stride), RPC_V(n_kv_heads),
					 RPC_V(head_dim), RPC_V(n_ctx), RPC_V(n_kv_heads_active), RPC_V(m));
}

static status_code stub_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
									int token, int dim, buffer *x_out) {
	return SEND_VALS(self, RPC_OP_embd_lookup, RPC_V(tok_embd), RPC_V(tok_embd_type), RPC_V(token),
					 RPC_V(dim), RPC_V(x_out));
}

static status_code stub_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
								float eps) {
	return SEND_VALS(self, RPC_OP_rmsnorm, RPC_V(x), RPC_V(w), RPC_V(y), RPC_V(n), RPC_V(eps));
}

static status_code stub_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							   buffer *y, int n, int k) {
	return SEND_VALS(self, RPC_OP_matmul, RPC_V(w), RPC_V(w_type), RPC_V(x), RPC_V(y), RPC_V(n),
					 RPC_V(k));
}

static status_code stub_matmul_residual(backend *self, const buffer *w, uint32_t w_type,
										const buffer *x, const buffer *residual, buffer *y, int n,
										int k) {
	return SEND_VALS(self, RPC_OP_matmul_residual, RPC_V(w), RPC_V(w_type), RPC_V(x),
					 RPC_V(residual), RPC_V(y), RPC_V(n), RPC_V(k));
}

static status_code stub_matmul_qonly(backend *self, const buffer *w, uint32_t w_type,
									 const buffer *x_q8, uint32_t q8_type, buffer *y, int n, int k,
									 int m) {
	return SEND_VALS(self, RPC_OP_matmul_qonly, RPC_V(w), RPC_V(w_type), RPC_V(x_q8),
					 RPC_V(q8_type), RPC_V(y), RPC_V(n), RPC_V(k), RPC_V(m));
}

static status_code stub_prequantize_x(backend *self, const buffer *x, int k, uint32_t q8_type,
									  buffer *x_q8_out) {
	return SEND_VALS(self, RPC_OP_prequantize_x, RPC_V(x), RPC_V(k), RPC_V(q8_type),
					 RPC_V(x_q8_out));
}

static status_code stub_add_inplace(backend *self, buffer *x, const buffer *y, int n) {
	return SEND_VALS(self, RPC_OP_add_inplace, RPC_V(x), RPC_V(y), RPC_V(n));
}

static status_code stub_scale_inplace(backend *self, buffer *x, float scale, int n) {
	return SEND_VALS(self, RPC_OP_scale_inplace, RPC_V(x), RPC_V(scale), RPC_V(n));
}

static status_code stub_softcap(backend *self, buffer *x, float cap, int n) {
	return SEND_VALS(self, RPC_OP_softcap, RPC_V(x), RPC_V(cap), RPC_V(n));
}

static status_code stub_split_qgate(backend *self, const buffer *mixed, buffer *q, buffer *gate,
									int n_heads, int head_dim, int n_rows) {
	return SEND_VALS(self, RPC_OP_split_qgate, RPC_V(mixed), RPC_V(q), RPC_V(gate), RPC_V(n_heads),
					 RPC_V(head_dim), RPC_V(n_rows));
}

static status_code stub_attn_output_gate(backend *self, buffer *out, const buffer *gate, int n,
										 int n_rows) {
	return SEND_VALS(self, RPC_OP_attn_output_gate, RPC_V(out), RPC_V(gate), RPC_V(n),
					 RPC_V(n_rows));
}

static status_code stub_copy_buffer(backend *self, const buffer *src, buffer *dst, int n) {
	return SEND_VALS(self, RPC_OP_copy_buffer, RPC_V(src), RPC_V(dst), RPC_V(n));
}

static status_code stub_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
									float combine_scale) {
	return SEND_VALS(self, RPC_OP_ple_combine, RPC_V(ple), RPC_V(proj), RPC_V(n),
					 RPC_V(combine_scale));
}

static status_code stub_ffn_activate(backend *self, const buffer *gate, const buffer *up,
									 buffer *out, int n) {
	return SEND_VALS(self, RPC_OP_ffn_activate, RPC_V(gate), RPC_V(up), RPC_V(out), RPC_V(n));
}

static status_code stub_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up,
										buffer *out, int n, int activation) {
	return SEND_VALS(self, RPC_OP_ffn_activate_ex, RPC_V(gate), RPC_V(up), RPC_V(out), RPC_V(n),
					 RPC_V(activation));
}

static status_code stub_rmsnorm_per_head(backend *self, const buffer *x, const buffer *w, buffer *y,
										 int n_heads, int head_dim, float eps) {
	return SEND_VALS(self, RPC_OP_rmsnorm_per_head, RPC_V(x), RPC_V(w), RPC_V(y), RPC_V(n_heads),
					 RPC_V(head_dim), RPC_V(eps));
}

static status_code stub_rmsnorm_noweight(backend *self, const buffer *x, buffer *y, int n,
										 float eps) {
	return SEND_VALS(self, RPC_OP_rmsnorm_noweight, RPC_V(x), RPC_V(y), RPC_V(n), RPC_V(eps));
}

static status_code stub_rmsnorm_noweight_per_head(backend *self, const buffer *x, buffer *y,
												  int n_heads, int head_dim, float eps) {
	return SEND_VALS(self, RPC_OP_rmsnorm_noweight_per_head, RPC_V(x), RPC_V(y), RPC_V(n_heads),
					 RPC_V(head_dim), RPC_V(eps));
}

static status_code stub_rmsnorm_add(backend *self, const buffer *x, const buffer *w,
									const buffer *residual, buffer *y, int n, float eps,
									float out_scale) {
	return SEND_VALS(self, RPC_OP_rmsnorm_add, RPC_V(x), RPC_V(w), RPC_V(residual), RPC_V(y),
					 RPC_V(n), RPC_V(eps), RPC_V(out_scale));
}

static status_code stub_rmsnorm_add_batch(backend *self, const buffer *x, const buffer *w,
										  const buffer *residual, buffer *y, int n, float eps,
										  float out_scale, int m) {
	return SEND_VALS(self, RPC_OP_rmsnorm_add_batch, RPC_V(x), RPC_V(w), RPC_V(residual), RPC_V(y),
					 RPC_V(n), RPC_V(eps), RPC_V(out_scale), RPC_V(m));
}

static status_code stub_rmsnorm_add_norm_batch(backend *self, const buffer *x, const buffer *w_add,
											   const buffer *residual, buffer *y_add,
											   const buffer *w_norm, buffer *y_norm, int n,
											   float eps, float out_scale, int m) {
	return SEND_VALS(self, RPC_OP_rmsnorm_add_norm_batch, RPC_V(x), RPC_V(w_add), RPC_V(residual),
					 RPC_V(y_add), RPC_V(w_norm), RPC_V(y_norm), RPC_V(n), RPC_V(eps),
					 RPC_V(out_scale), RPC_V(m));
}

static status_code stub_rmsnorm_per_head_batch(backend *self, const buffer *x, const buffer *w,
											   buffer *y, int n_heads, int head_dim, float eps,
											   int m) {
	return SEND_VALS(self, RPC_OP_rmsnorm_per_head_batch, RPC_V(x), RPC_V(w), RPC_V(y),
					 RPC_V(n_heads), RPC_V(head_dim), RPC_V(eps), RPC_V(m));
}

static status_code stub_rmsnorm_noweight_batch(backend *self, const buffer *x, buffer *y, int n,
											   float eps, int m) {
	return SEND_VALS(self, RPC_OP_rmsnorm_noweight_batch, RPC_V(x), RPC_V(y), RPC_V(n), RPC_V(eps),
					 RPC_V(m));
}

static status_code stub_rmsnorm_noweight_per_head_batch(backend *self, const buffer *x, buffer *y,
														int n_heads, int head_dim, float eps,
														int m) {
	return SEND_VALS(self, RPC_OP_rmsnorm_noweight_per_head_batch, RPC_V(x), RPC_V(y),
					 RPC_V(n_heads), RPC_V(head_dim), RPC_V(eps), RPC_V(m));
}

static status_code stub_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type,
										const buffer *gate, const buffer *up, buffer *y, int n,
										int k, int activation) {
	return SEND_VALS(self, RPC_OP_matmul_ffn_down, RPC_V(w), RPC_V(w_type), RPC_V(gate), RPC_V(up),
					 RPC_V(y), RPC_V(n), RPC_V(k), RPC_V(activation));
}

static status_code stub_matmul_ffn_down_batch(backend *self, const buffer *w, uint32_t w_type,
											  const buffer *gate, const buffer *up, buffer *y,
											  int n, int k, int activation, int m) {
	return SEND_VALS(self, RPC_OP_matmul_ffn_down_batch, RPC_V(w), RPC_V(w_type), RPC_V(gate),
					 RPC_V(up), RPC_V(y), RPC_V(n), RPC_V(k), RPC_V(activation), RPC_V(m));
}

static status_code stub_rmsnorm_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
									  int n, float eps, int m) {
	return SEND_VALS(self, RPC_OP_rmsnorm_batch, RPC_V(x), RPC_V(w), RPC_V(y), RPC_V(n), RPC_V(eps),
					 RPC_V(m));
}

static status_code stub_matmul_batch(backend *self, const buffer *w, uint32_t w_type,
									 const buffer *x, buffer *y, int n, int k, int m) {
	return SEND_VALS(self, RPC_OP_matmul_batch, RPC_V(w), RPC_V(w_type), RPC_V(x), RPC_V(y),
					 RPC_V(n), RPC_V(k), RPC_V(m));
}

static status_code stub_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
	return SEND_VALS(self, RPC_OP_add_batch, RPC_V(x), RPC_V(y), RPC_V(n), RPC_V(m));
}

static status_code stub_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up,
										   buffer *out, int n, int activation, int m) {
	return SEND_VALS(self, RPC_OP_ffn_activate_batch, RPC_V(gate), RPC_V(up), RPC_V(out), RPC_V(n),
					 RPC_V(activation), RPC_V(m));
}

static status_code stub_ffn_activate_fused_batch(backend *self, const buffer *fused, buffer *out,
												 int n, int activation, int m) {
	return SEND_VALS(self, RPC_OP_ffn_activate_fused_batch, RPC_V(fused), RPC_V(out), RPC_V(n),
					 RPC_V(activation), RPC_V(m));
}

static status_code stub_attention(backend *self, const buffer *q, const buffer *k_cache,
								  const buffer *v_cache, buffer *out, int layer, int pos,
								  int n_heads, int n_kv_heads, int head_dim, int n_ctx,
								  int flash_attn, float scale, int n_kv_heads_active) {
	return SEND_VALS(self, RPC_OP_attention, RPC_V(q), RPC_V(k_cache), RPC_V(v_cache), RPC_V(out),
					 RPC_V(layer), RPC_V(pos), RPC_V(n_heads), RPC_V(n_kv_heads), RPC_V(head_dim),
					 RPC_V(n_ctx), RPC_V(flash_attn), RPC_V(scale), RPC_V(n_kv_heads_active));
}

static status_code stub_attention_swa(backend *self, const buffer *q, const buffer *k_cache,
									  const buffer *v_cache, buffer *out, int layer, int pos,
									  int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									  int flash_attn, float scale, int sliding_window,
									  int n_kv_heads_active) {
	return SEND_VALS(self, RPC_OP_attention_swa, RPC_V(q), RPC_V(k_cache), RPC_V(v_cache),
					 RPC_V(out), RPC_V(layer), RPC_V(pos), RPC_V(n_heads), RPC_V(n_kv_heads),
					 RPC_V(head_dim), RPC_V(n_ctx), RPC_V(flash_attn), RPC_V(scale),
					 RPC_V(sliding_window), RPC_V(n_kv_heads_active));
}

static status_code stub_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
										const buffer *v_cache, buffer *out, int layer,
										int pos_start, int n_heads, int n_kv_heads, int head_dim,
										int n_ctx, int flash_attn, float scale,
										int n_kv_heads_active, int m) {
	return SEND_VALS(self, RPC_OP_attention_batch, RPC_V(q), RPC_V(k_cache), RPC_V(v_cache),
					 RPC_V(out), RPC_V(layer), RPC_V(pos_start), RPC_V(n_heads), RPC_V(n_kv_heads),
					 RPC_V(head_dim), RPC_V(n_ctx), RPC_V(flash_attn), RPC_V(scale),
					 RPC_V(n_kv_heads_active), RPC_V(m));
}

static status_code stub_attention_swa_batch(backend *self, const buffer *q, const buffer *k_cache,
											const buffer *v_cache, buffer *out, int layer,
											int pos_start, int n_heads, int n_kv_heads,
											int head_dim, int n_ctx, int flash_attn, float scale,
											int sliding_window, int n_kv_heads_active, int m) {
	return SEND_VALS(self, RPC_OP_attention_swa_batch, RPC_V(q), RPC_V(k_cache), RPC_V(v_cache),
					 RPC_V(out), RPC_V(layer), RPC_V(pos_start), RPC_V(n_heads), RPC_V(n_kv_heads),
					 RPC_V(head_dim), RPC_V(n_ctx), RPC_V(flash_attn), RPC_V(scale),
					 RPC_V(sliding_window), RPC_V(n_kv_heads_active), RPC_V(m));
}

static status_code stub_attention_full_batch(backend *self, const buffer *q, const buffer *k,
											 const buffer *v, buffer *out, int n_heads,
											 int n_kv_heads, int head_dim, int n_rows,
											 int flash_attn, float scale) {
	return SEND_VALS(self, RPC_OP_attention_full_batch, RPC_V(q), RPC_V(k), RPC_V(v), RPC_V(out),
					 RPC_V(n_heads), RPC_V(n_kv_heads), RPC_V(head_dim), RPC_V(n_rows),
					 RPC_V(flash_attn), RPC_V(scale));
}

static status_code stub_matmul_gate_act_batch(backend *self, const buffer *w_gate,
											  uint32_t gate_type, const buffer *w_up,
											  uint32_t up_type, const buffer *x, buffer *act_out,
											  buffer *up_out, int n, int k, int activation, int m) {
	return SEND_VALS(self, RPC_OP_matmul_gate_act_batch, RPC_V(w_gate), RPC_V(gate_type),
					 RPC_V(w_up), RPC_V(up_type), RPC_V(x), RPC_V(act_out), RPC_V(up_out), RPC_V(n),
					 RPC_V(k), RPC_V(activation), RPC_V(m));
}

static status_code stub_matmul_softcap_batch(backend *self, const buffer *w, uint32_t w_type,
											 const buffer *x, buffer *y, int n, int k, int m,
											 float cap) {
	return SEND_VALS(self, RPC_OP_matmul_softcap_batch, RPC_V(w), RPC_V(w_type), RPC_V(x), RPC_V(y),
					 RPC_V(n), RPC_V(k), RPC_V(m), RPC_V(cap));
}

static status_code stub_layernorm_modulate(backend *self, const buffer *x, const buffer *scale,
										   buffer *y, int rows, int dim, float eps) {
	return SEND_VALS(self, RPC_OP_layernorm_modulate, RPC_V(x), RPC_V(scale), RPC_V(y), RPC_V(rows),
					 RPC_V(dim), RPC_V(eps));
}

static status_code stub_gated_add(backend *self, buffer *x, const buffer *y, const buffer *gate,
								  int rows, int dim) {
	return SEND_VALS(self, RPC_OP_gated_add, RPC_V(x), RPC_V(y), RPC_V(gate), RPC_V(rows),
					 RPC_V(dim));
}

static status_code stub_add_bias(backend *self, buffer *x, const buffer *bias, int rows, int dim) {
	return SEND_VALS(self, RPC_OP_add_bias, RPC_V(x), RPC_V(bias), RPC_V(rows), RPC_V(dim));
}

static status_code stub_activate(backend *self, buffer *x, int n, int activation) {
	return SEND_VALS(self, RPC_OP_activate, RPC_V(x), RPC_V(n), RPC_V(activation));
}

static status_code stub_rope_table(backend *self, buffer *x, const buffer *table, int rows,
								   int heads, int head_dim) {
	return SEND_VALS(self, RPC_OP_rope_table, RPC_V(x), RPC_V(table), RPC_V(rows), RPC_V(heads),
					 RPC_V(head_dim));
}

static status_code stub_kv_put_mla(backend *self, buffer *kv_cache, int layer, int pos,
								   const buffer *kv_a_in, const buffer *kv_a_norm_w, int kv_lora,
								   int qk_rope, int n_ctx, float eps) {
	return SEND_VALS(self, RPC_OP_kv_put_mla, RPC_V(kv_cache), RPC_V(layer), RPC_V(pos),
					 RPC_V(kv_a_in), RPC_V(kv_a_norm_w), RPC_V(kv_lora), RPC_V(qk_rope),
					 RPC_V(n_ctx), RPC_V(eps));
}

static status_code stub_moe_activate(backend *self, const buffer *gate, const buffer *up,
									 buffer *out, int n, float gate_scale, float up_scale,
									 int use_gelu) {
	return SEND_VALS(self, RPC_OP_moe_activate, RPC_V(gate), RPC_V(up), RPC_V(out), RPC_V(n),
					 RPC_V(gate_scale), RPC_V(up_scale), RPC_V(use_gelu));
}

static status_code stub_moe_router_topk(backend *self, const buffer *logits, const buffer *bias,
										buffer *out, int rows, int n_experts, int top_k,
										int use_softmax, int norm_topk, float routed_scale,
										int n_group, int topk_group) {
	return SEND_VALS(self, RPC_OP_moe_router_topk, RPC_V(logits), RPC_V(bias), RPC_V(out),
					 RPC_V(rows), RPC_V(n_experts), RPC_V(top_k), RPC_V(use_softmax),
					 RPC_V(norm_topk), RPC_V(routed_scale), RPC_V(n_group), RPC_V(topk_group));
}
