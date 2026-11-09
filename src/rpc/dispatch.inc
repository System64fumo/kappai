case RPC_OP_kv_put:
return be->kv_put(be, RPC_VB(0), RPC_VB(1), RPC_I(2), RPC_I(3), RPC_CB(4), RPC_CB(5), RPC_I(6),
				  RPC_I(7), RPC_I(8), RPC_I(9));
case RPC_OP_kv_put_batch:
return be->kv_put_batch(be, RPC_VB(0), RPC_VB(1), RPC_I(2), RPC_I(3), RPC_CB(4), RPC_CB(5),
						RPC_I(6), RPC_I(7), RPC_I(8), RPC_I(9), RPC_I(10), RPC_I(11));
case RPC_OP_embd_lookup:
return be->embd_lookup(be, RPC_CB(0), RPC_U(1), RPC_I(2), RPC_I(3), RPC_VB(4));
case RPC_OP_rmsnorm:
return be->rmsnorm(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3), RPC_F(4));
case RPC_OP_matmul:
return be->matmul(be, RPC_CB(0), RPC_U(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_I(5));
case RPC_OP_matmul_residual:
return be->matmul_residual(be, RPC_CB(0), RPC_U(1), RPC_CB(2), RPC_CB(3), RPC_VB(4), RPC_I(5),
						   RPC_I(6));
case RPC_OP_matmul_qonly:
return be->matmul_qonly(be, RPC_CB(0), RPC_U(1), RPC_CB(2), RPC_U(3), RPC_VB(4), RPC_I(5), RPC_I(6),
						RPC_I(7));
case RPC_OP_prequantize_x:
return be->prequantize_x(be, RPC_CB(0), RPC_I(1), RPC_U(2), RPC_VB(3));
case RPC_OP_add_inplace:
return be->add_inplace(be, RPC_VB(0), RPC_CB(1), RPC_I(2));
case RPC_OP_scale_inplace:
return be->scale_inplace(be, RPC_VB(0), RPC_F(1), RPC_I(2));
case RPC_OP_softcap:
return be->softcap(be, RPC_VB(0), RPC_F(1), RPC_I(2));
case RPC_OP_split_qgate:
return be->split_qgate(be, RPC_CB(0), RPC_VB(1), RPC_VB(2), RPC_I(3), RPC_I(4), RPC_I(5));
case RPC_OP_attn_output_gate:
return be->attn_output_gate(be, RPC_VB(0), RPC_CB(1), RPC_I(2), RPC_I(3));
case RPC_OP_copy_buffer:
return be->copy_buffer(be, RPC_CB(0), RPC_VB(1), RPC_I(2));
case RPC_OP_ple_combine:
return be->ple_combine(be, RPC_VB(0), RPC_CB(1), RPC_I(2), RPC_F(3));
case RPC_OP_ffn_activate:
return be->ffn_activate(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3));
case RPC_OP_ffn_activate_ex:
return be->ffn_activate_ex(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3), RPC_I(4));
case RPC_OP_rmsnorm_per_head:
return be->rmsnorm_per_head(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3), RPC_I(4), RPC_F(5));
case RPC_OP_rmsnorm_noweight:
return be->rmsnorm_noweight(be, RPC_CB(0), RPC_VB(1), RPC_I(2), RPC_F(3));
case RPC_OP_rmsnorm_noweight_per_head:
return be->rmsnorm_noweight_per_head(be, RPC_CB(0), RPC_VB(1), RPC_I(2), RPC_I(3), RPC_F(4));
case RPC_OP_rmsnorm_add:
return be->rmsnorm_add(be, RPC_CB(0), RPC_CB(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_F(5),
					   RPC_F(6));
case RPC_OP_rmsnorm_add_batch:
return be->rmsnorm_add_batch(be, RPC_CB(0), RPC_CB(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_F(5),
							 RPC_F(6), RPC_I(7));
case RPC_OP_rmsnorm_add_norm_batch:
return be->rmsnorm_add_norm_batch(be, RPC_CB(0), RPC_CB(1), RPC_CB(2), RPC_VB(3), RPC_CB(4),
								  RPC_VB(5), RPC_I(6), RPC_F(7), RPC_F(8), RPC_I(9));
case RPC_OP_rmsnorm_per_head_batch:
return be->rmsnorm_per_head_batch(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3), RPC_I(4), RPC_F(5),
								  RPC_I(6));
case RPC_OP_rmsnorm_noweight_batch:
return be->rmsnorm_noweight_batch(be, RPC_CB(0), RPC_VB(1), RPC_I(2), RPC_F(3), RPC_I(4));
case RPC_OP_rmsnorm_noweight_per_head_batch:
return be->rmsnorm_noweight_per_head_batch(be, RPC_CB(0), RPC_VB(1), RPC_I(2), RPC_I(3), RPC_F(4),
										   RPC_I(5));
case RPC_OP_matmul_ffn_down:
return be->matmul_ffn_down(be, RPC_CB(0), RPC_U(1), RPC_CB(2), RPC_CB(3), RPC_VB(4), RPC_I(5),
						   RPC_I(6), RPC_I(7));
case RPC_OP_matmul_ffn_down_batch:
return be->matmul_ffn_down_batch(be, RPC_CB(0), RPC_U(1), RPC_CB(2), RPC_CB(3), RPC_VB(4), RPC_I(5),
								 RPC_I(6), RPC_I(7), RPC_I(8));
case RPC_OP_rmsnorm_batch:
return be->rmsnorm_batch(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3), RPC_F(4), RPC_I(5));
case RPC_OP_matmul_batch:
return be->matmul_batch(be, RPC_CB(0), RPC_U(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_I(5),
						RPC_I(6));
case RPC_OP_add_batch:
return be->add_batch(be, RPC_VB(0), RPC_CB(1), RPC_I(2), RPC_I(3));
case RPC_OP_ffn_activate_batch:
return be->ffn_activate_batch(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3), RPC_I(4), RPC_I(5));
case RPC_OP_ffn_activate_fused_batch:
return be->ffn_activate_fused_batch(be, RPC_CB(0), RPC_VB(1), RPC_I(2), RPC_I(3), RPC_I(4));
case RPC_OP_attention:
return be->attention(be, RPC_CB(0), RPC_CB(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_I(5), RPC_I(6),
					 RPC_I(7), RPC_I(8), RPC_I(9), RPC_I(10), RPC_F(11), RPC_I(12));
case RPC_OP_attention_swa:
return be->attention_swa(be, RPC_CB(0), RPC_CB(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_I(5),
						 RPC_I(6), RPC_I(7), RPC_I(8), RPC_I(9), RPC_I(10), RPC_F(11), RPC_I(12),
						 RPC_I(13));
case RPC_OP_attention_batch:
return be->attention_batch(be, RPC_CB(0), RPC_CB(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_I(5),
						   RPC_I(6), RPC_I(7), RPC_I(8), RPC_I(9), RPC_I(10), RPC_F(11), RPC_I(12),
						   RPC_I(13));
case RPC_OP_attention_swa_batch:
return be->attention_swa_batch(be, RPC_CB(0), RPC_CB(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_I(5),
							   RPC_I(6), RPC_I(7), RPC_I(8), RPC_I(9), RPC_I(10), RPC_F(11),
							   RPC_I(12), RPC_I(13), RPC_I(14));
case RPC_OP_attention_full_batch:
return be->attention_full_batch(be, RPC_CB(0), RPC_CB(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_I(5),
								RPC_I(6), RPC_I(7), RPC_I(8), RPC_F(9));
case RPC_OP_matmul_gate_act_batch:
return be->matmul_gate_act_batch(be, RPC_CB(0), RPC_U(1), RPC_CB(2), RPC_U(3), RPC_CB(4), RPC_VB(5),
								 RPC_VB(6), RPC_I(7), RPC_I(8), RPC_I(9), RPC_I(10));
case RPC_OP_matmul_softcap_batch:
return be->matmul_softcap_batch(be, RPC_CB(0), RPC_U(1), RPC_CB(2), RPC_VB(3), RPC_I(4), RPC_I(5),
								RPC_I(6), RPC_F(7));
case RPC_OP_layernorm_modulate:
return be->layernorm_modulate(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3), RPC_I(4), RPC_F(5));
case RPC_OP_gated_add:
return be->gated_add(be, RPC_VB(0), RPC_CB(1), RPC_CB(2), RPC_I(3), RPC_I(4));
case RPC_OP_add_bias:
return be->add_bias(be, RPC_VB(0), RPC_CB(1), RPC_I(2), RPC_I(3));
case RPC_OP_activate:
return be->activate(be, RPC_VB(0), RPC_I(1), RPC_I(2));
case RPC_OP_rope_table:
return be->rope_table(be, RPC_VB(0), RPC_CB(1), RPC_I(2), RPC_I(3), RPC_I(4));
case RPC_OP_kv_put_mla:
return be->kv_put_mla(be, RPC_VB(0), RPC_I(1), RPC_I(2), RPC_CB(3), RPC_CB(4), RPC_I(5), RPC_I(6),
					  RPC_I(7), RPC_F(8));
case RPC_OP_moe_activate:
return be->moe_activate(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3), RPC_F(4), RPC_F(5),
						RPC_I(6));
case RPC_OP_moe_router_topk:
return be->moe_router_topk(be, RPC_CB(0), RPC_CB(1), RPC_VB(2), RPC_I(3), RPC_I(4), RPC_I(5),
						   RPC_I(6), RPC_I(7), RPC_F(8), RPC_I(9), RPC_I(10));
