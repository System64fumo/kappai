#ifndef ARCH_H
#define ARCH_H

#include "common.h"
#include "gguf.h"

typedef enum {
	ARCH_UNKNOWN = 0,
	ARCH_LLAMA,
	ARCH_GEMMA4,
	ARCH_GEMMA4_MOE,
	ARCH_GLM_DSA,
	ARCH_QWEN35,
	ARCH_LFM2,
} model_arch;

typedef struct {
	const char *gguf_name;
	const char *key_prefix;
	model_arch	arch;

	float default_rope_theta;
	int	  sliding_window_period;
	int	  moe_first_dense_layers;

	bool has_scale_embeddings;
	bool has_qk_norm;
	bool has_per_layer_embeddings;
	bool has_post_norm_ple;
	bool uses_gelu_activation;
	bool uses_norm_v_without_weight;
	bool has_attn_post_norm;
	bool has_ffn_post_norm;
	bool has_layer_output_scale;
	bool has_variable_layer_dims;
	bool uses_neox_rope;
	bool is_moe;
	bool uses_moe_softmax_router;
	bool has_shared_expert;
	bool uses_moe_shared_dense_ffn;
	bool is_mla;
	bool is_hybrid_recurrent;
	bool hybrid_shortconv;
	bool has_attn_output_gate;
	bool uses_post_attn_norm_for_ffn;
	bool uses_moe_norm_topk_prob;
} arch_info;

const arch_info *arch_lookup(model_arch a);
const arch_info *arch_lookup_by_gguf_name(const char *name);

model_arch arch_detect(const gguf_ctx *g);

#endif
