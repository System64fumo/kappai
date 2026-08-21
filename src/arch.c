#include "arch.h"
#include "log.h"

#include <string.h>

static const arch_info k_registry[] = {
	{
		.arch						= ARCH_LLAMA,
		.gguf_name					= "llama",
		.key_prefix					= "llama",
		.has_scale_embeddings		= false,
		.default_rope_theta			= 10000.0f,
		.sliding_window_period		= 0,
		.has_qk_norm				= false,
		.has_per_layer_embeddings	= false,
		.has_post_norm_ple			= false,
		.uses_gelu_activation		= false,
		.uses_norm_v_without_weight = false,
		.has_attn_post_norm			= false,
		.has_ffn_post_norm			= false,
		.has_layer_output_scale		= false,
		.has_variable_layer_dims	= false,
		.uses_neox_rope				= false,
	},
	{
		.arch						= ARCH_GEMMA4,
		.gguf_name					= "gemma4",
		.key_prefix					= "gemma4",
		.has_scale_embeddings		= true,
		.default_rope_theta			= 1000000.0f,
		.sliding_window_period		= 5,
		.has_qk_norm				= true,
		.has_per_layer_embeddings	= true,
		.has_post_norm_ple			= true,
		.uses_gelu_activation		= true,
		.uses_norm_v_without_weight = true,
		.has_attn_post_norm			= true,
		.has_ffn_post_norm			= true,
		.has_layer_output_scale		= true,
		.has_variable_layer_dims	= true,
		.uses_neox_rope				= true,
	},
	{
		.arch						= ARCH_GEMMA4_MOE,
		.gguf_name					= "gemma4_moe",
		.key_prefix					= "gemma4",
		.has_scale_embeddings		= true,
		.default_rope_theta			= 1000000.0f,
		.sliding_window_period		= 5,
		.has_qk_norm				= true,
		.has_per_layer_embeddings	= true,
		.has_post_norm_ple			= true,
		.uses_gelu_activation		= true,
		.uses_norm_v_without_weight = true,
		.has_attn_post_norm			= true,
		.has_ffn_post_norm			= true,
		.has_layer_output_scale		= true,
		.has_variable_layer_dims	= true,
		.uses_neox_rope				= true,
		.is_moe						= true,
		.uses_moe_softmax_router	= true,
		.uses_moe_shared_dense_ffn	= true,
		.uses_moe_norm_topk_prob	= true,
	},
	{
		.arch					 = ARCH_GLM_DSA,
		.gguf_name				 = "glm-dsa",
		.key_prefix				 = "glm-dsa",
		.default_rope_theta		 = 10000.0f,
		.uses_neox_rope			 = true,
		.is_moe					 = true,
		.has_shared_expert		 = true,
		.is_mla					 = true,
		.moe_first_dense_layers	 = 3,
		.uses_moe_norm_topk_prob = true,
	},
	{
		.arch						 = ARCH_QWEN35,
		.gguf_name					 = "qwen35",
		.key_prefix					 = "qwen35",
		.default_rope_theta			 = 10000000.0f,
		.has_qk_norm				 = true,
		.has_attn_post_norm			 = true,
		.has_attn_output_gate		 = true,
		.uses_post_attn_norm_for_ffn = true,
		.is_hybrid_recurrent		 = true,
		.uses_neox_rope				 = true,
	},
};

const arch_info *arch_lookup(model_arch a) {
	if (a == ARCH_UNKNOWN)
		return NULL;
	for (size_t i = 0; i < ARRAY_LEN(k_registry); i++) {
		if (k_registry[i].arch == a)
			return &k_registry[i];
	}
	return NULL;
}

const arch_info *arch_lookup_by_gguf_name(const char *name) {
	if (!name)
		return NULL;
	for (size_t i = 0; i < ARRAY_LEN(k_registry); i++) {
		if (strcmp(k_registry[i].gguf_name, name) == 0)
			return &k_registry[i];
	}
	return NULL;
}

model_arch arch_detect(const gguf_ctx *g) {
	const char *name = NULL;
	if (gguf_get_str(g, "general.architecture", &name) != OK || !name) {
		WARN("model: missing 'general.architecture', assuming llama");
		return ARCH_LLAMA;
	}

	const arch_info *info = arch_lookup_by_gguf_name(name);
	if (!info) {
		ERROR("model: unrecognized architecture '%s'", name);
		return ARCH_UNKNOWN;
	}
	return info->arch;
}
