#ifndef MODEL_H
#define MODEL_H

#include "arch.h"
#include "backend/backend.h"
#include "common.h"
#include "gguf.h"

struct model_recipe;
typedef struct model_recipe model_recipe;

typedef struct weight_ref {
	const void *host_ptr;
	uint32_t	type;
	buffer		buf;
} weight_ref;

struct expert_desc {
	const void *gate_w;
	const void *up_w;
	const void *down_w;
	uint32_t	gate_type, up_type, down_type;
	bool		gate_up_fused;
	float		gate_scale;
	float		up_scale;
	float		down_scale;

	uint64_t gate_off;
	uint64_t up_off;
	uint64_t down_off;
};

typedef struct layer_weights {
	void			   *gate_up_fused_host;
	void			   *shexp_fused_host;
	void			   *qkv_fused_host;
	struct expert_desc *experts;

	weight_ref attn_norm_w;
	weight_ref wq, wk, wv, wo;
	weight_ref ffn_norm_w;
	weight_ref gate_w, up_w, down_w;

	weight_ref post_attn_norm_w;
	weight_ref post_ffn_norm_w;

	weight_ref gate_up_w;
	weight_ref qkv_w;

	weight_ref attn_q_norm_w;
	weight_ref attn_k_norm_w;
	weight_ref attn_qkv_w;
	weight_ref attn_gate_w;
	weight_ref ssm_conv1d_w;
	weight_ref ssm_dt_b;
	weight_ref ssm_a;
	weight_ref ssm_beta_w;
	weight_ref ssm_alpha_w;
	weight_ref ssm_norm_w;
	weight_ref ssm_out_w;
	weight_ref ple_post_norm_w;
	weight_ref ple_inp_gate_w;
	weight_ref ple_proj_w;
	weight_ref layer_out_scale_w;

	weight_ref q_a_w, q_b_w, q_a_norm_w;
	weight_ref kv_a_w, k_b_w, v_b_w, kv_a_norm_w;

	weight_ref router_w;
	weight_ref router_bias;
	weight_ref router_scale_w;

	weight_ref shexp_gate_w, shexp_up_w, shexp_down_w;

	weight_ref ffn_pre_norm_2_w;
	weight_ref ffn_post_norm_1_w;
	weight_ref ffn_post_norm_2_w;

	int32_t	 head_dim;
	int32_t	 intermediate;
	float	 layer_out_scale;
	int32_t	 n_kv_heads;
	int32_t	 rope_dim;
	uint32_t gate_q8_type;

	bool	gate_up_fused;
	bool	shexp_fused;
	bool	qkv_fused;
	bool	is_sliding;
	uint8_t is_global_layer;
	bool	has_own_v;
	bool	is_recurrent;
	bool	mla_kb_f32;
	bool	mla_vb_f32;
	bool	is_moe_layer;
	bool	any_fused_experts;
} layer_weights;

typedef struct model_mla_params {
	int32_t q_lora;
	int32_t kv_lora;
	int32_t qk_nope;
	int32_t qk_rope;
	int32_t qk_head;
	int32_t v_head;
} model_mla_params;

typedef struct model_moe_params {
	int32_t n_experts;
	int32_t n_experts_used;
	int32_t n_shared_experts;
	int32_t moe_intermediate;
	int32_t n_group;
	int32_t topk_group;
	float	routed_scale;
	bool	norm_topk_prob;
	int32_t first_dense_layer;
	float	router_dim_scale;
	bool	experts_resident;
} model_moe_params;

typedef struct model_layer_dims_params {
	int32_t	 head_dim_swa;
	int32_t	 head_dim_global;
	int32_t	 rope_dim_swa;
	int32_t	 rope_dim_global;
	float	 rope_theta_swa;
	float	 rope_theta_global;
	int32_t	 n_embd_per_layer;
	uint8_t *is_global_layer;
	int32_t *ffn_lengths;
	int32_t *n_kv_heads_per_layer;
	int32_t	 n_layer_kv_from_start;
	int32_t	 kv_layer_swa;
	int32_t	 kv_layer_global;

	weight_ref per_layer_tok_embd;
	weight_ref per_layer_model_proj;
	weight_ref per_layer_proj_norm_w;
} model_layer_dims_params;

typedef struct model_hybrid_params {
	int32_t conv_kernel;
	int32_t inner_size;
	int32_t state_size;
	int32_t n_value_heads;
	int32_t n_key_heads;
	int32_t full_attention_interval;
	int32_t key_dim;
	int32_t value_head_dim;
	int32_t value_dim;
	int32_t conv_dim;
	int32_t conv_channels;

	uint8_t *recurrent_layers;
} model_hybrid_params;

typedef struct model {
	const arch_info			*arch_info;
	const float				*rope_freqs;
	weight_ref				 rope_freqs_w;
	weight_ref				 tok_embd, output_norm_w, output_w;
	layer_weights			*layers;
	struct moe_stream_cache *moe_cache;
	backend					*backend;
	char					*model_path;
	const char				*repack_config;
	const char				*fuse_config;
	model_recipe			*recipe;
	weight_ref			   **wrefs_by_layer;
	backend				   **layer_backends;

	model_layer_dims_params layer_dims;
	model_hybrid_params		hybrid;
	gguf_ctx				gctx;

	model_mla_params mla;
	model_moe_params moe;

	model_arch arch;
	int32_t	   n_layers, n_ctx, dim, n_heads, n_kv_heads, head_dim;
	float	   dim_sqrt;
	int32_t	   intermediate, vocab_size, rope_dim;
	float	   norm_eps, rope_theta;
	float	   attn_logit_softcap;
	float	   final_logit_softcap;
	int32_t	   sliding_window;
	int32_t	   rope_freqs_count;
	int32_t	   qkv_fused_layers;
	int32_t	   n_layer_backends;

	bool tie_embeddings;
	bool has_per_layer_embeddings;
	bool moe_stream_enabled;
	bool owns_backend;
	bool use_mmap;
	bool batchable;
	bool mixed_backend_mode;
} model;

status_code model_load(model *m, const char *path);
status_code model_load_backend_ex_repack(model *m, const char *path, backend *bk, int use_mmap,
										 const char *repack_config, int requested_n_ctx);

status_code model_load_parse(model *m, const char *path, backend *bk, int use_mmap,
							 const char *repack_config, int requested_n_ctx);
status_code model_upload_weights(model *m);
status_code model_build_recipe(model *m);
void		model_free(model *m);

int model_should_repack(uint32_t type, const char *repack_config);

static inline backend *model_layer_backend(const model *m, int li) {
	if (!m || !m->layer_backends || li < 0 || li >= m->n_layer_backends)
		return m ? m->backend : NULL;
	backend *b = m->layer_backends[li];
	return b ? b : m->backend;
}

static inline int model_layer_backend_is_host(const model *m, int li) {
	backend *b = model_layer_backend(m, li);
	return b && backend_has_cap(b, BCAP_IS_HOST);
}

static inline int model_mixed_backend_mode(const model *m) {
	return m && m->mixed_backend_mode;
}

status_code model_set_layer_backend_range(model *m, int begin, int end, backend *b);
status_code model_build_weight_refs(model *m);

static inline int model_layer_is_sliding(const model *m, int li) {
	return m->sliding_window > 0 && m->layers[li].is_sliding;
}

static inline int model_layer_head_dim(const model *m, int li) {
	if (!m->arch_info->has_variable_layer_dims)
		return m->head_dim;
	if (li < 0 || li >= m->n_layers)
		return m->head_dim;
	return m->layers[li].head_dim;
}

static inline int model_layer_intermediate(const model *m, int li) {
	if (!m->arch_info->has_variable_layer_dims)
		return m->intermediate;
	return m->layers[li].intermediate;
}

static inline int model_layer_kv_heads(const model *m, int li) {
	if (!m->arch_info->has_variable_layer_dims)
		return m->n_kv_heads;
	return m->layers[li].n_kv_heads;
}

static inline int model_layer_has_own_v(const model *m, int li) {
	if (!m->arch_info->has_variable_layer_dims)
		return 1;
	return m->layers[li].has_own_v;
}

static inline int model_layer_has_kv(const model *m, int li) {
	if (!m->arch_info->has_variable_layer_dims)
		return 1;
	if (m->layer_dims.n_layer_kv_from_start <= 0)
		return 1;
	return li < m->layer_dims.n_layer_kv_from_start;
}

static inline int model_layer_rope_dim(const model *m, int li) {
	if (!m->arch_info->has_variable_layer_dims)
		return m->rope_dim;
	return m->layers[li].rope_dim;
}

static inline int model_layer_is_moe(const model *m, int li) {
	if (!m->arch_info->is_moe)
		return 0;
	if (li < 0 || li >= m->n_layers)
		return 0;
	return m->layers[li].is_moe_layer;
}

static inline int model_layer_is_recurrent(const model *m, int li) {
	if (!m || !m->arch_info || !m->arch_info->is_hybrid_recurrent)
		return 0;
	if (li < 0 || li >= m->n_layers)
		return 0;
	return m->layers[li].is_recurrent;
}

static inline int model_hybrid_proj_size(const model *m) {
	int q_out = m->n_heads * m->head_dim;
	int s	  = m->hybrid.conv_dim > 2 * q_out ? m->hybrid.conv_dim : 2 * q_out;
	return s;
}

static inline int model_hybrid_gate_size(const model *m) {
	int q_out = m->n_heads * m->head_dim;
	int s	  = m->hybrid.value_dim > q_out ? m->hybrid.value_dim : q_out;
	return s;
}

static inline tpool *model_get_pool(const model *m) {
	if (m->backend && m->backend->get_pool)
		return m->backend->get_pool(m->backend);
	return NULL;
}

#endif