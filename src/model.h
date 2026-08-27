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

typedef struct layer_weights {
	weight_ref attn_norm_w;
	weight_ref wq, wk, wv, wo;
	weight_ref ffn_norm_w;
	weight_ref gate_w, up_w, down_w;

	weight_ref post_attn_norm_w;
	weight_ref post_ffn_norm_w;

	weight_ref gate_up_w;
	int		   gate_up_fused;
	void	  *gate_up_fused_host;

	int	  shexp_fused;
	void *shexp_fused_host;

	weight_ref qkv_w;
	int		   qkv_fused;
	void	  *qkv_fused_host;

	int is_sliding;

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

	int		head_dim;
	int		intermediate;
	uint8_t is_global_layer;
	float	layer_out_scale;

	int n_kv_heads;
	int has_own_v;
	int rope_dim;
	int is_recurrent;

	weight_ref q_a_w, q_b_w, q_a_norm_w;
	weight_ref kv_a_w, k_b_w, v_b_w, kv_a_norm_w;
	int		   mla_kb_f32;
	int		   mla_vb_f32;

	weight_ref router_w;
	weight_ref router_bias;
	weight_ref router_scale_w;

	weight_ref shexp_gate_w, shexp_up_w, shexp_down_w;

	weight_ref ffn_pre_norm_2_w;
	weight_ref ffn_post_norm_1_w;
	weight_ref ffn_post_norm_2_w;

	struct expert_desc {
		const void *gate_w;
		const void *up_w;
		const void *down_w;
		uint32_t	gate_type, up_type, down_type;
		int			gate_up_fused;
		float		gate_scale;
		float		up_scale;
		float		down_scale;

		uint64_t gate_off;
		uint64_t up_off;
		uint64_t down_off;
	}		*experts;
	int		 is_moe_layer;
	int		 any_fused_experts;
	uint32_t gate_q8_type;
} layer_weights;

typedef struct model_mla_params {
	int q_lora;
	int kv_lora;
	int qk_nope;
	int qk_rope;
	int qk_head;
	int v_head;
} model_mla_params;

typedef struct model_moe_params {
	int	  n_experts;
	int	  n_experts_used;
	int	  n_shared_experts;
	int	  moe_intermediate;
	int	  n_group;
	int	  topk_group;
	float routed_scale;
	int	  norm_topk_prob;
	int	  first_dense_layer;
	float router_dim_scale;
} model_moe_params;

typedef struct model_layer_dims_params {
	int		 head_dim_swa;
	int		 head_dim_global;
	int		 rope_dim_swa;
	int		 rope_dim_global;
	float	 rope_theta_swa;
	float	 rope_theta_global;
	int		 n_embd_per_layer;
	uint8_t *is_global_layer;
	int		*ffn_lengths;
	int		*n_kv_heads_per_layer;
	int		 n_layer_kv_from_start;
	int		 kv_layer_swa;
	int		 kv_layer_global;

	weight_ref per_layer_tok_embd;
	weight_ref per_layer_model_proj;
	weight_ref per_layer_proj_norm_w;
} model_layer_dims_params;

typedef struct model_hybrid_params {
	int conv_kernel;
	int inner_size;
	int state_size;
	int n_value_heads;
	int n_key_heads;
	int full_attention_interval;
	int key_dim;
	int value_head_dim;
	int value_dim;
	int conv_dim;
	int conv_channels;

	uint8_t *recurrent_layers;
} model_hybrid_params;

typedef struct model {
	model_arch		 arch;
	const arch_info *arch_info;

	int	  n_layers, n_ctx, dim, n_heads, n_kv_heads, head_dim;
	float dim_sqrt;
	int	  intermediate, vocab_size, rope_dim;
	float norm_eps, rope_theta;
	int	  tie_embeddings;

	float attn_logit_softcap;
	float final_logit_softcap;
	int	  sliding_window;

	int			 has_per_layer_embeddings;
	const float *rope_freqs;
	weight_ref	 rope_freqs_w;
	int			 rope_freqs_count;

	weight_ref	   tok_embd, output_norm_w, output_w;
	layer_weights *layers;

	model_mla_params		mla;
	model_moe_params		moe;
	model_layer_dims_params layer_dims;
	model_hybrid_params		hybrid;

	struct moe_stream_cache *moe_cache;
	int						 moe_stream_enabled;

	backend	   *backend;
	int			owns_backend;
	int			use_mmap;
	char	   *model_path;
	const char *repack_config;
	const char *fuse_config;
	int			qkv_fused_layers;

	gguf_ctx gctx;

	model_recipe *recipe;
	int			  batchable;
	weight_ref	**wrefs_by_layer;

	backend **layer_backends;
	int		  n_layer_backends;
	int		  mixed_backend_mode;
} model;

status_code model_load(model *m, const char *path);
status_code model_load_backend_ex_repack(model *m, const char *path, backend *accel, int use_mmap,
										 const char *repack_config, int requested_n_ctx);

status_code model_load_parse(model *m, const char *path, backend *accel, int use_mmap,
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

#endif