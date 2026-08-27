#include "test_core.h"

typedef struct {
	model m;
	void *tok_embd_raw;
	void *output_norm_raw;
	void *output_w_raw;
	int	  n_layers;
	int	  n_ctx;
} synth_model;

typedef struct {
	int	  n_layers, dim, n_heads, n_kv_heads, head_dim, intermediate, vocab;
	int	  n_experts;
	int	  n_experts_used;
	int	  n_shared_experts;
	int	  moe_intermediate;
	int	  n_group;
	int	  topk_group;
	float routed_scale;
	int	  norm_topk_prob;
	int	  first_dense_layer;
	int	  n_layer_kv_from_start;
	int	  q_lora;
	int	  kv_lora;
	int	  qk_head;
	int	  qk_rope;
	int	  qk_nope;
	int	  v_head;
} synth_cfg;

static void *alloc_q4_k_weights(int rows, int cols) {
	if (cols % 256 != 0) {
		fprintf(stderr, "alloc_q4_k_weights: cols=%d not multiple of 256\n", cols);
		exit(2);
	}
	size_t blocks_per_row = (size_t)cols / 256;
	size_t n_blocks		  = (size_t)rows * blocks_per_row;
	void  *p			  = xcalloc(n_blocks, sizeof(q4_k_block));
	seed_test_rng(0x1234ULL + ((uint64_t)rows * 31) + ((uint64_t)cols * 7));
	fill_random_blocks(p, (int)n_blocks, sizeof(q4_k_block), GGML_TYPE_Q4_K);
	return p;
}

static void *alloc_f32_weights(int n, float bias) {
	float *p = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0xABCDULL + (uint64_t)n);
	fill_random_f32(p, n, 1.0f);
	for (int i = 0; i < n; i++)
		p[i] += bias;
	return p;
}

static const synth_cfg ARCH_CFG_SMALL = {
	.n_layers	  = 2,
	.dim		  = 512,
	.n_heads	  = 8,
	.n_kv_heads	  = 4,
	.head_dim	  = 64,
	.intermediate = 1024,
	.vocab		  = 512,
};

static const synth_cfg ARCH_CFG_DEEPER = {
	.n_layers	  = 4,
	.dim		  = 512,
	.n_heads	  = 8,
	.n_kv_heads	  = 4,
	.head_dim	  = 64,
	.intermediate = 1024,
	.vocab		  = 512,
};

static const synth_cfg ARCH_CFG_KV_SHARED = {
	.n_layers			   = 10,
	.dim				   = 512,
	.n_heads			   = 8,
	.n_kv_heads			   = 4,
	.head_dim			   = 64,
	.intermediate		   = 1024,
	.vocab				   = 512,
	.n_layer_kv_from_start = 5,
};

static const synth_cfg ARCH_CFG_MLA = {
	.n_layers		   = 2,
	.dim			   = 512,
	.n_heads		   = 8,
	.n_kv_heads		   = 8,
	.head_dim		   = 64,
	.intermediate	   = 1024,
	.vocab			   = 512,
	.n_experts		   = 16,
	.n_experts_used	   = 4,
	.n_shared_experts  = 1,
	.moe_intermediate  = 512,
	.routed_scale	   = 1.0f,
	.norm_topk_prob	   = 1,
	.first_dense_layer = 0,
	.q_lora			   = 256,
	.kv_lora		   = 128,
	.qk_head		   = 64,
	.qk_rope		   = 32,
	.qk_nope		   = 32,
	.v_head			   = 64,
};

static const synth_cfg ARCH_CFG_MLA_DENSE_FIRST = {
	.n_layers		   = 4,
	.dim			   = 512,
	.n_heads		   = 8,
	.n_kv_heads		   = 8,
	.head_dim		   = 64,
	.intermediate	   = 1024,
	.vocab			   = 512,
	.n_experts		   = 16,
	.n_experts_used	   = 4,
	.n_shared_experts  = 1,
	.moe_intermediate  = 512,
	.routed_scale	   = 1.0f,
	.norm_topk_prob	   = 1,
	.first_dense_layer = 2,
	.q_lora			   = 256,
	.kv_lora		   = 128,
	.qk_head		   = 64,
	.qk_rope		   = 32,
	.qk_nope		   = 32,
	.v_head			   = 64,
};

static const synth_cfg ARCH_CFG_MLA_WIDE = {
	.n_layers		   = 2,
	.dim			   = 512,
	.n_heads		   = 8,
	.n_kv_heads		   = 8,
	.head_dim		   = 96,
	.intermediate	   = 1024,
	.vocab			   = 512,
	.n_experts		   = 16,
	.n_experts_used	   = 4,
	.n_shared_experts  = 1,
	.moe_intermediate  = 512,
	.routed_scale	   = 1.0f,
	.norm_topk_prob	   = 1,
	.first_dense_layer = 0,
	.q_lora			   = 256,
	.kv_lora		   = 128,
	.qk_head		   = 96,
	.qk_rope		   = 32,
	.qk_nope		   = 64,
	.v_head			   = 96,
};

static const synth_cfg *g_synth_build_cfg = NULL;

static size_t synth_q4_k_buf_size(int rows, int cols) {
	if (cols % 256 != 0) {
		fprintf(stderr, "synth_q4_k_buf_size: cols=%d not multiple of 256\n", cols);
		exit(2);
	}
	size_t blocks_per_row = (size_t)cols / 256;
	return (size_t)rows * blocks_per_row * sizeof(q4_k_block);
}

static void *synth_alloc_q4_k(int rows, int cols) {
	return xcalloc(1, synth_q4_k_buf_size(rows, cols));
}

static float *synth_alloc_f32(int n) {
	return xmalloc((size_t)n * sizeof(float));
}

static void synth_fill_q4_k(void *p, int rows, int cols) {
	size_t blocks_per_row = (size_t)cols / 256;
	size_t n_blocks		  = (size_t)rows * blocks_per_row;
	seed_test_rng(0x1234ULL + ((uint64_t)rows * 31) + ((uint64_t)cols * 7));
	fill_random_blocks(p, (int)n_blocks, sizeof(q4_k_block), GGML_TYPE_Q4_K);
}

static void synth_fill_f32(float *p, int n, float bias) {
	seed_test_rng(0xABCDULL + (uint64_t)n);
	fill_random_f32(p, n, 1.0f);
	for (int i = 0; i < n; i++)
		p[i] += bias;
}

static void synth_alloc_layer(layer_weights *L, int li, const synth_cfg *cfg, model *m) {
	if (m->arch_info->has_variable_layer_dims) {
		L->is_global_layer = m->layer_dims.is_global_layer[li];
		L->head_dim =
			L->is_global_layer ? m->layer_dims.head_dim_global : m->layer_dims.head_dim_swa;
		L->intermediate = m->layer_dims.ffn_lengths[li];
		L->n_kv_heads	= m->layer_dims.n_kv_heads_per_layer[li];
	} else {
		L->head_dim		   = cfg->head_dim;
		L->intermediate	   = cfg->intermediate;
		L->is_global_layer = 0;
		L->n_kv_heads	   = cfg->n_kv_heads;
	}
	L->has_own_v = 1;

	int				 q_out		  = cfg->n_heads * L->head_dim;
	int				 kv_out		  = L->n_kv_heads * L->head_dim;
	const arch_info *ai			  = m->arch_info;
	int				 is_moe_layer = cfg->n_experts > 0 && li >= cfg->first_dense_layer;
	L->is_moe_layer				  = is_moe_layer;

	L->attn_norm_w.host_ptr = synth_alloc_f32(cfg->dim);
	L->wq.host_ptr			= synth_alloc_q4_k(q_out, cfg->dim);
	L->wk.host_ptr			= synth_alloc_q4_k(kv_out, cfg->dim);
	L->wv.host_ptr			= synth_alloc_q4_k(kv_out, cfg->dim);
	L->wo.host_ptr			= synth_alloc_q4_k(cfg->dim, q_out);
	L->ffn_norm_w.host_ptr	= synth_alloc_f32(cfg->dim);
	L->gate_w.host_ptr		= synth_alloc_q4_k(cfg->intermediate, cfg->dim);
	L->up_w.host_ptr		= synth_alloc_q4_k(cfg->intermediate, cfg->dim);
	L->down_w.host_ptr		= synth_alloc_q4_k(cfg->dim, cfg->intermediate);

	if (ai->has_qk_norm) {
		int hd					  = L->head_dim;
		L->attn_q_norm_w.host_ptr = synth_alloc_f32(hd);
		L->attn_k_norm_w.host_ptr = synth_alloc_f32(hd);
	}
	if (ai->has_attn_post_norm) {
		L->post_attn_norm_w.host_ptr = synth_alloc_f32(cfg->dim);
	}
	if (ai->has_ffn_post_norm) {
		L->post_ffn_norm_w.host_ptr = synth_alloc_f32(cfg->dim);
	}

	if (is_moe_layer) {
		int E = cfg->n_experts;

		L->router_w.host_ptr = synth_alloc_f32(cfg->dim * E);
		if (ai->uses_moe_softmax_router) {
			L->router_scale_w.host_ptr = synth_alloc_f32(cfg->dim);
		}
		L->ffn_pre_norm_2_w.host_ptr  = synth_alloc_f32(cfg->dim);
		L->ffn_post_norm_1_w.host_ptr = synth_alloc_f32(cfg->dim);
		L->ffn_post_norm_2_w.host_ptr = synth_alloc_f32(cfg->dim);

		int expert_block = cfg->dim * cfg->moe_intermediate;
		L->experts		 = xcalloc((size_t)E, sizeof(struct expert_desc));
		float *exp_gate	 = xmalloc((size_t)E * expert_block * sizeof(float));
		float *exp_up	 = xmalloc((size_t)E * expert_block * sizeof(float));
		float *exp_down	 = xmalloc((size_t)E * expert_block * sizeof(float));
		for (int e = 0; e < E; e++) {
			float *base					= exp_gate + ((size_t)e * expert_block);
			L->experts[e].gate_w		= base;
			L->experts[e].gate_type		= GGML_TYPE_F32;
			L->experts[e].gate_scale	= 1.0f;
			base						= exp_up + ((size_t)e * expert_block);
			L->experts[e].up_w			= base;
			L->experts[e].up_type		= GGML_TYPE_F32;
			L->experts[e].up_scale		= 1.0f;
			base						= exp_down + ((size_t)e * expert_block);
			L->experts[e].down_w		= base;
			L->experts[e].down_type		= GGML_TYPE_F32;
			L->experts[e].down_scale	= 1.0f;
			L->experts[e].gate_up_fused = 0;
		}

		if (cfg->n_shared_experts > 0) {
			int sh_inter			 = cfg->moe_intermediate * cfg->n_shared_experts;
			int sh_block			 = cfg->dim * sh_inter;
			L->shexp_gate_w.host_ptr = synth_alloc_f32(sh_block);
			L->shexp_gate_w.type	 = GGML_TYPE_F32;
			L->shexp_up_w.host_ptr	 = synth_alloc_f32(sh_block);
			L->shexp_up_w.type		 = GGML_TYPE_F32;
			L->shexp_down_w.host_ptr = synth_alloc_f32(sh_inter * cfg->dim);
			L->shexp_down_w.type	 = GGML_TYPE_F32;
		}
	}

	if (m->arch_info->is_mla && m->mla.q_lora > 0) {
		int q_lora	  = m->mla.q_lora;
		int kv_lora	  = m->mla.kv_lora;
		int qk_rope	  = m->mla.qk_rope;
		int kv_a_rows = kv_lora + qk_rope;
		int q_b_rows  = m->n_heads * m->mla.qk_head;
		int v_head	  = m->mla.v_head;
		int wo_in	  = m->n_heads * v_head;
		int dim		  = m->dim;

		L->q_a_w.host_ptr		= synth_alloc_f32(dim * q_lora);
		L->kv_a_w.host_ptr		= synth_alloc_f32(dim * kv_a_rows);
		L->q_a_norm_w.host_ptr	= synth_alloc_f32(q_lora);
		L->q_b_w.host_ptr		= synth_alloc_f32(q_lora * q_b_rows);
		L->kv_a_norm_w.host_ptr = synth_alloc_f32(kv_lora);
		L->k_b_w.host_ptr		= synth_alloc_f32(kv_lora * wo_in);
		L->v_b_w.host_ptr		= synth_alloc_f32(kv_lora * wo_in);
	}
}

static void synth_fill_layer_rng(layer_weights *L, int li, uint64_t *seed) {
	const synth_cfg *cfg	= g_synth_build_cfg;
	int				 q_out	= cfg->n_heads * cfg->head_dim;
	int				 kv_out = cfg->n_kv_heads * cfg->head_dim;

	synth_fill_f32((float *)L->attn_norm_w.host_ptr, cfg->dim, 1.0f);
	synth_fill_q4_k((void *)L->wq.host_ptr, q_out, cfg->dim);
	synth_fill_q4_k((void *)L->wk.host_ptr, kv_out, cfg->dim);
	synth_fill_q4_k((void *)L->wv.host_ptr, kv_out, cfg->dim);
	synth_fill_q4_k((void *)L->wo.host_ptr, cfg->dim, q_out);
	synth_fill_f32((float *)L->ffn_norm_w.host_ptr, cfg->dim, 1.0f);
	synth_fill_q4_k((void *)L->gate_w.host_ptr, cfg->intermediate, cfg->dim);
	synth_fill_q4_k((void *)L->up_w.host_ptr, cfg->intermediate, cfg->dim);
	synth_fill_q4_k((void *)L->down_w.host_ptr, cfg->dim, cfg->intermediate);

	if (L->attn_q_norm_w.host_ptr) {
		int hd = cfg->head_dim;
		synth_fill_f32((float *)L->attn_q_norm_w.host_ptr, hd, 1.0f);
		synth_fill_f32((float *)L->attn_k_norm_w.host_ptr, hd, 1.0f);
	}
	if (L->post_attn_norm_w.host_ptr) {
		synth_fill_f32((float *)L->post_attn_norm_w.host_ptr, cfg->dim, 1.0f);
	}
	if (L->post_ffn_norm_w.host_ptr) {
		synth_fill_f32((float *)L->post_ffn_norm_w.host_ptr, cfg->dim, 1.0f);
	}

	if (L->is_moe_layer) {
		int E = cfg->n_experts;
		synth_fill_f32((float *)L->router_w.host_ptr, cfg->dim * E, 0.0f);
		if (L->router_scale_w.host_ptr) {
			synth_fill_f32((float *)L->router_scale_w.host_ptr, cfg->dim, 1.0f);
		}
		synth_fill_f32((float *)L->ffn_pre_norm_2_w.host_ptr, cfg->dim, 1.0f);
		synth_fill_f32((float *)L->ffn_post_norm_1_w.host_ptr, cfg->dim, 1.0f);
		synth_fill_f32((float *)L->ffn_post_norm_2_w.host_ptr, cfg->dim, 1.0f);

		int expert_block = cfg->dim * cfg->moe_intermediate;
		seed_test_rng(*seed + ((uint64_t)li * 1000));
		fill_random_f32((float *)L->experts[0].gate_w, E * expert_block, 1.0f);
		fill_random_f32((float *)L->experts[0].up_w, E * expert_block, 1.0f);
		fill_random_f32((float *)L->experts[0].down_w, E * expert_block, 1.0f);

		if (L->shexp_gate_w.host_ptr) {
			int sh_inter = cfg->moe_intermediate * cfg->n_shared_experts;
			int sh_block = cfg->dim * sh_inter;
			synth_fill_f32((float *)L->shexp_gate_w.host_ptr, sh_block, 0.0f);
			synth_fill_f32((float *)L->shexp_up_w.host_ptr, sh_block, 0.0f);
			synth_fill_f32((float *)L->shexp_down_w.host_ptr, sh_inter * cfg->dim, 0.0f);
		}
	}

	if (L->q_a_w.host_ptr) {
		int q_lora	  = cfg->q_lora;
		int kv_lora	  = cfg->kv_lora;
		int qk_rope	  = cfg->qk_rope;
		int kv_a_rows = kv_lora + qk_rope;
		int q_b_rows  = cfg->n_heads * cfg->qk_head;
		int v_head	  = cfg->v_head;
		int wo_in	  = cfg->n_heads * v_head;
		int dim		  = cfg->dim;

		synth_fill_f32((float *)L->q_a_w.host_ptr, dim * q_lora, 0.0f);
		synth_fill_f32((float *)L->kv_a_w.host_ptr, dim * kv_a_rows, 0.0f);
		synth_fill_f32((float *)L->q_a_norm_w.host_ptr, q_lora, 1.0f);
		synth_fill_f32((float *)L->q_b_w.host_ptr, q_lora * q_b_rows, 0.0f);
		synth_fill_f32((float *)L->kv_a_norm_w.host_ptr, kv_lora, 1.0f);
		synth_fill_f32((float *)L->k_b_w.host_ptr, kv_lora * wo_in, 0.0f);
		synth_fill_f32((float *)L->v_b_w.host_ptr, kv_lora * wo_in, 0.0f);
	}
}

static void synth_upload_layer(model *m, int li) {
	backend			*a		= m->backend;
	layer_weights	*L		= &m->layers[li];
	const synth_cfg *cfg	= g_synth_build_cfg;
	int				 q_out	= cfg->n_heads * cfg->head_dim;
	int				 kv_out = cfg->n_kv_heads * cfg->head_dim;
	const arch_info *ai		= m->arch_info;
	char			 tname[64];

	snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", li);
	tensor_desc d_an = {.host_data = L->attn_norm_w.host_ptr,
						.type	   = GGML_TYPE_F32,
						.n_dims	   = 1,
						.dims	   = {cfg->dim},
						.name	   = tname};
	a->buffer_alloc_weight(a, &d_an, &L->attn_norm_w.buf);

	snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", li);
	tensor_desc dq = {.host_data = L->wq.host_ptr,
					  .type		 = GGML_TYPE_Q4_K,
					  .n_dims	 = 2,
					  .dims		 = {cfg->dim, q_out},
					  .name		 = tname};
	a->buffer_alloc_weight(a, &dq, &L->wq.buf);
	L->wq.type = GGML_TYPE_Q4_K;

	snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", li);
	tensor_desc dk = {.host_data = L->wk.host_ptr,
					  .type		 = GGML_TYPE_Q4_K,
					  .n_dims	 = 2,
					  .dims		 = {cfg->dim, kv_out},
					  .name		 = tname};
	a->buffer_alloc_weight(a, &dk, &L->wk.buf);
	L->wk.type = GGML_TYPE_Q4_K;

	snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", li);
	tensor_desc dv = {.host_data = L->wv.host_ptr,
					  .type		 = GGML_TYPE_Q4_K,
					  .n_dims	 = 2,
					  .dims		 = {cfg->dim, kv_out},
					  .name		 = tname};
	a->buffer_alloc_weight(a, &dv, &L->wv.buf);
	L->wv.type = GGML_TYPE_Q4_K;

	snprintf(tname, sizeof(tname), "blk.%d.attn_o.weight", li);
	tensor_desc dwo = {.host_data = L->wo.host_ptr,
					   .type	  = GGML_TYPE_Q4_K,
					   .n_dims	  = 2,
					   .dims	  = {q_out, cfg->dim},
					   .name	  = tname};
	a->buffer_alloc_weight(a, &dwo, &L->wo.buf);
	L->wo.type = GGML_TYPE_Q4_K;

	snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", li);
	tensor_desc dfn = {.host_data = L->ffn_norm_w.host_ptr,
					   .type	  = GGML_TYPE_F32,
					   .n_dims	  = 1,
					   .dims	  = {cfg->dim},
					   .name	  = tname};
	a->buffer_alloc_weight(a, &dfn, &L->ffn_norm_w.buf);

	snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", li);
	tensor_desc dg = {.host_data = L->gate_w.host_ptr,
					  .type		 = GGML_TYPE_Q4_K,
					  .n_dims	 = 2,
					  .dims		 = {cfg->dim, cfg->intermediate},
					  .name		 = tname};
	a->buffer_alloc_weight(a, &dg, &L->gate_w.buf);
	L->gate_w.type = GGML_TYPE_Q4_K;

	snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", li);
	tensor_desc du = {.host_data = L->up_w.host_ptr,
					  .type		 = GGML_TYPE_Q4_K,
					  .n_dims	 = 2,
					  .dims		 = {cfg->dim, cfg->intermediate},
					  .name		 = tname};
	a->buffer_alloc_weight(a, &du, &L->up_w.buf);
	L->up_w.type = GGML_TYPE_Q4_K;

	snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", li);
	tensor_desc dd = {.host_data = L->down_w.host_ptr,
					  .type		 = GGML_TYPE_Q4_K,
					  .n_dims	 = 2,
					  .dims		 = {cfg->intermediate, cfg->dim},
					  .name		 = tname};
	a->buffer_alloc_weight(a, &dd, &L->down_w.buf);
	L->down_w.type = GGML_TYPE_Q4_K;

	if (ai->has_qk_norm) {
		int hd = cfg->head_dim;
		snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", li);
		tensor_desc d = {.host_data = L->attn_q_norm_w.host_ptr,
						 .type		= GGML_TYPE_F32,
						 .n_dims	= 1,
						 .dims		= {hd},
						 .name		= tname};
		a->buffer_alloc_weight(a, &d, &L->attn_q_norm_w.buf);

		snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", li);
		tensor_desc d2 = {.host_data = L->attn_k_norm_w.host_ptr,
						  .type		 = GGML_TYPE_F32,
						  .n_dims	 = 1,
						  .dims		 = {hd},
						  .name		 = tname};
		a->buffer_alloc_weight(a, &d2, &L->attn_k_norm_w.buf);
	}
	if (ai->has_attn_post_norm) {
		snprintf(tname, sizeof(tname), "blk.%d.post_attn_norm.weight", li);
		tensor_desc d = {.host_data = L->post_attn_norm_w.host_ptr,
						 .type		= GGML_TYPE_F32,
						 .n_dims	= 1,
						 .dims		= {cfg->dim},
						 .name		= tname};
		a->buffer_alloc_weight(a, &d, &L->post_attn_norm_w.buf);
	}
	if (ai->has_ffn_post_norm) {
		snprintf(tname, sizeof(tname), "blk.%d.post_ffn_norm.weight", li);
		tensor_desc d = {.host_data = L->post_ffn_norm_w.host_ptr,
						 .type		= GGML_TYPE_F32,
						 .n_dims	= 1,
						 .dims		= {cfg->dim},
						 .name		= tname};
		a->buffer_alloc_weight(a, &d, &L->post_ffn_norm_w.buf);
	}

	if (L->is_moe_layer) {
		int E = cfg->n_experts;

		snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.weight", li);
		tensor_desc dr = {.host_data = L->router_w.host_ptr,
						  .type		 = GGML_TYPE_F32,
						  .n_dims	 = 2,
						  .dims		 = {cfg->dim, E},
						  .name		 = tname};
		a->buffer_alloc_weight(a, &dr, &L->router_w.buf);

		if (ai->uses_moe_softmax_router) {
			snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.scale", li);
			tensor_desc drs = {.host_data = L->router_scale_w.host_ptr,
							   .type	  = GGML_TYPE_F32,
							   .n_dims	  = 1,
							   .dims	  = {cfg->dim},
							   .name	  = tname};
			a->buffer_alloc_weight(a, &drs, &L->router_scale_w.buf);
		}

		snprintf(tname, sizeof(tname), "blk.%d.ffn_pre_norm_2.weight", li);
		tensor_desc dpn2 = {.host_data = L->ffn_pre_norm_2_w.host_ptr,
							.type	   = GGML_TYPE_F32,
							.n_dims	   = 1,
							.dims	   = {cfg->dim},
							.name	   = tname};
		a->buffer_alloc_weight(a, &dpn2, &L->ffn_pre_norm_2_w.buf);

		snprintf(tname, sizeof(tname), "blk.%d.ffn_post_norm_1.weight", li);
		tensor_desc dpn1 = {.host_data = L->ffn_post_norm_1_w.host_ptr,
							.type	   = GGML_TYPE_F32,
							.n_dims	   = 1,
							.dims	   = {cfg->dim},
							.name	   = tname};
		a->buffer_alloc_weight(a, &dpn1, &L->ffn_post_norm_1_w.buf);

		snprintf(tname, sizeof(tname), "blk.%d.ffn_post_norm_2.weight", li);
		tensor_desc dpn22 = {.host_data = L->ffn_post_norm_2_w.host_ptr,
							 .type		= GGML_TYPE_F32,
							 .n_dims	= 1,
							 .dims		= {cfg->dim},
							 .name		= tname};
		a->buffer_alloc_weight(a, &dpn22, &L->ffn_post_norm_2_w.buf);

		if (L->shexp_gate_w.host_ptr) {
			int sh_inter = cfg->moe_intermediate * cfg->n_shared_experts;

			snprintf(tname, sizeof(tname), "blk.%d.shexp_gate.weight", li);
			tensor_desc d_sg = {.host_data = L->shexp_gate_w.host_ptr,
								.type	   = GGML_TYPE_F32,
								.n_dims	   = 2,
								.dims	   = {cfg->dim, sh_inter},
								.name	   = tname};
			a->buffer_alloc_weight(a, &d_sg, &L->shexp_gate_w.buf);

			snprintf(tname, sizeof(tname), "blk.%d.shexp_up.weight", li);
			tensor_desc d_su = {.host_data = L->shexp_up_w.host_ptr,
								.type	   = GGML_TYPE_F32,
								.n_dims	   = 2,
								.dims	   = {cfg->dim, sh_inter},
								.name	   = tname};
			a->buffer_alloc_weight(a, &d_su, &L->shexp_up_w.buf);

			snprintf(tname, sizeof(tname), "blk.%d.shexp_down.weight", li);
			tensor_desc d_sd = {.host_data = L->shexp_down_w.host_ptr,
								.type	   = GGML_TYPE_F32,
								.n_dims	   = 2,
								.dims	   = {sh_inter, cfg->dim},
								.name	   = tname};
			a->buffer_alloc_weight(a, &d_sd, &L->shexp_down_w.buf);

			L->shexp_up_w.buf.offset   = 0;
			L->shexp_down_w.buf.offset = 0;
		}
	}

	if (m->arch_info->is_mla && m->mla.q_lora > 0) {
		int q_lora	  = m->mla.q_lora;
		int kv_lora	  = m->mla.kv_lora;
		int qk_rope	  = m->mla.qk_rope;
		int kv_a_rows = kv_lora + qk_rope;
		int q_b_rows  = m->n_heads * m->mla.qk_head;
		int v_head	  = m->mla.v_head;
		int wo_in	  = m->n_heads * v_head;
		int dim		  = m->dim;

		snprintf(tname, sizeof(tname), "blk.%d.q_a.weight", li);
		tensor_desc dqa = {.host_data = L->q_a_w.host_ptr,
						   .type	  = GGML_TYPE_F32,
						   .n_dims	  = 2,
						   .dims	  = {dim, q_lora},
						   .name	  = tname};
		a->buffer_alloc_weight(a, &dqa, &L->q_a_w.buf);

		snprintf(tname, sizeof(tname), "blk.%d.kv_a.weight", li);
		tensor_desc dkva = {.host_data = L->kv_a_w.host_ptr,
							.type	   = GGML_TYPE_F32,
							.n_dims	   = 2,
							.dims	   = {dim, kv_a_rows},
							.name	   = tname};
		a->buffer_alloc_weight(a, &dkva, &L->kv_a_w.buf);

		snprintf(tname, sizeof(tname), "blk.%d.q_a_norm.weight", li);
		tensor_desc dqan = {.host_data = L->q_a_norm_w.host_ptr,
							.type	   = GGML_TYPE_F32,
							.n_dims	   = 1,
							.dims	   = {q_lora},
							.name	   = tname};
		a->buffer_alloc_weight(a, &dqan, &L->q_a_norm_w.buf);

		snprintf(tname, sizeof(tname), "blk.%d.q_b.weight", li);
		tensor_desc dqb = {.host_data = L->q_b_w.host_ptr,
						   .type	  = GGML_TYPE_F32,
						   .n_dims	  = 2,
						   .dims	  = {q_lora, q_b_rows},
						   .name	  = tname};
		a->buffer_alloc_weight(a, &dqb, &L->q_b_w.buf);

		snprintf(tname, sizeof(tname), "blk.%d.kv_a_norm.weight", li);
		tensor_desc dkvan = {.host_data = L->kv_a_norm_w.host_ptr,
							 .type		= GGML_TYPE_F32,
							 .n_dims	= 1,
							 .dims		= {kv_lora},
							 .name		= tname};
		a->buffer_alloc_weight(a, &dkvan, &L->kv_a_norm_w.buf);

		snprintf(tname, sizeof(tname), "blk.%d.k_b.weight", li);
		tensor_desc dkb = {.host_data = L->k_b_w.host_ptr,
						   .type	  = GGML_TYPE_F32,
						   .n_dims	  = 2,
						   .dims	  = {kv_lora, wo_in},
						   .name	  = tname};
		a->buffer_alloc_weight(a, &dkb, &L->k_b_w.buf);

		snprintf(tname, sizeof(tname), "blk.%d.v_b.weight", li);
		tensor_desc dvb = {.host_data = L->v_b_w.host_ptr,
						   .type	  = GGML_TYPE_F32,
						   .n_dims	  = 2,
						   .dims	  = {kv_lora, wo_in},
						   .name	  = tname};
		a->buffer_alloc_weight(a, &dvb, &L->v_b_w.buf);
	}
}

static synth_model *synth_model_build(backend *a, model_arch arch, const synth_cfg *cfg,
									  int n_ctx) {
	synth_model *sm = xcalloc(1, sizeof(synth_model));
	model		*m	= &sm->m;
	sm->n_layers	= cfg->n_layers;
	sm->n_ctx		= n_ctx;

	m->backend		  = a;
	m->arch			  = arch;
	m->arch_info	  = arch_lookup(arch);
	m->n_layers		  = cfg->n_layers;
	m->dim			  = cfg->dim;
	m->n_heads		  = cfg->n_heads;
	m->n_kv_heads	  = cfg->n_kv_heads;
	m->head_dim		  = cfg->head_dim;
	m->intermediate	  = cfg->intermediate;
	m->vocab_size	  = cfg->vocab;
	m->n_ctx		  = n_ctx;
	m->rope_dim		  = cfg->n_heads * cfg->head_dim / 2;
	m->rope_theta	  = 10000.0f;
	m->norm_eps		  = 1e-5f;
	m->tie_embeddings = 0;
	m->owns_backend	  = 0;

	m->moe.n_experts		 = cfg->n_experts;
	m->moe.n_experts_used	 = cfg->n_experts_used;
	m->moe.n_shared_experts	 = cfg->n_shared_experts;
	m->moe.moe_intermediate	 = cfg->moe_intermediate;
	m->moe.n_group			 = cfg->n_group;
	m->moe.topk_group		 = cfg->topk_group;
	m->moe.routed_scale		 = cfg->routed_scale;
	m->moe.norm_topk_prob	 = cfg->norm_topk_prob;
	m->moe.first_dense_layer = cfg->first_dense_layer;

	m->mla.q_lora  = cfg->q_lora;
	m->mla.kv_lora = cfg->kv_lora;
	m->mla.qk_head = cfg->qk_head;
	m->mla.qk_rope = cfg->qk_rope;
	m->mla.qk_nope = cfg->qk_nope;
	m->mla.v_head  = cfg->v_head;

	if (m->arch_info->has_variable_layer_dims) {
		int hd_swa						   = cfg->head_dim;
		int hd_global					   = cfg->head_dim * 2;
		m->layer_dims.head_dim_swa		   = hd_swa;
		m->layer_dims.head_dim_global	   = hd_global;
		m->layer_dims.rope_dim_swa		   = cfg->n_heads * hd_swa / 2;
		m->layer_dims.rope_dim_global	   = cfg->n_heads * hd_global / 2;
		m->layer_dims.rope_theta_swa	   = 10000.0f;
		m->layer_dims.rope_theta_global	   = 1000000.0f;
		m->head_dim						   = hd_global;
		m->rope_dim						   = cfg->n_heads * hd_global / 2;
		m->sliding_window				   = m->arch_info->sliding_window_period > 0 ? hd_swa : 0;
		m->layer_dims.is_global_layer	   = xcalloc((size_t)cfg->n_layers, sizeof(uint8_t));
		m->layer_dims.ffn_lengths		   = xcalloc((size_t)cfg->n_layers, sizeof(int));
		m->layer_dims.n_kv_heads_per_layer = xcalloc((size_t)cfg->n_layers, sizeof(int));
		for (int li = 0; li < cfg->n_layers; li++) {
			int period							   = m->arch_info->sliding_window_period;
			m->layer_dims.is_global_layer[li]	   = period > 0 && (li % period) == (period - 1);
			m->layer_dims.ffn_lengths[li]		   = cfg->intermediate;
			m->layer_dims.n_kv_heads_per_layer[li] = cfg->n_kv_heads;
		}

		m->layer_dims.n_layer_kv_from_start =
			cfg->n_layer_kv_from_start > 0 ? cfg->n_layer_kv_from_start : cfg->n_layers;
		m->layer_dims.kv_layer_swa	  = -1;
		m->layer_dims.kv_layer_global = -1;
		for (int li = 0; li < m->layer_dims.n_layer_kv_from_start && li < cfg->n_layers; li++) {
			if (m->layer_dims.is_global_layer[li])
				m->layer_dims.kv_layer_global = li;
			else
				m->layer_dims.kv_layer_swa = li;
		}
	}

	sm->tok_embd_raw  = alloc_q4_k_weights(cfg->vocab, cfg->dim);
	tensor_desc d_tok = {
		.host_data = sm->tok_embd_raw,
		.type	   = GGML_TYPE_Q4_K,
		.n_dims	   = 2,
		.dims	   = {cfg->dim, cfg->vocab},
		.name	   = "token_embd.weight",
	};
	a->buffer_alloc_weight(a, &d_tok, &m->tok_embd.buf);
	m->tok_embd.type = GGML_TYPE_Q4_K;

	sm->output_norm_raw = alloc_f32_weights(cfg->dim, 1.0f);
	tensor_desc d_onorm = {
		.host_data = sm->output_norm_raw,
		.type	   = GGML_TYPE_F32,
		.n_dims	   = 1,
		.dims	   = {cfg->dim},
		.name	   = "output_norm.weight",
	};
	a->buffer_alloc_weight(a, &d_onorm, &m->output_norm_w.buf);

	sm->output_w_raw  = alloc_q4_k_weights(cfg->vocab, cfg->dim);
	tensor_desc d_out = {
		.host_data = sm->output_w_raw,
		.type	   = GGML_TYPE_Q4_K,
		.n_dims	   = 2,
		.dims	   = {cfg->dim, cfg->vocab},
		.name	   = "output.weight",
	};
	a->buffer_alloc_weight(a, &d_out, &m->output_w.buf);
	m->output_w.type = GGML_TYPE_Q4_K;

	m->layers = xcalloc((size_t)cfg->n_layers, sizeof(layer_weights));

	g_synth_build_cfg	= cfg;
	uint64_t layer_seed = 0xFEEDULL;
	for (int li = 0; li < cfg->n_layers; li++) {
		layer_weights *L = &m->layers[li];
		synth_alloc_layer(L, li, cfg, m);
		synth_fill_layer_rng(L, li, &layer_seed);
		synth_upload_layer(m, li);
	}
	g_synth_build_cfg = NULL;

	model_build_weight_refs(&sm->m);
	m->recipe			  = recipe_build(m);
	m->moe_stream_enabled = 0;
	return sm;
}

static void synth_free_layer(layer_weights *L, int li) {
	(void)li;
	backend *a = L->attn_norm_w.buf.owner;
	if (a) {
		a->buffer_free(a, &L->attn_norm_w.buf);
		a->buffer_free(a, &L->wq.buf);
		a->buffer_free(a, &L->wk.buf);
		a->buffer_free(a, &L->wv.buf);
		a->buffer_free(a, &L->wo.buf);
		a->buffer_free(a, &L->ffn_norm_w.buf);
		a->buffer_free(a, &L->gate_w.buf);
		a->buffer_free(a, &L->up_w.buf);
		a->buffer_free(a, &L->down_w.buf);
		a->buffer_free(a, &L->attn_q_norm_w.buf);
		a->buffer_free(a, &L->attn_k_norm_w.buf);
		a->buffer_free(a, &L->post_attn_norm_w.buf);
		a->buffer_free(a, &L->post_ffn_norm_w.buf);
		a->buffer_free(a, &L->router_w.buf);
		a->buffer_free(a, &L->router_scale_w.buf);
		a->buffer_free(a, &L->ffn_pre_norm_2_w.buf);
		a->buffer_free(a, &L->ffn_post_norm_1_w.buf);
		a->buffer_free(a, &L->ffn_post_norm_2_w.buf);
		a->buffer_free(a, &L->shexp_gate_w.buf);
		a->buffer_free(a, &L->shexp_up_w.buf);
		a->buffer_free(a, &L->shexp_down_w.buf);
		if (L->q_a_w.host_ptr) {
			a->buffer_free(a, &L->q_a_w.buf);
			a->buffer_free(a, &L->kv_a_w.buf);
			a->buffer_free(a, &L->q_a_norm_w.buf);
			a->buffer_free(a, &L->q_b_w.buf);
			a->buffer_free(a, &L->kv_a_norm_w.buf);
			a->buffer_free(a, &L->k_b_w.buf);
			a->buffer_free(a, &L->v_b_w.buf);
		}
	}

	free((void *)L->attn_norm_w.host_ptr);
	free((void *)L->wq.host_ptr);
	free((void *)L->wk.host_ptr);
	free((void *)L->wv.host_ptr);
	free((void *)L->wo.host_ptr);
	free((void *)L->ffn_norm_w.host_ptr);
	free((void *)L->gate_w.host_ptr);
	free((void *)L->up_w.host_ptr);
	free((void *)L->down_w.host_ptr);
	free((void *)L->attn_q_norm_w.host_ptr);
	free((void *)L->attn_k_norm_w.host_ptr);
	free((void *)L->post_attn_norm_w.host_ptr);
	free((void *)L->post_ffn_norm_w.host_ptr);
	free((void *)L->router_w.host_ptr);
	free((void *)L->router_scale_w.host_ptr);
	free((void *)L->ffn_pre_norm_2_w.host_ptr);
	free((void *)L->ffn_post_norm_1_w.host_ptr);
	free((void *)L->ffn_post_norm_2_w.host_ptr);
	free((void *)L->shexp_gate_w.host_ptr);
	free((void *)L->shexp_up_w.host_ptr);
	free((void *)L->shexp_down_w.host_ptr);

	if (L->experts) {
		free((void *)L->experts[0].gate_w);
		free((void *)L->experts[0].up_w);
		free((void *)L->experts[0].down_w);
		free(L->experts);
	}

	if (L->q_a_w.host_ptr) {
		free((void *)L->q_a_w.host_ptr);
		free((void *)L->kv_a_w.host_ptr);
		free((void *)L->q_a_norm_w.host_ptr);
		free((void *)L->q_b_w.host_ptr);
		free((void *)L->kv_a_norm_w.host_ptr);
		free((void *)L->k_b_w.host_ptr);
		free((void *)L->v_b_w.host_ptr);
	}
}

static void synth_model_free(synth_model *sm) {
	if (!sm)
		return;
	backend *a = sm->m.backend;
	a->buffer_free(a, &sm->m.tok_embd.buf);
	a->buffer_free(a, &sm->m.output_norm_w.buf);
	a->buffer_free(a, &sm->m.output_w.buf);
	for (int li = 0; li < sm->n_layers; li++)
		synth_free_layer(&sm->m.layers[li], li);
	free(sm->tok_embd_raw);
	free(sm->output_norm_raw);
	free(sm->output_w_raw);
	free(sm->m.layers);
	free(sm->m.wrefs_by_layer);
	sm->m.wrefs_by_layer = NULL;
	free(sm->m.layer_dims.is_global_layer);
	free(sm->m.layer_dims.ffn_lengths);
	free(sm->m.layer_dims.n_kv_heads_per_layer);
	if (sm->m.recipe)
		recipe_free(sm->m.recipe);
	if (sm->m.moe_cache)
		moe_stream_cache_free(sm->m.moe_cache);
	free(sm);
}

static void test_arch_pipeline(backend *cpu, backend *tgt, const synth_cfg *cfg, int n_prefill,
							   int flash_attn) {
	int n_ctx = n_prefill + 16;

	synth_model *sm_cpu = synth_model_build(cpu, ARCH_LLAMA, cfg, n_ctx);
	synth_model *sm_tgt = synth_model_build(tgt, ARCH_LLAMA, cfg, n_ctx);

	int32_t *prompt = xmalloc((size_t)n_prefill * sizeof(int32_t));
	seed_test_rng(0xBEEFULL);
	for (int i = 0; i < n_prefill; i++)
		prompt[i] = (int32_t)(next_u32() % (uint32_t)cfg->vocab);

	float *logits_cpu = xmalloc((size_t)cfg->vocab * sizeof(float));
	float *logits_tgt = xmalloc((size_t)cfg->vocab * sizeof(float));

	{
		kvcache kv;
		kvcache_init(&kv, &sm_cpu->m, n_ctx, KV_QUANT_F16);
		compute_scratch s;
		compute_scratch_init(&s);
		(void)compute_scratch_ensure(&s, &sm_cpu->m, n_ctx);
		for (int i = 0; i < n_prefill; i++) {
			float *out = (i == n_prefill - 1) ? logits_cpu : NULL;
			compute_forward(&sm_cpu->m, &kv, &s, prompt[i], i, flash_attn, out);
			kv.n_pos++;
		}
		if (cpu->synchronize)
			cpu->synchronize(cpu);
		compute_scratch_free(&s);
		kvcache_free(&kv);
	}
	{
		kvcache kv;
		kvcache_init(&kv, &sm_tgt->m, n_ctx, KV_QUANT_F16);
		compute_scratch s;
		compute_scratch_init(&s);
		(void)compute_scratch_ensure(&s, &sm_tgt->m, n_ctx);
		for (int i = 0; i < n_prefill; i++) {
			float *out = (i == n_prefill - 1) ? logits_tgt : NULL;
			compute_forward(&sm_tgt->m, &kv, &s, prompt[i], i, flash_attn, out);
			kv.n_pos++;
		}
		if (tgt->synchronize)
			tgt->synchronize(tgt);
		compute_scratch_free(&s);
		kvcache_free(&kv);
	}

	int32_t am_cpu = sampler_argmax(logits_cpu, cfg->vocab);
	int32_t am_tgt = sampler_argmax(logits_tgt, cfg->vocab);

	char label[160];
	char detail[320];
	snprintf(label, sizeof(label), "arch.pipeline %d-layer prefill(%d) flash=%d", cfg->n_layers,
			 n_prefill, flash_attn);
	verdict v =
		classify_output("loose", logits_cpu, logits_tgt, cfg->vocab, OK, detail, sizeof(detail));
	int dl = (int)strlen(detail);
	snprintf(detail + dl, sizeof(detail) - dl, " | argmax cpu=%d tgt=%d %s", am_cpu, am_tgt,
			 am_cpu == am_tgt ? "(agree)" : "(DIVERGE)");
	if (v != V_PASS && v != V_SKIP)
		compute_debug(logits_cpu, logits_tgt, cfg->vocab);
	record_result(OPFAM_ARCH_PIPELINE, label, v, detail);

	free(logits_cpu);
	free(logits_tgt);
	free(prompt);
	synth_model_free(sm_cpu);
	synth_model_free(sm_tgt);
}

static void test_arch_single_layer(backend *cpu, backend *tgt, const synth_cfg *cfg,
								   int flash_attn) {
	synth_cfg one = *cfg;
	one.n_layers  = 1;
	int n_ctx	  = 16;

	synth_model *sm_cpu = synth_model_build(cpu, ARCH_LLAMA, &one, n_ctx);
	synth_model *sm_tgt = synth_model_build(tgt, ARCH_LLAMA, &one, n_ctx);

	int32_t tok		   = 7;
	float  *logits_cpu = xmalloc((size_t)one.vocab * sizeof(float));
	float  *logits_tgt = xmalloc((size_t)one.vocab * sizeof(float));

	{
		kvcache kv;
		kvcache_init(&kv, &sm_cpu->m, n_ctx, KV_QUANT_F16);
		compute_scratch s;
		compute_scratch_init(&s);
		(void)compute_scratch_ensure(&s, &sm_cpu->m, n_ctx);
		compute_forward(&sm_cpu->m, &kv, &s, tok, 0, flash_attn, logits_cpu);
		if (cpu->synchronize)
			cpu->synchronize(cpu);
		compute_scratch_free(&s);
		kvcache_free(&kv);
	}
	{
		kvcache kv;
		kvcache_init(&kv, &sm_tgt->m, n_ctx, KV_QUANT_F16);
		compute_scratch s;
		compute_scratch_init(&s);
		(void)compute_scratch_ensure(&s, &sm_tgt->m, n_ctx);
		compute_forward(&sm_tgt->m, &kv, &s, tok, 0, flash_attn, logits_tgt);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
		compute_scratch_free(&s);
		kvcache_free(&kv);
	}

	char label[160];
	char detail[320];
	snprintf(label, sizeof(label), "arch.single_layer flash=%d", flash_attn);
	verdict v =
		classify_output("loose", logits_cpu, logits_tgt, one.vocab, OK, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(logits_cpu, logits_tgt, one.vocab);
	record_result(OPFAM_ARCH_LAYER, label, v, detail);

	free(logits_cpu);
	free(logits_tgt);
	synth_model_free(sm_cpu);
	synth_model_free(sm_tgt);
}

static void test_arch_decode_chain(backend *cpu, backend *tgt, const synth_cfg *cfg, int n_prefill,
								   int n_decode, int flash_attn) {
	int n_ctx = n_prefill + n_decode + 16;

	synth_model *sm_cpu = synth_model_build(cpu, ARCH_LLAMA, cfg, n_ctx);
	synth_model *sm_tgt = synth_model_build(tgt, ARCH_LLAMA, cfg, n_ctx);

	int32_t *prompt = xmalloc((size_t)n_prefill * sizeof(int32_t));
	seed_test_rng(0xDEC01ULL);
	for (int i = 0; i < n_prefill; i++)
		prompt[i] = (int32_t)(next_u32() % (uint32_t)cfg->vocab);

	float *logits_cpu = xmalloc((size_t)cfg->vocab * sizeof(float));
	float *logits_tgt = xmalloc((size_t)cfg->vocab * sizeof(float));

	kvcache kv_cpu;
	kvcache_init(&kv_cpu, &sm_cpu->m, n_ctx, KV_QUANT_F16);
	compute_scratch s_cpu;
	compute_scratch_init(&s_cpu);
	(void)compute_scratch_ensure(&s_cpu, &sm_cpu->m, n_ctx);
	for (int i = 0; i < n_prefill; i++) {
		float *out = (i == n_prefill - 1) ? logits_cpu : NULL;
		compute_forward(&sm_cpu->m, &kv_cpu, &s_cpu, prompt[i], i, flash_attn, out);
		kv_cpu.n_pos++;
	}
	if (cpu->synchronize)
		cpu->synchronize(cpu);

	kvcache kv_tgt;
	kvcache_init(&kv_tgt, &sm_tgt->m, n_ctx, KV_QUANT_F16);
	compute_scratch s_tgt;
	compute_scratch_init(&s_tgt);
	(void)compute_scratch_ensure(&s_tgt, &sm_tgt->m, n_ctx);
	for (int i = 0; i < n_prefill; i++) {
		float *out = (i == n_prefill - 1) ? logits_tgt : NULL;
		compute_forward(&sm_tgt->m, &kv_tgt, &s_tgt, prompt[i], i, flash_attn, out);
		kv_tgt.n_pos++;
	}
	if (tgt->synchronize)
		tgt->synchronize(tgt);

	int	  n_agree		  = 0;
	float max_abs_overall = 0;
	for (int step = 0; step < n_decode; step++) {
		int		pos = n_prefill + step;
		int32_t tok = (int32_t)(next_u32() % (uint32_t)cfg->vocab);
		compute_forward(&sm_cpu->m, &kv_cpu, &s_cpu, tok, pos, flash_attn, logits_cpu);
		if (cpu->synchronize)
			cpu->synchronize(cpu);
		compute_forward(&sm_tgt->m, &kv_tgt, &s_tgt, tok, pos, flash_attn, logits_tgt);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
		int32_t am_cpu = sampler_argmax(logits_cpu, cfg->vocab);
		int32_t am_tgt = sampler_argmax(logits_tgt, cfg->vocab);
		if (am_cpu == am_tgt)
			n_agree++;

		float step_max = max_abs_diff_at(logits_cpu, logits_tgt, cfg->vocab, NULL);
		if (step_max > max_abs_overall)
			max_abs_overall = step_max;
	}

	char label[160];
	char detail[320];
	snprintf(label, sizeof(label), "arch.decode_chain %d-layer prefill=%d decode=%d flash=%d",
			 cfg->n_layers, n_prefill, n_decode, flash_attn);
	verdict v =
		classify_output("loose", logits_cpu, logits_tgt, cfg->vocab, OK, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(logits_cpu, logits_tgt, cfg->vocab);
	int dl = (int)strlen(detail);
	snprintf(detail + dl, sizeof(detail) - dl, " | argmax agree %d/%d  max_abs_overall=%.4e",
			 n_agree, n_decode, max_abs_overall);
	record_result(OPFAM_ARCH_DECODE, label, v, detail);

	compute_scratch_free(&s_cpu);
	compute_scratch_free(&s_tgt);
	kvcache_free(&kv_cpu);
	kvcache_free(&kv_tgt);
	free(logits_cpu);
	free(logits_tgt);
	free(prompt);
	synth_model_free(sm_cpu);
	synth_model_free(sm_tgt);
}

static void test_arch_batch_vs_single(backend *cpu, model_arch arch, const synth_cfg *cfg,
									  int n_prefill, int flash_attn) {
	const char *arch_name = arch_lookup(arch)->gguf_name;
	int			n_ctx	  = n_prefill + 16;

	synth_model *sm_batch  = synth_model_build(cpu, arch, cfg, n_ctx);
	synth_model *sm_single = synth_model_build(cpu, arch, cfg, n_ctx);

	if (arch == ARCH_GEMMA4 || arch == ARCH_GEMMA4_MOE)
		sm_batch->m.batchable = -1;
	if (arch == ARCH_GLM_DSA)
		sm_batch->m.batchable = -1;

	int32_t *prompt = xmalloc((size_t)n_prefill * sizeof(int32_t));
	seed_test_rng(0xBA7C9B9ULL + (uint64_t)arch);
	for (int i = 0; i < n_prefill; i++)
		prompt[i] = (int32_t)(next_u32() % (uint32_t)cfg->vocab);

	float *logits_batch	 = xmalloc((size_t)cfg->vocab * sizeof(float));
	float *logits_single = xmalloc((size_t)cfg->vocab * sizeof(float));

	kvcache kv_batch, kv_single;
	kvcache_init(&kv_batch, &sm_batch->m, n_ctx, KV_QUANT_F16);
	kvcache_init(&kv_single, &sm_single->m, n_ctx, KV_QUANT_F16);
	compute_scratch s_batch, s_single;
	compute_scratch_init(&s_batch);
	compute_scratch_init(&s_single);
	(void)compute_scratch_ensure(&s_batch, &sm_batch->m, n_ctx);
	(void)compute_scratch_ensure(&s_single, &sm_single->m, n_ctx);

	status_code st_batch = compute_forward_batch(&sm_batch->m, &kv_batch, &s_batch, prompt,
												 n_prefill, 0, flash_attn, logits_batch);
	kv_batch.n_pos += n_prefill;

	status_code st_single = OK;
	for (int i = 0; i < n_prefill; i++) {
		st_single = compute_forward(&sm_single->m, &kv_single, &s_single, prompt[i], 0 + i,
									flash_attn, (i == n_prefill - 1) ? logits_single : NULL);
		kv_single.n_pos++;
		if (st_single != OK)
			break;
	}

	char label[160];
	char detail[320];
	if (st_batch != OK || st_single != OK) {
		snprintf(label, sizeof(label), "batch_vs_single[%s] status", arch_name);
		snprintf(detail, sizeof(detail), "batch_st=%d single_st=%d", st_batch, st_single);
		fprintf(stderr, "  [FAIL] batch_vs_single[%s]: %s\n", arch_name, detail);
		record_result(OPFAM_ARCH_GENERATE, label, V_FAIL, detail);
	} else {
		int nf_batch  = count_nonfinite(logits_batch, cfg->vocab);
		int nf_single = count_nonfinite(logits_single, cfg->vocab);
		if (nf_batch > 0 || nf_single > 0) {
			snprintf(label, sizeof(label), "batch_vs_single[%s] nonfinite", arch_name);
			snprintf(detail, sizeof(detail), "batch_nf=%d single_nf=%d", nf_batch, nf_single);
			record_result(OPFAM_ARCH_GENERATE, label, V_FAIL, detail);
		} else {
			float max_diff = 0;
			int	  max_idx  = 0;
			for (int i = 0; i < cfg->vocab; i++) {
				float d = fabsf(logits_batch[i] - logits_single[i]);
				if (d > max_diff) {
					max_diff = d;
					max_idx	 = i;
				}
			}
			snprintf(label, sizeof(label), "batch_vs_single[%s] logits", arch_name);
			if (max_diff > 0.1f) {
				snprintf(detail, sizeof(detail), "max_diff=%.4f at idx=%d (batch=%.4f single=%.4f)",
						 max_diff, max_idx, logits_batch[max_idx], logits_single[max_idx]);
				record_result(OPFAM_ARCH_GENERATE, label, V_FAIL, detail);
			} else {
				snprintf(detail, sizeof(detail), "max_diff=%.6f", max_diff);
				record_result(OPFAM_ARCH_GENERATE, label, V_PASS, detail);
			}
		}
	}

	compute_scratch_free(&s_batch);
	compute_scratch_free(&s_single);
	kvcache_free(&kv_batch);
	kvcache_free(&kv_single);
	free(logits_batch);
	free(logits_single);
	free(prompt);
	synth_model_free(sm_batch);
	synth_model_free(sm_single);
}

static void test_arch_error_compounding(backend *cpu, backend *tgt, const synth_cfg *base_cfg,
										int n_prefill, int flash_attn) {
	int layer_counts[] = {1, 2, 4, 8};
	int n_configs	   = (int)(sizeof(layer_counts) / sizeof(layer_counts[0]));
	int header_shown   = 0;

	for (int ci = 0; ci < n_configs; ci++) {
		int		  n_layers = layer_counts[ci];
		synth_cfg cfg	   = *base_cfg;
		cfg.n_layers	   = n_layers;
		int n_ctx		   = n_prefill + 16;

		synth_model *sm_cpu = synth_model_build(cpu, ARCH_LLAMA, &cfg, n_ctx);
		synth_model *sm_tgt = synth_model_build(tgt, ARCH_LLAMA, &cfg, n_ctx);

		int32_t *prompt = xmalloc((size_t)n_prefill * sizeof(int32_t));
		seed_test_rng(0xC0B01ULL);
		for (int i = 0; i < n_prefill; i++)
			prompt[i] = (int32_t)(next_u32() % (uint32_t)cfg.vocab);

		float *logits_cpu = xmalloc((size_t)cfg.vocab * sizeof(float));
		float *logits_tgt = xmalloc((size_t)cfg.vocab * sizeof(float));

		{
			kvcache kv;
			kvcache_init(&kv, &sm_cpu->m, n_ctx, KV_QUANT_F16);
			compute_scratch s;
			compute_scratch_init(&s);
			(void)compute_scratch_ensure(&s, &sm_cpu->m, n_ctx);
			for (int i = 0; i < n_prefill; i++) {
				float *out = (i == n_prefill - 1) ? logits_cpu : NULL;
				compute_forward(&sm_cpu->m, &kv, &s, prompt[i], i, flash_attn, out);
				kv.n_pos++;
			}
			if (cpu->synchronize)
				cpu->synchronize(cpu);
			compute_scratch_free(&s);
			kvcache_free(&kv);
		}
		{
			kvcache kv;
			kvcache_init(&kv, &sm_tgt->m, n_ctx, KV_QUANT_F16);
			compute_scratch s;
			compute_scratch_init(&s);
			(void)compute_scratch_ensure(&s, &sm_tgt->m, n_ctx);
			for (int i = 0; i < n_prefill; i++) {
				float *out = (i == n_prefill - 1) ? logits_tgt : NULL;
				compute_forward(&sm_tgt->m, &kv, &s, prompt[i], i, flash_attn, out);
				kv.n_pos++;
			}
			if (tgt->synchronize)
				tgt->synchronize(tgt);
			compute_scratch_free(&s);
			kvcache_free(&kv);
		}

		int32_t am_cpu = sampler_argmax(logits_cpu, cfg.vocab);
		int32_t am_tgt = sampler_argmax(logits_tgt, cfg.vocab);

		int		at;
		float	max_abs = max_abs_diff_at(logits_cpu, logits_tgt, cfg.vocab, &at);
		float	ratio = max_combined_ratio_at(logits_cpu, logits_tgt, cfg.vocab, ATOL_LOOSE * 10.0f,
											  RTOL_LOOSE, NULL);
		verdict v	  = (ratio <= 1.0f) ? V_PASS : V_FAIL;

		if (v != V_PASS) {
			if (!header_shown) {
				printf("    layers  max_abs      tol_ratio  argmax  verdict\n");
				printf("    ------  -----------  ---------  ------  -------\n");
				header_shown = 1;
			}
			printf("    %6d  %11.4e  %9.3f  %3s    %s\n", n_layers, max_abs, ratio,
				   am_cpu == am_tgt ? "yes" : "NO", "FAIL");
		}

		char label[160];
		char detail[320];
		snprintf(label, sizeof(label), "arch.error_compound %d layers prefill=%d", n_layers,
				 n_prefill);
		snprintf(detail, sizeof(detail), "max_abs=%.4e@%d tol_ratio=%.3f argmax cpu=%d tgt=%d %s",
				 max_abs, at, ratio, am_cpu, am_tgt, am_cpu == am_tgt ? "(agree)" : "(DIVERGE)");
		if (v != V_PASS)
			compute_debug(logits_cpu, logits_tgt, cfg.vocab);
		record_result(OPFAM_ARCH_COMPOUND, label, v, detail);

		free(logits_cpu);
		free(logits_tgt);
		free(prompt);
		synth_model_free(sm_cpu);
		synth_model_free(sm_tgt);
	}
}

static void test_arch_generate(backend *cpu, backend *tgt, model_arch arch, const synth_cfg *cfg,
							   int n_prefill, int n_decode, int flash_attn) {
	const char *arch_name = arch_lookup(arch)->gguf_name;
	int			n_ctx	  = n_prefill + n_decode + 16;

	synth_model *sm_cpu = synth_model_build(cpu, arch, cfg, n_ctx);
	synth_model *sm_tgt = tgt ? synth_model_build(tgt, arch, cfg, n_ctx) : NULL;

	int32_t *prompt = xmalloc((size_t)n_prefill * sizeof(int32_t));
	seed_test_rng(0x9E3779B9ULL + (uint64_t)arch);
	for (int i = 0; i < n_prefill; i++)
		prompt[i] = (int32_t)(next_u32() % (uint32_t)cfg->vocab);

	float *logits_cpu = xmalloc((size_t)cfg->vocab * sizeof(float));
	float *logits_tgt = tgt ? xmalloc((size_t)cfg->vocab * sizeof(float)) : NULL;

	kvcache			kv_cpu;
	status_code		kv_cpu_status = kvcache_init(&kv_cpu, &sm_cpu->m, n_ctx, KV_QUANT_F16);
	compute_scratch s_cpu;
	compute_scratch_init(&s_cpu);
	(void)compute_scratch_ensure(&s_cpu, &sm_cpu->m, n_ctx);

	kvcache			kv_tgt;
	compute_scratch s_tgt;
	status_code		kv_tgt_status = OK;
	if (sm_tgt) {
		kv_tgt_status = kvcache_init(&kv_tgt, &sm_tgt->m, n_ctx, KV_QUANT_F16);
		if (kv_tgt_status != OK) {
			synth_model_free(sm_tgt);
			sm_tgt = NULL;
		} else {
			compute_scratch_init(&s_tgt);
			(void)compute_scratch_ensure(&s_tgt, &sm_tgt->m, n_ctx);
		}
	}

	if (kv_cpu_status != OK) {
		char label[160];
		char detail[320];
		snprintf(label, sizeof(label), "arch.generate[%s] prefill (%d tokens)", arch_name,
				 n_prefill);
		snprintf(detail, sizeof(detail), "kvcache_init failed: status=%d", kv_cpu_status);
		record_result(OPFAM_ARCH_GENERATE, label, V_SKIP, detail);
		free(logits_cpu);
		free(logits_tgt);
		free(prompt);
		compute_scratch_free(&s_cpu);
		if (sm_tgt) {
			compute_scratch_free(&s_tgt);
			kvcache_free(&kv_tgt);
		}
		synth_model_free(sm_cpu);
		if (sm_tgt)
			synth_model_free(sm_tgt);
		return;
	}

	int n_agree		   = 0;
	int n_steps		   = 0;
	int self_test_fail = 0;

	char label[160];
	char detail[320];

	status_code s_cpu_status = compute_forward_batch(&sm_cpu->m, &kv_cpu, &s_cpu, prompt, n_prefill,
													 0, flash_attn, logits_cpu);
	kv_cpu.n_pos += n_prefill;
	if (cpu->synchronize)
		cpu->synchronize(cpu);
	status_code s_tgt_status = OK;
	if (sm_tgt) {
		s_tgt_status = compute_forward_batch(&sm_tgt->m, &kv_tgt, &s_tgt, prompt, n_prefill, 0,
											 flash_attn, logits_tgt);
		kv_tgt.n_pos += n_prefill;
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}

	if (s_cpu_status != OK) {
		self_test_fail = 1;
	} else {
		int nf = count_nonfinite(logits_cpu, cfg->vocab);
		if (nf > 0)
			self_test_fail = 1;
	}

	snprintf(label, sizeof(label), "arch.generate[%s] prefill (%d tokens)", arch_name, n_prefill);
	if (self_test_fail) {
		snprintf(detail, sizeof(detail), "CPU self-test failure: status=%d nonfinite=%d/%d",
				 s_cpu_status, s_cpu_status == OK ? count_nonfinite(logits_cpu, cfg->vocab) : -1,
				 cfg->vocab);
		record_result(OPFAM_ARCH_GENERATE, label, V_FAIL, detail);
	} else if (sm_tgt) {
		verdict v = classify_output("loose", logits_cpu, logits_tgt, cfg->vocab, s_tgt_status,
									detail, sizeof(detail));
		if (v != V_PASS && v != V_SKIP)
			compute_debug(logits_cpu, logits_tgt, cfg->vocab);
		record_result(OPFAM_ARCH_GENERATE, label, v, detail);
		if (v != V_FAIL) {
			int32_t am_cpu = sampler_argmax(logits_cpu, cfg->vocab);
			int32_t am_tgt = sampler_argmax(logits_tgt, cfg->vocab);
			n_agree += (am_cpu == am_tgt);
			n_steps++;
		}
	} else {
		snprintf(detail, sizeof(detail), "prefill produced finite output (CPU self-test only)");
		record_result(OPFAM_ARCH_GENERATE, label, V_PASS, detail);
	}

	int32_t tok_cpu = sampler_argmax(logits_cpu, cfg->vocab);
	int32_t tok_tgt = sm_tgt ? sampler_argmax(logits_tgt, cfg->vocab) : tok_cpu;

	for (int step = 0; step < n_decode && !self_test_fail; step++) {
		int pos = n_prefill + step;

		s_cpu_status =
			compute_forward(&sm_cpu->m, &kv_cpu, &s_cpu, tok_cpu, pos, flash_attn, logits_cpu);
		kv_cpu.n_pos++;
		if (cpu->synchronize)
			cpu->synchronize(cpu);
		if (s_cpu_status != OK || count_nonfinite(logits_cpu, cfg->vocab) > 0) {
			snprintf(label, sizeof(label), "arch.generate[%s] decode step %d", arch_name, step);
			snprintf(detail, sizeof(detail), "CPU self-test failure: status=%d nonfinite=%d/%d",
					 s_cpu_status,
					 s_cpu_status == OK ? count_nonfinite(logits_cpu, cfg->vocab) : -1, cfg->vocab);
			record_result(OPFAM_ARCH_GENERATE, label, V_FAIL, detail);
			break;
		}

		if (sm_tgt) {
			s_tgt_status =
				compute_forward(&sm_tgt->m, &kv_tgt, &s_tgt, tok_tgt, pos, flash_attn, logits_tgt);
			kv_tgt.n_pos++;
			if (tgt->synchronize)
				tgt->synchronize(tgt);
			snprintf(label, sizeof(label), "arch.generate[%s] decode step %d", arch_name, step);
			verdict v = classify_output("loose", logits_cpu, logits_tgt, cfg->vocab, s_tgt_status,
										detail, sizeof(detail));
			if (v != V_PASS && v != V_SKIP)
				compute_debug(logits_cpu, logits_tgt, cfg->vocab);
			record_result(OPFAM_ARCH_GENERATE, label, v, detail);

			int32_t am_cpu = sampler_argmax(logits_cpu, cfg->vocab);
			int32_t am_tgt = sampler_argmax(logits_tgt, cfg->vocab);
			n_agree += (am_cpu == am_tgt);
			n_steps++;
			tok_cpu = am_cpu;
			tok_tgt = am_tgt;
		} else {
			snprintf(label, sizeof(label), "arch.generate[%s] decode step %d", arch_name, step);
			snprintf(detail, sizeof(detail), "finite output (CPU self-test only)");
			record_result(OPFAM_ARCH_GENERATE, label, V_PASS, detail);
			tok_cpu = sampler_argmax(logits_cpu, cfg->vocab);
		}
	}

	if (sm_tgt && n_steps > 0 && n_agree < n_steps) {
		printf("    [%s] prefill=%d decode=%d  argmax agreement: %d/%d\n", arch_name, n_prefill,
			   n_decode, n_agree, n_steps);
	}

	free(logits_cpu);
	free(logits_tgt);
	free(prompt);
	compute_scratch_free(&s_cpu);
	kvcache_free(&kv_cpu);
	if (sm_tgt) {
		compute_scratch_free(&s_tgt);
		kvcache_free(&kv_tgt);
	}
	synth_model_free(sm_cpu);
	if (sm_tgt)
		synth_model_free(sm_tgt);
}

void run_arch_tests(backend *cpu, backend *tgt) {
	printf("\n========================================\n");
	if (tgt)
		printf("Architecture-level tests: %s  vs  cpu (reference)\n", tgt->name);
	else
		printf("Architecture-level tests: cpu (self-test)\n");
	printf("========================================\n");

	if (tgt) {
		test_arch_single_layer(cpu, tgt, &ARCH_CFG_SMALL, 0);
		test_arch_single_layer(cpu, tgt, &ARCH_CFG_SMALL, 1);
		flush_family(OPFAM_ARCH_LAYER);

		test_arch_pipeline(cpu, tgt, &ARCH_CFG_SMALL, 8, 0);
		test_arch_pipeline(cpu, tgt, &ARCH_CFG_SMALL, 8, 1);
		test_arch_pipeline(cpu, tgt, &ARCH_CFG_SMALL, 16, 1);
		test_arch_pipeline(cpu, tgt, &ARCH_CFG_DEEPER, 16, 1);
		test_arch_pipeline(cpu, tgt, &ARCH_CFG_DEEPER, 32, 1);
		flush_family(OPFAM_ARCH_PIPELINE);

		test_arch_decode_chain(cpu, tgt, &ARCH_CFG_SMALL, 8, 8, 1);
		test_arch_decode_chain(cpu, tgt, &ARCH_CFG_DEEPER, 16, 8, 1);
		flush_family(OPFAM_ARCH_DECODE);

		test_arch_error_compounding(cpu, tgt, &ARCH_CFG_SMALL, 16, 1);
		flush_family(OPFAM_ARCH_COMPOUND);
	}

	static const model_arch generate_archs[] = {
		ARCH_LLAMA,
		ARCH_GEMMA4,
		ARCH_GLM_DSA,
	};
	for (size_t gi = 0; gi < ARRAY_LEN(generate_archs); gi++) {
		test_arch_generate(cpu, tgt, generate_archs[gi], &ARCH_CFG_SMALL, 16,
						   ARCH_GENERATE_N_DECODE, 1);
	}
	test_arch_generate(cpu, tgt, ARCH_GEMMA4, &ARCH_CFG_KV_SHARED, 16, ARCH_GENERATE_N_DECODE, 1);
	test_arch_generate(cpu, tgt, ARCH_GLM_DSA, &ARCH_CFG_MLA, 16, ARCH_GENERATE_N_DECODE, 1);
	test_arch_generate(cpu, tgt, ARCH_GLM_DSA, &ARCH_CFG_MLA_WIDE, 16, ARCH_GENERATE_N_DECODE, 1);
	test_arch_batch_vs_single(cpu, ARCH_GLM_DSA, &ARCH_CFG_MLA, 8, 1);
	test_arch_batch_vs_single(cpu, ARCH_GLM_DSA, &ARCH_CFG_MLA_DENSE_FIRST, 8, 1);
	test_arch_batch_vs_single(cpu, ARCH_GLM_DSA, &ARCH_CFG_MLA_WIDE, 8, 1);
	test_arch_batch_vs_single(cpu, ARCH_GEMMA4, &ARCH_CFG_SMALL, 8, 1);
	test_arch_batch_vs_single(cpu, ARCH_GEMMA4, &ARCH_CFG_KV_SHARED, 8, 1);
	test_arch_batch_vs_single(cpu, ARCH_GEMMA4, &ARCH_CFG_KV_SHARED, 16, 1);
	flush_family(OPFAM_ARCH_GENERATE);
}
