#define _GNU_SOURCE
#include "memconfig.h"
#include "gguf.h"
#include "log.h"
#include "moe/moe_stream.h"
#include <string.h>
#include <sys/sysinfo.h>

size_t get_available_memory(void) {
	FILE *f = fopen("/proc/meminfo", "r");
	if (!f) {
		struct sysinfo si;
		if (sysinfo(&si) == 0)
			return (size_t)si.freeram * (size_t)si.mem_unit;
		return 0;
	}
	char   line[256];
	size_t mem_available = 0;
	size_t mem_free		 = 0;
	size_t mem_cached	 = 0;
	size_t mem_buffers	 = 0;
	while (fgets(line, sizeof(line), f)) {
		if (strncmp(line, "MemFree:", 8) == 0)
			mem_free = strtoull(line + 8, NULL, 10) * 1024;
		else if (strncmp(line, "MemAvailable:", 13) == 0)
			mem_available = strtoull(line + 13, NULL, 10) * 1024;
		else if (strncmp(line, "Cached:", 7) == 0)
			mem_cached = strtoull(line + 7, NULL, 10) * 1024;
		else if (strncmp(line, "Buffers:", 8) == 0)
			mem_buffers = strtoull(line + 8, NULL, 10) * 1024;
	}
	fclose(f);
	if (mem_available > 0)
		return mem_available;
	return mem_free + mem_cached + mem_buffers;
}

size_t get_total_memory(void) {
	struct sysinfo si;
	if (sysinfo(&si) != 0)
		return 0;
	return (size_t)si.totalram * (size_t)si.mem_unit;
}

static size_t calc_embeddings_bytes(const model *m) {
	size_t total = ggml_row_size(m->tok_embd.type, m->dim) * m->vocab_size;
	if (!m->tie_embeddings)
		total += ggml_row_size(m->output_w.type, m->dim) * m->vocab_size;
	return total;
}

static size_t wr_bytes(const weight_ref *w, uint64_t d0, uint64_t d1) {
	if (!w->host_ptr)
		return 0;
	size_t row = ggml_row_size(w->type, d0);
	return d1 ? row * d1 : row;
}

static size_t calc_attn_bytes(const model *m, int i) {
	const layer_weights *L	   = &m->layers[i];
	size_t				 total = wr_bytes(&L->attn_norm_w, m->dim, 0);
	if (m->arch_info->is_mla) {
		total += wr_bytes(&L->q_a_w, m->dim, m->mla.q_lora);
		total += wr_bytes(&L->q_b_w, m->mla.q_lora, (uint64_t)m->n_heads * m->mla.qk_head);
		total += wr_bytes(&L->q_a_norm_w, m->mla.q_lora, 0);
		total += wr_bytes(&L->kv_a_w, m->dim, (uint64_t)m->mla.kv_lora + m->mla.qk_rope);
		total += wr_bytes(&L->k_b_w, (uint64_t)m->mla.qk_nope * m->mla.kv_lora, m->n_heads);
		total += wr_bytes(&L->v_b_w, (uint64_t)m->mla.kv_lora * m->mla.v_head, m->n_heads);
		total += wr_bytes(&L->kv_a_norm_w, m->mla.kv_lora, 0);
		int wo_in = m->n_heads * m->mla.v_head;
		total += wr_bytes(&L->wo, wo_in, m->dim);
	} else if (L->attn_qkv_w.host_ptr) {
		const model_qwen35_params *p = &m->qwen35;
		total += wr_bytes(&L->attn_qkv_w, m->dim, p->conv_dim);
		total += wr_bytes(&L->attn_gate_w, m->dim, p->value_dim);
		total += wr_bytes(&L->ssm_conv1d_w, p->conv_kernel, p->conv_dim);
		total += wr_bytes(&L->ssm_dt_b, p->n_value_heads, 0);
		total += wr_bytes(&L->ssm_a, p->n_value_heads, 0);
		total += wr_bytes(&L->ssm_beta_w, m->dim, p->n_value_heads);
		total += wr_bytes(&L->ssm_alpha_w, m->dim, p->n_value_heads);
		total += wr_bytes(&L->ssm_norm_w, p->value_head_dim, 0);
		total += wr_bytes(&L->ssm_out_w, p->value_dim, m->dim);
	} else {
		int head_dim = L->head_dim;
		int q_out	 = m->n_heads * head_dim;
		int kv_out	 = L->n_kv_heads * head_dim;
		total += wr_bytes(&L->wq, m->dim, q_out);
		total += wr_bytes(&L->wk, m->dim, kv_out);
		total += wr_bytes(&L->wv, m->dim, kv_out);
		total += wr_bytes(&L->wo, q_out, m->dim);
		total += wr_bytes(&L->attn_q_norm_w, head_dim, 0);
		total += wr_bytes(&L->attn_k_norm_w, head_dim, 0);
	}
	total += wr_bytes(&L->post_attn_norm_w, m->dim, 0);
	return total;
}

static size_t calc_dense_ffn_bytes(const model *m, int i) {
	const layer_weights *L = &m->layers[i];
	if (L->is_moe_layer && !m->arch_info->uses_moe_shared_dense_ffn)
		return 0;
	int	   inter = L->intermediate;
	size_t total = 0;
	total += wr_bytes(&L->gate_w, m->dim, inter);
	total += wr_bytes(&L->up_w, m->dim, inter);
	total += wr_bytes(&L->down_w, inter, m->dim);
	total += wr_bytes(&L->ffn_norm_w, m->dim, 0);
	total += wr_bytes(&L->post_ffn_norm_w, m->dim, 0);
	return total;
}

static size_t calc_shared_expert_bytes(const model *m, int i) {
	const layer_weights *L = &m->layers[i];
	if (!L->is_moe_layer || m->moe.n_shared_experts <= 0)
		return 0;
	int	   sh	 = m->moe.moe_intermediate * m->moe.n_shared_experts;
	size_t total = 0;
	total += ggml_row_size(L->shexp_gate_w.type, m->dim) * sh;
	total += ggml_row_size(L->shexp_up_w.type, m->dim) * sh;
	total += ggml_row_size(L->shexp_down_w.type, sh) * m->dim;
	return total;
}

static size_t calc_router_bytes(const model *m, int i) {
	if (!m->layers[i].is_moe_layer)
		return 0;
	return m->dim * m->moe.n_experts * sizeof(float);
}

static size_t calc_per_expert_size(const model *m) {
	if (m->n_layers == 0)
		return 0;
	int first_moe = m->moe.first_dense_layer;
	if (first_moe < 0 || first_moe >= m->n_layers)
		first_moe = 0;
	return moe_calc_expert_bytes(m, first_moe, 0).total;
}

size_t model_kv_cache_bytes_quant(const model *m, int n_ctx, kv_quant_type kv_quant) {
	size_t kv_cache = 0;
	if (m->arch_info->is_mla) {
		kv_cache = ((size_t)m->mla.kv_lora + (size_t)m->mla.qk_rope) * (size_t)n_ctx *
				   (size_t)m->n_layers * sizeof(float);
		return kv_cache;
	}
	for (int i = 0; i < m->n_layers; i++) {
		int kv_heads = model_layer_kv_heads(m, i);
		int hdim	 = model_layer_head_dim(m, i);
		if (kv_quant == KV_QUANT_Q8_0) {
			size_t n_blocks = ((size_t)hdim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
			kv_cache += (size_t)kv_heads * n_blocks * KV_Q8_0_BLOCK_BYTES * (size_t)n_ctx * 2;
		} else {
			kv_cache += (size_t)kv_heads * (size_t)hdim * (size_t)n_ctx * sizeof(uint16_t) * 2;
		}
	}
	return kv_cache;
}

typedef enum {
	MEM_UNIT_MB,
	MEM_UNIT_GB,
} mem_unit;

static double to_unit(size_t bytes, mem_unit unit) {
	const double k		 = 1024.0;
	double		 divisor = (unit == MEM_UNIT_GB) ? k * k * k : k * k;
	return bytes / divisor;
}

void recommend_memory_config(const model *m, int n_ctx, size_t avail, kv_quant_type kv_quant) {
	if (n_ctx <= 0 || n_ctx > m->n_ctx)
		n_ctx = m->n_ctx;
	if (avail == 0)
		return;

	size_t embd_bytes	   = calc_embeddings_bytes(m);
	size_t attn_bytes	   = 0;
	size_t dense_ffn_bytes = 0;
	size_t shexp_bytes	   = 0;
	size_t router_bytes	   = 0;
	for (int i = 0; i < m->n_layers; i++) {
		attn_bytes += calc_attn_bytes(m, i);
		dense_ffn_bytes += calc_dense_ffn_bytes(m, i);
		shexp_bytes += calc_shared_expert_bytes(m, i);
		router_bytes += calc_router_bytes(m, i);
	}
	size_t non_expert = embd_bytes + m->dim * sizeof(float) + attn_bytes + dense_ffn_bytes +
						shexp_bytes + router_bytes;
	size_t per_expert = calc_per_expert_size(m);
	int	   n_experts  = m->moe.n_experts;
	int	   n_layers	  = m->n_layers - m->moe.first_dense_layer;
	if (n_layers < 0)
		n_layers = 0;
	int	   topk			= m->moe.n_experts_used;
	size_t kv_cache		= model_kv_cache_bytes_quant(m, n_ctx, kv_quant);
	size_t total_expert = per_expert * (size_t)n_experts * (size_t)n_layers;

	INFO("RAM: %.1f GB available", to_unit(avail, MEM_UNIT_GB));
	DEBUG("memory breakdown:");
	DEBUG("  embeddings:        %.1f MB", to_unit(embd_bytes, MEM_UNIT_MB));
	DEBUG("  attention weights: %.1f MB", to_unit(attn_bytes, MEM_UNIT_MB));
	if (dense_ffn_bytes > 0)
		DEBUG("  dense FFN weights: %.1f MB", to_unit(dense_ffn_bytes, MEM_UNIT_MB));
	if (shexp_bytes > 0)
		DEBUG("  shared experts:    %.1f MB", to_unit(shexp_bytes, MEM_UNIT_MB));
	if (router_bytes > 0)
		DEBUG("  MoE routers:       %.1f MB", to_unit(router_bytes, MEM_UNIT_MB));
	DEBUG("  non-expert total:  %.1f GB", to_unit(non_expert, MEM_UNIT_GB));
	if (per_expert > 0 && n_experts > 0) {
		DEBUG("  routed experts:    %.1f GB (%.1f MB/expert, %d experts x %d layers)",
			  to_unit(total_expert, MEM_UNIT_GB), to_unit(per_expert, MEM_UNIT_MB), n_experts,
			  n_layers);
	}
	DEBUG("  KV cache (ctx=%d): %.0f MB (%s)", n_ctx, to_unit(kv_cache, MEM_UNIT_MB),
		  kv_quant == KV_QUANT_Q8_0 ? "q8_0" : "f16");
	INFO("model total: %.1f GB", to_unit(non_expert + total_expert, MEM_UNIT_GB));

	if (per_expert == 0 || n_experts == 0) {
		if (non_expert > avail) {
			WARN("RAM: model doesn't fit in RAM (%.1f MB > %.1f GB available), will stream "
				 "from disk",
				 to_unit(non_expert, MEM_UNIT_MB), to_unit(avail, MEM_UNIT_GB));
		}
		return;
	}

	size_t overhead = (size_t)1 * 1024 * 1024 * 1024;
	if (avail < non_expert + kv_cache + overhead) {
		WARN("RAM: not enough for non-expert weights (%.1f GB) + KV (%.0f MB) + overhead (1 GB)",
			 to_unit(non_expert, MEM_UNIT_GB), to_unit(kv_cache, MEM_UNIT_MB));
		INFO("  Suggested: --mmap --moe-cache %d (let kernel manage page cache)", topk);
		return;
	}

	size_t expert_budget		= avail - non_expert - kv_cache - overhead;
	int	   affordable_experts	= (int)(expert_budget / per_expert);
	int	   affordable_per_layer = affordable_experts / n_layers;
	if (affordable_per_layer < 1)
		affordable_per_layer = 1;
	if (affordable_per_layer > n_experts)
		affordable_per_layer = n_experts;

	int suggested_cache = affordable_per_layer;
	if (suggested_cache < topk)
		suggested_cache = topk;

	double resident_gb =
		to_unit(per_expert * (size_t)affordable_per_layer * (size_t)n_layers, MEM_UNIT_GB);

	if (affordable_per_layer >= n_experts) {
		INFO("  All experts fit in RAM (%.1f GB)", resident_gb);
		INFO("  Suggested: --moe-stream off --moe-cache %d", n_experts);
	} else if (affordable_per_layer >= topk) {
		INFO("  Partial fit: %d/%d experts/layer (%.1f GB)", affordable_per_layer, n_experts,
			 resident_gb);
		INFO("  Suggested: --moe-cache %d", suggested_cache);
	} else {
		INFO("  Tight: %d experts/layer fit (%.1f GB), topk=%d", affordable_per_layer, resident_gb,
			 topk);
		INFO("  Suggested: --moe-cache %d", affordable_per_layer);
	}
}