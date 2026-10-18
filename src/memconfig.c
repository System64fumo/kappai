#define _GNU_SOURCE
#include "memconfig.h"
#include "config.h"
#include "gguf.h"
#include "kvcache.h"
#include "log.h"
#include "mempool.h"
#include "model.h"
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
		if (line[0] == 'M' && strncmp(line, "MemAvailable:", 13) == 0) {
			mem_available = strtoull(line + 13, NULL, 10) * 1024;
			break;
		}
		if (strncmp(line, "MemFree:", 8) == 0)
			mem_free = strtoull(line + 8, NULL, 10) * 1024;
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

static size_t calc_attn_bytes(const model *m, int i) {
	const layer_weights *L	   = &m->layers[i];
	size_t				 total = 0;
	if (m->arch_info->is_kda && model_layer_is_recurrent(m, i)) {
		const size_t d_inner = (size_t)m->hybrid.value_dim;
		total += 3 * ggml_row_size(L->wq.type, m->dim) * d_inner;
		total += ggml_row_size(L->wo.type, d_inner) * m->dim;
		total += 2 * ggml_row_size(L->kda_f_a_w.type, m->dim) * m->kda.head_dim;
		total += 2 * ggml_row_size(L->kda_f_b_w.type, m->kda.head_dim) * d_inner;
	} else if (m->arch_info->is_mla) {
		total += ggml_row_size(L->q_a_w.type, m->dim) * m->mla.q_lora;
		total += ggml_row_size(L->q_b_w.type, m->mla.q_lora) * (m->n_heads * m->mla.qk_head);
		total += m->mla.q_lora * sizeof(float);
		total += ggml_row_size(L->kv_a_w.type, m->dim) * (m->mla.kv_lora + m->mla.qk_rope);
		total += ggml_row_size(L->k_b_w.type, m->mla.qk_nope * m->mla.kv_lora) * m->n_heads;
		total += ggml_row_size(L->v_b_w.type, m->mla.kv_lora * m->mla.v_head) * m->n_heads;
		total += m->mla.kv_lora * sizeof(float);
		int wo_in = m->n_heads * m->mla.v_head;
		total += ggml_row_size(L->wo.type, wo_in) * m->dim;
	} else {
		int head_dim = m->layers[i].head_dim;
		int q_out	 = m->n_heads * head_dim;
		int kv_out	 = m->layers[i].n_kv_heads * head_dim;
		total += ggml_row_size(L->wq.type, m->dim) * q_out;
		total += ggml_row_size(L->wk.type, m->dim) * kv_out;
		if (m->layers[i].has_own_v)
			total += ggml_row_size(L->wv.type, m->dim) * kv_out;
		total += ggml_row_size(L->wo.type, q_out) * m->dim;
	}
	total += m->dim * sizeof(float) * 4;
	return total;
}

static size_t calc_dense_ffn_bytes(const model *m, int i) {
	const layer_weights *L = &m->layers[i];
	if (L->is_moe_layer && !m->arch_info->uses_moe_shared_dense_ffn)
		return 0;
	int	   inter = m->layers[i].intermediate;
	size_t total = 0;
	total += ggml_row_size(L->gate_w.type, m->dim) * inter;
	total += ggml_row_size(L->up_w.type, m->dim) * inter;
	total += ggml_row_size(L->down_w.type, inter) * m->dim;
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

int model_kv_layer_pos_cap(const model *m, int n_ctx, int kv_layer) {
	if (kv_layer < 0 || kv_layer >= m->n_layers)
		return n_ctx;
	if (m->sliding_window <= 0 || !model_layer_is_sliding(m, kv_layer))
		return n_ctx;
	long w	   = m->sliding_window;
	long slack = w / 4;
	if (slack < 8)
		slack = 8;
	long cap = w + slack;
	return cap >= (long)n_ctx ? n_ctx : (int)cap;
}

size_t model_kv_slot_bytes(const model *m, int n_ctx, kv_quant_type kv_quant, int compacted,
						   int slot) {
	int kv_heads = model_layer_kv_heads(m, slot);
	if (!m->arch_info->has_variable_layer_dims && m->layer_dims.n_kv_heads_per_layer &&
		slot < m->n_layers)
		kv_heads = m->layer_dims.n_kv_heads_per_layer[slot] > 0
					   ? m->layer_dims.n_kv_heads_per_layer[slot]
					   : 0;
	int hdim = model_layer_head_dim(m, slot);
	int pcap = compacted ? model_kv_layer_pos_cap(m, n_ctx, slot) : n_ctx;
	if (kv_quant == KV_QUANT_Q8_0) {
		size_t n_blocks = ((size_t)hdim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		return (size_t)kv_heads * n_blocks * KV_Q8_0_BLOCK_BYTES * (size_t)pcap * 2;
	}
	return (size_t)kv_heads * (size_t)hdim * (size_t)pcap * sizeof(uint16_t) * 2;
}

size_t model_kv_cache_bytes_quant(const model *m, int n_ctx, kv_quant_type kv_quant,
								  int compacted) {
	size_t kv_cache = 0;
	if (m->arch_info->is_mla) {
		kv_cache = ((size_t)m->mla.kv_lora + (size_t)m->mla.qk_rope) * (size_t)n_ctx *
				   (size_t)m->n_layers * sizeof(float);
		return kv_cache;
	}
	int n_kv_layers =
		m->layer_dims.n_layer_kv_from_start > 0 ? m->layer_dims.n_layer_kv_from_start : m->n_layers;
	if (n_kv_layers > m->n_layers)
		n_kv_layers = m->n_layers;
	for (int i = 0; i < n_kv_layers; i++)
		kv_cache += model_kv_slot_bytes(m, n_ctx, kv_quant, compacted, i);
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

static int moe_layer_count(const model *m) {
	int n_layers = m->n_layers - m->moe.first_dense_layer;
	return n_layers < 0 ? 0 : n_layers;
}

static int moe_cache_cap(const model *m, const config *cfg) {
	int cache_cap = cfg->moe_cache_cap > 0 ? cfg->moe_cache_cap : MOE_DEFAULT_CACHE_CAP;
	if (cache_cap > 1024)
		cache_cap = 1024;
	if (cache_cap > m->moe.n_experts)
		cache_cap = m->moe.n_experts;
	return cache_cap;
}

typedef struct {
	size_t embd;
	size_t attn;
	size_t dense_ffn;
	size_t shexp;
	size_t router;
	size_t total;
} non_expert_breakdown;

static non_expert_breakdown calc_non_expert_breakdown(const model *m) {
	non_expert_breakdown b = {0};
	b.embd				   = calc_embeddings_bytes(m);
	for (int i = 0; i < m->n_layers; i++) {
		b.attn += calc_attn_bytes(m, i);
		b.dense_ffn += calc_dense_ffn_bytes(m, i);
		b.shexp += calc_shared_expert_bytes(m, i);
		b.router += calc_router_bytes(m, i);
	}
	b.total = b.embd + m->dim * sizeof(float) + b.attn + b.dense_ffn + b.shexp + b.router;
	return b;
}

static size_t calc_non_expert_bytes(const model *m) {
	if (m->mem_estimates_valid)
		return m->mem_non_expert_bytes;
	return calc_non_expert_breakdown(m).total;
}

static size_t model_cached_per_expert_size(const model *m) {
	if (m->mem_estimates_valid)
		return m->mem_per_expert_bytes;
	return calc_per_expert_size(m);
}

void model_mem_estimates_ensure(model *m) {
	if (m->mem_estimates_valid)
		return;
	m->mem_non_expert_bytes = calc_non_expert_breakdown(m).total;
	m->mem_per_expert_bytes = calc_per_expert_size(m);
	m->mem_estimates_valid	= true;
}

size_t model_device_weight_bytes(const model *m, const backend *b) {
	size_t total = 0;
	for (int i = 0; i < m->n_layers; i++) {
		if (model_layer_backend(m, i) != b)
			continue;
		total += calc_attn_bytes(m, i) + calc_dense_ffn_bytes(m, i) +
				 calc_shared_expert_bytes(m, i) + calc_router_bytes(m, i);
	}
	if (b == m->backend)
		total += calc_embeddings_bytes(m) + m->dim * sizeof(float);
	return total;
}

static size_t model_total_weight_bytes(const model *m) {
	size_t non_expert	= calc_non_expert_bytes(m);
	size_t per_expert	= model_cached_per_expert_size(m);
	int	   n_layers		= moe_layer_count(m);
	size_t total_expert = per_expert * (size_t)m->moe.n_experts * (size_t)n_layers;
	return non_expert + total_expert;
}

static size_t hcs_pending_expert_bytes(const model *m, const config *cfg) {
	size_t per_expert	= model_cached_per_expert_size(m);
	int	   n_layers		= moe_layer_count(m);
	int	   n_experts	= m->moe.n_experts;
	bool   primary_host = cfg->device && backend_device_is_host(cfg->device);
	int	   on_primary	= 0;
	int	   assigned		= 0;
	for (int i = 0; i < cfg->n_expert_devices; i++) {
		int cnt = MIN(cfg->expert_counts[i], n_experts - assigned);
		assigned += cnt;
		bool is_host = backend_device_is_host(cfg->expert_devices[i]);
		if (is_host ? primary_host
					: (cfg->device && strcmp(cfg->expert_devices[i], cfg->device) == 0))
			on_primary += cnt;
	}
	if (primary_host) {
		int host_experts = on_primary + (n_experts - assigned);
		on_primary = cfg->moe_stream ? MIN(host_experts, MOE_DEFAULT_CACHE_CAP) : host_experts;
	}
	return per_expert * (size_t)on_primary * (size_t)n_layers;
}

size_t model_pending_weight_bytes(const model *m, const config *cfg) {
	if (m->arch_info->is_moe && cfg->n_expert_devices > 0)
		return hcs_pending_expert_bytes(m, cfg);
	int full_resident = !cfg->use_mmap && cfg->moe_stream == 0;
	if (full_resident || !m->arch_info->is_moe)
		return cfg->use_mmap ? model_total_weight_bytes(m) : 0;

	size_t per_expert = model_cached_per_expert_size(m);
	int	   n_layers	  = moe_layer_count(m);
	int	   cache_cap  = moe_cache_cap(m, cfg);

	return per_expert * (size_t)cache_cap * (size_t)n_layers;
}

static void format_pool_users(const mem_pool *p, char *out, size_t cap) {
	size_t len = 0;
	out[0]	   = '\0';
	for (int i = 0; i < p->n_members && len < cap; i++) {
		char name[BACKEND_DEVICE_NAME_MAX];
		backend_format_device(p->members[i]->name, p->members[i]->device_index, name, sizeof(name));
		len += (size_t)snprintf(out + len, cap - len, "%s%s", i ? ", " : "", name);
	}
}

static void log_pools(const mem_topology *t) {
	char   pool_names[BACKEND_MAX * 8];
	size_t names_len = 0;
	pool_names[0]	 = '\0';
	for (int i = 0; i < t->n_pools; i++) {
		char users[256];
		format_pool_users(&t->pools[i], users, sizeof(users));
		log_tag("MEM", "pool%d: %.4g GB | Users: %s", i,
				to_unit(mem_pool_available(&t->pools[i]), MEM_UNIT_GB), users);
		names_len += (size_t)snprintf(pool_names + names_len, sizeof(pool_names) - names_len,
									  "%spool%d", i ? ", " : "", i);
	}
	log_tag("MEM", "Available memory: %.4g GB (%s)",
			to_unit(mem_topology_available(t), MEM_UNIT_GB), pool_names);
}

void report_memory(const model *m, const mem_topology *t, backend *kv_backend, int n_ctx,
				   kv_quant_type kv_quant) {
	if (n_ctx <= 0 || n_ctx > m->n_ctx)
		n_ctx = m->n_ctx;

	non_expert_breakdown bd			= calc_non_expert_breakdown(m);
	size_t				 per_expert = model_cached_per_expert_size(m);
	size_t expert_total = per_expert * (size_t)m->moe.n_experts * (size_t)moe_layer_count(m);
	size_t kv_cache		= model_kv_cache_bytes_quant(m, n_ctx, kv_quant,
													 backend_has_cap(kv_backend, BCAP_KV_POS_CAP));

	log_pools(t);
	DEBUG("memory breakdown:");
	DEBUG("  embeddings:        %.1f MB", to_unit(bd.embd, MEM_UNIT_MB));
	DEBUG("  attention weights: %.1f MB", to_unit(bd.attn, MEM_UNIT_MB));
	if (bd.dense_ffn > 0)
		DEBUG("  dense FFN weights: %.1f MB", to_unit(bd.dense_ffn, MEM_UNIT_MB));
	if (bd.shexp > 0)
		DEBUG("  shared experts:    %.1f MB", to_unit(bd.shexp, MEM_UNIT_MB));
	if (bd.router > 0)
		DEBUG("  MoE routers:       %.1f MB", to_unit(bd.router, MEM_UNIT_MB));
	log_tag("MEM", "Non expert weights: %.4g GB", to_unit(bd.total, MEM_UNIT_GB));
	if (expert_total > 0)
		log_tag("MEM", "Expert weights: %.4g GB", to_unit(expert_total, MEM_UNIT_GB));
	log_tag("MEM", "KV cache: %.4g GB", to_unit(kv_cache, MEM_UNIT_GB));
	log_tag("MEM", "Required: %.4g GB", to_unit(bd.total + expert_total + kv_cache, MEM_UNIT_GB));
}
