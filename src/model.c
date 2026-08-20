#define _GNU_SOURCE
#include "model.h"
#include "backend/cpu/scalar/quants.h"
#include "config.h"
#include "log.h"
#include "memconfig.h"
#include "moe/moe_stream.h"
#include "monitor.h"
#include "recipe.h"
#include "threadpool.h"
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define REPACK_MIN_ROWS_PER_THREAD 8

#define PREFETCH_MAX_THREADS 64
#define PREFETCH_CHUNK_MIN ((size_t)4u * 1024 * 1024)

static void madvise_dontneed(const void *map_base, size_t map_size, const void *ptr, size_t bytes) {
	if (!map_base || map_size == 0)
		return;
	long ps = sysconf(_SC_PAGESIZE);
	if (ps <= 0)
		ps = 4096;
	uintptr_t addr		 = (uintptr_t)ptr;
	uintptr_t page_start = addr & ~((uintptr_t)ps - 1);
	uintptr_t page_end	 = (addr + bytes + ps - 1) & ~((uintptr_t)ps - 1);
	if (page_start >= (uintptr_t)map_base && page_end <= (uintptr_t)map_base + map_size) {
		madvise((void *)page_start, page_end - page_start, MADV_DONTNEED);
	}
}

typedef struct {
	const void *src;
	void	   *dst;
	uint32_t	type;
	int			k;
	int			rows_per_group;
	void (*repack_fn)(const void *src, void *dst, int begin, int end, int k);
} repack_job;

typedef struct {
	int	   fd;
	off_t  offset;
	size_t len;
} prefetch_range;

static status_code akey_i32(const gguf_ctx *g, const char *prefix, const char *suffix,
							int32_t *out) {
	char key[160];
	snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
	return gguf_get_i32(g, key, out);
}

static status_code akey_f32(const gguf_ctx *g, const char *prefix, const char *suffix, float *out) {
	char key[160];
	snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
	return gguf_get_f32(g, key, out);
}

static status_code req_i32(const gguf_ctx *g, const char *prefix, const char *suffix,
						   int32_t *out) {
	status_code s = akey_i32(g, prefix, suffix, out);
	if (s != OK)
		ERROR("model_load: missing required metadata key '%s.%s'", prefix, suffix);
	return s;
}

static status_code req_f32(const gguf_ctx *g, const char *prefix, const char *suffix, float *out) {
	status_code s = akey_f32(g, prefix, suffix, out);
	if (s != OK)
		ERROR("model_load: missing required metadata key '%s.%s'", prefix, suffix);
	return s;
}

static const gguf_tensor *find_tensor_fmt(const gguf_ctx *g, char *tname, size_t tname_sz,
										  const char *fmt, int i)
	__attribute__((format(printf, 4, 0)));

static const gguf_tensor *find_tensor_fmt(const gguf_ctx *g, char *tname, size_t tname_sz,
										  const char *fmt, int i) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	snprintf(tname, tname_sz, fmt, i);
#pragma GCC diagnostic pop
	const gguf_tensor *t = gguf_find_tensor(g, tname);
	if (!t)
		ERROR("model_load: missing required tensor '%s'", tname);
	return t;
}

static int check_dims_1d(const gguf_ctx *g, const char *name, uint64_t d0) {
	const gguf_tensor *t = gguf_find_tensor(g, name);
	if (!t)
		return -1;
	if (t->n_dims != 1)
		return -1;
	if (t->dims[0] != d0)
		return -1;
	return 0;
}

static int check_dims_2d(const gguf_ctx *g, const char *name, uint64_t d0, uint64_t d1) {
	const gguf_tensor *t = gguf_find_tensor(g, name);
	if (!t)
		return -1;
	if (t->n_dims != 2)
		return -1;
	if (t->dims[0] != d0 || t->dims[1] != d1)
		return -1;
	return 0;
}

static int require_dims_1d(const gguf_ctx *g, const char *name, uint64_t d0) {
	if (check_dims_1d(g, name, d0)) {
		ERROR("model: %s has wrong shape", name);
		return -1;
	}
	return 0;
}

static int require_dims_2d(const gguf_ctx *g, const char *name, uint64_t d0, uint64_t d1) {
	if (check_dims_2d(g, name, d0, d1)) {
		ERROR("model: %s has wrong shape", name);
		return -1;
	}
	return 0;
}

static int require_layer_dims_1d(const gguf_ctx *g, char *tname, size_t tname_sz, const char *fmt,
								 int i, uint64_t d0) __attribute__((format(printf, 4, 0)));

static int require_layer_dims_1d(const gguf_ctx *g, char *tname, size_t tname_sz, const char *fmt,
								 int i, uint64_t d0) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	snprintf(tname, tname_sz, fmt, i);
#pragma GCC diagnostic pop
	return require_dims_1d(g, tname, d0);
}

static int require_layer_dims_2d(const gguf_ctx *g, char *tname, size_t tname_sz, const char *fmt,
								 int i, uint64_t d0, uint64_t d1)
	__attribute__((format(printf, 4, 0)));

static int require_layer_dims_2d(const gguf_ctx *g, char *tname, size_t tname_sz, const char *fmt,
								 int i, uint64_t d0, uint64_t d1) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	snprintf(tname, tname_sz, fmt, i);
#pragma GCC diagnostic pop
	return require_dims_2d(g, tname, d0, d1);
}

static int validate_swa_support(const model *m) {
	if (m->sliding_window <= 0)
		return 0;
	int any_sliding = 0;
	for (int i = 0; i < m->n_layers; i++) {
		if (m->layers[i].is_sliding) {
			any_sliding = 1;
			break;
		}
	}
	if (!any_sliding)
		return 0;
	int has_swa = m->backend->attention_swa != NULL ||
				  (backend_host() != m->backend && backend_host()->attention_swa != NULL);
	if (!has_swa) {
		ERROR("model: '%s' has sliding-window layers but backend '%s' (and its CPU fallback) "
			  "have no attention_swa implementation; refusing to load to avoid silently "
			  "computing full (unwindowed) attention on SWA layers",
			  m->arch_info->gguf_name, m->backend->name ? m->backend->name : "unknown");
		return -1;
	}
	return 0;
}

static int validate_model_dims(const model *m, const gguf_ctx *g) {
	char tname[128];

	if (require_dims_2d(g, "token_embd.weight", m->dim, m->vocab_size))
		return -1;
	if (require_dims_1d(g, "output_norm.weight", m->dim))
		return -1;
	if (!m->tie_embeddings) {
		if (require_dims_2d(g, "output.weight", m->dim, m->vocab_size))
			return -1;
	}

	if (m->arch_info->is_mla) {
		if (m->mla.qk_head > HEAD_DIM_MAX || m->mla.v_head > HEAD_DIM_MAX ||
			m->mla.kv_lora > HEAD_DIM_MAX) {
			ERROR("model: MLA head dims exceed supported maximum %d "
				  "(qk_head=%d v_head=%d kv_lora=%d)",
				  HEAD_DIM_MAX, m->mla.qk_head, m->mla.v_head, m->mla.kv_lora);
			return -1;
		}
	}

	for (int i = 0; i < m->n_layers; i++) {
		int head_dim = model_layer_head_dim(m, i);
		if (head_dim > HEAD_DIM_MAX) {
			ERROR("model: layer %d head_dim=%d exceeds supported maximum %d", i, head_dim,
				  HEAD_DIM_MAX);
			return -1;
		}
		int q_out		 = m->n_heads * head_dim;
		int q_weight_out = m->arch_info->has_attn_output_gate ? 2 * q_out : q_out;
		int kv_heads	 = model_layer_kv_heads(m, i);
		int kv_out		 = kv_heads * head_dim;
		int intermediate = model_layer_intermediate(m, i);
		int is_mla_layer = m->arch_info->is_mla;
		int is_moe_layer = model_layer_is_moe(m, i);

		if (require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.attn_norm.weight", i, m->dim))
			return -1;
		if (model_layer_is_recurrent(m, i)) {
			const model_qwen35_params *q = &m->qwen35;
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_qkv.weight", i,
									  m->dim, q->conv_dim) ||
				require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_gate.weight", i,
									  m->dim, q->value_dim) ||
				require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ssm_conv1d.weight", i,
									  q->conv_kernel, q->conv_dim) ||
				require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.ssm_dt.bias", i,
									  q->n_value_heads) ||
				require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.ssm_a", i,
									  q->n_value_heads) ||
				require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ssm_beta.weight", i,
									  m->dim, q->n_value_heads) ||
				require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ssm_alpha.weight", i,
									  m->dim, q->n_value_heads) ||
				require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.ssm_norm.weight", i,
									  q->value_head_dim) ||
				require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ssm_out.weight", i,
									  q->value_dim, m->dim))
				return -1;
		} else if (is_mla_layer) {
			int q_lora	  = m->mla.q_lora;
			int kv_lora	  = m->mla.kv_lora;
			int kv_a_rows = kv_lora + m->mla.qk_rope;
			int q_b_rows  = m->n_heads * m->mla.qk_head;
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_q_a.weight", i, m->dim,
									  q_lora))
				return -1;
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_q_b.weight", i, q_lora,
									  q_b_rows))
				return -1;
			if (require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.attn_q_a_norm.weight", i,
									  q_lora))
				return -1;
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_kv_a_mqa.weight", i,
									  m->dim, kv_a_rows))
				return -1;
			snprintf(tname, sizeof(tname), "blk.%d.attn_kv_a_norm.weight", i);
			if (check_dims_1d(g, tname, kv_lora) != 0) {
				snprintf(tname, sizeof(tname), "blk.%d.attn_kv_a_lora_norm.weight", i);
				if (check_dims_1d(g, tname, kv_lora) != 0) {
					ERROR("model: missing or wrong-shape kv_a_norm tensor");
					return -1;
				}
			}
			int wo_in = m->n_heads * m->mla.v_head;
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_output.weight", i,
									  wo_in, m->dim))
				return -1;
		} else {
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_q.weight", i, m->dim,
									  q_weight_out))
				return -1;
			if (model_layer_has_kv(m, i))
				if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_k.weight", i,
										  m->dim, kv_out))
					return -1;
			if (model_layer_has_own_v(m, i) && model_layer_has_kv(m, i)) {
				if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_v.weight", i,
										  m->dim, kv_out))
					return -1;
			}
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.attn_output.weight", i,
									  q_out, m->dim))
				return -1;
		}
		if (!m->arch_info->uses_post_attn_norm_for_ffn)
			if (require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.ffn_norm.weight", i,
									  m->dim))
				return -1;
		if (is_moe_layer) {
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_gate_inp.weight", i,
									  m->dim, m->moe.n_experts))
				return -1;
			if (m->moe.n_shared_experts > 0) {
				int sh_inter = m->moe.moe_intermediate * m->moe.n_shared_experts;
				if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_gate_shexp.weight",
										  i, m->dim, sh_inter))
					return -1;
				if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_up_shexp.weight", i,
										  m->dim, sh_inter))
					return -1;
				if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_down_shexp.weight",
										  i, sh_inter, m->dim))
					return -1;
			}
			{
				char vname[160];
				snprintf(vname, sizeof(vname), "blk.%d.ffn_gate_up_exps.weight", i);
				const gguf_tensor *et = gguf_find_tensor(g, vname);
				if (et) {
					if (et->n_dims < 2 || et->dims[0] != (uint64_t)m->dim) {
						ERROR("model: %s has wrong shape", vname);
						return -1;
					}
				} else {
					snprintf(vname, sizeof(vname), "blk.%d.ffn_gate_exps.weight", i);
					et = gguf_find_tensor(g, vname);
					if (et) {
						if (et->n_dims < 2 || et->dims[0] != (uint64_t)m->dim) {
							ERROR("model: %s has wrong shape", vname);
							return -1;
						}
					}
				}
				snprintf(vname, sizeof(vname), "blk.%d.ffn_down_exps.weight", i);
				et = gguf_find_tensor(g, vname);
				if (et) {
					if (et->n_dims < 2 || et->dims[0] != (uint64_t)m->moe.moe_intermediate) {
						ERROR("model: %s has wrong shape", vname);
						return -1;
					}
				}
			}
			if (m->arch_info->uses_moe_shared_dense_ffn) {
				if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_gate.weight", i,
										  m->dim, intermediate))
					return -1;
				if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_up.weight", i,
										  m->dim, intermediate))
					return -1;
				if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_down.weight", i,
										  intermediate, m->dim))
					return -1;
			}
		} else {
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_gate.weight", i, m->dim,
									  intermediate))
				return -1;
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_up.weight", i, m->dim,
									  intermediate))
				return -1;
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.ffn_down.weight", i,
									  intermediate, m->dim))
				return -1;
		}

		if (m->arch_info->has_attn_post_norm) {
			if (require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.post_attention_norm.weight",
									  i, m->dim))
				return -1;
		}
		if (m->arch_info->has_ffn_post_norm) {
			if (require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.post_ffw_norm.weight", i,
									  m->dim))
				return -1;
		}
		if (m->arch_info->has_qk_norm && !model_layer_is_recurrent(m, i)) {
			if (require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.attn_q_norm.weight", i,
									  head_dim))
				return -1;
			if (model_layer_has_kv(m, i))
				if (require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.attn_k_norm.weight", i,
										  head_dim))
					return -1;
		}
		if (m->has_per_layer_embeddings && m->arch_info->has_post_norm_ple) {
			if (require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.post_norm.weight", i,
									  m->dim))
				return -1;
		}
		if (m->has_per_layer_embeddings) {
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.inp_gate.weight", i, m->dim,
									  m->layer_dims.n_embd_per_layer))
				return -1;
			if (require_layer_dims_2d(g, tname, sizeof(tname), "blk.%d.proj.weight", i,
									  m->layer_dims.n_embd_per_layer, m->dim))
				return -1;
		}
		if (m->arch_info->has_layer_output_scale) {
			if (require_layer_dims_1d(g, tname, sizeof(tname), "blk.%d.layer_output_scale.weight",
									  i, 1))
				return -1;
		}
	}

	return 0;
}

static status_code upload_tensor_to(model *m, const void *host_ptr, uint32_t type, int n_dims,
									uint64_t d0, uint64_t d1, weight_class wc,
									backend *target_backend, buffer *out) {
	tensor_desc desc = {
		.type	   = type,
		.n_dims	   = n_dims,
		.dims	   = {d0, d1, 0, 0},
		.host_data = host_ptr,
	};
	backend	   *home = backend_weight_home(target_backend ? target_backend : m->backend, wc);
	status_code s	 = home->buffer_alloc_weight(home, &desc, out);
	if (s != OK)
		return s;

	return OK;
}

static status_code upload_tensor(model *m, const void *host_ptr, uint32_t type, int n_dims,
								 uint64_t d0, uint64_t d1, weight_class wc, buffer *out) {
	return upload_tensor_to(m, host_ptr, type, n_dims, d0, d1, wc, m->backend, out);
}

static void release_original_weight_data(model *m, const void *host_ptr, size_t bytes) {
	if (!host_ptr || bytes == 0)
		return;

	if (m->tie_embeddings && host_ptr == m->tok_embd.host_ptr)
		return;
	if (m->gctx.map && !m->gctx.map_is_heap) {

		madvise_dontneed(m->gctx.map, m->gctx.map_size, host_ptr, bytes);
	} else if (m->gctx.map_is_heap) {

		for (size_t i = 0; i < m->gctx.n_tensors; i++) {
			if (m->gctx.tensors[i].data == host_ptr) {
				free((void *)m->gctx.tensors[i].data);
				m->gctx.tensors[i].data = NULL;
				return;
			}
		}
	}
}
int model_should_repack(uint32_t type, const char *repack_config) {
	if (!repack_config)
		return type == GGML_TYPE_IQ3_S || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL;
	if (strcmp(repack_config, "all") == 0)
		return type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q8_0 ||
			   type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_K;
	if (strcmp(repack_config, "none") == 0)
		return 0;

	const char *name = ggml_type_name(type);
	if (!name)
		return 0;
	const char *p = repack_config;
	while (*p) {
		const char *comma = strchr(p, ',');
		size_t		len	  = comma ? (size_t)(comma - p) : strlen(p);
		if (strlen(name) == len && strncmp(p, name, len) == 0)
			return 1;
		if (!comma)
			break;
		p = comma + 1;
	}
	return 0;
}

static void repack_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	repack_job *job		  = ctx;
	int			row_begin = begin * job->rows_per_group;
	int			row_end	  = end * job->rows_per_group;
	job->repack_fn(job->src, job->dst, row_begin, row_end, job->k);
}

static void repack_weight(backend *home, const void *src, void *dst, uint32_t type, int n_rows,
						  int k) {
	repack_job job	= {.src = src, .dst = dst, .type = type, .k = k};
	tpool	  *pool = (home && home->get_pool) ? home->get_pool(home) : NULL;

	if (type == GGML_TYPE_Q8_0) {
		job.rows_per_group = Q8_0_R8_ROWS;
		job.repack_fn	   = repack_q8_0_to_q8_0_r8_rows;
	} else if (type == GGML_TYPE_Q4_0) {
		job.rows_per_group = Q4_0_R8_ROWS;
		job.repack_fn	   = repack_q4_0_to_q4_0_r8_rows;
	} else if (type == GGML_TYPE_Q4_K) {
		job.rows_per_group = Q4_K_R8_ROWS;
		job.repack_fn	   = repack_q4_k_to_q4_k_r8_rows;
	} else if (type == GGML_TYPE_IQ3_S) {
		job.rows_per_group = IQ3_S_RE8_ROWS;
		job.repack_fn	   = repack_iq3_s_to_iq3_s_re8_rows;
	} else if (type == GGML_TYPE_IQ4_NL) {
		job.rows_per_group = IQ4_NL_R8_ROWS;
		job.repack_fn	   = repack_iq4_nl_to_iq4_nl_r8_rows;
	} else {
		job.rows_per_group = 1;
		job.repack_fn	   = repack_iq4_nl_to_q8_0_rows;
	}

	int n_groups = n_rows / job.rows_per_group;
	tpool_parallel_for(pool, n_groups, 1, repack_chunk, &job);
}

static void fill_expert_desc(struct expert_desc *ed, const void *gate_w, const void *up_w,
							 const void *down_w, uint32_t gate_type, uint32_t up_type,
							 uint32_t down_type, int gate_up_fused, uint64_t gate_off,
							 uint64_t up_off, uint64_t down_off) {
	ed->gate_w		  = gate_w;
	ed->up_w		  = up_w;
	ed->down_w		  = down_w;
	ed->gate_type	  = gate_type;
	ed->up_type		  = up_type;
	ed->down_type	  = down_type;
	ed->gate_up_fused = gate_up_fused;
	ed->gate_scale	  = 1.0f;
	ed->up_scale	  = 1.0f;
	ed->down_scale	  = 1.0f;
	ed->gate_off	  = gate_off;
	ed->up_off		  = up_off;
	ed->down_off	  = down_off;
}

static status_code upload_tensor_repack_to(model *m, const void *host_ptr, uint32_t *type_io,
										   int n_dims, uint64_t d0, uint64_t d1, weight_class wc,
										   backend *target_backend, buffer *out) {
	uint32_t type = type_io ? *type_io : (host_ptr ? GGML_TYPE_F32 : 0);
	backend *home = backend_weight_home(target_backend ? target_backend : m->backend, wc);

	int home_is_cpu = (home && home->name && strcmp(home->name, "cpu") == 0);

	int do_repack = home_is_cpu && n_dims == 2 && wc == WCLASS_MATMUL && d0 > 0 && d1 > 0 &&
					model_should_repack(type, m->repack_config);

	uint32_t re_type = 0;
	if (do_repack) {
		if (type == GGML_TYPE_IQ3_S && (d0 % 256) == 0 && (d1 % IQ3_S_RE8_ROWS) == 0) {
			re_type = GGML_TYPE_IQ3_S_RE8;
		} else if (type == GGML_TYPE_IQ4_NL && (d0 % 32) == 0 && (d1 % IQ4_NL_R8_ROWS) == 0) {
			re_type = GGML_TYPE_IQ4_NL_R8;
		} else if (type == GGML_TYPE_Q8_0 && (d0 % 32) == 0 && (d1 % Q8_0_R8_ROWS) == 0) {
			re_type = GGML_TYPE_Q8_0_R8;
		} else if (type == GGML_TYPE_Q4_0 && (d0 % 32) == 0 && (d1 % Q4_0_R8_ROWS) == 0) {
			re_type = GGML_TYPE_Q4_0_R8;
		} else if (type == GGML_TYPE_Q4_K && (d0 % 256) == 0 && (d1 % Q4_K_R8_ROWS) == 0) {
			re_type = GGML_TYPE_Q4_K_R8;
		}
	}

	if (re_type) {
		int	   k			 = (int)d0;
		int	   n_rows		 = (int)d1;
		size_t src_row_bytes = ggml_row_size(type, (size_t)k);
		size_t src_total	 = src_row_bytes * (size_t)n_rows;
		size_t dst_row_bytes = ggml_row_size(re_type, (size_t)k);
		size_t dst_total	 = dst_row_bytes * (size_t)n_rows;
		int	   writable		 = m->gctx.map_is_heap;

		if (re_type == GGML_TYPE_Q4_K_R8 && dst_total == src_total && writable && host_ptr) {
			int	   rpg		   = Q4_K_R8_ROWS;
			size_t group_bytes = (size_t)rpg * src_row_bytes;
			void  *tmp		   = xmalloc(group_bytes);
			uint8_t *base	   = (uint8_t *)host_ptr;
			for (int g = 0; g < n_rows; g += rpg) {
				uint8_t *grp = base + ((size_t)g * src_row_bytes);
				memcpy(tmp, grp, group_bytes);
				repack_q4_k_to_q4_k_r8_rows(tmp, grp, 0, rpg, k);
			}
			free(tmp);
			if (type_io)
				*type_io = re_type;
			return upload_tensor_to(m, host_ptr, re_type, n_dims, d0, d1, wc, target_backend, out);
		}
		if (re_type == GGML_TYPE_Q4_K_R8 && !writable)
			re_type = 0;
	}

	if (re_type) {
		int	   k			 = (int)d0;
		int	   n_rows		 = (int)d1;
		size_t src_row_bytes = ggml_row_size(type, (size_t)k);
		size_t src_total	 = src_row_bytes * (size_t)n_rows;
		size_t dst_row_bytes = ggml_row_size(re_type, (size_t)k);
		size_t dst_total	 = dst_row_bytes * (size_t)n_rows;

		void *repacked = xmalloc_aligned(dst_total, 64);
		repack_weight(home, host_ptr, repacked, type, n_rows, k);

		tensor_desc desc = {
			.type	   = re_type,
			.n_dims	   = n_dims,
			.dims	   = {d0, d1, 0, 0},
			.host_data = repacked,
		};
		status_code s = home->buffer_alloc_weight(home, &desc, out);
		if (s != OK) {
			free(repacked);
			return s;
		}

		out->host_ptr = NULL;
		out->size	  = dst_total;

		release_original_weight_data(m, host_ptr, src_total);

		if (type_io)
			*type_io = re_type;
		return OK;
	}

	status_code s = upload_tensor_to(m, host_ptr, type, n_dims, d0, d1, wc, target_backend, out);
	if (s != OK)
		return s;
	return OK;
}

static status_code upload_tensor_repack(model *m, const void *host_ptr, uint32_t *type_io,
										int n_dims, uint64_t d0, uint64_t d1, weight_class wc,
										buffer *out) {
	return upload_tensor_repack_to(m, host_ptr, type_io, n_dims, d0, d1, wc, m->backend, out);
}

static void dequant_weight_to_f32(const void **w, uint32_t *type, size_t row_len, size_t n_rows) {
	if (*type == GGML_TYPE_F32)
		return;
	size_t		   row_bytes = ggml_row_size(*type, row_len);
	float		  *f32_buf	 = xmalloc(n_rows * row_len * sizeof(float));
	const uint8_t *src		 = *w;
	for (size_t r = 0; r < n_rows; r++)
		dequant_row_dispatch(*type, src + (r * row_bytes), (int)row_len, f32_buf + (r * row_len));
	*w	  = f32_buf;
	*type = GGML_TYPE_F32;
}

static void convert_weight_f32_to_f16(const void **w, uint32_t *type, size_t n_elems) {
	if (*type != GGML_TYPE_F32)
		return;
	const float *src = *w;
	uint16_t *dst = xmalloc_aligned(n_elems * sizeof(uint16_t), 64);
	for (size_t i = 0; i < n_elems; i++)
		dst[i] = f32_to_f16(src[i]);
	*w = dst;
	*type = GGML_TYPE_F16;
}

static void *build_fused_gate_up(model *m, const void *gate_w, const void *up_w, uint32_t type,
								 uint64_t dim, uint64_t intermediate) {
	size_t	 row_stride = ggml_row_size(type, dim);
	size_t	 half_bytes = row_stride * intermediate;
	uint8_t *buf		= xmalloc(half_bytes * 2);
	memcpy(buf, gate_w, half_bytes);
	memcpy(buf + half_bytes, up_w, half_bytes);

	if (m->use_mmap && m->gctx.map && m->gctx.map_size > 0) {
		madvise_dontneed(m->gctx.map, m->gctx.map_size, gate_w, half_bytes);
		madvise_dontneed(m->gctx.map, m->gctx.map_size, up_w, half_bytes);
	}

	return buf;
}

static status_code upload_one(model *m, weight_ref *ref, uint32_t wtype, int ndims, uint64_t d0,
							  uint64_t d1, weight_class wc) {
	ref->type = wtype;
	return upload_tensor(m, ref->host_ptr, ref->type, ndims, d0, d1, wc, &ref->buf);
}

static status_code upload_one_repack(model *m, weight_ref *ref, int ndims, uint64_t d0, uint64_t d1,
									 weight_class wc) {
	return upload_tensor_repack(m, ref->host_ptr, &ref->type, ndims, d0, d1, wc, &ref->buf);
}

static void upload_embeddings(model *m) {
	status_code s;

	s = upload_one(m, &m->tok_embd, m->tok_embd.type, 2, m->dim, m->vocab_size, WCLASS_EMBEDDING);
	if (s != OK)
		return;
	s = upload_one(m, &m->output_norm_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_NORM);
	if (s != OK)
		return;

	s = upload_one_repack(m, &m->output_w, 2, m->dim, m->vocab_size, WCLASS_MATMUL);
	if (s != OK)
		return;

	if (m->has_per_layer_embeddings) {
		if (m->backend && strcmp(m->backend->name, "cpu") == 0) {
			s = upload_one(m, &m->layer_dims.per_layer_tok_embd,
						   m->layer_dims.per_layer_tok_embd.type, 2,
						   (uint64_t)m->layer_dims.n_embd_per_layer * m->n_layers, m->vocab_size,
						   WCLASS_EMBEDDING);
			if (s != OK)
				return;
		} else {
			m->layer_dims.per_layer_tok_embd.buf.handle = NULL;
			m->layer_dims.per_layer_tok_embd.buf.host_ptr =
				m->layer_dims.per_layer_tok_embd.host_ptr;
			m->layer_dims.per_layer_tok_embd.buf.owner = NULL;
		}
		if (m->layer_dims.per_layer_model_proj.type == GGML_TYPE_BF16 && m->backend &&
			strcmp(m->backend->name, "cpu") == 0) {
			s = upload_one(m, &m->layer_dims.per_layer_model_proj, GGML_TYPE_BF16, 2, m->dim,
						   (uint64_t)m->layer_dims.n_embd_per_layer * m->n_layers, WCLASS_MATMUL);
			if (s != OK)
				return;
		} else if (m->layer_dims.per_layer_model_proj.type == GGML_TYPE_BF16) {
			size_t n_elems = (size_t)m->dim * (size_t)m->layer_dims.n_embd_per_layer * m->n_layers;
			float *f32_buf = xmalloc(n_elems * sizeof(float));
			dequant_bf16_row(m->layer_dims.per_layer_model_proj.host_ptr, (int)n_elems, f32_buf);
			if (m->gctx.map && m->gctx.map_size > 0) {
				size_t bf16_bytes = n_elems * sizeof(uint16_t);
				madvise_dontneed(m->gctx.map, m->gctx.map_size,
								 m->layer_dims.per_layer_model_proj.host_ptr, bf16_bytes);
			}
			m->layer_dims.per_layer_model_proj.host_ptr = f32_buf;
			s = upload_one(m, &m->layer_dims.per_layer_model_proj, GGML_TYPE_F32, 2, m->dim,
						   (uint64_t)m->layer_dims.n_embd_per_layer * m->n_layers, WCLASS_MATMUL);
			if (s != OK)
				return;
		} else {
			s = upload_one_repack(m, &m->layer_dims.per_layer_model_proj, 2, m->dim,
								  (uint64_t)m->layer_dims.n_embd_per_layer * m->n_layers,
								  WCLASS_MATMUL);
			if (s != OK)
				return;
		}
		s = upload_one(m, &m->layer_dims.per_layer_proj_norm_w, GGML_TYPE_F32, 1,
					   m->layer_dims.n_embd_per_layer, 0, WCLASS_NORM);
		if (s != OK)
			return;
	}
	if (m->rope_freqs_count > 0) {
		m->rope_freqs_w.host_ptr = m->rope_freqs;
		s = upload_one(m, &m->rope_freqs_w, GGML_TYPE_F32, 1, m->rope_freqs_count, 0, WCLASS_MISC);
		if (s != OK)
			return;
	}
}

static status_code upload_layer_weights(model *m, int i, progress *prog) {
	status_code	   s;
	layer_weights *L		= &m->layers[i];
	backend		  *layer_be = model_layer_backend(m, i);

	int q_out		 = m->n_heads * model_layer_head_dim(m, i);
	int q_weight_out = m->arch_info->has_attn_output_gate ? 2 * q_out : q_out;
	int kv_out		 = model_layer_kv_heads(m, i) * model_layer_head_dim(m, i);
	int intermediate = model_layer_intermediate(m, i);
	int is_mla_layer = m->arch_info->is_mla;
	int is_moe_layer = model_layer_is_moe(m, i);

#define UPLOAD(ref, wtype, ndims, d0, d1, wc)                                                      \
	do {                                                                                           \
		(ref)->type = (wtype);                                                                     \
		s = upload_tensor_to(m, (ref)->host_ptr, (ref)->type, (ndims), (d0), (d1), (wc), layer_be, \
							 &(ref)->buf);                                                         \
		if (s != OK)                                                                               \
			return s;                                                                              \
	} while (0)

#define UPLOAD_REP(ref, ndims, d0, d1, wc)                                                         \
	do {                                                                                           \
		s = upload_tensor_repack_to(m, (ref)->host_ptr, &(ref)->type, (ndims), (d0), (d1), (wc),   \
									layer_be, &(ref)->buf);                                        \
		if (s != OK)                                                                               \
			return s;                                                                              \
	} while (0)

	UPLOAD(&L->attn_norm_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_NORM);

	if (model_layer_is_recurrent(m, i)) {
		UPLOAD_REP(&L->attn_qkv_w, 2, m->dim, m->qwen35.conv_dim, WCLASS_MATMUL);
		UPLOAD_REP(&L->attn_gate_w, 2, m->dim, m->qwen35.value_dim, WCLASS_MATMUL);
		UPLOAD(&L->ssm_conv1d_w, GGML_TYPE_F32, 2, m->qwen35.conv_kernel,
			   m->qwen35.conv_dim, WCLASS_MISC);
		UPLOAD(&L->ssm_dt_b, GGML_TYPE_F32, 1, m->qwen35.n_value_heads, 0, WCLASS_MISC);
		UPLOAD(&L->ssm_a, GGML_TYPE_F32, 1, m->qwen35.n_value_heads, 0, WCLASS_MISC);
		UPLOAD_REP(&L->ssm_beta_w, 2, m->dim, m->qwen35.n_value_heads, WCLASS_MATMUL);
		UPLOAD_REP(&L->ssm_alpha_w, 2, m->dim, m->qwen35.n_value_heads, WCLASS_MATMUL);
		UPLOAD(&L->ssm_norm_w, GGML_TYPE_F32, 1, m->qwen35.value_head_dim, 0, WCLASS_NORM);
		UPLOAD_REP(&L->ssm_out_w, 2, m->qwen35.value_dim, m->dim, WCLASS_MATMUL);
		memset(&L->wq.buf, 0, sizeof(L->wq.buf));
		memset(&L->wk.buf, 0, sizeof(L->wk.buf));
		memset(&L->wv.buf, 0, sizeof(L->wv.buf));
		memset(&L->wo.buf, 0, sizeof(L->wo.buf));
	} else if (is_mla_layer) {
		int q_lora	  = m->mla.q_lora;
		int kv_lora	  = m->mla.kv_lora;
		int kv_a_rows = kv_lora + m->mla.qk_rope;
		int q_b_rows  = m->n_heads * m->mla.qk_head;
		UPLOAD_REP(&L->q_a_w, 2, m->dim, q_lora, WCLASS_MATMUL);
		UPLOAD_REP(&L->q_b_w, 2, q_lora, q_b_rows, WCLASS_MATMUL);
		UPLOAD(&L->q_a_norm_w, GGML_TYPE_F32, 1, q_lora, 0, WCLASS_NORM);
		UPLOAD_REP(&L->kv_a_w, 2, m->dim, kv_a_rows, WCLASS_MATMUL);
		UPLOAD_REP(&L->k_b_w, 2, (uint64_t)m->mla.qk_nope * m->mla.kv_lora, m->n_heads,
				   WCLASS_MATMUL);
		UPLOAD_REP(&L->v_b_w, 2, (uint64_t)m->mla.kv_lora * m->mla.v_head, m->n_heads,
				   WCLASS_MATMUL);
		UPLOAD(&L->kv_a_norm_w, GGML_TYPE_F32, 1, kv_lora, 0, WCLASS_NORM);
		int wo_in = m->n_heads * m->mla.v_head;
		UPLOAD_REP(&L->wo, 2, wo_in, m->dim, WCLASS_MATMUL);
		memset(&L->wq.buf, 0, sizeof(L->wq.buf));
		memset(&L->wk.buf, 0, sizeof(L->wk.buf));
		memset(&L->wv.buf, 0, sizeof(L->wv.buf));
	} else {
		UPLOAD_REP(&L->wq, 2, m->dim, q_weight_out, WCLASS_MATMUL);
		if (model_layer_has_kv(m, i)) {
			UPLOAD_REP(&L->wk, 2, m->dim, kv_out, WCLASS_MATMUL);
		} else {
			memset(&L->wk.buf, 0, sizeof(L->wk.buf));
		}
		if (L->has_own_v) {
			UPLOAD_REP(&L->wv, 2, m->dim, kv_out, WCLASS_MATMUL);
		} else {
			memset(&L->wv.buf, 0, sizeof(L->wv.buf));
		}
		UPLOAD_REP(&L->wo, 2, q_out, m->dim, WCLASS_MATMUL);
	}
	if (!m->arch_info->uses_post_attn_norm_for_ffn)
		UPLOAD(&L->ffn_norm_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_NORM);

	if (m->arch_info->has_attn_post_norm) {
		UPLOAD(&L->post_attn_norm_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_NORM);
	}
	if (m->arch_info->has_ffn_post_norm) {
		UPLOAD(&L->post_ffn_norm_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_NORM);
	}

	if (m->arch_info->has_qk_norm && !model_layer_is_recurrent(m, i)) {
		UPLOAD(&L->attn_q_norm_w, GGML_TYPE_F32, 1, L->head_dim, 0, WCLASS_NORM);
		if (model_layer_has_kv(m, i))
			UPLOAD(&L->attn_k_norm_w, GGML_TYPE_F32, 1, L->head_dim, 0, WCLASS_NORM);
	}
	if (m->has_per_layer_embeddings) {
		UPLOAD(&L->ple_post_norm_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_NORM);
		int gate_owned = 0;
		if (layer_be && strcmp(layer_be->name, "cpu") == 0 &&
			L->ple_inp_gate_w.type == GGML_TYPE_F32) {
			convert_weight_f32_to_f16(&L->ple_inp_gate_w.host_ptr, &L->ple_inp_gate_w.type,
									 (size_t)m->layer_dims.n_embd_per_layer * (size_t)m->dim);
			gate_owned = 1;
		} else if (L->ple_inp_gate_w.type != GGML_TYPE_F32) {
			dequant_weight_to_f32(&L->ple_inp_gate_w.host_ptr, &L->ple_inp_gate_w.type,
								  (size_t)m->layer_dims.n_embd_per_layer, (size_t)m->dim);
			gate_owned = 1;
		}
		UPLOAD(&L->ple_inp_gate_w, L->ple_inp_gate_w.type, 2, m->dim,
			   m->layer_dims.n_embd_per_layer, WCLASS_MATMUL);
		if (gate_owned)
			L->ple_inp_gate_w.buf.host_ptr = NULL;
		int proj_owned = 0;
		if (layer_be && strcmp(layer_be->name, "cpu") == 0 && L->ple_proj_w.type == GGML_TYPE_F32) {
			convert_weight_f32_to_f16(&L->ple_proj_w.host_ptr, &L->ple_proj_w.type,
									 (size_t)m->dim * (size_t)m->layer_dims.n_embd_per_layer);
			proj_owned = 1;
		} else if (L->ple_proj_w.type != GGML_TYPE_F32) {
			dequant_weight_to_f32(&L->ple_proj_w.host_ptr, &L->ple_proj_w.type, (size_t)m->dim,
								  (size_t)m->layer_dims.n_embd_per_layer);
			proj_owned = 1;
		}
		UPLOAD(&L->ple_proj_w, L->ple_proj_w.type, 2, m->layer_dims.n_embd_per_layer, m->dim,
			   WCLASS_MATMUL);
		if (proj_owned)
			L->ple_proj_w.buf.host_ptr = NULL;
	}
	if (m->arch_info->has_layer_output_scale) {
		UPLOAD(&L->layer_out_scale_w, GGML_TYPE_F32, 1, 1, 0, WCLASS_MISC);
		const float *p	   = L->layer_out_scale_w.host_ptr;
		L->layer_out_scale = p ? p[0] : 1.0f;
	}

	L->gate_up_fused	  = 0;
	L->gate_up_fused_host = NULL;
	memset(&L->gate_w.buf, 0, sizeof(L->gate_w.buf));
	memset(&L->up_w.buf, 0, sizeof(L->up_w.buf));
	memset(&L->down_w.buf, 0, sizeof(L->down_w.buf));

	if (is_moe_layer) {
		UPLOAD_REP(&L->router_w, 2, m->dim, m->moe.n_experts, WCLASS_MATMUL);
		if (L->router_bias.host_ptr) {
			UPLOAD(&L->router_bias, GGML_TYPE_F32, 1, m->moe.n_experts, 0, WCLASS_MISC);
		} else {
			memset(&L->router_bias.buf, 0, sizeof(L->router_bias.buf));
		}
		if (m->moe.n_shared_experts > 0) {
			int sh_inter = m->moe.moe_intermediate * m->moe.n_shared_experts;
			if (L->shexp_gate_w.type == L->shexp_up_w.type) {
				void *fused =
					build_fused_gate_up(m, L->shexp_gate_w.host_ptr, L->shexp_up_w.host_ptr,
										L->shexp_gate_w.type, (uint64_t)m->dim, (uint64_t)sh_inter);
				uint32_t fused_type = L->shexp_gate_w.type;
				s = upload_tensor_repack_to(m, fused, &fused_type, 2, m->dim,
											(uint64_t)2 * sh_inter, WCLASS_MATMUL, layer_be,
											&L->shexp_gate_w.buf);
				if (s != OK) {
					free(fused);
					return s;
				}
				L->shexp_gate_w.type = fused_type;
				L->shexp_up_w.type	 = fused_type;
				if (L->shexp_gate_w.buf.host_ptr != fused) {
					free(fused);
					L->shexp_fused_host = NULL;
				} else {
					L->shexp_fused_host = fused;
				}
				L->shexp_up_w.buf = L->shexp_gate_w.buf;
				L->shexp_up_w.buf.offset =
					(size_t)sh_inter * ggml_row_size(L->shexp_gate_w.type, m->dim);

				if (L->shexp_up_w.buf.handle)
					L->shexp_up_w.buf.host_ptr = L->shexp_up_w.buf.handle;
			} else {
				UPLOAD_REP(&L->shexp_gate_w, 2, m->dim, sh_inter, WCLASS_MATMUL);
				UPLOAD_REP(&L->shexp_up_w, 2, m->dim, sh_inter, WCLASS_MATMUL);
			}
			UPLOAD_REP(&L->shexp_down_w, 2, sh_inter, m->dim, WCLASS_MATMUL);
		}
		if (L->router_scale_w.host_ptr) {
			UPLOAD(&L->router_scale_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_MISC);
		} else {
			memset(&L->router_scale_w.buf, 0, sizeof(L->router_scale_w.buf));
		}
		if (L->ffn_pre_norm_2_w.host_ptr) {
			UPLOAD(&L->ffn_pre_norm_2_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_NORM);
		}
		if (L->ffn_post_norm_1_w.host_ptr) {
			UPLOAD(&L->ffn_post_norm_1_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_NORM);
		}
		if (L->ffn_post_norm_2_w.host_ptr) {
			UPLOAD(&L->ffn_post_norm_2_w, GGML_TYPE_F32, 1, m->dim, 0, WCLASS_NORM);
		}
		if (m->arch_info->uses_moe_shared_dense_ffn) {
			UPLOAD_REP(&L->gate_w, 2, m->dim, intermediate, WCLASS_MATMUL);
			UPLOAD_REP(&L->up_w, 2, m->dim, intermediate, WCLASS_MATMUL);
			UPLOAD_REP(&L->down_w, 2, intermediate, m->dim, WCLASS_MATMUL);
		}
	} else {
		if (L->gate_w.type == L->up_w.type && !m->arch_info->has_variable_layer_dims) {
			void *fused =
				build_fused_gate_up(m, L->gate_w.host_ptr, L->up_w.host_ptr, L->gate_w.type,
									(uint64_t)m->dim, (uint64_t)intermediate);
			uint32_t fused_type = L->gate_w.type;
			s = upload_tensor_repack_to(m, fused, &fused_type, 2, m->dim,
										(uint64_t)2 * intermediate, WCLASS_MATMUL, layer_be,
										&L->gate_up_w.buf);
			if (s == OK) {
				L->gate_up_w.type = fused_type;
				L->gate_up_fused  = 1;
				if (L->gate_up_w.buf.host_ptr == fused) {
					L->gate_up_fused_host = fused;
				} else {
					free(fused);
					L->gate_up_fused_host = NULL;
				}
			} else {
				free(fused);
				return s;
			}
		}
		if (!L->gate_up_fused) {
			UPLOAD_REP(&L->gate_w, 2, m->dim, intermediate, WCLASS_MATMUL);
			UPLOAD_REP(&L->up_w, 2, m->dim, intermediate, WCLASS_MATMUL);
		}
		UPLOAD_REP(&L->down_w, 2, intermediate, m->dim, WCLASS_MATMUL);
	}

#undef UPLOAD
#undef UPLOAD_REP
	progress_update(prog, (uint64_t)i + 1);
	return OK;
}

static status_code upload_all_weights(model *m) {
	status_code s;

	upload_embeddings(m);

	progress prog;
	progress_start(&prog, "Preparing weights", (uint64_t)model_n_all_layers(m));

	if (m->gctx.fd >= 0 && m->gctx.map && m->gctx.n_tensors > 0) {
		uint64_t ra_t0 = (g_monitor && g_monitor->fd >= 0) ? time_us() : 0;
		long	 ps	   = sysconf(_SC_PAGESIZE);
		if (ps <= 0)
			ps = 4096;
		uintptr_t pm   = ~((uintptr_t)ps - 1);
		uintptr_t base = (uintptr_t)m->gctx.map;
		uintptr_t stop = base + m->gctx.map_size;
		for (size_t ti = 0; ti < m->gctx.n_tensors; ti++) {
			const gguf_tensor *gt = &m->gctx.tensors[ti];
			if (gguf_tensor_name_is_expert(gt->name))
				continue;
			size_t tbytes;
			if (gguf_tensor_byte_size(gt, &tbytes) != OK || tbytes == 0)
				continue;
			uintptr_t a = (uintptr_t)gt->data;
			uintptr_t b = a + tbytes;
			if (a >= base && b <= stop && b >= a)
				readahead(m->gctx.fd, (off_t)((a & pm) - base), ((b + ps - 1) & pm) - (a & pm));
		}
		if (g_monitor && g_monitor->fd >= 0) {
			monitor_send(g_monitor,
						 "{\"type\":\"load\",\"phase\":\"dense_readahead_done\",\"ms\":%llu}",
						 (unsigned long long)((time_us() - ra_t0) / 1000));
			monitor_poll(g_monitor);
		}
	}

	for (int i = 0; i < m->n_layers; i++) {
		if (g_monitor && g_monitor->fd >= 0) {
			monitor_send(g_monitor,
						 "{\"type\":\"load\",\"phase\":\"loading_weights\","
						 "\"layer\":%d,\"n_layers\":%d,\"pct\":%.1f}",
						 i, m->n_layers, 100.0 * (double)i / (double)m->n_layers);
			monitor_poll(g_monitor);
		}

		s = upload_layer_weights(m, i, &prog);
		if (s != OK) {
			progress_finish(&prog);
			return s;
		}
	}

	if (m->n_layer_nextn == 1) {
		s = upload_layer_weights(m, m->n_layers, &prog);
		if (s != OK) {
			progress_finish(&prog);
			return s;
		}
		backend *be = m->backend;
		s = upload_tensor_repack_to(m, m->mtp_eh_proj.host_ptr, &m->mtp_eh_proj.type, 2,
									(uint64_t)(2 * m->dim), (uint64_t)m->dim, WCLASS_MATMUL, be,
									&m->mtp_eh_proj.buf);
		if (s != OK) {
			progress_finish(&prog);
			return s;
		}
		m->mtp_enorm.type = GGML_TYPE_F32;
		s = upload_tensor_to(m, m->mtp_enorm.host_ptr, m->mtp_enorm.type, 1, (uint64_t)m->dim, 0,
							 WCLASS_NORM, be, &m->mtp_enorm.buf);
		if (s != OK) {
			progress_finish(&prog);
			return s;
		}
		m->mtp_hnorm.type = GGML_TYPE_F32;
		s = upload_tensor_to(m, m->mtp_hnorm.host_ptr, m->mtp_hnorm.type, 1, (uint64_t)m->dim, 0,
							 WCLASS_NORM, be, &m->mtp_hnorm.buf);
		if (s != OK) {
			progress_finish(&prog);
			return s;
		}
		if (m->mtp_shared_head_norm.host_ptr) {
			m->mtp_shared_head_norm.type = GGML_TYPE_F32;
			s = upload_tensor_to(m, m->mtp_shared_head_norm.host_ptr, m->mtp_shared_head_norm.type,
								 1, (uint64_t)m->dim, 0, WCLASS_NORM, be,
								 &m->mtp_shared_head_norm.buf);
			if (s != OK) {
				progress_finish(&prog);
				return s;
			}
		}
	}

	progress_finish(&prog);

	return OK;
}

static status_code load_gemma4_metadata(model *m, const gguf_ctx *g, const char *prefix) {
	int32_t v32;
	float	vf;

	if (akey_i32(g, prefix, "attention.key_length_swa", &v32) == OK) {
		m->layer_dims.head_dim_swa = v32;
	} else {
		m->layer_dims.head_dim_swa = m->dim / m->n_heads;
	}
	if (akey_i32(g, prefix, "attention.key_length", &v32) == OK) {
		m->layer_dims.head_dim_global = v32;
	} else {
		m->layer_dims.head_dim_global = m->dim / m->n_heads;
	}
	m->head_dim = m->layer_dims.head_dim_global;

	if (akey_f32(g, prefix, "rope.freq_base_swa", &vf) == OK) {
		m->layer_dims.rope_theta_swa = vf;
	} else {
		m->layer_dims.rope_theta_swa = 10000.0f;
	}
	m->layer_dims.rope_theta_global = m->rope_theta;

	if (akey_i32(g, prefix, "embedding_length_per_layer_input", &v32) == OK) {
		m->layer_dims.n_embd_per_layer = v32;
	} else {
		m->layer_dims.n_embd_per_layer = 0;
	}
	m->has_per_layer_embeddings = (m->layer_dims.n_embd_per_layer > 0);

	if (akey_i32(g, prefix, "rope.dimension_count", &v32) == OK) {
		m->layer_dims.rope_dim_global = v32;
	} else {
		m->layer_dims.rope_dim_global = m->layer_dims.head_dim_global;
	}
	if (akey_i32(g, prefix, "rope.dimension_count_swa", &v32) == OK) {
		m->layer_dims.rope_dim_swa = v32;
	} else {
		m->layer_dims.rope_dim_swa = m->layer_dims.head_dim_swa;
	}
	m->rope_dim = m->layer_dims.rope_dim_global;

	{
		char key1[128];
		char key2[128];
		snprintf(key1, sizeof(key1), "%s.attention.sliding_window_pattern", prefix);
		snprintf(key2, sizeof(key2), "%s.attention.layer_types", prefix);
		const int32_t *pattern;
		size_t		   n_pattern;
		if (gguf_get_arr_i32(g, key1, &pattern, &n_pattern) == OK ||
			gguf_get_arr_i32(g, key2, &pattern, &n_pattern) == OK) {
			m->layer_dims.is_global_layer = xcalloc(m->n_layers, sizeof(uint8_t));
			for (int i = 0; i < m->n_layers && i < (int)n_pattern; i++) {
				m->layer_dims.is_global_layer[i] = pattern[i] ? 0 : 1;
			}
		} else {
			m->layer_dims.is_global_layer = xcalloc(m->n_layers, sizeof(uint8_t));
			int period					  = m->arch_info->sliding_window_period;
			for (int i = 0; i < m->n_layers; i++) {
				m->layer_dims.is_global_layer[i] = ((i + 1) % period == 0) ? 1 : 0;
			}
			for (size_t i = 0; i < g->n_kv; i++) {
				if (strcmp(g->kv_keys[i], key1) == 0 && g->kv_types[i] == GGUF_TYPE_ARRAY &&
					g->kv_arr_type[i] == GGUF_TYPE_BOOL) {
					const uint8_t *bools = g->kv_arr_data[i];
					for (int j = 0; j < m->n_layers && j < (int)g->kv_arr_len[i]; j++) {
						m->layer_dims.is_global_layer[j] = bools[j] ? 0 : 1;
					}
					break;
				}
			}
		}
	}

	{
		char ffn_key[128];
		snprintf(ffn_key, sizeof(ffn_key), "%s.feed_forward_length", prefix);
		const int32_t *ffn_lens;
		size_t		   n_ffn;
		m->layer_dims.ffn_lengths = xcalloc(m->n_layers, sizeof(int));
		if (gguf_get_arr_i32(g, ffn_key, &ffn_lens, &n_ffn) == OK) {
			for (int i = 0; i < m->n_layers && i < (int)n_ffn; i++) {
				m->layer_dims.ffn_lengths[i] = ffn_lens[i];
			}
		} else {
			for (int i = 0; i < m->n_layers; i++) {
				m->layer_dims.ffn_lengths[i] = m->intermediate;
			}
		}
	}

	{
		char		   kv_key[128];
		const int32_t *kv_arr = NULL;
		size_t		   kv_n	  = 0;
		int32_t		   kv_scalar;
		snprintf(kv_key, sizeof(kv_key), "%s.attention.head_count_kv", prefix);
		m->layer_dims.n_kv_heads_per_layer = xcalloc(m->n_layers, sizeof(int));
		if (gguf_get_arr_i32(g, kv_key, &kv_arr, &kv_n) == OK && kv_n > 0) {
			for (int i = 0; i < m->n_layers; i++) {
				m->layer_dims.n_kv_heads_per_layer[i] =
					(i < (int)kv_n) ? kv_arr[i] : kv_arr[kv_n - 1];
			}
			char  summary[128];
			char *p	   = summary;
			char *end  = summary + sizeof(summary);
			int	  show = m->n_layers < 6 ? m->n_layers : 6;
			p += snprintf(p, (size_t)(end - p), "[");
			for (int i = 0; i < show && p < end; i++) {
				p += snprintf(p, (size_t)(end - p), "%s%d", i ? ", " : "",
							  m->layer_dims.n_kv_heads_per_layer[i]);
			}
			if (m->n_layers > show && p < end)
				p += snprintf(p, (size_t)(end - p), ", ...");
			if (p < end)
				snprintf(p, (size_t)(end - p), "]");
			DEBUG("n_kv_heads per layer: %s", summary);
		} else if (akey_i32(g, prefix, "attention.head_count_kv", &kv_scalar) == OK) {
			for (int i = 0; i < m->n_layers; i++) {
				m->layer_dims.n_kv_heads_per_layer[i] = kv_scalar;
			}
			DEBUG("n_kv_heads = %d (all layers)", kv_scalar);
		} else {
			for (int i = 0; i < m->n_layers; i++) {
				m->layer_dims.n_kv_heads_per_layer[i] = m->n_kv_heads;
			}
			DEBUG("n_kv_heads = %d (no per-layer metadata)", m->n_kv_heads);
		}
	}

	{
		const gguf_tensor *t = gguf_find_tensor(g, "rope_freqs.weight");
		if (t && t->data) {
			m->rope_freqs		= t->data;
			m->rope_freqs_count = (int)t->dims[0];
			int			 n_rot	= 0;
			const float *ff		= (const float *)t->data;
			for (int i = 0; i < m->rope_freqs_count; i++) {
				if (ff[i] < 1e10f)
					n_rot++;
			}
			n_rot *= 2;
			DEBUG("rope_freqs: %d values, n_rot=%d", m->rope_freqs_count, n_rot);
		} else {
			m->rope_freqs		= NULL;
			m->rope_freqs_count = 0;
		}
	}

	if (akey_i32(g, prefix, "attention.sliding_window", &v32) == OK) {
		m->sliding_window = v32;
	}

	{
		int32_t n_kv_shared = 0;
		if (akey_i32(g, prefix, "attention.shared_kv_layers", &n_kv_shared) == OK) {
			m->layer_dims.n_layer_kv_from_start = m->n_layers - (int)n_kv_shared;
		} else {
			m->layer_dims.n_layer_kv_from_start = m->n_layers;
		}
		m->layer_dims.kv_layer_swa	  = -1;
		m->layer_dims.kv_layer_global = -1;
		for (int i = 0; i < m->layer_dims.n_layer_kv_from_start && i < m->n_layers; i++) {
			if (m->layer_dims.is_global_layer[i]) {
				m->layer_dims.kv_layer_global = i;
			} else {
				m->layer_dims.kv_layer_swa = i;
			}
		}
		DEBUG("KV sharing: kv_layers=%d, swa=%d, global=%d", m->layer_dims.n_layer_kv_from_start,
			  m->layer_dims.kv_layer_swa, m->layer_dims.kv_layer_global);

		int needs_swa_target	= 0;
		int needs_global_target = 0;
		for (int i = m->layer_dims.n_layer_kv_from_start; i < m->n_layers; i++) {
			if (m->layer_dims.is_global_layer[i])
				needs_global_target = 1;
			else
				needs_swa_target = 1;
		}
		if ((needs_swa_target && m->layer_dims.kv_layer_swa < 0) ||
			(needs_global_target && m->layer_dims.kv_layer_global < 0)) {
			ERROR("KV sharing: shared_kv_layers=%d leaves no local KV-store layer of a "
				  "type used by later shared layers (swa_target=%d found=%d, "
				  "global_target=%d found=%d)",
				  (int)n_kv_shared, needs_swa_target, m->layer_dims.kv_layer_swa,
				  needs_global_target, m->layer_dims.kv_layer_global);
			return ERR_FORMAT;
		}
	}

	return OK;
}

static int expert_scale_f32(const gguf_ctx *g, const char *name, int n_experts,
							const float **out_scale) {
	const gguf_tensor *s = gguf_find_tensor(g, name);
	if (s && s->data && s->type == GGML_TYPE_F32 && s->n_dims == 1 &&
		(int)s->dims[0] == n_experts) {
		*out_scale = (const float *)s->data;
		return 1;
	}
	return 0;
}

status_code model_load(model *m, const char *path) {
	backend	   *cpu;
	status_code s = backend_create("cpu", 0, &cpu);
	if (s != OK)
		return s;

	s = model_load_backend_ex_repack(m, path, cpu, 1, NULL, 0);
	if (s != OK) {
		backend_destroy(cpu);
		return s;
	}
	m->owns_backend = 1;
	return OK;
}

static void *prefetch_worker(void *arg) {
	prefetch_range *r	  = arg;
	size_t			chunk = (size_t)128u * 1024 * 1024;
	if (chunk > r->len)
		chunk = r->len;
	for (size_t off = 0; off < r->len; off += chunk) {
		size_t n = r->len - off < chunk ? r->len - off : chunk;
		readahead(r->fd, r->offset + (off_t)off, n);
	}
	return NULL;
}

static void model_prefetch_mmap(gguf_ctx *g) {
	if (!g || g->fd < 0 || !g->map || g->map_size == 0 || g->map_is_heap)
		return;

	long n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
	if (n_cpu < 1)
		n_cpu = 1;
	int n_threads = (int)n_cpu * 4;
	if (n_threads > PREFETCH_MAX_THREADS)
		n_threads = PREFETCH_MAX_THREADS;

	size_t total = g->map_size;
	if (total / (size_t)n_threads < PREFETCH_CHUNK_MIN) {
		n_threads = (int)(total / PREFETCH_CHUNK_MIN);
		if (n_threads < 1)
			n_threads = 1;
	}

	pthread_t	   threads[PREFETCH_MAX_THREADS];
	prefetch_range ranges[PREFETCH_MAX_THREADS];
	size_t		   base_len	 = total / (size_t)n_threads;
	size_t		   off		 = 0;
	int			   n_spawned = 0;

	for (int i = 0; i < n_threads; i++) {
		size_t len = (i == n_threads - 1) ? (total - off) : base_len;
		ranges[i]  = (prefetch_range){.fd = g->fd, .offset = (off_t)off, .len = len};
		off += len;
		if (i == 0)
			continue;
		if (pthread_create(&threads[i], NULL, prefetch_worker, &ranges[i]) == 0)
			n_spawned++;
		else
			threads[i] = 0;
	}

	prefetch_worker(&ranges[0]);

	for (int i = 1; i < n_threads; i++) {
		if (threads[i])
			pthread_join(threads[i], NULL);
	}
	(void)n_spawned;
}

static status_code load_moe_metadata(model *m, const gguf_ctx *g, const char *prefix) {
	int32_t v32;
	if (m->arch_info->is_moe || akey_i32(g, prefix, "expert_count", &v32) == OK) {
		if (m->arch_info->is_moe) {
			if (req_i32(g, prefix, "expert_count", &v32) != OK) {
				return ERR_FORMAT;
			}
			m->moe.n_experts = v32;
		} else {
			m->moe.n_experts = v32;
		}
		if (req_i32(g, prefix, "expert_used_count", &v32) != OK) {
			if (akey_i32(g, prefix, "expert_shared_count", &v32) == OK) {
				m->moe.n_experts_used = v32;
			} else {
				ERROR("model_load: missing '%s.expert_used_count'", prefix);
				return ERR_FORMAT;
			}
		} else {
			m->moe.n_experts_used = v32;
		}
		if (akey_i32(g, prefix, "expert_shared_count", &v32) == OK) {
			m->moe.n_shared_experts = v32;
		} else {
			m->moe.n_shared_experts = m->arch_info->has_shared_expert ? 1 : 0;
		}
		if (akey_i32(g, prefix, "expert_feed_forward_length", &v32) == OK ||
			akey_i32(g, prefix, "moe_intermediate_size", &v32) == OK) {
			m->moe.moe_intermediate = v32;
		} else {
			m->moe.moe_intermediate = m->intermediate;
		}
		if (akey_i32(g, prefix, "expert_group_count", &v32) == OK) {
			m->moe.n_group = v32;
		} else {
			m->moe.n_group = 1;
		}
		if (akey_i32(g, prefix, "expert_group_used_count", &v32) == OK) {
			m->moe.topk_group = v32;
		} else {
			m->moe.topk_group = 1;
		}
		float rsf;
		if (akey_f32(g, prefix, "expert_weights_scale", &rsf) == OK ||
			akey_f32(g, prefix, "routed_scaling_factor", &rsf) == OK) {
			m->moe.routed_scale = rsf;
		} else {
			m->moe.routed_scale = 1.0f;
		}
		int	 bnorm;
		int	 bval;
		char norm_key[160];
		snprintf(norm_key, sizeof(norm_key), "%s.expert_weights_norm", prefix);
		if (gguf_get_bool(g, norm_key, &bval) == OK ||
			gguf_get_bool(g, "norm_topk_prob", &bval) == OK) {
			bnorm = bval ? 1 : 0;
		} else if (akey_i32(g, prefix, "expert_weights_norm", &v32) == OK ||
				   akey_i32(g, prefix, "norm_topk_prob", &v32) == OK) {
			bnorm = v32 ? 1 : 0;
		} else {
			bnorm = m->arch_info->uses_moe_norm_topk_prob;
		}
		if ((m->arch == ARCH_GEMMA4 || m->arch == ARCH_GEMMA4_MOE) && m->moe.n_experts > 0) {
			if (!bnorm) {
				DEBUG("norm_topk_prob forced to 1 (gemma4_moe convention)");
				bnorm = 1;
			}
		}
		m->moe.norm_topk_prob	 = bnorm;
		m->moe.first_dense_layer = m->arch_info->moe_first_dense_layers;
		if (akey_i32(g, prefix, "leading_dense_block_count", &v32) == OK ||
			akey_i32(g, prefix, "first_dense_layer_count", &v32) == OK) {
			m->moe.first_dense_layer = v32;
		}
		m->moe.router_dim_scale = 1.0f / sqrtf((float)m->dim);

		DEBUG("MoE: %d experts, topk=%d, shared=%d, inter=%d, scale=%g", m->moe.n_experts,
			  m->moe.n_experts_used, m->moe.n_shared_experts, m->moe.moe_intermediate,
			  m->moe.routed_scale);

		if (m->arch == ARCH_GEMMA4 && m->moe.n_experts > 0) {
			m->arch		 = ARCH_GEMMA4_MOE;
			m->arch_info = arch_lookup(ARCH_GEMMA4_MOE);
			DEBUG("arch: gemma4 -> gemma4_moe (%d experts)", m->moe.n_experts);
		}
	}
	return OK;
}

static void load_mla_metadata(model *m, const gguf_ctx *g, const char *prefix) {
	int32_t v32;
	int		q_lora	= 2048;
	int		kv_lora = 512;
	int		qk_head = 0;
	int		qk_rope = 0;
	int		qk_nope = 0;
	int		v_head	= 0;
	if (akey_i32(g, prefix, "attention.q_lora_rank", &v32) == OK)
		q_lora = v32;
	if (akey_i32(g, prefix, "attention.kv_lora_rank", &v32) == OK)
		kv_lora = v32;
	if (akey_i32(g, prefix, "attention.key_length_mla", &v32) == OK)
		qk_head = v32;
	if (akey_i32(g, prefix, "attention.value_length_mla", &v32) == OK)
		v_head = v32;
	if (akey_i32(g, prefix, "rope.dimension_count", &v32) == OK ||
		akey_i32(g, prefix, "attention.qk_rope_head_dim", &v32) == OK)
		qk_rope = v32;
	if (akey_i32(g, prefix, "attention.qk_nope_head_dim", &v32) == OK)
		qk_nope = v32;
	else if (qk_head > 0 && qk_rope > 0)
		qk_nope = qk_head - qk_rope;
	if (qk_head == 0)
		qk_head = qk_nope + qk_rope;
	if (v_head == 0) {
		if (akey_i32(g, prefix, "attention.v_head_dim", &v32) == OK)
			v_head = v32;
	}
	if (q_lora <= 0)
		q_lora = 2048;
	if (kv_lora <= 0)
		kv_lora = 512;
	if (qk_rope <= 0)
		qk_rope = 64;
	if (qk_nope <= 0)
		qk_nope = 192;
	if (qk_head <= 0)
		qk_head = 256;
	if (v_head <= 0)
		v_head = 256;
	m->mla.q_lora  = q_lora;
	m->mla.kv_lora = kv_lora;
	m->mla.qk_nope = qk_nope;
	m->mla.qk_rope = qk_rope;
	m->mla.qk_head = qk_head;
	m->mla.v_head  = v_head;
	m->head_dim	   = m->mla.qk_head;
	m->rope_dim	   = m->mla.qk_rope;
	DEBUG("glm-dsa: MLA q_lora=%d kv_lora=%d qk_nope=%d qk_rope=%d "
		  "qk_head=%d v_head=%d (head_dim=%d, rope_dim=%d)",
		  m->mla.q_lora, m->mla.kv_lora, m->mla.qk_nope, m->mla.qk_rope, m->mla.qk_head,
		  m->mla.v_head, m->head_dim, m->rope_dim);
}

static status_code load_qwen35_metadata(model *m, const gguf_ctx *g, const char *prefix) {
	int32_t v32;
	model_qwen35_params *q = &m->qwen35;

	if (req_i32(g, prefix, "ssm.conv_kernel", &v32) != OK)
		return ERR_FORMAT;
	q->conv_kernel = v32;
	if (req_i32(g, prefix, "ssm.inner_size", &v32) != OK)
		return ERR_FORMAT;
	q->inner_size = v32;
	if (req_i32(g, prefix, "ssm.state_size", &v32) != OK)
		return ERR_FORMAT;
	q->state_size = v32;
	if (req_i32(g, prefix, "ssm.time_step_rank", &v32) != OK)
		return ERR_FORMAT;
	q->n_value_heads = v32;
	if (req_i32(g, prefix, "ssm.group_count", &v32) != OK)
		return ERR_FORMAT;
	q->n_key_heads = v32;

	q->full_attention_interval = 4;
	if (akey_i32(g, prefix, "attention.full_attention_interval", &v32) == OK ||
		akey_i32(g, prefix, "full_attention_interval", &v32) == OK)
		q->full_attention_interval = v32;

	if (q->conv_kernel < 1 || q->inner_size < 1 || q->state_size < 1 ||
		q->n_value_heads < 1 || q->n_key_heads < 1 ||
		q->inner_size % q->n_value_heads != 0 ||
		q->n_value_heads % q->n_key_heads != 0 || q->full_attention_interval < 1) {
		ERROR("qwen35: invalid recurrent dimensions (conv=%d inner=%d state=%d "
			  "value_heads=%d key_heads=%d interval=%d)",
			  q->conv_kernel, q->inner_size, q->state_size, q->n_value_heads, q->n_key_heads,
			  q->full_attention_interval);
		return ERR_FORMAT;
	}

	q->key_dim = q->state_size * q->n_key_heads;
	q->value_head_dim = q->inner_size / q->n_value_heads;
	q->value_dim = q->inner_size;
	q->conv_dim = 2 * q->key_dim + q->value_dim;

	DEBUG("qwen35 GDN: conv=%d inner=%d state=%d key_heads=%d value_heads=%d interval=%d",
		  q->conv_kernel, q->inner_size, q->state_size, q->n_key_heads, q->n_value_heads,
		  q->full_attention_interval);
	return OK;
}

static status_code model_load_open(model *m, const char *path, int use_mmap,
								   const char *repack_config, int requested_n_ctx) {
	(void)requested_n_ctx;
	m->use_mmap		 = use_mmap;
	m->model_path	 = xstrdup(path);
	m->repack_config = repack_config ? xstrdup(repack_config) : NULL;
	status_code s	 = use_mmap ? gguf_load(&m->gctx, path) : gguf_load_metadata(&m->gctx, path);
	if (s != OK) {
		free(m->model_path);
		free((char *)m->repack_config);
		m->model_path	 = NULL;
		m->repack_config = NULL;
		return s;
	}

	if (g_monitor && g_monitor->fd >= 0) {
		monitor_send(g_monitor, "{\"type\":\"load\",\"phase\":\"prefetch_mmap\",\"path\":\"%s\"}",
					 path);
		monitor_poll(g_monitor);
	}
	uint64_t pf_t0 = time_us();

	if (use_mmap)
		model_prefetch_mmap(&m->gctx);

	if (g_monitor && g_monitor->fd >= 0) {
		monitor_send(g_monitor, "{\"type\":\"load\",\"phase\":\"prefetch_done\",\"ms\":%llu}",
					 (unsigned long long)((time_us() - pf_t0) / 1000));
		monitor_poll(g_monitor);
	}

	return OK;
}

static status_code model_load_metadata(model *m, const gguf_ctx *g, const char *prefix) {
	int32_t v32;
	float	vf;
	if (req_i32(g, prefix, "block_count", &v32) != OK) {
		return ERR_FORMAT;
	}
	int n_layer_all	  = v32;
	int n_layer_nextn = 0;
	if (akey_i32(g, prefix, "nextn_predict_layers", &v32) == OK) {
		n_layer_nextn = v32;
	}
	if (n_layer_nextn < 0 || n_layer_nextn >= n_layer_all) {
		WARN("model_load: '%s.nextn_predict_layers'=%d is out of range for "
			 "block_count=%d; ignoring (treating as 0)",
			 prefix, n_layer_nextn, n_layer_all);
		n_layer_nextn = 0;
	}
	m->n_layers		 = n_layer_all - n_layer_nextn;
	m->n_layer_nextn = 0;
	if (n_layer_nextn == 1) {
		if (config_get()->spec_type == SPEC_TYPE_MTP)
			m->n_layer_nextn = 1;
	} else if (n_layer_nextn > 1) {
		WARN("model_load: nextn_predict_layers=%d not supported (need 1); MTP disabled",
			 n_layer_nextn);
	}
	if (req_i32(g, prefix, "context_length", &v32) != OK) {
		return ERR_FORMAT;
	}
	m->n_ctx = v32;
	if (req_i32(g, prefix, "embedding_length", &v32) != OK) {
		return ERR_FORMAT;
	}
	m->dim		= v32;
	m->dim_sqrt = sqrtf((float)v32);
	if (m->arch_info->has_variable_layer_dims) {
		const int32_t *ffn_arr;
		size_t		   ffn_n;
		char		   ffn_key[64];
		snprintf(ffn_key, sizeof(ffn_key), "%s.feed_forward_length", prefix);
		if (gguf_get_arr_i32(g, ffn_key, &ffn_arr, &ffn_n) == OK && ffn_n > 0) {
			m->intermediate = ffn_arr[0];
		} else if (akey_i32(g, prefix, "feed_forward_length", &v32) == OK) {
			m->intermediate = v32;
			DEBUG("ffn_intermediate = %d (uniform)", v32);
		} else {
			ERROR("model_load: missing required metadata key '%s' "
				  "(neither array nor scalar form found)",
				  ffn_key);
			return ERR_FORMAT;
		}
	} else {
		if (req_i32(g, prefix, "feed_forward_length", &v32) != OK) {
			return ERR_FORMAT;
		}
		m->intermediate = v32;
	}
	if (req_i32(g, prefix, "attention.head_count", &v32) != OK) {
		return ERR_FORMAT;
	}
	m->n_heads = v32;
	if (akey_i32(g, prefix, "attention.head_count_kv", &v32) == OK) {
		m->n_kv_heads = v32;
	} else {
		const int32_t *kv_arr;
		size_t		   kv_n;
		char		   kv_key[128];
		snprintf(kv_key, sizeof(kv_key), "%s.attention.head_count_kv", prefix);
		if (gguf_get_arr_i32(g, kv_key, &kv_arr, &kv_n) == OK && kv_n > 0) {
			m->n_kv_heads = kv_arr[0];
			int uniform	  = 1;
			for (size_t i = 1; i < kv_n; i++) {
				if (kv_arr[i] != kv_arr[0]) {
					uniform = 0;
					break;
				}
			}
			if (!uniform) {
				if (m->arch_info->has_variable_layer_dims) {
					DEBUG("n_kv_heads varies per layer ([%d, %d, ...] -- per-layer dims supported)",
						  kv_arr[0], kv_n > 1 ? kv_arr[1] : kv_arr[0]);
				} else {
					ERROR("%s: %s varies per layer ([%d, %d, ...]); this engine "
						  "only supports a single model-wide KV head count for "
						  "this architecture",
						  m->arch_info->gguf_name, kv_key, kv_arr[0],
						  kv_n > 1 ? kv_arr[1] : kv_arr[0]);
					return ERR_FORMAT;
				}
			} else {
				DEBUG("n_kv_heads = %d (uniform)", m->n_kv_heads);
			}
		} else {
			m->n_kv_heads = m->n_heads;
		}
	}
	if (req_f32(g, prefix, "attention.layer_norm_rms_epsilon", &vf) != OK) {
		return ERR_FORMAT;
	}
	m->norm_eps = vf;

	if (akey_i32(g, prefix, "attention.key_length", &v32) == OK) {
		m->head_dim = v32;
	} else {
		m->head_dim = m->dim / m->n_heads;
	}
	m->rope_dim = m->head_dim;
	if (akey_i32(g, prefix, "rope.dimension_count", &v32) == OK) {
		m->rope_dim = v32;
	}

	if (akey_f32(g, prefix, "rope.freq_base", &vf) == OK) {
		m->rope_theta = vf;
	} else {
		m->rope_theta = m->arch_info->default_rope_theta;
	}

	m->attn_logit_softcap = 0.0f;
	akey_f32(g, prefix, "attn_logit_softcapping", &m->attn_logit_softcap);
	m->final_logit_softcap = 0.0f;
	akey_f32(g, prefix, "final_logit_softcapping", &m->final_logit_softcap);

	m->sliding_window = 0;
	if (m->arch_info->sliding_window_period > 0) {
		if (akey_i32(g, prefix, "attention.sliding_window", &v32) == OK) {
			m->sliding_window = v32;
		} else {
			WARN("arch '%s' expects a sliding window but '%s.attention.sliding_window' is "
				 "missing; disabling",
				 m->arch_info->gguf_name, prefix);
		}
	}

	if (m->arch_info->has_variable_layer_dims) {
		if (load_gemma4_metadata(m, g, prefix) != OK)
			return ERR_FORMAT;
	}

	if (m->arch_info->is_moe || akey_i32(g, prefix, "expert_count", &v32) == OK) {
		if (load_moe_metadata(m, g, prefix) != OK)
			return ERR_FORMAT;
	}

	if (m->arch_info->is_mla) {
		load_mla_metadata(m, g, prefix);
	}
	if (m->arch_info->is_hybrid_recurrent) {
		if (load_qwen35_metadata(m, g, prefix) != OK)
			return ERR_FORMAT;
	}

	return OK;
}

static status_code load_layer_tensor(const gguf_ctx *g, char *tname, size_t tname_sz, int i,
									 layer_weights *L, weight_ref *ref, const char *fmt,
									 int debug_first) __attribute__((format(printf, 7, 0)));

static status_code load_layer_tensor(const gguf_ctx *g, char *tname, size_t tname_sz, int i,
									 layer_weights *L, weight_ref *ref, const char *fmt,
									 int debug_first) {
	(void)L;
	const gguf_tensor *t = find_tensor_fmt(g, tname, tname_sz, fmt, i);
	if (!t)
		return ERR_FORMAT;
	ref->host_ptr = t->data;
	ref->type	  = t->type;
	if (debug_first && i == 0)
		DEBUG("%s type=%u (%s)", tname, t->type, ggml_type_name(t->type));
	return OK;
}

static int qwen35_layer_is_recurrent(const model *m, const gguf_ctx *g, int layer) {
	char key[128];
	snprintf(key, sizeof(key), "%s.attention.recurrent_layers", m->arch_info->key_prefix);
	for (size_t i = 0; i < g->n_kv; i++) {
		if (strcmp(g->kv_keys[i], key) != 0 || g->kv_types[i] != GGUF_TYPE_ARRAY ||
			layer < 0 || (size_t)layer >= g->kv_arr_len[i])
			continue;
		if (g->kv_arr_type[i] == GGUF_TYPE_BOOL)
			return ((const uint8_t *)g->kv_arr_data[i])[layer] != 0;
		if (g->kv_arr_type[i] == GGUF_TYPE_I32)
			return ((const int32_t *)g->kv_arr_data[i])[layer] != 0;
	}
	return ((layer + 1) % m->qwen35.full_attention_interval) != 0;
}

static status_code load_weight_named(const gguf_ctx *g, weight_ref *ref, const char *name,
									 int required) {
	const gguf_tensor *t = gguf_find_tensor(g, name);
	if (!t) {
		if (required) {
			ERROR("model_load: missing required tensor '%s'", name);
			return ERR_FORMAT;
		}
		*ref = (weight_ref){0};
		return OK;
	}
	ref->host_ptr = t->data;
	ref->type	  = t->type;
	return OK;
}

static status_code load_mtp_layer(model *m, const gguf_ctx *g) {
	if (m->n_layer_nextn != 1)
		return OK;
	int			   i = m->n_layers;
	layer_weights *L = &m->layers[i];
	char		   tname[160];

	L->head_dim		= m->head_dim;
	L->intermediate = m->intermediate;
	L->n_kv_heads	= m->n_kv_heads;
	L->rope_dim		= m->rope_dim;
	L->has_own_v	= 1;
	L->is_recurrent = 0;

	snprintf(tname, sizeof(tname), "blk.%d.nextn.eh_proj.weight", i);
	if (load_weight_named(g, &m->mtp_eh_proj, tname, 0) != OK || !m->mtp_eh_proj.host_ptr) {
		WARN("MTP: %s missing; draft head disabled", tname);
		m->n_layer_nextn = 0;
		return OK;
	}
	snprintf(tname, sizeof(tname), "blk.%d.nextn.enorm.weight", i);
	if (load_weight_named(g, &m->mtp_enorm, tname, 1) != OK)
		return ERR_FORMAT;
	snprintf(tname, sizeof(tname), "blk.%d.nextn.hnorm.weight", i);
	if (load_weight_named(g, &m->mtp_hnorm, tname, 1) != OK)
		return ERR_FORMAT;
	snprintf(tname, sizeof(tname), "blk.%d.nextn.shared_head_norm.weight", i);
	load_weight_named(g, &m->mtp_shared_head_norm, tname, 0);

	if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->attn_norm_w, "blk.%d.attn_norm.weight",
						  0) != OK ||
		load_layer_tensor(g, tname, sizeof(tname), i, L, &L->wq, "blk.%d.attn_q.weight", 0) != OK ||
		load_layer_tensor(g, tname, sizeof(tname), i, L, &L->wk, "blk.%d.attn_k.weight", 0) != OK ||
		load_layer_tensor(g, tname, sizeof(tname), i, L, &L->wo, "blk.%d.attn_output.weight", 0) !=
			OK ||
		load_layer_tensor(g, tname, sizeof(tname), i, L, &L->gate_w, "blk.%d.ffn_gate.weight", 0) !=
			OK ||
		load_layer_tensor(g, tname, sizeof(tname), i, L, &L->up_w, "blk.%d.ffn_up.weight", 0) !=
			OK ||
		load_layer_tensor(g, tname, sizeof(tname), i, L, &L->down_w, "blk.%d.ffn_down.weight", 0) !=
			OK)
		return ERR_FORMAT;

	snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", i);
	if (load_weight_named(g, &L->wv, tname, 1) != OK)
		return ERR_FORMAT;
	if (m->arch_info->has_attn_post_norm) {
		if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->post_attn_norm_w,
							  "blk.%d.post_attention_norm.weight", 0) != OK)
			return ERR_FORMAT;
	}
	if (m->arch_info->has_qk_norm) {
		if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->attn_q_norm_w,
							  "blk.%d.attn_q_norm.weight", 0) != OK ||
			load_layer_tensor(g, tname, sizeof(tname), i, L, &L->attn_k_norm_w,
							  "blk.%d.attn_k_norm.weight", 0) != OK)
			return ERR_FORMAT;
	}
	INFO("MTP: loaded blk.%d NextN draft head", i);
	return OK;
}

static void fill_wrefs_row(model *m, int li) {
	layer_weights *L   = &m->layers[li];
	weight_ref	 **row = &m->wrefs_by_layer[(size_t)li * WIDX_COUNT];
	row[WIDX_TOK_EMBD]		  = (weight_ref *)&m->tok_embd;
	row[WIDX_OUTPUT_NORM]	  = (weight_ref *)&m->output_norm_w;
	row[WIDX_OUTPUT_W]		  = (weight_ref *)&m->output_w;
	row[WIDX_ATTN_NORM]		  = &L->attn_norm_w;
	row[WIDX_WQ]			  = &L->wq;
	row[WIDX_WK]			  = &L->wk;
	row[WIDX_WV]			  = &L->wv;
	row[WIDX_WO]			  = &L->wo;
	row[WIDX_FFN_NORM]		  = &L->ffn_norm_w;
	row[WIDX_GATE]			  = &L->gate_w;
	row[WIDX_UP]			  = &L->up_w;
	row[WIDX_DOWN]			  = &L->down_w;
	row[WIDX_GATE_UP]		  = &L->gate_up_w;
	row[WIDX_POST_ATTN_NORM]  = &L->post_attn_norm_w;
	row[WIDX_POST_FFN_NORM]	  = &L->post_ffn_norm_w;
	row[WIDX_ATTN_Q_NORM]	  = &L->attn_q_norm_w;
	row[WIDX_ATTN_K_NORM]	  = &L->attn_k_norm_w;
	row[WIDX_PLE_POST_NORM]	  = &L->ple_post_norm_w;
	row[WIDX_PLE_INP_GATE]	  = &L->ple_inp_gate_w;
	row[WIDX_PLE_PROJ]		  = &L->ple_proj_w;
	row[WIDX_LAYER_OUT_SCALE] = &L->layer_out_scale_w;
	row[WIDX_ROPE_FREQS]	  = m->rope_freqs_count > 0 ? (weight_ref *)&m->rope_freqs_w : NULL;
	row[WIDX_PER_LAYER_TOK_EMBD]   = (weight_ref *)&m->layer_dims.per_layer_tok_embd;
	row[WIDX_PER_LAYER_MODEL_PROJ] = (weight_ref *)&m->layer_dims.per_layer_model_proj;
	row[WIDX_PER_LAYER_PROJ_NORM]  = (weight_ref *)&m->layer_dims.per_layer_proj_norm_w;
	row[WIDX_FFN_GATE_INP]		   = &L->router_w;
	row[WIDX_EXP_PROBS_BIAS]	   = &L->router_bias;
	row[WIDX_FFN_GATE_INP_S]	   = &L->router_scale_w;
	row[WIDX_FFN_PRE_NORM_2]	   = &L->ffn_pre_norm_2_w;
	row[WIDX_FFN_POST_NORM_1]	   = &L->ffn_post_norm_1_w;
	row[WIDX_FFN_POST_NORM_2]	   = &L->ffn_post_norm_2_w;
	row[WIDX_SHEXP_GATE]		   = &L->shexp_gate_w;
	row[WIDX_SHEXP_UP]			   = &L->shexp_up_w;
	row[WIDX_SHEXP_DOWN]		   = &L->shexp_down_w;
	row[WIDX_MLA_Q_A]			   = &L->q_a_w;
	row[WIDX_MLA_Q_B]			   = &L->q_b_w;
	row[WIDX_MLA_Q_A_NORM]		   = &L->q_a_norm_w;
	row[WIDX_MLA_KV_A]			   = &L->kv_a_w;
	row[WIDX_MLA_K_B]			   = &L->k_b_w;
	row[WIDX_MLA_V_B]			   = &L->v_b_w;
	row[WIDX_MLA_KV_A_NORM]		   = &L->kv_a_norm_w;
	row[WIDX_ATTN_QKV]			   = &L->attn_qkv_w;
	row[WIDX_ATTN_GATE]			   = &L->attn_gate_w;
	row[WIDX_SSM_CONV1D]			   = &L->ssm_conv1d_w;
	row[WIDX_SSM_DT]				   = &L->ssm_dt_b;
	row[WIDX_SSM_A]				   = &L->ssm_a;
	row[WIDX_SSM_BETA]			   = &L->ssm_beta_w;
	row[WIDX_SSM_ALPHA]			   = &L->ssm_alpha_w;
	row[WIDX_SSM_NORM]			   = &L->ssm_norm_w;
	row[WIDX_SSM_OUT]				   = &L->ssm_out_w;
	row[WIDX_MTP_EH_PROJ]		   = &m->mtp_eh_proj;
	row[WIDX_MTP_ENORM]			   = &m->mtp_enorm;
	row[WIDX_MTP_HNORM]			   = &m->mtp_hnorm;
	row[WIDX_MTP_HEAD_NORM]		   = m->mtp_shared_head_norm.host_ptr ? &m->mtp_shared_head_norm
																	  : (weight_ref *)&m->output_norm_w;
}

static status_code model_load_tensor_layout(model *m, const gguf_ctx *g) {
	if (!m->use_mmap) {
		status_code rs = gguf_sparse_read_tensors(&m->gctx, m->model_path);
		if (rs != OK)
			return rs;
	}

	{
		const gguf_tensor *t = gguf_find_tensor(g, "token_embd.weight");
		if (!t) {
			ERROR("model_load: missing required tensor 'token_embd.weight'");
			return ERR_FORMAT;
		}
		m->vocab_size		 = (int)t->dims[1];
		m->tok_embd.host_ptr = t->data;
		m->tok_embd.type	 = t->type;
		DEBUG("token_embd type=%u (%s)", t->type, ggml_type_name(t->type));
	}
	{
		const gguf_tensor *t = gguf_find_tensor(g, "output_norm.weight");
		if (!t) {
			ERROR("model_load: missing required tensor 'output_norm.weight'");
			return ERR_FORMAT;
		}
		m->output_norm_w.host_ptr = t->data;
	}
	{
		const gguf_tensor *t = gguf_find_tensor(g, "output.weight");
		m->tie_embeddings	 = (t == NULL);
		if (m->tie_embeddings) {
			m->output_w = m->tok_embd;
			DEBUG("output tied to token_embd type=%u", m->tok_embd.type);
		} else {
			m->output_w.host_ptr = t->data;
			m->output_w.type	 = t->type;
			DEBUG("output type=%u (%s)", t->type, ggml_type_name(t->type));
		}
	}

	if (m->has_per_layer_embeddings) {
		const gguf_tensor *t = gguf_find_tensor(g, "per_layer_token_embd.weight");
		if (!t) {
			ERROR("model: per_layer_token_embd.weight not found but PLE is enabled");
			return ERR_FORMAT;
		}
		m->layer_dims.per_layer_tok_embd.host_ptr = t->data;
		m->layer_dims.per_layer_tok_embd.type	  = t->type;

		t = gguf_find_tensor(g, "per_layer_model_proj.weight");
		if (!t) {
			ERROR("model: per_layer_model_proj.weight not found");
			return ERR_FORMAT;
		}
		m->layer_dims.per_layer_model_proj.host_ptr = t->data;
		m->layer_dims.per_layer_model_proj.type		= t->type;

		t = gguf_find_tensor(g, "per_layer_proj_norm.weight");
		if (!t) {
			ERROR("model: per_layer_proj_norm.weight not found");
			return ERR_FORMAT;
		}
		m->layer_dims.per_layer_proj_norm_w.host_ptr = t->data;
	}

	m->layers = xcalloc((size_t)model_n_all_layers(m), sizeof(layer_weights));
	char tname[128];
	for (int i = 0; i < m->n_layers; i++) {
		layer_weights	  *L = &m->layers[i];
		const gguf_tensor *t;

		if (m->arch_info->has_variable_layer_dims) {
			L->head_dim		   = m->layer_dims.is_global_layer[i] ? m->layer_dims.head_dim_global
																  : m->layer_dims.head_dim_swa;
			L->intermediate	   = m->layer_dims.ffn_lengths[i];
			L->is_global_layer = m->layer_dims.is_global_layer[i];
			L->rope_dim		   = m->layer_dims.is_global_layer[i] ? m->layer_dims.rope_dim_global
																  : m->layer_dims.rope_dim_swa;
			L->n_kv_heads =
				(m->layer_dims.n_kv_heads_per_layer && m->layer_dims.n_kv_heads_per_layer[i] > 0)
					? m->layer_dims.n_kv_heads_per_layer[i]
					: m->n_kv_heads;
		} else {
			L->head_dim		   = m->head_dim;
			L->intermediate	   = m->intermediate;
			L->is_global_layer = 0;
			L->rope_dim		   = m->rope_dim;
			L->n_kv_heads	   = m->n_kv_heads;
		}
		L->has_own_v = 1;
		if (m->arch_info->is_hybrid_recurrent)
			L->is_recurrent = qwen35_layer_is_recurrent(m, g, i);

		if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->attn_norm_w,
							  "blk.%d.attn_norm.weight", 0) != OK)
			return ERR_FORMAT;

		if (model_layer_is_recurrent(m, i)) {
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->attn_qkv_w,
								  "blk.%d.attn_qkv.weight", 1) != OK ||
				load_layer_tensor(g, tname, sizeof(tname), i, L, &L->attn_gate_w,
								  "blk.%d.attn_gate.weight", 1) != OK ||
				load_layer_tensor(g, tname, sizeof(tname), i, L, &L->ssm_conv1d_w,
								  "blk.%d.ssm_conv1d.weight", 0) != OK ||
				load_layer_tensor(g, tname, sizeof(tname), i, L, &L->ssm_dt_b,
								  "blk.%d.ssm_dt.bias", 0) != OK ||
				load_layer_tensor(g, tname, sizeof(tname), i, L, &L->ssm_a,
								  "blk.%d.ssm_a", 0) != OK ||
				load_layer_tensor(g, tname, sizeof(tname), i, L, &L->ssm_beta_w,
								  "blk.%d.ssm_beta.weight", 0) != OK ||
				load_layer_tensor(g, tname, sizeof(tname), i, L, &L->ssm_alpha_w,
								  "blk.%d.ssm_alpha.weight", 0) != OK ||
				load_layer_tensor(g, tname, sizeof(tname), i, L, &L->ssm_norm_w,
								  "blk.%d.ssm_norm.weight", 0) != OK ||
				load_layer_tensor(g, tname, sizeof(tname), i, L, &L->ssm_out_w,
								  "blk.%d.ssm_out.weight", 1) != OK)
				return ERR_FORMAT;
			L->wq = (weight_ref){0};
			L->wk = (weight_ref){0};
			L->wv = (weight_ref){0};
			L->wo = (weight_ref){0};
			L->has_own_v = 0;
		} else if (m->arch_info && m->arch_info->is_mla) {
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->q_a_w,
								  "blk.%d.attn_q_a.weight", 1) != OK)
				return ERR_FORMAT;
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->q_b_w,
								  "blk.%d.attn_q_b.weight", 1) != OK)
				return ERR_FORMAT;
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->q_a_norm_w,
								  "blk.%d.attn_q_a_norm.weight", 0) != OK)
				return ERR_FORMAT;
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->kv_a_w,
								  "blk.%d.attn_kv_a_mqa.weight", 1) != OK)
				return ERR_FORMAT;
			{
				char nname[128];
				snprintf(nname, sizeof(nname), "blk.%d.attn_kv_a_norm.weight", i);
				const gguf_tensor *nt = gguf_find_tensor(g, nname);
				if (!nt) {
					snprintf(nname, sizeof(nname), "blk.%d.attn_kv_a_lora_norm.weight", i);
					nt = gguf_find_tensor(g, nname);
				}
				if (!nt) {
					ERROR("model_load: missing required tensor 'blk.%d.attn_kv_a_norm.weight'", i);
					return ERR_FORMAT;
				}
				L->kv_a_norm_w.host_ptr = nt->data;
			}
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->k_b_w,
								  "blk.%d.attn_k_b.weight", 1) != OK)
				return ERR_FORMAT;
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->v_b_w,
								  "blk.%d.attn_v_b.weight", 1) != OK)
				return ERR_FORMAT;
			if (m->backend && strcmp(m->backend->name, "cpu") == 0) {
				L->mla_kb_vb_f32 = 1;
				dequant_weight_to_f32(&L->k_b_w.host_ptr, &L->k_b_w.type,
									  (size_t)m->mla.qk_nope * m->mla.kv_lora, (size_t)m->n_heads);
				L->k_b_w.buf.handle	  = (void *)L->k_b_w.host_ptr;
				L->k_b_w.buf.host_ptr = NULL;

				dequant_weight_to_f32(&L->v_b_w.host_ptr, &L->v_b_w.type,
									  (size_t)m->mla.kv_lora * m->mla.v_head, (size_t)m->n_heads);
				L->v_b_w.buf.handle	  = (void *)L->v_b_w.host_ptr;
				L->v_b_w.buf.host_ptr = NULL;
			}
			L->wq		 = (weight_ref){0};
			L->wk		 = (weight_ref){0};
			L->wv		 = (weight_ref){0};
			L->has_own_v = 0;
		} else {
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->wq, "blk.%d.attn_q.weight",
								  1) != OK)
				return ERR_FORMAT;
			if (model_layer_has_kv(m, i)) {
				if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->wk, "blk.%d.attn_k.weight",
									  1) != OK)
					return ERR_FORMAT;
			} else {
				L->wk = (weight_ref){0};
				DEBUG("blk.%d.attn_k.weight not present (shared-KV layer); K "
					  "reused from KV-store layer",
					  i);
			}
			{
				char vname[128];
				snprintf(vname, sizeof(vname), "blk.%d.attn_v.weight", i);
				const gguf_tensor *vt = gguf_find_tensor(g, vname);
				if (vt) {
					L->wv.host_ptr = vt->data;
					L->wv.type	   = vt->type;
					L->has_own_v   = 1;
				} else if (!model_layer_has_kv(m, i)) {
					L->wv		 = (weight_ref){0};
					L->has_own_v = 0;
					DEBUG("blk.%d.attn_v.weight not present (shared-KV layer); V "
						  "reused from KV-store layer",
						  i);
				} else if (m->arch_info->has_variable_layer_dims && L->is_global_layer) {
					L->wv		 = (weight_ref){0};
					L->has_own_v = 0;
					DEBUG("blk.%d.attn_v.weight not present (global attention "
						  "layer); using K=V weight sharing for this layer",
						  i);
				} else {
					ERROR("model_load: missing required tensor '%s'", vname);
					return ERR_FORMAT;
				}
			}
		}

		if (!model_layer_is_recurrent(m, i)) {
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->wo,
								  "blk.%d.attn_output.weight", 1) != OK)
				return ERR_FORMAT;
		}
		if (!m->arch_info->uses_post_attn_norm_for_ffn) {
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->ffn_norm_w,
								  "blk.%d.ffn_norm.weight", 0) != OK)
				return ERR_FORMAT;
		}

		if (m->arch_info->is_moe) {
			L->is_moe_layer = (i >= m->moe.first_dense_layer);
			if (L->is_moe_layer) {
				{
					char rname[128];
					snprintf(rname, sizeof(rname), "blk.%d.ffn_gate_inp.weight", i);
					const gguf_tensor *rt = gguf_find_tensor(g, rname);
					if (!rt) {
						ERROR("model_load: missing router tensor '%s'", rname);
						return ERR_FORMAT;
					}
					L->router_w.host_ptr = rt->data;
					L->router_w.type	 = rt->type;
					if (i == m->moe.first_dense_layer) {
						DEBUG("%s type=%u (%s)", rname, rt->type, ggml_type_name(rt->type));
					}
				}
				{
					char bname[128];
					snprintf(bname, sizeof(bname), "blk.%d.exp_probs_b.bias", i);
					const gguf_tensor *bt = gguf_find_tensor(g, bname);
					if (!bt) {
						snprintf(bname, sizeof(bname), "blk.%d.exp_probs_bias.bias", i);
						bt = gguf_find_tensor(g, bname);
					}
					if (!bt) {
						snprintf(bname, sizeof(bname), "blk.%d.ffn_gate_inp.bias", i);
						bt = gguf_find_tensor(g, bname);
					}
					L->router_bias.host_ptr = bt ? bt->data : NULL;
				}
				if (m->arch_info->uses_moe_softmax_router) {
					char sname[128];
					snprintf(sname, sizeof(sname), "blk.%d.ffn_gate_inp.scale", i);
					const gguf_tensor *st	   = gguf_find_tensor(g, sname);
					L->router_scale_w.host_ptr = st ? st->data : NULL;
					if (st && i == m->moe.first_dense_layer) {
						DEBUG("%s type=%u (%s) dim=%llu", sname, st->type, ggml_type_name(st->type),
							  (unsigned long long)st->dims[0]);
					}
				}
				if (m->moe.n_shared_experts > 0) {
					if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->shexp_gate_w,
										  "blk.%d.ffn_gate_shexp.weight", 1) != OK)
						return ERR_FORMAT;
					if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->shexp_up_w,
										  "blk.%d.ffn_up_shexp.weight", 1) != OK)
						return ERR_FORMAT;
					if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->shexp_down_w,
										  "blk.%d.ffn_down_shexp.weight", 1) != OK)
						return ERR_FORMAT;
					L->intermediate = m->moe.moe_intermediate * m->moe.n_shared_experts;
				}
				{
					char ename[160];
					snprintf(ename, sizeof(ename), "blk.%d.ffn_gate_up_exps.weight", i);
					const gguf_tensor *gut = gguf_find_tensor(g, ename);
					snprintf(ename, sizeof(ename), "blk.%d.ffn_down_exps.weight", i);
					const gguf_tensor *dt = gguf_find_tensor(g, ename);
					snprintf(ename, sizeof(ename), "blk.%d.ffn_gate_exps.weight", i);
					const gguf_tensor *gt = gguf_find_tensor(g, ename);
					snprintf(ename, sizeof(ename), "blk.%d.ffn_up_exps.weight", i);
					const gguf_tensor *ut = gguf_find_tensor(g, ename);

					if (gut && dt) {
						size_t gate_up_row =
							ggml_row_size(gut->type, (size_t)m->moe.moe_intermediate * 2 * m->dim);
						size_t down_row =
							ggml_row_size(dt->type, (size_t)m->dim * m->moe.moe_intermediate);
						L->experts = xcalloc(m->moe.n_experts, sizeof(*L->experts));
						for (int e = 0; e < m->moe.n_experts; e++) {
							fill_expert_desc(
								&L->experts[e], (const char *)gut->data + ((size_t)e * gate_up_row),
								NULL, (const char *)dt->data + ((size_t)e * down_row), gut->type,
								gut->type, dt->type, 1, gut->offset + ((uint64_t)e * gate_up_row),
								0, dt->offset + ((uint64_t)e * down_row));
						}
						{
							snprintf(ename, sizeof(ename), "blk.%d.ffn_gate_up_exps.scale", i);
							const float *sc = NULL;
							if (expert_scale_f32(g, ename, m->moe.n_experts, &sc)) {
								for (int e = 0; e < m->moe.n_experts; e++) {
									L->experts[e].gate_scale = sc[e];
									L->experts[e].up_scale	 = sc[e];
								}
								if (i == m->moe.first_dense_layer)
									DEBUG("MoE: gate_up scales loaded");
							}
							snprintf(ename, sizeof(ename), "blk.%d.ffn_down_exps.scale", i);
							sc = NULL;
							if (expert_scale_f32(g, ename, m->moe.n_experts, &sc)) {
								for (int e = 0; e < m->moe.n_experts; e++)
									L->experts[e].down_scale = sc[e];
								if (i == m->moe.first_dense_layer)
									DEBUG("MoE: down scales loaded");
							}
						}
						if (i == m->moe.first_dense_layer) {
							DEBUG("MoE: %d experts, fused_gate_up, %s", m->moe.n_experts,
								  ggml_type_name(L->experts[0].gate_type));
						}
					} else if (gt && ut && dt) {
						size_t gate_row =
							ggml_row_size(gt->type, (size_t)m->moe.moe_intermediate * m->dim);
						size_t up_row =
							ggml_row_size(ut->type, (size_t)m->moe.moe_intermediate * m->dim);
						size_t down_row =
							ggml_row_size(dt->type, (size_t)m->dim * m->moe.moe_intermediate);
						L->experts = xcalloc(m->moe.n_experts, sizeof(*L->experts));
						for (int e = 0; e < m->moe.n_experts; e++) {
							fill_expert_desc(
								&L->experts[e], (const char *)gt->data + ((size_t)e * gate_row),
								(const char *)ut->data + ((size_t)e * up_row),
								(const char *)dt->data + ((size_t)e * down_row), gt->type, ut->type,
								dt->type, 0, gt->offset + ((uint64_t)e * gate_row),
								ut->offset + ((uint64_t)e * up_row),
								dt->offset + ((uint64_t)e * down_row));
						}
						{
							snprintf(ename, sizeof(ename), "blk.%d.ffn_gate_exps.scale", i);
							const float *sc = NULL;
							if (expert_scale_f32(g, ename, m->moe.n_experts, &sc)) {
								for (int e = 0; e < m->moe.n_experts; e++)
									L->experts[e].gate_scale = sc[e];
							}
							snprintf(ename, sizeof(ename), "blk.%d.ffn_up_exps.scale", i);
							sc = NULL;
							if (expert_scale_f32(g, ename, m->moe.n_experts, &sc)) {
								for (int e = 0; e < m->moe.n_experts; e++)
									L->experts[e].up_scale = sc[e];
							}
							snprintf(ename, sizeof(ename), "blk.%d.ffn_down_exps.scale", i);
							sc = NULL;
							if (expert_scale_f32(g, ename, m->moe.n_experts, &sc)) {
								for (int e = 0; e < m->moe.n_experts; e++)
									L->experts[e].down_scale = sc[e];
							}
						}
						if (i == m->moe.first_dense_layer) {
							DEBUG("MoE: %d experts, merged, %s", m->moe.n_experts,
								  ggml_type_name(L->experts[0].gate_type));
						}
					} else {
						L->experts		  = xcalloc(m->moe.n_experts, sizeof(*L->experts));
						int per_expert_ok = 1;
						for (int e = 0; e < m->moe.n_experts; e++) {
							snprintf(ename, sizeof(ename), "blk.%d.ffn_gate.%d.weight", i, e);
							const gguf_tensor *get = gguf_find_tensor(g, ename);
							snprintf(ename, sizeof(ename), "blk.%d.ffn_up.%d.weight", i, e);
							const gguf_tensor *uet = gguf_find_tensor(g, ename);
							snprintf(ename, sizeof(ename), "blk.%d.ffn_down.%d.weight", i, e);
							const gguf_tensor *det = gguf_find_tensor(g, ename);
							if (!get || !uet || !det) {
								per_expert_ok = 0;
								break;
							}
							fill_expert_desc(&L->experts[e], get->data, uet->data, det->data,
											 get->type, uet->type, det->type, 0, get->offset,
											 uet->offset, det->offset);
						}
						if (!per_expert_ok) {
							ERROR("missing expert tensors for MoE layer %d", i);
							DEBUG("layer %d: checked fused (ffn_gate_up_exps), merged "
								  "(ffn_*_exps.weight), and per-expert (ffn_*.{j}.weight) "
								  "tensor names, found none",
								  i);
							return ERR_FORMAT;
						}
						if (i == m->moe.first_dense_layer) {
							DEBUG("MoE: %d experts, per-expert tensors, %s", m->moe.n_experts,
								  ggml_type_name(L->experts[0].gate_type));
						}
					}
				}
				L->any_fused_experts = 0;
				L->gate_q8_type		 = 0;
				for (int e = 0; e < m->moe.n_experts; e++) {
					if (L->experts[e].gate_up_fused)
						L->any_fused_experts = 1;
					if (!L->gate_q8_type) {
						uint32_t qt = wtype_to_q8type(L->experts[e].gate_type);
						if (qt)
							L->gate_q8_type = qt;
					}
				}
				if (!m->arch_info->uses_moe_shared_dense_ffn) {
					L->gate_w = (weight_ref){0};
					L->up_w	  = (weight_ref){0};
					L->down_w = (weight_ref){0};
				} else {
					if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->gate_w,
										  "blk.%d.ffn_gate.weight", 1) != OK)
						return ERR_FORMAT;
					if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->up_w,
										  "blk.%d.ffn_up.weight", 1) != OK)
						return ERR_FORMAT;
					if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->down_w,
										  "blk.%d.ffn_down.weight", 1) != OK)
						return ERR_FORMAT;
				}
				if (m->arch_info->uses_moe_softmax_router) {
					{
						char			   nname[128];
						const gguf_tensor *nt = NULL;
						snprintf(nname, sizeof(nname), "blk.%d.pre_ffw_norm_2.weight", i);
						nt = gguf_find_tensor(g, nname);
						if (!nt) {
							snprintf(nname, sizeof(nname), "blk.%d.ffn_pre_norm_2.weight", i);
							nt = gguf_find_tensor(g, nname);
						}
						L->ffn_pre_norm_2_w.host_ptr = nt ? nt->data : NULL;
					}
					{
						char			   nname[128];
						const gguf_tensor *nt = NULL;
						snprintf(nname, sizeof(nname), "blk.%d.post_ffw_norm_1.weight", i);
						nt = gguf_find_tensor(g, nname);
						if (!nt) {
							snprintf(nname, sizeof(nname), "blk.%d.ffn_post_norm_1.weight", i);
							nt = gguf_find_tensor(g, nname);
						}
						L->ffn_post_norm_1_w.host_ptr = nt ? nt->data : NULL;
					}
					{
						char			   nname[128];
						const gguf_tensor *nt = NULL;
						snprintf(nname, sizeof(nname), "blk.%d.post_ffw_norm_2.weight", i);
						nt = gguf_find_tensor(g, nname);
						if (!nt) {
							snprintf(nname, sizeof(nname), "blk.%d.ffn_post_norm_2.weight", i);
							nt = gguf_find_tensor(g, nname);
						}
						L->ffn_post_norm_2_w.host_ptr = nt ? nt->data : NULL;
					}
					if (i == m->moe.first_dense_layer) {
						DEBUG("MoE: additional norms loaded (pre2/post1/post2)");
					}
				}
			} else {
				if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->gate_w,
									  "blk.%d.ffn_gate.weight", 1) != OK)
					return ERR_FORMAT;
				if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->up_w,
									  "blk.%d.ffn_up.weight", 1) != OK)
					return ERR_FORMAT;
				if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->down_w,
									  "blk.%d.ffn_down.weight", 1) != OK)
					return ERR_FORMAT;
			}
		} else {
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->gate_w,
								  "blk.%d.ffn_gate.weight", 1) != OK)
				return ERR_FORMAT;
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->up_w, "blk.%d.ffn_up.weight",
								  1) != OK)
				return ERR_FORMAT;
			if (load_layer_tensor(g, tname, sizeof(tname), i, L, &L->down_w,
								  "blk.%d.ffn_down.weight", 1) != OK)
				return ERR_FORMAT;
		}

		if (m->arch_info->has_variable_layer_dims) {
			char kname[128];
			snprintf(kname, sizeof(kname), "blk.%d.attn_k.weight", i);
			const gguf_tensor *kt = gguf_find_tensor(g, kname);
			if (kt && kt->n_dims == 2 && L->head_dim > 0) {
				uint64_t kv_out_dim = kt->dims[1];
				if (kv_out_dim % (uint64_t)L->head_dim != 0) {
					WARN("blk.%d.attn_k.weight kv_out=%llu not divisible by "
						 "head_dim=%d; keeping n_kv_heads=%d (from metadata)",
						 i, (unsigned long long)kv_out_dim, L->head_dim, L->n_kv_heads);
				} else {
					int inferred = (int)(kv_out_dim / (uint64_t)L->head_dim);
					if (m->layer_dims.n_kv_heads_per_layer &&
						m->layer_dims.n_kv_heads_per_layer[i] > 0 &&
						m->layer_dims.n_kv_heads_per_layer[i] != inferred) {
						WARN("blk.%d: metadata head_count_kv[%d]=%d disagrees "
							 "with tensor shape (kv_out=%llu / head_dim=%d => %d); "
							 "trusting metadata",
							 i, i, m->layer_dims.n_kv_heads_per_layer[i],
							 (unsigned long long)kv_out_dim, L->head_dim, inferred);
					} else if (!m->layer_dims.n_kv_heads_per_layer ||
							   m->layer_dims.n_kv_heads_per_layer[i] == 0) {
						L->n_kv_heads = inferred;
					}
				}
			}
		}

		if (m->arch_info->has_attn_post_norm) {
			t = find_tensor_fmt(g, tname, sizeof(tname), "blk.%d.post_attention_norm.weight", i);
			if (!t) {
				return ERR_FORMAT;
			}
			L->post_attn_norm_w.host_ptr = t->data;
		}
		if (m->arch_info->has_ffn_post_norm) {
			t = find_tensor_fmt(g, tname, sizeof(tname), "blk.%d.post_ffw_norm.weight", i);
			if (!t) {
				return ERR_FORMAT;
			}
			L->post_ffn_norm_w.host_ptr = t->data;
		}

		if (m->arch_info->has_qk_norm && !model_layer_is_recurrent(m, i)) {
			t = find_tensor_fmt(g, tname, sizeof(tname), "blk.%d.attn_q_norm.weight", i);
			if (!t) {
				return ERR_FORMAT;
			}
			L->attn_q_norm_w.host_ptr = t->data;
			if (model_layer_has_kv(m, i)) {
				t = find_tensor_fmt(g, tname, sizeof(tname), "blk.%d.attn_k_norm.weight", i);
				if (!t) {
					return ERR_FORMAT;
				}
				L->attn_k_norm_w.host_ptr = t->data;
			} else {
				L->attn_k_norm_w = (weight_ref){0};
			}
		}
		if (m->has_per_layer_embeddings) {
			t = find_tensor_fmt(g, tname, sizeof(tname), "blk.%d.post_norm.weight", i);
			if (!t) {
				return ERR_FORMAT;
			}
			L->ple_post_norm_w.host_ptr = t->data;
			t = find_tensor_fmt(g, tname, sizeof(tname), "blk.%d.inp_gate.weight", i);
			if (!t) {
				return ERR_FORMAT;
			}
			L->ple_inp_gate_w.host_ptr = t->data;
			L->ple_inp_gate_w.type	   = t->type;
			t = find_tensor_fmt(g, tname, sizeof(tname), "blk.%d.proj.weight", i);
			if (!t) {
				return ERR_FORMAT;
			}
			L->ple_proj_w.host_ptr = t->data;
			L->ple_proj_w.type	   = t->type;
		}
		if (m->arch_info->has_layer_output_scale) {
			t = find_tensor_fmt(g, tname, sizeof(tname), "blk.%d.layer_output_scale.weight", i);
			if (!t) {
				return ERR_FORMAT;
			}
			L->layer_out_scale_w.host_ptr = t->data;
			L->layer_out_scale			  = ((const float *)L->layer_out_scale_w.host_ptr)[0];
		}

		if (m->arch_info->has_variable_layer_dims) {
			L->is_sliding = !L->is_global_layer;
		} else if (m->sliding_window > 0) {
			int period	  = m->arch_info->sliding_window_period;
			L->is_sliding = ((i + 1) % period) != 0;
		}
	}

	if (load_mtp_layer(m, g) != OK)
		return ERR_FORMAT;

	int n_all = model_n_all_layers(m);
	m->wrefs_by_layer = xcalloc((size_t)n_all * WIDX_COUNT, sizeof(weight_ref *));
	for (int li = 0; li < n_all; li++)
		fill_wrefs_row(m, li);

	return OK;
}

status_code model_set_layer_backend_range(model *m, int begin, int end, backend *b) {
	if (!m || !m->layers || begin < 0 || end <= begin)
		return ERR_INVALID_ARG;
	if (end > m->n_layers)
		end = m->n_layers;
	if (!m->layer_backends) {
		m->layer_backends	= xcalloc((size_t)m->n_layers, sizeof(backend *));
		m->n_layer_backends = m->n_layers;
		for (int i = 0; i < m->n_layers; i++)
			m->layer_backends[i] = m->backend;
	}
	for (int i = begin; i < end; i++)
		m->layer_backends[i] = b;
	if (b != m->backend)
		m->mixed_backend_mode = 1;
	return OK;
}

status_code model_load_parse(model *m, const char *path, backend *accel, int use_mmap,
							 const char *repack_config, int requested_n_ctx) {
	memset(m, 0, sizeof(*m));
	m->batchable = -1;
	m->backend	 = accel;

	size_t avail_before_load = get_available_memory();

	status_code s = model_load_open(m, path, use_mmap, repack_config, requested_n_ctx);
	if (s != OK) {
		goto fail;
	}

	const gguf_ctx *g = &m->gctx;

	m->arch = arch_detect(g);
	if (m->arch == ARCH_UNKNOWN) {
		goto fail;
	}
	m->arch_info = arch_lookup(m->arch);

	const char *gguf_arch = NULL;
	if (gguf_get_str(g, "general.architecture", &gguf_arch) != OK || !gguf_arch) {
		ERROR(
			"model_load: missing 'general.architecture' (should have been caught by arch_detect)");
		goto fail;
	}
	const char *prefix = gguf_arch;

	s = model_load_metadata(m, g, prefix);
	if (s != OK) {
		goto fail;
	}

	s = model_load_tensor_layout(m, g);
	if (s != OK) {
		goto fail;
	}

	if (validate_swa_support(m)) {
		goto fail;
	}

	if (validate_model_dims(m, g)) {
		goto fail;
	}

	int report_n_ctx = requested_n_ctx;
	if (report_n_ctx <= 0 || report_n_ctx > m->n_ctx)
		report_n_ctx = m->n_ctx;
	recommend_memory_config(m, report_n_ctx, avail_before_load,
							(kv_quant_type)config_get()->kv_quant);

	return OK;

fail:
	ERROR("model_load_parse: aborting load of '%s' due to above error", path);
	model_free(m);
	return ERR_FORMAT;
}

status_code model_upload_weights(model *m) {
	if (g_monitor && g_monitor->fd >= 0) {
		monitor_send(g_monitor, "{\"type\":\"load\",\"phase\":\"upload_weights_start\"}");
		monitor_poll(g_monitor);
	}
	uint64_t up_t0 = time_us();

	if (upload_all_weights(m) != OK) {
		return ERR_FORMAT;
	}

	INFO("weights uploaded in %llu ms", (unsigned long long)((time_us() - up_t0) / 1000));
	if (g_monitor && g_monitor->fd >= 0) {
		monitor_send(g_monitor, "{\"type\":\"load\",\"phase\":\"upload_weights_done\",\"ms\":%llu}",
					 (unsigned long long)((time_us() - up_t0) / 1000));
		monitor_poll(g_monitor);
	}
	return OK;
}

status_code model_build_recipe(model *m) {
	m->recipe = recipe_build(m);
	if (!m->recipe) {
		ERROR("model_load: no recipe builder registered for arch '%s' "
			  "— cannot load model",
			  m->arch_info->gguf_name);
		return ERR_FORMAT;
	}
	DEBUG("recipe: built for arch '%s' (pre=%d layer=%d post=%d ops)", m->arch_info->gguf_name,
		  m->recipe->n_pre_ops, m->recipe->layer.n_ops, m->recipe->n_post_ops);
	return OK;
}

status_code model_load_backend_ex_repack(model *m, const char *path, backend *accel, int use_mmap,
										 const char *repack_config, int requested_n_ctx) {
	status_code s = model_load_parse(m, path, accel, use_mmap, repack_config, requested_n_ctx);
	if (s != OK)
		return s;

	s = model_upload_weights(m);
	if (s != OK) {
		model_free(m);
		return s;
	}

	s = model_build_recipe(m);
	if (s != OK) {
		model_free(m);
		return s;
	}

	return OK;
}

static void free_weight_buf(buffer *buf) {
	if (!buf->owner)
		return;
	buf->owner->buffer_free(buf->owner, buf);
}

void model_free(model *m) {
	if (!m)
		return;

	if (m->moe_cache) {
		moe_stream_cache_free(m->moe_cache);
		m->moe_cache = NULL;
	}

	if (m->backend && m->layers) {
		for (int i = 0; i < model_n_all_layers(m); i++) {
			layer_weights *L = &m->layers[i];
			free_weight_buf(&L->attn_norm_w.buf);
			free_weight_buf(&L->wq.buf);
			free_weight_buf(&L->wk.buf);
			free_weight_buf(&L->wv.buf);
			free_weight_buf(&L->wo.buf);
			free_weight_buf(&L->attn_qkv_w.buf);
			free_weight_buf(&L->attn_gate_w.buf);
			free_weight_buf(&L->ssm_conv1d_w.buf);
			free_weight_buf(&L->ssm_dt_b.buf);
			free_weight_buf(&L->ssm_a.buf);
			free_weight_buf(&L->ssm_beta_w.buf);
			free_weight_buf(&L->ssm_alpha_w.buf);
			free_weight_buf(&L->ssm_norm_w.buf);
			free_weight_buf(&L->ssm_out_w.buf);
			free_weight_buf(&L->q_a_w.buf);
			free_weight_buf(&L->q_b_w.buf);
			free_weight_buf(&L->q_a_norm_w.buf);
			free_weight_buf(&L->kv_a_w.buf);
			free_weight_buf(&L->k_b_w.buf);
			free_weight_buf(&L->v_b_w.buf);
			if (L->mla_kb_vb_f32) {
				free((void *)L->k_b_w.host_ptr);
				free((void *)L->v_b_w.host_ptr);
				L->mla_kb_vb_f32 = 0;
			}
			free_weight_buf(&L->kv_a_norm_w.buf);
			free_weight_buf(&L->router_w.buf);
			free_weight_buf(&L->router_bias.buf);
			free_weight_buf(&L->router_scale_w.buf);
			free_weight_buf(&L->shexp_gate_w.buf);
			free_weight_buf(&L->shexp_up_w.buf);
			free_weight_buf(&L->shexp_down_w.buf);
			free(L->shexp_fused_host);
			L->shexp_fused_host = NULL;
			free_weight_buf(&L->ffn_pre_norm_2_w.buf);
			free_weight_buf(&L->ffn_post_norm_1_w.buf);
			free_weight_buf(&L->ffn_post_norm_2_w.buf);
			free(L->experts);
			L->experts = NULL;
			free_weight_buf(&L->ffn_norm_w.buf);
			free_weight_buf(&L->gate_w.buf);
			free_weight_buf(&L->up_w.buf);
			free_weight_buf(&L->gate_up_w.buf);
			free(L->gate_up_fused_host);
			L->gate_up_fused_host = NULL;
			free_weight_buf(&L->down_w.buf);
			if (m->arch_info->has_attn_post_norm)
				free_weight_buf(&L->post_attn_norm_w.buf);
			if (m->arch_info->has_ffn_post_norm)
				free_weight_buf(&L->post_ffn_norm_w.buf);
			if (m->arch_info->has_qk_norm) {
				free_weight_buf(&L->attn_q_norm_w.buf);
				free_weight_buf(&L->attn_k_norm_w.buf);
			}
			if (m->has_per_layer_embeddings) {
				free_weight_buf(&L->ple_post_norm_w.buf);
				free_weight_buf(&L->ple_inp_gate_w.buf);
				free_weight_buf(&L->ple_proj_w.buf);
			}
			if (m->arch_info->has_layer_output_scale) {
				free_weight_buf(&L->layer_out_scale_w.buf);
			}
		}
		free_weight_buf(&m->tok_embd.buf);
		free_weight_buf(&m->output_norm_w.buf);
		free_weight_buf(&m->output_w.buf);
		free_weight_buf(&m->mtp_eh_proj.buf);
		free_weight_buf(&m->mtp_enorm.buf);
		free_weight_buf(&m->mtp_hnorm.buf);
		free_weight_buf(&m->mtp_shared_head_norm.buf);
		if (m->has_per_layer_embeddings) {
			free_weight_buf(&m->layer_dims.per_layer_tok_embd.buf);
			free_weight_buf(&m->layer_dims.per_layer_model_proj.buf);
			if (m->layer_dims.per_layer_model_proj.type == GGML_TYPE_F32 &&
				m->layer_dims.per_layer_model_proj.host_ptr &&
				m->layer_dims.per_layer_model_proj.host_ptr !=
					m->layer_dims.per_layer_tok_embd.host_ptr) {
				free((void *)m->layer_dims.per_layer_model_proj.host_ptr);
				m->layer_dims.per_layer_model_proj.host_ptr = NULL;
			}
			free_weight_buf(&m->layer_dims.per_layer_proj_norm_w.buf);
		}
		if (m->rope_freqs_count > 0)
			free_weight_buf(&m->rope_freqs_w.buf);
	}

	free(m->layers);
	free(m->layer_dims.is_global_layer);
	free(m->layer_dims.ffn_lengths);
	free(m->layer_dims.n_kv_heads_per_layer);
	gguf_free(&m->gctx);
	free(m->model_path);
	m->model_path = NULL;
	free((char *)m->repack_config);
	m->repack_config = NULL;

	if (m->recipe) {
		recipe_free(m->recipe);
		m->recipe = NULL;
	}
	free(m->wrefs_by_layer);
	m->wrefs_by_layer = NULL;

	if (m->layer_backends) {
		if (m->owns_backend) {
			for (int i = 0; i < m->n_layer_backends; i++) {
				if (m->layer_backends[i] && m->layer_backends[i] != m->backend &&
					!backend_has_cap(m->layer_backends[i], BCAP_IS_HOST)) {
					backend_destroy(m->layer_backends[i]);
				}
			}
		}
		free(m->layer_backends);
		m->layer_backends	= NULL;
		m->n_layer_backends = 0;
	}

	if (m->owns_backend)
		backend_destroy(m->backend);

	memset(m, 0, sizeof(*m));
}
