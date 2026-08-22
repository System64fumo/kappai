#ifndef CPU_COMMON_H
#define CPU_COMMON_H

#include "backend/backend.h"
#include "backend/cpu/scalar/quants.h"
#include "log.h"
#include "threadpool.h"
#include <stdlib.h>

typedef struct {
	quant_scratch qscratch;
	float		 *scores;
	int			  scores_cap;
} cpu_thread_scratch;

typedef struct {
	float		*buf;
	size_t		 cap;
	const float *layer_c_cached;
	int			 layer_cached;
	int			 total_dim_cached;
	int			 kv_lora_cached;
	int			 qk_rope_cached;
	int			 half_rope_cached;
	int			 n_pos_cached;
} cpu_mla_krot_cache;

typedef struct {
	double theta_scale;
	double theta_cached;
	int	   head_dim_cached;

	float		*cs;
	int			 cs_cap;
	int			 pos_cached;
	float		 theta_cs_cached;
	int			 head_dim_cs_cached;
	const float *freq_factors_cached;
} cpu_rope_cs_cache;

typedef struct {
	quant_scratch qscratch;

	void		 *xq8_buf;
	size_t		  xq8_buf_cap;
	float		 *scores;
	int			  scores_cap;
	int			  kv_head_dim_max;
	kv_quant_type kv_quant;

	size_t kv_block_stride;
	size_t kv_layer_stride;
	size_t kv_kvh_stride;

	uint16_t *kv_k;
	uint16_t *kv_v;

	size_t *kv_layer_off;

	float *residual_tmp;
	size_t residual_tmp_cap;

	cpu_mla_krot_cache mla_krot;
	cpu_rope_cs_cache  rope_cs;

	tpool			   *pool;
	int					n_threads;
	cpu_thread_scratch *thread_scratch;

	quant_scratch matmul_multi_scratch[4];
} cpu_priv;

void feat_add(char *buf, size_t cap, const char *name);
void detect_features(char *buf, size_t cap);

static inline status_code cpu_scratch_grow(void **buf, size_t *cap_bytes, size_t need_bytes) {
	if (*cap_bytes >= need_bytes)
		return OK;
	free(*buf);
	*buf = malloc(need_bytes);
	if (!*buf) {
		*cap_bytes = 0;
		return ERR_OUT_OF_MEMORY;
	}
	*cap_bytes = need_bytes;
	return OK;
}

static inline void *cpu_ptr(const buffer *b) {
	if (b->owner && !backend_has_cap(b->owner, BCAP_IS_HOST) &&
		!backend_has_cap(b->owner, BCAP_HOST_VISIBLE_BUFFERS)) {
		ERROR("cpu_ptr: buffer owned by backend '%s' passed to CPU compute path; "
			  "this backend is missing a native op and silently fell back to host, "
			  "which corrupts memory for non-CPU-resident buffers",
			  b->owner->name);
		abort();
	}
	return (void *)((char *)b->handle + b->offset);
}

static inline void cpu_kv_put_q8_0_head(uint8_t *kd, uint8_t *vd, const float *kfh,
										const float *vfh, int head_dim) {
	int	  n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	float padded[KV_Q8_0_BLOCK];
	for (int b = 0; b < n_blocks; b++) {
		int base = b * KV_Q8_0_BLOCK;
		int n	 = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;
		const float *ksrc = kfh + base;
		const float *vsrc = vfh + base;
		if (n < KV_Q8_0_BLOCK) {
			memset(padded, 0, sizeof(padded));
			memcpy(padded, ksrc, (size_t)n * sizeof(float));
			quantize_q8_0(padded, (q8_0_block *)(kd + ((size_t)b * KV_Q8_0_BLOCK_BYTES)),
						  KV_Q8_0_BLOCK);
			memset(padded, 0, sizeof(padded));
			memcpy(padded, vsrc, (size_t)n * sizeof(float));
			quantize_q8_0(padded, (q8_0_block *)(vd + ((size_t)b * KV_Q8_0_BLOCK_BYTES)),
						  KV_Q8_0_BLOCK);
		} else {
			quantize_q8_0(ksrc, (q8_0_block *)(kd + ((size_t)b * KV_Q8_0_BLOCK_BYTES)),
						  KV_Q8_0_BLOCK);
			quantize_q8_0(vsrc, (q8_0_block *)(vd + ((size_t)b * KV_Q8_0_BLOCK_BYTES)),
						  KV_Q8_0_BLOCK);
		}
	}
}

static inline void cpu_kv_put_f16_head(uint16_t *kd, uint16_t *vd, const float *kfh,
									   const float *vfh, int head_dim) {
	for (int i = 0; i < head_dim; i++) {
		kd[i] = f32_to_f16(kfh[i]);
		vd[i] = f32_to_f16(vfh[i]);
	}
}

typedef struct {
	const uint16_t *kl_base, *vl_base;
	const float	   *qf;
	float		   *outf;
	int				n_groups, head_dim, hd_stride, n_pos, flash_attn;
	float			scale;
	size_t			kvh_stride;
	cpu_priv	   *p;
	kv_quant_type	kv_quant;
} cpu_attn_job;

typedef struct {
	const uint16_t *kl_base, *vl_base;
	const float	   *qf;
	float		   *outf;
	int				n_groups, head_dim, hd_stride;
	int				n_heads, pos_start, m, sliding_window, flash_attn;
	float			scale;
	size_t			kvh_stride;
	cpu_priv	   *p;
	const int	   *bitrev_perm;
	kv_quant_type	kv_quant;
} cpu_attn_batch_job;

typedef struct {
	const float *g;
	const float *u;
	float		*o;
	int			 activation;
} cpu_ffn_down_act_args;

typedef struct {
	float		*vec;
	int			 n_heads, head_dim, pos_start;
	const float *rope_cos_base, *rope_sin_base;
	int			 rope_neox;
} cpu_rope_batch_job;

typedef struct {
	float		*q, *k;
	int			 n_heads, n_kv_heads, head_dim, pos_start;
	const float *rope_cos_base, *rope_sin_base;
	int			 rope_neox;
} cpu_rope_qk_batch_job;

typedef struct {
	float		*x;
	const float *y;
	int			 n;
} cpu_add_batch_job;

typedef struct {
	const float *g, *u;
	float		*o;
	int			 n;
	int			 activation;
} cpu_ffn_act_batch_job;

static inline void cpu_run_batch(tpool *pool, int m, tpool_chunk_fn chunk, void *job) {
	if (tpool_current_tid() < 0 && pool && m >= 2)
		tpool_parallel_for(pool, m, 1, chunk, job);
	else
		chunk(0, m, 0, job);
}

#endif