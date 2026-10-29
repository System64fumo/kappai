#ifndef CPU_COMMON_H
#define CPU_COMMON_H

#include "backend/backend.h"
#include "backend/cpu/quants.h"
#include "log.h"
#include "threadpool.h"
#include <stdlib.h>

struct cpu_cache_info {
	int	   line_bytes;
	size_t l1d_bytes;
	size_t l2_bytes;
	size_t l3_bytes;
	int	   n_cpus;
	int	   detected;
};

extern cpu_cache_info g_cpu_cache;

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>

static inline size_t cpu_host_mem_available(void) {
	mach_port_t host = mach_host_self();
	if (host == MACH_PORT_NULL)
		return 0;
	vm_size_t			   page_size = 0;
	vm_statistics64_data_t vm		 = {0};
	mach_msg_type_number_t count	 = HOST_VM_INFO64_COUNT;
	kern_return_t		   status	 = host_page_size(host, &page_size);
	if (status == KERN_SUCCESS)
		status = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count);
	mach_port_deallocate(mach_task_self(), host);
	if (status != KERN_SUCCESS || page_size == 0 ||
		(size_t)count * sizeof(integer_t) <
			offsetof(vm_statistics64_data_t, inactive_count) + sizeof(vm.inactive_count))
		return 0;
	uint64_t pages = (uint64_t)vm.free_count + vm.inactive_count;
	return pages > SIZE_MAX / page_size ? SIZE_MAX : (size_t)pages * (size_t)page_size;
}

static inline size_t cpu_host_mem_total(void) {
	uint64_t total = 0;
	size_t	 len   = sizeof(total);
	if (sysctlbyname("hw.memsize", &total, &len, NULL, 0) != 0 || len != sizeof(total))
		return 0;
	return total > SIZE_MAX ? SIZE_MAX : (size_t)total;
}
#endif

#define CPU_ELEMWISE_MIN_PER_THREAD 4096
#define MLA_KROT_PARALLEL_MIN_POS 64
#define ATTN_BITREV_MIN_M 8

#define CPU_BACKEND_CAPS                                                                           \
	(BCAP_IS_HOST | BCAP_MULTI_MATMUL | BCAP_ROPE_QK_FUSED | BCAP_MATMUL_RESIDUAL |                \
	 BCAP_MATMUL_QONLY | BCAP_RMSNORM_ADD | BCAP_MATMUL_FFN_DOWN | BCAP_KV_QUANT_Q8_0 |            \
	 BCAP_KV_POS_CAP | BCAP_KV_SHIFT | BCAP_KV_LAYER_DIMS)

status_code cpu_backend_fill(backend *out, status_code (*init)(backend *, int));

#define CPU_BACKEND_REGISTER(name_str, ctor_fn, prio, desc_str, init_fn)                           \
	static status_code ctor_fn(backend *out) {                                                     \
		memset(out, 0, sizeof(*out));                                                              \
		out->name	  = name_str;                                                                  \
		out->priority = prio;                                                                      \
		out->caps	  = CPU_BACKEND_CAPS;                                                          \
		out->desc	  = desc_str;                                                                  \
		return cpu_backend_fill(out, init_fn);                                                     \
	}                                                                                              \
	BACKEND_REGISTER(name_str, ctor_fn)

int32_t cpu_argmax_f32(const float *logits, int vocab);

typedef struct {
	quant_scratch qscratch;
	float		 *scores;
	size_t		  scores_cap;
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
	double		 theta_scale;
	double		 theta_cached;
	int			 head_dim_cached;
	float		*cs;
	size_t		 cs_cap;
	int			 pos_cached;
	float		 theta_cs_cached;
	int			 head_dim_cs_cached;
	const float *freq_factors_cached;
} cpu_rope_cs_cache;

typedef struct {
	uint32_t w_type;
	status_code (*prepare)(const void *xq, size_t row_stride_blocks, int k, int m, void **out,
						   size_t *capacity);
	void (*matmul)(const void *w, const void *packed, float *y, int y_row_stride, int n, int k,
				   int m);
} cpu_matmul_packing;

typedef struct {
	quant_scratch			  qscratch;
	const cpu_matmul_packing *matmul_packing;
	tpool_chunk_fn			  shortconv_centered_chunk;
	void					 *matmul_packed;
	size_t					  matmul_packed_cap;

	void		 *xq8_buf;
	size_t		  xq8_buf_cap;
	float		 *scores;
	size_t		  scores_cap;
	size_t		  scores_want;
	kv_quant_type kv_quant;

	size_t *kv_pos_cap;
	size_t *kv_base_pos;

	size_t *kv_layer_off;
	size_t	kv_k_bytes;
	int		kv_n_layers;

	float *residual_tmp;
	size_t residual_tmp_cap;

	int *bitrev_perm_cache;
	int	 bitrev_perm_cap;

	float_buf		   full_attn_kv;
	cpu_mla_krot_cache mla_krot;
	cpu_rope_cs_cache  rope_cs;

	tpool			   *pool;
	int					n_threads;
	cpu_thread_scratch *thread_scratch;

	quant_scratch matmul_multi_scratch[4];
} cpu_priv;

static inline quant_scratch *cpu_scratch_for_tid(cpu_priv *p, int tid) {
	if (p->thread_scratch && tid >= 0 && tid < p->n_threads)
		return &p->thread_scratch[tid].qscratch;
	return &p->qscratch;
}

void feat_add(char *buf, size_t cap, const char *name);
void detect_features(char *buf, size_t cap);

static inline status_code cpu_buf_grow(void **p, size_t *cap_bytes, size_t need_bytes,
									   size_t align) {
	if (*cap_bytes >= need_bytes)
		return OK;
	free(*p);
	*p			 = NULL;
	size_t bytes = align > 1 ? ((need_bytes + align - 1) / align) * align : need_bytes;
	if (align > 1)
		*p = aligned_alloc(align, bytes);
	else
		*p = malloc(bytes);
	if (!*p) {
		*cap_bytes = 0;
		return ERR_OUT_OF_MEMORY;
	}
	*cap_bytes = bytes;
	return OK;
}

static inline float *cpu_grow_scores(cpu_priv *p, int tid, int need) {
	cpu_thread_scratch *ts =
		(p->thread_scratch && tid >= 0 && tid < p->n_threads) ? &p->thread_scratch[tid] : NULL;
	float **buf;
	size_t *cap;
	if (ts) {
		buf = &ts->scores;
		cap = &ts->scores_cap;
	} else if (p->thread_scratch && p->n_threads > 0) {
		buf = &p->thread_scratch[0].scores;
		cap = &p->thread_scratch[0].scores_cap;
	} else {
		buf = &p->scores;
		cap = &p->scores_cap;
	}
	if (*cap < (size_t)need) {
		size_t want = (size_t)need > p->scores_want ? (size_t)need : p->scores_want;
		free(*buf);
		*buf = xmalloc(want * sizeof(float));
		*cap = want;
	}
	return *buf;
}

static inline size_t cpu_kv_uniform_rows(const cpu_priv *p, const kv_desc *desc) {
	if (!p->kv_pos_cap || !desc->n_kv_layers)
		return (size_t)desc->n_ctx;
	size_t r = p->kv_pos_cap[0];
	for (int i = 1; i < desc->n_kv_layers; i++)
		if (p->kv_pos_cap[i] < r)
			r = p->kv_pos_cap[i];
	return r;
}

static inline void cpu_kv_layer_check(const cpu_priv *p, int layer) {
	if (p->kv_n_layers > 0 && (layer < 0 || layer >= p->kv_n_layers)) {
		ERROR("kv: layer %d is outside the %d allocated KV layers (shared-KV layers must be "
			  "remapped to their store layer)",
			  layer, p->kv_n_layers);
		abort();
	}
}

static inline size_t cpu_kv_layer_rows(const cpu_priv *p, int layer, int n_ctx) {
	cpu_kv_layer_check(p, layer);
	if (p->kv_pos_cap && p->kv_pos_cap[layer] > 0)
		return p->kv_pos_cap[layer];
	return (size_t)n_ctx;
}

static inline size_t cpu_kv_layer_base(const cpu_priv *p, int layer, size_t layer_stride,
									   size_t elem_bytes) {
	if (p->kv_layer_off)
		return p->kv_layer_off[layer] / elem_bytes;
	return (size_t)layer * layer_stride;
}

typedef struct {
	int			hd_stride;
	size_t		kvh_stride;
	const void *k_base;
	const void *v_base;
} cpu_kv_layer_view;

static inline cpu_kv_layer_view cpu_kv_layer_view_get(const cpu_priv *p, const buffer *k_cache,
													  const buffer *v_cache, int layer,
													  int n_kv_heads, int head_dim, size_t rows) {
	cpu_kv_layer_view v;
	if (p->kv_quant == KV_QUANT_Q8_0) {
		size_t n_blocks = ((size_t)head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		v.hd_stride		= (int)(n_blocks * KV_Q8_0_BLOCK_BYTES);
		v.kvh_stride	= rows * (size_t)v.hd_stride;
		size_t base		= cpu_kv_layer_base(p, layer, (size_t)n_kv_heads * v.kvh_stride, 1);
		v.k_base		= (const uint8_t *)k_cache->handle + base;
		v.v_base		= (const uint8_t *)v_cache->handle + base;
		return v;
	}
	v.hd_stride	 = head_dim;
	v.kvh_stride = rows * (size_t)head_dim;
	size_t base	 = cpu_kv_layer_base(p, layer, (size_t)n_kv_heads * v.kvh_stride, sizeof(uint16_t));
	v.k_base	 = (const uint16_t *)k_cache->handle + base;
	v.v_base	 = (const uint16_t *)v_cache->handle + base;
	return v;
}

static inline void cpu_kv_window_to_slots_base(size_t b, int *attn_start, int *n_pos) {
	if (!b)
		return;
	if (*attn_start < (int)b) {
		*n_pos -= (int)b - *attn_start;
		*attn_start = 0;
		if (*n_pos < 0)
			*n_pos = 0;
	} else {
		*attn_start -= (int)b;
	}
}

static inline void cpu_kv_window_to_slots(const cpu_priv *p, int layer, int *attn_start,
										  int *n_pos) {
	if (!p->kv_base_pos)
		return;
	cpu_kv_window_to_slots_base(p->kv_base_pos[layer], attn_start, n_pos);
}

static inline size_t cpu_kv_put_slot(cpu_priv *p, int layer, int pos, int n_ctx, void *kbase,
									 void *vbase, size_t layer_off, size_t kvh_stride,
									 size_t row_bytes, int n_kv_heads) {
	if (!p->kv_base_pos || !p->kv_pos_cap)
		return (size_t)pos;
	cpu_kv_layer_check(p, layer);
	size_t pcap = p->kv_pos_cap[layer];
	if (pcap == 0 || pcap >= (size_t)n_ctx)
		return (size_t)pos;
	size_t bpos = p->kv_base_pos[layer];
	if (p->kv_k_bytes) {
		size_t need = layer_off + (size_t)n_kv_heads * kvh_stride;
		if (need > p->kv_k_bytes) {
			ERROR("kv compaction: layer %d needs %zu bytes (off %zu + %d heads x %zu stride) "
				  "but the cache holds %zu -- layout does not match this buffer",
				  layer, need, layer_off, n_kv_heads, kvh_stride, p->kv_k_bytes);
			abort();
		}
	}
	if ((size_t)pos >= bpos + pcap) {
		size_t shift = (size_t)pos - bpos - pcap + 1;
		if (shift > pcap)
			shift = pcap;
		for (int kvh = 0; kvh < n_kv_heads; kvh++) {
			char *kr = (char *)kbase + layer_off + ((size_t)kvh * kvh_stride);
			char *vr = (char *)vbase + layer_off + ((size_t)kvh * kvh_stride);
			memmove(kr, kr + shift * row_bytes, (pcap - shift) * row_bytes);
			memmove(vr, vr + shift * row_bytes, (pcap - shift) * row_bytes);
		}
		bpos += shift;
		p->kv_base_pos[layer] = bpos;
	}
	if ((size_t)pos < bpos)
		return 0;
	return (size_t)pos - bpos;
}

static inline void *cpu_ptr(const buffer *b) {
	if (b->owner && !backend_has_cap(b->owner, BCAP_IS_HOST | BCAP_HOST_VISIBLE_BUFFERS |
												   BCAP_HOST_RESIDENT_BUFFERS)) {
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
	f32_to_f16_row(kfh, kd, head_dim);
	f32_to_f16_row(vfh, vd, head_dim);
}

static inline int *cpu_bitrev_perm_get(cpu_priv *p, int m_pow2) {
	if (p->bitrev_perm_cap == m_pow2 && p->bitrev_perm_cache)
		return p->bitrev_perm_cache;
	free(p->bitrev_perm_cache);
	p->bitrev_perm_cache = xmalloc((size_t)m_pow2 * sizeof(int));
	p->bitrev_perm_cap	 = m_pow2;
	unsigned bits		 = 0;
	while ((1u << bits) < (unsigned)m_pow2)
		bits++;
	for (int r = 0; r < m_pow2; r++) {
		unsigned rev = 0, tmp = (unsigned)r;
		for (unsigned b = 0; b < bits; b++) {
			rev = (rev << 1) | (tmp & 1u);
			tmp >>= 1;
		}
		p->bitrev_perm_cache[r] = (int)rev;
	}
	return p->bitrev_perm_cache;
}

typedef struct {
	const uint16_t *kl_base, *vl_base;
	const float	   *qf;
	float		   *outf;
	int				n_groups, head_dim, hd_stride, n_pos, flash_attn;
	int				n_heads;
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
	int				n_keys;
	float			scale;
	size_t			kvh_stride;
	cpu_priv	   *p;
	const int	   *bitrev_perm;
	kv_quant_type	kv_quant;
	size_t			kv_rows, kv_base;
} cpu_attn_batch_job;
typedef struct {
	const float *g, *u;
	float		*o;
} cpu_ffn_act_args;

typedef struct {
	const float *qf;
	float		*outf;
	const float *layer_c;
	const float *k_b;
	const float *v_b;
	const float *k_pe_rot_all;
	const float *rope_cos_base;
	const float *rope_sin_base;
	int			 qk_head, qk_rope, qk_nope, v_head, kv_lora, n_pos, half_rope, pos;
	float		 scale;
} cpu_mla_job;

typedef struct {
	const float *layer_c;
	float		*k_pe_rot_all;
	const float *rope_cos_base;
	const float *rope_sin_base;
	int			 total_dim;
	int			 kv_lora;
	int			 qk_rope;
	int			 half_rope;
} cpu_mla_krot_job;

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
	const float *x, *w, *residual;
	float		*y;
	int			 n;
	float		 eps;
	float		 out_scale;
} cpu_rmsnorm_add_batch_job;

typedef struct {
	const float *g, *u;
	float		*o;
	int			 n;
	int			 activation;
} cpu_ffn_act_batch_job;

typedef struct {
	float *x;
	float  inv_cap;
	float  cap;
} cpu_softcap_job;

typedef struct {
	const float *mixed;
	float		*q, *gate;
	int			 n_heads, head_dim;
} cpu_split_qgate_job;

typedef struct {
	float		*out;
	const float *gate;
	int			 n;
} cpu_attn_output_gate_job;

typedef struct {
	float		*q, *k;
	const float *cos_base, *sin_base;
	int			 qn, kn, half, rope_dim, n_heads, n_kv_heads, head_dim;
	int			 pos0;
} cpu_partial_rope_qk_job;

static inline void cpu_run_batch_full(tpool *pool, int m, int grain, int min_m,
									  tpool_chunk_fn chunk, void *job) {
	if (tpool_current_tid() < 0 && pool && m >= min_m)
		tpool_parallel_for(pool, m, grain, chunk, job);
	else
		chunk(0, m, 0, job);
}

static inline void cpu_run_batch(tpool *pool, int m, tpool_chunk_fn chunk, void *job) {
	cpu_run_batch_full(pool, m, 1, 2, chunk, job);
}

typedef struct {
	int			 dim, n_hc;
	const float *x;
	float		*hc;
} hc_init_host_desc;

typedef struct {
	int			 dim, n_hc;
	const float *hc;
	float		*x;
} hc_final_host_desc;

typedef struct {
	int			   dim, n_hc, sinkhorn_iters;
	float		   norm_eps, eps;
	const float	  *stream;
	backend_weight fn;
	const float	  *base, *scale;
	float		  *x, *coef;
	float		  *ws;
} hc_pre_host_desc;

typedef struct {
	int			 dim, n_hc;
	const float *sub, *coef, *hc_cur;
	float		*hc_next;
} hc_post_host_desc;

typedef struct {
	int			   dim, n_heads, head_dim, conv_kernel;
	float		   gate_lower_bound, norm_eps;
	const float	  *x;
	backend_weight wq, wk, wv, wo, f_a, f_b, g_a, g_b, beta;
	const float	  *conv_q, *conv_k, *conv_v, *dt_bias, *ssm_a, *out_norm;
	float		  *conv_state, *state;
	float		  *out;
	float		  *ws;
} kda_host_desc;

typedef struct {
	int			   dim, n_heads, kv_lora, qk_head, v_head, q_lora, pos, n_threads;
	float		   norm_eps;
	const float	  *x;
	backend_weight q_a, q_b, kv_a, wo;
	const float	  *q_a_norm, *kv_a_norm, *k_b, *v_b;
	float		  *latent_cache;
	float		  *out;
	float		  *ws;
} mla_absorbed_host_desc;

status_code cpu_hc_init(backend *self, const hc_init_host_desc *d);
status_code cpu_hc_final(backend *self, const hc_final_host_desc *d);
status_code cpu_hc_pre(backend *self, const hc_pre_host_desc *d);
status_code cpu_hc_post(backend *self, const hc_post_host_desc *d);
status_code cpu_kda(backend *self, const kda_host_desc *d);
status_code cpu_mla_absorbed(backend *self, const mla_absorbed_host_desc *d);

status_code cpu_probe(void);
status_code cpu_init(backend *self, int device_index);
void		cpu_free(backend *self);
status_code cpu_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out);
status_code cpu_buffer_alloc_scratch(backend *self, size_t size, buffer *out);
void		cpu_buffer_free(backend *self, buffer *buf);
status_code cpu_copy_buffer(backend *self, const buffer *src, buffer *dst, int n);
status_code cpu_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n);
status_code cpu_buffer_write_f32(backend *self, buffer *buf, const float *host_src, int n);
status_code cpu_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out);
status_code cpu_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos, const buffer *k_in,
					   const buffer *v_in, int n_kv_heads, int head_dim, int n_ctx,
					   int n_kv_heads_active);
status_code cpu_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
							 const buffer *k_in, const buffer *v_in, int in_row_stride,
							 int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active, int m);
status_code cpu_kv_shift(backend *self, buffer *k, buffer *v, int layer,
						 const kv_shift_chunk *chunks, int n_chunks, const float *cos_tbl,
						 const float *sin_tbl, int n_kv_heads, int head_dim, int n_ctx,
						 int n_kv_heads_active);
status_code cpu_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
							int token, int dim, buffer *x_out);
status_code cpu_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
						float eps);
status_code cpu_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x, buffer *y,
					   int n, int k);
status_code cpu_matmul_residual(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								const buffer *residual, buffer *y, int n, int k);
status_code cpu_matmul_batch(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							 buffer *y, int n, int k, int m);
status_code cpu_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
							 const buffer *x, buffer **y, const int *n_list, int k, int n_matmuls);
status_code cpu_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
								   const buffer *x, buffer **y, const int *n_list, int k,
								   int n_matmuls, int m);
status_code cpu_matmul_qonly(backend *self, const buffer *w, uint32_t w_type, const buffer *x_q8,
							 uint32_t q8_type, buffer *y, int n, int k, int m);
status_code cpu_prequantize_x(backend *self, const buffer *x, int k, uint32_t q8_type,
							  buffer *x_q8_out);
status_code cpu_rope(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
					 const float *rope_cos_base, const float *rope_sin_base);
status_code cpu_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
						int head_dim, int pos, const float *rope_cos_base,
						const float *rope_sin_base);
status_code cpu_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
						 const float *rope_cos_base, const float *rope_sin_base,
						 const float *freq_factors);
status_code cpu_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim, int pos_start,
							   const float *rope_cos_base, const float *rope_sin_base,
							   const float *freq_factors, int m);
status_code cpu_attention(backend *self, const buffer *q, const buffer *k_cache,
						  const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
						  int n_kv_heads, int head_dim, int n_ctx, int flash_attn, float scale,
						  int n_kv_heads_active);
status_code cpu_attention_swa(backend *self, const buffer *q, const buffer *k_cache,
							  const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
							  int n_kv_heads, int head_dim, int n_ctx, int flash_attn, float scale,
							  int sliding_window, int n_kv_heads_active);
status_code cpu_add_inplace(backend *self, buffer *x, const buffer *y, int n);
status_code cpu_ffn_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
							 int n);
status_code cpu_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up, buffer *out,
								int n, int activation);
status_code cpu_ffn_activate_fused_batch(backend *self, const buffer *fused, buffer *out, int n,
										 int activation, int m);
status_code cpu_rmsnorm_per_head(backend *self, const buffer *x, const buffer *w, buffer *y,
								 int n_heads, int head_dim, float eps);
status_code cpu_rmsnorm_per_head_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
									   int n_heads, int head_dim, float eps, int m);
status_code cpu_rmsnorm_noweight_batch(backend *self, const buffer *x, buffer *y, int n, float eps,
									   int m);
status_code cpu_rmsnorm_noweight_per_head_batch(backend *self, const buffer *x, buffer *y,
												int n_heads, int head_dim, float eps, int m);
status_code cpu_rmsnorm_noweight(backend *self, const buffer *x, buffer *y, int n, float eps);
status_code cpu_rmsnorm_noweight_per_head(backend *self, const buffer *x, buffer *y, int n_heads,
										  int head_dim, float eps);
status_code cpu_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx);
status_code cpu_gated_delta_net(backend *self, const gdn_desc *d);
status_code cpu_shortconv(backend *self, const shortconv_desc *d);
status_code cpu_repack_plan(backend *self, uint32_t type, uint64_t d0, uint64_t d1,
							uint32_t *re_type_out);
status_code cpu_repack_weight(backend *self, uint32_t type, const void *src, void *dst, int n_rows,
							  int k);
status_code cpu_dequant_row(backend *self, uint32_t type, const void *src, int n_elems, float *dst);
status_code cpu_moe_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
							 int n, float gate_scale, float up_scale, int use_gelu);
void		cpu_synchronize(backend *self);
status_code cpu_attention_mla(backend *self, const buffer *q, const buffer *kv_cache,
							  const buffer *k_b_w, const buffer *v_b_w, buffer *out, int layer,
							  int pos, int n_heads, int qk_head, int qk_rope, int qk_nope,
							  int v_head, int kv_lora, int n_ctx, const float *rope_cos_base,
							  const float *rope_sin_base, uint32_t k_b_type, uint32_t v_b_type,
							  float scale);
status_code cpu_kv_alloc_mla(backend *self, int n_layers, int n_ctx, int kv_lora, int qk_rope,
							 buffer *kv_out);
status_code cpu_kv_put_mla(backend *self, buffer *kv_cache, int layer, int pos,
						   const buffer *kv_a_in, const buffer *kv_a_norm_w, int kv_lora,
						   int qk_rope, int n_ctx, float eps);
tpool	   *cpu_get_pool(backend *self);
status_code cpu_matmul_thread_local(backend *self, const void *W, uint32_t w_type, const float *x,
									float *y, int n, int k, int tid);
size_t		cpu_mem_available(backend *self);
size_t		cpu_mem_total(backend *self);
status_code cpu_rmsnorm_batch(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
							  float eps, int m);
status_code cpu_add_batch(backend *self, buffer *x, const buffer *y, int n, int m);
status_code cpu_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n, int activation, int m);
status_code cpu_rope_batch(backend *self, buffer *vec, int n_heads, int head_dim, int pos_start,
						   const float *rope_cos_base, const float *rope_sin_base, int m);
status_code cpu_rope_qk_batch(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
							  int head_dim, int pos_start, const float *rope_cos_base,
							  const float *rope_sin_base, int m);
status_code cpu_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
								const buffer *v_cache, buffer *out, int layer, int pos_start,
								int n_heads, int n_kv_heads, int head_dim, int n_ctx,
								int flash_attn, float scale, int n_kv_heads_active, int m);
status_code cpu_attention_swa_batch(backend *self, const buffer *q, const buffer *k_cache,
									const buffer *v_cache, buffer *out, int layer, int pos_start,
									int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									int flash_attn, float scale, int sliding_window,
									int n_kv_heads_active, int m);
status_code cpu_rmsnorm_add(backend *self, const buffer *x, const buffer *w, const buffer *residual,
							buffer *y, int n, float eps, float out_scale);
status_code cpu_rmsnorm_add_batch(backend *self, const buffer *x, const buffer *w,
								  const buffer *residual, buffer *y, int n, float eps,
								  float out_scale, int m);
status_code cpu_scale_inplace(backend *self, buffer *x, float scale, int n);
status_code cpu_softcap(backend *self, buffer *x, float cap, int n);
status_code cpu_split_qgate(backend *self, const buffer *mixed, buffer *q, buffer *gate,
							int n_heads, int head_dim, int n_rows);
status_code cpu_attn_output_gate(backend *self, buffer *out, const buffer *gate, int n, int n_rows);
status_code cpu_partial_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
								int head_dim, int rope_dim, int pos_start,
								const float *rope_cos_base, const float *rope_sin_base, int n_rows);
status_code cpu_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
							float combine_scale);
status_code cpu_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type, const buffer *gate,
								const buffer *up, buffer *y, int n, int k, int activation);
status_code cpu_matmul_ffn_down_batch(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *gate, const buffer *up, buffer *y, int n, int k,
									  int activation, int m);
void		cpu_kv_free(backend *self, buffer *k, buffer *v);
void		cpu_ffn_down_act_chunk(int begin, int end, int tid, void *ctx);

void cpu_softcap_chunk(int begin, int end, int tid, void *ctx);
void cpu_split_qgate_chunk(int begin, int end, int tid, void *ctx);
void cpu_attn_output_gate_chunk(int begin, int end, int tid, void *ctx);
void cpu_partial_rope_qk_chunk(int begin, int end, int tid, void *ctx);
void cpu_rope_cs_fill(float *cs, int half, int pos, double theta_scale, const float *freq_factors);
status_code cpu_rope_cs_cached(cpu_priv *p, int head_dim, int pos, float theta,
							   const float *freq_factors, float **cs_out);

cpu_attn_batch_job cpu_full_attn_job(backend *self, const buffer *q, const buffer *k,
									 const buffer *v, buffer *out, int n_heads, int n_kv_heads,
									 int head_dim, int n_rows, int flash_attn, float scale);
status_code		   cpu_attention_full_batch(backend *self, const buffer *q, const buffer *k,
											const buffer *v, buffer *out, int n_heads, int n_kv_heads,
											int head_dim, int n_rows, int flash_attn, float scale);

#endif
