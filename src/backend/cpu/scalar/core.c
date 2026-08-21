#include "backend/backend.h"
#include "backend/cpu/common.h"
#include "backend/cpu/scalar/quants.h"
#include "memconfig.h"
#include "moe/moe_stream.h"
#include <execinfo.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CPU_MATMUL_MIN_ROWS_PER_THREAD 32
#define CPU_MATMUL_MULTI_MAX 8
#define ATTN_BITREV_MIN_M 8
#define ATTN_BITREV_STACK_MAX 256
#define CPU_QUANTIZE_MIN_ROWS_PER_THREAD 8
#define CPU_ELEMWISE_MIN_PER_THREAD 4096

typedef void (*qonly_fn)(const void *, const void *, size_t, float *, int, int, int, int);

typedef struct {
	qonly_fn qonly;
	uint32_t w_type;
	int		 q8_class;
} matmul_kernel;

typedef struct {
	const void			*W;
	uint32_t			 w_type;
	const void			*xq;
	const float			*xf;
	float				*y;
	int					 n, k, m;
	size_t				 row_stride;
	size_t				 xq_row_stride;
	const float			*bias;
	const float			*residual;
	int					 aliases_residual;
	int					 group_rows;
	const matmul_kernel *kernel;
	size_t				 xq_row_stride_blocks;
} cpu_matmul_job;

typedef struct {
	uint32_t	   w_type;
	int			   group_rows;
	tpool_chunk_fn chunk_fn;
} grouped_matmul_kernel;

typedef struct {
	quant_scratch *qs;
	int			   q8_class;
	const float	  *xf;
	int			   k;
	size_t		   row_stride;
	uint8_t		  *xq;
} cpu_quantize_x_rows_job;

typedef struct {
	cpu_matmul_job jobs[CPU_MATMUL_MULTI_MAX];
	int			   row_offset[CPU_MATMUL_MULTI_MAX + 1];
	int			   n_matmuls;
	int			   group_offset[CPU_MATMUL_MULTI_MAX + 1];
	int			   group_rows;
} cpu_matmul_multi_job;

typedef struct {
	const float *x, *w;
	float		*y;
	int			 n;
	float		 eps;
} cpu_rmsnorm_batch_job;

typedef struct {
	const float *qf;
	float		*outf;
	const float *k_b;
	const float *v_b;
	const float *layer_c;
	const float *k_pe_rot_all;
	const float *rope_cos_base;
	const float *rope_sin_base;
	int			 n_heads;
	int			 qk_head;
	int			 qk_rope;
	int			 qk_nope;
	int			 v_head;
	int			 kv_lora;
	int			 total_dim;
	int			 pos;
	float		 scale;
} cpu_attn_mla_job;

#define MATMUL_KERNEL(wt, fn, q8) {(qonly_fn)fn, wt, q8}

static const matmul_kernel k_kernels[] = {
	MATMUL_KERNEL(GGML_TYPE_Q4_0, matmul_q4_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_IQ4_NL, matmul_iq4_nl_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_Q8_0, matmul_q8_0_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_Q4_1, matmul_q4_1_q8_qonly_f32, 2),
	MATMUL_KERNEL(GGML_TYPE_Q5_0, matmul_q5_0_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_Q5_1, matmul_q5_1_q8_qonly_f32, 2),
	MATMUL_KERNEL(GGML_TYPE_Q6_K, matmul_q6_k_q8_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_Q4_K, matmul_q4_k_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_Q5_K, matmul_q5_k_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_IQ3_S, matmul_iq3_s_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_IQ3_S_RE, matmul_iq3_s_re_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_IQ3_S_RE8, matmul_iq3_s_re8_q8_k_qonly_f32, 3),
	MATMUL_KERNEL(GGML_TYPE_Q8_0_R8, matmul_q8_0_r8_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_Q4_0_R8, matmul_q4_0_r8_q8_qonly_f32, 1),
	MATMUL_KERNEL(GGML_TYPE_IQ4_NL_R8, matmul_iq4_nl_r8_q8_qonly_f32, 1),
};

static inline status_code cpu_scratch_grow_aligned(void **buf, size_t *cap_bytes, size_t need_bytes,
												   size_t align) {
	if (*cap_bytes >= need_bytes)
		return OK;
	free(*buf);
	*buf = aligned_alloc(align, need_bytes);
	if (!*buf) {
		*cap_bytes = 0;
		return ERR_OUT_OF_MEMORY;
	}
	*cap_bytes = need_bytes;
	return OK;
}

static inline status_code cpu_scratch_grow_floats(float **buf, int *cap_count, int need_count) {
	if (*cap_count >= need_count)
		return OK;
	free(*buf);
	*buf = malloc((size_t)need_count * sizeof(float));
	if (!*buf) {
		*cap_count = 0;
		return ERR_OUT_OF_MEMORY;
	}
	*cap_count = need_count;
	return OK;
}

static inline quant_scratch *cpu_scratch_for_tid(cpu_priv *p, int tid) {
	if (p->thread_scratch && tid >= 0 && tid < p->n_threads)
		return &p->thread_scratch[tid].qscratch;
	return &p->qscratch;
}

static status_code cpu_probe(void) {
	return OK;
}

static size_t cpu_mem_available(backend *self) {
	(void)self;
	return get_available_memory();
}

static size_t cpu_mem_total(backend *self) {
	(void)self;
	return get_total_memory();
}

__attribute__((weak)) void detect_features(char *buf, size_t cap) {
	snprintf(buf, cap, "generic");
}

static status_code cpu_init(backend *self, int device_index) {
	(void)device_index;
	cpu_priv *p = xcalloc(1, sizeof(cpu_priv));
	self->priv	= p;

	int n_threads = tpool_default_n_threads();
	if (n_threads > 1) {
		p->pool		 = tpool_create(n_threads);
		p->n_threads = tpool_n_threads(p->pool);
		tpool_set_worker_cleanup(p->pool, moe_stream_thread_cleanup);

		p->thread_scratch = xcalloc((size_t)p->n_threads, sizeof(cpu_thread_scratch));
	}

	char cpu_desc[128] = "";
	detect_features(cpu_desc, sizeof(cpu_desc));
	log_tag("CPU", "%d threads, %s", p->n_threads > 0 ? p->n_threads : 1, cpu_desc);

	return OK;
}

void feat_add(char *buf, size_t cap, const char *name) {
	size_t used = strlen(buf);
	int	   n	= snprintf(buf + used, cap - used, used ? ", %s" : "%s", name);
	if (n < 0 || (size_t)n >= cap - used)
		buf[used] = '\0';
}

static void cpu_free(backend *self) {
	cpu_priv *p = self->priv;
	if (!p)
		return;
	if (p->pool)
		tpool_destroy(p->pool);
	if (p->thread_scratch) {
		for (int i = 0; i < p->n_threads; i++) {
			free(p->thread_scratch[i].qscratch.q8_buf);
			free(p->thread_scratch[i].scores);
		}
		free(p->thread_scratch);
	}
	free(p->qscratch.q8_buf);
	free(p->xq8_buf);
	free(p->scores);

	for (int i = 0; i < 4; i++)
		free(p->matmul_multi_scratch[i].q8_buf);

	free(p->mla_krot.buf);
	free(p->residual_tmp);
	free(p->rope_cs.cs);
	free(p->kv_layer_off);
	free(p);
	self->priv = NULL;
}

static status_code cpu_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
	out->handle	  = (void *)desc->host_data;
	out->size	  = 0;
	out->offset	  = 0;
	out->host_ptr = desc->host_data;
	out->owner	  = self;
	return OK;
}

static status_code cpu_buffer_alloc_scratch(backend *self, size_t size, buffer *out) {
	out->handle	  = xmalloc(size);
	out->size	  = size;
	out->offset	  = 0;
	out->host_ptr = NULL;
	out->owner	  = self;
	return OK;
}

static void cpu_buffer_free(backend *self, buffer *buf) {
	(void)self;

	if (!buf || !buf->handle)
		return;
	if (buf->host_ptr)
		return;
	free(buf->handle);
	buf->handle = NULL;
	buf->size	= 0;
	buf->offset = 0;
}

static status_code cpu_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n) {
	(void)self;
	memcpy(host_dst, cpu_ptr(buf), (size_t)n * sizeof(float));
	return OK;
}

static status_code cpu_buffer_write_f32(backend *self, buffer *buf, const float *host_src, int n) {
	(void)self;
	memcpy(cpu_ptr(buf), host_src, (size_t)n * sizeof(float));
	return OK;
}

static status_code cpu_copy_buffer(backend *self, const buffer *src, buffer *dst, int n) {
	(void)self;

	memcpy(cpu_ptr(dst), cpu_ptr(src), (size_t)n * sizeof(float));
	return OK;
}

static status_code cpu_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out) {
	cpu_priv *p		   = self->priv;
	p->kv_head_dim_max = desc->head_dim;
	p->kv_quant		   = desc->kv_quant;

	free(p->kv_layer_off);
	p->kv_layer_off	   = NULL;
	int has_layer_dims = desc->layer_head_dim && desc->layer_n_kv_heads && desc->n_kv_layers > 0;
	if (has_layer_dims) {
		int n			= desc->n_kv_layers;
		p->kv_layer_off = xcalloc((size_t)n + 1, sizeof(size_t));
	}

	size_t total_bytes;
	if (desc->kv_quant == KV_QUANT_Q8_0) {
		size_t per_layer_uniform = 0;
		if (!has_layer_dims) {
			size_t n_blocks = ((size_t)desc->head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
			per_layer_uniform =
				(size_t)desc->n_kv_heads * desc->n_ctx * n_blocks * KV_Q8_0_BLOCK_BYTES;
		}
		total_bytes = 0;
		for (int i = 0; i < (has_layer_dims ? desc->n_kv_layers : 1); i++) {
			size_t per_layer = per_layer_uniform;
			if (has_layer_dims) {
				size_t hd		   = (size_t)desc->layer_head_dim[i];
				size_t n_blocks	   = (hd + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
				per_layer		   = (size_t)desc->layer_n_kv_heads[i] * desc->n_ctx * n_blocks *
									 KV_Q8_0_BLOCK_BYTES;
				p->kv_layer_off[i] = total_bytes;
			}
			total_bytes += per_layer;
		}
		if (!has_layer_dims)
			total_bytes =
				per_layer_uniform * (size_t)(desc->n_kv_layers > 0 ? desc->n_kv_layers : 1);
		if (has_layer_dims)
			p->kv_layer_off[desc->n_kv_layers] = total_bytes;
	} else {
		size_t per_layer_uniform = 0;
		if (!has_layer_dims) {
			per_layer_uniform = (size_t)desc->n_kv_heads * desc->n_ctx * desc->head_dim;
		}
		total_bytes = 0;
		for (int i = 0; i < (has_layer_dims ? desc->n_kv_layers : 1); i++) {
			size_t per_layer = per_layer_uniform;
			if (has_layer_dims) {
				per_layer		   = (size_t)desc->layer_n_kv_heads[i] * desc->n_ctx *
									 (size_t)desc->layer_head_dim[i];
				p->kv_layer_off[i] = total_bytes;
			}
			total_bytes += per_layer;
		}
		if (!has_layer_dims)
			total_bytes =
				per_layer_uniform * (size_t)(desc->n_kv_layers > 0 ? desc->n_kv_layers : 1);
		if (has_layer_dims)
			p->kv_layer_off[desc->n_kv_layers] = total_bytes;
		total_bytes *= sizeof(uint16_t);
		if (has_layer_dims) {
			for (int i = 0; i <= desc->n_kv_layers; i++)
				p->kv_layer_off[i] *= sizeof(uint16_t);
		}
	}

	k_out->handle	= xmalloc_aligned(total_bytes, 64);
	k_out->size		= total_bytes;
	k_out->host_ptr = NULL;
	k_out->owner	= self;
	v_out->handle	= xmalloc_aligned(total_bytes, 64);
	v_out->size		= total_bytes;
	v_out->host_ptr = NULL;
	v_out->owner	= self;

	madvise_hugepage(k_out->handle, total_bytes);
	madvise_hugepage(v_out->handle, total_bytes);
	prefault(k_out->handle, total_bytes);
	prefault(v_out->handle, total_bytes);

	p->kv_k = (uint16_t *)k_out->handle;
	p->kv_v = (uint16_t *)v_out->handle;

	if (desc->kv_quant == KV_QUANT_Q8_0) {
		size_t n_blocks	   = ((size_t)desc->head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		p->kv_block_stride = n_blocks * KV_Q8_0_BLOCK_BYTES;
	} else {
		p->kv_block_stride = (size_t)desc->head_dim * sizeof(uint16_t);
	}
	p->kv_layer_stride = (size_t)desc->n_kv_heads * desc->n_ctx * p->kv_block_stride;
	p->kv_kvh_stride   = (size_t)desc->n_ctx * p->kv_block_stride;

	if (desc->n_ctx > p->scores_cap) {
		free(p->scores);
		p->scores	  = xmalloc((size_t)desc->n_ctx * sizeof(float));
		p->scores_cap = desc->n_ctx;
	}

	if (p->thread_scratch) {
		for (int i = 0; i < p->n_threads; i++) {
			cpu_thread_scratch *ts = &p->thread_scratch[i];
			if (desc->n_ctx > ts->scores_cap) {
				free(ts->scores);
				ts->scores	   = xmalloc((size_t)desc->n_ctx * sizeof(float));
				ts->scores_cap = desc->n_ctx;
			}
		}
	}

	return OK;
}

static void cpu_kv_put_f16_head(uint16_t *kd, uint16_t *vd, const float *kfh, const float *vfh,
								int head_dim) {
	for (int i = 0; i < head_dim; i++) {
		kd[i] = f32_to_f16(kfh[i]);
		vd[i] = f32_to_f16(vfh[i]);
	}
}

__attribute__((weak)) status_code cpu_kv_put(backend *self, buffer *k, buffer *v, int layer,
											 int pos, const buffer *k_in, const buffer *v_in,
											 int n_kv_heads, int head_dim, int n_ctx,
											 int n_kv_heads_active) {
	cpu_priv	*p		  = self->priv;
	const float *kf		  = cpu_ptr(k_in);
	const float *vf		  = cpu_ptr(v_in);
	int			 n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

	if (p->kv_quant == KV_QUANT_Q8_0) {
		size_t	 pos_off = (size_t)pos * p->kv_block_stride;
		uint8_t *kd_base = (uint8_t *)k->handle;
		uint8_t *vd_base = (uint8_t *)v->handle;
		for (int kvh = 0; kvh < n_active; kvh++) {
			size_t layer_base =
				p->kv_layer_off ? p->kv_layer_off[layer] : ((size_t)layer * p->kv_layer_stride);
			size_t off = layer_base + ((size_t)kvh * p->kv_kvh_stride) + pos_off;
			cpu_kv_put_q8_0_head(kd_base + off, vd_base + off, kf + ((size_t)kvh * head_dim),
								 vf + ((size_t)kvh * head_dim), head_dim);
		}
		return OK;
	}

	uint16_t *kd_base	= k->handle;
	uint16_t *vd_base	= v->handle;
	int		  hd_stride = head_dim;
	size_t	  pos_off	= (size_t)pos * hd_stride;
	for (int kvh = 0; kvh < n_active; kvh++) {
		size_t layer_base = p->kv_layer_off
								? p->kv_layer_off[layer] / sizeof(uint16_t)
								: ((size_t)layer * (size_t)n_kv_heads * n_ctx * hd_stride);
		size_t off		  = layer_base + ((size_t)kvh * (size_t)n_ctx * hd_stride) + pos_off;
		cpu_kv_put_f16_head(kd_base + off, vd_base + off, kf + ((size_t)kvh * head_dim),
							vf + ((size_t)kvh * head_dim), head_dim);
	}
	return OK;
}

static void cpu_synchronize(backend *self) {
	(void)self;
}

static status_code cpu_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
								   int token, int dim, buffer *x_out) {
	(void)self;
	size_t		   row_stride = ggml_row_size(tok_embd_type, dim);
	const uint8_t *embd		  = (const uint8_t *)cpu_ptr(tok_embd) + ((size_t)token * row_stride);
	float		  *dst		  = (float *)cpu_ptr(x_out);
	dequant_row_dispatch(tok_embd_type, embd, dim, dst);
	return OK;
}

static status_code cpu_rmsnorm_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
									 int n, float eps, int m);
static status_code cpu_matmul_batch(backend *self, const buffer *w, uint32_t w_type,
									const buffer *x, buffer *y, int n, int k, int m);
__attribute__((weak)) status_code cpu_add_batch(backend *self, buffer *x, const buffer *y, int n,
												int m);
__attribute__((weak)) status_code cpu_ffn_activate_batch(backend *self, const buffer *gate,
														 const buffer *up, buffer *out, int n,
														 int activation, int m);

static status_code cpu_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
							   float eps) {
	return cpu_rmsnorm_batch(self, x, w, y, n, eps, 1);
}

static inline void cpu_matmul_add_bias_residual(float *restrict y, const float *restrict bias,
												const float *restrict residual, int n) {
	if (bias) {
		if (residual) {
			for (int i = 0; i < n; i++)
				y[i] += bias[i] + residual[i];
		} else {
			for (int i = 0; i < n; i++)
				y[i] += bias[i];
		}
	} else if (residual) {
		for (int i = 0; i < n; i++)
			y[i] += residual[i];
	}
}

static const matmul_kernel *matmul_kernel_lookup(uint32_t w_type) {
	for (size_t i = 0; i < (sizeof(k_kernels) / sizeof(k_kernels[0])); i++) {
		if (k_kernels[i].w_type == w_type)
			return &k_kernels[i];
	}
	return NULL;
}

static size_t cpu_matmul_q8_block_size(int q8_class) {
	switch (q8_class) {
	case 1:
		return sizeof(q8_0_block);
	case 2:
		return sizeof(q8_1_block);
	case 3:
		return sizeof(q8_k_block);
	default:
		return 0;
	}
}

static inline void cpu_matmul_job_prepare_kernel(cpu_matmul_job *j) {
	j->kernel				= matmul_kernel_lookup(j->w_type);
	j->xq_row_stride_blocks = 0;
	if (j->kernel)
		j->xq_row_stride_blocks =
			j->kernel->q8_class ? j->xq_row_stride / cpu_matmul_q8_block_size(j->kernel->q8_class)
								: 0;
}

static uint32_t cpu_matmul_q8_expected_type(int q8_class) {
	switch (q8_class) {
	case 1:
		return GGML_TYPE_Q8_0;
	case 2:
		return GGML_TYPE_Q8_1;
	case 3:
		return GGML_TYPE_Q8_K;
	default:
		return 0;
	}
}

static void cpu_matmul_rows_worker(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_matmul_job *j			  = ctx;
	int				n_sub		  = end - begin;
	const uint8_t *restrict W_sub = (const uint8_t *)j->W + ((size_t)begin * j->row_stride);
	float *restrict y_sub		  = j->y + begin;

	float		 save_stack[CPU_MATMUL_MIN_ROWS_PER_THREAD];
	const float *res_ptr;
	int			 res_off;
	if (j->m == 1 && j->aliases_residual && j->residual) {
		for (int i = 0; i < n_sub; i++)
			save_stack[i] = j->residual[begin + i];
		res_ptr = save_stack;
		res_off = 0;
	} else {
		res_ptr = j->residual;
		res_off = begin;
	}

	const matmul_kernel *entry = j->kernel;
	if (entry) {
		entry->qonly(W_sub, j->xq, j->xq_row_stride_blocks, y_sub, j->n, n_sub, j->k, j->m);
	} else if (j->w_type == GGML_TYPE_F32) {
		matmul_f32_f32_batch((const float *)W_sub, j->xf, y_sub, n_sub, j->k, j->m, j->k, j->n);
	} else if (j->w_type == GGML_TYPE_F16) {
		matmul_f16_f32_batch(W_sub, j->xf, y_sub, n_sub, j->k, j->m, j->k, j->n);
	} else if (j->w_type == GGML_TYPE_BF16) {
		matmul_bf16_f32_batch(W_sub, j->xf, y_sub, n_sub, j->k, j->m, j->k, j->n);
	} else {
		for (int row = 0; row < j->m; row++) {
			float *y = j->y + ((size_t)row * j->n) + begin;
			matmul_generic_f32(W_sub, j->w_type, j->xf + ((size_t)row * j->k), y, n_sub, j->k);
		}
	}

	if (j->m == 1)
		cpu_matmul_add_bias_residual(y_sub, j->bias ? j->bias + begin : NULL,
									 j->residual ? res_ptr + res_off : NULL, n_sub);
}

static void cpu_matmul_groups_worker(int begin, int end, int tid, void *ctx) {
	cpu_matmul_job *j  = ctx;
	int				gr = j->group_rows;
	cpu_matmul_rows_worker(begin * gr, end * gr, tid, ctx);
}

static const grouped_matmul_kernel k_grouped_kernels[] = {
	{GGML_TYPE_Q8_0_R8, Q8_0_R8_ROWS, cpu_matmul_groups_worker},
	{GGML_TYPE_Q4_0_R8, Q4_0_R8_ROWS, cpu_matmul_groups_worker},
	{GGML_TYPE_IQ3_S_RE8, IQ3_S_RE8_ROWS, cpu_matmul_groups_worker},
	{GGML_TYPE_IQ4_NL_R8, IQ4_NL_R8_ROWS, cpu_matmul_groups_worker},
};

static const grouped_matmul_kernel *grouped_matmul_kernel_lookup(uint32_t w_type) {
	for (size_t i = 0; i < (sizeof(k_grouped_kernels) / sizeof(k_grouped_kernels[0])); i++) {
		if (k_grouped_kernels[i].w_type == w_type)
			return &k_grouped_kernels[i];
	}
	return NULL;
}

static void cpu_matmul_dispatch_rows(tpool *pool, uint32_t w_type, int n_rows, void *job,
									 tpool_chunk_fn plain_chunk_fn) {
	const grouped_matmul_kernel *gk = grouped_matmul_kernel_lookup(w_type);
	if (!gk) {
		tpool_parallel_for(pool, n_rows, CPU_MATMUL_MIN_ROWS_PER_THREAD, plain_chunk_fn, job);
		return;
	}
	((cpu_matmul_job *)job)->group_rows = gk->group_rows;
	int n_groups						= n_rows / gk->group_rows;
	int min_groups_per_thr				= CPU_MATMUL_MIN_ROWS_PER_THREAD / gk->group_rows;
	if (min_groups_per_thr < 1)
		min_groups_per_thr = 1;
	tpool_parallel_for(pool, n_groups, min_groups_per_thr, gk->chunk_fn, job);
}

static size_t cpu_matmul_w_row_stride(uint32_t w_type, int k) {
	switch (w_type) {
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_0_R8:
		return (size_t)(k / 32) * sizeof(q4_0_block);
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ4_NL_R8:
		return (size_t)(k / 32) * sizeof(iq4_nl_block);
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_0_R8:
		return (size_t)(k / 32) * sizeof(q8_0_block);
	case GGML_TYPE_Q4_1:
		return (size_t)(k / 32) * sizeof(q4_1_block);
	case GGML_TYPE_Q5_0:
		return (size_t)(k / 32) * sizeof(q5_0_block);
	case GGML_TYPE_Q5_1:
		return (size_t)(k / 32) * sizeof(q5_1_block);
	case GGML_TYPE_F32:
		return (size_t)k * sizeof(float);
	case GGML_TYPE_F16:
		return (size_t)k * sizeof(uint16_t);
	case GGML_TYPE_BF16:
		return (size_t)k * sizeof(uint16_t);
	default:

		return ggml_row_size(w_type, (size_t)k);
	}
}

static size_t cpu_matmul_xq_row_stride(int q8_class, int k) {
	switch (q8_class) {
	case 1:
		return (size_t)(k / 32) * sizeof(q8_0_block);
	case 2:
		return (size_t)(k / 32) * sizeof(q8_1_block);
	case 3:
		return (size_t)(k / 256) * sizeof(q8_k_block);
	default:
		return 0;
	}
}

static void cpu_quantize_class(const float *restrict xf, void *restrict dst, int k, int q8_class) {
	switch (q8_class) {
	case 1:
		quantize_q8_0(xf, dst, k);
		break;
	case 2:
		quantize_q8_1(xf, dst, k);
		break;
	case 3:
		quantize_q8_k(xf, dst, k);
		break;
	default:
		break;
	}
}

static void cpu_quantize_x_rows_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_quantize_x_rows_job *j = ctx;
	for (int i = begin; i < end; i++) {
		const float *xi	 = j->xf + ((size_t)i * j->k);
		void		*dst = j->xq + ((size_t)i * j->row_stride);
		cpu_quantize_class(xi, dst, j->k, j->q8_class);
	}
}

static void *cpu_matmul_quantize_x_rows(backend *self, quant_scratch *qs, int q8_class,
										const float *xf, int k, int m, size_t row_stride) {
	quant_scratch_ensure(qs, (size_t)m * row_stride);
	uint8_t *xq = qs->q8_buf;

	cpu_priv *p			  = self->priv;
	int		  cur_tid	  = tpool_current_tid();
	int		  can_recurse = (cur_tid < 0);

	if (can_recurse && p->pool && m >= 2 * CPU_QUANTIZE_MIN_ROWS_PER_THREAD) {
		cpu_quantize_x_rows_job job = {
			.qs = qs, .q8_class = q8_class, .xf = xf, .k = k, .row_stride = row_stride, .xq = xq};
		tpool_parallel_for(p->pool, m, CPU_QUANTIZE_MIN_ROWS_PER_THREAD, cpu_quantize_x_rows_chunk,
						   &job);
		return xq;
	}

	for (int i = 0; i < m; i++) {
		const float *xi	 = xf + ((size_t)i * k);
		void		*dst = xq + ((size_t)i * row_stride);
		cpu_quantize_class(xi, dst, k, q8_class);
	}
	return xq;
}

static void *cpu_matmul_quantize_x(quant_scratch *qs, int q8_class, const float *x, int k) {
	if (q8_class < 1 || q8_class > 3)
		return NULL;
	quant_scratch_ensure(qs, cpu_matmul_xq_row_stride(q8_class, k));
	cpu_quantize_class(x, qs->q8_buf, k, q8_class);
	return qs->q8_buf;
}

void cpu_matmul_one(const void *restrict W, uint32_t w_type, const float *restrict x,
					float *restrict y, int n, int k, quant_scratch *qs) {
	const matmul_kernel *entry = matmul_kernel_lookup(w_type);
	if (entry && entry->q8_class) {
		void *xq = cpu_matmul_quantize_x(qs, entry->q8_class, x, k);
		if (xq) {
			entry->qonly(W, xq, 0, y, n, n, k, 1);
			return;
		}
	}
	switch (w_type) {
	case GGML_TYPE_F32:
		matmul_f32_f32(W, x, y, n, k);
		break;
	case GGML_TYPE_F16:
		matmul_f16_f32(W, x, y, n, k);
		break;
	case GGML_TYPE_BF16:
		matmul_bf16_f32(W, x, y, n, k);
		break;
	default:
		matmul_generic_f32(W, w_type, x, y, n, k);
		break;
	}
}

static void cpu_matmul_threaded_bias_residual(backend *self, const void *restrict W,
											  uint32_t w_type, const float *restrict x,
											  float *restrict y, int n, int k,
											  const float *restrict bias,
											  const float *restrict residual) {
	cpu_priv *p = self->priv;

	int cur_tid			 = tpool_current_tid();
	int can_recurse		 = (cur_tid < 0);
	int aliases_residual = (residual != NULL && residual == y);

	if (!can_recurse || !p->pool || n < 2 * CPU_MATMUL_MIN_ROWS_PER_THREAD) {
		const float *res = residual;
		if (aliases_residual) {
			status_code grow_st = cpu_scratch_grow_aligned(
				(void **)&p->residual_tmp, &p->residual_tmp_cap, (size_t)n * sizeof(float), 64);
			if (grow_st != OK)
				return;
			memcpy(p->residual_tmp, residual, (size_t)n * sizeof(float));
			res = p->residual_tmp;
		}
		quant_scratch *qs = cpu_scratch_for_tid(p, cur_tid);
		cpu_matmul_one(W, w_type, x, y, n, k, qs);
		cpu_matmul_add_bias_residual(y, bias, res, n);
		return;
	}

	cpu_matmul_job job = {.W				= W,
						  .w_type			= w_type,
						  .n				= n,
						  .k				= k,
						  .m				= 1,
						  .y				= y,
						  .xf				= x,
						  .bias				= bias,
						  .residual			= residual,
						  .aliases_residual = aliases_residual};
	cpu_matmul_job_prepare_kernel(&job);

	int q8_class = job.kernel ? job.kernel->q8_class : 0;
	if (q8_class) {
		job.xq = cpu_matmul_quantize_x(&p->qscratch, q8_class, x, k);
	} else if (w_type != GGML_TYPE_F32 && w_type != GGML_TYPE_BF16 && w_type != GGML_TYPE_F16) {
		const float *res = residual;
		if (aliases_residual) {
			status_code grow_st = cpu_scratch_grow_aligned(
				(void **)&p->residual_tmp, &p->residual_tmp_cap, (size_t)n * sizeof(float), 64);
			if (grow_st != OK)
				return;
			memcpy(p->residual_tmp, residual, (size_t)n * sizeof(float));
			res = p->residual_tmp;
		}
		cpu_matmul_one(W, w_type, x, y, n, k, &p->qscratch);
		cpu_matmul_add_bias_residual(y, bias, res, n);
		return;
	}
	job.row_stride = cpu_matmul_w_row_stride(w_type, k);

	cpu_matmul_dispatch_rows(p->pool, w_type, n, &job, cpu_matmul_rows_worker);
}

static void cpu_matmul_multi_worker(int begin, int end, int tid, void *ctx) {
	cpu_matmul_multi_job *mj = ctx;

	for (int i = 0; i < mj->n_matmuls; i++) {
		int lo = mj->row_offset[i];
		int hi = mj->row_offset[i + 1];
		if (end <= lo)
			break;
		int b = begin > lo ? begin : lo;
		int e = end < hi ? end : hi;
		if (b >= e)
			continue;
		cpu_matmul_rows_worker(b - lo, e - lo, tid, &mj->jobs[i]);
	}
}

static void cpu_matmul_multi_groups_worker(int begin, int end, int tid, void *ctx) {
	cpu_matmul_multi_job *mj = ctx;
	int					  gr = mj->group_rows;

	for (int i = 0; i < mj->n_matmuls; i++) {
		int lo = mj->group_offset[i];
		int hi = mj->group_offset[i + 1];
		if (end <= lo)
			break;
		int b = begin > lo ? begin : lo;
		int e = end < hi ? end : hi;
		if (b >= e)
			continue;
		int row_begin = (b - lo) * gr;
		int row_end	  = (e - lo) * gr;
		cpu_matmul_rows_worker(row_begin, row_end, tid, &mj->jobs[i]);
	}
}

static status_code cpu_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
									const buffer *x, buffer **y, const int *n_list, int k,
									int n_matmuls) {
	cpu_priv	*p	= self->priv;
	const float *xf = cpu_ptr(x);

	int total_rows = 0;
	for (int i = 0; i < n_matmuls; i++)
		total_rows += n_list[i];

	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);

	int has_grouped = 0;
	for (int i = 0; i < n_matmuls; i++)
		if (w_types[i] == GGML_TYPE_Q8_0_R8 || w_types[i] == GGML_TYPE_Q4_0_R8 ||
			w_types[i] == GGML_TYPE_IQ3_S_RE8 || w_types[i] == GGML_TYPE_IQ4_NL_R8)
			has_grouped = 1;

	if (!can_recurse || !p->pool || n_matmuls > CPU_MATMUL_MULTI_MAX ||
		total_rows < 2 * CPU_MATMUL_MIN_ROWS_PER_THREAD) {

		quant_scratch *qs = cpu_scratch_for_tid(p, cur_tid);

		void *xq_by_class[4] = {NULL, NULL, NULL, NULL};

		for (int i = 0; i < n_matmuls; i++) {
			float				*yf	   = cpu_ptr(y[i]);
			int					 n	   = n_list[i];
			const void			*W	   = cpu_ptr(w[i]);
			uint32_t			 w_t   = w_types[i];
			const matmul_kernel *entry = matmul_kernel_lookup(w_t);
			if (entry) {
				int q8_class = entry->q8_class;
				if (!xq_by_class[q8_class])
					xq_by_class[q8_class] = cpu_matmul_quantize_x(qs, q8_class, xf, k);
				entry->qonly(W, xq_by_class[q8_class], 0, yf, n, n, k, 1);
			} else {
				cpu_matmul_one(W, w_t, xf, yf, n, k, qs);
			}
		}
		return OK;
	}

	quant_scratch		*local_scratch	= p->matmul_multi_scratch;
	void				*xq_by_class[4] = {NULL, NULL, NULL, NULL};
	cpu_matmul_multi_job mj				= {.n_matmuls = n_matmuls};
	mj.row_offset[0]					= 0;
	for (int i = 0; i < n_matmuls; i++) {
		mj.row_offset[i + 1] = mj.row_offset[i] + n_list[i];

		cpu_matmul_job *j = &mj.jobs[i];
		j->W			  = cpu_ptr(w[i]);
		j->w_type		  = w_types[i];
		j->y			  = cpu_ptr(y[i]);
		j->k			  = k;
		j->n			  = n_list[i];
		j->m			  = 1;
		j->xf			  = xf;

		cpu_matmul_job_prepare_kernel(j);
		int q8_class = j->kernel ? j->kernel->q8_class : 0;
		if (q8_class) {
			if (!xq_by_class[q8_class])
				xq_by_class[q8_class] =
					cpu_matmul_quantize_x(&local_scratch[q8_class], q8_class, xf, k);
			j->xq = xq_by_class[q8_class];
		} else if (w_types[i] != GGML_TYPE_F32 && w_types[i] != GGML_TYPE_BF16) {
			for (int ii = 0; ii < n_matmuls; ii++)
				cpu_matmul_one(cpu_ptr(w[ii]), w_types[ii], xf, cpu_ptr(y[ii]), n_list[ii], k,
							   &p->qscratch);
			return OK;
		}
		j->row_stride = cpu_matmul_w_row_stride(w_types[i], k);
	}

	if (has_grouped) {
		int all_grouped = 1;
		int gr			= 0;
		for (int i = 0; i < n_matmuls; i++) {
			const grouped_matmul_kernel *gk = grouped_matmul_kernel_lookup(w_types[i]);
			if (!gk) {
				all_grouped = 0;
				break;
			}
			if (gr == 0)
				gr = gk->group_rows;
			else if (gk->group_rows != gr) {
				all_grouped = 0;
				break;
			}
		}

		if (all_grouped && gr > 0) {
			mj.group_rows	   = gr;
			mj.group_offset[0] = 0;
			int total_groups   = 0;
			for (int i = 0; i < n_matmuls; i++) {
				int n_groups_i = n_list[i] / gr;
				total_groups += n_groups_i;
				mj.group_offset[i + 1] = total_groups;
				mj.jobs[i].group_rows  = gr;
			}
			int min_groups_per_thr = CPU_MATMUL_MIN_ROWS_PER_THREAD / gr;
			if (min_groups_per_thr < 1)
				min_groups_per_thr = 1;
			tpool_parallel_for(p->pool, total_groups, min_groups_per_thr,
							   cpu_matmul_multi_groups_worker, &mj);
			return OK;
		}

		for (int i = 0; i < n_matmuls; i++)
			cpu_matmul_dispatch_rows(p->pool, w_types[i], n_list[i], &mj.jobs[i],
									 cpu_matmul_rows_worker);
		return OK;
	}

	tpool_parallel_for(p->pool, total_rows, CPU_MATMUL_MIN_ROWS_PER_THREAD, cpu_matmul_multi_worker,
					   &mj);

	return OK;
}

static status_code cpu_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
										  const buffer *x, buffer **y, const int *n_list, int k,
										  int n_matmuls, int m) {
	if (m <= 1) {
		buffer *y1[CPU_MATMUL_MULTI_MAX];
		for (int i = 0; i < n_matmuls; i++)
			y1[i] = y[i];
		return cpu_matmul_multi(self, w, w_types, x, y1, n_list, k, n_matmuls);
	}

	cpu_priv	*p	= self->priv;
	const float *xf = cpu_ptr(x);

	if (n_matmuls > CPU_MATMUL_MULTI_MAX)
		return ERR_INVALID_ARG;

	for (int i = 0; i < n_matmuls; i++) {
		if (grouped_matmul_kernel_lookup(w_types[i])) {
			for (int ii = 0; ii < n_matmuls; ii++) {
				status_code st =
					cpu_matmul_batch(self, w[ii], w_types[ii], x, y[ii], n_list[ii], k, m);
				if (st != OK)
					return st;
			}
			return OK;
		}
	}

	quant_scratch *local_scratch  = p->matmul_multi_scratch;
	void		  *xq_by_class[4] = {NULL, NULL, NULL, NULL};

	cpu_matmul_multi_job mj = {.n_matmuls = n_matmuls};
	mj.row_offset[0]		= 0;
	for (int i = 0; i < n_matmuls; i++) {
		mj.row_offset[i + 1] = mj.row_offset[i] + n_list[i];

		cpu_matmul_job *j = &mj.jobs[i];
		j->W			  = cpu_ptr(w[i]);
		j->w_type		  = w_types[i];
		j->y			  = cpu_ptr(y[i]);
		j->k			  = k;
		j->n			  = n_list[i];
		j->m			  = m;
		j->xf			  = xf;

		cpu_matmul_job_prepare_kernel(j);
		int q8_class = j->kernel ? j->kernel->q8_class : 0;
		if (q8_class) {
			if (!xq_by_class[q8_class]) {
				size_t row_stride	  = cpu_matmul_xq_row_stride(q8_class, k);
				xq_by_class[q8_class] = cpu_matmul_quantize_x_rows(self, &local_scratch[q8_class],
																   q8_class, xf, k, m, row_stride);
			}
			j->xq					= xq_by_class[q8_class];
			j->xq_row_stride		= cpu_matmul_xq_row_stride(q8_class, k);
			j->xq_row_stride_blocks = j->xq_row_stride / cpu_matmul_q8_block_size(q8_class);
		} else if (w_types[i] != GGML_TYPE_F32 && w_types[i] != GGML_TYPE_BF16) {
			for (int ii = 0; ii < n_matmuls; ii++)
				for (int row = 0; row < m; row++)
					cpu_matmul_one(cpu_ptr(w[ii]), w_types[ii], xf + ((size_t)row * k),
								   cpu_ptr(y[ii]) + ((size_t)row * n_list[ii]), n_list[ii], k,
								   &p->qscratch);
			return OK;
		}
		j->row_stride = cpu_matmul_w_row_stride(w_types[i], k);
	}

	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);
	int total_rows	= mj.row_offset[n_matmuls];

	if (!can_recurse || !p->pool ||
		(size_t)m * (size_t)total_rows < 2 * CPU_MATMUL_MIN_ROWS_PER_THREAD) {
		for (int i = 0; i < n_matmuls; i++)
			cpu_matmul_rows_worker(0, n_list[i], -1, &mj.jobs[i]);
		return OK;
	}

	tpool_parallel_for(p->pool, total_rows, CPU_MATMUL_MIN_ROWS_PER_THREAD, cpu_matmul_multi_worker,
					   &mj);
	return OK;
}

static status_code cpu_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							  buffer *y, int n, int k) {
	return cpu_matmul_batch(self, w, w_type, x, y, n, k, 1);
}
static status_code cpu_matmul_residual(backend *self, const buffer *w, uint32_t w_type,
									   const buffer *x, const buffer *residual, buffer *y, int n,
									   int k) {
	const void	*W	= cpu_ptr(w);
	const float *xf = cpu_ptr(x);
	float		*yf = cpu_ptr(y);
	const float *rf = cpu_ptr(residual);

	cpu_matmul_threaded_bias_residual(self, W, w_type, xf, yf, n, k, NULL, rf);
	return OK;
}

static status_code cpu_matmul_batch(backend *self, const buffer *w, uint32_t w_type,
									const buffer *x, buffer *y, int n, int k, int m) {
	cpu_priv	*p	= self->priv;
	const float *xf = cpu_ptr(x);
	float		*yf = cpu_ptr(y);
	const void	*W	= cpu_ptr(w);

	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);

	if (!can_recurse || !p->pool || !p->thread_scratch ||
		(size_t)m * (size_t)n < 2 * CPU_MATMUL_MIN_ROWS_PER_THREAD) {
		quant_scratch *qs = cpu_scratch_for_tid(p, cur_tid);
		for (int i = 0; i < m; i++)
			cpu_matmul_one(W, w_type, xf + ((size_t)i * k), yf + ((size_t)i * n), n, k, qs);
		return OK;
	}

	cpu_matmul_job *job = xmalloc(sizeof(*job));
	memset(job, 0, sizeof(*job));
	job->W		= W;
	job->w_type = w_type;
	job->xf		= xf;
	job->y		= yf;
	job->n		= n;
	job->k		= k;
	job->m		= m;
	cpu_matmul_job_prepare_kernel(job);

	int q8_class = job->kernel ? job->kernel->q8_class : 0;
	if (q8_class) {
		job->xq_row_stride = cpu_matmul_xq_row_stride(q8_class, k);
		job->xq_row_stride_blocks =
			job->xq_row_stride / cpu_matmul_q8_block_size(job->kernel->q8_class);
		job->xq =
			cpu_matmul_quantize_x_rows(self, &p->qscratch, q8_class, xf, k, m, job->xq_row_stride);
	} else if (w_type != GGML_TYPE_F32 && w_type != GGML_TYPE_BF16 && w_type != GGML_TYPE_F16) {
		for (int i = 0; i < m; i++)
			cpu_matmul_one(W, w_type, xf + ((size_t)i * k), yf + ((size_t)i * n), n, k,
						   &p->qscratch);
		free(job);
		return OK;
	}
	job->row_stride = cpu_matmul_w_row_stride(w_type, k);

	cpu_matmul_dispatch_rows(p->pool, w_type, n, job, cpu_matmul_rows_worker);
	free(job);
	return OK;
}

static status_code cpu_prequantize_x(backend *self, const buffer *x, int k, uint32_t q8_type,
									 buffer *x_q8_out) {
	cpu_priv *p = self->priv;

	int q8_class;
	switch (q8_type) {
	case GGML_TYPE_Q8_0:
		q8_class = 1;
		break;
	case GGML_TYPE_Q8_1:
		q8_class = 2;
		break;
	case GGML_TYPE_Q8_K:
		q8_class = 3;
		break;
	default:
		return ERR_UNSUPPORTED;
	}

	size_t		need	= cpu_matmul_xq_row_stride(q8_class, k);
	status_code grow_st = cpu_scratch_grow(&p->xq8_buf, &p->xq8_buf_cap, need);
	if (grow_st != OK)
		return grow_st;
	cpu_quantize_class(cpu_ptr(x), p->xq8_buf, k, q8_class);
	x_q8_out->handle   = p->xq8_buf;
	x_q8_out->size	   = need;
	x_q8_out->host_ptr = NULL;
	x_q8_out->owner	   = NULL;
	return OK;
}

static status_code cpu_matmul_qonly(backend *self, const buffer *w, uint32_t w_type,
									const buffer *x_q8, uint32_t q8_type, buffer *y, int n, int k,
									int m) {
	(void)self;
	const matmul_kernel *entry = matmul_kernel_lookup(w_type);
	if (!entry)
		return ERR_UNSUPPORTED;
	uint32_t want_q8 = cpu_matmul_q8_expected_type(entry->q8_class);
	if (!want_q8 || q8_type != want_q8)
		return ERR_INVALID_ARG;
	if (m <= 0)
		m = 1;
	size_t xq_row_stride_blocks = (size_t)(k / 32);
	int	   y_row_stride			= n;
	entry->qonly(cpu_ptr(w), x_q8->handle, xq_row_stride_blocks, cpu_ptr(y), y_row_stride, n, k, m);
	return OK;
}

__attribute__((weak)) status_code cpu_rope_batch(backend *self, buffer *vec, int n_heads,
												 int head_dim, int pos_start,
												 const float *rope_cos_base,
												 const float *rope_sin_base, int m);
__attribute__((weak)) status_code cpu_rope_qk_batch(backend *self, buffer *q, buffer *k,
													int n_heads, int n_kv_heads, int head_dim,
													int pos_start, const float *rope_cos_base,
													const float *rope_sin_base, int m);

static void cpu_rope_one(float *v, int n_heads, int head_dim, const float *rope_cos,
						 const float *rope_sin, const float *freq_factors, int neox) {
	int half = head_dim / 2;
	(void)freq_factors;
	if (neox) {
		rope_rotate_neox(v, n_heads, head_dim, head_dim, rope_cos, rope_sin);
		return;
	}
	for (int h = 0; h < n_heads; h++) {
		float *vh = v + ((size_t)h * head_dim);
		for (int j = 0; j < half; j++) {
			float c			= rope_cos[j];
			float s			= rope_sin[j];
			float v0		= vh[2 * j];
			float v1		= vh[(2 * j) + 1];
			vh[2 * j]		= (v0 * c) - (v1 * s);
			vh[(2 * j) + 1] = (v0 * s) + (v1 * c);
		}
	}
}

static status_code cpu_rope_one_factored(cpu_priv *p, float *v, int n_heads, int head_dim, int pos,
										 float theta, const float *freq_factors, int neox) {
	int half = head_dim / 2;

	double theta_scale;
	if (p->rope_cs.theta_cached == theta && p->rope_cs.head_dim_cached == head_dim) {
		theta_scale = p->rope_cs.theta_scale;
	} else {
		theta_scale				   = pow((double)theta, -2.0 / (double)head_dim);
		p->rope_cs.theta_scale	   = theta_scale;
		p->rope_cs.theta_cached	   = theta;
		p->rope_cs.head_dim_cached = head_dim;
	}

	int cache_valid =
		p->rope_cs.cs && p->rope_cs.pos_cached == pos && p->rope_cs.theta_cs_cached == theta &&
		p->rope_cs.head_dim_cs_cached == head_dim && p->rope_cs.freq_factors_cached == freq_factors;

	if ((size_t)p->rope_cs.cs_cap < (size_t)half * 2 * sizeof(float)) {
		status_code grow_st = cpu_scratch_grow_floats(&p->rope_cs.cs, &p->rope_cs.cs_cap, half * 2);
		if (grow_st != OK)
			return grow_st;
		cache_valid = 0;
	}

	float *cs = p->rope_cs.cs;
	if (!cache_valid) {
		double base_freq = 1.0;
		for (int j = 0; j < half; j++) {
			double ff;
			if (freq_factors) {
				float fv = freq_factors[j];
				ff		 = (fv >= 1e10f || fv == 0.0f) ? 0.0 : (double)fv;
			} else {
				ff = 1.0;
			}
			if (ff > 0.0) {
				double angle	= base_freq * ff * (double)pos;
				cs[2 * j]		= (float)cos(angle);
				cs[(2 * j) + 1] = (float)sin(angle);
			} else {
				cs[2 * j]		= 1.0f;
				cs[(2 * j) + 1] = 0.0f;
			}
			base_freq *= theta_scale;
		}
		p->rope_cs.pos_cached		   = pos;
		p->rope_cs.theta_cs_cached	   = theta;
		p->rope_cs.head_dim_cs_cached  = head_dim;
		p->rope_cs.freq_factors_cached = freq_factors;
	}

	for (int h = 0; h < n_heads; h++) {
		float *vh = v + ((size_t)h * head_dim);
		for (int j = 0; j < half; j++) {
			float c = cs[2 * j];
			float s = cs[(2 * j) + 1];
			if (neox) {
				float v0	 = vh[j];
				float v1	 = vh[j + half];
				vh[j]		 = (v0 * c) - (v1 * s);
				vh[j + half] = (v0 * s) + (v1 * c);
			} else {
				float v0		= vh[2 * j];
				float v1		= vh[(2 * j) + 1];
				vh[2 * j]		= (v0 * c) - (v1 * s);
				vh[(2 * j) + 1] = (v0 * s) + (v1 * c);
			}
		}
	}
	return OK;
}

__attribute__((weak)) status_code cpu_rope(backend *self, buffer *vec, int n_heads, int head_dim,
										   int pos, const float *rope_cos_base,
										   const float *rope_sin_base) {
	return cpu_rope_batch(self, vec, n_heads, head_dim, pos, rope_cos_base, rope_sin_base, 1);
}

__attribute__((weak)) status_code cpu_rope_qk(backend *self, buffer *q, buffer *k, int n_heads,
											  int n_kv_heads, int head_dim, int pos,
											  const float *rope_cos_base,
											  const float *rope_sin_base) {
	return cpu_rope_qk_batch(self, q, k, n_heads, n_kv_heads, head_dim, pos, rope_cos_base,
							 rope_sin_base, 1);
}

static void cpu_rope_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rope_batch_job *j	 = ctx;
	int					half = j->head_dim / 2;
	for (int row = begin; row < end; row++) {
		int			 pos = j->pos_start + row;
		const float *rc	 = j->rope_cos_base + ((size_t)pos * half);
		const float *rs	 = j->rope_sin_base + ((size_t)pos * half);
		cpu_rope_one(j->vec + ((size_t)row * j->n_heads * j->head_dim), j->n_heads, j->head_dim, rc,
					 rs, NULL, j->rope_neox);
	}
}

__attribute__((weak)) status_code cpu_rope_batch(backend *self, buffer *vec, int n_heads,
												 int head_dim, int pos_start,
												 const float *rope_cos_base,
												 const float *rope_sin_base, int m) {
	cpu_priv		  *p   = self->priv;
	cpu_rope_batch_job job = {.vec			 = (float *)cpu_ptr(vec),
							  .n_heads		 = n_heads,
							  .head_dim		 = head_dim,
							  .pos_start	 = pos_start,
							  .rope_cos_base = rope_cos_base,
							  .rope_sin_base = rope_sin_base,
							  .rope_neox	 = self->rope_neox};
	cpu_run_batch(p->pool, m, cpu_rope_batch_chunk, &job);
	return OK;
}

static void cpu_rope_qk_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rope_qk_batch_job *j	= ctx;
	int					   half = j->head_dim / 2;
	for (int row = begin; row < end; row++) {
		int			 pos = j->pos_start + row;
		const float *rc	 = j->rope_cos_base + ((size_t)pos * half);
		const float *rs	 = j->rope_sin_base + ((size_t)pos * half);
		cpu_rope_one(j->q + ((size_t)row * j->n_heads * j->head_dim), j->n_heads, j->head_dim, rc,
					 rs, NULL, j->rope_neox);
		cpu_rope_one(j->k + ((size_t)row * j->n_kv_heads * j->head_dim), j->n_kv_heads, j->head_dim,
					 rc, rs, NULL, j->rope_neox);
	}
}

__attribute__((weak)) status_code cpu_rope_qk_batch(backend *self, buffer *q, buffer *k,
													int n_heads, int n_kv_heads, int head_dim,
													int pos_start, const float *rope_cos_base,
													const float *rope_sin_base, int m) {
	cpu_priv			 *p	  = self->priv;
	cpu_rope_qk_batch_job job = {.q				= (float *)cpu_ptr(q),
								 .k				= (float *)cpu_ptr(k),
								 .n_heads		= n_heads,
								 .n_kv_heads	= n_kv_heads,
								 .head_dim		= head_dim,
								 .pos_start		= pos_start,
								 .rope_cos_base = rope_cos_base,
								 .rope_sin_base = rope_sin_base,
								 .rope_neox		= self->rope_neox};
	cpu_run_batch(p->pool, m, cpu_rope_qk_batch_chunk, &job);
	return OK;
}

static status_code cpu_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
								const float *rope_cos_base, const float *rope_sin_base,
								const float *freq_factors) {
	cpu_priv *p = self->priv;
	if (freq_factors) {
		status_code err = cpu_rope_one_factored(p, (float *)cpu_ptr(vec), n_heads, head_dim, pos,
												self->rope_theta, freq_factors, self->rope_neox);
		if (err)
			return err;
	} else {
		int			 half	  = head_dim / 2;
		const float *rope_cos = rope_cos_base + ((size_t)pos * half);
		const float *rope_sin = rope_sin_base + ((size_t)pos * half);
		cpu_rope_one((float *)cpu_ptr(vec), n_heads, head_dim, rope_cos, rope_sin, NULL,
					 self->rope_neox);
	}
	return OK;
}

__attribute__((weak)) float dot8(const float *restrict a, const float *restrict b, int head_dim) {
	float a0	  = 0;
	float a1	  = 0;
	float a2	  = 0;
	float a3	  = 0;
	int	  d		  = 0;
	int	  hd_main = head_dim - (head_dim % 8);
	for (; d < hd_main; d += 8) {
		a0 += (a[d] * b[d]) + (a[d + 4] * b[d + 4]);
		a1 += (a[d + 1] * b[d + 1]) + (a[d + 5] * b[d + 5]);
		a2 += (a[d + 2] * b[d + 2]) + (a[d + 6] * b[d + 6]);
		a3 += (a[d + 3] * b[d + 3]) + (a[d + 7] * b[d + 7]);
	}
	for (; d < head_dim; d++)
		a0 += a[d] * b[d];
	return (a0 + a1) + (a2 + a3);
}

__attribute__((weak)) float dot8_f16(const float *restrict a, const uint16_t *restrict b,
									 int head_dim) {
	float a0	  = 0;
	float a1	  = 0;
	float a2	  = 0;
	float a3	  = 0;
	int	  d		  = 0;
	int	  hd_main = head_dim - (head_dim % 8);
	for (; d < hd_main; d += 8) {
		float b0 = f16_to_f32(b[d]);
		float b1 = f16_to_f32(b[d + 1]);
		float b2 = f16_to_f32(b[d + 2]);
		float b3 = f16_to_f32(b[d + 3]);
		float b4 = f16_to_f32(b[d + 4]);
		float b5 = f16_to_f32(b[d + 5]);
		float b6 = f16_to_f32(b[d + 6]);
		float b7 = f16_to_f32(b[d + 7]);
		a0 += (a[d] * b0) + (a[d + 4] * b4);
		a1 += (a[d + 1] * b1) + (a[d + 5] * b5);
		a2 += (a[d + 2] * b2) + (a[d + 6] * b6);
		a3 += (a[d + 3] * b3) + (a[d + 7] * b7);
	}
	for (; d < head_dim; d++)
		a0 += a[d] * f16_to_f32(b[d]);
	return (a0 + a1) + (a2 + a3);
}

static inline float dot8_q8_0(const float *restrict a, const uint8_t *restrict block_ptr,
							  int head_dim) {
	int	  n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	float sum	   = 0.0f;
	for (int b = 0; b < n_blocks; b++) {
		const q8_0_block *blk = (const q8_0_block *)(block_ptr + ((size_t)b * KV_Q8_0_BLOCK_BYTES));
		float			  d	  = f16_to_f32(blk->d);
		int				  base = b * KV_Q8_0_BLOCK;
		int				  n	   = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;
		float partial = 0.0f;
		for (int j = 0; j < n; j++)
			partial += a[base + j] * (float)blk->qs[j];
		sum += partial * d;
	}
	return sum;
}

static inline void accum_v_q8_0(float *restrict out_h, const uint8_t *restrict block_ptr,
								float weight, int head_dim) {
	int n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	for (int b = 0; b < n_blocks; b++) {
		const q8_0_block *blk = (const q8_0_block *)(block_ptr + ((size_t)b * KV_Q8_0_BLOCK_BYTES));
		float			  d	  = f16_to_f32(blk->d) * weight;
		int				  base = b * KV_Q8_0_BLOCK;
		int				  n	   = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;
		for (int j = 0; j < n; j++)
			out_h[base + j] = fmaf((float)blk->qs[j], d, out_h[base + j]);
	}
}

static void cpu_attention_inner_q8_0(const uint8_t *k_slice, const uint8_t *v_slice,
									 size_t kv_stride_bytes, const float *qh, float *out_h,
									 int head_dim, int n_pos, float scale, int flash_attn,
									 float *scores) {
	if (flash_attn) {
		float M					= -INFINITY;
		float S					= 0.0f;
		float VKQ[HEAD_DIM_MAX] = {0};
		for (int t = 0; t < n_pos; t++) {
			const uint8_t *kt = k_slice + ((size_t)t * kv_stride_bytes);
			float		   ss = dot8_q8_0(qh, kt, head_dim) * scale;
			float		   ms = 1.0f;
			float		   vs = 1.0f;
			if (ss > M) {
				float Mold = M;
				M		   = ss;
				ms		   = expf(Mold - M);
				for (int d = 0; d < head_dim; d++)
					VKQ[d] *= ms;
			} else {
				vs = expf(ss - M);
			}
			const uint8_t *vt = v_slice + ((size_t)t * kv_stride_bytes);
			accum_v_q8_0(VKQ, vt, vs, head_dim);
			S = (S * ms) + vs;
		}
		float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
		for (int d = 0; d < head_dim; d++)
			out_h[d] = VKQ[d] * S_inv;
		return;
	}

	for (int t = 0; t < n_pos; t++) {
		const uint8_t *kt = k_slice + ((size_t)t * kv_stride_bytes);
		scores[t]		  = dot8_q8_0(qh, kt, head_dim) * scale;
	}
	softmax_masked(scores, n_pos);

	memset(out_h, 0, (size_t)head_dim * sizeof(float));
	for (int t = 0; t < n_pos; t++) {
		const uint8_t *vt = v_slice + ((size_t)t * kv_stride_bytes);
		accum_v_q8_0(out_h, vt, scores[t], head_dim);
	}
}

static void cpu_attention_inner(const uint16_t *k_slice, const uint16_t *v_slice, int kv_stride,
								const float *qh, float *out_h, int head_dim, int n_pos, float scale,
								int flash_attn, float *scores) {
	if (flash_attn) {
		float M					= -INFINITY;
		float S					= 0.0f;
		float VKQ[HEAD_DIM_MAX] = {0};
		for (int t = 0; t < n_pos; t++) {
			const uint16_t *kt = k_slice + ((size_t)t * kv_stride);
			float			ss = dot8_f16(qh, kt, head_dim) * scale;
			float			ms = 1.0f;
			float			vs = 1.0f;
			if (ss > M) {
				float Mold = M;
				M		   = ss;
				ms		   = expf(Mold - M);
				for (int d = 0; d < head_dim; d++)
					VKQ[d] *= ms;
			} else {
				vs = expf(ss - M);
			}
			const uint16_t *vt = v_slice + ((size_t)t * kv_stride);
			for (int d = 0; d < head_dim; d++)
				VKQ[d] = fmaf(f16_to_f32(vt[d]), vs, VKQ[d]);
			S = (S * ms) + vs;
		}
		float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
		for (int d = 0; d < head_dim; d++)
			out_h[d] = VKQ[d] * S_inv;
		return;
	}

	for (int t = 0; t < n_pos; t++) {
		const uint16_t *kt = k_slice + ((size_t)t * kv_stride);
		scores[t]		   = dot8_f16(qh, kt, head_dim) * scale;
	}
	softmax_masked(scores, n_pos);

	if (n_pos > 0) {
		float			sv0 = scores[0];
		const uint16_t *vt0 = v_slice;
		for (int d = 0; d < head_dim; d++)
			out_h[d] = sv0 * f16_to_f32(vt0[d]);
	}
	for (int t = 1; t < n_pos; t++) {
		float			sv = scores[t];
		const uint16_t *vt = v_slice + ((size_t)t * kv_stride);
		for (int d = 0; d < head_dim; d++)
			out_h[d] += sv * f16_to_f32(vt[d]);
	}
}

static void cpu_attn_head_chunk(int begin, int end, int tid, void *ctx) {
	cpu_attn_job *j = ctx;
	float		 *scores;
	if (tid == 0) {
		scores = j->p->scores;
	} else {
		cpu_thread_scratch *ts = &j->p->thread_scratch[tid];
		if (ts->scores_cap < j->n_pos) {
			free(ts->scores);
			ts->scores	   = xmalloc((size_t)j->n_pos * sizeof(float));
			ts->scores_cap = j->n_pos;
		}
		scores = ts->scores;
	}
	for (int h = begin; h < end; h++) {
		int			 kvh   = h / j->n_groups;
		const float *qh	   = j->qf + ((size_t)h * j->head_dim);
		float		*out_h = j->outf + ((size_t)h * j->head_dim);
		if (j->kv_quant == KV_QUANT_Q8_0) {
			const uint8_t *k_slice = (const uint8_t *)j->kl_base + ((size_t)kvh * j->kvh_stride);
			const uint8_t *v_slice = (const uint8_t *)j->vl_base + ((size_t)kvh * j->kvh_stride);
			cpu_attention_inner_q8_0(k_slice, v_slice, (size_t)j->hd_stride, qh, out_h, j->head_dim,
									 j->n_pos, j->scale, j->flash_attn, scores);
		} else {
			const uint16_t *k_slice = j->kl_base + ((size_t)kvh * j->kvh_stride);
			const uint16_t *v_slice = j->vl_base + ((size_t)kvh * j->kvh_stride);
			cpu_attention_inner(k_slice, v_slice, j->hd_stride, qh, out_h, j->head_dim, j->n_pos,
								j->scale, j->flash_attn, scores);
		}
	}
}

__attribute__((weak)) status_code cpu_attention_impl(backend *self, const buffer *q,
													 const buffer *k_cache, const buffer *v_cache,
													 buffer *out, int layer, int pos, int n_heads,
													 int n_kv_heads, int head_dim, int n_ctx,
													 int flash_attn, float scale,
													 int sliding_window, int n_kv_heads_active) {
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

	int			 n_groups = (n_heads + n_active - 1) / n_active;
	const float *qf		  = (const float *)cpu_ptr(q);
	float		*outf	  = (float *)cpu_ptr(out);
	(void)k_cache;
	(void)v_cache;

	int hd_stride_elems = head_dim;

	int attn_start = 0;
	int n_pos	   = pos + 1;
	if (sliding_window > 0 && n_pos > sliding_window) {
		attn_start = n_pos - sliding_window;
		n_pos	   = sliding_window;
	}

	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);

	if (p->kv_quant == KV_QUANT_Q8_0) {
		size_t n_blocks		= ((size_t)hd_stride_elems + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		size_t elem_stride	= n_blocks * KV_Q8_0_BLOCK_BYTES;
		size_t layer_stride = (size_t)n_kv_heads * n_ctx * elem_stride;
		size_t kvh_stride	= (size_t)n_ctx * elem_stride;
		size_t layer_base =
			p->kv_layer_off ? p->kv_layer_off[layer] : ((size_t)layer * layer_stride);
		const uint8_t *kl_base =
			(const uint8_t *)p->kv_k + layer_base + ((size_t)attn_start * elem_stride);
		const uint8_t *vl_base =
			(const uint8_t *)p->kv_v + layer_base + ((size_t)attn_start * elem_stride);

		if (can_recurse && p->pool && p->thread_scratch &&
			(size_t)n_heads * (size_t)n_pos >= 4096) {
			cpu_attn_job job = {.kl_base	= (const uint16_t *)kl_base,
								.vl_base	= (const uint16_t *)vl_base,
								.qf			= qf,
								.outf		= outf,
								.n_groups	= n_groups,
								.head_dim	= head_dim,
								.hd_stride	= (int)elem_stride,
								.n_pos		= n_pos,
								.flash_attn = flash_attn,
								.scale		= scale,
								.kvh_stride = kvh_stride,
								.p			= p,
								.kv_quant	= KV_QUANT_Q8_0};
			tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_head_chunk, &job);
			return OK;
		}

		float *scores;
		if (cur_tid >= 0 && p->thread_scratch && cur_tid < p->n_threads) {
			cpu_thread_scratch *ts = &p->thread_scratch[cur_tid];
			status_code grow_st	   = cpu_scratch_grow_floats(&ts->scores, &ts->scores_cap, n_pos);
			if (grow_st != OK)
				return grow_st;
			scores = ts->scores;
		} else {
			scores = p->scores;
		}
		for (int kvh = 0; kvh < n_active; kvh++) {
			const uint8_t *k_slice = kl_base + ((size_t)kvh * kvh_stride);
			const uint8_t *v_slice = vl_base + ((size_t)kvh * kvh_stride);
			for (int hg = 0; hg < n_groups; hg++) {
				int			 h	   = (kvh * n_groups) + hg;
				const float *qh	   = qf + ((size_t)h * head_dim);
				float		*out_h = outf + ((size_t)h * head_dim);
				cpu_attention_inner_q8_0(k_slice, v_slice, elem_stride, qh, out_h, head_dim, n_pos,
										 scale, flash_attn, scores);
			}
		}
		return OK;
	}

	int	   hd_stride	= hd_stride_elems;
	size_t layer_stride = (size_t)n_kv_heads * n_ctx * hd_stride;
	size_t kvh_stride	= (size_t)n_ctx * hd_stride;
	size_t layer_base	= p->kv_layer_off ? p->kv_layer_off[layer] / sizeof(uint16_t)
										  : ((size_t)layer * layer_stride);

	uint16_t *kl_base = p->kv_k + layer_base;
	uint16_t *vl_base = p->kv_v + layer_base;
	kl_base += (size_t)attn_start * hd_stride;
	vl_base += (size_t)attn_start * hd_stride;

	if (can_recurse && p->pool && p->thread_scratch && (size_t)n_heads * (size_t)n_pos >= 4096) {
		cpu_attn_job job = {.kl_base	= kl_base,
							.vl_base	= vl_base,
							.qf			= qf,
							.outf		= outf,
							.n_groups	= n_groups,
							.head_dim	= head_dim,
							.hd_stride	= hd_stride,
							.n_pos		= n_pos,
							.flash_attn = flash_attn,
							.scale		= scale,
							.kvh_stride = kvh_stride,
							.p			= p,
							.kv_quant	= KV_QUANT_F16};
		tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_head_chunk, &job);
		return OK;
	}

	float *scores;
	if (cur_tid >= 0 && p->thread_scratch && cur_tid < p->n_threads) {
		cpu_thread_scratch *ts		= &p->thread_scratch[cur_tid];
		status_code			grow_st = cpu_scratch_grow_floats(&ts->scores, &ts->scores_cap, n_pos);
		if (grow_st != OK)
			return grow_st;
		scores = ts->scores;
	} else {
		scores = p->scores;
	}
	for (int kvh = 0; kvh < n_active; kvh++) {
		uint16_t *k_slice = kl_base + ((size_t)kvh * kvh_stride);
		uint16_t *v_slice = vl_base + ((size_t)kvh * kvh_stride);

		for (int hg = 0; hg < n_groups; hg++) {
			int			 h	   = (kvh * n_groups) + hg;
			const float *qh	   = qf + ((size_t)h * head_dim);
			float		*out_h = outf + ((size_t)h * head_dim);
			cpu_attention_inner(k_slice, v_slice, hd_stride, qh, out_h, head_dim, n_pos, scale,
								flash_attn, scores);
		}
	}
	return OK;
}

__attribute__((weak)) status_code cpu_attention(backend *self, const buffer *q,
												const buffer *k_cache, const buffer *v_cache,
												buffer *out, int layer, int pos, int n_heads,
												int n_kv_heads, int head_dim, int n_ctx,
												int flash_attn, float scale,
												int n_kv_heads_active) {
	return cpu_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							  head_dim, n_ctx, flash_attn, scale, 0, n_kv_heads_active);
}

__attribute__((weak)) status_code cpu_attention_swa(backend *self, const buffer *q,
													const buffer *k_cache, const buffer *v_cache,
													buffer *out, int layer, int pos, int n_heads,
													int n_kv_heads, int head_dim, int n_ctx,
													int flash_attn, float scale, int sliding_window,
													int n_kv_heads_active) {
	return cpu_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							  head_dim, n_ctx, flash_attn, scale, sliding_window,
							  n_kv_heads_active);
}

static void cpu_attn_batch_chunk(int begin, int end, int tid, void *ctx) {
	cpu_attn_batch_job *j = ctx;
	float			   *scores;
	if (tid == 0) {
		scores = j->p->scores;
	} else {
		cpu_thread_scratch *ts	 = &j->p->thread_scratch[tid];
		int					need = j->pos_start + j->m;
		if (ts->scores_cap < need) {
			free(ts->scores);
			ts->scores	   = xmalloc((size_t)need * sizeof(float));
			ts->scores_cap = need;
		}
		scores = ts->scores;
	}

	for (int idx = begin; idx < end; idx++) {
		int dispatch_row = idx / j->n_heads;
		int h			 = idx % j->n_heads;

		int row = j->bitrev_perm ? j->bitrev_perm[dispatch_row] : dispatch_row;
		if (row >= j->m)
			continue;

		int pos		   = j->pos_start + row;
		int n_pos	   = pos + 1;
		int attn_start = 0;
		if (j->sliding_window > 0 && n_pos > j->sliding_window) {
			attn_start = n_pos - j->sliding_window;
			n_pos	   = j->sliding_window;
		}

		int			 kvh   = h / j->n_groups;
		const float *qh	   = j->qf + ((((size_t)row * j->n_heads) + h) * j->head_dim);
		float		*out_h = j->outf + ((((size_t)row * j->n_heads) + h) * j->head_dim);
		if (j->kv_quant == KV_QUANT_Q8_0) {
			const uint8_t *k_slice = (const uint8_t *)j->kl_base + ((size_t)kvh * j->kvh_stride) +
									 ((size_t)attn_start * (size_t)j->hd_stride);
			const uint8_t *v_slice = (const uint8_t *)j->vl_base + ((size_t)kvh * j->kvh_stride) +
									 ((size_t)attn_start * (size_t)j->hd_stride);
			cpu_attention_inner_q8_0(k_slice, v_slice, (size_t)j->hd_stride, qh, out_h, j->head_dim,
									 n_pos, j->scale, j->flash_attn, scores);
		} else {
			const uint16_t *k_slice =
				j->kl_base + ((size_t)kvh * j->kvh_stride) + ((size_t)attn_start * j->hd_stride);
			const uint16_t *v_slice =
				j->vl_base + ((size_t)kvh * j->kvh_stride) + ((size_t)attn_start * j->hd_stride);
			cpu_attention_inner(k_slice, v_slice, j->hd_stride, qh, out_h, j->head_dim, n_pos,
								j->scale, j->flash_attn, scores);
		}
	}
}

static status_code cpu_attention_batch_impl(backend *self, const buffer *q, const buffer *k_cache,
											const buffer *v_cache, buffer *out, int layer,
											int pos_start, int n_heads, int n_kv_heads,
											int head_dim, int n_ctx, int flash_attn, float scale,
											int sliding_window, int n_kv_heads_active, int m) {
	cpu_priv *p		   = self->priv;
	int		  n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	int		  n_groups = (n_heads + n_active - 1) / n_active;
	(void)k_cache;
	(void)v_cache;

	int			hd_stride;
	size_t		layer_stride;
	size_t		kvh_stride;
	const void *kl_base_raw;
	const void *vl_base_raw;
	if (p->kv_quant == KV_QUANT_Q8_0) {
		size_t n_blocks = ((size_t)head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
		hd_stride		= (int)(n_blocks * KV_Q8_0_BLOCK_BYTES);
		layer_stride	= (size_t)n_kv_heads * n_ctx * (size_t)hd_stride;
		kvh_stride		= (size_t)n_ctx * (size_t)hd_stride;
		size_t layer_base =
			p->kv_layer_off ? p->kv_layer_off[layer] : ((size_t)layer * layer_stride);
		kl_base_raw = (const uint8_t *)p->kv_k + layer_base;
		vl_base_raw = (const uint8_t *)p->kv_v + layer_base;
	} else {
		hd_stride		  = head_dim;
		layer_stride	  = (size_t)n_kv_heads * n_ctx * hd_stride;
		kvh_stride		  = (size_t)n_ctx * hd_stride;
		size_t layer_base = p->kv_layer_off ? p->kv_layer_off[layer] / sizeof(uint16_t)
											: ((size_t)layer * layer_stride);
		kl_base_raw		  = p->kv_k + layer_base;
		vl_base_raw		  = p->kv_v + layer_base;
	}

	int use_bitrev = (m >= ATTN_BITREV_MIN_M);
	int m_pow2	   = 1;
	while (m_pow2 < m)
		m_pow2 <<= 1;

	int	 bitrev_stack[ATTN_BITREV_STACK_MAX];
	int *bitrev_perm = NULL;
	if (use_bitrev) {
		if (m_pow2 <= ATTN_BITREV_STACK_MAX) {
			bitrev_perm = bitrev_stack;
		} else {
			bitrev_perm = xmalloc((size_t)m_pow2 * sizeof(int));
		}
		unsigned bits = 0;
		while ((1u << bits) < (unsigned)m_pow2)
			bits++;
		for (int r = 0; r < m_pow2; r++) {
			unsigned rev = 0, tmp = (unsigned)r;
			for (unsigned b = 0; b < bits; b++) {
				rev = (rev << 1) | (tmp & 1u);
				tmp >>= 1;
			}
			bitrev_perm[r] = (int)rev;
		}
	}

	cpu_attn_batch_job job = {.kl_base		  = kl_base_raw,
							  .vl_base		  = vl_base_raw,
							  .qf			  = (const float *)cpu_ptr(q),
							  .outf			  = (float *)cpu_ptr(out),
							  .n_groups		  = n_groups,
							  .head_dim		  = head_dim,
							  .hd_stride	  = hd_stride,
							  .n_heads		  = n_heads,
							  .pos_start	  = pos_start,
							  .m			  = m,
							  .sliding_window = sliding_window,
							  .flash_attn	  = flash_attn,
							  .scale		  = scale,
							  .kvh_stride	  = kvh_stride,
							  .p			  = p,
							  .bitrev_perm	  = bitrev_perm,
							  .kv_quant		  = p->kv_quant};

	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);
	if (can_recurse && p->pool && p->thread_scratch && n_heads * m >= 2) {
		int total = use_bitrev ? (n_heads * m_pow2) : (n_heads * m);
		tpool_parallel_for(p->pool, total, 1, cpu_attn_batch_chunk, &job);
	} else {
		cpu_attn_batch_chunk(0, n_heads * m, 0, &job);
	}

	if (bitrev_perm != bitrev_stack && bitrev_perm != NULL)
		free(bitrev_perm);

	return OK;
}

__attribute__((weak)) status_code cpu_attention_batch(backend *self, const buffer *q,
													  const buffer *k_cache, const buffer *v_cache,
													  buffer *out, int layer, int pos_start,
													  int n_heads, int n_kv_heads, int head_dim,
													  int n_ctx, int flash_attn, float scale,
													  int n_kv_heads_active, int m) {
	return cpu_attention_batch_impl(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
									n_kv_heads, head_dim, n_ctx, flash_attn, scale, 0,
									n_kv_heads_active, m);
}

__attribute__((weak)) status_code cpu_attention_swa_batch(
	backend *self, const buffer *q, const buffer *k_cache, const buffer *v_cache, buffer *out,
	int layer, int pos_start, int n_heads, int n_kv_heads, int head_dim, int n_ctx, int flash_attn,
	float scale, int sliding_window, int n_kv_heads_active, int m) {
	return cpu_attention_batch_impl(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
									n_kv_heads, head_dim, n_ctx, flash_attn, scale, sliding_window,
									n_kv_heads_active, m);
}

__attribute__((weak)) status_code cpu_add_inplace(backend *self, buffer *x, const buffer *y,
												  int n) {
	return cpu_add_batch(self, x, y, n, 1);
}

__attribute__((weak)) status_code cpu_ffn_activate(backend *self, const buffer *gate,
												   const buffer *up, buffer *out, int n) {
	return cpu_ffn_activate_batch(self, gate, up, out, n, 0, 1);
}

__attribute__((weak)) status_code cpu_ffn_activate_ex(backend *self, const buffer *gate,
													  const buffer *up, buffer *out, int n,
													  int activation) {
	return cpu_ffn_activate_batch(self, gate, up, out, n, activation, 1);
}

static void cpu_rmsnorm_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_rmsnorm_batch_job *j = ctx;
	for (int row = begin; row < end; row++)
		rmsnorm(j->x + ((size_t)row * j->n), j->w, j->y + ((size_t)row * j->n), j->n, j->eps);
}

static status_code cpu_rmsnorm_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
									 int n, float eps, int m) {
	cpu_priv			 *p	  = self->priv;
	cpu_rmsnorm_batch_job job = {
		.x = cpu_ptr(x), .w = cpu_ptr(w), .y = cpu_ptr(y), .n = n, .eps = eps};
	cpu_run_batch(p->pool, m, cpu_rmsnorm_batch_chunk, &job);
	return OK;
}

static void cpu_add_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_add_batch_job *j = ctx;
	for (int row = begin; row < end; row++) {
		float		*xr = j->x + ((size_t)row * j->n);
		const float *yr = j->y + ((size_t)row * j->n);
		for (int i = 0; i < j->n; i++)
			xr[i] += yr[i];
	}
}

__attribute__((weak)) status_code cpu_add_batch(backend *self, buffer *x, const buffer *y, int n,
												int m) {
	cpu_priv		 *p	  = self->priv;
	cpu_add_batch_job job = {.x = cpu_ptr(x), .y = cpu_ptr(y), .n = n};
	cpu_run_batch(p->pool, m, cpu_add_batch_chunk, &job);
	return OK;
}

static void cpu_ffn_act_batch_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_act_batch_job *j = ctx;
	for (int row = begin; row < end; row++) {
		const float *g = j->g + ((size_t)row * j->n);
		const float *u = j->u + ((size_t)row * j->n);
		float		*o = j->o + ((size_t)row * j->n);
		if (j->activation == 1) {
			for (int i = 0; i < j->n; i++)
				o[i] = gelu_tanh(g[i]) * u[i];
		} else {
			for (int i = 0; i < j->n; i++) {
				float gv = g[i];
				o[i]	 = gv / (1.0f + expf(-gv)) * u[i];
			}
		}
	}
}

__attribute__((weak)) status_code cpu_ffn_activate_batch(backend *self, const buffer *gate,
														 const buffer *up, buffer *out, int n,
														 int activation, int m) {
	cpu_priv			 *p	  = self->priv;
	cpu_ffn_act_batch_job job = {
		.g = cpu_ptr(gate), .u = cpu_ptr(up), .o = cpu_ptr(out), .n = n, .activation = activation};
	cpu_run_batch(p->pool, m, cpu_ffn_act_batch_chunk, &job);
	return OK;
}

static status_code cpu_rmsnorm_per_head(backend *self, const buffer *x, const buffer *w, buffer *y,
										int n_heads, int head_dim, float eps) {
	(void)self;
	const float *xf = cpu_ptr(x);
	const float *wf = cpu_ptr(w);
	float		*yf = cpu_ptr(y);
	rmsnorm_per_head(xf, wf, yf, n_heads, head_dim, eps);
	return OK;
}

static status_code cpu_rmsnorm_noweight(backend *self, const buffer *x, buffer *y, int n,
										float eps) {
	(void)self;
	const float *xf = cpu_ptr(x);
	float		*yf = cpu_ptr(y);
	rmsnorm_noweight(xf, yf, n, eps);
	return OK;
}

static status_code cpu_rmsnorm_noweight_per_head(backend *self, const buffer *x, buffer *y,
												 int n_heads, int head_dim, float eps) {
	(void)self;
	const float *xf = cpu_ptr(x);
	float		*yf = cpu_ptr(y);
	for (int h = 0; h < n_heads; h++) {
		rmsnorm_noweight(xf + ((size_t)h * head_dim), yf + ((size_t)h * head_dim), head_dim, eps);
	}
	return OK;
}

__attribute__((weak)) status_code cpu_argmax(backend *self, const buffer *logits, int n,
											 int32_t *out_idx) {
	(void)self;
	const float *lp	   = cpu_ptr(logits);
	int			 best  = 0;
	float		 bestv = lp[0];
	for (int i = 1; i < n; i++) {
		if (lp[i] > bestv) {
			bestv = lp[i];
			best  = i;
		}
	}
	*out_idx = best;
	return OK;
}

status_code cpu_kv_alloc_mla(backend *self, int n_layers, int n_ctx, int kv_lora, int qk_rope,
							 buffer *kv_out) {
	cpu_priv *p			= self->priv;
	size_t	  per_layer = (size_t)(kv_lora + qk_rope) * (size_t)n_ctx * sizeof(float);
	size_t	  total		= per_layer * (size_t)n_layers;
	kv_out->handle		= xmalloc_aligned(total, 64);
	kv_out->size		= total;
	kv_out->host_ptr	= NULL;
	kv_out->owner		= self;
	madvise_hugepage(kv_out->handle, total);
	prefault(kv_out->handle, total);

	if (p) {
		size_t		krot_need = (size_t)n_ctx * (size_t)qk_rope * sizeof(float);
		status_code grow_st =
			cpu_scratch_grow((void **)&p->mla_krot.buf, &p->mla_krot.cap, krot_need);
		if (grow_st != OK)
			return grow_st;
	}

	return OK;
}

status_code cpu_kv_put_mla(backend *self, buffer *kv_cache, int layer, int pos,
						   const buffer *kv_a_in, const buffer *kv_a_norm_w, int kv_lora,
						   int qk_rope, int n_ctx, float eps) {
	(void)self;
	int			 total_dim	  = kv_lora + qk_rope;
	const float *kv_a		  = (const float *)cpu_ptr(kv_a_in);
	const float *norm_w		  = (const float *)cpu_ptr(kv_a_norm_w);
	float		*cache_base	  = (float *)cpu_ptr(kv_cache);
	size_t		 layer_stride = (size_t)total_dim * (size_t)n_ctx;
	float		*slot = cache_base + ((size_t)layer * layer_stride) + ((size_t)pos * total_dim);
	float		 ss	  = 0.0f;
	for (int i = 0; i < kv_lora; i++)
		ss += kv_a[i] * kv_a[i];
	ss = 1.0f / sqrtf((ss / (float)kv_lora) + eps);
	for (int i = 0; i < kv_lora; i++)
		slot[i] = kv_a[i] * ss * norm_w[i];
	for (int i = 0; i < qk_rope; i++)
		slot[kv_lora + i] = kv_a[kv_lora + i];
	return OK;
}

static void cpu_attn_mla_head_compute(const cpu_attn_mla_job *j, int h) {
	const int	half_rope = j->qk_rope / 2;
	const int	n_pos	  = j->pos + 1;
	const float scale	  = j->scale;

	float q_h[HEAD_DIM_MAX];
	float q_absorbed[HEAD_DIM_MAX];
	float VKQ_latent[HEAD_DIM_MAX];

	const float *q_src = j->qf + ((size_t)h * j->qk_head);
	for (int d = 0; d < j->qk_head; d++)
		q_h[d] = q_src[d];
	const float *q_rope_cos = j->rope_cos_base + ((size_t)j->pos * half_rope);
	const float *q_rope_sin = j->rope_sin_base + ((size_t)j->pos * half_rope);
	for (int k = 0; k < half_rope; k++) {
		float c						  = q_rope_cos[k];
		float s						  = q_rope_sin[k];
		float a						  = q_h[j->qk_nope + (2 * k)];
		float b						  = q_h[j->qk_nope + (2 * k) + 1];
		q_h[j->qk_nope + (2 * k)]	  = (a * c) - (b * s);
		q_h[j->qk_nope + (2 * k) + 1] = (a * s) + (b * c);
	}

	const float *k_b_h = j->k_b + ((size_t)h * j->qk_nope * j->kv_lora);
	const float *v_b_h = j->v_b + ((size_t)h * j->kv_lora * j->v_head);

	for (int i = 0; i < j->kv_lora; i++) {
		float		 acc = 0.0f;
		const float *row = k_b_h + ((size_t)i * j->qk_nope);
		for (int d = 0; d < j->qk_nope; d++)
			acc += row[d] * q_h[d];
		q_absorbed[i] = acc;
	}
	const float *q_rope_part = q_h + j->qk_nope;

	float M = -INFINITY;
	float S = 0.0f;
	memset(VKQ_latent, 0, (size_t)j->kv_lora * sizeof(float));
	for (int t = 0; t < n_pos; t++) {
		const float *slot	  = j->layer_c + ((size_t)t * j->total_dim);
		const float *latent	  = slot;
		const float *k_pe_rot = j->k_pe_rot_all + ((size_t)t * j->qk_rope);

		float score = 0.0f;
		for (int i = 0; i < j->kv_lora; i++)
			score += latent[i] * q_absorbed[i];
		for (int d = 0; d < j->qk_rope; d++)
			score += q_rope_part[d] * k_pe_rot[d];
		score *= scale;

		float ms = 1.0f;
		float vs = 1.0f;
		if (score > M) {
			float Mold = M;
			M		   = score;
			ms		   = expf(Mold - M);
			for (int i = 0; i < j->kv_lora; i++)
				VKQ_latent[i] *= ms;
		} else {
			vs = expf(score - M);
		}
		for (int i = 0; i < j->kv_lora; i++)
			VKQ_latent[i] = fmaf(latent[i], vs, VKQ_latent[i]);
		S = (S * ms) + vs;
	}
	float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
	float VKQ[HEAD_DIM_MAX];
	for (int d = 0; d < j->v_head; d++) {
		float		 acc = 0.0f;
		const float *row = v_b_h + ((size_t)d * j->kv_lora);
		for (int i = 0; i < j->kv_lora; i++)
			acc += VKQ_latent[i] * row[i];
		VKQ[d] = acc * S_inv;
	}
	for (int d = 0; d < j->v_head; d++)
		j->outf[((size_t)h * j->v_head) + d] = VKQ[d];
}

static void cpu_attn_mla_head_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_attn_mla_job *j = ctx;
	for (int h = begin; h < end; h++)
		cpu_attn_mla_head_compute(j, h);
}

__attribute__((weak)) status_code cpu_attention_mla(backend *self, const buffer *q,
													const buffer *kv_cache, const buffer *k_b_w,
													const buffer *v_b_w, buffer *out, int layer,
													int pos, int n_heads, int qk_head, int qk_rope,
													int qk_nope, int v_head, int kv_lora, int n_ctx,
													const float *rope_cos_base,
													const float *rope_sin_base, float scale) {
	cpu_priv	*p			  = self->priv;
	int			 total_dim	  = kv_lora + qk_rope;
	const float *qf			  = (const float *)cpu_ptr(q);
	float		*outf		  = (float *)cpu_ptr(out);
	const float *cache		  = (const float *)cpu_ptr(kv_cache);
	const float *k_b		  = (const float *)cpu_ptr(k_b_w);
	const float *v_b		  = (const float *)cpu_ptr(v_b_w);
	size_t		 layer_stride = (size_t)total_dim * (size_t)n_ctx;
	const float *layer_c	  = cache + ((size_t)layer * layer_stride);

	int n_pos	  = pos + 1;
	int half_rope = qk_rope / 2;

	size_t		krot_need = (size_t)n_pos * qk_rope * sizeof(float);
	status_code grow_st = cpu_scratch_grow((void **)&p->mla_krot.buf, &p->mla_krot.cap, krot_need);
	if (grow_st != OK)
		return grow_st;
	float *k_pe_rot_all = p->mla_krot.buf;

	int cached_n_pos = p->mla_krot.n_pos_cached;
	int can_incr =
		(cached_n_pos == n_pos - 1) && (p->mla_krot.layer_cached == layer) &&
		(p->mla_krot.layer_c_cached == layer_c) && (p->mla_krot.total_dim_cached == total_dim) &&
		(p->mla_krot.kv_lora_cached == kv_lora) && (p->mla_krot.qk_rope_cached == qk_rope) &&
		(p->mla_krot.half_rope_cached == half_rope);

	if (can_incr && n_pos >= 1) {
		int			 t			= n_pos - 1;
		const float *slot		= layer_c + ((size_t)t * total_dim);
		const float *k_pe		= slot + kv_lora;
		const float *t_rope_cos = rope_cos_base + ((size_t)t * half_rope);
		const float *t_rope_sin = rope_sin_base + ((size_t)t * half_rope);
		float		*dst		= k_pe_rot_all + ((size_t)t * qk_rope);
		for (int j = 0; j < half_rope; j++) {
			float c			 = t_rope_cos[j];
			float s			 = t_rope_sin[j];
			float a			 = k_pe[2 * j];
			float b			 = k_pe[(2 * j) + 1];
			dst[2 * j]		 = (a * c) - (b * s);
			dst[(2 * j) + 1] = (a * s) + (b * c);
		}
		p->mla_krot.n_pos_cached = n_pos;
	} else {
		for (int t = 0; t < n_pos; t++) {
			const float *slot		= layer_c + ((size_t)t * total_dim);
			const float *k_pe		= slot + kv_lora;
			const float *t_rope_cos = rope_cos_base + ((size_t)t * half_rope);
			const float *t_rope_sin = rope_sin_base + ((size_t)t * half_rope);
			float		*dst		= k_pe_rot_all + ((size_t)t * qk_rope);
			for (int j = 0; j < half_rope; j++) {
				float c			 = t_rope_cos[j];
				float s			 = t_rope_sin[j];
				float a			 = k_pe[2 * j];
				float b			 = k_pe[(2 * j) + 1];
				dst[2 * j]		 = (a * c) - (b * s);
				dst[(2 * j) + 1] = (a * s) + (b * c);
			}
		}
		p->mla_krot.layer_c_cached	 = layer_c;
		p->mla_krot.layer_cached	 = layer;
		p->mla_krot.total_dim_cached = total_dim;
		p->mla_krot.kv_lora_cached	 = kv_lora;
		p->mla_krot.qk_rope_cached	 = qk_rope;
		p->mla_krot.half_rope_cached = half_rope;
		p->mla_krot.n_pos_cached	 = n_pos;
	}

	for (int i = 0; i < n_heads * v_head; i++)
		outf[i] = 0.0f;

	cpu_attn_mla_job job = {
		.qf			   = qf,
		.outf		   = outf,
		.k_b		   = k_b,
		.v_b		   = v_b,
		.layer_c	   = layer_c,
		.k_pe_rot_all  = k_pe_rot_all,
		.rope_cos_base = rope_cos_base,
		.rope_sin_base = rope_sin_base,
		.n_heads	   = n_heads,
		.qk_head	   = qk_head,
		.qk_rope	   = qk_rope,
		.qk_nope	   = qk_nope,
		.v_head		   = v_head,
		.kv_lora	   = kv_lora,
		.total_dim	   = total_dim,
		.pos		   = pos,
		.scale		   = scale,
	};

	int cur_tid		= tpool_current_tid();
	int can_recurse = (cur_tid < 0);

	if (can_recurse && p->pool && p->thread_scratch && n_heads >= 2) {
		tpool_parallel_for(p->pool, n_heads, 1, cpu_attn_mla_head_chunk, &job);
		return OK;
	}

	for (int h = 0; h < n_heads; h++)
		cpu_attn_mla_head_compute(&job, h);

	return OK;
}

static tpool *cpu_get_pool(backend *self) {
	cpu_priv *p = self->priv;
	return p ? p->pool : NULL;
}

static status_code cpu_matmul_thread_local(backend *self, const void *W, uint32_t w_type,
										   const float *x, float *y, int n, int k, int tid) {
	cpu_priv *p = self->priv;
	if (!p)
		return ERR_INVALID_ARG;
	quant_scratch *qs = cpu_scratch_for_tid(p, tid);
	cpu_matmul_one(W, w_type, x, y, n, k, qs);
	return OK;
}

static backend		 *g_host_backend = NULL;
static pthread_once_t g_host_once	 = PTHREAD_ONCE_INIT;

static _Atomic(backend *) g_host_override = NULL;

void backend_host_use(backend *b) {
	if (!b || !backend_has_cap(b, BCAP_IS_HOST))
		return;
	atomic_store(&g_host_override, b);
}

void backend_destroyed(backend *b) {
	if (b && atomic_load(&g_host_override) == b)
		atomic_store(&g_host_override, NULL);
}

static void host_backend_init(void) {
	if (backend_create("cpu", 0, &g_host_backend) != OK) {
		ERROR("could not create cpu fallback backend");
		abort();
	}
	if (atomic_load(&g_host_override) == NULL)
		atomic_store(&g_host_override, g_host_backend);
}

backend *backend_host(void) {
	backend *o = atomic_load(&g_host_override);
	if (o)
		return o;
	pthread_once(&g_host_once, host_backend_init);
	return g_host_backend;
}

backend *backend_weight_home(backend *b, weight_class wc) {
	int native;
	switch (wc) {
	case WCLASS_MATMUL:
		native = (b->matmul != NULL);
		break;
	case WCLASS_NORM:
		native = (b->rmsnorm != NULL);
		break;
	case WCLASS_EMBEDDING:
		native = (b->embd_lookup != NULL);
		break;
	case WCLASS_MISC:
	default:
		native = 1;
		break;
	}
	return native ? b : backend_host();
}

static status_code cpu_rmsnorm_add(backend *self, const buffer *x, const buffer *w,
								   const buffer *residual, buffer *y, int n, float eps) {
	(void)self;
	rmsnorm(cpu_ptr(x), cpu_ptr(w), cpu_ptr(y), n, eps);
	const float *rf = cpu_ptr(residual);
	float		*yf = cpu_ptr(y);
	for (int i = 0; i < n; i++)
		yf[i] += rf[i];
	return OK;
}

static status_code cpu_scale_inplace(backend *self, buffer *x, float scale, int n) {
	(void)self;
	float *xf = cpu_ptr(x);
	for (int i = 0; i < n; i++)
		xf[i] *= scale;
	return OK;
}

static status_code cpu_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
								   float combine_scale) {
	(void)self;
	float		*pf = cpu_ptr(ple);
	const float *pr = cpu_ptr(proj);
	for (int i = 0; i < n; i++)
		pf[i] = (pf[i] + pr[i]) * combine_scale;
	return OK;
}

__attribute__((weak)) void cpu_ffn_down_act_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	cpu_ffn_down_act_args *a = ctx;
	const float *restrict g	 = a->g;
	const float *restrict u	 = a->u;
	float *restrict o		 = a->o;
	if (a->activation == 1) {
		for (int i = begin; i < end; i++)
			o[i] = gelu_tanh(g[i]) * u[i];
	} else {
		for (int i = begin; i < end; i++) {
			float gv = g[i];
			o[i]	 = gv / (1.0f + expf(-gv)) * u[i];
		}
	}
}

static status_code cpu_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type,
									   const buffer *gate, const buffer *up, buffer *y, int n,
									   int k, int activation) {
	cpu_priv	*p		= self->priv;
	const void	*W		= cpu_ptr(w);
	const float *gate_f = cpu_ptr(gate);
	const float *up_f	= cpu_ptr(up);
	float		*yf		= cpu_ptr(y);

	status_code grow_st = cpu_scratch_grow_aligned((void **)&p->residual_tmp, &p->residual_tmp_cap,
												   (size_t)k * sizeof(float), 64);
	if (grow_st != OK)
		return grow_st;
	float *act = p->residual_tmp;

	cpu_ffn_down_act_args a		  = {.g = gate_f, .u = up_f, .o = act, .activation = activation};
	int					  cur_tid = tpool_current_tid();
	if (p->pool && cur_tid < 0 && k >= 2 * CPU_ELEMWISE_MIN_PER_THREAD) {
		tpool_parallel_for(p->pool, k, CPU_ELEMWISE_MIN_PER_THREAD, cpu_ffn_down_act_chunk, &a);
	} else {
		cpu_ffn_down_act_chunk(0, k, cur_tid, &a);
	}

	cpu_matmul_threaded_bias_residual(self, W, w_type, act, yf, n, k, NULL, NULL);
	return OK;
}

static void cpu_kv_free(backend *self, buffer *k, buffer *v) {
	cpu_buffer_free(self, k);
	cpu_buffer_free(self, v);
}

__attribute__((weak)) int32_t cpu_argmax_f32(const float *logits, int vocab) {
	if (vocab <= 0)
		return 0;
	int32_t best  = 0;
	float	bestv = logits[0];
	for (int i = 1; i < vocab; i++)
		if (logits[i] > bestv) {
			bestv = logits[i];
			best  = i;
		}
	return best;
}

static status_code cpu_ctor(backend *out) {
	memset(out, 0, sizeof(*out));
	out->name	  = "cpu";
	out->priority = 0;
	out->caps  = BCAP_IS_HOST | BCAP_MULTI_MATMUL | BCAP_ROPE_QK_FUSED | BCAP_MATMUL_RESIDUAL |
				 BCAP_MATMUL_QONLY | BCAP_RMSNORM_ADD | BCAP_MATMUL_FFN_DOWN | BCAP_KV_QUANT_Q8_0;
	out->probe = cpu_probe;
	out->init  = cpu_init;
	out->free  = cpu_free;
	out->buffer_alloc_weight	   = cpu_buffer_alloc_weight;
	out->buffer_alloc_scratch	   = cpu_buffer_alloc_scratch;
	out->buffer_free			   = cpu_buffer_free;
	out->copy_buffer			   = cpu_copy_buffer;
	out->buffer_read_f32		   = cpu_buffer_read_f32;
	out->buffer_write_f32		   = cpu_buffer_write_f32;
	out->kv_alloc				   = cpu_kv_alloc;
	out->kv_put					   = cpu_kv_put;
	out->embd_lookup			   = cpu_embd_lookup;
	out->rmsnorm				   = cpu_rmsnorm;
	out->matmul					   = cpu_matmul;
	out->matmul_residual		   = cpu_matmul_residual;
	out->matmul_batch			   = cpu_matmul_batch;
	out->matmul_multi			   = cpu_matmul_multi;
	out->matmul_multi_batch		   = cpu_matmul_multi_batch;
	out->matmul_qonly			   = cpu_matmul_qonly;
	out->prequantize_x			   = cpu_prequantize_x;
	out->rope					   = cpu_rope;
	out->rope_qk				   = cpu_rope_qk;
	out->rope_ext				   = cpu_rope_ext;
	out->attention				   = cpu_attention;
	out->attention_swa			   = cpu_attention_swa;
	out->add_inplace			   = cpu_add_inplace;
	out->ffn_activate			   = cpu_ffn_activate;
	out->ffn_activate_ex		   = cpu_ffn_activate_ex;
	out->rmsnorm_per_head		   = cpu_rmsnorm_per_head;
	out->rmsnorm_noweight		   = cpu_rmsnorm_noweight;
	out->rmsnorm_noweight_per_head = cpu_rmsnorm_noweight_per_head;
	out->argmax					   = cpu_argmax;
	out->synchronize			   = cpu_synchronize;
	out->attention_mla			   = cpu_attention_mla;
	out->kv_alloc_mla			   = cpu_kv_alloc_mla;
	out->kv_put_mla				   = cpu_kv_put_mla;
	out->get_pool				   = cpu_get_pool;
	out->matmul_thread_local	   = cpu_matmul_thread_local;
	out->mem_available			   = cpu_mem_available;
	out->mem_total				   = cpu_mem_total;
	out->rmsnorm_batch			   = cpu_rmsnorm_batch;
	out->add_batch				   = cpu_add_batch;
	out->ffn_activate_batch		   = cpu_ffn_activate_batch;
	out->rope_batch				   = cpu_rope_batch;
	out->rope_qk_batch			   = cpu_rope_qk_batch;
	out->attention_batch		   = cpu_attention_batch;
	out->attention_swa_batch	   = cpu_attention_swa_batch;
	out->rmsnorm_add			   = cpu_rmsnorm_add;
	out->scale_inplace			   = cpu_scale_inplace;
	out->ple_combine			   = cpu_ple_combine;
	out->matmul_ffn_down		   = cpu_matmul_ffn_down;
	out->kv_free				   = cpu_kv_free;
	return OK;
}

BACKEND_REGISTER("cpu", cpu_ctor)