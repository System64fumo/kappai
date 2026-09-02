#ifndef BACKEND_H
#define BACKEND_H

#include "common.h"
#include "gguf.h"
#include "threadpool.h"

#define BACKEND_MAX 8

typedef struct backend backend;

typedef status_code (*backend_ctor_fn)(backend *out);

typedef enum {
	BCAP_NONE				  = 0,
	BCAP_IS_HOST			  = 1 << 0,
	BCAP_MULTI_MATMUL		  = 1 << 1,
	BCAP_MATMUL_RESIDUAL	  = 1 << 2,
	BCAP_ROPE_QK_FUSED		  = 1 << 3,
	BCAP_RMSNORM_ADD		  = 1 << 4,
	BCAP_MATMUL_FFN_DOWN	  = 1 << 5,
	BCAP_MATMUL_QONLY		  = 1 << 6,
	BCAP_HOST_VISIBLE_BUFFERS = 1 << 7,
	BCAP_KV_QUANT_Q8_0		  = 1 << 8,
	BCAP_MOE_EXPERT_GPU		  = 1 << 9,
} backend_cap;

typedef struct {
	void	   *handle;
	size_t		size;
	const void *host_ptr;
	size_t		offset;
	backend	   *owner;
} buffer;

typedef struct {
	const buffer *gate_w;
	const buffer *up_w;
	const buffer *down_w;
	uint32_t	  gate_type;
	uint32_t	  up_type;
	uint32_t	  down_type;
	int			  gate_up_fused;
	int			  use_gelu;
	float		  gate_scale;
	float		  up_scale;
	float		  down_scale;
	float		  weight;
} moe_gpu_expert;
static inline buffer buffer_slice(const buffer *parent, size_t byte_off, size_t byte_len) {
	buffer s   = *parent;
	s.host_ptr = parent->host_ptr ? (const char *)parent->host_ptr + byte_off : NULL;
	s.offset   = parent->offset + byte_off;
	s.size	   = byte_len;
	return s;
}

typedef enum {
	WCLASS_MATMUL,
	WCLASS_NORM,
	WCLASS_EMBEDDING,
	WCLASS_MISC,
} weight_class;

typedef struct {
	uint32_t	type;
	int			n_dims;
	uint64_t	dims[4];
	const void *host_data;
	const char *name;
} tensor_desc;

typedef enum {
	KV_QUANT_F16 = 0,
	KV_QUANT_Q8_0,
} kv_quant_type;

#define KV_Q8_0_BLOCK 32
#define KV_Q8_0_BLOCK_BYTES 34

typedef struct {
	int			  n_layers, n_kv_layers, n_kv_heads, head_dim, n_ctx;
	kv_quant_type kv_quant;
	const int	 *layer_head_dim;
	const int	 *layer_n_kv_heads;
	const int	 *layer_pos_cap;
} kv_desc;

typedef struct {
	char	 name[32];
	int		 priority;
	int		 available;
	uint64_t caps;
} backend_info;

struct backend {
	const char *name;
	int			priority;
	uint64_t	caps;
	status_code (*probe)(void);
	status_code (*init)(backend *self, int device_index);
	void (*free)(backend *self);
	status_code (*buffer_alloc_weight)(backend *self, const tensor_desc *desc, buffer *out);
	status_code (*buffer_alloc_scratch)(backend *self, size_t size, buffer *out);
	void (*buffer_free)(backend *self, buffer *buf);
	status_code (*kv_alloc)(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out);
	void (*kv_free)(backend *self, buffer *k, buffer *v);
	status_code (*kv_put)(backend *self, buffer *k, buffer *v, int layer, int pos,
						  const buffer *k_in, const buffer *v_in, int n_kv_heads, int head_dim,
						  int n_ctx, int n_kv_heads_active);
	status_code (*kv_put_batch)(backend *self, buffer *k, buffer *v, int layer, int pos_start,
								const buffer *k_in, const buffer *v_in, int in_row_stride,
								int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
								int m);
	status_code (*embd_lookup)(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
							   int token, int dim, buffer *x_out);
	status_code (*rmsnorm)(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
						   float eps);
	status_code (*matmul)(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
						  buffer *y, int n, int k);
	int (*matmul_type_native)(backend *self, uint32_t w_type);
	status_code (*matmul_residual)(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								   const buffer *residual, buffer *y, int n, int k);
	status_code (*matmul_multi)(backend *self, const buffer **w, const uint32_t *w_types,
								const buffer *x, buffer **y, const int *n_list, int k,
								int n_matmuls);

	status_code (*matmul_qonly)(backend *self, const buffer *w, uint32_t w_type, const buffer *x_q8,
								uint32_t q8_type, buffer *y, int n, int k, int m);
	status_code (*prequantize_x)(backend *self, const buffer *x, int k, uint32_t q8_type,
								 buffer *x_q8_out);
	status_code (*rope)(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
						const float *rope_cos_base, const float *rope_sin_base);
	status_code (*rope_qk)(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
						   int head_dim, int pos, const float *rope_cos_base,
						   const float *rope_sin_base);
	status_code (*rope_ext)(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
							const float *rope_cos_base, const float *rope_sin_base,
							const float *freq_factors);
	status_code (*rope_ext_batch)(backend *self, buffer *vec, int n_heads, int head_dim,
								  int pos_start, const float *rope_cos_base,
								  const float *rope_sin_base, const float *freq_factors, int m);
	status_code (*attention)(backend *self, const buffer *q, const buffer *k_cache,
							 const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
							 int n_kv_heads, int head_dim, int n_ctx, int flash_attn, float scale,
							 int n_kv_heads_active);
	status_code (*attention_swa)(backend *self, const buffer *q, const buffer *k_cache,
								 const buffer *v_cache, buffer *out, int layer, int pos,
								 int n_heads, int n_kv_heads, int head_dim, int n_ctx,
								 int flash_attn, float scale, int sliding_window,
								 int n_kv_heads_active);
	status_code (*add_inplace)(backend *self, buffer *x, const buffer *y, int n);
	status_code (*scale_inplace)(backend *self, buffer *x, float scale, int n);
	status_code (*copy_buffer)(backend *self, const buffer *src, buffer *dst, int n);
	status_code (*ple_combine)(backend *self, buffer *ple, const buffer *proj, int n,
							   float combine_scale);
	status_code (*ffn_activate)(backend *self, const buffer *gate, const buffer *up, buffer *out,
								int n);
	status_code (*ffn_activate_ex)(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n, int activation);
	status_code (*rmsnorm_per_head)(backend *self, const buffer *x, const buffer *w, buffer *y,
									int n_heads, int head_dim, float eps);
	status_code (*rmsnorm_noweight)(backend *self, const buffer *x, buffer *y, int n, float eps);
	status_code (*rmsnorm_noweight_per_head)(backend *self, const buffer *x, buffer *y, int n_heads,
											 int head_dim, float eps);
	status_code (*rmsnorm_add)(backend *self, const buffer *x, const buffer *w,
							   const buffer *residual, buffer *y, int n, float eps);
	status_code (*rmsnorm_per_head_batch)(backend *self, const buffer *x, const buffer *w,
										  buffer *y, int n_heads, int head_dim, float eps, int m);
	status_code (*rmsnorm_noweight_batch)(backend *self, const buffer *x, buffer *y, int n,
										  float eps, int m);
	status_code (*rmsnorm_noweight_per_head_batch)(backend *self, const buffer *x, buffer *y,
												   int n_heads, int head_dim, float eps, int m);
	status_code (*matmul_ffn_down)(backend *self, const buffer *w, uint32_t w_type,
								   const buffer *gate, const buffer *up, buffer *y, int n, int k,
								   int activation);
	status_code (*buffer_read_f32)(backend *self, const buffer *buf, float *host_dst, int n);
	status_code (*buffer_write_f32)(backend *self, buffer *buf, const float *host_src, int n);
	status_code (*argmax)(backend *self, const buffer *logits, int n, int32_t *out_idx);
	void (*synchronize)(backend *self);
	void (*begin_batch)(backend *self);
	void (*end_batch)(backend *self);
	status_code (*rmsnorm_batch)(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
								 float eps, int m);
	status_code (*matmul_batch)(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								buffer *y, int n, int k, int m);
	status_code (*add_batch)(backend *self, buffer *x, const buffer *y, int n, int m);
	status_code (*ffn_activate_batch)(backend *self, const buffer *gate, const buffer *up,
									  buffer *out, int n, int activation, int m);
	status_code (*rope_batch)(backend *self, buffer *vec, int n_heads, int head_dim, int pos_start,
							  const float *rope_cos_base, const float *rope_sin_base, int m);
	status_code (*rope_qk_batch)(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
								 int head_dim, int pos_start, const float *rope_cos_base,
								 const float *rope_sin_base, int m);
	status_code (*attention_batch)(backend *self, const buffer *q, const buffer *k_cache,
								   const buffer *v_cache, buffer *out, int layer, int pos_start,
								   int n_heads, int n_kv_heads, int head_dim, int n_ctx,
								   int flash_attn, float scale, int n_kv_heads_active, int m);
	status_code (*attention_swa_batch)(backend *self, const buffer *q, const buffer *k_cache,
									   const buffer *v_cache, buffer *out, int layer, int pos_start,
									   int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									   int flash_attn, float scale, int sliding_window,
									   int n_kv_heads_active, int m);
	status_code (*matmul_multi_batch)(backend *self, const buffer **w, const uint32_t *w_types,
									  const buffer *x, buffer **y, const int *n_list, int k,
									  int n_matmuls, int m);
	void *priv;
	int	  rope_neox;
	float rope_theta;
	status_code (*attention_mla)(backend *self, const buffer *q, const buffer *kv_cache,
								 const buffer *k_b_w, const buffer *v_b_w, buffer *out, int layer,
								 int pos, int n_heads, int qk_head, int qk_rope, int qk_nope,
								 int v_head, int kv_lora, int n_ctx, const float *rope_cos_base,
								 const float *rope_sin_base, float scale);
	status_code (*kv_alloc_mla)(backend *self, int n_layers, int n_ctx, int kv_lora, int qk_rope,
								buffer *kv_out);
	status_code (*kv_put_mla)(backend *self, buffer *kv_cache, int layer, int pos,
							  const buffer *kv_a_in, const buffer *kv_a_norm_w, int kv_lora,
							  int qk_rope, int n_ctx, float eps);
	tpool *(*get_pool)(backend *self);
	status_code (*matmul_thread_local)(backend *self, const void *w, uint32_t w_type,
									   const float *x, float *y, int n, int k, int tid);
	status_code (*buffer_alloc_from_host)(backend *self, const void *host_data, size_t size,
										  buffer *out);
	status_code (*moe_expert_ffn_gpu)(backend *self, const buffer *x, buffer *out,
									  const moe_gpu_expert *e, int dim, int inter);
	status_code (*moe_experts_batch_gpu)(backend *self, const buffer *xb, buffer *out, int n_rows,
										 int dim, int inter, int use_gelu, int n_experts,
										 const moe_gpu_expert *experts, const int *counts,
										 const int *rows_packed, const float *weights_packed);
	size_t (*mem_available)(backend *self);
	size_t (*mem_total)(backend *self);
};

static inline int backend_has_cap(const backend *b, uint64_t cap) {
	return b && (b->caps & cap) != 0;
}

void backend_register(const char *name, backend_ctor_fn ctor);

int			backend_list(backend_info *out, int max);
status_code backend_create(const char *name, int device_index, backend **out);
status_code backend_create_best(int device_index, backend **out);
void		backend_destroy(backend *b);
void		backend_destroyed(backend *b);

status_code buffer_ensure_scratch(backend *a, buffer *b, size_t bytes);

backend *backend_host(void);

void backend_host_use(backend *b);

backend *backend_weight_home(backend *b, weight_class wc);

size_t backend_mem_available(const backend *b);
size_t backend_mem_total(const backend *b);

#define BACKEND_REGISTER(name_str, ctor_fn)                                                        \
	static void __attribute__((constructor)) backend_autoreg_##ctor_fn(void) {                     \
		backend_register(name_str, ctor_fn);                                                       \
	}

int32_t cpu_argmax_f32(const float *logits, int vocab);

#endif