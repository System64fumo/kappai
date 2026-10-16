#ifndef KAI_CUDA_INTERNAL_H
#define KAI_CUDA_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"
#include "backend/backend.h"

typedef struct cuda_host_alloc {
	void					*host_ptr;
	int						pinned;
	struct cuda_host_alloc *next;
} cuda_host_alloc;

#define CUDA_HOST_ALLOC_BUCKETS 256

void  cuda_host_alloc_note(void *host_ptr, int pinned);
int   cuda_host_alloc_pinned(void *host_ptr);
void  cuda_host_alloc_forget(void *host_ptr);

void cuda_weight_type_note(void *dev_ptr, uint32_t type);
uint32_t cuda_weight_type_of(const void *dev_ptr);
void cuda_weight_type_forget(void *dev_ptr);

#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#endif

typedef struct {
	uint16_t d;
	int8_t   qs[32];
} cuda_q8_0_block;

#define Q8_0_QM_GROUP 32

typedef struct {
	uint16_t d;
	uint16_t s;
	int8_t   qs[32];
} cuda_q8_1_block;

typedef struct {
	uint16_t d;
	uint8_t  qs[16];
} cuda_q4_0_block;

typedef struct {
	uint16_t d;
	uint16_t m;
	uint8_t  qs[16];
} cuda_q4_1_block;

typedef struct {
	uint16_t d;
	uint16_t dmin;
	uint8_t  scales[12];
	uint8_t  qs[128];
} cuda_q4_k_block;

typedef struct {
	float    d;
	int8_t   qs[256];
	int16_t  bsums[16];
} cuda_q8_k_block;

typedef struct {
	uint16_t d;
	uint8_t  qs[16];
} cuda_iq4_nl_block;

typedef struct {
	uint16_t d;
	uint8_t  qh[4];
	uint8_t  qs[16];
} cuda_q5_0_block;

typedef struct {
	uint16_t d;
	uint16_t m;
	uint8_t  qh[4];
	uint8_t  qs[16];
} cuda_q5_1_block;

typedef struct {
	uint16_t d;
	uint16_t dmin;
	uint8_t  scales[12];
	uint8_t  qh[32];
	uint8_t  qs[128];
} cuda_q5_k_block;

typedef struct {
	uint16_t d;
	uint8_t  qs[64];
	uint8_t  qh[8];
	uint8_t  signs[32];
	uint8_t  scales[4];
} cuda_iq3_s_block;

typedef struct {
	uint8_t  ql[128];
	uint8_t  qh[64];
	int8_t   scales[16];
	uint16_t d;
} cuda_q6_k_block;

#ifdef __cplusplus
extern "C" {
#endif
float        f16_to_f32(uint16_t h);
uint16_t     f32_to_f16(float f);
float        bf16_to_f32(uint16_t h);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void cuda_quantize_f32_to_q8_0(const float *x_dev, cuda_q8_0_block *xq_dev, int k);

void cuda_quantize_f32_to_q4_0(const float *x_dev, cuda_q4_0_block *xq_dev, int k);

void cuda_quantize_f32_to_q8_k(const float *x_dev, cuda_q8_k_block *xq_dev, int k);

void cuda_matmul_q8_0(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);

void cuda_matmul_q4_0(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);

void cuda_matmul_f32(const float *w_dev, const float *x_dev, float *y_dev, int n, int k,
                     cudaStream_t stream);

void cuda_matmul_f16(const uint16_t *w_dev, const float *x_dev, float *y_dev, int n, int k,
                     cudaStream_t stream);

void cuda_matmul_bf16(const uint16_t *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);

void cuda_rmsnorm(const float *x_dev, const float *w_dev, float *y_dev, int n, float eps,
                  cudaStream_t stream);
void cuda_rmsnorm_per_head(const float *x_dev, const float *w_dev, float *y_dev, int n_heads,
                           int head_dim, float eps, cudaStream_t stream);

void cuda_rmsnorm_add(const float *x_dev, const float *w_dev, const float *residual_dev,
                      float *y_dev, int n, float eps, float out_scale, cudaStream_t stream);

void cuda_rmsnorm_per_head_g(const float *x_dev, const float *w_dev, float *y_dev,
                             int n_heads, int head_dim, float eps,
                             const int *params_dev, cudaStream_t stream);

void cuda_rmsnorm_rope(const float *x_dev, const float *w_dev, float *y_dev,
                       int n_heads, int head_dim, const float *cos_dev,
                       const float *sin_dev, float eps, int neox, cudaStream_t stream);

void cuda_matmul_ffn_down(const void *w_dev, const float *gate_dev, const float *up_dev,
                          float *y_dev, int n, int k, cudaStream_t stream);

void cuda_matmul_ffn_down_q4(const void *w_dev, const float *gate_dev, const float *up_dev,
                             float *y_dev, int n, int k, cudaStream_t stream);

void cuda_matmul_ffn_down_q4_1(const void *w_dev, const float *gate_dev, const float *up_dev,
                               float *y_dev, int n, int k, cudaStream_t stream);

void cuda_matmul_q4_1(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);

void cuda_matmul_q8_0_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev,
                               int n, int k, cudaStream_t stream);
void cuda_matmul_q4_0_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev,
                               int n, int k, cudaStream_t stream);
void cuda_matmul_q4_1_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev,
                               int n, int k, cudaStream_t stream);

#ifdef __cplusplus
extern "C" {
#endif
void cuda_matmul_iq4_nl(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                        cudaStream_t stream);
void cuda_matmul_iq4_nl_residual(const void *w_dev, const float *x_dev,
                                 const float *residual_dev, float *y_dev, int n, int k,
                                 cudaStream_t stream);
void cuda_matmul_batch_iq4_nl(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                              int m, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
void cuda_matmul_ffn_down_batch_launch(const void *w_dev, const float *gate_dev,
                                       const float *up_dev, float *y_dev, int n, int k,
                                       int m, cudaStream_t stream);
void cuda_matmul_ffn_down_q4_batch_launch(const void *w_dev, const float *gate_dev,
                                          const float *up_dev, float *y_dev, int n, int k,
                                          int m, cudaStream_t stream);
void cuda_matmul_ffn_down_q4_1_batch_launch(const void *w_dev, const float *gate_dev,
                                            const float *up_dev, float *y_dev, int n, int k,
                                            int m, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

status_code cuda_moe_experts_batch(backend *self, const buffer *xb, buffer *out,
                                    int n_rows, int dim, int inter, int use_gelu,
                                    int n_experts, const moe_resident_expert *experts,
                                    const int *counts, const int *rows_packed,
                                    const float *weights_packed, cudaStream_t stream);

status_code cuda_moe_gate_proj(const void *w_dev, uint32_t w_type, const float *x_dev,
                                float *gate_dev, int n_rows, int dim, int inter,
                                cudaStream_t stream);
status_code cuda_moe_up_proj(const void *w_dev, uint32_t w_type, const float *x_dev,
                              float *up_dev, int n_rows, int dim, int inter,
                              cudaStream_t stream);
status_code cuda_moe_down_proj(const void *w_dev, uint32_t w_type, const float *act_dev,
                                float *y_dev, int n_rows, int inter, int dim,
                                cudaStream_t stream);
status_code cuda_moe_scale_accum(float *out, const float *src, int n_rows, int dim,
                                  float weight, float down_scale, cudaStream_t stream);

void cuda_matmul_q4_k(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);

#ifdef __cplusplus
extern "C" {
#endif
void cuda_matmul_q6_k(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);
void cuda_matmul_q6_k_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev, int n, int k,
                               cudaStream_t stream);
void cuda_matmul_batch_q6_k(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                            int m, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

void cuda_matmul_multi_q8_0(const void *const *w_dev, float *const *y_dev, const int *n_host,
                             const float *x_dev, int nm, int k, cudaStream_t stream);
void cuda_matmul_multi_q4_0(const void *const *w_dev, float *const *y_dev, const int *n_host,
                             const float *x_dev, int nm, int k, cudaStream_t stream);

void cuda_rmsnorm_matmul_multi_q8_0(const void *const *w_dev, float *const *y_dev,
                                    const int *n_host, const float *x_dev,
                                    const float *norm_w_dev, int nm, int k,
                                    float eps, cudaStream_t stream);
void cuda_rmsnorm_matmul_multi_q4_0(const void *const *w_dev, float *const *y_dev,
                                    const int *n_host, const float *x_dev,
                                    const float *norm_w_dev, int nm, int k,
                                    float eps, cudaStream_t stream);
void cuda_rmsnorm_matmul_multi_q4_1(const void *const *w_dev, float *const *y_dev,
                                    const int *n_host, const float *x_dev,
                                    const float *norm_w_dev, int nm, int k,
                                    float eps, cudaStream_t stream);

void cuda_attn_f16(const float *q_dev, const uint16_t *kc_dev,
                   const uint16_t *vc_dev, float *out_dev, int n_heads,
                   int n_kv_heads, int head_dim, size_t layer_base_bytes, size_t kvh_stride,
                   int attn_start, int n_pos_total, float scale, cudaStream_t stream);

void cuda_attn_f16_decode_split(const float *q_dev, const uint16_t *kc_dev,
                   const uint16_t *vc_dev, float *out_dev, int n_heads,
                   int n_kv_heads, int head_dim, size_t layer_base_bytes, size_t kvh_stride,
                   int n_pos, float scale, cudaStream_t stream);

void cuda_attn_f16_decode_split_g(const float *q_dev, const uint16_t *kc_dev,
                   const uint16_t *vc_dev, float *out_dev, int n_heads,
                   int n_kv_heads, int head_dim, size_t layer_base_bytes, size_t kvh_stride,
                   float scale, const int *params_dev, int sliding_window,
                   cudaStream_t stream);
void cuda_attn_dec_split_ensure(void);

void cuda_attn_q8(const float *q_dev, const uint8_t *kc_dev,
                  const uint8_t *vc_dev, float *out_dev, int n_heads,
                  int n_kv_heads, int head_dim, size_t layer_base_bytes,
                  size_t kvh_stride_bytes, int attn_start, int n_pos_total,
                  float scale, int n_blocks, cudaStream_t stream);
void cuda_attn_batch_q8(const float *q_dev, const uint8_t *kc_dev,
                        const uint8_t *vc_dev, float *out_dev, int n_heads,
                        int n_kv_heads, int head_dim, size_t layer_base_bytes,
                        size_t kvh_stride_bytes, int pos_start, int m,
                        int sliding_window, float scale, int n_blocks,
                        cudaStream_t stream);

void cuda_kv_put_f16(const float *k_in, const float *v_in,
                     uint16_t *kd, uint16_t *vd, int n_kv_heads, int head_dim,
                     size_t layer_base_bytes, size_t kvh_stride, int pos, int n_ctx,
                     cudaStream_t stream);

void cuda_kv_alloc_f16(uint16_t **kd_out, uint16_t **vd_out, size_t size, cudaStream_t stream);

void cuda_argmax(const float *logits_dev, int n, int32_t *out_idx_dev, cudaStream_t stream);

void cuda_add_inplace(float *x_dev, const float *y_dev, int n, cudaStream_t stream);

void cuda_scale_inplace(float *x_dev, float scale, int n, cudaStream_t stream);

void cuda_softcap(float *x_dev, float cap, long long n, cudaStream_t stream);
void cuda_attn_output_gate(float *out_dev, const float *gate_dev, long long n,
                           cudaStream_t stream);
void cuda_split_qgate(const float *mixed_dev, float *q_dev, float *gate_dev, int n_heads,
                      int head_dim, int n_rows, cudaStream_t stream);
void cuda_partial_rope(float *vec_dev, int n_heads, int head_dim, int rope_dim, int pos_start,
                       const float *cos_dev, const float *sin_dev, int n_rows,
                       cudaStream_t stream);
void cuda_moe_activate(const float *gate_dev, const float *up_dev, float *out_dev, long long n,
                       float gate_scale, float up_scale, int use_gelu, cudaStream_t stream);

void cuda_attn_batch_f16(const float *q_dev, const uint16_t *kc_dev,
                         const uint16_t *vc_dev, float *out_dev, int n_heads,
                         int n_kv_heads, int head_dim, size_t layer_base_bytes,
                         size_t kvh_stride, int pos_start, int m,
                         int sliding_window, float scale, cudaStream_t stream);
void cuda_kv_put_batch_f16(const float *k_in, const float *v_in,
                           uint16_t *kd, uint16_t *vd, int n_kv_heads, int head_dim,
                           size_t layer_base_bytes, size_t kvh_stride, int pos_start,
                           int n_ctx, int in_row_stride, int m, cudaStream_t stream);

void cuda_kv_put_q8(const float *k_in, const float *v_in,
                    uint8_t *kd, uint8_t *vd, int n_kv_heads, int head_dim,
                    size_t layer_base_bytes, size_t kvh_stride_bytes,
                    int pos, int n_blocks, cudaStream_t stream);
void cuda_kv_put_batch_q8(const float *k_in, const float *v_in,
                          uint8_t *kd, uint8_t *vd, int n_kv_heads, int head_dim,
                          size_t layer_base_bytes, size_t kvh_stride_bytes,
                          int pos_start, int n_seq, int n_blocks,
                          int in_row_stride, cudaStream_t stream);

void cuda_rmsnorm_batch(const float *x_dev, const float *w_dev, float *y_dev,
                        int n, float eps, int m, cudaStream_t stream);
void cuda_add_batch(float *x_dev, const float *y_dev, int n, int m, cudaStream_t stream);
void cuda_ffn_activate_batch(const float *gate_dev, const float *up_dev, float *out_dev,
                             int n, int activation, int m, cudaStream_t stream);

#ifdef __cplusplus
extern "C" {
#endif
void cuda_matmul_q5_0(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);
void cuda_matmul_q5_0_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev, int n, int k,
                               cudaStream_t stream);
void cuda_matmul_batch_q5_0(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                            int m, cudaStream_t stream);
void cuda_matmul_q5_1(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);
void cuda_matmul_q5_1_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev, int n, int k,
                               cudaStream_t stream);
void cuda_matmul_batch_q5_1(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                            int m, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
void cuda_matmul_q5_k(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);
void cuda_matmul_q5_k_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev, int n, int k,
                               cudaStream_t stream);
void cuda_matmul_batch_q5_k(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                            int m, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
void cuda_matmul_iq3_s(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                       cudaStream_t stream);
void cuda_matmul_iq3_s_residual(const void *w_dev, const float *x_dev,
                                const float *residual_dev, float *y_dev, int n, int k,
                                cudaStream_t stream);
void cuda_matmul_batch_iq3_s(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                             int m, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

void cuda_ffn_activate_fused_batch(const float *fused_dev, float *out_dev, int n,
                                   int activation, int m, cudaStream_t stream);
void cuda_rope_batch(float *vec_dev, int n_heads, int head_dim, int pos_start,
                     const float *rope_cos_dev, const float *rope_sin_dev,
                     int m, int neox, cudaStream_t stream);
void cuda_rope_qk_batch(float *q_dev, float *k_dev, int n_heads, int n_kv_heads,
                        int head_dim, int pos_start,
                        const float *rope_cos_dev, const float *rope_sin_dev,
                        int m, int neox, cudaStream_t stream);

void cuda_matmul_batch_q8_0(const void *w_dev, const float *x_dev, float *y_dev,
                             int n, int k, int m, cudaStream_t stream);

void cuda_matmul_batch_fp16(const void *w_dev, const float *x_dev, float *y_dev,
                            int n, int k, int m, cudaStream_t stream);
void cuda_matmul_batch_q4_0(const void *w_dev, const float *x_dev, float *y_dev,
                             int n, int k, int m, cudaStream_t stream);
void cuda_matmul_batch_q4_1(const void *w_dev, const float *x_dev, float *y_dev,
                             int n, int k, int m, cudaStream_t stream);
void cuda_matmul_batch_f32(const float *w_dev, const float *x_dev, float *y_dev,
                            int n, int k, int m, cudaStream_t stream);
void cuda_matmul_batch_q4_k(const void *w_dev, const float *x_dev, float *y_dev,
                            int n, int k, int m, cudaStream_t stream);
void cuda_matmul_batch_f16(const uint16_t *w_dev, const float *x_dev, float *y_dev,
                           int n, int k, int m, cudaStream_t stream);
void cuda_matmul_batch_bf16(const uint16_t *w_dev, const float *x_dev, float *y_dev,
                            int n, int k, int m, cudaStream_t stream);

void cuda_ffn_activate(const float *gate_dev, const float *up_dev, float *out_dev,
                        int n, int activation, cudaStream_t stream);

void cuda_rope_ext(float *vec_dev, int n_heads, int head_dim, int pos,
                   const float *rope_cos_base, const float *rope_sin_base,
                   int neox, cudaStream_t stream);

void cuda_embd_lookup_f32(const float *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_f16(const uint16_t *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_bf16(const uint16_t *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_q8_0(const cuda_q8_0_block *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_q4_0(const cuda_q4_0_block *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_q4_k(const cuda_q4_k_block *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);

typedef struct {
    int pos;
    int n_pos;
    int token;
    int pad;
} cuda_decode_params;

void cuda_ple_combine(float *ple_dev, const float *proj_dev, int n, float scale,
                      cudaStream_t stream);

void cuda_ple_norm_batch(float *proj_dev, const float *norm_w_dev, int n_rows,
                         int total_ple, int n_embd, int n_layers, float eps,
                         cudaStream_t stream);

void cuda_rmsnorm_add_batch(const float *x_dev, const float *w_dev,
                            const float *residual_dev, float *y_dev, int n, float eps,
                            int m, float out_scale, cudaStream_t stream);

void cuda_matmul_multi_batch_q4_0(const void *const *w_dev, float *const *y_dev,
                                  const int *n_host, const float *x_dev, int nm, int k,
                                  int m, const float *resid_dev, cudaStream_t stream);
void cuda_matmul_multi_batch_q8_0(const void *const *w_dev, float *const *y_dev,
                                  const int *n_host, const float *x_dev, int nm, int k,
                                  int m, const float *resid_dev, cudaStream_t stream);

void cuda_rope_ext_g(float *vec_dev, int n_heads, int head_dim,
                     const float *cos_base, const float *sin_base,
                     const int *params_dev, int neox, cudaStream_t stream);
void cuda_attn_f16_g(const float *q_dev, const uint16_t *kc_dev,
                     const uint16_t *vc_dev, float *out_dev, int n_heads,
                     int n_kv_heads, int head_dim, size_t layer_base_bytes,
                     size_t kvh_stride, int sliding_window, float scale,
                     const int *params_dev, cudaStream_t stream);

void cuda_attn_f16_split_g(const float *q_dev, const uint16_t *kc_dev,
                           const uint16_t *vc_dev, float *out_dev, int n_heads,
                           int n_kv_heads, int head_dim, size_t layer_base_bytes,
                           size_t kvh_stride, int sliding_window, float scale,
                           int n_queries, int n_splits, const int *params_dev,
                           cudaStream_t stream);

void cuda_attn_f16_gqa_g(const float *q_dev, const uint16_t *kc_dev,
                         const uint16_t *vc_dev, float *out_dev, int n_heads,
                         int n_kv_heads, int head_dim, size_t layer_base_bytes,
                         size_t kvh_stride, int sliding_window, float scale,
                         int n_queries, const int *params_dev, cudaStream_t stream);
void cuda_kv_put_f16_g(const float *k_in, const float *v_in,
                       uint16_t *kd, uint16_t *vd, int n_kv_heads, int head_dim,
                       size_t layer_base_bytes, size_t kvh_stride, int n_ctx,
                       const int *params_dev, cudaStream_t stream);
void cuda_embd_lookup_q4_0_g(const cuda_q4_0_block *embd_dev, float *out_dev,
                             int dim, const int *params_dev, cudaStream_t stream);
void cuda_embd_lookup_q8_0_g(const cuda_q8_0_block *embd_dev, float *out_dev,
                             int dim, const int *params_dev, cudaStream_t stream);
void cuda_embd_lookup_q4_k_g(const cuda_q4_k_block *embd_dev, float *out_dev,
                              int dim, const int *params_dev, cudaStream_t stream);
void cuda_kv_put_q8_g(const float *k_in, const float *v_in,
                      uint8_t *kd, uint8_t *vd, int n_kv_heads, int head_dim,
                      size_t layer_base_bytes, size_t kvh_stride_bytes,
                      int n_blocks, const int *params_dev, cudaStream_t stream);
void cuda_attn_q8_g(const float *q_dev, const uint8_t *kc_dev,
                    const uint8_t *vc_dev, float *out_dev, int n_heads,
                    int n_kv_heads, int head_dim, size_t layer_base_bytes,
                    size_t kvh_stride_bytes, int sliding_window, float scale,
                    int n_blocks, const int *params_dev, cudaStream_t stream);

void cuda_attn_q8_gqa_g(const float *q_dev, const uint8_t *kc_dev,
                        const uint8_t *vc_dev, float *out_dev, int n_heads,
                        int n_kv_heads, int head_dim, size_t layer_base_bytes,
                        size_t kvh_stride_bytes, int sliding_window, float scale,
                        int n_blocks, int n_queries, const int *params_dev,
                        cudaStream_t stream);

void cuda_qkv_norm_rope(float *q_dev, float *k_dev, float *v_dev,
                        const float *wq_norm_dev, const float *wk_norm_dev,
                        int n_heads, int n_kv_heads, int head_dim, float eps,
                        int pos, const float *cos_tbl_row, const float *sin_tbl_row,
                        int neox, cudaStream_t stream);
void cuda_qkv_norm_rope_g(float *q_dev, float *k_dev, float *v_dev,
                          const float *wq_norm_dev, const float *wk_norm_dev,
                          int n_heads, int n_kv_heads, int head_dim, float eps,
                          const float *cos_base, const float *sin_base,
                          const int *params_dev, int neox, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
