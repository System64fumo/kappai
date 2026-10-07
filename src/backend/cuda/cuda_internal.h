#ifndef KAI_CUDA_INTERNAL_H
#define KAI_CUDA_INTERNAL_H

/*
 * Shared definitions for the CUDA backend.
 *
 * These block layouts are duplicated from ggml (see src/gguf.h /
 * src/backend/cpu/scalar/quants.h) on purpose: the CUDA kernels operate on
 * raw device memory that must be byte-identical to what CPU code expects, so
 * we cannot rely on a host-side struct being ABI-compatible across the nvcc/gcc
 * boundary.  Keep these in sync with ggml's Q8_0 / Q4_0 block formats.
 */

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

/* Quad-major relayout types, engine-internal and CUDA-only: they exist solely
 * to give the Q4_0/Q8_0 GEMV kernels a coalesced byte order, and never appear in
 * a GGUF file. Upstream keeps them in the shared gguf.h enum; they are declared
 * here instead so the CUDA backend does not have to widen the shared type list.
 * The numeric values continue the 0x40+ internal-repack range (Q8_0_QM 0x4A,
 * Q4_0_QM 0x4B) and must not collide with a type the engine can actually see. */
enum {
	CUDA_GGML_TYPE_Q8_0_QM = 0x4A,
	CUDA_GGML_TYPE_Q4_0_QM = 0x4B,
};

/*
 * Per-buffer bookkeeping that the shared `buffer` struct does not carry.
 *
 * `host_ptr` is a void* into which we must remember how the host side was
 * allocated: cudaHostAlloc() memory has to go back through cudaFreeHost(),
 * while malloc()/managed memory goes through free() or cudaFree(). Upstream
 * tracks this with a `bool host_is_pinned` field on `buffer`; to keep this
 * backend self-contained (no edits to the shared header) the flag lives in a
 * small side table keyed by the host pointer instead. A hash table keeps the
 * lookup O(1) so cuda_buffer_free() stays cheap on the decode hot path.
 *
 * Not thread-safe by design: buffer alloc/free are serialised by the engine's
 * single-threaded buffer scratch accounting, exactly as they are upstream.
 */
typedef struct cuda_host_alloc {
	void					*host_ptr; /* key: the buffer's host_ptr */
	int						pinned;		/* 1 = cudaHostAlloc, 0 = malloc/managed */
	struct cuda_host_alloc *next;
} cuda_host_alloc;

#define CUDA_HOST_ALLOC_BUCKETS 256

void  cuda_host_alloc_note(void *host_ptr, int pinned);
int   cuda_host_alloc_pinned(void *host_ptr); /* -1 if unknown */
void  cuda_host_alloc_forget(void *host_ptr);

/*
 * Per-weight-buffer type override.
 *
 * The engine decides which matmul kernel runs by passing the *model's* declared
 * type to matmul()/matmul_batch(). That is correct for weights the backend
 * uploads verbatim, but this backend also relayouts some weights at upload
 * time (Q4_0 -> Q8_0 promotion, then the quad-major byte order). The engine
 * never learns about that, so it would hand the matmul a Q8_0 tag for data that
 * is actually Q8_0_QM and the wrong kernel would read it.
 *
 * Upstream avoids this by rewriting the type in model.c, which the engine here
 * does not do. Instead the backend records what it actually stored and the
 * matmul entry points prefer that record over the engine's tag. Keyed by the
 * device pointer, same lifetime as the buffer.
 */
void cuda_weight_type_note(void *dev_ptr, uint32_t type);
uint32_t cuda_weight_type_of(const void *dev_ptr); /* returns type unchanged if unknown */
void cuda_weight_type_forget(void *dev_ptr);

/* Forward declaration for CUDA stream type when not compiling with nvcc */
#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#endif

/* Q8_0 block: one dequant scale (F16) + 32 signed-quantized values. */
typedef struct {
	uint16_t d;
	int8_t   qs[32];
} cuda_q8_0_block;

/* Q8_0 quad-major (QM) relayout -- GGML_TYPE_Q8_0_QM. THE layout spec.
 *
 * Same 34B/32e geometry as Q8_0 (nb*34 row stride shared, so buffer sizes,
 * row_bytes, and VRAM estimates are IDENTICAL; only byte ORDER differs).
 * Writer: repack_q8_0_row_qmajor (src/model.c). Readers: all Q8_0 CUDA
 * kernels take `int qmajor` (0 = original, 1 = QM).
 *
 * A row of nb = k/32 blocks is stored as nb groups of 34 bytes. Group g
 * covers blocks [g*G, g*G+G) with G = 32 (tail group may be shorter).
 * Within a group, bytes are transposed 32x34: byte j of the 34-byte
 * block stride holds that byte for all G blocks in the group, i.e.
 *   group_base = row + g*34*G
 *   scale bits of block (g*G+r): u16 at group_base + 2*r
 *   quad t (t = 0..7) of block (g*G+r): 4 bytes at group_base + 2*G + t*4*G + 4*r
 *     (quads follow the G scales occupying the first 2*G bytes)
 * READER RULE: rows are nb*34 bytes, i.e. only 2B-aligned when nb is
 * odd. Readers MUST use 1B/2B loads (never u32 __ldg: misaligned
 * address faults poison the CUDA context for later launches).
 * Tail group (nb % 32 != 0): same pattern with G' = nb % 32.
 * Dequant identity: qval(b,i) == qval_qm(b,i) for every element.
 *
 * Invariants: embeddings/KV/x-quant scratch are NEVER repacked (class and
 * system separation in model.c upload); CPU must never see QM buffers
 * (host-fallback sites reject loudly); files never contain QM.
 *
 * Q4_0 quad-major (GGML_TYPE_Q4_0_QM) follows the same pattern with
 * 18B blocks: group of G blocks -> 2*G scale bytes, then 16 byte
 * planes of G bytes (qs byte j of block r at 2*G + j*G + r).
 * Nibble unpacking is unchanged (kernel-specific). Same 2B rule. */
#define Q8_0_QM_GROUP 32

/* Q8_1 block: one dequant scale (F16) + one sum (F16) + 32 signed-quantized values.
 * The sum enables dp4a to work directly without per-block scale multiplication. */
typedef struct {
	uint16_t d;
	uint16_t s;    /* FP16 sum of quantized values */
	int8_t   qs[32];
} cuda_q8_1_block;

/* Q4_0 block: one dequant scale (F16) + 16 unsigned nibble values. */
typedef struct {
	uint16_t d;
	uint8_t  qs[16];
} cuda_q4_0_block;

/* Q4_1 block: dequant scale (F16) + offset (F16) + 16 unsigned nibble values.
 * Dequant: v = q * d + m, same nibble layout as Q4_0. */
typedef struct {
	uint16_t d;
	uint16_t m;
	uint8_t  qs[16];
} cuda_q4_1_block;

/* Q4_K block: 256 elements, 4 groups of 64 elements each.
 * Structure: d (F16), dmin (F16), scales[12] (16 6-bit scale/min values),
 * qs[128] (256 4-bit values packed 2/byte). Total 144 bytes. */
typedef struct {
	uint16_t d;
	uint16_t dmin;
	uint8_t  scales[12];
	uint8_t  qs[128];
} cuda_q4_k_block;

/* Q8_K block: 256 elements.
 * Structure: d (float), qs[256] (int8), bsums[16] (int16). */
typedef struct {
	float    d;
	int8_t   qs[256];
	int16_t  bsums[16];
} cuda_q8_k_block;

/* Host-side helpers (resolved from the scalar backend's weak definitions so
 * that dequant scales match the CPU reference bit-for-bit). */
#ifdef __cplusplus
extern "C" {
#endif
float        f16_to_f32(uint16_t h);
uint16_t     f32_to_f16(float f);
float        bf16_to_f32(uint16_t h);
#ifdef __cplusplus
}
#endif

/* Kernel launchers implemented in cuda_kernels.cu (compiled with nvcc). */
#ifdef __cplusplus
extern "C" {
#endif

/* Quantize an F32 buffer of length k into Q8_0 blocks on the device. */
void cuda_quantize_f32_to_q8_0(const float *x_dev, cuda_q8_0_block *xq_dev, int k);

/* Quantize an F32 buffer of length k into Q4_0 blocks on the device. */
void cuda_quantize_f32_to_q4_0(const float *x_dev, cuda_q4_0_block *xq_dev, int k);

/* Quantize an F32 buffer of length k into Q8_K blocks on the device. */
void cuda_quantize_f32_to_q8_k(const float *x_dev, cuda_q8_k_block *xq_dev, int k);

/* Q8_0 weight matmul: quantizes x internally (Q8_0), then y = W * x.
 * qmajor: 1 = weight buffer is quad-major relayout (GGML_TYPE_Q8_0_QM),
 * 0 = original Q8_0 order. Readers land in A2; A0 threads the flag. */
void cuda_matmul_q8_0(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream, int qmajor);



/* Q4_0 weight matmul: quantizes x internally (Q8_0), then y = W * x.
 * qmajor: 1 = quad-major relayout (GGML_TYPE_Q4_0_QM), 0 = original. */
void cuda_matmul_q4_0(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream, int qmajor);

/* F32 weight matmul. */
void cuda_matmul_f32(const float *w_dev, const float *x_dev, float *y_dev, int n, int k,
                     cudaStream_t stream);

/* F16 weight matmul (W stored as raw F16 bit patterns). */
void cuda_matmul_f16(const uint16_t *w_dev, const float *x_dev, float *y_dev, int n, int k,
                     cudaStream_t stream);

/* BF16 weight matmul (W stored as raw BF16 bit patterns). */
void cuda_matmul_bf16(const uint16_t *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);

/* RMSNorm.  w_dev may be NULL for the noweight variants. */
void cuda_rmsnorm(const float *x_dev, const float *w_dev, float *y_dev, int n, float eps,
                  cudaStream_t stream);
void cuda_rmsnorm_per_head(const float *x_dev, const float *w_dev, float *y_dev, int n_heads,
                           int head_dim, float eps, cudaStream_t stream);

/* Fused RMSNorm + Residual Add: y = (x + residual) * scale * w */
void cuda_rmsnorm_add(const float *x_dev, const float *w_dev, const float *residual_dev,
                      float *y_dev, int n, float eps, float out_scale, cudaStream_t stream);

/* Graph-capture launchers (_g suffix): same kernels, take decode_params for
 * capture compatibility (params may be unused by kernel itself). */
void cuda_rmsnorm_per_head_g(const float *x_dev, const float *w_dev, float *y_dev,
                             int n_heads, int head_dim, float eps,
                             const int *params_dev, cudaStream_t stream);

/* Fused RMSNorm + RoPE: y = rmsnorm(x, w), then apply RoPE rotation */
void cuda_rmsnorm_rope(const float *x_dev, const float *w_dev, float *y_dev,
                       int n_heads, int head_dim, const float *cos_dev,
                       const float *sin_dev, float eps, int neox, cudaStream_t stream);

/* Fused Q8_0 down-projection with GELU activation:
 * y = W * (gelu_tanh(gate) * up).  The activation is folded into the Q8_0
 * quantization of the input so no host round-trip is needed. */
void cuda_matmul_ffn_down(const void *w_dev, const float *gate_dev, const float *up_dev,
                          float *y_dev, int n, int k, cudaStream_t stream,
                          int qmajor);

/* Same fused GELU down-projection with Q4_0 down-weights. */
void cuda_matmul_ffn_down_q4(const void *w_dev, const float *gate_dev, const float *up_dev,
                             float *y_dev, int n, int k, cudaStream_t stream,
                             int qmajor);

/* Same fused GELU down-projection with Q4_1 down-weights. */
void cuda_matmul_ffn_down_q4_1(const void *w_dev, const float *gate_dev, const float *up_dev,
                               float *y_dev, int n, int k, cudaStream_t stream);

/* Q4_1 weight matmul: quantizes x internally (Q8_0), then y = W * x. */
void cuda_matmul_q4_1(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);

/* Fused residual matmuls: y = W*x + residual. Quantize x once, dispatch
 * MMVQ (M=1) or DP4A residual kernels (M>1). */
void cuda_matmul_q8_0_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev,
                               int n, int k, cudaStream_t stream, int qmajor);
void cuda_matmul_q4_0_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev,
                               int n, int k, cudaStream_t stream, int qmajor);
void cuda_matmul_q4_1_residual(const void *w_dev, const float *x_dev,
                               const float *residual_dev, float *y_dev,
                               int n, int k, cudaStream_t stream);

/* Q4_K weight matmul: quantizes x internally (Q8_K), then y = W * x. */
void cuda_matmul_q4_k(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                      cudaStream_t stream);

/* Fused multi-column matmul.  All columns share x (quantized once);
 * w_dev/y_dev are device pointer arrays of length nm, n_host holds the
 * number of output rows per column, k is the shared input width. */
void cuda_matmul_multi_q8_0(const void *const *w_dev, float *const *y_dev, const int *n_host,
                             const float *x_dev, int nm, int k, cudaStream_t stream,
                             int qmajor);
void cuda_matmul_multi_q4_0(const void *const *w_dev, float *const *y_dev, const int *n_host,
                             const float *x_dev, int nm, int k, cudaStream_t stream,
                             int qmajor);

/* Fused RMSNorm + Multi-MatMul: single kernel does RMSNorm(x), quantize
 * to Q8_0 in shared memory, then multi-column matmul. */
void cuda_rmsnorm_matmul_multi_q8_0(const void *const *w_dev, float *const *y_dev,
                                    const int *n_host, const float *x_dev,
                                    const float *norm_w_dev, int nm, int k,
                                    float eps, cudaStream_t stream, int qmajor);
void cuda_rmsnorm_matmul_multi_q4_0(const void *const *w_dev, float *const *y_dev,
                                    const int *n_host, const float *x_dev,
                                    const float *norm_w_dev, int nm, int k,
                                    float eps, cudaStream_t stream, int qmajor);
void cuda_rmsnorm_matmul_multi_q4_1(const void *const *w_dev, float *const *y_dev,
                                    const int *n_host, const float *x_dev,
                                    const float *norm_w_dev, int nm, int k,
                                    float eps, cudaStream_t stream);

/* F16 KV Cache Attention */
void cuda_attn_f16(const float *q_dev, const uint16_t *kc_dev,
                   const uint16_t *vc_dev, float *out_dev, int n_heads,
                   int n_kv_heads, int head_dim, size_t layer_base_bytes, size_t kvh_stride,
                   int attn_start, int n_pos_total, float scale, cudaStream_t stream);

/* Split decode attention (eager long-context path): grid over
 * heads x splits + combine. Caller pre-offsets SWA into layer_base. */
void cuda_attn_f16_decode_split(const float *q_dev, const uint16_t *kc_dev,
                   const uint16_t *vc_dev, float *out_dev, int n_heads,
                   int n_kv_heads, int head_dim, size_t layer_base_bytes, size_t kvh_stride,
                   int n_pos, float scale, cudaStream_t stream);

/* Graph variant (n_pos/SWA from device params) + scratch pre-grow
 * (must run outside capture; reuse path is capture-safe). */
void cuda_attn_f16_decode_split_g(const float *q_dev, const uint16_t *kc_dev,
                   const uint16_t *vc_dev, float *out_dev, int n_heads,
                   int n_kv_heads, int head_dim, size_t layer_base_bytes, size_t kvh_stride,
                   float scale, const int *params_dev, int sliding_window,
                   cudaStream_t stream);
void cuda_attn_dec_split_ensure(void);

/* Q8_0 KV Cache Attention (kc/vc are byte buffers; strides in bytes;
 * n_blocks = ceil(head_dim/32)). */
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

/* F16 KV Cache write */
void cuda_kv_put_f16(const float *k_in, const float *v_in,
                     uint16_t *kd, uint16_t *vd, int n_kv_heads, int head_dim,
                     size_t layer_base_bytes, size_t kvh_stride, int pos, int n_ctx,
                     cudaStream_t stream);

/* KV Cache allocation */
void cuda_kv_alloc_f16(uint16_t **kd_out, uint16_t **vd_out, size_t size, cudaStream_t stream);

/* argmax: find index of maximum value in float array */
void cuda_argmax(const float *logits_dev, int n, int32_t *out_idx_dev, cudaStream_t stream);

/* add_inplace: x[i] += y[i] */
void cuda_add_inplace(float *x_dev, const float *y_dev, int n, cudaStream_t stream);

/* scale_inplace: x[i] *= scale */
void cuda_scale_inplace(float *x_dev, float scale, int n, cudaStream_t stream);

/* Ops the engine dispatches via OP_BACKEND: a NULL slot silently reroutes to
 * a host backend, which would then dereference a device pointer. Device
 * backends must implement these natively. */
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

/* Batched attention/KV: grid over rows. sliding_window <= 0 disables window. */
void cuda_attn_batch_f16(const float *q_dev, const uint16_t *kc_dev,
                         const uint16_t *vc_dev, float *out_dev, int n_heads,
                         int n_kv_heads, int head_dim, size_t layer_base_bytes,
                         size_t kvh_stride, int pos_start, int m,
                         int sliding_window, float scale, cudaStream_t stream);
void cuda_kv_put_batch_f16(const float *k_in, const float *v_in,
                           uint16_t *kd, uint16_t *vd, int n_kv_heads, int head_dim,
                           size_t layer_base_bytes, size_t kvh_stride, int pos_start,
                           int n_ctx, int in_row_stride, int m, cudaStream_t stream);

/* Q8_0 KV Cache write (kd/vd are byte buffers; kvh_stride in bytes;
 * n_blocks = ceil(head_dim/32)). */
void cuda_kv_put_q8(const float *k_in, const float *v_in,
                    uint8_t *kd, uint8_t *vd, int n_kv_heads, int head_dim,
                    size_t layer_base_bytes, size_t kvh_stride_bytes,
                    int pos, int n_blocks, cudaStream_t stream);
void cuda_kv_put_batch_q8(const float *k_in, const float *v_in,
                          uint8_t *kd, uint8_t *vd, int n_kv_heads, int head_dim,
                          size_t layer_base_bytes, size_t kvh_stride_bytes,
                          int pos_start, int n_seq, int n_blocks,
                          int in_row_stride, cudaStream_t stream);

/* True batched elementwise ops (prefill): all m rows in one launch. */
void cuda_rmsnorm_batch(const float *x_dev, const float *w_dev, float *y_dev,
                        int n, float eps, int m, cudaStream_t stream);
void cuda_add_batch(float *x_dev, const float *y_dev, int n, int m, cudaStream_t stream);
void cuda_ffn_activate_batch(const float *gate_dev, const float *up_dev, float *out_dev,
                             int n, int activation, int m, cudaStream_t stream);
void cuda_rope_batch(float *vec_dev, int n_heads, int head_dim, int pos_start,
                     const float *rope_cos_dev, const float *rope_sin_dev,
                     int m, int neox, cudaStream_t stream);
void cuda_rope_qk_batch(float *q_dev, float *k_dev, int n_heads, int n_kv_heads,
                        int head_dim, int pos_start,
                        const float *rope_cos_dev, const float *rope_sin_dev,
                        int m, int neox, cudaStream_t stream);

/* True batched matmul (prefill): M rows in one launch. */
void cuda_matmul_batch_q8_0(const void *w_dev, const float *x_dev, float *y_dev,
                             int n, int k, int m, cudaStream_t stream, int qmajor);
/* Pure-FP16 WMMA batch GEMM over an fp16_shadow (prefill). No dequant;
 * arithmetic differs from the quant path (exactness gated). */
void cuda_matmul_batch_fp16(const void *w_dev, const float *x_dev, float *y_dev,
                            int n, int k, int m, cudaStream_t stream);
void cuda_matmul_batch_q4_0(const void *w_dev, const float *x_dev, float *y_dev,
                             int n, int k, int m, cudaStream_t stream, int qmajor);
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

/* ffn_activate: out = activation(gate) * up. activation: 0=SwiGLU, 1=GELU */
void cuda_ffn_activate(const float *gate_dev, const float *up_dev, float *out_dev,
                        int n, int activation, cudaStream_t stream);

/* rope_ext: RoPE rotation using precomputed cos/sin tables */
void cuda_rope_ext(float *vec_dev, int n_heads, int head_dim, int pos,
                   const float *rope_cos_base, const float *rope_sin_base,
                   int neox, cudaStream_t stream);

/* embd_lookup: dequantize one row from embedding table */
void cuda_embd_lookup_f32(const float *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_f16(const uint16_t *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_bf16(const uint16_t *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_q8_0(const cuda_q8_0_block *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_q4_0(const cuda_q4_0_block *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);
void cuda_embd_lookup_q4_k(const cuda_q4_k_block *embd_dev, float *out_dev, int token, int dim, cudaStream_t stream);

/* Decode-graph params: per-step (pos, n_pos, token) read by _g kernels
 * from device memory, so a captured graph replays without node updates. */
typedef struct {
    int pos;
    int n_pos;
    int token;
    int pad;
} cuda_decode_params;

/* ple_combine: ple[i] = (ple[i] + proj[i]) * scale */
void cuda_ple_combine(float *ple_dev, const float *proj_dev, int n, float scale,
                      cudaStream_t stream);
/* ple_norm_batch: RMSNorm n_embd slices at row*total_ple + l*n_embd */
void cuda_ple_norm_batch(float *proj_dev, const float *norm_w_dev, int n_rows,
                         int total_ple, int n_embd, int n_layers, float eps,
                         cudaStream_t stream);
/* rmsnorm_add_batch: per-row RMSNorm + residual add in one kernel.
 * out_scale is applied to the sum as an output-side gain. */
void cuda_rmsnorm_add_batch(const float *x_dev, const float *w_dev,
                            const float *residual_dev, float *y_dev, int n, float eps,
                            int m, float out_scale, cudaStream_t stream);

/* multi-column batched matmul (one x-quant + one GEMM for all columns).
 * resid_dev adds a fused residual (NULL = none). */
void cuda_matmul_multi_batch_q4_0(const void *const *w_dev, float *const *y_dev,
                                  const int *n_host, const float *x_dev, int nm, int k,
                                  int m, const float *resid_dev, cudaStream_t stream,
                                  int qmajor);
void cuda_matmul_multi_batch_q8_0(const void *const *w_dev, float *const *y_dev,
                                  const int *n_host, const float *x_dev, int nm, int k,
                                  int m, const float *resid_dev, cudaStream_t stream,
                                  int qmajor);

/* Decode-graph (_g) variants: pos/token/n_pos from device params struct.
 * Bodies mirror the base kernels exactly; only index sources differ. */
void cuda_rope_ext_g(float *vec_dev, int n_heads, int head_dim,
                     const float *cos_base, const float *sin_base,
                     const int *params_dev, int neox, cudaStream_t stream);
void cuda_attn_f16_g(const float *q_dev, const uint16_t *kc_dev,
                     const uint16_t *vc_dev, float *out_dev, int n_heads,
                     int n_kv_heads, int head_dim, size_t layer_base_bytes,
                     size_t kvh_stride, int sliding_window, float scale,
                     const int *params_dev, cudaStream_t stream);
/* Split-K variant: n_splits position chunks + combine. Tolerance-gated
 * (not bit-exact vs single-pass). */
void cuda_attn_f16_split_g(const float *q_dev, const uint16_t *kc_dev,
                           const uint16_t *vc_dev, float *out_dev, int n_heads,
                           int n_kv_heads, int head_dim, size_t layer_base_bytes,
                           size_t kvh_stride, int sliding_window, float scale,
                           int n_queries, int n_splits, const int *params_dev,
                           cudaStream_t stream);
/* GQA-grouped variant: n_queries query heads per block share K/V tiles. */
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
/* GQA-grouped variant: n_queries query heads per block share staging +
 * dequant. */
void cuda_attn_q8_gqa_g(const float *q_dev, const uint8_t *kc_dev,
                        const uint8_t *vc_dev, float *out_dev, int n_heads,
                        int n_kv_heads, int head_dim, size_t layer_base_bytes,
                        size_t kvh_stride_bytes, int sliding_window, float scale,
                        int n_blocks, int n_queries, const int *params_dev,
                        cudaStream_t stream);

/* Fused Q/K per-head RMSNorm + V RMSNorm-noweight + RoPE on Q/K
 * (decode, in-place). Replaces 5 launches with 1. */
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

#endif /* KAI_CUDA_INTERNAL_H */
