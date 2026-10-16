#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#ifdef HAVE_CUBLAS
#include <cublas_v2.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include "backend/cuda/cuda_internal.h"
using namespace nvcuda;

extern "C" {
#include "log.h"
}

static __device__ __forceinline__ void store_half(uint16_t *dst, float val) {
    reinterpret_cast<__half &>(*dst) = __float2half(val);
}

static __device__ __forceinline__ int cuda_xq8_val(float x, float d) {
    const float inv = 1.0f / d;
    const float t   = x * inv;
    const int   q   = (int)(t >= 0.0f ? floorf(t + 0.5f) : ceilf(t - 0.5f));
    return q > 127 ? 127 : (q < -127 ? -127 : q);
}

__global__ void q8_0_quant_kernel(const float *x, cuda_q8_0_block *xq, int k) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k / 32)
        return;

    const float *xb = x + 32 * b;
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float a = fabsf(xb[j]);
        if (a > amax)
            amax = a;
    }

    cuda_q8_0_block *out = &xq[b];
    if (amax < 1e-30f) {
        out->d = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            out->qs[j] = 0;
        return;
    }

    float d = amax / 127.0f;
    store_half(&out->d, d);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out->qs[j] = (int8_t)cuda_xq8_val(xb[j], d);
    }
}

__global__ void q8_0_quant_qm_kernel(const float *x, uint8_t *xq8, int k) {
    const int nb = k / 32;
    const int b = blockIdx.x;
    if (b >= nb) return;
    const int lane = threadIdx.x & 31;

    float v = x[(size_t)b * 32 + lane];
    float a = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));

    float d = a / 127.0f;
    int q = (a < 1e-30f) ? 0 : cuda_xq8_val(v, d);

    int g = b >> 5, r = b & 31;
    int Gfull = nb - g * 32, G = Gfull > 32 ? 32 : Gfull;
    uint8_t *gb = xq8 + (size_t)g * 32 * 34;
    if (lane == 0) {
        uint16_t dbits;
        store_half(&dbits, (a < 1e-30f) ? 0.0f : d);
        gb[2 * r] = (uint8_t)dbits;
        gb[2 * r + 1] = (uint8_t)(dbits >> 8);
    }

    uint8_t *qb = gb + (size_t)2 * G;
    int t = lane >> 2, j = lane & 3;
    qb[((size_t)t * G + r) * 4 + j] = (uint8_t)(int8_t)q;
}

__global__ void q4_0_quant_kernel(const float *x, cuda_q4_0_block *xq, int k) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k / 32)
        return;

    const float *xb = x + 32 * b;
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float a = fabsf(xb[j]);
        if (a > amax)
            amax = a;
    }

    cuda_q4_0_block *out = &xq[b];
    if (amax < 1e-30f) {
        out->d = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++)
            out->qs[j] = 8;
        return;
    }

    float d = amax / 7.0f;
    store_half(&out->d, d);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int q = (int)floorf(xb[j] / d + 0.5f);
        if (q > 15) q = 15;
        if (q < 0)  q = 0;
        out->qs[j] = (uint8_t)q;
    }
}

__global__ void q8_1_quant_kernel(const float *x, cuda_q8_1_block *xq, int k) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k / 32)
        return;

    const float *xb = x + 32 * b;
    float amax = 0.0f;
    int32_t sum = 0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float a = fabsf(xb[j]);
        if (a > amax)
            amax = a;
    }

    cuda_q8_1_block *out = &xq[b];
    if (amax < 1e-30f) {
        out->d = 0;
        out->s = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            out->qs[j] = 0;
        return;
    }

    float d = amax / 127.0f;
    store_half(&out->d, d);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int q = cuda_xq8_val(xb[j], d);
        out->qs[j] = (int8_t)q;
        sum += q;
    }

    store_half(&out->s, d * (float)sum);
}

__global__ void q8_k_quant_kernel(const float *x, cuda_q8_k_block *xq, int k) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k / 256)
        return;

    const float *xb = x + 256 * b;
    cuda_q8_k_block *out = &xq[b];

    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < 256; j++) {
        float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax;
            max = xb[j];
        }
    }

    if (amax < 1e-30f) {
        out->d = 0.0f;
        for (int j = 0; j < 256; j++)
            out->qs[j] = 0;
        for (int j = 0; j < 16; j++)
            out->bsums[j] = 0;
        return;
    }

    float iscale = -127.0f / max;
    out->d = 1.0f / iscale;

    int32_t sum = 0;
    int16_t bsums[16] = {0};
    for (int j = 0; j < 256; j++) {
        int q = (int)roundf(iscale * xb[j]);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out->qs[j] = (int8_t)q;
        sum += q;
        bsums[j / 16] += q;
    }

    for (int j = 0; j < 16; j++)
        out->bsums[j] = bsums[j];
}

#define MM_LANES          32
#define MM_ROWS_PER_BLOCK 8
#define MM_TOK_TILE       8
#define ATTN_TILE_SIZE    32

static inline int attn_tile_size(int head_dim) {
    if (head_dim <= 183) return 32;
    if (head_dim <= 341) return 16;
    return 8;
}
#define MMVQ_NTHREADS   (MMVQ_NWARPS * 32)
#define MMVQ_VDR        2

static __device__ __forceinline__ uint16_t qm_q8_scale(const uint8_t *wrow8, int nb, int b);
static __device__ __forceinline__ int32_t qm_q8_quad(const uint8_t *wrow8, int nb, int b, int t);
static __device__ __forceinline__ int qm_q8_val(const uint8_t *wrow8, int nb, int b, int j);

__global__ void matmul_q8_0_dp4a_kernel(const cuda_q8_0_block *__restrict__ w,
                                        const cuda_q8_0_block *__restrict__ xq,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q8_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q8_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            sumi += (int)wb->qs[j] * (int)xb->qs[j];

        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)sumi, acc);
    }
    }

    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane == 0)
        y[row] = acc;
}

__global__ void matmul_q8_0_residual_kernel(const cuda_q8_0_block *__restrict__ w,
                                            const cuda_q8_0_block *__restrict__ xq,
                                            const float *__restrict__ residual,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q8_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q8_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            sumi += (int)wb->qs[j] * (int)xb->qs[j];

        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)sumi, acc);
    }
    }

    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane == 0)
        y[row] = acc + residual[row];
}

__global__ void matmul_q4_0_dp4a_kernel(const cuda_q4_0_block *__restrict__ w,
                                        const cuda_q8_0_block *__restrict__ xq,
                                        float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q4_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q4_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            sumi0 += ((int)(wb->qs[j] & 0xF) - 8) * (int)xb->qs[j];
            sumi1 += ((int)(wb->qs[j] >> 4) - 8) * (int)xb->qs[j + 16];
        }
        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)(sumi0 + sumi1), acc);
    }
    }

    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (threadIdx.x % MM_LANES == 0)
        y[row] = acc;
}

__global__ void matmul_q4_0_residual_kernel(const cuda_q4_0_block *__restrict__ w,
                                            const cuda_q8_0_block *__restrict__ xq,
                                            const float *__restrict__ residual,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q4_0_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    {
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q4_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;
        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            sumi0 += ((int)(wb->qs[j] & 0xF) - 8) * (int)xb->qs[j];
            sumi1 += ((int)(wb->qs[j] >> 4) - 8) * (int)xb->qs[j + 16];
        }
        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)(sumi0 + sumi1), acc);
    }
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (threadIdx.x % MM_LANES == 0)
        y[row] = acc + residual[row];
}

static __device__ __forceinline__ float cuda_q81_s(float d, int sum) {
    return __half2float(__float2half_rn(d * (float)sum));
}

static __device__ __forceinline__ void dot_q4_1_block(
    const cuda_q4_1_block *wb, const cuda_q8_0_block *xb,
    int *sum_qx, int *sum_x) {
    int8_t wdeq[32];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint8_t q = wb->qs[j];
        wdeq[j]      = (int8_t)(q & 0xF);
        wdeq[j + 16] = (int8_t)(q >> 4);
    }
    int32_t qx = 0, xs = 0;
    #pragma unroll
    for (int j = 0; j < 32; j += 4) {
        uint8_t w0 = (uint8_t)wdeq[j];
        uint8_t w1 = (uint8_t)wdeq[j + 1];
        uint8_t w2 = (uint8_t)wdeq[j + 2];
        uint8_t w3 = (uint8_t)wdeq[j + 3];
        uint8_t x0 = __ldg((const uint8_t *)&xb->qs[j]);
        uint8_t x1 = __ldg((const uint8_t *)&xb->qs[j + 1]);
        uint8_t x2 = __ldg((const uint8_t *)&xb->qs[j + 2]);
        uint8_t x3 = __ldg((const uint8_t *)&xb->qs[j + 3]);
        int32_t a = (int32_t)w0 | ((int32_t)w1 << 8) |
                    ((int32_t)w2 << 16) | ((int32_t)w3 << 24);
        int32_t bval = (int32_t)x0 | ((int32_t)x1 << 8) |
                       ((int32_t)x2 << 16) | ((int32_t)x3 << 24);
        qx = __dp4a(a, bval, qx);
        xs = __dp4a(bval, 0x01010101, xs);
    }
    *sum_qx = qx;
    *sum_x = xs;
}

__global__ void matmul_q4_1_dp4a_kernel(const cuda_q4_1_block *__restrict__ w,
                                        const cuda_q8_0_block *__restrict__ xq,
                                        float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;

    extern __shared__ char smem_gemv[];
    cuda_q8_0_block *xq_s = reinterpret_cast<cuda_q8_0_block *>(smem_gemv);
    for (int b = threadIdx.x; b < nb; b += blockDim.x)
        xq_s[b] = xq[b];
    __syncthreads();

    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q4_1_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q4_1_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq_s + b;
        int8_t wdeq[32];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t q = wb->qs[j];
            wdeq[j]      = (int8_t)(q & 0xF);
            wdeq[j + 16] = (int8_t)(q >> 4);
        }
        int32_t qx = 0, xs = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            uint8_t w0 = (uint8_t)wdeq[j];
            uint8_t w1 = (uint8_t)wdeq[j + 1];
            uint8_t w2 = (uint8_t)wdeq[j + 2];
            uint8_t w3 = (uint8_t)wdeq[j + 3];
            uint8_t x0 = (uint8_t)xb->qs[j];
            uint8_t x1 = (uint8_t)xb->qs[j + 1];
            uint8_t x2 = (uint8_t)xb->qs[j + 2];
            uint8_t x3 = (uint8_t)xb->qs[j + 3];
            int32_t a = (int32_t)w0 | ((int32_t)w1 << 8) |
                        ((int32_t)w2 << 16) | ((int32_t)w3 << 24);
            int32_t bval = (int32_t)x0 | ((int32_t)x1 << 8) |
                           ((int32_t)x2 << 16) | ((int32_t)x3 << 24);
            qx = __dp4a(a, bval, qx);
            xs = __dp4a(bval, 0x01010101, xs);
        }
        float dw = __half2float(__ushort_as_half(wb->d));
        float mw = __half2float(__ushort_as_half(wb->m));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)qx, mw * cuda_q81_s(dx, xs) + acc);
    }

    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

if (threadIdx.x % MM_LANES == 0)
        y[row] = acc;
}

__global__ void matmul_q4_1_residual_kernel(const cuda_q4_1_block *__restrict__ w,
                                            const cuda_q8_0_block *__restrict__ xq,
                                            const float *__restrict__ residual,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q4_1_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q4_1_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;
        int sum_qx = 0, sum_x = 0;
        dot_q4_1_block(wb, xb, &sum_qx, &sum_x);
        float dw = __half2float(__ushort_as_half(wb->d));
        float mw = __half2float(__ushort_as_half(wb->m));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)sum_qx, mw * cuda_q81_s(dx, sum_x) + acc);
    }

    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

if (threadIdx.x % MM_LANES == 0)
        y[row] = acc + residual[row];
}

static __device__ __forceinline__ int32_t q8_block_quad(const cuda_q8_0_block *wb, int t) {
    const uint8_t *pb = (const uint8_t *)wb;
    int j = t << 2;
    uint16_t hw0 = __ldg((const uint16_t *)(pb + 2 + j));
    uint16_t hw1 = __ldg((const uint16_t *)(pb + 2 + j + 2));
    return (int32_t)(hw0 & 0xFF) | ((int32_t)(hw0 >> 8) << 8) |
           ((int32_t)(hw1 & 0xFF) << 16) | ((int32_t)(hw1 >> 8) << 24);
}

static __device__ __forceinline__ int32_t q8_block_quad_wide(const cuda_q8_0_block *wb, int t) {
    return (int32_t)__ldg((const uint32_t *)((const uint8_t *)wb + 2 + (t << 2)));
}

static __device__ __forceinline__ uint16_t qm_q8_scale(const uint8_t *wrow8, int nb, int b) {
    int g = b >> 5, r = b & 31;
    int Gfull = nb - g * 32, G = Gfull > 32 ? 32 : Gfull;
    (void)G;
    return __ldg((const uint16_t *)(wrow8 + (size_t)g * 32 * 34 + (size_t)2 * r));
}

static __device__ __forceinline__ int32_t qm_q8_quad(const uint8_t *wrow8, int nb, int b, int t) {
    int g = b >> 5, r = b & 31;
    int Gfull = nb - g * 32, G = Gfull > 32 ? 32 : Gfull;
    const uint8_t *qb = wrow8 + (size_t)g * 32 * 34 + (size_t)2 * G + ((size_t)t * G + r) * 4;
    uint16_t hw0 = __ldg((const uint16_t *)qb);
    uint16_t hw1 = __ldg((const uint16_t *)(qb + 2));
    return (int32_t)(hw0 & 0xFF) | ((int32_t)(hw0 >> 8) << 8) |
           ((int32_t)(hw1 & 0xFF) << 16) | ((int32_t)(hw1 >> 8) << 24);
}

static __device__ __forceinline__ int qm_q8_val(const uint8_t *wrow8, int nb, int b, int j) {
    int32_t q = qm_q8_quad(wrow8, nb, b, j >> 2);
    return (int)(int8_t)(q >> ((j & 3) << 3));
}

static __device__ __forceinline__ int32_t qm_q8_quad_wide(const uint8_t *wrow8, int nb, int b,
                                                          int t) {
    int g = b >> 5, r = b & 31;
    int Gfull = nb - g * 32, G = Gfull > 32 ? 32 : Gfull;
    const uint8_t *qb = wrow8 + (size_t)g * 32 * 34 + (size_t)2 * G + ((size_t)t * G + r) * 4;
    return (int32_t)__ldg((const uint32_t *)qb);
}

__global__ void matmul_q8_0_dp4a_v2_kernel(const cuda_q8_0_block *__restrict__ w,
                                        const cuda_q8_0_block *__restrict__ xq,
                                        float *__restrict__ y, int n, int k,
                                        int use_stage, int xq_qm) {
    const int nb = k >> 5;

    extern __shared__ char smem_gemv[];
    cuda_q8_0_block *xq_s = reinterpret_cast<cuda_q8_0_block *>(smem_gemv);
    if (use_stage) {
        for (int b = threadIdx.x; b < nb; b += blockDim.x)
            xq_s[b] = xq[b];
    }
    __syncthreads();

    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q8_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {

    const uint8_t *xq8o = (const uint8_t *)xq;
    int xqm_o = (xq_qm && !use_stage);
    int xwide_o = (xqm_o && ((nb & 1) == 0));
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q8_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = use_stage ? xq_s + b : xq + b;
        uint16_t dwb = __ldg(&wb->d);

        int32_t dp_acc = 0;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int j = t << 2;
            int32_t a = q8_block_quad(wb, t);
            int32_t bval;
            if (xwide_o) {
                bval = qm_q8_quad_wide(xq8o, nb, b, t);
            } else if (xqm_o) {
                bval = qm_q8_quad(xq8o, nb, b, t);
            } else if (use_stage) {
                uint8_t b0 = xb->qs[j];
                uint8_t b1 = xb->qs[j + 1];
                uint8_t b2 = xb->qs[j + 2];
                uint8_t b3 = xb->qs[j + 3];
                bval = (int32_t)b0 | ((int32_t)b1 << 8) |
                       ((int32_t)b2 << 16) | ((int32_t)b3 << 24);
            } else {
                bval = q8_block_quad(xb, t);
            }
            dp_acc = __dp4a(a, bval, dp_acc);
        }
        float dw = __half2float(__ushort_as_half(dwb));
        uint16_t dxb = xqm_o ? qm_q8_scale(xq8o, nb, b) : xb->d;
        float dx = __half2float(__ushort_as_half(dxb));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }

    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane == 0)
        y[row] = acc;
}

__global__ void matmul_q8_0_residual_v2_kernel(const cuda_q8_0_block *__restrict__ w,
                                           const cuda_q8_0_block *__restrict__ xq,
                                           const float *__restrict__ residual,
                                           float *__restrict__ y, int n, int k, int xq_qm) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q8_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;

    const uint8_t *xq8r = (const uint8_t *)xq;
    int xr_wide = (xq_qm && ((nb & 1) == 0));
    int xr_qm = xq_qm;
    {
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q8_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int32_t dp_acc = 0;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int32_t a = q8_block_quad(wb, t);
            int32_t bval = xr_qm ? (xr_wide ? qm_q8_quad_wide(xq8r, nb, b, t)
                                            : qm_q8_quad(xq8r, nb, b, t))
                                 : q8_block_quad(xb, t);
            dp_acc = __dp4a(a, bval, dp_acc);
        }

        uint16_t dwb = __ldg(&wb->d);
        uint16_t dxb = xr_qm ? qm_q8_scale(xq8r, nb, b) : __ldg(&xb->d);
        float dw = __half2float(__ushort_as_half(dwb));
        float dx = __half2float(__ushort_as_half(dxb));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }

    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane == 0)
        y[row] = acc + residual[row];
}

__global__ void matmul_q4_0_dp4a_v2_kernel(const cuda_q4_0_block *__restrict__ w,
                                        const cuda_q8_0_block *__restrict__ xq,
                                        float *__restrict__ y, int n, int k, int use_stage) {
    const int nb = k >> 5;

    extern __shared__ char smem_gemv[];
    cuda_q8_0_block *xq_s = reinterpret_cast<cuda_q8_0_block *>(smem_gemv);
    if (use_stage) {
        for (int b = threadIdx.x; b < nb; b += blockDim.x)
            xq_s[b] = xq[b];
    }
    __syncthreads();

    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q4_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q4_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = use_stage ? xq_s + b : xq + b;

        int32_t raw = 0, xs = 0;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int jt = t << 2;
            uint8_t w0 = wb->qs[jt];
            uint8_t w1 = wb->qs[jt + 1];
            uint8_t w2 = wb->qs[jt + 2];
            uint8_t w3 = wb->qs[jt + 3];
            uint8_t x0 = xb->qs[jt];
            uint8_t x1 = xb->qs[jt + 1];
            uint8_t x2 = xb->qs[jt + 2];
            uint8_t x3 = xb->qs[jt + 3];
            int32_t alo = (int32_t)(w0 & 0xF) | ((int32_t)(w1 & 0xF) << 8) |
                          ((int32_t)(w2 & 0xF) << 16) | ((int32_t)(w3 & 0xF) << 24);
            int32_t vlo = (int32_t)x0 | ((int32_t)x1 << 8) |
                          ((int32_t)x2 << 16) | ((int32_t)x3 << 24);
            raw = __dp4a(alo, vlo, raw);
            xs = __dp4a(vlo, 0x01010101, xs);
            uint8_t c0 = xb->qs[jt + 16];
            uint8_t c1 = xb->qs[jt + 17];
            uint8_t c2 = xb->qs[jt + 18];
            uint8_t c3 = xb->qs[jt + 19];
            int32_t ahi = (int32_t)(w0 >> 4) | ((int32_t)(w1 >> 4) << 8) |
                          ((int32_t)(w2 >> 4) << 16) | ((int32_t)(w3 >> 4) << 24);
            int32_t vhi = (int32_t)c0 | ((int32_t)c1 << 8) |
                          ((int32_t)c2 << 16) | ((int32_t)c3 << 24);
            raw = __dp4a(ahi, vhi, raw);
            xs = __dp4a(vhi, 0x01010101, xs);
        }
        int32_t dp_acc = raw - 8 * xs;

        uint16_t dwb = __ldg(&wb->d);
        float dw = __half2float(__ushort_as_half(dwb));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }

    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (threadIdx.x % MM_LANES == 0)
        y[row] = acc;
}

__global__ void matmul_q4_0_residual_v2_kernel(const cuda_q4_0_block *__restrict__ w,
                                           const cuda_q8_0_block *__restrict__ xq,
                                           const float *__restrict__ residual,
                                           float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q4_0_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    {
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q4_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int8_t wdeq[32];
        #pragma unroll
        for (int h2 = 0; h2 < 8; h2++) {
            uint16_t hv = __ldg((const uint16_t *)((const uint8_t *)wb + 2 + (h2 << 1)));
            uint8_t q0 = (uint8_t)hv;
            uint8_t q1 = (uint8_t)(hv >> 8);
            wdeq[h2 << 1] = (int8_t)((q0 & 0xF) - 8);
            wdeq[(h2 << 1) + 1] = (int8_t)((q1 & 0xF) - 8);
            wdeq[16 + (h2 << 1)] = (int8_t)((q0 >> 4) - 8);
            wdeq[16 + (h2 << 1) + 1] = (int8_t)((q1 >> 4) - 8);
        }

        const int32_t *aw = reinterpret_cast<const int32_t *>(wdeq);
        int32_t dp_acc = 0;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int jt = t << 2;
            int32_t a = aw[t];
            uint8_t b0 = xb->qs[jt];
            uint8_t b1 = xb->qs[jt + 1];
            uint8_t b2 = xb->qs[jt + 2];
            uint8_t b3 = xb->qs[jt + 3];
            int32_t bval = (int32_t)b0 | ((int32_t)b1 << 8) |
                           ((int32_t)b2 << 16) | ((int32_t)b3 << 24);
            dp_acc = __dp4a(a, bval, dp_acc);
        }

        uint16_t dwb = __ldg(&wb->d);
        uint16_t dxb = __ldg(&xb->d);
        float dw = __half2float(__ushort_as_half(dwb));
        float dx = __half2float(__ushort_as_half(dxb));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (threadIdx.x % MM_LANES == 0)
        y[row] = acc + residual[row];
}

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WMMA_WARPS_PER_BLOCK 4
#define WMMA_THREADS_PER_BLOCK (WMMA_WARPS_PER_BLOCK * 32)
#define WMMA_ROWS_PER_BLOCK (WMMA_M * 2)
#define WMMA_COLS_PER_BLOCK (WMMA_N * 2)

#define WMMA_K_ELEMS 32

__global__ void matmul_q8_0_wmma_batch_kernel(const cuda_q8_0_block *__restrict__ w,
                                               const cuda_q8_0_block *__restrict__ xq,
                                               float *__restrict__ y, int n, int k, int m) {
    const int nb = k >> 5;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int row_tile = blockIdx.y * WMMA_ROWS_PER_BLOCK;
    const int col_tile = blockIdx.x * WMMA_COLS_PER_BLOCK;

    if (row_tile >= n || col_tile >= m) return;

    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int row_start = row_tile + warp_row * WMMA_M;
    const int col_start = col_tile + warp_col * WMMA_N;

    const bool warp_active = (row_start < n) && (col_start < m);

    const int rows_this_warp = warp_active ? min(WMMA_M, n - row_start) : 0;
    const int cols_this_warp = warp_active ? min(WMMA_N, m - col_start) : 0;

    extern __shared__ char smem_wmma[];
    __half *w_smem = reinterpret_cast<__half *>(smem_wmma);
    __half *x_smem = &w_smem[(size_t)WMMA_ROWS_PER_BLOCK * WMMA_K_ELEMS];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int kb = 0; kb < nb; kb++) {

        for (int i = threadIdx.x; i < WMMA_ROWS_PER_BLOCK * WMMA_K_ELEMS; i += WMMA_THREADS_PER_BLOCK) {
            int r = i / WMMA_K_ELEMS;
            int j = i % WMMA_K_ELEMS;
            int row = row_tile + r;
            if (row < n) {
                float scale, val;
                {
                    const cuda_q8_0_block *wb = w + (size_t)row * nb + kb;
                    scale = __half2float(__ushort_as_half(wb->d));
                    val = (int)wb->qs[j] * scale;
                }
                w_smem[(size_t)r * WMMA_K_ELEMS + j] = __float2half(val);
            } else {
                w_smem[(size_t)r * WMMA_K_ELEMS + j] = __float2half(0.0f);
            }
        }

        for (int i = threadIdx.x; i < WMMA_K_ELEMS * WMMA_COLS_PER_BLOCK; i += WMMA_THREADS_PER_BLOCK) {
            int kk_elem = i / WMMA_COLS_PER_BLOCK;
            int c = i % WMMA_COLS_PER_BLOCK;
            int col = col_tile + c;
            if (col < m) {
                const cuda_q8_0_block *xb = xq + (size_t)col * nb + kb;
                float scale = __half2float(__ushort_as_half(xb->d));
                float val = (int)xb->qs[kk_elem] * scale;
                x_smem[(size_t)kk_elem * WMMA_COLS_PER_BLOCK + c] = __float2half(val);
            } else {
                x_smem[(size_t)kk_elem * WMMA_COLS_PER_BLOCK + c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int kk = 0; kk < WMMA_K_ELEMS; kk += WMMA_K) {

            wmma::load_matrix_sync(a_frag,
                &w_smem[(size_t)warp_row * WMMA_M * WMMA_K_ELEMS + kk], WMMA_K_ELEMS);

            wmma::load_matrix_sync(b_frag,
                &x_smem[(size_t)kk * WMMA_COLS_PER_BLOCK + warp_col * WMMA_N], WMMA_COLS_PER_BLOCK);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    __shared__ float smem_results[WMMA_WARPS_PER_BLOCK][WMMA_M][WMMA_N];
    wmma::store_matrix_sync(&smem_results[warp_id][0][0], acc_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    for (int i = 0; i < rows_this_warp; i++) {
        for (int j = 0; j < cols_this_warp; j++) {
            if (lane == 0) {
                y[(size_t)(col_start + j) * n + (row_start + i)] = smem_results[warp_id][i][j];
            }
        }
    }
}

__global__ void matmul_q4_0_wmma_batch_kernel(const cuda_q4_0_block *__restrict__ w,
                                               const cuda_q8_0_block *__restrict__ xq,
                                               float *__restrict__ y, int n, int k, int m) {
    const int nb = k >> 5;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int row_tile = blockIdx.y * WMMA_ROWS_PER_BLOCK;
    const int col_tile = blockIdx.x * WMMA_COLS_PER_BLOCK;

    if (row_tile >= n || col_tile >= m) return;

    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int row_start = row_tile + warp_row * WMMA_M;
    const int col_start = col_tile + warp_col * WMMA_N;

    const bool warp_active = (row_start < n) && (col_start < m);

    const int rows_this_warp = warp_active ? min(WMMA_M, n - row_start) : 0;
    const int cols_this_warp = warp_active ? min(WMMA_N, m - col_start) : 0;

    extern __shared__ char smem_wmma[];
    __half *w_smem = reinterpret_cast<__half *>(smem_wmma);
    __half *x_smem = &w_smem[(size_t)WMMA_ROWS_PER_BLOCK * WMMA_K_ELEMS];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int kb = 0; kb < nb; kb++) {

        for (int i = threadIdx.x; i < WMMA_ROWS_PER_BLOCK * WMMA_K_ELEMS; i += WMMA_THREADS_PER_BLOCK) {
            int r = i / WMMA_K_ELEMS;
            int j = i % WMMA_K_ELEMS;
            int row = row_tile + r;
            if (row < n) {
                float scale;
                int q;
                {
                    const cuda_q4_0_block *wb = w + (size_t)row * nb + kb;
                    scale = __half2float(__ushort_as_half(wb->d));
                    q = (j < 16) ? ((wb->qs[j] & 0xF) - 8) : ((wb->qs[j - 16] >> 4) - 8);
                }
                w_smem[(size_t)r * WMMA_K_ELEMS + j] = __float2half(q * scale);
            } else {
                w_smem[(size_t)r * WMMA_K_ELEMS + j] = __float2half(0.0f);
            }
        }

        for (int i = threadIdx.x; i < WMMA_K_ELEMS * WMMA_COLS_PER_BLOCK; i += WMMA_THREADS_PER_BLOCK) {
            int kk_elem = i / WMMA_COLS_PER_BLOCK;
            int c = i % WMMA_COLS_PER_BLOCK;
            int col = col_tile + c;
            if (col < m) {
                const cuda_q8_0_block *xb = xq + (size_t)col * nb + kb;
                float scale = __half2float(__ushort_as_half(xb->d));
                float val = (int)xb->qs[kk_elem] * scale;
                x_smem[(size_t)kk_elem * WMMA_COLS_PER_BLOCK + c] = __float2half(val);
            } else {
                x_smem[(size_t)kk_elem * WMMA_COLS_PER_BLOCK + c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int kk = 0; kk < WMMA_K_ELEMS; kk += WMMA_K) {
            wmma::load_matrix_sync(a_frag,
                &w_smem[(size_t)warp_row * WMMA_M * WMMA_K_ELEMS + kk], WMMA_K_ELEMS);
            wmma::load_matrix_sync(b_frag,
                &x_smem[(size_t)kk * WMMA_COLS_PER_BLOCK + warp_col * WMMA_N], WMMA_COLS_PER_BLOCK);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    __shared__ float smem_results[WMMA_WARPS_PER_BLOCK][WMMA_M][WMMA_N];
    wmma::store_matrix_sync(&smem_results[warp_id][0][0], acc_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    for (int i = 0; i < rows_this_warp; i++) {
        for (int j = 0; j < cols_this_warp; j++) {
            if (lane == 0) {
                y[(size_t)(col_start + j) * n + (row_start + i)] = smem_results[warp_id][i][j];
            }
        }
    }
}

__global__ void fp16_copy_batch_kernel(const float *__restrict__ x, __half *__restrict__ xf,
                                       int k, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * k;
    if (i >= total) return;
    xf[i] = __float2half(x[i]);
}

__global__ void dequant_q8_0_to_fp16_kernel(const void *__restrict__ w_in, __half *__restrict__ out,
                                            int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;
    if (i >= total) return;
    int row = i / k, col = i % k;
    int blk = col >> 5, j = col & 31;
    float scale, v;
    {
        const cuda_q8_0_block *wb =
            (const cuda_q8_0_block *)w_in + (size_t)row * (k >> 5) + blk;
        scale = __half2float(__ushort_as_half(__ldg(&wb->d)));
        v = (float)wb->qs[j];
    }
    out[i] = __float2half(v * scale);
}

__global__ void dequant_q4_0_to_fp16_kernel(const void *__restrict__ w_in, __half *__restrict__ out,
                                            int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;
    if (i >= total) return;
    int row = i / k, col = i % k;
    int blk = col >> 5, j = col & 31;
    float scale;
    int q;
    {
        const cuda_q4_0_block *wb =
            (const cuda_q4_0_block *)w_in + (size_t)row * (k >> 5) + blk;
        scale = __half2float(__ushort_as_half(__ldg(&wb->d)));
        uint8_t qb = wb->qs[(j < 16) ? j : j - 16];
        q = ((j < 16) ? (qb & 0xF) : (qb >> 4)) - 8;
    }
    out[i] = __float2half((float)q * scale);
}

#define F16_TILE_M 64
#define F16_TILE_N 64
#define F16_WARPS 8
#define F16_THREADS (F16_WARPS * 32)

#define F16_TILE_WARPS 4
#define F16_TILE_THREADS (F16_TILE_WARPS * 32)

__global__ void matmul_fp16_wmma64_batch_kernel(const __half *__restrict__ w,
                                               const __half *__restrict__ xf,
                                               float *__restrict__ y, int n, int k, int m) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int row_tile = blockIdx.y * F16_TILE_M;
    const int col_tile = blockIdx.x * F16_TILE_N;

    if (row_tile >= n || col_tile >= m) return;

    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int row_start = row_tile + warp_row * 2 * WMMA_M;
    const int col_start = col_tile + warp_col * (2 * WMMA_N);

    const int warp_valid = (row_start < n && col_start < m &&
                            row_start < row_tile + F16_TILE_M);

    extern __shared__ char smem_f16b[];
    __half *w_smem = reinterpret_cast<__half *>(smem_f16b);
    __half *x_smem = &w_smem[(size_t)F16_TILE_M * WMMA_K_ELEMS];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b0, b1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc00, acc01, acc10, acc11;

    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    for (int kb = 0; kb < k; kb += WMMA_K_ELEMS) {
        for (int i = threadIdx.x; i < F16_TILE_M * WMMA_K_ELEMS; i += F16_TILE_THREADS) {
            int r = i / WMMA_K_ELEMS;
            int j = i % WMMA_K_ELEMS;
            int row = row_tile + r;
            w_smem[(size_t)r * WMMA_K_ELEMS + j] =
                (row < n && kb + j < k) ? w[(size_t)row * k + kb + j] : __float2half(0.0f);
        }

        for (int i = threadIdx.x; i < WMMA_K_ELEMS * F16_TILE_N; i += F16_TILE_THREADS) {
            int kk_elem = i / F16_TILE_N;
            int c = i % F16_TILE_N;
            int col = col_tile + c;
            x_smem[(size_t)kk_elem * F16_TILE_N + c] =
                (col < m && kb + kk_elem < k) ? xf[(size_t)col * k + kb + kk_elem]
                                              : __float2half(0.0f);
        }
        __syncthreads();

        for (int kk = 0; kk < WMMA_K_ELEMS; kk += WMMA_K) {
            wmma::load_matrix_sync(a0,
                &w_smem[(size_t)warp_row * 2 * WMMA_M * WMMA_K_ELEMS + kk], WMMA_K_ELEMS);
            wmma::load_matrix_sync(a1,
                &w_smem[(size_t)(warp_row * 2 * WMMA_M + WMMA_M) * WMMA_K_ELEMS + kk],
                WMMA_K_ELEMS);
            wmma::load_matrix_sync(b0,
                &x_smem[(size_t)kk * F16_TILE_N + warp_col * 2 * WMMA_N], F16_TILE_N);
            wmma::load_matrix_sync(b1,
                &x_smem[(size_t)kk * F16_TILE_N + (warp_col * 2 + 1) * WMMA_N],
                F16_TILE_N);
            wmma::mma_sync(acc00, a0, b0, acc00);
            wmma::mma_sync(acc01, a0, b1, acc01);
            wmma::mma_sync(acc10, a1, b0, acc10);
            wmma::mma_sync(acc11, a1, b1, acc11);
        }
        __syncthreads();
    }

    __shared__ float smem_res64[F16_TILE_WARPS][2 * WMMA_M][2 * WMMA_N];

    wmma::store_matrix_sync(&smem_res64[warp_id][0][0], acc00, 2 * WMMA_N,
                            wmma::mem_row_major);
    wmma::store_matrix_sync(&smem_res64[warp_id][0][WMMA_N], acc01, 2 * WMMA_N,
                            wmma::mem_row_major);
    wmma::store_matrix_sync(&smem_res64[warp_id][WMMA_M][0], acc10, 2 * WMMA_N,
                            wmma::mem_row_major);
    wmma::store_matrix_sync(&smem_res64[warp_id][WMMA_M][WMMA_N], acc11, 2 * WMMA_N,
                            wmma::mem_row_major);
    __syncthreads();

    for (int i = lane; i < 2 * WMMA_M * 2 * WMMA_N; i += MM_LANES) {
        int r = i / (2 * WMMA_N);
        int c = i % (2 * WMMA_N);
        int row = row_start + r;
        int col = col_start + c;
        if (warp_valid && row < n && col < m)
            y[(size_t)col * n + row] = smem_res64[warp_id][r][c];
    }
}

static __device__ __forceinline__ void get_scale_min_k4(int idx, const uint8_t *scales, uint8_t *scu8, uint8_t *mu8) {
    if (idx < 4) {
        *scu8 = scales[idx] & 63;
        *mu8  = scales[idx + 4] & 63;
    } else {
        *scu8 = (scales[idx + 4] & 0xF) | ((scales[idx - 4] >> 6) << 4);
        *mu8  = (scales[idx + 4] >> 4) | ((scales[idx - 0] >> 6) << 4);
    }
}

static __device__ __forceinline__ float q4_k_dot(const cuda_q4_k_block *blk, const int8_t *xq8, const int16_t *bs, float xd, float acc) {
    float d     = __half2float(__ushort_as_half(blk->d));
    float dmin  = __half2float(__ushort_as_half(blk->dmin));

    if (!isfinite(d) || d <= 0.0f) d = 1.0f;
    if (!isfinite(dmin)) dmin = 0.0f;

    if (!isfinite(xd)) xd = 1.0f;
    const uint8_t *qbytes = blk->qs;
    const uint8_t *sc   = blk->scales;
    int32_t sumi = 0;
    int32_t summ = 0;
    int ib = 0;

    uint32_t scw0 = __ldg((const uint32_t *)(sc + 0));
    uint32_t scw1 = __ldg((const uint32_t *)(sc + 4));
    uint32_t scw2 = __ldg((const uint32_t *)(sc + 8));
    uint8_t b0 = (uint8_t)scw0, b1 = (uint8_t)(scw0 >> 8);
    uint8_t b2 = (uint8_t)(scw0 >> 16), b3 = (uint8_t)(scw0 >> 24);
    uint8_t b4 = (uint8_t)scw1, b5 = (uint8_t)(scw1 >> 8);
    uint8_t b6 = (uint8_t)(scw1 >> 16), b7 = (uint8_t)(scw1 >> 24);
    uint8_t b8 = (uint8_t)scw2, b9 = (uint8_t)(scw2 >> 8);
    uint8_t b10 = (uint8_t)(scw2 >> 16), b11 = (uint8_t)(scw2 >> 24);
    int s01[4] = {(int)(b0 & 63), (int)(b1 & 63), (int)(b2 & 63), (int)(b3 & 63)};
    int m01[4] = {(int)(b4 & 63), (int)(b5 & 63), (int)(b6 & 63), (int)(b7 & 63)};
    int s23[4] = {(int)((b8 & 0xF) | ((b0 >> 6) << 4)), (int)((b9 & 0xF) | ((b1 >> 6) << 4)),
                  (int)((b10 & 0xF) | ((b2 >> 6) << 4)), (int)((b11 & 0xF) | ((b3 >> 6) << 4))};
    int m23[4] = {(int)((b8 >> 4) | ((b4 >> 6) << 4)), (int)((b9 >> 4) | ((b5 >> 6) << 4)),
                  (int)((b10 >> 4) | ((b6 >> 6) << 4)), (int)((b11 >> 4) | ((b7 >> 6) << 4))};
    for (int g = 0; g < 4; g++) {

        int pi = (g < 2) ? 2 * g : 2 * g - 4;
        int s0 = (g < 2) ? s01[pi] : s23[pi];
        int m0 = (g < 2) ? m01[pi] : m23[pi];
        int s1 = (g < 2) ? s01[pi + 1] : s23[pi + 1];
        int m1 = (g < 2) ? m01[pi + 1] : m23[pi + 1];
        const uint8_t *qg  = qbytes + (ptrdiff_t)(g * 32);
        const int8_t  *xq0 = xq8 + (ptrdiff_t)(g * 64);
        const int8_t  *xq1 = xq8 + (ptrdiff_t)(g * 64) + 32;

        int32_t dot0 = 0;
        int32_t dot1 = 0;

        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int jt = t << 2;
            uint32_t qw = __ldg((const uint32_t *)(qg + jt));
            uint32_t xw0 = __ldg((const uint32_t *)(xq0 + jt));
            uint32_t xw1 = __ldg((const uint32_t *)(xq1 + jt));
            int32_t wa = (int32_t)(qw & 0x0F0F0F0Fu);
            dot0 = __dp4a(wa, (int32_t)xw0, dot0);
            int32_t ha = (int32_t)((qw >> 4) & 0x0F0F0F0Fu);
            dot1 = __dp4a(ha, (int32_t)xw1, dot1);
        }
        sumi += (s0 * dot0) + (s1 * dot1);
        summ += m0 * (int)(bs[ib] + bs[ib + 1]);
        ib += 2;
        summ += m1 * (int)(bs[ib] + bs[ib + 1]);
        ib += 2;
    }
    float result = fmaf(xd, (d * (float)sumi) - (dmin * (float)summ), acc);
    return result;
}

__global__ void matmul_q4_k_kernel(const cuda_q4_k_block *__restrict__ w,
                                    const cuda_q8_k_block *__restrict__ xq,
                                    float *__restrict__ y, int n, int k) {
    const int nb = k / 256;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const int gr = lane >> 3;
    const int lane8 = lane & 7;
    const unsigned mask8 = 0xFFu << (gr << 3);
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK * 4 + warp * 4 + gr;

    float acc = 0.0f;
    const cuda_q4_k_block *wrow = NULL;
    if (row < n) {
        wrow = w + (size_t)row * nb;
        for (int b = lane8; b < nb; b += 8) {
            const cuda_q4_k_block *wb = wrow + b;
            const cuda_q8_k_block *xb = xq + b;
            float xd = xb->d;
            acc = q4_k_dot(wb, xb->qs, xb->bsums, xd, acc);
        }
    }

    #pragma unroll
    for (int off = 4; off > 0; off >>= 1)
        acc += __shfl_xor_sync(mask8, acc, off);

    if (row < n && lane8 == 0)
        y[row] = acc;
}

#define MMVQ_NWARPS     4
#define MMVQ_NTHREADS   (MMVQ_NWARPS * 32)
#define MMVQ_VDR        2

__global__ void matmul_q8_0_mmvq_kernel(const cuda_q8_0_block *__restrict__ w,
                                         const cuda_q8_0_block *__restrict__ xq,
                                         float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const cuda_q8_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {
    for (int b = tid; b < nb; b += MMVQ_NTHREADS) {
        const cuda_q8_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int32_t dp_acc = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            int32_t a = (int32_t)((uint8_t)wb->qs[j] | ((uint8_t)wb->qs[j+1] << 8) |
                                  ((uint8_t)wb->qs[j+2] << 16) | ((uint8_t)wb->qs[j+3] << 24));
            int32_t bval = (int32_t)((uint8_t)xb->qs[j] | ((uint8_t)xb->qs[j+1] << 8) |
                                     ((uint8_t)xb->qs[j+2] << 16) | ((uint8_t)xb->qs[j+3] << 24));
            dp_acc = __dp4a(a, bval, dp_acc);
        }

        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }

    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    __shared__ float smem[MMVQ_NWARPS];

    if (lane == 0)
        smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < MMVQ_NWARPS) ? smem[lane] : 0.0f;
        for (int off = MMVQ_NWARPS / 2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0)
            y[row] = val;
    }
}

__global__ void matmul_q8_0_residual_mmvq_kernel(const cuda_q8_0_block *__restrict__ w,
                                                const cuda_q8_0_block *__restrict__ xq,
                                                const float *__restrict__ residual,
                                                float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const cuda_q8_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {
    for (int b = tid; b < nb; b += MMVQ_NTHREADS) {
        const cuda_q8_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int32_t dp_acc = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            int32_t a = (int32_t)((uint8_t)wb->qs[j] | ((uint8_t)wb->qs[j+1] << 8) |
                                  ((uint8_t)wb->qs[j+2] << 16) | ((uint8_t)wb->qs[j+3] << 24));
            int32_t bval = (int32_t)((uint8_t)xb->qs[j] | ((uint8_t)xb->qs[j+1] << 8) |
                                     ((uint8_t)xb->qs[j+2] << 16) | ((uint8_t)xb->qs[j+3] << 24));
            dp_acc = __dp4a(a, bval, dp_acc);
        }

        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }

    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    __shared__ float smem[MMVQ_NWARPS];

    if (lane == 0)
        smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < MMVQ_NWARPS) ? smem[lane] : 0.0f;
        for (int off = MMVQ_NWARPS / 2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0)
            y[row] = val + residual[row];
    }
}

__global__ void matmul_q4_0_mmvq_kernel(const cuda_q4_0_block *__restrict__ w,
                                         const cuda_q8_0_block *__restrict__ xq,
                                         float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const cuda_q4_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {
    for (int b = tid; b < nb; b += MMVQ_NTHREADS) {
        const cuda_q4_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int8_t wdeq[32];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            wdeq[j]      = (int8_t)((wb->qs[j] & 0xF) - 8);
            wdeq[j + 16] = (int8_t)((wb->qs[j] >> 4) - 8);
        }

        int32_t dp_acc = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            int32_t a = (int32_t)((uint8_t)wdeq[j] | ((uint8_t)wdeq[j+1] << 8) |
                                  ((uint8_t)wdeq[j+2] << 16) | ((uint8_t)wdeq[j+3] << 24));
            int32_t bval = (int32_t)((uint8_t)xb->qs[j] | ((uint8_t)xb->qs[j+1] << 8) |
                                     ((uint8_t)xb->qs[j+2] << 16) | ((uint8_t)xb->qs[j+3] << 24));
            dp_acc = __dp4a(a, bval, dp_acc);
        }

        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    __shared__ float smem[MMVQ_NWARPS];

    if (threadIdx.x % 32 == 0)
        smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < MMVQ_NWARPS) ? smem[lane] : 0.0f;
        for (int off = MMVQ_NWARPS / 2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0)
            y[row] = val;
    }
}

__global__ void matmul_q4_0_residual_mmvq_kernel(const cuda_q4_0_block *__restrict__ w,
                                                const cuda_q8_0_block *__restrict__ xq,
                                                const float *__restrict__ residual,
                                                float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const cuda_q4_0_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {
    for (int b = tid; b < nb; b += MMVQ_NTHREADS) {
        const cuda_q4_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int8_t wdeq[32];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            wdeq[j]      = (int8_t)((wb->qs[j] & 0xF) - 8);
            wdeq[j + 16] = (int8_t)((wb->qs[j] >> 4) - 8);
        }

        int32_t dp_acc = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            int32_t a = (int32_t)((uint8_t)wdeq[j] | ((uint8_t)wdeq[j+1] << 8) |
                                  ((uint8_t)wdeq[j+2] << 16) | ((uint8_t)wdeq[j+3] << 24));
            int32_t bval = (int32_t)((uint8_t)xb->qs[j] | ((uint8_t)xb->qs[j+1] << 8) |
                                     ((uint8_t)xb->qs[j+2] << 16) | ((uint8_t)xb->qs[j+3] << 24));
            dp_acc = __dp4a(a, bval, dp_acc);
        }

        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    __shared__ float smem[MMVQ_NWARPS];

    if (threadIdx.x % 32 == 0)
        smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < MMVQ_NWARPS) ? smem[lane] : 0.0f;
        for (int off = MMVQ_NWARPS / 2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0)
            y[row] = val + residual[row];
    }
}

#define Q4K_MMVQ_NTHREADS 128
#define Q4K_MMVQ_NWARPS   4

__global__ void matmul_q4_k_mmvq_kernel(const cuda_q4_k_block *__restrict__ w,
                                         const cuda_q8_k_block *__restrict__ xq,
                                         float *__restrict__ y, int n, int k) {
    const int nb = k / 256;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const cuda_q4_k_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    {
        for (int b = tid; b < nb; b += Q4K_MMVQ_NTHREADS) {
            const cuda_q4_k_block *wb = wrow + b;
            const cuda_q8_k_block *xb = xq + b;

            int32_t sumi = q4_k_dot(wb, xb->qs, xb->bsums, 0, 0);
            float xd = xb->d;
            acc = fmaf(xd, (float)sumi, acc);
        }
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    __shared__ float smem[4];
    if (threadIdx.x < 4) smem[threadIdx.x] = 0.0f;
    __syncthreads();

    int warp_lane = warp_id;
    if (lane == 0) {
        atomicAdd(&smem[warp_lane], acc);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < 4; i++)
            total += smem[i];
        y[row] = total;
    }
}

__global__ void matmul_q4_1_mmvq_kernel(const cuda_q4_1_block *__restrict__ w,
                                         const cuda_q8_0_block *__restrict__ xq,
                                         float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const cuda_q4_1_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    for (int b = tid; b < nb; b += MMVQ_NTHREADS) {
        const cuda_q4_1_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int8_t wdeq[32];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            wdeq[j]      = (int8_t)(wb->qs[j] & 0xF);
            wdeq[j + 16] = (int8_t)(wb->qs[j] >> 4);
        }

        int32_t dp_acc = 0;
        int32_t x_sum = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            int32_t a = (int32_t)((uint8_t)wdeq[j] | ((uint8_t)wdeq[j+1] << 8) |
                                  ((uint8_t)wdeq[j+2] << 16) | ((uint8_t)wdeq[j+3] << 24));
            int32_t bval = (int32_t)((uint8_t)xb->qs[j] | ((uint8_t)xb->qs[j+1] << 8) |
                                     ((uint8_t)xb->qs[j+2] << 16) | ((uint8_t)xb->qs[j+3] << 24));
            dp_acc = __dp4a(a, bval, dp_acc);
            x_sum += (int)xb->qs[j] + (int)xb->qs[j+1] + (int)xb->qs[j+2] + (int)xb->qs[j+3];
        }

        float dw = __half2float(__ushort_as_half(wb->d));
        float mw = __half2float(__ushort_as_half(wb->m));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)dp_acc, mw * cuda_q81_s(dx, x_sum) + acc);
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    __shared__ float smem[MMVQ_NWARPS];

    if (threadIdx.x % 32 == 0)
        smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < MMVQ_NWARPS) ? smem[lane] : 0.0f;
        for (int off = MMVQ_NWARPS / 2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0)
            y[row] = val;
    }
}

__global__ void matmul_q4_1_residual_mmvq_kernel(const cuda_q4_1_block *__restrict__ w,
                                                const cuda_q8_0_block *__restrict__ xq,
                                                const float *__restrict__ residual,
                                                float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const cuda_q4_1_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    for (int b = tid; b < nb; b += MMVQ_NTHREADS) {
        const cuda_q4_1_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int8_t wdeq[32];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            wdeq[j]      = (int8_t)(wb->qs[j] & 0xF);
            wdeq[j + 16] = (int8_t)(wb->qs[j] >> 4);
        }

        int32_t dp_acc = 0;
        int32_t x_sum = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            int32_t a = (int32_t)((uint8_t)wdeq[j] | ((uint8_t)wdeq[j+1] << 8) |
                                  ((uint8_t)wdeq[j+2] << 16) | ((uint8_t)wdeq[j+3] << 24));
            int32_t bval = (int32_t)((uint8_t)xb->qs[j] | ((uint8_t)xb->qs[j+1] << 8) |
                                     ((uint8_t)xb->qs[j+2] << 16) | ((uint8_t)xb->qs[j+3] << 24));
            dp_acc = __dp4a(a, bval, dp_acc);
            x_sum += (int)xb->qs[j] + (int)xb->qs[j+1] + (int)xb->qs[j+2] + (int)xb->qs[j+3];
        }

        float dw = __half2float(__ushort_as_half(wb->d));
        float mw = __half2float(__ushort_as_half(wb->m));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)dp_acc, mw * cuda_q81_s(dx, x_sum) + acc);
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    __shared__ float smem[MMVQ_NWARPS];

    if (threadIdx.x % 32 == 0)
        smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < MMVQ_NWARPS) ? smem[lane] : 0.0f;
        for (int off = MMVQ_NWARPS / 2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0)
            y[row] = val + residual[row];
    }
}

#define MMVQ_TC_WARPS_PER_BLOCK 4
#define MMVQ_TC_THREADS_PER_BLOCK (MMVQ_TC_WARPS_PER_BLOCK * 32)
#define MMVQ_TC_OUT_PER_WARP 16

__global__ void matmul_q8_0_mmvq_tc_kernel(const cuda_q8_0_block *__restrict__ w,
                                            const cuda_q8_0_block *__restrict__ xq,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    if (warp_id >= MMVQ_TC_WARPS_PER_BLOCK) return;

    const int out_start = (blockIdx.x * MMVQ_TC_WARPS_PER_BLOCK + warp_id) * MMVQ_TC_OUT_PER_WARP;
    if (out_start >= n) return;
    const int out_len = min(MMVQ_TC_OUT_PER_WARP, n - out_start);

    const cuda_q8_0_block *wrows[MMVQ_TC_OUT_PER_WARP];
    for (int i = 0; i < out_len; i++) {
        wrows[i] = w + (size_t)(out_start + i) * nb;
    }

    extern __shared__ char smem_mmvq_tc[];

    __half *w_smem = reinterpret_cast<__half *>(smem_mmvq_tc);

    __half *x_smem = &w_smem[MMVQ_TC_OUT_PER_WARP * WMMA_K];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[MMVQ_TC_OUT_PER_WARP];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[MMVQ_TC_OUT_PER_WARP];

    for (int r = 0; r < out_len; r++) {
        wmma::fill_fragment(acc_frag[r], 0.0f);
    }

    for (int kb = 0; kb < nb; kb += max(1, WMMA_K / 32)) {
        const int k_blocks_this_tile = min(max(1, WMMA_K / 32), nb - kb);

        for (int i = lane; i < out_len * k_blocks_this_tile * 32; i += 32) {
            int r = i / (k_blocks_this_tile * 32);
            int idx_in_tile = i % (k_blocks_this_tile * 32);
            int kb_local = idx_in_tile / 32;
            int j = idx_in_tile % 32;
            int row_idx = out_start + r;
            if (row_idx < n) {
                float scale, val;
                {
                    const cuda_q8_0_block *wb = wrows[r] + kb + kb_local;
                    scale = __half2float(__ushort_as_half(wb->d));
                    val = (int)wb->qs[j] * scale;
                }
                w_smem[(size_t)r * WMMA_K + kb_local * 32 + j] = __float2half(val);
            } else {
                w_smem[(size_t)r * WMMA_K + kb_local * 32 + j] = __float2half(0.0f);
            }
        }

        for (int i = lane; i < k_blocks_this_tile * 32 * WMMA_N; i += 32) {
            int kb_local = i / (32 * WMMA_N);
            int idx_in_kb = i % (32 * WMMA_N);
            int j = idx_in_kb % 32;
            int c = idx_in_kb / 32;
            if (c < WMMA_N) {
                const cuda_q8_0_block *xb = xq + kb + kb_local;
                float scale = __half2float(__ushort_as_half(xb->d));
                float val = (int)xb->qs[j] * scale;
                x_smem[(size_t)(kb_local * 32 + j) * WMMA_N + c] = __float2half(val);
            } else {
                x_smem[(size_t)(kb_local * 32 + j) * WMMA_N + c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int r = 0; r < out_len; r++) {
            for (int kk = 0; kk < k_blocks_this_tile * 32; kk += WMMA_K) {

                wmma::load_matrix_sync(a_frag[r],
                    &w_smem[(size_t)r * WMMA_K + kk], WMMA_K);

                wmma::load_matrix_sync(b_frag,
                    &x_smem[(size_t)kk * WMMA_N], WMMA_N);

                wmma::mma_sync(acc_frag[r], a_frag[r], b_frag, acc_frag[r]);
            }
        }
        __syncthreads();
    }

    __shared__ float smem_results[MMVQ_TC_WARPS_PER_BLOCK][MMVQ_TC_OUT_PER_WARP][WMMA_N];
    for (int r = 0; r < out_len; r++) {
        wmma::store_matrix_sync(&smem_results[warp_id][r][0], acc_frag[r], WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    if (lane == 0) {
        for (int r = 0; r < out_len; r++) {
            y[out_start + r] = smem_results[warp_id][r][0];
        }
    }
}

__global__ void matmul_q4_0_mmvq_tc_kernel(const cuda_q4_0_block *__restrict__ w,
                                            const cuda_q8_0_block *__restrict__ xq,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    if (warp_id >= MMVQ_TC_WARPS_PER_BLOCK) return;

    const int out_start = (blockIdx.x * MMVQ_TC_WARPS_PER_BLOCK + warp_id) * MMVQ_TC_OUT_PER_WARP;
    if (out_start >= n) return;
    const int out_len = min(MMVQ_TC_OUT_PER_WARP, n - out_start);

    const cuda_q4_0_block *wrows[MMVQ_TC_OUT_PER_WARP];
    for (int i = 0; i < out_len; i++) {
        wrows[i] = w + (size_t)(out_start + i) * nb;
    }

    extern __shared__ char smem_mmvq_tc[];
    __half *w_smem = reinterpret_cast<__half *>(smem_mmvq_tc);
    __half *x_smem = &w_smem[MMVQ_TC_OUT_PER_WARP * WMMA_K];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[MMVQ_TC_OUT_PER_WARP];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[MMVQ_TC_OUT_PER_WARP];

    for (int r = 0; r < out_len; r++) {
        wmma::fill_fragment(acc_frag[r], 0.0f);
    }

    for (int kb = 0; kb < nb; kb += max(1, WMMA_K / 32)) {
        const int k_blocks_this_tile = min(max(1, WMMA_K / 32), nb - kb);

        for (int i = lane; i < out_len * k_blocks_this_tile * 32; i += 32) {
            int r = i / (k_blocks_this_tile * 32);
            int idx_in_tile = i % (k_blocks_this_tile * 32);
            int kb_local = idx_in_tile / 32;
            int j = idx_in_tile % 32;
            int row_idx = out_start + r;
            if (row_idx < n) {
                float scale;
                int q;
                {
                    const cuda_q4_0_block *wb = wrows[r] + kb + kb_local;
                    scale = __half2float(__ushort_as_half(wb->d));
                    q = (j < 16) ? ((wb->qs[j] & 0xF) - 8) : ((wb->qs[j - 16] >> 4) - 8);
                }
                w_smem[(size_t)r * WMMA_K + kb_local * 32 + j] = __float2half(q * scale);
            } else {
                w_smem[(size_t)r * WMMA_K + kb_local * 32 + j] = __float2half(0.0f);
            }
        }

        for (int i = lane; i < k_blocks_this_tile * 32 * WMMA_N; i += 32) {
            int kb_local = i / (32 * WMMA_N);
            int idx_in_kb = i % (32 * WMMA_N);
            int j = idx_in_kb % 32;
            int c = idx_in_kb / 32;
            if (c < WMMA_N) {
                const cuda_q8_0_block *xb = xq + kb + kb_local;
                float scale = __half2float(__ushort_as_half(xb->d));
                float val = (int)xb->qs[j] * scale;
                x_smem[(size_t)(kb_local * 32 + j) * WMMA_N + c] = __float2half(val);
            } else {
                x_smem[(size_t)(kb_local * 32 + j) * WMMA_N + c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int r = 0; r < out_len; r++) {
            for (int kk = 0; kk < k_blocks_this_tile * 32; kk += WMMA_K) {
                wmma::load_matrix_sync(a_frag[r],
                    &w_smem[(size_t)r * WMMA_K + kk], WMMA_K);
                wmma::load_matrix_sync(b_frag,
                    &x_smem[(size_t)kk * WMMA_N], WMMA_N);
                wmma::mma_sync(acc_frag[r], a_frag[r], b_frag, acc_frag[r]);
            }
        }
        __syncthreads();
    }

    __shared__ float smem_results[MMVQ_TC_WARPS_PER_BLOCK][MMVQ_TC_OUT_PER_WARP][WMMA_N];
    for (int r = 0; r < out_len; r++) {
        wmma::store_matrix_sync(&smem_results[warp_id][r][0], acc_frag[r], WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    if (lane == 0) {
        for (int r = 0; r < out_len; r++) {
            y[out_start + r] = smem_results[warp_id][r][0];
        }
    }
}

#define MMVQ_WS_NWARPS     4
#define MMVQ_WS_THREADS    (MMVQ_WS_NWARPS * 32)
#define MMVQ_WS_TILE_B     16

__global__ void matmul_q8_0_mmvq_ws_kernel(const cuda_q8_0_block *__restrict__ w,
                                            const cuda_q8_0_block *__restrict__ xq,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const cuda_q8_0_block *wrow = w + (size_t)row * nb;
    const uint8_t *wrow8 = (const uint8_t *)wrow;

    extern __shared__ char smem_ws_q8_0[];
    float (*ws_w_tiles)[MMVQ_WS_TILE_B][33] = reinterpret_cast<float (*)[MMVQ_WS_TILE_B][33]>(smem_ws_q8_0);
    int8_t (*ws_x_qs)[32] = reinterpret_cast<int8_t (*)[32]>(&smem_ws_q8_0[2 * MMVQ_WS_TILE_B * 33 * sizeof(float)]);

    float acc = 0.0f;

    for (int tile_base = 0; tile_base < nb; tile_base += MMVQ_WS_TILE_B) {
        const int tiles_this_round = min(MMVQ_WS_TILE_B, nb - tile_base);
        const int tile_idx = (tile_base / MMVQ_WS_TILE_B) & 1;

        if (warp_id == 0) {
            for (int b = lane; b < tiles_this_round; b += 32) {
                const int global_b = tile_base + b;
                float scale;
                {
                    const cuda_q8_0_block *wb = wrow + global_b;
                    scale = __half2float(__ushort_as_half(wb->d));
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        ws_w_tiles[tile_idx][b][j] = (int)wb->qs[j] * scale;
                    }
                }
                ws_w_tiles[tile_idx][b][32] = scale;
            }
        }
        __syncthreads();

        if (tile_base == 0) {
            for (int b = lane; b < tiles_this_round; b += 32) {
                const int global_b = tile_base + b;
                const cuda_q8_0_block *xb = xq + global_b;
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    ws_x_qs[b][j] = xb->qs[j];
                }
            }
        }
        __syncthreads();

        if (warp_id >= 1 && warp_id <= 3) {
            const int num_compute_warps = 3;
            const int blocks_per_warp = (tiles_this_round + num_compute_warps - 1) / num_compute_warps;
            const int start_b = (warp_id - 1) * blocks_per_warp;
            const int end_b = min(start_b + blocks_per_warp, tiles_this_round);

            for (int b = start_b + lane; b < end_b; b += 32) {
                const int global_b = tile_base + b;
                const float *w_tile = ws_w_tiles[tile_idx][b];
                const float scale_w = w_tile[32];

                const cuda_q8_0_block *xb = xq + global_b;
                const float scale_x = __half2float(__ushort_as_half(xb->d));

                int32_t dp_acc = 0;
                #pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    int32_t a = (int32_t)((uint8_t)w_tile[j] |
                                         ((uint8_t)w_tile[j+1] << 8) |
                                         ((uint8_t)w_tile[j+2] << 16) |
                                         ((uint8_t)w_tile[j+3] << 24));
                    int32_t bval = (int32_t)((uint8_t)xb->qs[j] |
                                            ((uint8_t)xb->qs[j+1] << 8) |
                                            ((uint8_t)xb->qs[j+2] << 16) |
                                            ((uint8_t)xb->qs[j+3] << 24));
                    dp_acc = __dp4a(a, bval, dp_acc);
                }
                acc = fmaf(scale_w * scale_x, (float)dp_acc, acc);
            }
        }
        __syncthreads();
    }

    if (warp_id == 3) {
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) {
            y[row] = acc;
        }
    }
}

__global__ void matmul_q4_0_mmvq_ws_kernel(const cuda_q4_0_block *__restrict__ w,
                                            const cuda_q8_0_block *__restrict__ xq,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const cuda_q4_0_block *wrow = w + (size_t)row * nb;
    const uint8_t *wrow8 = (const uint8_t *)wrow;

    extern __shared__ char smem_ws_q4_0[];
    float (*ws_w_tiles)[MMVQ_WS_TILE_B][33] = reinterpret_cast<float (*)[MMVQ_WS_TILE_B][33]>(smem_ws_q4_0);
    int8_t (*ws_x_qs)[32] = reinterpret_cast<int8_t (*)[32]>(&smem_ws_q4_0[2 * MMVQ_WS_TILE_B * 33 * sizeof(float)]);

    float acc = 0.0f;

    for (int tile_base = 0; tile_base < nb; tile_base += MMVQ_WS_TILE_B) {
        const int tiles_this_round = min(MMVQ_WS_TILE_B, nb - tile_base);
        const int tile_idx = (tile_base / MMVQ_WS_TILE_B) & 1;

        if (warp_id == 0) {
            for (int b = lane; b < tiles_this_round; b += 32) {
                const int global_b = tile_base + b;
                float scale;
                {
                    const cuda_q4_0_block *wb = wrow + global_b;
                    scale = __half2float(__ushort_as_half(wb->d));
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        int q_lo = (wb->qs[j] & 0xF) - 8;
                        int q_hi = (wb->qs[j] >> 4) - 8;
                        ws_w_tiles[tile_idx][b][j]      = q_lo * scale;
                        ws_w_tiles[tile_idx][b][j + 16] = q_hi * scale;
                    }
                }
                ws_w_tiles[tile_idx][b][32] = scale;
            }
        }
        __syncthreads();

        if (tile_base == 0) {
            for (int b = lane; b < tiles_this_round; b += 32) {
                const int global_b = tile_base + b;
                const cuda_q8_0_block *xb = xq + global_b;
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    ws_x_qs[b][j] = xb->qs[j];
                }
            }
        }
        __syncthreads();

        if (warp_id >= 1 && warp_id <= 3) {
            const int num_compute_warps = 3;
            const int blocks_per_warp = (tiles_this_round + 3 - 1) / 3;
            const int start_b = (warp_id - 1) * blocks_per_warp;
            const int end_b = min(start_b + blocks_per_warp, tiles_this_round);

            for (int b = start_b + lane; b < end_b; b += 32) {
                const int global_b = tile_base + b;
                const float *w_tile = ws_w_tiles[tile_idx][b];
                const float scale_w = w_tile[32];

                const cuda_q8_0_block *xb = xq + global_b;
                const float scale_x = __half2float(__ushort_as_half(xb->d));

                int32_t dp_acc = 0;
                #pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    int32_t a = (int32_t)((uint8_t)w_tile[j] |
                                         ((uint8_t)w_tile[j+1] << 8) |
                                         ((uint8_t)w_tile[j+2] << 16) |
                                         ((uint8_t)w_tile[j+3] << 24));
                    int32_t bval = (int32_t)((uint8_t)xb->qs[j] |
                                            ((uint8_t)xb->qs[j+1] << 8) |
                                            ((uint8_t)xb->qs[j+2] << 16) |
                                            ((uint8_t)xb->qs[j+3] << 24));
                    dp_acc = __dp4a(a, bval, dp_acc);
                }
                acc = fmaf(scale_w * scale_x, (float)dp_acc, acc);
            }
        }
        __syncthreads();
    }

    if (warp_id == 3) {
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) {
            y[row] = acc;
        }
    }
}

__global__ void matmul_f32_kernel(const float *__restrict__ w, const float *__restrict__ x,
                                   float *__restrict__ y, int n, int k) {
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const float *wrow = w + (size_t)row * k;
    float acc = 0.0f;
    for (int j = lane; j < k; j += MM_LANES)
        acc = fmaf(wrow[j], x[j], acc);
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0)
        y[row] = acc;
}

__global__ void matmul_f16_kernel(const uint16_t *__restrict__ w, const float *__restrict__ x,
                                   float *__restrict__ y, int n, int k) {
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const uint16_t *wrow = w + (size_t)row * k;
    float acc = 0.0f;
    for (int j = lane; j < k; j += MM_LANES) {
        __half hf = __ushort_as_half(wrow[j]);
        acc = fmaf(__half2float(hf), x[j], acc);
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0)
        y[row] = acc;
}

__global__ void matmul_bf16_kernel(const uint16_t *__restrict__ w, const float *__restrict__ x,
                                    float *__restrict__ y, int n, int k) {
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const uint16_t *wrow = w + (size_t)row * k;
    float acc = 0.0f;
    for (int j = lane; j < k; j += MM_LANES) {

        __nv_bfloat16 bf = __ushort_as_bfloat16(wrow[j]);
        acc = fmaf((float)bf, x[j], acc);
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0)
        y[row] = acc;
}

__device__ __forceinline__ float act_gelu_tanh(float x) {
    const float c   = 0.7978845608028654f;
    const float x3  = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
}

__global__ void act_gelu_quant_batch_kernel(const float *__restrict__ gate,
                                            const float *__restrict__ up,
                                            cuda_q8_0_block *__restrict__ xq,
                                            int k, int m) {
    int row = blockIdx.y;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k / 32 || row >= m)
        return;

    const float *g = gate + (size_t)row * k + 32 * b;
    const float *u = up + (size_t)row * k + 32 * b;
    float vals[32];
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        const float a = act_gelu_tanh(g[j]) * u[j];
        vals[j]       = a;
        const float va = fabsf(a);
        if (va > amax)
            amax = va;
    }
    cuda_q8_0_block *out = &xq[(size_t)row * (k / 32) + b];
    if (amax < 1e-30f) {
        out->d = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            out->qs[j] = 0;
        return;
    }
    const float d = amax / 127.0f;
    store_half(&out->d, d);
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out->qs[j] = (int8_t)cuda_xq8_val(vals[j], d);
    }
}

__global__ void act_gelu_quant_kernel(const float *__restrict__ gate,
                                      const float *__restrict__ up,
                                      cuda_q8_0_block *__restrict__ xq, int k) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k / 32)
        return;
    const float *g = gate + 32 * b;
    const float *u = up + 32 * b;
    float vals[32];
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        const float a = act_gelu_tanh(g[j]) * u[j];
        vals[j]       = a;
        const float va = fabsf(a);
        if (va > amax)
            amax = va;
    }
    cuda_q8_0_block *out = &xq[b];
    if (amax < 1e-30f) {
        out->d = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            out->qs[j] = 0;
        return;
    }
    const float d = amax / 127.0f;
    store_half(&out->d, d);
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out->qs[j] = (int8_t)cuda_xq8_val(vals[j], d);
    }
}

static __device__ __forceinline__ float d_from_half(const uint16_t *h) {
    return __half2float(reinterpret_cast<const __half &>(*h));
}

struct mm_multi_desc {
    const char *w[8];
    float      *y[8];
    int         n[8];
    int         nm;
    int         k;
    const float *resid;
};

__global__ void matmul_multi_q8_0_kernel(const mm_multi_desc d,
                                         const cuda_q8_0_block *__restrict__ xq, int xq_qm) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const int g    = blockIdx.x * MM_ROWS_PER_BLOCK + warp;

    int col = d.nm, row = 0, cum = 0;
    for (int c = 0; c < d.nm; c++) {
        const int nn = d.n[c];
        if (g < cum + nn) { col = c; row = g - cum; break; }
        cum += nn;
    }
    if (col >= d.nm)
        return;

    const int nb = d.k / 32;
    const cuda_q8_0_block *wrow = (const cuda_q8_0_block *)d.w[col] + (size_t)row * nb;
    float acc = 0.0f;

    const uint8_t *xq8m = (const uint8_t *)xq;
    int xw = (xq_qm && ((nb & 1) == 0));
    int xn = (xq_qm && ((nb & 1) != 0));
    {
    for (int j = lane; j < nb; j += MM_LANES) {
        const cuda_q8_0_block *xb = &xq[j];
        const cuda_q8_0_block *wb = &wrow[j];

        int32_t dp_acc = 0;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int32_t a = q8_block_quad(wb, t);
            int32_t bval = xw ? qm_q8_quad_wide(xq8m, nb, j, t)
                              : xn ? qm_q8_quad(xq8m, nb, j, t)
                                   : q8_block_quad(xb, t);
            dp_acc = __dp4a(a, bval, dp_acc);
        }
        float dw = d_from_half(&wb->d);
        uint16_t dxb = xq_qm ? qm_q8_scale(xq8m, nb, j) : xb->d;
        float dx = __half2float(__ushort_as_half(dxb));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0)
        d.y[col][row] = acc;
}

__global__ void matmul_multi_q4_0_kernel(const mm_multi_desc d,
                                         const cuda_q8_0_block *__restrict__ xq) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const int g    = blockIdx.x * MM_ROWS_PER_BLOCK + warp;

    int col = d.nm, row = 0, cum = 0;
    for (int c = 0; c < d.nm; c++) {
        const int nn = d.n[c];
        if (g < cum + nn) { col = c; row = g - cum; break; }
        cum += nn;
    }
    if (col >= d.nm)
        return;

    const int nb = d.k / 32;
    const cuda_q4_0_block *wrow = (const cuda_q4_0_block *)d.w[col] + (size_t)row * nb;
    float acc = 0.0f;
    {
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q4_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int32_t dp_acc = 0;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int jt = (t < 4) ? (t << 2) : ((t - 4) << 2);
            uint8_t w0 = wb->qs[jt];
            uint8_t w1 = wb->qs[jt + 1];
            uint8_t w2 = wb->qs[jt + 2];
            uint8_t w3 = wb->qs[jt + 3];
            uint8_t x0 = xb->qs[(t < 4) ? jt : jt + 16];
            uint8_t x1 = xb->qs[(t < 4) ? jt + 1 : jt + 17];
            uint8_t x2 = xb->qs[(t < 4) ? jt + 2 : jt + 18];
            uint8_t x3 = xb->qs[(t < 4) ? jt + 3 : jt + 19];
            int32_t a, bval;
            if (t < 4) {
                a = (((int)(w0 & 0xF) - 8) & 0xFF) |
                    ((((int)(w1 & 0xF) - 8) & 0xFF) << 8) |
                    ((((int)(w2 & 0xF) - 8) & 0xFF) << 16) |
                    ((((int)(w3 & 0xF) - 8) & 0xFF) << 24);
            } else {
                a = (((int)(w0 >> 4) - 8) & 0xFF) |
                    ((((int)(w1 >> 4) - 8) & 0xFF) << 8) |
                    ((((int)(w2 >> 4) - 8) & 0xFF) << 16) |
                    ((((int)(w3 >> 4) - 8) & 0xFF) << 24);
            }
            bval = (int32_t)x0 | ((int32_t)x1 << 8) |
                   ((int32_t)x2 << 16) | ((int32_t)x3 << 24);
            dp_acc = __dp4a(a, bval, dp_acc);
        }
        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0)
        d.y[col][row] = acc;
}

__global__ void rmsnorm_matmul_multi_q4_0_kernel(const mm_multi_desc d,
                                                 const float *__restrict__ x,
                                                 const float *__restrict__ norm_w,
                                                 float eps) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int k = d.k;
    const int nb = k >> 5;

    extern __shared__ char sh[];
    float *reduce = (float *)sh;
    cuda_q8_0_block *xq = (cuda_q8_0_block *)(reduce + nt);

    float s = 0.0f;
    for (int i = tid; i < k; i += nt)
        s += x[i] * x[i];
    reduce[tid] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (tid < sz) reduce[tid] += reduce[tid + sz];
        __syncthreads();
    }
    if (tid == 0)
        reduce[0] = 1.0f / sqrtf((reduce[0] / (float)k) + eps);
    __syncthreads();
    const float scale = reduce[0];

    for (int b = tid; b < nb; b += nt) {
        float vals[32];
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            int idx = (b << 5) + j;
            float xn = x[idx] * scale;
            if (norm_w) xn *= norm_w[idx];
            vals[j] = xn;
            float a = fabsf(xn);
            if (a > amax) amax = a;
        }
        cuda_q8_0_block *out = &xq[b];
        if (amax < 1e-30f) {
            out->d = 0;
            #pragma unroll
            for (int j = 0; j < 32; j++)
                out->qs[j] = 0;
        } else {
            float qd = amax / 127.0f;
            store_half(&out->d, qd);
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                int q = (int)floorf(vals[j] / qd + 0.5f);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                out->qs[j] = (int8_t)q;
            }
        }
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & (MM_LANES - 1);
    const int g = blockIdx.x * MM_ROWS_PER_BLOCK + warp;

    int col = d.nm, row = 0, cum = 0;
    for (int c = 0; c < d.nm; c++) {
        const int nn = d.n[c];
        if (g < cum + nn) { col = c; row = g - cum; break; }
        cum += nn;
    }
    if (col >= d.nm)
        return;

    const cuda_q4_0_block *wrow = (const cuda_q4_0_block *)d.w[col] + (size_t)row * nb;
    float acc = 0.0f;
    {
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q4_0_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;

        int32_t dp_acc = 0;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int jt = t << 2;
            uint8_t x0 = xb->qs[jt];
            uint8_t x1 = xb->qs[jt + 1];
            uint8_t x2 = xb->qs[jt + 2];
            uint8_t x3 = xb->qs[jt + 3];
            int32_t a, bval;
            if (t < 4) {
                uint8_t w0 = wb->qs[jt];
                uint8_t w1 = wb->qs[jt + 1];
                uint8_t w2 = wb->qs[jt + 2];
                uint8_t w3 = wb->qs[jt + 3];
                a = (((int)(w0 & 0xF) - 8) & 0xFF) | ((((int)(w1 & 0xF) - 8) & 0xFF) << 8) |
                    ((((int)(w2 & 0xF) - 8) & 0xFF) << 16) | ((((int)(w3 & 0xF) - 8) & 0xFF) << 24);
            } else {
                int ju = jt - 16;
                uint8_t w0 = wb->qs[ju];
                uint8_t w1 = wb->qs[ju + 1];
                uint8_t w2 = wb->qs[ju + 2];
                uint8_t w3 = wb->qs[ju + 3];
                a = (((int)(w0 >> 4) - 8) & 0xFF) | ((((int)(w1 >> 4) - 8) & 0xFF) << 8) |
                    ((((int)(w2 >> 4) - 8) & 0xFF) << 16) | ((((int)(w3 >> 4) - 8) & 0xFF) << 24);
            }
            bval = (int32_t)x0 | ((int32_t)x1 << 8) |
                   ((int32_t)x2 << 16) | ((int32_t)x3 << 24);
            dp_acc = __dp4a(a, bval, dp_acc);
        }
        float dw = __half2float(__ushort_as_half(wb->d));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0)
        d.y[col][row] = acc;
}

__global__ void rmsnorm_matmul_multi_q8_0_kernel(const mm_multi_desc d,
                                                 const float *__restrict__ x,
                                                 const float *__restrict__ norm_w,
                                                 float eps) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int k = d.k;
    const int nb = k >> 5;

    extern __shared__ char sh[];
    float *reduce = (float *)sh;
    cuda_q8_0_block *xq = (cuda_q8_0_block *)(reduce + nt);

    float s = 0.0f;
    for (int i = tid; i < k; i += nt)
        s += x[i] * x[i];
    reduce[tid] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (tid < sz) reduce[tid] += reduce[tid + sz];
        __syncthreads();
    }
    if (tid == 0)
        reduce[0] = 1.0f / sqrtf((reduce[0] / (float)k) + eps);
    __syncthreads();
    const float scale = reduce[0];

    for (int b = tid; b < nb; b += nt) {
        float vals[32];
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            int idx = (b << 5) + j;
            float xn = x[idx] * scale;
            if (norm_w) xn *= norm_w[idx];
            vals[j] = xn;
            float a = fabsf(xn);
            if (a > amax) amax = a;
        }
        cuda_q8_0_block *out = &xq[b];
        if (amax < 1e-30f) {
            out->d = 0;
            #pragma unroll
            for (int j = 0; j < 32; j++)
                out->qs[j] = 0;
        } else {
            float qd = amax / 127.0f;
            store_half(&out->d, qd);
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                int q = (int)floorf(vals[j] / qd + 0.5f);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                out->qs[j] = (int8_t)q;
            }
        }
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & (MM_LANES - 1);
    const int g = blockIdx.x * MM_ROWS_PER_BLOCK + warp;

    int col = d.nm, row = 0, cum = 0;
    for (int c = 0; c < d.nm; c++) {
        const int nn = d.n[c];
        if (g < cum + nn) { col = c; row = g - cum; break; }
        cum += nn;
    }
    if (col >= d.nm)
        return;

    const cuda_q8_0_block *wrow = (const cuda_q8_0_block *)d.w[col] + (size_t)row * nb;
    float acc = 0.0f;
    {
    for (int j = lane; j < nb; j += MM_LANES) {
        const cuda_q8_0_block *xb = &xq[j];
        const cuda_q8_0_block *wb = &wrow[j];
        int32_t dp_acc = 0;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int jt = t << 2;
            int32_t a = q8_block_quad(wb, t);
            uint8_t b0 = xb->qs[jt];
            uint8_t b1 = xb->qs[jt + 1];
            uint8_t b2 = xb->qs[jt + 2];
            uint8_t b3 = xb->qs[jt + 3];
            int32_t bval = (int32_t)b0 | ((int32_t)b1 << 8) |
                           ((int32_t)b2 << 16) | ((int32_t)b3 << 24);
            dp_acc = __dp4a(a, bval, dp_acc);
        }
        float dw = d_from_half(&wb->d);
        float dx = d_from_half(&xb->d);
        acc = fmaf(dw * dx, (float)dp_acc, acc);
    }
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0)
        d.y[col][row] = acc;
}

__global__ void rmsnorm_matmul_multi_q4_1_kernel(const mm_multi_desc d,
                                                 const float *__restrict__ x,
                                                 const float *__restrict__ norm_w,
                                                 float eps) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int k = d.k;
    const int nb = k >> 5;

    extern __shared__ char sh[];
    float *reduce = (float *)sh;
    cuda_q8_0_block *xq = (cuda_q8_0_block *)(reduce + nt);

    float s = 0.0f;
    for (int i = tid; i < k; i += nt)
        s += x[i] * x[i];
    reduce[tid] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (tid < sz) reduce[tid] += reduce[tid + sz];
        __syncthreads();
    }
    if (tid == 0)
        reduce[0] = 1.0f / sqrtf((reduce[0] / (float)k) + eps);
    __syncthreads();
    const float scale = reduce[0];

    for (int b = tid; b < nb; b += nt) {
        float vals[32];
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            int idx = (b << 5) + j;
            float xn = x[idx] * scale;
            if (norm_w) xn *= norm_w[idx];
            vals[j] = xn;
            float a = fabsf(xn);
            if (a > amax) amax = a;
        }
        cuda_q8_0_block *out = &xq[b];
        if (amax < 1e-30f) {
            out->d = 0;
            #pragma unroll
            for (int j = 0; j < 32; j++)
                out->qs[j] = 0;
        } else {
            float qd = amax / 127.0f;
            store_half(&out->d, qd);
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                int q = (int)floorf(vals[j] / qd + 0.5f);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                out->qs[j] = (int8_t)q;
            }
        }
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & (MM_LANES - 1);
    const int g = blockIdx.x * MM_ROWS_PER_BLOCK + warp;

    int col = d.nm, row = 0, cum = 0;
    for (int c = 0; c < d.nm; c++) {
        const int nn = d.n[c];
        if (g < cum + nn) { col = c; row = g - cum; break; }
        cum += nn;
    }
    if (col >= d.nm)
        return;

    const cuda_q4_1_block *wrow = (const cuda_q4_1_block *)d.w[col] + (size_t)row * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q4_1_block *wb = wrow + b;
        const cuda_q8_0_block *xb = xq + b;
        int sum_qx = 0, sum_x = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int qlo = (int)(wb->qs[j] & 0xF);
            int qhi = (int)(wb->qs[j] >> 4);
            sum_qx += qlo * (int)xb->qs[j] + qhi * (int)xb->qs[j + 16];
            sum_x  += (int)xb->qs[j] + (int)xb->qs[j + 16];
        }
        float dw = __half2float(__ushort_as_half(wb->d));
        float mw = __half2float(__ushort_as_half(wb->m));
        float dx = __half2float(__ushort_as_half(xb->d));
        acc = fmaf(dw * dx, (float)sum_qx, mw * cuda_q81_s(dx, sum_x) + acc);
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0)
        d.y[col][row] = acc;
}

__global__ void rmsnorm_kernel(const float *__restrict__ x, const float *__restrict__ w,
                               float *__restrict__ y, int n, float eps) {
    extern __shared__ float sm[];
    const int t  = threadIdx.x;
    const int nt = blockDim.x;
    float s = 0.0f;
    for (int i = t; i < n; i += nt)
        s += x[i] * x[i];
    sm[t] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (t < sz) sm[t] += sm[t + sz];
        __syncthreads();
    }
    if (t == 0)
        sm[0] = 1.0f / sqrtf((sm[0] / (float)n) + eps);
    __syncthreads();
    const float scale = sm[0];
    for (int i = t; i < n; i += nt)
        y[i] = (w ? x[i] * scale * w[i] : x[i] * scale);
}

__global__ void rmsnorm_per_head_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                        float *__restrict__ y, int head_dim, float eps) {
    const int h  = blockIdx.x;
    const int t  = threadIdx.x;
    const int nt = blockDim.x;
    const float *xh = x + (size_t)h * head_dim;
    float *yh       = y + (size_t)h * head_dim;
    extern __shared__ float sm[];
    float s = 0.0f;
    for (int i = t; i < head_dim; i += nt)
        s += xh[i] * xh[i];
    sm[t] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (t < sz) sm[t] += sm[t + sz];
        __syncthreads();
    }
    if (t == 0)
        sm[0] = 1.0f / sqrtf((sm[0] / (float)head_dim) + eps);
    __syncthreads();
    const float scale = sm[0];
    for (int i = t; i < head_dim; i += nt)
        yh[i] = (w ? xh[i] * scale * w[i] : xh[i] * scale);
}

__global__ void rmsnorm_batch_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                       float *__restrict__ y, int n, float eps, int m) {

    const int r = blockIdx.x;
    if (r >= m) return;
    extern __shared__ float sm[];
    const int t  = threadIdx.x;
    const int nt = blockDim.x;
    const float *xr = x + (size_t)r * n;
    float *yr = y + (size_t)r * n;
    float s = 0.0f;
    for (int i = t; i < n; i += nt)
        s += xr[i] * xr[i];
    sm[t] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (t < sz) sm[t] += sm[t + sz];
        __syncthreads();
    }
    if (t == 0)
        sm[0] = 1.0f / sqrtf((sm[0] / (float)n) + eps);
    __syncthreads();
    const float scale = sm[0];
    for (int i = t; i < n; i += nt)
        yr[i] = (w ? xr[i] * scale * w[i] : xr[i] * scale);
}

extern "C" void cuda_rmsnorm(const float *x_dev, const float *w_dev, float *y_dev, int n, float eps,
                             cudaStream_t stream) {
    const int nt = n < 256 ? 32 : 256;
    rmsnorm_kernel<<<1, nt, nt * sizeof(float), stream>>>(x_dev, w_dev, y_dev, n, eps);
}

extern "C" void cuda_rmsnorm_batch(const float *x_dev, const float *w_dev, float *y_dev,
                                   int n, float eps, int m, cudaStream_t stream) {
    const int nt = n < 256 ? 32 : 256;
    dim3 grid((unsigned)m);
    rmsnorm_batch_kernel<<<grid, nt, (size_t)nt * sizeof(float), stream>>>(
        x_dev, w_dev, y_dev, n, eps, m);
}

extern "C" void cuda_rmsnorm_per_head(const float *x_dev, const float *w_dev, float *y_dev,
                                      int n_heads, int head_dim, float eps, cudaStream_t stream) {
    const int nt = head_dim <= 512 ? head_dim : 256;
    rmsnorm_per_head_kernel<<<n_heads, nt, nt * sizeof(float), stream>>>(x_dev, w_dev, y_dev,
                                                                        head_dim, eps);
}

__global__ void rmsnorm_add_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                   const float *__restrict__ residual, float *__restrict__ y,
                                   int n, float eps, float out_scale) {
    extern __shared__ float sm[];
    const int t  = threadIdx.x;
    const int nt = blockDim.x;
    float s = 0.0f;
    for (int i = t; i < n; i += nt)
        s += x[i] * x[i];
    sm[t] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (t < sz) sm[t] += sm[t + sz];
        __syncthreads();
    }
    if (t == 0)
        sm[0] = 1.0f / sqrtf((sm[0] / (float)n) + eps);
    __syncthreads();
    const float scale = sm[0];
    for (int i = t; i < n; i += nt) {
        float normed = (w ? x[i] * scale * w[i] : x[i] * scale);

        y[i] = (normed + residual[i]) * out_scale;
    }
}

extern "C" void cuda_rmsnorm_add(const float *x_dev, const float *w_dev, const float *residual_dev,
                                 float *y_dev, int n, float eps, float out_scale,
                                 cudaStream_t stream) {
    const int nt = n < 256 ? 32 : 256;
    rmsnorm_add_kernel<<<1, nt, nt * sizeof(float), stream>>>(x_dev, w_dev, residual_dev, y_dev,
                                                              n, eps, out_scale);
}

extern "C" void cuda_rmsnorm_per_head_g(const float *x_dev, const float *w_dev, float *y_dev,
                                        int n_heads, int head_dim, float eps,
                                        const int *params_dev, cudaStream_t stream) {
    const int nt = head_dim <= 512 ? head_dim : 256;
    rmsnorm_per_head_kernel<<<n_heads, nt, nt * sizeof(float), stream>>>(
        x_dev, w_dev, y_dev, head_dim, eps);
}

__device__ __forceinline__ void qkv_norm_rope_head(float *vh, const float *wh,
                                                   int head_dim, float eps,
                                                   const float *cos_tbl, const float *sin_tbl,
                                                   int neox, float *sm, int t, int nt) {

    float s = 0.0f;
    for (int i = t; i < head_dim; i += nt)
        s += vh[i] * vh[i];
    sm[t] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (t < sz) sm[t] += sm[t + sz];
        __syncthreads();
    }
    if (t == 0)
        sm[0] = 1.0f / sqrtf((sm[0] / (float)head_dim) + eps);
    __syncthreads();
    const float scale = sm[0];
    for (int i = t; i < head_dim; i += nt)
        vh[i] = (wh ? vh[i] * scale * wh[i] : vh[i] * scale);
    __syncthreads();

    if (cos_tbl && sin_tbl) {
        const int half = head_dim / 2;
        for (int j = t; j < half; j += nt) {
            float c = cos_tbl[j];
            float s2 = sin_tbl[j];
            if (neox) {
                float v0 = vh[j];
                float v1 = vh[j + half];
                vh[j]        = v0 * c - v1 * s2;
                vh[j + half] = v0 * s2 + v1 * c;
            } else {
                float v0 = vh[2 * j];
                float v1 = vh[2 * j + 1];
                vh[2 * j]     = v0 * c - v1 * s2;
                vh[2 * j + 1] = v0 * s2 + v1 * c;
            }
        }
    }
}

__global__ void qkv_norm_rope_kernel(float *q, float *k, float *v,
                                     const float *wq_norm, const float *wk_norm,
                                     int n_heads, int n_kv_heads, int head_dim, float eps,
                                     const float *cos_tbl, const float *sin_tbl, int neox) {
    const int h  = blockIdx.x;
    const int t  = threadIdx.x;
    const int nt = blockDim.x;
    extern __shared__ float sm[];
    if (h < n_heads)
        qkv_norm_rope_head(q + (size_t)h * head_dim, wq_norm, head_dim, eps,
                           cos_tbl, sin_tbl, neox, sm, t, nt);
    __syncthreads();
    if (h < n_kv_heads) {
        qkv_norm_rope_head(k + (size_t)h * head_dim, wk_norm, head_dim, eps,
                           cos_tbl, sin_tbl, neox, sm, t, nt);
        __syncthreads();

        float *vh = v + (size_t)h * head_dim;
        float s = 0.0f;
        for (int i = t; i < head_dim; i += nt)
            s += vh[i] * vh[i];
        sm[t] = s;
        __syncthreads();
        for (int sz = nt >> 1; sz > 0; sz >>= 1) {
            if (t < sz) sm[t] += sm[t + sz];
            __syncthreads();
        }
        if (t == 0)
            sm[0] = 1.0f / sqrtf((sm[0] / (float)head_dim) + eps);
        __syncthreads();
        const float scale = sm[0];
        for (int i = t; i < head_dim; i += nt)
            vh[i] = vh[i] * scale;
    }
}

__global__ void qkv_norm_rope_g_kernel(float *q, float *k, float *v,
                                       const float *wq_norm, const float *wk_norm,
                                       int n_heads, int n_kv_heads, int head_dim, float eps,
                                       const float *cos_base, const float *sin_base,
                                       const cuda_decode_params *dp, int neox) {
    const int h  = blockIdx.x;
    const int t  = threadIdx.x;
    const int nt = blockDim.x;
    extern __shared__ float sm[];
    const int half = head_dim / 2;
    const float *cos_tbl = cos_base + (size_t)dp->pos * half;
    const float *sin_tbl = sin_base + (size_t)dp->pos * half;
    if (h < n_heads)
        qkv_norm_rope_head(q + (size_t)h * head_dim, wq_norm, head_dim, eps,
                           cos_tbl, sin_tbl, neox, sm, t, nt);
    __syncthreads();
    if (h < n_kv_heads) {
        qkv_norm_rope_head(k + (size_t)h * head_dim, wk_norm, head_dim, eps,
                           cos_tbl, sin_tbl, neox, sm, t, nt);
        __syncthreads();
        float *vh = v + (size_t)h * head_dim;
        float s = 0.0f;
        for (int i = t; i < head_dim; i += nt)
            s += vh[i] * vh[i];
        sm[t] = s;
        __syncthreads();
        for (int sz = nt >> 1; sz > 0; sz >>= 1) {
            if (t < sz) sm[t] += sm[t + sz];
            __syncthreads();
        }
        if (t == 0)
            sm[0] = 1.0f / sqrtf((sm[0] / (float)head_dim) + eps);
        __syncthreads();
        const float scale = sm[0];
        for (int i = t; i < head_dim; i += nt)
            vh[i] = vh[i] * scale;
    }
}

extern "C" void cuda_qkv_norm_rope(float *q_dev, float *k_dev, float *v_dev,
                                   const float *wq_norm_dev, const float *wk_norm_dev,
                                   int n_heads, int n_kv_heads, int head_dim, float eps,
                                   int pos, const float *cos_tbl_row, const float *sin_tbl_row,
                                   int neox, cudaStream_t stream) {
    const int nt = head_dim <= 512 ? head_dim : 256;
    const int ng = n_heads > n_kv_heads ? n_heads : n_kv_heads;
    qkv_norm_rope_kernel<<<ng, nt, (size_t)nt * sizeof(float), stream>>>(
        q_dev, k_dev, v_dev, wq_norm_dev, wk_norm_dev,
        n_heads, n_kv_heads, head_dim, eps, cos_tbl_row, sin_tbl_row, neox);
}

extern "C" void cuda_qkv_norm_rope_g(float *q_dev, float *k_dev, float *v_dev,
                                     const float *wq_norm_dev, const float *wk_norm_dev,
                                     int n_heads, int n_kv_heads, int head_dim, float eps,
                                     const float *cos_base, const float *sin_base,
                                     const int *params_dev, int neox, cudaStream_t stream) {
    const int nt = head_dim <= 512 ? head_dim : 256;
    const int ng = n_heads > n_kv_heads ? n_heads : n_kv_heads;
    qkv_norm_rope_g_kernel<<<ng, nt, (size_t)nt * sizeof(float), stream>>>(
        q_dev, k_dev, v_dev, wq_norm_dev, wk_norm_dev,
        n_heads, n_kv_heads, head_dim, eps, cos_base, sin_base,
        (const cuda_decode_params *)params_dev, neox);
}

__global__ void rmsnorm_add_batch_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                         const float *__restrict__ residual, float *__restrict__ y,
                                         int n, float eps, int m, float out_scale) {
    const int r = blockIdx.x;
    if (r >= m) return;
    extern __shared__ float sm[];
    const int t  = threadIdx.x;
    const int nt = blockDim.x;
    const float *xr = x + (size_t)r * n;
    const float *rr = residual + (size_t)r * n;
    float *yr = y + (size_t)r * n;
    float s = 0.0f;
    for (int i = t; i < n; i += nt)
        s += xr[i] * xr[i];
    sm[t] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (t < sz) sm[t] += sm[t + sz];
        __syncthreads();
    }
    if (t == 0)
        sm[0] = 1.0f / sqrtf((sm[0] / (float)n) + eps);
    __syncthreads();
    const float scale = sm[0];
    for (int i = t; i < n; i += nt) {
        float normed = (w ? xr[i] * scale * w[i] : xr[i] * scale);

        yr[i] = (normed + rr[i]) * out_scale;
    }
}

extern "C" void cuda_rmsnorm_add_batch(const float *x_dev, const float *w_dev,
                                       const float *residual_dev, float *y_dev, int n, float eps,
                                       int m, float out_scale, cudaStream_t stream) {
    const int nt = n < 256 ? 32 : 256;
    dim3 grid((unsigned)m);
    rmsnorm_add_batch_kernel<<<grid, nt, (size_t)nt * sizeof(float), stream>>>(
        x_dev, w_dev, residual_dev, y_dev, n, eps, m, out_scale);
}

__global__ void rmsnorm_rope_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                    float *__restrict__ y, int n_heads, int head_dim,
                                    const float *__restrict__ cos_tbl,
                                    const float *__restrict__ sin_tbl, float eps, int neox) {
    int half = head_dim / 2;
    int h = blockIdx.x;
    if (h >= n_heads) return;

    int t = threadIdx.x;
    int nt = blockDim.x;

    const float *xh = x + (size_t)h * head_dim;
    float *yh = y + (size_t)h * head_dim;

    extern __shared__ float sm[];
    float s = 0.0f;
    for (int i = t; i < head_dim; i += nt)
        s += xh[i] * xh[i];
    sm[t] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (t < sz) sm[t] += sm[t + sz];
        __syncthreads();
    }
    if (t == 0)
        sm[0] = 1.0f / sqrtf((sm[0] / (float)head_dim) + eps);
    __syncthreads();
    const float scale = sm[0];

    for (int i = t; i < head_dim; i += nt)
        yh[i] = (w ? xh[i] * scale * w[i] : xh[i] * scale);
    __syncthreads();

    for (int j = t; j < half; j += nt) {
        float c = cos_tbl[j];
        float s_val = sin_tbl[j];
        if (neox) {
            float v0 = yh[j];
            float v1 = yh[j + half];
            yh[j]        = v0 * c - v1 * s_val;
            yh[j + half] = v0 * s_val + v1 * c;
        } else {
            float v0 = yh[2 * j];
            float v1 = yh[2 * j + 1];
            yh[2 * j]     = v0 * c - v1 * s_val;
            yh[2 * j + 1] = v0 * s_val + v1 * c;
        }
    }
}

extern "C" void cuda_rmsnorm_rope(const float *x_dev, const float *w_dev, float *y_dev,
                                  int n_heads, int head_dim, const float *cos_dev,
                                  const float *sin_dev, float eps, int neox, cudaStream_t stream) {
    const int nt = head_dim <= 512 ? head_dim : 256;
    rmsnorm_rope_kernel<<<n_heads, nt, nt * sizeof(float), stream>>>(x_dev, w_dev, y_dev,
                                                                     n_heads, head_dim,
                                                                     cos_dev, sin_dev, eps, neox);
}

static cuda_q8_0_block *g_xq_scratch = nullptr;
static size_t           g_xq_cap      = 0;
static cuda_q8_k_block *g_q8k_scratch = nullptr;
static size_t           g_q8k_cap     = 0;

static cuda_q8_k_block *q8k_scratch_for(int k) {
    size_t need = (size_t)(k / 256) * sizeof(cuda_q8_k_block);
    if (need <= g_q8k_cap)
        return g_q8k_scratch;
    if (g_q8k_scratch)
        cudaFree(g_q8k_scratch);
    g_q8k_scratch = nullptr;
    g_q8k_cap     = 0;
    if (cudaMalloc((void **)&g_q8k_scratch, need) != cudaSuccess)
        return nullptr;
    g_q8k_cap = need;
    return g_q8k_scratch;
}

static cuda_q8_1_block *g_q81_scratch = nullptr;
static size_t          g_q81_cap     = 0;

static cuda_q8_1_block *q81_scratch_for(size_t nelem) {
    size_t need = (nelem / 32) * sizeof(cuda_q8_1_block);
    if (need <= g_q81_cap)
        return g_q81_scratch;
    if (g_q81_scratch)
        cudaFree(g_q81_scratch);
    g_q81_scratch = nullptr;
    g_q81_cap     = 0;
    if (cudaMalloc((void **)&g_q81_scratch, need) != cudaSuccess)
        return nullptr;
    g_q81_cap = need;
    return g_q81_scratch;
}

static cuda_q8_0_block *xq_scratch_for(int k) {
    size_t need = (size_t)(k / 32) * sizeof(cuda_q8_0_block);
    if (need <= g_xq_cap)
        return g_xq_scratch;
    if (g_xq_scratch)
        cudaFree(g_xq_scratch);
    g_xq_scratch = nullptr;
    g_xq_cap     = 0;
    if (cudaMalloc((void **)&g_xq_scratch, need) != cudaSuccess)
        return nullptr;
    g_xq_cap = need;
    return g_xq_scratch;
}

extern "C" void cuda_quantize_f32_to_q8_0(const float *x_dev, cuda_q8_0_block *xq_dev, int k) {
    int nb = k / 32;
    int block = 256;
    dim3 grid((nb + block - 1) / block);
    q8_0_quant_kernel<<<grid, block>>>(x_dev, xq_dev, k);
}

extern "C" void cuda_quantize_f32_to_q4_0(const float *x_dev, cuda_q4_0_block *xq_dev, int k) {
    int nb = k / 32;
    int block = 256;
    dim3 grid((nb + block - 1) / block);
    q4_0_quant_kernel<<<grid, block>>>(x_dev, xq_dev, k);
}

extern "C" void cuda_quantize_f32_to_q8_k(const float *x_dev, cuda_q8_k_block *xq_dev, int k) {
    int nb = k / 256;
    int block = 256;
    dim3 grid((nb + block - 1) / block);
    q8_k_quant_kernel<<<grid, block>>>(x_dev, xq_dev, k);
}

extern "C" void cuda_matmul_q8_0(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                                 cudaStream_t stream) {

    cuda_q8_0_block *xq_dev = xq_scratch_for(k);
    if (!xq_dev)
        return;

    const int use_v2e = 1;

    const int use_xqm = (n != 1 && use_v2e) ? 1 : 0;
    if (use_xqm)

        q8_0_quant_qm_kernel<<<(unsigned)(k / 32), 32, 0, stream>>>(
            x_dev, (uint8_t *)xq_dev, k);
    else
        q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);

    if (n == 1) {

        matmul_q8_0_mmvq_kernel<<<1, MMVQ_NTHREADS, 0, stream>>>(
            (const cuda_q8_0_block *)w_dev, xq_dev, y_dev, n, k);
    } else {
        dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        if (use_v2e) {

            const int use_stage = !use_xqm ? 1 : 0;
            size_t shmem = (size_t)(k >> 5) * sizeof(cuda_q8_0_block);
            matmul_q8_0_dp4a_v2_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, shmem, stream>>>(
                (const cuda_q8_0_block *)w_dev, xq_dev, y_dev, n, k, use_stage,
                use_xqm);
        } else
            matmul_q8_0_dp4a_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
                (const cuda_q8_0_block *)w_dev, xq_dev, y_dev, n, k);
    }
}

extern "C" void cuda_matmul_q4_0(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                                 cudaStream_t stream) {

    cuda_q8_0_block *xq_dev = xq_scratch_for(k);
    if (!xq_dev)
        return;
    q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);

    if (n == 1) {

        matmul_q4_0_mmvq_kernel<<<1, MMVQ_NTHREADS, 0, stream>>>(
            (const cuda_q4_0_block *)w_dev, xq_dev, y_dev, n, k);
    } else {
        const int use_v2 = 1;
        dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        if (use_v2) {

            const int use_stage = 1;
            size_t shmem = (size_t)(k >> 5) * (sizeof(cuda_q8_0_block) + sizeof(int));
            matmul_q4_0_dp4a_v2_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, shmem, stream>>>(
                (const cuda_q4_0_block *)w_dev, xq_dev, y_dev, n, k, use_stage);
        } else
            matmul_q4_0_dp4a_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
                (const cuda_q4_0_block *)w_dev, xq_dev, y_dev, n, k);
    }
}

__constant__ int8_t cuda_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static __device__ __forceinline__ float iq4_nl_dot(const cuda_iq4_nl_block *blk,
                                                   const int8_t *xq8, float d_xq, float acc) {
    const uint8_t *qs = blk->qs;
    int32_t        sumi0 = 0;
    int32_t        sumi1 = 0;

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        sumi0 += (int)xq8[j] * (int)cuda_kvalues_iq4nl[qs[j] & 0xF];
        sumi1 += (int)xq8[j + 16] * (int)cuda_kvalues_iq4nl[qs[j] >> 4];
    }
    const float d = __half2float(__ushort_as_half(blk->d));
    return fmaf(d * d_xq, (float)(sumi0 + sumi1), acc);
}

__global__ void matmul_iq4_nl_kernel(const cuda_iq4_nl_block *__restrict__ w,
                                     const cuda_q8_0_block *__restrict__ xq,
                                     float *__restrict__ y, int n, int k) {
    const int nb  = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_iq4_nl_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q8_0_block *xb = xq + b;
        acc = iq4_nl_dot(wrow + b, xb->qs, __half2float(__ushort_as_half(xb->d)), acc);
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc;
}

__global__ void matmul_iq4_nl_residual_kernel(const cuda_iq4_nl_block *__restrict__ w,
                                              const cuda_q8_0_block *__restrict__ xq,
                                              const float *__restrict__ residual,
                                              float *__restrict__ y, int n, int k) {
    const int nb  = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_iq4_nl_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q8_0_block *xb = xq + b;
        acc = iq4_nl_dot(wrow + b, xb->qs, __half2float(__ushort_as_half(xb->d)), acc);
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc + residual[row];
}

__global__ void matmul_iq4_nl_batch_kernel(const cuda_iq4_nl_block *__restrict__ w,
                                           const cuda_q8_0_block *__restrict__ xq,
                                           float *__restrict__ y, int n, int k, int m) {
    const int nb  = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_iq4_nl_block *wrow = w + (size_t)row * nb;

    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            const cuda_iq4_nl_block *wb = wrow + b;
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt)
                    break;
                const cuda_q8_0_block *xb = xq + ((size_t)(t0 + t) * nb + b);
                acc[t] = iq4_nl_dot(wb, xb->qs, __half2float(__ushort_as_half(xb->d)), acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt)
                break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if ((threadIdx.x & (MM_LANES - 1)) == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

extern "C" void cuda_matmul_q4_1(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                                 cudaStream_t stream) {

    cuda_q8_0_block *xq_dev = xq_scratch_for(k);
    if (!xq_dev)
        return;
    q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);
    if (n == 1) {
        matmul_q4_1_mmvq_kernel<<<1, MMVQ_NTHREADS, 0, stream>>>(
            (const cuda_q4_1_block *)w_dev, xq_dev, y_dev, n, k);
    } else {
        dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        size_t shmem = (size_t)(k >> 5) * sizeof(cuda_q8_0_block);
        matmul_q4_1_dp4a_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, shmem, stream>>>(
            (const cuda_q4_1_block *)w_dev, xq_dev, y_dev, n, k);
    }
}

extern "C" void cuda_matmul_q8_0_residual(const void *w_dev, const float *x_dev,
                                         const float *residual_dev, float *y_dev,
                                         int n, int k, cudaStream_t stream) {
    cuda_q8_0_block *xq_dev = xq_scratch_for(k);
    if (!xq_dev)
        return;

    const int use_v2r = 1;
    const int use_xqmr = (n != 1 && use_v2r) ? 1 : 0;
    if (use_xqmr)
        q8_0_quant_qm_kernel<<<(unsigned)(k / 32), 32, 0, stream>>>(
            x_dev, (uint8_t *)xq_dev, k);
    else
        q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);
    if (n == 1) {
        matmul_q8_0_residual_mmvq_kernel<<<1, MMVQ_NTHREADS, 0, stream>>>(
            (const cuda_q8_0_block *)w_dev, xq_dev, residual_dev, y_dev, n, k);
    } else {
        dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        if (use_v2r)
            matmul_q8_0_residual_v2_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
                (const cuda_q8_0_block *)w_dev, xq_dev, residual_dev, y_dev, n, k,
                use_xqmr);
        else
            matmul_q8_0_residual_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
                (const cuda_q8_0_block *)w_dev, xq_dev, residual_dev, y_dev, n, k);
    }
}

extern "C" void cuda_matmul_q4_0_residual(const void *w_dev, const float *x_dev,
                                         const float *residual_dev, float *y_dev,
                                         int n, int k, cudaStream_t stream) {
    cuda_q8_0_block *xq_dev = xq_scratch_for(k);
    if (!xq_dev)
        return;
    q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);
    if (n == 1) {
        matmul_q4_0_residual_mmvq_kernel<<<1, MMVQ_NTHREADS, 0, stream>>>(
            (const cuda_q4_0_block *)w_dev, xq_dev, residual_dev, y_dev, n, k);
    } else {
        const int use_v2 = 1;
        dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        if (use_v2)
            matmul_q4_0_residual_v2_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
                (const cuda_q4_0_block *)w_dev, xq_dev, residual_dev, y_dev, n, k);
        else
            matmul_q4_0_residual_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
                (const cuda_q4_0_block *)w_dev, xq_dev, residual_dev, y_dev, n, k);
    }
}

extern "C" void cuda_matmul_q4_1_residual(const void *w_dev, const float *x_dev,
                                         const float *residual_dev, float *y_dev,
                                         int n, int k, cudaStream_t stream) {
    cuda_q8_0_block *xq_dev = xq_scratch_for(k);
    if (!xq_dev)
        return;
    q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);
    if (n == 1) {
        matmul_q4_1_residual_mmvq_kernel<<<1, MMVQ_NTHREADS, 0, stream>>>(
            (const cuda_q4_1_block *)w_dev, xq_dev, residual_dev, y_dev, n, k);
    } else {
        dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        matmul_q4_1_residual_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
            (const cuda_q4_1_block *)w_dev, xq_dev, residual_dev, y_dev, n, k);
    }
}

static __device__ __forceinline__ void q6_k_unpack(const cuda_q6_k_block *blk, int8_t *a) {
    const uint8_t *ql = blk->ql;
    const uint8_t *qh = blk->qh;
    for (int half = 0; half < 2; half++) {
        for (int l = 0; l < 32; l++) {
            const uint8_t qhl = qh[l];
            const uint8_t q0  = ql[l];
            const uint8_t q1  = ql[l + 32];
            a[half * 128 + l]      = (int8_t)(((q0 & 0xF) | ((qhl >> 0) & 3) << 4) - 32);
            a[half * 128 + l + 32] = (int8_t)(((q1 & 0xF) | ((qhl >> 2) & 3) << 4) - 32);
            a[half * 128 + l + 64] = (int8_t)(((q0 >> 4) | ((qhl >> 4) & 3) << 4) - 32);
            a[half * 128 + l + 96] = (int8_t)(((q1 >> 4) | ((qhl >> 6) & 3) << 4) - 32);
        }
        ql += 64;
        qh += 32;
    }
}

static __device__ __forceinline__ float q6_k_dot(const cuda_q6_k_block *blk, const int8_t *xq8,
                                                 const int16_t *bsums, float d_xq, float acc) {
    int8_t a[256];
    q6_k_unpack(blk, a);

    int32_t total = 0;

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int32_t l0 = 0;
        int32_t l1 = 0;
        #pragma unroll
        for (int l = 0; l < 8; l++) {
            l0 += (int)xq8[l] * (int)a[j * 16 + l];
            l1 += (int)xq8[l + 8] * (int)a[j * 16 + l + 8];
        }
        total += (int)blk->scales[j] * (l0 + l1);
        xq8 += 16;
    }
    const float d = __half2float(__ushort_as_half(blk->d));
    return acc + (d * d_xq) * (float)total;
}

__global__ void matmul_q6_k_kernel(const cuda_q6_k_block *__restrict__ w,
                                   const cuda_q8_k_block *__restrict__ xq,
                                   float *__restrict__ y, int n, int k) {
    const int nb  = k >> 8;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q6_k_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q8_k_block *xb = xq + b;
        acc = q6_k_dot(wrow + b, xb->qs, xb->bsums, xb->d, acc);
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc;
}

__global__ void matmul_q6_k_residual_kernel(const cuda_q6_k_block *__restrict__ w,
                                            const cuda_q8_k_block *__restrict__ xq,
                                            const float *__restrict__ residual,
                                            float *__restrict__ y, int n, int k) {
    const int nb  = k >> 8;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q6_k_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES) {
        const cuda_q8_k_block *xb = xq + b;
        acc = q6_k_dot(wrow + b, xb->qs, xb->bsums, xb->d, acc);
    }
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc + residual[row];
}

__global__ void matmul_q6_k_batch_kernel(const cuda_q6_k_block *__restrict__ w,
                                         const cuda_q8_k_block *__restrict__ xq,
                                         float *__restrict__ y, int n, int k, int m) {
    const int nb  = k >> 8;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n)
        return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q6_k_block *wrow = w + (size_t)row * nb;

    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            const cuda_q6_k_block *wb = wrow + b;
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt)
                    break;
                const cuda_q8_k_block *xb = xq + ((size_t)(t0 + t) * nb + b);
                acc[t] = q6_k_dot(wb, xb->qs, xb->bsums, xb->d, acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt)
                break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if ((threadIdx.x & (MM_LANES - 1)) == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

#define MMVQ_WARPS 4

static __device__ __forceinline__ int8_t q6_k_elem(const cuda_q6_k_block *blk, int e) {
    const int      half   = e >> 7;
    const int      within = e & 127;
    const int      grp    = within >> 5;
    const int      l      = within & 31;
    const uint8_t *ql     = blk->ql + half * 64;
    const uint8_t *qh     = blk->qh + half * 32;
    const uint8_t  q0     = ql[l];
    const uint8_t  q1     = ql[l + 32];
    const uint8_t  qhl    = qh[l];
    int v;
    if (grp == 0)      v = ((q0 & 0xF) | (((qhl >> 0) & 3) << 4)) - 32;
    else if (grp == 1) v = ((q1 & 0xF) | (((qhl >> 2) & 3) << 4)) - 32;
    else if (grp == 2) v = ((q0 >> 4)  | (((qhl >> 4) & 3) << 4)) - 32;
    else               v = ((q1 >> 4)  | (((qhl >> 6) & 3) << 4)) - 32;
    return (int8_t)v;
}

__global__ void matmul_q6_k_mmvq_kernel(const cuda_q6_k_block *__restrict__ w,
                                        const cuda_q8_k_block *__restrict__ xq,
                                        float *__restrict__ y, int n, int k) {
    const int nb   = k >> 8;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int row = blockIdx.x * nwarps + wid;
    if (row >= n) return;

    const cuda_q6_k_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;

    for (int b = wid; b < nb; b += nwarps) {
        const cuda_q6_k_block *blk = wrow + b;
        const cuda_q8_k_block *xb  = xq + b;

        int32_t local = 0;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            const int e = lane * 8 + t;
            local += (int)xb->qs[e] * (int)q6_k_elem(blk, e);
        }
        int32_t part = (int)blk->scales[lane >> 1] * local;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            part += __shfl_xor_sync(0xffffffff, part, off);

        if (lane == 0)
            acc = fmaf(__half2float(__ushort_as_half(blk->d)) * xb->d, (float)part, acc);
    }

    if (lane == 0)
        y[row] = acc;
}

__global__ void matmul_iq4_nl_mmvq_kernel(const cuda_iq4_nl_block *__restrict__ w,
                                          const cuda_q8_0_block *__restrict__ xq,
                                          float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const cuda_iq4_nl_block *wrow = w + (size_t)row * nb;

    float acc = 0.0f;
    for (int b = tid; b < nb; b += blockDim.x)
        acc = iq4_nl_dot(wrow + b, xq[b].qs, __half2float(__ushort_as_half(xq[b].d)), acc);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    __shared__ float smem[MMVQ_WARPS];
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0)
        smem[wid] = acc;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < MMVQ_WARPS; i++)
            total += smem[i];
        y[row] = total;
    }
}
extern "C" void cuda_matmul_q6_k(const void *w_dev, const float *x_dev, float *y_dev, int n,
                                  int k, cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 255))
        return;
    cuda_q8_k_block *xq_dev = q8k_scratch_for(k);
    if (!xq_dev)
        return;
    q8_k_quant_kernel<<<(k / 256 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);
    if (n == 1) {

        matmul_q6_k_mmvq_kernel<<<(n + MMVQ_NWARPS - 1) / MMVQ_NWARPS,
                                  MMVQ_NWARPS * 32, 0, stream>>>(
            (const cuda_q6_k_block *)w_dev, xq_dev, y_dev, n, k);
    } else {
        dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        matmul_q6_k_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
            (const cuda_q6_k_block *)w_dev, xq_dev, y_dev, n, k);
    }
}

extern "C" void cuda_matmul_q6_k_residual(const void *w_dev, const float *x_dev,
                                          const float *residual_dev, float *y_dev, int n, int k,
                                          cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 255))
        return;
    cuda_q8_k_block *xq_dev = q8k_scratch_for(k);
    if (!xq_dev)
        return;
    q8_k_quant_kernel<<<(k / 256 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q6_k_residual_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q6_k_block *)w_dev, xq_dev, residual_dev, y_dev, n, k);
}

static __device__ __forceinline__ float q5_0_dot(const cuda_q5_0_block *wb,
                                                  const cuda_q8_0_block *xb, float acc) {
    const uint8_t *qs = wb->qs;
    uint32_t qh = (uint32_t)wb->qh[0] | ((uint32_t)wb->qh[1] << 8) |
                  ((uint32_t)wb->qh[2] << 16) | ((uint32_t)wb->qh[3] << 24);
    int32_t sumi0 = 0, sumi1 = 0;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        const int lo = (int)(qs[j] & 0xF) | (int)((qh >> j) & 1) << 4;
        const int hi = (int)(qs[j] >> 4) | (int)((qh >> (j + 16)) & 1) << 4;
        sumi0 += (lo - 16) * (int)xb->qs[j];
        sumi1 += (hi - 16) * (int)xb->qs[j + 16];
    }
    const float dw = __half2float(__ushort_as_half(wb->d));
    const float dx = __half2float(__ushort_as_half(xb->d));
    return fmaf(dw * dx, (float)(sumi0 + sumi1), acc);
}

static __device__ __forceinline__ float q5_1_dot(const cuda_q5_1_block *wb,
                                                  const cuda_q8_1_block *xb, float acc) {
    const uint8_t *qs = wb->qs;
    uint32_t qh = (uint32_t)wb->qh[0] | ((uint32_t)wb->qh[1] << 8) |
                  ((uint32_t)wb->qh[2] << 16) | ((uint32_t)wb->qh[3] << 24);
    int32_t sumi0 = 0, sumi1 = 0;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        const int lo = (int)(qs[j] & 0xF) | (int)((qh >> j) & 1) << 4;
        const int hi = (int)(qs[j] >> 4) | (int)((qh >> (j + 16)) & 1) << 4;
        sumi0 += lo * (int)xb->qs[j];
        sumi1 += hi * (int)xb->qs[j + 16];
    }
    const float dw = __half2float(__ushort_as_half(wb->d));
    const float mw = __half2float(__ushort_as_half(wb->m));
    const float dx = __half2float(__ushort_as_half(xb->d));
    const float sx = __half2float(__ushort_as_half(xb->s));
    return fmaf(dw * dx, (float)(sumi0 + sumi1), mw * sx + acc);
}

__global__ void matmul_q5_0_kernel(const cuda_q5_0_block *__restrict__ w,
                                   const cuda_q8_0_block *__restrict__ xq,
                                   float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q5_0_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES)
        acc = q5_0_dot(wrow + b, xq + b, acc);
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc;
}

__global__ void matmul_q5_0_residual_kernel(const cuda_q5_0_block *__restrict__ w,
                                            const cuda_q8_0_block *__restrict__ xq,
                                            const float *__restrict__ residual,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q5_0_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES)
        acc = q5_0_dot(wrow + b, xq + b, acc);
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc + residual[row];
}

__global__ void matmul_q5_0_batch_kernel(const cuda_q5_0_block *__restrict__ w,
                                         const cuda_q8_0_block *__restrict__ xq,
                                         float *__restrict__ y, int n, int k, int m) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q5_0_block *wrow = w + (size_t)row * nb;
    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            const cuda_q5_0_block *wb = wrow + b;
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                acc[t] = q5_0_dot(wb, xq + ((size_t)(t0 + t) * nb + b), acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if ((threadIdx.x & (MM_LANES - 1)) == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

__global__ void matmul_q5_1_kernel(const cuda_q5_1_block *__restrict__ w,
                                   const cuda_q8_1_block *__restrict__ xq,
                                   float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q5_1_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES)
        acc = q5_1_dot(wrow + b, xq + b, acc);
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc;
}

__global__ void matmul_q5_1_residual_kernel(const cuda_q5_1_block *__restrict__ w,
                                            const cuda_q8_1_block *__restrict__ xq,
                                            const float *__restrict__ residual,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q5_1_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES)
        acc = q5_1_dot(wrow + b, xq + b, acc);
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc + residual[row];
}

__global__ void matmul_q5_1_batch_kernel(const cuda_q5_1_block *__restrict__ w,
                                         const cuda_q8_1_block *__restrict__ xq,
                                         float *__restrict__ y, int n, int k, int m) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q5_1_block *wrow = w + (size_t)row * nb;
    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            const cuda_q5_1_block *wb = wrow + b;
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                acc[t] = q5_1_dot(wb, xq + ((size_t)(t0 + t) * nb + b), acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if ((threadIdx.x & (MM_LANES - 1)) == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

static __device__ __forceinline__ void get_scale_min_k4_dev(int j, const uint8_t *q,
                                                             uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static __device__ __forceinline__ float q5_k_dot(const cuda_q5_k_block *blk,
                                                 const cuda_q8_k_block *xb, float acc) {
    const float d    = __half2float(__ushort_as_half(blk->d));
    const float dmin = __half2float(__ushort_as_half(blk->dmin));
    const uint8_t *qbytes = blk->qs;
    const uint8_t *qh     = blk->qh;
    const uint8_t *sc     = blk->scales;
    int32_t sumi = 0, summ = 0;
    int is = 0, ib = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int g = 0; g < 4; g++) {
        uint8_t scu8, mu8;
        get_scale_min_k4_dev(is + 0, sc, &scu8, &mu8);
        const int s0 = scu8, m0 = mu8;
        get_scale_min_k4_dev(is + 1, sc, &scu8, &mu8);
        const int s1 = scu8, m1 = mu8;
        const uint8_t *qsg = qbytes + g * 32;
        const int8_t  *xq0 = xb->qs + g * 64;
        const int8_t  *xq1 = xb->qs + g * 64 + 32;
        int32_t dot0 = 0, dot1 = 0;
        #pragma unroll
        for (int l = 0; l < 32; l++) {
            const uint8_t byte = qsg[l];
            const int hi0 = (qh[l] & u1) ? 16 : 0;
            const int hi1 = (qh[l] & u2) ? 16 : 0;
            dot0 += (int)((byte & 0xF) + hi0) * (int)xq0[l];
            dot1 += (int)((byte >> 4) + hi1) * (int)xq1[l];
        }
        sumi += (s0 * dot0) + (s1 * dot1);
        summ += m0 * (int)(xb->bsums[ib] + xb->bsums[ib + 1]);
        ib += 2;
        summ += m1 * (int)(xb->bsums[ib] + xb->bsums[ib + 1]);
        ib += 2;
        is += 2;
        u1 = (uint8_t)(u1 << 2);
        u2 = (uint8_t)(u2 << 2);
    }

    return fmaf(xb->d, (d * (float)sumi) - (dmin * (float)summ), acc);
}

__global__ void matmul_q5_k_kernel(const cuda_q5_k_block *__restrict__ w,
                                   const cuda_q8_k_block *__restrict__ xq,
                                   float *__restrict__ y, int n, int k) {
    const int nb = k >> 8;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q5_k_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES)
        acc = q5_k_dot(wrow + b, xq + b, acc);
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc;
}

__global__ void matmul_q5_k_residual_kernel(const cuda_q5_k_block *__restrict__ w,
                                            const cuda_q8_k_block *__restrict__ xq,
                                            const float *__restrict__ residual,
                                            float *__restrict__ y, int n, int k) {
    const int nb = k >> 8;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q5_k_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES)
        acc = q5_k_dot(wrow + b, xq + b, acc);
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc + residual[row];
}

__global__ void matmul_q5_k_batch_kernel(const cuda_q5_k_block *__restrict__ w,
                                         const cuda_q8_k_block *__restrict__ xq,
                                         float *__restrict__ y, int n, int k, int m) {
    const int nb = k >> 8;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q5_k_block *wrow = w + (size_t)row * nb;
    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            const cuda_q5_k_block *wb = wrow + b;
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                acc[t] = q5_k_dot(wb, xq + ((size_t)(t0 + t) * nb + b), acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if ((threadIdx.x & (MM_LANES - 1)) == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

__constant__ uint32_t cuda_iq3s_grid[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u, 0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u, 0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u, 0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u, 0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u, 0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u, 0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u, 0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu, 0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u, 0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u, 0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u, 0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u, 0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u, 0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u, 0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};

__constant__ uint8_t cuda_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static __device__ __forceinline__ void iq3s_unpack(const cuda_iq3_s_block *b, int8_t *a) {
    const uint8_t *qs    = b->qs;
    const uint8_t *qh    = b->qh;
    const uint8_t *signs = b->signs;
    int8_t       *ap    = a;

    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        for (int l = 0; l < 4; l++) {
            const uint32_t g1 = cuda_iq3s_grid[qs[2 * l] | ((qh[0] << (8 - 2 * l)) & 256)];
            const uint32_t g2 = cuda_iq3s_grid[qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)];
            const uint8_t sb = signs[l];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int8_t v1 = (int8_t)((g1 >> (8 * j)) & 0xFF);
                const int8_t v2 = (int8_t)((g2 >> (8 * j)) & 0xFF);
                ap[j]     = (sb & cuda_kmask_iq2xs[j])     ? (int8_t)-v1 : v1;
                ap[j + 4] = (sb & cuda_kmask_iq2xs[j + 4]) ? (int8_t)-v2 : v2;
            }
            ap += 8;
        }
        qs += 8;
        signs += 4;

        for (int l = 0; l < 4; l++) {
            const uint32_t g1 = cuda_iq3s_grid[qs[2 * l] | ((qh[1] << (8 - 2 * l)) & 256)];
            const uint32_t g2 = cuda_iq3s_grid[qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)];
            const uint8_t sb = signs[l];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int8_t v1 = (int8_t)((g1 >> (8 * j)) & 0xFF);
                const int8_t v2 = (int8_t)((g2 >> (8 * j)) & 0xFF);
                ap[j]     = (sb & cuda_kmask_iq2xs[j])     ? (int8_t)-v1 : v1;
                ap[j + 4] = (sb & cuda_kmask_iq2xs[j + 4]) ? (int8_t)-v2 : v2;
            }
            ap += 8;
        }
        qh += 2;
        qs += 8;
        signs += 4;
    }
}

static __device__ __forceinline__ float iq3_s_dot(const cuda_iq3_s_block *blk,
                                                  const cuda_q8_k_block *xb, float acc) {
    int8_t a[256];
    iq3s_unpack(blk, a);

    const int8_t *q8 = xb->qs;
    int32_t total = 0;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        const uint8_t sb  = blk->scales[g / 2];
        const int      sc = (g & 1) ? (1 + 2 * (sb >> 4)) : (1 + 2 * (sb & 0xF));
        int32_t gs = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            gs += (int)a[g * 32 + j] * (int)q8[g * 32 + j];
        total += sc * gs;
    }
    const float d = __half2float(__ushort_as_half(blk->d));
    return fmaf(d * xb->d, (float)total, acc);
}

__global__ void matmul_iq3_s_kernel(const cuda_iq3_s_block *__restrict__ w,
                                    const cuda_q8_k_block *__restrict__ xq,
                                    float *__restrict__ y, int n, int k) {
    const int nb = k >> 8;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_iq3_s_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES)
        acc = iq3_s_dot(wrow + b, xq + b, acc);
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc;
}

__global__ void matmul_iq3_s_residual_kernel(const cuda_iq3_s_block *__restrict__ w,
                                             const cuda_q8_k_block *__restrict__ xq,
                                             const float *__restrict__ residual,
                                             float *__restrict__ y, int n, int k) {
    const int nb = k >> 8;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_iq3_s_block *wrow = w + (size_t)row * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += MM_LANES)
        acc = iq3_s_dot(wrow + b, xq + b, acc);
    #pragma unroll
    for (int off = MM_LANES >> 1; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & (MM_LANES - 1)) == 0)
        y[row] = acc + residual[row];
}

__global__ void matmul_iq3_s_batch_kernel(const cuda_iq3_s_block *__restrict__ w,
                                          const cuda_q8_k_block *__restrict__ xq,
                                          float *__restrict__ y, int n, int k, int m) {
    const int nb = k >> 8;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_iq3_s_block *wrow = w + (size_t)row * nb;
    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            const cuda_iq3_s_block *wb = wrow + b;
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                acc[t] = iq3_s_dot(wb, xq + ((size_t)(t0 + t) * nb + b), acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if ((threadIdx.x & (MM_LANES - 1)) == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

extern "C" void cuda_matmul_q5_0(const void *w_dev, const float *x_dev, float *y_dev, int n,
                                 int k, cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 31)) return;
    cuda_q8_0_block *xq = xq_scratch_for(k);
    if (!xq) return;
    q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q5_0_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q5_0_block *)w_dev, xq, y_dev, n, k);
}

extern "C" void cuda_matmul_q5_0_residual(const void *w_dev, const float *x_dev,
                                          const float *residual_dev, float *y_dev, int n, int k,
                                          cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 31)) return;
    cuda_q8_0_block *xq = xq_scratch_for(k);
    if (!xq) return;
    q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q5_0_residual_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q5_0_block *)w_dev, xq, residual_dev, y_dev, n, k);
}

extern "C" void cuda_matmul_q5_1(const void *w_dev, const float *x_dev, float *y_dev, int n,
                                 int k, cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 31)) return;
    cuda_q8_1_block *xq = q81_scratch_for(k);
    if (!xq) return;
    q8_1_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q5_1_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q5_1_block *)w_dev, xq, y_dev, n, k);
}

extern "C" void cuda_matmul_q5_1_residual(const void *w_dev, const float *x_dev,
                                          const float *residual_dev, float *y_dev, int n, int k,
                                          cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 31)) return;
    cuda_q8_1_block *xq = q81_scratch_for(k);
    if (!xq) return;
    q8_1_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q5_1_residual_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q5_1_block *)w_dev, xq, residual_dev, y_dev, n, k);
}

extern "C" void cuda_matmul_q5_k(const void *w_dev, const float *x_dev, float *y_dev, int n,
                                 int k, cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 255)) return;
    cuda_q8_k_block *xq = q8k_scratch_for(k);
    if (!xq) return;
    q8_k_quant_kernel<<<(k / 256 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q5_k_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q5_k_block *)w_dev, xq, y_dev, n, k);
}

extern "C" void cuda_matmul_q5_k_residual(const void *w_dev, const float *x_dev,
                                          const float *residual_dev, float *y_dev, int n, int k,
                                          cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 255)) return;
    cuda_q8_k_block *xq = q8k_scratch_for(k);
    if (!xq) return;
    q8_k_quant_kernel<<<(k / 256 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q5_k_residual_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q5_k_block *)w_dev, xq, residual_dev, y_dev, n, k);
}

extern "C" void cuda_matmul_iq3_s(const void *w_dev, const float *x_dev, float *y_dev, int n,
                                  int k, cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 255)) return;
    cuda_q8_k_block *xq = q8k_scratch_for(k);
    if (!xq) return;
    q8_k_quant_kernel<<<(k / 256 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_iq3_s_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_iq3_s_block *)w_dev, xq, y_dev, n, k);
}

extern "C" void cuda_matmul_iq3_s_residual(const void *w_dev, const float *x_dev,
                                           const float *residual_dev, float *y_dev, int n, int k,
                                           cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 255)) return;
    cuda_q8_k_block *xq = q8k_scratch_for(k);
    if (!xq) return;
    q8_k_quant_kernel<<<(k / 256 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_iq3_s_residual_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_iq3_s_block *)w_dev, xq, residual_dev, y_dev, n, k);
}

extern "C" void cuda_matmul_q4_k(const void *w_dev, const float *x_dev, float *y_dev, int n, int k,
                                 cudaStream_t stream) {

    cuda_q8_k_block *xq_dev = q8k_scratch_for(k);
    if (!xq_dev)
        return;
    q8_k_quant_kernel<<<(k / 256 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);
    if (n == 1) {

        matmul_q4_k_mmvq_kernel<<<n, Q4K_MMVQ_NTHREADS, 0, stream>>>(
            (const cuda_q4_k_block *)w_dev, xq_dev, y_dev, n, k);
    } else {

        dim3 grid((n + MM_ROWS_PER_BLOCK * 4 - 1) / (MM_ROWS_PER_BLOCK * 4));
        matmul_q4_k_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
            (const cuda_q4_k_block *)w_dev, xq_dev, y_dev, n, k);
    }
}

extern "C" void cuda_matmul_f32(const float *w_dev, const float *x_dev, float *y_dev, int n, int k,
                                cudaStream_t stream) {
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_f32_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(w_dev, x_dev, y_dev, n, k);
}

extern "C" void cuda_matmul_f16(const uint16_t *w_dev, const float *x_dev, float *y_dev, int n, int k,
                                cudaStream_t stream) {
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_f16_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(w_dev, x_dev, y_dev, n, k);
}

extern "C" void cuda_matmul_bf16(const uint16_t *w_dev, const float *x_dev, float *y_dev, int n, int k,
                                 cudaStream_t stream) {
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_bf16_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(w_dev, x_dev, y_dev, n, k);
}

extern "C" void cuda_matmul_ffn_down(const void *w_dev, const float *gate_dev, const float *up_dev,
                                     float *y_dev, int n, int k, cudaStream_t stream) {
    cuda_q8_0_block *act_q = xq_scratch_for(k);
    if (!act_q)
        return;
    const int nb = k / 32;
    act_gelu_quant_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(gate_dev, up_dev, act_q, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    size_t shmem = (size_t)(k >> 5) * sizeof(cuda_q8_0_block);
    matmul_q8_0_dp4a_v2_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, shmem, stream>>>(
        (const cuda_q8_0_block *)w_dev, act_q, y_dev, n, k, 0, 0);
}

extern "C" void cuda_matmul_ffn_down_q4(const void *w_dev, const float *gate_dev,
                                        const float *up_dev, float *y_dev, int n, int k,
                                        cudaStream_t stream) {
    cuda_q8_0_block *act_q = xq_scratch_for(k);
    if (!act_q)
        return;
    const int nb = k / 32;
    act_gelu_quant_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(gate_dev, up_dev, act_q, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    size_t shmem = (size_t)(k >> 5) * (sizeof(cuda_q8_0_block) + sizeof(int));
    matmul_q4_0_dp4a_v2_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, shmem, stream>>>(
        (const cuda_q4_0_block *)w_dev, act_q, y_dev, n, k, 0);
}

extern "C" void cuda_matmul_ffn_down_q4_1(const void *w_dev, const float *gate_dev,
                                          const float *up_dev, float *y_dev, int n, int k,
                                          cudaStream_t stream) {
    cuda_q8_0_block *act_q = xq_scratch_for(k);
    if (!act_q)
        return;
    const int nb = k / 32;
    act_gelu_quant_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(gate_dev, up_dev, act_q, k);
    if (n == 1) {
        matmul_q4_1_mmvq_kernel<<<1, MMVQ_NTHREADS, 0, stream>>>(
            (const cuda_q4_1_block *)w_dev, act_q, y_dev, n, k);
    } else {
        dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        size_t shmem = (size_t)(k >> 5) * sizeof(cuda_q8_0_block);
        matmul_q4_1_dp4a_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, shmem, stream>>>(
            (const cuda_q4_1_block *)w_dev, act_q, y_dev, n, k);
    }
}

extern "C" void cuda_matmul_multi_q8_0(const void *const *w_dev, float *const *y_dev,
                                       const int *n_host, const float *x_dev, int nm, int k,
                                       cudaStream_t stream) {
    cuda_q8_0_block *xq = xq_scratch_for(k);
    if (!xq)
        return;

    const int use_xqm = 1;
    if (use_xqm)
        q8_0_quant_qm_kernel<<<(unsigned)(k / 32), 32, 0, stream>>>(
            x_dev, (uint8_t *)xq, k);
    else
        q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);

    mm_multi_desc desc;
    if (nm < 1 || nm > 8)
        return;
    int total = 0;
    for (int c = 0; c < nm; c++) {
        desc.w[c] = (const char *)w_dev[c];
        desc.y[c] = y_dev[c];
        desc.n[c] = n_host[c];
        total += n_host[c];
    }
    desc.nm = nm;
    desc.k  = k;

    dim3 grid((total + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_multi_q8_0_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        desc, xq, use_xqm);
}

extern "C" void cuda_matmul_multi_q4_0(const void *const *w_dev, float *const *y_dev,
                                       const int *n_host, const float *x_dev, int nm, int k,
                                       cudaStream_t stream) {
    cuda_q8_0_block *xq = xq_scratch_for(k);
    if (!xq)
        return;
    q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq, k);

    mm_multi_desc desc;
    if (nm < 1 || nm > 8)
        return;
    int total = 0;
    for (int c = 0; c < nm; c++) {
        desc.w[c] = (const char *)w_dev[c];
        desc.y[c] = y_dev[c];
        desc.n[c] = n_host[c];
        total += n_host[c];
    }
    desc.nm = nm;
    desc.k  = k;

    dim3 grid((total + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_multi_q4_0_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(desc, xq);
}

extern "C" void cuda_rmsnorm_matmul_multi_q8_0(const void *const *w_dev, float *const *y_dev,
                                               const int *n_host, const float *x_dev,
                                               const float *norm_w_dev, int nm, int k,
                                               float eps, cudaStream_t stream) {

    mm_multi_desc desc;
    if (nm < 1 || nm > 8)
        return;
    int total = 0;
    for (int c = 0; c < nm; c++) {
        desc.w[c] = (const char *)w_dev[c];
        desc.y[c] = y_dev[c];
        desc.n[c] = n_host[c];
        total += n_host[c];
    }
    desc.nm = nm;
    desc.k  = k;

    const int nt = 256;
    const int nb = k >> 5;
    size_t shmem = (size_t)nt * sizeof(float) + (size_t)nb * sizeof(cuda_q8_0_block);
    dim3 grid((total + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    rmsnorm_matmul_multi_q8_0_kernel<<<grid, nt, shmem, stream>>>(
        desc, x_dev, norm_w_dev, eps);
}

extern "C" void cuda_rmsnorm_matmul_multi_q4_0(const void *const *w_dev, float *const *y_dev,
                                               const int *n_host, const float *x_dev,
                                               const float *norm_w_dev, int nm, int k,
                                               float eps, cudaStream_t stream) {

    mm_multi_desc desc;
    if (nm < 1 || nm > 8)
        return;
    int total = 0;
    for (int c = 0; c < nm; c++) {
        desc.w[c] = (const char *)w_dev[c];
        desc.y[c] = y_dev[c];
        desc.n[c] = n_host[c];
        total += n_host[c];
    }
    desc.nm = nm;
    desc.k  = k;

    const int nt = 256;
    const int nb = k >> 5;
    size_t shmem = (size_t)nt * sizeof(float) + (size_t)nb * sizeof(cuda_q8_0_block);
    dim3 grid((total + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    rmsnorm_matmul_multi_q4_0_kernel<<<grid, nt, shmem, stream>>>(
        desc, x_dev, norm_w_dev, eps);
}

extern "C" void cuda_rmsnorm_matmul_multi_q4_1(const void *const *w_dev, float *const *y_dev,
                                               const int *n_host, const float *x_dev,
                                               const float *norm_w_dev, int nm, int k,
                                               float eps, cudaStream_t stream) {

    mm_multi_desc desc;
    if (nm < 1 || nm > 8)
        return;
    int total = 0;
    for (int c = 0; c < nm; c++) {
        desc.w[c] = (const char *)w_dev[c];
        desc.y[c] = y_dev[c];
        desc.n[c] = n_host[c];
        total += n_host[c];
    }
    desc.nm = nm;
    desc.k  = k;

    const int nt = 256;
    const int nb = k >> 5;
    size_t shmem = (size_t)nt * sizeof(float) + (size_t)nb * sizeof(cuda_q8_0_block);
    dim3 grid((total + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    rmsnorm_matmul_multi_q4_1_kernel<<<grid, nt, shmem, stream>>>(
        desc, x_dev, norm_w_dev, eps);
}

__global__ void attn_f16_kernel(const float *__restrict__ q,
                                const uint16_t *__restrict__ kc,
                                const uint16_t *__restrict__ vc,
                                float *__restrict__ out,
                                int n_heads, int n_kv_heads, int head_dim,
                                size_t layer_base_bytes, size_t kvh_stride, int n_pos, float scale) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if (h >= n_heads) return;

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_k = &smem[head_dim];
    float *s_v = &smem[2 * head_dim];

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q_h[i];
    }
    __syncthreads();

    float M = -INFINITY;
    float S = 0.0f;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0};

    size_t layer_base = layer_base_bytes / sizeof(uint16_t);
    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    for (int t = 0; t < n_pos; t++) {
        const uint16_t *k_t = kc_head + (size_t)t * head_dim;
        const uint16_t *v_t = vc_head + (size_t)t * head_dim;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_k[i] = __half2float(__ushort_as_half(k_t[i]));
        }
        __syncthreads();

        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += s_q[i] * s_k[i];
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            score += __shfl_down_sync(0xffffffff, score, offset);

        float raw_score = score;
        float ms = 1.0f;
        float vs = 1.0f;
        if (lane == 0) {
            raw_score *= scale;
            if (raw_score > M) {
                float old_M = M;
                M = raw_score;
                ms = expf(old_M - M);
                S *= ms;
                vs = 1.0f;
            } else {
                ms = 1.0f;
                vs = expf(raw_score - M);
            }
            S += vs;
        }

        M = __shfl_sync(0xffffffff, M, 0);
        S = __shfl_sync(0xffffffff, S, 0);
        ms = __shfl_sync(0xffffffff, ms, 0);
        vs = __shfl_sync(0xffffffff, vs, 0);

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_v[i] = __half2float(__ushort_as_half(v_t[i]));
        }
        __syncthreads();

        for (int idx = start_idx; idx < end_idx; idx++) {
            int local_idx = idx - start_idx;
            O_local[local_idx] = O_local[local_idx] * ms + vs * s_v[idx];
        }
    }

    float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float *out_h = out + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

__global__ void attn_f16_tiled_kernel(const float *__restrict__ q,
                                       const uint16_t *__restrict__ kc,
                                       const uint16_t *__restrict__ vc,
                                       float *__restrict__ out,
                                       int n_heads, int n_kv_heads, int head_dim,
                                       size_t layer_base_offset_bytes, size_t kvh_stride,
                                       int n_pos, float scale, int tile_size) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if (h >= n_heads) return;

    extern __shared__ float smem[];
    float *s_q = smem;

    float *s_k_tile = &smem[1 * head_dim];
    float *s_v_tile = &smem[1 * head_dim + tile_size * head_dim];

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;
    {
        int chunk = (head_dim + blockDim.x - 1) / blockDim.x;
        int base = tid * chunk;
        int end = min(base + chunk, head_dim);
        for (int i = base; i < end; i++) {
            s_q[i] = q_h[i];
        }
    }
    __syncthreads();

    double M = -INFINITY;
    double S = 0.0;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0.0f};

    size_t layer_base = layer_base_offset_bytes / sizeof(uint16_t);
    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    for (int tile_start = 0; tile_start < n_pos; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, n_pos);
        int tile_len = tile_end - tile_start;

        {
            int tile_pairs = (tile_len * head_dim) >> 1;
            const __half2 *k2 = (const __half2 *)(kc_head + (size_t)tile_start * head_dim);
            const __half2 *v2 = (const __half2 *)(vc_head + (size_t)tile_start * head_dim);
            float2 *s_k2 = (float2 *)s_k_tile;
            float2 *s_v2 = (float2 *)s_v_tile;
            for (int idx = tid; idx < tile_pairs; idx += blockDim.x) {
                s_k2[idx] = __half22float2(k2[idx]);
                s_v2[idx] = __half22float2(v2[idx]);
            }
            if (((tile_len * head_dim) & 1) && tid == 0) {
                int tail = tile_len * head_dim - 1;
                int b = tail / head_dim;
                int i = tail % head_dim;
                const uint16_t *k_t = kc_head + (size_t)(tile_start + b) * head_dim;
                const uint16_t *v_t = vc_head + (size_t)(tile_start + b) * head_dim;
                s_k_tile[tail] = __half2float(__ushort_as_half(k_t[i]));
                s_v_tile[tail] = __half2float(__ushort_as_half(v_t[i]));
            }
        }
        __syncthreads();

        for (int t_local = 0; t_local < tile_len; t_local++) {
            const float *k_tile_row = &s_k_tile[t_local * head_dim];
            const float *v_tile_row = &s_v_tile[t_local * head_dim];

            float score = 0.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                score += s_q[i] * k_tile_row[i];
            }

            for (int offset = 16; offset > 0; offset >>= 1)
                score += __shfl_down_sync(0xffffffff, score, offset);

            double raw_score = (double)score * (double)scale;
            double ms = 1.0;
            double vs = 1.0;
            if (lane == 0) {
                if (raw_score > M) {
                    double old_M = M;
                    M = raw_score;
                    ms = exp(old_M - M);
                    S *= ms;
                    vs = 1.0;
                } else {
                    ms = 1.0;
                    vs = exp(raw_score - M);
                }
                S += vs;
            }

            {
                double tmp_M = M, tmp_S = S, tmp_ms = ms, tmp_vs = vs;
                tmp_M = __shfl_sync(0xffffffff, tmp_M, 0);
                tmp_S = __shfl_sync(0xffffffff, tmp_S, 0);
                tmp_ms = __shfl_sync(0xffffffff, tmp_ms, 0);
                tmp_vs = __shfl_sync(0xffffffff, tmp_vs, 0);
                M = tmp_M; S = tmp_S; ms = tmp_ms; vs = tmp_vs;
            }

            float f_ms = (float)ms;
            float f_vs = (float)vs;
            for (int idx = start_idx; idx < end_idx; idx++) {
                int local_idx = idx - start_idx;
                O_local[local_idx] = O_local[local_idx] * f_ms + f_vs * v_tile_row[idx];
            }
        }
        __syncthreads();
    }

    float inv_S = (S > 0.0) ? (1.0f / (float)S) : 0.0f;
    float *out_h = out + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

__global__ void kv_put_f16_kernel(const float *k_in, const float *v_in,
                                  uint16_t *kd, uint16_t *vd,
                                  int n_kv_heads, int head_dim,
                                  size_t layer_base_bytes, size_t kvh_stride,
                                  int pos, int n_ctx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_kv_heads * head_dim;
    if (idx >= total) return;

    int head = idx / head_dim;
    int dim = idx % head_dim;

    size_t layer_off = layer_base_bytes / sizeof(uint16_t) + (size_t)head * (size_t)n_ctx * head_dim;
    size_t pos_off = layer_off + (size_t)pos * head_dim + dim;

    kd[pos_off] = __half_as_ushort(__float2half_rn(k_in[idx]));
    vd[pos_off] = __half_as_ushort(__float2half_rn(v_in[idx]));
}

extern "C" void cuda_attn_f16(const float *q_dev, const uint16_t *kc_dev,
                              const uint16_t *vc_dev, float *out_dev, int n_heads,
                              int n_kv_heads, int head_dim, size_t layer_base_bytes, size_t kvh_stride,
                              int attn_start, int n_pos_total, float scale, cudaStream_t stream) {
    dim3 grid(n_heads);

    if (n_pos_total >= 256 && head_dim == 256) {
        cuda_attn_f16_decode_split(q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads,
                                   head_dim, layer_base_bytes, kvh_stride, n_pos_total,
                                   scale, stream);
        return;
    }
    if (n_pos_total >= 256) {
        int tile_size = attn_tile_size(head_dim);

        size_t shmem_tiled = (1 * head_dim + 2 * tile_size * head_dim) * sizeof(float);
        size_t layer_base_offset =
            layer_base_bytes + (size_t)attn_start * head_dim * sizeof(uint16_t);
        attn_f16_tiled_kernel<<<grid, 32, shmem_tiled, stream>>>(
            q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
            layer_base_offset, kvh_stride, n_pos_total, scale, tile_size);
    } else {
        (void)attn_start;
        size_t shmem = 3 * head_dim * sizeof(float);
        attn_f16_kernel<<<grid, 32, shmem, stream>>>(
            q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
            layer_base_bytes, kvh_stride, n_pos_total, scale);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: attn_f16_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

__global__ void attn_f16_batch_kernel(const float *__restrict__ q,
                                      const uint16_t *__restrict__ kc,
                                      const uint16_t *__restrict__ vc,
                                      float *__restrict__ out,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      size_t layer_base_bytes, size_t kvh_stride,
                                      int pos_start, int m, int sliding_window, float scale) {
    const int hh = blockIdx.x;
    const int total_h = m * n_heads;
    if (hh >= total_h) return;
    const int r = hh / n_heads;
    const int h = hh % n_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    int n_pos_r = pos_start + r + 1;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_r > sliding_window) {
        attn_start = n_pos_r - sliding_window;
        n_pos_r = sliding_window;
    }

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_k = &smem[head_dim];
    float *s_v = &smem[2 * head_dim];

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const int q_row_elems = n_heads * head_dim;
    const float *q_h = q + (size_t)r * q_row_elems + (size_t)h * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q_h[i];
    }
    __syncthreads();

    float M = -INFINITY;
    float S = 0.0f;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0};

    size_t layer_base = layer_base_bytes / sizeof(uint16_t) + (size_t)attn_start * head_dim;
    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    for (int t = 0; t < n_pos_r; t++) {
        const uint16_t *k_t = kc_head + (size_t)t * head_dim;
        const uint16_t *v_t = vc_head + (size_t)t * head_dim;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_k[i] = __half2float(__ushort_as_half(k_t[i]));
        }
        __syncthreads();

        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += s_q[i] * s_k[i];
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            score += __shfl_down_sync(0xffffffff, score, offset);

        float raw_score = score;
        float ms = 1.0f;
        float vs = 1.0f;
        if (lane == 0) {
            raw_score *= scale;
            if (raw_score > M) {
                float old_M = M;
                M = raw_score;
                ms = expf(old_M - M);
                S *= ms;
                vs = 1.0f;
            } else {
                ms = 1.0f;
                vs = expf(raw_score - M);
            }
            S += vs;
        }

        M = __shfl_sync(0xffffffff, M, 0);
        S = __shfl_sync(0xffffffff, S, 0);
        ms = __shfl_sync(0xffffffff, ms, 0);
        vs = __shfl_sync(0xffffffff, vs, 0);

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_v[i] = __half2float(__ushort_as_half(v_t[i]));
        }
        __syncthreads();

        for (int idx = start_idx; idx < end_idx; idx++) {
            int local_idx = idx - start_idx;
            O_local[local_idx] = O_local[local_idx] * ms + vs * s_v[idx];
        }
    }

    float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float *out_h = out + (size_t)r * q_row_elems + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

__global__ void attn_f16_batch_tiled_kernel(const float *__restrict__ q,
                                             const uint16_t *__restrict__ kc,
                                             const uint16_t *__restrict__ vc,
                                             float *__restrict__ out,
                                             int n_heads, int n_kv_heads, int head_dim,
                                             size_t layer_base_bytes, size_t kvh_stride,
                                             int pos_start, int m,
                                             int sliding_window, float scale, int tile_size) {
    const int hh = blockIdx.x;
    const int total_h = m * n_heads;
    if (hh >= total_h) return;
    const int r = hh / n_heads;
    const int h = hh % n_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    int n_pos_r = pos_start + r + 1;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_r > sliding_window) {
        attn_start = n_pos_r - sliding_window;
        n_pos_r = sliding_window;
    }

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_k = &smem[head_dim];
    float *s_v = &smem[2 * head_dim];
    float *s_k_tile = &smem[3 * head_dim];
    float *s_v_tile = &smem[3 * head_dim + tile_size * head_dim];

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const int q_row_elems = n_heads * head_dim;
    const float *q_h = q + (size_t)r * q_row_elems + (size_t)h * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q_h[i];
    }
    __syncthreads();

    double M = -INFINITY;
    double S = 0.0;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0.0f};

    size_t layer_base = layer_base_bytes / sizeof(uint16_t) + (size_t)attn_start * head_dim;
    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    for (int tile_start = 0; tile_start < n_pos_r; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, n_pos_r);
        int tile_len = tile_end - tile_start;

        for (int b = 0; b < tile_len; b++) {
            const int global_b = tile_start + b;
            const uint16_t *k_t = kc_head + (size_t)global_b * head_dim;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                s_k_tile[b * head_dim + i] = __half2float(__ushort_as_half(k_t[i]));
            }
        }

        for (int b = 0; b < tile_len; b++) {
            const int global_b = tile_start + b;
            const uint16_t *v_t = vc_head + (size_t)global_b * head_dim;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                s_v_tile[b * head_dim + i] = __half2float(__ushort_as_half(v_t[i]));
            }
        }
        __syncthreads();

        for (int t_local = 0; t_local < tile_len; t_local++) {
            const float *k_tile_row = &s_k_tile[t_local * head_dim];
            const float *v_tile_row = &s_v_tile[t_local * head_dim];

            float score = 0.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                score += s_q[i] * k_tile_row[i];
            }

            for (int offset = 16; offset > 0; offset >>= 1)
                score += __shfl_down_sync(0xffffffff, score, offset);

            double raw_score = (double)score * (double)scale;
            double ms = 1.0;
            double vs = 1.0;
            if (lane == 0) {
                if (raw_score > M) {
                    double old_M = M;
                    M = raw_score;
                    ms = exp(old_M - M);
                    S *= ms;
                    vs = 1.0;
                } else {
                    ms = 1.0;
                    vs = exp(raw_score - M);
                }
                S += vs;
            }

            {
                double tmp_M = M, tmp_S = S, tmp_ms = ms, tmp_vs = vs;
                tmp_M = __shfl_sync(0xffffffff, tmp_M, 0);
                tmp_S = __shfl_sync(0xffffffff, tmp_S, 0);
                tmp_ms = __shfl_sync(0xffffffff, tmp_ms, 0);
                tmp_vs = __shfl_sync(0xffffffff, tmp_vs, 0);
                M = tmp_M; S = tmp_S; ms = tmp_ms; vs = tmp_vs;
            }

            float f_ms = (float)ms;
            float f_vs = (float)vs;
            for (int idx = start_idx; idx < end_idx; idx++) {
                int local_idx = idx - start_idx;
                O_local[local_idx] = O_local[local_idx] * f_ms + f_vs * v_tile_row[idx];
            }
        }
        __syncthreads();
    }

    float inv_S = (S > 0.0) ? (1.0f / (float)S) : 0.0f;
    float *out_h = out + (size_t)r * q_row_elems + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

#define FLASH_QTILE 8
#define FLASH_KTILE 16
__global__ void attn_f16_batch_flash_kernel(const float *__restrict__ q,
                                            const uint16_t *__restrict__ kc,
                                            const uint16_t *__restrict__ vc,
                                            float *__restrict__ out,
                                            int n_heads, int n_kv_heads, int head_dim,
                                            size_t layer_base_bytes, size_t kvh_stride,
                                            int pos_start, int m, float scale);

extern "C" void cuda_attn_batch_f16(const float *q_dev, const uint16_t *kc_dev,
                                    const uint16_t *vc_dev, float *out_dev, int n_heads,
                                    int n_kv_heads, int head_dim, size_t layer_base_bytes,
                                    size_t kvh_stride, int pos_start, int m,
                                    int sliding_window, float scale, cudaStream_t stream) {
    dim3 grid((unsigned)m * (unsigned)n_heads);

    if (head_dim == 256 && sliding_window == 0) {
        dim3 grid(((unsigned)m + FLASH_QTILE - 1) / FLASH_QTILE, (unsigned)n_heads);
        size_t shmem = 2 * (size_t)FLASH_KTILE * head_dim * sizeof(__half);
        attn_f16_batch_flash_kernel<<<grid, 256, shmem, stream>>>(
            q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
            layer_base_bytes, kvh_stride, pos_start, m, scale);
        cudaError_t flash_err = cudaGetLastError();
        if (flash_err != cudaSuccess)
            ERROR("cuda: flash attn launch failed: %s", cudaGetErrorString(flash_err));
        return;
    }
        int n_pos_est = pos_start + m;
    if (sliding_window > 0 && n_pos_est > sliding_window)
        n_pos_est = sliding_window;
 else {
        size_t shmem = 3 * head_dim * sizeof(float);
        attn_f16_batch_kernel<<<grid, 32, shmem, stream>>>(
            q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
            layer_base_bytes, kvh_stride, pos_start, m, sliding_window, scale);
    }
}

__global__ void attn_f16_batch_flash_kernel(const float *__restrict__ q,
                                            const uint16_t *__restrict__ kc,
                                            const uint16_t *__restrict__ vc,
                                            float *__restrict__ out,
                                            int n_heads, int n_kv_heads, int head_dim,
                                            size_t layer_base_bytes, size_t kvh_stride,
                                            int pos_start, int m, float scale) {
    const int h = blockIdx.y;
    const int qr = blockIdx.x * FLASH_QTILE;
    if (h >= n_heads) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= FLASH_QTILE) return;

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    int row = qr + warp;
    int row_active = (row < m);
    int n_pos_r = row_active ? pos_start + row + 1 : 0;

    const int q_row_elems = n_heads * head_dim;
    const float *q_h = q + (size_t)(row_active ? row : 0) * q_row_elems + (size_t)h * head_dim;

    extern __shared__ __half smem_flash[];
    __half *s_k = smem_flash;
    __half *s_v = smem_flash + (size_t)FLASH_KTILE * head_dim;

    float M = -INFINITY, S = 0.0f;
    float O[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    size_t layer_base = layer_base_bytes / sizeof(uint16_t);
    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    int rmax = qr + FLASH_QTILE - 1;
    if (rmax >= m) rmax = m - 1;
    int n_pos_max = pos_start + rmax + 1;
    int tiles_max = (n_pos_max + FLASH_KTILE - 1) / FLASH_KTILE;

    for (int tile = 0; tile < tiles_max; tile++) {

        for (int i = threadIdx.x; i < FLASH_KTILE * head_dim; i += blockDim.x) {
            int t = i / head_dim, j = i % head_dim;
            int pos = tile * FLASH_KTILE + t;
            __half kv = __float2half(0.0f);
            __half vv = __float2half(0.0f);
            if (pos < n_pos_max) {
                kv = ((const __half *)kc_head)[(size_t)pos * head_dim + j];
                vv = ((const __half *)vc_head)[(size_t)pos * head_dim + j];
            }
            s_k[(size_t)t * head_dim + j] = kv;
            s_v[(size_t)t * head_dim + j] = vv;
        }
        __syncthreads();

        if (row_active) {
            for (int t = 0; t < FLASH_KTILE; t++) {
                int pos = tile * FLASH_KTILE + t;
                if (pos >= n_pos_r) break;
                float score = 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int j = lane * 8 + e;
                    score += q_h[j] * __half2float(s_k[(size_t)t * head_dim + j]);
                }

                for (int off = 16; off > 0; off >>= 1)
                    score += __shfl_xor_sync(0xffffffff, score, off);
                float raw = score * scale;
                float old_M = M;
                float m_new = raw > old_M ? raw : old_M;
                float a = expf(old_M - m_new);
                float b = expf(raw - m_new);
                M = m_new;
                S = S * a + b;
                float wgt = b;
                float resc = a;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int j = lane * 8 + e;
                    O[e] = O[e] * resc +
                           wgt * __half2float(s_v[(size_t)t * head_dim + j]);
                }
            }
        }
        __syncthreads();
    }

    if (row_active) {
        float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
        float *out_h = out + (size_t)row * q_row_elems + (size_t)h * head_dim;
        #pragma unroll
        for (int e = 0; e < 8; e++)
            out_h[lane * 8 + e] = O[e] * inv_S;
    }
}

#define KV_Q8_0_ATTN_BLK_BYTES 34

static __device__ __forceinline__ float kv_q8_dequant_elem(const uint8_t *row_base, int i) {
    int b = i >> 5;
    const uint8_t *blk = row_base + (size_t)b * KV_Q8_0_ATTN_BLK_BYTES;
    uint16_t dbits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    float d = __half2float(__ushort_as_half(dbits));
    return d * (float)((const int8_t *)(blk + 2))[i & 31];
}

__global__ void attn_q8_kernel(const float *__restrict__ q,
                               const uint8_t *__restrict__ kc,
                               const uint8_t *__restrict__ vc,
                               float *__restrict__ out,
                               int n_heads, int n_kv_heads, int head_dim,
                               size_t layer_base_bytes, size_t kvh_stride_bytes,
                               int n_pos, float scale, int n_blocks) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if (h >= n_heads) return;

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_k = &smem[head_dim];
    float *s_v = &smem[2 * head_dim];

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q_h[i];
    }
    __syncthreads();

    float M = -INFINITY;
    float S = 0.0f;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0};

    size_t row_bytes = (size_t)n_blocks * KV_Q8_0_ATTN_BLK_BYTES;
    const uint8_t *kc_head = kc + layer_base_bytes + (size_t)kv_head * kvh_stride_bytes;
    const uint8_t *vc_head = vc + layer_base_bytes + (size_t)kv_head * kvh_stride_bytes;

    for (int t = 0; t < n_pos; t++) {
        const uint8_t *k_t = kc_head + (size_t)t * row_bytes;
        const uint8_t *v_t = vc_head + (size_t)t * row_bytes;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_k[i] = kv_q8_dequant_elem(k_t, i);
        }
        __syncthreads();

        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += s_q[i] * s_k[i];
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            score += __shfl_down_sync(0xffffffff, score, offset);

        float raw_score = score;
        float ms = 1.0f;
        float vs = 1.0f;
        if (lane == 0) {
            raw_score *= scale;
            if (raw_score > M) {
                float old_M = M;
                M = raw_score;
                ms = expf(old_M - M);
                S *= ms;
                vs = 1.0f;
            } else {
                ms = 1.0f;
                vs = expf(raw_score - M);
            }
            S += vs;
        }

        M = __shfl_sync(0xffffffff, M, 0);
        S = __shfl_sync(0xffffffff, S, 0);
        ms = __shfl_sync(0xffffffff, ms, 0);
        vs = __shfl_sync(0xffffffff, vs, 0);

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_v[i] = kv_q8_dequant_elem(v_t, i);
        }
        __syncthreads();

        for (int idx = start_idx; idx < end_idx; idx++) {
            int local_idx = idx - start_idx;
            O_local[local_idx] = O_local[local_idx] * ms + vs * s_v[idx];
        }
    }

    float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float *out_h = out + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

__global__ void attn_q8_tiled_kernel(const float *__restrict__ q,
                                      const uint8_t *__restrict__ kc,
                                      const uint8_t *__restrict__ vc,
                                      float *__restrict__ out,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      size_t layer_base_offset_bytes, size_t kvh_stride_bytes,
                                      int n_pos, float scale, int n_blocks, int tile_size) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if (h >= n_heads) return;

    size_t row_bytes = (size_t)n_blocks * KV_Q8_0_ATTN_BLK_BYTES;

    extern __shared__ float smem[];
    float *s_q = smem;

    uint8_t *s_raw_k = (uint8_t *)&smem[1 * head_dim];
    uint8_t *s_raw_v = (uint8_t *)&smem[1 * head_dim] + (size_t)tile_size * row_bytes;
    float *s_k_tile = (float *)((uint8_t *)&smem[1 * head_dim] +
                                2 * (size_t)tile_size * row_bytes);
    float *s_v_tile = s_k_tile + (size_t)tile_size * head_dim;

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;
    {
        int chunk = (head_dim + blockDim.x - 1) / blockDim.x;
        int base = tid * chunk;
        int end = min(base + chunk, head_dim);
        for (int i = base; i < end; i++) {
            s_q[i] = q_h[i];
        }
    }
    __syncthreads();

    double M = -INFINITY;
    double S = 0.0;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0.0f};

    const uint8_t *kc_head = kc + layer_base_offset_bytes + (size_t)kv_head * kvh_stride_bytes;
    const uint8_t *vc_head = vc + layer_base_offset_bytes + (size_t)kv_head * kvh_stride_bytes;

    for (int tile_start = 0; tile_start < n_pos; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, n_pos);
        int tile_len = tile_end - tile_start;

        {
            size_t tile_raw = (size_t)tile_len * row_bytes;
            const uint8_t *kbase = kc_head + (size_t)tile_start * row_bytes;
            const uint8_t *vbase = vc_head + (size_t)tile_start * row_bytes;
            for (size_t by = tid; by < tile_raw; by += blockDim.x) {
                s_raw_k[by] = kbase[by];
                s_raw_v[by] = vbase[by];
            }
        }
        __syncthreads();
        for (int b = 0; b < tile_len; b++) {
            const uint8_t *k_t = s_raw_k + (size_t)b * row_bytes;
            const uint8_t *v_t = s_raw_v + (size_t)b * row_bytes;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                s_k_tile[b * head_dim + i] = kv_q8_dequant_elem(k_t, i);
                s_v_tile[b * head_dim + i] = kv_q8_dequant_elem(v_t, i);
            }
        }
        __syncthreads();

        for (int t_local = 0; t_local < tile_len; t_local++) {
            const float *k_tile_row = &s_k_tile[t_local * head_dim];
            const float *v_tile_row = &s_v_tile[t_local * head_dim];

            float score = 0.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                score += s_q[i] * k_tile_row[i];
            }

            for (int offset = 16; offset > 0; offset >>= 1)
                score += __shfl_down_sync(0xffffffff, score, offset);

            double raw_score = (double)score * (double)scale;
            double ms = 1.0;
            double vs = 1.0;
            if (lane == 0) {
                if (raw_score > M) {
                    double old_M = M;
                    M = raw_score;
                    ms = exp(old_M - M);
                    S *= ms;
                    vs = 1.0;
                } else {
                    ms = 1.0;
                    vs = exp(raw_score - M);
                }
                S += vs;
            }

            {
                double tmp_M = M, tmp_S = S, tmp_ms = ms, tmp_vs = vs;
                tmp_M = __shfl_sync(0xffffffff, tmp_M, 0);
                tmp_S = __shfl_sync(0xffffffff, tmp_S, 0);
                tmp_ms = __shfl_sync(0xffffffff, tmp_ms, 0);
                tmp_vs = __shfl_sync(0xffffffff, tmp_vs, 0);
                M = tmp_M; S = tmp_S; ms = tmp_ms; vs = tmp_vs;
            }

            float f_ms = (float)ms;
            float f_vs = (float)vs;
            for (int idx = start_idx; idx < end_idx; idx++) {
                int local_idx = idx - start_idx;
                O_local[local_idx] = O_local[local_idx] * f_ms + f_vs * v_tile_row[idx];
            }
        }
        __syncthreads();
    }

    float inv_S = (S > 0.0) ? (1.0f / (float)S) : 0.0f;
    float *out_h = out + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

extern "C" void cuda_attn_q8(const float *q_dev, const uint8_t *kc_dev,
                              const uint8_t *vc_dev, float *out_dev, int n_heads,
                              int n_kv_heads, int head_dim, size_t layer_base_bytes,
                              size_t kvh_stride_bytes, int attn_start, int n_pos_total,
                              float scale, int n_blocks, cudaStream_t stream) {
    dim3 grid(n_heads);

    if (n_pos_total >= 256) {
        int tile_size = attn_tile_size(head_dim);

        size_t row_bytes_q = (size_t)n_blocks * KV_Q8_0_ATTN_BLK_BYTES;
        size_t shmem = (size_t)head_dim * sizeof(float) +
                       2 * (size_t)tile_size * row_bytes_q +
                       2 * (size_t)tile_size * head_dim * sizeof(float);
        size_t layer_base_offset =
            layer_base_bytes + (size_t)attn_start * n_blocks * KV_Q8_0_ATTN_BLK_BYTES;
        attn_q8_tiled_kernel<<<grid, 32, shmem, stream>>>(
            q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
            layer_base_offset, kvh_stride_bytes, n_pos_total, scale, n_blocks, tile_size);
    } else {
        (void)attn_start;
        size_t shmem = 3 * head_dim * sizeof(float);
        attn_q8_kernel<<<grid, 32, shmem, stream>>>(
            q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
            layer_base_bytes, kvh_stride_bytes, n_pos_total, scale, n_blocks);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: attn_q8_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

__global__ void attn_q8_batch_kernel(const float *__restrict__ q,
                                      const uint8_t *__restrict__ kc,
                                      const uint8_t *__restrict__ vc,
                                      float *__restrict__ out,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      size_t layer_base_bytes, size_t kvh_stride_bytes,
                                      int pos_start, int m, int sliding_window,
                                      float scale, int n_blocks) {
    const int hh = blockIdx.x;
    const int total_h = m * n_heads;
    if (hh >= total_h) return;
    const int r = hh / n_heads;
    const int h = hh % n_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    int n_pos_r = pos_start + r + 1;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_r > sliding_window) {
        attn_start = n_pos_r - sliding_window;
        n_pos_r = sliding_window;
    }

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_k = &smem[head_dim];
    float *s_v = &smem[2 * head_dim];

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const int q_row_elems = n_heads * head_dim;
    const float *q_h = q + (size_t)r * q_row_elems + (size_t)h * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q_h[i];
    }
    __syncthreads();

    float M = -INFINITY;
    float S = 0.0f;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0};

    size_t row_bytes = (size_t)n_blocks * KV_Q8_0_ATTN_BLK_BYTES;
    size_t layer_base = layer_base_bytes + (size_t)attn_start * row_bytes;
    const uint8_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride_bytes;
    const uint8_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride_bytes;

    for (int t = 0; t < n_pos_r; t++) {
        const uint8_t *k_t = kc_head + (size_t)t * row_bytes;
        const uint8_t *v_t = vc_head + (size_t)t * row_bytes;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_k[i] = kv_q8_dequant_elem(k_t, i);
        }
        __syncthreads();

        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += s_q[i] * s_k[i];
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            score += __shfl_down_sync(0xffffffff, score, offset);

        float raw_score = score;
        float ms = 1.0f;
        float vs = 1.0f;
        if (lane == 0) {
            raw_score *= scale;
            if (raw_score > M) {
                float old_M = M;
                M = raw_score;
                ms = expf(old_M - M);
                S *= ms;
                vs = 1.0f;
            } else {
                ms = 1.0f;
                vs = expf(raw_score - M);
            }
            S += vs;
        }

        M = __shfl_sync(0xffffffff, M, 0);
        S = __shfl_sync(0xffffffff, S, 0);
        ms = __shfl_sync(0xffffffff, ms, 0);
        vs = __shfl_sync(0xffffffff, vs, 0);

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_v[i] = kv_q8_dequant_elem(v_t, i);
        }
        __syncthreads();

        for (int idx = start_idx; idx < end_idx; idx++) {
            int local_idx = idx - start_idx;
            O_local[local_idx] = O_local[local_idx] * ms + vs * s_v[idx];
        }
    }

    float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float *out_h = out + (size_t)r * q_row_elems + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

extern "C" void cuda_attn_batch_q8(const float *q_dev, const uint8_t *kc_dev,
                                    const uint8_t *vc_dev, float *out_dev, int n_heads,
                                    int n_kv_heads, int head_dim, size_t layer_base_bytes,
                                    size_t kvh_stride_bytes, int pos_start, int m,
                                    int sliding_window, float scale, int n_blocks,
                                    cudaStream_t stream) {
    dim3 grid((unsigned)m * (unsigned)n_heads);

    size_t shmem = 3 * head_dim * sizeof(float);
    attn_q8_batch_kernel<<<grid, 32, shmem, stream>>>(
        q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride_bytes, pos_start, m, sliding_window, scale, n_blocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: attn_q8_batch_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

__global__ void kv_put_f16_batch_kernel(const float *k_in, const float *v_in,
                                        uint16_t *kd, uint16_t *vd,
                                        int n_kv_heads, int head_dim,
                                        size_t layer_base_bytes, size_t kvh_stride,
                                        int pos_start, int n_ctx, int in_row_stride, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int per_row = n_kv_heads * head_dim;
    if (idx >= m * per_row) return;
    int row = idx / per_row;
    int elem = idx % per_row;
    int head = elem / head_dim;
    int dim = elem % head_dim;
    int pos = pos_start + row;

    size_t layer_off = layer_base_bytes / sizeof(uint16_t) + (size_t)head * (size_t)n_ctx * head_dim;
    size_t pos_off = layer_off + (size_t)pos * head_dim + dim;

    const float *k_row = k_in + (size_t)row * in_row_stride;
    const float *v_row = v_in + (size_t)row * in_row_stride;

    kd[pos_off] = __half_as_ushort(__float2half_rn(k_row[elem]));
    vd[pos_off] = __half_as_ushort(__float2half_rn(v_row[elem]));
}

extern "C" void cuda_kv_put_batch_f16(const float *k_in, const float *v_in,
                                      uint16_t *kd, uint16_t *vd, int n_kv_heads, int head_dim,
                                      size_t layer_base_bytes, size_t kvh_stride, int pos_start,
                                      int n_ctx, int in_row_stride, int m, cudaStream_t stream) {
    int total = m * n_kv_heads * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    kv_put_f16_batch_kernel<<<grid, block, 0, stream>>>(k_in, v_in, kd, vd, n_kv_heads, head_dim,
                                                        layer_base_bytes, kvh_stride, pos_start,
                                                        n_ctx, in_row_stride, m);
}

extern "C" void cuda_kv_put_f16(const float *k_in, const float *v_in,
                                uint16_t *kd, uint16_t *vd, int n_kv_heads, int head_dim,
                                size_t layer_base_bytes, size_t kvh_stride, int pos, int n_ctx,
                                cudaStream_t stream) {
    int total = n_kv_heads * head_dim;
    int block = 256;
    dim3 grid((total + block - 1) / block);
    kv_put_f16_kernel<<<grid, block, 0, stream>>>(k_in, v_in, kd, vd, n_kv_heads, head_dim,
                                                   layer_base_bytes, kvh_stride, pos, n_ctx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: kv_put_f16_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

extern "C" void cuda_kv_alloc_f16(uint16_t **kd_out, uint16_t **vd_out, size_t size, cudaStream_t stream) {
    size_t bytes = size * sizeof(uint16_t);
    cudaMallocAsync((void**)kd_out, bytes, stream);
    cudaMallocAsync((void**)vd_out, bytes, stream);
}

__global__ void add_inplace_kernel(float *x, const float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += y[i];
}

extern "C" void cuda_add_inplace(float *x_dev, const float *y_dev, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_inplace_kernel<<<grid, block, 0, stream>>>(x_dev, y_dev, n);
}

__global__ void scale_inplace_kernel(float *x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

extern "C" void cuda_scale_inplace(float *x_dev, float scale, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_inplace_kernel<<<grid, block, 0, stream>>>(x_dev, scale, n);
}

__global__ void softcap_kernel(float *x, float cap, float inv_cap, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] * inv_cap);
}

extern "C" void cuda_softcap(float *x_dev, float cap, long long n, cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    long long grid = (n + block - 1) / block;
    softcap_kernel<<<(unsigned)grid, block, 0, stream>>>(x_dev, cap, 1.0f / cap, n);
}

__global__ void attn_output_gate_kernel(float *o, const float *g, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] *= 1.0f / (1.0f + expf(-g[i]));
}

__global__ void partial_rope_kernel(float *vec, int n_heads, int head_dim, int rope_dim,
                                    const float *cos_base, const float *sin_base,
                                    int pos_start, int n_rows) {
    int half = rope_dim / 2;
    int per_row = n_heads * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * per_row) return;
    int row = idx / per_row;
    int within = idx - row * per_row;
    int h = within / half;
    int j = within - h * half;
    float *vh = vec + ((size_t)row * n_heads + h) * head_dim;
    float c = cos_base[(size_t)(pos_start + row) * half + j];
    float s = sin_base[(size_t)(pos_start + row) * half + j];
    float v0 = vh[j];
    float v1 = vh[j + half];
    vh[j]        = v0 * c - v1 * s;
    vh[j + half] = v0 * s + v1 * c;
}

extern "C" void cuda_partial_rope(float *vec_dev, int n_heads, int head_dim, int rope_dim,
                                  int pos_start, const float *cos_dev, const float *sin_dev,
                                  int n_rows, cudaStream_t stream) {
    int half = rope_dim / 2;
    if (n_rows <= 0 || n_heads <= 0 || half <= 0) return;
    int total = n_rows * n_heads * half;
    int block = 256;
    int grid = (total + block - 1) / block;
    partial_rope_kernel<<<grid, block, 0, stream>>>(vec_dev, n_heads, head_dim, rope_dim,
                                                    cos_dev, sin_dev, pos_start, n_rows);
}

extern "C" void cuda_attn_output_gate(float *out_dev, const float *gate_dev, long long n,
                                      cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    long long grid = (n + block - 1) / block;
    attn_output_gate_kernel<<<(unsigned)grid, block, 0, stream>>>(out_dev, gate_dev, n);
}

__global__ void split_qgate_kernel(const float *mixed, float *q, float *gate, int head_dim,
                                   long long q_out, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    long long row = i / q_out;
    long long rem = i - row * q_out;
    long long h   = rem / head_dim;
    long long d   = rem - h * head_dim;
    const float *src = mixed + row * 2 * q_out + h * 2 * head_dim;
    q[row * q_out + rem]    = src[d];
    gate[row * q_out + rem] = src[d + head_dim];
}

extern "C" void cuda_split_qgate(const float *mixed_dev, float *q_dev, float *gate_dev,
                                int n_heads, int head_dim, int n_rows, cudaStream_t stream) {
    if (n_rows <= 0 || n_heads <= 0 || head_dim <= 0) return;
    long long q_out = (long long)n_heads * head_dim;
    long long total = q_out * n_rows;
    int block = 256;
    long long grid = (total + block - 1) / block;
    split_qgate_kernel<<<(unsigned)grid, block, 0, stream>>>(mixed_dev, q_dev, gate_dev, head_dim,
                                                             q_out, total);
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu_tanh_f32(float x) {
    const float c  = 0.7978845608028654f;
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
}

__global__ void moe_activate_kernel(const float *gate, const float *up, float *out, long long n,
                                    float gate_scale, float up_scale, int use_gelu) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i] * gate_scale;
    float u = up[i] * up_scale;
    float a = use_gelu ? gelu_tanh_f32(g) : silu_f32(g);
    out[i] = a * u;
}

extern "C" void cuda_moe_activate(const float *gate_dev, const float *up_dev, float *out_dev,
                                  long long n, float gate_scale, float up_scale, int use_gelu,
                                  cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    long long grid = (n + block - 1) / block;
    moe_activate_kernel<<<(unsigned)grid, block, 0, stream>>>(gate_dev, up_dev, out_dev, n,
                                                              gate_scale, up_scale, use_gelu);
}

__global__ void ffn_activate_kernel(const float *gate, const float *up, float *out,
                                    int n, int activation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (activation == 1)
        out[i] = gelu_tanh_f32(g) * u;
    else
        out[i] = silu_f32(g) * u;
}

extern "C" void cuda_ffn_activate(const float *gate_dev, const float *up_dev, float *out_dev,
                                   int n, int activation, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    ffn_activate_kernel<<<grid, block, 0, stream>>>(gate_dev, up_dev, out_dev, n, activation);
}

__global__ void ffn_activate_batch_kernel(const float *gate, const float *up, float *out,
                                          int n, int activation, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) return;
    float g = gate[idx];
    float u = up[idx];
    if (activation == 1)
        out[idx] = gelu_tanh_f32(g) * u;
    else
        out[idx] = silu_f32(g) * u;
}

extern "C" void cuda_ffn_activate_batch(const float *gate_dev, const float *up_dev, float *out_dev,
                                        int n, int activation, int m, cudaStream_t stream) {
    int block = 256;
    int total = m * n;
    int grid = (total + block - 1) / block;
    ffn_activate_batch_kernel<<<grid, block, 0, stream>>>(gate_dev, up_dev, out_dev, n, activation, m);
}

__global__ void add_batch_kernel(float *x, const float *y, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    x[idx] += y[idx];
}

__global__ void ffn_activate_fused_batch_kernel(const float *fused, float *out, int n,
                                                 int activation, int m) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * m)
        return;
    const int row = idx / n;
    const int col = idx - row * n;
    const float *g = fused + (size_t)row * 2 * n + col;
    const float *u = g + n;
    const float  gv = *g;
    const float  uv = *u;
    out[idx] = (activation == 1) ? gelu_tanh_f32(gv) * uv : silu_f32(gv) * uv;
}

extern "C" void cuda_ffn_activate_fused_batch(const float *fused_dev, float *out_dev, int n,
                                               int activation, int m, cudaStream_t stream) {
    if (n <= 0 || m <= 0)
        return;
    const int total = n * m;
    const int block = 256;
    ffn_activate_fused_batch_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        fused_dev, out_dev, n, activation, m);
}

extern "C" void cuda_add_batch(float *x_dev, const float *y_dev, int n, int m, cudaStream_t stream) {
    int block = 256;
    int total = m * n;
    int grid = (total + block - 1) / block;
    add_batch_kernel<<<grid, block, 0, stream>>>(x_dev, y_dev, n, m);
}

__global__ void rope_ext_kernel(float *vec, int n_heads, int head_dim,
                                const float *cos_tbl, const float *sin_tbl,
                                int neox) {
    int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * half;
    if (idx >= total) return;

    int h = idx / half;
    int j = idx % half;
    float *vh = vec + (size_t)h * head_dim;
    float c = cos_tbl[j];
    float s = sin_tbl[j];

    if (neox) {
        float v0 = vh[j];
        float v1 = vh[j + half];
        vh[j]        = v0 * c - v1 * s;
        vh[j + half] = v0 * s + v1 * c;
    } else {
        float v0 = vh[2 * j];
        float v1 = vh[2 * j + 1];
        vh[2 * j]     = v0 * c - v1 * s;
        vh[2 * j + 1] = v0 * s + v1 * c;
    }
}

extern "C" void cuda_rope_ext(float *vec_dev, int n_heads, int head_dim, int pos,
                               const float *rope_cos_dev, const float *rope_sin_dev,
                               int neox, cudaStream_t stream) {
    int half = head_dim / 2;
    int total = n_heads * half;
    int block = 256;
    int grid = (total + block - 1) / block;

    const float *cos_tbl = rope_cos_dev + (size_t)pos * half;
    const float *sin_tbl = rope_sin_dev + (size_t)pos * half;
    rope_ext_kernel<<<grid, block, 0, stream>>>(vec_dev, n_heads, head_dim,
                                                 cos_tbl, sin_tbl, neox);
}

__global__ void rope_ext_batch_kernel(float *vec, int n_heads, int head_dim,
                                      const float *cos_base, const float *sin_base,
                                      int pos_start, int m, int neox) {
    int half = head_dim / 2;
    int per_row = n_heads * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * per_row) return;
    int row = idx / per_row;
    int within = idx % per_row;
    int h = within / half;
    int j = within % half;
    float *vh = vec + ((size_t)row * n_heads + h) * head_dim;
    float c = cos_base[(size_t)(pos_start + row) * half + j];
    float s = sin_base[(size_t)(pos_start + row) * half + j];

    if (neox) {
        float v0 = vh[j];
        float v1 = vh[j + half];
        vh[j]        = v0 * c - v1 * s;
        vh[j + half] = v0 * s + v1 * c;
    } else {
        float v0 = vh[2 * j];
        float v1 = vh[2 * j + 1];
        vh[2 * j]     = v0 * c - v1 * s;
        vh[2 * j + 1] = v0 * s + v1 * c;
    }
}

extern "C" void cuda_rope_batch(float *vec_dev, int n_heads, int head_dim, int pos_start,
                                const float *rope_cos_base, const float *rope_sin_base,
                                int m, int neox, cudaStream_t stream) {
    int half = head_dim / 2;
    int total = m * n_heads * half;
    int block = 256;
    int grid = (total + block - 1) / block;
    rope_ext_batch_kernel<<<grid, block, 0, stream>>>(vec_dev, n_heads, head_dim,
                                                      rope_cos_base, rope_sin_base,
                                                      pos_start, m, neox);
}

extern "C" void cuda_rope_qk_batch(float *q_dev, float *k_dev, int n_heads, int n_kv_heads,
                                   int head_dim, int pos_start,
                                   const float *rope_cos_base, const float *rope_sin_base,
                                   int m, int neox, cudaStream_t stream) {
    int half = head_dim / 2;
    int block = 256;
    int total_q = m * n_heads * half;
    int grid_q = (total_q + block - 1) / block;
    rope_ext_batch_kernel<<<grid_q, block, 0, stream>>>(q_dev, n_heads, head_dim,
                                                        rope_cos_base, rope_sin_base,
                                                        pos_start, m, neox);
    int total_k = m * n_kv_heads * half;
    int grid_k = (total_k + block - 1) / block;
    rope_ext_batch_kernel<<<grid_k, block, 0, stream>>>(k_dev, n_kv_heads, head_dim,
                                                        rope_cos_base, rope_sin_base,
                                                        pos_start, m, neox);
}

__global__ void embd_lookup_f32_kernel(const float *embd, float *out, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim)
        out[i] = embd[(size_t)token * dim + i];
}

__global__ void embd_lookup_f16_kernel(const uint16_t *embd, float *out, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim)
        out[i] = __half2float(__ushort_as_half(embd[(size_t)token * dim + i]));
}

__global__ void embd_lookup_bf16_kernel(const uint16_t *embd, float *out, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim)
        out[i] = (float)__ushort_as_bfloat16(embd[(size_t)token * dim + i]);
}

__global__ void embd_lookup_q8_0_kernel(const cuda_q8_0_block *embd, float *out,
                                         int token, int dim) {
    int nb = (dim + 31) / 32;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;

    const cuda_q8_0_block *blk = embd + (size_t)token * nb + b;
    float scale = __half2float(__ushort_as_half(blk->d));
    int base = b * 32;
    int n = min(32, dim - base);
    for (int j = 0; j < n; j++)
        out[base + j] = scale * (float)blk->qs[j];
}

__global__ void embd_lookup_q4_0_kernel(const cuda_q4_0_block *embd, float *out,
                                         int token, int dim) {
    int nb = (dim + 31) / 32;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;

    const cuda_q4_0_block *blk = embd + (size_t)token * nb + b;
    float scale = __half2float(__ushort_as_half(blk->d));
    int base = b * 32;
    int n = min(32, dim - base);
    for (int j = 0; j < n; j++) {
        int v;
        if (j < 16)
            v = (blk->qs[j] & 0xF) - 8;
        else
            v = (blk->qs[j - 16] >> 4) - 8;
        out[base + j] = scale * (float)v;
    }
}

extern "C" void cuda_embd_lookup_f32(const float *embd_dev, float *out_dev,
                                       int token, int dim, cudaStream_t stream) {
    int block = 256;
    int grid = (dim + block - 1) / block;
    embd_lookup_f32_kernel<<<grid, block, 0, stream>>>(embd_dev, out_dev, token, dim);
}

extern "C" void cuda_embd_lookup_f16(const uint16_t *embd_dev, float *out_dev,
                                       int token, int dim, cudaStream_t stream) {
    int block = 256;
    int grid = (dim + block - 1) / block;
    embd_lookup_f16_kernel<<<grid, block, 0, stream>>>(embd_dev, out_dev, token, dim);
}

extern "C" void cuda_embd_lookup_bf16(const uint16_t *embd_dev, float *out_dev,
                                        int token, int dim, cudaStream_t stream) {
    int block = 256;
    int grid = (dim + block - 1) / block;
    embd_lookup_bf16_kernel<<<grid, block, 0, stream>>>(embd_dev, out_dev, token, dim);
}

extern "C" void cuda_embd_lookup_q8_0(const cuda_q8_0_block *embd_dev, float *out_dev,
                                        int token, int dim, cudaStream_t stream) {
    int nb = (dim + 31) / 32;
    int block = 256;
    int grid = (nb + block - 1) / block;
    embd_lookup_q8_0_kernel<<<grid, block, 0, stream>>>(embd_dev, out_dev, token, dim);
}

extern "C" void cuda_embd_lookup_q4_0(const cuda_q4_0_block *embd_dev, float *out_dev,
                                       int token, int dim, cudaStream_t stream) {
    int nb = (dim + 31) / 32;
    int block = 256;
    int grid = (nb + block - 1) / block;
    embd_lookup_q4_0_kernel<<<grid, block, 0, stream>>>(embd_dev, out_dev, token, dim);
}

__global__ void embd_lookup_q4_k_kernel(const cuda_q4_k_block *embd, float *out,
                                        int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    int bi = i / 256;
    int o = i % 256;
    const cuda_q4_k_block *b = embd + (size_t)token * ((dim + 255) / 256) + bi;
    int g = o / 64;
    int k = o % 64;
    int is = g * 2 + (k >= 32 ? 1 : 0);
    int l = k >= 32 ? k - 32 : k;
    uint8_t sc, m;
    get_scale_min_k4(is, b->scales, &sc, &m);
    float d = __half2float(__ushort_as_half(b->d)) * (float)sc;
    float m1 = __half2float(__ushort_as_half(b->dmin)) * (float)m;
    uint8_t qb = b->qs[(size_t)g * 32 + l];
    int qv = (k >= 32) ? (qb >> 4) : (qb & 0xF);
    out[i] = d * (float)qv - m1;
}

__global__ void embd_lookup_q4_k_g_kernel(const cuda_q4_k_block *embd, float *out,
                                          int dim, const cuda_decode_params *dp) {
    int token = dp->token;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    int bi = i / 256;
    int o = i % 256;
    const cuda_q4_k_block *b = embd + (size_t)token * ((dim + 255) / 256) + bi;
    int g = o / 64;
    int k = o % 64;
    int is = g * 2 + (k >= 32 ? 1 : 0);
    int l = k >= 32 ? k - 32 : k;
    uint8_t sc, m;
    get_scale_min_k4(is, b->scales, &sc, &m);
    float d = __half2float(__ushort_as_half(b->d)) * (float)sc;
    float m1 = __half2float(__ushort_as_half(b->dmin)) * (float)m;
    uint8_t qb = b->qs[(size_t)g * 32 + l];
    int qv = (k >= 32) ? (qb >> 4) : (qb & 0xF);
    out[i] = d * (float)qv - m1;
}

extern "C" void cuda_embd_lookup_q4_k_g(const cuda_q4_k_block *embd_dev, float *out_dev,
                                        int dim, const int *params_dev, cudaStream_t stream) {

    int block = 256;
    int grid = (dim + block - 1) / block;
    embd_lookup_q4_k_g_kernel<<<grid, block, 0, stream>>>(
        embd_dev, out_dev, dim, (const cuda_decode_params *)params_dev);
}

extern "C" void cuda_embd_lookup_q4_k(const cuda_q4_k_block *embd_dev, float *out_dev,
                                      int token, int dim, cudaStream_t stream) {
    int block = 256;
    int grid = (dim + block - 1) / block;
    embd_lookup_q4_k_kernel<<<grid, block, 0, stream>>>(embd_dev, out_dev, token, dim);
}

__global__ void argmax_kernel(const float *__restrict__ logits, int n, int32_t *out_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_max = -INFINITY;
    int32_t local_idx = 0;

    for (int i = tid; i < n; i += stride) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, off);
        int32_t other_idx = __shfl_down_sync(0xffffffff, local_idx, off);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    __shared__ float smax[32];
    __shared__ int32_t sidx[32];
    int warp_id = tid >> 5;
    int lane = tid & 31;

    if (lane == 0) {
        smax[warp_id] = local_max;
        sidx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        float val = (lane < nw) ? smax[lane] : -INFINITY;
        int32_t idx = (lane < nw) ? sidx[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            float other_max = __shfl_down_sync(0xffffffff, val, off);
            int32_t other_idx = __shfl_down_sync(0xffffffff, idx, off);
            if (other_max > val) {
                val = other_max;
                idx = other_idx;
            }
        }
        if (lane == 0)
            *out_idx = idx;
    }
}

extern "C" void cuda_argmax(const float *logits_dev, int n, int32_t *out_idx_dev,
                             cudaStream_t stream) {

    int block = 1024;
    if (n < block) block = n;
    argmax_kernel<<<1, block, 0, stream>>>(logits_dev, n, out_idx_dev);
}

__global__ void q8_1_quant_rows_kernel(const float *__restrict__ x,
                                       cuda_q8_1_block *__restrict__ xq,
                                       int k, int m) {
    const int nb = k / 32;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= m * nb) return;
    const int tok = b / nb;
    const int blk = b - tok * nb;

    const float *xb = x + (size_t)tok * k + (size_t)blk * 32;
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++)
        amax = fmaxf(amax, fabsf(xb[j]));

    cuda_q8_1_block *out = &xq[(size_t)tok * nb + blk];
    if (amax < 1e-30f) {
        out->d = 0;
        out->s = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++) out->qs[j] = 0;
        return;
    }
    const float d = amax / 127.0f;
    store_half(&out->d, d);
    int32_t sum = 0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        const int q = cuda_xq8_val(xb[j], d);
        out->qs[j] = (int8_t)q;
        sum += q;
    }
    store_half(&out->s, d * (float)sum);
}

__global__ void q8_0_quant_batch_kernel(const float *__restrict__ x,
                                        cuda_q8_0_block *__restrict__ xq,
                                        int k, int m) {
    const int nb = k / 32;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * nb;
    if (b >= total) return;
    int tok = b / nb;
    int blk = b % nb;

    const float *xb = x + (size_t)tok * k + (size_t)blk * 32;
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float a = fabsf(xb[j]);
        if (a > amax)
            amax = a;
    }

    cuda_q8_0_block *out = &xq[(size_t)tok * nb + blk];
    if (amax < 1e-30f) {
        out->d = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            out->qs[j] = 0;
        return;
    }

    float d = amax / 127.0f;
    store_half(&out->d, d);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out->qs[j] = (int8_t)cuda_xq8_val(xb[j], d);
    }
}

__global__ void matmul_q8_0_dp4a_batch_kernel(const cuda_q8_0_block *__restrict__ w,
                                              const cuda_q8_0_block *__restrict__ xq,
                                              float *__restrict__ y, int n, int k, int m) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q8_0_block *wrow = w + (size_t)row * nb;
    const uint8_t *wrow8 = (const uint8_t *)wrow;

    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            float dw;
            int wqs[32];
            {
                const cuda_q8_0_block *wb = wrow + b;
                dw = __half2float(__ushort_as_half(wb->d));
                #pragma unroll
                for (int j = 0; j < 32; j++)
                    wqs[j] = (int)wb->qs[j];
            }
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                const cuda_q8_0_block *xb = xq + ((size_t)(t0 + t) * nb + b);
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 32; j++)
                    sumi += wqs[j] * (int)xb->qs[j];
                float dx = __half2float(__ushort_as_half(xb->d));
                acc[t] = fmaf(dw * dx, (float)sumi, acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

__global__ void matmul_q4_0_dp4a_batch_kernel(const cuda_q4_0_block *__restrict__ w,
                                              const cuda_q8_0_block *__restrict__ xq,
                                              float *__restrict__ y, int n, int k, int m) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q4_0_block *wrow = w + (size_t)row * nb;
    const uint8_t *wrow8 = (const uint8_t *)wrow;

    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            float dw;
            uint8_t wqb[16];
            {
                const cuda_q4_0_block *wb = wrow + b;
                dw = __half2float(__ushort_as_half(wb->d));
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    wqb[j] = wb->qs[j];
            }
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                const cuda_q8_0_block *xb = xq + ((size_t)(t0 + t) * nb + b);
                int sumi0 = 0, sumi1 = 0;
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    sumi0 += ((int)(wqb[j] & 0xF) - 8) * (int)xb->qs[j];
                    sumi1 += ((int)(wqb[j] >> 4) - 8) * (int)xb->qs[j + 16];
                }
                float dx = __half2float(__ushort_as_half(xb->d));
                acc[t] = fmaf(dw * dx, (float)(sumi0 + sumi1), acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (threadIdx.x % MM_LANES == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

__global__ void matmul_q4_1_dp4a_batch_kernel(const cuda_q4_1_block *__restrict__ w,
                                              const cuda_q8_0_block *__restrict__ xq,
                                              float *__restrict__ y, int n, int k, int m) {
    const int nb = k >> 5;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q4_1_block *wrow = w + (size_t)row * nb;

    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            const cuda_q4_1_block *wb = wrow + b;
            float dw = __half2float(__ushort_as_half(wb->d));
            float mw = __half2float(__ushort_as_half(wb->m));
            uint8_t wqb[16];
            #pragma unroll
            for (int j = 0; j < 16; j++)
                wqb[j] = wb->qs[j];

            int8_t wdeq[32];
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                wdeq[j]      = (int8_t)(wqb[j] & 0xF);
                wdeq[j + 16] = (int8_t)(wqb[j] >> 4);
            }
            int32_t wa[8];
            #pragma unroll
            for (int j = 0; j < 32; j += 4) {
                uint8_t w0 = (uint8_t)wdeq[j];
                uint8_t w1 = (uint8_t)wdeq[j + 1];
                uint8_t w2 = (uint8_t)wdeq[j + 2];
                uint8_t w3 = (uint8_t)wdeq[j + 3];
                wa[j >> 2] = (int32_t)w0 | ((int32_t)w1 << 8) |
                             ((int32_t)w2 << 16) | ((int32_t)w3 << 24);
            }
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                const cuda_q8_0_block *xb = xq + ((size_t)(t0 + t) * nb + b);
                int32_t qx = 0, xs = 0;
                #pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    uint8_t x0 = __ldg((const uint8_t *)&xb->qs[j]);
                    uint8_t x1 = __ldg((const uint8_t *)&xb->qs[j + 1]);
                    uint8_t x2 = __ldg((const uint8_t *)&xb->qs[j + 2]);
                    uint8_t x3 = __ldg((const uint8_t *)&xb->qs[j + 3]);
                    int32_t bval = (int32_t)x0 | ((int32_t)x1 << 8) |
                                   ((int32_t)x2 << 16) | ((int32_t)x3 << 24);
                    qx = __dp4a(wa[j >> 2], bval, qx);
                    xs = __dp4a(bval, 0x01010101, xs);
                }
                float dx = __half2float(__ushort_as_half(xb->d));
                acc[t] = fmaf(dw * dx, (float)qx, mw * cuda_q81_s(dx, xs) + acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (threadIdx.x % MM_LANES == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

__global__ void matmul_f32_batch_kernel(const float *__restrict__ w, const float *__restrict__ x,
                                        float *__restrict__ y, int n, int k, int m) {
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const float *wrow = w + (size_t)row * k;
    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int j = lane; j < k; j += MM_LANES) {
            float wv = wrow[j];
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                acc[t] = fmaf(wv, x[((size_t)(t0 + t) * k) + j], acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

static cuda_q8_0_block *xq_scratch_for_batch(int k, int m) {
    size_t need = (size_t)m * (size_t)(k / 32) * sizeof(cuda_q8_0_block);
    if (need <= g_xq_cap)
        return g_xq_scratch;
    if (g_xq_scratch)
        cudaFree(g_xq_scratch);
    g_xq_scratch = nullptr;
    g_xq_cap     = 0;
    if (cudaMalloc((void **)&g_xq_scratch, need) != cudaSuccess)
        return nullptr;
    g_xq_cap = need;
    return g_xq_scratch;
}

static __half *g_xf_scratch = nullptr;
static size_t g_xf_cap = 0;
static __half *xf_scratch_for_batch(int k, int m) {
    size_t need = (size_t)m * (size_t)k * sizeof(__half);
    if (need <= g_xf_cap)
        return g_xf_scratch;
    if (g_xf_scratch)
        cudaFree(g_xf_scratch);
    g_xf_scratch = nullptr;
    g_xf_cap     = 0;
    if (cudaMalloc((void **)&g_xf_scratch, need) != cudaSuccess)
        return nullptr;
    g_xf_cap = need;
    return g_xf_scratch;
}

static void *g_wf_scratch = nullptr;
static size_t g_wf_cap = 0;
static void *wf_scratch_for(size_t need) {
    if (need <= g_wf_cap)
        return g_wf_scratch;
    if (g_wf_scratch)
        cudaFree(g_wf_scratch);
    g_wf_scratch = nullptr;
    g_wf_cap     = 0;
    if (cudaMalloc(&g_wf_scratch, need) != cudaSuccess)
        return nullptr;
    g_wf_cap = need;
    return g_wf_scratch;
}

static int batch_fp16_enabled(void) {
    return 1;
}

#ifdef HAVE_CUBLAS

static cublasHandle_t g_cublas_h = nullptr;
static int g_cublas_state = 0;
static void *g_cublas_ws = nullptr;
static size_t g_cublas_ws_cap = 0;

static cublasHandle_t cublas_for(cudaStream_t stream) {
    if (g_cublas_state < 0)
        return nullptr;
        if (!g_cublas_h) {
        if (cublasCreate(&g_cublas_h) != CUBLAS_STATUS_SUCCESS) {
            g_cublas_state = -1;
            return nullptr;
        }

        cublasSetMathMode(g_cublas_h, CUBLAS_DEFAULT_MATH);

        size_t need = (size_t)4 * 1024 * 1024;
        if (cudaMalloc(&g_cublas_ws, need) != cudaSuccess) {
            g_cublas_state = -1;
            return nullptr;
        }
        g_cublas_ws_cap = need;
        if (cublasSetWorkspace(g_cublas_h, g_cublas_ws, g_cublas_ws_cap) !=
            CUBLAS_STATUS_SUCCESS) {
            g_cublas_state = -1;
            return nullptr;
        }
        g_cublas_state = 1;
    }
    cublasSetStream(g_cublas_h, stream);
    return g_cublas_h;
}
#endif

static void matmul_batch_fp16_impl(const __half *w_fp16, const float *x_dev, float *y_dev,
                                   int n, int k, int m, cudaStream_t stream) {
    __half *xf = xf_scratch_for_batch(k, m);
    if (!xf) return;
    int total = m * k;
    fp16_copy_batch_kernel<<<(total + 255) / 256, 256, 0, stream>>>(x_dev, xf, k, m);

#ifdef HAVE_CUBLAS

    if (1) {
        cublasHandle_t ch = cublas_for(stream);
        if (ch) {
            float alpha = 1.0f, beta = 0.0f;
            cublasStatus_t cst = cublasGemmEx(
                ch, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w_fp16, CUDA_R_16F, k,
                xf, CUDA_R_16F, k, &beta, y_dev, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (cst == CUBLAS_STATUS_SUCCESS)
                return;
        }
    }
#endif

    dim3 grid((m + F16_TILE_N - 1) / F16_TILE_N,
              (n + F16_TILE_M - 1) / F16_TILE_M);
    size_t shmem = ((size_t)F16_TILE_M * WMMA_K_ELEMS +
                    (size_t)WMMA_K_ELEMS * F16_TILE_N) * sizeof(__half);
    matmul_fp16_wmma64_batch_kernel<<<grid, F16_TILE_THREADS, shmem, stream>>>(
        w_fp16, xf, y_dev, n, k, m);
}

extern "C" void cuda_matmul_batch_fp16(const void *w_in, const float *x_dev, float *y_dev,
                                       int n, int k, int m, cudaStream_t stream) {
    matmul_batch_fp16_impl((const __half *)w_in, x_dev, y_dev, n, k, m, stream);
}

extern "C" void cuda_matmul_batch_q8_0(const void *w_dev, const float *x_dev, float *y_dev,
                                        int n, int k, int m, cudaStream_t stream) {
    const int nb = k / 32;

    if (batch_fp16_enabled() && m >= 16 && n >= 32) {
        __half *wf = (__half *)wf_scratch_for((size_t)n * k * sizeof(__half));
        if (!wf) return;
        int total = n * k;
        dequant_q8_0_to_fp16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            w_dev, wf, n, k);
        matmul_batch_fp16_impl(wf, x_dev, y_dev, n, k, m, stream);
        return;
    }
    cuda_q8_0_block *xq = xq_scratch_for_batch(k, m);
    if (!xq) return;
    int qblock = 256;
    int qgrid = (m * nb + qblock - 1) / qblock;
    q8_0_quant_batch_kernel<<<qgrid, qblock, 0, stream>>>(x_dev, xq, k, m);

    if (m >= 16 && n >= 32) {
        dim3 grid((m + WMMA_COLS_PER_BLOCK - 1) / WMMA_COLS_PER_BLOCK,
                  (n + WMMA_ROWS_PER_BLOCK - 1) / WMMA_ROWS_PER_BLOCK);
        size_t shmem = ((size_t)WMMA_ROWS_PER_BLOCK * WMMA_K_ELEMS +
                        (size_t)WMMA_K_ELEMS * WMMA_COLS_PER_BLOCK) * sizeof(__half);
        matmul_q8_0_wmma_batch_kernel<<<grid, WMMA_THREADS_PER_BLOCK, shmem, stream>>>(
            (const cuda_q8_0_block *)w_dev, xq, y_dev, n, k, m);
    } else {
        dim3 grid(((size_t)n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        matmul_q8_0_dp4a_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
            (const cuda_q8_0_block *)w_dev, xq, y_dev, n, k, m);
    }
}

extern "C" void cuda_matmul_batch_q4_0(const void *w_dev, const float *x_dev, float *y_dev,
                                        int n, int k, int m, cudaStream_t stream) {
    const int nb = k / 32;

    if (batch_fp16_enabled() && m >= 16 && n >= 32) {
        __half *wf = (__half *)wf_scratch_for((size_t)n * k * sizeof(__half));
        if (!wf) return;
        int total = n * k;
        dequant_q4_0_to_fp16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            w_dev, wf, n, k);
        matmul_batch_fp16_impl(wf, x_dev, y_dev, n, k, m, stream);
        return;
    }
    cuda_q8_0_block *xq = xq_scratch_for_batch(k, m);
    if (!xq) return;
    int qblock = 256;
    int qgrid = (m * nb + qblock - 1) / qblock;
    q8_0_quant_batch_kernel<<<qgrid, qblock, 0, stream>>>(x_dev, xq, k, m);

    if (m >= 16 && n >= 32) {
        dim3 grid((m + WMMA_COLS_PER_BLOCK - 1) / WMMA_COLS_PER_BLOCK,
                  (n + WMMA_ROWS_PER_BLOCK - 1) / WMMA_ROWS_PER_BLOCK);
        size_t shmem = ((size_t)WMMA_ROWS_PER_BLOCK * WMMA_K_ELEMS +
                        (size_t)WMMA_K_ELEMS * WMMA_COLS_PER_BLOCK) * sizeof(__half);
        matmul_q4_0_wmma_batch_kernel<<<grid, WMMA_THREADS_PER_BLOCK, shmem, stream>>>(
            (const cuda_q4_0_block *)w_dev, xq, y_dev, n, k, m);
    } else {
        dim3 grid(((size_t)n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        matmul_q4_0_dp4a_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
            (const cuda_q4_0_block *)w_dev, xq, y_dev, n, k, m);
    }
}

extern "C" void cuda_matmul_batch_q4_1(const void *w_dev, const float *x_dev, float *y_dev,
                                       int n, int k, int m, cudaStream_t stream) {
    const int nb = k / 32;
    cuda_q8_0_block *xq = xq_scratch_for_batch(k, m);
    if (!xq) return;
    int qblock = 256;
    int qgrid = (m * nb + qblock - 1) / qblock;
    q8_0_quant_batch_kernel<<<qgrid, qblock, 0, stream>>>(x_dev, xq, k, m);
    dim3 grid(((size_t)n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q4_1_dp4a_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q4_1_block *)w_dev, xq, y_dev, n, k, m);
}

extern "C" void cuda_matmul_batch_f32(const float *w_dev, const float *x_dev, float *y_dev,
                                      int n, int k, int m, cudaStream_t stream) {
    dim3 grid(((size_t)n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_f32_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        w_dev, x_dev, y_dev, n, k, m);
}

__global__ void q8_k_quant_batch_kernel(const float *__restrict__ x,
                                        cuda_q8_k_block *__restrict__ xq,
                                        int k, int m) {
    const int nb = k / 256;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * nb;
    if (b >= total) return;
    int tok = b / nb;
    int blk = b % nb;

    const float *xb = x + (size_t)tok * k + (size_t)blk * 256;
    cuda_q8_k_block *out = &xq[(size_t)tok * nb + blk];

    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < 256; j++) {
        float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax;
            max = xb[j];
        }
    }

    if (amax < 1e-30f) {
        out->d = 0.0f;
        for (int j = 0; j < 256; j++)
            out->qs[j] = 0;
        for (int j = 0; j < 16; j++)
            out->bsums[j] = 0;
        return;
    }

    float iscale = -127.0f / max;
    out->d = 1.0f / iscale;

    int32_t sum = 0;
    int16_t bsums[16] = {0};
    for (int j = 0; j < 256; j++) {
        int q = (int)roundf(iscale * xb[j]);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out->qs[j] = (int8_t)q;
        sum += q;
        bsums[j / 16] += q;
    }

    for (int j = 0; j < 16; j++)
        out->bsums[j] = bsums[j];
}

__global__ void matmul_q4_k_batch_kernel(const cuda_q4_k_block *__restrict__ w,
                                         const cuda_q8_k_block *__restrict__ xq,
                                         float *__restrict__ y, int n, int k, int m) {
    const int nb = k / 256;
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const cuda_q4_k_block *wrow = w + (size_t)row * nb;

    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            const cuda_q4_k_block *wb = wrow + b;
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                const cuda_q8_k_block *xb = xq + ((size_t)(t0 + t) * nb + b);
                float xd = xb->d;
                acc[t] = q4_k_dot(wb, xb->qs, xb->bsums, xd, acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

__global__ void matmul_f16_batch_kernel(const uint16_t *__restrict__ w, const float *__restrict__ x,
                                        float *__restrict__ y, int n, int k, int m) {
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const uint16_t *wrow = w + (size_t)row * k;
    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int j = lane; j < k; j += MM_LANES) {
            __half hf = __ushort_as_half(wrow[j]);
            float wv = __half2float(hf);
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                acc[t] = fmaf(wv, x[((size_t)(t0 + t) * k) + j], acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

__global__ void matmul_bf16_batch_kernel(const uint16_t *__restrict__ w, const float *__restrict__ x,
                                         float *__restrict__ y, int n, int k, int m) {
    const int row = blockIdx.x * MM_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;

    const int lane = threadIdx.x & (MM_LANES - 1);
    const uint16_t *wrow = w + (size_t)row * k;
    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int j = lane; j < k; j += MM_LANES) {
            __nv_bfloat16 bf = __ushort_as_bfloat16(wrow[j]);
            float wv = (float)bf;
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                acc[t] = fmaf(wv, x[((size_t)(t0 + t) * k) + j], acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0)
                y[(size_t)(t0 + t) * n + row] = v;
        }
    }
}

static cuda_q8_k_block *q8k_scratch_for_batch(int k, int m) {
    size_t need = (size_t)m * (size_t)(k / 256) * sizeof(cuda_q8_k_block);
    if (need <= g_q8k_cap)
        return g_q8k_scratch;
    if (g_q8k_scratch)
        cudaFree(g_q8k_scratch);
    g_q8k_scratch = nullptr;
    g_q8k_cap     = 0;
    if (cudaMalloc((void **)&g_q8k_scratch, need) != cudaSuccess)
        return nullptr;
    g_q8k_cap = need;
    return g_q8k_scratch;
}

extern "C" void cuda_matmul_batch_q4_k(const void *w_dev, const float *x_dev, float *y_dev,
                                       int n, int k, int m, cudaStream_t stream) {
    const int nb = k / 256;
    cuda_q8_k_block *xq = q8k_scratch_for_batch(k, m);
    if (!xq) return;
    int qblock = 256;
    int qgrid = (m * nb + qblock - 1) / qblock;
    q8_k_quant_batch_kernel<<<qgrid, qblock, 0, stream>>>(x_dev, xq, k, m);
    dim3 grid(((size_t)n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q4_k_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q4_k_block *)w_dev, xq, y_dev, n, k, m);
}

__global__ void matmul_multi_q4_0_batch_kernel(const mm_multi_desc d,
                                               const cuda_q8_0_block *__restrict__ xq,
                                               int m) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const int g    = blockIdx.x * MM_ROWS_PER_BLOCK + warp;

    int col = d.nm, row = 0, cum = 0;
    for (int c = 0; c < d.nm; c++) {
        const int nn = d.n[c];
        if (g < cum + nn) { col = c; row = g - cum; break; }
        cum += nn;
    }
    if (col >= d.nm)
        return;

    const int nb = d.k / 32;
    const int n_col = d.n[col];
    const cuda_q4_0_block *wrow = (const cuda_q4_0_block *)d.w[col] + (size_t)row * nb;
    const uint8_t *wrow8 = (const uint8_t *)wrow;
    float *ycol = d.y[col];

    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            float dw;
            uint8_t wqb[16];
            {
                const cuda_q4_0_block *wb = wrow + b;
                dw = __half2float(__ushort_as_half(wb->d));
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    wqb[j] = wb->qs[j];
            }
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                const cuda_q8_0_block *xb = xq + ((size_t)(t0 + t) * nb + b);
                int sumi0 = 0, sumi1 = 0;
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    sumi0 += ((int)(wqb[j] & 0xF) - 8) * (int)xb->qs[j];
                    sumi1 += ((int)(wqb[j] >> 4) - 8) * (int)xb->qs[j + 16];
                }
                float dx = __half2float(__ushort_as_half(xb->d));
                acc[t] = fmaf(dw * dx, (float)(sumi0 + sumi1), acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (threadIdx.x % MM_LANES == 0) {
                float rv = d.resid ? d.resid[(size_t)(t0 + t) * n_col + row] : 0.0f;
                ycol[(size_t)(t0 + t) * n_col + row] = v + rv;
            }
        }
    }
}

__global__ void matmul_multi_q8_0_batch_kernel(const mm_multi_desc d,
                                               const cuda_q8_0_block *__restrict__ xq,
                                               int m) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (MM_LANES - 1);
    const int g    = blockIdx.x * MM_ROWS_PER_BLOCK + warp;

    int col = d.nm, row = 0, cum = 0;
    for (int c = 0; c < d.nm; c++) {
        const int nn = d.n[c];
        if (g < cum + nn) { col = c; row = g - cum; break; }
        cum += nn;
    }
    if (col >= d.nm)
        return;

    const int nb = d.k / 32;
    const int n_col = d.n[col];
    const cuda_q8_0_block *wrow = (const cuda_q8_0_block *)d.w[col] + (size_t)row * nb;
    const uint8_t *wrow8 = (const uint8_t *)wrow;
    float *ycol = d.y[col];

    for (int t0 = 0; t0 < m; t0 += MM_TOK_TILE) {
        const int nt = min(MM_TOK_TILE, m - t0);
        float acc[MM_TOK_TILE];
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++)
            acc[t] = 0.0f;
        for (int b = lane; b < nb; b += MM_LANES) {
            float dw;
            int8_t wqs[32];
            {
                const cuda_q8_0_block *wb = wrow + b;
                dw = d_from_half(&wb->d);
                #pragma unroll
                for (int j = 0; j < 32; j++)
                    wqs[j] = wb->qs[j];
            }
            #pragma unroll
            for (int t = 0; t < MM_TOK_TILE; t++) {
                if (t >= nt) break;
                const cuda_q8_0_block *xb = xq + ((size_t)(t0 + t) * nb + b);
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 32; j++)
                    sumi += (int)wqs[j] * (int)xb->qs[j];
                float dx = d_from_half(&xb->d);
                acc[t] = fmaf(dw * dx, (float)sumi, acc[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < MM_TOK_TILE; t++) {
            if (t >= nt) break;
            float v = acc[t];
            #pragma unroll
            for (int off = MM_LANES >> 1; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (threadIdx.x % MM_LANES == 0) {
                float rv = d.resid ? d.resid[(size_t)(t0 + t) * n_col + row] : 0.0f;
                ycol[(size_t)(t0 + t) * n_col + row] = v + rv;
            }
        }
    }
}

extern "C" void cuda_matmul_multi_batch_q4_0(const void *const *w_dev, float *const *y_dev,
                                             const int *n_host, const float *x_dev, int nm, int k,
                                             int m, const float *resid_dev, cudaStream_t stream) {
    const int nb = k / 32;

    if (batch_fp16_enabled() && !resid_dev && m >= 16) {
        size_t total_need = 0;
        for (int c = 0; c < nm; c++)
            total_need += (size_t)n_host[c] * k * sizeof(__half);
        char *base = (char *)wf_scratch_for(total_need);
        if (base) {
            size_t off = 0;
            for (int c = 0; c < nm; c++) {
                __half *wf = (__half *)(base + off);
                int total = n_host[c] * k;
                dequant_q4_0_to_fp16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                    w_dev[c], wf, n_host[c], k);
                matmul_batch_fp16_impl(wf, x_dev, y_dev[c], n_host[c], k, m, stream);
                off += (size_t)n_host[c] * k * sizeof(__half);
            }
            return;
        }
    }
    cuda_q8_0_block *xq = xq_scratch_for_batch(k, m);
    if (!xq)
        return;
    int qblock = 256;
    int qgrid = (m * nb + qblock - 1) / qblock;
    q8_0_quant_batch_kernel<<<qgrid, qblock, 0, stream>>>(x_dev, xq, k, m);

    mm_multi_desc desc;
    if (nm < 1 || nm > 8)
        return;
    int total = 0;
    for (int c = 0; c < nm; c++) {
        desc.w[c] = (const char *)w_dev[c];
        desc.y[c] = y_dev[c];
        desc.n[c] = n_host[c];
        total += n_host[c];
    }
    desc.nm = nm;
    desc.k  = k;
    desc.resid = resid_dev;

    dim3 grid(((size_t)total + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_multi_q4_0_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(desc, xq, m);
}

extern "C" void cuda_matmul_multi_batch_q8_0(const void *const *w_dev, float *const *y_dev,
                                             const int *n_host, const float *x_dev, int nm, int k,
                                             int m, const float *resid_dev, cudaStream_t stream) {
    const int nb = k / 32;

    if (batch_fp16_enabled() && !resid_dev && m >= 16) {

        size_t total_need = 0;
        for (int c = 0; c < nm; c++)
            total_need += (size_t)n_host[c] * k * sizeof(__half);
        char *base = (char *)wf_scratch_for(total_need);
        if (base) {
            size_t off = 0;
            for (int c = 0; c < nm; c++) {
                __half *wf = (__half *)(base + off);
                int total = n_host[c] * k;
                dequant_q8_0_to_fp16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                    w_dev[c], wf, n_host[c], k);
                matmul_batch_fp16_impl(wf, x_dev, y_dev[c], n_host[c], k, m, stream);
                off += (size_t)n_host[c] * k * sizeof(__half);
            }
            return;
        }
    }
    cuda_q8_0_block *xq = xq_scratch_for_batch(k, m);
    if (!xq)
        return;
    int qblock = 256;
    int qgrid = (m * nb + qblock - 1) / qblock;
    q8_0_quant_batch_kernel<<<qgrid, qblock, 0, stream>>>(x_dev, xq, k, m);

    mm_multi_desc desc;
    if (nm < 1 || nm > 8)
        return;
    int total = 0;
    for (int c = 0; c < nm; c++) {
        desc.w[c] = (const char *)w_dev[c];
        desc.y[c] = y_dev[c];
        desc.n[c] = n_host[c];
        total += n_host[c];
    }
    desc.nm = nm;
    desc.k  = k;
    desc.resid = resid_dev;

    dim3 grid(((size_t)total + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_multi_q8_0_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(desc, xq, m);
}

extern "C" void cuda_matmul_batch_f16(const uint16_t *w_dev, const float *x_dev, float *y_dev,
                                      int n, int k, int m, cudaStream_t stream) {
    dim3 grid(((size_t)n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_f16_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        w_dev, x_dev, y_dev, n, k, m);
}

extern "C" void cuda_matmul_batch_bf16(const uint16_t *w_dev, const float *x_dev, float *y_dev,
                                       int n, int k, int m, cudaStream_t stream) {
    dim3 grid(((size_t)n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_bf16_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        w_dev, x_dev, y_dev, n, k, m);
}

__global__ void ple_combine_kernel(float *__restrict__ ple, const float *__restrict__ proj,
                                   int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        ple[i] = (ple[i] + proj[i]) * scale;
}

extern "C" void cuda_ple_combine(float *ple_dev, const float *proj_dev, int n, float scale,
                                 cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    ple_combine_kernel<<<grid, block, 0, stream>>>(ple_dev, proj_dev, n, scale);
}

__global__ void ple_norm_batch_kernel(float *__restrict__ proj, const float *__restrict__ norm_w,
                                      int n_rows, int total_ple, int n_embd, int n_layers,
                                      float eps) {
    int idx = blockIdx.x;
    int total = n_rows * n_layers;
    if (idx >= total) return;
    int row = idx / n_layers;
    int l = idx % n_layers;
    float *slice = proj + (size_t)row * total_ple + (size_t)l * n_embd;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    extern __shared__ float sm[];
    float s = 0.0f;
    for (int i = t; i < n_embd; i += nt)
        s += slice[i] * slice[i];
    sm[t] = s;
    __syncthreads();
    for (int sz = nt >> 1; sz > 0; sz >>= 1) {
        if (t < sz) sm[t] += sm[t + sz];
        __syncthreads();
    }
    if (t == 0)
        sm[0] = 1.0f / sqrtf((sm[0] / (float)n_embd) + eps);
    __syncthreads();
    const float scale = sm[0];
    for (int i = t; i < n_embd; i += nt)
        slice[i] = norm_w ? slice[i] * scale * norm_w[i] : slice[i] * scale;
}

extern "C" void cuda_ple_norm_batch(float *proj_dev, const float *norm_w_dev, int n_rows,
                                    int total_ple, int n_embd, int n_layers, float eps,
                                    cudaStream_t stream) {
    const int nt = 256;
    dim3 grid((unsigned)((size_t)n_rows * (size_t)n_layers));
    ple_norm_batch_kernel<<<grid, nt, (size_t)nt * sizeof(float), stream>>>(
        proj_dev, norm_w_dev, n_rows, total_ple, n_embd, n_layers, eps);
}

__global__ void rope_ext_g_kernel(float *vec, int n_heads, int head_dim,
                                  const float *cos_base, const float *sin_base,
                                  const cuda_decode_params *dp, int neox) {
    int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * half;
    if (idx >= total) return;

    const float *cos_tbl = cos_base + (size_t)dp->pos * half;
    const float *sin_tbl = sin_base + (size_t)dp->pos * half;
    int h = idx / half;
    int j = idx % half;
    float *vh = vec + (size_t)h * head_dim;
    float c = cos_tbl[j];
    float s = sin_tbl[j];

    if (neox) {
        float v0 = vh[j];
        float v1 = vh[j + half];
        vh[j]        = v0 * c - v1 * s;
        vh[j + half] = v0 * s + v1 * c;
    } else {
        float v0 = vh[2 * j];
        float v1 = vh[2 * j + 1];
        vh[2 * j]     = v0 * c - v1 * s;
        vh[2 * j + 1] = v0 * s + v1 * c;
    }
}

extern "C" void cuda_rope_ext_g(float *vec_dev, int n_heads, int head_dim,
                                const float *cos_base, const float *sin_base,
                                const int *params_dev, int neox, cudaStream_t stream) {

    int half = head_dim / 2;
    int total = n_heads * half;
    int block = 256;
    int grid = (total + block - 1) / block;
    rope_ext_g_kernel<<<grid, block, 0, stream>>>(
        vec_dev, n_heads, head_dim, cos_base, sin_base,
        (const cuda_decode_params *)params_dev, neox);
}

__global__ void attn_f16_g_kernel(const float *__restrict__ q,
                                  const uint16_t *__restrict__ kc,
                                  const uint16_t *__restrict__ vc,
                                  float *__restrict__ out,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  size_t layer_base_bytes, size_t kvh_stride,
                                  int sliding_window, float scale,
                                  const cuda_decode_params *dp) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if (h >= n_heads) return;

    int n_pos_total = dp->n_pos;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_total > sliding_window) {
        attn_start = n_pos_total - sliding_window;
        n_pos_total = sliding_window;
    }
    size_t layer_base = (layer_base_bytes + (size_t)attn_start * head_dim * sizeof(uint16_t)) /
                        sizeof(uint16_t);
    int n_pos = n_pos_total;

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_k = &smem[head_dim];
    float *s_v = &smem[2 * head_dim];

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q_h[i];
    }
    __syncthreads();

    float M = -INFINITY;
    float S = 0.0f;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0};

    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    for (int t = 0; t < n_pos; t++) {
        const uint16_t *k_t = kc_head + (size_t)t * head_dim;
        const uint16_t *v_t = vc_head + (size_t)t * head_dim;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_k[i] = __half2float(__ushort_as_half(k_t[i]));
        }
        __syncthreads();

        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += s_q[i] * s_k[i];
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            score += __shfl_down_sync(0xffffffff, score, offset);

        float raw_score = score;
        float ms = 1.0f;
        float vs = 1.0f;
        if (lane == 0) {
            raw_score *= scale;
            if (raw_score > M) {
                float old_M = M;
                M = raw_score;
                ms = expf(old_M - M);
                S *= ms;
                vs = 1.0f;
            } else {
                ms = 1.0f;
                vs = expf(raw_score - M);
            }
            S += vs;
        }

        M = __shfl_sync(0xffffffff, M, 0);
        S = __shfl_sync(0xffffffff, S, 0);
        ms = __shfl_sync(0xffffffff, ms, 0);
        vs = __shfl_sync(0xffffffff, vs, 0);

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_v[i] = __half2float(__ushort_as_half(v_t[i]));
        }
        __syncthreads();

        for (int idx = start_idx; idx < end_idx; idx++) {
            int local_idx = idx - start_idx;
            O_local[local_idx] = O_local[local_idx] * ms + vs * s_v[idx];
        }
    }

    float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float *out_h = out + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

__global__ void attn_f16_tiled_g_kernel(const float *__restrict__ q,
                                        const uint16_t *__restrict__ kc,
                                        const uint16_t *__restrict__ vc,
                                        float *__restrict__ out,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        size_t layer_base_bytes, size_t kvh_stride,
                                        int sliding_window, float scale,
                                        const cuda_decode_params *dp, int tile_size) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if (h >= n_heads) return;

    int n_pos_total = dp->n_pos;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_total > sliding_window) {
        attn_start = n_pos_total - sliding_window;
        n_pos_total = sliding_window;
    }
    size_t layer_base = (layer_base_bytes + (size_t)attn_start * head_dim * sizeof(uint16_t)) /
                        sizeof(uint16_t);
    int n_pos = n_pos_total;

    extern __shared__ float smem[];
    float *s_q = smem;

    float *s_k_tile = &smem[1 * head_dim];
    float *s_v_tile = &smem[1 * head_dim + tile_size * head_dim];

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;
    {
        int chunk = (head_dim + blockDim.x - 1) / blockDim.x;
        int base = tid * chunk;
        int end = min(base + chunk, head_dim);
        for (int i = base; i < end; i++) {
            s_q[i] = q_h[i];
        }
    }
    __syncthreads();

    double M = -INFINITY;
    double S = 0.0;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0.0f};

    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    for (int tile_start = 0; tile_start < n_pos; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, n_pos);
        int tile_len = tile_end - tile_start;

        {
            int tile_pairs = (tile_len * head_dim) >> 1;
            const __half2 *k2 = (const __half2 *)(kc_head + (size_t)tile_start * head_dim);
            const __half2 *v2 = (const __half2 *)(vc_head + (size_t)tile_start * head_dim);
            float2 *s_k2 = (float2 *)s_k_tile;
            float2 *s_v2 = (float2 *)s_v_tile;
            for (int idx = tid; idx < tile_pairs; idx += blockDim.x) {
                s_k2[idx] = __half22float2(k2[idx]);
                s_v2[idx] = __half22float2(v2[idx]);
            }
            if ((tile_len * head_dim) & 1) {
                int tail = tile_len * head_dim - 1;
                int b = tail / head_dim;
                int i = tail % head_dim;
                const uint16_t *k_t = kc_head + (size_t)(tile_start + b) * head_dim;
                const uint16_t *v_t = vc_head + (size_t)(tile_start + b) * head_dim;
                if (tid == 0) {
                    s_k_tile[tail] = __half2float(__ushort_as_half(k_t[i]));
                    s_v_tile[tail] = __half2float(__ushort_as_half(v_t[i]));
                }
            }
        }
        __syncthreads();

        for (int t_local = 0; t_local < tile_len; t_local++) {
            const float *k_tile_row = &s_k_tile[t_local * head_dim];
            const float *v_tile_row = &s_v_tile[t_local * head_dim];

            float score = 0.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                score += s_q[i] * k_tile_row[i];
            }

            for (int offset = 16; offset > 0; offset >>= 1)
                score += __shfl_down_sync(0xffffffff, score, offset);

            double raw_score = (double)score * (double)scale;
            double ms = 1.0;
            double vs = 1.0;
            if (lane == 0) {
                if (raw_score > M) {
                    double old_M = M;
                    M = raw_score;
                    ms = exp(old_M - M);
                    S *= ms;
                    vs = 1.0;
                } else {
                    ms = 1.0;
                    vs = exp(raw_score - M);
                }
                S += vs;
            }

            {
                double tmp_M = M, tmp_S = S, tmp_ms = ms, tmp_vs = vs;
                tmp_M = __shfl_sync(0xffffffff, tmp_M, 0);
                tmp_S = __shfl_sync(0xffffffff, tmp_S, 0);
                tmp_ms = __shfl_sync(0xffffffff, tmp_ms, 0);
                tmp_vs = __shfl_sync(0xffffffff, tmp_vs, 0);
                M = tmp_M; S = tmp_S; ms = tmp_ms; vs = tmp_vs;
            }

            float f_ms = (float)ms;
            float f_vs = (float)vs;
            for (int idx = start_idx; idx < end_idx; idx++) {
                int local_idx = idx - start_idx;
                O_local[local_idx] = O_local[local_idx] * f_ms + f_vs * v_tile_row[idx];
            }
        }
        __syncthreads();
    }

    float inv_S = (S > 0.0) ? (1.0f / (float)S) : 0.0f;
    float *out_h = out + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

__global__ void attn_f16_tiled_gqa_g_kernel(const float *__restrict__ q,
                                            const uint16_t *__restrict__ kc,
                                            const uint16_t *__restrict__ vc,
                                            float *__restrict__ out,
                                            int n_heads, int n_kv_heads, int head_dim,
                                            size_t layer_base_bytes, size_t kvh_stride,
                                            int sliding_window, float scale, int n_queries,
                                            const cuda_decode_params *dp, int tile_size) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int w = tid >> 5;

    int n_pos_total = dp->n_pos;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_total > sliding_window) {
        attn_start = n_pos_total - sliding_window;
        n_pos_total = sliding_window;
    }
    size_t layer_base = (layer_base_bytes + (size_t)attn_start * head_dim * sizeof(uint16_t)) /
                        sizeof(uint16_t);
    int n_pos = n_pos_total;

    int n_kv_groups = n_heads / n_kv_heads;
    int qb = (n_kv_groups + n_queries - 1) / n_queries;
    int kv_head = b / qb;
    int qslot0 = (b % qb) * n_queries;
    int nq = min(n_queries, n_kv_groups - qslot0);
    int active = (w < nq) ? 1 : 0;
    int h = kv_head * n_kv_groups + qslot0 + w;

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_k_tile = &smem[(size_t)n_queries * head_dim];
    float *s_v_tile = &smem[(size_t)n_queries * head_dim + tile_size * head_dim];

    float *s_qw = s_q + (size_t)w * head_dim;
    if (active) {
        const float *q_h = q + (size_t)h * head_dim;
        int chunk = (head_dim + 31) / 32;
        int base = lane * chunk;
        int end = min(base + chunk, head_dim);
        for (int i = base; i < end; i++) {
            s_qw[i] = q_h[i];
        }
    }
    __syncthreads();

    float M = -INFINITY;
    float S = 0.0f;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = lane * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0.0f};

    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    for (int tile_start = 0; tile_start < n_pos; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, n_pos);
        int tile_len = tile_end - tile_start;

        {
            int tile_pairs = (tile_len * head_dim) >> 1;
            const __half2 *k2 = (const __half2 *)(kc_head + (size_t)tile_start * head_dim);
            const __half2 *v2 = (const __half2 *)(vc_head + (size_t)tile_start * head_dim);
            float2 *s_k2 = (float2 *)s_k_tile;
            float2 *s_v2 = (float2 *)s_v_tile;
            for (int idx = tid; idx < tile_pairs; idx += blockDim.x) {
                s_k2[idx] = __half22float2(k2[idx]);
                s_v2[idx] = __half22float2(v2[idx]);
            }
            if (((tile_len * head_dim) & 1) && tid == 0) {
                int tail = tile_len * head_dim - 1;
                int bb = tail / head_dim;
                int ii = tail % head_dim;
                const uint16_t *k_t = kc_head + (size_t)(tile_start + bb) * head_dim;
                const uint16_t *v_t = vc_head + (size_t)(tile_start + bb) * head_dim;
                s_k_tile[tail] = __half2float(__ushort_as_half(k_t[ii]));
                s_v_tile[tail] = __half2float(__ushort_as_half(v_t[ii]));
            }
        }
        __syncthreads();

        if (active) {

            for (int t_local = 0; t_local < tile_len; t_local++) {
                const float *k_tile_row = &s_k_tile[t_local * head_dim];
                const float *v_tile_row = &s_v_tile[t_local * head_dim];

                float score = 0.0f;
                for (int i = lane; i < head_dim; i += 32) {
                    score += s_qw[i] * k_tile_row[i];
                }

                for (int offset = 16; offset > 0; offset >>= 1)
                    score += __shfl_down_sync(0xffffffff, score, offset);

                float raw_score = score * scale;
                float ms = 1.0f;
                float vs = 1.0f;
                if (lane == 0) {
                    if (raw_score > M) {
                        float old_M = M;
                        M = raw_score;
                        ms = expf(old_M - M);
                        S *= ms;
                        vs = 1.0f;
                    } else {
                        ms = 1.0f;
                        vs = expf(raw_score - M);
                    }
                    S += vs;
                }

                M = __shfl_sync(0xffffffff, M, 0);
                S = __shfl_sync(0xffffffff, S, 0);
                ms = __shfl_sync(0xffffffff, ms, 0);
                vs = __shfl_sync(0xffffffff, vs, 0);

                for (int idx = start_idx; idx < end_idx; idx++) {
                    int local_idx = idx - start_idx;
                    O_local[local_idx] = O_local[local_idx] * ms + vs * v_tile_row[idx];
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
        float *out_h = out + (size_t)h * head_dim;
        for (int idx = start_idx; idx < end_idx; idx++) {
            int local_idx = idx - start_idx;
            out_h[idx] = O_local[local_idx] * inv_S;
        }
    }
}

#define COOP_QTILE 16

__global__ void attn_f16_decode_coop_kernel(const float *__restrict__ q,
                                            const uint16_t *__restrict__ kc,
                                            const uint16_t *__restrict__ vc,
                                            float *__restrict__ out,
                                            int n_heads, int n_kv_heads, int head_dim,
                                            size_t layer_base_bytes, size_t kvh_stride,
                                            float scale, const int *params_dev,
                                            int sliding_window) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    int n_pos = params_dev ? params_dev[1] : 0;

    size_t layer_base = layer_base_bytes / sizeof(uint16_t);
    if (sliding_window > 0 && n_pos > sliding_window) {
        layer_base += (size_t)(n_pos - sliding_window) * head_dim;
        n_pos = sliding_window;
    }
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;

    extern __shared__ char smem_coop[];
    __half *s_k = reinterpret_cast<__half *>(smem_coop);
    __half *s_v = s_k + (size_t)COOP_QTILE * head_dim;
    __shared__ float s_sum[8];

    float M = -INFINITY, S = 0.0f, O = 0.0f;

    int j = warp * 32 + lane;
    float qv = (j < head_dim) ? q_h[j] : 0.0f;

    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    for (int tile = 0; tile < n_pos; tile += COOP_QTILE) {
        int tile_len = min(COOP_QTILE, n_pos - tile);
        for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
            int t = i / head_dim, jj = i % head_dim;
            s_k[(size_t)t * head_dim + jj] =
                ((const __half *)kc_head)[(size_t)(tile + t) * head_dim + jj];
            s_v[(size_t)t * head_dim + jj] =
                ((const __half *)vc_head)[(size_t)(tile + t) * head_dim + jj];
        }
        __syncthreads();

        for (int t = 0; t < tile_len; t++) {
            float score = qv * __half2float(s_k[(size_t)t * head_dim + j]);
            for (int off = 16; off > 0; off >>= 1)
                score += __shfl_down_sync(0xffffffff, score, off);

            if (lane == 0) s_sum[warp] = score;
            __syncthreads();
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; w++) tot += s_sum[w];

            float a = 1.0f, b = 1.0f;
            if (lane == 0) {
                float raw = tot * scale;
                float old_M = M;
                float m_new = raw > old_M ? raw : old_M;
                a = expf(old_M - m_new);
                b = expf(raw - m_new);
                M = m_new;
                S = S * a + b;
            }
            M = __shfl_sync(0xffffffff, M, 0);
            S = __shfl_sync(0xffffffff, S, 0);
            a = __shfl_sync(0xffffffff, a, 0);
            b = __shfl_sync(0xffffffff, b, 0);
            O = O * a + b * __half2float(s_v[(size_t)t * head_dim + j]);
            __syncthreads();
        }
        __syncthreads();
    }

    float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
    if (j < head_dim)
        out[(size_t)h * head_dim + j] = O * inv_S;
}

#define DEC_SPLIT_S 8

static float *g_attn_dec_split = nullptr;
static size_t g_attn_dec_split_cap = 0;

static float *attn_dec_split_for(size_t need) {
    if (need <= g_attn_dec_split_cap)
        return g_attn_dec_split;
    if (g_attn_dec_split)
        cudaFree(g_attn_dec_split);
    g_attn_dec_split = nullptr;
    g_attn_dec_split_cap = 0;
    if (cudaMalloc((void **)&g_attn_dec_split, need) != cudaSuccess)
        return nullptr;
    g_attn_dec_split_cap = need;
    return g_attn_dec_split;
}

__global__ void attn_f16_decode_split_kernel(const float *__restrict__ q,
                                             const uint16_t *__restrict__ kc,
                                             const uint16_t *__restrict__ vc,
                                             float *__restrict__ partials,
                                             int n_heads, int n_kv_heads, int head_dim,
                                             size_t layer_base_bytes, size_t kvh_stride,
                                             int n_pos, float scale, int n_splits) {
    const int bx = blockIdx.x;
    const int h = bx % n_heads;
    const int s = bx / n_heads;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;

    extern __shared__ char smem_decsplit[];
    __half *s_k = reinterpret_cast<__half *>(smem_decsplit);
    __half *s_v = s_k + (size_t)COOP_QTILE * head_dim;
    __shared__ float s_sum[8];

    float M = -INFINITY, S = 0.0f, O = 0.0f;
    int j = warp * 32 + lane;
    float qv = (j < head_dim) ? q_h[j] : 0.0f;

    size_t layer_base = layer_base_bytes / sizeof(uint16_t);
    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    int c0 = (int)((int64_t)s * n_pos / n_splits);
    int c1 = (int)((int64_t)(s + 1) * n_pos / n_splits);

    for (int tile = c0; tile < c1; tile += COOP_QTILE) {
        int tile_len = min(COOP_QTILE, c1 - tile);
        for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
            int t = i / head_dim, jj = i % head_dim;
            s_k[(size_t)t * head_dim + jj] =
                ((const __half *)kc_head)[(size_t)(tile + t) * head_dim + jj];
            s_v[(size_t)t * head_dim + jj] =
                ((const __half *)vc_head)[(size_t)(tile + t) * head_dim + jj];
        }
        __syncthreads();

        for (int t = 0; t < tile_len; t++) {
            float score = qv * __half2float(s_k[(size_t)t * head_dim + j]);
            for (int off = 16; off > 0; off >>= 1)
                score += __shfl_down_sync(0xffffffff, score, off);
            if (lane == 0) s_sum[warp] = score;
            __syncthreads();
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; w++) tot += s_sum[w];
            float a = 1.0f, b = 1.0f;
            if (lane == 0) {
                float raw = tot * scale;
                float old_M = M;
                float m_new = raw > old_M ? raw : old_M;
                a = expf(old_M - m_new);
                b = expf(raw - m_new);
                M = m_new;
                S = S * a + b;
            }
            M = __shfl_sync(0xffffffff, M, 0);
            S = __shfl_sync(0xffffffff, S, 0);
            a = __shfl_sync(0xffffffff, a, 0);
            b = __shfl_sync(0xffffffff, b, 0);
            O = O * a + b * __half2float(s_v[(size_t)t * head_dim + j]);
            __syncthreads();
        }
        __syncthreads();
    }

    float *p = partials + ((size_t)h * n_splits + s) * (head_dim + 2);
    if (tid == 0) {
        p[0] = M;
        p[1] = S;
    }
    if (j < head_dim)
        p[2 + j] = O;
}

__global__ void attn_f16_combine_kernel(const float *__restrict__ partials,
                                        float *__restrict__ out,
                                        int n_heads, int head_dim, int n_splits) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float gM = -INFINITY;
    for (int s = 0; s < n_splits; s++) {
        const float *p = partials + ((size_t)h * n_splits + s) * (head_dim + 2);
        float m = p[0];
        gM = m > gM ? m : gM;
    }

    int j = warp * 32 + lane;
    float accS = 0.0f, accO = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        const float *p = partials + ((size_t)h * n_splits + s) * (head_dim + 2);
        float w = expf(p[0] - gM);
        accS += w * p[1];
        if (j < head_dim)
            accO += w * p[2 + j];
    }
    if (j < head_dim)
        out[(size_t)h * head_dim + j] = (accS > 0.0f) ? (accO / accS) : 0.0f;
}

extern "C" void cuda_attn_f16_decode_split(const float *q_dev, const uint16_t *kc_dev,
                                           const uint16_t *vc_dev, float *out_dev,
                                           int n_heads, int n_kv_heads, int head_dim,
                                           size_t layer_base_bytes, size_t kvh_stride,
                                           int n_pos, float scale, cudaStream_t stream) {
    int S = DEC_SPLIT_S;
    float *partials = attn_dec_split_for((size_t)n_heads * S * (head_dim + 2) * sizeof(float));
    if (!partials) return;
    dim3 grid((unsigned)n_heads * (unsigned)S);
    size_t shmem = 2 * (size_t)COOP_QTILE * head_dim * sizeof(__half);
    attn_f16_decode_split_kernel<<<grid, 256, shmem, stream>>>(
        q_dev, kc_dev, vc_dev, partials, n_heads, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride, n_pos, scale, S);
    attn_f16_combine_kernel<<<n_heads, 256, 0, stream>>>(partials, out_dev, n_heads, head_dim, S);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess)
        ERROR("cuda: decode-split attn launch failed: %s", cudaGetErrorString(cerr));
}

extern "C" void cuda_attn_dec_split_ensure(void) {
    attn_dec_split_for((size_t)64 * DEC_SPLIT_S * (256 + 2) * sizeof(float));
}

__global__ void attn_f16_decode_split_g_kernel(const float *__restrict__ q,
                                               const uint16_t *__restrict__ kc,
                                               const uint16_t *__restrict__ vc,
                                               float *__restrict__ partials,
                                               int n_heads, int n_kv_heads, int head_dim,
                                               size_t layer_base_bytes, size_t kvh_stride,
                                               float scale,
                                               const cuda_decode_params *dp,
                                               int sliding_window, int n_splits) {
    const int bx = blockIdx.x;
    const int h = bx % n_heads;
    const int s = bx / n_heads;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    int n_pos = dp ? dp->n_pos : 0;
    size_t layer_base = layer_base_bytes / sizeof(uint16_t);
    if (sliding_window > 0 && n_pos > sliding_window) {
        layer_base += (size_t)(n_pos - sliding_window) * head_dim;
        n_pos = sliding_window;
    }

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;

    extern __shared__ char smem_decsplit_g[];
    __half *s_k = reinterpret_cast<__half *>(smem_decsplit_g);
    __half *s_v = s_k + (size_t)COOP_QTILE * head_dim;
    __shared__ float s_sum[8];

    float M = -INFINITY, S = 0.0f, O = 0.0f;
    int j = warp * 32 + lane;
    float qv = (j < head_dim) ? q_h[j] : 0.0f;

    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    int c0, c1;
    if (n_pos < 256) {
        c0 = 0;
        c1 = (s == 0) ? n_pos : 0;
    } else {
        c0 = (int)((int64_t)s * n_pos / n_splits);
        c1 = (int)((int64_t)(s + 1) * n_pos / n_splits);
    }

    for (int tile = c0; tile < c1; tile += COOP_QTILE) {
        int tile_len = min(COOP_QTILE, c1 - tile);
        for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
            int t = i / head_dim, jj = i % head_dim;
            s_k[(size_t)t * head_dim + jj] =
                ((const __half *)kc_head)[(size_t)(tile + t) * head_dim + jj];
            s_v[(size_t)t * head_dim + jj] =
                ((const __half *)vc_head)[(size_t)(tile + t) * head_dim + jj];
        }
        __syncthreads();

        for (int t = 0; t < tile_len; t++) {
            float score = qv * __half2float(s_k[(size_t)t * head_dim + j]);
            for (int off = 16; off > 0; off >>= 1)
                score += __shfl_down_sync(0xffffffff, score, off);
            if (lane == 0) s_sum[warp] = score;
            __syncthreads();
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; w++) tot += s_sum[w];
            float a = 1.0f, b = 1.0f;
            if (lane == 0) {
                float raw = tot * scale;
                float old_M = M;
                float m_new = raw > old_M ? raw : old_M;
                a = expf(old_M - m_new);
                b = expf(raw - m_new);
                M = m_new;
                S = S * a + b;
            }
            M = __shfl_sync(0xffffffff, M, 0);
            S = __shfl_sync(0xffffffff, S, 0);
            a = __shfl_sync(0xffffffff, a, 0);
            b = __shfl_sync(0xffffffff, b, 0);
            O = O * a + b * __half2float(s_v[(size_t)t * head_dim + j]);
            __syncthreads();
        }
        __syncthreads();
    }

    float *p = partials + ((size_t)h * n_splits + s) * (head_dim + 2);
    if (tid == 0) {
        p[0] = M;
        p[1] = S;
    }

    if (j < head_dim)
        p[2 + j] = (c1 > c0) ? O : 0.0f;
}

extern "C" void cuda_attn_f16_decode_split_g(const float *q_dev, const uint16_t *kc_dev,
                                             const uint16_t *vc_dev, float *out_dev,
                                             int n_heads, int n_kv_heads, int head_dim,
                                             size_t layer_base_bytes, size_t kvh_stride,
                                             float scale, const int *params_dev,
                                             int sliding_window, cudaStream_t stream) {
    int S = DEC_SPLIT_S;

    float *partials = attn_dec_split_for((size_t)n_heads * S * (head_dim + 2) * sizeof(float));
    if (!partials) return;
    dim3 grid((unsigned)n_heads * (unsigned)S);
    size_t shmem = 2 * (size_t)COOP_QTILE * head_dim * sizeof(__half);
    attn_f16_decode_split_g_kernel<<<grid, 256, shmem, stream>>>(
        q_dev, kc_dev, vc_dev, partials, n_heads, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride, scale,
        (const cuda_decode_params *)params_dev, sliding_window, S);
    attn_f16_combine_kernel<<<n_heads, 256, 0, stream>>>(partials, out_dev, n_heads, head_dim, S);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess)
        ERROR("cuda: decode-split-g attn launch failed: %s", cudaGetErrorString(cerr));
}

extern "C" void cuda_attn_f16_gqa_g(const float *q_dev, const uint16_t *kc_dev,
                                    const uint16_t *vc_dev, float *out_dev, int n_heads,
                                    int n_kv_heads, int head_dim, size_t layer_base_bytes,
                                    size_t kvh_stride, int sliding_window, float scale,
                                    int n_queries, const int *params_dev,
                                    cudaStream_t stream) {

    if (0) {
        dim3 grid((unsigned)n_heads);
        size_t shmem = 2 * (size_t)COOP_QTILE * head_dim * sizeof(__half);
        attn_f16_decode_coop_kernel<<<grid, 256, shmem, stream>>>(
            q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
            layer_base_bytes, kvh_stride, scale, params_dev, sliding_window);
        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess)
            ERROR("cuda: coop attn launch failed: %s", cudaGetErrorString(cerr));
        return;
    }

    if (head_dim == 256) {
        cuda_attn_f16_decode_split_g(q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads,
                                     head_dim, layer_base_bytes, kvh_stride, scale,
                                     params_dev, sliding_window, stream);
        return;
    }
    int n_kv_groups = n_heads / n_kv_heads;
    int qb = (n_kv_groups + n_queries - 1) / n_queries;
    dim3 grid((unsigned)n_kv_heads * (unsigned)qb);
    int tile_size = attn_tile_size(head_dim);
    size_t shmem = ((size_t)n_queries * head_dim + 2 * tile_size * head_dim) * sizeof(float);
    attn_f16_tiled_gqa_g_kernel<<<grid, 32 * n_queries, shmem, stream>>>(
        q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride, sliding_window, scale, n_queries,
        (const cuda_decode_params *)params_dev, tile_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: attn_f16_gqa_g_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

static float *g_attn_split = nullptr;
static size_t g_attn_split_cap = 0;

static float *attn_split_for(size_t need) {
    if (need <= g_attn_split_cap)
        return g_attn_split;
    if (g_attn_split)
        cudaFree(g_attn_split);
    g_attn_split = nullptr;
    g_attn_split_cap = 0;
    if (cudaMalloc((void **)&g_attn_split, need) != cudaSuccess)
        return nullptr;
    g_attn_split_cap = need;
    return g_attn_split;
}

__global__ void attn_f16_split_g_kernel(const float *__restrict__ q,
                                        const uint16_t *__restrict__ kc,
                                        const uint16_t *__restrict__ vc,
                                        float *__restrict__ partials,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        size_t layer_base_bytes, size_t kvh_stride,
                                        int sliding_window, float scale, int n_queries,
                                        int n_splits,
                                        const cuda_decode_params *dp, int tile_size) {
    const int bx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int w = tid >> 5;

    int n_kv_groups = n_heads / n_kv_heads;
    int qb = (n_kv_groups + n_queries - 1) / n_queries;
    int nqb = n_kv_heads * qb;
    int b = bx % nqb;
    int s = bx / nqb;

    int n_pos_total = dp->n_pos;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_total > sliding_window) {
        attn_start = n_pos_total - sliding_window;
        n_pos_total = sliding_window;
    }
    size_t layer_base = (layer_base_bytes + (size_t)attn_start * head_dim * sizeof(uint16_t)) /
                        sizeof(uint16_t);
    int n_pos = n_pos_total;

    int kv_head = b / qb;
    int qslot0 = (b % qb) * n_queries;
    int nq = min(n_queries, n_kv_groups - qslot0);
    int active = (w < nq) ? 1 : 0;
    int h = kv_head * n_kv_groups + qslot0 + w;

    int c0 = (int)((int64_t)s * n_pos / n_splits);
    int c1 = (int)((int64_t)(s + 1) * n_pos / n_splits);

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_k_tile = &smem[(size_t)n_queries * head_dim];
    float *s_v_tile = &smem[(size_t)n_queries * head_dim + tile_size * head_dim];

    float *s_qw = s_q + (size_t)w * head_dim;
    if (active) {
        const float *q_h = q + (size_t)h * head_dim;
        int chunk = (head_dim + 31) / 32;
        int base = lane * chunk;
        int end = min(base + chunk, head_dim);
        for (int i = base; i < end; i++) {
            s_qw[i] = q_h[i];
        }
    }
    __syncthreads();

    double M = -INFINITY;
    double S = 0.0;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = lane * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0.0f};

    const uint16_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride;
    const uint16_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride;

    for (int tile_start = c0; tile_start < c1; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, c1);
        int tile_len = tile_end - tile_start;

        {
            int tile_pairs = (tile_len * head_dim) >> 1;
            const __half2 *k2 = (const __half2 *)(kc_head + (size_t)tile_start * head_dim);
            const __half2 *v2 = (const __half2 *)(vc_head + (size_t)tile_start * head_dim);
            float2 *s_k2 = (float2 *)s_k_tile;
            float2 *s_v2 = (float2 *)s_v_tile;
            for (int idx = tid; idx < tile_pairs; idx += blockDim.x) {
                s_k2[idx] = __half22float2(k2[idx]);
                s_v2[idx] = __half22float2(v2[idx]);
            }
            if (((tile_len * head_dim) & 1) && tid == 0) {
                int tail = tile_len * head_dim - 1;
                int bb = tail / head_dim;
                int ii = tail % head_dim;
                const uint16_t *k_t = kc_head + (size_t)(tile_start + bb) * head_dim;
                const uint16_t *v_t = vc_head + (size_t)(tile_start + bb) * head_dim;
                s_k_tile[tail] = __half2float(__ushort_as_half(k_t[ii]));
                s_v_tile[tail] = __half2float(__ushort_as_half(v_t[ii]));
            }
        }
        __syncthreads();

        if (active) {
            for (int t_local = 0; t_local < tile_len; t_local++) {
                const float *k_tile_row = &s_k_tile[t_local * head_dim];
                const float *v_tile_row = &s_v_tile[t_local * head_dim];

                float score = 0.0f;
                for (int i = lane; i < head_dim; i += 32) {
                    score += s_qw[i] * k_tile_row[i];
                }

                for (int offset = 16; offset > 0; offset >>= 1)
                    score += __shfl_down_sync(0xffffffff, score, offset);

                double raw_score = (double)score * (double)scale;
                double ms = 1.0;
                double vs = 1.0;
                if (lane == 0) {
                    if (raw_score > M) {
                        double old_M = M;
                        M = raw_score;
                        ms = exp(old_M - M);
                        S *= ms;
                        vs = 1.0;
                    } else {
                        ms = 1.0;
                        vs = exp(raw_score - M);
                    }
                    S += vs;
                }

                {
                    double tmp_M = M, tmp_S = S, tmp_ms = ms, tmp_vs = vs;
                    tmp_M = __shfl_sync(0xffffffff, tmp_M, 0);
                    tmp_S = __shfl_sync(0xffffffff, tmp_S, 0);
                    tmp_ms = __shfl_sync(0xffffffff, tmp_ms, 0);
                    tmp_vs = __shfl_sync(0xffffffff, tmp_vs, 0);
                    M = tmp_M; S = tmp_S; ms = tmp_ms; vs = tmp_vs;
                }

                float f_ms = (float)ms;
                float f_vs = (float)vs;
                for (int idx = start_idx; idx < end_idx; idx++) {
                    int local_idx = idx - start_idx;
                    O_local[local_idx] = O_local[local_idx] * f_ms + f_vs * v_tile_row[idx];
                }
            }
        }
        __syncthreads();
    }

    {
        size_t stride = (size_t)n_queries * (2 + head_dim);
        float *dst = partials + ((size_t)s * nqb + b) * stride + (size_t)w * (2 + head_dim);
        if (active) {
            dst[0] = (float)M;
            dst[1] = (float)S;
            for (int idx = start_idx; idx < end_idx; idx++) {
                int local_idx = idx - start_idx;
                dst[2 + idx] = O_local[local_idx];
            }
        } else {
            dst[0] = -INFINITY;
            dst[1] = 0.0f;
            for (int idx = start_idx; idx < end_idx; idx++) {
                dst[2 + idx] = 0.0f;
            }
        }
    }
}

__global__ void attn_f16_combine_kernel(const float *__restrict__ partials,
                                        float *__restrict__ out,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        int n_queries, int n_splits) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int w = tid >> 5;

    int n_kv_groups = n_heads / n_kv_heads;
    int qb = (n_kv_groups + n_queries - 1) / n_queries;
    int nqb = n_kv_heads * qb;
    if (b >= nqb) return;
    int kv_head = b / qb;
    int qslot0 = (b % qb) * n_queries;
    int nq = min(n_queries, n_kv_groups - qslot0);
    if (w >= nq) return;
    int h = kv_head * n_kv_groups + qslot0 + w;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = lane * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);

    size_t stride = (size_t)n_queries * (2 + head_dim);
    double m = -INFINITY;
    for (int s = 0; s < n_splits; s++) {
        const float *src = partials + ((size_t)s * nqb + b) * stride + (size_t)w * (2 + head_dim);
        double Ms = (double)src[0];
        if (Ms > m) m = Ms;
    }
    double S = 0.0;
    float O_local[16] = {0.0f};
    for (int s = 0; s < n_splits; s++) {
        const float *src = partials + ((size_t)s * nqb + b) * stride + (size_t)w * (2 + head_dim);
        double Ms = (double)src[0];
        double Ss = (double)src[1];
        double wx = exp(Ms - m);
        S += Ss * wx;
        float fw = (float)wx;
        for (int idx = start_idx; idx < end_idx; idx++) {
            int local_idx = idx - start_idx;
            O_local[local_idx] += fw * src[2 + idx];
        }
    }
    float inv_S = (S > 0.0) ? (1.0f / (float)S) : 0.0f;
    float *out_h = out + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

extern "C" void cuda_attn_f16_split_g(const float *q_dev, const uint16_t *kc_dev,
                                      const uint16_t *vc_dev, float *out_dev, int n_heads,
                                      int n_kv_heads, int head_dim, size_t layer_base_bytes,
                                      size_t kvh_stride, int sliding_window, float scale,
                                      int n_queries, int n_splits, const int *params_dev,
                                      cudaStream_t stream) {
    int n_kv_groups = n_heads / n_kv_heads;
    int qb = (n_kv_groups + n_queries - 1) / n_queries;
    int nqb = n_kv_heads * qb;
    dim3 grid((unsigned)nqb * (unsigned)n_splits);
    int tile_size = attn_tile_size(head_dim);
    size_t shmem = ((size_t)n_queries * head_dim + 2 * tile_size * head_dim) * sizeof(float);

    float *partials = nullptr;
    {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &st);
        if (st == cudaStreamCaptureStatusNone) {
            size_t need = (size_t)n_splits * nqb * n_queries * (2 + head_dim) * sizeof(float);
            partials = attn_split_for(need);
        }
    }
    if (!partials) {
        cuda_attn_f16_gqa_g(q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
                            layer_base_bytes, kvh_stride, sliding_window, scale, n_queries,
                            params_dev, stream);
        return;
    }
    attn_f16_split_g_kernel<<<grid, 32 * n_queries, shmem, stream>>>(
        q_dev, kc_dev, vc_dev, partials, n_heads, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride, sliding_window, scale, n_queries, n_splits,
        (const cuda_decode_params *)params_dev, tile_size);
    dim3 cgrid(nqb);
    attn_f16_combine_kernel<<<cgrid, 32 * n_queries, 0, stream>>>(
        partials, out_dev, n_heads, n_kv_heads, head_dim, n_queries, n_splits);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: attn_f16_split_g_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

extern "C" void cuda_attn_f16_g(const float *q_dev, const uint16_t *kc_dev,
                                const uint16_t *vc_dev, float *out_dev, int n_heads,
                                int n_kv_heads, int head_dim, size_t layer_base_bytes,
                                size_t kvh_stride, int sliding_window, float scale,
                                const int *params_dev, cudaStream_t stream) {

    dim3 grid(n_heads);
    int tile_size = attn_tile_size(head_dim);
    size_t shmem = (1 * head_dim + 2 * tile_size * head_dim) * sizeof(float);
    attn_f16_tiled_g_kernel<<<grid, 32, shmem, stream>>>(
        q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride, sliding_window, scale,
        (const cuda_decode_params *)params_dev, tile_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: attn_f16_g_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

__global__ void attn_q8_tiled_g_kernel(const float *__restrict__ q,
                                       const uint8_t *__restrict__ kc,
                                       const uint8_t *__restrict__ vc,
                                       float *__restrict__ out,
                                       int n_heads, int n_kv_heads, int head_dim,
                                       size_t layer_base_bytes, size_t kvh_stride_bytes,
                                       int sliding_window, float scale, int n_blocks,
                                       const cuda_decode_params *dp, int tile_size) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if (h >= n_heads) return;

    int n_pos_total = dp->n_pos;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_total > sliding_window) {
        attn_start = n_pos_total - sliding_window;
        n_pos_total = sliding_window;
    }
    size_t row_bytes = (size_t)n_blocks * KV_Q8_0_ATTN_BLK_BYTES;
    size_t layer_base = layer_base_bytes + (size_t)attn_start * row_bytes;
    int n_pos = n_pos_total;

    extern __shared__ float smem[];
    float *s_q = smem;

    uint8_t *s_raw_k = (uint8_t *)&smem[1 * head_dim];
    uint8_t *s_raw_v = (uint8_t *)&smem[1 * head_dim] + (size_t)tile_size * row_bytes;
    float *s_k_tile = (float *)((uint8_t *)&smem[1 * head_dim] +
                                2 * (size_t)tile_size * row_bytes);
    float *s_v_tile = s_k_tile + (size_t)tile_size * head_dim;

    int n_kv_groups = n_heads / n_kv_heads;
    int kv_head = h / n_kv_groups;

    const float *q_h = q + (size_t)h * head_dim;
    {
        int chunk = (head_dim + blockDim.x - 1) / blockDim.x;
        int base = tid * chunk;
        int end = min(base + chunk, head_dim);
        for (int i = base; i < end; i++) {
            s_q[i] = q_h[i];
        }
    }
    __syncthreads();

    double M = -INFINITY;
    double S = 0.0;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = tid * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0.0f};

    const uint8_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride_bytes;
    const uint8_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride_bytes;

    for (int tile_start = 0; tile_start < n_pos; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, n_pos);
        int tile_len = tile_end - tile_start;

        {
            size_t tile_raw = (size_t)tile_len * row_bytes;
            const uint8_t *kbase = kc_head + (size_t)tile_start * row_bytes;
            const uint8_t *vbase = vc_head + (size_t)tile_start * row_bytes;
            for (size_t by = tid; by < tile_raw; by += blockDim.x) {
                s_raw_k[by] = kbase[by];
                s_raw_v[by] = vbase[by];
            }
        }
        __syncthreads();

        {
            int nq = tile_len * n_blocks * 8;
            for (int qq = tid; qq < nq; qq += blockDim.x) {
                int q = qq & 7;
                int blk = (qq >> 3) % n_blocks;
                int brow = qq / (n_blocks * 8);
                const uint8_t *kb = s_raw_k + (size_t)brow * row_bytes + (size_t)blk * 34;
                const uint8_t *vb = s_raw_v + (size_t)brow * row_bytes + (size_t)blk * 34;
                float kd = __half2float(__ushort_as_half(*(const uint16_t *)kb));
                float vd = __half2float(__ushort_as_half(*(const uint16_t *)vb));
                uint16_t kh0 = *(const uint16_t *)(kb + 2 + (q << 2));
                uint16_t kh1 = *(const uint16_t *)(kb + 2 + (q << 2) + 2);
                uint16_t vh0 = *(const uint16_t *)(vb + 2 + (q << 2));
                uint16_t vh1 = *(const uint16_t *)(vb + 2 + (q << 2) + 2);
                float *ok = s_k_tile + (size_t)brow * head_dim + blk * 32 + (q << 2);
                float *ov = s_v_tile + (size_t)brow * head_dim + blk * 32 + (q << 2);

                int base = blk * 32 + (q << 2);
                if (base + 4 <= head_dim) {
                    ok[0] = kd * (float)(int8_t)(kh0 & 0xFF);
                    ok[1] = kd * (float)(int8_t)(kh0 >> 8);
                    ok[2] = kd * (float)(int8_t)(kh1 & 0xFF);
                    ok[3] = kd * (float)(int8_t)(kh1 >> 8);
                    ov[0] = vd * (float)(int8_t)(vh0 & 0xFF);
                    ov[1] = vd * (float)(int8_t)(vh0 >> 8);
                    ov[2] = vd * (float)(int8_t)(vh1 & 0xFF);
                    ov[3] = vd * (float)(int8_t)(vh1 >> 8);
                } else if (base < head_dim) {
                    float kv_[4] = {kd * (float)(int8_t)(kh0 & 0xFF),
                                    kd * (float)(int8_t)(kh0 >> 8),
                                    kd * (float)(int8_t)(kh1 & 0xFF),
                                    kd * (float)(int8_t)(kh1 >> 8)};
                    float vv_[4] = {vd * (float)(int8_t)(vh0 & 0xFF),
                                    vd * (float)(int8_t)(vh0 >> 8),
                                    vd * (float)(int8_t)(vh1 & 0xFF),
                                    vd * (float)(int8_t)(vh1 >> 8)};
                    int rem = head_dim - base;
                    for (int e = 0; e < rem; e++) {
                        ok[e] = kv_[e];
                        ov[e] = vv_[e];
                    }
                }
            }
        }
        __syncthreads();

        for (int t_local = 0; t_local < tile_len; t_local++) {
            const float *k_tile_row = &s_k_tile[t_local * head_dim];
            const float *v_tile_row = &s_v_tile[t_local * head_dim];

            float score = 0.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                score += s_q[i] * k_tile_row[i];
            }

            for (int offset = 16; offset > 0; offset >>= 1)
                score += __shfl_down_sync(0xffffffff, score, offset);

            double raw_score = (double)score * (double)scale;
            double ms = 1.0;
            double vs = 1.0;
            if (lane == 0) {
                if (raw_score > M) {
                    double old_M = M;
                    M = raw_score;
                    ms = exp(old_M - M);
                    S *= ms;
                    vs = 1.0;
                } else {
                    ms = 1.0;
                    vs = exp(raw_score - M);
                }
                S += vs;
            }

            {
                double tmp_M = M, tmp_S = S, tmp_ms = ms, tmp_vs = vs;
                tmp_M = __shfl_sync(0xffffffff, tmp_M, 0);
                tmp_S = __shfl_sync(0xffffffff, tmp_S, 0);
                tmp_ms = __shfl_sync(0xffffffff, tmp_ms, 0);
                tmp_vs = __shfl_sync(0xffffffff, tmp_vs, 0);
                M = tmp_M; S = tmp_S; ms = tmp_ms; vs = tmp_vs;
            }

            float f_ms = (float)ms;
            float f_vs = (float)vs;
            for (int idx = start_idx; idx < end_idx; idx++) {
                int local_idx = idx - start_idx;
                O_local[local_idx] = O_local[local_idx] * f_ms + f_vs * v_tile_row[idx];
            }
        }
        __syncthreads();
    }

    float inv_S = (S > 0.0) ? (1.0f / (float)S) : 0.0f;
    float *out_h = out + (size_t)h * head_dim;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int local_idx = idx - start_idx;
        out_h[idx] = O_local[local_idx] * inv_S;
    }
}

__global__ void attn_q8_tiled_gqa_g_kernel(const float *__restrict__ q,
                                           const uint8_t *__restrict__ kc,
                                           const uint8_t *__restrict__ vc,
                                           float *__restrict__ out,
                                           int n_heads, int n_kv_heads, int head_dim,
                                           size_t layer_base_bytes, size_t kvh_stride_bytes,
                                           int sliding_window, float scale, int n_blocks,
                                           int n_queries,
                                           const cuda_decode_params *dp, int tile_size) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int w = tid >> 5;

    int n_pos_total = dp->n_pos;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_total > sliding_window) {
        attn_start = n_pos_total - sliding_window;
        n_pos_total = sliding_window;
    }
    size_t row_bytes = (size_t)n_blocks * KV_Q8_0_ATTN_BLK_BYTES;
    size_t layer_base = layer_base_bytes + (size_t)attn_start * row_bytes;
    int n_pos = n_pos_total;

    int n_kv_groups = n_heads / n_kv_heads;
    int qb = (n_kv_groups + n_queries - 1) / n_queries;
    int kv_head = b / qb;
    int qslot0 = (b % qb) * n_queries;
    int nq = min(n_queries, n_kv_groups - qslot0);
    int active = (w < nq) ? 1 : 0;
    int h = kv_head * n_kv_groups + qslot0 + w;

    extern __shared__ float smem[];
    float *s_q = smem;
    uint8_t *s_raw_k = (uint8_t *)&smem[(size_t)n_queries * head_dim];
    uint8_t *s_raw_v = (uint8_t *)&smem[(size_t)n_queries * head_dim] +
                       (size_t)tile_size * row_bytes;
    float *s_k_tile = (float *)((uint8_t *)&smem[(size_t)n_queries * head_dim] +
                                2 * (size_t)tile_size * row_bytes);
    float *s_v_tile = s_k_tile + (size_t)tile_size * head_dim;

    float *s_qw = s_q + (size_t)w * head_dim;
    if (active) {
        const float *q_h = q + (size_t)h * head_dim;
        int chunk = (head_dim + 31) / 32;
        int base = lane * chunk;
        int end = min(base + chunk, head_dim);
        for (int i = base; i < end; i++) {
            s_qw[i] = q_h[i];
        }
    }
    __syncthreads();

    double M = -INFINITY;
    double S = 0.0;

    const int per_thread = (head_dim + 31) / 32;
    const int start_idx = lane * per_thread;
    const int end_idx = min(start_idx + per_thread, head_dim);
    float O_local[16] = {0.0f};

    const uint8_t *kc_head = kc + layer_base + (size_t)kv_head * kvh_stride_bytes;
    const uint8_t *vc_head = vc + layer_base + (size_t)kv_head * kvh_stride_bytes;

    for (int tile_start = 0; tile_start < n_pos; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, n_pos);
        int tile_len = tile_end - tile_start;

        {
            size_t tile_raw = (size_t)tile_len * row_bytes;
            const uint8_t *kbase = kc_head + (size_t)tile_start * row_bytes;
            const uint8_t *vbase = vc_head + (size_t)tile_start * row_bytes;
            for (size_t by = tid; by < tile_raw; by += blockDim.x) {
                s_raw_k[by] = kbase[by];
                s_raw_v[by] = vbase[by];
            }
        }
        __syncthreads();
        {
            int nq2 = tile_len * n_blocks * 8;
            for (int qq = tid; qq < nq2; qq += blockDim.x) {
                int q = qq & 7;
                int blk = (qq >> 3) % n_blocks;
                int brow = qq / (n_blocks * 8);
                const uint8_t *kb = s_raw_k + (size_t)brow * row_bytes + (size_t)blk * 34;
                const uint8_t *vb = s_raw_v + (size_t)brow * row_bytes + (size_t)blk * 34;
                float kd = __half2float(__ushort_as_half(*(const uint16_t *)kb));
                float vd = __half2float(__ushort_as_half(*(const uint16_t *)vb));
                uint16_t kh0 = *(const uint16_t *)(kb + 2 + (q << 2));
                uint16_t kh1 = *(const uint16_t *)(kb + 2 + (q << 2) + 2);
                uint16_t vh0 = *(const uint16_t *)(vb + 2 + (q << 2));
                uint16_t vh1 = *(const uint16_t *)(vb + 2 + (q << 2) + 2);
                float *ok = s_k_tile + (size_t)brow * head_dim + blk * 32 + (q << 2);
                float *ov = s_v_tile + (size_t)brow * head_dim + blk * 32 + (q << 2);
                int base = blk * 32 + (q << 2);
                if (base + 4 <= head_dim) {
                    ok[0] = kd * (float)(int8_t)(kh0 & 0xFF);
                    ok[1] = kd * (float)(int8_t)(kh0 >> 8);
                    ok[2] = kd * (float)(int8_t)(kh1 & 0xFF);
                    ok[3] = kd * (float)(int8_t)(kh1 >> 8);
                    ov[0] = vd * (float)(int8_t)(vh0 & 0xFF);
                    ov[1] = vd * (float)(int8_t)(vh0 >> 8);
                    ov[2] = vd * (float)(int8_t)(vh1 & 0xFF);
                    ov[3] = vd * (float)(int8_t)(vh1 >> 8);
                } else if (base < head_dim) {
                    float kv_[4] = {kd * (float)(int8_t)(kh0 & 0xFF),
                                    kd * (float)(int8_t)(kh0 >> 8),
                                    kd * (float)(int8_t)(kh1 & 0xFF),
                                    kd * (float)(int8_t)(kh1 >> 8)};
                    float vv_[4] = {vd * (float)(int8_t)(vh0 & 0xFF),
                                    vd * (float)(int8_t)(vh0 >> 8),
                                    vd * (float)(int8_t)(vh1 & 0xFF),
                                    vd * (float)(int8_t)(vh1 >> 8)};
                    int rem = head_dim - base;
                    for (int e = 0; e < rem; e++) {
                        ok[e] = kv_[e];
                        ov[e] = vv_[e];
                    }
                }
            }
        }
        __syncthreads();

        if (active) {

            for (int t_local = 0; t_local < tile_len; t_local++) {
                const float *k_tile_row = &s_k_tile[t_local * head_dim];
                const float *v_tile_row = &s_v_tile[t_local * head_dim];

                float score = 0.0f;
                for (int i = lane; i < head_dim; i += 32) {
                    score += s_qw[i] * k_tile_row[i];
                }

                for (int offset = 16; offset > 0; offset >>= 1)
                    score += __shfl_down_sync(0xffffffff, score, offset);

                float raw_score = score * scale;
                float ms = 1.0f;
                float vs = 1.0f;
                if (lane == 0) {
                    if (raw_score > M) {
                        float old_M = M;
                        M = raw_score;
                        ms = expf(old_M - M);
                        S *= ms;
                        vs = 1.0f;
                    } else {
                        ms = 1.0f;
                        vs = expf(raw_score - M);
                    }
                    S += vs;
                }

                M = __shfl_sync(0xffffffff, M, 0);
                S = __shfl_sync(0xffffffff, S, 0);
                ms = __shfl_sync(0xffffffff, ms, 0);
                vs = __shfl_sync(0xffffffff, vs, 0);

                for (int idx = start_idx; idx < end_idx; idx++) {
                    int local_idx = idx - start_idx;
                    O_local[local_idx] = O_local[local_idx] * ms + vs * v_tile_row[idx];
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv_S = (S > 0.0) ? (1.0f / (float)S) : 0.0f;
        float *out_h = out + (size_t)h * head_dim;
        for (int idx = start_idx; idx < end_idx; idx++) {
            int local_idx = idx - start_idx;
            out_h[idx] = O_local[local_idx] * inv_S;
        }
    }
}

extern "C" void cuda_attn_q8_gqa_g(const float *q_dev, const uint8_t *kc_dev,
                                   const uint8_t *vc_dev, float *out_dev, int n_heads,
                                   int n_kv_heads, int head_dim, size_t layer_base_bytes,
                                   size_t kvh_stride_bytes, int sliding_window, float scale,
                                   int n_blocks, int n_queries, const int *params_dev,
                                   cudaStream_t stream) {
    int n_kv_groups = n_heads / n_kv_heads;
    int qb = (n_kv_groups + n_queries - 1) / n_queries;
    dim3 grid((unsigned)n_kv_heads * (unsigned)qb);
    size_t row_bytes = (size_t)n_blocks * KV_Q8_0_ATTN_BLK_BYTES;

    int tile_size = attn_tile_size(head_dim);
    size_t shmem = 0;
    for (; tile_size >= 1; tile_size /= 2) {
        shmem = (size_t)n_queries * head_dim * sizeof(float) +
                2 * (size_t)tile_size * row_bytes +
                2 * (size_t)tile_size * head_dim * sizeof(float);
        if (shmem <= 46 * 1024)
            break;
    }
    if (tile_size < 1)
        tile_size = 1;
    attn_q8_tiled_gqa_g_kernel<<<grid, 32 * n_queries, shmem, stream>>>(
        q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride_bytes, sliding_window, scale, n_blocks, n_queries,
        (const cuda_decode_params *)params_dev, tile_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: attn_q8_gqa_g_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

extern "C" void cuda_attn_q8_g(const float *q_dev, const uint8_t *kc_dev,
                                const uint8_t *vc_dev, float *out_dev, int n_heads,
                                int n_kv_heads, int head_dim, size_t layer_base_bytes,
                                size_t kvh_stride_bytes, int sliding_window, float scale,
                                int n_blocks, const int *params_dev, cudaStream_t stream) {

    dim3 grid(n_heads);
    int tile_size = attn_tile_size(head_dim);

    size_t row_bytes = (size_t)n_blocks * KV_Q8_0_ATTN_BLK_BYTES;
    size_t shmem = (size_t)head_dim * sizeof(float) +
                   2 * (size_t)tile_size * row_bytes +
                   2 * (size_t)tile_size * head_dim * sizeof(float);
    attn_q8_tiled_g_kernel<<<grid, 32, shmem, stream>>>(
        q_dev, kc_dev, vc_dev, out_dev, n_heads, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride_bytes, sliding_window, scale, n_blocks,
        (const cuda_decode_params *)params_dev, tile_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: attn_q8_g_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

__global__ void kv_put_f16_g_kernel(const float *k_in, const float *v_in,
                                    uint16_t *kd, uint16_t *vd,
                                    int n_kv_heads, int head_dim,
                                    size_t layer_base_bytes, size_t kvh_stride, int n_ctx,
                                    const cuda_decode_params *dp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_kv_heads * head_dim;
    if (idx >= total) return;

    int head = idx / head_dim;
    int dim = idx % head_dim;

    size_t layer_off = layer_base_bytes / sizeof(uint16_t) + (size_t)head * (size_t)n_ctx * head_dim;
    size_t pos_off = layer_off + (size_t)dp->pos * head_dim + dim;

    kd[pos_off] = __half_as_ushort(__float2half_rn(k_in[idx]));
    vd[pos_off] = __half_as_ushort(__float2half_rn(v_in[idx]));
}

extern "C" void cuda_kv_put_f16_g(const float *k_in, const float *v_in,
                                  uint16_t *kd, uint16_t *vd, int n_kv_heads, int head_dim,
                                  size_t layer_base_bytes, size_t kvh_stride, int n_ctx,
                                  const int *params_dev, cudaStream_t stream) {

    int total = n_kv_heads * head_dim;
    int block = 256;
    dim3 grid((total + block - 1) / block);
    kv_put_f16_g_kernel<<<grid, block, 0, stream>>>(
        k_in, v_in, kd, vd, n_kv_heads, head_dim, layer_base_bytes, kvh_stride, n_ctx,
        (const cuda_decode_params *)params_dev);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: kv_put_f16_g_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

#define KV_Q8_0_PUT_BLK 32
#define KV_Q8_0_PUT_BLK_BYTES 34

static __device__ __forceinline__ void kv_quant_q8_0_block(const float *x, int n,
                                                           uint8_t *blk_out) {
    float vals[KV_Q8_0_PUT_BLK];
#pragma unroll
    for (int j = 0; j < KV_Q8_0_PUT_BLK; j++)
        vals[j] = (j < n) ? x[j] : 0.0f;
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < KV_Q8_0_PUT_BLK; j++) {
        float a = fabsf(vals[j]);
        if (a > amax) amax = a;
    }
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    uint16_t dbits = __half_as_ushort(__float2half(d));
    blk_out[0] = (uint8_t)(dbits & 0xFF);
    blk_out[1] = (uint8_t)((dbits >> 8) & 0xFF);
#pragma unroll
    for (int j = 0; j < KV_Q8_0_PUT_BLK; j++) {
        int q = (int)roundf(vals[j] * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        blk_out[2 + j] = (uint8_t)(int8_t)q;
    }
}

__global__ void kv_put_q8_kernel(const float *k_in, const float *v_in,
                                 uint8_t *kd, uint8_t *vd,
                                 int n_kv_heads, int head_dim,
                                 size_t layer_base_bytes, size_t kvh_stride_bytes,
                                 int pos, int n_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_kv_heads * n_blocks;
    if (idx >= total) return;

    int head = idx / n_blocks;
    int b = idx % n_blocks;

    size_t slot_base = layer_base_bytes + (size_t)head * kvh_stride_bytes +
                       (size_t)pos * n_blocks * KV_Q8_0_PUT_BLK_BYTES;
    uint8_t *k_blk = kd + slot_base + (size_t)b * KV_Q8_0_PUT_BLK_BYTES;
    uint8_t *v_blk = vd + slot_base + (size_t)b * KV_Q8_0_PUT_BLK_BYTES;

    int base = b * KV_Q8_0_PUT_BLK;
    int n = min(KV_Q8_0_PUT_BLK, head_dim - base);
    kv_quant_q8_0_block(k_in + (size_t)head * head_dim + base, n, k_blk);
    kv_quant_q8_0_block(v_in + (size_t)head * head_dim + base, n, v_blk);
}

extern "C" void cuda_kv_put_q8(const float *k_in, const float *v_in,
                               uint8_t *kd, uint8_t *vd,
                               int n_kv_heads, int head_dim,
                               size_t layer_base_bytes, size_t kvh_stride_bytes,
                               int pos, int n_blocks, cudaStream_t stream) {
    int total = n_kv_heads * n_blocks;
    int block = 256;
    dim3 grid((total + block - 1) / block);
    kv_put_q8_kernel<<<grid, block, 0, stream>>>(
        k_in, v_in, kd, vd, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride_bytes, pos, n_blocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: kv_put_q8_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

__global__ void kv_put_q8_batch_kernel(const float *k_in, const float *v_in,
                                       uint8_t *kd, uint8_t *vd,
                                       int n_kv_heads, int head_dim,
                                       size_t layer_base_bytes, size_t kvh_stride_bytes,
                                       int pos_start, int n_seq, int n_blocks,
                                       int in_row_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_seq * n_kv_heads * n_blocks;
    if (idx >= total) return;

    int tmp = idx / n_blocks;
    int b = idx % n_blocks;
    int row = tmp / n_kv_heads;
    int head = tmp % n_kv_heads;
    int pos = pos_start + row;

    size_t slot_base = layer_base_bytes + (size_t)head * kvh_stride_bytes +
                       (size_t)pos * n_blocks * KV_Q8_0_PUT_BLK_BYTES;
    uint8_t *k_blk = kd + slot_base + (size_t)b * KV_Q8_0_PUT_BLK_BYTES;
    uint8_t *v_blk = vd + slot_base + (size_t)b * KV_Q8_0_PUT_BLK_BYTES;

    int base = b * KV_Q8_0_PUT_BLK;
    int n = min(KV_Q8_0_PUT_BLK, head_dim - base);
    size_t in_off = (size_t)row * in_row_stride + (size_t)head * head_dim + base;
    kv_quant_q8_0_block(k_in + in_off, n, k_blk);
    kv_quant_q8_0_block(v_in + in_off, n, v_blk);
}

extern "C" void cuda_kv_put_batch_q8(const float *k_in, const float *v_in,
                                     uint8_t *kd, uint8_t *vd,
                                     int n_kv_heads, int head_dim,
                                     size_t layer_base_bytes, size_t kvh_stride_bytes,
                                     int pos_start, int n_seq, int n_blocks,
                                     int in_row_stride, cudaStream_t stream) {
    int total = n_seq * n_kv_heads * n_blocks;
    int block = 256;
    dim3 grid((total + block - 1) / block);
    kv_put_q8_batch_kernel<<<grid, block, 0, stream>>>(
        k_in, v_in, kd, vd, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride_bytes, pos_start, n_seq, n_blocks, in_row_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: kv_put_batch_q8_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

__global__ void kv_put_q8_g_kernel(const float *k_in, const float *v_in,
                                   uint8_t *kd, uint8_t *vd,
                                   int n_kv_heads, int head_dim,
                                   size_t layer_base_bytes, size_t kvh_stride_bytes,
                                   int n_blocks, const cuda_decode_params *dp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_kv_heads * n_blocks;
    if (idx >= total) return;

    int head = idx / n_blocks;
    int b = idx % n_blocks;

    size_t slot_base = layer_base_bytes + (size_t)head * kvh_stride_bytes +
                       (size_t)dp->pos * n_blocks * KV_Q8_0_PUT_BLK_BYTES;
    uint8_t *k_blk = kd + slot_base + (size_t)b * KV_Q8_0_PUT_BLK_BYTES;
    uint8_t *v_blk = vd + slot_base + (size_t)b * KV_Q8_0_PUT_BLK_BYTES;

    int base = b * KV_Q8_0_PUT_BLK;
    int n = min(KV_Q8_0_PUT_BLK, head_dim - base);
    kv_quant_q8_0_block(k_in + (size_t)head * head_dim + base, n, k_blk);
    kv_quant_q8_0_block(v_in + (size_t)head * head_dim + base, n, v_blk);
}

extern "C" void cuda_kv_put_q8_g(const float *k_in, const float *v_in,
                                 uint8_t *kd, uint8_t *vd,
                                 int n_kv_heads, int head_dim,
                                 size_t layer_base_bytes, size_t kvh_stride_bytes,
                                 int n_blocks, const int *params_dev, cudaStream_t stream) {
    int total = n_kv_heads * n_blocks;
    int block = 256;
    dim3 grid((total + block - 1) / block);
    kv_put_q8_g_kernel<<<grid, block, 0, stream>>>(
        k_in, v_in, kd, vd, n_kv_heads, head_dim,
        layer_base_bytes, kvh_stride_bytes, n_blocks,
        (const cuda_decode_params *)params_dev);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ERROR("cuda: kv_put_q8_g_kernel launch failed: %s", cudaGetErrorString(err));
    }
}

__global__ void embd_lookup_q4_0_g_kernel(const cuda_q4_0_block *embd, float *out,
                                          int dim, const cuda_decode_params *dp) {
    int token = dp->token;
    int nb = (dim + 31) / 32;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;

    const cuda_q4_0_block *blk = embd + (size_t)token * nb + b;
    float scale = __half2float(__ushort_as_half(blk->d));
    int base = b * 32;
    int n = min(32, dim - base);
    for (int j = 0; j < n; j++) {
        int v;
        if (j < 16)
            v = (blk->qs[j] & 0xF) - 8;
        else
            v = (blk->qs[j - 16] >> 4) - 8;
        out[base + j] = scale * (float)v;
    }
}

extern "C" void cuda_embd_lookup_q4_0_g(const cuda_q4_0_block *embd_dev, float *out_dev,
                                        int dim, const int *params_dev, cudaStream_t stream) {

    int nb = (dim + 31) / 32;
    int block = 256;
    int grid = (nb + block - 1) / block;
    embd_lookup_q4_0_g_kernel<<<grid, block, 0, stream>>>(embd_dev, out_dev, dim,
                                                          (const cuda_decode_params *)params_dev);
}

__global__ void embd_lookup_q8_0_g_kernel(const cuda_q8_0_block *embd, float *out,
                                          int dim, const cuda_decode_params *dp) {
    int token = dp->token;
    int nb = (dim + 31) / 32;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;

    const cuda_q8_0_block *blk = embd + (size_t)token * nb + b;
    float scale = __half2float(__ushort_as_half(blk->d));
    int base = b * 32;
    int n = min(32, dim - base);
    for (int j = 0; j < n; j++)
        out[base + j] = scale * (float)blk->qs[j];
}

extern "C" void cuda_embd_lookup_q8_0_g(const cuda_q8_0_block *embd_dev, float *out_dev,
                                        int dim, const int *params_dev, cudaStream_t stream) {

    int nb = (dim + 31) / 32;
    int block = 256;
    int grid = (nb + block - 1) / block;
    embd_lookup_q8_0_g_kernel<<<grid, block, 0, stream>>>(embd_dev, out_dev, dim,
                                                          (const cuda_decode_params *)params_dev);
}

extern "C" void cuda_matmul_iq4_nl(const void *w_dev, const float *x_dev, float *y_dev, int n,
                                   int k, cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 31))
        return;
    cuda_q8_0_block *xq_dev = xq_scratch_for(k);
    if (!xq_dev)
        return;
    q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);
    if (n == 1) {

        matmul_iq4_nl_mmvq_kernel<<<1, MMVQ_NWARPS * 32, 0, stream>>>(
            (const cuda_iq4_nl_block *)w_dev, xq_dev, y_dev, n, k);
    } else {
        dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
        matmul_iq4_nl_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
            (const cuda_iq4_nl_block *)w_dev, xq_dev, y_dev, n, k);
    }
}

extern "C" void cuda_matmul_iq4_nl_residual(const void *w_dev, const float *x_dev,
                                            const float *residual_dev, float *y_dev, int n, int k,
                                            cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (k & 31))
        return;
    cuda_q8_0_block *xq_dev = xq_scratch_for(k);
    if (!xq_dev)
        return;
    q8_0_quant_kernel<<<(k / 32 + 255) / 256, 256, 0, stream>>>(x_dev, xq_dev, k);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_iq4_nl_residual_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_iq4_nl_block *)w_dev, xq_dev, residual_dev, y_dev, n, k);
}

extern "C" void cuda_matmul_ffn_down_batch_launch(const void *w_dev, const float *gate_dev,
                                           const float *up_dev, float *y_dev, int n, int k,
                                           int m, cudaStream_t stream) {

    cuda_q8_0_block *act_q = xq_scratch_for_batch(k, m);
    if (!act_q) return;
    const int nb = k / 32;
    dim3 grid((k / 32 + 255) / 256, m);
    act_gelu_quant_batch_kernel<<<grid, 256, 0, stream>>>(
        gate_dev, up_dev, act_q, k, m);

    matmul_q8_0_dp4a_batch_kernel<<<(n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK,
                                     MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q8_0_block *)w_dev, act_q, y_dev, n, k, m);
}

extern "C" void cuda_matmul_ffn_down_q4_batch_launch(const void *w_dev, const float *gate_dev,
                                              const float *up_dev, float *y_dev, int n, int k,
                                              int m, cudaStream_t stream) {

    cuda_q8_0_block *act_q = xq_scratch_for_batch(k, m);
    if (!act_q) return;
    const int nb = k / 32;
    dim3 grid((k / 32 + 255) / 256, m);
    act_gelu_quant_batch_kernel<<<grid, 256, 0, stream>>>(
        (const float *)gate_dev, (const float *)up_dev, act_q, k, m);

    matmul_q4_0_dp4a_batch_kernel<<<(n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK,
                                     MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q4_0_block *)w_dev, (const cuda_q8_0_block *)act_q,
        y_dev, n, k, m);
}

extern "C" void cuda_matmul_ffn_down_q4_1_batch_launch(const void *w_dev, const float *gate_dev,
                                                const float *up_dev, float *y_dev, int n, int k,
                                                int m, cudaStream_t stream) {

    cuda_q8_0_block *act_q = xq_scratch_for_batch(k, m);
    if (!act_q) return;
    const int nb = k / 32;
    dim3 grid((k / 32 + 255) / 256, m);
    act_gelu_quant_batch_kernel<<<grid, 256, 0, stream>>>(
        (const float *)gate_dev, (const float *)up_dev, act_q, k, m);

    matmul_q4_1_dp4a_batch_kernel<<<(n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK,
                                    MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q4_1_block *)w_dev, act_q, y_dev, n, k, m);
}

__global__ void moe_activate_fused_kernel(const float *gate, const float *up, float *out,
                                           long long n, float gate_scale, float up_scale,
                                           int use_gelu) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i] * gate_scale;
    float u = up[i] * up_scale;
    float a = use_gelu ? gelu_tanh_f32(g) : silu_f32(g);
    out[i] = a * u;
}

__global__ void moe_scale_accum_kernel(float *out, const float *src, int n_rows, int dim,
                                        float weight, float down_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * dim;
    if (idx >= total) return;
    float w = weight * down_scale;
    out[idx] = fmaf(w, src[idx], out[idx]);
}

__global__ void moe_scale_accum_batched_kernel(float *out, const float *src,
                                                int n_rows, int dim,
                                                const float *weights, const float *down_scales,
                                                int n_experts, int rows_per_expert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * dim;
    if (idx >= total) return;
    int row = idx / dim;
    int expert_idx = row / rows_per_expert;
    if (expert_idx >= n_experts) return;
    float w = weights[expert_idx] * down_scales[expert_idx];
    out[idx] = fmaf(w, src[idx], out[idx]);
}

__global__ void moe_gate_proj_kernel(const cuda_q8_0_block *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ gate,
                                      int n_rows, int dim, int inter) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || col >= dim) return;

}

extern "C" status_code cuda_moe_experts_batch_launch(backend *self, const buffer *xb, buffer *out,
                                        int n_rows, int dim, int inter, int use_gelu,
                                        int n_experts, const moe_resident_expert *experts,
                                        const int *counts, const int *rows_packed,
                                        const float *weights_packed, cudaStream_t stream) {

    (void)self; (void)xb; (void)out; (void)n_rows; (void)dim; (void)inter;
    (void)use_gelu; (void)n_experts; (void)experts; (void)counts;
    (void)rows_packed; (void)weights_packed; (void)stream;

    return ERR_UNSUPPORTED;
}

extern "C" void cuda_attention_mla(const float *q_dev, const uint16_t *kc_dev,
                                    const uint16_t *vc_dev, float *out_dev,
                                    int n_heads, int n_kv_heads, int head_dim,
                                    size_t layer_base_bytes, size_t kvh_stride,
                                    int pos_start, int n_pos, float scale,
                                    int sliding_window, cudaStream_t stream) {
    (void)q_dev; (void)kc_dev; (void)vc_dev; (void)out_dev;
    (void)n_heads; (void)n_kv_heads; (void)head_dim;
    (void)layer_base_bytes; (void)kvh_stride; (void)pos_start;
    (void)n_pos; (void)scale; (void)sliding_window; (void)stream;

}

extern "C" void cuda_kv_alloc_mla(int n_layers, int n_ctx, int kv_lora, int qk_rope,
                                   int qk_nope, int v_head, int n_ctx_kv,
                                   int kv_quant, int qk_lora_rank,
                                   cudaStream_t stream, buffer *k_out, buffer *v_out) {
    (void)n_layers; (void)n_ctx; (void)kv_lora; (void)qk_rope;
    (void)qk_nope; (void)v_head; (void)n_ctx_kv; (void)kv_quant;
    (void)qk_lora_rank; (void)stream; (void)k_out; (void)v_out;

}

extern "C" void cuda_kv_put_mla(const float *k_in, const float *v_in,
                                 uint8_t *k_dev, uint8_t *v_dev,
                                 int n_kv_heads, int head_dim,
                                 size_t layer_base_bytes, size_t kvh_stride_bytes,
                                 int pos, int n_blocks, cudaStream_t stream) {
    (void)k_in; (void)v_in; (void)k_dev; (void)v_dev;
    (void)n_kv_heads; (void)head_dim; (void)layer_base_bytes;
    (void)kvh_stride_bytes; (void)pos; (void)n_blocks; (void)stream;

}

extern "C" void cuda_matmul_batch_iq4_nl(const void *w_dev, const float *x_dev, float *y_dev,
                                         int n, int k, int m, cudaStream_t stream) {
    if (n <= 0 || m <= 0 || k <= 0 || (k & 31))
        return;
    const int nb = k / 32;
    cuda_q8_0_block *xq = xq_scratch_for_batch(k, m);
    if (!xq)
        return;
    q8_0_quant_batch_kernel<<<(((size_t)m * nb) + 255) / 256, 256, 0, stream>>>(x_dev, xq, k, m);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_iq4_nl_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_iq4_nl_block *)w_dev, xq, y_dev, n, k, m);
}

extern "C" void cuda_matmul_batch_q6_k(const void *w_dev, const float *x_dev, float *y_dev,
                                       int n, int k, int m, cudaStream_t stream) {
    if (n <= 0 || m <= 0 || k <= 0 || (k & 255))
        return;
    const int nb = k / 256;
    cuda_q8_k_block *xq = q8k_scratch_for_batch(k, m);
    if (!xq)
        return;
    q8_k_quant_batch_kernel<<<(((size_t)m * nb) + 255) / 256, 256, 0, stream>>>(x_dev, xq, k, m);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q6_k_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q6_k_block *)w_dev, xq, y_dev, n, k, m);
}

extern "C" void cuda_matmul_batch_q5_0(const void *w_dev, const float *x_dev, float *y_dev,
                                        int n, int k, int m, cudaStream_t stream) {
    if (n <= 0 || m <= 0 || k <= 0 || (k & 31)) return;
    const int nb = k / 32;
    cuda_q8_0_block *xq = xq_scratch_for_batch(k, m);
    if (!xq) return;
    q8_0_quant_batch_kernel<<<(((size_t)m * nb) + 255) / 256, 256, 0, stream>>>(x_dev, xq, k, m);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q5_0_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q5_0_block *)w_dev, xq, y_dev, n, k, m);
}

extern "C" void cuda_matmul_batch_q5_1(const void *w_dev, const float *x_dev, float *y_dev,
                                        int n, int k, int m, cudaStream_t stream) {
    if (n <= 0 || m <= 0 || k <= 0 || (k & 31)) return;
    const int nb = k / 32;
    cuda_q8_1_block *xq = q81_scratch_for((size_t)m * (size_t)k);
    if (!xq) return;

    const int total = (int)(((size_t)m * nb + 255) / 256) * 256;
    dim3 g(total / 256);
    q8_1_quant_rows_kernel<<<g, 256, 0, stream>>>(x_dev, xq, k, m);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q5_1_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q5_1_block *)w_dev, xq, y_dev, n, k, m);
}

extern "C" void cuda_matmul_batch_q5_k(const void *w_dev, const float *x_dev, float *y_dev,
                                        int n, int k, int m, cudaStream_t stream) {
    if (n <= 0 || m <= 0 || k <= 0 || (k & 255)) return;
    const int nb = k / 256;
    cuda_q8_k_block *xq = q8k_scratch_for_batch(k, m);
    if (!xq) return;
    q8_k_quant_batch_kernel<<<(((size_t)m * nb) + 255) / 256, 256, 0, stream>>>(x_dev, xq, k, m);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_q5_k_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_q5_k_block *)w_dev, xq, y_dev, n, k, m);
}

extern "C" void cuda_matmul_batch_iq3_s(const void *w_dev, const float *x_dev, float *y_dev,
                                         int n, int k, int m, cudaStream_t stream) {
    if (n <= 0 || m <= 0 || k <= 0 || (k & 255)) return;
    const int nb = k / 256;
    cuda_q8_k_block *xq = q8k_scratch_for_batch(k, m);
    if (!xq) return;
    q8_k_quant_batch_kernel<<<(((size_t)m * nb) + 255) / 256, 256, 0, stream>>>(x_dev, xq, k, m);
    dim3 grid((n + MM_ROWS_PER_BLOCK - 1) / MM_ROWS_PER_BLOCK);
    matmul_iq3_s_batch_kernel<<<grid, MM_LANES * MM_ROWS_PER_BLOCK, 0, stream>>>(
        (const cuda_iq3_s_block *)w_dev, xq, y_dev, n, k, m);
}
