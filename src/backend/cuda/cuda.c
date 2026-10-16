#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <cuda_runtime.h>

#include "backend/backend.h"
#include "compute.h"
#include "recipe.h"
#include "log.h"
#include "gguf.h"
#include "backend/cuda/cuda_internal.h"
#include "backend/cuda/cuda_repack.h"

static int cuda_lazy_d2h(void) {
    return 1;
}
static int cuda_lazy_d2h_for(const char *tag) {
    (void)tag;
    return 1;
}

static status_code cuda_op_add_inplace(backend *self, buffer *x, const buffer *y, int n);
static void *cuda_dev_ptr(const buffer *b);
static status_code cuda_op_matmul_batch(backend *self, const buffer *w, uint32_t w_type,
                                        const buffer *x, buffer *y, int n, int k, int m);
static status_code cuda_op_add_batch(backend *self, buffer *x, const buffer *y, int n, int m);
static status_code cuda_rope_ensure_cs(backend *self, int pos, int head_dim,
                                       const float *freq_factors);
static status_code cuda_rope_pre_ensure(backend *self, int pos_end, int head_dim,
                                        const float *rope_cos_base,
                                        const float *rope_sin_base,
                                        const float **cos_dev, const float **sin_dev);

struct cuda_priv {
    cudaStream_t stream;
    size_t *kv_layer_off;
    int kv_n_layers;
    int kv_n_kv_heads;
    int kv_n_ctx;
    int kv_head_dim;

    int kv_quant_q8;

#define ROPE_TBL_SLOTS 2
    struct {
        float *cos_dev;
        float *sin_dev;
        size_t cap;
        int half;
        int up_pos;

        const float *cos_src;
        const float *sin_src;
    } rope_tbl[ROPE_TBL_SLOTS];

    int32_t *argmax_idx_dev;

    float *rope_cs_host;
    size_t rope_cs_cap;
    int rope_cs_pos;
    float rope_cs_theta;
    const float *rope_cs_ff;
    int rope_cs_hd;

    float *rope_cs_cos_dev;
    float *rope_cs_sin_dev;
    size_t rope_cs_dev_cap;

    float *resid_tmp_dev;
    size_t resid_tmp_cap;

    float *rmm_tmp_dev;
    size_t rmm_tmp_cap;
};

static struct cuda_priv *cuda_priv(backend *self) {
    return (struct cuda_priv *)self->priv;
}

static size_t cuda_kv_layer_base(struct cuda_priv *priv, int layer) {
    if (priv->kv_layer_off)
        return priv->kv_layer_off[layer];
    return (size_t)layer * (size_t)priv->kv_n_kv_heads * (size_t)priv->kv_n_ctx *
           (size_t)priv->kv_head_dim;
}

static size_t cuda_kv_q8_nblocks(int head_dim) {
    return ((size_t)head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
}

static size_t cuda_kv_layer_base_bytes(struct cuda_priv *priv, int layer) {
    if (priv->kv_layer_off)
        return priv->kv_layer_off[layer];
    return (size_t)layer * (size_t)priv->kv_n_kv_heads * (size_t)priv->kv_n_ctx *
           cuda_kv_q8_nblocks(priv->kv_head_dim) * KV_Q8_0_BLOCK_BYTES;
}

static status_code cuda_probe(void) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess)
        return ERR_UNSUPPORTED;
    return device_count > 0 ? OK : ERR_UNSUPPORTED;
}

static status_code cuda_init(backend *self, int device_index) {
    if (device_index < 0)
        device_index = 0;
    cudaError_t e = cudaSetDevice(device_index);
    if (e != cudaSuccess)
        return ERR_UNSUPPORTED;

    struct cuda_priv *priv = calloc(1, sizeof(*priv));
    if (!priv)
        return ERR_OUT_OF_MEMORY;
    if (cudaStreamCreate(&priv->stream) != cudaSuccess) {
        free(priv);
        return ERR_UNSUPPORTED;
    }
    if (cudaMalloc((void **)&priv->argmax_idx_dev, sizeof(int32_t)) != cudaSuccess) {
        cudaStreamDestroy(priv->stream);
        free(priv);
        return ERR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < ROPE_TBL_SLOTS; i++) {
        priv->rope_tbl[i].half = -1;
        priv->rope_tbl[i].up_pos = -1;
    }
    self->priv = priv;
    return OK;
}

static void cuda_free(backend *self) {
    struct cuda_priv *priv = cuda_priv(self);
    if (priv) {
        if (priv->argmax_idx_dev)
            cudaFree(priv->argmax_idx_dev);
        if (priv->rope_cs_host)
            free(priv->rope_cs_host);
        if (priv->rope_cs_cos_dev)
            cudaFree(priv->rope_cs_cos_dev);
        if (priv->rope_cs_sin_dev)
            cudaFree(priv->rope_cs_sin_dev);
        if (priv->resid_tmp_dev)
            cudaFree(priv->resid_tmp_dev);
        if (priv->rmm_tmp_dev)
            cudaFree(priv->rmm_tmp_dev);
        for (int i = 0; i < ROPE_TBL_SLOTS; i++) {
            if (priv->rope_tbl[i].cos_dev)
                cudaFree(priv->rope_tbl[i].cos_dev);
            if (priv->rope_tbl[i].sin_dev)
                cudaFree(priv->rope_tbl[i].sin_dev);
        }
        if (priv->stream)
            cudaStreamDestroy(priv->stream);
        free(priv);
    }
}

static void cuda_synchronize(backend *self) {
    struct cuda_priv *priv = cuda_priv(self);
    if (priv && priv->stream)
        cudaStreamSynchronize(priv->stream);
    cudaDeviceSynchronize();
}

static status_code cuda_upload_weight(backend *self, const void *host_data, uint32_t type, int k,
                                      int n_rows, buffer *out, uint32_t declared_type);

static status_code cuda_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
    int k = (int)desc->dims[0];
    int n_rows = (int)desc->dims[1];
    if (n_rows <= 0)
        n_rows = 1;

    void       *converted = NULL;
    const void *host_data = desc->host_data;
    uint32_t    type      = desc->type;

    if (desc->n_dims == 2 && desc->type == GGML_TYPE_Q4_0 && k > 0 && (k % 32) == 0) {
        converted = cuda_convert_q4_0_to_q8_0(desc->host_data, n_rows, k);
        if (converted) {
            host_data = converted;
            type      = GGML_TYPE_Q8_0;
        }
    }

    status_code st = cuda_upload_weight(self, host_data, type, k, n_rows, out,
                                      desc->type);
    free(converted);
    return st;
}

static status_code cuda_upload_weight(backend *self, const void *host_data, uint32_t type, int k,
                                      int n_rows, buffer *out, uint32_t declared_type) {
    size_t row_bytes;
    switch (type) {
        case GGML_TYPE_Q8_0: row_bytes = (size_t)(k / 32) * 34; break;
        case GGML_TYPE_Q4_0: row_bytes = (size_t)(k / 32) * 18; break;
        case GGML_TYPE_Q4_1: row_bytes = (size_t)(k / 32) * 20; break;
        case GGML_TYPE_F32:  row_bytes = (size_t)k * 4;         break;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: row_bytes = (size_t)k * 2;         break;
        default:             row_bytes = ggml_row_size(type, k); break;
    }

    size_t w_bytes = (size_t)n_rows * row_bytes;

    void *w_dev = NULL;
    cudaError_t e = cudaMalloc(&w_dev, w_bytes);
    if (e == cudaSuccess) {
        e = cudaMemcpy(w_dev, host_data, w_bytes, cudaMemcpyHostToDevice);
        if (e == cudaSuccess) {
            void *w_host = malloc(w_bytes);
            if (w_host) {
                memcpy(w_host, host_data, w_bytes);
                out->handle  = w_dev;
                out->size    = w_bytes;
                out->offset  = 0;
                out->host_ptr = w_host;
                out->owner   = self;
                cuda_host_alloc_note(w_host, 0);

                if (type != declared_type)
                    cuda_weight_type_note(w_dev, type);
                return OK;
            }
        }
        cudaFree(w_dev);
    }

    void *w_managed = NULL;
    e = cudaMallocManaged(&w_managed, w_bytes, cudaMemAttachGlobal);
    if (e != cudaSuccess) return ERR_OUT_OF_MEMORY;
    e = cudaMemcpy(w_managed, host_data, w_bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { cudaFree(w_managed); return ERR_OUT_OF_MEMORY; }
    out->handle  = w_managed;
    out->size    = w_bytes;
    out->offset  = 0;
    out->host_ptr = NULL;
    out->owner   = self;
    if (type != declared_type)
        cuda_weight_type_note(w_managed, type);
    return OK;
}

static status_code cuda_buffer_alloc_scratch(backend *self, size_t size, buffer *out) {
    void *dev = NULL;
    cudaError_t e = cudaMalloc(&dev, size);
    if (e == cudaSuccess) {
        void *host = NULL;
        e = cudaHostAlloc(&host, size, cudaHostAllocDefault);
        if (e == cudaSuccess && host) {
            out->handle        = dev;
            out->size          = size;
            out->offset        = 0;
            out->host_ptr      = host;
            out->owner         = self;
            cuda_host_alloc_note(host, 1);
            return OK;
        }
        if (host) cudaFreeHost(host);

        host = malloc(size);
        if (host) {
            out->handle        = dev;
            out->size          = size;
            out->offset        = 0;
            out->host_ptr      = host;
            out->owner         = self;
            cuda_host_alloc_note(host, 0);
            return OK;
        }
        cudaFree(dev);
    }
    e = cudaMallocManaged(&dev, size, cudaMemAttachGlobal);
    if (e != cudaSuccess) return ERR_OUT_OF_MEMORY;
    out->handle        = dev;
    out->size          = size;
    out->offset        = 0;
    out->host_ptr      = NULL;
    out->owner         = self;
    return OK;
}

static void cuda_buffer_free(backend *self, buffer *buf) {
    (void)self;
    if (!buf || !buf->handle)
        return;

    void *handle = buf->handle;
    void *host_ptr = (void *)buf->host_ptr;

    cuda_weight_type_forget(handle);

    int   pinned = cuda_host_alloc_pinned(host_ptr);

    buf->handle = NULL;
    buf->host_ptr = NULL;
    buf->size   = 0;
    buf->offset = 0;
    if (host_ptr) {
        cudaFree(handle);
        if (pinned == 1) {
            cudaError_t e = cudaFreeHost(host_ptr);
            if (e != cudaSuccess) {

            }
        } else {

            free(host_ptr);
        }
        cuda_host_alloc_forget(host_ptr);
    } else {
        cudaFree(handle);
    }
}

static void *cuda_dev_ptr(const buffer *b) {
    return (char *)b->handle + b->offset;
}

static void *cuda_host_stage_in(const buffer *b, size_t bytes) {
    void *h = malloc(bytes ? bytes : 1);
    if (!h) return NULL;
    if (cudaMemcpy(h, cuda_dev_ptr((buffer *)b), bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(h);
        return NULL;
    }
    return h;
}

static status_code cuda_host_stage_out(buffer *b, const void *h, size_t bytes) {
    if (cudaMemcpy(cuda_dev_ptr(b), h, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return ERR_OUT_OF_MEMORY;
    if (b->host_ptr) memcpy((void *)b->host_ptr, h, bytes);
    return OK;
}

static buffer cuda_host_buf(void *p, size_t bytes, backend *owner) {
    buffer b = {0};
    b.handle   = p;
    b.host_ptr = p;
    b.size     = bytes;
    b.owner    = owner;
    return b;
}

static status_code cuda_copy_buffer(backend *self, const buffer *src, buffer *dst, int n) {
    (void)self;
    if (cudaMemcpy(cuda_dev_ptr(dst), cuda_dev_ptr(src), (size_t)n * sizeof(float),
                   cudaMemcpyDefault) != cudaSuccess)
        return ERR_OUT_OF_MEMORY;
    if (dst->host_ptr) {
        if (src->host_ptr) memcpy((void *)dst->host_ptr, src->host_ptr, (size_t)n * sizeof(float));
        else if (cudaMemcpy((void *)dst->host_ptr, cuda_dev_ptr(dst), (size_t)n * sizeof(float),
                            cudaMemcpyDeviceToHost) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
    }
    return OK;
}

static status_code cuda_copy_buffer_async(backend *self, const buffer *src, buffer *dst, int n) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || n <= 0) return ERR_INVALID_ARG;
    if (cudaMemcpyAsync(cuda_dev_ptr(dst), cuda_dev_ptr((buffer *)src),
                        (size_t)n * sizeof(float), cudaMemcpyDefault,
                        priv->stream) != cudaSuccess)
        return ERR_OUT_OF_MEMORY;
    if (dst->host_ptr) {
        if (src->host_ptr)
            memcpy((void *)dst->host_ptr, src->host_ptr, (size_t)n * sizeof(float));
        else if (cudaMemcpyAsync((void *)dst->host_ptr, cuda_dev_ptr(dst),
                                 (size_t)n * sizeof(float), cudaMemcpyDeviceToHost,
                                 priv->stream) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
    }
    return OK;
}

static status_code cuda_buffer_write_async(backend *self, buffer *buf, const float *host_src,
                                           int n) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || !host_src || n <= 0) return ERR_INVALID_ARG;
    if (buf->host_ptr)
        memcpy((void *)buf->host_ptr, host_src, (size_t)n * sizeof(float));
    if (cudaMemcpyAsync(cuda_dev_ptr(buf), buf->host_ptr ? buf->host_ptr : host_src,
                        (size_t)n * sizeof(float), cudaMemcpyHostToDevice,
                        priv->stream) != cudaSuccess)
        return ERR_OUT_OF_MEMORY;
    return OK;
}

static status_code cuda_op_copy_2d(backend *self, buffer *dst, size_t dst_row_floats,
                                   const buffer *src, size_t src_row_floats, int width_floats,
                                   int height) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || !dst || !src || width_floats <= 0 || height <= 0)
        return ERR_INVALID_ARG;
    void *dd = cuda_dev_ptr(dst);
    const void *sd = cuda_dev_ptr((buffer *)src);
    if (dst->owner == self && src->owner == self && dd && sd) {
        if (cudaMemcpy2DAsync(dd, dst_row_floats * sizeof(float), sd,
                              src_row_floats * sizeof(float), (size_t)width_floats * sizeof(float),
                              (size_t)height, cudaMemcpyDeviceToDevice,
                              priv->stream) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
        return OK;
    }
    for (int r = 0; r < height; r++) {
        buffer srow = *src;
        buffer drow = *dst;
        srow.offset += (size_t)r * src_row_floats * sizeof(float);
        drow.offset += (size_t)r * dst_row_floats * sizeof(float);
        status_code st = cuda_copy_buffer(self, &srow, &drow, width_floats);
        if (st != OK) return st;
    }
    return OK;
}

static status_code cuda_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n) {

    struct cuda_priv *priv = cuda_priv(self);
    if (priv && priv->stream)
        cudaStreamSynchronize(priv->stream);
    if (buf->handle) {
        if (buf->host_ptr) {
            if (cudaMemcpy((void *)buf->host_ptr, cuda_dev_ptr(buf),
                           (size_t)n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                return ERR_OUT_OF_MEMORY;
            memcpy(host_dst, buf->host_ptr, (size_t)n * sizeof(float));
            return OK;
        }
        if (cudaMemcpy(host_dst, cuda_dev_ptr(buf), (size_t)n * sizeof(float),
                       cudaMemcpyDefault) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
        return OK;
    }
    if (buf->host_ptr) {
        memcpy(host_dst, buf->host_ptr, (size_t)n * sizeof(float));
        return OK;
    }
    return ERR_INVALID_ARG;
}

static status_code cuda_buffer_write_f32(backend *self, buffer *buf, const float *host_src, int n) {
    (void)self;
    if (buf->host_ptr) {
        memcpy((void *)buf->host_ptr, host_src, (size_t)n * sizeof(float));
        if (cudaMemcpy(cuda_dev_ptr(buf), buf->host_ptr, (size_t)n * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
        return OK;
    }
    if (cudaMemcpy(cuda_dev_ptr(buf), host_src, (size_t)n * sizeof(float), cudaMemcpyDefault) !=
        cudaSuccess)
        return ERR_OUT_OF_MEMORY;
    return OK;
}

static int cuda_matmul_type_native(backend *self, uint32_t w_type) {
    (void)self;
    switch (w_type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return 1;
        default:
            return 0;
    }
}

static status_code cuda_op_add_inplace(backend *self, buffer *x, const buffer *y, int n);
static status_code cuda_matmul(backend *self, const buffer *w, uint32_t w_type,
                                const buffer *x, buffer *y, int n, int k) {

    uint32_t stored = cuda_weight_type_of(w ? cuda_dev_ptr(w) : NULL);
    if (stored != (uint32_t)-1)
        w_type = stored;
    if (!cuda_matmul_type_native(self, w_type)) {
        backend *host = backend_host();
        if (host && host->matmul) {
            size_t wb = w->size, xb = (size_t)k * sizeof(float), yb = (size_t)n * sizeof(float);
            void  *wh = cuda_host_stage_in(w, wb);
            void  *xh = cuda_host_stage_in(x, xb);
            void  *yh = malloc(yb ? yb : 1);
            status_code st = ERR_OUT_OF_MEMORY;
            if (wh && xh && yh) {
                buffer whb = cuda_host_buf(wh, wb, host);
                buffer xhb = cuda_host_buf(xh, xb, host);
                buffer yhb = cuda_host_buf(yh, yb, host);
                st = host->matmul(host, &whb, w_type, &xhb, &yhb, n, k);
                if (st == OK)
                    st = cuda_host_stage_out(y, yh, yb);
            }
            free(wh); free(xh); free(yh);
            return st;
        }
        return ERR_UNSUPPORTED;
    }

    struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;

    const void *w_dev = cuda_dev_ptr((buffer *)w);
    const float *x_dev = (const float *)cuda_dev_ptr((buffer *)x);
    float *y_dev = (float *)cuda_dev_ptr(y);

    switch (w_type) {
        case GGML_TYPE_Q8_0:
            cuda_matmul_q8_0(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q4_0:
            cuda_matmul_q4_0(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q4_1:
            cuda_matmul_q4_1(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q4_K:
            cuda_matmul_q4_k(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q6_K:
            cuda_matmul_q6_k(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q5_K:
            cuda_matmul_q5_k(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q5_0:
            cuda_matmul_q5_0(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q5_1:
            cuda_matmul_q5_1(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_IQ3_S:
            cuda_matmul_iq3_s(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            cuda_matmul_iq4_nl(w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_F32:
            cuda_matmul_f32((const float *)w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_F16:
            cuda_matmul_f16((const uint16_t *)w_dev, x_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_BF16:
            cuda_matmul_bf16((const uint16_t *)w_dev, x_dev, y_dev, n, k, stream);
            break;
        default:
            return ERR_UNSUPPORTED;
    }

        if (!cuda_lazy_d2h_for("mm") && y->host_ptr) {
        if (cudaMemcpy((void *)y->host_ptr, y_dev, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return ERR_INTERNAL;
    }

    return OK;
}

static status_code cuda_matmul_residual(backend *self, const buffer *w, uint32_t w_type,
                                        const buffer *x, const buffer *residual, buffer *y,
                                        int n, int k) {

    uint32_t stored = cuda_weight_type_of(w ? cuda_dev_ptr(w) : NULL);
    if (stored != (uint32_t)-1)
        w_type = stored;
    if (!cuda_matmul_type_native(self, w_type)) {
        backend *host = backend_host();
        if (host && host->matmul_residual) {
            size_t wb = w->size, xb = (size_t)k * sizeof(float), yb = (size_t)n * sizeof(float);
            void  *wh = cuda_host_stage_in(w, wb);
            void  *xh = cuda_host_stage_in(x, xb);
            void  *rh = cuda_host_stage_in(residual, yb);
            void  *yh = malloc(yb ? yb : 1);
            status_code st = ERR_OUT_OF_MEMORY;
            if (wh && xh && rh && yh) {
                buffer whb = cuda_host_buf(wh, wb, host);
                buffer xhb = cuda_host_buf(xh, xb, host);
                buffer rhb = cuda_host_buf(rh, yb, host);
                buffer yhb = cuda_host_buf(yh, yb, host);
                st = host->matmul_residual(host, &whb, w_type, &xhb, &rhb, &yhb, n, k);
                if (st == OK)
                    st = cuda_host_stage_out(y, yh, yb);
            }
            free(wh); free(xh); free(rh); free(yh);
            return st;
        }
        return ERR_UNSUPPORTED;
    }

    struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;

    const void *w_dev = cuda_dev_ptr((buffer *)w);
    const float *x_dev = (const float *)cuda_dev_ptr((buffer *)x);
    const float *r_dev = (const float *)cuda_dev_ptr((buffer *)residual);
    float *y_dev = (float *)cuda_dev_ptr(y);

    switch (w_type) {
        case GGML_TYPE_Q8_0:
            cuda_matmul_q8_0_residual(w_dev, x_dev, r_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q4_0:
            cuda_matmul_q4_0_residual(w_dev, x_dev, r_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q4_1:
            cuda_matmul_q4_1_residual(w_dev, x_dev, r_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            cuda_matmul_iq4_nl_residual(w_dev, x_dev, r_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q6_K:
            cuda_matmul_q6_k_residual(w_dev, x_dev, r_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q5_K:
            cuda_matmul_q5_k_residual(w_dev, x_dev, r_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q5_0:
            cuda_matmul_q5_0_residual(w_dev, x_dev, r_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_Q5_1:
            cuda_matmul_q5_1_residual(w_dev, x_dev, r_dev, y_dev, n, k, stream);
            break;
        case GGML_TYPE_IQ3_S:
            cuda_matmul_iq3_s_residual(w_dev, x_dev, r_dev, y_dev, n, k, stream);
            break;
        default: {

            status_code st = cuda_matmul(self, w, w_type, x, y, n, k);
            if (st != OK) return st;
            return cuda_op_add_inplace(self, y, residual, n);
        }
    }

    if (!cuda_lazy_d2h_for("mmres") && y->host_ptr) {
        if (cudaMemcpy((void *)y->host_ptr, y_dev, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return ERR_INTERNAL;
    }

    return OK;
}

static status_code cuda_op_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type,
                                        const buffer *gate, const buffer *up, buffer *y, int n,
                                        int k, int activation) {

    uint32_t stored = cuda_weight_type_of(w ? cuda_dev_ptr(w) : NULL);
    if (stored != (uint32_t)-1)
        w_type = stored;
    if (activation != ACTIVATION_GELU) {
        backend *host = backend_host();
        if (host && host->matmul_ffn_down) {
            size_t wb = w->size, gb = (size_t)k * sizeof(float),
                   yb = (size_t)n * sizeof(float);
            void  *wh = cuda_host_stage_in(w, wb);
            void  *gh = cuda_host_stage_in(gate, gb);
            void  *uh = cuda_host_stage_in(up, gb);
            void  *yh = malloc(yb ? yb : 1);
            status_code st = ERR_OUT_OF_MEMORY;
            if (wh && gh && uh && yh) {
                buffer whb = cuda_host_buf(wh, wb, host);
                buffer ghb = cuda_host_buf(gh, gb, host);
                buffer uhb = cuda_host_buf(uh, gb, host);
                buffer yhb = cuda_host_buf(yh, yb, host);
                st = host->matmul_ffn_down(host, &whb, w_type, &ghb, &uhb, &yhb, n, k,
                                           activation);
                if (st == OK)
                    st = cuda_host_stage_out(y, yh, yb);
            }
            free(wh); free(gh); free(uh); free(yh);
            return st;
        }
        return ERR_UNSUPPORTED;
    }
    if (w_type != GGML_TYPE_Q8_0 && w_type != GGML_TYPE_Q4_0 && w_type != GGML_TYPE_Q4_1) {

        backend *host = backend_host();
        if (host && host->matmul_ffn_down) {
            size_t wb = w->size, gb = (size_t)k * sizeof(float),
                   yb = (size_t)n * sizeof(float);
            void  *wh = cuda_host_stage_in(w, wb);
            void  *gh = cuda_host_stage_in(gate, gb);
            void  *uh = cuda_host_stage_in(up, gb);
            void  *yh = malloc(yb ? yb : 1);
            status_code st = ERR_OUT_OF_MEMORY;
            if (wh && gh && uh && yh) {
                buffer whb = cuda_host_buf(wh, wb, host);
                buffer ghb = cuda_host_buf(gh, gb, host);
                buffer uhb = cuda_host_buf(uh, gb, host);
                buffer yhb = cuda_host_buf(yh, yb, host);
                st = host->matmul_ffn_down(host, &whb, w_type, &ghb, &uhb, &yhb, n, k,
                                           activation);
                if (st == OK)
                    st = cuda_host_stage_out(y, yh, yb);
            }
            free(wh); free(gh); free(uh); free(yh);
            return st;
        }
        return ERR_UNSUPPORTED;
    }
        struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;

    const void *w_dev = cuda_dev_ptr((buffer *)w);
    const float *gate_dev = (const float *)cuda_dev_ptr((buffer *)gate);
    const float *up_dev = (const float *)cuda_dev_ptr((buffer *)up);
    float *y_dev = (float *)cuda_dev_ptr(y);

    if (w_type == GGML_TYPE_Q4_0 ) {
        cuda_matmul_ffn_down_q4(w_dev, gate_dev, up_dev, y_dev, n, k, stream);
    } else if (w_type == GGML_TYPE_Q4_1) {
        cuda_matmul_ffn_down_q4_1(w_dev, gate_dev, up_dev, y_dev, n, k, stream);
    } else {
        cuda_matmul_ffn_down(w_dev, gate_dev, up_dev, y_dev, n, k, stream);
    }

    if (!cuda_lazy_d2h_for("ffndown") && y->host_ptr) {
        if (cudaMemcpy((void *)y->host_ptr, y_dev, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_matmul_ffn_down_batch(backend *self, const buffer *w, uint32_t w_type,
                                                 const buffer *gate, const buffer *up, buffer *y,
                                                 int n, int k, int activation, int m) {
    uint32_t stored = cuda_weight_type_of(w ? cuda_dev_ptr(w) : NULL);
    if (stored != (uint32_t)-1)
        w_type = stored;

    if (activation != ACTIVATION_GELU) {
        backend *host = backend_host();
        if (host && host->matmul_ffn_down_batch) {
            size_t wb = w->size, gb = (size_t)m * (size_t)k * sizeof(float),
                   yb = (size_t)m * (size_t)n * sizeof(float);
            void *wh = cuda_host_stage_in(w, wb);
            void *gh = cuda_host_stage_in(gate, gb);
            void *uh = cuda_host_stage_in(up, gb);
            void *yh = malloc(yb ? yb : 1);
            status_code st = ERR_OUT_OF_MEMORY;
            if (wh && gh && uh && yh) {
                buffer whb = cuda_host_buf(wh, wb, host);
                buffer ghb = cuda_host_buf(gh, gb, host);
                buffer uhb = cuda_host_buf(uh, gb, host);
                buffer yhb = cuda_host_buf(yh, yb, host);
                st = host->matmul_ffn_down_batch(host, &whb, w_type, &ghb, &uhb, &yhb, n, k,
                                                 activation, m);
                if (st == OK)
                    st = cuda_host_stage_out(y, yh, yb);
            }
            free(wh); free(gh); free(uh); free(yh);
            return st;
        }
        return ERR_UNSUPPORTED;
    }
    if (w_type != GGML_TYPE_Q8_0 && w_type != GGML_TYPE_Q4_0 && w_type != GGML_TYPE_Q4_1) {
        backend *host = backend_host();
        if (host && host->matmul_ffn_down_batch) {
            size_t wb = w->size, gb = (size_t)m * (size_t)k * sizeof(float),
                   yb = (size_t)m * (size_t)n * sizeof(float);
            void *wh = cuda_host_stage_in(w, wb);
            void *gh = cuda_host_stage_in(gate, gb);
            void *uh = cuda_host_stage_in(up, gb);
            void *yh = malloc(yb ? yb : 1);
            status_code st = ERR_OUT_OF_MEMORY;
            if (wh && gh && uh && yh) {
                buffer whb = cuda_host_buf(wh, wb, host);
                buffer ghb = cuda_host_buf(gh, gb, host);
                buffer uhb = cuda_host_buf(uh, gb, host);
                buffer yhb = cuda_host_buf(yh, yb, host);
                st = host->matmul_ffn_down_batch(host, &whb, w_type, &ghb, &uhb, &yhb, n, k,
                                                 activation, m);
                if (st == OK)
                    st = cuda_host_stage_out(y, yh, yb);
            }
            free(wh); free(gh); free(uh); free(yh);
            return st;
        }
        return ERR_UNSUPPORTED;
    }
        struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;

    const void *w_dev = cuda_dev_ptr((buffer *)w);
    const float *gate_dev = (const float *)cuda_dev_ptr((buffer *)gate);
    const float *up_dev = (const float *)cuda_dev_ptr((buffer *)up);
    float *y_dev = (float *)cuda_dev_ptr(y);

    if (w_type == GGML_TYPE_Q4_0 ) {
        cuda_matmul_ffn_down_q4_batch_launch(w_dev, gate_dev, up_dev, y_dev, n, k, m, stream);
    } else if (w_type == GGML_TYPE_Q4_1) {
        cuda_matmul_ffn_down_q4_1_batch_launch(w_dev, gate_dev, up_dev, y_dev, n, k, m, stream);
    } else {
        cuda_matmul_ffn_down_batch_launch(w_dev, gate_dev, up_dev, y_dev, n, k, m, stream);
    }

    if (!cuda_lazy_d2h_for("ffndown") && y->host_ptr) {
        if (cudaMemcpy((void *)y->host_ptr, y_dev, (size_t)m * (size_t)n * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
                                    float eps) {
    struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;

    cuda_rmsnorm(cuda_dev_ptr((buffer *)x),
                 w ? (const float *)cuda_dev_ptr((buffer *)w) : NULL,
                 (float *)cuda_dev_ptr(y), n, eps, stream);

    if (!cuda_lazy_d2h_for("norm") && y->host_ptr)
        cudaMemcpy(y->host_ptr, cuda_dev_ptr(y), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);

    return OK;
}

static status_code cuda_op_rmsnorm_per_head(backend *self, const buffer *x, const buffer *w,
                                            buffer *y, int n_heads, int head_dim, float eps) {
    struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;
    int n = n_heads * head_dim;

    cuda_rmsnorm_per_head(cuda_dev_ptr((buffer *)x),
                          w ? (const float *)cuda_dev_ptr((buffer *)w) : NULL,
                          (float *)cuda_dev_ptr(y), n_heads, head_dim, eps, stream);

    if (!cuda_lazy_d2h_for("norm") && y->host_ptr)
        cudaMemcpy(y->host_ptr, cuda_dev_ptr(y), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    return OK;
}

static status_code cuda_op_rmsnorm_per_head_batch(backend *self, const buffer *x, const buffer *w,
                                                  buffer *y, int n_heads, int head_dim, float eps,
                                                  int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || m <= 0) return ERR_INTERNAL;
    cuda_rmsnorm_batch((const float *)cuda_dev_ptr((buffer *)x),
                       w ? (const float *)cuda_dev_ptr((buffer *)w) : NULL,
                       (float *)cuda_dev_ptr(y), head_dim, eps, m * n_heads, priv->stream);
    if (!cuda_lazy_d2h_for("norm") && y->host_ptr) {
        if (cudaMemcpyAsync((void *)y->host_ptr, cuda_dev_ptr(y),
                            (size_t)m * (size_t)n_heads * (size_t)head_dim * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_rmsnorm_noweight(backend *self, const buffer *x, buffer *y, int n,
                                            float eps) {
    return cuda_op_rmsnorm(self, x, NULL, y, n, eps);
}

static status_code cuda_op_rmsnorm_noweight_per_head_batch(backend *self, const buffer *x,
                                                           buffer *y, int n_heads, int head_dim,
                                                           float eps, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || m <= 0) return ERR_INTERNAL;
    cuda_rmsnorm_batch((const float *)cuda_dev_ptr((buffer *)x), NULL,
                       (float *)cuda_dev_ptr(y), head_dim, eps, m * n_heads, priv->stream);
    if (!cuda_lazy_d2h_for("norm") && y->host_ptr) {
        if (cudaMemcpyAsync((void *)y->host_ptr, cuda_dev_ptr(y),
                            (size_t)m * (size_t)n_heads * (size_t)head_dim * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_rmsnorm_noweight_per_head(backend *self, const buffer *x, buffer *y,
                                                     int n_heads, int head_dim, float eps) {
    return cuda_op_rmsnorm_per_head(self, x, NULL, y, n_heads, head_dim, eps);
}

static status_code cuda_op_rmsnorm_add(backend *self, const buffer *x, const buffer *w,
                                        const buffer *residual, buffer *y, int n, float eps,
                                        float out_scale) {
    struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;
    int n_total = n;

    cuda_rmsnorm_add(cuda_dev_ptr((buffer *)x),
                     w ? (const float *)cuda_dev_ptr((buffer *)w) : NULL,
                     (const float *)cuda_dev_ptr(residual),
                     (float *)cuda_dev_ptr(y), n_total, eps, out_scale, stream);

    if (!cuda_lazy_d2h_for("norm") && y->host_ptr)
        cudaMemcpy(y->host_ptr, cuda_dev_ptr(y), (size_t)n_total * sizeof(float),
                   cudaMemcpyDeviceToHost);
    return OK;
}

static status_code cuda_op_rmsnorm_add_batch(backend *self, const buffer *x, const buffer *w,
                                             const buffer *residual, buffer *y, int n, float eps,
                                             float out_scale, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || m <= 0) return ERR_INTERNAL;
    cuda_rmsnorm_add_batch((const float *)cuda_dev_ptr((buffer *)x),
                           w ? (const float *)cuda_dev_ptr((buffer *)w) : NULL,
                           (const float *)cuda_dev_ptr(residual),
                           (float *)cuda_dev_ptr(y), n, eps, m, out_scale, priv->stream);
    if (!cuda_lazy_d2h_for("norm") && y->host_ptr) {
        if (cudaMemcpyAsync((void *)y->host_ptr, cuda_dev_ptr(y),
                            (size_t)m * (size_t)n * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_rmsnorm_rope(backend *self, const buffer *x, const buffer *w,
                                         buffer *y, int n_heads, int head_dim, int pos,
                                         const float *rope_cos_base, const float *rope_sin_base,
                                         const float *freq_factors, float eps) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    int neox = self->rope_neox;
    int n = n_heads * head_dim;
    int half = head_dim / 2;

    const float *cos_dev = NULL, *sin_dev = NULL;
    if (freq_factors) {
        status_code cs_st = cuda_rope_ensure_cs(self, pos, head_dim, freq_factors);
        if (cs_st != OK) return cs_st;
        cos_dev = priv->rope_cs_cos_dev + (size_t)pos * half;
        sin_dev = priv->rope_cs_sin_dev + (size_t)pos * half;
    } else {
        status_code st = cuda_rope_pre_ensure(self, pos + 1, head_dim, rope_cos_base,
                                              rope_sin_base, &cos_dev, &sin_dev);
        if (st != OK) return st;
        cos_dev += (size_t)pos * half;
        sin_dev += (size_t)pos * half;
    }

    cuda_rmsnorm_rope(cuda_dev_ptr((buffer *)x),
                      w ? (const float *)cuda_dev_ptr((buffer *)w) : NULL,
                      (float *)cuda_dev_ptr(y), n_heads, head_dim,
                      cos_dev, sin_dev, eps, neox, priv->stream);

    if (!cuda_lazy_d2h_for("norm") && y->host_ptr) {
        cudaMemcpy(y->host_ptr, cuda_dev_ptr(y), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
                                     const buffer *x, buffer **y, const int *n_list, int k,
                                     int n_matmuls) {
    if (n_matmuls <= 0 || n_matmuls > 8)
        return ERR_UNSUPPORTED;

    uint32_t eff_types[8];
    for (int c = 0; c < n_matmuls; c++) {
        uint32_t stored = cuda_weight_type_of(cuda_dev_ptr(w[c]));
        eff_types[c]     = (stored != (uint32_t)-1) ? stored : w_types[c];
    }
    w_types = eff_types;

    uint32_t first = w_types[0];
    if (!cuda_matmul_type_native(self, first)) {
        for (int c = 0; c < n_matmuls; c++) {
            status_code st = cuda_matmul(self, w[c], w_types[c], x, y[c], n_list[c], k);
            if (st != OK) return st;
        }
        return OK;
    }
    for (int c = 0; c < n_matmuls; c++)
        if (w_types[c] != first) {
            for (int cc = 0; cc < n_matmuls; cc++) {
                status_code st = cuda_matmul(self, w[cc], w_types[cc], x, y[cc], n_list[cc], k);
                if (st != OK) return st;
            }
            return OK;
        }

    struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;

    const void *w_dev[8];
    float *y_dev[8];
    int n_list_copy[8];
    for (int c = 0; c < n_matmuls; c++) {
        w_dev[c]      = cuda_dev_ptr((buffer *)w[c]);
        y_dev[c]      = (float *)cuda_dev_ptr(y[c]);
        n_list_copy[c] = n_list[c];
    }

    const float *x_dev = (const float *)cuda_dev_ptr((buffer *)x);
    if (first == GGML_TYPE_Q8_0 )
        cuda_matmul_multi_q8_0(w_dev, y_dev, n_list_copy, x_dev, n_matmuls, k, stream);
    else if (first == GGML_TYPE_Q4_0 )
        cuda_matmul_multi_q4_0(w_dev, y_dev, n_list_copy, x_dev, n_matmuls, k, stream);
    else {

        for (int c = 0; c < n_matmuls; c++) {
            status_code st = cuda_matmul(self, w[c], w_types[c], x, y[c], n_list[c], k);
            if (st != OK) return st;
        }
        return OK;
    }

        for (int c = 0; c < n_matmuls; c++) {
        if (!cuda_lazy_d2h_for("multi") && y[c]->host_ptr) {
            if (cudaMemcpy((void *)y[c]->host_ptr, y_dev[c], (size_t)n_list[c] * sizeof(float),
                            cudaMemcpyDeviceToHost) != cudaSuccess)
                return ERR_INTERNAL;
        }
    }
    return OK;
}

static status_code cuda_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
                                           const buffer *x, buffer **y, const int *n_list, int k,
                                           int n_matmuls, int m) {
    if (n_matmuls <= 0 || n_matmuls > 8 || m <= 0)
        return ERR_UNSUPPORTED;

    uint32_t eff_types[8];
    for (int c = 0; c < n_matmuls; c++) {
        uint32_t stored = cuda_weight_type_of(cuda_dev_ptr(w[c]));
        eff_types[c]     = (stored != (uint32_t)-1) ? stored : w_types[c];
    }
    w_types = eff_types;

    uint32_t first = w_types[0];
    for (int c = 0; c < n_matmuls; c++)
        if (w_types[c] != first)
            first = UINT32_MAX;
    struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;
    const void *w_dev[8];
    float *y_dev[8];
    int n_list_copy[8];
    for (int c = 0; c < n_matmuls; c++) {
        w_dev[c]       = cuda_dev_ptr((buffer *)w[c]);
        y_dev[c]       = (float *)cuda_dev_ptr(y[c]);
        n_list_copy[c] = n_list[c];
    }
    const float *x_dev = (const float *)cuda_dev_ptr((buffer *)x);
    if (first == GGML_TYPE_Q4_0 )
        cuda_matmul_multi_batch_q4_0(w_dev, y_dev, n_list_copy, x_dev, n_matmuls, k, m, NULL,
                                     stream);
    else if (first == GGML_TYPE_Q8_0 )
        cuda_matmul_multi_batch_q8_0(w_dev, y_dev, n_list_copy, x_dev, n_matmuls, k, m, NULL,
                                     stream);
    else {
        for (int c = 0; c < n_matmuls; c++) {
            status_code st = cuda_op_matmul_batch(self, w[c], w_types[c], x, y[c], n_list[c], k, m);
            if (st != OK) return st;
        }
        return OK;
    }
    for (int c = 0; c < n_matmuls; c++) {
        if (!cuda_lazy_d2h_for("multi") && y[c]->host_ptr) {
            if (cudaMemcpyAsync((void *)y[c]->host_ptr, y_dev[c],
                                (size_t)m * (size_t)n_list[c] * sizeof(float),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return ERR_INTERNAL;
        }
    }
    return OK;
}

static status_code cuda_matmul_multi_batch_resid(backend *self, const buffer **w,
                                                 const uint32_t *w_types, const buffer *x,
                                                 buffer **y, const int *n_list, int k,
                                                 int n_matmuls, int m, const buffer *resid) {
    if (n_matmuls <= 0 || n_matmuls > 8 || m <= 0)
        return ERR_UNSUPPORTED;
    if (resid && n_matmuls != 1)
        return ERR_UNSUPPORTED;
    uint32_t first = w_types[0];
    for (int c = 0; c < n_matmuls; c++)
        if (w_types[c] != first)
            first = UINT32_MAX;
    struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;
    const void *w_dev[8];
    float *y_dev[8];
    int n_list_copy[8];
    for (int c = 0; c < n_matmuls; c++) {
        w_dev[c]       = cuda_dev_ptr((buffer *)w[c]);
        y_dev[c]       = (float *)cuda_dev_ptr(y[c]);
        n_list_copy[c] = n_list[c];
    }
    const float *x_dev = (const float *)cuda_dev_ptr((buffer *)x);
    const float *resid_dev = resid ? (const float *)cuda_dev_ptr((buffer *)resid) : NULL;
    if (first == GGML_TYPE_Q4_0 )
        cuda_matmul_multi_batch_q4_0(w_dev, y_dev, n_list_copy, x_dev, n_matmuls, k, m, resid_dev,
                                     stream);
    else if (first == GGML_TYPE_Q8_0 )
        cuda_matmul_multi_batch_q8_0(w_dev, y_dev, n_list_copy, x_dev, n_matmuls, k, m, resid_dev,
                                     stream);
    else {

        const float *rd = NULL;
        if (resid) {
            size_t need = (size_t)m * (size_t)n_list[0] * sizeof(float);
            if (!priv || need == 0)
                return ERR_UNSUPPORTED;
            if (priv->resid_tmp_cap < need) {
                if (priv->resid_tmp_dev)
                    cudaFree(priv->resid_tmp_dev);
                priv->resid_tmp_dev = NULL;
                priv->resid_tmp_cap = 0;
                if (cudaMalloc((void **)&priv->resid_tmp_dev, need) != cudaSuccess)
                    return ERR_OUT_OF_MEMORY;
                priv->resid_tmp_cap = need;
            }
            if (cudaMemcpyAsync(priv->resid_tmp_dev, resid_dev, need, cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess)
                return ERR_OUT_OF_MEMORY;
            rd = priv->resid_tmp_dev;
        }
        for (int c = 0; c < n_matmuls; c++) {
            status_code st = cuda_op_matmul_batch(self, w[c], w_types[c], x, y[c], n_list[c], k, m);
            if (st != OK) return st;
        }
        if (resid) {
            buffer rview = *resid;
            rview.handle = (void *)rd;
            rview.offset = 0;
            rview.host_ptr = NULL;
            for (int c = 0; c < n_matmuls; c++) {
                status_code st = cuda_op_add_batch(self, y[c], &rview, n_list[c], m);
                if (st != OK) return st;
            }
        }
        return OK;
    }
    for (int c = 0; c < n_matmuls; c++) {
        if (!cuda_lazy_d2h_for("multi") && y[c]->host_ptr) {
            if (cudaMemcpyAsync((void *)y[c]->host_ptr, y_dev[c],
                                (size_t)m * (size_t)n_list[c] * sizeof(float),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return ERR_INTERNAL;
        }
    }
    return OK;
}

static status_code cuda_rmsnorm_matmul_multi(backend *self, const buffer *norm_w, float eps,
                                             const buffer **w, const uint32_t *w_types,
                                             const buffer *x, buffer **y, const int *n_list,
                                             int k, int n_matmuls) {
    if (n_matmuls <= 0 || n_matmuls > 8)
        return ERR_UNSUPPORTED;

    uint32_t first = w_types[0];
    if (first != GGML_TYPE_Q8_0 && first != GGML_TYPE_Q4_0 && first != GGML_TYPE_Q4_1)
        return ERR_UNSUPPORTED;
    for (int c = 0; c < n_matmuls; c++)
        if (w_types[c] != first)
            return ERR_UNSUPPORTED;

    struct cuda_priv *priv = cuda_priv(self);
    cudaStream_t stream = priv ? priv->stream : 0;

    int total_rows = 0;
    for (int c = 0; c < n_matmuls; c++)
        total_rows += n_list[c];

    const int fused_max = 512;
    if (total_rows > fused_max) {
        size_t need = (size_t)k * sizeof(float);
        if (!priv || need == 0)
            return ERR_UNSUPPORTED;
        if (priv->rmm_tmp_cap < (size_t)k) {
            if (priv->rmm_tmp_dev)
                cudaFree(priv->rmm_tmp_dev);
            priv->rmm_tmp_dev = NULL;
            priv->rmm_tmp_cap = 0;
            if (cudaMalloc((void **)&priv->rmm_tmp_dev, need) != cudaSuccess)
                return ERR_OUT_OF_MEMORY;
            priv->rmm_tmp_cap = (size_t)k;
        }
        buffer xn = {0};
        xn.handle = priv->rmm_tmp_dev;
        xn.size = need;
        xn.host_ptr = NULL;
        xn.offset = 0;
        xn.owner = self;
        status_code st = cuda_op_rmsnorm(self, x, norm_w, &xn, k, eps);
        if (st != OK)
            return st;
        return cuda_matmul_multi(self, w, w_types, &xn, y, n_list, k, n_matmuls);
    }

    const void *w_dev[8];
    float *y_dev[8];
    int n_list_copy[8];
    for (int c = 0; c < n_matmuls; c++) {
        w_dev[c]       = cuda_dev_ptr((buffer *)w[c]);
        y_dev[c]       = (float *)cuda_dev_ptr(y[c]);
        n_list_copy[c] = n_list[c];
    }

    const float *x_dev = (const float *)cuda_dev_ptr((buffer *)x);
    const float *nw_dev = norm_w ? (const float *)cuda_dev_ptr((buffer *)norm_w) : NULL;

    if (first == GGML_TYPE_Q8_0 )
        cuda_rmsnorm_matmul_multi_q8_0(w_dev, y_dev, n_list_copy, x_dev, nw_dev,
                                       n_matmuls, k, eps, stream);
    else if (first == GGML_TYPE_Q4_0 )
        cuda_rmsnorm_matmul_multi_q4_0(w_dev, y_dev, n_list_copy, x_dev, nw_dev,
                                       n_matmuls, k, eps, stream);
    else if (first == GGML_TYPE_Q4_1)
        cuda_rmsnorm_matmul_multi_q4_1(w_dev, y_dev, n_list_copy, x_dev, nw_dev,
                                       n_matmuls, k, eps, stream);
    else
        return ERR_UNSUPPORTED;

    for (int c = 0; c < n_matmuls; c++) {
        if (!cuda_lazy_d2h_for("rmm") && y[c]->host_ptr) {
            if (cudaMemcpy((void *)y[c]->host_ptr, y_dev[c], (size_t)n_list[c] * sizeof(float),
                            cudaMemcpyDeviceToHost) != cudaSuccess)
                return ERR_INTERNAL;
        }
    }
    return OK;
}

static int cuda_attn_gqa_q(int n_heads, int n_kv_heads, int n_pos) {
    int q = 2;
    if (q < 2) return 1;
    int g = n_kv_heads > 0 ? n_heads / n_kv_heads : 1;
    if (g < 2) return 1;
    if (q > g) q = g;
    if (q > 8) q = 8;
    return q;
}

static status_code cuda_op_attention(backend *self, const buffer *q, const buffer *k_cache,
                                      const buffer *v_cache, buffer *out, int layer, int pos,
                                      int n_heads, int n_kv_heads, int head_dim, int n_ctx,
                                      int flash_attn, float scale, int n_kv_heads_active) {
    (void)flash_attn;

    if (n_kv_heads_active > 0)
        n_kv_heads = n_kv_heads_active;
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;

    cudaStream_t stream = priv->stream;
    int n_pos = pos + 1;
    int n = n_heads * head_dim;

    if (priv->kv_quant_q8) {
        int nb = (int)cuda_kv_q8_nblocks(head_dim);
        size_t lay = cuda_kv_layer_base_bytes(priv, layer);
        size_t stride = (size_t)n_ctx * (size_t)nb * KV_Q8_0_BLOCK_BYTES;

            cuda_attn_q8((const float *)cuda_dev_ptr(q), (const uint8_t *)cuda_dev_ptr(k_cache),
                         (const uint8_t *)cuda_dev_ptr(v_cache), (float *)cuda_dev_ptr(out),
                         n_heads, n_kv_heads, head_dim, lay, stride, 0, n_pos, scale, nb, stream);
        if (!cuda_lazy_d2h_for("attn") && out->host_ptr) {
            cudaMemcpy(out->host_ptr, cuda_dev_ptr(out), (size_t)n * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        return OK;
    }

    size_t lay = cuda_kv_layer_base(priv, layer) * sizeof(uint16_t);
    size_t stride = (size_t)n_ctx * (size_t)head_dim;

        cuda_attn_f16((const float *)cuda_dev_ptr(q), (const uint16_t *)cuda_dev_ptr(k_cache),
                      (const uint16_t *)cuda_dev_ptr(v_cache), (float *)cuda_dev_ptr(out),
                      n_heads, n_kv_heads, head_dim, lay, stride, 0, n_pos, scale, stream);

    if (!cuda_lazy_d2h_for("attn") && out->host_ptr) {
        cudaMemcpy(out->host_ptr, cuda_dev_ptr(out), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_op_attention_swa(backend *self, const buffer *q, const buffer *k_cache,
                                          const buffer *v_cache, buffer *out, int layer, int pos,
                                          int n_heads, int n_kv_heads, int head_dim, int n_ctx,
                                          int flash_attn, float scale, int sliding_window,
                                          int n_kv_heads_active) {
    (void)flash_attn;

    if (n_kv_heads_active > 0)
        n_kv_heads = n_kv_heads_active;

    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;

    int n = n_heads * head_dim;

    int n_pos_total = pos + 1;
    int attn_start = 0;
    if (sliding_window > 0 && n_pos_total > sliding_window) {
        attn_start = n_pos_total - sliding_window;
        n_pos_total = sliding_window;
    }

    if (priv->kv_quant_q8) {
        int nb = (int)cuda_kv_q8_nblocks(head_dim);
        size_t row_bytes = (size_t)nb * KV_Q8_0_BLOCK_BYTES;
        size_t lay = cuda_kv_layer_base_bytes(priv, layer);
        size_t stride = (size_t)n_ctx * row_bytes;

        size_t lay_off = lay + (size_t)attn_start * row_bytes;

            cuda_attn_q8((const float *)cuda_dev_ptr(q), (const uint8_t *)cuda_dev_ptr(k_cache),
                         (const uint8_t *)cuda_dev_ptr(v_cache), (float *)cuda_dev_ptr(out),
                         n_heads, n_kv_heads, head_dim, lay_off, stride, 0, n_pos_total, scale,
                         nb, priv->stream);
        if (!cuda_lazy_d2h_for("attn") && out->host_ptr) {
            cudaMemcpy(out->host_ptr, cuda_dev_ptr(out), (size_t)n * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        return OK;
    }

    size_t lay = cuda_kv_layer_base(priv, layer) * sizeof(uint16_t);
    size_t stride = (size_t)n_ctx * (size_t)head_dim;

    size_t lay_off = lay + (size_t)attn_start * (size_t)head_dim * sizeof(uint16_t);

    cuda_attn_f16((const float *)cuda_dev_ptr(q), (const uint16_t *)cuda_dev_ptr(k_cache),
                  (const uint16_t *)cuda_dev_ptr(v_cache), (float *)cuda_dev_ptr(out),
                  n_heads, n_kv_heads, head_dim, lay_off, stride, 0, n_pos_total, scale,
                  priv->stream);

    if (!cuda_lazy_d2h_for("attn") && out->host_ptr) {
        cudaMemcpy(out->host_ptr, cuda_dev_ptr(out), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!desc || !k_out || !v_out) return ERR_INVALID_ARG;
    if (desc->kv_quant != KV_QUANT_F16 && desc->kv_quant != KV_QUANT_Q8_0)
        return ERR_UNSUPPORTED;

    int is_q8 = (desc->kv_quant == KV_QUANT_Q8_0);
    int n_kv_layers = desc->n_kv_layers > 0 ? desc->n_kv_layers : 1;
    int n_heads = desc->n_kv_heads;
    int n_ctx = desc->n_ctx;
    int has_var = desc->layer_head_dim && desc->n_kv_layers > 0;

    free(priv->kv_layer_off);
    priv->kv_layer_off = NULL;
    size_t total = 0;

    if (has_var) {
        priv->kv_layer_off = (size_t *)calloc((size_t)n_kv_layers + 1, sizeof(size_t));
        if (!priv->kv_layer_off) return ERR_OUT_OF_MEMORY;
        for (int i = 0; i < n_kv_layers; i++) {
            priv->kv_layer_off[i] = total;
            if (is_q8)
                total += (size_t)n_heads * (size_t)n_ctx *
                         cuda_kv_q8_nblocks(desc->layer_head_dim[i]) * KV_Q8_0_BLOCK_BYTES;
            else
                total += (size_t)n_heads * (size_t)n_ctx * (size_t)desc->layer_head_dim[i];
        }
        priv->kv_layer_off[n_kv_layers] = total;
        priv->kv_head_dim = desc->layer_head_dim[0];
    } else {
        priv->kv_head_dim = desc->head_dim;
        if (is_q8)
            total = (size_t)n_heads * (size_t)n_ctx * cuda_kv_q8_nblocks(desc->head_dim) *
                    KV_Q8_0_BLOCK_BYTES * (size_t)n_kv_layers;
        else
            total = (size_t)n_heads * (size_t)n_ctx * (size_t)desc->head_dim *
                    (size_t)n_kv_layers;
    }

    if (!is_q8)
        total *= sizeof(uint16_t);
    priv->kv_quant_q8 = is_q8 ? 1 : 0;
    priv->kv_n_layers = n_kv_layers;
    priv->kv_n_kv_heads = n_heads;
    priv->kv_n_ctx = n_ctx;

    void *kd, *vd;
    if (cudaMalloc(&kd, total) != cudaSuccess) return ERR_OUT_OF_MEMORY;
    if (cudaMalloc(&vd, total) != cudaSuccess) { cudaFree(kd); return ERR_OUT_OF_MEMORY; }

    k_out->handle = kd;
    k_out->size = total;
    k_out->host_ptr = NULL;
    k_out->owner = self;
    v_out->handle = vd;
    v_out->size = total;
    v_out->host_ptr = NULL;
    v_out->owner = self;
    return OK;
}

static void cuda_kv_free(backend *self, buffer *k, buffer *v) {
    struct cuda_priv *priv = cuda_priv(self);
    if (priv) {
        free(priv->kv_layer_off);
        priv->kv_layer_off = NULL;
    }
    if (k && k->handle) { cudaFree(k->handle); k->handle = NULL; }
    if (v && v->handle) { cudaFree(v->handle); v->handle = NULL; }
}

static status_code cuda_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos,
                                const buffer *k_in, const buffer *v_in, int n_kv_heads,
                                int head_dim, int n_ctx, int n_kv_heads_active) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

    if (priv->kv_quant_q8) {
        int nb = (int)cuda_kv_q8_nblocks(head_dim);
        size_t lay = cuda_kv_layer_base_bytes(priv, layer);
        size_t stride = (size_t)n_ctx * (size_t)nb * KV_Q8_0_BLOCK_BYTES;
        cuda_kv_put_q8((const float *)cuda_dev_ptr(k_in), (const float *)cuda_dev_ptr(v_in),
                       (uint8_t *)cuda_dev_ptr(k), (uint8_t *)cuda_dev_ptr(v),
                       n_active, head_dim, lay, stride, pos, nb, priv->stream);
        return OK;
    }

    size_t lay = cuda_kv_layer_base(priv, layer) * sizeof(uint16_t);
    size_t stride = (size_t)n_ctx * (size_t)head_dim;

    cuda_kv_put_f16((const float *)cuda_dev_ptr(k_in), (const float *)cuda_dev_ptr(v_in),
                    (uint16_t *)cuda_dev_ptr(k), (uint16_t *)cuda_dev_ptr(v),
                    n_active, head_dim, lay, stride, pos, n_ctx, priv->stream);
    return OK;
}

static status_code cuda_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
                                       const buffer *k_in, const buffer *v_in, int in_row_stride,
                                       int n_kv_heads, int head_dim, int n_ctx,
                                       int n_kv_heads_active, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
    if (priv->kv_quant_q8) {
        int nb = (int)cuda_kv_q8_nblocks(head_dim);
        size_t lay = cuda_kv_layer_base_bytes(priv, layer);
        size_t stride = (size_t)n_ctx * (size_t)nb * KV_Q8_0_BLOCK_BYTES;
        cuda_kv_put_batch_q8((const float *)cuda_dev_ptr(k_in), (const float *)cuda_dev_ptr(v_in),
                             (uint8_t *)cuda_dev_ptr(k), (uint8_t *)cuda_dev_ptr(v),
                             n_active, head_dim, lay, stride, pos_start, m, nb,
                             in_row_stride, priv->stream);
        return OK;
    }
    size_t lay = cuda_kv_layer_base(priv, layer) * sizeof(uint16_t);
    size_t stride = (size_t)n_ctx * (size_t)head_dim;
    cuda_kv_put_batch_f16((const float *)cuda_dev_ptr(k_in), (const float *)cuda_dev_ptr(v_in),
                          (uint16_t *)cuda_dev_ptr(k), (uint16_t *)cuda_dev_ptr(v),
                          n_active, head_dim, lay, stride, pos_start, n_ctx,
                          in_row_stride, m, priv->stream);
    return OK;
}

static status_code cuda_op_add_inplace(backend *self, buffer *x, const buffer *y, int n) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;

    cuda_add_inplace((float *)cuda_dev_ptr(x), (const float *)cuda_dev_ptr(y), n, priv->stream);

    if (!cuda_lazy_d2h_for("elem") && x->host_ptr) {
        cudaMemcpy(x->host_ptr, cuda_dev_ptr(x), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_op_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    cuda_add_batch((float *)cuda_dev_ptr(x), (const float *)cuda_dev_ptr(y), n, m, priv->stream);
    if (x->host_ptr) {
        if (cudaMemcpyAsync((void *)x->host_ptr, cuda_dev_ptr(x), (size_t)m * (size_t)n * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_scale_inplace(backend *self, buffer *x, float scale, int n) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    cuda_scale_inplace((float *)cuda_dev_ptr(x), scale, n, priv->stream);
    if (!cuda_lazy_d2h_for("elem") && x->host_ptr) {
        cudaMemcpy(x->host_ptr, cuda_dev_ptr(x), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_op_softcap(backend *self, buffer *x, float cap, int n) {
	struct cuda_priv *priv = cuda_priv(self);
	if (!priv)
		return ERR_INTERNAL;
	if (cap <= 0.0f || n <= 0)
		return OK;
	cuda_softcap((float *)cuda_dev_ptr(x), cap, n, priv->stream);
	if (!cuda_lazy_d2h_for("elem") && x->host_ptr)
		cudaMemcpy(x->host_ptr, cuda_dev_ptr(x), (size_t)n * sizeof(float),
				   cudaMemcpyDeviceToHost);
	return OK;
}

static status_code cuda_op_attn_output_gate(backend *self, buffer *out, const buffer *gate, int n,
										   int n_rows) {
	struct cuda_priv *priv = cuda_priv(self);
	if (!priv)
		return ERR_INTERNAL;
	if (n <= 0 || n_rows <= 0)
		return OK;
	cuda_attn_output_gate((float *)cuda_dev_ptr(out), (const float *)cuda_dev_ptr(gate),
						  (long long)n * n_rows, priv->stream);
	return OK;
}

static status_code cuda_op_split_qgate(backend *self, const buffer *mixed, buffer *q, buffer *gate,
									  int n_heads, int head_dim, int n_rows) {
	struct cuda_priv *priv = cuda_priv(self);
	if (!priv)
		return ERR_INTERNAL;
	if (n_rows <= 0 || n_heads <= 0 || head_dim <= 0)
		return OK;
	cuda_split_qgate((const float *)cuda_dev_ptr(mixed), (float *)cuda_dev_ptr(q),
					 (float *)cuda_dev_ptr(gate), n_heads, head_dim, n_rows, priv->stream);
	return OK;
}

static status_code cuda_op_moe_activate(backend *self, const buffer *gate, const buffer *up,
									   buffer *out, int n, float gate_scale, float up_scale,
									   int use_gelu) {
	struct cuda_priv *priv = cuda_priv(self);
	if (!priv)
		return ERR_INTERNAL;
	if (n <= 0)
		return OK;
	cuda_moe_activate((const float *)cuda_dev_ptr(gate), (const float *)cuda_dev_ptr(up),
					  (float *)cuda_dev_ptr(out), n, gate_scale, up_scale, use_gelu ? 1 : 0,
					  priv->stream);
	return OK;
}

static status_code cuda_op_moe_experts_batch(backend *self, const buffer *xb, buffer *out,
                                             int n_rows, int dim, int inter, int use_gelu,
                                             int n_experts, const moe_resident_expert *experts,
                                             const int *counts, const int *rows_packed,
                                             const float *weights_packed) {

    (void)xb; (void)out; (void)n_rows; (void)dim; (void)inter; (void)use_gelu;
    (void)n_experts; (void)experts; (void)counts; (void)rows_packed; (void)weights_packed;
    return ERR_UNSUPPORTED;
}

static status_code cuda_op_partial_rope_qk(backend *self, buffer *q, buffer *k, int n_heads,
                                           int n_kv_heads, int head_dim, int rope_dim,
                                           int pos_start, const float *rope_cos_base,
                                           const float *rope_sin_base, int n_rows) {
	struct cuda_priv *priv = cuda_priv(self);
	if (!priv)
		return ERR_INTERNAL;
	if (n_rows <= 0 || rope_dim <= 0 || n_heads <= 0 || n_kv_heads <= 0)
		return OK;

	const float *cos_base = NULL, *sin_base = NULL;
	status_code st = cuda_rope_pre_ensure(self, pos_start + n_rows, rope_dim, rope_cos_base,
										  rope_sin_base, &cos_base, &sin_base);
	if (st != OK)
		return st;
	cuda_partial_rope((float *)cuda_dev_ptr(q), n_heads, head_dim, rope_dim, pos_start, cos_base,
					  sin_base, n_rows, priv->stream);
	cuda_partial_rope((float *)cuda_dev_ptr(k), n_kv_heads, head_dim, rope_dim, pos_start,
					  cos_base, sin_base, n_rows, priv->stream);
	if (!cuda_lazy_d2h_for("rope")) {
		if (q->host_ptr &&
			cudaMemcpyAsync(q->host_ptr, cuda_dev_ptr(q),
							(size_t)n_rows * (size_t)n_heads * (size_t)head_dim * sizeof(float),
							cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
			return ERR_INTERNAL;
		if (k->host_ptr &&
			cudaMemcpyAsync(k->host_ptr, cuda_dev_ptr(k),
							(size_t)n_rows * (size_t)n_kv_heads * (size_t)head_dim *
								sizeof(float),
							cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
			return ERR_INTERNAL;
	}
	return OK;
}

static status_code cuda_op_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
                                       float scale) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || n <= 0) return ERR_INTERNAL;
    cuda_ple_combine((float *)cuda_dev_ptr(ple), (const float *)cuda_dev_ptr((buffer *)proj), n,
                     scale, priv->stream);
    return OK;
}

static status_code cuda_op_ple_norm_batch(backend *self, buffer *proj, const buffer *norm_w,
                                          int n_rows, int total_ple, int n_embd, int n_layers,
                                          float eps) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || n_rows <= 0) return ERR_INTERNAL;
    cuda_ple_norm_batch((float *)cuda_dev_ptr(proj),
                        norm_w ? (const float *)cuda_dev_ptr((buffer *)norm_w) : NULL, n_rows,
                        total_ple, n_embd, n_layers, eps, priv->stream);
    return OK;
}

static status_code cuda_op_ffn_activate(backend *self, const buffer *gate, const buffer *up,
                                         buffer *out, int n) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;

    cuda_ffn_activate((const float *)cuda_dev_ptr(gate), (const float *)cuda_dev_ptr(up),
                      (float *)cuda_dev_ptr(out), n, 0, priv->stream);

    if (!cuda_lazy_d2h_for("elem") && out->host_ptr) {
        cudaMemcpy(out->host_ptr, cuda_dev_ptr(out), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_op_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up,
                                            buffer *out, int n, int activation) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;

    cuda_ffn_activate((const float *)cuda_dev_ptr(gate), (const float *)cuda_dev_ptr(up),
                      (float *)cuda_dev_ptr(out), n, activation, priv->stream);

    if (!cuda_lazy_d2h_for("elem") && out->host_ptr) {
        cudaMemcpy(out->host_ptr, cuda_dev_ptr(out), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_op_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up,
                                                buffer *out, int n, int activation, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    cuda_ffn_activate_batch((const float *)cuda_dev_ptr((buffer *)gate),
                            (const float *)cuda_dev_ptr((buffer *)up),
                            (float *)cuda_dev_ptr(out), n, activation, m, priv->stream);
    if (out->host_ptr) {
        if (cudaMemcpyAsync((void *)out->host_ptr, cuda_dev_ptr(out), (size_t)m * (size_t)n * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_ffn_activate_fused_batch(backend *self, const buffer *fused,
                                                    buffer *out, int n, int activation, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    cuda_ffn_activate_fused_batch((const float *)cuda_dev_ptr((buffer *)fused),
                                  (float *)cuda_dev_ptr(out), n, activation, m, priv->stream);
    if (out->host_ptr) {
        if (cudaMemcpyAsync((void *)out->host_ptr, cuda_dev_ptr(out),
                            (size_t)m * (size_t)n * sizeof(float), cudaMemcpyDeviceToHost,
                            priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_rope_pre_ensure(backend *self, int pos_end, int head_dim,
                                        const float *rope_cos_base,
                                        const float *rope_sin_base,
                                        const float **cos_dev, const float **sin_dev) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || !rope_cos_base || !rope_sin_base || !cos_dev || !sin_dev)
        return ERR_INTERNAL;
    int half = head_dim / 2;
    if (half <= 0 || pos_end <= 0) {

        *cos_dev = priv->rope_tbl[0].cos_dev;
        *sin_dev = priv->rope_tbl[0].sin_dev;
        return OK;
    }
    int slot = -1;
    for (int i = 0; i < ROPE_TBL_SLOTS; i++) {
        if (priv->rope_tbl[i].half == half) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        for (int i = 0; i < ROPE_TBL_SLOTS; i++) {
            if (priv->rope_tbl[i].half < 0) {
                slot = i;
                break;
            }
        }
    }
    if (slot < 0) {

        slot = 0;
        priv->rope_tbl[slot].half = -1;
        priv->rope_tbl[slot].up_pos = -1;
    }

    size_t needed_rows = (size_t)pos_end > 0 ? (size_t)pos_end : 0;
    size_t unit = (size_t)1024 * (size_t)half;
    size_t want_cap = unit;
    if (needed_rows * (size_t)half > want_cap)
        want_cap = ((needed_rows * (size_t)half + unit - 1) / unit) * unit;
    if (priv->rope_tbl[slot].half != half || priv->rope_tbl[slot].cap < want_cap ||
        priv->rope_tbl[slot].cos_src != rope_cos_base ||
        priv->rope_tbl[slot].sin_src != rope_sin_base) {

        if (priv->rope_tbl[slot].cos_dev)
            cudaFree(priv->rope_tbl[slot].cos_dev);
        if (priv->rope_tbl[slot].sin_dev)
            cudaFree(priv->rope_tbl[slot].sin_dev);
        priv->rope_tbl[slot].cos_dev = NULL;
        priv->rope_tbl[slot].sin_dev = NULL;
        if (cudaMalloc((void **)&priv->rope_tbl[slot].cos_dev,
                       want_cap * sizeof(float)) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
        if (cudaMalloc((void **)&priv->rope_tbl[slot].sin_dev,
                       want_cap * sizeof(float)) != cudaSuccess) {
            cudaFree(priv->rope_tbl[slot].cos_dev);
            priv->rope_tbl[slot].cos_dev = NULL;
            return ERR_OUT_OF_MEMORY;
        }
        priv->rope_tbl[slot].cap = want_cap;
        priv->rope_tbl[slot].half = half;
        priv->rope_tbl[slot].up_pos = -1;
        priv->rope_tbl[slot].cos_src = rope_cos_base;
        priv->rope_tbl[slot].sin_src = rope_sin_base;
    }
    int max_rows = (int)(priv->rope_tbl[slot].cap / (size_t)half);
    int end = pos_end < max_rows ? pos_end : max_rows;
    int up = priv->rope_tbl[slot].up_pos;
    if (end > up + 1) {
        size_t off = (size_t)(up + 1) * (size_t)half;
        size_t cnt = (size_t)(end - up - 1) * (size_t)half;
        if (cudaMemcpyAsync(priv->rope_tbl[slot].cos_dev + off, rope_cos_base + off,
                            cnt * sizeof(float), cudaMemcpyHostToDevice,
                            priv->stream) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
        if (cudaMemcpyAsync(priv->rope_tbl[slot].sin_dev + off, rope_sin_base + off,
                            cnt * sizeof(float), cudaMemcpyHostToDevice,
                            priv->stream) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
        priv->rope_tbl[slot].up_pos = end - 1;
    }
    *cos_dev = priv->rope_tbl[slot].cos_dev;
    *sin_dev = priv->rope_tbl[slot].sin_dev;
    return OK;
}

static status_code cuda_rope_ensure_cs(backend *self, int pos, int head_dim,
                                       const float *freq_factors) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || !freq_factors) return ERR_INTERNAL;
    float theta = self->rope_theta;
    int half = head_dim / 2;
    size_t table_cap = (size_t)1024 * (size_t)half;
    size_t needed = (size_t)(pos + 1) * (size_t)half;
    if (!priv->rope_cs_cos_dev || priv->rope_cs_dev_cap < table_cap) {
        if (priv->rope_cs_cos_dev) cudaFree(priv->rope_cs_cos_dev);
        if (priv->rope_cs_sin_dev) cudaFree(priv->rope_cs_sin_dev);
        if (cudaMalloc((void **)&priv->rope_cs_cos_dev, table_cap * sizeof(float)) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
        if (cudaMalloc((void **)&priv->rope_cs_sin_dev, table_cap * sizeof(float)) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
        priv->rope_cs_dev_cap = table_cap;
    }
    if (needed > table_cap) {

        size_t unit = (size_t)1024 * (size_t)half;
        size_t new_cap = ((needed + unit - 1) / unit) * unit;
        if (priv->rope_cs_cos_dev) cudaFree(priv->rope_cs_cos_dev);
        if (priv->rope_cs_sin_dev) cudaFree(priv->rope_cs_sin_dev);
        priv->rope_cs_cos_dev = NULL;
        priv->rope_cs_sin_dev = NULL;
        if (cudaMalloc((void **)&priv->rope_cs_cos_dev, new_cap * sizeof(float)) != cudaSuccess)
            return ERR_OUT_OF_MEMORY;
        if (cudaMalloc((void **)&priv->rope_cs_sin_dev, new_cap * sizeof(float)) != cudaSuccess) {
            cudaFree(priv->rope_cs_cos_dev);
            priv->rope_cs_cos_dev = NULL;
            return ERR_OUT_OF_MEMORY;
        }
        priv->rope_cs_dev_cap = new_cap;
        table_cap = new_cap;
        priv->rope_cs_pos = -1;
    }
    if (!priv->rope_cs_host || priv->rope_cs_hd != head_dim ||
        priv->rope_cs_theta != theta || priv->rope_cs_ff != freq_factors ||
        priv->rope_cs_cap < table_cap * 2) {

        cudaStreamSynchronize(priv->stream);
        free(priv->rope_cs_host);
        priv->rope_cs_host = (float *)malloc(table_cap * 2 * sizeof(float));
        if (!priv->rope_cs_host) return ERR_OUT_OF_MEMORY;
        priv->rope_cs_cap = table_cap * 2;
        priv->rope_cs_hd = head_dim;
        priv->rope_cs_theta = theta;
        priv->rope_cs_ff = freq_factors;
        priv->rope_cs_pos = -1;
    }
    if (pos <= priv->rope_cs_pos)
        return OK;

    float *cos_tbl = priv->rope_cs_host;
    float *sin_tbl = priv->rope_cs_host + table_cap;
    double theta_scale = pow((double)theta, -2.0 / (double)head_dim);
    for (int p = priv->rope_cs_pos + 1; p <= pos; p++) {
        double base_freq = 1.0;
        for (int j = 0; j < half; j++) {
            double ff = freq_factors[j];
            ff = (ff >= 1e10f || ff == 0.0f) ? 0.0 : (double)ff;
            if (ff > 0.0) {
                double angle = base_freq * ff * (double)p;
                cos_tbl[(size_t)p * half + j] = (float)cos(angle);
                sin_tbl[(size_t)p * half + j] = (float)sin(angle);
            } else {
                cos_tbl[(size_t)p * half + j] = 1.0f;
                sin_tbl[(size_t)p * half + j] = 0.0f;
            }
            base_freq *= theta_scale;
        }
    }
    size_t off = ((size_t)priv->rope_cs_pos + 1) * (size_t)half;
    size_t cnt = ((size_t)pos - (size_t)priv->rope_cs_pos) * (size_t)half;
    if (cudaMemcpyAsync(priv->rope_cs_cos_dev + off, cos_tbl + off, cnt * sizeof(float),
                        cudaMemcpyHostToDevice, priv->stream) != cudaSuccess)
        return ERR_OUT_OF_MEMORY;
    if (cudaMemcpyAsync(priv->rope_cs_sin_dev + off, sin_tbl + off, cnt * sizeof(float),
                        cudaMemcpyHostToDevice, priv->stream) != cudaSuccess)
        return ERR_OUT_OF_MEMORY;
    priv->rope_cs_pos = pos;
    return OK;
}

static status_code cuda_op_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim,
                                     int pos, const float *rope_cos_base, const float *rope_sin_base,
                                     const float *freq_factors) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    int neox = self->rope_neox;
    int n = n_heads * head_dim;
    int half = head_dim / 2;

    const float *cos_base = NULL, *sin_base = NULL;
    if (freq_factors) {
        status_code cs_st = cuda_rope_ensure_cs(self, pos, head_dim, freq_factors);
        if (cs_st != OK) return cs_st;
        cos_base = priv->rope_cs_cos_dev;
        sin_base = priv->rope_cs_sin_dev;
    } else {
        status_code st = cuda_rope_pre_ensure(self, pos + 1, head_dim, rope_cos_base,
                                              rope_sin_base, &cos_base, &sin_base);
        if (st != OK) return st;
    }

    cuda_rope_ext((float *)cuda_dev_ptr(vec), n_heads, head_dim, pos,
                      cos_base, sin_base, neox, priv->stream);

    if (!cuda_lazy_d2h_for("rope") && vec->host_ptr) {
        cudaMemcpy(vec->host_ptr, cuda_dev_ptr(vec), (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_op_rope(backend *self, buffer *vec, int n_heads, int head_dim,
                                 int pos, const float *rope_cos_base, const float *rope_sin_base) {
    return cuda_op_rope_ext(self, vec, n_heads, head_dim, pos, rope_cos_base, rope_sin_base, NULL);
}

static status_code cuda_rope_tables_ensure(backend *self, int pos_end, int head_dim,
                                           const float *rope_cos_base,
                                           const float *rope_sin_base,
                                           const float **cos_dev, const float **sin_dev) {
    return cuda_rope_pre_ensure(self, pos_end, head_dim, rope_cos_base,
                                rope_sin_base, cos_dev, sin_dev);
}

static status_code cuda_op_rope_qk(backend *self, buffer *q, buffer *k, int n_heads,
                                    int n_kv_heads, int head_dim, int pos,
                                    const float *rope_cos_base, const float *rope_sin_base) {
    int neox = self->rope_neox;
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;

    int q_n = n_heads * head_dim;
    int k_n = n_kv_heads * head_dim;

    const float *cos_base = NULL, *sin_base = NULL;
    status_code tbl_st = cuda_rope_pre_ensure(self, pos + 1, head_dim, rope_cos_base,
                                              rope_sin_base, &cos_base, &sin_base);
    if (tbl_st != OK) return tbl_st;

    {
        cuda_rope_ext((float *)cuda_dev_ptr(q), n_heads, head_dim, pos,
                      cos_base, sin_base, neox, priv->stream);
        cuda_rope_ext((float *)cuda_dev_ptr(k), n_kv_heads, head_dim, pos,
                      cos_base, sin_base, neox, priv->stream);
    }

    if (!cuda_lazy_d2h_for("rope") && q->host_ptr) {
        cudaMemcpy(q->host_ptr, cuda_dev_ptr(q), (size_t)q_n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    if (!cuda_lazy_d2h_for("rope") && k->host_ptr) {
        cudaMemcpy(k->host_ptr, cuda_dev_ptr(k), (size_t)k_n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_op_qkv_norm_rope(backend *self, buffer *q, buffer *k, buffer *v,
                                         const buffer *wq_norm, const buffer *wk_norm,
                                         int n_heads, int n_kv_heads, int head_dim, float eps,
                                         int pos, const float *rope_cos_base,
                                         const float *rope_sin_base,
                                         const float *freq_factors, int neox) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    int half = head_dim / 2;

    const float *cos_base = NULL, *sin_base = NULL;
    if (freq_factors) {
        status_code cs_st = cuda_rope_ensure_cs(self, pos, head_dim, freq_factors);
        if (cs_st != OK) return cs_st;
        cos_base = priv->rope_cs_cos_dev;
        sin_base = priv->rope_cs_sin_dev;
    } else {
        status_code st = cuda_rope_pre_ensure(self, pos + 1, head_dim, rope_cos_base,
                                              rope_sin_base, &cos_base, &sin_base);
        if (st != OK) return st;
    }

    {
        const float *cos_tbl = cos_base + (size_t)pos * half;
        const float *sin_tbl = sin_base + (size_t)pos * half;
        cuda_qkv_norm_rope((float *)cuda_dev_ptr(q), (float *)cuda_dev_ptr(k),
                           (float *)cuda_dev_ptr(v),
                           wq_norm ? (const float *)cuda_dev_ptr((buffer *)wq_norm) : NULL,
                           wk_norm ? (const float *)cuda_dev_ptr((buffer *)wk_norm) : NULL,
                           n_heads, n_kv_heads, head_dim, eps, pos, cos_tbl, sin_tbl,
                           neox, priv->stream);
    }

    int q_n = n_heads * head_dim, k_n = n_kv_heads * head_dim;
    if (!cuda_lazy_d2h_for("norm") && q->host_ptr)
        cudaMemcpy(q->host_ptr, cuda_dev_ptr(q), (size_t)q_n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    if (!cuda_lazy_d2h_for("norm") && k->host_ptr)
        cudaMemcpy(k->host_ptr, cuda_dev_ptr(k), (size_t)k_n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    if (!cuda_lazy_d2h_for("norm") && v->host_ptr)
        cudaMemcpy(v->host_ptr, cuda_dev_ptr(v), (size_t)k_n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    if (!cuda_lazy_d2h_for("rope") && q->host_ptr)
        cudaMemcpy(q->host_ptr, cuda_dev_ptr(q), (size_t)q_n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    if (!cuda_lazy_d2h_for("rope") && k->host_ptr)
        cudaMemcpy(k->host_ptr, cuda_dev_ptr(k), (size_t)k_n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    return OK;
}

static status_code cuda_op_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim,
                                              int pos_start, const float *rope_cos_base,
                                              const float *rope_sin_base,
                                              const float *freq_factors, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || m <= 0) return ERR_INTERNAL;
    int row_elems = n_heads * head_dim;
    const float *cos_base = NULL, *sin_base = NULL;
    if (freq_factors) {
        status_code cs_st = cuda_rope_ensure_cs(self, pos_start + m - 1, head_dim, freq_factors);
        if (cs_st != OK) return cs_st;
        cos_base = priv->rope_cs_cos_dev;
        sin_base = priv->rope_cs_sin_dev;
    } else {
        status_code st = cuda_rope_pre_ensure(self, pos_start + m, head_dim, rope_cos_base,
                                              rope_sin_base, &cos_base, &sin_base);
        if (st != OK) return st;
    }
    cuda_rope_batch((float *)cuda_dev_ptr(vec), n_heads, head_dim, pos_start, cos_base, sin_base,
                    m, self->rope_neox, priv->stream);
    if (!cuda_lazy_d2h_for("rope") && vec->host_ptr) {
        if (cudaMemcpyAsync((void *)vec->host_ptr, cuda_dev_ptr(vec),
                            (size_t)m * (size_t)row_elems * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_rope_batch(backend *self, buffer *vec, int n_heads, int head_dim,
                                        int pos_start, const float *rope_cos_base,
                                        const float *rope_sin_base, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    int row_elems = n_heads * head_dim;
    const float *cos_base = NULL, *sin_base = NULL;
    status_code st = cuda_rope_tables_ensure(self, pos_start + m, head_dim,
                                             rope_cos_base, rope_sin_base,
                                             &cos_base, &sin_base);
    if (st != OK) return st;
    cuda_rope_batch((float *)cuda_dev_ptr(vec), n_heads, head_dim, pos_start,
                    cos_base, sin_base, m, self->rope_neox, priv->stream);
    if (!cuda_lazy_d2h_for("rope") && vec->host_ptr) {
        if (cudaMemcpyAsync((void *)vec->host_ptr, cuda_dev_ptr(vec),
                            (size_t)m * (size_t)row_elems * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_rope_qk_batch(backend *self, buffer *q, buffer *k, int n_heads,
                                            int n_kv_heads, int head_dim, int pos_start,
                                            const float *rope_cos_base,
                                            const float *rope_sin_base, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    int q_row = n_heads * head_dim;
    int k_row = n_kv_heads * head_dim;
    const float *cos_base = NULL, *sin_base = NULL;
    status_code st = cuda_rope_tables_ensure(self, pos_start + m, head_dim,
                                             rope_cos_base, rope_sin_base,
                                             &cos_base, &sin_base);
    if (st != OK) return st;
    cuda_rope_qk_batch((float *)cuda_dev_ptr(q), (float *)cuda_dev_ptr(k), n_heads, n_kv_heads,
                       head_dim, pos_start, cos_base, sin_base,
                       m, self->rope_neox, priv->stream);
    if (!cuda_lazy_d2h_for("rope") && q->host_ptr) {
        if (cudaMemcpyAsync((void *)q->host_ptr, cuda_dev_ptr(q),
                            (size_t)m * (size_t)q_row * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    if (!cuda_lazy_d2h_for("rope") && k->host_ptr) {
        if (cudaMemcpyAsync((void *)k->host_ptr, cuda_dev_ptr(k),
                            (size_t)m * (size_t)k_row * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
                                        int token, int dim, buffer *x_out) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;

    uint32_t stored = cuda_weight_type_of(tok_embd ? cuda_dev_ptr((buffer *)tok_embd) : NULL);
    if (stored != (uint32_t)-1)
        tok_embd_type = stored;

    switch (tok_embd_type) {
        case GGML_TYPE_F32:
            cuda_embd_lookup_f32((const float *)cuda_dev_ptr(tok_embd),
                                 (float *)cuda_dev_ptr(x_out), token, dim, priv->stream);
            break;
        case GGML_TYPE_F16:
            cuda_embd_lookup_f16((const uint16_t *)cuda_dev_ptr(tok_embd),
                                 (float *)cuda_dev_ptr(x_out), token, dim, priv->stream);
            break;
        case GGML_TYPE_BF16:
            cuda_embd_lookup_bf16((const uint16_t *)cuda_dev_ptr(tok_embd),
                                 (float *)cuda_dev_ptr(x_out), token, dim, priv->stream);
            break;
        case GGML_TYPE_Q8_0:
            cuda_embd_lookup_q8_0((const cuda_q8_0_block *)cuda_dev_ptr(tok_embd),
                                 (float *)cuda_dev_ptr(x_out), token, dim, priv->stream);
            break;
        case GGML_TYPE_Q4_0:
            cuda_embd_lookup_q4_0((const cuda_q4_0_block *)cuda_dev_ptr(tok_embd),
                                 (float *)cuda_dev_ptr(x_out), token, dim, priv->stream);
            break;
        case GGML_TYPE_Q4_K:
            cuda_embd_lookup_q4_k((const cuda_q4_k_block *)cuda_dev_ptr(tok_embd),
                                 (float *)cuda_dev_ptr(x_out), token, dim, priv->stream);
            break;
        default: {

            backend *host = backend_host();
            if (!host || !host->embd_lookup)
                return ERR_UNSUPPORTED;
            size_t row_bytes = ggml_row_size(tok_embd_type, (size_t)dim);
            if (row_bytes == 0)
                return ERR_UNSUPPORTED;
            void *row_host = malloc(row_bytes);
            float *out_host = malloc((size_t)dim * sizeof(float));
            if (!row_host || !out_host) {
                free(row_host);
                free(out_host);
                return ERR_OUT_OF_MEMORY;
            }
            const void *src =
                (const char *)cuda_dev_ptr((buffer *)tok_embd) + (size_t)token * row_bytes;
            if (cudaMemcpy(row_host, src, row_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                free(row_host);
                free(out_host);
                return ERR_OUT_OF_MEMORY;
            }
            buffer hb = {0}, ob = {0};
            hb.handle = hb.host_ptr = row_host;
            hb.size	  = row_bytes;
            hb.owner  = host;
            ob.handle = ob.host_ptr = out_host;
            ob.size	  = (size_t)dim * sizeof(float);
            ob.owner  = host;
            status_code st = host->embd_lookup(host, &hb, tok_embd_type, 0, dim, &ob);
            if (st == OK) {
                if (cudaMemcpy(cuda_dev_ptr(x_out), out_host, (size_t)dim * sizeof(float),
                               cudaMemcpyHostToDevice) != cudaSuccess)
                    st = ERR_OUT_OF_MEMORY;
            }
            free(row_host);
            free(out_host);
            return st;
        }
    }

    if (!cuda_lazy_d2h_for("embd") && x_out->host_ptr) {
        cudaMemcpy(x_out->host_ptr, cuda_dev_ptr(x_out), (size_t)dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return OK;
}

static status_code cuda_op_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv || !priv->argmax_idx_dev) return ERR_INTERNAL;

    cuda_argmax((const float *)cuda_dev_ptr(logits), n, priv->argmax_idx_dev, priv->stream);

    cudaMemcpy(out_idx, priv->argmax_idx_dev, sizeof(int32_t), cudaMemcpyDeviceToHost);
    return OK;
}

static status_code cuda_op_rmsnorm_batch(backend *self, const buffer *x, const buffer *w,
                                           buffer *y, int n, float eps, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    cuda_rmsnorm_batch((const float *)cuda_dev_ptr((buffer *)x),
                       w ? (const float *)cuda_dev_ptr((buffer *)w) : NULL,
                       (float *)cuda_dev_ptr(y), n, eps, m, priv->stream);

    if (!cuda_lazy_d2h_for("norm") && y->host_ptr) {
        if (cudaMemcpyAsync((void *)y->host_ptr, cuda_dev_ptr(y), (size_t)m * (size_t)n * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_op_matmul_batch(backend *self, const buffer *w, uint32_t w_type,
                                          const buffer *x, buffer *y, int n, int k, int m) {

    uint32_t stored = cuda_weight_type_of(w ? cuda_dev_ptr(w) : NULL);
    if (stored != (uint32_t)-1)
        w_type = stored;
    if (!cuda_matmul_type_native(self, w_type)) {

        backend *host = backend_host();
        if (!host || !host->matmul_batch) return ERR_UNSUPPORTED;
        size_t wb = w->size, xb = (size_t)m * (size_t)k * sizeof(float),
               yb = (size_t)m * (size_t)n * sizeof(float);
        void *wh = cuda_host_stage_in(w, wb);
        void *xh = cuda_host_stage_in(x, xb);
        void *yh = malloc(yb ? yb : 1);
        status_code st = ERR_OUT_OF_MEMORY;
        if (wh && xh && yh) {
            buffer whb = cuda_host_buf(wh, wb, host);
            buffer xhb = cuda_host_buf(xh, xb, host);
            buffer yhb = cuda_host_buf(yh, yb, host);
            st = host->matmul_batch(host, &whb, w_type, &xhb, &yhb, n, k, m);
            if (st == OK)
                st = cuda_host_stage_out(y, yh, yb);
        }
        free(wh); free(xh); free(yh);
        return st;
    }
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    const void *w_dev = cuda_dev_ptr((buffer *)w);
    const float *x_dev = (const float *)cuda_dev_ptr((buffer *)x);
    float *y_dev = (float *)cuda_dev_ptr(y);
    switch (w_type) {
        case GGML_TYPE_Q8_0:
            cuda_matmul_batch_q8_0(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_Q4_0:
            cuda_matmul_batch_q4_0(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_Q4_1:
            cuda_matmul_batch_q4_1(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_F32:
            cuda_matmul_batch_f32((const float *)w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_Q4_K:
            cuda_matmul_batch_q4_k(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_Q6_K:
            cuda_matmul_batch_q6_k(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_Q5_K:
            cuda_matmul_batch_q5_k(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_Q5_0:
            cuda_matmul_batch_q5_0(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_Q5_1:
            cuda_matmul_batch_q5_1(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_IQ3_S:
            cuda_matmul_batch_iq3_s(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_IQ4_NL:
            cuda_matmul_batch_iq4_nl(w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_F16:
            cuda_matmul_batch_f16((const uint16_t *)w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        case GGML_TYPE_BF16:
            cuda_matmul_batch_bf16((const uint16_t *)w_dev, x_dev, y_dev, n, k, m, priv->stream);
            break;
        default: {

            backend *host = backend_host();
            if (!host || !host->matmul_batch)
                return ERR_UNSUPPORTED;
            size_t wb = w->size, xb = (size_t)m * (size_t)k * sizeof(float),
                   yb = (size_t)m * (size_t)n * sizeof(float);
            void *wh = cuda_host_stage_in(w, wb);
            void *xh = cuda_host_stage_in(x, xb);
            void *yh = malloc(yb ? yb : 1);
            status_code st = ERR_OUT_OF_MEMORY;
            if (wh && xh && yh) {
                buffer whb = cuda_host_buf(wh, wb, host);
                buffer xhb = cuda_host_buf(xh, xb, host);
                buffer yhb = cuda_host_buf(yh, yb, host);
                st = host->matmul_batch(host, &whb, w_type, &xhb, &yhb, n, k, m);
                if (st == OK)
                    st = cuda_host_stage_out(y, yh, yb);
            }
            free(wh); free(xh); free(yh);
            return st;
        }
    }
    if (y->host_ptr) {
        if (cudaMemcpyAsync((void *)y->host_ptr, cuda_dev_ptr(y), (size_t)m * (size_t)n * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
                                          const buffer *v_cache, buffer *out, int layer,
                                          int pos_start, int n_heads, int n_kv_heads,
                                          int head_dim,                                           int n_ctx, int flash_attn, float scale,
                                          int n_kv_heads_active, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    (void)flash_attn;

    if (n_kv_heads_active > 0)
        n_kv_heads = n_kv_heads_active;
    int row_elems = n_heads * head_dim;
    if (priv->kv_quant_q8) {
        int nb = (int)cuda_kv_q8_nblocks(head_dim);
        size_t lay = cuda_kv_layer_base_bytes(priv, layer);
        size_t stride = (size_t)n_ctx * (size_t)nb * KV_Q8_0_BLOCK_BYTES;
        cuda_attn_batch_q8((const float *)cuda_dev_ptr(q),
                           (const uint8_t *)cuda_dev_ptr(k_cache),
                           (const uint8_t *)cuda_dev_ptr(v_cache),
                           (float *)cuda_dev_ptr(out),
                           n_heads, n_kv_heads, head_dim, lay, stride,
                           pos_start, m, 0, scale, nb, priv->stream);
        if (out->host_ptr) {
            if (cudaMemcpyAsync((void *)out->host_ptr, cuda_dev_ptr(out),
                                (size_t)m * (size_t)row_elems * sizeof(float),
                                cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
                return ERR_INTERNAL;
        }
        return OK;
    }
    size_t lay = cuda_kv_layer_base(priv, layer) * sizeof(uint16_t);
    size_t stride = (size_t)n_ctx * (size_t)head_dim;
    cuda_attn_batch_f16((const float *)cuda_dev_ptr(q),
                        (const uint16_t *)cuda_dev_ptr(k_cache),
                        (const uint16_t *)cuda_dev_ptr(v_cache),
                        (float *)cuda_dev_ptr(out),
                        n_heads, n_kv_heads, head_dim, lay, stride,
                        pos_start, m, 0, scale, priv->stream);
    if (out->host_ptr) {
        if (cudaMemcpyAsync((void *)out->host_ptr, cuda_dev_ptr(out),
                            (size_t)m * (size_t)row_elems * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_attention_swa_batch(backend *self, const buffer *q, const buffer *k_cache,
                                              const buffer *v_cache, buffer *out, int layer,
                                              int pos_start, int n_heads, int n_kv_heads,
                                              int head_dim, int n_ctx, int flash_attn, float scale,
                                              int sliding_window, int n_kv_heads_active, int m) {
    struct cuda_priv *priv = cuda_priv(self);
    if (!priv) return ERR_INTERNAL;
    (void)flash_attn;

    if (n_kv_heads_active > 0)
        n_kv_heads = n_kv_heads_active;
    int row_elems = n_heads * head_dim;
    if (priv->kv_quant_q8) {
        int nb = (int)cuda_kv_q8_nblocks(head_dim);
        size_t lay = cuda_kv_layer_base_bytes(priv, layer);
        size_t stride = (size_t)n_ctx * (size_t)nb * KV_Q8_0_BLOCK_BYTES;
        cuda_attn_batch_q8((const float *)cuda_dev_ptr(q),
                           (const uint8_t *)cuda_dev_ptr(k_cache),
                           (const uint8_t *)cuda_dev_ptr(v_cache),
                           (float *)cuda_dev_ptr(out),
                           n_heads, n_kv_heads, head_dim, lay, stride,
                           pos_start, m, sliding_window, scale, nb, priv->stream);
        if (out->host_ptr) {
            if (cudaMemcpyAsync((void *)out->host_ptr, cuda_dev_ptr(out),
                                (size_t)m * (size_t)row_elems * sizeof(float),
                                cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
                return ERR_INTERNAL;
        }
        return OK;
    }
    size_t lay = cuda_kv_layer_base(priv, layer) * sizeof(uint16_t);
    size_t stride = (size_t)n_ctx * (size_t)head_dim;
    cuda_attn_batch_f16((const float *)cuda_dev_ptr(q),
                        (const uint16_t *)cuda_dev_ptr(k_cache),
                        (const uint16_t *)cuda_dev_ptr(v_cache),
                        (float *)cuda_dev_ptr(out),
                        n_heads, n_kv_heads, head_dim, lay, stride,
                        pos_start, m, sliding_window, scale, priv->stream);
    if (out->host_ptr) {
        if (cudaMemcpyAsync((void *)out->host_ptr, cuda_dev_ptr(out),
                            (size_t)m * (size_t)row_elems * sizeof(float),
                            cudaMemcpyDeviceToHost, priv->stream) != cudaSuccess)
            return ERR_INTERNAL;
    }
    return OK;
}

static status_code cuda_ctor(backend *out) {

    memset(out, 0, sizeof(*out));
    out->name     = "cuda";
    out->priority = 100;
    out->caps     = BCAP_MULTI_MATMUL | BCAP_MATMUL_RESIDUAL | BCAP_MATMUL_FFN_DOWN | BCAP_HOST_VISIBLE_BUFFERS | BCAP_RMSNORM_ADD | BCAP_ROPE_QK_FUSED | BCAP_KV_QUANT_Q8_0;
    out->probe    = cuda_probe;
    out->init     = cuda_init;
    out->free     = cuda_free;

    out->buffer_alloc_weight   = cuda_buffer_alloc_weight;
    out->buffer_alloc_scratch  = cuda_buffer_alloc_scratch;
    out->buffer_free           = cuda_buffer_free;
    out->copy_buffer           = cuda_copy_buffer;
    out->buffer_read_f32       = cuda_buffer_read_f32;
    out->buffer_write_f32      = cuda_buffer_write_f32;

    out->matmul             = cuda_matmul;
    out->matmul_residual    = cuda_matmul_residual;
    out->matmul_ffn_down    = cuda_op_matmul_ffn_down;
    out->matmul_ffn_down_batch = cuda_op_matmul_ffn_down_batch;
    out->matmul_multi       = cuda_matmul_multi;
    out->matmul_type_native = cuda_matmul_type_native;
    out->rmsnorm            = cuda_op_rmsnorm;
    out->rmsnorm_per_head   = cuda_op_rmsnorm_per_head;
    out->rmsnorm_per_head_batch = cuda_op_rmsnorm_per_head_batch;
    out->rmsnorm_noweight   = cuda_op_rmsnorm_noweight;
    out->rmsnorm_noweight_per_head = cuda_op_rmsnorm_noweight_per_head;
    out->rmsnorm_noweight_per_head_batch = cuda_op_rmsnorm_noweight_per_head_batch;
    out->rmsnorm_add        = cuda_op_rmsnorm_add;
    out->rmsnorm_add_batch  = cuda_op_rmsnorm_add_batch;
    out->attention          = cuda_op_attention;
    out->attention_swa      = cuda_op_attention_swa;
    out->add_inplace        = cuda_op_add_inplace;
    out->ffn_activate       = cuda_op_ffn_activate;
    out->ffn_activate_ex    = cuda_op_ffn_activate_ex;
    out->rope               = cuda_op_rope;
    out->rope_qk            = cuda_op_rope_qk;
    out->rope_ext           = cuda_op_rope_ext;
    out->rope_ext_batch     = cuda_op_rope_ext_batch;
    out->embd_lookup        = cuda_op_embd_lookup;
    out->kv_alloc           = cuda_kv_alloc;
    out->kv_free            = cuda_kv_free;
    out->kv_put             = cuda_kv_put;
    out->kv_put_batch       = cuda_kv_put_batch;
    out->rmsnorm_batch      = cuda_op_rmsnorm_batch;
    out->matmul_batch       = cuda_op_matmul_batch;
    out->matmul_multi_batch = cuda_matmul_multi_batch;
    out->add_batch          = cuda_op_add_batch;
    out->ffn_activate_batch = cuda_op_ffn_activate_batch;
    out->ffn_activate_fused_batch = cuda_op_ffn_activate_fused_batch;
    out->rope_batch         = cuda_op_rope_batch;
    out->rope_qk_batch      = cuda_op_rope_qk_batch;
    out->attention_batch    = cuda_attention_batch;
    out->attention_swa_batch = cuda_attention_swa_batch;
    out->scale_inplace      = cuda_op_scale_inplace;
    out->softcap             = cuda_op_softcap;
    out->attn_output_gate    = cuda_op_attn_output_gate;
    out->split_qgate         = cuda_op_split_qgate;
    out->partial_rope_qk    = cuda_op_partial_rope_qk;
    out->moe_activate       = cuda_op_moe_activate;
    out->moe_experts_batch = cuda_op_moe_experts_batch;
    out->ple_combine        = cuda_op_ple_combine;
    out->argmax             = cuda_op_argmax;
    out->synchronize        = cuda_synchronize;

    return OK;
}

BACKEND_REGISTER("cuda", cuda_ctor);
