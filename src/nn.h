#ifndef DIFFUSION_NN_H
#define DIFFUSION_NN_H

#include "backend/backend.h"
#include "weights.h"

typedef enum {
	NN_SILU,
	NN_GELU,
} nn_activation;

typedef struct {
	const float *k0, *v0;
	int			 n0;
	const float *k1, *v1;
	int			 n1;
	bool		 causal;
} nn_kv;

typedef struct {
	weight w;
	float *bias;
	void  *reordered;
	int	   ic, oc, k;
} nn_conv;

tpool *nn_pool(backend *be);

void nn_matmul(backend *be, const weight *w, const float *x, float *y, int n, int k, int m);
void nn_rmsnorm(backend *be, const float *x, const float *w, float *y, int dim, float eps,
				int rows);
void nn_rmsnorm_per_head(backend *be, const float *x, const float *w, float *y, int heads,
						 int head_dim, float eps, int rows);
void nn_add(backend *be, float *x, const float *y, int dim, int rows);

void nn_layernorm_modulate(backend *be, const float *x, float *y, int rows, int dim,
						   const float *scale);
void nn_gated_add(backend *be, float *x, const float *y, int rows, int dim, const float *gate);
void nn_activate(backend *be, nn_activation act, float *x, size_t n);
void nn_silu_mul(backend *be, const float *gate_up, float *out, int rows, int n);

void nn_rope_table(float *table, const int *pos, int rows, const int axes[3], float theta);
void nn_rope_apply(backend *be, float *x, int rows, int heads, int head_dim, const float *table);

void nn_attention(backend *be, float *out, const float *q, int nq, int heads, int head_dim,
				  const nn_kv *kv);

void nn_add_bias(backend *be, float *x, const float *bias, int dim, int rows);

status_code nn_conv_load(backend *be, const gguf_tensor *weight, const gguf_tensor *bias,
						 nn_conv *c);
void		nn_conv_free(backend *be, nn_conv *c);
void nn_conv_forward(backend *be, const nn_conv *c, const float *in, float *out, int H, int W,
					 int upsample_shift);

#endif
