#ifndef DIFFUSION_NN_H
#define DIFFUSION_NN_H

#include "backend/backend.h"
#include "weights.h"

typedef enum {
	NN_SILU = 0,
	NN_GELU = 1,
} nn_activation;

typedef struct {
	weight w;
	buffer bias;
	void  *reordered;
	int	   ic, oc, k;
} nn_conv;

#define NN_TRY(expr)                                                                               \
	do {                                                                                           \
		status_code nn_try_status_ = (expr);                                                       \
		if (nn_try_status_ != OK)                                                                  \
			return nn_try_status_;                                                                 \
	} while (0)

#define NN_GOTO(expr)                                                                              \
	do {                                                                                           \
		s = (expr);                                                                                \
		if (s != OK)                                                                               \
			goto done;                                                                             \
	} while (0)

bool nn_supported(backend *be);

status_code nn_alloc(backend *be, size_t floats, buffer *out);
status_code nn_upload(backend *be, const float *host, size_t floats, buffer *out);
status_code nn_vec_load(backend *be, const gguf_tensor *t, float offset, buffer *out);
void		nn_free(backend *be, buffer *b);

status_code nn_matmul(backend *be, const weight *w, const buffer *x, buffer *y, int n, int k,
					  int m);
status_code nn_rmsnorm(backend *be, const buffer *x, const buffer *w, buffer *y, int dim, float eps,
					   int rows);
status_code nn_add(backend *be, buffer *x, const buffer *y, int dim, int rows);

void nn_rope_table(float *table, const int *pos, int rows, const int axes[3], float theta);

status_code nn_conv_load(backend *be, const gguf_tensor *weight, const gguf_tensor *bias,
						 int row_begin, int rows, nn_conv *c);
void		nn_conv_free(backend *be, nn_conv *c);
status_code nn_conv_forward(backend *be, const nn_conv *c, const buffer *in, buffer *out, int H,
							int W, int upsample_shift);

#endif
