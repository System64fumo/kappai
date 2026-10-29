#ifndef RECIPE_CONV_H
#define RECIPE_CONV_H

#include "backend/backend.h"
#include "weights.h"

typedef struct {
	weight w;
	buffer bias;
	void  *reordered;
	buffer col_scratch;
	int	   ic, oc, k;
} conv_layer;

status_code conv_load(backend *be, const gguf_tensor *weight, const gguf_tensor *bias,
					  int row_begin, int rows, conv_layer *c);
void		conv_free(backend *be, conv_layer *c);
status_code conv_forward(backend *be, conv_layer *c, const buffer *in, buffer *out, int H, int W,
						 int upsample_shift);

#endif
