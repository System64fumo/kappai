#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "backend/backend.h"
#include "gguf.h"

typedef struct {
	buffer	 buf;
	uint32_t type;
	void	*owned;
} weight;

status_code weight_upload(backend *be, uint32_t type, size_t k, size_t rows, const void *data,
						  weight *out);
status_code weight_load(backend *be, const gguf_tensor *t, weight *out);
status_code weight_load_rows(backend *be, const gguf_tensor *t, size_t row_begin, size_t rows,
							 weight *out);
void		weight_free(weight *w);

float	   *tensor_to_f32(backend *be, const gguf_tensor *t);
status_code tensor_upload_f32(backend *be, const gguf_tensor *t, size_t first, size_t count,
							  float offset, buffer *out);

#endif
