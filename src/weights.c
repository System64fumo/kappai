#include "weights.h"
#include "log.h"
#include "model.h"

#define WEIGHT_ALIGN 64

static size_t tensor_rows(const gguf_tensor *t) {
	size_t rows = 1;
	for (uint32_t d = 1; d < t->n_dims; d++)
		rows *= t->dims[d];
	return rows;
}

static size_t tensor_elems(const gguf_tensor *t) {
	return (size_t)t->dims[0] * tensor_rows(t);
}

status_code weight_upload(backend *be, uint32_t type, size_t k, size_t rows, const void *data,
						  weight *out) {
	memset(out, 0, sizeof(*out));
	uint32_t re_type = 0;
	if (model_should_repack(type, NULL) && be->repack_plan)
		be->repack_plan(be, type, k, rows, &re_type);

	void *host = NULL;
	if (re_type) {
		host		  = xmalloc_aligned(ggml_row_size(re_type, k) * rows, WEIGHT_ALIGN);
		status_code s = be->repack_weight(be, type, data, host, (int)rows, (int)k);
		if (s != OK) {
			free(host);
			return s;
		}
		type = re_type;
	} else if ((uintptr_t)data % WEIGHT_ALIGN) {
		size_t bytes = ggml_row_size(type, k) * rows;
		host		 = xmalloc_aligned(bytes, WEIGHT_ALIGN);
		memcpy(host, data, bytes);
	}

	tensor_desc desc = {
		.type	   = type,
		.n_dims	   = 2,
		.dims	   = {k, rows, 0, 0},
		.host_data = host ? host : data,
	};
	status_code s = be->buffer_alloc_weight(be, &desc, &out->buf);
	if (s != OK) {
		free(host);
		return s;
	}
	out->type  = type;
	out->owned = host;
	return OK;
}

status_code weight_load_rows(backend *be, const gguf_tensor *t, size_t row_begin, size_t rows,
							 weight *out) {
	const char *data = (const char *)t->data + row_begin * ggml_row_size(t->type, t->dims[0]);
	return weight_upload(be, t->type, t->dims[0], rows, data, out);
}

status_code weight_load(backend *be, const gguf_tensor *t, weight *out) {
	return weight_load_rows(be, t, 0, tensor_rows(t), out);
}

void weight_free(weight *w) {
	buffer_release(&w->buf);
	free(w->owned);
	memset(w, 0, sizeof(*w));
}

float *tensor_to_f32(backend *be, const gguf_tensor *t) {
	if (!backend_has_cap(be, BCAP_IS_HOST))
		be = backend_host();
	size_t n   = tensor_elems(t);
	float *dst = xmalloc(n * sizeof(float));
	if (t->type == GGML_TYPE_F32)
		memcpy(dst, t->data, n * sizeof(float));
	else if (be->dequant_row(be, t->type, t->data, (int)n, dst) != OK) {
		free(dst);
		return NULL;
	}
	return dst;
}

status_code tensor_upload_f32(backend *be, const gguf_tensor *t, size_t first, size_t count,
							  float offset, buffer *out) {
	float *host = tensor_to_f32(be, t);
	if (!host)
		return ERR_FORMAT;
	if (!count)
		count = tensor_elems(t);
	for (size_t i = 0; i < count; i++)
		host[first + i] += offset;
	status_code s = buffer_upload_f32(be, host + first, count, out);
	free(host);
	return s;
}
