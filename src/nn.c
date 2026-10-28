#include "nn.h"
#include "log.h"

#include <math.h>

#define ROW_CHUNK 16384

status_code nn_alloc(backend *be, size_t floats, buffer *out) {
	memset(out, 0, sizeof(*out));
	return be->buffer_alloc_scratch(be, floats * sizeof(float), out);
}

status_code nn_upload(backend *be, const float *host, size_t floats, buffer *out) {
	status_code s = nn_alloc(be, floats, out);
	if (s == OK)
		s = be->buffer_write_f32(be, out, host, (int)floats);
	if (s != OK)
		nn_free(be, out);
	return s;
}

status_code nn_vec_load(backend *be, const gguf_tensor *t, float offset, buffer *out) {
	size_t n = 1;
	for (uint32_t d = 0; d < t->n_dims; d++)
		n *= t->dims[d];
	float *host = tensor_to_f32(be, t);
	if (!host)
		return ERR_FORMAT;
	for (size_t i = 0; i < n; i++)
		host[i] += offset;
	status_code s = nn_upload(be, host, n, out);
	free(host);
	return s;
}

void nn_free(backend *be, buffer *b) {
	if (b->owner)
		be->buffer_free(be, b);
	memset(b, 0, sizeof(*b));
}

status_code nn_matmul(backend *be, const weight *w, const buffer *x, buffer *y, int n, int k,
					  int m) {
	for (int r = 0; r < m; r += ROW_CHUNK) {
		int	   rows = MIN(ROW_CHUNK, m - r);
		buffer xs =
			buffer_slice(x, (size_t)r * k * sizeof(float), (size_t)rows * k * sizeof(float));
		buffer ys =
			buffer_slice(y, (size_t)r * n * sizeof(float), (size_t)rows * n * sizeof(float));
		NN_TRY(be->matmul_batch(be, &w->buf, w->type, &xs, &ys, n, k, rows));
	}
	return OK;
}

status_code nn_rmsnorm(backend *be, const buffer *x, const buffer *w, buffer *y, int dim, float eps,
					   int rows) {
	for (int r = 0; r < rows; r += ROW_CHUNK) {
		int	   n = MIN(ROW_CHUNK, rows - r);
		buffer xs =
			buffer_slice(x, (size_t)r * dim * sizeof(float), (size_t)n * dim * sizeof(float));
		buffer ys =
			buffer_slice(y, (size_t)r * dim * sizeof(float), (size_t)n * dim * sizeof(float));
		NN_TRY(be->rmsnorm_batch(be, &xs, w, &ys, dim, eps, n));
	}
	return OK;
}

status_code nn_add(backend *be, buffer *x, const buffer *y, int dim, int rows) {
	for (int r = 0; r < rows; r += ROW_CHUNK) {
		int	   n = MIN(ROW_CHUNK, rows - r);
		buffer xs =
			buffer_slice(x, (size_t)r * dim * sizeof(float), (size_t)n * dim * sizeof(float));
		buffer ys =
			buffer_slice(y, (size_t)r * dim * sizeof(float), (size_t)n * dim * sizeof(float));
		NN_TRY(be->add_batch(be, &xs, &ys, dim, n));
	}
	return OK;
}

void nn_rope_table(float *table, const int *pos, int rows, const int axes[3], float theta) {
	int half = (axes[0] + axes[1] + axes[2]) / 2;
	for (int r = 0; r < rows; r++) {
		float *cs  = table + (size_t)r * half * 2;
		int	   idx = 0;
		for (int a = 0; a < 3; a++)
			for (int j = 0; j < axes[a] / 2; j++, idx++) {
				float omega		= 1.0f / powf(theta, (float)(2 * j) / (float)axes[a]);
				float angle		= (float)pos[r * 3 + a] * omega;
				cs[idx * 2]		= cosf(angle);
				cs[idx * 2 + 1] = sinf(angle);
			}
	}
}

static void reorder_conv3x3(const gguf_tensor *t, int ic, int oc, char *dst) {
	size_t		esz = ggml_row_size(t->type, 1);
	const char *src = t->data;
	for (int o = 0; o < oc; o++)
		for (int c = 0; c < ic; c++)
			for (int p = 0; p < 9; p++)
				memcpy(dst + (((size_t)o * 9 + p) * ic + c) * esz,
					   src + (((size_t)o * ic + c) * 9 + p) * esz, esz);
}

status_code nn_conv_load(backend *be, const gguf_tensor *w, const gguf_tensor *bias, int row_begin,
						 int rows, nn_conv *c) {
	memset(c, 0, sizeof(*c));
	c->ic = (int)w->dims[w->n_dims - 2];
	c->oc = rows ? rows : (int)w->dims[w->n_dims - 1];
	c->k  = w->n_dims == 4 && w->dims[0] == 3 ? 3 : 1;

	float *host = tensor_to_f32(be, bias);
	if (!host)
		return ERR_FORMAT;
	status_code s = nn_upload(be, host + row_begin, (size_t)c->oc, &c->bias);
	free(host);
	if (s != OK)
		return s;

	if (c->k == 1)
		return weight_upload(
			be, w->type, c->ic, c->oc,
			(const char *)w->data + (size_t)row_begin * ggml_row_size(w->type, c->ic), &c->w);
	size_t esz	 = ggml_row_size(w->type, 1);
	c->reordered = xmalloc_aligned((size_t)c->oc * 9 * c->ic * esz, 64);
	reorder_conv3x3(w, c->ic, c->oc, c->reordered);
	return weight_upload(be, w->type, (size_t)c->ic * 9, c->oc, c->reordered, &c->w);
}

void nn_conv_free(backend *be, nn_conv *c) {
	if (c->w.buf.owner)
		weight_free(be, &c->w);
	nn_free(be, &c->bias);
	free(c->reordered);
	memset(c, 0, sizeof(*c));
}

status_code nn_conv_forward(backend *be, const nn_conv *c, const buffer *in, buffer *out, int H,
							int W, int shift) {
	if (c->k == 1) {
		NN_TRY(nn_matmul(be, &c->w, in, out, c->oc, c->ic, H * W));
		for (int r = 0; r < H * W; r += ROW_CHUNK) {
			int	   n  = MIN(ROW_CHUNK, H * W - r);
			buffer os = buffer_slice(out, (size_t)r * c->oc * sizeof(float),
									 (size_t)n * c->oc * sizeof(float));
			NN_TRY(be->add_bias(be, &os, &c->bias, n, c->oc));
		}
		return OK;
	}
	buffer col;
	NN_TRY(nn_alloc(be, (size_t)MIN(ROW_CHUNK, H * W) * 9 * c->ic, &col));
	status_code s = OK;
	for (int r = 0; r < H * W && s == OK; r += ROW_CHUNK) {
		int			n = MIN(ROW_CHUNK, H * W - r);
		im2col_desc d = {.in		  = in,
						 .col		  = &col,
						 .first_pixel = r,
						 .n_pixels	  = n,
						 .width		  = W,
						 .in_height	  = H >> shift,
						 .in_width	  = W >> shift,
						 .channels	  = c->ic,
						 .shift		  = shift};
		buffer		dst =
			buffer_slice(out, (size_t)r * c->oc * sizeof(float), (size_t)n * c->oc * sizeof(float));
		s = be->im2col3x3(be, &d);
		if (s == OK)
			s = nn_matmul(be, &c->w, &col, &dst, c->oc, 9 * c->ic, n);
		if (s == OK)
			s = be->add_bias(be, &dst, &c->bias, n, c->oc);
	}
	nn_free(be, &col);
	return s;
}
