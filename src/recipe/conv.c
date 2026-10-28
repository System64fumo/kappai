#include "recipe/conv.h"

#define CONV_COL_PIXELS 16384

static void reorder_conv3x3(const gguf_tensor *t, int ic, int oc, char *dst) {
	size_t		esz = ggml_row_size(t->type, 1);
	const char *src = t->data;
	for (int o = 0; o < oc; o++)
		for (int c = 0; c < ic; c++)
			for (int p = 0; p < 9; p++)
				memcpy(dst + (((size_t)o * 9 + p) * ic + c) * esz,
					   src + (((size_t)o * ic + c) * 9 + p) * esz, esz);
}

status_code conv_load(backend *be, const gguf_tensor *w, const gguf_tensor *bias, int row_begin,
					  int rows, conv_layer *c) {
	memset(c, 0, sizeof(*c));
	c->ic = (int)w->dims[w->n_dims - 2];
	c->oc = rows ? rows : (int)w->dims[w->n_dims - 1];
	c->k  = w->n_dims == 4 && w->dims[0] == 3 ? 3 : 1;

	status_code s = tensor_upload_f32(be, bias, row_begin, c->oc, 0.0f, &c->bias);
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

void conv_free(backend *be, conv_layer *c) {
	(void)be;
	weight_free(&c->w);
	buffer_release(&c->bias);
	buffer_release(&c->col_scratch);
	free(c->reordered);
	memset(c, 0, sizeof(*c));
}

status_code conv_forward(backend *be, conv_layer *c, const buffer *in, buffer *out, int H, int W,
						 int shift) {
	status_code s;
	if (c->k == 1) {
		s = be->matmul_batch(be, &c->w.buf, c->w.type, in, out, c->oc, c->ic, H * W);
		return s == OK ? be->add_bias(be, out, &c->bias, H * W, c->oc) : s;
	}
	s = buffer_ensure_scratch(be, &c->col_scratch,
							  (size_t)MIN(CONV_COL_PIXELS, H * W) * 9 * c->ic * sizeof(float));
	if (s != OK)
		return s;
	buffer col = c->col_scratch;
	for (int r = 0; r < H * W && s == OK; r += CONV_COL_PIXELS) {
		int			n	= MIN(CONV_COL_PIXELS, H * W - r);
		im2col_desc d	= {.in			= in,
						   .col			= &col,
						   .first_pixel = r,
						   .n_pixels	= n,
						   .width		= W,
						   .in_height	= H >> shift,
						   .in_width	= W >> shift,
						   .channels	= c->ic,
						   .shift		= shift};
		buffer		dst = buffer_rows(out, r, n, (size_t)c->oc * sizeof(float));
		s				= be->im2col3x3(be, &d);
		if (s == OK)
			s = be->matmul_batch(be, &c->w.buf, c->w.type, &col, &dst, c->oc, 9 * c->ic, n);
		if (s == OK)
			s = be->add_bias(be, &dst, &c->bias, n, c->oc);
	}
	c->col_scratch = col;
	return s;
}
