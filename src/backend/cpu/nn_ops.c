#include "backend/cpu/common.h"
#include <math.h>
#include <string.h>

#define ACT_GELU 1

typedef struct {
	const float *x, *vec, *src;
	float		*y;
	int			 dim, heads, head_dim, activation;
	float		 eps;
} row_job;

static float activate_one(int activation, float v) {
	if (activation == ACT_GELU)
		return gelu_tanh(v);
	return silu(v);
}

static void layernorm_modulate_rows(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const row_job *j = ctx;
	for (int r = begin; r < end; r++) {
		const float *x	 = j->x + (size_t)r * j->dim;
		float		*y	 = j->y + (size_t)r * j->dim;
		double		 sum = 0;
		for (int i = 0; i < j->dim; i++)
			sum += x[i];
		float  mean = (float)(sum / j->dim);
		double sq	= 0;
		for (int i = 0; i < j->dim; i++)
			sq += (double)(x[i] - mean) * (x[i] - mean);
		float inv = 1.0f / sqrtf((float)(sq / j->dim) + j->eps);
		for (int i = 0; i < j->dim; i++)
			y[i] = (x[i] - mean) * inv * (1.0f + j->vec[i]);
	}
}

status_code cpu_layernorm_modulate(backend *self, const buffer *x, const buffer *scale, buffer *y,
								   int rows, int dim, float eps) {
	row_job j = {.x = cpu_ptr(x), .vec = cpu_ptr(scale), .y = cpu_ptr(y), .dim = dim, .eps = eps};
	cpu_run_batch(cpu_get_pool(self), rows, layernorm_modulate_rows, &j);
	return OK;
}

static void gated_add_rows(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const row_job *j = ctx;
	for (int r = begin; r < end; r++)
		for (int i = 0; i < j->dim; i++)
			j->y[(size_t)r * j->dim + i] += j->src[(size_t)r * j->dim + i] * tanhf(j->vec[i]);
}

status_code cpu_gated_add(backend *self, buffer *x, const buffer *y, const buffer *gate, int rows,
						  int dim) {
	row_job j = {.y = cpu_ptr(x), .src = cpu_ptr(y), .vec = cpu_ptr(gate), .dim = dim};
	cpu_run_batch(cpu_get_pool(self), rows, gated_add_rows, &j);
	return OK;
}

static void add_bias_rows(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const row_job *j = ctx;
	for (int r = begin; r < end; r++)
		for (int i = 0; i < j->dim; i++)
			j->y[(size_t)r * j->dim + i] += j->vec[i];
}

status_code cpu_add_bias(backend *self, buffer *x, const buffer *bias, int rows, int dim) {
	row_job j = {.y = cpu_ptr(x), .vec = cpu_ptr(bias), .dim = dim};
	cpu_run_batch(cpu_get_pool(self), rows, add_bias_rows, &j);
	return OK;
}

static void activate_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const row_job *j = ctx;
	for (int i = begin; i < end; i++)
		j->y[i] = activate_one(j->activation, j->y[i]);
}

status_code cpu_activate(backend *self, buffer *x, int n, int activation) {
	row_job j = {.y = cpu_ptr(x), .activation = activation};
	cpu_run_batch_full(cpu_get_pool(self), n, 1024, 2048, activate_chunk, &j);
	return OK;
}

static void rope_table_rows(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const row_job *j	= ctx;
	int			   half = j->head_dim / 2;
	for (int r = begin; r < end; r++) {
		const float *cs = j->vec + (size_t)r * half * 2;
		rope_rotate_rows(j->y + (size_t)r * j->heads * j->head_dim, j->heads, j->head_dim, half, cs,
						 0);
	}
}

status_code cpu_rope_table(backend *self, buffer *x, const buffer *table, int rows, int heads,
						   int head_dim) {
	row_job j = {.y = cpu_ptr(x), .vec = cpu_ptr(table), .heads = heads, .head_dim = head_dim};
	cpu_run_batch(cpu_get_pool(self), rows, rope_table_rows, &j);
	return OK;
}

typedef struct {
	const dense_attention_desc *d;
	const float				   *q, *k0, *v0, *k1, *v1;
	float					   *out;
} attention_job;

static float dot(const float *a, const float *b, int n) {
	float s = 0;
	for (int i = 0; i < n; i++)
		s += a[i] * b[i];
	return s;
}

static void attention_items(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const attention_job		   *j	   = ctx;
	const dense_attention_desc *d	   = j->d;
	size_t						stride = (size_t)d->n_heads * d->head_dim;
	int							total  = d->n0 + d->n1;
	float						scale  = 1.0f / sqrtf((float)d->head_dim);
	float					   *scores = xmalloc((size_t)total * sizeof(float));
	for (int it = begin; it < end; it++) {
		int			 qi	  = it / d->n_heads;
		int			 h	  = it % d->n_heads;
		const float *q	  = j->q + (size_t)qi * stride + (size_t)h * d->head_dim;
		float		*o	  = j->out + (size_t)qi * stride + (size_t)h * d->head_dim;
		int			 vis  = d->causal ? qi + 1 : total;
		float		 maxv = -INFINITY;
		for (int c = 0; c < vis; c++) {
			const float *k =
				(c < d->n0 ? j->k0 + (size_t)c * stride : j->k1 + (size_t)(c - d->n0) * stride) +
				(size_t)h * d->head_dim;
			scores[c] = dot(q, k, d->head_dim) * scale;
			maxv	  = MAX(maxv, scores[c]);
		}
		float sum = 0;
		for (int c = 0; c < vis; c++) {
			scores[c] = expf(scores[c] - maxv);
			sum += scores[c];
		}
		memset(o, 0, (size_t)d->head_dim * sizeof(float));
		for (int c = 0; c < vis; c++) {
			const float *v =
				(c < d->n0 ? j->v0 + (size_t)c * stride : j->v1 + (size_t)(c - d->n0) * stride) +
				(size_t)h * d->head_dim;
			float w = scores[c] / sum;
			for (int e = 0; e < d->head_dim; e++)
				o[e] += w * v[e];
		}
	}
	free(scores);
}

status_code cpu_attention_dense(backend *self, const dense_attention_desc *d) {
	attention_job j = {.d	= d,
					   .q	= cpu_ptr(d->q),
					   .k0	= cpu_ptr(d->k0),
					   .v0	= cpu_ptr(d->v0),
					   .k1	= d->n1 ? cpu_ptr(d->k1) : NULL,
					   .v1	= d->n1 ? cpu_ptr(d->v1) : NULL,
					   .out = cpu_ptr(d->out)};
	cpu_run_batch(cpu_get_pool(self), d->n_q * d->n_heads, attention_items, &j);
	return OK;
}

typedef struct {
	const im2col_desc *d;
	const float		  *in;
	float			  *col;
} im2col_job;

static void im2col_rows(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const im2col_job  *j = ctx;
	const im2col_desc *d = j->d;
	for (int r = begin; r < end; r++) {
		int	   pixel = d->first_pixel + r;
		int	   y	 = pixel / d->width;
		int	   x	 = pixel % d->width;
		float *dst	 = j->col + (size_t)r * 9 * d->channels;
		for (int ky = 0; ky < 3; ky++)
			for (int kx = 0; kx < 3; kx++, dst += d->channels) {
				int sy = y + ky - 1, sx = x + kx - 1;
				int iy = sy >> d->shift, ix = sx >> d->shift;
				if (sy < 0 || sx < 0 || iy >= d->in_height || ix >= d->in_width)
					memset(dst, 0, (size_t)d->channels * sizeof(float));
				else
					memcpy(dst, j->in + ((size_t)iy * d->in_width + ix) * d->channels,
						   (size_t)d->channels * sizeof(float));
			}
	}
}

status_code cpu_im2col3x3(backend *self, const im2col_desc *d) {
	im2col_job j = {.d = d, .in = cpu_ptr(d->in), .col = cpu_ptr(d->col)};
	cpu_run_batch_full(cpu_get_pool(self), d->n_pixels, 64, 2, im2col_rows, &j);
	return OK;
}

typedef struct {
	const upsample_shortcut_desc *d;
	const float					 *in;
	float						 *out;
} shortcut_job;

static void upsample_shortcut_rows(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const shortcut_job			 *j			= ctx;
	const upsample_shortcut_desc *d			= j->d;
	int							  f			= d->factor;
	int							  out_width = d->width * f;
	for (int r = begin; r < end; r++) {
		int			 Y = r / out_width, X = r % out_width;
		int			 y = Y / f, b = Y % f, x = X / f, e = X % f;
		const float *src = j->in + ((size_t)y * d->width + x) * d->in_channels;
		float		*dst = j->out + (size_t)r * d->out_channels;
		for (int o = 0; o < d->out_channels; o++) {
			int group = ((o * d->frames_up + (d->frames_up - 1)) * f + b) * f + e;
			dst[o] += src[group / d->repeats];
		}
	}
}

status_code cpu_upsample_shortcut_add(backend *self, const upsample_shortcut_desc *d) {
	shortcut_job j = {.d = d, .in = cpu_ptr(d->in), .out = cpu_ptr(d->out)};
	cpu_run_batch_full(cpu_get_pool(self), d->height * d->factor * d->width * d->factor, 64, 2,
					   upsample_shortcut_rows, &j);
	return OK;
}
