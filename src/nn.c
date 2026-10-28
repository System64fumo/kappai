#include "nn.h"

#include <math.h>

tpool *nn_pool(backend *be) {
	tpool *pool = backend_get_pool(be);
	return pool ? pool : tpool_fetch_pool();
}

void nn_matmul(backend *be, const weight *w, const float *x, float *y, int n, int k, int m) {
	buffer xb = buffer_host_view(be, (void *)x, (size_t)m * k * sizeof(float));
	buffer yb = buffer_host_view(be, y, (size_t)m * n * sizeof(float));
	be->matmul_batch(be, &w->buf, w->type, &xb, &yb, n, k, m);
	ensure_sync(be);
}

void nn_rmsnorm(backend *be, const float *x, const float *w, float *y, int dim, float eps,
				int rows) {
	buffer xb = buffer_host_view(be, (void *)x, (size_t)rows * dim * sizeof(float));
	buffer wb = buffer_host_view(be, (void *)w, (size_t)dim * sizeof(float));
	buffer yb = buffer_host_view(be, y, (size_t)rows * dim * sizeof(float));
	be->rmsnorm_batch(be, &xb, &wb, &yb, dim, eps, rows);
	ensure_sync(be);
}

void nn_rmsnorm_per_head(backend *be, const float *x, const float *w, float *y, int heads,
						 int head_dim, float eps, int rows) {
	size_t bytes = (size_t)rows * heads * head_dim * sizeof(float);
	buffer xb	 = buffer_host_view(be, (void *)x, bytes);
	buffer wb	 = buffer_host_view(be, (void *)w, (size_t)head_dim * sizeof(float));
	buffer yb	 = buffer_host_view(be, y, bytes);
	be->rmsnorm_per_head_batch(be, &xb, &wb, &yb, heads, head_dim, eps, rows);
	ensure_sync(be);
}

void nn_add(backend *be, float *x, const float *y, int dim, int rows) {
	size_t bytes = (size_t)rows * dim * sizeof(float);
	buffer xb	 = buffer_host_view(be, x, bytes);
	buffer yb	 = buffer_host_view(be, (void *)y, bytes);
	be->add_batch(be, &xb, &yb, dim, rows);
	ensure_sync(be);
}

typedef struct {
	const float *x;
	float		*y;
	int			 dim;
	const float *vec;
} row_ctx;

static void layernorm_modulate_rows(int begin, int end, int tid, void *p) {
	(void)tid;
	const row_ctx *c = p;
	for (int r = begin; r < end; r++) {
		const float *x	 = c->x + (size_t)r * c->dim;
		float		*y	 = c->y + (size_t)r * c->dim;
		double		 sum = 0, sq = 0;
		for (int i = 0; i < c->dim; i++) {
			sum += x[i];
			sq += (double)x[i] * x[i];
		}
		float mean = (float)(sum / c->dim);
		float var  = (float)(sq / c->dim) - mean * mean;
		float inv  = 1.0f / sqrtf(var + 1e-6f);
		for (int i = 0; i < c->dim; i++)
			y[i] = (x[i] - mean) * inv * (1.0f + c->vec[i]);
	}
}

void nn_layernorm_modulate(backend *be, const float *x, float *y, int rows, int dim,
						   const float *scale) {
	row_ctx c = {x, y, dim, scale};
	tpool_parallel_for(nn_pool(be), rows, 1, layernorm_modulate_rows, &c);
}

typedef struct {
	float		*x;
	const float *y;
	int			 dim;
	const float *gate;
} gated_ctx;

static void gated_add_rows(int begin, int end, int tid, void *p) {
	(void)tid;
	const gated_ctx *c = p;
	for (int r = begin; r < end; r++)
		for (int i = 0; i < c->dim; i++)
			c->x[(size_t)r * c->dim + i] += c->y[(size_t)r * c->dim + i] * tanhf(c->gate[i]);
}

void nn_gated_add(backend *be, float *x, const float *y, int rows, int dim, const float *gate) {
	gated_ctx c = {x, y, dim, gate};
	tpool_parallel_for(nn_pool(be), rows, 1, gated_add_rows, &c);
}

typedef struct {
	float		 *x;
	nn_activation act;
} act_ctx;

static float activate_one(nn_activation act, float v) {
	if (act == NN_SILU)
		return v / (1.0f + expf(-v));
	const float k = 0.7978845608f;
	return 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
}

static void activate_chunk(int begin, int end, int tid, void *p) {
	(void)tid;
	const act_ctx *c = p;
	for (int i = begin; i < end; i++)
		c->x[i] = activate_one(c->act, c->x[i]);
}

void nn_activate(backend *be, nn_activation act, float *x, size_t n) {
	act_ctx c = {x, act};
	tpool_parallel_for(nn_pool(be), (int)n, 1024, activate_chunk, &c);
}

typedef struct {
	const float *gate_up;
	float		*out;
	int			 n;
} silu_mul_ctx;

static void silu_mul_rows(int begin, int end, int tid, void *p) {
	(void)tid;
	const silu_mul_ctx *c = p;
	for (int r = begin; r < end; r++) {
		const float *gate = c->gate_up + (size_t)r * 2 * c->n;
		const float *up	  = gate + c->n;
		float		*out  = c->out + (size_t)r * c->n;
		for (int i = 0; i < c->n; i++)
			out[i] = activate_one(NN_SILU, gate[i]) * up[i];
	}
}

void nn_silu_mul(backend *be, const float *gate_up, float *out, int rows, int n) {
	silu_mul_ctx c = {gate_up, out, n};
	tpool_parallel_for(nn_pool(be), rows, 1, silu_mul_rows, &c);
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

typedef struct {
	float		*x;
	int			 heads, head_dim;
	const float *table;
} rope_ctx;

static void rope_rows(int begin, int end, int tid, void *p) {
	(void)tid;
	const rope_ctx *c	 = p;
	int				half = c->head_dim / 2;
	for (int r = begin; r < end; r++) {
		const float *cs = c->table + (size_t)r * half * 2;
		for (int h = 0; h < c->heads; h++) {
			float *v = c->x + ((size_t)r * c->heads + h) * c->head_dim;
			for (int i = 0; i < half; i++) {
				float x0 = v[2 * i], x1 = v[2 * i + 1];
				float co = cs[i * 2], si = cs[i * 2 + 1];
				v[2 * i]	 = x0 * co - x1 * si;
				v[2 * i + 1] = x0 * si + x1 * co;
			}
		}
	}
}

void nn_rope_apply(backend *be, float *x, int rows, int heads, int head_dim, const float *table) {
	rope_ctx c = {x, heads, head_dim, table};
	tpool_parallel_for(nn_pool(be), rows, 1, rope_rows, &c);
}

typedef struct {
	float		*out;
	const float *q;
	int			 heads, head_dim;
	const nn_kv *kv;
} attn_ctx;

static float dot(const float *a, const float *b, int n) {
	float s = 0;
	for (int i = 0; i < n; i++)
		s += a[i] * b[i];
	return s;
}

static const float *kv_row(const float *p0, int n0, const float *p1, int j, size_t stride) {
	return j < n0 ? p0 + (size_t)j * stride : p1 + (size_t)(j - n0) * stride;
}

static void attention_items(int begin, int end, int tid, void *p) {
	(void)tid;
	const attn_ctx *c	   = p;
	const nn_kv	   *kv	   = c->kv;
	size_t			stride = (size_t)c->heads * c->head_dim;
	int				total  = kv->n0 + kv->n1;
	float			scale  = 1.0f / sqrtf((float)c->head_dim);
	float		   *scores = xmalloc((size_t)total * sizeof(float));
	for (int it = begin; it < end; it++) {
		int			 qi	  = it / c->heads;
		int			 h	  = it % c->heads;
		const float *q	  = c->q + (size_t)qi * stride + (size_t)h * c->head_dim;
		float		*o	  = c->out + (size_t)qi * stride + (size_t)h * c->head_dim;
		int			 vis  = kv->causal ? qi + 1 : total;
		float		 maxv = -INFINITY;
		for (int j = 0; j < vis; j++) {
			const float *k = kv_row(kv->k0, kv->n0, kv->k1, j, stride) + (size_t)h * c->head_dim;
			scores[j]	   = dot(q, k, c->head_dim) * scale;
			maxv		   = MAX(maxv, scores[j]);
		}
		float sum = 0;
		for (int j = 0; j < vis; j++) {
			scores[j] = expf(scores[j] - maxv);
			sum += scores[j];
		}
		memset(o, 0, (size_t)c->head_dim * sizeof(float));
		for (int j = 0; j < vis; j++) {
			const float *v = kv_row(kv->v0, kv->n0, kv->v1, j, stride) + (size_t)h * c->head_dim;
			float		 w = scores[j] / sum;
			for (int d = 0; d < c->head_dim; d++)
				o[d] += w * v[d];
		}
	}
	free(scores);
}

void nn_attention(backend *be, float *out, const float *q, int nq, int heads, int head_dim,
				  const nn_kv *kv) {
	attn_ctx c = {out, q, heads, head_dim, kv};
	tpool_parallel_for(nn_pool(be), nq * heads, 1, attention_items, &c);
}

typedef struct {
	float		*x;
	const float *bias;
	int			 dim;
} bias_ctx;

static void add_bias_rows(int begin, int end, int tid, void *p) {
	(void)tid;
	const bias_ctx *c = p;
	for (int r = begin; r < end; r++)
		for (int i = 0; i < c->dim; i++)
			c->x[(size_t)r * c->dim + i] += c->bias[i];
}

void nn_add_bias(backend *be, float *x, const float *bias, int dim, int rows) {
	bias_ctx c = {x, bias, dim};
	tpool_parallel_for(nn_pool(be), rows, 16, add_bias_rows, &c);
}

#define IM2COL_MAX_FLOATS ((size_t)32 << 20)

static void reorder_conv3x3(const gguf_tensor *t, int ic, int oc, char *dst) {
	size_t		esz = ggml_row_size(t->type, 1);
	const char *src = t->data;
	for (int o = 0; o < oc; o++)
		for (int c = 0; c < ic; c++)
			for (int p = 0; p < 9; p++)
				memcpy(dst + (((size_t)o * 9 + p) * ic + c) * esz,
					   src + (((size_t)o * ic + c) * 9 + p) * esz, esz);
}

status_code nn_conv_load(backend *be, const gguf_tensor *w, const gguf_tensor *bias, nn_conv *c) {
	memset(c, 0, sizeof(*c));
	c->ic	= (int)w->dims[w->n_dims - 2];
	c->oc	= (int)w->dims[w->n_dims - 1];
	c->k	= w->n_dims == 4 && w->dims[0] == 3 ? 3 : 1;
	c->bias = tensor_to_f32(be, bias);
	if (!c->bias)
		return ERR_FORMAT;
	if (c->k == 1)
		return weight_upload(be, w->type, c->ic, c->oc, w->data, &c->w);
	c->reordered = xmalloc_aligned((size_t)c->oc * 9 * c->ic * ggml_row_size(w->type, 1), 64);
	reorder_conv3x3(w, c->ic, c->oc, c->reordered);
	return weight_upload(be, w->type, (size_t)c->ic * 9, c->oc, c->reordered, &c->w);
}

void nn_conv_free(backend *be, nn_conv *c) {
	if (c->w.buf.owner)
		weight_free(be, &c->w);
	free(c->bias);
	free(c->reordered);
	memset(c, 0, sizeof(*c));
}

typedef struct {
	float		*col;
	const float *in;
	int			 y0, W, in_W, in_H, ic, shift;
} im2col_ctx;

static void im2col_rows(int begin, int end, int tid, void *p) {
	(void)tid;
	const im2col_ctx *c = p;
	for (int r = begin; r < end; r++) {
		int	   y   = c->y0 + r / c->W;
		int	   x   = r % c->W;
		float *dst = c->col + (size_t)r * 9 * c->ic;
		for (int ky = 0; ky < 3; ky++)
			for (int kx = 0; kx < 3; kx++, dst += c->ic) {
				int sy = y + ky - 1, sx = x + kx - 1;
				int iy = sy >> c->shift, ix = sx >> c->shift;
				if (sy < 0 || sx < 0 || iy >= c->in_H || ix >= c->in_W)
					memset(dst, 0, (size_t)c->ic * sizeof(float));
				else
					memcpy(dst, c->in + ((size_t)iy * c->in_W + ix) * c->ic,
						   (size_t)c->ic * sizeof(float));
			}
	}
}

void nn_conv_forward(backend *be, const nn_conv *c, const float *in, float *out, int H, int W,
					 int shift) {
	if (c->k == 1) {
		nn_matmul(be, &c->w, in, out, c->oc, c->ic, H * W);
		nn_add_bias(be, out, c->bias, c->oc, H * W);
		return;
	}
	int	   rows_per = (int)MAX((size_t)1, IM2COL_MAX_FLOATS / ((size_t)W * 9 * c->ic));
	float *col		= xmalloc((size_t)MIN(rows_per, H) * W * 9 * c->ic * sizeof(float));
	for (int y0 = 0; y0 < H; y0 += rows_per) {
		int		   n   = MIN(rows_per, H - y0) * W;
		im2col_ctx ctx = {col, in, y0, W, W >> shift, H >> shift, c->ic, shift};
		tpool_parallel_for(nn_pool(be), n, 64, im2col_rows, &ctx);
		float *dst = out + (size_t)y0 * W * c->oc;
		nn_matmul(be, &c->w, col, dst, c->oc, 9 * c->ic, n);
		nn_add_bias(be, dst, c->bias, c->oc, n);
	}
	free(col);
}
