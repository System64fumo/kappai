#include "recipe/internal.h"

static const float *encoder_weight(exec_ctx *ctx, uint8_t index, float_buf *stage, int n) {
	const buffer *w = exec_weight(ctx, index);
	if (buffer_is_host_resident(w))
		return buffer_f32(w);
	float *out = float_buf_ensure(stage, (size_t)n);
	ensure_sync(w->owner);
	return w->owner->buffer_read_f32(w->owner, w, out, n) == OK ? out : NULL;
}

typedef struct {
	const float *x, *w, *bias;
	float		*out;
	int			 n;
	float		 eps;
} layernorm_job;

static void layernorm_rows(int begin, int end, int tid, void *data) {
	(void)tid;
	layernorm_job *j = data;
	for (int row = begin; row < end; row++) {
		const float *x	 = j->x + (size_t)row * j->n;
		float		*out = j->out + (size_t)row * j->n;
		double		 sum = 0.0;
		for (int i = 0; i < j->n; i++)
			sum += x[i];
		float  mean		= (float)(sum / j->n);
		double variance = 0.0;
		for (int i = 0; i < j->n; i++) {
			float delta = x[i] - mean;
			variance += delta * delta;
		}
		float scale = 1.0f / sqrtf((float)(variance / j->n) + j->eps);
		for (int i = 0; i < j->n; i++)
			out[i] = ((x[i] - mean) * scale) * j->w[i] + j->bias[i];
	}
}

status_code op_layernorm(exec_ctx *ctx) {
	const recipe_op *op	   = ctx->op;
	int				 n	   = op->u.layernorm.n;
	int				 total = n * ctx->rows.n_rows;
	layernorm_job	 job   = {
			 .x	   = recipe_slot_read_f32(ctx, op->in[0], &ctx->s->run.stage_a, total),
			 .w	   = encoder_weight(ctx, op->w_idx, &ctx->s->run.op_ws, n),
			 .bias = encoder_weight(ctx, op->u.layernorm.bias_idx, &ctx->s->run.scale_tmp, n),
			 .out  = recipe_slot_write_stage(ctx, op->out, &ctx->s->run.stage_b, total),
			 .n	   = n,
			 .eps  = op->u.layernorm.eps,
	 };
	if (!job.x || !job.w || !job.bias || !job.out)
		return ERR_INVALID_ARG;
	tpool *pool = model_get_pool(ctx->m);
	if (pool && tpool_current_tid() < 0)
		tpool_parallel_for(pool, ctx->rows.n_rows, 1, layernorm_rows, &job);
	else
		layernorm_rows(0, ctx->rows.n_rows, 0, &job);
	return recipe_slot_write_commit(ctx, op->out, job.out, total);
}

status_code op_add_bias(exec_ctx *ctx) {
	const recipe_op *op	   = ctx->op;
	int				 n	   = op->u.ffn_act.n;
	int				 types = op->kind == OP_ADD_TYPE_EMBD ? op->u.ffn_act.n_types : 1;
	int				 type  = op->kind == OP_ADD_TYPE_EMBD ? ctx->rows.token_type : 0;
	if (type < 0 || type >= types)
		return ERR_INVALID_ARG;
	const float *weights = encoder_weight(ctx, op->w_idx, &ctx->s->run.op_ws, n * types);
	int			 total	 = n * ctx->rows.n_rows;
	const float *x		 = recipe_slot_read_f32(ctx, op->in[0], &ctx->s->run.stage_a, total);
	float		*out	 = recipe_slot_write_stage(ctx, op->out, &ctx->s->run.stage_b, total);
	if (!weights || !x || !out)
		return ERR_INVALID_ARG;
	const float *bias = weights + (size_t)type * n;
	for (int row = 0; row < ctx->rows.n_rows; row++)
		for (int i = 0; i < n; i++)
			out[(size_t)row * n + i] = x[(size_t)row * n + i] + bias[i];
	return recipe_slot_write_commit(ctx, op->out, out, total);
}

status_code op_split_qkv(exec_ctx *ctx) {
	int			 dim  = ctx->op->u.matmul.k;
	int			 rows = ctx->rows.n_rows;
	const float *input =
		recipe_slot_read_f32(ctx, ctx->op->in[0], &ctx->s->run.stage_a, rows * 3 * dim);
	if (!input)
		return ERR_INVALID_ARG;
	const uint8_t slots[] = {ctx->op->out, ctx->op->out1, ctx->op->out2};
	for (int part = 0; part < 3; part++) {
		float *out = recipe_slot_write_stage(ctx, slots[part], &ctx->s->run.stage_b, rows * dim);
		if (!out)
			return ERR_INVALID_ARG;
		for (int row = 0; row < rows; row++)
			memcpy(out + (size_t)row * dim, input + ((size_t)row * 3 + part) * dim,
				   (size_t)dim * sizeof(float));
		status_code st = recipe_slot_write_commit(ctx, slots[part], out, rows * dim);
		if (st != OK)
			return st;
	}
	return OK;
}

status_code op_activate(exec_ctx *ctx) {
	int			 n	 = ctx->op->u.ffn_act.n * ctx->rows.n_rows;
	const float *x	 = recipe_slot_read_f32(ctx, ctx->op->in[0], &ctx->s->run.stage_a, n);
	float		*out = recipe_slot_write_stage(ctx, ctx->op->out, &ctx->s->run.stage_b, n);
	if (!x || !out)
		return ERR_INVALID_ARG;
	for (int i = 0; i < n; i++) {
		if (ctx->op->u.ffn_act.activation == ACTIVATION_RELU)
			out[i] = MAX(x[i], 0.0f);
		else if (ctx->op->u.ffn_act.activation == ACTIVATION_GELU_ERF)
			out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.7071067811865475f));
		else
			return ERR_UNSUPPORTED;
	}
	return recipe_slot_write_commit(ctx, ctx->op->out, out, n);
}

status_code op_attention_full(exec_ctx *ctx) {
	const recipe_op *op = ctx->op;
	backend			*a	= OP_BACKEND(exec_layer_backend(ctx), attention_full_batch);
	if (!a->attention_full_batch)
		return ERR_UNSUPPORTED;
	int rows = ctx->rows.n_rows;
	if (a == exec_layer_backend(ctx))
		return a->attention_full_batch(
			a, exec_slot(ctx, op->in[0]), exec_slot(ctx, op->in[1]), exec_slot(ctx, op->in[2]),
			exec_slot(ctx, op->out), op->u.attention.n_heads, op->u.attention.n_kv_heads,
			op->u.attention.head_dim, rows, ctx->flash_attn, op->u.attention.scale);
	int			 qn	 = rows * op->u.attention.n_heads * op->u.attention.head_dim;
	int			 kn	 = rows * op->u.attention.n_kv_heads * op->u.attention.head_dim;
	const float *q	 = recipe_slot_read_f32(ctx, op->in[0], &ctx->s->run.stage_a, qn);
	const float *k	 = recipe_slot_read_f32(ctx, op->in[1], &ctx->s->hybrid.host, kn);
	const float *v	 = recipe_slot_read_f32(ctx, op->in[2], &ctx->s->hybrid.host2, kn);
	float		*out = recipe_slot_write_stage(ctx, op->out, &ctx->s->run.stage_b, qn);
	if (!q || !k || !v || !out)
		return ERR_INVALID_ARG;
	buffer		qb = buffer_host_view(a, (void *)q, (size_t)qn * sizeof(float));
	buffer		kb = buffer_host_view(a, (void *)k, (size_t)kn * sizeof(float));
	buffer		vb = buffer_host_view(a, (void *)v, (size_t)kn * sizeof(float));
	buffer		ob = buffer_host_view(a, out, (size_t)qn * sizeof(float));
	status_code st = a->attention_full_batch(a, &qb, &kb, &vb, &ob, op->u.attention.n_heads,
											 op->u.attention.n_kv_heads, op->u.attention.head_dim,
											 rows, ctx->flash_attn, op->u.attention.scale);
	return st == OK ? recipe_slot_write_commit(ctx, op->out, out, qn) : st;
}
