#include "backend/backend.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "recipe.h"
#include "threadpool.h"

#include <math.h>
#include <string.h>

static tpool *lfm_pool(const model *m) {
	if (m->backend && m->backend->get_pool)
		return m->backend->get_pool(m->backend);
	return NULL;
}

static int lfm_attn_buf(const model *m) {
	int buf	  = m->dim;
	int q_out = m->n_heads * m->head_dim;
	return buf > q_out ? buf : q_out;
}

typedef struct {
	float		*out;
	float		*state;
	const float *mixed;
	const float *conv_w;
	int			 dim, kernel, history;
	int			 in_stride, out_stride;
	int			 n_rows;
} shortconv_job;

static void shortconv_chunk(int begin, int end, int tid, void *ctx) {
	(void)tid;
	shortconv_job *j = ctx;
	const int	   K = j->kernel;
	for (int c = begin; c < end; c++) {
		const float *w	   = j->conv_w + (size_t)c * K;
		float		*hist  = j->state + (size_t)c * j->history;
		const float *b_col = j->mixed + c;
		const float *c_col = j->mixed + (size_t)j->dim + c;
		const float *x_col = j->mixed + 2 * (size_t)j->dim + c;
		float		*o_col = j->out + c;

		if (K == 3) {
			float h0 = hist[0], h1 = hist[1];
			for (int t = 0; t < j->n_rows; t++) {
				size_t off						 = (size_t)t * j->in_stride;
				float  bx						 = b_col[off] * x_col[off];
				float  sum						 = h0 * w[0] + h1 * w[1] + bx * w[2];
				h0								 = h1;
				h1								 = bx;
				o_col[(size_t)t * j->out_stride] = c_col[off] * sum;
			}
			hist[0] = h0;
			hist[1] = h1;
		} else {
			for (int t = 0; t < j->n_rows; t++) {
				size_t off = (size_t)t * j->in_stride;
				float  bx  = b_col[off] * x_col[off];
				float  sum = bx * w[K - 1];
				for (int k = 0; k < j->history; k++)
					sum += hist[k] * w[k];
				memmove(hist, hist + 1, (size_t)(j->history - 1) * sizeof(float));
				hist[j->history - 1]			 = bx;
				o_col[(size_t)t * j->out_stride] = c_col[off] * sum;
			}
		}
	}
}

status_code op_shortconv(exec_ctx *ctx) {
	if (!ctx || !ctx->m || !ctx->cache || !ctx->cache->hybrid || !ctx->s)
		return ERR_INVALID_ARG;
	model					  *m	  = ctx->m;
	layer_weights			  *L	  = &m->layers[ctx->li];
	kvcache_hybrid			  *hc	  = ctx->cache->hybrid;
	const model_hybrid_params *p	  = &m->hybrid;
	const float				  *conv_w = (const float *)L->ssm_conv1d_w.host_ptr;
	if (!conv_w || p->conv_kernel < 2 || ctx->li < 0 || ctx->li >= m->n_layers)
		return ERR_FORMAT;

	profile_scope ps = profile_begin(&ctx->s->prof, ctx->op->stage);

	int	   n_rows = recipe_exec_is_batch(ctx) ? ctx->n_rows : 1;
	float *state  = hc->conv_state + (size_t)ctx->li * hc->conv_stride;
	if (ctx->pos_start == 0)
		memset(state, 0, (size_t)m->dim * (size_t)(p->conv_kernel - 1) * sizeof(float));

	shortconv_job job = {
		.out		= recipe_slot_f32(ctx, ctx->op->out),
		.state		= state,
		.mixed		= recipe_slot_f32(ctx, ctx->op->in[0]),
		.conv_w		= conv_w,
		.dim		= m->dim,
		.kernel		= p->conv_kernel,
		.history	= p->conv_kernel - 1,
		.in_stride	= model_hybrid_proj_size(m),
		.out_stride = lfm_attn_buf(m),
		.n_rows		= n_rows,
	};
	if (!job.out || !job.mixed)
		return ERR_INVALID_ARG;

	tpool *pool = lfm_pool(m);
	if (pool && m->dim > 8 && tpool_current_tid() < 0)
		tpool_parallel_for(pool, m->dim, 8, shortconv_chunk, &job);
	else
		shortconv_chunk(0, m->dim, -1, &job);

	profile_end(&ctx->s->prof, &ps);
	return OK;
}

static int lfm_append_ffn(recipe_op *ops, int i, const model *m, int li) {
	int	  dim	= m->dim;
	int	  inter = m->intermediate;
	float eps	= m->norm_eps;

	ops[i++] = mk_add(RECIPE_SLOT_ATTN_OUT, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_ATTN_OUT, STAGE_ADD);
	ops[i++] = mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_FFN_NORM, eps, STAGE_RMSNORM);

	if (li >= 0 && m->layers[li].gate_up_fused) {
		ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE_UP, WIDX_GATE_UP, 2 * inter, dim,
							 STAGE_MATMUL);
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE_FUSED,
			.in		   = {RECIPE_SLOT_FFN_GATE_UP, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	} else {
		ops[i++] =
			mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_GATE, WIDX_GATE, inter, dim, STAGE_MATMUL);
		ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_FFN_UP, WIDX_UP, inter, dim, STAGE_MATMUL);
		ops[i++] = (recipe_op){
			.kind	   = OP_FFN_ACTIVATE,
			.in		   = {RECIPE_SLOT_FFN_GATE, RECIPE_SLOT_FFN_UP, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_FFN_ACT,
			.w_idx	   = RECIPE_NO_WEIGHT,
			.stage	   = STAGE_FFN_ACT,
			.u.ffn_act = {.n = inter, .activation = ACTIVATION_SILU},
		};
	}
	ops[i++] = mk_matmul(RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_XB2, WIDX_DOWN, dim, inter, STAGE_MATMUL);
	ops[i++] = mk_add(RECIPE_SLOT_XB2, RECIPE_SLOT_X, STAGE_ADD);
	ops[i++] = mk_swap(RECIPE_SLOT_X, RECIPE_SLOT_XB2, STAGE_ADD);
	return i;
}

static int lfm_append_conv_block(recipe_op *ops, const model *m, int li) {
	int						   i   = 0;
	int						   dim = m->dim;
	const model_hybrid_params *p   = &m->hybrid;

	ops[i++] =
		mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_HYB_PROJ, WIDX_ATTN_QKV, p->conv_dim, dim,
						 STAGE_MATMUL);
	ops[i++] = (recipe_op){
		.kind  = OP_SHORTCONV,
		.in	   = {RECIPE_SLOT_HYB_PROJ, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
	};
	ops[i++] = mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_SSM_OUT, dim, p->value_dim,
						 STAGE_MATMUL);
	return lfm_append_ffn(ops, i, m, li);
}

static int lfm_append_attention_block(recipe_op *ops, const model *m, int li) {
	int i	   = 0;
	int dim	   = m->dim;
	int q_out  = m->n_heads * m->head_dim;
	int kv_out = m->n_kv_heads * m->head_dim;
	int neox   = m->arch_info->uses_neox_rope;

	ops[i++] =
		mk_rmsnorm(RECIPE_SLOT_X, RECIPE_SLOT_XB, WIDX_ATTN_NORM, m->norm_eps, STAGE_RMSNORM);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_Q, WIDX_WQ, q_out, dim, STAGE_MATMUL);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_K, WIDX_WK, kv_out, dim, STAGE_MATMUL);
	ops[i++] = mk_matmul(RECIPE_SLOT_XB, RECIPE_SLOT_V, WIDX_WV, kv_out, dim, STAGE_MATMUL);

	ops[i++] = (recipe_op){
		.kind	   = OP_RMSNORM_PER_HEAD,
		.in		   = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	   = RECIPE_SLOT_Q,
		.w_idx	   = WIDX_ATTN_Q_NORM,
		.stage	   = STAGE_RMSNORM,
		.u.rmsnorm = {.eps = m->norm_eps, .n_heads = m->n_heads},
	};
	ops[i++] = (recipe_op){
		.kind	   = OP_RMSNORM_PER_HEAD,
		.in		   = {RECIPE_SLOT_K, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	   = RECIPE_SLOT_K,
		.w_idx	   = WIDX_ATTN_K_NORM,
		.stage	   = STAGE_RMSNORM,
		.u.rmsnorm = {.eps = m->norm_eps, .n_heads = m->n_kv_heads},
	};
	ops[i++] = (recipe_op){
		.kind	= OP_ROPE,
		.in		= {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	= RECIPE_SLOT_NONE,
		.w_idx	= RECIPE_NO_WEIGHT,
		.stage	= STAGE_ROPE,
		.u.rope = {.n_heads = m->n_heads, .head_dim = m->head_dim, .rope_neox = neox},
	};
	ops[i++] = (recipe_op){
		.kind	= OP_ROPE,
		.in		= {RECIPE_SLOT_K, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out	= RECIPE_SLOT_NONE,
		.w_idx	= RECIPE_NO_WEIGHT,
		.stage	= STAGE_ROPE,
		.u.rope = {.n_heads = m->n_kv_heads, .head_dim = m->head_dim, .rope_neox = neox},
	};
	ops[i++] = (recipe_op){
		.kind  = OP_KV_PUT,
		.in	   = {RECIPE_SLOT_K, RECIPE_SLOT_V, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_NONE,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_KVPUT,
	};
	ops[i++] = (recipe_op){
		.kind  = OP_ATTENTION,
		.in	   = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
		.out   = RECIPE_SLOT_XB2,
		.w_idx = RECIPE_NO_WEIGHT,
		.stage = STAGE_ATTN,
		.u.attention =
			{
				.n_heads		   = m->n_heads,
				.n_kv_heads		   = m->n_kv_heads,
				.head_dim		   = m->head_dim,
				.n_ctx			   = m->n_ctx,
				.scale			   = 1.0f / sqrtf((float)m->head_dim),
				.sliding_window	   = m->sliding_window,
				.n_kv_heads_active = m->n_kv_heads,
			},
	};
	ops[i++] = mk_matmul(RECIPE_SLOT_XB2, RECIPE_SLOT_ATTN_OUT, WIDX_WO, dim, q_out, STAGE_MATMUL);
	return lfm_append_ffn(ops, i, m, li);
}

enum { LFM_MAX_OPS_PER_LAYER = 24 };

static model_recipe *build_lfm2_recipe(const model *m) {
	model_recipe *r		= xcalloc(1, sizeof(*r));
	r->max_intermediate = m->intermediate;
	r->max_head_dim		= m->head_dim;
	r->max_kv_heads		= m->n_kv_heads;
	recipe_build_pre_ops(r, m);

	r->layer.ops	 = xcalloc(LFM_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	r->layer.n_ops	 = LFM_MAX_OPS_PER_LAYER;
	r->per_layer_ops = xcalloc((size_t)m->n_layers * LFM_MAX_OPS_PER_LAYER, sizeof(recipe_op));
	for (int li = 0; li < m->n_layers; li++) {
		recipe_op *ops = r->per_layer_ops + (size_t)li * LFM_MAX_OPS_PER_LAYER;
		int		   n   = model_layer_is_recurrent(m, li) ? lfm_append_conv_block(ops, m, li)
														 : lfm_append_attention_block(ops, m, li);
		if (n > LFM_MAX_OPS_PER_LAYER) {
			ERROR("lfm2: layer %d produced %d recipe ops but capacity is %d -- "
				  "raise LFM_MAX_OPS_PER_LAYER",
				  li, n, LFM_MAX_OPS_PER_LAYER);
			free(r->pre_ops);
			free(r->layer.ops);
			free(r->per_layer_ops);
			free(r);
			return NULL;
		}
	}
	recipe_build_post_ops(r, m);
	return r;
}

RECIPE_REGISTER(lfm2, "lfm2", build_lfm2_recipe)
