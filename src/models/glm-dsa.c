#include "backend/backend.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "moe/moe_stream.h"
#include "recipe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MLA_Q_A_STACK_CAP 4096
#define MLA_KV_A_STACK_CAP 2048

static buffer mla_scratch_buf_alloc(float *stack, int stack_cap, int n, int *out_heap) {
	float *ptr	 = (n <= stack_cap) ? stack : xmalloc((size_t)n * sizeof(float));
	*out_heap	 = (ptr != stack);
	buffer buf	 = {0};
	buf.handle	 = ptr;
	buf.size	 = (size_t)n * sizeof(float);
	buf.host_ptr = ptr;
	buf.owner	 = NULL;
	return buf;
}

static void mla_scratch_buf_free(buffer *buf, int heap) {
	if (heap)
		free(buf->handle);
}

status_code op_mla_qkv_proj_fused(exec_ctx *ctx) {
	struct model   *m		  = ctx->m;
	struct kvcache *cache	  = ctx->cache;
	int				pos		  = ctx->pos;
	int				li		  = ctx->li;
	buffer		   *slots	  = ctx->s->slots;
	backend		   *a		  = m->backend;
	layer_weights  *L		  = &m->layers[li];
	const int		dim		  = m->dim;
	const int		q_lora	  = m->mla.q_lora;
	const int		kv_lora	  = m->mla.kv_lora;
	const int		qk_rope	  = m->mla.qk_rope;
	const int		q_b_rows  = m->n_heads * m->mla.qk_head;
	const int		kv_a_rows = kv_lora + qk_rope;

	buffer *xb = &slots[RECIPE_SLOT_XB];

	float  q_a_stack[MLA_Q_A_STACK_CAP];
	int	   q_a_heap;
	buffer q_a_buf = mla_scratch_buf_alloc(q_a_stack, MLA_Q_A_STACK_CAP, q_lora, &q_a_heap);

	float  kv_a_stack[MLA_KV_A_STACK_CAP];
	int	   kv_a_heap;
	buffer kv_a_buf = mla_scratch_buf_alloc(kv_a_stack, MLA_KV_A_STACK_CAP, kv_a_rows, &kv_a_heap);

	status_code st;
	if (a->matmul_multi) {
		const buffer *w_list[2]	 = {&L->q_a_w.buf, &L->kv_a_w.buf};
		uint32_t	  w_types[2] = {L->q_a_w.type, L->kv_a_w.type};
		buffer		 *y_list[2]	 = {&q_a_buf, &kv_a_buf};
		int			  n_list[2]	 = {q_lora, kv_a_rows};
		st						 = a->matmul_multi(a, w_list, w_types, xb, y_list, n_list, dim, 2);
	} else {
		st = a->matmul(a, &L->q_a_w.buf, L->q_a_w.type, xb, &q_a_buf, q_lora, dim);
		if (st == OK)
			st = a->matmul(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, &kv_a_buf, kv_a_rows, dim);
	}
	if (st != OK) {
		mla_scratch_buf_free(&q_a_buf, q_a_heap);
		mla_scratch_buf_free(&kv_a_buf, kv_a_heap);
		return st;
	}

	st = a->rmsnorm(a, &q_a_buf, &L->q_a_norm_w.buf, &q_a_buf, q_lora, m->norm_eps);
	if (st == OK) {
		buffer *q_buf = &slots[RECIPE_SLOT_Q];
		st = a->matmul(a, &L->q_b_w.buf, L->q_b_w.type, &q_a_buf, q_buf, q_b_rows, q_lora);
	}
	mla_scratch_buf_free(&q_a_buf, q_a_heap);

	if (st != OK) {
		mla_scratch_buf_free(&kv_a_buf, kv_a_heap);
		return st;
	}

	st = a->kv_put_mla(a, &cache->mla->kv, li, pos, &kv_a_buf, &L->kv_a_norm_w.buf, kv_lora,
					   qk_rope, cache->n_ctx, m->norm_eps);
	mla_scratch_buf_free(&kv_a_buf, kv_a_heap);
	return st;
}

status_code op_mla_q_proj(exec_ctx *ctx) {
	struct model  *m		= ctx->m;
	int			   li		= ctx->li;
	buffer		  *slots	= ctx->s->slots;
	backend		  *a		= m->backend;
	layer_weights *L		= &m->layers[li];
	const int	   dim		= m->dim;
	const int	   q_lora	= m->mla.q_lora;
	const int	   q_b_rows = m->n_heads * m->mla.qk_head;

	buffer *xb = &slots[RECIPE_SLOT_XB];
	float	q_a_stack[MLA_Q_A_STACK_CAP];
	int		q_a_heap;
	buffer	q_a_buf = mla_scratch_buf_alloc(q_a_stack, MLA_Q_A_STACK_CAP, q_lora, &q_a_heap);

	status_code st = a->matmul(a, &L->q_a_w.buf, L->q_a_w.type, xb, &q_a_buf, q_lora, dim);
	if (st == OK)
		st = a->rmsnorm(a, &q_a_buf, &L->q_a_norm_w.buf, &q_a_buf, q_lora, m->norm_eps);
	if (st == OK) {
		buffer *q_buf = &slots[RECIPE_SLOT_Q];
		st = a->matmul(a, &L->q_b_w.buf, L->q_b_w.type, &q_a_buf, q_buf, q_b_rows, q_lora);
	}
	mla_scratch_buf_free(&q_a_buf, q_a_heap);
	return st;
}

status_code op_mla_kv_proj(exec_ctx *ctx) {
	struct model   *m		  = ctx->m;
	struct kvcache *cache	  = ctx->cache;
	int				pos		  = ctx->pos;
	int				li		  = ctx->li;
	buffer		   *slots	  = ctx->s->slots;
	backend		   *a		  = m->backend;
	layer_weights  *L		  = &m->layers[li];
	const int		dim		  = m->dim;
	const int		kv_lora	  = m->mla.kv_lora;
	const int		qk_rope	  = m->mla.qk_rope;
	const int		kv_a_rows = kv_lora + qk_rope;

	float  kv_a_stack[MLA_KV_A_STACK_CAP];
	int	   kv_a_heap;
	buffer kv_a_buf = mla_scratch_buf_alloc(kv_a_stack, MLA_KV_A_STACK_CAP, kv_a_rows, &kv_a_heap);

	buffer	   *xb = &slots[RECIPE_SLOT_XB];
	status_code st = a->matmul(a, &L->kv_a_w.buf, L->kv_a_w.type, xb, &kv_a_buf, kv_a_rows, dim);
	if (st == OK)
		st = a->kv_put_mla(a, &cache->mla->kv, li, pos, &kv_a_buf, &L->kv_a_norm_w.buf, kv_lora,
						   qk_rope, cache->n_ctx, m->norm_eps);
	mla_scratch_buf_free(&kv_a_buf, kv_a_heap);
	return st;
}

status_code op_attention_mla(exec_ctx *ctx) {
	const recipe_op		   *op	  = ctx->op;
	struct model		   *m	  = ctx->m;
	struct kvcache		   *cache = ctx->cache;
	struct compute_scratch *s	  = ctx->s;
	int						li	  = ctx->li;
	int						pos	  = ctx->pos;
	buffer				   *slots = ctx->s->slots;
	(void)ctx->flash_attn;
	backend		  *a	   = m->backend;
	layer_weights *L	   = &m->layers[li];
	int			   n_heads = op->u.attention.n_heads;
	int			   qk_head = m->mla.qk_head;
	int			   qk_rope = m->mla.qk_rope;
	int			   qk_nope = m->mla.qk_nope;
	int			   v_head  = m->mla.v_head;
	int			   kv_lora = m->mla.kv_lora;
	float		   scale   = op->u.attention.scale;

	return a->attention_mla(a, &slots[RECIPE_SLOT_Q], &cache->mla->kv, &L->k_b_w.buf, &L->v_b_w.buf,
							&slots[op->out], li, pos, n_heads, qk_head, qk_rope, qk_nope, v_head,
							kv_lora, cache->n_ctx, s->rope_cos, s->rope_sin, scale);
}

static model_recipe *build_glm_dsa_recipe(const model *m) {
	model_recipe *r = xcalloc(1, sizeof(model_recipe));

	const int	dim		  = m->dim;
	const int	n_heads	  = m->n_heads;
	const int	n_ctx	  = m->n_ctx;
	const float eps		  = m->norm_eps;
	const int	rope_neox = m->arch_info->uses_neox_rope;

	const int qk_head  = m->mla.qk_head;
	const int v_head   = m->mla.v_head;
	const int q_b_rows = n_heads * qk_head;
	const int wo_in	   = n_heads * v_head;

	r->max_intermediate =
		m->moe.moe_intermediate > m->intermediate ? m->moe.moe_intermediate : m->intermediate;
	r->max_head_dim = qk_head;
	r->max_kv_heads = n_heads;

	{
		recipe_op *ops = xcalloc(1, sizeof(recipe_op));
		ops[0]		   = (recipe_op){
			.kind  = OP_EMBD_LOOKUP,
			.in	   = {RECIPE_SLOT_NONE, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_X,
			.w_idx = WIDX_TOK_EMBD,
			.stage = STAGE_EMBD,
		};
		r->pre_ops	 = ops;
		r->n_pre_ops = 1;
	}

	{
		int		   cap = 32;
		recipe_op *ops = xcalloc(cap, sizeof(recipe_op));
		int		   i   = 0;

		ops[i++] = (recipe_op){
			.kind	   = OP_RMSNORM,
			.in		   = {RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_XB,
			.w_idx	   = WIDX_ATTN_NORM,
			.stage	   = STAGE_RMSNORM,
			.u.rmsnorm = {.eps = eps, .n_heads = 0},
		};

		ops[i++] = (recipe_op){
			.kind	  = OP_MLA_QKV_PROJ_FUSED,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_Q,
			.w_idx	  = WIDX_NONE,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = q_b_rows, .k = dim},
		};

		ops[i++] = (recipe_op){
			.kind  = OP_ATTENTION_MLA,
			.in	   = {RECIPE_SLOT_Q, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_XB2,
			.w_idx = WIDX_NONE,
			.stage = STAGE_ATTN,
			.u.attention =
				{
					.n_heads		   = n_heads,
					.n_kv_heads		   = n_heads,
					.head_dim		   = qk_head,
					.n_ctx			   = n_ctx,
					.scale			   = 1.0f / sqrtf((float)qk_head),
					.sliding_window	   = 0,
					.n_kv_heads_active = n_heads,
				},
		};

		ops[i++] = (recipe_op){
			.kind	  = OP_MATMUL_RESIDUAL,
			.in		  = {RECIPE_SLOT_XB2, RECIPE_SLOT_X, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_X,
			.w_idx	  = WIDX_WO,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = dim, .k = wo_in},
		};

		ops[i++] = (recipe_op){
			.kind	   = OP_RMSNORM,
			.in		   = {RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_XB,
			.w_idx	   = WIDX_FFN_NORM,
			.stage	   = STAGE_RMSNORM,
			.u.rmsnorm = {.eps = eps, .n_heads = 0},
		};

		if (m->moe.n_shared_experts > 0) {
			int sh_inter = m->moe.moe_intermediate * m->moe.n_shared_experts;
			ops[i++]	 = (recipe_op){
				.kind	  = OP_MOE_SHARED,
				.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
				.out	  = RECIPE_SLOT_FFN_ACT,
				.w_idx	  = WIDX_NONE,
				.stage	  = STAGE_MATMUL,
				.u.matmul = {.n = sh_inter, .k = dim},
			};
		}

		ops[i++] = (recipe_op){
			.kind	  = OP_MOE_ROUTER,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_ROUTER_IDS,
			.w_idx	  = WIDX_FFN_GATE_INP,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = m->moe.n_experts, .k = dim},
		};

		ops[i++] = (recipe_op){
			.kind	  = OP_MOE_EXPERTS,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_ROUTER_IDS, RECIPE_SLOT_ROUTER_W},
			.out	  = RECIPE_SLOT_XB2,
			.w_idx	  = WIDX_NONE,
			.stage	  = STAGE_MATMUL,
			.u.matmul = {.n = dim, .k = m->moe.moe_intermediate},
		};

		ops[i++] = (recipe_op){
			.kind  = OP_ADD,
			.in	   = {RECIPE_SLOT_X, RECIPE_SLOT_XB2, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_NONE,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_ADD,
		};
		if (m->moe.n_shared_experts > 0) {
			ops[i++] = (recipe_op){
				.kind  = OP_ADD,
				.in	   = {RECIPE_SLOT_X, RECIPE_SLOT_FFN_ACT, RECIPE_SLOT_NONE},
				.out   = RECIPE_SLOT_NONE,
				.w_idx = RECIPE_NO_WEIGHT,
				.stage = STAGE_ADD,
			};
		}

		r->layer.ops   = ops;
		r->layer.n_ops = i;
	}

	{
		recipe_op *ops = xcalloc(3, sizeof(recipe_op));
		int		   i   = 0;
		ops[i++]	   = (recipe_op){
			.kind	   = OP_RMSNORM,
			.in		   = {RECIPE_SLOT_X, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	   = RECIPE_SLOT_XB,
			.w_idx	   = WIDX_OUTPUT_NORM,
			.stage	   = STAGE_LOGITS_NORM,
			.u.rmsnorm = {.eps = eps, .n_heads = 0},
		};
		ops[i++] = (recipe_op){
			.kind	  = OP_MATMUL,
			.in		  = {RECIPE_SLOT_XB, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out	  = RECIPE_SLOT_LOGITS,
			.w_idx	  = WIDX_OUTPUT_W,
			.stage	  = STAGE_LOGITS_MATMUL,
			.u.matmul = {.n = m->vocab_size, .k = dim},
		};
		ops[i++] = (recipe_op){
			.kind  = OP_LOGITS_READBACK,
			.in	   = {RECIPE_SLOT_LOGITS, RECIPE_SLOT_NONE, RECIPE_SLOT_NONE},
			.out   = RECIPE_SLOT_NONE,
			.w_idx = RECIPE_NO_WEIGHT,
			.stage = STAGE_LOGITS_READBACK,
		};
		r->post_ops	  = ops;
		r->n_post_ops = i;
	}

	moe_stream_cache_init((struct model *)m);
	(void)rope_neox;
	return r;
}

RECIPE_REGISTER(glm_dsa, "glm-dsa", build_glm_dsa_recipe)