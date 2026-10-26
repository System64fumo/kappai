#ifndef RECIPE_INTERNAL_H
#define RECIPE_INTERNAL_H

#include "backend/backend.h"
#include "capability.h"
#include "compute.h"
#include "graph.h"
#include "kvcache.h"
#include "log.h"
#include "mem_plan.h"
#include "model.h"
#include "moe/moe_common.h"
#include "moe/moe_exec.h"
#include "moe/moe_stream.h"
#include "monitor.h"
#include "partition.h"
#include "recipe.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OP_BACKEND(be, field) op_backend_pick((be), ((be)->field != NULL), #field)

typedef struct {
	int n_calls;
	int rows;
} exec_segments;

#define EXEC_CAPS_RESOLVED 0x80000000u

enum { PLE_CAP_HEAP = 1u << 0, PLE_CAP_FUSED_DOWN = 1u << 1, PLE_CAP_FUSED_NORM = 1u << 2 };

enum {
	MOE_SH_FUSED_ACT = 1u << 0,
	MOE_SH_MULTI	 = 1u << 1,
	MOE_SH_ACT		 = 1u << 2,
	MOE_SH_COPY		 = 1u << 3
};

enum { MLA_QKV_MULTI = 1u << 0 };

enum { GDN_NATIVE = 1u << 0 };

typedef struct {
	float_buf fb;
	buffer	  b;
} bs_pair;

typedef struct {
	int cnt;
	int u;
} moe_order_pair;

struct batch_scratch {
	bs_pair pair[RECIPE_SLOT_MAX];
	uint8_t own[RECIPE_SLOT_MAX];
	uint8_t logits_slot;

	char	 *moe_pack;
	size_t	  moe_pack_cap;
	float_buf copy_tmp;

	buffer			 arena;
	float_buf		 arena_host;
	size_t			 arena_bytes;
	mem_layout		 layout;
	mem_layout_cache layouts;

	int	  *moe_router_ids;
	float *moe_router_w;
	int	  *moe_union_ids;
	int	   moe_n_union;

	int *moe_union_rows;
	int *moe_union_kidx;
	int *moe_union_offsets;
	int *moe_union_cursor;
	int *moe_union_count;

	int *moe_expert_union_idx;
	int *moe_expert_seen_gen;
	int	 moe_expert_gen;
	int	 moe_expert_alloc;

	int *moe_union_order;
	int	 moe_union_order_cap;
	int *moe_union_sorted_ids;
	int	 moe_union_sorted_ids_cap;

	moe_order_pair *moe_order_pairs;
	int				moe_order_pairs_cap;
	int				moe_rowslots_cap;

	float_buf moe_expert_out;
	float_buf moe_out;
	float_buf moe_router_logits;
	float_buf moe_router_inp;
	buffer	  moe_router_logits_dev;
	buffer	  moe_router_inp_dev;
	float_buf moe_xb_f;

	float_buf moe_gather_x;
	float_buf moe_exp_gu;
	float_buf moe_up_scratch;
	float_buf moe_exp_act;
	float_buf moe_exp_y;
	float_buf moe_fused_host;
	float_buf moe_fact_out;
	float_buf moe_zeros;

	moe_expert_slot *moe_per_token_slots;
	int				 moe_per_token_slots_cap;

	moe_expert_slot *moe_union_slots;
	int				 moe_union_slots_cap;
	int				 moe_union_pending_n;

	const int32_t *tokens;

	float_buf ple_buf;
	buffer	  ple_all;
	float_buf ple_inp;
	float_buf ple_proj;
	float_buf ple_slice;
	buffer	  ple_inp_dev;
	buffer	  ple_slice_dev;
	buffer	  ple_proj_dev;
};

int						matmul_type_ok(const backend *a, uint32_t t);
int						ple_build_dev_path_cached(const struct model *m, backend *a, int batched);
extern const weight_ref WEIGHT_REF_NONE;
weight_ref			   *resolve_weight_ref(const model *m, int li, uint8_t w_idx);
int						model_has_sliding_layers(const model *m);

recipe_op mk_rmsnorm_per_head(uint8_t io, uint8_t widx, float eps, int n_heads);

uint32_t	ple_inject_caps(const backend *a);
uint32_t	moe_shared_caps(const backend *a);
uint32_t	mla_qkv_caps(const backend *a);
uint32_t	gdn_caps(const backend *a);
void		exec_plan_free(struct exec_plan *p);
status_code exec_plan_build(model_recipe *r, const struct model *m);
int			recipe_is_decode_batchable(const model *m);

const char *recipe_op_name(op_kind k);

extern _Atomic(backend *) cached_host_backend;
const float *recipe_slot_read_f32(const exec_ctx *ctx, uint8_t idx, float_buf *stage, int n);
float		*recipe_slot_write_stage(const exec_ctx *ctx, uint8_t idx, float_buf *stage, int n);
status_code	 recipe_slot_write_commit(const exec_ctx *ctx, uint8_t idx, const float *staged, int n);
void		 profile_weight_bytes(profile *prof, stage s, uint32_t w_type, int rows, int k);
int	 exec_matmul_run_coalesced_batch(const recipe_op *ops, int n_ops, model *m, compute_scratch *s,
									 int li, buffer *xb, buffer *ys[], int n_rows,
									 status_code *out_status);
int	 exec_matmul_run_qonly(const recipe_op *ops, int n_ops, model *m, buffer *xb, buffer **y_list,
						   int li, int n_rows, status_code *out_status);
void op_fallback_note(backend *a, const char *op_name);
status_code op_ffn_activate_ex(exec_ctx *ctx);
status_code op_rope_ext_batched(exec_ctx *ctx);
status_code op_rope_ext(exec_ctx *ctx);
status_code op_qk_norm_rope_unfused(exec_ctx *ctx);
status_code op_qk_norm_rope(exec_ctx *ctx);
status_code scale_inplace_fallback(struct compute_scratch *s, backend *sync_be, buffer *xb, int n,
								   float scale);
status_code op_rmsnorm(exec_ctx *ctx);
status_code op_matmul_plain(exec_ctx *ctx);
status_code op_copy_k_to_v(exec_ctx *ctx);
status_code op_matmul_residual(exec_ctx *ctx);
status_code op_matmul_ffn_down(exec_ctx *ctx);
status_code op_matmul_ffn_down_fused(exec_ctx *ctx);
status_code op_matmul_ffn_down_composed(exec_ctx *ctx);
status_code op_rope(exec_ctx *ctx);
status_code op_rope_qk_fused(exec_ctx *ctx);
status_code op_kv_put(exec_ctx *ctx);
status_code op_attention_global(exec_ctx *ctx);
status_code op_attention_swa(exec_ctx *ctx);
status_code op_attention(exec_ctx *ctx);
status_code op_add(exec_ctx *ctx);
status_code op_swap(exec_ctx *ctx);
status_code op_ffn_activate(exec_ctx *ctx);
status_code op_ffn_activate_fused_device(exec_ctx *ctx);
status_code op_ffn_activate_fused_host(exec_ctx *ctx);
status_code op_ffn_activate_fused(exec_ctx *ctx);
status_code op_softcap_native(exec_ctx *ctx);
status_code op_softcap(exec_ctx *ctx);
status_code op_logits_readback(exec_ctx *ctx);
status_code op_rmsnorm_per_head_batch(exec_ctx *ctx);
status_code op_rmsnorm_per_head_rows(exec_ctx *ctx);
status_code op_rmsnorm_per_head(exec_ctx *ctx);
status_code op_rmsnorm_noweight_ph_batch(exec_ctx *ctx);
status_code op_rmsnorm_noweight_batch(exec_ctx *ctx);
status_code op_rmsnorm_noweight_ph_rows(exec_ctx *ctx);
status_code op_rmsnorm_noweight_rows(exec_ctx *ctx);
status_code op_rmsnorm_noweight(exec_ctx *ctx);
status_code op_rmsnorm_add_fused(exec_ctx *ctx);
status_code op_rmsnorm_add_composed(exec_ctx *ctx);
status_code op_rmsnorm_add(exec_ctx *ctx);
int			exec_rmsnorm_add_norm(exec_ctx *ctx, const recipe_op *add, const recipe_op *norm);
status_code op_scale(exec_ctx *ctx);
status_code op_scale_native(exec_ctx *ctx);
status_code op_scale_host(exec_ctx *ctx);
status_code op_matmul_multi_fused(exec_ctx *ctx);
status_code op_matmul_multi_split(exec_ctx *ctx);
status_code matmul_multi_batch_body(exec_ctx *ctx);

status_code	   op_split_qgate_native(exec_ctx *ctx);
status_code	   op_split_qgate(exec_ctx *ctx);
status_code	   op_partial_rope_qk_native(exec_ctx *ctx);
status_code	   op_partial_rope_qk(exec_ctx *ctx);
status_code	   op_attn_output_gate_native(exec_ctx *ctx);
status_code	   op_attn_output_gate(exec_ctx *ctx);
status_code	   op_gated_delta_net(exec_ctx *ctx);
status_code	   op_shortconv(exec_ctx *ctx);
backend_weight weight_view(const weight_ref *w);
status_code	   op_hc_init(exec_ctx *ctx);
status_code	   op_hc_final(exec_ctx *ctx);
status_code	   op_hc_pre(exec_ctx *ctx);
status_code	   op_hc_post(exec_ctx *ctx);
status_code	   op_kda(exec_ctx *ctx);

status_code op_mla_qkv_proj_fused(exec_ctx *ctx);
status_code op_mla_q_proj(exec_ctx *ctx);
status_code op_mla_kv_proj(exec_ctx *ctx);
status_code op_attention_mla(exec_ctx *ctx);
status_code op_mla_absorbed(exec_ctx *ctx);

status_code moe_activate(backend *a, float *act, const float *gate, const float *up, int n_i,
						 float gs, float us, int use_gelu);
status_code moe_router_batch(exec_ctx *ctx);
status_code moe_experts_batch(exec_ctx *ctx);
status_code moe_shared_batch(exec_ctx *ctx);

int			ple_build_dev_path_ok(backend *a, const struct model *m, int batched);
status_code ple_build_batch(exec_ctx *ctx);
status_code ple_build_device(exec_ctx *ctx);
status_code ple_build_host(exec_ctx *ctx);
status_code ple_proj_inject_batch(exec_ctx *ctx);

int			   bs_ensure_slot(batch_scratch *bs, backend *owner, uint8_t slot, size_t n_elems);
buffer		  *batch_slot(batch_scratch *bs, uint8_t slot);
buffer		   batch_row_view(const buffer *whole, int row, int row_elems);
status_code	   embd_fill_rows(backend *a, const buffer *w, uint32_t w_type, const int32_t *tokens,
							  buffer *xs, int n_rows, int dim, float scale);
buffer		   bs_cached_view(float_buf *fb, backend *owner, size_t n_elems);
batch_scratch *batch_scratch_prepare(model *m, compute_scratch *s, int n_rows);

static inline backend *op_host_backend(void) {
	backend *b = atomic_load_explicit(&cached_host_backend, memory_order_relaxed);
	if (!b) {
		b = backend_host();
		atomic_store_explicit(&cached_host_backend, b, memory_order_relaxed);
	}
	return b;
}

static inline backend *op_backend_fallback(backend *a, const char *op_name);
static inline backend *op_backend_pick(backend *a, int has_field, const char *op_name) {
	if (has_field)
		return a;
	if (backend_has_cap(a, BCAP_IS_HOST))
		return a;
	return op_backend_fallback(a, op_name);
}

static inline float	 *batch_buf_ptr(const buffer *b);
static inline buffer *exec_slot(const exec_ctx *ctx, uint8_t idx) {
	return batch_slot(ctx->bs, idx);
}

static inline exec_segments exec_segments_of(const exec_ctx *ctx) {
	if (ctx->rows.pos)
		return (exec_segments){ctx->rows.n_rows, 1};
	return (exec_segments){1, ctx->rows.n_rows};
}

static inline int exec_segment_pos(const exec_ctx *ctx, int call) {
	return ctx->rows.pos ? ctx->rows.pos[call] : ctx->rows.pos_start;
}

static inline int exec_row_pos(const exec_ctx *ctx, int row) {
	return ctx->rows.pos ? ctx->rows.pos[row] : ctx->rows.pos_start + row;
}

static inline kvcache *exec_segment_cache(const exec_ctx *ctx, int call) {
	return ctx->rows.caches ? ctx->rows.caches[call] : ctx->rows.cache;
}

static inline backend *exec_layer_backend(const exec_ctx *ctx) {
	if (ctx->layer_be)
		return ctx->layer_be;
	if (ctx->li >= 0 && ctx->m && ctx->m->mixed_backend_mode)
		return model_layer_backend(ctx->m, ctx->li);
	return ctx->m->backend;
}

static inline uint32_t exec_caps(const exec_ctx *ctx, uint32_t (*resolve)(const backend *)) {
	return ctx->caps ? ctx->caps : (resolve(exec_layer_backend(ctx)) | EXEC_CAPS_RESOLVED);
}

static inline backend *op_backend_fallback(backend *a, const char *op_name) {
	op_fallback_note(a, op_name);
	ensure_sync(a);
	return op_host_backend();
}

static inline int ffn_down_native(const backend *a) {
	return a->matmul_ffn_down_batch && backend_has_cap(a, BCAP_MATMUL_FFN_DOWN);
}

static inline float *batch_buf_ptr(const buffer *b) {
	if (b->host_ptr)
		return (float *)((const char *)b->host_ptr + b->offset);
	return (float *)((char *)b->handle + b->offset);
}

#endif
