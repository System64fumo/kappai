#ifndef COMPUTE_H
#define COMPUTE_H

#include <signal.h>

#include "backend/backend.h"
#include "common.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "moe/moe_stream.h"
#include "profile.h"
#include "recipe.h"

typedef void (*layer_progress_cb)(int layer_idx, int n_layers, int token_progress, int batched,
								  void *ud);

typedef struct compute_scratch {
	buffer		 router_softmax_inp_gpu;
	buffer		 router_logits_gpu;
	int			 router_ids_host[64];
	float		 router_w_host[64];
	float		*rope_cos, *rope_sin, *logits_host;
	float		*rope_cos_swa, *rope_sin_swa;
	buffer		 ple_inp;
	buffer		 ple_slice;
	buffer		 ple_all;
	buffer		 ple_proj_gpu;
	buffer		 ple_proj_norm_w_gpu;
	int			 ple_proj_norm_w_uploaded;
	float		*ple_buf;
	backend		*backend;
	int			 allocated_n_ctx;
	const model *last_model;
	int			 last_n_ctx;
	profile		 prof;

	buffer slots[RECIPE_SLOT_MAX];

	buffer	 mirror_slots[RECIPE_SLOT_MAX];
	int		 mirror_slots_alloced;
	backend *mirror_backend;
	backend *active_backend;
	int		 active_is_mirror;
	float	*transfer_buf;
	size_t	 transfer_buf_cap;

	float *ple_inp_host;
	float *cur_host;
	float *out_proj_host;
	float *vf_host;
	float *xf_host;
	int	   small_host_dim;
	int	   small_host_intermediate;
	int	   small_host_kv_out;

	float_buf ple_proj_host;
	float_buf inpL_host;

	float_buf moe_all_scratch;
	float_buf moe_all_outs;
	float_buf moe_scratch;
	float_buf moe_xb_f;
	float_buf moe_shared_y;
	float_buf qwen_host;

	moe_expert_slot moe_slot_buf[512];

	layer_progress_cb			 layer_cb;
	void						*layer_cb_ud;
	const volatile sig_atomic_t *interrupt;

	batch_scratch *bs;
	float_buf	   batch_logits_tmp;
} compute_scratch;

static inline int compute_model_changed(const compute_scratch *s, const model *m, int n_ctx) {
	return s->last_model != m || s->allocated_n_ctx < n_ctx;
}

void		compute_scratch_init(compute_scratch *s);
void		compute_scratch_free(compute_scratch *s);
status_code compute_scratch_ensure(compute_scratch *s, const model *m, int n_ctx);

void compute_small_host_ensure(compute_scratch *s, int dim, int intermediate, int kv_out);

status_code compute_forward(model *m, kvcache *cache, compute_scratch *s, int token, int pos,
							int flash_attn, float *logits_out);

status_code compute_forward_batch(model *m, kvcache *cache, compute_scratch *s,
								  const int32_t *tokens, int n_tokens, int pos_start,
								  int flash_attn, float *logits_out);

void compute_set_layer_progress_cb(compute_scratch *s, layer_progress_cb cb, void *ud);

status_code compute_scratch_ensure_mirror(compute_scratch *s, const model *m, int n_ctx);
status_code compute_switch_active_backend(compute_scratch *s, backend *target, int dim);
status_code compute_copy_buffer_cross(compute_scratch *s, const buffer *src, buffer *dst, int n);
static inline backend *compute_active_backend(compute_scratch *s) {
	return s->active_backend ? s->active_backend : s->backend;
}

static inline buffer *compute_slots_array(compute_scratch *s) {
	return s->active_is_mirror ? s->mirror_slots : s->slots;
}

#endif
