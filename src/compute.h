#ifndef COMPUTE_H
#define COMPUTE_H

#include <signal.h>

#include "backend/backend.h"
#include "common.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "moe/moe_common.h"
#include "moe/moe_stream.h"
#include "profile.h"
#include "recipe.h"

typedef void (*layer_progress_cb)(int layer_idx, int n_layers, int token_progress, int batched,
								  void *ud);

#define SCRATCH_MAX_MIRRORS 7

typedef struct {
	backend *be;
	buffer	 slots[RECIPE_SLOT_MAX];
	buffer	 ple_inp;
	buffer	 ple_slice;
} scratch_mirror;

typedef struct compute_scratch {
	buffer		 router_softmax_inp;
	buffer		 router_logits;
	buffer		 moe_shared_y_dev;
	int			 router_ids_host[64];
	float		 router_w_host[64];
	float		*rope_cos, *rope_sin, *logits_host;
	int			 logits_alias;
	float		*rope_cos_swa, *rope_sin_swa;
	buffer		 ple_inp;
	buffer		 ple_slice;
	buffer		 ple_all;
	buffer		 ple_proj;
	buffer		 ple_proj_norm_w;
	int			 ple_proj_norm_w_uploaded;
	float		*ple_buf;
	backend		*backend;
	int			 allocated_n_ctx;
	const model *last_model;
	int			 last_n_ctx;
	profile		 prof;

	buffer slots[RECIPE_SLOT_MAX];

	scratch_mirror mirror[SCRATCH_MAX_MIRRORS];
	int			   n_mirrors;
	int			   mirrors_alloced;
	backend		  *active_backend;
	int			   active_dev;
	float		  *transfer_buf;
	size_t		   transfer_buf_cap;

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
	float_buf hybrid_host;
	float_buf hybrid_host2;
	float_buf hybrid_host3;
	float_buf gdn_ws_host;
	float_buf op_ws;
	float_buf gdn_z_host;
	float_buf gdn_alpha_host;
	float_buf gdn_beta_host;
	float_buf gdn_out_host;
	float_buf op_st_in;
	float_buf op_st_out;
	float_buf op_st_cur;
	float_buf op_st_coef;

	moe_expert_slot *moe_slot_buf;

	layer_progress_cb			 layer_cb;
	void						*layer_cb_ud;
	const volatile sig_atomic_t *interrupt;

	batch_scratch *bs;
	float_buf	   batch_logits_tmp;
} compute_scratch;

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

status_code compute_scratch_ensure_mirror(compute_scratch *s, const model *m);
status_code compute_switch_active_backend(compute_scratch *s, backend *target, const model *m);
status_code compute_copy_buffer_cross(compute_scratch *s, const buffer *src, buffer *dst, int n);
static inline backend *compute_active_backend(compute_scratch *s) {
	return s->active_backend ? s->active_backend : s->backend;
}

int compute_device_index(const compute_scratch *s, const backend *b);

static inline buffer *compute_dev_slots(compute_scratch *s, int dev) {
	return dev > 0 ? s->mirror[dev - 1].slots : s->slots;
}

static inline buffer *compute_slots_array(compute_scratch *s) {
	return compute_dev_slots(s, s->active_dev);
}

#endif
