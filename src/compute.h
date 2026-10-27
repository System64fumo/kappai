#ifndef COMPUTE_H
#define COMPUTE_H

#include <signal.h>
#include <stdatomic.h>

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

#define EXPERT_DEV_MAX 8

#define MOE_STAGE_RING 3

typedef struct {
	buffer x;
	buffer out;
	size_t cap;
} expert_dev_entry;

typedef struct {
	expert_dev_entry entries[EXPERT_DEV_MAX];
	float_buf		 acc;
	float_buf		 tmp;
} expert_dev_scratch;

typedef struct {
	backend		  *be;
	batch_scratch *bs;
} scratch_mirror;

typedef struct {
	float		*rope_cos, *rope_sin;
	float		*rope_cos_swa, *rope_sin_swa;
	buffer		 ple_proj_norm_w;
	int			 ple_proj_norm_w_uploaded;
	backend		*backend;
	int			 allocated_n_ctx;
	const model *last_model;
} scratch_model;

typedef struct {
	float	 *transfer_buf;
	size_t	  transfer_buf_cap;
	float_buf op_ws;
	float_buf stage_a;
	float_buf stage_b;
	float_buf scale_tmp;
	uint8_t	 *last_only;
	size_t	  last_only_cap;
} scratch_run;

typedef struct {
	float_buf		   all_scratch;
	float_buf		   all_outs;
	float_buf		   scratch;
	float_buf		   xb_f;
	buffer			   stage[MOE_STAGE_RING];
	int				   stage_next;
	moe_expert_slot	  *slot_buf;
	moe_expert_slot	  *slot_dev;
	expert_dev_scratch expert_dev;
} scratch_moe;

typedef struct {
	float_buf host;
	float_buf host2;
	float_buf gdn_ws;
	float_buf gdn_z;
	float_buf gdn_alpha;
	float_buf gdn_beta;
	float_buf gdn_out;
} scratch_hybrid;

typedef struct {
	float_buf proj_host;
	float_buf inpL_host;
} scratch_ple;

typedef struct compute_scratch {
	scratch_model  model;
	scratch_run	   run;
	scratch_moe	   moe;
	scratch_hybrid hybrid;
	scratch_ple	   ple;

	scratch_mirror mirror[SCRATCH_MAX_MIRRORS];
	int			   n_mirrors;
	int			   mirrors_alloced;
	batch_scratch *bs;

	float  *logits_host;
	profile prof;

	layer_progress_cb			layer_cb;
	void					   *layer_cb_ud;
	const _Atomic sig_atomic_t *interrupt;
} compute_scratch;

void			  compute_scratch_init(compute_scratch *s);
void			  compute_scratch_free(compute_scratch *s);
void			  expert_dev_scratch_release(expert_dev_scratch *ds);
expert_dev_entry *expert_dev_scratch_get(expert_dev_scratch *ds, backend *eb, size_t floats);
status_code		  compute_scratch_ensure(compute_scratch *s, const model *m, int n_ctx);

status_code compute_forward(model *m, kvcache *cache, compute_scratch *s, int token, int pos,
							int flash_attn, float *logits_out);

status_code compute_forward_batch(model *m, kvcache *cache, compute_scratch *s,
								  const int32_t *tokens, int n_tokens, int pos_start,
								  int flash_attn, float *logits_out);

status_code compute_forward_decode_batch(model *m, kvcache **caches, compute_scratch *s,
										 const int32_t *tokens, const int *positions,
										 const uint8_t *needs_logits, int n_seqs, int flash_attn,
										 float *logits_out);

void compute_set_layer_progress_cb(compute_scratch *s, layer_progress_cb cb, void *ud);

status_code compute_scratch_ensure_mirror(compute_scratch *s, const model *m);
status_code compute_copy_buffer_cross(compute_scratch *s, const buffer *src, buffer *dst, int n);

int compute_device_index(const compute_scratch *s, const backend *b);

#endif
