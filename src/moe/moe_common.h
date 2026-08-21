#ifndef MOE_COMMON_H
#define MOE_COMMON_H

#include "backend/backend.h"
#include "common.h"
#include "moe/moe_stream.h"
#include "recipe.h"

int	 moe_topk_select(const float *scores, int n_experts, int top_k, int *top_idx, float *top_score);
void moe_apply_weights(float *weight, int n_k, int norm_topk, float routed_scale);
void moe_activate(float *act, const float *gate, const float *up, int n_i, float gs, float us,
				  int use_gelu);

typedef struct {
	backend		 *a;
	int			  inter;
	int			  dim;
	int			  use_gelu;
	int			  q8_gate_ok;
	uint32_t	  q8_gate_type;
	const buffer *xb_q8_gate;
	float		 *scratch;
	float		 *out;
} moe_expert_ctx;

status_code moe_expert_exec(moe_expert_ctx *cx, const moe_expert_slot *es, const float *xb,
							float weight, int tid);
void moe_router_normalize_input(const struct model *m, const struct layer_weights *L, int dim,
								float *router_input);
int moe_router_emit(int E, int K, int use_softmax, int norm_topk, float routed_scale, float *logits,
					const float *bias, float *scores_scratch, int *ids_out, float *w_out);

#define MOE_MAX_TOPK 64

#endif