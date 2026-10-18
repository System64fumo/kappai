#ifndef MOE_TIER_H
#define MOE_TIER_H

#include "mempool.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
	float promote_thr;
	float demote_thr;
	float evict_thr;
	float decay;
	int	  min_residency;
	float swap_margin;
} moe_tier_policy;

typedef struct {
	float	 score;
	uint64_t last_hit;
} moe_tier_score;

typedef struct {
	moe_tier_score score;
	uint64_t	   last_move;
} moe_tier_state;

typedef enum {
	MOE_TIER_STAY,
	MOE_TIER_PROMOTE,
	MOE_TIER_DEMOTE,
	MOE_TIER_EVICT,
} moe_tier_action;

moe_tier_policy moe_tier_policy_default(void);
bool			moe_tier_policy_valid(const moe_tier_policy *p);

float moe_tier_score_at(const moe_tier_policy *p, const moe_tier_score *s, uint64_t token);
void  moe_tier_score_hit(const moe_tier_policy *p, moe_tier_score *s, uint64_t token);

moe_tier_action moe_tier_decide(const moe_tier_policy *p, float score, int tier, int top_tier);
bool			moe_tier_may_move(const moe_tier_policy *p, uint64_t last_move, uint64_t token);
bool			moe_tier_beats(const moe_tier_policy *p, float challenger, float incumbent);

size_t moe_tier_pool_budget(size_t available);
int	   moe_tier_fit(size_t budget, size_t layer_bytes, int n_experts);
int	   moe_tier_pool_capacity(const mem_pool *p, size_t layer_bytes, int requested);

#endif
