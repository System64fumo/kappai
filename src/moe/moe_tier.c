#include "moe/moe_tier.h"
#include "common.h"

#include <math.h>

#define POOL_RESERVE_FLOOR ((size_t)2 << 30)

moe_tier_policy moe_tier_policy_default(void) {
	return (moe_tier_policy){
		.promote_thr   = 0.30f,
		.demote_thr	   = 0.15f,
		.evict_thr	   = 0.03f,
		.decay		   = 0.98f,
		.min_residency = 32,
		.swap_margin   = 1.25f,
	};
}

bool moe_tier_policy_valid(const moe_tier_policy *p) {
	if (!(p->decay > 0.0f && p->decay < 1.0f))
		return false;
	if (!(p->evict_thr > 0.0f && p->demote_thr > p->evict_thr && p->promote_thr > p->demote_thr))
		return false;
	return p->min_residency >= 0 && p->swap_margin >= 1.0f;
}

float moe_tier_score_at(const moe_tier_policy *p, const moe_tier_score *s, uint64_t token) {
	if (token <= s->last_hit)
		return s->score;
	return s->score * powf(p->decay, (float)(token - s->last_hit));
}

void moe_tier_score_hit(const moe_tier_policy *p, moe_tier_score *s, uint64_t token) {
	s->score	= MIN(1.0f, moe_tier_score_at(p, s, token) + (1.0f - p->decay));
	s->last_hit = token;
}

moe_tier_action moe_tier_decide(const moe_tier_policy *p, float score, int tier, int top_tier) {
	if (tier < top_tier && score >= p->promote_thr)
		return MOE_TIER_PROMOTE;
	if (tier > 0 && score < p->demote_thr)
		return MOE_TIER_DEMOTE;
	if (tier == 0 && score < p->evict_thr)
		return MOE_TIER_EVICT;
	return MOE_TIER_STAY;
}

bool moe_tier_may_move(const moe_tier_policy *p, uint64_t last_move, uint64_t token) {
	if (last_move == 0)
		return true;
	return token >= last_move && token - last_move >= (uint64_t)p->min_residency;
}

bool moe_tier_beats(const moe_tier_policy *p, float challenger, float incumbent) {
	return challenger > incumbent * p->swap_margin;
}

size_t moe_tier_pool_budget(size_t available) {
	size_t reserve = MAX(POOL_RESERVE_FLOOR, available / 10);
	return available > reserve ? available - reserve : 0;
}

int moe_tier_fit(size_t budget, size_t layer_bytes, int n_experts) {
	if (layer_bytes == 0)
		return n_experts;
	return (int)MIN(budget / layer_bytes, (size_t)n_experts);
}

int moe_tier_pool_capacity(const mem_pool *p, size_t layer_bytes, int requested) {
	size_t budget = moe_tier_pool_budget(mem_pool_available(p));
	return MIN(requested, moe_tier_fit(budget, layer_bytes, requested));
}
