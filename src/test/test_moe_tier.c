#include "moe/moe_tier.h"
#include "test_core.h"

#include <math.h>

#define record_tier(label, ok, ...) record_resultf(OPFAM_MOE_TIER, (label), (ok), __VA_ARGS__)

#define GB ((size_t)1 << 30)

static size_t g_available;

static size_t fake_available(backend *self) {
	(void)self;
	return g_available;
}

static mem_pool fake_pool(size_t available) {
	static backend b;
	g_available = available;
	b			= (backend){.mem_available = fake_available, .mem_total = fake_available};
	return (mem_pool){.capacity_source = &b};
}

static void test_default_policy_valid(void) {
	moe_tier_policy p = moe_tier_policy_default();
	record_tier("default_policy_valid", moe_tier_policy_valid(&p), "valid=%d",
				moe_tier_policy_valid(&p));
}

static void test_invalid_policies(void) {
	moe_tier_policy base = moe_tier_policy_default();
	moe_tier_policy a = base, b = base, c = base, d = base;
	a.demote_thr  = a.promote_thr;
	b.evict_thr	  = b.demote_thr;
	c.decay		  = 1.0f;
	d.swap_margin = 0.5f;
	int ok		  = !moe_tier_policy_valid(&a) && !moe_tier_policy_valid(&b) &&
					!moe_tier_policy_valid(&c) && !moe_tier_policy_valid(&d);
	record_tier("invalid_policies", ok, "a=%d b=%d c=%d d=%d", moe_tier_policy_valid(&a),
				moe_tier_policy_valid(&b), moe_tier_policy_valid(&c), moe_tier_policy_valid(&d));
}

static void test_score_converges_to_hit_rate(void) {
	moe_tier_policy p = moe_tier_policy_default();
	moe_tier_score	s = {0};
	for (uint64_t t = 1; t <= 2000; t++)
		moe_tier_score_hit(&p, &s, t);
	float steady = moe_tier_score_at(&p, &s, 2000);
	record_tier("score_saturates_on_every_token_hit", fabsf(steady - 1.0f) < 0.01f, "score=%.4f",
				(double)steady);
}

static void test_score_matches_per_token_ewma(void) {
	moe_tier_policy p	   = moe_tier_policy_default();
	moe_tier_score	s	   = {0};
	float			ref	   = 0.0f;
	uint64_t		hits[] = {1, 2, 5, 9, 10, 40};
	size_t			next   = 0;
	for (uint64_t t = 1; t <= 40; t++) {
		int hit = next < sizeof(hits) / sizeof(hits[0]) && hits[next] == t;
		ref		= p.decay * ref + (hit ? 1.0f - p.decay : 0.0f);
		if (hit) {
			moe_tier_score_hit(&p, &s, t);
			next++;
		}
	}
	float lazy = moe_tier_score_at(&p, &s, 40);
	record_tier("score_matches_per_token_ewma", fabsf(lazy - ref) < 1e-5f, "lazy=%.6f ref=%.6f",
				(double)lazy, (double)ref);
}

static void test_score_decays_without_hits(void) {
	moe_tier_policy p	  = moe_tier_policy_default();
	moe_tier_score	s	  = {.score = 0.5f, .last_hit = 100};
	float			later = moe_tier_score_at(&p, &s, 300);
	float			same  = moe_tier_score_at(&p, &s, 100);
	record_tier("score_decays_without_hits", later < p.evict_thr && same == 0.5f,
				"later=%.5f same=%.3f", (double)later, (double)same);
}

static void test_decide_thresholds(void) {
	moe_tier_policy p  = moe_tier_policy_default();
	int				ok = 1;
	ok &= moe_tier_decide(&p, p.promote_thr, 0, 1) == MOE_TIER_PROMOTE;
	ok &= moe_tier_decide(&p, p.promote_thr, 1, 1) == MOE_TIER_STAY;
	ok &= moe_tier_decide(&p, p.demote_thr - 0.01f, 1, 1) == MOE_TIER_DEMOTE;
	ok &= moe_tier_decide(&p, p.evict_thr - 0.01f, 0, 1) == MOE_TIER_EVICT;
	ok &= moe_tier_decide(&p, p.evict_thr - 0.01f, 1, 1) == MOE_TIER_DEMOTE;
	ok &= moe_tier_decide(&p, p.evict_thr - 0.01f, 0, 0) == MOE_TIER_EVICT;
	ok &= moe_tier_decide(&p, 1.0f, 0, 0) == MOE_TIER_STAY;
	record_tier("decide_thresholds", ok, "ok=%d", ok);
}

static void test_decide_hysteresis_band(void) {
	moe_tier_policy p	= moe_tier_policy_default();
	float			mid = (p.promote_thr + p.demote_thr) * 0.5f;
	float			low = (p.demote_thr + p.evict_thr) * 0.5f;
	int				ok	= moe_tier_decide(&p, mid, 0, 1) == MOE_TIER_STAY &&
						  moe_tier_decide(&p, mid, 1, 1) == MOE_TIER_STAY &&
						  moe_tier_decide(&p, low, 0, 1) == MOE_TIER_STAY;
	record_tier("decide_hysteresis_band", ok, "ok=%d", ok);
}

static void test_decide_single_step_tiers(void) {
	moe_tier_policy p  = moe_tier_policy_default();
	int				ok = moe_tier_decide(&p, 0.9f, 1, 2) == MOE_TIER_PROMOTE &&
						 moe_tier_decide(&p, 0.0f, 2, 2) == MOE_TIER_DEMOTE &&
						 moe_tier_decide(&p, 0.0f, 1, 2) == MOE_TIER_DEMOTE;
	record_tier("decide_single_step_tiers", ok, "ok=%d", ok);
}

static void test_min_residency_blocks_early_moves(void) {
	moe_tier_policy p  = moe_tier_policy_default();
	int				ok = moe_tier_may_move(&p, 0, 1) && !moe_tier_may_move(&p, 10, 10) &&
						 !moe_tier_may_move(&p, 10, 10 + (uint64_t)p.min_residency - 1) &&
						 moe_tier_may_move(&p, 10, 10 + (uint64_t)p.min_residency);
	record_tier("min_residency_blocks_early_moves", ok, "ok=%d", ok);
}

static void test_swap_margin_requires_clear_win(void) {
	moe_tier_policy p = moe_tier_policy_default();
	int				ok =
		!moe_tier_beats(&p, 0.40f, 0.40f) && !moe_tier_beats(&p, 0.40f * p.swap_margin, 0.40f) &&
		moe_tier_beats(&p, 0.40f * p.swap_margin + 0.01f, 0.40f) && moe_tier_beats(&p, 0.01f, 0.0f);
	record_tier("swap_margin_requires_clear_win", ok, "ok=%d", ok);
}

static void test_pool_budget(void) {
	int ok = moe_tier_pool_budget(1 * GB) == 0 && moe_tier_pool_budget(2 * GB) == 0 &&
			 moe_tier_pool_budget(10 * GB) == 8 * GB && moe_tier_pool_budget(100 * GB) == 90 * GB;
	record_tier("pool_budget", ok, "10GB->%zu 100GB->%zu", moe_tier_pool_budget(10 * GB),
				moe_tier_pool_budget(100 * GB));
}

static void test_fit_clamps_to_experts(void) {
	int ok = moe_tier_fit(100 * GB, 1 * GB, 64) == 64 && moe_tier_fit(10 * GB, 1 * GB, 64) == 10 &&
			 moe_tier_fit(0, 1 * GB, 64) == 0 && moe_tier_fit(1 * GB, 0, 64) == 64;
	record_tier("fit_clamps_to_experts", ok, "ok=%d", ok);
}

static void test_pool_capacity_follows_available(void) {
	mem_pool small = fake_pool(10 * GB);
	int		 a	   = moe_tier_pool_capacity(&small, 1 * GB, 64);
	int		 b	   = moe_tier_pool_capacity(&small, 1 * GB, 4);
	mem_pool big   = fake_pool(100 * GB);
	int		 c	   = moe_tier_pool_capacity(&big, 1 * GB, 64);
	record_tier("pool_capacity_follows_available", a == 8 && b == 4 && c == 64, "a=%d b=%d c=%d", a,
				b, c);
}

void run_moe_tier_tests(void) {
	test_default_policy_valid();
	test_invalid_policies();
	test_score_converges_to_hit_rate();
	test_score_matches_per_token_ewma();
	test_score_decays_without_hits();
	test_decide_thresholds();
	test_decide_hysteresis_band();
	test_decide_single_step_tiers();
	test_min_residency_blocks_early_moves();
	test_swap_margin_requires_clear_win();
	test_pool_budget();
	test_fit_clamps_to_experts();
	test_pool_capacity_follows_available();
}
