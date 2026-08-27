#include "test_core.h"

#include <math.h>

#define N_DRAWS 10000

static const float PROBS8[8] = {0.5f, 0.2f, 0.15f, 0.1f, 0.03f, 0.01f, 0.007f, 0.003f};

static void logits_from_probs(float *lg, const float *probs, int n) {
	for (int i = 0; i < n; i++)
		lg[i] = logf(probs[i]);
}

typedef struct {
	int32_t draws[N_DRAWS];
	int		n;
	int		seen[16];
} draw_set;

static void draw_set_run(draw_set *ds, uint64_t seed, const float *logits, int vocab, float temp,
						 int top_k, float top_p, float min_p, int ndraws) {
	sampler s;
	sampler_init(&s, seed);
	sampler_set_vocab(&s, vocab);
	sampler_set_params(&s, temp, top_k, top_p, min_p, 1.0f, 64);
	if (ndraws > N_DRAWS)
		ndraws = N_DRAWS;
	memset(ds->seen, 0, sizeof(ds->seen));
	ds->n = ndraws;
	for (int i = 0; i < ndraws; i++) {
		ds->draws[i] = sampler_sample(&s, logits, vocab);
		if (ds->draws[i] >= 0 && ds->draws[i] < 16)
			ds->seen[ds->draws[i]]++;
	}
	sampler_free(&s);
}

static int support_eq(const draw_set *ds, const int *want, int n_want) {
	for (int i = 0; i < 16; i++) {
		int should = 0;
		for (int j = 0; j < n_want; j++)
			if (want[j] == i)
				should = 1;
		if (should != (ds->seen[i] > 0))
			return 0;
	}
	return 1;
}

static void test_top_k_exact_support(void) {
	float lg[8];
	logits_from_probs(lg, PROBS8, 8);

	draw_set ds;
	draw_set_run(&ds, 42ULL, lg, 8, 1.0f, 3, 1.0f, 0.0f, N_DRAWS);

	const int want[3] = {0, 1, 2};
	int		  ok =
		support_eq(&ds, want, 3) && ds.seen[0] > 4500 && ds.seen[1] > 1500 && ds.seen[2] > 1000;
	char detail[256];
	snprintf(detail, sizeof(detail),
			 "top_k=3 support={%d,%d,%d} freq=[%d,%d,%d] of %d (expect ~%.2f/%.2f/%.2f)",
			 ds.seen[0] > 0, ds.seen[1] > 0, ds.seen[2] > 0, ds.seen[0], ds.seen[1], ds.seen[2],
			 ds.n, 0.588, 0.235, 0.176);
	record_result(OPFAM_SAMPLER, "top_k_keeps_exactly_k", ok ? V_PASS : V_FAIL, detail);
}

static void test_top_p_cut(void) {
	float lg[8];
	logits_from_probs(lg, PROBS8, 8);
	char detail[256];

	draw_set ds;
	draw_set_run(&ds, 7ULL, lg, 8, 1.0f, 0, 0.71f, 0.0f, N_DRAWS);
	const int want3[3] = {0, 1, 2};
	int		  ok3	   = support_eq(&ds, want3, 3);
	snprintf(detail, sizeof(detail), "top_p=0.71 support={0,1,2} seen=[%d,%d,%d]", ds.seen[0],
			 ds.seen[1], ds.seen[2]);
	record_result(OPFAM_SAMPLER, "top_p_cut_includes_crossing_token", ok3 ? V_PASS : V_FAIL,
				  detail);

	draw_set ds2;
	draw_set_run(&ds2, 7ULL, lg, 8, 1.0f, 0, 0.45f, 0.0f, N_DRAWS);
	const int want1[1] = {0};
	int		  ok1	   = support_eq(&ds2, want1, 1) && ds2.seen[0] == ds2.n;
	snprintf(detail, sizeof(detail), "top_p=0.45 all %d draws hit argmax (seen0=%d)", ds2.n,
			 ds2.seen[0]);
	record_result(OPFAM_SAMPLER, "top_p_below_first_mass_degenerates", ok1 ? V_PASS : V_FAIL,
				  detail);
}

static void test_min_p_threshold(void) {
	float lg[8];
	logits_from_probs(lg, PROBS8, 8);
	char detail[256];

	draw_set ds;
	draw_set_run(&ds, 99ULL, lg, 8, 1.0f, 0, 1.0f, 0.35f, N_DRAWS);
	const int want2[2] = {0, 1};
	int		  ok	   = support_eq(&ds, want2, 2);
	snprintf(detail, sizeof(detail), "min_p=0.35 floor=%.3f support={0,1} seen=[%d,%d]",
			 0.35f * 0.5f, ds.seen[0], ds.seen[1]);
	record_result(OPFAM_SAMPLER, "min_p_threshold_vs_global_max", ok ? V_PASS : V_FAIL, detail);

	draw_set ds3;
	draw_set_run(&ds3, 99ULL, lg, 8, 1.0f, 0, 1.0f, 0.21f, N_DRAWS);
	const int want3[3] = {0, 1, 2};
	ok				   = support_eq(&ds3, want3, 3);
	snprintf(detail, sizeof(detail), "min_p=0.21 floor=%.3f support={0,1,2} seen=[%d,%d,%d]",
			 0.21f * 0.5f, ds3.seen[0], ds3.seen[1], ds3.seen[2]);
	record_result(OPFAM_SAMPLER, "min_p_boundary_token_included", ok ? V_PASS : V_FAIL, detail);
}

static void test_temperature_monotonic(void) {
	float	 lg[2] = {1.0f, 0.0f};
	draw_set d05, d10, d20;
	draw_set_run(&d05, 1234ULL, lg, 2, 0.5f, 0, 1.0f, 0.0f, N_DRAWS);
	draw_set_run(&d10, 1234ULL, lg, 2, 1.0f, 0, 1.0f, 0.0f, N_DRAWS);
	draw_set_run(&d20, 1234ULL, lg, 2, 2.0f, 0, 1.0f, 0.0f, N_DRAWS);

	float f05 = (float)d05.seen[0] / d05.n;
	float f10 = (float)d10.seen[0] / d10.n;
	float f20 = (float)d20.seen[0] / d20.n;

	int	 ok = (f05 > 0.85f && f05 < 0.91f) && (f10 > 0.70f && f10 < 0.76f) &&
			  (f20 > 0.59f && f20 < 0.65f) && (f05 > f10 && f10 > f20);
	char detail[256];
	snprintf(detail, sizeof(detail),
			 "freq(id0): T=0.5 %.3f [0.85..0.91], T=1 %.3f [0.70..0.76], T=2 %.3f [0.59..0.65]",
			 f05, f10, f20);
	record_result(OPFAM_SAMPLER, "temperature_monotonicity", ok ? V_PASS : V_FAIL, detail);
}

static void test_argmax_identity(void) {
	float lg[8];
	logits_from_probs(lg, PROBS8, 8);
	char detail[256];

	draw_set ds;
	draw_set_run(&ds, 5ULL, lg, 8, 0.0f, 40, 0.9f, 0.1f, 500);
	int ok = ds.seen[0] == 500 && ds.n == 500;
	snprintf(detail, sizeof(detail), "temp=0: %d/%d draws are argmax id0", ds.seen[0], ds.n);
	record_result(OPFAM_SAMPLER, "temperature_zero_is_argmax", ok ? V_PASS : V_FAIL, detail);

	draw_set ds2;
	draw_set_run(&ds2, 5ULL, lg, 8, 0.8f, 1, 0.9f, 0.1f, 500);
	ok = ds2.seen[0] == 500;
	snprintf(detail, sizeof(detail), "top_k=1: %d/%d draws are argmax id0", ds2.seen[0], ds2.n);
	record_result(OPFAM_SAMPLER, "top_k_one_is_argmax", ok ? V_PASS : V_FAIL, detail);

	float	 tied[4] = {1.0f, 1.0f, 1.0f, 1.0f};
	draw_set ds3;
	draw_set_run(&ds3, 5ULL, tied, 4, 0.0f, 0, 1.0f, 0.0f, 64);
	ok = ds3.seen[0] == 64;
	snprintf(detail, sizeof(detail), "all-tie temp=0: id0 chosen %d/64", ds3.seen[0]);
	record_result(OPFAM_SAMPLER, "argmax_tie_breaks_lowest_index", ok ? V_PASS : V_FAIL, detail);
}

static void test_seed_reproducibility(void) {
	float lg[8];
	logits_from_probs(lg, PROBS8, 8);
	char detail[256];

	draw_set a, b, c;
	draw_set_run(&a, 2024ULL, lg, 8, 0.9f, 0, 1.0f, 0.0f, 2000);
	draw_set_run(&b, 2024ULL, lg, 8, 0.9f, 0, 1.0f, 0.0f, 2000);
	draw_set_run(&c, 777ULL, lg, 8, 0.9f, 0, 1.0f, 0.0f, 2000);

	int same = memcmp(a.draws, b.draws, sizeof(a.draws[0]) * (size_t)a.n) == 0;
	int diff = 0;
	for (int i = 0; i < a.n; i++)
		if (a.draws[i] != c.draws[i]) {
			diff = 1;
			break;
		}
	snprintf(detail, sizeof(detail), "seed2024 twice identical=%d; seed777 differs=%d", same, diff);
	record_result(OPFAM_SAMPLER, "seed_reproducibility", (same && diff) ? V_PASS : V_FAIL, detail);
}

static void test_nofilter_fastpath_agreement(void) {
	float lg[8];
	logits_from_probs(lg, PROBS8, 8);
	char detail[256];

	draw_set fast, filt;
	draw_set_run(&fast, 31337ULL, lg, 8, 1.0f, 0, 1.0f, 0.0f, 5000);
	draw_set_run(&filt, 31337ULL, lg, 8, 1.0f, 7, 1.0f, 0.0f, 5000);

	int				   ok		 = 1;
	float			   worst	 = 0;
	static const float expect[8] = {0.5f, 0.2f, 0.15f, 0.1f, 0.03f, 0.01f, 0.007f, 0.0f};
	for (int i = 0; i < 8; i++) {
		float ff = (float)fast.seen[i] / fast.n;
		float gf = (float)filt.seen[i] / filt.n;
		float d	 = fabsf(ff - gf);
		if (d > worst)
			worst = d;
		if (ff - expect[i] > 0.02f || expect[i] - ff > 0.02f)
			ok = 0;
		if (i < 7 && (gf - expect[i] > 0.02f || expect[i] - gf > 0.02f))
			ok = 0;
		if (i == 7 && filt.seen[i] != 0)
			ok = 0;
	}
	snprintf(detail, sizeof(detail),
			 "fast vs filtered max|dfreq|=%.4f; fast freq dev from theory <= band; "
			 "filtered never emits idx7 (%d)",
			 worst, filt.seen[7]);
	record_result(OPFAM_SAMPLER, "nofilter_fastpath_matches_filtered_path", ok ? V_PASS : V_FAIL,
				  detail);
}

static void test_repeat_penalty_flip(void) {
	char  detail[256];
	float lg[4] = {1.0f, 0.9f, 0.0f, -1.0f};

	sampler s;
	sampler_init(&s, 11ULL);
	sampler_set_vocab(&s, 4);
	sampler_set_params(&s, 0.0f, 40, 0.9f, 0.0f, 1.0f, 8);
	int32_t plain = sampler_sample(&s, lg, 4);
	sampler_observe(&s, 0);
	int32_t still0 = sampler_sample(&s, lg, 4);
	sampler_free(&s);

	sampler s2;
	sampler_init(&s2, 11ULL);
	sampler_set_vocab(&s2, 4);
	sampler_set_params(&s2, 0.0f, 40, 0.9f, 0.0f, 2.0f, 8);
	sampler_observe(&s2, 0);
	int32_t flipped = sampler_sample(&s2, lg, 4);
	sampler_free(&s2);

	int ok = plain == 0 && still0 == 0 && flipped == 1;
	snprintf(detail, sizeof(detail), "plain=%d re-draw(no penalty)=%d penalized=%d (expect 0,0,1)",
			 plain, still0, flipped);
	record_result(OPFAM_SAMPLER, "repeat_penalty_flips_greedy_choice", ok ? V_PASS : V_FAIL,
				  detail);
}

static void test_topk_helper_exact(void) {
	float lg[8];
	logits_from_probs(lg, PROBS8, 8);
	sampler_top_k_entry out[8];
	int					kept = sampler_top_k(lg, 8, 4, out);

	int ok = kept == 4;
	for (int i = 0; i < kept && ok; i++) {
		if (out[i].i != i)
			ok = 0;
		if (i > 0 && out[i].v > out[i - 1].v)
			ok = 0;
	}
	char detail[256];
	snprintf(detail, sizeof(detail), "kept=%d ids=[%d,%d,%d,%d] desc-sorted=%d", kept, out[0].i,
			 out[1].i, out[2].i, out[3].i, ok);
	record_result(OPFAM_SAMPLER, "sampler_top_k_helper_exact", ok ? V_PASS : V_FAIL, detail);
}

void run_sampler_tests(void) {
	test_top_k_exact_support();
	test_top_p_cut();
	test_min_p_threshold();
	test_temperature_monotonic();
	test_argmax_identity();
	test_seed_reproducibility();
	test_nofilter_fastpath_agreement();
	test_repeat_penalty_flip();
	test_topk_helper_exact();
}
