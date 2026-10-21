#include "test_core.h"
#include "test_synth_gguf.h"

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	model			m;
	kvcache			kv;
	compute_scratch s;
	backend		   *be;
	int				ok;
	float		   *logits;
	int				vocab;
} moe_sess;

static void moe_load(moe_sess *r, const char *path, int n_ctx) {
	memset(r, 0, sizeof(*r));
	if (backend_create("cpu", 0, &r->be) != OK)
		abort();
	if (model_load_backend_ex_repack(&r->m, path, r->be, 1, NULL) != OK)
		abort();
	r->vocab  = r->m.vocab_size;
	r->logits = xmalloc((size_t)r->vocab * sizeof(float));
	if (kvcache_init(&r->kv, &r->m, n_ctx, KV_QUANT_F16) != OK)
		abort();
	compute_scratch_init(&r->s);
	if (compute_scratch_ensure(&r->s, &r->m, n_ctx) != OK)
		abort();
	r->ok = 1;
}

static void moe_free(moe_sess *r) {
	if (!r->ok)
		return;
	compute_scratch_free(&r->s);
	kvcache_free(&r->kv);
	free(r->logits);
	model_free(&r->m);
	backend_destroy(r->be);
	r->ok = 0;
}

#define MOE_NPREFILL 12
#define MOE_NDECODE 4

static status_code moe_run(moe_sess *r, const int32_t *toks, float logits_out[MOE_NDECODE + 1][64],
						   int32_t *chain) {
	status_code st =
		compute_forward_batch(&r->m, &r->kv, &r->s, toks, MOE_NPREFILL, 0, 0, r->logits);
	r->kv.n_pos += MOE_NPREFILL;
	if (st != OK)
		return st;
	memcpy(logits_out[0], r->logits, (size_t)r->vocab * sizeof(float));
	for (int i = 0; i < MOE_NDECODE; i++) {
		int32_t tok = sampler_argmax(r->logits, r->vocab);
		if (chain)
			chain[i] = tok;
		st = compute_forward(&r->m, &r->kv, &r->s, tok, r->kv.n_pos, 0, r->logits);
		r->kv.n_pos++;
		if (st != OK)
			return st;
		memcpy(logits_out[i + 1], r->logits, (size_t)r->vocab * sizeof(float));
	}
	return OK;
}

void run_moe_stream_tests(void) {
	char detail[512];
	synth_suite_common_init();

	config saved_cfg = *(const config *)config_get();
	synth_apply_config();

	moe_sess streamed, resident;
	moe_load(&streamed, synth_dsa_model_path, 128);
	moe_load(&resident, synth_dsa_model_path, 128);

	int have_stream_cfg = streamed.m.moe_cache != NULL && streamed.m.moe_stream_enabled == 1;
	if (!have_stream_cfg) {
		snprintf(detail, sizeof(detail),
				 "streaming cache not active on this build/config "
				 "(cache=%p enabled=%d); skipping",
				 (void *)streamed.m.moe_cache, streamed.m.moe_stream_enabled);
		record_result(OPFAM_MOE_STREAM, "moe_stream.setup", V_SKIP, detail);
		moe_free(&streamed);
		moe_free(&resident);
		*((config *)config_get()) = saved_cfg;
		return;
	}
	resident.m.moe_stream_enabled = 0;

	int32_t toks[MOE_NPREFILL];
	tsg_seed(0xE7EA75ULL);
	for (int i = 0; i < MOE_NPREFILL; i++)
		toks[i] = (int32_t)(tsg_rand() % (uint32_t)streamed.vocab);

	static float lg_s[MOE_NDECODE + 1][64];
	static float lg_r[MOE_NDECODE + 1][64];
	int32_t		 chain_s[MOE_NDECODE], chain_r[MOE_NDECODE];

	status_code st_s = moe_run(&streamed, toks, lg_s, chain_s);
	status_code st_r = moe_run(&resident, toks, lg_r, chain_r);

	int	  ok		 = (st_s == OK && st_r == OK);
	float worst		 = 0;
	int	  worst_step = -1;
	if (ok) {
		for (int step = 0; step <= MOE_NDECODE; step++) {
			for (int i = 0; i < streamed.vocab; i++) {
				float d = fabsf(lg_s[step][i] - lg_r[step][i]);
				if (d > worst) {
					worst	   = d;
					worst_step = step;
				}
			}
			if (count_nonfinite(lg_s[step], streamed.vocab) > 0 ||
				count_nonfinite(lg_r[step], streamed.vocab) > 0)
				ok = 0;
		}
		ok = ok && memcmp(chain_s, chain_r, sizeof(chain_s)) == 0 && worst <= 2e-3f;
	}

	moe_stream_stats stats = {0};
	moe_stream_get_stats(streamed.m.moe_cache, &stats);
	int evicted_enough = stats.requests > 0 && stats.cache_misses > 0;

	ok = ok && evicted_enough;
	snprintf(detail, sizeof(detail),
			 "cap=2<topk=4 forced evictions: req=%llu misses=%llu hits=%llu | "
			 "max|dlogits|=%.3e @step %d; greedy chains equal=%d",
			 (unsigned long long)stats.requests, (unsigned long long)stats.cache_misses,
			 (unsigned long long)(stats.cache_hits + stats.lru_hits + stats.pin_hits), worst,
			 worst_step, memcmp(chain_s, chain_r, sizeof(chain_s)) == 0);
	record_result(OPFAM_MOE_STREAM, "streamed_eviction_equals_fully_resident", ok ? V_PASS : V_FAIL,
				  detail);

	{
		int last_moe_layer = -1;
		for (int li = streamed.m.n_layers - 1; li >= 0; li--) {
			if (streamed.m.layers[li].experts) {
				last_moe_layer = li;
				break;
			}
		}
		if (last_moe_layer < 0) {
			record_result(OPFAM_MOE_STREAM, "moe_stream.union_over_max_topk", V_SKIP,
						  "no MoE layer in synthetic model");
		} else {
			int				 n_k   = MOE_MAX_TOPK + 36;
			int				*ids   = xmalloc((size_t)n_k * sizeof(int));
			moe_expert_slot *slots = xcalloc((size_t)n_k, sizeof(*slots));
			int				 n_e   = streamed.m.moe.n_experts > 0 ? streamed.m.moe.n_experts : 1;
			for (int k = 0; k < n_k; k++)
				ids[k] = k % n_e;

			int ok_union = moe_stream_resolve(&streamed.m, last_moe_layer, ids, n_k, slots) == OK;
			if (ok_union) {
				for (int k = 0; k < n_k; k++) {
					if (slots[k].eid != ids[k] ||
						atomic_load_explicit(&slots[k].io_ready, memory_order_acquire) == 0) {
						ok_union = 0;
						break;
					}
				}
			}
			snprintf(detail, sizeof(detail),
					 "streamed cache: n_k=%d (> MOE_MAX_TOPK=%d) resolved ok=%d", n_k, MOE_MAX_TOPK,
					 ok_union);
			record_result(OPFAM_MOE_STREAM, "moe_stream.union_over_max_topk",
						  ok_union ? V_PASS : V_FAIL, detail);

			memset(slots, 0, (size_t)n_k * sizeof(*slots));
			int ok_nocache = moe_stream_resolve(&resident.m, last_moe_layer, ids, n_k, slots) == OK;
			if (ok_nocache) {
				for (int k = 0; k < n_k; k++) {
					if (slots[k].eid != ids[k] ||
						atomic_load_explicit(&slots[k].io_ready, memory_order_acquire) == 0) {
						ok_nocache = 0;
						break;
					}
				}
			}
			snprintf(detail, sizeof(detail),
					 "no-cache path: n_k=%d (> MOE_MAX_TOPK=%d) resolved ok=%d", n_k, MOE_MAX_TOPK,
					 ok_nocache);
			record_result(OPFAM_MOE_STREAM, "moe_stream.union_over_max_topk_nocache",
						  ok_nocache ? V_PASS : V_FAIL, detail);

			free(ids);
			free(slots);
		}
	}

	moe_free(&streamed);
	moe_free(&resident);
	*((config *)config_get()) = saved_cfg;
}
