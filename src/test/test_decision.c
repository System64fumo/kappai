#include "decision.h"
#include "test_core.h"
#include "test_synth_gguf.h"

#include <unistd.h>

static void test_full_attention(backend *b) {
	enum { ROWS = 5, HEADS = 4, KV = 2, DIM = 8 };
	float q[ROWS * HEADS * DIM], k[ROWS * KV * DIM], v[ROWS * KV * DIM];
	float result[ROWS * HEADS * DIM], expected[ROWS * HEADS * DIM];
	for (size_t i = 0; i < ARRAY_LEN(q); i++)
		q[i] = (float)((int)(i % 19) - 9) / 8.0f;
	for (size_t i = 0; i < ARRAY_LEN(k); i++) {
		k[i] = (float)((int)(i % 13) - 6) / 8.0f;
		v[i] = (float)((int)(i % 17) - 8) / 8.0f;
	}
	for (int row = 0; row < ROWS; row++) {
		for (int h = 0; h < HEADS; h++) {
			double weights[ROWS], sum = 0.0;
			for (int t = 0; t < ROWS; t++) {
				double dot = 0.0;
				for (int d = 0; d < DIM; d++)
					dot += q[(row * HEADS + h) * DIM + d] * k[(t * KV + h / 2) * DIM + d];
				weights[t] = exp(dot / sqrt(DIM));
				sum += weights[t];
			}
			for (int d = 0; d < DIM; d++) {
				double out = 0.0;
				for (int t = 0; t < ROWS; t++)
					out += weights[t] / sum * v[(t * KV + h / 2) * DIM + d];
				expected[(row * HEADS + h) * DIM + d] = (float)out;
			}
		}
	}
	buffer qb = buffer_host_view(b, q, sizeof(q));
	buffer kb = buffer_host_view(b, k, sizeof(k));
	buffer vb = buffer_host_view(b, v, sizeof(v));
	buffer ob = buffer_host_view(b, result, sizeof(result));
	for (int flash = 0; flash <= 1; flash++) {
		status_code st = b->attention_full_batch(b, &qb, &kb, &vb, &ob, HEADS, KV, DIM, ROWS, flash,
												 1.0f / sqrtf(DIM));
		float		error = 0.0f;
		for (size_t i = 0; i < ARRAY_LEN(result); i++)
			error = MAX(error, fabsf(result[i] - expected[i]));
		record_resultf(OPFAM_EDGE_CASE, "decision.full_attention", st == OK && error < 1e-5f,
					   "backend=%s flash=%d max_error=%.3g", b->name, flash, error);
	}
}

static void test_encoder(const char *path, backend *b, int use_mmap) {
	model		m;
	status_code st = model_load_backend_ex_repack(&m, path, b, use_mmap, "none");
	record_resultf(OPFAM_EDGE_CASE, "decision.load", st == OK, "backend=%s mmap=%d status=%d",
				   b->name, use_mmap, st);
	if (st != OK)
		return;
	compute_scratch scratch	   = {0};
	int32_t			tokens[9]  = {3, 5, 7, 9, 11, 13, 15, 17, 19};
	uint8_t			outputs[9] = {0, 1, 0, 1, 0, 0, 0, 0, 0};
	float			all[9], selected[2], again[2], other[2];
	st = compute_forward_sequence_recipe(&m, &scratch, tokens, 4, NULL, 0, 1, all);
	if (st == OK)
		st = compute_forward_sequence_recipe(&m, &scratch, tokens, 4, outputs, 0, 1, selected);
	int ok = st == OK && fabsf(selected[0] - all[1]) < 1e-6f && fabsf(selected[1] - all[3]) < 1e-6f;
	record_resultf(OPFAM_EDGE_CASE, "decision.marker_gather", ok, "backend=%s errors=%.3g,%.3g",
				   b->name, fabsf(selected[0] - all[1]), fabsf(selected[1] - all[3]));
	st = compute_forward_sequence_recipe(&m, &scratch, tokens, 9, outputs, 0, 1, other);
	if (st == OK)
		st = compute_forward_sequence_recipe(&m, &scratch, tokens, 4, outputs, 0, 1, again);
	ok = st == OK && !memcmp(selected, again, sizeof(selected));
	record_resultf(OPFAM_EDGE_CASE, "decision.scratch_reuse", ok, "backend=%s", b->name);
	ok = fabsf(selected[0] - other[0]) > 1e-6f;
	record_resultf(OPFAM_EDGE_CASE, "decision.noncausal", ok, "backend=%s", b->name);
	st = compute_forward_sequence_recipe(&m, &scratch, tokens, 4, outputs, 2, 1, other);
	ok = st == OK && fabsf(other[0] - selected[0]) > 1e-6f;
	record_resultf(OPFAM_EDGE_CASE, "decision.type_embedding", ok, "backend=%s", b->name);
	_Atomic sig_atomic_t interrupt = 1;
	scratch.interrupt			   = &interrupt;
	st = compute_forward_sequence_recipe(&m, &scratch, tokens, 4, outputs, 0, 1, again);
	record_resultf(OPFAM_EDGE_CASE, "decision.interrupt", st == ERR_INTERRUPTED, "backend=%s",
				   b->name);
	interrupt = 0;
	st		  = compute_forward_sequence_recipe(&m, &scratch, tokens, 4, outputs, 0, 1, again);
	ok		  = st == OK && !memcmp(selected, again, sizeof(selected));
	record_resultf(OPFAM_EDGE_CASE, "decision.recovery", ok, "backend=%s", b->name);
	st				   = compute_forward_sequence_recipe(&m, &scratch, tokens, 9, NULL, 0, 1, all);
	const int counts[] = {1, 2, 4, 9};
	for (size_t i = 0; i < ARRAY_LEN(counts); i++) {
		uint8_t mask[9] = {0};
		float	scores[9];
		int		count = counts[i];
		for (int row = 0; row < count; row++)
			mask[row * 8 / MAX(count - 1, 1)] = 1;
		status_code selected_st =
			compute_forward_sequence_recipe(&m, &scratch, tokens, 9, mask, 0, 1, scores);
		ok = st == OK && selected_st == OK;
		for (int row = 0; ok && row < count; row++)
			ok = fabsf(scores[row] - all[row * 8 / MAX(count - 1, 1)]) < 1e-6f;
		record_resultf(OPFAM_EDGE_CASE, "decision.marker_counts", ok, "backend=%s count=%d",
					   b->name, count);
	}
	kvcache cache;
	st = kvcache_init(&cache, &m, 64, KV_QUANT_F16);
	ok = st == OK && cache.n_kv_layers == 0 && cache.hybrid == NULL;
	record_resultf(OPFAM_EDGE_CASE, "decision.no_kv", ok, "backend=%s", b->name);
	kvcache_free(&cache);
	compute_scratch_free(&scratch);
	model_free(&m);
}

static void test_decision_parse(const char *path) {
	gguf_ctx g;
	if (gguf_load_metadata(&g, path) != OK)
		return;
	char			err[256];
	decision_model *dm = decision_open(&g, err, sizeof(err));
	record_resultf(OPFAM_EDGE_CASE, "decision.registry", dm != NULL, "%s", dm ? "loaded" : err);
	if (dm) {
		const char *inputs[] = {
			"{\"state\":null,\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"Check\"}}}",
			"{\"state\":[],\"questions\":{\"q\":{\"type\":\"score\",\"instructions\":\"Check\","
			"\"criteria\":[\"low\",\"high\"]}}}",
			"{\"state\":{},\"questions\":{\"q\":{\"type\":\"choice\",\"instructions\":\"Check\","
			"\"criteria\":{\"a\":\"A\",\"b\":\"B\"}}}}",
			"{\"state\":{},\"questions\":{\"q\":{\"type\":\"score\",\"instructions\":\"Check\","
			"\"criteria\":[\"one\"]}}}",
			"{\"state\":{},\"questions\":{\"q\":{\"type\":\"choice\",\"instructions\":\"Check\","
			"\"criteria\":{}}}}",
			"{\"state\":{},\"images\":[\"image\"],\"questions\":{\"q\":{\"type\":\"noul\","
			"\"instructions\":\"Check\"}}}",
		};
		const decision_parse_status expected[] = {
			DECISION_PARSE_OK,		DECISION_PARSE_OK,		DECISION_PARSE_OK,
			DECISION_PARSE_INVALID, DECISION_PARSE_INVALID, DECISION_PARSE_UNSUPPORTED};
		for (size_t i = 0; i < ARRAY_LEN(inputs); i++) {
			json_object			 *body = json_tokener_parse(inputs[i]);
			decision_request	  req;
			decision_parse_status st = decision_parse(dm, body, &req, err, sizeof(err));
			record_resultf(OPFAM_EDGE_CASE, "decision.request", st == expected[i],
						   "case=%zu status=%d", i, st);
			decision_request_free(&req);
			json_object_put(body);
		}
		decision_close(dm);
	}
	gguf_free(&g);
}

void run_decision_tests(void) {
	char path[128];
	snprintf(path, sizeof(path), "/tmp/kappai-omni-%ld.gguf", (long)getpid());
	const uint8_t conv[] = {1, 0, 0, 0};
	tsg_lfm2_spec spec	 = {.dim			   = 32,
							.n_heads		   = 4,
							.n_kv_heads		   = 2,
							.head_dim		   = 8,
							.n_layers		   = 4,
							.intermediate	   = 64,
							.conv_kernel	   = 3,
							.ctx			   = 64,
							.vocab			   = 32,
							.n_decision_layers = 2,
							.seed			   = 15813,
							.is_conv		   = conv};
	tsg_build_lfm2(path, &spec);
	test_decision_parse(path);
	const char *backends[] = {"cpu_scalar", "cpu"};
	for (size_t i = 0; i < ARRAY_LEN(backends); i++) {
		backend *b = NULL;
		if (backend_create(backends[i], 0, &b) != OK)
			continue;
		test_full_attention(b);
		for (int use_mmap = 0; use_mmap <= 1; use_mmap++)
			test_encoder(path, b, use_mmap);
		backend_destroy(b);
	}
	unlink(path);
}
