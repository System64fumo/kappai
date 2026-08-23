#include "test_core.h"

static const int g_model_flash = 1;

static int32_t tok_test_hash_lookup(const struct tok_hash_entry *ht, size_t cap, const char *key,
									size_t klen) {
	uint64_t h = fnv1a(key, klen) & (cap - 1);
	while (ht[h].used) {
		if (ht[h].key_len == klen && memcmp(ht[h].key, key, klen) == 0)
			return ht[h].id;
		h = (h + 1) & (cap - 1);
	}
	return -1;
}

typedef struct {
	const char *p;
	size_t		n;
	int32_t		id;
} tok_ref_piece;

static int32_t tok_ref_pair_rank(const tokenizer *t, const tok_ref_piece *a, const tok_ref_piece *b,
								 char *key) {
	size_t klen = a->n + b->n;
	memcpy(key, a->p, a->n);
	memcpy(key + a->n, b->p, b->n);
	if (t->has_merges)
		return tok_test_hash_lookup(t->merge_hash, t->merge_hash_capacity, key, klen);
	return tok_test_hash_lookup(t->hash, t->hash_capacity, key, klen);
}

static void tok_ref_bpe(const tokenizer *t, const char *text, size_t len, int32_t *out_ids,
						int *n_out) {
	tok_ref_piece *pcs	= xmalloc((len + 1) * sizeof(tok_ref_piece));
	int			   npcs = 0;
	size_t		   ci	= 0;
	while (ci < len) {
		unsigned char c	 = (unsigned char)text[ci];
		size_t		  cl = 1;
		if (c >= 0xF0)
			cl = 4;
		else if (c >= 0xE0)
			cl = 3;
		else if (c >= 0xC0)
			cl = 2;
		if (ci + cl > len)
			cl = 1;
		int32_t id = tok_test_hash_lookup(t->hash, t->hash_capacity, text + ci, cl);
		if (id < 0)
			id = (t->unk_id >= 0) ? t->unk_id : 0;
		pcs[npcs].p	 = text + ci;
		pcs[npcs].n	 = cl;
		pcs[npcs].id = id;
		npcs++;
		ci += cl;
	}

	char  *key		  = xmalloc(len * 2 + 4);
	size_t arena_cap  = len + 1;
	char  *arena	  = xmalloc(arena_cap);
	size_t arena_used = 0;

	while (npcs > 1) {
		int		best_i	  = -1;
		int32_t best_rank = INT32_MAX;
		size_t	best_klen = 0;
		for (int i = 0; i < npcs - 1; i++) {
			int32_t rank = tok_ref_pair_rank(t, &pcs[i], &pcs[i + 1], key);
			if (rank < 0 || rank >= best_rank)
				continue;
			best_rank = rank;
			best_i	  = i;
			best_klen = pcs[i].n + pcs[i + 1].n;
		}
		if (best_i < 0)
			break;
		if (arena_used + best_klen > arena_cap) {
			uintptr_t old_base = (uintptr_t)arena;
			size_t	  old_used = arena_used;
			while (arena_used + best_klen > arena_cap)
				arena_cap *= 2;
			char *new_arena = xrealloc(arena, arena_cap);
			if ((uintptr_t)new_arena != old_base) {
				for (int i = 0; i < npcs; i++) {
					uintptr_t p = (uintptr_t)pcs[i].p;
					if (p >= old_base && p < old_base + old_used)
						pcs[i].p = new_arena + (p - old_base);
				}
			}
			arena = new_arena;
		}
		char *merged = arena + arena_used;
		memcpy(merged, pcs[best_i].p, pcs[best_i].n);
		memcpy(merged + pcs[best_i].n, pcs[best_i + 1].p, pcs[best_i + 1].n);
		arena_used += best_klen;
		pcs[best_i].p  = merged;
		pcs[best_i].n  = best_klen;
		int32_t mid	   = tok_test_hash_lookup(t->hash, t->hash_capacity, merged, best_klen);
		pcs[best_i].id = (mid >= 0) ? mid : ((t->unk_id >= 0) ? t->unk_id : 0);
		for (int i = best_i + 1; i < npcs - 1; i++)
			pcs[i] = pcs[i + 1];
		npcs--;
	}

	*n_out = npcs;
	for (int i = 0; i < npcs; i++)
		out_ids[i] = pcs[i].id;
	free(pcs);
	free(key);
	free(arena);
}

static uint64_t tok_rng_state = 0x12345678ULL;

static uint64_t tok_rng_next(void) {
	tok_rng_state ^= tok_rng_state << 13;
	tok_rng_state ^= tok_rng_state >> 7;
	tok_rng_state ^= tok_rng_state << 17;
	return tok_rng_state;
}

static void run_tokenizer_bpe_differential(const tokenizer *t) {
	const int			n_lens	= 6;
	static const size_t lens[]	= {1, 2, 7, 33, 129, 511};
	int32_t			   *ids_new = xmalloc((lens[n_lens - 1] + 16) * sizeof(int32_t));
	int32_t			   *ids_ref = xmalloc((lens[n_lens - 1] + 16) * sizeof(int32_t));
	char			   *text	= xmalloc(lens[n_lens - 1] + 1);

	int mismatches = 0;
	for (int trial = 0; trial < 256; trial++) {
		size_t			  len		 = lens[trial % n_lens];
		static const char alphabet[] = " the quick brown fox  \xe2\x96\x81"
									   "abcdef";
		for (size_t i = 0; i < len; i++)
			text[i] = alphabet[tok_rng_next() % (sizeof(alphabet) - 1)];
		len = strnlen(text, len);
		if (len == 0)
			continue;
		int n_new = 0, n_ref = 0;
		tokenizer_bpe_encode((tokenizer *)t, text, len, ids_new, (int)(len + 8), &n_new);
		tok_ref_bpe(t, text, len, ids_ref, &n_ref);
		if (n_new != n_ref || memcmp(ids_new, ids_ref, (size_t)n_new * sizeof(int32_t)) != 0) {
			mismatches++;
			if (mismatches <= 3)
				fprintf(stderr, "BPE mismatch trial=%d len=%zu n_new=%d n_ref=%d\n", trial, len,
						n_new, n_ref);
		}
	}

	record_result(OPFAM_ARCH_PIPELINE, "tokenizer.bpe_differential",
				  mismatches == 0 ? V_PASS : V_FAIL,
				  mismatches == 0 ? "heap BPE matches reference merge order"
								  : "BPE output diverges from reference");
	free(ids_new);
	free(ids_ref);
	free(text);
}

static void run_tokenizer_roundtrip(tokenizer *t) {
	static const char *texts[] = {
		"Hello world!",
		"The quick brown fox jumps over the lazy dog.",
		"def fibonacci(n):\n    if n < 2:\n        return n\n    return "
		"fibonacci(n-1)+fibonacci(n-2)\n",
		"\xe2\x96\x81unicode \xc3\xa9\xc3\xa8 mix \xe2\x82\xac end",
	};
	int n_texts = (int)(sizeof(texts) / sizeof(texts[0]));
	for (int ti = 0; ti < n_texts; ti++) {
		const char *txt = texts[ti];
		int32_t		ids1[512], ids2[512];
		int			n1 = tokenizer_encode_with_specials(t, txt, 0, ids1, 512, NULL);
		int			n2 = tokenizer_encode_with_specials(t, txt, 0, ids2, 512, NULL);
		int ok = (n1 == n2 && n1 > 0 && memcmp(ids1, ids2, (size_t)n1 * sizeof(int32_t)) == 0);
		record_result(OPFAM_ARCH_PIPELINE, "tokenizer.encode_deterministic", ok ? V_PASS : V_FAIL,
					  ok ? "repeated encode identical" : "encode not stable");
	}
}

static void run_model_cross(const char *path, backend *cpu, backend *tgt, int n_prefill,
							int n_decode) {
	printf("\n========================================\n");
	printf("Real-model cross-check: %s vs %s  (%s)\n", tgt->name, cpu->name, path);
	printf("========================================\n");

	model m_cpu = {0};
	model m_tgt = {0};
	if (model_load_backend_ex_repack(&m_cpu, path, cpu, 1, NULL, 0) != OK) {
		fprintf(stderr, "ERROR: failed to load model on cpu backend\n");
		g_fail++;
		return;
	}
	if (model_load_backend_ex_repack(&m_tgt, path, tgt, 1, NULL, 0) != OK) {
		fprintf(stderr, "ERROR: failed to load model on %s backend\n", tgt->name);
		model_free(&m_cpu);
		g_fail++;
		return;
	}

	gguf_ctx  gctx	   = {0};
	tokenizer tok	   = {0};
	int		  have_tok = (gguf_load(&gctx, path) == OK) && (tokenizer_init(&tok, &gctx) == OK);
	if (have_tok) {
		tok_rng_state = 0x12345678ULL;
		run_tokenizer_bpe_differential(&tok);
		flush_family(OPFAM_ARCH_PIPELINE);
		run_tokenizer_roundtrip(&tok);
		flush_family(OPFAM_ARCH_PIPELINE);
	}

	int32_t prompt_buf[64];
	int		n_prompt;
	if (have_tok) {
		static const char *sample_text = "The quick brown fox jumps over the lazy dog.";
		n_prompt = tokenizer_encode_with_specials(&tok, sample_text, 1, prompt_buf,
												  (int)ARRAY_LEN(prompt_buf), NULL);
		if (n_prompt <= 0) {
			prompt_buf[0] = tok.bos_id >= 0 ? tok.bos_id : 1;
			n_prompt	  = 1;
		}
	} else {
		prompt_buf[0] = 1;
		n_prompt	  = 1;
	}
	n_prompt = MIN(n_prompt, n_prefill);

	int n_ctx	= n_prompt + n_decode + 16;
	int vocab	= m_cpu.vocab_size;
	m_cpu.n_ctx = n_ctx;
	m_tgt.n_ctx = n_ctx;
	if (m_cpu.recipe)
		recipe_free(m_cpu.recipe);
	m_cpu.recipe = recipe_build(&m_cpu);
	if (m_tgt.recipe)
		recipe_free(m_tgt.recipe);
	m_tgt.recipe = recipe_build(&m_tgt);

	kvcache kv_cpu = {0};
	kvcache kv_tgt = {0};
	if (kvcache_init(&kv_cpu, &m_cpu, n_ctx, KV_QUANT_F16) != OK ||
		kvcache_init(&kv_tgt, &m_tgt, n_ctx, KV_QUANT_F16) != OK) {
		fprintf(stderr, "ERROR: kvcache_init failed\n");
		g_fail++;
		goto cleanup_models;
	}

	compute_scratch s_cpu;
	compute_scratch s_tgt;
	compute_scratch_init(&s_cpu);
	compute_scratch_init(&s_tgt);
	(void)compute_scratch_ensure(&s_cpu, &m_cpu, n_ctx);
	(void)compute_scratch_ensure(&s_tgt, &m_tgt, n_ctx);

	float *logits_cpu = xmalloc((size_t)vocab * sizeof(float));
	float *logits_tgt = xmalloc((size_t)vocab * sizeof(float));

	printf("prompt: %d token(s), decoding %d additional token(s) greedily\n", n_prompt, n_decode);

	int32_t tok_cpu	   = prompt_buf[0];
	int32_t tok_tgt	   = prompt_buf[0];
	int		mismatches = 0;

	for (int i = 0; i < n_prompt + n_decode; i++) {
		int		pos		  = i;
		int		is_prompt = (i < n_prompt);
		int32_t feed_cpu  = is_prompt ? prompt_buf[i] : tok_cpu;
		int32_t feed_tgt  = is_prompt ? prompt_buf[i] : tok_tgt;

		compute_forward(&m_cpu, &kv_cpu, &s_cpu, feed_cpu, pos, g_model_flash, logits_cpu);
		if (cpu->synchronize)
			cpu->synchronize(cpu);

		compute_forward(&m_tgt, &kv_tgt, &s_tgt, feed_tgt, pos, g_model_flash, logits_tgt);
		if (tgt->synchronize)
			tgt->synchronize(tgt);

		int32_t am_cpu = sampler_argmax(logits_cpu, vocab);
		int32_t am_tgt = sampler_argmax(logits_tgt, vocab);

		char label[96];
		char detail[256];
		snprintf(label, sizeof(label), "model token[%d]", pos);
		verdict v =
			classify_output("loose", logits_cpu, logits_tgt, vocab, OK, detail, sizeof(detail));
		if (v != V_PASS && v != V_SKIP)
			compute_debug(logits_cpu, logits_tgt, vocab);
		record_result(OPFAM_ARCH_PIPELINE, label, v, detail);
		if (am_cpu != am_tgt)
			mismatches++;
		tok_cpu = am_cpu;
		tok_tgt = am_tgt;
	}

	printf("greedy-decode argmax agreement: %d/%d tokens matched\n",
		   n_prompt + n_decode - mismatches, n_prompt + n_decode);
	flush_family(OPFAM_ARCH_PIPELINE);

	free(logits_cpu);
	free(logits_tgt);
	compute_scratch_free(&s_cpu);
	compute_scratch_free(&s_tgt);
	kvcache_free(&kv_cpu);
	kvcache_free(&kv_tgt);

cleanup_models:
	if (have_tok)
		tokenizer_free(&tok);
	if (have_tok)
		gguf_free(&gctx);
	model_free(&m_cpu);
	model_free(&m_tgt);
}

int run_model_mode(int argc, char **argv, backend_info *infos, int n_backends) {
	log_set_level(LOG_INFO);
	stats_reset();
	const char *path	  = NULL;
	int			n_prefill = 32;
	int			n_decode  = 16;
	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--model") == 0 && ai + 1 < argc) {
			path = argv[++ai];
		} else if (strcmp(argv[ai], "--n-prefill") == 0 && ai + 1 < argc) {
			n_prefill = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--n-decode") == 0 && ai + 1 < argc) {
			n_decode = atoi(argv[++ai]);
		}
	}
	if (!path) {
		fprintf(stderr, "ERROR: --model requires a path to a .gguf file\n");
		return 1;
	}
	backend *cpu = NULL;
	if (backend_create("cpu", 0, &cpu) != OK) {
		fprintf(stderr, "ERROR: cpu backend (the reference) is unavailable\n");
		return 1;
	}
	int run_all = wants_all(argc, argv);
	int any_run = 0;
	for (int bi = 0; bi < n_backends; bi++) {
		if (strcmp(infos[bi].name, "cpu") == 0)
			continue;
		int want = run_all || matches_name(argc, argv, infos[bi].name);
		if (!want)
			continue;
		if (!infos[bi].available) {
			printf("\n=== %s: SKIPPED (not available) ===\n", infos[bi].name);
			continue;
		}
		backend *tgt = NULL;
		if (backend_create(infos[bi].name, 0, &tgt) != OK) {
			printf("\n=== %s: SKIPPED (failed to init) ===\n", infos[bi].name);
			continue;
		}
		any_run = 1;
		run_model_cross(path, cpu, tgt, n_prefill, n_decode);
		backend_destroy(tgt);
	}
	if (!any_run) {
		fprintf(stderr, "No matching/available backends were tested.\n");
		backend_destroy(cpu);
		return 1;
	}
	print_final_results();
	backend_destroy(cpu);
	return g_fail > 0 ? 1 : 0;
}
