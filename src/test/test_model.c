#include "test_core.h"

static const int g_model_flash = 1;

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
