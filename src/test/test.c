#include "test_core.h"

int run_per_op_mode(int argc, char **argv, backend_info *infos, int n_backends) {
	stats_reset();
	backend *cpu = NULL;
	if (backend_create("cpu", 0, &cpu) != OK) {
		fprintf(stderr, "ERROR: cpu backend (the reference) is unavailable\n");
		return 1;
	}

	synth_suite_common_init();

	run_arch_tests(cpu, NULL);

	run_sampler_tests();
	flush_family(OPFAM_SAMPLER);
	run_tokenizer_tests();
	flush_family(OPFAM_TOKENIZER);
	run_jinja_tests();
	flush_family(OPFAM_EDGE_CASE);
	run_hybrid_state_tests(cpu);
	flush_family(OPFAM_HYBRID_STATE);
	run_orchestration_tests();
	flush_family(OPFAM_ORCHESTRATION);
	run_moe_stream_tests();
	flush_family(OPFAM_MOE_STREAM);

	int run_all = wants_all(argc, argv);
	int any_run = 0;
	for (int bi = 0; bi < n_backends; bi++) {
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
		run_per_op_tests(cpu, tgt);
		run_arch_tests(cpu, tgt);
		backend_destroy(tgt);
	}
	if (!any_run) {
		fprintf(stderr, "No matching/available backends were tested.\n");
		usage(argv[0]);
		backend_destroy(cpu);
		return 1;
	}
	print_final_results();
	backend_destroy(cpu);
	return g_fail > 0 ? 1 : 0;
}

int main(int argc, char **argv) {
	log_init(log_default_config());
	log_set_level(getenv("KAPPAI_TEST_VERBOSE") ? LOG_INFO : LOG_WARN);

	color_init();
	backend_info infos[BACKEND_MAX];
	int			 n_backends = backend_list(infos, BACKEND_MAX);

	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "-h") == 0 || strcmp(argv[ai], "--help") == 0) {
			usage(argv[0]);
			return 0;
		}
	}
	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--model") == 0)
			return run_model_mode(argc, argv, infos, n_backends);
	}
	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--bench") == 0)
			return run_matmul_bench_mode(argc, argv, infos, n_backends);
	}
	for (int ai = 1; ai < argc; ai++) {
		if (argv[ai][0] == '-' && argv[ai][1] == '-' && argv[ai][2] != '\0' &&
			strcmp(argv[ai], "--all") != 0) {
			fprintf(stderr, "error: unknown option '%s' (use plain backend name, e.g. 'cpu')\n",
					argv[ai]);
			usage(argv[0]);
			return 1;
		}
	}
	return run_per_op_mode(argc, argv, infos, n_backends);
}
