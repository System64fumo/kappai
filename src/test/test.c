#include "test_core.h"

int test_parse_device_selection(int argc, char **argv, const test_device *devs, int n_devs,
								device_selection *sel) {
	memset(sel, 0, sizeof(*sel));
	const test_device *default_ref = test_default_reference(devs, n_devs);
	if (!default_ref) {
		fprintf(stderr, "error: no compute devices available\n");
		return -1;
	}

	const char *pos[TEST_MAX_DEVICES + 1];
	int			n_pos = 0;
	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--n-prefill") == 0 || strcmp(argv[ai], "--n-decode") == 0 ||
			strcmp(argv[ai], "--arch") == 0 || strcmp(argv[ai], "--quant") == 0) {
			ai++;
			continue;
		}
		if (argv[ai][0] == '-')
			continue;
		if (n_pos >= (int)ARRAY_LEN(pos))
			break;
		pos[n_pos++] = argv[ai];
	}

	for (int i = 0; i < n_pos; i++) {
		if (!test_find_device(devs, n_devs, pos[i])) {
			fprintf(stderr, "error: unknown device '%s'\n", pos[i]);
			return -1;
		}
	}

	if (n_pos == 0) {
		sel->ref = default_ref->spec;
		return 0;
	}
	if (n_pos == 1 && strcmp(pos[0], default_ref->spec) == 0) {
		const test_device *best		 = NULL;
		int				   best_prio = -1;
		for (int i = 0; i < n_devs; i++) {
			if (strcmp(devs[i].spec, pos[0]) == 0 || !backend_device_is_host(devs[i].spec))
				continue;
			int prio = test_device_priority(&devs[i]);
			if (prio > best_prio) {
				best_prio = prio;
				best	  = &devs[i];
			}
		}
		if (!best) {
			fprintf(stderr,
					"error: '%s' is the only host device; pass two devices to "
					"compare (e.g. '%s <other>')\n",
					pos[0], pos[0]);
			return -1;
		}
		sel->ref					   = best->spec;
		sel->targets[sel->n_targets++] = pos[0];
		return 0;
	}

	sel->ref = pos[0];
	for (int i = 1; i < n_pos; i++)
		sel->targets[sel->n_targets++] = pos[i];
	return 0;
}

int run_per_op_mode(int argc, char **argv, test_device *devs, int n_devs) {
	stats_reset();

	device_selection sel;
	if (test_parse_device_selection(argc, argv, devs, n_devs, &sel) != 0) {
		usage(argv[0]);
		return 1;
	}

	backend *ref = NULL;
	if (test_device_create(test_find_device(devs, n_devs, sel.ref), &ref) != OK) {
		fprintf(stderr, "ERROR: reference device '%s' is unavailable\n", sel.ref);
		return 1;
	}
	if (ref->desc)
		printf("reference device: %s  (%s)\n", sel.ref, ref->desc);
	else
		printf("reference device: %s\n", sel.ref);

	synth_suite_common_init();

	run_arch_tests(ref, NULL);

	run_sampler_tests();
	flush_family(OPFAM_SAMPLER);
	run_tokenizer_tests();
	flush_family(OPFAM_TOKENIZER);
	run_jinja_tests();
	run_portability_tests();
	run_model_lifecycle_tests();
	run_decision_tests();
	run_server_task_tests();
	flush_family(OPFAM_EDGE_CASE);
	run_hybrid_state_tests(ref);
	flush_family(OPFAM_HYBRID_STATE);
	run_orchestration_tests();
	flush_family(OPFAM_ORCHESTRATION);
	run_moe_stream_tests();
	flush_family(OPFAM_MOE_STREAM);
	run_toolcall_tests();
	flush_family(OPFAM_TOOLCALL);
	run_hcs_tests();
	flush_family(OPFAM_HCS);
	run_mempool_tests();
	flush_family(OPFAM_MEMPOOL);
	run_moe_tier_tests();
	flush_family(OPFAM_MOE_TIER);
	run_plan_tests();
	run_gguf_tests();
	flush_family(OPFAM_PLAN);

	int run_all = wants_all(argc, argv);
	int any_run = 0;
	for (int di = 0; di < n_devs; di++) {
		int want = 0;
		if (run_all) {
			want = 1;
		} else {
			for (int t = 0; t < sel.n_targets; t++)
				if (strcmp(sel.targets[t], devs[di].spec) == 0)
					want = 1;
		}
		if (!want || strcmp(devs[di].spec, sel.ref) == 0)
			continue;
		backend *tgt = NULL;
		if (test_device_create(&devs[di], &tgt) != OK) {
			any_run = 1;
			printf("\n=== %s: SKIPPED (failed to init) ===\n", devs[di].spec);
			continue;
		}
		any_run = 1;
		printf("\n=== device %s ===\n", devs[di].spec);
		run_per_op_tests(ref, tgt);
		run_arch_tests(ref, tgt);
		backend_destroy(tgt);
	}
	if (!any_run)
		fprintf(stderr, "No matching/available devices were tested.\n");
	print_final_results();
	backend_destroy(ref);
	return g_fail > 0 ? 1 : 0;
}

int main(int argc, char **argv) {
	log_init(log_default_config());
	log_set_level(getenv("KAPPAI_TEST_VERBOSE") ? LOG_INFO : LOG_WARN);

	color_init();
	test_device devs[TEST_MAX_DEVICES];
	int			n_devs = test_enum_devices(devs, TEST_MAX_DEVICES);

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
		if (strcmp(argv[ai], "--archs") == 0)
			return run_arch_cross_mode(argc, argv, devs, n_devs);
	}
	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--bench") == 0)
			return run_bench_mode(argc, argv, devs, n_devs);
	}
	for (int ai = 1; ai < argc; ai++) {
		if (argv[ai][0] == '-' && argv[ai][1] == '-' && argv[ai][2] != '\0' &&
			strcmp(argv[ai], "--all") != 0) {
			fprintf(stderr, "error: unknown option '%s' (devices are plain names, e.g. 'cpu0')\n",
					argv[ai]);
			usage(argv[0]);
			return 1;
		}
	}
	return run_per_op_mode(argc, argv, devs, n_devs);
}
