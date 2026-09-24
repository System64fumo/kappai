#include "test_core.h"

typedef struct {
	const char *ref;
	const char *targets[BACKEND_MAX];
	int			n_targets;
} backend_selection;

static const char *default_reference_name(void) {
	backend_info infos[BACKEND_MAX];
	int			 n = backend_list(infos, BACKEND_MAX);
	for (int i = 0; i < n; i++)
		if (infos[i].available && strcmp(infos[i].name, "cpu_scalar") == 0)
			return "cpu_scalar";
	return "cpu";
}

static int backend_known(const char *name, backend_info *infos, int n) {
	for (int i = 0; i < n; i++)
		if (strcmp(infos[i].name, name) == 0)
			return 1;
	return 0;
}

static int parse_backend_selection(int argc, char **argv, backend_info *infos, int n_backends,
								   backend_selection *sel) {
	memset(sel, 0, sizeof(*sel));
	const char *pos[BACKEND_MAX + 1];
	int			n_pos = 0;
	for (int ai = 1; ai < argc; ai++) {
		if (argv[ai][0] == '-')
			continue;
		if (n_pos >= (int)(sizeof(pos) / sizeof(pos[0])))
			break;
		pos[n_pos++] = argv[ai];
	}

	for (int i = 0; i < n_pos; i++) {
		if (!backend_known(pos[i], infos, n_backends)) {
			fprintf(stderr, "error: unknown backend '%s'\n", pos[i]);
			return -1;
		}
	}

	if (n_pos == 0) {
		sel->ref = default_reference_name();
		return 0;
	}
	if (n_pos == 1 && strcmp(pos[0], default_reference_name()) == 0) {
		const char *best	  = NULL;
		int			best_prio = -1;
		for (int i = 0; i < n_backends; i++) {
			if (!infos[i].available || strcmp(infos[i].name, pos[0]) == 0)
				continue;
			if (!(infos[i].caps & BCAP_IS_HOST))
				continue;
			if (infos[i].priority > best_prio) {
				best_prio = infos[i].priority;
				best	  = infos[i].name;
			}
		}
		if (!best) {
			fprintf(stderr,
					"error: '%s' is the only host backend; pass two backends to "
					"compare (e.g. '%s <other>')\n",
					pos[0], pos[0]);
			return -1;
		}
		sel->ref					   = best;
		sel->targets[sel->n_targets++] = pos[0];
		return 0;
	}

	sel->ref = pos[0];
	for (int i = 1; i < n_pos; i++)
		sel->targets[sel->n_targets++] = pos[i];
	return 0;
}

int run_per_op_mode(int argc, char **argv, backend_info *infos, int n_backends) {
	stats_reset();

	backend_selection sel;
	if (parse_backend_selection(argc, argv, infos, n_backends, &sel) != 0) {
		usage(argv[0]);
		return 1;
	}

	backend *cpu = NULL;
	if (backend_create(sel.ref, 0, &cpu) != OK) {
		fprintf(stderr, "ERROR: reference backend '%s' is unavailable\n", sel.ref);
		return 1;
	}
	if (cpu->desc)
		printf("reference backend: %s  (%s)\n", cpu->name, cpu->desc);
	else
		printf("reference backend: %s\n", cpu->name);

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
	run_toolcall_tests();
	flush_family(OPFAM_TOOLCALL);

	int run_all = wants_all(argc, argv);
	int any_run = 0;
	for (int bi = 0; bi < n_backends; bi++) {
		int want = 0;
		if (run_all) {
			want = 1;
		} else {
			for (int t = 0; t < sel.n_targets; t++)
				if (strcmp(sel.targets[t], infos[bi].name) == 0)
					want = 1;
		}
		if (!want)
			continue;
		if (strcmp(infos[bi].name, sel.ref) == 0)
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
