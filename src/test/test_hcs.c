#include "hcs.h"
#include "model.h"
#include "test_core.h"

#include <string.h>

#define record_hcs(label, ok, ...) record_resultf(OPFAM_HCS, (label), (ok), __VA_ARGS__)

static hcs_config single_device(const char *name, unsigned tasks, int layers, int experts) {
	hcs_config h = {.enabled = true, .n_devices = 1};
	snprintf(h.devices[0].name, HCS_NAME_MAX, "%s", name);
	h.devices[0].tasks	 = tasks;
	h.devices[0].layers	 = layers;
	h.devices[0].experts = experts;
	return h;
}

static int validate_has(const hcs_config *h, const char *needle) {
	char err[HCS_ERR_MAX];
	int	 rc = hcs_validate(h, synth_chat_model_path, err, sizeof(err));
	return rc < 0 && strstr(err, needle) != NULL;
}

static void test_count_and_tasks(void) {
	int		 n;
	unsigned t;
	int		 ok = hcs_parse_count("*", &n) == 0 && n == HCS_COUNT_ALL;
	ok &= hcs_parse_count(" 12 ", &n) == 0 && n == 12;
	ok &= hcs_parse_count("-1", &n) != 0 && hcs_parse_count("", &n) != 0 &&
		  hcs_parse_count("1x", &n) != 0;
	record_hcs("count.parse", ok, "ok=%d", ok);

	ok = hcs_parse_tasks("prefill, decode", &t) == 0 && t == HCS_TASKS_ALL;
	ok &= hcs_parse_tasks("decode", &t) == 0 && t == HCS_TASK_DECODE;
	ok &= hcs_parse_tasks("train", &t) != 0 && hcs_parse_tasks("", &t) != 0;
	record_hcs("tasks.parse", ok, "ok=%d", ok);
}

static void test_loader(void) {
	char	   err[256];
	hcs_loader l;
	hcs_config h;
	hcs_loader_init(&l);
	int a  = hcs_loader_open_section(&l, "cpu_0", err, sizeof(err));
	int ok = a == 0 && hcs_loader_set(&l, -1, "devices", "cpu_0", err, sizeof(err)) == 0 &&
			 hcs_loader_set(&l, a, "tasks", "prefill, decode", err, sizeof(err)) == 0 &&
			 hcs_loader_set(&l, a, "layers", "*", err, sizeof(err)) == 0 &&
			 hcs_loader_set(&l, a, "experts", "10", err, sizeof(err)) == 0 &&
			 hcs_loader_finish(&l, &h, err, sizeof(err)) == 0;
	ok &= ok && h.n_devices == 1 && h.devices[0].layers == HCS_COUNT_ALL &&
		  h.devices[0].experts == 10 && h.devices[0].tasks == HCS_TASKS_ALL;
	record_hcs("loader.valid", ok, "err='%s'", err);

	ok = hcs_loader_set(&l, a, "layers", "2", err, sizeof(err)) < 0 &&
		 strstr(err, "duplicate") != NULL;
	ok &= hcs_loader_set(&l, a, "bogus", "1", err, sizeof(err)) < 0;
	ok &= hcs_loader_open_section(&l, "cpu_0", err, sizeof(err)) < 0;
	record_hcs("loader.strict_keys", ok, "ok=%d", ok);

	hcs_loader_init(&l);
	hcs_loader_open_section(&l, "cpu_0", err, sizeof(err));
	hcs_loader_set(&l, -1, "devices", "cpu_0, cpu_1", err, sizeof(err));
	ok = hcs_loader_finish(&l, &h, err, sizeof(err)) < 0 && strstr(err, "cpu_1") != NULL;
	record_hcs("loader.missing_section", ok, "err='%s'", err);

	hcs_loader_init(&l);
	hcs_loader_open_section(&l, "cpu_0", err, sizeof(err));
	hcs_loader_open_section(&l, "extra_0", err, sizeof(err));
	hcs_loader_set(&l, -1, "devices", "cpu_0", err, sizeof(err));
	ok = hcs_loader_finish(&l, &h, err, sizeof(err)) < 0 && strstr(err, "extra_0") != NULL;
	record_hcs("loader.unlisted_section", ok, "err='%s'", err);
}

static void test_split(void) {
	int n_layers = 0, n_experts = 0;
	int probe_rc = model_probe_shape(synth_chat_model_path, &n_layers, &n_experts);

	hcs_config h = single_device("cpu_scalar_0", HCS_TASKS_ALL, 1, HCS_COUNT_UNSET);
	h.n_devices	 = 2;
	h.devices[1] =
		single_device("opengl_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET).devices[0];
	int held[HCS_MAX_DEVICES];
	int assigned = hcs_resolve_layers(&h, n_layers, held);
	int ok		 = assigned == n_layers && held[0] == 1 && held[1] == n_layers - 1;
	record_hcs("split.resolve_star_takes_remainder", ok, "assigned=%d probe=%d n=%d", assigned,
			   probe_rc, n_layers);

	config cfg = config_defaults();
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	ok = strcmp(cfg.device, "opengl_0") == 0 && cfg.ngl == n_layers - 1;
	record_hcs("split.order_irrelevant_host_listed_first", ok, "device=%s ngl=%d", cfg.device,
			   cfg.ngl);

	h			 = single_device("opengl_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	h.n_devices	 = 2;
	h.devices[1] = single_device("cpu_scalar_0", HCS_TASKS_ALL, 1, HCS_COUNT_UNSET).devices[0];
	cfg			 = config_defaults();
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	ok = strcmp(cfg.device, "opengl_0") == 0 && cfg.ngl == n_layers - 1;
	record_hcs("split.order_irrelevant_host_listed_last", ok, "device=%s ngl=%d", cfg.device,
			   cfg.ngl);

	h.devices[1].layers = 0;
	cfg					= config_defaults();
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	record_hcs("split.host_holds_no_layers", cfg.ngl == n_layers, "ngl=%d", cfg.ngl);
}

static void test_enabled_flag(void) {
	char	   err[256];
	hcs_loader l;
	hcs_config h;
	hcs_loader_init(&l);
	hcs_loader_set(&l, -1, "enabled", "off", err, sizeof(err));
	hcs_loader_set(&l, -1, "devices", "cpu_0", err, sizeof(err));
	int a = hcs_loader_open_section(&l, "cpu_0", err, sizeof(err));
	hcs_loader_set(&l, a, "tasks", "decode", err, sizeof(err));
	int ok = hcs_loader_finish(&l, &h, err, sizeof(err)) == 0 && !h.enabled;
	record_hcs("enabled.off_parsed", ok, "err='%s'", err);

	hcs_loader_init(&l);
	hcs_loader_set(&l, -1, "devices", "cpu_0", err, sizeof(err));
	a  = hcs_loader_open_section(&l, "cpu_0", err, sizeof(err));
	ok = hcs_loader_finish(&l, &h, err, sizeof(err)) == 0 && h.enabled;
	record_hcs("enabled.defaults_on_when_section_present", ok, "enabled=%d", h.enabled);

	hcs_loader_init(&l);
	ok = hcs_loader_set(&l, -1, "enabled", "maybe", err, sizeof(err)) < 0;
	record_hcs("enabled.rejects_garbage", ok, "err='%s'", err);

	h		   = single_device("cpu_0", HCS_TASK_DECODE, 1, 99);
	h.enabled  = false;
	config cfg = config_defaults();
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	ok = hcs_validate(&h, "/nonexistent/model.gguf", err, sizeof(err)) == 0 &&
		 strcmp(cfg.device, "cpu_0") == 0 && cfg.ngl == -1 && cfg.moe_cache_cap == 0;
	record_hcs("enabled.disabled_maps_to_single_device", ok, "err='%s' cap=%d", err,
			   cfg.moe_cache_cap);
}

static void test_legacy(void) {
	char	   err[256];
	hcs_config h;
	hcs_defaults(&h);
	hcs_legacy legacy = {.has_ngl = true, .ngl = 0, .has_cache = true, .cache = 8};
	int ok = hcs_migrate_legacy(&h, &legacy, err, sizeof(err)) == 0 &&
			 strcmp(h.devices[0].name, "cpu_0") == 0 && h.devices[0].experts == 8 && h.enabled;
	record_hcs("legacy.ngl0_and_cache", ok, "device='%s'", h.devices[0].name);

	hcs_defaults(&h);
	legacy = (hcs_legacy){.has_ngl = true, .ngl = 12};
	ok = hcs_migrate_legacy(&h, &legacy, err, sizeof(err)) < 0 && strstr(err, "partial") != NULL;
	record_hcs("legacy.partial_ngl_rejected", ok, "err='%s'", err);
}

static void test_validate(void) {
	int n_layers = 0, n_experts = 0;
	if (model_probe_shape(synth_chat_model_path, &n_layers, &n_experts) != OK) {
		record_hcs("validate.setup", 0, "cannot probe synthetic model");
		return;
	}

	hcs_config h = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	char	   err[HCS_ERR_MAX];
	int		   ok = hcs_validate(&h, synth_chat_model_path, err, sizeof(err)) == 0;
	record_hcs("validate.v1_baseline", ok, "err='%s'", err);

	h  = single_device("cpu_0", HCS_TASKS_ALL, n_layers, HCS_COUNT_UNSET);
	ok = hcs_validate(&h, synth_chat_model_path, err, sizeof(err)) == 0;
	record_hcs("validate.explicit_all_layers", ok, "err='%s'", err);

	h  = single_device("cpu_0", HCS_TASKS_ALL, 1, HCS_COUNT_UNSET);
	ok = n_layers > 1 && validate_has(&h, "add 'layers = *'");
	record_hcs("validate.under_subscription", ok, "n_layers=%d", n_layers);

	h = single_device("cpu_0", HCS_TASKS_ALL, n_layers + 1, HCS_COUNT_UNSET);
	record_hcs("validate.over_subscription", validate_has(&h, "remove 1"), "n=%d", n_layers);

	h = single_device("cpu_0", HCS_TASK_DECODE, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	record_hcs("validate.tasks_must_cover_both", validate_has(&h, "lacks 'prefill'"), "");

	h = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, 4);
	record_hcs("validate.experts_on_dense", validate_has(&h, "not a MoE"), "experts=%d", n_experts);

	h = single_device("nope_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	record_hcs("validate.unknown_device", validate_has(&h, "not an available device"), "");

	h = single_device("cpu", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	record_hcs("validate.bare_name_rejected", validate_has(&h, "<backend>_<n>"), "");

	h			 = single_device("cpu_scalar_0", HCS_TASKS_ALL, 1, 8);
	h.n_devices	 = 2;
	h.devices[1] = single_device("cpu_x86_64_0", HCS_TASKS_ALL, HCS_COUNT_ALL, 4).devices[0];
	ok			 = validate_has(&h, "2 host (cpu) devices") && validate_has(&h, "differs from");
	record_hcs("validate.two_hosts_rejected", ok, "ok=%d", ok);

	h			= single_device("opengl_0", HCS_TASKS_ALL, 1, HCS_COUNT_UNSET);
	h.n_devices = 2;
	h.devices[1] =
		single_device("vulkan_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET).devices[0];
	record_hcs("validate.two_accelerators_rejected", validate_has(&h, "2 non-host devices"), "");

	h  = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	ok = hcs_validate(&h, "/nonexistent/model.gguf", err, sizeof(err)) < 0 &&
		 strstr(err, "cannot validate HCS: model '/nonexistent/model.gguf' unreadable") != NULL;
	record_hcs("validate.unreadable_model", ok, "err='%s'", err);

	config cfg = config_defaults();
	h		   = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, 10);
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	ok = strcmp(cfg.device, "cpu_0") == 0 && cfg.ngl == -1 && cfg.moe_cache_cap == 10;
	record_hcs("materialize.v1", ok, "device=%s ngl=%d cap=%d", cfg.device, cfg.ngl,
			   cfg.moe_cache_cap);
}

void run_hcs_tests(void) {
	test_count_and_tasks();
	test_loader();
	test_split();
	test_enabled_flag();
	test_legacy();
	test_validate();
}
