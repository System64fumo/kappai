#include "backend/backend.h"
#include "hcs.h"
#include "model.h"
#include "test_core.h"
#include <limits.h>

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
	int	 rc = hcs_validate(h, synth_chat_model_path, true, err, sizeof(err));
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

static hcs_config two_devices(const char *a, int layers_a, const char *b, int layers_b) {
	hcs_config h = single_device(a, HCS_TASKS_ALL, layers_a, HCS_COUNT_UNSET);
	h.n_devices	 = 2;
	h.devices[1] = single_device(b, HCS_TASKS_ALL, layers_b, HCS_COUNT_UNSET).devices[0];
	return h;
}

static void test_layer_map(void) {
	int n_layers = 0, n_experts = 0, n_kv_layers = 0;
	int probe_rc = model_probe_shape(synth_chat_model_path, &n_layers, &n_experts, &n_kv_layers);

	hcs_config h = two_devices("cpu_scalar_0", 1, "opengl_0", HCS_COUNT_ALL);
	int		   held[HCS_MAX_DEVICES];
	int		   assigned = hcs_resolve_counts(&h, HCS_FIELD_LAYERS, n_layers, held);
	int		   ok		= assigned == n_layers && held[0] == 1 && held[1] == n_layers - 1;
	record_hcs("map.resolve_star_takes_remainder", ok, "assigned=%d probe=%d n=%d", assigned,
			   probe_rc, n_layers);

	config cfg = config_defaults();
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	ok = cfg.n_layer_devices == 2 && strcmp(cfg.layer_devices[0], "opengl_0") == 0 &&
		 strcmp(cfg.layer_devices[1], "cpu_scalar_0") == 0 && cfg.layer_counts[0] == n_layers - 1 &&
		 cfg.layer_counts[1] == 1 && strcmp(cfg.device, "opengl_0") == 0;
	record_hcs("map.host_listed_first_runs_last", ok, "n=%d device=%s", cfg.n_layer_devices,
			   cfg.device);

	h	= two_devices("opengl_0", HCS_COUNT_ALL, "cpu_scalar_0", 1);
	cfg = config_defaults();
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	ok = cfg.n_layer_devices == 2 && strcmp(cfg.layer_devices[0], "opengl_0") == 0 &&
		 cfg.layer_counts[0] == n_layers - 1;
	record_hcs("map.host_listed_last_runs_last", ok, "n=%d", cfg.n_layer_devices);

	h.devices[1].layers = 0;
	cfg					= config_defaults();
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	record_hcs("map.device_without_layers_skipped", cfg.n_layer_devices == 1, "n=%d",
			   cfg.n_layer_devices);

	h			 = two_devices("vulkan_0", 1, "opengl_0", HCS_COUNT_ALL);
	h.n_devices	 = 3;
	h.devices[2] = single_device("cpu_scalar_0", HCS_TASKS_ALL, 0, HCS_COUNT_UNSET).devices[0];
	cfg			 = config_defaults();
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	ok = cfg.n_layer_devices == 2 && strcmp(cfg.layer_devices[0], "vulkan_0") == 0 &&
		 strcmp(cfg.layer_devices[1], "opengl_0") == 0;
	record_hcs("map.accelerators_keep_listing_order", ok, "n=%d", cfg.n_layer_devices);
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
	ok = hcs_validate(&h, "/nonexistent/model.gguf", true, err, sizeof(err)) == 0 &&
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
	int n_layers = 0, n_experts = 0, n_kv_layers = 0;
	if (model_probe_shape(synth_chat_model_path, &n_layers, &n_experts, &n_kv_layers) != OK) {
		record_hcs("validate.setup", 0, "cannot probe synthetic model");
		return;
	}

	hcs_config h = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	char	   err[HCS_ERR_MAX];
	int		   ok = hcs_validate(&h, synth_chat_model_path, true, err, sizeof(err)) == 0;
	record_hcs("validate.v1_baseline", ok, "err='%s'", err);

	h  = single_device("cpu_0", HCS_TASKS_ALL, n_layers, HCS_COUNT_UNSET);
	ok = hcs_validate(&h, synth_chat_model_path, true, err, sizeof(err)) == 0;
	record_hcs("validate.explicit_all_layers", ok, "err='%s'", err);

	h  = single_device("cpu_0", HCS_TASKS_ALL, 1, HCS_COUNT_UNSET);
	ok = n_layers > 1 && validate_has(&h, "add 'layers = *'");
	record_hcs("validate.under_subscription", ok, "n_layers=%d", n_layers);

	h = single_device("cpu_0", HCS_TASKS_ALL, n_layers + 1, HCS_COUNT_UNSET);
	record_hcs("validate.over_subscription", validate_has(&h, "remove 1"), "n=%d", n_layers);

	h = single_device("cpu_0", HCS_TASK_DECODE, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	record_hcs("validate.tasks_must_cover_both",
			   validate_has(&h, "must run both prefill and decode"), "");

	h = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, 4);
	record_hcs("validate.experts_on_dense", validate_has(&h, "not a MoE"), "experts=%d", n_experts);

	h = single_device("nope_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	record_hcs("validate.unknown_device", validate_has(&h, "not an available device"), "");

	h = single_device("cpu", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	record_hcs("validate.bare_name_rejected", validate_has(&h, "<backend>_<n>"), "");

	h			= single_device("cpu_scalar_0", HCS_TASKS_ALL, 1, 8);
	h.n_devices = 2;
	h.devices[1] =
		single_device("cpu_x86_64_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_ALL).devices[0];
	int expert_held[HCS_MAX_DEVICES];
	int expert_total = hcs_resolve_counts(&h, HCS_FIELD_EXPERTS, 64, expert_held);
	record_hcs("map.experts_star_takes_remainder",
			   expert_total == 64 && expert_held[0] == 8 && expert_held[1] == 56, "total=%d",
			   expert_total);

	char available[4][BACKEND_DEVICE_NAME_MAX];
	if (backend_enum_devices(available, 4) >= 2 && n_layers >= 2) {
		h  = two_devices(available[0], 1, available[1], HCS_COUNT_ALL);
		ok = hcs_validate(&h, synth_chat_model_path, true, err, sizeof(err)) == 0;
		record_hcs("validate.two_devices_accepted", ok, "err='%s'", err);
	}

	h  = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	ok = hcs_validate(&h, "/nonexistent/model.gguf", true, err, sizeof(err)) < 0 &&
		 strstr(err, "cannot validate HCS: model '/nonexistent/model.gguf' unreadable") != NULL;
	record_hcs("validate.unreadable_model", ok, "err='%s'", err);

	config cfg = config_defaults();
	h		   = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, 10);
	hcs_materialize(&h, synth_chat_model_path, &cfg);
	ok = strcmp(cfg.device, "cpu_0") == 0 && cfg.ngl == -1 && cfg.moe_cache_cap == 10;
	record_hcs("materialize.v1", ok, "device=%s ngl=%d cap=%d", cfg.device, cfg.ngl,
			   cfg.moe_cache_cap);

	if (model_probe_shape(synth_dsa_model_path, &n_layers, &n_experts, &n_kv_layers) == OK &&
		n_experts > 4) {
		config cfg2			 = config_defaults();
		int	   experts_total = n_experts;

		h  = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, 4);
		ok = hcs_validate(&h, synth_dsa_model_path, true, err, sizeof(err)) == 0;
		record_hcs("validate.experts_partial_streams", ok, "err='%s' total=%d", err, experts_total);

		hcs_materialize(&h, synth_dsa_model_path, &cfg2);
		ok = cfg2.moe_pin == 0 && cfg2.moe_cache_cap == 4;
		record_hcs("materialize.experts_lru_partial", ok, "pin=%d cap=%d", cfg2.moe_pin,
				   cfg2.moe_cache_cap);

		h = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_ALL);
		hcs_materialize(&h, synth_dsa_model_path, &cfg2);
		ok = cfg2.moe_pin == 0 && cfg2.moe_cache_cap == INT_MAX;
		record_hcs("materialize.experts_star_all_cached", ok, "pin=%d cap=%d", cfg2.moe_pin,
				   cfg2.moe_cache_cap);
	}
}

static void test_priority(void) {
	int		   n;
	char	   err[256];
	hcs_loader l;
	hcs_config h;
	int ok = hcs_parse_priority(" 3 ", &n) == 0 && n == 3 && hcs_parse_priority("0", &n) == 0 &&
			 n == 0 && hcs_parse_priority("*", &n) != 0 && hcs_parse_priority("-1", &n) != 0 &&
			 hcs_parse_priority("", &n) != 0;
	record_hcs("priority.parse", ok, "ok=%d", ok);

	hcs_loader_init(&l);
	int a = hcs_loader_open_section(&l, "cpu_0", err, sizeof(err));
	int b = hcs_loader_open_section(&l, "cpu_1", err, sizeof(err));
	ok	  = hcs_loader_set(&l, -1, "devices", "cpu_0, cpu_1", err, sizeof(err)) == 0 &&
			hcs_loader_set(&l, a, "tasks", "decode", err, sizeof(err)) == 0 &&
			hcs_loader_set(&l, a, "layers", "*", err, sizeof(err)) == 0 &&
			hcs_loader_set(&l, a, "priority", "2", err, sizeof(err)) == 0 &&
			hcs_loader_set(&l, b, "tasks", "decode", err, sizeof(err)) == 0 &&
			hcs_loader_set(&l, b, "layers", "0", err, sizeof(err)) == 0 &&
			hcs_loader_finish(&l, &h, err, sizeof(err)) == 0;
	ok &=
		ok && h.devices[0].has_priority && h.devices[0].priority == 2 && !h.devices[1].has_priority;
	ok &= hcs_loader_set(&l, a, "priority", "3", err, sizeof(err)) < 0 &&
		  strstr(err, "duplicate") != NULL;
	record_hcs("priority.loader", ok, "err='%s'", err);

	h = single_device("cpu_0", HCS_TASKS_ALL, HCS_COUNT_ALL, HCS_COUNT_UNSET);
	h.devices[0].has_priority = true;
	h.devices[0].priority	  = 1;
	ok						  = validate_has(&h, "priority");
	record_hcs("priority.rejected_for_dense", ok, "ok=%d", ok);
}

void run_hcs_tests(void) {
	test_count_and_tasks();
	test_priority();
	test_loader();
	test_layer_map();
	test_enabled_flag();
	test_legacy();
	test_validate();
}
