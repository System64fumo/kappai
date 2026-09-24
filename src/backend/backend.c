#define _GNU_SOURCE
#include "backend.h"
#include "log.h"
#include "memconfig.h"
#include <dirent.h>
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HFB_MAX_ENTRIES 32
#define HFB_OP_CAP 40
#define HFB_DETAIL_CAP 160
#define HFB_PATH_CAP 512
#define HFB_LIB_MAX 16

typedef struct {
	uint64_t			 count;
	host_fallback_reason reason;
	int					 printed;
	char				 op[HFB_OP_CAP];
	char				 device[32];
	char				 detail[HFB_DETAIL_CAP];
} hfb_entry;

static hfb_entry	   g_hfb_entries[HFB_MAX_ENTRIES];
static int			   g_hfb_n_entries = 0;
static pthread_mutex_t g_hfb_mtx	   = PTHREAD_MUTEX_INITIALIZER;
static char			   g_hfb_last[192] = "";

static int g_hfb_reported = 0;
static int g_hfb_warn	  = 1;

static char g_lib_search_dirs[HFB_LIB_MAX][HFB_PATH_CAP];
static int	g_n_lib_search_dirs = 0;
static char g_loaded_lib_paths[HFB_LIB_MAX][HFB_PATH_CAP];
static int	g_n_loaded_libs = 0;
static struct {
	char path[HFB_PATH_CAP];
	char err[256];
} g_lib_load_fails[HFB_LIB_MAX];
static int g_n_lib_load_fails = 0;
static int g_backend_path_env = 0;

static void lib_note_search_dir(const char *dir) {
	if (!dir || !*dir || g_n_lib_search_dirs >= HFB_LIB_MAX)
		return;
	for (int i = 0; i < g_n_lib_search_dirs; i++)
		if (strcmp(g_lib_search_dirs[i], dir) == 0)
			return;
	snprintf(g_lib_search_dirs[g_n_lib_search_dirs++], HFB_PATH_CAP, "%s", dir);
}

static const char *hfb_reason_str(host_fallback_reason r) {
	switch (r) {
	case HFB_OP_NOT_NATIVE:
		return "not native";
	case HFB_BATCH_DESIGN:
		return "batch staging";
	case HFB_WEIGHT_TYPE:
		return "weight type not native";
	case HFB_BUF_HOST_RESIDENT:
		return "host-resident buffers";
	case HFB_CAPABILITY:
		return "missing capability";
	case HFB_ERROR:
		return "device error, retried on host";
	case HFB_LAYER_NOT_OFFLOADED:
		return "layer not offloaded";
	default:
		return "unknown";
	}
}

void backend_report_host_fallback(const backend *device, const char *op,
								  host_fallback_reason reason, const char *detail_fmt, ...) {
	if (backend_has_cap(device, BCAP_IS_HOST))
		return;

	char op_key[HFB_OP_CAP];
	snprintf(op_key, sizeof(op_key), "%s", op ? op : "unknown");

	char detail[HFB_DETAIL_CAP];
	if (detail_fmt && *detail_fmt) {
		va_list ap;
		va_start(ap, detail_fmt);
		vsnprintf(detail, sizeof(detail), detail_fmt, ap);
		va_end(ap);
	} else {
		detail[0] = '\0';
	}

	const char *dev = (device && device->name) ? device->name : "?";

	pthread_mutex_lock(&g_hfb_mtx);
	snprintf(g_hfb_last, sizeof(g_hfb_last), "op '%s' on backend '%s': %s%s%s", op_key, dev,
			 hfb_reason_str(reason), detail[0] ? " -- " : "", detail);

	hfb_entry *e = NULL;
	for (int i = 0; i < g_hfb_n_entries; i++) {
		hfb_entry *cand = &g_hfb_entries[i];
		if (cand->reason == reason && strcmp(cand->op, op_key) == 0 &&
			strcmp(cand->device, dev) == 0) {
			e = cand;
			break;
		}
	}
	int first = 0;
	if (!e && g_hfb_n_entries < HFB_MAX_ENTRIES) {
		e = &g_hfb_entries[g_hfb_n_entries++];
		snprintf(e->op, sizeof(e->op), "%s", op_key);
		snprintf(e->device, sizeof(e->device), "%s", dev);
		e->reason	 = reason;
		e->count	 = 0;
		e->detail[0] = '\0';
		first		 = 1;
	}
	if (e) {
		first = first || (e->count == 0);
		if (first && detail[0])
			snprintf(e->detail, sizeof(e->detail), "%s", detail);
		e->count++;
		if (first && !e->printed && (g_hfb_reported || reason == HFB_ERROR)) {
			if (g_hfb_warn)
				log_msg(LOG_WARN, "host fallback: %s on '%s': %s", op_key, dev,
						hfb_reason_str(reason));
			e->printed = 1;
		}
	}
	pthread_mutex_unlock(&g_hfb_mtx);
}

void backend_set_fallback_warn(int enable) {
	pthread_mutex_lock(&g_hfb_mtx);
	g_hfb_warn = enable ? 1 : 0;
	pthread_mutex_unlock(&g_hfb_mtx);
}

void backend_fallback_report(void) {
	pthread_mutex_lock(&g_hfb_mtx);
	g_hfb_reported = 1;
	int n_new	   = 0;
	for (int i = 0; i < g_hfb_n_entries; i++) {
		hfb_entry *e = &g_hfb_entries[i];
		if (e->printed)
			continue;
		e->printed = 1;
		n_new++;
		if (g_hfb_warn)
			log_msg(LOG_WARN, "%s on '%s': %s", e->op, e->device, hfb_reason_str(e->reason));
	}
	pthread_mutex_unlock(&g_hfb_mtx);
	if (n_new == 0)
		return;
	backend_host();
}

static void backend_dump_load_diagnosis(void) {
	if (g_n_lib_search_dirs > 0) {
		char dirs[HFB_LIB_MAX * HFB_PATH_CAP];
		dirs[0]	   = '\0';
		size_t off = 0;
		for (int i = 0; i < g_n_lib_search_dirs; i++) {
			int w = snprintf(dirs + off, sizeof(dirs) - off, "%s%s", i ? "; " : "",
							 g_lib_search_dirs[i]);
			if (w < 0 || (size_t)w >= sizeof(dirs) - off)
				break;
			off += (size_t)w;
		}
		ERROR("  backend library search dirs: %s", dirs);
	} else {
		ERROR("  backend library search dirs: none");
	}
	if (!g_backend_path_env)
		ERROR("  (set KAPPAI_BACKEND_PATH to the backend .so directory)");

	if (g_n_loaded_libs > 0) {
		for (int i = 0; i < g_n_loaded_libs; i++)
			ERROR("  loaded backend library: %s", g_loaded_lib_paths[i]);
	} else {
		ERROR("  no backend libraries loaded");
	}
	for (int i = 0; i < g_n_lib_load_fails; i++)
		ERROR("  failed to dlopen '%s': %s", g_lib_load_fails[i].path, g_lib_load_fails[i].err);

	if (g_hfb_last[0])
		ERROR("  host fallback was triggered by: %s", g_hfb_last);
	else
		ERROR("  no host fallback trigger recorded");
}

typedef struct {
	char			name[32];
	backend_ctor_fn ctor;
} backend_reg_entry;

static backend_reg_entry g_registry[BACKEND_MAX];
static int				 g_registry_count = 0;

void backend_register(const char *name, backend_ctor_fn ctor) {
	if (g_registry_count >= BACKEND_MAX)
		return;
	for (int i = 0; i < g_registry_count; i++) {
		if (strcmp(g_registry[i].name, name) != 0)
			continue;
		WARN("backend '%s' registered more than once; ignoring duplicate", name);
		return;
	}
	backend_reg_entry *e = &g_registry[g_registry_count++];
	snprintf(e->name, sizeof(e->name), "%s", name);
	e->ctor = ctor;
}

int backend_list(backend_info *out, int max) {
	backend_load();
	int n = MIN(g_registry_count, max);
	for (int i = 0; i < n; i++) {
		backend probe;
		memset(&probe, 0, sizeof(probe));
		snprintf(out[i].name, sizeof(out[i].name), "%s", g_registry[i].name);
		out[i].priority	 = 0;
		out[i].available = 0;
		out[i].n_devices = 1;
		out[i].caps		 = 0;
		if (g_registry[i].ctor(&probe) != OK)
			continue;
		out[i].priority	 = probe.priority;
		out[i].available = (!probe.probe || probe.probe() == OK);
		out[i].n_devices = probe.device_count ? probe.device_count() : 1;
		if (out[i].n_devices <= 0)
			out[i].available = 0;
		out[i].caps = probe.caps;
	}
	return g_registry_count;
}

static void log_op_homes(backend *b) {
	if (backend_has_cap(b, BCAP_IS_HOST))
		return;

	struct {
		const char *op;
		int			native;
	} ops[] = {
		{"kv_alloc", b->kv_alloc != NULL},
		{"kv_put", b->kv_put != NULL},
		{"embd_lookup", b->embd_lookup != NULL},
		{"rmsnorm", b->rmsnorm != NULL},
		{"rmsnorm_per_head", b->rmsnorm_per_head != NULL},
		{"rmsnorm_noweight", b->rmsnorm_noweight != NULL},
		{"rmsnorm_noweight_per_head", b->rmsnorm_noweight_per_head != NULL},
		{"matmul", b->matmul != NULL},
		{"rope", b->rope != NULL},
		{"rope_ext", b->rope_ext != NULL},
		{"rope_qk", b->rope_qk != NULL},
		{"attention", b->attention != NULL},
		{"attention_swa", b->attention_swa != NULL},
		{"add_inplace", b->add_inplace != NULL},
		{"scale_inplace", b->scale_inplace != NULL},
		{"softcap", b->softcap != NULL},
		{"split_qgate", b->split_qgate != NULL},
		{"attn_output_gate", b->attn_output_gate != NULL},
		{"partial_rope_qk", b->partial_rope_qk != NULL},
		{"ffn_activate", b->ffn_activate != NULL},
		{"ffn_activate_ex", b->ffn_activate_ex != NULL},
		{"argmax", b->argmax != NULL},
		{"rmsnorm_add", b->rmsnorm_add != NULL},
		{"matmul_residual", b->matmul_residual != NULL},
		{"matmul_multi", b->matmul_multi != NULL},
		{"kv_put_batch", b->kv_put_batch != NULL},
		{"matmul_batch", b->matmul_batch != NULL},
		{"rmsnorm_batch", b->rmsnorm_batch != NULL},
		{"rope_qk_batch", b->rope_qk_batch != NULL},
		{"attention_batch", b->attention_batch != NULL},
		{"attention_swa_batch", b->attention_swa_batch != NULL},
		{"ffn_activate_batch", b->ffn_activate_batch != NULL},
		{"matmul_multi_batch", b->matmul_multi_batch != NULL},
		{"moe_activate", b->moe_activate != NULL},
		{"moe_experts_batch", b->moe_experts_batch != NULL},
	};
	int	 n_native = 0;
	int	 n_total  = (int)ARRAY_LEN(ops);
	char missing[1024];
	missing[0] = '\0';
	for (int i = 0; i < n_total; i++) {
		if (ops[i].native) {
			n_native++;
			continue;
		}
		size_t len = strlen(missing);
		snprintf(missing + len, sizeof(missing) - len, "%s%s", len ? ", " : "", ops[i].op);
	}
	if (n_native == n_total) {
		DEBUG("backend '%s': %d/%d core ops native", b->name, n_native, n_total);
	} else {
		DEBUG("backend '%s': %d/%d core ops native", b->name, n_native, n_total);
		WARN("backend '%s': missing native ops: %s", b->name, missing);
	}

	if (b->matmul_type_native) {
		static const struct {
			uint32_t	t;
			const char *n;
		} types[] = {
			{GGML_TYPE_F32, "f32"},		{GGML_TYPE_F16, "f16"},	  {GGML_TYPE_BF16, "bf16"},
			{GGML_TYPE_Q4_0, "q4_0"},	{GGML_TYPE_Q4_1, "q4_1"}, {GGML_TYPE_Q5_0, "q5_0"},
			{GGML_TYPE_Q5_1, "q5_1"},	{GGML_TYPE_Q8_0, "q8_0"}, {GGML_TYPE_Q4_K, "q4_k"},
			{GGML_TYPE_Q5_K, "q5_k"},	{GGML_TYPE_Q6_K, "q6_k"}, {GGML_TYPE_IQ4_NL, "iq4_nl"},
			{GGML_TYPE_IQ3_S, "iq3_s"},
		};
		char native[256], nonnative[256];
		native[0] = nonnative[0] = '\0';
		for (size_t i = 0; i < ARRAY_LEN(types); i++) {
			int	   is_native = b->matmul_type_native((backend *)b, types[i].t);
			char  *dst		 = is_native ? native : nonnative;
			size_t len		 = strlen(dst);
			snprintf(dst + len, is_native ? sizeof(native) : sizeof(nonnative), "%s%s",
					 len ? ", " : "", types[i].n);
		}
		DEBUG("backend '%s': matmul native types: %s", b->name, native[0] ? native : "(none)");
		if (nonnative[0])
			WARN("backend '%s': matmul host-only types: %s", b->name, nonnative);
	}
}

static status_code make_backend(backend_reg_entry *e, int device_index, backend **out) {
	backend	   *b = xcalloc(1, sizeof(backend));
	status_code s = e->ctor(b);
	if (s != OK) {
		free(b);
		return s;
	}

	if (b->probe && b->probe() != OK) {
		free(b);
		return ERR_UNSUPPORTED;
	}

	s = b->init(b, device_index);
	if (s != OK) {
		free(b);
		return s;
	}

	log_op_homes(b);

	*out = b;
	return OK;
}

status_code backend_create(const char *name, int device_index, backend **out) {
	backend_load();
	if (strcmp(name, "cpu") == 0)
		return backend_create_host(out);
	for (int i = 0; i < g_registry_count; i++) {
		if (strcmp(g_registry[i].name, name) != 0)
			continue;
		return make_backend(&g_registry[i], device_index, out);
	}
	return ERR_NOT_FOUND;
}

status_code backend_create_host(backend **out) {
	backend_load();
	int				best_priority = -1;
	backend_ctor_fn best_ctor	  = NULL;

	for (int i = 0; i < g_registry_count; i++) {
		backend probe;
		memset(&probe, 0, sizeof(probe));
		if (g_registry[i].ctor(&probe) != OK)
			continue;
		if (!(probe.caps & BCAP_IS_HOST))
			continue;
		if (probe.probe && probe.probe() != OK)
			continue;
		if (probe.priority <= best_priority)
			continue;
		best_priority = probe.priority;
		best_ctor	  = g_registry[i].ctor;
	}

	if (!best_ctor) {
		ERROR("no host (cpu) backend available; was the cpu backend library built?");
		for (int i = 0; i < g_registry_count; i++) {
			backend probe;
			memset(&probe, 0, sizeof(probe));
			if (g_registry[i].ctor(&probe) != OK) {
				ERROR("  registered backend '%s': constructor failed", g_registry[i].name);
				continue;
			}
			if (!(probe.caps & BCAP_IS_HOST)) {
				ERROR("  backend '%s': not host-capable (caps=0x%llx)", g_registry[i].name,
					  (unsigned long long)probe.caps);
				continue;
			}
			if (probe.probe && probe.probe() != OK) {
				ERROR("  registered backend '%s': host-capable but hardware probe failed",
					  g_registry[i].name);
				continue;
			}
		}
		if (g_registry_count == 0)
			ERROR("  no backend libraries registered at all");
		backend_dump_load_diagnosis();
		return ERR_NOT_FOUND;
	}
	backend	   *b = xcalloc(1, sizeof(backend));
	status_code s = best_ctor(b);
	if (s != OK) {
		free(b);
		return s;
	}
	if (b->probe && b->probe() != OK) {
		free(b);
		return ERR_UNSUPPORTED;
	}
	s = b->init(b, 0);
	if (s != OK) {
		free(b);
		return s;
	}
	log_op_homes(b);
	*out = b;
	return OK;
}

status_code backend_create_best(backend **out) {
	backend_load();
	int best_priority = -1;
	int best_idx	  = -1;

	for (int i = 0; i < g_registry_count; i++) {
		backend probe;
		memset(&probe, 0, sizeof(probe));
		if (g_registry[i].ctor(&probe) != OK)
			continue;
		if (probe.probe && probe.probe() != OK)
			continue;
		if (probe.caps & BCAP_IS_HOST)
			continue;
		if (probe.priority <= best_priority)
			continue;
		best_priority = probe.priority;
		best_idx	  = i;
	}

	if (best_idx < 0)
		return backend_create_host(out);
	return make_backend(&g_registry[best_idx], 0, out);
}

int backend_parse_device(const char *spec, char *name, size_t name_cap, int *device_index) {
	if (!spec || !*spec)
		return -1;

	backend_load();

	for (int i = 0; i < g_registry_count; i++) {
		if (strcmp(g_registry[i].name, spec) == 0) {
			snprintf(name, name_cap, "%s", spec);
			*device_index = 0;
			return 0;
		}
	}

	const char *colon = strchr(spec, ':');
	if (colon && colon != spec) {
		size_t name_len = (size_t)(colon - spec);
		if (name_len >= name_cap)
			return -1;
		char *tail = NULL;
		long  v	   = strtol(colon + 1, &tail, 10);
		if (*tail != '\0' || v < 0 || v > INT_MAX)
			return -1;
		memcpy(name, spec, name_len);
		name[name_len] = '\0';
		*device_index  = (int)v;
		return 0;
	}

	int best_len = -1;
	int best_idx = 0;
	for (int i = 0; i < g_registry_count; i++) {
		size_t len = strlen(g_registry[i].name);
		if ((int)len <= best_len)
			continue;
		if (strncmp(spec, g_registry[i].name, len) != 0)
			continue;
		const char *rem		   = spec + len;
		int			all_digits = 1;
		for (const char *p = rem; *p; p++) {
			if (*p < '0' || *p > '9') {
				all_digits = 0;
				break;
			}
		}
		if (!all_digits)
			continue;
		best_len = (int)len;
		best_idx = (*rem) ? (int)strtol(rem, NULL, 10) : 0;
	}
	if (best_len > 0) {
		snprintf(name, name_cap, "%.*s", best_len, spec);
		*device_index = best_idx;
		return 0;
	}

	const char *end_all = spec + strlen(spec);
	const char *digits	= end_all;
	while (digits > spec && digits[-1] >= '0' && digits[-1] <= '9')
		digits--;
	size_t name_len = (size_t)(digits - spec);
	if (name_len == 0 || name_len >= name_cap)
		return -1;
	memcpy(name, spec, name_len);
	name[name_len] = '\0';
	int idx		   = 0;
	if (*digits) {
		char *tail = NULL;
		long  v	   = strtol(digits, &tail, 10);
		if (*tail != '\0' || v < 0 || v > INT_MAX)
			return -1;
		idx = (int)v;
	}
	*device_index = idx;
	return 0;
}

void backend_destroy(backend *b) {
	if (!b)
		return;
	backend_destroyed(b);
	if (b->free)
		b->free(b);
	free(b);
}

static backend			 *g_host_backend  = NULL;
static pthread_once_t	  g_host_once	  = PTHREAD_ONCE_INIT;
static _Atomic(backend *) g_host_override = NULL;

void backend_host_use(backend *b) {
	if (!b || !backend_has_cap(b, BCAP_IS_HOST))
		return;
	atomic_store(&g_host_override, b);
}

void backend_destroyed(backend *b) {
	if (b && atomic_load(&g_host_override) == b)
		atomic_store(&g_host_override, NULL);
}

static void host_backend_init(void) {
	if (backend_create_host(&g_host_backend) != OK) {
		pthread_mutex_lock(&g_hfb_mtx);
		char last[192];
		snprintf(last, sizeof(last), "%s", g_hfb_last);
		pthread_mutex_unlock(&g_hfb_mtx);
		ERROR("could not create host (cpu) fallback backend");
		if (last[0])
			ERROR("required by: %s", last);
		abort();
	}
	if (atomic_load(&g_host_override) == NULL)
		atomic_store(&g_host_override, g_host_backend);
}

backend *backend_host(void) {
	backend *o = atomic_load(&g_host_override);
	if (o)
		return o;
	pthread_once(&g_host_once, host_backend_init);
	return g_host_backend;
}

backend *backend_weight_home(backend *b, weight_class wc) {
	int native;
	switch (wc) {
	case WCLASS_MATMUL:
		native = (b->matmul != NULL);
		break;
	case WCLASS_NORM:
		native = (b->rmsnorm != NULL);
		break;
	case WCLASS_EMBEDDING:
		native = (b->embd_lookup != NULL);
		break;
	case WCLASS_MISC:
	default:
		native = 1;
		break;
	}
	return native ? b : backend_host();
}

static _Atomic(int)					   g_hk_priority = -1;
static _Atomic(host_matmul_generic_fn) g_hk_matmul	 = NULL;

void host_kernels_register(int priority, host_matmul_generic_fn mm) {
	int cur = atomic_load(&g_hk_priority);
	while (priority > cur) {
		if (atomic_compare_exchange_weak(&g_hk_priority, &cur, priority)) {
			atomic_store(&g_hk_matmul, mm);
			return;
		}
	}
}

void host_matmul_generic(const void *w, uint32_t w_type, const float *x, float *y, int n, int k) {
	host_matmul_generic_fn fn = atomic_load(&g_hk_matmul);
	if (fn) {
		fn(w, w_type, x, y, n, k);
		return;
	}
	backend *host = backend_host();
	if (host && host->matmul_thread_local)
		host->matmul_thread_local(host, w, w_type, x, y, n, k, 0);
}

status_code buffer_ensure_scratch(backend *a, buffer *b, size_t bytes) {
	if (!a || !a->buffer_alloc_scratch)
		return ERR_UNSUPPORTED;
	if (b->owner == a && b->size >= bytes)
		return OK;
	if (b->owner)
		b->owner->buffer_free(b->owner, b);
	memset(b, 0, sizeof(*b));
	return a->buffer_alloc_scratch(a, bytes, b);
}

size_t backend_mem_available(const backend *b) {
	if (b && b->mem_available)
		return b->mem_available((backend *)b);
	return get_available_memory();
}

size_t backend_mem_total(const backend *b) {
	if (b && b->mem_total)
		return b->mem_total((backend *)b);
	return get_total_memory();
}
#define BACKEND_LIB_PREFIX "libkappai_"
#define BACKEND_LIB_SUFFIX ".so"

static pthread_once_t g_backends_once = PTHREAD_ONCE_INIT;

static int str_ends_with(const char *s, const char *suffix) {
	size_t ls = strlen(s), lx = strlen(suffix);
	return ls >= lx && strcmp(s + ls - lx, suffix) == 0;
}

static int str_starts_with(const char *s, const char *prefix) {
	return strncmp(s, prefix, strlen(prefix)) == 0;
}

static void path_copy_trunc(char *dst, const char *src) {
	size_t n = strlen(src);
	if (n >= HFB_PATH_CAP)
		n = HFB_PATH_CAP - 1;
	memcpy(dst, src, n);
	dst[n] = '\0';
}

static void try_load_backend(const char *path) {
	void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		const char *err = dlerror();
		WARN("backend library '%s' failed to load: %s", path, err ? err : "unknown error");
		if (g_n_lib_load_fails < HFB_LIB_MAX) {
			path_copy_trunc(g_lib_load_fails[g_n_lib_load_fails].path, path);
			snprintf(g_lib_load_fails[g_n_lib_load_fails].err, 256, "%s",
					 err ? err : "unknown error");
			g_n_lib_load_fails++;
		}
		return;
	}
	DEBUG("loaded backend library: %s", path);
	if (g_n_loaded_libs < HFB_LIB_MAX)
		path_copy_trunc(g_loaded_lib_paths[g_n_loaded_libs++], path);
}

static int already_loaded(const char *candidate) {
	static char loaded[BACKEND_MAX][4096];
	static int	n_loaded = 0;
	char		real[4096];
	if (!realpath(candidate, real))
		snprintf(real, sizeof(real), "%s", candidate);
	for (int i = 0; i < n_loaded; i++)
		if (strcmp(loaded[i], real) == 0)
			return 1;
	if (n_loaded < BACKEND_MAX)
		snprintf(loaded[n_loaded++], sizeof(loaded[0]), "%s", real);
	return 0;
}

static void load_from_dir_dedup(const char *dir) {
	lib_note_search_dir(dir);
	DIR *d = opendir(dir);
	if (!d)
		return;
	struct dirent *ent;
	while ((ent = readdir(d)) != NULL) {
		if (!str_starts_with(ent->d_name, BACKEND_LIB_PREFIX))
			continue;
		if (!str_ends_with(ent->d_name, BACKEND_LIB_SUFFIX))
			continue;
		char path[4096];
		snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
		if (!already_loaded(path))
			try_load_backend(path);
	}
	closedir(d);
}

static void engine_dir(char *out, size_t cap) {
	out[0] = '\0';
	Dl_info info;
	if (dladdr((void *)backend_register, &info) && info.dli_fname) {
		const char *slash = strrchr(info.dli_fname, '/');
		if (slash) {
			size_t n = (size_t)(slash - info.dli_fname);
			if (n >= cap)
				n = cap - 1;
			memcpy(out, info.dli_fname, n);
			out[n] = '\0';
		}
	}
}

static void backends_init(void) {
	char path[4096];

	const char *env	   = getenv("KAPPAI_BACKEND_PATH");
	g_backend_path_env = (env && *env);
	if (env && *env) {
		char *dup = strdup(env);
		if (dup) {
			char *save = NULL;
			for (char *dir = strtok_r(dup, ":", &save); dir; dir = strtok_r(NULL, ":", &save))
				load_from_dir_dedup(dir);
			free(dup);
			return;
		}
	}

	char engdir[4096];
	engine_dir(engdir, sizeof(engdir));
	if (engdir[0]) {
		snprintf(path, sizeof(path), "%s/backends", engdir);
		load_from_dir_dedup(path);
	}

	load_from_dir_dedup("build/backends");
}

void backend_load(void) {
	pthread_once(&g_backends_once, backends_init);
}