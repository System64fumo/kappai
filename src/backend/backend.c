#include "backend.h"
#include "log.h"
#include "memconfig.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	char			name[32];
	backend_ctor_fn ctor;
} backend_reg_entry;

static backend_reg_entry g_registry[BACKEND_MAX];
static int				 g_registry_count = 0;

void backend_register(const char *name, backend_ctor_fn ctor) {
	if (g_registry_count >= BACKEND_MAX)
		return;
	backend_reg_entry *e = &g_registry[g_registry_count++];
	snprintf(e->name, sizeof(e->name), "%s", name);
	e->ctor = ctor;
}

int backend_list(backend_info *out, int max) {
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
		{"ffn_activate", b->ffn_activate != NULL},
		{"ffn_activate_ex", b->ffn_activate_ex != NULL},
		{"argmax", b->argmax != NULL},
	};
	int	 n_native = 0;
	int	 n_total  = (int)ARRAY_LEN(ops);
	char missing[512];
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
		WARN("backend '%s': %d/%d core ops native, falling back to host for: %s", b->name, n_native,
			 n_total, missing);
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
	for (int i = 0; i < g_registry_count; i++) {
		if (strcmp(g_registry[i].name, name) != 0)
			continue;
		return make_backend(&g_registry[i], device_index, out);
	}
	return ERR_NOT_FOUND;
}

status_code backend_create_best(backend **out) {
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
		return backend_create("cpu", 0, out);
	return make_backend(&g_registry[best_idx], 0, out);
}

int backend_parse_device(const char *spec, char *name, size_t name_cap, int *device_index) {
	if (!spec || !*spec)
		return -1;
	const char *end = spec + strlen(spec);
	while (end > spec && end[-1] >= '0' && end[-1] <= '9')
		end--;
	size_t name_len = (size_t)(end - spec);
	if (name_len == 0 || name_len >= name_cap)
		return -1;
	memcpy(name, spec, name_len);
	name[name_len] = '\0';
	int idx		   = 0;
	if (*end) {
		char *tail = NULL;
		long  v	   = strtol(end, &tail, 10);
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