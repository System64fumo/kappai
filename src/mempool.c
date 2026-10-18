#include "mempool.h"
#include <stdlib.h>
#include <string.h>

static bool shares_host_memory(const backend *b) {
	return backend_has_cap(b, BCAP_IS_HOST) || backend_has_cap(b, BCAP_UNIFIED_MEMORY);
}

static int compare_discrete(const void *a, const void *b) {
	const backend *x = *(backend *const *)a;
	const backend *y = *(backend *const *)b;
	if (x->priority != y->priority)
		return y->priority - x->priority;
	return x->device_index - y->device_index;
}

static void add_member(mem_pool *p, backend *b) {
	for (int i = 0; i < p->n_members; i++)
		if (p->members[i] == b)
			return;
	p->members[p->n_members++] = b;
}

void mem_topology_build(mem_topology *t, backend *const *backends, int n) {
	memset(t, 0, sizeof(*t));
	mem_pool *host = &t->pools[t->n_pools++];
	host->is_host  = true;

	backend *discrete[BACKEND_MAX];
	int		 n_discrete = 0;
	for (int i = 0; i < n; i++) {
		if (!backends[i])
			continue;
		if (!shares_host_memory(backends[i])) {
			if (n_discrete < BACKEND_MAX - 1)
				discrete[n_discrete++] = backends[i];
			continue;
		}
		add_member(host, backends[i]);
		if (backend_has_cap(backends[i], BCAP_IS_HOST) && !host->capacity_source)
			host->capacity_source = backends[i];
	}

	qsort(discrete, (size_t)n_discrete, sizeof(discrete[0]), compare_discrete);
	for (int i = 0; i < n_discrete; i++) {
		mem_pool *p		   = &t->pools[t->n_pools++];
		p->capacity_source = discrete[i];
		add_member(p, discrete[i]);
	}
}

int mem_topology_pool_of(const mem_topology *t, const backend *b) {
	for (int i = 0; i < t->n_pools; i++)
		for (int m = 0; m < t->pools[i].n_members; m++)
			if (t->pools[i].members[m] == b)
				return i;
	return -1;
}

size_t mem_pool_available(const mem_pool *p) {
	return backend_mem_available(p->capacity_source);
}

size_t mem_pool_total(const mem_pool *p) {
	return backend_mem_total(p->capacity_source);
}

size_t mem_topology_pool_available(const mem_topology *t, const backend *b) {
	int pool = mem_topology_pool_of(t, b);
	if (pool < 0)
		return backend_mem_available(b);
	return mem_pool_available(&t->pools[pool]);
}

size_t mem_topology_available(const mem_topology *t) {
	size_t total = 0;
	for (int i = 0; i < t->n_pools; i++)
		total += mem_pool_available(&t->pools[i]);
	return total;
}

size_t mem_topology_total(const mem_topology *t) {
	size_t total = 0;
	for (int i = 0; i < t->n_pools; i++)
		total += mem_pool_total(&t->pools[i]);
	return total;
}
