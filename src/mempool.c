#include "mempool.h"
#include "memconfig.h"
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

static bool discrete_identity_match(const backend *a, const backend *b) {
	char ia[64];
	char ib[64];
	if (backend_identity(a, ia, sizeof(ia)) != OK || backend_identity(b, ib, sizeof(ib)) != OK)
		return false;
	return ia[0] && strcmp(ia, ib) == 0;
}

void mem_topology_build(mem_topology *t, backend *const *backends, int n) {
	memset(t, 0, sizeof(*t));
	mem_pool *host = NULL;

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
		if (!host) {
			host		  = &t->pools[t->n_pools++];
			host->is_host = true;
		}
		add_member(host, backends[i]);
		if (backend_has_cap(backends[i], BCAP_IS_HOST) && !host->capacity_source)
			host->capacity_source = backends[i];
	}

	qsort(discrete, (size_t)n_discrete, sizeof(discrete[0]), compare_discrete);
	int pool_of[BACKEND_MAX];
	for (int i = 0; i < n_discrete; i++)
		pool_of[i] = i;
	for (int i = 0; i < n_discrete; i++) {
		for (int j = 0; j < i; j++) {
			if (pool_of[i] != i || !discrete_identity_match(discrete[i], discrete[j]))
				continue;
			int target = pool_of[j];
			for (int k = 0; k < n_discrete; k++)
				if (pool_of[k] == i)
					pool_of[k] = target;
		}
	}
	int reindex[BACKEND_MAX] = {0};
	int n_pools				 = 0;
	for (int i = 0; i < n_discrete; i++) {
		if (pool_of[i] != i)
			continue;
		reindex[i] = n_pools++;
	}
	for (int i = 0; i < n_discrete; i++)
		pool_of[i] = reindex[pool_of[i]];
	for (int p = 0; p < n_pools; p++) {
		mem_pool *pool = &t->pools[t->n_pools++];
		pool->is_host  = false;
		size_t best	   = 0;
		for (int i = 0; i < n_discrete; i++) {
			if (pool_of[i] != p)
				continue;
			add_member(pool, discrete[i]);
			if (!pool->capacity_source || backend_mem_total(discrete[i]) > best) {
				pool->capacity_source = discrete[i];
				best				  = backend_mem_total(discrete[i]);
			}
		}
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

size_t mem_topology_pool_total(const mem_pool *p) {
	return backend_mem_total(p->capacity_source);
}

size_t mem_pool_total(const mem_pool *p) {
	return backend_mem_total(p->capacity_source);
}

size_t mem_topology_host_available(const mem_topology *t) {
	if (t)
		for (int i = 0; i < t->n_pools; i++)
			if (t->pools[i].is_host)
				return backend_mem_available(t->pools[i].capacity_source);
	return get_available_memory();
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
