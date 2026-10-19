#ifndef MEMPOOL_H
#define MEMPOOL_H

#include "backend/backend.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
	backend *members[BACKEND_MAX];
	int		 n_members;
	backend *capacity_source;
	bool	 is_host;
} mem_pool;

typedef struct {
	mem_pool pools[BACKEND_MAX];
	int		 n_pools;
} mem_topology;

void   mem_topology_build(mem_topology *t, backend *const *backends, int n);
int	   mem_topology_pool_of(const mem_topology *t, const backend *b);
size_t mem_pool_available(const mem_pool *p);
size_t mem_pool_total(const mem_pool *p);
size_t mem_topology_host_available(const mem_topology *t);
size_t mem_topology_pool_available(const mem_topology *t, const backend *b);
size_t mem_topology_available(const mem_topology *t);
size_t mem_topology_total(const mem_topology *t);

#endif
