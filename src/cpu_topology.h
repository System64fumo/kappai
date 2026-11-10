#ifndef CPU_TOPOLOGY_H
#define CPU_TOPOLOGY_H

#include "config.h"

#include <stddef.h>

#define CPU_TOPOLOGY_MAX_CPUS 256

typedef struct {
	int n_cpus;
	int cpus[CPU_TOPOLOGY_MAX_CPUS];
	int max_mhz;
} cpu_cluster;

typedef struct {
	int			n_clusters;
	cpu_cluster clusters[CONFIG_MAX_CPU_CLUSTERS];
} cpu_topology;

const cpu_topology *cpu_topology_get(void);
void				cpu_cluster_describe(int cluster, char *out, size_t cap);

#endif
