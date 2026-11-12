#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "cpu_topology.h"

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CLUSTER_PERF_TOLERANCE_PCT 2

typedef struct {
	int	 cpu;
	long capacity;
	long khz;
} cpu_sample;

static cpu_topology	  g_topology;
static pthread_once_t g_topology_once = PTHREAD_ONCE_INIT;

static int collect_allowed_cpus(int *cpus, int cap) {
	int n = 0;
#if defined(__linux__)
	cpu_set_t set;
	CPU_ZERO(&set);
	if (sched_getaffinity(0, sizeof(set), &set) == 0)
		for (int cpu = 0; cpu < CPU_SETSIZE && n < cap; cpu++)
			if (CPU_ISSET(cpu, &set))
				cpus[n++] = cpu;
	if (n > 0)
		return n;
#endif
	long online = sysconf(_SC_NPROCESSORS_ONLN);
	for (int cpu = 0; cpu < online && n < cap; cpu++)
		cpus[n++] = cpu;
	if (n == 0)
		cpus[n++] = 0;
	return n;
}

static long read_cpu_sysfs(int cpu, const char *file) {
#if defined(__linux__)
	char path[96];
	snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/%s", cpu, file);
	FILE *f = fopen(path, "r");
	if (!f)
		return -1;
	long value = -1;
	if (fscanf(f, "%ld", &value) != 1)
		value = -1;
	fclose(f);
	return value;
#else
	(void)cpu;
	(void)file;
	return -1;
#endif
}

static long read_domain_capacity(int cpu) {
#if defined(__linux__)
	char path[96];
	snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/related_cpus", cpu);
	FILE *f = fopen(path, "r");
	if (!f)
		return read_cpu_sysfs(cpu, "cpu_capacity");
	long best = read_cpu_sysfs(cpu, "cpu_capacity");
	int	 sibling;
	while (fscanf(f, "%d", &sibling) == 1) {
		long capacity = read_cpu_sysfs(sibling, "cpu_capacity");
		if (capacity > best)
			best = capacity;
	}
	fclose(f);
	return best;
#else
	return read_cpu_sysfs(cpu, "cpu_capacity");
#endif
}

static long sample_perf(const cpu_sample *s, bool use_capacity, bool use_khz) {
	if (use_capacity)
		return s->capacity;
	return use_khz ? s->khz : 0;
}

static int compare_ints(const void *a, const void *b) {
	return *(const int *)a - *(const int *)b;
}

static void cluster_finish(cpu_cluster *cluster) {
	qsort(cluster->cpus, (size_t)cluster->n_cpus, sizeof(int), compare_ints);
}

static void build_topology(void) {
	int		   cpus[CPU_TOPOLOGY_MAX_CPUS];
	cpu_sample samples[CPU_TOPOLOGY_MAX_CPUS];
	int		   n			= collect_allowed_cpus(cpus, CPU_TOPOLOGY_MAX_CPUS);
	bool	   use_capacity = true;
	bool	   use_khz		= true;
	for (int i = 0; i < n; i++) {
		samples[i].cpu		= cpus[i];
		samples[i].capacity = read_domain_capacity(cpus[i]);
		samples[i].khz		= read_cpu_sysfs(cpus[i], "cpufreq/cpuinfo_max_freq");
		use_capacity &= samples[i].capacity > 0;
		use_khz &= samples[i].khz > 0;
	}

	for (int i = 1; i < n; i++) {
		cpu_sample moving = samples[i];
		int		   j	  = i;
		while (j > 0 && sample_perf(&samples[j - 1], use_capacity, use_khz) <
							sample_perf(&moving, use_capacity, use_khz)) {
			samples[j] = samples[j - 1];
			j--;
		}
		samples[j] = moving;
	}

	long leader_perf = 0;
	for (int i = 0; i < n; i++) {
		long perf	  = sample_perf(&samples[i], use_capacity, use_khz);
		bool new_tier = g_topology.n_clusters == 0 ||
						(perf * 100 < leader_perf * (100 - CLUSTER_PERF_TOLERANCE_PCT) &&
						 g_topology.n_clusters < CONFIG_MAX_CPU_CLUSTERS);
		if (new_tier) {
			leader_perf = perf;
			g_topology.n_clusters++;
		}
		cpu_cluster *cluster			 = &g_topology.clusters[g_topology.n_clusters - 1];
		cluster->cpus[cluster->n_cpus++] = samples[i].cpu;
		int mhz							 = (int)(samples[i].khz / 1000);
		if (mhz > cluster->max_mhz)
			cluster->max_mhz = mhz;
	}
	for (int c = 0; c < g_topology.n_clusters; c++)
		cluster_finish(&g_topology.clusters[c]);
}

const cpu_topology *cpu_topology_get(void) {
	pthread_once(&g_topology_once, build_topology);
	return &g_topology;
}

void cpu_cluster_describe(int cluster, char *out, size_t cap) {
	const cpu_cluster *c = &cpu_topology_get()->clusters[cluster];
	if (c->max_mhz > 0)
		snprintf(out, cap, "%d cores @ %.1f GHz", c->n_cpus, c->max_mhz / 1000.0);
	else
		snprintf(out, cap, "%d cores", c->n_cpus);
}
