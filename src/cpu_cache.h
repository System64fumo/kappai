#ifndef CPU_CACHE_H
#define CPU_CACHE_H

#include <stddef.h>

typedef struct {
	int	   line_bytes;
	size_t l1d_bytes;
	size_t l2_bytes;
	size_t l3_bytes;
	int	   n_cpus;
	int	   detected;
} cpu_cache_info;

extern cpu_cache_info g_cpu_cache;

void cpu_cache_init(void);

#endif
