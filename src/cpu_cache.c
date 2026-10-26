#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "cpu_cache.h"

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CACHE_DEFAULT_LINE 64
#define CACHE_DEFAULT_L1D (32 * 1024)
#define CACHE_DEFAULT_L2 (512 * 1024)
#define CACHE_MAX_INDEX 8

cpu_cache_info g_cpu_cache = {
	.line_bytes = CACHE_DEFAULT_LINE,
	.l1d_bytes	= CACHE_DEFAULT_L1D,
	.l2_bytes	= CACHE_DEFAULT_L2,
};

static int cache_read(int cpu, int idx, const char *field, char *buf, size_t cap) {
	char path[160];
	snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/%s", cpu, idx, field);
	FILE *f = fopen(path, "r");
	if (!f)
		return 0;
	char *ok = fgets(buf, (int)cap, f);
	fclose(f);
	if (!ok)
		return 0;
	buf[strcspn(buf, "\r\n")] = '\0';
	return buf[0] != '\0';
}

static size_t cache_parse_size(const char *s) {
	char		 *end;
	unsigned long v = strtoul(s, &end, 10);
	if (*end == 'K' || *end == 'k')
		v <<= 10;
	else if (*end == 'M' || *end == 'm')
		v <<= 20;
	else if (*end == 'G' || *end == 'g')
		v <<= 30;
	return (size_t)v;
}

static int cache_count_cpus(const char *list) {
	int n = 0;
	for (const char *p = list; *p;) {
		char *end;
		long  lo = strtol(p, &end, 10);
		if (end == p)
			break;
		long hi = lo;
		if (*end == '-')
			hi = strtol(end + 1, &end, 10);
		n += (int)(hi - lo + 1);
		p = *end == ',' ? end + 1 : end;
		if (p == end && *p)
			break;
	}
	return n > 0 ? n : 1;
}

static void cache_keep_min(size_t *dst, size_t v) {
	if (v && (!*dst || v < *dst))
		*dst = v;
}

void cpu_cache_init(void) {
	cpu_set_t set;
	CPU_ZERO(&set);
	if (sched_getaffinity(0, sizeof(set), &set) != 0)
		return;

	size_t l1d = 0, l2 = 0, l3 = 0, line = 0;
	int	   n_cpus = 0;
	for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
		if (!CPU_ISSET(cpu, &set))
			continue;
		n_cpus++;
		for (int idx = 0; idx < CACHE_MAX_INDEX; idx++) {
			char level[16], type[32], size[32], ln[16], shared[128];
			if (!cache_read(cpu, idx, "level", level, sizeof(level)))
				break;
			if (!cache_read(cpu, idx, "type", type, sizeof(type)) ||
				!cache_read(cpu, idx, "size", size, sizeof(size)) || !strcmp(type, "Instruction"))
				continue;
			size_t bytes = cache_parse_size(size);
			if (cache_read(cpu, idx, "coherency_line_size", ln, sizeof(ln)))
				cache_keep_min(&line, (size_t)atoi(ln));
			switch (atoi(level)) {
			case 1:
				cache_keep_min(&l1d, bytes);
				break;
			case 2: {
				int sharers = cache_read(cpu, idx, "shared_cpu_list", shared, sizeof(shared))
								  ? cache_count_cpus(shared)
								  : 1;
				cache_keep_min(&l2, bytes / (size_t)sharers);
				break;
			}
			case 3:
				cache_keep_min(&l3, bytes);
				break;
			}
		}
	}

	g_cpu_cache.n_cpus = n_cpus;
	if (line >= 32 && line <= 256 && (line & (line - 1)) == 0)
		g_cpu_cache.line_bytes = (int)line;
	if (l1d)
		g_cpu_cache.l1d_bytes = l1d;
	if (l2)
		g_cpu_cache.l2_bytes = l2;
	g_cpu_cache.l3_bytes = l3;
	g_cpu_cache.detected = l1d || l2 || l3;
}

__attribute__((constructor)) static void cpu_cache_ctor(void) {
	cpu_cache_init();
}
