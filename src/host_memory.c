#include "memconfig.h"

#include <stdio.h>
#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#endif

size_t get_available_memory(void) {
#if defined(__linux__)
	FILE *f = fopen("/proc/meminfo", "r");
	if (!f) {
		struct sysinfo si;
		if (sysinfo(&si) == 0)
			return (size_t)si.freeram * (size_t)si.mem_unit;
		return 0;
	}
	char   line[256];
	size_t mem_available = 0;
	size_t mem_free		 = 0;
	size_t mem_cached	 = 0;
	size_t mem_buffers	 = 0;
	while (fgets(line, sizeof(line), f)) {
		if (line[0] == 'M' && strncmp(line, "MemAvailable:", 13) == 0) {
			mem_available = strtoull(line + 13, NULL, 10) * 1024;
			break;
		}
		if (strncmp(line, "MemFree:", 8) == 0)
			mem_free = strtoull(line + 8, NULL, 10) * 1024;
		else if (strncmp(line, "Cached:", 7) == 0)
			mem_cached = strtoull(line + 7, NULL, 10) * 1024;
		else if (strncmp(line, "Buffers:", 8) == 0)
			mem_buffers = strtoull(line + 8, NULL, 10) * 1024;
	}
	fclose(f);
	if (mem_available > 0)
		return mem_available;
	return mem_free + mem_cached + mem_buffers;
#elif defined(__APPLE__)
	mach_port_t host = mach_host_self();
	if (host == MACH_PORT_NULL)
		return 0;
	vm_size_t			   page_size = 0;
	vm_statistics64_data_t vm		 = {0};
	mach_msg_type_number_t count	 = HOST_VM_INFO64_COUNT;
	kern_return_t		   status	 = host_page_size(host, &page_size);
	if (status == KERN_SUCCESS)
		status = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count);
	mach_port_deallocate(mach_task_self(), host);
	if (status != KERN_SUCCESS || page_size == 0 ||
		(size_t)count * sizeof(integer_t) <
			offsetof(vm_statistics64_data_t, inactive_count) + sizeof(vm.inactive_count))
		return 0;
	uint64_t pages = (uint64_t)vm.free_count + vm.inactive_count;
	return pages > SIZE_MAX / page_size ? SIZE_MAX : (size_t)pages * (size_t)page_size;
#elif defined(_SC_AVPHYS_PAGES)
	long pages = sysconf(_SC_AVPHYS_PAGES);
	return pages > 0 ? (size_t)pages * (size_t)page_size_cached() : 0;
#else
	return 0;
#endif
}

size_t get_total_memory(void) {
#if defined(__linux__)
	struct sysinfo si;
	if (sysinfo(&si) != 0)
		return 0;
	return (size_t)si.totalram * (size_t)si.mem_unit;
#elif defined(__APPLE__)
	uint64_t total = 0;
	size_t	 len   = sizeof(total);
	if (sysctlbyname("hw.memsize", &total, &len, NULL, 0) != 0 || len != sizeof(total))
		return 0;
	return total > SIZE_MAX ? SIZE_MAX : (size_t)total;
#elif defined(_SC_PHYS_PAGES)
	long pages = sysconf(_SC_PHYS_PAGES);
	return pages > 0 ? (size_t)pages * (size_t)page_size_cached() : 0;
#else
	return 0;
#endif
}
