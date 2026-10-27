#if defined(__APPLE__)

#include "test_core.h"

#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>

static struct {
	mach_port_t			   host;
	vm_size_t			   page_size;
	vm_statistics64_data_t stats;
	mach_msg_type_number_t count;
	kern_return_t		   page_status;
	kern_return_t		   stats_status;
	uint64_t			   installed;
	size_t				   installed_size;
	int					   sysctl_status;
	int					   releases;
	int					   queries;
	int					   errors;
} memory_mock;

static kern_return_t mock_page_size(mach_port_t host, vm_size_t *size) {
	memory_mock.errors += host != memory_mock.host;
	*size = memory_mock.page_size;
	return memory_mock.page_status;
}

static kern_return_t mock_statistics(mach_port_t host, int flavor, host_info64_t out,
									 mach_msg_type_number_t *count) {
	memory_mock.errors +=
		host != memory_mock.host || flavor != HOST_VM_INFO64 || *count != HOST_VM_INFO64_COUNT;
	memcpy(out, &memory_mock.stats, sizeof(memory_mock.stats));
	*count = memory_mock.count;
	memory_mock.queries++;
	return memory_mock.stats_status;
}

static kern_return_t mock_deallocate(mach_port_t task, mach_port_t host) {
	memory_mock.errors += task != 9 || host != memory_mock.host || host == MACH_PORT_NULL;
	memory_mock.releases++;
	return KERN_SUCCESS;
}

static int mock_sysctl(const char *name, void *out, size_t *len, const void *newval,
					   size_t newlen) {
	memory_mock.errors += strcmp(name, "hw.memsize") != 0 ||
						  *len != sizeof(memory_mock.installed) || newval != NULL || newlen != 0;
	memcpy(out, &memory_mock.installed, sizeof(memory_mock.installed));
	*len = memory_mock.installed_size;
	return memory_mock.sysctl_status;
}

#undef mach_task_self
#define mach_task_self() 9
#undef mach_host_self
#define mach_host_self() memory_mock.host
#define host_page_size mock_page_size
#define host_statistics64 mock_statistics
#define mach_port_deallocate mock_deallocate
#define sysctlbyname mock_sysctl
#define get_available_memory test_available_memory
#define get_total_memory test_total_memory
#include "../host_memory.c"
#undef get_total_memory
#undef get_available_memory
#undef sysctlbyname
#undef mach_port_deallocate
#undef host_statistics64
#undef host_page_size
#undef mach_host_self
#undef mach_task_self

static void memory_expect(const char *name, size_t actual, size_t expected) {
	record_resultf(OPFAM_EDGE_CASE, name, actual == expected, "actual=%zu expected=%zu", actual,
				   expected);
}

void run_host_memory_tests(void) {
	memset(&memory_mock, 0, sizeof(memory_mock));
	memory_mock.host					= 7;
	memory_mock.page_size				= 16384;
	memory_mock.stats.free_count		= 100000;
	memory_mock.stats.inactive_count	= 500000;
	memory_mock.stats.speculative_count = 100000;
	memory_mock.count					= HOST_VM_INFO64_COUNT;
	memory_mock.installed				= UINT64_C(16) << 30;
	memory_mock.installed_size			= sizeof(memory_mock.installed);
	memory_expect("memory.apple_total", test_total_memory(), (size_t)memory_mock.installed);
	memory_expect("memory.apple_16k_pages", test_available_memory(), (size_t)600000 * 16384);
	memory_mock.page_size = 4096;
	memory_expect("memory.apple_4k_pages", test_available_memory(), (size_t)600000 * 4096);
	memory_mock.page_status = 1;
	int queries				= memory_mock.queries;
	memory_expect("memory.apple_page_failure", test_available_memory(), 0);
	memory_expect("memory.apple_skip_stats", memory_mock.queries, queries);
	memory_mock.page_status	 = KERN_SUCCESS;
	memory_mock.stats_status = 1;
	memory_expect("memory.apple_stats_failure", test_available_memory(), 0);
	memory_mock.stats_status = KERN_SUCCESS;
	memory_mock.count		 = 0;
	memory_expect("memory.apple_short_stats", test_available_memory(), 0);
	memory_mock.count	  = HOST_VM_INFO64_COUNT;
	memory_mock.page_size = 0;
	memory_expect("memory.apple_zero_page", test_available_memory(), 0);
	memory_mock.page_size			 = 16384;
	memory_mock.stats.free_count	 = UINT32_MAX;
	memory_mock.stats.inactive_count = UINT32_MAX;
	uint64_t bytes					 = (UINT64_C(2) * UINT32_MAX) * 16384;
	memory_expect("memory.apple_wide_sum", test_available_memory(),
				  bytes > SIZE_MAX ? SIZE_MAX : (size_t)bytes);
	memory_mock.page_size = (vm_size_t)SIZE_MAX;
	memory_expect("memory.apple_overflow", test_available_memory(), SIZE_MAX);
	memory_expect("memory.apple_release_ports", memory_mock.releases, 8);
	memory_mock.host = MACH_PORT_NULL;
	memory_expect("memory.apple_missing_host", test_available_memory(), 0);
	memory_expect("memory.apple_no_null_release", memory_mock.releases, 8);
	memory_mock.sysctl_status = -1;
	memory_expect("memory.apple_sysctl_failure", test_total_memory(), 0);
	memory_mock.sysctl_status  = 0;
	memory_mock.installed_size = sizeof(memory_mock.installed) - 1;
	memory_expect("memory.apple_short_total", test_total_memory(), 0);
	memory_expect("memory.apple_api_arguments", memory_mock.errors, 0);
}

#endif
