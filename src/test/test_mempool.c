#include "memory.h"
#include "test_core.h"

#define record_mempool(label, ok, ...) record_resultf(OPFAM_MEMPOOL, (label), (ok), __VA_ARGS__)

#define GB ((size_t)1 << 30)

typedef struct {
	size_t available;
	size_t total;
} fake_mem;

static fake_mem g_fake[BACKEND_MAX];

static size_t fake_available(backend *self) {
	return g_fake[self->device_index].available;
}

static size_t fake_total(backend *self) {
	return g_fake[self->device_index].total;
}

static backend fake_backend(int slot, int priority, uint64_t caps, size_t available, size_t total) {
	g_fake[slot] = (fake_mem){available, total};
	return (backend){
		.priority	   = priority,
		.device_index  = slot,
		.caps		   = caps,
		.mem_available = fake_available,
		.mem_total	   = fake_total,
	};
}

static void test_host_only(void) {
	backend		 cpu   = fake_backend(0, 0, BCAP_IS_HOST, 60 * GB, 64 * GB);
	backend		*all[] = {&cpu};
	mem_topology t;
	mem_topology_build(&t, all, 1);
	int ok = t.n_pools == 1 && t.pools[0].is_host && mem_topology_pool_of(&t, &cpu) == 0;
	ok &= mem_topology_available(&t) == 60 * GB && mem_topology_total(&t) == 64 * GB;
	record_mempool("host_only", ok, "pools=%d", t.n_pools);
}

static void test_unified_gpu_shares_host_pool(void) {
	backend		 cpu   = fake_backend(0, 0, BCAP_IS_HOST, 60 * GB, 64 * GB);
	backend		 igpu  = fake_backend(1, 100, BCAP_UNIFIED_MEMORY, 999 * GB, 999 * GB);
	backend		*all[] = {&cpu, &igpu};
	mem_topology t;
	mem_topology_build(&t, all, 2);
	int ok = t.n_pools == 1 && t.pools[0].n_members == 2;
	ok &= mem_topology_pool_of(&t, &igpu) == 0 && mem_topology_available(&t) == 60 * GB;
	record_mempool("unified_gpu_shares_host", ok, "pools=%d avail=%zu", t.n_pools,
				   mem_topology_available(&t));
}

static void test_unified_gpu_without_host_member(void) {
	backend		 igpu  = fake_backend(0, 100, BCAP_UNIFIED_MEMORY, 999 * GB, 999 * GB);
	backend		*all[] = {&igpu};
	mem_topology t;
	mem_topology_build(&t, all, 1);
	int ok = t.n_pools == 1 && t.pools[0].is_host && mem_topology_pool_of(&t, &igpu) == 0;
	ok &= mem_topology_available(&t) == 999 * GB && mem_topology_total(&t) == 999 * GB;
	record_mempool("unified_gpu_no_host_member", ok, "pools=%d", t.n_pools);
}

static void test_host_capacity_source_order(void) {
	backend		 cpu   = fake_backend(0, 0, BCAP_IS_HOST, 60 * GB, 64 * GB);
	backend		 igpu  = fake_backend(1, 100, BCAP_UNIFIED_MEMORY, 999 * GB, 999 * GB);
	backend		*all[] = {&igpu, &cpu};
	mem_topology t;
	mem_topology_build(&t, all, 2);
	int ok = t.n_pools == 1 && t.pools[0].capacity_source == &cpu;
	ok &= mem_topology_available(&t) == 60 * GB && mem_topology_total(&t) == 64 * GB;
	record_mempool("host_capacity_source_order", ok, "pools=%d", t.n_pools);
}

static void test_unknown_backend_memory(void) {
	backend		 unknown = {0};
	backend		*all[]	 = {&unknown};
	mem_topology t;
	mem_topology_build(&t, all, 1);
	int ok = backend_mem_available(&unknown) == 0 && backend_mem_total(&unknown) == 0;
	ok &= backend_mem_available(NULL) == 0 && backend_mem_total(NULL) == 0;
	ok &= mem_topology_available(&t) == 0 && mem_topology_total(&t) == 0;
	record_mempool("unknown_backend_memory", ok, "pools=%d", t.n_pools);
}

static void test_discrete_gpu_adds_pool(void) {
	backend		 cpu   = fake_backend(0, 0, BCAP_IS_HOST, 60 * GB, 64 * GB);
	backend		 dgpu  = fake_backend(1, 100, 0, 7 * GB, 8 * GB);
	backend		*all[] = {&cpu, &dgpu};
	mem_topology t;
	mem_topology_build(&t, all, 2);
	int ok = t.n_pools == 2 && !t.pools[1].is_host;
	ok &= mem_topology_pool_of(&t, &cpu) == 0 && mem_topology_pool_of(&t, &dgpu) == 1;
	ok &= mem_pool_available(&t.pools[1]) == 7 * GB;
	ok &= mem_topology_available(&t) == 67 * GB && mem_topology_total(&t) == 72 * GB;
	record_mempool("discrete_gpu_adds_pool", ok, "pools=%d", t.n_pools);
}

static void test_igpu_and_dgpu(void) {
	backend		 cpu   = fake_backend(0, 0, BCAP_IS_HOST, 60 * GB, 64 * GB);
	backend		 igpu  = fake_backend(1, 100, BCAP_UNIFIED_MEMORY, 999 * GB, 999 * GB);
	backend		 dgpu  = fake_backend(2, 100, 0, 7 * GB, 8 * GB);
	backend		*all[] = {&cpu, &igpu, &dgpu};
	mem_topology t;
	mem_topology_build(&t, all, 3);
	int ok = t.n_pools == 2 && t.pools[0].n_members == 2 && t.pools[1].n_members == 1;
	ok &= mem_topology_pool_of(&t, &igpu) == 0 && mem_topology_pool_of(&t, &dgpu) == 1;
	ok &= mem_topology_available(&t) == 67 * GB;
	record_mempool("igpu_and_dgpu", ok, "pools=%d", t.n_pools);
}

static void test_discrete_ordering(void) {
	backend		 cpu   = fake_backend(0, 0, BCAP_IS_HOST, 60 * GB, 64 * GB);
	backend		 slow  = fake_backend(1, 80, 0, 4 * GB, 4 * GB);
	backend		 fast1 = fake_backend(3, 100, 0, 8 * GB, 8 * GB);
	backend		 fast0 = fake_backend(2, 100, 0, 8 * GB, 8 * GB);
	backend		*all[] = {&cpu, &slow, &fast1, &fast0};
	mem_topology t;
	mem_topology_build(&t, all, 4);
	int ok = t.n_pools == 4 && mem_topology_pool_of(&t, &fast0) == 1;
	ok &= mem_topology_pool_of(&t, &fast1) == 2 && mem_topology_pool_of(&t, &slow) == 3;
	record_mempool("discrete_ordering", ok, "fast0=%d fast1=%d slow=%d",
				   mem_topology_pool_of(&t, &fast0), mem_topology_pool_of(&t, &fast1),
				   mem_topology_pool_of(&t, &slow));
}

static void test_duplicate_and_null_backends(void) {
	backend		 cpu   = fake_backend(0, 0, BCAP_IS_HOST, 60 * GB, 64 * GB);
	backend		*all[] = {&cpu, NULL, &cpu};
	mem_topology t;
	mem_topology_build(&t, all, 3);
	int ok = t.n_pools == 1 && t.pools[0].n_members == 1;
	record_mempool("duplicate_and_null", ok, "members=%d", t.pools[0].n_members);
}

void run_mempool_tests(void) {
	test_host_only();
	test_unified_gpu_shares_host_pool();
	test_unified_gpu_without_host_member();
	test_host_capacity_source_order();
	test_unknown_backend_memory();
	test_discrete_gpu_adds_pool();
	test_igpu_and_dgpu();
	test_discrete_ordering();
	test_duplicate_and_null_backends();
}
