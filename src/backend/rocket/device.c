#include "device.h"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define PAGE_BYTES 4096UL
#define DMA_LIMIT 0xFFFFFFFFULL

static int64_t monotonic_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void gem_close(rkt_device *dev, int core, uint32_t handle) {
	struct drm_rocket_gem_close gc = {.handle = handle};
	ioctl(dev->fd[core], DRM_IOCTL_ROCKET_GEM_CLOSE, &gc);
}

int rkt_device_open(rkt_device *dev, const char *path, int n_cores, int64_t timeout_ns) {
	memset(dev, 0, sizeof(*dev));
	dev->n_cores	= n_cores;
	dev->timeout_ns = timeout_ns;
	for (int c = 0; c < n_cores; c++) {
		dev->fd[c] = open(path, O_RDWR | O_CLOEXEC);
		if (dev->fd[c] >= 0)
			continue;
		int err		 = errno;
		dev->n_cores = c;
		rkt_device_close(dev);
		return -err;
	}
	return 0;
}

void rkt_device_close(rkt_device *dev) {
	for (int c = 0; c < dev->n_cores; c++)
		if (dev->fd[c] >= 0)
			close(dev->fd[c]);
	dev->n_cores = 0;
}

int rkt_bo_create(rkt_device *dev, int core, size_t size, rkt_bo *bo) {
	memset(bo, 0, sizeof(*bo));
	size = (size + PAGE_BYTES - 1) & ~(PAGE_BYTES - 1);
	if (size > UINT32_MAX)
		return -EFBIG;

	struct drm_rocket_create_bo create = {.size = (uint32_t)size};
	if (ioctl(dev->fd[core], DRM_IOCTL_ROCKET_CREATE_BO, &create) < 0)
		return -errno;

	if (create.dma_address + size > DMA_LIMIT + 1) {
		gem_close(dev, core, create.handle);
		return -ERANGE;
	}

	void *map =
		mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd[core], (off_t)create.offset);
	if (map == MAP_FAILED) {
		int err = errno;
		gem_close(dev, core, create.handle);
		return -err;
	}

	bo->handle	 = create.handle;
	bo->dma_addr = create.dma_address;
	bo->size	 = size;
	bo->map		 = map;
	bo->core	 = core;
	return 0;
}

void rkt_bo_destroy(rkt_device *dev, rkt_bo *bo) {
	if (!bo->map)
		return;
	munmap(bo->map, bo->size);
	gem_close(dev, bo->core, bo->handle);
	memset(bo, 0, sizeof(*bo));
}

int rkt_bo_flush(rkt_device *dev, const rkt_bo *bo) {
	struct drm_rocket_fini_bo fini = {.handle = bo->handle};
	return ioctl(dev->fd[bo->core], DRM_IOCTL_ROCKET_FINI_BO, &fini) < 0 ? -errno : 0;
}

int rkt_bo_wait(rkt_device *dev, const rkt_bo *bo) {
	struct drm_rocket_prep_bo prep = {.handle	  = bo->handle,
									  .timeout_ns = monotonic_ns() + dev->timeout_ns};
	if (ioctl(dev->fd[bo->core], DRM_IOCTL_ROCKET_PREP_BO, &prep) == 0)
		return 0;
	return ioctl(dev->fd[bo->core], DRM_IOCTL_ROCKET_PREP_BO, &prep) < 0 ? -errno : 0;
}

int rkt_submit(rkt_device *dev, const uint32_t *regcmd_dma, const uint32_t *n_ops, int n_tasks,
			   const rkt_bo *regcmd_pool, const rkt_bo *in, const rkt_bo *out) {
	if (n_tasks < 1 || n_tasks > RKT_MAX_TASKS)
		return -EINVAL;

	struct drm_rocket_task tasks[RKT_MAX_TASKS];
	for (int i = 0; i < n_tasks; i++)
		tasks[i] = (struct drm_rocket_task){
			.regcmd		  = regcmd_dma[i],
			.regcmd_count = n_ops[i],
		};
	uint32_t in_handles[2]	= {in->handle, regcmd_pool->handle};
	uint32_t out_handles[1] = {out->handle};

	struct drm_rocket_job job = {
		.tasks				 = (uint64_t)(uintptr_t)tasks,
		.in_bo_handles		 = (uint64_t)(uintptr_t)in_handles,
		.out_bo_handles		 = (uint64_t)(uintptr_t)out_handles,
		.task_count			 = (uint32_t)n_tasks,
		.task_struct_size	 = sizeof(tasks[0]),
		.in_bo_handle_count	 = 2,
		.out_bo_handle_count = 1,
	};
	struct drm_rocket_submit submit = {
		.jobs			 = (uint64_t)(uintptr_t)&job,
		.job_count		 = 1,
		.job_struct_size = sizeof(job),
	};
	return ioctl(dev->fd[in->core], DRM_IOCTL_ROCKET_SUBMIT, &submit) < 0 ? -errno : 0;
}

int rkt_submit_bos(rkt_device *dev, int core, const uint32_t *regcmd_dma, const uint32_t *n_ops,
				   int n_tasks, const rkt_bo *const *in, int n_in, const rkt_bo *const *out,
				   int n_out) {
	if (n_tasks < 1 || n_tasks > RKT_MAX_TASKS || n_in > RKT_MAX_HANDLES || n_out > RKT_MAX_HANDLES)
		return -EINVAL;

	struct drm_rocket_task tasks[RKT_MAX_TASKS];
	for (int i = 0; i < n_tasks; i++)
		tasks[i] = (struct drm_rocket_task){
			.regcmd		  = regcmd_dma[i],
			.regcmd_count = n_ops[i],
		};
	uint32_t in_handles[RKT_MAX_HANDLES];
	uint32_t out_handles[RKT_MAX_HANDLES];
	for (int i = 0; i < n_in; i++)
		in_handles[i] = in[i]->handle;
	for (int i = 0; i < n_out; i++)
		out_handles[i] = out[i]->handle;

	struct drm_rocket_job job = {
		.tasks				 = (uint64_t)(uintptr_t)tasks,
		.in_bo_handles		 = (uint64_t)(uintptr_t)in_handles,
		.out_bo_handles		 = (uint64_t)(uintptr_t)out_handles,
		.task_count			 = (uint32_t)n_tasks,
		.task_struct_size	 = sizeof(tasks[0]),
		.in_bo_handle_count	 = (uint32_t)n_in,
		.out_bo_handle_count = (uint32_t)n_out,
	};
	struct drm_rocket_submit submit = {
		.jobs			 = (uint64_t)(uintptr_t)&job,
		.job_count		 = 1,
		.job_struct_size = sizeof(job),
	};
	return ioctl(dev->fd[core], DRM_IOCTL_ROCKET_SUBMIT, &submit) < 0 ? -errno : 0;
}
