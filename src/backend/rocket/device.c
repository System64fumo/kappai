#include "device.h"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define DMA_LIMIT 0xFFFFFFFFULL

static int64_t monotonic_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void gem_close(rkt_device *dev, int core, uint32_t handle) {
	struct drm_gem_close gc = {.handle = handle};
	ioctl(dev->fd[core], DRM_IOCTL_GEM_CLOSE, &gc);
}

int rkt_device_open(rkt_device *dev, const char *path, int n_cores, int64_t timeout_ns) {
	memset(dev, 0, sizeof(*dev));
	long page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0 || (page_size & (page_size - 1)) || (unsigned long)page_size > UINT32_MAX)
		return -EINVAL;
	dev->page_size	= (size_t)page_size;
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
	if (!size || size > UINT32_MAX - (dev->page_size - 1))
		return -EFBIG;
	size = (size + dev->page_size - 1) & ~(dev->page_size - 1);

	struct drm_rocket_create_bo create = {.size = (uint32_t)size};
	if (ioctl(dev->fd[core], DRM_IOCTL_ROCKET_CREATE_BO, &create) < 0)
		return -errno;

	if (create.dma_address > DMA_LIMIT + 1 - size) {
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
	return ioctl(dev->fd[bo->core], DRM_IOCTL_ROCKET_PREP_BO, &prep) < 0 ? -errno : 0;
}

int rkt_submit(rkt_device *dev, const rkt_bo *regcmd, const struct drm_rocket_task *tasks,
			   uint32_t count, const rkt_bo *in, const rkt_bo *out, const rkt_bo *weights) {
	uint32_t in_handles[3]	= {in->handle, regcmd->handle, weights->handle};
	uint32_t out_handles[1] = {out->handle};

	struct drm_rocket_job job = {
		.tasks				 = (uint64_t)(uintptr_t)tasks,
		.in_bo_handles		 = (uint64_t)(uintptr_t)in_handles,
		.out_bo_handles		 = (uint64_t)(uintptr_t)out_handles,
		.task_count			 = count,
		.task_struct_size	 = sizeof(*tasks),
		.in_bo_handle_count	 = 3,
		.out_bo_handle_count = 1,
	};
	struct drm_rocket_submit submit = {
		.jobs			 = (uint64_t)(uintptr_t)&job,
		.job_count		 = 1,
		.job_struct_size = sizeof(job),
	};
	return ioctl(dev->fd[in->core], DRM_IOCTL_ROCKET_SUBMIT, &submit) < 0 ? -errno : 0;
}
