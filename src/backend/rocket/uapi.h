#ifndef ROCKET_UAPI_H
#define ROCKET_UAPI_H

#include <stdint.h>
#include <sys/ioctl.h>

#define DRM_ROCKET_IOCTL_BASE 'd'
#define DRM_ROCKET_COMMAND_BASE 0x40

enum {
	DRM_ROCKET_CREATE_BO = 0,
	DRM_ROCKET_SUBMIT,
	DRM_ROCKET_PREP_BO,
	DRM_ROCKET_FINI_BO,
};

struct drm_rocket_create_bo {
	uint32_t size;
	uint32_t handle;
	uint64_t dma_address;
	uint64_t offset;
};

struct drm_rocket_prep_bo {
	uint32_t handle;
	uint32_t reserved;
	int64_t	 timeout_ns;
};

struct drm_rocket_fini_bo {
	uint32_t handle;
	uint32_t reserved;
};

struct drm_rocket_task {
	uint32_t regcmd;
	uint32_t regcmd_count;
};

struct drm_rocket_job {
	uint64_t tasks;
	uint64_t in_bo_handles;
	uint64_t out_bo_handles;
	uint32_t task_count;
	uint32_t task_struct_size;
	uint32_t in_bo_handle_count;
	uint32_t out_bo_handle_count;
};

struct drm_rocket_submit {
	uint64_t jobs;
	uint32_t job_count;
	uint32_t job_struct_size;
	uint64_t reserved;
};

struct drm_rocket_gem_close {
	uint32_t handle;
	uint32_t pad;
};

#define ROCKET_IOCTL_NR(n) (DRM_ROCKET_COMMAND_BASE + (n))

#define DRM_IOCTL_ROCKET_GEM_CLOSE _IOW(DRM_ROCKET_IOCTL_BASE, 0x09, struct drm_rocket_gem_close)
#define DRM_IOCTL_ROCKET_CREATE_BO                                                                 \
	_IOWR(DRM_ROCKET_IOCTL_BASE, ROCKET_IOCTL_NR(DRM_ROCKET_CREATE_BO), struct drm_rocket_create_bo)
#define DRM_IOCTL_ROCKET_SUBMIT                                                                    \
	_IOW(DRM_ROCKET_IOCTL_BASE, ROCKET_IOCTL_NR(DRM_ROCKET_SUBMIT), struct drm_rocket_submit)
#define DRM_IOCTL_ROCKET_PREP_BO                                                                   \
	_IOW(DRM_ROCKET_IOCTL_BASE, ROCKET_IOCTL_NR(DRM_ROCKET_PREP_BO), struct drm_rocket_prep_bo)
#define DRM_IOCTL_ROCKET_FINI_BO                                                                   \
	_IOW(DRM_ROCKET_IOCTL_BASE, ROCKET_IOCTL_NR(DRM_ROCKET_FINI_BO), struct drm_rocket_fini_bo)

#endif
