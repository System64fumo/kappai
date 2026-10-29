#ifndef ROCKET_DEVICE_H
#define ROCKET_DEVICE_H

#include <drm/rocket_accel.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/ioctl.h>

#define RKT_MAX_CORES 3

typedef struct {
	uint32_t handle;
	uint64_t dma_addr;
	size_t	 size;
	void	*map;
	int		 core;
} rkt_bo;

typedef struct {
	int		fd[RKT_MAX_CORES];
	int		n_cores;
	size_t	page_size;
	int64_t timeout_ns;
} rkt_device;

int	 rkt_device_open(rkt_device *dev, const char *path, int n_cores, int64_t timeout_ns);
void rkt_device_close(rkt_device *dev);

int	 rkt_bo_create(rkt_device *dev, int core, size_t size, rkt_bo *bo);
void rkt_bo_destroy(rkt_device *dev, rkt_bo *bo);
int	 rkt_bo_flush(rkt_device *dev, const rkt_bo *bo);
int	 rkt_bo_wait(rkt_device *dev, const rkt_bo *bo);

int rkt_submit(rkt_device *dev, const rkt_bo *regcmd, const struct drm_rocket_task *tasks,
			   uint32_t count, const rkt_bo *in, const rkt_bo *out, const rkt_bo *weights);

#endif
