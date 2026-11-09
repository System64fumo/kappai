#ifndef ROCKET_DEVICE_H
#define ROCKET_DEVICE_H

#include "uapi.h"
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#define RKT_MAX_CORES 3
#define RKT_MAX_TASKS 64
#define RKT_MAX_HANDLES 64

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
	int64_t timeout_ns;
} rkt_device;

int	 rkt_device_open(rkt_device *dev, const char *path, int n_cores, int64_t timeout_ns);
void rkt_device_close(rkt_device *dev);

int	 rkt_bo_create(rkt_device *dev, int core, size_t size, rkt_bo *bo);
void rkt_bo_destroy(rkt_device *dev, rkt_bo *bo);
int	 rkt_bo_flush(rkt_device *dev, const rkt_bo *bo);
int	 rkt_bo_wait(rkt_device *dev, const rkt_bo *bo);

int rkt_submit_bos(rkt_device *dev, int core, const uint32_t *regcmd_dma, const uint32_t *n_ops,
				   int n_tasks, const rkt_bo *const *in, int n_in, const rkt_bo *const *out,
				   int n_out);
int rkt_submit(rkt_device *dev, const uint32_t *regcmd_dma, const uint32_t *n_ops, int n_tasks,
			   const rkt_bo *regcmd_pool, const rkt_bo *in, const rkt_bo *out);

#endif
