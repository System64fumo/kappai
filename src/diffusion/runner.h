#ifndef DIFFUSION_RUNNER_H
#define DIFFUSION_RUNNER_H

#include "diffusion/diffusion.h"

typedef struct {
	const char *dir;
	const char *name;
	const char *path;
} diffusion_output;

int diffusion_run(const diffusion_files *files, const diffusion_params *params,
				  const diffusion_output *out, bool debug);

#endif
