#ifndef DIFFUSION_VAE_H
#define DIFFUSION_VAE_H

#include "backend/backend.h"
#include "diffusion/diffusion.h"
#include "recipe/exec.h"

typedef struct {
	int			 channels;
	const float *mean, *std;
} vae_latent_stats;

typedef struct diffusion_vae diffusion_vae;

diffusion_vae *diffusion_vae_open(backend *be, const char *path, const vae_latent_stats *stats);
void		   diffusion_vae_close(diffusion_vae *v);
status_code	   diffusion_vae_decode(diffusion_vae *v, cg_arena *arena, const float *latent,
									latent_shape shape, diffusion_image *out);

#endif
