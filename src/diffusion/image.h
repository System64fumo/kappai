#ifndef DIFFUSION_IMAGE_H
#define DIFFUSION_IMAGE_H

#include "diffusion/diffusion.h"

typedef struct {
	const float *rgb_proj, *rgb_bias, *alpha_proj;
	float		 alpha_bias;
} latent_preview_proj;

void	image_resize(const diffusion_image *src, int width, int height, diffusion_image *dst);
void	image_drop_opaque_alpha(diffusion_image *img);
uint8_t image_unit_to_byte(float v);
void diffusion_preview_linear(const latent_preview_proj *proj, const float *x0, latent_shape shape,
							  diffusion_image *out);

#endif
