#include "diffusion/image.h"

#define OPAQUE_ALPHA 255
#define PREVIEW_CHANNELS 4
#define PREVIEW_RGB 3

static float texel(const diffusion_image *img, int y, int x, int c) {
	return img->pixels[((size_t)y * img->width + x) * img->channels + c];
}

void image_resize(const diffusion_image *src, int width, int height, diffusion_image *dst) {
	dst->width	  = width;
	dst->height	  = height;
	dst->channels = src->channels;
	dst->pixels	  = xmalloc((size_t)width * height * src->channels);
	for (int y = 0; y < height; y++) {
		float fy = ((float)y + 0.5f) * src->height / height - 0.5f;
		fy		 = MAX(fy, 0.0f);
		int	  y0 = MIN((int)fy, src->height - 1);
		int	  y1 = MIN(y0 + 1, src->height - 1);
		float wy = fy - (float)y0;
		for (int x = 0; x < width; x++) {
			float fx = ((float)x + 0.5f) * src->width / width - 0.5f;
			fx		 = MAX(fx, 0.0f);
			int	  x0 = MIN((int)fx, src->width - 1);
			int	  x1 = MIN(x0 + 1, src->width - 1);
			float wx = fx - (float)x0;
			for (int c = 0; c < src->channels; c++) {
				float top = texel(src, y0, x0, c) * (1 - wx) + texel(src, y0, x1, c) * wx;
				float bot = texel(src, y1, x0, c) * (1 - wx) + texel(src, y1, x1, c) * wx;
				dst->pixels[((size_t)y * width + x) * src->channels + c] =
					(uint8_t)(top * (1 - wy) + bot * wy + 0.5f);
			}
		}
	}
}

void image_drop_opaque_alpha(diffusion_image *img) {
	if (img->channels != 4)
		return;
	size_t n = (size_t)img->width * img->height;
	for (size_t i = 0; i < n; i++)
		if (img->pixels[i * 4 + 3] != OPAQUE_ALPHA)
			return;
	for (size_t i = 0; i < n; i++)
		memcpy(img->pixels + i * 3, img->pixels + i * 4, 3);
	img->channels = 3;
}

uint8_t image_unit_to_byte(float v) {
	return (uint8_t)(MAX(0.0f, MIN(1.0f, v)) * 255.0f + 0.5f);
}

void diffusion_preview_linear(const latent_preview_proj *proj, const float *x0, latent_shape shape,
							  diffusion_image *out) {
	int pixels	  = shape.lw * shape.lh;
	out->width	  = shape.lw;
	out->height	  = shape.lh;
	out->channels = PREVIEW_CHANNELS;
	out->pixels	  = xmalloc((size_t)pixels * PREVIEW_CHANNELS);
	for (int p = 0; p < pixels; p++) {
		const float *z = x0 + (size_t)p * shape.lc;
		float		 c[PREVIEW_CHANNELS];
		memcpy(c, proj->rgb_bias, PREVIEW_RGB * sizeof(float));
		c[PREVIEW_RGB] = proj->alpha_bias;
		for (int d = 0; d < shape.lc; d++) {
			for (int j = 0; j < PREVIEW_RGB; j++)
				c[j] += z[d] * proj->rgb_proj[d * PREVIEW_RGB + j];
			c[PREVIEW_RGB] += z[d] * proj->alpha_proj[d];
		}
		for (int j = 0; j < PREVIEW_CHANNELS; j++)
			out->pixels[(size_t)p * PREVIEW_CHANNELS + j] =
				image_unit_to_byte(j < PREVIEW_RGB ? c[j] * 0.5f + 0.5f : c[j]);
	}
}
