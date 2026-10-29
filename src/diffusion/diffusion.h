#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "common.h"
#include "gguf.h"

#include <signal.h>

#define DIFFUSION_DIMENSION_STEP 32

typedef struct {
	int		 width, height, channels;
	uint8_t *pixels;
} diffusion_image;

typedef struct {
	const char *diffusion_model;
	const char *device;
	const char *vae;
	const char *llm;
} diffusion_files;

typedef struct {
	int	  base_seq, max_seq;
	float base_shift, max_shift, shift_terminal;
} diffusion_schedule;

typedef struct {
	int lw, lh, lc;
} latent_shape;

typedef struct diffusion_model diffusion_model;
typedef struct diffusion_cond  diffusion_cond;

typedef struct {
	const char		  *name;
	diffusion_schedule schedule;
	bool (*detect)(const gguf_ctx *dit);
	diffusion_model *(*open)(const diffusion_files *files, gguf_ctx *dit);
	void (*close)(diffusion_model *dm);
	latent_shape (*shape)(int width, int height);
	status_code (*encode)(diffusion_model *dm, const char *const *texts, int n,
						  diffusion_cond **out);
	void (*cond_free)(diffusion_cond *cond);
	status_code (*predict)(diffusion_model *dm, const float *latent, latent_shape shape,
						   float sigma, const diffusion_cond *cond, float *velocity);
	status_code (*decode)(diffusion_model *dm, const float *latent, latent_shape shape,
						  diffusion_image *out);
	void (*preview)(const float *x0, latent_shape shape, diffusion_image *out);
} diffusion_ops;

struct diffusion_model {
	const diffusion_ops *ops;
};

typedef struct {
	const char					*prompt;
	const char					*negative_prompt;
	int							 width, height, steps;
	float						 cfg_scale;
	uint64_t					 seed;
	const char					*preview_path;
	float						 preview_interval;
	const volatile sig_atomic_t *interrupt;
} diffusion_params;

void diffusion_register(const diffusion_ops *ops);

#define DIFFUSION_REGISTER(id, ops_ptr)                                                            \
	static void __attribute__((constructor)) diffusion_autoreg_##id(void) {                        \
		diffusion_register(ops_ptr);                                                               \
	}

bool			 diffusion_probe(const char *path);
const char		*diffusion_params_error(const diffusion_params *p);
diffusion_params diffusion_params_defaults(void);
diffusion_model *diffusion_open(const diffusion_files *files, char *err, size_t err_len);
void			 diffusion_close(diffusion_model *dm);

status_code diffusion_generate(diffusion_model *dm, const diffusion_params *p,
							   diffusion_image *out);

status_code diffusion_image_write_png(const char *path, const diffusion_image *img);
void		diffusion_image_free(diffusion_image *img);

#endif
