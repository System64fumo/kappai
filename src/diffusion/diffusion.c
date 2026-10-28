#include "diffusion/diffusion.h"
#include "diffusion/image.h"
#include "log.h"
#include "profile.h"

#include <math.h>
#include <stdio.h>

#define DIFFUSION_MAX_MODELS 8

static const diffusion_ops *g_registry[DIFFUSION_MAX_MODELS];
static int					g_registry_count;

void diffusion_register(const diffusion_ops *ops) {
	if (g_registry_count < DIFFUSION_MAX_MODELS)
		g_registry[g_registry_count++] = ops;
}

diffusion_model *diffusion_open(const diffusion_files *files, char *err, size_t err_len) {
	gguf_ctx dit;
	if (gguf_load(&dit, files->diffusion_model) != OK) {
		snprintf(err, err_len, "cannot load '%s'", files->diffusion_model);
		return NULL;
	}
	for (int i = 0; i < g_registry_count; i++) {
		if (!g_registry[i]->detect(&dit))
			continue;
		diffusion_model *dm = g_registry[i]->open(files, &dit);
		if (!dm)
			snprintf(err, err_len, "failed to load %s", g_registry[i]->name);
		return dm;
	}
	snprintf(err, err_len, "'%s' is not a supported diffusion model", files->diffusion_model);
	gguf_free(&dit);
	return NULL;
}

void diffusion_close(diffusion_model *dm) {
	if (dm)
		dm->ops->close(dm);
}

static void compute_sigmas(const diffusion_schedule *s, int seq_len, int steps, float *sigmas) {
	float slope = (s->max_shift - s->base_shift) / (float)(s->max_seq - s->base_seq);
	float mu	= (float)seq_len * slope + (s->base_shift - slope * (float)s->base_seq);
	for (int i = 0; i < steps; i++) {
		float t	  = 1.0f - (float)i / (float)steps;
		sigmas[i] = t <= 0 ? 0 : expf(mu) / (expf(mu) + (1.0f / t - 1.0f));
	}
	if (s->shift_terminal > 0 && steps > 1) {
		float scale = (1.0f - sigmas[steps - 1]) / (1.0f - s->shift_terminal);
		for (int i = 0; i < steps; i++)
			sigmas[i] = 1.0f - (1.0f - sigmas[i]) / scale;
	}
	sigmas[steps] = 0;
}

static uint64_t rng_next(uint64_t *state) {
	uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
	z		   = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
	z		   = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31);
}

static void fill_noise(float *x, size_t n, uint64_t seed) {
	uint64_t state = seed;
	for (size_t i = 0; i < n; i += 2) {
		double u1 = ((double)(rng_next(&state) >> 11) + 1.0) / 9007199254740993.0;
		double u2 = (double)(rng_next(&state) >> 11) / 9007199254740992.0;
		double r  = sqrt(-2.0 * log(u1));
		x[i]	  = (float)(r * cos(2.0 * M_PI * u2));
		if (i + 1 < n)
			x[i + 1] = (float)(r * sin(2.0 * M_PI * u2));
	}
}

static void write_preview(const diffusion_ops *ops, const diffusion_params *p, const float *x0,
						  latent_shape shape) {
	diffusion_image small = {0}, full = {0};
	ops->preview(x0, shape, &small);
	image_resize(&small, p->width, p->height, &full);
	image_drop_opaque_alpha(&full);
	diffusion_image_write_png(p->preview_path, &full);
	diffusion_image_free(&small);
	diffusion_image_free(&full);
}

static void log_range(const char *what, const float *x, size_t n) {
	float lo = INFINITY, hi = -INFINITY;
	for (size_t i = 0; i < n; i++) {
		lo = MIN(lo, x[i]);
		hi = MAX(hi, x[i]);
	}
	DEBUG("%s range [%g, %g]", what, lo, hi);
}

static status_code predict_guided(const diffusion_ops *ops, diffusion_model *dm, const float *x,
								  latent_shape shape, float sigma, float cfg_scale,
								  diffusion_cond **cond, float *v, float *v_uncond) {
	size_t		n = (size_t)shape.lw * shape.lh * shape.lc;
	status_code s = ops->predict(dm, x, shape, sigma, cond[0], v);
	if (s != OK || cfg_scale == 1.0f)
		return s;
	s = ops->predict(dm, x, shape, sigma, cond[1], v_uncond);
	for (size_t i = 0; s == OK && i < n; i++)
		v[i] = v_uncond[i] + cfg_scale * (v[i] - v_uncond[i]);
	return s;
}

status_code diffusion_generate(diffusion_model *dm, const diffusion_params *p,
							   diffusion_image *out) {
	const diffusion_ops *ops	 = dm->ops;
	latent_shape		 shape	 = ops->shape(p->width, p->height);
	size_t				 n		 = (size_t)shape.lw * shape.lh * shape.lc;
	bool				 guided	 = p->cfg_scale != 1.0f;
	const char			*texts[] = {p->prompt, p->negative_prompt[0] ? p->negative_prompt : " "};

	diffusion_cond *cond[2] = {0};
	status_code		s		= ops->encode(dm, texts, guided ? 2 : 1, cond);
	if (s != OK)
		return s;

	float *x		= xmalloc(n * sizeof(float));
	float *v		= xmalloc(n * sizeof(float));
	float *v_uncond = guided ? xmalloc(n * sizeof(float)) : NULL;
	float *x0		= xmalloc(n * sizeof(float));
	float *sigmas	= xmalloc((size_t)(p->steps + 1) * sizeof(float));
	compute_sigmas(&ops->schedule, shape.lw * shape.lh, p->steps, sigmas);
	fill_noise(x, n, p->seed);

	uint64_t last_preview = 0;
	for (int i = 0; i < p->steps && s == OK; i++) {
		uint64_t t0 = time_us();
		s			= predict_guided(ops, dm, x, shape, sigmas[i], p->cfg_scale, cond, v, v_uncond);
		if (s != OK)
			break;
		log_range("velocity", v, n);
		for (size_t j = 0; j < n; j++)
			x0[j] = x[j] - sigmas[i] * v[j];
		uint64_t now = time_us();
		if (p->preview_path &&
			(!last_preview || (double)(now - last_preview) / 1e6 >= p->preview_interval)) {
			write_preview(ops, p, x0, shape);
			last_preview = time_us();
		}
		for (size_t j = 0; j < n; j++)
			x[j] += (sigmas[i + 1] - sigmas[i]) * v[j];
		INFO("step %d/%d sigma %.4f (%.1fs)", i + 1, p->steps, sigmas[i], (double)(now - t0) / 1e6);
	}

	for (int i = 0; i < 2; i++)
		if (cond[i])
			ops->cond_free(cond[i]);
	if (s == OK) {
		log_range("latent", x, n);
		s = ops->decode(dm, x, shape, out);
		if (s == OK)
			image_drop_opaque_alpha(out);
	}
	free(x);
	free(v);
	free(v_uncond);
	free(x0);
	free(sigmas);
	return s;
}
