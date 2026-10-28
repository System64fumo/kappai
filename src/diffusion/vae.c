#include "diffusion/vae.h"
#include "diffusion/image.h"
#include "log.h"
#include "nn.h"
#include "recipe/exec.h"

#define VAE_STAGES 5
#define VAE_BLOCKS_PER_STAGE 3
#define VAE_RMS_EPS 1e-12f
#define VAE_UPSAMPLE 2

typedef struct {
	int		in, out;
	buffer	gamma1, gamma2;
	nn_conv conv1, conv2, shortcut;
	bool	has_shortcut;
} res_block;

typedef struct {
	int		  in, out, frames_up;
	res_block blocks[VAE_BLOCKS_PER_STAGE];
	bool	  up;
	nn_conv	  resample;
} up_stage;

struct diffusion_vae {
	backend				   *be;
	const vae_latent_stats *stats;
	gguf_ctx				g;
	nn_conv					post_quant, conv_in, head;
	res_block				mid0, mid2;
	buffer					attn_gamma;
	nn_conv					attn_q, attn_k, attn_v, attn_proj;
	up_stage				stages[VAE_STAGES];
	buffer					head_gamma;
};

static const gguf_tensor *vae_tensor(diffusion_vae *v, const char *prefix, const char *suffix) {
	char name[192];
	snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
	const gguf_tensor *t = gguf_find_tensor(&v->g, name);
	if (!t)
		ERROR("vae tensor '%s' not found", name);
	return t;
}

static status_code load_vec(diffusion_vae *v, const char *prefix, const char *suffix, buffer *out) {
	const gguf_tensor *t = vae_tensor(v, prefix, suffix);
	return t ? nn_vec_load(v->be, t, 0.0f, out) : ERR_NOT_FOUND;
}

static status_code load_conv(diffusion_vae *v, const char *prefix, nn_conv *c) {
	const gguf_tensor *w = vae_tensor(v, prefix, "weight");
	const gguf_tensor *b = vae_tensor(v, prefix, "bias");
	return w && b ? nn_conv_load(v->be, w, b, 0, 0, c) : ERR_NOT_FOUND;
}

static status_code load_res_block(diffusion_vae *v, const char *prefix, res_block *rb) {
	char name[160];
	snprintf(name, sizeof(name), "%s.residual.0", prefix);
	status_code s = load_vec(v, name, "gamma", &rb->gamma1);
	snprintf(name, sizeof(name), "%s.residual.3", prefix);
	if (s == OK)
		s = load_vec(v, name, "gamma", &rb->gamma2);
	snprintf(name, sizeof(name), "%s.residual.2", prefix);
	if (s == OK)
		s = load_conv(v, name, &rb->conv1);
	snprintf(name, sizeof(name), "%s.residual.6", prefix);
	if (s == OK)
		s = load_conv(v, name, &rb->conv2);
	rb->in	= rb->conv1.ic;
	rb->out = rb->conv2.oc;
	snprintf(name, sizeof(name), "%s.shortcut.weight", prefix);
	rb->has_shortcut = gguf_find_tensor(&v->g, name) != NULL;
	if (s == OK && rb->has_shortcut) {
		snprintf(name, sizeof(name), "%s.shortcut", prefix);
		s = load_conv(v, name, &rb->shortcut);
	}
	return s;
}

static void free_res_block(diffusion_vae *v, res_block *rb) {
	nn_free(v->be, &rb->gamma1);
	nn_free(v->be, &rb->gamma2);
	nn_conv_free(v->be, &rb->conv1);
	nn_conv_free(v->be, &rb->conv2);
	nn_conv_free(v->be, &rb->shortcut);
}

static status_code load_stage(diffusion_vae *v, int idx, up_stage *st) {
	char		prefix[128];
	status_code s = OK;
	for (int b = 0; b < VAE_BLOCKS_PER_STAGE && s == OK; b++) {
		snprintf(prefix, sizeof(prefix), "decoder.upsamples.%d.upsamples.%d", idx, b);
		s = load_res_block(v, prefix, &st->blocks[b]);
	}
	st->in	= st->blocks[0].in;
	st->out = st->blocks[VAE_BLOCKS_PER_STAGE - 1].out;
	snprintf(prefix, sizeof(prefix), "decoder.upsamples.%d.upsamples.%d.resample.1.weight", idx,
			 VAE_BLOCKS_PER_STAGE);
	st->up = gguf_find_tensor(&v->g, prefix) != NULL;
	if (s == OK && st->up) {
		snprintf(prefix, sizeof(prefix), "decoder.upsamples.%d.upsamples.%d.resample.1", idx,
				 VAE_BLOCKS_PER_STAGE);
		s = load_conv(v, prefix, &st->resample);
		snprintf(prefix, sizeof(prefix), "decoder.upsamples.%d.upsamples.%d.time_conv.weight", idx,
				 VAE_BLOCKS_PER_STAGE);
		st->frames_up = gguf_find_tensor(&v->g, prefix) ? 2 : 1;
	}
	return s;
}

static status_code load_attention(diffusion_vae *v) {
	status_code		   s = load_conv(v, "decoder.middle.1.proj", &v->attn_proj);
	const gguf_tensor *w = vae_tensor(v, "decoder.middle.1.to_qkv", "weight");
	const gguf_tensor *b = vae_tensor(v, "decoder.middle.1.to_qkv", "bias");
	if (s == OK && !(w && b))
		s = ERR_NOT_FOUND;
	nn_conv *parts[] = {&v->attn_q, &v->attn_k, &v->attn_v};
	for (int i = 0; i < 3 && s == OK; i++)
		s = nn_conv_load(v->be, w, b, i * v->attn_proj.oc, v->attn_proj.oc, parts[i]);
	return s;
}

static status_code load_all(diffusion_vae *v) {
	status_code s = load_conv(v, "conv2", &v->post_quant);
	if (s == OK)
		s = load_conv(v, "decoder.conv1", &v->conv_in);
	if (s == OK)
		s = load_res_block(v, "decoder.middle.0", &v->mid0);
	if (s == OK)
		s = load_res_block(v, "decoder.middle.2", &v->mid2);
	if (s == OK)
		s = load_attention(v);
	if (s == OK)
		s = load_vec(v, "decoder.middle.1.norm", "gamma", &v->attn_gamma);
	for (int i = 0; i < VAE_STAGES && s == OK; i++)
		s = load_stage(v, i, &v->stages[i]);
	if (s == OK)
		s = load_vec(v, "decoder.head.0", "gamma", &v->head_gamma);
	if (s == OK)
		s = load_conv(v, "decoder.head.2", &v->head);
	return s;
}

diffusion_vae *diffusion_vae_open(backend *be, const char *path, const vae_latent_stats *stats) {
	diffusion_vae *v = xcalloc(1, sizeof(*v));
	v->be			 = be;
	v->stats		 = stats;
	if (gguf_load(&v->g, path) != OK || load_all(v) != OK) {
		ERROR("failed to load vae '%s'", path);
		diffusion_vae_close(v);
		return NULL;
	}
	return v;
}

void diffusion_vae_close(diffusion_vae *v) {
	if (!v)
		return;
	nn_conv_free(v->be, &v->post_quant);
	nn_conv_free(v->be, &v->conv_in);
	nn_conv_free(v->be, &v->head);
	free_res_block(v, &v->mid0);
	free_res_block(v, &v->mid2);
	nn_conv_free(v->be, &v->attn_q);
	nn_conv_free(v->be, &v->attn_k);
	nn_conv_free(v->be, &v->attn_v);
	nn_conv_free(v->be, &v->attn_proj);
	nn_free(v->be, &v->attn_gamma);
	nn_free(v->be, &v->head_gamma);
	for (int i = 0; i < VAE_STAGES; i++) {
		for (int b = 0; b < VAE_BLOCKS_PER_STAGE; b++)
			free_res_block(v, &v->stages[i].blocks[b]);
		nn_conv_free(v->be, &v->stages[i].resample);
	}
	gguf_free(&v->g);
	free(v);
}

#define VAE_INPUT 0
#define VAE_OUTPUT 0

static int norm_silu(cg_builder *g, int x, const buffer *gamma, int channels, int pixels) {
	int t = cg_rmsnorm(g, cg_param_vec(g, gamma), x, CG_NEW, channels, VAE_RMS_EPS, pixels);
	return cg_activate(g, t, NN_SILU);
}

static int conv(cg_builder *g, const nn_conv *c, int x, int H, int W, int shift) {
	return cg_conv(g, cg_param_conv(g, c), x, CG_NEW, H, W, shift);
}

static int res_block_graph(cg_builder *g, const res_block *rb, int x, int H, int W) {
	int pixels = H * W;
	int t	   = norm_silu(g, x, &rb->gamma1, rb->in, pixels);
	int u	   = conv(g, &rb->conv1, t, H, W, 0);
	t		   = norm_silu(g, u, &rb->gamma2, rb->out, pixels);
	int y	   = conv(g, &rb->conv2, t, H, W, 0);
	int skip   = rb->has_shortcut ? conv(g, &rb->shortcut, x, H, W, 0) : x;
	return cg_add(g, y, skip, rb->out, pixels);
}

static int attention_graph(cg_builder *g, const diffusion_vae *v, int x, int H, int W) {
	int pixels	 = H * W;
	int channels = v->attn_proj.oc;
	int xn =
		cg_rmsnorm(g, cg_param_vec(g, &v->attn_gamma), x, CG_NEW, channels, VAE_RMS_EPS, pixels);
	int q	= conv(g, &v->attn_q, xn, H, W, 0);
	int k	= conv(g, &v->attn_k, xn, H, W, 0);
	int val = conv(g, &v->attn_v, xn, H, W, 0);
	int att = cg_attention(
		g, q, k, val, CG_NONE, CG_NONE, CG_NEW,
		(cg_attention_dims){.n_q = pixels, .n_heads = 1, .head_dim = channels, .n0 = pixels});
	int proj = conv(g, &v->attn_proj, att, H, W, 0);
	return cg_add(g, x, proj, channels, pixels);
}

static int up_stage_graph(cg_builder *g, const up_stage *st, int x, int *H, int *W) {
	int cur = x;
	for (int b = 0; b < VAE_BLOCKS_PER_STAGE; b++)
		cur = res_block_graph(g, &st->blocks[b], cur, *H, *W);
	if (!st->up)
		return cur;
	int oH = *H * VAE_UPSAMPLE, oW = *W * VAE_UPSAMPLE;
	int up	   = conv(g, &st->resample, cur, oH, oW, 1);
	int factor = st->frames_up * VAE_UPSAMPLE * VAE_UPSAMPLE;
	up		   = cg_upsample_shortcut_add(g, x, up,
										  (cg_upsample_dims){.height	   = *H,
															 .width		   = *W,
															 .in_channels  = st->in,
															 .out_channels = st->out,
															 .repeats	   = st->out * factor / st->in,
															 .frames_up	   = st->frames_up,
															 .factor	   = VAE_UPSAMPLE});
	*H		   = oH;
	*W		   = oW;
	return up;
}

static cg_builder *vae_graph(const diffusion_vae *v, int H, int W) {
	cg_builder *g = cg_builder_new();
	int			z = cg_input(g, (size_t)H * W * v->stats->channels, VAE_INPUT);
	int			q = conv(g, &v->post_quant, z, H, W, 0);
	int			x = conv(g, &v->conv_in, q, H, W, 0);
	x			  = res_block_graph(g, &v->mid0, x, H, W);
	x			  = attention_graph(g, v, x, H, W);
	x			  = res_block_graph(g, &v->mid2, x, H, W);
	for (int i = 0; i < VAE_STAGES; i++)
		x = up_stage_graph(g, &v->stages[i], x, &H, &W);
	x = norm_silu(g, x, &v->head_gamma, v->head.ic, H * W);
	cg_output_value(g, conv(g, &v->head, x, H, W, 0), VAE_OUTPUT);
	return g;
}

static float *denormalize_latent(const vae_latent_stats *st, const float *latent, int pixels) {
	float *z = xmalloc((size_t)pixels * st->channels * sizeof(float));
	for (int p = 0; p < pixels; p++)
		for (int c = 0; c < st->channels; c++)
			z[(size_t)p * st->channels + c] =
				latent[(size_t)p * st->channels + c] * st->std[c] + st->mean[c];
	return z;
}

static int output_scale(const diffusion_vae *v) {
	int scale = 1;
	for (int i = 0; i < VAE_STAGES; i++)
		scale *= v->stages[i].up ? VAE_UPSAMPLE : 1;
	return scale;
}

status_code diffusion_vae_decode(diffusion_vae *v, cg_arena *arena, const float *latent,
								 latent_shape shape, diffusion_image *out) {
	cg_plan	   *plan;
	status_code s = cg_plan_build(vae_graph(v, shape.lh, shape.lw), v->be, "vae", &plan);
	if (s != OK)
		return s;
	int			 W	  = shape.lw * output_scale(v);
	int			 H	  = shape.lh * output_scale(v);
	size_t		 n	  = (size_t)H * W * v->head.oc;
	float		*z	  = denormalize_latent(v->stats, latent, shape.lh * shape.lw);
	float		*host = xmalloc(n * sizeof(float));
	cg_exec_args args = {.host_in = {[VAE_INPUT] = z}, .host_out = {[VAE_OUTPUT] = host}};
	s				  = cg_graph_run(v->be, arena, plan, &args);
	cg_plan_free(plan);
	if (s == OK) {
		out->width	  = W;
		out->height	  = H;
		out->channels = v->head.oc;
		out->pixels	  = xmalloc(n);
		for (size_t i = 0; i < n; i++)
			out->pixels[i] = image_unit_to_byte((host[i] + 1.0f) * 0.5f);
	}
	free(z);
	free(host);
	return s;
}
