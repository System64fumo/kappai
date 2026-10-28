#include "compute.h"
#include "config.h"
#include "context.h"
#include "diffusion/diffusion.h"
#include "log.h"
#include "nn.h"

#include <math.h>
#include <stdio.h>

static const float qwen_image_preview_rgb_proj[192] = {
	0.00860495522f,	 0.01219501462f,  -0.00321337196f,	0.01889233090f,	  0.01246581216f,
	0.01074959482f,	 0.1255941446f,	  0.1176879344f,	-0.0332212352f,	  0.0418238528f,
	0.1043427754f,	 0.0121666316f,	  0.02025338f,		0.01453670296f,	  -0.000224336044f,
	-0.01896720702f, -0.0206099030f,  -0.0322728584f,	0.00438984796f,	  -0.01374969766f,
	0.02849196f,	 -0.0374495856f,  -0.0286777126f,	-0.0693192810f,	  0.01511914734f,
	0.0242979386f,	 0.0553878870f,	  -0.1138629518f,	-0.020391466f,	  0.001550520522f,
	-0.0233650696f,	 -0.0417292018f,  -0.0362361182f,	-0.0351603342f,	  -0.0243595924f,
	-0.00216261038f, 0.01093355288f,  -0.0373466924f,	0.00241315350f,	  0.01778704744f,
	-0.00401984678f, -0.0343259192f,  0.0486059334f,	0.0253144f,		  0.0672564966f,
	0.0309463558f,	 0.0277963166f,	  0.0520869622f,	0.0374485008f,	  0.0551753676f,
	0.0225853902f,	 -0.0090809962f,  -0.004756176f,	0.00636443612f,	  -0.0270455652f,
	-0.0384966954f,	 -0.00905908082f, -0.00553493756f,	0.01484553684f,	  -0.0211502468f,
	0.01319502562f,	 0.00948005666f,  0.0483789212f,	-0.00931847104f,  -0.00276452734f,
	-0.01011985302f, 0.0180478258f,	  0.01614954356f,	-0.0209424690f,	  -0.0214530434f,
	-0.00272961176f, 0.0217887476f,	  -0.0636772304f,	-0.0208893548f,	  0.0479167742f,
	-0.0250321236f,	 -0.0286715676f,  0.0530110146f,	-0.01853078078f,  0.01647272818f,
	-0.00207747588f, 0.0023101082f,	  0.01228800748f,	0.01303505006f,	  -0.01243671408f,
	-0.0258638728f,	 -0.0379116264f,  0.00598934710f,	0.00642563550f,	  -0.01234514304f,
	-0.0296733996f,	 -0.0234698050f,  0.00060018212f,	-0.0322019498f,	  -0.0529200462f,
	-0.00344987414f, -0.00205026458f, -0.00846599446f,	0.00455971038f,	  -0.01082227064f,
	0.0315661948f,	 -0.0677753362f,  0.0645553474f,	0.1109666998f,	  0.0674744864f,
	0.01036801108f,	 -0.00484841210f, -0.001529168474f, 0.01264353566f,	  0.01548126338f,
	-0.00966374324f, -0.0223892408f,  -0.00871751526f,	-0.000306421670f, 0.0271322742f,
	0.03496524f,	 -0.0089692858f,  0.0512178672f,	0.0173080034f,	  0.00804227746f,
	0.01210987192f,	 0.00758025926f,  -0.00281712586f,	0.1897278390f,	  0.1210261828f,
	0.062603892f,	 0.0208058822f,	  0.00547548182f,	0.01262955638f,	  0.00813332858f,
	0.01015930914f,	 0.01301771290f,  -0.000927236014f, -0.00152540594f,  -0.00599213302f,
	0.01663314616f,	 -0.00582789626f, 0.0163958132f,	-0.0252546342f,	  -0.0604193732f,
	-0.1606919922f,	 -0.091722686f,	  -0.0409201224f,	-0.0959576198f,	  0.0282963112f,
	-0.01387223872f, -0.01648814464f, 0.0552316818f,	0.0967547788f,	  0.0413586632f,
	0.00922849292f,	 0.00451467542f,  -0.0529172378f,	0.0558600768f,	  0.0122988308f,
	-0.01445942422f, 0.000210660902f, -0.01295958782f,	-0.01804761764f,  0.0358136250f,
	-0.0472505970f,	 -0.1156405142f,  -0.0506390696f,	-0.0471914842f,	  0.0349791468f,
	-0.0480143168f,	 0.00628389868f,  -0.0545163826f,	0.0315499582f,	  0.0564846606f,
	-0.0430850488f,	 -0.0362330316f,  -0.01267788554f,	0.0061024772f,	  0.0038627542f,
	0.00911055916f,	 -0.00758526008f, -0.0447103298f,	-0.00835411408f,  0.01545872328f,
	-0.015006738f,	 0.00270612302f,  -0.00784361356f,	-0.0221755048f,	  -0.0513344748f,
	-0.0475317424f,	 -0.01036656294f, -0.00422146068f,	-0.0213499052f,	  0.01788952706f,
	0.01191944190f,	 0.0397205238f,
};

static const float qwen_image_preview_rgb_bias[3] = {-0.043293118f, -0.02695978f, -0.11986706f};

static const float qwen_image_preview_alpha_proj[64] = {
	-0.0416241114f,	 -0.00678954612f, -0.0169095515f,  -0.0230551401f, 0.0100882595f,
	0.00655586802f,	 0.0401166874f,	  -0.0055510216f,  0.0224234441f,  -0.0389640963f,
	-0.0114492163f,	 -0.00721128977f, -0.0029064082f,  0.0150300547f,  -0.00321615308f,
	-0.0498856338f,	 -0.0215251401f,  0.0240220482f,   0.0117338008f,  -0.0460420624f,
	0.0387872889f,	 0.0131517207f,	  0.0147100836f,   0.0266985286f,  0.0153097324f,
	-0.0418119757f,	 0.0421013917f,	  0.0401724499f,   0.00972515915f, 0.011718495f,
	0.0117622291f,	 0.0136505134f,	  -0.0350017363f,  -0.0100692606f, -0.0131582529f,
	-0.00660639315f, 0.00253611396f,  -0.0195736368f,  -0.04240184f,   0.0321299262f,
	0.0106089414f,	 -0.0179845306f,  -0.00806212708f, 0.0135889057f,  -0.0157393098f,
	-0.0267791344f,	 0.0109068534f,	  0.0283931966f,   -0.0435370078f, 0.00187883536f,
	-0.0108995378f,	 -0.0450757676f,  -0.0699481501f,  0.0123562106f,  -0.0222592249f,
	0.0216155907f,	 0.0563755424f,	  -0.0073379912f,  0.0160012921f,  0.0411637742f,
	0.0189607258f,	 -0.024025029f,	  -0.0161487905f,  -0.016913203f,
};

static const float qwen_image_preview_alpha_bias = 0.871268134f;

static const float qwen_image_latent_mean[64] = {
	0.5126f,  0.7721f,	-0.0631f, 1.3506f,	-0.7855f, -2.1025f, -0.3458f, 1.3722f,
	1.8873f,  -1.7177f, -0.6510f, 0.2732f,	0.7562f,  -0.6163f, -1.0277f, 3.8363f,
	2.0210f,  0.0472f,	0.9320f,  2.0087f,	2.4954f,  -0.1391f, -1.4249f, 1.8464f,
	-0.5236f, 1.2826f,	3.7046f,  -1.3035f, 2.7286f,  -1.4518f, -1.9036f, -1.9955f,
	-0.0342f, -1.0265f, -0.7636f, 3.0555f,	0.0746f,  -3.0751f, -0.1076f, 1.7376f,
	-1.0914f, -1.9435f, -0.2784f, -1.3680f, 0.4809f,  -0.4433f, 0.3764f,  0.5729f,
	-2.0595f, 1.0960f,	-1.3260f, -2.0211f, -5.0179f, 0.5275f,	4.0162f,  1.8505f,
	0.3026f,  1.9373f,	1.4937f,  0.2632f,	0.5547f,  -1.7121f, -0.1562f, 0.0304f,
};

static const float qwen_image_latent_std[64] = {
	3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.7910f, 3.0785f, 3.6500f,
	3.9308f, 3.0904f, 2.8778f, 3.7675f, 3.7320f, 5.0756f, 3.2864f, 4.0397f, 3.1317f, 4.0443f,
	2.9249f, 3.9454f, 3.0988f, 4.2489f, 3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.2830f,
	3.5694f, 4.2467f, 3.9037f, 3.2947f, 5.0770f, 3.5075f, 3.2700f, 3.4767f, 2.8063f, 5.1125f,
	3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f, 3.9624f, 4.0168f,
	3.5643f, 4.0550f, 5.5614f, 4.2963f, 4.4080f, 3.4959f, 3.8747f, 3.7608f, 3.5735f, 3.1490f,
	3.7662f, 3.6746f, 3.4563f, 3.8161f,
};

#define VAE_STAGES 5
#define VAE_BLOCKS_PER_STAGE 3
#define VAE_RMS_EPS 1e-12f
#define VAE_LATENT_CHANNELS 64
#define VAE_OUT_CHANNELS 4
#define VAE_UPSAMPLE 2
#define TIME_EMBED_DIM 256
#define TIME_MAX_PERIOD 10000.0f
#define TIMESTEP_SCALE 1000.0f
#define NORM_EPS 1e-6f
#define MOD_CHUNKS 4
#define ROPE_THETA 10000.0f
#define ROPE_AXES 3
#define TEXT_CTX_TOKENS 1024
#define TEXT_SYSTEM "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
#define TEXT_USER_PREFIX "<|im_start|>user\n"
#define TEXT_USER_SUFFIX "<|im_end|>\n<|im_start|>assistant\n"

typedef struct {
	int		in, out;
	float  *gamma1, *gamma2;
	nn_conv conv1, conv2, shortcut;
	bool	has_shortcut;
} res_block;

typedef struct {
	int		  in, out, frames_up;
	res_block blocks[VAE_BLOCKS_PER_STAGE];
	bool	  up;
	nn_conv	  resample;
} up_stage;

typedef struct qwen_vae qwen_vae;

struct qwen_vae {
	backend	 *be;
	gguf_ctx  g;
	nn_conv	  post_quant, conv_in, head;
	res_block mid0, mid2;
	float	 *attn_gamma;
	nn_conv	  attn_qkv, attn_proj;
	up_stage  stages[VAE_STAGES];
	float	 *head_gamma;
};

static const gguf_tensor *vae_tensor(qwen_vae *v, const char *fmt, const char *prefix,
									 const char *suffix) {
	char name[192];
	snprintf(name, sizeof(name), fmt, prefix, suffix);
	const gguf_tensor *t = gguf_find_tensor(&v->g, name);
	if (!t)
		ERROR("vae tensor '%s' not found", name);
	return t;
}

static float *vae_vec(qwen_vae *v, const char *prefix, const char *suffix) {
	const gguf_tensor *t = vae_tensor(v, "%s.%s", prefix, suffix);
	return t ? tensor_to_f32(v->be, t) : NULL;
}

static status_code load_conv(qwen_vae *v, const char *prefix, nn_conv *c) {
	const gguf_tensor *w = vae_tensor(v, "%s.%s", prefix, "weight");
	const gguf_tensor *b = vae_tensor(v, "%s.%s", prefix, "bias");
	return w && b ? nn_conv_load(v->be, w, b, c) : ERR_NOT_FOUND;
}

static status_code load_res_block(qwen_vae *v, const char *prefix, res_block *rb) {
	char name[160];
	rb->gamma1 = vae_vec(v, prefix, "residual.0.gamma");
	rb->gamma2 = vae_vec(v, prefix, "residual.3.gamma");
	snprintf(name, sizeof(name), "%s.residual.2", prefix);
	status_code s = load_conv(v, name, &rb->conv1);
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
	return rb->gamma1 && rb->gamma2 ? s : ERR_NOT_FOUND;
}

static void free_res_block(qwen_vae *v, res_block *rb) {
	free(rb->gamma1);
	free(rb->gamma2);
	nn_conv_free(v->be, &rb->conv1);
	nn_conv_free(v->be, &rb->conv2);
	if (rb->has_shortcut)
		nn_conv_free(v->be, &rb->shortcut);
}

static status_code load_stage(qwen_vae *v, int idx, up_stage *st) {
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

static status_code load_all(qwen_vae *v) {
	status_code s = load_conv(v, "conv2", &v->post_quant);
	if (s == OK)
		s = load_conv(v, "decoder.conv1", &v->conv_in);
	if (s == OK)
		s = load_res_block(v, "decoder.middle.0", &v->mid0);
	if (s == OK)
		s = load_res_block(v, "decoder.middle.2", &v->mid2);
	if (s == OK)
		s = load_conv(v, "decoder.middle.1.to_qkv", &v->attn_qkv);
	if (s == OK)
		s = load_conv(v, "decoder.middle.1.proj", &v->attn_proj);
	v->attn_gamma = vae_vec(v, "decoder.middle.1.norm", "gamma");
	for (int i = 0; i < VAE_STAGES && s == OK; i++)
		s = load_stage(v, i, &v->stages[i]);
	v->head_gamma = vae_vec(v, "decoder.head.0", "gamma");
	if (s == OK)
		s = load_conv(v, "decoder.head.2", &v->head);
	return v->attn_gamma && v->head_gamma ? s : ERR_NOT_FOUND;
}

static void qwen_vae_close(qwen_vae *v);

static qwen_vae *qwen_vae_open(backend *be, const char *path) {
	qwen_vae *v = xcalloc(1, sizeof(*v));
	v->be		= be;
	if (gguf_load(&v->g, path) != OK || load_all(v) != OK) {
		ERROR("failed to load vae '%s'", path);
		qwen_vae_close(v);
		return NULL;
	}
	return v;
}

static void qwen_vae_close(qwen_vae *v) {
	if (!v)
		return;
	nn_conv_free(v->be, &v->post_quant);
	nn_conv_free(v->be, &v->conv_in);
	nn_conv_free(v->be, &v->head);
	free_res_block(v, &v->mid0);
	free_res_block(v, &v->mid2);
	nn_conv_free(v->be, &v->attn_qkv);
	nn_conv_free(v->be, &v->attn_proj);
	free(v->attn_gamma);
	free(v->head_gamma);
	for (int i = 0; i < VAE_STAGES; i++) {
		for (int b = 0; b < VAE_BLOCKS_PER_STAGE; b++)
			free_res_block(v, &v->stages[i].blocks[b]);
		if (v->stages[i].up)
			nn_conv_free(v->be, &v->stages[i].resample);
	}
	gguf_free(&v->g);
	free(v);
}

static float *norm_silu(backend *be, const float *x, const float *gamma, int ch, int pixels) {
	float *t = xmalloc((size_t)pixels * ch * sizeof(float));
	nn_rmsnorm(be, x, gamma, t, ch, VAE_RMS_EPS, pixels);
	nn_activate(be, NN_SILU, t, (size_t)pixels * ch);
	return t;
}

static float *res_block_forward(qwen_vae *v, const res_block *rb, const float *x, int H, int W) {
	backend *be = v->be;
	int		 P	= H * W;

	float *t = norm_silu(be, x, rb->gamma1, rb->in, P);
	float *u = xmalloc((size_t)P * rb->out * sizeof(float));
	nn_conv_forward(be, &rb->conv1, t, u, H, W, 0);
	free(t);

	t		 = norm_silu(be, u, rb->gamma2, rb->out, P);
	float *y = xmalloc((size_t)P * rb->out * sizeof(float));
	nn_conv_forward(be, &rb->conv2, t, y, H, W, 0);
	free(t);
	free(u);

	if (rb->has_shortcut) {
		float *sc = xmalloc((size_t)P * rb->out * sizeof(float));
		nn_conv_forward(be, &rb->shortcut, x, sc, H, W, 0);
		nn_add(be, y, sc, rb->out, P);
		free(sc);
	} else
		nn_add(be, y, x, rb->out, P);
	return y;
}

typedef struct {
	const float *qkv;
	float		*q, *k, *v;
	int			 ch;
} qkv_split_ctx;

static void split_qkv_rows(int begin, int end, int tid, void *p) {
	(void)tid;
	const qkv_split_ctx *c	 = p;
	size_t				 row = (size_t)c->ch * sizeof(float);
	for (int r = begin; r < end; r++) {
		const float *src = c->qkv + (size_t)r * 3 * c->ch;
		memcpy(c->q + (size_t)r * c->ch, src, row);
		memcpy(c->k + (size_t)r * c->ch, src + c->ch, row);
		memcpy(c->v + (size_t)r * c->ch, src + 2 * c->ch, row);
	}
}

static void attention_block(qwen_vae *v, float *x, int H, int W) {
	backend *be = v->be;
	int		 P	= H * W;
	int		 ch = v->attn_proj.oc;

	float *xn  = xmalloc((size_t)P * ch * sizeof(float));
	float *qkv = xmalloc((size_t)P * 3 * ch * sizeof(float));
	nn_rmsnorm(be, x, v->attn_gamma, xn, ch, VAE_RMS_EPS, P);
	nn_conv_forward(be, &v->attn_qkv, xn, qkv, H, W, 0);

	float *q = xn, *k = xmalloc((size_t)P * ch * sizeof(float)),
		  *val			= xmalloc((size_t)P * ch * sizeof(float));
	qkv_split_ctx split = {qkv, q, k, val, ch};
	tpool_parallel_for(nn_pool(be), P, 16, split_qkv_rows, &split);
	free(qkv);

	float *att = xmalloc((size_t)P * ch * sizeof(float));
	nn_kv  kv  = {.k0 = k, .v0 = val, .n0 = P};
	nn_attention(be, att, q, P, 1, ch, &kv);
	free(xn);
	free(k);
	free(val);

	float *proj = xmalloc((size_t)P * ch * sizeof(float));
	nn_conv_forward(be, &v->attn_proj, att, proj, H, W, 0);
	nn_add(be, x, proj, ch, P);
	free(att);
	free(proj);
}

typedef struct {
	const float *in;
	float		*out;
	int			 H, W, in_ch, out_ch, repeats, frames_up;
} dup_up_ctx;

static void dup_up_rows(int begin, int end, int tid, void *p) {
	(void)tid;
	const dup_up_ctx *c	 = p;
	int				  fs = VAE_UPSAMPLE;
	int				  oW = c->W * fs;
	for (int r = begin; r < end; r++) {
		int			 Y = r / oW, X = r % oW;
		int			 y = Y / fs, b = Y % fs, x = X / fs, e = X % fs;
		const float *src = c->in + ((size_t)y * c->W + x) * c->in_ch;
		float		*dst = c->out + (size_t)r * c->out_ch;
		for (int o = 0; o < c->out_ch; o++) {
			int group = ((o * c->frames_up + (c->frames_up - 1)) * fs + b) * fs + e;
			dst[o] += src[group / c->repeats];
		}
	}
}

static float *up_stage_forward(qwen_vae *v, const up_stage *st, float *x, int *H, int *W) {
	backend *be	 = v->be;
	float	*cur = x;
	for (int b = 0; b < VAE_BLOCKS_PER_STAGE; b++) {
		float *next = res_block_forward(v, &st->blocks[b], cur, *H, *W);
		if (cur != x)
			free(cur);
		cur = next;
	}
	if (!st->up) {
		free(x);
		return cur;
	}

	int	   oH = *H * VAE_UPSAMPLE, oW = *W * VAE_UPSAMPLE;
	float *up = xmalloc((size_t)oH * oW * st->out * sizeof(float));
	nn_conv_forward(be, &st->resample, cur, up, oH, oW, 1);
	free(cur);

	int		   factor = st->frames_up * VAE_UPSAMPLE * VAE_UPSAMPLE;
	dup_up_ctx dup	  = {x, up, *H, *W, st->in, st->out, st->out * factor / st->in, st->frames_up};
	tpool_parallel_for(nn_pool(be), oH * oW, 64, dup_up_rows, &dup);
	free(x);
	*H = oH;
	*W = oW;
	return up;
}

static float *denormalize_latent(const float *latent, int pixels) {
	float *z = xmalloc((size_t)pixels * VAE_LATENT_CHANNELS * sizeof(float));
	for (int p = 0; p < pixels; p++)
		for (int c = 0; c < VAE_LATENT_CHANNELS; c++)
			z[(size_t)p * VAE_LATENT_CHANNELS + c] =
				latent[(size_t)p * VAE_LATENT_CHANNELS + c] * qwen_image_latent_std[c] +
				qwen_image_latent_mean[c];
	return z;
}

static uint8_t to_byte(float v) {
	float s = (v + 1.0f) * 0.5f;
	return (uint8_t)(MAX(0.0f, MIN(1.0f, s)) * 255.0f + 0.5f);
}

static status_code qwen_vae_decode(qwen_vae *v, const float *latent, latent_shape shape,
								   diffusion_image *out) {
	backend *be = v->be;
	int		 H = shape.lh, W = shape.lw;

	float *z = denormalize_latent(latent, H * W);
	float *q = xmalloc((size_t)H * W * v->post_quant.oc * sizeof(float));
	nn_conv_forward(be, &v->post_quant, z, q, H, W, 0);
	free(z);

	float *x = xmalloc((size_t)H * W * v->conv_in.oc * sizeof(float));
	nn_conv_forward(be, &v->conv_in, q, x, H, W, 0);
	free(q);

	float *t = res_block_forward(v, &v->mid0, x, H, W);
	free(x);
	attention_block(v, t, H, W);
	x = res_block_forward(v, &v->mid2, t, H, W);
	free(t);

	for (int i = 0; i < VAE_STAGES; i++) {
		x = up_stage_forward(v, &v->stages[i], x, &H, &W);
		DEBUG("vae stage %d done (%dx%d)", i, W, H);
	}

	t = norm_silu(be, x, v->head_gamma, v->head.ic, H * W);
	free(x);
	float *rgba = xmalloc((size_t)H * W * VAE_OUT_CHANNELS * sizeof(float));
	nn_conv_forward(be, &v->head, t, rgba, H, W, 0);
	free(t);

	out->width	  = W;
	out->height	  = H;
	out->channels = VAE_OUT_CHANNELS;
	out->pixels	  = xmalloc((size_t)H * W * VAE_OUT_CHANNELS);
	for (size_t i = 0; i < (size_t)H * W * VAE_OUT_CHANNELS; i++)
		out->pixels[i] = to_byte(rgba[i]);
	free(rgba);
	return OK;
}

typedef struct {
	weight q, k, v, o, gate_up, down;
	float *norm_q, *norm_k;
} dit_layer;

typedef struct {
	diffusion_model base;
	backend		   *be;
	gguf_ctx		dit;
	char		   *llm_path;
	qwen_vae	   *vae;
	int				hidden, heads, head_dim, inter, n_layers, in_channels, out_channels;
	int				rope_axes[ROPE_AXES];
	weight			img_in, modulation, norm_out, proj_out, time_1, time_2, txt_in, txt_out;
	float		   *txt_norm;
	dit_layer	   *layers;
} qwen_image;

struct diffusion_cond {
	int	   n_tokens;
	float *k, *v;
};

typedef struct {
	float *q, *k, *v, *a, *h, *gu, *norm;
} layer_scratch;

static const gguf_tensor *find(const qwen_image *m, const char *fmt, int idx, const char *suffix) {
	char name[160];
	snprintf(name, sizeof(name), fmt, idx, suffix);
	const gguf_tensor *t = gguf_find_tensor(&m->dit, name);
	if (!t)
		ERROR("dit tensor '%s' not found", name);
	return t;
}

static status_code load_named(const qwen_image *m, const char *name, weight *out) {
	const gguf_tensor *t = gguf_find_tensor(&m->dit, name);
	if (!t) {
		ERROR("dit tensor '%s' not found", name);
		return ERR_NOT_FOUND;
	}
	return weight_load(m->be, t, out);
}

static status_code load_layer(const qwen_image *m, int li, dit_layer *l) {
	static const struct {
		const char *name;
		size_t		off;
	} mats[] = {
		{"attn.to_q.weight", offsetof(dit_layer, q)},
		{"attn.to_k.weight", offsetof(dit_layer, k)},
		{"attn.to_v.weight", offsetof(dit_layer, v)},
		{"attn.to_out.0.weight", offsetof(dit_layer, o)},
		{"img_mlp.gate_up.weight", offsetof(dit_layer, gate_up)},
		{"img_mlp.out.weight", offsetof(dit_layer, down)},
	};
	for (size_t i = 0; i < ARRAY_LEN(mats); i++) {
		const gguf_tensor *t = find(m, "transformer_blocks.%d.%s", li, mats[i].name);
		if (!t)
			return ERR_NOT_FOUND;
		status_code s = weight_load(m->be, t, (weight *)((char *)l + mats[i].off));
		if (s != OK)
			return s;
	}
	const gguf_tensor *nq = find(m, "transformer_blocks.%d.%s", li, "attn.norm_q.weight");
	const gguf_tensor *nk = find(m, "transformer_blocks.%d.%s", li, "attn.norm_k.weight");
	if (!nq || !nk)
		return ERR_NOT_FOUND;
	l->norm_q = tensor_to_f32(m->be, nq);
	l->norm_k = tensor_to_f32(m->be, nk);
	return OK;
}

static bool qwen_image_detect(const gguf_ctx *dit) {
	return gguf_find_tensor(dit, "transformer_blocks.0.img_mlp.gate_up.weight") &&
		   gguf_find_tensor(dit, "txt_in.in_layer.weight");
}

static status_code load_dit(qwen_image *m) {
	static const struct {
		const char *name;
		size_t		off;
	} globals[] = {
		{"img_in.weight", offsetof(qwen_image, img_in)},
		{"modulation.1.weight", offsetof(qwen_image, modulation)},
		{"norm_out.linear.weight", offsetof(qwen_image, norm_out)},
		{"proj_out.weight", offsetof(qwen_image, proj_out)},
		{"time_text_embed.timestep_embedder.linear_1.weight", offsetof(qwen_image, time_1)},
		{"time_text_embed.timestep_embedder.linear_2.weight", offsetof(qwen_image, time_2)},
		{"txt_in.in_layer.weight", offsetof(qwen_image, txt_in)},
		{"txt_in.out_layer.weight", offsetof(qwen_image, txt_out)},
	};
	for (size_t i = 0; i < ARRAY_LEN(globals); i++) {
		status_code s = load_named(m, globals[i].name, (weight *)((char *)m + globals[i].off));
		if (s != OK)
			return s;
	}

	const gguf_tensor *norm = gguf_find_tensor(&m->dit, "txt_in.text_norm.weight");
	if (!norm)
		return ERR_NOT_FOUND;
	m->txt_norm = tensor_to_f32(m->be, norm);
	for (int i = 0; i < m->hidden; i++)
		m->txt_norm[i] += 1.0f;

	m->layers = xcalloc((size_t)m->n_layers, sizeof(dit_layer));
	for (int li = 0; li < m->n_layers; li++) {
		status_code s = load_layer(m, li, &m->layers[li]);
		if (s != OK)
			return s;
	}
	return OK;
}

static void detect_config(qwen_image *m) {
	const gguf_tensor *img_in = gguf_find_tensor(&m->dit, "img_in.weight");
	const gguf_tensor *proj	  = gguf_find_tensor(&m->dit, "proj_out.weight");
	const gguf_tensor *norm_q =
		gguf_find_tensor(&m->dit, "transformer_blocks.0.attn.norm_q.weight");
	const gguf_tensor *gu =
		gguf_find_tensor(&m->dit, "transformer_blocks.0.img_mlp.gate_up.weight");
	m->in_channels	= (int)img_in->dims[0];
	m->hidden		= (int)img_in->dims[1];
	m->out_channels = (int)proj->dims[1];
	m->head_dim		= (int)norm_q->dims[0];
	m->heads		= m->hidden / m->head_dim;
	m->inter		= (int)gu->dims[1] / 2;

	char name[96];
	for (m->n_layers = 0;; m->n_layers++) {
		snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_q.weight", m->n_layers);
		if (!gguf_find_tensor(&m->dit, name))
			break;
	}
	m->rope_axes[0] = 16;
	m->rope_axes[1] = (m->head_dim - 16) / 2;
	m->rope_axes[2] = (m->head_dim - 16) / 2;
}

static void qwen_image_close(diffusion_model *dm) {
	qwen_image *m		  = (qwen_image *)dm;
	weight	   *singles[] = {&m->img_in, &m->modulation, &m->norm_out, &m->proj_out,
							 &m->time_1, &m->time_2,	 &m->txt_in,   &m->txt_out};
	for (size_t i = 0; i < ARRAY_LEN(singles); i++)
		if (singles[i]->buf.owner)
			weight_free(m->be, singles[i]);
	free(m->txt_norm);
	for (int li = 0; m->layers && li < m->n_layers; li++) {
		dit_layer *l	  = &m->layers[li];
		weight	  *mats[] = {&l->q, &l->k, &l->v, &l->o, &l->gate_up, &l->down};
		for (size_t i = 0; i < ARRAY_LEN(mats); i++)
			if (mats[i]->buf.owner)
				weight_free(m->be, mats[i]);
		free(l->norm_q);
		free(l->norm_k);
	}
	free(m->layers);
	qwen_vae_close(m->vae);
	gguf_free(&m->dit);
	free(m->llm_path);
	if (m->be)
		backend_destroy(m->be);
	free(m);
}

static diffusion_model *qwen_image_open(const diffusion_files *files, gguf_ctx *dit);

static float *time_modulation(qwen_image *m, float t, float *time_vec) {
	float emb[TIME_EMBED_DIM];
	int	  half = TIME_EMBED_DIM / 2;
	for (int j = 0; j < half; j++) {
		float freq	  = expf(-logf(TIME_MAX_PERIOD) * (float)j / (float)half);
		emb[j]		  = cosf(t * freq);
		emb[j + half] = sinf(t * freq);
	}
	float *mid = xmalloc((size_t)m->hidden * sizeof(float));
	nn_matmul(m->be, &m->time_1, emb, mid, m->hidden, TIME_EMBED_DIM, 1);
	nn_activate(m->be, NN_SILU, mid, (size_t)m->hidden);
	nn_matmul(m->be, &m->time_2, mid, time_vec, m->hidden, m->hidden, 1);
	nn_activate(m->be, NN_SILU, time_vec, (size_t)m->hidden);
	free(mid);

	float *mod = xmalloc((size_t)MOD_CHUNKS * m->hidden * sizeof(float));
	nn_matmul(m->be, &m->modulation, time_vec, mod, MOD_CHUNKS * m->hidden, m->hidden, 1);
	return mod;
}

static void layer_forward(qwen_image *m, const dit_layer *l, float *x, int rows, const float *mod,
						  const float *rope, float *k_out, float *v_out, const float *prefix_k,
						  const float *prefix_v, int prefix_n, layer_scratch *s) {
	backend		*be			= m->be;
	int			 dim		= m->hidden;
	const float *scale_attn = mod, *gate_attn = mod + dim;
	const float *scale_mlp = mod + 2 * dim, *gate_mlp = mod + 3 * dim;

	nn_layernorm_modulate(be, x, s->h, rows, dim, scale_attn);
	nn_matmul(be, &l->q, s->h, s->q, dim, dim, rows);
	nn_matmul(be, &l->k, s->h, s->norm, dim, dim, rows);
	nn_matmul(be, &l->v, s->h, v_out, dim, dim, rows);

	nn_rmsnorm_per_head(be, s->q, l->norm_q, s->q, m->heads, m->head_dim, NORM_EPS, rows);
	nn_rmsnorm_per_head(be, s->norm, l->norm_k, k_out, m->heads, m->head_dim, NORM_EPS, rows);
	nn_rope_apply(be, s->q, rows, m->heads, m->head_dim, rope);
	nn_rope_apply(be, k_out, rows, m->heads, m->head_dim, rope);

	nn_kv kv = prefix_k ? (nn_kv){.k0 = prefix_k,
								  .v0 = prefix_v,
								  .n0 = prefix_n,
								  .k1 = k_out,
								  .v1 = v_out,
								  .n1 = rows}
						: (nn_kv){.k0 = k_out, .v0 = v_out, .n0 = rows, .causal = true};
	nn_attention(be, s->a, s->q, rows, m->heads, m->head_dim, &kv);
	nn_matmul(be, &l->o, s->a, s->h, dim, dim, rows);
	nn_gated_add(be, x, s->h, rows, dim, gate_attn);

	nn_layernorm_modulate(be, x, s->h, rows, dim, scale_mlp);
	nn_matmul(be, &l->gate_up, s->h, s->gu, 2 * m->inter, dim, rows);
	nn_silu_mul(be, s->gu, s->a, rows, m->inter);
	nn_matmul(be, &l->down, s->a, s->h, dim, m->inter, rows);
	nn_gated_add(be, x, s->h, rows, dim, gate_mlp);
}

static layer_scratch scratch_alloc(const qwen_image *m, int rows) {
	size_t		  n = (size_t)rows * m->hidden;
	layer_scratch s = {
		.q	  = xmalloc(n * sizeof(float)),
		.a	  = xmalloc(MAX(n, (size_t)rows * m->inter) * sizeof(float)),
		.h	  = xmalloc(MAX(n, (size_t)rows * m->inter) * sizeof(float)),
		.gu	  = xmalloc((size_t)rows * 2 * m->inter * sizeof(float)),
		.norm = xmalloc(n * sizeof(float)),
	};
	return s;
}

static void scratch_free(layer_scratch *s) {
	free(s->q);
	free(s->a);
	free(s->h);
	free(s->gu);
	free(s->norm);
}

static float *project_text(qwen_image *m, const float *hidden, int rows) {
	size_t n = (size_t)rows * m->hidden;
	float *a = xmalloc(n * sizeof(float));
	float *b = xmalloc(n * sizeof(float));
	nn_rmsnorm(m->be, hidden, m->txt_norm, a, m->hidden, NORM_EPS, rows);
	nn_matmul(m->be, &m->txt_in, a, b, m->hidden, m->hidden, rows);
	nn_activate(m->be, NN_GELU, b, n);
	nn_matmul(m->be, &m->txt_out, b, a, m->hidden, m->hidden, rows);
	free(b);
	return a;
}

static diffusion_cond *build_prefix(qwen_image *m, const float *hidden, int rows) {
	size_t			layer_floats = (size_t)rows * m->hidden;
	diffusion_cond *c			 = xcalloc(1, sizeof(*c));
	c->n_tokens					 = rows;
	c->k						 = xmalloc(layer_floats * m->n_layers * sizeof(float));
	c->v						 = xmalloc(layer_floats * m->n_layers * sizeof(float));

	float *x	= project_text(m, hidden, rows);
	float *time = xmalloc((size_t)m->hidden * sizeof(float));
	float *mod	= time_modulation(m, 0.0f, time);
	int	  *pos	= xmalloc((size_t)rows * ROPE_AXES * sizeof(int));
	int	   half = m->head_dim / 2;
	float *rope = xmalloc((size_t)rows * half * 2 * sizeof(float));
	for (int i = 0; i < rows; i++)
		pos[i * ROPE_AXES] = pos[i * ROPE_AXES + 1] = pos[i * ROPE_AXES + 2] = i;
	nn_rope_table(rope, pos, rows, m->rope_axes, ROPE_THETA);

	layer_scratch s = scratch_alloc(m, rows);
	for (int li = 0; li < m->n_layers; li++)
		layer_forward(m, &m->layers[li], x, rows, mod, rope, c->k + li * layer_floats,
					  c->v + li * layer_floats, NULL, NULL, 0, &s);
	scratch_free(&s);
	free(x);
	free(time);
	free(mod);
	free(pos);
	free(rope);
	return c;
}

static status_code encode_text(qwen_image *m, const char *const *texts, int n, float **hidden,
							   int *rows) {
	config cfg	 = config_defaults();
	cfg.model	 = m->llm_path;
	cfg.device	 = "cpu0";
	cfg.ctx_size = TEXT_CTX_TOKENS;
	cfg.debug	 = log_get_level() <= LOG_DEBUG;
	config_init(&cfg);

	context		ctx;
	status_code s = context_init(&ctx, &cfg);
	if (s != OK)
		return s;

	int32_t ids[TEXT_CTX_TOKENS];
	int skip = tokenizer_encode_with_specials(&ctx.tok, TEXT_SYSTEM, 0, ids, TEXT_CTX_TOKENS, NULL);
	for (int i = 0; i < n && s == OK; i++) {
		char prompt[8192];
		snprintf(prompt, sizeof(prompt), "%s%s%s%s", TEXT_SYSTEM, TEXT_USER_PREFIX, texts[i],
				 TEXT_USER_SUFFIX);
		int total = tokenizer_encode_with_specials(&ctx.tok, prompt, 0, ids, TEXT_CTX_TOKENS, NULL);
		if (total <= skip || total >= TEXT_CTX_TOKENS) {
			ERROR("prompt does not fit in %d tokens", TEXT_CTX_TOKENS);
			s = ERR_INVALID_ARG;
			break;
		}
		context_reset(&ctx);
		float *all = xmalloc((size_t)total * ctx.m.dim * sizeof(float));
		s = compute_hidden_states(&ctx.m, &ctx.kv, &ctx.scratch, ids, total, ctx.flash_attn, all);
		rows[i]	  = total - skip;
		hidden[i] = xmalloc((size_t)rows[i] * ctx.m.dim * sizeof(float));
		memcpy(hidden[i], all + (size_t)skip * ctx.m.dim,
			   (size_t)rows[i] * ctx.m.dim * sizeof(float));
		free(all);
	}
	context_free(&ctx);
	return s;
}

static status_code qwen_image_encode(diffusion_model *dm, const char *const *texts, int n,
									 diffusion_cond **out) {
	qwen_image *m		  = (qwen_image *)dm;
	float	   *hidden[2] = {0};
	int			rows[2]	  = {0};
	status_code s		  = encode_text(m, texts, n, hidden, rows);
	for (int i = 0; i < n; i++) {
		if (s == OK)
			out[i] = build_prefix(m, hidden[i], rows[i]);
		free(hidden[i]);
	}
	return s;
}

static void qwen_image_cond_free(diffusion_cond *c) {
	free(c->k);
	free(c->v);
	free(c);
}

static latent_shape qwen_image_shape(int width, int height) {
	return (latent_shape){width / 16, height / 16, 64};
}

static status_code qwen_image_predict(diffusion_model *dm, const float *latent, latent_shape shape,
									  float sigma, const diffusion_cond *cond, float *velocity) {
	qwen_image *m			 = (qwen_image *)dm;
	int			rows		 = shape.lw * shape.lh;
	size_t		layer_floats = (size_t)cond->n_tokens * m->hidden;

	float *time = xmalloc((size_t)m->hidden * sizeof(float));
	float *mod	= time_modulation(m, sigma * TIMESTEP_SCALE, time);

	float *x = xmalloc((size_t)rows * m->hidden * sizeof(float));
	nn_matmul(m->be, &m->img_in, latent, x, m->hidden, m->in_channels, rows);

	int *pos = xmalloc((size_t)rows * ROPE_AXES * sizeof(int));
	for (int i = 0; i < rows; i++) {
		pos[i * ROPE_AXES]	   = cond->n_tokens;
		pos[i * ROPE_AXES + 1] = i / shape.lw - (shape.lh - shape.lh / 2);
		pos[i * ROPE_AXES + 2] = i % shape.lw - (shape.lw - shape.lw / 2);
	}
	int	   half = m->head_dim / 2;
	float *rope = xmalloc((size_t)rows * half * 2 * sizeof(float));
	nn_rope_table(rope, pos, rows, m->rope_axes, ROPE_THETA);

	layer_scratch s		= scratch_alloc(m, rows);
	float		 *k_img = xmalloc((size_t)rows * m->hidden * sizeof(float));
	float		 *v_img = xmalloc((size_t)rows * m->hidden * sizeof(float));
	for (int li = 0; li < m->n_layers; li++)
		layer_forward(m, &m->layers[li], x, rows, mod, rope, k_img, v_img,
					  cond->k + li * layer_floats, cond->v + li * layer_floats, cond->n_tokens, &s);

	float *scale = xmalloc((size_t)m->hidden * sizeof(float));
	nn_matmul(m->be, &m->norm_out, time, scale, m->hidden, m->hidden, 1);
	nn_layernorm_modulate(m->be, x, s.h, rows, m->hidden, scale);
	nn_matmul(m->be, &m->proj_out, s.h, velocity, m->out_channels, m->hidden, rows);

	scratch_free(&s);
	free(k_img);
	free(v_img);
	free(scale);
	free(rope);
	free(pos);
	free(x);
	free(mod);
	free(time);
	return OK;
}

static status_code qwen_image_decode(diffusion_model *dm, const float *latent, latent_shape shape,
									 diffusion_image *out) {
	return qwen_vae_decode(((qwen_image *)dm)->vae, latent, shape, out);
}

static void qwen_image_preview(const float *x0, latent_shape shape, diffusion_image *out) {
	out->width	  = shape.lw;
	out->height	  = shape.lh;
	out->channels = 4;
	out->pixels	  = xmalloc((size_t)shape.lw * shape.lh * 4);
	for (int p = 0; p < shape.lw * shape.lh; p++) {
		const float *z = x0 + (size_t)p * shape.lc;
		float		 c[4];
		memcpy(c, qwen_image_preview_rgb_bias, sizeof(qwen_image_preview_rgb_bias));
		c[3] = qwen_image_preview_alpha_bias;
		for (int d = 0; d < shape.lc; d++) {
			for (int j = 0; j < 3; j++)
				c[j] += z[d] * qwen_image_preview_rgb_proj[d * 3 + j];
			c[3] += z[d] * qwen_image_preview_alpha_proj[d];
		}
		for (int j = 0; j < 4; j++) {
			float v						   = j < 3 ? c[j] * 0.5f + 0.5f : c[j];
			out->pixels[(size_t)p * 4 + j] = (uint8_t)(MAX(0.0f, MIN(1.0f, v)) * 255.0f + 0.5f);
		}
	}
}

static const diffusion_ops qwen_image_ops = {
	.name	   = "qwen-image",
	.schedule  = {.base_seq		  = 256,
				  .max_seq		  = 8192,
				  .base_shift	  = 0.5f,
				  .max_shift	  = 0.9f,
				  .shift_terminal = 0.02f},
	.detect	   = qwen_image_detect,
	.open	   = qwen_image_open,
	.close	   = qwen_image_close,
	.shape	   = qwen_image_shape,
	.encode	   = qwen_image_encode,
	.cond_free = qwen_image_cond_free,
	.predict   = qwen_image_predict,
	.decode	   = qwen_image_decode,
	.preview   = qwen_image_preview,
};

static diffusion_model *qwen_image_open(const diffusion_files *files, gguf_ctx *dit) {
	qwen_image *m = xcalloc(1, sizeof(*m));
	m->base.ops	  = &qwen_image_ops;
	m->dit		  = *dit;
	m->llm_path	  = xstrdup(files->llm);
	if (backend_create("cpu", 0, &m->be) != OK) {
		qwen_image_close(&m->base);
		return NULL;
	}
	detect_config(m);
	INFO("qwen-image: %d layers, hidden %d, %d heads", m->n_layers, m->hidden, m->heads);
	if (load_dit(m) != OK || !(m->vae = qwen_vae_open(m->be, files->vae))) {
		qwen_image_close(&m->base);
		return NULL;
	}
	return &m->base;
}

DIFFUSION_REGISTER(qwen_image, &qwen_image_ops)
