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

typedef struct qwen_vae qwen_vae;

struct qwen_vae {
	backend	 *be;
	gguf_ctx  g;
	nn_conv	  post_quant, conv_in, head;
	res_block mid0, mid2;
	buffer	  attn_gamma;
	nn_conv	  attn_q, attn_k, attn_v, attn_proj;
	up_stage  stages[VAE_STAGES];
	buffer	  head_gamma;
};

static const gguf_tensor *vae_tensor(qwen_vae *v, const char *prefix, const char *suffix) {
	char name[192];
	snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
	const gguf_tensor *t = gguf_find_tensor(&v->g, name);
	if (!t)
		ERROR("vae tensor '%s' not found", name);
	return t;
}

static status_code load_vec(qwen_vae *v, const char *prefix, const char *suffix, buffer *out) {
	const gguf_tensor *t = vae_tensor(v, prefix, suffix);
	return t ? nn_vec_load(v->be, t, 0.0f, out) : ERR_NOT_FOUND;
}

static status_code load_conv(qwen_vae *v, const char *prefix, nn_conv *c) {
	const gguf_tensor *w = vae_tensor(v, prefix, "weight");
	const gguf_tensor *b = vae_tensor(v, prefix, "bias");
	return w && b ? nn_conv_load(v->be, w, b, 0, 0, c) : ERR_NOT_FOUND;
}

static status_code load_res_block(qwen_vae *v, const char *prefix, res_block *rb) {
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

static void free_res_block(qwen_vae *v, res_block *rb) {
	nn_free(v->be, &rb->gamma1);
	nn_free(v->be, &rb->gamma2);
	nn_conv_free(v->be, &rb->conv1);
	nn_conv_free(v->be, &rb->conv2);
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

static status_code load_attention(qwen_vae *v) {
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

static status_code load_all(qwen_vae *v) {
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

static status_code norm_silu(backend *be, const buffer *x, const buffer *gamma, int ch, int pixels,
							 buffer *out) {
	NN_TRY(nn_alloc(be, (size_t)pixels * ch, out));
	status_code s = nn_rmsnorm(be, x, gamma, out, ch, VAE_RMS_EPS, pixels);
	if (s == OK)
		s = be->activate(be, out, pixels * ch, NN_SILU);
	if (s != OK)
		nn_free(be, out);
	return s;
}

static status_code res_block_forward(qwen_vae *v, const res_block *rb, const buffer *x, int H,
									 int W, buffer *y) {
	backend	   *be = v->be;
	int			P  = H * W;
	buffer		t = {0}, u = {0}, sc = {0};
	status_code s;

	NN_GOTO(norm_silu(be, x, &rb->gamma1, rb->in, P, &t));
	NN_GOTO(nn_alloc(be, (size_t)P * rb->out, &u));
	NN_GOTO(nn_conv_forward(be, &rb->conv1, &t, &u, H, W, 0));
	nn_free(be, &t);

	NN_GOTO(norm_silu(be, &u, &rb->gamma2, rb->out, P, &t));
	NN_GOTO(nn_alloc(be, (size_t)P * rb->out, y));
	NN_GOTO(nn_conv_forward(be, &rb->conv2, &t, y, H, W, 0));

	if (rb->has_shortcut) {
		NN_GOTO(nn_alloc(be, (size_t)P * rb->out, &sc));
		NN_GOTO(nn_conv_forward(be, &rb->shortcut, x, &sc, H, W, 0));
		NN_GOTO(nn_add(be, y, &sc, rb->out, P));
	} else
		NN_GOTO(nn_add(be, y, x, rb->out, P));
done:
	nn_free(be, &t);
	nn_free(be, &u);
	nn_free(be, &sc);
	if (s != OK)
		nn_free(be, y);
	return s;
}

static status_code attention_block(qwen_vae *v, buffer *x, int H, int W) {
	backend	   *be = v->be;
	int			P  = H * W;
	int			ch = v->attn_proj.oc;
	buffer		xn = {0}, q = {0}, k = {0}, val = {0}, att = {0}, proj = {0};
	status_code s;

	NN_GOTO(nn_alloc(be, (size_t)P * ch, &xn));
	NN_GOTO(nn_alloc(be, (size_t)P * ch, &q));
	NN_GOTO(nn_alloc(be, (size_t)P * ch, &k));
	NN_GOTO(nn_alloc(be, (size_t)P * ch, &val));
	NN_GOTO(nn_rmsnorm(be, x, &v->attn_gamma, &xn, ch, VAE_RMS_EPS, P));
	NN_GOTO(nn_conv_forward(be, &v->attn_q, &xn, &q, H, W, 0));
	NN_GOTO(nn_conv_forward(be, &v->attn_k, &xn, &k, H, W, 0));
	NN_GOTO(nn_conv_forward(be, &v->attn_v, &xn, &val, H, W, 0));

	NN_GOTO(nn_alloc(be, (size_t)P * ch, &att));
	dense_attention_desc d = {.q		= &q,
							  .k0		= &k,
							  .v0		= &val,
							  .out		= &att,
							  .n_q		= P,
							  .n_heads	= 1,
							  .head_dim = ch,
							  .n0		= P};
	NN_GOTO(be->attention_dense(be, &d));

	NN_GOTO(nn_alloc(be, (size_t)P * ch, &proj));
	NN_GOTO(nn_conv_forward(be, &v->attn_proj, &att, &proj, H, W, 0));
	NN_GOTO(nn_add(be, x, &proj, ch, P));
done:
	nn_free(be, &xn);
	nn_free(be, &q);
	nn_free(be, &k);
	nn_free(be, &val);
	nn_free(be, &att);
	nn_free(be, &proj);
	return s;
}

static status_code up_stage_forward(qwen_vae *v, const up_stage *st, buffer *x, int *H, int *W) {
	backend	   *be	= v->be;
	buffer		cur = *x, next = {0}, up = {0};
	status_code s = OK;
	for (int b = 0; b < VAE_BLOCKS_PER_STAGE; b++) {
		NN_GOTO(res_block_forward(v, &st->blocks[b], &cur, *H, *W, &next));
		if (cur.handle != x->handle)
			nn_free(be, &cur);
		cur	 = next;
		next = (buffer){0};
	}
	if (st->up) {
		int oH = *H * VAE_UPSAMPLE, oW = *W * VAE_UPSAMPLE;
		NN_GOTO(nn_alloc(be, (size_t)oH * oW * st->out, &up));
		NN_GOTO(nn_conv_forward(be, &st->resample, &cur, &up, oH, oW, 1));
		int					   factor = st->frames_up * VAE_UPSAMPLE * VAE_UPSAMPLE;
		upsample_shortcut_desc d	  = {.in		   = x,
										 .out		   = &up,
										 .height	   = *H,
										 .width		   = *W,
										 .in_channels  = st->in,
										 .out_channels = st->out,
										 .repeats	   = st->out * factor / st->in,
										 .frames_up	   = st->frames_up,
										 .factor	   = VAE_UPSAMPLE};
		NN_GOTO(be->upsample_shortcut_add(be, &d));
		nn_free(be, &cur);
		nn_free(be, x);
		*x = up;
		up = (buffer){0};
		*H = oH;
		*W = oW;
		return OK;
	}
	nn_free(be, x);
	*x = cur;
	return OK;
done:
	if (cur.handle != x->handle)
		nn_free(be, &cur);
	nn_free(be, &up);
	nn_free(be, x);
	return s;
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

static status_code vae_run(qwen_vae *v, const float *latent, int H, int W, buffer *rgba) {
	backend	   *be = v->be;
	buffer		z = {0}, q = {0}, x = {0}, t = {0};
	status_code s;

	float *denorm = denormalize_latent(latent, H * W);
	s			  = nn_upload(be, denorm, (size_t)H * W * VAE_LATENT_CHANNELS, &z);
	free(denorm);
	if (s != OK)
		return s;

	NN_GOTO(nn_alloc(be, (size_t)H * W * v->post_quant.oc, &q));
	NN_GOTO(nn_conv_forward(be, &v->post_quant, &z, &q, H, W, 0));
	NN_GOTO(nn_alloc(be, (size_t)H * W * v->conv_in.oc, &x));
	NN_GOTO(nn_conv_forward(be, &v->conv_in, &q, &x, H, W, 0));
	nn_free(be, &z);
	nn_free(be, &q);

	NN_GOTO(res_block_forward(v, &v->mid0, &x, H, W, &t));
	nn_free(be, &x);
	NN_GOTO(attention_block(v, &t, H, W));
	NN_GOTO(res_block_forward(v, &v->mid2, &t, H, W, &x));
	nn_free(be, &t);

	for (int i = 0; i < VAE_STAGES; i++) {
		NN_GOTO(up_stage_forward(v, &v->stages[i], &x, &H, &W));
		DEBUG("vae stage %d done (%dx%d)", i, W, H);
	}

	NN_GOTO(norm_silu(be, &x, &v->head_gamma, v->head.ic, H * W, &t));
	NN_GOTO(nn_alloc(be, (size_t)H * W * VAE_OUT_CHANNELS, rgba));
	NN_GOTO(nn_conv_forward(be, &v->head, &t, rgba, H, W, 0));
done:
	if (s != OK)
		nn_free(be, rgba);
	nn_free(be, &z);
	nn_free(be, &q);
	nn_free(be, &x);
	nn_free(be, &t);
	return s;
}

static status_code qwen_vae_decode(qwen_vae *v, const float *latent, latent_shape shape,
								   diffusion_image *out) {
	int			H = shape.lh, W = shape.lw;
	buffer		rgba = {0};
	status_code s	 = vae_run(v, latent, H, W, &rgba);
	if (s != OK)
		return s;
	int scale = 1;
	for (int i = 0; i < VAE_STAGES; i++)
		scale *= v->stages[i].up ? VAE_UPSAMPLE : 1;
	H *= scale;
	W *= scale;

	size_t n	= (size_t)H * W * VAE_OUT_CHANNELS;
	float *host = xmalloc(n * sizeof(float));
	s			= v->be->buffer_read_f32(v->be, &rgba, host, (int)n);
	nn_free(v->be, &rgba);
	if (s == OK) {
		out->width	  = W;
		out->height	  = H;
		out->channels = VAE_OUT_CHANNELS;
		out->pixels	  = xmalloc(n);
		for (size_t i = 0; i < n; i++)
			out->pixels[i] = to_byte(host[i]);
	}
	free(host);
	return s;
}

typedef struct {
	weight q, k, v, o, gate, up, down;
	buffer norm_q, norm_k;
} dit_layer;

typedef struct {
	diffusion_model base;
	backend		   *be;
	gguf_ctx		dit;
	char		   *llm_path;
	char		   *device;
	qwen_vae	   *vae;
	int				hidden, heads, head_dim, inter, n_layers, in_channels, out_channels;
	int				rope_axes[ROPE_AXES];
	weight			img_in, modulation, norm_out, proj_out, time_1, time_2, txt_in, txt_out;
	buffer			txt_norm;
	dit_layer	   *layers;
} qwen_image;

struct diffusion_cond {
	backend *be;
	int		 n_tokens;
	buffer	 k, v;
};

typedef struct {
	buffer q, a, h, gate, up, raw;
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
		{"img_mlp.out.weight", offsetof(dit_layer, down)},
	};
	for (size_t i = 0; i < ARRAY_LEN(mats); i++) {
		const gguf_tensor *t = find(m, "transformer_blocks.%d.%s", li, mats[i].name);
		if (!t)
			return ERR_NOT_FOUND;
		NN_TRY(weight_load(m->be, t, (weight *)((char *)l + mats[i].off)));
	}
	const gguf_tensor *gu = find(m, "transformer_blocks.%d.%s", li, "img_mlp.gate_up.weight");
	if (!gu)
		return ERR_NOT_FOUND;
	NN_TRY(weight_load_rows(m->be, gu, 0, m->inter, &l->gate));
	NN_TRY(weight_load_rows(m->be, gu, m->inter, m->inter, &l->up));

	const gguf_tensor *nq = find(m, "transformer_blocks.%d.%s", li, "attn.norm_q.weight");
	const gguf_tensor *nk = find(m, "transformer_blocks.%d.%s", li, "attn.norm_k.weight");
	if (!nq || !nk)
		return ERR_NOT_FOUND;
	NN_TRY(nn_vec_load(m->be, nq, 0.0f, &l->norm_q));
	return nn_vec_load(m->be, nk, 0.0f, &l->norm_k);
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
	for (size_t i = 0; i < ARRAY_LEN(globals); i++)
		NN_TRY(load_named(m, globals[i].name, (weight *)((char *)m + globals[i].off)));

	const gguf_tensor *norm = gguf_find_tensor(&m->dit, "txt_in.text_norm.weight");
	if (!norm)
		return ERR_NOT_FOUND;
	NN_TRY(nn_vec_load(m->be, norm, 1.0f, &m->txt_norm));

	m->layers = xcalloc((size_t)m->n_layers, sizeof(dit_layer));
	for (int li = 0; li < m->n_layers; li++)
		NN_TRY(load_layer(m, li, &m->layers[li]));
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
	nn_free(m->be, &m->txt_norm);
	for (int li = 0; m->layers && li < m->n_layers; li++) {
		dit_layer *l	  = &m->layers[li];
		weight	  *mats[] = {&l->q, &l->k, &l->v, &l->o, &l->gate, &l->up, &l->down};
		for (size_t i = 0; i < ARRAY_LEN(mats); i++)
			if (mats[i]->buf.owner)
				weight_free(m->be, mats[i]);
		nn_free(m->be, &l->norm_q);
		nn_free(m->be, &l->norm_k);
	}
	free(m->layers);
	qwen_vae_close(m->vae);
	gguf_free(&m->dit);
	free(m->llm_path);
	free(m->device);
	if (m->be)
		backend_destroy(m->be);
	free(m);
}

static diffusion_model *qwen_image_open(const diffusion_files *files, gguf_ctx *dit);

static status_code time_modulation(qwen_image *m, float t, buffer *time_vec, buffer *mod) {
	float emb[TIME_EMBED_DIM];
	int	  half = TIME_EMBED_DIM / 2;
	for (int j = 0; j < half; j++) {
		float freq	  = expf(-logf(TIME_MAX_PERIOD) * (float)j / (float)half);
		emb[j]		  = cosf(t * freq);
		emb[j + half] = sinf(t * freq);
	}
	backend	   *be = m->be;
	buffer		in = {0}, mid = {0};
	status_code s;

	NN_GOTO(nn_upload(be, emb, TIME_EMBED_DIM, &in));
	NN_GOTO(nn_alloc(be, m->hidden, &mid));
	NN_GOTO(nn_alloc(be, m->hidden, time_vec));
	NN_GOTO(nn_alloc(be, (size_t)MOD_CHUNKS * m->hidden, mod));
	NN_GOTO(nn_matmul(be, &m->time_1, &in, &mid, m->hidden, TIME_EMBED_DIM, 1));
	NN_GOTO(be->activate(be, &mid, m->hidden, NN_SILU));
	NN_GOTO(nn_matmul(be, &m->time_2, &mid, time_vec, m->hidden, m->hidden, 1));
	NN_GOTO(be->activate(be, time_vec, m->hidden, NN_SILU));
	NN_GOTO(nn_matmul(be, &m->modulation, time_vec, mod, MOD_CHUNKS * m->hidden, m->hidden, 1));
done:
	nn_free(be, &in);
	nn_free(be, &mid);
	if (s != OK) {
		nn_free(be, time_vec);
		nn_free(be, mod);
	}
	return s;
}

static status_code layer_forward(qwen_image *m, const dit_layer *l, buffer *x, int rows,
								 const buffer *mod, const buffer *rope, buffer *k_out,
								 buffer *v_out, const buffer *prefix_k, const buffer *prefix_v,
								 int prefix_n, layer_scratch *s) {
	backend *be			= m->be;
	int		 dim		= m->hidden;
	size_t	 chunk		= (size_t)dim * sizeof(float);
	buffer	 scale_attn = buffer_slice(mod, 0, chunk), gate_attn = buffer_slice(mod, chunk, chunk);
	buffer	 scale_mlp = buffer_slice(mod, 2 * chunk, chunk),
			 gate_mlp  = buffer_slice(mod, 3 * chunk, chunk);

	NN_TRY(be->layernorm_modulate(be, x, &scale_attn, &s->h, rows, dim, NORM_EPS));
	NN_TRY(nn_matmul(be, &l->q, &s->h, &s->raw, dim, dim, rows));
	NN_TRY(be->rmsnorm_per_head_batch(be, &s->raw, &l->norm_q, &s->q, m->heads, m->head_dim,
									  NORM_EPS, rows));
	NN_TRY(nn_matmul(be, &l->k, &s->h, &s->raw, dim, dim, rows));
	NN_TRY(be->rmsnorm_per_head_batch(be, &s->raw, &l->norm_k, k_out, m->heads, m->head_dim,
									  NORM_EPS, rows));
	NN_TRY(nn_matmul(be, &l->v, &s->h, v_out, dim, dim, rows));
	NN_TRY(be->rope_table(be, &s->q, rope, rows, m->heads, m->head_dim));
	NN_TRY(be->rope_table(be, k_out, rope, rows, m->heads, m->head_dim));

	dense_attention_desc d = {
		.q = &s->q, .out = &s->a, .n_q = rows, .n_heads = m->heads, .head_dim = m->head_dim};
	if (prefix_k) {
		d.k0 = prefix_k;
		d.v0 = prefix_v;
		d.n0 = prefix_n;
		d.k1 = k_out;
		d.v1 = v_out;
		d.n1 = rows;
	} else {
		d.k0	 = k_out;
		d.v0	 = v_out;
		d.n0	 = rows;
		d.causal = 1;
	}
	NN_TRY(be->attention_dense(be, &d));
	NN_TRY(nn_matmul(be, &l->o, &s->a, &s->h, dim, dim, rows));
	NN_TRY(be->gated_add(be, x, &s->h, &gate_attn, rows, dim));

	NN_TRY(be->layernorm_modulate(be, x, &scale_mlp, &s->h, rows, dim, NORM_EPS));
	NN_TRY(nn_matmul(be, &l->gate, &s->h, &s->gate, m->inter, dim, rows));
	NN_TRY(nn_matmul(be, &l->up, &s->h, &s->up, m->inter, dim, rows));
	NN_TRY(be->ffn_activate_batch(be, &s->gate, &s->up, &s->a, m->inter, NN_SILU, rows));
	NN_TRY(nn_matmul(be, &l->down, &s->a, &s->h, dim, m->inter, rows));
	return be->gated_add(be, x, &s->h, &gate_mlp, rows, dim);
}

static status_code scratch_alloc(const qwen_image *m, int rows, layer_scratch *s) {
	memset(s, 0, sizeof(*s));
	size_t	n = (size_t)rows * m->hidden, wide = (size_t)rows * m->inter;
	buffer *bufs[]	= {&s->q, &s->a, &s->h, &s->gate, &s->up, &s->raw};
	size_t	sizes[] = {n, MAX(n, wide), MAX(n, wide), wide, wide, n};
	for (size_t i = 0; i < ARRAY_LEN(bufs); i++)
		NN_TRY(nn_alloc(m->be, sizes[i], bufs[i]));
	return OK;
}

static void scratch_free(const qwen_image *m, layer_scratch *s) {
	buffer *bufs[] = {&s->q, &s->a, &s->h, &s->gate, &s->up, &s->raw};
	for (size_t i = 0; i < ARRAY_LEN(bufs); i++)
		nn_free(m->be, bufs[i]);
}

static status_code project_text(qwen_image *m, const float *hidden, int rows, buffer *out) {
	backend	   *be = m->be;
	size_t		n  = (size_t)rows * m->hidden;
	buffer		in = {0}, a = {0}, b = {0};
	status_code s;

	NN_GOTO(nn_upload(be, hidden, n, &in));
	NN_GOTO(nn_alloc(be, n, &a));
	NN_GOTO(nn_alloc(be, n, &b));
	NN_GOTO(nn_alloc(be, n, out));
	NN_GOTO(nn_rmsnorm(be, &in, &m->txt_norm, &a, m->hidden, NORM_EPS, rows));
	NN_GOTO(nn_matmul(be, &m->txt_in, &a, &b, m->hidden, m->hidden, rows));
	NN_GOTO(be->activate(be, &b, (int)n, NN_GELU));
	NN_GOTO(nn_matmul(be, &m->txt_out, &b, out, m->hidden, m->hidden, rows));
done:
	nn_free(be, &in);
	nn_free(be, &a);
	nn_free(be, &b);
	if (s != OK)
		nn_free(be, out);
	return s;
}

static status_code upload_rope(qwen_image *m, const int *pos, int rows, buffer *out) {
	int	   half = m->head_dim / 2;
	float *host = xmalloc((size_t)rows * half * 2 * sizeof(float));
	nn_rope_table(host, pos, rows, m->rope_axes, ROPE_THETA);
	status_code s = nn_upload(m->be, host, (size_t)rows * half * 2, out);
	free(host);
	return s;
}

static void qwen_image_cond_free(diffusion_cond *c) {
	nn_free(c->be, &c->k);
	nn_free(c->be, &c->v);
	free(c);
}

static status_code build_prefix(qwen_image *m, const float *hidden, int rows,
								diffusion_cond **out) {
	backend		   *be			= m->be;
	size_t			layer_bytes = (size_t)rows * m->hidden * sizeof(float);
	diffusion_cond *c			= xcalloc(1, sizeof(*c));
	c->be						= be;
	c->n_tokens					= rows;
	buffer		  x = {0}, time = {0}, mod = {0}, rope = {0};
	layer_scratch sc;
	status_code	  s;

	memset(&sc, 0, sizeof(sc));
	NN_GOTO(nn_alloc(be, (size_t)rows * m->hidden * m->n_layers, &c->k));
	NN_GOTO(nn_alloc(be, (size_t)rows * m->hidden * m->n_layers, &c->v));
	NN_GOTO(project_text(m, hidden, rows, &x));
	NN_GOTO(time_modulation(m, 0.0f, &time, &mod));
	int *pos = xmalloc((size_t)rows * ROPE_AXES * sizeof(int));
	for (int i = 0; i < rows; i++)
		pos[i * ROPE_AXES] = pos[i * ROPE_AXES + 1] = pos[i * ROPE_AXES + 2] = i;
	s = upload_rope(m, pos, rows, &rope);
	free(pos);
	if (s != OK)
		goto done;
	NN_GOTO(scratch_alloc(m, rows, &sc));
	for (int li = 0; li < m->n_layers; li++) {
		buffer k = buffer_slice(&c->k, li * layer_bytes, layer_bytes);
		buffer v = buffer_slice(&c->v, li * layer_bytes, layer_bytes);
		NN_GOTO(
			layer_forward(m, &m->layers[li], &x, rows, &mod, &rope, &k, &v, NULL, NULL, 0, &sc));
	}
done:
	scratch_free(m, &sc);
	nn_free(be, &x);
	nn_free(be, &time);
	nn_free(be, &mod);
	nn_free(be, &rope);
	if (s == OK)
		*out = c;
	else
		qwen_image_cond_free(c);
	return s;
}

static status_code encode_text(qwen_image *m, const char *const *texts, int n, float **hidden,
							   int *rows) {
	config cfg	 = config_defaults();
	cfg.model	 = m->llm_path;
	cfg.device	 = m->device;
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
			s = build_prefix(m, hidden[i], rows[i], &out[i]);
		free(hidden[i]);
	}
	return s;
}

static latent_shape qwen_image_shape(int width, int height) {
	return (latent_shape){width / 16, height / 16, 64};
}

static status_code qwen_image_predict(diffusion_model *dm, const float *latent, latent_shape shape,
									  float sigma, const diffusion_cond *cond, float *velocity) {
	qwen_image	 *m			  = (qwen_image *)dm;
	backend		 *be		  = m->be;
	int			  rows		  = shape.lw * shape.lh;
	size_t		  layer_bytes = (size_t)cond->n_tokens * m->hidden * sizeof(float);
	buffer		  lat = {0}, time = {0}, mod = {0}, x = {0}, rope = {0}, k_img = {0}, v_img = {0},
				  scale = {0}, vel = {0};
	layer_scratch sc;
	status_code	  s;

	memset(&sc, 0, sizeof(sc));
	NN_GOTO(time_modulation(m, sigma * TIMESTEP_SCALE, &time, &mod));
	NN_GOTO(nn_upload(be, latent, (size_t)rows * m->in_channels, &lat));
	NN_GOTO(nn_alloc(be, (size_t)rows * m->hidden, &x));
	NN_GOTO(nn_matmul(be, &m->img_in, &lat, &x, m->hidden, m->in_channels, rows));

	int *pos = xmalloc((size_t)rows * ROPE_AXES * sizeof(int));
	for (int i = 0; i < rows; i++) {
		pos[i * ROPE_AXES]	   = cond->n_tokens;
		pos[i * ROPE_AXES + 1] = i / shape.lw - (shape.lh - shape.lh / 2);
		pos[i * ROPE_AXES + 2] = i % shape.lw - (shape.lw - shape.lw / 2);
	}
	s = upload_rope(m, pos, rows, &rope);
	free(pos);
	if (s != OK)
		goto done;

	NN_GOTO(scratch_alloc(m, rows, &sc));
	NN_GOTO(nn_alloc(be, (size_t)rows * m->hidden, &k_img));
	NN_GOTO(nn_alloc(be, (size_t)rows * m->hidden, &v_img));
	for (int li = 0; li < m->n_layers; li++) {
		buffer pk = buffer_slice(&cond->k, li * layer_bytes, layer_bytes);
		buffer pv = buffer_slice(&cond->v, li * layer_bytes, layer_bytes);
		NN_GOTO(layer_forward(m, &m->layers[li], &x, rows, &mod, &rope, &k_img, &v_img, &pk, &pv,
							  cond->n_tokens, &sc));
	}

	NN_GOTO(nn_alloc(be, m->hidden, &scale));
	NN_GOTO(nn_matmul(be, &m->norm_out, &time, &scale, m->hidden, m->hidden, 1));
	NN_GOTO(be->layernorm_modulate(be, &x, &scale, &sc.h, rows, m->hidden, NORM_EPS));
	NN_GOTO(nn_alloc(be, (size_t)rows * m->out_channels, &vel));
	NN_GOTO(nn_matmul(be, &m->proj_out, &sc.h, &vel, m->out_channels, m->hidden, rows));
	NN_GOTO(be->buffer_read_f32(be, &vel, velocity, rows * m->out_channels));
done:
	scratch_free(m, &sc);
	nn_free(be, &lat);
	nn_free(be, &time);
	nn_free(be, &mod);
	nn_free(be, &x);
	nn_free(be, &rope);
	nn_free(be, &k_img);
	nn_free(be, &v_img);
	nn_free(be, &scale);
	nn_free(be, &vel);
	return s;
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

static status_code open_device(qwen_image *m) {
	char name[BACKEND_DEVICE_NAME_MAX];
	int	 index = 0;
	if (backend_parse_device(m->device, name, sizeof(name), &index) != 0) {
		ERROR("invalid device spec '%s' (expected <backend><n>, e.g. vulkan0)", m->device);
		return ERR_INVALID_ARG;
	}
	status_code s = backend_create(name, index, &m->be);
	if (s != OK)
		ERROR("failed to initialize device '%s'", m->device);
	else if (!nn_supported(m->be))
		return ERR_UNSUPPORTED;
	else if (!backend_has_cap(m->be, BCAP_IS_HOST))
		backend_host_use(m->be);
	return s;
}

static diffusion_model *qwen_image_open(const diffusion_files *files, gguf_ctx *dit) {
	qwen_image *m = xcalloc(1, sizeof(*m));
	m->base.ops	  = &qwen_image_ops;
	m->dit		  = *dit;
	m->llm_path	  = xstrdup(files->llm);
	m->device	  = xstrdup(files->device);
	if (open_device(m) != OK) {
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
