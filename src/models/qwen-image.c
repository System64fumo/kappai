#include "diffusion/diffusion.h"
#include "diffusion/dit.h"
#include "diffusion/image.h"
#include "diffusion/text_encoder.h"
#include "diffusion/vae.h"
#include "log.h"
#include "recipe/exec.h"

#include <math.h>
#include <stdio.h>

#define LATENT_CHANNELS 64
#define LATENT_PATCH 16
#define TIME_EMBED_DIM 256
#define TIME_MAX_PERIOD 10000.0f
#define TIMESTEP_SCALE 1000.0f
#define NORM_EPS 1e-6f
#define MOD_CHUNKS 4
#define ROPE_THETA 10000.0f
#define ROPE_AXES 3
#define ROPE_FRAME_DIM 16
#define TEXT_CTX_TOKENS 1024
#define TEXT_SYSTEM "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
#define TEXT_USER_PREFIX "<|im_start|>user\n"
#define TEXT_USER_SUFFIX "<|im_end|>\n<|im_start|>assistant\n"

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

static const vae_latent_stats qwen_image_vae_stats = {
	.channels = LATENT_CHANNELS,
	.mean	  = qwen_image_latent_mean,
	.std	  = qwen_image_latent_std,
};

static const latent_preview_proj qwen_image_preview_proj = {
	.rgb_proj	= qwen_image_preview_rgb_proj,
	.rgb_bias	= qwen_image_preview_rgb_bias,
	.alpha_proj = qwen_image_preview_alpha_proj,
	.alpha_bias = qwen_image_preview_alpha_bias,
};

typedef struct {
	weight q, k, v, o, gate, up, down;
	buffer norm_q, norm_k;
} dit_layer;

enum { BIND_TIME, BIND_INPUT, BIND_ROPE };
enum { BIND_COND_K, BIND_COND_V };
#define BIND_VELOCITY 0

enum { LP_Q, LP_K, LP_V, LP_O, LP_GATE, LP_UP, LP_DOWN, LP_NORM_Q, LP_NORM_K, LP_COUNT };
#define LP_NONE 0xFF

typedef struct {
	diffusion_model base;
	backend		   *be;
	gguf_ctx		dit;
	char		   *llm_path;
	char		   *device;
	diffusion_vae  *vae;
	int				hidden, heads, head_dim, inter, n_layers, in_channels, out_channels;
	int				rope_axes[ROPE_AXES];
	weight			img_in, modulation, norm_out, proj_out, time_1, time_2, txt_in, txt_out;
	buffer			txt_norm;
	dit_layer	   *layers;
	cg_param	   *layer_params;
	cg_arena		arena;
	cg_plan_cache	step_plans, prefix_plans;
} qwen_image;

struct diffusion_cond {
	backend *be;
	int		 n_tokens;
	buffer	 k, v;
};

enum { LW_MAT, LW_GATE_UP, LW_VEC };

static const struct {
	const char *suffix;
	size_t		off;
	size_t		off2;
	uint8_t		slot;
	uint8_t		slot2;
	uint8_t		kind;
	float		scale;
} qwen_layer_tensors[] = {
	{"attn.to_q.weight", offsetof(dit_layer, q), 0, LP_Q, LP_NONE, LW_MAT, 0.0f},
	{"attn.to_k.weight", offsetof(dit_layer, k), 0, LP_K, LP_NONE, LW_MAT, 0.0f},
	{"attn.to_v.weight", offsetof(dit_layer, v), 0, LP_V, LP_NONE, LW_MAT, 0.0f},
	{"attn.to_out.0.weight", offsetof(dit_layer, o), 0, LP_O, LP_NONE, LW_MAT, 0.0f},
	{"img_mlp.out.weight", offsetof(dit_layer, down), 0, LP_DOWN, LP_NONE, LW_MAT, 0.0f},
	{"img_mlp.gate_up.weight", offsetof(dit_layer, gate), offsetof(dit_layer, up), LP_GATE, LP_UP,
	 LW_GATE_UP, 0.0f},
	{"attn.norm_q.weight", offsetof(dit_layer, norm_q), 0, LP_NORM_Q, LP_NONE, LW_VEC, 0.0f},
	{"attn.norm_k.weight", offsetof(dit_layer, norm_k), 0, LP_NORM_K, LP_NONE, LW_VEC, 0.0f},
};

static status_code load_layer(const qwen_image *m, int li, dit_layer *l) {
	for (size_t i = 0; i < ARRAY_LEN(qwen_layer_tensors); i++) {
		char name[160];
		snprintf(name, sizeof(name), "transformer_blocks.%d.%s", li, qwen_layer_tensors[i].suffix);
		const gguf_tensor *t = gguf_require_tensor(&m->dit, name);
		if (!t)
			return ERR_NOT_FOUND;
		void	   *dst = (char *)l + qwen_layer_tensors[i].off;
		status_code s	= OK;
		switch (qwen_layer_tensors[i].kind) {
		case LW_MAT:
			s = weight_load(m->be, t, dst);
			break;
		case LW_GATE_UP:
			s = weight_load_rows(m->be, t, 0, m->inter, dst);
			if (s == OK)
				s = weight_load_rows(m->be, t, m->inter, m->inter,
									 (weight *)((char *)l + qwen_layer_tensors[i].off2));
			break;
		case LW_VEC:
			s = tensor_upload_f32(m->be, t, 0, 0, qwen_layer_tensors[i].scale, dst);
			break;
		}
		if (s != OK)
			return s;
	}
	return OK;
}

static bool qwen_image_detect(const gguf_ctx *dit) {
	return gguf_find_tensor(dit, "transformer_blocks.0.img_mlp.gate_up.weight") &&
		   gguf_find_tensor(dit, "txt_in.in_layer.weight");
}

static void bind_layer_params(qwen_image *m) {
	m->layer_params = xcalloc((size_t)m->n_layers * LP_COUNT, sizeof(cg_param));
	for (int li = 0; li < m->n_layers; li++) {
		dit_layer *l   = &m->layers[li];
		cg_param  *row = &m->layer_params[(size_t)li * LP_COUNT];
		for (size_t i = 0; i < ARRAY_LEN(qwen_layer_tensors); i++) {
			if (qwen_layer_tensors[i].slot == LP_NONE)
				continue;
			cg_param *p = &row[qwen_layer_tensors[i].slot];
			if (qwen_layer_tensors[i].kind == LW_VEC)
				p->vec = (buffer *)((char *)l + qwen_layer_tensors[i].off);
			else
				p->mat = (weight *)((char *)l + qwen_layer_tensors[i].off);
			if (qwen_layer_tensors[i].slot2 != LP_NONE)
				row[qwen_layer_tensors[i].slot2].mat =
					(weight *)((char *)l + qwen_layer_tensors[i].off2);
		}
	}
}

enum { GW_WEIGHT, GW_VEC };

static const struct {
	const char *name;
	size_t		off;
	uint8_t		kind;
	float		scale;
} qwen_global_tensors[] = {
	{"img_in.weight", offsetof(qwen_image, img_in), GW_WEIGHT, 0.0f},
	{"modulation.1.weight", offsetof(qwen_image, modulation), GW_WEIGHT, 0.0f},
	{"norm_out.linear.weight", offsetof(qwen_image, norm_out), GW_WEIGHT, 0.0f},
	{"proj_out.weight", offsetof(qwen_image, proj_out), GW_WEIGHT, 0.0f},
	{"time_text_embed.timestep_embedder.linear_1.weight", offsetof(qwen_image, time_1), GW_WEIGHT,
	 0.0f},
	{"time_text_embed.timestep_embedder.linear_2.weight", offsetof(qwen_image, time_2), GW_WEIGHT,
	 0.0f},
	{"txt_in.in_layer.weight", offsetof(qwen_image, txt_in), GW_WEIGHT, 0.0f},
	{"txt_in.out_layer.weight", offsetof(qwen_image, txt_out), GW_WEIGHT, 0.0f},
	{"txt_in.text_norm.weight", offsetof(qwen_image, txt_norm), GW_VEC, 1.0f},
};

static status_code load_dit(qwen_image *m) {
	status_code s = OK;
	for (size_t i = 0; i < ARRAY_LEN(qwen_global_tensors) && s == OK; i++) {
		const gguf_tensor *t = gguf_require_tensor(&m->dit, qwen_global_tensors[i].name);
		if (!t)
			return ERR_NOT_FOUND;
		void *dst = (char *)m + qwen_global_tensors[i].off;
		if (qwen_global_tensors[i].kind == GW_WEIGHT)
			s = weight_load(m->be, t, dst);
		else
			s = tensor_upload_f32(m->be, t, 0, 0, qwen_global_tensors[i].scale, dst);
	}
	m->layers = xcalloc((size_t)m->n_layers, sizeof(dit_layer));
	for (int li = 0; li < m->n_layers && s == OK; li++)
		s = load_layer(m, li, &m->layers[li]);
	if (s == OK)
		bind_layer_params(m);
	return s;
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
	m->rope_axes[0] = ROPE_FRAME_DIM;
	m->rope_axes[1] = (m->head_dim - ROPE_FRAME_DIM) / 2;
	m->rope_axes[2] = (m->head_dim - ROPE_FRAME_DIM) / 2;
}

static void qwen_image_close(diffusion_model *dm) {
	qwen_image *m = (qwen_image *)dm;
	cg_plan_cache_free(&m->step_plans);
	cg_plan_cache_free(&m->prefix_plans);
	cg_arena_free(&m->arena);
	free(m->layer_params);
	for (size_t i = 0; i < ARRAY_LEN(qwen_global_tensors); i++) {
		void *p = (char *)m + qwen_global_tensors[i].off;
		if (qwen_global_tensors[i].kind == GW_WEIGHT)
			weight_free(p);
		else
			buffer_release(p);
	}
	for (int li = 0; m->layers && li < m->n_layers; li++) {
		dit_layer *l = &m->layers[li];
		for (size_t i = 0; i < ARRAY_LEN(qwen_layer_tensors); i++) {
			void *p = (char *)l + qwen_layer_tensors[i].off;
			if (qwen_layer_tensors[i].kind == LW_VEC)
				buffer_release(p);
			else
				weight_free(p);
			if (qwen_layer_tensors[i].slot2 != LP_NONE)
				weight_free((weight *)((char *)l + qwen_layer_tensors[i].off2));
		}
	}
	free(m->layers);
	diffusion_vae_close(m->vae);
	gguf_free(&m->dit);
	free(m->llm_path);
	free(m->device);
	if (m->be)
		backend_destroy(m->be);
	free(m);
}

static diffusion_model *qwen_image_open(const diffusion_files *files, gguf_ctx *dit);

static void qwen_image_cond_free(diffusion_cond *c) {
	buffer_release(&c->k);
	buffer_release(&c->v);
	free(c);
}

typedef struct {
	int x, mod, rope, k_out, v_out, prefix_k, prefix_v, prefix_n;
} layer_values;

static int time_modulation_graph(cg_builder *g, const qwen_image *m, int *time_vec) {
	int emb = cg_input(g, TIME_EMBED_DIM, BIND_TIME);
	int mid = cg_matmul(g, cg_param_mat(g, &m->time_1), emb, CG_NEW, m->hidden, TIME_EMBED_DIM, 1);
	mid		= cg_activate(g, mid, ACTIVATION_SILU);
	*time_vec = cg_matmul(g, cg_param_mat(g, &m->time_2), mid, CG_NEW, m->hidden, m->hidden, 1);
	*time_vec = cg_activate(g, *time_vec, ACTIVATION_SILU);
	return cg_matmul(g, cg_param_mat(g, &m->modulation), *time_vec, CG_NEW, MOD_CHUNKS * m->hidden,
					 m->hidden, 1);
}

static void layer_graph(cg_builder *g, const qwen_image *m, const layer_values *lv, int rows) {
	int dim		   = m->hidden;
	int scale_attn = cg_view(g, lv->mod, 0, dim);
	int gate_attn  = cg_view(g, lv->mod, dim, dim);
	int scale_mlp  = cg_view(g, lv->mod, 2 * dim, dim);
	int gate_mlp   = cg_view(g, lv->mod, 3 * dim, dim);

	int h	= cg_layernorm_modulate(g, lv->x, scale_attn, CG_NEW, dim, NORM_EPS, rows);
	int raw = cg_matmul(g, cg_layer_param(LP_Q), h, CG_NEW, dim, dim, rows);
	int q	= cg_rmsnorm_per_head(g, cg_layer_param(LP_NORM_Q), raw, CG_NEW, m->heads, m->head_dim,
								  NORM_EPS, rows);
	raw		= cg_matmul(g, cg_layer_param(LP_K), h, CG_NEW, dim, dim, rows);
	int k = cg_rmsnorm_per_head(g, cg_layer_param(LP_NORM_K), raw, lv->k_out, m->heads, m->head_dim,
								NORM_EPS, rows);
	int v = cg_matmul(g, cg_layer_param(LP_V), h, lv->v_out, dim, dim, rows);
	q	  = cg_rope_table(g, q, lv->rope, rows, m->heads, m->head_dim);
	k	  = cg_rope_table(g, k, lv->rope, rows, m->heads, m->head_dim);

	cg_attention_dims dims = {.n_q = rows, .n_heads = m->heads, .head_dim = m->head_dim};
	int				  attn;
	if (lv->prefix_k != CG_NONE) {
		dims.n0 = lv->prefix_n;
		dims.n1 = rows;
		attn	= cg_attention(g, q, lv->prefix_k, lv->prefix_v, k, v, CG_NEW, dims);
	} else {
		dims.n0		= rows;
		dims.causal = 1;
		attn		= cg_attention(g, q, k, v, CG_NONE, CG_NONE, CG_NEW, dims);
	}
	int o = cg_matmul(g, cg_layer_param(LP_O), attn, CG_NEW, dim, dim, rows);
	int x = cg_gated_add(g, lv->x, o, gate_attn, dim, rows);

	h		 = cg_layernorm_modulate(g, x, scale_mlp, CG_NEW, dim, NORM_EPS, rows);
	int gate = cg_matmul(g, cg_layer_param(LP_GATE), h, CG_NEW, m->inter, dim, rows);
	int up	 = cg_matmul(g, cg_layer_param(LP_UP), h, CG_NEW, m->inter, dim, rows);
	int act	 = cg_ffn_activate(g, gate, up, CG_NEW, m->inter, ACTIVATION_SILU, rows);
	int down = cg_matmul(g, cg_layer_param(LP_DOWN), act, CG_NEW, dim, m->inter, rows);
	cg_gated_add(g, x, down, gate_mlp, dim, rows);
}

static cg_builder *prefix_graph(const qwen_image *m, int rows) {
	cg_builder *g	   = cg_builder_new();
	size_t		floats = (size_t)rows * m->hidden;
	int			time_vec;
	int			mod	 = time_modulation_graph(g, m, &time_vec);
	int			text = cg_input(g, floats, BIND_INPUT);
	int a = cg_rmsnorm(g, cg_param_vec(g, &m->txt_norm), text, CG_NEW, m->hidden, NORM_EPS, rows);
	int b = cg_matmul(g, cg_param_mat(g, &m->txt_in), a, CG_NEW, m->hidden, m->hidden, rows);
	b	  = cg_activate(g, b, ACTIVATION_GELU);
	int x = cg_matmul(g, cg_param_mat(g, &m->txt_out), b, CG_NEW, m->hidden, m->hidden, rows);
	int rope = cg_input(g, (size_t)rows * m->head_dim, BIND_ROPE);

	cg_builder_section(g, CG_SECTION_LAYER);
	cg_builder_layers(g, m->n_layers, m->layer_params, LP_COUNT);
	layer_values lv = {.x		 = x,
					   .mod		 = mod,
					   .rope	 = rope,
					   .k_out	 = cg_layer_external(g, floats, floats, BIND_COND_K),
					   .v_out	 = cg_layer_external(g, floats, floats, BIND_COND_V),
					   .prefix_k = CG_NONE,
					   .prefix_v = CG_NONE};
	layer_graph(g, m, &lv, rows);
	return g;
}

static cg_builder *step_graph(const qwen_image *m, int rows, int n_tokens) {
	cg_builder *g			  = cg_builder_new();
	size_t		prefix_floats = (size_t)n_tokens * m->hidden;
	int			time_vec;
	int			mod	   = time_modulation_graph(g, m, &time_vec);
	int			latent = cg_input(g, (size_t)rows * m->in_channels, BIND_INPUT);
	int			x =
		cg_matmul(g, cg_param_mat(g, &m->img_in), latent, CG_NEW, m->hidden, m->in_channels, rows);
	int rope = cg_input(g, (size_t)rows * m->head_dim, BIND_ROPE);

	cg_builder_section(g, CG_SECTION_LAYER);
	cg_builder_layers(g, m->n_layers, m->layer_params, LP_COUNT);
	layer_values lv = {.x		 = x,
					   .mod		 = mod,
					   .rope	 = rope,
					   .k_out	 = CG_NEW,
					   .v_out	 = CG_NEW,
					   .prefix_k = cg_layer_external(g, prefix_floats, prefix_floats, BIND_COND_K),
					   .prefix_v = cg_layer_external(g, prefix_floats, prefix_floats, BIND_COND_V),
					   .prefix_n = n_tokens};
	layer_graph(g, m, &lv, rows);

	cg_builder_section(g, CG_SECTION_POST);
	int scale =
		cg_matmul(g, cg_param_mat(g, &m->norm_out), time_vec, CG_NEW, m->hidden, m->hidden, 1);
	int h = cg_layernorm_modulate(g, x, scale, CG_NEW, m->hidden, NORM_EPS, rows);
	cg_output_value(
		g, cg_matmul(g, cg_param_mat(g, &m->proj_out), h, CG_NEW, m->out_channels, m->hidden, rows),
		BIND_VELOCITY);
	return g;
}

static status_code prefix_plan(qwen_image *m, int rows, cg_plan **out) {
	uint64_t key = cg_plan_key(rows, 0);
	*out		 = cg_plan_cache_get(&m->prefix_plans, key);
	if (*out)
		return OK;
	status_code s = cg_plan_build(prefix_graph(m, rows), m->be, "qwen-image prefix", out);
	if (s == OK)
		cg_plan_cache_put(&m->prefix_plans, key, *out);
	return s;
}

static status_code step_plan(qwen_image *m, int rows, int n_tokens, cg_plan **out) {
	uint64_t key = cg_plan_key(rows, n_tokens);
	*out		 = cg_plan_cache_get(&m->step_plans, key);
	if (*out)
		return OK;
	status_code s = cg_plan_build(step_graph(m, rows, n_tokens), m->be, "qwen-image step", out);
	if (s == OK)
		cg_plan_cache_put(&m->step_plans, key, *out);
	return s;
}

static float *rope_from_positions(const qwen_image *m, int *pos, int rows) {
	int	   half	 = m->head_dim / 2;
	float *table = xmalloc((size_t)rows * m->head_dim * sizeof(float));
	for (int r = 0; r < rows; r++) {
		float *cs  = table + (size_t)r * half * 2;
		int	   idx = 0;
		for (int a = 0; a < ROPE_AXES; a++) {
			int	   axis_half   = m->rope_axes[a] / 2;
			double theta_scale = pow(ROPE_THETA, -2.0 / (double)m->rope_axes[a]);
			rope_cs_fill(cs + idx * 2, axis_half, pos[r * ROPE_AXES + a], theta_scale, NULL);
			idx += axis_half;
		}
	}
	free(pos);
	return table;
}

static float *text_rope(const qwen_image *m, int rows) {
	int *pos = xmalloc((size_t)rows * ROPE_AXES * sizeof(int));
	for (int i = 0; i < rows; i++)
		pos[i * ROPE_AXES] = pos[i * ROPE_AXES + 1] = pos[i * ROPE_AXES + 2] = i;
	return rope_from_positions(m, pos, rows);
}

static float *image_rope(const qwen_image *m, latent_shape shape, int n_tokens) {
	int	 rows = shape.lw * shape.lh;
	int *pos  = xmalloc((size_t)rows * ROPE_AXES * sizeof(int));
	for (int i = 0; i < rows; i++) {
		pos[i * ROPE_AXES]	   = n_tokens;
		pos[i * ROPE_AXES + 1] = i / shape.lw - (shape.lh - shape.lh / 2);
		pos[i * ROPE_AXES + 2] = i % shape.lw - (shape.lw - shape.lw / 2);
	}
	return rope_from_positions(m, pos, rows);
}

static status_code build_prefix(qwen_image *m, const float *hidden, int rows,
								diffusion_cond **out) {
	diffusion_cond *c  = xcalloc(1, sizeof(*c));
	c->be			   = m->be;
	c->n_tokens		   = rows;
	size_t		floats = (size_t)rows * m->hidden * m->n_layers;
	cg_plan	   *plan;
	status_code s = buffer_alloc_f32(m->be, floats, &c->k);
	if (s == OK)
		s = buffer_alloc_f32(m->be, floats, &c->v);
	if (s == OK)
		s = prefix_plan(m, rows, &plan);
	if (s == OK) {
		float emb[TIME_EMBED_DIM];
		dit_timestep_embedding(emb, TIME_EMBED_DIM, 0.0f, TIME_MAX_PERIOD);
		float		*rope = text_rope(m, rows);
		cg_exec_args args = {
			.host_in   = {[BIND_TIME] = emb, [BIND_INPUT] = hidden, [BIND_ROPE] = rope},
			.layer_ext = {[BIND_COND_K] = c->k, [BIND_COND_V] = c->v}};
		s = cg_graph_run(m->be, &m->arena, plan, &args);
		free(rope);
	}
	if (s == OK)
		*out = c;
	else
		qwen_image_cond_free(c);
	return s;
}

static status_code qwen_image_encode(diffusion_model *dm, const char *const *texts, int n,
									 diffusion_cond **out) {
	qwen_image		 *m			   = (qwen_image *)dm;
	text_encoder_spec spec		   = {.llm_path	   = m->llm_path,
									  .device	   = m->device,
									  .system	   = TEXT_SYSTEM,
									  .user_prefix = TEXT_USER_PREFIX,
									  .user_suffix = TEXT_USER_SUFFIX,
									  .ctx_tokens  = TEXT_CTX_TOKENS};
	text_encoding	  encodings[2] = {0};
	status_code		  s			   = text_encoder_run(&spec, texts, n, encodings);
	for (int i = 0; i < n; i++) {
		if (s == OK)
			s = build_prefix(m, encodings[i].hidden, encodings[i].rows, &out[i]);
		text_encoding_free(&encodings[i]);
	}
	return s;
}

static latent_shape qwen_image_shape(int width, int height) {
	return (latent_shape){width / LATENT_PATCH, height / LATENT_PATCH, LATENT_CHANNELS};
}

static status_code qwen_image_predict(diffusion_model *dm, const float *latent, latent_shape shape,
									  float sigma, const diffusion_cond *cond, float *velocity) {
	qwen_image *m = (qwen_image *)dm;
	cg_plan	   *plan;
	status_code s = step_plan(m, shape.lw * shape.lh, cond->n_tokens, &plan);
	if (s != OK)
		return s;
	float emb[TIME_EMBED_DIM];
	dit_timestep_embedding(emb, TIME_EMBED_DIM, sigma * TIMESTEP_SCALE, TIME_MAX_PERIOD);
	float		*rope = image_rope(m, shape, cond->n_tokens);
	cg_exec_args args = {.host_in  = {[BIND_TIME] = emb, [BIND_INPUT] = latent, [BIND_ROPE] = rope},
						 .host_out = {[BIND_VELOCITY] = velocity},
						 .layer_ext = {[BIND_COND_K] = cond->k, [BIND_COND_V] = cond->v}};
	s				  = cg_graph_run(m->be, &m->arena, plan, &args);
	free(rope);
	return s;
}

static status_code qwen_image_decode(diffusion_model *dm, const float *latent, latent_shape shape,
									 diffusion_image *out) {
	qwen_image *m = (qwen_image *)dm;
	return diffusion_vae_decode(m->vae, &m->arena, latent, shape, out);
}

static void qwen_image_preview(const float *x0, latent_shape shape, diffusion_image *out) {
	diffusion_preview_linear(&qwen_image_preview_proj, x0, shape, out);
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
	m->device	  = xstrdup(files->device);
	if (cg_backend_open(m->device, &m->be) != OK) {
		qwen_image_close(&m->base);
		return NULL;
	}
	detect_config(m);
	INFO("qwen-image: %d layers, hidden %d, %d heads", m->n_layers, m->hidden, m->heads);
	if (load_dit(m) != OK) {
		qwen_image_close(&m->base);
		return NULL;
	}
	m->vae = diffusion_vae_open(m->be, files->vae, &qwen_image_vae_stats);
	if (!m->vae) {
		qwen_image_close(&m->base);
		return NULL;
	}
	return &m->base;
}

DIFFUSION_REGISTER(qwen_image, &qwen_image_ops)
