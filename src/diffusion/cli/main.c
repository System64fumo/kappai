#include "diffusion/diffusion.h"
#include "log.h"

#include <getopt.h>
#include <stdio.h>

#define MIN_DIMENSION_STEP 32

static void usage(const char *prog) {
	fprintf(stderr,
			"usage: %s --diffusion-model PATH --vae PATH --llm PATH -p PROMPT [options]\n"
			"  -n, --negative-prompt TEXT\n"
			"  -W, --width N             (default 512, multiple of %d)\n"
			"  -H, --height N            (default 512, multiple of %d)\n"
			"      --steps N             (default 20)\n"
			"      --cfg-scale F         (default 6.0)\n"
			"      --seed N              (default 0)\n"
			"  -o, --out PATH            (default image.png)\n"
			"      --out-preview PATH\n"
			"      --preview-interval S  (default 1.0)\n"
			"      --debug\n",
			prog, MIN_DIMENSION_STEP, MIN_DIMENSION_STEP);
}

enum {
	OPT_DIFFUSION_MODEL = 256,
	OPT_VAE,
	OPT_LLM,
	OPT_STEPS,
	OPT_CFG,
	OPT_SEED,
	OPT_PREVIEW,
	OPT_PREVIEW_INTERVAL,
	OPT_DEBUG,
};

static const struct option long_opts[] = {
	{"diffusion-model", required_argument, 0, OPT_DIFFUSION_MODEL},
	{"vae", required_argument, 0, OPT_VAE},
	{"llm", required_argument, 0, OPT_LLM},
	{"prompt", required_argument, 0, 'p'},
	{"negative-prompt", required_argument, 0, 'n'},
	{"width", required_argument, 0, 'W'},
	{"height", required_argument, 0, 'H'},
	{"steps", required_argument, 0, OPT_STEPS},
	{"cfg-scale", required_argument, 0, OPT_CFG},
	{"seed", required_argument, 0, OPT_SEED},
	{"out", required_argument, 0, 'o'},
	{"out-preview", required_argument, 0, OPT_PREVIEW},
	{"preview-interval", required_argument, 0, OPT_PREVIEW_INTERVAL},
	{"debug", no_argument, 0, OPT_DEBUG},
	{"help", no_argument, 0, 'h'},
	{0, 0, 0, 0},
};

static bool parse_args(int argc, char **argv, diffusion_files *files, diffusion_params *p,
					   const char **out_path, bool *debug) {
	int opt;
	while ((opt = getopt_long(argc, argv, "p:n:W:H:o:h", long_opts, NULL)) != -1) {
		switch (opt) {
		case OPT_DIFFUSION_MODEL:
			files->diffusion_model = optarg;
			break;
		case OPT_VAE:
			files->vae = optarg;
			break;
		case OPT_LLM:
			files->llm = optarg;
			break;
		case 'p':
			p->prompt = optarg;
			break;
		case 'n':
			p->negative_prompt = optarg;
			break;
		case 'W':
			p->width = atoi(optarg);
			break;
		case 'H':
			p->height = atoi(optarg);
			break;
		case OPT_STEPS:
			p->steps = atoi(optarg);
			break;
		case OPT_CFG:
			p->cfg_scale = (float)atof(optarg);
			break;
		case OPT_SEED:
			p->seed = strtoull(optarg, NULL, 10);
			break;
		case 'o':
			*out_path = optarg;
			break;
		case OPT_PREVIEW:
			p->preview_path = optarg;
			break;
		case OPT_PREVIEW_INTERVAL:
			p->preview_interval = (float)atof(optarg);
			break;
		case OPT_DEBUG:
			*debug = true;
			break;
		default:
			return false;
		}
	}
	return files->diffusion_model && files->vae && files->llm && p->prompt;
}

int main(int argc, char **argv) {
	diffusion_files	 files = {0};
	diffusion_params p	   = {
		.prompt			  = NULL,
		.negative_prompt  = "",
		.width			  = 512,
		.height			  = 512,
		.steps			  = 20,
		.cfg_scale		  = 6.0f,
		.preview_interval = 1.0f,
	};
	const char *out_path = "image.png";
	bool		debug	 = false;

	if (!parse_args(argc, argv, &files, &p, &out_path, &debug)) {
		usage(argv[0]);
		return 1;
	}
	if (p.width <= 0 || p.height <= 0 || p.width % MIN_DIMENSION_STEP ||
		p.height % MIN_DIMENSION_STEP || p.steps < 1) {
		fprintf(stderr, "width and height must be multiples of %d and steps at least 1\n",
				MIN_DIMENSION_STEP);
		return 1;
	}

	log_init(log_default_config());
	if (debug)
		log_set_level(LOG_DEBUG);

	char			 err[256] = {0};
	diffusion_model *dm		  = diffusion_open(&files, err, sizeof(err));
	if (!dm) {
		ERROR("%s", err);
		return 1;
	}

	diffusion_image img = {0};
	status_code		s	= diffusion_generate(dm, &p, &img);
	diffusion_close(dm);
	if (s != OK) {
		ERROR("generation failed: %s", status_str(s));
		return 1;
	}

	s = diffusion_image_write_png(out_path, &img);
	diffusion_image_free(&img);
	return s == OK ? 0 : 1;
}
