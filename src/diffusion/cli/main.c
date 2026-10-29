#include "diffusion/runner.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

static void usage(const char *prog) {
	fprintf(stderr,
			"usage: %s --diffusion-model PATH --vae PATH --llm PATH [-p PROMPT] [options]\n"
			"  without -p an interactive session starts, each prompt generates a new image\n"
			"  -n, --negative-prompt TEXT\n"
			"  -W, --width N             (default 512, multiple of %d)\n"
			"  -H, --height N            (default 512, multiple of %d)\n"
			"      --steps N             (default 20)\n"
			"      --cfg-scale F         (default 6.0)\n"
			"      --seed N              (default 0 = random)\n"
			"      --device SPEC         (default cpu0, e.g. vulkan0)\n"
			"  -o, --out PATH            exact output file, only with -p\n"
			"      --out-dir DIR         (default ., images are named automatically)\n"
			"      --out-preview PATH\n"
			"      --preview-interval S  (default 1.0)\n"
			"      --debug\n",
			prog, DIFFUSION_DIMENSION_STEP, DIFFUSION_DIMENSION_STEP);
}

enum {
	OPT_DIFFUSION_MODEL = 256,
	OPT_VAE,
	OPT_LLM,
	OPT_STEPS,
	OPT_CFG,
	OPT_SEED,
	OPT_DEVICE,
	OPT_OUT_DIR,
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
	{"device", required_argument, 0, OPT_DEVICE},
	{"out", required_argument, 0, 'o'},
	{"out-dir", required_argument, 0, OPT_OUT_DIR},
	{"out-preview", required_argument, 0, OPT_PREVIEW},
	{"preview-interval", required_argument, 0, OPT_PREVIEW_INTERVAL},
	{"debug", no_argument, 0, OPT_DEBUG},
	{"help", no_argument, 0, 'h'},
	{0, 0, 0, 0},
};

static bool parse_args(int argc, char **argv, diffusion_files *files, diffusion_params *p,
					   diffusion_output *out, bool *debug) {
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
		case OPT_DEVICE:
			files->device = optarg;
			break;
		case 'o':
			out->path = optarg;
			break;
		case OPT_OUT_DIR:
			out->dir = optarg;
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
	return files->diffusion_model != NULL;
}

int main(int argc, char **argv) {
	diffusion_files	 files = {.device = "cpu0"};
	diffusion_params p	   = diffusion_params_defaults();
	diffusion_output out   = {.dir = ".", .name = "image"};
	bool			 debug = false;

	if (!parse_args(argc, argv, &files, &p, &out, &debug)) {
		usage(argv[0]);
		return 1;
	}
	return diffusion_run(&files, &p, &out, debug);
}
