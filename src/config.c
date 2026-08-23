#include "config.h"
#include "backend/backend.h"
#include "log.h"

#include <ctype.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>

static config g_cfg;
static int	  g_cfg_initialized = 0;

config config_defaults(void) {
	config c;
	memset(&c, 0, sizeof(c));

	c.backend		= "auto";
	c.use_mmap		= true;
	c.flash_attn	= true;
	c.reasoning		= true;
	c.moe_stream	= true;
	c.moe_cache_cap = 0;
	c.ngl			= -1;

	return c;
}

void config_init(const config *cfg) {
	if (g_cfg_initialized) {
		return;
	}
	g_cfg			  = cfg ? *cfg : config_defaults();
	g_cfg_initialized = 1;

	if (g_cfg.debug) {
		log_set_level(LOG_DEBUG);
	}
}

const config *config_get(void) {
	if (!g_cfg_initialized) {
		g_cfg			  = config_defaults();
		g_cfg_initialized = 1;
	}
	return &g_cfg;
}

void usage(FILE *fp) {
	fprintf(fp,
			"Usage: kappai-cli --model <path> [options] [prompt]\n\n"
			"Runs an interactive chat REPL by default. Use -p for a single\n"
			"non-interactive turn (scripting, testing, experimentation).\n\n"
			"Engine:\n"
			"  -m, --model <path>       model file (required)\n"
			"  -c, --ctx-size <n>       context size (0 = from model)\n"
			"  --accel <name>           accelerator backend (default: auto)\n"
			"  --gpu-device <n>         GPU device index\n"
			"  --ngl <n>                offload the first <n> layers to the --accel backend\n"
			"                           and keep the remaining layers on CPU (-1 = all on accel,\n"
			"                           0 = all on CPU; default: -1)\n"
			"  --threads <n>            CPU worker threads (default: auto)\n"
			"  -f, --flash-attn [on|off]  flash attention kernel (default: on)\n"
			"  --kv-quant [f16|q8_0]    KV cache precision (default: f16)\n"
			"  --mmap [on|off]          mmap model weights (default: on)\n"
			"  --repack MODE            weight repacking: all|none|<types> (default: smart)\n"
			"  --fuse LIST              load-time weight fusion, comma list: qkv\n"
			"  --moe-stream [on|off]    streaming expert cache for MoE models (default: on)\n"
			"  --moe-cache <n>          per-layer LRU capacity for expert cache\n"
			"  --moe-preload            eagerly load all expert weights at startup\n"
			"  --moe-pin <n>            pin first n experts per layer (never evicted)\n"
			"  --moe-pin-list <ids>     pin specific expert ids, e.g. \"3,17,42\"\n"
			"  --stream [on|off]        stream partial output as it's generated (default: on)\n\n"
			"LLM behavior:\n"
			"  -p, --prompt <text>      one-shot prompt (or last positional arg)\n"
			"  --system <text>          system prompt\n"
			"  -n, --n-predict <n>      max tokens to generate (-1 = unlimited)\n"
			"  -t, --temperature <f>    sampling temperature (default 0.80, 0.0 = greedy)\n"
			"  -k, --top-k <n>          top-k sampling (default 40, 0 = disabled)\n"
			"  -P, --top-p <f>          top-p sampling (default 0.90, 1.0 = disabled)\n"
			"  -M, --min-p <f>          min-p sampling (default 0.10, 0.0 = disabled)\n"
			"  -s, --seed <n>           RNG seed (0 = random)\n"
			"  --reasoning [on|off]     allow model reasoning/thinking (default: on)\n\n"
			"Metrics / startup:\n"
			"  --metrics <list>         comma-separated metrics to print after generation,\n"
			"                           in any order: pp,tg,ttft (default: pp,tg)\n"
			"  --warmup [on|off]        pre-process the static prefix of the chat template\n"
			"                           at startup (default: off)\n\n"
			"Tools / debug:\n"
			"  --list-accels            list available accelerator backends and exit\n"
			"  --dump-metadata          print raw GGUF metadata and exit\n"
			"  --show-template          print the rendered chat template before prefill\n"
			"  --grep-vocab <substr>    print vocab entries containing <substr> and exit\n"
			"  --monitor [path]         enable monitoring socket (default: /tmp/kappai.monitor)\n"
			"  --time                   print per-stage timing breakdown\n"
			"  --debug-forward          print per-layer activation stats for first token\n"
			"  --debug                  enable debug-level logging\n"
			"  --disable-failsafes      skip memory guardrails (unsafe; debug only)\n"
			"  -h, --help               show this help\n\n");
}

static int parse_bool_flag(const char *optarg, int default_if_bare) {
	if (!optarg)
		return default_if_bare;
	if (!strcmp(optarg, "off") || !strcmp(optarg, "0") || !strcmp(optarg, "false") ||
		!strcmp(optarg, "disabled"))
		return 0;
	return 1;
}

static const char *peek_optional_bool_arg(const char *optarg, int argc, char **argv, int *optind) {
	if (optarg)
		return optarg;
	if (*optind >= argc)
		return NULL;
	const char *candidate = argv[*optind];
	if (!strcmp(candidate, "on") || !strcmp(candidate, "off") || !strcmp(candidate, "0") ||
		!strcmp(candidate, "1") || !strcmp(candidate, "true") || !strcmp(candidate, "false") ||
		!strcmp(candidate, "disabled")) {
		(*optind)++;
		return candidate;
	}
	return NULL;
}

static void parse_kv_quant(const char *optarg, config *cfg) {
	if (strcmp(optarg, "f16") == 0) {
		cfg->kv_quant = KV_QUANT_F16;
	} else if (strcmp(optarg, "q8_0") == 0) {
		cfg->kv_quant = KV_QUANT_Q8_0;
	} else {
		fprintf(stderr, "invalid --kv-quant value '%s' (expected f16 or q8_0)\n", optarg);
		exit(1);
	}
}

static void parse_monitor(const char *optarg, int argc, char **argv, int *optind, config *cfg) {
	(void)argc;
	if (optarg) {
		cfg->monitor = optarg;
	} else if (*optind < argc && argv[*optind][0] != '-') {
		cfg->monitor = argv[*optind];
		(*optind)++;
	} else {
		cfg->monitor = "";
	}
}

int parse_args(int argc, char **argv, config *cfg, cli_args *a) {
	*cfg = config_defaults();
	memset(a, 0, sizeof(*a));

	a->n_predict	 = -1;
	a->temperature	 = 0.80f;
	a->top_k		 = 40;
	a->top_p		 = 0.90f;
	a->min_p		 = 0.10f;
	a->output_stream = true;
	snprintf(a->metrics, sizeof(a->metrics), "pp,tg");
	a->warmup = false;

	setenv("POSIXLY_CORRECT", "1", 1);

	enum {
		OPT_SYSTEM = 256,
		OPT_STREAM,
		OPT_TIME,
		OPT_ACCEL,
		OPT_GPU_DEV,
		OPT_LIST_ACCELS,
		OPT_DUMP_METADATA,
		OPT_DEBUG_FWD,
		OPT_DEBUG,
		OPT_THREADS,
		OPT_MOE_CACHE,
		OPT_MOE_STREAM,
		OPT_MOE_PRELOAD,
		OPT_MOE_PIN,
		OPT_MOE_PIN_LIST,
		OPT_MONITOR,
		OPT_MMAP,
		OPT_REPACK,
		OPT_FUSE,
		OPT_REASONING,
		OPT_DISABLE_FAILSAFES,
		OPT_SHOW_TEMPLATE,
		OPT_GREP_VOCAB,
		OPT_KV_QUANT,
		OPT_NGL,
		OPT_METRICS,
		OPT_WARMUP
	};

	static struct option long_opts[] = {
		{"model", required_argument, NULL, 'm'},
		{"prompt", required_argument, NULL, 'p'},
		{"n-predict", required_argument, NULL, 'n'},
		{"temperature", required_argument, NULL, 't'},
		{"top-k", required_argument, NULL, 'k'},
		{"top-p", required_argument, NULL, 'P'},
		{"min-p", required_argument, NULL, 'M'},
		{"ctx-size", required_argument, NULL, 'c'},
		{"seed", required_argument, NULL, 's'},
		{"flash-attn", required_argument, NULL, 'f'},
		{"help", no_argument, NULL, 'h'},
		{"system", required_argument, NULL, OPT_SYSTEM},
		{"stream", optional_argument, NULL, OPT_STREAM},
		{"time", no_argument, NULL, OPT_TIME},
		{"accel", required_argument, NULL, OPT_ACCEL},
		{"gpu-device", required_argument, NULL, OPT_GPU_DEV},
		{"list-accels", no_argument, NULL, OPT_LIST_ACCELS},
		{"dump-metadata", no_argument, NULL, OPT_DUMP_METADATA},
		{"debug-forward", no_argument, NULL, OPT_DEBUG_FWD},
		{"debug", no_argument, NULL, OPT_DEBUG},
		{"threads", required_argument, NULL, OPT_THREADS},
		{"moe-cache", required_argument, NULL, OPT_MOE_CACHE},
		{"moe-stream", optional_argument, NULL, OPT_MOE_STREAM},
		{"moe-preload", no_argument, NULL, OPT_MOE_PRELOAD},
		{"moe-pin", required_argument, NULL, OPT_MOE_PIN},
		{"moe-pin-list", required_argument, NULL, OPT_MOE_PIN_LIST},
		{"monitor", optional_argument, NULL, OPT_MONITOR},
		{"mmap", optional_argument, NULL, OPT_MMAP},
		{"repack", required_argument, NULL, OPT_REPACK},
		{"fuse", required_argument, NULL, OPT_FUSE},
		{"reasoning", optional_argument, NULL, OPT_REASONING},
		{"disable-failsafes", no_argument, NULL, OPT_DISABLE_FAILSAFES},
		{"show-template", no_argument, NULL, OPT_SHOW_TEMPLATE},
		{"grep-vocab", required_argument, NULL, OPT_GREP_VOCAB},
		{"kv-quant", required_argument, NULL, OPT_KV_QUANT},
		{"ngl", required_argument, NULL, OPT_NGL},
		{"metrics", required_argument, NULL, OPT_METRICS},
		{"warmup", optional_argument, NULL, OPT_WARMUP},
		{0, 0, 0, 0}};

	int c;
	while ((c = getopt_long(argc, argv, "m:p:n:t:k:P:M:c:s:f:h", long_opts, NULL)) != -1) {
		switch (c) {
		case 'm':
			cfg->model = optarg;
			break;
		case 'p':
			a->prompt = optarg;
			break;
		case 'n':
			a->n_predict = atoi(optarg);
			break;
		case 't':
			a->temperature = (float)atof(optarg);
			break;
		case 'k':
			a->top_k = atoi(optarg);
			break;
		case 'P':
			a->top_p = (float)atof(optarg);
			break;
		case 'M':
			a->min_p = (float)atof(optarg);
			break;
		case 'c':
			cfg->ctx_size = atoi(optarg);
			break;
		case 's':
			cfg->seed = (uint64_t)strtoull(optarg, NULL, 10);
			break;
		case 'f':
			cfg->flash_attn = parse_bool_flag(optarg, 1);
			break;
		case 'h':
			usage(stdout);
			exit(0);
		case OPT_SYSTEM:
			a->system = optarg;
			break;
		case OPT_STREAM:
			a->output_stream =
				parse_bool_flag(peek_optional_bool_arg(optarg, argc, argv, &optind), 1);
			break;
		case OPT_TIME:
			cfg->profile_time = true;
			break;
		case OPT_ACCEL:
			cfg->backend = optarg;
			break;
		case OPT_GPU_DEV:
			cfg->gpu_device = atoi(optarg);
			break;
		case OPT_LIST_ACCELS:
			a->list_accels = true;
			break;
		case OPT_DUMP_METADATA:
			a->dump_metadata = true;
			break;
		case OPT_DEBUG_FWD:
			cfg->debug_forward = true;
			break;
		case OPT_SHOW_TEMPLATE:
			a->show_template = true;
			break;
		case OPT_GREP_VOCAB:
			a->grep_vocab = optarg;
			break;
		case OPT_KV_QUANT:
			parse_kv_quant(optarg, cfg);
			break;
		case OPT_NGL:
			cfg->ngl = atoi(optarg);
			break;
		case OPT_DEBUG:
			cfg->debug = 1;
			break;
		case OPT_THREADS:
			cfg->n_threads = atoi(optarg);
			if (cfg->n_threads < 0)
				cfg->n_threads = 0;
			break;
		case OPT_MOE_CACHE:
			cfg->moe_cache_cap = atoi(optarg);
			break;
		case OPT_MOE_STREAM:
			cfg->moe_stream =
				parse_bool_flag(peek_optional_bool_arg(optarg, argc, argv, &optind), 1);
			break;
		case OPT_MOE_PRELOAD:
			cfg->moe_preload = 1;
			break;
		case OPT_MOE_PIN:
			cfg->moe_pin = atoi(optarg);
			if (cfg->moe_pin < 0)
				cfg->moe_pin = 0;
			break;
		case OPT_MOE_PIN_LIST:
			cfg->moe_pin_list = optarg;
			break;
		case OPT_MONITOR:
			parse_monitor(optarg, argc, argv, &optind, cfg);
			break;
		case OPT_MMAP:
			cfg->use_mmap = parse_bool_flag(peek_optional_bool_arg(optarg, argc, argv, &optind), 1);
			break;
		case OPT_REPACK:
			cfg->repack = optarg;
			break;
		case OPT_FUSE:
			cfg->fuse = optarg;
			break;
		case OPT_REASONING:
			cfg->reasoning =
				parse_bool_flag(peek_optional_bool_arg(optarg, argc, argv, &optind), 1);
			break;
		case OPT_DISABLE_FAILSAFES:
			cfg->disable_failsafes = 1;
			break;
		case OPT_METRICS: {
			size_t l = strlen(optarg);
			if (l == 0 || l >= sizeof(a->metrics)) {
				fprintf(stderr, "invalid --metrics value '%s' (max %zu chars)\n", optarg,
						sizeof(a->metrics) - 1);
				return -1;
			}
			for (size_t i = 0; i < l; i++) {
				char ch = optarg[i];
				if (ch == ',' || ch == ' ' || (ch >= 'a' && ch <= 'z') ||
					(ch >= 'A' && ch <= 'Z')) {
					a->metrics[i] = (char)tolower((unsigned char)ch);
				} else {
					fprintf(stderr, "invalid character '%c' in --metrics value '%s'\n", ch, optarg);
					return -1;
				}
			}
			a->metrics[l] = '\0';
			break;
		}
		case OPT_WARMUP:
			a->warmup = parse_bool_flag(peek_optional_bool_arg(optarg, argc, argv, &optind), 1);
			break;
		case '?':
			return -1;
		default:
			break;
		}
	}

	if (optind < argc && !a->prompt)
		a->prompt = argv[optind];
	if (!a->prompt)
		a->interactive = true;
	if (a->list_accels)
		return 0;
	if (!cfg->model) {
		ERROR("--model is required");
		return -1;
	}
	return 0;
}
