#include "config.h"
#include "backend/backend.h"
#include "log.h"

#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static config g_cfg;
static int	  g_cfg_initialized = 0;

config config_defaults(void) {
	config c;
	memset(&c, 0, sizeof(c));

	c.device		= "auto";
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

void usage(FILE *fp, bool is_server) {
	fprintf(fp,
			"Usage: %s --model <path> [options]%s\n\n"
			"%s",
			is_server ? "kappai-server" : "kappai-cli", is_server ? "" : " [prompt]",
			is_server ? "Starts an OpenAI-compatible HTTP server.\n\n"
					  : "Runs an interactive chat REPL by default. Use -p for a single\n"
						"non-interactive turn (scripting, testing, experimentation).\n\n");

	fprintf(fp,
			"Engine:\n"
			"  -m, --model <path>       model file (required)\n"
			"  -c, --ctx-size <n>       context size (0 = from model)\n"
			"  --device <spec>          compute device (default: auto)\n"
			"                           <backend><n>, e.g. vulkan0, vulkan1, cpu0;\n"
			"                           a bare backend name means device 0\n"
			"  --ngl <n>                offload the first <n> layers to the --device backend\n"
			"                           and keep the remaining layers on CPU (-1 = all on the\n"
			"                           device, 0 = all on CPU; default: -1)\n"
			"  --threads <n>            CPU worker threads (default: auto)\n"
			"  -f, --flash-attn [bool]  flash attention kernel (default: on)\n"
			"  --kv-quant [f16|q8_0]    KV cache precision (default: f16)\n"
			"  --mmap [bool]            mmap model weights (default: on)\n"
			"  --repack MODE            weight repacking: all|none|<types> (default: smart)\n"
			"  --fuse LIST              load-time weight fusion, comma list: qkv\n"
			"  --moe-stream [bool]      streaming expert cache for MoE models (default: on)\n"
			"  --moe-cache <n>          per-layer LRU capacity for expert cache\n"
			"  --moe-preload            eagerly load all expert weights at startup\n"
			"  --moe-pin <n>            pin first n experts per layer (never evicted)\n"
			"  --moe-pin-list <ids>     pin specific expert ids, e.g. \"3,17,42\"\n"
			"  --stream [bool]          stream partial output as it's generated (default: on)\n"
			"  --metrics <list>         comma-separated metrics to print after generation,\n"
			"                           in any order: pp,tg,ttft (default: pp,tg)\n"
			"  --warmup [bool]          pre-process the static prefix of the chat template\n"
			"                           at startup (default: on)\n\n"
			"LLM behavior:\n"
			"  -p, --prompt <text>      one-shot prompt (or last positional arg)\n"
			"  --system <text>          system prompt\n"
			"  -n, --n-predict <n>      max tokens to generate (-1 = unlimited)\n"
			"  -t, --temperature <f>    sampling temperature (default 0.80, 0.0 = greedy)\n"
			"  -k, --top-k <n>          top-k sampling (default 40, 0 = disabled)\n"
			"  -P, --top-p <f>          top-p sampling (default 0.90, 1.0 = disabled)\n"
			"  -M, --min-p <f>          min-p sampling (default 0.10, 0.0 = disabled)\n"
			"  --repeat-penalty <f>     repetition penalty (default 1.0 = disabled)\n"
			"  --repeat-last-n <n>      last-n tokens considered by repeat penalty (default 64)\n"
			"  -s, --seed <n>           RNG seed (0 = random)\n"
			"  --reasoning [bool]       allow model reasoning/thinking (default: on)\n\n");

	if (is_server) {
		fprintf(fp, "Server:\n"
					"  --host <addr>            bind address (default: 127.0.0.1)\n"
					"  --port <n>               listen port (default: 8080)\n"
					"  --api-key <key>          require 'Authorization: Bearer <key>' on /v1 "
					"requests\n\n");
	}

	fprintf(fp,
			"Tools / debug:\n"
			"  --list-devices           list available compute devices and exit\n"
			"  --dump-metadata          print raw GGUF metadata and exit\n"
			"  --debug-prompt           print exact tokens/text fed to the model for every "
			"prefill (chat turns, idle prefill, warmup, cache resync)\n"
			"  --grep-vocab <substr>    print vocab entries containing <substr> and exit\n"
			"  --monitor [path]         enable monitoring socket (default: /tmp/kappai.monitor)\n"
			"  --time                   print per-stage timing breakdown\n"
			"  --debug-forward          print per-layer activation stats for first token\n"
			"  --debug                  enable debug-level logging\n"
			"  --disable-failsafes      skip memory guardrails (unsafe; debug only)\n"
			"  -h, --help               show this help\n\n");
}

static int parse_bool_flag(const char *flag, const char *optarg, int default_if_bare, int *out) {
	if (!optarg) {
		*out = default_if_bare;
		return 0;
	}
	if (!strcmp(optarg, "yes") || !strcmp(optarg, "on") || !strcmp(optarg, "true") ||
		!strcmp(optarg, "1")) {
		*out = 1;
		return 0;
	}
	if (!strcmp(optarg, "no") || !strcmp(optarg, "off") || !strcmp(optarg, "false") ||
		!strcmp(optarg, "0")) {
		*out = 0;
		return 0;
	}
	fprintf(stderr,
			"invalid %s value '%s' (expected one of: yes, on, true, 1, no, off, false, 0)\n", flag,
			optarg);
	return -1;
}

static int parse_int_arg(const char *optarg, const char *flag, long minv, long maxv, int *out) {
	char *end = NULL;
	errno	  = 0;
	long v	  = strtol(optarg, &end, 10);
	if (end == optarg || *end != '\0') {
		fprintf(stderr, "invalid %s value '%s' (expected an integer)\n", flag, optarg);
		return -1;
	}
	if (errno == ERANGE || v < minv || v > maxv) {
		fprintf(stderr, "%s value '%s' out of range [%ld, %ld]\n", flag, optarg, minv, maxv);
		return -1;
	}
	*out = (int)v;
	return 0;
}

static int parse_seed_arg(const char *optarg, const char *flag, uint64_t *out) {
	char *end  = NULL;
	errno	   = 0;
	uint64_t v = strtoull(optarg, &end, 10);
	if (end == optarg || *end != '\0') {
		fprintf(stderr, "invalid %s value '%s' (expected an unsigned integer)\n", flag, optarg);
		return -1;
	}
	if (errno == ERANGE) {
		fprintf(stderr, "%s value '%s' out of range [0, %llu]\n", flag, optarg,
				(unsigned long long)UINT64_MAX);
		return -1;
	}
	*out = v;
	return 0;
}

static int parse_float_arg(const char *optarg, const char *flag, float minv, float maxv,
						   float *out) {
	char *end = NULL;
	errno	  = 0;
	double v  = strtod(optarg, &end);
	if (end == optarg || *end != '\0') {
		fprintf(stderr, "invalid %s value '%s' (expected a number)\n", flag, optarg);
		return -1;
	}
	if (!isfinite(v)) {
		fprintf(stderr, "invalid %s value '%s' (must be finite)\n", flag, optarg);
		return -1;
	}
	if (errno == ERANGE || (double)v < (double)minv || (double)v > (double)maxv) {
		fprintf(stderr, "%s value '%s' out of range [%g, %g]\n", flag, optarg, (double)minv,
				(double)maxv);
		return -1;
	}
	*out = (float)v;
	return 0;
}

static int bool_token_is_valid(const char *s) {
	return !strcmp(s, "yes") || !strcmp(s, "on") || !strcmp(s, "true") || !strcmp(s, "1") ||
		   !strcmp(s, "no") || !strcmp(s, "off") || !strcmp(s, "false") || !strcmp(s, "0");
}

static const char *peek_optional_bool_arg(const char *optarg, int argc, char **argv, int *optind) {
	if (optarg)
		return optarg;
	if (*optind >= argc)
		return NULL;
	const char *candidate = argv[*optind];
	if (bool_token_is_valid(candidate)) {
		(*optind)++;
		return candidate;
	}
	return NULL;
}

static int parse_kv_quant(const char *optarg, config *cfg) {
	if (strcmp(optarg, "f16") == 0) {
		cfg->kv_quant = KV_QUANT_F16;
	} else if (strcmp(optarg, "q8_0") == 0) {
		cfg->kv_quant = KV_QUANT_Q8_0;
	} else {
		fprintf(stderr, "invalid --kv-quant value '%s' (expected f16 or q8_0)\n", optarg);
		return -1;
	}
	return 0;
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

	const char *prog	= (argc > 0 && argv[0]) ? argv[0] : "";
	const char *prog_bn = strrchr(prog, '/');
	prog_bn				= prog_bn ? prog_bn + 1 : prog;
	a->is_server		= strstr(prog_bn, "server") != NULL;

	a->n_predict	  = -1;
	a->temperature	  = 0.80f;
	a->top_k		  = 40;
	a->top_p		  = 0.90f;
	a->min_p		  = 0.10f;
	a->repeat_penalty = 1.0f;
	a->repeat_last_n  = 64;
	a->output_stream  = true;
	snprintf(a->metrics, sizeof(a->metrics), "pp,tg");
	a->warmup	   = true;
	a->server_host = "127.0.0.1";
	a->server_port = 8080;

	enum {
		OPT_SYSTEM = 256,
		OPT_STREAM,
		OPT_TIME,
		OPT_DEVICE,
		OPT_LIST_DEVICES,
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
		OPT_DEBUG_PROMPT,
		OPT_GREP_VOCAB,
		OPT_KV_QUANT,
		OPT_NGL,
		OPT_METRICS,
		OPT_WARMUP,
		OPT_HOST,
		OPT_PORT,
		OPT_API_KEY,
		OPT_REPEAT_PENALTY,
		OPT_REPEAT_LAST_N
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
		{"flash-attn", optional_argument, NULL, 'f'},
		{"help", no_argument, NULL, 'h'},
		{"system", required_argument, NULL, OPT_SYSTEM},
		{"stream", optional_argument, NULL, OPT_STREAM},
		{"time", no_argument, NULL, OPT_TIME},
		{"device", required_argument, NULL, OPT_DEVICE},
		{"list-devices", no_argument, NULL, OPT_LIST_DEVICES},
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
		{"debug-prompt", no_argument, NULL, OPT_DEBUG_PROMPT},
		{"grep-vocab", required_argument, NULL, OPT_GREP_VOCAB},
		{"kv-quant", required_argument, NULL, OPT_KV_QUANT},
		{"ngl", required_argument, NULL, OPT_NGL},
		{"metrics", required_argument, NULL, OPT_METRICS},
		{"warmup", optional_argument, NULL, OPT_WARMUP},
		{"host", required_argument, NULL, OPT_HOST},
		{"port", required_argument, NULL, OPT_PORT},
		{"api-key", required_argument, NULL, OPT_API_KEY},
		{"repeat-penalty", required_argument, NULL, OPT_REPEAT_PENALTY},
		{"repeat-last-n", required_argument, NULL, OPT_REPEAT_LAST_N},
		{0, 0, 0, 0}};

	int c;
	while ((c = getopt_long(argc, argv, "m:p:n:t:k:P:M:c:s:f::h", long_opts, NULL)) != -1) {
		switch (c) {
		case 'm':
			cfg->model = optarg;
			break;
		case 'p':
			a->prompt = optarg;
			break;
		case 'n':
			if (parse_int_arg(optarg, "--n-predict", -1, INT_MAX, &a->n_predict) < 0)
				return -1;
			break;
		case 't':
			if (parse_float_arg(optarg, "--temperature", 0.0f, 1000.0f, &a->temperature) < 0)
				return -1;
			break;
		case 'k':
			if (parse_int_arg(optarg, "--top-k", 0, INT_MAX, &a->top_k) < 0)
				return -1;
			break;
		case 'P':
			if (parse_float_arg(optarg, "--top-p", 0.0f, 1.0f, &a->top_p) < 0)
				return -1;
			break;
		case 'M':
			if (parse_float_arg(optarg, "--min-p", 0.0f, 1.0f, &a->min_p) < 0)
				return -1;
			break;
		case 'c':
			if (parse_int_arg(optarg, "--ctx-size", 0, INT_MAX, &cfg->ctx_size) < 0)
				return -1;
			break;
		case 's':
			if (parse_seed_arg(optarg, "--seed", &cfg->seed) < 0)
				return -1;
			break;
		case 'f': {
			int v;
			if (parse_bool_flag("--flash-attn", peek_optional_bool_arg(optarg, argc, argv, &optind),
								1, &v) < 0)
				return -1;
			cfg->flash_attn = v;
			break;
		}
		case 'h':
			usage(stdout, a->is_server);
			exit(0);
		case OPT_SYSTEM:
			a->system = optarg;
			break;
		case OPT_STREAM: {
			int v;
			if (parse_bool_flag("--stream", peek_optional_bool_arg(optarg, argc, argv, &optind), 1,
								&v) < 0)
				return -1;
			a->output_stream = v;
			break;
		}
		case OPT_TIME:
			cfg->profile_time = true;
			break;
		case OPT_DEVICE:
			cfg->device = optarg;
			break;
		case OPT_LIST_DEVICES:
			a->list_devices = true;
			break;
		case OPT_DUMP_METADATA:
			a->dump_metadata = true;
			break;
		case OPT_DEBUG_FWD:
			cfg->debug_forward = true;
			break;
		case OPT_DEBUG_PROMPT:
			a->debug_prompt = true;
			break;
		case OPT_GREP_VOCAB:
			a->grep_vocab = optarg;
			break;
		case OPT_KV_QUANT:
			if (parse_kv_quant(optarg, cfg) < 0)
				return -1;
			break;
		case OPT_NGL:
			if (parse_int_arg(optarg, "--ngl", -1, INT_MAX, &cfg->ngl) < 0)
				return -1;
			break;
		case OPT_DEBUG:
			cfg->debug = 1;
			break;
		case OPT_THREADS:
			if (parse_int_arg(optarg, "--threads", 0, INT_MAX, &cfg->n_threads) < 0)
				return -1;
			break;
		case OPT_MOE_CACHE:
			if (parse_int_arg(optarg, "--moe-cache", 0, INT_MAX, &cfg->moe_cache_cap) < 0)
				return -1;
			break;
		case OPT_MOE_STREAM: {
			int v;
			if (parse_bool_flag("--moe-stream", peek_optional_bool_arg(optarg, argc, argv, &optind),
								1, &v) < 0)
				return -1;
			cfg->moe_stream = v;
			break;
		}
		case OPT_MOE_PRELOAD:
			cfg->moe_preload = 1;
			break;
		case OPT_MOE_PIN:
			if (parse_int_arg(optarg, "--moe-pin", 0, INT_MAX, &cfg->moe_pin) < 0)
				return -1;
			break;
		case OPT_MOE_PIN_LIST:
			cfg->moe_pin_list = optarg;
			break;
		case OPT_MONITOR:
			parse_monitor(optarg, argc, argv, &optind, cfg);
			break;
		case OPT_MMAP: {
			int v;
			if (parse_bool_flag("--mmap", peek_optional_bool_arg(optarg, argc, argv, &optind), 1,
								&v) < 0)
				return -1;
			cfg->use_mmap = v;
			break;
		}
		case OPT_REPACK:
			cfg->repack = optarg;
			break;
		case OPT_FUSE:
			cfg->fuse = optarg;
			break;
		case OPT_REASONING: {
			int v;
			if (parse_bool_flag("--reasoning", peek_optional_bool_arg(optarg, argc, argv, &optind),
								1, &v) < 0)
				return -1;
			cfg->reasoning = v;
			break;
		}
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
		case OPT_WARMUP: {
			int v;
			if (parse_bool_flag("--warmup", peek_optional_bool_arg(optarg, argc, argv, &optind), 1,
								&v) < 0)
				return -1;
			a->warmup = v;
			break;
		}
		case OPT_HOST:
			a->server_host = optarg;
			break;
		case OPT_PORT:
			a->server_port = atoi(optarg);
			if (a->server_port <= 0 || a->server_port > 65535) {
				fprintf(stderr, "invalid --port value '%s'\n", optarg);
				return -1;
			}
			break;
		case OPT_API_KEY:
			a->server_api_key = optarg;
			break;
		case OPT_REPEAT_PENALTY:
			if (parse_float_arg(optarg, "--repeat-penalty", 1e-6f, 1000.0f, &a->repeat_penalty) < 0)
				return -1;
			break;
		case OPT_REPEAT_LAST_N:
			if (parse_int_arg(optarg, "--repeat-last-n", 0, INT_MAX, &a->repeat_last_n) < 0)
				return -1;
			break;
		case '?':
			return -1;
		default:
			break;
		}
	}

	if (optind < argc && !a->prompt)
		a->prompt = argv[optind++];
	if (optind < argc) {
		ERROR("unexpected extra argument '%s' (quote the prompt if it contains spaces)",
			  argv[optind]);
		return -1;
	}
	if (!a->prompt)
		a->interactive = true;
	if (a->list_devices)
		return 0;
	if (!cfg->model) {
		ERROR("--model is required");
		return -1;
	}
	return 0;
}
