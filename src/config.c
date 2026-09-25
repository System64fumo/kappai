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

void config_usage(FILE *fp, bool is_server) {
	fprintf(fp,
			"Usage: %s --model <path> [options]%s\n\n"
			"%s",
			is_server ? "kappai-server" : "kappai-cli", is_server ? "" : " [prompt]",
			is_server ? "Starts an OpenAI-compatible HTTP server.\n\n"
					  : "Runs an interactive chat REPL by default. Use -p for a single\n"
						"non-interactive turn (scripting, testing, experimentation).\n");

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
			"  --reasoning [bool]       allow model reasoning/thinking (default: on)\n");

	if (is_server) {
		fprintf(fp, "Server:\n"
					"  --host <addr>            bind address (default: 127.0.0.1)\n"
					"  --port <n>               listen port (default: 8080)\n"
					"  --api-key <key>          require 'Authorization: Bearer <key>' on /v1 "
					"requests\n");
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
			"  -h, --help               show this help\n");
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
	ERROR("invalid %s value '%s' (expected one of: yes, on, true, 1, no, off, false, 0)", flag,
		  optarg);
	return -1;
}

static int parse_int_arg(const char *optarg, const char *flag, long minv, long maxv, int *out) {
	char *end = NULL;
	errno	  = 0;
	long v	  = strtol(optarg, &end, 10);
	if (end == optarg || *end != '\0') {
		ERROR("invalid %s value '%s' (expected an integer)", flag, optarg);
		return -1;
	}
	if (errno == ERANGE || v < minv || v > maxv) {
		ERROR("%s value '%s' out of range [%ld, %ld]", flag, optarg, minv, maxv);
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
		ERROR("invalid %s value '%s' (expected an unsigned integer)", flag, optarg);
		return -1;
	}
	if (errno == ERANGE) {
		ERROR("%s value '%s' out of range [0, %llu]", flag, optarg, (unsigned long long)UINT64_MAX);
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
		ERROR("invalid %s value '%s' (expected a number)", flag, optarg);
		return -1;
	}
	if (!isfinite(v)) {
		ERROR("invalid %s value '%s' (must be finite)", flag, optarg);
		return -1;
	}
	if (errno == ERANGE || (double)v < (double)minv || (double)v > (double)maxv) {
		ERROR("%s value '%s' out of range [%g, %g]", flag, optarg, (double)minv, (double)maxv);
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
		ERROR("invalid --kv-quant value '%s' (expected f16 or q8_0)", optarg);
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

typedef enum {
	CLI_OPT_STRING,
	CLI_OPT_INT,
	CLI_OPT_FLOAT,
	CLI_OPT_BOOL_FLAG,
	CLI_OPT_FLAG_TRUE,
	CLI_OPT_SEED,
	CLI_OPT_KV_QUANT,
	CLI_OPT_MONITOR,
	CLI_OPT_METRICS,
	CLI_OPT_PORT,
	CLI_OPT_HELP,
} cli_opt_kind;

typedef struct {
	const char	*name;
	int			 val;
	cli_opt_kind kind;
	void		*target;
	double		 lo;
	double		 hi;
} cli_option_desc;

static int cli_opt_arg_type(cli_opt_kind kind) {
	switch (kind) {
	case CLI_OPT_BOOL_FLAG:
	case CLI_OPT_MONITOR:
		return optional_argument;
	case CLI_OPT_FLAG_TRUE:
	case CLI_OPT_HELP:
		return no_argument;
	default:
		return required_argument;
	}
}

static int handle_cli_option(const cli_option_desc *o, const char *optarg, int argc, char **argv,
							 int *optind, config *cfg, cli_args *a) {
	char flag[64];
	snprintf(flag, sizeof(flag), "--%s", o->name);
	switch (o->kind) {
	case CLI_OPT_STRING:
		*(const char **)o->target = optarg;
		return 0;
	case CLI_OPT_INT:
		return parse_int_arg(optarg, flag, (long)o->lo, (long)o->hi, (int *)o->target);
	case CLI_OPT_FLOAT:
		return parse_float_arg(optarg, flag, (float)o->lo, (float)o->hi, (float *)o->target);
	case CLI_OPT_BOOL_FLAG: {
		int v;
		if (parse_bool_flag(flag, peek_optional_bool_arg(optarg, argc, argv, optind), 1, &v) < 0)
			return -1;
		*(bool *)o->target = v ? true : false;
		return 0;
	}
	case CLI_OPT_FLAG_TRUE:
		*(bool *)o->target = true;
		return 0;
	case CLI_OPT_SEED:
		return parse_seed_arg(optarg, flag, (uint64_t *)o->target);
	case CLI_OPT_KV_QUANT:
		return parse_kv_quant(optarg, cfg);
	case CLI_OPT_MONITOR:
		parse_monitor(optarg, argc, argv, optind, cfg);
		return 0;
	case CLI_OPT_METRICS: {
		size_t l = strlen(optarg);
		if (l == 0 || l >= sizeof(a->metrics)) {
			ERROR("invalid --metrics value '%s' (max %zu chars)", optarg, sizeof(a->metrics) - 1);
			return -1;
		}
		for (size_t i = 0; i < l; i++) {
			char ch = optarg[i];
			if (ch == ',' || ch == ' ' || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) {
				a->metrics[i] = (char)tolower((unsigned char)ch);
			} else {
				ERROR("invalid character '%c' in --metrics value '%s'", ch, optarg);
				return -1;
			}
		}
		a->metrics[l] = '\0';
		return 0;
	}
	case CLI_OPT_PORT:
		a->server_port = atoi(optarg);
		if (a->server_port <= 0 || a->server_port > 65535) {
			ERROR("invalid --port value '%s'", optarg);
			return -1;
		}
		return 0;
	case CLI_OPT_HELP:
		config_usage(stdout, a->is_server);
		exit(0);
	}
	return -1;
}

static const cli_option_desc *find_cli_option(const cli_option_desc *opts, size_t n, int val) {
	for (size_t i = 0; i < n; i++)
		if (opts[i].val == val)
			return &opts[i];
	return NULL;
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

	cli_option_desc opts[] = {
		{"model", 'm', CLI_OPT_STRING, &cfg->model, 0, 0},
		{"prompt", 'p', CLI_OPT_STRING, &a->prompt, 0, 0},
		{"n-predict", 'n', CLI_OPT_INT, &a->n_predict, -1, INT_MAX},
		{"temperature", 't', CLI_OPT_FLOAT, &a->temperature, 0.0, 1000.0},
		{"top-k", 'k', CLI_OPT_INT, &a->top_k, 0, INT_MAX},
		{"top-p", 'P', CLI_OPT_FLOAT, &a->top_p, 0.0, 1.0},
		{"min-p", 'M', CLI_OPT_FLOAT, &a->min_p, 0.0, 1.0},
		{"ctx-size", 'c', CLI_OPT_INT, &cfg->ctx_size, 0, INT_MAX},
		{"seed", 's', CLI_OPT_SEED, &cfg->seed, 0, 0},
		{"flash-attn", 'f', CLI_OPT_BOOL_FLAG, &cfg->flash_attn, 0, 0},
		{"help", 'h', CLI_OPT_HELP, NULL, 0, 0},
		{"system", OPT_SYSTEM, CLI_OPT_STRING, &a->system, 0, 0},
		{"stream", OPT_STREAM, CLI_OPT_BOOL_FLAG, &a->output_stream, 0, 0},
		{"time", OPT_TIME, CLI_OPT_FLAG_TRUE, &cfg->profile_time, 0, 0},
		{"device", OPT_DEVICE, CLI_OPT_STRING, &cfg->device, 0, 0},
		{"list-devices", OPT_LIST_DEVICES, CLI_OPT_FLAG_TRUE, &a->list_devices, 0, 0},
		{"dump-metadata", OPT_DUMP_METADATA, CLI_OPT_FLAG_TRUE, &a->dump_metadata, 0, 0},
		{"debug-forward", OPT_DEBUG_FWD, CLI_OPT_FLAG_TRUE, &cfg->debug_forward, 0, 0},
		{"debug", OPT_DEBUG, CLI_OPT_FLAG_TRUE, &cfg->debug, 0, 0},
		{"threads", OPT_THREADS, CLI_OPT_INT, &cfg->n_threads, 0, INT_MAX},
		{"moe-cache", OPT_MOE_CACHE, CLI_OPT_INT, &cfg->moe_cache_cap, 0, INT_MAX},
		{"moe-stream", OPT_MOE_STREAM, CLI_OPT_BOOL_FLAG, &cfg->moe_stream, 0, 0},
		{"moe-preload", OPT_MOE_PRELOAD, CLI_OPT_FLAG_TRUE, &cfg->moe_preload, 0, 0},
		{"moe-pin", OPT_MOE_PIN, CLI_OPT_INT, &cfg->moe_pin, 0, INT_MAX},
		{"moe-pin-list", OPT_MOE_PIN_LIST, CLI_OPT_STRING, &cfg->moe_pin_list, 0, 0},
		{"monitor", OPT_MONITOR, CLI_OPT_MONITOR, NULL, 0, 0},
		{"mmap", OPT_MMAP, CLI_OPT_BOOL_FLAG, &cfg->use_mmap, 0, 0},
		{"repack", OPT_REPACK, CLI_OPT_STRING, &cfg->repack, 0, 0},
		{"fuse", OPT_FUSE, CLI_OPT_STRING, &cfg->fuse, 0, 0},
		{"reasoning", OPT_REASONING, CLI_OPT_BOOL_FLAG, &cfg->reasoning, 0, 0},
		{"disable-failsafes", OPT_DISABLE_FAILSAFES, CLI_OPT_FLAG_TRUE, &cfg->disable_failsafes, 0,
		 0},
		{"debug-prompt", OPT_DEBUG_PROMPT, CLI_OPT_FLAG_TRUE, &a->debug_prompt, 0, 0},
		{"grep-vocab", OPT_GREP_VOCAB, CLI_OPT_STRING, &a->grep_vocab, 0, 0},
		{"kv-quant", OPT_KV_QUANT, CLI_OPT_KV_QUANT, NULL, 0, 0},
		{"ngl", OPT_NGL, CLI_OPT_INT, &cfg->ngl, -1, INT_MAX},
		{"metrics", OPT_METRICS, CLI_OPT_METRICS, NULL, 0, 0},
		{"warmup", OPT_WARMUP, CLI_OPT_BOOL_FLAG, &a->warmup, 0, 0},
		{"host", OPT_HOST, CLI_OPT_STRING, &a->server_host, 0, 0},
		{"port", OPT_PORT, CLI_OPT_PORT, NULL, 0, 0},
		{"api-key", OPT_API_KEY, CLI_OPT_STRING, &a->server_api_key, 0, 0},
		{"repeat-penalty", OPT_REPEAT_PENALTY, CLI_OPT_FLOAT, &a->repeat_penalty, 1e-6, 1000.0},
		{"repeat-last-n", OPT_REPEAT_LAST_N, CLI_OPT_INT, &a->repeat_last_n, 0, INT_MAX},
	};
	size_t n_opts = sizeof(opts) / sizeof(opts[0]);

	struct option *long_opts = xmalloc((n_opts + 1) * sizeof(*long_opts));
	for (size_t i = 0; i < n_opts; i++) {
		long_opts[i].name	 = opts[i].name;
		long_opts[i].has_arg = cli_opt_arg_type(opts[i].kind);
		long_opts[i].flag	 = NULL;
		long_opts[i].val	 = opts[i].val;
	}
	memset(&long_opts[n_opts], 0, sizeof(long_opts[n_opts]));

	int c;
	while ((c = getopt_long(argc, argv, "m:p:n:t:k:P:M:c:s:f::h", long_opts, NULL)) != -1) {
		if (c == '?') {
			free(long_opts);
			return -1;
		}
		const cli_option_desc *o = find_cli_option(opts, n_opts, c);
		if (!o)
			break;
		if (handle_cli_option(o, optarg, argc, argv, &optind, cfg, a) < 0) {
			free(long_opts);
			return -1;
		}
	}
	free(long_opts);

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
