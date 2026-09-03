#include "engine.h"

#include "backend/backend.h"
#include "common.h"
#include "gguf.h"
#include "log.h"
#include "moe/moe_stream.h"

#include <malloc.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define COLOR_RESET "\033[0m"
#define COLOR_GRAY "\033[90m"

typedef struct {
	bool	 output_stream;
	int		*p_gen;
	bool	 in_thinking;
	bool	 first_token;
	bool	 skip_label;
	int32_t	 think_start_id;
	int32_t	 think_end_id;
	context *c;
} on_token_ud;

static int g_use_color	   = 0;
static int g_stdout_isatty = 0;

static context *g_ctx = NULL;

static void on_sigint(int sig) {
	(void)sig;
	if (g_ctx) {
		g_ctx->interrupt = 1;
	}
}

static void print_start_thinking(void) {
	fputc('\n', stdout);
	if (g_use_color)
		fputs(COLOR_GRAY, stdout);
	fputs("[Start thinking]\n", stdout);
	fflush(stdout);
}

static void print_end_thinking(void) {
	fputs("\n[End thinking]\n", stdout);
	if (g_use_color)
		fputs(COLOR_RESET, stdout);
	fputc('\n', stdout);
	fflush(stdout);
}

static void on_token_cb(int32_t id, const char *piece, int n, void *ud) {
	on_token_ud *u = (on_token_ud *)ud;
	(*u->p_gen)++;

	if (!u->output_stream)
		return;

	if (u->first_token && u->c && u->c->chat.think_open) {
		u->in_thinking = true;
		print_start_thinking();
	}
	u->first_token = false;

	if (id == u->think_start_id) {
		u->in_thinking = true;
		u->skip_label  = true;
		print_start_thinking();
		return;
	}
	if (id == u->think_end_id) {
		if (u->in_thinking) {
			u->in_thinking = false;
			print_end_thinking();
		}
		return;
	}

	if (u->skip_label) {
		const char *nl = memchr(piece, '\n', (size_t)n);
		if (!nl)
			return;
		n			  = n - (int)(nl - piece) - 1;
		piece		  = nl + 1;
		u->skip_label = false;
	}

	if (n > 0) {
		fwrite(piece, 1, n, stdout);
		if (g_stdout_isatty)
			fflush(stdout);
		u->first_token = false;
	}
}

static status_code warmup_run(context *c, const char *system) {
	if (!c)
		return ERR_INVALID_ARG;

	const char *sys = (system && *system) ? system : "";

	size_t prefix_bytes = chat_template_detect_static_prefix(&c->chat, sys);
	if (prefix_bytes == 0)
		return OK;

	char *prev_render = xstrdup(c->chat.last_render);

	char	   *render = NULL;
	char		errbuf[512];
	status_code rc =
		chat_template_add_turn(&c->chat, "system", sys, 0, &render, errbuf, sizeof(errbuf));
	if (rc != OK) {
		free(prev_render);
		return OK;
	}

	int		 cap = c->n_ctx;
	int32_t *ids = context_ids_scratch(c, cap + 1);
	int		 n	 = tokenizer_encode_with_specials(&c->tok, render, 0, ids, cap, &c->scratch.prof);
	if (n < 0) {
		free(render);
		goto restore;
	}

	int count = tokenizer_token_count_for_bytes(&c->tok, ids, n, prefix_bytes);
	if (count == 0) {
		free(render);
		goto restore;
	}

	context_monitor_send_start(c);
	prefill_result pf = context_prefill_tokens(c, ids, count, "warmup", true);
	if (pf.rc < 0) {
		free(render);
		free(prev_render);
		c->session_poisoned = true;
		return ERR_INTERNAL;
	}

	c->warmup_done = true;

	DEBUG("warmup: prefilled %d tokens", count);

	free(render);
	free(prev_render);
	return OK;

restore:
	free(c->chat.last_render);
	c->chat.last_render = prev_render;
	return OK;
}

static int run_chat_turn(context *c, cli_args *a, const char *text) {
	int			   gen	= 0;
	on_token_ud	   ud	= {a->output_stream,	 &gen, false, true, false, c->chat.think_start_id,
						   c->chat.think_end_id, c};
	sampler_params samp = {a->temperature, a->top_k,		  a->top_p,
						   a->min_p,	   a->repeat_penalty, a->repeat_last_n};
	return context_chat_turn(c, "user", text, true, a->n_predict, &samp, on_token_cb, &ud,
							 a->metrics[0] ? a->metrics : "pp,tg");
}

static int run_one_shot(context *c, cli_args *a) {
	context_idle_prefill_wait(c);

	if (!c->warmup_done) {
		context_reset(c);
		if (a->warmup)
			warmup_run(c, a->system);
	}
	c->warmup_done = false;

	if (c->session_poisoned) {
		ERROR("startup warmup left inconsistent state; refusing to generate");
		return -1;
	}

	if (a->output_stream)
		fflush(stderr);
	int r = run_chat_turn(c, a, a->prompt);
	printf("\n");

	int rc = 0;
	if (r < 0)
		rc = -1;
	if (c->context_limit_hit) {
		c->context_limit_hit = false;
		WARN("context window exhausted (n_ctx=%d). Shorten the prompt, lower "
			 "--n-predict, or raise --ctx-size.",
			 c->n_ctx);
		rc = ENGINE_EXIT_CONTEXT_FULL;
	}
	if (c->session_poisoned) {
		ERROR("generation left inconsistent cache state; session must be reset");
		rc = -1;
	}
	return rc;
}

static void reset_and_warmup(context *c, cli_args *a, const char *system) {
	context_idle_prefill_wait(c);
	context_reset(c);
	if (a->warmup) {
		warmup_run(c, system);
		if (!c->session_poisoned)
			context_idle_prefill_start(c);
	}
}

static int run_interactive(context *c, cli_args *a) {
	printf("kappai interactive chat.\n");
	printf("Commands: /exit /reset /system <text>  (Ctrl+C interrupts generation)\n\n");

	if (!c->warmup_done) {
		context_reset(c);
		if (a->warmup)
			warmup_run(c, a->system);
	}
	c->warmup_done = false;

	if (a->warmup && !c->session_poisoned)
		context_idle_prefill_start(c);

	char line[4096];
	while (1) {
		printf("> ");
		fflush(stdout);
		if (!fgets(line, sizeof(line), stdin))
			break;
		size_t len = strlen(line);
		while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
			line[--len] = '\0';
		if (len == 0)
			continue;

		if (!strcmp(line, "/exit") || !strcmp(line, "/quit"))
			break;
		if (!strcmp(line, "/reset")) {
			reset_and_warmup(c, a, a->system);
			printf("[context reset]\n\n");
			continue;
		}
		if (!strncmp(line, "/system ", 8)) {
			reset_and_warmup(c, a, line + 8);
			printf("[system prompt updated]\n\n");
			continue;
		}

		context_idle_prefill_wait(c);

		c->interrupt = 0;

		int r = run_chat_turn(c, a, line);

		if (c->interrupt) {
			c->interrupt = 0;
			printf("\n[interrupted]");
		}
		if (c->context_limit_hit) {
			c->context_limit_hit = false;
			printf("\n[context limit reached]");
		}
		printf("\n\n");
		if (c->session_poisoned || r < 0) {
			if (c->session_poisoned)
				ERROR("turn left inconsistent cache state; resetting context");
			else
				ERROR("turn failed; resetting context");
			reset_and_warmup(c, a, a->system);
		} else if (a->warmup) {
			context_idle_prefill_start(c);
		}
	}

	context_idle_prefill_wait(c);
	return 0;
}

static int list_devices(void) {
	backend_info infos[BACKEND_MAX];
	int			 n = backend_list(infos, BACKEND_MAX);
	printf("available compute devices:\n");
	for (int i = 0; i < n; i++) {
		if (!infos[i].available)
			continue;
		for (int d = 0; d < infos[i].n_devices; d++) {
			bool is_host = (infos[i].caps & BCAP_IS_HOST) != 0;
			printf("  %s%d%s\n", infos[i].name, d, is_host ? " (host fallback)" : "");
		}
	}
	return 0;
}

static int dump_metadata(const char *model_path) {
	gguf_ctx	gctx;
	status_code s = gguf_load_metadata(&gctx, model_path);
	if (s != OK) {
		ERROR("failed to parse GGUF metadata: %s", model_path);
		return 1;
	}
	gguf_dump(&gctx, stdout);
	gguf_free(&gctx);
	return 0;
}

static int grep_vocab(const char *model_path, const char *substr) {
	gguf_ctx	gctx;
	status_code s = gguf_load_metadata(&gctx, model_path);
	if (s != OK) {
		ERROR("failed to parse GGUF metadata: %s", model_path);
		return 1;
	}
	const char *const *toks;
	size_t			   n_toks;
	if (gguf_get_arr_str(&gctx, "tokenizer.ggml.tokens", &toks, &n_toks) != OK) {
		ERROR("GGUF has no tokenizer.ggml.tokens array");
		gguf_free(&gctx);
		return 1;
	}
	const int32_t *types   = NULL;
	size_t		   n_types = 0;
	gguf_get_arr_i32(&gctx, "tokenizer.ggml.token_type", &types, &n_types);
	for (size_t i = 0; i < n_toks; i++) {
		if (!strstr(toks[i], substr))
			continue;
		printf("%6zu  type=%-2d  \"%s\"\n", i, (types && i < n_types) ? types[i] : -1, toks[i]);
	}
	gguf_free(&gctx);
	return 0;
}

status_code engine_init(context *ctx, cli_args *a, int argc, char **argv) {
	mallopt(M_MMAP_THRESHOLD, 64 * 1024 * 1024);
	mallopt(M_MMAP_MAX, 0);
	mallopt(M_TRIM_THRESHOLD, -1);

	log_level saved_log_level = log_get_level();
	log_init(log_default_config());
	log_set_level(saved_log_level);

	config cfg;
	if (parse_args(argc, argv, &cfg, a) < 0) {
		usage(stderr, a->is_server);
		return ERR_INVALID_ARG;
	}

	if (a->list_devices) {
		list_devices();
		return ENGINE_EXIT;
	}
	if (a->dump_metadata) {
		int rc = dump_metadata(cfg.model);
		return rc == 0 ? ENGINE_EXIT : ERR_NOT_FOUND;
	}
	if (a->grep_vocab) {
		int rc = grep_vocab(cfg.model, a->grep_vocab);
		return rc == 0 ? ENGINE_EXIT : ERR_NOT_FOUND;
	}

	config_init(&cfg);
	const config *ec = config_get();

	const char *model_base = strrchr(ec->model, '/');
	model_base			   = model_base ? model_base + 1 : ec->model;

	DEBUG("model path: %s", ec->model);
	DEBUG("device=%s use_mmap=%d seed=%llu", ec->device ? ec->device : "auto", ec->use_mmap,
		  (unsigned long long)ec->seed);
	uint64_t	t0 = time_ms();
	status_code s  = context_init(ctx, ec);
	if (s != OK) {
		ERROR("failed to load model: %d", s);
		return s;
	}
	INFO("Loaded: %s in %llu ms", model_base, (unsigned long long)(time_ms() - t0));
	INFO("n_layers=%d dim=%d vocab=%d ctx=%d flash_attn=%s", ctx->m.n_layers, ctx->m.dim,
		 ctx->m.vocab_size, ctx->n_ctx, ctx->flash_attn ? "on" : "off");
	if (ctx->m.arch_info->is_moe && ec->moe_stream) {
		INFO("moe: streaming enabled cache_cap=%d pin=%d", ec->moe_cache_cap, ec->moe_pin);
	}

	ctx->chat.enable_thinking = ec->reasoning;
	DEBUG("reasoning=%s", ec->reasoning ? "on" : "off");

	if (ec->debug_forward) {
		ctx->debug_forward = true;
	}

	if (ec->moe_preload && ctx->m.arch_info->is_moe) {
		uint64_t pl_t0 = time_ms();
		moe_stream_preload_all(&ctx->m);
		INFO("MoE preload completed in %llu ms", (unsigned long long)(time_ms() - pl_t0));
	}

	ctx->scratch.prof.enabled = (ec->profile_time) ? 1 : 0;
	if (!(ec->profile_time))
		profile_reset(&ctx->scratch.prof);

	context_set_debug_prompt(ctx, a->debug_prompt);

	malloc_trim(0);

	return OK;
}

int engine_run(context *ctx, cli_args *a) {
	g_use_color		= isatty(STDOUT_FILENO) && isatty(STDERR_FILENO);
	g_stdout_isatty = isatty(STDOUT_FILENO);

	g_ctx = ctx;
	signal(SIGINT, on_sigint);

	int rc;
	if (a->interactive) {
		rc = run_interactive(ctx, a);
	} else {
		rc = run_one_shot(ctx, a);
	}

	signal(SIGINT, SIG_DFL);
	g_ctx = NULL;

	return rc;
}

void engine_shutdown(context *ctx) {
	context_free(ctx);
}