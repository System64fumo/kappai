#include "diffusion/runner.h"

#include "log.h"
#include "profile.h"

#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define PROMPT_MAX 4096
#define ERR_MAX 512

static volatile sig_atomic_t g_interrupt;

static void on_sigint(int sig) {
	(void)sig;
	g_interrupt = 1;
}

static void install_sigint_handler(int flags) {
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = on_sigint;
	sa.sa_flags	  = flags;
	sigemptyset(&sa.sa_mask);
	sigaction(SIGINT, &sa, NULL);
}

static bool auto_output_path(const diffusion_output *out, uint64_t seed, char *buf, size_t cap) {
	time_t	  now = time(NULL);
	struct tm tm;
	char	  stamp[32];
	localtime_r(&now, &tm);
	strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);
	return snprintf(buf, cap, "%s/%s-%s-%llu.png", out->dir, out->name, stamp,
					(unsigned long long)seed) < (int)cap;
}

static status_code generate_image(diffusion_model *dm, const diffusion_params *params,
								  const diffusion_output *out, const char *prompt) {
	diffusion_params run = *params;
	run.prompt			 = prompt;
	run.seed			 = params->seed ? params->seed : time_us();
	run.interrupt		 = &g_interrupt;

	char		auto_path[PATH_MAX];
	const char *path = out->path;
	if (!path) {
		if (!auto_output_path(out, run.seed, auto_path, sizeof(auto_path))) {
			fprintf(stderr, "output path too long\n\n");
			return ERR_INVALID_ARG;
		}
		path = auto_path;
	}

	diffusion_image img = {0};
	g_interrupt			= 0;
	install_sigint_handler(SA_RESTART);
	status_code s = diffusion_generate(dm, &run, &img);
	install_sigint_handler(0);
	if (s == ERR_INTERRUPTED) {
		printf("[interrupted]\n\n");
		return s;
	}
	if (s != OK) {
		fprintf(stderr, "generation failed: %s\n\n", status_str(s));
		return s;
	}
	s = diffusion_image_write_png(path, &img);
	diffusion_image_free(&img);
	if (s != OK) {
		fprintf(stderr, "cannot write '%s': %s\n\n", path, status_str(s));
		return s;
	}
	printf("[saved %s]\n\n", path);
	return OK;
}

static bool read_prompt(char *line, size_t cap) {
	printf("> ");
	fflush(stdout);
	if (!fgets(line, (int)cap, stdin))
		return false;
	size_t len		= strlen(line);
	bool   complete = len > 0 && line[len - 1] == '\n';
	if (!complete && !feof(stdin)) {
		int ch;
		while ((ch = getchar()) != '\n' && ch != EOF)
			;
		fprintf(stderr, "prompt too long (max %d characters)\n\n", PROMPT_MAX - 1);
		line[0] = '\0';
		return true;
	}
	while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
		line[--len] = '\0';
	return true;
}

static int run_chat(diffusion_model *dm, const diffusion_params *params,
					const diffusion_output *out) {
	printf("kappai diffusion chat.\n");
	printf("Each prompt generates a new image. Commands: /exit\n");
	printf("(Ctrl+C cancels generation, or exits at the prompt)\n\n");

	install_sigint_handler(0);
	char line[PROMPT_MAX];
	while (read_prompt(line, sizeof(line))) {
		if (!line[0])
			continue;
		if (!strcmp(line, "/exit") || !strcmp(line, "/quit"))
			break;
		generate_image(dm, params, out, line);
	}
	return 0;
}

int diffusion_run(const diffusion_files *files, const diffusion_params *params,
				  const diffusion_output *out, bool debug) {
	const char *params_error = diffusion_params_error(params);
	if (params_error) {
		fprintf(stderr, "kappai: %s\n", params_error);
		return 1;
	}
	if (!files->vae || !files->vae[0] || !files->llm || !files->llm[0]) {
		fprintf(stderr, "kappai: diffusion models need both a VAE and an LLM file\n");
		return 1;
	}

	char err[ERR_MAX];
	if (!out->path && mkdir_p(out->dir, err, sizeof(err)) < 0) {
		fprintf(stderr, "kappai: %s\n", err);
		return 1;
	}

	log_init(log_default_config());
	if (debug)
		log_set_level(LOG_DEBUG);

	diffusion_model *dm = diffusion_open(files, err, sizeof(err));
	if (!dm) {
		ERROR("%s", err);
		return 1;
	}

	int rc = 0;
	if (params->prompt)
		rc = generate_image(dm, params, out, params->prompt) == OK ? 0 : 1;
	else
		rc = run_chat(dm, params, out);

	signal(SIGINT, SIG_DFL);
	diffusion_close(dm);
	return rc;
}
