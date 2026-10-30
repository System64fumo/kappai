#define _GNU_SOURCE
#include "dialog.h"
#include "editor.h"
#include "engine.h"
#include "hcs.h"
#include "menu.h"
#include "schema.h"
#include "server/openai.h"
#include "settings.h"
#include "setup.h"
#include "ui.h"

#include <locale.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int run_server(context *ctx, const cli_args *a) {
	ctx->quiet_progress = true;

	openai_state *oa = openai_init(ctx, a);
	if (!oa) {
		engine_shutdown(ctx);
		return 1;
	}
	if (!openai_bind(oa, a->server_host, a->server_port)) {
		fprintf(stderr, "failed to bind %s:%d\n", a->server_host, a->server_port);
		openai_free(oa);
		engine_shutdown(ctx);
		return 1;
	}

	char errbuf[256];
	if (!openai_serve(oa, errbuf, sizeof(errbuf))) {
		fprintf(stderr, "failed to start server on %s:%d: %s\n", a->server_host, a->server_port,
				errbuf);
		openai_free(oa);
		engine_shutdown(ctx);
		return 1;
	}

	printf("listening on http://%s:%d (press Ctrl+C to stop)\n", a->server_host, a->server_port);
	openai_wait_for_signal();
	openai_free(oa);
	engine_shutdown(ctx);
	return 0;
}

static void wait_for_keypress(void) {
	if (!isatty(STDIN_FILENO))
		return;
	printf("\n[press Enter to return to the menu]\n");
	fflush(stdout);
	int c;
	while ((c = getchar()) != '\n' && c != EOF)
		;
}

static int run_saved_config(saved_config *sc, bool server_mode) {
	sc_resolve_model_paths(sc);
	char hcs_err[HCS_ERR_MAX];
	if (hcs_validate(&sc->hcs, sc->model, sc->moe_stream, hcs_err, sizeof(hcs_err)) < 0) {
		fprintf(stderr, "kappai: malformed/misconfigured config '%s':\n%s", sc->name, hcs_err);
		return 1;
	}

	if (diffusion_probe(sc->model)) {
		if (server_mode) {
			fprintf(stderr, "kappai: server mode is not supported for diffusion models\n");
			return 1;
		}
		diffusion_files	 files;
		diffusion_params params;
		diffusion_output out;
		sc_materialize_diffusion(sc, &files, &params, &out);
		return diffusion_run(&files, &params, &out, sc->debug);
	}

	config	 cfg;
	cli_args a;
	sc_materialize(sc, &cfg, &a);
	a.interactive = true;
	a.is_server	  = server_mode;

	if (server_mode)
		openai_install_signals();
	else
		signal(SIGINT, SIG_DFL);

	context ctx;
	if (engine_init_config(&ctx, &a, &cfg) != OK)
		return 1;
	if (server_mode)
		return run_server(&ctx, &a);

	int rc = engine_run(&ctx, &a);
	engine_shutdown(&ctx);
	return rc;
}

static void tui_run_config(const sc_entry *entry, bool server_mode) {
	saved_config sc;
	char		 err[SC_ERR_MAX];
	if (sc_load(entry->path, &sc, err, sizeof(err)) < 0) {
		ui_message("Could not load config", err, true);
		return;
	}

	sigset_t mask;
	sigemptyset(&mask);
	sigaddset(&mask, SIGWINCH);
	pthread_sigmask(SIG_BLOCK, &mask, NULL);

	def_prog_mode();
	endwin();

	int rc = run_saved_config(&sc, server_mode);
	if (rc != 0)
		wait_for_keypress();

	sigemptyset(&mask);
	sigaddset(&mask, SIGINT);
	sigaddset(&mask, SIGTERM);
	pthread_sigmask(SIG_UNBLOCK, &mask, NULL);

	reset_prog_mode();
	pthread_sigmask(SIG_UNBLOCK, &mask, NULL);
	ui_after_resume();
}

static int launch_config(const char *name, const char *mode) {
	bool server_mode = strcmp(mode, "server") == 0;
	if (!server_mode && strcmp(mode, "chat") != 0) {
		fprintf(stderr, "kappai: unknown run type '%s' (expected chat or server)\n", mode);
		return 1;
	}

	const char *dir = sc_configs_dir();
	char		path[SC_PATH_MAX];
	if (!dir || strchr(name, '/') ||
		snprintf(path, sizeof(path), "%s/%s.conf", dir, name) >= (int)sizeof(path)) {
		fprintf(stderr, "kappai: unknown config '%s'\n", name);
		return 1;
	}

	saved_config sc;
	char		 err[SC_ERR_MAX];
	if (sc_load(path, &sc, err, sizeof(err)) < 0) {
		fprintf(stderr, "kappai: %s\n", err);
		return 1;
	}
	return run_saved_config(&sc, server_mode);
}

int main(int argc, char **argv) {
	if (argc == 2) {
		fprintf(stderr, "kappai: missing run type for config '%s' (chat or server)\n", argv[1]);
		return 1;
	}
	if (argc > 3) {
		fprintf(stderr, "usage: %s [config_name chat|server]\n", argv[0]);
		return 1;
	}
	if (argc == 3) {
		char settings_err[SC_ERR_MAX];
		if (tui_settings_load(settings_err, sizeof(settings_err)) < 0) {
			fprintf(stderr, "kappai: %s\n", settings_err);
			return 1;
		}
		return launch_config(argv[1], argv[2]);
	}

	setlocale(LC_ALL, "");
	if (ui_init() < 0) {
		fprintf(stderr, "failed to initialize the terminal\n");
		return 1;
	}

	char settings_err[SC_ERR_MAX];
	if (tui_settings_load(settings_err, sizeof(settings_err)) < 0)
		ui_message("Could not load TUI settings", settings_err, true);
	if (!tui_settings_get()->setup_done)
		setup_run();

	int rc = 0;
	for (;;) {
		menu_result mr;
		if (menu_run(&mr) < 0) {
			rc = 1;
			break;
		}
		if (mr.action == MENU_QUIT)
			break;
		if (mr.action == MENU_NEW || mr.action == MENU_EDIT) {
			char saved[SC_NAME_MAX];
			editor_run(mr.action == MENU_EDIT ? &mr.entry : NULL, saved, sizeof(saved));
			if (saved[0])
				menu_select(saved);
			continue;
		}
		if (mr.action == MENU_RUN) {
			tui_run_config(&mr.entry, mr.server_mode);
			continue;
		}
	}

	ui_shutdown();
	return rc;
}
