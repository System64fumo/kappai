#define _GNU_SOURCE
#include "dialog.h"
#include "editor.h"
#include "engine.h"
#include "menu.h"
#include "schema.h"
#include "server/openai.h"
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

static void tui_run_config(const sc_entry *entry, bool server_mode) {
	saved_config sc;
	char		 err[SC_ERR_MAX];
	if (sc_load(entry->path, &sc, err, sizeof(err)) < 0) {
		ui_message("Could not load config", err, true);
		return;
	}

	config	 cfg;
	cli_args a;
	sc_materialize(&sc, &cfg, &a);
	a.interactive = true;
	a.is_server	  = server_mode;

	context ctx;
	int		rc;

	def_prog_mode();
	endwin();

	if (server_mode) {
		openai_install_signals();
	} else {
		signal(SIGINT, SIG_DFL);
	}

	int s = engine_init_config(&ctx, &a, &cfg);
	if (s != OK) {
		rc = 1;
	} else if (server_mode) {
		rc = run_server(&ctx, &a);
	} else {
		rc = engine_run(&ctx, &a);
		engine_shutdown(&ctx);
	}

	if (rc != 0)
		wait_for_keypress();

	sigset_t mask;
	sigemptyset(&mask);
	sigaddset(&mask, SIGINT);
	sigaddset(&mask, SIGTERM);
	pthread_sigmask(SIG_UNBLOCK, &mask, NULL);

	reset_prog_mode();
	ui_after_resume();
}

int main(int argc, char **argv) {
	if (argc > 1) {
		fprintf(stderr, "kappai takes no launch flags\n");
		fprintf(stderr, "run it without arguments to pick a saved configuration\n");
		return 1;
	}

	setlocale(LC_ALL, "");
	if (ui_init() < 0) {
		fprintf(stderr, "failed to initialize the terminal\n");
		return 1;
	}

	int rc = 0;
	for (;;) {
		menu_result mr;
		if (menu_run(&mr) < 0) {
			rc = 1;
			break;
		}
		if (mr.action == MENU_QUIT)
			break;
		if (mr.action == MENU_NEW) {
			editor_run(NULL);
			continue;
		}
		if (mr.action == MENU_EDIT) {
			editor_run(&mr.entry);
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
