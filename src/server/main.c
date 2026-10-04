#include "config.h"
#include "engine.h"
#include "log.h"
#include "openai.h"

#include <errno.h>
#include <string.h>

int main(int argc, char **argv) {
	context	 ctx;
	cli_args a;

	openai_install_signals();

	int s = engine_init(&ctx, &a, argc, argv);
	if (s == ENGINE_EXIT)
		return 0;
	if (s != OK)
		return 1;

	ctx.quiet_progress = true;

	openai_state *oa = openai_init(&ctx, &a);
	if (!oa) {
		ERROR("out of memory");
		goto shutdown_engine;
	}

	if (!openai_bind(oa, a.server_host, a.server_port))
		goto shutdown_server;

	char errbuf[256];
	if (!openai_serve(oa, errbuf, sizeof(errbuf))) {
		ERROR("failed to start server on %s:%d: %s", a.server_host, a.server_port, errbuf);
		goto shutdown_server;
	}

	INFO("listening on http://%s:%d", a.server_host, a.server_port);
	INFO("model: %s", config_get()->model);
	INFO("endpoints: GET /health | GET /v1/models | POST /v1/chat/completions | "
		 "POST /v1/completions");

	openai_wait_for_signal();

	openai_free(oa);
	INFO("shut down complete");

	engine_shutdown(&ctx);
	return 0;

shutdown_server:
	openai_free(oa);
shutdown_engine:
	engine_shutdown(&ctx);
	return 1;
}
