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

	status_code s = engine_init(&ctx, &a, argc, argv);
	if (s == ENGINE_EXIT)
		return 0;
	if (s != OK)
		return 1;

	ctx.quiet_progress = true;

	openai_state *oa = openai_init(&ctx, &a);
	if (!oa) {
		ERROR("out of memory");
		engine_shutdown(&ctx);
		return 1;
	}

	if (!openai_bind(oa, a.server_host, a.server_port)) {
		engine_shutdown(&ctx);
		return 1;
	}

	char errbuf[256];
	if (!openai_serve(oa, errbuf, sizeof(errbuf))) {
		ERROR("failed to start server on %s:%d: %s", a.server_host, a.server_port, errbuf);
		openai_free(oa);
		engine_shutdown(&ctx);
		return 1;
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
}
