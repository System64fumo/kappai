#include "engine.h"

int main(int argc, char **argv) {
	context	 ctx;
	cli_args a;

	status_code s = engine_init(&ctx, &a, argc, argv);
	if (s == ENGINE_EXIT)
		return 0;
	if (s != OK)
		return 1;

	int rc = engine_run(&ctx, &a);

	engine_shutdown(&ctx);
	return rc;
}
