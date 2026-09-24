#ifndef ENGINE_H
#define ENGINE_H

#include "config.h"
#include "context.h"

typedef enum {
	ENGINE_EXIT				 = 1,
	ENGINE_EXIT_CONTEXT_FULL = 2,
} engine_status;

int	 engine_init(context *ctx, cli_args *a, int argc, char **argv);
int	 engine_run(context *ctx, cli_args *a);
void engine_shutdown(context *ctx);

#endif
