#ifndef ENGINE_H
#define ENGINE_H

#include "config.h"
#include "context.h"

#define ENGINE_EXIT 1

status_code engine_init(context *ctx, cli_args *a, int argc, char **argv);
int			engine_run(context *ctx, cli_args *a);
void		engine_shutdown(context *ctx);

#endif
