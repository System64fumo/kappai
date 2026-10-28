#ifndef RECIPE_EXEC_H
#define RECIPE_EXEC_H

#include "recipe/plan.h"

typedef struct {
	buffer buf;
	size_t floats;
} cg_arena;

typedef struct {
	const float *host_in[CG_MAX_BINDS];
	buffer		 layer_ext[CG_MAX_BINDS];
	float		*host_out[CG_MAX_BINDS];
} cg_exec_args;

struct cg_exec {
	backend			   *be;
	const cg_builder   *graph;
	cg_plan			   *plan;
	cg_arena		   *arena;
	const cg_exec_args *args;
	int					layer;
};

bool		cg_backend_supported(backend *be);
status_code cg_backend_open(const char *spec, backend **out);
status_code cg_graph_run(backend *be, cg_arena *arena, cg_plan *plan, const cg_exec_args *args);
void		cg_arena_free(cg_arena *arena);

#endif
