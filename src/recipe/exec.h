#ifndef RECIPE_EXEC_H
#define RECIPE_EXEC_H

#include "recipe/plan.h"

/*
 * Raw-graph execution.
 *
 * cg_graph_run executes a cg_plan on a backend: it uploads host inputs,
 * walks the graph sections (pre / layers / post) dispatching each node
 * through the shared per-op handler table in recipe/exec.c, and reads
 * results back.  The LLM recipe path drives the same backend ops via
 * its exec_plan schedules; the two paths share the op_kind enum,
 * capability resolution and every backend entry point.
 */

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
void		cg_arena_free(backend *be, cg_arena *arena);

#endif
