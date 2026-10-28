#ifndef RECIPE_PLAN_H
#define RECIPE_PLAN_H

#include "recipe/graph.h"

#define MEM_ARENA_ALIGN 256
#define MEM_LAYOUT_CACHE 8

typedef struct {
	int dim;
	int attn_buf;
	int q_out;
	int kv_out;
	int max_inter;
	int ffn_act;
	int gate_row;
	int up_row;
	int hc_flat;
	int hc_coef;
	int hyb_proj;
	int hyb_gate;
	int hyb_heads;
	int max_head_dim;
	int max_kv_heads;
} mem_dims;

typedef struct mem_plan {
	mem_dims dims;
	uint8_t	 planned[RECIPE_SLOT_MAX];
	uint8_t	 rep[RECIPE_SLOT_MAX];
	size_t	 row_elems[RECIPE_SLOT_MAX];
	size_t	 group_row_elems[RECIPE_SLOT_MAX];
	int		 n_slots;
	int		 n_buffers;
} mem_plan;

typedef struct {
	size_t offset[RECIPE_SLOT_MAX];
	size_t cap[RECIPE_SLOT_MAX];
	size_t total;
	int	   bucket;
} mem_layout;

typedef struct {
	const mem_plan *plan;
	mem_layout		entries[MEM_LAYOUT_CACHE];
	int				n;
	int				next;
} mem_layout_cache;

void			  mem_dims_init(mem_dims *d, const struct model *m);
mem_plan		 *mem_plan_build(const cg_set *graphs, const struct model *m);
void			  mem_plan_free(mem_plan *p);
int				  mem_rows_bucket(int n_rows);
const mem_layout *mem_layout_get(mem_layout_cache *cache, const mem_plan *p, int n_rows);

#define PART_CARRY_MAX 2

typedef enum { PART_COMPUTE = 0, PART_TRANSFER, PART_GATHER } part_kind;

typedef struct {
	part_kind kind;
	backend	 *be;
	backend	 *src;
	int		  first_layer;
	int		  n_layers;
} part_node;

typedef struct {
	uint8_t slot;
	int		row_elems;
} part_carry;

typedef struct part_plan {
	part_node *nodes;
	int		   n_nodes;
	part_carry carry[PART_CARRY_MAX];
	int		   n_carry;
} part_plan;

part_plan *part_plan_build(const struct model *m);
void	   part_plan_free(part_plan *p);
int		   part_plan_n_transfers(const part_plan *p);
int		   part_plan_check_carry(const part_plan *p, const cg_set *graphs);

/* ------------------------------------------------------------------ *
 * Value-level plans for raw cg_graph execution.
 *
 * The slot-based mem_plan above is the LLM-specialized planner (fixed
 * slots, batch-size buckets).  cg_plan is the general value-level
 * planner: it computes liveness over explicit graph values and packs
 * them into a first-fit arena.  Both paths share cg_graph, the op_kind
 * enum and the backend dispatch; diffusion models plan their graphs
 * with cg_plan and run them through cg_graph_run (recipe/exec.h).
 * ------------------------------------------------------------------ */

#define CG_NO_OFFSET ((size_t)-1)
#define CG_ARENA_ALIGN_FLOATS 64
#define CG_PLAN_CACHE_SIZE 3

typedef struct cg_exec cg_exec;
typedef status_code (*cg_node_handler)(const cg_exec *x, const cg_node *n);

typedef struct {
	cg_node_handler fn;
	uint64_t		us, calls;
} cg_node_step;

typedef struct {
	size_t *offset;			  /* [n_values + 1], CG_NO_OFFSET where not arena-backed */
	size_t	arena_floats;	  /* total arena size in floats */
	size_t	unplanned_floats; /* sum of live value sizes without reuse */
} cg_arena_layout;

void		cg_arena_layout_plan(cg_arena_layout *l, const cg_graph *c, const cg_range *ranges);
status_code cg_arena_layout_build(cg_arena_layout *l, const cg_graph *c);
void		cg_arena_layout_release(cg_arena_layout *l);

typedef struct {
	cg_builder	   *graph;
	cg_node_step   *steps;
	cg_arena_layout layout;
	const char	   *label;
	bool			profiling;
} cg_plan;

status_code cg_plan_build(cg_builder *g, backend *be, const char *label, cg_plan **out);
void		cg_plan_free(cg_plan *p);
void		cg_plan_report(cg_plan *p, FILE *fp);

typedef struct {
	cg_plan *plans[CG_PLAN_CACHE_SIZE];
	uint64_t keys[CG_PLAN_CACHE_SIZE];
	int		 count, next;
} cg_plan_cache;

cg_plan *cg_plan_cache_get(const cg_plan_cache *c, uint64_t key);
void	 cg_plan_cache_put(cg_plan_cache *c, uint64_t key, cg_plan *plan);
void	 cg_plan_cache_free(cg_plan_cache *c);

static inline uint64_t cg_plan_key(int a, int b) {
	return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}

cg_node_handler cg_node_handler_for(op_kind kind);

#endif
