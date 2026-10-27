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

#endif
