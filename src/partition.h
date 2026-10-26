#ifndef PARTITION_H
#define PARTITION_H

#include "graph.h"

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
