#ifndef PLAN_H
#define PLAN_H

#include "recipe.h"

typedef struct model_plan model_plan;

void plan_attach(model_recipe *r, const struct model *m);
void plan_free(model_plan *p);
int	 plan_fuse_residuals(model_recipe *r, const struct model *m);
int	 plan_fuse_residual_variant(recipe_op *ops, int n_ops);

#endif
