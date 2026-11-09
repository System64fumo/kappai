#ifndef RECIPE_CAPABILITY_H
#define RECIPE_CAPABILITY_H

#include "recipe.h"

typedef enum {
	CAP_NATIVE = 0,
	CAP_HOST,
} cap_impl;

typedef struct {
	cap_impl	impl;
	const char *fn;
} cap_resolution;

cap_resolution cap_resolve(backend *be, const struct model *m, op_kind kind, const weight_ref *w,
						   int batched);
const char	  *cap_host_only_fn(op_kind kind);

#endif
