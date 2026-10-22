#ifndef CAPABILITY_H
#define CAPABILITY_H

#include "recipe.h"

typedef enum {
	CAP_NATIVE = 0,
	CAP_HOST,
} cap_impl;

typedef struct {
	cap_impl			 impl;
	const char			*fn;
	host_fallback_reason reason;
	const char			*detail;
	uint32_t			 type;
} cap_resolution;

cap_resolution cap_resolve(backend *be, const struct model *m, op_kind kind, const weight_ref *w,
						   int batched);
void		   cap_report(backend *be, const cap_resolution *res);
const char	  *cap_host_only_fn(op_kind kind);

#endif
