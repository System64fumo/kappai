#ifndef COOP_H
#define COOP_H

#include "backend/backend.h"
#include "common.h"
#include "config.h"
#include "context.h"
#include "log.h"

typedef struct coop coop;

status_code coop_open(coop **out, context *c, const config *cfg, backend *const *devices,
					  int n_devices);
void		coop_close(coop *k);
bool		coop_usable(const coop *k, const context *c, int n_tokens);
status_code coop_prefill(coop *k, context *c, const int32_t *tokens, int n, int pos, int base_rows,
						 progress *prog, layer_progress_cb layer_cb, void *layer_cb_ud);

#endif
