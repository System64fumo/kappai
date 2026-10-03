#ifndef MEMCONFIG_H
#define MEMCONFIG_H

#include "backend/backend.h"
#include "common.h"

typedef struct model  model;
typedef struct config config;

size_t get_available_memory(void);
size_t get_total_memory(void);

int	   model_kv_layer_pos_cap(const model *m, int n_ctx, int kv_layer);
size_t model_kv_cache_bytes_quant(const model *m, int n_ctx, kv_quant_type kv_quant, int compacted);
size_t model_pending_weight_bytes(const model *m, const config *cfg);

void model_mem_estimates_ensure(model *m);

void recommend_memory_config(const model *m, backend *kv_backend, int n_ctx, size_t avail,
							 kv_quant_type kv_quant, int is_host);

#endif
