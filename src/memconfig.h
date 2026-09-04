#ifndef MEMCONFIG_H
#define MEMCONFIG_H

#include "common.h"
#include "config.h"
#include "kvcache.h"
#include "model.h"

size_t get_available_memory(void);
size_t get_total_memory(void);

size_t model_kv_cache_bytes_quant(const model *m, int n_ctx, kv_quant_type kv_quant);
size_t model_total_weight_bytes(const model *m);
size_t model_resident_weight_bytes(const model *m, const config *cfg);
size_t model_pending_weight_bytes(const model *m, const config *cfg);

void recommend_memory_config(const model *m, int n_ctx, size_t avail, kv_quant_type kv_quant,
							 int is_host);

#endif