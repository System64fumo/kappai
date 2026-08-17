#ifndef MEMCONFIG_H
#define MEMCONFIG_H

#include "common.h"
#include "kvcache.h"
#include "model.h"

size_t get_available_memory(void);
size_t get_total_memory(void);

size_t model_kv_cache_bytes_quant(const model *m, int n_ctx, kv_quant_type kv_quant);

void recommend_memory_config(const model *m, int n_ctx, size_t avail, kv_quant_type kv_quant);

#endif