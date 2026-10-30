#ifndef TUI_MODEL_NAMES_H
#define TUI_MODEL_NAMES_H

#include <stdbool.h>
#include <stddef.h>

#define MODEL_TYPE_COUNT 6
#define MODEL_TYPE_OTHER (MODEL_TYPE_COUNT - 1)
#define MODEL_QUANT_MAX 16

extern const char *const MODEL_TYPE_NAMES[MODEL_TYPE_COUNT];

int	 model_type_index(const char *name);
bool model_has_gguf_suffix(const char *name);
bool model_has_safetensors_suffix(const char *name);
void model_detect_quant(const char *path, char *out, size_t cap);
void model_format_bytes(double bytes, char *out, size_t cap);
bool model_shard_key(const char *path, char *key, size_t cap, int *index);

#endif
