#ifndef TUI_MODEL_LIBRARY_H
#define TUI_MODEL_LIBRARY_H

#include "model_names.h"

#include <stdbool.h>
#include <stddef.h>

#define LIB_PATH_MAX 512
#define LIB_ARCH_MAX 48
#define LIB_MAX_MODELS 2048

typedef struct {
	char   path[LIB_PATH_MAX];
	char   arch[LIB_ARCH_MAX];
	char   quant[MODEL_QUANT_MAX];
	size_t size;
	int	   parts;
	int	   type;
	bool   custom;
	bool   missing;
} lib_model;

typedef struct {
	lib_model *items;
	int		   count;
} model_lib;

void model_lib_load(model_lib *lib);
void model_lib_free(model_lib *lib);
void model_lib_absolute(const lib_model *m, char *out, size_t cap);
int	 model_lib_delete(const lib_model *m, char *err, size_t errlen);

#endif
