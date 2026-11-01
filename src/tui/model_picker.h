#ifndef TUI_MODEL_PICKER_H
#define TUI_MODEL_PICKER_H

#include <stddef.h>

typedef enum {
	MODEL_PICK_CANCEL,
	MODEL_PICK_SELECTED,
} model_pick_result;

model_pick_result model_picker_run(const char *title, const char *suggest_type, const char *current,
								   char *out, size_t cap);
void			  model_manager_run(void);

#endif
