#ifndef TUI_DIALOG_H
#define TUI_DIALOG_H

#include "ui.h"

#include <stdbool.h>
#include <stddef.h>

#define UI_ERR_MAX 256
#define UI_MATCH_MAX 96

typedef int (*ui_validate_fn)(const char *value, char *err, size_t errlen, void *ud);

typedef struct {
	char insert[UI_MATCH_MAX];
} ui_completion;

typedef int (*ui_complete_fn)(const char *buf, size_t pos, ui_completion *out);

int	 ui_input(const char *title, const char *initial, char *buf, size_t buflen, ui_validate_fn v,
			  void *ud, ui_complete_fn compl);
int	 ui_path_complete(const char *buf, size_t pos, ui_completion *out);
int	 ui_model_path_complete(const char *buf, size_t pos, ui_completion *out);
int	 ui_select(const char *title, const char *const choices[], int n, int cur);
bool ui_toggle(const char *title, const char *const choices[], int n, bool *sel);
bool ui_confirm(const char *title, const char *msg);
void ui_message(const char *title, const char *msg, bool error);

#endif
