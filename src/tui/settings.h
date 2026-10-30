#ifndef TUI_SETTINGS_H
#define TUI_SETTINGS_H

#include <stdbool.h>
#include <stddef.h>

#define TUI_PATH_MAX 4096
#define TUI_CUSTOM_MAX 32
#define TUI_CUSTOM_PATH_MAX 1024

typedef struct {
	char models_dir[TUI_PATH_MAX];
	bool default_server;
	bool confirm_delete;
	bool setup_done;
	int	 n_custom;
	char custom_models[TUI_CUSTOM_MAX][TUI_CUSTOM_PATH_MAX];
} tui_settings;

const tui_settings *tui_settings_get(void);
const char		   *tui_models_dir(void);
int					tui_settings_load(char *err, size_t errlen);
int					tui_settings_save(const tui_settings *next, char *err, size_t errlen);
int	 tui_expand_path(const char *value, char *out, size_t cap, char *err, size_t errlen);
int	 tui_models_dir_check(const char *value, char *expanded, size_t cap, char *err, size_t errlen);
int	 tui_custom_check(const char *value, char *expanded, size_t cap, char *err, size_t errlen);
int	 tui_settings_add_custom(const char *path, char *err, size_t errlen);
int	 tui_settings_remove_custom(const char *path, char *err, size_t errlen);
void tui_resolve_path(const char *path, char *out, size_t cap);

#endif
