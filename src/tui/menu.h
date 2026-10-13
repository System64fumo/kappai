#ifndef TUI_MENU_H
#define TUI_MENU_H

#include "schema.h"

#include <stdbool.h>

typedef enum {
	MENU_NONE = 0,
	MENU_RUN,
	MENU_EDIT,
	MENU_NEW,
	MENU_QUIT,
} menu_action;

typedef struct {
	menu_action action;
	bool		server_mode;
	sc_entry	entry;
} menu_result;

int menu_run(menu_result *out);

#endif
