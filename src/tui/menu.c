#include "menu.h"

#include "common.h"
#include "dialog.h"
#include "settings.h"
#include "settings_page.h"
#include "ui.h"

#include <string.h>

#define MENU_CHROME_ROWS 6
#define MENU_MIN_COL_W 30

typedef struct {
	sc_entry entries[SC_MAX_CONFIGS];
	int		 n;
	int		 sel;
	int		 vis_top;
	bool	 server_mode;
	int		 col_w;
} menu_state;

static struct {
	char name[SC_NAME_MAX];
	bool server_mode;
} memory;

void menu_select(const char *name) {
	snprintf(memory.name, sizeof(memory.name), "%s", name);
}

static void menu_measure(menu_state *m) {
	int w = MENU_MIN_COL_W;
	for (int i = 0; i < m->n; i++) {
		int name_w = ui_disp_width(m->entries[i].name) + 4;
		if (name_w > w)
			w = name_w;
	}
	m->col_w = w;
}

static void menu_reload(menu_state *m) {
	char err[SC_ERR_MAX];
	int	 n = sc_list(m->entries, SC_MAX_CONFIGS, err, sizeof(err));
	if (n < 0) {
		ui_message("Could not list configs", err, true);
		n = 0;
	}
	m->n = n;
	menu_measure(m);
	m->sel = MIN(m->sel, MAX(m->n - 1, 0));
}

static void menu_select_remembered(menu_state *m) {
	for (int i = 0; i < m->n; i++) {
		if (strcmp(m->entries[i].name, memory.name) == 0) {
			m->sel = i;
			return;
		}
	}
}

static int menu_shown(const menu_state *m) {
	return MIN(m->n, MAX(LINES - MENU_CHROME_ROWS, 1));
}

static void menu_scroll(menu_state *m) {
	if (m->n == 0)
		return;
	int shown = menu_shown(m);
	if (m->sel < m->vis_top)
		m->vis_top = m->sel;
	if (m->sel >= m->vis_top + shown)
		m->vis_top = m->sel - shown + 1;
	m->vis_top = MIN(m->vis_top, m->n - shown);
	m->vis_top = MAX(m->vis_top, 0);
}

static void menu_draw_mode_line(int y, bool server_mode) {
	static const char chat_label[]	 = "[chat]";
	static const char server_label[] = "[server]";
	int				  total			 = (int)strlen(chat_label) + 3 + (int)strlen(server_label);
	int				  x0			 = (COLS - total) / 2;
	if (x0 < 0)
		x0 = 0;
	ui_putn(y, x0, chat_label, (int)strlen(chat_label),
			server_mode ? ui_attr(UI_DIM) : (A_REVERSE | A_BOLD));
	int xs = x0 + (int)strlen(chat_label) + 3;
	ui_putn(y, xs, server_label, (int)strlen(server_label),
			server_mode ? (A_REVERSE | A_BOLD) : ui_attr(UI_DIM));
}

static void menu_draw(menu_state *m) {
	erase();

	int shown	  = menu_shown(m);
	int list_rows = MAX(shown, 1);
	int total_h	  = MENU_CHROME_ROWS - 2 + list_rows;
	int y		  = MAX((LINES - 1 - total_h) / 2, 0);
	int x		  = MAX((COLS - m->col_w) / 2, 0);

	ui_center(y++, "kappai", ui_attr(UI_TITLE));
	y++;
	menu_draw_mode_line(y++, m->server_mode);
	y++;

	if (m->n == 0)
		ui_cell(y, x, m->col_w, "(no saved configs)", ui_attr(UI_DIM));
	for (int i = 0; i < shown; i++) {
		int	   idx	= m->vis_top + i;
		chtype attr = (idx == m->sel) ? (A_REVERSE | A_BOLD) : 0;
		ui_cell(y + i, x, m->col_w, m->entries[idx].name, attr);
	}
	if (m->n > shown) {
		char pos[16];
		snprintf(pos, sizeof(pos), "(%d/%d)", m->sel + 1, m->n);
		ui_center(y + list_rows, pos, ui_attr(UI_DIM));
	}
	ui_putn(LINES - 1, 2,
			"Up/Down: select   Enter: run   E: edit   N: new   X: delete   S: settings   "
			"Left/Right: chat/server   Esc: quit",
			COLS - 4, ui_attr(UI_DIM));
	refresh();
}

static void menu_help(void) {
	ui_message("Help",
			   "Up/Down, PgUp/PgDn, Home/End: select   Enter: run\nE: edit   N: new   X: delete "
			   "(with confirm)\nS: settings page (models directory, Hugging Face downloads)\n"
			   "Left/Right: switch chat / OpenAI-server run mode\nR: rescan the config directory\n"
			   "H or F1: this help   Esc: quit",
			   false);
}

static void menu_delete(menu_state *m) {
	char msg[SC_NAME_MAX + 64];
	snprintf(msg, sizeof(msg), "Are you sure you want to delete '%s'?", m->entries[m->sel].name);
	if (tui_settings_get()->confirm_delete && !ui_confirm("Delete config", msg))
		return;
	char err[SC_ERR_MAX];
	if (sc_delete(&m->entries[m->sel], err, sizeof(err)) < 0) {
		ui_message("Delete failed", err, true);
		return;
	}
	menu_reload(m);
}

static int menu_finish(const menu_state *m, menu_result *out, menu_action action) {
	out->action		 = action;
	out->server_mode = m->server_mode;
	if (m->n > 0) {
		out->entry = m->entries[m->sel];
		menu_select(m->entries[m->sel].name);
	}
	memory.server_mode = m->server_mode;
	return 0;
}

static bool menu_has_selection(const menu_state *m) {
	return m->n > 0;
}

int menu_run(menu_result *out) {
	menu_state m;
	memset(&m, 0, sizeof(m));
	static bool started;
	if (!started) {
		memory.server_mode = tui_settings_get()->default_server;
		started			   = true;
	}
	m.server_mode = memory.server_mode;
	menu_reload(&m);
	menu_select_remembered(&m);

	for (;;) {
		menu_scroll(&m);
		menu_draw(&m);

		int ch = ui_getch();
		if (ch == KBD_INTERRUPT)
			return menu_finish(&m, out, MENU_QUIT);
		if (ch == KEY_RESIZE)
			continue;

		switch (ch) {
		case KEY_UP:
			if (menu_has_selection(&m))
				m.sel = (m.sel + m.n - 1) % m.n;
			break;
		case KEY_DOWN:
			if (menu_has_selection(&m))
				m.sel = (m.sel + 1) % m.n;
			break;
		case KEY_PPAGE:
			m.sel = MAX(m.sel - menu_shown(&m), 0);
			break;
		case KEY_NPAGE:
			m.sel = MIN(m.sel + menu_shown(&m), MAX(m.n - 1, 0));
			break;
		case KEY_HOME:
			m.sel = 0;
			break;
		case KEY_END:
			m.sel = MAX(m.n - 1, 0);
			break;
		case KEY_RIGHT:
		case KEY_LEFT:
			m.server_mode = !m.server_mode;
			break;
		case '\n':
		case KEY_ENTER:
			if (!menu_has_selection(&m)) {
				beep();
				break;
			}
			return menu_finish(&m, out, MENU_RUN);
		case 'e':
		case 'E':
			if (!menu_has_selection(&m)) {
				beep();
				break;
			}
			return menu_finish(&m, out, MENU_EDIT);
		case 'x':
		case 'X':
			if (menu_has_selection(&m))
				menu_delete(&m);
			else
				beep();
			break;
		case 'n':
		case 'N':
			return menu_finish(&m, out, MENU_NEW);
		case 's':
		case 'S':
			settings_page_run();
			break;
		case 'r':
		case 'R':
			menu_reload(&m);
			break;
		case 'h':
		case 'H':
		case KEY_F(1):
			menu_help();
			break;
		case 27:
			return menu_finish(&m, out, MENU_QUIT);
		default:
			break;
		}
	}
}
