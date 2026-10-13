#include "menu.h"

#include "common.h"
#include "dialog.h"
#include "ui.h"

#include <string.h>

#define MENU_MAX_VISIBLE 5

typedef struct {
	sc_entry entries[SC_MAX_CONFIGS];
	int		 n;
	int		 sel;
	int		 vis_top;
	bool	 expanded;
	bool	 server_mode;
	int		 col_w;
} menu_state;

static void menu_measure(menu_state *m) {
	int w = 26;
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
	m->vis_top = 0;
	if (m->sel >= m->n + 2)
		m->sel = 0;
	if (m->sel > m->n)
		m->sel = m->n;
}

static int menu_more_row(const menu_state *m) {
	return m->n > MENU_MAX_VISIBLE ? m->n : -1;
}

static int menu_new_row(const menu_state *m) {
	return menu_more_row(m) >= 0 ? m->n + 1 : m->n;
}

static int menu_total_rows(const menu_state *m) {
	return menu_new_row(m) + 1;
}

static int menu_shown(const menu_state *m) {
	if (!m->expanded)
		return MIN(m->n, MENU_MAX_VISIBLE);
	int budget = LINES - 9;
	if (budget < 1)
		budget = 1;
	return MIN(m->n, budget);
}

static void menu_scroll(menu_state *m) {
	int shown = menu_shown(m);
	if (m->n == 0)
		return;
	if (m->sel < m->vis_top)
		m->vis_top = m->sel;
	if (m->sel >= m->vis_top + shown)
		m->vis_top = m->sel - shown + 1;
	if (m->vis_top + shown > m->n)
		m->vis_top = MAX(0, m->n - shown);
	if (m->vis_top < 0)
		m->vis_top = 0;
}

static void menu_draw_mode_line(int y, bool server_mode) {
	static const char lead[]		 = "run mode:  ";
	static const char chat_label[]	 = "[chat]";
	static const char server_label[] = "[server]";
	int total = (int)strlen(lead) + (int)strlen(chat_label) + 3 + (int)strlen(server_label);
	int x0	  = (COLS - total) / 2;
	if (x0 < 0)
		x0 = 0;
	ui_putn(y, x0, lead, (int)strlen(lead), ui_attr(UI_DIM));
	int xc = x0 + (int)strlen(lead);
	ui_putn(y, xc, chat_label, (int)strlen(chat_label),
			server_mode ? ui_attr(UI_DIM) : (A_REVERSE | A_BOLD));
	int xs = xc + (int)strlen(chat_label) + 3;
	ui_putn(y, xs, server_label, (int)strlen(server_label),
			server_mode ? (A_REVERSE | A_BOLD) : ui_attr(UI_DIM));
}

static void menu_draw(menu_state *m) {
	erase();

	int shown	 = menu_shown(m);
	int has_more = menu_more_row(m) >= 0 ? 1 : 0;
	int total_h	 = 8 + shown + has_more;
	int y		 = (LINES - total_h) / 2;
	if (y < 0)
		y = 0;
	int x = (COLS - m->col_w) / 2;
	if (x < 0)
		x = 0;

	ui_center(y++, "kappai", ui_attr(UI_TITLE));
	y++;
	menu_draw_mode_line(y++, m->server_mode);
	y++;
	ui_cell(y++, x, m->col_w, "saved configurations", ui_attr(UI_DIM));

	for (int i = 0; i < shown; i++) {
		int idx = m->vis_top + i;
		if (idx >= m->n)
			break;
		chtype attr = (idx == m->sel) ? (A_REVERSE | A_BOLD) : 0;
		ui_cell(y++, x, m->col_w, m->entries[idx].name, attr);
	}

	if (m->n == 0)
		ui_cell(y++, x, m->col_w, "(no saved configs)", ui_attr(UI_DIM));

	if (has_more) {
		const char *label = m->expanded ? "[ view less ]" : "[ view more ]";
		chtype		attr  = (m->sel == m->n) ? (A_REVERSE | A_BOLD) : ui_attr(UI_DIM);
		ui_cell(y++, x, m->col_w, label, attr);
	}

	chtype new_attr = (m->sel == menu_new_row(m)) ? (A_REVERSE | A_BOLD) : ui_attr(UI_OK);
	ui_cell(y++, x, m->col_w, "[ new config ]", new_attr);

	y++;
	ui_center(y,
			  "Up/Down: select   Enter: run   E: edit   D: delete   N: new   Tab: chat/server"
			  "   Q: quit",
			  ui_attr(UI_DIM));
	refresh();
}

static void menu_help(void) {
	ui_message("Help",
			   "Up/Down: select   Enter: run\nE: edit   N: new   D: delete (with confirm)\n"
			   "Tab: switch chat / OpenAI-server run mode\nR: rescan the config directory\n"
			   "H or F1: this help   Q: quit",
			   false);
}

static void menu_delete(menu_state *m) {
	char msg[SC_PATH_MAX + 96];
	snprintf(msg, sizeof(msg), "delete '%s'?\nthe file below will be removed:\n%s",
			 m->entries[m->sel].name, m->entries[m->sel].path);
	if (!ui_confirm("Delete config", msg))
		return;
	char err[SC_ERR_MAX];
	if (sc_delete(&m->entries[m->sel], err, sizeof(err)) < 0) {
		ui_message("Delete failed", err, true);
		return;
	}
	if (m->sel > 0)
		m->sel--;
	menu_reload(m);
}

int menu_run(menu_result *out) {
	menu_state m;
	memset(&m, 0, sizeof(m));
	m.sel = 0;
	menu_reload(&m);

	for (;;) {
		menu_scroll(&m);
		menu_draw(&m);

		int ch = ui_getch();
		if (ch == KBD_INTERRUPT) {
			out->action = MENU_QUIT;
			return 0;
		}
		if (ch == KEY_RESIZE)
			continue;

		int total = menu_total_rows(&m);
		switch (ch) {
		case KEY_UP:
			m.sel = (m.sel + total - 1) % total;
			break;
		case KEY_DOWN:
			m.sel = (m.sel + 1) % total;
			break;
		case '\t':
		case KEY_BTAB:
			m.server_mode = !m.server_mode;
			break;
		case '\n':
		case KEY_ENTER:
			if (m.sel < m.n) {
				out->action		 = MENU_RUN;
				out->server_mode = m.server_mode;
				out->entry		 = m.entries[m.sel];
				return 0;
			}
			if (m.sel == menu_more_row(&m)) {
				m.expanded = !m.expanded;
				m.vis_top  = 0;
			} else {
				out->action		 = MENU_NEW;
				out->server_mode = m.server_mode;
				return 0;
			}
			break;
		case 'e':
		case 'E':
			if (m.sel < m.n) {
				out->action		 = MENU_EDIT;
				out->server_mode = m.server_mode;
				out->entry		 = m.entries[m.sel];
				return 0;
			}
			beep();
			break;
		case 'd':
		case 'D':
			if (m.sel < m.n)
				menu_delete(&m);
			else
				beep();
			break;
		case 'n':
		case 'N':
			out->action		 = MENU_NEW;
			out->server_mode = m.server_mode;
			return 0;
		case 'r':
		case 'R':
			menu_reload(&m);
			break;
		case 'h':
		case 'H':
		case KEY_F(1):
			menu_help();
			break;
		case 'q':
		case 'Q':
		case 27:
			out->action = MENU_QUIT;
			return 0;
		default:
			break;
		}
	}
}
