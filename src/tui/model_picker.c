#define _GNU_SOURCE
#include "model_picker.h"

#include "common.h"
#include "dialog.h"
#include "hf.h"
#include "model_library.h"
#include "schema.h"
#include "settings.h"
#include "ui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BROWSER_PANEL_W 40
#define BROWSER_PANEL_MIN_COLS 96
#define BROWSER_LIST_TOP 3
#define BROWSER_TAB_NAME_MAX 24
#define BROWSER_MAX_TABS (MODEL_TYPE_COUNT + 2)

enum { TAB_ALL = -1, TAB_CUSTOM = -2 };

typedef struct {
	char label[BROWSER_TAB_NAME_MAX];
	int	 type;
} browser_tab;

typedef struct {
	model_lib	lib;
	browser_tab tabs[BROWSER_MAX_TABS];
	int			n_tabs;
	int			tab;
	int		   *visible;
	int			n_visible;
	int			sel;
	int			top;
	bool		manage;
	const char *current;
	char		status[SC_ERR_MAX];
	int			status_kind;
} browser;

static bool tab_matches(const browser_tab *tab, const lib_model *m) {
	if (tab->type == TAB_ALL)
		return true;
	if (tab->type == TAB_CUSTOM)
		return m->custom;
	return !m->custom && m->type == tab->type;
}

static bool tab_has_models(const browser *b, int type) {
	browser_tab probe = {.type = type};
	for (int i = 0; i < b->lib.count; i++)
		if (tab_matches(&probe, &b->lib.items[i]))
			return true;
	return false;
}

static void add_tab(browser *b, const char *name, int type) {
	browser_tab *tab = &b->tabs[b->n_tabs++];
	snprintf(tab->label, sizeof(tab->label), "%s", name);
	tab->type = type;
}

static void build_tabs(browser *b) {
	b->n_tabs = 0;
	for (int t = 0; t < MODEL_TYPE_COUNT; t++)
		if (tab_has_models(b, t))
			add_tab(b, MODEL_TYPE_NAMES[t], t);
	if (tab_has_models(b, TAB_CUSTOM))
		add_tab(b, "custom", TAB_CUSTOM);
	add_tab(b, "all", TAB_ALL);
}

static void build_visible(browser *b) {
	const browser_tab *tab = &b->tabs[b->tab];
	b->n_visible		   = 0;
	for (int i = 0; i < b->lib.count; i++)
		if (tab_matches(tab, &b->lib.items[i]))
			b->visible[b->n_visible++] = i;
	b->sel = MIN(b->sel, MAX(b->n_visible - 1, 0));
}

static int find_tab(const browser *b, int type) {
	for (int t = 0; t < b->n_tabs; t++)
		if (b->tabs[t].type == type)
			return t;
	return -1;
}

static void reload(browser *b) {
	int previous_type = b->n_tabs ? b->tabs[b->tab].type : TAB_ALL;
	model_lib_free(&b->lib);
	model_lib_load(&b->lib);
	build_tabs(b);
	int tab = find_tab(b, previous_type);
	b->tab	= tab >= 0 ? tab : b->n_tabs - 1;
	build_visible(b);
}

static void select_initial(browser *b, const char *suggest_type) {
	int type = suggest_type ? model_type_index(suggest_type) : -1;
	int tab	 = type >= 0 ? find_tab(b, type) : -1;
	if (tab >= 0)
		b->tab = tab;
	build_visible(b);
	for (int i = 0; i < b->n_visible; i++) {
		if (b->current && strcmp(b->lib.items[b->visible[i]].path, b->current) == 0) {
			b->sel = i;
			return;
		}
	}
}

static void set_status(browser *b, int kind, const char *msg) {
	snprintf(b->status, sizeof(b->status), "%s", msg);
	b->status_kind = kind;
}

static bool panel_visible(void) {
	return COLS >= BROWSER_PANEL_MIN_COLS;
}

static int list_rows(void) {
	return MAX(LINES - BROWSER_LIST_TOP - 2, 1);
}

static int list_width(void) {
	return panel_visible() ? COLS - BROWSER_PANEL_W - 4 : COLS - 4;
}

static void keep_visible(browser *b) {
	int rows = list_rows();
	if (b->sel < b->top)
		b->top = b->sel;
	if (b->sel >= b->top + rows)
		b->top = b->sel - rows + 1;
	b->top = MAX(MIN(b->top, b->n_visible - rows), 0);
}

static void draw_tabs(const browser *b) {
	int tx = 2;
	for (int t = 0; t < b->n_tabs; t++) {
		char label[BROWSER_TAB_NAME_MAX + 4];
		snprintf(label, sizeof(label), " %s ", b->tabs[t].label);
		ui_putn(0, tx, label, (int)strlen(label),
				t == b->tab ? (A_REVERSE | A_BOLD) : ui_attr(UI_DIM));
		tx += (int)strlen(label) + 2;
	}
	move(1, 1);
	attron(ui_attr(UI_DIM));
	for (int i = 1; i < COLS - 1; i++)
		addch(ACS_HLINE);
	attroff(ui_attr(UI_DIM));
}

static void format_label(const browser *b, const lib_model *m, char *out, size_t cap) {
	char size[32];
	model_format_bytes((double)m->size, size, sizeof(size));
	const char *mark = !b->manage && b->current && strcmp(m->path, b->current) == 0 ? "* " : "  ";
	snprintf(out, cap, "%s%-10s %s%s", mark, m->missing ? "missing" : size, m->path,
			 m->parts > 1 ? "  (multi-part)" : "");
}

static void draw_list(const browser *b) {
	int rows = MIN(list_rows(), b->n_visible - b->top);
	for (int i = 0; i < rows; i++) {
		int	 idx	  = b->top + i;
		int	 y		  = BROWSER_LIST_TOP + i;
		bool selected = idx == b->sel;
		char label[LIB_PATH_MAX + 64];
		format_label(b, &b->lib.items[b->visible[idx]], label, sizeof(label));
		if (selected) {
			ui_putn(y, 1, ">", 1, A_BOLD);
			ui_fill(y, 3, 1, list_width(), A_REVERSE);
		}
		ui_putn(y, 3, label, list_width(), selected ? (A_REVERSE | A_BOLD) : 0);
	}
	if (b->n_visible == 0)
		ui_putn(BROWSER_LIST_TOP, 3,
				b->manage ? "(no models yet; press D to download or C to add a custom one)"
						  : "(no models here; press D to download or C to add a custom one)",
				list_width(), ui_attr(UI_DIM));
}

static int put_field(int y, int x, int w, const char *label, const char *value) {
	char line[SC_PATH_MAX + 32];
	snprintf(line, sizeof(line), "%-9s %s", label, value);
	ui_draw_wrapped(y, x, w, line, 0);
	return y + ui_wrap_count(line, w);
}

static void draw_panel(const browser *b) {
	int px = COLS - BROWSER_PANEL_W;
	attron(ui_attr(UI_DIM));
	for (int y = 2; y <= LINES - 3; y++)
		mvaddch(y, px - 2, ACS_VLINE);
	attroff(ui_attr(UI_DIM));
	if (b->n_visible == 0)
		return;
	int				 w = BROWSER_PANEL_W - 2;
	const lib_model *m = &b->lib.items[b->visible[b->sel]];
	char			 size[48], abs[SC_PATH_MAX];
	model_format_bytes((double)m->size, size, sizeof(size));
	if (m->parts > 1)
		snprintf(size + strlen(size), sizeof(size) - strlen(size), " (%d parts)", m->parts);
	model_lib_absolute(m, abs, sizeof(abs));
	const char *base = strrchr(m->path, '/');
	const char *name = base ? base + 1 : m->path;
	ui_draw_wrapped(2, px, w, name, A_BOLD);
	int y = 2 + ui_wrap_count(name, w) + 1;
	y	  = put_field(y, px, w, "Type", m->custom ? "custom" : MODEL_TYPE_NAMES[m->type]);
	y	  = put_field(y, px, w, "Arch", m->arch[0] ? m->arch : "-");
	y	  = put_field(y, px, w, "Quant", m->quant);
	y	  = put_field(y, px, w, "Size", m->missing ? "file is missing" : size);
	y	  = put_field(y, px, w, "Path", abs) + 1;
	if (m->custom)
		ui_draw_wrapped(
			y, px, w, "custom model: outside the models directory; removing it only unregisters it",
			ui_attr(UI_DIM));
}

static const char *key_bar(const browser *b) {
	return b->manage
			   ? "D: download   X: delete   C: add custom model   Tab: switch tab   Esc: back"
			   : "Enter: select   C: custom model   D: download   Tab: switch tab   Esc: cancel";
}

static void draw(const browser *b, const char *title) {
	erase();
	draw_tabs(b);
	char header[SC_PATH_MAX + 96];
	if (tui_models_dir()[0])
		snprintf(header, sizeof(header), "%s   (%d models, library: %s)", title, b->n_visible,
				 tui_models_dir());
	else
		snprintf(header, sizeof(header),
				 "%s   (no models directory set; configure one in settings)", title);
	ui_putn(2, 3, header, COLS - 6, A_BOLD);
	draw_list(b);
	if (panel_visible())
		draw_panel(b);
	if (b->status[0])
		ui_putn(LINES - 2, 2, b->status, COLS - 4, ui_attr(b->status_kind));
	ui_putn(LINES - 1, 2, key_bar(b), COLS - 4, ui_attr(UI_DIM));
	refresh();
}

static void switch_tab(browser *b, int dir) {
	b->tab = (b->tab + dir + b->n_tabs) % b->n_tabs;
	b->sel = 0;
	b->top = 0;
	build_visible(b);
}

static void move_selection(browser *b, int delta) {
	b->sel		 = MAX(MIN(b->sel + delta, b->n_visible - 1), 0);
	b->status[0] = '\0';
}

static int validate_custom(const char *value, char *err, size_t errlen, void *ud) {
	(void)ud;
	char expanded[SC_PATH_MAX];
	return tui_custom_check(value, expanded, sizeof(expanded), err, errlen);
}

static bool add_custom(browser *b, char *added, size_t cap) {
	char buf[SC_PATH_MAX], expanded[SC_PATH_MAX], err[SC_ERR_MAX];
	if (ui_input("Custom model path (.gguf file)", "", buf, sizeof(buf), validate_custom, NULL,
				 ui_path_complete) <= 0)
		return false;
	if (tui_custom_check(buf, expanded, sizeof(expanded), err, sizeof(err)) < 0 ||
		tui_settings_add_custom(expanded, err, sizeof(err)) < 0) {
		set_status(b, UI_ERR, err);
		return false;
	}
	snprintf(added, cap, "%s", expanded);
	reload(b);
	set_status(b, UI_OK, "custom model added");
	return true;
}

static void delete_selected(browser *b) {
	if (b->n_visible == 0)
		return;
	const lib_model *m = &b->lib.items[b->visible[b->sel]];
	char			 msg[SC_PATH_MAX + 160], size[32], err[SC_ERR_MAX];
	model_format_bytes((double)m->size, size, sizeof(size));
	if (m->custom)
		snprintf(msg, sizeof(msg),
				 "remove this custom model from the list?\n%s\nthe file itself stays on disk",
				 m->path);
	else
		snprintf(msg, sizeof(msg), "delete this model from disk?\n%s\n%s%s", m->path, size,
				 m->parts > 1 ? ", all parts" : "");
	bool custom = m->custom;
	if (!ui_confirm(custom ? "Remove custom model" : "Delete model", msg))
		return;
	if (model_lib_delete(m, err, sizeof(err)) < 0) {
		set_status(b, UI_ERR, err);
		return;
	}
	reload(b);
	set_status(b, UI_OK, custom ? "removed from the list" : "model deleted");
}

static model_pick_result run_loop(browser *b, const char *title, char *out, size_t cap) {
	for (;;) {
		keep_visible(b);
		draw(b, title);
		int ch = ui_getch();
		switch (ch) {
		case KBD_INTERRUPT:
		case 27:
			return MODEL_PICK_CANCEL;
		case KEY_UP:
			move_selection(b, -1);
			break;
		case KEY_DOWN:
			move_selection(b, 1);
			break;
		case KEY_PPAGE:
			move_selection(b, -list_rows());
			break;
		case KEY_NPAGE:
			move_selection(b, list_rows());
			break;
		case KEY_HOME:
			move_selection(b, -LIB_MAX_MODELS);
			break;
		case KEY_END:
			move_selection(b, LIB_MAX_MODELS);
			break;
		case '	':
			switch_tab(b, 1);
			break;
		case KEY_BTAB:
			switch_tab(b, -1);
			break;
		case 'c':
		case 'C':
			if (add_custom(b, out, cap) && !b->manage)
				return MODEL_PICK_SELECTED;
			break;
		case 'd':
		case 'D':
			hf_page_run();
			reload(b);
			break;
		case 'x':
		case 'X':
		case KEY_DC:
			if (b->manage)
				delete_selected(b);
			break;
		case '\n':
		case KEY_ENTER:
			if (b->manage || b->n_visible == 0)
				break;
			snprintf(out, cap, "%s", b->lib.items[b->visible[b->sel]].path);
			return MODEL_PICK_SELECTED;
		default:
			break;
		}
	}
}

static model_pick_result browse(bool manage, const char *title, const char *suggest_type,
								const char *current, char *out, size_t cap) {
	browser b;
	memset(&b, 0, sizeof(b));
	b.visible = xcalloc(LIB_MAX_MODELS, sizeof(int));
	b.manage  = manage;
	b.current = current;
	model_lib_load(&b.lib);
	build_tabs(&b);
	b.tab = b.n_tabs - 1;
	select_initial(&b, suggest_type);

	model_pick_result r = run_loop(&b, title, out, cap);
	model_lib_free(&b.lib);
	free(b.visible);
	return r;
}

model_pick_result model_picker_run(const char *title, const char *suggest_type, const char *current,
								   char *out, size_t cap) {
	return browse(false, title, suggest_type, current, out, cap);
}

void model_manager_run(void) {
	char unused[SC_PATH_MAX];
	browse(true, "Models", NULL, NULL, unused, sizeof(unused));
}
