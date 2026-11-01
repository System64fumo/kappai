#include "settings_page.h"

#include "dialog.h"
#include "model_picker.h"
#include "schema.h"
#include "settings.h"
#include "setup.h"
#include "ui.h"

#include <string.h>

#define PAGE_LABEL_W 24

typedef enum {
	ITEM_SECTION,
	ITEM_MODELS_DIR,
	ITEM_DEFAULT_MODE,
	ITEM_CONFIRM_DELETE,
	ITEM_MODELS,
	ITEM_SETUP,
} item_kind;

typedef struct {
	item_kind	kind;
	const char *label;
	const char *hint;
} page_item;

typedef struct {
	tui_settings cfg;
	int			 row;
	char		 status[SC_ERR_MAX];
	int			 status_kind;
} page_state;

static const page_item ITEMS[] = {
	{ITEM_SECTION, "General", ""},
	{ITEM_MODELS_DIR, "Models directory",
	 "base directory for relative model paths in configs; downloads are stored here, empty = off"},
	{ITEM_DEFAULT_MODE, "Default run mode", "run mode selected when the menu starts"},
	{ITEM_CONFIRM_DELETE, "Confirm before delete", "ask before deleting a saved config"},
	{ITEM_SETUP, "Run setup guide...", "walk through the first-run setup and quick tour again"},
	{ITEM_SECTION, "Models", ""},
	{ITEM_MODELS, "Manage models...",
	 "browse installed models, download new ones from Hugging Face, delete or add custom models"},
};
#define ITEM_COUNT ((int)(sizeof(ITEMS) / sizeof(ITEMS[0])))

static const char *const MODE_CHOICES[] = {"chat", "server"};

static bool item_selectable(int row) {
	return ITEMS[row].kind != ITEM_SECTION;
}

static void move_row(page_state *st, int dir) {
	int row = st->row;
	do {
		row = (row + dir + ITEM_COUNT) % ITEM_COUNT;
	} while (!item_selectable(row));
	st->row = row;
}

static void set_status(page_state *st, int kind, const char *msg) {
	snprintf(st->status, sizeof(st->status), "%s", msg);
	st->status_kind = kind;
}

static void commit(page_state *st) {
	char err[SC_ERR_MAX];
	if (tui_settings_save(&st->cfg, err, sizeof(err)) < 0) {
		st->cfg = *tui_settings_get();
		set_status(st, UI_ERR, err);
		beep();
		return;
	}
	set_status(st, UI_OK, "saved");
}

static void format_value(const page_state *st, int row, char *out, size_t cap) {
	const tui_settings *c = &st->cfg;
	out[0]				  = '\0';
	switch (ITEMS[row].kind) {
	case ITEM_MODELS_DIR:
		snprintf(out, cap, "%s", c->models_dir[0] ? c->models_dir : "(not set)");
		break;
	case ITEM_DEFAULT_MODE:
		snprintf(out, cap, "%s", MODE_CHOICES[c->default_server ? 1 : 0]);
		break;
	case ITEM_CONFIRM_DELETE:
		snprintf(out, cap, "%s", c->confirm_delete ? "on" : "off");
		break;
	default:
		break;
	}
}

static void draw(const page_state *st) {
	erase();
	ui_putn(0, 2, " settings ", 10, A_REVERSE | A_BOLD);
	move(1, 1);
	attron(ui_attr(UI_DIM));
	for (int i = 1; i < COLS - 1; i++)
		addch(ACS_HLINE);
	attroff(ui_attr(UI_DIM));

	int value_x = 3 + PAGE_LABEL_W + 2;
	for (int row = 0; row < ITEM_COUNT; row++) {
		int y = 3 + row;
		if (ITEMS[row].kind == ITEM_SECTION) {
			ui_putn(y, 2, ITEMS[row].label, COLS - 4, A_BOLD | ui_attr(UI_TITLE));
			continue;
		}
		bool selected = row == st->row;
		char value[SC_PATH_MAX + 16];
		format_value(st, row, value, sizeof(value));
		chtype attr = selected ? (A_REVERSE | A_BOLD) : 0;
		if (selected) {
			ui_putn(y, 1, ">", 1, A_BOLD);
			ui_fill(y, 3, 1, COLS - 5, A_REVERSE);
		}
		ui_putn(y, 3, ITEMS[row].label, PAGE_LABEL_W, selected ? attr : ui_attr(UI_DIM));
		ui_putn(y, value_x, value, COLS - value_x - 2, attr);
	}

	ui_putn(LINES - 3, 2, ITEMS[st->row].hint, COLS - 4, ui_attr(UI_DIM));
	if (st->status[0])
		ui_putn(LINES - 2, 2, st->status, COLS - 4, ui_attr(st->status_kind));
	ui_putn(LINES - 1, 2, "Up/Down: select   Enter: edit   M: manage models   Esc: back", COLS - 4,
			ui_attr(UI_DIM));
	refresh();
}

static int validate_models_dir(const char *value, char *err, size_t errlen, void *ud) {
	(void)ud;
	char expanded[TUI_PATH_MAX];
	return tui_models_dir_check(value, expanded, sizeof(expanded), err, errlen);
}

static void edit_models_dir(page_state *st) {
	char buf[TUI_PATH_MAX];
	if (ui_input("Models directory (empty = off)", st->cfg.models_dir, buf, sizeof(buf),
				 validate_models_dir, NULL, ui_path_complete) <= 0)
		return;
	char err[SC_ERR_MAX];
	if (tui_models_dir_check(buf, st->cfg.models_dir, sizeof(st->cfg.models_dir), err,
							 sizeof(err)) < 0) {
		set_status(st, UI_ERR, err);
		return;
	}
	commit(st);
}

static void edit_default_mode(page_state *st) {
	int idx = ui_select("Default run mode", MODE_CHOICES, 2, st->cfg.default_server ? 1 : 0);
	if (idx < 0)
		return;
	st->cfg.default_server = idx == 1;
	commit(st);
}

static void activate(page_state *st) {
	st->status[0] = '\0';
	switch (ITEMS[st->row].kind) {
	case ITEM_MODELS_DIR:
		edit_models_dir(st);
		break;
	case ITEM_DEFAULT_MODE:
		edit_default_mode(st);
		break;
	case ITEM_CONFIRM_DELETE:
		st->cfg.confirm_delete = !st->cfg.confirm_delete;
		commit(st);
		break;
	case ITEM_MODELS:
		model_manager_run();
		break;
	case ITEM_SETUP:
		setup_run();
		st->cfg = *tui_settings_get();
		break;
	default:
		break;
	}
}

static int models_row(void) {
	for (int i = 0; i < ITEM_COUNT; i++)
		if (ITEMS[i].kind == ITEM_MODELS)
			return i;
	return 0;
}

void settings_page_run(void) {
	page_state st;
	memset(&st, 0, sizeof(st));
	st.cfg = *tui_settings_get();
	st.row = 1;

	for (;;) {
		draw(&st);
		int ch = ui_getch();
		if (ch == KBD_INTERRUPT || ch == 27)
			return;
		switch (ch) {
		case KEY_UP:
			move_row(&st, -1);
			st.status[0] = '\0';
			break;
		case KEY_DOWN:
			move_row(&st, 1);
			st.status[0] = '\0';
			break;
		case '\n':
		case KEY_ENTER:
			activate(&st);
			break;
		case 'm':
		case 'M':
			st.row = models_row();
			activate(&st);
			break;
		default:
			break;
		}
	}
}
