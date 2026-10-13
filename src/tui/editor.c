#include "editor.h"

#include "common.h"
#include "dialog.h"
#include "ui.h"

#include <string.h>

#define EDITOR_LABEL_W 18
#define EDITOR_NAME_LABEL "Name"
#define EDITOR_NAME_HINT "config file name, invalid characters are replaced with '_'"
#define EDITOR_DEV_MAX 16
#define EDITOR_MULTI_MAX 32

typedef struct {
	saved_config sc;
	char		 orig_name[SC_NAME_MAX];
	int			 tab;
	int			 row;
	int			 scroll;
	bool		 dirty;
	char		 status[160];
	int			 status_kind;
} editor_state;

typedef struct {
	const field_desc *f;
} field_validate_ud;

static int editor_tab_rows(int tab) {
	return SCHEMA_TABS[tab].n_fields + (tab == 0 ? 1 : 0);
}

static const field_desc *editor_field_at(int tab, int row) {
	return &SCHEMA_TABS[tab].fields[row - (tab == 0 ? 1 : 0)];
}

static int editor_validate_field(const char *value, char *err, size_t errlen, void *ud) {
	field_validate_ud *vu = (field_validate_ud *)ud;
	saved_config	   probe;
	return sc_parse_value(&probe, vu->f, value, err, errlen);
}

static int editor_rows(void) {
	int rows = LINES - 5;
	if (rows < 1)
		rows = 1;
	return rows;
}

static void editor_scroll(editor_state *st) {
	int rows = editor_rows();
	if (st->row < st->scroll)
		st->scroll = st->row;
	if (st->row >= st->scroll + rows)
		st->scroll = st->row - rows + 1;
	if (st->scroll < 0)
		st->scroll = 0;
}

static void editor_set_status(editor_state *st, int kind, const char *msg) {
	snprintf(st->status, sizeof(st->status), "%s", msg);
	st->status_kind = kind;
}

static void editor_draw_value(int y, int x, int w, const char *val, bool selected) {
	if (w < 8)
		w = 8;
	int vw = w - 4;
	if (selected) {
		ui_putn(y, x, "  ", 2, A_REVERSE);
		ui_putn(y, x + 2, val, vw, A_REVERSE);
		int used = MIN(ui_disp_width(val), vw);
		ui_fill(y, x + 2 + used, 1, vw - used, A_REVERSE);
		ui_putn(y, x + w - 2, "  ", 2, A_REVERSE);
	} else {
		ui_putn(y, x + 2, val, vw, 0);
	}
}

static void editor_draw(const editor_state *st) {
	erase();

	int tx = 2;
	for (int t = 0; t < SCHEMA_TAB_COUNT; t++) {
		char label[32];
		snprintf(label, sizeof(label), " %s ", SCHEMA_TABS[t].name);
		int	   w	= (int)strlen(label);
		chtype attr = (t == st->tab) ? (A_REVERSE | A_BOLD) : ui_attr(UI_DIM);
		ui_putn(0, tx, label, w, attr);
		tx += w + 2;
	}

	move(1, 1);
	attron(ui_attr(UI_DIM));
	for (int i = 1; i < COLS - 1; i++)
		addch(ACS_HLINE);
	attroff(ui_attr(UI_DIM));

	int value_x	 = 3 + EDITOR_LABEL_W + 2;
	int value_w	 = COLS - value_x - 2;
	int rows	 = editor_rows();
	int tab_rows = editor_tab_rows(st->tab);
	int shown	 = MIN(rows, tab_rows - st->scroll);

	for (int i = 0; i < shown; i++) {
		int				  row		  = st->scroll + i;
		int				  y			  = 2 + i;
		bool			  is_name_row = (st->tab == 0 && row == 0);
		const field_desc *f			  = is_name_row ? NULL : editor_field_at(st->tab, row);
		const char		 *label		  = is_name_row ? EDITOR_NAME_LABEL : f->label;
		bool			  selected	  = (row == st->row);

		if (selected)
			ui_putn(y, 1, ">", 1, A_BOLD);
		ui_putn(y, 3, label, EDITOR_LABEL_W, selected ? A_BOLD : ui_attr(UI_DIM));

		char		vbuf[SC_SYSTEM_MAX + 16];
		const char *val =
			is_name_row ? st->sc.name : sc_format_value(&st->sc, f, vbuf, sizeof(vbuf));
		editor_draw_value(y, value_x, value_w, val, selected);
	}

	if (st->status[0])
		ui_putn(LINES - 2, 2, st->status, COLS - 4, ui_attr(st->status_kind));

	if (st->dirty) {
		static const char dirty[] = "* unsaved changes";
		int				  dx	  = COLS - 2 - (int)strlen(dirty);
		if (dx < 2)
			dx = 2;
		if (dx >= 2 + ui_disp_width(st->status))
			ui_putn(LINES - 2, dx, dirty, COLS - 2 - dx, ui_attr(UI_DIM));
	}

	static const char key_bar[] = "Enter: edit   Tab: section   S: save   H: help   Esc: back";
	ui_putn(LINES - 1, 2, key_bar, COLS - 4, ui_attr(UI_DIM));
	refresh();
}

static void editor_edit_name(editor_state *st) {
	char buf[SC_NAME_MAX];
	int	 r =
		ui_input("Config name (a-z 0-9 - _ .)", st->sc.name, buf, sizeof(buf), NULL, NULL, NULL);
	if (r <= 0) {
		st->status[0] = '\0';
		return;
	}
	sc_sanitize_name(buf, st->sc.name, sizeof(st->sc.name));
	st->dirty = true;
	editor_set_status(st, UI_OK, "name updated");
}

static int editor_choice_count(const field_desc *f) {
	int count = 0;
	while (f->choices[count])
		count++;
	return count;
}

static void editor_edit_select(editor_state *st, const field_desc *f) {
	int n = editor_choice_count(f);
	if (n <= 0)
		return;
	int *val = (int *)sc_field_ptr(&st->sc, f);
	int	 idx = ui_select(f->label, f->choices, n, *val);
	if (idx < 0) {
		st->status[0] = '\0';
		return;
	}
	if (idx != *val) {
		*val	  = idx;
		st->dirty = true;
	}
	st->status[0] = '\0';
}

static void editor_edit_device(editor_state *st, const field_desc *f) {
	char		specs[EDITOR_DEV_MAX][SC_TEXT_MAX];
	const char *choices[EDITOR_DEV_MAX + 2];
	int			n	  = sc_enum_devices(specs, EDITOR_DEV_MAX);
	char	   *cur	  = (char *)sc_field_ptr(&st->sc, f);
	int			extra = 0;

	for (int i = 0; i < n; i++)
		choices[i] = specs[i];
	if (cur[0]) {
		bool found = false;
		for (int i = 0; i < n; i++)
			if (strcmp(specs[i], cur) == 0)
				found = true;
		if (!found)
			choices[n + extra++] = cur;
	}
	int cur_idx = 0;
	for (int i = 0; i < n + extra; i++)
		if (strcmp(choices[i], cur) == 0)
			cur_idx = i;
	static const char custom[] = "custom...";
	choices[n + extra]		   = custom;
	int total				   = n + extra + 1;

	int idx = ui_select(f->label, choices, total, cur[0] ? cur_idx : 0);
	if (idx < 0) {
		st->status[0] = '\0';
		return;
	}

	if (idx == total - 1) {
		char buf[SC_TEXT_MAX];
		int	 r = ui_input(f->label, cur, buf, sizeof(buf), NULL, NULL, NULL);
		if (r <= 0) {
			st->status[0] = '\0';
			return;
		}
		snprintf(cur, f->max_len, "%s", buf);
		st->dirty = true;
	} else if (strcmp(cur, choices[idx]) != 0) {
		snprintf(cur, f->max_len, "%s", choices[idx]);
		st->dirty = true;
	}
	st->status[0] = '\0';
}

static void editor_edit_multi(editor_state *st, const field_desc *f) {
	int n = editor_choice_count(f);
	if (n <= 0 || n > EDITOR_MULTI_MAX)
		return;
	bool  sel[EDITOR_MULTI_MAX];
	char *cur = (char *)sc_field_ptr(&st->sc, f);

	sc_multi_parse(cur, f->choices, n, f->empty_all, sel);
	if (!ui_toggle(f->label, f->choices, n, sel)) {
		st->status[0] = '\0';
		return;
	}
	sc_multi_format(cur, f->max_len, f->choices, n, f->empty_all, sel);
	st->dirty	  = true;
	st->status[0] = '\0';
}

static void editor_edit_field(editor_state *st, const field_desc *f) {
	switch (f->uik) {
	case UIK_SELECT:
		editor_edit_select(st, f);
		return;
	case UIK_DEVICE:
		editor_edit_device(st, f);
		return;
	case UIK_MULTI:
		editor_edit_multi(st, f);
		return;
	default:
		break;
	}

	if (f->type == FLD_BOOL) {
		*(bool *)sc_field_ptr(&st->sc, f) = !*(bool *)sc_field_ptr(&st->sc, f);
		st->dirty						  = true;
		st->status[0]					  = '\0';
		return;
	}

	char		buf[SC_SYSTEM_MAX + 16];
	const char *cur = sc_format_value(&st->sc, f, buf, sizeof(buf));
	char		initial[SC_SYSTEM_MAX + 16];
	snprintf(initial, sizeof(initial), "%s", cur);

	field_validate_ud ud	 = {f};
	ui_validate_fn	  v		 = (f->type == FLD_STR) ? NULL : editor_validate_field;
	size_t			  buflen = (f->type == FLD_STR) ? f->max_len : sizeof(buf);
	ui_complete_fn	  compl	 = (f->uik == UIK_PATH) ? ui_path_complete : NULL;
	int				  r		 = ui_input(f->label, initial, buf, buflen, v, &ud, compl);
	if (r <= 0) {
		st->status[0] = '\0';
		return;
	}
	char perr[UI_ERR_MAX];
	if (sc_parse_value(&st->sc, f, buf, perr, sizeof(perr)) < 0) {
		editor_set_status(st, UI_ERR, perr);
		beep();
		return;
	}
	st->dirty	  = true;
	st->status[0] = '\0';
}

static void editor_save(editor_state *st) {
	char err[SC_ERR_MAX];
	if (sc_save(&st->sc, st->orig_name, err, sizeof(err)) < 0) {
		editor_set_status(st, UI_ERR, err);
		beep();
		return;
	}
	char stem[SC_NAME_MAX];
	sc_sanitize_name(st->sc.name, stem, sizeof(stem));
	snprintf(st->sc.name, sizeof(st->sc.name), "%s", stem);
	snprintf(st->orig_name, sizeof(st->orig_name), "%s", stem);
	st->dirty = false;
	editor_set_status(st, UI_OK, "saved");
}

static bool editor_confirm_exit(editor_state *st) {
	if (!st->dirty)
		return true;
	return ui_confirm("Discard changes", "leave the editor and discard unsaved changes?");
}

static void editor_help(const editor_state *st) {
	bool			  is_name_row = (st->tab == 0 && st->row == 0);
	const field_desc *f			  = is_name_row ? NULL : editor_field_at(st->tab, st->row);
	const char		 *label		  = is_name_row ? EDITOR_NAME_LABEL : f->label;
	const char		 *hint		  = is_name_row ? EDITOR_NAME_HINT : sc_field_hint(f);

	char body[SC_SYSTEM_MAX + 192];
	snprintf(body, sizeof(body),
			 "Up/Down: select field\nPgUp/PgDn: page\nHome/End: first/last field\n"
			 "Tab: next section\nShift+Tab: previous section\nEnter: edit field\n"
			 "S: save\nH or F1: help\nEsc: back to menu\n\n%s: %s",
			 label, hint);
	ui_message("Help", body, false);
}

int editor_run(const sc_entry *entry) {
	editor_state st;
	memset(&st, 0, sizeof(st));

	char err[SC_ERR_MAX];
	if (entry) {
		if (sc_load(entry->path, &st.sc, err, sizeof(err)) < 0) {
			ui_message("Could not load config", err, true);
			return -1;
		}
		snprintf(st.orig_name, sizeof(st.orig_name), "%s", entry->name);
	} else {
		sc_defaults(&st.sc);
		sc_default_name(st.sc.name, sizeof(st.sc.name));
		snprintf(st.orig_name, sizeof(st.orig_name), "%s", st.sc.name);
		st.dirty = true;
	}

	for (;;) {
		editor_scroll(&st);
		editor_draw(&st);

		int ch = ui_getch();
		if (ch == KBD_INTERRUPT)
			return 0;
		if (ch == KEY_RESIZE)
			continue;

		switch (ch) {
		case KEY_UP:
			if (st.row > 0)
				st.row--;
			st.status[0] = '\0';
			break;
		case KEY_DOWN:
			if (st.row < editor_tab_rows(st.tab) - 1)
				st.row++;
			st.status[0] = '\0';
			break;
		case KEY_PPAGE:
			st.row -= editor_rows();
			if (st.row < 0)
				st.row = 0;
			st.status[0] = '\0';
			break;
		case KEY_NPAGE:
			st.row += editor_rows();
			if (st.row > editor_tab_rows(st.tab) - 1)
				st.row = editor_tab_rows(st.tab) - 1;
			st.status[0] = '\0';
			break;
		case KEY_HOME:
			st.row		 = 0;
			st.status[0] = '\0';
			break;
		case KEY_END:
			st.row		 = editor_tab_rows(st.tab) - 1;
			st.status[0] = '\0';
			break;
		case '\t':
			st.tab++;
			if (st.tab >= SCHEMA_TAB_COUNT)
				st.tab = 0;
			st.row		 = MIN(st.row, editor_tab_rows(st.tab) - 1);
			st.scroll	 = 0;
			st.status[0] = '\0';
			break;
		case KEY_BTAB:
			st.tab--;
			if (st.tab < 0)
				st.tab = SCHEMA_TAB_COUNT - 1;
			st.row		 = MIN(st.row, editor_tab_rows(st.tab) - 1);
			st.scroll	 = 0;
			st.status[0] = '\0';
			break;
		case '\n':
		case KEY_ENTER:
			if (st.tab == 0 && st.row == 0)
				editor_edit_name(&st);
			else
				editor_edit_field(&st, editor_field_at(st.tab, st.row));
			break;
		case 's':
		case 'S':
			editor_save(&st);
			break;
		case 'h':
		case 'H':
		case KEY_F(1):
			editor_help(&st);
			st.status[0] = '\0';
			break;
		case 27:
			if (editor_confirm_exit(&st))
				return 0;
			break;
		default:
			break;
		}
	}
}
