#include "editor.h"

#include "backend/backend.h"
#include "common.h"
#include "dialog.h"
#include "hcs.h"
#include "ui.h"

#include <string.h>

#define EDITOR_LABEL_W 18
#define EDITOR_NAME_LABEL "Name"
#define EDITOR_NAME_HINT "config file name, invalid characters are replaced with '_'"
#define EDITOR_DEV_MAX 16
#define EDITOR_HCS_BANNER_LINES 2
#define EDITOR_HCS_PANEL_W 34
#define EDITOR_HCS_PANEL_MIN_COLS 100
#define EDITOR_HCS_TAG "[W.I.P]"
#define EDITOR_MULTI_MAX 32

typedef struct {
	saved_config sc;
	char		 orig_name[SC_NAME_MAX];
	int			 tab;
	int			 row;
	int			 scroll;
	bool		 dirty;
	char		 probe_path[SC_PATH_MAX];
	bool		 probed;
	bool		 probe_ok;
	int			 n_layers;
	int			 n_experts;
	char		 status[160];
	int			 status_kind;
} editor_state;

typedef struct {
	const field_desc *f;
} field_validate_ud;

typedef enum {
	HCS_REF_ENABLE,
	HCS_REF_SPACER,
	HCS_REF_ADD,
	HCS_REF_DEVICE,
	HCS_REF_TASKS,
	HCS_REF_LAYERS,
	HCS_REF_EXPERTS,
} hcs_ref_kind;

typedef struct {
	hcs_ref_kind kind;
	int			 device;
} hcs_ref;

#define HCS_REF_FIELD_COUNT 4

static const char EDITOR_HCS_ENABLE_HINT[] =
	"off: the model tab device runs everything; on: place layers and experts per device here";
static const char EDITOR_MODEL_DEVICE_HINT[] =
	"compute device to run on; with HCS enabled it is set in the hcs tab";

static const char *const HCS_FIELD_HINTS[HCS_REF_FIELD_COUNT] = {
	"compute device, <backend>_<n>",
	"phases this device runs for the weights it holds",
	"transformer layers held, a number or '*' for the remainder (one '*' per config)",
	"per-layer expert cache size, number, '*' or empty for the default",
};

static int editor_tab_count(void) {
	return SCHEMA_TAB_COUNT + 1;
}

static bool editor_tab_is_hcs(int tab) {
	return tab == SCHEMA_TAB_COUNT;
}

static const schema_tab *editor_schema_tab(int tab) {
	return &SCHEMA_TABS[tab];
}

static int editor_header_lines(const editor_state *st, int tab) {
	if (!editor_tab_is_hcs(tab))
		return 0;
	return EDITOR_HCS_BANNER_LINES;
}

static bool editor_model_is_dense(const editor_state *st) {
	return st->probe_ok && st->n_experts == 0;
}

static int editor_hcs_fields_per_device(const editor_state *st) {
	return editor_model_is_dense(st) ? 3 : 4;
}

static bool editor_hcs_can_add(const editor_state *st) {
	return st->sc.hcs.n_devices < HCS_MAX_DEVICES;
}

static int editor_hcs_total_rows(const editor_state *st) {
	int stride = editor_hcs_fields_per_device(st) + 1;
	return 2 + st->sc.hcs.n_devices * stride + (editor_hcs_can_add(st) ? 1 : -1);
}

static hcs_ref editor_hcs_ref(const editor_state *st, int row) {
	if (row == 0)
		return (hcs_ref){HCS_REF_ENABLE, -1};
	int per	   = editor_hcs_fields_per_device(st);
	int stride = per + 1;
	int r	   = row - 2;
	if (r < 0)
		return (hcs_ref){HCS_REF_SPACER, -1};
	if (r < st->sc.hcs.n_devices * stride) {
		int k = r % stride;
		if (k == per)
			return (hcs_ref){HCS_REF_SPACER, -1};
		return (hcs_ref){(hcs_ref_kind)(HCS_REF_DEVICE + k), r / stride};
	}
	if (r == st->sc.hcs.n_devices * stride && editor_hcs_can_add(st))
		return (hcs_ref){HCS_REF_ADD, -1};
	return (hcs_ref){HCS_REF_SPACER, -1};
}

static int editor_tab_rows(const editor_state *st, int tab) {
	if (editor_tab_is_hcs(tab))
		return st->sc.hcs.enabled ? editor_hcs_total_rows(st) : 1;
	return editor_schema_tab(tab)->n_fields + (tab == 0 ? 1 : 0) +
		   (tab == 0 && !st->sc.hcs.enabled ? 1 : 0);
}

static int editor_last_row(const editor_state *st, int tab) {
	return MAX(editor_tab_rows(st, tab) - 1, 0);
}

static bool editor_is_name_row(int tab, int row) {
	return tab == 0 && row == 0;
}

static bool editor_is_model_device_row(const editor_state *st, int tab, int row) {
	return tab == 0 && !st->sc.hcs.enabled && row == editor_schema_tab(0)->n_fields + 1;
}

static const field_desc *editor_field_at(int tab, int row) {
	return &editor_schema_tab(tab)->fields[row - (tab == 0 ? 1 : 0)];
}

static bool editor_row_is_spacer(const editor_state *st, int tab, int row) {
	return editor_tab_is_hcs(tab) && st->sc.hcs.enabled &&
		   editor_hcs_ref(st, row).kind == HCS_REF_SPACER;
}

static void editor_hcs_row_text(const editor_state *st, int row, char *label, size_t label_cap,
								char *val, size_t val_cap) {
	const hcs_config *h	  = &st->sc.hcs;
	hcs_ref			  ref = editor_hcs_ref(st, row);
	label[0]			  = '\0';
	val[0]				  = '\0';
	if (ref.kind == HCS_REF_ENABLE) {
		snprintf(label, label_cap, "Enable HCS");
		snprintf(val, val_cap, "%s", h->enabled ? "on" : "off");
		return;
	}
	if (ref.kind == HCS_REF_ADD) {
		snprintf(label, label_cap, "+ Add device");
		return;
	}
	if (ref.kind == HCS_REF_SPACER)
		return;

	const hcs_device *d = &h->devices[ref.device];
	char			  count[16];
	switch (ref.kind) {
	case HCS_REF_DEVICE:
		snprintf(label, label_cap, "Device %d", ref.device + 1);
		snprintf(val, val_cap, "%s", d->name);
		break;
	case HCS_REF_TASKS:
		snprintf(label, label_cap, "  Tasks");
		hcs_format_tasks(d->tasks, val, val_cap);
		break;
	case HCS_REF_LAYERS:
		snprintf(label, label_cap, "  Layers");
		snprintf(val, val_cap, "%s",
				 d->layers == HCS_COUNT_UNSET ? "(not set)"
											  : hcs_format_count(d->layers, count, sizeof(count)));
		break;
	default:
		snprintf(label, label_cap, "  Experts");
		snprintf(val, val_cap, "%s",
				 d->experts == HCS_COUNT_UNSET
					 ? "(default)"
					 : hcs_format_count(d->experts, count, sizeof(count)));
		break;
	}
}

static int editor_validate_field(const char *value, char *err, size_t errlen, void *ud) {
	field_validate_ud *vu = (field_validate_ud *)ud;
	saved_config	   probe;
	return sc_parse_value(&probe, vu->f, value, err, errlen);
}

static int editor_rows(const editor_state *st, int tab) {
	int rows = LINES - 5 - editor_header_lines(st, tab);
	if (rows < 1)
		rows = 1;
	return rows;
}

static void editor_refresh_probe(editor_state *st) {
	if (st->probed && strcmp(st->probe_path, st->sc.model) == 0)
		return;
	snprintf(st->probe_path, sizeof(st->probe_path), "%s", st->sc.model);
	st->probed	 = true;
	st->probe_ok = hcs_probe_model(st->sc.model, &st->n_layers, &st->n_experts) == 0;
	if (!editor_model_is_dense(st))
		return;
	for (int i = 0; i < st->sc.hcs.n_devices; i++) {
		if (st->sc.hcs.devices[i].experts == HCS_COUNT_UNSET)
			continue;
		st->sc.hcs.devices[i].experts = HCS_COUNT_UNSET;
		st->dirty					  = true;
	}
}

static void editor_scroll(editor_state *st) {
	int rows = editor_rows(st, st->tab);
	if (st->row < st->scroll)
		st->scroll = st->row;
	if (st->row >= st->scroll + rows)
		st->scroll = st->row - rows + 1;
	if (st->scroll < 0)
		st->scroll = 0;
}

static void editor_move_row(editor_state *st, int target, int dir) {
	int last = editor_last_row(st, st->tab);
	target	 = MIN(MAX(target, 0), last);
	while (target >= 0 && target <= last && editor_row_is_spacer(st, st->tab, target))
		target += dir;
	if (target >= 0 && target <= last)
		st->row = target;
}

static void editor_snap_row(editor_state *st) {
	st->row = MIN(st->row, editor_last_row(st, st->tab));
	if (!editor_row_is_spacer(st, st->tab, st->row))
		return;
	editor_move_row(st, st->row + 1, 1);
	if (editor_row_is_spacer(st, st->tab, st->row))
		editor_move_row(st, st->row - 1, -1);
}

static void editor_switch_tab(editor_state *st, int tab) {
	st->tab		  = tab;
	st->scroll	  = 0;
	st->status[0] = '\0';
	editor_snap_row(st);
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

static void editor_draw_tabs(const editor_state *st) {
	int tx = 2;
	for (int t = 0; t < editor_tab_count(); t++) {
		char label[48];
		snprintf(label, sizeof(label), " %s ",
				 editor_tab_is_hcs(t) ? "hcs" : editor_schema_tab(t)->name);
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
}

static bool editor_hcs_panel_visible(const editor_state *st) {
	return editor_tab_is_hcs(st->tab) && st->sc.hcs.enabled && COLS >= EDITOR_HCS_PANEL_MIN_COLS;
}

static void editor_draw_rows(const editor_state *st) {
	int value_x	 = 3 + EDITOR_LABEL_W + 2;
	int right	 = editor_hcs_panel_visible(st) ? COLS - EDITOR_HCS_PANEL_W - 3 : COLS - 2;
	int value_w	 = right - value_x;
	int rows	 = editor_rows(st, st->tab);
	int tab_rows = editor_tab_rows(st, st->tab);
	int shown	 = MIN(rows, tab_rows - st->scroll);
	int top		 = 2 + editor_header_lines(st, st->tab);

	for (int i = 0; i < shown; i++) {
		int row = st->scroll + i;
		int y	= top + i;
		if (editor_row_is_spacer(st, st->tab, row))
			continue;
		bool			  is_name_row = editor_is_name_row(st->tab, row);
		bool			  selected	  = (row == st->row);
		const field_desc *f			  = NULL;
		char			  label[EDITOR_LABEL_W + 8];
		char			  vbuf[SC_SYSTEM_MAX + 16];
		const char		 *val;

		if (editor_tab_is_hcs(st->tab)) {
			editor_hcs_row_text(st, row, label, sizeof(label), vbuf, sizeof(vbuf));
			val = vbuf;
		} else if (is_name_row) {
			snprintf(label, sizeof(label), "%s", EDITOR_NAME_LABEL);
			val = st->sc.name;
		} else if (editor_is_model_device_row(st, st->tab, row)) {
			snprintf(label, sizeof(label), "Device");
			val = st->sc.hcs.devices[0].name;
		} else {
			f = editor_field_at(st->tab, row);
			snprintf(label, sizeof(label), "%s", f->label);
			val = sc_format_value(&st->sc, f, vbuf, sizeof(vbuf));
		}

		if (selected)
			ui_putn(y, 1, ">", 1, A_BOLD);
		ui_putn(y, 3, label, EDITOR_LABEL_W, selected ? A_BOLD : ui_attr(UI_DIM));
		editor_draw_value(y, value_x, value_w, val, selected);
	}
}

static void editor_hcs_summary(const editor_state *st, int assigned, char *out, size_t cap) {
	int n = st->n_layers;
	if (assigned == n)
		snprintf(out, cap, "all satisfied");
	else if (assigned < n)
		snprintf(out, cap, "%d unassigned", n - assigned);
	else
		snprintf(out, cap, "%d over-assigned", assigned - n);
}

static void editor_draw_hcs_compact(const editor_state *st) {
	if (!st->probe_ok || st->n_layers <= 0) {
		ui_putn(3, 3, "layers: unknown (model unreadable)", COLS - 6, ui_attr(UI_DIM));
		return;
	}
	int	 held[HCS_MAX_DEVICES];
	int	 assigned = hcs_resolve_layers(&st->sc.hcs, st->n_layers, held);
	char summary[48];
	char line[128];
	editor_hcs_summary(st, assigned, summary, sizeof(summary));
	snprintf(line, sizeof(line), "layers: %d total, %d assigned, %s", st->n_layers,
			 MIN(assigned, st->n_layers), summary);
	ui_putn(3, 3, line, COLS - 6, assigned == st->n_layers ? ui_attr(UI_OK) : ui_attr(UI_ERR));
}

static void editor_draw_hcs_layer_grid(const int *held, const int *order, int n_order, int n_layers,
									   int x, int y, int width, int max_rows) {
	static const char symbols[] = "#=+~";
	int				  cells		= MIN(n_layers, width * max_rows);
	int				  owner_end[HCS_MAX_DEVICES];
	int				  cumulative = 0;
	for (int i = 0; i < n_order; i++) {
		cumulative += held[order[i]];
		owner_end[i] = cumulative;
	}
	char line[64];
	for (int row = 0; row * width < cells; row++) {
		int len = MIN(width, cells - row * width);
		for (int c = 0; c < len; c++) {
			int layer = (int)((long)(row * width + c) * n_layers / cells);
			int owner = 0;
			while (owner < n_order && layer >= owner_end[owner])
				owner++;
			line[c] = owner < n_order ? symbols[order[owner] % 4] : '.';
		}
		line[len] = '\0';
		ui_putn(y + row, x, line, width, 0);
	}
}

static void editor_draw_hcs_panel(const editor_state *st) {
	static const char symbols[] = "#=+~";
	const hcs_config *h			= &st->sc.hcs;
	int				  px		= COLS - EDITOR_HCS_PANEL_W;
	int				  pw		= EDITOR_HCS_PANEL_W - 2;
	int				  last		= LINES - 3;

	attron(ui_attr(UI_DIM));
	for (int y = 2; y <= last; y++)
		mvaddch(y, px - 2, ACS_VLINE);
	attroff(ui_attr(UI_DIM));

	ui_putn(2, px, "Layer status", pw, A_BOLD);
	if (!st->probe_ok || st->n_layers <= 0) {
		ui_putn(4, px, "model unreadable", pw, ui_attr(UI_DIM));
		ui_putn(5, px, "set a valid model path", pw, ui_attr(UI_DIM));
		return;
	}

	int	 n = st->n_layers;
	int	 held[HCS_MAX_DEVICES];
	int	 assigned = hcs_resolve_layers(h, n, held);
	char summary[48];
	editor_hcs_summary(st, assigned, summary, sizeof(summary));
	bool ok = assigned == n;

	char value[16];
	snprintf(value, sizeof(value), "%d", n);
	ui_putn(4, px, "Model layers", pw, ui_attr(UI_DIM));
	ui_putn(4, px + pw - (int)strlen(value), value, pw, 0);
	snprintf(value, sizeof(value), "%d", MIN(assigned, n));
	ui_putn(5, px, "Assigned", pw, ui_attr(UI_DIM));
	ui_putn(5, px + pw - (int)strlen(value), value, pw, 0);
	ui_putn(6, px, summary, pw, ok ? ui_attr(UI_OK) : ui_attr(UI_ERR));

	int order[HCS_MAX_DEVICES];
	int n_order	  = hcs_layout_order(h, order);
	int grid_y	  = 8;
	int grid_rows = MAX(MIN((n + pw - 1) / pw, last - grid_y - n_order - 3), 1);
	editor_draw_hcs_layer_grid(held, order, n_order, n, px, grid_y, pw, grid_rows);

	int y = grid_y + (MIN(n, pw * grid_rows) + pw - 1) / pw + 1;
	for (int i = 0; i < n_order && y <= last; i++, y++) {
		int	 d = order[i];
		char name[HCS_NAME_MAX + 4];
		snprintf(name, sizeof(name), "%c %s", symbols[d % 4], h->devices[d].name);
		snprintf(value, sizeof(value), "%d", held[d]);
		ui_putn(y, px, name, pw - 5, 0);
		ui_putn(y, px + pw - (int)strlen(value), value, pw, 0);
	}
	if (assigned < n && y <= last)
		ui_putn(y, px, ". unassigned", pw, ui_attr(UI_DIM));
}

static void editor_draw_hcs_banner(const editor_state *st) {
	static const char warn[] = EDITOR_HCS_TAG " experimental, advanced users only";
	static const char off[]	 = "disabled: the device selected in the model tab runs everything";
	ui_putn(2, 3, warn, COLS - 6, A_BOLD | ui_attr(UI_ERR));
	if (!st->sc.hcs.enabled)
		ui_putn(3, 3, off, COLS - 6, ui_attr(UI_DIM));
	else if (editor_hcs_panel_visible(st))
		editor_draw_hcs_panel(st);
	else
		editor_draw_hcs_compact(st);
}

static void editor_draw(const editor_state *st) {
	erase();
	editor_draw_tabs(st);

	if (editor_tab_is_hcs(st->tab))
		editor_draw_hcs_banner(st);
	editor_draw_rows(st);

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
	static const char hcs_bar[] =
		"Enter: edit   A: add device   X: remove device   Tab: section   S: save   Esc: back";
	bool use_hcs_bar = editor_tab_is_hcs(st->tab) && st->sc.hcs.enabled;
	ui_putn(LINES - 1, 2, use_hcs_bar ? hcs_bar : key_bar, COLS - 4, ui_attr(UI_DIM));
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

static void editor_edit_device(editor_state *st, const char *title, char *cur, size_t cap) {
	char		specs[EDITOR_DEV_MAX][BACKEND_DEVICE_NAME_MAX];
	const char *choices[EDITOR_DEV_MAX + 2];
	int			n	  = backend_enum_devices(specs, EDITOR_DEV_MAX);
	int			extra = 0;

	for (int i = 0; i < n; i++)
		choices[i] = specs[i];
	bool found = false;
	for (int i = 0; i < n; i++)
		if (strcmp(specs[i], cur) == 0)
			found = true;
	if (cur[0] && !found)
		choices[n + extra++] = cur;
	int cur_idx = 0;
	for (int i = 0; i < n + extra; i++)
		if (strcmp(choices[i], cur) == 0)
			cur_idx = i;
	static const char custom[] = "custom...";
	choices[n + extra]		   = custom;
	int total				   = n + extra + 1;

	int idx		  = ui_select(title, choices, total, cur_idx);
	st->status[0] = '\0';
	if (idx < 0)
		return;

	if (idx == total - 1) {
		char buf[BACKEND_DEVICE_NAME_MAX];
		if (ui_input(title, cur, buf, sizeof(buf), NULL, NULL, NULL) <= 0)
			return;
		snprintf(cur, cap, "%s", buf);
		st->dirty = true;
	} else if (strcmp(cur, choices[idx]) != 0) {
		snprintf(cur, cap, "%s", choices[idx]);
		st->dirty = true;
	}
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

typedef struct {
	bool			  allow_unset;
	const hcs_config *h;
	int				  self;
	size_t			  field_offset;
} count_validate_ud;

static int editor_count_of(const hcs_device *d, size_t field_offset) {
	return *(const int *)((const char *)d + field_offset);
}

static int editor_validate_count(const char *value, char *err, size_t errlen, void *ud) {
	const count_validate_ud *u = ud;
	int						 parsed;
	if (!value[0] && u->allow_unset)
		return 0;
	if (hcs_parse_count(value, &parsed) != 0) {
		snprintf(err, errlen, "expected a non-negative integer or '*'%s",
				 u->allow_unset ? ", or empty for the default" : "");
		return -1;
	}
	if (parsed != HCS_COUNT_ALL)
		return 0;
	for (int i = 0; i < u->h->n_devices; i++) {
		if (i != u->self && editor_count_of(&u->h->devices[i], u->field_offset) == HCS_COUNT_ALL) {
			snprintf(err, errlen, "'*' is already used by %s; set that device to a number first",
					 u->h->devices[i].name);
			return -1;
		}
	}
	return 0;
}

static void editor_edit_count(editor_state *st, const char *title, int device, size_t field_offset,
							  bool allow_unset) {
	int *value = (int *)((char *)&st->sc.hcs.devices[device] + field_offset);
	char count[16];
	char initial[16];
	char buf[16];
	snprintf(initial, sizeof(initial), "%s", hcs_format_count(*value, count, sizeof(count)));

	count_validate_ud ud = {allow_unset, &st->sc.hcs, device, field_offset};
	int r		  = ui_input(title, initial, buf, sizeof(buf), editor_validate_count, &ud, NULL);
	st->status[0] = '\0';
	if (r <= 0)
		return;
	int parsed = HCS_COUNT_UNSET;
	if (buf[0])
		hcs_parse_count(buf, &parsed);
	if (parsed == *value)
		return;
	*value	  = parsed;
	st->dirty = true;
}

static void editor_edit_tasks(editor_state *st, hcs_device *d) {
	static const char *const choices[] = {"prefill", "decode"};
	bool sel[]	  = {(d->tasks & HCS_TASK_PREFILL) != 0, (d->tasks & HCS_TASK_DECODE) != 0};
	st->status[0] = '\0';
	if (!ui_toggle("Tasks", choices, 2, sel))
		return;
	d->tasks  = (sel[0] ? HCS_TASK_PREFILL : 0) | (sel[1] ? HCS_TASK_DECODE : 0);
	st->dirty = true;
}

static void editor_toggle_hcs(editor_state *st) {
	hcs_config *h = &st->sc.hcs;
	st->status[0] = '\0';
	if (!h->enabled &&
		!ui_confirm("Enable HCS?", "HCS (Heterogeneous Compute Scheduler) is a work in progress "
								   "and intended for advanced users only.\n\n"
								   "Misconfiguration can prevent the model from loading."))
		return;
	h->enabled = !h->enabled;
	if (h->enabled && h->n_devices == 0)
		hcs_set_simple(h, h->devices[0].name);
	st->dirty = true;
}

static void editor_add_hcs_device(editor_state *st) {
	hcs_config *h = &st->sc.hcs;
	if (!h->enabled || !editor_hcs_can_add(st))
		return;

	char		avail[16][BACKEND_DEVICE_NAME_MAX];
	int			n	  = backend_enum_devices(avail, 16);
	const char *pick  = "cpu_0";
	bool		found = false;
	for (int i = 0; i < n && !found; i++) {
		bool used = false;
		for (int d = 0; d < h->n_devices; d++)
			used |= strcmp(h->devices[d].name, avail[i]) == 0;
		if (!used) {
			pick  = avail[i];
			found = true;
		}
	}
	for (int i = 0; i < h->n_devices; i++)
		if (h->devices[i].layers == HCS_COUNT_ALL)
			h->devices[i].layers = st->probe_ok ? (st->n_layers + 1) / 2 : 0;
	hcs_device *d = &h->devices[h->n_devices++];
	snprintf(d->name, sizeof(d->name), "%s", pick);
	d->tasks	  = HCS_TASKS_ALL;
	d->layers	  = HCS_COUNT_ALL;
	d->experts	  = HCS_COUNT_UNSET;
	st->dirty	  = true;
	st->status[0] = '\0';
}

static void editor_remove_hcs_device(editor_state *st) {
	hcs_config *h	= &st->sc.hcs;
	hcs_ref		ref = editor_hcs_ref(st, st->row);
	if (!h->enabled || ref.device < 0)
		return;
	if (h->n_devices <= 1) {
		editor_set_status(st, UI_ERR, "at least one device is required");
		return;
	}
	int idx = ref.device;
	memmove(&h->devices[idx], &h->devices[idx + 1],
			(size_t)(h->n_devices - idx - 1) * sizeof(h->devices[0]));
	h->n_devices--;
	st->row		  = MIN(st->row, editor_last_row(st, st->tab));
	st->dirty	  = true;
	st->status[0] = '\0';
}

static void editor_edit_hcs_row(editor_state *st, int row) {
	hcs_ref ref = editor_hcs_ref(st, row);
	switch (ref.kind) {
	case HCS_REF_ENABLE:
		editor_toggle_hcs(st);
		return;
	case HCS_REF_ADD:
		editor_add_hcs_device(st);
		return;
	case HCS_REF_DEVICE:
		editor_edit_device(st, "Device", st->sc.hcs.devices[ref.device].name,
						   sizeof(st->sc.hcs.devices[ref.device].name));
		return;
	case HCS_REF_TASKS:
		editor_edit_tasks(st, &st->sc.hcs.devices[ref.device]);
		return;
	case HCS_REF_LAYERS:
		editor_edit_count(st, "Layers", ref.device, offsetof(hcs_device, layers), false);
		return;
	case HCS_REF_EXPERTS:
		editor_edit_count(st, "Experts", ref.device, offsetof(hcs_device, experts), true);
		return;
	default:
		return;
	}
}

static void editor_save(editor_state *st) {
	char hcs_err[HCS_ERR_MAX];
	if (hcs_validate(&st->sc.hcs, st->sc.model, hcs_err, sizeof(hcs_err)) < 0) {
		ui_message("Cannot save: invalid HCS configuration", hcs_err, true);
		editor_set_status(st, UI_ERR, "not saved: HCS validation failed");
		beep();
		return;
	}
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
	const char *label;
	const char *hint;
	char		label_buf[EDITOR_LABEL_W + 8];
	char		value_buf[SC_SYSTEM_MAX + 16];
	if (editor_tab_is_hcs(st->tab)) {
		hcs_ref ref = editor_hcs_ref(st, st->row);
		editor_hcs_row_text(st, st->row, label_buf, sizeof(label_buf), value_buf,
							sizeof(value_buf));
		label = label_buf;
		hint  = ref.kind == HCS_REF_ENABLE ? EDITOR_HCS_ENABLE_HINT
				: ref.kind == HCS_REF_ADD  ? "add another device; it takes '*' (the remainder)"
										   : HCS_FIELD_HINTS[ref.kind - HCS_REF_DEVICE];
	} else if (editor_is_model_device_row(st, st->tab, st->row)) {
		label = "Device";
		hint  = EDITOR_MODEL_DEVICE_HINT;
	} else if (editor_is_name_row(st->tab, st->row)) {
		label = EDITOR_NAME_LABEL;
		hint  = EDITOR_NAME_HINT;
	} else {
		const field_desc *f = editor_field_at(st->tab, st->row);
		label				= f->label;
		hint				= sc_field_hint(f);
	}

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
		if (editor_tab_is_hcs(st.tab) && st.sc.hcs.enabled)
			editor_refresh_probe(&st);
		editor_snap_row(&st);
		editor_scroll(&st);
		editor_draw(&st);

		int ch = ui_getch();
		if (ch == KBD_INTERRUPT)
			return 0;
		if (ch == KEY_RESIZE)
			continue;

		switch (ch) {
		case KEY_UP:
			editor_move_row(&st, st.row - 1, -1);
			st.status[0] = '\0';
			break;
		case KEY_DOWN:
			editor_move_row(&st, st.row + 1, 1);
			st.status[0] = '\0';
			break;
		case KEY_PPAGE:
			editor_move_row(&st, st.row - editor_rows(&st, st.tab), 1);
			st.status[0] = '\0';
			break;
		case KEY_NPAGE:
			editor_move_row(&st, st.row + editor_rows(&st, st.tab), -1);
			st.status[0] = '\0';
			break;
		case KEY_HOME:
			editor_move_row(&st, 0, 1);
			st.status[0] = '\0';
			break;
		case KEY_END:
			editor_move_row(&st, editor_last_row(&st, st.tab), -1);
			st.status[0] = '\0';
			break;
		case '\t':
			editor_switch_tab(&st, (st.tab + 1) % editor_tab_count());
			break;
		case KEY_BTAB:
			editor_switch_tab(&st, (st.tab + editor_tab_count() - 1) % editor_tab_count());
			break;
		case 'a':
		case 'A':
			if (editor_tab_is_hcs(st.tab))
				editor_add_hcs_device(&st);
			break;
		case 'x':
		case 'X':
			if (editor_tab_is_hcs(st.tab))
				editor_remove_hcs_device(&st);
			break;
		case '\n':
		case KEY_ENTER:
			if (editor_tab_is_hcs(st.tab))
				editor_edit_hcs_row(&st, st.row);
			else if (editor_is_name_row(st.tab, st.row))
				editor_edit_name(&st);
			else if (editor_is_model_device_row(&st, st.tab, st.row))
				editor_edit_device(&st, "Device", st.sc.hcs.devices[0].name,
								   sizeof(st.sc.hcs.devices[0].name));
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
