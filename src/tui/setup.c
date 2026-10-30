#include "setup.h"

#include "common.h"
#include "dialog.h"
#include "model_picker.h"
#include "schema.h"
#include "settings.h"
#include "ui.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define SETUP_TEXT_MAX_W 72
#define SETUP_MAX_PARAGRAPHS 5
#define SETUP_DEFAULT_MODELS_DIR "~/models"

typedef enum {
	STEP_INFO,
	STEP_MODELS_DIR,
	STEP_MODE,
	STEP_FINISH,
} step_kind;

typedef struct {
	step_kind	kind;
	const char *title;
	const char *paragraphs[SETUP_MAX_PARAGRAPHS];
} setup_step;

typedef struct {
	tui_settings cfg;
	int			 step;
} setup_state;

static const char *const MODE_CHOICES[] = {"chat", "server"};

static const setup_step STEPS[] = {
	{STEP_INFO,
	 "Welcome",
	 {"kappai runs local language and image models on your own hardware.",
	  "This short setup picks where your models live and chooses a default run mode. "
	  "Everything can be changed later in Settings (S).",
	  "Keys that work everywhere: Up/Down move, Enter chooses or edits, Tab switches tabs, Esc "
	  "goes back (or quits from the main menu), H or F1 shows help."}},
	{STEP_MODELS_DIR,
	 "Models directory",
	 {"Pick one folder for all your models. Configs can then use short relative paths such as "
	  "language/llama/Model-Q4_K_M.gguf, and downloads are organised there as "
	  "type/architecture/file.",
	  "Enter: choose a folder (a missing folder can be created).   Right: skip."}},
	{STEP_MODE,
	 "Default run mode",
	 {"Chat opens an interactive terminal chat with the model. Server exposes an "
	  "OpenAI-compatible API.",
	  "You can still switch per run with Tab in the main menu.", "Enter: choose.   Right: skip."}},
	{STEP_FINISH,
	 "You're all set",
	 {"From here: press N in the main menu to create a config, pick its model with the model "
	  "picker, then run it with Enter.",
	  "Settings (S) holds the model manager (M) for downloading, deleting and adding models, and "
	  "can run this guide again.",
	  "Enter: finish.   M: finish and open the model manager to download your first model."}},
};
#define STEP_COUNT ((int)(sizeof(STEPS) / sizeof(STEPS[0])))

static bool step_is_input(step_kind kind) {
	return kind == STEP_MODELS_DIR || kind == STEP_MODE;
}

static void format_current(const setup_state *st, step_kind kind, char *out, size_t cap) {
	const tui_settings *c = &st->cfg;
	if (kind == STEP_MODELS_DIR)
		snprintf(out, cap, "Current: %s", c->models_dir[0] ? c->models_dir : "(not set)");
	else
		snprintf(out, cap, "Current: %s", MODE_CHOICES[c->default_server ? 1 : 0]);
}

static void draw(const setup_state *st) {
	const setup_step *step = &STEPS[st->step];
	erase();

	char progress[32];
	snprintf(progress, sizeof(progress), "step %d/%d", st->step + 1, STEP_COUNT);
	ui_putn(0, 2, " setup ", 7, A_REVERSE | A_BOLD);
	ui_putn(0, COLS - (int)strlen(progress) - 2, progress, (int)strlen(progress), ui_attr(UI_DIM));
	move(1, 1);
	attron(ui_attr(UI_DIM));
	for (int i = 1; i < COLS - 1; i++)
		addch(ACS_HLINE);
	attroff(ui_attr(UI_DIM));

	int w = MIN(COLS - 8, SETUP_TEXT_MAX_W);
	int x = (COLS - w) / 2;
	int y = 4;
	ui_putn(y++, x, step->title, w, A_BOLD | ui_attr(UI_TITLE));
	y++;
	for (int i = 0; i < SETUP_MAX_PARAGRAPHS && step->paragraphs[i]; i++) {
		ui_draw_wrapped(y, x, w, step->paragraphs[i], 0);
		y += ui_wrap_count(step->paragraphs[i], w) + 1;
	}
	if (step_is_input(step->kind)) {
		char current[SC_PATH_MAX + 16];
		format_current(st, step->kind, current, sizeof(current));
		ui_putn(y, x, current, w, A_BOLD);
	}

	const char *bar =
		step->kind == STEP_FINISH
			? "Enter: finish   M: finish and open models   Left: back   Esc: skip setup"
			: "Enter: continue/edit   Right: skip   Left: back   Esc: skip setup";
	ui_putn(LINES - 1, 2, bar, COLS - 4, ui_attr(UI_DIM));
	refresh();
}

static void commit(setup_state *st) {
	char err[SC_ERR_MAX];
	if (tui_settings_save(&st->cfg, err, sizeof(err)) < 0) {
		ui_message("Could not save settings", err, true);
		st->cfg = *tui_settings_get();
	}
}

static int validate_absolute(const char *value, char *err, size_t errlen, void *ud) {
	(void)ud;
	char expanded[TUI_PATH_MAX];
	return tui_expand_path(value, expanded, sizeof(expanded), err, errlen);
}

static bool ensure_directory(const char *dir) {
	struct stat sb;
	char		err[SC_ERR_MAX];
	if (stat(dir, &sb) == 0)
		return S_ISDIR(sb.st_mode);
	char msg[TUI_PATH_MAX + 48];
	snprintf(msg, sizeof(msg), "'%s' does not exist. Create it?", dir);
	if (!ui_confirm("Create directory", msg))
		return false;
	if (mkdir_p(dir, err, sizeof(err)) < 0) {
		ui_message("Could not create directory", err, true);
		return false;
	}
	return true;
}

static bool edit_models_dir(setup_state *st) {
	char		buf[TUI_PATH_MAX], dir[TUI_PATH_MAX], err[SC_ERR_MAX];
	const char *initial = st->cfg.models_dir[0] ? st->cfg.models_dir : SETUP_DEFAULT_MODELS_DIR;
	if (ui_input("Models directory", initial, buf, sizeof(buf), validate_absolute, NULL,
				 ui_path_complete) <= 0)
		return false;
	if (tui_expand_path(buf, dir, sizeof(dir), err, sizeof(err)) < 0)
		return false;
	if (dir[0] && !ensure_directory(dir))
		return false;
	snprintf(st->cfg.models_dir, sizeof(st->cfg.models_dir), "%s", dir);
	commit(st);
	return true;
}

static bool edit_mode(setup_state *st) {
	int idx = ui_select("Default run mode", MODE_CHOICES, 2, st->cfg.default_server ? 1 : 0);
	if (idx < 0)
		return false;
	st->cfg.default_server = idx == 1;
	commit(st);
	return true;
}

static bool edit_step(setup_state *st) {
	switch (STEPS[st->step].kind) {
	case STEP_MODELS_DIR:
		return edit_models_dir(st);
	case STEP_MODE:
		return edit_mode(st);
	default:
		return true;
	}
}

static bool confirm_skip(void) {
	return ui_confirm("Skip setup", "skip the rest of setup? you can run it again from Settings.");
}

static void finish(setup_state *st, bool open_models) {
	st->cfg.setup_done = true;
	commit(st);
	if (open_models && tui_models_dir()[0])
		model_manager_run();
}

void setup_run(void) {
	setup_state st;
	memset(&st, 0, sizeof(st));
	st.cfg = *tui_settings_get();

	for (;;) {
		draw(&st);
		int	 ch	  = ui_getch();
		bool last = st.step == STEP_COUNT - 1;
		if (ch == KBD_INTERRUPT || ch == 27) {
			if (!confirm_skip())
				continue;
			finish(&st, false);
			return;
		}
		if (ch == KEY_LEFT || ch == KEY_BACKSPACE) {
			st.step = MAX(st.step - 1, 0);
		} else if (ch == KEY_RIGHT && !last) {
			st.step++;
		} else if ((ch == 'm' || ch == 'M') && last) {
			finish(&st, true);
			return;
		} else if (ch == '\n' || ch == KEY_ENTER) {
			if (last) {
				finish(&st, false);
				return;
			}
			if (edit_step(&st))
				st.step++;
		}
	}
}
