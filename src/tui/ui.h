#ifndef TUI_UI_H
#define TUI_UI_H

#include "common.h"

#include <curses.h>
#include <signal.h>

#define KBD_INTERRUPT (KEY_MAX + 8)

enum {
	UI_TITLE = 0,
	UI_ERR,
	UI_OK,
	UI_DIM,
	UI_WARN,
};

chtype ui_attr(int kind);

extern volatile sig_atomic_t ui_interrupt;

int	 ui_init(void);
void ui_shutdown(void);
void ui_after_resume(void);
void ui_install_interrupt(void);
int	 ui_getch(void);

int	 ui_disp_width(const char *s);
void ui_center(int y, const char *s, chtype attr);
void ui_putn(int y, int x, const char *s, int max_cols, chtype attr);
void ui_putn_bytes(int y, int x, const char *s, size_t n, int max_cols, chtype attr);
void ui_box(int y, int x, int h, int w, const char *title, chtype title_attr);
void ui_fill(int y, int x, int rows, int cols, chtype attr);
void ui_cell(int y, int x, int w, const char *text, chtype attr);
int	 ui_wrap_count(const char *text, int width);
void ui_draw_wrapped(int y, int x, int width, const char *text, chtype attr);

#endif
