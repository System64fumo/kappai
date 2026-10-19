#define _GNU_SOURCE
#include "ui.h"

#include "log.h"

#include <locale.h>
#include <stdio.h>
#include <string.h>

volatile sig_atomic_t ui_interrupt = 0;

static int g_colors = 0;

enum {
	UI_PAIR_TITLE = 1,
	UI_PAIR_ERR,
	UI_PAIR_OK,
	UI_PAIR_WARN,
};

static void on_sigint(int sig) {
	(void)sig;
	ui_interrupt = 1;
}

void ui_install_interrupt(void) {
	ui_interrupt = 0;
	signal(SIGINT, on_sigint);
}

chtype ui_attr(int kind) {
	switch (kind) {
	case UI_TITLE:
		return g_colors ? (COLOR_PAIR(UI_PAIR_TITLE) | A_BOLD) : A_BOLD;
	case UI_ERR:
		return g_colors ? COLOR_PAIR(UI_PAIR_ERR) : A_REVERSE;
	case UI_OK:
		return g_colors ? COLOR_PAIR(UI_PAIR_OK) : A_BOLD;
	case UI_WARN:
		return g_colors ? COLOR_PAIR(UI_PAIR_WARN) : A_BOLD;
	default:
		return A_DIM;
	}
}

int ui_init(void) {
	if (initscr() == NULL)
		return -1;
	if (!getenv("ESCDELAY"))
		set_escdelay(30);
	cbreak();
	noecho();
	keypad(stdscr, TRUE);
	meta(stdscr, TRUE);
	curs_set(0);
	timeout(500);

	if (has_colors()) {
		if (start_color() == OK) {
			g_colors = 1;
			use_default_colors();
			init_pair(UI_PAIR_TITLE, COLOR_CYAN, -1);
			init_pair(UI_PAIR_ERR, COLOR_RED, -1);
			init_pair(UI_PAIR_OK, COLOR_GREEN, -1);
			init_pair(UI_PAIR_WARN, COLOR_YELLOW, -1);
		}
	}
	ui_install_interrupt();
	return 0;
}

void ui_shutdown(void) {
	signal(SIGINT, SIG_DFL);
	endwin();
}

void ui_after_resume(void) {
	ui_install_interrupt();
	touchwin(stdscr);
	refresh();
	curs_set(0);
}

int ui_getch(void) {
	for (;;) {
		if (ui_interrupt)
			return KBD_INTERRUPT;
		int ch = getch();
		if (ch == ERR)
			continue;
		return ch;
	}
}

int ui_disp_width(const char *s) {
	int w = 0;
	for (const unsigned char *p = (const unsigned char *)s; *p; p++)
		if ((*p & 0xC0) != 0x80)
			w++;
	return w;
}

void ui_putn_bytes(int y, int x, const char *s, size_t n, int max_cols, chtype attr) {
	if (y < 0 || y >= LINES || x < 0 || x >= COLS || max_cols <= 0)
		return;
	move(y, x);
	attron(attr);
	int cols = 0;
	for (const unsigned char *p = (const unsigned char *)s;
		 (size_t)(p - (const unsigned char *)s) < n && cols < max_cols; p++) {
		if ((*p & 0xC0) != 0x80)
			cols++;
		addch(*p);
	}
	attroff(attr);
}

void ui_putn(int y, int x, const char *s, int max_cols, chtype attr) {
	ui_putn_bytes(y, x, s, strlen(s), max_cols, attr);
}

void ui_center(int y, const char *s, chtype attr) {
	int w = ui_disp_width(s);
	int x = (COLS - w) / 2;
	if (x < 0)
		x = 0;
	ui_putn(y, x, s, COLS - x, attr);
}

void ui_fill(int y, int x, int rows, int cols, chtype attr) {
	static const char spaces[] = "                                                            "
								 "                                        ";
	if (rows <= 0)
		return;
	for (int r = 0; r < rows; r++) {
		int rem = cols;
		int cx	= x;
		while (rem > 0) {
			int c = MIN(rem, (int)sizeof(spaces) - 1);
			ui_putn_bytes(y + r, cx, spaces, c, c, attr);
			cx += c;
			rem -= c;
		}
	}
}

void ui_cell(int y, int x, int w, const char *text, chtype attr) {
	if (w < 2)
		return;
	ui_putn(y, x, " ", 1, attr);
	ui_putn(y, x + 1, text, w - 2, attr);
	int used = MIN(ui_disp_width(text), w - 2);
	ui_fill(y, x + 1 + used, 1, w - 2 - used, attr);
	ui_putn(y, x + w - 1, " ", 1, attr);
}

void ui_box(int y, int x, int h, int w, const char *title, chtype title_attr) {
	if (h < 2 || w < 4)
		return;
	ui_fill(y + 1, x + 1, h - 2, w - 2, 0);
	for (int i = 1; i < w - 1; i++) {
		mvaddch(y, x + i, ACS_HLINE);
		mvaddch(y + h - 1, x + i, ACS_HLINE);
	}
	for (int j = 1; j < h - 1; j++) {
		mvaddch(y + j, x, ACS_VLINE);
		mvaddch(y + j, x + w - 1, ACS_VLINE);
	}
	mvaddch(y, x, ACS_ULCORNER);
	mvaddch(y, x + w - 1, ACS_URCORNER);
	mvaddch(y + h - 1, x, ACS_LLCORNER);
	mvaddch(y + h - 1, x + w - 1, ACS_LRCORNER);
	if (title && *title) {
		int tw = ui_disp_width(title) + 2;
		int tx = x + (w - tw) / 2;
		if (tx < x + 1)
			tx = x + 1;
		ui_putn(y, tx, title, w - 2, title_attr);
	}
}

typedef struct {
	const char *p;
	size_t		bytes;
	int			cols;
} wrap_span;

static int wrap_step(const char **cursor, int width, wrap_span *out) {
	if (width < 1)
		width = 1;
	const char *p = *cursor;
	while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
		p++;
	if (!*p)
		return 0;

	const char *line_start = p;
	const char *line_end   = p;
	int			cols	   = 0;

	for (;;) {
		const char *wp	  = p;
		int			wcols = 0;
		while (*wp && *wp != ' ' && *wp != '\t' && *wp != '\n' && *wp != '\r') {
			if ((*wp & 0xC0) != 0x80)
				wcols++;
			wp++;
		}

		if (wcols > width && cols == 0) {
			int			used = 0;
			const char *hp	 = p;
			while (used < width) {
				if ((*hp & 0xC0) != 0x80)
					used++;
				hp++;
			}
			out->p	   = line_start;
			out->bytes = (size_t)(hp - line_start);
			out->cols  = used;
			*cursor	   = hp;
			return 1;
		}

		if (cols > 0 && cols + 1 + wcols > width) {
			break;
		}
		if (cols > 0)
			cols++;
		cols += wcols;
		line_end = wp;
		p		 = wp;

		while (*p == ' ' || *p == '\t')
			p++;
		if (*p == '\n') {
			p++;
			break;
		}
		if (!*p)
			break;
	}

	out->p	   = line_start;
	out->bytes = (size_t)(line_end - line_start);
	out->cols  = cols;
	*cursor	   = p;
	return 1;
}

int ui_wrap_count(const char *text, int width) {
	const char *cur = text;
	wrap_span	span;
	int			n = 0;
	while (wrap_step(&cur, width, &span))
		n++;
	return n;
}

void ui_draw_wrapped(int y, int x, int width, const char *text, chtype attr) {
	const char *cur = text;
	wrap_span	span;
	while (wrap_step(&cur, width, &span)) {
		ui_putn_bytes(y, x, span.p, span.bytes, width, attr);
		y++;
	}
}
