#include "dialog.h"

#include "settings.h"

#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static size_t cp_back(const char *s, size_t pos) {
	if (pos == 0)
		return 0;
	size_t p = pos - 1;
	while (p > 0 && ((unsigned char)s[p] & 0xC0) == 0x80)
		p--;
	return p;
}

static size_t cp_fwd(const char *s, size_t pos) {
	if (!s[pos])
		return pos;
	size_t p = pos + 1;
	while (s[p] && ((unsigned char)s[p] & 0xC0) == 0x80)
		p++;
	return p;
}

static int width_between(const char *s, size_t from, size_t to) {
	int w = 0;
	for (size_t i = from; i < to; i++)
		if (((unsigned char)s[i] & 0xC0) != 0x80)
			w++;
	return w;
}

static void clamp_view(const char *buf, size_t len, size_t pos, size_t *win, int vw) {
	while (pos < *win)
		*win = cp_back(buf, *win);
	while (width_between(buf, *win, pos) > vw) {
		size_t next = cp_fwd(buf, *win);
		if (next <= *win)
			break;
		*win = next;
	}
	if (*win > len)
		*win = len;
}

static bool ghost_accept(char *buf, size_t buflen, size_t *len, size_t *pos, char *err,
						 size_t errlen, const char *ghost) {
	size_t glen = strlen(ghost);
	if (*len + glen + 1 > buflen) {
		snprintf(err, errlen, "completion does not fit");
		beep();
		return false;
	}
	memcpy(buf + *pos, ghost, glen);
	*len += glen;
	*pos	  = *len;
	buf[*len] = '\0';
	return true;
}

static void list_place(int w, int h, int *bx, int *by) {
	*bx = (COLS - w) / 2;
	*by = (LINES - h) / 2;
	if (*bx < 0)
		*bx = 0;
	if (*by < 0)
		*by = 0;
}

static void input_geometry(int *w, int *h, int *bx, int *by) {
	int iw = COLS - 6;
	if (iw > 64)
		iw = 64;
	if (iw < 24)
		iw = 24;
	*w = iw;
	*h = 4;
	list_place(*w, *h, bx, by);
}

static void input_draw(const char *title, const char *buf, size_t len, size_t pos, size_t *win,
					   const char *err, const char *ghost) {
	int w, h, bx, by;
	input_geometry(&w, &h, &bx, &by);

	ui_box(by, bx, h, w, title, ui_attr(UI_TITLE));
	int vw = w - 8;
	clamp_view(buf, len, pos, win, vw - 1);
	ui_putn_bytes(by + 1, bx + 2, buf + *win, len - *win, vw - 1, 0);

	if (ghost[0] && pos == len) {
		int gx	 = bx + 2 + width_between(buf, *win, pos);
		int gmax = bx + w - 3 - gx;
		if (gmax > 0)
			ui_putn(by + 1, gx, ghost, gmax, ui_attr(UI_DIM));
	}

	static const char plain_hint[] = "Enter: accept   Esc: cancel";
	static const char compl_hint[] = "Enter: accept   Esc: cancel   Tab: complete";
	ui_putn(by + 2, bx + 2, err[0] ? err : (ghost[0] ? compl_hint : plain_hint), w - 6,
			err[0] ? ui_attr(UI_ERR) : ui_attr(UI_DIM));
}

int ui_input(const char *title, const char *initial, char *buf, size_t buflen, ui_validate_fn v,
			 void *ud, ui_complete_fn compl) {
	if (buflen < 2)
		return 0;

	size_t len = initial ? strlen(initial) : 0;
	if (len >= buflen)
		len = buflen - 1;
	if (initial && initial != buf)
		memmove(buf, initial, len);
	buf[len] = '\0';

	size_t pos			   = len;
	size_t win			   = 0;
	char   err[UI_ERR_MAX] = {0};
	char   ghost[UI_MATCH_MAX];
	int	   rc = 0;

	curs_set(1);
	for (;;) {
		ghost[0] = '\0';
		if (compl && pos == len) {
			size_t wstart = pos;
			while (wstart > 0 && buf[wstart - 1] != '/')
				wstart--;
			if (pos > wstart) {
				ui_completion c;
				if (compl(buf, pos, &c) > 0 && c.insert[0])
					snprintf(ghost, sizeof(ghost), "%s", c.insert);
			}
		}

		input_draw(title, buf, len, pos, &win, err, ghost);

		int w, h, bx, by;
		input_geometry(&w, &h, &bx, &by);
		int cx = bx + 2 + width_between(buf, win, pos);
		if (cx >= bx + 2 + w - 8)
			cx = bx + 2 + w - 9;
		move(by + 1, cx);
		refresh();

		int ch = ui_getch();
		if (ch == KBD_INTERRUPT) {
			rc = -1;
			break;
		}
		if (ch == KEY_RESIZE)
			continue;
		err[0] = '\0';

		if (ch == '\t' && compl) {
			if (ghost[0])
				ghost_accept(buf, buflen, &len, &pos, err, sizeof(err), ghost);
			else
				beep();
			continue;
		}

		if (ch == '\n' || ch == KEY_ENTER) {
			if (v) {
				char verr[UI_ERR_MAX];
				verr[0] = '\0';
				if (v(buf, verr, sizeof(verr), ud) != 0) {
					snprintf(err, sizeof(err), "%s", verr);
					beep();
					continue;
				}
			}
			rc = 1;
			break;
		}
		if (ch == 27) {
			rc = 0;
			break;
		}
		if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
			if (pos == 0) {
				beep();
				continue;
			}
			size_t ns = cp_back(buf, pos);
			memmove(buf + ns, buf + pos, len - pos);
			len -= pos - ns;
			pos		 = ns;
			buf[len] = '\0';
			continue;
		}
		if (ch == KEY_DC) {
			if (pos < len) {
				size_t ne = cp_fwd(buf, pos);
				memmove(buf + pos, buf + ne, len - ne);
				len -= ne - pos;
				buf[len] = '\0';
			}
			continue;
		}
		if (ch == KEY_LEFT) {
			pos = cp_back(buf, pos);
			continue;
		}
		if (ch == KEY_RIGHT) {
			if (pos == len && ghost[0]) {
				ghost_accept(buf, buflen, &len, &pos, err, sizeof(err), ghost);
				continue;
			}
			pos = cp_fwd(buf, pos);
			continue;
		}
		if (ch == KEY_HOME) {
			pos = 0;
			continue;
		}
		if (ch == KEY_END) {
			pos = len;
			continue;
		}
		if (ch >= 32 && ch < 256 && ch != 127) {
			if (len + 1 < buflen) {
				memmove(buf + pos + 1, buf + pos, len - pos);
				buf[pos] = (char)ch;
				pos++;
				len++;
				buf[len] = '\0';
			} else {
				beep();
			}
			continue;
		}
	}
	curs_set(0);
	return rc;
}

static int path_complete(const char *base, const char *buf, size_t pos, ui_completion *out) {
	memset(out, 0, sizeof(*out));

	size_t from = pos;
	while (from > 0 && buf[from - 1] != '/')
		from--;

	char dir[PATH_MAX];
	if (from >= sizeof(dir))
		return 0;
	memcpy(dir, buf, from);
	dir[from] = '\0';

	const char *word = buf + from;
	size_t		wlen = pos - from;

	char lookup[PATH_MAX];
	bool relative = base && base[0] && buf[0] != '/';
	if (snprintf(lookup, sizeof(lookup), "%s%s%s", relative ? base : "", relative ? "/" : "",
				 dir) >= (int)sizeof(lookup))
		return 0;

	DIR *d = opendir(lookup[0] ? lookup : ".");
	if (!d)
		return 0;

	char   ref[256];
	size_t common  = 0;
	int	   n_total = 0;

	struct dirent *de;
	while ((de = readdir(d)) != NULL) {
		const char *nm = de->d_name;
		size_t		nl = strlen(nm);
		if (strcmp(nm, ".") == 0 || strcmp(nm, "..") == 0)
			continue;
		if (nm[0] == '.' && !(wlen > 0 && word[0] == '.'))
			continue;
		if (nl < wlen || strncmp(nm, word, wlen) != 0)
			continue;
		if (n_total == 0) {
			snprintf(ref, sizeof(ref), "%s", nm);
			common = nl;
		} else {
			size_t k = wlen;
			while (k < common && k < nl && ref[k] == nm[k])
				k++;
			common = k;
		}
		n_total++;
	}
	closedir(d);

	if (n_total == 0)
		return 0;

	while (common > wlen && ((unsigned char)ref[common] & 0xC0) == 0x80)
		common--;

	if (n_total == 1) {
		char		full[PATH_MAX];
		struct stat st;
		bool is_dir = snprintf(full, sizeof(full), "%s%s", lookup, ref) < (int)sizeof(full) &&
					  stat(full, &st) == 0 && S_ISDIR(st.st_mode);
		snprintf(out->insert, sizeof(out->insert), "%s%s", ref + wlen, is_dir ? "/" : "");
	} else {
		size_t n = common - wlen;
		if (n >= sizeof(out->insert))
			n = sizeof(out->insert) - 1;
		memcpy(out->insert, ref + wlen, n);
		out->insert[n] = '\0';
	}
	return n_total;
}

int ui_path_complete(const char *buf, size_t pos, ui_completion *out) {
	return path_complete(NULL, buf, pos, out);
}

int ui_model_path_complete(const char *buf, size_t pos, ui_completion *out) {
	return path_complete(tui_models_dir(), buf, pos, out);
}

static void list_geometry(const char *title, const char *const choices[], int n, int prefix, int *w,
						  int *h, int *max_vis) {
	int item_w = 0;
	for (int i = 0; i < n; i++) {
		int cw = ui_disp_width(choices[i]) + prefix;
		if (cw > item_w)
			item_w = cw;
	}
	*w		 = MAX(item_w + 6, (int)strlen(title) + 6);
	*w		 = MAX(*w, 38);
	*w		 = MIN(*w, COLS - 4);
	*max_vis = MIN(n, LINES - 5);
	*max_vis = MIN(*max_vis, 12);
	if (*max_vis < 1)
		*max_vis = 1;
	*h = *max_vis + 3;
}

static void list_scroll(int *top, int sel, int n, int max_vis) {
	if (sel < *top)
		*top = sel;
	if (sel >= *top + max_vis)
		*top = sel - max_vis + 1;
	if (*top > n - max_vis)
		*top = MAX(0, n - max_vis);
	if (*top < 0)
		*top = 0;
}

static bool list_nav(int ch, int *sel, int n) {
	if (ch == KEY_UP)
		*sel = (*sel + n - 1) % n;
	else if (ch == KEY_DOWN)
		*sel = (*sel + 1) % n;
	else if (ch == KEY_HOME)
		*sel = 0;
	else if (ch == KEY_END)
		*sel = n - 1;
	else
		return false;
	return true;
}

int ui_select(const char *title, const char *const choices[], int n, int cur) {
	if (n <= 0)
		return -1;
	if (cur < 0 || cur >= n)
		cur = 0;

	int sel = cur;
	int top = 0;

	for (;;) {
		int w, h, max_vis;
		list_geometry(title, choices, n, 0, &w, &h, &max_vis);
		int bx, by;
		list_place(w, h, &bx, &by);
		list_scroll(&top, sel, n, max_vis);

		ui_box(by, bx, h, w, title, ui_attr(UI_TITLE));
		for (int i = 0; i < max_vis && top + i < n; i++) {
			int	   idx	= top + i;
			chtype attr = (idx == sel) ? (A_REVERSE | A_BOLD) : 0;
			ui_cell(by + 1 + i, bx + 2, w - 4, choices[idx], attr);
		}
		ui_putn(by + h - 2, bx + 3, "Up/Down: select   Enter: accept   Esc: cancel", w - 6,
				ui_attr(UI_DIM));
		refresh();

		int ch = ui_getch();
		if (ch == KBD_INTERRUPT || ch == 27)
			return -1;
		if (ch == KEY_RESIZE)
			continue;
		if (list_nav(ch, &sel, n))
			continue;
		if (ch == '\n' || ch == KEY_ENTER)
			return sel;
		if (ch >= 32 && ch < 256 && ch != 127) {
			for (int i = 0; i < n; i++) {
				if (tolower((unsigned char)choices[i][0]) == tolower((unsigned char)ch)) {
					sel = i;
					break;
				}
			}
		}
	}
}

bool ui_toggle(const char *title, const char *const choices[], int n, bool *sel) {
	if (n <= 0)
		return false;

	bool *work = xmalloc((size_t)n * sizeof(bool));
	memcpy(work, sel, (size_t)n * sizeof(bool));
	int	 cur	= 0;
	int	 top	= 0;
	bool result = false;

	for (;;) {
		int w, h, max_vis;
		list_geometry(title, choices, n, 4, &w, &h, &max_vis);
		int bx, by;
		list_place(w, h, &bx, &by);
		list_scroll(&top, cur, n, max_vis);

		ui_box(by, bx, h, w, title, ui_attr(UI_TITLE));
		for (int i = 0; i < max_vis && top + i < n; i++) {
			int	 idx = top + i;
			char row[300];
			snprintf(row, sizeof(row), "[%c] %s", work[idx] ? 'x' : ' ', choices[idx]);
			chtype attr = (idx == cur) ? (A_REVERSE | A_BOLD) : 0;
			ui_cell(by + 1 + i, bx + 2, w - 4, row, attr);
		}
		ui_putn(by + h - 2, bx + 3,
				"Space: toggle   A: all   N: none   Enter: accept   Esc: cancel", w - 6,
				ui_attr(UI_DIM));
		refresh();

		int ch = ui_getch();
		if (ch == KBD_INTERRUPT || ch == 27)
			break;
		if (ch == KEY_RESIZE)
			continue;
		if (list_nav(ch, &cur, n))
			continue;
		if (ch == ' ')
			work[cur] = !work[cur];
		else if (ch == 'a' || ch == 'A') {
			for (int i = 0; i < n; i++)
				work[i] = true;
		} else if (ch == 'n' || ch == 'N') {
			for (int i = 0; i < n; i++)
				work[i] = false;
		} else if (ch == '\n' || ch == KEY_ENTER) {
			memcpy(sel, work, (size_t)n * sizeof(bool));
			result = true;
			break;
		}
	}
	free(work);
	return result;
}

static void msg_geometry(const char *msg, int cap, int *w, int *text_w, int *h, int *bx, int *by) {
	*w = COLS - 4;
	if (*w > cap)
		*w = cap;
	if (*w < 30)
		*w = 30;
	*text_w	  = *w - 6;
	int lines = ui_wrap_count(msg, *text_w);
	*h		  = lines + 4;
	*bx		  = (COLS - *w) / 2;
	*by		  = (LINES - *h) / 2;
	if (*by < 0)
		*by = 0;
}

bool ui_confirm(const char *title, const char *msg) {
	int w, text_w, h, bx, by;
	msg_geometry(msg, 56, &w, &text_w, &h, &bx, &by);

	static const char yes_label[] = "[ Yes ]";
	static const char no_label[]  = "[ No ]";
	int				  gap		  = 4;
	int				  buttons_w	  = (int)strlen(yes_label) + gap + (int)strlen(no_label);
	int				  b0		  = bx + (w - buttons_w) / 2;
	int				  by_row	  = by + h - 2;

	int	 sel	= 1;
	bool result = false;

	for (;;) {
		ui_box(by, bx, h, w, title, ui_attr(UI_ERR));
		ui_draw_wrapped(by + 1, bx + 3, text_w, msg, 0);
		ui_putn(by_row, b0, yes_label, (int)strlen(yes_label),
				sel == 0 ? (A_REVERSE | A_BOLD) : ui_attr(UI_DIM));
		ui_putn(by_row, b0 + (int)strlen(yes_label) + gap, no_label, (int)strlen(no_label),
				sel == 1 ? (A_REVERSE | A_BOLD) : ui_attr(UI_DIM));
		refresh();

		int ch = ui_getch();
		if (ch == KBD_INTERRUPT || ch == 27) {
			result = false;
			break;
		}
		if (ch == KEY_RESIZE)
			continue;
		if (ch == '\n' || ch == KEY_ENTER) {
			result = (sel == 0);
			break;
		}
		if (ch == KEY_UP || ch == KEY_DOWN || ch == KEY_LEFT || ch == KEY_RIGHT || ch == '\t' ||
			ch == KEY_BTAB)
			sel = 1 - sel;
		else if (ch == 'y' || ch == 'Y') {
			result = true;
			break;
		} else if (ch == 'n' || ch == 'N') {
			result = false;
			break;
		}
	}
	return result;
}

void ui_message(const char *title, const char *msg, bool error) {
	int w, text_w, h, bx, by;
	msg_geometry(msg, 60, &w, &text_w, &h, &bx, &by);

	for (;;) {
		ui_box(by, bx, h, w, title, error ? ui_attr(UI_ERR) : ui_attr(UI_TITLE));
		ui_draw_wrapped(by + 1, bx + 3, text_w, msg, error ? ui_attr(UI_ERR) : 0);
		ui_putn(by + h - 2, bx + 3, "press any key to continue", text_w, ui_attr(UI_DIM));
		refresh();

		int ch = ui_getch();
		if (ch == KBD_INTERRUPT)
			break;
		if (ch == KEY_RESIZE || ch == ERR)
			continue;
		break;
	}
}
