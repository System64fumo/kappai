#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <locale.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <curses.h>
#include <json-c/json.h>

#define DEFAULT_PATH "/tmp/kappai.monitor"
#define MAX_OUTPUT 16384
#define MAX_EXPERTS 256
#define MAX_LAYERS 128
#define MAX_TOPK 64
#define MAX_LOAD_STEPS 24

enum { V_OUTPUT = 0, V_HEATMAP, V_SORTED };

enum {
	C_PHASE = 1,
	C_GOOD,
	C_ERR,
};

typedef struct {
	char arch[64];
	char model_path[128];
	int	 n_layers;
	int	 dim;
	int	 n_ctx;
	int	 vocab;
	int	 is_moe;
	int	 n_experts;
	int	 topk;

	char   phase[16];
	int	   decoding;
	int	   token_idx;
	int	   cur_layer;
	double pct;
	double cumulative_tps;

	double pp_tps;
	double tg_tps;
	int	   n_prefill;
	int	   n_generated;

	char	 load_phase[32];
	int		 load_ms;
	int		 load_done;
	int		 load_error;
	uint64_t load_t0_ms;

	int phase_tokens;
	int tokens_done;

	double			   moe_hit;
	double			   moe_pin;
	double			   moe_lru;
	unsigned long long moe_misses;

	char output[MAX_OUTPUT];
	int	 output_len;

	int expert_hits[MAX_LAYERS][MAX_EXPERTS];

	int	  cur_experts[MAX_LAYERS][MAX_TOPK];
	int	  cur_n_experts[MAX_LAYERS];
	float cur_weights[MAX_LAYERS][MAX_TOPK];
	int	  has_current_token;

	int expert_total[MAX_EXPERTS];

	char load_steps[MAX_LOAD_STEPS][48];
	int	 n_load_steps;

	int scroll_x;
	int scroll_y;
} monitor_state;

static volatile sig_atomic_t g_running = 1;

static void on_sigint(int sig) {
	(void)sig;
	g_running = 0;
}

static uint64_t now_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

static void append_output(monitor_state *st, const char *text) {
	int n = (int)strlen(text);
	if (st->output_len + n >= MAX_OUTPUT - 1) {
		int half = st->output_len / 2;
		memmove(st->output, st->output + half, (size_t)(st->output_len - half));
		st->output_len -= half;
	}
	memcpy(st->output + st->output_len, text, (size_t)n);
	st->output_len += n;
	st->output[st->output_len] = '\0';
}

static int wrap_text(const char *text, int width, char *lines[], int max_lines) {
	int			count = 0;
	const char *p	  = text;
	while (*p && count < max_lines) {
		while (*p == ' ')
			p++;
		const char *nl	= strchr(p, '\n');
		int			len = nl ? (int)(nl - p) : (int)strlen(p);
		if (len == 0) {
			if (nl) {
				p = nl + 1;
				continue;
			}
			break;
		}
		if (len <= width) {
			char *line = malloc((size_t)len + 1);
			if (!line)
				break;
			memcpy(line, p, (size_t)len);
			line[len]	   = '\0';
			lines[count++] = line;
			p			   = nl ? nl + 1 : p + len;
		} else {
			int ba = width;
			while (ba > 0 && p[ba] != ' ')
				ba--;
			if (ba == 0)
				ba = width;
			char *line = malloc((size_t)ba + 1);
			if (!line)
				break;
			memcpy(line, p, (size_t)ba);
			line[ba]	   = '\0';
			lines[count++] = line;
			p += ba;
			if (*p == ' ')
				p++;
		}
	}
	return count;
}

static void free_lines(char *lines[], int n) {
	for (int i = 0; i < n; i++)
		free(lines[i]);
}

static void pretty_phase(char *out, size_t n, const char *p) {
	size_t j   = 0;
	int	   cap = 1;
	for (size_t i = 0; p[i] && j < n - 1; i++) {
		char c = p[i];
		if (c == '_') {
			if (j > 0 && out[j - 1] != ' ')
				out[j++] = ' ';
			cap = 1;
		} else {
			out[j++] = cap ? (char)toupper((unsigned char)c) : c;
			cap		 = 0;
		}
	}
	out[j] = '\0';
}

static void curses_colors(void) {
	if (!has_colors())
		return;
	start_color();
	use_default_colors();
	init_pair(C_PHASE, COLOR_YELLOW, -1);
	init_pair(C_GOOD, COLOR_GREEN, -1);
	init_pair(C_ERR, COLOR_RED, -1);
}

static void draw_ubar(WINDOW *win, int y, int x, int w, double pct) {
	if (w <= 0)
		return;
	int filled = (int)(pct / 100.0 * (double)w + 0.5);
	if (filled < 0)
		filled = 0;
	if (filled > w)
		filled = w;
	if (filled > 0) {
		char *fb = malloc((size_t)filled * 3 + 1);
		if (fb) {
			char *q = fb;
			for (int i = 0; i < filled; i++) {
				memcpy(q, "\xe2\x96\x93", 3);
				q += 3;
			}
			*q = '\0';
			wattron(win, COLOR_PAIR(C_GOOD));
			mvwprintw(win, y, x, "%s", fb);
			wattroff(win, COLOR_PAIR(C_GOOD));
			free(fb);
		}
	}
	if (w - filled > 0) {
		char *ub = malloc((size_t)(w - filled) * 3 + 1);
		if (ub) {
			char *q = ub;
			for (int i = filled; i < w; i++) {
				memcpy(q, "\xe2\x96\x91", 3);
				q += 3;
			}
			*q = '\0';
			wattron(win, A_DIM);
			mvwprintw(win, y, x + filled, "%s", ub);
			wattroff(win, A_DIM);
			free(ub);
		}
	}
}

static int is_prompt_phase(const monitor_state *st) {
	if (st->n_generated > 0)
		return 0;
	if (strcmp(st->phase, "prefill") == 0 || strcmp(st->phase, "warmup") == 0)
		return 1;
	return st->phase[0] && st->phase_tokens > 0 && st->tokens_done >= 0;
}

static int is_decode_phase(const monitor_state *st) {
	return st->decoding || (st->phase[0] && strcmp(st->phase, "decode") == 0);
}

static void draw_divider(WINDOW *win, int y, int cols) {
	mvwaddch(win, y, 0, ACS_LTEE);
	mvwhline(win, y, 1, ACS_HLINE, cols - 2);
	mvwaddch(win, y, cols - 1, ACS_RTEE);
}

static void draw_vscrollbar(WINDOW *win, int y0, int y1, int cols, int pos, int view, int total) {
	int h = y1 - y0 + 1;
	if (h < 2 || view >= total)
		return;
	int x  = cols - 2;
	int th = h * view / total;
	if (th < 1)
		th = 1;
	if (th > h)
		th = h;
	int ts = (h - th) * pos / (total - view);
	if (ts < 0)
		ts = 0;
	for (int i = 0; i < h; i++) {
		int y = y0 + i;
		if (i >= ts && i < ts + th) {
			wattron(win, A_REVERSE);
			mvwaddch(win, y, x, ' ');
			wattroff(win, A_REVERSE);
		} else {
			wattron(win, A_DIM);
			mvwprintw(win, y, x, "\xe2\x94\x83");
			wattroff(win, A_DIM);
		}
	}
}

static void draw_hscrollbar(WINDOW *win, int y, int x0, int x1, int pos, int view, int total) {
	int w = x1 - x0 + 1;
	if (w < 2 || view >= total)
		return;
	int tw = w * view / total;
	if (tw < 1)
		tw = 1;
	if (tw > w)
		tw = w;
	int ts = (w - tw) * pos / (total - view);
	if (ts < 0)
		ts = 0;
	for (int i = 0; i < w; i++) {
		int x = x0 + i;
		if (i >= ts && i < ts + tw) {
			wattron(win, A_REVERSE);
			mvwaddch(win, y, x, ' ');
			wattroff(win, A_REVERSE);
		} else {
			wattron(win, A_DIM);
			mvwprintw(win, y, x, "\xe2\x94\x81");
			wattroff(win, A_DIM);
		}
	}
}

static void status_right(WINDOW *win, int y, int cols, const char *rhs) {
	int x = cols - 1 - (int)strlen(rhs);
	if (x < 1)
		x = 1;
	mvwprintw(win, y, x, "%s", rhs);
}

static void clip_print(WINDOW *win, int y, int x, int cols, const char *s) {
	int maxw = cols - 1 - x;
	if (maxw <= 0)
		return;
	mvwprintw(win, y, x, "%.*s", maxw, s);
}

static void phase_label(WINDOW *win, int y, int x, const char *label, int active) {
	if (active) {
		wattron(win, COLOR_PAIR(C_PHASE) | A_BOLD);
		mvwprintw(win, y, x, "%s", label);
		wattroff(win, COLOR_PAIR(C_PHASE) | A_BOLD);
	} else {
		wattron(win, A_DIM);
		mvwprintw(win, y, x, "%s", label);
		wattroff(win, A_DIM);
	}
}

static int render_top(WINDOW *win, monitor_state *st, int cols) {
	int y = 1;

	int active = !st->load_done || st->phase[0] != 0;
	if (st->load_error) {
		wattron(win, COLOR_PAIR(C_ERR) | A_BOLD);
		mvwprintw(win, y, 1, "\xe2\x97\x8f");
		wattroff(win, COLOR_PAIR(C_ERR) | A_BOLD);
	} else if (active) {
		wattron(win, COLOR_PAIR(C_GOOD) | A_BOLD);
		mvwprintw(win, y, 1, "\xe2\x97\x8f");
		wattroff(win, COLOR_PAIR(C_GOOD) | A_BOLD);
	} else {
		wattron(win, A_DIM);
		mvwprintw(win, y, 1, "\xe2\x97\x8f");
		wattroff(win, A_DIM);
	}

	char rhs[64];
	int	 x = 3;

	if (!st->load_done) {
		char ph[48];
		pretty_phase(ph, sizeof(ph), st->load_phase[0] ? st->load_phase : "starting");
		phase_label(win, y, x, "Loading", 1);
		wattron(win, A_DIM);
		mvwprintw(win, y, 11, "%s", ph);
		wattroff(win, A_DIM);

		uint64_t elapsed = st->load_t0_ms ? now_ms() - st->load_t0_ms : 0;
		snprintf(rhs, sizeof(rhs), "%llu s", (unsigned long long)(elapsed / 1000));
		status_right(win, y, cols, rhs);
		y++;

		int bw = cols - 28;
		if (bw < 8)
			bw = 8;
		if (strcmp(st->load_phase, "loading_weights") == 0 && st->n_layers > 0) {
			draw_ubar(win, y, 1, bw, st->pct);
			char tail[48];
			snprintf(tail, sizeof(tail), " %3.0f%%", st->pct);
			clip_print(win, y, bw + 3, cols, tail);
			y++;
			wattron(win, A_DIM);
			mvwprintw(win, y, 1, "layer %d/%d", st->cur_layer, st->n_layers);
			wattroff(win, A_DIM);
			y++;
		} else {
			wattron(win, A_DIM);
			mvwprintw(win, y, 1, "(working...)");
			wattroff(win, A_DIM);
			y += 2;
		}

		draw_divider(win, y, cols);
		y++;
		if (st->model_path[0]) {
			wattron(win, A_DIM);
			clip_print(win, y, 1, cols, st->model_path);
			wattroff(win, A_DIM);
		}
		y++;
		if (st->load_error) {
			wattron(win, COLOR_PAIR(C_ERR) | A_BOLD);
			mvwprintw(win, y, 1, "load failed");
			wattroff(win, COLOR_PAIR(C_ERR) | A_BOLD);
			y++;
		}
		draw_divider(win, y, cols);
		return y + 1;
	}

	int running = st->phase[0] != 0;
	int prompt	= is_prompt_phase(st);
	int decode	= is_decode_phase(st);

	if (prompt) {
		double show = st->cumulative_tps > 0 ? st->cumulative_tps : st->pp_tps;
		snprintf(rhs, sizeof(rhs), "%.2f t/s", show);
		phase_label(win, y, x, "Prompt", 1);
	} else if (decode) {
		double show = st->cumulative_tps > 0 ? st->cumulative_tps : st->tg_tps;
		snprintf(rhs, sizeof(rhs), "%.2f t/s", show);
		phase_label(win, y, x, "Generating", 1);
	} else if (running) {
		char lbl[16];
		pretty_phase(lbl, sizeof(lbl), st->phase);
		snprintf(rhs, sizeof(rhs), "%.2f t/s", st->cumulative_tps);
		phase_label(win, y, x, lbl, 1);
	} else {
		if (st->pp_tps > 0 || st->tg_tps > 0)
			snprintf(rhs, sizeof(rhs), "PP %.2f / TG %.2f t/s", st->pp_tps, st->tg_tps);
		else
			snprintf(rhs, sizeof(rhs), "--");
		phase_label(win, y, x, "Idle", 0);
	}
	status_right(win, y, cols, rhs);
	y++;

	int bw = cols - 28;
	if (bw < 8)
		bw = 8;
	if (prompt) {
		int	   total = st->phase_tokens > 0 ? st->phase_tokens : st->n_prefill;
		int	   done	 = st->tokens_done >= 0 ? st->tokens_done : 0;
		double pct	 = 0.0;
		if (total > 0) {
			pct = 100.0 * (double)done / (double)total;
			if (pct > 100.0)
				pct = 100.0;
		}
		draw_ubar(win, y, 1, bw, pct);
		char tail[48];
		if (total > 0)
			snprintf(tail, sizeof(tail), " %3.0f%% tokens %d/%d", pct, done, total);
		else
			snprintf(tail, sizeof(tail), " %3.0f%%", pct);
		clip_print(win, y, bw + 3, cols, tail);
	} else if (decode) {
		wattron(win, A_DIM);
		mvwprintw(win, y, 1, "tokens: %d generated", st->n_generated);
		wattroff(win, A_DIM);
	} else {
		wattron(win, A_DIM);
		mvwprintw(win, y, 1, "tokens: (idle)");
		wattroff(win, A_DIM);
	}
	y++;

	if (running && st->n_layers > 0) {
		draw_ubar(win, y, 1, bw, st->pct);
		char tail[48];
		snprintf(tail, sizeof(tail), " %3.0f%% layer %d/%d", st->pct, st->cur_layer, st->n_layers);
		clip_print(win, y, bw + 3, cols, tail);
	} else {
		wattron(win, A_DIM);
		mvwprintw(win, y, 1, "layer: (n/a)");
		wattroff(win, A_DIM);
	}
	y++;

	draw_divider(win, y, cols);
	y++;

	char info[192];
	int	 n = snprintf(info, sizeof(info), "%s | %dL | %dd | ctx %d | vocab %d",
					  st->arch[0] ? st->arch : "?", st->n_layers, st->dim, st->n_ctx, st->vocab);
	if (st->is_moe)
		snprintf(info + n, sizeof(info) - (size_t)n, " | MoE %de/%d", st->n_experts, st->topk);
	wattron(win, A_DIM);
	clip_print(win, y, 1, cols, info);
	wattroff(win, A_DIM);
	y++;

	if (st->is_moe) {
		char vals[128];
		snprintf(vals, sizeof(vals), "hit=%.1f%% pin=%.1f%% lru=%.1f%% miss=%llu", st->moe_hit,
				 st->moe_pin, st->moe_lru, st->moe_misses);
		wattron(win, A_DIM);
		mvwprintw(win, y, 1, "cache");
		wattroff(win, A_DIM);
		mvwprintw(win, y, 7, "%.*s", cols - 8, vals);
		y++;
	}

	draw_divider(win, y, cols);
	return y + 1;
}

static void render_loading(WINDOW *win, monitor_state *st, int y0, int max_y, int cols) {
	(void)cols;
	if (y0 >= max_y)
		return;
	int n	  = st->n_load_steps;
	int show  = max_y - y0;
	int start = n > show ? n - show : 0;

	for (int i = start; i < n && y0 < max_y; i++) {
		char ph[48];
		pretty_phase(ph, sizeof(ph), st->load_steps[i]);
		int is_last = (i == n - 1);

		if (st->load_error && is_last) {
			wattron(win, COLOR_PAIR(C_ERR) | A_BOLD);
			mvwprintw(win, y0, 1, "[!] %s", ph);
			wattroff(win, COLOR_PAIR(C_ERR) | A_BOLD);
		} else if (is_last) {
			wattron(win, COLOR_PAIR(C_PHASE) | A_BOLD);
			mvwprintw(win, y0, 1, "[*] %s", ph);
			wattroff(win, COLOR_PAIR(C_PHASE) | A_BOLD);
		} else {
			wattron(win, COLOR_PAIR(C_GOOD));
			mvwprintw(win, y0, 1, "[+] %s", ph);
			wattroff(win, COLOR_PAIR(C_GOOD));
		}
		y0++;
	}
}

static void render_output(WINDOW *win, monitor_state *st, int y0, int max_y, int cols) {
	int uc = cols > 2 ? cols - 2 : 1;
	if (y0 >= max_y)
		return;
	int avail = max_y - y0;
	if (st->output_len == 0) {
		wattron(win, A_DIM);
		mvwprintw(win, y0, 1, "(waiting for output...)");
		wattroff(win, A_DIM);
		return;
	}

	char *lines[128];
	int	  nl = wrap_text(st->output, uc, lines, 128);
	int	  sl = nl > avail ? nl - avail : 0;
	for (int i = sl; i < nl && y0 < max_y; i++)
		mvwprintw(win, y0++, 1, "%s", lines[i]);
	free_lines(lines, nl);
}

static void render_sorted(WINDOW *win, monitor_state *st, int y0, int max_y, int cols, int *sy) {
	int n_experts = st->n_experts;
	if (n_experts <= 0 || n_experts > MAX_EXPERTS) {
		mvwprintw(win, y0, 1, "(no MoE data)");
		return;
	}

	int idx[MAX_EXPERTS];
	for (int i = 0; i < n_experts; i++)
		idx[i] = i;
	for (int i = 1; i < n_experts; i++) {
		int key		 = idx[i];
		int key_hits = st->expert_total[key];
		int j		 = i - 1;
		while (j >= 0 && st->expert_total[idx[j]] < key_hits) {
			idx[j + 1] = idx[j];
			j--;
		}
		idx[j + 1] = key;
	}

	unsigned long long total_dispatches = 0;
	for (int i = 0; i < n_experts; i++)
		total_dispatches += st->expert_total[i];
	if (total_dispatches == 0)
		total_dispatches = 1;

	int vw = cols - 3;
	if (vw < 20)
		vw = 20;
	int body_rows = max_y - y0 - 1;
	if (body_rows < 1)
		return;

	int max_sy = n_experts - body_rows;
	if (max_sy < 0)
		max_sy = 0;
	if (*sy > max_sy)
		*sy = max_sy;
	if (*sy < 0)
		*sy = 0;
	int has_v = n_experts > body_rows;
	int top	  = y0;

	mvwprintw(win, y0, 1, "%-6s %-7s %-7s %s", "EID", "Hits", "Share", "Bar");
	y0++;

	int max_hits = st->expert_total[idx[0]];
	if (max_hits <= 0)
		max_hits = 1;
	int bar_w = vw - 22;
	if (bar_w < 5)
		bar_w = 5;

	for (int i = 0; i < body_rows; i++) {
		int row = *sy + i;
		if (row >= n_experts)
			break;
		int	   eid		 = idx[row];
		int	   hits		 = st->expert_total[eid];
		double share_pct = 100.0 * (double)hits / (double)total_dispatches;
		double rel_pct	 = 100.0 * (double)hits / (double)max_hits;
		int	   filled	 = (int)(rel_pct / 100.0 * (double)bar_w);
		if (filled > bar_w)
			filled = bar_w;

		mvwprintw(win, y0, 1, "%-6d %-7d %5.1f%% ", eid, hits, share_pct);
		for (int j = 0; j < bar_w; j++) {
			if (j < filled) {
				wattron(win, A_REVERSE);
				waddch(win, ' ');
				wattroff(win, A_REVERSE);
			} else {
				wattron(win, A_DIM);
				waddch(win, '-');
				wattroff(win, A_DIM);
			}
		}
		y0++;
	}

	if (has_v)
		draw_vscrollbar(win, top, max_y - 1, cols, *sy, body_rows, n_experts);
}

static void render_heatmap(WINDOW *win, monitor_state *st, int y0, int max_y, int cols, int *sx,
						   int *sy) {
	int n_layers  = st->n_layers;
	int n_experts = st->n_experts;
	if (n_layers <= 0 || n_experts <= 0) {
		mvwprintw(win, y0, 1, "(no MoE data)");
		return;
	}
	if (n_layers > MAX_LAYERS)
		n_layers = MAX_LAYERS;
	if (n_experts > MAX_EXPERTS)
		n_experts = MAX_EXPERTS;

	const int label_w	= 5;
	int		  view_cols = cols - 8;
	if (view_cols < 1)
		view_cols = 1;

	int has_h	   = n_experts > view_cols;
	int legend_row = has_h ? max_y - 2 : max_y - 1;
	if (legend_row <= y0)
		legend_row = y0;

	int y = y0;

	if (y < legend_row) {
		wattron(win, A_DIM);
		mvwprintw(win, y, 1, "%-*s", label_w, "exp");
		wattroff(win, A_DIM);
		for (int j = 0; j < view_cols; j++) {
			int e = *sx + j;
			if (e >= n_experts)
				break;
			if (e % 8 == 0) {
				int x = 1 + label_w + j;
				if (x < cols - 5) {
					wattron(win, A_DIM);
					mvwprintw(win, y, x, "%d", e);
					wattroff(win, A_DIM);
				}
			}
		}
		y++;
	}

	int body_rows = legend_row - y;
	if (body_rows < 1)
		body_rows = 1;
	if (body_rows > n_layers)
		body_rows = n_layers;

	int max_sy = n_layers - body_rows;
	if (max_sy < 0)
		max_sy = 0;
	if (*sy > max_sy)
		*sy = max_sy;
	if (*sy < 0)
		*sy = 0;
	int max_sx = n_experts - view_cols;
	if (max_sx < 0)
		max_sx = 0;
	if (*sx > max_sx)
		*sx = max_sx;
	if (*sx < 0)
		*sx = 0;

	int has_v = n_layers > body_rows;

	int max_hits = 1;
	for (int l = 0; l < n_layers; l++)
		for (int e = 0; e < n_experts; e++)
			if (st->expert_hits[l][e] > max_hits)
				max_hits = st->expert_hits[l][e];

	for (int i = 0; i < body_rows; i++) {
		int l = *sy + i;
		if (l >= n_layers)
			break;

		wattron(win, A_DIM);
		mvwprintw(win, y, 1, "L%3d ", l);
		wattroff(win, A_DIM);

		int is_active = st->has_current_token && l < st->n_layers && st->cur_n_experts[l] > 0;

		for (int j = 0; j < view_cols; j++) {
			int e = *sx + j;
			if (e >= n_experts)
				break;
			int x = 1 + label_w + j;

			int is_cur = 0;
			if (is_active) {
				for (int k = 0; k < st->cur_n_experts[l]; k++) {
					if (st->cur_experts[l][k] == e) {
						is_cur = 1;
						break;
					}
				}
			}

			if (is_cur) {
				wattron(win, COLOR_PAIR(C_GOOD) | A_BOLD);
				mvwprintw(win, y, x, "#");
				wattroff(win, COLOR_PAIR(C_GOOD) | A_BOLD);
			} else {
				int hits = st->expert_hits[l][e];
				if (hits <= 0) {
					wattron(win, A_DIM);
					mvwprintw(win, y, x, ".");
					wattroff(win, A_DIM);
				} else {
					int lvl = (hits * 3) / max_hits;
					if (lvl > 2)
						lvl = 2;
					mvwprintw(win, y, x, "%s", "\xe2\x96\x91\xe2\x96\x92\xe2\x96\x93" + lvl * 3);
				}
			}
		}
		y++;
	}

	if (y <= legend_row && legend_row < max_y) {
		wattron(win, A_DIM);
		mvwprintw(win, legend_row, 1,
				  ". empty  \xe2\x96\x91\xe2\x96\x92\xe2\x96\x93 intensity  # active");
		wattroff(win, A_DIM);
	}

	if (has_v)
		draw_vscrollbar(win, y0, max_y - 1, cols, *sy, body_rows, n_layers);
	if (has_h)
		draw_hscrollbar(win, max_y - 1, 1, cols - 3, *sx, view_cols, n_experts);
}

static void render(WINDOW *win, monitor_state *st, int view_mode) {
	int rows, cols;
	getmaxyx(win, rows, cols);
	werase(win);
	if (rows < 10 || cols < 30)
		return;

	box(win, ACS_VLINE, ACS_HLINE);

	int content_start = render_top(win, st, cols);

	int keybar_row	= rows - 2;
	int bottom_div	= keybar_row - 1;
	int content_end = bottom_div;
	if (content_end < content_start)
		content_end = content_start;

	if (!st->load_done) {
		render_loading(win, st, content_start, content_end, cols);
	} else if (view_mode == V_OUTPUT) {
		render_output(win, st, content_start, content_end, cols);
	} else if (view_mode == V_HEATMAP) {
		render_heatmap(win, st, content_start, content_end, cols, &st->scroll_x, &st->scroll_y);
	} else {
		render_sorted(win, st, content_start, content_end, cols, &st->scroll_y);
	}

	draw_divider(win, bottom_div, cols);
	wattron(win, A_DIM);
	mvwprintw(win, keybar_row, 1, " q:quit  c:clear  o:output  h:heatmap  s:sorted  space:cycle ");
	wattroff(win, A_DIM);
	wrefresh(win);
}

static void record_load_step(monitor_state *st, const char *phase) {
	if (st->n_load_steps == 0 || strcmp(st->load_steps[st->n_load_steps - 1], phase) != 0) {
		if (st->n_load_steps < MAX_LOAD_STEPS) {
			snprintf(st->load_steps[st->n_load_steps], sizeof(st->load_steps[0]), "%s", phase);
			st->n_load_steps++;
		}
	}
}

static void process_event(monitor_state *st, struct json_object *root) {
	struct json_object *jtype;
	if (!json_object_object_get_ex(root, "type", &jtype))
		return;
	const char *type = json_object_get_string(jtype);

	if (strcmp(type, "start") == 0) {
		struct json_object *j;
		if (json_object_object_get_ex(root, "arch", &j))
			snprintf(st->arch, sizeof(st->arch), "%s", json_object_get_string(j));
		if (json_object_object_get_ex(root, "n_layers", &j))
			st->n_layers = json_object_get_int(j);
		if (json_object_object_get_ex(root, "dim", &j))
			st->dim = json_object_get_int(j);
		if (json_object_object_get_ex(root, "n_ctx", &j))
			st->n_ctx = json_object_get_int(j);
		if (json_object_object_get_ex(root, "vocab", &j))
			st->vocab = json_object_get_int(j);
		if (json_object_object_get_ex(root, "is_moe", &j))
			st->is_moe = json_object_get_int(j);
		if (json_object_object_get_ex(root, "n_experts", &j))
			st->n_experts = json_object_get_int(j);
		if (json_object_object_get_ex(root, "topk", &j))
			st->topk = json_object_get_int(j);

		st->load_done		  = 1;
		st->phase[0]		  = '\0';
		st->decoding		  = 0;
		st->token_idx		  = 0;
		st->cur_layer		  = 0;
		st->pct				  = 0;
		st->cumulative_tps	  = 0;
		st->pp_tps			  = 0;
		st->tg_tps			  = 0;
		st->n_prefill		  = 0;
		st->n_generated		  = 0;
		st->phase_tokens	  = 0;
		st->tokens_done		  = 0;
		st->has_current_token = 0;
	} else if (strcmp(type, "load") == 0) {
		struct json_object *j;
		if (json_object_object_get_ex(root, "phase", &j))
			snprintf(st->load_phase, sizeof(st->load_phase), "%s", json_object_get_string(j));
		if (json_object_object_get_ex(root, "ms", &j))
			st->load_ms = json_object_get_int(j);
		if (json_object_object_get_ex(root, "layers", &j))
			st->n_layers = json_object_get_int(j);
		if (json_object_object_get_ex(root, "dim", &j))
			st->dim = json_object_get_int(j);
		if (json_object_object_get_ex(root, "vocab", &j))
			st->vocab = json_object_get_int(j);
		if (json_object_object_get_ex(root, "path", &j))
			snprintf(st->model_path, sizeof(st->model_path), "%s", json_object_get_string(j));
		if (!st->load_t0_ms)
			st->load_t0_ms = now_ms();

		record_load_step(st, st->load_phase);

		if (strcmp(st->load_phase, "loading_weights") == 0) {
			if (json_object_object_get_ex(root, "layer", &j))
				st->cur_layer = json_object_get_int(j);
			if (json_object_object_get_ex(root, "n_layers", &j))
				st->n_layers = json_object_get_int(j);
			if (json_object_object_get_ex(root, "pct", &j))
				st->pct = json_object_get_double(j);
		}

		if (strcmp(st->load_phase, "model_load_done") == 0)
			st->load_done = 1;
		else if (strcmp(st->load_phase, "model_load_failed") == 0)
			st->load_error = 1;
	} else if (strcmp(type, "prefill") == 0) {
		struct json_object *j;
		if (json_object_object_get_ex(root, "tps", &j))
			st->pp_tps = json_object_get_double(j);
		if (json_object_object_get_ex(root, "n_tokens", &j)) {
			st->n_prefill	 = json_object_get_int(j);
			st->phase_tokens = st->n_prefill;
			st->tokens_done	 = st->n_prefill;
		}
		st->cumulative_tps = 0;
		snprintf(st->phase, sizeof(st->phase), "prefill");
	} else if (strcmp(type, "layer") == 0) {
		struct json_object *j;
		if (json_object_object_get_ex(root, "phase", &j))
			snprintf(st->phase, sizeof(st->phase), "%s", json_object_get_string(j));
		if (strcmp(st->phase, "decode") == 0)
			st->decoding = 1;
		if (json_object_object_get_ex(root, "token_idx", &j))
			st->token_idx = json_object_get_int(j);
		if (json_object_object_get_ex(root, "n_tokens", &j))
			st->phase_tokens = json_object_get_int(j);
		if (json_object_object_get_ex(root, "tokens_done", &j))
			st->tokens_done = json_object_get_int(j);
		if (json_object_object_get_ex(root, "layer", &j))
			st->cur_layer = json_object_get_int(j);
		if (json_object_object_get_ex(root, "n_layers", &j))
			st->n_layers = json_object_get_int(j);
		if (json_object_object_get_ex(root, "pct", &j))
			st->pct = json_object_get_double(j);
		if (json_object_object_get_ex(root, "cumulative_tps", &j))
			st->cumulative_tps = json_object_get_double(j);
		if (json_object_object_get_ex(root, "moe_hit", &j))
			st->moe_hit = json_object_get_double(j);
		if (json_object_object_get_ex(root, "moe_pin", &j))
			st->moe_pin = json_object_get_double(j);
		if (json_object_object_get_ex(root, "moe_lru", &j))
			st->moe_lru = json_object_get_double(j);
		if (json_object_object_get_ex(root, "moe_miss", &j))
			st->moe_misses = (unsigned long long)json_object_get_int64(j);
	} else if (strcmp(type, "token") == 0) {
		struct json_object *j;
		if (json_object_object_get_ex(root, "text", &j))
			append_output(st, json_object_get_string(j));
		if (json_object_object_get_ex(root, "token_idx", &j))
			st->token_idx = json_object_get_int(j);
		st->n_generated		  = st->token_idx + 1;
		st->decoding		  = 1;
		st->has_current_token = 1;
	} else if (strcmp(type, "moe_experts") == 0) {
		struct json_object *jlayer;
		struct json_object *jexperts;
		struct json_object *jweights;
		int					layer = -1;
		if (json_object_object_get_ex(root, "layer", &jlayer))
			layer = json_object_get_int(jlayer);
		if (layer < 0 || layer >= MAX_LAYERS)
			return;

		st->cur_n_experts[layer] = 0;

		if (json_object_object_get_ex(root, "experts", &jexperts) &&
			json_object_is_type(jexperts, json_type_array)) {
			int n = json_object_array_length(jexperts);
			if (n > MAX_TOPK)
				n = MAX_TOPK;
			st->cur_n_experts[layer] = n;
			for (int k = 0; k < n; k++) {
				struct json_object *je	  = json_object_array_get_idx(jexperts, k);
				int					eid	  = json_object_get_int(je);
				st->cur_experts[layer][k] = eid;
				if (eid >= 0 && eid < MAX_EXPERTS) {
					st->expert_hits[layer][eid]++;
					st->expert_total[eid]++;
				}
			}
		}
		if (json_object_object_get_ex(root, "weights", &jweights) &&
			json_object_is_type(jweights, json_type_array)) {
			int n = json_object_array_length(jweights);
			if (n > MAX_TOPK)
				n = MAX_TOPK;
			for (int k = 0; k < n; k++) {
				struct json_object *jw	  = json_object_array_get_idx(jweights, k);
				st->cur_weights[layer][k] = (float)json_object_get_double(jw);
			}
		}
	} else if (strcmp(type, "end") == 0) {
		struct json_object *j;
		if (json_object_object_get_ex(root, "pp_tps", &j))
			st->pp_tps = json_object_get_double(j);
		if (json_object_object_get_ex(root, "tg_tps", &j))
			st->tg_tps = json_object_get_double(j);
		if (json_object_object_get_ex(root, "tokens_generated", &j))
			st->n_generated = json_object_get_int(j);
		st->cumulative_tps	  = 0;
		st->phase[0]		  = '\0';
		st->decoding		  = 0;
		st->has_current_token = 0;

		if (json_object_object_get_ex(root, "moe_hit", &j))
			st->moe_hit = json_object_get_double(j);
		if (json_object_object_get_ex(root, "moe_pin", &j))
			st->moe_pin = json_object_get_double(j);
		if (json_object_object_get_ex(root, "moe_lru", &j))
			st->moe_lru = json_object_get_double(j);
		if (json_object_object_get_ex(root, "moe_miss", &j))
			st->moe_misses = (unsigned long long)json_object_get_int64(j);
	}
}

static int connect_monitor(const char *path, int retry_ms, int max_wait_ms) {
	uint64_t waited = 0;
	while (1) {
		int fd = socket(AF_UNIX, SOCK_STREAM, 0);
		if (fd < 0) {
			perror("socket");
			return -1;
		}
		struct sockaddr_un addr;
		memset(&addr, 0, sizeof(addr));
		addr.sun_family = AF_UNIX;
		strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
		if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0)
			return fd;
		close(fd);
		if (max_wait_ms >= 0 && waited >= (uint64_t)max_wait_ms) {
			fprintf(stderr, "Cannot connect to %s: %s\n", path, strerror(errno));
			return -1;
		}
		if (waited == 0)
			fprintf(stderr, "Waiting for monitor socket at %s...\n", path);
		usleep((useconds_t)retry_ms * 1000);
		waited += retry_ms;
		if (!g_running)
			return -1;
	}
}

int main(int argc, char **argv) {
	const char *path = (argc > 1) ? argv[1] : DEFAULT_PATH;
	signal(SIGINT, on_sigint);

	int fd = connect_monitor(path, 200, 10000);
	if (fd < 0)
		return 1;

	setlocale(LC_ALL, "");
	WINDOW *win = initscr();
	cbreak();
	noecho();
	nodelay(win, TRUE);
	keypad(win, TRUE);
	mousemask(ALL_MOUSE_EVENTS, NULL);
	curs_set(0);
	curses_colors();

	monitor_state st;
	memset(&st, 0, sizeof(st));

	char linebuf[8192];
	int	 linepos   = 0;
	int	 view_mode = 0;
	int	 prev_view = 0;

	while (g_running) {
		fd_set rfds;
		FD_ZERO(&rfds);
		FD_SET(fd, &rfds);
		struct timeval tv = {0, 100000};
		int			   rv = select(fd + 1, &rfds, NULL, NULL, &tv);
		if (rv > 0) {
			char	buf[4096];
			ssize_t n = read(fd, buf, sizeof(buf));
			if (n <= 0)
				break;
			for (ssize_t i = 0; i < n; i++) {
				if (buf[i] == '\n' || linepos >= (int)sizeof(linebuf) - 1) {
					linebuf[linepos] = '\0';
					if (linepos > 0) {
						struct json_object *root = json_tokener_parse(linebuf);
						if (root) {
							process_event(&st, root);
							json_object_put(root);
						}
					}
					linepos = 0;
				} else {
					linebuf[linepos++] = buf[i];
				}
			}
		}

		int scroll_dx = 0, scroll_dy = 0, quit = 0;
		int ch;
		while ((ch = getch()) != ERR) {
			if (ch == 'q' || ch == 'Q') {
				quit = 1;
				break;
			}
			if (ch == 'c' || ch == 'C') {
				st.output_len = 0;
				st.output[0]  = '\0';
			}
			if (ch == 'o' || ch == 'O' || ch == 't' || ch == 'T')
				view_mode = V_OUTPUT;
			if (ch == 'h' || ch == 'H')
				view_mode = V_HEATMAP;
			if (ch == 's' || ch == 'S')
				view_mode = V_SORTED;
			if (ch == ' ' || ch == '\t' || ch == 'v' || ch == 'V')
				view_mode = (view_mode + 1) % 3;

			if (view_mode == V_HEATMAP || view_mode == V_SORTED) {
				switch (ch) {
				case KEY_UP:
					scroll_dy--;
					break;
				case KEY_DOWN:
					scroll_dy++;
					break;
				case KEY_LEFT:
					scroll_dx--;
					break;
				case KEY_RIGHT:
					scroll_dx++;
					break;
				case KEY_PPAGE:
					scroll_dy -= 10;
					break;
				case KEY_NPAGE:
					scroll_dy += 10;
					break;
				case KEY_HOME:
					scroll_dx = INT_MIN;
					scroll_dy = INT_MIN;
					break;
				case KEY_END:
					scroll_dx = INT_MAX;
					scroll_dy = INT_MAX;
					break;
				case KEY_MOUSE: {
					MEVENT me;
					if (getmouse(&me) == OK) {
						if (me.bstate & BUTTON4_PRESSED)
							scroll_dy--;
						else if (me.bstate & BUTTON5_PRESSED)
							scroll_dy++;
					}
					break;
				}
				}
			}
		}
		if (quit)
			break;

		if (view_mode != prev_view) {
			prev_view	= view_mode;
			st.scroll_x = 0;
			st.scroll_y = 0;
		}
		if (scroll_dx || scroll_dy) {
			st.scroll_x += scroll_dx;
			st.scroll_y += scroll_dy;
		}

		render(win, &st, view_mode);
	}

	endwin();
	close(fd);
	return 0;
}
