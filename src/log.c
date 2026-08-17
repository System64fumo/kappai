#include "log.h"
#include "common.h"
#include "profile.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define COLOR_RESET "\x1b[0m"
#define PROGRESS_BAR_WIDTH 30

static log_config	   g_log	 = {.level = LOG_INFO, .color = 1, .stream = NULL};
static pthread_mutex_t g_log_mtx = PTHREAD_MUTEX_INITIALIZER;

static int g_progress_on_screen = 0;

static const char *level_name(log_level level) {
	static const char *names[] = {
		[LOG_TRACE] = "TRC", [LOG_DEBUG] = "DBG", [LOG_INFO] = "INF",
		[LOG_WARN] = "WRN",	 [LOG_ERROR] = "ERR",
	};
	if ((int)level < 0 || level >= ARRAY_LEN(names))
		return "?";
	return names[level];
}

static const char *level_color(log_level level) {
	static const char *colors[] = {
		[LOG_TRACE] = "\x1b[90m", [LOG_DEBUG] = "\x1b[36m", [LOG_INFO] = "\x1b[32m",
		[LOG_WARN] = "\x1b[33m",  [LOG_ERROR] = "\x1b[31m",
	};
	if ((int)level < 0 || level >= ARRAY_LEN(colors))
		return "";
	return colors[level];
}

log_config log_default_config(void) {
	log_config cfg = {.level = LOG_INFO, .color = isatty(fileno(stderr)), .stream = stderr};
	return cfg;
}

void log_init(log_config cfg) {
	if (!cfg.stream)
		cfg.stream = stderr;
	g_log = cfg;
}

log_level log_get_level(void) {
	return g_log.level;
}

void log_set_level(log_level level) {
	g_log.level = level;
}

static void log_line_locked(FILE *out, const char *color, const char *tag, const char *fmt,
							va_list ap) {
	if (g_log.color && color && *color)
		fprintf(out, "%s[ %-3s ]" COLOR_RESET " ", color, tag);
	else
		fprintf(out, "[ %-3s ] ", tag);
	vfprintf(out, fmt, ap);
	fputc('\n', out);
}

static void log_locked(log_level level, const char *color, const char *tag, const char *fmt,
					   va_list ap) {
	if (level < g_log.level)
		return;

	FILE *out = g_log.stream ? g_log.stream : stderr;

	pthread_mutex_lock(&g_log_mtx);
	if (g_progress_on_screen && out == g_log.stream) {
		fputs("\n", out);
		g_progress_on_screen = 0;
	}
	log_line_locked(out, color, tag, fmt, ap);
	pthread_mutex_unlock(&g_log_mtx);
}

void log_msgv(log_level level, const char *fmt, va_list ap) {
	log_locked(level, level_color(level), level_name(level), fmt, ap);
}

void log_msg(log_level level, const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	log_msgv(level, fmt, ap);
	va_end(ap);
}

void log_tag(const char *tag, const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	log_locked(LOG_INFO, "\x1b[34m", tag, fmt, ap);
	va_end(ap);
}

static void progress_render(progress *p) {
	FILE *out = g_log.stream ? g_log.stream : stderr;
	if (!p->active)
		return;

	double pct = p->total > 0 ? (100.0 * (double)p->current / (double)p->total) : 0.0;
	if (pct > 100.0)
		pct = 100.0;

	int filled = (int)(pct / 100.0 * PROGRESS_BAR_WIDTH);

	char bar[PROGRESS_BAR_WIDTH + 1];
	for (int i = 0; i < PROGRESS_BAR_WIDTH; i++)
		bar[i] = i < filled ? '#' : '-';
	bar[PROGRESS_BAR_WIDTH] = '\0';

	char line[256];
	char pct_buf[32];
	if (g_log.color)
		snprintf(pct_buf, sizeof(pct_buf), "\r\x1b[35m[ %3.0f%% ]" COLOR_RESET " ", pct);
	else
		snprintf(pct_buf, sizeof(pct_buf), "\r[ %3.0f%% ] ", pct);

	int n = snprintf(line, sizeof(line), "%s[%s] (%llu/%llu) %s", pct_buf, bar,
					 (unsigned long long)p->current, (unsigned long long)p->total, p->label);
	if (n < 0)
		return;

	int pad = p->last_len - n;
	fputs(line, out);
	for (int i = 0; i < pad; i++)
		fputc(' ', out);
	fflush(out);

	p->last_len			 = n;
	g_progress_on_screen = (out == g_log.stream);
}

void progress_start(progress *p, const char *label, uint64_t total) {
	memset(p, 0, sizeof(*p));
	snprintf(p->label, sizeof(p->label), "%s", label ? label : "");
	p->total		= total;
	p->current		= 0;
	p->active		= (g_log.level <= LOG_INFO);
	p->last_draw_ms = 0;
	progress_render(p);
}

void progress_update(progress *p, uint64_t current) {
	if (!p->active)
		return;
	p->current = current;

	uint64_t now = time_ms();
	if (current < p->total && now - p->last_draw_ms < 33)
		return;
	p->last_draw_ms = now;

	progress_render(p);
}

void progress_finish(progress *p) {
	if (!p->active)
		return;
	FILE *out = g_log.stream ? g_log.stream : stderr;
	fputc('\r', out);
	for (int i = 0; i < p->last_len; i++)
		fputc(' ', out);
	fputc('\r', out);
	fflush(out);
	if (out == g_log.stream)
		g_progress_on_screen = 0;
	p->active = 0;
}