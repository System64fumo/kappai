#ifndef LOG_H
#define LOG_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define TRACE(...)                                                                                 \
	do {                                                                                           \
		if (LOG_TRACE >= log_get_level())                                                          \
			log_msg(LOG_TRACE, __VA_ARGS__);                                                       \
	} while (0)
#define DEBUG(...)                                                                                 \
	do {                                                                                           \
		if (LOG_DEBUG >= log_get_level())                                                          \
			log_msg(LOG_DEBUG, __VA_ARGS__);                                                       \
	} while (0)
#define INFO(...)                                                                                  \
	do {                                                                                           \
		if (LOG_INFO >= log_get_level())                                                           \
			log_msg(LOG_INFO, __VA_ARGS__);                                                        \
	} while (0)
#define WARN(...)                                                                                  \
	do {                                                                                           \
		if (LOG_WARN >= log_get_level())                                                           \
			log_msg(LOG_WARN, __VA_ARGS__);                                                        \
	} while (0)
#define ERROR(...)                                                                                 \
	do {                                                                                           \
		if (LOG_ERROR >= log_get_level())                                                          \
			log_msg(LOG_ERROR, __VA_ARGS__);                                                       \
	} while (0)

typedef enum {
	LOG_TRACE = 0,
	LOG_DEBUG,
	LOG_INFO,
	LOG_WARN,
	LOG_ERROR,
} log_level;

typedef struct {
	log_level level;
	bool	  color;
	FILE	 *stream;
} log_config;

typedef struct {
	char	 label[64];
	uint64_t total;
	uint64_t current;
	uint64_t last_draw_ms;
	int		 last_len;
	int		 active;
} progress;

void	   log_init(log_config cfg);
log_config log_default_config(void);

void	  log_set_level(log_level level);
log_level log_get_level(void);

void log_msg(log_level level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void log_msgv(log_level level, const char *fmt, va_list ap);

void log_tag(const char *tag, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
bool log_color_enabled(void);

void progress_start(progress *p, const char *label, uint64_t total);
void progress_update(progress *p, uint64_t current);
void progress_finish(progress *p);

#endif
