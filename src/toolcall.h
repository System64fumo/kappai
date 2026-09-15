#ifndef TOOLCALL_H
#define TOOLCALL_H

#include "common.h"
#include "markers.h"

#include <json-c/json.h>
#include <stddef.h>

typedef struct {
	char  *p;
	size_t len;
	size_t cap;
} toolcall_buf;

void toolcall_buf_append(toolcall_buf *b, const char *s, size_t n);
void toolcall_buf_reset(toolcall_buf *b);

typedef void (*toolcall_content_fn)(void *ud, const char *piece, size_t n);
typedef void (*toolcall_call_fn)(void *ud, int index, const char *id, const char *name,
								 const char *args_json);

typedef struct toolcall_scanner toolcall_scanner;

toolcall_scanner *toolcall_scanner_new(const marker_pair *fmt, toolcall_buf *content,
									   toolcall_content_fn on_content, toolcall_call_fn on_call,
									   void *ud);

void toolcall_scanner_feed(toolcall_scanner *sc);

void toolcall_scanner_feed_capture(toolcall_scanner *sc, const char *piece, size_t n);

bool toolcall_scanner_in_capture(const toolcall_scanner *sc);

bool toolcall_scanner_suppressed(const toolcall_scanner *sc);

void toolcall_scanner_finish(toolcall_scanner *sc);

size_t		 toolcall_scanner_n_calls(const toolcall_scanner *sc);
json_object *toolcall_scanner_calls(const toolcall_scanner *sc);

void toolcall_scanner_free(toolcall_scanner *sc);

status_code toolcall_parse(const marker_pair *fmt, const char *text, size_t len, char **name_out,
						   json_object **args_out, size_t *consumed, int *saw_close);

#endif
