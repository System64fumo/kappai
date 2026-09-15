#ifndef MARKERS_H
#define MARKERS_H

#include "common.h"
#include "tokenizer.h"

typedef enum {
	MARKER_THINKING,
	MARKER_TOOL_CALL,
} marker_role;

typedef enum {
	PAYLOAD_NONE = 0,
	PAYLOAD_CALLCOLON,
	PAYLOAD_AUTO,
	PAYLOAD_FUNCARGS,
	PAYLOAD_XMLFUNC,
} marker_payload;

typedef struct {
	const char	  *open;
	const char	  *close;
	marker_role	   role;
	marker_payload payload;
	const char	  *stop;
} marker_pair;

const marker_pair *marker_registry(size_t *n_pairs);

const marker_pair *marker_probe(const tokenizer *tok, marker_role role);

const marker_pair *marker_probe_text(const char *text, marker_role role);

const marker_pair *marker_find_open(const char *open);

#endif
