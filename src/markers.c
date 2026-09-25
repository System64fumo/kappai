#include "markers.h"

#include <string.h>

static const marker_pair marker_pairs[] = {
	{"<think>", "</think>", MARKER_THINKING, PAYLOAD_NONE, NULL, NULL},
	{"<|channel>", "<channel|>", MARKER_THINKING, PAYLOAD_NONE, NULL, NULL},
	{"<|think|>", "<|/think|>", MARKER_THINKING, PAYLOAD_NONE, NULL, NULL},
	{"<start_of_thought>", "<|end_of_thought|>", MARKER_THINKING, PAYLOAD_NONE, NULL, NULL},

	{"<|tool_call>", "<tool_call|>", MARKER_TOOL_CALL, PAYLOAD_CALLCOLON, "<|tool_response>", NULL},
	{"<tool_call>", "</tool_call>", MARKER_TOOL_CALL, PAYLOAD_XMLARGS, NULL, "<arg_key>"},
	{"<tool_call>", "</tool_call>", MARKER_TOOL_CALL, PAYLOAD_XMLFUNC, NULL, NULL},
	{"<|tool_call_start|>", "<|tool_call_end|>", MARKER_TOOL_CALL, PAYLOAD_FUNCARGS,
	 "<|tool_call_end|>", NULL},
	{"<function=", "</function>", MARKER_TOOL_CALL, PAYLOAD_XMLFUNC, "<|im_end|>", NULL},
	{"", "", MARKER_TOOL_CALL, PAYLOAD_AUTO, "<|eot_id|>", NULL},
};

const marker_pair *marker_registry(size_t *n_pairs) {
	if (n_pairs)
		*n_pairs = sizeof(marker_pairs) / sizeof(marker_pairs[0]);
	return marker_pairs;
}

typedef int (*marker_accept_fn)(const marker_pair *mp, const void *mode_ctx);

static const marker_pair *probe_walk(marker_role role, marker_accept_fn accept,
									 const void *mode_ctx) {
	size_t			   n   = 0;
	const marker_pair *all = marker_registry(&n);
	for (size_t i = 0; i < n; i++) {
		if (all[i].role != role)
			continue;
		if (!all[i].open[0])
			continue;
		if (accept(&all[i], mode_ctx))
			return &all[i];
	}
	for (size_t i = 0; i < n; i++) {
		if (all[i].role != role)
			continue;
		if (all[i].open[0])
			continue;
		return &all[i];
	}
	return NULL;
}

static int probe_accept_token(const marker_pair *mp, const void *mode_ctx) {
	const tokenizer *tok = (const tokenizer *)mode_ctx;
	if (mp->probe_hint && tokenizer_find_token(tok, mp->probe_hint) < 0)
		return 0;
	if (tokenizer_find_token(tok, mp->open) < 0)
		return 0;
	if (mp->close && tokenizer_find_token(tok, mp->close) < 0)
		return 0;
	return 1;
}

static int probe_accept_text(const marker_pair *mp, const void *mode_ctx) {
	const char *text = (const char *)mode_ctx;
	if (mp->probe_hint && (!text || !strstr(text, mp->probe_hint)))
		return 0;
	return text && strstr(text, mp->open) != NULL;
}

const marker_pair *marker_probe(const tokenizer *tok, marker_role role) {
	return probe_walk(role, probe_accept_token, tok);
}

const marker_pair *marker_probe_text(const char *text, marker_role role) {
	return probe_walk(role, probe_accept_text, text);
}