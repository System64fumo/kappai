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

static const marker_pair *probe(const tokenizer *tok, const char *text, marker_role role) {
	size_t			   n   = 0;
	const marker_pair *all = marker_registry(&n);
	for (size_t i = 0; i < n; i++) {
		if (all[i].role != role)
			continue;
		if (!all[i].open[0])
			continue;
		if (all[i].probe_hint) {
			if (tok) {
				if (tokenizer_find_token(tok, all[i].probe_hint) < 0)
					continue;
			} else if (!text || !strstr(text, all[i].probe_hint)) {
				continue;
			}
		}
		if (tok) {
			if (tokenizer_find_token(tok, all[i].open) < 0)
				continue;
			if (all[i].close && tokenizer_find_token(tok, all[i].close) < 0)
				continue;
			return &all[i];
		}
		if (text && strstr(text, all[i].open))
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

const marker_pair *marker_probe(const tokenizer *tok, marker_role role) {
	return probe(tok, NULL, role);
}

const marker_pair *marker_probe_text(const char *text, marker_role role) {
	return probe(NULL, text, role);
}