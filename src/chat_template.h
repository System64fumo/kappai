#ifndef CHAT_TEMPLATE_H
#define CHAT_TEMPLATE_H

#include "common.h"
#include "gguf.h"
#include "jinja.h"
#include "markers.h"
#include "tokenizer.h"

#include <json-c/json_types.h>

typedef struct {
	char		*role;
	char		*content;
	char		*reasoning_content;
	json_object *tool_calls;
	char		*tool_call_id;
	char		*name;
} chat_message;

typedef struct {
	jinja_program *prog;
	char		  *bos_token;
	char		  *eos_token;

	chat_message *messages;
	size_t		  n_messages;
	size_t		  cap_messages;

	bool enable_thinking;

	int32_t		think_start_id;
	int32_t		think_end_id;
	const char *think_start_text;
	const char *think_end_text;
	bool		think_label_line;
	bool		think_open;

	bool keep_thinking_in_history;

	void (*invalidate_cb)(void *);
	void *invalidate_ud;

	const marker_pair *tool_fmt;

	json_object *tools;
	char		*tool_choice;

	char *last_render;
} chat_template_state;

void chat_template_set_invalidate_cb(chat_template_state *cts, void (*cb)(void *), void *ud);

status_code chat_template_init(chat_template_state *cts, const gguf_ctx *g, const tokenizer *tok);
void		chat_template_free(chat_template_state *cts);
void		chat_template_clear_messages(chat_template_state *cts);

void chat_template_set_tools(chat_template_state *cts, json_object *tools, const char *tool_choice);

void chat_template_add_message(chat_template_state *cts, const char *role, const char *content);
void chat_template_add_message_ex(chat_template_state *cts, const chat_message *msg);

status_code chat_template_add_turn(chat_template_state *cts, const char *role, const char *content,
								   int add_generation_prompt, char **out, char *errbuf,
								   size_t errbuf_len);

status_code chat_template_add_turn_ex(chat_template_state *cts, const chat_message *msg,
									  int add_generation_prompt, char **out, char *errbuf,
									  size_t errbuf_len);

void chat_template_rewrite_last_assistant(chat_template_state *cts, const char *content,
										  const char *reasoning, json_object *tool_calls);

void chat_template_split_thinking(const chat_template_state *cts, const char *raw,
								  char **out_reasoning, char **out_content);

typedef enum {
	THINK_EMIT,
	THINK_START,
	THINK_END,
	THINK_SWALLOW,
} think_filter_event;

typedef struct {
	bool in_thinking;
	bool skip_label;
	bool first_token;
} think_filter;

static inline think_filter_event think_filter_feed(think_filter *f, int32_t id, int32_t start_id,
												   int32_t end_id, int think_open, bool label_line,
												   const char **piece, int *n) {
	if (f->first_token && think_open)
		f->in_thinking = true;
	f->first_token = false;
	if (id == start_id) {
		f->in_thinking = true;
		f->skip_label  = label_line;
		return THINK_START;
	}
	if (id == end_id) {
		f->in_thinking = false;
		return THINK_END;
	}
	if (f->skip_label) {
		const char *nl = memchr(*piece, '\n', (size_t)*n);
		if (!nl)
			return THINK_SWALLOW;
		*n			  = *n - (int)(nl - *piece) - 1;
		*piece		  = nl + 1;
		f->skip_label = false;
	}
	return THINK_EMIT;
}

size_t chat_template_detect_static_prefix(chat_template_state *cts, const char *system);

status_code chat_template_preview_next_turn(chat_template_state *cts, const char *role,
											const char *content, int add_generation_prompt,
											char **out, char *errbuf, size_t errbuf_len);

#endif
