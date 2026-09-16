#ifndef CHAT_TEMPLATE_H
#define CHAT_TEMPLATE_H

#include "common.h"
#include "gguf.h"
#include "jinja.h"
#include "markers.h"
#include "tokenizer.h"

#include <json-c/json.h>

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
	bool		think_open;

	const marker_pair *tool_fmt;

	json_object *tools;
	char		*tool_choice;

	char *last_render;
} chat_template_state;

status_code chat_template_init(chat_template_state *cts, const gguf_ctx *g, const tokenizer *tok);
void		chat_template_free(chat_template_state *cts);
void		chat_template_clear_messages(chat_template_state *cts);

void chat_template_set_tools(chat_template_state *cts, json_object *tools, const char *tool_choice);

void chat_template_add_message(chat_template_state *cts, const char *role, const char *content);
void chat_template_add_message_ex(chat_template_state *cts, const chat_message *msg);

status_code chat_template_render(chat_template_state *cts, int add_generation_prompt, char **out,
								 char *errbuf, size_t errbuf_len);

status_code chat_template_add_turn(chat_template_state *cts, const char *role, const char *content,
								   int add_generation_prompt, char **out, char *errbuf,
								   size_t errbuf_len);

status_code chat_template_add_turn_ex(chat_template_state *cts, const chat_message *msg,
									  int add_generation_prompt, char **out, char *errbuf,
									  size_t errbuf_len);

void chat_template_rewrite_last_assistant(chat_template_state *cts, const char *content,
										  const char *reasoning, json_object *tool_calls);

size_t chat_template_detect_static_prefix(chat_template_state *cts, const char *system);

status_code chat_template_preview_next_turn(chat_template_state *cts, const char *role,
											const char *content, int add_generation_prompt,
											char **out, char *errbuf, size_t errbuf_len);

#endif
