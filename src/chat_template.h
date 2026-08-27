#ifndef CHAT_TEMPLATE_H
#define CHAT_TEMPLATE_H

#include "common.h"
#include "gguf.h"
#include "jinja.h"
#include "tokenizer.h"

typedef struct {
	char *role;
	char *content;
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

	char *last_render;
} chat_template_state;

status_code chat_template_init(chat_template_state *cts, const gguf_ctx *g, const tokenizer *tok);
void		chat_template_free(chat_template_state *cts);
void		chat_template_clear_messages(chat_template_state *cts);

void chat_template_add_message(chat_template_state *cts, const char *role, const char *content);

status_code chat_template_add_turn(chat_template_state *cts, const char *role, const char *content,
								   int add_generation_prompt, char **out, char *errbuf,
								   size_t errbuf_len);

size_t chat_template_detect_static_prefix(chat_template_state *cts, const char *system);

status_code chat_template_preview_next_turn(chat_template_state *cts, const char *role,
											const char *content, int add_generation_prompt,
											char **out, char *errbuf, size_t errbuf_len);

#endif
