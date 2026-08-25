#ifndef OPENAI_H
#define OPENAI_H

#include "config.h"
#include "context.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct openai_state openai_state;

openai_state *openai_init(context *ctx, const cli_args *args);

bool openai_bind(openai_state *st, const char *host, int port);

bool openai_serve(openai_state *st, char *errbuf, size_t errbuf_len);

void openai_install_signals(void);

void openai_wait_for_signal(void);

void openai_stop(openai_state *st);

void openai_free(openai_state *st);

#endif
