#ifndef CONTEXT_H
#define CONTEXT_H

#include "chat_template.h"
#include "common.h"
#include "compute.h"
#include "config.h"
#include "kvcache.h"
#include "model.h"
#include "monitor.h"
#include "sampler.h"
#include "tokenizer.h"
#include <signal.h>

#define CTX_INTERRUPTED (-2)
#define CTX_COMPUTE_ERROR (-3)

typedef struct {
	model				  m;
	tokenizer			  tok;
	kvcache				  kv;
	compute_scratch		  scratch;
	sampler				  samp;
	monitor				  monitor;
	int					  n_ctx;
	bool				  flash_attn;
	bool				  debug_forward;
	volatile sig_atomic_t interrupt;
	bool				  context_limit_hit;
	bool				  show_template;
	backend				 *backend;
	monitor_layer_tracker layer_tracker;
	chat_template_state	  chat;
} context;

typedef struct {
	int		 rc;
	uint64_t us;
	double	 tps;
} prefill_result;

typedef struct {
	float temperature;
	int	  top_k;
	float top_p;
	float min_p;
} sampler_params;

void context_set_show_template(context *c, bool enabled);

status_code context_init(context *c, const config *cfg);
void		context_free(context *c);
void		context_reset(context *c);

void context_monitor_send_start(context *c);

int context_prefill_preamble(context *c, const char *system);

prefill_result context_prefill_tokens(context *c, const int32_t *tokens, int n_tokens,
									  const char *phase);

int context_chat_turn(context *c, const char *role, const char *content, bool add_generation_prompt,
					  int max_tokens, const sampler_params						 *samp,
					  void (*on_token)(int32_t, const char *, int, void *), void *ud);

#endif