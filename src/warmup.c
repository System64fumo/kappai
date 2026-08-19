#include "warmup.h"

#include "chat_template.h"
#include "log.h"

status_code warmup_run(context *c, const char *system) {
	if (!c)
		return ERR_INVALID_ARG;

	const char *sys = (system && *system) ? system : "";

	size_t prefix_bytes = chat_template_detect_static_prefix(&c->chat, sys);
	if (prefix_bytes == 0)
		return OK;

	char	   *render = NULL;
	char		errbuf[512];
	status_code rc =
		chat_template_add_turn(&c->chat, "system", sys, 0, &render, errbuf, sizeof(errbuf));
	if (rc != OK)
		return OK;

	int		 cap = c->n_ctx;
	int32_t *ids = xmalloc((size_t)(cap + 1) * sizeof(int32_t));
	int		 n	 = tokenizer_encode_with_specials(&c->tok, render, 0, ids, cap, &c->scratch.prof);
	if (n < 0) {
		free(ids);
		free(render);
		return OK;
	}

	int count = tokenizer_token_count_for_bytes(&c->tok, ids, n, prefix_bytes);
	if (count == 0) {
		free(ids);
		free(render);
		return OK;
	}

	context_monitor_send_start(c);
	prefill_result pf = context_prefill_tokens(c, ids, count, "warmup", true);
	if (pf.rc < 0) {
		free(ids);
		free(render);
		return ERR_INTERNAL;
	}

	char *decoded = tokenizer_decode_prefix(&c->tok, ids, count);
	if (decoded) {
		free(c->chat.last_render);
		c->chat.last_render = decoded;
	}
	c->warmup_done = true;

	DEBUG("warmup: prefilled %d tokens", count);

	free(ids);
	free(render);
	return OK;
}
