#include "diffusion/text_encoder.h"
#include "compute.h"
#include "config.h"
#include "context.h"
#include "log.h"

#include <stdio.h>

#define PROMPT_MAX_CHARS 8192

static status_code encode_one(context *ctx, const text_encoder_spec *spec, const char *text,
							  int skip, int32_t *ids, text_encoding *out) {
	char prompt[PROMPT_MAX_CHARS];
	snprintf(prompt, sizeof(prompt), "%s%s%s%s", spec->system, spec->user_prefix, text,
			 spec->user_suffix);
	int total = tokenizer_encode_with_specials(&ctx->tok, prompt, 0, ids, spec->ctx_tokens, NULL);
	if (total <= skip || total >= spec->ctx_tokens) {
		ERROR("prompt does not fit in %d tokens", spec->ctx_tokens);
		return ERR_INVALID_ARG;
	}
	context_reset(ctx);
	size_t		dim = (size_t)ctx->m.dim;
	float	   *all = xmalloc((size_t)total * dim * sizeof(float));
	status_code s =
		compute_hidden_states(&ctx->m, &ctx->kv, &ctx->scratch, ids, total, ctx->flash_attn, all);
	out->rows	= total - skip;
	out->hidden = xmalloc((size_t)out->rows * dim * sizeof(float));
	memcpy(out->hidden, all + (size_t)skip * dim, (size_t)out->rows * dim * sizeof(float));
	free(all);
	return s;
}

status_code text_encoder_run(const text_encoder_spec *spec, const char *const *texts, int n,
							 text_encoding *out) {
	config cfg		   = config_defaults();
	cfg.model		   = spec->llm_path;
	cfg.device		   = spec->device;
	cfg.shared_backend = spec->backend;
	cfg.ctx_size	   = spec->ctx_tokens;
	cfg.debug		   = log_get_level() <= LOG_DEBUG;
	config_init(&cfg);

	context		ctx;
	status_code s = context_init(&ctx, &cfg);
	if (s != OK)
		return s;

	int32_t *ids = xmalloc((size_t)spec->ctx_tokens * sizeof(int32_t));
	int		 skip =
		tokenizer_encode_with_specials(&ctx.tok, spec->system, 0, ids, spec->ctx_tokens, NULL);
	for (int i = 0; i < n && s == OK; i++)
		s = encode_one(&ctx, spec, texts[i], skip, ids, &out[i]);
	free(ids);
	context_free(&ctx);
	return s;
}

void text_encoding_free(text_encoding *e) {
	free(e->hidden);
	memset(e, 0, sizeof(*e));
}
