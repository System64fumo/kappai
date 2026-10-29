#include "diffusion/text_encoder.h"
#include "compute.h"
#include "config.h"
#include "context.h"
#include "log.h"

#include <stdio.h>

#define PROMPT_MAX_CHARS 8192

struct text_encoder {
	context			  ctx;
	text_encoder_spec spec;
	char			 *llm_path, *device;
	int32_t			 *ids;
	int				  skip;
};

static status_code encode_one(context *ctx, const text_encoder_spec *spec, const char *text,
							  int skip, int32_t *ids, text_encoding *out) {
	char prompt[PROMPT_MAX_CHARS];
	int	 len = snprintf(prompt, sizeof(prompt), "%s%s%s%s", spec->system, spec->user_prefix, text,
						spec->user_suffix);
	if (len < 0 || (size_t)len >= sizeof(prompt)) {
		ERROR("prompt longer than %d characters", PROMPT_MAX_CHARS - 1);
		return ERR_INVALID_ARG;
	}
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

status_code text_encoder_open(const text_encoder_spec *spec, text_encoder **out) {
	text_encoder *enc  = xcalloc(1, sizeof(*enc));
	enc->llm_path	   = xstrdup(spec->llm_path);
	enc->device		   = xstrdup(spec->device);
	enc->spec		   = *spec;
	enc->spec.llm_path = enc->llm_path;
	enc->spec.device   = enc->device;

	config cfg		   = config_defaults();
	cfg.model		   = enc->llm_path;
	cfg.device		   = enc->device;
	cfg.shared_backend = spec->backend;
	cfg.ctx_size	   = spec->ctx_tokens;
	cfg.debug		   = log_get_level() <= LOG_DEBUG;
	config_init(&cfg);

	status_code s = context_init(&enc->ctx, &cfg);
	if (s != OK) {
		free(enc->llm_path);
		free(enc->device);
		free(enc);
		return s;
	}
	enc->ids  = xmalloc((size_t)spec->ctx_tokens * sizeof(int32_t));
	enc->skip = tokenizer_encode_with_specials(&enc->ctx.tok, spec->system, 0, enc->ids,
											   spec->ctx_tokens, NULL);
	*out	  = enc;
	return OK;
}

status_code text_encoder_run(text_encoder *enc, const char *const *texts, int n,
							 text_encoding *out) {
	status_code s = OK;
	for (int i = 0; i < n && s == OK; i++)
		s = encode_one(&enc->ctx, &enc->spec, texts[i], enc->skip, enc->ids, &out[i]);
	return s;
}

void text_encoder_close(text_encoder *enc) {
	if (!enc)
		return;
	context_free(&enc->ctx);
	free(enc->ids);
	free(enc->llm_path);
	free(enc->device);
	free(enc);
}

void text_encoding_free(text_encoding *e) {
	free(e->hidden);
	memset(e, 0, sizeof(*e));
}
