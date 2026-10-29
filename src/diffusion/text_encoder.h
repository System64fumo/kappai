#ifndef DIFFUSION_TEXT_ENCODER_H
#define DIFFUSION_TEXT_ENCODER_H

#include "common.h"

typedef struct {
	const char	   *llm_path, *device;
	struct backend *backend;
	const char	   *system, *user_prefix, *user_suffix;
	int				ctx_tokens;
} text_encoder_spec;

typedef struct {
	float *hidden;
	int	   rows;
} text_encoding;

typedef struct text_encoder text_encoder;

status_code text_encoder_open(const text_encoder_spec *spec, text_encoder **out);
status_code text_encoder_run(text_encoder *enc, const char *const *texts, int n,
							 text_encoding *out);
void		text_encoder_close(text_encoder *enc);
void		text_encoding_free(text_encoding *e);

#endif
