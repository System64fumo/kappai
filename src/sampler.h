#ifndef SAMPLER_H
#define SAMPLER_H

#include "backend/backend.h"
#include "common.h"

typedef struct {
	uint64_t s[4];
} rng;

typedef struct {
	float	 temperature;
	int		 top_k;
	float	 top_p;
	float	 min_p;
	float	 repeat_penalty;
	int		 repeat_last_n;
	rng		 rng;
	int32_t *recent;
	int		 recent_count;
	int		 recent_capacity;
	int		 recent_head;
	float	*logits_buf;
	void	*cand_buf;
	int		 buf_vocab;
	int		 cand_vocab;
} sampler;

void	sampler_init(sampler *s, uint64_t seed);
void	sampler_free(sampler *s);
void	sampler_set_params(sampler *s, float temp, int top_k, float top_p, float min_p,
						   float repeat_penalty, int repeat_last_n);
void	sampler_set_vocab(sampler *s, int vocab_size);
void	sampler_observe(sampler *s, int32_t token);
int32_t sampler_sample(sampler *s, const float *logits, int vocab);
int32_t sampler_argmax(const float *logits, int vocab);

typedef struct {
	float v;
	int	  i;
} sampler_top_k_entry;

int sampler_top_k(const float *logits, int vocab, int k, sampler_top_k_entry *out);

#endif