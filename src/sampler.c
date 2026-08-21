#include "sampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void grow_buf(void **buf, int *cur_count, int need_count, size_t elem_size) {
	if (*cur_count >= need_count)
		return;
	*buf	   = xrealloc(*buf, (size_t)need_count * elem_size);
	*cur_count = need_count;
}

static inline uint64_t rotl64(uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static void rng_seed(rng *r, uint64_t seed) {
	if (seed == 0)
		seed = 0xDEADBEEFCAFEBABEULL;
	uint64_t z = seed;
	for (int i = 0; i < 4; i++) {
		z += 0x9E3779B97F4A7C15ULL;
		uint64_t t = z;
		t		   = (t ^ (t >> 30)) * 0xBF58476D1CE4E5B9ULL;
		t		   = (t ^ (t >> 27)) * 0x94D049BB133111EBULL;
		r->s[i]	   = t ^ (t >> 31);
	}
}

static float rng_uniform(rng *r) {
	uint64_t *s		 = r->s;
	uint64_t  result = rotl64(s[0] + s[3], 23) + s[0];
	uint64_t  t		 = s[1] << 17;
	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];
	s[2] ^= t;
	s[3] = rotl64(s[3], 45);
	return (result >> 40) * 0x1.0p-24f;
}

void sampler_init(sampler *s, uint64_t seed) {
	memset(s, 0, sizeof(*s));
	if (seed == 0) {
		FILE *f = fopen("/dev/urandom", "rb");
		if (f) {
			if (fread(&seed, sizeof(seed), 1, f) != 1)
				seed = 0;
			fclose(f);
		}
	}
	rng_seed(&s->rng, seed);
	s->temperature	   = 1.0f;
	s->top_p		   = 1.0f;
	s->recent_capacity = 256;
	s->recent		   = xmalloc(s->recent_capacity * sizeof(int32_t));
}

void sampler_free(sampler *s) {
	free(s->recent);
	free(s->logits_buf);
	free(s->cand_buf);
	memset(s, 0, sizeof(*s));
}

void sampler_set_vocab(sampler *s, int vocab_size) {
	grow_buf((void **)&s->logits_buf, &s->buf_vocab, vocab_size, sizeof(float));
}

void sampler_set_params(sampler *s, float temp, int top_k, float top_p, float min_p,
						float repeat_penalty, int repeat_last_n) {
	s->temperature	  = temp;
	s->top_k		  = top_k;
	s->top_p		  = top_p;
	s->min_p		  = min_p;
	s->repeat_penalty = repeat_penalty;
	if (repeat_last_n > 0) {
		s->repeat_last_n = repeat_last_n;
		if (repeat_last_n > s->recent_capacity) {
			int32_t *grown = xmalloc((size_t)repeat_last_n * sizeof(int32_t));
			for (int i = 0; i < s->recent_count; i++)
				grown[i] = s->recent[(s->recent_head + i) % s->recent_capacity];
			free(s->recent);
			s->recent		   = grown;
			s->recent_head	   = 0;
			s->recent_capacity = repeat_last_n;
		}
	}
}

void sampler_observe(sampler *s, int32_t token) {
	if (s->recent_count >= s->recent_capacity) {
		s->recent[s->recent_head] = token;
		s->recent_head			  = (s->recent_head + 1) % s->recent_capacity;
	} else {
		s->recent[(s->recent_head + s->recent_count) % s->recent_capacity] = token;
		s->recent_count++;
	}
}

int32_t sampler_argmax(const float *logits, int vocab) {
	return cpu_argmax_f32(logits, vocab);
}

static int top_k_heap(const float *logits, int vocab, int k, sampler_top_k_entry *out) {
	static float *hs;
	static int	 *hi;
	static int	  cap;

	if (k > vocab)
		k = vocab;
	if (k <= 0)
		return 0;
	if (cap < k) {
		hs	= xrealloc(hs, (size_t)k * sizeof(float));
		hi	= xrealloc(hi, (size_t)k * sizeof(int));
		cap = k;
	}
	int hn = topk_heap_select(logits, vocab, k, hs, hi);
	for (int i = 0; i < hn; i++) {
		out[i].v = hs[i];
		out[i].i = hi[i];
	}
	return hn;
}

static int cmp_desc(const void *a, const void *b) {
	float va = ((const sampler_top_k_entry *)a)->v;
	float vb = ((const sampler_top_k_entry *)b)->v;
	return va < vb ? 1 : va > vb ? -1 : 0;
}

int sampler_top_k(const float *logits, int vocab, int k, sampler_top_k_entry *out) {
	int kept = top_k_heap(logits, vocab, k, out);
	qsort(out, kept, sizeof(sampler_top_k_entry), cmp_desc);
	return kept;
}

static int top_all_desc(const float *logits, int vocab, sampler_top_k_entry *out, int max_keep) {
	if (max_keep > vocab)
		max_keep = vocab;
	if (max_keep <= 0)
		return 0;

	int kept = top_k_heap(logits, vocab, max_keep, out);
	qsort(out, kept, sizeof(sampler_top_k_entry), cmp_desc);
	return kept;
}

static void apply_repeat_penalty(sampler *s, float *mut, int vocab) {
	int n = MIN(s->repeat_last_n, s->recent_count);
	for (int i = s->recent_count - n; i < s->recent_count; i++) {
		int32_t tok = s->recent[(s->recent_head + i) % s->recent_capacity];
		if (tok < 0 || tok >= vocab)
			continue;
		if (mut[tok] > 0)
			mut[tok] /= s->repeat_penalty;
		else
			mut[tok] *= s->repeat_penalty;
	}
}

static int collect_candidates(sampler *s, const float *logits, int vocab,
							  sampler_top_k_entry *arr) {
	int kept;
	if (s->top_k > 0 && s->top_k < vocab) {
		kept = top_k_heap(logits, vocab, s->top_k, arr);
		qsort(arr, kept, sizeof(sampler_top_k_entry), cmp_desc);
	} else {
		int cap;
		if (s->top_p < 1.0f || s->min_p > 0.0f)
			cap = vocab < 1024 ? vocab : 1024;
		else
			cap = vocab;
		kept = top_all_desc(logits, vocab, arr, cap);
	}
	return kept;
}

static void apply_temperature_softmax(sampler_top_k_entry *arr, int kept, float temperature) {
	float inv_temp = 1.0f / temperature;
	float mx	   = arr[0].v * inv_temp;
	for (int i = 0; i < kept; i++)
		arr[i].v = expf((arr[i].v * inv_temp) - mx);
}

static int apply_top_p(sampler_top_k_entry *arr, int kept, float top_p, float sum) {
	if (top_p >= 1.0f)
		return kept;
	float cumulative = 0.0f;
	for (int i = 0; i < kept; i++) {
		cumulative += arr[i].v / sum;
		if (cumulative > top_p)
			return i + 1;
	}
	return kept;
}

static int apply_min_p(sampler_top_k_entry *arr, int kept, float min_p, float sum) {
	if (min_p <= 0.0f)
		return kept;
	float min_prob = min_p * (arr[0].v / sum);
	int	  n		   = 0;
	while (n < kept && (arr[n].v / sum) >= min_prob)
		n++;
	return n > 0 ? n : 1;
}

static int32_t sample_from_candidates(sampler_top_k_entry *arr, int kept, rng *r) {
	if (kept <= 0)
		kept = 1;
	float psum = 0.0f;
	for (int i = 0; i < kept; i++)
		psum += arr[i].v;
	float rnd	   = rng_uniform(r) * psum;
	float acc	   = 0.0f;
	int	  picked_i = kept - 1;
	for (int i = 0; i < kept; i++) {
		acc += arr[i].v;
		if (rnd < acc) {
			picked_i = i;
			break;
		}
	}
	return arr[picked_i].i;
}

int32_t sampler_sample(sampler *s, const float *logits_in, int vocab) {
	const float *logits	   = logits_in;
	int			 penalized = s->repeat_penalty != 1.0f && s->recent_count > 0;

	if (penalized) {
		float *mut = s->logits_buf;
		memcpy(mut, logits_in, (size_t)vocab * sizeof(float));
		logits = mut;
		apply_repeat_penalty(s, mut, vocab);
	}

	if (s->temperature <= 0.0f || s->top_k == 1)
		return sampler_argmax(logits, vocab);

	int need_cands = (s->top_k > 0 && s->top_k < vocab) ? s->top_k : vocab;
	grow_buf(&s->cand_buf, &s->cand_vocab, need_cands, sizeof(sampler_top_k_entry));
	sampler_top_k_entry *arr  = s->cand_buf;
	int					 kept = collect_candidates(s, logits, vocab, arr);
	if (kept <= 0)
		return -1;

	apply_temperature_softmax(arr, kept, s->temperature);

	float sum = 0.0f;
	for (int i = 0; i < kept; i++)
		sum += arr[i].v;

	kept = apply_top_p(arr, kept, s->top_p, sum);
	kept = apply_min_p(arr, kept, s->min_p, sum);
	if (kept <= 0)
		kept = 1;

	return sample_from_candidates(arr, kept, &s->rng);
}