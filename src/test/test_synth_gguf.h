#ifndef TEST_SYNTH_GGUF_H
#define TEST_SYNTH_GGUF_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define TSG_MAGIC 0x46554747u
#define TSG_VERSION 3u
#define TSG_T_U32 4u
#define TSG_T_I32 5u
#define TSG_T_F32 6u
#define TSG_T_BOOL 7u
#define TSG_T_STRING 8u
#define TSG_T_ARRAY 9u

typedef struct {
	uint8_t *buf;
	size_t	 len;
	size_t	 cap;
} tsg_buf;

static __attribute__((unused)) void tsg_put(tsg_buf *w, const void *p, size_t n) {
	if (w->len + n > w->cap) {
		size_t cap = w->cap ? w->cap * 2 : 4096;
		while (cap < w->len + n)
			cap *= 2;
		w->buf = realloc(w->buf, cap);
		w->cap = cap;
	}
	memcpy(w->buf + w->len, p, n);
	w->len += n;
}
static __attribute__((unused)) void tsg_u32(tsg_buf *w, uint32_t v) {
	tsg_put(w, &v, 4);
}
static __attribute__((unused)) void tsg_i32(tsg_buf *w, int32_t v) {
	tsg_put(w, &v, 4);
}
static __attribute__((unused)) void tsg_u64(tsg_buf *w, uint64_t v) {
	tsg_put(w, &v, 8);
}
static __attribute__((unused)) void tsg_f32v(tsg_buf *w, float v) {
	tsg_put(w, &v, 4);
}
static __attribute__((unused)) void tsg_str(tsg_buf *w, const char *s) {
	size_t n = strlen(s);
	tsg_u64(w, n);
	tsg_put(w, s, n);
}

static __attribute__((unused)) uint64_t tsg_rng_state = 0x853C49E6748FEA9BULL;

static __attribute__((unused)) void tsg_seed(uint64_t s) {
	tsg_rng_state = s ? s : 0x853C49E6748FEA9BULL;
}

static __attribute__((unused)) uint32_t tsg_rand(void) {
	uint64_t x = tsg_rng_state;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	tsg_rng_state = x;
	return (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
}

static __attribute__((unused)) float tsg_uniform(float lo, float hi) {
	return lo + (hi - lo) * ((float)(tsg_rand() % 1000001) / 1000001.0f);
}

static __attribute__((unused)) void tsg_fill_rand_f32(float *p, size_t n, float scale) {
	for (size_t i = 0; i < n; i++)
		p[i] = tsg_uniform(-scale, scale);
}

static __attribute__((unused)) void tsg_fill_norm_f32(float *p, size_t n) {
	for (size_t i = 0; i < n; i++)
		p[i] = 1.0f + tsg_uniform(-0.15f, 0.15f);
}

#define TSG_MAX_TENSORS 128
#define TSG_ALIGN 32u

typedef struct {
	char	 name[120];
	uint32_t ndims;
	uint64_t dims[4];
	float	*data;
	size_t	 nelem;
	size_t	 off_field_pos;
} tsg_tensor;

typedef struct {
	tsg_buf	   body;
	tsg_tensor tensors[TSG_MAX_TENSORS];
	int		   n_tensors;
	int		   n_kv;
} tsg_writer;

static __attribute__((unused)) void tsg_winit(tsg_writer *w) {
	memset(w, 0, sizeof(*w));
	tsg_u32(&w->body, 0);
	tsg_u32(&w->body, 0);
	tsg_u64(&w->body, 0);
	tsg_u64(&w->body, 0);
}

static __attribute__((unused)) void tsg_kv_begin(tsg_writer *w, const char *key, uint32_t type) {
	tsg_str(&w->body, key);
	tsg_u32(&w->body, type);
	w->n_kv++;
}
static __attribute__((unused)) void tsg_kv_str(tsg_writer *w, const char *key, const char *val) {
	tsg_kv_begin(w, key, TSG_T_STRING);
	tsg_str(&w->body, val);
}
static __attribute__((unused)) void tsg_kv_i32(tsg_writer *w, const char *key, int32_t val) {
	tsg_kv_begin(w, key, TSG_T_I32);
	tsg_i32(&w->body, val);
}
static __attribute__((unused)) void tsg_kv_u32(tsg_writer *w, const char *key, uint32_t val) {
	tsg_kv_begin(w, key, TSG_T_U32);
	tsg_u32(&w->body, val);
}
static __attribute__((unused)) void tsg_kv_f32(tsg_writer *w, const char *key, float val) {
	tsg_kv_begin(w, key, TSG_T_F32);
	tsg_f32v(&w->body, val);
}
static __attribute__((unused)) void tsg_kv_bool(tsg_writer *w, const char *key, int val) {
	tsg_kv_begin(w, key, TSG_T_BOOL);
	unsigned char b = (unsigned char)(val ? 1 : 0);
	tsg_put(&w->body, &b, 1);
}
static __attribute__((unused)) void tsg_kv_arr_i32(tsg_writer *w, const char *key, const int32_t *v,
												   size_t n) {
	tsg_kv_begin(w, key, TSG_T_ARRAY);
	tsg_u32(&w->body, TSG_T_I32);
	tsg_u64(&w->body, n);
	for (size_t i = 0; i < n; i++)
		tsg_i32(&w->body, v[i]);
}
static __attribute__((unused)) void tsg_kv_arr_f32(tsg_writer *w, const char *key, const float *v,
												   size_t n) {
	tsg_kv_begin(w, key, TSG_T_ARRAY);
	tsg_u32(&w->body, TSG_T_F32);
	tsg_u64(&w->body, n);
	for (size_t i = 0; i < n; i++)
		tsg_f32v(&w->body, v[i]);
}
static __attribute__((unused)) void tsg_kv_arr_str(tsg_writer *w, const char *key,
												   const char *const *v, size_t n) {
	tsg_kv_begin(w, key, TSG_T_ARRAY);
	tsg_u32(&w->body, TSG_T_STRING);
	tsg_u64(&w->body, n);
	for (size_t i = 0; i < n; i++)
		tsg_str(&w->body, v[i]);
}

static __attribute__((unused)) float *
tsg_tensor_new(tsg_writer *w, const char *name, uint32_t ndims, const uint64_t *dims, float scale) {
	if (w->n_tensors >= TSG_MAX_TENSORS)
		abort();
	tsg_tensor *t = &w->tensors[w->n_tensors++];
	snprintf(t->name, sizeof(t->name), "%s", name);
	t->ndims = ndims;
	t->nelem = 1;
	for (uint32_t d = 0; d < ndims; d++) {
		t->dims[d] = dims[d];
		t->nelem *= (size_t)dims[d];
	}
	t->data = malloc(t->nelem * sizeof(float));
	if (!t->data)
		abort();
	if (scale < 0.0f)
		tsg_fill_norm_f32(t->data, t->nelem);
	else
		tsg_fill_rand_f32(t->data, t->nelem, scale);
	return t->data;
}

static __attribute__((unused)) float *tsg_add2(tsg_writer *w, const char *name, uint64_t d0,
											   uint64_t d1, float scale) {
	uint64_t dims[2] = {d0, d1};
	return tsg_tensor_new(w, name, 2, dims, scale);
}

static __attribute__((unused)) float *tsg_add1(tsg_writer *w, const char *name, uint64_t d0,
											   float scale) {
	return tsg_tensor_new(w, name, 1, &d0, scale);
}

static __attribute__((unused)) float *tsg_add3(tsg_writer *w, const char *name, uint64_t d0,
											   uint64_t d1, uint64_t d2, float scale) {
	uint64_t dims[3] = {d0, d1, d2};
	return tsg_tensor_new(w, name, 3, dims, scale);
}

static __attribute__((unused)) void tsg_tensor_free_all(tsg_writer *w) {
	for (int i = 0; i < w->n_tensors; i++)
		free(w->tensors[i].data);
	w->n_tensors = 0;
	free(w->body.buf);
	w->body.buf = NULL;
	w->body.len = w->body.cap = 0;
}

static __attribute__((unused)) void tsg_ensure_parent_dirs(const char *path) {
	char   dirs[512];
	size_t len = strlen(path);
	if (len == 0 || len >= sizeof(dirs))
		return;
	memcpy(dirs, path, len + 1);
	char *last_slash = strrchr(dirs, '/');
	if (!last_slash)
		return;
	*last_slash = '\0';
	for (char *p = dirs + 1;; p++) {
		if (*p == '/' || *p == '\0') {
			char next = *p;
			*p		  = '\0';
			mkdir(dirs, 0755);
			if (!next)
				break;
			*p = '/';
		}
	}
}

static __attribute__((noreturn, unused)) void tsg_write_failed(const char *path) {
	fprintf(stderr, "error: cannot write gguf fixture '%s'\n", path);
	abort();
}

static __attribute__((unused)) int tsg_finish(tsg_writer *w, const char *path) {
	tsg_ensure_parent_dirs(path);

	uint32_t magic = TSG_MAGIC, version = TSG_VERSION;
	uint64_t ntens = (uint64_t)w->n_tensors, nkv = (uint64_t)w->n_kv;
	memcpy(w->body.buf + 0, &magic, 4);
	memcpy(w->body.buf + 4, &version, 4);
	memcpy(w->body.buf + 8, &ntens, 8);
	memcpy(w->body.buf + 16, &nkv, 8);

	for (int i = 0; i < w->n_tensors; i++) {
		tsg_str(&w->body, w->tensors[i].name);
		tsg_u32(&w->body, w->tensors[i].ndims);
		for (uint32_t d = 0; d < w->tensors[i].ndims; d++)
			tsg_u64(&w->body, w->tensors[i].dims[d]);
		tsg_u32(&w->body, 0u);
		w->tensors[i].off_field_pos = w->body.len;
		tsg_u64(&w->body, 0);
	}

	while (w->body.len % TSG_ALIGN)
		tsg_put(&w->body, "\0", 1);
	size_t data_start = w->body.len;
	for (int i = 0; i < w->n_tensors; i++) {
		while (w->body.len % TSG_ALIGN)
			tsg_put(&w->body, "\0", 1);
		uint64_t off = (uint64_t)(w->body.len - data_start);
		tsg_put(&w->body, w->tensors[i].data, w->tensors[i].nelem * sizeof(float));
		memcpy(w->body.buf + w->tensors[i].off_field_pos, &off, 8);
	}

	FILE *fp = fopen(path, "wb");
	if (!fp)
		return -1;
	size_t wr = fwrite(w->body.buf, 1, w->body.len, fp);
	fclose(fp);
	return wr == w->body.len ? 0 : -1;
}

#define TSG_TOK_NORMAL 1
#define TSG_TOK_UNKNOWN 2
#define TSG_TOK_CONTROL 3
#define TSG_TOK_BYTE 6

static __attribute__((unused)) void tsg_build_vocab_file(const char *path, const char *model_name,
														 const char *pre, int has_space_flag,
														 int space_flag, const char *const *tokens,
														 const int32_t *types, size_t n) {
	tsg_writer w;
	tsg_winit(&w);
	tsg_kv_str(&w, "tokenizer.ggml.model", model_name);
	tsg_kv_arr_str(&w, "tokenizer.ggml.tokens", tokens, n);
	tsg_kv_arr_i32(&w, "tokenizer.ggml.token_type", types, n);
	if (pre)
		tsg_kv_str(&w, "tokenizer.ggml.pre", pre);
	if (has_space_flag)
		tsg_kv_bool(&w, "tokenizer.ggml.add_space_prefix", space_flag);
	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}

#define TSG_CHAT_VOCAB 357

static __attribute__((unused)) int32_t tsg_chat_char_id(int c) {
	if (c >= 32 && c <= 126)
		return 1 + (c - 32);
	if (c == '\n')
		return 96;
	if (c == '\t')
		return 97;
	if (c == '\r')
		return 98;
	return -1;
}

static __attribute__((unused)) void tsg_build_chat_vocab_arrays(char (*storage)[8],
																const char **toks, int32_t *types) {
	int id = 0;
	snprintf(storage[id], 8, "<unk>");
	toks[id]	= storage[id];
	types[id++] = TSG_TOK_UNKNOWN;
	for (int c = 32; c <= 126; c++) {
		storage[id][0] = (char)c;
		storage[id][1] = '\0';
		toks[id]	   = storage[id];
		types[id++]	   = TSG_TOK_NORMAL;
	}
	const char specials3[] = {'\n', '\t', '\r'};
	for (int i = 0; i < 3; i++) {
		storage[id][0] = specials3[i];
		storage[id][1] = '\0';
		toks[id]	   = storage[id];
		types[id++]	   = TSG_TOK_NORMAL;
	}
	for (int b = 0; b < 256; b++) {
		snprintf(storage[id], 8, "<0x%02X>", b);
		toks[id]	= storage[id];
		types[id++] = TSG_TOK_BYTE;
	}
	snprintf(storage[id], 8, "<bos>");
	toks[id]	= storage[id];
	types[id++] = TSG_TOK_CONTROL;
	snprintf(storage[id], 8, "<eos>");
	toks[id]	= storage[id];
	types[id++] = TSG_TOK_CONTROL;
}

typedef struct {
	int		 dim;
	int		 n_heads;
	int		 n_kv_heads;
	int		 head_dim;
	int		 n_layers;
	int		 intermediate;
	int		 ctx;
	uint64_t seed;
} tsg_llama_spec;

static __attribute__((unused)) const char *tsg_chat_template =
	"{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n"
	"{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}";

static __attribute__((unused)) void tsg_build_chat_llama(const char			  *path,
														 const tsg_llama_spec *s) {
	static char		   storage[TSG_CHAT_VOCAB][8];
	static const char *toks[TSG_CHAT_VOCAB];
	static int32_t	   types[TSG_CHAT_VOCAB];
	tsg_build_chat_vocab_arrays(storage, toks, types);

	tsg_writer w;
	tsg_winit(&w);
	tsg_seed(s->seed);

	tsg_kv_str(&w, "general.architecture", "llama");
	tsg_kv_i32(&w, "llama.context_length", s->ctx);
	tsg_kv_i32(&w, "llama.block_count", s->n_layers);
	tsg_kv_i32(&w, "llama.embedding_length", s->dim);
	tsg_kv_i32(&w, "llama.feed_forward_length", s->intermediate);
	tsg_kv_i32(&w, "llama.attention.head_count", s->n_heads);
	tsg_kv_i32(&w, "llama.attention.head_count_kv", s->n_kv_heads);
	tsg_kv_i32(&w, "llama.attention.key_length", s->head_dim);
	tsg_kv_f32(&w, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
	tsg_kv_arr_str(&w, "tokenizer.ggml.tokens", toks, TSG_CHAT_VOCAB);
	tsg_kv_arr_i32(&w, "tokenizer.ggml.token_type", types, TSG_CHAT_VOCAB);
	tsg_kv_str(&w, "tokenizer.ggml.model", "gpt2");
	tsg_kv_i32(&w, "tokenizer.ggml.unknown_token_id", 0);
	tsg_kv_str(&w, "tokenizer.chat_template", tsg_chat_template);

	int	 V = TSG_CHAT_VOCAB;
	char nm[128];
	tsg_add2(&w, "token_embd.weight", (uint64_t)s->dim, (uint64_t)V, 0.10f);
	tsg_add1(&w, "output_norm.weight", (uint64_t)s->dim, -1.0f);
	tsg_add2(&w, "output.weight", (uint64_t)s->dim, (uint64_t)V, 0.10f);
	int q_out  = s->n_heads * s->head_dim;
	int kv_out = s->n_kv_heads * s->head_dim;
	for (int i = 0; i < s->n_layers; i++) {
		snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", i);
		tsg_add1(&w, nm, (uint64_t)s->dim, -1.0f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)q_out, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)kv_out, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)kv_out, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", i);
		tsg_add2(&w, nm, (uint64_t)q_out, (uint64_t)s->dim, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", i);
		tsg_add1(&w, nm, (uint64_t)s->dim, -1.0f);
		snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)s->intermediate, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)s->intermediate, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->intermediate, (uint64_t)s->dim, 0.08f);
	}

	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}

typedef struct {
	int			   dim;
	int			   n_heads;
	int			   n_kv_heads;
	int			   head_dim;
	int			   n_layers;
	int			   intermediate;
	int			   conv_kernel;
	int			   ctx;
	int			   vocab;
	uint64_t	   seed;
	const uint8_t *is_conv;
} tsg_lfm2_spec;

static __attribute__((unused)) void tsg_build_lfm2(const char *path, const tsg_lfm2_spec *s) {
	tsg_writer w;
	tsg_winit(&w);
	tsg_seed(s->seed);

	int32_t *kv_arr = malloc((size_t)s->n_layers * sizeof(int32_t));
	for (int i = 0; i < s->n_layers; i++)
		kv_arr[i] = s->is_conv[i] ? 0 : s->n_kv_heads;

	tsg_kv_str(&w, "general.architecture", "lfm2");
	tsg_kv_i32(&w, "lfm2.context_length", s->ctx);
	tsg_kv_i32(&w, "lfm2.block_count", s->n_layers);
	tsg_kv_i32(&w, "lfm2.embedding_length", s->dim);
	tsg_kv_i32(&w, "lfm2.feed_forward_length", s->intermediate);
	tsg_kv_i32(&w, "lfm2.attention.head_count", s->n_heads);
	tsg_kv_arr_i32(&w, "lfm2.attention.head_count_kv", kv_arr, (size_t)s->n_layers);
	tsg_kv_i32(&w, "lfm2.attention.key_length", s->head_dim);
	tsg_kv_f32(&w, "lfm2.attention.layer_norm_rms_epsilon", 1e-5f);
	tsg_kv_i32(&w, "lfm2.shortconv.l_cache", s->conv_kernel);
	free(kv_arr);

	int	 V		  = s->vocab;
	int	 conv_dim = 3 * s->dim;
	int	 q_out	  = s->n_heads * s->head_dim;
	int	 kv_out	  = s->n_kv_heads * s->head_dim;
	char nm[128];
	tsg_add2(&w, "token_embd.weight", (uint64_t)s->dim, (uint64_t)V, 0.10f);
	tsg_add1(&w, "output_norm.weight", (uint64_t)s->dim, -1.0f);
	tsg_add2(&w, "output.weight", (uint64_t)s->dim, (uint64_t)V, 0.10f);
	for (int i = 0; i < s->n_layers; i++) {
		snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", i);
		tsg_add1(&w, nm, (uint64_t)s->dim, -1.0f);
		if (s->is_conv[i]) {
			snprintf(nm, sizeof(nm), "blk.%d.shortconv.in_proj.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)conv_dim, 0.08f);
			snprintf(nm, sizeof(nm), "blk.%d.shortconv.conv.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->conv_kernel, (uint64_t)s->dim, 0.20f);
			snprintf(nm, sizeof(nm), "blk.%d.shortconv.out_proj.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)s->dim, 0.08f);
		} else {
			snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)q_out, 0.08f);
			snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)kv_out, 0.08f);
			snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)kv_out, 0.08f);
			snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", i);
			tsg_add2(&w, nm, (uint64_t)q_out, (uint64_t)s->dim, 0.08f);
			snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", i);
			tsg_add1(&w, nm, (uint64_t)s->head_dim, -1.0f);
			snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", i);
			tsg_add1(&w, nm, (uint64_t)s->head_dim, -1.0f);
		}
		snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", i);
		tsg_add1(&w, nm, (uint64_t)s->dim, -1.0f);
		snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)s->intermediate, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)s->intermediate, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->intermediate, (uint64_t)s->dim, 0.08f);
	}

	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}

typedef struct {
	int		 dim;
	int		 n_heads;
	int		 head_dim;
	int		 n_layers;
	int		 dense_layers;
	int		 intermediate;
	int		 moe_inter;
	int		 n_experts;
	int		 n_experts_used;
	int		 q_lora, kv_lora, qk_rope, qk_nope, qk_head, v_head;
	int		 ctx;
	int		 vocab;
	uint64_t seed;
} tsg_dsa_spec;

static __attribute__((unused)) void tsg_build_glm_dsa(const char *path, const tsg_dsa_spec *s) {
	tsg_writer w;
	tsg_winit(&w);
	tsg_seed(s->seed);

	tsg_kv_str(&w, "general.architecture", "glm-dsa");
	tsg_kv_i32(&w, "glm-dsa.context_length", s->ctx);
	tsg_kv_i32(&w, "glm-dsa.block_count", s->n_layers);
	tsg_kv_i32(&w, "glm-dsa.embedding_length", s->dim);
	tsg_kv_i32(&w, "glm-dsa.feed_forward_length", s->intermediate);
	tsg_kv_i32(&w, "glm-dsa.attention.head_count", s->n_heads);
	tsg_kv_i32(&w, "glm-dsa.attention.head_count_kv", s->n_heads);
	tsg_kv_i32(&w, "glm-dsa.attention.key_length", s->head_dim);
	tsg_kv_f32(&w, "glm-dsa.attention.layer_norm_rms_epsilon", 1e-5f);
	tsg_kv_i32(&w, "glm-dsa.attention.q_lora_rank", s->q_lora);
	tsg_kv_i32(&w, "glm-dsa.attention.kv_lora_rank", s->kv_lora);
	tsg_kv_i32(&w, "glm-dsa.attention.key_length_mla", s->qk_head);
	tsg_kv_i32(&w, "glm-dsa.attention.value_length_mla", s->v_head);
	tsg_kv_i32(&w, "glm-dsa.attention.qk_rope_head_dim", s->qk_rope);
	tsg_kv_i32(&w, "glm-dsa.attention.qk_nope_head_dim", s->qk_nope);
	tsg_kv_i32(&w, "glm-dsa.expert_count", s->n_experts);
	tsg_kv_i32(&w, "glm-dsa.expert_used_count", s->n_experts_used);
	tsg_kv_i32(&w, "glm-dsa.expert_shared_count", 1);
	tsg_kv_i32(&w, "glm-dsa.moe_intermediate_size", s->moe_inter);
	tsg_kv_i32(&w, "glm-dsa.leading_dense_block_count", s->dense_layers);

	int	 V		   = s->vocab;
	int	 kv_a_rows = s->kv_lora + s->qk_rope;
	int	 q_b_rows  = s->n_heads * s->qk_head;
	int	 wo_in	   = s->n_heads * s->v_head;
	int	 sh_inter  = s->moe_inter;
	char nm[128];

	tsg_add2(&w, "token_embd.weight", (uint64_t)s->dim, (uint64_t)V, 0.10f);
	tsg_add1(&w, "output_norm.weight", (uint64_t)s->dim, -1.0f);
	tsg_add2(&w, "output.weight", (uint64_t)s->dim, (uint64_t)V, 0.10f);

	for (int i = 0; i < s->n_layers; i++) {
		snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", i);
		tsg_add1(&w, nm, (uint64_t)s->dim, -1.0f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_q_a.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)s->q_lora, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_q_a_norm.weight", i);
		tsg_add1(&w, nm, (uint64_t)s->q_lora, -1.0f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_q_b.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->q_lora, (uint64_t)q_b_rows, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_kv_a_mqa.weight", i);
		tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)kv_a_rows, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_kv_a_norm.weight", i);
		tsg_add1(&w, nm, (uint64_t)s->kv_lora, -1.0f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_k_b.weight", i);
		tsg_add2(&w, nm, (uint64_t)(s->qk_nope * s->kv_lora), (uint64_t)s->n_heads, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_v_b.weight", i);
		tsg_add2(&w, nm, (uint64_t)(s->kv_lora * s->v_head), (uint64_t)s->n_heads, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", i);
		tsg_add2(&w, nm, (uint64_t)wo_in, (uint64_t)s->dim, 0.08f);
		snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", i);
		tsg_add1(&w, nm, (uint64_t)s->dim, -1.0f);

		if (i >= s->dense_layers) {
			snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)s->n_experts, 0.30f);
			snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)sh_inter, 0.06f);
			snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)sh_inter, 0.06f);
			snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", i);
			tsg_add2(&w, nm, (uint64_t)sh_inter, (uint64_t)s->dim, 0.06f);
			snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", i);
			tsg_add3(&w, nm, (uint64_t)s->dim, (uint64_t)s->moe_inter, (uint64_t)s->n_experts,
					 0.06f);
			snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", i);
			tsg_add3(&w, nm, (uint64_t)s->dim, (uint64_t)s->moe_inter, (uint64_t)s->n_experts,
					 0.06f);
			snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", i);
			tsg_add3(&w, nm, (uint64_t)s->moe_inter, (uint64_t)s->dim, (uint64_t)s->n_experts,
					 0.06f);
		} else {
			snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)s->intermediate, 0.08f);
			snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->dim, (uint64_t)s->intermediate, 0.08f);
			snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", i);
			tsg_add2(&w, nm, (uint64_t)s->intermediate, (uint64_t)s->dim, 0.08f);
		}
	}

	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}

#endif
