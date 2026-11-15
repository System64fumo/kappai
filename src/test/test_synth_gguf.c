#include "test_synth_gguf.h"

#include <math.h>

static const char *tsg_chat_template =
	"{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n"
	"{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}";

static const char *tsg_chat_merged[TSG_CHAT_N_MERGES] = {
	"th", "he", "the", "an", "and", "in", "er", "re",
	"on", "at", "es",  "or", "le",	"me", "is", "it",
};

static const char *tsg_chat_merges[TSG_CHAT_N_MERGES] = {
	"t h", "h e", "th e", "a n", "an d", "i n", "e r", "r e",
	"o n", "a t", "e s",  "o r", "l e",	 "m e", "i s", "i t",
};

static uint64_t tsg_rng_state = 0x853C49E6748FEA9BULL;

static void tsg_put(tsg_buf *w, const void *p, size_t n) {
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
static void tsg_u32(tsg_buf *w, uint32_t v) {
	tsg_put(w, &v, 4);
}
static void tsg_i32(tsg_buf *w, int32_t v) {
	tsg_put(w, &v, 4);
}
static void tsg_u64(tsg_buf *w, uint64_t v) {
	tsg_put(w, &v, 8);
}
static void tsg_f32v(tsg_buf *w, float v) {
	tsg_put(w, &v, 4);
}
static void tsg_str(tsg_buf *w, const char *s) {
	size_t n = strlen(s);
	tsg_u64(w, n);
	tsg_put(w, s, n);
}
void tsg_seed(uint64_t s) {
	tsg_rng_state = s ? s : 0x853C49E6748FEA9BULL;
}
uint32_t tsg_rand(void) {
	uint64_t x = tsg_rng_state;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	tsg_rng_state = x;
	return (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
}
static float tsg_uniform(float lo, float hi) {
	return lo + (hi - lo) * ((float)(tsg_rand() % 1000001) / 1000001.0f);
}
static void tsg_fill_rand_f32(float *p, size_t n, float scale) {
	for (size_t i = 0; i < n; i++)
		p[i] = tsg_uniform(-scale, scale);
}
static void tsg_fill_norm_f32(float *p, size_t n) {
	for (size_t i = 0; i < n; i++)
		p[i] = 1.0f + tsg_uniform(-0.15f, 0.15f);
}
static void tsg_winit(tsg_writer *w) {
	memset(w, 0, sizeof(*w));
	tsg_u32(&w->body, 0);
	tsg_u32(&w->body, 0);
	tsg_u64(&w->body, 0);
	tsg_u64(&w->body, 0);
}
static void tsg_kv_begin(tsg_writer *w, const char *key, uint32_t type) {
	tsg_str(&w->body, key);
	tsg_u32(&w->body, type);
	w->n_kv++;
}
static void tsg_kv_str(tsg_writer *w, const char *key, const char *val) {
	tsg_kv_begin(w, key, TSG_T_STRING);
	tsg_str(&w->body, val);
}
static void tsg_kv_i32(tsg_writer *w, const char *key, int32_t val) {
	tsg_kv_begin(w, key, TSG_T_I32);
	tsg_i32(&w->body, val);
}
static void tsg_kv_f32(tsg_writer *w, const char *key, float val) {
	tsg_kv_begin(w, key, TSG_T_F32);
	tsg_f32v(&w->body, val);
}
static void tsg_kv_bool(tsg_writer *w, const char *key, int val) {
	tsg_kv_begin(w, key, TSG_T_BOOL);
	unsigned char b = (unsigned char)(val ? 1 : 0);
	tsg_put(&w->body, &b, 1);
}
static void tsg_kv_arr_i32(tsg_writer *w, const char *key, const int32_t *v, size_t n) {
	tsg_kv_begin(w, key, TSG_T_ARRAY);
	tsg_u32(&w->body, TSG_T_I32);
	tsg_u64(&w->body, n);
	for (size_t i = 0; i < n; i++)
		tsg_i32(&w->body, v[i]);
}
static void tsg_kv_arr_str(tsg_writer *w, const char *key, const char *const *v, size_t n) {
	tsg_kv_begin(w, key, TSG_T_ARRAY);
	tsg_u32(&w->body, TSG_T_STRING);
	tsg_u64(&w->body, n);
	for (size_t i = 0; i < n; i++)
		tsg_str(&w->body, v[i]);
}
static float *tsg_tensor_new(tsg_writer *w, const char *name, uint32_t ndims, const uint64_t *dims,
							 float scale) {
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
static float *tsg_add2(tsg_writer *w, const char *name, uint64_t d0, uint64_t d1, float scale) {
	uint64_t dims[2] = {d0, d1};
	return tsg_tensor_new(w, name, 2, dims, scale);
}
static float *tsg_add1(tsg_writer *w, const char *name, uint64_t d0, float scale) {
	return tsg_tensor_new(w, name, 1, &d0, scale);
}
static float *tsg_add3(tsg_writer *w, const char *name, uint64_t d0, uint64_t d1, uint64_t d2,
					   float scale) {
	uint64_t dims[3] = {d0, d1, d2};
	return tsg_tensor_new(w, name, 3, dims, scale);
}
static void tsg_tensor_free_all(tsg_writer *w) {
	for (int i = 0; i < w->n_tensors; i++)
		free(w->tensors[i].data);
	w->n_tensors = 0;
	free(w->body.buf);
	w->body.buf = NULL;
	w->body.len = w->body.cap = 0;
}
static void tsg_ensure_parent_dirs(const char *path) {
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
static void tsg_write_failed(const char *path) {
	fprintf(stderr, "error: cannot write gguf fixture '%s'\n", path);
	abort();
}
static int tsg_finish(tsg_writer *w, const char *path) {
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
void tsg_build_array_file(const char *path, uint32_t type, uint64_t count, const void *data,
						  size_t bytes) {
	tsg_writer w;
	tsg_winit(&w);
	tsg_kv_begin(&w, "array", TSG_T_ARRAY);
	tsg_u32(&w.body, type);
	tsg_u64(&w.body, count);
	if (bytes)
		tsg_put(&w.body, data, bytes);
	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}

void tsg_build_vocab_file(const char *path, const char *model_name, const char *pre,
						  int has_space_flag, int space_flag, const char *const *tokens,
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
static void tsg_build_chat_vocab_arrays(char (*storage)[8], const char **toks, int32_t *types) {
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
	for (int i = 0; i < TSG_CHAT_N_MERGES; i++) {
		snprintf(storage[id], 8, "%s", tsg_chat_merged[i]);
		toks[id]	= storage[id];
		types[id++] = TSG_TOK_NORMAL;
	}
}
void tsg_build_chat_llama(const char *path, const tsg_llama_spec *s) {
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
	tsg_kv_arr_str(&w, "tokenizer.ggml.merges", tsg_chat_merges, TSG_CHAT_N_MERGES);
	tsg_kv_str(&w, "tokenizer.ggml.model", "gpt2");
	tsg_kv_i32(&w, "tokenizer.ggml.unknown_token_id", 0);
	tsg_kv_str(&w, "tokenizer.chat_template", tsg_chat_template);

	int	   V = TSG_CHAT_VOCAB;
	char   nm[128];
	float *embd = tsg_add2(&w, "token_embd.weight", (uint64_t)s->dim, (uint64_t)V, 0.10f);
	tsg_add1(&w, "output_norm.weight", (uint64_t)s->dim, -1.0f);
	if (!s->tied) {
		float *outw = tsg_add2(&w, "output.weight", (uint64_t)s->dim, (uint64_t)V, 0.10f);
		if (s->output_copies_embd && embd && outw)
			memcpy(outw, embd, sizeof(float) * (size_t)s->dim * (size_t)V);
	} else {
		float *discard = malloc((size_t)s->dim * (size_t)V * sizeof(float));
		if (!discard)
			abort();
		tsg_fill_rand_f32(discard, (size_t)s->dim * (size_t)V, 0.10f);
		free(discard);
	}
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
void tsg_build_lfm2(const char *path, const tsg_lfm2_spec *s) {
	tsg_writer w;
	tsg_winit(&w);
	tsg_seed(s->seed);

	int32_t *kv_arr = malloc((size_t)s->n_layers * sizeof(int32_t));
	for (int i = 0; i < s->n_layers; i++)
		kv_arr[i] = i >= s->n_layers - s->n_decision_layers ? s->n_heads
					: s->is_conv[i]							? 0
															: s->n_kv_heads;

	tsg_kv_str(&w, "general.architecture", "lfm2");
	tsg_kv_i32(&w, "lfm2.context_length", s->ctx);
	tsg_kv_i32(&w, "lfm2.block_count", s->n_layers);
	tsg_kv_i32(&w, "lfm2.embedding_length", s->dim);
	if (s->n_decision_layers) {
		int32_t *ffn = malloc((size_t)s->n_layers * sizeof(int32_t));
		for (int i = 0; i < s->n_layers; i++)
			ffn[i] =
				i >= s->n_layers - s->n_decision_layers ? 2 * s->intermediate : s->intermediate;
		tsg_kv_arr_i32(&w, "lfm2.feed_forward_length", ffn, (size_t)s->n_layers);
		free(ffn);
		tsg_kv_str(&w, "lfm2.decision.type", "lfm2-d1-omni");
		tsg_kv_i32(&w, "lfm2.decision.block_count", s->n_decision_layers);
		tsg_kv_bool(&w, "lfm2.attention.causal", 0);
		tsg_kv_f32(&w, "lfm2.attention.layer_norm_epsilon", 1e-5f);
		tsg_kv_str(&w, "tokenizer.chat_template.systemone", "{{ state }}");
	} else
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
	if (s->n_decision_layers) {
		tsg_add2(&w, "token_types.weight", s->dim, 3, 0.1f);
		tsg_add1(&w, "cls.norm.weight", s->dim, -1.0f);
		tsg_add1(&w, "cls.norm.bias", s->dim, 0.1f);
		tsg_add2(&w, "cls.weight", s->dim, s->dim, 0.1f);
		tsg_add1(&w, "cls.bias", s->dim, 0.1f);
		tsg_add1(&w, "cls.output.weight", s->dim, 0.1f);
		tsg_add1(&w, "cls.output.bias", 1, 0.1f);
	} else
		tsg_add2(&w, "output.weight", (uint64_t)s->dim, (uint64_t)V, 0.10f);
	for (int i = 0; i < s->n_layers; i++) {
		snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", i);
		tsg_add1(&w, nm, (uint64_t)s->dim, -1.0f);
		if (i >= s->n_layers - s->n_decision_layers) {
			static const char *const norms[] = {"attn_norm", "ffn_norm"};
			for (size_t j = 0; j < 2; j++) {
				if (j) {
					snprintf(nm, sizeof(nm), "blk.%d.%s.weight", i, norms[j]);
					tsg_add1(&w, nm, s->dim, -1.0f);
				}
				snprintf(nm, sizeof(nm), "blk.%d.%s.bias", i, norms[j]);
				tsg_add1(&w, nm, s->dim, 0.1f);
			}
			const char *names[]	  = {"attn_qkv", "attn_output", "ffn_up", "ffn_down"};
			int			inputs[]  = {s->dim, s->dim, s->dim, 2 * s->intermediate};
			int			outputs[] = {3 * s->dim, s->dim, 2 * s->intermediate, s->dim};
			for (size_t j = 0; j < 4; j++) {
				snprintf(nm, sizeof(nm), "blk.%d.%s.weight", i, names[j]);
				tsg_add2(&w, nm, inputs[j], outputs[j], 0.08f);
				snprintf(nm, sizeof(nm), "blk.%d.%s.bias", i, names[j]);
				tsg_add1(&w, nm, outputs[j], 0.1f);
			}
			continue;
		}
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
void tsg_build_glm_dsa(const char *path, const tsg_dsa_spec *s) {
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

static float *tsg_blk1(tsg_writer *w, int i, const char *suffix, uint64_t d0, float scale) {
	char nm[128];
	snprintf(nm, sizeof(nm), "blk.%d.%s", i, suffix);
	return tsg_add1(w, nm, d0, scale);
}

static float *tsg_blk2(tsg_writer *w, int i, const char *suffix, uint64_t d0, uint64_t d1,
					   float scale) {
	char nm[128];
	snprintf(nm, sizeof(nm), "blk.%d.%s", i, suffix);
	return tsg_add2(w, nm, d0, d1, scale);
}

static float *tsg_blk3(tsg_writer *w, int i, const char *suffix, uint64_t d0, uint64_t d1,
					   uint64_t d2, float scale) {
	char nm[128];
	snprintf(nm, sizeof(nm), "blk.%d.%s", i, suffix);
	return tsg_add3(w, nm, d0, d1, d2, scale);
}

static void tsg_kvp_i32(tsg_writer *w, const char *prefix, const char *suffix, int32_t v) {
	char key[160];
	snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
	tsg_kv_i32(w, key, v);
}

static void tsg_kvp_f32(tsg_writer *w, const char *prefix, const char *suffix, float v) {
	char key[160];
	snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
	tsg_kv_f32(w, key, v);
}

static void tsg_kvp_arr_i32(tsg_writer *w, const char *prefix, const char *suffix, const int32_t *v,
							size_t n) {
	char key[160];
	snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
	tsg_kv_arr_i32(w, key, v, n);
}

static void tsg_arch_header(tsg_writer *w, const char *arch, const char *prefix,
							const tsg_arch_spec *s) {
	tsg_kv_str(w, "general.architecture", arch);
	tsg_kvp_i32(w, prefix, "context_length", s->ctx);
	tsg_kvp_i32(w, prefix, "block_count", s->n_layers);
	tsg_kvp_i32(w, prefix, "embedding_length", s->dim);
	tsg_kvp_i32(w, prefix, "attention.head_count", s->n_heads);
	tsg_kvp_f32(w, prefix, "attention.layer_norm_rms_epsilon", 1e-5f);
}

static void tsg_arch_embeddings(tsg_writer *w, const tsg_arch_spec *s, int with_output) {
	tsg_add2(w, "token_embd.weight", (uint64_t)s->dim, (uint64_t)s->vocab, 0.10f);
	tsg_add1(w, "output_norm.weight", (uint64_t)s->dim, -1.0f);
	if (with_output)
		tsg_add2(w, "output.weight", (uint64_t)s->dim, (uint64_t)s->vocab, 0.10f);
}

static void tsg_blk_dense_ffn(tsg_writer *w, int i, int dim, int inter) {
	tsg_blk2(w, i, "ffn_gate.weight", (uint64_t)dim, (uint64_t)inter, 0.08f);
	tsg_blk2(w, i, "ffn_up.weight", (uint64_t)dim, (uint64_t)inter, 0.08f);
	tsg_blk2(w, i, "ffn_down.weight", (uint64_t)inter, (uint64_t)dim, 0.08f);
}

static void tsg_blk_attention(tsg_writer *w, int i, int dim, int q_out, int kv_out, int with_k,
							  int with_v) {
	tsg_blk2(w, i, "attn_q.weight", (uint64_t)dim, (uint64_t)q_out, 0.08f);
	if (with_k)
		tsg_blk2(w, i, "attn_k.weight", (uint64_t)dim, (uint64_t)kv_out, 0.08f);
	if (with_v)
		tsg_blk2(w, i, "attn_v.weight", (uint64_t)dim, (uint64_t)kv_out, 0.08f);
	tsg_blk2(w, i, "attn_output.weight", (uint64_t)q_out, (uint64_t)dim, 0.08f);
}

static void tsg_blk_mla(tsg_writer *w, int i, int dim, int n_heads, int q_lora, int kv_lora,
						int qk_nope, int qk_head, int v_head) {
	tsg_blk2(w, i, "attn_q_a.weight", (uint64_t)dim, (uint64_t)q_lora, 0.08f);
	tsg_blk1(w, i, "attn_q_a_norm.weight", (uint64_t)q_lora, -1.0f);
	tsg_blk2(w, i, "attn_q_b.weight", (uint64_t)q_lora, (uint64_t)(n_heads * qk_head), 0.08f);
	tsg_blk2(w, i, "attn_kv_a_mqa.weight", (uint64_t)dim, (uint64_t)kv_lora, 0.08f);
	tsg_blk1(w, i, "attn_kv_a_norm.weight", (uint64_t)kv_lora, -1.0f);
	tsg_blk2(w, i, "attn_k_b.weight", (uint64_t)(qk_nope * kv_lora), (uint64_t)n_heads, 0.08f);
	tsg_blk2(w, i, "attn_v_b.weight", (uint64_t)(kv_lora * v_head), (uint64_t)n_heads, 0.08f);
	tsg_blk2(w, i, "attn_output.weight", (uint64_t)(n_heads * v_head), (uint64_t)dim, 0.08f);
}

void tsg_build_qwen3vl(const char *path, const tsg_arch_spec *s) {
	tsg_writer w;
	tsg_winit(&w);
	tsg_seed(s->seed);

	tsg_arch_header(&w, "qwen3vl", "qwen3vl", s);
	tsg_kvp_i32(&w, "qwen3vl", "feed_forward_length", s->intermediate);
	tsg_kvp_i32(&w, "qwen3vl", "attention.head_count_kv", s->n_kv_heads);
	tsg_kvp_i32(&w, "qwen3vl", "attention.key_length", s->head_dim);
	tsg_arch_embeddings(&w, s, 1);

	int q_out  = s->n_heads * s->head_dim;
	int kv_out = s->n_kv_heads * s->head_dim;
	for (int i = 0; i < s->n_layers; i++) {
		tsg_blk1(&w, i, "attn_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk_attention(&w, i, s->dim, q_out, kv_out, 1, 1);
		tsg_blk1(&w, i, "attn_q_norm.weight", (uint64_t)s->head_dim, -1.0f);
		tsg_blk1(&w, i, "attn_k_norm.weight", (uint64_t)s->head_dim, -1.0f);
		tsg_blk1(&w, i, "ffn_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk_dense_ffn(&w, i, s->dim, s->intermediate);
	}

	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}

void tsg_build_lfm3(const char *path, const tsg_arch_spec *s) {
	enum { CONV_KERNEL = 4, SLIDING_WINDOW = 12 };
	tsg_writer w;
	tsg_winit(&w);
	tsg_seed(s->seed);

	int32_t *kv_arr = malloc((size_t)s->n_layers * sizeof(int32_t));
	for (int i = 0; i < s->n_layers; i++)
		kv_arr[i] = i % 3 == 2 ? s->n_kv_heads : 0;

	tsg_arch_header(&w, "lfm3", "lfm3", s);
	tsg_kvp_i32(&w, "lfm3", "feed_forward_length", s->intermediate);
	tsg_kvp_arr_i32(&w, "lfm3", "attention.head_count_kv", kv_arr, (size_t)s->n_layers);
	tsg_kvp_i32(&w, "lfm3", "attention.key_length", s->head_dim);
	tsg_kvp_i32(&w, "lfm3", "attention.sliding_window", SLIDING_WINDOW);
	tsg_kvp_i32(&w, "lfm3", "shortconv.l_cache", CONV_KERNEL);
	tsg_arch_embeddings(&w, s, 1);

	int q_out  = s->n_heads * s->head_dim;
	int kv_out = s->n_kv_heads * s->head_dim;
	for (int i = 0; i < s->n_layers; i++) {
		tsg_blk1(&w, i, "attn_norm.weight", (uint64_t)s->dim, -1.0f);
		if (kv_arr[i] == 0) {
			tsg_blk2(&w, i, "shortconv.in_proj.weight", (uint64_t)s->dim, (uint64_t)(3 * s->dim),
					 0.08f);
			tsg_blk2(&w, i, "shortconv.conv.weight", CONV_KERNEL, (uint64_t)s->dim, 0.20f);
			tsg_blk2(&w, i, "shortconv.out_proj.weight", (uint64_t)s->dim, (uint64_t)s->dim, 0.08f);
		} else {
			tsg_blk_attention(&w, i, s->dim, q_out, kv_out, 1, 1);
			tsg_blk1(&w, i, "attn_q_norm.weight", (uint64_t)s->head_dim, -1.0f);
			tsg_blk1(&w, i, "attn_k_norm.weight", (uint64_t)s->head_dim, -1.0f);
		}
		tsg_blk1(&w, i, "post_attention_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk1(&w, i, "ffn_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk1(&w, i, "post_ffw_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk_dense_ffn(&w, i, s->dim, s->intermediate);
	}
	free(kv_arr);

	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}

static void tsg_build_gemma4_impl(const char *path, const tsg_arch_spec *s, int moe) {
	enum {
		SLIDING_WINDOW	 = 12,
		SHARED_KV_LAYERS = 2,
		PLE_DIM			 = 8,
		N_EXPERTS		 = 4,
		N_USED			 = 2,
		EXPERT_INTER	 = 16
	};
	tsg_writer w;
	tsg_winit(&w);
	tsg_seed(s->seed);

	int32_t *sliding = malloc((size_t)s->n_layers * sizeof(int32_t));
	for (int i = 0; i < s->n_layers; i++)
		sliding[i] = i % 3 == 2 ? 0 : 1;
	int head_swa	= s->head_dim;
	int head_global = 2 * s->head_dim;

	tsg_arch_header(&w, "gemma4", "gemma4", s);
	tsg_kvp_i32(&w, "gemma4", "feed_forward_length", s->intermediate);
	tsg_kvp_i32(&w, "gemma4", "attention.head_count_kv", s->n_kv_heads);
	tsg_kvp_i32(&w, "gemma4", "attention.key_length", head_global);
	tsg_kvp_i32(&w, "gemma4", "attention.key_length_swa", head_swa);
	tsg_kvp_f32(&w, "gemma4", "rope.freq_base", 1000000.0f);
	tsg_kvp_f32(&w, "gemma4", "rope.freq_base_swa", 10000.0f);
	tsg_kvp_i32(&w, "gemma4", "attention.sliding_window", SLIDING_WINDOW);
	tsg_kvp_arr_i32(&w, "gemma4", "attention.sliding_window_pattern", sliding, (size_t)s->n_layers);
	tsg_kvp_i32(&w, "gemma4", "attention.shared_kv_layers", SHARED_KV_LAYERS);
	tsg_kvp_i32(&w, "gemma4", "embedding_length_per_layer_input", PLE_DIM);
	tsg_kvp_f32(&w, "gemma4", "final_logit_softcapping", 30.0f);
	if (moe) {
		tsg_kvp_i32(&w, "gemma4", "expert_count", N_EXPERTS);
		tsg_kvp_i32(&w, "gemma4", "expert_used_count", N_USED);
		tsg_kvp_i32(&w, "gemma4", "expert_feed_forward_length", EXPERT_INTER);
	}

	tsg_arch_embeddings(&w, s, 0);
	tsg_add2(&w, "per_layer_token_embd.weight", (uint64_t)(PLE_DIM * s->n_layers),
			 (uint64_t)s->vocab, 0.05f);
	tsg_add2(&w, "per_layer_model_proj.weight", (uint64_t)s->dim, (uint64_t)(PLE_DIM * s->n_layers),
			 0.05f);
	tsg_add1(&w, "per_layer_proj_norm.weight", PLE_DIM, -1.0f);

	for (int i = 0; i < s->n_layers; i++) {
		int global = !sliding[i];
		int hd	   = global ? head_global : head_swa;
		int q_out  = s->n_heads * hd;
		int kv_out = s->n_kv_heads * hd;
		int has_kv = i < s->n_layers - SHARED_KV_LAYERS;

		tsg_blk1(&w, i, "attn_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk_attention(&w, i, s->dim, q_out, kv_out, has_kv, has_kv && !global);
		tsg_blk1(&w, i, "attn_q_norm.weight", (uint64_t)hd, -1.0f);
		if (has_kv)
			tsg_blk1(&w, i, "attn_k_norm.weight", (uint64_t)hd, -1.0f);
		tsg_blk1(&w, i, "post_attention_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk1(&w, i, "ffn_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk1(&w, i, "post_ffw_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk_dense_ffn(&w, i, s->dim, s->intermediate);
		if (moe) {
			tsg_blk2(&w, i, "ffn_gate_inp.weight", (uint64_t)s->dim, N_EXPERTS, 0.30f);
			tsg_blk1(&w, i, "ffn_gate_inp.scale", (uint64_t)s->dim, -1.0f);
			tsg_blk3(&w, i, "ffn_gate_up_exps.weight", (uint64_t)s->dim, 2 * EXPERT_INTER,
					 N_EXPERTS, 0.06f);
			tsg_blk3(&w, i, "ffn_down_exps.weight", EXPERT_INTER, (uint64_t)s->dim, N_EXPERTS,
					 0.06f);
			tsg_blk1(&w, i, "pre_ffw_norm_2.weight", (uint64_t)s->dim, -1.0f);
			tsg_blk1(&w, i, "post_ffw_norm_1.weight", (uint64_t)s->dim, -1.0f);
			tsg_blk1(&w, i, "post_ffw_norm_2.weight", (uint64_t)s->dim, -1.0f);
		}
		tsg_blk1(&w, i, "post_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk2(&w, i, "inp_gate.weight", (uint64_t)s->dim, PLE_DIM, 0.08f);
		tsg_blk2(&w, i, "proj.weight", PLE_DIM, (uint64_t)s->dim, 0.08f);
		tsg_blk1(&w, i, "layer_output_scale.weight", 1, 0.05f)[0] = 0.9f;
	}
	free(sliding);

	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}

void tsg_build_gemma4(const char *path, const tsg_arch_spec *s) {
	tsg_build_gemma4_impl(path, s, 0);
}

void tsg_build_gemma4_moe(const char *path, const tsg_arch_spec *s) {
	tsg_build_gemma4_impl(path, s, 1);
}

void tsg_build_qwen35(const char *path, const tsg_arch_spec *s) {
	enum { CONV_KERNEL = 4, STATE = 16, KEY_HEADS = 2, VALUE_HEADS = 4, INTERVAL = 4 };
	int inner	 = STATE * VALUE_HEADS;
	int key_dim	 = STATE * KEY_HEADS;
	int conv_dim = 2 * key_dim + inner;
	int value_hd = inner / VALUE_HEADS;

	tsg_writer w;
	tsg_winit(&w);
	tsg_seed(s->seed);

	tsg_arch_header(&w, "qwen35", "qwen35", s);
	tsg_kvp_i32(&w, "qwen35", "feed_forward_length", s->intermediate);
	tsg_kvp_i32(&w, "qwen35", "attention.head_count_kv", s->n_kv_heads);
	tsg_kvp_i32(&w, "qwen35", "attention.key_length", s->head_dim);
	tsg_kvp_i32(&w, "qwen35", "rope.dimension_count", s->head_dim / 2);
	tsg_kvp_i32(&w, "qwen35", "ssm.conv_kernel", CONV_KERNEL);
	tsg_kvp_i32(&w, "qwen35", "ssm.inner_size", inner);
	tsg_kvp_i32(&w, "qwen35", "ssm.state_size", STATE);
	tsg_kvp_i32(&w, "qwen35", "ssm.time_step_rank", VALUE_HEADS);
	tsg_kvp_i32(&w, "qwen35", "ssm.group_count", KEY_HEADS);
	tsg_kvp_i32(&w, "qwen35", "attention.full_attention_interval", INTERVAL);
	tsg_arch_embeddings(&w, s, 1);

	int q_out  = s->n_heads * s->head_dim;
	int kv_out = s->n_kv_heads * s->head_dim;
	for (int i = 0; i < s->n_layers; i++) {
		int recurrent = ((i + 1) % INTERVAL) != 0;
		tsg_blk1(&w, i, "attn_norm.weight", (uint64_t)s->dim, -1.0f);
		if (recurrent) {
			tsg_blk2(&w, i, "attn_qkv.weight", (uint64_t)s->dim, (uint64_t)conv_dim, 0.08f);
			tsg_blk2(&w, i, "attn_gate.weight", (uint64_t)s->dim, (uint64_t)inner, 0.08f);
			tsg_blk2(&w, i, "ssm_conv1d.weight", CONV_KERNEL, (uint64_t)conv_dim, 0.20f);
			tsg_blk1(&w, i, "ssm_dt.bias", VALUE_HEADS, 0.10f);
			float *a = tsg_blk1(&w, i, "ssm_a", VALUE_HEADS, 0.50f);
			for (int h = 0; h < VALUE_HEADS; h++)
				a[h] = -(0.2f + fabsf(a[h]));
			tsg_blk2(&w, i, "ssm_beta.weight", (uint64_t)s->dim, VALUE_HEADS, 0.08f);
			tsg_blk2(&w, i, "ssm_alpha.weight", (uint64_t)s->dim, VALUE_HEADS, 0.08f);
			tsg_blk1(&w, i, "ssm_norm.weight", (uint64_t)value_hd, -1.0f);
			tsg_blk2(&w, i, "ssm_out.weight", (uint64_t)inner, (uint64_t)s->dim, 0.08f);
		} else {
			tsg_blk2(&w, i, "attn_q.weight", (uint64_t)s->dim, (uint64_t)(2 * q_out), 0.08f);
			tsg_blk2(&w, i, "attn_k.weight", (uint64_t)s->dim, (uint64_t)kv_out, 0.08f);
			tsg_blk2(&w, i, "attn_v.weight", (uint64_t)s->dim, (uint64_t)kv_out, 0.08f);
			tsg_blk2(&w, i, "attn_output.weight", (uint64_t)q_out, (uint64_t)s->dim, 0.08f);
			tsg_blk1(&w, i, "attn_q_norm.weight", (uint64_t)s->head_dim, -1.0f);
			tsg_blk1(&w, i, "attn_k_norm.weight", (uint64_t)s->head_dim, -1.0f);
		}
		tsg_blk1(&w, i, "post_attention_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk_dense_ffn(&w, i, s->dim, s->intermediate);
	}

	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}

void tsg_build_glm5_next(const char *path, const tsg_arch_spec *s) {
	enum {
		CONV_KERNEL	 = 4,
		KDA_HEAD	 = 16,
		HC			 = 4,
		SINKHORN	 = 2,
		N_EXPERTS	 = 4,
		N_USED		 = 2,
		MOE_INTER	 = 16,
		DENSE_LAYERS = 1,
		Q_LORA		 = 32,
		KV_LORA		 = 32,
		QK_HEAD		 = 32,
		V_HEAD		 = 32
	};
	const char *p		= "glm5-next";
	int			d_inner = KDA_HEAD * s->n_heads;
	uint64_t	hc_dim	= (uint64_t)HC * (uint64_t)s->dim;
	uint64_t	hc_mix	= (uint64_t)(2 + HC) * HC;

	int32_t *kv_arr = malloc((size_t)s->n_layers * sizeof(int32_t));
	for (int i = 0; i < s->n_layers; i++)
		kv_arr[i] = i % 4 == 3 ? s->n_heads : 0;

	tsg_writer w;
	tsg_winit(&w);
	tsg_seed(s->seed);

	tsg_arch_header(&w, p, p, s);
	tsg_kvp_i32(&w, p, "feed_forward_length", s->intermediate);
	tsg_kvp_arr_i32(&w, p, "attention.head_count_kv", kv_arr, (size_t)s->n_layers);
	tsg_kvp_i32(&w, p, "attention.q_lora_rank", Q_LORA);
	tsg_kvp_i32(&w, p, "attention.kv_lora_rank", KV_LORA);
	tsg_kvp_i32(&w, p, "attention.key_length_mla", QK_HEAD);
	tsg_kvp_i32(&w, p, "attention.value_length_mla", V_HEAD);
	tsg_kvp_i32(&w, p, "rope.dimension_count", 0);
	tsg_kvp_i32(&w, p, "expert_count", N_EXPERTS);
	tsg_kvp_i32(&w, p, "expert_used_count", N_USED);
	tsg_kvp_i32(&w, p, "expert_shared_count", 1);
	tsg_kvp_i32(&w, p, "expert_feed_forward_length", MOE_INTER);
	tsg_kvp_i32(&w, p, "leading_dense_block_count", DENSE_LAYERS);
	tsg_kvp_i32(&w, p, "ssm.conv_kernel", CONV_KERNEL);
	tsg_kvp_i32(&w, p, "kda.head_dim", KDA_HEAD);
	tsg_kvp_i32(&w, p, "hyper_connection.count", HC);
	tsg_kvp_i32(&w, p, "hyper_connection.sinkhorn_iterations", SINKHORN);
	tsg_kvp_f32(&w, p, "hyper_connection.epsilon", 1e-6f);
	tsg_arch_embeddings(&w, s, 1);

	for (int i = 0; i < s->n_layers; i++) {
		tsg_blk1(&w, i, "attn_norm.weight", (uint64_t)s->dim, -1.0f);
		tsg_blk2(&w, i, "hc_attn_fn.weight", hc_dim, hc_mix, 0.05f);
		tsg_blk1(&w, i, "hc_attn_base.weight", hc_mix, 0.05f);
		tsg_blk1(&w, i, "hc_attn_scale.weight", 3, 0.20f);
		tsg_blk2(&w, i, "hc_ffn_fn.weight", hc_dim, hc_mix, 0.05f);
		tsg_blk1(&w, i, "hc_ffn_base.weight", hc_mix, 0.05f);
		tsg_blk1(&w, i, "hc_ffn_scale.weight", 3, 0.20f);

		if (kv_arr[i] == 0) {
			tsg_blk2(&w, i, "attn_q.weight", (uint64_t)s->dim, (uint64_t)d_inner, 0.08f);
			tsg_blk2(&w, i, "attn_k.weight", (uint64_t)s->dim, (uint64_t)d_inner, 0.08f);
			tsg_blk2(&w, i, "attn_v.weight", (uint64_t)s->dim, (uint64_t)d_inner, 0.08f);
			tsg_blk2(&w, i, "attn_output.weight", (uint64_t)d_inner, (uint64_t)s->dim, 0.08f);
			tsg_blk2(&w, i, "ssm_f_a.weight", (uint64_t)s->dim, KDA_HEAD, 0.08f);
			tsg_blk2(&w, i, "ssm_f_b.weight", KDA_HEAD, (uint64_t)d_inner, 0.08f);
			tsg_blk2(&w, i, "ssm_g_a.weight", (uint64_t)s->dim, KDA_HEAD, 0.08f);
			tsg_blk2(&w, i, "ssm_g_b.weight", KDA_HEAD, (uint64_t)d_inner, 0.08f);
			tsg_blk2(&w, i, "ssm_beta.weight", (uint64_t)s->dim, (uint64_t)s->n_heads, 0.08f);
			tsg_blk1(&w, i, "ssm_a", (uint64_t)s->n_heads, 0.20f);
			tsg_blk1(&w, i, "ssm_dt.bias", (uint64_t)d_inner, 0.10f);
			tsg_blk1(&w, i, "ssm_norm.weight", KDA_HEAD, -1.0f);
			tsg_blk2(&w, i, "ssm_conv1d_q.weight", CONV_KERNEL, (uint64_t)d_inner, 0.20f);
			tsg_blk2(&w, i, "ssm_conv1d_k.weight", CONV_KERNEL, (uint64_t)d_inner, 0.20f);
			tsg_blk2(&w, i, "ssm_conv1d_v.weight", CONV_KERNEL, (uint64_t)d_inner, 0.20f);
		} else {
			tsg_blk_mla(&w, i, s->dim, s->n_heads, Q_LORA, KV_LORA, QK_HEAD, QK_HEAD, V_HEAD);
		}

		tsg_blk1(&w, i, "ffn_norm.weight", (uint64_t)s->dim, -1.0f);
		if (i < DENSE_LAYERS) {
			tsg_blk_dense_ffn(&w, i, s->dim, s->intermediate);
			continue;
		}
		tsg_blk2(&w, i, "ffn_gate_inp.weight", (uint64_t)s->dim, N_EXPERTS, 0.30f);
		tsg_blk2(&w, i, "ffn_gate_shexp.weight", (uint64_t)s->dim, MOE_INTER, 0.06f);
		tsg_blk2(&w, i, "ffn_up_shexp.weight", (uint64_t)s->dim, MOE_INTER, 0.06f);
		tsg_blk2(&w, i, "ffn_down_shexp.weight", MOE_INTER, (uint64_t)s->dim, 0.06f);
		tsg_blk3(&w, i, "ffn_gate_exps.weight", (uint64_t)s->dim, MOE_INTER, N_EXPERTS, 0.06f);
		tsg_blk3(&w, i, "ffn_up_exps.weight", (uint64_t)s->dim, MOE_INTER, N_EXPERTS, 0.06f);
		tsg_blk3(&w, i, "ffn_down_exps.weight", MOE_INTER, (uint64_t)s->dim, N_EXPERTS, 0.06f);
	}
	free(kv_arr);

	if (tsg_finish(&w, path) != 0)
		tsg_write_failed(path);
	tsg_tensor_free_all(&w);
}
