#include "hf_meta.h"

#include "gguf.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HM_MAGIC "GGUF"
#define HM_TYPE_STRING 8
#define HM_TYPE_ARRAY 9
#define HM_TYPE_COUNT 13
#define HM_KEY_MAX 128
#define HM_STRING_LIMIT (1u << 20)
#define HM_MAX_ENTRIES (1u << 20)
#define HM_MAX_DIMS 4
#define MIX_SHOWN 3

typedef struct {
	const unsigned char *p;
	size_t				 len;
	size_t				 pos;
	bool				 short_read;
	bool				 bad;
} reader;

typedef struct {
	bool	  ok;
	bool	  array;
	long long sum;
	long long count;
} int_value;

typedef struct {
	const char *suffix;
	size_t		offset;
} slot_desc;

static const size_t TYPE_SIZE[HM_TYPE_COUNT] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

static const slot_desc SLOTS[] = {
	{"block_count", offsetof(hf_meta, n_layer)},
	{"embedding_length", offsetof(hf_meta, n_embd)},
	{"context_length", offsetof(hf_meta, n_ctx)},
	{"expert_count", offsetof(hf_meta, n_expert)},
	{"attention.head_count", offsetof(hf_meta, n_head)},
	{"attention.head_count_kv", offsetof(hf_meta, head_kv)},
	{"attention.key_length", offsetof(hf_meta, key_len)},
	{"attention.value_length", offsetof(hf_meta, val_len)},
	{"nextn_predict_layers", offsetof(hf_meta, n_nextn)},
};

static const struct {
	unsigned	id;
	const char *name;
} TYPE_NAMES[] = {
	{0, "F32"},		{1, "F16"},	   {2, "Q4_0"},		{3, "Q4_1"},	{6, "Q5_0"},	 {7, "Q5_1"},
	{8, "Q8_0"},	{9, "Q8_1"},   {10, "Q2_K"},	{11, "Q3_K"},	{12, "Q4_K"},	 {13, "Q5_K"},
	{14, "Q6_K"},	{15, "Q8_K"},  {16, "IQ2_XXS"}, {17, "IQ2_XS"}, {18, "IQ3_XXS"}, {19, "IQ1_S"},
	{20, "IQ4_NL"}, {21, "IQ3_S"}, {22, "IQ2_S"},	{23, "IQ4_XS"}, {24, "I8"},		 {25, "I16"},
	{26, "I32"},	{27, "I64"},   {28, "F64"},		{29, "IQ1_M"},	{30, "BF16"},	 {34, "TQ1_0"},
	{35, "TQ2_0"},	{39, "MXFP4"},
};

void hf_meta_init(hf_meta *m) {
	memset(m, 0, sizeof(*m));
	m->n_layer = m->n_embd = m->n_head = m->head_kv = m->kv_sum = -1;
	m->key_len = m->val_len = m->n_ctx = m->n_expert = m->n_nextn = -1;
}

static bool need(reader *r, size_t n) {
	if (r->short_read || r->len - r->pos < n) {
		r->short_read = true;
		return false;
	}
	return true;
}

static uint64_t read_uint(reader *r, size_t bytes) {
	if (!need(r, bytes))
		return 0;
	uint64_t v = 0;
	for (size_t i = 0; i < bytes; i++)
		v |= (uint64_t)r->p[r->pos + i] << (8 * i);
	r->pos += bytes;
	return v;
}

static bool skip_bytes(reader *r, uint64_t n) {
	if (r->short_read || n > r->len - r->pos) {
		r->short_read = true;
		return false;
	}
	r->pos += (size_t)n;
	return true;
}

static bool skip_value(reader *r, uint32_t type) {
	if (type >= HM_TYPE_COUNT)
		return false;
	if (type == HM_TYPE_STRING)
		return skip_bytes(r, read_uint(r, 8)) && !r->short_read;
	if (type != HM_TYPE_ARRAY)
		return skip_bytes(r, TYPE_SIZE[type]);
	uint32_t elem  = (uint32_t)read_uint(r, 4);
	uint64_t count = read_uint(r, 8);
	if (r->short_read || elem >= HM_TYPE_COUNT)
		return false;
	if (TYPE_SIZE[elem] && count > (r->len - r->pos) / TYPE_SIZE[elem]) {
		r->short_read = true;
		return false;
	}
	if (TYPE_SIZE[elem])
		return skip_bytes(r, count * TYPE_SIZE[elem]);
	for (uint64_t i = 0; i < count; i++)
		if (!skip_value(r, elem))
			return false;
	return true;
}

static bool is_signed_int(uint32_t type) {
	return type == 1 || type == 3 || type == 5 || type == 11;
}

static bool is_int(uint32_t type) {
	return type <= 5 || type == 10 || type == 11;
}

static long long read_int(reader *r, uint32_t type) {
	uint64_t raw = read_uint(r, TYPE_SIZE[type]);
	if (!is_signed_int(type))
		return (long long)raw;
	int shift = (int)(64 - 8 * TYPE_SIZE[type]);
	return (long long)(raw << shift) >> shift;
}

static int_value read_int_value(reader *r, uint32_t type) {
	int_value v = {0};
	if (type < HM_TYPE_COUNT && is_int(type)) {
		v.ok	= true;
		v.sum	= read_int(r, type);
		v.count = 1;
		return v;
	}
	if (type != HM_TYPE_ARRAY)
		return v;
	uint32_t elem  = (uint32_t)read_uint(r, 4);
	uint64_t count = read_uint(r, 8);
	if (r->short_read || elem >= HM_TYPE_COUNT || !is_int(elem) || count > (1u << 20))
		return v;
	v.ok	= true;
	v.array = true;
	v.count = (long long)count;
	for (uint64_t i = 0; i < count && !r->short_read; i++)
		v.sum += read_int(r, elem);
	return v;
}

static bool read_key(reader *r, char *key, size_t cap) {
	uint64_t len = read_uint(r, 8);
	if (r->short_read || len >= HM_KEY_MAX)
		return false;
	if (!need(r, (size_t)len))
		return false;
	memcpy(key, r->p + r->pos, (size_t)len);
	key[len] = '\0';
	r->pos += (size_t)len;
	(void)cap;
	return true;
}

static bool read_string(reader *r, char *out, size_t cap) {
	uint64_t len = read_uint(r, 8);
	if (r->short_read || len >= cap || len >= HM_STRING_LIMIT || !need(r, (size_t)len))
		return false;
	memcpy(out, r->p + r->pos, (size_t)len);
	out[len] = '\0';
	r->pos += (size_t)len;
	return true;
}

static const slot_desc *find_slot(const hf_meta *m, const char *key) {
	size_t alen = strlen(m->arch);
	if (!alen || strncmp(key, m->arch, alen) != 0 || key[alen] != '.')
		return NULL;
	for (size_t i = 0; i < sizeof(SLOTS) / sizeof(SLOTS[0]); i++)
		if (strcmp(key + alen + 1, SLOTS[i].suffix) == 0)
			return &SLOTS[i];
	return NULL;
}

static void store_slot(hf_meta *m, const slot_desc *slot, int_value v) {
	long long *dst = (long long *)((char *)m + slot->offset);
	if (slot->offset == offsetof(hf_meta, head_kv) && v.array) {
		m->kv_sum = v.sum;
		*dst	  = v.count > 0 ? v.sum / v.count : -1;
		return;
	}
	*dst = v.array && v.count > 0 ? v.sum / v.count : v.sum;
}

static bool has_essentials(const hf_meta *m) {
	return m->arch[0] && m->n_layer > 0 && m->n_embd > 0 && m->n_head > 0;
}

static bool read_entry(reader *r, hf_meta *m, char *key) {
	if (!read_key(r, key, HM_KEY_MAX))
		return false;
	uint32_t type = (uint32_t)read_uint(r, 4);
	if (r->short_read)
		return false;
	if (strcmp(key, "general.architecture") == 0 && type == HM_TYPE_STRING)
		return read_string(r, m->arch, sizeof(m->arch));
	const slot_desc *slot = find_slot(m, key);
	if (!slot)
		return skip_value(r, type);
	int_value v = read_int_value(r, type);
	if (!v.ok)
		return false;
	store_slot(m, slot, v);
	return true;
}

static bool is_mtp_tensor(const hf_meta *m, const char *name) {
	if (strstr(name, ".nextn.") || strstr(name, "mtp"))
		return true;
	int layer;
	if (m->n_nextn <= 0 || m->n_layer <= 0 || sscanf(name, "blk.%d.", &layer) != 1)
		return false;
	return layer >= m->n_layer - m->n_nextn;
}

static void record_bad_type(hf_meta *m, unsigned type) {
	for (int i = 0; i < m->n_bad_types; i++)
		if (m->bad_types[i] == type)
			return;
	if (m->n_bad_types < HF_META_BAD_TYPES)
		m->bad_types[m->n_bad_types++] = type;
}

static void tally_tensor(hf_meta *m, const char *name, unsigned type, const uint64_t *dims,
						 uint32_t n_dims) {
	m->n_tensors++;
	if (!ggml_type_loadable(type)) {
		if (is_mtp_tensor(m, name)) {
			m->n_bad_mtp_tensors++;
			return;
		}
		m->n_bad_tensors++;
		record_bad_type(m, type);
		return;
	}
	if (type >= HF_META_TYPE_SLOTS)
		return;
	double bytes = (double)ggml_row_size(type, (size_t)dims[0]);
	for (uint32_t d = 1; d < n_dims; d++)
		bytes *= (double)dims[d];
	m->type_bytes[type] += bytes;
}

static bool read_tensor(reader *r, hf_meta *m, char *name) {
	if (!read_key(r, name, HM_KEY_MAX))
		return false;
	uint32_t n_dims = (uint32_t)read_uint(r, 4);
	if (r->short_read)
		return false;
	if (n_dims == 0 || n_dims > HM_MAX_DIMS) {
		r->bad = true;
		return false;
	}
	uint64_t dims[HM_MAX_DIMS];
	for (uint32_t i = 0; i < n_dims; i++)
		dims[i] = read_uint(r, 8);
	uint32_t type = (uint32_t)read_uint(r, 4);
	read_uint(r, 8);
	if (r->short_read)
		return false;
	tally_tensor(m, name, type, dims, n_dims);
	return true;
}

static int scan_failure(const reader *r) {
	return r->short_read ? 1 : -1;
}

int hf_meta_scan(const unsigned char *buf, size_t len, hf_meta *m) {
	reader r = {buf, len, 0, false, false};
	if (len < 24 || memcmp(buf, HM_MAGIC, 4) != 0)
		return -1;
	r.pos			 = 4;
	uint32_t version = (uint32_t)read_uint(&r, 4);
	if (version < 2 || version > 3)
		return -1;
	uint64_t n_tensors = read_uint(&r, 8);
	uint64_t n_kv	   = read_uint(&r, 8);
	if (n_tensors > HM_MAX_ENTRIES || n_kv > HM_MAX_ENTRIES)
		return -1;

	char key[HM_KEY_MAX];
	for (uint64_t i = 0; i < n_kv; i++) {
		if (!read_entry(&r, m, key)) {
			m->kv_ok = has_essentials(m);
			return scan_failure(&r);
		}
	}
	m->kv_ok = has_essentials(m);
	for (uint64_t i = 0; i < n_tensors; i++)
		if (!read_tensor(&r, m, key))
			return scan_failure(&r);
	m->tensors_ok = true;
	return m->kv_ok ? 0 : -1;
}

double hf_meta_kv_per_token(const hf_meta *m, double elem_bytes) {
	if (m->n_layer <= 0 || m->n_head <= 0)
		return -1.0;
	double kv_heads = m->kv_sum > 0
						  ? (double)m->kv_sum
						  : (double)m->n_layer * (double)(m->head_kv > 0 ? m->head_kv : m->n_head);
	double k		= m->key_len > 0 ? (double)m->key_len : (double)m->n_embd / (double)m->n_head;
	double v		= m->val_len > 0 ? (double)m->val_len : k;
	return kv_heads * (k + v) * elem_bytes;
}

bool hf_meta_blocked(const hf_meta *m) {
	return m->tensors_ok && m->n_bad_tensors > 0;
}

static const char *type_name(unsigned id, char *buf, size_t cap) {
	for (size_t i = 0; i < sizeof(TYPE_NAMES) / sizeof(TYPE_NAMES[0]); i++)
		if (TYPE_NAMES[i].id == id)
			return TYPE_NAMES[i].name;
	snprintf(buf, cap, "type %u", id);
	return buf;
}

void hf_meta_bad_list(const hf_meta *m, char *out, size_t cap) {
	size_t used = 0;
	out[0]		= '\0';
	for (int i = 0; i < m->n_bad_types && used + 1 < cap; i++) {
		char		tmp[16];
		const char *name = type_name(m->bad_types[i], tmp, sizeof(tmp));
		used += (size_t)snprintf(out + used, cap - used, "%s%s", i ? ", " : "", name);
	}
}

void hf_meta_mix(const hf_meta *m, char *out, size_t cap) {
	double total = 0;
	for (int t = 0; t < HF_META_TYPE_SLOTS; t++)
		total += m->type_bytes[t];
	out[0] = '\0';
	if (total <= 0)
		return;
	bool   taken[HF_META_TYPE_SLOTS] = {false};
	size_t used						 = 0;
	for (int shown = 0; shown < MIX_SHOWN; shown++) {
		int best = -1;
		for (int t = 0; t < HF_META_TYPE_SLOTS; t++)
			if (!taken[t] && m->type_bytes[t] > 0 &&
				(best < 0 || m->type_bytes[t] > m->type_bytes[best]))
				best = t;
		if (best < 0 || used + 1 >= cap)
			return;
		taken[best] = true;
		char tmp[16];
		used += (size_t)snprintf(out + used, cap - used, "%s%s %.0f%%", shown ? "  " : "",
								 type_name((unsigned)best, tmp, sizeof(tmp)),
								 m->type_bytes[best] * 100.0 / total);
	}
}
