#include "tokenizer.h"
#include "log.h"
#include <ctype.h>

#define TOK_DECODE_STACK_CAP 256

typedef struct {
	const char *p;
	size_t		n;
	int32_t		id;
	int			locked;
} piece;

static const int g_byte_to_cp[256] = {
	256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274,
	275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 33,  34,	35,	 36,  37,
	38,	 39,  40,  41,	42,	 43,  44,  45,	46,	 47,  48,  49,	50,	 51,  52,  53,	54,	 55,  56,
	57,	 58,  59,  60,	61,	 62,  63,  64,	65,	 66,  67,  68,	69,	 70,  71,  72,	73,	 74,  75,
	76,	 77,  78,  79,	80,	 81,  82,  83,	84,	 85,  86,  87,	88,	 89,  90,  91,	92,	 93,  94,
	95,	 96,  97,  98,	99,	 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
	114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 289, 290, 291, 292, 293, 294,
	295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313,
	314, 315, 316, 317, 318, 319, 320, 321, 322, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
	171, 172, 323, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
	190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
	209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
	228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
	247, 248, 249, 250, 251, 252, 253, 254, 255,
};

static const int g_cp_to_byte[512] = {
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  33,  34,	35,	 36,  37,
	38,	 39,  40,  41,	42,	 43,  44,  45,	46,	 47,  48,  49,	50,	 51,  52,  53,	54,	 55,  56,
	57,	 58,  59,  60,	61,	 62,  63,  64,	65,	 66,  67,  68,	69,	 70,  71,  72,	73,	 74,  75,
	76,	 77,  78,  79,	80,	 81,  82,  83,	84,	 85,  86,  87,	88,	 89,  90,  91,	92,	 93,  94,
	95,	 96,  97,  98,	99,	 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
	114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
	171, 172, -1,  174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
	190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
	209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
	228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
	247, 248, 249, 250, 251, 252, 253, 254, 255, 0,	  1,   2,	3,	 4,	  5,   6,	7,	 8,	  9,
	10,	 11,  12,  13,	14,	 15,  16,  17,	18,	 19,  20,  21,	22,	 23,  24,  25,	26,	 27,  28,
	29,	 30,  31,  32,	127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
	142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
	173, -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,
	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,  -1,  -1,	-1,	 -1,
};

static int cp_to_utf8(int cp, char *out) {
	if (cp < 0x80) {
		out[0] = (char)cp;
		return 1;
	}
	if (cp < 0x800) {
		out[0] = (char)(0xC0 | (cp >> 6));
		out[1] = (char)(0x80 | (cp & 0x3F));
		return 2;
	}
	if (cp < 0x10000) {
		out[0] = (char)(0xE0 | (cp >> 12));
		out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
		out[2] = (char)(0x80 | (cp & 0x3F));
		return 3;
	}
	out[0] = (char)(0xF0 | (cp >> 18));
	out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
	out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
	out[3] = (char)(0x80 | (cp & 0x3F));
	return 4;
}

static int utf8_to_cp(const char *s, int len, int *cp) {
	if (len <= 0) {
		*cp = 0;
		return 0;
	}
	unsigned char c = (unsigned char)s[0];
	if (c < 0x80) {
		*cp = c;
		return 1;
	}
	if ((c & 0xE0) == 0xC0 && len >= 2) {
		*cp = ((c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
		return 2;
	}
	if ((c & 0xF0) == 0xE0 && len >= 3) {
		*cp =
			((c & 0x0F) << 12) | (((unsigned char)s[1] & 0x3F) << 6) | ((unsigned char)s[2] & 0x3F);
		return 3;
	}
	if ((c & 0xF8) == 0xF0 && len >= 4) {
		*cp = ((c & 0x07) << 18) | (((unsigned char)s[1] & 0x3F) << 12) |
			  (((unsigned char)s[2] & 0x3F) << 6) | ((unsigned char)s[3] & 0x3F);
		return 4;
	}
	*cp = c;
	return 1;
}

static char *gpt2_encode_bytes(const char *in, size_t in_len, size_t *out_len) {
	char  *out = xmalloc((in_len * 2) + 1);
	size_t n   = 0;
	for (size_t i = 0; i < in_len; i++) {
		char buf[4];
		int	 k = cp_to_utf8(g_byte_to_cp[(unsigned char)in[i]], buf);
		memcpy(out + n, buf, k);
		n += k;
	}
	out[n]	 = '\0';
	*out_len = n;
	return out;
}

static size_t gpt2_decode_to_bytes_buf(const char *in, const unsigned char *byte_mark,
									   size_t in_len, char *out) {
	const int *cp_to_byte = g_cp_to_byte;
	size_t	   n		  = 0;
	size_t	   i		  = 0;
	while (i < in_len) {
		if (byte_mark && byte_mark[i]) {
			out[n++] = in[i];
			i++;
			continue;
		}
		int cp;
		int k = utf8_to_cp(in + i, (int)(in_len - i), &cp);
		if (k <= 0)
			break;
		int b = (cp < 512) ? cp_to_byte[cp] : -1;
		if (b < 0) {
			out[n++] = in[i];
		} else {
			out[n++] = (char)b;
		}
		i += k;
	}
	out[n] = '\0';
	return n;
}

static int is_utf8_letter(unsigned char c) {
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= 0x80);
}

static int is_utf8_digit(unsigned char c) {
	return c >= '0' && c <= '9';
}

static int cp_is_letter(int cp) {
	if (cp < 0x80)
		return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
	if (cp >= 0x2000 && cp <= 0x2BFF)
		return 0; /* general punctuation, symbols, arrows, dingbats, misc technical */
	if (cp >= 0x1F000 && cp <= 0x1FFFF)
		return 0; /* emoji & pictograph astral blocks */
	if ((cp >= 0x80 && cp <= 0xA9) || (cp >= 0xAB && cp <= 0xB4) || (cp >= 0xB6 && cp <= 0xB9) ||
		(cp >= 0xBB && cp <= 0xBF) || cp == 0xD7 || cp == 0xF7)
		return 0; /* Latin-1 punctuation/symbols */
	return 1;
}

static int cp_is_digit(int cp) {
	return cp >= '0' && cp <= '9';
}

static int is_space_char(unsigned char c) {
	return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

static int is_newline(unsigned char c) {
	return c == '\n' || c == '\r';
}

static int is_apostrophe(unsigned char c) {
	return c == '\'';
}

static int hex_nibble(unsigned char c) {
	if (c >= '0' && c <= '9')
		return c - '0';
	if (c >= 'a' && c <= 'f')
		return c - 'a' + 10;
	if (c >= 'A' && c <= 'F')
		return c - 'A' + 10;
	return -1;
}

static int byte_token_value(const vocab_token *tok) {
	if (tok->text_len != 6 || memcmp(tok->text, "<0x", 3) != 0 || tok->text[5] != '>')
		return -1;
	int hi = hex_nibble((unsigned char)tok->text[3]);
	int lo = hex_nibble((unsigned char)tok->text[4]);
	if (hi < 0 || lo < 0)
		return -1;
	return (hi << 4) | lo;
}

static void hash_insert(tok_hash_entry *ht, size_t cap, const char *key, size_t klen, int32_t id) {
	uint64_t h = fnv1a(key, klen) & (cap - 1);
	while (ht[h].used) {
		if (ht[h].key_len == klen && memcmp(ht[h].key, key, klen) == 0) {
			return;
		}
		h = (h + 1) & (cap - 1);
	}
	ht[h].key	  = key;
	ht[h].key_len = klen;
	ht[h].id	  = id;
	ht[h].used	  = 1;
}

static int32_t hash_lookup(const tok_hash_entry *ht, size_t cap, const char *key, size_t klen) {
	uint64_t h = fnv1a(key, klen) & (cap - 1);
	while (ht[h].used) {
		if (ht[h].key_len == klen && memcmp(ht[h].key, key, klen) == 0) {
			return ht[h].id;
		}
		h = (h + 1) & (cap - 1);
	}
	return -1;
}

static size_t next_pretoken(const char *s, size_t len, size_t *pos) {
	size_t i = *pos;
	if (i >= len)
		return 0;

	unsigned char c = (unsigned char)s[i];

	if (is_apostrophe(c)) {
		if (i + 1 < len) {
			unsigned char c1 = (unsigned char)s[i + 1];
			if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
				*pos = i + 2;
				return 2;
			}
			if (i + 2 < len) {
				unsigned char c2 = (unsigned char)s[i + 2];
				if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
					(c1 == 'l' && c2 == 'l')) {
					*pos = i + 3;
					return 3;
				}
			}
		}
	}

	if (c == ' ' && i + 1 < len && is_utf8_digit((unsigned char)s[i + 1])) {
		size_t j = i + 1;
		while (j < len && is_utf8_digit((unsigned char)s[j]))
			j++;
		*pos = j;
		return j - i;
	}

	{
		size_t start = i;
		size_t j	 = i;
		if (j < len) {
			unsigned char ch = (unsigned char)s[j];
			if (!is_newline(ch) && !is_utf8_letter(ch) && !is_utf8_digit(ch)) {
				j++;
			}
		}
		if (j < len && is_utf8_letter((unsigned char)s[j])) {
			while (j < len && is_utf8_letter((unsigned char)s[j]))
				j++;
			*pos = j;
			return j - start;
		}
	}

	if (is_utf8_digit(c)) {
		size_t j = i;
		while (j < len && is_utf8_digit((unsigned char)s[j]))
			j++;
		*pos = j;
		return j - i;
	}

	{
		size_t start = i;
		size_t j	 = i;
		if (j < len && s[j] == ' ')
			j++;
		if (j < len && !is_space_char((unsigned char)s[j]) &&
			!is_utf8_letter((unsigned char)s[j]) && !is_utf8_digit((unsigned char)s[j])) {
			while (j < len && !is_space_char((unsigned char)s[j]) &&
				   !is_utf8_letter((unsigned char)s[j]) && !is_utf8_digit((unsigned char)s[j]))
				j++;
			while (j < len && is_newline((unsigned char)s[j]))
				j++;
			*pos = j;
			return j - start;
		}
	}

	{
		size_t start = i;
		size_t j	 = i;
		while (j < len && (s[j] == ' ' || s[j] == '\t'))
			j++;
		if (j < len && is_newline((unsigned char)s[j])) {
			while (j < len && is_newline((unsigned char)s[j]))
				j++;
			*pos = j;
			return j - start;
		}
	}

	if (is_space_char(c)) {
		size_t j = i;
		while (j < len && is_space_char((unsigned char)s[j]))
			j++;
		if (j < len && !is_space_char((unsigned char)s[j])) {
			if (j > i + 1) {
				j--;
			}
		}
		*pos = j;
		return j - i;
	}

	{
		size_t j = i + 1;
		*pos	 = j;
		return 1;
	}
}

static int contains_newline_before_non_newline_run(const char *s, size_t len, size_t i) {
	while (i < len && is_space_char((unsigned char)s[i])) {
		if (is_newline((unsigned char)s[i]))
			return 1;
		i++;
	}
	return 0;
}

static int cp_at(const char *s, size_t len, size_t j, int *cp_len) {
	int cp;
	int k	= utf8_to_cp(s + j, (int)(len - j), &cp);
	*cp_len = k > 0 ? k : 1;
	return k > 0 ? cp : (unsigned char)s[j];
}

static size_t next_pretoken_unicode(const char *s, size_t len, size_t *pos, int digit_run) {
	size_t i = *pos;
	if (i >= len)
		return 0;

	unsigned char c = (unsigned char)s[i];

	if (is_apostrophe(c) && i + 1 < len) {
		unsigned char c1 = (unsigned char)tolower((unsigned char)s[i + 1]);
		if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
			*pos = i + 2;
			return 2;
		}
		if (i + 2 < len) {
			unsigned char c2 = (unsigned char)tolower((unsigned char)s[i + 2]);
			if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
				*pos = i + 3;
				return 3;
			}
		}
	}

	{
		size_t start   = i;
		size_t j	   = i;
		int	   cp_len0 = 1;
		int	   cp0	   = cp_at(s, len, j, &cp_len0);
		if (!is_newline(c) && !cp_is_letter(cp0) && !cp_is_digit(cp0))
			j = i + (size_t)cp_len0;
		int j_cp_len = 1;
		if (j < len && cp_is_letter(cp_at(s, len, j, &j_cp_len))) {
			while (j < len) {
				int cl;
				int cp = cp_at(s, len, j, &cl);
				if (!cp_is_letter(cp))
					break;
				j += (size_t)cl;
			}
			*pos = j;
			return j - start;
		}
	}

	if (is_utf8_digit(c)) {
		size_t j = i;
		int	   n = 0;
		while (j < len && is_utf8_digit((unsigned char)s[j]) && n < digit_run) {
			j++;
			n++;
		}
		*pos = j;
		return j - i;
	}

	{
		size_t start = i;
		size_t j	 = i;
		if (j < len && s[j] == ' ')
			j++;
		if (j < len) {
			int cl;
			int cp0 = cp_at(s, len, j, &cl);
			if (!is_space_char((unsigned char)s[j]) && !cp_is_letter(cp0) && !cp_is_digit(cp0)) {
				while (j < len) {
					int			  cl2;
					int			  cpn = cp_at(s, len, j, &cl2);
					unsigned char cj  = (unsigned char)s[j];
					if (is_space_char(cj) || cp_is_letter(cpn) || cp_is_digit(cpn))
						break;
					j += (size_t)cl2;
				}
				while (j < len && is_newline((unsigned char)s[j]))
					j++;
				*pos = j;
				return j - start;
			}
		}
	}

	if (is_space_char(c)) {
		size_t start = i;

		if (contains_newline_before_non_newline_run(s, len, i)) {
			size_t j		   = i;
			size_t last_nl_end = i;
			while (j < len && is_space_char((unsigned char)s[j])) {
				if (is_newline((unsigned char)s[j])) {
					j++;
					last_nl_end = j;
				} else {
					size_t k = j;
					while (k < len && is_space_char((unsigned char)s[k]) &&
						   !is_newline((unsigned char)s[k]))
						k++;
					if (k < len && is_newline((unsigned char)s[k])) {
						j = k;
					} else {
						break;
					}
				}
			}
			*pos = last_nl_end;
			return last_nl_end - start;
		}

		{
			size_t j = i;
			while (j < len && is_space_char((unsigned char)s[j]))
				j++;
			if (j < len && j > i + 1) {
				j--;
			}
			*pos = j;
			return j - start;
		}
	}

	*pos = i + 1;
	return 1;
}

typedef struct {
	int32_t	 rank;
	int32_t	 node;
	uint32_t ver;
} bpe_heap_entry;

typedef struct {
	piece	 *pcs;
	int		  npcs;
	int		  head;
	int		 *prev;
	int		 *next;
	uint8_t	 *alive;
	uint32_t *ver;

	int32_t	 *hrank;
	int32_t	 *hnode;
	uint32_t *hver;
	int		  hn;

	char  *key;
	size_t key_cap;
} bpe_state;

static int32_t bpe_pair_rank(tokenizer *t, bpe_state *bs, int i) {
	const piece *a = &bs->pcs[i];
	const piece *b = &bs->pcs[bs->next[i]];
	if (a->locked || b->locked)
		return -1;
	size_t klen = a->n + b->n;
	if (klen > t->bpe_rank_cap) {
		size_t cap = t->bpe_rank_cap > 0 ? t->bpe_rank_cap : 128;
		while (cap < klen)
			cap *= 2;
		free(t->bpe_rank_buf);
		t->bpe_rank_buf = xmalloc(cap);
		t->bpe_rank_cap = cap;
	}
	bs->key		= t->bpe_rank_buf;
	bs->key_cap = t->bpe_rank_cap;
	memcpy(bs->key, a->p, a->n);
	memcpy(bs->key + a->n, b->p, b->n);
	if (t->has_merges)
		return hash_lookup((const tok_hash_entry *)t->merge_hash, t->merge_hash_capacity, bs->key,
						   klen);
	return hash_lookup(t->hash, t->hash_capacity, bs->key, klen);
}

static void bpe_heap_swap(bpe_state *bs, int i, int j) {
	int32_t tr	 = bs->hrank[i];
	bs->hrank[i] = bs->hrank[j];
	bs->hrank[j] = tr;
	int32_t tn	 = bs->hnode[i];
	bs->hnode[i] = bs->hnode[j];
	bs->hnode[j] = tn;
	uint32_t tv	 = bs->hver[i];
	bs->hver[i]	 = bs->hver[j];
	bs->hver[j]	 = tv;
}

static void bpe_heap_push(bpe_state *bs, int32_t rank, int node, uint32_t ver) {
	int i		 = ++bs->hn;
	bs->hrank[i] = rank;
	bs->hnode[i] = node;
	bs->hver[i]	 = ver;
	while (i > 1) {
		int par = i / 2;
		if (bs->hrank[par] < bs->hrank[i] ||
			(bs->hrank[par] == bs->hrank[i] && bs->hnode[par] <= bs->hnode[i]))
			break;
		bpe_heap_swap(bs, i, par);
		i = par;
	}
}

static void bpe_heap_pop(bpe_state *bs, int32_t *rank, int *node, uint32_t *ver) {
	*rank = bs->hrank[1];
	*node = bs->hnode[1];
	*ver  = bs->hver[1];
	int n = bs->hn--;
	if (n <= 1)
		return;
	bpe_heap_swap(bs, 1, n);
	int i = 1;
	for (;;) {
		int l = 2 * i, r = l + 1, best = i;
		if (l <= bs->hn && (bs->hrank[l] < bs->hrank[best] ||
							(bs->hrank[l] == bs->hrank[best] && bs->hnode[l] < bs->hnode[best])))
			best = l;
		if (r <= bs->hn && (bs->hrank[r] < bs->hrank[best] ||
							(bs->hrank[r] == bs->hrank[best] && bs->hnode[r] < bs->hnode[best])))
			best = r;
		if (best == i)
			return;
		bpe_heap_swap(bs, i, best);
		i = best;
	}
}

static int bpe_work_reserve(tokenizer *t, size_t need) {
	if (need <= t->bpe_work_cap)
		return 0;
	free(t->bpe_work);
	t->bpe_work		= xmalloc(need);
	t->bpe_work_cap = need;
	return 0;
}

int tokenizer_bpe_encode(tokenizer *t, const char *text, size_t len, int32_t *out_ids, int max_out,
						 int *n_out) {
	if (len == 0) {
		*n_out = 0;
		return 0;
	}
	if (t->bpe_pcs_cache_cap < len) {
		free(t->bpe_pcs_cache);
		t->bpe_pcs_cache	 = xmalloc(len * sizeof(piece));
		t->bpe_pcs_cache_cap = len;
	}
	piece *pcs	= t->bpe_pcs_cache;
	int	   npcs = 0;

	size_t char_idx = 0;
	while (char_idx < len) {
		unsigned char c		   = (unsigned char)text[char_idx];
		size_t		  char_len = 1;
		if (c >= 0xF0)
			char_len = 4;
		else if (c >= 0xE0)
			char_len = 3;
		else if (c >= 0xC0)
			char_len = 2;

		if (char_idx + char_len > len)
			char_len = 1;

		int32_t id = hash_lookup(t->hash, t->hash_capacity, text + char_idx, char_len);
		if (id < 0 && t->n_byte_fallback > 0) {
			size_t off = 0;
			while (off < char_len) {
				int			  cp;
				int			  k;
				unsigned char raw;
				if (t->is_sentencepiece) {
					k	= 1;
					cp	= -1;
					raw = (unsigned char)text[char_idx + off];
				} else {
					k = utf8_to_cp(text + char_idx + off, (int)(char_len - off), &cp);
					if (k <= 0) {
						k  = 1;
						cp = -1;
					}
					raw = (unsigned char)text[char_idx + off];
					if (cp >= 0 && cp < 512 && g_cp_to_byte[cp] >= 0)
						raw = (unsigned char)g_cp_to_byte[cp];
				}
				int32_t fid		 = t->byte_fallback_ids[raw];
				pcs[npcs].p		 = text + char_idx + off;
				pcs[npcs].n		 = (size_t)k;
				pcs[npcs].id	 = (fid >= 0) ? fid : ((t->unk_id >= 0) ? t->unk_id : 0);
				pcs[npcs].locked = (fid >= 0);
				npcs++;
				off += (size_t)k;
			}
			char_idx += char_len;
			continue;
		}
		if (id < 0)
			id = (t->unk_id >= 0) ? t->unk_id : 0;
		pcs[npcs].p		 = text + char_idx;
		pcs[npcs].n		 = char_len;
		pcs[npcs].id	 = id;
		pcs[npcs].locked = 0;
		npcs++;
		char_idx += char_len;
	}

	size_t heap_cap	 = (size_t)(3 * npcs + 8);
	size_t work_need = ((size_t)npcs * sizeof(int)) * 2 +
					   ((size_t)npcs * (sizeof(uint8_t) + sizeof(uint32_t))) +
					   (heap_cap * (sizeof(int32_t) + sizeof(int) + sizeof(uint32_t))) + 8 * 8;
	bpe_work_reserve(t, work_need);

	bpe_state bs;
	memset(&bs, 0, sizeof(bs));
	{
		uintptr_t cursor = (uintptr_t)t->bpe_work;
#define BPE_TAKE(ptr, type, count)                                                                 \
	do {                                                                                           \
		cursor = (cursor + (_Alignof(type) - 1)) & ~(uintptr_t)(_Alignof(type) - 1);               \
		(ptr)  = (type *)cursor;                                                                   \
		cursor += (size_t)(count) * sizeof(type);                                                  \
	} while (0)
		BPE_TAKE(bs.prev, int, npcs);
		BPE_TAKE(bs.next, int, npcs);
		BPE_TAKE(bs.alive, uint8_t, npcs);
		BPE_TAKE(bs.ver, uint32_t, npcs);
		BPE_TAKE(bs.hrank, int32_t, heap_cap);
		BPE_TAKE(bs.hnode, int, heap_cap);
		BPE_TAKE(bs.hver, uint32_t, heap_cap);
#undef BPE_TAKE
		bs.pcs	= pcs;
		bs.npcs = npcs;
		bs.head = 0;
		bs.key	= t->bpe_rank_buf;
	}

	for (int i = 0; i < npcs; i++) {
		bs.prev[i]	= i - 1;
		bs.next[i]	= (i + 1 < npcs) ? i + 1 : -1;
		bs.alive[i] = 1;
		bs.ver[i]	= 0;
	}
	for (int i = 0; i + 1 < npcs; i++) {
		int32_t r = bpe_pair_rank(t, &bs, i);
		if (r >= 0)
			bpe_heap_push(&bs, r, i, 0);
	}

	size_t arena_cap = t->bpe_arena_cap;
	if (arena_cap < len) {
		size_t cap = arena_cap > 0 ? arena_cap : 64;
		while (cap < len)
			cap *= 2;
		free(t->bpe_arena);
		t->bpe_arena	 = xmalloc(cap);
		t->bpe_arena_cap = cap;
	}
	char  *arena	  = t->bpe_arena;
	size_t arena_used = 0;

	while (bs.hn > 0) {
		int32_t	 top_rank;
		int		 nd;
		uint32_t top_ver;
		bpe_heap_pop(&bs, &top_rank, &nd, &top_ver);
		if (!bs.alive[nd] || bs.ver[nd] != top_ver || bs.next[nd] < 0)
			continue;
		int32_t cur = bpe_pair_rank(t, &bs, nd);
		if (cur != top_rank) {
			if (cur >= 0)
				bpe_heap_push(&bs, cur, nd, bs.ver[nd]);
			continue;
		}
		int nx		  = bs.next[nd];
		int best_klen = (int)(bs.pcs[nd].n + bs.pcs[nx].n);
		if (arena_used + (size_t)best_klen > arena_cap) {
			uintptr_t old_base = (uintptr_t)arena;
			size_t	  old_used = arena_used;
			size_t	  new_cap  = t->bpe_arena_cap * 2;
			while (arena_used + (size_t)best_klen > new_cap)
				new_cap *= 2;
			char *new_arena	 = xrealloc(arena, new_cap);
			t->bpe_arena	 = new_arena;
			t->bpe_arena_cap = new_cap;
			if ((uintptr_t)new_arena != old_base) {
				for (int i = bs.head; i >= 0; i = bs.next[i]) {
					uintptr_t p = (uintptr_t)pcs[i].p;
					if (p >= old_base && p < old_base + old_used)
						pcs[i].p = new_arena + (p - old_base);
				}
			}
			arena	  = new_arena;
			arena_cap = new_cap;
		}
		char *merged = arena + arena_used;
		arena_used += (size_t)best_klen;
		memcpy(merged, pcs[nd].p, pcs[nd].n);
		memcpy(merged + pcs[nd].n, pcs[nx].p, pcs[nx].n);
		pcs[nd].p				= merged;
		pcs[nd].n				= (size_t)best_klen;
		int32_t merged_vocab_id = hash_lookup(t->hash, t->hash_capacity, merged, (size_t)best_klen);
		pcs[nd].id = (merged_vocab_id >= 0) ? merged_vocab_id : ((t->unk_id >= 0) ? t->unk_id : 0);

		bs.alive[nx] = 0;
		bs.ver[nx]++;
		bs.ver[nd]++;
		bs.next[nd] = bs.next[nx];
		if (bs.next[nx] >= 0)
			bs.prev[bs.next[nx]] = nd;

		if (bs.prev[nd] >= 0) {
			int32_t r = bpe_pair_rank(t, &bs, bs.prev[nd]);
			if (r >= 0)
				bpe_heap_push(&bs, r, bs.prev[nd], bs.ver[bs.prev[nd]]);
		}
		if (bs.next[nd] >= 0) {
			int32_t r = bpe_pair_rank(t, &bs, nd);
			if (r >= 0)
				bpe_heap_push(&bs, r, nd, bs.ver[nd]);
		}
	}

	int written = 0;
	for (int i = bs.head; i >= 0; i = bs.next[i]) {
		if (written >= max_out)
			goto fail;
		out_ids[written++] = pcs[i].id;
	}
	*n_out = written;
	return 0;

fail:
	return -1;
}

static int encode_sp_chunk(tokenizer *t, const char *text, size_t start, size_t end,
						   int32_t *out_ids, int max_out, int *written) {
	size_t sub_len = end - start;
	if (t->bpe_sp_cap < (sub_len + 1) * 3 + 1) {
		size_t cap = t->bpe_sp_cap > 0 ? t->bpe_sp_cap : 256;
		while (cap < (sub_len + 1) * 3 + 1)
			cap *= 2;
		free(t->bpe_sp_text);
		t->bpe_sp_text = xmalloc(cap);
		t->bpe_sp_cap  = cap;
	}
	char  *sp_text = t->bpe_sp_text;
	size_t sp_len  = 0;
	if (t->add_space_prefix) {
		sp_text[sp_len++] = '\xe2';
		sp_text[sp_len++] = '\x96';
		sp_text[sp_len++] = '\x81';
	}
	for (size_t i = start; i < end; i++) {
		if (text[i] == ' ') {
			sp_text[sp_len++] = '\xe2';
			sp_text[sp_len++] = '\x96';
			sp_text[sp_len++] = '\x81';
		} else {
			sp_text[sp_len++] = text[i];
		}
	}
	sp_text[sp_len] = '\0';
	int n;
	if (tokenizer_bpe_encode(t, sp_text, sp_len, out_ids + *written, max_out - *written, &n) < 0)
		return -1;
	*written += n;
	return 0;
}

static int encode_gpt2_chunk(tokenizer *t, const char *text, size_t start, size_t end,
							 int32_t *out_ids, int max_out, int *written) {
	size_t sub_pos = start;
	while (sub_pos < end) {
		size_t pstart = sub_pos;
		size_t plen;
		if (t->pre_type == TOK_PRE_LLAMA3)
			plen = next_pretoken_unicode(text, end, &sub_pos, 3);
		else if (t->pre_type == TOK_PRE_QWEN35)
			plen = next_pretoken_unicode(text, end, &sub_pos, 1);
		else
			plen = next_pretoken(text, end, &sub_pos);
		if (plen == 0)
			break;
		size_t enc_len;
		char  *enc = gpt2_encode_bytes(text + pstart, plen, &enc_len);
		int	   n;
		if (tokenizer_bpe_encode(t, enc, enc_len, out_ids + *written, max_out - *written, &n) < 0) {
			free(enc);
			return -1;
		}
		free(enc);
		*written += n;
	}
	return 0;
}

static int encode_text_chunk(tokenizer *t, const char *text, size_t start, size_t end,
							 int32_t *out_ids, int max_out, int *written) {
	if (t->is_sentencepiece)
		return encode_sp_chunk(t, text, start, end, out_ids, max_out, written);
	return encode_gpt2_chunk(t, text, start, end, out_ids, max_out, written);
}

static int emit_special_token(int32_t *out_ids, int max_out, int *written, int32_t id) {
	if (*written >= max_out)
		return -1;
	out_ids[(*written)++] = id;
	return 0;
}

static int32_t find_next_special(const tokenizer *t, const char *text, size_t len, size_t from,
								 size_t *out_at) {
	const unsigned char *bitmap = t->special_first_byte_bitmap;
	size_t				 p		= from;
	while (p < len) {
		unsigned char c = (unsigned char)text[p];
		if ((bitmap[c >> 3] & (unsigned char)(1u << (c & 7))) == 0) {
			size_t next = len;
			for (size_t i = 0; i < t->n_special_first_bytes; i++) {
				const char *hit = memchr(text + p, t->special_first_bytes[i], len - p);
				if (hit) {
					size_t at = (size_t)(hit - text);
					if (at < next)
						next = at;
				}
			}
			if (next >= len)
				return -1;
			p = next;
			continue;
		}
		size_t	b0		= t->special_by_first_byte_off[c];
		size_t	b1		= t->special_by_first_byte_off[(size_t)c + 1];
		int32_t best_id = -1;
		size_t	best_n	= 0;
		for (size_t bi = b0; bi < b1; bi++) {
			int32_t sid	 = t->special_by_first_byte[bi];
			size_t	nlen = t->tokens[sid].text_len;
			if (nlen <= best_n)
				continue;
			if (p + nlen > len)
				continue;
			if (memcmp(text + p, t->tokens[sid].text, nlen) == 0) {
				best_id = sid;
				best_n	= nlen;
			}
		}
		if (best_id >= 0) {
			*out_at = p;
			return best_id;
		}
		p++;
	}
	return -1;
}

static bool g_warned_no_merges;

status_code tokenizer_init(tokenizer *t, const gguf_ctx *g) {
	memset(t, 0, sizeof(*t));

	const char *model_name = NULL;
	if (gguf_get_str(g, "tokenizer.ggml.model", &model_name) != OK) {
		ERROR("tokenizer: missing 'tokenizer.ggml.model'");
		return ERR_FORMAT;
	}

	const char *const *toks;
	size_t			   n_toks;
	if (gguf_get_arr_str(g, "tokenizer.ggml.tokens", &toks, &n_toks) != OK) {
		ERROR("tokenizer: missing 'tokenizer.ggml.tokens'");
		return ERR_FORMAT;
	}

	const float *scores	  = NULL;
	size_t		 n_scores = 0;
	gguf_get_arr_f32(g, "tokenizer.ggml.scores", &scores, &n_scores);

	const int32_t *types;
	size_t		   n_types;
	if (gguf_get_arr_i32(g, "tokenizer.ggml.token_type", &types, &n_types) != OK) {
		ERROR("tokenizer: missing 'tokenizer.ggml.token_type'");
		return ERR_FORMAT;
	}

	if (n_toks != n_types) {
		ERROR("tokenizer: vocab array size mismatch");
		return ERR_FORMAT;
	}
	(void)n_scores;

	t->n_tokens = n_toks;
	t->tokens	= xcalloc(n_toks, sizeof(vocab_token));
	for (size_t i = 0; i < n_toks; i++) {
		t->tokens[i].id		  = (int32_t)i;
		t->tokens[i].text	  = toks[i];
		t->tokens[i].text_len = strlen(toks[i]);
		t->tokens[i].score	  = scores ? scores[i] : (float)i;
		t->tokens[i].type	  = types[i];
	}

	size_t cap = 1;
	while (cap < n_toks * 2)
		cap <<= 1;
	t->hash_capacity = cap;
	t->hash			 = xcalloc(cap, sizeof(tok_hash_entry));
	for (size_t i = 0; i < n_toks; i++) {
		hash_insert((tok_hash_entry *)t->hash, cap, t->tokens[i].text, t->tokens[i].text_len,
					(int32_t)i);
	}

	for (int bi = 0; bi < 256; bi++)
		t->byte_fallback_ids[bi] = -1;
	t->n_byte_fallback = 0;
	for (size_t i = 0; i < n_toks; i++) {
		if (t->tokens[i].type != TOK_TYPE_BYTE)
			continue;
		int bv = byte_token_value(&t->tokens[i]);
		if (bv >= 0 && t->byte_fallback_ids[bv] < 0) {
			t->byte_fallback_ids[bv] = (int32_t)i;
			t->n_byte_fallback++;
		}
	}
	if (t->n_byte_fallback > 0)
		DEBUG("tokenizer: %zu byte-fallback pieces registered", t->n_byte_fallback);

	const char *const *merges	= NULL;
	size_t			   n_merges = 0;
	if (gguf_get_arr_str(g, "tokenizer.ggml.merges", &merges, &n_merges) == OK && n_merges > 0) {
		size_t mcap = 1;
		while (mcap < n_merges * 2)
			mcap <<= 1;
		t->merge_hash_capacity = mcap;
		t->merge_hash		   = xcalloc(mcap, sizeof(tok_hash_entry));
		t->merge_keys		   = (char **)xcalloc(n_merges, sizeof(char *));
		t->n_merge_keys		   = 0;
		for (size_t i = 0; i < n_merges; i++) {
			const char *entry = merges[i];
			const char *sp	  = strchr(entry, ' ');
			if (!sp)
				continue;
			size_t left_len	 = (size_t)(sp - entry);
			size_t right_len = strlen(sp + 1);
			size_t klen		 = left_len + right_len;
			char  *key		 = xmalloc(klen + 1);
			memcpy(key, entry, left_len);
			memcpy(key + left_len, sp + 1, right_len);
			key[klen]						 = '\0';
			t->merge_keys[t->n_merge_keys++] = key;
			hash_insert((tok_hash_entry *)t->merge_hash, mcap, key, klen, (int32_t)i);
		}
		t->has_merges = 1;
		DEBUG("tokenizer: loaded %zu BPE merge rules", n_merges);
	} else {
		t->merge_hash		   = NULL;
		t->merge_hash_capacity = 0;
		t->has_merges		   = 0;
		if (!g_warned_no_merges) {
			g_warned_no_merges = true;
			WARN("tokenizer: no merge rules in GGUF, using vocab-ID heuristic instead");
		}
	}

	int32_t v;
	if (gguf_get_i32(g, "tokenizer.ggml.bos_token_id", &v) == OK)
		t->bos_id = v;
	else
		t->bos_id = -1;
	if (gguf_get_i32(g, "tokenizer.ggml.eos_token_id", &v) == OK)
		t->eos_id = v;
	else
		t->eos_id = -1;
	if (gguf_get_i32(g, "tokenizer.ggml.eot_token_id", &v) == OK) {
		t->eot_id = v;
	} else {
		static const char *eot_strs[] = {
			"<|eot_id|>",
			"<|im_end|>",
			"<|end|>",
			"<end_of_turn>",
			"<|endoftext|>",
			"<|end_of_text|>",
			"<EOT>",
			"_<EOT>",
			"[EOT]",
			"<\357\275\234end\342\226\201of\342\226\201sentence\357\275\234>",
			"<end_of_utterance>",
			"<turn|>",
		};
		t->eot_id = -1;
		for (size_t ei = 0; ei < ARRAY_LEN(eot_strs) && t->eot_id < 0; ei++) {
			int32_t id = hash_lookup((const tok_hash_entry *)t->hash, cap, eot_strs[ei],
									 strlen(eot_strs[ei]));
			if (id >= 0)
				t->eot_id = id;
		}
	}
	if (gguf_get_i32(g, "tokenizer.ggml.padding_token_id", &v) == OK)
		t->pad_id = v;
	else
		t->pad_id = -1;
	if (gguf_get_i32(g, "tokenizer.ggml.unknown_token_id", &v) == OK)
		t->unk_id = v;
	else
		t->unk_id = -1;

	int b;
	if (gguf_get_bool(g, "tokenizer.ggml.add_bos_token", &b) == OK)
		t->add_bos = b;

	t->is_sentencepiece = 0;
	if (model_name) {
		if (strstr(model_name, "gemma") || strstr(model_name, "spm") || strstr(model_name, "t5") ||
			strstr(model_name, "llama")) {
			t->is_sentencepiece = 1;
		}
	} else {
		t->add_bos = (t->bos_id >= 0);
	}

	t->add_space_prefix = 0;
	if (gguf_get_bool(g, "tokenizer.ggml.add_space_prefix", &b) == OK) {
		t->add_space_prefix = b;
	} else if (t->is_sentencepiece && model_name && strstr(model_name, "llama")) {
		t->add_space_prefix = 1;
	}

	t->pre_type			   = TOK_PRE_GPT2;
	const char *pre_str	   = NULL;
	status_code pre_status = gguf_get_str(g, "tokenizer.ggml.pre", &pre_str);
	if (pre_status != OK)
		pre_status = gguf_get_str(g, "tokenizer.ggml.pretokenizer", &pre_str);
	if (pre_status == OK && pre_str) {
		if (strstr(pre_str, "llama3") || strstr(pre_str, "llama-bpe")) {
			t->pre_type = TOK_PRE_LLAMA3;
		} else if (strstr(pre_str, "qwen35")) {
			t->pre_type = TOK_PRE_QWEN35;
		}
	} else if (!t->is_sentencepiece && model_name && (strstr(model_name, "gpt2") == NULL)) {
		WARN("tokenizer: pretokenizer unset, assuming GPT-2 regex (wrong for some models)");
	}
	if (gguf_get_bool(g, "tokenizer.ggml.add_eos_token", &b) == OK)
		t->add_eos = b;
	else
		t->add_eos = 0;

	t->n_special_ids = 0;
	t->special_ids	 = xmalloc(n_toks * sizeof(int32_t));
	for (size_t i = 0; i < n_toks; i++) {
		if (t->tokens[i].type == TOK_TYPE_CONTROL || t->tokens[i].type == TOK_TYPE_USER_DEFINED) {
			t->special_ids[t->n_special_ids++] = (int32_t)i;
		}
	}

	{
		size_t counts[256] = {0};
		for (size_t i = 0; i < t->n_special_ids; i++) {
			int32_t sid = t->special_ids[i];
			if (t->tokens[sid].text_len == 0)
				continue;
			unsigned char fb = (unsigned char)t->tokens[sid].text[0];
			counts[fb]++;
			t->special_first_byte_bitmap[fb >> 3] |= (unsigned char)(1u << (fb & 7));
		}
		for (int b2 = 0; b2 < 256; b2++) {
			if (counts[b2])
				t->special_first_bytes[t->n_special_first_bytes++] = (unsigned char)b2;
		}
		t->special_by_first_byte_off[0] = 0;
		for (int b2 = 0; b2 < 256; b2++) {
			t->special_by_first_byte_off[b2 + 1] = t->special_by_first_byte_off[b2] + counts[b2];
		}
		size_t total			 = t->special_by_first_byte_off[256];
		t->special_by_first_byte = total ? xmalloc(total * sizeof(int32_t)) : NULL;
		size_t cursor[256];
		for (int b2 = 0; b2 < 256; b2++)
			cursor[b2] = t->special_by_first_byte_off[b2];
		for (size_t i = 0; i < t->n_special_ids; i++) {
			int32_t sid = t->special_ids[i];
			if (t->tokens[sid].text_len == 0)
				continue;
			unsigned char fb					   = (unsigned char)t->tokens[sid].text[0];
			t->special_by_first_byte[cursor[fb]++] = sid;
		}
	}

	return OK;
}

void tokenizer_free(tokenizer *t) {
	free(t->tokens);
	free((void *)t->hash);
	free((void *)t->merge_hash);
	for (size_t i = 0; i < t->n_merge_keys; i++)
		free(t->merge_keys[i]);
	free((void *)t->merge_keys);
	free(t->special_ids);
	free(t->special_by_first_byte);
	free(t->bpe_pcs_cache);
	free(t->bpe_work);
	free(t->bpe_arena);
	free(t->bpe_rank_buf);
	free(t->bpe_sp_text);
	memset(t, 0, sizeof(*t));
}

int tokenizer_is_eog(const tokenizer *t, int32_t id) {
	if (id < 0)
		return 0;
	if (t->eos_id >= 0 && id == t->eos_id)
		return 1;
	if (t->eot_id >= 0 && id == t->eot_id)
		return 1;
	return 0;
}

size_t tokenizer_token_decoded_len(const tokenizer *t, int32_t id) {
	if (id < 0 || (size_t)id >= t->n_tokens)
		return 0;
	const vocab_token *tok = &t->tokens[id];
	if (tok->type == TOK_TYPE_BYTE)
		return (byte_token_value(tok) >= 0) ? 1 : tok->text_len;
	size_t decoded = 0;
	size_t i	   = 0;
	while (i < tok->text_len) {
		int cp;
		int k = utf8_to_cp(tok->text + i, (int)(tok->text_len - i), &cp);
		if (k <= 0)
			break;
		int b = (cp < 512) ? g_cp_to_byte[cp] : -1;
		decoded += (b >= 0) ? 1 : (size_t)k;
		i += (size_t)k;
	}
	return decoded;
}

int tokenizer_token_count_for_bytes(const tokenizer *t, const int32_t *ids, int n,
									size_t max_bytes) {
	char		   stack_acc[TOK_DECODE_STACK_CAP];
	unsigned char  stack_mark[TOK_DECODE_STACK_CAP];
	int			   stack_owner[TOK_DECODE_STACK_CAP];
	char		  *acc		= stack_acc;
	unsigned char *mark		= stack_mark;
	int			  *owner	= stack_owner;
	size_t		   acc_cap	= TOK_DECODE_STACK_CAP;
	size_t		   acc_len	= 0;
	int			   acc_heap = 0;

	for (int i = 0; i < n; i++) {
		int32_t id = ids[i];
		if (id < 0 || (size_t)id >= t->n_tokens)
			break;
		const vocab_token *tok = &t->tokens[id];
		int				   bv  = -1;
		size_t			   len = tok->text_len;
		if (tok->type == TOK_TYPE_BYTE) {
			bv	= byte_token_value(tok);
			len = (bv >= 0) ? 1 : tok->text_len;
		}
		if (acc_len + len > acc_cap) {
			size_t new_cap = acc_cap;
			while (acc_len + len > new_cap)
				new_cap <<= 1;
			char		  *new_acc	 = xmalloc(new_cap);
			unsigned char *new_mark	 = xmalloc(new_cap);
			int			  *new_owner = xmalloc(new_cap * sizeof(int));
			memcpy(new_acc, acc, acc_len);
			memcpy(new_mark, mark, acc_len);
			memcpy(new_owner, owner, acc_len * sizeof(int));
			if (acc_heap) {
				free(acc);
				free(mark);
				free(owner);
			}
			acc		 = new_acc;
			mark	 = new_mark;
			owner	 = new_owner;
			acc_cap	 = new_cap;
			acc_heap = 1;
		}
		if (bv >= 0) {
			acc[acc_len]  = (char)bv;
			mark[acc_len] = 1;
		} else {
			memcpy(acc + acc_len, tok->text, len);
			memset(mark + acc_len, 0, len);
		}
		for (size_t j = 0; j < len; j++)
			owner[acc_len + j] = i;
		acc_len += len;
	}

	int result = 0;

	if (t->is_sentencepiece) {
		size_t rd		 = 0;
		int	   cur_owner = -1;
		size_t out_bytes = 0;
		while (rd < acc_len) {
			size_t step;
			if (rd + 2 < acc_len && (unsigned char)acc[rd] == 0xE2 &&
				(unsigned char)acc[rd + 1] == 0x96 && (unsigned char)acc[rd + 2] == 0x81) {
				step = 3;
			} else {
				step = 1;
			}
			if (out_bytes + 1 > max_bytes)
				break;
			out_bytes += 1;
			int this_owner = owner[rd + step - 1];
			rd += step;
			if (rd >= acc_len || owner[rd] != this_owner)
				cur_owner = this_owner;
		}
		result = cur_owner + 1;
	} else {
		size_t out_bytes = 0;
		int	   cur_owner = -1;
		size_t i		 = 0;
		while (i < acc_len) {
			size_t step;
			if (mark[i]) {
				step = 1;
			} else {
				int cp;
				int k = utf8_to_cp(acc + i, (int)(acc_len - i), &cp);
				if (k <= 0)
					break;
				step = (size_t)k;
			}
			if (out_bytes + 1 > max_bytes)
				break;
			out_bytes += 1;
			int this_owner = owner[i + step - 1];
			i += step;
			if (i >= acc_len || owner[i] != this_owner)
				cur_owner = this_owner;
		}
		result = cur_owner + 1;
	}

	if (acc_heap) {
		free(acc);
		free(mark);
		free(owner);
	}
	return result;
}

int32_t tokenizer_find_token(const tokenizer *t, const char *text) {
	if (t->hash) {
		int32_t id =
			hash_lookup((const tok_hash_entry *)t->hash, t->hash_capacity, text, strlen(text));
		if (id >= 0 && (size_t)id < t->n_tokens)
			return id;
	}

	size_t needle_len = strlen(text);
	for (size_t i = 0; i < t->n_tokens; i++) {
		if (t->tokens[i].type != TOK_TYPE_CONTROL && t->tokens[i].type != TOK_TYPE_USER_DEFINED)
			continue;
		if (t->tokens[i].text_len == needle_len && memcmp(t->tokens[i].text, text, needle_len) == 0)
			return (int32_t)i;
	}
	return -1;
}

int tokenizer_encode_with_specials(tokenizer *t, const char *text, int add_specials,
								   int32_t *out_ids, int max_out, profile *prof) {
	profile_scope ps	  = profile_begin(prof, STAGE_TOKENIZE_ENCODE);
	int			  written = 0;

	if (add_specials && t->add_bos && t->bos_id >= 0 &&
		emit_special_token(out_ids, max_out, &written, t->bos_id) < 0) {
		profile_end(prof, &ps);
		return -1;
	}

	size_t len = strlen(text);
	size_t pos = 0;
	while (pos < len) {
		size_t	best_at;
		int32_t best_id = find_next_special(t, text, len, pos, &best_at);

		if (best_id < 0) {
			if (encode_text_chunk(t, text, pos, len, out_ids, max_out, &written) < 0) {
				profile_end(prof, &ps);
				return -1;
			}
			break;
		}

		if (best_at > pos) {
			if (encode_text_chunk(t, text, pos, best_at, out_ids, max_out, &written) < 0) {
				profile_end(prof, &ps);
				return -1;
			}
		}

		if (emit_special_token(out_ids, max_out, &written, best_id) < 0) {
			profile_end(prof, &ps);
			return -1;
		}
		pos = best_at + t->tokens[best_id].text_len;
	}

	if (add_specials && t->add_eos && t->eos_id >= 0 &&
		emit_special_token(out_ids, max_out, &written, t->eos_id) < 0) {
		profile_end(prof, &ps);
		return -1;
	}

	profile_end(prof, &ps);
	return written;
}

int tokenizer_decode(tokenizer *t, const int32_t *ids, int n_ids, char *out, int max_out,
					 profile *prof) {
	profile_scope ps = profile_begin(prof, STAGE_TOKENIZE_DECODE);
	int			  result;
	char		  stack_acc[TOK_DECODE_STACK_CAP];
	unsigned char stack_mark[TOK_DECODE_STACK_CAP];
	memset(stack_acc, 0, sizeof(stack_acc));
	memset(stack_mark, 0, sizeof(stack_mark));
	char		  *acc		 = stack_acc;
	unsigned char *mark		 = stack_mark;
	size_t		   acc_cap	 = TOK_DECODE_STACK_CAP;
	size_t		   acc_len	 = 0;
	int			   acc_heap	 = 0;
	int			   mark_heap = 0;

	for (int i = 0; i < n_ids; i++) {
		int32_t id = ids[i];
		if (id < 0 || (size_t)id >= t->n_tokens)
			continue;
		const vocab_token *tok = &t->tokens[id];
		int				   bv  = -1;
		size_t			   n   = tok->text_len;
		if (tok->type == TOK_TYPE_BYTE) {
			bv = byte_token_value(tok);
			n  = (bv >= 0) ? 1 : tok->text_len;
		}
		if (acc_len + n > acc_cap) {
			size_t new_cap = acc_cap;
			while (acc_len + n > new_cap)
				new_cap <<= 1;
			if (acc_heap) {
				acc = xrealloc(acc, new_cap);
			} else {
				char *heap_acc = xmalloc(new_cap);
				memcpy(heap_acc, acc, acc_len);
				acc		 = heap_acc;
				acc_heap = 1;
			}
			if (mark_heap) {
				mark = xrealloc(mark, new_cap);
			} else {
				unsigned char *heap_mark = xmalloc(new_cap);
				memcpy(heap_mark, mark, acc_len);
				mark	  = heap_mark;
				mark_heap = 1;
			}
			memset(mark + acc_len, 0, new_cap - acc_len);
			acc_cap = new_cap;
		}
		if (bv >= 0) {
			acc[acc_len]  = (char)bv;
			mark[acc_len] = 1;
		} else {
			memcpy(acc + acc_len, tok->text, n);
			memset(mark + acc_len, 0, n);
		}
		acc_len += n;
	}

	if (t->is_sentencepiece && acc_len >= 3) {
		size_t rd = 0;
		size_t wr = 0;
		while (rd + 2 < acc_len) {
			if ((unsigned char)acc[rd] == 0xE2 && (unsigned char)acc[rd + 1] == 0x96 &&
				(unsigned char)acc[rd + 2] == 0x81) {
				acc[wr++] = ' ';
				rd += 3;
			} else {
				acc[wr++] = acc[rd++];
			}
		}
		while (rd < acc_len)
			acc[wr++] = acc[rd++];
		acc_len = wr;
	}

	char   stack_raw[TOK_DECODE_STACK_CAP];
	char  *raw		= stack_raw;
	int	   raw_heap = 0;
	size_t raw_len;
	if (t->is_sentencepiece) {
		if (acc_len + 1 > sizeof(stack_raw)) {
			raw		 = xmalloc(acc_len + 1);
			raw_heap = 1;
		}
		memcpy(raw, acc, acc_len);
		raw_len		 = acc_len;
		raw[raw_len] = '\0';
	} else {
		if (acc_len + 1 > sizeof(stack_raw)) {
			raw		 = xmalloc(acc_len + 1);
			raw_heap = 1;
		}
		raw_len = gpt2_decode_to_bytes_buf(acc, mark, acc_len, raw);
	}

	if (acc_heap)
		free(acc);
	if (mark_heap)
		free(mark);

	if ((int)raw_len >= max_out) {
		if (raw_heap)
			free(raw);
		result = -1;
		profile_end(prof, &ps);
		return result;
	}
	memcpy(out, raw, raw_len);
	out[raw_len] = '\0';
	if (raw_heap)
		free(raw);
	result = (int)raw_len;
	profile_end(prof, &ps);
	return result;
}