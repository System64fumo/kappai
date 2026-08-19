#include "tokenizer.h"
#include "log.h"
#include <ctype.h>

#define TOK_DECODE_STACK_CAP 256

typedef struct tok_hash_entry {
	const char *key;
	size_t		key_len;
	int32_t		id;
	int			used;
} tok_hash_entry;

typedef struct {
	const char *p;
	size_t		n;
	int32_t		id;
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

static size_t gpt2_decode_to_bytes_buf(const char *in, size_t in_len, char *out) {
	const int *cp_to_byte = g_cp_to_byte;
	size_t	   n		  = 0;
	size_t	   i		  = 0;
	while (i < in_len) {
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
			unsigned char c1 = (unsigned char)tolower((unsigned char)s[i + 1]);
			if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
				*pos = i + 2;
				return 2;
			}
			if (i + 2 < len) {
				unsigned char c2 = (unsigned char)tolower((unsigned char)s[i + 2]);
				if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
					(c1 == 'l' && c2 == 'l')) {
					*pos = i + 3;
					return 3;
				}
			}
		}
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
		int	   n = 0;
		while (j < len && is_utf8_digit((unsigned char)s[j]) && n < 3) {
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

static size_t next_pretoken_llama3(const char *s, size_t len, size_t *pos) {
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
		while (j < len && is_utf8_digit((unsigned char)s[j]) && n < 3) {
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

static int bpe_encode(tokenizer *t, const char *text, size_t len, int32_t *out_ids, int max_out,
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
		if (id < 0)
			id = (t->unk_id >= 0) ? t->unk_id : 0;
		pcs[npcs].p	 = text + char_idx;
		pcs[npcs].n	 = char_len;
		pcs[npcs].id = id;
		npcs++;
		char_idx += char_len;
	}

	char  *merge_key	 = NULL;
	size_t merge_key_cap = 0;

	size_t arena_cap  = len;
	char  *arena	  = xmalloc(arena_cap);
	size_t arena_used = 0;

	while (npcs > 1) {
		int		best_i	  = -1;
		int32_t best_rank = INT32_MAX;
		size_t	best_klen = 0;

		for (int i = 0; i < npcs - 1; i++) {
			size_t klen = pcs[i].n + pcs[i + 1].n;
			if (klen > merge_key_cap) {
				merge_key_cap = klen * 2;
				merge_key	  = xrealloc(merge_key, merge_key_cap);
			}
			memcpy(merge_key, pcs[i].p, pcs[i].n);
			memcpy(merge_key + pcs[i].n, pcs[i + 1].p, pcs[i + 1].n);

			int32_t rank;
			if (t->has_merges) {
				rank = hash_lookup((const tok_hash_entry *)t->merge_hash, t->merge_hash_capacity,
								   merge_key, klen);
				if (rank < 0)
					continue;
			} else {
				rank = hash_lookup(t->hash, t->hash_capacity, merge_key, klen);
				if (rank < 0)
					continue;
			}
			if (rank < best_rank) {
				best_rank = rank;
				best_i	  = i;
				best_klen = klen;
			}
		}

		if (best_i < 0)
			break;

		if (arena_used + best_klen > arena_cap) {
			uintptr_t old_base = (uintptr_t)arena;
			size_t	  old_used = arena_used;
			size_t	  new_cap  = arena_cap * 2;
			while (arena_used + best_klen > new_cap)
				new_cap *= 2;
			char *new_arena = xrealloc(arena, new_cap);
			if ((uintptr_t)new_arena != old_base) {
				for (int i = 0; i < npcs; i++) {
					uintptr_t p = (uintptr_t)pcs[i].p;
					if (p >= old_base && p < old_base + old_used) {
						pcs[i].p = new_arena + (p - old_base);
					}
				}
			}
			arena	  = new_arena;
			arena_cap = new_cap;
		}

		char *merged = arena + arena_used;
		arena_used += best_klen;
		memcpy(merged, pcs[best_i].p, pcs[best_i].n);
		memcpy(merged + pcs[best_i].n, pcs[best_i + 1].p, pcs[best_i + 1].n);

		pcs[best_i].p			= merged;
		pcs[best_i].n			= best_klen;
		int32_t merged_vocab_id = hash_lookup(t->hash, t->hash_capacity, merged, best_klen);
		pcs[best_i].id =
			(merged_vocab_id >= 0) ? merged_vocab_id : ((t->unk_id >= 0) ? t->unk_id : 0);

		for (int i = best_i + 1; i < npcs - 1; i++)
			pcs[i] = pcs[i + 1];
		npcs--;
	}

	int written = 0;
	for (int i = 0; i < npcs; i++) {
		if (written >= max_out)
			goto fail;
		out_ids[written++] = pcs[i].id;
	}
	free(merge_key);
	free(arena);
	*n_out = written;
	return 0;

fail:
	free(merge_key);
	free(arena);
	return -1;
}

static int encode_sp_chunk(tokenizer *t, const char *text, size_t start, size_t end,
						   int32_t *out_ids, int max_out, int *written) {
	size_t sub_len = end - start;
	char  *sp_text = xmalloc((sub_len * 3) + 1);
	size_t sp_len  = 0;
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
	if (bpe_encode(t, sp_text, sp_len, out_ids + *written, max_out - *written, &n) < 0) {
		free(sp_text);
		return -1;
	}
	free(sp_text);
	*written += n;
	return 0;
}

static int encode_gpt2_chunk(tokenizer *t, const char *text, size_t start, size_t end,
							 int32_t *out_ids, int max_out, int *written) {
	size_t sub_pos = start;
	while (sub_pos < end) {
		size_t pstart = sub_pos;
		size_t plen	  = (t->pre_type == TOK_PRE_LLAMA3) ? next_pretoken_llama3(text, end, &sub_pos)
														: next_pretoken(text, end, &sub_pos);
		if (plen == 0)
			break;
		size_t enc_len;
		char  *enc = gpt2_encode_bytes(text + pstart, plen, &enc_len);
		int	   n;
		if (bpe_encode(t, enc, enc_len, out_ids + *written, max_out - *written, &n) < 0) {
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
	for (size_t p = from; p < len; p++) {
		unsigned char fb = (unsigned char)text[p];
		size_t		  b0 = t->special_by_first_byte_off[fb];
		size_t		  b1 = t->special_by_first_byte_off[(size_t)fb + 1];
		for (size_t bi = b0; bi < b1; bi++) {
			int32_t sid	 = t->special_by_first_byte[bi];
			size_t	nlen = t->tokens[sid].text_len;
			if (p + nlen > len)
				continue;
			if (memcmp(text + p, t->tokens[sid].text, nlen) == 0) {
				*out_at = p;
				return sid;
			}
		}
	}
	return -1;
}

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
		WARN("tokenizer: no merge rules in GGUF, using vocab-ID heuristic instead");
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

	t->pre_type			= TOK_PRE_GPT2;
	const char *pre_str = NULL;
	if (gguf_get_str(g, "tokenizer.ggml.pretokenizer", &pre_str) == OK && pre_str) {
		if (strstr(pre_str, "llama3") || strstr(pre_str, "llama-bpe")) {
			t->pre_type = TOK_PRE_LLAMA3;
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
		return 1;
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
	size_t cumulative = 0;
	for (int i = 0; i < n; i++) {
		int32_t id = ids[i];
		if (id < 0 || (size_t)id >= t->n_tokens)
			return i;
		size_t tlen = tokenizer_token_decoded_len(t, id);
		if (cumulative + tlen > max_bytes)
			return i;
		cumulative += tlen;
	}
	return n;
}

char *tokenizer_decode_prefix(const tokenizer *t, const int32_t *ids, int count) {
	size_t max_decoded = 0;
	for (int i = 0; i < count; i++) {
		int32_t id = ids[i];
		if (id < 0 || (size_t)id >= t->n_tokens)
			break;
		max_decoded += t->tokens[id].text_len;
	}
	max_decoded += 16;

	char *out = xmalloc(max_decoded + 1);
	int	  len = tokenizer_decode((tokenizer *)t, ids, count, out, (int)max_decoded + 1, NULL);
	if (len < 0) {
		free(out);
		return NULL;
	}
	return out;
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
	memset(stack_acc, 0, sizeof(stack_acc));
	char  *acc		= stack_acc;
	size_t acc_cap	= TOK_DECODE_STACK_CAP;
	size_t acc_len	= 0;
	int	   acc_heap = 0;

	for (int i = 0; i < n_ids; i++) {
		int32_t id = ids[i];
		if (id < 0 || (size_t)id >= t->n_tokens)
			continue;
		size_t n = t->tokens[id].text_len;
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
			acc_cap = new_cap;
		}
		memcpy(acc + acc_len, t->tokens[id].text, n);
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
		raw_len = gpt2_decode_to_bytes_buf(acc, acc_len, raw);
	}

	if (acc_heap)
		free(acc);

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