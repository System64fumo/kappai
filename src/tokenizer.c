#include "tokenizer.h"
#include "log.h"
#include <ctype.h>

typedef struct {
	const char *p;
	size_t		n;
	uint64_t	h;
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

static uint64_t g_byte_hash[256];
static uint8_t	g_is_letter_tab[256];
static uint8_t	g_is_digit_tab[256];
static uint8_t	g_is_space_tab[256];
static uint8_t	g_is_newline_tab[256];
static uint8_t	g_utf8_len_tab[256];
static int		g_tables_ready = 0;

static void tokenizer_tables_ensure(void) {
	if (g_tables_ready)
		return;
	for (int i = 0; i < 256; i++) {
		unsigned char c = (unsigned char)i;
		char		  b = (char)c;
		g_byte_hash[i]	= fnv1a(&b, 1);
		g_is_letter_tab[i] =
			(uint8_t)((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= 0x80));
		g_is_digit_tab[i] = (uint8_t)(c >= '0' && c <= '9');
		g_is_space_tab[i] =
			(uint8_t)(c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r');
		g_is_newline_tab[i] = (uint8_t)(c == '\n' || c == '\r');
		g_utf8_len_tab[i]	= (uint8_t)(c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1);
	}
	g_tables_ready = 1;
}

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

static char	   g_byte_utf8[256][4];
static uint8_t g_byte_utf8_len[256];
static uint8_t g_byte_encode_passthrough[256];
static uint8_t g_byte_decode_passthrough[256];
static int	   g_byte_utf8_ready = 0;

static void gpt2_byte_table_ensure(void) {
	if (g_byte_utf8_ready)
		return;
	for (int i = 0; i < 256; i++) {
		g_byte_utf8_len[i] = (uint8_t)cp_to_utf8(g_byte_to_cp[i], g_byte_utf8[i]);
		g_byte_encode_passthrough[i] =
			(uint8_t)(g_byte_utf8_len[i] == 1 &&
					  (unsigned char)g_byte_utf8[i][0] == (unsigned char)i);
	}
	for (int i = 0; i < 256; i++)
		g_byte_decode_passthrough[i] = (uint8_t)(i < 0x80 && g_cp_to_byte[i] == i);
	g_byte_utf8_ready = 1;
}

static const char *gpt2_encode_bytes(tokenizer *t, const char *in, size_t in_len, size_t *out_len) {
	gpt2_byte_table_ensure();
	const unsigned char *src = (const unsigned char *)in;
	size_t				 i	 = 0;
	while (i < in_len && g_byte_encode_passthrough[src[i]])
		i++;
	if (i == in_len) {
		*out_len = in_len;
		return in;
	}
	sb_reset(&t->gpt2_scratch);
	sb_reserve(&t->gpt2_scratch, in_len * 2 + 1);
	char *dst = t->gpt2_scratch.p;
	memcpy(dst, src, i);
	size_t at = i;
	while (i < in_len) {
		size_t run_start = i;
		while (i < in_len && g_byte_encode_passthrough[src[i]])
			i++;
		if (i > run_start) {
			memcpy(dst + at, src + run_start, i - run_start);
			at += i - run_start;
		}
		if (i >= in_len)
			break;
		unsigned b = src[i];
		size_t	 n = g_byte_utf8_len[b];
		memcpy(dst + at, g_byte_utf8[b], n);
		at += n;
		i++;
	}
	dst[at]				= '\0';
	t->gpt2_scratch.len = at;
	*out_len			= at;
	return dst;
}

static int cp_is_letter(int cp) {
	if (cp < 0x80)
		return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
	if (cp >= 0x2000 && cp <= 0x2BFF)
		return 0;
	if (cp >= 0x1F000 && cp <= 0x1FFFF)
		return 0;
	if ((cp >= 0x80 && cp <= 0xA9) || (cp >= 0xAB && cp <= 0xB4) || (cp >= 0xB6 && cp <= 0xB9) ||
		(cp >= 0xBB && cp <= 0xBF) || cp == 0xD7 || cp == 0xF7)
		return 0;
	return 1;
}

static int cp_is_digit(int cp) {
	return cp >= '0' && cp <= '9';
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
	while (ht[h].key) {
		if (ht[h].key_len == klen && memcmp(ht[h].key, key, klen) == 0) {
			return;
		}
		h = (h + 1) & (cap - 1);
	}
	ht[h].key	  = key;
	ht[h].key_len = (uint32_t)klen;
	ht[h].id	  = id;
}

static int32_t hash_lookup_h(const tok_hash_entry *ht, size_t cap, uint64_t h, const char *key,
							 size_t klen) {
	h &= (cap - 1);
	while (ht[h].key) {
		if (ht[h].key_len == klen && memcmp(ht[h].key, key, klen) == 0) {
			return ht[h].id;
		}
		h = (h + 1) & (cap - 1);
	}
	return -1;
}

static int32_t hash_lookup(const tok_hash_entry *ht, size_t cap, const char *key, size_t klen) {
	return hash_lookup_h(ht, cap, fnv1a(key, klen), key, klen);
}

static int32_t hash_lookup_pair_h(const tok_hash_entry *ht, size_t cap, uint64_t h, const char *a,
								  size_t an, const char *b, size_t bn) {
	size_t klen = an + bn;
	h &= (cap - 1);
	while (ht[h].key) {
		if (ht[h].key_len == klen && memcmp(ht[h].key, a, an) == 0 &&
			memcmp(ht[h].key + an, b, bn) == 0) {
			return ht[h].id;
		}
		h = (h + 1) & (cap - 1);
	}
	return -1;
}

static int32_t hash_lookup_pair(const tok_hash_entry *ht, size_t cap, const char *a, size_t an,
								const char *b, size_t bn) {
	return hash_lookup_pair_h(ht, cap, fnv1a_update(fnv1a(a, an), b, bn), a, an, b, bn);
}

static size_t next_pretoken(const char *s, size_t len, size_t *pos) {
	size_t i = *pos;
	if (i >= len)
		return 0;

	unsigned char c = (unsigned char)s[i];

	if (c == '\'') {
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

	if (c == ' ' && i + 1 < len && g_is_digit_tab[(unsigned char)s[i + 1]]) {
		size_t j = i + 1;
		while (j < len && g_is_digit_tab[(unsigned char)s[j]])
			j++;
		*pos = j;
		return j - i;
	}

	{
		size_t start = i;
		size_t j	 = i;
		if (j < len) {
			unsigned char ch = (unsigned char)s[j];
			if (!g_is_newline_tab[ch] && !g_is_letter_tab[ch] && !g_is_digit_tab[ch]) {
				j++;
			}
		}
		if (j < len && g_is_letter_tab[(unsigned char)s[j]]) {
			while (j < len && g_is_letter_tab[(unsigned char)s[j]])
				j++;
			*pos = j;
			return j - start;
		}
	}

	if (g_is_digit_tab[c]) {
		size_t j = i;
		while (j < len && g_is_digit_tab[(unsigned char)s[j]])
			j++;
		*pos = j;
		return j - i;
	}

	{
		size_t start = i;
		size_t j	 = i;
		if (j < len && s[j] == ' ')
			j++;
		if (j < len && !g_is_space_tab[(unsigned char)s[j]] &&
			!g_is_letter_tab[(unsigned char)s[j]] && !g_is_digit_tab[(unsigned char)s[j]]) {
			while (j < len && !g_is_space_tab[(unsigned char)s[j]] &&
				   !g_is_letter_tab[(unsigned char)s[j]] && !g_is_digit_tab[(unsigned char)s[j]])
				j++;
			while (j < len && g_is_newline_tab[(unsigned char)s[j]])
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
		if (j < len && g_is_newline_tab[(unsigned char)s[j]]) {
			while (j < len && g_is_newline_tab[(unsigned char)s[j]])
				j++;
			*pos = j;
			return j - start;
		}
	}

	if (g_is_space_tab[c]) {
		size_t j = i;
		while (j < len && g_is_space_tab[(unsigned char)s[j]])
			j++;
		if (j < len && !g_is_space_tab[(unsigned char)s[j]]) {
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
	while (i < len && g_is_space_tab[(unsigned char)s[i]]) {
		if (g_is_newline_tab[(unsigned char)s[i]])
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

	if (c == '\'' && i + 1 < len) {
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
		if (!g_is_newline_tab[c] && !cp_is_letter(cp0) && !cp_is_digit(cp0))
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

	if (g_is_digit_tab[c]) {
		size_t j = i;
		int	   n = 0;
		while (j < len && g_is_digit_tab[(unsigned char)s[j]] && n < digit_run) {
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
			if (!g_is_space_tab[(unsigned char)s[j]] && !cp_is_letter(cp0) && !cp_is_digit(cp0)) {
				while (j < len) {
					int			  cl2;
					int			  cpn = cp_at(s, len, j, &cl2);
					unsigned char cj  = (unsigned char)s[j];
					if (g_is_space_tab[cj] || cp_is_letter(cpn) || cp_is_digit(cpn))
						break;
					j += (size_t)cl2;
				}
				while (j < len && g_is_newline_tab[(unsigned char)s[j]])
					j++;
				*pos = j;
				return j - start;
			}
		}
	}

	if (g_is_space_tab[c]) {
		size_t start = i;

		if (contains_newline_before_non_newline_run(s, len, i)) {
			size_t j		   = i;
			size_t last_nl_end = i;
			while (j < len && g_is_space_tab[(unsigned char)s[j]]) {
				if (g_is_newline_tab[(unsigned char)s[j]]) {
					j++;
					last_nl_end = j;
				} else {
					size_t k = j;
					while (k < len && g_is_space_tab[(unsigned char)s[k]] &&
						   !g_is_newline_tab[(unsigned char)s[k]])
						k++;
					if (k < len && g_is_newline_tab[(unsigned char)s[k]]) {
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
			while (j < len && g_is_space_tab[(unsigned char)s[j]])
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
} bpe_state;

static int32_t bpe_pair_rank(tokenizer *t, bpe_state *bs, int i) {
	const piece *a = &bs->pcs[i];
	const piece *b = &bs->pcs[bs->next[i]];
	if (a->locked || b->locked)
		return -1;
	uint64_t h = fnv1a_update(a->h, b->p, b->n);
	if (t->has_merges)
		return hash_lookup_pair_h((const tok_hash_entry *)t->merge_hash, t->merge_hash_capacity, h,
								  a->p, a->n, b->p, b->n);
	return hash_lookup_pair_h(t->hash, t->hash_capacity, h, a->p, a->n, b->p, b->n);
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
	size_t new_cap = t->bpe_work_cap ? (size_t)t->bpe_work_cap : 4096;
	while (new_cap < need)
		new_cap *= 2;
	t->bpe_work		= xrealloc(t->bpe_work, new_cap);
	t->bpe_work_cap = (uint32_t)new_cap;
	return 0;
}

#define TOK_CHUNK_CACHE_SLOTS (1u << 14)
#define TOK_CHUNK_CACHE_MAX_ENTRIES 4096
#define TOK_CHUNK_CACHE_MAX_KEY 64

static const tok_cache_entry *chunk_cache_find(const tok_cache_entry *tab, uint32_t cap,
											   const char *key, size_t klen) {
	uint32_t mask = cap - 1;
	uint32_t i	  = (uint32_t)(fnv1a(key, klen) & mask);
	for (uint32_t probe = 0; probe <= mask; probe++) {
		const tok_cache_entry *e = &tab[i];
		if (e->key == NULL)
			return NULL;
		if (e->key_len == klen && memcmp(e->key, key, klen) == 0)
			return e;
		i = (i + 1) & mask;
	}
	return NULL;
}

static tok_cache_entry *chunk_cache_slot_for_insert(tok_cache_entry *tab, uint32_t cap,
													const char *key, size_t klen) {
	uint32_t mask = cap - 1;
	uint32_t i	  = (uint32_t)(fnv1a(key, klen) & mask);
	for (uint32_t probe = 0; probe <= mask; probe++) {
		tok_cache_entry *e = &tab[i];
		if (e->key == NULL || (e->key_len == klen && memcmp(e->key, key, klen) == 0))
			return e;
		i = (i + 1) & mask;
	}
	return NULL;
}

static void *chunk_cache_arena_alloc(str_arena *a, size_t n) {
	void *p = str_arena_alloc(a, (n + 7u) & ~(size_t)7u);
	return p;
}

static int bpe_encode_uncached(tokenizer *t, const char *text, size_t len, int32_t *out_ids,
							   int max_out, int *n_out) {
	if (len == 0) {
		*n_out = 0;
		return 0;
	}
	if (t->bpe_pcs_cache_cap < len) {
		size_t new_cap = t->bpe_pcs_cache_cap ? (size_t)t->bpe_pcs_cache_cap : 64;
		while (new_cap < len)
			new_cap *= 2;
		t->bpe_pcs_cache	 = xrealloc(t->bpe_pcs_cache, new_cap * sizeof(piece));
		t->bpe_pcs_cache_cap = (uint32_t)new_cap;
	}
	piece *pcs	= t->bpe_pcs_cache;
	int	   npcs = 0;

	size_t char_idx = 0;
	while (char_idx < len) {
		unsigned char c		   = (unsigned char)text[char_idx];
		size_t		  char_len = g_utf8_len_tab[c];

		if (char_idx + char_len > len)
			char_len = 1;

		uint64_t h;
		int32_t	 id;
		if (char_len == 1) {
			h  = g_byte_hash[c];
			id = t->byte_vocab_ids[c];
		} else {
			h  = fnv1a(text + char_idx, char_len);
			id = hash_lookup_h(t->hash, t->hash_capacity, h, text + char_idx, char_len);
		}

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
				pcs[npcs].h		 = fnv1a(pcs[npcs].p, pcs[npcs].n);
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
		pcs[npcs].h		 = h;
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

	ARR_ENSURE(t->bpe_arena, len, t->bpe_arena_cap);
	size_t arena_cap  = t->bpe_arena_cap;
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
		pcs[nd].p	   = merged;
		pcs[nd].n	   = (size_t)best_klen;
		pcs[nd].h	   = fnv1a_update(pcs[nd].h, pcs[nx].p, pcs[nx].n);
		int32_t new_id = t->has_merges ? t->merge_token_ids[cur] : cur;
		pcs[nd].id	   = (new_id >= 0) ? new_id : ((t->unk_id >= 0) ? t->unk_id : 0);

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

int tokenizer_bpe_encode(tokenizer *t, const char *text, size_t len, int32_t *out_ids, int max_out,
						 int *n_out) {
	if (len == 0) {
		*n_out = 0;
		return 0;
	}
	if (len > TOK_CHUNK_CACHE_MAX_KEY)
		return bpe_encode_uncached(t, text, len, out_ids, max_out, n_out);

	const tok_cache_entry *hit = chunk_cache_find(t->chunk_cache, t->chunk_cache_cap, text, len);
	if (hit) {
		if ((int)hit->n_ids > max_out)
			return -1;
		memcpy(out_ids, hit->ids, hit->n_ids * sizeof(int32_t));
		*n_out = (int)hit->n_ids;
		return 0;
	}

	int32_t tmp[TOK_CHUNK_CACHE_MAX_KEY + 1];
	int		n = 0;
	if (bpe_encode_uncached(t, text, len, tmp, TOK_CHUNK_CACHE_MAX_KEY + 1, &n) != 0)
		return bpe_encode_uncached(t, text, len, out_ids, max_out, n_out);

	if (n <= (int)ARRAY_LEN(tmp) && t->chunk_cache_used < TOK_CHUNK_CACHE_MAX_ENTRIES) {
		tok_cache_entry *slot =
			chunk_cache_slot_for_insert(t->chunk_cache, t->chunk_cache_cap, text, len);
		if (slot && slot->key == NULL) {
			char *key = str_arena_alloc(&t->chunk_cache_pool, len);
			memcpy(key, text, len);
			int32_t *ids =
				chunk_cache_arena_alloc(&t->chunk_cache_pool, (size_t)n * sizeof(int32_t));
			memcpy(ids, tmp, (size_t)n * sizeof(int32_t));
			slot->key	  = key;
			slot->ids	  = ids;
			slot->key_len = (uint32_t)len;
			slot->n_ids	  = (uint32_t)n;
			t->chunk_cache_used++;
		}
	}

	if (n > max_out)
		return -1;
	memcpy(out_ids, tmp, (size_t)n * sizeof(int32_t));
	*n_out = n;
	return 0;
}

static int encode_sp_chunk(tokenizer *t, const char *text, size_t start, size_t end,
						   int32_t *out_ids, int max_out, int *written) {
	size_t sub_len	= end - start;
	int	   need_buf = t->add_space_prefix;
	if (!need_buf) {
		for (size_t i = start; i < end; i++) {
			if (text[i] == ' ') {
				need_buf = 1;
				break;
			}
		}
	}
	if (!need_buf) {
		int n;
		if (tokenizer_bpe_encode(t, text + start, sub_len, out_ids + *written, max_out - *written,
								 &n) < 0)
			return -1;
		*written += n;
		return 0;
	}
	sb_reset(&t->bpe_sp);
	sb_reserve(&t->bpe_sp, (sub_len + 1) * 3 + 1);
	if (t->add_space_prefix)
		sb_putb(&t->bpe_sp, "\xe2\x96\x81", 3);
	{
		size_t i = start;
		while (i < end) {
			if (text[i] == ' ') {
				sb_putb(&t->bpe_sp, "\xe2\x96\x81", 3);
				i++;
			} else {
				size_t j = i;
				while (j < end && text[j] != ' ')
					j++;
				sb_putb(&t->bpe_sp, text + i, j - i);
				i = j;
			}
		}
	}
	int n;
	if (tokenizer_bpe_encode(t, t->bpe_sp.p, t->bpe_sp.len, out_ids + *written, max_out - *written,
							 &n) < 0)
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
		size_t		enc_len;
		const char *enc = gpt2_encode_bytes(t, text + pstart, plen, &enc_len);
		int			n;
		if (tokenizer_bpe_encode(t, enc, enc_len, out_ids + *written, max_out - *written, &n) < 0) {
			return -1;
		}
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
		size_t b0 = t->special_by_first_byte_off[c];
		size_t b1 = t->special_by_first_byte_off[(size_t)c + 1];
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
		p++;
	}
	return -1;
}

static size_t decoded_token_len(const tokenizer *t, int32_t id) {
	const vocab_token *tok = &t->tokens[id];
	if (tok->type == TOK_TYPE_BYTE) {
		int bv = t->token_id_to_byte[id];
		if (bv >= 0)
			return 1;
	}
	const char *src = tok->text;
	size_t		n	= tok->text_len;
	size_t		k	= 0;
	size_t		out = 0;
	if (t->is_sentencepiece) {
		while (k < n) {
			if (k + 2 < n && (unsigned char)src[k] == 0xE2 && (unsigned char)src[k + 1] == 0x96 &&
				(unsigned char)src[k + 2] == 0x81) {
				out++;
				k += 3;
			} else {
				out++;
				k++;
			}
		}
		return out;
	}
	while (k < n) {
		int cp;
		int kl = utf8_to_cp(src + k, (int)(n - k), &cp);
		if (kl <= 0)
			break;
		int b = (cp < 512) ? g_cp_to_byte[cp] : -1;
		out += (b >= 0) ? 1 : (size_t)kl;
		k += (size_t)kl;
	}
	return out;
}

static size_t decode_token_text_into(const tokenizer *t, int32_t id, char *out) {
	const vocab_token *tok = &t->tokens[id];
	if (tok->type == TOK_TYPE_BYTE) {
		int bv = t->token_id_to_byte[id];
		if (bv >= 0) {
			out[0] = (char)bv;
			return 1;
		}
	}
	const char *src		= tok->text;
	size_t		n		= tok->text_len;
	size_t		written = 0;
	size_t		k		= 0;
	if (t->is_sentencepiece) {
		while (k < n) {
			if (k + 2 < n && (unsigned char)src[k] == 0xE2 && (unsigned char)src[k + 1] == 0x96 &&
				(unsigned char)src[k + 2] == 0x81) {
				out[written++] = ' ';
				k += 3;
			} else {
				out[written++] = src[k++];
			}
		}
		return written;
	}
	while (k < n) {
		int cp;
		int kl = utf8_to_cp(src + k, (int)(n - k), &cp);
		if (kl <= 0)
			break;
		int b = (cp < 512) ? g_cp_to_byte[cp] : -1;
		if (b >= 0) {
			out[written++] = (char)b;
		} else {
			memcpy(out + written, src + k, (size_t)kl);
			written += (size_t)kl;
		}
		k += (size_t)kl;
	}
	return written;
}

static bool g_warned_no_merges;

status_code tokenizer_init(tokenizer *t, const gguf_ctx *g) {
	memset(t, 0, sizeof(*t));
	str_arena_init(&t->merge_pool);
	str_arena_init(&t->chunk_cache_pool);
	t->chunk_cache_cap = TOK_CHUNK_CACHE_SLOTS;
	t->chunk_cache	   = xcalloc(t->chunk_cache_cap, sizeof(tok_cache_entry));

	tokenizer_tables_ensure();
	gpt2_byte_table_ensure();

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

	for (int bi = 0; bi < 256; bi++) {
		t->byte_fallback_ids[bi] = -1;
		t->byte_vocab_ids[bi]	 = -1;
	}
	for (size_t i = 0; i < n_toks; i++) {
		if (t->tokens[i].text_len != 1)
			continue;
		unsigned char b = (unsigned char)t->tokens[i].text[0];
		if (t->byte_vocab_ids[b] < 0)
			t->byte_vocab_ids[b] = (int32_t)i;
	}
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
		DEBUG("tokenizer: %u byte-fallback pieces registered", t->n_byte_fallback);

	t->token_id_to_byte = xmalloc(n_toks * sizeof(int16_t));
	for (size_t i = 0; i < n_toks; i++)
		t->token_id_to_byte[i] =
			(t->tokens[i].type == TOK_TYPE_BYTE) ? (int16_t)byte_token_value(&t->tokens[i]) : -1;

	t->token_decoded_len = xmalloc(n_toks * sizeof(int32_t));
	t->token_char_count	 = xmalloc(n_toks * sizeof(int32_t));
	for (size_t i = 0; i < n_toks; i++) {
		const vocab_token *tok = &t->tokens[i];
		if (tok->type == TOK_TYPE_BYTE && t->token_id_to_byte[i] >= 0) {
			t->token_char_count[i] = 1;
			continue;
		}
		int32_t chars = 0;
		size_t	k	  = 0;
		while (k < tok->text_len) {
			int cp;
			int kl = utf8_to_cp(tok->text + k, (int)(tok->text_len - k), &cp);
			if (kl <= 0)
				break;
			chars++;
			k += (size_t)kl;
		}
		t->token_char_count[i] = chars;
	}
	for (size_t i = 0; i < n_toks; i++)
		t->token_decoded_len[i] = 0;

	const char *const *merges	= NULL;
	size_t			   n_merges = 0;
	if (gguf_get_arr_str(g, "tokenizer.ggml.merges", &merges, &n_merges) == OK && n_merges > 0) {
		size_t mcap = 1;
		while (mcap < n_merges * 2)
			mcap <<= 1;
		t->merge_hash_capacity = mcap;
		t->merge_hash		   = xcalloc(mcap, sizeof(tok_hash_entry));
		t->merge_keys		   = (char **)xcalloc(n_merges, sizeof(char *));
		t->merge_token_ids	   = xmalloc(n_merges * sizeof(int32_t));
		t->n_merge_keys		   = 0;
		for (size_t i = 0; i < n_merges; i++)
			t->merge_token_ids[i] = -1;
		size_t pool_need = 0;
		for (size_t i = 0; i < n_merges; i++) {
			const char *sp = strchr(merges[i], ' ');
			if (!sp)
				continue;
			pool_need += (size_t)(sp - merges[i]) + strlen(sp + 1) + 1;
		}
		if (pool_need > 0)
			str_arena_reserve(&t->merge_pool, pool_need);
		for (size_t i = 0; i < n_merges; i++) {
			const char *entry = merges[i];
			const char *sp	  = strchr(entry, ' ');
			if (!sp)
				continue;
			size_t left_len	 = (size_t)(sp - entry);
			size_t right_len = strlen(sp + 1);
			size_t klen		 = left_len + right_len;
			char  *key		 = str_arena_alloc(&t->merge_pool, klen + 1);
			memcpy(key, entry, left_len);
			memcpy(key + left_len, sp + 1, right_len);
			key[klen]						 = '\0';
			t->merge_keys[t->n_merge_keys++] = key;
			hash_insert((tok_hash_entry *)t->merge_hash, mcap, key, klen, (int32_t)i);
			t->merge_token_ids[i] =
				hash_lookup((const tok_hash_entry *)t->hash, t->hash_capacity, key, klen);
		}
		t->has_merges = 1;
		DEBUG("tokenizer: loaded %zu BPE merge rules", n_merges);
	} else {
		t->merge_hash		   = NULL;
		t->merge_hash_capacity = 0;
		t->merge_token_ids	   = NULL;
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

	{
		size_t total = 0;
		for (size_t i = 0; i < n_toks; i++)
			total += decoded_token_len(t, (int32_t)i);
		t->decoded_pool = xmalloc(total ? total : 1);
		t->decoded_off	= xmalloc(n_toks * sizeof(int32_t));
		size_t off		= 0;
		for (size_t i = 0; i < n_toks; i++) {
			t->decoded_off[i] = (int32_t)off;
			size_t dlen		  = decode_token_text_into(t, (int32_t)i, t->decoded_pool + off);
			off += dlen;
			t->token_decoded_len[i] = (int32_t)dlen;
		}
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
		for (int b2 = 0; b2 < 256; b2++) {
			size_t start = t->special_by_first_byte_off[b2];
			size_t end	 = t->special_by_first_byte_off[b2 + 1];
			size_t cnt	 = end - start;
			if (cnt <= 1)
				continue;
			int32_t *arr = t->special_by_first_byte + start;
			for (size_t i = 1; i < cnt; i++) {
				int32_t key		= arr[i];
				size_t	key_len = t->tokens[key].text_len;
				size_t	j		= i;
				while (j > 0 && t->tokens[arr[j - 1]].text_len < key_len) {
					arr[j] = arr[j - 1];
					j--;
				}
				arr[j] = key;
			}
		}
	}

	return OK;
}

void tokenizer_free(tokenizer *t) {
	free(t->tokens);
	free(t->token_id_to_byte);
	free(t->token_decoded_len);
	free(t->token_char_count);
	free(t->decoded_off);
	free(t->decoded_pool);
	free((void *)t->hash);
	free((void *)t->merge_hash);
	free(t->merge_token_ids);
	str_arena_free(&t->merge_pool);
	str_arena_free(&t->chunk_cache_pool);
	free(t->chunk_cache);
	free((void *)t->merge_keys);
	free(t->special_ids);
	free(t->special_by_first_byte);
	free(t->bpe_pcs_cache);
	free(t->bpe_work);
	free(t->bpe_arena);
	sb_free(&t->bpe_sp);
	sb_free(&t->gpt2_scratch);
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
	return (size_t)t->token_decoded_len[id];
}

int tokenizer_token_count_for_bytes(const tokenizer *t, const int32_t *ids, int n,
									size_t max_bytes) {
	size_t out		 = 0;
	int	   cur_owner = -1;
	for (int i = 0; i < n; i++) {
		int32_t id = ids[i];
		if (id < 0 || (size_t)id >= t->n_tokens)
			break;
		if (out >= max_bytes)
			break;
		size_t c	   = (size_t)t->token_char_count[id];
		size_t can_add = max_bytes - out;
		size_t add	   = c < can_add ? c : can_add;
		out += add;
		cur_owner = i;
		if (out >= max_bytes)
			break;
	}
	return cur_owner + 1;
}

int32_t tokenizer_find_token(const tokenizer *t, const char *text) {
	if (t->hash) {
		int32_t id =
			hash_lookup((const tok_hash_entry *)t->hash, t->hash_capacity, text, strlen(text));
		if (id >= 0 && (size_t)id < t->n_tokens)
			return id;
	}

	return -1;
}

int tokenizer_starts_with_special(const tokenizer *t, const char *s) {
	if (!t || !s || !s[0])
		return 0;
	size_t at = 0;
	return find_next_special(t, s, strlen(s), 0, &at) >= 0 && at == 0;
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
	if (max_out <= 0)
		goto fail;
	size_t pos = 0;
	for (int i = 0; i < n_ids; i++) {
		int32_t id = ids[i];
		if (id < 0 || (size_t)id >= t->n_tokens)
			continue;
		size_t dlen = (size_t)t->token_decoded_len[id];
		if (pos + dlen >= (size_t)max_out)
			goto fail;
		memcpy(out + pos, t->decoded_pool + t->decoded_off[id], dlen);
		pos += dlen;
	}
	out[pos] = '\0';
	profile_end(prof, &ps);
	return (int)pos;

fail:
	profile_end(prof, &ps);
	return -1;
}
