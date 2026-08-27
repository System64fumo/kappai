#include "test_core.h"
#include "test_synth_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TK_NORMAL = 1, TK_UNKNOWN = 2, TK_CONTROL = 3, TK_BYTE = 6 };

static char tok_dir[300];

static tokenizer load_vocab(tokenizer *t, gguf_ctx *g, const char *path) {
	memset(t, 0, sizeof(*t));
	memset(g, 0, sizeof(*g));
	if (gguf_load(g, path) != OK || tokenizer_init(t, g) != OK)
		abort();
	return *t;
}

static void drop_vocab(tokenizer *t, gguf_ctx *g) {
	tokenizer_free(t);
	gguf_free(g);
}

static int enc(tokenizer *t, const char *text, int32_t *ids, int max) {
	return tokenizer_encode_with_specials(t, text, 0, ids, max, NULL);
}

static int ids_match(const int32_t *got, int n_got, const int32_t *want, int n_want) {
	if (n_got != n_want)
		return 0;
	return memcmp(got, want, (size_t)n_want * sizeof(int32_t)) == 0;
}

static void record_tok(const char *label, int ok, const char *detail) {
	record_result(OPFAM_TOKENIZER, label, ok ? V_PASS : V_FAIL, detail);
}

static int roundtrip(tokenizer *t, const char *text, char *out, size_t out_cap) {
	int32_t ids[512];
	int		n = enc(t, text, ids, 512);
	if (n < 0)
		return -1;
	int len = tokenizer_decode(t, ids, n, out, (int)out_cap, NULL);
	if (len < 0 || (size_t)len != strlen(text))
		return -1;
	if (memcmp(out, text, strlen(text)) != 0)
		return -1;
	return n;
}

static const char *const tok_a[] = {
	"<unk>",
	"i",
	"e",
	"d",
	"\xc4\xa0"
	"4",
	"\xc4\xa0",
	"4",
	"IT",
	"'",
	"S",
	"it",
	"'s",
	"3",
	"33",
	"333",
	"3333",
	"<0xF0>",
	"<0x9F>",
	"<0xA6>",
	"<0x99>",
	"<0x41>",
	"<0xZZ>",
	"I",
	"T",
	"s",
};
static const int32_t typ_a[] = {
	TK_UNKNOWN, TK_NORMAL, TK_NORMAL, TK_NORMAL, TK_NORMAL, TK_NORMAL, TK_NORMAL,
	TK_NORMAL,	TK_NORMAL, TK_NORMAL, TK_NORMAL, TK_NORMAL, TK_NORMAL, TK_NORMAL,
	TK_NORMAL,	TK_NORMAL, TK_BYTE,	  TK_BYTE,	 TK_BYTE,	TK_BYTE,   TK_BYTE,
	TK_BYTE,	TK_NORMAL, TK_NORMAL, TK_NORMAL,
};

static void run_vocab_a(void) {
	char path[400];
	snprintf(path, sizeof(path), "%s/tok_vocabA.gguf", tok_dir);
	tsg_build_vocab_file(path, "gpt2", NULL, 0, 0, tok_a, typ_a, ARRAY_LEN(tok_a));

	tokenizer t;
	gguf_ctx  g;
	load_vocab(&t, &g, path);
	char	detail[256];
	int32_t ids[128];

	record_tok("vocabA.byte_fallback_table_populated", t.n_byte_fallback == 5, "");

	int n = enc(&t, "3333", ids, 128);
	snprintf(detail, sizeof(detail), "'3333' -> %d token(s), id=%d", n, n == 1 ? ids[0] : -1);
	record_tok("gpt2.uncapped_digit_run_single_token", n == 1 && ids[0] == 15, detail);

	n		= enc(&t, "ied 4", ids, 128);
	int ok4 = ids_match(ids, n, (const int32_t[]){1, 2, 3, 4}, 4);
	snprintf(detail, sizeof(detail), "'ied 4' -> [%d,%d,%d,%d]", ids[0], ids[1], ids[2], ids[3]);
	record_tok("gpt2.space_attaches_to_digit_run", ok4, detail);

	n		  = enc(&t, "IT'S", ids, 128);
	int ok_up = ids_match(ids, n, (const int32_t[]){7, 8, 9}, 3);
	n		  = enc(&t, "it's", ids, 128);
	int ok_lo = ids_match(ids, n, (const int32_t[]){10, 11}, 2);
	record_tok("gpt2.contraction_case_sensitive", ok_up && ok_lo, "IT'S -> IT|'|S ; it's -> it|'s");

	n		  = enc(&t, "\xf0\x9f\xa6\x99", ids, 128);
	int ok_fb = ids_match(ids, n, (const int32_t[]){16, 17, 18, 19}, 4);
	snprintf(detail, sizeof(detail), "U+1F999 -> %d pieces (byte fallback)", n);
	record_tok("byte_fallback_unknown_char_not_unk", ok_fb, detail);

	char out[64];
	int	 len = tokenizer_decode(&t, ids, n, out, (int)sizeof(out), NULL);
	record_tok("byte_fallback_roundtrip_emoji", len == 4 && memcmp(out, "\xf0\x9f\xa6\x99", 4) == 0,
			   "");

	int32_t bad	  = 21;
	len			  = tokenizer_decode(&t, &bad, 1, out, (int)sizeof(out), NULL);
	size_t dlen	  = tokenizer_token_decoded_len(&t, bad);
	int	   ok_bad = len == 6 && memcmp(out, "<0xZZ>", 6) == 0 && dlen == 6;
	snprintf(detail, sizeof(detail), "<0xZZ> decoded_len=%zu bytes='%.*s'", dlen, len > 0 ? len : 0,
			 out);
	record_tok("malformed_byte_escape_literal_safe", ok_bad, detail);

	drop_vocab(&t, &g);
}

static char		   f_store[600][8];
static const char *f_toks[600];
static int32_t	   f_types[600];

static size_t build_full_byte_vocab(void) {
	int id		   = 0;
	f_store[id][0] = '\0';
	snprintf(f_store[id], 8, "<unk>");
	f_toks[id]	  = f_store[id];
	f_types[id++] = TK_UNKNOWN;
	for (int c = 33; c <= 126; c++) {
		f_store[id][0] = (char)c;
		f_store[id][1] = '\0';
		f_toks[id]	   = f_store[id];
		f_types[id++]  = TK_NORMAL;
	}
	for (int b = 0; b < 256; b++) {
		snprintf(f_store[id], 8, "<0x%02X>", b);
		f_toks[id]	  = f_store[id];
		f_types[id++] = TK_BYTE;
	}
	return (size_t)id;
}

static void run_vocab_f_roundtrip(void) {
	size_t nvocab = build_full_byte_vocab();
	char   path[400];
	snprintf(path, sizeof(path), "%s/tok_vocabF.gguf", tok_dir);
	tsg_build_vocab_file(path, "gpt2", NULL, 0, 0, f_toks, f_types, nvocab);

	tokenizer t;
	gguf_ctx  g;
	load_vocab(&t, &g, path);

	static const char *samples[] = {
		"Hello world!",
		"the quick brown fox 42 jumps",
		"def f(n):\n\treturn n*2\n",
		"caf\xc3\xa9 na\xc3\xafve r\xc3\xa9sum\xc3\xa9",
		"\xe6\xb5\xaa\xe6\xbc\xab CJK \xe4\xb8\xad\xe6\x96\x87",
		"\xf0\x9f\xa6\x99\xf0\x9f\x9a\x80 emoji",
		"\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x92\xbb ZWJ sequence",
		"mixed ascii + \xe2\x82\xac euro + \xf0\x9f\x8c\xb2 pine",
	};
	int	 fails		= 0;
	char worst[128] = "";
	char out[512];
	for (size_t i = 0; i < ARRAY_LEN(samples); i++) {
		if (roundtrip(&t, samples[i], out, sizeof(out)) < 0) {
			fails++;
			if (!worst[0])
				snprintf(worst, sizeof(worst), "%.100s", samples[i]);
		}
	}
	record_tok("roundtrip_decode_encode_utf8_matrix", fails == 0,
			   fails ? "first failing sample recorded in debug" : "8/8 exact roundtrips");

	uint64_t st		   = 0xC0FFEE123ULL;
	int		 fuzz_fail = 0;
	char	 buf[97];
	for (int iter = 0; iter < 200; iter++) {
		size_t len = (size_t)(st >> 60) % 64 + 1;
		for (size_t j = 0; j < len; j++) {
			st ^= st << 13;
			st ^= st >> 7;
			st ^= st << 17;
			buf[j] = (char)(33 + ((st >> 16) % 94));
		}
		buf[len] = '\0';
		if (roundtrip(&t, buf, out, sizeof(out)) < 0) {
			fuzz_fail++;
			break;
		}
	}
	record_tok("roundtrip_ascii_fuzz_200_iters_seeded", fuzz_fail == 0,
			   fuzz_fail ? "seeded fuzz found mismatch" : "200/200 exact roundtrips");

	drop_vocab(&t, &g);
}

static void run_vocab_d(void) {
	char path[400];
	snprintf(path, sizeof(path), "%s/tok_vocabD.gguf", tok_dir);
	tsg_build_vocab_file(path, "gpt2", NULL, 0, 0, (const char *const[]){"<unk>", "i"},
						 (const int32_t[]){TK_UNKNOWN, TK_NORMAL}, 2);
	tokenizer t;
	gguf_ctx  g;
	load_vocab(&t, &g, path);
	int32_t ids[16];
	int		n = enc(&t, "\xf0\x9f\xa6\x99", ids, 16);
	record_tok("no_byte_table_falls_back_to_unk",
			   t.n_byte_fallback == 0 && n == 4 && ids[0] == 0 && ids[3] == 0, "");
	drop_vocab(&t, &g);
}

static const char *const tok_sp[] = {
	"<unk>",
	"<s>",
	"</s>",
	"\xe2\x96\x81Hello",
	"\xe2\x96\x81world",
	"Hello",
	"world",
	"\xe2\x96\x81",
	"H",
	"e",
	"l",
	"o",
	"w",
	"r",
	"d",
	"<s2>",
	"<0xE6>",
	"<0xB5>",
	"<0xAA>",
};
static const int32_t typ_sp[] = {TK_UNKNOWN, TK_CONTROL, TK_CONTROL, TK_NORMAL, TK_NORMAL,
								 TK_NORMAL,	 TK_NORMAL,	 TK_NORMAL,	 TK_NORMAL, TK_NORMAL,
								 TK_NORMAL,	 TK_NORMAL,	 TK_NORMAL,	 TK_NORMAL, TK_NORMAL,
								 TK_CONTROL, TK_BYTE,	 TK_BYTE,	 TK_BYTE};

static void run_spm_variant(const char *name, const char *model, int has_flag, int flag,
							int expect_prefix) {
	char path[400];
	snprintf(path, sizeof(path), "%s/tok_%s.gguf", tok_dir, name);
	tsg_build_vocab_file(path, model, NULL, has_flag, flag, tok_sp, typ_sp, ARRAY_LEN(tok_sp));

	tokenizer t;
	gguf_ctx  g;
	load_vocab(&t, &g, path);
	char detail[256];

	snprintf(detail, sizeof(detail), "%s add_space_prefix KV %s -> resolved %d (want %d)", model,
			 has_flag ? (flag ? "true" : "false") : "absent", t.add_space_prefix, expect_prefix);
	record_tok(name, t.is_sentencepiece == 1 && t.add_space_prefix == expect_prefix, detail);

	int32_t ids[128];
	int		n = enc(&t, "Hello world", ids, 128);
	if (expect_prefix) {
		int ok =
			ids_match(ids, n, (const int32_t[]){7, 8, 9, 10, 10, 11, 7, 12, 11, 13, 10, 14}, 12);
		record_tok("spm.dummy_prefix_prepended", ok, "leading metaspace piece present");
	} else {
		int ok = ids_match(ids, n, (const int32_t[]){8, 9, 10, 10, 11, 7, 12, 11, 13, 10, 14}, 11);
		record_tok("spm.no_dummy_prefix", ok, "plain encoding without metaspace lead");
	}

	n						= enc(&t, "\xe6\xb5\xaa", ids, 128);
	const int32_t *tail		= ids + (t.add_space_prefix ? 1 : 0);
	int			   tail_n	= n - (t.add_space_prefix ? 1 : 0);
	int			   ok_bytes = ids_match(tail, tail_n, (const int32_t[]){16, 17, 18}, 3);
	if (t.add_space_prefix)
		ok_bytes = ok_bytes && ids[0] == 7;
	char out[64];
	int	 len = tokenizer_decode(&t, tail, tail_n, out, (int)sizeof(out), NULL);
	record_tok("spm.byte_fallback_roundtrip",
			   ok_bytes && len == 3 && memcmp(out, "\xe6\xb5\xaa", 3) == 0, "");

	drop_vocab(&t, &g);
}

static const char *const tok_s[] = {
	"<unk>", "x", "y", "c", "<ab", "<abc",
};
static const int32_t typ_s[] = {TK_UNKNOWN, TK_NORMAL,	TK_NORMAL,
								TK_NORMAL,	TK_CONTROL, TK_CONTROL};

static void run_specials(void) {
	char path[400];
	snprintf(path, sizeof(path), "%s/tok_specials.gguf", tok_dir);
	tsg_build_vocab_file(path, "gpt2", NULL, 0, 0, tok_s, typ_s, ARRAY_LEN(tok_s));
	tokenizer t;
	gguf_ctx  g;
	load_vocab(&t, &g, path);
	char	detail[256];
	int		ok;
	int32_t ids[64];
	int		n = enc(&t, "<abcxy", ids, 64);
	char	out[64];
	int		len				 = tokenizer_decode(&t, ids, n, out, (int)sizeof(out), NULL);
	int		rt				 = len == 6 && memcmp(out, "<abcxy", 6) == 0;
	int		split_at_longest = n >= 1 && ids[0] == 5;
	ok						 = rt && split_at_longest;
	snprintf(detail, sizeof(detail), "'<abcxy' roundtrip=%d longest-match-split=%d (M-20 fixed)",
			 rt, split_at_longest);
	record_tok("special_prefix_collision_longest_match", ok, detail);

	n		= enc(&t, "x<aby", ids, 64);
	int ok2 = ids_match(ids, n, (const int32_t[]){1, 4, 2}, 3);
	snprintf(detail, sizeof(detail), "'x<aby' -> [x,<ab>,y]");
	record_tok("special_split_mid_text", ok2, detail);

	drop_vocab(&t, &g);
}

static const char *const tok_lm[] = {
	"<unk>", "x", "y", "z", "<|a", "<|ab", "w<|a",
};
static const int32_t typ_lm[] = {TK_UNKNOWN, TK_NORMAL,	 TK_NORMAL, TK_NORMAL,
								 TK_CONTROL, TK_CONTROL, TK_NORMAL};

static void run_specials_longest_match(void) {
	char path[400];
	snprintf(path, sizeof(path), "%s/tok_specials_longest.gguf", tok_dir);
	tsg_build_vocab_file(path, "gpt2", NULL, 0, 0, tok_lm, typ_lm, ARRAY_LEN(tok_lm));
	tokenizer t;
	gguf_ctx  g;
	load_vocab(&t, &g, path);
	char	detail[256];
	int32_t ids[64];

	int n = enc(&t, "<|abz", ids, 64);
	snprintf(detail, sizeof(detail), "'<|abz' -> %d token(s), first id=%d", n, n > 0 ? ids[0] : -1);
	record_tok("special_longest_match_beats_lower_id", n == 2 && ids[0] == 5 && ids[1] == 3,
			   detail);

	n				= enc(&t, "x<|ay", ids, 64);
	int short_alone = ids_match(ids, n, (const int32_t[]){1, 4, 2}, 3);
	snprintf(detail, sizeof(detail), "'x<|ay' -> [x,<|a>,y]");
	record_tok("special_shorter_token_still_matches", short_alone, detail);

	drop_vocab(&t, &g);
}

void run_tokenizer_tests(void) {
	snprintf(tok_dir, sizeof(tok_dir), "%s", synth_fixture_dir);
	run_vocab_a();
	run_vocab_d();
	run_vocab_f_roundtrip();
	run_spm_variant("spm_absent_llama", "llama", 0, 0, 1);
	run_spm_variant("spm_false_llama", "llama", 1, 0, 0);
	run_spm_variant("spm_absent_gemma", "gemma", 0, 0, 0);
	run_spm_variant("spm_true_gemma", "gemma", 1, 1, 1);
	run_specials();
	run_specials_longest_match();
}
