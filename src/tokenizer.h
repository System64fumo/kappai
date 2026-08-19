#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "common.h"
#include "gguf.h"
#include "profile.h"

enum {
	TOK_TYPE_NORMAL = 1,
	TOK_TYPE_UNKNOWN,
	TOK_TYPE_CONTROL,
	TOK_TYPE_USER_DEFINED,
	TOK_TYPE_UNUSED,
	TOK_TYPE_BYTE,
};

enum {
	TOK_PRE_GPT2 = 0,
	TOK_PRE_LLAMA3,
};

typedef struct {
	int32_t		id;
	const char *text;
	size_t		text_len;
	float		score;
	int			type;
} vocab_token;

typedef struct {
	vocab_token			  *tokens;
	size_t				   n_tokens;
	int32_t				   bos_id, eos_id, eot_id, pad_id, unk_id;
	int					   add_bos, add_eos;
	int					   is_sentencepiece;
	int					   pre_type;
	struct tok_hash_entry *hash;
	size_t				   hash_capacity;
	struct tok_hash_entry *merge_hash;
	size_t				   merge_hash_capacity;
	int					   has_merges;
	char				 **merge_keys;
	size_t				   n_merge_keys;
	int32_t				  *special_ids;
	size_t				   n_special_ids;
	int32_t				  *special_by_first_byte;
	size_t				   special_by_first_byte_off[257];
	void				  *bpe_pcs_cache;
	size_t				   bpe_pcs_cache_cap;
} tokenizer;

status_code tokenizer_init(tokenizer *t, const gguf_ctx *g);
void		tokenizer_free(tokenizer *t);
int			tokenizer_encode_with_specials(tokenizer *t, const char *text, int add_specials,
										   int32_t *out_ids, int max_out, profile *prof);
int			tokenizer_decode(tokenizer *t, const int32_t *ids, int n_ids, char *out, int max_out,
							 profile *prof);
size_t		tokenizer_token_decoded_len(const tokenizer *t, int32_t id);
int			tokenizer_token_count_for_bytes(const tokenizer *t, const int32_t *ids, int n,
											size_t max_bytes);
char	   *tokenizer_decode_prefix(const tokenizer *t, const int32_t *ids, int count);
int			tokenizer_is_eog(const tokenizer *t, int32_t id);
int32_t		tokenizer_find_token(const tokenizer *t, const char *text);

#endif