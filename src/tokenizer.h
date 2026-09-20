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
	TOK_PRE_QWEN35,
};

typedef struct {
	int32_t		id;
	const char *text;
	uint32_t	text_len;
	float		score;
	uint8_t		type;
} vocab_token;

typedef struct tok_hash_entry {
	const char *key;
	uint32_t	key_len;
	int32_t		id;
} tok_hash_entry;

typedef struct {
	vocab_token			  *tokens;
	struct tok_hash_entry *hash;
	struct tok_hash_entry *merge_hash;
	char				 **merge_keys;
	int32_t				  *special_ids;
	int32_t				  *special_by_first_byte;
	void				  *bpe_work;
	char				  *bpe_arena;
	char				  *bpe_sp_text;
	void				  *bpe_pcs_cache;
	str_arena			   merge_pool;
	uint32_t			   n_tokens;
	int32_t				   bos_id, eos_id, eot_id, pad_id, unk_id;
	uint32_t			   n_byte_fallback;
	uint32_t			   hash_capacity;
	uint32_t			   merge_hash_capacity;
	uint32_t			   n_merge_keys;
	uint32_t			   n_special_ids;
	uint32_t			   n_special_first_bytes;
	uint32_t			   bpe_work_cap;
	uint32_t			   bpe_arena_cap;
	uint32_t			   bpe_sp_cap;
	uint32_t			   bpe_pcs_cache_cap;
	int32_t				   byte_fallback_ids[256];
	uint32_t			   special_by_first_byte_off[257];
	bool				   add_bos, add_eos;
	bool				   is_sentencepiece;
	bool				   add_space_prefix;
	uint8_t				   pre_type;
	bool				   has_merges;
	uint8_t				   special_first_byte_bitmap[32];
	uint8_t				   special_first_bytes[256];
} tokenizer;

status_code tokenizer_init(tokenizer *t, const gguf_ctx *g);
void		tokenizer_free(tokenizer *t);
int			tokenizer_encode_with_specials(tokenizer *t, const char *text, int add_specials,
										   int32_t *out_ids, int max_out, profile *prof);
int tokenizer_bpe_encode(tokenizer *t, const char *text, size_t len, int32_t *out_ids, int max_out,
						 int *n_out);
int tokenizer_decode(tokenizer *t, const int32_t *ids, int n_ids, char *out, int max_out,
					 profile *prof);
size_t	tokenizer_token_decoded_len(const tokenizer *t, int32_t id);
int		tokenizer_token_count_for_bytes(const tokenizer *t, const int32_t *ids, int n,
										size_t max_bytes);
int		tokenizer_is_eog(const tokenizer *t, int32_t id);
int32_t tokenizer_find_token(const tokenizer *t, const char *text);
int		tokenizer_starts_with_special(const tokenizer *t, const char *s);

#endif