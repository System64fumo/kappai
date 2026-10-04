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

#define TSG_TOK_NORMAL 1
#define TSG_TOK_UNKNOWN 2
#define TSG_TOK_CONTROL 3
#define TSG_TOK_BYTE 6

#define TSG_CHAT_VOCAB 373
#define TSG_CHAT_N_MERGES 16

typedef struct {
	int		 dim;
	int		 n_heads;
	int		 n_kv_heads;
	int		 head_dim;
	int		 n_layers;
	int		 intermediate;
	int		 ctx;
	uint64_t seed;
	int		 tied;
	int		 output_copies_embd;
} tsg_llama_spec;

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

void	 tsg_seed(uint64_t s);
uint32_t tsg_rand(void);
void	 tsg_build_vocab_file(const char *path, const char *model_name, const char *pre,
							  int has_space_flag, int space_flag, const char *const *tokens,
							  const int32_t *types, size_t n);
void	 tsg_build_chat_llama(const char *path, const tsg_llama_spec *s);
void	 tsg_build_lfm2(const char *path, const tsg_lfm2_spec *s);
void	 tsg_build_glm_dsa(const char *path, const tsg_dsa_spec *s);

#endif
