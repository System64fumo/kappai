#ifndef GGUF_H
#define GGUF_H

#include "common.h"

#define GGUF_MAGIC 0x46554747u
#define GGUF_VERSION 3u

typedef enum {
	GGUF_TYPE_U8 = 0,
	GGUF_TYPE_I8,
	GGUF_TYPE_U16,
	GGUF_TYPE_I16,
	GGUF_TYPE_U32,
	GGUF_TYPE_I32,
	GGUF_TYPE_F32,
	GGUF_TYPE_BOOL,
	GGUF_TYPE_STRING,
	GGUF_TYPE_ARRAY,
	GGUF_TYPE_U64,
	GGUF_TYPE_I64,
	GGUF_TYPE_F64,
} gguf_type;

typedef enum {
	GGML_TYPE_F32			= 0,
	GGML_TYPE_F16			= 1,
	GGML_TYPE_Q4_0			= 2,
	GGML_TYPE_Q4_1			= 3,
	GGML_TYPE_Q4_2			= 4,
	GGML_TYPE_Q4_3			= 5,
	GGML_TYPE_Q5_0			= 6,
	GGML_TYPE_Q5_1			= 7,
	GGML_TYPE_Q8_0			= 8,
	GGML_TYPE_Q8_1			= 9,
	GGML_TYPE_Q2_K			= 10,
	GGML_TYPE_Q3_K			= 11,
	GGML_TYPE_Q4_K			= 12,
	GGML_TYPE_Q5_K			= 13,
	GGML_TYPE_Q6_K			= 14,
	GGML_TYPE_Q8_K			= 15,
	GGML_TYPE_IQ2_XXS		= 16,
	GGML_TYPE_IQ2_XS			= 17,
	GGML_TYPE_IQ3_XXS		= 18,
	GGML_TYPE_IQ1_S			= 19,
	GGML_TYPE_IQ4_NL		= 20,
	GGML_TYPE_IQ3_S			= 21,
	GGML_TYPE_IQ2_S			= 22,
	GGML_TYPE_IQ4_XS		= 23,
	GGML_TYPE_IQ1_M			= 29,
	GGML_TYPE_BF16			= 30,

	/* 0x40+ = engine-internal repacked types */
	GGML_TYPE_IQ3_S_RE	= 0x42,
	GGML_TYPE_Q8_0_R8	= 0x43,
	GGML_TYPE_Q4_0_R8	= 0x44,
	GGML_TYPE_IQ3_S_RE8 = 0x45,
	GGML_TYPE_IQ4_NL_R8 = 0x46,
	GGML_TYPE_Q4_K_R8	= 0x47,

} ggml_type;

typedef struct {
	uint64_t	len;
	const char *data;
} gguf_str;

typedef struct {
	char		name[128];
	uint32_t	n_dims;
	uint64_t	dims[4];
	uint32_t	type;
	uint64_t	offset;
	const void *data;
} gguf_tensor;

typedef struct {
	int			 fd;
	void		*map;
	size_t		 map_size;
	int			 map_is_heap;
	int			 owns_tensor_data;
	int			 valid;
	char	   **kv_keys;
	uint32_t	*kv_types;
	uint64_t	*kv_vals;
	gguf_str	*kv_strs;
	uint32_t	*kv_arr_type;
	uint64_t	*kv_arr_len;
	void	   **kv_arr_data;
	size_t		 n_kv;
	gguf_tensor *tensors;
	size_t		 n_tensors;
	const void	*data_start;
	size_t		 data_size;
	uint64_t	 data_file_offset;
	struct {
		const char *name;
		size_t		idx;
		int			used;
	}	  *tensor_hash;
	size_t tensor_hash_cap;
	struct {
		const char *key;
		size_t		idx;
		int			used;
	}	  *kv_hash;
	size_t kv_hash_cap;
} gguf_ctx;

const char *ggml_type_name(uint32_t t);

status_code gguf_load(gguf_ctx *ctx, const char *path);

status_code gguf_load_metadata(gguf_ctx *ctx, const char *path);

status_code gguf_sparse_read_tensors(gguf_ctx *ctx, const char *path);

void gguf_free(gguf_ctx *ctx);

int gguf_tensor_byte_size(const gguf_tensor *t, size_t *out_size);

status_code gguf_get_i32(const gguf_ctx *c, const char *key, int32_t *out);
status_code gguf_get_f32(const gguf_ctx *c, const char *key, float *out);
status_code gguf_get_bool(const gguf_ctx *c, const char *key, int *out);
status_code gguf_get_str(const gguf_ctx *c, const char *key, const char **out);

status_code gguf_get_arr_i32(const gguf_ctx *c, const char *key, const int32_t **out,
							 size_t *out_count);
status_code gguf_get_arr_f32(const gguf_ctx *c, const char *key, const float **out,
							 size_t *out_count);
status_code gguf_get_arr_str(const gguf_ctx *c, const char *key, const char *const **out,
							 size_t *out_count);

const gguf_tensor *gguf_find_tensor(const gguf_ctx *c, const char *name);
size_t			   ggml_row_size(uint32_t type, size_t n);

int gguf_tensor_name_is_expert(const char *name);

void gguf_dump(const gguf_ctx *c, FILE *fp);

#endif