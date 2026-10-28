#include "gguf.h"
#include "json_helpers.h"
#include "log.h"

#include <sys/mman.h>

#define SAFETENSORS_LEN_BYTES 8

bool safetensors_detect(const void *data, size_t fsize) {
	if (fsize <= SAFETENSORS_LEN_BYTES)
		return false;
	uint64_t header_len;
	memcpy(&header_len, data, sizeof(header_len));
	return header_len < fsize - SAFETENSORS_LEN_BYTES &&
		   ((const char *)data)[SAFETENSORS_LEN_BYTES] == '{';
}

static bool dtype_to_type(const char *dtype, uint32_t *out) {
	static const struct {
		const char *name;
		uint32_t	type;
	} map[] = {{"F32", GGML_TYPE_F32}, {"F16", GGML_TYPE_F16}, {"BF16", GGML_TYPE_BF16}};
	for (size_t i = 0; i < ARRAY_LEN(map); i++) {
		if (!strcmp(dtype, map[i].name)) {
			*out = map[i].type;
			return true;
		}
	}
	return false;
}

static bool parse_tensor(gguf_tensor *t, const char *name, json_object *info, const uint8_t *base,
						 size_t data_size, int header_only) {
	json_object *shape	 = json_get_arr(info, "shape");
	json_object *offsets = json_get_arr(info, "data_offsets");
	uint32_t	 type;
	if (!shape || !offsets || json_object_array_length(offsets) != 2 ||
		!dtype_to_type(json_get_str(info, "dtype", ""), &type))
		return false;
	if (strlen(name) >= sizeof(t->name))
		return false;

	snprintf(t->name, sizeof(t->name), "%s", name);
	t->type		= type;
	size_t rank = json_object_array_length(shape);
	t->n_dims	= 0;
	for (size_t i = rank; i-- > 0;) {
		uint64_t dim = (uint64_t)json_object_get_int64(json_object_array_get_idx(shape, i));
		if (dim == 1 && rank > ARRAY_LEN(t->dims))
			continue;
		if (t->n_dims == ARRAY_LEN(t->dims))
			return false;
		t->dims[t->n_dims++] = dim;
	}
	if (t->n_dims == 0)
		t->dims[t->n_dims++] = 1;

	uint64_t begin = (uint64_t)json_object_get_int64(json_object_array_get_idx(offsets, 0));
	uint64_t end   = (uint64_t)json_object_get_int64(json_object_array_get_idx(offsets, 1));
	if (gguf_tensor_byte_size(t, &t->byte_size) != 0 || end < begin || end > data_size ||
		end - begin != t->byte_size)
		return false;
	t->offset = begin;
	t->data	  = header_only ? NULL : base + begin;
	return true;
}

static void release_mapping(int fd, void *map_ptr, size_t fsize) {
	if (map_ptr && map_ptr != MAP_FAILED)
		munmap(map_ptr, fsize);
	if (fd >= 0)
		close(fd);
}

status_code safetensors_parse(gguf_ctx *ctx, void *data, size_t fsize, int fd, void *map_ptr,
							  int header_only) {
	uint64_t header_len;
	memcpy(&header_len, data, sizeof(header_len));
	const uint8_t *base		 = (const uint8_t *)data + SAFETENSORS_LEN_BYTES + header_len;
	size_t		   data_size = fsize - SAFETENSORS_LEN_BYTES - header_len;

	const char	*err = NULL;
	json_object *root =
		json_parse_len((const char *)data + SAFETENSORS_LEN_BYTES, header_len, &err);
	if (!root || !json_object_is_type(root, json_type_object)) {
		ERROR("safetensors: invalid header: %s", err ? err : "not an object");
		json_object_put(root);
		release_mapping(fd, map_ptr, fsize);
		return ERR_FORMAT;
	}

	json_object *meta	   = json_get_obj(root, "__metadata__");
	size_t		 n_meta	   = meta ? (size_t)json_object_object_length(meta) : 0;
	size_t		 n_all	   = (size_t)json_object_object_length(root);
	size_t		 n_tensors = n_all - (meta ? 1 : 0);

	str_arena_init(&ctx->strs);
	gguf_tensor *ts = xcalloc(n_tensors ? n_tensors : 1, sizeof(*ts));
	size_t		 ti = 0;
	json_object_object_foreach(root, name, info) {
		if (!strcmp(name, "__metadata__"))
			continue;
		if (!parse_tensor(&ts[ti], name, info, base, data_size, header_only)) {
			DEBUG("safetensors: skipping tensor '%s' (unsupported dtype or bad layout)", name);
			memset(&ts[ti], 0, sizeof(ts[ti]));
			continue;
		}
		ti++;
	}

	ctx->tensors	 = ts;
	ctx->n_tensors	 = ti;
	ctx->n_kv		 = n_meta;
	ctx->kv_keys	 = xcalloc(n_meta ? n_meta : 1, sizeof(char *));
	ctx->kv_types	 = xcalloc(n_meta ? n_meta : 1, sizeof(uint32_t));
	ctx->kv_vals	 = xcalloc(n_meta ? n_meta : 1, sizeof(uint64_t));
	ctx->kv_strs	 = xcalloc(n_meta ? n_meta : 1, sizeof(gguf_str));
	ctx->kv_arr_type = xcalloc(n_meta ? n_meta : 1, sizeof(uint32_t));
	ctx->kv_arr_len	 = xcalloc(n_meta ? n_meta : 1, sizeof(uint64_t));
	ctx->kv_arr_data = xcalloc(n_meta ? n_meta : 1, sizeof(void *));
	size_t ki		 = 0;
	if (meta) {
		json_object_object_foreach(meta, key, val) {
			const char *text	  = json_object_get_string(val);
			size_t		len		  = strlen(text);
			ctx->kv_keys[ki]	  = str_arena_dup(&ctx->strs, key, strlen(key));
			ctx->kv_types[ki]	  = GGUF_TYPE_STRING;
			ctx->kv_strs[ki].len  = len;
			ctx->kv_strs[ki].data = str_arena_dup(&ctx->strs, text, len);
			ki++;
		}
	}
	json_object_put(root);

	ctx->data_start		  = base;
	ctx->data_file_offset = SAFETENSORS_LEN_BYTES + header_len;
	ctx->data_size		  = data_size;
	ctx->fd				  = fd;
	ctx->map			  = map_ptr;
	ctx->map_size		  = fsize;
	ctx->valid			  = 1;
	gguf_ctx_build_index(ctx);
	return OK;
}
