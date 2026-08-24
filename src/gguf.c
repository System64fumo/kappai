#define _GNU_SOURCE
#include "gguf.h"
#include "log.h"
#include "profile.h"
#include <ctype.h>
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define GGUF_METADATA_INITIAL_CHUNK ((size_t)(4 << 20))
#define GGUF_METADATA_MAX_CHUNK ((size_t)(256 << 20))

#define GGUF_LOAD_CHUNK_TARGET_BYTES ((size_t)(8 * 1024 * 1024))
#define GGUF_LOAD_MAX_THREADS 64

static inline size_t align_up_bytes(size_t v, size_t a) {
	return (v + a - 1) & ~(a - 1);
}

typedef struct {
	uint32_t	type;
	const char *name;
	size_t		block_size;
	size_t		block_elems;
} ggml_type_info;

typedef struct {
	const uint8_t *p;
	const uint8_t *end;
} gguf_reader;

typedef struct {
	char	*key;
	uint32_t type;
	uint64_t ival;
	double	 fval;
	gguf_str str;
	uint32_t arr_type;
	uint64_t arr_len;
	void	*arr_data;
	gguf_str arr_str;
} kv_entry;

typedef struct {
	kv_entry *e;
	size_t	  n;
} kv_table;

typedef struct {
	const gguf_tensor *t;
	uint64_t		   file_off;
	size_t			   len;
	void			  *dst;
	size_t			   tidx;
	int				   plain;
} gguf_load_chunk;

typedef struct {
	gguf_ctx		*ctx;
	int				 plain_fd;
	int				 direct_fd;
	size_t			 align;
	_Atomic size_t	 next;
	gguf_load_chunk *chunks;
	size_t			 n_chunks;
	_Atomic int		 first_err;
	_Atomic size_t	 bytes_read;
	_Atomic size_t	 tensors_done;
	_Atomic size_t	*chunk_remaining;
} gguf_load_job;

static inline void gguf_ctx_init(gguf_ctx *ctx) {
	memset(ctx, 0, sizeof(*ctx));
	ctx->fd = -1;
}

static int gguf_open_ro(const char *path, size_t *fsize) {
	int fd = open(path, O_RDONLY);
	if (fd < 0)
		return -1;
	struct stat st;
	if (fstat(fd, &st) < 0) {
		close(fd);
		return -1;
	}
	*fsize = (size_t)st.st_size;
	return fd;
}

static const ggml_type_info *ggml_type_lookup(uint32_t t) {
	static const ggml_type_info table[] = {
		[GGML_TYPE_F32]		  = {GGML_TYPE_F32, "f32", 4, 1},
		[GGML_TYPE_F16]		  = {GGML_TYPE_F16, "f16", 2, 1},
		[GGML_TYPE_BF16]	  = {GGML_TYPE_BF16, "bf16", 2, 1},
		[GGML_TYPE_Q4_0]	  = {GGML_TYPE_Q4_0, "q4_0", 18, 32},
		[GGML_TYPE_Q4_1]	  = {GGML_TYPE_Q4_1, "q4_1", 20, 32},
		[GGML_TYPE_Q5_0]	  = {GGML_TYPE_Q5_0, "q5_0", 22, 32},
		[GGML_TYPE_Q5_1]	  = {GGML_TYPE_Q5_1, "q5_1", 24, 32},
		[GGML_TYPE_Q8_0]	  = {GGML_TYPE_Q8_0, "q8_0", 34, 32},
		[GGML_TYPE_Q8_1]	  = {GGML_TYPE_Q8_1, "q8_1", 36, 32},
		[GGML_TYPE_Q3_K]	  = {GGML_TYPE_Q3_K, "q3_K", 110, 256},
		[GGML_TYPE_Q4_K]	  = {GGML_TYPE_Q4_K, "q4_K", 144, 256},
		[GGML_TYPE_Q5_K]	  = {GGML_TYPE_Q5_K, "q5_K", 176, 256},
		[GGML_TYPE_Q6_K]	  = {GGML_TYPE_Q6_K, "q6_K", 210, 256},
		[GGML_TYPE_IQ4_NL]	  = {GGML_TYPE_IQ4_NL, "iq4_nl", 18, 32},
		[GGML_TYPE_IQ3_S]	  = {GGML_TYPE_IQ3_S, "iq3_s", 110, 256},
		[GGML_TYPE_IQ3_S_RE]  = {GGML_TYPE_IQ3_S_RE, "iq3_s_re", 134, 256},
		[GGML_TYPE_Q8_0_R8]	  = {GGML_TYPE_Q8_0_R8, "q8_0_r8", 34, 32},
		[GGML_TYPE_Q4_0_R8]	  = {GGML_TYPE_Q4_0_R8, "q4_0_r8", 18, 32},
		[GGML_TYPE_IQ3_S_RE8] = {GGML_TYPE_IQ3_S_RE8, "iq3_s_re8", 134, 256},
		[GGML_TYPE_IQ4_NL_R8] = {GGML_TYPE_IQ4_NL_R8, "iq4_nl_r8", 18, 32},
		[GGML_TYPE_Q4_K_R8]	  = {GGML_TYPE_Q4_K_R8, "q4_k_r8", 148, 256},
		[GGML_TYPE_Q5_K_R8]	  = {GGML_TYPE_Q5_K_R8, "q5_k_r8", 180, 256},
		[GGML_TYPE_Q6_K_R8]	  = {GGML_TYPE_Q6_K_R8, "q6_k_r8", 210, 256},
	};
	if (t >= ARRAY_LEN(table))
		return NULL;
	return table[t].name ? &table[t] : NULL;
}

const char *ggml_type_name(uint32_t t) {
	const ggml_type_info *info = ggml_type_lookup(t);
	return info ? info->name : "?";
}

static ptrdiff_t gguf_reader_left(const gguf_reader *r) {
	return (ptrdiff_t)(r->end - r->p);
}

static int gguf_reader_u8(gguf_reader *r, uint8_t *o) {
	if (gguf_reader_left(r) < (ptrdiff_t)1)
		return -1;
	*o = r->p[0];
	r->p += 1;
	return 0;
}

static int gguf_reader_u16(gguf_reader *r, uint16_t *o) {
	if (gguf_reader_left(r) < (ptrdiff_t)2)
		return -1;
	uint16_t v;
	memcpy(&v, r->p, 2);
	*o = le16toh(v);
	r->p += 2;
	return 0;
}

static int gguf_reader_u32(gguf_reader *r, uint32_t *o) {
	if (gguf_reader_left(r) < (ptrdiff_t)4)
		return -1;
	uint32_t v;
	memcpy(&v, r->p, 4);
	*o = le32toh(v);
	r->p += 4;
	return 0;
}

static int gguf_reader_u64(gguf_reader *r, uint64_t *o) {
	if (gguf_reader_left(r) < (ptrdiff_t)8)
		return -1;
	uint64_t v;
	memcpy(&v, r->p, 8);
	*o = le64toh(v);
	r->p += 8;
	return 0;
}

static int gguf_reader_f32(gguf_reader *r, float *o) {
	if (gguf_reader_left(r) < (ptrdiff_t)4)
		return -1;
	uint32_t v;
	memcpy(&v, r->p, 4);
	v = le32toh(v);
	memcpy(o, &v, 4);
	r->p += 4;
	return 0;
}

static int gguf_reader_f64(gguf_reader *r, double *o) {
	if (gguf_reader_left(r) < (ptrdiff_t)8)
		return -1;
	uint64_t v;
	memcpy(&v, r->p, 8);
	v = le64toh(v);
	memcpy(o, &v, 8);
	r->p += 8;
	return 0;
}

static int gguf_reader_str(gguf_reader *r, gguf_str *o) {
	uint64_t n;
	if (gguf_reader_u64(r, &n))
		return -1;
	if (gguf_reader_left(r) < (ptrdiff_t)n)
		return -1;
	o->len	= n;
	o->data = (const char *)r->p;
	r->p += n;
	return 0;
}

static int gguf_reader_bool(gguf_reader *r, int *o) {
	uint8_t v;
	if (gguf_reader_u8(r, &v))
		return -1;
	*o = v != 0;
	return 0;
}

static int parse_kv_value(gguf_reader *r, kv_entry *e, uint32_t type) {
	e->type = type;
	switch (type) {
	case GGUF_TYPE_U8: {
		uint8_t v;
		if (gguf_reader_u8(r, &v))
			return -1;
		e->ival = v;
		return 0;
	}
	case GGUF_TYPE_I8: {
		uint8_t v;
		if (gguf_reader_u8(r, &v))
			return -1;
		e->ival = (unsigned char)(int8_t)v;
		return 0;
	}
	case GGUF_TYPE_U16: {
		uint16_t v;
		if (gguf_reader_u16(r, &v))
			return -1;
		e->ival = v;
		return 0;
	}
	case GGUF_TYPE_I16: {
		uint16_t v;
		if (gguf_reader_u16(r, &v))
			return -1;
		e->ival = (int16_t)v;
		return 0;
	}
	case GGUF_TYPE_U32: {
		uint32_t v;
		if (gguf_reader_u32(r, &v))
			return -1;
		e->ival = v;
		return 0;
	}
	case GGUF_TYPE_I32: {
		uint32_t v;
		if (gguf_reader_u32(r, &v))
			return -1;
		e->ival = (int32_t)v;
		return 0;
	}
	case GGUF_TYPE_U64: {
		uint64_t v;
		if (gguf_reader_u64(r, &v))
			return -1;
		e->ival = v;
		return 0;
	}
	case GGUF_TYPE_I64: {
		uint64_t v;
		if (gguf_reader_u64(r, &v))
			return -1;
		e->ival = (int64_t)v;
		return 0;
	}
	case GGUF_TYPE_F32: {
		float v;
		if (gguf_reader_f32(r, &v))
			return -1;
		e->ival = 0;
		memcpy(&e->ival, &v, 4);
		return 0;
	}
	case GGUF_TYPE_F64: {
		double v;
		if (gguf_reader_f64(r, &v))
			return -1;
		e->fval = v;
		return 0;
	}
	case GGUF_TYPE_BOOL: {
		int v;
		if (gguf_reader_bool(r, &v))
			return -1;
		e->ival = v;
		return 0;
	}
	case GGUF_TYPE_STRING: {
		gguf_str s;
		if (gguf_reader_str(r, &s))
			return -1;
		char *owned = xmalloc(s.len + 1);
		memcpy(owned, s.data, s.len);
		owned[s.len] = '\0';
		e->str.len	 = s.len;
		e->str.data	 = owned;
		return 0;
	}
	case GGUF_TYPE_ARRAY: {
		uint32_t et;
		uint64_t n;
		if (gguf_reader_u32(r, &et))
			return -1;
		if (gguf_reader_u64(r, &n))
			return -1;
		e->arr_type = et;
		e->arr_len	= n;
		switch (et) {
		case GGUF_TYPE_U8:
		case GGUF_TYPE_I8:
		case GGUF_TYPE_BOOL:
			e->arr_data = xmalloc(n);
			for (uint64_t i = 0; i < n; i++) {
				uint8_t v;
				if (gguf_reader_u8(r, &v))
					return -1;
				((uint8_t *)e->arr_data)[i] = v;
			}
			break;
		case GGUF_TYPE_U16:
		case GGUF_TYPE_I16:
			e->arr_data = xmalloc(n * 2);
			for (uint64_t i = 0; i < n; i++) {
				uint16_t v;
				if (gguf_reader_u16(r, &v))
					return -1;
				((uint16_t *)e->arr_data)[i] = v;
			}
			break;
		case GGUF_TYPE_U32:
		case GGUF_TYPE_I32:
		case GGUF_TYPE_F32:
			e->arr_data = xmalloc(n * 4);
			for (uint64_t i = 0; i < n; i++) {
				uint32_t v;
				if (gguf_reader_u32(r, &v))
					return -1;
				((uint32_t *)e->arr_data)[i] = v;
			}
			break;
		case GGUF_TYPE_U64:
		case GGUF_TYPE_I64:
		case GGUF_TYPE_F64:
			e->arr_data = xmalloc(n * 8);
			for (uint64_t i = 0; i < n; i++) {
				uint64_t v;
				if (gguf_reader_u64(r, &v))
					return -1;
				((uint64_t *)e->arr_data)[i] = v;
			}
			break;
		case GGUF_TYPE_STRING: {
			char **arr = (char **)xmalloc(n * sizeof(char *));
			for (uint64_t i = 0; i < n; i++) {
				gguf_str s;
				if (gguf_reader_str(r, &s)) {
					for (uint64_t j = 0; j < i; j++)
						free(arr[j]);
					free(arr);
					return -1;
				}
				arr[i] = xmalloc(s.len + 1);
				memcpy(arr[i], s.data, s.len);
				arr[i][s.len] = '\0';
			}
			e->arr_data = (void *)arr;
			break;
		}
		default:
			return -1;
		}
		return 0;
	}
	default:
		return -1;
	}
}

static void free_kv(kv_entry *e, size_t n) {
	for (size_t i = 0; i < n; i++) {
		free(e[i].key);
		if (e[i].type == GGUF_TYPE_STRING) {
			free((void *)e[i].str.data);
		}
		if (e[i].type == GGUF_TYPE_ARRAY) {
			if (e[i].arr_type == GGUF_TYPE_STRING && e[i].arr_data) {
				char **arr = (char **)e[i].arr_data;
				for (uint64_t j = 0; j < e[i].arr_len; j++)
					free(arr[j]);
			}
			free(e[i].arr_data);
		}
	}
	free(e);
}

size_t ggml_row_size(uint32_t t, size_t n) {
	const ggml_type_info *info = ggml_type_lookup(t);
	if (!info)
		return 0;
	return ((n + info->block_elems - 1) / info->block_elems) * info->block_size;
}

int gguf_tensor_byte_size(const gguf_tensor *t, size_t *out_size) {
	if (t->n_dims == 0)
		return -1;
	const ggml_type_info *info = ggml_type_lookup(t->type);
	if (!info)
		return -1;
	if (t->dims[0] % info->block_elems != 0)
		return -1;
	size_t row = ggml_row_size(t->type, t->dims[0]);
	if (row == 0)
		return -1;
	size_t total = row;
	for (uint32_t d = 1; d < t->n_dims; d++) {
		uint64_t dims = t->dims[d];
		if (dims != 0 && dims > SIZE_MAX / total)
			return -1;
		total *= dims ? dims : 1;
	}
	*out_size = total;
	return 0;
}

static status_code gguf_parse_common(gguf_ctx *ctx, void *data, size_t fsize, int fd, void *map_ptr,
									 int header_only) {
	gguf_reader r = {(const uint8_t *)data, (const uint8_t *)data + fsize};

	uint32_t magic;
	uint32_t version;
	if (gguf_reader_u32(&r, &magic))
		goto bad;
	if (magic != GGUF_MAGIC)
		goto bad;
	if (gguf_reader_u32(&r, &version))
		goto bad;
	if (version != GGUF_VERSION)
		goto bad;
	if (gguf_reader_u64(&r, &ctx->n_tensors))
		goto bad;
	if (gguf_reader_u64(&r, &ctx->n_kv))
		goto bad;

	{
		size_t remaining = gguf_reader_left(&r) > 0 ? (size_t)gguf_reader_left(&r) : 0;
		if (ctx->n_kv > remaining / 13)
			goto bad;
		if (ctx->n_tensors > remaining / 25)
			goto bad;
	}

	kv_entry *kv = xcalloc(ctx->n_kv, sizeof(kv_entry));
	for (size_t i = 0; i < ctx->n_kv; i++) {
		gguf_str key;
		if (gguf_reader_str(&r, &key)) {
			free_kv(kv, i);
			goto bad;
		}
		kv[i].key = xmalloc(key.len + 1);
		memcpy(kv[i].key, key.data, key.len);
		kv[i].key[key.len] = '\0';
		uint32_t t;
		if (gguf_reader_u32(&r, &t)) {
			free_kv(kv, i + 1);
			goto bad;
		}
		if (parse_kv_value(&r, &kv[i], t)) {
			free_kv(kv, i + 1);
			goto bad;
		}
	}

	gguf_tensor *ts = xcalloc(ctx->n_tensors, sizeof(gguf_tensor));
	for (size_t i = 0; i < ctx->n_tensors; i++) {
		gguf_str name;
		if (gguf_reader_str(&r, &name))
			goto bad_kv_ts;
		size_t nl = MIN(name.len, sizeof(ts[i].name) - 1);
		memcpy(ts[i].name, name.data, nl);
		ts[i].name[nl] = '\0';
		if (gguf_reader_u32(&r, &ts[i].n_dims))
			goto bad_kv_ts;
		if (ts[i].n_dims > 4)
			goto bad_kv_ts;
		for (uint32_t d = 0; d < ts[i].n_dims; d++)
			if (gguf_reader_u64(&r, &ts[i].dims[d]))
				goto bad_kv_ts;
		if (gguf_reader_u32(&r, &ts[i].type))
			goto bad_kv_ts;
		if (gguf_reader_u64(&r, &ts[i].offset))
			goto bad_kv_ts;
	}

	size_t align = 32;
	for (size_t i = 0; i < ctx->n_kv; i++) {
		if (strcmp(kv[i].key, "general.alignment") != 0)
			continue;
		if (kv[i].type != GGUF_TYPE_U32 && kv[i].type != GGUF_TYPE_I32)
			goto bad_kv_ts;
		uint32_t a = (uint32_t)kv[i].ival;
		if (a == 0 || (a & (a - 1)) != 0)
			goto bad_kv_ts;
		align = a;
		break;
	}
	size_t cur	   = (size_t)(r.p - (const uint8_t *)data);
	size_t aligned = (cur + align - 1) & ~(align - 1);
	if (aligned > fsize)
		goto bad_kv_ts;
	ctx->data_start		  = (const uint8_t *)data + aligned;
	ctx->data_file_offset = (uint64_t)aligned;
	ctx->data_size		  = fsize - aligned;

	for (size_t i = 0; i < ctx->n_tensors; i++) {
		if (header_only) {
			ts[i].data = NULL;
			continue;
		}
		if (ts[i].offset > ctx->data_size)
			goto bad_kv_ts;
		size_t tsize;
		if (gguf_tensor_byte_size(&ts[i], &tsize) == 0)
			if (tsize > ctx->data_size - ts[i].offset)
				goto bad_kv_ts;
		ts[i].data = (const uint8_t *)ctx->data_start + ts[i].offset;
	}

	{
		size_t cap = 1;
		while (cap < ctx->n_tensors * 2)
			cap <<= 1;
		ctx->tensor_hash	 = xcalloc(cap, sizeof(*ctx->tensor_hash));
		ctx->tensor_hash_cap = cap;
		for (size_t i = 0; i < ctx->n_tensors; i++) {
			uint64_t h = fnv1a_str(ts[i].name) & (cap - 1);
			while (ctx->tensor_hash[h].used)
				h = (h + 1) & (cap - 1);
			ctx->tensor_hash[h].name = ts[i].name;
			ctx->tensor_hash[h].idx	 = i;
			ctx->tensor_hash[h].used = 1;
		}
	}

	ctx->kv_keys	 = (char **)xcalloc(ctx->n_kv, sizeof(char *));
	ctx->kv_types	 = xcalloc(ctx->n_kv, sizeof(uint32_t));
	ctx->kv_vals	 = xcalloc(ctx->n_kv, sizeof(uint64_t));
	ctx->kv_strs	 = xcalloc(ctx->n_kv, sizeof(gguf_str));
	ctx->kv_arr_type = xcalloc(ctx->n_kv, sizeof(uint32_t));
	ctx->kv_arr_len	 = xcalloc(ctx->n_kv, sizeof(uint64_t));
	ctx->kv_arr_data = (void **)xcalloc(ctx->n_kv, sizeof(void *));
	for (size_t i = 0; i < ctx->n_kv; i++) {
		ctx->kv_keys[i]		= kv[i].key;
		ctx->kv_types[i]	= kv[i].type;
		ctx->kv_vals[i]		= kv[i].ival;
		ctx->kv_strs[i]		= kv[i].str;
		ctx->kv_arr_type[i] = kv[i].arr_type;
		ctx->kv_arr_len[i]	= kv[i].arr_len;
		ctx->kv_arr_data[i] = kv[i].arr_data;
	}
	free(kv);

	{
		size_t cap = 1;
		while (cap < (ctx->n_kv * 2) + 1)
			cap <<= 1;
		ctx->kv_hash	 = xcalloc(cap, sizeof(*ctx->kv_hash));
		ctx->kv_hash_cap = cap;
		for (size_t i = 0; i < ctx->n_kv; i++) {
			uint64_t h	 = fnv1a_str(ctx->kv_keys[i]) & (cap - 1);
			int		 dup = 0;
			while (ctx->kv_hash[h].used) {
				if (strcmp(ctx->kv_hash[h].key, ctx->kv_keys[i]) == 0) {
					dup = 1;
					break;
				}
				h = (h + 1) & (cap - 1);
			}
			if (dup)
				continue;
			ctx->kv_hash[h].key	 = ctx->kv_keys[i];
			ctx->kv_hash[h].idx	 = i;
			ctx->kv_hash[h].used = 1;
		}
	}

	ctx->tensors  = ts;
	ctx->fd		  = fd;
	ctx->map	  = map_ptr;
	ctx->map_size = fsize;
	ctx->valid	  = 1;
	return OK;

bad_kv_ts:
	free_kv(kv, ctx->n_kv);
	free(ts);
bad:
	free(ctx->tensor_hash);
	free(ctx->kv_hash);
	if (map_ptr) {
		if (ctx->map_is_heap)
			free(map_ptr);
		else if (map_ptr != MAP_FAILED)
			munmap(map_ptr, fsize);
	}
	if (fd >= 0)
		close(fd);
	memset(ctx, 0, sizeof(*ctx));
	ctx->fd = -1;
	return ERR_FORMAT;
}

status_code gguf_load(gguf_ctx *ctx, const char *path) {
	gguf_ctx_init(ctx);

	size_t fsize;
	int	   fd = gguf_open_ro(path, &fsize);
	if (fd < 0)
		return ERR_IO;

	void *map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
	if (map == MAP_FAILED) {
		close(fd);
		return ERR_IO;
	}
	madvise(map, fsize, MADV_SEQUENTIAL);
#ifdef MADV_DONTDUMP
	madvise(map, fsize, MADV_DONTDUMP);
#endif

	return gguf_parse_common(ctx, map, fsize, fd, map, 0);
}

static int gguf_range_read(int plain_fd, int direct_fd, size_t align, uint64_t file_off, size_t len,
						   void *dst) {
	if (len == 0)
		return 0;

	if (direct_fd >= 0 && align > 0) {
		int dst_aligned = (((uintptr_t)dst & (align - 1)) == 0);
		int off_aligned = ((file_off & (align - 1)) == 0);
		int len_aligned = ((len & (align - 1)) == 0);

		if (dst_aligned && off_aligned && len_aligned) {
			size_t total = 0;
			while (total < len) {
				ssize_t n =
					pread(direct_fd, (char *)dst + total, len - total, (off_t)(file_off + total));
				if (n < 0) {
					if (errno == EINTR)
						continue;
					goto bounce_fallback;
				}
				if (n == 0)
					break;
				total += (size_t)n;
			}
			if (total >= len) {
				posix_fadvise(direct_fd, (off_t)file_off, (off_t)len, POSIX_FADV_DONTNEED);
				return 0;
			}
		}

	bounce_fallback: {
		uintptr_t am		  = ~((uintptr_t)align - 1);
		uint64_t  aligned_off = file_off & am;
		size_t	  head_slop	  = (size_t)(file_off - aligned_off);
		size_t	  aligned_len = (head_slop + len + align - 1) & ~(align - 1);

		void *bounce = NULL;
		if (posix_memalign(&bounce, align, aligned_len) == 0 && bounce) {
			size_t	 total	   = 0;
			int		 io_failed = 0;
			uint8_t *bp		   = bounce;
			while (total < aligned_len) {
				ssize_t n =
					pread(direct_fd, bp + total, aligned_len - total, (off_t)(aligned_off + total));
				if (n < 0) {
					if (errno == EINTR)
						continue;
					io_failed = 1;
					break;
				}
				if (n == 0)
					break;
				total += (size_t)n;
			}
			if (!io_failed && total >= head_slop + len) {
				memcpy(dst, bp + head_slop, len);
				free(bounce);
				posix_fadvise(direct_fd, (off_t)aligned_off, (off_t)aligned_len,
							  POSIX_FADV_DONTNEED);
				return 0;
			}
			free(bounce);
		}
	}
	}

	if (plain_fd < 0)
		return -1;
	size_t total = 0;
	while (total < len) {
		ssize_t n = pread(plain_fd, (char *)dst + total, len - total, (off_t)(file_off + total));
		if (n < 0) {
			if (errno == EINTR)
				continue;
			return -1;
		}
		if (n == 0)
			return -1;
		total += (size_t)n;
	}
	posix_fadvise(plain_fd, (off_t)file_off, (off_t)len, POSIX_FADV_DONTNEED);
	return 0;
}

static void *gguf_load_worker(void *arg) {
	gguf_load_job *j = (gguf_load_job *)arg;
	for (;;) {
		size_t i = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed);
		if (i >= j->n_chunks)
			break;
		if (atomic_load_explicit(&j->first_err, memory_order_relaxed) != OK)
			break;

		gguf_load_chunk *ch = &j->chunks[i];
		int rr = gguf_range_read(j->plain_fd, ch->plain ? -1 : j->direct_fd, j->align, ch->file_off,
								 ch->len, ch->dst);
		if (rr != 0) {
			atomic_fetch_sub_explicit(&j->chunk_remaining[ch->tidx], 1, memory_order_acq_rel);
			atomic_store(&j->first_err, ERR_IO);
			break;
		}
		atomic_fetch_add_explicit(&j->bytes_read, ch->len, memory_order_relaxed);
		if (atomic_fetch_sub_explicit(&j->chunk_remaining[ch->tidx], 1, memory_order_acq_rel) == 1)
			atomic_fetch_add_explicit(&j->tensors_done, 1, memory_order_relaxed);
	}
	return NULL;
}

static int gguf_affinity_thread_count(void) {
	cpu_set_t set;
	CPU_ZERO(&set);
	int n_cpu = 0;
	if (sched_getaffinity(0, sizeof(set), &set) == 0)
		n_cpu = CPU_COUNT(&set);
	if (n_cpu <= 0) {
		long online = sysconf(_SC_NPROCESSORS_ONLN);
		n_cpu		= online < 1 ? 1 : (int)online;
	}
	int n = n_cpu * 4;
	if (n < 8)
		n = 8;
	return n;
}

status_code gguf_load_metadata(gguf_ctx *ctx, const char *path) {
	gguf_ctx_init(ctx);

	int fd = open(path, O_RDONLY);
	if (fd < 0)
		return ERR_IO;

	struct stat st;
	if (fstat(fd, &st) < 0) {
		close(fd);
		return ERR_IO;
	}
	size_t real_fsize = (size_t)st.st_size;
	if (real_fsize == 0) {
		close(fd);
		return ERR_IO;
	}

	size_t		cap	 = MIN(GGUF_METADATA_INITIAL_CHUNK, real_fsize);
	void	   *buf	 = xmalloc(cap);
	size_t		have = 0;
	status_code s	 = ERR_FORMAT;

	for (;;) {
		while (have < cap) {
			ssize_t n = pread(fd, (char *)buf + have, cap - have, (off_t)have);
			if (n < 0) {
				free(buf);
				close(fd);
				return ERR_IO;
			}
			if (n == 0)
				break;
			have += (size_t)n;
		}

		gguf_ctx trial;
		memset(&trial, 0, sizeof(trial));
		s = gguf_parse_common(&trial, buf, have, -1, NULL, 1);
		if (s == OK) {
			*ctx = trial;
			break;
		}
		gguf_free(&trial);

		if (have >= real_fsize) {
			free(buf);
			close(fd);
			return ERR_FORMAT;
		}

		size_t grow	   = MIN(cap, GGUF_METADATA_MAX_CHUNK);
		size_t new_cap = MIN(cap + grow, real_fsize);
		if (new_cap <= cap) {
			free(buf);
			close(fd);
			return ERR_FORMAT;
		}
		void *nbuf = realloc(buf, new_cap);
		if (!nbuf) {
			free(buf);
			close(fd);
			return ERR_OUT_OF_MEMORY;
		}
		buf = nbuf;
		cap = new_cap;
	}

	for (size_t i = 0; i < ctx->n_tensors; i++)
		ctx->tensors[i].data = NULL;

	ctx->data_start = NULL;
	ctx->data_size	= real_fsize;

	posix_fadvise(fd, 0, (off_t)have, POSIX_FADV_DONTNEED);
	close(fd);
	ctx->fd = -1;
	free(buf);
	ctx->map		 = NULL;
	ctx->map_size	 = 0;
	ctx->map_is_heap = 0;
	return OK;
}

status_code gguf_sparse_read_tensors(gguf_ctx *ctx, const char *path) {
	int plain_fd = open(path, O_RDONLY);
	if (plain_fd < 0) {
		gguf_free(ctx);
		return ERR_IO;
	}

	int	   direct_fd = -1;
	size_t align	 = 0;
	{
		int fd = open(path, O_RDONLY | O_DIRECT);
		if (fd < 0) {
			DEBUG("gguf sparse-load: O_DIRECT unavailable (%s), using buffered reads",
				  strerror(errno));
		} else {
			struct stat st;
			long		blk = 4096;
			if (fstat(fd, &st) == 0 && st.st_blksize > 0)
				blk = st.st_blksize < 4096 ? st.st_blksize : 4096;
			void  *probe;
			size_t a = (size_t)blk;
			if (posix_memalign(&probe, a, a) != 0 || !probe) {
				close(fd);
				DEBUG("gguf sparse-load: O_DIRECT probe alloc failed, using buffered reads");
			} else {
				ssize_t rc = pread(fd, probe, a, 0);
				free(probe);
				if (rc < 0) {
					close(fd);
					DEBUG("gguf sparse-load: O_DIRECT probe failed (%s), using buffered reads",
						  strerror(errno));
				} else {
					direct_fd = fd;
					align	  = a;
					DEBUG("gguf sparse-load: O_DIRECT enabled (align=%zu)", align);
				}
			}
		}
	}

	status_code ret = OK;

	size_t n_load			= 0;
	size_t total_load_bytes = 0;
	for (size_t i = 0; i < ctx->n_tensors; i++) {
		gguf_tensor *t = &ctx->tensors[i];
		t->data		   = NULL;
		if (gguf_tensor_name_is_expert(t->name))
			continue;

		size_t tbytes;
		if (gguf_tensor_byte_size(t, &tbytes) != OK) {
			ERROR("gguf_load_sparse: cannot compute size of tensor '%s'", t->name);
			ret = ERR_FORMAT;
			break;
		}
		if (tbytes == 0)
			continue;
		if (t->offset > ctx->data_size || tbytes > ctx->data_size - t->offset) {
			ERROR("gguf_load_sparse: tensor '%s' extends past end of file", t->name);
			ret = ERR_FORMAT;
			break;
		}

		size_t alloc_align = align > 0 ? align : sizeof(void *);
		size_t pad		   = 0;
		if (align > 0) {
			uint64_t off64 = ctx->data_file_offset + t->offset;
			size_t	 slop  = (size_t)(off64 & (uint64_t)(align - 1));
			pad			   = align_up_bytes(slop + tbytes, align) - tbytes;
		}
		void *buf = NULL;
		if (posix_memalign(&buf, alloc_align, tbytes + pad) != 0) {
			ERROR("gguf_load_sparse: OOM allocating %zu bytes for tensor '%s'", tbytes, t->name);
			ret = ERR_OUT_OF_MEMORY;
			break;
		}
		t->data = buf;
		n_load++;
		total_load_bytes += tbytes;
	}

	if (ret == OK && n_load > 0) {
		DEBUG("gguf sparse-load: reading %zu non-expert tensors (%.1f MB)", n_load,
			  total_load_bytes / (1024.0 * 1024.0));
		uint64_t read_t0	 = time_us();
		size_t	 chunk_bytes = GGUF_LOAD_CHUNK_TARGET_BYTES;
		if (align > 0) {
			chunk_bytes = (chunk_bytes / align) * align;
			if (chunk_bytes == 0)
				chunk_bytes = align;
		}

		size_t n_chunks_cap = 0;
		for (size_t i = 0; i < ctx->n_tensors; i++) {
			gguf_tensor *t = &ctx->tensors[i];
			if (!t->data)
				continue;
			size_t tbytes;
			if (gguf_tensor_byte_size(t, &tbytes) != OK || tbytes == 0)
				continue;
			n_chunks_cap += (tbytes / chunk_bytes) + (align > 0 ? 2 : 1);
		}

		int has_experts = 0;
		for (size_t i = 0; i < ctx->n_tensors && !has_experts; i++)
			if (gguf_tensor_name_is_expert(ctx->tensors[i].name))
				has_experts = 1;

		gguf_load_chunk *chunks			 = xmalloc(n_chunks_cap * sizeof(*chunks));
		_Atomic size_t	*chunk_remaining = xcalloc(n_load, sizeof(*chunk_remaining));
		size_t			 n_chunks		 = 0;
		size_t			 load_idx		 = 0;
		struct stat		 fst;
		uint64_t file_end = fstat(plain_fd, &fst) == 0 ? (uint64_t)fst.st_size
													   : (ctx->data_file_offset + ctx->data_size);
		for (size_t i = 0; i < ctx->n_tensors; i++) {
			gguf_tensor *t = &ctx->tensors[i];
			if (!t->data)
				continue;
			size_t tbytes;
			if (gguf_tensor_byte_size(t, &tbytes) != OK || tbytes == 0)
				continue;

			uint64_t base_off		 = ctx->data_file_offset + t->offset;
			size_t	 off			 = 0;
			size_t	 n_tensor_chunks = 0;
			if (align > 0) {
				uint64_t a_base = base_off & ~(uint64_t)(align - 1);
				size_t	 slop	= (size_t)(base_off - a_base);
				size_t	 need	= align_up_bytes(slop + tbytes, align);
				uint64_t limit	= file_end - a_base;
				if (need > limit)
					need = (size_t)limit;
				size_t roff = 0;
				while (roff < need) {
					size_t piece = need - roff;
					if (piece > chunk_bytes)
						piece = chunk_bytes;
					int plain				  = (piece & (align - 1)) != 0;
					chunks[n_chunks].t		  = t;
					chunks[n_chunks].file_off = a_base + roff;
					chunks[n_chunks].len	  = piece;
					chunks[n_chunks].dst	  = (char *)t->data + roff;
					chunks[n_chunks].tidx	  = load_idx;
					chunks[n_chunks].plain	  = plain;
					n_chunks++;
					n_tensor_chunks++;
					roff += piece;
				}
				chunk_remaining[load_idx] = n_tensor_chunks;
				load_idx++;
				continue;
			}
			while (off < tbytes) {
				size_t piece = tbytes - off;
				if (piece > chunk_bytes)
					piece = chunk_bytes;
				chunks[n_chunks].t		  = t;
				chunks[n_chunks].file_off = base_off + off;
				chunks[n_chunks].len	  = piece;
				chunks[n_chunks].dst	  = (char *)t->data + off;
				chunks[n_chunks].tidx	  = load_idx;
				chunks[n_chunks].plain	  = 0;
				n_chunks++;
				n_tensor_chunks++;
				off += piece;
			}
			chunk_remaining[load_idx] = n_tensor_chunks;
			load_idx++;
		}

		gguf_load_job job = {
			.ctx			 = ctx,
			.plain_fd		 = plain_fd,
			.direct_fd		 = direct_fd,
			.align			 = align,
			.chunks			 = chunks,
			.n_chunks		 = n_chunks,
			.chunk_remaining = chunk_remaining,
		};
		atomic_store(&job.next, 0);
		atomic_store(&job.first_err, OK);
		atomic_store(&job.bytes_read, 0);
		atomic_store(&job.tensors_done, 0);

		int n_threads = gguf_affinity_thread_count();
		if (n_threads > GGUF_LOAD_MAX_THREADS)
			n_threads = GGUF_LOAD_MAX_THREADS;
		if ((size_t)n_threads > n_chunks)
			n_threads = n_chunks > 0 ? (int)n_chunks : 1;
		if (n_threads < 1)
			n_threads = 1;

		pthread_t threads[GGUF_LOAD_MAX_THREADS];
		for (int t = 0; t < n_threads; t++) {
			if (pthread_create(&threads[t], NULL, gguf_load_worker, &job) != 0) {
				threads[t] = 0;
				n_threads  = t;
				break;
			}
		}

		const char *load_label = has_experts ? "Loading non expert weights" : "Loading weights";
		progress	prog;
		progress_start(&prog, load_label, (uint64_t)n_load);
		for (;;) {
			size_t done = atomic_load(&job.tensors_done);
			progress_update(&prog, (uint64_t)done);
			if (done >= n_load)
				break;
			struct timespec ts = {.tv_sec = 0, .tv_nsec = 20 * 1000 * 1000};
			nanosleep(&ts, NULL);
		}

		for (int t = 0; t < n_threads; t++) {
			if (threads[t])
				pthread_join(threads[t], NULL);
		}
		progress_finish(&prog);

		int err = atomic_load(&job.first_err);
		if (err != OK)
			ret = err;

		if (err == OK && align > 0) {
			for (size_t i = 0; i < ctx->n_tensors; i++) {
				gguf_tensor *t = &ctx->tensors[i];
				if (!t->data)
					continue;
				size_t tbytes;
				if (gguf_tensor_byte_size(t, &tbytes) != OK || tbytes == 0)
					continue;
				uint64_t off64 = ctx->data_file_offset + t->offset;
				size_t	 slop  = (size_t)(off64 & (uint64_t)(align - 1));
				if (slop)
					memmove(t->data, (char *)t->data + slop, tbytes);
			}
		}

		if (err == OK) {
			double elapsed_ms = (time_us() - read_t0) / 1000.0;
			double mb		  = total_load_bytes / (1024.0 * 1024.0);
			INFO("gguf sparse-load: %zu tensors, %.1f MB in %.0f ms (%.0f MB/s)", n_load, mb,
				 elapsed_ms, elapsed_ms > 0 ? mb / (elapsed_ms / 1000.0) : 0.0);
		}

		free(chunks);
		free(chunk_remaining);
	}

	if (direct_fd >= 0)
		close(direct_fd);
	close(plain_fd);

	if (ret != OK) {
		for (size_t i = 0; i < ctx->n_tensors; i++)
			free((void *)ctx->tensors[i].data);
		ctx->n_tensors = 0;
		gguf_free(ctx);
		return ret;
	}

	ctx->owns_tensor_data = 1;
	return OK;
}

void gguf_free(gguf_ctx *ctx) {
	if (ctx && ctx->valid && ctx->owns_tensor_data) {
		for (size_t i = 0; i < ctx->n_tensors; i++)
			free((void *)ctx->tensors[i].data);
	}
	if (!ctx || !ctx->valid)
		return;
	for (size_t i = 0; i < ctx->n_kv; i++) {
		free(ctx->kv_keys[i]);
		if (ctx->kv_types[i] == GGUF_TYPE_STRING) {
			free((void *)ctx->kv_strs[i].data);
		}
		if (ctx->kv_types[i] == GGUF_TYPE_ARRAY) {
			if (ctx->kv_arr_type[i] == GGUF_TYPE_STRING && ctx->kv_arr_data[i]) {
				char **arr = (char **)ctx->kv_arr_data[i];
				for (uint64_t j = 0; j < ctx->kv_arr_len[i]; j++)
					free(arr[j]);
			}
			free(ctx->kv_arr_data[i]);
		}
	}
	free((void *)ctx->kv_keys);
	free(ctx->kv_types);
	free(ctx->kv_vals);
	free(ctx->kv_strs);
	free(ctx->kv_arr_type);
	free(ctx->kv_arr_len);
	free((void *)ctx->kv_arr_data);
	free(ctx->tensors);
	free(ctx->tensor_hash);
	free(ctx->kv_hash);
	if (ctx->map) {
		if (ctx->map_is_heap)
			free(ctx->map);
		else if (ctx->map != MAP_FAILED)
			munmap(ctx->map, ctx->map_size);
	}
	if (ctx->fd >= 0)
		close(ctx->fd);
	memset(ctx, 0, sizeof(*ctx));
	ctx->fd = -1;
}

static ptrdiff_t find_kv(const gguf_ctx *c, const char *key) {
	if (!c->kv_hash)
		return -1;
	uint64_t h = fnv1a_str(key) & (c->kv_hash_cap - 1);
	while (c->kv_hash[h].used) {
		if (strcmp(c->kv_hash[h].key, key) == 0)
			return (ptrdiff_t)c->kv_hash[h].idx;
		h = (h + 1) & (c->kv_hash_cap - 1);
	}
	return -1;
}

status_code gguf_get_i32(const gguf_ctx *c, const char *k, int32_t *o) {
	ptrdiff_t i = find_kv(c, k);
	if (i < 0)
		return ERR_NOT_FOUND;
	if (c->kv_types[i] == GGUF_TYPE_I32) {
		*o = (int32_t)c->kv_vals[i];
		return OK;
	}
	if (c->kv_types[i] == GGUF_TYPE_U32) {
		*o = (int32_t)(uint32_t)c->kv_vals[i];
		return OK;
	}
	return ERR_INVALID_ARG;
}

status_code gguf_get_f32(const gguf_ctx *c, const char *k, float *o) {
	ptrdiff_t i = find_kv(c, k);
	if (i < 0)
		return ERR_NOT_FOUND;
	if (c->kv_types[i] == GGUF_TYPE_F32) {
		float v;
		memcpy(&v, &c->kv_vals[i], 4);
		*o = v;
		return OK;
	}
	return ERR_INVALID_ARG;
}

status_code gguf_get_bool(const gguf_ctx *c, const char *k, int *o) {
	ptrdiff_t i = find_kv(c, k);
	if (i < 0)
		return ERR_NOT_FOUND;
	if (c->kv_types[i] == GGUF_TYPE_BOOL) {
		*o = (int)c->kv_vals[i];
		return OK;
	}
	return ERR_INVALID_ARG;
}

status_code gguf_get_str(const gguf_ctx *c, const char *k, const char **o) {
	ptrdiff_t i = find_kv(c, k);
	if (i < 0)
		return ERR_NOT_FOUND;
	if (c->kv_types[i] == GGUF_TYPE_STRING) {
		*o = c->kv_strs[i].data;
		return OK;
	}
	return ERR_INVALID_ARG;
}

static status_code gguf_get_arr_raw(const gguf_ctx *c, const char *k, uint32_t elem_type,
									const void **o, size_t *out_count) {
	ptrdiff_t i = find_kv(c, k);
	if (i < 0)
		return ERR_NOT_FOUND;
	if (c->kv_types[i] != GGUF_TYPE_ARRAY)
		return ERR_INVALID_ARG;
	if (c->kv_arr_type[i] != elem_type)
		return ERR_INVALID_ARG;
	*o = c->kv_arr_data[i];
	if (out_count)
		*out_count = (size_t)c->kv_arr_len[i];
	return OK;
}

status_code gguf_get_arr_i32(const gguf_ctx *c, const char *k, const int32_t **o,
							 size_t *out_count) {
	const void *raw = NULL;
	status_code st	= gguf_get_arr_raw(c, k, GGUF_TYPE_I32, &raw, out_count);
	if (st == OK)
		*o = (const int32_t *)raw;
	return st;
}

status_code gguf_get_arr_f32(const gguf_ctx *c, const char *k, const float **o, size_t *out_count) {
	const void *raw = NULL;
	status_code st	= gguf_get_arr_raw(c, k, GGUF_TYPE_F32, &raw, out_count);
	if (st == OK)
		*o = (const float *)raw;
	return st;
}

status_code gguf_get_arr_str(const gguf_ctx *c, const char *k, const char *const **o,
							 size_t *out_count) {
	const void *raw = NULL;
	status_code st	= gguf_get_arr_raw(c, k, GGUF_TYPE_STRING, &raw, out_count);
	if (st == OK)
		*o = (const char *const *)raw;
	return st;
}

const gguf_tensor *gguf_find_tensor(const gguf_ctx *c, const char *name) {
	if (!c->tensor_hash)
		return NULL;
	uint64_t h = fnv1a_str(name) & (c->tensor_hash_cap - 1);
	while (c->tensor_hash[h].used) {
		if (strcmp(c->tensor_hash[h].name, name) == 0)
			return &c->tensors[c->tensor_hash[h].idx];
		h = (h + 1) & (c->tensor_hash_cap - 1);
	}
	return NULL;
}

static const char *type_name(uint32_t t) {
	static const char *names[] = {
		[GGUF_TYPE_U8] = "u8",	   [GGUF_TYPE_I8] = "i8",	  [GGUF_TYPE_U16] = "u16",
		[GGUF_TYPE_I16] = "i16",   [GGUF_TYPE_U32] = "u32",	  [GGUF_TYPE_I32] = "i32",
		[GGUF_TYPE_U64] = "u64",   [GGUF_TYPE_I64] = "i64",	  [GGUF_TYPE_F32] = "f32",
		[GGUF_TYPE_F64] = "f64",   [GGUF_TYPE_BOOL] = "bool", [GGUF_TYPE_STRING] = "str",
		[GGUF_TYPE_ARRAY] = "arr",
	};
	if (t >= ARRAY_LEN(names) || !names[t])
		return "?";
	return names[t];
}

int gguf_tensor_name_is_expert(const char *name) {
	if (strstr(name, "ffn_gate_exps") || strstr(name, "ffn_up_exps") ||
		strstr(name, "ffn_down_exps") || strstr(name, "ffn_gate_up_exps"))
		return 1;

	static const char *bases[] = {"ffn_gate.", "ffn_up.", "ffn_down."};
	for (size_t i = 0; i < ARRAY_LEN(bases); i++) {
		const char *p = strstr(name, bases[i]);
		if (!p)
			continue;
		const char *digits = p + strlen(bases[i]);
		if (!isdigit((unsigned char)digits[0]))
			continue;
		const char *q = digits;
		while (isdigit((unsigned char)*q))
			q++;
		if (strcmp(q, ".weight") == 0)
			return 1;
	}
	return 0;
}

void gguf_dump(const gguf_ctx *c, FILE *fp) {
	fprintf(fp, "=== GGUF dump ===\n");
	fprintf(fp, "n_kv=%zu n_tensors=%zu data_size=%zu\n", c->n_kv, c->n_tensors, c->data_size);
	fprintf(fp, "--- metadata ---\n");
	for (size_t i = 0; i < c->n_kv; i++) {
		fprintf(fp, "  [%3zu] %-40s : %-5s = ", i, c->kv_keys[i], type_name(c->kv_types[i]));
		switch (c->kv_types[i]) {
		case GGUF_TYPE_U8:
		case GGUF_TYPE_BOOL:
			fprintf(fp, "%u", (unsigned)c->kv_vals[i]);
			break;
		case GGUF_TYPE_I8:
			fprintf(fp, "%d", (int)(int8_t)c->kv_vals[i]);
			break;
		case GGUF_TYPE_U16:
			fprintf(fp, "%u", (unsigned)(uint16_t)c->kv_vals[i]);
			break;
		case GGUF_TYPE_I16:
			fprintf(fp, "%d", (int)(int16_t)c->kv_vals[i]);
			break;
		case GGUF_TYPE_U32:
			fprintf(fp, "%u", (unsigned)(uint32_t)c->kv_vals[i]);
			break;
		case GGUF_TYPE_I32:
			fprintf(fp, "%d", (int)(int32_t)c->kv_vals[i]);
			break;
		case GGUF_TYPE_U64:
			fprintf(fp, "%llu", (unsigned long long)c->kv_vals[i]);
			break;
		case GGUF_TYPE_I64:
			fprintf(fp, "%lld", (long long)(int64_t)c->kv_vals[i]);
			break;
		case GGUF_TYPE_F32: {
			float v;
			memcpy(&v, &c->kv_vals[i], 4);
			fprintf(fp, "%g", v);
			break;
		}
		case GGUF_TYPE_F64:
			fprintf(fp, "%g", *(double *)&c->kv_vals[i]);
			break;
		case GGUF_TYPE_STRING:
			fprintf(fp, "\"%.*s\"", (int)c->kv_strs[i].len, c->kv_strs[i].data);
			break;
		case GGUF_TYPE_ARRAY: {
			const char *et = type_name(c->kv_arr_type[i]);
			fprintf(fp, "[%s x %llu]", et, (unsigned long long)c->kv_arr_len[i]);
			if (c->kv_arr_type[i] == GGUF_TYPE_STRING && c->kv_arr_len[i] <= 8) {
				char **arr = (char **)c->kv_arr_data[i];
				fprintf(fp, " {");
				for (uint64_t j = 0; j < c->kv_arr_len[i]; j++)
					fprintf(fp, " \"%s\"", arr[j]);
				fprintf(fp, " }");
			} else if (c->kv_arr_type[i] == GGUF_TYPE_F32 && c->kv_arr_len[i] <= 8) {
				float *arr = c->kv_arr_data[i];
				fprintf(fp, " {");
				for (uint64_t j = 0; j < c->kv_arr_len[i]; j++)
					fprintf(fp, " %g", arr[j]);
				fprintf(fp, " }");
			} else if (c->kv_arr_type[i] == GGUF_TYPE_I32 && c->kv_arr_len[i] <= 8) {
				int32_t *arr = c->kv_arr_data[i];
				fprintf(fp, " {");
				for (uint64_t j = 0; j < c->kv_arr_len[i]; j++)
					fprintf(fp, " %d", arr[j]);
				fprintf(fp, " }");
			}
			break;
		}
		default:
			break;
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "--- tensors ---\n");
	for (size_t i = 0; i < c->n_tensors; i++) {
		const gguf_tensor *t  = &c->tensors[i];
		const char		  *tn = ggml_type_name(t->type);
		if (tn[0] == '?') {
			fprintf(fp, "  [%3zu] %-40s : type=%u dims=[", i, t->name, t->type);
		} else {
			fprintf(fp, "  [%3zu] %-40s : %-5s dims=[", i, t->name, tn);
		}
		for (uint32_t d = 0; d < t->n_dims; d++) {
			fprintf(fp, "%s%llu", d ? "," : "", (unsigned long long)t->dims[d]);
		}
		fprintf(fp, "] offset=%llu\n", (unsigned long long)t->offset);
	}
}