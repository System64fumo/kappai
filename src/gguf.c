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
	if ((uint64_t)gguf_reader_left(r) < n)
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

static int gguf_arr_len_ok(const gguf_reader *r, uint64_t n, size_t elem_size) {
	if (elem_size == 0 || n > (uint64_t)(SIZE_MAX / elem_size))
		return 0;
	return n * (uint64_t)elem_size <= (uint64_t)gguf_reader_left(r);
}

static int parse_kv_value(gguf_reader *r, kv_entry *e, uint32_t type, str_arena *sa) {
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
		char *owned = str_arena_dup(sa, s.data, s.len);
		e->str.len	= s.len;
		e->str.data = owned;
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
			if (!gguf_arr_len_ok(r, n, 1))
				return -1;
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
			if (!gguf_arr_len_ok(r, n, 2))
				return -1;
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
			if (!gguf_arr_len_ok(r, n, 4))
				return -1;
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
			if (!gguf_arr_len_ok(r, n, 8))
				return -1;
			e->arr_data = xmalloc(n * 8);
			for (uint64_t i = 0; i < n; i++) {
				uint64_t v;
				if (gguf_reader_u64(r, &v))
					return -1;
				((uint64_t *)e->arr_data)[i] = v;
			}
			break;
		case GGUF_TYPE_STRING: {
			if (!gguf_arr_len_ok(r, n, sizeof(char *)))
				return -1;
			char **arr = (char **)xmalloc(n * sizeof(char *));
			for (uint64_t i = 0; i < n; i++) {
				gguf_str s;
				if (gguf_reader_str(r, &s)) {
					free(arr);
					return -1;
				}
				arr[i] = str_arena_dup(sa, s.data, s.len);
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
		if (e[i].type == GGUF_TYPE_ARRAY)
			free(e[i].arr_data);
	}
	free(e);
}

const uint32_t ggml_iq3s_grid[512] = {
	0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
	0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
	0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
	0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
	0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
	0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
	0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
	0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
	0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
	0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
	0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
	0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
	0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
	0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
	0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
	0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
	0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
	0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
	0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
	0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
	0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
	0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
	0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
	0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
	0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
	0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
	0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
	0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
	0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
	0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
	0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
	0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
	0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
	0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
	0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
	0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
	0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
	0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
	0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
	0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
	0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
	0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
	0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
	0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
	0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
	0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
	0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
	0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
	0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
	0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
	0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
	0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
	0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
	0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
	0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
	0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
	0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
	0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
	0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
	0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
	0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
	0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
	0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
	0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

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

static ptrdiff_t strtab_find(const gguf_strtab_entry *table, size_t cap, const char *key) {
	if (!table)
		return -1;
	uint64_t h = fnv1a_str(key) & (cap - 1);
	while (table[h].used) {
		if (strcmp(table[h].key, key) == 0)
			return (ptrdiff_t)table[h].idx;
		h = (h + 1) & (cap - 1);
	}
	return -1;
}

static void strtab_build(gguf_strtab_entry **out_table, size_t *out_cap, size_t count,
						 size_t		 min_cap, const char *(*key_of)(const void *ctx, size_t i),
						 const void *ctx, const char *dup_warn_label) {
	size_t cap = 1;
	while (cap < min_cap)
		cap <<= 1;
	gguf_strtab_entry *table = xcalloc(cap, sizeof(*table));
	for (size_t i = 0; i < count; i++) {
		const char *key = key_of(ctx, i);
		uint64_t	h	= fnv1a_str(key) & (cap - 1);
		int			dup = 0;
		while (table[h].used) {
			if (strcmp(table[h].key, key) == 0) {
				dup = 1;
				break;
			}
			h = (h + 1) & (cap - 1);
		}
		if (dup) {
			WARN("gguf: duplicate %s '%s' ignored", dup_warn_label, key);
			continue;
		}
		table[h].key  = key;
		table[h].idx  = i;
		table[h].used = 1;
	}
	*out_table = table;
	*out_cap   = cap;
}

static const char *tensor_name_at(const void *ctx, size_t i) {
	return ((const gguf_tensor *)ctx)[i].name;
}

static const char *kv_key_at(const void *ctx, size_t i) {
	return ((const char *const *)ctx)[i];
}

static status_code gguf_parse_common(gguf_ctx *ctx, void *data, size_t fsize, int fd, void *map_ptr,
									 int header_only) {
	gguf_reader r = {(const uint8_t *)data, (const uint8_t *)data + fsize};
	str_arena_init(&ctx->strs);

	uint32_t magic;
	uint32_t version;
	if (gguf_reader_u32(&r, &magic))
		goto bad;
	if (magic != GGUF_MAGIC)
		goto bad;
	if (gguf_reader_u32(&r, &version))
		goto bad;
	if (version != 2u && version != GGUF_VERSION)
		goto bad;
	if (version != GGUF_VERSION)
		DEBUG("gguf: accepting v%u header (layout identical to v%u)", version, GGUF_VERSION);
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
		kv[i].key = str_arena_dup(&ctx->strs, key.data, key.len);
		uint32_t t;
		if (gguf_reader_u32(&r, &t)) {
			free_kv(kv, i + 1);
			goto bad;
		}
		if (parse_kv_value(&r, &kv[i], t, &ctx->strs)) {
			free_kv(kv, i + 1);
			goto bad;
		}
	}

	gguf_tensor *ts = xcalloc(ctx->n_tensors, sizeof(gguf_tensor));
	for (size_t i = 0; i < ctx->n_tensors; i++) {
		gguf_str name;
		if (gguf_reader_str(&r, &name))
			goto bad_kv_ts;
		if (name.len >= sizeof(ts[i].name)) {
			ERROR("gguf: tensor name too long (%llu bytes, max %zu)", (unsigned long long)name.len,
				  sizeof(ts[i].name) - 1);
			goto bad_kv_ts;
		}
		memcpy(ts[i].name, name.data, name.len);
		ts[i].name[name.len] = '\0';
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
		size_t tsize;
		if (gguf_tensor_byte_size(&ts[i], &tsize) != 0) {
			ERROR("gguf: cannot size tensor '%s' (unknown type or bad dims)", ts[i].name);
			goto bad_kv_ts;
		}
		if (ts[i].offset > ctx->data_size || tsize > ctx->data_size - ts[i].offset)
			goto bad_kv_ts;
		ts[i].data = (const uint8_t *)ctx->data_start + ts[i].offset;
	}

	strtab_build(&ctx->tensor_hash, &ctx->tensor_hash_cap, ctx->n_tensors, ctx->n_tensors * 2,
				 tensor_name_at, ts, "tensor name");

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

	strtab_build(&ctx->kv_hash, &ctx->kv_hash_cap, ctx->n_kv, ctx->n_kv * 2 + 1, kv_key_at,
				 ctx->kv_keys, "metadata key");

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
	str_arena_free(&ctx->strs);
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
	ctx->data_size	= real_fsize >= (size_t)ctx->data_file_offset
						  ? real_fsize - (size_t)ctx->data_file_offset
						  : 0;

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
			ERROR("gguf_load_sparse: tensor '%s' extends past end of data section", t->name);
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
			if (atomic_load(&job.first_err) != OK)
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
					memmove((void *)t->data, (const char *)t->data + slop, tbytes);
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
		if (ctx->kv_types[i] == GGUF_TYPE_ARRAY)
			free(ctx->kv_arr_data[i]);
	}
	str_arena_free(&ctx->strs);
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
	return strtab_find(c->kv_hash, c->kv_hash_cap, key);
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
	ptrdiff_t i = strtab_find(c->tensor_hash, c->tensor_hash_cap, name);
	return i < 0 ? NULL : &c->tensors[i];
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
		case GGUF_TYPE_F64: {
			double v;
			memcpy(&v, &c->kv_vals[i], sizeof(v));
			fprintf(fp, "%g", v);
			break;
		}
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