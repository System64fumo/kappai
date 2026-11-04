#include "backend/backend.h"
#include "common.h"
#include "device.h"
#include "emit.h"
#include "gguf.h"
#include "hw.h"
#include "log.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define RKT_FFNA_SILU 0
#define RKT_FFNA_GELU 1

#define RKT_DEVICE_PATH "/dev/accel/accel0"
#define RKT_WEIGHT_MAGIC 0x524B5457u
#define RKT_TIMEOUT_NS 10000000000LL
#define RKT_POISON_WORD 0xCAFEBABEu
#define RKT_PRIORITY 50
#define RKT_F16_BLOCK_BYTES (CBUF_BANK_BYTES * 2)
#define RKT_TILE_ROWS 16
#define RKT_TILE_COLS 32

#define RKT_FOLD_K_LEN RKT_K_ALIGN

#define RKT_ACT_CHUNK 256
#define RKT_ACT_BLOCK_ROWS RKT_TILE_ROWS
#define RKT_ACT_MAX_ROUNDS 4

#define RKT_QUEUE_ARENA_CAP ((size_t)4 << 20)
#define RKT_QUEUE_ALIGN 64
#define RKT_STREAM_POOL_BYTES ((size_t)2 << 20)
#define RKT_STREAM_TABLE_SIZE 4096
#define RKT_STREAM_MAX_ENTRIES (RKT_STREAM_TABLE_SIZE * 3 / 4)
#define RKT_ATTN_BATCH_FLOATS ((size_t)1 << 20)

typedef struct {
	rkt_bo bo;
	int	   row_start;
	int	   row_count;
} rkt_part;

typedef struct {
	int		 k_off;
	int		 k_len;
	rkt_part part[RKT_MAX_CORES];
	int		 fold_ready;
	rkt_bo	 fold_bo[RKT_MAX_CORES];
} rkt_kchunk;

typedef struct rkt_weight rkt_weight;
struct rkt_weight {
	uint32_t	magic;
	int			n;
	int			k;
	int			n_pad;
	int			k_pad;
	uint32_t	src_type;
	size_t		src_row_bytes;
	int			n_kchunks;
	rkt_kchunk *kchunks;
	void	   *host_copy;
	rkt_weight *fused_parent;
	int			fused_row0;
	int			fuse_tried;
};

typedef struct {
	rkt_bo in;
	rkt_bo out;
} rkt_slot;

typedef enum {
	SLOT_DECODE,
	SLOT_BATCH,
	SLOT_CLASSES,
} slot_class;

struct rkt_act_cache {
	int			ready;
	int			uploaded_kind;
	rkt_lut_cfg cfg;
	rkt_bo		identity[RKT_MAX_CORES];
	rkt_bo		in[RKT_MAX_CORES];
	rkt_bo		out[RKT_MAX_CORES];
	rkt_bo		regcmd[RKT_MAX_CORES];
};
typedef struct {
	int core;
	int row_start;
	int rows;
	int w_row_off;
} job_desc;

typedef struct {
	const rkt_weight *w;
	const rkt_kchunk *kc;
	int				  row0;
	int				  n;
	int				  m;
	int				  m_done;
	int				  accumulate;
	float			 *y;
} block_ctx;

typedef struct {
	block_ctx ctx;
	int		  blk;
	job_desc  job;
	size_t	  in_off;
	size_t	  out_off;
	uint32_t  n_ops;
	uint32_t  regcmd_dma;
} rkt_task;

typedef struct {
	int		 blk;
	int		 k_len;
	int		 rows;
	uint32_t in_dma;
	uint32_t w_dma;
	uint32_t out_dma;
} stream_key;

typedef struct {
	stream_key key;
	uint32_t   offset;
	uint32_t   n_ops;
	int		   used;
} stream_entry;

typedef struct {
	rkt_bo		 bo;
	size_t		 bytes_used;
	int			 n_entries;
	int			 dirty;
	stream_entry entries[RKT_STREAM_TABLE_SIZE];
} rkt_stream_cache;

typedef struct {
	slot_class cls;
	int		   n_tasks;
	size_t	   out_bytes;
	rkt_task   tasks[RKT_MAX_TASKS];
} rkt_core_queue;

typedef struct rkt_act_cache rkt_act_cache;

typedef struct {
	rkt_device		 dev;
	pthread_mutex_t	 lock;
	int				 max_k;
	int				 poison;
	int				 npu_failed;
	rkt_slot		 slots[SLOT_CLASSES][RKT_MAX_CORES];
	uint16_t		*x16;
	size_t			 x16_cap;
	uint16_t		*x_tile;
	size_t			 x_tile_cap;
	uint16_t		*x_chunk;
	size_t			 x_chunk_cap;
	rkt_weight	   **weights;
	size_t			 n_weights;
	size_t			 cap_weights;
	int				 fuse;
	int				 fused_groups;
	int				 fused_weights;
	float			*y_tmp;
	size_t			 y_tmp_cap;
	float			*attn_ws;
	size_t			 attn_ws_cap;
	rkt_act_cache	 act;
	float			*act_tmp;
	size_t			 act_tmp_cap;
	int				 max_tasks;
	rkt_core_queue	 queue[RKT_MAX_CORES];
	rkt_stream_cache streams[RKT_MAX_CORES];
	uint8_t			*arena;
	size_t			 arena_cap;
	size_t			 arena_len;
} rkt_priv;

static inline uint16_t f32_to_f16_bits(float v) {
	_Float16 h = (_Float16)v;
	uint16_t bits;
	memcpy(&bits, &h, sizeof(bits));
	return bits;
}

static inline float f16_bits_to_f32(uint16_t bits) {
	_Float16 h;
	memcpy(&h, &bits, sizeof(h));
	return (float)h;
}

static inline void *rkt_ptr(const buffer *b) {
	return (char *)b->handle + b->offset;
}

static int env_int(const char *name, int fallback, int lo, int hi) {
	const char *e = getenv(name);
	if (!e)
		return fallback;
	int v = atoi(e);
	return v < lo ? lo : (v > hi ? hi : v);
}

static const char *device_path(void) {
	const char *e = getenv("KAPPAI_ROCKET_DEVICE");
	return e && *e ? e : RKT_DEVICE_PATH;
}

static int ensure_capacity(void **buf, size_t *cap, size_t need, size_t elem) {
	if (*cap >= need)
		return 1;
	void *grown = realloc(*buf, need * elem);
	if (!grown)
		return 0;
	*buf = grown;
	*cap = need;
	return 1;
}

static int partition_rows(int n, int n_cores, int *start, int *count) {
	int cores = n_cores;
	while (cores > 1 && n / cores < RKT_N_ALIGN)
		cores--;
	int base = ((n / cores) + RKT_N_ALIGN - 1) / RKT_N_ALIGN * RKT_N_ALIGN;
	int used = 0;
	for (int c = 0; c < cores; c++) {
		int s = c * base;
		int e = (c == cores - 1 || s + base > n) ? n : s + base;
		if (s >= n)
			break;
		start[used] = s;
		count[used] = e - s;
		used++;
	}
	return used;
}

static int max_batch_rows(int k_len) {
	int m = RKT_F16_BLOCK_BYTES / (k_len * 2);
	m	  = m / 4 * 4;
	return m < 4 ? 4 : m;
}

static void tile_weight_row(uint16_t *bo, const uint16_t *row, int k_len, int local_row) {
	int n_kpg	= k_len / RKT_TILE_COLS;
	int row_off = (local_row % RKT_TILE_ROWS) * RKT_TILE_COLS;
	int group	= local_row / RKT_TILE_ROWS;
	for (int kpg = 0; kpg < n_kpg; kpg++) {
		size_t tile = ((size_t)group * n_kpg + kpg) * (RKT_TILE_ROWS * RKT_TILE_COLS);
		memcpy(bo + tile + row_off, row + kpg * RKT_TILE_COLS, RKT_TILE_COLS * sizeof(uint16_t));
	}
}

static void tile_input(uint16_t *dst, const uint16_t *src, int m, int k) {
	for (int g = 0; g < k / RKT_FEATURE_C2_F16; g++) {
		uint16_t *plane = dst + (size_t)g * m * RKT_FEATURE_C2_F16;
		for (int r = 0; r < m; r++)
			memcpy(plane + (size_t)r * RKT_FEATURE_C2_F16,
				   src + (size_t)r * k + (size_t)g * RKT_FEATURE_C2_F16,
				   RKT_FEATURE_C2_F16 * sizeof(uint16_t));
	}
}

static void convert_to_f16(uint16_t *dst, const float *src, size_t n) {
	for (size_t i = 0; i < n; i++)
		dst[i] = f32_to_f16_bits(src[i]);
}

static void register_weight(rkt_priv *p, rkt_weight *w) {
	if (p->n_weights == p->cap_weights) {
		p->cap_weights = p->cap_weights ? p->cap_weights * 2 : 256;
		p->weights	   = xrealloc(p->weights, p->cap_weights * sizeof(*p->weights));
	}
	p->weights[p->n_weights++] = w;
}

static int unregister_weight(rkt_priv *p, const rkt_weight *w) {
	for (size_t i = 0; i < p->n_weights; i++) {
		if (p->weights[i] != w)
			continue;
		p->weights[i] = p->weights[--p->n_weights];
		return 1;
	}
	return 0;
}

static void destroy_weight(rkt_priv *p, rkt_weight *w) {
	for (int ci = 0; ci < w->n_kchunks; ci++) {
		for (int c = 0; c < RKT_MAX_CORES; c++) {
			rkt_bo_destroy(&p->dev, &w->kchunks[ci].part[c].bo);
			rkt_bo_destroy(&p->dev, &w->kchunks[ci].fold_bo[c]);
		}
		w->kchunks[ci].fold_ready = 0;
	}
	free(w->kchunks);
	free(w->host_copy);
	w->magic = 0;
	free(w);
}

static status_code row_to_f16(backend *host, uint32_t type, const void *src, int k, float *tmp,
							  uint16_t *dst) {
	if (type == GGML_TYPE_F16) {
		memcpy(dst, src, (size_t)k * sizeof(uint16_t));
		return OK;
	}
	if (type == GGML_TYPE_F32) {
		convert_to_f16(dst, src, (size_t)k);
		return OK;
	}
	if (!host || !host->dequant_row)
		return ERR_UNSUPPORTED;
	status_code s = host->dequant_row(host, type, src, k, tmp);
	if (s != OK)
		return s;
	convert_to_f16(dst, tmp, (size_t)k);
	return OK;
}

static int type_is_group_repacked(uint32_t type) {
	switch (type) {
	case GGML_TYPE_Q8_0_REPACKED:
	case GGML_TYPE_Q4_0_REPACKED:
	case GGML_TYPE_IQ3_S_REPACKED:
	case GGML_TYPE_IQ4_NL_REPACKED:
	case GGML_TYPE_Q4_K_REPACKED:
	case GGML_TYPE_Q5_K_REPACKED:
	case GGML_TYPE_Q6_K_REPACKED:
		return 1;
	default:
		return 0;
	}
}

static int weight_shape_supported(const rkt_priv *p, const rkt_weight *w) {
	return w->n > 0 && w->k > 0 && p->dev.n_cores > 0 && !type_is_group_repacked(w->src_type);
}

static int build_layout(rkt_priv *p, rkt_weight *w) {
	int start[RKT_MAX_CORES], count[RKT_MAX_CORES];
	int used = partition_rows(w->n_pad, p->dev.n_cores, start, count);

	w->n_kchunks = (w->k_pad + p->max_k - 1) / p->max_k;
	w->kchunks	 = xcalloc((size_t)w->n_kchunks, sizeof(*w->kchunks));
	for (int ci = 0; ci < w->n_kchunks; ci++) {
		rkt_kchunk *kc = &w->kchunks[ci];
		kc->k_off	   = ci * p->max_k;
		kc->k_len	   = w->k_pad - kc->k_off < p->max_k ? w->k_pad - kc->k_off : p->max_k;
		for (int c = 0; c < used; c++) {
			kc->part[c].row_start = start[c];
			kc->part[c].row_count = count[c];
			size_t bytes		  = (size_t)count[c] * kc->k_len * sizeof(uint16_t);
			if (rkt_bo_create(&p->dev, c, bytes, &kc->part[c].bo) != 0)
				return 0;
		}
	}
	return 1;
}

static const rkt_part *part_for_row(const rkt_kchunk *kc, int row) {
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		const rkt_part *pt = &kc->part[c];
		if (pt->bo.map && row >= pt->row_start && row < pt->row_start + pt->row_count)
			return pt;
	}
	return NULL;
}

static int flush_weight(rkt_priv *p, rkt_weight *w) {
	for (int ci = 0; ci < w->n_kchunks; ci++)
		for (int c = 0; c < RKT_MAX_CORES; c++)
			if (w->kchunks[ci].part[c].bo.map && rkt_bo_flush(&p->dev, &w->kchunks[ci].part[c].bo))
				return 0;
	return 1;
}

typedef status_code (*row_source_fn)(void *ctx, int row, uint16_t *dst);

static status_code fill_weight(rkt_priv *p, rkt_weight *w, row_source_fn source, void *ctx) {
	uint16_t   *row = xcalloc((size_t)w->k_pad, sizeof(uint16_t));
	status_code s	= OK;
	for (int r = 0; r < w->n_pad && s == OK; r++) {
		if (r < w->n)
			s = source(ctx, r, row);
		else
			memset(row, 0, (size_t)w->k_pad * sizeof(uint16_t));
		for (int ci = 0; ci < w->n_kchunks && s == OK; ci++) {
			const rkt_kchunk *kc = &w->kchunks[ci];
			const rkt_part	 *pt = part_for_row(kc, r);
			if (!pt) {
				s = ERR_INTERNAL;
				break;
			}
			tile_weight_row(pt->bo.map, row + kc->k_off, kc->k_len, r - pt->row_start);
		}
	}
	free(row);
	if (s == OK && !flush_weight(p, w))
		s = ERR_IO;
	return s;
}

typedef struct {
	backend		  *host;
	uint32_t	   type;
	const uint8_t *data;
	size_t		   row_bytes;
	int			   k;
	float		  *tmp;
} quant_rows;

static status_code quant_row_source(void *ctx, int row, uint16_t *dst) {
	quant_rows *q = ctx;
	return row_to_f16(q->host, q->type, q->data + (size_t)row * q->row_bytes, q->k, q->tmp, dst);
}

static rkt_weight *new_weight(const tensor_desc *desc) {
	rkt_weight *w	 = xcalloc(1, sizeof(*w));
	w->magic		 = RKT_WEIGHT_MAGIC;
	w->k			 = (int)desc->dims[0];
	w->n			 = (int)desc->dims[1];
	w->n_pad		 = (w->n + RKT_N_ALIGN - 1) / RKT_N_ALIGN * RKT_N_ALIGN;
	w->k_pad		 = (w->k + RKT_K_ALIGN - 1) / RKT_K_ALIGN * RKT_K_ALIGN;
	w->src_type		 = desc->type;
	w->src_row_bytes = ggml_row_size(desc->type, (size_t)w->k);
	return w;
}

static status_code upload_to_npu(rkt_priv *p, rkt_weight *w, const tensor_desc *desc) {
	if (!build_layout(p, w))
		return ERR_OUT_OF_MEMORY;
	quant_rows q = {
		.host	   = backend_host(),
		.type	   = desc->type,
		.data	   = desc->host_data,
		.row_bytes = w->src_row_bytes,
		.k		   = w->k,
		.tmp	   = xmalloc((size_t)w->k * sizeof(float)),
	};
	status_code s = fill_weight(p, w, quant_row_source, &q);
	free(q.tmp);
	return s;
}

static void keep_on_host(rkt_weight *w, const tensor_desc *desc) {
	size_t bytes = w->src_row_bytes * (size_t)w->n;
	w->host_copy = xmalloc(bytes);
	memcpy(w->host_copy, desc->host_data, bytes);
}

static void release_npu_state(rkt_priv *p, rkt_weight *w) {
	for (int ci = 0; ci < w->n_kchunks; ci++) {
		for (int c = 0; c < RKT_MAX_CORES; c++) {
			rkt_bo_destroy(&p->dev, &w->kchunks[ci].part[c].bo);
			rkt_bo_destroy(&p->dev, &w->kchunks[ci].fold_bo[c]);
		}
		w->kchunks[ci].fold_ready = 0;
	}
	free(w->kchunks);
	w->kchunks	 = NULL;
	w->n_kchunks = 0;
}

#define RKT_KV_MAGIC 0x524B5456u

typedef struct {
	rkt_weight *k;
	rkt_weight *v;
	int			hd;
	int			kvh;
	int			pos_pad;
	int			hd_pad;
} rkt_kv_slot;

typedef struct {
	uint32_t	 magic;
	int			 n_slots;
	rkt_kv_slot *slots;
} rkt_kv;

static int rkt_align_up(int v, int a) {
	return (v + a - 1) / a * a;
}

static void zero_weight_bos(rkt_priv *p, rkt_weight *w) {
	for (int ci = 0; ci < w->n_kchunks; ci++)
		for (int c = 0; c < RKT_MAX_CORES; c++) {
			rkt_part *pt = &w->kchunks[ci].part[c];
			if (!pt->bo.map)
				continue;
			memset(pt->bo.map, 0, pt->bo.size);
			rkt_bo_flush(&p->dev, &pt->bo);
		}
}

static rkt_weight *kv_weight_build(rkt_priv *p, int n_rows, int k, int k_pad) {
	rkt_weight *w	 = xcalloc(1, sizeof(*w));
	w->magic		 = RKT_WEIGHT_MAGIC;
	w->n			 = n_rows;
	w->n_pad		 = n_rows;
	w->k			 = k;
	w->k_pad		 = k_pad;
	w->src_type		 = GGML_TYPE_F16;
	w->src_row_bytes = (size_t)k * sizeof(uint16_t);
	if (!build_layout(p, w)) {
		release_npu_state(p, w);
		free(w);
		return NULL;
	}
	zero_weight_bos(p, w);
	return w;
}

static void rkt_kv_destroy(rkt_priv *p, rkt_kv *kv) {
	for (int i = 0; i < kv->n_slots; i++) {
		if (kv->slots[i].k)
			destroy_weight(p, kv->slots[i].k);
		if (kv->slots[i].v)
			destroy_weight(p, kv->slots[i].v);
	}
	free(kv->slots);
	kv->magic = 0;
	free(kv);
}

static status_code rkt_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
	rkt_priv *p = self->priv;
	size_t	  bytes;

	if (desc->n_dims != 2) {
		bytes	   = ggml_row_size(desc->type, (size_t)desc->dims[0]);
		void *copy = xmalloc(bytes);
		memcpy(copy, desc->host_data, bytes);
		*out = (buffer){.handle = copy, .size = bytes, .host_ptr = copy, .owner = self};
		return OK;
	}

	rkt_weight *w = new_weight(desc);
	bytes		  = w->src_row_bytes * (size_t)w->n;

	status_code s = ERR_UNSUPPORTED;
	if (weight_shape_supported(p, w)) {
		s = upload_to_npu(p, w, desc);
		if (s != OK)
			release_npu_state(p, w);
	}
	if (s != OK) {
		if (s != ERR_UNSUPPORTED)
			WARN("rocket: weight %dx%d (%s) stays on host: status %d", w->k, w->n,
				 ggml_type_name(desc->type), s);
		keep_on_host(w, desc);
	}

	pthread_mutex_lock(&p->lock);
	register_weight(p, w);
	pthread_mutex_unlock(&p->lock);

	*out = (buffer){.handle = w, .size = bytes, .owner = self};
	return OK;
}

static status_code rkt_buffer_alloc_scratch(backend *self, size_t size, buffer *out) {
	void *mem = xcalloc(1, size ? size : 1);
	*out	  = (buffer){.handle = mem, .size = size, .host_ptr = mem, .owner = self};
	return OK;
}

static void rkt_buffer_free(backend *self, buffer *buf) {
	if (!buf || !buf->handle)
		return;
	rkt_priv *p = self->priv;
	if (((rkt_kv *)buf->handle)->magic == RKT_KV_MAGIC) {
		*buf = (buffer){0};
		return;
	}
	pthread_mutex_lock(&p->lock);
	rkt_weight *w		  = buf->handle;
	int			is_weight = unregister_weight(p, w);
	pthread_mutex_unlock(&p->lock);

	if (is_weight)
		destroy_weight(p, w);
	else
		free(buf->handle);
	*buf = (buffer){0};
}

static status_code rkt_buffer_read_f32(backend *self, const buffer *buf, float *dst, int n) {
	(void)self;
	memcpy(dst, rkt_ptr(buf), (size_t)n * sizeof(float));
	return OK;
}

static status_code rkt_buffer_write_f32(backend *self, buffer *buf, const float *src, int n) {
	(void)self;
	memcpy(rkt_ptr(buf), src, (size_t)n * sizeof(float));
	return OK;
}

static int ensure_bo(rkt_priv *p, int core, size_t bytes, rkt_bo *bo) {
	if (bo->map && bo->size >= bytes)
		return 1;
	rkt_bo_destroy(&p->dev, bo);
	return rkt_bo_create(&p->dev, core, bytes, bo) == 0;
}

static int ensure_slot(rkt_priv *p, slot_class cls, int core, size_t in_bytes, size_t out_bytes) {
	rkt_slot *s = &p->slots[cls][core];
	return ensure_bo(p, core, in_bytes, &s->in) && ensure_bo(p, core, out_bytes, &s->out);
}

static int poison_hits(const uint32_t *words, size_t count) {
	for (size_t i = 0; i < count; i++)
		if (words[i] == RKT_POISON_WORD)
			return 1;
	return 0;
}

static size_t align_queue(size_t v) {
	return (v + RKT_QUEUE_ALIGN - 1) & ~((size_t)RKT_QUEUE_ALIGN - 1);
}

static const rkt_part *core_part(const rkt_kchunk *kc, int core) {
	const rkt_part *pt = &kc->part[0];
	for (int c = 0; c < RKT_MAX_CORES; c++)
		if (kc->part[c].bo.map && kc->part[c].bo.core == core)
			pt = &kc->part[c];
	return pt;
}

static void reset_queue(rkt_priv *p) {
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		p->queue[c].n_tasks	  = 0;
		p->queue[c].out_bytes = 0;
	}
	p->arena_len = 0;
}

static stream_entry *stream_lookup(rkt_stream_cache *sc, const stream_key *key) {
	const uint32_t *words = (const uint32_t *)key;
	uint32_t		h	  = 2166136261u;
	for (size_t i = 0; i < sizeof(*key) / sizeof(uint32_t); i++)
		h = (h ^ words[i]) * 16777619u;

	uint32_t idx = h & (RKT_STREAM_TABLE_SIZE - 1);
	while (sc->entries[idx].used && memcmp(&sc->entries[idx].key, key, sizeof(*key)))
		idx = (idx + 1) & (RKT_STREAM_TABLE_SIZE - 1);
	return &sc->entries[idx];
}

static status_code stage_streams(rkt_priv *p, int core, rkt_core_queue *q, const rkt_slot *sl) {
	rkt_stream_cache *sc = &p->streams[core];
	if (!sc->bo.map && rkt_bo_create(&p->dev, core, RKT_STREAM_POOL_BYTES, &sc->bo))
		return ERR_OUT_OF_MEMORY;

	size_t worst_bytes = (size_t)q->n_tasks * RKT_REGCMD_BYTES;
	if (sc->bytes_used + worst_bytes > sc->bo.size ||
		sc->n_entries + q->n_tasks > RKT_STREAM_MAX_ENTRIES) {
		memset(sc->entries, 0, sizeof(sc->entries));
		sc->bytes_used = 0;
		sc->n_entries  = 0;
	}

	for (int i = 0; i < q->n_tasks; i++) {
		rkt_task	   *t	= &q->tasks[i];
		const rkt_part *pt	= core_part(t->ctx.kc, core);
		stream_key		key = {
			.blk	= t->blk,
			.k_len	= t->ctx.kc->k_len,
			.rows	= t->job.rows,
			.in_dma = (uint32_t)(sl->in.dma_addr + t->in_off),
			.w_dma	= (uint32_t)(pt->bo.dma_addr +
								 (size_t)(t->job.row_start - pt->row_start) * t->ctx.kc->k_len * 2),
			.out_dma = (uint32_t)(sl->out.dma_addr + t->out_off),
		};
		stream_entry *e = stream_lookup(sc, &key);
		if (!e->used) {
			size_t	  offset = align_queue(sc->bytes_used);
			uint64_t *ops	 = (uint64_t *)((uint8_t *)sc->bo.map + offset);
			uint32_t  n_ops	 = (uint32_t)rkt_emit_matmul_f16(ops, key.blk, key.k_len, key.rows,
															 key.in_dma, key.w_dma, key.out_dma);
			if (n_ops > RKT_REGCMD_MAX_OPS)
				return ERR_INTERNAL;
			*e = (stream_entry){.key = key, .offset = (uint32_t)offset, .n_ops = n_ops, .used = 1};
			sc->bytes_used = offset + (size_t)n_ops * sizeof(uint64_t);
			sc->n_entries++;
			sc->dirty = 1;
		}
		t->n_ops	  = e->n_ops;
		t->regcmd_dma = (uint32_t)(sc->bo.dma_addr + e->offset);
	}
	if (!sc->dirty)
		return OK;
	sc->dirty = 0;
	return rkt_bo_flush(&p->dev, &sc->bo) ? ERR_IO : OK;
}

static status_code stage_core_queue(rkt_priv *p, int core) {
	rkt_core_queue *q = &p->queue[core];
	q->cls			  = SLOT_DECODE;
	for (int i = 0; i < q->n_tasks; i++)
		if (q->tasks[i].blk > 1)
			q->cls = SLOT_BATCH;
	if (!ensure_slot(p, q->cls, core, p->arena_len, q->out_bytes))
		return ERR_OUT_OF_MEMORY;

	rkt_slot *sl = &p->slots[q->cls][core];
	memcpy(sl->in.map, p->arena, p->arena_len);
	status_code s = stage_streams(p, core, q, sl);
	if (s != OK)
		return s;
	if (rkt_bo_flush(&p->dev, &sl->in))
		return ERR_IO;

	if (!p->poison)
		return OK;
	uint32_t *ow = sl->out.map;
	for (size_t i = 0; i < q->out_bytes / sizeof(uint32_t); i++)
		ow[i] = RKT_POISON_WORD;
	return rkt_bo_flush(&p->dev, &sl->out) ? ERR_IO : OK;
}

static status_code collect_task(const rkt_priv *p, const rkt_task *t, const float *om) {
	const block_ctx *b	 = &t->ctx;
	const job_desc	*j	 = &t->job;
	int				 blk = t->blk;
	if (p->poison && poison_hits((const uint32_t *)om, (size_t)blk * j->rows)) {
		ERROR("rocket: core %d output still holds poison (npu did not write)", j->core);
		return ERR_COMPUTE_FAIL;
	}

	int lo = (b->row0 > j->row_start ? b->row0 : j->row_start) - j->row_start;
	int hi = (b->row0 + b->n < j->row_start + j->rows ? b->row0 + b->n : j->row_start + j->rows) -
			 j->row_start;
	if (hi <= lo)
		return OK;
	for (int r = 0; r < blk; r++) {
		float *dst = &b->y[(size_t)(b->m_done + r) * b->n + (j->row_start + lo - b->row0)];
		for (int nn = lo; nn < hi; nn++) {
			float v = om[(size_t)(nn / RKT_OUT_C2_F32) * blk * RKT_OUT_C2_F32 +
						 (size_t)r * RKT_OUT_C2_F32 + nn % RKT_OUT_C2_F32];
			if (b->accumulate)
				dst[nn - lo] += v;
			else
				dst[nn - lo] = v;
		}
	}
	return OK;
}

static status_code flush_queue(rkt_priv *p) {
	status_code s					  = OK;
	int			active[RKT_MAX_CORES] = {0};
	int			n_active			  = 0;
	for (int c = 0; c < RKT_MAX_CORES; c++)
		if (p->queue[c].n_tasks)
			active[n_active++] = c;

	for (int i = 0; i < n_active && s == OK; i++)
		s = stage_core_queue(p, active[i]);

	int submitted = 0;
	for (; submitted < n_active && s == OK; submitted++) {
		int					  c	 = active[submitted];
		const rkt_core_queue *q	 = &p->queue[c];
		const rkt_slot		 *sl = &p->slots[q->cls][c];
		uint32_t			  n_ops[RKT_MAX_TASKS];
		uint32_t			  regcmd_dma[RKT_MAX_TASKS];
		for (int t = 0; t < q->n_tasks; t++) {
			n_ops[t]	  = q->tasks[t].n_ops;
			regcmd_dma[t] = q->tasks[t].regcmd_dma;
		}
		if (rkt_submit(&p->dev, regcmd_dma, n_ops, q->n_tasks, &p->streams[c].bo, &sl->in,
					   &sl->out))
			s = ERR_COMPUTE_FAIL;
	}
	for (int i = 0; i < submitted; i++) {
		int c = active[i];
		if (rkt_bo_wait(&p->dev, &p->slots[p->queue[c].cls][c].out) && s == OK)
			s = ERR_COMPUTE_FAIL;
	}
	for (int i = 0; i < n_active && s == OK; i++) {
		const rkt_core_queue *q	 = &p->queue[active[i]];
		const rkt_slot		 *sl = &p->slots[q->cls][active[i]];
		for (int t = 0; t < q->n_tasks && s == OK; t++)
			s = collect_task(p, &q->tasks[t],
							 (const float *)((const uint8_t *)sl->out.map + q->tasks[t].out_off));
	}
	reset_queue(p);
	return s;
}

static int plan_jobs(const block_ctx *b, int lo, int hi, int round, job_desc *jobs) {
	int n_jobs = 0;
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		const rkt_part *pt = &b->kc->part[c];
		if (!pt->bo.map)
			continue;
		int a	  = lo > pt->row_start ? lo : pt->row_start;
		int e	  = hi < pt->row_start + pt->row_count ? hi : pt->row_start + pt->row_count;
		int start = a + round * RKT_MAX_N_CHUNK;
		if (start >= e)
			continue;
		int rows	   = e - start < RKT_MAX_N_CHUNK ? e - start : RKT_MAX_N_CHUNK;
		jobs[n_jobs++] = (job_desc){.core = c, .row_start = start, .rows = rows};
	}
	return n_jobs;
}

static status_code queue_block(rkt_priv *p, const block_ctx *b, int blk, const uint16_t *x_block) {
	int lo = b->row0 & ~(RKT_N_ALIGN - 1);
	int hi = (b->row0 + b->n + RKT_N_ALIGN - 1) & ~(RKT_N_ALIGN - 1);
	if (hi > b->w->n_pad)
		hi = b->w->n_pad;

	size_t in_bytes = (size_t)blk * b->kc->k_len * sizeof(uint16_t);
	if (!ensure_capacity((void **)&p->x_tile, &p->x_tile_cap, (size_t)blk * b->kc->k_len,
						 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	tile_input(p->x_tile, x_block, blk, b->kc->k_len);

	size_t	 in_off = 0;
	int		 staged = 0;
	job_desc jobs[RKT_MAX_CORES];
	for (int round = 0;; round++) {
		int n_jobs = plan_jobs(b, lo, hi, round, jobs);
		if (n_jobs == 0)
			return OK;
		for (int i = 0; i < n_jobs; i++) {
			rkt_core_queue *q = &p->queue[jobs[i].core];
			if (q->n_tasks == p->max_tasks ||
				(!staged && p->arena_len && p->arena_len + in_bytes > RKT_QUEUE_ARENA_CAP)) {
				status_code s = flush_queue(p);
				if (s != OK)
					return s;
				staged = 0;
			}
			if (!staged) {
				in_off = align_queue(p->arena_len);
				if (!ensure_capacity((void **)&p->arena, &p->arena_cap, in_off + in_bytes, 1))
					return ERR_OUT_OF_MEMORY;
				memcpy(p->arena + in_off, p->x_tile, in_bytes);
				p->arena_len = in_off + in_bytes;
				staged		 = 1;
			}
			q->tasks[q->n_tasks++] = (rkt_task){
				.ctx	 = *b,
				.blk	 = blk,
				.job	 = jobs[i],
				.in_off	 = in_off,
				.out_off = q->out_bytes,
			};
			q->out_bytes += align_queue((size_t)blk * jobs[i].rows * sizeof(float));
		}
	}
}

static int next_block_rows(int remaining, int max_rows) {
	if (remaining < 4)
		return 1;
	int blk = remaining / 4 * 4;
	return blk > max_rows ? max_rows : blk;
}

static const uint16_t *chunk_input(rkt_priv *p, const rkt_kchunk *kc, int k, int m,
								   int *stride_ok) {
	*stride_ok = 1;
	if (kc->k_len == k)
		return p->x16;
	if (!ensure_capacity((void **)&p->x_chunk, &p->x_chunk_cap, (size_t)m * kc->k_len,
						 sizeof(uint16_t))) {
		*stride_ok = 0;
		return NULL;
	}
	for (int r = 0; r < m; r++)
		memcpy(p->x_chunk + (size_t)r * kc->k_len, p->x16 + (size_t)r * k + kc->k_off,
			   (size_t)kc->k_len * sizeof(uint16_t));
	return p->x_chunk;
}

static status_code enqueue_kchunks(rkt_priv *p, const rkt_weight *w, int row0, int n, float *y,
								   int m) {
	for (int ci = 0; ci < w->n_kchunks; ci++) {
		const rkt_kchunk *kc = &w->kchunks[ci];
		int				  ok;
		const uint16_t	 *xc = chunk_input(p, kc, w->k_pad, m, &ok);
		if (!ok)
			return ERR_OUT_OF_MEMORY;

		block_ctx b = {
			.w = w, .kc = kc, .row0 = row0, .n = n, .m = m, .y = y, .accumulate = ci > 0};
		int max_rows = max_batch_rows(kc->k_len);
		for (int done = 0; done < m;) {
			int blk		  = next_block_rows(m - done, max_rows);
			b.m_done	  = done;
			status_code s = queue_block(p, &b, blk, xc + (size_t)done * kc->k_len);
			if (s != OK)
				return s;
			done += blk;
		}
	}
	return OK;
}

static status_code enqueue_fold(rkt_priv *p, const rkt_weight *w, const rkt_kchunk *fold_kc,
								int row0, int n, const uint16_t *x16, float *y, int m) {
	if (!ensure_capacity((void **)&p->x_chunk, &p->x_chunk_cap, (size_t)m * RKT_FOLD_K_LEN,
						 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	memset(p->x_chunk, 0, (size_t)m * RKT_FOLD_K_LEN * sizeof(uint16_t));
	for (int r = 0; r < m; r++)
		p->x_chunk[(size_t)r * RKT_FOLD_K_LEN] = x16[(size_t)r * w->k_pad + w->k];

	block_ctx fb = {.w = w, .kc = fold_kc, .row0 = row0, .n = n, .m = m, .y = y, .accumulate = 1};
	for (int done = 0; done < m;) {
		int blk		  = next_block_rows(m - done, RKT_TILE_ROWS);
		fb.m_done	  = done;
		status_code s = queue_block(p, &fb, blk, p->x_chunk + (size_t)done * RKT_FOLD_K_LEN);
		if (s != OK)
			return s;
		done += blk;
	}
	return OK;
}

static status_code npu_matmul_fold(rkt_priv *p, const rkt_weight *w, int row0, int n,
								   const uint16_t *x16, float *y, int m) {
	rkt_kchunk fold_kc = {
		.k_off = w->k,
		.k_len = RKT_FOLD_K_LEN,
	};
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		fold_kc.part[c]	   = w->kchunks[0].part[c];
		fold_kc.part[c].bo = w->kchunks[0].fold_bo[c];
	}

	status_code s = enqueue_kchunks(p, w, row0, n, y, m);
	if (s == OK)
		s = enqueue_fold(p, w, &fold_kc, row0, n, x16, y, m);
	if (s == OK)
		return flush_queue(p);
	reset_queue(p);
	return s;
}

static status_code npu_matmul_enqueue(rkt_priv *p, const rkt_weight *w, int row0, int n,
									  const float *x, float *y, int m) {
	if (!ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)m * w->k_pad, sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	for (int r = 0; r < m; r++) {
		uint16_t *dst = p->x16 + (size_t)r * w->k_pad;
		convert_to_f16(dst, x + (size_t)r * w->k, (size_t)w->k);
		memset(dst + w->k, 0, (size_t)(w->k_pad - w->k) * sizeof(uint16_t));
	}
	return enqueue_kchunks(p, w, row0, n, y, m);
}

static status_code npu_matmul(rkt_priv *p, const rkt_weight *w, int row0, int n, const float *x,
							  float *y, int m) {
	status_code s = npu_matmul_enqueue(p, w, row0, n, x, y, m);
	if (s == OK)
		return flush_queue(p);
	reset_queue(p);
	return s;
}

static float tiled_weight_at(const rkt_part *pt, int k_len, int row, int col) {
	int				local = row - pt->row_start;
	size_t			tile  = ((size_t)(local / RKT_TILE_ROWS) * (k_len / RKT_TILE_COLS) +
							 (size_t)(col / RKT_TILE_COLS)) *
							(RKT_TILE_ROWS * RKT_TILE_COLS);
	const uint16_t *bits  = pt->bo.map;
	return f16_bits_to_f32(bits[tile + (size_t)(local % RKT_TILE_ROWS) * RKT_TILE_COLS +
								(size_t)(col % RKT_TILE_COLS)]);
}

static void tiled_matmul_cpu(const rkt_weight *w, int row0, int n, const float *x, float *y,
							 int m) {
	for (int r = 0; r < m; r++) {
		for (int j = 0; j < n; j++) {
			float acc = 0.0f;
			for (int ci = 0; ci < w->n_kchunks; ci++) {
				const rkt_kchunk *kc = &w->kchunks[ci];
				const rkt_part	 *pt = part_for_row(kc, row0 + j);
				for (int c = 0; c < kc->k_len; c++) {
					int col = kc->k_off + c;
					if (col >= w->k)
						break;
					acc += tiled_weight_at(pt, kc->k_len, row0 + j, c) *
						   f16_bits_to_f32(f32_to_f16_bits(x[(size_t)r * w->k + col]));
				}
			}
			y[(size_t)r * n + j] = acc;
		}
	}
}

static status_code host_matmul(const rkt_weight *w, int row0, int n, const buffer *x, buffer *y,
							   int m) {
	backend *host = backend_host();
	size_t	 off  = (size_t)row0 * w->src_row_bytes;
	buffer	 wv	  = buffer_host_view(host, (uint8_t *)w->host_copy + off, w->src_row_bytes * n);
	buffer	 xv	  = buffer_host_view(host, rkt_ptr(x), (size_t)m * w->k * sizeof(float));
	buffer	 yv	  = buffer_host_view(host, rkt_ptr(y), (size_t)m * n * sizeof(float));
	return host->matmul_batch(host, &wv, w->src_type, &xv, &yv, n, w->k, m);
}

static status_code rkt_matmul_batch(backend *self, const buffer *wbuf, uint32_t w_type,
									const buffer *x, buffer *y, int n, int k, int m) {
	(void)w_type;
	rkt_priv   *p = self->priv;
	rkt_weight *w = wbuf->handle;
	if (!w || w->magic != RKT_WEIGHT_MAGIC || w->k != k)
		return ERR_INVALID_ARG;
	if (wbuf->offset % w->src_row_bytes != 0)
		return ERR_INVALID_ARG;
	int row0 = (int)(wbuf->offset / w->src_row_bytes);
	if (row0 + n > w->n)
		return ERR_INVALID_ARG;
	if (w->fused_parent) {
		row0 += w->fused_row0;
		w = w->fused_parent;
	}

	const float *xf = rkt_ptr(x);
	float		*yf = rkt_ptr(y);
	if (w->host_copy)
		return host_matmul(w, row0, n, x, y, m);

	pthread_mutex_lock(&p->lock);
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL : npu_matmul(p, w, row0, n, xf, yf, m);
	if (s != OK && !p->npu_failed) {
		p->npu_failed = 1;
		WARN("rocket: npu matmul failed (n=%d k=%d m=%d status=%d, errno=%d), npu disabled, using "
			 "cpu on "
			 "tiled weights",
			 n, k, m, s, errno);
	}
	if (s != OK) {
		tiled_matmul_cpu(w, row0, n, xf, yf, m);
		s = OK;
	}
	pthread_mutex_unlock(&p->lock);
	return s;
}

static status_code rkt_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							  buffer *y, int n, int k) {
	return rkt_matmul_batch(self, w, w_type, x, y, n, k, 1);
}

#define RKT_MAX_FUSE 8

static void copy_weight_row(const rkt_part *sp, int s_row, const rkt_part *dp, int d_row,
							int k_len) {
	int				n_kpg	= k_len / RKT_TILE_COLS;
	int				s_local = s_row - sp->row_start;
	int				d_local = d_row - dp->row_start;
	const uint16_t *src		= sp->bo.map;
	uint16_t	   *dst		= dp->bo.map;
	for (int kpg = 0; kpg < n_kpg; kpg++) {
		size_t s_tile =
			((size_t)(s_local / RKT_TILE_ROWS) * n_kpg + kpg) * (RKT_TILE_ROWS * RKT_TILE_COLS) +
			(size_t)(s_local % RKT_TILE_ROWS) * RKT_TILE_COLS;
		size_t d_tile =
			((size_t)(d_local / RKT_TILE_ROWS) * n_kpg + kpg) * (RKT_TILE_ROWS * RKT_TILE_COLS) +
			(size_t)(d_local % RKT_TILE_ROWS) * RKT_TILE_COLS;
		memcpy(dst + d_tile, src + s_tile, RKT_TILE_COLS * sizeof(uint16_t));
	}
}

static rkt_weight *fuse_weights(rkt_priv *p, rkt_weight **ws, int count) {
	int total = 0;
	for (int i = 0; i < count; i++) {
		const rkt_weight *w = ws[i];
		if (!w || w->magic != RKT_WEIGHT_MAGIC || w->host_copy || w->fused_parent ||
			w->n_kchunks == 0 || w->k != ws[0]->k || w->k_pad != ws[0]->k_pad ||
			w->n_kchunks != ws[0]->n_kchunks)
			return NULL;
		for (int j = 0; j < i; j++)
			if (ws[j] == w)
				return NULL;
		total += w->n_pad;
	}

	rkt_weight *f	 = xcalloc(1, sizeof(*f));
	f->magic		 = RKT_WEIGHT_MAGIC;
	f->n			 = total;
	f->n_pad		 = total;
	f->k			 = ws[0]->k;
	f->k_pad		 = ws[0]->k_pad;
	f->src_type		 = ws[0]->src_type;
	f->src_row_bytes = ws[0]->src_row_bytes;
	if (!build_layout(p, f)) {
		release_npu_state(p, f);
		free(f);
		return NULL;
	}

	int ok = 1;
	for (int ci = 0; ci < f->n_kchunks && ok; ci++) {
		const rkt_kchunk *fk  = &f->kchunks[ci];
		int				  off = 0;
		for (int i = 0; i < count && ok; i++) {
			const rkt_kchunk *sk = &ws[i]->kchunks[ci];
			for (int r = 0; r < ws[i]->n_pad; r++) {
				const rkt_part *sp = part_for_row(sk, r);
				const rkt_part *dp = part_for_row(fk, off + r);
				if (!sp || !dp) {
					ok = 0;
					break;
				}
				copy_weight_row(sp, r, dp, off + r, fk->k_len);
			}
			off += ws[i]->n_pad;
		}
	}
	if (!ok || !flush_weight(p, f)) {
		release_npu_state(p, f);
		free(f);
		return NULL;
	}

	register_weight(p, f);
	int off = 0;
	for (int i = 0; i < count; i++) {
		release_npu_state(p, ws[i]);
		ws[i]->fused_parent = f;
		ws[i]->fused_row0	= off;
		off += ws[i]->n_pad;
	}
	p->fused_groups++;
	p->fused_weights += count;
	if (p->fused_groups == 1)
		log_tag("RKT", "fusing projections that share an input into larger jobs "
					   "(KAPPAI_ROCKET_FUSE=0 disables)");
	return f;
}

static status_code run_npu_or_cpu(rkt_priv *p, const rkt_weight *w, int row0, int n,
								  const float *xf, float *yf, int m) {
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL : npu_matmul(p, w, row0, n, xf, yf, m);
	if (s != OK && !p->npu_failed) {
		p->npu_failed = 1;
		WARN("rocket: npu matmul failed (n=%d k=%d m=%d status=%d, errno=%d), npu disabled, using "
			 "cpu on tiled weights",
			 n, w->k, m, s, errno);
	}
	if (s != OK) {
		tiled_matmul_cpu(w, row0, n, xf, yf, m);
		s = OK;
	}
	return s;
}

static status_code multi_fused(backend *self, const buffer **w, const buffer *x, buffer **y,
							   const int *n_list, int k, int n_matmuls, int m) {
	rkt_priv   *p = self->priv;
	rkt_weight *ws[RKT_MAX_FUSE];
	int			row_in[RKT_MAX_FUSE];
	if (n_matmuls > RKT_MAX_FUSE)
		return ERR_UNSUPPORTED;
	for (int i = 0; i < n_matmuls; i++) {
		ws[i] = w[i]->handle;
		if (!ws[i] || ws[i]->magic != RKT_WEIGHT_MAGIC || ws[i]->k != k ||
			w[i]->offset % ws[i]->src_row_bytes != 0)
			return ERR_UNSUPPORTED;
		row_in[i] = (int)(w[i]->offset / ws[i]->src_row_bytes);
		if (row_in[i] + n_list[i] > ws[i]->n)
			return ERR_INVALID_ARG;
	}

	pthread_mutex_lock(&p->lock);
	rkt_weight *parent = ws[0]->fused_parent;
	if (!parent && !p->npu_failed && !ws[0]->fuse_tried) {
		for (int i = 0; i < n_matmuls; i++)
			ws[i]->fuse_tried = 1;
		parent = fuse_weights(p, ws, n_matmuls);
	}
	int fusable = parent && !p->npu_failed;
	for (int i = 0; i < n_matmuls && fusable; i++)
		fusable = ws[i]->fused_parent == parent;
	if (!fusable) {
		pthread_mutex_unlock(&p->lock);
		return ERR_UNSUPPORTED;
	}

	int row0 = INT32_MAX, hi = 0;
	for (int i = 0; i < n_matmuls; i++) {
		int r0 = ws[i]->fused_row0 + row_in[i];
		if (r0 < row0)
			row0 = r0;
		if (r0 + n_list[i] > hi)
			hi = r0 + n_list[i];
	}
	int span = hi - row0;
	if (!ensure_capacity((void **)&p->y_tmp, &p->y_tmp_cap, (size_t)m * span, sizeof(float))) {
		pthread_mutex_unlock(&p->lock);
		return ERR_OUT_OF_MEMORY;
	}

	status_code s = run_npu_or_cpu(p, parent, row0, span, rkt_ptr(x), p->y_tmp, m);
	if (s == OK) {
		for (int i = 0; i < n_matmuls; i++) {
			float *dst = rkt_ptr(y[i]);
			int	   off = ws[i]->fused_row0 + row_in[i] - row0;
			for (int r = 0; r < m; r++)
				memcpy(dst + (size_t)r * n_list[i], p->y_tmp + (size_t)r * span + off,
					   (size_t)n_list[i] * sizeof(float));
		}
	}
	pthread_mutex_unlock(&p->lock);
	return s;
}

static status_code rkt_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
										  const buffer *x, buffer **y, const int *n_list, int k,
										  int n_matmuls, int m) {
	rkt_priv *p = self->priv;
	if (p->fuse && n_matmuls >= 2) {
		status_code s = multi_fused(self, w, x, y, n_list, k, n_matmuls, m);
		if (s != ERR_UNSUPPORTED)
			return s;
	}
	for (int i = 0; i < n_matmuls; i++) {
		status_code s = rkt_matmul_batch(self, w[i], w_types[i], x, y[i], n_list[i], k, m);
		if (s != OK)
			return s;
	}
	return OK;
}

static status_code rkt_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
									const buffer *x, buffer **y, const int *n_list, int k,
									int n_matmuls) {
	return rkt_matmul_multi_batch(self, w, w_types, x, y, n_list, k, n_matmuls, 1);
}

static int ensure_fold_bos(rkt_priv *p, rkt_kchunk *kc) {
	if (kc->fold_ready)
		return 1;
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		if (!kc->part[c].bo.map || kc->part[c].row_count <= 0)
			continue;
		size_t bytes = (size_t)kc->part[c].row_count * RKT_FOLD_K_LEN * sizeof(uint16_t);
		if (rkt_bo_create(&p->dev, kc->part[c].bo.core, bytes, &kc->fold_bo[c]))
			return 0;
		memset(kc->fold_bo[c].map, 0, kc->fold_bo[c].size);
	}
	kc->fold_ready = 1;
	return 1;
}

static status_code rkt_matmul_residual(backend *self, const buffer *w, uint32_t w_type,
									   const buffer *x, const buffer *residual, buffer *y, int n,
									   int k) {
	(void)w_type;
	rkt_priv   *p  = self->priv;
	rkt_weight *wt = w->handle;
	if (!wt || wt->magic != RKT_WEIGHT_MAGIC || wt->k != k)
		return ERR_INVALID_ARG;
	if (w->offset % wt->src_row_bytes != 0)
		return ERR_INVALID_ARG;
	int row0 = (int)(w->offset / wt->src_row_bytes);
	if (row0 + n > wt->n)
		return ERR_INVALID_ARG;
	if (wt->fused_parent) {
		row0 += wt->fused_row0;
		wt = wt->fused_parent;
	}
	if (wt->host_copy || wt->n_kchunks != 1 || wt->kchunks[0].k_len + RKT_FOLD_K_LEN > p->max_k ||
		wt->k_pad != wt->k || rkt_ptr(residual) == NULL) {
		const float *rr = rkt_ptr(residual);
		float		*rc = NULL;
		if (rr && rr == rkt_ptr(y)) {
			rc = xmalloc((size_t)n * sizeof(float));
			memcpy(rc, rr, (size_t)n * sizeof(float));
			rr = rc;
		}
		status_code s2 = rkt_matmul(self, w, w_type, x, y, n, k);
		if (s2 == OK && rr) {
			float *y2 = rkt_ptr(y);
			for (int i = 0; i < n; i++)
				y2[i] += rr[i];
		}
		free(rc);
		return s2;
	}

	const float *xf	   = rkt_ptr(x);
	float		*yf	   = rkt_ptr(y);
	const float *r	   = rkt_ptr(residual);
	float		*rcopy = NULL;
	if (r == yf) {
		rcopy = xmalloc((size_t)n * sizeof(float));
		memcpy(rcopy, r, (size_t)n * sizeof(float));
		r = rcopy;
	}

	pthread_mutex_lock(&p->lock);
	status_code s = ERR_COMPUTE_FAIL;
	if (!p->npu_failed) {
		rkt_kchunk *kc = &wt->kchunks[0];
		if (ensure_fold_bos(p, kc)) {
			for (int c = 0; c < RKT_MAX_CORES; c++) {
				if (!kc->fold_bo[c].map)
					continue;
				uint16_t *fold = kc->fold_bo[c].map;
				for (int rr = 0; rr < kc->part[c].row_count; rr++) {
					int global	  = kc->part[c].row_start + rr;
					int tile_slot = (rr / RKT_TILE_ROWS) * (RKT_TILE_ROWS * RKT_TILE_COLS) +
									(rr % RKT_TILE_ROWS) * RKT_TILE_COLS;
					fold[tile_slot] =
						global >= row0 && global < row0 + n ? f32_to_f16_bits(r[global - row0]) : 0;
				}
				if (rkt_bo_flush(&p->dev, &kc->fold_bo[c]))
					s = ERR_IO;
			}
			if (s == OK) {
				if (ensure_capacity((void **)&p->x16, &p->x16_cap,
									(size_t)1 * (wt->k + RKT_FOLD_K_LEN), sizeof(uint16_t))) {
					convert_to_f16(p->x16, xf, (size_t)k);
					memset(p->x16 + k, 0, (size_t)(wt->k_pad - wt->k) * sizeof(uint16_t));
					p->x16[k] = 0x3C00u;
					s		  = npu_matmul_fold(p, wt, row0, n, p->x16, yf, 1);
				} else {
					s = ERR_OUT_OF_MEMORY;
				}
			}
		}
	}
	if (s != OK && !p->npu_failed) {
		p->npu_failed = 1;
		WARN("rocket: npu residual-fold matmul failed (n=%d k=%d status=%d, errno=%d), npu "
			 "disabled, using cpu",
			 n, k, s, errno);
	}
	if (s != OK) {
		tiled_matmul_cpu(wt, row0, n, xf, yf, 1);
		float *y2 = rkt_ptr(y);
		for (int i = 0; i < n; i++)
			y2[i] += r[i];
		s = OK;
	}
	pthread_mutex_unlock(&p->lock);
	free(rcopy);
	return s;
}

#define RKT_LUT_HQ_SEL 4

static double rkt_lut_fn(rkt_lut_kind kind, double x) {
	if (kind == RKT_LUT_GELU_TANH)
		return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
	return x / (1.0 + exp(-x));
}

static void rkt_lut_configure(rkt_lut_cfg *cfg, rkt_lut_kind kind) {
	cfg->kind		= kind;
	cfg->sel		= RKT_LUT_HQ_SEL;
	cfg->S			= 256;
	cfg->F			= 2048;
	cfg->start		= -(RKT_LUT_WINDOW_HALF << cfg->sel);
	cfg->end		= (RKT_LUT_WINDOW_HALF << cfg->sel) - 1;
	cfg->tail_slope = cfg->F / cfg->S;
}

static void rkt_lut_fill(const rkt_lut_cfg *cfg, uint16_t *le) {
	long step = 1L << cfg->sel;
	for (int i = 0; i < RKT_LUT_ENTRIES; i++) {
		long   y_int = (long)cfg->start + (long)i * step;
		double v	 = rkt_lut_fn(cfg->kind, (double)y_int / cfg->S) * cfg->F;
		long   r	 = lround(v);
		if (r > 32767)
			r = 32767;
		if (r < -32768)
			r = -32768;
		le[i] = (uint16_t)((int32_t)r & 0xFFFF);
	}
}

static void rkt_act_cache_reset(rkt_priv *p, rkt_act_cache *a) {
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		rkt_bo_destroy(&p->dev, &a->identity[c]);
		rkt_bo_destroy(&p->dev, &a->in[c]);
		rkt_bo_destroy(&p->dev, &a->out[c]);
		rkt_bo_destroy(&p->dev, &a->regcmd[c]);
	}
	a->ready		 = 0;
	a->uploaded_kind = -1;
}

static status_code rkt_lut_upload_tables(rkt_priv *p, rkt_act_cache *a, const uint16_t *le) {
	for (int off = 0; off < RKT_LUT_ENTRIES; off += RKT_LUT_UPLOAD_CHUNK) {
		int count = RKT_LUT_ENTRIES - off < RKT_LUT_UPLOAD_CHUNK ? RKT_LUT_ENTRIES - off
																 : RKT_LUT_UPLOAD_CHUNK;
		for (int c = 0; c < p->dev.n_cores; c++) {
			uint32_t n_ops = (uint32_t)rkt_emit_lut_upload(
				a->regcmd[c].map, 0, off, le + off, count, (uint32_t)a->in[c].dma_addr,
				(uint32_t)a->identity[c].dma_addr, (uint32_t)a->out[c].dma_addr);
			if (!n_ops || n_ops > RKT_REGCMD_MAX_OPS)
				return ERR_INTERNAL;
			if (rkt_bo_flush(&p->dev, &a->regcmd[c]))
				return ERR_IO;
			uint32_t regcmd_dma = (uint32_t)a->regcmd[c].dma_addr;
			if (rkt_submit(&p->dev, &regcmd_dma, &n_ops, 1, &a->regcmd[c], &a->in[c], &a->out[c]))
				return ERR_COMPUTE_FAIL;
			if (rkt_bo_wait(&p->dev, &a->out[c]))
				return ERR_COMPUTE_FAIL;
		}
	}
	return OK;
}

static status_code rkt_act_cache_init(rkt_priv *p, rkt_act_cache *a) {
	if (a->ready)
		return OK;
	rkt_lut_configure(&a->cfg, RKT_LUT_NONE);
	size_t	 identity_bytes = (size_t)RKT_ACT_CHUNK * RKT_ACT_CHUNK * sizeof(uint16_t);
	size_t	 in_bytes		= (size_t)RKT_ACT_BLOCK_ROWS * RKT_ACT_CHUNK * sizeof(uint16_t);
	size_t	 out_bytes		= (size_t)RKT_ACT_BLOCK_ROWS * RKT_ACT_CHUNK * sizeof(float);
	uint16_t row[RKT_ACT_CHUNK];
	for (int c = 0; c < p->dev.n_cores; c++) {
		if (rkt_bo_create(&p->dev, c, identity_bytes, &a->identity[c]) ||
			rkt_bo_create(&p->dev, c, in_bytes, &a->in[c]) ||
			rkt_bo_create(&p->dev, c, out_bytes, &a->out[c]) ||
			rkt_bo_create(&p->dev, c, RKT_REGCMD_BYTES, &a->regcmd[c])) {
			rkt_act_cache_reset(p, a);
			return ERR_OUT_OF_MEMORY;
		}
		for (int j = 0; j < RKT_ACT_CHUNK; j++) {
			memset(row, 0, sizeof(row));
			row[j] = f32_to_f16_bits((float)a->cfg.S);
			tile_weight_row(a->identity[c].map, row, RKT_ACT_CHUNK, j);
		}
		if (rkt_bo_flush(&p->dev, &a->identity[c])) {
			rkt_act_cache_reset(p, a);
			return ERR_IO;
		}
	}
	a->uploaded_kind = -1;
	a->ready		 = 1;
	return OK;
}

static status_code rkt_act_cache_select(rkt_priv *p, rkt_act_cache *a, rkt_lut_kind kind) {
	status_code s = rkt_act_cache_init(p, a);
	if (s != OK)
		return s;
	rkt_lut_configure(&a->cfg, kind);
	if (a->uploaded_kind == (int)kind)
		return OK;
	uint16_t le[RKT_LUT_ENTRIES];
	rkt_lut_fill(&a->cfg, le);
	s = rkt_lut_upload_tables(p, a, le);
	if (s == OK)
		a->uploaded_kind = (int)kind;
	return s;
}

typedef struct {
	int first;
	int rows;
} act_block;

static status_code rkt_act_round(rkt_priv *p, rkt_act_cache *a, const uint16_t *x16,
								 const act_block *blocks, int n_blocks, float *out, int n,
								 int chunks_per_row) {
	uint32_t n_ops[RKT_MAX_CORES];
	for (int c = 0; c < n_blocks; c++) {
		const act_block *b = &blocks[c];
		tile_input(a->in[c].map, x16 + (size_t)b->first * RKT_ACT_CHUNK, b->rows, RKT_ACT_CHUNK);
		n_ops[c] = (uint32_t)rkt_emit_matmul_f16_lut(
			a->regcmd[c].map, b->rows, RKT_ACT_CHUNK, RKT_ACT_CHUNK, (uint32_t)a->in[c].dma_addr,
			(uint32_t)a->identity[c].dma_addr, (uint32_t)a->out[c].dma_addr, &a->cfg);
		if (n_ops[c] > RKT_REGCMD_MAX_OPS)
			return ERR_INTERNAL;
		if (rkt_bo_flush(&p->dev, &a->in[c]) || rkt_bo_flush(&p->dev, &a->regcmd[c]))
			return ERR_IO;
	}

	status_code s		  = OK;
	int			submitted = 0;
	for (; submitted < n_blocks && s == OK; submitted++) {
		uint32_t regcmd_dma = (uint32_t)a->regcmd[submitted].dma_addr;
		if (rkt_submit(&p->dev, &regcmd_dma, &n_ops[submitted], 1, &a->regcmd[submitted],
					   &a->in[submitted], &a->out[submitted]))
			s = ERR_COMPUTE_FAIL;
	}
	for (int c = 0; c < submitted; c++)
		if (rkt_bo_wait(&p->dev, &a->out[c]) && s == OK)
			s = ERR_COMPUTE_FAIL;
	if (s != OK)
		return s;

	float inv = 1.0f / (float)a->cfg.F;
	for (int c = 0; c < n_blocks; c++) {
		const act_block *b	= &blocks[c];
		const float		*om = a->out[c].map;
		for (int r = 0; r < b->rows; r++) {
			int	   pixel = b->first + r;
			int	   col0	 = (pixel % chunks_per_row) * RKT_ACT_CHUNK;
			int	   cols	 = n - col0 < RKT_ACT_CHUNK ? n - col0 : RKT_ACT_CHUNK;
			float *dst	 = out + (size_t)(pixel / chunks_per_row) * n + col0;
			for (int nn = 0; nn < cols; nn++)
				dst[nn] = om[(size_t)(nn / RKT_OUT_C2_F32) * b->rows * RKT_OUT_C2_F32 +
							 (size_t)r * RKT_OUT_C2_F32 + nn % RKT_OUT_C2_F32] *
						  inv;
		}
	}
	return OK;
}

static status_code rkt_lut_act_rows(rkt_priv *p, rkt_act_cache *a, rkt_lut_kind kind,
									const float *gate, int gate_stride, float *out, int n, int m) {
	status_code s = rkt_act_cache_select(p, a, kind);
	if (s != OK)
		return s;

	int chunks_per_row = (n + RKT_ACT_CHUNK - 1) / RKT_ACT_CHUNK;
	int pixels		   = m * chunks_per_row;
	if (!ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)pixels * RKT_ACT_CHUNK,
						 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	for (int r = 0; r < m; r++) {
		uint16_t *dst = p->x16 + (size_t)r * chunks_per_row * RKT_ACT_CHUNK;
		convert_to_f16(dst, gate + (size_t)r * gate_stride, (size_t)n);
		memset(dst + n, 0, ((size_t)chunks_per_row * RKT_ACT_CHUNK - (size_t)n) * sizeof(uint16_t));
	}

	act_block blocks[RKT_MAX_CORES];
	for (int done = 0; done < pixels && s == OK;) {
		int n_blocks = 0;
		while (n_blocks < p->dev.n_cores && done < pixels) {
			int rows		   = next_block_rows(pixels - done, RKT_ACT_BLOCK_ROWS);
			blocks[n_blocks++] = (act_block){.first = done, .rows = rows};
			done += rows;
		}
		s = rkt_act_round(p, a, p->x16, blocks, n_blocks, out, n, chunks_per_row);
	}
	return s;
}

static status_code rkt_ffn_activate_kind(backend *self, const buffer *gate, const buffer *up,
										 buffer *out, int n, int activation, int m, int gate_stride,
										 int up_stride, int fused) {
	rkt_priv *p		= self->priv;
	float	 *ucopy = NULL;
	if (n <= 0 || m <= 0)
		return OK;
	if (activation != RKT_FFNA_SILU && activation != RKT_FFNA_GELU)
		goto host;
	rkt_lut_kind kind = activation == RKT_FFNA_GELU ? RKT_LUT_GELU_TANH : RKT_LUT_SILU;

	const float *gf = rkt_ptr(gate);
	const float *uf = rkt_ptr(up);
	float		*of = rkt_ptr(out);
	if (uf == of) {
		ucopy = xmalloc((size_t)n * (size_t)m * sizeof(float));
		for (int r = 0; r < m; r++)
			memcpy(ucopy + (size_t)r * n, uf + (size_t)r * up_stride, (size_t)n * sizeof(float));
		uf = ucopy;
	}

	int chunks_per_row = (n + RKT_ACT_CHUNK - 1) / RKT_ACT_CHUNK;
	if ((size_t)m * chunks_per_row >
		(size_t)p->dev.n_cores * RKT_ACT_BLOCK_ROWS * RKT_ACT_MAX_ROUNDS) {
	host:
		free(ucopy);
		if (fused || gate_stride != n || up_stride != n)
			return ERR_UNSUPPORTED;
		backend_report_host_fallback(self, "ffn_activate", HFB_OP_NOT_NATIVE,
									 "shape n=%d m=%d act=%d", n, m, activation);
		backend *host = backend_host();
		if (!host || !host->ffn_activate_batch)
			return ERR_UNSUPPORTED;
		return host->ffn_activate_batch(host, gate, up, out, n, activation, m);
	}

	pthread_mutex_lock(&p->lock);
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL
								  : rkt_lut_act_rows(p, &p->act, kind, gf, gate_stride, of, n, m);
	if (s != OK && !p->npu_failed) {
		p->npu_failed = 1;
		WARN("rocket: npu ffn_activate failed (n=%d m=%d status=%d, errno=%d), npu disabled", n, m,
			 s, errno);
	}
	pthread_mutex_unlock(&p->lock);
	if (s != OK)
		goto host;

	int u_stride = ucopy ? n : up_stride;
	for (int r = 0; r < m; r++) {
		const float *g = of + (size_t)r * n;
		const float *u = uf + (size_t)r * u_stride;
		float		*o = of + (size_t)r * n;
		for (int i = 0; i < n; i++)
			o[i] = g[i] * u[i];
	}
	free(ucopy);
	return OK;
}

static status_code rkt_ffn_activate(backend *self, const buffer *gate, const buffer *up,
									buffer *out, int n) {
	return rkt_ffn_activate_kind(self, gate, up, out, n, RKT_FFNA_SILU, 1, n, n, 0);
}

static status_code rkt_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up,
									   buffer *out, int n, int activation) {
	return rkt_ffn_activate_kind(self, gate, up, out, n, activation, 1, n, n, 0);
}

static status_code rkt_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up,
										  buffer *out, int n, int activation, int m) {
	return rkt_ffn_activate_kind(self, gate, up, out, n, activation, m, n, n, 0);
}

static status_code rkt_ffn_activate_fused_batch(backend *self, const buffer *fused, buffer *out,
												int n, int activation, int m) {
	buffer up =
		buffer_slice(fused, (size_t)n * sizeof(float), (size_t)m * (size_t)n * sizeof(float));
	return rkt_ffn_activate_kind(self, fused, &up, out, n, activation, m, 2 * n, 2 * n, 1);
}

static status_code rkt_matmul_ffn_down_batch(backend *self, const buffer *w, uint32_t w_type,
											 const buffer *gate, const buffer *up, buffer *y, int n,
											 int k, int activation, int m) {
	rkt_priv *p = self->priv;
	if (m <= 0)
		return ERR_INVALID_ARG;
	if (!ensure_capacity((void **)&p->act_tmp, &p->act_tmp_cap, (size_t)m * k, sizeof(float)))
		return ERR_OUT_OF_MEMORY;

	buffer		act_buf = buffer_host_view(self, p->act_tmp, (size_t)m * k * sizeof(float));
	status_code s = rkt_ffn_activate_kind(self, gate, up, &act_buf, k, activation, m, k, k, 0);
	if (s != OK)
		return s;
	return rkt_matmul_batch(self, w, w_type, &act_buf, y, n, k, m);
}

static status_code rkt_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type,
									   const buffer *gate, const buffer *up, buffer *y, int n,
									   int k, int activation) {
	return rkt_matmul_ffn_down_batch(self, w, w_type, gate, up, y, n, k, activation, 1);
}

static status_code rkt_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out) {
	rkt_priv *p = self->priv;
	if (desc->kv_quant != KV_QUANT_F16 || desc->n_layers < 1 || desc->n_ctx < 1)
		return ERR_UNSUPPORTED;
	if (rkt_align_up(desc->n_ctx, RKT_K_ALIGN) > RKT_MAX_K)
		return ERR_UNSUPPORTED;
	rkt_kv *kv		= xcalloc(1, sizeof(*kv));
	kv->magic		= RKT_KV_MAGIC;
	kv->n_slots		= desc->n_layers;
	kv->slots		= xcalloc((size_t)desc->n_layers, sizeof(*kv->slots));
	int			pos = rkt_align_up(desc->n_ctx, RKT_K_ALIGN);
	status_code s	= OK;
	for (int i = 0; i < desc->n_layers && s == OK; i++) {
		rkt_kv_slot *sl = &kv->slots[i];
		sl->hd			= desc->layer_head_dim ? desc->layer_head_dim[i] : desc->head_dim;
		sl->kvh			= desc->layer_n_kv_heads ? desc->layer_n_kv_heads[i] : desc->n_kv_heads;
		sl->pos_pad		= pos;
		sl->hd_pad		= rkt_align_up(sl->hd, RKT_K_ALIGN);
		if (sl->hd < 1 || sl->kvh < 1 || sl->hd_pad > RKT_MAX_K ||
			(p->max_k < sl->pos_pad || p->max_k < sl->hd_pad)) {
			s = ERR_INVALID_ARG;
			break;
		}
		sl->k = kv_weight_build(p, sl->kvh * sl->pos_pad, sl->hd, sl->hd_pad);
		sl->v = kv_weight_build(p, sl->kvh * sl->hd_pad, sl->pos_pad, sl->pos_pad);
		if (!sl->k || !sl->v)
			s = ERR_OUT_OF_MEMORY;
	}
	if (s != OK) {
		for (int i = 0; i < desc->n_layers; i++) {
			if (kv->slots[i].k)
				destroy_weight(p, kv->slots[i].k);
			if (kv->slots[i].v)
				destroy_weight(p, kv->slots[i].v);
		}
		free(kv->slots);
		free(kv);
		return s;
	}
	size_t bytes = (size_t)desc->n_layers * desc->n_kv_heads * pos *
				   rkt_align_up(desc->head_dim, RKT_K_ALIGN) * sizeof(uint16_t) * 2;
	*k_out		 = (buffer){.handle = kv, .size = bytes, .owner = self};
	*v_out		 = *k_out;
	return OK;
}

static void rkt_kv_free(backend *self, buffer *k, buffer *v) {
	(void)v;
	rkt_priv *p	 = self->priv;
	rkt_kv	 *kv = k ? k->handle : NULL;
	if (!kv || kv->magic != RKT_KV_MAGIC)
		return;
	rkt_kv_destroy(p, kv);
	*k = (buffer){0};
	if (v)
		*v = (buffer){0};
}

static const rkt_kv_slot *kv_slot_of(const buffer *cache, int layer) {
	rkt_kv *kv = cache ? cache->handle : NULL;
	if (!kv || kv->magic != RKT_KV_MAGIC || layer < 0 || layer >= kv->n_slots)
		return NULL;
	return &kv->slots[layer];
}

static status_code rkt_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
									const buffer *k_in, const buffer *v_in, int in_row_stride,
									int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
									int m) {
	(void)n_ctx;
	(void)n_kv_heads;
	rkt_priv		  *p  = self->priv;
	const rkt_kv_slot *sl = kv_slot_of(k, layer);
	if (!sl || sl->hd != head_dim || !v || v->handle != k->handle) {
		ERROR("rocket: kv_put rejected: slot=%d sl=%p sl.hd=%d head_dim=%d handle_match=%d", layer,
			  (const void *)sl, sl ? sl->hd : -1, head_dim, v && k && v->handle == k->handle);
		return ERR_INVALID_ARG;
	}
	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : sl->kvh;
	if (n_active > sl->kvh || m < 1 || pos_start < 0 || pos_start + m > sl->pos_pad) {
		ERROR("rocket: kv_put rejected: slot=%d kvh=%d sl.kvh=%d n_active=%d m=%d pos=%d "
			  "pos_pad=%d",
			  layer, n_kv_heads_active, sl->kvh, n_active, m, pos_start, sl->pos_pad);
		return ERR_INVALID_ARG;
	}
	rkt_kchunk *kc = &sl->k->kchunks[0];
	rkt_kchunk *vc = &sl->v->kchunks[0];
	if (kc->k_len > RKT_MAX_K || vc->k_len > RKT_MAX_K)
		return ERR_UNSUPPORTED;
	const float *kf	 = rkt_ptr(k_in);
	const float *vf	 = rkt_ptr(v_in);
	uint16_t	*row = NULL;
	status_code	 s	 = OK;
	pthread_mutex_lock(&p->lock);
	if (!ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)kc->k_len, sizeof(uint16_t)))
		s = ERR_OUT_OF_MEMORY;
	row = (uint16_t *)p->x16;
	for (int r = 0; s == OK && r < m; r++) {
		int pos = pos_start + r;
		for (int h = 0; s == OK && h < n_active; h++) {
			const float *krow = kf + (size_t)r * in_row_stride + (size_t)h * head_dim;
			const float *vrow = vf + (size_t)r * in_row_stride + (size_t)h * head_dim;
			convert_to_f16(row, krow, (size_t)head_dim);
			memset(row + head_dim, 0, (size_t)(kc->k_len - head_dim) * sizeof(uint16_t));
			int				k_global = h * sl->pos_pad + pos;
			const rkt_part *pt		 = part_for_row(kc, k_global);
			if (!pt) {
				s = ERR_INTERNAL;
				break;
			}
			tile_weight_row(pt->bo.map, row, kc->k_len, k_global - pt->row_start);
			int v_col = pos / RKT_TILE_COLS, v_rem = pos % RKT_TILE_COLS;
			for (int d = 0; d < head_dim; d++) {
				int				v_global = h * sl->hd_pad + d;
				const rkt_part *vp		 = part_for_row(vc, v_global);
				if (!vp) {
					s = ERR_INTERNAL;
					break;
				}
				int		  local = v_global - vp->row_start;
				size_t	  tile	= ((size_t)(local / RKT_TILE_ROWS) * (vc->k_len / RKT_TILE_COLS) +
								   (size_t)v_col) *
								  (RKT_TILE_ROWS * RKT_TILE_COLS);
				uint16_t *vmap	= (uint16_t *)vp->bo.map;
				vmap[tile + (size_t)(local % RKT_TILE_ROWS) * RKT_TILE_COLS + v_rem] =
					f32_to_f16_bits(vrow[d]);
			}
		}
	}
	if (s == OK) {
		for (int c = 0; c < RKT_MAX_CORES; c++) {
			if (kc->part[c].bo.map && rkt_bo_flush(&p->dev, &kc->part[c].bo))
				s = ERR_IO;
			if (vc->part[c].bo.map && rkt_bo_flush(&p->dev, &vc->part[c].bo))
				s = ERR_IO;
		}
	}
	pthread_mutex_unlock(&p->lock);
	return s;
}

static status_code rkt_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos,
							  const buffer *k_in, const buffer *v_in, int n_kv_heads, int head_dim,
							  int n_ctx, int n_kv_heads_active) {
	return rkt_kv_put_batch(self, k, v, layer, pos, k_in, v_in, n_kv_heads * head_dim, n_kv_heads,
							head_dim, n_ctx, n_kv_heads_active, 1);
}

static void attention_softmax(const float *scores, float *probs, int m, int n_i, int span16,
							  int pos_pad, int pos_start, int window, float scale) {
	for (int r = 0; r < m; r++) {
		int lim	 = pos_start + r + 1;
		int t_lo = window > 0 ? pos_start + r + 1 - window : 0;
		if (t_lo < 0)
			t_lo = 0;
		for (int i = 0; i < n_i; i++) {
			const float *sc = scores + (size_t)(r * n_i + i) * span16;
			float		*pr = probs + (size_t)(r * n_i + i) * pos_pad;
			float		 mx = -INFINITY;
			for (int t = t_lo; t < lim; t++) {
				float v = sc[t] * scale;
				if (v > mx)
					mx = v;
			}
			double sum = 0.0;
			for (int t = t_lo; t < lim; t++) {
				float e = expf(sc[t] * scale - mx);
				pr[t]	= e;
				sum += e;
			}
			float inv = (float)(1.0 / sum);
			for (int t = 0; t < lim; t++)
				pr[t] = f16_bits_to_f32(f32_to_f16_bits(pr[t] * inv));
		}
	}
}

static status_code rkt_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
									   const buffer *v_cache, buffer *out, int layer, int pos_start,
									   int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									   int flash_attn, float scale, int n_kv_heads_active, int m,
									   int window) {
	(void)flash_attn;
	(void)n_ctx;
	(void)n_kv_heads;
	rkt_priv		  *p  = self->priv;
	const rkt_kv_slot *sl = kv_slot_of(k_cache, layer);
	if (!sl || sl->hd != head_dim || !v_cache || v_cache->handle != k_cache->handle)
		return ERR_INVALID_ARG;
	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : sl->kvh;
	if (n_active > sl->kvh || n_heads < n_active || n_heads % n_active != 0 || m < 1)
		return ERR_INVALID_ARG;
	int n_groups = n_heads / n_active;
	int span	 = pos_start + m;
	if (pos_start < 0 || span > sl->pos_pad)
		return ERR_INVALID_ARG;
	int	   span16		= rkt_align_up(span, RKT_N_ALIGN);
	int	   M			= m * n_groups;
	size_t group_floats = (size_t)M * (2 * head_dim + span16 + sl->pos_pad);
	int	   batch		= (int)(RKT_ATTN_BATCH_FLOATS / group_floats);
	batch				= batch < 1 ? 1 : batch > n_active ? n_active : batch;
	float *qf			= rkt_ptr(q);
	float *outf			= rkt_ptr(out);
	if (!ensure_capacity((void **)&p->attn_ws, &p->attn_ws_cap, group_floats * batch,
						 sizeof(float)))
		return ERR_OUT_OF_MEMORY;

	pthread_mutex_lock(&p->lock);
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL : OK;
	for (int g0 = 0; s == OK && g0 < n_active; g0 += batch) {
		int count = n_active - g0 < batch ? n_active - g0 : batch;
		for (int i = 0; i < count && s == OK; i++) {
			int	   g	  = g0 + i;
			float *q_buf  = p->attn_ws + (size_t)i * group_floats;
			float *scores = q_buf + (size_t)2 * M * head_dim;
			float *probs  = scores + (size_t)M * span16;
			memset(probs, 0, (size_t)M * sl->pos_pad * sizeof(float));
			for (int r = 0; r < m; r++)
				for (int h = 0; h < n_groups; h++)
					memcpy(q_buf + (size_t)(r * n_groups + h) * head_dim,
						   qf + ((size_t)r * n_heads + (size_t)g * n_groups + h) * head_dim,
						   (size_t)head_dim * sizeof(float));
			s = npu_matmul_enqueue(p, sl->k, g * sl->pos_pad, span16, q_buf, scores, M);
		}
		if (s == OK)
			s = flush_queue(p);

		for (int i = 0; i < count && s == OK; i++) {
			float *q_buf   = p->attn_ws + (size_t)i * group_floats;
			float *out_buf = q_buf + (size_t)M * head_dim;
			float *scores  = q_buf + (size_t)2 * M * head_dim;
			float *probs   = scores + (size_t)M * span16;
			attention_softmax(scores, probs, m, n_groups, span16, sl->pos_pad, pos_start, window,
							  scale);
			s = npu_matmul_enqueue(p, sl->v, (g0 + i) * sl->hd_pad, head_dim, probs, out_buf, M);
		}
		if (s == OK)
			s = flush_queue(p);

		for (int i = 0; i < count && s == OK; i++) {
			const float *out_buf = p->attn_ws + (size_t)i * group_floats + (size_t)M * head_dim;
			for (int r = 0; r < m; r++)
				for (int h = 0; h < n_groups; h++)
					memcpy(outf +
							   ((size_t)r * n_heads + (size_t)(g0 + i) * n_groups + h) * head_dim,
						   out_buf + (size_t)(r * n_groups + h) * head_dim,
						   (size_t)head_dim * sizeof(float));
		}
	}
	if (s != OK) {
		reset_queue(p);
		if (!p->npu_failed) {
			p->npu_failed = 1;
			WARN("rocket: npu attention failed (status %d, errno=%d), npu disabled", s, errno);
		}
	}
	pthread_mutex_unlock(&p->lock);
	return s;
}

static status_code rkt_attention(backend *self, const buffer *q, const buffer *k_cache,
								 const buffer *v_cache, buffer *out, int layer, int pos,
								 int n_heads, int n_kv_heads, int head_dim, int n_ctx,
								 int flash_attn, float scale, int n_kv_heads_active) {
	return rkt_attention_batch(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							   head_dim, n_ctx, flash_attn, scale, n_kv_heads_active, 1, 0);
}

static status_code rkt_attention_swa(backend *self, const buffer *q, const buffer *k_cache,
									 const buffer *v_cache, buffer *out, int layer, int pos,
									 int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									 int flash_attn, float scale, int sliding_window,
									 int n_kv_heads_active) {
	if (sliding_window < 1)
		return ERR_INVALID_ARG;
	return rkt_attention_batch(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							   head_dim, n_ctx, flash_attn, scale, n_kv_heads_active, 1,
							   sliding_window);
}

static status_code rkt_attention_swa_batch(backend *self, const buffer *q, const buffer *k_cache,
										   const buffer *v_cache, buffer *out, int layer,
										   int pos_start, int n_heads, int n_kv_heads, int head_dim,
										   int n_ctx, int flash_attn, float scale,
										   int sliding_window, int n_kv_heads_active, int m) {
	if (sliding_window < 1)
		return ERR_INVALID_ARG;
	return rkt_attention_batch(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
							   n_kv_heads, head_dim, n_ctx, flash_attn, scale, n_kv_heads_active, m,
							   sliding_window);
}

static status_code rkt_attention_batch_fn(backend *self, const buffer *q, const buffer *k_cache,
										  const buffer *v_cache, buffer *out, int layer,
										  int pos_start, int n_heads, int n_kv_heads, int head_dim,
										  int n_ctx, int flash_attn, float scale,
										  int n_kv_heads_active, int m) {
	return rkt_attention_batch(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
							   n_kv_heads, head_dim, n_ctx, flash_attn, scale, n_kv_heads_active, m,
							   0);
}

static status_code rkt_probe(void) {
	return access(device_path(), R_OK | W_OK) == 0 ? OK : ERR_UNSUPPORTED;
}

static void free_slots(rkt_priv *p) {
	for (int cls = 0; cls < SLOT_CLASSES; cls++)
		for (int c = 0; c < RKT_MAX_CORES; c++) {
			rkt_bo_destroy(&p->dev, &p->slots[cls][c].in);
			rkt_bo_destroy(&p->dev, &p->slots[cls][c].out);
		}
	for (int c = 0; c < RKT_MAX_CORES; c++)
		rkt_bo_destroy(&p->dev, &p->streams[c].bo);
}

static void rkt_free(backend *self) {
	rkt_priv *p = self->priv;
	if (!p)
		return;
	for (size_t i = 0; i < p->n_weights; i++)
		destroy_weight(p, p->weights[i]);
	if (p->fused_groups)
		log_tag("RKT", "fused %d weights into %d larger job sets", p->fused_weights,
				p->fused_groups);
	free_slots(p);
	rkt_act_cache_reset(p, &p->act);
	rkt_device_close(&p->dev);
	pthread_mutex_destroy(&p->lock);
	free(p->weights);
	free(p->x16);
	free(p->x_tile);
	free(p->x_chunk);
	free(p->y_tmp);
	free(p->attn_ws);
	free(p->act_tmp);
	free(p->arena);
	free(p);
	self->priv = NULL;
}

static status_code rkt_init(backend *self, int device_index) {
	(void)device_index;
	rkt_priv *p = xcalloc(1, sizeof(*p));
	p->max_k =
		env_int("KAPPAI_ROCKET_MAX_K", RKT_MAX_K, RKT_K_ALIGN, RKT_MAX_K) & ~(RKT_K_ALIGN - 1);
	p->poison	 = env_int("KAPPAI_ROCKET_POISON", 0, 0, 1);
	p->fuse		 = env_int("KAPPAI_ROCKET_FUSE", 1, 0, 1);
	p->max_tasks = env_int("KAPPAI_ROCKET_CHAIN", 1, 0, 1) ? RKT_MAX_TASKS : 1;
	pthread_mutex_init(&p->lock, NULL);
	self->priv = p;

	int n_cores = env_int("KAPPAI_ROCKET_NCORES", RKT_MAX_CORES, 1, RKT_MAX_CORES);
	int err		= rkt_device_open(&p->dev, device_path(), n_cores, RKT_TIMEOUT_NS);
	if (err) {
		WARN("rocket: cannot open %s: %s", device_path(), strerror(-err));
		rkt_free(self);
		return ERR_UNSUPPORTED;
	}
	log_tag("RKT", "%s: %d core(s), max_k=%d, poison=%d, tasks/job=%d", device_path(),
			p->dev.n_cores, p->max_k, p->poison, p->max_tasks);
	return OK;
}

static status_code rkt_ctor(backend *out) {
	memset(out, 0, sizeof(*out));
	out->name				 = "rocket";
	out->desc				 = "Rockchip RK3588 NPU (rocket)";
	out->priority			 = RKT_PRIORITY;
	out->caps				 = BCAP_HOST_VISIBLE_BUFFERS | BCAP_MULTI_MATMUL | BCAP_MATMUL_RESIDUAL;
	out->probe				 = rkt_probe;
	out->init				 = rkt_init;
	out->free				 = rkt_free;
	out->buffer_alloc_weight = rkt_buffer_alloc_weight;
	out->buffer_alloc_scratch = rkt_buffer_alloc_scratch;
	out->buffer_free		  = rkt_buffer_free;
	out->buffer_read_f32	  = rkt_buffer_read_f32;
	out->buffer_write_f32	  = rkt_buffer_write_f32;
	out->matmul				  = rkt_matmul;
	out->matmul_batch		  = rkt_matmul_batch;
	out->matmul_multi		  = rkt_matmul_multi;
	out->matmul_multi_batch	  = rkt_matmul_multi_batch;
	out->matmul_residual	  = rkt_matmul_residual;
	out->ffn_activate		  = rkt_ffn_activate;
	out->ffn_activate_ex	  = rkt_ffn_activate_ex;
	out->ffn_activate_batch	  = rkt_ffn_activate_batch;
	// BUG: Temporarily disabled: out->ffn_activate_fused_batch = rkt_ffn_activate_fused_batch;
	out->matmul_ffn_down	   = rkt_matmul_ffn_down;
	out->matmul_ffn_down_batch = rkt_matmul_ffn_down_batch;
	out->attention			   = rkt_attention;
	out->attention_batch	   = rkt_attention_batch_fn;
	out->attention_swa		   = rkt_attention_swa;
	out->attention_swa_batch   = rkt_attention_swa_batch;
	out->kv_alloc			   = rkt_kv_alloc;
	out->kv_free			   = rkt_kv_free;
	out->kv_put				   = rkt_kv_put;
	out->kv_put_batch		   = rkt_kv_put_batch;
	return OK;
}

BACKEND_REGISTER("rocket", rkt_ctor)
