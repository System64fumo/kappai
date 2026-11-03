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
	rkt_bo regcmd;
} rkt_slot;

typedef enum {
	SLOT_DECODE,
	SLOT_BATCH,
	SLOT_CLASSES,
} slot_class;

struct rkt_act_cache {
	int			n;
	int			k;
	int			uploaded_kind;
	rkt_lut_cfg cfg;
	int			n_weight_cores;
	int			weight_core_start[RKT_MAX_CORES];
	int			weight_core_count[RKT_MAX_CORES];
	rkt_bo		weight_core[RKT_MAX_CORES];
	rkt_bo		in[RKT_MAX_CORES];
	rkt_bo		out[RKT_MAX_CORES];
	rkt_bo		regcmd[RKT_MAX_CORES];
};
typedef struct rkt_act_cache rkt_act_cache;

typedef struct {
	rkt_device		dev;
	pthread_mutex_t lock;
	int				max_k;
	int				poison;
	int				npu_failed;
	rkt_slot		slots[SLOT_CLASSES][RKT_MAX_CORES];
	uint16_t	   *x16;
	size_t			x16_cap;
	uint16_t	   *x_tile;
	size_t			x_tile_cap;
	uint16_t	   *x_chunk;
	size_t			x_chunk_cap;
	rkt_weight	  **weights;
	size_t			n_weights;
	size_t			cap_weights;
	int				fuse;
	int				fused_groups;
	int				fused_weights;
	float		   *y_tmp;
	size_t			y_tmp_cap;
	float		   *attn_ws;
	size_t			attn_ws_cap;
	rkt_act_cache	act;
	float		   *act_tmp;
	size_t			act_tmp_cap;
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
	for (int r = 0; r < m; r++)
		for (int c = 0; c < k; c++)
			dst[(size_t)(c / RKT_FEATURE_C2_F16) * m * RKT_FEATURE_C2_F16 +
				(size_t)r * RKT_FEATURE_C2_F16 + c % RKT_FEATURE_C2_F16] = src[(size_t)r * k + c];
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
	return ensure_bo(p, core, in_bytes, &s->in) && ensure_bo(p, core, out_bytes, &s->out) &&
		   ensure_bo(p, core, RKT_REGCMD_BYTES, &s->regcmd);
}

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
	slot_class		  cls;
	float			 *y;
} block_ctx;

static int poison_hits(const uint32_t *words, size_t count) {
	for (size_t i = 0; i < count; i++)
		if (words[i] == RKT_POISON_WORD)
			return 1;
	return 0;
}

static status_code prepare_job(rkt_priv *p, const block_ctx *b, int blk, const job_desc *j,
							   uint32_t *n_ops) {
	size_t in_bytes	 = (size_t)blk * b->kc->k_len * sizeof(uint16_t);
	size_t out_bytes = (size_t)blk * j->rows * sizeof(float);
	if (!ensure_slot(p, b->cls, j->core, in_bytes, out_bytes))
		return ERR_OUT_OF_MEMORY;

	rkt_slot	   *s  = &p->slots[b->cls][j->core];
	const rkt_part *pt = &b->kc->part[0];
	for (int c = 0; c < RKT_MAX_CORES; c++)
		if (b->kc->part[c].bo.map && b->kc->part[c].bo.core == j->core)
			pt = &b->kc->part[c];

	uint32_t w_dma =
		(uint32_t)(pt->bo.dma_addr + (size_t)(j->row_start - pt->row_start) * b->kc->k_len * 2);
	*n_ops =
		(uint32_t)rkt_emit_matmul_f16(s->regcmd.map, blk, b->kc->k_len, j->rows,
									  (uint32_t)s->in.dma_addr, w_dma, (uint32_t)s->out.dma_addr);
	if (*n_ops > RKT_REGCMD_MAX_OPS)
		return ERR_INTERNAL;

	memcpy(s->in.map, p->x_tile, in_bytes);
	if (rkt_bo_flush(&p->dev, &s->in) || rkt_bo_flush(&p->dev, &s->regcmd))
		return ERR_IO;

	if (!p->poison)
		return OK;
	uint32_t *ow = s->out.map;
	for (size_t i = 0; i < (size_t)blk * j->rows; i++)
		ow[i] = RKT_POISON_WORD;
	return rkt_bo_flush(&p->dev, &s->out) ? ERR_IO : OK;
}

static status_code collect_job(rkt_priv *p, const block_ctx *b, int blk, const job_desc *j) {
	const float *om = p->slots[b->cls][j->core].out.map;
	if (p->poison && poison_hits((const uint32_t *)om, (size_t)blk * j->rows)) {
		ERROR("rocket: core %d output still holds poison (npu did not write)", j->core);
		return ERR_COMPUTE_FAIL;
	}

	for (int nn = 1; nn <= j->rows; nn++) {
		int row = j->row_start + nn - 1;
		if (row < b->row0 || row >= b->row0 + b->n)
			continue;
		for (int r = 1; r <= blk; r++) {
			int	   idx = blk == 1 ? nn - 1 : rkt_feature_index(blk, RKT_OUT_C2_F32, nn, r);
			float *dst = &b->y[(size_t)(b->m_done + r - 1) * b->n + (row - b->row0)];
			if (b->accumulate)
				*dst += om[idx];
			else
				*dst = om[idx];
		}
	}
	return OK;
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

static status_code run_jobs(rkt_priv *p, const block_ctx *b, int blk, job_desc *jobs, int n_jobs) {
	uint32_t	n_ops[RKT_MAX_CORES];
	status_code s = OK;

	for (int i = 0; i < n_jobs && s == OK; i++)
		s = prepare_job(p, b, blk, &jobs[i], &n_ops[i]);

	int submitted = 0;
	for (; submitted < n_jobs && s == OK; submitted++) {
		rkt_slot *sl = &p->slots[b->cls][jobs[submitted].core];
		if (rkt_submit(&p->dev, &sl->regcmd, n_ops[submitted], &sl->in, &sl->out))
			s = ERR_COMPUTE_FAIL;
	}
	for (int i = 0; i < submitted; i++) {
		if (rkt_bo_wait(&p->dev, &p->slots[b->cls][jobs[i].core].out) && s == OK)
			s = ERR_COMPUTE_FAIL;
	}
	for (int i = 0; i < n_jobs && s == OK; i++)
		s = collect_job(p, b, blk, &jobs[i]);
	return s;
}

static status_code run_block(rkt_priv *p, block_ctx *b, int blk, const uint16_t *x_block) {
	int lo = b->row0 & ~(RKT_N_ALIGN - 1);
	int hi = (b->row0 + b->n + RKT_N_ALIGN - 1) & ~(RKT_N_ALIGN - 1);
	if (hi > b->w->n_pad)
		hi = b->w->n_pad;

	if (!ensure_capacity((void **)&p->x_tile, &p->x_tile_cap, (size_t)blk * b->kc->k_len,
						 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	tile_input(p->x_tile, x_block, blk, b->kc->k_len);

	job_desc jobs[RKT_MAX_CORES];
	for (int round = 0;; round++) {
		int n_jobs = plan_jobs(b, lo, hi, round, jobs);
		if (n_jobs == 0)
			return OK;
		status_code s = run_jobs(p, b, blk, jobs, n_jobs);
		if (s != OK)
			return s;
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

static status_code npu_matmul_fold(rkt_priv *p, const rkt_weight *w, int row0, int n,
								   const uint16_t *x16, float *y, int m) {
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
			b.cls		  = blk == 1 ? SLOT_DECODE : SLOT_BATCH;
			status_code s = run_block(p, &b, blk, xc + (size_t)done * kc->k_len);
			if (s != OK)
				return s;
			done += blk;
		}
	}

	const rkt_kchunk *kc = &w->kchunks[0];
	if (!ensure_capacity((void **)&p->x_chunk, &p->x_chunk_cap, (size_t)m * RKT_FOLD_K_LEN,
						 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	memset(p->x_chunk, 0, (size_t)m * RKT_FOLD_K_LEN * sizeof(uint16_t));
	for (int r = 0; r < m; r++)
		p->x_chunk[(size_t)r * RKT_FOLD_K_LEN] = x16[(size_t)r * w->k_pad + w->k];

	rkt_kchunk fold_kc = {
		.k_off = w->k,
		.k_len = RKT_FOLD_K_LEN,
	};
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		fold_kc.part[c]	   = kc->part[c];
		fold_kc.part[c].bo = kc->fold_bo[c];
	}
	block_ctx fb = {.w = w, .kc = &fold_kc, .row0 = row0, .n = n, .m = m, .y = y, .accumulate = 1};
	for (int done = 0; done < m;) {
		int blk		  = next_block_rows(m - done, RKT_TILE_ROWS);
		fb.m_done	  = done;
		fb.cls		  = blk == 1 ? SLOT_DECODE : SLOT_BATCH;
		status_code s = run_block(p, &fb, blk, p->x_chunk + (size_t)done * RKT_FOLD_K_LEN);
		if (s != OK)
			return s;
		done += blk;
	}
	return OK;
}

static status_code npu_matmul(rkt_priv *p, const rkt_weight *w, int row0, int n, const float *x,
							  float *y, int m) {
	if (!ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)m * w->k_pad, sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	for (int r = 0; r < m; r++) {
		uint16_t *dst = p->x16 + (size_t)r * w->k_pad;
		convert_to_f16(dst, x + (size_t)r * w->k, (size_t)w->k);
		memset(dst + w->k, 0, (size_t)(w->k_pad - w->k) * sizeof(uint16_t));
	}

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
			b.cls		  = blk == 1 ? SLOT_DECODE : SLOT_BATCH;
			status_code s = run_block(p, &b, blk, xc + (size_t)done * kc->k_len);
			if (s != OK)
				return s;
			done += blk;
		}
	}
	return OK;
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
		log_tag("ROCKET", "fusing projections that share an input into larger jobs "
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
				memset(kc->fold_bo[c].map, 0, kc->fold_bo[c].size);
				uint16_t wr[RKT_FOLD_K_LEN];
				for (int rr = 0; rr < kc->part[c].row_count; rr++) {
					int global = kc->part[c].row_start + rr;
					memset(wr, 0, sizeof(wr));
					if (global >= row0 && global < row0 + n)
						wr[0] = f32_to_f16_bits(r[global - row0]);
					tile_weight_row(kc->fold_bo[c].map, wr, RKT_FOLD_K_LEN, rr);
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

static void rkt_lut_build(rkt_lut_cfg *cfg, uint16_t *le) {
	cfg->kind		= RKT_LUT_NONE;
	cfg->sel		= RKT_LUT_HQ_SEL;
	cfg->S			= 256;
	cfg->F			= 2048;
	cfg->start		= -(RKT_LUT_WINDOW_HALF << cfg->sel);
	cfg->end		= (RKT_LUT_WINDOW_HALF << cfg->sel) - 1;
	cfg->tail_slope = cfg->F / cfg->S;
	if (!le)
		return;

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
		rkt_bo_destroy(&p->dev, &a->weight_core[c]);
		rkt_bo_destroy(&p->dev, &a->in[c]);
		rkt_bo_destroy(&p->dev, &a->out[c]);
		rkt_bo_destroy(&p->dev, &a->regcmd[c]);
	}
	a->n = a->k		  = 0;
	a->n_weight_cores = 0;
	a->uploaded_kind  = -1;
}

static status_code rkt_lut_upload_tables(rkt_priv *p, rkt_act_cache *a, const uint16_t *le) {
	uint16_t chunk[RKT_LUT_UPLOAD_CHUNK];
	for (int off = 0; off < RKT_LUT_ENTRIES; off += RKT_LUT_UPLOAD_CHUNK) {
		int count = RKT_LUT_ENTRIES - off < RKT_LUT_UPLOAD_CHUNK ? RKT_LUT_ENTRIES - off
																 : RKT_LUT_UPLOAD_CHUNK;
		for (int i = 0; i < count; i++)
			chunk[i] = le[off + i];
		for (int c = 0; c < a->n_weight_cores; c++) {
			uint32_t n_ops = (uint32_t)rkt_emit_lut_upload(
				a->regcmd[c].map, 0, off, chunk, count, (uint32_t)a->in[c].dma_addr,
				(uint32_t)a->weight_core[c].dma_addr, (uint32_t)a->out[c].dma_addr);
			if (!n_ops || n_ops > RKT_REGCMD_MAX_OPS)
				return ERR_INTERNAL;
			if (rkt_bo_flush(&p->dev, &a->regcmd[c]))
				return ERR_IO;
			if (rkt_submit(&p->dev, &a->regcmd[c], n_ops, &a->in[c], &a->out[c]))
				return ERR_COMPUTE_FAIL;
			if (rkt_bo_wait(&p->dev, &a->out[c]))
				return ERR_COMPUTE_FAIL;
		}
	}
	return OK;
}

static status_code rkt_act_cache_build(rkt_priv *p, rkt_act_cache *a, int n, int k,
									   rkt_lut_kind kind) {
	if (a->n == n && a->k == k && a->uploaded_kind == (int)kind)
		return OK;
	if (a->n != n || a->k != k) {
		rkt_act_cache_reset(p, a);
		if (k % RKT_K_ALIGN != 0 || k > p->max_k || n % RKT_N_ALIGN != 0)
			return ERR_UNSUPPORTED;
		a->n = n;
		a->k = k;
		rkt_lut_build(&a->cfg, NULL);
		a->cfg.kind = kind;
		int start[RKT_MAX_CORES], count[RKT_MAX_CORES];
		int used = partition_rows(n, p->dev.n_cores, start, count);
		for (int c = 0; c < used; c++) {
			size_t w_bytes	 = (size_t)count[c] * k * sizeof(uint16_t);
			size_t in_bytes	 = (size_t)RKT_TILE_ROWS * k * sizeof(uint16_t);
			size_t out_bytes = (size_t)RKT_TILE_ROWS * count[c] * sizeof(float);
			if (rkt_bo_create(&p->dev, c, w_bytes, &a->weight_core[c]) ||
				rkt_bo_create(&p->dev, c, in_bytes, &a->in[c]) ||
				rkt_bo_create(&p->dev, c, out_bytes, &a->out[c]) ||
				rkt_bo_create(&p->dev, c, RKT_REGCMD_BYTES, &a->regcmd[c]))
				return ERR_OUT_OF_MEMORY;
			a->weight_core_count[c] = count[c];
			a->weight_core_start[c] = start[c];
		}
		a->n_weight_cores = used;

		uint16_t *row = xcalloc((size_t)k, sizeof(uint16_t));
		for (int c = 0; c < used; c++) {
			uint16_t *bo = a->weight_core[c].map;
			for (int r = 0; r < count[c]; r++) {
				memset(row, 0, (size_t)k * sizeof(uint16_t));
				row[start[c] + r] = f32_to_f16_bits((float)a->cfg.S);
				tile_weight_row(bo, row, k, r);
			}
			if (rkt_bo_flush(&p->dev, &a->weight_core[c]))
				return ERR_IO;
		}
		free(row);
	}

	uint16_t le[RKT_LUT_ENTRIES];
	rkt_lut_build(&a->cfg, le);
	a->cfg.kind = kind;
	if (a->uploaded_kind != (int)kind) {
		status_code s = rkt_lut_upload_tables(p, a, le);
		if (s != OK)
			return s;
		a->uploaded_kind = (int)kind;
	}
	return OK;
}

static status_code rkt_lut_act_rows(rkt_priv *p, rkt_act_cache *a, rkt_lut_kind kind,
									const float *gate, int gate_stride, float *out, int n, int m) {
	status_code s = rkt_act_cache_build(p, a, n, n, kind);
	if (s != OK)
		return s;

	if (!ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)RKT_TILE_ROWS * n,
						 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;

	int max_rows = max_batch_rows(n);
	if (max_rows > RKT_TILE_ROWS)
		max_rows = RKT_TILE_ROWS;
	for (int done = 0; done < m && s == OK;) {
		int blk = next_block_rows(m - done, max_rows);

		memset(p->x16, 0, (size_t)blk * n * sizeof(uint16_t));
		for (int r = 0; r < blk; r++)
			convert_to_f16(p->x16 + (size_t)r * n, gate + (size_t)(done + r) * gate_stride,
						   (size_t)n);
		for (int c = 0; c < a->n_weight_cores; c++) {
			tile_input(a->in[c].map, p->x16, blk, a->k);
			if (rkt_bo_flush(&p->dev, &a->in[c]))
				return ERR_IO;
		}

		int n_ops[RKT_MAX_CORES];
		int active[RKT_MAX_CORES];
		int n_active = 0;
		for (int c = 0; c < a->n_weight_cores; c++) {
			int start = a->weight_core_start[c];
			if (start >= n)
				continue;
			int rows = a->weight_core_count[c];
			if (start + rows > n)
				rows = n - start;
			n_ops[c] = (uint32_t)rkt_emit_matmul_f16_lut(
				a->regcmd[c].map, blk, a->k, rows, (uint32_t)a->in[c].dma_addr,
				(uint32_t)a->weight_core[c].dma_addr, (uint32_t)a->out[c].dma_addr, &a->cfg);
			if (n_ops[c] > RKT_REGCMD_MAX_OPS)
				return ERR_INTERNAL;
			if (rkt_bo_flush(&p->dev, &a->regcmd[c]))
				return ERR_IO;
			active[n_active++] = c;
		}
		for (int i = 0; i < n_active && s == OK; i++) {
			int c = active[i];
			if (rkt_submit(&p->dev, &a->regcmd[c], n_ops[c], &a->in[c], &a->out[c]))
				s = ERR_COMPUTE_FAIL;
		}
		for (int i = 0; i < n_active && s == OK; i++) {
			int c = active[i];
			if (rkt_bo_wait(&p->dev, &a->out[c]))
				s = ERR_COMPUTE_FAIL;
		}
		if (s != OK)
			return s;

		for (int i = 0; i < n_active; i++) {
			int c	  = active[i];
			int start = a->weight_core_start[c];
			int rows  = a->weight_core_count[c];
			if (start + rows > n)
				rows = n - start;
			const float *om = a->out[c].map;
			for (int nn = 0; nn < rows; nn++) {
				int col = start + nn;
				for (int r = 0; r < blk; r++) {
					int idx = blk == 1 ? nn : rkt_feature_index(blk, RKT_OUT_C2_F32, nn + 1, r + 1);
					out[(size_t)(done + r) * n + col] = om[idx] / (float)a->cfg.F;
				}
			}
		}
		done += blk;
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

	if (n % RKT_N_ALIGN || n > RKT_MAX_N_CHUNK) {
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
	free(ucopy);
	if (s != OK)
		goto host;

	for (int r = 0; r < m; r++) {
		const float *g = of + (size_t)r * n;
		const float *u = uf + (size_t)r * up_stride;
		float		*o = of + (size_t)r * n;
		for (int i = 0; i < n; i++)
			o[i] = g[i] * u[i];
	}
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
	if ((activation != RKT_FFNA_SILU && activation != RKT_FFNA_GELU) || k % RKT_K_ALIGN ||
		k > RKT_MAX_K || n % RKT_N_ALIGN) {
		backend_report_host_fallback(self, "matmul_ffn_down", HFB_OP_NOT_NATIVE,
									 "shape n=%d k=%d m=%d act=%d", n, k, m, activation);
		backend *host = backend_host();
		if (!host || !host->matmul_ffn_down_batch)
			return ERR_UNSUPPORTED;
		return host->matmul_ffn_down_batch(host, w, w_type, gate, up, y, n, k, activation, m);
	}
	rkt_lut_kind kind = activation == RKT_FFNA_GELU ? RKT_LUT_GELU_TANH : RKT_LUT_SILU;

	const float *gf = rkt_ptr(gate);
	const float *uf = rkt_ptr(up);
	if (!ensure_capacity((void **)&p->act_tmp, &p->act_tmp_cap, (size_t)m * k, sizeof(float)))
		return ERR_OUT_OF_MEMORY;

	pthread_mutex_lock(&p->lock);
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL
								  : rkt_lut_act_rows(p, &p->act, kind, gf, k, p->act_tmp, k, m);
	if (s != OK && !p->npu_failed) {
		p->npu_failed = 1;
		WARN("rocket: npu ffn down activation failed (k=%d m=%d status=%d, errno=%d), npu disabled",
			 k, m, s, errno);
	}
	pthread_mutex_unlock(&p->lock);
	if (s != OK) {
		backend_report_host_fallback(self, "matmul_ffn_down", HFB_OP_NOT_NATIVE,
									 "npu activation failed (k=%d m=%d status=%d)", k, m, s);
		backend *host = backend_host();
		if (!host || !host->matmul_ffn_down_batch)
			return ERR_UNSUPPORTED;
		return host->matmul_ffn_down_batch(host, w, w_type, gate, up, y, n, k, activation, m);
	}

	for (int r = 0; r < m; r++) {
		const float *g = p->act_tmp + (size_t)r * k;
		const float *u = uf + (size_t)r * k;
		float		*o = p->act_tmp + (size_t)r * k;
		for (int i = 0; i < k; i++)
			o[i] = g[i] * u[i];
	}

	buffer act_buf = buffer_host_view(self, p->act_tmp, (size_t)m * k * sizeof(float));
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
	int n_groups = (n_heads + n_active - 1) / n_active;
	int span	 = pos_start + m;
	if (pos_start < 0 || span > sl->pos_pad)
		return ERR_INVALID_ARG;
	int	   span16 = rkt_align_up(span, RKT_N_ALIGN);
	int	   M	  = m * n_groups;
	float *qf	  = rkt_ptr(q);
	float *outf	  = rkt_ptr(out);
	if (!ensure_capacity((void **)&p->attn_ws, &p->attn_ws_cap,
						 (size_t)M * (2 * head_dim + span16 + sl->pos_pad), sizeof(float)))
		return ERR_OUT_OF_MEMORY;
	float *q_buf   = p->attn_ws;
	float *out_buf = q_buf + (size_t)M * head_dim;
	float *scores  = out_buf + (size_t)M * head_dim;
	float *probs   = scores + (size_t)M * span16;
	memset(probs, 0, (size_t)M * sl->pos_pad * sizeof(float));

	pthread_mutex_lock(&p->lock);
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL : OK;
	for (int g = 0; s == OK && g < n_active; g++) {
		int n_i	 = n_groups * (g + 1) <= n_heads ? n_groups : n_heads - g * n_groups;
		int rows = m * n_i;
		for (int r = 0; r < m; r++)
			for (int i = 0; i < n_i; i++)
				memcpy(q_buf + (size_t)(r * n_i + i) * head_dim,
					   qf + ((size_t)r * n_heads + (size_t)g * n_groups + i) * head_dim,
					   (size_t)head_dim * sizeof(float));
		s = npu_matmul(p, sl->k, g * sl->pos_pad, span16, q_buf, scores, rows);
		if (s != OK)
			break;
		for (int r = 0; r < m && s == OK; r++) {
			int lim	 = pos_start + r + 1;
			int t_lo = window > 0 ? pos_start + r + 1 - window : 0;
			if (t_lo < 0)
				t_lo = 0;
			for (int i = 0; i < n_i; i++) {
				const float *sc = scores + (size_t)(r * n_i + i) * span16;
				float		 mx = -INFINITY;
				for (int t = t_lo; t < lim; t++) {
					float v = sc[t] * scale;
					if (v > mx)
						mx = v;
				}
				double sum = 0.0;
				for (int t = t_lo; t < lim; t++) {
					float e										   = expf(sc[t] * scale - mx);
					probs[(size_t)(r * n_i + i) * sl->pos_pad + t] = e;
					sum += e;
				}
				float  inv = (float)(1.0 / sum);
				float *pr  = probs + (size_t)(r * n_i + i) * sl->pos_pad;
				for (int t = 0; t < lim; t++)
					pr[t] = f16_bits_to_f32(f32_to_f16_bits(pr[t] * inv));
			}
		}
		if (s != OK)
			break;
		s = npu_matmul(p, sl->v, g * sl->hd_pad, head_dim, probs, out_buf, rows);
		if (s != OK)
			break;
		for (int r = 0; r < m; r++)
			for (int i = 0; i < n_i; i++)
				memcpy(outf + ((size_t)r * n_heads + (size_t)g * n_groups + i) * head_dim,
					   out_buf + (size_t)(r * n_i + i) * head_dim,
					   (size_t)head_dim * sizeof(float));
	}
	if (s != OK && !p->npu_failed) {
		p->npu_failed = 1;
		WARN("rocket: npu attention failed (status %d, errno=%d), npu disabled", s, errno);
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
			rkt_bo_destroy(&p->dev, &p->slots[cls][c].regcmd);
		}
}

static void rkt_free(backend *self) {
	rkt_priv *p = self->priv;
	if (!p)
		return;
	for (size_t i = 0; i < p->n_weights; i++)
		destroy_weight(p, p->weights[i]);
	if (p->fused_groups)
		log_tag("ROCKET", "fused %d weights into %d larger job sets", p->fused_weights,
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
	free(p);
	self->priv = NULL;
}

static status_code rkt_init(backend *self, int device_index) {
	(void)device_index;
	rkt_priv *p = xcalloc(1, sizeof(*p));
	p->max_k =
		env_int("KAPPAI_ROCKET_MAX_K", RKT_MAX_K, RKT_K_ALIGN, RKT_MAX_K) & ~(RKT_K_ALIGN - 1);
	p->poison = env_int("KAPPAI_ROCKET_POISON", 0, 0, 1);
	p->fuse	  = env_int("KAPPAI_ROCKET_FUSE", 1, 0, 1);
	pthread_mutex_init(&p->lock, NULL);
	self->priv = p;

	int n_cores = env_int("KAPPAI_ROCKET_NCORES", RKT_MAX_CORES, 1, RKT_MAX_CORES);
	int err		= rkt_device_open(&p->dev, device_path(), n_cores, RKT_TIMEOUT_NS);
	if (err) {
		WARN("rocket: cannot open %s: %s", device_path(), strerror(-err));
		rkt_free(self);
		return ERR_UNSUPPORTED;
	}
	log_tag("ROCKET", "%s: %d core(s), max_k=%d, poison=%d", device_path(), p->dev.n_cores,
			p->max_k, p->poison);
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
	out->buffer_alloc_scratch	  = rkt_buffer_alloc_scratch;
	out->buffer_free			  = rkt_buffer_free;
	out->buffer_read_f32		  = rkt_buffer_read_f32;
	out->buffer_write_f32		  = rkt_buffer_write_f32;
	out->matmul					  = rkt_matmul;
	out->matmul_batch			  = rkt_matmul_batch;
	out->matmul_multi			  = rkt_matmul_multi;
	out->matmul_multi_batch		  = rkt_matmul_multi_batch;
	out->matmul_residual		  = rkt_matmul_residual;
	out->ffn_activate			  = rkt_ffn_activate;
	out->ffn_activate_ex		  = rkt_ffn_activate_ex;
	out->ffn_activate_batch		  = rkt_ffn_activate_batch;
	// BUG: Temporarily disabled: out->ffn_activate_fused_batch = rkt_ffn_activate_fused_batch;
	out->matmul_ffn_down		  = rkt_matmul_ffn_down;
	out->matmul_ffn_down_batch	  = rkt_matmul_ffn_down_batch;
	out->attention				  = rkt_attention;
	out->attention_batch		  = rkt_attention_batch_fn;
	out->attention_swa			  = rkt_attention_swa;
	out->attention_swa_batch	  = rkt_attention_swa_batch;
	out->kv_alloc				  = rkt_kv_alloc;
	out->kv_free				  = rkt_kv_free;
	out->kv_put					  = rkt_kv_put;
	out->kv_put_batch			  = rkt_kv_put_batch;
	return OK;
}

BACKEND_REGISTER("rocket", rkt_ctor)
