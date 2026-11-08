#include "backend/backend.h"
#include "common.h"
#include "device.h"
#include "emit.h"
#include "gguf.h"
#include "hw.h"
#include "log.h"
#include "threadpool.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <string.h>
#include <unistd.h>

#define RKT_FFNA_SILU 0
#define RKT_FFNA_GELU 1

#define RKT_DEVICE_PATH "/dev/accel/accel0"
#define RKT_WEIGHT_MAGIC 0x524B5457u
#define RKT_TIMEOUT_NS 10000000000LL
#define RKT_POISON_WORD 0xCAFEBABEu
#define RKT_PRIORITY 50
#define RKT_TILE_ROWS 16
#define RKT_TILE_COLS 32

#define RKT_FOLD_K_LEN RKT_K_ALIGN

#define RKT_MAX_JOB_WEIGHT_BYTES ((size_t)12582912)
#define RKT_LUT_CHUNK 256
#define RKT_LUT_BLOCK_ROWS RKT_TILE_ROWS
#define RKT_LUT_BLOCKS_PER_CORE 32
#define RKT_LUT_UPLOAD_TASKS (RKT_LUT_ENTRIES / RKT_LUT_UPLOAD_CHUNK)
#define RKT_LUT_TASKS_PER_CORE (RKT_LUT_UPLOAD_TASKS + RKT_LUT_BLOCKS_PER_CORE)
#define RKT_EW_OPERAND_BLOCK_BYTES                                                                 \
	((size_t)RKT_LUT_BLOCK_ROWS * (RKT_LUT_CHUNK / RKT_EW_SURFACE_CHANNELS) *                      \
	 RKT_EW_SURFACE_SLOTS * sizeof(uint16_t))
#define RKT_LUT_IN_BLOCK_BYTES ((size_t)RKT_LUT_BLOCK_ROWS * RKT_LUT_CHUNK * sizeof(uint16_t))
#define RKT_LUT_OUT_BLOCK_BYTES ((size_t)RKT_LUT_BLOCK_ROWS * RKT_LUT_CHUNK * sizeof(float))
#define RKT_LUT_KINDS 4
#define RKT_LUT_TIERS 2
#define RKT_LUT_SMALL_BLOCKS_PER_CORE 2

#define RKT_QUEUE_ARENA_CAP ((size_t)4 << 20)
#define RKT_QUEUE_BANKS 2
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
typedef struct rkt_quant  rkt_quant;
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
	buffer	   *host_copy;
	void	   *host_source;
	rkt_quant  *quant;
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

typedef struct {
	int			 blocks_per_core;
	rkt_lut_kind uploaded_kind;
	rkt_bo		 in[RKT_MAX_CORES];
	rkt_bo		 ew[RKT_MAX_CORES];
	rkt_bo		 out[RKT_MAX_CORES];
	rkt_bo		 reg_upload[RKT_MAX_CORES];
	rkt_bo		 reg_blocks[RKT_MAX_CORES];
} rkt_lut_tier;

struct rkt_lut_state {
	int				ready;
	int				post_active;
	int				scale_verified;
	int				add_verified;
	int				gate_act_verified[RKT_LUT_KINDS];
	rkt_lut_kind	resident;
	float			in_scale;
	float			out_scale;
	const uint16_t *ew16;
	const float	   *ew_src;
	rkt_post_cfg	post;
	int				verified[RKT_LUT_KINDS];
	uint32_t		upload_ops[RKT_LUT_UPLOAD_TASKS];
	rkt_lut_cfg		cfg;
	rkt_bo			identity[RKT_MAX_CORES];
	rkt_bo			identity_unit[RKT_MAX_CORES];
	rkt_lut_tier	tier[RKT_LUT_TIERS];
};
static void lut_configure(rkt_lut_cfg *cfg, rkt_lut_kind kind);

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
	int				  epi;
	float			  epi_cap;
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
	uint32_t epi;
	uint32_t epi_cap_bits;
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

typedef struct {
	int			   valid;
	int			   bank;
	int			   n_active;
	int			   active[RKT_MAX_CORES];
	rkt_core_queue queue[RKT_MAX_CORES];
} rkt_pending;

typedef struct rkt_lut_state rkt_lut_state;

typedef struct {
	rkt_bo	  in;
	rkt_bo	  out;
	rkt_bo	  reg;
	uint32_t *n_ops;
	size_t	  n_ops_cap;
} rkt_quant_scratch;

enum {
	PROF_MATMUL,
	PROF_QUANT_WAIT,
	PROF_QUANT_HOST,
	PROF_QUANT_PRE,
	PROF_QUANT_SUBMIT,
	PROF_QUANT_REDUCE,
	PROF_ATTN,
	PROF_LUT,
	PROF_PREFILL_MATMUL,
	PROF_N
};

typedef struct {
	rkt_device		  dev;
	pthread_mutex_t	  lock;
	int				  max_k;
	int				  poison;
	int				  npu_failed;
	int				  npu_consecutive_failures;
	rkt_slot		  slots[RKT_QUEUE_BANKS][SLOT_CLASSES][RKT_MAX_CORES];
	int				  bank;
	rkt_pending		  pending[RKT_QUEUE_BANKS];
	uint16_t		 *x16;
	size_t			  x16_cap;
	int				  epi_kind;
	float			  epi_cap;
	uint16_t		 *y16;
	size_t			  y16_cap;
	uint8_t			 *quant_pack[2];
	size_t			  quant_pack_cap[2];
	float			 *quant_s_act;
	float			 *scale_tmp;
	size_t			  scale_tmp_cap;
	size_t			  quant_s_act_cap;
	uint16_t		 *x_chunk;
	size_t			  x_chunk_cap;
	rkt_weight		**weights;
	size_t			  n_weights;
	size_t			  cap_weights;
	int				  fuse;
	int				  quant_on;
	int				  quant_group;
	int				  quant_failed;
	int				  quant_prefill_failed;
	int				  quant_check;
	int				  quant_prefill;
	int				  quant_accepted[2];
	rkt_quant_scratch quant_pf[RKT_MAX_CORES];
	int				  quant_logged;
	int				  quant_checked;
	int				  prof;
	double			  prof_ms[PROF_N];
	long			  prof_n[PROF_N];
	int				  fused_groups;
	int				  fused_weights;
	float			 *y_tmp;
	size_t			  y_tmp_cap;
	float			 *attn_ws;
	size_t			  attn_ws_cap;
	rkt_lut_state	  lut;
	float			 *ffn_act_tmp;
	size_t			  ffn_act_tmp_cap;
	rkt_core_queue	  queue[RKT_MAX_CORES];
	rkt_stream_cache  streams[RKT_MAX_CORES];
	size_t			  in_len;
	slot_class		  queue_cls;
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
	int m = (CBUF_BANKS - 1) * CBUF_BANK_BYTES / (k_len * 2);
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

static void pack_ew_operand(uint16_t *dst, const uint16_t *src, int m, int channels) {
	memset(dst, 0,
		   (size_t)(channels / RKT_EW_SURFACE_CHANNELS) * RKT_EW_SURFACE_SLOTS * m *
			   sizeof(uint16_t));
	for (int r = 0; r < m; r++)
		for (int c = 0; c < channels; c++)
			dst[rkt_ew_operand_index(m, r, c)] = src[(size_t)r * channels + c];
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

static void quant_free(rkt_priv *p, rkt_quant *q);

static void destroy_weight(rkt_priv *p, rkt_weight *w) {
	for (int ci = 0; ci < w->n_kchunks; ci++) {
		for (int c = 0; c < RKT_MAX_CORES; c++) {
			rkt_bo_destroy(&p->dev, &w->kchunks[ci].part[c].bo);
			rkt_bo_destroy(&p->dev, &w->kchunks[ci].fold_bo[c]);
		}
		w->kchunks[ci].fold_ready = 0;
	}
	free(w->kchunks);
	if (w->host_copy) {
		backend *host = backend_host();
		host->buffer_free(host, w->host_copy);
		free(w->host_copy);
	}
	free(w->host_source);
	quant_free(p, w->quant);
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

#define RKT_PAR_MIN_ROWS 16

typedef struct {
	int status;
	status_code (*fn)(void *ctx, int lo, int hi);
	void *ctx;
} par_job;

static void par_chunk(int begin, int end, int tid, void *arg) {
	(void)tid;
	par_job *j = arg;
	if (__atomic_load_n(&j->status, __ATOMIC_RELAXED) != OK)
		return;
	status_code s = j->fn(j->ctx, begin, end);
	if (s != OK)
		__atomic_store_n(&j->status, (int)s, __ATOMIC_RELAXED);
}

static status_code parallel_rows(int n, status_code (*fn)(void *ctx, int lo, int hi), void *ctx) {
	par_job job = {.status = OK, .fn = fn, .ctx = ctx};
	tpool_parallel_for(tpool_fetch_pool(), n, RKT_PAR_MIN_ROWS, par_chunk, &job);
	return (status_code)job.status;
}

typedef status_code (*row_source_fn)(void *ctx, int row, uint16_t *dst, float *tmp);

typedef struct {
	rkt_weight	 *w;
	row_source_fn source;
	void		 *ctx;
} fill_job;

static status_code fill_rows(void *arg, int lo, int hi) {
	fill_job   *job = arg;
	rkt_weight *w	= job->w;
	uint16_t   *row = xcalloc((size_t)w->k_pad, sizeof(uint16_t));
	float	   *tmp = xmalloc((size_t)w->k * sizeof(float));
	status_code s	= OK;
	for (int r = lo; r < hi && s == OK; r++) {
		if (r < w->n)
			s = job->source(job->ctx, r, row, tmp);
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
	free(tmp);
	free(row);
	return s;
}

static status_code fill_weight(rkt_priv *p, rkt_weight *w, row_source_fn source, void *ctx) {
	fill_job	job = {.w = w, .source = source, .ctx = ctx};
	status_code s	= parallel_rows(w->n_pad, fill_rows, &job);
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
} gguf_rows;

static status_code gguf_row_source(void *ctx, int row, uint16_t *dst, float *tmp) {
	gguf_rows *q = ctx;
	return row_to_f16(q->host, q->type, q->data + (size_t)row * q->row_bytes, q->k, tmp, dst);
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
	gguf_rows q = {
		.host	   = backend_host(),
		.type	   = desc->type,
		.data	   = desc->host_data,
		.row_bytes = w->src_row_bytes,
		.k		   = w->k,
	};
	return fill_weight(p, w, gguf_row_source, &q);
}

static status_code keep_on_host(rkt_weight *w, const tensor_desc *desc) {
	backend *host		 = backend_host();
	size_t	 bytes		 = w->src_row_bytes * (size_t)w->n;
	size_t	 padded_rows = ((size_t)w->n + 7) & ~(size_t)7;
	w->host_source		 = xcalloc(1, w->src_row_bytes * padded_rows + 64);
	memcpy(w->host_source, desc->host_data, bytes);
	tensor_desc owned		  = *desc;
	owned.host_data			  = w->host_source;
	owned.release_host_source = 0;
	owned.read_fn			  = NULL;
	owned.read_ctx			  = NULL;
	w->host_copy			  = xcalloc(1, sizeof(*w->host_copy));
	status_code s			  = host->buffer_alloc_weight(host, &owned, w->host_copy);
	if (s == OK)
		return OK;
	free(w->host_copy);
	free(w->host_source);
	w->host_copy   = NULL;
	w->host_source = NULL;
	return s;
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

static int poison_hits(const uint32_t *words, size_t count);

#define RKT_QUANT_NTILE 4096
#define RKT_QUANT_HEAD_ROWS 65536
#define RKT_MAX_FUSE 8
#define RKT_QUANT_MAX_CHILDREN RKT_MAX_FUSE
#define RKT_QUANT_PREFILL_TOKENS 16
#define RKT_QUANT_GROUP 480
#define RKT_QUANT_REDUCE_MIN_BLOCKS 16
#define RKT_QUANT_PACK_MIN_TOKENS 2

typedef struct {
	int		  n0;
	int		  n;
	int		  n_tasks;
	rkt_bo	  in;
	rkt_bo	  out;
	rkt_bo	  wt;
	rkt_bo	  reg;
	size_t	 *w_off;
	size_t	 *out_off;
	uint32_t *n_ops;
} rkt_quant_core;

struct rkt_quant {
	int			   bits;
	int			   group;
	int			   n_ktiles;
	int			   n64;
	rkt_quant_core core[RKT_MAX_CORES];
	float		  *s_w;
	int16_t		  *corr;
};

static int quant_tile_len(const rkt_weight *w, int group, int kt) {
	int rest = w->k_pad - kt * group;
	return rest < group ? rest : group;
}

static void quant_free(rkt_priv *p, rkt_quant *q) {
	if (!q)
		return;
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		rkt_bo_destroy(&p->dev, &q->core[c].in);
		rkt_bo_destroy(&p->dev, &q->core[c].out);
		rkt_bo_destroy(&p->dev, &q->core[c].wt);
		rkt_bo_destroy(&p->dev, &q->core[c].reg);
		free(q->core[c].w_off);
		free(q->core[c].out_off);
		free(q->core[c].n_ops);
	}
	free(q->s_w);
	free(q->corr);
	free(q);
}

static void quant_put_nibble(uint8_t *row16, int j, int v) {
	uint8_t nib = (uint8_t)(v & 15);
	if (j & 1)
		row16[j / 2] |= nib;
	else
		row16[j / 2] = (uint8_t)(nib << 4);
}

static int quant_alloc_cores(rkt_priv *p, rkt_weight *w, rkt_quant *q) {
	int blocks = q->n64 / 64;
	int used   = blocks < p->dev.n_cores ? blocks : p->dev.n_cores;
	int base   = blocks / used;
	int rem	   = blocks % used;
	int n0	   = 0;
	for (int c = 0; c < used; c++) {
		rkt_quant_core *wc = &q->core[c];
		wc->n0			   = n0;
		wc->n			   = (base + (c < rem ? 1 : 0)) * 64;
		n0 += wc->n;
		int n_ntiles = (wc->n + RKT_QUANT_NTILE - 1) / RKT_QUANT_NTILE;
		wc->n_tasks	 = n_ntiles * q->n_ktiles;
		wc->w_off	 = xcalloc((size_t)wc->n_tasks, sizeof(size_t));
		wc->out_off	 = xcalloc((size_t)wc->n_tasks, sizeof(size_t));
		wc->n_ops	 = xcalloc((size_t)wc->n_tasks, sizeof(uint32_t));
		size_t w_acc = 0, o_acc = 0;
		for (int nt = 0; nt < n_ntiles; nt++) {
			int ncols = wc->n - nt * RKT_QUANT_NTILE < RKT_QUANT_NTILE
							? wc->n - nt * RKT_QUANT_NTILE
							: RKT_QUANT_NTILE;
			for (int kt = 0; kt < q->n_ktiles; kt++) {
				int t		   = nt * q->n_ktiles + kt;
				wc->w_off[t]   = w_acc;
				wc->out_off[t] = o_acc;
				w_acc += (size_t)quant_tile_len(w, q->group, kt) * ncols * q->bits / 8;
				o_acc += (size_t)4 * ncols;
			}
		}
		if (rkt_bo_create(&p->dev, c, (size_t)w->k_pad, &wc->in) ||
			rkt_bo_create(&p->dev, c, o_acc, &wc->out) ||
			rkt_bo_create(&p->dev, c, w_acc, &wc->wt) ||
			rkt_bo_create(&p->dev, c, (size_t)wc->n_tasks * RKT_REGCMD_BYTES, &wc->reg))
			return 0;
		memset(wc->wt.map, 0, wc->wt.size);
	}
	return 1;
}

static int quant_bits_for_type(uint32_t type) {
	switch (type) {
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q3_K:
	case GGML_TYPE_Q2_K:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ4_XS:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ3_XXS:
	case GGML_TYPE_IQ2_S:
	case GGML_TYPE_IQ2_XXS:
		return 4;
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_K:
		return 8;
	default:
		return 0;
	}
}

static void quant_put_weight(const rkt_quant *q, uint8_t *base, int col, int kl, int kb, int j,
							 int v) {
	int		 atom_cols = 256 / q->bits;
	uint8_t *row =
		base + (((size_t)(col / atom_cols) * (kl / 32) + kb / 32) * atom_cols + col % atom_cols) *
				   (4 * q->bits);
	if (q->bits == 4)
		quant_put_nibble(row, j, v);
	else
		row[j] = (uint8_t)(int8_t)v;
}

static int quant_token_rows(const rkt_quant *q, int tokens) {
	return (q->bits == 4 ? 2 : 1) * tokens;
}

typedef struct {
	const rkt_weight  *w;
	rkt_quant		  *q;
	const tensor_desc *desc;
	backend			  *host;
} quant_job;

static status_code quant_build_rows(void *arg, int lo, int hi) {
	quant_job		  *job	= arg;
	const rkt_weight  *w	= job->w;
	rkt_quant		  *q	= job->q;
	const tensor_desc *desc = job->desc;
	backend			  *host = job->host;
	int				   bits = q->bits;
	float			  *tmp	= xmalloc((size_t)w->k * sizeof(float));
	status_code		   s	= OK;
	for (int r = lo; r < hi && s == OK; r++) {
		const uint8_t *src = (const uint8_t *)desc->host_data + (size_t)r * w->src_row_bytes;
		if (desc->type == GGML_TYPE_F32) {
			memcpy(tmp, src, (size_t)w->k * sizeof(float));
		} else if (desc->type == GGML_TYPE_F16) {
			for (int i = 0; i < w->k; i++) {
				uint16_t bits;
				memcpy(&bits, src + (size_t)i * 2, sizeof(bits));
				tmp[i] = f16_bits_to_f32(bits);
			}
		} else if (host && host->dequant_row) {
			s = host->dequant_row(host, desc->type, src, w->k, tmp);
		} else {
			s = ERR_UNSUPPORTED;
		}
		if (s != OK)
			break;

		int c = 0;
		while (c + 1 < RKT_MAX_CORES && q->core[c + 1].n > 0 && r >= q->core[c + 1].n0)
			c++;
		rkt_quant_core *wc	= &q->core[c];
		int				lr	= r - wc->n0;
		int				nt	= lr / RKT_QUANT_NTILE;
		int				col = lr % RKT_QUANT_NTILE;
		for (int kt = 0; kt < q->n_ktiles; kt++) {
			int	  k0 = kt * q->group, kl = quant_tile_len(w, q->group, kt);
			float amax = 0.0f;
			for (int j = 0; j < kl && k0 + j < w->k; j++)
				if (fabsf(tmp[k0 + j]) > amax)
					amax = fabsf(tmp[k0 + j]);
			int		 qmax  = (1 << (bits - 1)) - 1;
			int		 qmin  = bits == 4 ? -qmax - 1 : -qmax;
			float	 scale = amax / (float)qmax;
			float	 inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
			uint8_t *base  = (uint8_t *)wc->wt.map + wc->w_off[nt * q->n_ktiles + kt];
			int		 sum   = 0;
			for (int kb = 0; kb < kl; kb += 32)
				for (int j = 0; j < 32; j++) {
					int	  kk = k0 + kb + j;
					float v	 = kk < w->k ? tmp[kk] : 0.0f;
					long  qv = lrintf(v * inv);
					qv		 = qv > qmax ? qmax : qv < qmin ? qmin : qv;
					sum += (int)qv;
					quant_put_weight(q, base, col, kl, kb, j, (int)qv);
				}
			q->s_w[(size_t)kt * q->n64 + r] = scale;
			if (q->corr)
				q->corr[(size_t)kt * q->n64 + r] = (int16_t)(8 * sum);
		}
	}
	free(tmp);
	return s;
}

static status_code quant_build(rkt_priv *p, rkt_weight *w, const tensor_desc *desc, int bits) {
	if (w->n < 1 || w->k < 1 || p->dev.n_cores < 1)
		return ERR_UNSUPPORTED;
	rkt_quant *q = xcalloc(1, sizeof(*q));
	q->bits		 = bits;
	q->group	 = p->quant_group;
	q->n_ktiles	 = (w->k_pad + q->group - 1) / q->group;
	q->n64		 = (w->n + 63) / 64 * 64;
	q->s_w		 = xcalloc((size_t)q->n_ktiles * q->n64, sizeof(float));
	q->corr		 = bits == 4 ? xcalloc((size_t)q->n_ktiles * q->n64, sizeof(int16_t)) : NULL;
	if (!quant_alloc_cores(p, w, q)) {
		quant_free(p, q);
		return ERR_OUT_OF_MEMORY;
	}

	quant_job	job = {.w = w, .q = q, .desc = desc, .host = backend_host()};
	status_code s	= parallel_rows(w->n, quant_build_rows, &job);

	for (int c = 0; c < RKT_MAX_CORES && s == OK; c++) {
		rkt_quant_core *wc = &q->core[c];
		if (wc->n == 0)
			continue;
		int n_ntiles = (wc->n + RKT_QUANT_NTILE - 1) / RKT_QUANT_NTILE;
		for (int nt = 0; nt < n_ntiles; nt++) {
			int ncols = wc->n - nt * RKT_QUANT_NTILE < RKT_QUANT_NTILE
							? wc->n - nt * RKT_QUANT_NTILE
							: RKT_QUANT_NTILE;
			for (int kt = 0; kt < q->n_ktiles; kt++) {
				int t		 = nt * q->n_ktiles + kt;
				wc->n_ops[t] = (uint32_t)rkt_emit_quant(
					(uint64_t *)((uint8_t *)wc->reg.map + (size_t)t * RKT_REGCMD_BYTES), bits,
					quant_token_rows(q, 1), quant_tile_len(w, q->group, kt), ncols,
					(uint32_t)(wc->in.dma_addr + (size_t)kt * q->group),
					(uint32_t)(wc->wt.dma_addr + wc->w_off[t]),
					(uint32_t)(wc->out.dma_addr + wc->out_off[t]));
				if (wc->n_ops[t] > RKT_REGCMD_MAX_OPS)
					s = ERR_INTERNAL;
			}
		}
		if (s == OK && (rkt_bo_flush(&p->dev, &wc->wt) || rkt_bo_flush(&p->dev, &wc->reg)))
			s = ERR_IO;
	}
	if (s != OK) {
		quant_free(p, q);
		return s;
	}
	w->quant = q;
	p->quant_accepted[bits == 8]++;
	return OK;
}

static status_code quant_build_best(rkt_priv *p, rkt_weight *w, const tensor_desc *desc) {
	int bits = quant_bits_for_type(desc->type);
	if (bits == 0)
		return ERR_UNSUPPORTED;
	return quant_build(p, w, desc, bits);
}

static double now_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void prof_add(rkt_priv *p, int kind, double ms) {
	p->prof_ms[kind] += ms;
	p->prof_n[kind]++;
}

static void prof_print(const rkt_priv *p) {
	static const char *names[PROF_N] = {"decode matmul (all paths)",
										"quant npu wait",
										"quant host total",
										"quant host pre",
										"quant submit",
										"quant host reduce",
										"decode attention",
										"decode ffn activation",
										"prefill matmul (all paths)"};
	for (int i = 0; i < PROF_N; i++)
		log_tag("ROCKET-PROF", "%-26s calls=%-7ld total=%9.1f ms  avg=%7.3f ms", names[i],
				p->prof_n[i], p->prof_ms[i], p->prof_n[i] ? p->prof_ms[i] / p->prof_n[i] : 0.0);
}

typedef struct {
	const rkt_bo   *in;
	const rkt_bo   *out;
	const rkt_bo   *reg;
	const uint32_t *n_ops;
} quant_io;

typedef struct {
	const quant_io *io;
	int				task;
} quant_job_task;

static int ensure_bo(rkt_priv *p, int core, size_t bytes, rkt_bo *bo);

static int quant_prefill_prepare(rkt_priv *p, const rkt_weight *w, int tokens, quant_io *io) {
	int m = quant_token_rows(w->quant, tokens);
	for (int c = 0; c < p->dev.n_cores; c++) {
		const rkt_quant_core *wc = &w->quant->core[c];
		if (wc->n == 0)
			continue;
		rkt_quant_scratch *sc = &p->quant_pf[c];
		if (!ensure_bo(p, c, (size_t)w->k_pad * tokens, &sc->in) ||
			!ensure_bo(p, c, wc->out.size * tokens, &sc->out) ||
			!ensure_bo(p, c, (size_t)wc->n_tasks * RKT_REGCMD_BYTES, &sc->reg) ||
			!ensure_capacity((void **)&sc->n_ops, &sc->n_ops_cap, (size_t)wc->n_tasks,
							 sizeof(uint32_t)))
			return 0;
		for (int t = 0; t < wc->n_tasks; t++) {
			int kt		 = t % w->quant->n_ktiles;
			int nt		 = t / w->quant->n_ktiles;
			int ncols	 = wc->n - nt * RKT_QUANT_NTILE < RKT_QUANT_NTILE
							   ? wc->n - nt * RKT_QUANT_NTILE
							   : RKT_QUANT_NTILE;
			sc->n_ops[t] = (uint32_t)rkt_emit_quant(
				(uint64_t *)((uint8_t *)sc->reg.map + (size_t)t * RKT_REGCMD_BYTES), w->quant->bits,
				m, quant_tile_len(w, w->quant->group, kt), ncols,
				(uint32_t)(sc->in.dma_addr + (size_t)kt * w->quant->group * tokens),
				(uint32_t)(wc->wt.dma_addr + wc->w_off[t]),
				(uint32_t)(sc->out.dma_addr + wc->out_off[t] * tokens));
			if (sc->n_ops[t] > RKT_REGCMD_MAX_OPS)
				return 0;
		}
		if (rkt_bo_flush(&p->dev, &sc->reg))
			return 0;
		io[c] = (quant_io){&sc->in, &sc->out, &sc->reg, sc->n_ops};
	}
	return 1;
}

static float quant_absmax(const float *x, int n) {
	float m = 0.0f;
	int	  i = 0;
#if defined(__ARM_NEON)
	float32x4_t v = vdupq_n_f32(0.0f);
	for (; i + 4 <= n; i += 4)
		v = vmaxq_f32(v, vabsq_f32(vld1q_f32(x + i)));
	m = vmaxvq_f32(v);
#endif
	for (; i < n; i++)
		if (fabsf(x[i]) > m)
			m = fabsf(x[i]);
	return m;
}

static void quant_activation_scales(const rkt_weight *w, const float *x, int tokens, float *s_act) {
	int group = w->quant->group, n_ktiles = w->quant->n_ktiles;
	for (int tk = 0; tk < tokens; tk++)
		for (int kt = 0; kt < n_ktiles; kt++) {
			int	  k0 = kt * group, kl = quant_tile_len(w, group, kt);
			int	  valid = kl < w->k - k0 ? kl : w->k - k0;
			float amax	= quant_absmax(x + (size_t)tk * w->k + k0, valid > 0 ? valid : 0);
			s_act[tk * n_ktiles + kt] = amax > 0.0f ? amax / 127.0f : 1.0f;
		}
}

static int quant_quantize_activation(float v, float scale) {
	long qv = lrintf(v / scale);
	return qv > 127 ? 127 : qv < -127 ? -127 : (int)qv;
}

static void quant_quantize_tile(const float *x, int valid, int kl, float scale, int8_t *q) {
	int j = 0;
#if defined(__ARM_NEON)
	float32x4_t sv = vdupq_n_f32(scale);
	for (; j + 8 <= valid; j += 8) {
		int32x4_t a = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + j), sv));
		int32x4_t b = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + j + 4), sv));
		int8x8_t  o = vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)));
		vst1_s8(q + j, vmax_s8(o, vdup_n_s8(-127)));
	}
#endif
	for (; j < valid; j++)
		q[j] = (int8_t)quant_quantize_activation(x[j], scale);
	for (; j < kl; j++)
		q[j] = 0;
}

static void quant_pack_token(const rkt_weight *w, const float *x, int tk, int tokens,
							 const float *s_act, int bits, uint8_t *packed) {
	int	   group = w->quant->group, n_ktiles = w->quant->n_ktiles, m = tokens * (bits == 4 ? 2 : 1);
	int8_t q[RKT_QUANT_GROUP];
	const float *xt = x + (size_t)tk * w->k;
	for (int kt = 0; kt < n_ktiles; kt++) {
		int	  k0 = kt * group, kl = quant_tile_len(w, group, kt);
		int	  valid = kl < w->k - k0 ? kl : w->k - k0;
		float scale = s_act[tk * n_ktiles + kt];
		quant_quantize_tile(xt + k0, valid > 0 ? valid : 0, kl, scale, q);
		if (bits == 8) {
			for (int plane = 0; plane < kl / 16; plane++)
				memcpy(packed + (size_t)k0 * tokens + ((size_t)plane * tokens + tk) * 16,
					   q + plane * 16, 16);
			continue;
		}
		for (int kb = 0; kb < kl; kb += 32)
			for (int row = 0; row < 2; row++) {
				uint8_t *dst =
					packed + (size_t)k0 * tokens + ((size_t)(kb / 32) * m + 2 * tk + row) * 16;
				for (int j = 0; j < 32; j++) {
					int u = q[kb + j] + 128;
					quant_put_nibble(dst, j, row == 0 ? (u >> 4) - 8 : (u & 15) - 8);
				}
			}
	}
}

typedef struct {
	const rkt_weight *w;
	const float		 *x;
	const float		 *s_act;
	uint8_t			 *packed;
	int				  tokens;
	int				  bits;
} quant_pack_job;

static void quant_pack_chunk(int begin, int end, int tid, void *arg) {
	(void)tid;
	const quant_pack_job *j = arg;
	for (int tk = begin; tk < end; tk++)
		quant_pack_token(j->w, j->x, tk, j->tokens, j->s_act, j->bits, j->packed);
}

static void quant_pack_activation(const rkt_weight *w, const float *x, int tokens,
								  const float *s_act, int bits, uint8_t *packed) {
	quant_pack_job job = {w, x, s_act, packed, tokens, bits};
	tpool_parallel_for(tpool_fetch_pool(), tokens, RKT_QUANT_PACK_MIN_TOKENS, quant_pack_chunk,
					   &job);
}

static void quant_reduce_i4(const rkt_quant *q, const uint8_t *tile, size_t tile_stride, int m,
							int tk, int r0, int col, const float *sa, float *dst) {
	const int16_t *o0 = (const int16_t *)tile;
#if defined(__ARM_NEON)
	float32x4_t y0 = vdupq_n_f32(0.0f), y1 = vdupq_n_f32(0.0f);
	for (int kt = 0; kt < q->n_ktiles; kt++) {
		const int16_t *hi = o0 + (size_t)kt * tile_stride + (size_t)col * m + tk * 16;
		int16x8_t	   h  = vld1q_s16(hi);
		int16x8_t	   l  = vld1q_s16(hi + 8);
		int16x8_t	   cv = vld1q_s16(q->corr + (size_t)kt * q->n64 + r0 + col);
		int32x4_t	   a0 =
			vmlal_n_s16(vaddl_s16(vget_low_s16(l), vget_low_s16(cv)), vget_low_s16(h), 16);
		int32x4_t a1 =
			vmlal_n_s16(vaddl_s16(vget_high_s16(l), vget_high_s16(cv)), vget_high_s16(h), 16);
		const float *sw = q->s_w + (size_t)kt * q->n64 + r0 + col;
		y0				= vfmaq_f32(y0, vmulq_n_f32(vld1q_f32(sw), sa[kt]), vcvtq_f32_s32(a0));
		y1				= vfmaq_f32(y1, vmulq_n_f32(vld1q_f32(sw + 4), sa[kt]), vcvtq_f32_s32(a1));
	}
	vst1q_f32(dst, y0);
	vst1q_f32(dst + 4, y1);
#else
	for (int j = 0; j < 8; j++) {
		float v = 0.0f;
		for (int kt = 0; kt < q->n_ktiles; kt++) {
			const int16_t *hi = o0 + (size_t)kt * tile_stride + (size_t)col * m + tk * 16;
			int32_t		   a = 16 * hi[j] + hi[8 + j] + q->corr[(size_t)kt * q->n64 + r0 + col + j];
			v += q->s_w[(size_t)kt * q->n64 + r0 + col + j] * sa[kt] * (float)a;
		}
		dst[j] = v;
	}
#endif
}

static void quant_reduce_i8(const rkt_quant *q, const uint8_t *tile, size_t tile_stride, int m,
							int tk, int r0, int col, const float *sa, float *dst) {
	const int32_t *o0 = (const int32_t *)tile;
#if defined(__ARM_NEON)
	float32x4_t y0 = vdupq_n_f32(0.0f), y1 = vdupq_n_f32(0.0f);
	for (int kt = 0; kt < q->n_ktiles; kt++) {
		const int32_t *lo = o0 + (size_t)kt * tile_stride + (size_t)col * m + tk * 4;
		const float	  *sw = q->s_w + (size_t)kt * q->n64 + r0 + col;
		y0 = vfmaq_f32(y0, vmulq_n_f32(vld1q_f32(sw), sa[kt]), vcvtq_f32_s32(vld1q_s32(lo)));
		y1 = vfmaq_f32(y1, vmulq_n_f32(vld1q_f32(sw + 4), sa[kt]),
					   vcvtq_f32_s32(vld1q_s32(lo + 4 * m)));
	}
	vst1q_f32(dst, y0);
	vst1q_f32(dst + 4, y1);
#else
	for (int j = 0; j < 8; j++) {
		float v = 0.0f;
		for (int kt = 0; kt < q->n_ktiles; kt++) {
			const int32_t *lo = o0 + (size_t)kt * tile_stride + (size_t)col * m + tk * 4 +
								(size_t)(j / 4) * 4 * m + j % 4;
			v += q->s_w[(size_t)kt * q->n64 + r0 + col + j] * sa[kt] * (float)*lo;
		}
		dst[j] = v;
	}
#endif
}

typedef struct {
	const rkt_quant *q;
	const uint8_t	*tile;
	const float		*s_act;
	float			*y;
	size_t			 tile_stride;
	int				 n;
	int				 m;
	int				 r0;
	int				 col_blocks;
} quant_reduce_job;

static void quant_reduce_chunk(int begin, int end, int tid, void *arg) {
	(void)tid;
	const quant_reduce_job *j = arg;
	float					tail[8];
	for (int blk = begin; blk < end; blk++) {
		int tk	= blk / j->col_blocks;
		int col = (blk % j->col_blocks) * 8;
		int row = j->r0 + col;
		if (row >= j->n)
			continue;
		float		*dst = j->y + (size_t)tk * j->n + row;
		float		*out = row + 8 <= j->n ? dst : tail;
		const float *sa	 = j->s_act + (size_t)tk * j->q->n_ktiles;
		if (j->q->bits == 4)
			quant_reduce_i4(j->q, j->tile, j->tile_stride, j->m, tk, j->r0, col, sa, out);
		else
			quant_reduce_i8(j->q, j->tile, j->tile_stride, j->m, tk, j->r0, col, sa, out);
		if (out == tail)
			memcpy(dst, tail, (size_t)(j->n - row) * sizeof(float));
	}
}

static void quant_accumulate_core(const rkt_weight *w, const quant_io *io, int core, int tokens,
								  const float *s_act, float *y) {
	const rkt_quant		 *q		   = w->quant;
	const rkt_quant_core *wc	   = &q->core[core];
	int					  m		   = quant_token_rows(q, tokens);
	int					  n_ntiles = (wc->n + RKT_QUANT_NTILE - 1) / RKT_QUANT_NTILE;
	tpool				 *pool	   = tpool_fetch_pool();
	for (int nt = 0; nt < n_ntiles; nt++) {
		int ncols = wc->n - nt * RKT_QUANT_NTILE < RKT_QUANT_NTILE ? wc->n - nt * RKT_QUANT_NTILE
																   : RKT_QUANT_NTILE;
		quant_reduce_job job = {
			.q			 = q,
			.tile		 = (const uint8_t *)io->out->map + wc->out_off[nt * q->n_ktiles] * tokens,
			.s_act		 = s_act,
			.y			 = y,
			.tile_stride = (size_t)m * ncols,
			.n			 = w->n,
			.m			 = m,
			.r0			 = wc->n0 + nt * RKT_QUANT_NTILE,
			.col_blocks	 = ncols / 8,
		};
		tpool_parallel_for(pool, tokens * job.col_blocks, RKT_QUANT_REDUCE_MIN_BLOCKS,
						   quant_reduce_chunk, &job);
	}
}

static status_code quant_run_inner(rkt_priv *p, rkt_weight **ws, float **ys, int count,
								   const float *x, int tokens, double *wait_ms) {
	double			  t_start = now_ms();
	const rkt_weight *w0	  = ws[0];
	int				  k_pad = w0->k_pad, group = w0->quant->group;
	if (tokens > 1 && count != 1)
		return ERR_UNSUPPORTED;
	for (int i = 1; i < count; i++)
		if (ws[i]->k_pad != k_pad || ws[i]->quant->group != group)
			return ERR_UNSUPPORTED;

	quant_io io[RKT_QUANT_MAX_CHILDREN][RKT_MAX_CORES] = {0};
	if (tokens > 1) {
		if (!quant_prefill_prepare(p, w0, tokens, io[0]))
			return ERR_OUT_OF_MEMORY;
	} else {
		for (int i = 0; i < count; i++)
			for (int c = 0; c < RKT_MAX_CORES; c++) {
				const rkt_quant_core *wc = &ws[i]->quant->core[c];
				io[i][c]				 = (quant_io){&wc->in, &wc->out, &wc->reg, wc->n_ops};
			}
	}

	if (!ensure_capacity((void **)&p->quant_s_act, &p->quant_s_act_cap,
						 (size_t)tokens * w0->quant->n_ktiles, sizeof(float)))
		return ERR_OUT_OF_MEMORY;
	float	*s_act	   = p->quant_s_act;
	uint8_t *packed[2] = {NULL, NULL};
	quant_activation_scales(w0, x, tokens, s_act);
	for (int i = 0; i < count; i++) {
		int bits = ws[i]->quant->bits;
		int slot = bits == 8;
		if (packed[slot])
			continue;
		if (!ensure_capacity((void **)&p->quant_pack[slot], &p->quant_pack_cap[slot],
							 (size_t)k_pad * tokens, 1))
			return ERR_OUT_OF_MEMORY;
		packed[slot] = p->quant_pack[slot];
		memset(packed[slot], 0, (size_t)k_pad * tokens);
		quant_pack_activation(w0, x, tokens, s_act, bits, packed[slot]);
	}

	status_code s = OK;
	for (int i = 0; i < count && s == OK; i++)
		for (int c = 0; c < RKT_MAX_CORES && s == OK; c++) {
			if (ws[i]->quant->core[c].n == 0)
				continue;
			memcpy(io[i][c].in->map, packed[ws[i]->quant->bits == 8], (size_t)k_pad * tokens);
			if (rkt_bo_flush(&p->dev, io[i][c].in))
				s = ERR_IO;
		}

	if (p->poison)
		for (int i = 0; i < count && s == OK; i++)
			for (int c = 0; c < RKT_MAX_CORES && s == OK; c++) {
				if (ws[i]->quant->core[c].n == 0)
					continue;
				uint32_t *ow = io[i][c].out->map;
				for (size_t j = 0; j < io[i][c].out->size / sizeof(uint32_t); j++)
					ow[j] = RKT_POISON_WORD;
				if (rkt_bo_flush(&p->dev, io[i][c].out))
					s = ERR_IO;
			}
	double t_submit						 = now_ms();
	int	   submitted_core[RKT_MAX_CORES] = {0};
	for (int c = 0; c < p->dev.n_cores && s == OK; c++) {
		int n_list = 0;
		for (int i = 0; i < count; i++)
			n_list += ws[i]->quant->core[c].n > 0 ? ws[i]->quant->core[c].n_tasks : 0;
		if (n_list == 0)
			continue;
		quant_job_task *list = xmalloc((size_t)n_list * sizeof(*list));
		int				at	 = 0;
		for (int i = 0; i < count; i++) {
			const rkt_quant_core *wc = &ws[i]->quant->core[c];
			for (int t = 0; wc->n > 0 && t < wc->n_tasks; t++)
				list[at++] = (quant_job_task){.io = &io[i][c], .task = t};
		}
		for (int start = 0; start < n_list && s == OK; start += RKT_MAX_TASKS) {
			int			  n_tasks = n_list - start < RKT_MAX_TASKS ? n_list - start : RKT_MAX_TASKS;
			uint32_t	  dma[RKT_MAX_TASKS], ops[RKT_MAX_TASKS];
			const rkt_bo *in[RKT_MAX_HANDLES], *out[RKT_MAX_HANDLES];
			int			  n_in = 0, n_out = 0;
			const quant_io *last = NULL;
			for (int j = 0; j < n_tasks; j++) {
				const quant_io *tio = list[start + j].io;
				int				t	= list[start + j].task;
				dma[j]				= (uint32_t)(tio->reg->dma_addr + (size_t)t * RKT_REGCMD_BYTES);
				ops[j]				= tio->n_ops[t];
				if (tio != last) {
					last		 = tio;
					in[n_in++]	 = tio->in;
					in[n_in++]	 = tio->reg;
					out[n_out++] = tio->out;
				}
			}
			if (rkt_submit_bos(&p->dev, c, dma, ops, n_tasks, in, n_in, out, n_out))
				s = ERR_COMPUTE_FAIL;
			submitted_core[c] = 1;
		}
		free(list);
	}
	double reduce_ms = 0.0;
	if (p->prof) {
		prof_add(p, PROF_QUANT_PRE, t_submit - t_start);
		prof_add(p, PROF_QUANT_SUBMIT, now_ms() - t_submit);
	}
	*wait_ms = 0.0;
	for (int i = 0; i < count; i++)
		for (int c = 0; c < p->dev.n_cores; c++) {
			if (!submitted_core[c] || ws[i]->quant->core[c].n == 0)
				continue;
			double t_wait = now_ms();
			if (rkt_bo_wait(&p->dev, io[i][c].out) && s == OK)
				s = ERR_COMPUTE_FAIL;
			*wait_ms += now_ms() - t_wait;
			if (s != OK)
				continue;
			if (p->poison &&
				poison_hits(io[i][c].out->map, io[i][c].out->size / sizeof(uint32_t))) {
				WARN("rocket: quant core %d output k=%d n=%d still holds poison", c, ws[i]->k,
					 ws[i]->n);
				s = ERR_COMPUTE_FAIL;
				continue;
			}
			double t_reduce = now_ms();
			quant_accumulate_core(ws[i], &io[i][c], c, tokens, s_act, ys[i]);
			reduce_ms += now_ms() - t_reduce;
		}
	if (p->prof)
		prof_add(p, PROF_QUANT_REDUCE, reduce_ms);
	return s;
}

static status_code quant_run_many(rkt_priv *p, rkt_weight **ws, float **ys, int count,
								  const float *x, int tokens) {
	double		t0		= now_ms();
	double		wait_ms = 0.0;
	status_code s		= quant_run_inner(p, ws, ys, count, x, tokens, &wait_ms);
	double		total	= now_ms() - t0;
	if (p->prof) {
		prof_add(p, PROF_QUANT_WAIT, wait_ms);
		prof_add(p, PROF_QUANT_HOST, total - wait_ms);
	}
	if (s != OK && p->quant_logged < 24) {
		p->quant_logged++;
		WARN("rocket: quant call k=%d n=%d count=%d status=%d: %.1f ms (wait %.1f ms)", ws[0]->k,
			 ws[0]->n, count, s, total, wait_ms);
	}
	return s;
}

static void quant_disable(rkt_priv *p, status_code s) {
	if (p->quant_failed)
		return;
	p->quant_failed = 1;
	WARN("rocket: quant decode path failed (status %d, errno=%d), back to f16 weights", s, errno);
}

static void quant_report(rkt_priv *p, const rkt_weight *w, const float *x, const float *y4,
						 const float *yref, int prefill) {
	if (p->quant_checked >= 400)
		return;
	p->quant_checked++;
	double num = 0.0, den = 0.0, xx = 0.0, xmax = 0.0;
	for (int i = 0; i < w->n; i++) {
		double d = (double)y4[i] - (double)yref[i];
		num += d * d;
		den += (double)yref[i] * (double)yref[i];
	}
	for (int i = 0; i < w->k; i++) {
		xx += (double)x[i] * (double)x[i];
		if (fabs((double)x[i]) > xmax)
			xmax = fabs((double)x[i]);
	}
	double rms = sqrt(xx / (double)w->k);
	char   tag[16];
	snprintf(tag, sizeof(tag), prefill ? "Q%dCHK-PF" : "Q%dCHK", w->quant->bits);
	log_tag(tag, "k=%d n=%d rel_rmse=%.4f x_max/rms=%.1f", w->k, w->n,
			den > 0.0 ? sqrt(num / den) : 0.0, rms > 0.0 ? xmax / rms : 0.0);
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

	int quant_head =
		p->quant_on && w->n >= RKT_QUANT_HEAD_ROWS && quant_bits_for_type(desc->type) != 0;

	status_code s = ERR_UNSUPPORTED;
	if (!quant_head && weight_shape_supported(p, w)) {
		s = upload_to_npu(p, w, desc);
		if (s != OK)
			release_npu_state(p, w);
	}
	if (s != OK) {
		if (s != ERR_UNSUPPORTED)
			WARN("rocket: weight %dx%d (%s) stays on host: status %d", w->k, w->n,
				 ggml_type_name(desc->type), s);
		status_code host_s = keep_on_host(w, desc);
		if (host_s != OK) {
			destroy_weight(p, w);
			return host_s;
		}
	}
	if ((s == OK || quant_head) && p->quant_on) {
		status_code s4 = quant_build_best(p, w, desc);
		if (s4 != OK && s4 != ERR_UNSUPPORTED)
			WARN("rocket: quant build for %dx%d (%s) failed: status %d, stays f16", w->k, w->n,
				 ggml_type_name(desc->type), s4);
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

static int grow_slot_input(rkt_priv *p, slot_class cls, int core, size_t used, size_t bytes) {
	rkt_bo *in = &p->slots[p->bank][cls][core].in;
	if (in->map && in->size >= bytes)
		return 1;
	size_t size = in->size * 2 > bytes ? in->size * 2 : bytes;
	rkt_bo grown;
	if (rkt_bo_create(&p->dev, core, size, &grown))
		return 0;
	if (in->map)
		memcpy(grown.map, in->map, used < in->size ? used : in->size);
	rkt_bo_destroy(&p->dev, in);
	*in = grown;
	return 1;
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
	p->in_len = 0;
}

static status_code drain_all_pending(rkt_priv *p);

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
		status_code drained = drain_all_pending(p);
		if (drained != OK)
			return drained;
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
			.epi	 = (uint32_t)t->ctx.epi,
		};
		if (t->ctx.epi_cap > 0.0f)
			memcpy(&key.epi_cap_bits, &t->ctx.epi_cap, sizeof(key.epi_cap_bits));
		stream_entry *e = stream_lookup(sc, &key);
		if (!e->used) {
			size_t	  offset = align_queue(sc->bytes_used);
			uint64_t *ops	 = (uint64_t *)((uint8_t *)sc->bo.map + offset);
			uint32_t  n_ops;
			if (key.epi) {
				rkt_lut_cfg cfg;
				lut_configure(&cfg, (rkt_lut_kind)key.epi);
				n_ops = (uint32_t)rkt_emit_matmul_f16_act_lut(
					ops, key.blk, key.k_len, key.rows, key.in_dma, key.w_dma, key.out_dma, &cfg,
					t->ctx.epi_cap > 0.0f ? (float)cfg.S / t->ctx.epi_cap : (float)cfg.S);
			} else {
				n_ops = (uint32_t)rkt_emit_matmul_f16(ops, key.blk, key.k_len, key.rows, key.in_dma,
													  key.w_dma, key.out_dma);
			}
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
	q->cls			  = p->queue_cls;
	rkt_slot *sl	  = &p->slots[p->bank][q->cls][core];
	if (!ensure_bo(p, core, q->out_bytes, &sl->out))
		return ERR_OUT_OF_MEMORY;
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

	float out_scale = 1.0f;
	if (b->epi) {
		rkt_lut_cfg cfg;
		lut_configure(&cfg, (rkt_lut_kind)b->epi);
		out_scale = (b->epi_cap > 0.0f ? b->epi_cap : 1.0f) / (float)cfg.F;
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
			v *= out_scale;
			if (b->accumulate)
				dst[nn - lo] += v;
			else
				dst[nn - lo] = v;
		}
	}
	return OK;
}

#define RKT_NPU_MAX_CONSECUTIVE_FAILURES 3

static void npu_record_failure(rkt_priv *p) {
	if (++p->npu_consecutive_failures < RKT_NPU_MAX_CONSECUTIVE_FAILURES)
		return;
	p->npu_failed = 1;
	WARN("rocket: %d consecutive npu failures, npu disabled", p->npu_consecutive_failures);
}

static status_code drain_bank(rkt_priv *p, int bank) {
	rkt_pending *pd = &p->pending[bank];
	if (!pd->valid)
		return OK;
	pd->valid	  = 0;
	status_code s = OK;
	for (int i = 0; i < pd->n_active; i++) {
		int c	= pd->active[i];
		int err = rkt_bo_wait(&p->dev, &p->slots[bank][pd->queue[c].cls][c].out);
		if (err && s == OK) {
			ERROR("rocket: wait on core %d failed: %s", c, strerror(-err));
			s = ERR_COMPUTE_FAIL;
		}
	}
	for (int i = 0; i < pd->n_active && s == OK; i++) {
		const rkt_core_queue *q	 = &pd->queue[pd->active[i]];
		const rkt_slot		 *sl = &p->slots[bank][q->cls][pd->active[i]];
		for (int t = 0; t < q->n_tasks && s == OK; t++)
			s = collect_task(p, &q->tasks[t],
							 (const float *)((const uint8_t *)sl->out.map + q->tasks[t].out_off));
	}
	return s;
}

static status_code drain_all_pending(rkt_priv *p) {
	status_code s = OK;
	for (int bank = 0; bank < RKT_QUEUE_BANKS; bank++) {
		status_code bs = drain_bank(p, (p->bank + 1 + bank) % RKT_QUEUE_BANKS);
		if (s == OK)
			s = bs;
	}
	return s;
}

static void abort_queue(rkt_priv *p) {
	drain_all_pending(p);
	reset_queue(p);
}

static status_code submit_queue(rkt_priv *p, int overlap) {
	int active[RKT_MAX_CORES] = {0};
	int n_active			  = 0;
	for (int c = 0; c < RKT_MAX_CORES; c++)
		if (p->queue[c].n_tasks)
			active[n_active++] = c;
	if (n_active == 0) {
		status_code s = drain_all_pending(p);
		reset_queue(p);
		return s;
	}

	status_code s = OK;
	for (int i = 0; i < n_active && s == OK; i++)
		s = stage_core_queue(p, active[i]);

	int			 bank	   = p->bank;
	rkt_pending *pd		   = &p->pending[bank];
	int			 submitted = 0;
	for (; submitted < n_active && s == OK; submitted++) {
		int					  c	 = active[submitted];
		const rkt_core_queue *q	 = &p->queue[c];
		const rkt_slot		 *sl = &p->slots[bank][q->cls][c];
		uint32_t			  n_ops[RKT_MAX_TASKS];
		uint32_t			  regcmd_dma[RKT_MAX_TASKS];
		for (int t = 0; t < q->n_tasks; t++) {
			n_ops[t]	  = q->tasks[t].n_ops;
			regcmd_dma[t] = q->tasks[t].regcmd_dma;
		}
		int err = rkt_submit(&p->dev, regcmd_dma, n_ops, q->n_tasks, &p->streams[c].bo, &sl->in,
							 &sl->out);
		if (err) {
			ERROR("rocket: submit on core %d failed (%d tasks, cls %d): %s", c, q->n_tasks, q->cls,
				  strerror(-err));
			s = ERR_COMPUTE_FAIL;
		}
	}

	pd->valid	 = submitted > 0;
	pd->bank	 = bank;
	pd->n_active = submitted;
	for (int i = 0; i < submitted; i++) {
		int					  c	  = active[i];
		const rkt_core_queue *src = &p->queue[c];
		rkt_core_queue		 *dst = &pd->queue[c];
		pd->active[i]			  = c;
		dst->cls				  = src->cls;
		dst->n_tasks			  = src->n_tasks;
		dst->out_bytes			  = src->out_bytes;
		memcpy(dst->tasks, src->tasks, (size_t)src->n_tasks * sizeof(src->tasks[0]));
	}
	reset_queue(p);

	status_code older = drain_bank(p, (bank + 1) % RKT_QUEUE_BANKS);
	if (s == OK)
		s = older;
	if (s != OK || !overlap) {
		status_code mine = drain_bank(p, bank);
		if (s == OK)
			s = mine;
	} else {
		p->bank = (bank + 1) % RKT_QUEUE_BANKS;
	}
	return s;
}

static status_code flush_queue(rkt_priv *p) {
	status_code s = submit_queue(p, 0);
	if (s == OK)
		p->npu_consecutive_failures = 0;
	return s;
}

static int job_max_rows(int k_len) {
	int rows = (int)(RKT_MAX_JOB_WEIGHT_BYTES / ((size_t)k_len * sizeof(uint16_t)));
	rows	 = rows & ~(RKT_N_ALIGN - 1);
	if (rows < RKT_N_ALIGN)
		return RKT_N_ALIGN;
	return rows < RKT_MAX_N_CHUNK ? rows : RKT_MAX_N_CHUNK;
}

static int plan_jobs(const block_ctx *b, int lo, int hi, int round, job_desc *jobs) {
	int n_jobs	 = 0;
	int max_rows = job_max_rows(b->kc->k_len);
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		const rkt_part *pt = &b->kc->part[c];
		if (!pt->bo.map)
			continue;
		int a	  = lo > pt->row_start ? lo : pt->row_start;
		int e	  = hi < pt->row_start + pt->row_count ? hi : pt->row_start + pt->row_count;
		int start = a + round * max_rows;
		if (start >= e)
			continue;
		int rows	   = e - start < max_rows ? e - start : max_rows;
		jobs[n_jobs++] = (job_desc){.core = c, .row_start = start, .rows = rows};
	}
	return n_jobs;
}

static status_code queue_block(rkt_priv *p, const block_ctx *b, int blk, const uint16_t *x_block) {
	int lo = b->row0 & ~(RKT_N_ALIGN - 1);
	int hi = (b->row0 + b->n + RKT_N_ALIGN - 1) & ~(RKT_N_ALIGN - 1);
	if (hi > b->w->n_pad)
		hi = b->w->n_pad;

	size_t	   in_bytes				  = (size_t)blk * b->kc->k_len * sizeof(uint16_t);
	slot_class cls					  = blk > 1 ? SLOT_BATCH : SLOT_DECODE;
	size_t	   in_off				  = 0;
	int		   staged				  = 0;
	uint8_t	   written[RKT_MAX_CORES] = {0};
	job_desc   jobs[RKT_MAX_CORES];
	for (int round = 0;; round++) {
		int n_jobs = plan_jobs(b, lo, hi, round, jobs);
		if (n_jobs == 0)
			return OK;
		for (int i = 0; i < n_jobs; i++) {
			rkt_core_queue *q = &p->queue[jobs[i].core];
			if (q->n_tasks == RKT_MAX_TASKS || (p->in_len && p->queue_cls != cls) ||
				(!staged && p->in_len && p->in_len + in_bytes > RKT_QUEUE_ARENA_CAP)) {
				status_code s = submit_queue(p, 1);
				if (s != OK)
					return s;
				staged = 0;
				memset(written, 0, sizeof(written));
			}
			if (!staged) {
				in_off		 = align_queue(p->in_len);
				p->in_len	 = in_off + in_bytes;
				p->queue_cls = cls;
				staged		 = 1;
			}
			if (!written[jobs[i].core]) {
				if (!grow_slot_input(p, cls, jobs[i].core, in_off, p->in_len))
					return ERR_OUT_OF_MEMORY;
				tile_input(
					(uint16_t *)((uint8_t *)p->slots[p->bank][cls][jobs[i].core].in.map + in_off),
					x_block, blk, b->kc->k_len);
				written[jobs[i].core] = 1;
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

		block_ctx b		   = {.w		  = w,
							  .kc		  = kc,
							  .row0		  = row0,
							  .n		  = n,
							  .m		  = m,
							  .y		  = y,
							  .accumulate = ci > 0,
							  .epi		  = p->epi_kind,
							  .epi_cap	  = p->epi_cap};
		int		  max_rows = max_batch_rows(kc->k_len);
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

static status_code dense_matmul_fold(rkt_priv *p, const rkt_weight *w, int row0, int n,
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
	abort_queue(p);
	return s;
}

static status_code dense_matmul_enqueue(rkt_priv *p, const rkt_weight *w, int row0, int n,
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

static status_code dense_matmul(rkt_priv *p, const rkt_weight *w, int row0, int n, const float *x,
								float *y, int m) {
	status_code s = dense_matmul_enqueue(p, w, row0, n, x, y, m);
	if (s == OK)
		return flush_queue(p);
	abort_queue(p);
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
	buffer	 wv	  = *w->host_copy;
	wv.offset += off;
	buffer xv = buffer_host_view(host, rkt_ptr(x), (size_t)m * w->k * sizeof(float));
	buffer yv = buffer_host_view(host, rkt_ptr(y), (size_t)m * n * sizeof(float));
	return host->matmul_batch(host, &wv, w->src_type, &xv, &yv, n, w->k, m);
}

static status_code rkt_matmul_batch(backend *self, const buffer *wbuf, uint32_t w_type,
									const buffer *x, buffer *y, int n, int k, int m);

static status_code rkt_matmul_batch_inner(backend *self, const buffer *wbuf, uint32_t w_type,
										  const buffer *x, buffer *y, int n, int k, int m) {
	(void)w_type;
	rkt_priv   *p = self->priv;
	rkt_weight *w = wbuf->handle;
	if (!w || w->magic != RKT_WEIGHT_MAGIC) {
		ERROR("rocket: matmul weight is not a rocket weight (owner=%s)",
			  wbuf->owner ? wbuf->owner->name : "none");
		return ERR_INVALID_ARG;
	}
	if (w->k != k || wbuf->offset % w->src_row_bytes != 0) {
		ERROR("rocket: matmul shape mismatch (k=%d weight k=%d offset=%zu row_bytes=%zu)", k, w->k,
			  (size_t)wbuf->offset, (size_t)w->src_row_bytes);
		return ERR_INVALID_ARG;
	}
	int row0 = (int)(wbuf->offset / w->src_row_bytes);
	if (row0 + n > w->n) {
		ERROR("rocket: matmul rows out of range (row0=%d n=%d weight n=%d)", row0, n, w->n);
		return ERR_INVALID_ARG;
	}
	if (m == 1 && w->quant && p->quant_on && !p->quant_failed && !p->npu_failed && row0 == 0 &&
		n == w->n) {
		rkt_weight *one = w;
		float	   *yo	= p->quant_check ? xmalloc((size_t)n * sizeof(float)) : rkt_ptr(y);
		pthread_mutex_lock(&p->lock);
		status_code s4 = quant_run_many(p, &one, &yo, 1, rkt_ptr(x), 1);
		pthread_mutex_unlock(&p->lock);
		if (s4 == OK && !p->quant_check)
			return OK;
		if (s4 == OK) {
			p->quant_on	   = 0;
			status_code sr = rkt_matmul_batch(self, wbuf, w_type, x, y, n, k, m);
			p->quant_on	   = 1;
			quant_report(p, w, rkt_ptr(x), yo, rkt_ptr(y), 0);
			free(yo);
			return sr;
		}
		if (p->quant_check)
			free(yo);
		quant_disable(p, s4);
	}
	if (m > 1 && w->quant && p->quant_on && p->quant_prefill && !p->quant_failed &&
		!p->quant_prefill_failed && !p->npu_failed && row0 == 0 && n == w->n) {
		rkt_weight *one = w;
		status_code s4	= OK;
		float	   *yw	= p->quant_check ? xmalloc((size_t)m * n * sizeof(float)) : rkt_ptr(y);
		pthread_mutex_lock(&p->lock);
		for (int t0 = 0; t0 < m && s4 == OK; t0 += RKT_QUANT_PREFILL_TOKENS) {
			int	   tokens = m - t0 < RKT_QUANT_PREFILL_TOKENS ? m - t0 : RKT_QUANT_PREFILL_TOKENS;
			float *yo	  = yw + (size_t)t0 * n;
			s4 =
				quant_run_many(p, &one, &yo, 1, (const float *)rkt_ptr(x) + (size_t)t0 * k, tokens);
		}
		pthread_mutex_unlock(&p->lock);
		if (s4 == OK && !p->quant_check)
			return OK;
		if (s4 == OK) {
			p->quant_on		= 0;
			status_code sr	= rkt_matmul_batch(self, wbuf, w_type, x, y, n, k, m);
			p->quant_on		= 1;
			const float *xf = rkt_ptr(x), *yr = rkt_ptr(y);
			quant_report(p, w, xf, yw, yr, 1);
			quant_report(p, w, xf + (size_t)(m - 1) * k, yw + (size_t)(m - 1) * n,
						 yr + (size_t)(m - 1) * n, 1);
			free(yw);
			return sr;
		}
		if (p->quant_check)
			free(yw);
		p->quant_prefill_failed = 1;
		WARN("rocket: quant prefill path failed (status %d, errno=%d), prefill back to f16 weights",
			 s4, errno);
	}
	if (w->fused_parent) {
		row0 += w->fused_row0;
		w = w->fused_parent;
	}

	const float *xf = rkt_ptr(x);
	float		*yf = rkt_ptr(y);
	if (w->host_copy)
		return host_matmul(w, row0, n, x, y, m);

	pthread_mutex_lock(&p->lock);
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL : dense_matmul(p, w, row0, n, xf, yf, m);
	if (s != OK && !p->npu_failed) {
		npu_record_failure(p);
		WARN("rocket: npu matmul failed (n=%d k=%d m=%d status=%d, errno=%d), using "
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

static status_code rkt_matmul_batch(backend *self, const buffer *wbuf, uint32_t w_type,
									const buffer *x, buffer *y, int n, int k, int m) {
	rkt_priv   *p  = self->priv;
	double		t0 = p->prof ? now_ms() : 0.0;
	status_code s  = rkt_matmul_batch_inner(self, wbuf, w_type, x, y, n, k, m);
	if (p->prof)
		prof_add(p, m == 1 ? PROF_MATMUL : PROF_PREFILL_MATMUL, now_ms() - t0);
	return s;
}

static status_code rkt_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							  buffer *y, int n, int k) {
	return rkt_matmul_batch(self, w, w_type, x, y, n, k, 1);
}

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
	return f;
}

static status_code run_npu_or_cpu(rkt_priv *p, const rkt_weight *w, int row0, int n,
								  const float *xf, float *yf, int m) {
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL : dense_matmul(p, w, row0, n, xf, yf, m);
	if (s != OK && !p->npu_failed) {
		npu_record_failure(p);
		WARN("rocket: npu matmul failed (n=%d k=%d m=%d status=%d, errno=%d), using "
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
	if (m == 1 && p->quant_on && p->quant_check) {
		for (int i = 0; i < n_matmuls; i++) {
			status_code s = rkt_matmul_batch(self, w[i], w_types[i], x, y[i], n_list[i], k, m);
			if (s != OK)
				return s;
		}
		return OK;
	}
	if (m == 1 && p->quant_on && !p->quant_failed && !p->npu_failed &&
		n_matmuls <= RKT_QUANT_MAX_CHILDREN) {
		rkt_weight *ws[RKT_QUANT_MAX_CHILDREN];
		float	   *ys[RKT_QUANT_MAX_CHILDREN];
		uint8_t		is_quant[RKT_QUANT_MAX_CHILDREN] = {0};
		int			count							 = 0;
		for (int i = 0; i < n_matmuls; i++) {
			rkt_weight *wi = w[i]->handle;
			is_quant[i] = wi && wi->magic == RKT_WEIGHT_MAGIC && wi->quant && w[i]->offset == 0 &&
						  n_list[i] == wi->n && wi->k == k;
			if (is_quant[i]) {
				ws[count] = wi;
				ys[count] = rkt_ptr(y[i]);
				count++;
			}
		}
		if (count > 0) {
			pthread_mutex_lock(&p->lock);
			status_code s4 = quant_run_many(p, ws, ys, count, rkt_ptr(x), 1);
			pthread_mutex_unlock(&p->lock);
			if (s4 == OK) {
				for (int i = 0; i < n_matmuls; i++) {
					if (is_quant[i])
						continue;
					status_code s =
						rkt_matmul_batch(self, w[i], w_types[i], x, y[i], n_list[i], k, m);
					if (s != OK)
						return s;
				}
				return OK;
			}
			quant_disable(p, s4);
		}
	}
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
	int use_quant =
		p->quant_on && !p->quant_failed && !p->npu_failed && wt->quant && row0 == 0 && n == wt->n;
	if (wt->fused_parent) {
		row0 += wt->fused_row0;
		wt = wt->fused_parent;
	}
	if (use_quant || wt->host_copy || wt->n_kchunks != 1 ||
		wt->kchunks[0].k_len + RKT_FOLD_K_LEN > p->max_k || wt->k_pad != wt->k ||
		rkt_ptr(residual) == NULL) {
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
					s		  = dense_matmul_fold(p, wt, row0, n, p->x16, yf, 1);
				} else {
					s = ERR_OUT_OF_MEMORY;
				}
			}
		}
	}
	if (s != OK && !p->npu_failed) {
		npu_record_failure(p);
		WARN("rocket: npu residual-fold matmul failed (n=%d k=%d status=%d, errno=%d), "
			 "using cpu",
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
#define RKT_ADD_MAX_SCALED 65000.0f
#define RKT_ADD_VERIFY_REL_TOL 4.0e-3
#define RKT_ADD_VERIFY_ABS_TOL 1.0e-3
#define RKT_LUT_HOST_LIMIT 15.5f
#define RKT_LUT_VERIFY_N 512
#define RKT_LUT_VERIFY_RANGE 12.0f
#define RKT_LUT_VERIFY_ABS_TOL 2.0e-3
#define RKT_LUT_VERIFY_REL_TOL 2.0e-3

static const char *lut_name(rkt_lut_kind kind) {
	if (kind == RKT_LUT_TANH)
		return "tanh";
	return kind == RKT_LUT_GELU_TANH ? "gelu" : "silu";
}

static rkt_lut_kind lut_kind_for_activation(int activation) {
	if (activation == RKT_FFNA_SILU)
		return RKT_LUT_SILU;
	if (activation == RKT_FFNA_GELU)
		return RKT_LUT_GELU_TANH;
	return RKT_LUT_NONE;
}

static double lut_eval(rkt_lut_kind kind, double x) {
	if (kind == RKT_LUT_TANH)
		return tanh(x);
	if (x < -40.0)
		return 0.0;
	if (kind == RKT_LUT_GELU_TANH)
		return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
	return x / (1.0 + exp(-x));
}

static void lut_configure(rkt_lut_cfg *cfg, rkt_lut_kind kind) {
	cfg->kind		= kind;
	cfg->sel		= RKT_LUT_HQ_SEL;
	cfg->S			= 256;
	cfg->F			= 2048;
	cfg->start		= -(RKT_LUT_WINDOW_HALF << cfg->sel);
	cfg->end		= (RKT_LUT_WINDOW_HALF << cfg->sel) - 1;
	cfg->tail_slope = kind == RKT_LUT_TANH ? 0 : cfg->F / cfg->S;
}

static void lut_fill(const rkt_lut_cfg *cfg, uint16_t *le) {
	long step = 1L << cfg->sel;
	for (int i = 0; i < RKT_LUT_ENTRIES; i++) {
		long   y_int = (long)cfg->start + (long)i * step;
		double v	 = lut_eval(cfg->kind, (double)y_int / cfg->S) * cfg->F;
		long   r	 = lround(v);
		if (r > 32767)
			r = 32767;
		if (r < -32768)
			r = -32768;
		le[i] = (uint16_t)((int32_t)r & 0xFFFF);
	}
}

static void lut_tier_destroy(rkt_priv *p, rkt_lut_tier *t) {
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		rkt_bo_destroy(&p->dev, &t->in[c]);
		rkt_bo_destroy(&p->dev, &t->ew[c]);
		rkt_bo_destroy(&p->dev, &t->out[c]);
		rkt_bo_destroy(&p->dev, &t->reg_upload[c]);
		rkt_bo_destroy(&p->dev, &t->reg_blocks[c]);
	}
	t->uploaded_kind = RKT_LUT_NONE;
}

static status_code lut_tier_create(rkt_priv *p, rkt_lut_tier *t, int blocks_per_core) {
	t->blocks_per_core = blocks_per_core;
	t->uploaded_kind   = RKT_LUT_NONE;
	for (int c = 0; c < p->dev.n_cores; c++)
		if (rkt_bo_create(&p->dev, c, (size_t)blocks_per_core * RKT_LUT_IN_BLOCK_BYTES,
						  &t->in[c]) ||
			rkt_bo_create(&p->dev, c, (size_t)blocks_per_core * RKT_EW_OPERAND_BLOCK_BYTES,
						  &t->ew[c]) ||
			rkt_bo_create(&p->dev, c, (size_t)blocks_per_core * RKT_LUT_OUT_BLOCK_BYTES,
						  &t->out[c]) ||
			rkt_bo_create(&p->dev, c, (size_t)RKT_LUT_UPLOAD_TASKS * RKT_REGCMD_BYTES,
						  &t->reg_upload[c]) ||
			rkt_bo_create(&p->dev, c, (size_t)blocks_per_core * RKT_REGCMD_BYTES,
						  &t->reg_blocks[c]))
			return ERR_OUT_OF_MEMORY;
	return OK;
}

static void lut_state_reset(rkt_priv *p, rkt_lut_state *a) {
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		rkt_bo_destroy(&p->dev, &a->identity[c]);
		rkt_bo_destroy(&p->dev, &a->identity_unit[c]);
	}
	for (int t = 0; t < RKT_LUT_TIERS; t++)
		lut_tier_destroy(p, &a->tier[t]);
	a->ready	= 0;
	a->resident = RKT_LUT_NONE;
}

static status_code lut_state_init(rkt_priv *p, rkt_lut_state *a) {
	if (a->ready)
		return OK;
	lut_configure(&a->cfg, RKT_LUT_NONE);
	size_t	 identity_bytes = (size_t)RKT_LUT_CHUNK * RKT_LUT_CHUNK * sizeof(uint16_t);
	uint16_t row[RKT_LUT_CHUNK];
	for (int c = 0; c < p->dev.n_cores; c++) {
		if (rkt_bo_create(&p->dev, c, identity_bytes, &a->identity[c])) {
			lut_state_reset(p, a);
			return ERR_OUT_OF_MEMORY;
		}
		for (int j = 0; j < RKT_LUT_CHUNK; j++) {
			memset(row, 0, sizeof(row));
			row[j] = f32_to_f16_bits((float)a->cfg.S);
			tile_weight_row(a->identity[c].map, row, RKT_LUT_CHUNK, j);
		}
		if (rkt_bo_flush(&p->dev, &a->identity[c])) {
			lut_state_reset(p, a);
			return ERR_IO;
		}
		if (rkt_bo_create(&p->dev, c, identity_bytes, &a->identity_unit[c])) {
			lut_state_reset(p, a);
			return ERR_OUT_OF_MEMORY;
		}
		for (int j = 0; j < RKT_LUT_CHUNK; j++) {
			memset(row, 0, sizeof(row));
			row[j] = f32_to_f16_bits(1.0f);
			tile_weight_row(a->identity_unit[c].map, row, RKT_LUT_CHUNK, j);
		}
		if (rkt_bo_flush(&p->dev, &a->identity_unit[c])) {
			lut_state_reset(p, a);
			return ERR_IO;
		}
	}
	if (lut_tier_create(p, &a->tier[0], RKT_LUT_SMALL_BLOCKS_PER_CORE) != OK ||
		lut_tier_create(p, &a->tier[1], RKT_LUT_BLOCKS_PER_CORE) != OK) {
		lut_state_reset(p, a);
		return ERR_OUT_OF_MEMORY;
	}
	a->ready = 1;
	return OK;
}

static status_code lut_tier_prepare(rkt_priv *p, rkt_lut_state *a, rkt_lut_tier *t) {
	if (t->uploaded_kind == a->cfg.kind)
		return OK;
	uint16_t le[RKT_LUT_ENTRIES];
	lut_fill(&a->cfg, le);
	for (int c = 0; c < p->dev.n_cores; c++) {
		for (int task = 0; task < RKT_LUT_UPLOAD_TASKS; task++) {
			uint32_t n_ops = (uint32_t)rkt_emit_lut_upload(
				(uint64_t *)((uint8_t *)t->reg_upload[c].map + (size_t)task * RKT_REGCMD_BYTES), 0,
				task * RKT_LUT_UPLOAD_CHUNK, le + task * RKT_LUT_UPLOAD_CHUNK, RKT_LUT_UPLOAD_CHUNK,
				(uint32_t)t->in[c].dma_addr, (uint32_t)a->identity[c].dma_addr,
				(uint32_t)t->out[c].dma_addr);
			if (!n_ops || n_ops > RKT_REGCMD_MAX_OPS)
				return ERR_INTERNAL;
			a->upload_ops[task] = n_ops;
		}
		if (rkt_bo_flush(&p->dev, &t->reg_upload[c]))
			return ERR_IO;
	}
	t->uploaded_kind = a->cfg.kind;
	return OK;
}

static status_code lut_state_select(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind) {
	status_code s = lut_state_init(p, a);
	if (s != OK)
		return s;
	lut_configure(&a->cfg, kind);
	a->post_active = 0;
	return OK;
}

typedef struct {
	int first;
	int rows;
} lut_block;

static status_code lut_run_wave(rkt_priv *p, rkt_lut_state *a, rkt_lut_tier *t, const uint16_t *x16,
								const lut_block *blocks, int n_blocks) {
	int			n_upload = a->post_active ? 0 : RKT_LUT_UPLOAD_TASKS;
	status_code s		 = n_upload ? lut_tier_prepare(p, a, t) : OK;
	if (s != OK)
		return s;

	int		 n_cores				 = p->dev.n_cores;
	int		 per_core[RKT_MAX_CORES] = {0};
	uint32_t n_ops[RKT_MAX_CORES][RKT_LUT_TASKS_PER_CORE];
	uint32_t regcmd_dma[RKT_MAX_CORES][RKT_LUT_TASKS_PER_CORE];

	for (int c = 0; c < n_cores; c++)
		for (int task = 0; task < n_upload; task++) {
			n_ops[c][task] = a->upload_ops[task];
			regcmd_dma[c][task] =
				(uint32_t)(t->reg_upload[c].dma_addr + (size_t)task * RKT_REGCMD_BYTES);
		}

	for (int i = 0; i < n_blocks; i++) {
		const lut_block *b	  = &blocks[i];
		int				 c	  = i % n_cores;
		int				 slot = i / n_cores;
		int				 task = n_upload + slot;
		tile_input((uint16_t *)((uint8_t *)t->in[c].map + (size_t)slot * RKT_LUT_IN_BLOCK_BYTES),
				   x16 + (size_t)b->first * RKT_LUT_CHUNK, b->rows, RKT_LUT_CHUNK);
		uint64_t *reg =
			(uint64_t *)((uint8_t *)t->reg_blocks[c].map + (size_t)slot * RKT_REGCMD_BYTES);
		uint32_t in_dma	 = (uint32_t)(t->in[c].dma_addr + (size_t)slot * RKT_LUT_IN_BLOCK_BYTES);
		uint32_t out_dma = (uint32_t)(t->out[c].dma_addr + (size_t)slot * RKT_LUT_OUT_BLOCK_BYTES);
		uint32_t w_dma =
			(uint32_t)(a->ew16 ? a->identity_unit[c].dma_addr : a->identity[c].dma_addr);
		uint32_t ew_dma = 0;
		if (a->ew16) {
			size_t ew_slot = (size_t)slot * RKT_EW_OPERAND_BLOCK_BYTES;
			pack_ew_operand((uint16_t *)((uint8_t *)t->ew[c].map + ew_slot),
							a->ew16 + (size_t)b->first * RKT_LUT_CHUNK, b->rows, RKT_LUT_CHUNK);
			ew_dma = (uint32_t)(t->ew[c].dma_addr + ew_slot);
		}
		n_ops[c][task] =
			a->post_active && a->cfg.kind != RKT_LUT_NONE && !a->ew16
				? (uint32_t)rkt_emit_matmul_f16_act_lut(reg, b->rows, RKT_LUT_CHUNK, RKT_LUT_CHUNK,
														in_dma, w_dma, out_dma, &a->cfg,
														a->post.mulc)
			: a->ew16
				? (uint32_t)rkt_emit_matmul_f16_ewadd(reg, b->rows, RKT_LUT_CHUNK, RKT_LUT_CHUNK,
													  in_dma, w_dma, out_dma, ew_dma)
			: a->post_active
				? (uint32_t)rkt_emit_matmul_f16_act(reg, b->rows, RKT_LUT_CHUNK, RKT_LUT_CHUNK,
													in_dma, w_dma, out_dma, &a->post)
				: (uint32_t)rkt_emit_matmul_f16_lut(reg, b->rows, RKT_LUT_CHUNK, RKT_LUT_CHUNK,
													in_dma, w_dma, out_dma, &a->cfg);
		if (!n_ops[c][task] || n_ops[c][task] > RKT_REGCMD_MAX_OPS)
			return ERR_INTERNAL;
		regcmd_dma[c][task] =
			(uint32_t)(t->reg_blocks[c].dma_addr + (size_t)slot * RKT_REGCMD_BYTES);
		per_core[c]++;
	}

	for (int c = 0; c < n_cores; c++)
		if (per_core[c] &&
			(rkt_bo_flush(&p->dev, &t->in[c]) || rkt_bo_flush(&p->dev, &t->reg_blocks[c]) ||
			 (a->ew16 && rkt_bo_flush(&p->dev, &t->ew[c]))))
			return ERR_IO;

	int submitted = 0;
	for (; submitted < n_cores && (per_core[submitted] || n_upload) && s == OK; submitted++) {
		const rkt_bo *in_list[5] = {
			&t->in[submitted], a->ew16 ? &a->identity_unit[submitted] : &a->identity[submitted],
			&t->reg_upload[submitted], &t->reg_blocks[submitted], &t->ew[submitted]};
		const rkt_bo *out_list[1] = {&t->out[submitted]};
		if (rkt_submit_bos(&p->dev, submitted, regcmd_dma[submitted], n_ops[submitted],
						   n_upload + per_core[submitted], in_list, a->ew16 ? 5 : 4, out_list, 1))
			s = ERR_COMPUTE_FAIL;
	}
	for (int c = 0; c < submitted; c++)
		if (rkt_bo_wait(&p->dev, &t->out[c]) && s == OK)
			s = ERR_COMPUTE_FAIL;
	if (s == OK && n_upload)
		a->resident = a->cfg.kind;
	return s;
}

typedef struct {
	const rkt_lut_tier *tier;
	const lut_block	   *blocks;
	const uint16_t	   *x16;
	const float		   *up;
	float			   *out;
	int					up_stride;
	int					n;
	int					chunks_per_row;
	int					n_cores;
	rkt_lut_kind		kind;
	float				inv;
	float				out_scale;
} lut_gather;

static void lut_gather_chunk(int begin, int end, int tid, void *arg) {
	(void)tid;
	const lut_gather *g = arg;
	for (int i = begin; i < end; i++) {
		const lut_block *b	  = &g->blocks[i];
		int				 c	  = i % g->n_cores;
		int				 slot = i / g->n_cores;
		const float		*om	  = (const float *)((const uint8_t *)g->tier->out[c].map +
												(size_t)slot * RKT_LUT_OUT_BLOCK_BYTES);
		for (int r = 0; r < b->rows; r++) {
			int				pixel = b->first + r;
			int				row	  = pixel / g->chunks_per_row;
			int				col0  = (pixel % g->chunks_per_row) * RKT_LUT_CHUNK;
			int				cols  = g->n - col0 < RKT_LUT_CHUNK ? g->n - col0 : RKT_LUT_CHUNK;
			const uint16_t *x	  = g->x16 + (size_t)pixel * RKT_LUT_CHUNK;
			const float	   *u	  = g->up ? g->up + (size_t)row * g->up_stride + col0 : NULL;
			float		   *dst	  = g->out + (size_t)row * g->n + col0;
			for (int j = 0; j < cols; j++) {
				float v = om[(size_t)(j / RKT_OUT_C2_F32) * b->rows * RKT_OUT_C2_F32 +
							 (size_t)r * RKT_OUT_C2_F32 + j % RKT_OUT_C2_F32] *
						  g->inv;
				if (g->kind != RKT_LUT_NONE) {
					float xv = f16_bits_to_f32(x[j]);
					if (fabsf(xv) >= RKT_LUT_HOST_LIMIT)
						v = (float)lut_eval(g->kind, (double)xv);
				}
				dst[j] = (u ? v * u[j] : v) * g->out_scale;
			}
		}
	}
}

static status_code lut_run_rows(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind, const float *gate,
								int gate_stride, const float *up, int up_stride, float *out, int n,
								int m, float inv) {
	status_code s			   = OK;
	int			chunks_per_row = (n + RKT_LUT_CHUNK - 1) / RKT_LUT_CHUNK;
	int			pixels		   = m * chunks_per_row;
	if (!ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)pixels * RKT_LUT_CHUNK,
						 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	for (int r = 0; r < m; r++) {
		uint16_t	*dst = p->x16 + (size_t)r * chunks_per_row * RKT_LUT_CHUNK;
		const float *src = gate + (size_t)r * gate_stride;
		if (a->in_scale > 0.0f)
			for (int j = 0; j < n; j++)
				dst[j] = f32_to_f16_bits(src[j] * a->in_scale);
		else
			convert_to_f16(dst, src, (size_t)n);
		memset(dst + n, 0, ((size_t)chunks_per_row * RKT_LUT_CHUNK - (size_t)n) * sizeof(uint16_t));
	}

	if (a->ew_src) {
		if (!ensure_capacity((void **)&p->y16, &p->y16_cap, (size_t)pixels * RKT_LUT_CHUNK,
							 sizeof(uint16_t)))
			return ERR_OUT_OF_MEMORY;
		for (int r = 0; r < m; r++) {
			uint16_t *dst = p->y16 + (size_t)r * chunks_per_row * RKT_LUT_CHUNK;
			for (int j = 0; j < n; j++) {
				float v = a->ew_src[(size_t)r * n + j];
				if (!(fabsf(v) <= RKT_ADD_MAX_SCALED))
					return ERR_UNSUPPORTED;
				dst[j] = f32_to_f16_bits(v);
			}
			memset(dst + n, 0,
				   ((size_t)chunks_per_row * RKT_LUT_CHUNK - (size_t)n) * sizeof(uint16_t));
		}
		a->ew16 = p->y16;
	}

	lut_block blocks[RKT_MAX_CORES * RKT_LUT_BLOCKS_PER_CORE];
	int		  wave_capacity = p->dev.n_cores * RKT_LUT_BLOCKS_PER_CORE;
	tpool	 *pool			= tpool_fetch_pool();
	for (int done = 0; done < pixels;) {
		int n_blocks = 0;
		while (n_blocks < wave_capacity && done < pixels) {
			int rows		   = next_block_rows(pixels - done, RKT_LUT_BLOCK_ROWS);
			blocks[n_blocks++] = (lut_block){.first = done, .rows = rows};
			done += rows;
		}
		rkt_lut_tier *tier =
			n_blocks <= p->dev.n_cores * RKT_LUT_SMALL_BLOCKS_PER_CORE ? &a->tier[0] : &a->tier[1];
		s = lut_run_wave(p, a, tier, p->x16, blocks, n_blocks);
		if (s != OK)
			return s;
		lut_gather g = {.tier			= tier,
						.blocks			= blocks,
						.x16			= p->x16,
						.up				= up,
						.out			= out,
						.up_stride		= up_stride,
						.n				= n,
						.chunks_per_row = chunks_per_row,
						.n_cores		= p->dev.n_cores,
						.kind			= kind,
						.inv			= inv,
						.out_scale		= a->out_scale > 0.0f ? a->out_scale : 1.0f};
		tpool_parallel_for(pool, n_blocks, 4, lut_gather_chunk, &g);
	}
	return OK;
}

static status_code lut_activate_rows(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind,
									 const float *gate, int gate_stride, const float *up,
									 int up_stride, float *out, int n, int m) {
	status_code s = lut_state_select(p, a, kind);
	if (s != OK)
		return s;
	return lut_run_rows(p, a, kind, gate, gate_stride, up, up_stride, out, n, m,
						1.0f / (float)a->cfg.F);
}

static status_code lut_scale_rows(rkt_priv *p, rkt_lut_state *a, const float *x, float *out, int n,
								  float scale) {
	if (!isfinite(scale))
		return ERR_INVALID_ARG;
	status_code s = lut_state_init(p, a);
	if (s != OK)
		return s;
	int	  exponent;
	float mantissa = frexpf(scale, &exponent);
	lut_configure(&a->cfg, RKT_LUT_NONE);
	a->post_active = 1;
	a->post		   = (rkt_post_cfg){.act = RKT_ACT_MULC, .mulc = 2.0f * mantissa};
	return lut_run_rows(p, a, RKT_LUT_NONE, x, n, NULL, 0, out, n, 1,
						ldexpf(1.0f, exponent - 1) / (float)a->cfg.S);
}

#define RKT_SCALE_VERIFY_REL_TOL 4.0e-3
#define RKT_SCALE_VERIFY_ABS_TOL 1.0e-4
#define RKT_SCALE_VERIFY_VALUE 0.0371f

static status_code lut_scale_verify(rkt_priv *p, rkt_lut_state *a) {
	float ramp[RKT_LUT_VERIFY_N], got[RKT_LUT_VERIFY_N];
	for (int i = 0; i < RKT_LUT_VERIFY_N; i++)
		ramp[i] = -RKT_LUT_VERIFY_RANGE +
				  2.0f * RKT_LUT_VERIFY_RANGE * (float)i / (float)(RKT_LUT_VERIFY_N - 1);
	status_code s = lut_scale_rows(p, a, ramp, got, RKT_LUT_VERIFY_N, RKT_SCALE_VERIFY_VALUE);
	if (s != OK)
		return s;
	double worst   = 0.0;
	float  worst_x = 0.0f;
	int	   bad	   = 0;
	for (int i = 0; i < RKT_LUT_VERIFY_N; i++) {
		float  x   = f16_bits_to_f32(f32_to_f16_bits(ramp[i]));
		double ref = (double)x * (double)RKT_SCALE_VERIFY_VALUE;
		double err = fabs((double)got[i] - ref);
		if (err > worst) {
			worst	= err;
			worst_x = x;
		}
		if (!(err <= RKT_SCALE_VERIFY_ABS_TOL + RKT_SCALE_VERIFY_REL_TOL * fabs(ref)))
			bad++;
	}
	if (bad) {
		WARN("rocket: npu scale verification failed: %d/%d points off, max err %.6f at x=%.4f, "
			 "scale stays on host",
			 bad, RKT_LUT_VERIFY_N, worst, (double)worst_x);
		return ERR_COMPUTE_FAIL;
	}
	return OK;
}

static status_code lut_verify(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind) {
	float ramp[RKT_LUT_VERIFY_N], ones[RKT_LUT_VERIFY_N], got[RKT_LUT_VERIFY_N];
	for (int i = 0; i < RKT_LUT_VERIFY_N; i++) {
		ramp[i] = -RKT_LUT_VERIFY_RANGE +
				  2.0f * RKT_LUT_VERIFY_RANGE * (float)i / (float)(RKT_LUT_VERIFY_N - 1);
		ones[i] = 1.0f;
	}
	status_code s = lut_activate_rows(p, a, kind, ramp, RKT_LUT_VERIFY_N, ones, RKT_LUT_VERIFY_N,
									  got, RKT_LUT_VERIFY_N, 1);
	if (s != OK)
		return s;

	double worst   = 0.0;
	float  worst_x = 0.0f;
	int	   bad	   = 0;
	for (int i = 0; i < RKT_LUT_VERIFY_N; i++) {
		float  x   = f16_bits_to_f32(f32_to_f16_bits(ramp[i]));
		double ref = lut_eval(kind, (double)x);
		double err = fabs((double)got[i] - ref);
		if (err > worst) {
			worst	= err;
			worst_x = x;
		}
		if (!(err <= RKT_LUT_VERIFY_ABS_TOL + RKT_LUT_VERIFY_REL_TOL * fabs(ref)))
			bad++;
	}
	if (bad) {
		WARN("rocket: lut %s verification failed: %d/%d points off, max err %.5f at x=%.4f, "
			 "activation stays on host",
			 lut_name(kind), bad, RKT_LUT_VERIFY_N, worst, (double)worst_x);
		return ERR_COMPUTE_FAIL;
	}
	return OK;
}

static status_code lut_ensure_verified(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind) {
	if (a->verified[kind] > 0)
		return OK;
	if (a->verified[kind] < 0)
		return ERR_UNSUPPORTED;
	status_code s	  = lut_verify(p, a, kind);
	a->verified[kind] = s == OK ? 1 : -1;
	return s;
}

static status_code ffn_activate_host(backend *self, const buffer *gate, const buffer *up,
									 buffer *out, int n, int activation, int m, int gate_stride,
									 int up_stride, int fused) {
	if (fused || gate_stride != n || up_stride != n)
		return ERR_UNSUPPORTED;
	backend_report_host_fallback(self, "ffn_activate", HFB_OP_NOT_NATIVE, "shape n=%d m=%d act=%d",
								 n, m, activation);
	backend *host = backend_host();
	if (!host || !host->ffn_activate_batch)
		return ERR_UNSUPPORTED;
	return host->ffn_activate_batch(host, gate, up, out, n, activation, m);
}

static status_code lut_softcap_rows(rkt_priv *p, rkt_lut_state *a, const float *x, float *out,
									int n, float cap) {
	status_code s = lut_state_select(p, a, RKT_LUT_TANH);
	if (s != OK)
		return s;
	a->in_scale	 = 1.0f / cap;
	a->out_scale = cap;
	s = lut_run_rows(p, a, RKT_LUT_TANH, x, n, NULL, 0, out, n, 1, 1.0f / (float)a->cfg.F);
	a->in_scale	 = 0.0f;
	a->out_scale = 0.0f;
	return s;
}

static status_code rkt_softcap(backend *self, buffer *x, float cap, int n) {
	rkt_priv *p = self->priv;
	if (n <= 0)
		return OK;
	status_code s = ERR_UNSUPPORTED;
	if (cap > 0.0f && !p->npu_failed) {
		pthread_mutex_lock(&p->lock);
		s = lut_ensure_verified(p, &p->lut, RKT_LUT_TANH);
		if (s == OK &&
			!ensure_capacity((void **)&p->scale_tmp, &p->scale_tmp_cap, (size_t)n, sizeof(float)))
			s = ERR_OUT_OF_MEMORY;
		if (s == OK) {
			s = lut_softcap_rows(p, &p->lut, rkt_ptr(x), p->scale_tmp, n, cap);
			if (s == OK)
				memcpy(rkt_ptr(x), p->scale_tmp, (size_t)n * sizeof(float));
		}
		if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed &&
			p->lut.verified[RKT_LUT_TANH] > 0) {
			npu_record_failure(p);
			WARN("rocket: npu softcap failed (n=%d status=%d, errno=%d)", n, s, errno);
		}
		pthread_mutex_unlock(&p->lock);
	}
	if (s == OK)
		return OK;
	backend *host = backend_host();
	if (!host || !host->softcap)
		return ERR_UNSUPPORTED;
	backend_report_host_fallback(self, "softcap", HFB_OP_NOT_NATIVE, "n=%d", n);
	return host->softcap(host, x, cap, n);
}

static status_code lut_make_resident(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind) {
	status_code s = lut_state_select(p, a, kind);
	if (s != OK || a->resident == kind)
		return s;
	return lut_run_wave(p, a, &a->tier[0], NULL, NULL, 0);
}

#define RKT_GATE_ACT_VERIFY_RANGE 20.0f
#define RKT_GATE_ACT_VERIFY_REL_TOL 1.0e-2

static status_code lut_gate_act_verify(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind) {
	float ramp[RKT_LUT_VERIFY_N], got[RKT_LUT_VERIFY_N];
	for (int i = 0; i < RKT_LUT_VERIFY_N; i++)
		ramp[i] = -RKT_GATE_ACT_VERIFY_RANGE +
				  2.0f * RKT_GATE_ACT_VERIFY_RANGE * (float)i / (float)(RKT_LUT_VERIFY_N - 1);
	status_code s = lut_make_resident(p, a, kind);
	if (s != OK)
		return s;
	a->post_active = 1;
	a->post		   = (rkt_post_cfg){.act = RKT_ACT_MULC, .mulc = 1.0f};
	s = lut_run_rows(p, a, RKT_LUT_NONE, ramp, RKT_LUT_VERIFY_N, NULL, 0, got, RKT_LUT_VERIFY_N, 1,
					 1.0f / (float)a->cfg.F);
	if (s != OK)
		return s;
	double worst   = 0.0;
	float  worst_x = 0.0f;
	int	   bad	   = 0;
	for (int i = 0; i < RKT_LUT_VERIFY_N; i++) {
		float  x   = f16_bits_to_f32(f32_to_f16_bits(ramp[i]));
		double ref = lut_eval(kind, (double)x);
		double err = fabs((double)got[i] - ref);
		if (err > worst) {
			worst	= err;
			worst_x = x;
		}
		if (!(err <= RKT_LUT_VERIFY_ABS_TOL + RKT_GATE_ACT_VERIFY_REL_TOL * fabs(ref)))
			bad++;
	}
	if (bad) {
		WARN("rocket: gate+%s epilogue verification failed: %d/%d points off, max err %.5f at "
			 "x=%.4f, gate activation stays separate",
			 lut_name(kind), bad, RKT_LUT_VERIFY_N, worst, (double)worst_x);
		return ERR_COMPUTE_FAIL;
	}
	return OK;
}

static status_code lut_gate_act_prepare(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind) {
	if (a->gate_act_verified[kind] < 0)
		return ERR_UNSUPPORTED;
	if (a->gate_act_verified[kind] == 0) {
		status_code v			   = lut_gate_act_verify(p, a, kind);
		a->gate_act_verified[kind] = v == OK ? 1 : -1;
		if (v != OK)
			return v;
	}
	return lut_make_resident(p, a, kind);
}

static int gate_act_chainable(const rkt_priv *p, const buffer *wbuf, int n, int k,
							  const rkt_weight **dw, int *row0) {
	const rkt_weight *w = wbuf->handle;
	if (p->npu_failed || !w || w->magic != RKT_WEIGHT_MAGIC || w->k != k || w->host_copy)
		return 0;
	if (wbuf->offset % w->src_row_bytes != 0 || (int)(wbuf->offset / w->src_row_bytes) + n > w->n)
		return 0;
	if (w->quant && p->quant_on && !p->quant_failed)
		return 0;
	*row0 = (int)(wbuf->offset / w->src_row_bytes);
	if (w->fused_parent) {
		*row0 += w->fused_row0;
		w = w->fused_parent;
	}
	*dw = w;
	return 1;
}

static status_code rkt_matmul_gate_act_batch(backend *self, const buffer *w_gate,
											 uint32_t gate_type, const buffer *w_up,
											 uint32_t up_type, const buffer *x, buffer *act_out,
											 buffer *up_out, int n, int k, int activation, int m) {
	(void)gate_type;
	(void)up_type;
	rkt_priv		 *p		= self->priv;
	rkt_lut_kind	  kind	= lut_kind_for_activation(activation);
	const rkt_weight *gw	= NULL;
	const rkt_weight *uw	= NULL;
	int				  grow0 = 0;
	int				  urow0 = 0;
	if (kind == RKT_LUT_NONE || n <= 0 || m <= 0 ||
		!gate_act_chainable(p, w_gate, n, k, &gw, &grow0) ||
		!gate_act_chainable(p, w_up, n, k, &uw, &urow0) || gw->n_kchunks != 1 || uw->n_kchunks != 1)
		return ERR_UNSUPPORTED;

	float *act = rkt_ptr(act_out);
	float *up  = rkt_ptr(up_out);
	pthread_mutex_lock(&p->lock);
	status_code s = lut_gate_act_prepare(p, &p->lut, kind);
	if (s == OK) {
		p->epi_kind = (int)kind;
		s			= dense_matmul_enqueue(p, gw, grow0, n, rkt_ptr(x), act, m);
		p->epi_kind = 0;
		if (s == OK)
			s = dense_matmul_enqueue(p, uw, urow0, n, rkt_ptr(x), up, m);
		if (s == OK)
			s = flush_queue(p);
		else
			abort_queue(p);
	}
	if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed && p->lut.gate_act_verified[kind] > 0) {
		npu_record_failure(p);
		WARN("rocket: npu gate+%s matmul failed (n=%d k=%d m=%d status=%d, errno=%d)",
			 lut_name(kind), n, k, m, s, errno);
	}
	pthread_mutex_unlock(&p->lock);
	if (s != OK)
		return ERR_UNSUPPORTED;
	size_t total = (size_t)n * (size_t)m;
	for (size_t i = 0; i < total; i++)
		act[i] *= up[i];
	return OK;
}

static status_code rkt_matmul_softcap_fallback(backend *self, const buffer *w, uint32_t w_type,
											   const buffer *x, buffer *y, int n, int k, int m,
											   float cap) {
	status_code s = rkt_matmul_batch(self, w, w_type, x, y, n, k, m);
	if (s != OK)
		return s;
	return rkt_softcap(self, y, cap, n * m);
}

static status_code rkt_matmul_softcap_batch(backend *self, const buffer *wbuf, uint32_t w_type,
											const buffer *x, buffer *y, int n, int k, int m,
											float cap) {
	rkt_priv		 *p	   = self->priv;
	const rkt_weight *dw   = NULL;
	int				  row0 = 0;
	if (!(cap > 0.0f) || n <= 0 || m <= 0 || !gate_act_chainable(p, wbuf, n, k, &dw, &row0) ||
		dw->n_kchunks != 1)
		return rkt_matmul_softcap_fallback(self, wbuf, w_type, x, y, n, k, m, cap);

	pthread_mutex_lock(&p->lock);
	status_code s = lut_gate_act_prepare(p, &p->lut, RKT_LUT_TANH);
	if (s == OK) {
		p->epi_kind = RKT_LUT_TANH;
		p->epi_cap	= cap;
		s			= dense_matmul(p, dw, row0, n, rkt_ptr(x), rkt_ptr(y), m);
		p->epi_kind = 0;
		p->epi_cap	= 0.0f;
	}
	if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed &&
		p->lut.gate_act_verified[RKT_LUT_TANH] > 0) {
		npu_record_failure(p);
		WARN("rocket: npu matmul+softcap failed (n=%d k=%d m=%d status=%d, errno=%d)", n, k, m, s,
			 errno);
	}
	pthread_mutex_unlock(&p->lock);
	if (s == OK)
		return OK;
	return rkt_matmul_softcap_fallback(self, wbuf, w_type, x, y, n, k, m, cap);
}

static status_code lut_add_rows(rkt_priv *p, rkt_lut_state *a, const float *x, const float *y,
								float *out, int n, int m, float out_scale) {
	status_code s = lut_state_init(p, a);
	if (s != OK)
		return s;
	lut_configure(&a->cfg, RKT_LUT_NONE);
	a->post_active = 1;
	a->post		   = (rkt_post_cfg){.act = RKT_ACT_NONE};
	a->ew_src	   = y;
	s			   = lut_run_rows(p, a, RKT_LUT_NONE, x, n, NULL, 0, out, n, m, out_scale);
	a->ew_src	   = NULL;
	a->ew16		   = NULL;
	return s;
}

static status_code lut_add_verify(rkt_priv *p, rkt_lut_state *a) {
	float x[RKT_LUT_VERIFY_N], y[RKT_LUT_VERIFY_N], got[RKT_LUT_VERIFY_N];
	for (int i = 0; i < RKT_LUT_VERIFY_N; i++) {
		float t = (float)i / (float)(RKT_LUT_VERIFY_N - 1);
		x[i]	= -RKT_LUT_VERIFY_RANGE + 2.0f * RKT_LUT_VERIFY_RANGE * t;
		y[i]	= 0.37f * RKT_LUT_VERIFY_RANGE * (1.0f - 2.0f * t * t);
	}
	status_code s = lut_add_rows(p, a, x, y, got, RKT_LUT_VERIFY_N, 1, 1.0f);
	if (s != OK)
		return s;
	double worst   = 0.0;
	int	   bad	   = 0;
	int	   worst_i = 0;
	for (int i = 0; i < RKT_LUT_VERIFY_N; i++) {
		double ref = (double)f16_bits_to_f32(f32_to_f16_bits(x[i])) +
					 (double)f16_bits_to_f32(f32_to_f16_bits(y[i]));
		double err = fabs((double)got[i] - ref);
		if (err > worst) {
			worst	= err;
			worst_i = i;
		}
		if (!(err <= RKT_ADD_VERIFY_ABS_TOL + RKT_ADD_VERIFY_REL_TOL * fabs(ref)))
			bad++;
	}
	if (bad) {
		WARN("rocket: npu add verification failed: %d/%d points off, max err %.6f at i=%d "
			 "(x=%.4f y=%.4f got=%.4f), add stays on host",
			 bad, RKT_LUT_VERIFY_N, worst, worst_i, (double)x[worst_i], (double)y[worst_i],
			 (double)got[worst_i]);
		return ERR_COMPUTE_FAIL;
	}
	return OK;
}

static status_code lut_add_checked(rkt_priv *p, rkt_lut_state *a, float *x, const float *y, int n,
								   int m, float out_scale) {
	if (a->add_verified < 0)
		return ERR_UNSUPPORTED;
	if (a->add_verified == 0) {
		status_code v	= lut_add_verify(p, a);
		a->add_verified = v == OK ? 1 : -1;
		if (v != OK)
			return v;
	}
	size_t total = (size_t)n * (size_t)m;
	if (!ensure_capacity((void **)&p->scale_tmp, &p->scale_tmp_cap, total, sizeof(float)))
		return ERR_OUT_OF_MEMORY;
	status_code s = lut_add_rows(p, a, x, y, p->scale_tmp, n, m, out_scale);
	if (s == OK)
		memcpy(x, p->scale_tmp, total * sizeof(float));
	return s;
}

static status_code rkt_add_scaled(backend *self, buffer *x, const buffer *y, int n, int m,
								  float out_scale) {
	rkt_priv *p = self->priv;
	if (n <= 0 || m <= 0)
		return OK;
	status_code s = ERR_UNSUPPORTED;
	if (!p->npu_failed) {
		pthread_mutex_lock(&p->lock);
		s = lut_add_checked(p, &p->lut, rkt_ptr(x), rkt_ptr(y), n, m, out_scale);
		if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed && p->lut.add_verified > 0) {
			npu_record_failure(p);
			WARN("rocket: npu add failed (n=%d m=%d status=%d, errno=%d)", n, m, s, errno);
		}
		pthread_mutex_unlock(&p->lock);
	}
	if (s == OK)
		return OK;
	backend *host = backend_host();
	if (!host)
		return ERR_UNSUPPORTED;
	if (out_scale == 1.0f && host->add_batch) {
		backend_report_host_fallback(self, "add", HFB_OP_NOT_NATIVE, "n=%d m=%d", n, m);
		return host->add_batch(host, x, y, n, m);
	}
	if (out_scale != 1.0f && host->ple_combine) {
		backend_report_host_fallback(self, "ple_combine", HFB_OP_NOT_NATIVE, "n=%d", n);
		return host->ple_combine(host, x, y, n * m, out_scale);
	}
	return ERR_UNSUPPORTED;
}

static status_code rkt_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
	return rkt_add_scaled(self, x, y, n, m, 1.0f);
}

static status_code rkt_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
								   float combine_scale) {
	return rkt_add_scaled(self, ple, proj, n, 1, combine_scale);
}

static status_code rkt_add_inplace(backend *self, buffer *x, const buffer *y, int n) {
	return rkt_add_batch(self, x, y, n, 1);
}

static status_code rkt_ffn_activate_kind_inner(backend *self, const buffer *gate, const buffer *up,
											   buffer *out, int n, int activation, int m,
											   int gate_stride, int up_stride, int fused) {
	rkt_priv *p = self->priv;
	if (n <= 0 || m <= 0)
		return OK;
	rkt_lut_kind kind = lut_kind_for_activation(activation);
	if (kind == RKT_LUT_NONE || p->npu_failed)
		return ffn_activate_host(self, gate, up, out, n, activation, m, gate_stride, up_stride,
								 fused);

	const float *gf		  = rkt_ptr(gate);
	const float *uf		  = rkt_ptr(up);
	float		*of		  = rkt_ptr(out);
	float		*ucopy	  = NULL;
	const float *up_end	  = uf + (size_t)(m - 1) * up_stride + n;
	const float *out_end  = of + (size_t)m * n;
	int			 u_stride = up_stride;
	if (of < up_end && uf < out_end) {
		ucopy = xmalloc((size_t)n * (size_t)m * sizeof(float));
		for (int r = 0; r < m; r++)
			memcpy(ucopy + (size_t)r * n, uf + (size_t)r * up_stride, (size_t)n * sizeof(float));
		uf		 = ucopy;
		u_stride = n;
	}

	pthread_mutex_lock(&p->lock);
	status_code s = lut_ensure_verified(p, &p->lut, kind);
	if (s == OK)
		s = lut_activate_rows(p, &p->lut, kind, gf, gate_stride, uf, u_stride, of, n, m);
	if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed && p->lut.verified[kind] > 0) {
		npu_record_failure(p);
		WARN("rocket: npu ffn_activate failed (n=%d m=%d status=%d, errno=%d)", n, m, s, errno);
	}
	pthread_mutex_unlock(&p->lock);
	free(ucopy);
	if (s != OK)
		return ffn_activate_host(self, gate, up, out, n, activation, m, gate_stride, up_stride,
								 fused);
	return OK;
}

static status_code rkt_ffn_activate_kind(backend *self, const buffer *gate, const buffer *up,
										 buffer *out, int n, int activation, int m, int gate_stride,
										 int up_stride, int fused) {
	rkt_priv   *p  = self->priv;
	double		t0 = p->prof && m == 1 ? now_ms() : 0.0;
	status_code s  = rkt_ffn_activate_kind_inner(self, gate, up, out, n, activation, m, gate_stride,
												 up_stride, fused);
	if (p->prof && m == 1)
		prof_add(p, PROF_LUT, now_ms() - t0);
	return s;
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
	(void)self;
	backend *host = backend_host();
	if (!host || !host->ffn_activate_fused_batch)
		return ERR_UNSUPPORTED;
	return host->ffn_activate_fused_batch(host, fused, out, n, activation, m);
}

static status_code rkt_matmul_ffn_down_batch(backend *self, const buffer *w, uint32_t w_type,
											 const buffer *gate, const buffer *up, buffer *y, int n,
											 int k, int activation, int m) {
	rkt_priv *p = self->priv;
	if (m <= 0)
		return ERR_INVALID_ARG;
	if (!ensure_capacity((void **)&p->ffn_act_tmp, &p->ffn_act_tmp_cap, (size_t)m * k,
						 sizeof(float)))
		return ERR_OUT_OF_MEMORY;

	buffer		act_buf = buffer_host_view(self, p->ffn_act_tmp, (size_t)m * k * sizeof(float));
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

static status_code lut_scale_checked(rkt_priv *p, rkt_lut_state *a, float *x, int n, float scale) {
	if (a->scale_verified < 0)
		return ERR_UNSUPPORTED;
	if (a->scale_verified == 0) {
		status_code v	  = lut_scale_verify(p, a);
		a->scale_verified = v == OK ? 1 : -1;
		if (v != OK)
			return v;
	}
	if (!ensure_capacity((void **)&p->scale_tmp, &p->scale_tmp_cap, (size_t)n, sizeof(float)))
		return ERR_OUT_OF_MEMORY;
	status_code s = lut_scale_rows(p, a, x, p->scale_tmp, n, scale);
	if (s == OK)
		memcpy(x, p->scale_tmp, (size_t)n * sizeof(float));
	return s;
}

static status_code rkt_scale_inplace(backend *self, buffer *x, float scale, int n) {
	rkt_priv *p = self->priv;
	if (n <= 0 || scale == 1.0f)
		return OK;
	status_code s = ERR_UNSUPPORTED;
	if (!p->npu_failed) {
		pthread_mutex_lock(&p->lock);
		s = lut_scale_checked(p, &p->lut, rkt_ptr(x), n, scale);
		if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed && p->lut.scale_verified > 0) {
			npu_record_failure(p);
			WARN("rocket: npu scale failed (n=%d status=%d, errno=%d)", n, s, errno);
		}
		pthread_mutex_unlock(&p->lock);
	}
	if (s == OK)
		return OK;
	backend *host = backend_host();
	if (!host || !host->scale_inplace)
		return ERR_UNSUPPORTED;
	backend_report_host_fallback(self, "scale_inplace", HFB_OP_NOT_NATIVE, "n=%d scale=%g", n,
								 (double)scale);
	return host->scale_inplace(host, x, scale, n);
}

static status_code rkt_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out) {
	rkt_priv *p = self->priv;
	if (desc->kv_quant != KV_QUANT_F16 || desc->n_layers < 1 || desc->n_ctx < 1)
		return ERR_UNSUPPORTED;
	if (rkt_align_up(desc->n_ctx, RKT_K_ALIGN) > RKT_MAX_K)
		return ERR_UNSUPPORTED;
	int		n_slots = desc->n_kv_layers > 0 ? desc->n_kv_layers : desc->n_layers;
	rkt_kv *kv		= xcalloc(1, sizeof(*kv));
	kv->magic		= RKT_KV_MAGIC;
	kv->n_slots		= n_slots;
	kv->slots		= xcalloc((size_t)n_slots, sizeof(*kv->slots));
	int			pos = rkt_align_up(desc->n_ctx, RKT_K_ALIGN);
	status_code s	= OK;
	for (int i = 0; i < n_slots && s == OK; i++) {
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
		for (int i = 0; i < n_slots; i++) {
			if (kv->slots[i].k)
				destroy_weight(p, kv->slots[i].k);
			if (kv->slots[i].v)
				destroy_weight(p, kv->slots[i].v);
		}
		free(kv->slots);
		free(kv);
		return s;
	}
	int max_hd = desc->head_dim;
	if (desc->layer_head_dim)
		for (int i = 0; i < n_slots; i++)
			if (desc->layer_head_dim[i] > max_hd)
				max_hd = desc->layer_head_dim[i];
	size_t bytes = (size_t)n_slots * desc->n_kv_heads * pos * rkt_align_up(max_hd, RKT_K_ALIGN) *
				   sizeof(uint16_t) * 2;
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
			int				v_col = pos / RKT_TILE_COLS, v_rem = pos % RKT_TILE_COLS;
			const rkt_part *vp = NULL;
			for (int d = 0; d < head_dim; d++) {
				int v_global = h * sl->hd_pad + d;
				if (!vp || v_global < vp->row_start || v_global >= vp->row_start + vp->row_count)
					vp = part_for_row(vc, v_global);
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

#if defined(__ARM_NEON)
static inline float32x4_t neon_expf(float32x4_t x) {
	x			   = vmaxq_f32(x, vdupq_n_f32(-87.3f));
	int32x4_t	n  = vcvtnq_s32_f32(vmulq_f32(x, vdupq_n_f32(1.44269504f)));
	float32x4_t nf = vcvtq_f32_s32(n);
	float32x4_t r  = vfmsq_f32(x, nf, vdupq_n_f32(0.693145751953125f));
	r			   = vfmsq_f32(r, nf, vdupq_n_f32(1.428606765330187e-06f));
	float32x4_t p  = vdupq_n_f32(1.0f / 720.0f);
	p			   = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
	p			   = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
	p			   = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
	p			   = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
	p			   = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
	p			   = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
	int32x4_t e	   = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
	return vmulq_f32(p, vreinterpretq_f32_s32(e));
}
#endif

static float softmax_scaled_max(const float *sc, int lo, int hi, float scale) {
	float mx = -INFINITY;
	int	  t	 = lo;
#if defined(__ARM_NEON)
	float32x4_t vm = vdupq_n_f32(-INFINITY);
	for (; t + 4 <= hi; t += 4)
		vm = vmaxq_f32(vm, vmulq_n_f32(vld1q_f32(sc + t), scale));
	mx = vmaxvq_f32(vm);
#endif
	for (; t < hi; t++) {
		float v = sc[t] * scale;
		if (v > mx)
			mx = v;
	}
	return mx;
}

static double softmax_exp_sum(const float *sc, float *pr, int lo, int hi, float scale, float mx) {
	double sum = 0.0;
	int	   t   = lo;
#if defined(__ARM_NEON)
	float32x4_t vmx = vdupq_n_f32(mx);
	while (t + 4 <= hi) {
		float32x4_t acc = vdupq_n_f32(0.0f);
		int			end = t + 4 * 256 < hi ? t + 4 * 256 : hi;
		for (; t + 4 <= end; t += 4) {
			float32x4_t e = neon_expf(vsubq_f32(vmulq_n_f32(vld1q_f32(sc + t), scale), vmx));
			vst1q_f32(pr + t, e);
			acc = vaddq_f32(acc, e);
		}
		sum += (double)vaddvq_f32(acc);
	}
#endif
	for (; t < hi; t++) {
		float e = expf(sc[t] * scale - mx);
		pr[t]	= e;
		sum += e;
	}
	return sum;
}

static void softmax_round_f16(float *pr, int n, float inv) {
	int t = 0;
#if defined(__ARM_NEON)
	for (; t + 4 <= n; t += 4)
		vst1q_f32(pr + t, vcvt_f32_f16(vcvt_f16_f32(vmulq_n_f32(vld1q_f32(pr + t), inv))));
#endif
	for (; t < n; t++)
		pr[t] = f16_bits_to_f32(f32_to_f16_bits(pr[t] * inv));
}

static void attention_softmax(const float *scores, float *probs, int m, int n_i, int span16,
							  int pos_pad, int pos_start, int window, float scale) {
	for (int r = 0; r < m; r++) {
		int lim	 = pos_start + r + 1;
		int t_lo = window > 0 ? pos_start + r + 1 - window : 0;
		if (t_lo < 0)
			t_lo = 0;
		for (int i = 0; i < n_i; i++) {
			const float *sc	 = scores + (size_t)(r * n_i + i) * span16;
			float		*pr	 = probs + (size_t)(r * n_i + i) * pos_pad;
			float		 mx	 = softmax_scaled_max(sc, t_lo, lim, scale);
			double		 sum = softmax_exp_sum(sc, pr, t_lo, lim, scale, mx);
			softmax_round_f16(pr, lim, (float)(1.0 / sum));
		}
	}
}

static status_code rkt_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
									   const buffer *v_cache, buffer *out, int layer, int pos_start,
									   int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									   int flash_attn, float scale, int n_kv_heads_active, int m,
									   int window);

static status_code rkt_attention_batch_inner(backend *self, const buffer *q, const buffer *k_cache,
											 const buffer *v_cache, buffer *out, int layer,
											 int pos_start, int n_heads, int n_kv_heads,
											 int head_dim, int n_ctx, int flash_attn, float scale,
											 int n_kv_heads_active, int m, int window) {
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
			s = dense_matmul_enqueue(p, sl->k, g * sl->pos_pad, span16, q_buf, scores, M);
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
			s = dense_matmul_enqueue(p, sl->v, (g0 + i) * sl->hd_pad, head_dim, probs, out_buf, M);
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
		abort_queue(p);
		if (!p->npu_failed) {
			npu_record_failure(p);
			WARN("rocket: npu attention failed (status %d, errno=%d)", s, errno);
		}
	}
	pthread_mutex_unlock(&p->lock);
	return s;
}

static status_code rkt_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
									   const buffer *v_cache, buffer *out, int layer, int pos_start,
									   int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									   int flash_attn, float scale, int n_kv_heads_active, int m,
									   int window) {
	rkt_priv *p = self->priv;
	if (!kv_slot_of(k_cache, layer) && k_cache->owner && k_cache->owner != self) {
		backend *host = backend_host();
		if (window > 0 && host && host->attention_swa_batch)
			return host->attention_swa_batch(host, q, k_cache, v_cache, out, layer, pos_start,
											 n_heads, n_kv_heads, head_dim, n_ctx, flash_attn,
											 scale, window, n_kv_heads_active, m);
		if (window <= 0 && host && host->attention_batch)
			return host->attention_batch(host, q, k_cache, v_cache, out, layer, pos_start, n_heads,
										 n_kv_heads, head_dim, n_ctx, flash_attn, scale,
										 n_kv_heads_active, m);
		return ERR_UNSUPPORTED;
	}
	double		t0 = p->prof && m == 1 ? now_ms() : 0.0;
	status_code s  = rkt_attention_batch_inner(self, q, k_cache, v_cache, out, layer, pos_start,
											   n_heads, n_kv_heads, head_dim, n_ctx, flash_attn,
											   scale, n_kv_heads_active, m, window);
	if (p->prof && m == 1)
		prof_add(p, PROF_ATTN, now_ms() - t0);
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
	for (int bank = 0; bank < RKT_QUEUE_BANKS; bank++)
		for (int cls = 0; cls < SLOT_CLASSES; cls++)
			for (int c = 0; c < RKT_MAX_CORES; c++) {
				rkt_bo_destroy(&p->dev, &p->slots[bank][cls][c].in);
				rkt_bo_destroy(&p->dev, &p->slots[bank][cls][c].out);
			}
	for (int c = 0; c < RKT_MAX_CORES; c++)
		rkt_bo_destroy(&p->dev, &p->streams[c].bo);
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		rkt_bo_destroy(&p->dev, &p->quant_pf[c].in);
		rkt_bo_destroy(&p->dev, &p->quant_pf[c].out);
		rkt_bo_destroy(&p->dev, &p->quant_pf[c].reg);
		free(p->quant_pf[c].n_ops);
	}
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
	if (p->quant_on)
		log_tag("RKT", "quantized weights: %d int4, %d int8", p->quant_accepted[0],
				p->quant_accepted[1]);
	free_slots(p);
	lut_state_reset(p, &p->lut);
	rkt_device_close(&p->dev);
	pthread_mutex_destroy(&p->lock);
	free(p->weights);
	free(p->x16);
	free(p->x_chunk);
	free(p->y_tmp);
	free(p->quant_pack[0]);
	free(p->quant_pack[1]);
	free(p->quant_s_act);
	free(p->scale_tmp);
	free(p->attn_ws);
	free(p->ffn_act_tmp);
	if (p->prof)
		prof_print(p);
	free(p);
	self->priv = NULL;
}

static status_code rkt_init(backend *self, int device_index) {
	(void)device_index;
	rkt_priv *p = xcalloc(1, sizeof(*p));
	p->max_k =
		env_int("KAPPAI_ROCKET_MAX_K", RKT_MAX_K, RKT_K_ALIGN, RKT_MAX_K) & ~(RKT_K_ALIGN - 1);
	p->poison		 = env_int("KAPPAI_ROCKET_POISON", 0, 0, 1);
	p->fuse			 = 1;
	p->quant_on		 = 1;
	p->quant_group	 = RKT_QUANT_GROUP;
	p->quant_check	 = env_int("KAPPAI_ROCKET_QUANT_CHECK", 0, 0, 1);
	p->quant_prefill = 1;
	p->prof			 = env_int("KAPPAI_ROCKET_PROF", 0, 0, 1);
	pthread_mutex_init(&p->lock, NULL);
	self->priv = p;

	int n_cores = env_int("KAPPAI_ROCKET_NCORES", RKT_MAX_CORES, 1, RKT_MAX_CORES);
	int err		= rkt_device_open(&p->dev, device_path(), n_cores, RKT_TIMEOUT_NS);
	if (err) {
		WARN("rocket: cannot open %s: %s", device_path(), strerror(-err));
		rkt_free(self);
		return ERR_UNSUPPORTED;
	}
	log_tag(
		"RKT",
		"%s: %d core(s), max_k=%d, poison=%d, tasks/job=%d, group=%d prefill=%d prefill_tokens=%d",
		device_path(), p->dev.n_cores, p->max_k, p->poison, RKT_MAX_TASKS, p->quant_group,
		p->quant_prefill, RKT_QUANT_PREFILL_TOKENS);
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
	out->scale_inplace			  = rkt_scale_inplace;
	out->add_inplace			  = rkt_add_inplace;
	out->matmul_softcap_batch	  = rkt_matmul_softcap_batch;
	out->matmul_gate_act_batch	  = rkt_matmul_gate_act_batch;
	out->softcap				  = rkt_softcap;
	out->add_batch				  = rkt_add_batch;
	out->ple_combine			  = rkt_ple_combine;
	out->ffn_activate_ex		  = rkt_ffn_activate_ex;
	out->ffn_activate_batch		  = rkt_ffn_activate_batch;
	out->ffn_activate_fused_batch = rkt_ffn_activate_fused_batch;
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
