#ifndef ROCKET_H
#define ROCKET_H

#include "backend/backend.h"
#include "common.h"
#include "device.h"
#include "emit.h"
#include "gguf.h"
#include "hw.h"
#include "log.h"
#include "threadpool.h"
#include <arm_neon.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define RKT_FFNA_SILU 0
#define RKT_FFNA_GELU 1

#define RKT_DEVICE_PATH "/dev/accel/accel0"
#define RKT_WEIGHT_MAGIC 0x524B5457u
#define RKT_TIMEOUT_NS 10000000000LL
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
#define RKT_COLLECT_PAR_MIN_BYTES ((size_t)65536)
#define RKT_RESIDUAL_MAX 65000.0f

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
	uint32_t	  magic;
	int			  n;
	int			  k;
	int			  n_pad;
	int			  k_pad;
	uint32_t	  src_type;
	size_t		  src_row_bytes;
	int			  n_kchunks;
	rkt_kchunk	 *kchunks;
	buffer		 *host_copy;
	void		 *host_source;
	rkt_quant	 *quant;
	rkt_weight	 *fused_parent;
	int			  fused_row0;
	int			  fuse_tried;
	rkt_float_fmt fmt;
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
	rkt_bo		 reg_mul[RKT_MAX_CORES];
} rkt_lut_tier;

struct rkt_lut_state {
	int				ready;
	int				post_active;
	rkt_lut_kind	resident;
	float			in_scale;
	float			out_scale;
	const uint16_t *ew16;
	const float	   *ew_src;
	uint16_t	   *out16;
	int				out16_stride;
	rkt_post_cfg	post;
	uint32_t		upload_ops[RKT_LUT_UPLOAD_TASKS];
	rkt_lut_cfg		cfg;
	rkt_bo			identity[RKT_MAX_CORES];
	rkt_bo			identity_unit[RKT_MAX_CORES];
	rkt_lut_tier	tier[RKT_LUT_TIERS];
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
	int				  epi;
	float			  epi_cap;
	float			 *y;
	uint16_t		 *y16;
	int				  y16_stride;
	const uint16_t	 *ew16;
} block_ctx;

typedef struct {
	block_ctx ctx;
	int		  blk;
	job_desc  job;
	size_t	  in_off;
	size_t	  out_off;
	size_t	  ew_off;
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
	uint32_t out16;
	uint32_t ew_dma;
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

typedef struct {
	rkt_device		  dev;
	pthread_mutex_t	  lock;
	int				  npu_failed;
	int				  npu_consecutive_failures;
	rkt_slot		  slots[RKT_QUEUE_BANKS][SLOT_CLASSES][RKT_MAX_CORES];
	int				  bank;
	rkt_pending		  pending[RKT_QUEUE_BANKS];
	uint16_t		 *x16;
	const uint16_t	 *x_src;
	size_t			  x16_cap;
	int				  epi_kind;
	float			  epi_cap;
	uint16_t		 *out16_dst;
	int				  out16_stride;
	uint16_t		 *nat_gate;
	size_t			  nat_gate_cap;
	uint16_t		 *nat_up;
	size_t			  nat_up_cap;
	uint16_t		 *nat_act;
	size_t			  nat_act_cap;
	float			 *nat_f32;
	const uint16_t	 *quant_x16;
	int				  nat_n;
	int				  nat_m;
	uint16_t		 *y16;
	size_t			  y16_cap;
	uint16_t		 *res16;
	size_t			  res16_cap;
	const uint16_t	 *ew_res16;
	int				  ew_residual_failed;
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
	int				  bf16_native;
	int				  quant_failed;
	int				  quant_prefill_failed;
	rkt_quant_scratch quant_pf[RKT_MAX_CORES];
	int				  quant_logged;
	int				  pipelines_logged;
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

static inline uint16_t f32_to_bf16_bits(float v) {
	uint32_t bits;
	memcpy(&bits, &v, sizeof(bits));
	if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
		return (uint16_t)((bits >> 16) | 0x40u);
	bits += 0x7FFFu + ((bits >> 16) & 1u);
	return (uint16_t)(bits >> 16);
}

static inline float bf16_bits_to_f32(uint16_t bits) {
	uint32_t wide = (uint32_t)bits << 16;
	float	 v;
	memcpy(&v, &wide, sizeof(v));
	return v;
}

static const char *rkt_fmt_name(rkt_float_fmt fmt) {
	static const char *names[RKT_FMT_COUNT] = {
		[RKT_FMT_F16] = "f16", [RKT_FMT_BF16] = "bf16", [RKT_FMT_TF32] = "tf32"};
	return names[fmt];
}

extern const void *rkt_native_key;
void			   rkt_native_settle(void);

static inline void *rkt_ptr_raw(const buffer *b) {
	return (char *)b->handle + b->offset;
}

static inline void *rkt_ptr(const buffer *b) {
	void *ptr = rkt_ptr_raw(b);
	if (ptr == rkt_native_key)
		rkt_native_settle();
	return ptr;
}

#define RKT_QUANT_NTILE 4096
#define RKT_QUANT_HEAD_ROWS 65536
#define RKT_MAX_FUSE 8
#define RKT_QUANT_MAX_CHILDREN RKT_MAX_FUSE
#define RKT_QUANT_PREFILL_TOKENS 32
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
	rkt_bo	  pf_reg;
	uint32_t *pf_n_ops;
	size_t	  pf_cap;
	int		  pf_tokens;
	uint32_t  pf_in_dma;
	uint32_t  pf_out_dma;
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

int			rkt_ensure_bo(rkt_priv *p, int core, size_t bytes, rkt_bo *bo);
int			rkt_ensure_capacity(void **buf, size_t *cap, size_t need, size_t elem);
status_code rkt_parallel_rows(int n, status_code (*fn)(void *ctx, int lo, int hi), void *ctx);
status_code rkt_quant_build_best(rkt_priv *p, rkt_weight *w, const tensor_desc *desc);
void rkt_quant_cpu_matmul(const rkt_weight *w, int row0, int n, const float *x, float *y, int m);
void rkt_quant_disable(rkt_priv *p, status_code s);
void rkt_quant_disable_prefill(rkt_priv *p, status_code s);
void rkt_quant_free(rkt_priv *p, rkt_quant *q);
status_code rkt_quant_matmul(rkt_priv *p, rkt_weight *w, const float *x, float *y, int m);
status_code rkt_quant_run_many(rkt_priv *p, rkt_weight **ws, float **ys, int count, const float *x,
							   int tokens);
int			rkt_quant_servable(const rkt_priv *p, const rkt_weight *w, int row0, int n, int m);

void		 rkt_queue_abort(rkt_priv *p);
status_code	 rkt_queue_flush(rkt_priv *p);
status_code	 rkt_queue_flush_async(rkt_priv *p);
status_code	 rkt_queue_sync(rkt_priv *p);
status_code	 rkt_dense_matmul(rkt_priv *p, const rkt_weight *w, int row0, int n, const float *x,
							  float *y, int m);
status_code	 rkt_dense_matmul_enqueue_k(rkt_priv *p, const rkt_weight *w, int row0, int n,
										const float *x, float *y, int m, int k_used);
status_code	 rkt_dense_matmul_enqueue(rkt_priv *p, const rkt_weight *w, int row0, int n,
									  const float *x, float *y, int m);
int			 rkt_next_block_rows(int remaining, int max_rows);
void		 rkt_npu_record_failure(rkt_priv *p);
status_code	 rkt_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x, buffer *y,
						int n, int k);
status_code	 rkt_matmul_batch(backend *self, const buffer *wbuf, uint32_t w_type, const buffer *x,
							  buffer *y, int n, int k, int m);
status_code	 rkt_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
							  const buffer *x, buffer **y, const int *n_list, int k, int n_matmuls);
status_code	 rkt_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
									const buffer *x, buffer **y, const int *n_list, int k,
									int n_matmuls, int m);
status_code	 rkt_matmul_residual(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								 const buffer *residual, buffer *y, int n, int k);
int			 rkt_build_layout(rkt_priv *p, rkt_weight *w);
int			 rkt_flush_weight(rkt_priv *p, rkt_weight *w);
void		 rkt_register_weight(rkt_priv *p, rkt_weight *w);
void		 rkt_release_npu_state(rkt_priv *p, rkt_weight *w);
void		 rkt_lut_configure(rkt_lut_cfg *cfg, rkt_lut_kind kind);
const char	*rkt_lut_name(rkt_lut_kind kind);
rkt_lut_kind rkt_lut_kind_for_activation(int activation);
void		 rkt_lut_state_reset(rkt_priv *p, rkt_lut_state *a);
status_code	 rkt_lut_state_init(rkt_priv *p, rkt_lut_state *a);
status_code	 rkt_lut_activate_rows(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind,
								   const float *gate, int gate_stride, const float *up,
								   int up_stride, float *out, int n, int m);
status_code	 rkt_lut_softcap_rows(rkt_priv *p, rkt_lut_state *a, const float *x, float *out, int n,
								  float cap);
status_code	 rkt_lut_gate_act_prepare(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind);
status_code	 rkt_lut_add_inplace(rkt_priv *p, rkt_lut_state *a, float *x, const float *y, int n,
								 int m, float out_scale);
status_code	 rkt_lut_scale_inplace(rkt_priv *p, rkt_lut_state *a, float *x, int n, float scale);
status_code	 rkt_native_gate_act(rkt_priv *p, const rkt_weight *gw, int grow0, const rkt_weight *uw,
								 int urow0, const float *x, float *act, int n, int m,
								 rkt_lut_kind kind);
int			 rkt_native_pending(const rkt_priv *p, const buffer *x, int k, int m);
int rkt_native_matches(const rkt_priv *p, const buffer *x, const rkt_weight *w, int k, int m);
status_code rkt_native_ffn_activate(rkt_priv *p, rkt_lut_kind kind, const float *gate,
									int gate_stride, const float *up, int up_stride, float *out,
									int n, int m);
status_code rkt_quant_matmul_native(rkt_priv *p, rkt_weight *w, float *y, int m);
status_code rkt_lut_activate_rows_f16(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind,
									  const float *gate, int gate_stride, const float *up,
									  int up_stride, uint16_t *out16, int out_stride, int n, int m);
void		rkt_native_free(rkt_priv *p);
void		rkt_native_release(rkt_priv *p);
status_code rkt_dense_matmul_enqueue_f16(rkt_priv *p, const rkt_weight *w, int row0, int n,
										 const float *x, uint16_t *y16, int y16_stride, int m);
status_code rkt_dense_matmul_native(rkt_priv *p, const rkt_weight *w, int row0, int n, float *y,
									int m);
status_code rkt_lut_activate_native_mul(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind,
										const uint16_t *gate16, const uint16_t *up16,
										uint16_t *out16, int out_stride, int n, int m);
void		rkt_convert_from_f16(float *dst, const uint16_t *src, size_t n);
void		rkt_tile_weight_row(uint16_t *bo, const uint16_t *row, int k_len, int local_row);
void		rkt_tile_input(uint16_t *dst, const uint16_t *src, int m, int k);
void		rkt_pack_ew_operand(uint16_t *dst, const uint16_t *src, int m, int channels);
void		rkt_convert_to_bf16(uint16_t *dst, const float *src, size_t n);
void		rkt_convert_to_f16(uint16_t *dst, const float *src, size_t n);
void		rkt_tile_input_esz(uint8_t *dst, const uint8_t *src, int m, int k, int esz);
int			rkt_weight_esz(const rkt_weight *w);
int			rkt_max_batch_rows(int k_len, int esz);
const rkt_part *rkt_part_for_row(const rkt_kchunk *kc, int row);
void			rkt_log_pipeline_usage(rkt_priv *p);

#endif
