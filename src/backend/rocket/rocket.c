#include "backend/backend.h"
#include "config.h"
#include "device.h"
#include "emit.h"
#include "log.h"
#include "quant.h"
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define RKT_DEVICE_PATH "/dev/accel/accel0"
#define RKT_TIMEOUT_NS 10000000000LL
#define RKT_CACHE_BYTES ((size_t)2 << 30)
#define RKT_BATCH_ROWS 128
#define RKT_TILE_COLS 8192
#define RKT_TILE_K 1280
#define RKT_LOW_SCALE 1024.0f
#define RKT_MIN_ROWS 32
#define RKT_MAX_K 32768
#define RKT_MAX_CHUNKS ((RKT_MAX_K + RKT_INT8_K - 1) / RKT_INT8_K)
#define RKT_SPLIT_K_MAX_COLS 2048
#define RKT_BO_BINS 20

enum { RKT_INPUT_BUFFER, RKT_OUTPUT_BUFFER, RKT_COMMAND_BUFFER, RKT_BUFFER_COUNT };

enum { RKT_HIGH_PRODUCT, RKT_INPUT_CORRECTION, RKT_WEIGHT_CORRECTION, RKT_PRODUCT_COUNT };
_Static_assert(RKT_PRODUCT_COUNT == RKT_MAX_CORES, "");

typedef struct {
	rkt_bo bo;
	int	   first, cols, first_chunk, chunks;
} rkt_part;

typedef struct rkt_weight rkt_weight;
struct rkt_weight {
	const void *allocation, *source;
	int			n, k, max_cols, tile_k, chunks, split_k;
	size_t		bytes;
	rkt_part	parts[RKT_MAX_CORES];
	float	   *scales, *rotation;
	rkt_weight *next;
};

typedef struct {
	rkt_bo *input, *output, *commands;
	rkt_bo	buffers[RKT_BUFFER_COUNT][RKT_BO_BINS];
} rkt_slot;

typedef struct {
	backend *host;
	status_code (*host_write_f32)(backend *, buffer *, const float *, int);
	status_code (*host_copy_buffer)(backend *, const buffer *, buffer *, int);
	rkt_device				dev;
	pthread_mutex_t			lock;
	rkt_weight			   *weights;
	rkt_slot				slots[RKT_MAX_CORES];
	struct drm_rocket_task *tasks;
	size_t					task_capacity;
	size_t					cache_bytes, cache_limit;
	rkt_precision			precision;
	float					input_scales[RKT_BATCH_ROWS * RKT_MAX_CHUNKS];
	int						failed;
} rkt_priv;

static int split_half(float value, _Float16 *high, _Float16 *low) {
	if (!isfinite(value) || fabsf(value) > 65504.0f)
		return 0;
	*high = (_Float16)value;
	*low  = (_Float16)((value - (float)*high) * RKT_LOW_SCALE);
	return 1;
}

static void destroy_weight(rkt_priv *p, rkt_weight *w) {
	for (int c = 0; c < RKT_MAX_CORES; c++)
		rkt_bo_destroy(&p->dev, &w->parts[c].bo);
	p->cache_bytes -= w->bytes;
	free(w->scales);
	free(w->rotation);
	free(w);
}

static void invalidate_weights(rkt_priv *p, const void *allocation) {
	rkt_weight **link = &p->weights;
	while (*link) {
		rkt_weight *w = *link;
		if (w->allocation == allocation) {
			*link = w->next;
			destroy_weight(p, w);
		} else {
			link = &w->next;
		}
	}
}

static int pack_weight_f16(rkt_weight *w) {
	const q8_0_block *src  = w->source;
	_Float16		 *high = w->parts[RKT_HIGH_PRODUCT].bo.map;
	_Float16		 *low  = w->parts[RKT_WEIGHT_CORRECTION].bo.map;
	int				  cols = w->max_cols;
	for (int r = 0; r < w->n; r++)
		for (int b = 0; b < w->k / 32; b++) {
			const q8_0_block *q		= &src[(size_t)r * (w->k / 32) + b];
			float			  d		= rkt_half_to_float(q->d);
			int				  chunk = b / (w->tile_k / 32);
			int				  kb	= b % (w->tile_k / 32);
			size_t			  index = (size_t)chunk * cols * w->tile_k +
						   ((size_t)(r / 16) * (w->tile_k / 32) + kb) * 512 + (r % 16) * 32;
			for (int j = 0; j < 32; j++)
				if (!split_half(d * q->qs[j], &high[index + j], &low[index + j]))
					return 0;
		}
	memcpy(w->parts[RKT_INPUT_CORRECTION].bo.map, high,
		   (size_t)cols * w->chunks * w->tile_k * sizeof(_Float16));
	return 1;
}

static rkt_weight *prepare_weight(rkt_priv *p, const buffer *buf, int n, int k) {
	const void *source = (const char *)buf->handle + buf->offset;
	for (rkt_weight *w = p->weights; w; w = w->next)
		if (w->source == source && w->n == n && w->k == k)
			return w;
	int	   int8	   = p->precision == RKT_INT8;
	int	   max_k   = int8 ? RKT_INT8_K : RKT_TILE_K;
	int	   chunks  = (k + max_k - 1) / max_k;
	int	   tile_k  = ((k + chunks - 1) / chunks + 31) / 32 * 32;
	int	   split_k = int8 && n <= RKT_SPLIT_K_MAX_COLS && chunks >= p->dev.n_cores;
	int	   align   = int8 ? 32 : 16;
	int	   groups  = (n + align - 1) / align;
	size_t bytes   = (size_t)groups * align * chunks * tile_k * (int8 ? 1 : 6) +
				   (int8 ? ((size_t)n * chunks + k) * sizeof(float) : 0) +
				   RKT_MAX_CORES * p->dev.page_size;
	if (bytes > p->cache_limit - p->cache_bytes)
		return NULL;
	rkt_weight *w = calloc(1, sizeof(*w));
	if (!w)
		return NULL;
	*w = (rkt_weight){.allocation = buf->handle,
					  .source	  = source,
					  .n		  = n,
					  .k		  = k,
					  .tile_k	  = tile_k,
					  .chunks	  = chunks,
					  .split_k	  = split_k};
	if (int8) {
		w->scales	= malloc((size_t)n * chunks * sizeof(float));
		w->rotation = malloc((size_t)k * sizeof(float));
		if (!w->scales || !w->rotation) {
			destroy_weight(p, w);
			return NULL;
		}
		rkt_rotation_factors(w->rotation, k);
	}
	int cores = p->dev.n_cores;
	if (int8 && !split_k && groups < cores)
		cores = groups;
	int first = 0;
	for (int c = 0; c < cores; c++) {
		rkt_part *part	  = &w->parts[c];
		int		  count	  = split_k ? (chunks - first + cores - c - 1) / (cores - c)
							: int8	? (groups - first + cores - c - 1) / (cores - c)
									: groups;
		part->first		  = int8 && !split_k ? first * align : 0;
		part->cols		  = (split_k ? groups : count) * align;
		part->first_chunk = split_k ? first : 0;
		part->chunks	  = split_k ? count : chunks;
		first += count;
		if (part->cols > w->max_cols)
			w->max_cols = part->cols;
		if (rkt_bo_create(&p->dev, c,
						  (size_t)part->cols * part->chunks * tile_k *
							  (int8 ? 1 : sizeof(_Float16)),
						  &part->bo)) {
			destroy_weight(p, w);
			return NULL;
		}
		memset(part->bo.map, 0, part->bo.size);
		if (int8) {
			rkt_weight_pack pack = {.source		 = source,
									.data		 = part->bo.map,
									.scales		 = w->scales,
									.rotation	 = w->rotation,
									.n			 = n,
									.k			 = k,
									.first		 = part->first,
									.cols		 = part->cols,
									.tile_k		 = tile_k,
									.first_chunk = part->first_chunk,
									.chunks		 = part->chunks};
			for (int row = 0; row < part->cols && part->first + row < n; row++)
				if (!rkt_pack_weight_i8(&pack, row)) {
					destroy_weight(p, w);
					return NULL;
				}
		}
	}
	if (!int8 && !pack_weight_f16(w)) {
		destroy_weight(p, w);
		return NULL;
	}
	for (int c = 0; c < cores; c++)
		if (rkt_bo_flush(&p->dev, &w->parts[c].bo)) {
			destroy_weight(p, w);
			return NULL;
		}
	w->bytes = int8 ? ((size_t)n * chunks + k) * sizeof(float) : 0;
	for (int c = 0; c < cores; c++)
		w->bytes += w->parts[c].bo.size;
	p->cache_bytes += w->bytes;
	w->next	   = p->weights;
	p->weights = w;
	return w;
}

static rkt_bo *prepare_bo(rkt_priv *p, int core, size_t bytes, rkt_bo *buffers) {
	int	   bin	= 0;
	size_t size = p->dev.page_size;
	while (size < bytes && bin + 1 < RKT_BO_BINS) {
		size *= 2;
		bin++;
	}
	if (size < bytes)
		return NULL;
	rkt_bo *bo = &buffers[bin];
	if (!bo->map && rkt_bo_create(&p->dev, core, size, bo))
		return NULL;
	return bo;
}

static status_code prepare_slots(rkt_priv *p, const rkt_weight *w, int m) {
	size_t tasks = (size_t)w->chunks * RKT_MAX_CORES;
	if (tasks > p->task_capacity) {
		void *grown = realloc(p->tasks, tasks * sizeof(*p->tasks));
		if (!grown)
			return ERR_OUT_OF_MEMORY;
		p->tasks		 = grown;
		p->task_capacity = tasks;
	}
	int bytes = p->precision == RKT_INT8 ? 1 : sizeof(_Float16);
	for (int c = 0; c < p->dev.n_cores; c++) {
		if (!w->parts[c].cols)
			continue;
		int		  cols = w->parts[c].cols < RKT_TILE_COLS ? w->parts[c].cols : RKT_TILE_COLS;
		rkt_slot *s	   = &p->slots[c];
		s->input	   = prepare_bo(p, c, (size_t)m * w->chunks * w->tile_k * bytes,
									s->buffers[RKT_INPUT_BUFFER]);
		s->output	   = prepare_bo(p, c, (size_t)m * cols * w->parts[c].chunks * sizeof(float),
									s->buffers[RKT_OUTPUT_BUFFER]);
		s->commands	   = prepare_bo(p, c, (size_t)w->parts[c].chunks * RKT_REGCMD_BYTES,
									s->buffers[RKT_COMMAND_BUFFER]);
		if (!s->input || !s->output || !s->commands)
			return ERR_OUT_OF_MEMORY;
	}
	return OK;
}

static status_code prepare_input_f16(rkt_priv *p, const rkt_weight *w, const float *x, int rows,
									 int m) {
	size_t	  bytes = (size_t)m * w->chunks * w->tile_k * sizeof(_Float16);
	_Float16 *high	= p->slots[RKT_HIGH_PRODUCT].input->map;
	_Float16 *low	= p->slots[RKT_INPUT_CORRECTION].input->map;
	memset(high, 0, bytes);
	memset(low, 0, bytes);
	for (int r = 0; r < rows; r++) {
		buffer input =
			buffer_host_view(p->host, (void *)(x + (size_t)r * w->k), (size_t)w->k * sizeof(float));
		buffer		packed = {0};
		status_code st	   = p->host->prequantize_x(p->host, &input, w->k, GGML_TYPE_Q8_0, &packed);
		if (st != OK)
			return st;
		const q8_0_block *q = packed.handle;
		for (int b = 0; b < w->k / 32; b++) {
			float d		= rkt_half_to_float(q[b].d);
			int	  chunk = b / (w->tile_k / 32);
			int	  kb	= b % (w->tile_k / 32);
			for (int j = 0; j < 32; j++) {
				int	   col = kb * 32 + j;
				size_t index =
					(size_t)chunk * m * w->tile_k + (size_t)(col / 8) * m * 8 + r * 8 + col % 8;
				if (!split_half(d * q[b].qs[j], &high[index], &low[index]))
					return ERR_UNSUPPORTED;
			}
		}
	}
	memcpy(p->slots[RKT_WEIGHT_CORRECTION].input->map, high, bytes);
	for (int c = 0; c < p->dev.n_cores; c++)
		if (rkt_bo_flush(&p->dev, p->slots[c].input))
			return ERR_IO;
	return OK;
}

typedef struct {
	const rkt_weight *weight;
	const float		 *x;
	int8_t			 *dest;
	float			 *scales;
	int				 *valid;
	int				  padded;
} rkt_input_job;

static void prepare_input_rows(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const rkt_input_job *j = ctx;
	const rkt_weight	*w = j->weight;
	for (int r = begin; r < end; r++)
		j->valid[r] = rkt_pack_input_i8(j->x + (size_t)r * w->k, j->dest, j->scales, w->k,
										j->padded, r, w->tile_k, w->rotation);
}

static status_code prepare_input(rkt_priv *p, const rkt_weight *w, const float *x, int rows,
								 int m) {
	if (p->precision == RKT_FLOAT16)
		return prepare_input_f16(p, w, x, rows, m);
	size_t	bytes = (size_t)m * w->chunks * w->tile_k;
	int8_t *dest  = p->slots[0].input->map;
	memset(dest, 0, bytes);
	memset(p->input_scales, 0, (size_t)m * w->chunks * sizeof(float));
	int			  valid[RKT_BATCH_ROWS] = {0};
	rkt_input_job job					= {
						  .weight = w, .x = x, .dest = dest, .scales = p->input_scales, .valid = valid, .padded = m};
	tpool *pool = backend_get_pool(p->host);
	if (pool && tpool_current_tid() < 0 && rows >= 16)
		tpool_parallel_for(pool, rows, 8, prepare_input_rows, &job);
	else
		prepare_input_rows(0, rows, 0, &job);
	for (int r = 0; r < rows; r++)
		if (!valid[r])
			return ERR_UNSUPPORTED;
	for (int c = 0; c < p->dev.n_cores; c++) {
		if (!w->parts[c].cols)
			continue;
		if (c)
			memcpy(p->slots[c].input->map, dest, bytes);
		if (rkt_bo_flush(&p->dev, p->slots[c].input))
			return ERR_IO;
	}
	return OK;
}

typedef struct {
	const rkt_weight *weight;
	const float		 *partial[RKT_MAX_CORES];
	const float		 *input_scales;
	const int32_t	**integer;
	float			 *out;
	int				  first, cols, rows, padded;
} rkt_reduce_job;

static void reduce_f16(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const rkt_reduce_job *j = ctx;
	const rkt_weight	 *w = j->weight;
	for (int group = begin; group < end; group++) {
		int col	  = group * 4;
		int lanes = w->n - j->first - col;
		if (lanes > 4)
			lanes = 4;
		for (int row = 0; row < j->rows; row += 16) {
			int	  nr		  = j->rows - row < 16 ? j->rows - row : 16;
			float sums[16][4] = {{0}};
			for (int b = 0; b < w->chunks; b++) {
				size_t offset =
					(size_t)b * j->padded * j->cols + (size_t)group * j->padded * 4 + row * 4;
				const float *high		= j->partial[RKT_HIGH_PRODUCT] + offset;
				const float *input_low	= j->partial[RKT_INPUT_CORRECTION] + offset;
				const float *weight_low = j->partial[RKT_WEIGHT_CORRECTION] + offset;
				for (int r = 0; r < nr; r++)
					for (int lane = 0; lane < lanes; lane++) {
						int index = r * 4 + lane;
						sums[r][lane] +=
							high[index] + (input_low[index] + weight_low[index]) / RKT_LOW_SCALE;
					}
			}
			for (int r = 0; r < nr; r++)
				memcpy(j->out + (size_t)(row + r) * w->n + j->first + col, sums[r],
					   (size_t)lanes * sizeof(float));
		}
	}
}

static void reduce_i8(int begin, int end, int tid, void *ctx) {
	(void)tid;
	const rkt_reduce_job *j = ctx;
	const rkt_weight	 *w = j->weight;
	for (int group = begin; group < end; group++) {
		int col	  = group * 4;
		int lanes = w->n - j->first - col;
		if (lanes > 4)
			lanes = 4;
#if defined(__aarch64__)
		if (lanes == 4) {
			int row = 0;
			for (; row + 8 <= j->rows; row += 8) {
				float32x4_t sums[8] = {0};
				for (int b = 0; b < w->chunks; b++) {
					size_t		 offset	 = (size_t)group * j->padded * 4 + row * 4;
					float32x4_t	 weights = vld1q_f32(w->scales + (size_t)b * w->n + j->first + col);
					const float *input	 = j->input_scales + (size_t)b * j->padded + row;
					for (int r = 0; r < 8; r++) {
						float32x4_t scales = vmulq_n_f32(weights, input[r]);
						sums[r]			   = vfmaq_f32(
							   sums[r], vcvtq_f32_s32(vld1q_s32(j->integer[b] + offset + r * 4)),
							   scales);
					}
				}
				for (int r = 0; r < 8; r++)
					vst1q_f32(j->out + (size_t)(row + r) * w->n + j->first + col, sums[r]);
			}
			for (; row < j->rows; row++) {
				float32x4_t sum = vdupq_n_f32(0.0f);
				for (int b = 0; b < w->chunks; b++) {
					size_t		offset	= (size_t)group * j->padded * 4 + row * 4;
					float32x4_t weights = vld1q_f32(w->scales + (size_t)b * w->n + j->first + col);
					float32x4_t scale =
						vmulq_n_f32(weights, j->input_scales[(size_t)b * j->padded + row]);
					sum = vfmaq_f32(sum, vcvtq_f32_s32(vld1q_s32(j->integer[b] + offset)), scale);
				}
				vst1q_f32(j->out + (size_t)row * w->n + j->first + col, sum);
			}

			continue;
		}
#endif
		for (int row = 0; row < j->rows; row++) {
			for (int lane = 0; lane < lanes; lane++) {
				float sum = 0.0f;
				for (int b = 0; b < w->chunks; b++) {
					size_t offset = (size_t)group * j->padded * 4 + row * 4 + lane;
					float  scale  = j->input_scales[(size_t)b * j->padded + row] *
								  w->scales[(size_t)b * w->n + j->first + col + lane];
					sum = fmaf((float)j->integer[b][offset], scale, sum);
				}
				j->out[(size_t)row * w->n + j->first + col + lane] = sum;
			}
		}
	}
}

typedef struct {
	rkt_reduce_job jobs[RKT_MAX_CORES];
	int			   groups[RKT_MAX_CORES];
	rkt_precision  precision;
} rkt_reduce_batch;

static void reduce_batch(int begin, int end, int tid, void *ctx) {
	rkt_reduce_batch *batch = ctx;
	int				  first = 0;
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		int start = begin > first ? begin - first : 0;
		int stop  = end - first < batch->groups[c] ? end - first : batch->groups[c];
		if (start < stop) {
			if (batch->precision == RKT_INT8)
				reduce_i8(start, stop, tid, &batch->jobs[c]);
			else
				reduce_f16(start, stop, tid, &batch->jobs[c]);
		}
		first += batch->groups[c];
	}
}

static status_code run_tiles(rkt_priv *p, const rkt_weight *w, int rows, int m, int offset,
							 float *y) {
	int			int8				  = p->precision == RKT_INT8;
	int			bytes				  = int8 ? 1 : sizeof(_Float16);
	int			counts[RKT_MAX_CORES] = {0};
	status_code st					  = OK;
	for (int c = 0; c < p->dev.n_cores; c++) {
		int cols = w->parts[c].cols - offset;
		if (cols <= 0)
			continue;
		if (cols > RKT_TILE_COLS)
			cols = RKT_TILE_COLS;
		counts[c]					  = cols;
		rkt_slot			   *s	  = &p->slots[c];
		struct drm_rocket_task *tasks = p->tasks + (size_t)c * w->chunks;
		for (int b = 0; b < w->parts[c].chunks; b++) {
			int		 global_chunk = w->parts[c].first_chunk + b;
			uint32_t commands = (uint32_t)(s->commands->dma_addr + (size_t)b * RKT_REGCMD_BYTES);
			int		 words	  = rkt_emit_matmul(
				(uint64_t *)((char *)s->commands->map + (size_t)b * RKT_REGCMD_BYTES), p->precision,
				m, w->tile_k, cols,
				(uint32_t)(s->input->dma_addr + (size_t)global_chunk * m * w->tile_k * bytes),
				(uint32_t)(w->parts[c].bo.dma_addr +
						   ((size_t)b * w->parts[c].cols + offset) * w->tile_k * bytes),
				(uint32_t)(s->output->dma_addr + (size_t)b * m * cols * sizeof(float)));
			tasks[b] =
				(struct drm_rocket_task){.regcmd = commands, .regcmd_count = (uint32_t)words};
		}
		if (rkt_bo_flush(&p->dev, s->commands))
			return ERR_IO;
	}
	int submitted[RKT_MAX_CORES] = {0};
	for (int c = 0; c < p->dev.n_cores && st == OK; c++) {
		if (!counts[c])
			continue;
		rkt_slot *s = &p->slots[c];
		if (rkt_submit(&p->dev, s->commands, p->tasks + (size_t)c * w->chunks, w->parts[c].chunks,
					   s->input, s->output, &w->parts[c].bo))
			st = ERR_COMPUTE_FAIL;
		else
			submitted[c] = 1;
	}
	for (int c = 0; c < p->dev.n_cores; c++)
		if (submitted[c] && rkt_bo_wait(&p->dev, p->slots[c].output))
			st = ERR_COMPUTE_FAIL;
	if (st != OK)
		return st;
	const int32_t *planes[RKT_MAX_CORES][RKT_MAX_CHUNKS] = {{0}};
	for (int c = 0; c < p->dev.n_cores; c++) {
		if (!counts[c])
			continue;
		for (int b = 0; b < w->parts[c].chunks; b++)
			planes[w->split_k ? 0 : c][w->parts[c].first_chunk + b] =
				(const int32_t *)p->slots[c].output->map + (size_t)b * m * counts[c];
	}

	rkt_reduce_batch batch = {.precision = p->precision};
	int				 total = 0;
	for (int c = 0; c < (int8 && !w->split_k ? p->dev.n_cores : 1); c++) {
		if (!counts[c])
			continue;
		int				first = w->parts[c].first + offset;
		int				cols  = w->n - first < counts[c] ? w->n - first : counts[c];
		rkt_reduce_job *job	  = &batch.jobs[c];
		*job				  = (rkt_reduce_job){.weight	   = w,
												 .out		   = y,
												 .first		   = first,
												 .cols		   = counts[c],
												 .rows		   = rows,
												 .padded	   = m,
												 .input_scales = p->input_scales,
												 .integer	   = planes[c]};
		if (int8)
			job->partial[0] = p->slots[c].output->map;
		else
			for (int core = 0; core < p->dev.n_cores; core++)
				job->partial[core] = p->slots[core].output->map;
		batch.groups[c] = (cols + 3) / 4;
		total += batch.groups[c];
	}
	tpool *pool = backend_get_pool(p->host);
	if (pool && tpool_current_tid() < 0 && total >= 16)
		tpool_parallel_for(pool, total, 4, reduce_batch, &batch);
	else
		reduce_batch(0, total, 0, &batch);

	return OK;
}

static status_code rkt_matmul_batch(backend *self, const buffer *w, uint32_t type, const buffer *x,
									buffer *y, int n, int k, int m) {
	if (n < 0 || k < 0 || m < 0)
		return ERR_INVALID_ARG;
	if (!n || !m)
		return OK;
	if (!w || !x || !y)
		return ERR_INVALID_ARG;
	rkt_priv *p = self->device_priv;
	if (!k || type != GGML_TYPE_Q8_0 || k % 32 || k > RKT_MAX_K || m < RKT_MIN_ROWS ||
		n > INT_MAX - 32)
		return p->host->matmul_batch(p->host, w, type, x, y, n, k, m);
	pthread_mutex_lock(&p->lock);
	rkt_weight *packed = p->failed ? NULL : prepare_weight(p, w, n, k);
	status_code st	   = OK;
	if (packed)
		for (int start = 0; start < m && st == OK; start += RKT_BATCH_ROWS) {
			int	   rows	  = m - start < RKT_BATCH_ROWS ? m - start : RKT_BATCH_ROWS;
			int	   padded = (rows + 3) / 4 * 4;
			float *out	  = buffer_f32(y) + (size_t)start * n;
			st			  = prepare_slots(p, packed, padded);
			if (st == OK)
				st = prepare_input(p, packed, buffer_f32(x) + (size_t)start * k, rows, padded);
			if (st != OK)
				break;
			for (int offset = 0; offset < packed->max_cols && st == OK; offset += RKT_TILE_COLS)
				st = run_tiles(p, packed, rows, padded, offset, out);
		}
	if (st != OK && st != ERR_UNSUPPORTED) {
		p->failed = 1;
		backend_report_host_fallback(self, "matmul_batch", HFB_ERROR, "NPU job failed (status=%d)",
									 st);
	}
	pthread_mutex_unlock(&p->lock);
	if (!packed || st != OK)
		return p->host->matmul_batch(p->host, w, type, x, y, n, k, m);
	return OK;
}

static status_code rkt_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *types,
										  const buffer *x, buffer **y, const int *n, int k,
										  int count, int m) {
	rkt_priv *p = self->device_priv;
	if (m < RKT_MIN_ROWS)
		return p->host->matmul_multi_batch(p->host, w, types, x, y, n, k, count, m);
	int	   total = 0, contiguous = count > 1 && k > 0 && k <= RKT_MAX_K && k % 32 == 0;
	size_t row_bytes = contiguous ? ggml_row_size(GGML_TYPE_Q8_0, k) : 0;
	size_t offset	 = contiguous ? w[0]->offset : 0;
	for (int i = 0; i < count && contiguous; i++) {
		if (types[i] != GGML_TYPE_Q8_0 || w[i]->handle != w[0]->handle || w[i]->offset != offset ||
			n[i] <= 0 || n[i] > INT_MAX - total) {
			contiguous = 0;
			break;
		}
		total += n[i];
		offset += (size_t)n[i] * row_bytes;
	}
	if (contiguous && (size_t)m <= SIZE_MAX / sizeof(float) / (size_t)total) {
		size_t bytes = (size_t)m * total * sizeof(float);
		float *data	 = malloc(bytes);
		if (data) {
			buffer combined	   = *w[0];
			combined.size	   = (size_t)total * row_bytes;
			buffer		output = buffer_host_view(self, data, bytes);
			status_code st =
				rkt_matmul_batch(self, &combined, GGML_TYPE_Q8_0, x, &output, total, k, m);
			if (st == OK) {
				int col = 0;
				for (int i = 0; i < count; i++) {
					for (int row = 0; row < m; row++)
						memcpy(buffer_f32(y[i]) + (size_t)row * n[i],
							   data + (size_t)row * total + col, (size_t)n[i] * sizeof(float));
					col += n[i];
				}
			}
			free(data);
			return st;
		}
	}
	for (int i = 0; i < count; i++) {
		status_code st = rkt_matmul_batch(self, w[i], types[i], x, y[i], n[i], k, m);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code rkt_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
	if (!desc || desc->n_dims < 1 || desc->n_dims > 4 || !desc->dims[0])
		return ERR_INVALID_ARG;
	size_t bytes = ggml_row_size(desc->type, desc->dims[0]);
	if (!bytes || (!desc->host_data && !desc->read_fn))
		return ERR_INVALID_ARG;
	for (int i = 1; i < desc->n_dims; i++) {
		size_t rows = desc->dims[i];
		if (desc->type >= GGML_TYPE_Q8_0_REPACKED && desc->type <= GGML_TYPE_Q6_K_REPACKED &&
			desc->n_dims == 2) {
			if (rows > SIZE_MAX - 7)
				return ERR_INVALID_ARG;
			rows = (rows + 7) / 8 * 8;
		}
		if (!rows || rows > SIZE_MAX / bytes)
			return ERR_INVALID_ARG;
		bytes *= rows;
	}
	void *data = malloc(bytes);
	if (!data)
		return ERR_OUT_OF_MEMORY;
	status_code st = tensor_desc_read(desc, 0, data, bytes);
	if (st != OK) {
		free(data);
		return st;
	}
	rkt_priv *p = self->device_priv;
	*out		= (buffer){.handle = data, .size = bytes, .owner = p->host};
	return OK;
}

static void rkt_buffer_free(backend *self, buffer *buf) {
	if (!buf)
		return;
	rkt_priv *p = self->device_priv;
	pthread_mutex_lock(&p->lock);
	invalidate_weights(p, buf->handle);
	pthread_mutex_unlock(&p->lock);
	if (!buf->host_ptr)
		free(buf->handle);
	*buf = (buffer){0};
}

static status_code rkt_buffer_write_f32(backend *self, buffer *buf, const float *src, int n) {
	rkt_priv *p = self->device_priv;
	pthread_mutex_lock(&p->lock);
	invalidate_weights(p, buf->handle);
	status_code st = p->host_write_f32(p->host, buf, src, n);
	pthread_mutex_unlock(&p->lock);
	return st;
}

static status_code rkt_copy_buffer(backend *self, const buffer *src, buffer *dst, int n) {
	rkt_priv *p = self->device_priv;
	pthread_mutex_lock(&p->lock);
	invalidate_weights(p, dst->handle);
	status_code st = p->host_copy_buffer(p->host, src, dst, n);
	pthread_mutex_unlock(&p->lock);
	return st;
}

static status_code rkt_kv_alloc(backend *self, const kv_desc *desc, buffer *k, buffer *v) {
	rkt_priv   *p  = self->device_priv;
	status_code st = p->host->kv_alloc(self, desc, k, v);
	if (st == OK) {
		memset(k->handle, 0, k->size);
		memset(v->handle, 0, v->size);
	}
	return st;
}

static status_code rkt_probe(void) {
	char compatible[256] = {0};
	int	 f				 = open("/proc/device-tree/compatible", O_RDONLY | O_CLOEXEC);
	if (f < 0)
		return ERR_UNSUPPORTED;
	ssize_t bytes = read(f, compatible, sizeof(compatible) - 1);
	close(f);
	int supported = 0;
	for (ssize_t i = 0; i + 15 <= bytes; i++)
		if (memcmp(compatible + i, "rockchip,rk3588", 14) == 0)
			supported = 1;
	if (!supported)
		return ERR_UNSUPPORTED;
	f = open(RKT_DEVICE_PATH, O_RDWR | O_CLOEXEC);
	if (f < 0)
		return ERR_UNSUPPORTED;
	char			   name[32] = {0};
	struct drm_version version	= {.name_len = sizeof(name) - 1, .name = name};
	int valid = ioctl(f, DRM_IOCTL_VERSION, &version) == 0 && strcmp(name, "rocket") == 0;
	close(f);
	return valid ? OK : ERR_UNSUPPORTED;
}

static void rkt_free(backend *self) {
	rkt_priv *p = self->device_priv;
	if (!p)
		return;
	while (p->weights) {
		rkt_weight *w = p->weights;
		p->weights	  = w->next;
		destroy_weight(p, w);
	}
	for (int c = 0; c < RKT_MAX_CORES; c++)
		for (int kind = 0; kind < RKT_BUFFER_COUNT; kind++)
			for (int b = 0; b < RKT_BO_BINS; b++)
				rkt_bo_destroy(&p->dev, &p->slots[c].buffers[kind][b]);
	rkt_device_close(&p->dev);
	free(p->tasks);
	backend_destroy(p->host);
	pthread_mutex_destroy(&p->lock);
	free(p);
	self->priv		  = NULL;
	self->device_priv = NULL;
}

static status_code rkt_init(backend *self, int index) {
	if (index || rkt_probe() != OK)
		return ERR_UNSUPPORTED;
	rkt_priv *p = calloc(1, sizeof(*p));
	if (!p)
		return ERR_OUT_OF_MEMORY;
	pthread_mutex_init(&p->lock, NULL);
	self->device_priv = p;
	status_code st	  = backend_create_host(&p->host);
	if (st != OK) {
		rkt_free(self);
		return st;
	}
	if (rkt_device_open(&p->dev, RKT_DEVICE_PATH, RKT_MAX_CORES, RKT_TIMEOUT_NS)) {
		rkt_free(self);
		return ERR_UNSUPPORTED;
	}
	size_t available = p->host->mem_available ? p->host->mem_available(p->host) : 0;
	p->cache_limit	 = available / 4;
	p->precision	 = config_get()->npu_int8 ? RKT_INT8 : RKT_FLOAT16;
	if (p->cache_limit > RKT_CACHE_BYTES)
		p->cache_limit = RKT_CACHE_BYTES;
	p->host_write_f32		  = p->host->buffer_write_f32;
	p->host_copy_buffer		  = p->host->copy_buffer;
	p->host->device_priv	  = p;
	p->host->buffer_write_f32 = rkt_buffer_write_f32;
	p->host->buffer_free	  = rkt_buffer_free;
	p->host->copy_buffer	  = rkt_copy_buffer;
	*self					  = *p->host;
	self->name				  = "rocket";
	self->desc				  = "Rockchip RK3588 NPU (mainline rocket)";
	self->priority			  = -1;
	self->caps = (self->caps & ~(BCAP_IS_HOST | BCAP_MATMUL_QONLY)) | BCAP_HOST_VISIBLE_BUFFERS |
				 BCAP_HOST_RESIDENT_BUFFERS | BCAP_UNIFIED_MEMORY;
	self->probe				  = rkt_probe;
	self->init				  = rkt_init;
	self->free				  = rkt_free;
	self->kv_alloc			  = rkt_kv_alloc;
	self->buffer_alloc_weight = rkt_buffer_alloc_weight;
	self->buffer_write_f32	  = rkt_buffer_write_f32;
	self->matmul_batch		  = rkt_matmul_batch;
	self->matmul_multi_batch  = rkt_matmul_multi_batch;
	self->matmul_qonly		  = NULL;
	self->prequantize_x		  = NULL;
	self->repack_plan		  = NULL;
	log_tag("ROCKET", "%d contexts, Q8_0 %s matmul, %zu MiB weight cache", p->dev.n_cores,
			p->precision == RKT_INT8 ? "experimental rotated INT8" : "compensated FP16",
			p->cache_limit >> 20);
	return OK;
}

static status_code rkt_ctor(backend *out) {
	*out = (backend){.name	   = "rocket",
					 .desc	   = "Rockchip RK3588 NPU (mainline rocket)",
					 .priority = -1,
					 .caps	   = BCAP_HOST_VISIBLE_BUFFERS | BCAP_HOST_RESIDENT_BUFFERS |
							 BCAP_UNIFIED_MEMORY,
					 .probe = rkt_probe,
					 .init	= rkt_init,
					 .free	= rkt_free};
	return OK;
}

BACKEND_REGISTER("rocket", rkt_ctor)
