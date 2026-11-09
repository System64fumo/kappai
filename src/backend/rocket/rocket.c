#include "rocket.h"

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

int rkt_ensure_capacity(void **buf, size_t *cap, size_t need, size_t elem) {
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

int rkt_weight_esz(const rkt_weight *w) {
	return rkt_fmt_bytes(w->fmt);
}

int rkt_max_batch_rows(int k_len, int esz) {
	int weight_banks = (RKT_TILE_ROWS * k_len * esz + CBUF_BANK_BYTES - 1) / CBUF_BANK_BYTES;
	int data_banks	 = CBUF_BANKS - weight_banks;
	if (data_banks > CBUF_BANKS - 1)
		data_banks = CBUF_BANKS - 1;
	if (data_banks < 1)
		data_banks = 1;
	int m = data_banks * CBUF_BANK_BYTES / (k_len * esz);
	m	  = m / 4 * 4;
	return m < 4 ? 4 : m;
}

static void tile_weight_row_esz(uint8_t *bo, const uint8_t *row, int k_len, int local_row,
								int esz) {
	int cols	= RKT_TILE_COLS * 2 / esz;
	int n_kpg	= k_len / cols;
	int row_off = (local_row % RKT_TILE_ROWS) * cols;
	int group	= local_row / RKT_TILE_ROWS;
	for (int kpg = 0; kpg < n_kpg; kpg++) {
		size_t tile = ((size_t)group * n_kpg + kpg) * (RKT_TILE_ROWS * cols);
		memcpy(bo + (tile + row_off) * esz, row + (size_t)kpg * cols * esz, (size_t)cols * esz);
	}
}

void rkt_tile_weight_row(uint16_t *bo, const uint16_t *row, int k_len, int local_row) {
	tile_weight_row_esz((uint8_t *)bo, (const uint8_t *)row, k_len, local_row, 2);
}

void rkt_tile_input_esz(uint8_t *dst, const uint8_t *src, int m, int k, int esz) {
	int c2 = 16 / esz;
	for (int g = 0; g < k / c2; g++) {
		uint8_t *plane = dst + (size_t)g * m * 16;
		for (int r = 0; r < m; r++)
			memcpy(plane + (size_t)r * 16, src + ((size_t)r * k + (size_t)g * c2) * esz, 16);
	}
}

void rkt_tile_input(uint16_t *dst, const uint16_t *src, int m, int k) {
	rkt_tile_input_esz((uint8_t *)dst, (const uint8_t *)src, m, k, 2);
}

void rkt_pack_ew_operand(uint16_t *dst, const uint16_t *src, int m, int channels) {
	memset(dst, 0,
		   (size_t)(channels / RKT_EW_SURFACE_CHANNELS) * RKT_EW_SURFACE_SLOTS * m *
			   sizeof(uint16_t));
	for (int block = 0; block < channels / 8; block++) {
		uint16_t *base = dst + (size_t)((block + 1) / 2) * 16 * m + (size_t)(block / 2) * 8 * m;
		for (int r = 0; r < m; r++)
			memcpy(base + (size_t)r * 8, src + (size_t)r * channels + (size_t)block * 8,
				   8 * sizeof(uint16_t));
	}
}

void rkt_convert_to_bf16(uint16_t *dst, const float *src, size_t n) {
	const uint32x4_t abs_mask = vdupq_n_u32(0x7FFFFFFFu);
	const uint32x4_t inf_bits = vdupq_n_u32(0x7F800000u);
	const uint32x4_t round	  = vdupq_n_u32(0x7FFFu);
	const uint32x4_t quiet	  = vdupq_n_u32(0x40u);
	const uint32x4_t one	  = vdupq_n_u32(1u);
	size_t			 i		  = 0;
	for (; i + 4 <= n; i += 4) {
		uint32x4_t bits	  = vreinterpretq_u32_f32(vld1q_f32(src + i));
		uint32x4_t high	  = vshrq_n_u32(bits, 16);
		uint32x4_t is_nan = vcgtq_u32(vandq_u32(bits, abs_mask), inf_bits);
		uint32x4_t rounded =
			vshrq_n_u32(vaddq_u32(vaddq_u32(bits, round), vandq_u32(high, one)), 16);
		vst1_u16(dst + i, vmovn_u32(vbslq_u32(is_nan, vorrq_u32(high, quiet), rounded)));
	}
	for (; i < n; i++)
		dst[i] = f32_to_bf16_bits(src[i]);
}

void rkt_convert_from_f16(float *dst, const uint16_t *src, size_t n) {
	size_t i = 0;
	for (; i + 4 <= n; i += 4)
		vst1q_f32(dst + i, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(src + i))));
	for (; i < n; i++)
		dst[i] = f16_bits_to_f32(src[i]);
}

void rkt_convert_to_f16(uint16_t *dst, const float *src, size_t n) {
	size_t i = 0;
	for (; i + 8 <= n; i += 8) {
		float16x8_t h =
			vcombine_f16(vcvt_f16_f32(vld1q_f32(src + i)), vcvt_f16_f32(vld1q_f32(src + i + 4)));
		vst1q_u16(dst + i, vreinterpretq_u16_f16(h));
	}
	for (; i < n; i++)
		dst[i] = f32_to_f16_bits(src[i]);
}

void rkt_register_weight(rkt_priv *p, rkt_weight *w) {
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
	rkt_release_npu_state(p, w);
	if (w->host_copy) {
		backend *host = backend_host();
		host->buffer_free(host, w->host_copy);
		free(w->host_copy);
	}
	free(w->host_source);
	rkt_quant_free(p, w->quant);
	w->magic = 0;
	free(w);
}

static status_code row_to_f32(backend *host, uint32_t type, const void *src, int k, float *tmp) {
	if (type == GGML_TYPE_F32) {
		memcpy(tmp, src, (size_t)k * sizeof(float));
		return OK;
	}
	if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
		for (int i = 0; i < k; i++) {
			uint16_t bits;
			memcpy(&bits, (const uint8_t *)src + (size_t)i * 2, sizeof(bits));
			tmp[i] = type == GGML_TYPE_F16 ? f16_bits_to_f32(bits) : bf16_bits_to_f32(bits);
		}
		return OK;
	}
	if (!host || !host->dequant_row)
		return ERR_UNSUPPORTED;
	return host->dequant_row(host, type, src, k, tmp);
}

static status_code row_to_f16(backend *host, uint32_t type, const void *src, int k, float *tmp,
							  uint16_t *dst) {
	if (type == GGML_TYPE_F16) {
		memcpy(dst, src, (size_t)k * sizeof(uint16_t));
		return OK;
	}
	if (type == GGML_TYPE_F32) {
		rkt_convert_to_f16(dst, src, (size_t)k);
		return OK;
	}
	if (!host || !host->dequant_row)
		return ERR_UNSUPPORTED;
	status_code s = host->dequant_row(host, type, src, k, tmp);
	if (s != OK)
		return s;
	rkt_convert_to_f16(dst, tmp, (size_t)k);
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

int rkt_build_layout(rkt_priv *p, rkt_weight *w) {
	int start[RKT_MAX_CORES], count[RKT_MAX_CORES];
	int used = partition_rows(w->n_pad, p->dev.n_cores, start, count);

	int esz		 = rkt_weight_esz(w);
	int max_k	 = w->fmt == RKT_FMT_TF32 ? RKT_MAX_K_TF32 : RKT_MAX_K;
	w->n_kchunks = (w->k_pad + max_k - 1) / max_k;
	w->kchunks	 = xcalloc((size_t)w->n_kchunks, sizeof(*w->kchunks));
	for (int ci = 0; ci < w->n_kchunks; ci++) {
		rkt_kchunk *kc = &w->kchunks[ci];
		kc->k_off	   = ci * max_k;
		kc->k_len	   = w->k_pad - kc->k_off < max_k ? w->k_pad - kc->k_off : max_k;
		for (int c = 0; c < used; c++) {
			kc->part[c].row_start = start[c];
			kc->part[c].row_count = count[c];
			size_t bytes		  = (size_t)count[c] * kc->k_len * esz;
			if (rkt_bo_create(&p->dev, c, bytes, &kc->part[c].bo) != 0)
				return 0;
		}
	}
	return 1;
}

const rkt_part *rkt_part_for_row(const rkt_kchunk *kc, int row) {
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		const rkt_part *pt = &kc->part[c];
		if (pt->bo.map && row >= pt->row_start && row < pt->row_start + pt->row_count)
			return pt;
	}
	return NULL;
}

int rkt_flush_weight(rkt_priv *p, rkt_weight *w) {
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

status_code rkt_parallel_rows(int n, status_code (*fn)(void *ctx, int lo, int hi), void *ctx) {
	par_job job = {.status = OK, .fn = fn, .ctx = ctx};
	tpool_parallel_for(tpool_fetch_pool(), n, RKT_PAR_MIN_ROWS, par_chunk, &job);
	return (status_code)job.status;
}

typedef status_code (*row_source_fn)(void *ctx, int row, void *dst, float *tmp);

typedef struct {
	rkt_weight	 *w;
	row_source_fn source;
	void		 *ctx;
} fill_job;

static status_code fill_rows(void *arg, int lo, int hi) {
	fill_job   *job = arg;
	rkt_weight *w	= job->w;
	int			esz = rkt_weight_esz(w);
	uint8_t	   *row = xcalloc((size_t)w->k_pad, esz);
	float	   *tmp = xmalloc((size_t)w->k * sizeof(float));
	status_code s	= OK;
	for (int r = lo; r < hi && s == OK; r++) {
		if (r < w->n)
			s = job->source(job->ctx, r, row, tmp);
		else
			memset(row, 0, (size_t)w->k_pad * esz);
		for (int ci = 0; ci < w->n_kchunks && s == OK; ci++) {
			const rkt_kchunk *kc = &w->kchunks[ci];
			const rkt_part	 *pt = rkt_part_for_row(kc, r);
			if (!pt) {
				s = ERR_INTERNAL;
				break;
			}
			tile_weight_row_esz(pt->bo.map, row + (size_t)kc->k_off * esz, kc->k_len,
								r - pt->row_start, esz);
		}
	}
	free(tmp);
	free(row);
	return s;
}

static status_code fill_weight(rkt_priv *p, rkt_weight *w, row_source_fn source, void *ctx) {
	fill_job	job = {.w = w, .source = source, .ctx = ctx};
	status_code s	= rkt_parallel_rows(w->n_pad, fill_rows, &job);
	if (s == OK && !rkt_flush_weight(p, w))
		s = ERR_IO;
	return s;
}

typedef struct {
	backend		  *host;
	uint32_t	   type;
	const uint8_t *data;
	size_t		   row_bytes;
	int			   k;
	rkt_float_fmt  fmt;
} gguf_rows;

static status_code gguf_row_source(void *ctx, int row, void *dst, float *tmp) {
	gguf_rows	  *q   = ctx;
	const uint8_t *src = q->data + (size_t)row * q->row_bytes;
	if (q->fmt == RKT_FMT_F16)
		return row_to_f16(q->host, q->type, src, q->k, tmp, dst);
	if (q->fmt == RKT_FMT_BF16 && q->type == GGML_TYPE_BF16) {
		memcpy(dst, src, (size_t)q->k * sizeof(uint16_t));
		return OK;
	}
	status_code s = row_to_f32(q->host, q->type, src, q->k, tmp);
	if (s != OK)
		return s;
	if (q->fmt == RKT_FMT_BF16)
		rkt_convert_to_bf16(dst, tmp, (size_t)q->k);
	else
		memcpy(dst, tmp, (size_t)q->k * sizeof(float));
	return OK;
}

static rkt_float_fmt native_fmt_for_type(const rkt_priv *p, uint32_t type) {
	if (type == GGML_TYPE_BF16 && p->bf16_native)
		return RKT_FMT_BF16;
	if (type == GGML_TYPE_F32)
		return RKT_FMT_TF32;
	return RKT_FMT_F16;
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
	if (!rkt_build_layout(p, w))
		return ERR_OUT_OF_MEMORY;
	gguf_rows q = {
		.host	   = backend_host(),
		.type	   = desc->type,
		.data	   = desc->host_data,
		.row_bytes = w->src_row_bytes,
		.k		   = w->k,
		.fmt	   = w->fmt,
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

void rkt_release_npu_state(rkt_priv *p, rkt_weight *w) {
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
	if (!rkt_build_layout(p, w)) {
		rkt_release_npu_state(p, w);
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
		bytes = ggml_row_size(desc->type, (size_t)desc->dims[0]);
		for (int d = 1; d < desc->n_dims; d++)
			bytes *= (size_t)desc->dims[d];
		void *copy = xmalloc(bytes);
		memcpy(copy, desc->host_data, bytes);
		*out = (buffer){.handle = copy, .size = bytes, .host_ptr = copy, .owner = self};
		return OK;
	}

	rkt_weight *w = new_weight(desc);
	w->fmt		  = native_fmt_for_type(p, desc->type);
	bytes		  = w->src_row_bytes * (size_t)w->n;

	status_code s = rkt_quant_build_best(p, w, desc);
	if (s != OK && s != ERR_UNSUPPORTED)
		WARN("rocket: quant build for %dx%d (%s) failed: status %d, stays f16", w->k, w->n,
			 ggml_type_name(desc->type), s);
	if (s != OK) {
		status_code f16_s =
			weight_shape_supported(p, w) ? upload_to_npu(p, w, desc) : ERR_UNSUPPORTED;
		if (f16_s != OK)
			rkt_release_npu_state(p, w);
		if (f16_s != OK && f16_s != ERR_UNSUPPORTED)
			WARN("rocket: weight %dx%d (%s) stays on host: status %d", w->k, w->n,
				 ggml_type_name(desc->type), f16_s);
		if (f16_s != OK) {
			status_code host_s = keep_on_host(w, desc);
			if (host_s != OK) {
				destroy_weight(p, w);
				return host_s;
			}
		}
	}

	pthread_mutex_lock(&p->lock);
	rkt_register_weight(p, w);
	pthread_mutex_unlock(&p->lock);

	*out = (buffer){.handle = w, .size = bytes, .owner = self};
	return OK;
}

static int rkt_matmul_type_native(backend *self, uint32_t w_type) {
	(void)self;
	return !type_is_group_repacked(w_type);
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

static void rkt_synchronize(backend *self) {
	(void)self;
	rkt_native_settle();
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

static status_code rkt_softcap(backend *self, buffer *x, float cap, int n) {
	rkt_priv *p = self->priv;
	if (n <= 0)
		return OK;
	status_code s = ERR_UNSUPPORTED;
	if (cap > 0.0f && !p->npu_failed) {
		pthread_mutex_lock(&p->lock);
		s = rkt_ensure_capacity((void **)&p->scale_tmp, &p->scale_tmp_cap, (size_t)n, sizeof(float))
				? OK
				: ERR_OUT_OF_MEMORY;
		if (s == OK) {
			s = rkt_lut_softcap_rows(p, &p->lut, rkt_ptr(x), p->scale_tmp, n, cap);
			if (s == OK)
				memcpy(rkt_ptr(x), p->scale_tmp, (size_t)n * sizeof(float));
		}
		if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed) {
			rkt_npu_record_failure(p);
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

static int gate_act_chainable(const rkt_priv *p, const buffer *wbuf, int n, int k,
							  const rkt_weight **dw, int *row0) {
	const rkt_weight *w = wbuf->handle;
	if (p->npu_failed || !w || w->magic != RKT_WEIGHT_MAGIC || w->k != k || w->host_copy)
		return 0;
	if (wbuf->offset % w->src_row_bytes != 0 || (int)(wbuf->offset / w->src_row_bytes) + n > w->n)
		return 0;
	if (w->n_kchunks == 0 || w->fmt != RKT_FMT_F16 || (w->quant && !p->quant_failed))
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
	rkt_lut_kind	  kind	= rkt_lut_kind_for_activation(activation);
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
	if (rkt_native_gate_act(p, gw, grow0, uw, urow0, rkt_ptr(x), act, n, m, kind) == OK)
		return OK;
	pthread_mutex_lock(&p->lock);
	status_code s = rkt_lut_gate_act_prepare(p, &p->lut, kind);
	if (s == OK) {
		p->epi_kind = (int)kind;
		s			= rkt_dense_matmul_enqueue(p, gw, grow0, n, rkt_ptr(x), act, m);
		p->epi_kind = 0;
		if (s == OK)
			s = rkt_dense_matmul_enqueue(p, uw, urow0, n, rkt_ptr(x), up, m);
		if (s == OK)
			s = rkt_queue_flush(p);
		else
			rkt_queue_abort(p);
	}
	if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed) {
		rkt_npu_record_failure(p);
		WARN("rocket: npu gate+%s matmul failed (n=%d k=%d m=%d status=%d, errno=%d)",
			 rkt_lut_name(kind), n, k, m, s, errno);
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
	status_code s = rkt_lut_gate_act_prepare(p, &p->lut, RKT_LUT_TANH);
	if (s == OK) {
		p->epi_kind = RKT_LUT_TANH;
		p->epi_cap	= cap;
		s			= rkt_dense_matmul(p, dw, row0, n, rkt_ptr(x), rkt_ptr(y), m);
		p->epi_kind = 0;
		p->epi_cap	= 0.0f;
	}
	if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed) {
		rkt_npu_record_failure(p);
		WARN("rocket: npu matmul+softcap failed (n=%d k=%d m=%d status=%d, errno=%d)", n, k, m, s,
			 errno);
	}
	pthread_mutex_unlock(&p->lock);
	if (s == OK)
		return OK;
	return rkt_matmul_softcap_fallback(self, wbuf, w_type, x, y, n, k, m, cap);
}

static status_code rkt_add_scaled(backend *self, buffer *x, const buffer *y, int n, int m,
								  float out_scale) {
	rkt_priv *p = self->priv;
	if (n <= 0 || m <= 0)
		return OK;
	status_code s = ERR_UNSUPPORTED;
	if (!p->npu_failed) {
		pthread_mutex_lock(&p->lock);
		s = rkt_lut_add_inplace(p, &p->lut, rkt_ptr(x), rkt_ptr(y), n, m, out_scale);
		if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed) {
			rkt_npu_record_failure(p);
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

static status_code rkt_ffn_activate_kind(backend *self, const buffer *gate, const buffer *up,
										 buffer *out, int n, int activation, int m, int gate_stride,
										 int up_stride, int fused) {
	rkt_priv *p = self->priv;
	if (n <= 0 || m <= 0)
		return OK;
	rkt_lut_kind kind = rkt_lut_kind_for_activation(activation);
	if (kind == RKT_LUT_NONE || p->npu_failed)
		return ffn_activate_host(self, gate, up, out, n, activation, m, gate_stride, up_stride,
								 fused);

	const float *gf = rkt_ptr(gate);
	const float *uf = rkt_ptr(up);
	float		*of = rkt_ptr(out);
	if (rkt_native_ffn_activate(p, kind, gf, gate_stride, uf, up_stride, of, n, m) == OK)
		return OK;
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
	status_code s =
		rkt_lut_activate_rows(p, &p->lut, kind, gf, gate_stride, uf, u_stride, of, n, m);
	if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed) {
		rkt_npu_record_failure(p);
		WARN("rocket: npu ffn_activate failed (n=%d m=%d status=%d, errno=%d)", n, m, s, errno);
	}
	pthread_mutex_unlock(&p->lock);
	free(ucopy);
	if (s != OK)
		return ffn_activate_host(self, gate, up, out, n, activation, m, gate_stride, up_stride,
								 fused);
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
	if (!rkt_ensure_capacity((void **)&p->ffn_act_tmp, &p->ffn_act_tmp_cap, (size_t)m * k,
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

static status_code rkt_scale_inplace(backend *self, buffer *x, float scale, int n) {
	rkt_priv *p = self->priv;
	if (n <= 0 || scale == 1.0f)
		return OK;
	status_code s = ERR_UNSUPPORTED;
	if (!p->npu_failed) {
		pthread_mutex_lock(&p->lock);
		s = rkt_lut_scale_inplace(p, &p->lut, rkt_ptr(x), n, scale);
		if (s != OK && s != ERR_UNSUPPORTED && !p->npu_failed) {
			rkt_npu_record_failure(p);
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
		if (sl->hd < 1 || sl->kvh < 1) {
			sl->hd	   = 0;
			sl->kvh	   = 0;
			sl->hd_pad = 0;
			continue;
		}
		sl->hd_pad = rkt_align_up(sl->hd, RKT_K_ALIGN);
		if (sl->hd_pad > RKT_MAX_K || sl->pos_pad > RKT_MAX_K)
			s = ERR_UNSUPPORTED;
		else {
			sl->k = kv_weight_build(p, sl->kvh * sl->pos_pad, sl->hd, sl->hd_pad);
			sl->v = kv_weight_build(p, sl->kvh * sl->hd_pad, sl->pos_pad, sl->pos_pad);
			if (!sl->k || !sl->v)
				s = ERR_OUT_OF_MEMORY;
		}
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
	if (!rkt_ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)kc->k_len, sizeof(uint16_t)))
		s = ERR_OUT_OF_MEMORY;
	row = (uint16_t *)p->x16;
	for (int r = 0; s == OK && r < m; r++) {
		int pos = pos_start + r;
		for (int h = 0; s == OK && h < n_active; h++) {
			const float *krow = kf + (size_t)r * in_row_stride + (size_t)h * head_dim;
			const float *vrow = vf + (size_t)r * in_row_stride + (size_t)h * head_dim;
			rkt_convert_to_f16(row, krow, (size_t)head_dim);
			memset(row + head_dim, 0, (size_t)(kc->k_len - head_dim) * sizeof(uint16_t));
			int				k_global = h * sl->pos_pad + pos;
			const rkt_part *pt		 = rkt_part_for_row(kc, k_global);
			if (!pt) {
				s = ERR_INTERNAL;
				break;
			}
			rkt_tile_weight_row(pt->bo.map, row, kc->k_len, k_global - pt->row_start);
			int				v_col = pos / RKT_TILE_COLS, v_rem = pos % RKT_TILE_COLS;
			const rkt_part *vp = NULL;
			for (int d = 0; d < head_dim; d++) {
				int v_global = h * sl->hd_pad + d;
				if (!vp || v_global < vp->row_start || v_global >= vp->row_start + vp->row_count)
					vp = rkt_part_for_row(vc, v_global);
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

static float softmax_scaled_max(const float *sc, int lo, int hi, float scale) {
	float		mx = -INFINITY;
	int			t  = lo;
	float32x4_t vm = vdupq_n_f32(-INFINITY);
	for (; t + 4 <= hi; t += 4)
		vm = vmaxq_f32(vm, vmulq_n_f32(vld1q_f32(sc + t), scale));
	mx = vmaxvq_f32(vm);
	for (; t < hi; t++) {
		float v = sc[t] * scale;
		if (v > mx)
			mx = v;
	}
	return mx;
}

static double softmax_exp_sum(const float *sc, float *pr, int lo, int hi, float scale, float mx) {
	double		sum = 0.0;
	int			t	= lo;
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
	for (; t < hi; t++) {
		float e = expf(sc[t] * scale - mx);
		pr[t]	= e;
		sum += e;
	}
	return sum;
}

static void softmax_round_f16(float *pr, int n, float inv) {
	int t = 0;
	for (; t + 4 <= n; t += 4)
		vst1q_f32(pr + t, vcvt_f32_f16(vcvt_f16_f32(vmulq_n_f32(vld1q_f32(pr + t), inv))));
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

typedef struct {
	rkt_priv		  *p;
	const rkt_kv_slot *sl;
	const float		  *qf;
	float			  *outf;
	size_t			   group_floats;
	float			   scale;
	int				   n_active;
	int				   n_heads;
	int				   n_groups;
	int				   m;
	int				   rows;
	int				   span;
	int				   span16;
	int				   pos_start;
	int				   window;
	int				   head_dim;
} attn_job;

typedef struct {
	float *q;
	float *out;
	float *scores;
	float *probs;
} attn_head_buf;

static attn_head_buf attn_head(const attn_job *j, float *ws, int i) {
	float *q	  = ws + (size_t)i * j->group_floats;
	float *scores = q + (size_t)2 * j->rows * j->head_dim;
	return (attn_head_buf){.q	   = q,
						   .out	   = q + (size_t)j->rows * j->head_dim,
						   .scores = scores,
						   .probs  = scores + (size_t)j->rows * j->span16};
}

static status_code attn_enqueue_scores(const attn_job *j, float *ws, int g0, int count) {
	for (int i = 0; i < count; i++) {
		int			  g = g0 + i;
		attn_head_buf b = attn_head(j, ws, i);
		for (int row = 0; row < j->rows; row++)
			memset(b.probs + (size_t)row * j->sl->pos_pad, 0, (size_t)j->span * sizeof(float));
		for (int r = 0; r < j->m; r++)
			for (int h = 0; h < j->n_groups; h++)
				memcpy(b.q + (size_t)(r * j->n_groups + h) * j->head_dim,
					   j->qf + ((size_t)r * j->n_heads + (size_t)g * j->n_groups + h) * j->head_dim,
					   (size_t)j->head_dim * sizeof(float));
		status_code s = rkt_dense_matmul_enqueue(j->p, j->sl->k, g * j->sl->pos_pad, j->span16, b.q,
												 b.scores, j->rows);
		if (s != OK)
			return s;
	}
	return OK;
}

static status_code attn_enqueue_values(const attn_job *j, float *ws, int g0, int count) {
	for (int i = 0; i < count; i++) {
		attn_head_buf b = attn_head(j, ws, i);
		attention_softmax(b.scores, b.probs, j->m, j->n_groups, j->span16, j->sl->pos_pad,
						  j->pos_start, j->window, j->scale);
		status_code s = rkt_dense_matmul_enqueue_k(j->p, j->sl->v, (g0 + i) * j->sl->hd_pad,
												   j->head_dim, b.probs, b.out, j->rows, j->span);
		if (s != OK)
			return s;
	}
	return OK;
}

static void attn_copy_out(const attn_job *j, float *ws, int g0, int count) {
	for (int i = 0; i < count; i++) {
		attn_head_buf b = attn_head(j, ws, i);
		for (int r = 0; r < j->m; r++)
			for (int h = 0; h < j->n_groups; h++)
				memcpy(j->outf + ((size_t)r * j->n_heads + (size_t)(g0 + i) * j->n_groups + h) *
									 j->head_dim,
					   b.out + (size_t)(r * j->n_groups + h) * j->head_dim,
					   (size_t)j->head_dim * sizeof(float));
	}
}

static int attn_group_count(const attn_job *j, int g0, int batch) {
	return j->n_active - g0 < batch ? j->n_active - g0 : batch;
}

static status_code attn_run_serial(const attn_job *j, float *ws, int batch) {
	for (int g0 = 0; g0 < j->n_active; g0 += batch) {
		int			count = attn_group_count(j, g0, batch);
		status_code s	  = attn_enqueue_scores(j, ws, g0, count);
		if (s == OK)
			s = rkt_queue_flush(j->p);
		if (s == OK)
			s = attn_enqueue_values(j, ws, g0, count);
		if (s == OK)
			s = rkt_queue_flush(j->p);
		if (s != OK)
			return s;
		attn_copy_out(j, ws, g0, count);
	}
	return OK;
}

static status_code attn_run_pipelined(const attn_job *j, int batch) {
	rkt_priv   *p	  = j->p;
	float	   *ws[2] = {p->attn_ws, p->attn_ws + j->group_floats * (size_t)batch};
	status_code s	  = attn_enqueue_scores(j, ws[0], 0, attn_group_count(j, 0, batch));
	if (s == OK)
		s = rkt_queue_flush_async(p);
	for (int g0 = 0, turn = 0; s == OK && g0 < j->n_active; g0 += batch, turn ^= 1) {
		int	   count   = attn_group_count(j, g0, batch);
		int	   next_g0 = g0 + batch;
		float *current = ws[turn];
		if (next_g0 < j->n_active) {
			s = attn_enqueue_scores(j, ws[turn ^ 1], next_g0, attn_group_count(j, next_g0, batch));
			if (s == OK)
				s = rkt_queue_flush_async(p);
		} else {
			s = rkt_queue_sync(p);
		}
		if (s == OK)
			s = attn_enqueue_values(j, current, g0, count);
		if (s == OK)
			s = rkt_queue_flush_async(p);
		if (s == OK)
			s = rkt_queue_sync(p);
		if (s == OK)
			attn_copy_out(j, current, g0, count);
	}
	return s;
}

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
	int pipelined		= n_active > batch;
	if (!rkt_ensure_capacity((void **)&p->attn_ws, &p->attn_ws_cap,
							 group_floats * batch * (pipelined ? 2 : 1), sizeof(float)))
		return ERR_OUT_OF_MEMORY;

	attn_job job = {.p			  = p,
					.sl			  = sl,
					.qf			  = rkt_ptr(q),
					.outf		  = rkt_ptr(out),
					.group_floats = group_floats,
					.scale		  = scale,
					.n_active	  = n_active,
					.n_heads	  = n_heads,
					.n_groups	  = n_groups,
					.m			  = m,
					.rows		  = M,
					.span		  = span,
					.span16		  = span16,
					.pos_start	  = pos_start,
					.window		  = window,
					.head_dim	  = head_dim};

	pthread_mutex_lock(&p->lock);
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL
					: pipelined	  ? attn_run_pipelined(&job, batch)
								  : attn_run_serial(&job, p->attn_ws, batch);
	if (s != OK) {
		rkt_queue_abort(p);
		if (!p->npu_failed) {
			rkt_npu_record_failure(p);
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
	const rkt_kv_slot *slot = kv_slot_of(k_cache, layer);
	if ((!slot || slot->kvh < 1 || slot->hd < 1) && k_cache->owner && k_cache->owner != self) {
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
	return rkt_attention_batch_inner(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
									 n_kv_heads, head_dim, n_ctx, flash_attn, scale,
									 n_kv_heads_active, m, window);
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

static size_t rkt_mem_available(backend *self) {
	(void)self;
	return backend_mem_available(backend_host());
}

static size_t rkt_mem_total(backend *self) {
	(void)self;
	return backend_mem_total(backend_host());
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

static const char *weight_pipeline_name(const rkt_weight *w) {
	if (w->quant)
		return w->quant->bits == 4 ? "int4" : "int8";
	if (w->host_copy)
		return "host";
	return rkt_fmt_name(w->fmt);
}

void rkt_log_pipeline_usage(rkt_priv *p) {
	if (p->pipelines_logged)
		return;
	p->pipelines_logged		   = 1;
	static const char *names[] = {"f16", "bf16", "tf32", "int8", "int4", "host"};
	int				   counts[sizeof(names) / sizeof(names[0])] = {0};
	for (size_t i = 0; i < p->n_weights; i++) {
		const char *name = weight_pipeline_name(p->weights[i]);
		for (size_t j = 0; j < sizeof(names) / sizeof(names[0]); j++)
			counts[j] += strcmp(names[j], name) == 0;
	}
	for (size_t j = 0; j < sizeof(names) / sizeof(names[0]); j++)
		if (counts[j])
			DEBUG("rocket: pipeline %s: %d weights", names[j], counts[j]);
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
	rkt_log_pipeline_usage(p);
	free_slots(p);
	rkt_lut_state_reset(p, &p->lut);
	rkt_device_close(&p->dev);
	pthread_mutex_destroy(&p->lock);
	free(p->weights);
	rkt_native_free(p);
	free(p->x16);
	free(p->y16);
	free(p->res16);
	free(p->x_chunk);
	free(p->y_tmp);
	free(p->quant_pack[0]);
	free(p->quant_pack[1]);
	free(p->quant_s_act);
	free(p->scale_tmp);
	free(p->attn_ws);
	free(p->ffn_act_tmp);
	free(p);
	self->priv = NULL;
}

static status_code rkt_init(backend *self, int device_index) {
	(void)device_index;
	rkt_priv *p	   = xcalloc(1, sizeof(*p));
	p->bf16_native = env_int("KAPPAI_ROCKET_BF16", 0, 0, 1);
	rkt_emit_set_bf16_dpu_precision(
		(uint32_t)env_int("KAPPAI_ROCKET_BF16_DPU", PREC_BFLOAT16, 2, 5));
	pthread_mutex_init(&p->lock, NULL);
	self->priv = p;

	int n_cores = env_int("KAPPAI_ROCKET_NCORES", RKT_MAX_CORES, 1, RKT_MAX_CORES);
	int err		= rkt_device_open(&p->dev, device_path(), n_cores, RKT_TIMEOUT_NS);
	if (err) {
		WARN("rocket: cannot open %s: %s", device_path(), strerror(-err));
		rkt_free(self);
		return ERR_UNSUPPORTED;
	}
	log_tag("RKT", "%s: %d core(s)", device_path(), p->dev.n_cores);
	return OK;
}

static status_code rkt_ctor(backend *out) {
	memset(out, 0, sizeof(*out));
	out->name	  = "rocket";
	out->desc	  = "Rockchip RK3588 NPU (rocket)";
	out->priority = RKT_PRIORITY;
	out->caps =
		BCAP_HOST_VISIBLE_BUFFERS | BCAP_UNIFIED_MEMORY | BCAP_MULTI_MATMUL | BCAP_MATMUL_RESIDUAL;
	out->probe					  = rkt_probe;
	out->init					  = rkt_init;
	out->free					  = rkt_free;
	out->mem_available			  = rkt_mem_available;
	out->mem_total				  = rkt_mem_total;
	out->matmul_type_native		  = rkt_matmul_type_native;
	out->buffer_alloc_weight	  = rkt_buffer_alloc_weight;
	out->buffer_alloc_scratch	  = rkt_buffer_alloc_scratch;
	out->buffer_free			  = rkt_buffer_free;
	out->buffer_read_f32		  = rkt_buffer_read_f32;
	out->buffer_write_f32		  = rkt_buffer_write_f32;
	out->synchronize			  = rkt_synchronize;
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
