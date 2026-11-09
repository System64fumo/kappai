#include "rocket.h"

#define RKT_LUT_HQ_SEL 4
#define RKT_ADD_MAX_SCALED 65000.0f
#define RKT_LUT_HOST_LIMIT 15.5f

const char *rkt_lut_name(rkt_lut_kind kind) {
	if (kind == RKT_LUT_TANH)
		return "tanh";
	return kind == RKT_LUT_GELU_TANH ? "gelu" : "silu";
}

rkt_lut_kind rkt_lut_kind_for_activation(int activation) {
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

void rkt_lut_configure(rkt_lut_cfg *cfg, rkt_lut_kind kind) {
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
		rkt_bo_destroy(&p->dev, &t->reg_mul[c]);
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
						  &t->reg_blocks[c]) ||
			rkt_bo_create(&p->dev, c, (size_t)blocks_per_core * RKT_REGCMD_BYTES, &t->reg_mul[c]))
			return ERR_OUT_OF_MEMORY;
	return OK;
}

void rkt_lut_state_reset(rkt_priv *p, rkt_lut_state *a) {
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		rkt_bo_destroy(&p->dev, &a->identity[c]);
		rkt_bo_destroy(&p->dev, &a->identity_unit[c]);
	}
	for (int t = 0; t < RKT_LUT_TIERS; t++)
		lut_tier_destroy(p, &a->tier[t]);
	a->ready	= 0;
	a->resident = RKT_LUT_NONE;
}

status_code rkt_lut_state_init(rkt_priv *p, rkt_lut_state *a) {
	if (a->ready)
		return OK;
	rkt_lut_configure(&a->cfg, RKT_LUT_NONE);
	size_t	 identity_bytes = (size_t)RKT_LUT_CHUNK * RKT_LUT_CHUNK * sizeof(uint16_t);
	uint16_t row[RKT_LUT_CHUNK];
	for (int c = 0; c < p->dev.n_cores; c++) {
		if (rkt_bo_create(&p->dev, c, identity_bytes, &a->identity[c])) {
			rkt_lut_state_reset(p, a);
			return ERR_OUT_OF_MEMORY;
		}
		for (int j = 0; j < RKT_LUT_CHUNK; j++) {
			memset(row, 0, sizeof(row));
			row[j] = f32_to_f16_bits((float)a->cfg.S);
			rkt_tile_weight_row(a->identity[c].map, row, RKT_LUT_CHUNK, j);
		}
		if (rkt_bo_flush(&p->dev, &a->identity[c])) {
			rkt_lut_state_reset(p, a);
			return ERR_IO;
		}
		if (rkt_bo_create(&p->dev, c, identity_bytes, &a->identity_unit[c])) {
			rkt_lut_state_reset(p, a);
			return ERR_OUT_OF_MEMORY;
		}
		for (int j = 0; j < RKT_LUT_CHUNK; j++) {
			memset(row, 0, sizeof(row));
			row[j] = f32_to_f16_bits(1.0f);
			rkt_tile_weight_row(a->identity_unit[c].map, row, RKT_LUT_CHUNK, j);
		}
		if (rkt_bo_flush(&p->dev, &a->identity_unit[c])) {
			rkt_lut_state_reset(p, a);
			return ERR_IO;
		}
	}
	if (lut_tier_create(p, &a->tier[0], RKT_LUT_SMALL_BLOCKS_PER_CORE) != OK ||
		lut_tier_create(p, &a->tier[1], RKT_LUT_BLOCKS_PER_CORE) != OK) {
		rkt_lut_state_reset(p, a);
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
	status_code s = rkt_lut_state_init(p, a);
	if (s != OK)
		return s;
	rkt_lut_configure(&a->cfg, kind);
	a->post_active = 0;
	return OK;
}

typedef struct {
	int first;
	int rows;
} lut_block;

static status_code lut_run_wave(rkt_priv *p, rkt_lut_state *a, rkt_lut_tier *t, const uint16_t *x16,
								const lut_block *blocks, int n_blocks) {
	int n_upload  = a->post_active || (a->cfg.kind != RKT_LUT_NONE && a->resident == a->cfg.kind)
						? 0
						: RKT_LUT_UPLOAD_TASKS;
	status_code s = n_upload ? lut_tier_prepare(p, a, t) : OK;
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
		rkt_tile_input(
			(uint16_t *)((uint8_t *)t->in[c].map + (size_t)slot * RKT_LUT_IN_BLOCK_BYTES),
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
			rkt_pack_ew_operand((uint16_t *)((uint8_t *)t->ew[c].map + ew_slot),
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
	const uint16_t	   *up16;
	uint16_t		   *out16;
	int					out_stride;
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
			const uint16_t *u16	  = g->up16 ? g->up16 + (size_t)row * g->up_stride + col0 : NULL;
			float		   *dst	  = g->out ? g->out + (size_t)row * g->n + col0 : NULL;
			uint16_t	   *dst16 = g->out16 ? g->out16 + (size_t)row * g->out_stride + col0 : NULL;
			size_t			group_stride = (size_t)b->rows * RKT_OUT_C2_F32;
			const float	   *om_row		 = om + (size_t)r * RKT_OUT_C2_F32;
			int				j			 = 0;
			for (; j + RKT_OUT_C2_F32 <= cols; j += RKT_OUT_C2_F32) {
				float32x4_t v = vmulq_n_f32(
					vld1q_f32(om_row + (size_t)(j / RKT_OUT_C2_F32) * group_stride), g->inv);
				if (g->kind != RKT_LUT_NONE) {
					float32x4_t xv	   = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x + j)));
					uint32x4_t	beyond = vcageq_f32(xv, vdupq_n_f32(RKT_LUT_HOST_LIMIT));
					if (vmaxvq_u32(beyond)) {
						float vs[RKT_OUT_C2_F32], xs[RKT_OUT_C2_F32];
						vst1q_f32(vs, v);
						vst1q_f32(xs, xv);
						for (int l = 0; l < RKT_OUT_C2_F32; l++)
							if (fabsf(xs[l]) >= RKT_LUT_HOST_LIMIT)
								vs[l] = (float)lut_eval(g->kind, (double)xs[l]);
						v = vld1q_f32(vs);
					}
				}
				if (u)
					v = vmulq_f32(v, vld1q_f32(u + j));
				else if (u16)
					v = vmulq_f32(v, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(u16 + j))));
				v = vmulq_n_f32(v, g->out_scale);
				if (dst16)
					vst1_u16(dst16 + j, vreinterpret_u16_f16(vcvt_f16_f32(v)));
				else
					vst1q_f32(dst + j, v);
			}
			for (; j < cols; j++) {
				float v = om_row[(size_t)(j / RKT_OUT_C2_F32) * group_stride + j % RKT_OUT_C2_F32] *
						  g->inv;
				if (g->kind != RKT_LUT_NONE) {
					float xv = f16_bits_to_f32(x[j]);
					if (fabsf(xv) >= RKT_LUT_HOST_LIMIT)
						v = (float)lut_eval(g->kind, (double)xv);
				}
				float up_j	 = u ? u[j] : u16 ? f16_bits_to_f32(u16[j]) : 1.0f;
				float result = v * up_j * g->out_scale;
				if (dst16)
					dst16[j] = f32_to_f16_bits(result);
				else
					dst[j] = result;
			}
		}
	}
}

static status_code lut_run_blocks(rkt_priv *p, rkt_lut_state *a, const uint16_t *x16, int pixels,
								  int chunks_per_row, const lut_gather *proto) {
	status_code s = OK;
	lut_block	blocks[RKT_MAX_CORES * RKT_LUT_BLOCKS_PER_CORE];
	int			wave_capacity = p->dev.n_cores * RKT_LUT_BLOCKS_PER_CORE;
	tpool	   *pool		  = tpool_fetch_pool();
	for (int done = 0; done < pixels;) {
		int n_blocks = 0;
		while (n_blocks < wave_capacity && done < pixels) {
			int rows		   = rkt_next_block_rows(pixels - done, RKT_LUT_BLOCK_ROWS);
			blocks[n_blocks++] = (lut_block){.first = done, .rows = rows};
			done += rows;
		}
		rkt_lut_tier *tier =
			n_blocks <= p->dev.n_cores * RKT_LUT_SMALL_BLOCKS_PER_CORE ? &a->tier[0] : &a->tier[1];
		s = lut_run_wave(p, a, tier, x16, blocks, n_blocks);
		if (s != OK)
			return s;
		lut_gather g	 = *proto;
		g.tier			 = tier;
		g.blocks		 = blocks;
		g.x16			 = x16;
		g.chunks_per_row = chunks_per_row;
		g.n_cores		 = p->dev.n_cores;
		g.out_scale		 = a->out_scale > 0.0f ? a->out_scale : 1.0f;
		tpool_parallel_for(pool, n_blocks, 4, lut_gather_chunk, &g);
	}
	return OK;
}

static status_code lut_run_rows(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind, const float *gate,
								int gate_stride, const float *up, int up_stride, float *out, int n,
								int m, float inv) {
	status_code s			   = OK;
	int			chunks_per_row = (n + RKT_LUT_CHUNK - 1) / RKT_LUT_CHUNK;
	int			pixels		   = m * chunks_per_row;
	if (!rkt_ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)pixels * RKT_LUT_CHUNK,
							 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	for (int r = 0; r < m; r++) {
		uint16_t	*dst = p->x16 + (size_t)r * chunks_per_row * RKT_LUT_CHUNK;
		const float *src = gate + (size_t)r * gate_stride;
		if (a->in_scale > 0.0f)
			for (int j = 0; j < n; j++)
				dst[j] = f32_to_f16_bits(src[j] * a->in_scale);
		else
			rkt_convert_to_f16(dst, src, (size_t)n);
		memset(dst + n, 0, ((size_t)chunks_per_row * RKT_LUT_CHUNK - (size_t)n) * sizeof(uint16_t));
	}

	if (a->ew_src) {
		if (!rkt_ensure_capacity((void **)&p->y16, &p->y16_cap, (size_t)pixels * RKT_LUT_CHUNK,
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

	lut_gather proto = {.up			= up,
						.out		= out,
						.out16		= a->out16,
						.out_stride = a->out16_stride,
						.up_stride	= up_stride,
						.n			= n,
						.kind		= kind,
						.inv		= inv};
	return lut_run_blocks(p, a, p->x16, pixels, chunks_per_row, &proto);
}

status_code rkt_lut_activate_rows(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind,
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
	status_code s = rkt_lut_state_init(p, a);
	if (s != OK)
		return s;
	int	  exponent;
	float mantissa = frexpf(scale, &exponent);
	rkt_lut_configure(&a->cfg, RKT_LUT_NONE);
	a->post_active = 1;
	a->post		   = (rkt_post_cfg){.act = RKT_ACT_MULC, .mulc = 2.0f * mantissa};
	return lut_run_rows(p, a, RKT_LUT_NONE, x, n, NULL, 0, out, n, 1,
						ldexpf(1.0f, exponent - 1) / (float)a->cfg.S);
}

status_code rkt_lut_softcap_rows(rkt_priv *p, rkt_lut_state *a, const float *x, float *out, int n,
								 float cap) {
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

static status_code lut_make_resident(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind) {
	status_code s = lut_state_select(p, a, kind);
	if (s != OK || a->resident == kind)
		return s;
	return lut_run_wave(p, a, &a->tier[0], NULL, NULL, 0);
}

status_code rkt_lut_gate_act_prepare(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind) {
	return lut_make_resident(p, a, kind);
}

static status_code lut_add_rows(rkt_priv *p, rkt_lut_state *a, const float *x, const float *y,
								float *out, int n, int m, float out_scale) {
	status_code s = rkt_lut_state_init(p, a);
	if (s != OK)
		return s;
	rkt_lut_configure(&a->cfg, RKT_LUT_NONE);
	a->post_active = 1;
	a->post		   = (rkt_post_cfg){.act = RKT_ACT_NONE};
	a->ew_src	   = y;
	s			   = lut_run_rows(p, a, RKT_LUT_NONE, x, n, NULL, 0, out, n, m, out_scale);
	a->ew_src	   = NULL;
	a->ew16		   = NULL;
	return s;
}

status_code rkt_lut_add_inplace(rkt_priv *p, rkt_lut_state *a, float *x, const float *y, int n,
								int m, float out_scale) {
	size_t total = (size_t)n * (size_t)m;
	if (!rkt_ensure_capacity((void **)&p->scale_tmp, &p->scale_tmp_cap, total, sizeof(float)))
		return ERR_OUT_OF_MEMORY;
	status_code s = lut_add_rows(p, a, x, y, p->scale_tmp, n, m, out_scale);
	if (s == OK)
		memcpy(x, p->scale_tmp, total * sizeof(float));
	return s;
}

status_code rkt_lut_scale_inplace(rkt_priv *p, rkt_lut_state *a, float *x, int n, float scale) {
	if (!rkt_ensure_capacity((void **)&p->scale_tmp, &p->scale_tmp_cap, (size_t)n, sizeof(float)))
		return ERR_OUT_OF_MEMORY;
	status_code s = lut_scale_rows(p, a, x, p->scale_tmp, n, scale);
	if (s == OK)
		memcpy(x, p->scale_tmp, (size_t)n * sizeof(float));
	return s;
}

status_code rkt_lut_activate_rows_f16(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind,
									  const float *gate, int gate_stride, const float *up,
									  int up_stride, uint16_t *out16, int out_stride, int n,
									  int m) {
	status_code s = lut_state_select(p, a, kind);
	if (s != OK)
		return s;
	a->out16		= out16;
	a->out16_stride = out_stride;
	s				= lut_run_rows(p, a, kind, gate, gate_stride, up, up_stride, NULL, n, m,
								   1.0f / (float)a->cfg.F);
	a->out16		= NULL;
	return s;
}

#define RKT_MUL_BLOCKS_PER_CORE 28
#define RKT_MUL_HALF_BYTES ((size_t)RKT_LUT_BLOCK_ROWS * RKT_LUT_CHUNK * sizeof(uint16_t))

static void mul_gather(const rkt_lut_tier *t, const lut_block *blocks, int n_blocks, int n_cores,
					   uint16_t *out16, int out_stride, int n, int chunks_per_row) {
	for (int i = 0; i < n_blocks; i++) {
		const lut_block *b	 = &blocks[i];
		const uint16_t	*fin = (const uint16_t *)((const uint8_t *)t->out[i % n_cores].map +
												  (size_t)(i / n_cores) * RKT_LUT_OUT_BLOCK_BYTES +
												  RKT_MUL_HALF_BYTES);
		for (int r = 0; r < b->rows; r++) {
			int		  pixel = b->first + r;
			int		  row	= pixel / chunks_per_row;
			int		  col0	= (pixel % chunks_per_row) * RKT_LUT_CHUNK;
			int		  cols	= n - col0 < RKT_LUT_CHUNK ? n - col0 : RKT_LUT_CHUNK;
			uint16_t *dst	= out16 + (size_t)row * out_stride + col0;
			for (int j = 0; j < cols; j += 8) {
				const uint16_t *src = fin + (size_t)(j / 8) * b->rows * 8 + (size_t)r * 8;
				int				cnt = cols - j < 8 ? cols - j : 8;
				memcpy(dst + j, src, (size_t)cnt * sizeof(uint16_t));
			}
		}
	}
}

static status_code lut_mul_wave(rkt_priv *p, rkt_lut_state *a, rkt_lut_tier *t,
								const uint16_t *gate16, const uint16_t *up16,
								const lut_block *blocks, int n_blocks) {
	int			n_cores	 = p->dev.n_cores;
	int			n_upload = a->resident == a->cfg.kind ? 0 : RKT_LUT_UPLOAD_TASKS;
	status_code s		 = n_upload ? lut_tier_prepare(p, a, t) : OK;
	if (s != OK)
		return s;
	int		 per_core[RKT_MAX_CORES] = {0};
	uint32_t n_ops[RKT_MAX_CORES][RKT_MAX_TASKS];
	uint32_t dma[RKT_MAX_CORES][RKT_MAX_TASKS];
	uint32_t ops_b[RKT_MAX_CORES][RKT_MUL_BLOCKS_PER_CORE];
	for (int c = 0; c < n_cores; c++)
		for (int task = 0; task < n_upload; task++) {
			n_ops[c][task] = a->upload_ops[task];
			dma[c][task] = (uint32_t)(t->reg_upload[c].dma_addr + (size_t)task * RKT_REGCMD_BYTES);
		}
	float mulc = 1.0f / (float)a->cfg.F;
	for (int i = 0; i < n_blocks; i++) {
		const lut_block *b		 = &blocks[i];
		int				 c		 = i % n_cores;
		int				 slot	 = i / n_cores;
		size_t			 in_off	 = (size_t)slot * RKT_LUT_IN_BLOCK_BYTES;
		size_t			 out_off = (size_t)slot * RKT_LUT_OUT_BLOCK_BYTES;
		size_t			 ew_off	 = (size_t)slot * RKT_EW_OPERAND_BLOCK_BYTES;
		size_t			 reg_off = (size_t)slot * RKT_REGCMD_BYTES;
		rkt_tile_input((uint16_t *)((uint8_t *)t->in[c].map + in_off),
					   gate16 + (size_t)b->first * RKT_LUT_CHUNK, b->rows, RKT_LUT_CHUNK);
		rkt_pack_ew_operand((uint16_t *)((uint8_t *)t->ew[c].map + ew_off),
							up16 + (size_t)b->first * RKT_LUT_CHUNK, b->rows, RKT_LUT_CHUNK);
		uint32_t mid_dma = (uint32_t)(t->out[c].dma_addr + out_off);
		uint32_t na		 = (uint32_t)rkt_emit_matmul_f16_lut_out16(
			(uint64_t *)((uint8_t *)t->reg_blocks[c].map + reg_off), b->rows, RKT_LUT_CHUNK,
			RKT_LUT_CHUNK, (uint32_t)(t->in[c].dma_addr + in_off),
			(uint32_t)a->identity[c].dma_addr, mid_dma, &a->cfg, 2);
		uint32_t nb = (uint32_t)rkt_emit_matmul_f16_ewmul_out16(
			(uint64_t *)((uint8_t *)t->reg_mul[c].map + reg_off), b->rows, RKT_LUT_CHUNK,
			RKT_LUT_CHUNK, mid_dma, (uint32_t)a->identity_unit[c].dma_addr,
			mid_dma + (uint32_t)RKT_MUL_HALF_BYTES, (uint32_t)(t->ew[c].dma_addr + ew_off), mulc,
			2);
		if (!na || !nb || na > RKT_REGCMD_MAX_OPS || nb > RKT_REGCMD_MAX_OPS)
			return ERR_INTERNAL;
		n_ops[c][n_upload + slot] = na;
		dma[c][n_upload + slot]	  = (uint32_t)(t->reg_blocks[c].dma_addr + reg_off);
		ops_b[c][slot]			  = nb;
		per_core[c]++;
	}
	for (int c = 0; c < n_cores; c++)
		for (int slot = 0; slot < per_core[c]; slot++) {
			n_ops[c][n_upload + per_core[c] + slot] = ops_b[c][slot];
			dma[c][n_upload + per_core[c] + slot] =
				(uint32_t)(t->reg_mul[c].dma_addr + (size_t)slot * RKT_REGCMD_BYTES);
		}
	for (int c = 0; c < n_cores; c++)
		if (per_core[c] &&
			(rkt_bo_flush(&p->dev, &t->in[c]) || rkt_bo_flush(&p->dev, &t->ew[c]) ||
			 rkt_bo_flush(&p->dev, &t->reg_blocks[c]) || rkt_bo_flush(&p->dev, &t->reg_mul[c])))
			return ERR_IO;
	int submitted = 0;
	for (; submitted < n_cores && (per_core[submitted] || n_upload) && s == OK; submitted++) {
		int			  c			  = submitted;
		const rkt_bo *in_list[7]  = {&t->in[c],			&a->identity[c],   &a->identity_unit[c],
									 &t->reg_upload[c], &t->reg_blocks[c], &t->reg_mul[c],
									 &t->ew[c]};
		const rkt_bo *out_list[1] = {&t->out[c]};
		if (rkt_submit_bos(&p->dev, c, dma[c], n_ops[c], n_upload + 2 * per_core[c], in_list, 7,
						   out_list, 1))
			s = ERR_COMPUTE_FAIL;
	}
	for (int c = 0; c < submitted; c++)
		if (rkt_bo_wait(&p->dev, &t->out[c]) && s == OK)
			s = ERR_COMPUTE_FAIL;
	if (s == OK && n_upload)
		a->resident = a->cfg.kind;
	return s;
}

status_code rkt_lut_activate_native_mul(rkt_priv *p, rkt_lut_state *a, rkt_lut_kind kind,
										const uint16_t *gate16, const uint16_t *up16,
										uint16_t *out16, int out_stride, int n, int m) {
	status_code s = lut_state_select(p, a, kind);
	if (s != OK)
		return s;
	int		  chunks_per_row = (n + RKT_LUT_CHUNK - 1) / RKT_LUT_CHUNK;
	int		  pixels		 = m * chunks_per_row;
	int		  wave_capacity	 = p->dev.n_cores * RKT_MUL_BLOCKS_PER_CORE;
	lut_block blocks[RKT_MAX_CORES * RKT_MUL_BLOCKS_PER_CORE];
	for (int done = 0; done < pixels;) {
		int n_blocks = 0;
		while (n_blocks < wave_capacity && done < pixels) {
			int rows		   = rkt_next_block_rows(pixels - done, RKT_LUT_BLOCK_ROWS);
			blocks[n_blocks++] = (lut_block){.first = done, .rows = rows};
			done += rows;
		}
		rkt_lut_tier *tier =
			n_blocks <= p->dev.n_cores * RKT_LUT_SMALL_BLOCKS_PER_CORE ? &a->tier[0] : &a->tier[1];
		s = lut_mul_wave(p, a, tier, gate16, up16, blocks, n_blocks);
		if (s != OK)
			return s;
		mul_gather(tier, blocks, n_blocks, p->dev.n_cores, out16, out_stride, n, chunks_per_row);
	}
	return OK;
}
