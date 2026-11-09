#include "rocket.h"

static int quant_ncols(const rkt_quant_core *wc, int nt) {
	int rest = wc->n - nt * RKT_QUANT_NTILE;
	return rest < RKT_QUANT_NTILE ? rest : RKT_QUANT_NTILE;
}

static int quant_ntiles(const rkt_quant_core *wc) {
	return (wc->n + RKT_QUANT_NTILE - 1) / RKT_QUANT_NTILE;
}

static int quant_tile_len(const rkt_weight *w, int group, int kt) {
	int rest = w->k_pad - kt * group;
	return rest < group ? rest : group;
}

static int quant_emit_regs(const rkt_quant *q, const rkt_weight *w, rkt_quant_core *wc,
						   uint8_t *map, uint32_t *n_ops, int m, int tokens, uint32_t in_dma,
						   uint32_t out_dma) {
	for (int t = 0; t < wc->n_tasks; t++) {
		int kt = t % q->n_ktiles, nt = t / q->n_ktiles;
		n_ops[t] =
			(uint32_t)rkt_emit_quant((uint64_t *)(map + (size_t)t * RKT_REGCMD_BYTES), q->bits, m,
									 quant_tile_len(w, q->group, kt), quant_ncols(wc, nt),
									 (uint32_t)(in_dma + (size_t)kt * q->group * tokens),
									 (uint32_t)(wc->wt.dma_addr + wc->w_off[t]),
									 (uint32_t)(out_dma + wc->out_off[t] * tokens));
		if (n_ops[t] > RKT_REGCMD_MAX_OPS)
			return 0;
	}
	return 1;
}

void rkt_quant_free(rkt_priv *p, rkt_quant *q) {
	if (!q)
		return;
	for (int c = 0; c < RKT_MAX_CORES; c++) {
		rkt_bo_destroy(&p->dev, &q->core[c].in);
		rkt_bo_destroy(&p->dev, &q->core[c].out);
		rkt_bo_destroy(&p->dev, &q->core[c].wt);
		rkt_bo_destroy(&p->dev, &q->core[c].reg);
		rkt_bo_destroy(&p->dev, &q->core[c].pf_reg);
		free(q->core[c].w_off);
		free(q->core[c].out_off);
		free(q->core[c].n_ops);
		free(q->core[c].pf_n_ops);
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
		int n_ntiles = quant_ntiles(wc);
		wc->n_tasks	 = n_ntiles * q->n_ktiles;
		wc->w_off	 = xcalloc((size_t)wc->n_tasks, sizeof(size_t));
		wc->out_off	 = xcalloc((size_t)wc->n_tasks, sizeof(size_t));
		wc->n_ops	 = xcalloc((size_t)wc->n_tasks, sizeof(uint32_t));
		size_t w_acc = 0, o_acc = 0;
		for (int nt = 0; nt < n_ntiles; nt++) {
			int ncols = quant_ncols(wc, nt);
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

static void quant_pack_nibbles(uint8_t *dst, int8x16_t codes) {
	uint8x16_t u	= vreinterpretq_u8_s8(codes);
	uint8x8_t  even = vget_low_u8(vuzp1q_u8(u, u));
	uint8x8_t  odd	= vget_low_u8(vuzp2q_u8(u, u));
	vst1_u8(dst, vorr_u8(vshl_n_u8(even, 4), vand_u8(odd, vdup_n_u8(15))));
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
	quant_job		  *job		 = arg;
	const rkt_weight  *w		 = job->w;
	rkt_quant		  *q		 = job->q;
	const tensor_desc *desc		 = job->desc;
	backend			  *host		 = job->host;
	int				   bits		 = q->bits;
	int				   atom_cols = 256 / bits;
	float			  *tmp		 = xmalloc((size_t)w->k * sizeof(float));
	status_code		   s		 = OK;
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
			int k0 = kt * q->group, kl = quant_tile_len(w, q->group, kt);
			int valid = kl < w->k - k0 ? kl : w->k - k0;
			if (valid < 0)
				valid = 0;
			float		amax = 0.0f;
			int			i	 = 0;
			float32x4_t vm	 = vdupq_n_f32(0.0f);
			for (; i + 4 <= valid; i += 4)
				vm = vmaxq_f32(vm, vabsq_f32(vld1q_f32(tmp + k0 + i)));
			amax = vmaxvq_f32(vm);
			for (; i < valid; i++)
				if (fabsf(tmp[k0 + i]) > amax)
					amax = fabsf(tmp[k0 + i]);
			int				  qmax	 = (1 << (bits - 1)) - 1;
			int				  qmin	 = bits == 4 ? -qmax - 1 : -qmax;
			float			  scale	 = amax / (float)qmax;
			float			  inv	 = scale > 0.0f ? 1.0f / scale : 0.0f;
			uint8_t			 *base	 = (uint8_t *)wc->wt.map + wc->w_off[nt * q->n_ktiles + kt];
			int				  blocks = kl / 32;
			int				  sum	 = 0;
			const float32x4_t sv	 = vdupq_n_f32(inv);
			const int32x4_t	  sqmax = vdupq_n_s32(qmax), sqmin = vdupq_n_s32(qmin);
			for (int b = 0; b < blocks; b++) {
				int		 kb	 = b * 32;
				uint8_t *row = base + (((size_t)(col / atom_cols) * blocks + b) * atom_cols +
									   col % atom_cols) *
										  (4 * bits);
				if (valid - kb >= 32) {
					const float *xp = tmp + k0 + kb;
					int32x4_t	 a0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp), sv));
					int32x4_t	 a1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 4), sv));
					int32x4_t	 a2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 8), sv));
					int32x4_t	 a3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 12), sv));
					int32x4_t	 a4 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 16), sv));
					int32x4_t	 a5 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 20), sv));
					int32x4_t	 a6 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 24), sv));
					int32x4_t	 a7 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 28), sv));
					a0				= vminq_s32(sqmax, vmaxq_s32(sqmin, a0));
					a1				= vminq_s32(sqmax, vmaxq_s32(sqmin, a1));
					a2				= vminq_s32(sqmax, vmaxq_s32(sqmin, a2));
					a3				= vminq_s32(sqmax, vmaxq_s32(sqmin, a3));
					a4				= vminq_s32(sqmax, vmaxq_s32(sqmin, a4));
					a5				= vminq_s32(sqmax, vmaxq_s32(sqmin, a5));
					a6				= vminq_s32(sqmax, vmaxq_s32(sqmin, a6));
					a7				= vminq_s32(sqmax, vmaxq_s32(sqmin, a7));
					sum += vaddvq_s32(vaddq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)),
												vaddq_s32(vaddq_s32(a4, a5), vaddq_s32(a6, a7))));
					int8x16_t c0 =
						vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(a0), vqmovn_s32(a1))),
									vqmovn_s16(vcombine_s16(vqmovn_s32(a2), vqmovn_s32(a3))));
					int8x16_t c1 =
						vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(a4), vqmovn_s32(a5))),
									vqmovn_s16(vcombine_s16(vqmovn_s32(a6), vqmovn_s32(a7))));
					if (bits == 4) {
						quant_pack_nibbles(row, c0);
						quant_pack_nibbles(row + 8, c1);
					} else {
						vst1q_s8((int8_t *)row, c0);
						vst1q_s8((int8_t *)row + 16, c1);
					}
					continue;
				}
				for (int j = 0; j < 32; j++) {
					int	  kk = k0 + kb + j;
					float v	 = kk < w->k ? tmp[kk] : 0.0f;
					long  qv = lrintf(v * inv);
					qv		 = qv > qmax ? qmax : qv < qmin ? qmin : qv;
					sum += (int)qv;
					quant_put_weight(q, base, col, kl, kb, j, (int)qv);
				}
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
	if (w->k_pad % 32 != 0)
		return ERR_UNSUPPORTED;
	rkt_quant *q = xcalloc(1, sizeof(*q));
	q->bits		 = bits;
	q->group	 = RKT_QUANT_GROUP;
	q->n_ktiles	 = (w->k_pad + q->group - 1) / q->group;
	q->n64		 = (w->n + 63) / 64 * 64;
	q->s_w		 = xcalloc((size_t)q->n_ktiles * q->n64, sizeof(float));
	q->corr		 = bits == 4 ? xcalloc((size_t)q->n_ktiles * q->n64, sizeof(int16_t)) : NULL;
	if (!quant_alloc_cores(p, w, q)) {
		rkt_quant_free(p, q);
		return ERR_OUT_OF_MEMORY;
	}

	quant_job	job = {.w = w, .q = q, .desc = desc, .host = backend_host()};
	status_code s	= rkt_parallel_rows(w->n, quant_build_rows, &job);

	for (int c = 0; c < RKT_MAX_CORES && s == OK; c++) {
		rkt_quant_core *wc = &q->core[c];
		if (wc->n == 0)
			continue;
		if (!quant_emit_regs(q, w, wc, wc->reg.map, wc->n_ops, quant_token_rows(q, 1), 1,
							 wc->in.dma_addr, wc->out.dma_addr))
			s = ERR_INTERNAL;
		if (s == OK && (rkt_bo_flush(&p->dev, &wc->wt) || rkt_bo_flush(&p->dev, &wc->reg)))
			s = ERR_IO;
	}
	if (s != OK) {
		rkt_quant_free(p, q);
		return s;
	}
	w->quant = q;
	return OK;
}

status_code rkt_quant_build_best(rkt_priv *p, rkt_weight *w, const tensor_desc *desc) {
	int bits = quant_bits_for_type(desc->type);
	if (bits == 0)
		return ERR_UNSUPPORTED;
	return quant_build(p, w, desc, bits);
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

static int quant_pf_valid(const rkt_quant_core *wc, int tokens, const rkt_quant_scratch *sc) {
	return wc->pf_tokens == tokens && wc->pf_in_dma == (uint32_t)sc->in.dma_addr &&
		   wc->pf_out_dma == (uint32_t)sc->out.dma_addr;
}

static int quant_prefill_prepare(rkt_priv *p, const rkt_weight *w, int tokens, quant_io *io) {
	int m = quant_token_rows(w->quant, tokens);
	for (int c = 0; c < p->dev.n_cores; c++) {
		rkt_quant_core	  *wc = &w->quant->core[c];
		rkt_quant_scratch *sc = &p->quant_pf[c];
		if (wc->n == 0)
			continue;
		if (!rkt_ensure_bo(p, c, (size_t)w->k_pad * tokens, &sc->in) ||
			!rkt_ensure_bo(p, c, wc->out.size * tokens, &sc->out) ||
			!rkt_ensure_capacity((void **)&sc->n_ops, &sc->n_ops_cap, (size_t)wc->n_tasks,
								 sizeof(uint32_t)))
			return 0;
		rkt_bo	 *reg	= &sc->reg;
		uint32_t *n_ops = sc->n_ops;
		if (tokens == RKT_QUANT_PREFILL_TOKENS &&
			rkt_ensure_capacity((void **)&wc->pf_n_ops, &wc->pf_cap, (size_t)wc->n_tasks,
								sizeof(uint32_t)) &&
			rkt_ensure_bo(p, c, (size_t)wc->n_tasks * RKT_REGCMD_BYTES, &wc->pf_reg)) {
			reg	  = &wc->pf_reg;
			n_ops = wc->pf_n_ops;
		} else if (!rkt_ensure_bo(p, c, (size_t)wc->n_tasks * RKT_REGCMD_BYTES, &sc->reg))
			return 0;
		if (reg != &wc->pf_reg || !quant_pf_valid(wc, tokens, sc)) {
			if (!quant_emit_regs(w->quant, w, wc, reg->map, n_ops, m, tokens,
								 (uint32_t)sc->in.dma_addr, (uint32_t)sc->out.dma_addr))
				return 0;
			if (rkt_bo_flush(&p->dev, reg))
				return 0;
			wc->pf_tokens = 0;
			if (reg == &wc->pf_reg) {
				wc->pf_tokens  = tokens;
				wc->pf_in_dma  = (uint32_t)sc->in.dma_addr;
				wc->pf_out_dma = (uint32_t)sc->out.dma_addr;
			}
		}
		io[c] = (quant_io){&sc->in, &sc->out, reg, n_ops};
	}
	return 1;
}

static float quant_absmax(const float *x, int n) {
	float		m = 0.0f;
	int			i = 0;
	float32x4_t v = vdupq_n_f32(0.0f);
	for (; i + 4 <= n; i += 4)
		v = vmaxq_f32(v, vabsq_f32(vld1q_f32(x + i)));
	m = vmaxvq_f32(v);
	for (; i < n; i++)
		if (fabsf(x[i]) > m)
			m = fabsf(x[i]);
	return m;
}

typedef struct {
	const float	   *f32;
	const uint16_t *f16;
} quant_src;

static const float *quant_src_tile(const quant_src *src, size_t offset, int n, float *tmp) {
	if (src->f32)
		return src->f32 + offset;
	rkt_convert_from_f16(tmp, src->f16 + offset, (size_t)n);
	return tmp;
}

static void quant_activation_scales(const rkt_weight *w, const quant_src *x, int tokens,
									float *s_act) {
	int group = w->quant->group, n_ktiles = w->quant->n_ktiles;
	for (int tk = 0; tk < tokens; tk++)
		for (int kt = 0; kt < n_ktiles; kt++) {
			int	  k0 = kt * group, kl = quant_tile_len(w, group, kt);
			int	  valid = kl < w->k - k0 ? kl : w->k - k0;
			float tile[RKT_QUANT_GROUP];
			float amax =
				quant_absmax(quant_src_tile(x, (size_t)tk * w->k + k0, valid > 0 ? valid : 0, tile),
							 valid > 0 ? valid : 0);
			s_act[tk * n_ktiles + kt] = amax > 0.0f ? amax / 127.0f : 1.0f;
		}
}

static int quant_quantize_activation(float v, float scale) {
	long qv = lrintf(v / scale);
	return qv > 127 ? 127 : qv < -127 ? -127 : (int)qv;
}

static void quant_quantize_tile(const float *x, int valid, int kl, float scale, int8_t *q) {
	int			j  = 0;
	float32x4_t sv = vdupq_n_f32(scale);
	for (; j + 8 <= valid; j += 8) {
		int32x4_t a = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + j), sv));
		int32x4_t b = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + j + 4), sv));
		int8x8_t  o = vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)));
		vst1_s8(q + j, vmax_s8(o, vdup_n_s8(-127)));
	}
	for (; j < valid; j++)
		q[j] = (int8_t)quant_quantize_activation(x[j], scale);
	for (; j < kl; j++)
		q[j] = 0;
}

static void quant_pack_token(const rkt_weight *w, const quant_src *x, int tk, int tokens,
							 const float *s_act, int bits, uint8_t *packed) {
	int	   group = w->quant->group, n_ktiles = w->quant->n_ktiles, m = tokens * (bits == 4 ? 2 : 1);
	int8_t q[RKT_QUANT_GROUP];
	for (int kt = 0; kt < n_ktiles; kt++) {
		int	  k0 = kt * group, kl = quant_tile_len(w, group, kt);
		int	  valid = kl < w->k - k0 ? kl : w->k - k0;
		float scale = s_act[tk * n_ktiles + kt];
		float tile[RKT_QUANT_GROUP];
		quant_quantize_tile(quant_src_tile(x, (size_t)tk * w->k + k0, valid > 0 ? valid : 0, tile),
							valid > 0 ? valid : 0, kl, scale, q);
		if (bits == 8) {
			for (int plane = 0; plane < kl / 16; plane++)
				memcpy(packed + (size_t)k0 * tokens + ((size_t)plane * tokens + tk) * 16,
					   q + plane * 16, 16);
			continue;
		}
		for (int kb = 0; kb < kl; kb += 32) {
			uint8_t *p0 = packed + (size_t)k0 * tokens + ((size_t)(kb / 32) * m + 2 * tk) * 16;
			const uint8x16_t off = vdupq_n_u8(8), m15 = vdupq_n_u8(15);
			for (int half = 0; half < 2; half++) {
				uint8x16_t c =
					veorq_u8(vld1q_u8((const uint8_t *)q + kb + half * 16), vdupq_n_u8(0x80));
				uint8x16_t hi = vandq_u8(vaddq_u8(vshrq_n_u8(c, 4), off), m15);
				uint8x16_t lo = vandq_u8(vaddq_u8(vandq_u8(c, m15), off), m15);
				quant_pack_nibbles(p0, vreinterpretq_s8_u8(hi));
				quant_pack_nibbles(p0 + 16, vreinterpretq_s8_u8(lo));
				p0 += 8;
			}
			continue;
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
}

typedef struct {
	const rkt_weight *w;
	const quant_src	 *x;
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

static void quant_pack_activation(const rkt_weight *w, const quant_src *x, int tokens,
								  const float *s_act, int bits, uint8_t *packed) {
	quant_pack_job job = {w, x, s_act, packed, tokens, bits};
	tpool_parallel_for(tpool_fetch_pool(), tokens, RKT_QUANT_PACK_MIN_TOKENS, quant_pack_chunk,
					   &job);
}

static void quant_reduce_i4(const rkt_quant *q, const uint8_t *tile, size_t tile_stride, int m,
							int tk, int r0, int col, const float *sa, float *dst) {
	const int16_t *o0 = (const int16_t *)tile;
	float32x4_t	   y0 = vdupq_n_f32(0.0f), y1 = vdupq_n_f32(0.0f);
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
}

static void quant_reduce_i8(const rkt_quant *q, const uint8_t *tile, size_t tile_stride, int m,
							int tk, int r0, int col, const float *sa, float *dst) {
	const int32_t *o0 = (const int32_t *)tile;
	float32x4_t	   y0 = vdupq_n_f32(0.0f), y1 = vdupq_n_f32(0.0f);
	for (int kt = 0; kt < q->n_ktiles; kt++) {
		const int32_t *lo = o0 + (size_t)kt * tile_stride + (size_t)col * m + tk * 4;
		const float	  *sw = q->s_w + (size_t)kt * q->n64 + r0 + col;
		y0 = vfmaq_f32(y0, vmulq_n_f32(vld1q_f32(sw), sa[kt]), vcvtq_f32_s32(vld1q_s32(lo)));
		y1 = vfmaq_f32(y1, vmulq_n_f32(vld1q_f32(sw + 4), sa[kt]),
					   vcvtq_f32_s32(vld1q_s32(lo + 4 * m)));
	}
	vst1q_f32(dst, y0);
	vst1q_f32(dst + 4, y1);
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
	int					  n_ntiles = quant_ntiles(wc);
	tpool				 *pool	   = tpool_fetch_pool();
	for (int nt = 0; nt < n_ntiles; nt++) {
		int				 ncols = quant_ncols(wc, nt);
		quant_reduce_job job   = {
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

typedef struct {
	rkt_priv	*p;
	rkt_weight **ws;
	quant_io (*io)[RKT_MAX_CORES];
	uint8_t	   *packed[2];
	size_t		input_bytes;
	int			count;
	int			core;
	int			submitted;
	status_code status;
} quant_core_submit;

static int quant_core_task_count(rkt_weight **ws, int count, int core) {
	int n_list = 0;
	for (int i = 0; i < count; i++)
		n_list += ws[i]->quant->core[core].n > 0 ? ws[i]->quant->core[core].n_tasks : 0;
	return n_list;
}

static status_code quant_stage_core_input(const quant_core_submit *job) {
	for (int i = 0; i < job->count; i++) {
		if (job->ws[i]->quant->core[job->core].n == 0)
			continue;
		const quant_io *io = &job->io[i][job->core];
		memcpy(io->in->map, job->packed[job->ws[i]->quant->bits == 8], job->input_bytes);
		if (rkt_bo_flush(&job->p->dev, io->in))
			return ERR_IO;
	}
	return OK;
}

static status_code quant_submit_core_tasks(quant_core_submit *job) {
	int				c	   = job->core;
	int				n_list = quant_core_task_count(job->ws, job->count, c);
	quant_job_task *list   = xmalloc((size_t)n_list * sizeof(*list));
	int				at	   = 0;
	for (int i = 0; i < job->count; i++) {
		const rkt_quant_core *wc = &job->ws[i]->quant->core[c];
		for (int t = 0; wc->n > 0 && t < wc->n_tasks; t++)
			list[at++] = (quant_job_task){.io = &job->io[i][c], .task = t};
	}
	status_code s = OK;
	for (int start = 0; start < n_list && s == OK; start += RKT_MAX_TASKS) {
		int				n_tasks = n_list - start < RKT_MAX_TASKS ? n_list - start : RKT_MAX_TASKS;
		uint32_t		dma[RKT_MAX_TASKS], ops[RKT_MAX_TASKS];
		const rkt_bo   *in[RKT_MAX_HANDLES], *out[RKT_MAX_HANDLES];
		int				n_in = 0, n_out = 0;
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
		if (rkt_submit_bos(&job->p->dev, c, dma, ops, n_tasks, in, n_in, out, n_out))
			s = ERR_COMPUTE_FAIL;
		job->submitted = 1;
	}
	free(list);
	return s;
}

static void quant_core_run(void *arg) {
	quant_core_submit *job = arg;
	job->status			   = quant_stage_core_input(job);
	if (job->status == OK)
		job->status = quant_submit_core_tasks(job);
}

static status_code quant_run_inner(rkt_priv *p, rkt_weight **ws, float **ys, int count,
								   const float *x, int tokens) {
	const rkt_weight *w0	= ws[0];
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

	if (!rkt_ensure_capacity((void **)&p->quant_s_act, &p->quant_s_act_cap,
							 (size_t)tokens * w0->quant->n_ktiles, sizeof(float)))
		return ERR_OUT_OF_MEMORY;
	float	 *s_act		= p->quant_s_act;
	uint8_t	 *packed[2] = {NULL, NULL};
	quant_src src		= {.f32 = x, .f16 = x ? NULL : p->quant_x16};
	quant_activation_scales(w0, &src, tokens, s_act);
	for (int i = 0; i < count; i++) {
		int bits = ws[i]->quant->bits;
		int slot = bits == 8;
		if (packed[slot])
			continue;
		if (!rkt_ensure_capacity((void **)&p->quant_pack[slot], &p->quant_pack_cap[slot],
								 (size_t)k_pad * tokens, 1))
			return ERR_OUT_OF_MEMORY;
		packed[slot] = p->quant_pack[slot];
		quant_pack_activation(w0, &src, tokens, s_act, bits, packed[slot]);
	}

	quant_core_submit jobs[RKT_MAX_CORES];
	int				  n_active = 0;
	for (int c = 0; c < p->dev.n_cores; c++) {
		if (quant_core_task_count(ws, count, c) == 0)
			continue;
		jobs[n_active++] = (quant_core_submit){.p			= p,
											   .ws			= ws,
											   .io			= io,
											   .count		= count,
											   .core		= c,
											   .input_bytes = (size_t)k_pad * tokens,
											   .status		= OK};
		memcpy(jobs[n_active - 1].packed, packed, sizeof(packed));
	}
	for (int i = 0; i < n_active; i++)
		quant_core_run(&jobs[i]);

	status_code s							  = OK;
	int			submitted_core[RKT_MAX_CORES] = {0};
	for (int i = 0; i < n_active; i++) {
		const quant_core_submit *job = &jobs[i];
		submitted_core[job->core]	 = job->submitted;
		if (s == OK)
			s = job->status;
	}
	for (int i = 0; i < count; i++)
		for (int c = 0; c < p->dev.n_cores; c++) {
			if (!submitted_core[c] || ws[i]->quant->core[c].n == 0)
				continue;
			if (rkt_bo_wait(&p->dev, io[i][c].out) && s == OK)
				s = ERR_COMPUTE_FAIL;
			if (s != OK)
				continue;
			quant_accumulate_core(ws[i], &io[i][c], c, tokens, s_act, ys[i]);
		}
	return s;
}

status_code rkt_quant_run_many(rkt_priv *p, rkt_weight **ws, float **ys, int count, const float *x,
							   int tokens) {
	status_code s = quant_run_inner(p, ws, ys, count, x, tokens);
	if (s != OK && p->quant_logged < 24) {
		p->quant_logged++;
		WARN("rocket: quant call k=%d n=%d count=%d status=%d", ws[0]->k, ws[0]->n, count, s);
	}
	return s;
}

void rkt_quant_disable(rkt_priv *p, status_code s) {
	if (p->quant_failed)
		return;
	p->quant_failed = 1;
	WARN("rocket: quant decode path failed (status %d, errno=%d), back to f16 weights", s, errno);
}

void rkt_quant_disable_prefill(rkt_priv *p, status_code s) {
	if (p->quant_prefill_failed)
		return;
	p->quant_prefill_failed = 1;
	WARN("rocket: quant prefill path failed (status %d, errno=%d), prefill on cpu", s, errno);
}

int rkt_quant_servable(const rkt_priv *p, const rkt_weight *w, int row0, int n, int m) {
	if (!w->quant || p->quant_failed || p->npu_failed || row0 != 0 || n != w->n)
		return 0;
	return m == 1 || !p->quant_prefill_failed;
}

status_code rkt_quant_matmul(rkt_priv *p, rkt_weight *w, const float *x, float *y, int m) {
	rkt_weight *one = w;
	status_code s	= OK;
	pthread_mutex_lock(&p->lock);
	for (int t0 = 0; t0 < m && s == OK; t0 += RKT_QUANT_PREFILL_TOKENS) {
		int	   tokens = m - t0 < RKT_QUANT_PREFILL_TOKENS ? m - t0 : RKT_QUANT_PREFILL_TOKENS;
		float *yo	  = y + (size_t)t0 * w->n;
		s			  = rkt_quant_run_many(p, &one, &yo, 1, x + (size_t)t0 * w->k, tokens);
	}
	pthread_mutex_unlock(&p->lock);
	return s;
}

status_code rkt_quant_matmul_native(rkt_priv *p, rkt_weight *w, float *y, int m) {
	rkt_weight *one = w;
	status_code s	= OK;
	pthread_mutex_lock(&p->lock);
	for (int t0 = 0; t0 < m && s == OK; t0 += RKT_QUANT_PREFILL_TOKENS) {
		int	   tokens = m - t0 < RKT_QUANT_PREFILL_TOKENS ? m - t0 : RKT_QUANT_PREFILL_TOKENS;
		float *yo	  = y + (size_t)t0 * w->n;
		p->quant_x16  = p->nat_act + (size_t)t0 * w->k;
		s			  = rkt_quant_run_many(p, &one, &yo, 1, NULL, tokens);
	}
	p->quant_x16 = NULL;
	pthread_mutex_unlock(&p->lock);
	return s;
}

static void quant_dequant_row(const rkt_weight *w, int row, float *dst) {
	const rkt_quant *q = w->quant;
	int				 c = 0;
	while (c + 1 < RKT_MAX_CORES && q->core[c + 1].n > 0 && row >= q->core[c + 1].n0)
		c++;
	const rkt_quant_core *wc	 = &q->core[c];
	int					  local	 = row - wc->n0;
	int					  nt	 = local / RKT_QUANT_NTILE;
	int					  col	 = local % RKT_QUANT_NTILE;
	int					  atom	 = 256 / q->bits;
	size_t				  stride = (size_t)4 * q->bits;
	for (int kt = 0; kt < q->n_ktiles; kt++) {
		int			   kl	 = quant_tile_len(w, q->group, kt);
		int			   k0	 = kt * q->group;
		const uint8_t *base	 = (const uint8_t *)wc->wt.map + wc->w_off[nt * q->n_ktiles + kt];
		float		   scale = q->s_w[(size_t)kt * q->n64 + row];
		for (int kb = 0; kb < kl; kb += 32) {
			const uint8_t *atom_row =
				base + (((size_t)(col / atom) * (kl / 32) + kb / 32) * atom + col % atom) * stride;
			for (int j = 0; j < 32 && k0 + kb + j < w->k; j++) {
				int v;
				if (q->bits == 8) {
					v = (int8_t)atom_row[j];
				} else {
					int nib = (j & 1) ? atom_row[j / 2] & 15 : atom_row[j / 2] >> 4;
					v		= nib >= 8 ? nib - 16 : nib;
				}
				dst[k0 + kb + j] = (float)v * scale;
			}
		}
	}
}

typedef struct {
	const rkt_weight *w;
	const float		 *x;
	float			 *y;
	int				  row0;
	int				  n;
	int				  m;
} quant_cpu_job;

static status_code quant_cpu_rows(void *arg, int lo, int hi) {
	const quant_cpu_job *j	 = arg;
	const rkt_weight	*w	 = j->w;
	float				*row = xmalloc((size_t)w->k * sizeof(float));
	for (int r = lo; r < hi; r++) {
		quant_dequant_row(w, j->row0 + r, row);
		for (int t = 0; t < j->m; t++) {
			const float *xt	 = j->x + (size_t)t * w->k;
			float		 acc = 0.0f;
			for (int i = 0; i < w->k; i++)
				acc += row[i] * xt[i];
			j->y[(size_t)t * j->n + r] = acc;
		}
	}
	free(row);
	return OK;
}

void rkt_quant_cpu_matmul(const rkt_weight *w, int row0, int n, const float *x, float *y, int m) {
	quant_cpu_job job = {.w = w, .x = x, .y = y, .row0 = row0, .n = n, .m = m};
	rkt_parallel_rows(n, quant_cpu_rows, &job);
}

int rkt_ensure_bo(rkt_priv *p, int core, size_t bytes, rkt_bo *bo) {
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
			.w_dma	= (uint32_t)(pt->bo.dma_addr + (size_t)(t->job.row_start - pt->row_start) *
													   t->ctx.kc->k_len * rkt_weight_esz(t->ctx.w)),
			.out_dma = (uint32_t)(sl->out.dma_addr + t->out_off),
			.epi	 = (uint32_t)t->ctx.epi,
			.out16	 = t->ctx.y16 != NULL,
			.ew_dma	 = t->ctx.ew16 ? (uint32_t)(sl->in.dma_addr + t->ew_off) : 0,
		};
		if (t->ctx.epi_cap > 0.0f)
			memcpy(&key.epi_cap_bits, &t->ctx.epi_cap, sizeof(key.epi_cap_bits));
		stream_entry *e = stream_lookup(sc, &key);
		if (!e->used) {
			size_t	  offset = align_queue(sc->bytes_used);
			uint64_t *ops	 = (uint64_t *)((uint8_t *)sc->bo.map + offset);
			uint32_t  n_ops;
			if (key.ew_dma) {
				n_ops = (uint32_t)rkt_emit_matmul_f16_ewadd(ops, key.blk, key.k_len, key.rows,
															key.in_dma, key.w_dma, key.out_dma,
															key.ew_dma);
			} else if (key.out16) {
				n_ops = (uint32_t)rkt_emit_matmul_f16_out16(ops, key.blk, key.k_len, key.rows,
															key.in_dma, key.w_dma, key.out_dma, 2);
			} else if (key.epi) {
				rkt_lut_cfg cfg;
				rkt_lut_configure(&cfg, (rkt_lut_kind)key.epi);
				n_ops = (uint32_t)rkt_emit_matmul_f16_act_lut(
					ops, key.blk, key.k_len, key.rows, key.in_dma, key.w_dma, key.out_dma, &cfg,
					t->ctx.epi_cap > 0.0f ? (float)cfg.S / t->ctx.epi_cap : (float)cfg.S);
			} else {
				n_ops = (uint32_t)rkt_emit_matmul_fmt(ops, t->ctx.w->fmt, key.blk, key.k_len,
													  key.rows, key.in_dma, key.w_dma, key.out_dma);
			}
			if (!n_ops || n_ops > RKT_REGCMD_MAX_OPS)
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
	if (!rkt_ensure_bo(p, core, q->out_bytes, &sl->out))
		return ERR_OUT_OF_MEMORY;
	status_code s = stage_streams(p, core, q, sl);
	if (s != OK)
		return s;
	return rkt_bo_flush(&p->dev, &sl->in) ? ERR_IO : OK;
}

static inline void collect_store(float *dst, float v, int accumulate) {
	if (accumulate)
		*dst += v;
	else
		*dst = v;
}

static inline float collect_at(const float *src_row, size_t group_stride, int nn) {
	return src_row[(size_t)(nn / RKT_OUT_C2_F32) * group_stride + nn % RKT_OUT_C2_F32];
}

static void collect_task_f16(const rkt_task *t, const uint16_t *om, int lo, int hi) {
	const block_ctx *b	   = &t->ctx;
	const job_desc	*j	   = &t->job;
	size_t			 plane = (size_t)t->blk * 8;
	for (int r = 0; r < t->blk; r++) {
		uint16_t *dst =
			b->y16 + (size_t)(b->m_done + r) * b->y16_stride + (j->row_start + lo - b->row0);
		int nn = lo;
		for (; nn < hi && nn % 8; nn++)
			dst[nn - lo] = om[(size_t)(nn / 8) * plane + (size_t)r * 8 + nn % 8];
		for (; nn + 8 <= hi; nn += 8)
			memcpy(dst + (nn - lo), om + (size_t)(nn / 8) * plane + (size_t)r * 8,
				   8 * sizeof(uint16_t));
		for (; nn < hi; nn++)
			dst[nn - lo] = om[(size_t)(nn / 8) * plane + (size_t)r * 8 + nn % 8];
	}
}

static status_code collect_task(const rkt_task *t, const float *om) {
	const block_ctx *b		   = &t->ctx;
	const job_desc	*j		   = &t->job;
	int				 blk	   = t->blk;
	float			 out_scale = 1.0f;
	if (b->epi) {
		rkt_lut_cfg cfg;
		rkt_lut_configure(&cfg, (rkt_lut_kind)b->epi);
		out_scale = (b->epi_cap > 0.0f ? b->epi_cap : 1.0f) / (float)cfg.F;
	}
	int lo = (b->row0 > j->row_start ? b->row0 : j->row_start) - j->row_start;
	int hi = (b->row0 + b->n < j->row_start + j->rows ? b->row0 + b->n : j->row_start + j->rows) -
			 j->row_start;
	if (hi <= lo)
		return OK;
	if (b->y16) {
		collect_task_f16(t, (const uint16_t *)om, lo, hi);
		return OK;
	}
	size_t group_stride = (size_t)blk * RKT_OUT_C2_F32;
	for (int r = 0; r < blk; r++) {
		float		*dst = &b->y[(size_t)(b->m_done + r) * b->n + (j->row_start + lo - b->row0)];
		const float *src_row = om + (size_t)r * RKT_OUT_C2_F32;
		int			 nn		 = lo;
		for (; nn < hi && nn % RKT_OUT_C2_F32; nn++)
			collect_store(dst + (nn - lo), collect_at(src_row, group_stride, nn) * out_scale,
						  b->accumulate);
		for (; nn + RKT_OUT_C2_F32 <= hi; nn += RKT_OUT_C2_F32) {
			float32x4_t v = vmulq_n_f32(
				vld1q_f32(src_row + (size_t)(nn / RKT_OUT_C2_F32) * group_stride), out_scale);
			float *d = dst + (nn - lo);
			vst1q_f32(d, b->accumulate ? vaddq_f32(vld1q_f32(d), v) : v);
		}
		for (; nn < hi; nn++)
			collect_store(dst + (nn - lo), collect_at(src_row, group_stride, nn) * out_scale,
						  b->accumulate);
	}
	return OK;
}

#define RKT_NPU_MAX_CONSECUTIVE_FAILURES 3

void rkt_npu_record_failure(rkt_priv *p) {
	if (++p->npu_consecutive_failures < RKT_NPU_MAX_CONSECUTIVE_FAILURES)
		return;
	p->npu_failed = 1;
	WARN("rocket: %d consecutive npu failures, npu disabled", p->npu_consecutive_failures);
}

typedef struct {
	const rkt_priv	  *p;
	const rkt_pending *pd;
	int				   bank;
	int				   status;
} collect_drain;

static status_code collect_core(const collect_drain *d, int core) {
	const rkt_core_queue *q	 = &d->pd->queue[core];
	const rkt_slot		 *sl = &d->p->slots[d->bank][q->cls][core];
	for (int t = 0; t < q->n_tasks; t++) {
		status_code s = collect_task(
			&q->tasks[t], (const float *)((const uint8_t *)sl->out.map + q->tasks[t].out_off));
		if (s != OK)
			return s;
	}
	return OK;
}

static void collect_drain_chunk(int begin, int end, int tid, void *arg) {
	(void)tid;
	collect_drain *d = arg;
	for (int i = begin; i < end; i++) {
		status_code s = collect_core(d, d->pd->active[i]);
		if (s != OK)
			__atomic_store_n(&d->status, (int)s, __ATOMIC_RELAXED);
	}
}

static status_code collect_pending(const rkt_priv *p, const rkt_pending *pd, int bank) {
	collect_drain d			  = {.p = p, .pd = pd, .bank = bank, .status = OK};
	size_t		  total_bytes = 0;
	for (int i = 0; i < pd->n_active; i++)
		total_bytes += pd->queue[pd->active[i]].out_bytes;
	int min_items = total_bytes >= RKT_COLLECT_PAR_MIN_BYTES ? 1 : pd->n_active + 1;
	tpool_parallel_for(tpool_fetch_pool(), pd->n_active, min_items, collect_drain_chunk, &d);
	return (status_code)d.status;
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
	if (s != OK)
		return s;
	return collect_pending(p, pd, bank);
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

void rkt_queue_abort(rkt_priv *p) {
	drain_all_pending(p);
	reset_queue(p);
}

typedef struct {
	rkt_priv *p;
	int		  bank;
	int		  core;
	int		  err;
} core_submit;

static void submit_core_queue(void *arg) {
	core_submit			 *job = arg;
	rkt_priv			 *p	  = job->p;
	const rkt_core_queue *q	  = &p->queue[job->core];
	const rkt_slot		 *sl  = &p->slots[job->bank][q->cls][job->core];
	uint32_t			  n_ops[RKT_MAX_TASKS];
	uint32_t			  regcmd_dma[RKT_MAX_TASKS];
	for (int t = 0; t < q->n_tasks; t++) {
		n_ops[t]	  = q->tasks[t].n_ops;
		regcmd_dma[t] = q->tasks[t].regcmd_dma;
	}
	job->err = rkt_submit(&p->dev, regcmd_dma, n_ops, q->n_tasks, &p->streams[job->core].bo,
						  &sl->in, &sl->out);
	if (job->err)
		ERROR("rocket: submit on core %d failed (%d tasks, cls %d): %s", job->core, q->n_tasks,
			  q->cls, strerror(-job->err));
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

	int			 bank = p->bank;
	rkt_pending *pd	  = &p->pending[bank];
	core_submit	 jobs[RKT_MAX_CORES];
	int			 submitted = 0;
	if (s == OK) {
		for (int i = 0; i < n_active; i++)
			jobs[i] = (core_submit){.p = p, .bank = bank, .core = active[i]};
		for (int i = 0; i < n_active; i++)
			submit_core_queue(&jobs[i]);
		for (int i = 0; i < n_active; i++)
			if (jobs[i].err)
				s = ERR_COMPUTE_FAIL;
		submitted = n_active;
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

status_code rkt_queue_flush(rkt_priv *p) {
	status_code s = submit_queue(p, 0);
	if (s == OK)
		p->npu_consecutive_failures = 0;
	return s;
}

status_code rkt_queue_flush_async(rkt_priv *p) {
	status_code s = submit_queue(p, 1);
	if (s == OK)
		p->npu_consecutive_failures = 0;
	return s;
}

status_code rkt_queue_sync(rkt_priv *p) {
	return drain_all_pending(p);
}

static int job_max_rows(int k_len, int esz) {
	int rows = (int)(RKT_MAX_JOB_WEIGHT_BYTES / ((size_t)k_len * esz));
	rows	 = rows & ~(RKT_N_ALIGN - 1);
	if (rows < RKT_N_ALIGN)
		return RKT_N_ALIGN;
	return rows < RKT_MAX_N_CHUNK ? rows : RKT_MAX_N_CHUNK;
}

static int plan_jobs(const block_ctx *b, int lo, int hi, int round, job_desc *jobs) {
	int n_jobs	 = 0;
	int max_rows = job_max_rows(b->kc->k_len, rkt_weight_esz(b->w));
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

static status_code stage_ew_operand(rkt_priv *p, slot_class cls, const block_ctx *b, int blk,
									const job_desc *job, size_t *off) {
	if (blk != 1 || job->rows % RKT_EW_SURFACE_CHANNELS || job->rows > RKT_MAX_N_CHUNK)
		return ERR_UNSUPPORTED;
	size_t bytes =
		(size_t)(job->rows / RKT_EW_SURFACE_CHANNELS) * RKT_EW_SURFACE_SLOTS * sizeof(uint16_t);
	size_t at = align_queue(p->in_len);
	if (!grow_slot_input(p, cls, job->core, at, at + bytes))
		return ERR_OUT_OF_MEMORY;
	uint16_t residual[RKT_MAX_N_CHUNK];
	for (int channel = 0; channel < job->rows; channel++) {
		int col			  = job->row_start + channel - b->row0;
		residual[channel] = col >= 0 && col < b->n ? b->ew16[col] : 0;
	}
	rkt_pack_ew_operand((uint16_t *)((uint8_t *)p->slots[p->bank][cls][job->core].in.map + at),
						residual, 1, job->rows);
	p->in_len = at + bytes;
	*off	  = at;
	return OK;
}

static status_code queue_block(rkt_priv *p, const block_ctx *b, int blk, const void *x_block) {
	int lo = b->row0 & ~(RKT_N_ALIGN - 1);
	int hi = (b->row0 + b->n + RKT_N_ALIGN - 1) & ~(RKT_N_ALIGN - 1);
	if (hi > b->w->n_pad)
		hi = b->w->n_pad;

	size_t	   in_bytes				  = (size_t)blk * b->kc->k_len * rkt_weight_esz(b->w);
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
				rkt_tile_input_esz((uint8_t *)p->slots[p->bank][cls][jobs[i].core].in.map + in_off,
								   x_block, blk, b->kc->k_len, rkt_weight_esz(b->w));
				written[jobs[i].core] = 1;
			}
			size_t ew_off = 0;
			if (b->ew16) {
				status_code s = stage_ew_operand(p, cls, b, blk, &jobs[i], &ew_off);
				if (s != OK)
					return s;
			}
			q->tasks[q->n_tasks++] = (rkt_task){
				.ctx	 = *b,
				.blk	 = blk,
				.job	 = jobs[i],
				.in_off	 = in_off,
				.out_off = q->out_bytes,
				.ew_off	 = ew_off,
			};
			q->out_bytes += align_queue((size_t)blk * jobs[i].rows *
										(b->y16 ? sizeof(uint16_t) : sizeof(float)));
		}
	}
}

int rkt_next_block_rows(int remaining, int max_rows) {
	if (remaining < 4)
		return 1;
	int blk = remaining / 4 * 4;
	return blk > max_rows ? max_rows : blk;
}

static const uint8_t *chunk_input(rkt_priv *p, const rkt_kchunk *kc, int k, int m, int esz,
								  int *stride_ok) {
	*stride_ok			= 1;
	const uint8_t *base = (const uint8_t *)(p->x_src ? p->x_src : p->x16);
	if (kc->k_len == k)
		return base;
	if (!rkt_ensure_capacity((void **)&p->x_chunk, &p->x_chunk_cap, (size_t)m * kc->k_len * esz / 2,
							 sizeof(uint16_t))) {
		*stride_ok = 0;
		return NULL;
	}
	for (int r = 0; r < m; r++)
		memcpy((uint8_t *)p->x_chunk + (size_t)r * kc->k_len * esz,
			   base + ((size_t)r * k + kc->k_off) * esz, (size_t)kc->k_len * esz);
	return (const uint8_t *)p->x_chunk;
}

static status_code enqueue_kchunks(rkt_priv *p, const rkt_weight *w, int row0, int n, float *y,
								   int m) {
	int esz = rkt_weight_esz(w);
	for (int ci = 0; ci < w->n_kchunks; ci++) {
		const rkt_kchunk *kc = &w->kchunks[ci];
		int				  ok;
		const uint8_t	 *xc = chunk_input(p, kc, w->k_pad, m, esz, &ok);
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
							  .epi_cap	  = p->epi_cap,
							  .y16		  = p->out16_dst,
							  .y16_stride = p->out16_stride,
							  .ew16		  = ci == 0 ? p->ew_res16 : NULL};
		int		  max_rows = rkt_max_batch_rows(kc->k_len, esz);
		for (int done = 0; done < m;) {
			int blk		  = rkt_next_block_rows(m - done, max_rows);
			b.m_done	  = done;
			status_code s = queue_block(p, &b, blk, xc + (size_t)done * kc->k_len * esz);
			if (s != OK)
				return s;
			done += blk;
		}
	}
	return OK;
}

static status_code enqueue_fold(rkt_priv *p, const rkt_weight *w, const rkt_kchunk *fold_kc,
								int row0, int n, const uint16_t *x16, float *y, int m) {
	if (!rkt_ensure_capacity((void **)&p->x_chunk, &p->x_chunk_cap, (size_t)m * RKT_FOLD_K_LEN,
							 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	memset(p->x_chunk, 0, (size_t)m * RKT_FOLD_K_LEN * sizeof(uint16_t));
	for (int r = 0; r < m; r++)
		p->x_chunk[(size_t)r * RKT_FOLD_K_LEN] = x16[(size_t)r * w->k_pad + w->k];

	block_ctx fb = {.w = w, .kc = fold_kc, .row0 = row0, .n = n, .m = m, .y = y, .accumulate = 1};
	for (int done = 0; done < m;) {
		int blk		  = rkt_next_block_rows(m - done, RKT_TILE_ROWS);
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
		return rkt_queue_flush(p);
	rkt_queue_abort(p);
	return s;
}

static void convert_activation_row(rkt_float_fmt fmt, uint8_t *dst, const float *src, int k) {
	if (fmt == RKT_FMT_F16)
		rkt_convert_to_f16((uint16_t *)dst, src, (size_t)k);
	else if (fmt == RKT_FMT_BF16)
		rkt_convert_to_bf16((uint16_t *)dst, src, (size_t)k);
	else
		memcpy(dst, src, (size_t)k * sizeof(float));
}

status_code rkt_dense_matmul_enqueue_k(rkt_priv *p, const rkt_weight *w, int row0, int n,
									   const float *x, float *y, int m, int k_used) {
	int esz = rkt_weight_esz(w);
	if (!rkt_ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)m * w->k_pad * esz / 2,
							 sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	for (int r = 0; r < m; r++) {
		uint8_t *dst = (uint8_t *)p->x16 + (size_t)r * w->k_pad * esz;
		convert_activation_row(w->fmt, dst, x + (size_t)r * w->k, k_used);
		memset(dst + (size_t)k_used * esz, 0, (size_t)(w->k_pad - k_used) * esz);
	}
	return enqueue_kchunks(p, w, row0, n, y, m);
}

status_code rkt_dense_matmul_enqueue(rkt_priv *p, const rkt_weight *w, int row0, int n,
									 const float *x, float *y, int m) {
	return rkt_dense_matmul_enqueue_k(p, w, row0, n, x, y, m, w->k);
}

status_code rkt_dense_matmul_enqueue_f16(rkt_priv *p, const rkt_weight *w, int row0, int n,
										 const float *x, uint16_t *y16, int y16_stride, int m) {
	p->out16_dst	= y16;
	p->out16_stride = y16_stride;
	status_code s	= rkt_dense_matmul_enqueue(p, w, row0, n, x, NULL, m);
	p->out16_dst	= NULL;
	return s;
}

status_code rkt_dense_matmul_native(rkt_priv *p, const rkt_weight *w, int row0, int n, float *y,
									int m) {
	if (w->k == w->k_pad) {
		p->x_src = p->nat_act;
	} else {
		if (!rkt_ensure_capacity((void **)&p->x16, &p->x16_cap, (size_t)m * w->k_pad,
								 sizeof(uint16_t)))
			return ERR_OUT_OF_MEMORY;
		for (int r = 0; r < m; r++) {
			uint16_t *dst = p->x16 + (size_t)r * w->k_pad;
			memcpy(dst, p->nat_act + (size_t)r * w->k, (size_t)w->k * sizeof(uint16_t));
			memset(dst + w->k, 0, (size_t)(w->k_pad - w->k) * sizeof(uint16_t));
		}
	}
	status_code s = enqueue_kchunks(p, w, row0, n, y, m);
	p->x_src	  = NULL;
	if (s == OK)
		return rkt_queue_flush(p);
	rkt_queue_abort(p);
	return s;
}

status_code rkt_dense_matmul(rkt_priv *p, const rkt_weight *w, int row0, int n, const float *x,
							 float *y, int m) {
	status_code s = rkt_dense_matmul_enqueue(p, w, row0, n, x, y, m);
	if (s == OK)
		return rkt_queue_flush(p);
	rkt_queue_abort(p);
	return s;
}

static float tiled_weight_at(const rkt_weight *w, const rkt_part *pt, int k_len, int row, int col) {
	int	   esz	 = rkt_weight_esz(w);
	int	   cols	 = RKT_TILE_COLS * 2 / esz;
	int	   local = row - pt->row_start;
	size_t tile	 = ((size_t)(local / RKT_TILE_ROWS) * (k_len / cols) + (size_t)(col / cols)) *
				   (RKT_TILE_ROWS * cols);
	size_t index = tile + (size_t)(local % RKT_TILE_ROWS) * cols + (size_t)(col % cols);
	if (esz == 4) {
		float v;
		memcpy(&v, (const uint8_t *)pt->bo.map + index * sizeof(float), sizeof(v));
		return v;
	}
	uint16_t bits = ((const uint16_t *)pt->bo.map)[index];
	return w->fmt == RKT_FMT_BF16 ? bf16_bits_to_f32(bits) : f16_bits_to_f32(bits);
}

static float round_activation(rkt_float_fmt fmt, float v) {
	if (fmt == RKT_FMT_F16)
		return f16_bits_to_f32(f32_to_f16_bits(v));
	if (fmt == RKT_FMT_BF16)
		return bf16_bits_to_f32(f32_to_bf16_bits(v));
	return v;
}

static void tiled_matmul_cpu(const rkt_weight *w, int row0, int n, const float *x, float *y,
							 int m) {
	for (int r = 0; r < m; r++) {
		for (int j = 0; j < n; j++) {
			float acc = 0.0f;
			for (int ci = 0; ci < w->n_kchunks; ci++) {
				const rkt_kchunk *kc = &w->kchunks[ci];
				const rkt_part	 *pt = rkt_part_for_row(kc, row0 + j);
				for (int c = 0; c < kc->k_len; c++) {
					int col = kc->k_off + c;
					if (col >= w->k)
						break;
					acc += tiled_weight_at(w, pt, kc->k_len, row0 + j, c) *
						   round_activation(w->fmt, x[(size_t)r * w->k + col]);
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

status_code rkt_matmul_batch(backend *self, const buffer *wbuf, uint32_t w_type, const buffer *x,
							 buffer *y, int n, int k, int m) {
	(void)w_type;
	rkt_priv *p = self->priv;
	rkt_log_pipeline_usage(p);
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
	if (rkt_quant_servable(p, w, row0, n, m)) {
		if (rkt_native_pending(p, x, k, m)) {
			status_code ns = rkt_quant_matmul_native(p, w, rkt_ptr(y), m);
			if (ns == OK) {
				rkt_native_release(p);
				return OK;
			}
			rkt_native_settle();
		}
		status_code s4 = rkt_quant_matmul(p, w, rkt_ptr(x), rkt_ptr(y), m);
		if (s4 == OK)
			return OK;
		if (m == 1)
			rkt_quant_disable(p, s4);
		else
			rkt_quant_disable_prefill(p, s4);
	}
	if (!w->n_kchunks && w->quant) {
		rkt_quant_cpu_matmul(w, row0, n, rkt_ptr(x), rkt_ptr(y), m);
		return OK;
	}
	if (w->fused_parent) {
		row0 += w->fused_row0;
		w = w->fused_parent;
	}

	if (rkt_native_matches(p, x, w, k, m)) {
		pthread_mutex_lock(&p->lock);
		status_code ns = rkt_dense_matmul_native(p, w, row0, n, rkt_ptr(y), m);
		pthread_mutex_unlock(&p->lock);
		if (ns == OK) {
			rkt_native_release(p);
			return OK;
		}
		rkt_native_settle();
	}

	const float *xf = rkt_ptr(x);
	float		*yf = rkt_ptr(y);
	if (w->host_copy)
		return host_matmul(w, row0, n, x, y, m);

	pthread_mutex_lock(&p->lock);
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL : rkt_dense_matmul(p, w, row0, n, xf, yf, m);
	if (s != OK && !p->npu_failed) {
		rkt_npu_record_failure(p);
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

status_code rkt_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x, buffer *y,
					   int n, int k) {
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
			w->fmt != RKT_FMT_F16 || w->n_kchunks == 0 || w->k != ws[0]->k ||
			w->k_pad != ws[0]->k_pad || w->n_kchunks != ws[0]->n_kchunks)
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
	if (!rkt_build_layout(p, f)) {
		rkt_release_npu_state(p, f);
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
				const rkt_part *sp = rkt_part_for_row(sk, r);
				const rkt_part *dp = rkt_part_for_row(fk, off + r);
				if (!sp || !dp) {
					ok = 0;
					break;
				}
				copy_weight_row(sp, r, dp, off + r, fk->k_len);
			}
			off += ws[i]->n_pad;
		}
	}
	if (!ok || !rkt_flush_weight(p, f)) {
		rkt_release_npu_state(p, f);
		free(f);
		return NULL;
	}

	rkt_register_weight(p, f);
	int off = 0;
	for (int i = 0; i < count; i++) {
		rkt_release_npu_state(p, ws[i]);
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
	status_code s = p->npu_failed ? ERR_COMPUTE_FAIL : rkt_dense_matmul(p, w, row0, n, xf, yf, m);
	if (s != OK && !p->npu_failed) {
		rkt_npu_record_failure(p);
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
	if (!rkt_ensure_capacity((void **)&p->y_tmp, &p->y_tmp_cap, (size_t)m * span, sizeof(float))) {
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

status_code rkt_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
								   const buffer *x, buffer **y, const int *n_list, int k,
								   int n_matmuls, int m) {
	rkt_priv *p = self->priv;
	if (m == 1 && !p->quant_failed && !p->npu_failed && n_matmuls <= RKT_QUANT_MAX_CHILDREN) {
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
			status_code s4 = rkt_quant_run_many(p, ws, ys, count, rkt_ptr(x), 1);
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
			rkt_quant_disable(p, s4);
		}
	}
	if (n_matmuls >= 2) {
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

status_code rkt_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
							 const buffer *x, buffer **y, const int *n_list, int k, int n_matmuls) {
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

static int residual_ew_eligible(const rkt_priv *p, const rkt_weight *w, int use_quant) {
	return !p->ew_residual_failed && !p->npu_failed && !use_quant && !w->host_copy &&
		   w->fmt == RKT_FMT_F16 && w->n_kchunks > 0;
}

static status_code residual_to_f16(rkt_priv *p, const float *r, int n) {
	if (!rkt_ensure_capacity((void **)&p->res16, &p->res16_cap, (size_t)n, sizeof(uint16_t)))
		return ERR_OUT_OF_MEMORY;
	for (int i = 0; i < n; i++) {
		if (!(fabsf(r[i]) <= RKT_RESIDUAL_MAX))
			return ERR_UNSUPPORTED;
		p->res16[i] = f32_to_f16_bits(r[i]);
	}
	return OK;
}

static status_code residual_ew_matmul(rkt_priv *p, rkt_weight *wt, int row0, int n, int k,
									  const buffer *x, const buffer *residual, buffer *y) {
	int			 native = rkt_native_matches(p, x, wt, k, 1);
	const float *xf		= native ? NULL : rkt_ptr(x);
	float		*yf		= rkt_ptr(y);
	const float *r		= rkt_ptr(residual);
	float		*rcopy	= NULL;
	if (r == yf) {
		rcopy = xmalloc((size_t)n * sizeof(float));
		memcpy(rcopy, r, (size_t)n * sizeof(float));
		r = rcopy;
	}

	pthread_mutex_lock(&p->lock);
	status_code s = residual_to_f16(p, r, n);
	if (s == OK) {
		p->ew_res16 = p->res16;
		s			= native ? rkt_dense_matmul_native(p, wt, row0, n, yf, 1)
							 : rkt_dense_matmul(p, wt, row0, n, xf, yf, 1);
		p->ew_res16 = NULL;
		if (s != OK) {
			WARN("rocket: npu ew residual add failed (n=%d k=%d status=%d), using fold path", n, k,
				 s);
			p->ew_residual_failed = 1;
		}
	}
	if (s == OK && native)
		rkt_native_release(p);
	if (s != OK) {
		if (rcopy)
			memcpy(yf, rcopy, (size_t)n * sizeof(float));
		if (native)
			rkt_native_settle();
	}
	pthread_mutex_unlock(&p->lock);
	free(rcopy);
	return s;
}

status_code rkt_matmul_residual(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								const buffer *residual, buffer *y, int n, int k) {
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
	int use_quant = !p->quant_failed && !p->npu_failed && wt->quant && row0 == 0 && n == wt->n;
	if (wt->fused_parent) {
		row0 += wt->fused_row0;
		wt = wt->fused_parent;
	}
	if (residual_ew_eligible(p, wt, use_quant) && rkt_ptr(residual) != NULL &&
		residual_ew_matmul(p, wt, row0, n, k, x, residual, y) == OK)
		return OK;
	if (use_quant || wt->host_copy || wt->fmt != RKT_FMT_F16 || wt->n_kchunks != 1 ||
		wt->kchunks[0].k_len + RKT_FOLD_K_LEN > RKT_MAX_K || wt->k_pad != wt->k ||
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

	int			 native = rkt_native_matches(p, x, wt, k, 1);
	const float *xf		= native ? NULL : rkt_ptr(x);
	float		*yf		= rkt_ptr(y);
	const float *r		= rkt_ptr(residual);
	float		*rcopy	= NULL;
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
				if (rkt_ensure_capacity((void **)&p->x16, &p->x16_cap,
										(size_t)1 * (wt->k + RKT_FOLD_K_LEN), sizeof(uint16_t))) {
					if (native)
						memcpy(p->x16, p->nat_act, (size_t)k * sizeof(uint16_t));
					else
						rkt_convert_to_f16(p->x16, xf, (size_t)k);
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
		rkt_npu_record_failure(p);
		WARN("rocket: npu residual-fold matmul failed (n=%d k=%d status=%d, errno=%d), "
			 "using cpu",
			 n, k, s, errno);
	}
	if (s == OK && native)
		rkt_native_release(p);
	if (s != OK && native) {
		rkt_native_settle();
		xf = rkt_ptr_raw(x);
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
