#include "emit.h"
#include "hw.h"

#include <string.h>

typedef struct {
	uint64_t *ops;
	int		  n;
} regcmd;

static void put(regcmd *c, uint32_t block_op, uint32_t reg, uint32_t value) {
	c->ops[c->n++] =
		((uint64_t)(block_op & 0xFFFF) << 48) | ((uint64_t)value << 16) | (reg & 0xFFFF);
}

static void put_zeros(regcmd *c, uint32_t block_op, uint32_t first_reg, int count) {
	for (int i = 0; i < count; i++)
		put(c, block_op, first_reg + (uint32_t)i * 4, 0);
}

static uint32_t pack(uint32_t hi, uint32_t lo) {
	return (hi << 16) | lo;
}

int rkt_feature_index(int height, int c2, int channel, int row) {
	return ((channel - 1) / c2) * height * c2 + c2 * (row - 1) + (channel - 1) % c2;
}

static uint32_t feature_banks(int m, int k) {
	uint32_t banks = ((uint32_t)(m * k * 2) + CBUF_BANK_BYTES - 1) / CBUF_BANK_BYTES;
	if (banks < 1)
		return 1;
	return banks > CBUF_BANKS - 1 ? CBUF_BANKS - 1 : banks;
}

static int surface_stride(int m) {
	int stride = 4 * (m / 4 - 1);
	return stride < 0 ? stride + 1 : stride;
}

static void emit_cna(regcmd *c, int m, int k, int n, uint32_t in_dma, uint32_t w_dma) {
	uint32_t fd_banks = feature_banks(m, k);
	uint32_t w_banks  = CBUF_BANKS - fd_banks;

	put(c, OP_REG_CNA, CNA_CONV_CON1, (PREC_FLOAT16 << 7) | (PREC_FLOAT16 << 4));
	put(c, OP_REG_CNA, CNA_CONV_CON2, ((uint32_t)(m + 1) & 0x3FF) << 4);
	put(c, OP_REG_CNA, CNA_CONV_CON3, (1 << 3) | 1);
	put(c, OP_REG_CNA, CNA_DATA_SIZE0, pack(1, (uint32_t)m & 0x7FF));
	put(c, OP_REG_CNA, CNA_DATA_SIZE1, pack((uint32_t)(k - 1) & 0xFFFF, (uint32_t)k & 0xFFFF));
	put(c, OP_REG_CNA, CNA_DATA_SIZE2, 1);
	put(c, OP_REG_CNA, CNA_DATA_SIZE3, (uint32_t)m & 0x3FFFF);
	put(c, OP_REG_CNA, CNA_WEIGHT_SIZE0, (uint32_t)(k * 2 * n));
	put(c, OP_REG_CNA, CNA_WEIGHT_SIZE1, (uint32_t)(k * 2) & 0x7FFFF);
	put(c, OP_REG_CNA, CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | ((uint32_t)n & 0x3FFF));
	put(c, OP_REG_CNA, CNA_CBUF_CON0, ((w_banks & 0xF) << 4) | (fd_banks & 0xF));
	put(c, OP_REG_CNA, CNA_CBUF_CON1, (uint32_t)((k + 31) / 32) & 0x1FFF);
	put(c, OP_REG_CNA, CNA_CVT_CON0, (1 << 3) | (1 << 1) | 1);
	put(c, OP_REG_CNA, CNA_CVT_CON1, 1u << 16);
	put(c, OP_REG_CNA, CNA_CVT_CON2, 1u << 16);
	put(c, OP_REG_CNA, CNA_CVT_CON3, 1u << 16);
	put(c, OP_REG_CNA, CNA_CVT_CON4, 1u << 16);
	put(c, OP_REG_CNA, CNA_FC_CON0, 0);
	put(c, OP_REG_CNA, CNA_FC_CON1, 0);
	put(c, OP_REG_CNA, CNA_PAD_CON0, 0);
	put(c, OP_REG_CNA, CNA_FEATURE_DATA_ADDR, in_dma);
	put(c, OP_REG_CNA, CNA_FC_CON2, 0);
	put(c, OP_REG_CNA, CNA_DMA_CON0, pack(0xF, 0xF));
	put(c, OP_REG_CNA, CNA_DMA_CON1, 4);
	put(c, OP_REG_CNA, CNA_DMA_CON2, (uint32_t)surface_stride(m) & 0xFFFFFFF);
	put(c, OP_REG_CNA, CNA_FC_DATA_SIZE0, pack(1, (uint32_t)m & 0x7FF));
	put(c, OP_REG_CNA, CNA_FC_DATA_SIZE1, (uint32_t)k & 0xFFFF);
	put(c, OP_REG_CNA, CNA_DCOMP_CTRL, 0);
	put(c, OP_REG_CNA, CNA_DCOMP_REGNUM, 0);
	put(c, OP_REG_CNA, CNA_DCOMP_ADDR0, w_dma);
	put_zeros(c, OP_REG_CNA, CNA_DCOMP_AMOUNT, 16);
	put(c, OP_REG_CNA, CNA_CVT_CON5, 0);
	put(c, OP_REG_CNA, CNA_PAD_CON1, 0);
}

static void emit_core(regcmd *c, int m, int n) {
	put(c, OP_REG_CORE, CORE_MISC_CFG, (PREC_FLOAT16 << 8) | 1);
	put(c, OP_REG_CORE, CORE_DATAOUT_SIZE_0, pack((uint32_t)(m - 1) & 0xFFFF, 0));
	put(c, OP_REG_CORE, CORE_DATAOUT_SIZE_1, (uint32_t)(n - 1) & 0xFFFF);
	put(c, OP_REG_CORE, CORE_CLIP_TRUNCATE, 0);
	put(c, OP_REG_CORE, CORE_3030, 0);
}

static void emit_dpu_output(regcmd *c, int m, int n, uint32_t out_dma) {
	uint32_t channel = (uint32_t)(n - 1) & 0x1FFF;

	put(c, OP_REG_DPU, DPU_FEATURE_MODE_CFG, (0xF << 5) | (2 << 1));
	put(c, OP_REG_DPU, DPU_DATA_FORMAT,
		((uint32_t)PREC_FLOAT32 << 29) | ((uint32_t)PREC_FLOAT16 << 26) | PREC_FLOAT16);
	put(c, OP_REG_DPU, DPU_OFFSET_PEND, 0);
	put(c, OP_REG_DPU, DPU_DST_BASE_ADD, out_dma);
	put(c, OP_REG_DPU, DPU_DST_SURF_STRIDE, ((uint32_t)m & 0xFFFFFFF) << 4);
	put(c, OP_REG_DPU, DPU_DATA_CUBE_WIDTH, 0);
	put(c, OP_REG_DPU, DPU_DATA_CUBE_HEIGHT, (uint32_t)(m - 1) & 0x1FFF);
	put(c, OP_REG_DPU, DPU_DATA_CUBE_NOTCH_ADDR, 0);
	put(c, OP_REG_DPU, DPU_DATA_CUBE_CHANNEL, pack(channel, channel));
}

static void emit_dpu_bypass(regcmd *c, int m, int n) {
	uint32_t bypass_all = (1 << 6) | (1 << 4) | (1 << 1) | 1;

	put(c, OP_REG_DPU, DPU_BS_CFG, bypass_all);
	put(c, OP_REG_DPU, DPU_BS_ALU_CFG, 0);
	put(c, OP_REG_DPU, DPU_BS_MUL_CFG, 0);
	put(c, OP_REG_DPU, DPU_BS_RELUX_CMP_VALUE, 0);
	put(c, OP_REG_DPU, DPU_BS_OW_CFG, (3 << 8) | (3 << 5) | (3 << 2) | (1 << 1));
	put(c, OP_REG_DPU, DPU_BS_OW_OP, 0);
	put(c, OP_REG_DPU, DPU_WDMA_SIZE_0, (uint32_t)(n - 1) & 0x1FFF);
	put(c, OP_REG_DPU, DPU_WDMA_SIZE_1, pack((uint32_t)(m - 1) & 0x1FFF, 0));
	put(c, OP_REG_DPU, DPU_BN_CFG, bypass_all);
	put(c, OP_REG_DPU, DPU_BN_ALU_CFG, 0);
	put(c, OP_REG_DPU, DPU_BN_MUL_CFG, 0);
	put(c, OP_REG_DPU, DPU_BN_RELUX_CMP_VALUE, 0);
	put(c, OP_REG_DPU, DPU_EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1);
	put(c, OP_REG_DPU, DPU_EW_CVT_OFFSET_VALUE, 0);
	put(c, OP_REG_DPU, DPU_EW_CVT_SCALE_VALUE, 1);
	put(c, OP_REG_DPU, DPU_EW_RELUX_CMP_VALUE, 0);
	put(c, OP_REG_DPU, DPU_OUT_CVT_OFFSET, 0);
	put(c, OP_REG_DPU, DPU_OUT_CVT_SCALE, 1);
	put(c, OP_REG_DPU, DPU_OUT_CVT_SHIFT, 0);
	put_zeros(c, OP_REG_DPU, DPU_EW_OP_VALUE_0, 8);
	put(c, OP_REG_DPU, DPU_SURFACE_ADD, ((uint32_t)(m * 4) & 0xFFFFFFF) << 4);
	put(c, OP_REG_DPU, DPU_40C4, 0);
}

static void emit_dpu_lut_idle(regcmd *c) {
	static const uint32_t regs[] = {
		DPU_LUT_ACCESS_CFG,		DPU_LUT_ACCESS_DATA,	DPU_LUT_CFG,
		DPU_LUT_INFO,			DPU_LUT_LE_START,		DPU_LUT_LE_END,
		DPU_LUT_LO_START,		DPU_LUT_LO_END,			DPU_LUT_LE_SLOPE_SCALE,
		DPU_LUT_LE_SLOPE_SHIFT, DPU_LUT_LO_SLOPE_SCALE, DPU_LUT_LO_SLOPE_SHIFT,
	};
	for (unsigned i = 0; i < sizeof(regs) / sizeof(regs[0]); i++)
		put(c, OP_REG_DPU, regs[i], 0);
}

int rkt_emit_matmul_f16(uint64_t *ops, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
						uint32_t out_dma) {
	regcmd c = {.ops = ops};

	put(&c, OP_REG_DPU, DPU_S_POINTER, 0xE);
	emit_cna(&c, m, k, n, in_dma, w_dma);
	emit_core(&c, m, n);
	emit_dpu_output(&c, m, n, out_dma);
	emit_dpu_bypass(&c, m, n);
	emit_dpu_lut_idle(&c);
	put(&c, OP_NONE, 0, 0);
	put(&c, OP_REG_PC, PC_REGISTER_AMOUNTS, 0);
	put(&c, OP_40, 0, 0);
	put(&c, OP_ENABLE, PC_OPERATION_ENABLE, PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE);
	return c.n;
}

static uint64_t *find_op(uint64_t *ops, int n, uint32_t block_op, uint32_t reg) {
	for (int i = 0; i < n; i++) {
		uint64_t w = ops[i];
		if ((uint32_t)(w >> 48) == block_op && (uint32_t)(w & 0xFFFFu) == reg)
			return &ops[i];
	}
	return NULL;
}

static void set_op(uint64_t *ops, int n, uint32_t block_op, uint32_t reg, uint32_t value) {
	uint64_t *slot = find_op(ops, n, block_op, reg);
	if (slot)
		*slot = ((uint64_t)block_op << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
}

static uint32_t f32_bits(float v) {
	uint32_t bits;
	memcpy(&bits, &v, sizeof(bits));
	return bits;
}

static uint16_t f32_to_f16_bits(float v) {
	_Float16 h = (_Float16)v;
	uint16_t bits;
	memcpy(&bits, &h, sizeof(bits));
	return bits;
}

int rkt_emit_matmul_f16_act(uint64_t *ops, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
							uint32_t out_dma, const rkt_post_cfg *post) {
	int n_ops = rkt_emit_matmul_f16(ops, m, k, n, in_dma, w_dma, out_dma);
	if (!post || post->act == RKT_ACT_NONE)
		return n_ops;

	uint32_t bs = 0, alu = 0, mul = 0, cmp = 0;
	switch (post->act) {
	case RKT_ACT_RELU:
		bs = DPU_BS_CFG_ALU_BYPASS(1) | DPU_BS_CFG_MUL_BYPASS(1);
		break;
	case RKT_ACT_RELUX:
		bs	= DPU_BS_CFG_ALU_BYPASS(1) | DPU_BS_CFG_MUL_BYPASS(1) | DPU_BS_CFG_RELUX_EN(1);
		cmp = f32_bits(post->relux_max);
		break;
	case RKT_ACT_PRELU:
		bs	= DPU_BS_CFG_ALU_BYPASS(1) | DPU_BS_CFG_RELU_BYPASS(1) | DPU_BS_CFG_MUL_PRELU(1);
		mul = (uint32_t)f32_to_f16_bits(post->prelu_alpha) << 16;
		break;
	case RKT_ACT_ADDC:
		bs	= DPU_BS_CFG_ALU_ALGO(2) | DPU_BS_CFG_MUL_BYPASS(1) | DPU_BS_CFG_RELU_BYPASS(1);
		alu = f32_bits(post->addc);
		break;
	case RKT_ACT_MULC:
		bs	= DPU_BS_CFG_ALU_BYPASS(1) | DPU_BS_CFG_RELU_BYPASS(1);
		mul = (uint32_t)f32_to_f16_bits(post->mulc) << 16;
		break;
	case RKT_ACT_AFFINE:
		bs	= DPU_BS_CFG_ALU_ALGO(2) | DPU_BS_CFG_RELU_BYPASS(1);
		alu = f32_bits(post->addc);
		mul = (uint32_t)f32_to_f16_bits(post->mulc) << 16;
		break;
	case RKT_ACT_SUBC:
		bs	= DPU_BS_CFG_ALU_ALGO(4) | DPU_BS_CFG_MUL_BYPASS(1) | DPU_BS_CFG_RELU_BYPASS(1);
		alu = f32_bits(post->addc);
		break;
	case RKT_ACT_RELU6_ALU:
		bs	= DPU_BS_CFG_ALU_ALGO(2) | DPU_BS_CFG_RELUX_EN(1) | DPU_BS_CFG_MUL_BYPASS(1);
		alu = f32_bits(post->addc);
		cmp = f32_bits(post->relux_max);
		break;
	case RKT_ACT_RELU6_MUL:
		bs	= DPU_BS_CFG_ALU_BYPASS(1) | DPU_BS_CFG_RELUX_EN(1);
		mul = (uint32_t)f32_to_f16_bits(post->mulc) << 16;
		cmp = f32_bits(post->relux_max);
		break;
	case RKT_ACT_MUL_RELU:
		bs	= DPU_BS_CFG_ALU_BYPASS(1);
		mul = (uint32_t)f32_to_f16_bits(post->mulc) << 16;
		break;
	default:
		return n_ops;
	}
	set_op(ops, n_ops, OP_REG_DPU, DPU_BS_CFG, bs);
	set_op(ops, n_ops, OP_REG_DPU, DPU_BS_ALU_CFG, alu);
	set_op(ops, n_ops, OP_REG_DPU, DPU_BS_MUL_CFG, mul);
	set_op(ops, n_ops, OP_REG_DPU, DPU_BS_RELUX_CMP_VALUE, cmp);
	return n_ops;
}

int rkt_emit_matmul_f16_lut(uint64_t *ops, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
							uint32_t out_dma, const rkt_lut_cfg *cfg) {
	int n_ops = rkt_emit_matmul_f16(ops, m, k, n, in_dma, w_dma, out_dma);
	if (!cfg || cfg->kind == RKT_LUT_NONE)
		return n_ops;

	set_op(ops, n_ops, OP_REG_DPU, DPU_EW_CFG, 0x302);
	set_op(ops, n_ops, OP_REG_DPU, DPU_LUT_CFG, DPU_LUT_CFG_ROAD_SEL(0));
	set_op(ops, n_ops, OP_REG_DPU, DPU_LUT_INFO, DPU_LUT_INFO_LE_INDEX_SELECT((uint32_t)cfg->sel));
	set_op(ops, n_ops, OP_REG_DPU, DPU_LUT_LE_START, (uint32_t)cfg->start);
	set_op(ops, n_ops, OP_REG_DPU, DPU_LUT_LE_END, (uint32_t)cfg->end);
	set_op(ops, n_ops, OP_REG_DPU, DPU_LUT_LE_SLOPE_SCALE, cfg->tail_slope << 16);
	set_op(ops, n_ops, OP_REG_DPU, DPU_LUT_LE_SLOPE_SHIFT, 0);
	return n_ops;
}

int rkt_emit_lut_upload(uint64_t *ops, int table, int start, const uint16_t *words, int count,
						uint32_t in_dma, uint32_t w_dma, uint32_t out_dma) {
	if (count < 1 || count > RKT_LUT_UPLOAD_CHUNK)
		return 0;
	int n	  = rkt_emit_matmul_f16(ops, 1, RKT_K_ALIGN, RKT_N_ALIGN, in_dma, w_dma, out_dma);
	int extra = count + 1;
	if (n + extra > RKT_REGCMD_MAX_OPS)
		return 0;

	memmove(ops + n - 4 + extra, ops + n - 4, 4 * sizeof(uint64_t));
	int idx = n - 4;
	ops[idx++] =
		((uint64_t)OP_REG_DPU << 48) |
		((uint64_t)(DPU_LUT_ACCESS_CFG_TYPE(1) | DPU_LUT_ACCESS_CFG_TABLE_ID((uint32_t)table) |
					DPU_LUT_ACCESS_CFG_ADDR((uint32_t)start))
		 << 16) |
		(uint64_t)DPU_LUT_ACCESS_CFG;
	for (int i = 0; i < count; i++)
		ops[idx++] = ((uint64_t)OP_REG_DPU << 48) | ((uint64_t)words[i] << 16) |
					 (uint64_t)DPU_LUT_ACCESS_DATA;
	return n + extra;
}
