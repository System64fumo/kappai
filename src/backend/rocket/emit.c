#include "emit.h"
#include "hw.h"

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

static uint32_t feature_banks(int m, int k, int bytes) {
	uint32_t banks = ((uint32_t)(m * k * bytes) + CBUF_BANK_BYTES - 1) / CBUF_BANK_BYTES;
	if (banks < 1)
		return 1;
	return banks > CBUF_BANKS - 1 ? CBUF_BANKS - 1 : banks;
}

static int surface_stride(int m) {
	int stride = 4 * (m / 4 - 1);
	return stride < 0 ? stride + 1 : stride;
}

static void emit_cna(regcmd *c, rkt_precision precision, int m, int k, int n, uint32_t in_dma,
					 uint32_t w_dma) {
	int		 bytes	  = precision == RKT_INT8 ? 1 : 2;
	uint32_t prec	  = precision == RKT_INT8 ? PREC_INT8 : PREC_FLOAT16;
	uint32_t fd_banks = feature_banks(m, k, bytes);
	uint32_t w_banks  = CBUF_BANKS - fd_banks;

	put(c, OP_REG_CNA, CNA_CONV_CON1, (prec << 7) | (prec << 4));
	put(c, OP_REG_CNA, CNA_CONV_CON2, ((uint32_t)(m + 1) & 0x3FF) << 4);
	put(c, OP_REG_CNA, CNA_CONV_CON3, (1 << 3) | 1);
	put(c, OP_REG_CNA, CNA_DATA_SIZE0, pack(1, (uint32_t)m & 0x7FF));
	put(c, OP_REG_CNA, CNA_DATA_SIZE1, pack((uint32_t)(k - 1) & 0xFFFF, (uint32_t)k & 0xFFFF));
	put(c, OP_REG_CNA, CNA_DATA_SIZE2, 1);
	put(c, OP_REG_CNA, CNA_DATA_SIZE3, (uint32_t)m & 0x3FFFF);
	put(c, OP_REG_CNA, CNA_WEIGHT_SIZE0, (uint32_t)(k * bytes * n));
	put(c, OP_REG_CNA, CNA_WEIGHT_SIZE1, (uint32_t)(k * bytes) & 0x7FFFF);
	put(c, OP_REG_CNA, CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | ((uint32_t)n & 0x3FFF));
	put(c, OP_REG_CNA, CNA_CBUF_CON0, ((w_banks & 0xF) << 4) | (fd_banks & 0xF));
	put(c, OP_REG_CNA, CNA_CBUF_CON1, (uint32_t)((k + 64 / bytes - 1) / (64 / bytes)) & 0x1FFF);
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

static void emit_core(regcmd *c, rkt_precision precision, int m, int n) {
	uint32_t prec = precision == RKT_INT8 ? PREC_INT8 : PREC_FLOAT16;
	put(c, OP_REG_CORE, CORE_MISC_CFG, (prec << 8) | 1);
	put(c, OP_REG_CORE, CORE_DATAOUT_SIZE_0, pack((uint32_t)(m - 1) & 0xFFFF, 0));
	put(c, OP_REG_CORE, CORE_DATAOUT_SIZE_1, (uint32_t)(n - 1) & 0xFFFF);
	put(c, OP_REG_CORE, CORE_CLIP_TRUNCATE, 0);
	put(c, OP_REG_CORE, CORE_3030, 0);
}

static void emit_dpu_output(regcmd *c, rkt_precision precision, int m, int n, uint32_t out_dma) {
	uint32_t prec	  = precision == RKT_INT8 ? PREC_INT8 : PREC_FLOAT16;
	uint32_t out_prec = precision == RKT_INT8 ? PREC_INT32 : PREC_FLOAT32;
	uint32_t channel  = (uint32_t)(n - 1) & 0x1FFF;

	put(c, OP_REG_DPU, DPU_FEATURE_MODE_CFG, (0xF << 5) | (2 << 1));
	put(c, OP_REG_DPU, DPU_DATA_FORMAT, (out_prec << 29) | (prec << 26) | prec);
	put(c, OP_REG_DPU, DPU_OFFSET_PEND, 0);
	put(c, OP_REG_DPU, DPU_DST_BASE_ADD, out_dma);
	put(c, OP_REG_DPU, DPU_DST_SURF_STRIDE, ((uint32_t)m & 0xFFFFFFF) << 4);
	put(c, OP_REG_DPU, DPU_DATA_CUBE_WIDTH, 0);
	put(c, OP_REG_DPU, DPU_DATA_CUBE_HEIGHT, (uint32_t)(m - 1) & 0x1FFF);
	put(c, OP_REG_DPU, DPU_DATA_CUBE_NOTCH_ADDR, 0);
	put(c, OP_REG_DPU, DPU_DATA_CUBE_CHANNEL, pack(channel, channel));
}

static void emit_dpu_bypass(regcmd *c, rkt_precision precision, int m, int n) {
	uint32_t ow			   = precision == RKT_INT8 ? 7 : 3;
	int		 surface_bytes = precision == RKT_INT8 ? 8 : 4;
	uint32_t bypass_all	   = (1 << 6) | (1 << 4) | (1 << 1) | 1;

	put(c, OP_REG_DPU, DPU_BS_CFG, bypass_all);
	put(c, OP_REG_DPU, DPU_BS_ALU_CFG, 0);
	put(c, OP_REG_DPU, DPU_BS_MUL_CFG, 0);
	put(c, OP_REG_DPU, DPU_BS_RELUX_CMP_VALUE, 0);
	put(c, OP_REG_DPU, DPU_BS_OW_CFG, (ow << 8) | (ow << 5) | (ow << 2) | (1 << 1));
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
	put(c, OP_REG_DPU, DPU_SURFACE_ADD, ((uint32_t)(m * surface_bytes) & 0xFFFFFFF) << 4);
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

int rkt_emit_matmul(uint64_t *ops, rkt_precision precision, int m, int k, int n, uint32_t in_dma,
					uint32_t w_dma, uint32_t out_dma) {
	regcmd c = {.ops = ops};

	put(&c, OP_REG_DPU, DPU_S_POINTER, 0xE);
	emit_cna(&c, precision, m, k, n, in_dma, w_dma);
	emit_core(&c, precision, m, n);
	emit_dpu_output(&c, precision, m, n, out_dma);
	emit_dpu_bypass(&c, precision, m, n);
	emit_dpu_lut_idle(&c);
	put(&c, OP_NONE, 0, 0);
	put(&c, OP_REG_PC, PC_REGISTER_AMOUNTS, 0);
	put(&c, OP_40, 0, 0);
	put(&c, OP_ENABLE, PC_OPERATION_ENABLE, PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE);
	return c.n;
}
