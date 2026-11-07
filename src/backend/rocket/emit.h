#ifndef ROCKET_EMIT_H
#define ROCKET_EMIT_H

#include <stdint.h>

#define RKT_PRECISION_INT8 0
#define RKT_PRECISION_INT4 6
#define RKT_PRECISION_INT32 4
#define RKT_REGCMD_BYTES 4096
#define RKT_REGCMD_MAX_OPS 512
#define RKT_FEATURE_C2_F16 8
#define RKT_OUT_C2_F32 4
#define RKT_N_ALIGN 16
#define RKT_K_ALIGN 32
#define RKT_MAX_N_CHUNK 8192
#define RKT_MAX_K 8192

typedef enum {
	RKT_ACT_NONE = 0,
	RKT_ACT_RELU,
	RKT_ACT_RELUX,
	RKT_ACT_PRELU,
	RKT_ACT_ADDC,
	RKT_ACT_MULC,
	RKT_ACT_AFFINE,
	RKT_ACT_SUBC,
	RKT_ACT_RELU6_ALU,
	RKT_ACT_RELU6_MUL,
	RKT_ACT_MUL_RELU,
} rkt_post_act;

typedef struct {
	rkt_post_act act;
	float		 relux_max;
	float		 prelu_alpha;
	float		 addc;
	float		 mulc;
} rkt_post_cfg;

typedef enum {
	RKT_LUT_NONE = 0,
	RKT_LUT_GELU_TANH,
	RKT_LUT_SILU,
} rkt_lut_kind;

#define RKT_LUT_WINDOW_HALF 256
#define RKT_LUT_ENTRIES 512
#define RKT_LUT_UPLOAD_CHUNK 128

typedef struct {
	rkt_lut_kind kind;
	int			 sel;
	uint32_t	 S;
	uint32_t	 F;
	int32_t		 start;
	int32_t		 end;
	uint32_t	 tail_slope;
} rkt_lut_cfg;

int rkt_emit_matmul_f16(uint64_t *ops, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
						uint32_t out_dma);

int rkt_emit_quant(uint64_t *ops, int bits, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
				   uint32_t out_dma);

int rkt_emit_matmul_f16_act(uint64_t *ops, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
							uint32_t out_dma, const rkt_post_cfg *post);

int rkt_emit_matmul_f16_lut(uint64_t *ops, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
							uint32_t out_dma, const rkt_lut_cfg *cfg);

int rkt_emit_matmul_f16_ewadd(uint64_t *ops, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
							  uint32_t out_dma, uint32_t operand_dma);

int rkt_ew_operand_index(int m, int row, int channel);

int rkt_emit_lut_upload(uint64_t *ops, int table, int start, const uint16_t *words, int count,
						uint32_t in_dma, uint32_t w_dma, uint32_t out_dma);

int rkt_feature_index(int height, int c2, int channel, int row);

#endif
