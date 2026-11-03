#ifndef ROCKET_EMIT_H
#define ROCKET_EMIT_H

#include <stdint.h>

#define RKT_REGCMD_BYTES 4096
#define RKT_REGCMD_MAX_OPS 512
#define RKT_FEATURE_C2_F16 8
#define RKT_OUT_C2_F32 4
#define RKT_N_ALIGN 16
#define RKT_K_ALIGN 32
#define RKT_MAX_N_CHUNK 8192
#define RKT_MAX_K 8192

int rkt_emit_matmul_f16(uint64_t *ops, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
						uint32_t out_dma);

int rkt_feature_index(int height, int c2, int channel, int row);

#endif
