#ifndef ROCKET_EMIT_H
#define ROCKET_EMIT_H

#include <stdint.h>

#define RKT_REGCMD_BYTES 4096

typedef enum { RKT_FLOAT16, RKT_INT8 } rkt_precision;

int rkt_emit_matmul(uint64_t *ops, rkt_precision precision, int m, int k, int n, uint32_t in_dma,
					uint32_t w_dma, uint32_t out_dma);

#endif
