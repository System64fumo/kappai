#ifndef ROCKET_QUANT_H
#define ROCKET_QUANT_H

#include "backend/cpu/quants.h"

#define RKT_INT8_K 512
#define RKT_ROTATION_BLOCK 512

typedef struct {
	const q8_0_block *source;
	const float		 *rotation;
	int8_t			 *data;
	float			 *scales;
	int				  n, k, first, cols, tile_k, first_chunk, chunks;
} rkt_weight_pack;

void  rkt_rotation_factors(float *out, int k);
float rkt_half_to_float(uint16_t bits);
int	  rkt_pack_weight_i8(const rkt_weight_pack *pack, int row);
int	  rkt_pack_input_i8(const float *x, int8_t *dest, float *scales, int k, int m, int row,
						int tile_k, const float *rotation);

#endif
