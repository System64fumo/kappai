#ifndef KAPPAI_CUDA_REPACK_H
#define KAPPAI_CUDA_REPACK_H

#include <stdint.h>

void *cuda_convert_q4_0_to_q8_0(const void *src, int n_rows, int k);

#endif
