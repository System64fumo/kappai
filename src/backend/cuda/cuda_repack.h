#ifndef KAPPAI_CUDA_REPACK_H
#define KAPPAI_CUDA_REPACK_H

#include <stdint.h>

/* Host-side load-time weight conversion used by the CUDA backend.
 *
 * This lives in the CUDA backend (not the CPU quant module) so the CPU and
 * Vulkan backends stay completely untouched.
 *
 * Declarations are plain C (no CUDA headers). */

/* Lossless Q4_0 -> Q8_0 promotion. Values map exactly (qs=(nib-8)*2, s=d/2),
 * so a Q4_0 tensor can run on the faster Q8_0 kernels with identical
 * numerics. Returns an aligned malloc'd buffer (caller frees) or NULL. */
void *cuda_convert_q4_0_to_q8_0(const void *src, int n_rows, int k);

#endif /* KAPPAI_CUDA_REPACK_H */
