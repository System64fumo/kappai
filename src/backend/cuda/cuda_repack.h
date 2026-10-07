#ifndef KAPPAI_CUDA_REPACK_H
#define KAPPAI_CUDA_REPACK_H

#include <stdint.h>

/* Host-side load-time weight relayouts used by the CUDA backend.
 *
 * These live in the CUDA backend (not the CPU quant module) so the CPU and
 * Vulkan backends stay completely untouched: the quad-major (QM) layouts
 * only exist to feed the CUDA GEMV kernels, and nothing else consumes them.
 *
 * Declarations are plain C (no CUDA headers) so shared code such as model.c
 * can call them under #ifdef BACKEND_CUDA. */

/* Quad-major Q8_0: in-place-compatible, size-preserving relayout of rows
 * [row_begin, row_end). Byte-exact spec in cuda_internal.h. */
void cuda_repack_q8_0_qm_rows(const void *src, void *dst, int row_begin, int row_end, int k);

/* Quad-major Q4_0: size-preserving relayout (2B scales x G, then 16
 * byte-planes x G per 32-block group). */
void cuda_repack_q4_0_qm_rows(const void *src, void *dst, int row_begin, int row_end, int k);

/* Lossless Q4_0 -> Q8_0 promotion. Values map exactly (qs=(nib-8)*2, s=d/2),
 * so a Q4_0 tensor can run on the faster Q8_0 kernels with identical
 * numerics. Returns an aligned malloc'd buffer (caller frees) or NULL. */
void *cuda_convert_q4_0_to_q8_0(const void *src, int n_rows, int k);

#endif /* KAPPAI_CUDA_REPACK_H */
