#ifndef CPU_AARCH64_COMMON_H
#define CPU_AARCH64_COMMON_H

#include "backend/cpu/common.h"
#include "numeric_neon.h"

#include <arm_neon.h>
#include <string.h>

#if defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
extern const cpu_matmul_packing cpu_q8_x4_packing;
#endif

#define CACHE_LINE_BYTES (g_cpu_cache.line_bytes)
#define L1D_SIZE_BYTES (g_cpu_cache.l1d_bytes)
#define L2_SIZE_BYTES (g_cpu_cache.l2_bytes)

#ifndef PREFETCH_LOCALITY
#define PREFETCH_LOCALITY 1
#endif

#define PREFETCH(addr) __builtin_prefetch((addr), 0, PREFETCH_LOCALITY)

#define PF_ELEMS(base_elems_64b_line) (((base_elems_64b_line) * (CACHE_LINE_BYTES)) / 64)

static inline float f16_to_f32_fast(uint16_t h) {
	__fp16 v;
	memcpy(&v, &h, sizeof(v));
	return (float)v;
}

static inline float32x4_t silu_mul_vec4(float32x4_t g, float32x4_t u) {
	return vmulq_f32(vsiluq_f32(g), u);
}

static inline float32x4_t gelu_mul_vec4(float32x4_t g, float32x4_t u) {
	return vmulq_f32(vgeluq_f32(g), u);
}

#endif
