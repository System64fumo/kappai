/* CUDA backend host-side weight relayouts (see cuda_repack.h).
 *
 * Deliberately a plain C translation unit: no CUDA runtime headers, so it is
 * compiled by the ordinary host compiler rule and can be called from shared
 * code. Keeps the CPU/Vulkan backends free of CUDA-only layout knowledge. */
#include "backend/cuda/cuda_repack.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "common.h"
#include "backend/cpu/scalar/quants.h"
#include "backend/cuda/cuda_internal.h"

/* f16 <-> f32 helpers.
 *
 * This translation unit is linked into the engine (model.c calls it), while
 * the scalar backend's cvt_f16_to_f32/cvt_f32_to_f16 now live in a dlopen'd
 * libkappai_cpu_scalar.so, so the engine cannot link against them. Keep
 * bit-exact local copies instead: cvt_f32_to_f16 mirrors the scalar reference
 * algorithm verbatim, and cvt_f16_to_f32 is the exact IEEE half expansion
 * (equal to the reference lookup table for all 65536 encodings). */
static inline uint32_t cvt_f32_to_bits(float f) {
	union {
		float	 f;
		uint32_t i;
	} v;
	v.f = f;
	return v.i;
}

static inline float cvt_f32_from_bits(uint32_t b) {
	union {
		uint32_t i;
		float	 f;
	} v;
	v.i = b;
	return v.f;
}

static inline float cvt_f16_to_f32(uint16_t h) {
	const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
	uint32_t	   exp	 = ((uint32_t)h >> 10) & 0x1Fu;
	uint32_t	   mant = (uint32_t)h & 0x03FFu;
	uint32_t	   u;
	if (exp == 0) {
		if (mant == 0) {
			u = sign;
		} else {
			/* subnormal half: renormalize into the float normal range */
			exp = 1;
			while ((mant & 0x0400u) == 0) {
				mant <<= 1;
				exp--;
			}
			mant &= 0x03FFu;
			u = sign | ((exp + 112) << 23) | (mant << 13);
		}
	} else if (exp == 0x1F) {
		u = sign | 0x7F800000u | (mant << 13);
	} else {
		u = sign | ((exp + 112) << 23) | (mant << 13);
	}
	return cvt_f32_from_bits(u);
}

static inline uint16_t cvt_f32_to_f16(float f) {
	const float	base	= (fabsf(f) * 0x1.0p+112f) * 0x1.0p-110f;
	const uint32_t w		= cvt_f32_to_bits(f);
	const uint32_t shl1_w = w + w;
	const uint32_t sign	= w & 0x80000000u;
	uint32_t		bias	= shl1_w & 0xFF000000u;
	if (bias < 0x71000000u)
		bias = 0x71000000u;
	const float		rounded = cvt_f32_from_bits((bias >> 1) + 0x07800000u) + base;
	const uint32_t bits		= cvt_f32_to_bits(rounded);
	const uint32_t exp_bits  = (bits >> 13) & 0x7C00u;
	const uint32_t mant_bits = bits & 0x0FFFu;
	const uint32_t nonsign   = exp_bits + mant_bits;
	return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

/* ---- Quad-major Q8_0 (34B/32e blocks) ------------------------------- */
/* ---- Quad-major Q4_0 (18B/32e blocks) ------------------------------- */
/* ---- Lossless Q4_0 -> Q8_0 promotion -------------------------------- */
void *cuda_convert_q4_0_to_q8_0(const void *src, int n_rows, int k) {
	const int	 nb	 = k / 32;
	q8_0_block	*dst = xmalloc_aligned((size_t)n_rows * nb * sizeof(q8_0_block), 64);
	if (!dst)
		return NULL;
	const q4_0_block *s = src;
	for (int r = 0; r < n_rows; r++) {
		for (int b = 0; b < nb; b++) {
			const q4_0_block *sb = s + (size_t)r * nb + b;
			q8_0_block		 *db = dst + (size_t)r * nb + b;
			float			  d	 = cvt_f16_to_f32(sb->d);
			db->d				 = cvt_f32_to_f16(d * 0.5f);
			/* Nibble order: low -> elems 0-15, high -> elems 16-31. */
			for (int j = 0; j < 16; j++) {
				uint8_t qb	   = sb->qs[j];
				db->qs[j]	   = (int8_t)(((qb & 0xF) << 1) - 16);
				db->qs[j + 16] = (int8_t)(((qb >> 4) << 1) - 16);
			}
		}
	}
	return dst;
}

/* ------------------------------------------------------------------ */
/* Host-allocation side table (see cuda_internal.h for why this exists) */
/* ------------------------------------------------------------------ */

/*
 * `buffer.host_ptr` may be cudaHostAlloc() memory, which must be released with
 * cudaFreeHost(), or malloc() memory, which must be released with free(). The
 * shared `buffer` struct carries no flag saying which, and this backend does not
 * edit that struct, so the answer is recorded here keyed by host pointer.
 *
 * cudaHostAlloc'd blocks are 64-byte aligned, so the low 6 bits of the pointer
 * are free and give a cheap bucket index without a modulo.
 */
static cuda_host_alloc *g_host_buckets[CUDA_HOST_ALLOC_BUCKETS];

static unsigned cuda_host_bucket(void *p) {
	return (unsigned)(((uintptr_t)p >> 6) % CUDA_HOST_ALLOC_BUCKETS);
}

void cuda_host_alloc_note(void *host_ptr, int pinned) {
	if (!host_ptr)
		return;
	cuda_host_alloc_forget(host_ptr);
	unsigned                 b = cuda_host_bucket(host_ptr);
	cuda_host_alloc *e = xmalloc(sizeof(*e));
	e->host_ptr	    = host_ptr;
	e->pinned		    = pinned;
	e->next		    = g_host_buckets[b];
	g_host_buckets[b] = e;
}

int cuda_host_alloc_pinned(void *host_ptr) {
	if (!host_ptr)
		return -1;
	for (cuda_host_alloc *e = g_host_buckets[cuda_host_bucket(host_ptr)]; e; e = e->next)
		if (e->host_ptr == host_ptr)
			return e->pinned;
	return -1;
}

void cuda_host_alloc_forget(void *host_ptr) {
	if (!host_ptr)
		return;
	unsigned                 b = cuda_host_bucket(host_ptr);
	cuda_host_alloc **link = &g_host_buckets[b];
	while (*link) {
		if ((*link)->host_ptr == host_ptr) {
			cuda_host_alloc *dead = *link;
			*link				 = dead->next;
			free(dead);
			return;
		}
		link = &(*link)->next;
	}
}

/* ------------------------------------------------------------------ */
/* Weight-type side table (see cuda_internal.h for why this exists)     */
/* ------------------------------------------------------------------ */

typedef struct cuda_weight_type {
	void					*dev_ptr;
	uint32_t				type;
	struct cuda_weight_type *next;
} cuda_weight_type;

static cuda_weight_type *g_type_buckets[CUDA_HOST_ALLOC_BUCKETS];

static unsigned cuda_type_bucket(const void *p) {
	return (unsigned)(((uintptr_t)p >> 4) % CUDA_HOST_ALLOC_BUCKETS);
}

void cuda_weight_type_note(void *dev_ptr, uint32_t type) {
	if (!dev_ptr)
		return;
	cuda_weight_type_forget(dev_ptr);
	unsigned               b = cuda_type_bucket(dev_ptr);
	cuda_weight_type *e = xmalloc(sizeof(*e));
	e->dev_ptr         = dev_ptr;
	e->type            = type;
	e->next            = g_type_buckets[b];
	g_type_buckets[b]  = e;
}

uint32_t cuda_weight_type_of(const void *dev_ptr) {
	if (!dev_ptr)
		return (uint32_t)-1;
	for (cuda_weight_type *e = g_type_buckets[cuda_type_bucket(dev_ptr)]; e; e = e->next)
		if (e->dev_ptr == dev_ptr)
			return e->type;
	return (uint32_t)-1;
}

void cuda_weight_type_forget(void *dev_ptr) {
	if (!dev_ptr)
		return;
	unsigned                b = cuda_type_bucket(dev_ptr);
	cuda_weight_type **link = &g_type_buckets[b];
	while (*link) {
		if ((*link)->dev_ptr == dev_ptr) {
			cuda_weight_type *dead = *link;
			*link                 = dead->next;
			free(dead);
			return;
		}
		link = &(*link)->next;
	}
}
