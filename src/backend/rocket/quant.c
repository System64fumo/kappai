#include "quant.h"
#include <math.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

float rkt_half_to_float(uint16_t bits) {
	_Float16 value;
	memcpy(&value, &bits, sizeof(value));
	return (float)value;
}

static void decode_block(const q8_0_block *block, float *out) {
	float d = rkt_half_to_float(block->d);
#if defined(__aarch64__)
	float32x4_t scale = vdupq_n_f32(d);
	for (int j = 0; j < 32; j += 8) {
		int16x8_t values = vmovl_s8(vld1_s8(block->qs + j));
		vst1q_f32(out + j, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(values))), scale));
		vst1q_f32(out + j + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(values))), scale));
	}
#else
	for (int j = 0; j < 32; j++)
		out[j] = d * block->qs[j];
#endif
}

void rkt_rotation_factors(float *out, int k) {
	int block = 1;
	while (block < RKT_ROTATION_BLOCK && k % (block * 2) == 0)
		block *= 2;
	float scale = 1.0f / sqrtf((float)block);
	for (int j = 0; j < k; j++) {
		uint32_t bits = (uint32_t)j + 0x9e3779b9u;
		bits ^= bits >> 16;
		bits *= 0x85ebca6bu;
		bits ^= bits >> 13;
		out[j] = bits & 1 ? -scale : scale;
	}
}

static void rotate_channels(float *values, int k, const float *rotation) {
	int block = 1;
	while (block < RKT_ROTATION_BLOCK && k % (block * 2) == 0)
		block *= 2;
#if defined(__aarch64__)
	for (int j = 0; j < k; j += 4)
		vst1q_f32(values + j, vmulq_f32(vld1q_f32(values + j), vld1q_f32(rotation + j)));
	for (int base = 0; base < k; base += block) {
		for (int j = 0; j < block; j += 8) {
			float32x4x2_t pair = vld2q_f32(values + base + j);
			float32x4x2_t out  = {
				 {vaddq_f32(pair.val[0], pair.val[1]), vsubq_f32(pair.val[0], pair.val[1])}};
			vst2q_f32(values + base + j, out);
		}
		for (int j = 0; j < block; j += 4) {
			float32x4_t pair = vld1q_f32(values + base + j);
			float32x2_t a = vget_low_f32(pair), b = vget_high_f32(pair);
			vst1q_f32(values + base + j, vcombine_f32(vadd_f32(a, b), vsub_f32(a, b)));
		}
		for (int half = 4; half < block; half *= 2)
			for (int start = 0; start < block; start += half * 2)
				for (int j = 0; j < half; j += 4) {
					float32x4_t a = vld1q_f32(values + base + start + j);
					float32x4_t b = vld1q_f32(values + base + start + half + j);
					vst1q_f32(values + base + start + j, vaddq_f32(a, b));
					vst1q_f32(values + base + start + half + j, vsubq_f32(a, b));
				}
	}
#else
	for (int j = 0; j < k; j++)
		values[j] *= rotation[j];
	for (int base = 0; base < k; base += block)
		for (int half = 1; half < block; half *= 2)
			for (int start = 0; start < block; start += half * 2)
				for (int j = 0; j < half; j++) {
					float a = values[base + start + j], b = values[base + start + half + j];
					values[base + start + j]		= a + b;
					values[base + start + half + j] = a - b;
				}
#endif
}

static float max_abs(const float *values, int n) {
#if defined(__aarch64__)
	float32x4_t maximum = vdupq_n_f32(0.0f);
	for (int j = 0; j < n; j += 4)
		maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(values + j)));
	return vmaxvq_f32(maximum);
#else
	float maximum = 0.0f;
	for (int j = 0; j < n; j++) {
		if (!isfinite(values[j]))
			return INFINITY;
		maximum = fmaxf(maximum, fabsf(values[j]));
	}
	return maximum;
#endif
}

static void quantize_16(const float *values, int8_t *out, float inverse) {
#if defined(__aarch64__)
	float32x4_t scale = vdupq_n_f32(inverse);
	int32x4_t	a	  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(values), scale));
	int32x4_t	b	  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(values + 4), scale));
	int32x4_t	c	  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(values + 8), scale));
	int32x4_t	d	  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(values + 12), scale));
	int16x8_t	ab	  = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
	int16x8_t	cd	  = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
	int8x16_t	codes = vcombine_s8(vqmovn_s16(ab), vqmovn_s16(cd));
	vst1q_s8(out, vmaxq_s8(codes, vdupq_n_s8(-127)));
#else
	for (int j = 0; j < 16; j++) {
		long code = lroundf(values[j] * inverse);
		out[j]	  = (int8_t)(code < -127 ? -127 : code > 127 ? 127 : code);
	}
#endif
}

int rkt_pack_weight_i8(const rkt_weight_pack *pack, int row) {
	int	  column = pack->first + row;
	float values[pack->k];
	for (int b = 0; b < pack->k / 32; b++)
		decode_block(pack->source + (size_t)column * (pack->k / 32) + b, values + b * 32);
	rotate_channels(values, pack->k, pack->rotation);
	for (int chunk = pack->first_chunk; chunk < pack->first_chunk + pack->chunks; chunk++) {
		int	  begin	  = chunk * pack->tile_k;
		int	  count	  = pack->k - begin < pack->tile_k ? pack->k - begin : pack->tile_k;
		float maximum = max_abs(values + begin, count);
		float inverse = maximum ? 127.0f / maximum : 0.0f;
		if (!isfinite(maximum) || !isfinite(inverse))
			return 0;
		pack->scales[(size_t)chunk * pack->n + column] = maximum / 127.0f;
		for (int j = 0; j < count; j += 16) {
			size_t index = (size_t)(chunk - pack->first_chunk) * pack->cols * pack->tile_k +
						   ((size_t)(row / 32) * (pack->tile_k / 32) + j / 32) * 1024 +
						   row % 32 * 32 + j % 32;
			quantize_16(values + begin + j, pack->data + index, inverse);
		}
	}
	return 1;
}

int rkt_pack_input_i8(const float *x, int8_t *dest, float *scales, int k, int m, int row,
					  int tile_k, const float *rotation) {
	float values[k];
	memcpy(values, x, (size_t)k * sizeof(float));
	rotate_channels(values, k, rotation);
	for (int begin = 0, chunk = 0; begin < k; begin += tile_k, chunk++) {
		int	  count	  = k - begin < tile_k ? k - begin : tile_k;
		float maximum = max_abs(values + begin, count);
		float inverse = maximum ? 127.0f / maximum : 0.0f;
		if (!isfinite(maximum) || !isfinite(inverse))
			return 0;
		scales[(size_t)chunk * m + row] = maximum / 127.0f;
		for (int j = 0; j < count; j += 16) {
			size_t index = (size_t)chunk * m * tile_k + (size_t)(j / 16) * m * 16 + row * 16;
			quantize_16(values + begin + j, dest + index, inverse);
		}
	}
	return 1;
}
