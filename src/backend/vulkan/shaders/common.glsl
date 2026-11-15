#ifndef COMMON_GLSL
#define COMMON_GLSL

#define Q4_0_BLOCK_BYTES 18u
#define Q4_1_BLOCK_BYTES 20u
#define Q5_0_BLOCK_BYTES 22u
#define Q5_1_BLOCK_BYTES 24u
#define Q8_0_BLOCK_BYTES 34u
#define IQ4_NL_BLOCK_BYTES 18u
#define Q4_K_BLOCK_BYTES 144u
#define Q5_K_BLOCK_BYTES 176u
#define Q6_K_BLOCK_BYTES 210u

#define BLOCK_ELEMS_32 32
#define BLOCK_ELEMS_256 256

float bf16_to_float(uint b) {
	return uintBitsToFloat(b << 16u);
}

float half_to_float(uint h) {
	uint  w_bits			  = h << 16u;
	uint  sign				  = w_bits & 0x80000000u;
	uint  two_w				  = w_bits + w_bits;
	uint  exp_offset		  = 0xE0u << 23u;
	float exp_scale			  = uintBitsToFloat(0x07800000u);
	float normalized_value	  = uintBitsToFloat((two_w >> 4u) + exp_offset) * exp_scale;
	uint  magic_mask		  = 126u << 23u;
	float denormalized_value  = uintBitsToFloat((two_w >> 17u) | magic_mask) - 0.5;
	uint  denormalized_cutoff = 1u << 27u;
	uint  bits				  = two_w < denormalized_cutoff ? floatBitsToUint(denormalized_value)
															: floatBitsToUint(normalized_value);
	return uintBitsToFloat(sign | bits);
}

float neg_max_float() {
	return uintBitsToFloat(0xFF7FFFFFu);
}

int round_away(float f) {
	float a = abs(f);
	float t = trunc(a);
	int	  r = int(t) + ((a - t) >= 0.5 ? 1 : 0);
	return f < 0.0 ? -r : r;
}

int nearest_int(float f) {
	return round_away(f);
}

float div_exact(float a, float b) {
	uint nb	  = floatBitsToUint(a);
	uint db	  = floatBitsToUint(b);
	uint sign = (nb ^ db) & 0x80000000u;
	if ((nb & 0x7FFFFFFFu) == 0u || (db & 0x7FFFFFFFu) == 0u)
		return uintBitsToFloat(sign);
	uint Nn	 = (nb & 0x7FFFFFu) | 0x800000u;
	uint Dd	 = (db & 0x7FFFFFu) | 0x800000u;
	int	 eq	 = int((nb >> 23u) & 0xFFu) - int((db >> 23u) & 0xFFu);
	uint M24 = 0u;
	uint rem;
	int	 me;
	if (Nn >= Dd) {
		me	= eq;
		rem = Nn - Dd;
	} else {
		me	= eq - 1;
		rem = (Nn << 1u) - Dd;
	}
	for (int i = 0; i < 23; i++) {
		rem <<= 1u;
		M24 <<= 1u;
		if (rem >= Dd) {
			rem -= Dd;
			M24 |= 1u;
		}
	}
	uint r2 = rem << 1u;
	if (r2 > Dd || (r2 == Dd && (M24 & 1u) == 1u))
		M24++;
	if (M24 >= (1u << 23)) {
		M24 = 0u;
		me++;
	}
	int fe = me + 127;
	if (fe >= 255)
		return uintBitsToFloat(sign | 0x7F800000u);
	if (fe <= 0)
		return uintBitsToFloat(sign);
	return uintBitsToFloat(sign | (uint(fe) << 23u) | M24);
}

#define num_fma(a, b, c) fma(a, b, c)

float num_exp_split(float x, out float pow2n) {
	float xc = min(max(x, -88.0), 88.0);
	float n	 = roundEven(xc * uintBitsToFloat(0x3FB8AA3Bu));
	float r	 = num_fma(-n, uintBitsToFloat(0x3F317218u), xc);
	r		 = num_fma(-n, uintBitsToFloat(0xB102E308u), r);
	float t	 = uintBitsToFloat(0x39500D01u);
	t		 = num_fma(r, t, uintBitsToFloat(0x3AB60B61u));
	t		 = num_fma(r, t, uintBitsToFloat(0x3C088889u));
	t		 = num_fma(r, t, uintBitsToFloat(0x3D2AAAABu));
	t		 = num_fma(r, t, uintBitsToFloat(0x3E2AAAABu));
	t		 = num_fma(r, t, 0.5);
	float r2 = r * r;
	pow2n	 = uintBitsToFloat(uint(int(n) + 127) << 23u);
	return num_fma(r2, t, r);
}

float num_exp(float x) {
	float pow2n;
	float em1 = num_exp_split(x, pow2n);
	return num_fma(pow2n, em1, pow2n);
}

vec2 num_exp(vec2 x) {
	return vec2(num_exp(x.x), num_exp(x.y));
}

vec4 num_exp(vec4 x) {
	return vec4(num_exp(x.x), num_exp(x.y), num_exp(x.z), num_exp(x.w));
}

float num_expm1(float x) {
	float pow2n;
	float em1 = num_exp_split(x, pow2n);
	return num_fma(pow2n, em1, pow2n - 1.0);
}

float num_tanh(float x) {
	float a	  = min(abs(x), 10.0);
	float em1 = num_expm1(a + a);
	float t	  = div_exact(em1, em1 + 2.0);
	return uintBitsToFloat(floatBitsToUint(t) | (floatBitsToUint(x) & 0x80000000u));
}

float num_sigmoid(float x) {
	float e = num_exp(-abs(x));
	return div_exact(x >= 0.0 ? 1.0 : e, 1.0 + e);
}

float num_silu(float x) {
	return div_exact(x, 1.0 + num_exp(-x));
}

float num_gelu_tanh(float x) {
	float x2 = x * x;
	float inner =
		(x * num_fma(x2, uintBitsToFloat(0x3D372713u), 1.0)) * uintBitsToFloat(0x3F4C422Au);
	return (x * 0.5) * (1.0 + num_tanh(inner));
}

float silu(float x) {
	return num_silu(x);
}

float gelu_tanh(float x) {
	return num_gelu_tanh(x);
}

float num_sqrt(float x) {
	uint b	= floatBitsToUint(x);
	uint ef = (b >> 23u) & 0xFFu;
	if (ef == 0u || b >= 0x7F800000u)
		return x;
	uint m	= (b & 0x7FFFFFu) | 0x800000u;
	int	 ee = int(ef) - 127;
	if ((ee & 1) != 0) {
		m <<= 1u;
		ee -= 1;
	}
	uint root = 0u;
	uint rem  = 0u;
	for (int i = 24; i >= 0; i--) {
		int	 j1	   = 2 * i + 1;
		int	 j0	   = 2 * i;
		uint b1	   = j1 >= 25 ? (m >> uint(j1 - 25)) & 1u : 0u;
		uint b0	   = j0 >= 25 ? (m >> uint(j0 - 25)) & 1u : 0u;
		rem		   = (rem << 2u) | (b1 << 1u) | b0;
		uint trial = (root << 2u) | 1u;
		root <<= 1u;
		if (rem >= trial) {
			rem -= trial;
			root |= 1u;
		}
	}
	uint mant  = root >> 1u;
	uint guard = root & 1u;
	if (guard == 1u && (rem != 0u || (mant & 1u) == 1u))
		mant++;
	int e2 = ee / 2;
	if (mant == 0x1000000u) {
		mant = 0x800000u;
		e2++;
	}
	return uintBitsToFloat((uint(e2 + 127) << 23u) | (mant & 0x7FFFFFu));
}

float num_rms_scale(float sum_sq, int n, float eps) {
	float v = div_exact(sum_sq, float(n)) + eps;
	return v > 0.0 ? div_exact(1.0, num_sqrt(v)) : 0.0;
}

uint float_to_half(float f) {
	uint bits = floatBitsToUint(f);
	uint sign = (bits >> 16u) & 0x8000u;
	uint a	  = bits & 0x7FFFFFFFu;
	if (a >= 0x7F800000u)
		return sign | (a > 0x7F800000u ? 0x7E00u : 0x7C00u);
	if (a >= 0x477FF000u)
		return sign | 0x7C00u;
	if (a <= 0x33000000u)
		return sign;
	if (a < 0x38800000u) {
		uint shift = 126u - (a >> 23u);
		uint m	   = (a & 0x7FFFFFu) | 0x800000u;
		uint h	   = m >> shift;
		uint rem   = m & ((1u << shift) - 1u);
		uint half_ = 1u << (shift - 1u);
		if (rem > half_ || (rem == half_ && (h & 1u) == 1u))
			h++;
		return sign | h;
	}
	uint r = a + 0xFFFu + ((a >> 13u) & 1u);
	return sign | ((r - 0x38000000u) >> 13u);
}

float f32_to_f16_to_f32(float f) {
	return half_to_float(float_to_half(f));
}

#endif
