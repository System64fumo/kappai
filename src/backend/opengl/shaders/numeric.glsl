#ifndef NUMERIC_GLSL
#define NUMERIC_GLSL

#extension GL_EXT_gpu_shader5 : enable
#ifdef GL_EXT_gpu_shader5
#define num_fma(a, b, c) fma(a, b, c)
#else
#define num_fma(a, b, c) ((a) * (b) + (c))
#endif

float half_to_float(uint h) {
	return unpackHalf2x16(h & 0xFFFFu).x;
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

float num_exp_split(float x, out float pow2n) {
	precise float xc  = min(max(x, -88.0), 88.0);
	precise float n	  = roundEven(xc * uintBitsToFloat(0x3FB8AA3Bu));
	precise float r	  = num_fma(-n, uintBitsToFloat(0x3F317218u), xc);
	r				  = num_fma(-n, uintBitsToFloat(0xB102E308u), r);
	precise float t	  = uintBitsToFloat(0x39500D01u);
	t				  = num_fma(r, t, uintBitsToFloat(0x3AB60B61u));
	t				  = num_fma(r, t, uintBitsToFloat(0x3C088889u));
	t				  = num_fma(r, t, uintBitsToFloat(0x3D2AAAABu));
	t				  = num_fma(r, t, uintBitsToFloat(0x3E2AAAABu));
	t				  = num_fma(r, t, 0.5);
	precise float r2  = r * r;
	precise float p2  = uintBitsToFloat(uint(int(n) + 127) << 23u);
	pow2n			  = p2;
	precise float res = num_fma(r2, t, r);
	return res;
}

float num_exp(float x) {
	float		  pow2n;
	precise float em1 = num_exp_split(x, pow2n);
	precise float r	  = num_fma(pow2n, em1, pow2n);
	return r;
}

vec2 num_exp(vec2 x) {
	return vec2(num_exp(x.x), num_exp(x.y));
}

vec4 num_exp(vec4 x) {
	return vec4(num_exp(x.x), num_exp(x.y), num_exp(x.z), num_exp(x.w));
}

float num_expm1(float x) {
	float		  pow2n;
	precise float em1 = num_exp_split(x, pow2n);
	precise float r	  = num_fma(pow2n, em1, pow2n - 1.0);
	return r;
}

float num_tanh(float x) {
	precise float a	  = min(abs(x), 10.0);
	precise float em1 = num_expm1(a + a);
	precise float t	  = div_exact(em1, em1 + 2.0);
	return uintBitsToFloat(floatBitsToUint(t) | (floatBitsToUint(x) & 0x80000000u));
}

float num_sigmoid(float x) {
	float e = num_exp(-abs(x));
	return div_exact(x >= 0.0 ? 1.0 : e, 1.0 + e);
}

float num_silu(float x) {
	precise float e = num_exp(-x);
	precise float r = div_exact(x, 1.0 + e);
	return r;
}

float num_gelu_tanh(float x) {
	precise float x2 = x * x;
	precise float inner =
		(x * num_fma(x2, uintBitsToFloat(0x3D372713u), 1.0)) * uintBitsToFloat(0x3F4C422Au);
	precise float t = 1.0 + num_tanh(inner);
	precise float r = (x * 0.5) * t;
	return r;
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
