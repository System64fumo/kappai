#ifndef QBLOCK_GLSL
#define QBLOCK_GLSL

#if defined(Q4_0) || defined(IQ4_NL)
#define BLOCK_U16 9u
#define QS_U16 1u
#elif defined(Q4_1)
#define BLOCK_U16 10u
#define QS_U16 2u
#elif defined(Q5_0)
#define BLOCK_U16 11u
#define QS_U16 3u
#elif defined(Q5_1)
#define BLOCK_U16 12u
#define QS_U16 4u
#elif defined(Q8_0)
#define BLOCK_U16 17u
#define QS_U16 1u
#endif

#if defined(Q4_1) || defined(Q5_1)
#define HAS_MIN
#endif

#ifdef Q8_0
#define XW_A(p) (2u * (p))
#define XW_B(p) (2u * (p) + 1u)
#else
#define XW_A(p) (p)
#define XW_B(p) (4u + (p))
#endif

#define XQ_BLOCK_WORDS 12u
#define XQ_D_WORD 8u
#define XQ_S_WORD 9u

#ifdef IQ4_NL
shared uint s_kv[16];
const uint	KV_BYTES[16] = uint[16](0x81u, 0x98u, 0xADu, 0xBFu, 0xCFu, 0xDDu, 0xEAu, 0xF6u, 0x01u,
									0x0Du, 0x19u, 0x26u, 0x35u, 0x45u, 0x59u, 0x71u);

void qblock_init() {
	if (gl_LocalInvocationIndex < 16u)
		s_kv[gl_LocalInvocationIndex] = KV_BYTES[gl_LocalInvocationIndex];
}

uint kv4(uint nib) {
	return s_kv[nib & 0xFu] | (s_kv[(nib >> 8u) & 0xFu] << 8u) |
		   (s_kv[(nib >> 16u) & 0xFu] << 16u) | (s_kv[nib >> 24u] << 24u);
}
#else
void qblock_init() {}
#endif

uint sub_bytes(uint v, uint k) {
	return ((v | 0x80808080u) - k) ^ 0x80808080u;
}

uint spread4(uint h) {
	return ((h & 1u) << 4u) | ((h & 2u) << 11u) | ((h & 4u) << 18u) | ((h & 8u) << 25u);
}

uint wq16(uint i) {
	return uint(W16[i]);
}

uint wq32(uint i) {
	return wq16(i) | (wq16(i + 1u) << 16u);
}

void load_wpart(uint b, uint p, out uint wa, out uint wb, out float d, out float mn) {
	d = unpackHalf2x16(wq16(b)).x;
#ifdef HAS_MIN
	mn = unpackHalf2x16(wq16(b + 1u)).x;
#else
	mn = 0.0;
#endif
#ifdef Q8_0
	wa = wq32(b + QS_U16 + 4u * p);
	wb = wq32(b + QS_U16 + 4u * p + 2u);
#else
	uint q	= wq32(b + QS_U16 + 2u * p);
	uint lo = q & 0x0F0F0F0Fu;
	uint hi = (q >> 4u) & 0x0F0F0F0Fu;
#if defined(Q5_0) || defined(Q5_1)
	uint qh = wq32(b + QS_U16 - 2u);
	lo |= spread4((qh >> (4u * p)) & 0xFu);
	hi |= spread4((qh >> (16u + 4u * p)) & 0xFu);
#endif
#if defined(IQ4_NL)
	wa = kv4(lo);
	wb = kv4(hi);
#elif defined(Q4_0)
	wa = sub_bytes(lo, 0x08080808u);
	wb = sub_bytes(hi, 0x08080808u);
#elif defined(Q5_0)
	wa = sub_bytes(lo, 0x10101010u);
	wb = sub_bytes(hi, 0x10101010u);
#else
	wa = lo;
	wb = hi;
#endif
#endif
}

#endif
