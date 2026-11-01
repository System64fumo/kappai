layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer WBuf {
	uint w[];
};
layout(set = 0, binding = 1) readonly buffer XBuf {
	float x[];
};
layout(set = 0, binding = 2) buffer YBuf {
	float y[];
};
layout(set = 0, binding = 3) readonly buffer RBuf {
	float res[];
};

layout(push_constant) uniform Params {
	int n;
	int kb;
	int m;
	int res_mode;
}
pc;

#if defined(Q8_0)
#define SUB_BYTES 34u
#define NRAW 3
#elif defined(Q4_0) || defined(IQ4_NL)
#define SUB_BYTES 18u
#define NRAW 3
#elif defined(Q4_K)
#define SUB_BYTES 18u
#define NRAW 6
#elif defined(Q6_K)
#define BLOCK256 210u
#define NRAW 6
#elif defined(IQ2_XXS)
#define BLOCK256 66u
#define NRAW 3
#elif defined(IQ2_S)
#define BLOCK256 82u
#define NRAW 5
#elif defined(IQ3_XXS)
#define BLOCK256 98u
#define NRAW 4
#elif defined(IQ3_S)
#define BLOCK256 110u
#define NRAW 6
#elif defined(F32)
#define SUB_BYTES 128u
#define NRAW 8
#else
#define SUB_BYTES 64u
#define NRAW 4
#endif

#if defined(IQ2_XXS) || defined(IQ2_S) || defined(IQ3_XXS)
#define IQX_GRID_BINDING 4
#include "iqx_grids.glsl"
#elif defined(IQ3_S)
#define IQ3S_GRID_BINDING 4
#include "iq3s_grid.glsl"
#endif

#ifdef IQ4_NL
const float KV[16] = float[16](-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0,
							   25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
#endif

uint elem_base(uint qd) {
#if defined(Q4_0) || defined(IQ4_NL)
	return (qd >> 1u) * 16u + (qd & 1u) * 8u;
#else
	return qd * 8u;
#endif
}

#if defined(Q8_0) || defined(Q4_0) || defined(IQ4_NL) || defined(BLOCK256)
uint word_at(uint bo) {
	uint wi = bo >> 2u;
	if ((bo & 2u) == 0u)
		return w[wi];
	return (w[wi] >> 16u) | (w[wi + 1u] << 16u);
}
#endif

#ifdef BLOCK256
uint byte_at(uint addr) {
	return (word_at(addr & ~1u) >> ((addr & 1u) * 8u)) & 0xFFu;
}
#endif

uint row_base(uint row, uint kb) {
#ifdef BLOCK256
	return row * (kb >> 3u) * BLOCK256;
#else
	return row * kb * SUB_BYTES;
#endif
}

void fetch_raw(uint rbase, uint sb, uint qd, out uint raw[NRAW]) {
#if defined(Q4_K)
	uint wi = (rbase + (sb >> 3u) * 144u) >> 2u;
	uint qo = wi + 4u + ((sb & 6u) << 2u) + qd * 2u;
	for (uint i = 0u; i < 4u; i++)
		raw[i] = w[wi + i];
	raw[4] = w[qo];
	raw[5] = w[qo + 1u];
#elif defined(Q6_K)
	uint bo	 = rbase + (sb >> 3u) * BLOCK256;
	uint sub = sb & 7u;
	uint qq	 = sub & 3u;
	uint ql	 = bo + (sub >> 2u) * 64u + (qq & 1u) * 32u + qd * 8u;
	uint qh	 = bo + 128u + (sub >> 2u) * 32u + qd * 8u;
	raw[0]	 = word_at(ql);
	raw[1]	 = word_at(ql + 4u);
	raw[2]	 = word_at(qh);
	raw[3]	 = word_at(qh + 4u);
	raw[4]	 = byte_at(bo + 192u + (sub >> 2u) * 8u + (qd >> 1u) + qq * 2u);
	raw[5]	 = word_at(bo + 208u);
#elif defined(IQ2_XXS)
	uint bo	 = rbase + (sb >> 3u) * BLOCK256;
	uint aux = bo + 2u + (sb & 7u) * 8u;
	raw[0]	 = word_at(aux);
	raw[1]	 = word_at(aux + 4u);
	raw[2]	 = word_at(bo);
#elif defined(IQ2_S)
	uint bo = rbase + (sb >> 3u) * BLOCK256;
	uint ib = sb & 7u;
	raw[0]	= byte_at(bo + 2u + ib * 4u + qd);
	raw[1]	= byte_at(bo + 34u + ib * 4u + qd);
	raw[2]	= byte_at(bo + 66u + ib);
	raw[3]	= byte_at(bo + 74u + ib);
	raw[4]	= word_at(bo);
#elif defined(IQ3_XXS)
	uint bo = rbase + (sb >> 3u) * BLOCK256;
	uint ib = sb & 7u;
	raw[0]	= byte_at(bo + 2u + ib * 8u + qd * 2u);
	raw[1]	= byte_at(bo + 3u + ib * 8u + qd * 2u);
	raw[2]	= word_at(bo + 66u + ib * 4u);
	raw[3]	= word_at(bo);
#elif defined(IQ3_S)
	uint bo	 = rbase + (sb >> 3u) * BLOCK256;
	uint ib	 = sb & 7u;
	uint qsb = bo + 2u + (ib >> 1u) * 16u + (ib & 1u) * 8u + qd * 2u;
	raw[0]	 = byte_at(qsb);
	raw[1]	 = byte_at(qsb + 1u);
	raw[2]	 = byte_at(bo + 66u + ib);
	raw[3]	 = byte_at(bo + 74u + ib * 4u + qd);
	raw[4]	 = byte_at(bo + 106u + (ib >> 1u));
	raw[5]	 = word_at(bo);
#elif defined(Q8_0) || defined(Q4_0) || defined(IQ4_NL)
	uint bo = rbase + sb * SUB_BYTES;
#ifdef Q8_0
	uint qoff = 2u + qd * 8u;
#else
	uint qoff = 2u + (qd & 1u) * 8u;
#endif
	raw[0] = word_at(bo);
	raw[1] = word_at(bo + qoff);
	raw[2] = word_at(bo + qoff + 4u);
#else
	uint wi = (rbase + sb * SUB_BYTES + qd * (SUB_BYTES / 4u)) >> 2u;
	for (uint i = 0u; i < uint(NRAW); i++)
		raw[i] = w[wi + i];
#endif
}

#ifdef Q4_K
uint scale_byte(uint i, uint s0, uint s1, uint s2) {
	uint word = i < 4u ? s0 : (i < 8u ? s1 : s2);
	return (word >> ((i & 3u) * 8u)) & 0xFFu;
}
#endif

void dequant8(uint raw[NRAW], uint sb, uint qd, out float o[8]) {
#if defined(Q4_K)
	uint sub = sb & 7u;
	vec2 dm	 = unpackHalf2x16(raw[0]);
	uint sc, mn;
	if (sub < 4u) {
		sc = scale_byte(sub, raw[1], raw[2], raw[3]) & 63u;
		mn = scale_byte(sub + 4u, raw[1], raw[2], raw[3]) & 63u;
	} else {
		uint a = scale_byte(sub + 4u, raw[1], raw[2], raw[3]);
		uint b = scale_byte(sub - 4u, raw[1], raw[2], raw[3]);
		uint c = scale_byte(sub, raw[1], raw[2], raw[3]);
		sc	   = (a & 15u) | ((b >> 6u) << 4u);
		mn	   = (a >> 4u) | ((c >> 6u) << 4u);
	}
	float scale = dm.x * float(sc);
	float min_v = dm.y * float(mn);
	bool  high	= (sub & 1u) != 0u;
	for (uint j = 0u; j < 8u; j++) {
		uint b	 = (raw[4 + (j >> 2u)] >> ((j & 3u) * 8u)) & 0xFFu;
		uint nib = high ? b >> 4u : b & 0xFu;
		o[j]	 = scale * float(nib) - min_v;
	}
#elif defined(Q6_K)
	float d		= unpackHalf2x16(raw[5] & 0xFFFFu).x;
	float scale = d * float(bitfieldExtract(int(raw[4]), 0, 8));
	uint  shift = (sb & 3u) * 2u;
	bool  high	= ((sb >> 1u) & 1u) != 0u;
	for (uint j = 0u; j < 8u; j++) {
		uint b	= (raw[j >> 2u] >> ((j & 3u) * 8u)) & 0xFFu;
		uint hb = (raw[2u + (j >> 2u)] >> ((j & 3u) * 8u)) & 0xFFu;
		uint lo = high ? b >> 4u : b & 0xFu;
		o[j]	= scale * float(int(lo | (((hb >> shift) & 3u) << 4u)) - 32);
	}
#elif defined(IQ2_XXS) || defined(IQ2_S) || defined(IQ3_XXS) || defined(IQ3_S)
#if defined(IQ2_XXS)
	float d		= unpackHalf2x16(raw[2] & 0xFFFFu).x;
	float db	= d * (0.5 + float(raw[1] >> 28u)) * 0.25;
	uvec2 grid	= iqx_iq2xxs_grid((raw[0] >> (qd * 8u)) & 0xFFu);
	uint  signs = iqx_ksign((raw[1] >> (qd * 7u)) & 127u);
#elif defined(IQ2_S)
	float d		= unpackHalf2x16(raw[4] & 0xFFFFu).x;
	float db	= d * (0.5 + float(qd < 2u ? raw[3] & 0xFu : raw[3] >> 4u)) * 0.25;
	uvec2 grid	= iqx_iq2s_grid(raw[0] | ((raw[2] << (8u - 2u * qd)) & 0x300u));
	uint  signs = raw[1];
#endif
#if defined(IQ2_XXS) || defined(IQ2_S)
	for (uint j = 0u; j < 8u; j++) {
		float g = float(((j < 4u ? grid.x : grid.y) >> ((j & 3u) * 8u)) & 0xFFu);
		o[j]	= ((signs >> j) & 1u) != 0u ? -db * g : db * g;
	}
#elif defined(IQ3_XXS)
	float d		= unpackHalf2x16(raw[3] & 0xFFFFu).x;
	float db	= d * (0.5 + float(raw[2] >> 28u)) * 0.5;
	uint  gw[2] = uint[2](iqx_iq3xxs_grid(raw[0]), iqx_iq3xxs_grid(raw[1]));
	uint  signs = iqx_ksign((raw[2] >> (qd * 7u)) & 127u);
	for (uint j = 0u; j < 8u; j++) {
		float g = float((gw[j >> 2u] >> ((j & 3u) * 8u)) & 0xFFu);
		o[j]	= ((signs >> j) & 1u) != 0u ? -db * g : db * g;
	}
#else
	float d		= unpackHalf2x16(raw[5] & 0xFFFFu).x;
	float db	= d * float(1u + 2u * ((sb & 1u) != 0u ? raw[4] >> 4u : raw[4] & 0xFu));
	uint  gw[2] = uint[2](iq3s_grid_word(raw[0] | ((raw[2] << (8u - 2u * qd)) & 256u)),
						  iq3s_grid_word(raw[1] | ((raw[2] << (7u - 2u * qd)) & 256u)));
	for (uint j = 0u; j < 8u; j++) {
		float g = float((gw[j >> 2u] >> ((j & 3u) * 8u)) & 0xFFu);
		o[j]	= ((raw[3] >> j) & 1u) != 0u ? -db * g : db * g;
	}
#endif
#elif defined(Q8_0) || defined(Q4_0) || defined(IQ4_NL)
	float d = unpackHalf2x16(raw[0] & 0xFFFFu).x;
	for (uint j = 0u; j < 8u; j++) {
		uint word = raw[1 + (j >> 2u)];
#if defined(Q8_0)
		o[j] = d * float(bitfieldExtract(int(word), int((j & 3u) * 8u), 8));
#else
		uint b	 = (word >> ((j & 3u) * 8u)) & 0xFFu;
		uint nib = (qd >> 1u) != 0u ? b >> 4u : b & 0xFu;
#ifdef IQ4_NL
		o[j] = d * KV[nib];
#else
		o[j] = d * float(int(nib) - 8);
#endif
#endif
	}
#elif defined(F32)
	for (uint j = 0u; j < 8u; j++)
		o[j] = uintBitsToFloat(raw[j]);
#else
	for (uint j = 0u; j < 8u; j++) {
		uint word = raw[j >> 1u];
		uint h	  = (j & 1u) == 0u ? word & 0xFFFFu : word >> 16u;
#ifdef F16
		o[j] = unpackHalf2x16(h).x;
#else
		o[j] = uintBitsToFloat(h << 16u);
#endif
	}
#endif
}
