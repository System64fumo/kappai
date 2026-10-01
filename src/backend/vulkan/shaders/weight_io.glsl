#ifndef W_NAME
#define W_NAME w0
#endif
#ifndef W_PFX
#define W_PFX w0_
#endif

#define WIO_CAT_INNER(a, b) a##b
#define WIO_CAT(a, b) WIO_CAT_INNER(a, b)

uint WIO_CAT(W_PFX, read_u8)(uint byte_offset) {
	uint word  = W_NAME[byte_offset >> 2u];
	uint shift = (byte_offset & 3u) * 8u;
	return (word >> shift) & 0xFFu;
}

uint WIO_CAT(W_PFX, read_u16)(uint byte_offset) {
	uint word_idx = byte_offset >> 2u;
	uint shift	  = (byte_offset & 3u) * 8u;
	uint lo		  = W_NAME[word_idx];
	if (shift + 16u <= 32u) {
		return (lo >> shift) & 0xFFFFu;
	}
	uint hi = W_NAME[word_idx + 1u];
	return ((lo >> shift) | (hi << (32u - shift))) & 0xFFFFu;
}

uint WIO_CAT(W_PFX, read_u32)(uint byte_offset) {
	uint word_idx = byte_offset >> 2u;
	uint shift	  = (byte_offset & 3u) * 8u;
	uint lo		  = W_NAME[word_idx];
	if (shift == 0u)
		return lo;
	uint hi = W_NAME[word_idx + 1u];
	return (lo >> shift) | (hi << (32u - shift));
}

void WIO_CAT(W_PFX, read_u32x4)(uint byte_offset, out uint v[4]) {
	uint base  = byte_offset & ~3u;
	uint wi	   = base >> 2u;
	uint shift = (byte_offset & 3u) * 8u;
	uint l0	   = W_NAME[wi];
	if (shift == 0u) {
		v[0] = l0;
		v[1] = W_NAME[wi + 1u];
		v[2] = W_NAME[wi + 2u];
		v[3] = W_NAME[wi + 3u];
		return;
	}
	uint l1	 = W_NAME[wi + 1u];
	uint l2	 = W_NAME[wi + 2u];
	uint l3	 = W_NAME[wi + 3u];
	uint l4	 = W_NAME[wi + 4u];
	uint inv = 32u - shift;
	v[0]	 = (l0 >> shift) | (l1 << inv);
	v[1]	 = (l1 >> shift) | (l2 << inv);
	v[2]	 = (l2 >> shift) | (l3 << inv);
	v[3]	 = (l3 >> shift) | (l4 << inv);
}

int WIO_CAT(W_PFX, read_i8)(uint byte_offset) {
	uint v = WIO_CAT(W_PFX, read_u8)(byte_offset);
	return v >= 128u ? int(v) - 256 : int(v);
}

void WIO_CAT(W_PFX, get_scale_min_k4)(uint s_off, int j, out uint d_out, out uint m_out) {
	if (j < 4) {
		d_out = WIO_CAT(W_PFX, read_u8)(s_off + uint(j)) & 63u;
		m_out = WIO_CAT(W_PFX, read_u8)(s_off + uint(j + 4)) & 63u;
	} else {
		uint qj4  = WIO_CAT(W_PFX, read_u8)(s_off + uint(j + 4));
		uint qjm4 = WIO_CAT(W_PFX, read_u8)(s_off + uint(j - 4));
		uint qj0  = WIO_CAT(W_PFX, read_u8)(s_off + uint(j));
		d_out	  = (qj4 & 0xFu) | ((qjm4 >> 6u) << 4u);
		m_out	  = (qj4 >> 4u) | ((qj0 >> 6u) << 4u);
	}
}

void WIO_CAT(W_PFX, get_all_scale_min_k4)(uint s_off, out uint d_out[8], out uint m_out[8]) {
	uint sb[12];
	for (int t = 0; t < 12; t++)
		sb[t] = WIO_CAT(W_PFX, read_u8)(s_off + uint(t));

	for (int j = 0; j < 4; j++) {
		d_out[j] = sb[j] & 63u;
		m_out[j] = sb[j + 4] & 63u;
	}
	for (int j = 4; j < 8; j++) {
		uint qj4  = sb[j + 4];
		uint qjm4 = sb[j - 4];
		uint qj0  = sb[j];
		d_out[j]  = (qj4 & 0xFu) | ((qjm4 >> 6u) << 4u);
		m_out[j]  = (qj4 >> 4u) | ((qj0 >> 6u) << 4u);
	}
}

#undef WIO_CAT_INNER
#undef WIO_CAT
