#ifndef W_NAME
#define W_NAME w0
#endif
#ifndef W_PFX
#define W_PFX w0_
#endif

#define WIO_CAT_INNER(a, b) a##b
#define WIO_CAT(a, b) WIO_CAT_INNER(a, b)

uint WIO_CAT(W_PFX, read_u8)(uint byte_offset) {
    uint word = W_NAME[byte_offset >> 2u];
    uint shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

uint WIO_CAT(W_PFX, read_u16)(uint byte_offset) {
    uint word_idx = byte_offset >> 2u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = W_NAME[word_idx];
    if (shift + 16u <= 32u) {
        return (lo >> shift) & 0xFFFFu;
    }
    uint hi = W_NAME[word_idx + 1u];
    return ((lo >> shift) | (hi << (32u - shift))) & 0xFFFFu;
}

uint WIO_CAT(W_PFX, read_u32)(uint byte_offset) {
    uint word_idx = byte_offset >> 2u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = W_NAME[word_idx];
    if (shift == 0u) return lo;
    uint hi = W_NAME[word_idx + 1u];
    return (lo >> shift) | (hi << (32u - shift));
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
        uint qj4 = WIO_CAT(W_PFX, read_u8)(s_off + uint(j + 4));
        uint qjm4 = WIO_CAT(W_PFX, read_u8)(s_off + uint(j - 4));
        uint qj0 = WIO_CAT(W_PFX, read_u8)(s_off + uint(j));
        d_out = (qj4 & 0xFu) | ((qjm4 >> 6u) << 4u);
        m_out = (qj4 >> 4u)  | ((qj0  >> 6u) << 4u);
    }
}

#undef WIO_CAT_INNER
#undef WIO_CAT
