#ifndef COMMON_GLSL
#define COMMON_GLSL

#define Q4_0_BLOCK_BYTES   18u
#define Q4_1_BLOCK_BYTES   20u
#define Q5_0_BLOCK_BYTES   22u
#define Q5_1_BLOCK_BYTES   24u
#define Q8_0_BLOCK_BYTES   34u
#define IQ4_NL_BLOCK_BYTES 18u
#define Q4_K_BLOCK_BYTES   144u
#define Q5_K_BLOCK_BYTES   176u
#define Q6_K_BLOCK_BYTES   210u

#define BLOCK_ELEMS_32  32
#define BLOCK_ELEMS_256 256

float half_to_float(uint h) {
    uint w_bits = h << 16u;
    uint sign   = w_bits & 0x80000000u;
    uint two_w  = w_bits + w_bits;
    uint  exp_offset       = 0xE0u << 23u;
    float exp_scale        = uintBitsToFloat(0x07800000u);
    float normalized_value = uintBitsToFloat((two_w >> 4u) + exp_offset) * exp_scale;
    uint  magic_mask       = 126u << 23u;
    float denormalized_value = uintBitsToFloat((two_w >> 17u) | magic_mask) - 0.5;
    uint  denormalized_cutoff = 1u << 27u;
    uint  bits = two_w < denormalized_cutoff
                 ? floatBitsToUint(denormalized_value)
                 : floatBitsToUint(normalized_value);
    return uintBitsToFloat(sign | bits);
}

float f32_to_f16_to_f32(float f) {
    float base = (abs(f) * uintBitsToFloat(0x77800000u)) * uintBitsToFloat(0x08800000u);
    uint w = floatBitsToUint(f);
    uint shl1_w = w + w;
    uint sign = w & 0x80000000u;
    uint bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;
    base = uintBitsToFloat((bias >> 1u) + 0x07800000u) + base;
    uint bits = floatBitsToUint(base);
    uint exp_bits = (bits >> 13u) & 0x7C00u;
    uint mant_bits = bits & 0x0FFFu;
    uint nonsign = exp_bits + mant_bits;
    uint h = (sign >> 16u) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign);
    uint w_bits = h << 16u;
    uint h_sign = w_bits & 0x80000000u;
    uint two_w = w_bits + w_bits;
    uint exp_offset = 0xE0u << 23u;
    float exp_scale = uintBitsToFloat(0x07800000u);
    float normalized = uintBitsToFloat((two_w >> 4u) + exp_offset) * exp_scale;
    uint magic_mask = 126u << 23u;
    float denormalized = uintBitsToFloat((two_w >> 17u) | magic_mask) - 0.5;
    uint denorm_cutoff = 1u << 27u;
    uint result_bits = two_w < denorm_cutoff
                       ? floatBitsToUint(denormalized)
                       : floatBitsToUint(normalized);
    return uintBitsToFloat(h_sign | result_bits);
}

uint float_to_half(float f) {
    uint fbits = floatBitsToUint(f);
    uint sign = (fbits >> 16u) & 0x8000u;
    uint exp_mant = fbits & 0x7FFFFFFFu;
    uint exp_field = (fbits >> 23u) & 0xFFu;
    if (exp_field == 0u || exp_field < 113u) return sign;
    if (exp_field >= 142u) return sign | 0x7C00u;
    uint mant_round = exp_mant + 0x00001000u;
    uint exp_new = (mant_round >> 23u) & 0xFFu;
    if (exp_new >= 142u) return sign | 0x7C00u;
    uint h_exp = ((mant_round >> 23u) + 15u - 127u) << 10u;
    uint h_mant = (mant_round >> 13u) & 0x3FFu;
    return sign | h_exp | h_mant;
}

int round_away(float f) {
    return f >= 0.0 ? int(floor(f + 0.5)) : int(ceil(f - 0.5));
}

int nearest_int(float f) {
    return f >= 0.0 ? int(floor(f + 0.5)) : -int(floor(-f + 0.5));
}

#endif
