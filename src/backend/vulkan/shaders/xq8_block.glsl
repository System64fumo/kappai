int xq8_quant_block(int xb, out float d_out) {
	float amax = 0.0;
	for (int j = 0; j < 32; j++)
		amax = max(amax, abs(x_shared[xb + j]));

	float d	  = amax / 127.0;
	d_out	  = d;
	float inv = d > 0.0 ? 1.0 / d : 0.0;

	int wp	= xb >> 2;
	int sum = 0;
	for (int g = 0; g < 8; g++) {
		int a = clamp(round_away(x_shared[xb + g * 4 + 0] * inv), -128, 127);
		int b = clamp(round_away(x_shared[xb + g * 4 + 1] * inv), -128, 127);
		int c = clamp(round_away(x_shared[xb + g * 4 + 2] * inv), -128, 127);
		int e = clamp(round_away(x_shared[xb + g * 4 + 3] * inv), -128, 127);
		sum += a + b + c + e;
		xq_packed[wp + g] = uint(pack32(i8vec4(int8_t(a), int8_t(b), int8_t(c), int8_t(e))));
	}
	return sum;
}
