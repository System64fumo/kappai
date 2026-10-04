void xq8_quant_superblock(int bi, int block_start, int tile_start) {
	int	  xb   = bi * 256 - tile_start;
	float amax = 0.0, maxv = 0.0;
	for (int j = 0; j < 256; j++) {
		float v = x_shared[xb + j];
		float a = abs(v);
		if (a > amax) {
			amax = a;
			maxv = v;
		}
	}

	int local_bi = bi - block_start;
	int wp		 = xb >> 2;
	if (amax == 0.0) {
		xq_d[local_bi] = 0.0;
		for (int w = 0; w < 64; w++)
			xq_packed[wp + w] = 0u;
#ifdef XQ_HAVE_BSUMS
		for (int j = 0; j < 16; j++)
			xq_bsums[local_bi * 16 + j] = 0;
#endif
		return;
	}

	float iscale = -127.0 / maxv;
	for (int w = 0; w < 64; w++) {
		int a			  = clamp(nearest_int(iscale * x_shared[xb + w * 4 + 0]), -127, 127);
		int b			  = clamp(nearest_int(iscale * x_shared[xb + w * 4 + 1]), -127, 127);
		int c			  = clamp(nearest_int(iscale * x_shared[xb + w * 4 + 2]), -127, 127);
		int d			  = clamp(nearest_int(iscale * x_shared[xb + w * 4 + 3]), -127, 127);
		xq_packed[wp + w] = uint(pack32(i8vec4(int8_t(a), int8_t(b), int8_t(c), int8_t(d))));
	}
#ifdef XQ_HAVE_BSUMS
	for (int j = 0; j < 16; j++) {
		int sum = 0;
		for (int w = j * 4; w < j * 4 + 4; w++) {
			uint word = xq_packed[wp + w];
			sum += int(word << 24u) >> 24u;
			sum += int(word << 16u) >> 24u;
			sum += int(word << 8u) >> 24u;
			sum += int(word) >> 24u;
		}
		xq_bsums[local_bi * 16 + j] = sum;
	}
#endif
	xq_d[local_bi] = 1.0 / iscale;
}
