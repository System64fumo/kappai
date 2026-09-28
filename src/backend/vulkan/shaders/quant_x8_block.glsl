int	  xq_x_base = bi * 256 - tile_start;
float xq_maxv = 0.0, xq_amax = 0.0;
for (int j = 0; j < 256; j++) {
	float ax = abs(x_shared[xq_x_base + j]);
	if (ax > xq_amax) {
		xq_amax = ax;
		xq_maxv = x_shared[xq_x_base + j];
	}
}
int xq_local_bi = bi - block_start;
if (xq_amax == 0.0) {
	xq_d[xq_local_bi] = 0.0;
	for (int j = 0; j < 256; j++)
		xq_shared[xq_x_base + j] = 0;
#ifdef XQ_HAVE_BSUMS
	for (int j = 0; j < 16; j++)
		xq_bsums[xq_local_bi * 16 + j] = 0;
#endif
} else {
	float xq_iscale = -127.0 / xq_maxv;
	for (int j = 0; j < 256; j++) {
		int v = nearest_int(xq_iscale * x_shared[xq_x_base + j]);
		if (v > 127)
			v = 127;
		if (v < -127)
			v = -127;
		xq_shared[xq_x_base + j] = v;
	}
#ifdef XQ_HAVE_BSUMS
	for (int j = 0; j < 16; j++) {
		int sum = 0;
		for (int ii = 0; ii < 16; ii++)
			sum += xq_shared[xq_x_base + j * 16 + ii];
		xq_bsums[xq_local_bi * 16 + j] = sum;
	}
#endif
	xq_d[xq_local_bi] = 1.0 / xq_iscale;
}
