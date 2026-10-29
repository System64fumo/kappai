#include "diffusion/dit.h"

#include <math.h>

void dit_timestep_embedding(float *out, int dim, float t, float max_period) {
	int half = dim / 2;
	for (int j = 0; j < half; j++) {
		float freq	  = expf(-logf(max_period) * (float)j / (float)half);
		out[j]		  = cosf(t * freq);
		out[j + half] = sinf(t * freq);
	}
}
