#ifndef ACTIVATE_GLSL
#define ACTIVATE_GLSL

float silu(float x) {
	return x / (1.0 + exp(-x));
}

float gelu_tanh(float x) {
	const float t = 0.7978845608 * (x + 0.044715 * x * x * x);
	return 0.5 * x * (1.0 + tanh(t));
}

#endif
