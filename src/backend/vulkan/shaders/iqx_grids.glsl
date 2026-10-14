#ifndef IQX_GRIDS_GLSL
#define IQX_GRIDS_GLSL

#ifndef IQX_GRID_BINDING
#define IQX_GRID_BINDING 3
#endif

layout(set = 0, binding = IQX_GRID_BINDING) readonly buffer IQXGridBuf {
	uint iqx_grid[];
};

uint iqx_u32(uint word_idx) {
	return iqx_grid[word_idx];
}

uvec2 iqx_u64(uint idx) {
	return uvec2(iqx_grid[idx * 2u], iqx_grid[idx * 2u + 1u]);
}

uvec2 iqx_iq2s_grid(uint idx) {
	return iqx_u64(idx);
}

uvec2 iqx_iq2xxs_grid(uint idx) {
	uint base = 2048u + idx * 2u;
	return uvec2(iqx_grid[base], iqx_grid[base + 1u]);
}

uint iqx_iq3xxs_grid(uint idx) {
	return iqx_grid[2560u + idx];
}

uint iqx_ksign(uint i) {
	uint word = iqx_grid[2816u + (i >> 2u)];
	return (word >> ((i & 3u) * 8u)) & 0xFFu;
}

#endif
