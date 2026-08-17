#ifndef IQ3S_GRID_GLSL
#define IQ3S_GRID_GLSL

#ifndef IQ3S_GRID_BINDING
#define IQ3S_GRID_BINDING 3
#endif

layout(set = 0, binding = IQ3S_GRID_BINDING) readonly buffer IQ3SGridBuf {
    uint iq3s_grid[];
};

uint iq3s_grid_word(uint grid_idx) {
    return iq3s_grid[grid_idx];
}

int iq3s_grid_byte(uint grid_idx, int byte_idx) {
    uint word = iq3s_grid[grid_idx];
    return int((word >> uint(byte_idx * 8)) & 0xFFu);
}

#endif
