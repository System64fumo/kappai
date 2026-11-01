# Support tables

## Platforms

| OS    | Architecture                   |
|-------|--------------------------------|
| Linux | `aarch64`, `riscv64`, `x86_64` |
| macOS | `aarch64`                      |

## Backends

| Backend       | State                                            |
|---------------|--------------------------------------------------|
| `cpu_scalar`  | Fully working, reference for all other backends  |
| `cpu_aarch64` | Fully working                                    |
| `cpu_x86_64`  | Works, requires modern CPUs, lacks some features |
| `vulkan`      | Works, requires modern GPUs, lacks some features |
| `opengl`      | Works, lacks some features                       |

## Quantizations

| Family   | Quants                                                     |
|----------|------------------------------------------------------------|
| Legacy   | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`                     |
| K-quants | `Q4_K`, `Q5_K`, `Q6_K`                                     |
| I-quants | `IQ2_S`, `IQ2_XXS`, `IQ3_S`, `IQ3_XXS`, `IQ4_XS`, `IQ4_NL` |
| Float    | `F32`, `F16`, `BF16`                                       |

## Architectures

| Architecture  | Models                       |
|---------------|------------------------------|
| `llama`       | Llama 3.2                    |
| `gemma4`      | Gemma 4 (E2B, A4B, 12B, 26B) |
| `lfm`         | LFM 2.5                      |
| `lfm-d1`      | LFM 2.5 D1                   |
| `lfm-d1-omni` | LFM 2 D1 Omni                |
| `qwen35`      | Qwen 3.5, Qwen 3.6, Qwen 3.8 |
| `glm-dsa`     | GLM 5.2, GLM 5.3             |
| `glm5-next`   | GLM 5.3 Flash                |
| `qwen-image`  | Qwen Image 2.1               |