# kappai

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/System64fumo/kappai)](https://github.com/System64fumo/kappai/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/System64fumo/kappai)](https://github.com/System64fumo/kappai/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/System64fumo/kappai)](https://github.com/System64fumo/kappai/pulls)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/y6p3ua995Y)
[![Matrix](https://img.shields.io/badge/Matrix-Join-000000?logo=matrix&logoColor=white)](https://matrix.to/#/#kappai:matrix.org)
[![Donate](https://img.shields.io/badge/Donate-PayPal-00457C?logo=paypal&logoColor=white)](https://www.paypal.com/ncp/payment/EXAB48VXV3XJJ)

Kappai is a local AI inference engine that heavily prioritizes performance and efficiency.

It doesn't try to replace bigger or more serious engines. It just tries to make inference more accessible by making lower-end systems go faster. In some scenarios you can expect up to **800%** speedups. Results vary from system to system, model architecture, and quantization.

## Features

- High-performance CPU inference
- Vulkan backend
- OpenGL backend
- Heterogeneous Compute Scheduler
- MoE streaming
- OpenAI-compatible server
- Tool calling
- Decision models

## Support matrix

| Family   | Quants                                                     |
|----------|------------------------------------------------------------|
| Legacy   | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`                     |
| K-quants | `Q4_K`, `Q5_K`, `Q6_K`                                     |
| I-quants | `IQ2_S`, `IQ2_XXS`, `IQ3_S`, `IQ3_XXS`, `IQ4_XS`, `IQ4_NL` |
| Float    | `F32`, `F16`, `BF16`                                       |

| Architecture | Models                       |
|--------------|------------------------------|
| `llama`      | Llama 3.2                    |
| `gemma4`     | Gemma 4 (E2B, A4B, 12B, 26B) |
| `lfm`        | LFM 2.5                      |
| `lfm-d1`     | LFM 2.5 D1                   |
| `qwen35`     | Qwen 3.5, Qwen 3.6, Qwen 3.8 |
| `glm-dsa`    | GLM 5.2, GLM 5.3             |
| `glm5-next`  | GLM 5.3 Flash                |

| Backend       | State                                            |
|---------------|--------------------------------------------------|
| `cpu_scalar`  | Fully working, Reference for all other backends  |
| `cpu_aarch64` | Fully working                                    |
| `cpu_x86_64`  | Works, Requires modern CPUs, Lacks some features |
| `vulkan`      | Works, Requires modern GPUs, Lacks some features |
| `opengl`      | Works, Lacks some features                       |

| OS    | Architecture                   |
|-------|--------------------------------|
| Linux | `aarch64`, `riscv64`, `x86_64` |
| MacOS | `aarch64`                      |

## Building

```bash
make config BUILD=release
make -j 4
```

With some backends enabled:

```bash
make config BUILD=release BACKENDS=vulkan,opengl
make -j 4
```

## Usage

The easiest way to use kappai is to just run:

```bash
./build/kappai
```

For a less interactive manual CLI use:

```bash
./build/kappai-cli -m model.gguf
```

For server use:

```bash
./build/kappai-server -m model.gguf
```

## Contribution

If you like the project and want to support it, you can:

- Star the repository
- Contribute code
- Offer hardware or monetary donations

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the GGUF format and for being the best local LLM inference engine.
- [colibri](https://github.com/JustVugg/colibri) for motivating me to finish the streaming feature.
- [z.ai](https://z.ai) for their excellent models that helped throughout this journey.
- You for taking interest in this project.


> [!NOTE]
> This project makes use of AI-generated/assisted code. Do not use this in any serious or production environment. This is for educational purposes only.

## License

MIT License. See `LICENSE`.
