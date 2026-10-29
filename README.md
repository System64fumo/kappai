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
| `lfm-d1-omni` | LFM 2 D1 Omni (text/JSON)     |
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
| `rocket`      | Experimental RK3588 NPU, batched Q8_0 matmul     |

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

The RK3588 NPU backend uses the mainline Linux `rocket` driver and the kernel's
`drm/rocket_accel.h` header. Build it with `BACKENDS=rocket` or
`BACKENDS=vulkan,rocket`, then select it explicitly with `--device rocket0`.
The user running kappai needs read and write access to `/dev/accel/accel0`.
No Rockchip runtime or model converter is required.

The experimental backend offloads `Q8_0` matrix operations with at least 32 rows.
It reuses CPU implementations for small batches, other weight formats, attention
and remaining operations. Prepared weights are cached within one quarter of
available memory, capped at 2 GiB; weights exceeding that budget use the CPU.
Weights are prepared on first use, so the first request is slower.

By default, it retains GGUF block scales and uses three compensated FP16 products
on the NPU. For faster INT8 execution, use
`--device rocket0 --npu-int8 --fuse qkv`. This option rotates channels and
quantizes activations and weights in groups of up to 512 channels, then combines
INT32 partial results on the CPU. It changes model numerics and remains opt-in.
INT8 prepared weights take roughly the canonical Q8_0 storage, plus scales and
padding; compensated FP16 takes about six times as
much. Both modes share device buffers, scheduling and CPU fallback. Contiguous
Q/K/V weights created by `--fuse qkv` run as one matrix operation.

Select `rocket0` explicitly: it is excluded from automatic device selection.
On RK3588 at 600 MHz, the D1 Omni 600M Q8_0 decision benchmark with a single
128-token NOUL question takes about 200 ms after warmup with INT8, QKV fusion and
eight CPU threads, versus 270 ms on CPU alone with QKV fusion. The fused INT8 pass
uses 470 device tasks in 210 submissions. Contraction groups for narrow outputs run
across the three cores, reducing partial products without changing quantization.
Nine existing decision cases pass the benchmark's 0.03 probability and confidence
tolerance; this is a limited numerical check, not a general model quality evaluation.
The default compensated FP16 path takes about 451 ms.
The 100 ms target requires further reductions in device work and CPU processing.

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

D1 and D1 Omni models accept `state` and `questions` at `POST /v1/systemone`. D1 Omni supports text and JSON input with `choice`, `score`, and `noul` questions.

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
