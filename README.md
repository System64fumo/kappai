# kappai

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/System64fumo/kappai)](https://github.com/System64fumo/kappai/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/System64fumo/kappai)](https://github.com/System64fumo/kappai/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/System64fumo/kappai)](https://github.com/System64fumo/kappai/pulls)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/y6p3ua995Y)
[![Matrix](https://img.shields.io/badge/Matrix-Join-000000?logo=matrix&logoColor=white)](https://matrix.to/#/#kappai:matrix.org)
[![Donate](https://img.shields.io/badge/Donate-PayPal-00457C?logo=paypal&logoColor=white)](https://www.paypal.com/ncp/payment/EXAB48VXV3XJJ)

Kappai is a local AI inference engine that heavily prioritizes performance and efficiency.

The aim is to cut the hardware, time and electricity it takes to run models locally. It is usually faster than other engines, though how much depends on your hardware, model and quant. See [benchmarks](docs/benchmarks.md) for numbers.

## Features

- CPU inference tuned for `aarch64` and `x86_64`
- Vulkan and OpenGL backends
- A scheduler that mixes CPU and GPU
- MoE expert streaming
- An OpenAI-compatible server with tool calling
- Decision models
- Image generation

## Getting started

Head to [installation](docs/installation.md) to build kappai, then [usage](docs/usage.md) to run something.

See [support tables](docs/support-tables.md) for supported quants, models, backends and platforms.

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
