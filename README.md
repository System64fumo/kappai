# kappai

> [!NOTE]  
> AI use disclosure: This project makes use of AI-generated/assisted code.

> [!WARNING]  
> This project is very early in its development cycle and is highly experimental.

Kappai is a local AI inference engine that heavily focuses on CPU performance.

It doesn’t invent anything fundamentally new. Instead, it takes what already works and tries to do it better.<br>
What started as a fun side project has grown into something much larger than originally anticipated.

### Features
- High-performance CPU inference
- Vulkan backend
- Mixed backend / partial offloading
- MoE streaming
- OpenAI-compatible server

### Supported Model Architectures
- Llama
- Gemma 4 (Dense and MoE)
- GLM (DSA)
- LFM

### Platform Support (Linux only)

| Architecture | Status       |
|--------------|--------------|
| aarch64      | First-class  |
| x86_64       | Supported    |
| riscv64      | Experimental |

## Getting Started

### Building

```bash
make config BUILD=release
make -j$(nproc)
```

With Vulkan: (W.I.P, Slower than CPU in some cases)

```bash
make config BUILD=release BACKENDS=vulkan
make -j$(nproc)
```

### Usage

Interactive mode:
```bash
./build/kappai-cli -m model.gguf
```

One-shot:
```bash
./build/kappai-cli -m model.gguf -p "Hello"
```

### Notes

This project exists for educational purposes and experimentation.

If you like the project and want to support it, you can:
- Star the repository
- Contribute code
- Offer hardware donations

### Thank You

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the GGUF format and for being the best local LLM inference engine.
- [colibri](https://github.com/JustVugg/colibri) for motivating me to finish the streaming feature.
- [z.ai](https://z.ai) for their excellent models that helped throughout this journey.
- My friends and future contributors for taking interest in this project.

### License

MIT License. See `LICENSE`.