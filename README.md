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

### CPU performance notes

On x86_64 release builds with `CPU_ARCH_OPT=1`, the default weight policy keeps Q4_K and Q6_K matmul weights in their native GGUF layout. Their native x86 kernels outperform the generic R8 batch kernels on tested hardware. Q5_K remains repacked by default because its R8 kernel performed better in our tests. Other architectures and scalar-only builds retain their existing repacking defaults. `--repack all` explicitly restores Q4_K/Q6_K R8 repacking, and `--repack none` disables all weight repacking.

To compare the K-quant kernels on your CPU, run `./build/test --bench --n 2048 --k 2560 --m 1 16 128 --iters 5`. This is a matmul microbenchmark, not a measure of complete model throughput. For an end-to-end comparison, run the same GGUF and prompt with and without `--repack all`, keeping thread count, context, generated token count and other options identical. Report prompt processing and generation separately; speed relative to llama.cpp depends on the model, workload and CPU.

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