# kappai

Kappai is a local AI inference engine that heavily prioritizes performance and efficiency.

This engine does not aim to replace bigger/more serious engines rather it tries to make inference more accessible by making lower end systems go faster.<br>
You can expect up to **800%** speedups in some scenarios. (Results vary from system to system + model architecture + quantization)<br>


<table>
  <thead>
	<tr>
	  <th width="500px"> Features</th>
	  <th width="500px">Support table</th>
	</tr>
  </thead>
  <tbody>
  <tr width="600px">
	  <td valign="top">

- High-performance CPU inference
- Vulkan backend
- Mixed backend / partial offloading
- MoE streaming
- OpenAI-compatible server
- Tool calling

</td>
<td halign="right" valign="top">

| Quant family     | Models       | OS/Architecture    |
|------------------|--------------|--------------------|
| Q4_0, Q4_1, Q4_K | GLM 5.3      | linux / aarch64    |
| Q5_0, Q5_1, Q5_K | GLM 5.2      | linux / x86_64     |
| Q6_K             | Qwen 3.8     | linux / riscv64    |
| Q8_0             | Qwen 3.5     |                    |
| IQ3_S            | Gemma 4      |                    |
| IQ4_NL           | Llama 3.2    |                    |
| F32, F16, BF16   | LFM 2.5      |                    |

</td>
</tr>

  </tbody>
</table>

> [!WARNING]  
> This project is very early in its development cycle and is highly experimental.

<br>

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

---
<br>

> [!NOTE]  
> This project makes use of AI-generated/assisted code.<br>
> Do not use this in any serious/production environment, This is for educational purposes only.

### Contribution

If you like the project and want to support it, you can:
- Star the repository
- Contribute code
- Offer hardware donations

### Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the GGUF format and for being the best local LLM inference engine.
- [colibri](https://github.com/JustVugg/colibri) for motivating me to finish the streaming feature.
- [z.ai](https://z.ai) for their excellent models that helped throughout this journey.
- My friends and future contributors for taking interest in this project.

### License

MIT License. See `LICENSE`.