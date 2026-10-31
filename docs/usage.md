# Usage

Everything lives in `build/` once you've followed [installation.md](installation.md). Every binary has `--help` if you want the full list of options.

## kappai

```bash
./build/kappai
```

This is the one to start with. `kappai` rolls the other tools into a single terminal app, so instead of remembering a pile of command line flags you set things up once in a form, save it, and run it whenever you like.

The first launch walks you through a short setup: where your models live and whether you usually want chat or server mode. After that you get a list of your saved configs:

| Key | Action |
|-----|--------|
| `Enter` | Run the selected config |
| `←` `→` | Switch between chat and server |
| `N` | New config |
| `E` | Edit |
| `X` | Delete |
| `S` | Settings, including the model manager |
| `H` | Help |

The model manager can browse what you have installed and download models from Hugging Face, so you don't need to fetch anything by hand.

Configs are plain `.conf` files in `~/.config/sys64/kappai/configs/`, and the TUI's own settings are in `~/.config/sys64/kappai/tui.conf`. If `XDG_CONFIG_HOME` is set, that's used instead of `~/.config`.

Once a config exists you can skip the menu entirely:

```bash
./build/kappai my-config chat
./build/kappai my-config server
```

## kappai-cli

Terminal chat. You'll want a model first, for example Gemma 4 E2B from Unsloth ([unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF), the `IQ4_NL` file):

```bash
curl -LO https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-IQ4_NL.gguf
```

Then chat with it:

```bash
./build/kappai-cli -m gemma-4-E2B-it-IQ4_NL.gguf
```

Or ask a single question and exit:

```bash
./build/kappai-cli -m gemma-4-E2B-it-IQ4_NL.gguf -p "Why is the sky blue?"
```

A few flags you'll probably reach for:

- `--device vulkan0` picks a compute device. `--list-devices` shows what kappai found.
- `--ngl 20` puts the first 20 layers on the device and keeps the rest on the CPU.
- `-c 8192` sets the context size.
- `--system "..."` sets a system prompt.
- `-t 0` makes sampling greedy.

## kappai-server

An OpenAI-compatible server. Grab a model, for example Llama 3.2 3B from Unsloth ([unsloth/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF)):

```bash
curl -LO https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Start it:

```bash
./build/kappai-server -m Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

It listens on `127.0.0.1:8080` by default. Try it out:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

The engine flags from `kappai-cli` all work here too, plus:

- `--host` and `--port` change where it listens (`--host 0.0.0.0` exposes it to your network).
- `--api-key <key>` requires a Bearer token on `/v1` requests.
- `--slots <n>` sets how many sessions can run at once (default 4), each with its own KV cache.

## kappai-diffusion

Image generation with Qwen Image 2.1. It needs three files:

- Diffusion model: [qwen_image_2.1-Q4_K.gguf](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF)
- VAE: [qwen_image_2.1_vae_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) (in the `vae` folder)
- Text encoder: [Qwen3VL-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)

Generate something:

```bash
./build/kappai-diffusion \
  --diffusion-model qwen_image_2.1-Q4_K.gguf \
  --vae qwen_image_2.1_vae_bf16.safetensors \
  --llm Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  -p "a lighthouse on a cliff at sunset, oil painting" \
  --device vulkan0 --out-dir images
```

Leave out `-p` and you get an interactive session where every prompt makes a new image. Handy options: `-W` / `-H` for size (default 512, multiples of 32), `--steps` (default 20), `--cfg-scale` (default 6.0), `--seed`, and `-n` for a negative prompt. Without `--device` it runs on the CPU.