# Benchmarks

All benchmarks here are done with the same seed (1234), Context size (1024), Temperature (0.0 aka greedy).

## Radxa Orion O6 (With an rx 7600 xt)

**Model:** `gemma-4-E2B-it-IQ4_NL.gguf`<br>
**Backend:** `vulkan`<br>
**Kappai:** `[ PP: 435.22 t/s | TG: 118.92 t/s ]`<br>
**Llama.cpp:** `[ Prompt: 145.3 t/s | Generation: 56.7 t/s ]`<br>

## Radxa Orion O6

**Model:** `gemma-4-E2B-it-IQ4_NL.gguf`<br>
**Backend:** `cpu`<br>
**Kappai:** `[ PP: 83.87 t/s | TG: 25.94 t/s ]`<br>
**Llama.cpp:** `[ Prompt: 63.4 t/s | Generation: 16.8 t/s ]`<br>

## Orange Pi 5 Plus

**Model:** `gemma-4-E2B-it-IQ4_NL.gguf`<br>
**Backend:** `cpu`<br>
**Kappai:** `[ PP: 38.32 t/s | TG: 15.08 t/s ]`<br>
**Llama.cpp:** `[ Prompt: 31.0 t/s | Generation: 8.3 t/s ]`<br>
