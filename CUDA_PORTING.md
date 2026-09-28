# CUDA backend port — status

Branch `cuda-port` (based on `cuda-wip`). Goal: bring the CUDA acceleration
backend into this tree **without touching CPU or Vulkan**.

## Landed (in order)

| Commit | Contents |
|--------|----------|
| `build: add CUDA backend build system` | Guarded CUDA Makefile section (no effect unless `BACKENDS=cuda`): backend auto-detection, nvcc `.cu` rule + arch auto-detect, WSL2 libcuda bind fix, optional cuBLAS (`HAVE_CUBLAS`). |
| `types: add Q8_0_QM / Q4_0_QM` | Two engine-internal enum values (`0x4A`/`0x4B`) + type-table rows. Never in GGUF files. |
| `backend: add vtable ops needed by CUDA` | Additive vtable slots + caps + buffer fields (`is_slice`, `host_is_pinned`, `fp16_shadow`). CPU/Vulkan memset the vtable, so new slots stay NULL. |
| `cuda: add CUDA backend` | `src/backend/cuda/{cuda.c,cuda_kernels.cu,cuda_internal.h}` (~12.5k lines). Compiles, links, probes, initialises. |
| `cuda: QM weight repack + Q4_0->Q8_0 promotion` | `cuda_repack.c/.h` (plain C, in the CUDA dir so CPU stays clean); `model.c` calls them under `#ifdef BACKEND_CUDA`. Both default-on; `KAPPAI_QMAJOR=0` / `KAPPAI_Q4_TO_Q8=0` opt out. |

## Not yet ported (blocking end-to-end run)

**Fixed:** the batch scratch was host-only. `bs_ensure_slot` allocated every
batch slot (X, Q/K/V, gate/up, attn_out, ...) with `float_buf_ensure_nocopy`
and set `handle = host_ptr`, so the CUDA backend received host pointers as
device handles and the first kernel faulted (`cudaErrorIllegalAddress`,
surfacing later as `status=-5`). Ported the device-aware allocation
(`owner->buffer_alloc_scratch`) + matching free. `ple_build_batch` and
`ple_proj_inject_batch` likewise aliased host float_bufs as device buffers;
ported the device-scratch paths (`ple_proj_gpu`, `ple_inp_dev`,
`ple_slice_dev`, `ple_norm_batch`), keeping the original host path for
CPU/Vulkan (gated on `BCAP_IS_HOST` / op presence) so neither is affected.

**Result:** the model loads, batch prefill runs, and **CUDA generation now
produces correct output** for gemma4-E2B (Q8_0 and Q4_0): coherent responses
and **byte-identical greedy output vs the CPU backend** (60 tokens, both
quants, `-t 0.0 -s 42`). `--warmup 0` no longer crashes.

Remaining host/device assumptions that are **latent** for gemma4-E2B (dense)
but will break MoE / fused-activate recipes: `op_moe_experts`,
`op_moe_shared`, `op_moe_router_batch` use `batch_buf_ptr()` in host loops.
Port those before targeting a MoE model.

Still to port from the development tree for full parity:

1. **CUDA-graph replay — now working (default ON).** The state machine +
   `recipe_ple_fill_host` are ported, and the capture-illegal calls are
   fixed: KV-shared K→V copy now prefers `copy_buffer_async`, and the PLE
   projection slice copy uses an async `copy_2d` (was a sync
   `compute_copy_buffer_cross`). Capture completes (**902 nodes**,
   866 kernel / 36 memcpy) and replays; greedy output stays byte-identical
   to CPU. TG 41 -> **68** (short prompt); sampled decode still runs eager
   (no sampled-graph glue here yet, so it does not arm graphs).
   `KAPPAI_CUDA_GRAPH_DISABLE=1` force-disables.
2. **`--gemv` decode-GEMV rig** — ported (`test_bench.c` + dispatch + usage).
   Run: `./build/test --gemv cuda`. It runs before the arch self-test, so it
   works even in `BUILD=release` (whose `./build/test` still segfaults — item 3).
   Porting it required a small build fix: `compute.c` now provides no-op stubs
   for the `cuda_graph_dbg_*` helpers in non-CUDA builds (they are CUDA-only;
   this also fixes the pre-existing CPU-only `make test` link failure).

   Fork `--gemv cuda` results (short-prompt machine, ~same as dev tree):
   `ffn-up-6k 138.7us/72GB/s`, `logits 1304.9us/328GB/s`, FFN+logits ~14.4ms.
3. Note: the fork's `BUILD=release` `./build/test` segfault (pre-existing,
   CUDA-independent) still blocks the per-op suite in release mode; use the
   default ASan build for CPU/Vulkan suites (464/0/9), and e2e runs or
   `--gemv` for CUDA.

Remaining planned commits:

1. `recipe.c/h`: rmsnorm-matmul-multi op, fused-split threshold, capability
   routing (adapt to this tree's `recipe.c`, do not paste).
2. `compute.c`: CUDA-graph replay state machine (capability-gated; port last).
3. `context.c`: sampled-path graph hook (2 lines, gated on `graph_launch`).
4. `test_bench.c`: `--gemv` decode-GEMV rig + CUDA test cases.
5. Resolve the `OP_PLE_BUILD` batch path against the ported backend.

## Pre-existing bugs found (NOT introduced by this port)

1. **FIXED** — `BUILD=release` `./build/test cpu` segfaulted. Root cause: the
   x86_64 Q4_K tiled matmul kept AVX scratch caches (`__m256i[MR][4]`) in
   `realloc` memory (16-byte aligned) but GCC emits 32-byte aligned
   `vmovdqa` stores → misaligned-access fault as soon as the tiled path ran
   (m ≥ NR). ASan's allocator masked it. Fixed by allocating all 19 AVX
   caches 64-byte aligned (`cache_alloc`, `x86_64/quants.c`). Release CPU
   suite now runs: **464/0/9**; `--bench cpu` completes. (Same code lineage
   exists in the development tree.)
2. **Flaky test**: `arch.generate[glm-dsa] decode step 2` alternates pass/fail
   between identical runs of the same binary.

## Remaining CUDA-suite issues (fork test suite is newer/stricter than the dev tree)

**Host-fallback staging — FIXED.** With the release suite fixed, `./build/test
cuda` used to crash because several CUDA-backend fallbacks passed device
buffers to the CPU reference (which dereferenced device pointers). Added
`cuda_host_stage_in/out/buf` and routed every fallback through host scratch
(`cuda_matmul`, `cuda_matmul_residual`, `cuda_op_matmul_ffn_down` x3,
`cuda_op_matmul_batch`; `cuda_matmul_multi` already degrades to the staged
`cuda_matmul`; `cuda_op_embd_lookup` fixed earlier). Result: **per-op CUDA
suite runs to completion in release, 399/0/71** (was a segfault). No CPU/Vulkan
edits.

**Q4_K_M / exotic quants — now works.** `gemma-4-E2B-it-Q4_K_M` (Q4_K/Q6_K
tensors, all served by the staged host fallback) previously produced empty
output; the staged `matmul_ffn_down` sized gate/up from `n` instead of `k`,
feeding garbage to the CPU reference. Fixed. It now generates coherent text
and tracks the CPU output closely (can flip a wording choice mid-run — the
same thin-race class as the graph path, since it goes through host math on
GPU-produced activations). It is slow by design (host fallback), ~PP 40 /
TG 5.5. Native-quant models (Q8_0/Q4_0) remain byte-identical to CPU.

## Note on graph replay numerics

Greedy decode with graphs ON is deterministic and coherent, but can differ
from the CPU reference by a thin-race flip in long generations (the documented
dev-tree S28 cascade class: the captured/replayed op ordering yields ulp-level
differences). With `KAPPAI_CUDA_GRAPH_DISABLE=1` the eager path is
byte-identical to CPU.

3. `make` (default target) fails on the `server` target when `microhttpd` is
   absent; `make cli test` is the working subset here.

## Verification notes / gotchas

- **Do not pipe test output through `stdbuf`** with the default ASan build —
  `stdbuf`'s `LD_PRELOAD` makes ASan abort with "runtime does not come first".
- Prefer `BUILD=release` for CUDA work; the ASan build cannot initialise CUDA
  in this environment (`cuda(unavailable)`).
- The Makefile enforces a config signature in `build/.build-config`; switching
  `BACKENDS` requires `rm -rf build` first (by design).
- No CUDA code has been copied from llama.cpp/ggml; the backend is original,
  sharing only the GGUF quant block layouts (file-format compatibility).

## Baseline numbers (this machine, RTX 2070 Max-Q / WSL2)

- Fork, CPU-only, `make BUILD=release cli test` then `./build/test cpu`:
  see `RUN` notes above; ASan build is the reliable one (464/0/9).
- Development tree (source of this port): Q8 pp500 PP ~362 / TG ~42,
  Q4 pp500 PP ~337 / TG ~42, teacher-force 115/116, suite 389/0.
