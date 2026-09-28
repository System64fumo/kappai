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

1. **CUDA-graph replay** — the state machine + `recipe_ple_fill_host` are now
   ported (`compute.c`, `recipe.c`), but **capture currently aborts** during
   the decode forward ("operation failed due to a previous error during
   capture") and cleanly falls back to eager. Graphs are therefore **opt-in**
   here (`KAPPAI_CUDA_GRAPH=1`); the dev tree has them default-on. Next step:
   audit the single-token forward for capture-illegal calls (sync
   `cudaMemcpy`/`cudaMalloc`) the way the dev tree does — several async
   variants (`buffer_write_async`, `copy_buffer_async`) exist in the ported
   backend and the PLE upload already uses the async form.
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

1. **`BUILD=release` segfaults in the arch self-test** (`run_arch_tests(cpu,
   NULL)`, `test_arch.c`) on pristine `cuda-wip` — reproducible in a clean
   worktree at `45f5d98` with no CUDA code present. The default `release-rdbg`
   (ASan) build passes 464/0/9, so this looks like uninitialised-memory / UB
   that ASan masks. Worth a dedicated hunt (MSan or code review).
2. **Flaky test**: `arch.generate[glm-dsa] decode step 2` alternates pass/fail
   between identical runs of the same binary.
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
