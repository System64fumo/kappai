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

`gfortran`-free build passes and the model loads, but prefill aborts with
`status=-5` (`ERR_OUT_OF_MEMORY`) in the **`OP_PLE_BUILD` batch op**.

**Root cause (confirmed 2026-09-27):** the fork's `ple_build_batch` aliases a
host pointer as a device buffer:

```c
float_buf_ensure(&ctx->bs->ple_proj, (size_t)n_rows * total_ple);
buffer proj_buf = {0};
proj_buf.handle   = ctx->bs->ple_proj.p;   /* host pointer used as device handle */
proj_buf.host_ptr = ctx->bs->ple_proj.p;
```

It predates device-only backends (CPU/Vulkan share host memory, so this worked
there). The CUDA backend treats `handle` as a device pointer, so the following
`matmul_batch(..., &proj_buf, ...)` fails and bubbles up as `-5`.

**Fix:** port the development tree's `ple_build_batch` device-scratch path:
allocate `ple_proj_gpu` via `buffer_ensure_scratch`, use the `ple_norm_batch`
op, keep a host fallback for backends without it. Needs new scratch fields
(`batch_scratch.ple_proj_gpu`, `compute_scratch.ple_proj_norm_w_gpu` /
`ple_proj_norm_w_uploaded` / `ple_proj_host` / `inpL_host`). Backend-neutral
(NULL checks), so CPU/Vulkan keep their path unchanged.

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
