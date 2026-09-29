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

**MoE paths:** diffed `op_moe_experts`, `op_moe_shared`, `moe_router_batch`,
`moe_weight_view`, `moe_experts_batch` against the development tree — they are
byte-identical, so there is nothing to port; MoE device support is exactly the
same as in the dev tree (and untested here, no MoE model available).

Still to port from the development tree for full parity:

1. **CUDA-graph replay — working (default ON).** The state machine +
   `recipe_ple_fill_host` are ported, capture-illegal calls fixed (async
   KV-shared copy, async PLE slice copy), and the **sampled-path glue** is
   ported too (NULL logits + explicit host readback). Greedy AND sampled
   decode now replay the captured graph: TG ~41 -> 67 (short prompt).
   Capture completes (902 nodes). `KAPPAI_CUDA_GRAPH_DISABLE=1` disables.
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

## Known differences vs the development tree (engine-level)

The CUDA *backend* is at parity with the development tree (kernels byte-identical;
`cuda.c` = dev + the staging/robustness fixes above; identical vtable).

**Prefill chunk sizing — FIXED (the real PP gap).** The fork used an 8 MB
prefill workspace target and clamped chunks to 512; the dev tree uses 64 MB /
2048. Ported the dev values (`PREFILL_CHUNK_WS_TARGET_BYTES`, clamp). Measured
Q8_0 pp500 **PP 223 -> 353 (+58%)**; pp1000 322. Chunk size is the dominant PP
lever here (it sets the batch GEMM M).

**Fused engine ops (`rmsnorm_matmul_multi`, `qkv_norm_rope`, `rmsnorm_add_batch`,
`matmul_multi_batch_resid`)** — the fork's engine doesn't emit them, but they are
**not the PP gap**: for prefill the dev tree also *decomposes*
`rmsnorm_matmul_multi` into `rmsnorm_batch` + `matmul_multi_batch` (my delegating
port measured no gain), and decode is covered by CUDA graphs (whole step replays
as one launch). An apparent "+17% PP" from wiring the FFN fusion was a bug
artifact (zeroed `n_out` skipped the GEMM). Porting them is therefore low value;
not pursued.

## Port status after merging misc/improvements (2406645)

`cuda-wip` now contains `misc/improvements` (and therefore `main`), so the CUDA
backend is a dlopen'd library like the CPU and Vulkan ones. The engine dropped
from 1.93 MB to 878 KB and has zero CUDA runtime references; the CUDA code lives
in `build/backends/libkappai_cuda.so`.

### Ops the CUDA backend implements natively

All 29 pre-merge ops, plus these four that `misc/improvements` introduced and
that the engine dispatches through `OP_BACKEND`:

| Op | Why it must be native |
| --- | --- |
| `softcap` | gemma-4 final logit softcap; NULL slot segfaults on first prefill |
| `split_qgate` | passes Q/K slot buffers with no staging |
| `attn_output_gate` | in-place on the attention output slot |
| `partial_rope_qk` | Qwen3.5; rope over the first `rope_dim` dims only |

`OP_BACKEND` only *reroutes* to a host backend, it does not stage, so a NULL slot
for an op that receives device buffers is a segfault rather than a slow path.
All four are covered by tests, and the tests were verified to fail when the
kernels are broken.

`moe_activate` is also native now. `moe_experts_batch` stays NULL on purpose: it
is gated on `BCAP_MOE_EXPERT_RESIDENT`, so the engine's own device-fallback path
handles it with reporting. Wiring it up would mean keeping expert weights
resident, which is a much larger change.

`gated_delta_net` stays NULL and is correct: `op_gated_delta_net` stages its
inputs to host and the hybrid state is plain host `float *`, so the host fallback
is the intended path. It costs prefill bandwidth on recurrent layers, not
correctness.

### repack_plan / repack_weight are intentionally NULL

`misc/improvements` moved weight repacking behind two new backend hooks.
Implementing them on CUDA would be **dead code**: `model.c` only consults them
when `do_repack` is set, and `do_repack` requires `home_is_cpu`.
`backend_weight_home()` returns the device backend whenever it has a `matmul`
op, so for CUDA `home_is_cpu` is false, `do_repack` is false, and
`repack_plan` is never called.

CUDA's quad-major relayout and the lossless Q4_0 -> Q8_0 promotion therefore
stay in `model.c`'s `if (!re_type)` block, which is exactly the path CUDA takes.
The hooks also cannot express the promotion: `repack_weight` takes one input
type and allocates `ggml_row_size(re_type)`, so an 18 B/32-element Q4_0 row
cannot become a 34 B/32-element Q8_0 row through that contract.

### Performance vs the pre-merge fork (gemma-4 E2B Q8_0, RTX 2070 Max-Q)

Measurement caveat: this machine's CUDA PP variance is large, so A/B runs are
**interleaved** (fork, wip, fork, wip, ...) rather than blocked, and CPU PP varies
by ±17% run to run. An earlier blocked comparison reported a CPU regression that
did not survive interleaving.

| | fork e35ba92 | merged (before) | merged (after) |
| --- | --- | --- | --- |
| PP, 408 tok, CUDA (interleaved, 4 pairs) | 305-384 (mean 359) | 321 (3 runs) | 334-342 (mean 338) |
| PP, 408 tok, CPU (3 runs) | 32.4-45.3 (mean 37.4) | 37.4-38.1 (mean 37.8) | unchanged |
| TG, CUDA | 45.5-46.1 | 44.8-46.1 | parity |

CPU and decode are at parity; the regression is **CUDA-prefill-specific**.

**Root cause: one kernel launch per token in the PLE batch path.** gemma-4 has
per-layer embeddings, and `ple_build_batch()` in the merged tree ran the
per-layer norm as a nested `for layer { for row { a->rmsnorm(...) } }` loop —
`n_layers * n_rows` launches (35 x 408 = 14280 for one 408-token prompt) — and
gathered the per-layer token embeddings with a `for row { a->embd_lookup(...) }`
loop, one launch per token. The pre-merge fork instead issued **one**
`ple_norm_batch()` and **one** bulk `buffer_write_f32()` for the whole batch.

Note the batching was already written but unreachable: `ple_norm_batch` is
implemented in `cuda.c` (`cuda_op_ple_norm_batch`, one block per row/layer) yet
was never called from the engine. `misc/improvements` replaced the batched call
with the nested loop.

Fix: call `a->ple_norm_batch` when the backend provides it (falling back to the
nested loop otherwise, so CPU is unaffected), and gather the PLE rows on the host
with a single bulk upload when `n_rows > 1` on a device backend. Single-row
decode keeps the device `embd_lookup` (no host round trip). `ple_host_decode_row`
already folds `n_embd_sqrt` into the row, so the host path skips the separate
`scale_inplace`. Result: **PP 321 -> 338 t/s** on CUDA, and bit-identical
generations to the pre-fix binary on Q8_0 / Q4_0 / Q4_K_M / IQ4_NL. The gate is
`!backend_has_cap(a, BCAP_IS_HOST)`, so CPU provably keeps the old path (verified:
identical output hash). Vulkan also picks up the bulk gather, but could not be
validated end to end because gemma-4 exceeds its `maxStorageBufferRange`
(427 MB vs 128 MB) and the suite has no PLE batch coverage.

Not causes, ruled out by measurement: repack layout (`--repack none` 56.2 vs 45.4
t/s fork-vs-merged *widens* the gap; `--repack all` 31.9 vs 37.0 favours merged;
`model_should_repack` and the R8 row geometry are byte-identical between trees);
chunk sizing (unchanged, single chunk for a 408-token prompt); `OP_BACKEND`
indirection and `profile_scope` per op (CPU is at parity, so the shared engine
dispatch is not the cost).

### Fixes carried on this branch

- `log_op_homes` snprintf overflow (`_FORTIFY_SOURCE=3` aborts any device backend
  reporting more than one missing op). Pre-existing on `misc/improvements`.
- 32-byte alignment for every AVX scratch buffer in `x86_64/quants.c`.
  `misc/improvements` aligns only `q_ymm_cache`; the other 15 per-tile caches were
  still grown with plain `xrealloc` and indexed as `__m256i`. IQ4_NL models
  segfaulted on CUDA because the staging path loses the allocator alignment
  lottery; CPU-only runs happened to win it.
- vulkan `partial_rope_qk` read the rope table past its end for hybrid-recurrent
  archs (strided with `head_dim/2` where `compute.c` builds it with
  `rope_dim/2`).

### Still broken upstream, not fixed here

- vulkan `partial_rope_qk` still disagrees with the scalar reference on hybrid
  archs after the out-of-bounds fix; the test reports those as a known-bug SKIP.
- The QAT `gemma-4-E2B_q4_0-it.gguf` fails to load: `jinja.c` `parse_call_args`
  does not support adjacent string-literal concatenation, which the canonical
  Gemma 4 template uses in `raise_exception("..." "...")`.
- `kappai-test --all` (several backends in one process) segfaults; reproduced on
  pristine `misc/improvements` with vulkan only.
- `lfm2.kvcache_reset_virgin_state` FAIL, 2x `arch.generate[glm-dsa]` FAIL, the
  orchestration prefix-reuse SKIP, and a UBSan null-deref in scalar
  `quantize_q8_0` are all pre-existing on `misc/improvements`.
