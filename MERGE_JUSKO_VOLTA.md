# Tesla V100 (Volta) from the `jusko` fork — integration notes

Port of the **Volta / SM70** pieces of
[`jusko-llama-volta-qwen3flash`](https://github.com/jackjusko/jusko-llama-volta-qwen3flash)
(local clone `C:/Users/confu/llama/jusko-llama-volta-qwen3flash`, tip `c3d51d71d`)
into this tree, on top of the turboquant + adaptive-KV-streaming integration and
on top of the earlier `anyei` V100 port recorded in `MERGE_V100.md`.

| source | branch in this repo | upstream base | what was taken |
|---|---|---|---|
| `jusko-llama-volta-qwen3flash` @ `c3d51d71d` (local) | `jusko-volta-port` → fast-forwarded into `turbo-kvstream-merge` | `465e49b9c` (2026-09-07) | the Volta/SM70 items only — 15 files |

Result: **15 files, +1070 / -49**, landed on the integration branch
`turbo-kvstream-merge` as `25d31990b`.

```
d420d3bfc  turbo-kvstream-merge   (turboquant KV cache + adaptive KV streaming + anyei V100 port)
  └─ 25d31990b  ggml-cuda: port the Tesla V100 (Volta) pieces of the jusko fork
       └─ (this file, MERGE_JUSKO_VOLTA.md)
```

## Why the fork is not the base fork of the earlier port

The two V100 forks in this workspace are unrelated, and they turned out to be
**complementary rather than competing**. Checked item by item before porting:

| mechanism | `llamacpp-v100` (anyei, already merged) | `jusko` (this merge) |
|---|---|---|
| sm70 MMVQ tuning | an `MMVQ_PARAMETERS_VOLTA` parameter table | new GEMV *variants* (`GGML_CUDA_VOLTA_Q5_X4`, `_Q6_W4R4`) |
| multi-GPU | one-shot P2P NVLink AllReduce for `-sm tensor` | — |
| FlashAttention | — | sm70 q8_0 tensor-core kernel, 256x256 compact specialization |
| GatedDeltaNet | — | `128x4` Volta prefill kernel |

Neither fork supersedes the other, so nothing already merged had to be undone.

## Source shape

The fork is a distribution, not a single feature. Against the merge base
`465e49b9c` it carries **57 commits / 82 files / +12107 / -434**, and only a
small part of that is Volta-specific. It has two layers:

- **base layer** — Jirka Svítil's `v100-optimized` lineage (`perf(cuda): add V100
  optimized runtime stack`, `perf(cuda): tune Volta Qwen prompt attention`, …).
  This is the actual "for V100" work.
- **local layer** — the Qwen3.8-Flash-Next (`qwen4exp`) optimization layer by
  Jack/John Jusko: QSA block-sparse indexer modes, PLE residency, MoE
  hot-expert VRAM cache, server slot work. This is a *model* project that
  happens to run on V100; it is not a V100 optimization.

The fork documents both layers itself in `README-FORK.md` and
`HANDOFF_2026-09-05_SM70_SM75.md`. It never touches upstream non-Volta paths
except through the shared FA template, so the port is additive.

Two orientation facts that shaped the port:

- **Upstream had already built part of this itself.** `VOLTA_MMA_AVAILABLE`
  exists in our tree from upstream `31c511a96` ("CUDA: Volta tensor core support
  for MMF", #16843) and `lightning-indexer.cu` from upstream `3b5321936`
  (#25545). The fork's sm70 work sits *on top of* those, not instead of them.
- **Upstream fixed the same divergent-barrier bug more cleanly.** The fork
  restructures the meta-combine block of `flash_attn_ext_f16_process_tile` to
  un-divergence the `__syncthreads()`. Upstream did the same thing in
  `b74f590ea` ("ggml-cuda: fix divergent barrier in f16 flash attention",
  #27870) by hoisting the whole block under `if (np > 1)`. **We kept upstream's
  structure and inserted only the fork's `combine_scratch` delta into it**,
  rather than reinstating the fork's version of that block.

## What was taken

### A. FlashAttention (the main speed and context item)

1. **`ggml/src/ggml-cuda/fattn-q8-volta.cuh` — new file, 336 lines.**
   `namespace ggml_q8v`: a W=4 sm70 tensor-core attention path over *ordinary
   q8_0 KV*. The persistent cache stays q8_0; K/V are widened to FP16 only
   while a 16-key tile is loaded into shared memory, then consumed by
   `mma.sync.m8n8k4`. Selected by `GGML_CUDA_VOLTA_Q8_FATTN_TC=1` (opt-in,
   default off) via a new `BEST_FATTN_KERNEL_VOLTA_Q8_W4` dispatch case.
   This is also the **context** item: q8_0 KV is half of f16, so it doubles the
   KV capacity available at a given context length on the same card.

2. **`fattn-mma-f16.cuh`.** Three new config rows in `get_config_volta`
   (NInfer-derived D256 32- and 64-column tiles, plus a 128x128x64 tile), an
   `ggml_cuda_fattn_mma_get_nbatch_fa_layout()` override for the exact
   D=192/DV=128/4x16 geometry, and **`flash_attn_ext_f16_volta_compact`** — a
   Volta-only specialization using 96-wide K tiles, a 64-half2 combine scratch
   and a forward K walk, for 48 KiB of dynamic shared memory instead of the
   generic layout. The three generic call sites of `_process_tile` pass
   `-1, -1, false`, so every pre-existing instantiation is byte-identical.

3. **`fattn.cu`.** The `BEST_FATTN_KERNEL_VOLTA_Q8_W4` enum/alloc/dispatch
   wiring, `GGML_CUDA_VOLTA_GQA8_NCOLS2=2` (exact long-K GQA8 geometry), and the
   measured 16-column crossover for 24-head / D256 / GQA4 q8_0 appends
   (`Q->ne[1] > 16 && Q->ne[1] <= 512`).

### B. Decode and prefill matmuls

4. **`mmvq.cu` / `.cuh`.** `vec_dot_q5_K_q8_1_x4()` decodes one Q5_K weight
   fragment and reuses it across four q8_1 activation columns
   (`GGML_CUDA_VOLTA_Q5_X4`, ncols_dst == 4, no ids, no fusion); a
   four-rows-per-block Q6_K launch (`GGML_CUDA_VOLTA_Q6_W4R4`) built from the
   generic warp count. Implemented as two new template parameters
   (`q5_x4`, `rows_per_block_override`) with defaults, so all existing
   instantiations and the tuned `MMVQ_PARAMETERS_VOLTA` table are untouched.

5. **`mmq.cu` / `.cuh`.** On Volta, a Q6_K MMQ with `J >= 48` now takes the
   Pascal DP4A config (I: 128 -> 64, occupancy: 1 -> 2) rather than the Ampere
   one — smaller row tiles keep the register count down. Also
   `GGML_CUDA_VOLTA_FORCE_MMQ=moe`, an opt-in routed-MoE override.
   *Note:* the DP4A table only covers `J ∈ {8,16,24,32,40,48,64}` for Q6_K, so
   `J >= 72` resolves to the `GGML_TYPE_COUNT` sentinel. That is safe rather
   than silently wrong: `mul_mat_q_switch_J()` and `ggml_cuda_mmq_get_J_max()`
   both skip sentinel configs, so those instantiations are never selected.

6. **`ggml-cuda.cu` + `common.cuh` + `--prefill-reuse`.** Volta-gated GEMM
   tiling for quantized prompt matmuls: dequantize the weight to F16 once and
   then submit independent output-column tiles of `ctx.prefill_reuse` columns to
   cuBLAS. This keeps the *small-ubatch N shape* — and therefore the
   floating-point accumulation order — while amortizing the weight conversion
   across a large prefill. Wired through `ggml_backend_cuda_set_prefill_reuse`
   (a proc address, so the backend ABI is unchanged) and a new
   `llama_context_params::prefill_reuse`.

### C. GatedDeltaNet

7. **`gated_delta_net.cu`.** `gated_delta_net_cuda_128x4_volta`: for `S_v == 128`,
   scalar gates (`KDA == false`) and `n_tokens > 1`, each warp owns four state
   columns and reuses q/k/g/beta across them, so the recurrent state is
   streamed once instead of four times.
   **Deliberate narrowing:** the fork enables this on Volta *and* Turing; we
   gate on `cc == GGML_CUDA_CC_VOLTA` only, because SM75 was never measured
   here (see the fork's own `HANDOFF`: "the SM75 paths ... remain unproven").

## Usage

```bash
# --- A1: q8_0 KV tensor-core attention (opt-in) ---
GGML_CUDA_VOLTA_Q8_FATTN_TC=1 ./build/bin/llama-server -m model.gguf -fa on -ctk q8_0 -ctv q8_0 -c 65536

# --- A3: exact long-K GQA8 geometry (opt-in) ---
GGML_CUDA_VOLTA_GQA8_NCOLS2=2 ./build/bin/llama-server -m model.gguf -fa on -c 163840

# --- B4: decode GEMV variants (opt-in) ---
GGML_CUDA_VOLTA_Q5_X4=1 ./build/bin/llama-cli -m model-q5_k_m.gguf
GGML_CUDA_VOLTA_Q6_W4R4=1 ./build/bin/llama-cli -m model-q6_k.gguf

# --- B5: routed-MoE MMQ override (opt-in; leave unset for dense models) ---
GGML_CUDA_VOLTA_FORCE_MMQ=moe ./build/bin/llama-server -m moe-model.gguf

# --- B6: prefill weight reuse (CLI flag) ---
./build/bin/llama-server -m model.gguf -b 1536 -ub 768 --prefill-reuse 768
```

Everything above defaults to **off**, except the FA config rows and the compact
specialization, which are selected automatically for the shapes they were tuned
for and are inert elsewhere.

## Deliberately *not* ported

Judged against one criterion, agreed with the user: **does it make llama.cpp
faster on a V100 and/or enlarge the usable context?**

| item | size | why not |
|---|---|---|
| Qwen3.8-Flash-Next / `qwen4exp` layer — QSA block-sparse indexer modes (`QWEN4EXP_QSA_*`), `qwen4exp.cpp` (+760), HC combine, `lightning-indexer` QSA geometry, KV high-water compaction, PLE VirtualLock, CPU `GET_ROWS` threading | ~40 files | This is *model* support, not V100 tuning. It needs the Qwen3.8 graphs and checkpoint layout; it is not inert code you can carry in a general tree. Would be a separate project. |
| `--moe-cache` (hot-expert VRAM cache): `ggml-backend-moe-cache.h`, `moe-cache.cu` (3256 lines), `.cuh`, sched sessions in `ggml-backend.cpp`, CPU `mul_mat_id` hooks, fused `mmvq` entry | ~4200 lines | Hardware-agnostic backend feature whose measured win is in the *CPU-spill* regime (2x V100 + P40), not on a V100 that holds all experts in VRAM. The fork itself runs it with `--moe-cache off`. |
| all-`-inf` FA tile skip (`fattn-tile.cuh`, `fattn-vec.cuh`) | +111 | Benefit requires a mask with **interior** all-`-inf` tiles, i.e. the QSA block-sparse path; on a dense causal mask it only adds a per-tile mask peek. The fork's gain (TG-60k 23.7 -> 24.8) was measured on Qwen3.8 QSA, whose layer is out of scope. |
| `RMS_NORM` + `SCALE` fusion (`norm.cu`, `ggml-cuda.cu` can_fuse) | +78 | No reproducible hit shape in either tree: the fork's own `qwen4exp` graph emits `rms_norm -> reshape -> mul`, and in our tree `GGML_OP_SCALE` appears only for attention sinks and in the samplers. Unmeasured, so not carried. |
| `GGML_CUDA_VOLTA_DECODE_MMA` / `_DECODE_VEC` | +10 | The fork's own README marks both **killed**: `_DECODE_MMA` crashed CUDA init, `_DECODE_VEC` gave no TG win. Carrying a knob that is documented to crash is worse than not having it. |
| MTP shortlist (`data/mtp-shortlists/*.i32`, `ggml_cuda_try_mul_mat_vec_q_mtp_shortlist`) | +250 + 512 KiB data | Hard-coded to one checkpoint's geometry (`src0->ne[0] == 5120 && src0->ne[1] == 248320`) plus a sidecar file. Not a general optimization. |
| mirrored-input async copy, `--pipeline-copies` / `ggml_backend_sched_new_ex(n_copies)`, `ggml_backend_meta_tensor_is_mirrored` | ~150 in `ggml-backend.cpp` | Sits on the tensor-parallel meta backend that `MERGE_V100.md` deliberately did not take, and the fork's own handoff says useful overlap "has not been demonstrated". |
| Turing / SM75 tuning: `GGML_CUDA_TURING_CUBLAS_MIN_BATCH`, Turing GQA6 `ncols2=2`, the 256x256 Turing config change (`nbatch_K2` 64 -> 32), the `__CUDA_ARCH__ == 750` rescale skip | ~30 | Target hardware is a V100. The fork's author is explicit that SM70 and SM75 paths are separate and geometry-gated, and that SM75 gains are unproven. |
| server/speculative/kv-cache work: shared slot prefixes, slot `erase` without `--slot-save-path`, recurrent checkpoint/replay, `--spec-mtp-defer-prompt` | ~900 | Not Volta-specific. `src/llama-kv-cache.cpp` in particular is exactly where our adaptive-KV-streaming fork already diverges, so merging it would be a second, unrelated project. |

## Integration decisions worth recording

1. **Upstream's barrier fix won.** See above — the fork's `__syncthreads()`
   restructuring was dropped in favour of upstream `b74f590ea`, and only
   `combine_scratch` was threaded into upstream's block.

2. **The port target had moved.** Our `fattn.cu` is +2705 lines and
   `ggml-cuda.cu` +1790 relative to the fork's sync point, because turboquant
   and adaptive-KV-streaming rewrote them. The fork's hunks were re-anchored by
   hand rather than merged, and `flash_attn_ext_f16_case` had been refactored by
   the KV-streaming merge into `..._case_impl<..., output_partial>` — the
   `use_volta_2cta` block was therefore placed inside `_case_impl` and gated on
   `!output_partial`, since the compact kernel does not implement the partial
   output path.

3. **The FA template changes are additive by construction.** All new template
   parameters have defaults; the only call sites that had to be touched are the
   three inside `flash_attn_ext_f16` itself, which now pass `-1, -1, false`.

4. **`--prefill-reuse` is Volta-gated at the point of use** — i.e. `reuse_n` is
   forced to 0 unless `volta_mma_available(cc)`, so the flag is inert elsewhere
   and needs no user-side guard.

## Verification

Performed (Windows, MSVC 14.40, CUDA 12.4, `CMAKE_CUDA_ARCHITECTURES=70`,
`-allow-unsupported-compiler`):

1. **Targeted CUDA build, sm_70 — clean.** The five touched translation units
   (`fattn.cu`, `mmvq.cu`, `mmq.cu`, `gated_delta_net.cu`, `ggml-cuda.cu`)
   compile with no errors. `fattn.cu` is the TU that pulls in
   `fattn-mma-f16.cuh` and the new `fattn-q8-volta.cuh`, so those headers are
   covered by a real compile, not just a syntax read.
2. **Template-instance coverage.** `ggml/src/ggml-cuda/template-instances/` is
   picked up by a CMake **glob** and is where the MMQ and FA kernels are really
   instantiated. Two representative instance TUs —
   `mmq-instance-q6_k.cu` and `fattn-mma-f16-instance-ncols1_32-ncols2_2.cu`
   (the latter is exactly the D256/32x2 case the compact kernel targets) — were
   compiled to PTX at both `sm_70` and `sm_80` as part of the A/B below, so they
   compile. The remaining instance files use the same `DECL_*` macros, which
   name only the top-level case function, so the added template parameters
   cannot shift positionally.
3. **CPU build — clean and complete** (`GGML_CUDA=OFF`, tests on, tools on).
   Scope note: the whole diff except the `--prefill-reuse` plumbing is under
   `ggml/src/ggml-cuda/`, which a CPU build does not compile at all. The CPU
   build only exercised `common/arg.cpp`, `common/common.cpp`, `common.h`,
   `include/llama.h`, `src/llama-context.cpp` and `src/llama-cparams.h`; it is
   *not* evidence about the CUDA diff.
4. **PTX `.maxntid` A/B** (`sm_70` vs `sm_80`, current tree vs a git worktree
   checked out at the pre-port commit — the worktree matters, because a
   header-only change is invisible if you only copy the `.cu`):
   - `template-instances/mmq-instance-q6_k.cu`, `sm_70`: kernel count and
     `.maxntid` histogram **unchanged** (`{128: 32, 256: 32}`). That first looks
     like a no-op, and the reason is instructive: `.maxntid` is only the CASE
     field `nthreads`, which is 256 in *both* the Ampere and the Pascal DP4A
     Q6_K rows. The rows differ in `I` (128 vs 64) and occupancy (1 vs 2).
     Measuring per-kernel PTX body size instead shows the change exactly where
     the `J >= 48` guard says it should:
     `J = 8/16/24/32/40` — **byte-identical**; `J = 48` 9137 -> 2571 lines,
     `J = 64` 11197 -> 3138 and 12126 -> 3327 lines; `J >= 72` collapse to the
     74-line sentinel stubs that the dispatcher skips (see B5 above).
   - `mmvq.cu`: 317 -> 319 kernels, i.e. exactly the two new instantiations
     (`q5_x4`, `rows_per_block_override == 4`), each still at `.maxntid 128`;
     the other 542 entries are pure mangled-name renames from the appended
     defaulted template parameters, with `.maxntid` preserved in every pair.
   - `fattn-mma-f16-instance-ncols1_32-ncols2_2.cu`: 36 -> 38 kernels — exactly
     the two `flash_attn_ext_f16_volta_compact<256,256,32,2,use_logit_softcap
     =false/true>` instantiations, at `.maxntid 128`. All 36 pre-existing
     kernels are byte-identical. (The symbol is emitted for any arch that
     defines `VOLTA_MMA_AVAILABLE` in that TU; on non-Volta its body is
     `NO_DEVICE_CODE`, and what actually gates the behaviour is the runtime
     `cc == GGML_CUDA_CC_VOLTA` check at the dispatch site.)

**Not verified — read this before trusting the port on hardware:**

- **Nothing was executed.** This machine has no NVIDIA device, so there is no
  end-to-end generation, no `test-backend-ops`, and no token/shape cross-check
  against the fork. Every speed figure quoted above is the **fork's own
  measurement**, not a re-measurement.
- **The `sm_80` control side of the A/B above is incomplete**: the FA instance
  and `gated_delta_net` A/B runs were interrupted by environment problems
  (stale `.ninja_log` / `index.lock` handles on this box, and a wrapper that
  silently re-parsed stale PTX artefacts before it was hardened). The Volta
  gating is therefore argued from source — every use is behind
  `cc == GGML_CUDA_CC_VOLTA` or `__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA` — and
  directly demonstrated only for `mmq-instance-q6_k.cu`.
- **Not compiled here:** a full `ggml-cuda` library build (all ~370 targets),
  and the Metal / Vulkan / SYCL backends. Our targeted build covers the touched
  TUs, not the whole backend.
- **The compact kernel's 48 KiB `nbytes_shared_compact`** is the fork's
  constant; it was not re-derived from `ggml_cuda_fattn_smem_swizzle::tile_stride`
  for the 96/64 overrides. If it is wrong the symptom is shared-memory
  corruption, which only a device run can show.
- **`GGML_CUDA_VOLTA_Q8_FATTN_TC`** (the q8_0 tensor-core path) and
  `GGML_CUDA_VOLTA_Q5_X4` / `_Q6_W4R4` are all opt-in and off by default, so the
  default configuration is what the A/B above actually validates.
- **`REPACK` interaction:** the fork's launch config uses `REPACK=1`, and
  `--prefill-reuse` reuses one dequantized weight across column tiles. Whether
  the repacked layout is still what the cuBLAS path expects at `ub 768` is a
  runtime question.
