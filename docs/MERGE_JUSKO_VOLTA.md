# Tesla V100 (Volta) from the `jusko` fork — integration notes

Port of the **Volta / SM70** pieces of
[`jusko-llama-volta-qwen3flash`](https://github.com/jackjusko/jusko-llama-volta-qwen3flash)
(local clone `C:/Users/confu/llama/jusko-llama-volta-qwen3flash`, tip `c3d51d71d`)
into this tree, on top of the turboquant + adaptive-KV-streaming integration and
on top of the earlier `anyei` V100 port recorded in `MERGE_V100.md`.

| source | branch in this repo | upstream base | what was taken |
|---|---|---|---|
| [`jusko-llama-volta-qwen3flash`](https://github.com/jackjusko/jusko-llama-volta-qwen3flash) @ `c3d51d71d` | `jusko-volta-port` → fast-forwarded into `turbo-kvstream-merge` | `465e49b9c` (2026-09-07) | the Volta/SM70 items only — 15 files |

Result: **15 files, +1070 / -49**, landed on the integration branch
`turbo-kvstream-merge` as `25d31990b`.

```
d420d3bfc  turbo-kvstream-merge   (turboquant KV cache + adaptive KV streaming + anyei V100 port)
  └─ 25d31990b  ggml-cuda: port the Tesla V100 (Volta) pieces of the jusko fork
       └─ (this file, MERGE_JUSKO_VOLTA.md)
```

## 激活方式汇总（重点速查）

**编译前置（硬性）**： `GGML_CUDA=ON` 且 `CMAKE_CUDA_ARCHITECTURES` 含 `70`。否则 Volta 设备代码是 `NO_DEVICE_CODE`，全部无效。

**本端口的全部开关都是运行时 `getenv` 读取的环境变量或 CLI 参数，没有一个是 cmake 编译选项**——同一个二进制即可 A/B，无需重编。

| 项 | 环境变量 / 参数 | 默认 | 语义陷阱 |
|---|---|---|---|
| A1 q8_0 KV 张量核注意力 | `GGML_CUDA_VOLTA_Q8_FATTN_TC=1` | 关 | 判定为 `getenv(...) != nullptr` → **`=0` 仍然启用**，必须 `unset` 才关 |
| A2 compact FA 特化（256×256） | `GGML_CUDA_VOLTA_FA_COMPACT=0` | **开** | 我们加的 kill-switch；`unset`/`=1` 保持移植行为，`=0` 退回通用 MMA |
| A3 长 K GQA8 精确几何 | `GGML_CUDA_VOLTA_GQA8_NCOLS2=2` | 关 | 按值 `atoi(env)==2` 判定 |
| B4 decode GEMV 变体 | `GGML_CUDA_VOLTA_Q5_X4=1`、`GGML_CUDA_VOLTA_Q6_W4R4=1` | 关 | 同 A1，`!= nullptr` 判定，`=0` 仍启用 |
| B5 路由 MoE MMQ 覆盖 | `GGML_CUDA_VOLTA_FORCE_MMQ=moe` | 关 | `strcmp(env,"moe")==0`，密集模型不要设 |
| B6 prefill 权重复用 | CLI `--prefill-reuse <N>`（如 `-ub 768 --prefill-reuse 768`） | 关 | 代码内 Volta 门控：非 sm70 强制 0，其它平台自动 inert |

### C 段：recurrent checkpoint tail replay（`--checkpoint-recurrent-prev`，后端无关、无需 CUDA/sm70）

| 项 | 参数 | 默认 | 激活条件 |
|---|---|---|---|
| 循环状态快照回放 | CLI `--checkpoint-recurrent-prev`（仅 server；`--no-checkpoint-recurrent-prev` 关闭） | 关 | 见下方自门控条件 |

- **编译前置不同**：该项**不依赖 CUDA，也不需要 sm_70**，CPU 构建即可编译并验证（与本文档其它 Volta 项不同，是纯后端无关的改动）。
- **自门控（满足才生效，否则静默 no-op）**：
  - 必须是 **recurrent / hybrid 模型**（`llama_n_rs_seq(ctx) > 0`）；普通 transformer 上 `n_rs_seq` 被钳为 0，什么都不做；
  - 必须设 `--ctx-checkpoints > 0`；
  - **不能**启用投机解码（`spec == nullptr`）——草稿-MTP 有自己独立的机制；
  - 仅在追加的新 token 数 **> 64** 时才打快照；
  - 设了 flag 且投机未占用时，`common_context_params_to_llama()` 会把 `n_rs_seq` 至少抬到 1，并仅在此时设置 `rs_rollback_prompt_only`。
- 示例：`./build/bin/llama-server -m qwen3.8-27b.gguf --ctx-checkpoints 32 --checkpoint-min-step 8192 --checkpoint-recurrent-prev`

A1/A2/A3 还需 `-fa on`；A1 还需 `-ctk q8_0 -ctv q8_0` 与大 `-c` 上下文；C 段无需 `-fa`，且后端无关。

**两个“默认开、无运行时开关”的项**（只能改源码回退，因 host/device 共享内存布局必须一致）：
- `fattn-mma-f16.cuh` 中三个新的 `get_config_volta` tile 行（按形状自动选中，其它形状惰性）；
- `mmq.cuh` 中 Q6_K 在 `J >= 48` 时路由到 Pascal DP4A 配置（I:128→64，occupancy:1→2）。

⚠ 复盘要点：`Q8_FATTN_TC`、`Q5_X4`、`Q6_W4R4` 是“**设置了就启用**”语义，`=0` 不等于关闭，必须 `unset`；compact FA 才是 `=0` 关闭。

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

# --- A2: turn the compact FA specialization *off* (it is on by default) ---
GGML_CUDA_VOLTA_FA_COMPACT=0 ./build/bin/llama-server -m model.gguf -fa on -c 65536

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

Every switch in this list is read at runtime with `getenv` — none of them is a
build option, so a single binary can be A/B'd without rebuilding. Be aware that
the *semantics* differ and this is inherited from the fork:

| switch | test | so |
|---|---|---|
| `GGML_CUDA_VOLTA_Q8_FATTN_TC` | `getenv(...) != nullptr` | `=0` still **enables** it |
| `GGML_CUDA_VOLTA_Q5_X4`, `_Q6_W4R4` | `getenv(...) != nullptr` | `=0` still **enables** it |
| `GGML_CUDA_VOLTA_GQA8_NCOLS2` | `atoi(env) == 2` | value-based |
| `GGML_CUDA_VOLTA_FORCE_MMQ` | `strcmp(env, "moe") == 0` | value-based |
| `GGML_CUDA_VOLTA_FA_COMPACT` | `atoi(env) == 0` ⇒ off | value-based; **on by default** |

`GGML_CUDA_VOLTA_FA_COMPACT` is our addition, not the fork's: the compact
specialization is selected automatically and the fork ships no way to disable
it, which makes a suspicion about it unrecoverable without a rebuild. `unset`
and `=1` keep the ported behaviour, `=0` falls through to the generic MMA kernel
(the compact block ends in `return`, so nothing else changes).

The remaining two default-on pieces of the port have **no** runtime switch, because
both are compile-time constants that the host and the device must agree on (host
`nbatch_fa`/`nbatch_K2`/... must equal the device `constexpr` values, or the shared
memory layout mismatches):
- the three new `get_config_volta` tile rows, and
- the `J >= 48` → Pascal DP4A routing for Q6_K in `mmq.cuh`.

Reverting either means editing source. See *Verification → Not verified* for what
that implies for the V100 bring-up.

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

5. **We added a kill switch the fork does not have.** `GGML_CUDA_VOLTA_FA_COMPACT=0`
   disables the compact specialization at runtime. The fork selects it purely from
   `cc`/geometry, so on a V100 the only way to test "is it the compact kernel's
   fault?" was to rebuild. Since the port makes the compact kernel part of the
   *default* V100 path (i.e. this is no longer an opt-in experiment), a
   no-rebuild escape hatch is worth the one `getenv`. Scope is deliberately one
   `if`: the kernel choice is a host-side branch, so the switch cannot desync the
   host-computed shared-memory size from the device `constexpr` values.

6. **The two other default-on pieces are not switchable, on purpose.** The
   `get_config_volta` rows and the Q6_K `J >= 48` DP4A routing are read by both the
   host (to size shared memory and pick `J`) and the device (as `constexpr`). A
   runtime switch there would have to change the host answer while leaving the
   device `constexpr` alone — that is a shared-memory size mismatch, i.e. exactly
   the failure mode we are trying to protect against. Reverting them must be a
   source edit.

7. **The two V100 forks do not collide on decode.** Both are in this tree now —
   `MERGE_V100.md` ported `llamacpp-v100`'s Volta GEMV *parameter table*, this port
   adds jusko's new GEMV *variants* — so their interaction had to be checked rather
   than assumed. It is safe, and mechanically so:

   - The Volta table differs from `MMVQ_PARAMETERS_GENERIC` at exactly **one** point:
     `ncols_dst == 1` for Q2_K…Q6_K, `nwarps` 4 -> 2. Every other cell, including all
     of `ncols_dst` 2..8, is identical (`mmvq.cu`, both switch blocks).
   - jusko's variants exist only at `ncols_dst == 4`: `_Q5_X4` is gated on
     `type == GGML_TYPE_Q5_K && c_ncols_dst == 4` in the kernel dispatcher, and
     `_Q6_W4R4` lives in the `case 4:` arm with `rows_per_block = 4`. Note `_Q6_W4R4`
     also deliberately launches with `calc_nwarps(..., MMVQ_PARAMETERS_GENERIC)`.
   - At `ncols_dst == 4` both tables return `nwarps == 4`, so the launch's
     `block_dims.y` on the host still equals the kernel's `__launch_bounds__` on the
     device — and equals the value jusko measured in a tree where Volta fell through
     to GENERIC.

   So no combination is unreachable, mis-sized, or silently re-tuned. Had either
   variant sat at `ncols_dst == 1`, the two ports would have changed each other's
   geometry — which is the trap worth remembering, since neither fork's own
   benchmarks could have shown it.

## Verification

Performed (Windows, MSVC 14.40, CUDA 12.4, `CMAKE_CUDA_ARCHITECTURES=70`,
`-allow-unsupported-compiler`):

1. **Targeted CUDA build, sm_70 — clean.** The five touched translation units
   (`fattn.cu`, `mmvq.cu`, `mmq.cu`, `gated_delta_net.cu`, `ggml-cuda.cu`)
   compile with no errors. `fattn.cu` is the TU that pulls in
   `fattn-mma-f16.cuh` and the new `fattn-q8-volta.cuh`, so those headers are
   covered by a real compile, not just a syntax read. `fattn.cu` is also the only
   TU that includes `fattn-mma-f16.cuh` (verified by grep), which is why the
   `GGML_CUDA_VOLTA_FA_COMPACT` change — a host-side branch in that header —
   needed only that one object rebuilt.
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

---

## 附：recurrent checkpoint tail replay（`--checkpoint-recurrent-prev`）合并记录

> 本段原载于 `MERGE_RECURRENT_CKPT.md`，因同属 `jusko-llama-volta-qwen3flash` 仓库的合并记录，已并入本文档。激活条件见上方「激活方式汇总 · C 段」。

Port of the `jusko-llama-volta-qwen3flash` commit that lets a long cached recurrent
context resume from a snapshot taken *just before* the appended suffix, instead of
replaying from the last periodic checkpoint. Unlike the other two ports in this
repo (`MERGE_V100.md`, this document's Volta parts) this one is **entirely backend
agnostic** — no CUDA, no Volta gating — so it is validated by a CPU build and can
be exercised on hardware without a V100.

| source | branch in this repo | upstream base | what was taken |
|---|---|---|---|
| `jusko-llama-volta-qwen3flash` @ `c3d51d71d`, commit `be6567cea` "perf(server): avoid recurrent checkpoint tail replay" | `jusko-recurrent-ckpt` | `465e49b9c`, cherry-picked as a real 3-way merge via the existing `jusko` remote | the mechanism, re-expressed into this tree's checkpoint policy |

Result: **12 files, +285 / −25**, plus these notes, landed on
`turbo-kvstream-merge`.

```
7692b3457  turbo-kvstream-merge   (V100 ports + docs, see the other MERGE_*.md)
  └─ <this>  server: add --checkpoint-recurrent-prev (recurrent checkpoint tail replay)
       └─ docs: 已并入 MERGE_JUSKO_VOLTA.md
```

## 为什么这个 commit 不是自包含的

The first attempt was a straight patch application, which reported conflicts in
5 files and looked like a light re-anchor job. That measurement was **wrong**: it
counted patch-application failures, not dependency closure. Reading the source
showed `be6567cea` calls `create_checkpoint(..., is_replay_boundary, ...)`, and
`is_replay_boundary` does not exist anywhere in this tree — nor in the fork's
base `465e49b9c`, nor upstream. It comes from the fork's **own** `0471a9885`
("perf: add V100 optimized runtime stack") — the same commit whose CUDA parts
(the FA compact kernel, GDN 128x4, `--prefill-reuse`) were already ported in
this document's Volta part, but whose **server-side half was deliberately excluded**.

So the closure is: `be6567cea` + the checkpoint-policy part of `0471a9885`. The
two options were to import that policy wholesale or to re-express only the
increment the feature actually needs. **This port does the latter** — see
"刻意未移植的部分" below for exactly what that costs.

## 取用了什么（What was taken）

Three independent pieces, plus the server wiring:

1. **`rs_rollback_prompt_only`** — a new `llama_context_params` field. During
   token generation a recurrent rollback plane is dead weight, so
   `models/delta-net-base.cpp` now builds the state tensors without it whenever
   `ubatch.n_seq_tokens == 1`. This is a graph-shape change, and it is the reason
   `n_rs_seq` alone is not enough: the feature wants snapshots *during prompt
   processing* and none afterwards.

2. **`LLAMA_STATE_SEQ_FLAGS_RECURRENT_PREV`** — a new `llama_state_seq_flags` bit.
   `llama_memory_recurrent::state_write` can now serialize rollback plane 1
   (the state immediately before the live one) instead of the live state, and
   `state_write_meta` gained a `pos_shift` so the snapshot's positions are
   reported one token earlier. That is what lets the server hand a suffix a
   prefix that ends exactly where the suffix begins.

3. **Device-backed checkpoints** — `common_prompt_checkpoint` gained
   `data_tgt_on_device` and `data_tgt_logical_size`. An `ON_DEVICE` checkpoint
   keeps its tensor payload in context-owned storage and carries only the
   serialization envelope in `data_tgt`, so `size()` must report the logical size
   (that is the eviction/accounting weight) instead of the envelope length;
   `load_tgt` must re-inject the flag when restoring; and such checkpoints cannot
   be cloned into a child sequence or persisted into the prompt cache, because
   they are transient views into one context's memory. Four call sites drop them
   (`server_slot` copy, `server_prompt_cache::alloc`).

4. **Server wiring** — `--checkpoint-recurrent-prev` / `--no-checkpoint-recurrent-prev`
   (server only), the gating, and the two `snapshot_prev` sites: one skips the
   redundant `4 + ubatch` boundary checkpoint when a preceding snapshot will be
   taken, the other creates the snapshot once the prompt reaches `SLOT_STATE_DONE_PROMPT`.

## 用法（Usage）

```bash
./build/bin/llama-server -m qwen3.8-27b.gguf \
  --ctx-checkpoints 32 --checkpoint-min-step 8192 \
  --checkpoint-recurrent-prev
```

激活条件（是否真正生效）见上方「激活方式汇总 · C 段」：需 recurrent/hybrid 模型、
`--ctx-checkpoints > 0`、无投机解码，且追加 >64 token 才打快照；普通 transformer 上为静默 no-op。

## 刻意未移植的部分（Deliberately *not* ported）

| item | why not |
|---|---|
| The fork's **value-based checkpoint eviction** (`0471a9885`): victim scoring by `gap_l * gap_r` for periodic checkpoints and `n_tokens * checkpoint_min_step * 8` for semantic boundaries, both scaled by `1 + 4 * replay_hits`; plus refresh-in-place. | This is a general checkpoint-cache policy, not part of this feature, and this tree's version solves the same "don't duplicate a prefix" problem differently (upstream's `supersede at the same n_tokens` block, kept here). Importing the policy would be a separate change with its own measurements. **Consequence: the fork's headline numbers do not transfer** — see "已知限制". |
| `replay_hits` | Only consumed by the value-based scoring above. Dead without it. |
| The fork's `ctx_dft_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART` guard on `update_dft` | An optimization (a plain attention draft can be trimmed after restore, so duplicating its KV into every recurrent checkpoint is waste). Upstream stashes it unconditionally; keeping upstream's call is the smaller diff. Both the target and draft `seq_rm_type` members do exist in this tree, so this is a choice, not a constraint. |
| `--pipeline-copies` / `n_pipeline_copies` | Belongs to the mirrored-input tensor-parallel work excluded in `MERGE_V100.md`. |
| The MTP draft-ubatch rework in the same fork commit | It is the substrate for the *separate* `--spec-mtp-defer-prompt` feature, still un-ported. |

## 值得记录的集成决策（Integration decisions）

1. **`is_replay_boundary` is carried in, but only the semantics this feature needs.**
   The field, the min-step eviction exemption, a preference for evicting ordinary
   checkpoints in the capacity pass (falling back to the original `front()` rule
   when every checkpoint is a boundary), and carrying the flag across a supersede
   at the same `n_tokens`. Without the exemption the snapshot would not survive:
   it sits at `n_tokens == prompt_n - 1`, which is within `checkpoint_min_step`
   of the last periodic checkpoint, so the *unmodified* upstream policy evicts it
   immediately on the very next request. The exemption is load-bearing, not
   cosmetic.

2. **Keeping upstream's eviction structure over the fork's.** The fork replaces
   the whole `create_checkpoint` body (refresh-in-place + value eviction). This
   port keeps upstream's flow and threads the two boundary-aware conditions into
   it. The 3-way merge made this explicit: upstream had also independently added a
   de-duplication block, so restoring the fork's version would have removed it.

3. **`is_replay_boundary` got a default value (`= false`)** so the pre-existing
   4-argument call site is untouched. The fork updates every call site instead;
   with defaults, the only changed call site is the new one.

4. **The `include/llama.h` field is placed next to our existing `prefill_reuse`,
   not appended at the struct end** as the fork does ("Fork extensions are
   appended to preserve the layout of all upstream fields"). This tree's
   `llama_context_params` is not ABI-stable and has a single in-tree consumer, so
   the fork's convention buys nothing here and grouping our two extensions is
   clearer. Recorded because it means the fork's line numbers will not line up for
   the next port.

5. **A pre-existing gap in the `--prefill-reuse` port got fixed as a side effect.**
   `0471a9885` also makes the checkpoint boundary spacing use
   `min(n_ubatch, prefill_reuse)` instead of plain `n_ubatch`. `--prefill-reuse` is
   meant to be used *together with* a raised `-ub` (the fork's own example is
   `-b 1536 -ub 768 --prefill-reuse 768`), so without this the flag silently
   changes prompt segmentation — and therefore the checkpoint and MTP
   trajectories — even though the CUDA math is bit-identical. That hunk is in
   `tools/server/server-context.cpp`, outside the CUDA file set of the earlier
   port, which is why it was missed. It is included here.

6. **The test file was adapted, not copied.** The fork's `be6567cea` adds
   `test_previous_recurrent_snapshot` *where upstream now has `test_rollback`* —
   upstream added `test_rollback` plus a `uint8_t fill` parameter in `9dcf84e5a`
   ("model: support Kimi-K3 recurrent-state rollback", after the fork's base).
   Both are kept: the new test is added, upstream's `test_rollback` and the
   `fill`-parameterised helpers are untouched, and `main()` calls the new test
   before the existing `for (uint8_t fill : { 0, 0x3e })` loop.

## 验证（Verification）

Performed (Windows, MSVC 14.40, Ninja, CPU-only build — the correct scope here,
since the diff contains no CUDA):

- **CPU full build — clean.** 321 targets, 0 errors, `llama-server.exe` and
  `llama-cli.exe` link. This is real evidence for this change, unlike the CUDA
  ports where a CPU build compiles none of the diff.
- **CLI registration confirmed at runtime**, not by reading the source:
  `llama-server --help` lists `--checkpoint-recurrent-prev, --no-checkpoint-recurrent-prev`
  next to the existing `--ctx-checkpoints` / `--checkpoint-min-step`.
- **`test-recurrent-state-rollback` builds, links and runs.** Against a non-recurrent
  GGUF it exits 0 with `skipping for non-recurrent model`.
- **The feature was then exercised and it works.** With a `qwen35` hybrid model
  (Qwen3.5-0.8B-Q4_K_M) the new test passes and reports
  `max logit drift=0 at token -1, top1=314/314` — the state restored from the
  preceding snapshot is *numerically identical* to the control's full-prompt run.
  That is the correctness claim the fork makes, now reproduced here.
- **Server-side mechanism confirmed in the log**, not inferred: with the flag on, a
  checkpoint is created at `n_tokens = 10801` — i.e. `prompt_n - 1` — carrying
  `replay_boundary = 1`, and the following request restores *that* one instead of the
  `prompt_n - 4` checkpoint that the flag-off run restores.
- **A/B throughput measurement** on the same model (10 turns per side, feature flag
  the only difference):

  | | tokens processed per turn | prompt ms per turn | cache_n |
  |---|---|---:|---:|
  | feature off | **21** (9x) / 22 (1x) | ~426 | 10798 |
  | feature on | **18** (9x) / 19 (1x) | ~382 | 10801 |

  The two distributions do not overlap across ten turns. See "已知限制" for
  why this is a 3-token saving rather than the fork's percentage.
- **`test_rollback` fails on this tree — pre-existing, not caused by this port.**
  The dirty-context phase reports `logits mismatch at position 6, token 0
  (6.14735 != 7.3468)`. Two independent checks pin it on upstream:
  reverting all 12 files to the pre-port commit, rebuilding and re-running produces
  **byte-identical numbers**; and none of the three forks already merged here
  (`turboquant`, `adaptive-KV-streaming`, the two V100 ports) ever touched
  `src/llama-memory-recurrent.{cpp,h}` or `src/models/delta-net-base.cpp`
  (`git diff master <pre-port> -- …` is empty), nor the test file. So it is an
  upstream dirty-recurrent-state-restore defect on `qwen35`, worth reporting
  separately. Note: the test needs a small `-c` here, because the default sizes the
  KV cache from the model's training length (~3 GiB, allocation fails).
- **Port completeness by set difference** against the fork's own diff (normalised
  added lines): 238 fork lines vs 285 ours; the only fork lines not present
  verbatim are (a) one `cur.update_tgt(... | tgt_extra_flags)` occurrence whose
  counterpart exists at a different indentation in upstream's structure, and
  (b) the `include/llama.h` field declaration, which we re-align and re-place per
  decision 4. Everything else matches.
- Line endings of all 12 files re-checked as LF after a `python` rewrite of the
  test file silently converted it to CRLF (the same trap as `gated_delta_net.cu`
  in the previous port).

**Still not verified:**

- The `--prefill-reuse` + `--checkpoint-recurrent-prev` combination, including the
  boundary-spacing fix from decision 5.
- Interaction with the adaptive-KV-streaming fork end to end; that fork rewrites
  `llama-kv-cache.cpp`, while this change touches the *recurrent* memory and the
  server checkpoint list, so they are disjoint by construction but not by
  measurement.
- Any GPU behaviour. All of the above is CPU.

## 已知限制（Known limitations）

- **The fork's percentages do not reproduce, and the reason is measured.** The saving
  is an absolute **3 tokens of replay per turn**, not a proportion: the flag-off
  server restores upstream's checkpoint at `prompt_n - 4` and the flag-on server
  restores the snapshot at `prompt_n - 1`. Upstream already places boundary
  checkpoints at `4 + n_ubatch` and `4` before the end of a prompt
  (`checkpoint_offsets`, upstream `a7b3dee7a` / PR #20288) — and the fork's own base
  `465e49b9c` carries that same array — so this port effectively *replaces* the `-4`
  checkpoint with a `-1` snapshot.

  Consequence: the gain scales inversely with suffix length. Measured
  `-14%` of tokens / `-10%` wall at a ~17-token suffix; that becomes ~`-2.3%` at a
  128-token suffix. **The fork's `+14.08%` at `+128` is not reproduced and should not
  be used for planning.** The plausible explanation for the gap is a regime this test
  could not synthesize: one where the nearest *surviving* checkpoint is far from the
  prompt end (long context, checkpoint list filled, eviction active). Neither the
  4-turn nor the 10-turn run filled a 4-slot list.
- Gains only exist in a specific workload shape: a large cached context receiving
  a short append. The fork measures 64 tokens as neutral and the benefit decays as the
  suffix grows (`+3.33%` at 1000) — which is consistent with a fixed-size saving rather
  than a proportional one, and is worth keeping in mind when reading the fork's table.
- Requires a hybrid/recurrent model **and** `--ctx-checkpoints > 0` **and** no
  speculative decoding. On a plain attention model the flag is a no-op.
- `ON_DEVICE` checkpoints are not portable: they are dropped on slot
  copy and on prompt-cache save. That is intended (they alias one context's
  storage), but it means a restored prompt-cache entry falls back to the host
  checkpoints and therefore to the replay behaviour this feature was meant to
  avoid.
