# Adaptive KV cache streaming — integration notes

Merge of the [`RaymondHuang210129/llama.cpp-adaptive-kv-streaming`](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) downstream fork into upstream `llama.cpp`.

| source | branch in this repo | upstream base | what was taken |
|---|---|---|---|
| [`RaymondHuang210129/llama.cpp-adaptive-kv-streaming`](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) @ `d7680c67b` | `akv-feature` | `465e49b9c` (145 commits behind) | the whole branch (KV streaming) |

This fork was merged together with [`AtomicBot-ai/atomic-llama-cpp-turboquant`](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) onto branch **`turbo-kvstream-merge`** (113 files, `+23040 / -318` against `master`). This note covers only the KV-streaming part; see [`MERGE_TURBOQUANT.md`](MERGE_TURBOQUANT.md) for the TurboQuant part.

```
4c9233c03  master (upstream)
  └─ e27c32739  Merge branch 'akv-feature'          (KV streaming)
       └─ 94afa1bbe  Merge branch 'turbo-only'      (turboquant subset)
            └─ a69c11f33  cleanup, type numbering, build fixes
                 └─ 70009cf81  CUDA-side merge-damage fixes
```

## Usage

```bash
# adaptive KV streaming (CUDA arena in VRAM, remainder in host memory)
llama-server -m model.gguf --cache-reuse 256 --kv-stream-stage-mib 1024 -np 1
```

## 激活方式（编译 / 参数）

**编译前置**：必须 `GGML_CUDA=ON` 才能用到 CUDA 上的 KV-streaming 内核与 arena；架构需覆盖目标卡（如 sm_80 / sm_70）。

**运行时参数（CLI）**：

| 功能 | 参数 | 说明 |
|---|---|---|
| 自适应 KV 流式 | `--cache-reuse 256 --kv-stream-stage-mib 1024 -np 1` | CUDA arena 放 VRAM，其余落主机内存；**不再限定架构**，改由上下文创建时探测（见下文 "Eligibility"）；SWA 模型（gemma2/3 等）只流式 **dense 半边**，SWA 半边常驻。可与 turbo KV（`-ctk turbo4 -ctv turbo3`）和 MTP 推测解码（`--spec-type draft-mtp`）叠加，见 [`MERGE_TURBOQUANT.md`](MERGE_TURBOQUANT.md) 与 [`FIX_KVSTREAM_TURBO_MTP.md`](FIX_KVSTREAM_TURBO_MTP.md) |

**环境变量**：无 KV-streaming 专属变量（注意力旋转 / `TURBO_AUTO_ASYMMETRIC` 等见 [`MERGE_TURBOQUANT.md`](MERGE_TURBOQUANT.md)）。

## Integration decisions worth knowing

- **`peer_access` map restored** in `ggml_cuda_device_info` (`common.cuh`);
  upstream dropped it with the multi-GPU split infrastructure but the KV-streaming
  cross-device copy path uses it.

## Verification performed

Build: MSVC 14.40 + Ninja + CMake 3.28, static libs.

- `test-kv-stream-plan` / `-config` / `-softmax` / `-bench-config`:
  54 tests, 325 assertions, 0 failures.
- `--cache-reuse 256 --kv-stream-stage-mib 1024 -np 1` are parsed and reach
  context creation. See the limitation below.
- **CUDA compile** (`CUDA 12.4`, `-allow-unsupported-compiler`, arch `sm_80`):
  the KV-streaming CUDA sources compile with 0 errors (the full 367-target
  `ggml-cuda` link was not run to completion; untouched template instances
  depend only on the headers compiled above).

## Eligibility (was: Qwen3.5 only)

The upstream fork gated block KV streaming on `model.arch == LLM_ARCH_QWEN35` in
`llama-kv-stream-config.cpp`. That was a proxy for the geometry the fork had validated, not a
capability: it even rejected `LLM_ARCH_QWEN35MOE`. The gate is now a probe, run once in
`llama_context` when `--kv-stream-stage-mib` is set:

| condition | reported as |
|---|---|
| every streamed attention (KV-carrying, non-recurrent, **non-SWA**) layer on **one** CUDA device | `requires every attention layer on one CUDA device` |
| same padded K and V head dim, with a streamed instance for that `(K,V)` pair at that dim (64 / 128 / 256 / 512 for every streamed pair) | `no streamed attention kernel for layer N (head_dim_k = …, head_dim_v = …)` |
| one page geometry for all streamed layers (same head dim / head count / K,V types) | `requires uniform attention-layer geometry` |
| a streamed attention mode for the `(K,V)` type pair | `invalid block KV streaming page geometry` |
| at least one full-attention layer to stream | `found no full-attention layers to stream` |
| no sliding-window cache, or a sliding-window cache with a streamable dense half | `does not support hybrid models with sliding-window layers yet` / `does not combine with --swa-full` |
| `-np 1`, Flash Attention, `--no-kv-offload` absent, target context only | as before |

Head dim 512 (gemma-4's dense half) is a partial-only instance for **every** streamed pair: block
streaming always runs the chunked partial kernel, never the regular vec case, and at D = 512 that
regular case does not fit the register budget. Two consequences worth knowing:

- The pair set at 512 is the same as at 64/128/256, so a turbo/q8 KV cache can stream a 512-dim
  model instead of being forced onto f16.
- Turbo V needs the columns-per-iteration setting halved at this dim: the per-warp transpose buffer
  is `nwarps * V_cols_per_iter * D` floats, i.e. 64 KiB at eight columns, over the 48 KiB a kernel
  may allocate statically. `D >= 512` keeps four columns (32 KiB, no spill); 128/256 codegen is
  unchanged.

Memory types that carry the runtime: `llama_memory_hybrid` (attention child), the plain
`llama_kv_cache` of a **dense, non-SWA** model, and `llama_kv_cache_iswa` (**dense half only**).
The last one is what makes gemma2/gemma3-class models work: their layers are split into a dense
cache of `n_ctx` cells and a sliding-window cache of `PAD(min(n_ctx, n_swa + n_ubatch), 256)`
cells, so the SWA half is O(1) in the context length and stays resident while `get_base()`
streams. The probe therefore walks the **non-SWA** layers only; counting the SWA half would size
the resident region for pages the runtime never holds (and on gemma4 it would mix two different
SWA head dims into a uniformity check that applies to neither cache).

`llama_memory_hybrid_iswa` and `llama_memory_hybrid_idx` still take no streaming parameters at
all, and `--swa-full` is rejected: it sizes the SWA half like the dense one, which is the half the
streaming is supposed to move out of VRAM.

Not eligible, by construction: MLA caches (K/V head dims differ), attention sinks (`gpt-oss`),
head dims without a compiled instance (e.g. 96 — unless TurboQuant padding lifts them to 128),
`-np > 1`, MTP/draft contexts, and `llama_memory_hybrid_iswa` models (hybrid **and** SWA).

### What base-only streaming buys on a sliding-window model

The streamed half is the only one that grows, so the crossover is
`n_ctx > (n_layer - n_dense)/n_dense * (n_swa + n_ubatch)`: ~4k tokens for gemma2 (every 2nd
layer dense, `n_swa` 4096), ~5k for gemma3 and the 4B `spark2_5` in this tree (every 6th/4th
layer dense, `n_swa` 1024/512). Below that the SWA half dominates and streaming saves nothing.
Above it, every `n_ctx`-sized page lives in host memory and the decode cost is the resident
window's complement: each step reads `(active - resident) * page_bytes * n_dense` over PCIe, so
1M context on a model with 9 dense layers is ~9 GiB per step, i.e. ~1 s/token on PCIe 3.0 x16.
That is the same curve Qwen3.5 has, scaled by the number of streamed layers.

## Known limitations

- **Turbo KV + KV streaming and MTP + KV streaming were broken until
  [`FIX_KVSTREAM_TURBO_MTP.md`](FIX_KVSTREAM_TURBO_MTP.md)**: turbo types were
  missing from the CUDA KV-stream capability table (`invalid block KV streaming
  page geometry` at startup), and the phase arena reserved decode compute for
  one token per sequence, which a speculative verify batch cannot satisfy
  (`phase arena currently supports TG1 without speculative batches`). Both are
  fixed; the GPU-side numerics of the turbo conversion fallback still need a
  run on real hardware. See [`MERGE_TURBOQUANT.md`](MERGE_TURBOQUANT.md) for the
  TurboQuant side.
- **CUDA GPU validation is still outstanding.** No end-to-end run has happened on
  a real GPU (this machine has none); the KV-streaming path is compile-verified
  only.
