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
| 自适应 KV 流式 | `--cache-reuse 256 --kv-stream-stage-mib 1024 -np 1` | CUDA arena 放 VRAM，其余落主机内存；**不再限定架构**，改由上下文创建时探测（见下文 "Eligibility"）。可与 turbo KV（`-ctk turbo4 -ctv turbo3`）和 MTP 推测解码（`--spec-type draft-mtp`）叠加，见 [`MERGE_TURBOQUANT.md`](MERGE_TURBOQUANT.md) 与 [`FIX_KVSTREAM_TURBO_MTP.md`](FIX_KVSTREAM_TURBO_MTP.md) |

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
| every attention (KV-carrying, non-recurrent) layer on **one** CUDA device | `requires every attention layer on one CUDA device` |
| same padded K and V head dim, with a compiled streamed instance (64 / 128 / 256) | `no streamed attention kernel for layer N (head_dim_k = …, head_dim_v = …)` |
| one page geometry for all layers (same head dim / head count / K,V types) | `requires uniform attention-layer geometry` |
| a streamed attention mode for the `(K,V)` type pair | `invalid block KV streaming page geometry` |
| no sliding-window cache (`hparams.is_swa_any()` is false) | `does not support sliding-window attention caches yet` |
| `-np 1`, Flash Attention, `--no-kv-offload` absent, target context only | as before |

Memory types that carry the runtime: `llama_memory_hybrid` (attention child) and, since this
change, the plain `llama_kv_cache` of a **dense, non-SWA** model. `llama_kv_cache_iswa`,
`llama_memory_hybrid_iswa` and `llama_memory_hybrid_idx` still take no streaming parameters:
iswa keeps its layers in two caches (`get_base()` for the dense ones, `get_swa()` for the
sliding-window ones) whose KV lengths differ, while the runtime has exactly one page geometry
and one resident layout. Streaming only the dense half would be the way in — the SWA half is
already bounded by `n_swa + n_ubatch` and stays resident — but that needs the arena ownership
(the third `llama_kv_cache` owner) and a probe that counts dense layers only.

Not eligible, by construction: MLA caches (K/V head dims differ), attention sinks (`gpt-oss`),
head dims without a compiled instance (e.g. 96 — unless TurboQuant padding lifts them to 128),
`-np > 1`, and MTP/draft contexts.

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
