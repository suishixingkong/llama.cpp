# TurboQuant KV cache + adaptive KV streaming — integration notes

Merge of two downstream forks into upstream `llama.cpp`:

| source | branch in this repo | upstream base | what was taken |
|---|---|---|---|
| [`AtomicBot-ai/atomic-llama-cpp-turboquant`](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) @ `044dff242` | `tq-master` | `1c3c9674d` (706 commits behind) | TurboQuant KV cache + TQ weights only |
| [`RaymondHuang210129/llama.cpp-adaptive-kv-streaming`](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) @ `d7680c67b` | `akv-feature` | `465e49b9c` (145 commits behind) | the whole branch (KV streaming) |

Result: branch **`turbo-kvstream-merge`**, 113 files, `+23040 / -318` against `master`.

```
4c9233c03  master (upstream)
  └─ e27c32739  Merge branch 'akv-feature'          (KV streaming)
       └─ 94afa1bbe  Merge branch 'turbo-only'      (turboquant subset)
            └─ a69c11f33  cleanup, type numbering, build fixes
                 └─ 70009cf81  CUDA-side merge-damage fixes
```

## Usage

```bash
# turbo KV cache (requires flash attention; llama_context enables it automatically)
llama-server -m model.gguf -ctk turbo4 -ctv turbo4 -fa on

# adaptive KV streaming (CUDA arena in VRAM, remainder in host memory)
llama-server -m model.gguf --cache-reuse 256 --kv-stream-stage-mib 1024 -np 1
```

`-ctk/-ctv` now accept `turbo2`, `turbo3`, `turbo4` in addition to the upstream
types. Weight quantisation gained `--outtype tq3_1s` / `tq4_1s`.

## What was deliberately *not* ported

The turboquant fork is a full distribution (308 changed files). Excluded:

- **MTP / NextN** — the fork's `master` tree contains no implementation at all
  (`mtp_assistant`, `decode_mtp`, `nextn_draft` appear in no source file); only
  `MTP.md`, `NEXTN.md` and `scripts/run-gemma4-*-mtp-*` remain, the code lives on
  the unmerged `feature/gemma-mtp` / `b1-mtp-qwen-rebase` branches.
- kimi-k3 / inkling / bailingmoe3 architectures, UDT quant recipes
- `GGML_OP_FLASH_ATTN_EXT_BANDED` (banded flash attention for the Inkling arch)
  and its CPU/CUDA hooks
- CI workflows, ROCm/NVFP4 build machinery, release scripts, docs, benchmarks
- **Metal and Vulkan turbo kernels** — chose CPU + CUDA. Those backends reject
  turbo cache types at `supports_op`, and `ggml/src/ggml-metal`,
  `ggml/src/ggml-vulkan` are untouched.

## Type numbering

Upstream ids are preserved so upstream GGUFs stay readable; turbo types are
appended:

```
GGML_TYPE_Q2_0    = 42   (upstream id, unchanged)
GGML_TYPE_TURBO2_0= 43   GGML_TYPE_TURBO3_0 = 44   GGML_TYPE_TURBO4_0 = 45
GGML_TYPE_TQ3_1S  = 46   GGML_TYPE_TQ4_1S   = 47   GGML_TYPE_COUNT    = 48
GGML_OP_TURBO_WHT inserted in the op enum -> GGML_OP_COUNT = 102,
RPC_PROTO_PATCH_VERSION = 1
```

⚠ Consequence: GGUFs quantised with the fork's `TQ3_1S`/`TQ4_1S` weights are *not*
compatible with this build (the fork had renumbered `Q2_0` to 47 and used 42-46).
Turbo KV cache types are runtime-only and never stored in a GGUF.

## Integration decisions worth knowing

- **FA vector-kernel dispatch.** Upstream replaced the `#ifdef
  GGML_CUDA_FA_ALL_QUANTS` macro dispatch with a table in
  `ggml_cuda_get_fattn_vec_case()` driven by `if constexpr
  (GGML_CUDA_FA_<K>_<V>)`. The 21 turbo combinations are registered there and
  always compiled via `ggml_cuda_fattn_vec_instances()` in
  `ggml/cmake/common.cmake`; they are intentionally outside
  `GGML_CUDA_FA_QUANTS`.
- **D=640 MMA config.** Turbo KV zero-pads K/V head_dim to a multiple of 128
  (`llama-kv-cache.cpp`), so MLA models (576 → 640) need MMA config cases, which
  were added to all five `ggml_cuda_fattn_mma_get_config_*()`.
- **Attention rotation default changed to OFF.** This is the fork's policy
  (empirically model- and quant-specific). Enable per side with
  `LLAMA_ATTN_ROT_K_OVERRIDE=1` / `LLAMA_ATTN_ROT_V_OVERRIDE=1`;
  `LLAMA_ATTN_ROT_DISABLE=1` remains a hard lock-out. DeepSeek / GLM-DSA /
  DOTS3NOTE indexer models still force rotation on.
  ⚠ This also affects non-turbo quantised KV (e.g. `q8_0`) compared to plain
  upstream. Revert by restoring upstream's unconditional
  `attn_rot_k = !attn_rot_disable && ...` in `llama_kv_cache::llama_kv_cache()`.
- **`TURBO_AUTO_ASYMMETRIC`** (fork, default on): for a high GQA ratio the K cache
  is upgraded from `turbo*` to `q8_0` to protect quality. Set
  `TURBO_AUTO_ASYMMETRIC=0` to force symmetric turbo K/V.
- **`peer_access` map restored** in `ggml_cuda_device_info` (`common.cuh`);
  upstream dropped it with the multi-GPU split infrastructure but the KV-streaming
  cross-device copy path uses it.
- The fork's `GGML_PREC_F32_PEDANTIC` plumbing was dropped: neither that enum
  value nor the `ggml_prec()` accessor exists upstream, and it was only ever set
  by the excluded `build_lora_mm(..., prec)` parameter.

## Verification performed

Build: MSVC 14.40 + Ninja + CMake 3.28, static libs.

- **CPU: `314/314` targets, 0 errors.**
- `test-turbo-quant` (turbo3/turbo4 round trip), `test-quantize-fns`
  (incl. `tq3_1s`, `tq4_1s`), `test-arg-parser`, `test-chat`: pass.
- `test-kv-stream-plan` / `-config` / `-softmax` / `-bench-config`:
  54 tests, 325 assertions, 0 failures.
- End-to-end generation, `Qwen2.5-0.5B-Instruct Q4_K_M`, `-ctk turbo4 -ctv turbo4
  -fa on` — both `llama-completion` and `llama-server`
  (`/v1/chat/completions`) answer `"The capital of France is Paris."`, matching
  the f16-KV baseline.
- `--cache-reuse 256 --kv-stream-stage-mib 1024 -np 1` are parsed and reach
  context creation. See the limitation below.
- **CUDA compile** (`CUDA 12.4`, `-allow-unsupported-compiler`, arch `sm_80`):
  `ggml-cuda.cu`, `fattn.cu`, `convert.cu`, `getrows.cu`, `set-rows.cu`,
  `mmvq-tq.cu`, `turbo-wht.cu`, `turbo-innerq.cu` and all **21** turbo
  flash-attention instances compile with 0 errors. The full 367-target CUDA
  link was not run to completion (many hours); untouched template instances
  depend only on the headers compiled above.

## Known limitations

- **KV streaming is restricted to Qwen3.5** by
  `llama-kv-stream-config.cpp` (`block KV streaming currently supports only
  Qwen3.5`). Other architectures fail context creation when
  `--kv-stream-stage-mib` is set. This is the upstream fork's own restriction.
- Turbo KV cache is unusable on Metal/Vulkan in this tree (no kernels ported).
- `test-backend-ops` cannot validate the turbo kernels with a CPU-only device —
  the harness skips the CPU backend, so CUDA GPU validation is still outstanding.
