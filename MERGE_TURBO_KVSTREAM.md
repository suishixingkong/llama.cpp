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

## 激活方式（编译 / 环境变量 / 参数）

**编译前置**：必须 `GGML_CUDA=ON` 才能用到 CUDA 上的 turbo 内核，默认编译包含turbo常用组合；CPU 后端也自带 turbo 参考实现（可被 `test-backend-ops` 覆盖）。架构需覆盖目标卡（如 sm_80 / sm_70）。

**运行时参数（CLI）**：

| 功能 | 参数 | 说明 |
|---|---|---|
| Turbo KV 缓存 | `-ctk turbo4 -ctv turbo4 -fa on` | 需开启 FlashAttention；`llama_context` 会自动启用 FA。KV 类型支持 `turbo2`/`turbo3`/`turbo4` |
| 权重量化 | `--outtype tq3_1s` / `tq4_1s` | 转换模型权重时选用 turbo 权重量化 |
| 自适应 KV 流式 | `--cache-reuse 256 --kv-stream-stage-mib 1024 -np 1` | CUDA arena 放 VRAM，其余落主机内存；**仅 Qwen3.5 可用**（其它架构设了 `--kv-stream-stage-mib` 会建上下文失败，这是上游 fork 自身限制）。可与 turbo KV（`-ctk turbo4 -ctv turbo3`）和 MTP 推测解码（`--spec-type draft-mtp`）叠加，见 [`FIX_KVSTREAM_TURBO_MTP.md`](FIX_KVSTREAM_TURBO_MTP.md) |

**环境变量**：

| 变量 | 作用 | 默认 |
|---|---|---|
| `LLAMA_ATTN_ROT_K_OVERRIDE=1` | 按侧开启注意力旋转（fork 策略默认已改为 **OFF**） | OFF（本树默认关） |
| `LLAMA_ATTN_ROT_V_OVERRIDE=1` | 同上，V 侧 | OFF |
| `LLAMA_ATTN_ROT_DISABLE=1` | 硬性锁定关闭旋转 | — |
| `TURBO_AUTO_ASYMMETRIC=0` | 强制对称 turbo K/V（默认 ON：高 GQA 比时把 K 缓存从 turbo* 升级到 q8_0 保护质量） | ON |

⚠ 注意力旋转默认值被本树改成了 OFF（与纯上游不同），这也会影响非 turbo 的量化 KV（如 `q8_0`）。如需恢复上游行为，把 `llama_kv_cache::llama_kv_cache()` 里 `attn_rot_k = !attn_rot_disable && ...` 改回无条件开启。

⚠ Turbo KV 缓存类型**仅运行时使用、不写入 GGUF**；但用 fork 的 `TQ3_1S`/`TQ4_1S` 权重量化出的 GGUF 与本构建**不兼容**（类型编号被重新排过）。Metal/Vulkan 后端无 turbo 内核，会拒绝 turbo 缓存类型。

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
- Traces the exclusions had left in *shared* files were cleaned up in a
  follow-up: a duplicate `MODEL_TENSOR.SSM_G` in `gguf-py/gguf/constants.py`
  (raised at import time, so `import gguf` and every gguf-py tool were broken),
  the `inkling` entries in `MODEL_ARCH` / `MODEL_TENSORS` / `MODEL_TENSOR` /
  `VisionProjectorType`, the missing `TURBO{2,3,4}_0` rows in
  `GGML_QUANT_SIZES`, and the now-callerless `set_input_pos_rel_flat()` in
  `llama-kv-cache.{h,cpp}`.

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
  were added to all five `ggml_cuda_fattn_mma_get_config_*()`. The matching
  explicit instantiations of `(640, 512, {1,2}, 16)` were missing from
  `template-instances/` — `fattn.cu` called them and it still linked only because
  the kernel bodies live in `fattn-mma-f16.cuh` — so a follow-up added them,
  together with the `generate_cu_files.py` rule that regenerates them (the
  generator could not produce 640 before, so a regeneration would silently drop
  them again).
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
- **`test-backend-ops` turbo cases on the CPU backend** (`-b CPU`, i.e. the
  reference implementations): `TURBO_WHT` **27/27** — including
  `test_turbo_wht_roundtrip`, which bounds the error of the full
  f32 → WHT → PolarQuant → turbo3/turbo4 → f32 chain — `SET_ROWS` turbo3
  **21/21**, `FLASH_ATTN_EXT` `turbo3` **8/8**. The cases themselves come from
  the fork and were ported in a follow-up to this merge; see the correction
  under *Known limitations*.
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
- **Turbo KV + KV streaming and MTP + KV streaming were broken until
  [`FIX_KVSTREAM_TURBO_MTP.md`](FIX_KVSTREAM_TURBO_MTP.md)**: turbo types were
  missing from the CUDA KV-stream capability table (`invalid block KV streaming
  page geometry` at startup), and the phase arena reserved decode compute for
  one token per sequence, which a speculative verify batch cannot satisfy
  (`phase arena currently supports TG1 without speculative batches`). Both are
  fixed; the GPU-side numerics of the turbo conversion fallback still need a
  run on real hardware.
- **CUDA GPU validation is still outstanding.** The turbo kernels *are* covered
  by `test-backend-ops` on the CPU backend (see *Verification performed*); the
  harness does not skip it, it runs the reference implementations. An earlier
  revision of this note claimed a CPU-only device cannot validate the turbo
  kernels — that was wrong, and the actual gap was simply that no test cases
  existed in this tree. What genuinely remains unverified is the CUDA path:
  the full 367-target `ggml-cuda` link was never run, and no run has happened on
  a real GPU (this machine has none), so an end-to-end MLA model (576 → 640) with
  `-ctk turbo4` is still unmeasured.
