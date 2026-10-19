# Prism (Bonsai) 私有量化格式移植到本树 — 阶段一

> 源：`PrismML-Eng/llama.cpp` 分支 `prism`（工作副本 `C:/Users/confu/llama/prism-llama`），
> 基线 `5ea87ddad`（2026-08-25 上游 #27538，本树 `master` 已包含）。
> 目标树：本地 `master`（buun-port lineage，HEAD `e17ba5745`）。移植在分支 `prism-quant` 上进行。

## 1 目标与验收

目标：让本树编译出的 `llama-server` / `llama-cli` 能直接加载 Prism 用 `PQ2_0` / `PTQ1_0` 量化的模型。

| 验收项 | 状态 |
|---|---|
| 识别并加载 GGUF 里的 type 142/143、ftype 141/142/143 | ✅ |
| CPU 上完整推理（编码/解码/vec_dot/get_rows/mul_mat） | ✅ |
| CUDA 反量化 + get_rows（matmul 走 dequant→cuBLAS 兜底） | ✅ 编译通过，未上卡验证 |
| `llama-quantize` 能产出这两种格式 | ✅ |
| 与 prism 自身的数值行为一致 | ✅ 逐字节 + 逐位 PPL |
| 直接跑 prism 发布的 Bonsai 2 权重 | ❌ **被 Hadamard 折叠挡住**，见 §5 |

## 2 移植清单（阶段一实际改动）

**格式核心**

| 文件 | 内容 |
|---|---|
| `ggml/include/ggml.h` | `GGML_TYPE_PQ2_0 = 142`、`GGML_TYPE_PTQ1_0 = 143`、`GGML_TYPE_COUNT = 144`；`GGML_FTYPE_MOSTLY_PQ2_0 = 128`、`..._PTQ1_0 = 129` |
| `ggml/src/ggml-common.h` | `block_pq2_0`（2 B scale + 32 B qs / 128 权重，2.125 bpw）、`block_ptq1_0`（24 B qs + 2 B qh + 2 B scale，1.75 bpw）、`QI_*/QR_*` |
| `ggml/src/ggml-quants.c/.h` | 参考编解码、`quantize_pq2_0` / `quantize_ptq1_0`、`ggml_validate_row_data` 两个 case |
| `ggml/src/ggml.c` | `type_traits` 两项、`ggml_ftype_to_ggml_type`、`ggml_quantize_chunk` |
| `ggml/src/gguf.cpp` | 旧版 "group-128 存成 type 42" 的布局检测与明确报错 |

**CPU**

| 文件 | 内容 |
|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c` | traits（PQ2_0→Q8_K，PTQ1_0→Q8_0）、`ggml_cpu_vec_dot[_type]()`（PQ2_0 行宽非 256 倍数时退回 Q8_0，否则 Q8_K 会越界）、4 个调用点改走 src0 |
| `ggml/src/ggml-cpu/quants.c` | `ggml_vec_dot_{pq2_0_q8_0,pq2_0_q8_K,ptq1_0_q8_0}_generic` |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | SSE2/SSSE3/AVX2/AVX-VNNI/AVX-512-VNNI 实现（含 PQ2_0×Q8_K、PTQ1_0×Q8_0） |
| `ggml/src/ggml-cpu/arch/arm/quants.c` | NEON / dotprod / i8mm 实现 |
| `ggml/src/ggml-cpu/arch-fallback.h` | 通用/ppc/loongarch/riscv/s390x/wasm 分支的 generic 别名（未加 repack 相关符号） |
| `ggml/src/ggml-cpu/ops.cpp` | add/add1/acc/out_prod/set/get_rows/clamp 七处 case 标签 |

**CUDA**

| 文件 | 内容 |
|---|---|
| `ggml/src/ggml-cuda/dequantize.cuh` | `dequantize_pq2_0` / `dequantize_ptq1_0` |
| `ggml/src/ggml-cuda/common.cuh` | `ggml_cuda_type_traits<PQ2_0/PTQ1_0>`、`ptq1_0_trit()` |
| `ggml/src/ggml-cuda/convert.cu` | PTQ1_0 专用反量化核（`__byte_perm`/`__vsub4` + shared 转置，非 HIP）与 6 张 to_fp16/fp32/bf16 派发表 |
| `ggml/src/ggml-cuda/getrows.cu` | `get_rows_cuda_q<QK, QR, dequant_fn>` 两个 case |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | MUL_MAT / GET_ROWS 的 `supports_op` 白名单；`mul_mat_id` 显式绕开 mmvq（原生核未移植前走 dequant+cuBLAS） |

**llama 侧**

| 文件 | 内容 |
|---|---|
| `include/llama.h` | `LLAMA_FTYPE_MOSTLY_PQ2_0 = 141`、`_LEGACY = 142`、`PTQ1_0 = 143`（与线上 gguf 对齐） |
| `src/llama-quant.cpp` | ftype→type、不可整除时的回退表 |
| `src/llama-model-loader.cpp` | ftype 名称、type→ftype、**`prism.hadamard` 硬失败守卫** |
| `tools/quantize/quantize.cpp` | CLI 选项 `PQ2_0` / `PTQ1_0` |

未移植（明确排除）：mmq/mmvq 原生核（`mmq-*`、`mmvq-pq2_0.cuh`、`mmvq-ptq1_0.cuh`、`vecdotq.cuh`）、
CPU repack（`repack.cpp` 743 行）、Metal/Vulkan/SYCL/HIP/WebGPU 后端、`gguf-py` 常量与 HF 转换脚本。
⇒ 这些后端上这两种类型不可用（`supports_op` 会拒绝，回落到其它后端/CPU）。

## 3 验证证据

单变量对照，全部在同一台机器（无 GPU）上跑 CPU 构建。

**(1) 量化输出逐字节一致** — 同一源模型（`qwen2.5-0.5b-q4km.gguf`）、同一命令、两侧二进制：

```
357b0f27ae698dcc  m_pq2.gguf    (本树 llama-quantize)
357b0f27ae698dcc  p_pq2.gguf    (prism llama-quantize)
16d70caa47be3c83  m_ptq1.gguf
16d70caa47be3c83  p_ptq1.gguf
```

**(2) 推理数值逐位一致** — 同一 gguf、同一 `llama-perplexity` 参数（`-c 128 --chunks 2 -ngl 0`）：

| 模型 | 本树 PPL | prism PPL |
|---|---|---|
| `q2.5-pq2_0.gguf` | 6392297.5327 | 6392297.5327 |
| `q2.5-ptq1_0.gguf` | 8301865.3832 | 8301865.3832 |

（PPL 巨大是因为把 4-bit 模型硬再量化成 2-bit/三值 —— 两者一致才是判据；同源 Q8_0 为 20.70、原模型 20.81。）

**(3) 编解码往返自检**（独立小程序，`ggml_quantize_chunk` → `type_traits::to_float`）：PQ2_0/PTQ1_0 在
n=1024/896 上的重建误差与上游 Q2_0、TQ1_0 同量级（0.5·max_d），行宽 896（非 256 倍数）走 Q8_0 分支不越界。

**(4) 二进制可用**：`llama-cli` / `llama-server` 加载两种格式并完成 decode；`llama-server`
端点返回的是模型输出被 server 侧格式解析拒收（质量崩塌所致），不是加载/计算失败。

## 4 关键调用点（改动后）

- `ggml_row_size(PQ2_0, n) = 34 * n/128`；`PTQ1_0 = 28 * n/128`
- CPU mul_mat 的 vec_dot 类型由 `ggml_cpu_vec_dot_type(src0)` 决定，**不是**静态 trait
- CUDA 下这两个类型**只**走 `ggml_cuda_mul_mat_cublas`（dequantize→F16→cuBLAS），mmq/mmvq 均返回 false

## 5 阻断项：官方 Bonsai 2 权重是 Hadamard 折叠的

下载 `prism-ml/Ternary-Bonsai-2-2B-gguf` 的 `PTQ1_0`（409 MiB）读元数据：

```
general.architecture              = qwen35
general.file_type                 = 143
prism.hadamard.version            = 2
prism.hadamard.tied_output        = True
prism.hadamard.block_size         = 512
prism.hadamard.transform          = normalized-sylvester-walsh-hadamard
prism.hadamard.sign_mode          = explicit
prism.hadamard.weight_names       = [blk.*.attn_qkv/attn_gate/ssm_out/ffn_{down,gate,up}.weight, ...]
prism.hadamard.inverse_weight_names = [token_embd.weight]
prism.hadamard.sign_widths        = [2048, 6144]
```

即：绝大多数权重在**转换时**已被旋转进 Hadamard 基，运行时必须把激活按 512 一块做 FWHT 并乘符号向量；
`token_embd` 还要当输出头用（tied output，`output.weight` 不存在）。少了这层运行时，模型能加载但**算错**。

因此阶段一在 loader 里加了硬失败守卫（不允许静默算错）：

```
E llama_model_loader: ... declares 'prism.hadamard.weight_names': this model uses Prism's
  Hadamard weight folding, which is not ported into this build. Only the PQ2_0/PTQ1_0 weight
  formats are supported; load a model without prism.hadamard metadata (or port the folding runtime first)
```

**判定**：格式层是必要条件、不是充分条件。要真正跑通官方 Bonsai 2 权重，必须做阶段二。

## 6 阶段二（Hadamard 折叠）可行性评估

本树已具备底座，明显便宜于在原生上游上做：

- `GGML_HINT_SRC0_IS_HADAMARD`（`ggml.h:471`）与 `ggml_mul_mat_set_hint`（`src/llama-impl.h:71`）已在；
- 各后端 unsigned FWHT 已实现：CUDA `fwht.cu` 支持 **n=512**（正是 Bonsai 的 block_size）、CPU
  `ggml_compute_forward_fwht_f32`、Metal、SYCL、Vulkan、BLAS；
- 缺的是：`prism.hadamard.*` 解析与旋转张量、图内注入、图级校验、tied output v2，以及**符号向量**。

两种实现路线：

1. **图组合**（推荐先做）：`x' = FWHT_blocks(x ⊙ s)` —— 用 `ggml_mul` 施加符号、`ggml_reshape_4d` 成
   `[512, n/512, ...]`、`GGML_HINT_SRC0_IS_HADAMARD` 触发现有 unsigned FWHT。**零内核改动**，代价是每个
   折叠权重多一个 elementwise MUL。需要确认各后端 FWHT 的 n=512 路径都可用（CUDA 已确认）。
2. **内核内符号**（prism 原样）：FWHT 核内乘 `signs`，省掉那次 MUL，但要改 5 个后端。

合入顺序建议：`prism.hadamard` 解析 + 图组合 + tied output v2 → 先用 CPU 与 prism 对 PPL（逐位一致）→
再放开 CUDA。

## 7 未做 / 已知缺口

- CUDA 未上卡跑（开发机无 GPU）；`-ngl` 全部 offload 后的数值与性能待 V100 验证。
- 原生 mmq/mmvq 未移植 ⇒ GPU 上 2-bit 权重每层都要反量化成 F16，显存与带宽代价约 8×，性能不可用于生产。
- `gguf-py` 未加这两种类型 ⇒ `convert_hf_to_gguf.py` 无法直接产出，只能 `llama-quantize`。
- `-ctk/-ctv` 若选了这两种类型：KV cache 路径不支持（它们只用于权重）。
