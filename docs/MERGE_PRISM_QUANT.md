# Prism (Bonsai) 私有量化格式移植到本树

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
| 直接跑 prism 发布的 Bonsai 2 权重 | ✅ 阶段二（Hadamard 折叠）已落地，见 §5 |
| CUDA 原生 mmq/mmvq | ✅ 已移植并编译通过，未上卡验证，见 §6 |
| `gguf-py` 常量 / HF 转换脚本 | ✅ 见 §7 |
| Vulkan / SYCL 后端 | ❌ 未移植，见 §8 |

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

## 5 阶段二：Hadamard 折叠（已落地）

prism 发布的 Bonsai 权重不是"直接量化"的：转换期把 150 个矩阵旋转进 Hadamard 基
（`W' = W H S`，H 为 block 512 的归一化 Sylvester Hadamard，S 为每输入通道 ±1 符号），
运行时必须把激活送进同一个基。少这一层 = 模型能加载但算错，所以阶段一曾用 loader 硬失败拦住。

**实现（按本树既有设施走"图组合"，零新内核）**

| 位置 | 内容 |
|---|---|
| `llama-model.cpp`（`load_hparams`） | 解析并校验 `prism.hadamard.*`：版本 1/2、`tied_output` 与版本耦合、block 必须是 2 的幂、`transform`/`axis` 白名单、sign 宽度必须是 block 的整数倍且取值只能 ±1、折叠张量的 *种类* 白名单（只有 `build_lora_mm(_id)` 消费的权重）、架构白名单 |
| `llama-model.cpp`（`load_tensors` 尾部） | 按 (block_size, buft) 建一张 H，按 (width, buft) 建符号向量，放在**权重所在的那个 buffer type** 上；H 对称 ⇒ 同一张张量兼作前向与逆向旋转；逆向（查表）优先跟随前向所在的 buft，否则逐 token 变换要来回过 PCIe；tied output 把 LM head 记在 `output`（loader 复制的那个）上、把查表记在 `tok_embd` 上 |
| `llama-graph.cpp` | `build_hadamard_activation()`：可选 GDN 头重排 → 乘 signs → `llama_mul_mat_hadamard()`（复用本树已有的 `GGML_HINT_SRC0_IS_HADAMARD` FWHT 提示），按 (激活, rot, signs, perm) 记忆化（qkv/gate/up 共用同一个激活）；接进 `build_lora_mm` 与 `build_lora_mm_id`；`build_inp_embd` 在 token 查表后立刻做逆变换 |
| `llama-context.cpp` | 把两张 map 传进 `llm_graph_params`，并在**首次**带折叠权重的图计算前做一次图级校验：每个折叠权重必须被"src1 是指向同一 rot 的 hint matmul"消费，每个 latent 查表必须先旋转。这条守卫把"静默算错"变成报错 |

**验收（CPU，同一台机器）**

| 项 | 本树 | prism 参照 |
|---|---|---|
| 加载汇总 | `loaded 152 Hadamard-folded weight(s) (1 inverse-lookup) using 1 rotation(s) and 2 sign vector(s)` | 同 |
| PPL（同一 2×512 token 片段） | **13.9223** | **13.9484** |
| 贪心续写 `The capital of France is` | 同一段文本 | 同一段文本 |
| 回归（无折叠元数据的 qwen2.5-0.5b Q4_K_M） | 20.8087（= 阶段一基线） | — |

差异 0.19% 来自本树（buun lineage）自身在 CPU 数学/repack 上的发散，不是折叠错误：折叠错了会掉到几千以上。

**反向控制**：把 `build_hadamard_activation` 短路后，进程在首次 compute 处直接终止、无输出（守卫生效）；
还原后 PPL 精确复现 13.9223。

## 6 阶段三：CUDA 原生 mmq/mmvq（已落地，未上卡）

| 类型 | mmvq | mmq |
|---|---|---|
| PQ2_0 | `vec_dot_pq2_0_q8_1`，VDR 1（4 个 32 元素块/128 权重，同一 scale） | `load_tiles_pq2_0` = Q2_0 布局 + group-128 常量 |
| PTQ1_0 | `vec_dot_ptq1_0_q8_1`，VDR 4（一次吞整块 128 权重，qi/vdr=1）；四字节并行走 base-3 链，dp4a 累加数字 + 并行累加激活和来抵消 −1 偏置 | `load_tiles_ptq1_0` 把 qs/qh 展开成 int8 写在 smem，8 lane/块，全部走同一条 5 步链 ⇒ 无分支发散；lane 6 处理 qh（读 offset 24 会顺带读到 scale 前两字节，靠 `v_hi = v_lo` 丢弃） |

`should_use_mmq`：PQ2_0 并入 Q1_0/Q2_0 组；PTQ1_0 只在 `turing_mma_available` 时接受，但**默认全 batch 都走 MMQ**
（fork 记录 fp16 反量化 + cuBLAS 正是 PTQ1_0 在 CUDA 上掉精度的来源），`GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH` 留作 A/B 开关。
`mul_mat_id` 不再绕开这两种类型。

**未移植**：fork 的 SoA/exact-isum 单列路径与 PT 列共享核（`mmvq-pq2_0.cuh`、`mmvq-ptq1_0.cuh`）。
后果只在多列（speculative 验证）与单列 PTQ1 的线程利用率上，正确性不受影响。

**验证**：`ggml-cuda.lib` 静态构建 72/72 链接通过（0 error），两个 `mmq-instance-{pq2_0,ptq1_0}.cu.obj` 生成。
无 GPU，未上卡；解码数学改用"模拟 `__byte_perm`/`__vsub4` + 索引算术"对照参考反量化：
`vec_dot_ptq1_0_q8_1` 300/300 随机块逐元素一致，mmq 瓦片 lane 映射 100/100 一致。

## 7 `gguf-py` 常量与 HF 转换脚本（已落地）

- `constants.py`：`GGMLQuantizationType.PQ2_0=142 / PTQ1_0=143`、`GGML_QUANT_SIZES (128,34)/(128,28)`、
  `LlamaFileType.MOSTLY_PQ2_0=141 / MOSTLY_PTQ1_0=143`。
  ⚠️ fork 的 gguf-py 写的是 128/129，与它自己的 `llama.h`（141/143）和线上 gguf（`general.file_type=143`）都矛盾，
  这里按 `llama.h` 对齐。
- `conversion/base.py`：`hadamard_folded_names()` + `add_hadamard_metadata()`，把外置 packer 留下的
  `hadamard_packing.json` 转成 GGUF 元数据，校验规则与运行时一致（写出的 11 个键与运行时读取的 11 个键完全对齐）。
- `conversion/qwen.py`：折叠过的 `linear_attn.in_proj_qkv` / `out_proj` 跳过 V 头重排，改置 `gdn_v_grouped`。
- 未端到端跑：复现 Bonsai gguf 需要打包好的 checkpoint 和外置 packer，本树都没有。

## 8 未移植：Vulkan / SYCL

两者都是**重量级重写**而非机械搬运，原因可量化：

**Vulkan**（fork 侧 +235/−19 `ggml-vulkan.cpp`、+750 行 5 个 shader）
- 本树 `ggml-vulkan.cpp` 相对同一基线已发散 **+2393/−1216**；fork 的改动依赖它自己那套按类型编译的
  `mul_mat_vec`，而本树已经把量化 matmul 收成**单个 shader + `MmTypeA` spec constant 的 switch**。
  因此不是"把 shader 拷过来"，而是要把解码逻辑接进这个 switch，再加 C++ 侧的 pipeline/type 表 —— 等于重写。
- 环境是具备的（`D://Program Files\Vulkan`、glslc 都在），所以这件事可做，但要单独排期。

**SYCL**（+约 6000 行，其中 TernSYCL/Xe2 kernel 3000+ 行）
- 需要 Intel oneAPI（`icpx`）才能编译，本机没有 ⇒ 盲改无任何验证。
- Xe2 的 int2/int8 dpas、w8a8、pq2_xmx 都是体系结构专属优化，没有对应硬件时风险远大于收益。

**当前状态**：Vulkan/SYCL 上这两种类型会在 `supports_op` 处被拒绝，权重留在 CPU 跑 —— 正确但无加速。

## 9 仍存缺口

- **CUDA 未上卡**：CUDA 树在本机无法整体链接（见下），只能验证到 `ggml-cuda.lib` + 数学模拟。
- ⚠️ **本树静态构建当前是断的（与本移植无关）**：`ggml-backend-meta.cpp` 把
  `ggml_backend_dev_is_meta` / `ggml_backend_meta_dev_n_devs` / `ggml_backend_meta_buft_n_bufts` 等定义成 `static`，
  而 `src/llama-kv-cache.cpp`、`common/fit.cpp` 在调用它们 ⇒ 静态构建链接 `llama.lib` 必挂（LNK2019，
  `nm` 在 ggml-base 自己的 obj 里是 `t`）。共享构建（`BUILD_SHARED_LIBS=ON`）因为带上 `GGML_SHARED` 宏反而正常，
  CPU 验证就是在共享树上做的。
- `-np > 1`、KV cache 用这两种类型当 `-ctk/-ctv`：不支持（它们只用于权重）。
- CPU repack 的 `PQ2_0 4x8` 变体（AVX-512+VNNI 专用）未移植。
