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
| Vulkan 后端 | ✅ 已落地；折叠模型 `-ngl 99` 真跑通，PPL 与 CPU 一致，见 §8 / §8.1；q8_1 mat-vec + coopmat 接入见 §6.2 |
| SYCL 后端 | ❌ 放弃（缺 oneAPI 工具链，3000+ 行 Xe2 专属），见 §8 |

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

### 6.1 PTQ1_0 的 small_k warp-per-row 几何（后补）

fork 的 `small_k` 参数本身早就有了（`calc_nwarps(..., small_k)` +
`calc_rows_per_block()` 里 `small_k ? nwarps : 1`），本树也已具备。缺的是 fork 针对
**PTQ1_0 + 1 列 + SoA** 的专用 K 循环：通用 small_k 循环按 `kbx = tid; kbx += 128` 让整个 block 瓜分
K-block，K=5120（40 块）时只有 0..39 号线程在搬数据 —— warp 0 满载、warp 1 只剩 8 lane、
warp 2..3 在 `__syncthreads` 空占 SM warp 槽。fork 用 CUPTI 在 RTX 4070 上量的组内 GEMV 带宽是
**K=5120 全部档位 350–390 GB/s，而 K=17408（四个 warp 都在流）460 GB/s**。

改法（`ggml/src/ggml-cuda/mmvq.cu`）：warp `w` 独占第 `row0+w` 行，32 个 lane 分担该行的 K-block，
归约用 shuffle，没有共享内存也没有 block barrier；激活会被每个 warp 重读，但只有 20 KB 且常驻 L1/L2。
依赖全部齐备（`vec_dot_ptq1_0_q8_1_multi` 与 fork 同签名，fusion/gate/bias 变量名一致），
只多补了本树独有的 `GGML_GLU_OP_SWIGLU_CLAMP` 分支。

**与 fork 的唯一偏差**：fork 用编译期模板参数 `y_soa` 把这段代码只实例化给 SoA；本树的激活布局是
运行时参数 `y_layout`（见 6.2），所以门退化成一次运行时判断，非 SoA 的 launch 落到下面的通用循环。

⏳ **未上卡验证**：本机无 GPU，只做到 `ggml-cuda` 编译通过。

### 6.2 Vulkan：q8_1 mat-vec 与 coopmat 接入（后补）

移植前先量了一遍差距，结论是**缺的从来不是 MMQ/coopmat 引擎本身**：

| 面 | 本树移植前 | fork |
|---|---|---|
| `mul_mmq.comp`（int MMQ） | 有 | **有，且两边 `mul_mmq_funcs.glsl` 逐字节相同（488 行），fork 也没加 PTQ1_0/PQ2_0 分支** |
| coopmat（`mul_mm.cm1`） | 有（327 处引用） | 有，fork 多 14 条 `CREATE_MM2` |
| q8_1 mat-vec（GEMV） | **PQ2_0/PTQ1_0 缺失** | `mul_mat_vecq_ptq1_0.comp` 404 行 + `ptq1_0.glsl` 38 行 |

⇒ 实际工作量只有两块：把两种格式接进既有的 coopmat 与 q8_1 mat-vec 管线。

**q8_1 mat-vec（decode 主力）**：`mul_mat_vecq_ptq1_0.comp` / `mul_mat_vec_ptq1_0.comp` /
`ptq1_0.glsl` 从 fork 直拷（都走 `data_a_u32` 原始 dword 视图，不依赖 `A_TYPE`，
与本树 `types.glsl` 无冲突；`mul_mat_vec_base.glsl` 两边逐字节相同）。`PQ2_0` 复用通用
`mul_mat_vecq.comp`，在 `mul_mat_vecq_funcs.glsl` 加 `DATA_A_PQ2_0` 分支：

- ⚠️ **`get_dm` 与 `repack4` 都要按 128/32 = 4 折叠块号**。`ib` 是 q8_1 块索引（32 激活），
  而 PQ2_0 块是 128 权重 ⇒ 同一 PQ2_0 块被 4 个不同 `ib` 命中，块内窗口号 = `ib % 4`。
  Q2_0 不需要这一步是因为它的两个 64 宽半块**交错**同一段字节（`qs_idx = (ib&1)*4 + iqs*2`），
  PQ2_0 的 qs 是顺序排布的。**第一版漏了这个折叠 ⇒ 24 个 MUL_MAT 用例全 `ERR = inf`。**
- `mul_q8_1` 与 `unpack_q2_0` 提到 `DATA_A_Q2_0 || DATA_A_PQ2_0` 公共守卫下（否则 PQ2_0
  分支看不到它们，`no matching overloaded function`）。

**生成器的两处白名单必须同步**：编译循环（`string_to_spv`）与 `arr_dmmv_*_q8_1_f32` 符号产出
各有一份类型清单，漏改后者只会得到 `C2065: 未声明的标识符`（GLSL 全过、C++ 才炸）。
已提取成 `has_q8_1_mmvq()` 单一判据。

**coopmat 注册要限定在 coopmat 块内**：⚠️ **`non_lut_quant_types[]` 是全后端共享清单，有 10 个
消费点**（coopatmat2 subgroup、subgroup、`mul_mmq` int-dot 等）。把两种格式加进去会让
**无 coopmat 的设备**（本机 Intel iGPU）也创建 `sg_create_quant` pipeline，而
`mul_mmq_funcs.glsl` 没有对应分支 ⇒ shader 什么都不解 ⇒ **19 个 PTQ1_0 用例 `ERR = inf`**。
正解是新开 `coopmat_only_quant_types[]`，只在 `cm1_create_quant` 那个循环里用（对齐 fork 把
`CREATE_MM2` 放在 coopmat 块内的做法）。

**验证（Intel UHD 730）**：`test-backend-ops -b Vulkan0` 的 `MUL_MAT` 全量 **0 FAIL**
（ptq1_0 / pq2_0 定向各 0 FAIL），`GET_ROWS` 0 FAIL。折叠模型
`Ternary-Bonsai-2-2B-PTQ1_0.gguf -ngl 99` 的 PPL = **2.8440 +/- 0.31715**，与移植前
（dequant → f16 路径）**逐位相同**。

⏳ **未上卡验证**：本机无 coopmat（`device->coopmat_support == false`），coopmat 注册
只做到编译通过，prefill 的 GEMM 收益要在 V100 / 950 上重测。

## 6.3 激活布局：SoA 单列 / PT 列共享 / PQ2 多列（已落地）

fork 的多列核不是"再写一个 vec_dot"，它换掉了**激活的存放顺序**。本树按同一套设计接进来：
`ggml_cuda_q8_1_layout`（AOS / SOA_ISUM / PT / PQ2），由 `ggml_cuda_q8_1_layout_host(type, ncols, ids, plain_2d)`
一次决定，量化器、padding 与 kernel 派发三方共用同一个值（`common.cuh`）。

| 布局 | 用在 | 存放方式 |
|---|---|---|
| `SOA_ISUM` | PTQ1_0、1 列、非 Ampere、非 batch-invariant | 按 32 个 K-block 分组，组内 word w 连续（`ggml_cuda_ptq1_q8_word`），`ds.y` 里放**精确整数和** |
| `PT` | PTQ1_0、2–8 列或 MoE ids，以及 Ampere/batch-invariant 的 1 列 | 每个 128 权重的 8 个 16 字节平面 + 一个 `(d,s)` 平面；列跨度仍是 `block_q8_1` 的字节数 |
| `PQ2` | PQ2_0、3–8 列、**仅 sm_89** | 列内 qs 在前（每 16 元素组内置换），随后每 32-block 一个 `half2(d, 整数和)` |
| `AOS` | 其他一切，以及两条遗留入口（多卡切分 `ggml_cuda_op_mul_mat_vec_q`、moe-cache） | 普通 `block_q8_1` |

- `quantize.cu`：`quantize_q8_1<layout>` 四路模板；`quantize_row_q8_1_cuda` 改吃 layout 而不是 `type_src0`。
  `SOA_ISUM` 需要把 K padding 到 `GGML_CUDA_PTQ1_K_PAD`(4096)，否则分组不完整。
- `vecdotq.cuh`：`vec_dot_ptq1_0_q8_1_multi<ncols>`（SoA 版，一次吃整块并减去精确整数和）。
- `mmvq-ptq1_0.cuh`：`ptq1_0_pt_block_dot<ncols,nrows>`（PT 版的共享解码）+ 专用的
  `mul_mat_vec_ptq1_0_pt`（1–8 列、`rows_per_cta` 填满 128 线程、partial 写 smem 后统一归约，
  `GGML_CUDA_BATCH_INVARIANT` 时走 warp 归约以保证列数无关）。
- `mmvq-pq2_0.cuh`：`mul_mat_vec_pq2_0_mc`（3–8 列，4 行/warp，激活切片跨行复用）。
- `mmvq.cu`：新增 `y_layout` 参数贯穿 `switch_type → switch_ncols_dst → switch_fusion → kernel`；
  kernel 里 `if constexpr (type == GGML_TYPE_PTQ1_0)` 三路（SoA / PT / AoS 兜底）；
  `calc_nwarps`/`calc_rows_per_block` 增加 PTQ1_0 多列规则；`should_use_mmvq` 对 PTQ1_0 判 `ne11 <= 4`
  （fork 在 Ada 上量到 8 列时 MMQ 反超）。

⚠️ **AoS 兜底不是可有可无的**：`ggml_cuda_op_mul_mat_vec_q`（多卡切分）与 `ggml_cuda_moe_cache_mmv`
喂进来的激活是别处按 AoS 量化的，若只写"SoA/PT 两分支"，这两条路径会静默算错。

**未对齐 fork 的一处**：fork 在 `small_k && PTQ1_0 && 1 列 && SoA` 时另有一套 warp-per-row 几何
（每个 warp 独占一行、无 smem、无 barrier），这里没搬 —— 正确性不受影响，只是 K 较小的 1 列 GEMV
少了那点并行度。V100 上是否值得补，等 §测试方案 1.2/3.3 的实测数字再定。

**验证**：`ggml-cuda.lib` 静态构建 72/72 链接通过（0 error），两个 `mmq-instance-{pq2_0,ptq1_0}.cu.obj` 生成。
无 GPU，未上卡；解码数学改用"模拟 `__byte_perm`/`__vsub4` + 索引算术"对照参考反量化：
`vec_dot_ptq1_0_q8_1` 300/300 随机块逐元素一致，mmq 瓦片 lane 映射 100/100 一致。

## 6.4 Vulkan：int MMQ（`mul_mmq.comp`）接入（已落地）

`mul_mmq.comp` 本身按类型分支（`block_a_to_shmem` / `block_a_to_registers` / `mmq_dot_product`），
所以接入点是"补类型"：

| 文件 | 改动 |
|---|---|
| `mul_mmq_shmem_types.glsl` | 两种格式各一份 `block_a_cache`：`int32_t qs[8]` + `FLOAT_TYPE dm`，`QUANT_R_MMQ 1`（与 Q2_0 同宽：32 权重/子块，128 权重/块 ⇒ 4 个子块共享一个 scale） |
| `mul_mmq_funcs.glsl` | PQ2_0：`block_idx = ib/4`、`byte_idx = (ib&3)*8 + iqs`，取码与 Q2_0 相同，−1 偏置仍靠 `- cache_b.ds.y` 抵消；PTQ1_0：`ptq1_0_trit()` 用与 `dequantize_row_ptq1_0` 同一套 base-3 余数递推（`v = (v*3)&0xFF`、`t = (v*3)>>8`）逐元素出 −1/0/1，`mmq_dot_product` **不减** `ds.y`（trits 自带符号） |
| `vulkan-shaders-gen.cpp` | 新增 `has_q8_1_mmq()` 单判据（= `has_q8_1_mmvq()` 去掉 `iq1_*`），编译循环只认它 |
| `ggml-vulkan.cpp` | `ggml_vk_matmul_int_shmem_support` 加两档尺寸；三处注册（`matmul_{pq2_0,ptq1_0}_q8_1` / `matmul_id_subgroup_*` / `matmul_id_*`） |

**验证（Intel UHD 730 真跑）**

| 项 | 结果 |
|---|---|
| `test-backend-ops -o MUL_MAT -b Vulkan0` | 0 FAIL；pq2_0 / ptq1_0 各 104 例，其中 17+17 例命中 `matmul_{pq2_0,ptq1_0}_q8_1`（临时探针实测，n≥9 时 `quantize_y=1`） |
| `-o MUL_MAT_ID -b Vulkan0` | 0 FAIL；152 例，38+38 走 id 版 |
| 折叠模型 PPL（`-ngl 99 -c 512 -b 512`，同一语料） | int MMQ **53.2516** vs `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`（dequant→f16）**53.2835**，差 0.06% |

⚠️ **这一档在 iGPU 上不快**：同一 PPL 的 prefill 是 **9.39 s/pass**（MMQ）对 **8.78 s/pass**（dequant→f16），
约 **+7%**。PTQ1_0 的取码在 `block_a_to_shmem` 里是逐元素余数递推（每 4 元素一次内层循环），
比 Q2_0 那种位取码贵；V100 上是否反超未测。真要提速，把"同一 word 内 digit 序号 n 恒定"这一点用上
（`qs[0..79]` 段按 16 步进、`qs[80..119]` 段按 8 步进），可以把内层循环换成一次 `b*3^n mod 256`。

## 6.5 CUDA 首跑崩溃：mmvq 的分派表少了这两型（已修）

**症状**：V100 上 `llama-server -ngl 99`（`-ctk/-ctv f16`、`--flash-attn on`）启动即崩，日志末行是
`ggml/src/ggml-cuda/mmvq.cu:1762: fatal error`；`-ngl 0` 能起。ptq1_0 与 pq2_0 两个模型表现相同。

**根因**：`1762` 是 `mul_mat_vec_q_switch_type()` 的 `default: GGML_ABORT("fatal error")`。这一型的
vec_dot、转换核、`mul_mat_q_case`、`should_use_mmvq`（`ne11 <= 8`）都已就位，**唯独那张 case 列表没有
`PQ2_0` / `PTQ1_0`** ⇒ 走到 decode 步（warmup 就是 1 token）直接 abort。又一处"函数在 + 有单测 ≠ 接线了"：
`test-quantize-fns` 只测 codec，`test-backend-ops` 从未在这台卡上跑过。

`ggml_cuda_op_mul_mat_vec_q()`（多卡切分）走的是同一个 `mul_mat_vec_q_switch_type`，所以一处补齐两处都通。

**验证**：`ggml-cuda.lib`（`CMAKE_CUDA_ARCHITECTURES=70`）0 error；⏳ 待 V100 上先跑
`test-backend-ops -b CUDA0 -o MUL_MAT`（这是唯一能在真卡上逐元素验收 `SOA_ISUM` / `PT` / mmq 三条路的东西）。

## 7 `gguf-py` 常量与 HF 转换脚本（已落地）
- `constants.py`：`GGMLQuantizationType.PQ2_0=142 / PTQ1_0=143`、`GGML_QUANT_SIZES (128,34)/(128,28)`、
  `LlamaFileType.MOSTLY_PQ2_0=141 / MOSTLY_PTQ1_0=143`。
  ⚠️ fork 的 gguf-py 写的是 128/129，与它自己的 `llama.h`（141/143）和线上 gguf（`general.file_type=143`）都矛盾，
  这里按 `llama.h` 对齐。
- `conversion/base.py`：`hadamard_folded_names()` + `add_hadamard_metadata()`，把外置 packer 留下的
  `hadamard_packing.json` 转成 GGUF 元数据，校验规则与运行时一致（写出的 11 个键与运行时读取的 11 个键完全对齐）。
- `conversion/qwen.py`：折叠过的 `linear_attn.in_proj_qkv` / `out_proj` 跳过 V 头重排，改置 `gdn_v_grouped`。
- 未端到端跑：复现 Bonsai gguf 需要打包好的 checkpoint 和外置 packer，本树都没有。

## 8 Vulkan（已落地）/ SYCL（放弃）

**Vulkan**：本树的 `mul_mat_vec.comp` 是**单 shader + `MmTypeA` spec constant** 的结构（fork 那套按类型
编译的 `mul_mat_vec_<type>.comp` 在本树只有 `_k`/`tq` 还在），所以接入点是"补类型"，不是"搬 shader"：

| 文件 | 改动 |
|---|---|
| `types.glsl` | `QUANT_K_PQ2_0/PTQ1_0 = 128`、`QUANT_R = 1`、`block_pq2_0`（`d` + 32B qs）、`block_ptq1_0`（24B qs + 2B qh + `d`）、`A_TYPE` 绑定 |
| `dequant_funcs.glsl` | `dequantize`/`dequantize4`/`get_dm`；PTQ1_0 用 `ptq1_0_val()` 逐元素走 base-3 余数递推（与 CPU codec 同构），PQ2_0 直接复用 Q2_0 的 2-bit 取码 |
| `dequant_{pq2_0,ptq1_0}.comp` | 反量化核；PTQ1_0 版一次出 8 个元素、把 `pow3[n]` 提到 8 次循环外 |
| `vulkan-shaders-gen.cpp` | `type_names` 加两项 → 自动生成 `dequant_*`、`get_rows_*`、`mul_mat_vec_{pq2_0,ptq1_0}_{f32,f16}_f32*`、`mul_mat_vec_id_*` |
| `ggml-vulkan.cpp` | 5 处类型白名单（`ggml_vk_get_to_fp16`、`dequantize_mul_mat_vec{,_id}`、`supports_op` 的 MUL_MAT 与 GET_ROWS）+ dmmv/dequant/get_rows 管线创建（`2*rm_stdq`，与 Q1_0/Q2_0 同档） |

**取舍**：decode 走 `mul_mat_vec`（q8_1，见 §6.2），prefill 走 coopmat（§6.2）与 int MMQ（§6.4）。
PTQ1_0 的反量化是逐元素而非按字节并行，1.75 bpw 的 GEMV 仍然是带宽受限。

**已验证（开发机 Intel UHD 730，真跑）**：
`test-backend-ops -o MUL_MAT -b Vulkan0` **1236/1236 通过**，
`-o GET_ROWS -b Vulkan0` **255/255 通过**，两种新格式全部覆盖（m=67 行尾、n=1..64、k∈{128,256,1024,5120}、
batch 维、`mul_mat_id`）。同机 CPU 侧 1465/1465。

> 这一步抓到一个真缺陷：`dequant_pq2_0.comp` 最初按 Q2_0 的 64 宽块写了 "4 块/组"，128 宽应该是
> "2 块/组"（64 invocation × 4 元素 = 256 元素 = 2 个 128 权重的块）。表现是 n≤8 全绿（走 mul_mat_vec、
> 不经过这个 shader），n≥9 与带 batch 维的用例全红。修正后 0 失败 —— 这就是"构建通过 ≠ 正确"的实例。

### 8.1 折叠模型在 Vulkan 上"静默退出"——已定位并修复

**症状**：`Ternary-Bonsai-2-2B-PTQ1_0.gguf -ngl 99` 无任何 error 输出直接退出（退出码 `0xC0000409`），
日志停在 `computation-frontier ratchet`。**日志"干净截断"本身就是判据**：末尾是完整普通行、无 error ⇒
不是 abort 打印失败，而是异常在**第一条日志之前**抛出。

**定位手段**（可复用）：`common/log.cpp` 的日志是**异步 worker 线程**打印的，进程被终止时队列里的内容全丢
⇒ 任何"静默"故障都看不到上下文。加一个 `GGML_LOG_SYNC=1` 的同步旁路后，日志行行落地，
再按 `graph_compute → hadamard 校验 → sched` 三段插桩即可二分。

**根因 1（校验时机）**：`llama_verify_hadamard_graph()` 原本在 `llama_context::graph_compute()` 里调用，
即 **`ggml_backend_sched_reserve()` 已经改写过这张图之后**。图切分会在 split 边界插入 copy 并把**消费者的
`src[]` 重写成 copy 张量**，于是"embedding 的 `GET_ROWS` 经 reshape 喂给带 hint 的 `MUL_MAT`"这条链在
指针层面断了，校验误判成 `Hadamard-latent table 'token_embd.weight' is read without the inverse transform`
并抛异常（无人 catch ⇒ terminate）。实测该图 `graph splits = 2 / sched copies = 1`，被改写的正是
embedding 的 `MUL_MAT.src[1]`（指向调度器缓冲区，`op == GGML_OP_NONE`，因此 `unwrap()` 停在它上面）。
`-ngl 1` 之所以通过是巧合：那一档图恰好没在该边界插 copy。

**修法**：把校验移到 `llama_context::graph_reserve()` 里 `model.build_graph()` 之后、调度器接手之前 ——
折叠契约是"模型构图器产出的那张图"的性质，不是"调度后那张图"的性质。

**根因 2（dequant 全量 dispatch 溢出）**：修完根因 1 后 prefill 立刻撞第二个断言
`GGML_ASSERT(wg0 <= maxComputeWorkGroupCount)`。插桩显示罪魁是
`dequant_ptq1_0 els=(508559360,1,1) wg=(496640,1,1)`，而设备上限是 65536：
`token_embd.weight` 是 2048 × 248320，**整个权重一次性反量化**。触发条件是
`ggml_vk_mul_mat_q_f16()` 里 `qx_needs_dequant = mmp_map == nullptr || x_non_contig` ——
本机（Intel UHD 730）没有 coopmat ⇒ `mmp_map == nullptr` ⇒ 恒为真。上游类型没暴露这个问题，
是因为它们能走 Q8_1 mat-vec（`ggml_vk_get_dequantize_mul_mat_vec(..., GGML_TYPE_Q8_1, ...)` 的白名单里有），
不必 dequant 权重；PQ2_0/PTQ1_0 不在该白名单里。

**修法**：dispatch 侧分块（不动 shader）。每次发 `min(剩余, maxComputeWorkGroupCount[0] * wg_denoms[0])`
个元素，**两个 binding 的 offset 同步前进**、`p.nel` 缩到本次块数 —— shader 本来就从 0 开始索引绑定区间，
所以每个元素仍恰好被写一次。非连续的 `cpy` 路径保持单次 dispatch。

**验证（Intel UHD 730 真跑，同一语料 1355 token / `-c 512 -b 512`）**：

| 运行 | PPL |
|---|---|
| `-ngl 99`（Vulkan 全卡） | **2.8440 +/- 0.31715** |
| `-dev none -ngl 0`（纯 CPU 参照） | **2.8522 +/- 0.31860** |

⇒ Vulkan 与 CPU 一致（差 0.3%，浮点累加顺序）。`llama-cli -ngl 99` 亦跑通
（prefill 11 token + 24 token decode，`graphs reused = 31`）。

**SYCL**：**放弃**。需要 Intel oneAPI `icpx`（本机没有 ⇒ 盲改零验证），且 fork 那 3000+ 行是 Xe2 专属
dpas/w8a8/pq2_xmx，没有对应硬件时风险远大于收益。

### 8.2 `-ngl 0 -nkvo` 撞 descriptor 断言——已修

**症状**：折叠模型在 `-ngl 0 -nkvo`（权重全 CPU、KV 回 host，但 Vulkan 仍留在 scheduler 里）下
`GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size())` 失败；同参数的 Q4_K_M 模型与 `-dev none` 都正常。

**定位**（临时插在 `ggml_vk_dispatch_pipeline` 的断言前，未提交）：
`idx=6 size=6 req=3 pipeline=dequant_ptq1_0 node=result_output` —— 该节点只请求了 1 个 descriptor set，
却派发了 6 次以上。

**根因**：§8.1 根因 2 的修法把 dequant 改成**按 chunk 循环 dispatch**，但请求处仍写死 `1`。
这两件事必须同时改，否则"请求数 < 派发数"只在图的 req 累加刚好贴着 size 时才暴露 ——
`-ngl 99` 的大图有富余，所以一直没红；`-ngl 0 -nkvo` 时 Vulkan 侧只剩少数节点，第 7 次 dispatch 就撞线。

**修法**：chunk 大小抽成 `ggml_vk_dequant_chunk()`，请求与循环共用同一个值，
请求数 = `CEIL_DIV(x_ne, chunk)`。

**验证**（`-ngl 0 -nkvo -c 512 -b 512`，同一语料）：
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` 强制走 dequant→f16 分块路径 → 7 个 chunk 全跑完，
PPL **53.3759**（= 修前那条命令要算的数值），RC=0、无 assert；默认参数下 output 层已改走 int MMQ（§6.4），
`-ngl 0` / `-ngl 99` 分别 53.2621 / 53.2516，均 RC=0。
修前同一命令在第一个 pass 内就撞 `idx=6 size=6 req=3`。

## 9 仍存缺口

- **CUDA 仍未上卡**：本机没有 NVIDIA 卡，CUDA 侧到 `ggml-cuda.lib` + 解码数学模拟为止，
  实机判据见 `docs/MERGE_PRISM_TESTPLAN.md`。Vulkan 侧已在 Intel UHD 730 上真跑（§8、§6.4）。
- `-np > 1`、把 PQ2_0/PTQ1_0 当 `-ctk/-ctv`：不支持（它们只用于权重）。
- CPU repack 的 `PQ2_0 4x8` 变体（AVX-512+VNNI 专用）未移植；本机 i5-12400 无 AVX-512，该路径本来也不启用。
- 测试里的 `test-ptq1_0-cuda-dot.cpp` / `test-ptq1_0-element-map.cpp` 是**转写件**（把 CUDA 侧逻辑抄到主机上跑），
  它们绿只能证明转写件与 CPU codec 一致；与真实 kernel 的一致性靠转写时的人工核对 + 实机 test-backend-ops。
