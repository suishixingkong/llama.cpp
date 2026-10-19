# prism-llama 相对官方 llama.cpp 的新增功能（AUDIT）

> 口径：`prism-llama` 远端 `PrismML-Eng/llama.cpp` 分支 `prism`（开发名 `prism-v7`），
> 以官方 `ggml-org/llama.cpp` 为上游。本文只统计 **fork 独有提交**，
> 不含上游同期演进（上游在同期前进了 1107 个提交，两者不同步，见 §8）。

## 0 基线与规模

| 项 | 值 |
|---|---|
| merge-base | `5ea87ddad`（2026-08-25，上游 #27538 `webgpu: fix ARGSORT/TOP_K inf`） |
| prism HEAD | `118091848`（2026-10-18，`cuda: add opt-in staged GDN verification kernel (#369)`） |
| fork 独有提交 | 167 个，无 merge 提交 ⇒ 纯线性叠在上游之上，未做上游同步 |
| 改动规模 | 281 文件 / +40179 / −1473；其中新增文件 97 个 |
| 改动分布（插入行） | SYCL 9284 / tests 5868 / CUDA 5503 / llama 核心 5083 / CPU 3320 / common 2839 / Metal 2523 / CI 1153 / Vulkan 1135 |
| 复现 | `git remote add prism https://github.com/PrismML-Eng/llama.cpp && git fetch prism prism`；`git log --no-merges 5ea87ddad..prism/prism` |

一句话：**这不是"加了几个开关"的补丁集，而是给一个私有低比特格式族（Bonsai）配套的完整栈** ——
新量化格式 + 与之绑定的 Hadamard 权重折叠 + 低比特内核（每后端一套）+ 为这套模型量身做的投机解码与
混合注意力（GDN）运行时。

五条主线：

1. **私有量化格式** `PQ2_0` / `PTQ1_0`（ggml type 142 / 143）
2. **Hadamard 权重折叠运行时**（`prism.hadamard.*` GGUF 契约 v1/v2）
3. **投机解码**：DSpark / DFly / DFlash2 / MTP 链式草稿
4. **GDN（gated delta net）**：rows 读、raw gates 折叠、回滚改为重放
5. **K 缓存量化补强** `--kv-mean-center` + 内存/服务器若干延迟优化

---

## 1 私有量化格式（能跑 Bonsai 的前提）

`ggml/include/ggml.h` 把 `GGML_TYPE_COUNT` 从 43 抬到 144，插入两个高 id 的私有类型：

| 类型 | id | 结构 | 码率 | 作用 |
|---|---|---|---|---|
| `GGML_TYPE_PQ2_0` | 142 | group-128 的 2-bit（每 128 权重一个 fp16 scale） | 2.125 bpw | 上游 `Q2_0` 是 group-64（id 42）；g128 比 g64 小约 6%，是 Bonsai 三元/2-bit 权重的主力格式。**官方 llama.cpp 无法加载** |
| `GGML_TYPE_PTQ1_0` | 143 | group-128 三值（-1/0/1），base-3 trit 打包 | 1.75 bpw | 对"本身就是 group-128 三值"的 checkpoint **无损**；上游 `TQ1_0`（id 34）是 group-256，一个 scale 跨两个 128 组，必然丢一个组尺度 ⇒ 表达不了这类权重 |

配套：`LLAMA_FTYPE_MOSTLY_PQ2_0 = 141`、`..._PQ2_0_LEGACY = 142`（改名前的历史值，线上 gguf 里还有）、
`..._PTQ1_0 = 143`（`include/llama.h`）；量化/反量化、`get_rows`、`type_traits`、CPU/CUDA/Metal/Vulkan/SYCL/HIP/WebGPU
各后端 dispatch 全部补齐（提交 `67c1af28b` → `8bbb28b76` → `01fd9521c` 一条链）。

还有一层"防呆"：`gguf: detect legacy Prism Q2_0 (group 128) files and point to PQ2_0 / group-64 downloads`（`f2cf28855`）——
旧仓库里叫 `*-Q2_0.gguf` 的文件实际是 group-128 存在 id 42 下，加载时直接报错并告诉用户下哪个文件，
而不是静默算错。`README` 给了三种候选文件的取舍表。

## 2 Hadamard 权重折叠（`prism.hadamard.*`）

上游只有 **KV-cache 侧的** Hadamard 旋转（`attn_rot_k/v`，抑制 K/V 量化离群值）。本 fork 加的是
**权重侧折叠**，两者是不同机制：

- 权重在转换时被旋转到 Hadamard 基（`H·W`），运行时只需把**激活**旋转一次：`x' = H x`，再走普通 `mul_mat`。
  旋转用 FWHT hint 表达（`ggml_mul_mat_hadamard`），成本 O(n log n)，远低于一个额外矩阵乘。
- 元数据契约 `prism.hadamard.{block_size, transform, axis, sign_mode, weight_names, inverse_weight_names, version, tied_output}`，
  加载期校验非法组合并**响亮失败**（`src/llama-model.cpp:1207-1270`）。
- v1：显式输出头；**v2 `tied_output`**：`token_embd.weight` 同时当输出投影用（无 `output.weight`），
  lookup 侧 `s*(H z)`、输出侧 `H(s*h)` 共用同一批行 —— 省一整份 `n_vocab × n_embd` 权重（`docs/development/hadamard-tied-output.md`）。
- `-sm` 符号张量支持（Bonsai 2 / qwen35 折叠权重）：符号不再逐个存 sign 向量，而是由图里的张量提供（`71c147320`）；
  Metal 把 sign 翻转**折进 FWHT kernel**，省掉每层一个 MUL（`00e818761`）。
- 多个折叠权重共享同一输入时共享同一次激活变换（`633168fb6`），`hadamard_memo` 去重。
- 图级校验：折叠张量被 `mul_mat` 消费却没有注册变换 ⇒ 拒绝建图（`src/llama-context.cpp:36-90`）。
- 折叠张量不进 `CPU_REPACK` 缓冲（`0324c6652`），避免 repack 打乱旋转基。

**作用**：1–2 bit 权重的误差几乎全来自离群值；随机正交（Hadamard）旋转把能量摊平，让 ternary / 2-bit 分组量化显著更准，
而运行时只多一次 FWHT。这是"私有格式能被接受"的关键一环。

## 3 投机解码（fork 最"功能"的部分）

上游已有 `draft-dflash` 与 `draft-dspark`（在 `src/models/dflash.cpp` 里以 markov head 区分）。fork 在其上：

| 能力 | 内容 | 作用 / 证据 |
|---|---|---|
| **多层隐藏态 tap** | `llama_set_capture_layers` / `llama_get_embeddings_capture_ith`（`src/llama-ext.h`）：注册任意中间层集合，按位置拼成 `[n_layers*n_embd]` 一行 | EAGLE3-proper 与 dspark 共享的原语：dspark 要吃多层 target 特征，不是单个最终层向量 |
| **DSpark 独立化 + 专用内核** | 新 arch `LLM_ARCH_DSPARK` 与 `src/models/dspark.cpp`(573 行)；`common/dspark-markov.cu`(484) / `.mm`(240) 为 Markov head 写 CUDA/Metal 专用核 | 上游在 dflash.cpp 内用 `mul_mat` 串出来；专用核 + `LLAMA_DSPARK_MARKOV_CUDA/METAL` 开关 |
| **DSpark 新条件/头** | log-SNR 噪声级条件（`conversion/dspark.py`、`LLAMA_DSPARK_SNR_TWO_ROWS`）、confidence head 元数据、hidden correction、`LLAMA_DSPARK_SHARED_HEAD` / `dspark_head_source` 借用 target 输出头（不拷贝权重）、`CORRECTION_PREFIX` / `BALANCED_OUTPUT` / `GREEDY_IDS` / `REUSE_MASK` | 更好的接受率与小显存：头共享省一份 `n_vocab×n_embd`；SNR 条件让草稿质量随噪声级自适应 |
| **DFly（AngelSpec）** | DFlash 变体：① 每 **draft 层**各自的 context 融合（而非一层共享），② 跨块"前驱修正"链（TreeFlash）——位置 i 的 logits 用 i-1 已草稿 token 的 embedding 修正 | 更长块、更高接受率；靠 `layer_fusion` 检测，不会被误认作普通 DFlash；`docs/speculative.md` 给了 `--spec-draft-n-max` 扫描表（6 最优 42.0 tok/s vs 5 的 35.7 / 7 的 37.5） |
| **DFlash2** | 局部卷积 + 候选选择器（`p_min`）、beam selection、小 batch 的 PQ2_0 verify 优化 | 提升块内 token 选择质量 |
| **prompt lookup 放宽** | `LLAMA_DFLASH_LOOKUP` 覆盖所有非 DSpark 的 DFlash 系 drafter；命中即跳过 noise-block 前向 | 长重复 prompt（代码/日志）白捡 token |
| **MTP 链式草稿（qwen35 NextN）** | `llama_set_mtp_chain(ctx,n)`：**一次 decode 出一张图 n 个 draft token**（`llama_mtp_chain_support` 报能力位；trimmed 词表时 in-graph lookup 留在设备上）；K/V-only prompt pass；`LLAMA_MTP_EAGER_CATCHUP`；CUDA 上默认 draft depth=2 | 链式取代逐 token 往返 ⇒ 草稿阶段少 n-1 次 host/device 同步。`#354` 给出理由：CUDA 上 verify batch 成本随深度增长快于 Metal，depth 3 不划算 |
| **`--spec-draft-depth-max`** | 序列长于 N 后停止草稿（0=总是草稿） | 深上下文里 draft+多列 verify 的代价超过收益，server 可自动退化为逐 token |
| **捕获期图预约跳过** | 层捕获未变化时不重建/不预约计算图（`0082ac24a`） | 多层 tap 常驻时省掉每次 decode 的图预约开销 |
| **可观测性** | server 明确打印"启用了哪些 spec 实现""请求了但没初始化的是哪些、为什么"（`9a5c96acb`） | 以前"开了但没在草稿"是完全静默的 |

## 4 GDN / 混合注意力（qwen35 线性注意力层）

| 项 | 内容 | 作用 |
|---|---|---|
| `ggml_gated_delta_net_rows` | 按 row 索引直接读 live state（`rows[seq]`），不再为每层做一次 `get_rows` gather | 去掉 recurrent decode 图里每层一次 gather（`bed6e9a6b` / `08d8c9035`） |
| `ggml_gated_delta_net_set_raw_gates` | 把 `sigmoid(beta)`、`softplus(g + dt_bias)*A` 折进算子（scalar gate） | `#165` 六个融合之一，op/token 2359→2103（−25%），Metal 2B decode +7.7% |
| `ggml_gated_delta_net_replay` | 回滚改为**重放**：保留 1 份 base state + 每 token 的更新记录 `(delta, k, g)`，回滚 r 步 = 重放 `m−r` 条记录 | 取代 K 份 state 快照 ⇒ 长 context / 大 K 时省显存（ARC-LAB，`LLAMA_GDN_REPLAY`） |
| qwen35 ring decode | `gated_delta_net_rows` + snapshot write-fold 的 ring 路径 | 混合模型 decode 不再走快照环 |
| CUDA staged l8 verify 核 | `GGML_CUDA_GDN_L8=1`，Ampere+、32 lane warp、state 128、8–16 token verify | 8 lane 协作算一个 state 列，k/v/gate 预登 shared（Yukon 移植）。**默认关**：H200 字节级全过，但 32 头 1 seq 只有 1.219x、48 头 2 seq 是 0.864x 回归 |
| Metal | 4 个 GDN state 行/SIMD group（`445fa8209`） | state 更新吞吐 |
| Vulkan | 对 raw gates **明确拒绝**而不是算错（`a419d438a`） | 正确性优先，宁可 fallback |

## 5 KV cache 与内存

| 项 | 内容 | 作用 |
|---|---|---|
| `--kv-mean-center` | 写 K cache 前按 `(kv-head, channel)` 减去标定 bias，**仅 Q4_0** | Q4_0 是无零点的对称量化，通道均值会白吃动态范围；softmax 对整行加常数不变 ⇒ 数学等价、只提升量化保真度。配 `tools/kv-mean-center` 标定工具、`kv_mean_center.k_rot` 基校验（基不匹配拒绝加载）、`docs/kv-mean-center.md` |
| `seq_rm` 只走已用 cell 区间 | `56f21fb10` | 大 context + 投机回滚时 `seq_rm` 遍历全 cache 是 host 侧热点 |
| 待处理回滚跨 `seq_cp` 保留 | `rs_idx` 随 `seq_cp` 复制；DSV4 compressor 同理（`5ae2e169e`） | 修 bug：回滚后 fork 出的序列会读到回滚前的 state |
| server checkpoint 缓冲复用 | 被淘汰 checkpoint 的缓冲直接交给新 checkpoint | 每个 checkpoint ~150 MiB，原来每次从 0 重新 realloc+zero；多轮对话省约 18 ms/turn（645→627） |

## 6 后端内核（低比特为主，每个后端一套）

| 后端 | 主要新增 |
|---|---|
| CUDA | PTQ1_0/PQ2_0 MMQ tile loader + MMVQ（planar-transposed 激活布局：1–4 列 47/104/154/201 µs → 46/55/65/75 µs）、Hopper wgmma prefill（sm_90a，先 opt-in 后默认）、PTQ1_0 MMQ 全 batch（KLD 9.2e-4→5.6e-4、top-1 97.78→98.39%，代价 pp512 ≈ −7%）、MoE reduce 融合（bit-identical，省掉 `n_embd×n_expert_used×n_tokens` 的中间 F32 往返）、量化 K/V 进 short-query MMA FA（q4_0/q8_0 模板实例）、`GGML_FA_QUANTS` 编译裁剪、GB10 低比特专用（dual-RMS/Virtual RMS 等）、FWHT 宽块、`cudaDeviceGetAttribute` 取设备属性 |
| CPU | AVX-512 VNNI repack GEMV/GEMM（Q1_0/PQ2_0）、AVX2/AVX-VNNI、NEON+dotprod/i8mm（PTQ1_0/PQ2_0）、SSE2/SSSE3、AVX-512 tiled 单 token FA、PQ2_0 8 行 VNNI GEMM |
| Metal | PTQ1_0 exact-float trit 提取（+30–38% decode）、多列 matvec（2–4 列）、PQ2_0/Q2_0 精确 float 点积、FWHT 全宽度（+4–36%）、SwiGLU 折进 signed FWHT、im2col/set_rows 修复 |
| Vulkan | PTQ1_0 `mul_mat_vec` 专用 shader、Intel Xe2 整数点积（Arc B390 decode 8.3x）、无 DP4A 硬件回退（消除反量化分歧） |
| SYCL | 最大一块（9.3k 行）：TernSYCL（XMX DPAS int2）PTQ1_0、PQ2_0 XMX、decode attention（`fattn-dec`）、w8a8、prompt batch 反量化到 FP16、oneDNN 运行期开关 —— 面向 Intel Arc 跑 Bonsai 2 |
| WebGPU | ≤8 列走 matvec |
| HIP/ROCm | PTQ1_0 向量化 `vec_dot`、PQ2_0 原生 unpack、Q1_0/Q2_0 decode 提速 |
| x86/ARM 通用 | PQ2_0 行数非 256 倍数时留在 Q8_0 激活路径（避免契约破裂）、非 x86 链接修复 |

## 7 common / server / 工具 / 发布

- **采样**：top-k 若位于采样链首，直接从 logits 取 top-k，跳过 softmax 全量排序（`527ea4d9f`）。
- **语法**：JSON Schema → GBNF 允许按 schema 声明顺序输出属性（`e8fd209c9`）。
- **新工具**：`tools/kv-mean-center`（K 均值标定 + `make-calib-corpus.sh`）、`tools/mtp-export`（跑 trunk 导出 MTP 头训练/验证数据：`h/tok/tgt/idx`）。
- **新 example**：`examples/rs-rollback`（`n_rs_seq>0` 快照环正确性：pass A 直生成 vs pass B 每 S 步多生成 R 再 `seq_rm` 回滚，比对 token 序列）。
- **新测试**：`test-dspark-{forward,loop,logsNR-meta,real-eval}`、`test-dfly-fusion`、`test-kv-mean-center`、`test-ptq1_0-*`、`test-pq2-row-shapes`、`test-moe-weighted-reduce`、`test-mtp-catchup-batch`。
- **发布**：`release-prism.yml`；CUDA/HIP 包里带 CPU backend 变体；SYCL Linux/Windows 包；android-arm64；macOS arm64 在 macos-26 上构建（Metal 4 tensor API）。

## 8 开关速查

**CLI**：`--kv-mean-center <bias.gguf>`（需 `-ctk q4_0`）、`--spec-draft-depth-max N`、
`--spec-type draft-dspark|draft-dflash`、`-md <drafter.gguf>`（DFly/DFlash2 由 checkpoint 自动识别）。

**环境变量（部分）**：`GGML_CUDA_GDN_L8`、`GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH`、`GGML_CUDA_FWHT_FUSION`、
`GGML_HOPPER_Q1_DISABLE`、`LLAMA_MTP_CHAIN`（及 `_GPU_EMBD/_TIGHT`）、`LLAMA_GDN_REPLAY`、`LLAMA_SYNC_TRIM`
（bit1 跳过空同步、bit2 旋转矩阵常驻设备，SYCL 默认 3）、`LLAMA_DSPARK_MARKOV_CUDA/METAL`、`LLAMA_DFLASH_LOOKUP`、
`GGML_METAL_PQ2_0_*`、`GGML_SYCL_DISABLE_XMX`。多数低比特路径默认开，GDN l8 与部分实验项默认关。

## 9 注意事项 / 风险

1. **二进制与库不可混用**：fork 改了 `ggml.h` 的 type 枚举与 op 枚举（`GGML_OP_GATED_DELTA_NET_REPLAY` 追加在末尾），
   `README` 明确禁止把本 fork 的 `ggml-*` 与官方 llama.cpp 混链。
2. **type id 私有**：`COUNT=144`，142/143 被占用、43..141 留白；若上游将来把类型 id 推到 ≥142 会撞号。ftype 侧
   `llama.h` 用 141/142/143，而 `ggml.h` 内部另有一套 128/129 映射，两套并存。
3. **落后上游 1107 提交**：最后一次 rebase 到上游是 2026-08-25（新提交 commit date 全为该日），此后再未同步；
   上游 HEAD（`31288c9c6`）与 merge-base 之间 1107 个提交（#27xxx → #30xxx）本 fork 全都没有。
   "tracks current mainline" 只是意图，不是当前事实。
4. **低比特路径大量依赖机型**：Hopper wgmma 需 sm_90a；`PTQ1_0` MMQ 排除 HIP；GDN l8 需 Ampere+ 且 state=128；
   部分条目在提交信息里标 `[UNVERIFIED - needs Apple HW]` / `[needs ARM HW]`（`73f0fed04`/`bef5fd7ea`）。
5. **`--kv-mean-center` 有基耦合**：标定与服务的 `-ctk` 和 Hadamard 旋转设置必须一致，否则拒绝加载（不静默降级）。
