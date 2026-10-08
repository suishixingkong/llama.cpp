# turbo4 三方对照：TheTom 上游 vs AtomicBot-ai fork vs 我们的 master

日期：2026-09-23
对象：
- `C:\Users\confu\llama\llama-cpp-turboquant`（TheTom/llama-cpp-turboquant，`feature/turboquant-kv-cache`，HEAD `4deec5587`）
- `C:\Users\confu\llama\atomic-llama-cpp-turboquant`（AtomicBot-ai fork，`master`，HEAD `044dff242`）
- `C:\Users\confu\llama\llama.cpp`（我们的集成 `master`）

## 0. 结论

1. **TheTom 与 atomic 的 turbo4 不是同一个东西。** 核心 turbo 文件 6 个里 **5 个内容不同**，
   差异集中在**码本**与**块位宽**——正好是我们上一轮判定为“质量主因”的那一项。
2. **我们的 turbo4 与 atomic 逐字节相同**（`ggml-turbo-quant.c` / `ggml-cuda/turbo-quant.cuh` /
   `turbo-wht.cu` / `turbo-innerq.*` / `turbo-rotation-data.h` / `test-turbo-quant.c` md5 全等）
   ⇒ 我们移植的是 **atomic 那一版（旧的）**，而上游已经往前走了。
3. **TheTom 的 turbo4 质量更好**，且改进手段 = `AUDIT_TURBO4_BUUN_VS_MASTER.md` §2 预测的那一条
   （换 Lloyd-Max 码本）⇒ **上游独立验证了我们的结论**。atomic 血统的码本确实不是最优。
4. TheTom **没有** meansub / alpha / sink（那些是 buun 独有）⇒ TheTom 是“纯编解码器最优”版；
   若目标是极限质量，buun 仍多一层装置。
5. TheTom 还附送 3 项性能改动 + turbo4 MMA 路径 + 一个栈溢出修复，**都值得取**。
6. ⚠️ **两版格式不互通**：`sizeof(block_turbo4_0)` 66 vs 68，且码本不同 ⇒ 用 turbo4 量化过的
   GGUF 跨版本不可读（KV 是运行时写，无影响）。

## 1. 文件级 md5 对照（`tr -d '\r'` 归一后）

| 文件 | TheTom | atomic | 我们 master | 判定 |
|---|---|---|---|---|
| `ggml/src/ggml-turbo-quant.c` | 9054aa35 | 05fc31b0 | 05fc31b0 | TheTom 独新；= atomic |
| `ggml/src/ggml-cuda/turbo-quant.cuh` | afc80100 | a268507f | a268507f | 同上 |
| `ggml/src/ggml-cuda/turbo-wht.cu` | da86f2b5 | 13f52054 | 13f52054 | 同上 |
| `ggml/src/ggml-cuda/turbo-innerq.cu` | 1ab3a2c2 | 1ab3a2c2 | 1ab3a2c2 | **三方全同** |
| `ggml/src/ggml-cuda/turbo-innerq.cuh` | 3654e71b | 3654e71b | 3654e71b | **三方全同** |
| `src/turbo-rotation-data.h` | 5d1213d1 | 5d1213d1 | 5d1213d1 | **三方全同** |
| `tests/test-turbo-quant.c` | 275f98a0 | 778ae6b5 | 778ae6b5 | TheTom 独新 |
| `ggml/src/ggml-cuda/set-rows.cu` | 857bc638 | 52769814 | 7d84f593 | 三方各不同（我们有自己的移植改动） |
| `ggml/src/ggml-cuda/fattn-common.cuh` | c71ad790 | ac2ddcb5 | 5bf13a71 | 同上 |
| `ggml/src/ggml-cuda/fattn-mma-turbo.cuh` | 6871 B | **不存在** | 我们手写的版本 | TheTom 独有 |
| `ggml/src/ggml-common.h` | 66 B 块 | 68 B 块 | 68 B 块 | TheTom 独新 |

## 2. 差异逐条（TheTom 相对 atomic）

### 2.1 质量（主因）

| 项 | TheTom（上游） | atomic / 我们 |
|---|---|---|
| **4-bit 码本** | `−0.241529 … +0.241529`（最外级 = **2.732σ**，σ=1/√128）= N(0,1/√128) 的教科书 **Lloyd-Max** | `−0.173926 … +0.173926`（最外级 **1.968σ**） |
| 4-bit 判决中点 `TURBO_MID_4BIT` | 新表（`±0.212203, ±0.162947, …`） | 旧表（`±0.145561, ±0.103361, …`） |
| **3-bit 码本** | `0.190207/0.118786/0.066822/0.021663` | `0.190685/0.117832/0.065717/0.021460` |
| 3-bit 中点 | 随新表更新 | 旧表 |

⇒ 与 buun 的 `d_turbo_centroids_4bit`（`0.241556…`）**互相独立地收敛到同一个 Lloyd-Max 解**
（末位 0.241529 vs 0.241556 是各自重算的舍入差）。这是本次最重要的发现。

### 2.2 体积 / 格式

| 项 | TheTom | atomic / 我们 |
|---|---|---|
| `block_turbo4_0`（4-bit） | `norm + qs[64]` = **66 B = 4.125 bpv**（注释里写着“dropped dead rnorm”，但块上方那段说明文字还是旧的 68 B 描述） | `norm + rnorm + qs[64]` = 68 B = 4.25 bpv |
| 编码器收尾 | 不写 rnorm | `y[block].rnorm = 0` / `blk->rnorm = 0` |
| legacy 3-bit+QJL 分支 | 不变（仍 68 B） | 不变 |

### 2.3 性能（数值等价，已抽样校验）

| 项 | TheTom | atomic / 我们 |
|---|---|---|
| `TURBO_WHT_SIGNBITS1/2` | 新增 **打包符号位**（各 4×uint32；element e → word `e>>5` 的 bit `e&31`，1 = −1.0f），避免每 lane 读 32 个常量地址被常量缓存串行化 | 无 |
| `turbo-wht.cu` 蝶形 | 前 5 级用 **warp shuffle**（`__shfl_xor_sync`），只有 h=32/64 走 smem ⇒ `__syncthreads` 从 **7 次降到 2 次** | 每级一次 `__syncthreads` |
| `fattn-common.cuh` turbo4 V 解码 | 新增 **`ne == 8`** 快路径（4 个 qs 字节一次解 8 元，写 4×half2/float2） | 只有 `ne ≤ 4` |
| `ggml_cuda_pdl_sync()` | 有（PDL 依赖启动） | 无 |

**校验**：用 `D:\llama-build\verify_signbits.py` 从 atomic 的 `TURBO_WHT_SIGNS1/2` 按 TheTom 的打包
约定重算位模式，与 TheTom 的 `SIGNBITS1/2` 常量逐字比对 ⇒ **MATCH**（两张表都对）。所以 signbits 是
纯粹的性能改动，不是数值改动。

### 2.4 功能 / 稳定性

| 项 | TheTom | atomic / 我们 |
|---|---|---|
| `fattn-mma-turbo.cuh` | **已有**：turbo4-only（`type_K/type_V = TURBO4_0`）、`nstages=0`、`need_f16=false`、
注释明确写“Q 已在 graph 层旋转、输出在 graph 层逆旋转，本路径不做 inline FWHT、不做 src swap”
+ **14 个实例**（`fattn-mma-turbo-instance-dkq{128,256}-ncols1_A-ncols2_B.cu`，文件名不含类型） | **无**（我们手写了一个同设计的版本，覆盖 20 对） |
| `turbo_init_rotation` | 直接生成进 `turbo_rotation`，**删掉了 64 KB 栈局部 `G[]`**（注释：worker 线程栈小会溢出） | 仍用栈局部 `G[]` + `memcpy` |
| `turbo3_cpu_wht_group_size` | 声明（`GGML_API`）与定义分离（`-Wextern-initializer` 在 `-Werror` 下报错） | 合并成一行 |
| meansub / alpha / sink | **没有** | 没有（buun 才有） |
| CPU 4-bit 旋转 | `turbo_cpu_fwht`（与 CUDA 同域） | `turbo_cpu_fwht`（同） | 

> 注：`matvec(turbo_rotation_t, …)` 在 TheTom 的 CPU 里只出现在 **legacy 3-bit+QJL** 分支
> （第 530/635 行），4-bit 路径仍是 FWHT ⇒ 不存在 buun 那种 CPU/CUDA 不同域的问题。

## 3. 行动建议（按性价比排序）

1. **先只搬码本 + 中点**（最小、可立即验证）：`ggml-turbo-quant.c` 的 `CENTROIDS_4BIT`（编码/解码
   两处）+ `nearest_centroid_4bit` 中点；`/ggml-cuda/turbo-quant.cuh` 的 `TURBO_CENTROIDS_4BIT`
   + `TURBO_MID_4BIT`；3-bit 两处同理。**不动块结构**。→ 上 V100 跑 27B/16K KLD，预期
   turbo4 从 **0.0769 → ≈0.04**（反超 q4_0 的 0.0491）。
2. 再取体积/性能四项：66 B 块（连带删 `rnorm=0` 写入）、`SIGNBITS`、warp-shuffle 蝶形、
   `ne==8` 解码快路径。
3. 用 TheTom 的 `fattn-mma-turbo.cuh` 与我们的手写版做一次逐行对比，取更合适的
   （它只覆盖 turbo4；我们的覆盖 20 对，功能面更宽）。
4. `turbo_init_rotation` 栈溢出修复独立成一个提交（与质量无关，但对我们启动 multi-thread 时
   崩溃的排查有价值）。
5. ⚠️ 接受 66 B/新码本意味着：**此前用 atomic 版 turbo4 量化出的任何 GGUF 都要重做**。

## 4. 局限

- 本次只做了静态对照（md5 + 逐 hunk 语义），**没有跑数值**；§3.1 的 KLD 预期来自
  `AUDIT_TURBO4_BUUN_VS_MASTER.md` §2 的受控 RMSE 实验，需 V100 实测确认。
- 未核对 TheTom 相对 atomic 的 **非 turbo4** 差异（turbo8 / TCQ / turbo2）与上游合并面
  （`TURBOQUANT_UPSTREAM_MERGE.md` 未读）。
