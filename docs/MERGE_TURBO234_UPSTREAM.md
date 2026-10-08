# turbo2/3/4 上游对齐移植（TheTom 码本/块 + buun mean-sub）：实施记录与标定方法论

日期：2026-09-23 ｜ 分支：`buun-port`
依据：`AUDIT_TURBO234_THREEWAY.md`（排序与实测）、`AUDIT_TURBO4_THETOM_VS_ATOMIC.md`（差异清单）

## 1. 本轮已落地（代码）

| 改动 | 文件 | 说明 |
|---|---|---|
| turbo4 块 → 66 B | `ggml/src/ggml-common.h` | 4-bit 变体 `norm + qs[64]`，删 `rnorm`；legacy 分支不动 |
| 删 rnorm 写入 ×2 | `ggml/src/ggml-cuda/set-rows.cu`、`ggml/src/ggml-turbo-quant.c` | 4-bit 模式不再写死字段 |
| **turbo4 码本 → Lloyd-Max** | `ggml-turbo-quant.c`（编码/解码各 1 处）、`ggml-cuda/turbo-quant.cuh` | `±0.241529 … ±0.011349`（最外 2.7326σ） |
| turbo4 中点 → LM 中点 | 同上两文件 | `±0.212203 … ±0.022824` |
| **turbo3 码本 → 精确 Lloyd-Max** | 同上两文件 | `±0.190207/±0.118786/±0.066822/±0.021663` |
| turbo3 中点 | 同上两文件 | `±0.154496/±0.092804/±0.044243` |

**一致性校验**：全树 grep 旧码本值（`0.173926 / 0.190685 / 0.154259 / 0.14556`）→ **零残留**；
`rnorm` 仅剩 legacy 3-bit+QJL 分支（`ggml-turbo-quant.c:623`）与注释，该分支的 struct 仍有此字段。
⇒ 无其它后端副本（Metal/Vulkan/SYCL 的 turbo 文件只在 buun 侧）。

**影响面**：turbo4 的 KV 行字节数 68→66 ⇒ 行步长变化，FA 实例/`ggml_row_size` 自动跟随；
⚠️ **用旧版 turbo4 量化过的 GGUF 不可读**（KV 是运行时写，不受影响）。

## 2. 待做 A：可选经验（非对称）码本 —— buun 的机制

buun 的做法（`scripts/tq_gen_candidate_books.py` + `TURBO_CB_T2/T3/T4/T8` 环境变量）：

- 脚本从**真实 27B 行**（`.f32` dump）出发，用与 GPU **同一变换**
  `ROT = diag(S1) · (Hadamard(128)/√128) · diag(S2)`（脚本里硬编码 S1/S2，与设备表一致），
  跑 `lloyd(x, n_levels, symmetric)` 重训练，产出若干候选书：
  **pooled(K+V) / K-only / V-only** 三个家族，加尺度扰动（内/外分离），加 t8 companded；
- 产出 `.f32` 候选书，运行时经 env 覆盖设备常量表；buun 的 turbo3 因此有 3 套表，
  其中一套是**非对称**的（`-0.186344…+0.188239`，用于 K 或 V 单侧）。

**移植方案**：搬脚本 + 加一个候选书加载器（`TURBO_CB_T3=<file.f32>` 等），
**编码与解码两侧同时替换**（否则同型不同书 ⇒ 系统性偏差）。默认不加载 ⇒ 行为与现在一致。

## 3. 待做 B：mean-sub —— 机制真相（K 与 V 的作用**不同**）

源码事实（buun `ggml-turbo-meansub-data.inc` 头 + `ggml-turbo-meansub.cpp` + `llama-graph.cpp`）：

- 表 = **per-(layer, channel)** 的 KV 均值 `mu = sum/cnt`，坐标 `C = kv_head·head_dim`（紧凑），
  `L = n_layer`；只存 live 层，其余层 `mu = 0`（`expand_dense` 用 `calloc` 补零）；
  注册键 = **`arch + n_layer + n_embd`**（跨微调/量化档复用）；数据面 618 KB（16 层 × 2048 通道）。
- **K**：编码时减 mu，**解码不回加**。理由（源码注释）：`<q, k_i - mu> = <q,k_i> - <q,mu>`，
  平移项只依赖 query ⇒ softmax 不变 ⇒ **注意力输出不变**；减均值的作用纯粹是**缩小 K 的动态范围
  以降低量化误差**。⚠️ 前提：**模型不能有 learned attention sink**（sink logit 独立插入，
  整体平移会改变 sink 与真实 token 的相对权重）。
- **V**：编码时减 mu，**解码必须回加**：`y = Σ w_i (v_i - mu) = Σ w_i v_i - mu·Σw_i`，
  只有 `Σ w_i = 1` 时才精确还原 ⇒ 图侧在逆 WHT 之后 `ggml_add(out, mu_v)`。

### ⚠️ alpha（`TURBO4_ALPHA`）不是质量装置，**建议不移植**

grep 全树：`d_turbo4_alpha`（默认 1.0）只在**编码侧**乘进 `blk->norm`
（`set-rows.cu:1177`、`turbo-quant-cuda.cuh:1033`），**解码侧没有任何补偿**。
⇒ 它同时缩放 logits（温度）与注意力输出，是调参/调试旋钮，不是质量器件。

## 4. 我们的实现决策：**V 走图级 affine tap（CPU/CUDA 一致）；K 走内核 tap（可选、默认关）**

| | 做法 | 为什么 |
|---|---|---|
| **V** | 图级：`cpy_v` 前 `v_cur = ggml_sub(v_cur, mu_v)`；`ggml_turbo_wht(dir=1)` 之后 `out = ggml_add(out, mu_v)` | V **无 rope**，tap 与还原都能做成与后端无关的图节点 ⇒ **CPU 与 CUDA 完全一致**（满足"只有一份实现"的要求） |
| **K** | **内核内 tap**（CUDA set-rows，默认关，`TURBO_KMEAN_SUB` 类开关） | K 存的是 **post-rope** 值，而 rope 被**融合进 cpy_k**（图上的 `k_cur` 是 pre-rope）⇒ 图级 tap 会落在错误的域（`rope(k-mu) ≠ rope(k) - const`，会真的改变注意力）。 |
| K 的 CPU 不一致 | **可证明无害** | CPU 少减一个 mu ⇒ 缓存里 K 相差一个**逐通道常数** ⇒ logits 整体平移 ⇒ softmax 输出不变。混合缓存（部分行 CPU 写、部分 CUDA 写）同样正确；差别仅是 K 的量化误差。 |

⇒ 结论：**V 侧拿满收益且严格一致；K 侧收益（=K 量化误差）只在 CUDA 路径上体现，不产生错误结果。**
若要求 K 也严格一致，唯一干净做法是**关闭 rope 融合**让 `k_cur` 成为图上 post-rope 节点
（性能代价，需要实测）。

## 5. **怎么为 Qwen3.8-27B 补一张离线标定表**（回答提问）

### 5.1 前提检查（先做，否则白跑）

| 检查 | 要求 | Qwen3.8-27B 现状 |
|---|---|---|
| learned attention sink | **必须没有**（否则 K 的 tap 不成立，需 sink-aware 契约） | 需确认 GGUF 里无 `attn_sink/sinks` 张量 |
| `head_dim % 128 == 0` | 必需（V 侧坐标契约） | 256 ✔ |
| `C = n_head_kv · head_dim ≤ max_c` | 当前 `GGML_TURBO_MEANSUB_MAX_C = 2048` | 若 n_head_kv=8 → C=2048 ✔ 正好；更多 KV 头需抬高上限 |
| 类型范围 | buun 的 tap 只挂 turbo4/turbo8 | 我们只对 turbo4 挂（观察对象） |

### 5.2 三步产出链

**第 1 步：捕获（capture）**
- 目标量：per-(layer, channel) 的 **K（post-rope）** 与 **V** 的 `sum` 与 `count`。
- 落点：**CUDA set-rows 的 turbo4 编码内核**（不能放图级——见 §4 K 的域问题）。
  加一份可选累加器（仿 `d_innerq_sq_accum` 但改成 sum、且**按层分开**：`d_kmean_sum[n_layer][C]`、
  `d_kmean_cnt[n_layer]`），在 `v = src_row[ch]` **被 tap 之前**累加原值；
  env（如 `TURBO_MEANSUB_CAPTURE=k.bin,v.bin`）+ 收尾 dump。
- 语料：**长上下文**（buun 的文件名即 `..._27b_long.bin`）。建议单条 16K–32K token、
  `-c 16384 -b 512` 跑 **1–2 遍**；混合自然语言/代码/对话。均值只需量级正确，对语料不敏感，
  但**位置覆盖**（长上下文）重要。
- 预算：一次 perplexity/生成跑即可产出（捕获本身几乎零成本，只是 atomicAdd）。

**第 2 步：生成表条目**
- `mu[l][c] = sum/cnt`；写成
  `static const float ms_<arch>_<L>l_kval[L][C]` + `g_meansub_table[]` 条目
  `{arch, n_layer, n_embd, max_l, max_c, k_live, v_live, klay, kval, vlay, vval}`。
- ⚠️ 浮点字面量精度要够（buun 用 ~8 位有效数字），否则重编会得到略不同结果。
- 可选：像 buun 一样只存 live 层（每 4 层取 1）以省体积；其余层 `mu=0` 即"不校正"。

**第 3 步：注册与验证**
- `ggml_turbo_meansub_model_id(arch, n_layer, n_embd)` 命中才启用；表为进程生命周期只读。
- 校验：同一 27B/16K 基线下跑 `--kl-divergence`，与 §10.12 面板对比。
  预期：**V 侧 tap 降低 KLD**（V 的量化误差下降）；K 侧只影响 K 的量化误差。
  若 KLD 变差 → 先查 V 的 `Σw_i = 1` 假设（sink / softmax 归一化 / 是否走了非标准 attn 路径）。

## 6. 风险与回退

- 66 B 与新码本是**格式级**变更：需要一次全量重建；旧 turbo4 GGUF 不可读。
- 码本/块改动后 `tests/test-turbo-mma-fa` 与 `tests/test-turbo-quant` 的期望值不是硬编码的
  （用 `ggml_row_size`/`ggml_quantize_chunk` 自适应），无需改测试。
- mean-sub 全程"无表 = no-op"，可安全先合入框架，标定表后补。

## 7. 进展更新（2026-09-23 晚）：V 图级 tap 已实现 + 捕获工具就绪

### 7.1 sink 判定方法（回答提问）

`attn_sinks` 的权威判据是三层的，**从弱到强**：

1. **字节级**（最快，能确认"文件里有没有"）：GGUF 的张量名是明文 UTF-8，直接
   `grep -a -c "attn_sinks" model.gguf`；或仓库自带工具
   `python gguf-py/gguf/scripts/gguf_dump.py model.gguf` 后看张量名列表（规范名
   `blk.<i>.attn_sinks.weight`，见 `src/llama-arch.cpp:663` + `src/llama-model.cpp:391` 的
   regex `blk\.\d*\.attn_sinks\.weight`）。
2. **加载日志**：未使用/未知张量会打 `W model has unused tensor <name> -- ignoring`
   （`src/llama-model-loader.cpp:1177`）。若 GGUF 有 sink 但当前 arch 不消费它，会出现在这条 W 里。
   注意这条**只能发现"有但没用"**，发现不了"有且在用"。
3. **权威判据**（决定是否真的走 sink 路径）：该模型 arch 是否创建并把它传进 attention。
   `grep -rln LLM_TENSOR_ATTN_SINKS src/models/` 当前命中：
   **`deepseek4` / `dflash` / `granite-swa` / `hy-v4` / `mimo2` / `openai-moe`**。
   **Qwen 系列（`qwen35` / `qwen35moe` / `qwen3*`）没有** ⇒ **Qwen3.8-27B 满足 K tap 的前提** ✔。

### 7.2 V 图级 tap 实现（代码已落地）

| 文件 | 改动 |
|---|---|
| `src/llama-memory.h` | 新增虚接口 `get_turbo_vmean_table()`（默认 nullptr） |
| `src/llama-kv-cache.h` | 成员 `turbo_vmean_table` + 基类 getter + context override 声明 |
| `src/llama-kv-cache.cpp` | ① `#include "ggml-turbo-meansub.h"`；② 构造时 `load_env()`（缓存比任何 set_rows 更早建立）+ 建表张量 `(width, n_layer)`（门控：V 是 turbo 型 且 width 与 `n_embd_v_gqa_eff` 一致）；③ 缓冲区就绪后按层填充；④ **`cpy_v` 内做 tap**：`v_cur = ggml_sub(v_cur, mu_view)`（mu reshape 成 `(head_dim, n_head, 1)`，与 `cpy_v` 里 `v_cur = (head_dim, n_head, n_tokens)` 的布局对齐）；⑤ context override 实现 |
| `src/llama-memory-hybrid.{h,cpp}` | 转发到 `ctx_attn`（否则混合内存下还原会静默丢失而编码已 tap ⇒ 输出错误） |
| `src/llama-graph.cpp` | `build_attn_mha` 的 **2 个逆 WHT 站点**（FA 路径的 `cur`、mul_mat 路径的 `kqv`）之后加回 `mu_v`：reshape `(head_dim, n_head, 1)` 后 `ggml_add` 广播 |

设计要点：**tap 与还原都在图级**（`cpy_v` 是 KV cache 的方法，与后端无关）⇒
**CPU 与 CUDA 对 V 的处理完全一致**（满足"不写 CPU 实现也一致"的要求）；
无表 / width 不匹配 / V 非 turbo 时全为 no-op。

### 7.3 验证状态

- **CPU：`ggml` + `llama-cli` 两次构建均 BUILD OK，0 errors**（覆盖新增 .cpp、KV cache、
  graph、memory/hybrid 的全部改动）
- **CUDA：sm_70 全量构建 BUILD OK，460/460 目标、0 errors**；`ggml-cuda.dll` /
  `llama-perplexity.exe` / `test-turbo-mma-fa.exe` 已重链；`set-rows.cu.obj`（捕获 + K tap 的
  device 侧）由正常 nvcc 编译通过 ✔
- **已提交 `d3c9c71a2`**（10 文件 558+/2−），工作区干净
- ⚠️ 过程中连续三次 `BUILD FAILED` 且**日志零错误正文**，根因是宿主环境块里重复的 PATH
  大小写变体（`Path`/`path`/`PATH`）让 nvcc 静默 exit 1 —— **与本次代码无关**。已修并记入
  `FIX_NVCC_SILENT_FAIL.md`（构建脚本内先折叠 PATH）

### 7.5 尚未验证（需 V100）

- 标定捕获链路（`TURBO_MEANSUB_CAPTURE`）与启用表后的 KLD 变化；
- V 图级 tap 的还原精度（对照 §10.12 面板，V tap 应只降不升）。

### 7.4 V100 上的捕获命令（可直接用）

```bat
set TURBO_MEANSUB_CAPTURE=64,4096,D:/qwen38_27b
.\llama-perplexity.exe -m "C:/.../Qwen3.8-27B-...gguf" -f "<16K+ 语料>" ^
   -ctk turbo4 -ctv turbo4 -c 16384 -b 512 --kl-divergence --kl-divergence-base "E:/base.logits"
:: 命中阈值后日志：tool: wrote 64-layer x N-channel means to 'D:/qwen38_27b.bin' (k=yes v=yes) -- ok
:: 然后：set TURBO_MEANSUB=D:/qwen38_27b.bin  （K 走内核 tap，V 走图级 tap+还原）
```

### §8 结论：可选经验码本**不值得做**（2026-09-23）

问题：能否明显提高 turbo3 在 Qwen3.8-27B 上的质量？

**定量上限实验**（`D:/llama-build/measure_empirical_book_gain.py`）：经验码本的经验收益 =
真实系数分布偏离 N(0,1/√128) 的程度。下表是**上限**（refit 与评测用同一批数据）：

| 系数分布 | turbo3 (8 级) 收益 | turbo4 (16 级) 收益 |
|---|---|---|
| 各向同性高斯（理论即最优） | 0.0% | 0.1% |
| 通道 σ ~ logN(0,1) | **0.9%** | 1.6% |
| 6 个主导通道 ×12 | **4.2%** | 5.8% |
| 行范数 e^N(0,1.5) | 0.0% | 0.1% |
| Student-t (df=4) | 0.1% | 0.2% |

**为什么这么小**：每行先做 L2 归一化、再过 WHT —— 旋转的**目的正是把系数高斯化**，
所以理论 Lloyd-Max 表本就接近最优。只有"少数通道极端主导"这种非高斯情形才能到 4–5%，
而那正是均值/离群通道装置（meansub / InnerQ）处理的对象。旁证：buun 自己把 tap 对 turbo8
关掉时写的原因是 *"measured zero median gain"* —— 同一类平台期现象。

**结论**：turbo3 现实预期收益 **1–4%（RMSE）**，且是 in-sample 上限 ⇒ 27B/16K 的
KLD 0.0918 → 约 0.088–0.091（在噪声量级内），**改变不了 turbo3 在 27B 的地位**
（q4_0 4.5bpv 是 0.0491）。而成本不小：需要**逐层 raw-row dump 工具**（buun 的
`{k,v}_l*_w1024.f32`）+ 生成器 + `TURBO_CB_T3` 加载器（**编解码两侧同书**）+ 一份
**与 meansub 配置耦合**的 per-model 资产（buun 的脚本是在 `raw - mu` 之后拟合的，
开/关 meansub 就得重拟合）。

⇒ **不做**。turbo4 上真正大到值得做的码本收益（理论最优 vs 我们旧表 = **2.12× MSE**）**已经在
`3a96f07f2` 里拿到了**；经验 refit 只能在这之上再加 0.1–1.6%。
