# FIX_MTP_DRAFT_DEPTH —— MTP 的草稿深度没有控制器

现场（`buun-port`，V100-32GB，Qwen3.8-27B，`--kv-stream-stage-mib` + `--spec-type draft-mtp`）：

| 配置 | token 输出 |
|---|---|
| 不开 MTP、不开 kv-stream | 170+ t/s |
| **仅** MTP | **140 ~ 220 t/s** |
| **仅** kv-stream | 170+ t/s |
| MTP + kv-stream | **140 t/s** |

## 0 先分清「已证实」与「未证实」

**已证实（本条就是本文件的修复对象）**：本树没有 MTP 草稿深度控制器 ⇒ 无论
acceptance 多差都按 `n_max = 3` 全深度起草。这正是「140 ~ 220」这条**带宽**的来源，
也是 140 会落到**非推测基线 170 以下**的机制：猜输的那一步，草稿前向 + catch-up decode
全白付，而只换回 0 个 token。本文件修的是这个下界。

**未证实（不要在下一轮当成结论）**：kv-stream 是「两者同开 = 140」的原因。
`140` 落在「仅 MTP」自带的 `140 ~ 220` 区间**之内**，所以现有四个数字**无法**把
「kv-stream 拖慢 MTP」与「MTP 本来就常常落在下界」分开。§5 给出把它分开的实验；
在拿到那组数据之前，本文件的改动**不主张**解决 kv-stream 那一侧。

## 1 缺口：三个 fork 都有，上游 master 与本树都没有

| 树 | 机制 | 位置 |
|---|---|---|
| `buun-llama-cpp` | `common_speculative_mtp_adaptive`，3↔2 可逆，`GGML_MTP_DRAFT_ADAPTIVE=0` 可关 | `common/speculative-mtp-adaptive.h` + `draft_mtp` 的 5 处接线 |
| `llamacpp-v100` | acceptance EMA 驱动的 adaptive draft cap | `common/speculative.cpp`（`acc_ema`） |
| `llama-cpp-turboquant` | 独立的 `draft-mtp-adaptive` 类型 + 滞后状态机（climb/drop-pressure） | `common/speculative-adaptive.h` |
| **官方 `master`** | **无**（`git show master:common/speculative.cpp \| grep adaptive` 为空） | — |
| **本树 `buun-port`** | **无** | — |

本树自己的 `MERGE_V100.md` 早就把它登记为**刻意延后**，不是遗漏：

> **Backend-agnostic perf work** — multi-shape decode graph cache, speculative draft padding,
> server prompt-batch defragmentation and prompt-cache checkpoint pruning, **the adaptive draft
> cap**, the env-gated diagnostics, and the robustness fixes … These are explicitly *not*
> V100-specific in the fork's own README, so they belong in their own change, not this one.

本文件就是那个 "own change"。选 `buun-llama-cpp` 那一版而不是另两版：它是本树的直接祖先，
无新 CLI 旗标、只作用于 `draft_mtp`，且**自带单测**；另两版（尤其 turboquant 的
`draft-mtp-adaptive` 类型 + `--spec-draft-n-min-adaptive`）面更大，留作后续。

## 2 搬运内容

新增 `common/speculative-mtp-adaptive.h`（整文件，与 fork 逐字相同，另加来源注记）。
控制器只在**完全等于 fork 调好的那组配置**时启用，其余一律保持上游「永远全深度」：

```
n_mtp_layers == 1 && !is_mem_shared && params.n_max == 3 && GGML_MTP_DRAFT_ADAPTIVE != 0
```

（`chain_heads` 下每一步本来就在换 head，深度不是自由量；`is_mem_shared` 的助手没有独立草稿
decode 可省。`minimum_depth` 由 `--spec-draft-n-min` 定，`>= 3` 时控制器在 `accept()` 里直接
短路 ⇒ 等于关闭。）

`common/speculative.cpp` 的 5 处接线，全部带 `fork 9l-mtp-adaptive:` 标记：

| 位置 | 动作 |
|---|---|
| `begin()` | 新请求允许恢复（`adaptive[].begin()`，保留学到的深度，清请求内计数） |
| `draft()` | 起草上限改为 `n_max_eff`；收尾把**实际交付宽度**交给 `accept()` |
| `accept()` | 在 verify-row 守卫**之前**观测，让「没有 hidden row 的那一步」也算一次实测未中 |
| `set_state()` | 恢复 checkpoint 时清请求内计数（学到的深度保留） |
| `sequence_transition()` | 每个序列事件清在途宽度 + `begin()` |

## 3 顺带修掉的一处：MTP 忽略服务端的 per-sequence 上限

原 `draft()` 的停止条件是纯 `params.n_max`（= 启动参数），**完全无视 `dp.n_max`**。
服务端是为每个 slot 算过的上限：

```cpp
// tools/server/server-context.cpp
const int n_draft_max = slot.get_n_draft_max();   // min(n_ctx - prompt - 2, n_remaining - 1)
common_speculative_get_draft_params(spec.get(), slot.id) = { /* .n_max = */ n_draft_max, ... };
```

`draft_simple`（`speculative.cpp:367`）与 DFlash（`:3010`）都早已尊重它，只有 MTP 没有 ⇒
接近上下文末端时，MTP 会按全深度起草到窗口之外的位置。现在与 fork 一致：
`n_max_eff = min(params.n_max, adaptive.depth())`，再 `min(n_max_eff, dp.n_max)`。

安全性已核：服务端 `spec_i_batch` 由**实际返回**的 `spec_draft.size()` 派生
（`server-context.cpp:1630-1636`），`GGML_ASSERT(spec_i_batch.size() == n_draft + 1)`
用的也是这个实际值 ⇒ 少起草不会撞断言。

## 4 证据（本机 = Windows / MSVC，**无 NVIDIA 设备**）

| 项 | 结果 |
|---|---|
| CPU 构建（`D:\llama-build\cpu`，shared） | `test-speculative-mtp-adaptive` 13 目标，`RC_MTPADAPT=0` |
| 新单测实跑 | `OK`（exit 0）—— 决策表逐条，含 fork 原文的全部边界行 |
| 反向控制（`revctl_9l_mtp_adaptive.py`，三个单变量变异） | `threshold`/`period`/`hold` **三次全 RED**，还原后 GREEN，且 sha256 与 pristine 一致 |
| 守卫体检 | `guard_sweep.py` 见提交时的输出 |
| **真机吞吐** | **未测**。本机没有可跑的 CUDA 设备，所以「下界抬高多少」这句话本文件**不说** |

新单测：`tests/test-speculative-mtp-adaptive.cpp`（纯主机，不需要模型/后端/GPU），
`tests/CMakeLists.txt` 里无条件注册（不放进 `NOT WIN32 OR NOT BUILD_SHARED_LIBS` 那一块，
因为它不依赖未导出的 libllama 内部符号）。

## 5 把 kv-stream 那一侧分开的实验（同一次 V100 会话内做完）

**先做这一步，否则其余四步没有基准。** 同一 prompt、**同一上下文长度**、同一采样参数，
`--spec-draft-n-max 3`：

| # | 命令 | 读什么 | 若结果 ⇒ 结论 |
|---|---|---|---|
| A0 | 只 `--spec-type draft-mtp`（**不带** kv-stream） | t/s、`accepted x/y draft tokens` | 与 A1 同上下文；若 A0 ≈ 220 而 A1 ≈ 140 ⇒ kv-stream **确为因**，进 A2–A4；若两者都 ≈ 140 ⇒ kv-stream 清白，§0 的「未证实」成立 |
| A1 | 加 `--kv-stream-stage-mib N` | 同上 | 基准 |
| A2 | A1 + `GGML_CUDA_KV_STREAM_MMA_PREFILL=1` | t/s | 回升 ⇒ 元凶是**多 query 走进 vec-partial + 分块合并**（默认 `partial_count = 16`）；不变 ⇒ 不在这一格 |
| A3 | A1 + `GGML_CUDA_KV_STREAM_PARTS=2` | t/s | 同上判据的另一种切法（只取 2/4/8/16，其它值回落 16） |
| A4 | A1 + `GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS=1` | t/s | 回升 ⇒ 自适应重划分（`kv_stream_adapt` 的反馈回路）在 MTP 下选错了；MTP 每步覆盖 `1+accepted` 个 token，喂给 span tuner 的 `elapsed_ms` 与 TG1 不同源，这是它可能选错的地方 |

每次都要抓的启动/运行行：

- `phase arena decode: KV … MiB, compute … MiB, resident N pages/layer, ring M pages, tokens 4`
  —— `tokens` = `n_seq_max*(1+n_draft_max)`。若它不等于 4，`n_draft_max` 没传到目标上下文。
- `process_ubatch: phase arena reserved compute for 4 generation tokens … but the batch has N`
  —— **出现即失败**，说明 MTP 的验证批超了预留，不是慢，是算错。
- `LLAMA_KV_STREAM_TRACE=1` 的 `active/resident/ring/samples/misses/copy busy`
  —— `copy busy` 高 = 落在 PCIe 带宽侧，这时 §7 的「direct 收益被压掉」成立，别再指望内核侧。
- `selected bounded/unbounded decode spans from end-to-end latency (…)`
  —— ⚠️ 读它时记住：`fattn.cu:2067` 把上界门在 **`Q->ne[1] == 1`**。MTP 的 4-query 验证批
  **永远走 unbounded**，这个 tuner 的选择对 MTP **不生效**。所以 A4 若有效，原因在
  `kv_stream_adapt` 的重划分，不在 span 上界。

**已知的不对称**（不是结论，是下一轮读代码的入口）：`Q->ne[1] == 1` 时才启用 bounded
decode span（每 span 最多 `KV_STREAM_COPY_BATCH_PAGES = 32` 页）；多 query 走 `UINT32_MAX`
⇒ 一个 chunk-run 的整段 KV 都落进同一个 attention 调用。fork 就是这么写的（`akv` 同样如此），
本树**逐行相同**（已与 `llama.cpp-adaptive-kv-streaming` 对齐 diff），所以这不是移植遗漏。
要动它必须先有 A2/A3 的数字。

## 6 改动文件

```
common/speculative-mtp-adaptive.h            新增：fork 的 3↔2 深度控制器（整文件）
common/speculative.cpp                       draft_mtp 的 5 处接线 + 尊重 dp.n_max
tests/test-speculative-mtp-adaptive.cpp      新增：fork 原文的单测 + 生产构造路径
tests/CMakeLists.txt                         注册新测试（无条件）
```
