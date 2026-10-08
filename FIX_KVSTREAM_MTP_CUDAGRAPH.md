# MTP × kv-stream：CUDA graph 被禁用导致的 140 t/s 平台

> 症状：**仅开 MTP** 时吞吐从 140 t/s **爬升**到 220；**同开 MTP + kv-stream** 时稳定停在 140。
> 设备 V100，模型 Qwen3.5 2B。
> 结论：140 不是 kv-stream 的「直接拷贝开销」，而是**整个 decode 图失去 CUDA graph 重放**的价格。
> kv-stream 的图兼容门要求「一个 decode step 只写一个 token」，而 MTP 的验证批是 `n_draft + 1` 行。

---

## 1 ⚠️ 对上一条结论的更正

上一轮（`FIX_MTP_DRAFT_DEPTH.md`）我写的是：

> `140` 落在「仅 MTP」自带的 `140~220` **区间之内** ⇒ 四个数字**证明不了** kv-stream 拖慢了 MTP。

**这条更正为：不成立。** 用户澄清 `140` 不是抖动下界，而是**冷启动值**：仅 MTP 时它会**爬升**到 220。
于是 `140` = 「没有 CUDA graph 重放」的值，`220` = 「有重放」的值。kv-stream 并没有「拖慢」MTP，
它是**挡住了那次爬升**。归因对象因此是 kv-stream 的图捕获门，而不是某个每步拷贝量。

---

## 2 机制链（每一条都是可在源码里核对的事实）

### 2.1 为什么会有一次「爬升」

`src/llama-kv-cache.cpp:12700-12714`
```cpp
uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    // pad the n_kv value so that the graph remains constant across batches and can be reused
    const uint32_t n_pad_cur = std::max(n_pad, get_pad_floor());
    ...
    result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
}
```
`n_kv` 被**对齐到 pad 粒度**（默认 256），所以在一个 256 token 的窗口内 `n_kv` 不变 ⇒ `kq_mask`
的 `ne[0]` 不变（`llama-graph.cpp:34-41`、`can_reuse_kq_mask` 要求 `ne[0] == n_kv`）⇒ 图拓扑稳定。

`ggml/src/ggml-cuda/ggml-cuda.cu:5894-5902`（改前行号 5843-5851）
```cpp
if (!graph->warmup_complete) {
    // Warmup: need at least 2 calls with no property change on the 2nd call
    if (!properties_changed) { graph->warmup_complete = true; ... use_cuda_graph = true; }
    // else: properties changed or first call - execute directly (use_cuda_graph stays false)
}
```
**前两次调用 eager、之后 CUDA graph 重放** —— 这就是 `140 → 220` 的爬升形状。

### 2.2 为什么 kv-stream + MTP 升不过热身

`ggml/src/ggml-cuda/ggml-cuda.cu:3922` `ggml_cuda_graph_check_compability`（改前行号 3886）
有两处 `== 1`，都编码了「一个 decode step = 一个 token」：

| 落点 | 条件 | MTP 的取值 | 行号（改前 / 改后） |
|---|---|---|---|
| kv-stream 写 | `node_runtime->dirty_rows.size() == 1` | **4**（`n_draft+1`） | 3902 / **3942** |
| kv-stream 注意力 | `node->src[0]->ne[1] == 1` | **4**（Q 有 4 行） | 3923 / **3969** |

两处的数据来源：

- **dirty_rows**：`src/llama-kv-cache.cpp:13063-13082`
  ```cpp
  void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
      const uint32_t n_tokens = ubatch->n_tokens;   // 13064
      ...
      GGML_ASSERT(kv_stream_runtime.mark_dirty_rows_fn(kv_stream_runtime.runtime, data, n_tokens)); // 13080
  }
  ```
  `n_tokens` 就是本 ubatch 写进 KV 的行数。普通 decode = 1；**MTP 验证批 = 4**。
- **Q 行数**：MTP 让 target 一次验证 `n_draft+1` 个 token，Q 自然有 4 行。

任意一处不满足 ⇒ `use_cuda_graph = false`，**且不打印任何日志**（对比 `MUL_MAT_ID` 那条有
`GGML_LOG_DEBUG`），所以现场只有「慢」，没有「错」。

### 2.3 同一假设的另外两处落点（旁证，本次未改）

| 位置 | 内容 |
|---|---|
| `fattn.cu:2067` | `maximum_streamed_span_pages = Q->ne[1] == 1 ? graph_decode_span_pages : UINT32_MAX;` ⇒ 多 query 批永远用无界 span 合并 |
| `fattn.cu:2155` | 驻留页刷新 `... : (dst->src[0]->ne[1] > 1 \|\| chunk == nchunks - 1);` ⇒ 多 query 批在**非**精确脏跟踪时整层重传 |
| `fattn.cu:2206-2209` | MMA prefill 分支 `Q->ne[1] > 1 && Q->ne[0] == 256`：**这是为多 query 设计的**，但默认关闭，且要求 head_dim==256 |

### 2.4 环境侧确认本机能走到这条代码

- V100 = cc 7.0；`common.cuh:52 #define GGML_CUDA_CC_VOLTA 700`；`ggml-cuda.cu:5807` 只在
  `cc < VOLTA` 时禁用 ⇒ **V100 不禁用**。
- `CMakeLists.txt:169-170 set(GGML_CUDA_GRAPHS_DEFAULT ON)` ⇒ 构建默认启用；本机 `cuda70d`
  实测 `GGML_CUDA_GRAPHS:BOOL=ON`。

---

## 3 四个数字如何同时成立

| 配置 | target FA 的 Q 行数 | 图里有 kv-stream 节点 | CUDA graph | t/s |
|---|---|---|---|---|
| 不开 MTP、不开 kv-stream | 1 | 否 | ✅（热身 2 步后） | 170+ |
| **仅 MTP** | 4 | 否 | ✅ | **140 → 220** |
| 仅 kv-stream | 1 | 是（`dirty_rows=1`、`ne[1]=1`） | ✅ | 170+ |
| **MTP + kv-stream** | 4 | 是 | ❌ **恒禁用** | **140 恒** |

注意「仅 kv-stream」也拿不到 220：它没有推测解码可加速，170 就是它的天花板 —— 与它等于基线
这一点一致。kv-stream 的收益是**省显存**（KV 放 host），不是吞吐。

---

## 4 这不是移植遗漏

| 树 | 是否有流式 FA | 该门是否存在 |
|---|---|---|
| `llama.cpp`（port） | 是（`flash_attn_ext_streamed` ×7） | **是** |
| `llama.cpp-adaptive-kv-streaming`（akv，kv-stream 的来源） | 是（×4） | **逐行相同**（akv 3636-3644 ≡ port 3921-3929） |
| `buun-llama-cpp` / `llama-cpp-turboquant` / `atomic-…` / `llamacpp-v100` / `prism-llama` | **无此功能**（`maximum_streamed_span_pages` 计数 0） | 不适用 |

⇒ 那些 fork 计数为 0 是因为**根本没有 kv-stream**，不能读成「它们处理了多 query」。
⇒ 这是 kv-stream 特性的设计前提，不是 port 少搬了什么。

---

## 5 本次改动

### 5.1 诊断（默认生效，零风险）

`ggml/src/ggml-cuda/ggml-cuda.cu:3910` 新增 `kv_stream_graph_note_disabled()`：当某张图的
CUDA graph 因 kv-stream 节点被禁用时，**一次性** WARN，并带上 `op` 与行数。
只对 `rows <= 16` 的批次触发（`KV_STREAM_MULTIROW_GRAPH_PROBE_ROWS`），以免 prefill 批
把这一次性告警吃掉 —— prefill 本来就不用捕获图，那次禁用是正常的。

```
kv-stream: disabling CUDA graphs for this graph: <原因> (<op>, 4 rows) - this graph runs
without CUDA-graph replay; set GGML_CUDA_KV_STREAM_MULTIROW_GRAPH=1 to admit multi-row batches
```

### 5.2 实验性放宽（**默认关闭**）

`GGML_CUDA_KV_STREAM_MULTIROW_GRAPH=1`（`ggml-cuda.cu:3898`）解除上面两处 `== 1`：

- `dirty_rows.size() == 1` → `!dirty_rows.empty()`
- `ne[1] == 1` → `ne[1] >= 1`

**保留** `ggml_cuda_kv_stream_resident_cache_all_layers_fit()`。理由：全层驻留时所有 KV 页都
在 `resident_cache->pool_data` 里的**固定**地址上，图里不会烘进任何「运行期才选定」的页地址；
反之一旦需要 ring streaming，页地址就依赖运行期状态，捕获必然不正确。

⚠️ **这是实验开关，不是修复**。开错时它不会报错，只会**静默产出错误的 logits**。用之前必须先
比 token，不只是比 t/s。

---

## 6 V100 判别协议

### A0 —— 零改码反向控制（**必须最先做**）

`GGML_CUDA_DISABLE_GRAPHS` 的判据是 `getenv(...) != nullptr`（`common.cuh:1303-1305`），
**任何值都生效，包括 `0`**：

1. `仅 MTP，不带 kv-stream` → 记 t/s（期望 220）
2. 同一条命令加 `GGML_CUDA_DISABLE_GRAPHS=1` → 记 t/s

- 落到 ≈140 ⇒ **归因成立**：220 与 140 的差就是 CUDA graph 重放。
- 仍然是 220 ⇒ 我的归因错误，回头再查（例如 span tuner 那条路）。

### A1 —— 看诊断行

同一条 `仅 MTP，不带 kv-stream` 再加 `kv-stream`，看是否出现上面的 WARN：
- 出现且写的是 `4 rows` ⇒ 命中的正是 MTP 验证批。
- 出现但原因是「not fully resident」⇒ 先解决驻留，放宽才有意义（此时放宽无效）。
- 完全不出现 ⇒ 图其实被捕获了，问题不在这一层。

### A2 —— 放宽后 A/B

1. **先比正确性**：固定 prompt、贪心采样，比较 `kv-stream + MTP` 与 `kv-stream 关、MTP 开`
   的输出文本。**不一致就不要再往下测**。
2. 一致后再加 `GGML_CUDA_KV_STREAM_MULTIROW_GRAPH=1` 测吞吐，期望逼近 220。
3. 若正确但没提速，说明瓶颈不在 `check_compability`，转 A3。

### A3 —— 若瓶颈转移到流式 FA 内部

按嫌疑度试（均为现成开关）：

| 开关 | 作用 |
|---|---|
| `GGML_CUDA_KV_STREAM_MMA_PREFILL=1` | 多 query 的 MMA span；**仅 `head_dim==256` 生效**，2B 模型多半是 128 ⇒ 大概不适用，先确认 head_dim |
| `GGML_CUDA_KV_STREAM_PARTS=2` | 每 chunk 的 partial 数（默认见 `kv_stream_parts_per_chunk()`） |
| `GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS=1` | 固定 ring slot 数 |
| `LLAMA_KV_STREAM_TRACE=1` / `GGML_CUDA_KV_STREAM_TRACE_SPAN=1` | 观察 span 选择与数值 |

⚠️ 读 trace 时记住：`fattn.cu:2067` 的 span 上界门在 `Q->ne[1] == 1`，**MTP 的 4-query 批永远
走无界 span**，`selected bounded/unbounded decode spans` 那行对 MTP 不生效。

### 立即可用的绕行（不改码）

kv-stream 单独开只有 170（= 基线，无吞吐收益），它的价值是省显存。**若显存不紧，MTP 场景下
不要开 kv-stream，直接拿 220。**

---

## 7 未证实 / 风险

- 本机无 CUDA 设备 ⇒ 上述归属**只在代码层面闭合**，真机吞吐与放宽后的正确性**均未验证**。
- 5.2 的放宽是**设计扩展**（超出 akv 的既有前提），正确性未证；`get_mirror` 对多行会逐行做容量
  检查（`fattn.cu:828-833`），这是支持多行的迹象，但不等于已经安全。
- `KV_STREAM_MULTIROW_GRAPH_PROBE_ROWS = 16` 是个经验阈值，只影响**告警是否打印**，不影响行为。
