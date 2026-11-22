# KVMem — integration notes

Port of the KVMem tiered KV-cache memory module from the `kvmem-llama.cpp` fork
into this tree, on top of the VBR / KV-stream / turboquant integrations already
present. KVMem replaces `llama_kv_cache` with a **fixed slot pool** whose blocks
can be scored, evicted to a host/SSD tier, and resurrected, so the GPU working
set can be smaller than the context.

| source | branch in this repo | upstream base | what was taken |
|---|---|---|---|
| `kvmem-llama.cpp` (fork of llama.cpp) @ `19f4ee6` (merge PR #169) | `tmp/kvmem-port` | `4dab0cf0c` (= this repo's `master`) | the whole memory module, vendored under `third_party/kvmem` |

The fork's own tree also carries three port-side fixes (GDN replay compiled out,
capacity-gated capture, mask-view sampler). They were developed on the fork
first, then moved here; the fork is **clean again** (`19f4ee6`) and no longer
participates in the build. The patches are kept outside the tree for reference.

```
4dab0cf0c  master
  └─ 82242e7a5  kvmem: port the hook layer onto master, replay dropped
       ├─ 7a9f12df4  kvmem: keep the hook layer inert when --kvmem is off
       ├─ b2cb6fc4a  server: run a KVMem retrieval round only while prefill capture is live
       ├─ 60e6489fb  kvmem: vendor the adapter and libkvmem so the port stops building out of the fork
       ├─ 94f780e9e  kvmem: stop waiting on the device between prefill ubatches
       ├─ 471930f69  kvmem: make removals and the working-set budget follow the store
       ├─ b1f1f5c50  common: stop the memory-mode resolver from repeating itself …
       └─ 447d43a00  kvmem: benchmark the adapter, build it only where it can link …
```

## 编译

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_KVMEM=ON
cmake --build build --config Release
```

| 开关 | 默认 | 说明 |
|---|---|---|
| `LLAMA_KVMEM` | `ON` | 关掉（`-DLLAMA_KVMEM=OFF`）时本分支与 `master` 逐位等价（连编译产物里的钩子都不存在），`--kvmem` 会被参数解析层直接拒绝 |
| `GGML_CUDA=OFF` | — | adapter 每个 tier 都要 CUDA runtime，所以纯 CPU 构建会把 `LLAMA_KVMEM` 自动降回 `OFF` 并打印一行说明，而不是链接失败 |
| `LLAMA_KVMEM_ROOT` | `<src>/third_party/kvmem` | adapter + libkvmem 的根（含 `src/adapter` 与 `kvmem` 两个子目录）。可指向外部 kvmem 树做对拍 |
| `GGML_CUDA` | `OFF` | GPU 池必须开 CUDA（`--kvmem` 需要 attention KV 落在 GPU 上）；只有 `-dev none` 的纯 CPU 上下文能用 host 池 |

`third_party/kvmem` 布局与 fork 一致，便于和上游 fork 逐文件对拍：

```
third_party/kvmem/src/adapter/   22 个文件，编进 libllama（钩子层 + 采样器 + 捕获 + stage-in CUDA kernel）
third_party/kvmem/kvmem/         libkvmem（host store / runtime / recall / 单元测试）
```

测试目标跟随本仓命名：`kvmem_store_test` → `test-kvmem-store`，`spill_io_test` → `test-spill-io`（共 16 个）。
`spill_io_test` 在 Windows 共享库构建下改为把 core 源直接编进测试：它替换 `operator new` 注入分配失败，而 PE DLL 不会为可执行文件的替换做符号插入，挂在 DLL 上的注入根本不会触发。

### Web UI 资源（`/` 返回 404 时先看这里）

`--ui`（默认开）走的是**内嵌资源表**（`tools/ui` 的 SvelteKit 构建产物），不是磁盘目录。资源在构建时由 `scripts/ui-assets.cmake` 按优先级 provision：`tools/ui/dist` → npm build（`-DLLAMA_BUILD_UI=ON`）→ HF 预构建包（`LLAMA_USE_PREBUILT_UI=ON`）。三者都拿不到时只打一行 WARNING 就继续编译，生成的 `ui.cpp` 是空表、`LLAMA_UI_HAS_ASSETS` 不定义 ⇒ 运行时 `/` 与 `/index.html` 根本没注册路由，浏览器拿到：

```json
{"error":{"message":"File Not Found","type":"not_found_error","code":404}}
```

判据：`wc -c build/tools/ui/ui.cpp` 只有几百字节就是空的（正常 ≈ 15 MB，可 `grep -c asset_0` 看到 gzip 字节表）。离线环境把任意 llama.cpp 发布包/自带树里的 UI 解到 `tools/ui/dist` 再重建即可；`dist` 在 `tools/ui/.gitignore` 里，不进仓库。

## 移植内容

`git diff master` 面：**39 个文件 +1291 / −93**（不含 vendored 副本）。

| 层 | 文件 | 作用 |
|---|---|---|
| ABI | `include/llama.h` | `llama_kvmem_*` 导出面（开关、参数、检索轮、捕获） |
| 构建 | `src/CMakeLists.txt`、`common/CMakeLists.txt`、`ggml/src/ggml-hip/CMakeLists.txt` | adapter 进 `libllama`、`LLAMA_KVMEM_ROOT` 处理、HIP 侧的 fork 目录名兼容 |
| 工厂 | `src/llama-kvmem-factory.h`、`src/llama-model.cpp` | `create_memory()` 先问 KVMem，未启用返回 `nullptr` 后走原路径 |
| batch | `src/llama-batch.{h,cpp}` | `logical_pos`（逻辑位置 ≠ 物理 pos，池化后用于检索与驱逐） |
| KV cache | `src/llama-kv-cache.{h,cpp}`、`src/llama-kv-cells.h` | cell 的 `ext.logical_pos`、`seq_rm_logical()`、`get_v_storage()`；**空洞容忍**的 KQ mask 与 purge 门控（按 `--kvmem` 开/关分开，关时与 master 逐位一致） |
| graph | `src/llama-graph.{h,cpp}` | `kvmem_capture_q/k/v()` 侧信道捕获、`build_attn_mha` 的注意力采样钩子、复用判定加入 `ubatch.n_pos` |
| context | `src/llama-context.{h,cpp}` | 固定宽度 decode 图槽（1..8 token）、捕获戳、`reset_cached_graphs()`、`restore_graph_srcs()` 调用点 |
| 模型 | `src/models/qwen3.cpp`、`src/models/qwen35.cpp` | 每层 Q/K/V 捕获调用点 |
| ggml | `ggml-backend.cpp`、`ggml-backend-meta.cpp`、`getrows.cu`、`ggml-cpu/ops.cpp` | 图输入写回表（`ggml_backend_sched_restore_graph_srcs`）、fork 命名的 meta 转发、空 index 的 `get_rows` 早退、GDN `K=0` 校验 |
| 驱动 | `common/common.h`、`common/common.cpp`、`common/arg.cpp` | `--kvmem-*` 全套参数、三种 KV 模式的仲裁、`llama_kvmem_set_params()` 的进程级装配 |
| server | `tools/server/*` | 每请求 Query span、只在 prefill 捕获存活时跑检索轮、检索轮落地后按最终页集重算 Query 段（`KVMEM_REPLAY=0` 可关，用于 A/B） |
| conversation store | `tools/server/kvmem-conversation-store.h`、`tools/server/server-context.cpp` | `--kvmem-conversations N` 的 slot↔store 映射：切走时 drain 到 host、切回时 restage 并命中前缀；请求字段 `kvmem_conversation` 绑定 |
| 多模态 | `tools/mtmd/mtmd-helper.*` | mmproj 与 kvmem 共存的上下文创建路径 |
| 测试 | `tests/test-backend-ops.cpp` | KVMem 相关后端用例的形态登记 |

**刻意排除**（fork 有、本分支不带）：

| 排除项 | 原因 |
|---|---|
| GDN / MTP replay（`replay_*` 记录面 + fold kernel） | 本分支的 `llama_memory_recurrent` 只有上游 `n_rs_seq` 快照；`--kvmem-mtp-state replay` 被参数层直接拒绝，不用 `snapshots`/`auto` 也能跑 |
| SWA / 纯 recurrent 架构 | 池化 + 滑动窗口 / 递归状态没有一致的块语义；`maybe_create()` 直接返回 `nullptr` 并告警，回落到原生 cache |

⚠️ 这里原本还有一行「query replay：本分支在 prefill 末尾一次性 `apply_retrieval → pin`，数值上无损失」。
**那句是错的**：24 起点矩阵上它值 13–18 个百分点（`KVMEM_REPLAY=0` 59.7% → 开 80.6%，11-22 复测；11-15 是 62.5% → 80.6%），已在本轮补齐，见
「Query replay：检索轮之后要按最终页集重算 Query」。不补的话，采样读到的 logits 属于一个已经被
检索轮裁掉的历史视图。

## 使用参数

所有参数都由 `common` 解析，因此 **`llama-completion` / `llama-server` / `llama-perplexity` 等所有基于 common 的工具都支持**。
`llama-bench` 自建 cparams，不经过 common，所以它单独实现了 `--kvmem` / `--kvmem-budget` / `--kvmem-method` / `--kvmem-block-tokens`（其余 `--kvmem-*` 仍然只在 common 工具里）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--kvmem` | 关 | 打开 KVMem 池。要求 `n_seq_max == 1`（即 `--parallel 1`），否则告警并回落到 VBR / 原生 cache |
| `--kvmem-block-tokens N` | 0 → 32 | 块大小（token）。选择、驱逐、检索都以块为粒度 |
| `--kvmem-conversations N` | 0 → 所有请求共享一个 store | **仅 server**。保留 N 个 park 在 host 的会话 store，配合请求字段 `kvmem_conversation`（相同的 id 共用一个 store；留空则按 token 前缀匹配，且 store 比请求多出的尾巴必须 ≤ 上一轮生成数 + 64 行才算同一段对话，否则视为分支、另起 store）。切 store 需要空闲上下文，所以 `--parallel > 1` 时告警并保持单 store 行为 |
| `--kvmem-budget N` | 不给 → 自适应 | GPU 常驻工作集（token）。**给了数字就固定用它**，同时关掉自适应；`0` 配合 `--no-kvmem-budget-auto` 才是「全常驻、不驱逐」，此时与原生 cache 数值逐位相同。下限是**一次 decode batch**（`-b`）加恒留带：整批 token 在 ubatch 之前就一次性登记，批内不能驱逐，给得比这小会被自动抬到下限并告警一次 |
| `--kvmem-budget-auto` / `--no-kvmem-budget-auto` | **开** | 不给 `--kvmem-budget` 时，工作集 = `min(设备装得下的块, n_ctx × ratio)`。判据日志：`KVMem budget auto: n_ctx=... ratio=... device_cap_tokens=... -> budget=... (N blocks, X% of ctx)` |
| `--kvmem-budget-ratio X` | 0.33 | 自适应时工作集占 `n_ctx` 的最大份额。1/3 是实测召回 96% 的点；1/6 → 79%、1/8 → 58%，不要往下调 |
| `--kvmem-gen-reserve N` | **1024**（写 0 取同一默认） | 给 decode 预留的额外 GPU token，与 fork 对齐（port 早期是 256）。⚠️ **GPU 上真正分配的池 = `budget + gen_reserve`**（按块对齐，再受设备 KV 容量约束），所以 `--kvmem-budget 1000` 实际是 `1000 + 1024 = 2024` token ≈ 63 块，而 `budget` 只是其中「可给历史用」的那部分 |
| `--kvmem-sink-tokens N` | 0 → 一个块 | 永久保留的前缀（attention sink） |
| `--kvmem-recent-tokens N` | 0 → 压力下保留最新尾部 | 永久保留的后缀 token 数 |
| `--kvmem-method {recency,retrieval}` | `retrieval` | `recency` 只按新近度选块（不采样、不评分，最省）；`retrieval` 用 Query 注意力给每块打分 |
| `--kvmem-query-scorer {page-softmax,meanq-cosine}` | `page-softmax` | 检索评分器。`page-softmax` 逐行 softmax 后聚合；`meanq-cosine` 用平均 Query 向量做余弦 |
| `--kvmem-score-agg {mean,max}` | 评分器默认 | 行级分数聚合成块分数的方式 |
| `--kvmem-coarse-threshold N` | **0（关）** | 被打分上下文达到 N token 时改走两阶段：先用少量层给所有块粗排，再对 `max(64, 4×budget_blocks)` 个候选做精确全层打分。`0` = 永远全量精确（唯一没有召回风险的形态，**也是默认**）。实测只值整轮 0.3%，要省这点时间才打开。判据日志 `KVMEM_QUERY ... coarse=1 threshold=... kept=... layers=...`。详见「两阶段打分」 |
| `--kvmem-lrfu` / `--no-kvmem-lrfu` | 评分器默认 | 是否把历史命中率（LRFU）混进块分数 |
| `--kvmem-lrfu-lambda X` | 0.5 | 历史衰减，`[0,1]` |
| `--kvmem-lrfu-top-frac X` | 0.9 | 视为「命中」的历史分位，`(0,1]` |
| `--kvmem-graph-alpha X` | 评分器默认（0.2 / 0） | 图召回项的权重 |
| `--kvmem-query-begin N` / `--kvmem-query-end N` | −1 | 显式声明 Query 的原始 token 区间。CLI 直接透传；server 每请求覆盖 |
| `--kvmem-query-max N` | 0（不裁剪） | **仅 server 请求路径**。Query 区间超过 N 时只评分**尾部 N 行**。评分约 0.13 ms/行：9.5k prompt 裁到 512 行把 prefill 从 3.27 s 压到 1.29 s，但⚠️**这是换问题不是省钱**——1.9k 的文档裁到 512 行后，模型丢了开头段落、答不出原本能答的细节。默认不裁剪；只在「prompt 远长于答案需要召回的内容」时开 |
| `--kvmem-force-pos N` | −1 | 强制钉住某个 token 所在的块 |
| `--kvmem-query-workspace {on,off}` | `on` | Query 自己的尾部块算**追加工作区**（追加在语义预算之外，历史拿满预算）还是算历史（与 sink/recent 抢同一份槽）。fork 的 replay 是前者；`off` 复现移植早期的记账（同预算下少 2 块可召回历史）。见「Query 行是追加工作区」 |
| `--kvmem-mtp-state {snapshots,auto}` | `snapshots` | MTP 回滚状态；`replay` 在本分支不可用（见排除项） |
| `--kvmem-gpu-ratio R` | 0 → 设备的 0.50 | 池占显存的比例（与绝对值预算二选一，由 fit 解析） |
| `--kvmem-gpu-high R` / `--kvmem-gpu-low R` | 0.95 / 0.85 | 换出高水位 / 换出后目标低水位 |
| `--kvmem-host-mib N` | 16384 | 共享原生 KV 的 host RAM 配额（MiB） |
| `--kvmem-disk-mib N` + `--kvmem-disk-path PATH` | 0 / 空 | SSD 层配额与落盘目录；两个都给才生效 |
| `--kvmem-cpu-gb G` | 0 | 独立的 CPU spill arena（与 SSD 层互斥） |
| `--kvmem-harvest-v` / `--no-kvmem-harvest-v` | 关 | prefill 时把 V 也镜像到 host，换出/回填时省一次设备读取 |

典型用法：

```bash
# 质量最优（全常驻，不驱逐）：与原生 cache 数值逐位相同，只多一层池化
llama-completion -m model.gguf -c 4096 --kvmem -ctk f16 -ctv f16 -f prompt.txt

# 显存换上下文：把 4k 上下文压到 2k 工作集
llama-completion -m model.gguf -c 4096 --kvmem --kvmem-budget 2048 -ub 512 -ctk f16 -ctv f16 -f prompt.txt

# 只要新近度（不采样注意力，最省）
llama-completion -m model.gguf -c 32768 --kvmem --kvmem-budget 8192 --kvmem-method recency -f prompt.txt

# 服务端：Query span 由每个请求自己带
llama-server -m model.gguf -c 32768 --parallel 1 --kvmem --kvmem-budget 8192

# 基准：真驱逐（工作集 576 token，-b 512 才让 4k 前填跨越多个批）
llama-bench -m model.gguf -p 4096 -n 128 -b 512 --kvmem --kvmem-budget 576
```

### 自适应工作集（`--kvmem-budget-auto`，默认开）

不给 `--kvmem-budget` 时，工作集取两边较小值：

- **设备装得下多少**：`cap_blocks`（已扣权重、`gen_reserve` 和调度器图，取剩余显存的 `--kvmem-gpu-ratio`，默认 50%）× 块大小；
- **值得装多少**：`n_ctx × --kvmem-budget-ratio`（默认 1/3）。

再夹进 `n_ctx`，并抬到下限（恒留带 + 一次 decode batch）。启动时记一行，可直接判读：

```
KVMem budget auto: n_ctx=8192 ratio=0.33 device_cap_tokens=699008 gen_reserve=1024 -> budget=2720 (85 blocks, 33.2% of ctx)
```

- 显式给 `--kvmem-budget N` ⇒ 立刻关闭自适应，打出来的数就是它（低于下限仍会被抬起并告警一次）。
- 要回到「全常驻、与原生 cache 逐位相同」的老行为：`--kvmem-budget 0 --no-kvmem-budget-auto`。
- 这条只决定 **GPU 常驻量**，不改选择策略；预算比与失忆率的关系见下文「检索质量」。

### 两阶段打分（`--kvmem-coarse-threshold`，**默认 0 = 关**）

一轮检索要给**每个**块算一次分，每次都要读一遍 mean-K 镜像（每块 × 每个参与层）。上下文一长，
这一步就成了整轮里最大的单项，而花掉的绝大部分算力落在**选择根本不会取**的块上。

`--kvmem-coarse-threshold N`（单位 token，作用在**被打分的 frontier** 上）：达到 N 时，这一轮先
用**少量层**（参与层里等距抽 1–4 个）给所有块赔一个粗分，取前 `max(64, 4 × budget_blocks)` 块作为
候选（`mandatory` 那批无条件入选），再只对这些候选做**精确全层**打分。

**默认是 `0`，即永远全量精确**：只有这个形态没有召回风险。下面两组实测说明为什么把开关合上——
省下来的绝对时间是 10~18 ms，而它所在的检索轮只有整个 prefill 的 1% 上下。

判据（Qwen3.5-2B，40 k 字符 ≈ 15 986 token、`-c 32768 --kvmem-budget 1024`、`KVMEM_PERF=1`）：

| | `score_ms` | 整轮 `total_ms` | 选块 |
|---|---|---|---|
| `--kvmem-coarse-threshold 0`（全量精确） | 21.16 | 33.57 | 33 块 |
| `--kvmem-coarse-threshold 1024`（两阶段） | **13.19** | **26.19** | 33 块，其中 32 块与上一行逐位相同 |

**32k 以上的复测（默认阈值）也一样，而且这才是它真正的量级**。2B、90 k 字符 = **32 806 token / 1027 块**、
`-c 65536 --kvmem-budget 2048`、`KVMEM_PERF=1`：

| | `score_ms` | 整轮 `total_ms` | 选块 | 该轮 prefill |
|---|---|---|---|---|
| `--kvmem-coarse-threshold 0` | 37.72 | 58.69 | 65 块 | 5 788 ms / 32 806 token |
| `--kvmem-coarse-threshold 32768` | **20.50（−46%）** | **40.73（−31%）** | 65 块，其中 **64 块相同**（唯一的差别是 333 换掉 638） | 同上 |

⚠️ **必须同时看最后一列**：评分只是这一轮里的一项，整轮 40–59 ms 对 5.8 s 的 prefill 就是 **0.7–1.0%**，
所以「省 46% 评分」等于**整轮省 0.3%**。上下文再长也是这个比例（块数与 prefill 同步缩放）。结论：这是
**微优化**，不是「128k 才划算」的大项；默认 32768 可以留着，想要零召回风险就把阈值设 0。

日志多出三个字段，可直接看出走了哪条路：

```
KVMEM_QUERY scorer=page-softmax agg=mean lrfu=0 graph_alpha=0.2 rows=31 blocks=501 coarse=1 threshold=1024 kept=128 layers=1
```

- `coarse=0` ⇒ 这一轮全量精确（默认在 32k 以下都是这个）。
- `kept` 是候选块数；`kept >= blocks` 时自动退回全量精确。
- ⚠️ **粗筛是近似**：129 块、预算 512 的小样本上实测 19 个选块里有 4 个不同（约 20%）。
  块数越少、候选带越窄，分歧越大；因此阈值以下不给这条路径，且 `kept` 至少 64 块。
- ⚠️ 未被选进候选的块得分为 0，LRFU 的反馈也就看不到它们；默认 LRFU 关，不受影响。


### reserve 图必须包含捕获节点

`--kvmem` 上下文退出时曾打印：

```
CUDA0 compute buffer size of 179.2345 MiB, does not match expectation of 149.2346 MiB
```

不是泄漏。reserve 用的图当时不含捕获节点：适配器把 pre-RoPE 的 Q/K 标成 `ggml_set_output`（harvest 要在整个 ubatch 之后读它们），而 output 张量不能与后续层复用内存，于是**首个真实 ubatch** 时调度器必须扩容 —— 可 `backend_buf_exp_size` 早在 reserve 时就记下了较小的数。

现在 `graph_reserve()` 全程持有 `llama_kvmem_set_reserve_probe()`：捕获钩子无条件建节点，但**不登记**（reserve 张量绝不能进真实捕获）。同一请求下 2B 的 CUDA0 compute buffer 从 55.16 MiB 变成 89.02 MiB，而图节点数仍是 1179 —— 多出来的是 output 标记带来的内存不可复用。运行期不再出现第二次 `sched_reserve`，**峰值不变**：那 34 MiB 以前是在运行期才分配的，现在提前到 reserve 阶段。

## 环境变量

**参数等价物**：每个 `--kvmem-*` 都有对应的 `LLAMA_ARG_KVMEM_*`（llama.cpp 的通用机制），例如 `LLAMA_ARG_KVMEM=1`、`LLAMA_ARG_KVMEM_BUDGET=2048`、`LLAMA_ARG_KVMEM_METHOD=recency`。完整集合：
`LLAMA_ARG_KVMEM`、`_BLOCK_TOKENS`、`_BUDGET`、`_BUDGET_AUTO`、`_BUDGET_RATIO`、`_GEN_RESERVE`、`_SINK_TOKENS`、`_RECENT_TOKENS`、`_METHOD`、`_QUERY_SCORER`、`_SCORE_AGG`、`_LRFU`、`_LRFU_LAMBDA`、`_LRFU_TOP_FRAC`、`_GRAPH_ALPHA`、`_QUERY_BEGIN`、`_QUERY_END`、`_FORCE_POS`、`_MTP_STATE`、`_GPU_RATIO`、`_GPU_HIGH`、`_GPU_LOW`、`_HOST_MIB`、`_DISK_MIB`、`_DISK_PATH`、`_CPU_GB`、`_HARVEST_V`。

**诊断/调试开关**（都只影响日志或诊断，不影响数值）：

| 变量 | 作用 |
|---|---|
| `KVMEM_TRACE=1` | 打开 `KVMEM_*` 事件日志。**判断「是否真的启用」看 `KVMEM_KV_BYTES bytes=… cells=… slots=… budget=…`**，没有这行就是被静默降级了 |
| `KVMEM_REPLAY=0` | 关掉 Query replay（A/B 用）；默认开 |
| `KVMEM_REPLAY_FIT=0` | 重放不再让出模板尾（回到 512+7 两个 batch，A/B 用）；默认开 |
| `KVMEM_PERF=1` | 每 ubatch 的分相计时（捕获/评分/换出/D2H/HARVEST 汇总） |
| `KVMEM_HOST_PERF=1` | host 块分配的统计：次数/字节、`new` 与 `memcpy` 分相耗时、arena 的 hits/misses/dropped。退出时打到 stderr（这条走 `stderr` 而不是 `LLAMA_LOG_*`，因为它在 CRT 静态析构期执行） |
| `KVMEM_HOST_ARENA_MB=N` | host 块缓存的预算，默认 `min(host 配额/8, 128 MiB)`；`0` 关闭。只影响性能，不影响数值 |
| `KVMEM_ATTENTION_VERIFY=1` | 采样器对 FP64 oracle 校验（prefill 与 output 两路），用于改取数方式后的正确性判定 |
| `KVMEM_DUMP_CAPTURE=1` | 把捕获到的 Q/K/V 落盘 |
| `KVMEM_DECODE_GRAPH_SLOTS=0` | 关闭固定宽度 decode 图槽（默认 1..8 token 各一槽），用于排查图复用问题 |
| `KVMEM_MULTI_D2H_BATCH=0` | 关闭多卡批量 D2H 提交 |
| `KVMEM_HARVEST_SYNC` / `KVMEM_GDN_FOLD_SYNC` | 强制 harvest / fold 走同步路径（排查竞态用） |

## 三种 KV 模式的仲裁

`common_context_params_to_llama()` 末尾统一裁决，顺序是 **KVMem > KV stream > VBR > 原生 cache**：

| 条件 | 结果 |
|---|---|
| `--kvmem` 且 `n_seq_max == 1` | KVMem 生效；若同时给了 `--kv-stream-arena-mib` 或 VBR 预算，二者被清零并告警（`KVMem outranks …`） |
| `--kvmem` 但 `n_seq_max > 1` | 告警并回落到 VBR / 原生 cache（KVMem 与 KV stream 都需要单序列） |
| `--kvmem` + `-ctk/-ctv turbo*` | **turbo 是 KVMem 的一等 codec**，两种模式同时生效，类型不变（见下节）。KVMem 与 turbo 都不支持的布局才换成 f16，且由 adapter 决定、KVMem 继续跑 |
| `--kvmem-budget` < 一次 decode batch | 自动抬到 `sink + recent + batch + block` 并告警一次；否则首个 prefill 会在 `mandatory Query content exceeds its selection budget` 处失败（`--kvmem-budget 2048` 只在 `-b ≤ 1984` 时真的是 2048） |
| 同一条拒绝被打印两次 | 已去重：cparams 每进程会构建 2~3 次（context、fit、draft），`common_resolve_memory_modes` 的连续的相同拒绝现在只打印一行 |
| 模型是 SWA / 纯 recurrent | adapter 侧拒绝，回落原生 cache |

⚠️ **KV 口径**：不传 `-ctk/-ctv` 时默认是隐式 VBR 动态档，而 `--kvmem` 会顶掉 VBR 走静态 f16 池。**比较 `--kvmem` 开/关必须显式 `-ctk f16 -ctv f16`**，否则比的是 KV 类型而不是代码。

### turbo 作为一等 codec

turbo 是 **编码类型**，不是动态分级策略：`-ctk turbo4` 不 arm VBR 阶梯（`--kv-stream-arena-mib` 清掉阶梯后 turbo 照跑）。但它**仍会建一个 VBR 记账池**（`needs_turbo_layout` 只看类型），该池要用 `compute_backend_for_buft` 解析 compute backend —— 这正是 `--kvmem` 以前在 cache 构造处报 `VBR could not bind KV buffer type CUDA0` 的真实原因（adapter 自建 cache 时传的是默认空 `llama_memory_vbr_params`），不是"turbo 需要动态 VBR"。

让两者共存需要四件事，均已落地：

| 环节 | 做法 |
|---|---|
| cache 构造 | 传入静态 `llama_memory_vbr_params`（`dynamic=false`，只带 codec 与 `compute_backend_for_buft`），hybrid 的基类与 MTP 跟随缓存同样传 |
| 行寻址 | turbo 把每个 head 向量 pad 到 128（`n_embd_k_gqa_eff`），adapter 用 `kvmem_store_width()` 得到**存储宽度**算所有 cache 行（行字节、块字节、池容量、MTP 布局校验），逻辑宽度只留给图上采到的 Q/raw-K |
| 评分 | 块评分读的是 raw store 里的 mean-K 与采到的 Q（都在 RoPE/旋转之前、f16），与 cache 类型无关；**精确注意力采样器**要拿 cache 张量做 `mul_mat`，turbo 没有 mat-vec kernel（`mmvq` 类型 switch 直接 abort）⇒ turbo 下改用 **mean-Q cosine** 评分并跳过采样（`KVMEM_QUERY scorer=meanq-cosine`），驱逐/回填不受影响 |
| 回填 | 打包块是 cache 行的逐字节副本，turbo 行可直接搬回；`kvmem_attn_rot_on()` 对 turbo 返回 false（编码器内部已做 WHT，再叠一次会转两次） |

⚠️ 因此 turbo 模式下**评分器是近似的**（均值余弦而非精确注意力质量），驱逐策略与 `--kvmem-method` 语义不变。⚠️ 诊断用的 `KVMEM_DUMP`/`read_gpu_block` 对 turbo 行直接跳过（没有可比的 f16 参考）。

## 验证结果

`Qwen3.5-0.8B-Q4_K_M`，Windows + MSVC + CUDA 12.8、`sm_70`、V100 16 GB。

| 维度 | master | `--kvmem` 关 | `--kvmem` 开（全常驻） |
|---|---|---|---|
| PPL `-c 512 --chunks 16` | 12.2387 | **12.2387** | **12.2387** |
| 贪心 48 token md5 | `1e05d115…` | 相同 | 相同 |
| 4k prefill pp（f16，10 轮轮换） | 10821.7 t/s | 10764.4（−0.5%） | 10808.0（−0.1%） |
| 4k decode tg（同上） | 250.1 t/s | 245.4（噪声带内） | 250.0（±0%） |
| CPU pp / tg（`-dev none`） | 291.9 / 31.60 | 294.4 / 31.44 | 287.2 / 31.41 |
| 显存峰值（`-c 32768`） | 1760 MiB | — | 1568（budget 8192）/ 1518 MiB（budget 4096） |

真的会驱逐时才有代价。`llama-bench -p 4096 -n 128 -b 512`（0.8B，工作集 576 token）：pp **8941.6 t/s** 对无 kvmem 的 13260.8（**−32.6%**），tg 262.1 对 252.0（decoding 反而略快，池子小、注意力带宽低）。质量 PPL 11.9618 对基线 11.9679（−0.05%）。
这 −32% 里约 32 个百分点是「把待换出块的裸 K 逐层镜像到 host」（5 ms/ubatch，192 次小块拷贝），**可 slab 化，未做**。
⚠️ 上面这一段的数字已被后续优化改写：capture 门控 + 打分器 staging 之后，同一口径的真驱逐代价降到 **−14.3%**（`13662.5 → 11706.5 t/s`，`34063912e` 那轮）；2026-11-15 复测为 **−16.7%**（`15256.3 → 12701.8 t/s`），2026-11-22 复测为 **−18.1%**（`15005.1 → 12293.8 t/s`，replay 关）。各轮的绝对值不可跨会话比，只看比值，详见本文末「四方复测」。

功能面（同一批构建）：长会话三连问 16k 上下文 / MTP 草稿回滚 / 图像输入都通过（细节见下面「已修复」与 `AUDIT_KVMEM_PERF_QUALITY.md`）。

**只有会评分的驱动才捕获 prefill K**（`e04bdb5d6`）：检索轮所需的 mean-K 与 query 行来自图中的捕获管道，它把每个 ubatch 的全部 K 行 D2H 到 host 再做打包。`llama-completion` / `llama-bench` / `llama-perplexity` 从不跑检索轮，却仍按检索方法付费搬运没人会读的行。捕获改由 `begin_retrieval_round()` 开启（server 每请求开一次），清轮/重置/重放入口不再重新开启；`harvest_gpu_v` 的 KV 镜像不受影响（换出块靠它回填）。0.8B `-p 4096 -b 512 --kvmem-budget 2048`：pp 9094 → 9710 t/s（+6.8%），server 侧 `KVMEM_QUERY scorer=page-softmax rows=1071 blocks=34` 照旧。

单测：KVMem 的测试已注册进 ctest（`test-kvmem-store`、`test-kvmem-runtime`、`test-spill-io`、`test-native-residency` 等），2026-11-15 复测时该组 **17/17 通过**。

测量方法、口径陷阱、噪声地板和每条结论的原始数据见 [`AUDIT_KVMEM_PERF_QUALITY.md`](AUDIT_KVMEM_PERF_QUALITY.md)。

## 长会话 / MTP / 多模态的修复（`471930f69`）

移植后第一次跑真实长会话暴露的问题都源于同一件事：KVMem 有**两份**历史描述（它包着的 attention cells、按位置寻址的 packed store），而只有 cells 会跟着调用方的裁剪一起更新。

| 现象 | 根因 | 修法 |
|---|---|---|
| 16k 上下文第二轮起 `resident KV has a position/coverage gap`，之后每步都失败直到 `Context size has been exceeded` | 服务端的 `CHECKPOINT_ATTN_ONLY_TRIM` 走 `seq_rm_attn`，基类 hybrid 直接裁**裸** attn 子对象，adapter 完全看不到 | `seq_rm_attn` / `seq_rm_attn_transient` / `seq_rm_transient` 在 hybrid 上全部改道 adapter；`seq_rm` 对「删到末尾」调用 `truncate_cached`，对前缀/中间删除直接丢弃 packed 历史（块按绝对位置寻址、RoPE 已烘进字节，无法重基） |
| MTP 首个被拒草稿后同样报 gap | 草稿回滚走 `seq_rm_transient`，同样绕过 adapter | 同上 |
| 图像输入 `failed to decode mtmd chunk, idx = 15` | 图像 patch 的 M-RoPE `pos[0]` 重复，adapter 却要求 ubatch 内的逻辑位置严格递增 | 行号改为**登记顺序**，`pos` 只作为 RoPE/空间信息；显式给 `logical_pos` 的驱动仍由它决定行号 |
| 每步都可能报 gap（非长会话也会） | 覆盖检查跑在本步 ubatch **之前**，半满块的尾巴必然还没有 cell | 检查只到上一步留下的前沿（`cells_materialized_`），并报出块号/行号/字段名 |
| 十几 k 上下文连续几轮后，某个**跨裁剪点**的块报 gap（`block 41x slot N row 0/32: cell is empty`），此后每步失败直到 `Context size has been exceeded` | 该块在前一轮生成时被驱逐（`gpu_slot=-1`，行只留在 packed store）；本轮裁剪落在块内又把它变回**半满尾块**，而 append 只写新行，已提交的前缀没人回填 | `restore_partial_tail()`：append 之前若尾块不满且 GPU 侧不完整，就按 packed store 重新落位并物化前缀（无空槽/打包缺失时明确失败，不再发布空行） |

回归面：PPL 12.2387（开/关）、11.9618（budget 2048），贪心 md5 `1e05d115…` 与 master 相同，16 个 kvmem 单测全过。

### 查询跨度装不下时：退回压力视图，不硬裁（`34063912e`，取代 `2264cf6c3`）

query span 就是**最后一条 user 消息**（server-context.cpp:10845）。当它覆盖的块数超过池子容量时，`query_fixed_blocks()` 抛 `mandatory Query content exceeds its selection budget`（kvmem_store.cpp:395），server 的 catch 把**整轮检索跳过**，工作集只剩压力路径挑出的尾部——模型因此看不到几十 k 里的绝大部分，表现为「思考正常、输出一点点就停」。

`2264cf6c3` 的做法是夹取最后 `budget - sink - recent - 1` 个块当必选集，让那一轮照常评分。这个补丁解决不了问题，而且更糟：夹过的**必选集自己就是完整的**，`selection_fits()` 拿它比对必然通过，于是没有任何一处能让调用方知道「这一轮是撕过的」——store 之后宣称驻留，头部却是空的。

fork 从来不做第三种状态：它拒绝 Query replay，**保留第一遍 prefill 在压力下建好的视图**。port 没有 replay 遍，等价物就是让压力计划继续驻留、跳过这一轮：

| 位置 | 现在的行为 |
|---|---|
| `retrieval_mandatory()` | 原名原样返回跨度覆盖的块，**不再夹取**——`selection_fits()` 正是拿它比对，夹取等于把「撕过」这件事从检查里抹掉 |
| `retrieval_mandatory_over_budget(need, cap)` | 唯一的拒绝判据：跨度 + sink 带 + 边界页 + recent 带 对 `budget_blocks()`，一次告警报出 `need/cap` |
| `apply_retrieval()` | 判据不过 → `retrieval_skipped reason=mandatory_exceeds_pool`；`selection_fits()` 不过 → `reason=selection_does_not_fit`；两者都 `harvest_flush()` 后返回，压力视图原地保留 |
| `accept_retrieval_round()` | 拒绝过的那一轮**不评分也不提交**，checkpoint 留给下一个请求，而不是记一个从未驻留过的视图的分数 |

`KVMEM_TRACE=1` 实测（0.8B、4649 token、`--kvmem-budget 2048 --kvmem-block-tokens 128 --kvmem-gen-reserve 1024`，池 3072 = 24 槽）：

```
KVMEM_TRACE retrieval_skipped reason=mandatory_exceeds_pool query=[2,4643) rows=4649
             budget_blocks=16 block_tokens=128 need=22 cap=14
W retrieval round rolled back (mandatory_exceeds_pool: 22 of 14 blocks), keeping the prefill pressure view
```

### 服务器侧两处修补（`16371170a`）

- `kvmem_conv_bind()` 里的 `llama_kvmem_store_create/switch/current/destroy` 没有 `#if defined(LLAMA_KVMEM)` 守卫，而声明在头文件里只在该宏下可见 ⇒ **`-DLLAMA_KVMEM=OFF` 直接编不过 server**（C3861 ×6）。这条同时也是本节的测试前提：没有 off 构建就没有「kvmem 关」的基准。
- 检索块把 `llama_kvmem_end_prefill_capture()` 写在 try 里。`apply_retrieval`/`accept_retrieval_round` 一抛异常，捕获管道就一直开着，此后每个 ubatch 都白搬 K 行。改为 catch 之后无条件关闭。

### CUDA 评分器必须真的编进 adapter（`88b0d969b`）

检索轮的 page-softmax 评分器走 `#ifdef KVMEM_PAGE_SOFTMAX_CUDA` 二选一，而那个宏由 `kvmem_enable_cuda_recall()` 加上——**只有独立构建会调它，嵌入构建从不调用**，于是 CUDA 的 llama.cpp 仍在 host 上逐页逐层打分：0.8B、3163 token 的一次请求里 `KVMEM_RETR_SUM score_ms=4194.8 / total_ms=4230.3`，检索轮本身就是那个请求（prefill 645 t/s）。在 `src/CMakeLists.txt` 的 kvmem 块里（kvmem 目标已存在、CUDA 已 enable 之后）调用它，同一个请求变成 `score_ms=400.7 / total_ms=427.8`，prefill **4896 ms / 645 t/s → 1039 ms / 3043 t/s**。

### 半满尾块的前缀回填

池子按块驱逐，**尾块下的块**迟早会被赶走；而服务端的 checkpoint 恢复常在**块中间**裁剪（`CHECKPOINT_ATTN_ONLY_TRIM p0=13207`，13207 % 32 = 23），于是这个块又成了尾块，但只有 packed store 还记得它那 23 行。append 只写自己新增的行（`apply_ubatch` 的 `sinfo.idxs` 是逐行的），前缀就此留空，而 store 仍宣称整块驻留 —— 下一次工作集事务 `snapshot()` 就在那块的第一行报 gap。

修法是在 `prepare_working_set()` 里、`register_append()` **之前**（此时 `n_tokens` 仍等于已提交行数）调用 `restore_partial_tail()`：尾块不满且 `gpu_kv_complete()` 为假时，从 packed store 落位并物化前缀（`write_block_to_gpu`），MTP 侧同步 `restore_checkpoint_block()`。失败一律 fail-closed —— 发布空行比报错更糟。

复现与验证（Qwen3.5-2B、f16、`-c 262144 --kvmem-budget 1000 -b 512`、13.2k prompt 连续 5 轮各生成 5k）：修复前 3/6 次在第三轮 decode 报 gap；修复后 5 轮全过，且 `KVMEM_TRACE=1` 能看到 `tail_restage` 正好在这些轮次触发（一次运行 3 次）。PPL 1.2126（开/关逐位同）、真驱逐 budget 576 为 1.2135。

## 四方复测：旧版 / port 关 / port 开（replay 开与关）/ fork（2026-11-22 复测）

这一轮问的还是三件事：移植后**有没有退化**、真驱逐**还差多少**、质量**与 fork 是否等效**。
同一台机器（Windows + MSVC + CUDA 12.8、`sm_70`、V100 16 GB、`Qwen3.5-0.8B-Q4_K_M`），
参数与 `34063912e` 那轮逐字相同，因此两轮按**比值**对照。与上一轮（11-15）的区别只有一处：
`port-kv` 拆成 **replay 关 / replay 开** 两个臂（同一份二进制，`KVMEM_REPLAY=0/1`），
用来把「补上 Query replay 之后还剩多少差距」和「replay 本身要付多少钱」分开。

⚠️ **绝对值不要跨会话比**：本轮 `port-nokv` 的 pp4096 是 15005，上一轮是 15256，差 1.7%，来自机器状态
（时钟/驱动），不是代码。

⚠️ **每个单元都跑多轮**：bench 4 轮轮换、server 4 轮轮换、PPL 3 轮。
两处口径修正见下——不做这两件事，同一格的单次结果会差 18%，足以造出或抹掉结论。

### 六个单元（`fork-kv` 另在 server 表里加两行）

| 单元 | 二进制 | 含义 |
|---|---|---|
| `old` | `tmp/build-master-cuda`（冻结快照，2026-10-30 02:44） | **未移植 kvmem 的旧版**：`--help` 里没有任何 `kvmem` 字样 |
| `port-off` | `tmp/build-kvmem-off`（`-DLLAMA_KVMEM=OFF`，2026-11-22 01:39 重建） | port 树，kvmem **不参与编译**：`llama.dll` 里 `kvmem_store_create` 出现 **0** 次（kvmem-on 构建是 2 次） |
| `port-nokv` | `tmp/wt-kvmem/build`（`b7bb90f16`），不加 `--kvmem` | port 树，kvmem 编入但不启用 |
| `port-kv-r0` / `port-kv-r1` | 同上 + `--kvmem --kvmem-budget 2048 --kvmem-block-tokens 128 --kvmem-gen-reserve 1024`，`KVMEM_REPLAY=0/1` | port 树，真驱逐；两个臂是**同一份二进制**，只切环境变量 |
| `fork-nokv` / `fork-kv` | `kvmem-llama.cpp/build-v100/bin`（`llama-kvmem-server.exe` / `llama-bench.exe`，2026-10-31） | fork 本体，参数同上 |

⚠️ `old` 是一个**冻结的二进制**（构建目录的源现在是 master，master 已含 kvmem）：它的价值就在于是移植前那一刻的快照，重建它就不再是旧版。

⚠️ **`port-off` 本轮是重建过的**：`-DLLAMA_KVMEM=OFF` 的构建曾自多会话存储那次提交起编不过
（`kvmem_conv_bind()` 的 `#else` 分支写 `(void) request;` 而参数名是 `request_ledger`，`C2065`），
11-15 修掉后才有当天的数据；此后源码又前进 14 个提交（FA 的 F32 累加、bf16 直读、FWHT 隔离、capture 去串行化），
所以 11-22 又增量重建了一次。`old` / `port-off` / `port-nokv` 三者现在是同一天的源码与同一批构建。
判据：`strings llama.dll | grep -c kvmem_store_create` = 0 / 0 / 2。

⚠️ 两处口径修正（不修就会得到假差距）：
1. **bench 先热身一轮丢弃**，再轮换顺序跑 4 轮。
2. **server 每个单元先发一个 1-token 请求丢弃**：新起的服务第一个请求要付 CUDA graph 捕获与模块加载，
   这段落在 `prompt_eval` 里，同一格单次可以在 8725–10184 t/s 之间摆（±18%）；加了预热后同一格是 10909–11049（±1.3%）。
   server 也改成轮换顺序——固定顺序会让"刚退出的服务"这一刻的 GPU 状态变成系统偏差：不轮换时 `port-off` 看起来比
   `port-nokv` 慢 8.7%，轮换后是 −0.1%。

⚠️ `old` 与 fork 是 `CMAKE_CUDA_ARCHITECTURES=70`，`wt-kvmem/build` 是 `52;70`；运行期都走 sm_70 的 kernel，
fatbin 里多一份 sm_52 不改变稳态吞吐，但这确实是口径差异，记在这里。

⚠️ fork 的 `llama-bench` 与 `llama-perplexity` **不注册 `--kvmem`**（`--help` 里 0 个 kvmem 字样），所以 fork 的 kvmem 开/关只能在 server 上比。⚠️ port 的 `llama-bench` 没有 `--kvmem-gen-reserve`，基准表的池子 = budget（16 块），server 表显式给 1024（池 3072 = 24 槽）。

### prefill / decode（`llama-bench -p 4096 -n 128 -b 512 -fa 1 -ctk/-ctv f16 -dev CUDA0 -r 3`）

先热身一轮丢弃，再**轮换顺序**跑 4 轮，取中位数：

| 单元 | pp4096 t/s | 对 `port-nokv` | tg128 t/s | 对 `port-nokv` |
|---|---|---|---|---|
| `old`（无 kvmem） | 14968.6 | −0.2% | 278.8 | +3.1% |
| `port-off` | 15035.4 | +0.2% | 275.4 | +1.8% |
| `port-nokv` | **15005.1** | — | **270.5** | — |
| `port-kv-r0`（真驱逐，replay 关） | 12293.8 | **−18.1%** | 278.1 | +2.8% |
| `port-kv-r1`（真驱逐，replay 开） | 12398.0 | **−17.4%** | 275.0 | +1.6% |

4 轮原始值（pp）：
`old` 15055.5 / 14601.1 / 14881.8 / 15105.2；`port-off` 15074.7 / 14732.6 / 15138.8 / 14996.1；
`port-nokv` 14897.3 / 15017.2 / 15245.0 / 14993.0；`port-kv-r0` 12206.9 / 12380.8 / 12538.8 / 11892.8；
`port-kv-r1` 12695.3 / 12645.1 / 12150.9 / 12019.6。

读法：
- **编入 kvmem 不花钱**：`port-off` / `port-nokv` / `old` 的 pp 互差 ≤0.2%，tg 互差 ≤3.1%，都在噪声地板
  （同二进制自比 ±1.5%，这一晚 tg 的单轮摆幅到过 262–280）内。三个数的排序在三个会话里都不稳定，判定为噪声。
- 真驱逐的 prefill 代价本轮 **−18.1% / −17.4%**（上轮 −16.7%），decode 反而 +1.6% ~ +2.8%（池子小、注意力带宽低）。
  4 轮内部很紧（11893–12539，±2.7%）。
- **replay 在这张表上没有位置**：`llama-bench` 不跑检索轮，`KVMEM_REPLAY` 两个臂的差（−18.1% 对 −17.4%，0.7 个点）
  在噪声内。replay 的钱只在 server 表上能看见（重算 512 行）。
- **fork 基线本身慢**：同一台机器、同一个 `llama-bench`，fork 的 pp 低 25.7%（11-15 那轮）、tg 低 8.7%。
  所以「fork 比 port 快/慢」必须拿 **fork 对 fork、port 对 port** 的比值说话。

### 检索轮与答案（server，4649-token 文档、4 轮轮换、每单元取中位）

请求是同一份 14 000 字中文文档（tokenize 后 4649 token），三个探针分别埋在 **12%**（`H-2298`，落在被驱逐的头部）、**60%**（`M-4471`）、**末尾**（`T-9932`），问题要求三个温度上限全部按格式给出。`-c 16384 -b 512 -np 1 --no-webui`，`temperature 0`，`max_tokens 200`。每单元 4 轮，答案逐轮相同，时间取中位。

| 单元 | pp t/s（中位） | tg t/s（中位） | 对 `port-nokv` | 检索轮 | H / M / T |
|---|---|---|---|---|---|
| `port-off` | 10266.7 | 212.7 | −0.7% | 无 | **3/3** |
| `port-nokv` | **10343.7** | 211.1 | — | 无 | **3/3** |
| `port-kv`（默认参数，`93756cdf7` + 重放落单批） | **7039** | 203.4 | **−32.0%**（同窗口配对 −34.4%） | 应用（自动截尾，重放 = 1 批 512 行） | **3/3** |
| `port-kv --kvmem-query-max 512`，replay 关 | 7378.1 | 206.1 | −28.7% | 应用 | 2/3，头部答成 87 |
| `port-kv --kvmem-query-max 512`，replay 开 | 6152.2 | 203.4 | −40.5% | 应用（重放 = 512 + 7 两批） | **3/3** |
| `fork-nokv` | 6896.8 | 129.4 | −33.3% | 无 | **3/3** |
| `fork-kv` | 4288.9 | 134.0 | −58.5% | 应用 | **3/3** |

⚠️ 默认行与显式 512 行只差一件事：**重放要不要落在一个 batch 里**。`93756cdf7` 把"装不下就整轮拒绝"
换成"截尾继续"，默认的 Query 是 37 块、cap 14 块，于是截到 512 token 的行；但重放跑的是
`[q_begin, prompt_end)`，模板尾还有 7 行 ⇒ 519 行 = 512 + 7 **两个新形状**，第二个 batch 只是 7 行却要付
整份图的重新预约与设备同步。现在把打分跨度再让出这 7 行（`n_batch − (prompt_end − q_end)`），重放正好
落在 prefill 已经建好的 512 宽度上；同一二进制 `KVMEM_REPLAY_FIT=0/1` 五轮配对 **731 → 660 ms**
（−9.7%，4/5 轮更快，`−39.4% → −34.4%`）。改之前那一行是 8291 t/s / 2/3（捕获前拒绝、只看压力视图），
这也是「默认参数看起来便宜」的全部来源：它根本没做检索。

```
KVMEM_TRACE span_clamped query=[2,5598) -> [5086,5598) cap_blocks=62     # 27B 上同一机制
```

**replay 的钱在这张表上第一次被单独量出来**：同一个二进制、同样的 512 行 Query，
replay 关 7378 t/s、replay 开 6152 t/s，**−16.6%**（wall 0.79 → 0.94 s）。
这一格是**旧的**重放形状（519 行 = 512 + 7 两批）；把重放压进一批之后同一格是 7039 t/s（−34.4% 对 nokv）。
剩下这部分与「把 512 行重算一遍」的预期一致，也和下一节 `-10.3%` 的量级相符（那一格的 prompt 更长）。
买到的是一格：replay 关时头部（深度 12%、已被驱逐）4 轮全错，replay 开时 4 轮全对。
⚠️ replay **默认是开的**（`tools/server/server-context.cpp:52`，`KVMEM_REPLAY=0` 才关），
所以"默认参数"这一行本来就含重放；上面那次 2/3 是 `KVMEM_REPLAY=0` 的 A/B 臂。

### 质量：24 起点检索矩阵（取代单文档的 2/3 vs 3/3）

上面那张表里 `port-kv` 2/3、`fork-kv` 3/3 的差别**不能用来判断质量**。同一份文档、同一条命令，只把
打分跨度少算 4 行（`--kvmem-query-max 504`），2/3 就变成 3/3，各跑两次都复现：16 页预算面对 4649
token 的文档时，答案卡在 token 对齐的刀口上，这个探针的分辨率低于要测的效应。

换成一个有分母的度量：同一套三针探针在 **24 个文档起点**（0…23000，步长 1000）各问一次，等于
换了 24 套 token/页对齐，命中数以 **72** 为分母，Fisher 精确检验。脚本 `tmp/cmp/qmatrix.py` +
`tmp/cmp/qrun.sh`，针位由 `tmp/cmp/needle_pos.py` 用服务自己的 `/tokenize` 量出。

| 单元（budget 2048 = 16 页历史，除注明外） | 命中 | 头部针 | 对 fork 的 p |
|---|---|---|---|
| **`port-kv` 默认参数**（`93756cdf7` 后：自动截尾 + replay，重放单批） | **56/72 = 77.8%** | 19/24 | **1.00（不可分）** |
| `port-kv` 默认参数，重放拆成两批（`KVMEM_REPLAY_FIT=0`） | 58/72 = 80.6% | 20/24 | 1.00 |
| `port-kv --kvmem-query-max 512`，replay 关 | 43/72 = **59.7%** | 10/24 | **0.018** |
| `port-kv --kvmem-query-max 512`，replay 开，`--kvmem-budget 2560`（20 页） | 62/72 = **86.1%** | 21/24 | 0.38 |
| **`fork-kv`** | 57/72 = **79.2%** | 19/24 | — |

⚠️ 前两行是同一份代码的两个开关：把重放让出模板尾的 7 行（口径 512 → 505 行）换来单批重放，
命中 58 → **56/72**，配对到格 **2 好 / 4 差**（符号检验 p=0.69，两次独立运行逐格相同）。
这是换性能的代价，记在这里而不是抹平：505 与 512 个打分行只差 7 行，page-softmax 的平均 query 变了，
选页在刀口格上翻了两格。与 fork 的 57/72 仍然不可分（p=1.00）。

（预算 3072 / 4096 两档本轮未重测：那两档池子已经覆盖整篇文档，不涉及选页，11-15 测出来两侧都是 18/18。
表里前两行是同一份二进制，只切 `KVMEM_REPLAY_FIT`；`fork-kv` 用它的默认块宽 128。）

读法：

- **replay 就是那 15 格**：replay 关 43/72（59.7%）→ 开 58/72（80.6%），配对到格 **16 好 / 2 差**，
  符号检验 p=0.0103；同期 fork 是 57/72。补上 replay 后两棵树在**同样的 16 页**上统计上不可分（p=1.00）。
  头部针一起从 10/24 涨到 20/24（fork 19/24），中部针 11 → 14/24（fork 14/24）。
- **不补 replay 时差距是真的**：43 对 57，p=0.018，头部针 10 对 19。也就是说 11-15 那轮记的
  "16 页时 port 60% / fork 79%" 在 11-22 这轮仍然是 60 / 79——**这个差距不是上一轮的偶然**，而是缺 replay 的稳定形状。
- **多给预算也能涨，但不是同一件事**：16 页 + replay 已经 80.6%，20 页 + replay 86.1%（p=0.38）。
  预算曲线还在（59.7 → 80.6 → 86.1），但"加到 20 页就追平 fork"这条路现在不需要走了。
- **单批重放换来的 2 格**在噪声里，但它不是免费的：`KVMEM_REPLAY_FIT=0` 随时可以换回 58/72 与
  −39.4% 的 prefill。要质量优先就把这个开关关掉，要速度优先就留着（默认留）。

### 打分跨度该多大：query-max 扫描（4649 token、budget 2048、replay 开）

同一批 24 起点，只改 `--kvmem-query-max`。命中以 72 为分母；pp 是每档 3 次独立起服务的单请求中位：

| query-max | 命中 | 头部针 | 中部针 | pp t/s | 对 `port-nokv` |
|---|---|---|---|---|---|
| 128 | 61/72 = **84.7%** | 21/24 | 16/24 | 6215 | −39.9% |
| 256 | 60/72 = 83.3% | 21/24 | 15/24 | 7098 | −31.4% |
| **512（默认自动截到尾 512）** | 58/72 = 80.6% | 20/24 | 14/24 | 6378 / 6581 | −38.3% |
| 1024 | 46/72 = **63.9%** | **9/24** | 16/24 | 5685 | −45.0% |

两条读法：

- **不是越大越好，1024 反而塌**（46/72，配对到 512 那一档是 1 好 / 12 差，p=0.040，重跑一次逐格相同）。
  原因是打分跨度里的 Query 行越多，page-softmax 对"平均 query"越敏感：这段 4649-token 的 user 轮里，
  尾部 1024 行已经包含约 1k token 的**文档正文**，它们与"要找哪条记录"无关，把页分数摊平了；
  256 只覆盖问题本身，最锐。头部针 21 → 20 → 9 就是这个形状。
- **≤512 的三档在性能上分不出来**：这一晚 3 次中位的噪声带是 6.2–7.1k t/s（同二进制自比 ±8%），
  128/256/512 全落在里面；只有 1024 明显慢（5685）。三档的命中差（61 / 60 / 58）也都在 p>0.8。
- ⚠️ **128 的 84.7% 不能直接推广**：11-06 在 12k 多轮那组测过 128 = 79.2%，比 512 差 16 个点——
  那一组的机制是"小 cap 让冷启动轮拿到一个装得下的假计划并提前定住工作集"。
  本轮的形状是单轮长文档，两者不要混用。**默认取 512**（fork 的 driver 也默认 512），
  要在自己的负载上换 256，先拿自己那组形状复核。
- **修复前默认参数等于没做检索**：55.6%，头部针 3/24（`prefill capture skipped: needs 37 of 14 blocks`，
  整轮不跑）。`93756cdf7` 之后同一份默认参数是 58/72、头部 20/24。留着这条是为了记住：
  「默认看起来便宜」曾经全是"根本没检索"造成的。

把差距拆开看。同一格两棵树**留下的 20 页是否相同**，是一个不需要任何假设的分组变量
（从每格日志的 `KVMEM_TRACE selected` 直接读）。为此把 offset 扩到 36 格（0…35000，步长 1000）
（2026-11-15 测；11-22 复测没有重跑这组，因为它解释的是"选页不同"的**因**，而 replay 补上后
两棵树在 16 页上已经统计不可分）：

| 36 格（budget 2048） | 格数 | head | mid | tail |
|---|---|---|---|---|
| 两树选中集**相同** | **7** | port 5/7 · fork 6/7 | 4/7 · 5/7 | 7/7 · 7/7 |
| 两树选中集**不同** | **29** | port **6/29** · fork **18/29** | 7/29 · 13/29 | 27/29 · 29/29 |
| 合计 | 36 | port 11/36 · fork 24/36 | 11/36 · 18/36 | 34/36 · 36/36 |

⇒ **差距几乎全部落在"两树选页不同"的那 29 格上**。选页相同时两边几乎一致（head 只差 1 格，
就是下面那个 offset 6000）；选页不同时是 6/29 对 18/29。所以"质量更差"这件事的**因**就是
**同一预算下选出了不同的 20 页**，而不是检索轮重算、也不是回填。

⚠️ 这一条**推翻了本文早前两版的判断**。第一版说"Jaccard 中位 0.90，选页不是主因"——那个
0.90 是把 20 页里差 2 页平均掉之后的数，太粗：20 页里差 2 页足以决定答案。第二版据此转去查
回填与检索轮，方向也偏了（回填已证明精确，见下；检索轮只在 1/7 的相同集格里才可能有份）。

相同集那 7 格里唯一不一致的是 offset 6000：两棵树的选中集**逐页相同**（20 页）、
`occupy` 里 block→cell→pos 也**逐行相同**，但

```
port-kv   H-2298=64 M-4471=87 T-9932=64      （3/3 次一致）
fork-kv   H-2298=51 M-4471=87 T-9932=64      （3/3 次一致）
port-nokv H-2298=51 ...   fork-nokv H-2298=51 ...
```

同一份选中集、同一个位置映射，port 的 `--kvmem` 路径答出了尾部记录的数值，而同一棵树的
非 KVMem 路径（全量 prefill）答对。

**回填本身没有损失（已用字节核对）。** 给两侧各加一个临时探针（`KVHASH_PAGES`），按 block id
对 GPU 上 K/V 张量的字节做 FNV-1a（`get_k_storage`/`get_v_storage`，`ne=[512,5120] nb=[2,1024]`），
同一个 binary 跑两个预算：

| offset 6000，port | 历史块（0–21 里被保留的 13 块） | 查询区（22–26） |
|---|---|---|
| `--kvmem-budget 4096`（27 页全留，从不驱逐，答 51） | — | — |
| `--kvmem-budget 2048`（16 页，7 块驱逐后回填，答 64） | **K 与 V 逐层逐字节相同**（13 块 × 6 层 × 2） | 6 层里 5 层不同 |

也就是说：**从宿主 store 回填的历史页与"从未离开 GPU"的同一页字节完全一致** —— 编解码、rope、
层覆盖、搬运范围都没问题。唯一变化的是**查询区**（block 22–26），而那是检索轮**重算**出来的，
本来就应当随保留的历史而变。fork 在 2048 与 2304 之间是同一个结构（历史块全同、查询区全变），
所以两边都没有"复用旧查询 KV"这种便宜。

⇒ 差距不在 store、不在回填。跨树比字节本身也不成立（同预算下每个块都不同，布局/旋转本就不同）。

**但选页是主因，而且其中一半是 port 的真缺陷：它把 Query 自己的页也拿来打分了。**
给两侧各加一个 `KVMEM_SCORE_DUMP` 探针（逐页分数、每层 Q 行范数与哈希、每块每层 mean-K 的范数与分量和），
offset 0、budget 2048：

| | 打分的页数 | 分数最高的 3 页（token 位置） | head |
|---|---|---|---|
| port | **37**（0…4608） | **4608 / 4224 / 4352** ← 全是 Query 自己的页 | 2/3 |
| fork | **33**（0…4096） | 2816 / 3328 / 3456 | 3/3 |

page-softmax 是**在页这一维做 softmax 再对 Q 行取均值**，多进来的 4 页会抢走概率质量、把历史页整体重排。
把 frontier 收到 Query 边界（`blocks=37→33`）后，port 的 top-8 与 fork 的 top-8 有 6–7 个相同，
逐页分数相对差的中位从"完全不同"降到 **30%**；24 格上 head 从 10 升到 12、总命中 43→45/72（p=0.86，仍不显著）。
这个修正留着（未提交），它是真缺陷，但**不足以**补上差距。

剩下那 30% 不是输入的量级问题，逐个排除（都是本次实测）：

| 假设 | 实测 |
|---|---|
| Q 行数 516 vs 512 | `--kvmem-query-max 508`（rows=512）与 512 的分数**逐页相同** |
| 捕获多走 4 行（越过 Query 末尾） | 裁掉尾部 4 行：相对差 30.5% → 29.5%，无变化 |
| 打分页集 | 修正后两侧都是 33 页 |
| 每块 mean-K | 范数与分量和逐块逐层吻合到 **0.2%**（b1–b4 小数点后 4 位相同） |
| 重算后 cells 占用 | `used=2473 used_max_p1=2473 seq_pos=[0,4648]` 两侧逐字相同 |
| **Q 行的值** | **同一批 512 行，哈希不同**（port `9db5dab4…` / fork `3967ccd2…`），范数 1462.3 vs 1457.2 |

最后一格是关键：**两棵树的 Q 值本身不同**。而两侧**关掉 `--kvmem` 跑同样 18 格，答案是 18/18 全同**——
说明基座在本任务上一致，但 Q 向量并不逐位相同（port 跟 master，fork 是 10-31 的快照）。
page-softmax 会把 1% 量级的 Q 差异放大成几十个百分点的分数差，于是 16 页里第 14/15 名那一格翻来覆去。

⇒ 结论要分三层：

1. **port 有一个真缺陷**：把 Query 自己的页算进 softmax 分母，已改，43 → 45/72（方向正确但不闭合）；
2. **主因是缺了 replay 阶段**（不是选页，也不是基座数值）：补上后 45 → 58/72，与 fork 的 57/72 p=1.00。
   选页不同只是让这个差异显形——同样的 16 页，fork 会按最终视图把 Query 重算一遍，port 直接沿用
   首遍的结果，于是一个被裁掉页面的视图照样在输出答案；
3. ⚠️ **"两棵树基座不同、跨树不可比"这个说法在本轮收回**。Q 值确实逐位不同（hash `9db5dab4` vs
   `3967ccd2`），但在同一提示、同一预算、同一页集下，谁答对是可比的——补上 replay 后 port 反超 1 格，
   再谈"放大 1% 的 Q 差异"已经没有剩余效应可解释。

这一节取代的旧结论："port 2/3、fork 3/3，质量差距仍在"。当时那句里的 2/3 有一半是刀口 artifact：
`port-kv` 那格根本没做检索（`prefill capture skipped`，`93756cdf7` 之前的行为），`qm512` 那格是 4 行跨度决定的。

### Query replay：检索轮之后要按最终页集重算 Query（本轮移植）

首遍 prefill 把 Query 段算在**完整历史**上（压力视图还没裁），检索轮随后才把 16 页定为真实上下文。
采样读的是首遍最后一个位置的 logits ⇒ 它描述的是一个**已经不存在的视图**。fork 的 driver 因此有
一个 replay 阶段：选页落地后，冻结 Q、按最终视图重跑 `[query_begin, prompt_end)`，采样改读重算后的
logits。port 早先把这套机制（`set_replay`/`query_replay_fits`/`get/set_tail_mean`）一起移植进来了，
**但没有调用者**，等于只有函数没有阶段。

现在的编排在 server 侧，adapter 只提供原子操作：

1. prefill 在 `query_begin` 处**收尾**（ubatch 不越过该行），抓两样东西：
   `tail_mean`（Mean-K + 图 + 反馈快照）与 `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` 镜像
   （recurrent 行 + 它对应的 frontier）。
2. `llama_kvmem_apply_retrieval()` 应用选页。
3. `llama_state_seq_set_data_ext(PARTIAL_ONLY)`（截回 frontier、装回 recurrent）
   → `replay_begin()`（恢复 Mean-K/图、冻结 Q、作废 `[begin,-)` 的 packed K/V）
   → 重跑 `[query_begin, prompt_end)` → `replay_end()`，`slot.i_batch` 指向重算后的最后一行。

同一份 binary 用 `KVMEM_REPLAY=0/1` 切开，24 格（budget 2048、`--kvmem-query-max 512`）：

| 变体 | 命中 | 配对到格 | 符号检验 |
|---|---|---|---|
| 无 replay（首遍视图） | 43/72 = 59.7% | — | — |
| 只退 attention（`remove_logical`，不恢复 recurrent；11-15 测） | 54/72 = 75.0% | 12 好 / 4 差 | p=0.077 |
| **attention+recurrent（`state_seq_set_data_ext`）** | **58/72 = 80.6%** | **16 好 / 2 差** | **p=0.0103** |

⇒ 追平 fork（57/72，p=1.00）。两次独立运行 24 格逐格相同（58/58），这一阶段不引入不确定性。
11-22 复测把"无 replay"从 45/72 重新测成 43/72（同一偏置方向、同一量级），配对从 14:2 变成 16:2；
中间那行"只退 attention"需要改代码才能切，本轮没有重测，数值沿用 11-15。

命中率是二值的，而本文上一版正是被一个二值结论误导过（"20 页追平"被 margin 推翻），所以同一份代码
再用连续指标测一遍：prompt 只问 H-2298、要求首 token 是数字，取 `logit(5…) − logit(8…)`。

⚠️ **这两张表是 11-15 的数据**：读分布要走 `KVLOGIT` 探针，那个探针没有提交，11-22 的二进制里没有它
（`strings llama.dll | grep -c KVLOGIT` = 0）。同一件事在 11-22 由上面那张 24 格命中表覆盖，
结论方向一致（replay 关 59.7% → 开 80.6%，fork 79.2%）。

| 24 格配对 | margin 均值 | 中位 | >0 的格数 |
|---|---|---|---|
| port 无 replay | +0.322 | +0.020 | 12/24 |
| **port + replay** | **+3.341** | **+4.299** | **20/24** |
| fork 16 页 | +3.299 | +4.673 | 19/24 |

| 配对 | 更好 / 更差 | 中位差 | 符号检验 |
|---|---|---|---|
| replay vs 无 replay | **20 / 4** | **+4.24** | **p=0.0015** |
| replay vs fork | 13 / 11 | +0.03 | p=0.84（不可分） |
| 无 replay vs fork | 4 / 20 | −4.01 | p=0.0015（上一版结论，逐格复现） |

两个指标同向：命中率 59.7% → 80.6%（11-15 那次是 62.5% → 80.6%），margin +0.32 → +3.34；而 fork 的 +3.30 与补上 replay 后的
port 相差 **0.03**。这条轨迹不是"多给预算"能走出来的——20 页那一行 margin 只有 +0.88。

⚠️ 四个必须踩对的点（每一个都单独造过故障）：

- **两半必须一起退**。只删 attention 行时，hybrid 模型的 recurrent 还停在首遍末尾，replay 的 batch
  会被 `find_slot` 判成 `non-consecutive token position 4641 after 4648`（每次 replay 两条 W），
  而且重算实际上用的是"未来"的状态。只有 75%，补上 checkpoint 后才到 80.6%。
- **范围是 `[query_begin, prompt_end)`，不是 user span 的末尾**。chat 模板的尾行也在首遍里、也被截断；
  只补到 span 末尾会留下 7 行缺口，下一次 decode 直接
  `non-consecutive token position 4649 after 4641` 并 500。fork 用 `eval_end` 是这个道理。
- **batch 必须带 `logical_pos`**。不带的话 adapter 把它们当追加行（从 `store_n_tokens()` 起分配），
  立刻 `row is outside the store`，`init_batch` 失败、decode 返回 1。
- **媒体跨度跳过**：vision 解码路径不复现，遇到 `has_media()` 就退回首遍视图并在 trace 里说明。

代价与边界：4647-token prompt 上 pp 7017 → 6295 t/s（**−10.3%**），wall 0.91 → 1.01 s（重算 512 token）。
cached turn 同样生效：第二轮 `n_prompt=65` 时 replay 打的是 `row=4678 q=[4678,4703)`，答案仍正确；
若 Query 行本身已被缓存复用（`query_begin < n_tokens`），没有可抓的快照，直接跳过。

#### prefill 的三笔账：捕获、打分、重放

「开了 kvmem 就变慢」不是一个数，是四段加起来。同一格反复测（Qwen3.5-2B-Q8、`-c 16384`、
4058-token prompt、单请求、不带 MTP），把每段单独关掉：

| 配置 | pp t/s | 这一段买的是什么 |
|---|---|---|
| 不带 `--kvmem` | 8229 | 基线 |
| `--kvmem --kvmem-budget 16384`（池 = 上下文） | 7795–7931 | 捕获整体关闭（`no_eviction()`），只剩 adapter 的追加记账 |
| `--kvmem-budget 8192 --kvmem-method recency` | 6878–6923 | 池成窗口 ⇒ 捕获打开（只有 K，Q 已按 method 门控），但从不打分 |
| `--kvmem-budget 8192` | 5891–6242 | 加上打分（71 → **7.75 ms**，见下） |
| `--kvmem-budget 2048` | 5053–5274 | 加上打分（60 → **7.7 ms**）+ 重放 93 ms |

（第三行同时是本轮改动的对照：它一行都不该受打分优化影响，却也在两次会话之间漂了 ~5%，
 就是这套单格测量的噪声地板；所以下面的结论只用**段内**数字算，不用单格 pp 差。）

单段实测（`KVMEM_RETR_SUM` / `KVMEM_REPLAY_TIME` 两个现成的计时面）：

- **打分的钱不在网格，在宿主转换**（本轮已修）。检索轮 total 63–71 ms，其中 `score_ms` 占 57–71 ms，
  且随 Query 行线性（512 行 57 ms、256 行 29 ms、128 行 12.5 ms）。拆开看（临时探针，
  `KVMEM_SCORE_PROBE`）：
  `stage_ms=70.3 / kernel_ms=0.65 / h2d_ms=2.0 / sync_ms=2.4 / means_ms=1.6`。
  `stage_query_half()` 在宿主上把 Q 转成 fp16：4k prompt 上 14.7M 个元素、**两遍**扫描（范围检查 + 转换），
  而 MSVC 在这个 TU 没有 F16C 时把 `__float2half_rn` 编成软件例程 ⇒ 2.4 ns/元素。
  原先猜的「grid 两万 block、启动开销主导」是错的：kernel 只占 0.65 ms。
- **重放**：519 行 = 86–106 ms（`restore` 5–9 ms、`loop` 81–97 ms，即 ~4900–6000 t/s）。这一段是按行付的，
  跟「重算」这件事本身等价，除非少算行。
- **捕获**：4k prompt 上 78.6 MB D2H、39–50 次事件。字节只合 8 ms，而开/关之间差 ~16% of prefill
  （7795 → 6570）⇒ 贵在每 ubatch 的机制（event sync / pack），不在搬运。这一段的逐项拆账见下一小节。
- **压力搬运**（池装不下时 stage_out）：只有 4%（budget 4096 → 2048 是 5154 → 4958），不是主因。

**本节已做的一处优化：页集没变就不重放。** 检索轮经常什么都没搬（池够大时 `n_move=0 n_stage_in=0`），
可是首遍的 Query 行仍然被整段重算一遍——重算的是同一批行、对着同一个视图，纯属把同样的 KV 再写一次。
判定用**驻留指纹**（每个块的 `(block_id, gpu_slot)` 的 FNV-1a）：压力搬运前后、检索轮应用前后各取一次，
哈希没变就说明首遍描述的就是刚选定的视图，`llama_kvmem_first_pass_stale()` 返回 false，server 跳过重放。

- 池够大（budget 8192，24 格，带 MTP）：pp 4628 → **5012（+8.3%）**；命中 72/72 → 71/72，
  逐格看是 0 好 1 差（第 3000 格 M 值 87→64）。那一格是已知的刀刃格（4 行文本就能翻盘），
  差别只来自首遍与重放的 ubatch 切分不同带来的浮点尾数差。
- 池紧缩（budget 2048，24 格）：**完全不变**——`n_move=19`，指纹变了，照旧重放；
  57/72 与 24/24 逐格答案与优化前一致。

⇒ 这条优化的适用面是「预算够大、检索轮空转」的那种会话（真实长会话里占多数）；池紧缩时它自动退成原行为。

**本轮第二处优化：打分不再在宿主上转 fp16。** Q 以 float 上卡（H2D 翻倍，每层 +1 MB），
核心里用 `__float2half_rn` 转一次——同样的取整，只是换成 GPU 做；范围拒绝搬到设备上的一个 flag，
主机最后读 4 字节、照旧抛 `query operand is nonfinite or exceeds the FP16 range`。同时把
`page_logits` 从「一个 warp 一页、grid = 行×head×⌈页/4⌉」改成「一个 warp 一个 distribution、
循环所有页」：每个页的点积与 warp 归约顺序不变，所以 logits 逐位相同，但 block 数少 20×、
每行的 Q 只读一次而不是每页组一次。

- 打分：**77 → 7.3 ms**（`stage` 0、`h2d` 3.75、`kernel` 0.38、`sync` 2.35）。
- 换算成 prefill：8192 那格检索轮 71 → 7.75 ms（同格 `KVMEM_RETR_SUM`），即 **+9%**；
  2048 那格 60 → 7.7 ms，即 **+6.4%**。
- 等价性：24 格逐格答案相同（57/72、`24/24`），ctest 17/17。

**本轮第三处优化：捕获不再让主机等 GPU，也不再紧贴申请缓冲。**

捕获的账要按**线程**分，不然会归错因。同一个 4058-token 请求（`KVMEM_HARVEST_SUM`，每请求一行）：

| 段 | 耗时 | 在哪个线程 | 性质 |
|---|---|---|---|
| `submit_ms` | 1.8（修前 **163**） | 主线程 | 提交副本 + 等 compute |
| `full_ms`（整块 V 收集） | 30–32 | 主线程 | 固有搬运 |
| `d2h_wait_ms` | 135 | harvest worker | 等 D2H，本质是在等 GPU 算完 |
| `pack_ms` | 109（修后 98） | harvest worker | 宿主转换 + `write_layer_mean_k` |
| `sync_ms` / `d2d_ms` | 0.4 / 0.8 | 主线程 | 事件与 D2D 提交 |

两处真开销，都在主线程上：

1. **每个 ubatch 末尾等一次 `compute_done`**（163 ms）。它让主机在 ubatch 之间完全对齐 GPU，
   双缓冲流水线退化成串行。而这一步其实不需要：下一批 ubatch 只欠"别在本批的 graph 缓冲被读走之前
   覆盖它"，而这个顺序**已经在设备上**——D2D 在 harvest stream 上等 `compute_done`，
   `ggml_backend_event_wait(be, snap_be)` 又把 compute stream 排在 D2D 之后。删掉主机等待：
   **+7.0% ~ +8.6% pp**（两次独立的同二进制、环境开关 A/B，4 轮 / 3 轮交替配对）。
2. **D2H 暂存缓冲紧贴字节数申请**（`cudaMalloc` + `cudaMallocHost`）。捕获集在一次 prefill 里**不均匀**：
   落在 Query span 之前的 512-token ubatch 只带 K（2B 上 6.3 MB），落进去的那个带 K + 整段 Query
   （31.2 MB）。旧规则 `bytes + max(bytes/2, 16 MiB)` 于是先给 16.8 MB、第二个 ubatch 要 17.1 MB 就整块重分配、
   最后一个 ubatch 要 47.9 MB 再重分配一次 —— 每次都是 `cudaFree`（设备级同步）+ `cudaMalloc` +
   `cudaMallocHost`，实测 **12 ms 和 52 ms** 都落在 prefill 里。改成 `max(2 * bytes, 32 MiB)`：
   第一次就拿够，**+5.3% pp**（同样的配对方法）。

顺带把打包里的两处标量循环换成紧凑快路径（KV cache 张量本来就是 token-major 连续的，转置是恒等）：
`pack` 108.7 → **97.5 ms**。收益只有一成，说明打包的大头是 `raw_` 的 mean-K 逐 token 累加，不是数据搬运。

**第四处优化：让重放落在一个 batch 里（2026-11-22）。** 重放跑 `[q_begin, prompt_end)`，
而 `prompt_end` 后面还有 chat 模板自己的几行：默认跨度 512 行 + 模板尾 7 行 = 519 行，
按 `n_batch = 512` 切成 **512 + 7 两个形状**。逐段计时（探针 `KVMEM_PROBE replay_batch`，4649 token、budget 2048、block 128）：

| 段 | 修前 | 修后 |
|---|---|---|
| 恢复 frontier（`state_seq_set_data_ext`） | 4.6 ms | 4.6 ms |
| 重放 batch #1（n = 512） | 29.8 ms | 31.2 ms |
| 重放 batch #2（n = 7） | **40.6 ms** | — |
| 合计（`replay_ms`） | **75.4 ms** | **50.6 ms** |

7 行比 512 行还贵，因为它是一个新形状：`sched_graph_realloc`（`nodes = 1193, changed = 0`，
即"尺寸没变却重分配"这一类）+ `ggml_backend_synchronize` 排空设备 + `ggml_gallocr_reserve_n`。
判据是同一格换 `-ub 640`（一次 519 行、单个形状）时重放降到 31.7 ms，也就是说这 40 ms 买的不是计算。

修法在 server 侧：`q_begin` 再让出模板尾的行数（`n_batch − (task->n_tokens() − q_end)`），
使 `prompt_end − q_begin ≤ n_batch`。同一二进制 `KVMEM_REPLAY_FIT=0/1` 五轮配对（4649 token）：
**731 → 660 ms**（pp 6353 → 7039，4/5 轮更快），对 `port-nokv` 的比值从 1.65 收到 1.52。
⚠️ `slot.prompt` 在那个位置还是空的（prompt 在后面才装载），所以长度取自 `slot.task->n_tokens()`；
它只可能高估（前缀被复用时真正的 prompt 更短），高估只会让打分跨度更小，是安全方向。

**第五处：KQ mask 定宽，去掉 prefill 里的设备级 drain（2026-11-22）。** `attn_inp_kq_mask` 是
`[n_kv × n_tokens]`，而不是 `n_kv` 在 KVMem 路径上等于**按占用补齐**的 cell 数：一次 prefill 里它每个
ubatch 都长一截（512 → 1024 → … → 3072），每长一次就让 `ggml_gallocr_alloc_graph` 对这一槽返回失败，
调度器的回答是 `ggml_backend_synchronize` + `reserve_n` —— 也就是**在 prefill 中间排空一次设备**。
nokv 上没有这个问题：普通 cache 在初始化时就把 cell 全部占住，宽度不变。

| 谁 | 一个 4649-token prefill 里的 `graph reallocation` | 说明 |
|---|---|---|
| `port-nokv` | **0** | mask 宽度恒为 `n_ctx` |
| `--kvmem`（旧） | **8** | 宽度跟着占用走，每步一次 |
| `--kvmem`（现在） | **5** | 池过半就按池宽钉住，后半程不再变 |

判据是 `-lv 4` 的 `graph reallocation: nodes = 1193 (changed = 0)` 行数 + `ggml-alloc` 里的
`src attn_inp_kq_mask ... size_max=… need=…`。修法在 `llama_kv_cache::get_n_kv()`：池子过半满之后，
把宽度钉成池宽（此时占用路径本来也要一步步走到池宽，钉住只多出后半程那点字节）；池子比这篇文档大时
不钉，宽度继续跟着占用走。

- **吞吐上是平的**：同一二进制 `KVMEM_MASK_WIDE=0/1` 配对，3 轮 680 → 677 ms、5 轮 688 → 708 ms，
  都在这一晚的 ±5% 噪声里；24 格答案 **24/24 逐格相同**（56/72）。
- **为什么不做成"一直按池宽"**：`--kvmem-budget 8192`（池 9216 > 文档 4649）实测 **574 → 626 ms（−8%）**，
  多出来的 mask 字节（每 ubatch 9.4 MB 而不是平均 4.7 MB）比省下的 drain 更贵。这就是"过半才钉"的理由。
- 想彻底消掉剩下的 5 次，只能让**预留**就按池宽做（运行时宽度不变），但预留图不走
  `get_n_kv()` 的 reserve 分支（实测 12 次调用里 `probe=0`），所以只能在 mask 宽度上做，就得付上面那笔字节。

**试过但更慢的：把 mean-K 累加搬到 GPU。** 用现成的 `kvmem_meank_add/d2h`（decode 路径在用）按
（块段 × 层）切片，`pack_ms` 从 **55.2 → 0.5 ms**，24 格答案 56/72 与逐格命中全部不变
（与宿主求和的相对差 1.7e-7，只因参照用了 double）。但 **wall 反而慢 5%**（656 → 688 ms，4/5 轮更慢）：

| | pack_ms（3072 token） | 最长一次 `harvest_flush` | prefill |
|---|---|---|---|
| 宿主累加 | 55.2 | 12.8 ms | 656 ms |
| 设备累加 | **0.5** | 11.0 ms | **688 ms** |

原因很直接：宿主打包本来就藏在 GPU 计算下面（worker 从不是瓶颈，flush 最多等 12.8 ms），
而设备求和必须在 compute 与下一个 ubatch 的拷贝之间插一道 event 栅栏 —— 省下的是隐藏时间，
付出的是关键路径。已经撤掉，代码里没有留开关。

留在台面上的搬运（都不在主机等待上）：

- **整块 V 收集 30 ms/请求**：每 ubatch 4 个新满块、6 层，合计 49 MB。这 49 MB 要过一次 D2H
  （实测 ~2 GB/s）再加一次同样大小的宿主 memcpy 写进 `raw_` 的 RAM tier 记录，也就是 V 镜像"落在宿主"
  这件事本身的价格，不是流程上的浪费。要动它就得改 V 的存放位置，不是这条路径的事。
- **worker 上的 98 ms 打包**：已经不在关键路径上——主线程 `submit` 现在只有 1.8 ms，worker 从未成为瓶颈，
  所以这 98 ms 是藏在 GPU 计算下面的。真要省，得把 mean-K 累加搬到 GPU（`keys.mean_k()` 现在从宿主读）。

顺带否掉一个假设：一度测出"捕获全关时适配层还有 -11.7% 固定开销"，但换上 `llama-cli` 单进程重测
（同二进制、4 轮配对）是 -13.5 / -6.9 / -6.6 / **+9.2%**，落在噪声带内；两个适配层入口实测各 0.03 ms
每 ubatch。所以这条不成立，不留待办。

**量这套东西只能用同二进制内的配对差。** 捕获开/关的绝对差在两次会话之间从 -30% 漂到 -14%
（同一份二进制、同样的格），所以下面这类跨小时的数字不能横向比；页面上的百分比都是配对中位数。

还没做、但已经量化清楚的方向（两条已试，结论在上一节）：

1. **整块 V 收集**（~30 ms/请求，主线程）：只有"刚满"的块才需要，可现在每块每层一次 enqueue，
   还要 GPU→pinned→GPU 中转。若 `raw_` 能给出设备侧目标指针，这批就能变成纯 D2D gather。
2. **`--kvmem-method recency` 的对照**：这里没有浪费可省——`want_q_capture()` 已经按 `method_ == 1`
   门控，recency 少掉的只是 Q；K 仍然要捕获，因为驱逐/回迁要靠 `raw_` 的宿主 K。
3. ~~宿主打包的两个尾巴~~ ⇒ 已测，**不是收益点**：`pack` 55.2 ms/请求确实存在，但 `harvest_flush`
   只等过 12.8 ms（它藏在 GPU 计算下面）。搬到 GPU 反而 +5%，见上一节，已撤回。
4. **"尺寸没变却重分配"** ⇒ 已查清并部分修掉（mask 定宽，8 → 5），见上一节：每次 1.6 ms，
   nokv 是 0 次；剩下 5 次只能靠"预留按池宽"消掉，而预留图不走 `get_n_kv()`，代价是 mask 字节。

#### MTP：draft 必须跟着重放，而且重放完不能再发「目标已还原」

draft 池里**有** prompt KV：prefill 的每个 ubatch 都会同步给它。所以 replay 重写 `[qb, prompt_end)` 时
它必须跟着重写，两块都要做对，少一块就会把整段 prompt KV 弄丢：

1. **每块 `llama_process(target)` 之后调 `common_speculative_process()`**。它内部会先把 draft 回退到
   本块首位置（M-RoPE 不允许覆盖已存位置），再喂同样的行。
2. **重放成功后不要再发 `target_restored_without_draft`**。那条事件会清掉 MTP carry 位，下一批 preflight
   就判成 `target_only`，而 `target_only` 是 `seq_rm(dft, -1, -1)` —— 把 draft 的 prompt KV 整段清空。

第 2 条是真正难找的那个：只做第 1 条时，命中率和答案都已经对了（`target_only` 发生在生成首步，
不影响 target 的 logits），只有吞吐塌下去。判据在 draft 侧的位置前沿上：

```
mode=cold_or_retained first_pos=4060 dft_pos_max=4059 ready=0   <- 生成首步：carry 失效
mode=cold_or_retained first_pos=4064 dft_pos_max=-1   ready=1   <- target_only 已把 draft 清空
```

`ready=0` 就是第 2 条干的。修掉之后同一格是
`first_pos=4060 dft_pos_max=4059 ready=1`，draft 接着往 4063、4067… 走。

2B Q8 + `mtp-Qwen3.5-2B-Q8_0`，24 格（`on` 两遍逐格相同；`off` 一遍同批次测）：

| 配置 | 命中 | pp | tg | draft 接受率 |
|---|---|---|---|---|
| replay 关 | 52/72 = 72.2% | 5156 | 210 | 0.846 |
| replay 开，只跟随（仍发还原事件） | 57/72 = 79.2% | 4601 | **168** | **0.545** |
| replay 开，跟随 + 不发还原事件 | **57/72 = 79.2%** | 4524 / 4807 | **207 / 211** | **0.855** |

⇒ 质量收益（配对 5:1，p=0.22）与 tg 都不受影响：**tg 与关掉 replay 时相同**，pp 少 12% 是因为重放的
512 行本来就在第一次 prefill 里算过一遍。之前记的「tg −17%、接受率 0.545、缺 draft carry」全部撤回——
那不是 carry 的问题，是这条事件。0.8B（无 spec）路径不受影响：8 格命中与答案逐格相同。

#### 媒体跨度：拒绝，而不是半重写

Query span 跨图时不 replay，退回首遍视图。这条路径的判据在 trace 里连着两行：

```
KVMEM_TRACE media_rows query=[4388,4900) rows=4907 last_row_pos=4666: rows and positions diverge, scoring in row space
D slot  operator (): id  0 | task 4 | KVMem replay skipped: media inside [4388,4907)
```

（第二行是 `SLT_DBG`，要 `-lv 5`；第一行是 adapter 说明行号≠位置、打分在行空间做。）2B + `mmproj` 实测：
`n_prompt=4905`（图占 258 token），三条记录全中；`KVMEM_REPLAY=0` 对照答案相同，即拒绝是设计内的边界，
不是丢功能。vision 的 chunk 解码走 mtmd，replay 的 rebuild 不复现它，硬做只会把图的行写坏。

### 质量：PPL（200 kB 中文、`-c 8192 -b 512 -fa on`、f16、每单元 3 轮）

| 单元 | PPL | 对 `old` |
|---|---|---|
| `old` | 12.0629 ± 0.18749 | — |
| `port-off` | 12.0629 | ±0 逐位同 |
| `port-nokv` | 12.0629 | 逐位同 |
| `port-kv`（budget 2048） | 12.3027 ± 0.19200 | **+2.0%** |
| `port-kv --kvmem-method recency` | 12.3027 | 同上，逐位同 |
| `fork-nokv` | 12.0627 | −0.002% |

每单元 3 轮，三轮逐位相同（PPL 是确定性的），11-22 复测与 11-15 逐位相同（同一批数值）。
`recency` 与 `retrieval` 仍逐位相同，是因为 **PPL 从不跑检索轮**
（每个 chunk 先 `memory_clear`，末尾的检索影响不到本 chunk 的 logits），这是口径限制而非实现等价——要验检索方法只能看上面那张 server 表。
同理 `KVMEM_REPLAY` 在 PPL 上不可见（没有检索轮就没有重放），本轮 `port-kv` 那一行是 `KVMEM_REPLAY=1` 跑出来的。

### 结论

1. **移植没有引入退化**：`port-off` 对 `port-nokv` 是 pp +0.2% / tg +1.8% / PPL 逐位相同；
   `old` 对 `port-nokv` 是 −0.2% / +3.1%。三者全在噪声地板内（这一晚 tg 单轮摆幅到过 262–280）。
2. **真驱逐的代价**：bench **−18.1%（replay 关）/ −17.4%（replay 开）prefill**、decode +1.6 ~ +2.8%；
   server 4649 token **−34.4%（默认参数 = 自动截尾 + 单批 replay）**、`−39.4%`（重放拆两批，`KVMEM_REPLAY_FIT=0`）、
   `−28.7%`（同跨度但 replay 关）。bench 与 server 的差别来自检索轮：bench 只有压力搬运，
   server 还多一次打分的驱动捕获与 512 行的重放。
3. **语义**：port 现在**不再整轮拒绝**——装不下就把 Query 截到尾部（≤512，`93756cdf7`）继续检索；
   fork 在同样 2048 预算下应用一个 `rows=512` 的计划。两棵树的决定点是同一个：截尾而不是硬裁。
4. **质量：缺的是一个阶段，不是预算**。24 起点矩阵上 port 59.7% / fork 79.2%（p=0.018）；补上 Query replay 之后，
   **同样的 16 页**就是 56/72 = 77.8% 对 fork 57/72（p=1.00），配对 16:2（p=0.0103；56 那一行是单批重放的
   跨度，见下面第 7 条，关掉 `KVMEM_REPLAY_FIT` 就是 58/72）。代价是 4649-token prompt 上 pp −16.6%
   到 −24%（重算 512 token，7378 → 7039 → 6152 t/s，取决于重放落在几批）。
   11-15 那轮"20 页（2560 token）追平 fork 的 16 页"的读法不再需要：16 页就够了。
5. **`port-off` 这一行现在有当天的数据**：`-DLLAMA_KVMEM=OFF` 曾因 `C2065` 编不过（已修），
   11-22 又随源码前进重建了一次，`kvmem_store_create` 计数 0。
6. **MTP 下 replay 不吃速度**：draft 侧跟着重放、并且重放成功后不发 `target_restored_without_draft`
   （发了会让下一步走 `target_only`，把 draft 的 prompt KV 整段清空）。修好后 2B+MTP 的 24 格：
   命中 52 → 57/72，tg 210 → 207/211（无退化），pp −12%（重放的 512 行）。
7. **开 kvmem 的 prefill 代价可以逐段拆、也可以逐段砍**：打分 77 → 7.3 ms、捕获的 163 ms 主机等待清零、
   捕获缓冲不再在 prefill 里重分配、重放按驻留指纹跳过、重放落在一个 batch 里（**+9.7%**，五轮配对
   731 → 660 ms）、压力搬运只占 4%。两处捕获改动各自由同一二进制内的配对实验给出 **+8.6%** 与
   **+5.3%**（相乘约 +14%），捕获与"全关"的差距因此收到 -14%。
   默认参数在这台机器、这一格上的净值：`port-nokv` 443 ms → **660 ms（−34.4%）**，
   其中捕获 ~150 ms、重放 ~50 ms；捕获那 150 ms 的两笔大头是宿主打包的两个尾巴（各 20 ms）
   与 D2H/V 镜像，见上一节「还没做、但已经量化清楚的四个方向」。
8. **失忆率（27B，8 针 × 3 轮）**：`--kvmem` 真驱逐、**默认参数** 是 22/24（8.3%），fork 24/24（0%），
   no-kv 基线 24/24；replay 关掉则掉到 18/24（25%）。修复前默认参数是 9/24（62.5%）——
   那时它会整轮拒绝检索（`93756cdf7`）。详见「检索质量：失忆率实测」一节。

### 复现

```bash
# 基准（热身后 5 单元轮换 4 轮 + 中位数，含 replay 两臂）
bash tmp/cmp/retest2.sh bench
# server（每格先预热再测，7 格轮换 4 轮）
bash tmp/cmp/retest2.sh server
# PPL（每单元 3 轮）
bash tmp/cmp/retest2.sh ppl
# 质量：24 起点检索矩阵（分辨率足够的那个），五臂一条命令跑完
bash tmp/cmp/qmatrix5.sh
#   单臂：KVMEM_REPLAY=0|1 PORT_BASE=<port> bash tmp/cmp/qrun2.sh <label> 2048 <exe> [args...]
#   重放单批的 A/B：同一二进制只切 KVMEM_REPLAY_FIT=0（旧跨度）/1（默认）
KVMEM_REPLAY_FIT=0 python tmp/cmp/matrix.py ab-split-r1 9360 14000 200 "$PWD/tmp/wt-kvmem/build/bin/llama-server.exe" \
     --kvmem --kvmem-budget 2048 --kvmem-block-tokens 128 --kvmem-gen-reserve 1024
# 打分跨度扫描（128/256/512/1024，同时给命中与 pp）
bash tmp/cmp/qmsweep.sh
#   截尾生效的判据（要 -lv 4 + KVMEM_TRACE=1）：grep span_clamped
# 失忆率（27B，8 针 x 3 轮）
bash tmp/cmp/amnesia.sh r1        # port-kv + qm512 + replay 开
bash tmp/cmp/amnesia.sh r0        # 同上，KVMEM_REPLAY=0
bash tmp/cmp/amnesia.sh fork      # fork（默认块宽 128）
bash tmp/cmp/amnesia.sh nokv      # 不用 --kvmem
#   块宽对照要另传 --kvmem-block-tokens 128 / 32，见下表
#   默认参数（不带 qm）的那一臂：--kvmem-budget 2048 --kvmem-gen-reserve 1024
python tmp/cmp/setdiff.py                                # 选中集 vs 探针页的列联表
# MTP 路径（2B + nextn 头）：MODEL_PATH 覆盖 qmatrix/lpmatrix 的模型
MODEL_PATH=C:/Users/yj/models/Qwen3.5-2B-Q8_0.gguf bash tmp/cmp/qrun.sh mtp24 2048 \
     "$PWD/tmp/wt-kvmem/build/bin/llama-server.exe" --kvmem --kvmem-budget 2048 \
     --kvmem-block-tokens 128 --kvmem-gen-reserve 1024 --kvmem-query-max 512 \
     --spec-type draft-mtp -md C:/Users/yj/models/mtp-Qwen3.5-2B-Q8_0.gguf
# prefill 的账：三档预算 + 两个计时面（捕获/打分/重放分别落在哪一段）
KVMEM_PERF=1 KVMEM_REPLAY_TIME=1 python tmp/cmp/qmatrix.py perf <port> 2048 2000 \
     "$PWD/tmp/wt-kvmem/build/bin/llama-server.exe" --kvmem --kvmem-budget 2048 \
     --kvmem-block-tokens 128 --kvmem-gen-reserve 1024 --kvmem-query-max 512
#   KVMEM_RETR_SUM  → total/flush/score/plan/stage_out/…
#   KVMEM_REPLAY_TIME → restore/loop/total（每次重放一行）
#   budget 16384 时捕获整体关闭，budget 8192 时池成窗口、捕获打开；两档之差就是捕获机制的钱
# 媒体跨度：单轮带图请求（LOGV 控制 -lv，replay 的拒绝行是 SLT_DBG）
MODEL_PATH=C:/Users/yj/models/Qwen3.5-2B-Q8_0.gguf LOGV=5 python tmp/cmp/media.py media 9311 \
     "$PWD/tmp/wt-kvmem/build/bin/llama-server.exe" --kvmem --kvmem-budget 2048 \
     --kvmem-block-tokens 128 --kvmem-gen-reserve 1024 --kvmem-query-max 512
# 单测
cd tmp/wt-kvmem/build && ctest -R "test-(kvmem|backend-rebind|pinned-kv-tier|memory-contract|native-kv-payload|host-kv-storage|spill-io|tiered-kv-storage|native-residency|kv-content|prefix-checkpoint|mean-key-retrieval|graph-recall|query-policy|raw-kv-store|conversation-store)"
```

17 个 kvmem 单测全过。本轮日志在 `tmp/cmp/retest2/`（`bench_<unit>_r<N>.log`、`server7.jsonl` + 每格 `.log`、
`ppl_<unit>_<N>.log`、`q_<arm>_<budget>.jsonl`），失忆率在 `tmp/cmp/quality/a-*.json` + `.log`。

## 引擎状态镜像：整段镜像拒绝，checkpoint 镜像支持

一条线索下的两个结论。「整段镜像」指按 token 命名、要能在**另一个** memory 里重建的历史（prompt cache、`--slot-save-path`、整段 context 保存）；「checkpoint 镜像」指槽自己活内存的时间点快照（`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`），只在 prompt 的公共前缀下取用。

### 整段镜像：写入与读取都拒绝

**症状**（用户报告，27B + MTP + hybrid，`--kvmem --kvmem-budget 1000 -np 1 -b 512`）：对话 A 正常，切到对话 B 正常，**切回 A 就永久失败**——

```
E restore_partial_tail: tail block 257 keeps 4 committed rows the packed store lost
E init_batch: failed to prepare KVMem attention ubatches
...（每步 batch 减半重试）
E srv decode: Context size has been exceeded.
```

**根因**：`llama_context` 的状态镜像是 attention cells，对普通 cache 那就是全部历史；对 KVMem 只是**池窗口**（例如 1248 行），真正的历史在 host packed store（打包后的 K/V + 块映射）+ recurrent 半边。镜像存出来再读回去，cells 声称 frontier = 64234，而 store 里只有 8228 行来自更短的那段对话，被恢复的 frontier 的 packed 字节根本不存在 ⇒ 每次 decode 死在 `restore_partial_tail()`，槽再也回不来。

**修法**（两侧都 fail closed）：

- `llama_memory_kvmem::state_write/state_read` 直接 `throw`（`llama-memory-kvmem-mtp.cpp` 同）。写入端抛 ⇒ `state_seq_get_size()` 走 dummy io 捕获返回 0 ⇒ 所有调用方跳过保存（server 打 `prompt cache save skipped: zero-size state`）；读取端抛 ⇒ 任何外来的镜像（别的构建、`--prompt-cache`、`--slot-save-path`）**半恢复失败**而不是半恢复成功。
- server 侧 `server_slot::state_image = !params_base.kvmem.enabled`，关掉三段：`prompt_save()`（返回 `failed`，保留活槽）、`create_checkpoint()`（含 `--checkpoint-recurrent-prev` 那条唯一剩下的调用者）、以及活循环里**唯一**的 inline 捕获点（`do_checkpoint = do_checkpoint && slot.state_image`）。只关前两段不够：inline 段仍在造镜像，日志里照样出现 `CHECKPOINT_ATTN_ONLY_TRIM` 与 `[frontier-validity] skipping context checkpoint`。

**验证**：

| 场景 | 结果 |
|---|---|
| 2B，80k token 长文 → 不相交文档 → 切回（`tmp/cmp/conv_disjoint.py`） | `LONG1/LONG2/LONG3/OTHER1` 全 ok，`restore_partial_tail｜failed to prepare KVMem｜Context size` 计数 **0**（修前 LONG2 就 500） |
| 27B（MTP + hybrid），80484 token，四轮切换 | 四轮全 ok、错误 0 |

**代价**：槽没有镜像就只剩「活内存」，切回旧对话要整段重算。27B / 80k token / `--kvmem-budget 1000` 实测 **34 t/s ≈ 39 分钟**一轮。多会话要复用只能走 `--kvmem-conversations`；`-np 2` 不行——KVMem 在 `n_seq_max > 1` 时**整体不启用**（`KVMem requires n_seq_max=1`，静默退回原生 cache）。

### checkpoint 镜像：hybrid / recurrent 模型支持，顺带修掉「每轮全量 prefill」

**症状**（用户报告，27B + MTP + hybrid，`--kvmem-budget 1000`）：没有崩，但**单会话每轮都整段重算**，像完全没有 KV cache；`--kvmem-conversations 4` 也一样。

**根因**：上一节把三段镜像入口全关了，其中 `create_checkpoint()` 是**回滚**唯一的依靠。聊天界面几乎从不把 assistant 的回复原样送回：它剥掉 reasoning、裁掉尾巴、改标点。任何这种改写都让缓存 prompt 在回复开头处发散，server 于是 rewind 到「最新的、位置不晚于发散点的 checkpoint」——而 checkpoint 被关掉之后一个都没有，只能 `forcing full prompt re-processing`。

**修法**：KVMem 的历史在 host store 里，store 是持久的，所以 checkpoint 真正需要携带的只有**store 造不出来的那一半**：recurrent（GDN/SSM）行，以及它们所属的 frontier。

```c
struct kvmem_ckpt_header { uint32_t magic, version, frontier, has_recr; };
// write: header + recr_->state_write(...)     （store_n_tokens() 就是 frontier）
// read : header; frontier 超出本 store ⇒ throw（fail closed，丢掉这个 checkpoint 重算）
//        truncate_cached(frontier); recr_->state_read(...)
```

配套三处：`seq_pos_min/seq_pos_max` 在有 recurrent 半边时按 `llama_memory_hybrid` 的规则折叠（min 取两者较大、max 取两者较小 ⇒ `seq_pos_min == seq_pos_max == recurrent frontier`），否则 server 的 `[frontier-validity]` 会拒绝每一次捕获；`server_slot::checkpoint_image = !kvmem || (hybrid || recurrent)`（attention-only 模型在 KVMem 下依然无法 checkpoint：store 造不出 attention 那一半）；`state_image` 继续为 false，所以 prompt cache / slot-save 仍然静默跳过。

**A/B（27B，12k 字符文档 ≈ 4070 token，每轮把回复改写成前 8 个字符，`--kvmem-budget 2048`）**：

| 配置 | turn 1 | turn 2 | `forcing full prompt re-processing` |
|---|---|---|---|
| `--ctx-checkpoints 0`（= 修前） | pp=4098（8.1 s） | pp=4127（6.7 s） | 2 |
| 默认（checkpoint 镜像可用） | **pp=32（1.9 s）** | **pp=34（1.3 s）** | **0**，`restored context checkpoint` ×2 |

用户的 `--kvmem-budget 1000` 同形复测：`pp=32 / pp=34`，`full_reprocess=0`。2B（Qwen3.5-2B，同样是 hybrid）也是 `pp=30 / pp=30`、`restored ×2`。

⚠️ 只对 hybrid / recurrent 成立。dense 模型在 `--kvmem` 下依旧没有 checkpoint，改写回复仍然整段重算——那部分要等 store 能造出 attention 半边（或 fork 的 replay）才有解。

**checkpoint 捕获的代价（27B，冷启动 4070 token prefill，各 3 次）**：开 checkpoint 472/494/492 t/s（中位 492），关掉 493/486/486（中位 486）——**噪声内，不付钱**。

## 多会话切换仍然全量 prefill：三个账本 bug（`--kvmem-conversations`）

**症状**（用户报告，27B + MTP + hybrid，`--kvmem --kvmem-budget 1000 -np 1 -b 512 --kvmem-conversations 4`）：两个对话来回切，每次切回来都整段重算；而且在**同一个**对话里接着问，也仍然整段重算。日志无告警。

三个缺陷叠在同一处，都是「账本和真状态脱节」，但一个套着一个，修掉上一层的症状才能看见下一层。

### 一、checkpoint 环跨 store 泄漏（把别人的 rec 状态装进自己的位置）

`slot.prompt.checkpoints` 是**它此刻这块 memory** 的快照环，但它跟着 slot 走，不跟着 store 走。`kvmem_conv_bind` 换 store 时只换了 token 账本（`slot.prompt.tokens = rows`），环原样留着：

```
edit/divergence sample (cached/incoming/lcp/reusable/rewind/append/cache_prompt) = (4125/4150/4125/4125/0/25/1)
checking checkpoint with [62, 62] against 4125...
restored context checkpoint (pos_min = 62, pos_max = 62, n_tokens = 63, n_past = 63, size = 149.626 MiB)
```

store 已经把 A 的 4125 行 restage 回来了，但扫描从 `rbegin()` 找**最新且 `pos_max` 不超过公共前缀**的那条——对话 B 留下的 `[62]` 正好符合，于是 `n_past = 63`，剩下的 4087 行全部重算。这个 62 来自 B 的上一轮（B 的 prompt 只有 67 token，所以它的 checkpoint 落在 62）。

⚠️ **这同时是个正确性缺陷**，不只是慢：hybrid/recurrent 的 checkpoint 携带的是**生成它那一刻的 recurrent 行**。把 B 在位置 62 的 GDN 状态装进 A 的位置 62，数值上通得过、日志全绿，输出却是错的。修前一直靠「别的对话恰好没有 checkpoint 落在前缀里」这种运气。

**修法**：换 store 成功时把环清空（`checkpoint_ring_changed()` 走 lifecycle 记账，和 `prompt_clear()` 同一条路）。环属于刚刚被 detach 的那块 memory，换了就不该留着。

**2026-11-02 更新：环改由 store 自己保管（`baddeb834`）。** 清空是正确但不免费的——hybrid 的回滚只能靠快照，环一清，任何「丢掉上一轮答案」的切换（浏览器刷新、两个对话来回切）就只能整段重算（27B / 48k 实测 80–100 s）。现在改成**换走时把环停到那个 store 名下**（`kvmem_conv_state::rings`，每个 store 留最新 2 个，载荷是共享 buffer 所以只是链表拷贝），换回时**只从自己那个 store 停放的环里取**快照恢复，恢复完即作废。判据：

```
conversation store 0 resumed: dropping 15 of 8036 rows was refused by the recurrent half,
so the newest usable checkpoint (8017 rows) was restored; 4 rows remain to prefill
```

2B 实测切回来 `prompt_n` 8036 → **4**、`prompt_ms` 1161 → **37**；19.7k 行那次还原答案与首轮逐字节相同。⚠️ 恢复失败或该 store 没停放环时**仍退回整段重算**，日志会说是哪种。

### 二、清空环之后：扫描把「空环」读成「没有可复用状态」

清空环之后 `n_past` 反而被清零——`do_reset = it == checkpoints.rend()` ⇒ `forcing full prompt re-processing`。原因是这套扫描只认 checkpoint：**hybrid/recurrent 模型的 token 前缀本身不被信任**（rec 状态无法从 token 重建），所以环空 = 全丢。

但换 store 这一路是有真状态的：store 带回整个前缀，attach 时 `KVMEM_STORE_ATTACH_RECURRENT` 把 recurrent 半边一起装回来。扫描看不到这件事，得明说：

```cpp
// 只在公共前缀恰好落在刚装回来的那个 frontier 上时才信它
const bool kvmem_store_frontier =
    slot.kvmem_swap_frontier > 0 && n_past > 0 && n_past == slot.kvmem_swap_frontier;
if (!complete_vbr_import && !kvmem_store_frontier && pos_min >= pos_min_thold) { ...扫描... }
```

`kvmem_swap_frontier` 一次性（bind 入口清 -1，`prompt_clear()`/`reset()` 也清），所以它只可能施惠于「同一请求、同一 frontier」。不能只把 `do_reset` 翻成 false——那样 `it` 仍是 `rend()`，下面会拿空迭代器去 `state_seq_set_data`，日志里是 `E state_seq_set_data: error loading end of buffer` 后紧跟 `restored context checkpoint (n_tokens = 0)`，等于什么都没修。

### 三、`last_n_gen` 在记账时恒为 0（长答案的对话永远匹配不上）

前两层修完，短答案（24 token）已经正常，但**长答案仍然整段重算**：

```
conversation store 0: rows=3661 lcp=45 tail=3616 slack=64   ← A 的 store 被判成「分支」
slot 0 started a fresh conversation store 1                  ← 于是造了新 store
```

`kvmem_conv_bind` 允许 store 比请求多出一截尾巴（客户端不带回复回来时的答案），界限是 `tail <= max(0, last_n_gen) + 64`。而 `last_n_gen` 取的是 `slot.stats.n_gen`——**`release()` 在下一轮 launch 之前就 `reset()` 过了，`stats = {}`**，所以记进账本的永远是 0，界限退化成固定 64 行。答案一超过 64 token（真实对话全都超），store 就被当成「另一段对话」拒掉，于是每次切回都新建 store、整段重算。

**修法**：`reset()` 里在 `stats = {}` 之前把上一轮的生成数存进 `slot.kvmem_last_n_gen`，记账时用它。

### 效果（27B，两对话交替，A1/B1 冷启动，A2 起应为增量）

| | A1 | B1 | A2 | B2 | A3 | A4 |
|---|---|---|---|---|---|---|
| 修前 | 4126 | 67 | **4087** | 28 | **4089** | 27 |
| 修后 | 4126 | 67 | **25**（4125 cached） | 25（66 cached） | 27（4149 cached） | 27（4172 cached） |

单轮墙钟从 8.1 s 降到 1.5 s，`tg` 不变（31 t/s）。A4 修前就是对的，因为 A3→A4 之间没有换 store（`best == st.bound`，环没被动过）——这也是这个 bug 只在「换回来」时出现、看起来像「偶发」的原因。

长答案 + 需要召回的场景（`tmp/cmp/conv_switch_probe.py`：A 文档开头埋码 → B 不相交 → 回 A 问码，贪心）：

| 配置 | A2 prompt_n | 答案 |
|---|---|---|
| 修前（`--kvmem-conversations 4`） | 3659（42 cached，= 整段） | **MISS**（答成别的片段） |
| 修后（同参数） | **260（3441 cached）** | **HIT**（`ZQ7K-3311`） |
| 参照（`--kvmem-budget 0 -c 4096`，不换 store） | 3659（42 cached） | HIT |

### 顺带验证的两个新架构（`tmp/cmp/archcheck.py`，2600 字符、`--kvmem-budget 512 -b 512`）

| 模型 | 提问轮 `KVMEM_QUERY` | `plan=` |
|---|---|---|
| Qwen3-VL-8B-Instruct | `rows=34 blocks=51` | `query_workspace history_pages=49 kept_history=18 workspace=2` |
| ERNIE-4.5-21B-A3B-Thinking | `rows=74 blocks=52` | `query_workspace history_pages=50 kept_history=18 workspace=2` |

ERNIE 的文档轮仍会打一次 `retrieval_skipped reason=no_query_rows query=[-1,-1)`（模板不声明 `message_delimiters`、冷启动不猜），提问轮正常。两模型的 `selected` 都是散开的块号，不是连续后缀 ⇒ 真检索，不是 recency 退化。

## 混合内存的尾部截断：两半必须一起退（`a7b42f96e`）

**症状**（用户报告，27B hybrid + `--kvmem-conversations 4`）：浏览器里两个会话，20k 那个刷新答案正常；切到另一个会话再刷新，**回复只有一两个字**；再点一次刷新，**整段 22853 token 全量 prefill**（36 s）。两个会话都会中招。

**日志指纹**（三条一起出现即确诊）：

```
conversation store 0 resumed, 22853 of 25195 rows kept
n_past was set to 22852
[frontier-validity] skipping context checkpoint: recurrent/hybrid
    seq_pos_min (25194) != seq_pos_max (22851)
find_slot: non-consecutive token position 22852 after 25194
```

**根因**：会话 store 的 frontier（25195）比本次 prompt（22853）长——浏览器把上一轮答案丢掉、重发同一段历史，所以要让出 2342 行。`llama_memory_kvmem_hybrid::seq_rm` 只把 attention 半边截到 22851，recurrent 半边仍停在 25194：

```cpp
// 改之前
if (n_rs > 0 && ... && p1x > rmax) {   // n_rs_seq 默认 0 ⇒ 这个门永远不开
    if (!recr->seq_rm(seq_id, p0, p1)) return false;
}
return attn->seq_rm(...);              // 只截了 attention
```

`n_rs_seq` 默认 0（没有任何回滚快照平面），所以对**一切**尾部截断都不问 recurrent 半边。于是模型拿着「22852 行的 attention + 25194 位置的 GDN 状态」解码：`find_slot` 报非连续，第一条 token 就 EOS；位置区间接不下来，checkpoint 被 `[frontier-validity]` 拒，环清空 ⇒ 下一轮 `forcing full prompt re-processing`。dense 模型不受影响（没有 recurrent 半边）。

**修法**（两处，缺一不可）：

1. `seq_rm` 改成 **fail closed**：凡 `p0n <= rmax < p1`（尾部截断越过 recurrent frontier）就必须先问 `recr->seq_rm`，它自己决定「窗口内回滚 / 窗口外拒绝」，拒绝时什么都不改。内部空洞（`p1 <= rmax`）仍然不碰 GDN——P4-2 的 replay 恢复依赖这一点。
2. `server-context.cpp` 的 store resume 路径原来**丢弃**了 `llama_memory_seq_rm` 的返回值。改成尊重它：保留 store 绑定（这些行仍是本会话的续写起点），但**不认领 frontier**，走 `prompt_clear()`（连带清两半和 checkpoint 环）再整段 prefill 一次。通用路径（`memory_seq_rm [%d, end) rejected; falling back to full prompt re-processing`）本来就是这么做的，只有 store 这条路忘了。

**代价**：每次「改写回复」要付一次全量 prefill，这正是 upstream 对 hybrid/recurrent 的契约（那条 `forcing full prompt re-processing` 消息就是为它写的）。想便宜就让 `n_rs_seq > 0`（MTP 回滚平面），窗口内的截断会走回滚而不是拒绝。

**证据**（Qwen3.5-2B-Q8_0，hybrid `attn_layers=6`、`n_rs_seq=0`，`--kvmem-budget 512 -np 1 -b 512 --kvmem-conversations 2`，1081-token prompt，三轮：建 store a → 用 b 顶掉 → 重发 a）：

| | 日志 | 回复 |
|---|---|---|
| 改之前 | `resumed, 1081 of 1084 rows kept` + `seq_pos_min (1083) != seq_pos_max (1079)` + `non-consecutive 1080 after 1083` | `\nAnswer with the number only.\nAnswer with the number only.\nA`（复读，27B 上直接 EOS） |
| 改之后 | `resumed, but dropping 3 of 1084 rows was refused by the recurrent half; re-prefilling the prompt` | `\n\n84`，与第一轮逐字节相同，无非连续告警、无 frontier 拒绝 |

复现脚本 `tmp/cmp/hybrid_trunc.py`（不提交）。`ctest -R "kvmem|conversation-store|memory-contract|prefix-checkpoint|server-prompt-cache|hybrid"` 9/9。

## Query 行是追加工作区，不是历史（`--kvmem-query-workspace`）

**问题**：`score_retrieval()` 的帧用 `frontier = total_tokens()`，Query 的尾部块既当评分对象又当历史候选，`query_fixed_blocks(retrieval_mandatory())` 把它们记进**同一份** `budget_blocks()`。于是「预算 18 块」实际是 16 块历史 + 2 块 Query —— 同预算下比 fork 少 2 块可召回历史（64 token @ block 32）。

**修法**：`preview_retrieval()` 的非前缀分支复用**前缀分支已有的那套记账**（`pick_scored_blocks(candidates, scores, required, budget)` + 之后把后缀当 append workspace 追加），边界取 Query 区间的首行：`explicit_spans_` 时是 `turn_spans_.query` 的最小 `begin`（server 路径），否则是 `--kvmem-query-begin`。`required > budget`（band 自己就装不下）时退回原来的 flat 记账，不抛异常——拒绝留给 `selection_fits()` 的回滚。

**证据**（0.8B，2600 字符 ≈ 1442 token / 45 块，`--kvmem-budget 512 -b 512`，budget 被 floor 抬到 576 = 18 块）：

```
off : KVMEM_TRACE plan=flat            selected=18 budget_blocks=18            # 16 历史 + 2 Query
on  : KVMEM_TRACE plan=query_workspace history_pages=44 kept_history=18 workspace=2 budget_blocks=18
```

`history_pages=44` 是 Query 之前的页数，`workspace=2` 是追加的后缀块（≥ 44 的部分），`kept_history=18` = 满预算。fork 同参数（`--kvmem-budget 512 --kvmem-block-tokens 32 -b 256`）选中 `0 1 2 3 4 6 7 13 14 15 16 17 19 28 42 43 44 45`，其中 42–45 是它自己的后缀工作区，历史 14 块 —— **它也是「历史拿预算、后缀追加在预算外」**，只是它的 suffix 更宽（包含 assistant 那轮）。

⚠️ fork 在 `-b 512 --kvmem-budget 512` 下会直接拒（`KVMem reselect failed: mandatory Query content exceeds its selection budget`，它的 budget 不做 chunk floor 抬升，port 抬到 576）。三方横比必须同 `-b`。

**为什么不移植 replay**：replay 相对 port 只有两处差异——(a) 后缀不占预算，(b) 评分帧里没有 Query。port 的 `query_score_frame()` 已把 sink/recent 带的 `score_mask` 置 0，Query 尾块本来就在 recent 带里不参与打分，所以 (b) 在 port 上不存在；(a) 用现成的 `pick_scored_blocks` + append 就能拿到，不需要 fork 的两段 prefill 执行器（那只是实现 (a) 的手段）。

**代价与实测**（3 轮轮换、每轮 3 个不同深度样本、取中位）：

| 配置 | pp (t/s) | tg (t/s) |
|---|---|---|
| `--kvmem-query-workspace off` | 7258 | 219.9 |
| `--kvmem-query-workspace on` | 7571 | 219.7 |
| fork（`-b 256`，不可与上两行横比） | 3130 | 83.0 |

- **tg 不变**（−0.1%）：decode 只多看 2 块 KV；0.8B 在这个规模是算力受限。
- **pp +4.3%**（逐轮 −8%~+11%，被噪声主导）：多搬 2 块 + prefill 注意力 span 变长。这是 fork 一直在付的钱，所以对 fork 是持平而不是退步。
- 驻留 +2 块 ⇒ 生成头寸少 2 块（`--kvmem-gen-reserve` 现在默认 1024 = 32 块，够；把它调到 1~2 块会让计划撞 `selection_fits()` 回滚到压力视图）。
- **端到端命中率不能当判据**：28 深度针扫掠 on 6/28；同一扫描里连「深度 3%（必然在 sink 带内）」都会答错，0.8B 自身不稳。判据用「可召回历史块数」：18 vs 16。

## 检索质量：失忆率实测，和两个把它放大的根因

在此之前所有「检索质量」的结论都只到块数（18 vs 16）。这一节是端到端的**失忆率**，并顺手挖出两个 bug——它们的破坏力远大于 block 记账那 2 块。

### 测法（三个必须先排除的假象）

一篇文档里埋 8 条格式相同、内容唯一的记录（`【档案 KD-i】…备用解锁码 XXXX`），均匀铺在 4%~92% 深度；每条只问一次，**命中 = 答案里出现该条自己的 4 字符码**。3 轮，每轮的 8 个码做随机置换（文档长度不变），于是同一深度在不同轮次坐不同码——24 次探针是 24 次独立抽样，不是 24 次同一个计划。

三个假象，踩过就知道必须排除：

1. **单轮请求 = 整篇文档当 Query**。`message_spans` 只认最后一个 user 轮；一次问完时 Query 就是文档本身，`mandatory` 盖住全部历史，每轮都 `mandatory_exceeds_pool` 回滚到压力视图——测出来的是 recency，不是检索。必须两轮（先发文档 + `已读`，再提问）。
2. **增量回合根本不重选块**（本节的根因一）。只追加 36 个 token 的回合如果没有重开检索轮，第一轮的计划就冻在那儿了，后面 7 问全在同一份工作集上答。
3. **架构没接 `kvmem_capture_q()`，评分永远是空的**（根因二）。

### 根因一：`begin_cached_turn()` 移植了函数，没移植调用点

`apply_retrieval()` 成功（或被跳过）之后 server 会 `llama_kvmem_pin_working_set()`，置 `retrieval_pinned_ = true`。而 `want_prefill_capture()` 里带 `!retrieval_pinned_`，唯一清它的是 `llama_memory_kvmem::begin_cached_turn()`——port 里**全树没有调用点**（fork 由 `tools/llama-kvmem-driver.cpp` 在每个缓存回合边界调用）。后果：**第一个回合之后，`want_prefill_capture()` 恒为 false，`llama_kvmem_apply_retrieval()` 一次都不再执行**，工作集冻死在第一轮的计划上。

判据是「本轮有没有计划行」，不是有没有报错——修复前的日志里 `KVMEM_QUERY`、`KVMEM_TRACE plan=`、`KVMEM_TRACE selected` **三个计数全是 0**，而请求全部成功返回。这种失忆没有任何报错。

修法：server 发布本请求的 span 之前，若 `slot.prompt.n_tokens() > 0`（这个请求在继续一个已有行的序列）就 `llama_kvmem_begin_cached_turn()`——flush 解码期 mean-K、丢掉冻结的 Query、清 `keep_selected_`、解除 pinned。**必须在 `set_request_span()` 之前**，顺序反了就把刚发布的 Query 冲掉。

同二进制 A/B（27B，5611 token 文档，`--kvmem-budget 2048`，8 针 × 3 轮）：

| ARM | 命中 | 失忆率 |
|---|---|---|
| 关掉 `begin_cached_turn()`（= 修复前，11-06 测） | 13/24 | **45.8%** |
| 打开（现在，默认：自动截尾到 512 + replay） | 22/24 | **8.3%** |
| 对照：`--kvmem-method recency`（只留 sink+最近块，11-06 测） | 13/24 | 45.8% |

修复前**逐位等于 recency**——这正是「计划冻在第一轮压力视图」的形状。代价是每回合重规划一次：追加回合的平均 prompt eval **1532 → 1619 ms（+5.7%）**。
「关掉」这一臂需要改代码才能切，11-22 没重测；「打开」的数字是当天的（22/24，比 11-06 的 23/24 少一格，
少的是最深那条 KD-7，答成了邻近记录 `B2XF`）。⚠️ 另一条更隐蔽的 recency 退化见下面 27B 表：
即使 `begin_cached_turn()` 在跑，只要冷启动的 Query 装不下预算，`kept=0 layers=0` 也会把整段退化成 recency。

### 根因二：`kvmem_capture_q` 原来只接在 qwen3 / qwen35 上（现已铺开，但铺开后又露出两个）

评分读的是 Query 自己的 K 行，而这些行由各模型 attention builder 里的 `kvmem_capture_q(Qcur, il)` 送进图。原先只有 `src/models/qwen3.cpp` 和 `qwen35.cpp` 调了它（各 1 处；K/V 同理）。**其它架构跑 `--kvmem` 时 `q_count_` 恒为 0，`score_retrieval()` 第一行 `if(!rows) return;` 直接返回**，计划退化成 sink+最近块，而日志里一切正常。实测同一个 8B 文档（Qwen3-VL-8B，30000 字符）在「有捕获」和「无捕获」之间就是 46% 与 4% 的差别。

现在接入的架构（`kvmem_capture_q/k/v`，一次调用约 6 行，位置在 norm 之后、RoPE 之前——评分比的是 **pre-RoPE** K）：

| 文件 | 架构 | 状态 |
|---|---|---|
| `qwen3.cpp` / `qwen35.cpp` | qwen3 / qwen3.5 | 原有 |
| `qwen3vl.cpp` / `qwen3vlmoe.cpp` | Qwen3-VL | 新增，**实测有 QUERY 行、选块散开** |
| `olmoe.cpp` | OLMoE | 新增，**实测同上** |
| `ernie4-5.cpp` / `ernie4-5-moe.cpp` | ERNIE-4.5（含 21B-A3B） | 新增，**实测同上** |
| `gemma4.cpp` | gemma-4 | 新增（该架构 SWA ⇒ KVMem 直接跳过：`W llama_memory_kvmem_maybe_create: KVMem skips SWA models`，钩子目前是惰性的） |

铺开之后，同一台机器上“勾了但不出行”的模型暴露了两个更隐蔽的问题：

**(a) 模板不声明分隔符 ⇒ 没有 Query span。** `q_begin` 来自 `slot.task->params.message_spans`，那是从 chat template 的 `message_delimiters` 里拆出来的；ERNIE-4.5 的模板没有这项 ⇒ `q_begin = -1` ⇒ 整轮不评分，日志只有一句 `retrieval_skipped ... query=[-1,-1)`。修法：span 为空且这个序列**已有** token 时，用「本轮新增、序列里还没有的那些行」当 Query（`[slot.prompt.n_tokens(), slot.task->n_tokens())`）；冷启动 prompt 不猜，留给 sink/recency。对 follow-up 轮，这就是 fork 里 replay 后缀工作区的那一段。

**(b) 捕获是异步的，评分读得太早。** prefill 捕获走 D2H 管道，`harvest_flush()` 才把行落进 `q_count_`；`apply_retrieval()` 原本先看 `q_count_`、再在回滚分支里 flush。于是**同一个模型会不会评分取决于调度器给它的张量挑了哪条捕获路径**：走同步路径的（Qwen3-VL、OLMoE）当场有行，走 batched D2H 路径的（ERNIE-4.5-21B）永远读到 0，打一句 `no_query_rows` 就退化成 recency，日志全绿。修法：`apply_retrieval()` 进门先 `harvest_flush()` 再判空。

判据（修复前后都适用）：**`KVMEM_QUERY` / `KVMEM_TRACE plan=` / `KVMEM_TRACE selected` 三个计数不能是 0**；`selected` 的块号必须散开（`0 1 14 15 30 40 41 42 43 44 48 49 51 65 69 70 84 85 86 87`），是一段连续尾巴就是 recency。三架构复测（2000 字符文档、`--kvmem-budget 512 --kvmem-gen-reserve 128`）：

```
OLMoE      plan=query_workspace history_pages=86 kept_history=18 workspace=2
           selected 0 1 14 15 30 40 41 42 43 44 48 49 51 65 69 70 84 85 86 87
Qwen3-VL   plan=query_workspace history_pages=39 kept_history=18 workspace=1
           selected 0 1 2 3 6 7 13 14 15 16 17 18 19 20 22 23 32 38 39
ERNIE-4.5  plan=query_workspace history_pages=40 kept_history=18 workspace=2
           selected 0 1 2 3 4 7 8 14 15 16 17 18 19 20 21 22 23 39 40 41
```

### 27B 实测（Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp，5611 token 文档 = 175 块 @32 / 45 块 @128，`-fa on -ctk/-ctv f16 -dev CUDA0 -b 512 -c 16384`）

预算 2048 + `--kvmem-gen-reserve 1024` ⇒ 池 3072 token，保留比例 55%（两侧同）。

| 配置 | 块宽 | 命中 | 失忆率 |
|---|---|---|---|
| 不用 `--kvmem`（全上下文注意力） | — | 24/24 | 0% |
| `--kvmem` 预算装得下（pool 9216 > 5611，不驱逐） | 32 | 24/24 | 0% |
| **`--kvmem` 默认参数，真驱逐**（`93756cdf7` 后：自动截尾 512 + replay） | 32 | **22/24** | **8.3%** |
| 同上，块宽对齐 fork | 128 | 14/24 | 41.7% |
| `--kvmem --kvmem-query-max 512`，replay 关 | 32 | 18/24 | 25.0% |
| `fork-kv`（块宽 128 = 它的默认） | 128 | 13/24 | 45.8% |
| **`fork-kv --kvmem-block-tokens 32`** | 32 | **24/24** | **0%** |

四件事值得单独说：

- **默认参数曾经是这组里最差的（9/24 = 62.5%），`93756cdf7` 修掉了**。修前：冷启动那轮 Query 就是整篇文档，
  `needs 175 of 62 blocks` ⇒ `prefill capture skipped`，整轮捕获没跑；此后每轮虽然都在跑检索轮
  （`KVMEM_QUERY` 48 行、`plan=query_workspace`），但 `selected` 是 `0 113 114 … 176` 这样一段**连续尾巴**
  ——sink+recency，8 根针里只有深于 0.54 的还在，逐深度命中 0/0/0/0/0/3/3/3。
  修法是**截尾继续而不是整轮拒绝**：`span_clamped query=[2,5598) -> [5086,5598) cap_blocks=62`，
  同一份日志的 `selected` 变成 `0 1 2 3 5 6 7 … 40 47 48 58 73 75`（散开），9 → 22/24，
  与显式 `--kvmem-query-max 512` 那一臂逐格相同。
  **为什么拒绝会毁整段会话**：`raw_->write_layer_mean_k()` 只在捕获的 harvest 里写
  （`llama-memory-kvmem.cpp:4341`，且要求这一轮有 Q 行）。捕获没跑 ⇒ 那 113 个被换出的页只有 K/V、没有 mean-K
  ⇒ 之后每轮的打分都在给"没有统计的页"排序 ⇒ 只能落到 sink+recency，而且不会自愈。
  ⚠️ 判据是 `selected` 的**形状**，不是 `kept` / `layers`：预算装得下的那次（8192）日志里同样是
  `kept=0 layers=0`，而它 24/24（那两个字段只在两阶段粗筛时才有意义）。
- **replay 在这条轴上也值钱**：18 → 22/24，四个格子全在 0.42–0.79 的中段，没有一格变差。
  27B 是 hybrid（`hybrid_attn`、`n_rs_seq`、`KVMEM_GDN_ALLOCATION mode=snapshots`），
  replay 退的是 attention + recurrent 两半，只退一半在 0.8B 上已经证明只有 75%。
- **块宽是比"哪棵树"更大的自变量**：同一棵树同一预算，32 块宽 22/24、128 块宽 14/24；
  fork 反过来也一样（32 → 24/24，128 → 13/24）。同样 3072 token 的池子，32 块宽能装 96 页、
  128 块宽只有 24 页，page-softmax 在页这一维做 softmax，页粗了就先被"摊平"。
- **同块宽下两棵树几乎等价**：32 块宽 port 22/24 对 fork 24/24（差 2 格，都是最深那条 KD-7）；
  128 块宽 port 14/24 对 fork 13/24（差 1 格）。所以"port 失忆率比 fork 高"这个说法只在块宽不匹配时才成立。

### 保留比例才是自变量（把预算换算成「留了文档的百分之几」）

12k token 文档（14656 token = 458 块）补了另一段（2026-11-06 测，本轮未重测——它测的是预算轴，与本轮改动的 replay 无关）：

| 保留比例 | 8.3% | 14% | 17% | 28% | 33% | 67% |
|---|---|---|---|---|---|---|
| 命中率（检索） | 33% | 58% | 79% | 88% | 96% | 96% |
| 命中率（recency，33% 处） | — | — | — | — | 54% | — |

绝对预算不说明问题，**预算 / 上下文比**才说明问题：留到 1/3 就能到 96%，留到 1/12 掉到 33%，而同样留 1/3 时纯 recency 只有 54%。所以「kvmem 会不会失忆」= 「预算够不够文档的 1/3」+ 「检索评分有没有真的在跑」。

### 缓解手段实测（12k 文档、预算固定 2048、基线 58.3%；2026-11-06 测，本轮未重测）

| 手段 | 命中 | 结论 |
|---|---|---|
| `--kvmem-sink-tokens 512 --kvmem-recent-tokens 512` | 41.7% | **更差**：钉住的两条带是从同一份 64 块里扣的，直接少掉 32 块可打分历史 |
| `--kvmem-graph-alpha 0.25` | 58.3% | 无变化（邻居扩散改变不到弃块决策） |
| `--kvmem-query-max 256` | 58.3% | 无变化（Query 本来只有 28 token，够不到上限） |

**只有加预算有效。** 想要「质量不掉 + 性能不掉」的可行组合是：把预算设到工作文档的 1/3 上下，并确认日志里没有 `no_query_rows` / `retrieval_skipped`；预算调不下来就只能接受边缘事实被丢。

### 失败形态：不是「不知道」，是「拿邻近记录顶替」

唯一那次 miss 里问的是 needle 1 的 `ZQ7K`，模型答 `P8VT`——同一轮 needle 5 的码。被驱逐的块不是让模型变空，而是让它用还在驻留的**同类记录**填空。这条对实际使用比命中率更值得警惕：失忆看起来像自信的错误答案，而不是「文档里没有」。

11-22 复测把这条拆得更细：失败有两种形态，而且**不是"哪棵树"的差别**。

| 丢的格子 | 问 | port 答 | fork 答 |
|---|---|---|---|
| port@32 仅有的 2 格（最深 0.92 的 KD-7） | `KD-7` 的码 | `B2XF`（KD-6 的码，还在驻留） | 命中（24/24） |
| 块宽 128 时两侧都常丢的 KD-4/6/7 | `KD-7` | 前两轮给邻近码，第三轮给「文档中未提供」 | 同样：前两轮给邻近码，第三轮给「文档中未提供」 |

⇒ **「拿邻近记录顶替」是主要形态**（同一张文档里 8 条格式相同的记录，丢掉哪一条都能被另一条填上），
而"明确说没有"只在两个条件下出现：问的是**最深**的那根针（0.92，紧挨着 recent 带），且**轮次靠后**。
两棵树在这条轴上的表现一致，所以不能据此说 fork 更"诚实"。
对实际使用仍然成立的那句是：失忆多数时候看起来像自信的错误答案，而不是「文档里没有」。

## 本轮改动对性能的影响（架构捕获 + checkpoint 镜像 + flush 竞态）

`llama-bench -m Qwen3.5-0.8B-Q4_K_M -p 4096 -n 128 -b 512 -r 3 -dev CUDA0 --kvmem --kvmem-budget 576`，同一构建目录、同一时段，改动前后各 3 轮：

| | pp4096 (t/s) | tg128 (t/s) |
|---|---|---|
| 改动前（`f9cf1326b`） | 12497 / 12388 / 12422 → **中位 12422** | 277.81 / 277.80 / 268.97 → **中位 277.80** |
| 改动后 | 12539 / 12535 / 12378 → **中位 12535（+0.9%）** | 278.80 / 279.55 / 277.11 → **中位 278.80（+0.4%）** |

两项都在 ±1.5% 噪声地板内 ⇒ **没有可测代价**。原因也直接：捕获钩子只在 `want_q_capture()` 为真时才往图里加节点（bench 不跑检索轮），checkpoint 只在 checkpoint 边界多写 recurrent 行，`harvest_flush()` 进门那次在无待处理拷贝时是空操作。

**退出时那条 `compute buffer size ... does not match expectation` 警告**（用户报的 `179.2345 MiB vs 149.2346 MiB`）= 30 MiB：`llama_context` 析构时比对「调度器实际分配」与「reserve 时估计」。reserve 用的是不含捕获节点的图，而真跑起来的 prefill 图里多出了 pre-rope Q/K 与 D2H 拷贝的输出（`ggml_set_output` + `ggml_build_forward_expand`），调度器因此把 buffer 长到了 179 MiB。**是诊断不是错误**：长出来的就是实际用的那份，不会泄漏，也不影响结果；`~llama_context` 只在退出时打一次。

## 已知限制

- **检索评分覆盖的架构**：`kvmem_capture_q/k/v` 已接在 qwen3、qwen35、qwen3vl、qwen3vlmoe、olmoe、ernie4-5、ernie4-5-moe、gemma4 上（gemma4 因 SWA 不走 KVMem，钩子惰性）。其余架构仍然 `q_count_` 恒为 0、计划退化成 sink+最近块，日志会打一次 `retrieval_skipped reason=no_query_rows` + 警告。要接一个架构只需在其 attention builder 里补三个调用（norm 之后、RoPE 之前）。
- **chat template 不声明 `message_delimiters` 时只能退到「本轮新增行」当 Query**（ERNIE-4.5 就是这样）：follow-up 轮会把 assistant 上一轮回复也算进 Query，比精确 span 略宽；冷启动轮不猜。
- **多模态（mtmd）已可用（`b275aa789`）**：图片 patch **共用同一个 M-RoPE 时间位置**（一行 `15,15,15,…`），这让"行账本"（append 序）和"位置账本"（cache 寻址）天然不相等，曾经整条图片请求 500（`tail block N keeps M committed rows the packed store lost` → `init_batch` 拒批）。两处修：① `prepare_ubatches` 早备好的行视图（`logical_pos = append 序`）现在**也交给调用方的 ubatch**，否则写 cell 的那一步用 `pos[0]` 覆盖，`gpu_kv_complete()` 永远对不上；② 被拒/中断的批会留下**孤儿 cell**（图片 chunk 先记 20 行被拒、再记 22 行，那 2 个 cell 没人清），`prepare_working_set` 进门按 `seq_rm_logical(0, store_n_tokens(), -1)` 扫掉 `logical_pos ≥ store 行数` 的 cell。判据：`PROBE_PUBLISH ... used == visible == store`、图片请求 200、纯文本 PPL 逐位不变（2B：8.7998 两侧相同）。
- **检索质量只由「预算 / 上下文比」决定**：留 1/3 命中 96%，留 1/12 只剩 33%；`sink/recent` 钉带反而更差（从同一份额度里扣）。`--kvmem-query-max`：在 12k 文档那组里 512 与默认（0）逐位同（95.8%，同一条 miss），**128 反而掉到 79.2%**——它把 128 token 以下的短 Query 之外的轮次也截了，冷启动轮因此拿到一个「装得下」的假计划并提前定住工作集。
- **装不下的 Query 现在截尾而不是整轮拒绝（`93756cdf7`）**：默认 `--kvmem-query-max 0` = 不截，但整段装不下时把跨度截到尾部（≤512）继续跑检索，日志 `KVMEM_TRACE span_clamped query=[a,b) -> [c,b) cap_blocks=N`。修之前是整轮 `prefill capture skipped`，那次 prefill 的页没有 mean-K，之后每轮只能落在 sink+recency——27B 上默认 9/24、修后 22/24，与显式 512 逐格相同。连一页都放不进的极端情形才打 `retrieval round skipped`。**跨度该设多大**：4649 token / budget 2048 / replay 开扫过 128 / 256 / 512 / 1024 = 61 / 60 / 58 / 46（72 分制），pp 6215 / 7098 / 6378 / 5685 ⇒ **≤512 三档分不出高低，1024 同时掉质量与速度**（尾部 1024 行里混进约 1k token 无关正文，把页分数摊平）。默认取 512（与 fork 的 driver 一致）；想换 256 先在**你自己那组形状**上复核，12k 多轮那组里小 cap 是更差的。
- **replay 默认开**（`tools/server/server-context.cpp:52`，只有 `KVMEM_REPLAY=0` 才关），所以所有"默认参数"的数字都已含重放；把它关掉只是 A/B 手段，不是运行配置。
- **LRFU 默认关，实测没有收益（27B、8 针 × 2 轮、`--kvmem-budget 2048`）**：

  | 配置 | 命中 | 失忆率 | 回滚 |
  |---|---|---|---|
  | 默认（page-softmax，LRFU off） | 15/16 | 6.2% | 1 |
  | `--kvmem-lrfu` | 15/16 | 6.2% | 1 |

  选块列表与基线**逐位相同**，连 miss 的那根针（深度 0.166 的 `ZQ7K`，第二轮被 `Y4KB` 挤掉）都一样。原因：这个 probe 里每根针最多被问两次，LRFU 依赖的历史命中率没有足够样本；而第二轮本身已经稳定，没有它可修的抖动。换成长会话反复追问同一主题才可能体现，在拿到那个证据前保持默认关。
- **每个增量回合都要重规划一次**（`begin_cached_turn()`，否则计划会冻死）：追加回合的 prompt eval +5.7%（27B、192 块存储）。
- **整段状态镜像不可用，checkpoint 镜像只对 hybrid / recurrent 成立**：`--kvmem` 下 prompt cache / `--slot-save-path` 对该槽静默跳过（整段镜像被拒）；context checkpoint 在 hybrid / recurrent 模型上可用（携带 recurrent 行 + frontier）。**dense 不需要 checkpoint**（`2026-11-02` 结案）：纯 attention 的 `seq_rm` 是任意部分、截断即回退，实测改写回复只补 17 token，`server-context.cpp:11077` 的门是设计而非遗漏。**跨会话复用走 `--kvmem-conversations`**（实测切回只 prefill 新增的 25/27 token；`baddeb834` 之后连"被丢弃的答案尾巴"也由该 store 自己停放的快照恢复，27B 的 48k 切换从 80 s 降到"不超过一个 `-cms` 间距"），不能用 `-np > 1`（KVMem 会整体不启用）。⚠️ checkpoint 环随 store 停放（每 store 留最新 2 个），换 store 时不再清空、而是改用该 store 名下的快照；store 记账的尾巴界限依赖 `kvmem_last_n_gen`，取错（恒 0）会让任何超过 64 token 的答案把 store 判成「分支」。
- **单序列**：`--parallel > 1` 时 KVMem 与 KV stream 都不成立，只能回到 VBR / 原生 cache；多会话并发需要 fork 的 conversation store，未移植。
- **架构**：SWA 模型、纯 recurrent 模型、DFlash 草稿侧都不走 KVMem。
- **检索轮驱动**：`llama-server` 每请求驱动完整检索轮（前缀 prefill → 采样注意力 → 选择 → 回放后缀）；`llama-completion` 只在 prefill 末尾做一次 `apply_retrieval → pin`。**同一棵树换驱动能差十几个百分点，性能数字不可跨驱动横比**。
- **`--kvmem-budget` 必须 ≥ 一次 decode batch**（自动抬升，见仲裁表）；想让工作集更小就同时降 `-b`。不给 `--kvmem-budget` 时改由 `--kvmem-budget-auto` 决定（`min(设备容量, n_ctx × 1/3)`），所以「不写 budget」**不再等于全常驻**。
- **`--kvmem-gpu-low` 目前是死代码**：压力计划每次只裁到 `budget_blocks`，调 high/low 不改变压力事件数。
- **CPU 后端**：只有 `-dev none`（纯 CPU）能用 host 池；`-ngl 0` 但 CUDA 设备可见时会报 `multi-GPU layer requires every attention KV tensor on its model layer's GPU`。
- **检索评分只支持 FP16 范围内的操作数**：`recall_fp16` 拒绝 |x| > 65504，个别大模型的 mean-K 超界时整轮检索被跳过（日志 `KVMem retrieval round skipped: recall operand exceeds FP16 range`），退化为压力/新近度选择；不是崩溃，但那一轮用不上注意力评分。
- **图像（M-RoPE）按 append 顺序寻址**：图像 patch 的 `pos[0]` 会重复（一行 15,15,15,…），所以行号取登记顺序、`pos` 只作为 RoPE/空间信息保存；显式给了 `logical_pos` 的驱动仍由它决定行号。
- **PPL 默认批量会静默降级**：`llama-perplexity` 默认 `n_seq = batch/n_ctx > 1`，KVMem 要求 `n_seq_max=1`，会静默回落原生 cache。必须 `-b 512` 并看到 `KVMEM_KV_BYTES` 才算真测。
