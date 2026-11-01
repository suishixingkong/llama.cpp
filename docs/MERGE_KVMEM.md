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
| server | `tools/server/*` | 每请求 Query span、只在 prefill 捕获存活时跑检索轮 |
| conversation store | `tools/server/kvmem-conversation-store.h`、`tools/server/server-context.cpp` | `--kvmem-conversations N` 的 slot↔store 映射：切走时 drain 到 host、切回时 restage 并命中前缀；请求字段 `kvmem_conversation` 绑定 |
| 多模态 | `tools/mtmd/mtmd-helper.*` | mmproj 与 kvmem 共存的上下文创建路径 |
| 测试 | `tests/test-backend-ops.cpp` | KVMem 相关后端用例的形态登记 |

**刻意排除**（fork 有、本分支不带）：

| 排除项 | 原因 |
|---|---|
| GDN / MTP replay（`replay_*` 记录面 + fold kernel） | 本分支的 `llama_memory_recurrent` 只有上游 `n_rs_seq` 快照；`--kvmem-mtp-state replay` 被参数层直接拒绝，不用 `snapshots`/`auto` 也能跑 |
| query replay | 服务的是 fork「prefix/query 两段 prefill + 工具续接复用 Q」的 server 执行器；本分支在 prefill 末尾一次性 `apply_retrieval → pin`，数值上无损失 |
| SWA / 纯 recurrent 架构 | 池化 + 滑动窗口 / 递归状态没有一致的块语义；`maybe_create()` 直接返回 `nullptr` 并告警，回落到原生 cache |

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
| `--kvmem-coarse-threshold N` | **32768** | 被打分上下文达到 N token 时改走两阶段：先用少量层给所有块粗排，再对 `max(64, 4×budget_blocks)` 个候选做精确全层打分。`0` = 永远全量精确（唯一没有召回风险的形态）。判据日志 `KVMEM_QUERY ... coarse=1 threshold=... kept=... layers=...`。详见「两阶段打分」 |
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

### 两阶段打分（`--kvmem-coarse-threshold`，默认 32768）

一轮检索要给**每个**块算一次分，每次都要读一遍 mean-K 镜像（每块 × 每个参与层）。上下文一长，
这一步就成了整轮里最大的单项，而花掉的绝大部分算力落在**选择根本不会取**的块上。

`--kvmem-coarse-threshold N`（单位 token，作用在**被打分的 frontier** 上）：达到 N 时，这一轮先
用**少量层**（参与层里等距抽 1–4 个）给所有块赔一个粗分，取前 `max(64, 4 × budget_blocks)` 块作为
候选（`mandatory` 那批无条件入选），再只对这些候选做**精确全层**打分。低于 N 一律全量精确 ——
这是唯一没有召回风险的形态，所以默认值放在「全量精确还只是噪声」的边界之上（实测 16k 上下
两阶段能省掉 38% 的评分时间，谈不上值得为此改语义）。

判据（Qwen3.5-2B，40 k 字符 ≈ 15 986 token、`-c 32768 --kvmem-budget 1024`、`KVMEM_PERF=1`）：

| | `score_ms` | 整轮 `total_ms` | 选块 |
|---|---|---|---|
| `--kvmem-coarse-threshold 0`（全量精确） | 21.16 | 33.57 | 33 块 |
| `--kvmem-coarse-threshold 1024`（两阶段） | **13.19** | **26.19** | 33 块，其中 32 块与上一行逐位相同 |

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
⚠️ 上面这一段的数字已被后续优化改写：capture 门控 + 打分器 staging 之后，同一口径的真驱逐代价是 **−14.3%**（`13662.5 → 11706.5 t/s`），详见本文末「四方复测」。

功能面（同一批构建）：长会话三连问 16k 上下文 / MTP 草稿回滚 / 图像输入都通过（细节见下面「已修复」与 `AUDIT_KVMEM_PERF_QUALITY.md`）。

**只有会评分的驱动才捕获 prefill K**（`e04bdb5d6`）：检索轮所需的 mean-K 与 query 行来自图中的捕获管道，它把每个 ubatch 的全部 K 行 D2H 到 host 再做打包。`llama-completion` / `llama-bench` / `llama-perplexity` 从不跑检索轮，却仍按检索方法付费搬运没人会读的行。捕获改由 `begin_retrieval_round()` 开启（server 每请求开一次），清轮/重置/重放入口不再重新开启；`harvest_gpu_v` 的 KV 镜像不受影响（换出块靠它回填）。0.8B `-p 4096 -b 512 --kvmem-budget 2048`：pp 9094 → 9710 t/s（+6.8%），server 侧 `KVMEM_QUERY scorer=page-softmax rows=1071 blocks=34` 照旧。

单测：KVMem 的 16 个测试已注册进 ctest（`test-kvmem-store`、`test-kvmem-runtime`、`test-spill-io`、`test-native-residency` 等，`ctest -N` 由 113 涨到 129），16/16 通过。

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

## 四方复测：旧版 / port 开 / port 关 / fork 开 / fork 关（`34063912e`）

这一轮问的是三件事：移植后**有没有退化**、真驱逐**还差多少**、回滚语义**与 fork 是否等效**。四棵树、五个单元，同一台机器（Windows + MSVC + CUDA 12.8、`sm_70`、V100 16 GB、`Qwen3.5-0.8B-Q4_K_M`），跑同一批参数。

### 五个单元

| 单元 | 二进制 | 含义 |
|---|---|---|
| `old` | `tmp/build-master-cuda`（2026-10-30 02:44） | **未移植 kvmem 的旧版**：`--help` 里没有任何 `kvmem` 字样 |
| `port-off` | `tmp/build-kvmem-off`（`-DLLAMA_KVMEM=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`） | port 树，kvmem 代码**不参与编译** |
| `port-nokv` | `tmp/build-kvmem-on`，不加 `--kvmem` | port 树，kvmem 编入但不启用 |
| `port-kv` | 同上 + `--kvmem --kvmem-budget 2048 --kvmem-block-tokens 128 --kvmem-gen-reserve 1024` | port 树，真驱逐 |
| `fork-nokv` / `fork-kv` | `kvmem-llama.cpp/build-v100/bin/llama-kvmem-server.exe` / `llama-bench.exe` | fork 本体，参数同上 |

⚠️ `old` 是一个**冻结的二进制**（构建目录的源现在是 master，master 已含 kvmem）：它的价值就在于是移植前那一刻的快照，重建它就不再是旧版。

⚠️ fork 的 `llama-bench` 与 `llama-perplexity` **不注册 `--kvmem`**（`--help` 里 0 个 kvmem 字样），所以 fork 的 kvmem 开/关只能在 server 上比。⚠️ port 的 `llama-bench` 没有 `--kvmem-gen-reserve`，基准表的池子 = budget（16 块），server 表显式给 1024（池 3072 = 24 槽）。

### prefill / decode（`llama-bench -p 4096 -n 128 -b 512 -fa 1 -ctk/-ctv f16 -dev CUDA0 -r 3`）

先热身一轮丢弃（GPU 空闲后的第一次跑的是时钟爬坡，不是代码：同二进制首轮 10728 / 次轮 13530），再**轮换顺序**跑 4 轮（固定顺序会造假差距），取中位数：

| 单元 | pp4096 t/s | 对 `port-nokv` | tg128 t/s | 对 `port-nokv` |
|---|---|---|---|---|
| `old`（无 kvmem） | 13637.3 | −0.2% | 273.6 | −0.4% |
| `port-off` | 13659.7 | −0.02% | 270.4 | −1.6% |
| `port-nokv` | **13662.5** | — | **274.7** | — |
| `port-kv`（真驱逐） | 11706.5 | **−14.3%** | 277.6 | +1.0% |
| `fork-nokv` | 10273.6 | −24.8% | 241.1 | −12.2% |

4 轮原始值（pp）：
`old` 13628.2 / 13646.4 / 13616.5 / 13648.1；`port-off` 13673.5 / 13584.4 / 13651.6 / 13667.8；`port-nokv` 13650.0 / 13657.0 / 13668.1 / 13668.1；`port-kv` 11841.2 / 11608.2 / 11674.4 / 11738.5；`fork-nokv` 10369.0 / 10171.0 / 10391.0 / 10178.2。

读法：
- **编入 kvmem 不花钱**：`port-off` / `port-nokv` / `old` 三者互差 ≤0.2%，都在噪声地板（同二进制自比 ±1.5%）内。
- **真驱逐的代价从 −32.6% 收到 −14.3%**（`94f780e9e` 去掉两处整设备等待、`e04bdb5d6` 只让会评分的驱动捕获 prefill K，`35e773a8c` 的打分器 staging + batch512）。剩下的 −14.3% 全在压力路径：块换出/回填 + 记账。
- **fork 基线本身慢**：同一台机器、同一个 `llama-bench`，fork 的 pp 低 24.8%、tg 低 12.2%。所以「fork 比 port 快/慢」必须拿 **fork 对 fork、port 对 port** 的比值说话，跨树横比会把 fork 的基线差算到 kvmem 头上。

### 检索轮与答案（server，4649-token 文档、3 次/单元）

请求是同一份 14 000 字中文文档（tokenize 后 4649 token），三个探针分别埋在 **12%**（`H-2298`，落在被驱逐的头部）、**60%**（`M-4471`）、**末尾**（`T-9932`），问题要求三个温度上限全部按格式给出。`-c 16384 -b 512 -np 1 --no-webui`，`temperature 0`，`max_tokens 200`。

| 单元 | pp t/s（中位） | tg t/s（中位） | 检索轮 | H / M / T |
|---|---|---|---|---|
| `old` | 9724.6 | 214.4 | 无 | **3/3** |
| `port-off` | 9680.3 | 209.8 | 无 | **3/3** |
| `port-nokv` | 9030.3 | 199.9 | 无 | **3/3** |
| `port-kv`（回滚） | 5655.8 | 202.4 | `mandatory_exceeds_pool`，**回滚** | 2/3，头部答成 87 |
| `port-kv --kvmem-query-max 512` ¹ | 6027.6 | 205.3 | 应用，`rows=512` | 2/3，头部答成 87 |
| `port-kv --kvmem-query-max 1024` ¹ | 5681.3 | 201.6 | 应用，`rows=1024 score_ms=83.8` | 2/3，头部答成 87 |
| `fork-off` | 6154.1 | 143.2 | 无 | **3/3** |
| `fork-kv` | 4062.9 | 146.3 | 应用，`rows=512 score_ms=66.5` | **3/3** |

¹ 这两档只跑了 1 次（用来分离「上限」与「回滚语义」两个变量），时间数字按单次读，不经中位数。

三次的答案逐字相同（`H-2298=87, M-4471=64, T-9932=64` 或正确的 `51/87/64`），说明这不是采样抖动而是工作集差异：池 3072 token 只装得下 4649 的尾部，12% 处的块被压力路径按新近度扔掉了。**回滚语义本身是对的**——port 与 fork 在「装不下」时都不硬裁；差别在下一节。

### 质量：PPL（200 kB 中文、`-c 8192 -b 512 -fa on`、f16）

| 单元 | PPL | 对 `old` |
|---|---|---|
| `old` | 12.0629 ± 0.18749 | — |
| `port-off` | 12.0629 | ±0 逐位同 |
| `port-nokv` | 12.0629 | 逐位同 |
| `port-kv`（budget 2048） | 12.3027 ± 0.19200 | **+2.0%** |
| `port-kv --kvmem-method recency` | 12.3027 | 同上，逐位同 |
| `fork-nokv` | 12.0627 | −0.002% |

`recency` 与 `retrieval` 在 PPL 上逐位相同，是因为 **PPL 从不跑检索轮**（每个 chunk 先 `memory_clear`，末尾的检索影响不到本 chunk 的 logits），这是口径限制而非实现等价——要验检索方法只能看上面那张 server 表。

### 结论

1. 移植没有引入退化：`port-nokv` / `port-off` / `old` 的 pp、tg、PPL 三者互相在噪声内（PPL 甚至逐位相同）。
2. 真驱逐的代价是 **−14.3% prefill / +1.0% decode**（bench）与 **−37% 首字速度**（server，4649 token：9724 → 5656），后者是大头，因为 server 的每请求检索轮 + 压力搬运都压在 prefill 上。
3. 回滚语义生效且可观测：`need=22 cap=14` → 保留压力视图，不评分、不提交。
4. **query 行在 fork 里是工作区，在 port 里算历史** —— 已修（`--kvmem-query-workspace`，默认 on）。修前同参数下 fork 选中 18 块、port 选中 16 块：

```
port : selected 0 19 20 21 22 23 24 25 26 27 29 32 33 34 35 36        (16)
fork : selected 0 7 18 19 20 21 22 23 24 25 26 27 28 29 30 32 33 34 35 36  (18)
```

port 少掉的两块里，**第 7 块正是埋着头部探针的那一块**。成因是 fork 的检索轮走 Query replay：它的 `fixed` 集由 `retrieval_prefix_mandatory(frontier)` 生成，`frontier` 是**提问之前**的历史末端，所以 Query 自己那几块属于 append 工作区（追加到 16 块之外），16 个语义槽全部留给历史；port 没有 replay 遍，`fixed` 用 `query_fixed_blocks(retrieval_mandatory())` 且 `frontier = total_tokens()`，Query 的尾部块要占同一份 16 个槽，历史因此少两块，第 7 块落榜。**补这一处不需要移植 replay**，只需要把记账改对——见下一节。

### 复现

```bash
# 基准（轮换、热身、中位数）
bash tmp/cmp/bench_matrix.sh
# server 检索轮 + 三段探针答案
python tmp/cmp/matrix.py m-port-kv 8302 14000 200 tmp/build-kvmem-on/bin/llama-server.exe \
    --kvmem --kvmem-budget 2048 --kvmem-block-tokens 128 --kvmem-gen-reserve 1024
# 单测
cd tmp/build-kvmem-on && ctest -R "test-(kvmem|backend-rebind|pinned-kv-tier|memory-contract|native-kv-payload|host-kv-storage|spill-io|tiered-kv-storage|native-residency|kv-content|prefix-checkpoint|mean-key-retrieval|graph-recall|query-policy|raw-kv-store|conversation-store)"
```

16 个 kvmem 单测全过。日志在 `tmp/cmp/matrix/`、`tmp/cmp/bench/`、`tmp/cmp/ppl_*.log`。

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

同二进制 A/B（27B，6.1k token 文档，`--kvmem-budget 2048`，8 针 × 3 轮）：

| ARM | 命中 | 失忆率 |
|---|---|---|
| 关掉 `begin_cached_turn()`（= 修复前） | 13/24 | **45.8%** |
| 打开（现在） | 23/24 | **4.2%** |
| 对照：`--kvmem-method recency`（只留 sink+最近块） | 13/24 | 45.8% |

修复前**逐位等于 recency**——这正是「计划冻在第一轮压力视图」的形状。代价是每回合重规划一次：追加回合的平均 prompt eval **1532 → 1619 ms（+5.7%）**。

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

### 27B 实测（Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp，6.1k token 文档 = 192 块，`-fa on -ctk/-ctv f16 -dev CUDA0 -b 512 -c 16384`）

| 配置 | 预算 | 保留比例 | 命中 | 失忆率 |
|---|---|---|---|---|
| 不用 `--kvmem`（全上下文注意力） | — | 100% | 24/24 | 0% |
| `--kvmem`，预算装得下（不驱逐） | 8192 | 100% | 24/24 | 0% |
| `--kvmem` 真驱逐 | 4096 | 67% | 23/24 | 4.2% |
| `--kvmem` 真驱逐 | 2048 | 33% | 23/24 | 4.2% |
| 同上 + `--kvmem-query-workspace off` | 2048 | 33% | 23/24 | 4.2% |
| 同上 + `--kvmem-method recency` | 2048 | 33% | 13/24 | **45.8%** |
| `--kvmem` 真驱逐 | 1024 | 17% | 19/24 | 20.8% |
| `--kvmem` 真驱逐 | 512 | 8% | 8/24 | **66.7%** |

两件事值得单独说：

- **kvmem 本身不伤质量**：不驱逐时 100%，与不用 kvmem 逐位相同（24/24）。伤质量的只有驱逐。
- **新记账（Query 工作区）在这条轴上测不出差别**：on/off 都是 23/24，且 miss 是同一条记录。它不是为端到端命中率准备的，它是补「同预算少 2 块历史」的记账缺口，在 64 块的预算上占 3%，针探针分辨不出来。

### 保留比例才是自变量（把预算换算成「留了文档的百分之几」）

12k token 文档（14656 token = 458 块）补了另一段：

| 保留比例 | 8.3% | 14% | 17% | 28% | 33% | 67% |
|---|---|---|---|---|---|---|
| 命中率（检索） | 33% | 58% | 79% | 88% | 96% | 96% |
| 命中率（recency，33% 处） | — | — | — | — | 54% | — |

绝对预算不说明问题，**预算 / 上下文比**才说明问题：留到 1/3 就能到 96%，留到 1/12 掉到 33%，而同样留 1/3 时纯 recency 只有 54%。所以「kvmem 会不会失忆」= 「预算够不够文档的 1/3」+ 「检索评分有没有真的在跑」。

### 缓解手段实测（12k 文档、预算固定 2048、基线 58.3%）

| 手段 | 命中 | 结论 |
|---|---|---|
| `--kvmem-sink-tokens 512 --kvmem-recent-tokens 512` | 41.7% | **更差**：钉住的两条带是从同一份 64 块里扣的，直接少掉 32 块可打分历史 |
| `--kvmem-graph-alpha 0.25` | 58.3% | 无变化（邻居扩散改变不到弃块决策） |
| `--kvmem-query-max 256` | 58.3% | 无变化（Query 本来只有 28 token，够不到上限） |

**只有加预算有效。** 想要「质量不掉 + 性能不掉」的可行组合是：把预算设到工作文档的 1/3 上下，并确认日志里没有 `no_query_rows` / `retrieval_skipped`；预算调不下来就只能接受边缘事实被丢。

### 失败形态：不是「不知道」，是「拿邻近记录顶替」

唯一那次 miss 里问的是 needle 1 的 `ZQ7K`，模型答 `P8VT`——同一轮 needle 5 的码。被驱逐的块不是让模型变空，而是让它用还在驻留的**同类记录**填空。这条对实际使用比命中率更值得警惕：失忆看起来像自信的错误答案，而不是「文档里没有」。

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
- **检索质量只由「预算 / 上下文比」决定**：留 1/3 命中 96%，留 1/12 只剩 33%；`sink/recent` 钉带反而更差（从同一份额度里扣）。`--kvmem-query-max`：512 与默认（0）逐位同（95.8%，同一条 miss），**128 反而掉到 79.2%**——它把 128 token 以下的短 Query 之外的轮次也截了，冷启动轮因此拿到一个「装得下」的假计划并提前定住工作集。除非要卡上限，别调它。
- **LRFU 默认关，实测没有收益（27B、8 针 × 2 轮、`--kvmem-budget 2048`）**：

  | 配置 | 命中 | 失忆率 | 回滚 |
  |---|---|---|---|
  | 默认（page-softmax，LRFU off） | 15/16 | 6.2% | 1 |
  | `--kvmem-lrfu` | 15/16 | 6.2% | 1 |

  选块列表与基线**逐位相同**，连 miss 的那根针（深度 0.166 的 `ZQ7K`，第二轮被 `Y4KB` 挤掉）都一样。原因：这个 probe 里每根针最多被问两次，LRFU 依赖的历史命中率没有足够样本；而第二轮本身已经稳定，没有它可修的抖动。换成长会话反复追问同一主题才可能体现，在拿到那个证据前保持默认关。
- **每个增量回合都要重规划一次**（`begin_cached_turn()`，否则计划会冻死）：追加回合的 prompt eval +5.7%（27B、192 块存储）。
- **整段状态镜像不可用，checkpoint 镜像只对 hybrid / recurrent 成立**：`--kvmem` 下 prompt cache / `--slot-save-path` 对该槽静默跳过（整段镜像被拒）；context checkpoint 在 hybrid / recurrent 模型上可用（携带 recurrent 行 + frontier），dense 模型仍然没有 checkpoint、改写回复要整段重算。**跨会话复用走 `--kvmem-conversations`**（实测切回只 prefill 新增的 25/27 token），不能用 `-np > 1`（KVMem 会整体不启用）。⚠️ 换 store 会把该槽的 checkpoint 环清空（环属于被 detach 的那块 memory），改用一次性 `kvmem_swap_frontier` 认账；store 记账的尾巴界限依赖 `kvmem_last_n_gen`，取错（恒 0）会让任何超过 64 token 的答案把 store 判成「分支」。
- **单序列**：`--parallel > 1` 时 KVMem 与 KV stream 都不成立，只能回到 VBR / 原生 cache；多会话并发需要 fork 的 conversation store，未移植。
- **架构**：SWA 模型、纯 recurrent 模型、DFlash 草稿侧都不走 KVMem。
- **检索轮驱动**：`llama-server` 每请求驱动完整检索轮（前缀 prefill → 采样注意力 → 选择 → 回放后缀）；`llama-completion` 只在 prefill 末尾做一次 `apply_retrieval → pin`。**同一棵树换驱动能差十几个百分点，性能数字不可跨驱动横比**。
- **`--kvmem-budget` 必须 ≥ 一次 decode batch**（自动抬升，见仲裁表）；想让工作集更小就同时降 `-b`。不给 `--kvmem-budget` 时改由 `--kvmem-budget-auto` 决定（`min(设备容量, n_ctx × 1/3)`），所以「不写 budget」**不再等于全常驻**。
- **`--kvmem-gpu-low` 目前是死代码**：压力计划每次只裁到 `budget_blocks`，调 high/low 不改变压力事件数。
- **CPU 后端**：只有 `-dev none`（纯 CPU）能用 host 池；`-ngl 0` 但 CUDA 设备可见时会报 `multi-GPU layer requires every attention KV tensor on its model layer's GPU`。
- **检索评分只支持 FP16 范围内的操作数**：`recall_fp16` 拒绝 |x| > 65504，个别大模型的 mean-K 超界时整轮检索被跳过（日志 `KVMem retrieval round skipped: recall operand exceeds FP16 range`），退化为压力/新近度选择；不是崩溃，但那一轮用不上注意力评分。
- **图像（M-RoPE）按 append 顺序寻址**：图像 patch 的 `pos[0]` 会重复（一行 15,15,15,…），所以行号取登记顺序、`pos` 只作为 RoPE/空间信息保存；显式给了 `logical_pos` 的驱动仍由它决定行号。
- **PPL 默认批量会静默降级**：`llama-perplexity` 默认 `n_seq = batch/n_ctx > 1`，KVMem 要求 `n_seq_max=1`，会静默回落原生 cache。必须 `-b 512` 并看到 `KVMEM_KV_BYTES` 才算真测。
