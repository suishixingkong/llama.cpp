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
| 多模态 | `tools/mtmd/mtmd-helper.*` | mmproj 与 kvmem 共存的上下文创建路径 |
| 测试 | `tests/test-backend-ops.cpp` | KVMem 相关后端用例的形态登记 |

**刻意排除**（fork 有、本分支不带）：

| 排除项 | 原因 |
|---|---|
| GDN / MTP replay（`replay_*` 记录面 + fold kernel） | 本分支的 `llama_memory_recurrent` 只有上游 `n_rs_seq` 快照；`--kvmem-mtp-state replay` 被参数层直接拒绝，不用 `snapshots`/`auto` 也能跑 |
| conversation store（`--kvmem-conversations N`） | fork 为「KVMem 强制 `n_parallel=1`」设计的多会话热切换，与上游 slot 机制正面冲突；只影响多会话 TTFT，不影响质量 |
| query replay | 服务的是 fork「prefix/query 两段 prefill + 工具续接复用 Q」的 server 执行器；本分支在 prefill 末尾一次性 `apply_retrieval → pin`，数值上无损失 |
| SWA / 纯 recurrent 架构 | 池化 + 滑动窗口 / 递归状态没有一致的块语义；`maybe_create()` 直接返回 `nullptr` 并告警，回落到原生 cache |

## 使用参数

所有参数都由 `common` 解析，因此 **`llama-completion` / `llama-server` / `llama-perplexity` 等所有基于 common 的工具都支持**。
`llama-bench` 自建 cparams，不经过 common，所以它单独实现了 `--kvmem` / `--kvmem-budget` / `--kvmem-method` / `--kvmem-block-tokens`（其余 `--kvmem-*` 仍然只在 common 工具里）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--kvmem` | 关 | 打开 KVMem 池。要求 `n_seq_max == 1`（即 `--parallel 1`），否则告警并回落到 VBR / 原生 cache |
| `--kvmem-block-tokens N` | 0 → 32 | 块大小（token）。选择、驱逐、检索都以块为粒度 |
| `--kvmem-budget N` | 0 → `n_ctx` | GPU 常驻工作集（token）。**0 = 全常驻、不驱逐**，此时与原生 cache 数值逐位相同。下限是**一次 decode batch**（`-b`）加恒留带：整批 token 在 ubatch 之前就一次性登记，批内不能驱逐，给得比这小会被自动抬到下限并告警一次 |
| `--kvmem-gen-reserve N` | 0 → 256 | 给 decode 预留的额外 GPU token |
| `--kvmem-sink-tokens N` | 0 → 一个块 | 永久保留的前缀（attention sink） |
| `--kvmem-recent-tokens N` | 0 → 压力下保留最新尾部 | 永久保留的后缀 token 数 |
| `--kvmem-method {recency,retrieval}` | `retrieval` | `recency` 只按新近度选块（不采样、不评分，最省）；`retrieval` 用 Query 注意力给每块打分 |
| `--kvmem-query-scorer {page-softmax,meanq-cosine}` | `page-softmax` | 检索评分器。`page-softmax` 逐行 softmax 后聚合；`meanq-cosine` 用平均 Query 向量做余弦 |
| `--kvmem-score-agg {mean,max}` | 评分器默认 | 行级分数聚合成块分数的方式 |
| `--kvmem-lrfu` / `--no-kvmem-lrfu` | 评分器默认 | 是否把历史命中率（LRFU）混进块分数 |
| `--kvmem-lrfu-lambda X` | 0.5 | 历史衰减，`[0,1]` |
| `--kvmem-lrfu-top-frac X` | 0.9 | 视为「命中」的历史分位，`(0,1]` |
| `--kvmem-graph-alpha X` | 评分器默认（0.2 / 0） | 图召回项的权重 |
| `--kvmem-query-begin N` / `--kvmem-query-end N` | −1 | 显式声明 Query 的原始 token 区间。CLI 直接透传；server 每请求覆盖 |
| `--kvmem-force-pos N` | −1 | 强制钉住某个 token 所在的块 |
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

## 环境变量

**参数等价物**：每个 `--kvmem-*` 都有对应的 `LLAMA_ARG_KVMEM_*`（llama.cpp 的通用机制），例如 `LLAMA_ARG_KVMEM=1`、`LLAMA_ARG_KVMEM_BUDGET=2048`、`LLAMA_ARG_KVMEM_METHOD=recency`。完整集合：
`LLAMA_ARG_KVMEM`、`_BLOCK_TOKENS`、`_BUDGET`、`_GEN_RESERVE`、`_SINK_TOKENS`、`_RECENT_TOKENS`、`_METHOD`、`_QUERY_SCORER`、`_SCORE_AGG`、`_LRFU`、`_LRFU_LAMBDA`、`_LRFU_TOP_FRAC`、`_GRAPH_ALPHA`、`_QUERY_BEGIN`、`_QUERY_END`、`_FORCE_POS`、`_MTP_STATE`、`_GPU_RATIO`、`_GPU_HIGH`、`_GPU_LOW`、`_HOST_MIB`、`_DISK_MIB`、`_DISK_PATH`、`_CPU_GB`、`_HARVEST_V`。

**诊断/调试开关**（都只影响日志或诊断，不影响数值）：

| 变量 | 作用 |
|---|---|
| `KVMEM_TRACE=1` | 打开 `KVMEM_*` 事件日志。**判断「是否真的启用」看 `KVMEM_KV_BYTES bytes=… cells=… slots=… budget=…`**，没有这行就是被静默降级了 |
| `KVMEM_PERF=1` | 每 ubatch 的分相计时（捕获/评分/换出/D2H/HARVEST 汇总） |
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

功能面（同一批构建）：长会话三连问 16k 上下文 / MTP 草稿回滚 / 图像输入都通过（细节见下面「已修复」与 `AUDIT_KVMEM_PERF_QUALITY.md`）。

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

### 半满尾块的前缀回填

池子按块驱逐，**尾块下的块**迟早会被赶走；而服务端的 checkpoint 恢复常在**块中间**裁剪（`CHECKPOINT_ATTN_ONLY_TRIM p0=13207`，13207 % 32 = 23），于是这个块又成了尾块，但只有 packed store 还记得它那 23 行。append 只写自己新增的行（`apply_ubatch` 的 `sinfo.idxs` 是逐行的），前缀就此留空，而 store 仍宣称整块驻留 —— 下一次工作集事务 `snapshot()` 就在那块的第一行报 gap。

修法是在 `prepare_working_set()` 里、`register_append()` **之前**（此时 `n_tokens` 仍等于已提交行数）调用 `restore_partial_tail()`：尾块不满且 `gpu_kv_complete()` 为假时，从 packed store 落位并物化前缀（`write_block_to_gpu`），MTP 侧同步 `restore_checkpoint_block()`。失败一律 fail-closed —— 发布空行比报错更糟。

复现与验证（Qwen3.5-2B、f16、`-c 262144 --kvmem-budget 1000 -b 512`、13.2k prompt 连续 5 轮各生成 5k）：修复前 3/6 次在第三轮 decode 报 gap；修复后 5 轮全过，且 `KVMEM_TRACE=1` 能看到 `tail_restage` 正好在这些轮次触发（一次运行 3 次）。PPL 1.2126（开/关逐位同）、真驱逐 budget 576 为 1.2135。

## 已知限制

- **单序列**：`--parallel > 1` 时 KVMem 与 KV stream 都不成立，只能回到 VBR / 原生 cache；多会话并发需要 fork 的 conversation store，未移植。
- **架构**：SWA 模型、纯 recurrent 模型、DFlash 草稿侧都不走 KVMem。
- **检索轮驱动**：`llama-server` 每请求驱动完整检索轮（前缀 prefill → 采样注意力 → 选择 → 回放后缀）；`llama-completion` 只在 prefill 末尾做一次 `apply_retrieval → pin`。**同一棵树换驱动能差十几个百分点，性能数字不可跨驱动横比**。
- **`--kvmem-budget` 必须 ≥ 一次 decode batch**（自动抬升，见仲裁表）；想让工作集更小就同时降 `-b`。
- **`--kvmem-gpu-low` 目前是死代码**：压力计划每次只裁到 `budget_blocks`，调 high/low 不改变压力事件数。
- **CPU 后端**：只有 `-dev none`（纯 CPU）能用 host 池；`-ngl 0` 但 CUDA 设备可见时会报 `multi-GPU layer requires every attention KV tensor on its model layer's GPU`。
- **检索评分只支持 FP16 范围内的操作数**：`recall_fp16` 拒绝 |x| > 65504，个别大模型的 mean-K 超界时整轮检索被跳过（日志 `KVMem retrieval round skipped: recall operand exceeds FP16 range`），退化为压力/新近度选择；不是崩溃，但那一轮用不上注意力评分。
- **图像（M-RoPE）按 append 顺序寻址**：图像 patch 的 `pos[0]` 会重复（一行 15,15,15,…），所以行号取登记顺序、`pos` 只作为 RoPE/空间信息保存；显式给了 `logical_pos` 的驱动仍由它决定行号。
- **PPL 默认批量会静默降级**：`llama-perplexity` 默认 `n_seq = batch/n_ctx > 1`，KVMem 要求 `n_seq_max=1`，会静默回落原生 cache。必须 `-b 512` 并看到 `KVMEM_KV_BYTES` 才算真测。
