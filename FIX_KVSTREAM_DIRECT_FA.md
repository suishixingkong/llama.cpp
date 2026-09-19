# KV streaming × direct attention —— 让原生内核重新可被选中

现场：`llama.cpp` 集成分支（`master`）在并入 `llama.cpp-adaptive-kv-streaming` 之后，
`--kv-stream-stage-mib` 一旦开启，**任何** KV 量化类型都只会走 "把整页 KV 解量化成 F16 再交给
F16 FA" 的转换回退路径，即使该 (K,V) 对的**原生量化内核明明已经编进了二进制**。
换句话说 direct attention 这个能力在这棵树里**不可达**。

本分支：`kv-stream-direct-fa`（base = `master` @ `6bc377c9f`）。

---

## 1. 根因：判据挂在一个已经不存在的宏上

`ggml_backend_cuda_kv_stream_get_attention_mode()`（`ggml/src/ggml-cuda/fattn.cu`）是
"这个 (K,V) 对用哪种流式注意力" 的唯一判据，修复前它的 DIRECT 分支是：

```cpp
#ifdef GGML_CUDA_FA_ALL_QUANTS
    if (capabilities_k.direct_attention && capabilities_v.direct_attention) {
        return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS
```

`GGML_CUDA_FA_ALL_QUANTS` 这个**宏**从上游 `5a4d0feca`（"CUDA: replace GGML_FA_ALL_QUANTS with
GGML_FA_QUANTS"，2026-09-09）起就再没有任何地方 `add_compile_definitions` 它了：
CMake 选项还在，但它只被翻译成 `GGML_CUDA_FA_QUANTS=all`
（`ggml/cmake/common.cmake:58-61`），发宏的活儿交给了
`ggml_cuda_fattn_vec_instances()` 的**逐对**定义：

```cmake
foreach (TYPE_V IN LISTS FA_TYPES)          # q4_0 … f16 turbo2_0 turbo3_0 turbo4_0
    foreach (TYPE_K IN LISTS FA_TYPES)
        string(TOUPPER "GGML_CUDA_FA_${TYPE_K}_${TYPE_V}" COMBINATION_DEF)
        add_compile_definitions(${COMBINATION_DEF}=${COMPILED})   # 1 或 0
```

所以 `#ifdef` 永远为假 → DIRECT 永远不被返回 → 所有类型都落到 F16 转换分支。
akv fork 2026-08-29 写下这个 `#ifdef` 时它**是成立**的（当时 `GGML_CUDA_FA_ALL_QUANTS=ON`
会 `GLOB` 全部实例并发这个宏），是并入新基线让它失效。

同一个失效还带来两处"半通"：

| 位置 | 修复前 | 后果 |
|---|---|---|
| `kv_stream_resolve_native_partial()` 整个函数 | 被同一个 `#ifdef` 包着 | 函数不存在，任何直连尝试都编不过 |
| 该函数的分发表 | 只有 7 个基础类型，没有 turbo | 即使放开 DIRECT，turbo 对也拿不到内核 |
| `ggml_cuda_flash_attn_ext_streamed()` 的 else 分支 | `GGML_ABORT("native quantized KV streaming requires GGML_CUDA_FA_ALL_QUANTS")` | 只改模式那一处的话，"静默退 F16" 会变成 "首帧 abort" |

---

## 2. 修复：判据落在 (K,V) 对上，读逐对宏

三件事，全部在 CUDA 侧：

### 2.1 按对解析原生内核（`kv_stream_resolve_native_partial()`）

去掉 `#ifdef`，改成对**每一对**问一次 `if constexpr`，读的正是
`ggml_cuda_fattn_vec_instances()` 发的那个逐对宏：

```cpp
#define KV_STREAM_NATIVE_PARTIAL_CASE(type_K_case, type_V_case)                        \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                        \
        if (type_k == GGML_TYPE_##type_K_case && type_v == GGML_TYPE_##type_V_case) {  \
            return ggml_cuda_flash_attn_ext_vec_partial_case<                          \
                KV_STREAM_HEAD_DIM, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>; \
        }                                                                             \
    }
```

覆盖范围与上游自己的运行时选择器 `ggml_cuda_get_fattn_vec_case()`
**完全一致**：7×7 基础类型 49 对 + turbo 21 对（turbo×turbo 对角线、turbo×{f16,q8_0}
两个方向、turbo×turbo 的 6 个交叉），合计 70 对。`F32` 刻意不在表里：它没有任何
partner 的 partial 实例（上游的通用选择器是把 F32 映射到 F16 case，流式原生路径不做这件事，
它只吃 KV 缓存里真实写下的类型）。

`if constexpr (0)` 的分支被丢弃 → 不会 odr-use 未编译的实例 → **未编译的对不会变成
链接错误**，只会让函数返回 `nullptr`。这正是上游 `ggml_cuda_get_fattn_vec_case()`
的语义（返回 nullptr 就 warning 并退回 f16-f16）。

### 2.2 模式判定改为"能力 + 内核都在"

```cpp
    if (capabilities_k.direct_attention && capabilities_v.direct_attention &&
            kv_stream_resolve_native_partial(type_k, type_v) != nullptr) {
        return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT;
    }
```

**为什么必须加后半句**：`direct_attention` 是 per-type 标志，而内核是 per-pair 的。
只判两个 per-type 标志的话，`-ctk turbo4 -ctv q4_0` 会宣称 DIRECT，然后在
`ggml_cuda_flash_attn_ext_streamed()` 里撞上
`GGML_ASSERT(native_partial != nullptr)` —— 把"静默退 F16"换成"首帧崩溃"。
加了后半句之后，同一个对会**优雅降级**到 F16 转换，与上游通用选择器对未编译对的处理一致。

### 2.3 turbo 补上 `direct_attention`，清掉两处 `#ifdef`

- `get_type_capabilities()` 的 `direct_attention` 名单增加 `TURBO2_0 / TURBO3_0 / TURBO4_0`。
  这是**真的能力**：turbo 缓存放的是 WHT 旋转域的值，Q 的前向旋转与注意力输出的逆旋转都是
  图上的 `GGML_OP_TURBO_WHT`（`src/llama-graph.cpp`），与 FA 选哪个内核无关；自然，
  `ggml_cuda_flash_attn_ext_vec_partial_case<256, TURBO4_0, TURBO3_0>` 的实例本来就在
  `template-instances/fattn-vec-instance-turbo4_0-turbo3_0.cu` 里，且 `DECL_FATTN_VEC_CASE`
  同时实例化 `_case` 与 `_partial_case`（`fattn-vec.cuh:944-949`）。
- `ggml_cuda_flash_attn_ext_streamed()` 里第 4 处 `#ifdef`（except/abort 那个）删除，
  只剩 `GGML_ASSERT(native_partial != nullptr)`；第 3 处（取函数指针）去掉 `#ifdef` 保留。
  这两处现在是**冗余证据检查**，真正的判定已经前移到 2.2。

---

## 3. CMake 选项 `GGML_CUDA_FA_ALL_QUANTS` 保留，语义不变

本分支**没有**动它：`option(GGML_CUDA_FA_ALL_QUANTS ...)` 与
`common.cmake` 里的 deprecated 别名都原样保留。它仍然等价于
`GGML_CUDA_FA_QUANTS=all`，而 `all` 会把每个 `fattn-vec-instance-*.cu` 都 GLOB 进来并给
对应逐对宏发 `=1` —— 于是新的逐对分发表自然把**所有**对解析成 DIRECT。

也就是说：

| 想要的效果 | 怎么做 | 修复前 | 修复后 |
|---|---|---|---|
| 只放开默认集（16 对）的 direct | 什么都不用做 | 不起作用（宏不存在） | 16 对走 direct |
| 放开全部对的 direct | `-DGGML_CUDA_FA_ALL_QUANTS=ON` 或 `-DGGML_CUDA_FA_QUANTS=all` | 不起作用 | 全部有实例的对走 direct |
| 自定义集合 | `-DGGML_CUDA_FA_QUANTS=f16-f16;q8_0-q8_0;...` | 只有 F16 回退 | 列出且已编译的对走 direct，其余 F16 |

代码里对宏的**唯一**剩余接触点为零：`grep -rn GGML_CUDA_FA_ALL_QUANTS --include=*.cu
--include=*.cuh --include=*.cpp --include=*.h .` 为空。

---

## 4. 行为变化（默认 `GGML_CUDA_FA_QUANTS`）

默认集（`ggml/CMakeLists.txt:207`）的 16 对全部落在 10 个 direct-capable 类型之间，
所以它们**全部**从 F16 转换变成 DIRECT；其余对不受影响。

| 类别 | 例子 | 修复前 | 修复后 |
|---|---|---|---|
| 默认集内 | `f16-f16`、`q8_0-q4_0`、`turbo4_0-turbo3_0`、`f16-turbo4_0` | F16 转换 | **DIRECT**（原生 `<K,V>` 内核） |
| 有实例但未列入 FA_QUANTS | `q4_1-q5_1`、`turbo4_0-f16` | F16 转换 | F16 转换（不变） |
| 无 `online_write`（Q2_K / Q8_1 …） | `Q2_K-Q4_0` | UNSUPPORTED | UNSUPPORTED（不变） |

随之"复活"的既有分支（它们本来就在 akv fork 里，只是在这棵树上是死代码）：

- `ggml_cuda_flash_attn_ext_streamed()` 的 resident-span 合并（`desc.token_count =
  resident_span_pages*block_tokens`，`fattn.cu:2150-2173`）与随后的
  "全 resident 或单页 → 直接调普通 `ggml_cuda_flash_attn_ext`" 短路
  （`fattn.cu:2331`）。这条路径的**目的**就是"让 logits 与非流式缓存逐位一致"，
  它读的是合并后的完整 resident 前缀，而不是第一页。
- prefill 的 `use_mma_prefill` → `ggml_cuda_flash_attn_ext_mma_f16_partial_case<256,256,8,8>`
  （`fattn.cu:2355`）——**但这条必须加前置条件，见 §4a**。
- `ggml_cuda_kv_stream_workspace_bytes()` 对 DIRECT 对返回 **0**（不需要转换工作区），
  `llama_context` / `llama_kv_cache` 早已按这个语义申请内存。

### 4a. 顺带堵掉的前置项：`use_mma_prefill` 依赖的 f16 scratch 未必被预留

`use_mma_prefill` 的判据里有 `!convert_to_f16`，所以它在修复前是**死代码**，一旦打开 direct 就
活的。它调的是 `ggml_cuda_flash_attn_ext_mma_f16_partial_case<256,256,8,8>`，而后者在
`launch_fattn()` 上传的是 `need_f16_K = need_f16_V = true`（`fattn-mma-f16.cuh:2391-2393`）。

**先澄清一个容易误判的点**：这**不会**把量化页直接喂给 f16 内核。
`launch_fattn()`（`fattn-common.cuh:1381-1440`）会用 `ggml_get_to_fp16_cuda()` /
`ggml_get_to_fp16_nc_cuda()` 把 K/V 转成 f16，turbo 类型在里面也有分支
（`convert.cu` 的 `to_fp16_nc` 对 TURBO2/3/4 返回 `dequantize_block_cuda<...>`）。
数值上是对的。

**真正的坑在内存**：那份 f16 scratch 由 `ggml_cuda_flash_attn_ext_get_f16_extra_data()`
算成 `dst->data + ggml_nbytes(dst)` 之后的一段，**只有**当非流式的选核也要求 f16 时，
图分配器（`ggml_cuda_flash_attn_ext_get_alloc_size()`）才会把这几个 MiB 留在 KQV 后面。
而一个 direct 的量化对，在同一个 shape 上完全可能被选核判成**不需要 f16**：

| 选核结果 | 条件（Volta） | `get_alloc_size` 是否预留 f16 |
|---|---|---|
| `VEC` | `can_use_vector_kernel && Q->ne[1]*gqa_ratio_eff <= 2` | **否**（对该对已编译、且非 F32） |
| `TILE` / `MMA_F16` | 其余 | 是 |
| `VOLTA_Q8_W4` | 设了 `GGML_CUDA_VOLTA_Q8_FATTN_TC` 且 K=V=q8_0、`Q->ne[1]==4`、head_dim 256… | **否**（走 `ggml_q8v::get_alloc_size`，早退） |

第一行对应 `Q->ne[1] == 2`（GQA 比数为奇数，如 MHA 或 gqa=3）；第三行恰好是
**Qwen3.8-27B 的 4-token MTP 验证批 + q8_0 KV** 的形状——也就是说在用户的 V100 上，
只要开了那个实验开关，`use_mma_prefill` 就会在没有预留的地址上写 K/V 的 f16 副本。

**修复**：把 `get_alloc_size()` 里那段 `switch (kernel)` 提成
`ggml_cuda_flash_attn_ext_needs_f16()`，两处共用同一个答案，然后给 `use_mma_prefill`
加上 `f16_scratch_reserved`（= `need_f16_K && need_f16_V`）：

```cpp
    bool need_f16_K = false;
    bool need_f16_V = false;
    ggml_cuda_flash_attn_ext_needs_f16(dst, &need_f16_K, &need_f16_V);
    const bool f16_scratch_reserved = need_f16_K && need_f16_V;

    const bool use_mma_prefill = !convert_to_f16 && f16_scratch_reserved && ...
```

不在预留里时自动退回原生 partial 内核（与今天的行为一致，只是不再有越界写）。
判据取 `&&` 是**保守**的：`launch_fattn` 只对"类型不是 F16"的那一路做转换，
所以 `need_f16_K && need_f16_V` 覆盖得比必要更严，不会漏。

> ⚠️ 这条是 fork 自带的**潜在**缺陷（`GGML_CUDA_FA_ALL_QUANTS=ON` 时同样会被激活），
> 不是本次改动引入的；但打开 direct 会在本树里把它激活，所以一并堵掉。
> 与本次主线改动**互不依赖**，可以单独 revert。

---

## 5. 验证

本机 Windows / MSVC 14.40 / CUDA 12.4 / **无 NVIDIA 设备**，只能编 + 链接 + 跑纯 host 侧用例。
真机部分见 §6。

### 5.1 两种 FA 配置各编一遍（重要的是"逐对门真的按对生效"）

| 构建目录 | `GGML_CUDA_FA_QUANTS` | `GGML_CUDA_FA_*` 逐对宏 | `fattn.cu` | 链接 |
|---|---|---|---|---|
| `D:\llama-build\cuda70` | `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`（**4 对**，51 个 turbo 宏 `=0`） | 4 个 `=1` | 0 error | `test-kv-stream-cuda-set-rows` / `-attn` 均 **链接成功** |
| `D:\llama-build\cuda70d` | 默认（**16 对**，含 turbo） | 16 个 `=1` | 0 error | 同上，**链接成功**（且是 `BUILD_SHARED_LIBS=ON` 的 DLL 版本） |

第一行是本分支最要紧的证据：**在 turbo 内核一个都没编的 build 里链接消费者仍然成功**，
说明 `if constexpr (GGML_CUDA_FA_TURBO4_0_TURBO3_0)` 为 0 时那个分支确实没有被 odr-use，
未编译的对只会让解析器返回 `nullptr`，而不是变成未定义符号。

### 5.2 免设备执行（`nvcuda` 垫片，见 `FIX_KVSTREAM_TURBO_MTP.md` 的复现方式）

`test-kv-stream-cuda-set-rows`，两个 build 用同一份二进制逻辑：

| 用例 | cuda70（4 对） | cuda70d（默认 16 对） |
|---|---|---|
| `KV stream quant types are classified` | PASS (33) | PASS (33) |
| `Q8 K and Q4 V retain direct streamed attention` | PASS (3)（走 `else`：非 UNSUPPORTED） | PASS (3)（DIRECT） |
| `no native CUDA flash-attention KV pair loses its execution class` | PASS (49) | PASS (49) |
| **`the curated FA set is exactly the set of direct streamed pairs`** | PASS（配置不匹配，跳过） | **PASS (162)** |
| `every exposed KV-cache type has a GPU online writer` | PASS (27) | PASS (27) |
| `CUDA registry answers turbo KV stream queries without a device` | PASS (8) | PASS (8) |
| 其余 4 个需要真实设备的用例 | FAIL：`CUDA backend initializes` | FAIL：`CUDA backend initializes` |

最后一行 4 个失败是**环境限制**（无驱动、`ggml_backend_cuda_init(0)` 返回 nullptr），
与本次改动无关，也与 `FIX_KVSTREAM_TURBO_MTP.md` 里记录的修复前状态一致。

`the curated FA set ...`（162 条断言）是这次的关键验证：它对 12×12 = 144 对逐一断言
"是 DIRECT 或 F16、且 DIRECT 只出现在两个类型都 direct-capable 的对上"，
**并断言 DIRECT 恰好 16 对、F16 恰好 128 对** —— 与从源码推出的默认集合逐对吻合。

### 5.3 没有做的验证

- **未跑 CPU 构建**：本次改动的可编译部分全部在 `ggml/src/ggml-cuda/*.cu` 与
  `ggml/include/ggml-cuda.h`（仅注释）里，`tests/test-kv-stream-cuda-*.cpp` 也由
  `LABEL cuda` 守卫；CPU 构建一个改动文件都不编，属于"树一致性检查"而非证据，故略。
- **未跑真机数值**（见 §6）。
- **MMA prefill span 仍未验证**（见 §5.4）。

### 5.4 MMA prefill span（`use_mma_prefill`）的验证状态：至今没有一次干净测量

`use_mma_prefill` 是 fork 自带的加速路径，修复前它是**死代码**（判据含 `!convert_to_f16`），
所以在本轮之前从未在 GPU 上执行过。它的开关是 `GGML_CUDA_KV_STREAM_MMA_PREFILL`，
**默认关（opt-in）**，理由如下——其中包含一次被污染的测量和一次无效的复核：

| 轮次 | 条件 | 结果 | 可否作依据 |
|---|---|---|---|
| ① 2026-09-19 | span 开，**同一张 V100 上有 llama-server 正在推理** | 23 failures / 1037 断言（2e-3 ~ 1.7e-2，随 KV 跨度增长） | ✗ 受同设备并发干扰 |
| ② 同一次会话 | span 关（`=0`） | 0 failures / 1036 | ✓（与后来的干净轮一致） |
| ③ 干净复测第一轮 | 无 env | 0 failures / **1036** | ✗ **无效：span 根本没执行** |
| ④ 干净复测第二轮 | `=0` | 0 failures / 1036 | ✓ |

③ 无效的原因：span 断言只在开关打开时执行，所以开关开/关的**总断言数差 1**（1037 vs 1036）；
③ 与 ④ 都是 1036，说明两轮都是 span 关，等于**一次干净测量都没有**。
为杜绝这类误读，测试现在开头打印一行 `kv-stream MMA prefill span: ENABLED|DISABLED`，
并在 server-shaped / wide-query / sixteen-layer 三个用例的诊断行里带上 `mma_spans=`。

有依据的正面结论只有一条：**span 关掉时**，native/转换 vec partial 路径在整个
`test-kv-stream-cuda-attn` 里都能复现非流式参考（100 对含 turbo 全过 5e-4、
server-shaped 1.4e-4、1024-query 2.5e-4、其余 780+ 断言全绿）。

在拿到一次（GPU 空闲、显式 `=1`）的干净轮之前，默认保持关闭：span 只影响多 token 批
（含 MTP 验证批）的 prefill，而"多 token 批 × turbo 流式"恰好是现场报缺陷的那条组合，
宁可先慢不要先错。

> **测量纪律**：本机无 GPU，数值必须在 V100 上采；采集时该卡必须空闲。
> 2026-09-19 那次在 llama-server 推理占卡时采到的 2e-3~1.7e-2 已在空闲复测中消失，
> 整轮结论被推翻（详见上表）。GPU 上有其它进程时采到的数值一律不作为结论依据。


## 6. 需要 V100 确认的清单

1. **数值正确性（最关键）**：跑 `tests/test-kv-stream-cuda-attn`，重点三条
   - `all native CUDA KV pairs preserve streamed prefill results`（100 对，501 断言，容差 5e-4）
   - `fully resident multi-page prefill stays bit-identical to ordinary CUDA attention`
     （q8_0 K + q4_0 V，512 KV / 2 页全 resident，断言**逐位相等**）——这条现在是
     **真·跑在 direct 上**的，修复前它实际走的是 F16 转换路径；
   - `resident pages survive between evaluations while the tail is refreshed` 等 resident 用例。
   > 注意其中 q8_0/q4_0 混合对、`Q->ne[1]==4` 的 4 个 query 用例在 Volta 上会被选核判成
   > `TILE`，所以 `use_mma_prefill` 的 `f16_scratch_reserved` 前置条件为真，路径照旧
   > （不会被降级）；只有 `Q->ne[1]==2` + 奇数 GQA 比、或 `GGML_CUDA_VOLTA_Q8_FATTN_TC`
   > 那个形状才会被拦到原生 partial 内核上。
2. **turbo 流式端到端**：`-ctk turbo4 -ctv turbo3 --kv-stream-stage-mib 3800 -fa on -np 1`
   能否起来 + 输出与非流式 turbo 逐 token 一致（同一 prompt、同一采样参数）。
3. **该对到底走没走 direct**：`LLAMA_KV_STREAM_TRACE=1` 之外，最直接的证据是
   `ggml_cuda_kv_stream_workspace_bytes()` 对 DIRECT 对返回 0 —— 启动日志里
   `phase arena` 那几行的 workspace/convert 字段应当消失或归零；
   也可以对比 `resident N pages/layer`（direct 不再需要转换工作区，切片会变）。
4. **收益**：对比 direct 与 F16 回退的 decode 吞吐。用 `LLAMA_KV_STREAM_TRACE=1` 看
   `copy busy %` / `deadline misses` 判断处在 H2D-bound 还是 compute-bound 一侧 ——
   长上下文（150k）时 KV 搬运约 8.6 GB/token、PCIe Gen3 x16 ≈ 10 GB/s，转换与复制重叠，
   direct 的收益可能被压到个位数百分比；短上下文 / KV 大量 resident 时才可能显出来。
   另外注意 **prefill 在 direct 下仍然会转 f16**（走 `use_mma_prefill` 的 MMA 内核），
   所以 prefill 的收益预期为 0，收益只可能出现在 decode / resident 一侧。
5. **回归**：`test-kv-stream-cuda-set-rows` 全绿（本机已跑，见 §5.2）。
6. **span 的 A/B（先确认这张卡没有别的进程在用）**：

   ```bat
   test-kv-stream-cuda-attn.exe
   set GGML_CUDA_KV_STREAM_MMA_PREFILL=1 && test-kv-stream-cuda-attn.exe
   ```

   两轮都应 0 failures；第二轮断言数多 1 且在 server-shaped / wide-query / sixteen-layer
   的诊断行里 `mma_spans` 非 0（开头那行会打印 span 是 ENABLED 还是 DISABLED，据此确认
   配置真的生效）。**若第二轮出现这三处的数值失败**，就是 span 的 partial 合并有实错，
   按 `max_abs` 与 KV 跨度的关系定位（见 §4）。若干净轮通过，可把默认改回开启——
   那是一个独立的性能决策，需要先有 tok/s 对比。见 §5.4。

## 7. 改动文件

```
ggml/src/ggml-cuda/fattn.cu           direct 判定改按 (K,V) 对解析；turbo 进 direct_attention；
                                      4 处 #ifdef GGML_CUDA_FA_ALL_QUANTS 全部移除；
                                      新增 ggml_cuda_flash_attn_ext_needs_f16() 并给
                                      use_mma_prefill 加 f16_scratch_reserved 前置条件（§4a）
ggml/include/ggml-cuda.h              direct_attention 的语义注释（per-type vs per-pair）
tests/test-kv-stream-cuda-set-rows.cpp 期望值改成逐对；新增 curated 集 == direct 集 的断言
tests/test-kv-stream-cuda-attn.cpp    direct 对的原生等价性覆盖（7→10 类型 = 100 对）、
                                      转换工作区补齐、span 开关的自述输出（§5.4）
docs/build.md                         GGML_CUDA_FA_QUANTS 默认值/合法类型列表按源码订正
FIX_KVSTREAM_TURBO_MTP.md             同步"direct 不可达"的过期结论
MERGE_TURBO_KVSTREAM.md               勘误：turbo 组自 f531b24b7 起就在 GGML_CUDA_FA_QUANTS 里
FIX_KVSTREAM_DIRECT_FA.md             本文
```

未改动（刻意）：`ggml/CMakeLists.txt` 的 `option(GGML_CUDA_FA_ALL_QUANTS)`、
`ggml/cmake/common.cmake` 的 deprecated 别名。

§4a 那一项与主线改动**互不依赖**，如果只想保留"打开 direct"这一件事，单独回退
`use_mma_prefill` 的 `f16_scratch_reserved` 前置条件与 `ggml_cuda_flash_attn_ext_needs_f16()`
提取即可（其余改动不受影响）。
