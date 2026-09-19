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
- **MMA prefill span 实测有数值错误**（见 §5.4）。

### 5.4 MMA prefill span（`use_mma_prefill`）：实测错误，默认关

`use_mma_prefill` 是 fork 自带的加速路径，修复前它是**死代码**（判据含 `!convert_to_f16`），
所以在本轮之前从未在 GPU 上执行过。开关是 `GGML_CUDA_KV_STREAM_MMA_PREFILL`，**默认关**。

**四轮 V100 记录（同一二进制，只切开关）**：

| 轮次 | 条件 | 结果 | 判定 |
|---|---|---|---|
| ① 09-19 白天 | span 开；**同卡有 llama-server 在推理** | 23 failures / 1037（2e-3 ~ 1.7e-2） | 数值与 ③ **逐位相同** |
| ② 同一次会话 | span 关（`=0`） | 0 failures / 1036 | 对照 |
| ③ 09-19 13:0x，空闲卡 | span 开（显式 `=1`） | **23 failures / 1037**，各用例 `max_abs` 与 ① 相同（0.00202951 / 0.016892 / 0.00382153 / 0.0130784 / 0.0353837） | **权威结果** |
| ④ 中间一次"干净复测" | 无 env，但二进制已含 `00aa10c12`（span 默认关） | 0 failures / **1036** | **无效：span 未执行** |

④ 的识别方法：span 断言只在开关打开时执行，所以开关开/关的**总断言数差 1**（1037 vs 1036）；
④ 报 1036 即与对照轮同档，等于没测到 span。测试现在开头打印
`kv-stream MMA prefill span: ENABLED|DISABLED`，各用例诊断行带 `mma_spans=`。

**结论：span 是错的，不是"未验证"。** 误差随 KV 跨度增长（257 query → 3.8e-3、
1024 query → 1.3e-2、十六层 32 个 span → 3.5e-2），而同一套用例关掉 span 后全部落在
1.5e-4~3.4e-4。① 与 ③ 逐位相同 ⇒ 那次同卡并发**没有**污染数值，我一度把 ① 判为污染、
把 ④ 判为干净，方向恰好相反，已更正。

**已排除的两个嫌疑（别重复挖）**：

- **partial 数量不匹配**：`launch_fattn()` 在 `output_partial` 时强制
  `blocks_num = (ntiles_dst, 1, 1)`（`fattn-common.cuh:1557-1564`，注释写明"preserving exact
  partial numerator/meta output"），每行**恰好一个** partial，写的是未归一化分子 + `(max, sum)`
  （`fattn-mma-f16.cuh:1794-1803`），与 `kv_stream_accumulate_chunk_results` 的读法一致。
  所以 `partial_count = 1`（`fattn.cu:2081`）是对的。
- **Volta compact 内核**：`if constexpr (DKQ == 256 && DV == 256 && ncols1 == 32 && ncols2 == 2 &&
  !output_partial)`（`fattn-mma-f16.cuh:2234`）已把 partial 变体排除，且 span 用的是
  `<256, 256, 8, 8>`，两个条件都不成立。

**计数器盲区（本次踩到的坑）**：`++resident_cache->stats.mma_prefill_attention_spans` 外面套着
`if (resident_cache != nullptr)`（`fattn.cu:2407-2413`），所以**没有 resident cache 的 runtime
即使跑了 span 也报 0** —— native-pairs 用例正是这种 runtime，它那 16 个失败的行一度显示
`mma=0`，把人引向"span 与此无关"。测试已改为同时打印开关状态（`span=ON|off`）与该计数器，
并以开关为准。

默认继续保持关。仍然开着的是"span 在 sm_70 + 流式分页下的具体错因"——partial 约定已核对过，
下一步要么在同一颗内核上做 `GGML_CUDA_VOLTA_FA_COMPACT` / 单 span 最短回归的二分，要么先按
"小批走 vec、大批才允许 span"收窄它。**span 只影响多 token 批（含 MTP 验证批）的 prefill**，
而"多 token 批 × 流式"正是现场报缺陷的那条组合，所以关掉它同时也是一个可能的修复动作。

> **测量纪律**：本机无 GPU，数值必须在 V100 上采，且该卡必须空闲。
> 另：**判读配置是否真的生效，别只看"跑起来了"** —— 断言总数、stats 计数器、开关的自述输出
> 都要对一遍（本轮两个坑都出在这里）。


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
6. **span 已定性，不用再跑 A/B**：见 §5.4——span 开 = 23 failures（可复现），span 关 = 全绿。
   **首选错因（源码级定位，2026-09-19，待真机确认）**：span 的 partial meta 语义在
   `np > 1` 时是 `(scale, rowsum)`，而 `kv_stream_accumulate_chunk_results` 要的是 `(max, rowsum)`。

   - span 实例化为 `<DKQ=256, DV=256, ncols1=8, ncols2=8>` → `ncols = 64`；Volta 配置表里
     `nthreads = 128`、`nbatch_fa = 32`（`fattn-mma-f16.cuh:139`），`cols_per_warp = 32`
     （`get_cols_per_warp()`，Volta 分支）⇒ `np = nwarps*cols_per_warp/ncols = 4*32/64 = 2 > 1`。
   - `np > 1` 时组合步骤把 meta 槽改写为 `make_float2(KQ_cms, KQ_crs)` = **scale**（`exp(warpmax - 合并max)`）
     + 合并 rowsum（`fattn-mma-f16.cuh:1664-1670`）；真正的 `(max, rowsum)` 只在
     `needs_fixup/is_fixup` 分支里写进 `dstk_fixup_meta`。
   - 而 partial 的收尾写的是 `dstk_fixup[row] = make_float2(meta_j[0], meta_j[1])`
     （`fattn-mma-f16.cuh:1800-1803`）⇒ 拿到的正是那个 **scale**，不是 max。
   - 合并内核于是按 `weight = exp(meta.x - maximum) ≈ 1` 做**无权合并**；chunk 的 max 差异被丢掉，
     误差随 chunk 数/跨度增长 —— 与实测（257q 3.8e-3 → 1024q 1.3e-2 → 十六层 3.5e-2）一致。
   - **`np == 1` 和单 chunk 的情形不受影响**，这正好解释了为什么 `one-page` /
     `fully resident` 两个用例在 span 开时仍然**逐位相等**（没有可合并的第二块）。
   - vec 侧是直接写 `make_float2(KQ_max[jc], KQ_sum[jc])`（`fattn-vec.cuh:773`）⇒ 语义正确，
     这也是 span 关时全绿的原因。

   验证/修复步骤（按顺序，每步都能证伪上一步）：
   (a) **一次尝试已回退**（`b51ba9e11`）：按"partial 的 meta 应该是 `(max, rowsum)`"改过一版，
       真机结果**不是变好而是整体变差**：

       | 用例 | 修改前 | 修改后 |
       |---|---|---|
       | native 对（16 个） | 2.0e-3 | 2.2e-3（turbo 侧 3.1e-3 → 5.8e-3） |
       | wide-query 1024 | 1.31e-2 | **0.632758** |
       | four-query 页边界 | 1.75e-3 | 2.83e-3 |
       | server-shaped / 257q / 512q | 1.69e-2 / 3.8e-3 / 2.5e-3 | 1.47e-2 / 2.3e-3 / 2.2e-3（略好） |

       数字确实被这段改到了（方向对），但"分子与 rowsum 相对**合并后的 max** 归一"这个前提
       不成立；或者写者守卫漏了行 —— 1024q 那次 0.63 更像是整行 meta 没被写。已回退该改动。
   (a2) **改用测量代替猜测**：新增 `GGML_CUDA_KV_STREAM_TRACE_SPAN=1`，在归一化之前打印合并
       累积器的 `meta0/meta1 = (max, denominator)` 与前两行的前 4 个分子值，并带上
       `span=` / `nchunks=` / `parts=` / `nrows=`。**同一负载跑两次**（span 开 / 关），
       以 span 关（vec 路径）那次为基准，就能读出 span 真正发布的是什么、差在 max 还是分母。
   (b) 真机：`set GGML_CUDA_KV_STREAM_MMA_PREFILL=1 && test-kv-stream-cuda-attn.exe`，
       期望 23 failures → **0**，且各 `max_abs` 掉到 vec 同量级（1e-4）；若仍失败，
       下一步查 numerator 的约定（`FATTN_KQ_MAX_OFFSET` 与 `KQ_cmr`）。
   (c) 若 (b) 通过，再决定默认是否翻回"开"，并先量 prefill tok/s（见上面第 4 条的量级预期）；
       收益不明显就直接删掉这条路径。
   （原 (a)(b)(c) 的三条候选 —— `VOLTA_FA_COMPACT=0` 复跑、最短回归、阈值收窄 ——
   留作 (b) 失败后的兜底二分手段。）

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
