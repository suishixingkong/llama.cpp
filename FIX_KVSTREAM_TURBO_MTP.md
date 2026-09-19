# KV streaming × TurboQuant / MTP —— 两处实测报错的定位与修复

现场：`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-v1.1.gguf`，V100-32GB，
`--kv-stream-stage-mib 3800 -np 1 -ctk turbo4 -ctv turbo3 --flash-attn on
--spec-type draft-mtp --spec-draft-n-max 3 -c 150000`。两条报错，一个起不来、一个跑崩：

| # | 触发 | 现象 |
|---|---|---|
| 1 | KV 流式 + turbo 量化同时开启 | 建上下文失败：`llama_init_from_model: failed to initialize the context: invalid block KV streaming page geometry` |
| 2 | KV 流式长上下文 + 推测解码（MTP）首次 decode | `process_ubatch: phase arena currently supports TG1 without speculative batches` → `llama_decode: failed to decode, ret = -3` → `srv decode: Compute error.` |

两条都是**集成的空白区**，不是参数写错，也不是数据损坏后才发现的那种偶发。

---

## 1. 问题 1：turbo KV 没有登记进 KV streaming 的类型表

### 根因链

`ggml_backend_cuda_kv_stream_get_type_capabilities()`（`ggml/src/ggml-cuda/fattn.cu`）
是 KV streaming 唯一的"这个 KV 类型能不能流式"的判据，它由一张白名单驱动：

```cpp
switch (type) {
    case GGML_TYPE_F32: ... case GGML_TYPE_NVFP4:
        result.classified = true;   // turbo2/3/4 不在其中
        break;
    default:
        return result;              // classified = false, storage = false
}
```

`ggml_cuda_kv_stream_page_bytes()` 的第一道门就是 `storage`：

```cpp
if (!capabilities_k.storage || !capabilities_v.storage) {
    return false;
}
```

于是 `llama_context` 在算 arena 页面几何时（`src/llama-context.cpp:456`）和
`llama_kv_cache` 在创建 stream runtime 时（`src/llama-kv-cache.cpp:371`）都拿到 `false`，
抛出那句与真实原因无关的 `invalid block KV streaming page geometry`。

**这个白名单从来没有人补过 turbo**：turboquant fork 里根本没有 kv-stream 代码
（`grep -rl kv_stream atomic-llama-cpp-turboquant/ggml/src/ggml-cuda` 为空），
akv fork 里也没有 turbo。两个 fork 各自都自洽，是这次合并把它们凑到了一起。

**可复现的证据**（不需要 GPU）：`tests/test-kv-stream-cuda-set-rows.cpp` 的第一个测试
`KV stream quant types are classified` 本身就断言"所有 `ggml_is_quantized()` 的类型都必须
classified"，而 `ggml.c` 给 turbo2/3/4 都标了 `.is_quantized = true` —— 该测试在修复前必然失败。

### 修复

给 turbo2/3/4 补上 `classified` + `storage` + `online_write` + `decode_f16`，
**刻意不补 `direct_attention`**（走 F16 转换回退路径）。两条能力都有现成实现，不是"顺手打开"：

| 能力 | 依据 |
|---|---|
| `online_write`（设备侧写量化缓存） | `set-rows.cu` 的 `k_set_rows_turbo2/3/4` 就是设备侧量化，含前向 WHT 与 InnerQ 标定；dispatch 在 `set_rows_cuda()` 的 `else if (dst->type == GGML_TYPE_TURBO*_0)` |
| `decode_f16`（按块解量化） | `convert.cu` 的 `ggml_get_to_fp16_cuda()` 对 turbo* 返回 `dequantize_block_cont_cuda<..., dequantize_turbo*_0>` |

**为什么 F16 回退在数值上与非流式一致**：turbo 缓存里存的是 WHT 旋转域的值，而 Q 的前向旋转、
注意力输出的逆旋转都由图上的 `GGML_OP_TURBO_WHT` 完成（`src/llama-graph.cpp` 2922/3047/3166/3279/3384
是 Q 侧前向，2651/2729 是 V/输出的逆变换），**与 FA 用哪个内核无关**。回退路径把整块 KV 页
（包括 resident 前缀）逐块 dequant 成 F16 再交给普通 F16 FA，读到的数值与 turbo direct 内核
是同一批（同一张质心表 + 同一个 norm 修正）。因此"先能用"取回退路径是安全的，
direct turbo FA 在流式 resident/ring 几何上的行为留待真机验证后再开。

### 顺带堵掉的几何错位（turbo 一旦可流式就会变得可达）

1. **head_dim 填充**：turbo 类型在 KV 缓存里把每个 head 零填充到 128 的倍数
   （`llama_kv_cache` 的 TurboQuant zero-padding 段）。几何查询必须用填充后的 head_dim，
   否则 head_dim=192/576 这类模型会被 `page_bytes_fn` 的 `head_dim % 128` 校验误拒。
   → 新增 `llama_kv_cache_padded_head_dim()`（`src/llama-kv-cache.{h,cpp}`），
   四个查询点（arena 2 处 + runtime 2 处）统一改用它。
2. **K 类型解析**：`TURBO_AUTO_ASYMMETRIC` 会在 GQA≥6 且 K/V 同为 turbo 时把 K 升级成 q8_0。
   这条规则原本只在 `llama_kv_cache` 构造函数里生效，而 arena 的页面几何用的是
   `params.type_k` —— 一旦 turbo 可流式化，`-ctk turbo4 -ctv turbo4` 在高 GQA 模型上就会出现
   "arena 按 turbo4 算页（每 head 100B）、缓存实际是 q8_0（每 head 272B）"的切页错位。
   → 把规则提成 `llama_kv_cache_resolve_type_k()`，context 在算几何和建内存模块之前先解析，
   两处永远一致（构造函数里的那次变成幂等空转）。

---

## 2. 问题 2：decode 阶段按 TG1 预留 compute

### 根因链

block KV streaming 的 arena 把 VRAM 切成 prefill / token-generation 两套布局，两套布局都在
`sched_reserve()` 里**预先量好**：

```cpp
// src/llama-context.cpp（修复前）
auto * gf_tg = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), true, sizes_tg.data());
```

`n_tokens == n_seqs` 就是"每序列 1 个 token"，即 TG1。而 `process_ubatch()` 用一条硬守卫
把这个前提钉死：

```cpp
if (kv_stream_phase_arena.configured && generation &&
        ubatch.n_tokens != cparams.n_seq_max) {
    LLAMA_LOG_ERROR("%s: phase arena currently supports TG1 without speculative batches\n", __func__);
```

推测解码恰好破坏这个前提：**采样 token 和草稿 token 在同一个 generation ubatch 里验证**。
server 侧 `GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1)`（`tools/server/server-context.cpp:3989`），
`--spec-draft-n-max 3` ⇒ 4 个 token ≠ `n_seq_max`(1) ⇒ 直接失败（`ret = -3` 在 server 侧被翻译成
`Compute error.`）。这条守卫是 akv fork 原样带来的（`llama.cpp-adaptive-kv-streaming/src/llama-context.cpp:1798`），
不是合并引入的缺陷，但它把"MTP + 长上下文流式"这条组合彻底锁死了。

顺带说明：CUDA 侧**本来就是支持多 query 的** —— `ggml_cuda_flash_attn_ext_streamed()` 按
`workspace_queries` 对 Q 的多行做 tile 循环（`fattn.cu:2243`），
`kv_stream_adapt()` 也用 `MAX_DECODE_QUERY_TOKENS = 32` 把 ≤32 query 当作 decode 处理。
限制只在于 **arena 只给 decode 预留了 1 个 token 的 compute 切片**。

### 修复

decode 阶段改按 `n_seq_max * (1 + n_draft_max)` 预留，上限为物理 ubatch：

```cpp
const uint32_t n_tokens_tg = std::min<uint32_t>(
    n_seqs*(1u + cparams.n_draft_max), cparams.n_ubatch);
const uint32_t n_outputs_tg = std::min(n_tokens_tg, cparams.n_outputs_max);
auto * gf_tg = graph_reserve(n_tokens_tg, n_seqs, n_outputs_tg, mctx.get(), true, sizes_tg.data());
```

- 新增 `llama_context_params.n_draft_max`（`[EXPERIMENTAL]`，默认 0），`llama_cparams` 同步新增字段。
- `common/common.cpp` 用**现成的** `common_speculative_n_max(&params.speculative)`
  （`common/speculative.cpp:2335`，server 自己也在用）填充：MTP/Eagle3/DFlash/DSpark/Simple 取
  `draft.n_max`，ngram 系列取各自的 `size_m`/`n_max`，无推测解码时为 0。
- 上限存进 arena owner（`max_generation_tokens`），`kv_stream_switch_phase()` 复用它重建 decode 调度器与图。
- 守卫改成"超过预留上限才报错"，并把真实数值打进日志：

```
phase arena reserved compute for 4 generation tokens (n_seq_max = 1, n_draft_max = 3) but the batch has 8
```

顺带修掉一个更隐蔽的情形：多序列（`-np > 1`）下只有部分序列在生成时，
`ubatch.n_tokens < n_seq_max`，原来的 `!=` 判断同样会误报。

**默认行为不变**：`n_draft_max == 0` 时 `n_tokens_tg == n_seqs`，与修复前逐字节一致
（日志里的 `tokens %u` 会多打印一个字段，`bs=%d` 在默认配置下仍是 `bs=1`）。

---

## 3. 激活方式

| 项 | 需要做什么 |
|---|---|
| 问题 1（turbo + 流式） | **什么都不用做**，重新编译即可；`-ctk turbo4 -ctv turbo3 --kv-stream-stage-mib N -fa on -np 1` 就能建上下文 |
| 问题 2（MTP + 流式） | **什么都不用做**，`common_context_params_to_llama()` 自动从 `--spec-type` / `--spec-draft-n-max` 取上限 |
| 手动 API | 直接设 `llama_context_params.n_draft_max`（0 = 旧行为） |
| 回退到旧行为 | `n_draft_max = 0`（即关掉推测解码）；turbo 侧删掉 `fattn.cu` 里的 turbo 条目即可回到"起不来" |

**`n_draft_max` 不是一个新的启动参数**（`common/arg.cpp` 未改动，CLI 参数表一个没变），
它是从既有推测解码参数**推导**出来的容量上界：

| 启用的 spec 类型 | 取值来源（既有 CLI 参数） |
|---|---|
| `draft-simple` / `draft-mtp` / `draft-eagle3` / `draft-dflash` / `draft-dspark` | `--spec-draft-n-max`（默认 3） |
| `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` | `--spec-ngram-{simple,map-k,map-k4v}-size-m` |
| `ngram-mod` | `--spec-ngram-mod-n-max` |
| `ngram-cache` | 固定 8 |
| 无（`none`） | 0 → decode 预留退化回"每序列 1 token"，与修复前逐字节一致 |

推导用的是 server 早就在用的同一个函数 `common_speculative_n_max()`（`common/speculative.cpp:2335`），
没有新语义。新增的只是 libllama 侧的**结构体字段** `llama_context_params.n_draft_max`
（`[EXPERIMENTAL]`，默认 0），供绕过 common 直接调库的嵌入方显式设置 —— 不设就是 0，
结果是守卫在遇到多 token 的 generation 批时报错（宁可失败，不会算错）。

新日志（`sched_reserve` 阶段）：

```
phase arena prefill: KV ... MiB, compute ... MiB, resident N pages/layer, ring M pages
phase arena decode:  KV ... MiB, compute ... MiB, resident N pages/layer, ring M pages, tokens 4
```

`tokens 4` = `n_seq_max(1) × (1 + n_draft_max(3))`，即这次 MTP 配置下 decode 预留的形状。

---

## 4. 验证情况

本机（Windows / MSVC 14.40 / CUDA 12.4，**无 NVIDIA 设备**）能做的验证都做了，
真机部分需要回 V100。

| 项 | 结果 |
|---|---|
| CPU 构建（`D:\llama-build\cpu`，Release） | `llama`、`llama-common`、`llama-cli` 编译通过，0 error |
| `test-kv-stream-config` / `-plan` / `-softmax` / `-bench-config` | 全绿（0 failures） |
| CUDA 编译（nvcc 12.4、`-allow-unsupported-compiler`、**arch sm_70**） | 单独编译 `fattn.cu`（含全部 turbo FA 实例化）0 error |
| `test-kv-stream-cuda-set-rows`（`D:\llama-build\cuda70`，`GGML_BACKEND_DL=OFF`，tests 已打开） | 6 个免设备用例全绿：270 assertions / 仅 4 个用例失败，且失败原因都是 `CUDA backend initializes`（无驱动、`ggml_backend_cuda_init(0)` 返回 nullptr），属于"本机没有 GPU"而非代码问题 |
| **真机数值与性能** | **未验证** —— 本机无 NVIDIA 设备，需要回 V100-32GB |

### 修复前后的 A/B（同一测试二进制、同一命令，只切换 `fattn.cu`）

为了让"修复前必然失败"变成实测而不是推理，把 `ggml/src/ggml-cuda/fattn.cu` 单独回退到 `HEAD`
（其余文件不动）重新编译后跑同一套测试：

| 断言 | 修复前（HEAD 的 fattn.cu） | 修复后 |
|---|---|---|
| `KV stream quant types are classified` | **FAIL** — `turbo2` / `turbo3` / `turbo4` 三个类型未登记 | PASS（33 assertions） |
| `every exposed KV-cache type has a GPU online writer` | **FAIL** — turbo2/3/4 的 `online_write` 与 `decode_f16` 都是 false | PASS（27 assertions） |
| `all exposed KV-cache pairs select an optimized execution class` | **FAIL** — 第一个 turbo 对就 `mode mismatch K=f32 V=turbo2`（mode = 0 = UNSUPPORTED） | PASS（146 assertions，12×12 矩阵） |
| `CUDA registry answers turbo KV stream queries without a device` | **FAIL** — `turbo K is streamable` 断言不成立 | PASS（8 assertions） |

最后一行就是运行时的那条路径：`ggml_backend_reg_by_name("CUDA")` → proc-address
`ggml_backend_cuda_kv_stream_type_pair_supported` / `page_bytes` / `workspace_bytes`
—— 与 `llama_context` / `llama_kv_cache` 在启动时调用的**是同一批注册项**，
所以启动即报 `invalid block KV streaming page geometry` 这件事在本机被完整复现并验证修好。

GPU-free 运行的复现方式（本机无驱动时需要一层垫片）：

```bat
:: 1) 让只做主机侧查询的测试二进制能加载（它静态依赖 nvcuda.dll）
::    D:\llama-build\nvcuda_stub\ 是一个只导出 11 个 stub 符号的假 nvcuda.dll，
::    语义是"驱动在、设备 0 个"，正好走 ggml_cuda_init() 的 0 设备分支
set PATH=D:\llama-build\nvcuda_stub;"D:\Program Files\cuda\bin";%PATH%
D:\llama-build\cuda70\bin\test-kv-stream-cuda-set-rows.exe
```

### 需要 V100 确认的清单

1. **能不能起来**：原命令跑一次，期望看到 `phase arena decode: ... tokens 4`，不再出现
   `invalid block KV streaming page geometry`。
2. **数值正确性**（最关键，F16 回退路径）：同一 prompt、同一采样参数下对比
   `-ctk turbo4 -ctv turbo3 --kv-stream-stage-mib 3800` vs 去掉 `--kv-stream-stage-mib`
   （非流式 turbo）的输出；两者应逐 token 一致（或至少 KL ≈ 0）。
3. **MTP**：确认 decode 不再返回 -3，`accepted x/y draft tokens` 正常出现。
4. **代价**：对比修复前后 `phase arena decode` 的 `resident N pages/layer` —— decode compute
   切片变大（1 → 4 token），resident 页数会略减，属于预期。
5. **回归**：不设 `--kv-stream-stage-mib` 的普通 turbo 路径、以及 `-np 1` 无推测解码的流式路径
   应与修复前表现一致。

> 注：**本文写于 direct attention 还不可达的时候，该结论已由
> `kv-stream-direct-fa` 分支推翻，见 `FIX_KVSTREAM_DIRECT_FA.md`。**
> 当时两处构建的 `GGML_CUDA_FA_ALL_QUANTS` 都只是 OFF，而 `#ifdef GGML_CUDA_FA_ALL_QUANTS`
> 恒假（该宏自上游 `5a4d0feca` 起不再被任何人定义），所以
> `ggml_backend_cuda_kv_stream_get_attention_mode()` 对**所有** KV 类型都返回 F16 转换路径。
> 也就是说：那时 turbo KV 在流式下走的就是 q8_0 / f16 这些类型同样走的那条路，没有额外假设。
> 测试里 DIRECT 相关的期望值已按构建实际能力参数化，各种配置下都应全绿。

### 测试文件本身顺带修掉的两处"必然失败"

`tests/test-kv-stream-cuda-set-rows.cpp` 在修复前就有两条断言在**任何**配置下都不成立，
它们和 turbo 无关，但会让"跑一次测试"这件事失去意义：

1. `KV stream quant types are classified` 要求**所有** `ggml_is_quantized()` 的类型都登记进
   KV 流式类型表 —— 连权重专用格式 `TQ3_1S` / `TQ4_1S` 也算（`-ctk/-ctv` 不提供它们，
   它们也没有 KV 路径）。已按 `common/arg.cpp` 的 `kv_cache_types` 语义跳过这两个。
2. DIRECT 相关用例把期望值写死成 `GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT`，
   而默认构建根本没编出 direct 内核 —— 改为运行时探测。`kv-stream-direct-fa` 之后这项
   进一步细化为"逐对探测 + 只在用默认 `GGML_CUDA_FA_QUANTS` 时断言精确集合"，
   详见 `FIX_KVSTREAM_DIRECT_FA.md`。

---

## 5. 仍然存在的限制

- KV streaming 依然**只支持 Qwen3.5 架构**、要求 `-np 1` / FlashAttention / GPU KV offload
  （akv fork 自身限制，见 `llama-kv-stream-config.cpp`）；MTP/draft 上下文永远不用 arena
  （`common/speculative.cpp:2502` 把 `kv_stream_arena_mib` 置 0）。
- turbo 当时走的是 **F16 转换回退**，不是 turbo direct 内核：多一次 per-page dequant + 一次
  workspace 往返，吞吐增益未测。
  > **已修复，见 `FIX_KVSTREAM_DIRECT_FA.md`。** 下面这段记录的是当时的诊断，作为
  > "为什么这个门会失效"的证据链保留下来。
  这**不是 turbo 特有的待遇，也不是精度上的妥协**：当时
  `ggml_backend_cuda_kv_stream_get_attention_mode()` 的 DIRECT 分支被
  `#ifdef GGML_CUDA_FA_ALL_QUANTS` 包着，而这个宏**任何构建都不会定义** ——
  CMake 选项 `GGML_CUDA_FA_ALL_QUANTS` 已 deprecated（`ggml/cmake/common.cmake:58-61`
  只把它翻成 `GGML_CUDA_FA_QUANTS=all`），再没有任何 `add_compile_definitions` 发这个宏；
  `compile_commands.json` 里出现 0 次。所以 `q8_0` / `f16` 这些"有 direct 内核"的类型
  在流式下走的**也是**同一条转换回退路径。
  （来源：上游 `5a4d0feca` "CUDA: replace GGML_FA_ALL_QUANTS with GGML_FA_QUANTS"，
  2026-09-09；在此之前 `ggml/src/ggml-cuda/CMakeLists.txt:115-118` 是
  `if (GGML_CUDA_FA_ALL_QUANTS) → GLOB 所有实例 + add_compile_definitions(...)`，
  所以 akv fork 2026-08-29 写的 `#ifdef` 在当时是**成立**的，是这次并入新基线让它失效。）
  另外注意：那个 `#ifdef` 在 `fattn.cu` 里包的是 **4 处**，不只是模式选择 ——
  1019-1068（`kv_stream_native_partial_fn` 类型 + `kv_stream_resolve_native_partial()`）、
  1214-1218（模式选择）、1823-1829（assert）、2300-2305（**真正的启动点**，`GGML_ABORT(
  "native quantized KV streaming requires GGML_CUDA_FA_ALL_QUANTS")`）。只改模式选择那一处
  不是"半通"，而是把"静默退 F16"换成 abort —— 必须四处一起改成 per-pair 判定。
  当时列的"想让 turbo 真正走 direct 需要三件事"，`kv-stream-direct-fa` 已照此实现：
  1. 修掉那个失效的门，判据落在 **(K,V) 对**上：
     `ggml_cuda_fattn_vec_instances()`（`ggml/cmake/common.cmake:53`）为 `FA_TYPES` 的
     每一对发一个 `GGML_CUDA_FA_<K>_<V>=0/1` 宏（`common.cmake:99-110`），并且只把
     `FA_COMBINATIONS` 里列出的实例文件加进编译（`file(GLOB)` 只在
     `GGML_CUDA_FA_QUANTS=all` 时走）—— `FATTN_VEC_CASE` 的 `if constexpr` 用的正是这些
     逐对宏，所以"按对判定"本来就是上游自己的语义。
     ⚠️ 两个坑：(a) 这些宏由 `add_compile_definitions` 发在 `ggml/src/ggml-cuda`，
     作用域只到该目录 —— 测试/其它 target 看不到，host 侧判断只能运行时问后端；
     (b) 别拿 `GGML_CUDA_FA_QUANTS` 字符串当判据：在 `f531b24b7` 之前该字符串只含"用户给的
     或 curated 基础集"（实测 6 对），而 cmake 另外追加的 turbo 组只体现在逐对宏上
     （同一次编译有 16 个 `=1`，含 `GGML_CUDA_FA_TURBO4_0_TURBO3_0=1`）；
     `f531b24b7` 之后默认值本身就是 16 对显式列表，两者才重新一致；
  2. 给 `kv_stream_resolve_native_partial()` 补 turbo 的 K/V 分支 —— 已补（70 对全覆盖）；
  3. 把 turbo 标成 `direct_attention` —— 已标；并且把该判断从"两个 per-type 标志"
     收紧为"标志 + 该对的内核确实存在"，因为 `direct_attention` 是 per-type 标志、
     而内核是 per-pair 的。
  内核本身其实已经在了：默认 FA 组合集（`ggml/CMakeLists.txt:207` 的
  `GGML_CUDA_FA_QUANTS` 默认值，自 `f531b24b7` 起是一份 16 对的显式列表）就含 10 个 turbo
  组合，其中就有 `turbo4_0-turbo3_0`；对应的
  `template-instances/fattn-vec-instance-turbo4_0-turbo3_0.cu` 调用
  `DECL_FATTN_VEC_CASE`，而该宏**同时**实例化 `..._case` 与 `..._partial_case`
  （`fattn-vec.cuh:944-949`），D=64/128/256 各一份。kv-stream 只会用到 D=256
  （`KV_STREAM_HEAD_DIM = 256`，且 `ggml_cuda_flash_attn_ext_streamed_supported()` 要求
  Q、V 的 `ne[0]` 都是 256）。所以这是一步"接线 + 验证"，不是"重写内核"。
  ⚠️ 反例：`D:\llama-build\cuda70` 的 cache 里显式写了
  `GGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`，因此那个目录**一个 turbo
  实例都没编**（实测 `GGML_CUDA_FA_*` 只有 4 个 `=1`）—— 这也说明"这套内核在不在二进制里"
  是构建配置决定的，写文档/测试时要以 `compile_commands.json` 为准，也说明逐对判定必须
  能优雅降级而不是断言失败。
- `n_draft_max` 由 `common_speculative_n_max()` 推出；如果绕过 common 直接调 libllama 且把
  `n_draft_max` 留成 0，则仍然回到"守卫直接报错"的行为（宁可失败，不会算错）。
- 多序列 + 推测解码的组合会按 `n_seq_max × (1 + n_draft_max)` 预留，arena 吃紧时可能报
  `token-generation phase does not fit shared CUDA arena`，此时需要加大 `--kv-stream-stage-mib`。

### direct_attention 与本次两个报错的关系（避免误记）

| | 问题 1（page geometry） | 问题 2（TG1 守卫） |
|---|---|---|
| 直接原因 | turbo 未登记 → `storage = false` → `ggml_cuda_kv_stream_page_bytes()` 第一道门 `return false` | arena 的 decode compute 按 `n_seqs` 预留，且 `ubatch.n_tokens != n_seq_max` 是硬判 |
| `direct_attention` 的角色 | **无因果作用**：几何检查只读 `storage`，模式判断的 F16 分支只读 `storage/online_write/decode_f16`。只标 direct、不标 storage，问题 1 分毫不动 | **完全无关**：不同文件、不同层（FA 内核选择 vs arena compute 预留） |
| 与修复的关系 | 属同一张能力表的第 5 个标志：当时必须**刻意不标**。若连它一起标，等那个死门被修好就会撞 `GGML_ASSERT(native_partial != nullptr)`，把"启动报错"变成"首帧崩溃" | 不参与 |
| 后续（`kv-stream-direct-fa`） | 那个死门已按 (K,V) 对修好，turbo 现在**已标** `direct_attention`，同时 `get_attention_mode()` 收紧为"标志 + 该对内核确实存在"，所以上面那个 assert 依然不可能命中 | 不参与 |
| 共同"元原因" | 本次集成把 turbo KV / kv-stream / MTP 三个特性放进同一棵树，却没有验证**两两组合**。`direct_attention` 落在 turbo×kv-stream 这一格里，TG1 守卫落在 kv-stream×MTP 那一格里 —— 是同一张"特性组合矩阵"上的两个兄弟缺口，技术上彼此独立 | 同左 |

### direct 的收益在哪、有多大（量级估算，未实测）

- **不需要"所有组合都编"**：正确的粒度是 per-pair。`ggml_cuda_fattn_vec_instances()`
  只把 `FA_COMBINATIONS` 里那几对编进来，上游的运行时回退也已经按这一事实工作
  （`ggml_cuda_get_fattn_vec_case()` 返回 nullptr 就警告并退回 f16-f16）。fork 用
  "全量编译"这个粗判据，既不必要（只用一对就够了）也不充分（`kv_stream_resolve_native_partial()`
  的白名单是第二道门）。
- 转换是 **per-chunk（每 256-token 页）** 的开销，与 query 行数无关：`convert_page()` 在
  `fattn.cu:2215`，位于 `for (int chunk = 0; chunk < nchunks; ++chunk)` 之内、
  `fattn.cu:2266` 的 query-tile 循环之外。所以：
  **TG1（1 个 query）= 100% 落在关键路径；prefill（最多 512 行）摊薄 512 倍；
  MTP 4-token 验证批摊薄 4 倍。**
- 以本实例的几何（turbo4 K + turbo3 V，head_dim 256，4 个 KV head，页 256 token）算：
  一页 KV = 139,264 B(K) + 102,400 B(V) = 241,664 B ≈ 236 KiB；F16 转换工作区
  = 2 × 512 KiB = 1 MiB。F16 回退每页每层多做"读 236 KiB + 写 1 MiB"，之后注意力再读那 1 MiB
  （而不是 236 KiB）。
- 但**净符号不保证为正**：direct 把 "F16 VEC 内核" 换成 "量化域 VEC 内核"，后者每元素 ALU 更多、
  D=256 上寄存器压力更大（`MERGE_QWEN38_V100.md` 记录过 252–255 寄存器），量化内核完全可能反超。
  最终只能靠真机 A/B。
- 上下文长度决定谁在关键路径上：150k 上下文时每个 token 要搬 ~8.6 GB 量级的 KV
  （即 `MERGE_QWEN38_V100.md` 里 turbo KV 的 8.59 GB/token 实测数），PCIe Gen3 x16 ≈ 10 GB/s，
  decode 被 H2D 复制主导 → 转换与复制重叠，direct 收益被压到个位数百分比；短上下文
  （KV 大量 resident）时 decode 是 compute-bound，收益才显出来。用
  `LLAMA_KV_STREAM_TRACE=1` 看 `copy busy %` / `deadline misses` 可以直接判断处在哪一侧。

## 7. 追加（2026-09-19）：turbo × 流式 × MTP「答非所问」的根因 = staged SET_ROWS 丢行基址

问题 1、2 修好之后，现场又报：turbo（+流式+MTP）输出只有 20 t/s 且答非所问；去掉 turbo、
去掉流式、或去掉 MTP 都基本正常。定位时先怀疑流式的 MMA span（`use_mma_prefill`），
但用 `GGML_CUDA_KV_STREAM_MMA_PREFILL=0/1` 在真机对照，**两种设置表现完全相同** ⇒ span 无关。

### 根因

MTP 的验证批（采样 token + 草稿 = 4 个 token）让一次 ubatch 写多行 KV，于是 SET_ROWS 走
**staged 写路径**（`ggml-cuda.cu:3216` `ggml_cuda_kv_stream_staged_set_rows_range()`）：先把行写进
只有 `row_count` 行的临时 staging 缓冲，再整块 `cudaMemcpyAsync` 到
`dst->data + first_row*row_bytes`（`set-rows.cu:1338-1370`）。因此写内核必须把 `src1` 里的
**绝对行号**减去 `dst_row_base`：

| 内核 | 行号 |
|---|---|
| `k_set_rows_quant`（q8_0/q4_0/…）、`k_set_rows`（f32/f16） | `*(src1+…) - dst_row_base` ✅ |
| `k_set_rows_turbo2/3/4` 及两个 `_tail` | `*(src1+…)`，**没有减 base** ❌ |

分发处（`set-rows.cu:1246-1251`）也只给量化/f32 分支透传 `dst_row_base`（模板形参有默认值 0），
turbo 三个分支**根本没接收它**。结果：staged 写时 turbo 用绝对行号写进小缓冲 →
**越界写进 pool 的其它区域，目标行反而没写**（随后 staging 缓冲被整块拷回缓存）。

### 为什么三个条件缺一不可

| 条件 | 作用 |
|---|---|
| turbo KV | 只有 turbo 分支丢 base（q8_0/q4_0/… 都透传） |
| `--kv-stream-stage-mib` | staged 路径只存在于 KV-stream 的 SET_ROWS 分派里 |
| MTP / 多 token 批 | staged 要求 `runtime->dirty_rows.size() > 1`；单 token decode 走普通 SET_ROWS（base 0，正确） |

这三条正好等于现场观察（q8_0 正常、不流式正常、去掉 MTP 只是"质量一般"）。
并且 staged 路径还要求 `get_type_capabilities(dst->type).online_write` —— 也就是说
**这个 latent bug 是被问题 1 的修复点亮的**：在给 turbo 补上 `online_write` 之前，turbo 不会走
staged 写（那时它连上下文都建不起来）。

### 修复

`set_rows_cuda_turbo{2,3,4}` 增加 `dst_row_base` 形参并透传给 5 个内核（turbo4 ×1、
turbo2 ×2、turbo3 ×2 含 tail），内核行号改为减 base；分发处传 `dst_row_base`。

### 回归覆盖

`tests/test-kv-stream-cuda-attn.cpp` 的 `resident staged writes support every exposed KV-cache
format` 原来只覆盖 9 个基础类型（**没有 turbo**），它断言 `staged_set_rows == 10` 与字节数、
并与非流式基线做数值比较（5e-4）。已加入 turbo2/3/4：修复前该用例对 turbo 会失败
（行写错位 → 数值与计数都对不上），修复后应通过。

## 6. 改动文件

```
ggml/src/ggml-cuda/fattn.cu           turbo2/3/4 登记进 KV streaming 类型能力表（storage/online_write/decode_f16）
src/llama-kv-cache.h                  + llama_kv_cache_padded_head_dim() / llama_kv_cache_resolve_type_k()
src/llama-kv-cache.cpp                同上实现；几何统一用填充 head_dim；K 类型解析复用助手
src/llama-context.cpp                 arena 几何用解析后类型 + 填充 head_dim；decode 按 n_draft_max 预留；守卫放宽
src/llama-context.h                   arena owner + max_generation_tokens
src/llama-cparams.h                   + n_draft_max
include/llama.h                       llama_context_params + n_draft_max [EXPERIMENTAL]
common/common.cpp                     从 common_speculative_n_max() 填充 n_draft_max
tests/test-kv-stream-cuda-set-rows.cpp turbo 类型矩阵 + GPU-free 的 registry 查询用例
ggml/src/ggml-cuda/set-rows.cu        turbo2/3/4 的 SET_ROWS 内核与启动器补 dst_row_base（§7）
tests/test-kv-stream-cuda-attn.cpp    staged 写用例加 turbo2/3/4；span 开关自述输出
```
