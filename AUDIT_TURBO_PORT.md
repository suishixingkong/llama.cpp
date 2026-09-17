# TurboQuant 移植审计 — `turbo-kvstream-merge` vs `atomic-llama-cpp-turboquant`

审计对象：`1a2b5ebf4`（turbo KV-cache 子集）、`a69c11f33`（类型号 + 构建修正）、
`70009cf81`（CUDA 侧上游 API 漂移修正）三个提交，以及它们落在 `turbo-kvstream-merge` 上的结果。
方法：文件集合差集 → 逐文件 md5（CRLF 归一）→ 文件内按符号的行集合比对 → 对每个疑似点读全文。

对照源：`C:\Users\confu\llama\atomic-llama-cpp-turboquant` @ `044dff242`。

## 结论

**量化算法本身移植正确，没有数值偏移。问题集中在"接口面"——测试覆盖、模板实例化清单、
以及被排除架构的残留定义。** 没有发现会导致推理结果错误的缺陷。

> **2026-09-17 更新：下面"二、发现的问题"中的 5 项已全部修复，并追加发现 1 个 P0
> （`gguf-py` 原本无法导入）。验证结果见文末"五、修复与验证记录"。**

## 二·补 P0 — `gguf-py/gguf/constants.py` 无法导入（原审计漏掉的）

`MODEL_TENSOR` 里 `SSM_G` 被定义了两次：

- 上游 `master`：`SSM_G = auto() # Kimi K3 (full-rank KDA gate…)`
- fork：`SSM_F` + `SSM_G # bailingmoe3 (no_kda_lora)`

移植时把 fork 的两个条目塞进了已经有上游 `SSM_G` 的树 → 重复成员 →
`TypeError: 'SSM_G' already defined as 104`。**已用 HEAD 版本复现确认这是既有问题。**

后果：`import gguf.constants` 直接失败，`convert_hf_to_gguf.py` 和所有 gguf-py 工具全废。
原审计只做文本比对、没有真正 `import` 过 Python 文件，所以漏掉——教训见文末。

修复：删掉 fork 那个 `SSM_G`（保留上游的），注释标明它同时服务 Kimi K3 与 bailingmoe3；
`TENSOR_NAMES` 同样删重复项。两者要的字符串都是 `blk.{bid}.ssm_g`，合成一个成员是正确的。

## 一、核对无误（逐字节级证据）

把两侧文件做 `tr -d '\r'` 归一后比对 md5，**以下文件完全相同**：

| 文件 | 行数 | 含义 |
|---|---:|---|
| `ggml/src/ggml-cuda/turbo-quant.cuh` | 453 | 2/3/4-bit Lloyd-Max 质心、WHT 符号表、FWHT、InnerQ |
| `ggml/src/ggml-cuda/turbo-innerq.cuh` / `.cu` | 34 / 32 | InnerQ 跨 TU 状态 |
| `ggml/src/ggml-cuda/turbo-wht.cu` / `.cuh` | 189 / 5 | WHT CUDA kernel |
| `ggml/src/ggml-turbo-quant.c` | 1030 | 参考量化/反量化实现 |
| `ggml/src/ggml-quants.c` / `.h` | 5675 / 136 | 含 turbo 参考入口 |
| `ggml/src/ggml-cuda/convert.cu` / `dequantize.cuh` / `getrows.cu` | — | 类型转换与反量化 |
| `src/turbo-rotation-data.h` / `-32.h` | 4103 / 71 | attention rotation 数据（确被 `llama-kv-cache.cpp:613,761` include） |
| `tests/test-turbo-quant.c` | 67 | turbo3/turbo4 往返测试 |
| `template-instances/fattn-vec-instance-*turbo*.cu` | 21 个文件 | 全部 21 种 K/V 组合 |

→ **WHT 旋转、质心码本、FWHT 归一化、InnerQ 的数值行为与 fork 一致**，不存在"移植时算错"的可能。

其余差异均为合法适配，逐条确认：

- `ggml/src/ggml-common.h` 唯一差异是上游自己的守卫演进
  `defined(GGML_COMMON_IMPL_C)` → `|| defined(GGML_COMMON_IMPL_CPP)`。**保留是对的**。
  即 `block_turbo2_0`(34B/128)、`block_turbo3_0`(50B/128)、`block_turbo4_0`(68B/128) 与 fork 完全一致。
- `ggml/src/ggml-cuda/mmvq-tq.cu` 唯一差异：`ctx.cublas_handle(id)` → `ctx.cublas_handle()`（上游 API 改名）。
- `ggml/src/ggml-cuda/fattn.cu` 的全部"差异"是注释改写 + 上游命名方案换代
  （`GGML_TYPE_TURBO2_0` → 宏缩写 `TURBO2_0`）。**21 种组合一个不少**，
  且 `ggml/cmake/common.cmake::ggml_cuda_fattn_vec_instances()` 的
  `TQ_FA_COMBINATIONS` 生成的 `GGML_CUDA_FA_TURBO2_0_TURBO3_0` 与
  `fattn.cu::FATTN_VEC_CASE` 展开名严格匹配；该 cmake 函数对缺失实例文件是
  `FATAL_ERROR`，所以少一个文件会在 configure 期就失败。
- `src/llama-graph.cpp` 的 `ggml_turbo_wht` 调用点比 fork **多**（7 处 vs 5 处）：
  上游新增的注意力变体（ISWA / top-k / DSA）都被补上了。这是正确的扩散，不是遗漏。
- 类型号重排自洽：`ggml.h` `Q2_0=42`(保留上游) / `TURBO2_0=43` / `TURBO3_0=44` / `TURBO4_0=45` /
  `TQ3_1S=46` / `TQ4_1S=47`；`llama.h` `LLAMA_FTYPE_MOSTLY_TQ3_1S=43` / `TQ4_1S=44` 与
  `gguf-py` 一致；`common/arg.cpp`(KV 类型表)、`tools/quantize/quantize.cpp`(outtype 表)、
  `src/llama-model-loader.cpp`、`src/llama-quant.cpp` 全部同步。
- `GGML_OP_COUNT=102` + `RPC_PROTO_PATCH_VERSION=1`。fork 用的是 4，但那是 fork 自己的
  op 枚举布局决定的；本树枚举不同，用 1 正确。且该 `static_assert` 确实可达
  （`ggml/src/ggml-backend-reg.cpp:70` include 了 `ggml-rpc.h`）。
- `GGML_OP_TURBO_WHT` 覆盖完整：`ggml.h` 枚举、`ggml.c` 名称/builder、
  `ggml-cpu/ggml-cpu.c`(3 处)、`ggml-cpu/ops.{h,cpp}`、`ggml-cuda/ggml-cuda.cu`(2 处)。
- `src/llama-memory.h`（6 处）、`src/llama-memory-hybrid.{cpp,h}`（6/4 处）、
  `tools/llama-bench/llama-bench.cpp`（7 处）的差异**全部是注释改写**，
  符号集合一致，无遗漏。
- 旋转开关语义正确（value-based，不是 presence-based）：
  `atoi(LLAMA_ATTN_ROT_K_OVERRIDE) != 0`、`TURBO_AUTO_ASYMMETRIC` 用 `env[0]=='0'` 判定关闭、
  `TURBO_LAYER_ADAPTIVE` 用 `atoi`。没有 `X=0` 反而打开之类的反转。
- Metal/Vulkan 无 turbo kernel **是干净拒绝**：`ggml-vulkan.cpp:19308` 的 `fa_kv_ok`
  白名单 `default: return false`，turbo KV 落到 CPU 回落，不会静默算错。与文档一致。
- `test-quantize-fns` 对 turbo2/3/4 打印 `skipped: rotated-domain KV quant` —
  **fork 里是同样的跳过**（两侧代码相同，只是行号不同），不是本树丢失的覆盖。

### 实测证据

```
D:\llama-build\cpu\bin\test-turbo-quant.exe   → exit 0
  turbo3 e0:      Cosine=1.000000  OutNorm=0.999808
  turbo3 sin*10:  Cosine=0.986448  MSE=1.345
  turbo4 cos*5:   Cosine=0.988753  MSE=0.286
D:\llama-build\cpu\bin\test-quantize-fns.exe → exit 0
```

## 二、发现的问题

### P1 — `tests/test-backend-ops.cpp` 完全没有 turbo（0 处引用）

fork 在该文件里有：

- `test_turbo_wht`（forward/inverse WHT 算子测试）
- `test_turbo_wht_roundtrip`（forward∘inverse = identity）
- `test_set_rows_turbo3`（带误差上限，验证 `f32 → WHT → PolarQuant → dequant → f32` 全链路）
- FA 的 KV 类型列表含 `GGML_TYPE_TURBO3_0`

**后果**：本树没有任何后端级测试能覆盖 turbo kernel。`MERGE_TURBO_KVSTREAM.md` 把它归因为
"CPU-only 设备下 harness 跳过 CPU 后端"，这低估了问题——即使插上 GPU，也没有 turbo 用例可跑。
CUDA 侧 21 个 FA 实例 + `set_rows` + `TURBO_WHT` 目前只有"能编译"这一层证据。

修：把 fork 的这 4 组用例搬回来（注意它引用的 `ggml_turbo_wht` 签名与上游一致，
`GGML_TYPE_TURBO2_0/3_0/4_0` 只是编号不同）。

### P1 — D=640 的显式模板实例化缺失

fork：
- `template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_16.cu` 含
  `DECL_FATTN_MMA_F16_CASE(640, 512, 1, 16);`
- `template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_16.cu` 含
  `DECL_FATTN_MMA_F16_CASE(640, 512, 2, 16);`
- `template-instances/fattn-tile-instance-dkq640-dv512.cu` 整个文件

本树：前两处都没有，第三个文件也不存在。而 `fattn.cu:2629/2631` 正是调用
`ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 1, 16>` / `<640, 512, 2, 16>`。

**根因**：两侧 `template-instances/generate_cu_files.py` **逐字节相同**，而该脚本的
skip 规则 `head_size_kq not in (192, 320, 576) and ncols2 in (16, 32): continue`
永远不可能生成 640 → fork 那几行是**手改过的生成产物**，脚本无法复现。
本树丢掉它们不是"忘了打补丁"，而是重新生成过实例文件。

**实际危害：不是 link error。** `ggml_cuda_flash_attn_ext_mma_f16_case` 的**定义体就在
`fattn-mma-f16.cuh` 头文件里**，而 `fattn.cu:4` `#include "fattn-mma-f16.cuh"`，
所以调用点会隐式实例化并正常链接；`fattn-mma-f16.cuh` 里也没有针对 640 的
`extern template` 声明。（`fattn-tile` 那条更无害：本树 `fattn-tile.cu` 里没有 `case 640`。）

**危害在于脆弱**：下次重跑生成脚本或上游同步，这两个隐式实例化会被继续遗忘；
而且它掩盖了"实例清单与分派表已不一致"这个事实——如果哪天上游给 640 补上
`extern template` 声明，立刻变成 undefined symbol。

修：在 `fattn-mma-f16.cuh` 的 extern 列表和两个实例文件里补 640 条目，
**并同时改 `generate_cu_files.py` 的 skip 元组**（把 640 加进 `(192, 320, 576)`），
否则下次生成又会丢。若 `fattn-tile` 的 640 也要，同样补文件 + 脚本。

### P2 — `fattn-vec.cuh` 丢了 `if constexpr (!V_is_turbo)` 守卫

fork：

```cpp
#ifndef GGML_USE_HIP
        if constexpr (!V_is_turbo) { __syncwarp(); }
#endif // GGML_USE_HIP
```

本树（`fattn-vec.cuh:426`）：无条件 `ggml_cuda_syncwarp();`

`ggml_cuda_syncwarp()` 在 CUDA 上就是 `__syncwarp()`、HIP 上为空，等价于 fork 的
`#ifndef GGML_USE_HIP`；而此处控制流是 uniform（不在任何发散分支里），
所以**正确性无害**。但 turbo V 路径每次外层循环会多执行一个 no-op warp barrier，
而这正是 fork 刻意用 `if constexpr (!V_is_turbo)` 省掉的开销
（fork 注释明确提到 turbo 路径对 warp 级开销敏感）。

修：加回守卫 `if constexpr (!V_is_turbo) { ggml_cuda_syncwarp(); }`。

### P2 — `gguf-py/gguf/constants.py` 两处不一致

1. 把 `TURBO2_0 = 43` / `TURBO3_0 = 44` / `TURBO4_0 = 45` 加进了 `GGMLQuantizationType`，
   但 **`GGML_QUANT_SIZES` 里没有对应条目**（fork 的 gguf-py 根本没有这三个枚举，
   所以是本树新引入的半拉子状态）。`gguf_reader.py:351` 和 `quants.py:15,22,94` 都是
   **直接下标** `GGML_QUANT_SIZES[t]` → 真遇到 type 43/44/45 会抛裸 `KeyError`。
   影响低（turbo 是运行时类型，不会进 GGUF），但枚举与尺寸表应当一致。
2. 同一个 commit 把 fork 的 `constants.py` 整文件搬了过来，带进了被
   `MERGE_TURBO_KVSTREAM.md` 明确声明"已排除"的架构定义：
   `MODEL_ARCH.INKLING`、完整的 `MODEL_TENSORS[MODEL_ARCH.INKLING]`、
   `MODEL_TENSOR.ATTN_R / ATTN_REL_PROJ / SHORTCONV_K / SHORTCONV_V / SHORTCONV_ATTN /
   SHORTCONV_MLP / FFN_GSCALE`、`TENSOR_NAMES` 对应项、`VisionProjectorType.INKLING`。
   而 C++ 侧 `grep -rn "LLM_ARCH_INKLING" src/` **命中 0** →
   转换器会产出运行时认不出的 GGUF（回落 `LLM_ARCH_UNKNOWN`）。
   （对照：`KIMI_K3` / `BAILINGMOE3` 在 C++ 侧有 `LLM_ARCH_KIMI_K3` 等，是自洽的，不算问题。）

修：删掉 INKLING 相关条目（含它的 MODEL_TENSORS/TENSOR_NAMES/VisionProjectorType），
或给 turbo 三个类型补 `GGML_QUANT_SIZES`（`TURBO2_0: (128, 2+128//4)` 等）。

### P3 — inkling 残留死代码

`src/llama-kv-cache.h:256` 声明 + `src/llama-kv-cache.cpp:2511`（约 86 行实现）
+ `src/llama-kv-cache.cpp:3591`（context 转发）的 `set_input_pos_rel_flat()`，
唯一调用方是已被排除的 `src/models/inkling.cpp`。本树里是纯死代码。

这是"排除 fork 特性时钩子留在共享文件里"的典型残留。删掉即可（或保留，
但至少要意识到它没有任何调用方）。

## 三、本次未验证 / 无法验证

- **CUDA 未跑通链路**：本机无 NVIDIA 设备。上面关于 640 隐式实例化的结论来自
  "定义体在头文件 + 调用点 include 了该头文件"的源码事实，**不是**一次成功链接的实证。
  建议在 V100 机器上做一次 `ninja ggml-cuda` 全量链接 + 一个 MLA 模型
  （`glm-4.7-flash` 一类，head_dim 576→640）的 `-ctk turbo4` 端到端生成。
- **没有重量级数值验证**：fork 自己的 PPL/KLD 数字（7 个模型家族、旋转策略对比）
  没有复测，`MERGE_TURBO_KVSTREAM.md` 里那些数字仍然只是 fork 的数字。
- Metal / Vulkan 未编译（本就未移植）。
- `--cache-reuse` / KV streaming 路径与 turbo 的组合未验证（不属本次审计范围）。

## 四、建议的修复顺序

1. 补齐 `tests/test-backend-ops.cpp` 的 turbo 用例（P1）。
2. 修 `gguf-py` 的 `SSM_G` 重复（P0，否则转换器完全不可用）。
3. 补 640 的实例化条目 **并同步改 `generate_cu_files.py`**（P1）。
4. 清理 `gguf-py` 的 INKLING 残留 + 补 `GGML_QUANT_SIZES`（P2）。
5. 加回 `fattn-vec.cuh` 的 `!V_is_turbo` 守卫（P2，性能）。
6. 删 `set_input_pos_rel_flat`（P3）。

## 五、修复与验证记录（2026-09-17 已全部应用）

| # | 改动 | 文件 | 验证 |
|---|---|---|---|
| P0 | 删重复的 `SSM_G`（保留上游 Kimi K3 那个，注释标明同时服务 bailingmoe3）；`TENSOR_NAMES` 同步 | `gguf-py/gguf/constants.py` | `constants.py` 导入成功（HEAD 版本导入失败，已复现对照） |
| P1 | 新增 `test_turbo_wht` / `test_turbo_wht_roundtrip` / `test_set_rows_turbo3` 三个用例类 + 注册；新增 turbo3 FA 小循环（hs 128/256 × kv 113/512 × nb 1/32） | `tests/test-backend-ops.cpp` | CPU 后端：**TURBO_WHT 27/27**、**SET_ROWS_TURBO3 21/21**、**FA turbo3 8/8** 全通过 |
| P1 | 两个实例文件补 `DECL_FATTN_MMA_F16_CASE(640,512,{1,2},16)`；头文件补对应 `extern`；生成器加 `MMA_EXTRA_HEAD_SIZES_KQ=[640]`、`V_OVERRIDE 640→512`、`skip` 规则 | `fattn-mma-f16-instance-ncols1_{1,2}-ncols2_16.cu`、`fattn-mma-f16.cuh`、`generate_cu_files.py` | 在临时目录跑生成器：输出的这两个文件与手工编辑**逐字节一致**；未生成 `fattn-tile-dkq640`（符合预期）；其余生成文件与树一致 |
| P2 | 无条件 `ggml_cuda_syncwarp()` → `if constexpr (!V_is_turbo) { ggml_cuda_syncwarp(); }` | `ggml/src/ggml-cuda/fattn-vec.cuh` | sm_80 编译通过（见下） |
| P2 | 删 `MODEL_ARCH.INKLING`、`MODEL_TENSORS[INKLING]`、7 个 inkling-only `MODEL_TENSOR`、`TENSOR_NAMES` 对应项、`VisionProjectorType.INKLING`；给 `GGML_QUANT_SIZES` 补 `TURBO2_0 (128,34)` / `TURBO3_0 (128,50)` / `TURBO4_0 (128,68)` | `gguf-py/gguf/constants.py` | 导入验证：`TURBO2_0/3_0/4_0` 尺寸与 C++ `block_turbo*_{0}` 一致（34/50/68 B per 128）；`INKLING` 归零；`KIMI_K3`/`BAILINGMOE3` 保留 |
| P3 | 删除 `set_input_pos_rel_flat` 的声明、86 行实现、context 转发 | `src/llama-kv-cache.{h,cpp}` | `grep` 残留 0；CPU 全量重编通过（`llama.dll` 链接成功） |

### 编译 / 运行验证

```
CPU  (D:\llama-build\cpu,  MSVC 14.40, Release)
  ninja test-backend-ops                 → BUILD OK（134 步，含 llama-kv-cache.cpp 重编）
  test-backend-ops -b CPU -o TURBO_WHT        →  27/27 passed,  Backend CPU: OK
  test-backend-ops -b CPU -o SET_ROWS_TURBO3  →  21/21 passed,  Backend CPU: OK
  test-backend-ops -b CPU -o FLASH_ATTN_EXT -p turbo3 → 8/8 passed, Backend CPU: OK
  test-turbo-quant                            → 通过（turbo3 cos 1.000/0.986, turbo4 0.989）
  test-quantize-fns                           → 通过

CUDA (D:\llama-build\cuda, arch sm_80, 无 GPU，仅编译)
  fattn-vec-instance-turbo3_0-turbo3_0.cu.obj      → OK
  fattn-vec-instance-turbo3_0-turbo4_0.cu.obj      → OK
  fattn-mma-f16-instance-ncols1_1-ncols2_16.cu.obj → OK（含新增的 640 显式实例化）
  fattn-mma-f16-instance-ncols1_2-ncols2_16.cu.obj → OK
```

**重要更正**：`MERGE_TURBO_KVSTREAM.md` 说 "test-backend-ops cannot validate the turbo
kernels with a CPU-only device" ——**这个说法不成立**。CPU 后端能跑 turbo FA、
SET_ROWS turbo3、TURBO_WHT（走 CPU 参考实现），上面 56 个用例就是证据。
原来的缺口纯粹是"没有测试用例"，不是"设备不支持"。

### 仍然未验证

- **CUDA 全量链接**：本机无 NVIDIA 设备，只做了 4 个目标文件的编译，没跑完整 `ninja ggml-cuda`
  和链接。D=640 那条现在有了显式实例化，理论上更稳，但仍未在真实 GPU 上验证。
- 未在 V100 上跑 MLA 模型（576→640）+ `-ctk turbo4` 端到端。
- fork 的 PPL/KLD 数字未复测。
- Metal / Vulkan 未编译。
- 全量 `test-backend-ops`（所有 op）没跑完，只跑了 turbo 相关 + 构建期回归。

### 教训（写给自己）

1. **审计 Python 文件时必须真的 `import` 一次**，不能只做文本比对。本例中 `SSM_G` 重复
   是纯粹的运行时错误，任何 grep/diff 都看不出来，而它让整个 gguf-py 不可用。
2. **"fork 有 / 本树没有"不等于缺陷**：`test-quantize-fns.cpp` 的 `dot_product_error`
   看起来少了一个缓冲区修复，实际是上游重构后用 `ggml_row_size()` 按类型直接算对，更优。
   必须读全文 + 查 `git log master..HEAD` 才能判定方向。
3. **模板实例文件要验生成器，不验文件**：在临时目录跑一遍生成脚本再 diff，
   是唯一能证明"这些条目可复现"的方法（脚本会 `os.remove` 当前目录所有 `*.cu`，
   **千万不要在真实目录里跑**）。
