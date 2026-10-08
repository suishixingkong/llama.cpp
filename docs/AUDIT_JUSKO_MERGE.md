# 合并审计：jusko-llama-volta-qwen3flash → llama.cpp（turbo-kvstream-merge）

审计时间：2026-09-17
审计范围：`25d31990b`（Volta 端口，15 文件 +1070/−49）与 `057085a32`（recurrent-checkpoint 端口，12 文件 +285/−25）
方法：静态核对合并文档 `MERGE_JUSKO_VOLTA.md` 的每一项声明，对照 `llama.cpp` 集成分支当前代码（HEAD = `cb599aae4`，工作树干净）。

## 结论先行

**合并整体忠实、完整，文档与代码一致，未发现确凿的代码错误。**
所有 7 项 Volta 特性 + C 段 recurrent-checkpoint 均已在代码中落地，开关接线、架构守卫、模板默认参数、CLI 注册、依赖闭包全部正确。
之前工作记忆中标记的「`--prefill-reuse` 漏 `checkpoint_ubatch`」问题**已修复**（`server-context.cpp:3602-3604`）。

剩余的全部是**风险/陷阱/范围声明**，而非已证实的 bug。按严重程度排列如下。

---

## 核对清单（逐项确认已落地）

| # | 声明项 | 代码位置 | 结果 |
|---|---|---|---|
| 1 | `fattn-q8-volta.cuh` 新文件，sm70 守卫 | `fattn-q8-volta.cuh:16,209` `#if !defined(__CUDA_ARCH__) \|\| __CUDA_ARCH__ == 700` | ✅ |
| 2 | `get_config_volta` 三行新 tile + `flash_attn_ext_f16_volta_compact` + `nbatch_fa_layout` override + 3 处 `-1,-1,false` 调用点 | `fattn-mma-f16.cuh:123-139, 2043, 1971/1976/2020` | ✅ |
| 3 | `fattn.cu` 调度：`BEST_FATTN_KERNEL_VOLTA_Q8_W4` 枚举/分配/派发、Q8 精确几何判定、GQA8_NCOLS2=2 | `fattn.cu:2771, 2950-2959, 3047-3092, 2419-2427` | ✅ |
| 4 | `mmvq.cu/.cuh` `q5_x4` 变体 + `Q6_W4R4`（case 4 臂 `rows_per_block=4`），默认模板参数 | `mmvq.cu:624, 670, 813, 1121-1123, 1318-1327` | ✅ |
| 5 | `mmq.cuh` Q6_K `J>=48` → Pascal DP4A（host + device）；`FORCE_MMQ=moe` | `mmq.cuh:254-257, 285`；`mmq.cu:334-337` | ✅ |
| 6 | `ggml-cuda.cu` + `common.cuh` + `--prefill-reuse`：proc-address 接线、Volta 门控 `reuse_n=0` | `ggml-cuda.cu:2643-2648, 7104, 7324`；`src/llama-context.cpp:376-379` | ✅ |
| 7 | `gated_delta_net.cu` `128x4_volta`，仅 `cc==VOLTA` 派发（无 Turing 门控） | `gated_delta_net.cu:369, 373` | ✅ |
| 8 | `checkpoint_ubatch` 修复：`min(n_ubatch, prefill_reuse)` | `server-context.cpp:3602-3604` | ✅（此前标记项已解决） |
| 9 | recurrent ckpt：`rs_rollback_prompt_only` / `LLAMA_STATE_SEQ_FLAGS_RECURRENT_PREV` / `is_replay_boundary` 默认 false / `snapshot_prev` 两处 | `llama-cparams.h:17`, `include/llama.h:374,927`, `server-context.cpp:2316,3613,3880-3891` | ✅ |
| 10 | CLI 注册：`--prefill-reuse`、`--checkpoint-recurrent-prev` / `--no-checkpoint-recurrent-prev` | `common/arg.cpp:1685,1716-1717` | ✅ |

---

## 风险 / 陷阱（非已证实 bug，但值得关注）

### 🔴 HIGH — 整个 CUDA 端口未经硬件执行验证（最大风险）
文档已诚实声明：本机无 NVIDIA 设备，只过了 sm_70 编译，没有任何端到端生成、token/形状交叉校验。
- **compact 内核共享内存尺寸未重新推导**：`fattn-mma-f16.cuh:2275` `nbytes_shared_compact = 49152`（48 KiB）直接取自 fork 常量，`cudaFuncSetAttribute(..., 49152)` 已设置；若偏小，症状是共享内存静默损坏，只有设备运行才暴露。
- **`--prefill-reuse` 的 REPACK/cuBLAS 交互未验证**：`ub 768` 下复用一次反量化权重的列瓦片，repackage 布局是否仍为 cuBLAS 路径所期望，是运行期问题。
- 依赖闭包上：fork `be6567cea` 依赖 `0471a9885` 的 server 半壁，本端口按「只取该功能所需增量」处理（未整体引入 value-based eviction），这点已记录在文档，逻辑自洽。

### 🟠 MEDIUM — 语义陷阱，最可能导致误用
- **`GGML_CUDA_VOLTA_Q8_FATTN_TC` / `Q5_X4` / `Q6_W4R4` 用 `getenv(...) != nullptr` 判定**：设 `=0` **仍然启用**，必须 `unset` 才关。若用户或脚本设 `=0` 想关闭，会静默保持开启。文档已注明，但这是真实 footgun，建议改判据为 `atoi(env) != 0`，或在 help 里醒目标注「设 0 不关」。
- **A1（Q8_FATTN_TC）只在唯一几何下触发**：实际判定（`fattn.cu:2951-2955`）要求 Qwen3.8-27B 精确形状（Q ne[0]=256, ne[1]=4, ne[2]=24；K ne[2]=4；q8_0 KV；无 softcap；有 mask；无 KQV src[4]）。其它模型/形状静默退回通用 FA。激活表写成「q8_0 KV 张量核注意力」过于宽泛——它本质是**单模型单形状特化**，不要误以为通用 q8_0 KV 加速。

### 🟡 LOW — 范围/清晰度
- `GGML_CUDA_VOLTA_FA_COMPACT` 的 `=0` 关闭逻辑**安全**：`fattn-mma-f16.cuh:2242-2243` 先判 `getenv == nullptr` 再 `atoi`，无空指针崩溃——已确认与文档一致，不是 bug。
- `--prefill-reuse` 需 F16 compute + 量化权重 + Volta（`ggml-cuda.cu:2643-2648`），非 Volta 上 `reuse_n=0` 静默 inert——符合文档，但用户可能误以为生效。
- `tests/test-recurrent-state-rollback` 中的 `test_rollback` 在本树失败（上游 qwen35 dirty-restore 缺陷，非本端口引入；回退到 pre-port 提交复现相同数值）——属已知失败测试，运行回归时注意区分。

---

## 建议（按优先级）

1. **最高优先：在 V100 上实跑验证。** 重点：compact 内核共享内存尺寸、Q8 路径数值正确性、`--prefill-reuse` 在 `ub 768` 下的 REPACK/cuBLAS 数值一致性。文档里的所有速度数字都是 fork 自测值，本树未复测。
2. **消除 footgun**：把三个 `!=nullptr` 语义开关改为 `atoi(env) != 0`（或显式 "0/1"），至少在校核文档/help 用醒目方式写「设 0 不关」。
3. **修正范围声明**：在文档与 `--help` 明确 A1 的精确触发形状，避免误以为通用 q8_0 KV 加速。
4. **后续若验证 SM75/Turing**：再放开 `gated_delta_net_cuda_128x4_volta` 的 Turing 门控（当前仅 `cc==VOLTA`，符合「仅 Volta 测过」原则）。

---

## 已排除的疑虑（复核记录）
- 工作记忆曾标记「`--prefill-reuse` 漏 `checkpoint_ubatch`」：已确认在 `057085a32` 中修复（`min(n_ubatch, prefill_reuse)`），不再成立。
- `fattn-q8-volta.cuh` 是否会被非 sm70 误编译：有 `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700` 守卫，仅 sm70 与 host 实例化 mma.sync，安全。
- `process_tile` 新增模板参数：仅 compact 内核内部传 `96,64,true`，三处通用调用传 `-1,-1,false`；既有 `template-instances` 通过顶层 case 宏调用，签名未变，实例化不受影响。
