# `--mmproj-gpu-swap` 移植记录（buun → buun-port）

日期：2026-09-24 ｜ 分支：`buun-port`
来源：`buun-llama-cpp` 的 `8e64d7a3a`（引入提交）+ 后续精化提交中属于本特性的部分
计划依据：`BUUN_MERGE_PLAN.md` §三 功能 #3（难度「中」）

## 1. 特性语义

在显存受限的卡上，MTP 草稿上下文与 GPU 上的 mmproj（多模态投影器）**不能同时驻留**。
`--mmproj-gpu-swap` 让二者**相位互斥**：

1. 启动时投影器载入 **CPU**（不占显存）；
2. 请求首次遇到图像时，**销毁 MTP 草稿上下文**腾出显存；
3. 在当前进程内把投影器**重载到 GPU**，完成图像编码（在 CPU 上需 30–60 s，GPU 上 1–2 s）；
4. 媒体段结束后**卸载 GPU 投影器**并**重建 MTP 上下文**，恢复投机解码。

只在 `LLAMA_EXAMPLE_SERVER` 下注册，无环境变量。

## 2. 上游提交是个 bundle —— 拆解与取舍

`8e64d7a3a` 的 commit message 自己列了四件事，本次移植只取第一件：

| bundle 成员 | 处置 | 理由 |
|---|---|---|
| **`--mmproj-gpu-swap` 本体** | ✅ 移植 | 目标特性 |
| auto-fit：在 n_parallel 翻倍**之前**跑 fitter | ❌ 未移植 | 依赖 buun 的 `server_recurrent_expansion_state` / `recurrent_expanded` / `n_parallel_user` 体系，本树没有该重构。**后果**：本树下自动 fit 的页边距不考虑投影器的相位需求，需要 `-c` 显式指定时才精确。属已知局限，非缺陷。 |
| np=1 + spec 时自动启用 `kv_unified` | ❌ 未移植 | 与投影器无关的独立改动，buun 自己也随后把它拆成单独提交（`fee9ddd54`）。是否要这条应单独决策。 |
| `has_type()` helper（替代 verbose `std::find`） | ✅ 移植 | 本特性到处用它 |
| 提取 `make_mmproj_params()` / `reload_mmproj()` | ✅ 移植 | 重载路径的前提 |

后续精化提交（`4c714197f` / `143150400` / `396469835` / `fea2e08f0`）里与本特性相关的部分
是对 **external DFlash** 的支持（相位互斥扩到可从磁盘重载的外部 drafter）。**未移植**，理由见 §4。

## 3. 本树结构差异与适配

buun 的 `server-context.cpp` 是 22995 行（4× 本树 5634 行），draft 生命周期完全不同。
逐 hunk 强行移植不可行，因此按本树结构**重表达**，语义对齐上游：

| 关注点 | buun | 本树（适配后） |
|---|---|---|
| draft 上下文所有权 | `ctx_dft` 为 `shared_ptr`，`ctx_dft.reset(create_mtp_context())` 独立创建 | `spec_init`（`common_speculative_init_result`）持有 model+context，重建走 `common_speculative_init_from_params()` |
| 草稿上下文创建 | 独立 `create_mtp_context()` 直接 `llama_init_from_model` | 复用 `common_base_params_to_speculative()` + `common_speculative_init_from_params()`，与 `load_model()` 的首建路径同源（**避免两条创建路径漂移**） |
| slot 绑定 | `slot.spec_shared` + `slot.get_spec()` | `slot.spec`（原始指针）+ 成员 `spec` 统一 |
| slot 上的投影器状态 | `slot.mbatch` | 同，且**本树 `mbatch` 与 `mctx` 强绑定**（`mtmd_batch_init(mctx)`）⇒ `reload_mmproj()` 必须先 `slot.mbatch.reset()`，否则悬空 |
| 泛化到外部 drafter | `swap_spec_to_mmproj_gpu()` 按 `model_dft` 分支 | 只在 `!has_draft && spec_mtp`（纯 MTP）时启用；不满足则**告警并保持两者常驻**（fail-open，不改变原有行为） |

### 3.1 一个被移植进来的真实缺陷

上游的 `unbind` 只清 `slot.spec_ckpt`（他们的 slot 结构不同）。本树**必须成组清**
`spec_draft` + `spec_i_batch` + `spec_ckpt`：`server-context.cpp:3260` 在复用「partial draft」时
断言 `GGML_ASSERT(!slot.spec_ckpt.empty())`——

```cpp
if (!slot.spec_draft.empty()) {
    if (use_ckpt_tgt) { GGML_ASSERT(!slot.spec_ckpt.empty()); }   // ← 只清 ckpt 会在此 abort
}
```

即「清了 checkpoint 但留下 draft」会让恢复后的第一次草稿生成直接 abort。
本树 `slot.reset()` 清的就是这三件套（:383-388），`bind_slots_to_draft_context()` 对齐它。

### 3.2 交换期间的一致性论证

交换期间 `spec`（成员）为 null、`slot.spec` 为 null，本树所有投机入口都有守卫，逐条核对：

| 入口 | 守卫 | 位置 |
|---|---|---|
| 草稿生成 | `if (spec)` | :3246 |
| 草稿参数 / `can_speculate()` 断言 | 同上，且 `can_speculate()` 即 `!!spec` | :3247-3255, :471 |
| 采样与接受 | `iterate(...)` 内 `!slot.can_speculate()` 早退 | :4217 |
| `common_speculative_begin` | `if (slot.can_speculate())` | :4157 |
| mmproj 编码回调传 `slot.spec` | `common_speculative_process(nullptr, …)` 安全返回 true | speculative.cpp |

⇒ 交换期间相关 slot **退化为纯目标解码**（正确性由目标验证保证），不会崩。

### 3.3 异常安全

`server_mmproj_restore_guard`（RAII，`restore_now()` 先清 armed 再回调 ⇒ 恰好恢复一次）
在媒体段开始即 arm。`process_mtmd_chunk` 是经 `queue_tasks.yield_to_queue()` 调用的，可能
把媒体异常**或并发回调的异常**抛回来；guard 的析构处理这些路径，正常路径则在媒体循环后
显式 `restore_now()`。

## 4. 刻意未移植 / 已知局限

1. **外部 drafter（DFlash 等）不参与交换**。它会从磁盘重新加载模型，代价与本特性目标
   （省显存、别重建太多）冲突。不满足条件时打警告并保持常驻 —— **不静默改变行为**。
2. **auto-fit 的相位感知页边距未移植**（见 §2）。用 `-c` 显式指定上下文时无影响。
3. **`kv_unified` 自动启用未移植**（见 §2）。与本特性正交。
4. 上游后来把「投影器/草稿交换」的待处理状态接进了投机调度（`projector_swap_pending`，
   避免对着即将销毁的 owner 排队草稿）。本树不需要：交换**发生在该 slot 的媒体段之前**，
   期间 `spec` 已为 null，草稿入口整体短路（§3.2）。

## 5. 验证状态（重要）

**仅编译验证，未运行。** 本机无 GPU、无多模态模型。

- `llama-server` 目标 BUILD OK，**实测重编 102 个 TU**（含 `server-context.cpp` / `common.cpp` /
  `speculative.cpp`），产物时间戳刷新 —— 非「脚本假报 OK」。
- 连带重编 `llama-cli` / `llama-bench` / `llama-perplexity` / `llama-mtmd-cli`（各 6 TU，含重链）
  BUILD OK，确认 `common.h` 改动无回归。

**未验证、需在有卡机器上确认的**：

1. 交换的实际显存释放量是否足够放下投影器（取决于模型组合）；
2. `reload_mmproj(true)` → 图像编码 → `reload_mmproj(false)` → MTP 重建的端到端时延；
3. 交换后**其他 slot** 的投机接受率（其草稿状态按 §3.3 被丢弃，预期退化为无投机，**不影响输出正确性**）；
4. `slot.mem.init(ctx_tgt, nullptr)` 之后本树各路径对 `mem.ctx_dft == nullptr` 的处理是否完备
   （`common_memory::seq_*` 已有 `if (ctx_dft)` 守卫，其余调用点未逐一枚举）。

建议的验证命令（有卡机器）：

```bat
llama-server -m <MTP 模型> --mmproj <投影器> --spec-type draft-mtp --spec-draft-n-max 3 ^
  -ctk turbo4 -ctv turbo4 -c 32768 -ngl all -fa on --mmproj-gpu-swap
:: 预期启动日志：
::   loaded multimodal model on CPU (GPU swap enabled), '<mmproj>'
:: 首次带图请求的预期日志：
::   swapping MTP out of VRAM, loading mmproj to GPU...
::   MTP->mmproj swap done in <N> ms
::   unloading mmproj from the GPU, recreating MTP...
::   mmproj->MTP swap done in <N> ms
```

## 6. 后续

- 若需要外部 drafter 也参与交换，需先量「重新加载 drafter 的代价」是否值得；
- 若要移植 auto-fit 相位感知页边距，需先合入 buun 的 `recurrent_expansion` / `n_parallel_user` 体系
  （或在本树的 fit 路径上另写等价物）。
