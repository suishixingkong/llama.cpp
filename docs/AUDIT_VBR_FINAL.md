# VBR 移植终审：三面裁定与残差清单（buun-port）

日期：2026-10-04
对象：`llama.cpp` @ 分支 `buun-port`，HEAD **`4f0be29f9`**（slice `9l-d4b9u`）· 移植源 `buun-llama-cpp` · 三方祖先 `bfdc32183`
尺子：`D:\llama-build\` 下 6 把（§3–§6 各表给出命令）· 体检 `guard_sweep.py` · 过滤 `ctest`
本机限制：**无 N 卡**（数值/性能证据需回 V100；CUDA 测试的证据上限 = 编译+链接）

---

## 0. 裁定

**引擎 = 完成 ｜ server = 完成 ｜ 接线（`create_memory` 的 `vbr` 投递）= 完成 ｜ 测试面设备无关残差 = 0。**

本机可验证的 VBR 工作已到顶：残余 6 项**没有一件是设备无关的 VBR 工作**——R1–R3 是设备门控，
R4 已落地并在 V100 上验证（`1c8ff2e31`，2026-10-12），R5–R6 根本不是 VBR（§6）。

⚠️ **本终审不主张「fork 移植完成」**：fork 的 **QSA 生产代码**（`qsa_selection_safe` 等，非 VBR）**仍未移植**（R5），
该文件的其余 delta 是否排期需另行裁定。

---

## 1. 基线（可复现）

| 项 | 命令 | 结果 |
|---|---|---|
| 分支 / HEAD | `git -C llama.cpp log --oneline -1` | `4f0be29f9 vbr: deliver the runtime VBR params to the indexed hybrid memory` |
| 工作区 | `git -C llama.cpp status --porcelain \| wc -l` | `0` |
| 全量体检 | `python -u guard_sweep.py` | **136 守卫 / 0 failing**（`sweep_final_vbr.txt`）|
| 过滤 ctest | `cmd //c run_ctest_proj.bat` | **48/48 passed, 0 failed**，`CTEST_RC=0` |

---

## 2. 尺子清单（各面用哪把）

| 面 | 尺子 | 形态 |
|---|---|---|
| 引擎 `src/` + `ggml/src/` | 内联标识符覆盖（§3 给出全文脚本） | 标识符集合差 |
| server `tools/server/` | `vbr_server_triage.py` | 行级 + 角色/裁定分类 |
| 接线 `create_memory` | `scan_vbr_callsites.py`（**声明定义取自参考 fork**） | 调用点 + 声明差 |
| 测试面（`test-llama-archs`） | `attribute_llama_archs_vbr.py` | 按**宿主函数**归组 |
| 测试面（fork 独有文件 / 同名异量） | `classify_test_gap.py` | 文件级 + `fvbr` 列 |
| 成员/方法级（排期用） | `member_map.py` / `method_map.py` / `classify_member_gap.py` / `body_hunks.py` | 需给文件与行段，是**钻取**工具不是全局裁定 |
| ~~族缺口~~ | ~~`family_gap.py`（180/0）~~ | **作废**：正则抽样、多轮未动、对类成员盲。**不得引用** |

---

## 3. 面 1 —— 引擎（`src/` + `ggml/src/`）

度量（可复现：抽取两树 `src/`+`ggml/src/` 全部源码标识符中含 `vbr`（不分大小写）者，取集合差）：

| 度量 | 值 |
|---|---|
| fork 含 vbr 标识符 | **1061** |
| port 含 vbr 标识符 | **1063**（port 更多——如本片新增的参数） |
| **port 缺席** | **1** = `vbr_alpha_v_decode` |

`vbr_alpha_v_decode` 的裁定：**有意适配，0 工作量**。两条证据：

1. 它在 fork 里是 `ggml/src/ggml-cuda/fattn.cu` 的 `static inline`，唯一用途是为 VBR transcode 的 **V 侧**取
   **目标 tier 的解码 α**：`vscale = is_v ? 1.0f / vbr_alpha_v_decode(type_B, n_cells) : 1.0f`；
2. port **有** `vbr_dequant_turbo_to_f32`（`ggml/src/ggml-cuda/fattn.cu:3401`）与 `k_vbr_f16_to_f32_scaled`，
   其体内**写明**了适配与归属：
   `GGML_UNUSED(type_B);  // no decode alpha in this tree, so the target tier adds no scale`
   ⇒ port 里 α 恒为 `1.0`，助手函数不存在是**结论**而不是遗漏。归因切片 `9l-d4b8l`。

### ⚠️ 3.1 修正一条旧记：「CUDA 逐字节相同」不成立

`ggml/src/ggml-cuda/fattn.cu` 两树**差异 5,283 行**（port 163,383 B vs fork 153,424 B），
且 `tcq_compute_alpha_v` 在 port **0 命中**、fork **12 命中** ⇒ **port 的该文件是更新的上游**，不是 fork 的副本。

**正确表述**：*fork 的 VBR CUDA 入口在 port 齐备（`vbr_dequant_turbo_to_f32`、`k_vbr_f16_to_f32_scaled` 均在），
其函数体或与 fork 逐字相同、或带有 in-source 归属标记的适配*。旧记「逐字节相同」**标作废**。

### ⚠️ 3.2 修正一条旧记：CLI 旗标不是「24 个全在」

| 度量 | 值 |
|---|---|
| port 全树 `--vbr-[a-z0-9-]*` 不同**字符串** | **14**（含 1 个帮助文本裸前缀 `--vbr-`）|
| port **真正不同的旗标**（3 个是别名：`--vbr-budget` / `--vbr-min-bits` / `--vbr-vram-budget`）| **10**，另加前置开关 `-ctk vbr` / `-ctv vbr` |
| 上游 `master` 的 `common/arg.cpp` | **0**（⇒ 全部来自 fork 移植）|
| fork 全树 | **19** |
| **fork 有而 port 没有** | **5**，且**全部**是 `--vbr-adopt-cuda` / `--vbr-live-schedule-cuda` / `--vbr-occupied-restore-cuda` / `--vbr-transform-cuda` / `--vbr-transformed-destination-cuda` |

那 5 个不是产品 CLI，而是 `test-vbr-artifact-adopt` 的**测试驱动模式**（= **R3**，设备门控）。
⇒ 旧记「**24 个 `--vbr-*` CLI 全在**」**两处都错**（数字不是 24；也并非「全在」）：正确表述是
**10 个产品旗标全在（上游 0），fork 多出的 5 个是测试驱动的 CUDA 模式**。

> ⚠️ 计数口径要自洽：早先写下的「13」是把 **3 个别名串**各计了一次（14 − 裸前缀 = 13 串，
> 再去别名 = **10 个旗标**）。**记数时必须声明是否含别名**（`§68`）。

验证命令：`grep -roh -- '--vbr-[a-z0-9-]*' <tree>/ | sort -u | wc -l`，再 `comm -13` 求差。

---

## 4. 面 2 —— server（`tools/server/`）

`python -u vbr_server_triage.py`：缺席 **16 行**，role `decl=13 / include=1 / stmt=2`，
裁定 `COMPILE-TIME=14 / HOST-ABSENT=2`。

**逐名 grep 复核（不采信分类器，`§49.25`「仪器报的每条缺失都值得一次 grep」）**：

| 行 | 分类器说 | grep 复核 | 结论 |
|---|---|---|---|
| `server-context.cpp:6559 common_params_resolve_vbr_codec_auto(params_base);` | HOST-ABSENT | 该函数在 port **已定义且已被调用**：`common/common.cpp:1412`（定义）、`:1434`（调用） | **行缺失 ≠ 语义缺失** ⇒ 0 |
| `server-context.cpp:18344 !params_base.vbr_prompt_cache &&` | HOST-ABSENT | `vbr_prompt_cache` 在 port 命中 **13 个文件**（如 `common/arg.cpp:2539`） | **行缺失 ≠ 语义缺失** ⇒ 0 |
| `server-task.cpp:9 #include "server-vbr-artifact-store.h"` | COMPILE-TIME | 目标头文件**存在**：`tools/server/server-vbr-artifact-store.h`（28,436 B） | 编译期冗余 ⇒ 0 |
| `server-task.h` 12 × `enum class X;` + 1 × `friend class X;` | COMPILE-TIME | 抽验 3 名在 port 均命中：`vbr_adopt_phase` 87、`vbr_import_decision` 128、`server_prompt_cache_vbr_publication_metadata` 96（= fork 96） | 前置声明冗余 ⇒ 0 |

⇒ **server 面 16 行全部 0 工作量。**

---

## 5. 面 3 —— 接线（`create_memory` 的 `vbr` 投递）

`python -u scan_vbr_callsites.py`（**「谁接受 vbr」的定义取自参考 fork 的头文件**，不取自被测树）：

| 度量 | 值 |
|---|---|
| 声明点（6 个类 + 1 个自由函数） | port **7** / fork **7** |
| `DECLARATION delta`（fork 声明而 port 不声明） | **0** |
| 「fork 的声明要求 `vbr` 而调用点省略」 | **0** |
| 仪器自检 | `unbalanced signatures skipped: 0`；`known classes found: True`；退出码 **0** |

**尺子自身有分辨力**（在一次性临时树 `_ruler_mut/` 上做，工作区零风险）：

| 变异 | 目标 | 期望 | 实测 |
|---|---|---|---|
| baseline | 未改的副本 | delta 0 / verdict 0 | green |
| **mutA** | 删掉 ctor 的 `vbr` 参数 | delta 1 | **RED** |
| **mutB** | 保留参数、把 `fork's` 塞回块注释 | delta 0 | green（**该 bug 的回归测试**）|
| **mutC** | 保留声明、删掉调用点的 `vbr` | verdict 1 | **RED** |

⚠️ 该尺子在本轮**自己坏过一次**（注释里的撇号 `fork's` 使 hand-rolled 解析器吞掉 ctor 的 `)`），
修法与适用判据见技能 `§66`；复现脚本 `mut_scan_vbr_callsites.py`。

---

## 6. 面 4 —— 测试面

### 6.1 `attribute_llama_archs_vbr.py`（按宿主函数归组）

fork 独有代码行 **2511**，其中含 vbr **95** 行（= `fvbr` 列）：

| 宿主函数 / 作用域 | 行数 | 性质 |
|---|---|---|
| `test_qwen4_vbr_cuda` | 56 | 设备 |
| `test_backends`（`supports_vbr_vmm` 门后） | 19 | 设备 |
| `<file scope>`（`llama_kv_cache_vbr_epoch_test` 仪器结构） | 15 | 设备仪器 |
| `devices_support_vbr_vmm` | 4 | 设备能力探测 |
| `main`（调用点） | 1 | 调用点 |
| **合计** | **95** | **全部设备门控** |

### 6.2 `classify_test_gap.py`

- fork 独有测试 **46** 个，其中候选 **18** 个（6680 行）—— 全部判为 `DEVICE (not verifiable here)`；
- 同名异量行 **10** 条，其中**带 fork 独有 vbr 行的 3 条**：

| 文件 | fork → port | `fvbr` | 本终审的复核 |
|---|---|---|---|
| `test-llama-archs.cpp` | 4387 → 1145 | 95 | §6.1：全设备门控 |
| `test-vbr-artifact-adopt.cpp` | 6139 → 3676 | 296 | §6.3 |
| `test-recurrent-state-rollback.cpp` | 1838 → 599 | 27 | §6.4 |

### 6.3 `test-vbr-artifact-adopt.cpp`（本次终审新做的归因）

含 vbr 的 fork 独有行共 **345**（分类器用更严的 token 规则计得 296）：

| 位置 | 行数 | 归因 |
|---|---|---|
| 带内 `61-583`（model-adoption 骨架） | 42 | 既定排除 |
| 带内 `4042-5872`（GPU 半边） | 265 | 既定排除（设备） |
| 带内 `5938-5962`（DFlash） | 0 | 既定排除 |
| **带外** `5964-6136` = 驱动尾 | **37** | 五个模式 `--vbr-adopt-cuda` / `--vbr-transform-cuda` / `--vbr-live-schedule-cuda` / `--vbr-transformed-destination-cuda` / `--vbr-occupied-restore-cuda` —— **全部以 `-cuda` 结尾**（判据：`all(m.endswith('-cuda') for m in modes)` = `True`）⇒ 是**被排除的 CUDA 用例的入口** |
| **带外** `20` = `#include "server-vbr-artifact-store.h"` | **1** | 目标头文件存在（§4）⇒ 编译期冗余 |

⇒ 307 行在既定排除带内，38 行带外**同为 0 设备无关工作量**（此前未逐行归因，本次补齐）。

### 6.4 `test-recurrent-state-rollback.cpp`

port 的副本对 `vbr_controller_active|vbr_dynamic` **0 命中** ⇒ fork 的 VBR 臂**确实不在 port 里**（不是「同名异量」造成的假象）。
但那些臂**以「控制器已武装」为前置**——fork 原文的失败信息为
`"VBR sharing gate requires an armed ordinary hybrid"`；而「已武装」要求活的 VMM pool
⇒ **无设备时该前置恒不成立** ⇒ 照搬只会得到**会 PASS 而不断言**的测试（本项目明令禁止，比缺口更糟）
⇒ 记录为**设备 / VMM 门控**，不是可移植的断言。

**2026-10-12 更新（`1c8ff2e31`）：上述「不移植」的裁定被用户改判为「落地 + 交给设备取证」，理由随之修正。**

三条臂已按 fork 原文进入 `tests/test-recurrent-state-rollback.cpp`（opt-in：`--media-swa-prefix-only` /
`--live-swa-prefix-only` / `--attn-prefix-only`，仍然**不进 ctest**）。原文担心的「会 PASS 而不断言的测试」
在本 port 的实现里**不成立**——每条臂前提不满足时都 **fail-closed 且具名**：

- `share_media_swa_prefix`：`n_swa < 256` 直接 `return false`（→ 退出码 1）；
- `share_live_prefix_swa`：`n_swa == 0` 直接 `return false`；live 分支再查 `get_swa()->vbr_controller_active()`
  与 `tier_epoch(_swa) != 0`，不满足即打印 `"Live SWA gate needs actual VBR retiering"` 并 `return false`；
- `share_attn_prefix_vbr`：`!hybrid->get_mem_attn()->vbr_controller_active()` 时打印
  `"VBR sharing gate requires an armed ordinary hybrid"` 后 `return false`。

⇒ **缺口性质变化：从「未移植」变为「设备侧未取证」。代码残差 0；本机（无 GPU）只有编译证据，
数值/行为证据仍必须回 V100。** fixture：n_swa≥256 → gemma-4 E2B（两条 SWA 臂）；`-c ≥ 2048` + 显式 `--vbr-vram`（VBR 臂）。

**2026-10-12 后续（V100 实测）：三个臂的终态。** 完整标定过程与判据见 `docs/vbr.md`
「Deterministic tier schedules」；此处只留结论，不再保留中间几次改判。

- ✅ `--attn-prefix-only --vbr-vram 25M` → **6/6 全过**（`tier_epoch=0->12`，`source_first × arm` 六格）
  ⇒ `test_share_attn_prefix_vbr` 的 VBR 臂**已在设备上验证**。
- ✅ `--live-swa-prefix-only`（`VBR_FREEZE=1 VBR_BUDGET_MIB=25`）→ **6/6 PASS** + 窗口回收拒绝
  ⇒ 该臂的 VBR 分支已设备验证（它只要求 `tier_epoch_swa != 0`，**不要求递增**）。
- ✅ 两条 SWA 臂的 historical（非 VBR）分支：`-ctk f16 -ctv f16` **通过** ⇒ 行集构造 /
  `try_share_attn_prefix_rows` / companion 恢复 / 精确重放 / 回收窗口拒绝**均已设备验证**。
- ⛔ `--media-swa-prefix-only` 的 **VBR 分支：任何预算都不可满足**。根因不在预算，在
  `vbr_vmm_projected_bytes` 的**逐 extent 页粒度**：SWA 池每 extent 逻辑跨度只有 1 MiB、**小于一个
  2048 KiB 页** ⇒ 其 projection 对**任何 tier、任何 watermark** 都是 `24 extent × 2 MiB = 48 MiB`
  （= 其 f16 数据 24 MiB 的 2 倍；日志并列打印 `48.00 MiB VA reserved` 与 `size = 24.00 MiB`）
  ⇒ `vbr_over_budget` 对它只能**恒真或恒假** ⇒ 梯子在首边界走光（`#1..#168`）而收益为零。
  base 池每 extent 8 MiB（> 一页）则是 **12 → 24 MiB 的台阶**，台阶点 wm=2049 正落在断言窗口内
  （1278→2302）⇒ **卡住的一直只有 SWA**。该断言要求两个 child 都在窗口内新增一步 ⇒
  这是**一类模型**（SWA 跨度 < VMM 页）的共性，不是本夹具特例；该臂的实质内容由上面的 historical 分支覆盖。
- 相关机制（供复核）：降级门 `vbr_vmm_active() && vbr_budget_bytes_ > 0`（`llama-kv-cache.cpp:3708`）；
  `VBR_BUDGET_MIB` **覆盖** `--vbr-vram`；`vbr_tier_epoch_++` 三处自增中只有 `vbr_degrade_next()`
  （`:9669`，配套日志 `:9659`）会打 `VBR degrade #`，`vbr_promote_next()`（`:9340`）与 `vbr_full_reset()`
  （`:9088`）**静默** ⇒ **「没有那行日志」≠「没有 retier」**。可靠读数 = `-v` 下的 per-pool trace
  （`:3845`，每边界每池一行）；**没有这一行 = 该缓存已 settled、梯子已花光**。
- 观察（**未追、与本次移植无关**）：retier 跑的 teardown 会打
  `CUDA0 compute buffer size of 495.0000 MiB, does not match expectation of 489.0000 MiB`
  （`llama-context.cpp:845`，每次 arm 一次）。本次全部提交只动 `tests/` 与 `docs/`，
  `src/` 相对 `415b14280` 未改 ⇒ 该告警**先于本移植存在**。

移植时的**已声明偏差（均为消息/调用形式，功能等价）**：`make_ctx` → `make_ctx_sharing`（本 port 的 fill=0 / 不投毒约定）；
`test_share_attn_prefix_vbr` 里 fork 的公开包装 `llama_memory_try_seq_cp(mem, …)` → 内部虚方法 `mem->try_seq_cp(…)`，
即 §R5 与 9l-d4b7r 既有的 deviation 1（包装体只是 null 检查 + 调用）⇒ **`include/llama.h` 保持逐字节不变、无 170-TU 重建**；第三处 = media gate 的失败消息补打 `(base,swa)` 的 tier-epoch 对（fork 的 live-SWA gate 本来就在打这个 pair，此处只是对齐）。
先试过补公开 C 包装，被 `test_9l_d4b7o/p/q/r`（"include/llama.h is untouched"）判红 ⇒ 按既有约定改走内部虚方法，
**不是**放宽那四条守卫。

### 6.5 其余

`test-cuda-rdna2-vmm-policy.cpp`（7 行含 vbr）DEVICE ｜ `test-mtp-vocab-trim.cpp`（3 行）分类器裁定 **RULED-OUT**（非 VBR）。

⇒ **`test-llama-archs.cpp` 设备无关 VBR 残差 = 0；全测试面设备无关残差 = 0。**

---

## 7. ⛔ 残差清单（每条都是可执行归因，不是散文）

| # | 残差 | 量 | 归类 | 可执行依据 |
|---|---|---|---|---|
| **R1** | `test_qwen4_vbr_cuda` 等 95 行 | 95 行 | **设备** | 宿主全在 `supports_vbr_vmm` / CUDA 门后（§6.1） |
| **R2** | artifact-adopt 的 GPU 半边 `4042-5872` | 265 行 | **设备** | 带状归因（§6.3） |
| **R3** | artifact-adopt 驱动尾 `--vbr-*-cuda` 五模式 | 38 行 | **设备** | 五模式全 `-cuda` 结尾（§6.3） |
| **R4** | `test-recurrent-state-rollback` 的 VBR 臂 —— **已移植 + 已设备验证**（`1c8ff2e31`，2026-10-12） | 27 行 → **0** | **已闭环**（1/3 臂未取证，成因已定位，见 §6.4） | `share_attn_prefix_vbr` **6/6 PASS**（`--vbr-vram 25M`）；`share_live_prefix_swa` **6/6 PASS**（`VBR_FREEZE=1 VBR_BUDGET_MIB=25`，只要 `tier_epoch_swa != 0`）；`share_media_swa_prefix`：historical 分支 **PASS**，VBR 分支 **定性不可达**（根因 = 页粒度：SWA 每 extent 逻辑 1 MiB < 2048 KiB 页 ⇒ projection 恒 48 MiB，非「不降级」即「首边界花光」；base 是 12→24 MiB 台阶、台阶点 wm2049 落在窗口内，故它本可越线；**一类** SWA 池跨度小于页的模型都不可满足） |
| **R5** | **QSA 生产代码**：`qsa_selection_safe`、`set_input_qsa` 重写的 direct/fallback 算法、瞬态 `seq_rm_*` / `try_seq_cp`、`seq_add_raw_mrope` | 未量 | **非 VBR** | `qsa_selection_safe` 在 port **全树 0 命中**；fork 的 `src/models/qwen4exp.cpp` 调用 **2 次** |
| **R6** | `load_impl<Observed>`、DFlash、Bonsai、`.safetensors` / EXL3 | 未量 | **既定排除** | 见 `MERGE_BUUN_PLAN.md`「已定取舍」（用户裁定） |
| **R6b** | `test-moe-cache*`（MoE 专家缓存） | 已量（§10.92） | **2026-10-15 改判：纳入移植** | 用户改判；它与 VBR 零耦合（对 `vbr` 命中 0），是独立切片线，**不影响本审计的 VBR 残差结论** |

**R1–R3 = 本机不可达（无 N 卡）；R4 已落地并在 V100 上验证（`1c8ff2e31`，2026-10-12）；R5–R6 不是 VBR。**

---

## 8. 什么会推翻本终审

1. 任何一把尺子的自检报 `unbalanced > 0` 或 `known classes found: False` ⇒ **仪器坏 ⇒ 其裁定无效**；
2. `git -C llama.cpp show 4f0be29f9 -- src/llama-memory-hybrid-idx.h` 不再包含 `llama_memory_vbr_params` ⇒ 接线被回退；
3. `classify_test_gap.py` 的候选数 > 18，或**出现新的含 vbr 的 fork 独有测试文件**；
4. **有了一块 VBR 能力设备** ⇒ R1–R4 全部变为可达，本终审的「设备门控」理由随之失效，必须逐条落地并重新取证。

---

## 9. 诚实开口（只记实测事实，未断言机制）

以下为探针运行中观察到的现象，**机制未查清，不硬凑**：

- `pinned at q8_0` 的 WARN 在**修复跑与反向控制跑里都打印**（它在 `static bool warned` 之后，每进程一次）；
- 缓存尺寸日志行把 K 标成 `(f16)`，而 `kv_bpv()` 的求和是 **8.5**；
- `VBR_VMM gate` 调试行**无论 `vbr_dynamic` 取什么都打印 `dynamic=0`**（被公开默认值否决）。

---

## 10. 复现步骤（从零重跑本终审）

```bash
cd /d/llama-build
python -u guard_sweep.py                      # 期望 136 guards / 0 failing
python -u attribute_llama_archs_vbr.py        # 期望 95 行，分组 15+4+56+19+1
python -u classify_test_gap.py                # 期望候选 18、同名带 vbr 3 条
python -u vbr_server_triage.py                # 期望 16 行，14 compile-time / 2 host-absent
python -u scan_vbr_callsites.py               # 期望 delta 0、verdict 0、自检全过
python -u mut_scan_vbr_callsites.py           # 期望 baseline green / mutA red / mutB green / mutC red
cmd //c run_ctest_proj.bat                    # 看日志的 CTEST_RC（.bat 的包装退出码不可信）
# 引擎面标识符覆盖：见 §3 的脚本（1061 / 1063 / 缺席 1 = vbr_alpha_v_decode）
```
