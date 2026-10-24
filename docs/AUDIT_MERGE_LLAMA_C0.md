# AUDIT_MERGE_LLAMA_C0 — C0 演练：合并会静默删掉什么

> **2026-10-24**｜方法：冻结 master（tag `pre-llama-merge-2026-10-24`）→ 新分支 `merge-llama` → 独立 worktree
> `tmp/wt-c0` → `git merge --no-ff llama`（`llama` = `6b9ba4f2e`）→ **只读**三方树比对 → 记录。
> 本文件是 C0 的唯一交付物；演练**未提交、未解任何冲突**，结束后分支 `reset --hard` 回 master，
> 真正的解冲突从 C1 开始。配套计划：`PLAN_MERGE_LLAMA_2026-10-22.md`（v2）。

---

## 0 一句话结论

**master 自有功能面在文本上几乎无损**：关键标识 1:1、master 新增文件 0 丢失、0 个文件被整体回退到 base、
VBR / kv-stream / PQ2_0 / PTQ1_0 / `fork 9l-` 标记全部零损失。

但有 **4 处真实静默删除或结构断裂**（§2），成因统一：**上游重写了同一区域，git 无冲突地取了上游版本**。
这四处在合并后**必然编译失败或丢用例**，且都不会在 `git status` 的冲突列表里出现任何提示——

> ⭐ **C0 的核心教训：冲突列表只覆盖「双方都改了同一段」。上游单方面重写、master 单方面落点被覆盖的那一类，
> 冲突列表里一个记号都没有。C0 的机械比对是唯一能发现它的手段。**

---

## 1 基线

| 项 | 值 |
|---|---|
| `master`（冻结） | `ce579daec` 2026-10-24 `cuda argsort: give CUB a distinct keys-out buffer…` |
| `llama` | `6b9ba4f2e` 2026-10-24 `ci : fix data race between runners sharing the model cache (#31046)`，与 `refs/remotes/llama/master` 0/0 |
| merge-base | `4c9233c03` 2026-09-15（未变） |
| 上游独有 / master 独有 | **880** / **527** 提交 |
| 改动文件：上游 / master / 重叠 | 1479 / 946 / **126** |
| 冲突 | **50 文件**（49 `UU` + 1 `UD`），**147 个冲突段**（merge-tree 口径 155） |
| 合并删除的文件 | **35**（全部是上游侧删除，见 §3 A3） |

---

## 2 真丢清单（4 项，必须处置）

| # | 位置 | 现象 | 后果 | 处置段 |
|---|---|---|---|---|
| **S1** | `ggml/src/ggml-cuda/mmq.cuh` | master 的 `ggml_cuda_mmq_get_J_max()`（master `mmq.cuh:383`）**定义消失**，合并后 `ggml_cuda/mmq.cuh` 里 0 命中，但 `ggml-cuda.cu:3411`、`ggml-cuda.cu:3472`、`mmq.cu:245` 仍在调用 | **编译/链接必挂**；这是 V100 `cap=200` 动态 J 尺寸的唯一入口，丢了等于丢 R4-d 的性能开关 | C2 |
| **S2** | `tests/test-recurrent-state-rollback.cpp` + `tests/CMakeLists.txt` | 文件是 **`UD`**（上游 `a17e94d22` 删、master 改了 +898/−114）；`tests/CMakeLists.txt` 里 `test-recurrent-state-rollback` 的 **12 处引用 → 0 处** | **6 个用例整体消失**，其中 3 个（`test_share_media_swa_prefix` / `test_share_live_prefix_swa` / `test_share_attn_prefix_vbr`）依赖 master-only 的 SWA 池与 VBR cells，**无替代** | C4 |
| **S3** | `ggml/src/ggml-cuda/fattn-mma-f16.cuh:9` | `#include "fattn-swizzle.cuh"` 指向上游 `2ca15f540` 已删除的文件（R1 复现，行号 8→9 因为 master 前面加了一行） | **编译必挂** | C3 |
| **S4** | `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | 上游给 `ggml_cuda_mmq_get_{nthreads,I,sram_stride}()` 加了第 4 个参数 `GGML_PREC_Q8`；合并结果 **42 处四参 + 4 处三参**——这 4 处正是 master 为 `PQ2_0/PTQ1_0` 新加的实例化 | **编译必挂**（`no matching function`），且位置恰在 PQ2_0/PTQ1_0 的 MMQ 路径上 | C2 |

> ⚠️ S1 与 S4 是同一件事的两半：上游把 MMQ 的 tile 参数面改成「带 prec 的四参」，master 在同一批文件里加了
> 两套私有类型。**C2 必须一次改到底**：`get_J_max` 补回并接受 prec 参数 + 4 处调用点补第四参，
> 否则「编译过了但 PQ2_0 走了 Q8 精度」比编译失败更糟。

---

## 3 假警报（3 项，已排除，不必返工）

### A1 `ggml/src/ggml-cuda/argsort.cu`：整文件取了上游 = **正确去重**

- 现象：合并后该文件 blob **逐字节等于上游**，master 的 `ce579daec`（本次冻结前的最后一个提交）不见了。
- 原因：master 那条提交的正文自己写着 *"Ported from upstream `b23701f77` (#28389), which this tree was missing
  together with three other argsort/top-k commits"* ⇒ **master 本来就是上游补丁的手工 port**。
- 判定：**这正是本次合并要的"去重"**。上游版多带 3 个提交（top-k 选择、宽块 bitonic、CCCL 版本守卫），
  是超集。**不回贴 master 版本**，但 C5 收尾要用 `test-backend-ops -o ARGSORT/TOP_K` 在 GTX 950（sm_52）复验一次
  ——master 那条提交的动机正是 Maxwell 上的非法访存。

### A2「56 个 master 新增文件缺失」= 假警报

- 机械统计 `git log --diff-filter=A base..master` 得到 792 个 master 新增文件，其中 56 个在合并树里不存在。
- 复查：这 56 个**在 master 自己的树里也已经不存在**（master 后来改名/删除，例如
  `tests/test-turbo-mma-fa.cpp` → `db342edab test-mma-fa: rename it off "turbo"`；
  `fattn-vec-instance-turbo*` 一批模板实例被 master 自己清理）。
- ⭐ **判据修正**：统计"新增文件是否存活"必须先用 `master` 的 `ls-tree` 收敛集合，否则把 master 自己的改名算成合并丢失。
- 真实数字：**master 新增且最终仍在 master 树里的 736 个文件，合并后缺失 0 个**。

### A3 上游删除的 35 个文件：master 侧引用为 0（除 S3）

- 35 个被合并删除的路径，**每一个的 `master_commits(base..master) = 0`**（master 从未碰过）⇒ 纯上游删除，可接受。
- 其中 3 个是 rename 的源（`sheet-header/footer.svelte`、`models/utils.ts`）。
- 逐个 grep 上游删掉的 12 个源/头文件：`iqp.{h,cpp}`、`hex-dma.h`、`hvx-mm-kernels-flat.h`、`unary_softplus.cpp`、
  `fattn-buffers.{cpp,hpp}`、`test-json-serialization.cpp`、`test-save-load-state.cpp`、
  `test-state-restore-fragmented.cpp` — **引用全部为 0**（R2 自愈确认，`ggml/src` 内 `\biqp` 零命中）。
- **唯一例外 = S3（`fattn-swizzle.cuh`）**。

---

## 4 结构性变化（不丢功能，但落点变了，必须按新结构接线）

| # | 变化 | 影响 |
|---|---|---|
| **V1** | 上游把 `ggml-vulkan.cpp` **拆成 3 个文件**：`ggml-vulkan.cpp` / `ggml-vulkan-buffers.cpp` / `ggml-vulkan-debug.cpp`（另加 `ggml-vulkan-{common,types,push-constants}.h`） | master 在该文件上有 **+1479/−5699** 的改动面（Vulkan Hadamard + PQ2_0/PTQ1_0）；`ggml-vulkan.cpp` 是 `UU`，拆分本身不丢代码，但**解冲突时要按新文件边界重新落点**。Vulkan 侧 `PQ2_0` 46 处 / `PTQ1_0` 88 处合并后**计数不变**，说明着色器与 `vulkan-shaders-gen.cpp` 的引用完好 |
| **V2** | 上游 MMQ 全体函数追加 `prec_src1` / `GGML_PREC_Q8` 维度 | 见 S4；另 `ggml_cuda_op_mul_mat` 的 MMQ 分支签名变化，master 的 `get_J_max` 调用点要跟着改 |
| **V3** | 上游测试重组为 `tests/llama-context/`（`cases-kv.cpp` / `cases-rs.cpp` / `test-case.*` / `test-runner.*`），注册 `test-llama-context` | master 专有测试目标**除 rollback 外全部存活**：`test-vbr` 61 处、`test-kv-stream` 11、`test-moe` 9、`test-turbo` 8（与 master 一一对齐）。`test-llama-context` 在合并结果里出现 5 次，与 master 目标并存无冲突 |

---

## 5 重复功能现状：合并后**两份并存**（去重是 C2–C4 的主线）

| 标识 | master | 合并结果 | 解读 |
|---|---|---|---|
| `prism.hadamard` | 58 / 6 文件 | **113 / 11 文件** | master 的裸字符串读取 + 上游的 `LLM_KV` 枚举**同时存在** ⇒ 同一个键两套读法，必须二选一 |
| `hadamard_rotations` | 21 / 5 | 28 / 5 | master 侧结构存活，上游的 `hdmd.rot` 并存 |
| `llama_verify_hadamard_graph` | 5 / 3 | 7 / 3 | 上游版多一个 `moe_cache` 参数，两个同名重载并存 |
| `fork 9l-` | 593 / 58 | **593 / 58** | 零损失 |
| `GGML_TYPE_PQ2_0` / `PTQ1_0` | 117/27、113/24 | 117/27、113/24 | 零损失 |
| `vbr_pools_` / `kv_stream` | 154/8、1371/50 | 154/8、1371/50 | 零损失（按文件逐一比对 `vbr` 计数：只有 `tools/ui/package-lock.json` 差 1，无关） |
| `MMVQ_PARAMETERS_VOLTA` | 11 / 5 | 11 / 5 | 零损失 |

⭐ **去重取舍（用户已定规矩）**：**同域重复功能优先保留 master 的**，只有上游**性能/覆盖更优**时才取上游。
按此规矩，§5 里 Hadamard 一条的默认答案是"**保留 master 的 `hadamard_rotations`/`hadamard_memo` 实现，把上游的
`hdmd`/`LLM_KV` 侧当作纯元数据层接入**"，而不是 plan v2 §R4-a 原先写的"整条以上游为准"。

---

## 6 结构性问题处置决定（写进 C1–C5 的必查项）

| 项 | 落在哪段 | 具体动作 |
|---|---|---|
| S4 四参/三参混用 | **C2** | 4 处三参补 `GGML_PREC_Q8`（或按该 op 的 prec 语义传），**不许**用默认值蒙过去；改完对 `mmq-load-tiles.cuh` 逐个实例 TU 单编 |
| S1 `get_J_max` 消失 | **C2** | 把 master `mmq.cuh:383` 的函数补回，签名对齐上游四参风格；确认 `ggml-cuda.cu` 两处调用（cap=200 路径）与 `mmq.cu` 一处仍在 |
| S3 悬空 include | **C3** | 删 `fattn-mma-f16.cuh:9` 的 include；**但** master 在该文件里用了 `load_ldmatrix`（26 处）/`fattn-mma-turbo.cuh` 2 处 /`mmf.cuh` 5 处，上游把它们搬进 `mma.cuh`（7 处 / 1 处）且**签名可能不同** ⇒ 必须逐签名比对，不能只删 include |
| S2 rollback 用例 | **C4** | `git checkout master -- tests/test-recurrent-state-rollback.cpp`，并在 `tests/CMakeLists.txt` 里与 `test-llama-context` 并存；6 个用例逐个数一遍 |
| V1 Vulkan 拆分 | **C4**（Hadamard 落点） | 按拆分后的文件边界落 master 的 Vulkan 改动；`grep -c PQ2_0/PTQ1_0 ggml/src/ggml-vulkan` 必须回到 46/88 |
| V3 测试重组 | **C4** | `grep -c` 四个 master 目标（61/11/9/8）与合并前对齐 |

---

## 7 C1–C5 切点（按 880 提交重算）

| 段 | 段末提交 | 日期 | 累积（文件/段） | 本段增量 |
|---|---|---|---|---|
| C1 | `bd4f514db` | 09-24 | 9 / 35 | 35 |
| C2 | `552f18f91` | 10-01 | 27 / 85 | **50**（最重） |
| C3 | `03aa006ac` | 10-08 | 35 / 114 | 29 |
| C4 | `6afa9a313` | 10-16 | 46 / 142 | 28 |
| C5 | `6b9ba4f2e` | 10-24 | 50 / 155 | 13（含今日 29 个上游提交） |

---

## 8 复现命令

```bash
cd llama.cpp
ls .git/shallow                       # 必须不存在
git merge-base master llama           # 必须是 4c9233c03…
git worktree add -b merge-llama ../tmp/wt-c0 master
cd ../tmp/wt-c0 && git merge --no-ff --no-commit llama
# 机械比对（脚本见 tmp/c0_analyze.py、tmp/c0_lost_lines.py）
git diff --diff-filter=D --name-only master        # 35 个删除
git diff --name-only --diff-filter=U | wc -l       # 50
```

---

## 9 C1 执行记录（段末 `bd4f514db`，9 文件 / 31 段）

### 9.1 逐文件处置

| 文件 | 段 | 处置 |
|---|---|---|
| `common/fit.cpp` | 2 | 全取 ours：保留 `moe_tensors` / `vbr_costs` / `plan_hint` 形参；`n_ctx_max` 维持 master 的 `hp_nct * n_streams`（非统一 KV 每流各占一份训练上下文），不采上游的 `* n_seq_max` |
| `ggml-cuda/allreduce.cu` | 2 | 外层守卫改为上游 `#if !defined(GGML_USE_MUSA)`（上游给整文件加了 HIP 支持，是超集），master 的 one-shot P2P AllReduce 主体与 MUSA 桩全部保留 |
| `ggml-vulkan/ggml-vulkan.cpp` | 5 | 五处均为「两侧同锚点追加注册项」⇒ 并集：`PQ2_0/PTQ1_0`(master) + `IQ4_XS`(上游) |
| `ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | 3 | 全取 ours，但把 `has_q8_1_mmq` 的排除面从 `!iq*` 收窄为 `!= iq1_s && != iq1_m`（见 9.3-②） |
| `ggml-cuda/fattn-mma-f16.cuh` | 7 | FA 架构决策，见 9.2 |
| `ggml-cuda/fattn.cu` | 1 | 并集：master 的 `!requires_ncols_ge_32`（Volta 最小 32 列）+ 上游的 wide-sparse 优选 + 新签名 `shall_use_sparse(cc, dst, 1, ncols2)` |
| `src/llama-context.cpp` | 3 | 并集：master 的 `llama_model_has_cacheable_moe_weights` / kv-stream 反馈块 / Hadamard 校验前移，与上游的 `llama_graph_n_input_tensors` / `get_gf_res_prev()` / `n_input_tensors` 并存 |
| `tests/test-llama-archs.cpp` | 2 | master 的 Qwen4 VBR 契约块 + 上游的 `stdev` 形参与 `arch_filter` |
| `tests/test-recurrent-state-rollback.cpp` | 6 | 全取 ours（推迟到 C4，见 9.3-⑤） |

### 9.2 FA 那对冲突：判定为「同一功能的上游更高版本」

`fattn-swizzle.cuh` **在 C1 段末（09-24）就已被删**（v2 计划以为在 C3）。master 的 `swz_K/swz_V` 源自上游 `e4b9af007`(#25635)，上游 `1884824fd`(#28536) 是同一 PR 的官方 follow-up，把 swizzle 搬进了 `mma.cuh`。⇒ 取上游为基底：

- 换挂点：`ggml_cuda_fattn_smem_swizzle::{tile_stride,bytes_rc}` → `ggml_cuda_fattn_mma_get_swizzled/get_stride_tile` 与 `swizzle_bytes<swz,half2>`（XOR 映射逐位相同：`off ^ ((i & 7) << 4)`）。
- `swz_K`/`swz_V` 合并为单一 `swz`：上游把「是否 padding」提升为整核决策（要求 K/V 的 nbatch 都 32 对齐），比 master 的逐 tile 判定更粗但自洽（写入与读取用同一 stride）。
- master 独有且必须保留的两块：`flash_attn_ext_turbo_load_tile`（turbo/VBR KV 类型的前提，21 处实例化）与 Volta compact 特化（`GGML_CUDA_VOLTA_FA_COMPACT`，`DKQ==256 && DV==256 && ncols1==32 && ncols2==2`）。
- sparse 索引公式统一取上游 `(sequence % ne33)*iter_j + jt`（master/base 是 `ne31 + jt*ncols1`）：上游 wide-sparse 改了 `sparse_indices` 的布局，两阶段必须同口径。
- 两处合并副产物必须清掉：误落进 Volta compact kernel 的 `indices`/`use_sparse` 块（该 kernel 无 sparse 变体），以及 launcher 里已由 `ggml_cuda_fattn_mma_f16_shared_bytes()` 取代的 `nbytes_shared_*` 计算块。

### 9.3 覆盖上游的偏离（每条都要理由）

| # | 位置 | 偏离 | 理由 |
|---|---|---|---|
| ① | `fit.cpp` | `n_ctx_max = hp_nct * n_streams` 而非 `* n_seq_max` | 统一 KV 时 `n_streams==1`；只有非统一 KV 才需要「每流各占一份训练上下文」。上游式在统一 KV 下会把上限放大 `n_seq_max` 倍 |
| ② | `vulkan-shaders-gen.cpp` | `has_q8_1_mmq` 只排除 `iq1_s/iq1_m` | host 侧 `ggml_vk_matmul_int_shmem_support` 已含 `IQ3_S`/`IQ4_XS`（上游新增），用 master 的 `!iq*` 会出现「表里承诺、shader 没生成」 |
| ③ | `allreduce.cu` | 外层改上游 `!MUSA` | 上游把 HIP 纳入同一实现（仅 sleep 内建分叉）；master 的 P2P 只用 hipify 可映射的 `cudaEvent*`/`cudaMemcpyAsync`/`cudaMalloc` |
| ④ | `test-llama-archs.cpp` | `arch_matches(arch_filter, LLM_ARCH_QWEN4EXP)` | 上游把 `--arch` 从 enum 改成正则过滤；空 filter 恒匹配 ⇒ 默认「全跑」语义不变 |
| ⑤ | `test-recurrent-state-rollback.cpp` | 全取 ours | 上游该文件仅 `161755f29`(#29133) 的 88/35 改动；`a17e94d22`(#30023) 在 **C4** 会把它并入 `tests/llama-context/cases-rs.cpp`，届时随 R3 一次性对齐，避免同一文件改两遍 |
| ⑥ | `server-cache-plan-preflight` | 暴露门改收 `hostnames` 列表 | 上游把 `--host` 变成逗号列表；只取首项会让 `127.0.0.1,0.0.0.0` 蒙过 trusted-local 门 |

### 9.4 验证

- 配置：Ninja + MSVC 14.44 + CUDA 12.8，`CMAKE_CUDA_ARCHITECTURES=70`，`GGML_VULKAN=OFF`（本机无 Vulkan SDK，Vulkan 侧改动仅静态审查）。
- 构建：`cmake --build` 全量 **RC=0，0 FAILED**，产出 143 个 exe + `ggml-cuda.dll`（含 `llama-cli` / `llama-server` / `test-llama-archs` / `test-recurrent-state-rollback`）。
- ctest：**102 / 110 通过**。8 红逐条归因，全部为环境：`test-jinja-py`（缺 python jinja2）、`test-download-model` / `test-eval-callback-download-model`（需联网）、`test-thread-safety` / `test-state-restore-fragmented` / `test-eval-callback`（依赖前述，Not Run）、`test-backend-meta-replay` / `test-opt`（`ggml-cuda.cu:113: CUDA error` —— 开发机无 GPU）。
- 零丢失复核（C1 树 vs master）：`fork 9l-` 550/550、`PTQ1_0` 305/305、`PQ2_0` 233/233、`GGML_TYPE_TURBO` 857/857。
- ⏳ 待 V100：`Ternary-Bonsai-2-2B-{PQ2_0,PTQ1_0}.gguf` 加载出 token（硬验收门）、FA turbo 分支与 Volta compact 的数值/性能、双卡 allreduce。

### 9.5 环境复现（本机）

```bash
source tmp/msvc_env.sh          # 手工 INCLUDE/LIB，PATH 必须用 /d/... 风格
cmake -S tmp/wt-c0 -B tmp/build-c1 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DGGML_VULKAN=OFF
cmake --build tmp/build-c1 -- -k 0
```
⚠️ 本机 `cl.exe` 传 `/nologo` 会被 MSYS 改写路径 ⇒ 一律用 `-nologo` 形式。
