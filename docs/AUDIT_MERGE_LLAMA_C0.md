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

## 10 C2 执行记录（段末 `552f18f91`，23 文件 / 56 段，提交 `56565e159`）

### 10.1 逐文件处置（按「重复功能保留 master」规矩）
| 文件 | 处置 | 理由 |
|---|---|---|
| `mmq.cuh` | ours | master Volta `== GGML_CUDA_CC_VOLTA` Q6_K dp4a 特例保留，上游只加 `!defined(GGML_USE_MUSA)` 守卫 |
| `norm.cu`/`norm.cuh` | ours | master `post_scale` / `rms_norm_scale_f32_cuda`；上游 scale_out 是重复功能 |
| `mmvq.cu` | ours | master `MMVQ_PARAMETERS_VOLTA` 表 |
| `fwht.cu`/`fwht.cuh` | ours + 上游谓词 | master 富版 `fwht_cuda<N,Tout,has_signs>`+block butterfly（64–4096）；补上游 `ggml_cuda_op_mul_mat_use_fwht` 谓词 + **TSrc 模板参**（F16 输入），否则上游 impl<T> 与 master fwht_launch 混拼 `const half*`→`const float*` 直接崩 |
| `ggml-cuda.cu` | 并集 | `GGML_OP_TURBO_WHT`+`GGML_OP_SCALE` 都留；删重复 `_fused` RMS_NORM+SCALE dispatch |
| `generate_cu_files.py` | 并集 | PTQ1_0 CUDA-only 实例 + MXFP4/NVFP4 W4A4 |
| `ggml-cpu.c` | ours | `ggml_cpu_vec_dot_type`（PQ2_0→Q8_0）是 Bonsai 验收门硬依赖 |
| `llama-kv-cells.h` | 并集 | `seq_pos_get`（上游）+ master `seq_pos_tok_le` 注释 |
| `include/llama.h` | 并集 | master `llama_decode_phase`+`llama_set_decode_phase` + 上游 `llama_batch_ext` 块 |
| `llama-graph.{h,cpp}` | 并集 | hadamard_rotations/inverses + prec_policy；build_attn 保留 TurboQuant Q 预旋转 |
| `llama-context.cpp` | ours 骨架 + 上游签名 | `vbr_decode_txn` 保留；`decode()` 收 `llama_batch_ext` |
| `common/common.{h,cpp}` | 并集 + 恢复 | fs_* 返回 path；`common_moe_cache_profile_file` 转 utf8；**恢复 `common_batch_add`/`common_batch_clear`** |
| `common/fit.cpp` | ours | moe_tensors / fit_selected / vbr_costs |
| `common/speculative.cpp` | ours 逻辑 + 上游 API | MTP preflight 生命周期适配 extended batch |
| `conversion/base.py` | 并集 | hadamard_folded_names/add_hadamard_metadata + `_tag_prec_a4` |
| `server-context.cpp` | ours + 上游 API | decode() hard-seal 三特性保留；`llama_batch_view`→`llama_process`/`common_batch`；swap-aware mmproj fit |
| `test-llama-archs.cpp` | 并集 | `test_qwen4_vbr_*`×3 + `check_causal_attn_toggle` |
| `test-recurrent-state-rollback.cpp` | ours（延 C4） | `a17e94d22`(#30023) 在 C4 并入 cases-rs.cpp |
| `llama-bench.cpp` | 并集 | kv-stream/n-parallel + repack；quiet 日志回调取 master |
| `docs/build.md` | 并集 | FA_MMA + MMQ_PREC 两段都留 |

### 10.2 ⭐ 新发现：上游删除 `common_batch_add`/`common_batch_clear`（C0 漏抓的静默删除）
上游把 `llama_batch` 迁移到 `common_batch` 类时删掉了这两个 helper，但 master 的
`test-recurrent-state-rollback` / `test-vbr-representation-epoch` / `test-vbr-swa-window` 以及
`server-context.cpp` 的「尾部 frontier 补解码」仍走 `llama_batch`+compat `llama_decode`，共 14 处调用点
静默失效（构建 C3861）。修法 = 在 common.h/cpp 恢复这两个 20 行 helper，不重写调用点（`llama_batch` 结构仍在）。

### 10.3 验证
- `cmake --build -- -k 0`：**RC=0，0 FAILED**（sm_70 全量）。
- ctest：**103/111**；8 红与 C1 完全同集（无 GPU `0xc0000409` / 无网 / 无 jinja2），无新回归。
- 零损失标记 vs master：`fork 9l-` 593/593、PQ2_0 292→301、PTQ1_0 372→380、TURBO_WHT 59/59、vbr_pools_ 154→155、kv_stream 1371→1372（全 merged≥master）。

## 11 C3 执行记录（段末 `03aa006ac`，21 文件 / 53 段，提交 `c07179456`）

### 11.1 唯一需要「重挂」而非「取边」的文件：`fattn-mma-f16.cuh`
上游 C3 第三次重构了 `mma.cuh` 的 swizzle/ldmatrix API（`mma.cuh` 本身 fork 从未改过，故直接取上游）：
删 `ggml_cuda_fattn_mma_bank_aligned` / `_get_stride_tile`，`get_swizzled()` 改收 `ncols`，
`swizzle_bytes<swz,T>` → `swizzle<stride,T>`，`load_ldmatrix<swz>` → `load_ldmatrix_swizzled<stride>`。
fork 的 turbo loader 写于旧 API 上，`git checkout --ours` 后无法编译。
处置：**把上游的 API 迁移应用到 ours**（而非把 fork 特性重挂到上游文件上——上游 delta 仅 247 行，
fork delta 1113 行，方向取小的一侧）：
| 变更 | 说明 |
|---|---|
| 删 `bank_aligned`/`get_stride_tile` | stride 内联为 `swz ? nbatch_2 : nbatch_2 + 4` |
| 两个 loader 去掉 `bool swz` 模板参 | 写入端改走 `swizzle<stride_tile>`；`+4` 填充使非 swizzle 瓦片按构造落回恒等映射 |
| `load_ldmatrix{,_trans}<swz>` → `_swizzled<stride_tile>` | 含 Volta 分支（上游已删 `static_assert(!swz, "Volta has no ldmatrix")`） |
| `tile_V_i` 指针 → `tile_V_offset_i` 整数 | 配合上游 `load_ldmatrix_trans_swizzled` 的偏移式 |
| 新增 Volta `(320,256)` 两档 + `async_kv_preload` 实参 | 上游纯新增，全保留（后者只在 DGX Spark + scan_mask 生效） |

### 11.2 ⭐ 上游 C3 引入第二套 MoE 缓存（决策：保留休眠）
| | 上游 `src/llama-moe-cache.cpp`（550 行） | fork `ggml/src/ggml-cuda/moe-cache.cu`（4474 行） |
|---|---|---|
| 机制 | host slot map + 设备 bank，miss 在 scheduler copy callback 里上传 | CPU `MUL_MAT_ID` 拦截 `begin/plan/dispatch/collect` |
| 命中行算在哪 | GPU（用 bank + slot ids 替换专家张量） | GPU（MMV 直接派发） |
| 未命中行算在哪 | —（全部行都在 GPU，miss 每步重传） | **CPU**（`miss_mask = valid_mask & ~hit_mask`）⇒ 热专家永不重传 |
| 批量上限 | `max_batch = 32` | 无此硬门 |
| 融合 | 无 | `fused_begin`：up*GLU(+down) 单核 |
| 其它 | — | expert-parallel / cpu-overlap / heatmap profile |
结论：**上游不是超集**，且在「主机↔设备流量」这一关键轴上更差（每 miss 重传专家）。
两者都改写同一批 `MUL_MAT_ID` 节点，**绝不能同时启用** ⇒ 上游类保留编译（便于下次 rebase），
用 `use_upstream_moe_cache = false` 挡死实例化；`--moe-cache-mib`（上游旋钮）改喂 provider 预算。

### 11.3 其余处置
- `gated_delta_net.cu` / `fwht.cu` / `norm.cu`：ours（Volta prefill / per-block FWHT / post_scale）。
- `ggml-backend.cpp`：ours（MoE session 钩子），补上游 helper 拆分。
- `llama-cparams.h` / `llama-context.cpp` / `llama-graph.cpp`：并集（`vbr_*`+`hadamard_*` + 上游 `moe_cache_size`）。
- `llama-kv-cache.cpp`：上游 `n_rot_k/n_rot_v` + fork `attn_rot_k_whole_head`，`build_input_k_rot` 保留 `LLAMA_ATTN_ROT_K_NROT`。
- `qwen35.cpp`：上游 `crop_before_nextn/crop_after_nextn` 取代本地 `narrow_last`（真值表相同）。
- `test-quantize-fns.cpp`：ternary dot + 上游 degenerate-imatrix 并集。
- `tests/test-recurrent-state-rollback.cpp`：ours（延 C4）。

### 11.4 验证
- `cmake --build -- -k 0`：**RC=0，228/228 链接，0 错误**，162 处 fattn 重编（sm_70）。
- ctest：**103/111**，8 红与 C1/C2 同集（无 GPU `0xc0000409` / 无网 / 无 jinja2），无新回归。
- 零损失：`fork 9l-` 593→598、PQ2_0 292→303、PTQ1_0 372→382、TURBO_WHT 59→61、
  vbr_pools_ 154→156、kv_stream 1371→1373、MMVQ_PARAMETERS_VOLTA 11→13、hadamard_rotations 21→24。

## 12 C4 执行记录（段末 `6afa9a313`，21 文件，提交 `eb867e467`）

### 12.1 ⭐ hadamard 存储统一（fork 的并行 plumbing 退场）
fork 把折叠权重的变换挂在 `llama_model::hadamard_rotations/hadamard_inverses` 两个裸 map 上；
上游早已把同一对 map 连同元数据收进 `llama_hadamard hdmd`，并由 `load_tensors_hadamard()` 建张量。
C4 把该容器带进来 ⇒ 树里同时存在两套：fork 的 map 由 fork 的 hparams 读取段填充，**但已无人读**。
处置：`hdmd` 作唯一存储，删 fork 的 map 成员 / `llm_graph_params` 字段 / `build_hadamard_activation`
（改上游 `build_hadamard_input`）与验证调用形态；**保留 fork 独有的 `build_hadamard_inverse`**（接 `hdmd->inv`），
它的调用者是 qwen35 / qwen35moe / qwen3next 的自有 embedding 查表，上游从无对应。
图时契约检查仍跑，改用 `hdmd.rot/inv` + 上游的 MoE-cache forward map（穿过缓存的折叠专家也被验到）。

### 12.2 ⭐ MMQ API 迁移 + 被上游静默删除的 fork 符号
C4 给 `get_config/get_nthreads/get_I/get_sram_stride` 加 `prec_src1`，`mul_mat_q_case` 变双模板参（`<type, prec>`）。
- PQ2_0 / PTQ1_0 的 case 与两个 tile loader 补 `GGML_PREC_Q8`（这两种类型的配置表只有这一档）。
- **`ggml_cuda_mmq_get_J_max` 被上游整段删掉**（fork 的 FWHT→q8_1 生产者用它算尾部余量）⇒ 按新签名重加，
  语义不变。传 Q8 是精确的：`ggml_cuda_fwht_mmq_layout()` 拒绝 NVFP4/MXFP4，而只有这两种类型的配置随 prec 收窄。

### 12.3 ⭐ fork master 自带的重复字典键（顺带修掉）
`gguf-py/gguf/constants.py` 的 `MODEL_TENSORS` 里 `KIMI_K3` 与 `BAILINGMOE3` 各出现两次（字面量后者覆盖前者）。
两个 KIMI_K3 内容不同（`OUTPUT/ATTN/FFN_RES_NORM+RES_PROJ` 旧版 vs `*_RES_SCORE` 新版，后者才与 `src/models/kimi-k3.cpp` 对齐），
两个 BAILINGMOE3 也不同。处置：留生效的那份、删被遮蔽的，并删掉被 fork 副本遮蔽的上游 BAILINGMOE3 副本。
`MODEL_TENSORS` 157 项、零重复键。

### 12.4 其它
| 文件 | 处置 |
|---|---|
| `llama-model.cpp` | 删 fork 的内联 hadamard 建张量段（175 行），改调上游 `load_tensors_hadamard()`；`buf_ptrs` 与 `GGML_CUDA_PREFER_MODEL_WEIGHTS` 并集 |
| `server-context.cpp` | 槽文件 checkpoint appendix（C4 新增）改用 `common_shared_byte_buffer`：补两个 `ckpt_{read,write}_buf` 重载 + `data_spec` 成员；`prompt_save` 返回枚举 ⇒ `prompt_save_durable()` |
| `server-context.cpp` | `gated_delta_net` 的 `ggml_cuda_gdn_mma_launch` 早退：上游按架构/形状自门（Volta 恒 false）⇒ 与 fork 编译期分派并集，Volta 行为不变 |
| `llama-memory.cpp` / `llama-ext.h` / `llama-hparams.h` / `test-arg-parser.cpp` / `ggml-vulkan.cpp` | 并集 |
| `tests/test-recurrent-state-rollback.cpp` | **保留 ours 独立 target**：上游 `a17e94d22` 把它并进 `test-llama-context`，但 fork 的 VBR / SWA-prefix / 多序列重放用例上游没有等价物 |
| `common/chat.cpp` | 恢复被 C4 chat 重构丢弃的 `common_chat_msg_delimiters_parse`（fork 的 cache-plan 预检在用） |

### 12.5 ⚠️ 已知残留（合并后再清）
fork 的 `hadamard_weight_blocks / hadamard_inverse_blocks / hadamard_sign_data / hadamard_gdn_v_grouped /
hadamard_tied_output` 成员与 `llama-model.cpp` 里对应的 hparams 装载段仍在，**已被 `hdmd` 取代，实际是死代码**
（无读者）。删除属独立清理提交，避免在合并段里动 160 行验证逻辑。

### 12.6 验证
- `cmake --build -- -k 0`：**RC=0，228/228**，0 错误。
- ctest：**102–103/112**。稳定失败仍为 C1–C3 同一组环境前提（无 GPU `0xc0000409` / 无网 / 无 jinja2）；
  `test-arg-parser` 新增进集合但属**网络抖动**：它自己的出网探测通过、紧接着的 GET 失败，重跑直接 exit 0（无 SKIP 分支）。
- 零损失：`fork 9l-` 593→599、PQ2_0 292→304、PTQ1_0 372→383、TURBO_WHT 59→62、vbr_pools_ 154→157、
  kv_stream 1371→1374、MMVQ_PARAMETERS_VOLTA 11→14、prism.hadamard 58→105。
  `hadamard_rotations` 21→11 = 有意的 `hdmd` 归并（typedef 保留，fork 的并行 map 成员不再存在）。

## 13 C5 执行记录（段末 `6b9ba4f2e`，176 提交 / 525 文件 / 9 冲突文件 / 20 冲突段）

本段是计划里的最后一段，覆盖 10-16 → 10-24 的上游。规模上 `git diff --shortstat` 相对 C4 tip 为
`525 files changed, 60126 insertions(+), 29128 deletions(-)`；真正落到人工判断的只有 9 个文件。

### 13.1 ⭐⭐⭐ 双方各加一个 `ggml_op` ⇒ 合并后必须是 103，不是任何一侧的 102
`ggml-rpc.h` 的冲突只有三行注释，但三行注释下面压着一个**双方都不可能单独算对**的常量：

| 侧 | base | 该侧新增 | 该侧 `GGML_OP_COUNT` |
|---|---|---|---|
| base `6afa9a313` | — | — | 101（patch 0）|
| master | 101 | `GGML_OP_TURBO_WHT` | 102（patch 1）|
| 上游 | 101 | `GGML_OP_TOP_K_MASK` | 102（patch 1）|
| **合并结果** | | 两者都在 | **103（patch 2）** |

两边的 `static_assert` 都写 102，`ggml.h` 的枚举却被 git 自动并成了 103 项（两个新增落在不同位置，
互不冲突），于是**自动合并没有冲突、断言值两边一致、但两者都是错的**。这类冲突在 diff 里看不到任何标记。
处置：`RPC_PROTO_PATCH_VERSION 1 → 2`、`GGML_OP_COUNT == 103`；`ggml.c` 里两处同名断言
（`GGML_OP_NAME` / `GGML_OP_SYMBOL` 表各一条）同步改 103 —— 它们同样由自动合并留下陈旧值。
两张表都已含 `TOP_K_MASK`（枚举、名称、符号三条一致，符号表 103 项）。

### 13.2 ⭐⭐ `norm.cu`：上游的 float4 路径整体接管，fork 的 warp-per-row 只补在融合 scale 分支
上游把 6 处启动点收敛成 `rms_norm_f32_dispatch<do_multiply, do_add, do_scale>`，并在 kernel 内加了
`do_vec` + `vpt` 的 float4 快路；fork 这边的改动是**相反方向**的：把 base 里
`for (sample) { for (channel) { ... } }` 的钳位循环拆掉（改 1 block = 1 行直接索引）、`scale_out → post_scale`、
`do_scale` 单独一轮乘。两侧改的是同一段的两端，硬拼会得到「没有钳位循环、却引用 `xc/dstc`」的半成品。

判据：fork 拆循环在 master 历史里 `git log -S` 追不到对应提交（是整文件搬入时带进来的），没有测量记录；
而上游的快路有明确收益，且保留了 `grid.y/z` 钳位（`nchannels > 65535` 不再越界）。⇒ **整文件取上游**，
只把 fork 唯一有测量的改动补回去：`rms_norm_scale_f32_cuda` 的 `ncols < 1024` 分支用 `WARP_SIZE` 而不是 256
（`523b6cf03`，42.0 → 31.9 us/launch）。补在 dispatch 的 else-if 里并以 `if constexpr (do_scale)` 收窄，
其它融合形态（mul / mul+add）行为不变。`do_scale` 的两次舍入语义两侧本就等价（`scale*x` 一次、`*post_scale` 一次）。

### 13.3 ⭐⭐ `ggml_cuda_op_rms_norm_scale` 改名：链接器抓到的静默不一致
`norm.cu` 取上游后，定义是 `ggml_cuda_op_rms_norm_scale_fused`（base 与上游同名），而 `norm.cuh` 声明和
`ggml-cuda.cu:6679` 的调用点仍是 master 改名后的 `ggml_cuda_op_rms_norm_scale`。
`norm.cuh` 的改名单侧变更、自动合并保留 → 头文件与实现不同名，**编译期全绿，链接期 `LNK2019`**。
判定：该改名是纯装饰（base 就是 `_fused`），fork 侧无第二用途 ⇒ 声明与调用点都改回 `_fused`，归一为上游命名。

### 13.4 `mmvq.cu`：两处都是「上游重构 + fork 表项」的并集
- `calc_rows_per_block`：上游把 `ncols_dst == 1` 拆出 `mmvq_two_rows_ncols1()`（Q1_0/Q2_K/Q3_K/IQ1_S/IQ1_M 两行）
  与 GB10 的 IQ2_S/IQ3_S 特例；fork 加了 PTQ1_0 的 PT 分支与 `MMVQ_PARAMETERS_VOLTA`。**并集**：
  PTQ 门在前，表列表含 VOLTA（否则 Volta 掉到 `return 1`），`case 1` 取上游新分支 ——
  VOLTA 命中最后的 `table_id == GENERIC && ...` 为假，仍返回 1，与 master 行为一致。
- 内核里 `rows_per_cuda_block`：保留 fork 的 `rows_per_block_override` 模板参数（上游是纯 constexpr）。

### 13.5 `ggml-cuda.cu`：`WARP_SIZE` 让位给运行时 `warp_size`
`ggml_cuda_should_use_mmf(...)` 的第 3 参 master 传编译期 `WARP_SIZE`，上游传运行时
`ggml_cuda_info().devices[ctx.device].warp_size`。取上游：CDNA 是 wave64，编译期 32 会让 mmf 判定错。

### 13.6 其余并集
| 文件 | 处置 |
|---|---|
| `src/llama-impl.h` | `gguf_kv_to_str(..., size_t max_len = SIZE_MAX)`（上游的截断参数）+ fork 的 `llama_crc32` 声明，两条并存；定义侧 `llama-impl.cpp:169` 已带 `max_len` |
| `tools/server/server-task.h` | fork 的 `server-cache-lease.h` / `server-retention-sidecar.h` / `server_cache_plan_payload_kind` / `server_cache_lease_build_identity` 与上游新增的 `mtmd-helper.h` 并存 |
| `tests/CMakeLists.txt` | fork 的 20+ 个 target（vbr / kv-stream / turbo / cache-control / llama-archs…）全部保留；采纳上游新增的 `LABEL behavior` 分级与 `test-gelu-quick.cpp`；`test-arg-parser` / `test-model-resolution` 用带上游 LABEL 的形式，避免同一 target 注册两次 |
| `server-context.cpp` §1 | 上游新插的 `process_tts_slots()` try/catch 与 fork 的「空 batch 告警（含 diffusion 豁免）」是同一锚点上的两段独立代码 ⇒ 顺序并集 |
| `server-context.cpp` §2 | checkpoint 失效判定取 fork：`prompt_pos_end`/`true LCP` 双判据 + `checkpoint_drop(..., checkpoint_invalidated)` 记账，上游那版只比 `pos_next / n_prefix` 且直接 `erase` |
| `server-context.cpp` §3 | 投机解码：fork 的 `verified_output_rows` 快照 + 上游把 `slot.spec_i_batch` move 进局部 `spec_i_batch`。**先快照再 move**，两处读者都能拿到 |
| `server-context.cpp` §4 | `server_context_meta` 指定初始化器：上游新增 `has_cap_chat/has_cap_tts/has_audio_stream` + `server_model_output_modalities(model, chat, tts)`，fork 的 17 个 VBR 字段全保留，顺序按 `server-context.h:250` 的声明（C++20 指定初始化器要求声明序） |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | 保留 fork 的 `if constexpr (!V_is_turbo)` 屏障豁免，采纳上游的注释 |
| `ggml/src/ggml.c` | 两处 `static_assert(GGML_OP_COUNT == ...)` 见 13.1 |

### 13.7 ⭐ 补记 C2 的一处静默删除（本段核对时发现）
`ggml/src/ggml-cpu/iqp.cpp` / `iqp.h`（1253 + 39 行）在 **C2（`56565e159`）** 就没了：上游
`d834d44e6`（tiled mul_mat for k-quants #27851）在 base 之前就把这两个文件删了，C2 合 `ggml-cpu/CMakeLists.txt`
时按上游的源码列表走，于是 fork 树里残留的副本一起消失。
判定**不是损失**：master 里 `IQP` 只出现在 `iqp.cpp` 自己（61 次），没有 `GGML_TYPE_IQP`、没有 type traits、
没有任何调用点，唯一的外部引用就是那份 CMake 源列表。属上游已废弃的实验实现，跟随上游删除。

### 13.8 逐段核过但判为「非损失」的两处
- `ggml/src/ggml-cuda/argsort.cu`：合并结果与上游整文件相同（`101afeba2`），master 侧是 `24115da0`。
  原因是**上游先修了同一个 bug**（`b23701f77` "fix CUB argsort corruption caused by in-place keys" #28389），
  base 已含该修复，master 的 `ce579daec` 是同一修复的独立实现；上游版本还多带 `ggml_cuda_chunk_nrows` 重构与
  bitonic 辅助函数整理 ⇒ 取上游严格更优。`temp_keys_out` 在合并结果里 14 处，修复仍在。
- `ggml/src/ggml-cuda/norm.cuh`：与上游逐字节相同 —— 即 13.3 的改名归一，是有意为之。

### 13.9 验证
- `cmake --build tmp/build-c1 -- -k 0`：**RC=0，159/159**（增量），0 error / 0 LNK。
  中途两次失败的成因都记一笔，避免下次误判：①`ninja: failed recompaction: Permission denied` = 上一段遗留的
  C4 后台 `ninja` 进程仍持有 `.ninja_log`（PID 存活但 CPU 不再增长）⇒ 杀进程重跑；②`LNK2019` 见 13.3。
- ctest：**104/113 通过，9 失败**，与 C4 的失败集合**逐项相同**（`diff` 空集差异）：无 GPU `0xc0000409`
  （`test-arg-parser` / `test-backend-meta-replay` / `test-opt`）、无网（`test-download-model` /
  `test-eval-callback-download-model`）、无模型（`test-llama-context-model` / `test-thread-safety` /
  `test-eval-callback`）、无 jinja2（`test-jinja-py`）。总数 112 → 113 是上游新增 `test-gelu-quick`。
- 零损失：`fork 9l-` 593→601、PQ2_0 292→306、PTQ1_0 372→385、`GGML_TYPE_PQ2_0` 117→118、
  `GGML_TYPE_PTQ1_0` 113→113、TURBO_WHT 59→63、`vbr_pools_` 154→158、kv_stream 1371→1375、
  `MMVQ_PARAMETERS_VOLTA` 11→15、`prism.hadamard` 58→106、`llama_verify_hadamard_graph` 5→6，
  以及 test-vbr / test-kv-stream / test-moe 全部 ≥ master。
  `hadamard_rotations` 21→13 = C4 §12.1 的 `hdmd` 归并（fork 的并行 map 成员不再存在，typedef 与读取点保留）。
- M2（整文件退回 base 的 master 改动）= 0；M5（master 新增文件在结果中缺失）= 0（过滤掉被 master 自己
  改名/删除后又出现在 `--diff-filter=A` 里的噪声项后实测 0）。
