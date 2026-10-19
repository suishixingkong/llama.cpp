# Prism 移植 — 实机测试方案（V100 / GTX 950）

> 对象：分支 `prism-quant` 上的 Prism 移植内容 —— 私有量化格式 PQ2_0/PTQ1_0、Hadamard 权重折叠、
> CUDA 原生 mmq/mmvq、激活布局（SoA 单列 / PT 列共享 / PQ2 多列）、Vulkan 后端。
> 开发机无 GPU，下面每一条都是**开发机上做不到、必须在有卡机器上补**的判据。
> 权威实现记录见 `docs/MERGE_PRISM_QUANT.md`；本文件只讲**怎么验、验到什么算过、过不了怎么归因**。

---

## 0 环境与前置

| 项 | 值 |
|---|---|
| 机器 | V100-SXM2-16GB（sm_70）+ GTX 950 2GB（sm_52），Windows，MSVC 2022 14.40，CUDA 12.x |
| 模型 | `models/Ternary-Bonsai-2-2B-PTQ1_0.gguf`（真实折叠模型）；`models/qwen2.5-0.5b-q4km.gguf`（无折叠回归）；`D:\llama-build\models\q2.5-{pq2_0,ptq1_0}.gguf`（自量化对照） |
| 参照 | `D:\llama-build\prism-cpu\bin\*`（fork 自己的二进制，CPU）—— 数值只跟它比，不跟"文本通不通顺"比 |
| 构建 | `D:\llama-build\cuda70`（**静态**，`BUILD_SHARED_LIBS=OFF`）、`D:\llama-build\vulkan`（Vulkan） |

**前置 0.1 — 静态链接必须先过**（本次已修，但要复核）

```bat
cmake --build D:\llama-build\cuda70 --target llama-cli llama-server -- -j 12
```

判据：**0 个 LNK2019**。此前 `ggml-backend-meta.cpp` 把
`ggml_backend_dev_is_meta` / `ggml_backend_meta_dev_n_devs` / `ggml_backend_meta_buft_n_bufts` 等 8 个
GGML_API 符号定义成 `static`，静态构建必挂。若这里仍然红，**不是本次移植的问题**，先报它。

**前置 0.2 — 环境变量纪律**

- `--device` 只管"放哪儿"，`CUDA_VISIBLE_DEVICES` 才管"看得见哪块"。单卡测试统一 `set CUDA_VISIBLE_DEVICES=0`。
- cmd 里 `set X=` 是**删变量**，不是设为空串。
- 手跑一律补 `-c 4096`（不传 ⇒ n_ctx=262144 ⇒ 单 context KV 3072 MiB，显存贴边必红）。
- 16 GB 卡上 2B 模型 + `-ngl 99` 之外不要开 `-np > 1`（交换是进程级的）。

---

## 1 CUDA 内核正确性：先用 CPU 当参照

这是整个方案里**判别力最强**的一步，因为 `test-backend-ops` 逐元素比对 GPU 与 CPU 的结果，
而 CPU 侧已在开发机上与 prism 参照二进制逐位对齐过。

### 1.1 全类型 mat-mul / get_rows

```bat
cd /d D:\llama-build\cuda70\tests
set GGML_CUDA_ENABLE_UNIFIED_MEMORY=0
test-backend-ops.exe -o MUL_MAT        -b CUDA0 2>&1 | findstr /C:"not supported" /C:"FAIL" /C:"OK"
test-backend-ops.exe -o MUL_MAT_ID     -b CUDA0 2>&1 | findstr /C:"not supported" /C:"FAIL" /C:"OK"
test-backend-ops.exe -o GET_ROWS       -b CUDA0
```

`-o` 用 `MUL_MAT` 会覆盖 `all_types/base_types/other_types` 里新增的 PQ2_0/PTQ1_0，
以及我们补的 Bonsai 形状用例（m=67 的行尾、n=1..8、k∈{128,384} 的非 256 倍数行宽）。

| 判据 | 通过条件 |
|---|---|
| FAIL 计数 | **0**（`not supported` 不算 FAIL，但要单独记账） |
| PQ2_0/PTQ1_0 是否 SKIP | 不应 SKIP。若 SKIP，看 `-o` 输出的原因：多半是 `supports_op` 白名单没接上 |
| 逐元素误差 | 由 test-backend-ops 自带的 NMSE 门限判定，不要自己改门限 |

> ⚠️ **cwd 就是环境前提**：`tests/CMakeLists.txt` 里 `WORKING_DIRECTORY` 默认是构建目录，
> 而部分用例用相对路径读夹具。手跑先 `cd` 到 `D:\llama-build\cuda70\tests`。
> 同一个二进制在 `bin\` 与源根下跑出的结果可以一个是 225 个 OK、一个是 0 个 —— 那不是移植问题。

### 1.2 布局分支是否真的被选中（需要一次性探针，**不提交**）

三套激活布局的选取是纯 host 逻辑，正确性靠 1.1，**覆盖性**必须单独证：

| 类型 | 列数 | 设备 | 期望布局 |
|---|---|---|---|
| PTQ1_0 | 1（普通 decode） | V100 sm_70 | `SOA_ISUM` |
| PTQ1_0 | 2–8（投机验证 / MTP） | 任意 | `PT` |
| PTQ1_0 | 任意 | 任意 + `GGML_CUDA_BATCH_INVARIANT=1` | `PT`（含 1 列） |
| PTQ1_0 | MoE（有 ids） | 任意 | `PT` |
| PQ2_0 | 3–8 | **只有 sm_89** | `PQ2` |
| PQ2_0 | 任意 | V100 / 950 | `AOS`（多列核不可用） |

取法（临时改，验完 `git checkout --`）：

```cpp
// ggml/src/ggml-cuda/mmvq.cu, ggml_cuda_mul_mat_vec_q 里 layout 定完之后
static std::once_flag once_;
std::call_once(once_, [&]{ LLAMA_LOG_INFO("prism probe: type=%s ncols=%lld ids=%d layout=%d cc=%d\n",
    ggml_type_name(src0->type), (long long)(ids?ne2:ne11), ids!=nullptr, (int)y_layout,
    ggml_cuda_info().devices[ggml_cuda_get_device()].cc); });
```

判据：
- V100 上 PTQ1_0 单 token decode ⇒ `layout=1`（SOA_ISUM）；MTP/投机（ncols 2..8）⇒ `layout=2`（PT）。
- `GGML_CUDA_BATCH_INVARIANT=1` 与默认在**同一 prompt 同一 n_predict** 下输出 logits/文本一致
  （两条路径数学等价，仅在求和顺序上有差别；若文本发生跳变，说明两路至少有一路算错了）。
- PQ2_0 在 V100/950 上**永远不出现 `layout=3`**。若出现，说明 `ggml_cuda_pq2_multicol()` 的 cc 门失效。

> 探针会替被测对象干活吗？不会 —— 它只读已算好的 `y_layout`，不改任何状态。
> 但按纪律：**探针不提交**，交付时工作区必须干净。

---

## 2 Hadamard 折叠：与 prism 参照逐值对齐

折叠错了不会崩，只会静默算错，所以**必须**和参照实现对数值。开发机已做过 CPU 对照，这里补 GPU。

### 2.1 CPU 侧复现（基线，先做）

```bat
cd /d D:\llama-build\cpu\bin
llama-perplexity.exe -m D:\llama-build\models\Ternary-Bonsai-2-2B-PTQ1_0.gguf -f D:\temp\ppl.txt -c 512 --chunks 2 -ngl 0
cd /d D:\llama-build\prism-cpu\bin
llama-perplexity.exe -m D:\llama-build\models\Ternary-Bonsai-2-2B-PTQ1_0.gguf -f D:\temp\ppl.txt -c 512 --chunks 2 -ngl 0
```

期望（开发机实测值，复现用它当门限基线）：

| 构建 | PPL |
|---|---|
| 本树 | **13.9223** |
| prism 参照 | **13.9484** |

两者差 0.19%，来源是 buun lineage 自身在 CPU 数学上的发散。**门限：相差 < 1%。**
折叠错了会掉到几千以上，不会落在这个区间。

### 2.2 GPU 侧

```bat
cd /d D:\llama-build\cuda70\bin
llama-perplexity.exe -m D:\llama-build\models\Ternary-Bonsai-2-2B-PTQ1_0.gguf -f D:\temp\ppl.txt -c 512 --chunks 2 -ngl 99
```

| 项 | 通过条件 |
|---|---|
| 加载汇总行 | `loaded 152 Hadamard-folded weight(s) (1 inverse-lookup) using 1 rotation(s) and 2 sign vector(s)` |
| PPL | 与 2.1 的 CPU 值相差 **< 1%** |
| 折叠旋转变换是否上卡 | 若 `-ngl 99` 后 PPL 明显变差（>5%），先怀疑旋转张量落在 CPU buffer 而激活在 GPU（反向：`--no-kv-offload` 之类无关，看 `-v` 里 `prism.hadamard.*` 张量的 buffer 名） |
| 反向控制 | 见 2.3 |

### 2.3 反向控制（每次改判据都要重做）

把 `build_hadamard_activation()` 开头短路成 `if (true) { return cur; }`，重编，重跑 2.2：

- 期望：进程在**首次 compute 处直接终止、无输出**（`llama_verify_hadamard_graph` 抛异常，把静默算错变成报错）。
- 若"短路后仍然正常出数"⇒ 图级校验没生效，**折叠路径不可信**，后续所有数值都不作数。
- 还原后 PPL 必须精确复现（同一二进制同一参数，逐位相同）。

---

## 3 CUDA mmq/mmvq 原生核

### 3.1 正确性

已由 1.1 覆盖（`test-backend-ops -o MUL_MAT` 会分别走到 mmvq（n≤8）与 mmq（更大 n））。

补充判据：**PTQ1_0 默认全 batch 走 MMQ**（fork 记录 fp16 反量化 + cuBLAS 是它掉精度的来源）。
A/B 开关：

```bat
set GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0     & rem 强制回 cuBLAS 反量化路径
set GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=1000000
```

两条路径的 PPL **可以**不同（这是 fork 记录的真实精度差），但**都必须**低于"折叠错"的量级；
如果关闭 MMQ 后 PPL 反而更好，说明我们的 MMQ 实现有 bug，回来查 §3.2。

### 3.2 解码数学对不上时怎么定位

`vec_dot_ptq1_0_q8_1`（4 字节并行 base-3 链）与 `ptq1_0_pt_block_dot` 在开发机上已用
"模拟 `__byte_perm`/`__vsub4` + 索引算术"对照参考反量化做过 300/300、100/100 逐元素一致。
实机若不一致，按这个顺序查：

1. `test-ptq1_0-element-map.exe`、`test-ptq1_0-cuda-dot.exe` 必须绿（纯主机逻辑，与 GPU 无关）。
   它们红 ⇒ 转写件与 `ggml-quants.c` 已经漂移，先把测试里的转写件更新。
2. `test-backend-ops -o MUL_MAT -b CUDA0` 里 **PTQ1_0 n=1** 红而 n≥2 绿 ⇒ 问题在 SoA 布局
   （`ggml_cuda_ptq1_q8_word` 或 `vec_dot_ptq1_0_q8_1_multi`）。
3. n≥2 红而 n=1 绿 ⇒ 问题在 PT 布局（`ptq1_0_pt_nblk` 用了未 padding 的 K，或 `quantize_q8_1<PT>` 的
   `nblk` 与 kernel 侧不一致）。
4. 全红 ⇒ 先看 `MmTypeA`/`vdr` 的派发表，再看 `should_use_mmvq` 是否把该形状送错了路径。

### 3.3 性能（不作为通过判据，只记录）

```bat
llama-bench.exe -m D:\llama-build\models\Ternary-Bonsai-2-2B-PTQ1_0.gguf -ngl 99 -p 512 -n 128 -r 3
```

记录 `pp512` / `tg128`，与"关掉原生核"的对照（`GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0`）比。
V100 的结论**不要**照搬 fork 在 Ada 上的数字（fork 的多列核调的是 Ada）。

---

## 4 Vulkan 后端（GTX 950 / V100）

### 4.1 编译期（开发机已可验）

```bat
cmake --build D:\llama-build\vulkan --target ggml-vulkan -- -j 12
```

判据：`ggml-vulkan-shaders.cpp` 生成成功（`dequant_pq2_0` / `dequant_ptq1_0` /
`mul_mat_vec_{pq2_0,ptq1_0}_{f32,f16}_f32` / `mul_mat_vec_id_*` 全部产出），`ggml-vulkan.lib` 链接通过。
任何 shader 的 glslc 报错 = 我们的 GLSL 有问题（`dequant_funcs.glsl` 的 `ptq1_0_val`、
`types.glsl` 的块结构、`dequant_ptq1_0.comp` 的 pow3 索引）。

### 4.2 运行期

```bat
cd /d D:\llama-build\vulkan\bin
set GGML_VK_VISIBLE_DEVICES=0
test-backend-ops.exe -o MUL_MAT  -b Vulkan0
test-backend-ops.exe -o GET_ROWS -b Vulkan0
llama-cli.exe -m D:\llama-build\models\Ternary-Bonsai-2-2B-PTQ1_0.gguf -ngl 99 -c 4096 -n 32 -p "The capital of France is"
```

| 项 | 通过条件 |
|---|---|
| `test-backend-ops -b Vulkan0` 的 PQ2_0/PTQ1_0 | 与 CPU 参照在 NMSE 门限内（Vulkan 侧是**同一套 dequant 数学**，只是并行方式不同） |
| 若被 SKIP | 打印 `not supported`：说明 `supports_op` 白名单或管线查找没接上，**不是**"Vulkan 不支持"，回来查 |
| 端到端文本 | 与 CPU 结果一致（`-ngl 0` 对照） |

**开发机已跑过的部分（Intel UHD 730）**：`MUL_MAT` 1236/1236、`GET_ROWS` 255/255 通过，
所以上面两条 `test-backend-ops` 判据在 iGPU 上已经满足；**V100 / GTX 950 上要重跑**（不同 subgroup 宽度、
不同 shared memory 上限，走的是不同的 DMMV workgroup 档位）。

**折叠模型（Bonsai 2）已通**（2026-10-19，Intel UHD 730）：`-ngl 99` 真跑完
（`llama-cli` prefill 11 token + 24 token decode；`llama-perplexity -c 512 -b 512` 1025 token，
PPL **2.8440**，纯 CPU 参照 `-dev none -ngl 0` 为 **2.8522**）。期间修掉两个缺陷，
根因与证据见 `docs/MERGE_PRISM_QUANT.md` §8.1：
① `llama_verify_hadamard_graph()` 原本在 `graph_compute()` 里跑，那张图已被 `sched_reserve` 插过 copy，
   embedding 的 `MUL_MAT.src[1]` 被改写 ⇒ 误判"没做逆变换" ⇒ 抛异常终止（**静默**是因为
   `common/log.cpp` 的日志走异步 worker 线程，进程一死队列全丢）；
② 无 coopmat 的设备上 `qx_needs_dequant` 恒真，`token_embd.weight`（2048 × 248320 = 5.09 亿元素）
   一次性 dequant ⇒ `wg = 496640 > maxComputeWorkGroupCount = 65536`。

⇒ **V100 / GTX 950 上重跑时，把这条当回归项**：同一模型 `-ngl 99` 必须能出 PPL，且与 `-dev none -ngl 0`
在同一语料上落在 ±1% 内。若再次出现"无 error 直接退出"，先用 `GGML_LOG_SYNC=1` 打开同步日志
（环境变量，配合一处临时旁路；日志变同步后异常前的最后一行才有意义）再二分。

**已知取舍**：Vulkan 侧目前只走 `mul_mat_vec`（decode 快路径）+ `dequant → f16`（prefill）。
没有 coopmat/MMQ 版本，所以 `pp512` 会明显慢于 CUDA —— 这是设计取舍，不是缺陷。
PTQ1_0 的 `dequantize` 是逐元素 base-3 解码（未做 fork 那种按字节并行），带宽受限场景下够用，
但**不要**拿它和 fork 的 Vulkan 数值比较性能。

⚠️ GTX 950 只有 2 GB：Bonsai 2B（1.75 bpw ≈ 0.5 GB 权重）能放下，但要加 `-c 2048`；
2B + `-c 4096` 可能 OOM，OOM 不是移植缺陷。

---

## 5 端到端：llama-server

```bat
cd /d D:\llama-build\cuda70\bin
llama-server.exe -m D:\llama-build\models\Ternary-Bonsai-2-2B-PTQ1_0.gguf -ngl 99 -c 4096 --port 18097 --host 127.0.0.1
```

```bat
curl http://127.0.0.1:18097/health
curl -X POST http://127.0.0.1:18097/completion -H "Content-Type: application/json" -d "{\"prompt\":\"2+2=\",\"n_predict\":8,\"temperature\":0}"
```

| 项 | 通过条件 |
|---|---|
| `/health` | `{"status":"ok"}` |
| `/completion` | 返回 JSON，`content` 为 `4\n3+3=6\n`（开发机 CPU 实测值；GPU 上允许 token 级差异，但不允许空/乱码） |
| 加载日志 | 有 `loaded 152 Hadamard-folded weight(s) ...`，且**没有**任何 `prism.hadamard` 拒绝信息 |
| 并发 | `-np 1` 只测一遍；`-np > 1` 不属于本次验收范围 |

---

## 6 回归（每次都要跑，防止把既有能力弄坏）

| 用例 | 命令 | 门限 |
|---|---|---|
| 无折叠模型 | `llama-perplexity -m qwen2.5-0.5b-q4km.gguf -ngl 0` | PPL **20.8087**（阶段一基线，逐位） |
| 普通类型 CUDA | `test-backend-ops -o MUL_MAT -b CUDA0`，只看非 Prism 类型 | FAIL 0 |
| 静态链接 | §0.1 | LNK2019 **0** |
| 既有守卫 | `python D:\llama-build\guard_sweep.py` | 与基线同为 2 failing（这是既存债务，不是本次引入）；⚠️ 必须固定 cwd，单跑 1 vs sweep 2 |

---

## 7 归因顺序（红了先分三类）

1. **用法/参数**：`-c` 漏传、`-ngl` 与 `--device` 混用、`CUDA_VISIBLE_DEVICES` 没设、cwd 不对。
2. **环境前提**：显存不足（贴边）、静态 vs 共享构建混用、Vulkan 设备不选对。
3. **真缺陷**：以上都排除后，按 §2.3 → §1.2 → §3.2 的顺序，从"最容易被静默跳过的那一层"往上查。

**"日志干净截断"也是判据**：末行是完整普通日志且无 error ⇒ 多半是访问违例，不是 abort。
定位法 = 在源码里找该行之后的第一个打印，崩溃必在其间，再看窗口内谁消费外部指针
（阶段一就踩过：`load_model()` 把局部 `load_progress_data` 的地址留在长寿命 `params_base` 里）。

**"没有 FAIL" ≠ "绿"**：守卫脚本可能 `SyntaxError` 静默退出，必须同时看到 `RC=` 与 `PASS`/`SKIP`。

---

## 8 一页纸清单

```
[ ] 0.1 静态构建 0 个 LNK2019
[ ] 1.1 test-backend-ops MUL_MAT / MUL_MAT_ID / GET_ROWS on CUDA0：FAIL 0，Prism 两型不 SKIP
[ ] 1.2 探针确认 V100 上 PTQ1_0 1列=SOA_ISUM、2..8列=PT、BI=1 时=PT；PQ2_0 永不 PQ2
[ ] 2.1 CPU PPL 复现 13.9223 / 参照 13.9484
[ ] 2.2 GPU PPL 与 CPU 差 <1%，加载行 152/1/1/2
[ ] 2.3 短路反向控制：首次 compute 终止；还原后逐位复现
[ ] 3.1 PTQ1_0 MMQ 默认全 batch；MAX_BATCH=0 的 A/B 记录在案
[ ] 3.3 llama-bench 数字记录（不设门限）
[ ] 4.1 Vulkan shaders 全部生成 + lib 链接
[ ] 4.2 Vulkan0 上 MUL_MAT/GET_ROWS 与 CPU 一致；端到端文本与 CPU 一致
[ ] 5   server /health + /completion
[ ] 6   回归四项
[ ] 7   失败项按三类归因，附证据行
```
