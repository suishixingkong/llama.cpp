# Qwen3.8-27B on V100 (sm70) GQA packing — integration notes

来自 `qwen38-v100-serve-main` 仓库的 T2-001 补丁：`flash_attn_ext_vec` 增加 `ncols2 = 3`
的查询头打包，针对 GQA 比含因子 3（如 Qwen3.8-27B 的 24/4 = 6）在 Volta 上消除 KV 头
的重复读取。

| 来源 | 本树 commit | 上游 base | 取用范围 |
|---|---|---|---|
| [`jackinthebox52/qwen38-v100-serve`](https://github.com/jackinthebox52/qwen38-v100-serve) | `bfd6c4416` | `4f75b8735` | 1 文件 |

结果：**1 文件，`+96 / -32`**（`ggml/src/ggml-cuda/fattn-vec.cuh`），落在 `turbo-kvstream-merge`
上，作者 yangj，2026-09-15。

原补丁针对 `b10793`（`d230ddd`）的 `flash_attn_ext_vec`；因本树在 turbo-quant 合并之上，
该文件无法直接 `git apply`，故按手移植到 `4f75b8735`。功能一致，唯一差异（与仓库内
`patches/README.md` 记录一致）：

1. `nthreads_KQ` / `nthreads_V` 沿用 fork 的 turbo-aware selector 形式（`nthreads_f` 替代
   `128 / cpy_nb`），在打包路径上语义相同（打包路径仅限于 F16/BF16 K/V）；
2. 上游在 `b10793` 之后新增的 `ggml_cuda_flash_attn_ext_vec_partial_case_impl` 也转发
   `ncols2 = 1` 到内核模板；
3. 保留 turboquant 分支的 `__launch_bounds__(..., 2)`（原补丁假设 `..., 1`）。

## 激活方式（重点）

**无需任何编译开关、无需环境变量、无需运行时参数。**

该改动是对 `flash_attn_ext_vec` 行为的自动修改，由 GQA 几何在运行时直接触发：

- 仅当 `gqa_ratio % 3 == 0 && Q->ne[2] % 3 == 0` 时走 `ncols2 = 3` 打包路径
  （例如 Qwen3.8-27B 的 24/4 = 6）；
- 通过 `if constexpr (packing_supported)` 限定于**未量化 K/V（F16/BF16）**；`q8_0` KV 仍走
  原未打包内核（量化路径在 D=256 已占满 ~252–255 寄存器，再打包会 spill 到本地内存）；
- 由 `__CUDA_ARCH__ == sm_70` 的寄存器预算决定收益——主要在 Volta 上见效，其它架构编译
  存在但无此收益；
- 需要 `-fa on`（FlashAttention）；`q8_0` KV 不会触发打包。

换句话说：**给 Qwen3.8-27B 用 F16/BF16 KV 跑 FlashAttention 即可自动获得，什么都不用设。**
没有 kill-switch，要回退只能改源码（删掉 `fattn-vec.cuh` 里 `packing_supported` 那段）。

## 性能与质量（来自 fork 在 V100-32GB 上的实测）

| 上下文深度 | 原生 decode (tok/s) | 补丁后 (tok/s) | 加速 | KV DRAM/词 |
|---|---:|---:|---:|---:|
| 1K   | 35.95 | 35.22 | -2.0%   | — |
| 8K   | 34.42 | 34.39 | 噪声内  | — |
| 32K  | 30.29 | 32.04 | +5.8%   | — |
| 64K  | 25.63 | 29.04 | +13.3%  | — |
| 128K | 16.45 | **23.83** | **+44.9%** | **26.37 GB → 8.59 GB** |

- 数值等价：平均 KL 散度 = 0，Top-1 token 一致率 = 100%。
- 权衡：浅上下文（<8K）约 -2.0%（打包块更少），换取深上下文 +44.9%。

## 部署注意（fork 的 serve.sh 选择，非代码限制）

- MTP（`--spec-type draft-mtp`）一次验证 3–7 个 token，4-wide 验证天然摊薄 KV 头读取，
  叠加 T2-001 仅 ~1% 提升，故 fork 的 `./serve.sh`（默认 MTP 开）跑**未打补丁的 stock 构建**；
  `./serve.sh --no-mtp` 才跑打补丁构建。这是部署取舍，内核本身不禁止 MTP 下打包。

## 验证情况

- 原补丁验证：25 个 `fattn-vec-instance-*.cu` + `fattn.cu` 干净编译；`sm_70, D=256,
  ncols1=1, ncols2=3, F16 K/F16 V` 的 ptxas 报告 **207 寄存器，0 spill**（softcap 开：135
  寄存器，0 spill），即 3 头打包落在 Volta 255 寄存器预算内。
- 本树：改动已合入 `turbo-kvstream-merge`（`bfd6c4416`），但本机无 NVIDIA 设备，
  **未做端到端数值验证**。
