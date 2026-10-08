# turbo4 实现对比审计：`master`（tq 血统）vs `buun-llama-cpp`

日期：2026-09-23 ｜ 对照提交面：本地 `master`（tq/atomic 血统集成）与 `buun-master`（远程 `buun` → `C:\Users\confu\llama\buun-llama-cpp`）

## 0. 结论

**buun 的 turbo4 在“每比特质量”上占优，且优势可量化、可复现。** 两边跑的是**同一个变换**（同一 FWHT、
同一对 ±1 符号表、同一 `1/√128` 归一化、同一归一化+norm 校正结构），差别只在：

| 组件 | master（我们） | buun | 影响 |
|---|---|---|---|
| 16 级码本 | `TURBO_CENTROIDS_4BIT`，最外级 = **1.968σ** | `d_turbo_centroids_4bit`，最外级 = **2.733σ**（= N(0,1/√128) 的教科书 Lloyd-Max） | **主因**：实测高斯系数下 MSE **2.13×**；现实通道分布下 RMSE **1.13–1.40×** |
| 均值扣除（mean-sub） | **无**（完全没移植） | 逐层逐通道 `kmean_mu` 仿射 tap，表随运行时附带（覆盖 `qwen35`/`qwen35moe`） | 实测再降 **24.3%** RMSE（均值偏移分布） |
| 解码增益 alpha | **无** | `TURBO4_ALPHA` env → 存端 norm × alpha | 标量校正，量级小 |
| 每块字节数 | **68 B = 4.25 bpv**（`rnorm` 在 4-bit 模式下是死字段） | **66 B = 4.125 bpv** | 我们白花 0.125 bpv（≈ 该类型 KV 的 2.9%） |
| CPU/CUDA 一致性 | ✅ CPU `turbo_cpu_fwht` 与 CUDA butterfly 同域 | ❌ CPU `from_float = quantize_row_turbo4_0_ref` 用**稠密随机正交矩阵**，CUDA 用 FWHT | 跨设备（层 offload）会解码不符 |

**定量预测**：KLD 在低噪声区近似 ∝ 量化噪声功率。仅把码本换成 buun 的，turbo4 的 27B/16K KLD
应从 **0.0769 → ≈0.039–0.049**（即**反超 q4_0 的 0.0491**）；再叠加 mean-sub 可到 ≈0.03。

## 1. 逐层核对：先证明“变换本身一致”

两边的 CUDA 编码代数（逐行核对）：

```
# master  set-rows.cu  k_set_rows_turbo4
x[j] = src_row[i_blk*128 + j]
[可选 InnerQ: x[j] *= scale[j]]
inv_norm = 1/||x|| ; x[j] *= inv_norm
x[j] *= SIGNS1[j]
FWHT butterfly ×7 级
x[j] = x[j] * inv_sqrt_128 * SIGNS2[j]
idx = turbo_nearest_centroid_4bit(x[j])
blk->norm = fp16(grp_norm / recon_norm) ; blk->rnorm = 0

# buun    set-rows.cu  k_set_rows_turbo4_coop
v = src_row[ch] ; if (kmean_mu) v -= kmean_mu[ch]      # ← 差异
norm = ||v|| ; xr = ragged_fwht128_butterfly_inplace(v*inv_norm * d_turbo_wht_signs1[tid], smem)
xr = xr * 0.08838834764831845f * d_turbo_wht_signs2[tid]   # 0.08838834764831845 = 1/√128
idx = turbo_find_nearest_4bit(xr)
blk->norm = fp16((norm / recon_norm) * d_turbo4_alpha)      # ← 差异
```

⇒ 变换、符号、归一化、norm 校正**完全相同**；“最近码字”两边都用 15 个中点判决
（我们用线性 if 链 `TURBO_MID_4BIT`，buun 用同一组中点的二分，语义等价）。

## 2. 码本：唯一的主质量差异（数值 + 实测）

```
master : ±0.006938 ±0.020989 ±0.035597 ±0.051262 ±0.068756 ±0.089527 ±0.117195 ±0.173926
buun   : ±0.011353 ±0.034311 ±0.058069 ±0.083317 ±0.111065 ±0.143047 ±0.182907 ±0.241556
σ_theory = 1/√128 = 0.08839
最外级/σ :  master 1.968        buun 2.733
```
buun 的表就是 **N(0, 1/√128) 的 Lloyd-Max 最优解**（16 级高斯最优最外级 = 2.7326σ ✔）。
我们的表不是该分布的最优解——它的内/外间距比 0.040 vs buun 的 0.047，属于另一个形状的分布。

受控实验（`D:\llama-build\compare_turbo4_codecs.py`，两边同一变换、同一数据、只换码本）：

| 系数分布 | master rel-err | buun rel-err | 该分布上 Lloyd-Max 重拟合（理论最好） |
|---|---|---|---|
| 各向同性高斯 | 0.13404 | **0.09568** | 0.09560 |
| 重尾通道 σ | 0.12684 | **0.09391** | 0.09311 |
| 少数离群通道（σ=25×） | 0.10210 | **0.09009** | 0.07700 |
| 纯高斯系数 N(0,1/128) 的 MSE | 1.587e-04 | **7.436e-05** | — （比值 **2.134**） |

buun 的表在两种最现实的分布上几乎等于理论最优（0.09568 vs 0.09560；0.09391 vs 0.09311），
我们的表差 1.36–1.40× RMSE。**这就是“网上说 turbo4 很好、我们实测打不过 q4_0”的最可能解释**：
我们的 turbo4 是 tq 血统里**没有校准装置的裸编解码器**，而“网上”的数字多半来自带装置的实现。

## 3. mean-sub：buun 独有的第二个质量装置

- 机制：逐层、逐通道的 KV 均值向量（`kmean_mu`），在**raw 域**先减掉再走归一化+FWHT；
  解码侧加回。表是**离线 PFH1 标定烘进运行时的**（`ggml-turbo-meansub-data.inc`，618 KB），
  按键 `arch + n_layer + n_embd` 匹配，**不放在 GGUF 里**——理由是“同一架构的表可跨微调/量化档复用”。
  当前覆盖 `qwen35`、`qwen35moe` 两个架构；可用 `TURBO_KMEAN_SUB` / `TURBO_VMEAN_SUB` 覆盖。
- 收益实测：均值偏移分布上 rel-err **0.09580 → 0.07257（−24.3%）**。
- 成本：需要为每个目标架构做一次离线标定并随二进制附带（新增一个 618 KB 级数据面 + 注册表）。
- 我们这边：**一行都没有**（当初移植时按“丢调参装置”策略丢掉了）。

## 4. 我们反而占优的点（别被上面的结论盖掉）

- **CPU/CUDA 同域**：我们 CPU 参考用 `turbo_cpu_fwht`、CUDA 用 butterfly，两边都是 FWHT 域；
  buun 的 CPU `from_float = quantize_row_turbo4_0_ref` 走 `matvec(turbo_rotation, ...)`（稠密随机
  正交矩阵，`TURBO_SEED_ROTATION 42` + 修改 Gram-Schmidt），而 CUDA 走 FWHT
  ⇒ buun 的 **CPU 编码与 CUDA 解码不互通**，层 offload / CPU KV 会解出垃圾。
  这是 buun 侧的一个真 bug（也是我们移植时选择“统一 FWHT”的正确性理由）。
- 我们已有 InnerQ 框架（虽然按 §10.15 在 head_dim=256 上守卫失效），buun 侧也有 InnerQ
  且已识别同类风险（它用 `g_turbo_innerq_calibrated` 把 `turbo1_tcq` 切到 materialize 路径）。

## 5. 建议的验证（很便宜，两条命令级）

1. **只换码本 A/B（最高性价比）**：把 `TURBO_CENTROIDS_4BIT[16]`（`ggml-cuda/turbo-quant.cuh:297`）
   与 CPU 孪生表（`ggml-turbo-quant.c:493`，共 2 处）换成 buun 的 16 个值，重编后在 27B/16K 上
   重跑同一 base logits 的 `--kl-divergence`。预测：**0.0769 → ≈0.04**（反超 q4_0）。
   这是纯表替换，编解码自洽，零格式改动、零新增机制。
2. 若 1 命中，再评估 mean-sub：需要为 Qwen3.8-27B 的 arch 做一次离线标定（或先试
   `TURBO_KMEAN_SUB` 用现成 `qwen35` 表看趋势）。
3. 之前定的 MMA NaN bug（§10.14）与此无关，继续按 `--only` 单例排查。

## 6. 局限（必须声明）

- §2 的实验测的是**编解码器的 RMSE**，不是 KLD 面板。它是**受控对照**（同一变换/同一数据/只换码本），
  所以对“码本谁更好”这个结论是可靠的；但“KLD 0.0769 → 0.04”的换算依赖 KLD ∝ 噪声²的近似，
  以及合成分布对真实 K/V 的逼近程度 ⇒ 必须用第 5 节的实测确认。
- 未核对：`turbo-sink`、`turbo-tcq-alpha` 的完整语义，以及 buun 侧 V 的 mean-sub 与 VBR 转码的交互
  （`g_turbo_meansub_suppress` 等旁路）。
