# turbo2 / turbo3 / turbo4 三方对比：TheTom 上游 · buun · 我们的 master

日期：2026-09-23
对象：`llama-cpp-turboquant`（TheTom, `4deec5587`）· `buun-llama-cpp`（`buun-master`）· `llama.cpp`（我们 `master`，turbo2/3/4 与 `atomic-llama-cpp-turboquant` **逐字节相同**）
实验脚本：`D:\llama-build\compare_turbo234_threeway.py`（同一 FWHT、同一数据、逐码本对比）

## 0. 排序结论

**默认配置下：`TheTom ≥ buun > 我们(atomic 血统)`。**

- **TheTom**：三类格式都取到理论最优码本 + 最紧凑块 + 附带性能修复与 turbo4 MMA 路径。没有多余资产，也没有冗余/桩。
- **buun**：turbo4 与 TheTom 同级，**且额外带 meansub/alpha（端到端天花板最高）**；但 turbo2/3 白花 0.375 bpv、turbo2/3 的 CPU 参考是**桩函数**、CPU 与 CUDA 的 turbo4 旋转域**不一致**（层 offload 会解出垃圾），还要背 618 KB 离线标定资产。
- **我们**：turbo2/3 与 TheTom 同级；**turbo4 码本次优（唯一实质质量差距）** + 68 B 浪费 + InnerQ 在 head_dim=256 失效。

## 1. 三方事实表

| | TheTom（上游） | buun | 我们 master（= atomic） |
|---|---|---|---|
| **turbo2 块** | QK=128，34 B = **2.125 bpv** | QK=32×4，40 B = **2.5 bpv** | QK=128，34 B = **2.125 bpv** |
| turbo2 码本 | `±0.133462/±0.039994`（4 级 LM，最外 1.51σ） | **完全相同** | **完全相同** |
| turbo2 流水线 | 128 组归一化 → WHT → 2-bit | 同（但组范数**复制到 4 个子块**） | 同 |
| **turbo3 块** | QK=128，50 B = **3.125 bpv** | QK=32×4，56 B = **3.5 bpv** | QK=128，50 B = **3.125 bpv** |
| turbo3 码本 | `0.190207…`（**精确 LM**，最外 2.152σ） | `0.190685…`（= 我们的）＋可选经验/**非对称** K/V 专用表；`TURBO3_SKEW_EXP` 指数偏移 | `0.190685…` |
| **turbo4 块** | QK=128，66 B = **4.125 bpv** | QK=128，66 B = **4.125 bpv** | QK=128，68 B = **4.25 bpv**（死字段 rnorm） |
| turbo4 码本 | `0.241529…`（**LM**，最外 2.733σ） | `0.241556…`（**LM**，独立重算） | `0.173926…`（最外 **1.968σ**，次优） |
| 额外质量装置 | 无 | **mean-sub**（离线烘表）＋**alpha**（`TURBO4_ALPHA`） | InnerQ（默认关；head_dim=256 上守卫失效） |
| CPU 参考 | 真实（`turbo_cpu_fwht`，与 CUDA 同域） | turbo4 用**稠密随机旋转**（与 CUDA 的 FWHT 不同域）；turbo2/3 是**桩** | 真实（`turbo_cpu_fwht`，同域） |
| 绝对 bpv（tok/128） | 34 / 50 / 66 | 40 / 56 / 66 | 34 / 50 / 68 |

## 2. 受控实测（同一 FWHT、同一重尾通道数据，6000 块）

| 变体 | 字节/128 | bpv | rel-err | 相对最优 |
|---|---|---|---|---|
| turbo2 TheTom & 我们 | 34 | 2.125 | **0.33013** | 1.000× |
| turbo2 buun | 40 | 2.500 | **0.33013** | 1.000×（失真相同，多花 0.375 bpv） |
| turbo3 TheTom（精确 LM） | 50 | 3.125 | **0.17715** | 1.000× |
| turbo3 atomic/我们/buun | 50 | 3.125 | 0.17749 | 1.002× |
| turbo3 buun（QK=32 格式） | 56 | 3.500 | 0.17749 | 1.002×（多花 0.375 bpv） |
| turbo4 TheTom & buun（LM） | 66 | 4.125 | **0.09304** | 1.000× |
| turbo4 atomic/我们 | 68 | 4.250 | 0.12092 | **1.300×** |

纯高斯系数 N(0,1/128) 上的理论 MSE（码本本身的优劣）：

| 码本 | MSE | 说明 |
|---|---|---|
| 2bit | 9.1606e-04 | 三方同表 |
| 3bit TheTom | 2.6949e-04 | 精确 LM |
| 3bit atomic/我们/buun | 2.6965e-04 | 差 **0.06%**（可忽略） |
| **4bit LM（TheTom & buun）** | **7.4230e-05** | 理论最优 |
| **4bit atomic/我们** | **1.5765e-04** | **差 2.12×** |

## 3. 逐项判读

### turbo2 —— 算法完全相同，输赢只在字节
三方**同一个码本、同一条流水线**（128 组归一化 → FWHT → 4 级标量 → 组范数校正），实测 rel-err
**逐位相同（0.33013）**。差别只在块大小：

- TheTom/我们：`QK_TURBO2 = 128` ⇒ 一个块就是整个旋转组，**一把范数** ⇒ 34 B。
- buun：保留 `QK_TURBO2 = 32` + `QK_TURBO2_GROUP = 128` ⇒ 4 个子块，而编码器把
  **同一个组范数写进 4 个子块**（`for (b) dst[b].norm = corrected_norm;`）⇒ 冗余 3×2 B = 6 B/128
  = **0.375 bpv 纯浪费**。修它只需让 4 个子块共享一个范数。

### turbo3 —— 码本差异可忽略，字节差异一样
TheTom 的 3-bit 表是精确 Lloyd-Max，理论上比 atomic/我们/buun 的表好 **0.06%**（实测 0.2%），
**实际可忽略**。字节上同样是 buun 多 0.375 bpv。
buun 另有额外杠杆（默认关）：`d_turbo_centroids_3bit` 有 3 套表，其中一套是**非对称**经验拟合
（`-0.186344…+0.188239`，K/V 可分开），加 `TURBO3_SKEW_EXP` 指数偏移。**这是 buun 唯一在
turbo3 上可能反超的设计**，但默认不开、且需要标定。

### turbo4 —— 唯一实质质量差距在我们这边
TheTom 与 buun **各自独立**把码本换成 N(0,1/√128) 的 Lloyd-Max（最外 2.733σ，末位
0.241529 vs 0.241556 是重算舍入差），我们仍是 1.968σ 的次优表 ⇒ 理论 MSE **2.12×**、
现实数据 RMSE **1.30×**。加上我们多存 2 B 死字段（4.25 vs 4.125 bpv）。
buun 在此之上还有 **mean-sub**（实测再降 24.3% RMSE）与 **alpha**，所以 turbo4 的端到端天花板
是 **buun 最高**；代价是离线标定资产（618 KB 表，只覆盖 `qwen35`/`qwen35moe`）+ 它的
CPU/CUDA 不同域 bug。

## 4. 行动建议

1. **turbo4 码本 + 块结构**取 TheTom（2 处 CPU 表 + 2 处设备表 + 2 处中点表 + 66 B 块）——
   这是唯一能实质提升我们 turbo4 质量的改动，预期 27B/16K KLD **0.0769 → ≈0.04**。
2. **turbo3 码本**顺手取 TheTom 的精确 LM（收益 0.06%，但零成本、零风险）。
3. **turbo2 无需改**（三方同表），但不要把 buun 的 QK=32 冗余设计搬进来。
4. 若要追 turbo4 极限质量，再评估 buun 的 mean-sub（需要为 Qwen3.8-27B 补一张离线标定表）。
5. **不要取** buun 的 CPU/CUDA 旋转域不一致与 turbo2/3 CPU 桩。

## 5. 局限

- §2 测的是编解码器 RMSE（受控对照：同一变换/同一数据），不是 KLD 面板；turbo4 的
  「0.0769 → 0.04」换算依赖 KLD ∝ 噪声² 的近似，需 V100 实测确认。
- turbo3 的 buun 非对称码本 / skew 未实测（默认关，且需要标定数据）。
- 未核对 TheTom 的 `TURBOQUANT_UPSTREAM_MERGE.md` 与 turbo8/TCQ 面。
