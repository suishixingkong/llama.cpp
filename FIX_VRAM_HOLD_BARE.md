# 显存不足时的「重复 alloc」来自 `hold_bare()` 的固定 8 s 空等

> 症状：`llama-server` 在显存不够指定上下文时，**每隔约 2.16 s 重复失败一次同一个分配**，
> 连续 5 次后退出。`v1.0` 不会这样。
> 结论：这是 fork 独有文件 `src/llama-vram-demand.cpp` 的 `hold_bare()`。
> 它在**没有任何协商通道**的情况下按固定节拍重试，除了给失败报告加 ~8.7 s 延迟之外没有作用。
> 已改回**单次尝试**（与 `v1.0` 一致）。

---

## 1 判据：这是「原地重试」，不是「降档重试」

用户日志里最关键的一点：**每次要的字节数完全相同**（`2510290944` = 2394.00 MiB），
且间隔**均匀**（2.158 / 2.163 / 2.168 / 2.182 s）。

- 若是「降档后重试」，尺寸应当变化。
- 若是「抖动退避」，间隔不应这么整齐。

⇒ 这是一个**固定节拍、同参数**的重试循环。

## 2 定位

`src/llama-vram-demand.cpp`：

```cpp
ggml_backend_buffer_t llama_vram_hold_alloc_ctx_tensors(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);   // 第 1 次
    ...
    if (buf == nullptr) {
        size_t need = ...;
        while (buf == nullptr && llama_vram_demand_hold(dev, need)) {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);                  // 重试同一个
        }
    }
```

`llama_vram_demand_hold()` 在**未武装**（unarmed）时转发到 `hold_bare()`：

| 常量 | 值 | 出处 |
|---|---|---|
| `LLAMA_VRAM_LEDGER_SHORT_MS` | **8000** | `llama-vram-ledger.h:27` |
| `LLAMA_VRAM_LEDGER_BEAT_MS` | **2000** | `llama-vram-ledger.h:28` |

```cpp
bool hold_bare() {
    if (bare_deadline == 0) bare_deadline = now + 8000ms;   // 首次调用开窗
    if (now >= bare_deadline) { bare_deadline = 0; return false; }
    sleep_for(2000ms);
    return true;
}
```

### 次数复算（⚠️ 初次分配在循环之外）

8 s 窗口 ÷ 2 s 节拍 = **4 次 sleep 返回 true** ⇒ 循环内 4 次重试，
加上进入循环前的那 1 次 = **共 5 次尝试**，节拍 2000 ms + 约 160 ms 的失败/日志开销 = **2.16 s**。

与日志逐项对上：

| 日志时刻 | 第几次 | 距上次 |
|---|---|---|
| 10.805 | 1（循环外） | — |
| 12.963 | 2 | 2.158 |
| 15.126 | 3 | 2.163 |
| 17.294 | 4 | 2.168 |
| 19.476 | 5 | 2.182 |
| 19.547 | 抛 `failed to allocate buffer for rs cache` | 0.071 |

总延迟 **8.67 s ≈ 4 × 2.16**，之后才报出那个「本来立刻就能报」的错误。

### 为什么这台机器必然走这里

`src/llama-vram-ledger.cpp:656`：

```cpp
#else // !__linux__ — substrate disabled: armed() = false, every caller takes the clean-fail path
bool llama_vram_ledger_armed() { return false; }
```

用户日志里的模型路径是 `C:\Users\yj\models\...` ⇒ **Windows** ⇒ `armed()` 恒 false ⇒ `hold_bare()` 恒生效。
（Linux 上只有在未配置 ledger 目录时才走这条。）

三条调用点都会经过它：`llama-kv-cache.cpp:1976`（注意力 KV）、
`llama-memory-recurrent.cpp:721`（RS cache）、`llama-model.cpp:1807`。

## 3 这个等待有没有必要 —— 没有，而且作者自己也是这么写的

⚠️ **先把定性说准**：这**不是**一个 bug 式的重试。`llama-vram-demand.h` 开头的设计文档写明这是
「loader patience / VBR 多租户协议」的**有意行为**，并强调它 *"never called on decode paths —
the runtime fast-fail wall stays intact by construction"*（只在模型加载 / 上下文初始化期，不在
decode 路径上）。所以问题不是「为什么要重试」，而是**无通道时的那个分支退化了**。

`llama-vram-demand.cpp` 原注释（**改动前**）：

> no substrate: honor SHORT patience with plain backoff — **the grace for a resident that cannot
> be seen is pointless**, but a clean bounded wait keeps behavior uniform (design: clean fail +
> SHORT patience)

补一条更硬的论证：`llama_vram_ledger_armed()` 的定义是 `!vrlg_dir().empty()`。
**没有 ledger 目录 ⇒ 没有 claim/marker 通道** ⇒ 系统里**不存在任何**能告诉本进程
「这块显存的空闲量即将变化」的信息源。既然没有信息源，等固定时长后重新发起**同一个**请求，
不可能因为「我们本来会知道的原因」而成功。它唯一的效果是：

- 把 1 条错误变成 5 条同样的错误（日志像重试风暴）；
- 把失败报告推迟 8.7 s；
- 在 `llama-server` 上看起来像「反复尝试分配」。

## 4 与 `v1.0` 的对照

| 树 | `src/llama-vram-demand.cpp` |
|---|---|
| `v1.0` | **不存在** |
| `llama` / `master`（上游） | **不存在** |
| `buun-port` | 存在，由 `1830cb5f9`（*stage 2.9 slice 5a: model/graph VBR wiring, vram demand + ledger subsystem*）引入 |

⇒ 所以 `v1.0` 不重试是**因为它根本没有这套东西**，不是它显式选择了单次。这次改动就是把
fork 的默认行为对齐到 `v1.0` 的净效果。

## 5 改动

`src/llama-vram-demand.cpp` `hold_bare()`：**默认直接返回 `false`**（单次尝试），
原 8 s 有界等待改为 opt-in：

```
LLAMA_VRAM_HOLD_BARE_WAIT=1     # 恢复原来的「8 s / 2 s 节拍」行为
```

同时补一行 INFO，让这次行为变化**在日志里可见**（原来唯一痕迹就是那串重复的 cudaMalloc 失败，
读起来像重试风暴而不是一个决定）：

```
vram-demand: no VRAM ledger substrate - failing immediately instead of waiting 8000 ms
for a resident that cannot be observed (set LLAMA_VRAM_HOLD_BARE_WAIT=1 to restore the bounded wait)
```

⇒ 改动后一次失败的特征是「**1 条 E + 1 条这条 INFO**」，改动前是「5 条 E」。
两个方向都可判读，不用猜。

### 为什么不是一刀切删掉整个机制

**武装**（armed，Linux + 配了 ledger 目录）的那条路径是**知情**的，必须保留：
它发布 claim、扫描对端 marker 与 claim，`tie_break()` / `dev_ready()` / `jointly_sufficient()`
都有真实的提前判定，失败时打 `vram-demand: … giving up`。那里的等待能观察到捐赠方的释放动作，
是真正的多租户协作。本次只动「无通道」的那一半。

⚠️ 一个**未改动**的同类点，留作观察：武装路径在 **无对端** 时窗口也是 `SHORT_MS`（8 s），
若真实场景是「单租户但配了 ledger 目录」，那里同样会有 8 s 的无谓等待（带 extension cap 时理论上更长）。
它的 `progress_event`/截止条件我没有完整追踪，**不下结论**。

## 6 证据

| 项 | 结果 |
|---|---|
| CPU 共享构建，target `llama` | 见提交信息 `Evidence.` |
| 全量守卫体检 `guard_sweep.py` | 见提交信息 |
| 真机复现（显存不足 → 只失败 1 次） | **未验证**：本机无 CUDA 设备，且需要构造「显存不足」的模型/上下文组合 |

复现判据（用户侧，一条命令即可）：
用一个**故意超过显存**的 `-c` 启动 `llama-server`，看日志里
`allocating … cudaMalloc failed` 出现 **1 次**（改动后）而不是 5 次（改动前）。
加 `LLAMA_VRAM_HOLD_BARE_WAIT=1` 应恢复 5 次 —— 这就是本次改动的反向控制。
