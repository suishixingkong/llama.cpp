# Tesla V100 (Volta) CUDA — integration notes

Port of the CUDA/Volta-specific pieces of the `llamacpp-v100` fork into this
tree, on top of the existing turboquant + adaptive-KV-streaming integration.

| source | branch in this repo | upstream base | what was taken |
|---|---|---|---|
| `llamacpp-v100` @ `15e91bbbb` (local `C:/Users/confu/llama/llamacpp-v100`) | `fork-master` | `6d5a910c5` (881 commits behind `master`) | the CUDA/Volta items only — 4 files |

Result: **4 code files, +249 / -5**, plus these notes, landed on the integration
branch `turbo-kvstream-merge`.

```
bfd6c4416  turbo-kvstream-merge   (turboquant KV cache + adaptive KV streaming)
  └─ 1c5965914  ggml-cuda: port the Tesla V100 (Volta) specific pieces
       └─ bcc6ddd6f  docs: MERGE_V100.md
            └─ 3de3e117a  docs: exact diffstat for the excluded meta rewrite
```

The work was done on a side branch `v100-cuda-port` (identical tree at
`3de3e117a`) and fast-forwarded into `turbo-kvstream-merge`, which is 3 commits
ahead of `origin/master` and **not pushed**.

The fork itself is a distribution: 110 files between the merge base and its tip.
Despite the README's "V100 tuning" framing, only a small part of that is
Volta-specific, so the port was scoped by reviewing every hunk rather than by
taking the fork's file list.

## What was taken

1. **One-shot P2P NVLink AllReduce for 2-GPU tensor mode** —
   `ggml-cuda/allreduce.{cu,cuh}` taken byte-identical from the fork, plus the
   wiring in `ggml-cuda.cu`. Each device copies its partial into one of two
   alternating scratch slots, then a single kernel adds the peer's slot (read
   over NVLink/PCIe P2P) into its own tensor; cross-device ordering uses one
   event per (device, slot). Falls back by construction for anything it cannot
   handle (non-F32, non-contiguous, larger than the cap).

2. **MMVQ sm70 parameter table** — `mmvq.cu`. Volta was previously served by
   the untuned `MMVQ_PARAMETERS_GENERIC` path; it now has its own table. The
   only behavioural change is batch-1 (`ncols_dst == 1`) K-quants
   (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K) going from `nwarps = 4` to `nwarps = 2`; every
   other path keeps the generic values.

## What was deliberately *not* ported

The fork carries 110 changed files. Excluded, with the reason:

- **The fork's own `ggml-backend-meta.cpp` rewrite** (`+1607/-141`). This is the
  single biggest item and the reason the fork's README reads as a V100 story.
  The fork's base already had `-sm tensor` (`LLAMA_SPLIT_MODE_TENSOR = 3`,
  marked EXPERIMENTAL in `arg.cpp`) and a meta backend that already used split
  states — but upstream then rewrote the same file in parallel
  (`bf0a29cc1` "Deepseek 4: `-sm tensor`", `d3371929b` "\[Tensor parallel\] fix
  meta tensor split state propagation", `5d5cb4c3a` "ggml-meta: propagate
  buffer usage": `+271/-31` on that file since the merge base). So the two
  sides independently built out the same mechanism, and upstream's version
  already drives AllReduce through `ggml_backend_comm_init` /
  `ggml_backend_comm_allreduce_tensor`. The fork's P2P therefore plugs into
  upstream's *existing* mechanism as one more variant instead of requiring its
  parallel implementation — i.e. the fork's calls are rewritten into upstream's
  scheme rather than restoring the fork's older mechanism. The rest of the
  fork's meta work (split-state machinery, mirrored MLA attention,
  delayed-AllReduce matcher, `GGML_META_*` diagnostics, expert-parallel
  placement) is entangled with its distributed stack and is not in this tree.
- **Distributed inference / RPC 4.2 … 4.10** — `ggml/src/ggml-rpc/*`,
  `tools/rpc/rpc-server.cpp`, `tools/server/server-*`. RDMA transport,
  split-state upload, worker weight hash cache, TP islands, fleet
  orchestration, `--rpc-auto-weight`. Large, protocol-versioned, and the
  fork's own README notes the RPC protocol has **no authentication or TLS**
  and that several distributed claims remain hardware-gated.
- **SSD streaming** — `ggml/src/ggml-ssd-stream.cpp` (1415 new lines: a whole
  `ggml_backend_buffer_type`), `ggml/include/ggml-ssd-stream.h`,
  `common/ssd-streaming.*`, plus the model-loader/`arg.cpp` hooks and
  `run-ssd-*.sh`. Linux-only (`O_DIRECT`), self-labelled beta in the README.
- **Backend-agnostic perf work** — multi-shape decode graph cache
  (`src/llama-context`), speculative draft padding (`common/speculative`),
  server prompt-batch defragmentation and prompt-cache checkpoint pruning
  (`tools/server`), the adaptive draft cap, the env-gated diagnostics, and the
  robustness fixes (null-pointer crash on failed context/lora init, lora-path
  double-free). These are explicitly *not* V100-specific in the fork's own
  README, so they belong in their own change, not this one.
- **Fleet web UI** — `tools/ui` (17 files: fleet screens, store, service,
  types, plus `static/loading.html`).
- **Docs, compose files, Dockerfiles, run scripts** (~40 files) —
  `docs/perf-tuning-v100.md` & co., `docker-compose.*.yml`,
  `.devops/*.Dockerfile`, `TASKS.md`, `REBUILD-IMAGE.md`, `run-*.sh`.

Three further CUDA-side items were reviewed hunk by hunk and excluded **on the
merits**, not for scope reasons:

- **FA MMA shared-memory-failure fallback** (`fattn-common.cuh`,
  `fattn-mma-f16.cuh`, `fattn.cu`, `vendors/hip.h`). It does not affect V100.
  The fork measured, on V100 with Qwen3-0.6B and a 300-token prefill, that
  baseline / `GGML_CUDA_FA_NO_MMA` / injected failure all exit 0 and generate
  *identical* text at the same speed (~260 t/s) — i.e. the fallback never fires
  there. Its target device is the GTX 16xx (TU116/117, cc 7.5 *without* tensor
  cores). The one sub-fix that genuinely touched Volta was
  `ggml_cuda_should_use_wmma_fattn` returning true for NVIDIA Volta
  (`NO_DEVICE_CODE`, "no device code compatible with CUDA arch 700"), and that
  was only reachable *because* the feature introduced an MMA-disable knob;
  upstream has since **deleted `fattn-wmma-f16.cuh` entirely**, so the latent
  crash does not exist in this tree. Upstream deliberately has no such fix
  (discussion #5329: tensor-core availability is not queryable via the CUDA
  API), and master's `ggml_cuda_flash_attn_ext_mma_f16_case` now has five
  setattr sites across sparse-kernel variants, where the fork's patch assumed
  two.
- **CUDA-graph arch-gate removal** in `ggml_cuda_graph_set_enabled()`. The
  deleted `graph->disable_due_to_gpu_arch = true;` sits inside the
  `cc < GGML_CUDA_CC_VOLTA` branch, and V100 is cc 700, so it never enters that
  branch — **zero effect on V100**. Its actual effect is to enable CUDA graphs
  below Volta. Upstream deliberately drew that line in PR #25749 ("Enable CUDA
  graphs on volta+turing", 2026-07-16), which moved the threshold from
  `GGML_CUDA_CC_AMPERE` down to `GGML_CUDA_CC_VOLTA` and kept everything below
  off — and note that commit is an **ancestor of the fork's merge base**, so the
  fork (and this tree) already inherited Volta CUDA graphs from upstream; the
  fork's hunk overrides exactly the boundary that same PR drew. The fork
  documents a
  `GGML_CUDA_FORCE_GRAPHS` knob for this that **is not implemented anywhere in
  its source tree** (it appears only in `README.md`, `TASKS.md` and
  `docs/`), and its own note records "Tested: no gain".
- **`GGML_CUDA_ERROR_CONTAIN` worker error containment and
  `GGML_CUDA_INJECT_COMPUTE_FAIL` fault injection** — both exist to serve the
  excluded RPC worker path, and the SSD GPU-landing hook
  (`ggml_ssd_stream_*` in `device_offload_op`) to serve the excluded SSD tier.

## Integration decisions worth knowing

- **P2P is a fourth variant in upstream's chain, not a replacement.** The
  init chain stays `nccl -> internal -> none`; `p2p` runs
  `comm_init_nccl()` first and only then installs itself, keeping the previous
  `try_allreduce` in a new `fallback_allreduce` member so large
  (bandwidth-bound) and unsupported reductions still go to NCCL/internal.
  `GGML_CUDA_ALLREDUCE=p2p` selects it; anything else is unchanged.
- **`MMVQ_PARAMETERS_VOLTA` is a local enum inside `mmvq.cu`** and is not part
  of any ABI, GGUF or RPC surface — so unlike a `GGML_TYPE_*` addition there is
  no `GGML_TYPE_COUNT` / `RPC_PROTO_PATCH_VERSION` / `gguf-py` bookkeeping.
- **The fork's `mmvq.cu` hunk could not be applied verbatim.** Upstream
  rewrote `calc_nwarps()` since the merge base (`+307/-45`: added `small_k` and
  `halve_iters` parameters, added `MMVQ_PARAMETERS_GB10`). The Volta case was
  re-expressed in the current scheme, and its `calc_rows_per_block()` guard had
  to gain `MMVQ_PARAMETERS_GB10` as well, which did not exist at the fork's
  base.
- Two comments (`ggml_backend_cuda_comm_context`'s field doc and the
  "Top-level init" header) were updated to mention the new fourth path, since
  the change targets exactly the mechanism they describe. These are the only
  lines added that are not in the fork's diff.
- `allreduce.{cu,cuh}` needed no adaptation: neither upstream nor the
  integration branch had touched them since the merge base, so the fork's blobs
  apply as-is (verified byte-identical).

## Verification performed

Build: MSVC 14.40 + Ninja + CMake, CUDA 12.4 with `-allow-unsupported-compiler`,
static libs. Logs under `D:\llama-build\`.

- **CPU**: `335/335` targets, 0 errors, 14 warnings (all pre-existing
  conversion warnings elsewhere in the tree). Note this is a
  tree-consistency check only — the change is entirely under
  `ggml/src/ggml-cuda/`, so no CPU target compiles any of it.
- **CUDA, `sm_70` (the actual V100 arch)**: `mmvq.cu`, `allreduce.cu`,
  `ggml-cuda.cu` → 3/3 objects, 0 errors.
- **CUDA, `sm_80` (control)**: same three objects, 0 errors.
- NCCL was **not** found on this machine, so the compiled chain is the
  no-NCCL one (`comm_init_nccl` → warn → `comm_init_internal` → p2p chained on
  top). The NCCL-enabled variant of the assignment is not compiled here.
- **Mechanical proof that the Volta table is actually selected** (no GPU
  needed). `nwarps` is a `__launch_bounds__` parameter, so it is visible in
  PTX as `.maxntid`: nwarps=2 → 64. Compiling `mmvq.cu` to PTX twice per arch
  from the exact CMake command, before and after the change:

  | build | `.maxntid 64` (nwarps 2) | `.maxntid 128` (nwarps 4) | kernels |
  |---|---:|---:|---:|
  | pre-change, `sm_70` | 92 | 179 | 317 |
  | **ported, `sm_70`** | **118** | **153** | 317 |
  | pre-change, `sm_80` | 92 | 179 | 317 |
  | **ported, `sm_80`** | **92** | **179** | 317 |

  `sm_80` is **identical** before and after → no effect on non-Volta devices.
  `sm_70` moves exactly **26** kernels from nwarps 4 to nwarps 2, and the
  demangled signatures of those 26 are precisely the `ncols_dst == 1`
  instantiations of `ggml_type` 10/11/12/13/14 — GGML_TYPE_Q2_K/Q3_K/Q4_K/Q5_K/Q6_K
  (4+4+6+6+6). That is exactly the table's whitelist and nothing else.
- **Completeness of the port** was checked by set difference against the fork's
  own diff (normalised added lines, per the usual method): `mmvq.cu` 31 vs 31
  with a single explainable delta (the `calc_rows_per_block` guard line, which
  must also carry GB10); `ggml-cuda.cu`'s residuals are exactly the three
  excluded features plus the five comment lines. Every P2P line matches
  bidirectionally.

## Known limitations

- **Nothing was executed.** There is no NVIDIA device in this environment, so
  the merge is compile-verified only. Specifically unverified at runtime: the
  P2P AllReduce's numerical result, the P2P→NCCL/internal fallback transition,
  and the interaction with the meta backend's own butterfly path.
- The performance figures are the fork's, not re-measured here: `tg 48.8 →
  49.4 (+1.2%), ppl identical` for P2P (fork commit `d2b706d1b`) and
  `+1.8% tg, perplexity-identical` for the MMVQ Volta table.
- `GGML_CUDA_ALLREDUCE=p2p` needs exactly 2 CUDA devices with bidirectional
  peer access; otherwise init returns null, it warns, and the existing chain is
  left untouched.
- The P2P path handles contiguous `F32` tensors up to
  `GGML_CUDA_AR_P2P_MAX_BYTES` (default 4 MiB) only; everything else defers to
  the fallback.
- HIP/MUSA get no-op stubs from the fork; not compiled here (no ROCm toolkit),
  and `vendors/hip.h` was intentionally left alone.
- The NCCL-enabled build of the new chain was not compiled (no NCCL on
  Windows), though the added code there is only a function-pointer assignment.
- Consequence for the un-ported items: this tree's tensor-split support is
  upstream's own `-sm tensor` path, with the ported P2P AllReduce plugged into
  it — the fork's extra meta capabilities (expert-parallel placement, mirrored
  MLA attention, delayed AllReduce, `GGML_META_*` diagnostics) are absent, and
  so are SSD streaming, the RPC fleet, and the fleet web UI. Volta CUDA graphs
  remain available, as upstream ships them; only the fork's *below*-Volta gate
  override is left out.
