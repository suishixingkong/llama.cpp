# Audit — `llamacpp-v100` → `llama.cpp` (Volta) merge

Independent review of whether the port of the `llamacpp-v100` fork's CUDA/Volta
pieces into the `turbo-kvstream-merge` integration branch is complete and correct.

- **Fork under review**: `anyei/llamacpp-v100` @ `15e91bbbb` (merge base `6d5a910c5`)
- **Landing commit**: `1c5965914` "ggml-cuda: port the Tesla V100 (Volta) specific pieces"
  - doc commit `bcc6ddd6f` (MERGE_V100.md)
- **Method**: 3-way diff — fork `base..tip`, fork `tip` vs this tree's landing
  commit, this tree's landing `diffstat`, plus symbol-existence probes in this tree.

## Verdict

**Functionally complete and correct. No code bugs found.**
The port (4 files, +249 / -5) matches the fork for every V100-relevant code path.
All "deliberately excluded" items were re-checked on their merits and are safe for
V100. One documentation inaccuracy (line-count wording) was present and has since
been corrected in `MERGE_V100.md`.

## 1. Ported code — file-by-file

### 1.1 `allreduce.cu` / `allreduce.cuh` — byte-identical ✓

- `diff` of fork tip `15e91bbbb` vs this tree's landing commit on both files: **empty**.
- Fork blobs (CUDA impl + HIP/MUSA no-op stubs) apply verbatim.
- Dependencies present in this tree:
  - `GGML_TENSOR_FLAG_COMPUTE` (`ggml.h:675`)
  - `ggml_is_contiguously_allocated` (`ggml.h:811`)
  - `ggml_backend_cuda_context::stream()` (`common.cuh:1546`)
  - `ggml_cuda_highest_compiled_arch` / `GGML_CUDA_CC_VOLTA` (used by mmvq host branch)
  → compiles clean (matches documented sm_70: 3 objects, 0 errors).

### 1.2 `mmvq.cu` — VOLTA table functionally complete ✓ (one doc wording fixed)

Landed correctly:
- `enum MMVQ_PARAMETERS_VOLTA`
- `get_device_table_id()` device + host branches select VOLTA **only in the sm70
  window** (`VOLTA … < TURING`), on both the device `__CUDA_ARCH__` and the host
  `ggml_cuda_highest_compiled_arch(cc)` side.
- `calc_nwarps()` batch-1 K-quant (`ncols_dst == 1`) → `nwarps = 2`.
- `calc_rows_per_block()` guard carries `MMVQ_PARAMETERS_GB10` (required because
  upstream added GB10 *after* the fork's base).

Difference found vs fork:
- Fork `calc_nwarps` has an extra tail `switch (ncols_dst) { case 2..4: return 4;
  case 5..8: return 2; default: return 1; }` for `ncols_dst >= 2`; this tree does
  **not** carry it.
- **This tail is a redundant no-op.** This tree's `GENERIC`/`TURING` already return
  exactly `2–4 → 4 / 5–8 → 2 / default → 1` for those `ncols_dst` values, so the
  VOLTA case falling through to them yields identical `nwarps`. The fork's own
  `TASKS.md` (#18) records the `ncols_dst` 2–4 tuning as "investigated but NOT
  shipped / reverted" — confirming it is not a behavioural change.
- Net effect: the previously quoted "mmvq.cu 31 vs 31" line count was loose
  (fork hunk ≈ 44 added vs ≈ 39 in this tree); the gap *is* exactly that no-op
  tail. `MERGE_V100.md` wording corrected.

### 1.3 `ggml-cuda.cu` — P2P wiring correct ✓

- 4th init path `GGML_CUDA_ALLREDUCE=p2p` present.
- `init_p2p()` order matches fork **line-for-line**:
  `init_nccl()` (sets `try_allreduce` = nccl/internal) → `ar_p2p_init()` → on
  success `fallback_allreduce = try_allreduce` and `try_allreduce = p2p`.
- `comm_ctx->backends` is `assign`-filled in `comm_init`; `try_allreduce_p2p`
  reads `backends.data()` safely.
- `ar_p2p` freed in destructor via `ggml_cuda_ar_p2p_free`.
- P2P algorithm (two alternating scratch slots + one event per `(device, slot)`)
  orders correctly across calls; no race (2-slot separation is sufficient).

## 2. Excluded items — re-verified on the merits

| Excluded item | This-tree reality | Safe for V100? |
|---|---|---|
| CUDA-graph arch gate removal | `ggml-cuda.cu:5748` already `cc < GGML_CUDA_CC_VOLTA` | ✓ V100 keeps graphs |
| FA MMA wmma Volta fix (`should_use_wmma_fattn` / `fattn-wmma-f16.cuh`) | file deleted upstream; `should_use_wmma_fattn` absent in this tree | ✓ no WMMA dispatch hits Volta, no `NO_DEVICE_CODE` |
| FA MMA shared-mem fallback | fork measured: on V100 baseline / `FA_NO_MMA` / injected-fail all identical ~260 t/s → fallback never fires; target = GTX 16xx | ✓ correctly excluded |
| meta-backend rewrite (+1607/-141) | upstream already drives AllReduce via `ggml_backend_comm_init` / `_allreduce_tensor`; fork plugs in as one more variant | ✓ scoped correctly |
| RPC 4.2–4.10 / SSD streaming / fleet UI / docs / compose / Dockerfiles | excluded by scope; fork's `common/arg.cpp` and `ggml/src/CMakeLists.txt` contain **no** AllReduce/p2p/Volta change needing a port | ✓ nothing missed |

## 3. Documentation inaccuracy (fixed)

`MERGE_V100.md` "Completeness of the port" previously stated `mmvq.cu` 31 vs 31
with "a single explainable delta". Corrected to name the two real deltas:
(1) the GB10 guard line that genuinely must exist, and (2) the `ncols_dst >= 2`
tail switch which is an equivalent no-op. See edit in `1c5965914`'s follow-up.

## 4. Runtime-unverified (environment limit, not a defect)

This environment has **no NVIDIA device**. The following are compile-verified only
and should be confirmed on a 2× V100 box:

- P2P AllReduce numerical correctness.
- P2P → NCCL/internal fallback transition for bandwidth-bound / unsupported reductions.
- Interaction with the meta backend's butterfly path.
- Re-measure the fork's perf claims (P2P `+1.2%` tg; MMVQ Volta `+1.8%` tg) on
  actual hardware.

Suggested runtime check:

```sh
GGML_CUDA_ALLREDUCE=p2p ./llama-cli -m <model> -ngl 99 -sm row   # or -sm tensor
# confirm: small reductions take the p2p path, large ones fall back,
# output text matches a single-GPU run.
```
