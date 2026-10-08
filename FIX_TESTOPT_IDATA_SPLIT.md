# FIX: `ctest -R test-opt` fails — `test_idata_split(high_level=yes, ..., results_forward)`

Status: **fixed by a locally scoped workaround; the underlying defect is NOT in this fork's code.**
Scope: `ggml/src/CMakeLists.txt` (one guarded block). No source logic touched.

---

## 1 Symptom

```
39:   test_idata_split(high_level=no,  epoch=1..4, subtest=results_forward, optimizer=adamw): OK
39:   test_idata_split(high_level=yes, epoch=1, subtest=results_forward, optimizer=adamw): FAIL
39:   test_idata_split(high_level=yes, epoch=2, subtest=results_forward, optimizer=adamw): FAIL
39:   test_idata_split(high_level=yes, epoch=3, subtest=results_forward, optimizer=adamw): FAIL
39:   test_idata_split(high_level=yes, epoch=4, subtest=results_forward, optimizer=adamw): FAIL
The following tests FAILED:
    39 - test-opt (Failed)
```

Everything else in `test-opt` passes. Locally reproduced (`1/2 backend*optimizer passed`,
`Backend CPU sgd: OK`, `Backend CPU adamw: FAIL`), so it is deterministic per build.

`test_idata_split(..., high_level=yes, ...)` calls the **high-level API** exactly once per epoch:

```cpp
ggml_opt_epoch(cd.opt_ctx, cd.dataset_unsupervised, cd.result, cd.result2, idata_split, nullptr, nullptr);
```

`ndata = 6`, `idata_split = ndata*2/3 = 4`, so the epoch is a train half `[0,4)` and an
**eval half `[4,6)`**. `results_backward` checks `cd.result` (train) and passes;
`results_forward` checks `cd.result2` (**eval**) and fails.

## 2 Root cause: the eval loop of `ggml_opt_epoch()` is never entered

`ggml/src/ggml-opt.cpp`, `ggml_opt_epoch()`:

```cpp
int64_t ibatch = 0;
for (; ibatch < ibatch_split; ++ibatch) { ...train... }   // 0 -> 4
t_loop_start = ggml_time_us();
for (; ibatch < nbatches; ++ibatch)     { ...eval ... }   // 4 -> 6  (should run twice)
```

Instrumented run (`[PROBE]`/`[CANARY]` fprintf probes, removed afterwards):

```
[CANARY] entry ibatch=0 nbatches=6 ibatch_split=4 ndata=6 ndata_batch=1
[CANARY] it=0 after alloc:   nbatches=6 ibatch_split=4 ndata_batch=1
...
[CANARY] it=3 after eval:    nbatches=6 ibatch_split=4 ndata_batch=1
[CANARY] pre-eval-loop ibatch=4 nbatches=6          <-- 4 < 6, must iterate
[CANARY] entry ...            (next ggml_opt_epoch call)
```

* the eval-loop body print never appears, and `ibatch` is still `4` after the loop;
* across a whole run: **1020** `ggml_opt_epoch()` calls, `pre-eval-loop` printed 1020 times,
  but the loop body ran only **24** times — all 24 from the 4 calls whose `ibatch_split == 0`
  (train loop empty). Every call with `ibatch_split == 4` iterated **zero** times;
* the enclosing `result2` consequently stays empty. The observed
  `ndata=0 loss=0 loss_unc=nan` is exactly `ggml_opt_result_loss()`'s `nbatches == 0` branch:

```
[PROBE results_forward high_level=0 epoch=1] ndata=2 (exp 2) loss=7   ...  <- low-level path, correct
[PROBE results_forward high_level=1 epoch=1] ndata=0 (exp 2) loss=0 loss_unc=nan acc=-nan(ind)
```

So this is **control flow**, not numeric drift: the loop condition `4 < 6` evaluates false.

### 2.1 Layout sensitivity (the decisive tell)

The outcome flips when *any* code is added near it:

| build | change | eval loop | test |
|---|---|---|---|
| build3 | `fprintf` as first statement of the eval-loop body | **ran** (iter 4, 5) | **OK** |
| canary | the same body print + canaries in the train loop | **not entered** | FAIL |

A defect that disappears when a `printf` is inserted is a codegen defect (or latent UB),
not a logic error.

## 3 Evidence that this is NOT this fork's code

| # | Check | Result |
|---|---|---|
| 1 | `git diff llama -- tests/test-opt.cpp` | **empty** (byte-identical to upstream `llama`) |
| 2 | `git diff llama -- ggml/src/ggml-opt.cpp` | **empty** |
| 3 | Fresh build dir (configure 15:42, all objects 15:42–15:47) | still FAILs ⇒ not a stale-object artifact |
| 4 | `CMAKE_BUILD_TYPE=Debug` (`/Od`, `/RTC1`) on the same tree | **`2/2 backend*optimizer passed` / `OK`** |
| 5 | Revert `ggml/ src/ common/ tests/ tools/ examples/ include/ CMakeLists.txt` to upstream `llama`, rebuild Release | `RC_TESTOPT=0`, 0 errors — **still FAIL** |
| 6 | `/Od` on `ggml-opt.cpp` **alone** (Release elsewhere) | **`2/2 backend*optimizer passed` / `OK`** |

Row 5 is the decisive one: with **every** file the fork touched replaced by the upstream
`llama` version (only unreferenced fork-only files left on disk), a Release build still fails
identically. So the defect is upstream's, exposed by this toolchain.

### 3.1 Single-variable reverts that do NOT fix it

| Reverted to upstream | Result |
|---|---|
| `ggml-backend.cpp`'s relaxed `ggml_gallocr_reserve_n` (the fork discards its return value) | still FAIL |
| `ggml/src/ggml-cpu/ops.cpp` + `ggml/src/ggml-cpu/ggml-cpu.c` | still FAIL |

## 4 Toolchain

MSVC 19.40.33811.0 (VS 2022 17.10, `VC/Tools/MSVC/14.40.33807`), x64, Ninja,
`CMAKE_BUILD_TYPE=Release` (`/O2 /Ob2 /DNDEBUG`), `GGML_CUDA=OFF`.

## 5 Fix

`ggml/src/CMakeLists.txt`, one guarded block:

```cmake
if (MSVC)
    set_source_files_properties(ggml-opt.cpp PROPERTIES COMPILE_OPTIONS "/Od")
endif()
```

`ggml_opt` is the **training / optimiser** API. It is not on the inference path, so giving up
`/O2` for this single translation unit has no runtime cost for this project. The block carries
the full diagnosis inline and is meant to be deleted once the codegen defect is fixed upstream.

## 6 Verify

```
cmake --build D:/llama-build/cpu --target test-opt -j 8
D:/llama-build/cpu/bin/test-opt.exe        # expect "2/2 backend*optimizer passed" then "OK"
ctest -R test-opt --output-on-failure      # in the user's own build dir
```

## 7 Open items

* The exact UB/optimiser trigger inside `ggml_opt_epoch()` was **not** identified. `/Od` is a
  workaround, not a root-cause fix; if a future MSVC or upstream change makes it fail again,
  the same probe recipe (fprintf canaries around the eval loop) localises it in minutes.
* Worth reporting upstream: `test-opt` + MSVC 19.40 + Release + `ggml_opt_epoch` with a partial
  `idata_split`. Nothing in the failing path is fork-specific (rows 1–5 above).

## 8 Measurement scripts / evidence (in `D:\llama-build\`)

| file | what |
|---|---|
| `build_9l_testopt_cpu.bat` | build `test-opt` in the CPU tree (Release) |
| `debug_build_testopt.bat` | same, `CMAKE_BUILD_TYPE=Debug` |
| `upstream_ggml_testopt.bat` | same, Release, used for the all-upstream revert |
| `_canary.log`, `/tmp/can.err` | canary-probe run (1020 calls / 24 eval iterations) |
| `_upstream3.log` | all-upstream Release build log (`RC_TESTOPT=0`) |
| `_expOd.log` | `/Od` on `ggml-opt.cpp` only (`RC_TESTOPT=0`) |
