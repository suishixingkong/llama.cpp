# `--checkpoint-recurrent-prev` (recurrent checkpoint tail replay) — integration notes

Port of the `jusko-llama-volta-qwen3flash` commit that lets a long cached recurrent
context resume from a snapshot taken *just before* the appended suffix, instead of
replaying from the last periodic checkpoint. Unlike the other two ports in this
repo (`MERGE_V100.md`, `MERGE_JUSKO_VOLTA.md`) this one is **entirely backend
agnostic** — no CUDA, no Volta gating — so it is validated by a CPU build and can
be exercised on hardware without a V100.

| source | branch in this repo | upstream base | what was taken |
|---|---|---|---|
| `jusko-llama-volta-qwen3flash` @ `c3d51d71d`, commit `be6567cea` "perf(server): avoid recurrent checkpoint tail replay" | `jusko-recurrent-ckpt` | `465e49b9c`, cherry-picked as a real 3-way merge via the existing `jusko` remote | the mechanism, re-expressed into this tree's checkpoint policy |

Result: **12 files, +285 / −25**, plus these notes, landed on
`turbo-kvstream-merge`.

```
7692b3457  turbo-kvstream-merge   (V100 ports + docs, see the other MERGE_*.md)
  └─ <this>  server: add --checkpoint-recurrent-prev (recurrent checkpoint tail replay)
       └─ docs: MERGE_RECURRENT_CKPT.md
```

## Why this commit is not self-contained

The first attempt was a straight patch application, which reported conflicts in
5 files and looked like a light re-anchor job. That measurement was **wrong**: it
counted patch-application failures, not dependency closure. Reading the source
showed `be6567cea` calls `create_checkpoint(..., is_replay_boundary, ...)`, and
`is_replay_boundary` does not exist anywhere in this tree — nor in the fork's
base `465e49b9c`, nor upstream. It comes from the fork's **own** `0471a9885`
("perf: add V100 optimized runtime stack") — the same commit whose CUDA parts
(the FA compact kernel, GDN 128x4, `--prefill-reuse`) were already ported in
`MERGE_JUSKO_VOLTA.md`, but whose **server-side half was deliberately excluded**.

So the closure is: `be6567cea` + the checkpoint-policy part of `0471a9885`. The
two options were to import that policy wholesale or to re-express only the
increment the feature actually needs. **This port does the latter** — see
"Deliberately not ported" below for exactly what that costs.

## What was taken

Three independent pieces, plus the server wiring:

1. **`rs_rollback_prompt_only`** — a new `llama_context_params` field. During
   token generation a recurrent rollback plane is dead weight, so
   `models/delta-net-base.cpp` now builds the state tensors without it whenever
   `ubatch.n_seq_tokens == 1`. This is a graph-shape change, and it is the reason
   `n_rs_seq` alone is not enough: the feature wants snapshots *during prompt
   processing* and none afterwards.

2. **`LLAMA_STATE_SEQ_FLAGS_RECURRENT_PREV`** — a new `llama_state_seq_flags` bit.
   `llama_memory_recurrent::state_write` can now serialize rollback plane 1
   (the state immediately before the live one) instead of the live state, and
   `state_write_meta` gained a `pos_shift` so the snapshot's positions are
   reported one token earlier. That is what lets the server hand a suffix a
   prefix that ends exactly where the suffix begins.

3. **Device-backed checkpoints** — `common_prompt_checkpoint` gained
   `data_tgt_on_device` and `data_tgt_logical_size`. An `ON_DEVICE` checkpoint
   keeps its tensor payload in context-owned storage and carries only the
   serialization envelope in `data_tgt`, so `size()` must report the logical size
   (that is the eviction/accounting weight) instead of the envelope length;
   `load_tgt` must re-inject the flag when restoring; and such checkpoints cannot
   be cloned into a child sequence or persisted into the prompt cache, because
   they are transient views into one context's memory. Four call sites drop them
   (`server_slot` copy, `server_prompt_cache::alloc`).

4. **Server wiring** — `--checkpoint-recurrent-prev` / `--no-checkpoint-recurrent-prev`
   (server only), the gating, and the two `snapshot_prev` sites: one skips the
   redundant `4 + ubatch` boundary checkpoint when a preceding snapshot will be
   taken, the other creates the snapshot once the prompt reaches `SLOT_STATE_DONE_PROMPT`.

## Usage

```bash
./build/bin/llama-server -m qwen3.8-27b.gguf \
  --ctx-checkpoints 32 --checkpoint-min-step 8192 \
  --checkpoint-recurrent-prev
```

Self-gating, so it is inert unless the workload fits:

- requires a **recurrent or hybrid** model (`llama_n_rs_seq(ctx) > 0`); on a plain
  transformer `n_rs_seq` is clamped to 0 and nothing happens
- requires `--ctx-checkpoints > 0`
- requires **no** speculative decoding (`spec == nullptr`) — the fork's README is
  explicit that this is the non-speculative path, and that draft-MTP has its own
  separate mechanism instead
- snapshots only when more than 64 new tokens are being appended
- `common_context_params_to_llama()` raises `n_rs_seq` to at least 1 when the flag
  is set and speculation did not already ask for it, and only then sets
  `rs_rollback_prompt_only`

## Deliberately *not* ported

| item | why not |
|---|---|
| The fork's **value-based checkpoint eviction** (`0471a9885`): victim scoring by `gap_l * gap_r` for periodic checkpoints and `n_tokens * checkpoint_min_step * 8` for semantic boundaries, both scaled by `1 + 4 * replay_hits`; plus refresh-in-place. | This is a general checkpoint-cache policy, not part of this feature, and this tree's version solves the same "don't duplicate a prefix" problem differently (upstream's `supersede at the same n_tokens` block, kept here). Importing the policy would be a separate change with its own measurements. **Consequence: the fork's headline numbers do not transfer** — see "Known limitations". |
| `replay_hits` | Only consumed by the value-based scoring above. Dead without it. |
| The fork's `ctx_dft_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART` guard on `update_dft` | An optimization (a plain attention draft can be trimmed after restore, so duplicating its KV into every recurrent checkpoint is waste). Upstream stashes it unconditionally; keeping upstream's call is the smaller diff. Both the target and draft `seq_rm_type` members do exist in this tree, so this is a choice, not a constraint. |
| `--pipeline-copies` / `n_pipeline_copies` | Belongs to the mirrored-input tensor-parallel work excluded in `MERGE_V100.md`. |
| The MTP draft-ubatch rework in the same fork commit | It is the substrate for the *separate* `--spec-mtp-defer-prompt` feature, still un-ported. |

## Integration decisions worth recording

1. **`is_replay_boundary` is carried in, but only the semantics this feature needs.**
   The field, the min-step eviction exemption, a preference for evicting ordinary
   checkpoints in the capacity pass (falling back to the original `front()` rule
   when every checkpoint is a boundary), and carrying the flag across a supersede
   at the same `n_tokens`. Without the exemption the snapshot would not survive:
   it sits at `n_tokens == prompt_n - 1`, which is within `checkpoint_min_step`
   of the last periodic checkpoint, so the *unmodified* upstream policy evicts it
   immediately on the very next request. The exemption is load-bearing, not
   cosmetic.

2. **Keeping upstream's eviction structure over the fork's.** The fork replaces
   the whole `create_checkpoint` body (refresh-in-place + value eviction). This
   port keeps upstream's flow and threads the two boundary-aware conditions into
   it. The 3-way merge made this explicit: upstream had also independently added a
   de-duplication block, so restoring the fork's version would have removed it.

3. **`is_replay_boundary` got a default value (`= false`)** so the pre-existing
   4-argument call site is untouched. The fork updates every call site instead;
   with defaults, the only changed call site is the new one.

4. **The `include/llama.h` field is placed next to our existing `prefill_reuse`,
   not appended at the struct end** as the fork does ("Fork extensions are
   appended to preserve the layout of all upstream fields"). This tree's
   `llama_context_params` is not ABI-stable and has a single in-tree consumer, so
   the fork's convention buys nothing here and grouping our two extensions is
   clearer. Recorded because it means the fork's line numbers will not line up for
   the next port.

5. **A pre-existing gap in the `--prefill-reuse` port got fixed as a side effect.**
   `0471a9885` also makes the checkpoint boundary spacing use
   `min(n_ubatch, prefill_reuse)` instead of plain `n_ubatch`. `--prefill-reuse` is
   meant to be used *together with* a raised `-ub` (the fork's own example is
   `-b 1536 -ub 768 --prefill-reuse 768`), so without this the flag silently
   changes prompt segmentation — and therefore the checkpoint and MTP
   trajectories — even though the CUDA math is bit-identical. That hunk is in
   `tools/server/server-context.cpp`, outside the CUDA file set of the earlier
   port, which is why it was missed. It is included here.

6. **The test file was adapted, not copied.** The fork's `be6567cea` adds
   `test_previous_recurrent_snapshot` *where upstream now has `test_rollback`* —
   upstream added `test_rollback` plus a `uint8_t fill` parameter in `9dcf84e5a`
   ("model: support Kimi-K3 recurrent-state rollback", after the fork's base).
   Both are kept: the new test is added, upstream's `test_rollback` and the
   `fill`-parameterised helpers are untouched, and `main()` calls the new test
   before the existing `for (uint8_t fill : { 0, 0x3e })` loop.

## Verification

Performed (Windows, MSVC 14.40, Ninja, CPU-only build — the correct scope here,
since the diff contains no CUDA):

- **CPU full build — clean.** 321 targets, 0 errors, `llama-server.exe` and
  `llama-cli.exe` link. This is real evidence for this change, unlike the CUDA
  ports where a CPU build compiles none of the diff.
- **CLI registration confirmed at runtime**, not by reading the source:
  `llama-server --help` lists `--checkpoint-recurrent-prev, --no-checkpoint-recurrent-prev`
  next to the existing `--ctx-checkpoints` / `--checkpoint-min-step`.
- **`test-recurrent-state-rollback` builds, links and runs.** Against a non-recurrent
  GGUF it exits 0 with `skipping for non-recurrent model`.
- **The feature was then exercised and it works.** With a `qwen35` hybrid model
  (Qwen3.5-0.8B-Q4_K_M) the new test passes and reports
  `max logit drift=0 at token -1, top1=314/314` — the state restored from the
  preceding snapshot is *numerically identical* to the control's full-prompt run.
  That is the correctness claim the fork makes, now reproduced here.
- **Server-side mechanism confirmed in the log**, not inferred: with the flag on, a
  checkpoint is created at `n_tokens = 10801` — i.e. `prompt_n - 1` — carrying
  `replay_boundary = 1`, and the following request restores *that* one instead of the
  `prompt_n - 4` checkpoint that the flag-off run restores.
- **A/B throughput measurement** on the same model (10 turns per side, feature flag
  the only difference):

  | | tokens processed per turn | prompt ms per turn | cache_n |
  |---|---|---:|---:|
  | feature off | **21** (9x) / 22 (1x) | ~426 | 10798 |
  | feature on | **18** (9x) / 19 (1x) | ~382 | 10801 |

  The two distributions do not overlap across ten turns. See "Known limitations" for
  why this is a 3-token saving rather than the fork's percentage.
- **`test_rollback` fails on this tree — pre-existing, not caused by this port.**
  The dirty-context phase reports `logits mismatch at position 6, token 0
  (6.14735 != 7.3468)`. Two independent checks pin it on upstream:
  reverting all 12 files to the pre-port commit, rebuilding and re-running produces
  **byte-identical numbers**; and none of the three forks already merged here
  (`turboquant`, `adaptive-KV-streaming`, the two V100 ports) ever touched
  `src/llama-memory-recurrent.{cpp,h}` or `src/models/delta-net-base.cpp`
  (`git diff master <pre-port> -- …` is empty), nor the test file. So it is an
  upstream dirty-recurrent-state-restore defect on `qwen35`, worth reporting
  separately. Note: the test needs a small `-c` here, because the default sizes the
  KV cache from the model's training length (~3 GiB, allocation fails).
- **Port completeness by set difference** against the fork's own diff (normalised
  added lines): 238 fork lines vs 285 ours; the only fork lines not present
  verbatim are (a) one `cur.update_tgt(... | tgt_extra_flags)` occurrence whose
  counterpart exists at a different indentation in upstream's structure, and
  (b) the `include/llama.h` field declaration, which we re-align and re-place per
  decision 4. Everything else matches.
- Line endings of all 12 files re-checked as LF after a `python` rewrite of the
  test file silently converted it to CRLF (the same trap as `gated_delta_net.cu`
  in the previous port).

**Still not verified:**

- The `--prefill-reuse` + `--checkpoint-recurrent-prev` combination, including the
  boundary-spacing fix from decision 5.
- Interaction with the adaptive-KV-streaming fork end to end; that fork rewrites
  `llama-kv-cache.cpp`, while this change touches the *recurrent* memory and the
  server checkpoint list, so they are disjoint by construction but not by
  measurement.
- Any GPU behaviour. All of the above is CPU.

## Known limitations

- **The fork's percentages do not reproduce, and the reason is measured.** The saving
  is an absolute **3 tokens of replay per turn**, not a proportion: the flag-off
  server restores upstream's checkpoint at `prompt_n - 4` and the flag-on server
  restores the snapshot at `prompt_n - 1`. Upstream already places boundary
  checkpoints at `4 + n_ubatch` and `4` before the end of a prompt
  (`checkpoint_offsets`, upstream `a7b3dee7a` / PR #20288) — and the fork's own base
  `465e49b9c` carries that same array — so this port effectively *replaces* the `-4`
  checkpoint with a `-1` snapshot.

  Consequence: the gain scales inversely with suffix length. Measured
  `-14%` of tokens / `-10%` wall at a ~17-token suffix; that becomes ~`-2.3%` at a
  128-token suffix. **The fork's `+14.08%` at `+128` is not reproduced and should not
  be used for planning.** The plausible explanation for the gap is a regime this test
  could not synthesize: one where the nearest *surviving* checkpoint is far from the
  prompt end (long context, checkpoint list filled, eviction active). Neither the
  4-turn nor the 10-turn run filled a 4-slot list.
- Gains only exist in a specific workload shape: a large cached context receiving
  a short append. The fork measures 64 tokens as neutral and the benefit decays as the
  suffix grows (`+3.33%` at 1000) — which is consistent with a fixed-size saving rather
  than a proportional one, and is worth keeping in mind when reading the fork's table.
- Requires a hybrid/recurrent model **and** `--ctx-checkpoints > 0` **and** no
  speculative decoding. On a plain attention model the flag is a no-op.
- `ON_DEVICE` checkpoints are not portable: they are dropped on slot
  copy and on prompt-cache save. That is intended (they alias one context's
  storage), but it means a restored prompt-cache entry falls back to the host
  checkpoints and therefore to the replay behaviour this feature was meant to
  avoid.
