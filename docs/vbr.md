# VBR: variable-bit-rate KV cache

VBR changes KV representations **at runtime**. The default ladder is `auto`: f16 → turbo8 → q5_1 →
q4_1 → turbo4 → turbo3_tcq → turbo2_tcq → turbo1_tcq (16 → 1.25 effective bits/value), which interleaves the two
standard-quant rungs between the turbo ones so each descent step buys the cheapest representation
still better than the one above it. It is a ladder in its own right, not a preference order over
the other two. For BailingMoE3/Ling geometry that cannot host turbo KV rows at all, `auto` is
downgraded to the classic ladder f16 → q8_0 → q5_1 → q4_1; that is a capability fallback, never a
reinterpretation. The cache starts at the selected entry and individual physical KV tensors are
transcoded down the chosen ladder in place as it fills.

## Quickstart

```sh
llama-server -m model.gguf -ctk vbr
```

That single flag is the complete product:

- the cache starts at **f16** — full quality and f16 decode speed until there is actual
  budget pressure (layers keep f16 speed until their own degrade step fires);
- a KV VRAM budget is derived automatically from the memory left after weights/compute
  (the fit pass), or falls back to the floor-layout cost of the full context;
- when `-c` is unset, the advertised context is the budget's capacity at the **floor**
  tier (the selected ladder's own endpoint: t1 for `auto` and `turbo`, q4_1 for
  `classic`), capped at the model's training context;
- as mapped KV approaches the budget, the runtime controller degrades tensors down the
  price order, cheapest quality-per-byte first. With `-v` you can watch the
  `VBR degrade #…` steps fire.

The cache resets losslessly when it empties (tiers return to entry), and fully-degraded
runs stay coherent — the price orders were measured on KLD panels per model.

## Flags

| flag | meaning |
|---|---|
| no cache flags | common CLI tools default to dynamic VBR on both sides with a t4 quality floor; the tensors still enter at F16 and degrade only under pressure. |
| `-ctk vbr` / `-ctv vbr` / `-ct vbr` | explicitly select VBR and, without `--vbr-floor`, open the selected codec's complete ladder. An explicitly non-VBR side (`-ctv q8_0`) is pinned — it never degrades and counts in the aggregate at its fixed bits/value. |
| `--vbr-codec <auto\|turbo\|classic>` | representation ladder. `auto` is the default: f16 → turbo8 → q5_1 → q4_1 → turbo4 → turbo3_tcq → turbo2_tcq → turbo1_tcq. It is only downgraded, to classic, when the model's KV geometry cannot host turbo rows. `turbo` (f16 → turbo8 → turbo4 → turbo3_tcq → turbo2_tcq → turbo1_tcq) is the pure turbo ladder and the only one that may read the measured turbo orders and policies. Explicit families are strict. |
| `--vbr-budget <tier\|number>` (`--vbr-bits`) | `dynamic` (default) = runtime controller. A tier (`t8/t4/t3/t2/t1`) or a number selects a **fixed** static tier instead — no runtime degrades. |
| `--vbr-entry <tier>` | dynamic entry tier, any rung of the selected ladder: `auto` accepts `f16`, `t8`, `q5_1`, `q4_1`, `t4`, `t3`, `t2`, `t1`; `turbo` accepts `f16`, `t8`, `t4`, `t3`, `t2`, `t1`; `classic` accepts `f16`, `q8_0`, `q5_1`, `q4_1`. A lower entry is an explicit quality-for-bandwidth trade. |
| `--vbr-floor <bits\|tier>` (`--vbr-min-bits`) | LITERAL aggregate K+V bits/value floor for dynamic mode. The degrade order stops at the last step whose aggregate stays ≥ the floor — e.g. `4.25` means "t4 layout with a few units held one tier higher", **not** a snap-up to the next tier. Clamped to the reachable range of the selected ladder — minimum 1.25 for `auto` (its endpoint is the t1tcq rung), 1.25 for `turbo`, 5.0 for `classic`; maximum depends on the selected entry and any pinned side. Implicit VBR defaults to t4 (4.125); explicit `-ct vbr` without this flag uses the ladder's own endpoint (`auto` t1tcq = 1.25, `turbo` t1 = 1.25, `classic` q4_1 = 5.0). |
| `--vbr-vram <SIZE>` | explicit KV VRAM budget (e.g. `8G`). Default `auto` = derived by the fit pass. |
| `--vbr-policy <json>` | fixed mode only: a measured policy ladder; the best rung ≤ the fixed budget (and ≥ the floor) is selected and its per-layer schedule applied. |
| `--cache-ram <MiB>` | on the server, a nonzero value enables automatic projected-artifact host caching for supported dynamic-VBR topologies (default: 8192). `--cache-ram 0` is the zero-cost opt-out. |

Notes:

- The C API mirror is `llama_context_params.vbr_dynamic` / `vbr_codec` /
  `vbr_vram_budget_bytes` / `vbr_min_bits` — that struct is the runtime channel.
- A non-tier floor costs slightly more than the tier below it; capacity estimates account
  for this (the advertised context is scaled to the literal floor's cost).
- When both K and V are movable, the floor cannot exceed the selected entry tier. With one
  pinned side, the floor is aggregate and may be higher than the movable side's entry bpv.
- `/props` and `/models` on the server expose a `vbr` object (codec, mode, floor, budget,
  realized/selected bits — `null` where a value is not a fixed number under the dynamic
  controller).

## Requirements and limitations

- **Backend**: turbo-typed KV needs a backend that exports the TurboQuant interface
  (`ggml-vbr.h`) — currently CUDA (and ROCm for the VMM pool). KV on other backends is
  refused at init; layers whose KV lands on the CPU (partial offload, `--no-kv-offload`)
  fall back to q8_0 per layer with a warning.
- **Classic ladder**: available to any model whose attention KV cache has head dimensions that
  are multiples of 64 (the classic dynamic path enables a Hadamard rotation), on a VMM-capable
  CUDA/ROCm device. It keeps the native head width, uses no Turbo rotation/mean/tap/stash state, and uses
  the generic banded order by default: all movable layers reach q8_0 before any advances to q5_1,
  and all reach q5_1 before any advances to q4_1. `VBR_DEGRADE_ORDER` may supply an explicit
  classic order.
- **The turbo-*layout* and the turbo-*artefacts* are different things.** The `auto` ladder contains
  turbo rungs, so it needs the turbo row layout (padded rows, rotation tensors, sink-stash, no
  upstream Hadamard rotation) exactly as `turbo` does. It may not, however, read anything measured
  on the turbo ladder — the baked degrade orders, `--vbr-policy` ladders and `VBR_LAYER_SCHEDULE`
  are all keyed by band number, and band 1 means `t8→q5_1` on `auto` but `t8→t4` on `turbo`. Those
  features require `--vbr-codec turbo`; with `auto` the cache uses the conservative generic order
  and refuses capture. The hard-seal boundary is likewise derived from the ladder rather than fixed
  at band 1, so `auto` seals at the turbo4 crossing (band 3) where `turbo` seals at band 1.
- **Flash attention** is required and force-enabled (turbo KV stores rotated-space data
  that only the FA paths decode).
- **Unified KV**: dynamic mode needs single-stream KV; with parallel sequences
  (`-np > 1`, perplexity/imatrix multi-sequence batching) unified KV is forced
  automatically, with a warning.
- **Qwen4 Flash Next**: live static Turbo and dynamic VBR are supported for its ordinary
  attention child. The QSA index cache remains a separate fixed-F16 allocation and the
  recurrent child remains fixed-layout. Host prompt artifacts carry both fixed-layout children
  as authenticated companions and adopt the complete tree atomically. Qwen4 currently uses the
  generic degradation order; a measured model-specific order is still pending.
- **Context shift / self-extend** is disabled under dynamic VBR (the shift graph would
  touch unmapped pool pages); generation stops cleanly when the context fills.
- **Host prompt cache**: supported server topologies use authenticated projected VBR
  artifacts under the ordinary `--cache-ram` budget. The fixed-layout state serializer,
  manual slot files, and KV shifting/cache reuse remain disabled because their snapshots
  carry tier-typed KV. If artifact construction is unavailable for a topology, startup
  reports the typed refusal and continues live-only. Context checkpoints stay enabled on
  hybrid models (recurrent state only) but are disabled on SWA models.
  The artifact path is codec-agnostic — it keys identity on a digest over the KV type
  vector — so `classic` and `auto` keep the host cache whenever the model can host that
  codec's ladder; only a model that cannot is refused, by capability rather than by codec
  name. An occupied slot whose incoming prefix has no exact authenticated host entry logs
  `automatic occupied VBR restore refused: no exact host candidate`; that launch keeps the
  live KV's common prefix instead of importing the artifact, and the output is unaffected.
  Publication is not the same as consumption: see *Which restore route can actually fire*
  below for which routes accept which artifacts.
- **Exact continuation**: complete-frontier artifacts also authenticate the terminal raw
  vocabulary-logit row. Restoring an exact prompt therefore reports `prompt_n = 0`
  instead of replaying its last token solely to regenerate logits. With MTP or DFlash2,
  that cached first sample feeds the normal speculative loop; drafting is not disabled.
  While VBR prompt caching is enabled, the target keeps a full raw row instead of using
  backend-only sampling or the DFlash target-argmax shortcut.
- Degrade progress lines are `INFO`-level: visible with `-v`. Greedy-identical output does
  NOT mean no degrades happened.
- `--vbr-policy` exports its per-layer schedule through a process-global env: with a
  speculative **draft model** in the same process, the schedule also applies to the
  draft's cache (overriding `-ctkd/-ctvd`). Avoid `--vbr-policy` together with `-md`.

### Automatic host-cache support matrix

The default-on server route is intentionally narrower than the explicit artifact APIs.
It supports text-only, non-aLoRA slots on a dynamic-VBR target whose artifact store can
bind every attention child, device lane, and accounting domain. Ordinary LoRA identity,
attention-only targets, hybrid/recurrent targets, SWA/iSWA trees, MTP, external draft
models including DFlash2, single-GPU placement, and bound multi-GPU placement use that
route. Stateful trees and speculative slots are captured as exact complete-frontier
artifacts with their required companion state. Shortened stem publication remains
attention-only; unsupported stem shapes retain the complete live frontier.

The following automatic cases are typed live-only fallbacks today:

| Case | Typed reason | Behavior |
| --- | --- | --- |
| media prompt/projector state | `media_prompt_unsupported` | that live slot is neither captured nor displaced |
| aLoRA invocation boundary | `alora_invocation_unsupported` | that live slot is neither captured nor displaced |
| model cannot host the requested codec's ladder | `codec_unsupported` | automatic startup continues live-only (`--cache-ram` disabled) |
| missing portable topology or pool/lane binding | `artifact_topology_unavailable` | automatic startup continues live-only |
| incomplete accounting or budget evidence | `accounting_unavailable` | automatic startup continues live-only |
| pinned-ring/store construction refusal | `artifact_store_unavailable` plus the store failure/status | automatic startup continues live-only |
| missing, corrupt, or mismatched Qwen4 QSA-index companion | `required_companion_unavailable` | the artifact is refused before publication; the live slot is retained or the prompt replays cold |

Automatic fallback never clears the live source. An explicit `--vbr-prompt-cache`
request is strict instead: an unsupported global topology or unavailable store fails
startup with the same typed reason. Transient capture/admission failures retain the live
source and retry; permanent per-artifact incompatibilities are terminal only for that
unchanged frontier.

#### Which restore route can actually fire

Publication and restoration are asymmetric on purpose. Publication may shorten an
attention prefix to fit the host budget (`longest_attention_stem`), while restoration
must be able to prove it can roll back whatever it displaces. The automatic routes
therefore have different admission rules, and not every published artifact is consumable
by every route:

| Route | Admission | Consequence |
| --- | --- | --- |
| empty-slot handoff (whole-artifact import) | `slot_count == 1`, no hard lease, no deferred task, incumbent supported, and the candidate prefix is longer than the live incumbent's common prefix | the only route that can consume an artifact carrying fixed-layout companions. Multi-slot unified KV cannot clear one sequence out of a shared tree, so this route is closed for `-np > 1`. |
| occupied-slot replacement | requires a **durable host copy of the live incumbent at the same token count** — that copy is the rollback witness — plus a candidate prefix longer than the live common prefix | fails closed whenever no equal-length sealed copy of the live frontier exists. A shortened stem is shorter than the live frontier by construction, so it can never serve as its own rollback witness. |
| prefix projection (shortened stem) | attention-only: refuses any artifact whose package carries a companion descriptor | hybrid/recurrent targets always carry a recurrent companion, so their artifacts can only be imported whole. |

Two practical consequences follow. First, with `-np > 1` the empty-slot route is closed
and hybrid artifacts cannot be projected, so automatic host restore has no usable path
(`--cache-ram` still publishes); with `-np 1` the empty-slot route works and is the route
to rely on. Second, a confirmed import is a fixed-cost operation — dominated by the
device-to-host transfer and adoption of the artifact rather than by its token count — so
it only beats a cold prefill on long prompts; short prompts are cheaper replayed.

### Host-cache control-plane payloads

The cache control plane uses one payload vocabulary everywhere: `fixed_state` is the
historical serialized target/draft state image and `vbr_artifact` is an authenticated,
immutable projected artifact. `/cache/plan` reports `payload_kind` on every host candidate;
cache lifecycle and destruction receipts use the same field. Lease inspection and event
records derive it from their typed `host_snapshot` or `vbr_reference` subject. Families
apply to both representations and therefore report
`payload_scope: ["fixed_state", "vbr_artifact"]`; they are lineage policy, not a second
payload owner. Live prefixes and checkpoints report a null payload kind because their state
has not entered the host tier.

## Degrade orders

The Turbo price order — which (layer, side) to degrade next, per tier band — is measured per
model on KLD panels and baked into the binary for the supported fleet
(`src/llama-vbr-degrade-orders.inc`). Models without a baked order use a **generic**
cross-model order synthesized from normalized-position rank curves averaged over the
measured fleet (dense, MoE-hybrid and SWA-mixed layouts); it is a good hedge but a
measured order is better. Those baked tables are keyed by **band number**, so only the `turbo`
ladder may consume them: `auto` and `classic` use the generic curve positionally — `auto`'s five
bands take rows 0..4, `classic`'s three take rows 0..2. Override or supply an order with:

```sh
VBR_DEGRADE_ORDER=order.txt llama-server ...   # tokens: <layer><k|v>:<tier>
```

The accepted tier names are the selected ladder's own rungs: `t8`/`t4`/`t3`/`t2`/`t1` for `turbo`,
`t8`/`q5_1`/`q4_1`/`t4`/`t3` for `auto` (canonical names and the short `q5`/`q4` spellings both
work), and `q8`/`q8_0`, `q5`/`q5_1`, `q4`/`q4_1` for `classic`. They are derived from the ladder,
so a tier name always denotes the band that lands on that rung — a name is never reinterpreted by
a rung being inserted above it.

`VBR_FORCE_GENERIC=1` forces the generic order on a supported model (A/B instrument).

## Environment appendix

CLI-managed internals (set nothing yourself; these are developer overrides on top of the
cparams channel): `VBR_VMM` (=0 forces the controller off, =1 on), `VBR_MODE`,
`VBR_BUDGET_MIB`, `VBR_MIN_BITS`.

Product-adjacent knobs: `VBR_DEGRADE_ORDER` (order file), `VBR_LAYER_SCHEDULE`
(per-layer static schedule, `<il0>-<il1>:<k|v>:<tier>;…` or `@file`; `VBR_SCHEDULE_CTX`
sets the discovery window the row coordinates refer to, `VBR_LAYER_STRICT=1` errors on
unsupported bands), `VBR_POLICY_LADDER` (default for `--vbr-policy auto`),
`VBR_STASH_ROWS` (any turbo-layout cache — `turbo` and `auto`; f16 sink-stash rows per tensor,
default 128; 0 disables),
`VBR_FORCE_GENERIC`.

Developer instruments (off by default, safe to ignore): `VBR_VRAM_HEADROOM_MIB` (free-VRAM
clamp headroom, default 192), `VBR_PROMOTE` (container promotion kill switch, default on),
`VBR_TRANSCODE_TEST[_N]` (in-binary transcode oracle), `VBR_TRANSCODE_FIDELITY`
(per-transcode error report; slow), `VBR_TRANSCODE_NOTILE` (tiling isolator; unbounded
scratch), `VBR_STASH_CAPTURE_ONLY` (stash ablation).

Telemetry exports for wrapper scripts (written, never read): `VBR_SELECTED_*`,
`VBR_CAPACITY_BITS`, `VBR_VRAM_BUDGET`, `VBR_BUDGET`.

## Deterministic tier schedules (test/gating only)

`VBR_FREEZE=1` disables the live-VRAM clamp and the ledger, so with a fixed budget the degrade waves
become a pure function of occupancy — "scripted" waves. It is only deterministic together with an
explicit budget: set `VBR_BUDGET_MIB=<MiB>` as well, otherwise the auto budget keeps re-deriving from
live free VRAM (the code warns in that case). `VBR_FREEZE_PRESERVE_EMPTY_TIERS=1` retains the current
tier vector across empty boundaries. Both are read once per cache at construction
(`llama-kv-cache.cpp`). For a fully hermetic schedule also pin `VBR_FORCE_GENERIC=1` (arch-independent
degrade order), `VBR_PROMOTE=0` and `VBR_STASH_ROWS=0`, as `tests/test-vbr-representation-epoch.cpp`
does.

⚠️ Two traps when reading a `tier_epoch` of 0 back from a short run:

1. `VBR_BUDGET_MIB` **overrides** `--vbr-vram` outright (`llama_memory_vbr_budget_bytes_resolve`), so
   passing both does not mean what it looks like.
2. `vbr_tier_epoch` only advances inside the transcode path, and the degrade trigger needs the cache
   watermark to **cross** the budget (`vbr_vmm_active() && vbr_budget_bytes_ > 0`, then
   `prepare_with_slots`). Any budget above the run's KV watermark — including an "explicit" but
   generous `--vbr-vram`, and including the auto budget, which tracks live free device memory and is
   re-derived every boundary — therefore produces **no** degrade at all and `tier_epoch` stays 0
   forever. Anything that asserts on a retier has to be given a budget that actually binds.

**Calibrate small -- and note the band is two-sided.** Measured on a V100 at `-c 8192`: a plain
(non-SWA) hybrid model took **12** degrade steps at `--vbr-vram 25M`, but **0** at `--vbr-vram 2G`
and at `VBR_BUDGET_MIB=160`.

Both ends fail, and the log says which end you are on:

| What you see | What it means | Which knob |
|---|---|---|
| no `VBR degrade #` at all | budget too **big**: the watermark never crossed it | shrink the budget |
| `VBR budget X MiB exceeded with the degrade order clamped at the --vbr-floor` | budget too **small for the floor layout**: the ladder saturates at the floor on the very first boundary, so **no** step is left to take later and nothing can increase inside the assertion window | grow the budget. Lowering `--vbr-floor` alone only helps if the ladder still has steps left -- measured, it changed the message to `exhausted` and the allocation still did not fit |
| the same line but saying `exhausted` (not `clamped at the --vbr-floor`) | the ladder itself ran out (or the floor was already at the endpoint) | only a bigger budget helps |

`--vbr-floor BITS` is the second lever: the implicit default stops the ladder at **t4 = 4.125**, while
an explicit low floor (the ladder's own endpoint -- **1.25** for turbo and auto) leaves a lot of
headroom. It also moves the pool layout, so it changes the split: on a measured iSWA model
`VBR_BUDGET_MIB=25` split **1/3 : 2/3** (8.33 / 16.67 MiB) at the default floor but **1/4 : 3/4**
(5.00 / 20.00 MiB) with `--vbr-floor 1.25`. Below the floor-layout cost the reservation is refused
instead, so the usable band is an interval, not "smaller is always better".

**The binding quantity is each child's share against its own cost.** On the iSWA fixture the two
children are asymmetric: their projections at the same watermark were **12.00 MiB** (base) and
**48.00 MiB** (SWA) -- a 1 : 4 cost ratio -- while the scalar budget splits between them **1 : 2**
(base : SWA; measured at both 1/3 : 2/3 and 1/4 : 3/4). Against its own cost the base child is
over-provisioned and the SWA child under-provisioned by construction. Full sweep, `-c 8192`,
`VBR_FREEZE=1`:

| `VBR_BUDGET_MIB` | child shares (base : SWA) | `prepare_with_slots` warnings | outcome |
|---|---|---|---|
| 25 | 8.33 : 16.67 | 2, both `clamped at the --vbr-floor` | fail: both children saturate at once |
| 25 + `--vbr-floor 1.25` | 5.00 : 20.00 | 2, now `exhausted` | fail: the floor stopped limiting, the ladder ran out |
| 64 | 21.33 : 42.67 | **1** (SWA only) | fail: base now fits (21.33 > 12.00), SWA is 42.67 < 48.00 |
| 96 | 32.00 : 64.00 | 0 | fail the other way: neither child ever exceeds its share |
| 128 | 42.67 : 85.33 | 0 | same |
| 160 | (same layout) | 0 | same |

So watch the warning **count** as well as its wording -- but read it correctly:

1. The warning is **one-shot per cache** (`vbr_budget_warned_`; only `vbr_full_reset()` clears it).
   The count dropping 2 -> 1 -> 0 does not measure "how close you are": it says how many of the two
   children have **already saturated at least once**. `0` means "neither child ever exceeded its
   share", which is also a failure, just the opposite one.
2. `vbr_tier_epoch_++` has **three** sites and only one of them logs: `vbr_degrade_next()`
   (`llama-kv-cache.cpp:9669`, logging `VBR degrade #` at `:9659`), `vbr_promote_next()` (`:9340`) and
   `vbr_full_reset()` (`:9088`) advance the epoch silently. The absence of a `VBR degrade #` line is
   therefore **not** evidence that nothing retiered -- a promote bumps the same counter.

The definitive instrument is already in the tree: the per-pool trace, enabled by `-v`.

    VBR pool #<i> (device <d>): projected <X> / budget <Y> MiB (mapped <Z>) at <N> cells

One line per pool per boundary (`LLAMA_LOG_DEBUG`). A pool retiers on a boundary exactly when
`projected > budget`, so this line answers "does child C cross inside window W" directly, instead of
inferring it from the budget value or from the warning count.

**What actually decides reachability: where each child's cost crosses its share.** A pool retiers on
the boundary where `projected > budget`, so an arm that asserts on `tier_epoch` is satisfiable only if
the crossing falls inside **its own** assertion window (~1278 -> 2302 cells of watermark for the two
SWA arms). Each child's projection grows linearly in the watermark up to its own capacity, and the two
capacities are not comparable:

| child | projection @ 256 cells | capacity |
|---|---|---|
| base | 12.00 MiB | `kv_size` ~= n_ctx (8192 cells) |
| SWA | 48.00 MiB | `PAD(n_swa * n_seq_max + n_ubatch, 256)` = **2048** cells here |

Do not assume `size_swa == n_swa`: the sharing arms build a **3-sequence unified** context
(`make_ctx_sharing(..., 3)` plus `kv_unified = true`), so the SWA cache is still filling for roughly
60% of the window and its crossing can land inside it. (An earlier revision of this section assumed
`size_swa == n_swa` and wrongly called the arm unreachable.)

Two consequences:

1. **The band is two-sided** and the log says which side you are on -- but read the warning count
   right. It is **one-shot per cache** (`vbr_budget_warned_`, cleared only by `vbr_full_reset()`), so
   2 -> 1 -> 0 does *not* measure "how close you are": it only says which children have already
   saturated. **Zero warnings is also a failure**, just the opposite one.
2. **Which budget works depends on the SPLIT**, because the two children's per-cell costs differ 4x.
   Writing the split as `s_base : s_swa`, both crossings land inside one window only when
   `(s_swa / s_base) / 4` falls within `(ahead / advanced, advanced / ahead] = (0.56, 1.79)`. The
   sweep above is exactly that constraint failing:

| split (base : SWA) | crossing ratio | verdict |
|---|---|---|
| 1 : 2 (measured at 25 / 64 / 96 / 128 / 160 MiB) | 0.50 | **impossible**: below `1 / 1.79` |
| 1 : 4 (measured at 25 MiB + `--vbr-floor 1.25`) | 1.00 | **feasible**, if the total puts both crossings in the window |

**The split is piecewise, and that is what closes the arm.** `llama_memory_vbr_budget_partition` has
three regimes (see above). Feeding it the measured floor costs (12.00 / 48.00 MiB, printed by the
exhausted warnings) and fitting its output to the measured splits at 25 / 80 / 120 MiB identifies the
entry costs as `(48, 48)` -- every printed pair is reproduced exactly, including `[5, 20]` at 25 and
`[32, 48]` at 80:

| `VBR_BUDGET_MIB` | regime | base share | SWA share | observed |
|---|---|---|---|---|
| 25 | `B < floor_total` (60) | `B/5` = 5.00 | `4B/5` = 20.00 | both ladders walked at the **first** boundary (`#1..#42`, `#1..#168`), then frozen |
| 80 | `floor_total <= B < entry_total` (96) | `B-48` = 32.00 | **48.00** (constant -- the SWA's weight here is `entry-floor` = 0) | **zero** degradation, gate `(0,0)->(0,0)` |
| 120 | `B >= entry_total` | 60.00 | 60.00 | **zero** degradation |

So the SWA's share is `4B/5` below 60, exactly **48.00** from 60 to 96, and `B/2` above 96 -- and 48.00
is precisely its floor cost *and* its 48 MiB VA pool, while its projection never exceeds 48 (at a
share of 48.00 it does not degrade at all, so `projected > budget` never held during the whole run).

That leaves the SWA child only two states, and neither is a crossing:

* `share >= 48` -> its projection never exceeds the share -> `tier_epoch_swa` stays 0;
* `share < 48` -> it is over from the start and the ladder cannot bring its cost below the floor ->
  the whole order is walked immediately and the epoch jumps to 168 and freezes.

Meanwhile the base needs `share >= 12` not to be pinned. Walk the regimes:

| regime | base pinned? | SWA under pressure? |
|---|---|---|
| `B < 60` | **yes** (`B/5 < 12` for every `B < 60`) | yes |
| `60 <= B < 96` | no | **no** (exactly 48.00) |
| `B >= 96` | no | **no** (`>= 48`) |

**No budget makes the base free and the SWA under pressure at the same time.** Under a
floor-proportional split that is structural, not bad luck: `s_i = B * floor_i / floor_total` makes
`s_i >= floor_i` equivalent to `B >= floor_total` for *every* pool, so all children cross from
"pinned" to "free" on the same total. With `share_media_swa_prefix` requiring **both** children to
retier strictly inside the window, the arm's VBR branch is **unreachable in this fixture**.

That is only the *share* half of the argument -- it says no budget leaves the right combination of
pressures. The next section says why the SWA child can never cross at all, which is independent of
the split and is the actual root cause.

**Root cause: page granularity locks the SWA pool's footprint.** The numbers above are not a
statistical pattern, they follow from `vbr_vmm_projected_bytes`:

```cpp
const size_t need = ggml_row_size(t->type, t->ne[0]) * wm_cells;
total += std::min(vbr_slot_span(t, p.gran), GGML_PAD(need, p.gran));   // per (layer, side)
```

`vbr_slot_span` is `GGML_PAD(slot_bytes, gran)`, and on this box `gran` is the 2048 KiB VMM page
(the log says so). Filling in the fixture's real geometry:

| child | extents | logical span per extent | page-padded | projection |
|---|---|---|---|---|
| base | 3 layers x (k,v) = 6 | 8.00 MiB (f16, 8192 cells, dim 512) | 8.00 MiB | `min(8, PAD(need))` -> **2 MiB/extent below wm 2048, 4 MiB above** |
| SWA | 12 layers x (k,v) = 24 | **1.00 MiB** (f16, 2048 cells, dim 256) | **2.00 MiB** | `min(2, PAD(need))` = **2 MiB/extent always** |

which reproduces both measured warnings exactly -- base `12.00 MiB` (6 x 2) and SWA `48.00 MiB`
(24 x 2) -- and shows why they behave differently:

* **SWA**: its logical per-extent span (1 MiB) is *smaller than one VMM page*, so
  `GGML_PAD(need, gran)` is 2 MiB for **every tier and every watermark**, and so is the slot span.
  Its projected footprint is therefore a **constant 48 MiB = 2x its f16 data size (24 MiB)** -- both
  numbers are printed side by side by the log itself (`48.00 MiB VA reserved` vs
  `size = 24.00 MiB (2048 cells, 12 layers)`). `vbr_over_budget` can then only be permanently true
  (`share < 48`) or permanently false (`share >= 48`); it can never change mid-run, so the ladder is
  walked to `exhausted` at the first boundary (`#1..#168`) and buys nothing.
* **base**: its 8 MiB span exceeds a page, so `min(8, PAD(need))` *steps* -- 2 MiB below wm 2048,
  4 MiB above -- i.e. its projection is a staircase `12 -> 24 MiB`, and the step at wm 2049 falls
  **inside** the arm's window (1278 -> 2302). The base child can therefore cross inside the window;
  the SWA child cannot, ever.

⇒ `test_share_media_swa_prefix` requires a *new* retier for **both** children inside that window, so
on any iSWA model whose SWA per-extent span is below one VMM page the assertion is unsatisfiable --
this is a **class of models**, not one unlucky fixture. `test_share_live_prefix_swa` passes only
because it asserts `!= 0`, which the burnout satisfies.

**One run confirms it** (prediction stated so it can be refuted): `VBR_BUDGET_MIB=61` with
`--vbr-floor 1.25` gives shares 13 / 48, so the base crosses at wm 2049 while the SWA cannot -- the
failure message should print a **non-zero base with a zero SWA** (e.g. `(0,0)->(2,0)`). A non-zero
SWA there, or a zero base, refutes the whole page-granularity reading.

Either way the arm's historical branch (`-ctk f16 -ctv f16`, which skips the gate) passes and covers
its real subject: row-set construction, sharing, companion recovery, exact replay and the
expired-window refusal.
