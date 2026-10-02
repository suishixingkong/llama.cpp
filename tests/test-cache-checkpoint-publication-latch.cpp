// Unit test for the checkpoint-publication latch trio (slice 9l-d4b8i).
//
// The slice's one *definition* group in server-task.cpp:
//
//     server_cache_checkpoint_publication_attempt_begin
//     server_cache_checkpoint_refusal_state_changed
//     server_cache_checkpoint_publication_skipped
//
// All three were declared in server-cache-authority.h since 8q and never DEFINED
// anywhere in this tree -- the same latent-dangling-declaration shape the 9k-a
// ring-changed test documents for its sibling.  The VBR capture kernel landed by
// the same slice is their first caller, so before this test the only evidence for
// them was "it links".  Each is a pure transition on a caller-owned context, so
// they can be pinned without a model.
//
// Why each effect is load-bearing:
//   * attempt_begin latches the `publication_check` lane once per ring generation.
//     An unchanged ring must keep paying ONE integer comparison per pass rather
//     than rebuilding every identity and lease; a changed ring must not suppress a
//     publication it has never evaluated.  The two lanes must also stay
//     independent, or a capacity pass would silently disable publication.
//   * refusal_state_changed is the two-lane refusal latch.  It decides whether a
//     skip is worth reporting at all, and the lane split is what keeps a
//     publication skip from consuming the capacity-floor lane's evidence.
//   * publication_skipped turns a refused publication into ONE destruction
//     receipt and one ledger-sequence bump, and must never delete an incumbent.
//     Its first guard is `!context.authority` -- and because that check
//     short-circuits BEFORE the refusal latch, a context with no authority must
//     leave the latch untouched.  If the two conditions were swapped, a
//     null-authority pass would eat the next real skip's evidence.
//
// Deliberately NOT assert-based: a Release build defines NDEBUG, which turns
// assert() into a no-op and an assert-shaped test into a vacuous one.  Failures
// are counted, printed, and the exit code is derived from the count.
//
// Model-free and CPU-only: no GGUF is loaded, no llama_context exists.  The
// authority is a heap object because server_cache_authority owns a
// server_retention_sidecar_store with a real destructor.

#include "server-cache-authority.h"
#include "server-cache-lifecycle.h"
#include "common.h"

#include <cstdio>
#include <list>
#include <memory>

static int  g_checks   = 0;
static int  g_failures = 0;
// One deliberately-false check is injected as a negative control; `g_report` is
// cleared for that single evaluation so a *passing* run never prints FAIL.
// Judging by output text (not an exit code) is this tree's convention, and a stray
// FAIL line in a green run would read as a red one.  `g_muted` counts the reports
// that were produced but not printed, so the control still proves the
// failure/report path ran.
static bool g_report   = true;
static int  g_muted    = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            if (g_report) {                                                  \
                std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
            } else {                                                         \
                ++g_muted;                                                   \
            }                                                                \
        }                                                                    \
    } while (0)

#define CHECK_EQ(a, b)                                                       \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!((a) == (b))) {                                                 \
            ++g_failures;                                                    \
            if (g_report) {                                                  \
                std::printf("FAIL %s:%d  %s == %s\n", __FILE__, __LINE__,     \
                            #a, #b);                                         \
            } else {                                                         \
                ++g_muted;                                                   \
            }                                                                \
        }                                                                    \
    } while (0)

using reason_t = common_cache_plan_destruction_reason;
using lane_t   = server_cache_checkpoint_attempt_lane;
using state_t  = common_cache_plan_destruction_state;
using effect_t = common_cache_plan_destruction_effect;

static constexpr size_t REFUSED_TIER = size_t(common_cache_plan_selection::none);

// Everything the context points at.  `authority` is opt-in so a test can assert a
// null-authority call touches nothing at all.
struct fixture {
    std::list<common_prompt_checkpoint>      checkpoints;
    server_cache_checkpoint_attempt_latch    attempts;
    server_cache_destruction_observer        destruction;
    const common_prompt_checkpoint *         seam = nullptr;
    reason_t                                 refusal = reason_t::none;
    std::unique_ptr<server_cache_authority>  authority;
    server_cache_checkpoint_authority_context ctx;

    explicit fixture(bool with_authority = false) :
            ctx{ 11, checkpoints, nullptr, nullptr, &destruction, nullptr,
                 attempts, seam, refusal, false } {
        checkpoints.emplace_back();
        seam = &checkpoints.front();
        if (with_authority) {
            authority.reset(new server_cache_authority());
            ctx.authority = authority.get();
        }
    }
};

// ---------------------------------------------------------------- oracles
// Independent restatements, written from the documented intent rather than copied
// out of the landed bodies, so a transcription slip in either cannot hide behind
// itself.  The per-lane generation is seeded to 0 on purpose: the latch's own
// initial state is "generation 1, nothing attempted", so a first call at
// generation 1 must report a change.

struct attempt_oracle {
    bool begin(uint64_t generation, lane_t lane) noexcept {
        const size_t index = size_t(lane);
        if (per_lane_generation[index] == generation) {
            return false;
        }
        per_lane_generation[index] = generation;
        return true;
    }
    uint64_t per_lane_generation[size_t(lane_t::_count)] = {};
};

struct refusal_oracle {
    bool changed(uint64_t generation, reason_t reason, bool publication_skip) noexcept {
        const size_t lane = publication_skip ? 1u : 0u;
        if (per_lane_generation[lane] == generation &&
            per_lane_reason[lane] == reason) {
            return false;
        }
        per_lane_generation[lane] = generation;
        per_lane_reason[lane]     = reason;
        return true;
    }
    uint64_t per_lane_generation[2] = {};
    reason_t per_lane_reason[2] = { reason_t::none, reason_t::none };
};

// publication_skipped's contract, restated as scalars: gate on authority, gate on
// the publication-skip lane, then exactly one receipt.
struct skip_oracle {
    explicit skip_oracle(bool with_authority, bool with_destruction) :
            authority(with_authority), destruction(with_destruction) {}

    void skipped(uint64_t generation, reason_t reason) noexcept {
        if (!authority) {
            return;
        }
        if (!latch.changed(generation, reason, true)) {
            return;
        }
        ++receipts;
        ++sequence;
        if (destruction) {
            ++skips;
        }
    }

    bool     authority;
    bool     destruction;
    refusal_oracle latch;
    uint64_t receipts    = 0;   // has_receipt transitions
    uint64_t sequence    = 0;   // destruction_quote_sequence
    uint64_t skips       = 0;   // destruction->checkpoint_publication_skips
};

// ---------------------------------------------------------------- sections

static void section_attempt_begin() {
    std::printf("-- server_cache_checkpoint_publication_attempt_begin\n");

    // Arms exactly once per generation: the second pass in the same generation is
    // the "one integer comparison" the lane exists for.
    for (int trial = 0; trial < 64; ++trial) {
        fixture f;
        const uint64_t before = f.attempts.generation();
        CHECK(server_cache_checkpoint_publication_attempt_begin(f.ctx));
        CHECK(!server_cache_checkpoint_publication_attempt_begin(f.ctx));
        CHECK_EQ(f.attempts.generation(), before);
    }

    // A ring change re-arms it -- and only a ring change does.
    {
        fixture f;
        CHECK(server_cache_checkpoint_publication_attempt_begin(f.ctx));
        CHECK(!server_cache_checkpoint_publication_attempt_begin(f.ctx));
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK(server_cache_checkpoint_publication_attempt_begin(f.ctx));
        CHECK(!server_cache_checkpoint_publication_attempt_begin(f.ctx));
    }

    // The two lanes are independent.  If attempt_begin consumed (or was consumed
    // by) the capacity-floor lane, a refused capacity pass would suppress the next
    // publication evaluation.
    {
        fixture f;
        CHECK(server_cache_checkpoint_publication_attempt_begin(f.ctx));
        CHECK(f.attempts.begin(lane_t::capacity_floor));
        CHECK(!f.attempts.begin(lane_t::capacity_floor));
        CHECK(!server_cache_checkpoint_publication_attempt_begin(f.ctx));
    }
    {
        fixture f;
        CHECK(f.attempts.begin(lane_t::capacity_floor));
        CHECK(server_cache_checkpoint_publication_attempt_begin(f.ctx));
        CHECK(!server_cache_checkpoint_publication_attempt_begin(f.ctx));
        CHECK(!f.attempts.begin(lane_t::capacity_floor));
    }

    // It is a latch, not a slot rebind: nothing else may move.
    {
        fixture f;
        const size_t n_before = f.checkpoints.size();
        const auto * seam_before = f.seam;
        f.ctx.debug_observability = true;
        CHECK(server_cache_checkpoint_publication_attempt_begin(f.ctx));
        CHECK_EQ(f.ctx.slot_id, 11);
        CHECK_EQ(f.checkpoints.size(), n_before);
        CHECK(f.seam == seam_before);
        CHECK(f.ctx.authority == nullptr);
        CHECK(f.ctx.retention == nullptr);
        CHECK(f.ctx.leases == nullptr);
        CHECK(f.ctx.debug_observability);
        CHECK_EQ(f.ctx.floor_refusal, reason_t::none);
        CHECK_EQ(f.destruction.checkpoint_publication_skips, 0u);
    }

    // Oracle agreement over a scripted sequence that interleaves both lanes and a
    // ring bump.  128 rounds x 5 steps.
    {
        int disagreements = 0;
        fixture f;
        attempt_oracle o;
        for (int round = 0; round < 128; ++round) {
            bool got, want;
            got  = server_cache_checkpoint_publication_attempt_begin(f.ctx);
            want = o.begin(f.attempts.generation(), lane_t::publication_check);
            if (got != want) { ++disagreements; }

            got  = server_cache_checkpoint_publication_attempt_begin(f.ctx);
            want = o.begin(f.attempts.generation(), lane_t::publication_check);
            if (got != want) { ++disagreements; }

            got  = f.attempts.begin(lane_t::capacity_floor);
            want = o.begin(f.attempts.generation(), lane_t::capacity_floor);
            if (got != want) { ++disagreements; }

            server_cache_checkpoint_ring_changed(f.ctx);

            got  = server_cache_checkpoint_publication_attempt_begin(f.ctx);
            want = o.begin(f.attempts.generation(), lane_t::publication_check);
            if (got != want) { ++disagreements; }
        }
        CHECK_EQ(disagreements, 0);
    }
}

static void section_refusal_state_changed() {
    std::printf("-- server_cache_checkpoint_refusal_state_changed\n");

    // Dedup is on the PAIR (generation, reason) per lane.
    {
        fixture f;
        CHECK(server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::mandatory_anchor));
        CHECK(!server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::mandatory_anchor));
        // The declaration's default is `publication_skip = false`, i.e. lane 0.  A
        // default of `true` would land on lane 1 and report a change here.
        CHECK(!server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::mandatory_anchor, false));
        // A different reason re-fires inside one generation ...
        CHECK(server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::lifecycle_disabled));
        CHECK(!server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::lifecycle_disabled));
        // ... and so does a different lane carrying the same reason.
        CHECK(server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::lifecycle_disabled, true));
        CHECK(!server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::lifecycle_disabled, true));
        // The ring bump re-arms both lanes.
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK(server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::lifecycle_disabled));
        CHECK(server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::lifecycle_disabled, true));
    }

    // `reason_t::none` is a legal lane value, not a "disabled" sentinel: the latch
    // compares the enum, so none-vs-none still dedups.
    {
        fixture f;
        CHECK(server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::none, true));
        CHECK(!server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::none, true));
        CHECK(server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::none, false));
    }

    // No side effects on the context's pointers or the physical ring.
    {
        fixture f;
        const size_t n_before = f.checkpoints.size();
        const auto * seam_before = f.seam;
        f.ctx.debug_observability = true;
        CHECK(server_cache_checkpoint_refusal_state_changed(f.ctx, reason_t::mandatory_anchor, true));
        CHECK_EQ(f.checkpoints.size(), n_before);
        CHECK(f.seam == seam_before);
        CHECK(f.ctx.authority == nullptr);
        CHECK(f.ctx.retention == nullptr);
        CHECK(f.ctx.leases == nullptr);
        CHECK_EQ(f.ctx.floor_refusal, reason_t::none);
        CHECK_EQ(f.destruction.checkpoint_publication_skips, 0u);
    }

    // Oracle agreement over a scripted sequence: reasons cycle across both lanes
    // and a ring bump lands in the middle.  256 rounds x 5 steps.
    {
        int disagreements = 0;
        fixture f;
        refusal_oracle o;
        const reason_t reasons[3] = {
            reason_t::mandatory_anchor, reason_t::lifecycle_disabled, reason_t::none,
        };
        for (int round = 0; round < 256; ++round) {
            const reason_t a = reasons[round % 3];
            const reason_t b = reasons[(round + 1) % 3];
            bool got, want;

            got  = server_cache_checkpoint_refusal_state_changed(f.ctx, a, false);
            want = o.changed(f.attempts.generation(), a, false);
            if (got != want) { ++disagreements; }

            got  = server_cache_checkpoint_refusal_state_changed(f.ctx, a, true);
            want = o.changed(f.attempts.generation(), a, true);
            if (got != want) { ++disagreements; }

            got  = server_cache_checkpoint_refusal_state_changed(f.ctx, b, true);
            want = o.changed(f.attempts.generation(), b, true);
            if (got != want) { ++disagreements; }

            got  = server_cache_checkpoint_refusal_state_changed(f.ctx, b, false);
            want = o.changed(f.attempts.generation(), b, false);
            if (got != want) { ++disagreements; }

            server_cache_checkpoint_ring_changed(f.ctx);
        }
        CHECK_EQ(disagreements, 0);
    }
}

static void section_publication_skipped() {
    std::printf("-- server_cache_checkpoint_publication_skipped\n");

    // ---- no authority: zero side effects, and the refusal latch is NOT consumed.
    // The first guard short-circuits before refusal_state_changed; if the order
    // were swapped a null-authority pass would eat the next real skip's evidence.
    {
        fixture f;
        const size_t n_before = f.checkpoints.size();
        const auto * seam_before = f.seam;
        f.ctx.debug_observability = true;          // open the log door too
        server_cache_checkpoint_publication_skipped(f.ctx, reason_t::mandatory_anchor);
        CHECK_EQ(f.destruction.checkpoint_publication_skips, 0u);
        CHECK_EQ(f.checkpoints.size(), n_before);
        CHECK(f.seam == seam_before);
        CHECK_EQ(f.ctx.floor_refusal, reason_t::none);
        CHECK(f.authority == nullptr);
        // the latch is untouched:
        CHECK(f.attempts.refusal_changed(reason_t::mandatory_anchor, true));
        CHECK(!f.attempts.refusal_changed(reason_t::mandatory_anchor, true));
        // ... and so is the publication-check lane.
        CHECK(f.attempts.begin(lane_t::publication_check));
    }

    // ---- authority, no observer: exactly one receipt per newly-refused reason.
    {
        fixture f(true);
        f.ctx.destruction = nullptr;               // the opt-out door
        const size_t n_before = f.checkpoints.size();
        const auto * seam_before = f.seam;
        const size_t r = size_t(reason_t::mandatory_anchor);
        CHECK(seam_before != nullptr);

        server_cache_checkpoint_publication_skipped(f.ctx, reason_t::mandatory_anchor);
        CHECK_EQ(f.authority->destruction_quote_sequence, 1u);
        CHECK(f.authority->destruction_counters.has_receipt);
        const auto & receipt = f.authority->destruction_counters.last_receipt;
        CHECK_EQ(receipt.state, state_t::refused);
        CHECK_EQ(receipt.reason, reason_t::mandatory_anchor);
        CHECK(common_cache_plan_destruction_effect_has(
            receipt.effects, effect_t::checkpoint_member_drop));
        CHECK_EQ(receipt.admission_sequence, 1u);
        CHECK_EQ(f.authority->destruction_counters.refused[REFUSED_TIER][r], 1u);
        // no observer attached => no skip counter to bump, and no crash
        CHECK_EQ(f.destruction.checkpoint_publication_skips, 0u);

        // The same reason in the same generation is latched: no second receipt.
        server_cache_checkpoint_publication_skipped(f.ctx, reason_t::mandatory_anchor);
        CHECK_EQ(f.authority->destruction_quote_sequence, 1u);
        CHECK_EQ(f.authority->destruction_counters.refused[REFUSED_TIER][r], 1u);

        // A different reason is new evidence.
        server_cache_checkpoint_publication_skipped(f.ctx, reason_t::lifecycle_disabled);
        CHECK_EQ(f.authority->destruction_quote_sequence, 2u);
        CHECK_EQ(f.authority->destruction_counters.refused[REFUSED_TIER]
                     [size_t(reason_t::lifecycle_disabled)], 1u);

        // Never deletes an incumbent, and never rebinds the seam: the physical ring
        // is the caller's business.  A std::list node address is stable, so the
        // seam pointer identifies the very node that was there before.
        CHECK_EQ(f.checkpoints.size(), n_before);
        CHECK_EQ(f.checkpoints.size(), 1u);
        CHECK(f.seam == seam_before);
        CHECK(f.seam == &f.checkpoints.front());

        // The ring bump re-arms the lane, so the first reason is reportable again --
        // and note what the bump DOES do: clearing the seam pointer is
        // server_cache_checkpoint_ring_changed's own third effect (pinned by the
        // 9k-a test), not something publication_skipped is allowed to do.
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK(f.seam == nullptr);
        server_cache_checkpoint_publication_skipped(f.ctx, reason_t::mandatory_anchor);
        CHECK_EQ(f.authority->destruction_quote_sequence, 3u);
        CHECK_EQ(f.authority->destruction_counters.refused[REFUSED_TIER][r], 2u);
        // ... and publication_skipped still did not touch the ring.
        CHECK_EQ(f.checkpoints.size(), n_before);
    }

    // ---- authority + observer: the skip counter is bumped exactly once per
    // published refusal, and the receipt is still one.
    {
        fixture f(true);
        f.ctx.destruction = &f.destruction;
        const size_t n_before = f.checkpoints.size();

        server_cache_checkpoint_publication_skipped(f.ctx, reason_t::lifecycle_disabled);
        CHECK_EQ(f.destruction.checkpoint_publication_skips, 1u);
        CHECK_EQ(f.authority->destruction_quote_sequence, 1u);

        server_cache_checkpoint_publication_skipped(f.ctx, reason_t::lifecycle_disabled);
        CHECK_EQ(f.destruction.checkpoint_publication_skips, 1u);   // latched
        CHECK_EQ(f.authority->destruction_quote_sequence, 1u);

        server_cache_checkpoint_publication_skipped(f.ctx, reason_t::mandatory_anchor);
        CHECK_EQ(f.destruction.checkpoint_publication_skips, 2u);
        CHECK_EQ(f.authority->destruction_quote_sequence, 2u);

        CHECK_EQ(f.checkpoints.size(), n_before);
        CHECK(f.seam == &f.checkpoints.front());
    }

    // ---- the debug observer is a serialization layer only: flipping
    // debug_observability must not move one unit of accounting.
    {
        fixture a(true);
        fixture b(true);
        a.ctx.debug_observability = true;
        const reason_t seq[4] = {
            reason_t::mandatory_anchor, reason_t::mandatory_anchor,
            reason_t::lifecycle_disabled, reason_t::none,
        };
        for (int i = 0; i < 4; ++i) {
            server_cache_checkpoint_publication_skipped(a.ctx, seq[i]);
            server_cache_checkpoint_publication_skipped(b.ctx, seq[i]);
        }
        CHECK_EQ(a.authority->destruction_quote_sequence,
                 b.authority->destruction_quote_sequence);
        CHECK_EQ(a.authority->destruction_counters.has_receipt,
                 b.authority->destruction_counters.has_receipt);
        CHECK(a.authority->destruction_counters.has_receipt);
        CHECK_EQ(a.authority->destruction_quote_sequence, 3u);
    }

    // ---- oracle agreement on the whole contract, over a scripted mix of reasons
    // and null/non-null observers.  512 rounds.
    {
        int disagreements = 0;
        fixture f(true);
        f.ctx.destruction = &f.destruction;
        skip_oracle o(true, true);
        const reason_t reasons[3] = {
            reason_t::mandatory_anchor, reason_t::lifecycle_disabled, reason_t::none,
        };
        for (int round = 0; round < 512; ++round) {
            const reason_t r = reasons[(round * 7) % 3];
            server_cache_checkpoint_publication_skipped(f.ctx, r);
            o.skipped(f.attempts.generation(), r);
            if (f.authority->destruction_quote_sequence != o.sequence ||
                f.destruction.checkpoint_publication_skips != o.skips) {
                ++disagreements;
            }
            if (round % 17 == 0) {
                server_cache_checkpoint_ring_changed(f.ctx);
            }
        }
        CHECK_EQ(disagreements, 0);
        CHECK_EQ(f.authority->destruction_quote_sequence, o.sequence);
        CHECK_EQ(f.destruction.checkpoint_publication_skips, o.skips);
    }

    // ---- and the null-authority half of the same oracle: the sequence and skip
    // counters must both stay at zero however many times it is called.
    {
        int disagreements = 0;
        fixture f(false);
        skip_oracle o(false, true);
        for (int round = 0; round < 512; ++round) {
            server_cache_checkpoint_publication_skipped(f.ctx, reason_t::mandatory_anchor);
            o.skipped(f.attempts.generation(), reason_t::mandatory_anchor);
            if (f.destruction.checkpoint_publication_skips != o.skips) {
                ++disagreements;
            }
        }
        CHECK_EQ(disagreements, 0);
        CHECK_EQ(o.skips, 0u);
        CHECK_EQ(f.destruction.checkpoint_publication_skips, 0u);
        CHECK(f.authority == nullptr);
    }
}

int main() {
    section_attempt_begin();
    section_refusal_state_changed();
    section_publication_skipped();

    // ---------------------------------------------------------------- negative control
    // Prove the counter and the report path are really wired: inject one
    // deliberately-false check, confirm the failure count moved by exactly one and
    // that the failure branch reported (g_muted), then roll both counters back so
    // the injection never reaches the exit code or the reported total.  This is the
    // only way to tell a `0 failures` line from a harness whose counter is not
    // connected -- and this test's whole output is a failure count, so that
    // distinction is the difference between evidence and decoration.
    std::printf("== negative control: the counter really can fail\n");
    {
        const int before       = g_failures;
        const int muted_before = g_muted;
        g_report = false;
        CHECK(false);                                  // deliberately false, not reported
        g_report = true;
        CHECK(g_failures == before + 1);               // the counter moved by exactly 1
        CHECK(g_muted == muted_before + 1);            // ... and the failure branch ran
        g_failures = before;                           // do not let it reach the exit code
        g_checks  -= 3;                                // ... nor the reported total
    }

    std::printf("test-cache-checkpoint-publication-latch: %d checks, %d failures\n",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
