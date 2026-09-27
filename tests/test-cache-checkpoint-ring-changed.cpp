// Unit test for server_cache_checkpoint_ring_changed (slice 9k-a).
//
// Why this one function out of the slice: it is the only *definition* the slice
// added. Everything else it landed is a splice. Before the slice the function had
// no definition anywhere in the tree even though server-cache-authority.h had
// declared it since 8q -- a latent dangling declaration (its would-be caller,
// server_slot::checkpoint_ring_changed, was itself never emitted). The restore/trim
// transaction odr-uses that member, so the linker finally demanded the definition:
//
//   server-context.lib(server-context.cpp.obj) : error LNK2019:
//   无法解析的外部符号 server_cache_checkpoint_ring_changed
//
// The function is a pure three-effect transition on a caller-owned context, so it
// can be pinned without a model. Pinning it matters because all three effects are
// load-bearing for the checkpoint ring:
//   * the generation bump is what re-arms BOTH latches -- an unchanged ring must
//     keep paying one integer comparison per pass, and a changed ring must not
//     suppress a publication it has never evaluated;
//   * clearing seam_heuristic forbids reusing a seam decision taken against a ring
//     that no longer exists;
//   * floor_refusal = mandatory_anchor makes the next capacity pass fail closed
//     instead of evicting a member it has not yet re-evaluated.
//
// Deliberately NOT assert-based: a Release build defines NDEBUG, which turns
// assert() into a no-op and an assert-shaped test into a vacuous one. Failures are
// counted, printed, and the exit code is derived from the count.
//
// Model-free and CPU-only: no GGUF is loaded.

#include "server-cache-authority.h"
#include "common.h"

#include <cstdio>
#include <list>

static int  g_checks   = 0;
static int  g_failures = 0;
// One deliberately-false check is injected as a negative control; `g_report` is
// cleared for that single evaluation so a *passing* run never prints FAIL. Judging by
// output text (not an exit code) is this tree's convention, and a stray FAIL line in a
// green run would read as a red one. `g_muted` counts the reports that were produced
// but not printed, so the control still proves the failure/report path ran.
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

// Everything the context points at, so a test can assert the function touched
// none of it.
struct fixture {
    std::list<common_prompt_checkpoint>  checkpoints;
    server_cache_checkpoint_attempt_latch attempts;
    const common_prompt_checkpoint *      seam = nullptr;
    reason_t                              refusal = reason_t::none;
    server_cache_checkpoint_authority_context ctx;

    fixture() :
            ctx{ 7, checkpoints, nullptr, nullptr, nullptr, nullptr,
                 attempts, seam, refusal, false } {
        checkpoints.emplace_back();
        seam = &checkpoints.front();
    }
};

// An independent restatement of the fork's own three statements (fork
// server-task.cpp:4439-4442), written from the intent rather than copied from the
// implementation, so a transcription slip in the landed body cannot hide behind
// itself.
static void oracle_ring_changed(
        server_cache_checkpoint_attempt_latch & attempts,
        const common_prompt_checkpoint *& seam,
        reason_t & floor_refusal) noexcept {
    attempts.ring_changed();
    seam = nullptr;
    floor_refusal = reason_t::mandatory_anchor;
}

int main() {
    // ---------------------------------------------------------------- effect 1: the bump
    for (int trial = 0; trial < 64; ++trial) {
        fixture f;
        const uint64_t before = f.attempts.generation();
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK_EQ(f.attempts.generation(), before + 1);
    }

    // ---------------------------------------------------------------- effects 2 and 3
    for (int trial = 0; trial < 64; ++trial) {
        fixture f;
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK(f.seam == nullptr);
        CHECK_EQ(f.refusal, reason_t::mandatory_anchor);
    }

    // The function must not reach into anything else. A ring change is not a slot
    // rebind: the physical checkpoint list and the authority pointers are the
    // caller's business.
    {
        fixture f;
        const size_t n_before = f.checkpoints.size();
        f.ctx.debug_observability = true;
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK_EQ(f.ctx.slot_id, 7);
        CHECK_EQ(f.checkpoints.size(), n_before);
        CHECK(f.ctx.authority == nullptr);
        CHECK(f.ctx.retention == nullptr);
        CHECK(f.ctx.destruction == nullptr);
        CHECK(f.ctx.leases == nullptr);
        CHECK(f.ctx.debug_observability);
        CHECK(f.ctx.seam_heuristic == nullptr);
    }

    // ---------------------------------------------------------------- latch re-arm
    // This is the reason the generation exists: an unchanged ring pays one integer
    // comparison per pass, and only a membership change re-opens the lane.
    {
        fixture f;
        CHECK(f.attempts.begin(server_cache_checkpoint_attempt_lane::publication_check));
        CHECK(!f.attempts.begin(server_cache_checkpoint_attempt_lane::publication_check));
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK(f.attempts.begin(server_cache_checkpoint_attempt_lane::publication_check));
        CHECK(!f.attempts.begin(server_cache_checkpoint_attempt_lane::publication_check));
    }
    {
        fixture f;
        CHECK(f.attempts.begin(server_cache_checkpoint_attempt_lane::capacity_floor));
        CHECK(!f.attempts.begin(server_cache_checkpoint_attempt_lane::capacity_floor));
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK(f.attempts.begin(server_cache_checkpoint_attempt_lane::capacity_floor));
    }
    // The two lanes are independent in both directions.
    {
        fixture f;
        CHECK(f.attempts.begin(server_cache_checkpoint_attempt_lane::publication_check));
        CHECK(f.attempts.begin(server_cache_checkpoint_attempt_lane::capacity_floor));
        CHECK(!f.attempts.begin(server_cache_checkpoint_attempt_lane::publication_check));
        CHECK(!f.attempts.begin(server_cache_checkpoint_attempt_lane::capacity_floor));
    }
    // A refusal latch is re-armed by the same bump -- the fork's comment claims the
    // generation "re-arms every lane", so the refusal lanes must be included.
    {
        fixture f;
        CHECK(f.attempts.refusal_changed(reason_t::mandatory_anchor, false));
        CHECK(!f.attempts.refusal_changed(reason_t::mandatory_anchor, false));
        // a different reason changes the latch state even within one generation
        CHECK(f.attempts.refusal_changed(reason_t::lifecycle_disabled, false));
        CHECK(!f.attempts.refusal_changed(reason_t::lifecycle_disabled, false));
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK(f.attempts.refusal_changed(reason_t::lifecycle_disabled, false));
    }
    {
        fixture f;
        CHECK(f.attempts.refusal_changed(reason_t::none, true));
        CHECK(!f.attempts.refusal_changed(reason_t::none, true));
        server_cache_checkpoint_ring_changed(f.ctx);
        CHECK(f.attempts.refusal_changed(reason_t::none, true));
    }

    // ---------------------------------------------------------------- oracle agreement
    // 512 independent rounds: the landed body and the restated intent must agree on
    // all three effects every time.
    {
        int disagreements = 0;
        for (int trial = 0; trial < 512; ++trial) {
            fixture a;
            fixture b;
            server_cache_checkpoint_ring_changed(a.ctx);
            oracle_ring_changed(b.attempts, b.seam, b.refusal);
            if (a.attempts.generation() != b.attempts.generation() ||
                a.seam != b.seam ||
                a.refusal != b.refusal) {
                ++disagreements;
            }
        }
        CHECK_EQ(disagreements, 0);
    }

    // ---------------------------------------------------------------- generation is monotone
    // The implementation guards a 64-bit wrap by clamping to 1 and clearing both
    // latch arrays. That branch cannot be reached from a test without 2^64 calls, so
    // it is NOT claimed here; what is checked is what is reachable: the generation
    // strictly increases and is never 0 across a few hundred ring changes.
    {
        fixture f;
        uint64_t prev = f.attempts.generation();
        CHECK(prev != 0);
        int violations = 0;
        for (int i = 0; i < 500; ++i) {
            server_cache_checkpoint_ring_changed(f.ctx);
            const uint64_t now = f.attempts.generation();
            if (now == 0 || now <= prev) {
                ++violations;
            }
            prev = now;
        }
        CHECK_EQ(violations, 0);
        CHECK_EQ(f.attempts.generation(), 501);
    }

    // ---------------------------------------------------------------- negative control
    // Prove the counter and the report path are really wired: inject one
    // deliberately-false check, confirm the failure count moved by exactly one and
    // that the failure branch reported (g_muted), then roll both counters back so the
    // injection never reaches the exit code or the reported total. This is the only
    // way to tell a `0 failures` line from a harness whose counter is not connected.
    // The injection is silenced (see g_report) so a green run prints no "FAIL" text.
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

    std::printf("test-cache-checkpoint-ring-changed: %d checks, %d failures\n",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
