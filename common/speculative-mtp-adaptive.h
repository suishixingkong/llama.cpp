#pragma once

#include <algorithm>

// fork 9l-mtp-adaptive: ported from buun-llama-cpp common/speculative-mtp-adaptive.h.
// Reversible per-request draft-depth control for the single-head MTP drafter.
//
// Why this exists at all: the depth is a *bet*. Every drafted token costs a full
// draft decode, and the target only accepts the prefix it agrees with -- often
// none. When the bet loses, the step is slower than a plain decode, which is why
// an unguarded MTP run shows a wide throughput band (measured on the V100:
// 140..220 t/s) and a floor *below* the non-speculative baseline. The fork
// measured this and added the controller below; upstream master has no such
// mechanism, so this is a genuine fork-only feature, not an upstream behaviour.
//
// Keep the established conservative 3 -> 2 decision, but make it reversible
// within a request. Depth 1 did not earn its exploration cost in serving tests.
class common_speculative_mtp_adaptive {
public:
    explicit common_speculative_mtp_adaptive(int minimum = 1)
        : minimum_depth(std::max(2, std::min(3, minimum))) {}

    int depth() const { return cap; }

    void reset() { *this = common_speculative_mtp_adaptive(minimum_depth); }

    // Keep a learned depth across requests, but treat the new prefix as an
    // opportunity to recover. Restarting every request at 3 needlessly repeats
    // the losing probe on a stream of low-match prose requests.
    void begin() {
        attempts = full = prefix_full = full_streak = 0;
        // Preserve the periodic countdown too: many short requests must not
        // postpone a full-depth probe indefinitely.
    }

    void accept(int drafted, int accepted, bool other) {
        // Clipped/failed drafts and another implementation's proposals do not
        // measure our selected depth. Duplicate carry refreshes have no draft.
        if (other || drafted != cap || accepted < 0 || accepted > drafted || minimum_depth == 3) {
            return;
        }

        if (cap == 2) {
            full_streak = accepted == 2 ? full_streak + 1 : 0;
            // A matching streak signals a phase change only if the first two
            // positions were not already near-perfect in the full-depth probe.
            // Periodic probes also recover when no such streak is observed.
            if (--hold == 0 || (prefix_full < 12 && full_streak >= 8)) {
                reset();
            }
            return;
        }

        full += accepted == 3;
        prefix_full += accepted >= 2;
        if (++attempts == 16) {
            if (full < 8) {
                cap = 2;
                hold = 256;
            } else {
                prefix_full = 0;
            }
            attempts = full = 0;
        }
    }

private:
    int minimum_depth;
    int cap = 3;
    int attempts = 0;
    int full = 0;
    int prefix_full = 0;
    int full_streak = 0;
    int hold = 0;
};
