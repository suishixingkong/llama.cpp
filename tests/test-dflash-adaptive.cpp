#include "../common/speculative-dflash-adaptive.h"

#include <cassert>
#include <cstdio>

// Synthetic cycle costs isolate policy from GPU timing noise. Live gates still
// measure the search overhead, which is deliberately not hidden in these tests.
static int settle(common_speculative_dflash_adaptive & policy, int maximum, int winner) {
    for (int i = 0; i < 100; ++i) {
        const int depth = policy.depth(maximum);
        const int accepted = depth > 0 ? std::min(depth, winner > 2 ? 6 : 1) : 0;
        const int tokens = accepted + 1;
        policy.observe((depth == winner ? 5000 : 12000) * tokens, depth, accepted, tokens);
        if (!policy.searching()) {
            return policy.depth(maximum);
        }
    }
    assert(false);
    return -1;
}

int main() {
    // Reuse hardware cost, never a prior request's acceptance or chosen depth.
    common_speculative_dflash_adaptive::calibration timing;
    for (int i = 0; i < 3; ++i) {
        timing.record(12, 100000);
        timing.record(7, 40000);
        timing.record(2, 30000);
        timing.record(0, 20000);
    }
    common_speculative_dflash_adaptive warm;
    for (int i = 0; i < 4; ++i) {
        assert(warm.depth(12, 1, &timing) == 12);
        warm.observe(100000, 12, 6, 7);
    }
    assert(warm.depth(12, 1, &timing) == 7 && !warm.searching());
    common_speculative_dflash_adaptive fresh;
    assert(fresh.depth(12, 1, &timing) == 12 && fresh.searching());
    // An entirely rejected observed prefix must extinguish the tail allowance.
    // The prior is not permission to fabricate acceptance in low-match prose.
    common_speculative_dflash_adaptive reject;
    for (int i = 0; i < 4; ++i) {
        reject.depth(12, 1, &timing);
        reject.observe(100000, 12, 0, 1);
    }
    assert(reject.depth(12, 1, &timing) == 2);
    common_speculative_dflash_adaptive::calibration recovery_timing;
    for (int i = 0; i < 3; ++i) {
        recovery_timing.record(12, 24000);
        recovery_timing.record(7, 15000);
        recovery_timing.record(2, 10000);
        recovery_timing.record(0, 20000);
    }
    common_speculative_dflash_adaptive censored;
    for (int i = 0; i < 40 && censored.depth(12, 1, &recovery_timing) != 2; ++i) {
        const int d = censored.depth(12, 1, &recovery_timing);
        censored.observe(d == 12 ? 24000 : 15000, d, 1, 2);
    }
    assert(censored.depth(12, 1, &recovery_timing) == 2);
    for (int i = 0; i < 16; ++i) {
        const int accepted = i % 2 ? 2 : 0;
        censored.observe(10000, 2, accepted, accepted + 1);
    }
    // Only 50% acceptance at depth two, but successful prefixes could conceal
    // enough accepted tail tokens to make the known depth-seven cost worthwhile.
    const int recovered = censored.depth(12, 1, &recovery_timing);
    assert(recovered >= 7);
    censored.observe(recovered == 7 ? 15000 : 24000, recovered, recovered, recovered + 1);
    assert(!censored.searching());
    for (int maximum : {0, 1, 2, 3, 4, 7, 12}) {
        for (int winner : {0, 2, 4, 7, 12}) {
            if (winner == 0 && maximum != 0) { continue; }
            if (winner > maximum) { continue; }
            if (winner == 4 && maximum != 4) { continue; }
            common_speculative_dflash_adaptive policy;
            assert(settle(policy, maximum, winner) == winner);
        }
    }
    // Finding middle depth must not depend on poor acceptance at full depth.
    common_speculative_dflash_adaptive policy;
    assert(settle(policy, 12, 7) == 7);
    // A changed request ceiling invalidates the old search; never exceed it.
    assert(policy.depth(2) == 2);
    assert(settle(policy, 2, 2) == 2);
    // Fully accepted shallow drafts trigger exploration rather than permanent
    // depth-2 lock-in. Rejected drafts are a separate phase-change signal.
    common_speculative_dflash_adaptive shallow;
    assert(settle(shallow, 12, 2) == 2);
    for (int i = 0; i < 16; ++i) {
        shallow.observe(15000, 2, 2, 3);
    }
    // Recovery measures the unseen positions immediately; it must not spend
    // another incumbent window at two after the high-match window just did so.
    assert(shallow.depth(12) > 2);
    // An explicit zero ceiling must not prevent recovery when the caller
    // changes its limit. It is not an automatically selected off-ramp.
    common_speculative_dflash_adaptive off;
    assert(settle(off, 0, 0) == 0);
    for (int i = 0; i < 32; ++i) { off.observe(12000, 0, 0, 1); }
    assert(off.searching());
    assert(settle(off, 12, 12) == 12);
    // Clipped/failed attempts do not price the configured shape.
    common_speculative_dflash_adaptive clipped;
    assert(clipped.depth(12) == 12);
    for (int i = 0; i < 100; ++i) { clipped.observe(10, 1, 1, 2); }
    assert(clipped.depth(12) == 12);
    assert(settle(clipped, 12, 7) == 7);
    // The depth-7 arm encounters an unusually easy sentence. Pooling prefix
    // observations must not mistake that difference for a faster kernel.
    common_speculative_dflash_adaptive uneven;
    for (int i = 0; i < 100; ++i) {
        const int depth = uneven.depth(12);
        const int accepted = depth == 7 ? 3 : depth == 0 || depth == 12 ? 0 : 1;
        const int time = depth == 12 ? 80000 : depth == 7 ? 50000 :
                         depth == 4 ? 45000 : depth == 2 ? 30000 : 20000;
        uneven.observe(time, depth, accepted, accepted + 1);
        if (!uneven.searching()) { break; }
    }
    assert(uneven.depth(12) == 2 && !uneven.searching());
    // High-match short responses avoid a tour of smaller batches. A periodic
    // recheck must still test middle depth even when acceptance stays perfect.
    common_speculative_dflash_adaptive code;
    for (int i = 0; i < 20; ++i) {
        const int depth = code.depth(12);
        assert(depth == 12 || depth == 7);
        code.observe(depth == 12 ? 100000 : 45000, depth, depth, depth + 1);
        if (!code.searching()) { break; }
    }
    assert(code.depth(12) == 12 && !code.searching());
    for (int i = 0; i < 256; ++i) { code.observe(100000, 12, 12, 13); }
    assert(code.searching());
    for (int i = 0; i < 20; ++i) {
        const int depth = code.depth(12);
        code.observe(depth == 12 ? 100000 : 45000, depth, depth, depth + 1);
        if (!code.searching()) { break; }
    }
    assert(code.depth(12) == 7 && !code.searching());
    // Falling acceptance is reversible too; a perfect code window doesn't
    // entitle the controller to keep its depth throughout subsequent prose.
    for (int i = 0; i < 8; ++i) { code.observe(45000, 7, 0, 1); }
    assert(code.searching());
    assert(settle(code, 12, 2) == 2);
    common_speculative_dflash_adaptive minimum;
    for (int i = 0; i < 40; ++i) {
        const int depth = minimum.depth(12, 5);
        assert(depth == 0 || depth >= 5);
        minimum.observe(10000, depth, 0, 1);
    }
    // Independent slots must not share learned depths.
    assert(policy.depth(2) == 2);
    common_speculative_dflash_adaptive reset;
    assert(reset.depth(12) == 12);
    // Exercise changing limits, request resets, warm timing reuse, CopySpec
    // progress, and clipped proposals independently of GPU scheduling.
    common_speculative_dflash_adaptive walk;
    common_speculative_dflash_adaptive::calibration walk_timing;
    uint32_t random = 17;
    int maximum = 12, floor = 1;
    for (int i = 0; i < 20000; ++i) {
        random = random * 1664525u + 1013904223u;
        if (i % 500 == 0) {
            maximum = (random >> 16) % 25;
            floor = 1 + random % std::max(1, maximum);
            walk = {};
        }
        const int depth = walk.depth(maximum, floor, &walk_timing);
        assert(depth >= 0 && depth <= maximum);
        assert(maximum == 0 ? depth == 0 : depth >= floor);
        const int accepted = random % (depth + 1);
        if (i % 7 == 0) {
            walk.observe(1, depth - 1, std::max(0, accepted - 1), accepted + 1);
        } else {
            walk.observe(5000 + random % 30000, depth, accepted, accepted + 1 + i % 3);
        }
        for (const auto & sample : walk_timing.costs) {
            assert(sample.count == 0 || sample.us >= 5000);
        }
    }
    std::puts("DFlash2 adaptive policy tests passed");
}
