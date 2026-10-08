// fork 9l-mtp-adaptive: the fork's MTP draft-depth controller, ported with its test.
// Source of both: buun-llama-cpp common/speculative-mtp-adaptive.h and its
// tests/test-sampling.cpp::test_mtp_adaptive(). The test body is kept verbatim so a future
// re-sync against the fork is a diff, not a re-derivation.
//
// This is a pure host test: the class has no dependencies beyond <algorithm>, so it runs
// without a model, a backend, or a GPU. What it pins is the *decision table* (3 -> 2 after a
// losing probe, recovery on a streak or on the periodic countdown, no measurement of clipped
// or foreign drafts) -- i.e. exactly the part of the MTP throughput fix that can regress
// silently, since the throughput itself is only observable on a device.

#include "speculative-mtp-adaptive.h"

#include "ggml.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cstdio>

static void test_mtp_adaptive() {
    common_speculative_mtp_adaptive state;
    auto cycles = [&](int n, int accepted) {
        for (int i = 0; i < n; ++i) {
            state.accept(state.depth(), accepted, false);
            GGML_ASSERT(state.depth() >= 2 && state.depth() <= 3);
        }
    };
    cycles(64, 3); // high-match code retains the full depth
    GGML_ASSERT(state.depth() == 3);
    cycles(16, 0);
    GGML_ASSERT(state.depth() == 2);
    cycles(7, 2);
    GGML_ASSERT(state.depth() == 2);
    cycles(1, 2); // prose -> code, recover without waiting for another request
    GGML_ASSERT(state.depth() == 3);
    cycles(16, 3);
    GGML_ASSERT(state.depth() == 3);

    cycles(16, 2); // perfect first two rows but an unhelpful third
    GGML_ASSERT(state.depth() == 2);
    cycles(255, 2); // not a phase change: do not repeatedly probe every 8 cycles
    GGML_ASSERT(state.depth() == 2);
    cycles(1, 2); // bounded periodic recovery, even without a new streak
    GGML_ASSERT(state.depth() == 3);
    cycles(16, 0);
    cycles(256, 0); // periodic recovery also works with no accepted proposals
    GGML_ASSERT(state.depth() == 3);

    for (int i = 0; i < 32; ++i) {
        state.accept(2, 0, false); // clipped draft
        state.accept(0, 0, false); // failed/duplicate carry refresh
        state.accept(3, 0, true);  // another implementation
        state.accept(3, 4, false); // invalid count
    }
    GGML_ASSERT(state.depth() == 3);
    cycles(15, 0);
    GGML_ASSERT(state.depth() == 3);
    cycles(1, 0);
    GGML_ASSERT(state.depth() == 2);
    state.begin(); // learned depth survives, but a new request can recover
    GGML_ASSERT(state.depth() == 2);
    cycles(8, 2);
    GGML_ASSERT(state.depth() == 3);

    for (int matched = 7; matched <= 8; ++matched) {
        state.reset();
        cycles(matched, 3);
        cycles(16 - matched, 0);
        GGML_ASSERT(state.depth() == (matched == 7 ? 2 : 3));
    }
    for (int prefix = 11; prefix <= 12; ++prefix) {
        state.reset();
        cycles(prefix, 2);
        cycles(16 - prefix, 0);
        cycles(8, 2);
        GGML_ASSERT(state.depth() == (prefix == 11 ? 3 : 2));
    }
    state.reset();
    cycles(15, 0);
    state.begin(); // partial probe cannot leak into the next request
    cycles(1, 0);
    GGML_ASSERT(state.depth() == 3);

    state.reset();
    cycles(16, 0);
    for (int i = 0; i < 255; ++i) {
        state.begin();
        cycles(1, 0);
        GGML_ASSERT(state.depth() == 2);
    }
    state.begin();
    cycles(1, 0); // even one-token requests cannot postpone periodic recovery
    GGML_ASSERT(state.depth() == 3);

    common_speculative_mtp_adaptive slots[2];
    for (int i = 0; i < 16; ++i) {
        slots[0].accept(3, 0, false);
        slots[1].accept(3, 3, false);
    }
    GGML_ASSERT(slots[0].depth() == 2 && slots[1].depth() == 3);
    for (int i = 0; i < 8; ++i) {
        // Same prefix clamp as MTP's CopySpec-composition integration.
        const int drafted = slots[0].depth();
        slots[0].accept(drafted, std::min(3, drafted), false);
    }
    GGML_ASSERT(slots[0].depth() == 3 && slots[1].depth() == 3);

    for (int minimum = 0; minimum <= 3; ++minimum) {
        state = common_speculative_mtp_adaptive(minimum);
        for (int i = 0; i < 1024; ++i) {
            if (i == 512) {
                state.reset();
            }
            int drafted = state.depth();
            if (drafted < minimum) {
                drafted = 0; // production minimum-size boundary
            }
            GGML_ASSERT(drafted > 0);
            state.accept(drafted, 0, false);
            if (minimum == 3) {
                GGML_ASSERT(state.depth() == 3);
            }
        }
    }
}

// The port passes the drafter's own --spec-draft-n-min (default 0), not the class default.
// Pin the two construction paths the production code actually uses, so a future change to
// that argument cannot silently freeze the controller at depth 3 (which is what a
// minimum_depth of 3 does inside accept()).
static void test_production_construction_paths() {
    common_speculative_mtp_adaptive default_constructed;           // class default: minimum = 1
    common_speculative_mtp_adaptive from_n_min_zero(0);            // common_params_speculative_draft::n_min == 0
    for (auto * state : { &default_constructed, &from_n_min_zero }) {
        GGML_ASSERT(state->depth() == 3);
        for (int i = 0; i < 16; ++i) {
            state->accept(3, 0, false);
        }
        GGML_ASSERT(state->depth() == 2);
    }
}

int main() {
    test_mtp_adaptive();
    test_production_construction_paths();
    printf("OK\n");
    return 0;
}
