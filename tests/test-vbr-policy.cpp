#include "../src/llama-vbr-policy.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using llama_vbr_policy::child;
using llama_vbr_policy::result;
using llama_vbr_policy::selection;
using llama_vbr_policy::shortest_prefix_stream;
using llama_vbr_policy::step;

static step gain(int64_t bytes, size_t order = 0) {
    return { order, order, 1, 2, bytes };
}

static std::vector<size_t> drain(std::vector<child> children) {
    shortest_prefix_stream stream(std::move(children));
    std::vector<size_t> result_children;
    selection s;
    for (;;) {
        const result r = stream.next(s);
        if (r == result::exhausted) {
            return result_children;
        }
        assert(r == result::selected);
        result_children.push_back(s.child_index);
    }
}

// Literal state-copy reference for the existing mutation-while-sizing selector: choose the least
// current logical-progress / full-terminal-progress ratio, then mutate that child's progress.
static std::vector<size_t> reference_mutating_order(std::vector<child> children) {
    std::vector<size_t> next(children.size(), 0);
    std::vector<int64_t> progress;
    progress.reserve(children.size());
    for (const auto & c : children) {
        progress.push_back(c.initial_progress);
    }
    std::vector<size_t> out;
    for (;;) {
        size_t pick = children.size();
        for (size_t i = 0; i < children.size(); ++i) {
            if (children[i].terminal_progress <= 0 || next[i] >= children[i].steps.size()) {
                continue;
            }
            if (pick == children.size()) {
                pick = i;
                continue;
            }
            const uint64_t ni = progress[i] > 0 ? (uint64_t) progress[i] : 0;
            const uint64_t np = progress[pick] > 0 ? (uint64_t) progress[pick] : 0;
#if defined(__SIZEOF_INT128__)
            if ((__uint128_t) ni * (uint64_t) children[pick].terminal_progress <
                (__uint128_t) np * (uint64_t) children[i].terminal_progress) {
#else
            if ((double) ni / (double) children[i].terminal_progress <
                (double) np / (double) children[pick].terminal_progress) {
#endif
                pick = i;
            }
        }
        if (pick == children.size()) {
            return out;
        }
        int64_t updated = 0;
        assert(llama_vbr_policy::checked_add(
                progress[pick], children[pick].steps[next[pick]].logical_gain, updated));
        progress[pick] = updated;
        next[pick]++;
        out.push_back(pick);
    }
}

static void test_root_first_stable_tie() {
    // The caller supplies root first.  In the successor's SWA-root topology that is SWA, so SWA
    // must win every exact proportional-progress tie without an allocator-dependent tiebreak.
    child swa_root = { 0, 100, { gain(25, 0), gain(25, 1), gain(50, 2) }, {} };
    child base_peer = { 0, 200, { gain(50, 0), gain(50, 1), gain(100, 2) }, {} };
    const std::vector<size_t> got = drain({ swa_root, base_peer });
    const std::vector<size_t> expected = { 0, 1, 0, 1, 0, 1 };
    assert(got == expected);
}

static void test_negative_initial_progress_is_clamped_only_for_ranking() {
    child root = { -50, 50, { gain(50, 0), gain(50, 1) }, {} };
    child peer = {   0, 50, { gain(25, 0), gain(25, 1) }, {} };
    // Both initially rank as zero, so root wins.  Its signed progress becomes zero, producing
    // another exact tie that root wins again.  Only then can peer advance.
    const std::vector<size_t> got = drain({ root, peer });
    const std::vector<size_t> expected = { 0, 0, 1, 1 };
    assert(got == expected);
}

static void test_shortest_prefix_callback_runs_after_each_real_step() {
    shortest_prefix_stream stream({
        { 0, 100, { gain(25, 0), gain(25, 1), gain(50, 2) }, {} },
        { 0, 100, { gain(25, 0), gain(25, 1), gain(50, 2) }, {} },
    });
    std::vector<selection> prefix;
    size_t prices = 0;
    const result r = stream.shortest_prefix([&](const std::vector<selection> & p) {
        prices++;
        // Stand-in for later physical repricing/collateral checks.
        return p.size() == 3;
    }, prefix);
    assert(r == result::selected);
    assert(prices == 3);
    assert(prefix.size() == 3);
    assert(prefix[0].child_index == 0);
    assert(prefix[1].child_index == 1);
    assert(prefix[2].child_index == 0);
}

static void test_checked_page_padded_logical_progress() {
    uint64_t bytes = 0;
    assert(llama_vbr_policy::logical_endpoint_bytes(300, 10, 8192, 2048, bytes));
    assert(bytes == 4096);
    assert(llama_vbr_policy::logical_endpoint_bytes(300, 40, 8192, 2048, bytes));
    assert(bytes == 8192); // fixed slot cap
    assert(llama_vbr_policy::logical_endpoint_bytes(300, 0, 8192, 2048, bytes));
    assert(bytes == 0);
    assert(!llama_vbr_policy::logical_endpoint_bytes(
            std::numeric_limits<uint64_t>::max(), 2, 8192, 2048, bytes));
    assert(!llama_vbr_policy::logical_endpoint_bytes(1, 1, 8192, 0, bytes));
}

static void test_checked_progress_overflow_and_invalid_gain() {
    shortest_prefix_stream overflow({
        { std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), { gain(1) }, {} },
    });
    selection s;
    assert(overflow.next(s) == result::overflow);
    assert(overflow.selected().empty());

    shortest_prefix_stream invalid({ { 0, 1, { gain(-1) }, {} } });
    assert(invalid.next(s) == result::invalid);
}

static void test_fraction_comparison_never_cross_multiplies() {
    assert(llama_vbr_policy::fraction_less(0, 1, 1, 1));
    assert(!llama_vbr_policy::fraction_less(1, 1, 1, 1));
    assert(llama_vbr_policy::fraction_less(
            std::numeric_limits<uint64_t>::max() - 1,
            std::numeric_limits<uint64_t>::max(),
            std::numeric_limits<uint64_t>::max(),
            std::numeric_limits<uint64_t>::max()));

#if defined(__SIZEOF_INT128__)
    std::mt19937_64 rng(0x6275756eULL);
    for (size_t i = 0; i < 100000; ++i) {
        const uint64_t a = rng();
        const uint64_t b = rng() | 1;
        const uint64_t c = rng();
        const uint64_t d = rng() | 1;
        const bool reference = (__uint128_t) a * d < (__uint128_t) c * b;
        assert(llama_vbr_policy::fraction_less(a, b, c, d) == reference);
    }
#endif
}

static void test_equivalence_with_mutating_reference() {
    std::mt19937_64 rng(0x706f6c696379ULL);
    for (size_t trial = 0; trial < 10000; ++trial) {
        std::vector<child> children;
        for (size_t ci = 0; ci < 2; ++ci) {
            child c;
            c.initial_progress = -(int64_t) (rng() % 50);
            const size_t n = 1 + rng() % 8;
            int64_t total = c.initial_progress;
            for (size_t si = 0; si < n; ++si) {
                // Gapped order indices model entries skipped by the authoritative vbr_sim_step
                // adapter; only real steps reach this pure stream.
                const int64_t delta = 1 + rng() % 1000;
                c.steps.push_back(gain(delta, si*3 + ci));
                assert(llama_vbr_policy::checked_add(total, delta, total));
            }
            c.terminal_progress = total;
            children.push_back(std::move(c));
        }
        assert(drain(children) == reference_mutating_order(children));
    }
}

// Focused reproduction of the integration hazard: if current physical residency is used as the
// proportional numerator, identical logical ladders choose a different first victim after a
// partial-map history.  The pure stream has no residency input, so both histories stay root-first.
// The e3c7002 live executor predates the tree stream; this is intentionally a regression model,
// not a claim that this clean base currently executes the contaminated selector.
static size_t contaminated_first_pick(uint64_t root_resident_progress, uint64_t peer_resident_progress) {
    return llama_vbr_policy::fraction_less(
            peer_resident_progress, 100, root_resident_progress, 100) ? 1 : 0;
}

static void test_allocator_history_contamination_reproduction() {
    assert(contaminated_first_pick(60, 0) == 1);
    assert(contaminated_first_pick(0, 60) == 0);

    const child root = { 0, 100, { gain(50) }, {} };
    const child peer = { 0, 100, { gain(50) }, {} };
    const auto history_a = drain({ root, peer });
    const auto history_b = drain({ root, peer });
    assert(history_a == history_b);
    assert(!history_a.empty() && history_a[0] == 0);
}

static void test_nonpositive_terminal_child_is_ineligible() {
    child ineligible = { -100, 0, { gain(100) }, {} };
    child root = { 0, 10, { gain(10) }, {} };
    const auto got = drain({ root, ineligible });
    assert(got.size() == 1 && got[0] == 0);
}

int main() {
    test_root_first_stable_tie();
    test_negative_initial_progress_is_clamped_only_for_ranking();
    test_shortest_prefix_callback_runs_after_each_real_step();
    test_checked_page_padded_logical_progress();
    test_checked_progress_overflow_and_invalid_gain();
    test_fraction_comparison_never_cross_multiplies();
    test_equivalence_with_mutating_reference();
    test_allocator_history_contamination_reproduction();
    test_nonpositive_terminal_child_is_ineligible();
    std::cout << "VBR pure policy stream tests passed\n";
    return 0;
}
