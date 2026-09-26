#include "llama-kv-cells.h"

#include <cstdio>
#include <stdexcept>

static void test_swap() {
    static_assert(std::is_nothrow_swappable_v<llama_kv_cells>);
    llama_kv_cells a, b;
    a.resize(2);
    b.resize(130);
    a.pos_set(1, 10);
    a.ext_set(1, {11, 12, 13});
    a.seq_add(1, 0);
    a.pos_add(1, 3);
    b.pos_set(129, 20);
    b.seq_add(129, LLAMA_MAX_SEQ-1);
    const auto check = [&]() {
        if (b.size() != 2 || b.get_used() != 1 || b.used_min() != 1 || b.used_max_p1() != 2 ||
            !b.seq_has(1, 0) || b.seq_pos_min(0) != 13 || b.seq_pos_max(0) != 13 ||
            !b.get_has_shift() || b.get_shift(1) != 3 ||
            b.seq_pos_tok_le(0, 13) != 13 || b.ext_get(1).x != 11 || b.ext_get(1).y != 12 ||
            a.size() != 130 || a.get_used() != 1 || a.used_min() != 129 || a.used_max_p1() != 130 ||
            !a.seq_has(129, LLAMA_MAX_SEQ-1) || a.seq_pos_min(LLAMA_MAX_SEQ-1) != 20 ||
            a.get_has_shift()) {
            throw std::runtime_error("cell swap lost metadata");
        }
    };
    using std::swap;
    swap(a, b);
    check();
    b.swap(b);
    check();
    a.swap(b);
    swap(a, b);
    check();
    a.rm(129);
    b.rm(1);
    if (a.get_used() || b.get_used() || a.seq_pos_min(LLAMA_MAX_SEQ-1) != -1 || b.seq_pos_min(0) != -1) {
        throw std::runtime_error("swapped cell indices are inconsistent");
    }
}

int main() {
    try {
        test_swap();
        llama_kv_cells cells;
        cells.resize(1);
        const auto check = [&](const std::vector<llama_seq_id> & expected) {
            std::vector<llama_seq_id> actual;
            cells.seq_for_each(0, [&](llama_seq_id seq) { actual.push_back(seq); });
            if (actual != expected) { throw std::runtime_error("owner iteration differs"); }
        };
        check({});
        // Every singleton, including high IDs, must be visited exactly once.
        for (llama_seq_id seq = 0; seq < LLAMA_MAX_SEQ; ++seq) {
            cells.pos_set(0, 0);
            cells.seq_add(0, seq);
            check({seq});
            cells.seq_rm(0, seq);
            check({});
        }
        cells.pos_set(0, 0);
        for (llama_seq_id seq : {LLAMA_MAX_SEQ-1, 64, 0, 63}) { cells.seq_add(0, seq); }
        check({0, 63, 64, LLAMA_MAX_SEQ-1});
        cells.seq_rm(0, 64);
        check({0, 63, LLAMA_MAX_SEQ-1});
        cells.seq_rm(0, LLAMA_MAX_SEQ-1);
        check({0, 63});
        cells.seq_rm(0, 0);
        cells.seq_rm(0, 63);
        check({});
        std::puts("KV cell owner iteration PASS");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
