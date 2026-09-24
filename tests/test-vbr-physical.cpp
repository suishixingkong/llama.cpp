#include "../src/llama-vbr-physical.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <utility>
#include <vector>

using llama_vbr_physical::interval;
using llama_vbr_physical::projection;

static void expect_intervals(
        const std::vector<interval> & got,
        const std::vector<std::pair<size_t, size_t>> & expected) {
    assert(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        assert(got[i].begin == expected[i].first);
        assert(got[i].end == expected[i].second);
    }
}

static void test_deferred_ranges_match_full_page_unmap() {
    constexpr size_t g = 16;
    std::vector<interval> got;
    assert(llama_vbr_physical::normalize_deferred({
        {  1, 30 }, // no complete page
        { 16, 32 }, // [16, 48)
        { 32, 32 }, // overlaps, extending through 64
        { 64, 16 }, // adjacent: same union
        { 97, 14 }, // no complete page
        {112, 32 }, // separate range
    }, g, got));
    expect_intervals(got, { {16, 80}, {112, 144} });

    assert(!llama_vbr_physical::normalize_deferred({}, 0, got));
}

struct page_set {
    size_t granularity;
    std::set<size_t> pages;

    size_t operator()(size_t off, size_t len) const {
        assert(off % granularity == 0 && len % granularity == 0);
        size_t count = 0;
        for (size_t page : pages) {
            if (page >= off && page < off + len) {
                count++;
            }
        }
        return count * granularity;
    }
};

static void test_deferred_union_is_subtracted_once() {
    constexpr size_t g = 16;
    const page_set resident = { g, {0, 16, 32, 48, 80} };
    std::vector<interval> deferred;
    assert(llama_vbr_physical::normalize_deferred({
        {16, 32}, // pages 16 and 32
        {32, 32}, // pages 32 and 48; page 32 must not be double-counted
    }, g, deferred));

    size_t mapped = 0;
    assert(llama_vbr_physical::mapped_after_deferred(
            0, 96, g, deferred, resident, mapped));
    assert(mapped == 2*g); // pages 0 and 80 remain
}

static void test_deferred_absent_pages_do_not_reduce_residency() {
    constexpr size_t g = 16;
    const page_set resident = { g, {0, 48} };
    std::vector<interval> deferred;
    assert(llama_vbr_physical::normalize_deferred({ {16, 32} }, g, deferred));

    size_t mapped = 0;
    assert(llama_vbr_physical::mapped_after_deferred(
            0, 64, g, deferred, resident, mapped));
    assert(mapped == 2*g);

    // Clearing a no-op deferred queue does not bump the backend residency epoch.  That is
    // cache-safe because the projected state is byte-identical with or without the queue.
    const std::vector<interval> cleared;
    size_t mapped_after_clear = 0;
    assert(llama_vbr_physical::mapped_after_deferred(
            0, 64, g, cleared, resident, mapped_after_clear));
    assert(mapped_after_clear == mapped);
}

static void test_checked_endpoint_arithmetic_and_zero_endpoint() {
    uint64_t bytes = 1;
    assert(llama_vbr_physical::endpoint_bytes(300, 0, 8192, 2048, bytes));
    assert(bytes == 0);
    assert(llama_vbr_physical::endpoint_bytes(300, 10, 8192, 2048, bytes));
    assert(bytes == 4096);
    assert(llama_vbr_physical::endpoint_bytes(300, 40, 8192, 2048, bytes));
    assert(bytes == 8192);
    assert(!llama_vbr_physical::endpoint_bytes(
            std::numeric_limits<uint64_t>::max(), 2, 8192, 2048, bytes));
    assert(!llama_vbr_physical::endpoint_bytes(
            std::numeric_limits<uint64_t>::max() - 1, 1,
            std::numeric_limits<uint64_t>::max(), 4096, bytes));
    assert(!llama_vbr_physical::endpoint_bytes(1, 1, 8191, 2048, bytes));

    const page_set resident = { 16, {0} };
    const std::vector<interval> none;
    size_t mapped = 1;
    assert(llama_vbr_physical::mapped_after_deferred(
            0, 0, 16, none, resident, mapped));
    assert(mapped == 0);
}

static void test_endpoint_reports_release_growth_and_signed_delta() {
    constexpr uint64_t g = 2u << 20;
    projection p;

    // A partially resident endpoint: two existing tail pages disappear, while one hole
    // inside the terminal prefix must be grown.
    assert(llama_vbr_physical::add_endpoint(p, 4*g, 2*g, 3*g));
    assert(p.release == 2*g);
    assert(p.growth  == 1*g);
    assert(p.delta   == (int64_t) g);

    // Equal gross release and growth produces zero physical capacity even though pages move.
    assert(llama_vbr_physical::add_endpoint(p, 2*g, 1*g, 2*g));
    assert(p.release == 3*g);
    assert(p.growth  == 2*g);
    assert(p.delta   == (int64_t) g);

    projection growth_only;
    assert(llama_vbr_physical::add_endpoint(growth_only, 2*g, 2*g, 4*g));
    assert(growth_only.release == 0);
    assert(growth_only.growth  == 2*g);
    assert(growth_only.delta   == -(int64_t) (2*g));
}

static void test_invalid_physical_inputs_are_rejected() {
    projection p;
    assert(!llama_vbr_physical::add_endpoint(p, 1, 2, 2));
    assert(!llama_vbr_physical::add_endpoint(p, 2, 2, 1));

    size_t mapped = 0;
    const page_set resident = { 16, {0} };
    const std::vector<interval> none;
    assert(!llama_vbr_physical::mapped_after_deferred(
            1, 16, 16, none, resident, mapped));

    projection aggregate_overflow;
    aggregate_overflow.release = (uint64_t) std::numeric_limits<int64_t>::max();
    aggregate_overflow.delta   = std::numeric_limits<int64_t>::max();
    assert(!llama_vbr_physical::add_endpoint(aggregate_overflow, 1, 0, 0));

    projection zero_endpoint;
    assert(llama_vbr_physical::add_endpoint(zero_endpoint, 16, 0, 0));
    assert(zero_endpoint.release == 16);
    assert(zero_endpoint.growth == 0);
    assert(zero_endpoint.delta == 16);
}

int main() {
    test_deferred_ranges_match_full_page_unmap();
    test_deferred_union_is_subtracted_once();
    test_deferred_absent_pages_do_not_reduce_residency();
    test_checked_endpoint_arithmetic_and_zero_endpoint();
    test_endpoint_reports_release_growth_and_signed_delta();
    test_invalid_physical_inputs_are_rejected();
    std::cout << "VBR physical endpoint tests passed\n";
    return 0;
}
