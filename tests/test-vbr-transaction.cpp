#include "../src/llama-vbr-transaction.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

using llama_vbr_transaction::device_cost;
using llama_vbr_transaction::workspace_request;

static device_cost priced(
        uint64_t release,
        uint64_t kv_growth,
        uint64_t scratch_growth,
        uint64_t workspace_growth,
        uint64_t stash_growth) {
    device_cost result { release, kv_growth, scratch_growth, workspace_growth, stash_growth, 0 };
    assert(llama_vbr_transaction::finalize(result));
    return result;
}

static void test_shortest_prefix_accounts_for_activation_discontinuity() {
    // One nominal 10-page KV hop cannot satisfy an 8-page target when the first tapped hop
    // activates a 3-page workspace and 2-page stash.  The next hop makes the prefix feasible.
    std::map<int, device_cost> first = {{ 0, priced(10, 0, 0, 3, 2) }};
    assert(!llama_vbr_transaction::prefix_feasible(first, 0, 8));
    std::map<int, device_cost> second = {{ 0, priced(16, 0, 0, 3, 2) }};
    assert(llama_vbr_transaction::prefix_feasible(second, 0, 8));
}

static void test_collateral_capacity_is_signed() {
    std::map<int, device_cost> costs = {
        { 0, priced(20, 2, 1, 1, 0) },
        { 1, priced( 2, 3, 1, 0, 0) },
    };
    assert(costs.at(0).capacity_signed == 16);
    assert(costs.at(1).capacity_signed == -2);
    assert(!llama_vbr_transaction::prefix_feasible(costs, 0, 8));

    costs.at(1) = priced(5, 3, 1, 0, 0);
    assert(llama_vbr_transaction::prefix_feasible(costs, 0, 8));
}

static void test_workspace_projection_uses_only_real_tuples() {
    const std::vector<workspace_request> requests = {
        { 4096, 1024,   0 },
        {  256, 8192, 128 },
    };
    bool saw_synthetic = false;
    uint64_t now = 0;
    uint64_t endpoint = 0;
    assert(llama_vbr_transaction::workspace_endpoint(
            requests,
            [&](const workspace_request & request, uint64_t & physical_now, uint64_t & projected) {
                saw_synthetic = saw_synthetic ||
                        (request.n_cells == 4096 && request.ne0 == 8192 && request.stash_rows == 128);
                physical_now = 7;
                projected = (uint64_t) request.n_cells + (uint64_t) request.ne0 +
                        (uint64_t) request.stash_rows;
                return true;
            }, now, endpoint));
    assert(!saw_synthetic);
    assert(now == 7);
    assert(endpoint == 8576);
}

static void test_distributed_grant_amortizes_one_aggregate_credit() {
    // Reproduction of the pre-transaction row-local scheme: both 50-byte rows start at the same
    // bytes_now, so one +50 update clears 100 bytes of decrement.
    const uint64_t old_remaining =
            llama_vbr_transaction::grant_row_remaining(50, 0, 50) +
            llama_vbr_transaction::grant_row_remaining(50, 0, 50);
    assert(old_remaining == 0);

    uint64_t first = 0;
    uint64_t second = 0;
    assert(llama_vbr_transaction::grant_threshold(0, 0, first));
    assert(llama_vbr_transaction::grant_threshold(0, 50, second));
    const uint64_t aggregate_remaining =
            llama_vbr_transaction::grant_row_remaining(50, first, 50) +
            llama_vbr_transaction::grant_row_remaining(50, second, 50);
    assert(aggregate_remaining == 50);
}

int main() {
    test_shortest_prefix_accounts_for_activation_discontinuity();
    test_collateral_capacity_is_signed();
    test_workspace_projection_uses_only_real_tuples();
    test_distributed_grant_amortizes_one_aggregate_credit();
    std::cout << "VBR atomic transaction tests passed\n";
    return 0;
}
