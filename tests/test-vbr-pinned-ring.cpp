// Tests for llama-vbr-pinned-ring.h/.cpp - the bounded pinned transport ring's creation contract.
//
// Hand-written rather than ported: upstream has no test for this file, and it is the one stage-2.6
// component that genuinely needs a ggml backend for its interesting behaviour. The pump itself
// (chunk leases, events, backpressure, the pump callbacks) allocates host-pinned buffers through
// ggml_backend_dev_host_buffer_type(), so it cannot run in a host-only harness that does not link
// ggml. What CAN be verified without a backend is everything that decides whether a ring may exist
// at all - and that ladder is where the load-bearing rules live:
//
//   - geometry, which is checked before anything is reserved;
//   - the accounting binding, which pins the ring to a ledger/budget/domain triple;
//   - the lane binding, which rejects a device/backend pair that disagrees;
//   - the process-wide capacity ceiling, and the fact that a failure anywhere below the reservation
//     releases it again (unique_ptr<impl> owns the reservation, so the leak check is a real one);
//   - the synthetic-lane success path, where both device and backend are null and the ring backs its
//     chunks with plain storage - which is the only path a host-only harness can drive to success.
//
// Deliberately not covered, because reaching them requires a real ggml device: duplicate lane
// detection (only reached for lanes with a non-null device), every host_buffer_* code,
// lane_underprovisioned and accounting_charge_failed. They are listed in MERGE_BUUN_PLAN §10.33 so
// the gap is visible rather than assumed.
//
// Self-reporting (status from main plus explicit PASS lines), so NDEBUG cannot hollow it out and it
// needs no /UNDEBUG.

#include "llama-vbr-pinned-ring.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

// A device/backend pointer is only ever *compared* on the paths taken here: the lane-binding
// rejection short-circuits before ggml_backend_get_device() is called, so a representative
// non-null value is enough and no backend is ever touched.
static ggml_backend_dev_t fake_device(int slot) {
    static const unsigned char slots[4] = { 0, 0, 0, 0 };
    return reinterpret_cast<ggml_backend_dev_t>(
        const_cast<unsigned char *>(&slots[slot]));
}

static ggml_backend_t fake_backend(int slot) {
    static const unsigned char slots[4] = { 0, 0, 0, 0 };
    return reinterpret_cast<ggml_backend_t>(
        const_cast<unsigned char *>(&slots[slot]));
}

// A lane with no device and no backend: the ring falls back to synthetic storage, which is the only
// success path a host-only harness can drive.
static std::vector<vbr_pinned_ring_lane> synthetic_lanes(size_t count) {
    return std::vector<vbr_pinned_ring_lane>(count, vbr_pinned_ring_lane{});
}

// Every creation attempt must report its failure code AND leave the process-wide capacity
// untouched. The reservation is owned by the half-built impl, so a failure that returned without
// unwinding would leak the ceiling for the whole process.
static void expect_failure(const std::vector<vbr_pinned_ring_lane> & lanes,
                           uint64_t total_bytes, size_t chunk_bytes,
                           const vbr_pinned_ring_accounting * accounting,
                           vbr_pinned_ring_create_failure expected,
                           const char * what) {
    const uint64_t before = vbr_pinned_ring_live_capacity_bytes();
    vbr_pinned_ring_create_failure failure = vbr_pinned_ring_create_failure::none;
    const auto ring = vbr_bounded_pinned_ring_core::create(
        lanes, total_bytes, chunk_bytes, accounting, failure);
    if (ring != nullptr) {
        throw std::runtime_error(std::string("creation succeeded when it should not have: ") + what);
    }
    if (failure != expected) {
        throw std::runtime_error(std::string("wrong failure code for ") + what + ": got " +
                                 std::to_string(static_cast<int>(failure)) + ", expected " +
                                 std::to_string(static_cast<int>(expected)));
    }
    check(vbr_pinned_ring_live_capacity_bytes() == before,
          (std::string("a failed creation leaked ring capacity: ") + what).c_str());
}

int main() {
    try {
        // [1] Geometry, checked before anything is reserved.
        {
            check(VBR_PINNED_RING_MAX_BYTES == 256ull*1024*1024,
                  "the process-wide ring ceiling changed");
            const std::vector<vbr_pinned_ring_lane> one = synthetic_lanes(1);
            const std::vector<vbr_pinned_ring_lane> two = synthetic_lanes(2);

            expect_failure({}, 4096, 4096, nullptr,
                           vbr_pinned_ring_create_failure::invalid_geometry, "no lanes");
            expect_failure(one, 4096, 0, nullptr,
                           vbr_pinned_ring_create_failure::invalid_geometry, "zero chunk size");
            expect_failure(one, 0, 4096, nullptr,
                           vbr_pinned_ring_create_failure::invalid_geometry, "zero total bytes");
            expect_failure(one, VBR_PINNED_RING_MAX_BYTES + 4096, 4096, nullptr,
                           vbr_pinned_ring_create_failure::invalid_geometry,
                           "total above the process ceiling");
            // Fewer than two chunks per lane: a ring that cannot hold one chunk in flight and one
            // being filled is not a ring.
            expect_failure(two, 8192, 4096, nullptr,
                           vbr_pinned_ring_create_failure::invalid_geometry,
                           "one chunk for two lanes");
            expect_failure(one, 4096, 4096, nullptr,
                           vbr_pinned_ring_create_failure::invalid_geometry,
                           "a single chunk for a single lane");
            std::puts("VBR pinned ring PASS: geometry rejection");
        }

        // [2] Accounting binding. The domain must be pinned-host residency so the ledger rows a ring
        // charges are the ones the budget actually prices.
        {
            const std::vector<vbr_pinned_ring_lane> one = synthetic_lanes(1);
            llama_cache_acct_ledger ledger;
            const llama_cache_budget_config budget {};
            const auto pinned = llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pinned_host);
            const auto pageable = llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host);

            vbr_pinned_ring_accounting no_ledger {};
            no_ledger.budget  = &budget;
            no_ledger.domain  = pinned;
            expect_failure(one, 8192, 4096, &no_ledger,
                           vbr_pinned_ring_create_failure::invalid_accounting_binding,
                           "a binding with no ledger");

            vbr_pinned_ring_accounting no_budget {};
            no_budget.ledger = &ledger;
            no_budget.domain = pinned;
            expect_failure(one, 8192, 4096, &no_budget,
                           vbr_pinned_ring_create_failure::invalid_accounting_binding,
                           "a binding with no budget");

            vbr_pinned_ring_accounting wrong_residency {};
            wrong_residency.ledger = &ledger;
            wrong_residency.budget = &budget;
            wrong_residency.domain = pageable;
            expect_failure(one, 8192, 4096, &wrong_residency,
                           vbr_pinned_ring_create_failure::invalid_accounting_binding,
                           "a binding on pageable-host residency");

            // A properly bound accounting still fails when the ledger has no certified row for the
            // ring's category/domain: an unpriced ring is never admitted.
            vbr_pinned_ring_accounting bound {};
            bound.ledger   = &ledger;
            bound.budget   = &budget;
            bound.domain   = pinned;
            bound.category = llama_cache_acct_category::pinned_preimage_ring;
            expect_failure(one, 8192, 4096, &bound,
                           vbr_pinned_ring_create_failure::accounting_update_failed,
                           "a binding against an uncertified ledger");
            std::puts("VBR pinned ring PASS: accounting binding rejection");
        }

        // [3] Lane binding. A device and a backend must agree: neither may be set alone, because the
        // ring submits on the backend it was given and waits on the device it was given.
        {
            vbr_pinned_ring_lane device_only {};
            device_only.device = fake_device(0);
            expect_failure({ device_only }, 8192, 4096, nullptr,
                           vbr_pinned_ring_create_failure::invalid_lane_binding,
                           "a lane with a device but no backend");

            vbr_pinned_ring_lane backend_only {};
            backend_only.backend = fake_backend(1);
            expect_failure({ backend_only }, 8192, 4096, nullptr,
                           vbr_pinned_ring_create_failure::invalid_lane_binding,
                           "a lane with a backend but no device");

            // The rejection must fire even when a later lane is fine, and it must not depend on the
            // order the lanes were given in.
            vbr_pinned_ring_lane good {};
            expect_failure({ good, device_only }, 16384, 4096, nullptr,
                           vbr_pinned_ring_create_failure::invalid_lane_binding,
                           "a bad lane after a good one");
            std::puts("VBR pinned ring PASS: lane binding rejection");
        }

        // [4] Synthetic-lane success and the process-wide capacity ceiling.
        {
            const std::vector<vbr_pinned_ring_lane> two = synthetic_lanes(2);
            const uint64_t before = vbr_pinned_ring_live_capacity_bytes();

            vbr_pinned_ring_create_failure failure = vbr_pinned_ring_create_failure::none;
            auto ring = vbr_bounded_pinned_ring_core::create(two, 16384, 4096, nullptr, failure);
            check(ring != nullptr, "a synthetic-lane ring was refused");
            check(failure == vbr_pinned_ring_create_failure::none,
                  "a successful creation still reported a failure code");
            check(ring->capacity_bytes() == 16384, "the ring rounded its capacity");
            check(ring->chunk_bytes() == 4096, "the ring lost its chunk size");
            check(ring->lane_count() == 2, "the ring lost its lane count");
            check(vbr_pinned_ring_live_capacity_bytes() == before + 16384,
                  "a live ring did not appear in the process-wide ceiling");

            // Capacity is rounded down to whole chunks, and that rounded figure - not the requested
            // byte count - is what the ceiling charges.
            vbr_pinned_ring_create_failure rounded = vbr_pinned_ring_create_failure::none;
            auto partial = vbr_bounded_pinned_ring_core::create(two, 16384 + 1, 4096, nullptr, rounded);
            check(partial != nullptr, "a non-multiple total was refused");
            check(partial->capacity_bytes() == 16384, "a non-multiple total was not rounded down");
            check(vbr_pinned_ring_live_capacity_bytes() == before + 32768,
                  "the ceiling was charged the requested bytes instead of the rounded capacity");

            partial.reset();
            check(vbr_pinned_ring_live_capacity_bytes() == before + 16384,
                  "destroying a ring did not release its ceiling");
            ring.reset();
            check(vbr_pinned_ring_live_capacity_bytes() == before,
                  "destroying every ring did not drain the ceiling");

            // The ceiling itself: 128 MiB live, then ask for more than the remainder. This is the
            // only failure code that needs a fixture large enough to touch the ceiling, so it is
            // the one place this test allocates real memory.
            {
                constexpr uint64_t chunk = 1ull << 20;
                constexpr uint64_t half  = VBR_PINNED_RING_MAX_BYTES / 2;
                vbr_pinned_ring_create_failure big = vbr_pinned_ring_create_failure::none;
                auto held = vbr_bounded_pinned_ring_core::create(two, half, size_t(chunk), nullptr, big);
                check(held != nullptr, "the ceiling fixture could not be allocated");
                check(held->capacity_bytes() == half, "the ceiling fixture was not sized as asked");
                expect_failure(two, half + chunk, size_t(chunk), nullptr,
                               vbr_pinned_ring_create_failure::global_capacity_exceeded,
                               "a request past the remaining ceiling");
                held.reset();
                check(vbr_pinned_ring_live_capacity_bytes() == before,
                      "the ceiling fixture did not release its capacity");
            }
            std::puts("VBR pinned ring PASS: synthetic lanes and capacity ceiling");
        }

        // [5] Move-only ownership tokens. They are what stop the two direction adapters from
        // consuming each other's outstanding chunks, and what releases the ring when dropped.
        {
            vbr_pinned_ring_operation empty;
            check(!empty, "a default operation token reports as set");
            {
                vbr_pinned_ring_operation moved = std::move(empty);
                check(!moved, "moving an empty token produced a set token");
                // A moved-from empty token still destructs and is still assignable: the reset path
                // must be a no-op when there is no owner.
                vbr_pinned_ring_operation target;
                target = std::move(moved);
                check(!target, "move-assigning an empty token produced a set token");
                target = std::move(target);  // self-assignment must not release anything
                check(!target, "self-move-assignment broke the token");
            }

            vbr_pinned_chunk_lease lease;
            check(!lease, "a default chunk lease reports as set");
            check(lease.data() == nullptr && lease.capacity() == 0 && lease.valid() == 0,
                  "a default chunk lease exposes nonzero state");
            {
                vbr_pinned_chunk_lease moved = std::move(lease);
                check(!moved, "moving an empty lease produced a set lease");
                check(moved.data() == nullptr && moved.capacity() == 0 && moved.valid() == 0,
                      "a moved empty lease exposed state");
                vbr_pinned_chunk_lease target;
                target = std::move(moved);
                check(!target, "move-assigning an empty lease produced a set lease");
                target = std::move(target);
                check(!target, "self-move-assignment broke the lease");
            }
            std::puts("VBR pinned ring PASS: move-only ownership tokens");
        }

        std::puts("VBR pinned ring tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "VBR pinned ring FAIL: %s\n", error.what());
        return 1;
    }
}
