// Tests for llama-vbr-extent.h/.cpp - the committed-extent store and the dual-view ownership index.
//
// Hand-written rather than ported: upstream has no test for this file at all (the only upstream
// VBR tests are hard-seal, physical, policy, transaction, downward, representation-identity,
// representation-epoch, artifact*, swa-window and vmm), and the epoch test is the nearest thing to
// a consumer of the ownership index - but it reaches llama-kv-cache.h inside a single 765-line
// function. See BUUN_MERGE_PLAN §10.33.
//
// Two independent oracles are used, because a test that re-derives its expectation from the code
// under test proves nothing:
//
//   - the extent store is a four-state machine (prepared -> submitted -> committed/failed, with
//     ABA-safe generation validation), so every transition is asserted against the documented rule
//     rather than against whatever the implementation happens to return;
//   - the ownership index keeps a Fenwick tree over logical positions and a page-mask set over
//     cells. Both are derived views of one ground truth, so this test drives a deterministic
//     sequence of add/remove/move operations against a plain std::map oracle and compares
//     rank_below() and enumerate_owned() against it after every step. An off-by-one in the Fenwick
//     index arithmetic shows up immediately; nothing in the test recomputes the tree.
//
// Self-reporting (status from main plus explicit PASS lines), so NDEBUG cannot hollow it out and it
// needs no /UNDEBUG.

#include "llama-vbr-extent.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

// ------------------------------------------------------------------ oracle

// The ground truth the index's two derived views must both agree with: one entry per owned cell,
// carrying the logical position the index was last told about.
using oracle_map = std::map<uint32_t, llama_pos>;

static std::vector<uint32_t> oracle_cells(const oracle_map & owned) {
    std::vector<uint32_t> cells;
    cells.reserve(owned.size());
    for (const auto & entry : owned) {
        cells.push_back(entry.first);
    }
    return cells;
}

static uint32_t oracle_rank(const oracle_map & owned, uint32_t frontier) {
    uint32_t rank = 0;
    for (const auto & entry : owned) {
        if (static_cast<uint32_t>(entry.second) < frontier) {
            ++rank;
        }
    }
    return rank;
}

int main() {
    try {
        // [1] Extent store: lifecycle, field round-trip, reclaim on last release.
        {
            check(sizeof(vbr_extent_entry) == 24, "extent entry is not 24 bytes");
            check(vbr_extent_store::CAPACITY == 8192, "extent store capacity changed");

            vbr_extent_store store;
            check(store.live_entries() == 0, "a fresh store already has live entries");
            check(!store.exhausted_latched(), "a fresh store reports exhaustion");
            check(store.lookup_committed(vbr_extent_ref{}) == nullptr,
                  "the default reference resolved to an entry");
            check(!store.submit(vbr_extent_handle{}), "the default handle was submitted");
            check(!store.commit(vbr_extent_handle{}), "the default handle was committed");
            check(!store.fail(vbr_extent_handle{}), "the default handle was failed");
            check(!store.add_ref(vbr_extent_handle{}), "the default handle was referenced");

            // reset_all() from generation 0 rebases every slot to an even (resolved) generation, so
            // a default-constructed handle can never validate. The first reservation is odd.
            const vbr_extent_handle first = store.reserve(
                vbr_mutation_family::append, vbr_operation_class::ordinary_decode, 0, 2, 5, 9);
            check(bool(first), "the first reservation was refused");
            check(first.index == 0, "the first reservation did not take the first free slot");
            check(first.expected_gen == 3, "the first reservation did not bump to an odd generation");
            check(store.live_entries() == 1, "a live reservation was not counted");

            const vbr_extent_handle second = store.reserve(
                vbr_mutation_family::trim, vbr_operation_class::explicit_destructive_trim, 1, 3, 1, 2);
            check(bool(second) && second.index == 1 && second.expected_gen == 3,
                  "the second reservation did not take the next slot");

            // A committed entry is visible only through a reference, and only while the generation
            // still matches. The reference has to be taken before the commit: committing an
            // uncited entry reclaims it immediately.
            const vbr_extent_ref ref = store.add_ref(first);
            check(bool(ref) && ref.index == first.index && ref.expected_gen == first.expected_gen,
                  "add_ref did not mint a matching reference");
            check(store.lookup_committed(ref) == nullptr,
                  "a prepared entry was visible as committed evidence");
            check(store.submit(first), "a prepared entry refused to submit");
            check(store.lookup_committed(ref) == nullptr,
                  "a submitted entry was visible as committed evidence");
            check(store.commit(first), "a submitted entry refused to commit");
            const vbr_extent_entry * entry = store.lookup_committed(ref);
            check(entry != nullptr, "a committed, cited entry was not visible as evidence");
            check(entry->state == vbr_extent_state::committed, "the committed state was not stored");
            check(entry->family == vbr_mutation_family::append &&
                      entry->operation_class == vbr_operation_class::ordinary_decode &&
                      entry->stream == 0 && entry->seq_id == 2 &&
                      entry->p0 == 5 && entry->p1 == 9,
                  "the reservation did not round-trip its full extent");
            check(entry->slot_gen == first.expected_gen, "the entry lost its slot generation");

            // Releasing the last reference reclaims the slot, which bumps the generation and
            // therefore retires both the handle and the reference in one step.
            store.release_ref(ref);
            check(store.lookup_committed(ref) == nullptr,
                  "a released reference still resolved to evidence");
            check(!store.fail(first), "a reclaimed handle was still valid");
            check(store.live_entries() == 1, "reclaim did not return the slot to the free list");

            // A failed entry is never evidence, and it is reclaimable the same way.
            const vbr_extent_ref second_ref = store.add_ref(second);
            check(store.fail(second), "a prepared entry refused to fail");
            check(store.lookup_committed(second_ref) == nullptr,
                  "a failed entry was visible as committed evidence");
            store.release_ref(second_ref);
            check(store.live_entries() == 0, "the store did not drain after both entries resolved");
            std::puts("VBR extent PASS: store lifecycle and reclaim");
        }

        // [2] Extent store: the transition matrix.
        {
            vbr_extent_store store;
            const vbr_extent_handle handle = store.reserve(
                vbr_mutation_family::import, vbr_operation_class::host_import, 0, -1, -1, -1);
            check(bool(handle), "the transition fixture could not reserve");
            const vbr_extent_ref ref = store.add_ref(handle);

            check(store.submit(handle), "submit was refused while prepared");
            check(!store.submit(handle), "submit was accepted twice");
            check(store.commit(handle), "commit was refused while submitted");
            check(!store.submit(handle), "submit was accepted after commit");
            check(!store.commit(handle), "commit was accepted twice");
            check(store.lookup_committed(ref) != nullptr, "the committed entry vanished");
            check(store.fail(handle), "fail was refused from committed");
            check(store.lookup_committed(ref) == nullptr, "a failed entry stayed as evidence");
            check(!store.commit(handle), "commit was accepted after fail");

            store.release_ref(ref);
            check(store.live_entries() == 0, "the transition fixture did not drain");

            // commit() is also legal straight from prepared (the non-async families).
            const vbr_extent_handle direct = store.reserve(
                vbr_mutation_family::clear, vbr_operation_class::state_api, 0, 0, 0, 1);
            const vbr_extent_ref direct_ref = store.add_ref(direct);
            check(store.commit(direct), "commit was refused from prepared");
            check(store.lookup_committed(direct_ref) != nullptr,
                  "a committed-from-prepared entry was not evidence");
            store.release_ref(direct_ref);
            std::puts("VBR extent PASS: transition matrix");
        }

        // [3] Extent store: stale handles and stale references after slot reuse (the ABA contract).
        {
            vbr_extent_store store;
            const vbr_extent_handle original = store.reserve(
                vbr_mutation_family::append, vbr_operation_class::ordinary_decode, 0, 0, 0, 4);
            const vbr_extent_ref original_ref = store.add_ref(original);
            check(store.fail(original), "the ABA fixture refused to fail");
            store.release_ref(original_ref);
            check(store.live_entries() == 0, "the ABA slot was not reclaimed");

            // The same slot comes back with a bumped generation, so every value minted before the
            // reuse must be inert - in particular a stale release must not decrement the new
            // occupant's reference count.
            const vbr_extent_handle reused = store.reserve(
                vbr_mutation_family::append, vbr_operation_class::ordinary_decode, 0, 0, 6, 8);
            check(bool(reused) && reused.index == original.index,
                  "the free list did not hand the reclaimed slot back");
            check(reused.expected_gen != original.expected_gen,
                  "slot reuse did not bump the generation");
            check(!store.fail(original), "a stale handle was still valid after reuse");
            check(!store.add_ref(original), "a stale handle still minted references");
            check(store.lookup_committed(original_ref) == nullptr,
                  "a stale reference resolved against the new occupant");

            const vbr_extent_ref fresh_ref = store.add_ref(reused);
            check(bool(fresh_ref), "the new occupant could not be referenced");
            store.release_ref(original_ref);  // must be a no-op
            check(store.submit(reused) && store.commit(reused),
                  "a stale release disturbed the new occupant");
            check(store.lookup_committed(fresh_ref) != nullptr,
                  "a stale release retired the new occupant's reference");
            store.release_ref(fresh_ref);
            check(store.live_entries() == 0, "the ABA fixture did not drain");
            std::puts("VBR extent PASS: ABA safety across slot reuse");
        }

        // [4] Extent store: exhaustion latch and global-reset rebase.
        {
            vbr_extent_store store;
            for (uint32_t i = 0; i < vbr_extent_store::CAPACITY; ++i) {
                const vbr_extent_handle handle = store.reserve(
                    vbr_mutation_family::append, vbr_operation_class::ordinary_decode, 0, 0, 0, 1);
                check(bool(handle), "the store filled up early");
                check(!store.exhausted_latched(),
                      "exhaustion latched while slots were still available");
            }
            check(store.live_entries() == vbr_extent_store::CAPACITY,
                  "the store did not report full occupancy");

            // An off-side transaction passes latch_exhaustion = false: the refusal must leave both
            // the return value and the latch alone...
            check(!store.reserve(vbr_mutation_family::append,
                                 vbr_operation_class::ordinary_decode, 0, 0, 0, 1, false),
                  "a full store handed out a slot");
            check(!store.exhausted_latched(),
                  "an off-side refusal latched exhaustion although it was told not to");

            // ...while a normal reservation latches, because the caller now has to take the
            // invalidate-before-mutate path.
            const vbr_extent_handle refused = store.reserve(
                vbr_mutation_family::append, vbr_operation_class::ordinary_decode, 0, 0, 0, 1);
            check(!refused, "a full store handed out a slot");
            check(store.exhausted_latched(), "a normal refusal did not latch exhaustion");

            // The global reset rebases every slot past the outstanding generations, so a reference
            // taken before the reset can never validate afterwards.
            const vbr_extent_ref stale = vbr_extent_ref{ 0, 3 };
            store.reset_all();
            check(store.live_entries() == 0, "reset_all did not reclaim every slot");
            check(!store.exhausted_latched(), "reset_all did not clear the exhaustion latch");
            check(store.lookup_committed(stale) == nullptr,
                  "a pre-reset reference validated after the global rebase");
            const vbr_extent_handle after = store.reserve(
                vbr_mutation_family::append, vbr_operation_class::ordinary_decode, 0, 0, 0, 1);
            check(bool(after) && after.expected_gen != stale.expected_gen,
                  "reset_all did not rebase the slot generation");
            std::puts("VBR extent PASS: exhaustion latch and global rebase");
        }

        // [5] Ownership index against the oracle: randomized add/remove/move, checked after every
        // step. Positions deliberately cover a domain larger than the cell count so that several
        // cells share a position and the rank is not a proxy for the owned count.
        {
            constexpr uint32_t N_CELLS     = 512;
            constexpr uint32_t N_POSITIONS = 700;
            constexpr uint32_t N_STEPS     = 4000;

            vbr_ownership_index index(1, 2, N_CELLS, N_POSITIONS);
            oracle_map owned;
            std::mt19937 rng(0x5eed1234u);
            std::uniform_int_distribution<uint32_t> cell_dist(0, N_CELLS - 1);
            std::uniform_int_distribution<uint32_t> pos_dist(0, N_POSITIONS - 1);
            std::uniform_int_distribution<uint32_t> op_dist(0, 99);

            check(!index.initialized(0, 0), "a view is initialized before its first cell");
            check(!index.available(0, 0), "an unpopulated view reports available");

            // Seed the view: remove_cell/move_cell on a view that was never populated are refusals,
            // which is correct but would make the first randomly chosen operation meaningless.
            check(index.add_cell(0, 0, 0, 0), "the oracle view could not be seeded");
            owned[0] = 0;

            for (uint32_t step = 0; step < N_STEPS; ++step) {
                const uint32_t cell = cell_dist(rng);
                const uint32_t op   = op_dist(rng);
                const auto found = owned.find(cell);
                if (op < 60) {
                    const llama_pos pos = static_cast<llama_pos>(pos_dist(rng));
                    const bool was_owned = owned.find(cell) != owned.end();
                    check(index.add_cell(0, 0, cell, pos), "an in-domain add was refused");
                    // Idempotent: re-adding an owned cell is accepted and must NOT move it, so the
                    // oracle keeps the position the cell was first added with.
                    check(index.add_cell(0, 0, cell, pos), "a repeated in-domain add was refused");
                    if (!was_owned) {
                        owned[cell] = pos;
                    }
                } else if (op < 85) {
                    const llama_pos pos = found == owned.end()
                                              ? static_cast<llama_pos>(pos_dist(rng))
                                              : found->second;
                    check(index.remove_cell(0, 0, cell, pos), "an in-domain remove was refused");
                    owned.erase(cell);
                } else {
                    const llama_pos old_pos = found == owned.end()
                                                  ? static_cast<llama_pos>(pos_dist(rng))
                                                  : found->second;
                    const llama_pos new_pos = static_cast<llama_pos>(pos_dist(rng));
                    check(index.move_cell(0, 0, cell, old_pos, new_pos),
                          "an in-domain move was refused");
                    if (found != owned.end()) {
                        owned[cell] = new_pos;
                    }
                }

                // Both derived views must agree with the oracle exactly.
                std::vector<uint32_t> cells;
                check(index.enumerate_owned(0, 0, cells),
                      "a populated view reported itself unavailable");
                check(cells == oracle_cells(owned), "enumerate_owned diverged from the oracle");
                check(index.available(0, 0), "a populated view reported unavailable");
                check(index.initialized(0, 0), "a populated view reported uninitialized");

                // The frontier is resampled rather than fixed, so the comparison covers the whole
                // domain instead of one boundary.
                const uint32_t frontier = pos_dist(rng);
                uint32_t rank = 0xffffffffu;
                check(index.rank_below(0, 0, static_cast<llama_pos>(frontier), rank),
                      "rank_below refused an in-domain frontier");
                check(rank == oracle_rank(owned, frontier), "rank_below diverged from the oracle");
            }

            // Frontier clamping: the domain bound is n_positions, so anything at or above it is
            // the full rank, and an unknown (-1) frontier is refused outright.
            uint32_t rank = 0xffffffffu;
            check(index.rank_below(0, 0, static_cast<llama_pos>(N_POSITIONS), rank) &&
                      rank == static_cast<uint32_t>(owned.size()),
                  "rank at the domain bound is not the full owned count");
            check(index.rank_below(0, 0, static_cast<llama_pos>(N_POSITIONS) + 100, rank) &&
                      rank == static_cast<uint32_t>(owned.size()),
                  "rank above the domain bound was not clamped");
            check(index.rank_below(0, 0, 0, rank) && rank == 0,
                  "rank below position 0 is not zero");
            check(!index.rank_below(0, 0, -1, rank), "rank_below accepted an unknown frontier");
            check(index.rank_below(0, 0, static_cast<llama_pos>(N_POSITIONS), rank) &&
                      rank == oracle_rank(owned, N_POSITIONS),
                  "clamped rank disagrees with the oracle");
            std::puts("VBR extent PASS: ownership index against the oracle");
        }

        // [6] Ownership index: fail-closed domain restriction, availability, clear and clone.
        {
            constexpr uint32_t N_CELLS     = 1024;
            constexpr uint32_t N_POSITIONS = 64;

            vbr_ownership_index index(1, 2, N_CELLS, N_POSITIONS);
            check(index.position_capacity() == N_POSITIONS, "the logical domain is not exposed");
            check(index.add_cell(0, 0, 0, 0), "a first cell was refused");
            check(index.add_cell(0, 0, 1, 1), "a second cell was refused");

            // Priced while one destination view is still unallocated and one is out of domain.
            {
                const size_t populated = index.clone_storage_bytes(0, 0);
                const size_t untouched = index.clone_storage_bytes(0, 1);
                const size_t outside   = index.clone_storage_bytes(9, 0);
                check(populated > 0, "clone_storage_bytes priced a populated index at zero");
                check(untouched > populated,
                      "clone_storage_bytes does not account for a lazily allocated destination");
                check(outside == untouched,
                      "an out-of-domain destination was not priced like an unallocated one");
            }

            // Out-of-domain requests are refused up front - the physical arguments without
            // touching the view, and a logical position by marking the whole view unavailable.
            check(!index.add_cell(0, 0, N_CELLS, 5), "a cell beyond n_cells was accepted");
            check(!index.add_cell(1, 0, 0, 5), "a stream beyond n_stream was accepted");
            check(index.available(0, 0), "a refused physical add disabled a usable view");
            check(!index.add_cell(0, 0, 2, static_cast<llama_pos>(N_POSITIONS)),
                  "a position at the domain bound was accepted");
            check(!index.available(0, 0),
                  "an out-of-domain position did not mark the view unavailable");
            uint32_t rank = 0;
            std::vector<uint32_t> cells;
            check(!index.rank_below(0, 0, 5, rank),
                  "an unavailable view still produced a logical rank");
            check(!index.enumerate_owned(0, 0, cells),
                  "an unavailable view still enumerated its cells");
            check(index.initialized(0, 0), "an unavailable view reported uninitialized");

            // A view that was never populated is distinct from one that failed the domain contract.
            check(!index.initialized(0, 1) && !index.available(0, 1),
                  "an untouched view is not distinguishable from an unavailable one");
            check(!index.rank_below(0, 1, 5, rank), "an untouched view produced a rank");
            check(!index.remove_cell(0, 1, 0, 0), "remove_cell on an untouched view returned false");
            check(!index.move_cell(0, 1, 0, 0, 1), "move_cell on an untouched view returned false");

            // The physical domain is checked before the sequence domain, so a bad cell on an
            // untouched view is still a refusal rather than an accidental population.
            check(!index.remove_cell(0, 1, N_CELLS, 0), "remove_cell accepted a cell beyond n_cells");
            check(!index.initialized(0, 1), "a refused remove populated a view");

            // remove_cell of a cell the view does not own is a no-op that reports success, and a
            // move of an unowned cell likewise changes nothing.
            check(index.remove_cell(0, 0, 999, 0), "removing an unowned cell was refused");
            check(index.move_cell(0, 0, 999, 0, 1), "moving an unowned cell was refused");

            // clear_seq forgets the view but keeps the slot reusable, and clear_all resets every
            // slot at once.
            index.clear_seq(0, 0);
            check(!index.initialized(0, 0), "clear_seq left the view initialized");
            check(!index.available(0, 0), "clear_seq left the view available");
            check(index.add_cell(0, 0, 7, 7), "a cleared view could not be repopulated");
            std::vector<uint32_t> repopulated;
            check(index.enumerate_owned(0, 0, repopulated) &&
                      repopulated == std::vector<uint32_t>({ 7 }),
                  "a repopulated view kept cells from its previous occupation");

            check(index.add_cell(0, 1, 3, 3), "the source index stopped accepting cells");
            std::unique_ptr<vbr_ownership_index> clone = index.clone();
            check(clone != nullptr, "clone returned nothing");
            check(clone->position_capacity() == index.position_capacity(),
                  "the clone has a different logical domain");
            std::vector<uint32_t> cloned_cells;
            check(clone->enumerate_owned(0, 0, cloned_cells) &&
                      cloned_cells == std::vector<uint32_t>({ 7 }),
                  "the clone disagrees about the owned cells");
            check(clone->rank_below(0, 0, static_cast<llama_pos>(N_POSITIONS), rank) && rank == 1,
                  "the clone disagrees about the rank");
            check(clone->add_cell(0, 0, 9, 9), "the clone refused a cell");
            std::vector<uint32_t> source_cells;
            check(index.enumerate_owned(0, 0, source_cells) &&
                      source_cells == std::vector<uint32_t>({ 7 }),
                  "writing to the clone changed the source index");
            index.clear_all();
            check(!index.initialized(0, 0) && !index.initialized(0, 1),
                  "clear_all left views initialized");
            check(clone->initialized(0, 0) && clone->initialized(0, 1),
                  "clear_all reached into a clone");

            // The sequence domain is validated by remove_cell/move_cell but not by add_cell, whose
            // lazy allocation goes through an unchecked index. In the production configuration
            // (n_stream == 1, which armed VBR forces) the out-of-domain index is out of the slot
            // array, so the attempt cannot silently land in a real view - which is the invariant
            // asserted here. The multi-stream aliasing hazard this asymmetry implies is recorded in
            // BUUN_MERGE_PLAN §10.33 rather than encoded as expected behaviour.
            bool accepted = false;
            bool threw = false;
            try {
                accepted = index.add_cell(0, static_cast<llama_seq_id>(2), 0, 0);
            } catch (const std::exception &) {
                threw = true;
            }
            check(!accepted, "add_cell accepted an out-of-domain sequence id");
            if (threw) {
                std::puts("VBR extent NOTE: add_cell throws for an out-of-domain sequence id, "
                          "while remove_cell/move_cell fail closed");
            }
            std::puts("VBR extent PASS: domain restriction, availability, clear and clone");
        }

        std::puts("VBR extent tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "VBR extent FAIL: %s\n", error.what());
        return 1;
    }
}
