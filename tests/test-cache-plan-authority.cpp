#include "server-cache-plan-authority.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(COND) do { if (!(COND)) { \
    std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #COND); \
    std::abort(); \
} } while (0)

static common_cache_plan_candidate * add_viable(
        common_cache_plan_record & rec,
        common_cache_plan_provider provider,
        int32_t source,
        int32_t target,
        common_cache_plan_selection origin =
            common_cache_plan_selection::by_id) {
    auto * row = rec.find_or_add(provider, source, 0, target, origin);
    CHECK(row != nullptr);
    row->accept();
    return row;
}

struct host_checkpoint_chain_fixture {
    common_cache_plan_candidate * host = nullptr;
    common_cache_plan_candidate * checkpoint = nullptr;
    common_cache_plan_candidate * chain = nullptr;
};

static host_checkpoint_chain_fixture add_host_checkpoint_chain(
        common_cache_plan_record & rec,
        int32_t target,
        int32_t host_source,
        int32_t checkpoint_ordinal,
        common_cache_plan_selection origin =
            common_cache_plan_selection::by_id) {
    host_checkpoint_chain_fixture out;
    out.host = add_viable(rec, common_cache_plan_provider::host_cache_entry,
        host_source, target, origin);
    out.checkpoint = add_viable(
        rec, common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(
            host_source, checkpoint_ordinal), target, origin);
    out.checkpoint->component_only = true;
    out.checkpoint->dependent_host_source_id = host_source;
    out.chain = rec.add_chain(
        common_cache_plan_provider::host_cache_entry,
        int32_t(out.host - rec.inventory.data()),
        int32_t(out.checkpoint - rec.inventory.data()));
    CHECK(out.chain != nullptr);
    out.chain->accept();
    return out;
}

static void test_candidate_classifiers() {
    CHECK(common_cache_plan_strict_similarity(0.75, 0.50));
    CHECK(!common_cache_plan_strict_similarity(0.50, 0.50));
    CHECK(!common_cache_plan_strict_similarity(0.75, 0.0));
    CHECK(common_cache_plan_origin_in_domain(
        common_cache_plan_selection::similarity,
        common_cache_plan_selection::similarity));
    CHECK(!common_cache_plan_origin_in_domain(
        common_cache_plan_selection::route_home,
        common_cache_plan_selection::similarity));

    const auto busy = server_cache_plan_evaluate_live(true, true, 8, 16);
    CHECK(busy.reason == COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY);
    const auto live = server_cache_plan_evaluate_live(false, true, 8, 16);
    CHECK(live.reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(live.lcp_tokens == 8);
    common_cache_plan_candidate live_row;
    server_cache_plan_apply_live(&live_row, live);
    CHECK(!live_row.f_keep_known);
    CHECK(live_row.f_keep == -1.0f);

    const auto host = server_cache_plan_evaluate_host(
        true, true, 20, 40, 80, 1024);
    CHECK(host.reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(host.f_keep == 0.25f);
    // A short positive match is still eligible: the source-retention fraction
    // is not a minimum reuse threshold. Checkpoint validity is checked separately.
    for (const uint64_t prefix : {uint64_t(1), uint64_t(128), uint64_t(500), uint64_t(7916)}) {
        CHECK(server_cache_plan_evaluate_host(
            true, true, prefix, 16384, 98609, 1024).reason ==
            COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    }
    CHECK(server_cache_plan_evaluate_host(
        true, true, 0, 16384, 98609, 1024).reason ==
        COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT);
    CHECK(server_cache_plan_evaluate_host(
        false, true, 1, 16384, 98609, 1024).reason ==
        COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY);
    CHECK(server_cache_plan_evaluate_host(
        true, false, 1, 16384, 98609, 1024).reason ==
        COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);
    CHECK(server_cache_plan_evaluate_host(
        true, false, 20, 40, 80, 1024).reason ==
        COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);

    const auto ckpt = server_cache_plan_evaluate_checkpoint(
        true, true, true, true, 30, 39, 40, 0, 512);
    CHECK(ckpt.reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(ckpt.lcp_tokens == 39);
    CHECK(server_cache_plan_evaluate_checkpoint(
        true, true, true, false, 30, 39, 40, 0, 512).reason ==
        COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED);
    CHECK(server_cache_plan_viable(ckpt.reason));
    CHECK(!server_cache_plan_viable(
        COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT));
    CHECK(server_cache_plan_host_checkpoint_source_id(2, 3) == 1020003);
}

static void test_checkpoint_orientation_and_host_identity() {
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 0) == 2);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 1) == 1);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 2) == 0);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 3) == -1);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 0, 7) ==
          server_cache_plan_host_checkpoint_source_id(7, 2));
    CHECK(server_cache_plan_checkpoint_ordinal_from_source_id(2) == 2);
    CHECK(server_cache_plan_checkpoint_ordinal_from_source_id(
              server_cache_plan_host_checkpoint_source_id(7, 2), 7) == 2);
    CHECK(server_cache_plan_checkpoint_ordinal_from_source_id(
              server_cache_plan_host_checkpoint_source_id(8, 2), 7) == -1);
    CHECK(server_cache_plan_checkpoint_reverse_position_from_source_id(
              3, server_cache_plan_host_checkpoint_source_id(7, 2), 7) == 0);

    common_cache_plan_record checkpoint_rec;
    auto * oldest = checkpoint_rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 0, 0, 5);
    auto * middle = checkpoint_rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 1, 0, 5);
    auto * newest = checkpoint_rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 2, 0, 5);
    CHECK(oldest && middle && newest);
    CHECK(checkpoint_rec.find_or_add(
              common_cache_plan_provider::live_context_checkpoint,
              server_cache_plan_checkpoint_source_id_from_reverse(3, 0),
              COMMON_CACHE_PLAN_PHASE_CKPT_SCAN, 5) == newest);

    int32_t next_source = 0;
    int32_t old_prefix_instance = -1;
    int32_t survivor_instance = -1;
    int32_t replacement_instance = -1;
    int32_t source = -1;
    CHECK(server_cache_plan_assign_source_id(
        old_prefix_instance, next_source, source));
    const int32_t old_prefix = source;
    CHECK(source == 0);
    CHECK(server_cache_plan_assign_source_id(
        survivor_instance, next_source, source));
    const int32_t survivor = source;
    CHECK(source == 1);
    // Save-time prefix dedup removes old_prefix. The surviving physical node
    // must keep its identity after its list ordinal shifts from 1 to 0.
    CHECK(server_cache_plan_assign_source_id(
        survivor_instance, next_source, source));
    CHECK(source == 1);
    // The freshly saved replacement cannot inherit either removed ordinal.
    CHECK(server_cache_plan_assign_source_id(
        replacement_instance, next_source, source));
    const int32_t replacement = source;
    CHECK(source == 2);

    common_cache_plan_record host_rec;
    auto * removed_row = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, old_prefix, 0, 6);
    auto * survivor_row = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, survivor, 0, 6);
    CHECK(removed_row && survivor_row && removed_row != survivor_row);
    // After dedup shifts the survivor to list ordinal zero, its immutable id
    // still rejoins its own pre-mutation row, never removed_row.
    CHECK(host_rec.find_or_add(
              common_cache_plan_provider::host_cache_entry, survivor,
              COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 6) == survivor_row);
    CHECK(host_rec.find_or_add(
              common_cache_plan_provider::host_cache_entry, replacement,
              COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 6) != survivor_row);
}

static void test_compose_excludes_destroyed_live_checkpoint() {
    common_cache_plan_record rec;
    auto * host = rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 4, 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(host != nullptr);
    host->accept();
    host->delivered = true;
    rec.select(common_cache_plan_provider::host_cache_entry, host);

    auto * destroyed_live = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 0, 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(destroyed_live != nullptr);
    destroyed_live->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);

    auto * host_checkpoint = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(4, 0), 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(host_checkpoint != nullptr);
    host_checkpoint->accept();
    host_checkpoint->delivered = true;
    host_checkpoint->component_only = true;
    host_checkpoint->dependent_host_source_id = host->source_id;
    rec.select(common_cache_plan_provider::live_context_checkpoint,
               host_checkpoint);
    rec.chosen = common_cache_plan_provider::live_context_checkpoint;

    const uint32_t before = rec.n_inventory;
    common_cache_plan_compose_chains(rec);
    CHECK(rec.n_inventory == before + 1);
    CHECK(!destroyed_live->component_only);
    CHECK(rec.inventory[before].component_ids[0] ==
          int32_t(host - rec.inventory.data()));
    CHECK(rec.inventory[before].component_ids[1] ==
          int32_t(host_checkpoint - rec.inventory.data()));
}

static void test_composed_chain_reuses_inventory_identity() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::similarity;

    const auto fixture = add_host_checkpoint_chain(
        rec, 2, 4, 0, common_cache_plan_selection::similarity);
    auto * host = fixture.host;
    auto * selected = fixture.checkpoint;
    auto * selected_chain = fixture.chain;
    host->accept();
    host->delivered = true;
    rec.select(common_cache_plan_provider::host_cache_entry, host);
    auto * sibling = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(4, 1), 0, 2,
        common_cache_plan_selection::similarity);
    auto * foreign = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(5, 0), 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(selected && sibling && foreign);
    selected->accept();
    selected->delivered = true;
    sibling->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    sibling->component_only = true;
    sibling->dependent_host_source_id = 4;
    foreign->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    foreign->component_only = true;
    foreign->dependent_host_source_id = 5;
    rec.select(common_cache_plan_provider::live_context_checkpoint, selected);
    rec.chosen = common_cache_plan_provider::live_context_checkpoint;

    auto * sibling_chain = rec.add_chain(
        common_cache_plan_provider::host_cache_entry,
        int32_t(host - rec.inventory.data()),
        int32_t(sibling - rec.inventory.data()));
    CHECK(sibling_chain);
    selected_chain->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    sibling_chain->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    const int32_t selected_chain_id =
        int32_t(selected_chain - rec.inventory.data());
    const uint32_t inventory_before = rec.n_inventory;

    // Finalize preserves the observed chain ordinal rather than appending a duplicate.
    common_cache_plan_compose_chains(rec);

    CHECK(rec.n_inventory == inventory_before);
    CHECK(rec.shipped_plan_candidate == selected_chain_id);
    CHECK(selected_chain->delivered);
    CHECK(selected_chain->disposition ==
          common_cache_plan_disposition::accepted);
    CHECK(!sibling_chain->delivered);
    CHECK(sibling_chain->disposition ==
          common_cache_plan_disposition::valid_not_chosen_cost);
}

static void test_inventory_saturation() {
    common_cache_plan_record rec;
    rec.n_prompt_tokens = llama_cache_acct_value::measured(32);
    for (size_t i = 0; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; ++i) {
        auto * row = rec.find_or_add(
            common_cache_plan_provider::cold_replay, int32_t(i), 0,
            int32_t(i), common_cache_plan_selection::lru);
        CHECK(row != nullptr);
        row->accept();
    }
    CHECK(!rec.inventory_saturated());
    CHECK(rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 1000, 0, 1000,
        common_cache_plan_selection::lru) == nullptr);
    CHECK(rec.inventory_saturated());
    CHECK(rec.inventory_states[size_t(
              common_cache_plan_provider::host_cache_entry)] ==
          common_cache_plan_inventory_state::overflowed);


}

static void test_legacy_candidate() {
    constexpr int32_t target = 2;
    common_cache_plan_record rec;
    auto * live = add_viable(
        rec, common_cache_plan_provider::live_slot, target, target);
    live->f_keep = 0.4; live->f_keep_known = true;
    live->sim = 0.4; live->sim_known = true;
    const auto fixture = add_host_checkpoint_chain(rec, target, 5, 1);
    auto * host = fixture.host;
    host->f_keep = 0.8; host->f_keep_known = true;
    host->sim = 0.7; host->sim_known = true;
    auto * chain = fixture.chain;
    add_viable(
        rec, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, target);

    const int32_t legacy = server_cache_plan_legacy_candidate(rec, target);
    CHECK(legacy == int32_t(chain - rec.inventory.data()));
    CHECK(server_cache_plan_legacy_candidate(rec, target, false) ==
          int32_t(live - rec.inventory.data()));

}

int main() {
    test_candidate_classifiers();
    test_checkpoint_orientation_and_host_identity();
    test_compose_excludes_destroyed_live_checkpoint();
    test_composed_chain_reuses_inventory_identity();
    test_inventory_saturation();
    test_legacy_candidate();
    return 0;
}
