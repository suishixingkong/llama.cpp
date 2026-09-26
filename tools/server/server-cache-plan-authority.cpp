#include "server-cache-plan-authority.h"


int32_t server_cache_plan_host_source(
        const common_cache_plan_record & rec,
        int32_t candidate) noexcept {
    if (candidate < 0 || uint32_t(candidate) >= rec.n_inventory) {
        return -1;
    }
    const auto & row = rec.inventory[size_t(candidate)];
    if (row.is_chain()) {
        const int32_t base = row.component_ids[0];
        return base >= 0 && uint32_t(base) < rec.n_inventory
            ? rec.inventory[size_t(base)].source_id : -1;
    }
    return row.provider == common_cache_plan_provider::host_cache_entry
        ? row.source_id : -1;
}

common_cache_plan_destruction_effect_set server_cache_destruction_effects_for(
        const common_cache_plan_record & rec,
        int32_t candidate,
        int32_t legacy_candidate,
        common_cache_plan_destruction_effect_set permitted_effects) noexcept {
    if (candidate < 0 || legacy_candidate < 0 ||
        uint32_t(candidate) >= rec.n_inventory ||
        uint32_t(legacy_candidate) >= rec.n_inventory) {
        return 0;
    }
    const auto & planned = rec.inventory[size_t(candidate)];
    const auto & legacy = rec.inventory[size_t(legacy_candidate)];
    common_cache_plan_destruction_effect_set effects = 0;
    if (planned.target_slot_id != legacy.target_slot_id) {
        if (rec.selection == common_cache_plan_selection::similarity &&
            planned.provider == common_cache_plan_provider::live_slot &&
            planned.f_keep_known && planned.f_keep >= 1.0) {
            // The sole zero-destruction cross-target case.
        } else {
            effects |= common_cache_plan_destruction_effect_bit(
                rec.selection == common_cache_plan_selection::similarity
                    ? common_cache_plan_destruction_effect::
                          destructive_similarity_retarget
                    : common_cache_plan_destruction_effect::
                          cross_target_displacement);
        }
    }
    const bool legacy_uses_live_target =
        common_cache_plan_provider_is_live(legacy.provider);
    const bool destruction_certification_available =
        (permitted_effects &
         server_cache_plan_nonconsuming_host_effects(true)) != 0;
    if (planned.target_slot_id == legacy.target_slot_id &&
        ((planned.provider == common_cache_plan_provider::cold_replay &&
          legacy.provider != common_cache_plan_provider::cold_replay) ||
         (destruction_certification_available &&
          (planned.provider == common_cache_plan_provider::host_cache_entry ||
           planned.is_chain()) && legacy_uses_live_target))) {
        // Cold replacement is the established planner effect. Occupied restoration adds host restore
        // to the same physical class only when lifecycle certification exists;
        // lifecycle-off therefore preserves the previously authorized
        // same-target host-restore behavior byte-for-byte.
        // The schema-v6 name predates non-consuming host restore. Its physical
        // class is the stable contract: any certified same-target whole-state
        // replacement destroys the live slot, whether replacement bytes come
        // from cold replay or an immutable host snapshot.
        effects |= common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::same_target_cold_replacement);
    }
    const int32_t planned_host = server_cache_plan_host_source(rec, candidate);
    const int32_t legacy_host = server_cache_plan_host_source(rec, legacy_candidate);
    if (planned_host >= 0 && planned_host != legacy_host) {
        effects |= common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::different_host_source_consumption);
    }
    // Physical non-effects (lifecycle's non-consuming host restore) and
    // mutation-boundary destruction certificates share this single row-opening mask.
    return effects & ~permitted_effects;
}

bool server_cache_plan_assign_source_id(
        int32_t & instance_source_id,
        int32_t & next_source_id,
        int32_t & source_id) noexcept {
    if (instance_source_id >= 0) {
        source_id = instance_source_id;
        return true;
    }
    if (next_source_id < 0 ||
        next_source_id > SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID ||
        next_source_id >= int32_t(COMMON_CACHE_PLAN_MAX_CANDIDATES)) {
        source_id = -1;
        return false;
    }
    instance_source_id = next_source_id++;
    source_id = instance_source_id;
    return true;
}

int32_t server_cache_plan_legacy_candidate(
        const common_cache_plan_record & rec,
        int32_t target_slot_id,
        bool host_lookup_enabled) noexcept {
    int32_t live = -1;
    int32_t host = -1;
    double f_keep = -1.0;
    double sim = 0.0;

    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id != target_slot_id || candidate.is_chain()) {
            continue;
        }
        if (candidate.provider == common_cache_plan_provider::live_slot) {
            live = int32_t(i);
            if (candidate.f_keep_known) {
                f_keep = candidate.f_keep;
            }
            if (candidate.sim_known) {
                sim = candidate.sim;
            }
            break;
        }
    }

    // Reproduce the legacy host selector's strict two-axis improvement and
    // insertion-order tie behavior. Invalid rows never enter that selector.
    for (uint32_t i = 0; host_lookup_enabled && i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id != target_slot_id || candidate.is_chain() ||
            candidate.provider != common_cache_plan_provider::host_cache_entry ||
            !candidate.viable() || !candidate.f_keep_known ||
            !candidate.sim_known) {
            continue;
        }
        if (f_keep < candidate.f_keep && sim < candidate.sim) {
            f_keep = candidate.f_keep;
            sim = candidate.sim;
            host = int32_t(i);
        }
    }

    const int32_t host_source = host >= 0
        ? rec.inventory[size_t(host)].source_id : -1;
    int32_t checkpoint = -1;
    int32_t checkpoint_ordinal = -1;
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id != target_slot_id || candidate.is_chain() ||
            candidate.provider != common_cache_plan_provider::live_context_checkpoint ||
            !candidate.viable()) {
            continue;
        }
        const int32_t ordinal =
            server_cache_plan_checkpoint_ordinal_from_source_id(
                candidate.source_id, host >= 0 ? host_source : -1);
        if (host >= 0) {
            if (!candidate.component_only ||
                candidate.dependent_host_source_id != host_source) {
                continue;
            }
        } else if (candidate.component_only) {
            continue;
        }
        // The shipped selector scans newest-to-oldest, so the greatest forward
        // ordinal is its first viable checkpoint.
        if (ordinal >= 0 && ordinal > checkpoint_ordinal) {
            checkpoint_ordinal = ordinal;
            checkpoint = int32_t(i);
        }
    }

    if (checkpoint >= 0) {
        if (host < 0) {
            return checkpoint;
        }
        const auto * chain = rec.find_chain(
            common_cache_plan_provider::host_cache_entry, host, checkpoint);
        const int32_t chain_id = chain
            ? int32_t(chain - rec.inventory.data()) : -1;
        return chain_id >= 0 ? chain_id : host;
    }
    if (host >= 0) {
        return host;
    }
    if (live >= 0 && rec.inventory[size_t(live)].viable()) {
        return live;
    }
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id == target_slot_id &&
            candidate.provider == common_cache_plan_provider::cold_replay &&
            !candidate.is_chain() && candidate.viable()) {
            return int32_t(i);
        }
    }
    return -1;
}

server_cache_plan_live_evaluation server_cache_plan_evaluate_live(
        bool busy,
        bool has_payload,
        uint64_t lcp_tokens,
        uint64_t prompt_tokens,
        uint64_t source_tokens) noexcept {
    server_cache_plan_live_evaluation out;
    out.lcp_tokens = lcp_tokens;
    out.sim = prompt_tokens ? float(lcp_tokens) / float(prompt_tokens) : 0.0f;
    out.f_keep = source_tokens ? float(lcp_tokens) / float(source_tokens) : -1.0f;
    out.reason = busy ? COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY :
                 !has_payload ? COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE :
                 lcp_tokens == 0 ? COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT :
                 COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
    return out;
}

void server_cache_plan_apply_live(
        common_cache_plan_candidate * row,
        const server_cache_plan_live_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->sim = evaluation.sim;
    row->sim_known = true;
    row->f_keep = evaluation.f_keep;
    row->f_keep_known = evaluation.f_keep >= 0.0f;
    row->note_reject(evaluation.reason);
}

server_cache_plan_host_evaluation server_cache_plan_evaluate_host(
        bool payload_present,
        bool identity_matches,
        uint64_t lcp_tokens,
        uint64_t prompt_tokens,
        uint64_t source_tokens,
        uint64_t payload_bytes) noexcept {
    server_cache_plan_host_evaluation out;
    out.lcp_tokens = lcp_tokens;
    out.payload_bytes = payload_bytes;
    out.sim = prompt_tokens ? float(lcp_tokens) / float(prompt_tokens) : 0.0f;
    out.f_keep = source_tokens ? float(lcp_tokens) / float(source_tokens) : 0.0f;
    out.reason = !payload_present ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY :
                 !identity_matches ? COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH :
                 lcp_tokens == 0 ? COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT :
                 COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
    return out;
}

void server_cache_plan_apply_host(
        common_cache_plan_candidate * row,
        const server_cache_plan_host_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->payload_bytes = llama_cache_acct_value::measured(evaluation.payload_bytes);
    row->sim = evaluation.sim;
    row->sim_known = true;
    row->f_keep = evaluation.f_keep;
    row->f_keep_known = true;
    row->note_reject(evaluation.reason);
}

server_cache_plan_checkpoint_evaluation server_cache_plan_evaluate_checkpoint(
        bool payload_present,
        bool frontier_current,
        bool recurrent,
        bool checkpoint_lineage_matches,
        int64_t pos_min,
        int64_t pos_max,
        int64_t next_position,
        int64_t min_position_threshold,
        uint64_t payload_bytes) noexcept {
    server_cache_plan_checkpoint_evaluation out;
    out.lcp_tokens = pos_max >= 0 ? uint64_t(pos_max) : 0;
    out.payload_bytes = payload_bytes;
    out.reason = !payload_present ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY :
                 !frontier_current ? COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID :
                 !checkpoint_lineage_matches ? COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED :
                 recurrent
                    ? (pos_max < next_position
                        ? COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL
                        : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT)
                    : (pos_max <= next_position &&
                       (pos_min < min_position_threshold || pos_min == 0)
                        ? COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL
                        : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT);
    return out;
}

void server_cache_plan_apply_checkpoint(
        common_cache_plan_candidate * row,
        const server_cache_plan_checkpoint_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->payload_bytes = llama_cache_acct_value::measured(evaluation.payload_bytes);
    row->note_reject(evaluation.reason);
}
