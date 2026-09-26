#pragma once

#include "common-cache-plan.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// One compiled classification door shared by planner authority and
// destruction-quote assembly. A nonzero bit is precisely a destruction certificate
// the envelope must refuse until its ratchet is enabled.
int32_t server_cache_plan_host_source(
    const common_cache_plan_record & rec,
    int32_t candidate) noexcept;

constexpr common_cache_plan_destruction_effect_set
    SERVER_CACHE_LIVE_DISPLACEMENT_EFFECTS =
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::cross_target_displacement) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                destructive_similarity_retarget) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                same_target_cold_replacement);

constexpr common_cache_plan_destruction_effect_set
server_cache_plan_nonconsuming_host_effects(bool lifecycle) noexcept {
    return lifecycle
        ? common_cache_plan_destruction_effect_bit(
              common_cache_plan_destruction_effect::
                  different_host_source_consumption)
        : 0;
}

common_cache_plan_destruction_effect_set server_cache_destruction_effects_for(
    const common_cache_plan_record & rec,
    int32_t candidate,
    int32_t legacy_candidate,
    common_cache_plan_destruction_effect_set permitted_effects = 0) noexcept;

// Counterfactual forced-slot legacy provider sequence over the complete
// pre-mutation inventory. This is telemetry identity only; it never executes.
int32_t server_cache_plan_legacy_candidate(
    const common_cache_plan_record & rec,
    int32_t target_slot_id,
    bool host_lookup_enabled = true) noexcept;

constexpr int32_t SERVER_CACHE_PLAN_HOST_CHECKPOINT_BASE = 1000000;
constexpr int32_t SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE = 10000;
constexpr int32_t SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID =
    (INT32_MAX - SERVER_CACHE_PLAN_HOST_CHECKPOINT_BASE -
     (SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE - 1)) /
    SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE;

constexpr int32_t server_cache_plan_host_checkpoint_source_id(
        int32_t host_source_id,
        int32_t checkpoint_ordinal = 0) noexcept {
    return host_source_id < 0 ||
           host_source_id > SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID ||
           checkpoint_ordinal < 0 ||
           checkpoint_ordinal >= SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE
        ? -1
        : SERVER_CACHE_PLAN_HOST_CHECKPOINT_BASE +
          host_source_id*SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE +
          checkpoint_ordinal;
}

// Decode a checkpoint source id to the inventory's stable forward ordinal.
// A host-qualified source must remain inside that host's namespace.
constexpr int32_t server_cache_plan_checkpoint_ordinal_from_source_id(
        int32_t source_id,
        int32_t host_source_id = -1) noexcept {
    if (source_id < 0) {
        return -1;
    }
    if (host_source_id < 0) {
        return source_id < SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE
            ? source_id : -1;
    }
    const int32_t base = server_cache_plan_host_checkpoint_source_id(
        host_source_id, 0);
    if (base < 0 || source_id < base) {
        return -1;
    }
    const int32_t ordinal = source_id - base;
    return ordinal < SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE
        ? ordinal : -1;
}

constexpr int32_t server_cache_plan_checkpoint_reverse_position_from_source_id(
        size_t checkpoint_count,
        int32_t source_id,
        int32_t host_source_id = -1) noexcept {
    const int32_t ordinal = server_cache_plan_checkpoint_ordinal_from_source_id(
        source_id, host_source_id);
    return ordinal < 0 || size_t(ordinal) >= checkpoint_count
        ? -1 : int32_t(checkpoint_count - 1 - size_t(ordinal));
}

// Observer-only request-local host-state identity. The id is stored on the
// list node, so surviving nodes remain stable across save dedup/splice and an
// allocator-reused address cannot inherit a consumed source id.
bool server_cache_plan_assign_source_id(
    int32_t & instance_source_id,
    int32_t & next_source_id,
    int32_t & source_id) noexcept;

// Checkpoint containers are enumerated forward by the authority inventory but
// reverse by the shipped selector. Translate reverse visit position back to
// the forward ordinal used by both live and host-composed inventory rows.
constexpr int32_t server_cache_plan_checkpoint_source_id_from_reverse(
    size_t checkpoint_count,
    uint32_t reverse_ordinal,
    int32_t host_source_id = -1) noexcept {
    if (checkpoint_count == 0 || reverse_ordinal >= checkpoint_count) {
        return -1;
    }
    const size_t forward = checkpoint_count - 1 - reverse_ordinal;
    if (forward >= size_t(SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE)) {
        return -1;
    }
    return host_source_id >= 0
        ? server_cache_plan_host_checkpoint_source_id(
              host_source_id, int32_t(forward))
        : int32_t(forward);
}

constexpr bool server_cache_plan_viable(
        common_cache_plan_reason reason) noexcept {
    return reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
}

struct server_cache_plan_live_evaluation {
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE;
    uint64_t lcp_tokens = 0;
    float sim = 0.0f;
    float f_keep = 0.0f;
};

server_cache_plan_live_evaluation server_cache_plan_evaluate_live(
    bool busy,
    bool has_payload,
    uint64_t lcp_tokens,
    uint64_t prompt_tokens,
    uint64_t source_tokens = 0) noexcept;

void server_cache_plan_apply_live(
    common_cache_plan_candidate * row,
    const server_cache_plan_live_evaluation & evaluation) noexcept;

struct server_cache_plan_host_evaluation {
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE;
    uint64_t lcp_tokens = 0;
    uint64_t payload_bytes = 0;
    float sim = 0.0f;
    float f_keep = 0.0f;
};

server_cache_plan_host_evaluation server_cache_plan_evaluate_host(
    bool payload_present,
    bool identity_matches,
    uint64_t lcp_tokens,
    uint64_t prompt_tokens,
    uint64_t source_tokens,
    uint64_t payload_bytes) noexcept;

void server_cache_plan_apply_host(
    common_cache_plan_candidate * row,
    const server_cache_plan_host_evaluation & evaluation) noexcept;

struct server_cache_plan_checkpoint_evaluation {
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE;
    uint64_t lcp_tokens = 0;
    uint64_t payload_bytes = 0;
};

server_cache_plan_checkpoint_evaluation server_cache_plan_evaluate_checkpoint(
    bool payload_present,
    bool frontier_current,
    bool recurrent,
    bool checkpoint_lineage_matches,
    int64_t pos_min,
    int64_t pos_max,
    int64_t next_position,
    int64_t min_position_threshold,
    uint64_t payload_bytes) noexcept;

void server_cache_plan_apply_checkpoint(
    common_cache_plan_candidate * row,
    const server_cache_plan_checkpoint_evaluation & evaluation) noexcept;
