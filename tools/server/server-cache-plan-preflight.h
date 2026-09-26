#pragma once

#include "common-cache-plan.h"

#include <nlohmann/json_fwd.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// This is an internal scheduler result only. The public adapter owns the JSON schema;
// none of these values is a reservation, capability, or replayable claim.
enum class server_cache_plan_preflight_status : uint8_t {
    ok = 0,
    no_target,
    internal_fault,
    _count,
};

enum class server_cache_plan_preflight_target_relation : uint8_t {
    unavailable = 0,
    forced_slot,
    same_as_legacy,
    _count,
};

enum class server_cache_plan_preflight_cache_hit : uint8_t {
    unavailable = 0,
    miss,
    partial,
    full,
    _count,
};

struct server_cache_plan_preflight_miss_reason {
    common_cache_plan_provider provider =
        common_cache_plan_provider::cold_replay;
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_NONE;
    uint32_t count = 0;
};

struct server_cache_plan_preflight_view {
    server_cache_plan_preflight_status status =
        server_cache_plan_preflight_status::internal_fault;
    common_cache_plan_selection selection_tier =
        common_cache_plan_selection::none;
    common_cache_plan_provider provider =
        common_cache_plan_provider::cold_replay;
    bool provider_available = false;
    server_cache_plan_preflight_target_relation target_relation =
        server_cache_plan_preflight_target_relation::unavailable;
    server_cache_plan_preflight_cache_hit cache_hit =
        server_cache_plan_preflight_cache_hit::unavailable;
    llama_cache_acct_value prompt_tokens;
    llama_cache_acct_value reuse_tokens;
    llama_cache_acct_value replay_tokens;
    llama_cache_acct_value restore_bytes;
    std::vector<server_cache_plan_preflight_miss_reason> miss_reasons;
};

// Public v2 snapshot. This is deliberately independent of the debug
// serializer: identities, digests, accounting rows, ordinals, serials,
// leases, and recovery-source handles have no representation here.
nlohmann::ordered_json server_cache_plan_preflight_json(
    const server_cache_plan_preflight_view & view);

// Exposure remains opt-in and single-principal. The existing API-key
// middleware authenticates zero/one configured key;  refuses configurations
// where that middleware represents multiple principals.
bool server_cache_plan_preflight_exposure_allowed(
    const std::string & hostname,
    size_t api_key_count) noexcept;

bool server_cache_plan_preflight_request_field_allowed(
    std::string_view field) noexcept;

bool server_cache_plan_preflight_build_view(
    const common_cache_plan_record & rec,
    int32_t legacy_target_slot_id,
    server_cache_plan_preflight_view & out) noexcept;
