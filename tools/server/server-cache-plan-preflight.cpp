#include "server-cache-plan-preflight-internal.h"

#include "server-cache-plan-authority.h"

#include <algorithm>
#include <array>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

json public_value(const llama_cache_acct_value & value) {
    return value.state == llama_cache_acct_known::known
        ? json(value.value) : json(nullptr);
}

const char * preflight_status_name(
        server_cache_plan_preflight_status status) noexcept {
    switch (status) {
        case server_cache_plan_preflight_status::ok:             return "ok";
        case server_cache_plan_preflight_status::no_target:      return "no_target";
        case server_cache_plan_preflight_status::internal_fault: return "internal_fault";
        case server_cache_plan_preflight_status::_count:         break;
    }
    return "invalid";
}

const char * target_relation_name(
        server_cache_plan_preflight_target_relation relation) noexcept {
    switch (relation) {
        case server_cache_plan_preflight_target_relation::unavailable:
            return "unavailable";
        case server_cache_plan_preflight_target_relation::forced_slot:
            return "forced_slot";
        case server_cache_plan_preflight_target_relation::same_as_legacy:
            return "same_as_legacy";
        case server_cache_plan_preflight_target_relation::_count:
            break;
    }
    return "invalid";
}

const char * cache_hit_name(
        server_cache_plan_preflight_cache_hit hit) noexcept {
    switch (hit) {
        case server_cache_plan_preflight_cache_hit::unavailable:
            return "unavailable";
        case server_cache_plan_preflight_cache_hit::miss:    return "miss";
        case server_cache_plan_preflight_cache_hit::partial: return "partial";
        case server_cache_plan_preflight_cache_hit::full:    return "full";
        case server_cache_plan_preflight_cache_hit::_count:  break;
    }
    return "invalid";
}

} // namespace

bool server_cache_plan_local_source_registry::get_or_assign(
        uintptr_t instance,
        int32_t & source_id) {
    auto [it, inserted] = source_ids_.emplace(instance, -1);
    (void) inserted;
    return server_cache_plan_assign_source_id(
        it->second, next_source_id_, source_id);
}

bool server_cache_plan_local_source_registry::find(
        uintptr_t instance,
        int32_t & source_id) const noexcept {
    const auto found = source_ids_.find(instance);
    if (found == source_ids_.end() || found->second < 0) {
        source_id = -1;
        return false;
    }
    source_id = found->second;
    return true;
}

server_cache_plan_preflight_semantics server_cache_plan_preflight_semantics_for(
        bool is_preflight,
        bool native_completion,
        bool update_cache,
        bool prompt_cache_available,
        bool adapter_matches) noexcept {
    server_cache_plan_preflight_semantics out;
    out.completion_semantics = is_preflight || native_completion;
    out.host_lookup_enabled = update_cache && prompt_cache_available &&
                              out.completion_semantics && adapter_matches;
    return out;
}

bool server_cache_plan_preflight_build_view(
        const common_cache_plan_record & rec,
        int32_t legacy_target_slot_id,
        server_cache_plan_preflight_view & out) noexcept {
    try {
        out = {};
        out.status = server_cache_plan_preflight_status::ok;
        out.selection_tier = rec.selection;
        out.prompt_tokens = rec.n_prompt_tokens;
        for (uint32_t i = 0; i < rec.n_inventory; ++i) {
            const auto & candidate = rec.inventory[i];
            if (candidate.reason == COMMON_CACHE_PLAN_REASON_NONE) continue;
            auto found = std::find_if(out.miss_reasons.begin(), out.miss_reasons.end(),
                [&](const auto & row) {
                    return row.provider == candidate.provider && row.reason == candidate.reason;
                });
            if (found == out.miss_reasons.end()) {
                out.miss_reasons.push_back({ candidate.provider, candidate.reason, 1 });
            } else {
                found->count++;
            }
        }
        // The inventory builder records the shipped selector's snapshot choice.
        // Do not substitute a hypothetical optimum or infer a missing provider.
        const int32_t chosen = rec.destruction_legacy_plan_candidate;
        if (rec.inventory_saturated() || chosen < 0 || uint32_t(chosen) >= rec.n_inventory) return true;
        const auto & selected = rec.inventory[size_t(chosen)];
        if (!selected.viable() || selected.target_slot_id != legacy_target_slot_id) return true;
        out.provider = selected.provider;
        out.provider_available = true;
        out.target_relation = rec.selection == common_cache_plan_selection::by_id
            ? server_cache_plan_preflight_target_relation::forced_slot
            : server_cache_plan_preflight_target_relation::same_as_legacy;
        out.reuse_tokens = selected.provider == common_cache_plan_provider::cold_replay
            ? llama_cache_acct_value::measured(0) : selected.lcp_tokens;
        if (out.prompt_tokens.state == llama_cache_acct_known::known &&
            out.reuse_tokens.state == llama_cache_acct_known::known &&
            out.reuse_tokens.value <= out.prompt_tokens.value) {
            out.replay_tokens = llama_cache_acct_value::measured(
                out.prompt_tokens.value - out.reuse_tokens.value);
            out.cache_hit = out.reuse_tokens.value == 0
                ? server_cache_plan_preflight_cache_hit::miss
                : out.replay_tokens.value == 0
                    ? server_cache_plan_preflight_cache_hit::full
                    : server_cache_plan_preflight_cache_hit::partial;
        }
        out.restore_bytes = selected.provider == common_cache_plan_provider::cold_replay ||
                            selected.provider == common_cache_plan_provider::live_slot
            ? llama_cache_acct_value::measured(0) : selected.payload_bytes;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

json server_cache_plan_preflight_json(
        const server_cache_plan_preflight_view & view) {
    json miss_reasons = json::array();
    for (const auto & row : view.miss_reasons) {
        miss_reasons.push_back({
            { "provider", common_cache_plan_provider_name(row.provider) },
            { "reason", common_cache_plan_reason_name(row.reason) },
            { "count", row.count },
        });
    }

    json selection = {
        { "selection_tier", common_cache_plan_selection_name(view.selection_tier) },
        { "provider", view.provider_available
              ? json(common_cache_plan_provider_name(view.provider)) : json(nullptr) },
        { "target_relation", view.target_relation ==
                  server_cache_plan_preflight_target_relation::unavailable
              ? json(nullptr) : json(target_relation_name(view.target_relation)) },
        { "cache_hit", view.cache_hit == server_cache_plan_preflight_cache_hit::unavailable
              ? json(nullptr) : json(cache_hit_name(view.cache_hit)) },
        { "prompt_tokens", public_value(view.prompt_tokens) },
        { "reuse_tokens", public_value(view.reuse_tokens) },
        { "replay_tokens", public_value(view.replay_tokens) },
        { "restore_bytes", public_value(view.restore_bytes) },
    };

    return json {
        { "object", "cache_plan_preflight" },
        { "schema_version", 2 },
        { "cache_plan_schema_version", COMMON_CACHE_PLAN_SCHEMA_VERSION },
        { "authoritative", false },
        { "reservation", "none" },
        { "valid_until", nullptr },
        { "status", preflight_status_name(view.status) },
        { "selection", std::move(selection) },
        { "miss_reasons", std::move(miss_reasons) },
        { "limitations", json::array({
            "point_in_time",
            "no_reservation",
            "displacement_not_projected",
            "queue_and_contention_not_modeled",
            "post_generation_maintenance_not_modeled",
        }) },
    };
}

bool server_cache_plan_preflight_exposure_allowed(
        const std::string & hostname,
        size_t api_key_count) noexcept {
    const bool local = hostname == "127.0.0.1" || hostname == "::1" ||
                       hostname == "localhost" ||
                       (hostname.size() >= 5 &&
                        hostname.compare(hostname.size() - 5, 5, ".sock") == 0);
    return local && api_key_count <= 1;
}

bool server_cache_plan_preflight_request_field_allowed(
        std::string_view field) noexcept {
    static constexpr std::array<std::string_view, 5> accepted = {
        "prompt", "id_slot", "cache_prompt", "lora",
        "message_delimiters",
    };
    return std::find(accepted.begin(), accepted.end(), field) !=
           accepted.end();
}
