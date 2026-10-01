#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-cache-authority.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "../../src/llama-context.h"
#include "../../src/llama-io.h"
#include "../../src/llama-vbr-qsa-index.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <cinttypes>
#include <exception>
#include <memory>
#include <optional>
#include <filesystem>
#include <random>
#include <utility>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include "server-cache-destruction-quote.h"
#include "server-cache-plan-authority.h"
#include "server-cache-plan-preflight-internal.h"
#include "server-cache-yield.h"
#include "server-vbr-artifact-store.h"
#include "server-vbr-capture-readiness.h"
#include "server-vbr-prompt-cache-support.h"
#include "server-recurrent-expansion.h"
#include "common-cache-plan.h"
#include "gguf.h"
#include "../../src/llama-vbr-swa-window.h"
#include "../../src/llama-ext.h"
#include "../../src/llama-memory-hybrid-idx.h"
#include "../../src/llama-sha256.h"
#include <nlohmann/json.hpp>

// group A phase 2: file-scope helper, verbatim from the fork

// fork server-context.cpp:77
static json server_json_from_ordered(nlohmann::ordered_json value) {
    return json::parse(value.dump());
}

// fork 9l-d4b7e: the inverse helper (fork server-context.cpp 81-83)
static nlohmann::ordered_json server_ordered_from_json(const json & value) {
    return nlohmann::ordered_json::parse(value.dump());
}

#endif

constexpr int HTTP_POLLING_SECONDS = 1;

// Keep the C-style memory API balanced across every server early return. A scope only
// becomes active for a dynamic-VBR memory; non-VBR checkpoints preserve their old path exactly.
class server_vbr_retier_freeze_scope {
public:
    server_vbr_retier_freeze_scope(llama_memory_t mem, const char * owner) :
        mem(mem), owner(owner),
        operation_id(llama_memory_vbr_retier_freeze_begin(mem, owner)) {
    }

    ~server_vbr_retier_freeze_scope() {
        if (operation_id != 0) {
            llama_memory_vbr_retier_freeze_end(mem, owner, operation_id);
        }
    }

    server_vbr_retier_freeze_scope(const server_vbr_retier_freeze_scope &) = delete;
    server_vbr_retier_freeze_scope & operator=(const server_vbr_retier_freeze_scope &) = delete;

    bool active() const {
        return operation_id != 0;
    }

private:
    llama_memory_t mem;
    const char * owner;
    uint64_t operation_id;
};

// defined near get_model_info(); shared by the /props and /models responses
static json server_vbr_meta_json(const server_context_meta * meta);

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    // fork 9l-d4b6t: SLOT_STATE_WAIT_VBR_CAPTURE (fork server-context.cpp 1249)
    SLOT_STATE_WAIT_VBR_CAPTURE,

    SLOT_STATE_GENERATING,
};

server_speculative_decode_terminal
server_speculative_decode_terminal_resolve(
        int decode_result,
        bool hard_seal_terminal,
        bool single_token_batch,
        bool selected_exception,
        bool speculative_ok) noexcept {
    if (decode_result != 0) {
        if (decode_result == 1 && hard_seal_terminal) {
            return server_speculative_decode_terminal::preserve_hard_seal;
        }
        if (decode_result == -1 || decode_result < -1 ||
                (decode_result == 1 && single_token_batch)) {
            return server_speculative_decode_terminal::ordinary_ret_error;
        }
        return server_speculative_decode_terminal::retry;
    }
    if (selected_exception || !speculative_ok) {
        return server_speculative_decode_terminal::reset_committed_then_throw;
    }
    return server_speculative_decode_terminal::success;
}

// fork 9l-d2a: family-binding resolution for launch (fork server-context.cpp 1474-1485)
static server_cache_control_status server_cache_family_resolve_for_launch(
        server_cache_control_authority * authority,
        server_cache_control_token token,
        common_cache_family_binding & out) noexcept {
    out = {};
    if (!token) {
        return server_cache_control_status::ok;
    }
    return authority
        ? authority->resolve_family_binding(token, out)
        : server_cache_control_status::not_found;
}

bool server_vbr_empty_handoff_lookup_allowed(
        const server_vbr_empty_handoff_gate & gate) noexcept {
    return gate.slot_count == 1 && !gate.hard_lease &&
        !gate.deferred_task &&
        gate.incumbent_supported;
}

bool server_vbr_empty_handoff_allowed(
        const server_vbr_empty_handoff_gate & gate) noexcept {
    return server_vbr_empty_handoff_lookup_allowed(gate) &&
        gate.incoming_prefix > gate.incumbent_lcp &&
        gate.durable_incumbent_prefix > gate.incumbent_lcp &&
        !gate.exact_incumbent_durable && gate.family_matches;
}

bool server_vbr_live_source_displacement_allowed(
        bool kv_unified,
        size_t slot_count) noexcept {
    // Empty import is a whole-child contract. Clearing one sequence out of a
    // multi-slot unified tree leaves foreign rows behind, so retain the live
    // source and let the atomic occupied-replacement route preserve them.
    return !kv_unified || slot_count <= 1;
}

bool server_vbr_stem_matches_capture_source(
        bool valid,
        const std::array<uint8_t, 32> & stem_source,
        const std::array<uint8_t, 32> & capture_source) noexcept {
    return valid && stem_source == capture_source;
}

// fork 9l-d4b7k: the slot-envelope layer (fork server-context.cpp 155-237).
constexpr char SERVER_SLOT_ENVELOPE_MAGIC[8] = {
    'B', 'U', 'U', 'N', 'S', 'L', 'O', 'T',
};
constexpr uint32_t SERVER_SLOT_ENVELOPE_VERSION = 2;
constexpr size_t SERVER_SLOT_ENVELOPE_HEADER_SIZE = 224;
constexpr uint32_t SERVER_SLOT_ENVELOPE_HAS_LOGITS = 1u << 0;
constexpr uint32_t SERVER_SLOT_ENVELOPE_RUNTIME_FAMILY_BOUND = 1u << 1;

enum class server_slot_frontier_logits_status {
    loaded,
    not_present,
    legacy_cold,
    io_error,
    format_mismatch,
    model_family_mismatch,
    adapter_mismatch,
    token_count_mismatch,
    next_position_mismatch,
    token_digest_mismatch,
    vocabulary_mismatch,
    serialized_payload_mismatch,
    payload_mismatch,
    logits_digest_mismatch,
};

const char * server_slot_frontier_logits_status_name(
        server_slot_frontier_logits_status status) noexcept {
    switch (status) {
        case server_slot_frontier_logits_status::loaded:
            return "loaded";
        case server_slot_frontier_logits_status::not_present:
            return "not_present";
        case server_slot_frontier_logits_status::legacy_cold:
            return "legacy_cold";
        case server_slot_frontier_logits_status::io_error:
            return "io_error";
        case server_slot_frontier_logits_status::format_mismatch:
            return "format_mismatch";
        case server_slot_frontier_logits_status::model_family_mismatch:
            return "model_family_mismatch";
        case server_slot_frontier_logits_status::adapter_mismatch:
            return "adapter_mismatch";
        case server_slot_frontier_logits_status::token_count_mismatch:
            return "token_count_mismatch";
        case server_slot_frontier_logits_status::next_position_mismatch:
            return "next_position_mismatch";
        case server_slot_frontier_logits_status::token_digest_mismatch:
            return "token_digest_mismatch";
        case server_slot_frontier_logits_status::vocabulary_mismatch:
            return "vocabulary_mismatch";
        case server_slot_frontier_logits_status::serialized_payload_mismatch:
            return "serialized_payload_mismatch";
        case server_slot_frontier_logits_status::payload_mismatch:
            return "payload_mismatch";
        case server_slot_frontier_logits_status::logits_digest_mismatch:
            return "logits_digest_mismatch";
    }
    return "unknown";
}

std::array<uint8_t, 32> server_slot_frontier_tagged_digest(
        const char * domain,
        const void * data,
        size_t size) {
    llama_sha256_writer writer;
    writer.string(domain, std::strlen(domain));
    writer.string(data, size);
    return writer.finish();
}

std::array<uint8_t, 32> server_slot_frontier_string_digest(
        const char * domain,
        const std::string & value) {
    return server_slot_frontier_tagged_digest(
        domain, value.data(), value.size());
}

std::array<uint8_t, 32> server_slot_frontier_logits_digest(
        const std::vector<float> & logits) {
    return server_slot_frontier_tagged_digest(
        "buun.server.slot-frontier-logits/v1",
        logits.data(), logits.size()*sizeof(float));
}

// fork 9l-d4b7j: the slot-file runtime identity (fork server-context.cpp 239-350).
struct server_slot_runtime_identity {
    std::array<uint8_t, 32> digest = {};
    // Capability policy, not weight identity: "exact" slot continuation is
    // exact only within this semantic-family + runtime configuration contract.
    bool family_compatible = false;
};

struct server_slot_runtime_geometry {
    uint32_t n_ctx = 0;
    uint32_t n_ctx_seq = 0;
    uint32_t n_ctx_orig_yarn = 0;
};

server_slot_runtime_geometry server_slot_runtime_geometry_from_owner(
        uint32_t actual_n_ctx,
        uint32_t actual_n_ctx_seq,
        const common_params & params,
        uint32_t model_n_ctx_train,
        uint32_t model_n_ctx_orig_yarn) noexcept {
    return {
        actual_n_ctx,
        actual_n_ctx_seq,
        params.yarn_orig_ctx != 0 ? uint32_t(params.yarn_orig_ctx) :
            model_n_ctx_orig_yarn != 0 ? model_n_ctx_orig_yarn :
                                         model_n_ctx_train,
    };
}

server_slot_runtime_identity server_slot_runtime_identity_from_owner_receipts(
        const std::array<uint8_t, 32> & model_digest,
        bool model_digest_valid,
        const llama_model * model,
        const common_params & params,
        server_slot_runtime_geometry runtime_geometry) {
    llama_sha256_writer writer;
    static constexpr char domain[] =
        "buun.server.slot-file-runtime-identity/v2";
    writer.string(domain, sizeof(domain) - 1);
    const char * commit = llama_commit();
    writer.string(commit, std::strlen(commit));
    bool family_compatible = model_digest_valid;
    writer.u32(family_compatible ? 1u : 0u);
    writer.bytes(model_digest.data(), model_digest.size());
    const bool has_mmproj = !params.no_mmproj && !params.mmproj.path.empty();
    writer.u32(has_mmproj ? 1u : 0u);
    // The multimodal projector has no semantic-family receipt yet.
    // Media slot files retain their existing cold semantic restore, but can
    // never acquire the exact frontier-logits capability from a path hash.
    if (has_mmproj) {
        family_compatible = false;
    }
    writer.u32(uint32_t(params.control_vectors.size()));
    if (!params.control_vectors.empty()) {
        family_compatible = family_compatible &&
            params.control_vector_applied_digest_valid;
        writer.bytes(params.control_vector_applied_digest.data(),
                     params.control_vector_applied_digest.size());
    }
    writer.u32(uint32_t(params.control_vector_layer_start));
    writer.u32(uint32_t(params.control_vector_layer_end));
    // LoRA authority is intentionally not duplicated here.  The loaded
    // adapter registry can change through POST /lora-adapters; each envelope
    // binds the canonical per-slot lora_config_identity(), which already owns
    // loaded adapter content identity and scale at save/restore time.
    writer.u32(model ? uint32_t(llama_vocab_n_tokens(
        llama_model_get_vocab(model))) : 0);
    const auto write_f32 = [&](float value) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        writer.u32(bits);
    };
    writer.u32(runtime_geometry.n_ctx);
    writer.u32(runtime_geometry.n_ctx_seq);
    writer.u32(uint32_t(params.grp_attn_n));
    writer.u32(uint32_t(params.grp_attn_w));
    writer.u32(uint32_t(params.rope_scaling_type));
    write_f32(params.rope_freq_base);
    write_f32(params.rope_freq_scale);
    write_f32(params.yarn_ext_factor);
    write_f32(params.yarn_attn_factor);
    write_f32(params.yarn_beta_fast);
    write_f32(params.yarn_beta_slow);
    writer.u32(runtime_geometry.n_ctx_orig_yarn);
    writer.u32(uint32_t(params.cache_type_k));
    writer.u32(uint32_t(params.cache_type_v));
    writer.u32(uint32_t(params.flash_attn_type));
    writer.u32(params.no_fused_gdn ? 1u : 0u);
    writer.u32(params.swa_full ? 1u : 0u);
    writer.u32(params.kv_unified ? 1u : 0u);
    writer.u32(params.logits_all ? 1u : 0u);
    return { writer.finish(), family_compatible };
}

server_slot_runtime_identity server_slot_file_runtime_identity_build(
        const llama_model * model,
        const llama_context * ctx,
        const common_params & params) {
    std::array<uint8_t, 32> model_digest = {};
    const bool valid = model && llama_model_semantic_family_digest(
        model, model_digest.data());
    const uint32_t n_ctx_train = model
        ? uint32_t(llama_model_n_ctx_train(model)) : 0;
    const uint32_t n_ctx_orig_yarn = model
        ? uint32_t(llama_model_n_ctx_orig_yarn(model)) : 0;
    return server_slot_runtime_identity_from_owner_receipts(
        model_digest, valid, model, params,
        server_slot_runtime_geometry_from_owner(
            ctx ? llama_n_ctx(ctx) : 0,
            ctx ? llama_n_ctx_seq(ctx) : 0,
            params, n_ctx_train, n_ctx_orig_yarn));
}

// fork 9l-d4b7k: the slot-envelope layer, part 2 (fork server-context.cpp
// 352-638).  The LE helpers (B1) come first, then the envelope itself (B2),
// then the token-binding check (B3) -- fork order, and also the only order in
// which every file-scope free function is defined before its use.
void server_slot_store_le_u32(std::vector<char> & out, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        out.push_back(char(uint8_t(value >> (8*i))));
    }
}

void server_slot_store_le_u64(std::vector<char> & out, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out.push_back(char(uint8_t(value >> (8*i))));
    }
}

bool server_slot_load_le_u32(
        const char * data, size_t size, size_t & offset,
        uint32_t & value) noexcept {
    if (offset + 4 > size) {
        return false;
    }
    value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= uint32_t(uint8_t(data[offset++])) << (8*i);
    }
    return true;
}

bool server_slot_load_le_u64(
        const char * data, size_t size, size_t & offset,
        uint64_t & value) noexcept {
    if (offset + 8 > size) {
        return false;
    }
    value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= uint64_t(uint8_t(data[offset++])) << (8*i);
    }
    return true;
}

void server_slot_state_payload_checksum_update(
        uint64_t & hash, const uint8_t * data, size_t size) noexcept {
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
}

uint64_t server_slot_state_payload_checksum(
        const uint8_t * data, size_t size) noexcept {
    uint64_t hash = UINT64_C(14695981039346656037);
    server_slot_state_payload_checksum_update(hash, data, size);
    return hash;
}

struct server_slot_envelope {
    server_slot_frontier_logits_status status =
        server_slot_frontier_logits_status::format_mismatch;
    std::vector<char> serialized_tokens;
    std::vector<float> logits;
    std::array<uint8_t, 32> token_digest = {};
    uint64_t token_count = 0;
    int64_t next_position = 0;
};

bool server_slot_envelope_build(
        const std::vector<char> & serialized_tokens,
        const server_slot_runtime_identity & runtime_identity,
        const std::string & adapter_identity,
        uint64_t token_count,
        int64_t next_position,
        const std::array<uint8_t, 32> & token_digest,
        const std::vector<float> * logits,
        std::vector<char> & output) noexcept {
    output.clear();
    try {
        if (serialized_tokens.empty() || serialized_tokens.size() % 4 != 0 ||
            serialized_tokens.size() > UINT64_MAX ||
            (logits && (logits->empty() || logits->size() > UINT32_MAX ||
                !std::all_of(logits->begin(), logits->end(),
                    [](float value) { return std::isfinite(value); })))) {
            return false;
        }
        const bool include_logits = logits && runtime_identity.family_compatible;
        const uint64_t logits_bytes = include_logits
            ? uint64_t(logits->size())*sizeof(float) : 0;
        if (serialized_tokens.size() > SIZE_MAX - SERVER_SLOT_ENVELOPE_HEADER_SIZE ||
            logits_bytes > SIZE_MAX - SERVER_SLOT_ENVELOPE_HEADER_SIZE -
                serialized_tokens.size()) {
            return false;
        }
        output.reserve(SERVER_SLOT_ENVELOPE_HEADER_SIZE +
                       serialized_tokens.size() + size_t(logits_bytes));
        output.insert(output.end(), std::begin(SERVER_SLOT_ENVELOPE_MAGIC),
                      std::end(SERVER_SLOT_ENVELOPE_MAGIC));
        server_slot_store_le_u32(output, SERVER_SLOT_ENVELOPE_VERSION);
        server_slot_store_le_u32(output,
                                 uint32_t(SERVER_SLOT_ENVELOPE_HEADER_SIZE));
        uint32_t flags = runtime_identity.family_compatible
            ? SERVER_SLOT_ENVELOPE_RUNTIME_FAMILY_BOUND : 0;
        if (include_logits) {
            flags |= SERVER_SLOT_ENVELOPE_HAS_LOGITS;
        }
        server_slot_store_le_u32(output, flags);
        server_slot_store_le_u32(output, 0);
        server_slot_store_le_u64(output, serialized_tokens.size());
        server_slot_store_le_u64(output, token_count);
        server_slot_store_le_u64(output, uint64_t(next_position));
        server_slot_store_le_u32(output,
            include_logits ? uint32_t(logits->size()) : 0);
        server_slot_store_le_u32(output, 0);
        const auto adapter_digest = server_slot_frontier_string_digest(
            "buun.server.slot-frontier-adapter/v1", adapter_identity);
        const auto serialized_digest = server_slot_frontier_tagged_digest(
            "buun.server.slot-token-envelope/v2", serialized_tokens.data(),
            serialized_tokens.size());
        const auto logits_digest = include_logits
            ? server_slot_frontier_logits_digest(*logits)
            : std::array<uint8_t, 32> {};
        for (const auto & digest : { runtime_identity.digest, adapter_digest,
                token_digest, serialized_digest, logits_digest }) {
            output.insert(output.end(),
                          reinterpret_cast<const char *>(digest.data()),
                          reinterpret_cast<const char *>(digest.data() + digest.size()));
        }
        server_slot_store_le_u64(output, logits_bytes);
        GGML_ASSERT(output.size() == SERVER_SLOT_ENVELOPE_HEADER_SIZE);
        output.insert(output.end(), serialized_tokens.begin(), serialized_tokens.end());
        if (include_logits) {
            const char * begin = reinterpret_cast<const char *>(logits->data());
            output.insert(output.end(), begin, begin + logits_bytes);
        }
        return output.size() % sizeof(llama_token) == 0;
    } catch (...) {
        output.clear();
        return false;
    }
}

server_slot_envelope server_slot_envelope_parse_bytes(
        const void * packed_data,
        size_t packed_size,
        const server_slot_runtime_identity & runtime_identity,
        const std::string & adapter_identity,
        uint32_t vocabulary_size) noexcept {
    server_slot_envelope result;
    try {
        const char * data = reinterpret_cast<const char *>(packed_data);
        const size_t size = packed_size;
        if (size < sizeof(SERVER_SLOT_ENVELOPE_MAGIC) ||
            std::memcmp(data, SERVER_SLOT_ENVELOPE_MAGIC,
                        sizeof(SERVER_SLOT_ENVELOPE_MAGIC)) != 0) {
            result.status = server_slot_frontier_logits_status::legacy_cold;
            return result;
        }
        size_t offset = sizeof(SERVER_SLOT_ENVELOPE_MAGIC);
        uint32_t version = 0;
        uint32_t header_size = 0;
        uint32_t flags = 0;
        uint32_t reserved = 0;
        uint64_t token_bytes = 0;
        uint64_t next_position = 0;
        uint32_t saved_vocab = 0;
        uint32_t reserved2 = 0;
        if (!server_slot_load_le_u32(data, size, offset, version) ||
            !server_slot_load_le_u32(data, size, offset, header_size) ||
            !server_slot_load_le_u32(data, size, offset, flags) ||
            !server_slot_load_le_u32(data, size, offset, reserved) ||
            !server_slot_load_le_u64(data, size, offset, token_bytes) ||
            !server_slot_load_le_u64(data, size, offset, result.token_count) ||
            !server_slot_load_le_u64(data, size, offset, next_position) ||
            !server_slot_load_le_u32(data, size, offset, saved_vocab) ||
            !server_slot_load_le_u32(data, size, offset, reserved2) ||
            version != SERVER_SLOT_ENVELOPE_VERSION ||
            header_size != SERVER_SLOT_ENVELOPE_HEADER_SIZE ||
            reserved != 0 || reserved2 != 0 ||
            (flags & ~(SERVER_SLOT_ENVELOPE_HAS_LOGITS |
                       SERVER_SLOT_ENVELOPE_RUNTIME_FAMILY_BOUND)) != 0) {
            return result;
        }
        std::array<uint8_t, 32> saved_runtime = {};
        std::array<uint8_t, 32> saved_adapter = {};
        std::array<uint8_t, 32> saved_serialized = {};
        std::array<uint8_t, 32> saved_logits = {};
        for (auto * digest : { &saved_runtime, &saved_adapter,
                &result.token_digest, &saved_serialized, &saved_logits }) {
            if (offset + digest->size() > size) {
                return result;
            }
            std::memcpy(digest->data(), data + offset, digest->size());
            offset += digest->size();
        }
        uint64_t logits_bytes = 0;
        if (!server_slot_load_le_u64(data, size, offset, logits_bytes) ||
            offset != SERVER_SLOT_ENVELOPE_HEADER_SIZE ||
            token_bytes == 0 || token_bytes % 4 != 0 ||
            token_bytes > size - offset || logits_bytes > size - offset - token_bytes ||
            offset + token_bytes + logits_bytes != size) {
            return result;
        }
        if (saved_runtime != runtime_identity.digest) {
            result.status = server_slot_frontier_logits_status::model_family_mismatch;
            return result;
        }
        if (saved_adapter != server_slot_frontier_string_digest(
                "buun.server.slot-frontier-adapter/v1", adapter_identity)) {
            result.status = server_slot_frontier_logits_status::adapter_mismatch;
            return result;
        }
        result.next_position = int64_t(next_position);
        result.serialized_tokens.assign(data + offset, data + offset + token_bytes);
        if (saved_serialized != server_slot_frontier_tagged_digest(
                "buun.server.slot-token-envelope/v2",
                result.serialized_tokens.data(), result.serialized_tokens.size())) {
            result.status =
                server_slot_frontier_logits_status::serialized_payload_mismatch;
            return result;
        }
        offset += token_bytes;
        const bool has_logits = (flags & SERVER_SLOT_ENVELOPE_HAS_LOGITS) != 0;
        if (!has_logits) {
            if (logits_bytes != 0 || saved_vocab != 0 ||
                saved_logits != std::array<uint8_t, 32> {}) {
                result.status = server_slot_frontier_logits_status::payload_mismatch;
                return result;
            }
            result.status = server_slot_frontier_logits_status::not_present;
            return result;
        }
        if ((flags & SERVER_SLOT_ENVELOPE_RUNTIME_FAMILY_BOUND) == 0 ||
            !runtime_identity.family_compatible || saved_vocab != vocabulary_size) {
            result.status = saved_vocab != vocabulary_size
                ? server_slot_frontier_logits_status::vocabulary_mismatch
                : server_slot_frontier_logits_status::not_present;
            return result;
        }
        if (logits_bytes != uint64_t(saved_vocab)*sizeof(float)) {
            result.status = server_slot_frontier_logits_status::payload_mismatch;
            return result;
        }
        result.logits.resize(saved_vocab);
        std::memcpy(result.logits.data(), data + offset, size_t(logits_bytes));
        if (saved_logits != server_slot_frontier_logits_digest(result.logits)) {
            result.status = server_slot_frontier_logits_status::logits_digest_mismatch;
            return result;
        }
        if (!std::all_of(result.logits.begin(), result.logits.end(),
                [](float value) { return std::isfinite(value); })) {
            result.status = server_slot_frontier_logits_status::payload_mismatch;
            return result;
        }
        result.status = server_slot_frontier_logits_status::loaded;
        return result;
    } catch (...) {
        result = {};
        result.status = server_slot_frontier_logits_status::io_error;
        return result;
    }
}

server_slot_envelope server_slot_envelope_parse(
        const llama_tokens & packed,
        const server_slot_runtime_identity & runtime_identity,
        const std::string & adapter_identity,
        uint32_t vocabulary_size) noexcept {
    return server_slot_envelope_parse_bytes(
        packed.data(), packed.size()*sizeof(llama_token), runtime_identity,
        adapter_identity, vocabulary_size);
}

server_slot_frontier_logits_status server_slot_envelope_token_binding_status(
        const server_slot_envelope & envelope,
        uint64_t token_count,
        int64_t next_position,
        const std::array<uint8_t, 32> & token_digest) noexcept {
    if (envelope.status != server_slot_frontier_logits_status::loaded) {
        return envelope.status;
    }
    if (envelope.token_count != token_count) {
        return server_slot_frontier_logits_status::token_count_mismatch;
    }
    if (envelope.next_position != next_position) {
        return server_slot_frontier_logits_status::next_position_mismatch;
    }
    if (envelope.token_digest != token_digest) {
        return server_slot_frontier_logits_status::token_digest_mismatch;
    }
    return server_slot_frontier_logits_status::loaded;
}

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

// fork server-context.cpp:640-643 (anonymous namespace there; file scope here)
struct server_slot_exact_prompt_action {
    bool use_frontier_logits = false;
    size_t reusable_tokens = 0;
};

// fork server-context.cpp:755-766 (anonymous namespace there; `static` here)
static server_slot_exact_prompt_action server_slot_exact_prompt_action_resolve(
        size_t prompt_tokens,
        bool need_sampling,
        bool diffusion_self_speculation,
        bool frontier_logits_match) noexcept {
    server_slot_exact_prompt_action result;
    result.use_frontier_logits = prompt_tokens > 0 && need_sampling &&
        !diffusion_self_speculation && frontier_logits_match;
    result.reusable_tokens = prompt_tokens -
        ((prompt_tokens > 0 && !result.use_frontier_logits) ? 1u : 0u);
    return result;
}

struct server_frontier_logits_state {
    std::vector<float> values;
    std::array<uint8_t, 32> token_digest = {};
    uint64_t sequence_epoch = 0;
    bool pending = false;

    server_frontier_logits_state() noexcept : values() {}

    void clear() noexcept {
        std::vector<float>().swap(values);
        token_digest = {};
        sequence_epoch = 0;
        pending = false;
    }

    void install(
            std::vector<float> && source,
            const std::array<uint8_t, 32> & digest,
            uint64_t epoch) noexcept {
        values = std::move(source);
        token_digest = digest;
        sequence_epoch = epoch;
        pending = false;
    }

    bool matches(
            const server_tokens & tokens,
            uint64_t epoch,
            size_t vocabulary_size) const noexcept {
        if (tokens.empty() || tokens.has_media() || values.size() != vocabulary_size ||
            sequence_epoch != epoch) {
            return false;
        }
        std::array<uint8_t, 32> digest = {};
        return tokens.retention_token_digest(digest) && digest == token_digest;
    }
};

// fork server-context.cpp:1487
struct server_vbr_idle_refusal_witness {
    std::array<uint8_t, 32> attempt_identity = {};
    uint64_t accounting_serial = 0;
    uint64_t retention_epoch = 0;
    uint64_t lease_event_ordinal = 0;
    uint64_t lease_next_expiry_ns = 0;
    bool lease_tracked = false;
    bool valid = false;
};

// fork server-context.cpp:1497
struct server_swa_window_checkpoint {
    uint64_t sequence_epoch = 0;
    std::shared_ptr<const vbr_swa_window_image> image;
};

// fork server-context.cpp:1423
struct server_cache_plan_observer {
    uint64_t records_finalized  = 0;
    uint64_t shadow_unavailable = 0;
};

// Retention policy stays at the server's logical-operation boundary. These functions
// are the closed physical seams allowed to invoke the low-level destructive primitives.
// Callers must first admit their logical manifest, or be an explicitly transient operation.
static bool server_cache_live_range_drop_impl(
        llama_memory_t mem,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        bool attention_only = false) {
    return attention_only
        ? llama_memory_seq_rm_attn(mem, seq_id, p0, p1)
        : llama_memory_seq_rm(mem, seq_id, p0, p1);
}

static void server_cache_live_range_drop_impl(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    common_context_seq_rm(ctx, seq_id, p0, p1);
}

static bool server_cache_mandatory_recovery_reset_impl(
        llama_memory_t mem,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    return llama_memory_seq_rm(mem, seq_id, p0, p1);
}

static void server_cache_mandatory_recovery_reset_impl(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    common_context_seq_rm(ctx, seq_id, p0, p1);
}

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;

    // Canonical identity of the adapter set this slot is executing under. Computed
    // on first use and cached, because a capture identity is built per request while
    // the adapter set changes only when a request installs different adapters.
    // Full raw vocabulary logits produced at the exact retained prompt
    // frontier. They are captured only at a successful generation stop and
    // travel as an authenticated VBR companion with the same token/KV image.
    server_frontier_logits_state frontier_logits;
    std::string vbr_adapter_config_identity;
    bool vbr_adapter_config_identity_valid = false;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        auto * cur_fixed = cur->payload.fixed_state();
        if (cur_fixed == nullptr) {
            SLT_WRN(*this, "%s", "prompt cache entry has no fixed state to save into\n");
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur_fixed->main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur_fixed->drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    // fork server-context.cpp:2046-2052 (replaces this port's pre-door prompt_clear)
    // The fork routes slot clearing through the censused door; this port still had
    // the raw eraser, which the file's own comment at 434-436 forbids.  Default
    // argument keeps every existing 0-arg call site unchanged.
    void prompt_clear(
            server_cache_destruction_reason reason =
                server_cache_destruction_reason::slot_rebind) {
        (void) observe_full_slot(
            server_cache_destruction_class::slot_drop, reason);
        server_cache_slot_drop_impl();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        // live effective KV bits/value (moves under dynamic VBR); pollable via GET /slots
        if (ctx_tgt != nullptr) {
            const double kv_bpv = llama_memory_kv_bpv(llama_get_memory(ctx_tgt));
            if (kv_bpv >= 0.0) {
                res["kv_bpv"] = kv_bpv;
            }
        }

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            if (!cache_status.empty()) {
                res["cache_status"] = cache_status; // [obs] why the last prompt (partially) reprocessed
            }
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        // ON_DEVICE checkpoints refer to the source sequence's transient context storage and
        // cannot be cloned to a child sequence. The copied live memory is sufficient; portable
        // host checkpoints remain available for deeper replay.
        other.prompt.checkpoints.remove_if([](const common_prompt_checkpoint & ckpt) {
            return ckpt.data_tgt_on_device;
        });
        other.init_sampler();
    }
    // --- computation-frontier and speculative accessors (slice 7d part 3b) ---

    bool frontier_logits_matches() const noexcept {
        try {
            return ctx_tgt && frontier_logits.matches(
                prompt.tokens, prompt.sequence_epoch,
                size_t(llama_vocab_n_tokens(
                    llama_model_get_vocab(llama_get_model(ctx_tgt)))));
        } catch (...) {
            return false;
        }
    }

    // fork 9l-d4b6w: server_slot::capture_frontier_logits + capture_synchronized_frontier_logits
    // (fork server-context.cpp 1911-1967)
    bool capture_frontier_logits(
            int32_t output_index, bool synchronize) noexcept {
        try {
            if (synchronize) {
                llama_synchronize(ctx_tgt);
            }
            const int32_t n_vocab = llama_vocab_n_tokens(
                llama_model_get_vocab(llama_get_model(ctx_tgt)));
            llama_memory_i * memory = llama_get_memory(ctx_tgt);
            const llama_pos live_pos_max = memory
                ? memory->seq_pos_max(id) : -1;
            const llama_pos token_pos_next = prompt.tokens.pos_next();
            if (n_vocab <= 0 || prompt.n_tokens() <= 0 ||
                prompt.tokens.has_media() || live_pos_max < 0 ||
                live_pos_max + 1 != token_pos_next) {
                return false;
            }
            // The public getter may prefer a backend-sampled/top-k row whose
            // valid prefix is smaller than n_vocab.  Persistent frontier
            // authority must always copy the full raw pre-sampler row.
            if (ctx_tgt->get_sampled_logits_ith(output_index) != nullptr) {
                SLT_DBG(*this, "%s",
                    "frontier logits unavailable: backend-sampled row is not raw authority\n");
                return false;
            }
            const float * logits = ctx_tgt->get_logits_ith(output_index);
            if (!logits) {
                SLT_DBG(*this,
                        "frontier logits unavailable: vocab=%d raw=%d prompt=%d media=%d\n",
                        n_vocab, logits != nullptr,
                        prompt.n_tokens(), prompt.tokens.has_media());
                return false;
            }
            std::array<uint8_t, 32> digest = {};
            if (!prompt.tokens.retention_token_digest(digest)) {
                return false;
            }
            std::vector<float> captured((size_t(n_vocab)));
            std::memcpy(captured.data(), logits,
                        captured.size()*sizeof(float));
            if (!std::all_of(captured.begin(), captured.end(),
                    [](float value) { return std::isfinite(value); })) {
                return false;
            }
            frontier_logits.install(
                std::move(captured), digest, prompt.sequence_epoch);
            return true;
        } catch (...) {
            frontier_logits.clear();
            return false;
        }
    }

    bool capture_synchronized_frontier_logits(
            int32_t output_index) noexcept {
        return capture_frontier_logits(output_index, false);
    }

    void bind_frontier_logits_to_prompt() noexcept {
        if (frontier_logits.values.empty()) {
            return;
        }
        try {
            std::array<uint8_t, 32> digest = {};
            if (prompt.tokens.retention_token_digest(digest)) {
                frontier_logits.token_digest = digest;
                frontier_logits.sequence_epoch = prompt.sequence_epoch;
                return;
            }
        } catch (...) {
        }
        frontier_logits.clear();
    }

    // FORK-DIFF: the fork keeps an owned speculative handle plus a raw
    // spec_shared and picks whichever is live; the port's slot holds one raw
    // pointer to the shared owner (see bind_slots_to_draft_context), so the
    // accessor is the identity.
    // fork: server-context.cpp:2571 `return spec ? spec.get() : spec_shared;`
    common_speculative * get_spec() const {
        return spec;
    }

    // ---------------------------------------------------------------------
    // group A phase 2: slot state, verbatim from the fork
    // ---------------------------------------------------------------------

    // fork server-context.cpp:2294-2303 + 2305-2374
    // (the port already had 2303 alone; the fork defines all seven counters
    //  and both ratchet methods together)
    // Computation-frontier shadow-to-authority ratchet. Only checkpoint-selection
    // decisions qualify the agreement streak; other consistency checks may
    // reset/fallback it but cannot make a vacuous workload flip authority.
    uint64_t frontier_ratchet_min_agreements = 1024;
    uint64_t frontier_ratchet_agreement_streak = 0;
    uint64_t frontier_ratchet_agreements_total = 0;
    uint64_t frontier_ratchet_disagreements_total = 0;
    uint64_t frontier_ratchet_flips_total = 0;
    uint64_t frontier_ratchet_fallbacks_total = 0;
    bool frontier_ratchet_flipped = false;

    void frontier_ratchet_disagreement(
            const char * site,
            int64_t legacy_choice,
            int64_t frontier_choice) {
        frontier_ratchet_disagreements_total++;
        frontier_ratchet_agreement_streak = 0;

        if (frontier_ratchet_flipped) {
            frontier_ratchet_flipped = false;
            frontier_ratchet_fallbacks_total++;
            SLT_ERR(*this,
                    "FRONTIER_RATCHET event=post_flip_disagreement action=fallback_legacy "
                    "site=%s legacy=%" PRId64 " frontier=%" PRId64
                    " disagreements=%" PRIu64 " fallbacks=%" PRIu64 "\n",
                    site, legacy_choice, frontier_choice,
                    frontier_ratchet_disagreements_total,
                    frontier_ratchet_fallbacks_total);
        } else {
            SLT_WRN(*this,
                    "FRONTIER_RATCHET event=shadow_disagreement action=keep_legacy "
                    "site=%s legacy=%" PRId64 " frontier=%" PRId64
                    " disagreements=%" PRIu64 "\n",
                    site, legacy_choice, frontier_choice,
                    frontier_ratchet_disagreements_total);
        }
    }

    // Returns whether the frontier path owns this selection. `eligible` means
    // at least one dual-written record was actually evaluated; empty/legacy-only
    // scans do not count toward "sustained".
    bool frontier_ratchet_selection(
            bool eligible,
            bool agreement,
            int64_t legacy_choice,
            int64_t frontier_choice) {
        if (!eligible) {
            if (frontier_ratchet_flipped) {
                frontier_ratchet_disagreement(
                    "checkpoint_selection_no_frontier_record",
                    legacy_choice, frontier_choice);
            }
            return false;
        }

        if (!agreement) {
            frontier_ratchet_disagreement(
                "checkpoint_selection", legacy_choice, frontier_choice);
            return false;
        }

        frontier_ratchet_agreements_total++;
        if (!frontier_ratchet_flipped &&
            frontier_ratchet_min_agreements > 0) {
            frontier_ratchet_agreement_streak++;
            if (frontier_ratchet_agreement_streak >=
                    frontier_ratchet_min_agreements) {
                frontier_ratchet_flipped = true;
                frontier_ratchet_flips_total++;
                SLT_INF(*this,
                        "FRONTIER_RATCHET event=flip action=frontier_reads "
                        "streak=%" PRIu64 " threshold=%" PRIu64
                        " flips=%" PRIu64 "\n",
                        frontier_ratchet_agreement_streak,
                        frontier_ratchet_min_agreements,
                        frontier_ratchet_flips_total);
            }
        }

        return frontier_ratchet_flipped;
    }
    // fork server-context.cpp:1511
    // fork server-context.cpp:1510
    server_cache_destruction_observer * destruction_obs = nullptr;
    server_retention_sidecar_store * retention_obs = nullptr;
    // fork server-context.cpp:1512
    server_cache_lease_table * lease_obs = nullptr;
    // fork server-context.cpp:1517
    const std::string * lease_execution_identity = nullptr;
    // fork server-context.cpp:1518
    server_cache_authority * lifecycle_authority = nullptr;

    // fork server-context.cpp:1675  (4 lines)
    bool hard_lease_blocks_live_prefix() const {
        return hard_lease_blocks_live_range(
            0, static_cast<uint64_t>(std::max(0, prompt.n_tokens())));
    }

    // fork server-context.cpp:1655  (19 lines)
    bool hard_lease_blocks_live_range(
            uint64_t first_token, uint64_t token_count) const {
        if (!lifecycle_authority || !lease_obs || !retention_obs ||
            !lease_execution_identity ||
            prompt.n_tokens() <= 0 || token_count == 0) {
            return false;
        }
        const auto artifact = retention_obs->artifact_id(
            server_retention_instance_key::for_slot(id));
        server_cache_lease_identity identity;
        if (artifact.v == 0 || !server_cache_lease_build_identity(
                *lease_execution_identity, lora_config_identity(lora),
                prompt.tokens, prompt.n_tokens(), identity)) {
            return false;
        }
        return server_cache_hard_lease_blocks_range(
            lease_obs, artifact, identity, prompt.sequence_epoch,
            first_token, token_count);
    }


    // fork server-context.cpp:1998
    server_cache_destruction_admission observe_full_slot(
            server_cache_destruction_class cls,
            server_cache_destruction_reason reason) const {
        GGML_ASSERT(
            cls == server_cache_destruction_class::slot_drop ||
            cls == server_cache_destruction_class::mandatory_recovery_reset);
        server_cache_destruction_admission admission;
        admission.cls    = cls;
        admission.reason = reason;
        admission.issued = true;
        if (!destruction_obs) {
            return admission;
        }
        server_cache_destruction_request request;
        request.cls    = cls;
        request.reason = reason;
        for (const auto kind : {
                server_cache_destruction_target_kind::live_target,
                server_cache_destruction_target_kind::token_ledger,
                server_cache_destruction_target_kind::checkpoint_ring,
                server_cache_destruction_target_kind::rolling_window,
                server_cache_destruction_target_kind::typed_accelerator }) {
            if (kind == server_cache_destruction_target_kind::live_target &&
                retention_obs) {
                request.add_target(
                    kind,
                    id,
                    retention_obs->artifact_id(
                        server_retention_instance_key::for_slot(id)));
            } else {
                request.add_target(kind, id);
            }
        }
        if (ctx_dft) {
            request.add_target(
                server_cache_destruction_target_kind::live_draft, id);
        }
        for (const auto category : {
                llama_cache_acct_category::live_attention_state,
                llama_cache_acct_category::live_recurrent_state,
                llama_cache_acct_category::checkpoint_state_payload,
                llama_cache_acct_category::typed_accelerator_payload,
                llama_cache_acct_category::rolling_window_tape }) {
            request.add_yield(category);
        }
        return server_cache_retention_admit(destruction_obs, request);
    }

    // ---------------------------------------------------------------------
    // group A drop face: the state and the erasers the destruction-admission
    // path needs (slice 8q).  Bodies verbatim from the fork; fork coordinates
    // are on each header line.
    // ---------------------------------------------------------------------

    // fork server-context.cpp:1506
    common_cache_family_binding cache_family;
    // fork server-context.cpp:1519
    bool cache_debug_observability = false;
    // fork server-context.cpp:1521
    bool retention_reuse_pending = false;
    // fork server-context.cpp:1522
    uint64_t retention_reuse_tokens = 0;
    // fork server-context.cpp:1527
    uint64_t vbr_reuse_capture_frontier = 0;
    // fork server-context.cpp:1528
    server_retention_lineage_ticket retention_reuse_source;
    // fork server-context.cpp:1529
    server_retention_lineage_ticket retention_destination;
    // fork server-context.cpp:1530
    bool retention_branch_pending = false;
    // fork server-context.cpp:1531
    bool retention_request_succeeded = false;
    // fork server-context.cpp:1532
    bool retention_geometry_failed = false;
    // fork server-context.cpp:1536
    bool vbr_prompt_cache_enabled = false;
    // fork server-context.cpp:1537
    server_cache_checkpoint_attempt_latch checkpoint_attempts;
    // fork server-context.cpp:1538
    const common_prompt_checkpoint * checkpoint_seam_heuristic = nullptr;
    // fork server-context.cpp:1539
    common_cache_plan_destruction_reason checkpoint_floor_refusal =
        common_cache_plan_destruction_reason::mandatory_anchor;
    // fork server-context.cpp:1541
    std::array<uint8_t, 32> vbr_idle_capture_attempt_identity = {};
    // fork server-context.cpp:1542
    int64_t vbr_idle_capture_retry_after_ms = 0;
    // fork server-context.cpp:1543
    bool vbr_idle_capture_terminal = false;
    // fork server-context.cpp:1544
    bool vbr_idle_capture_isolated_retry = false;
    // fork server-context.cpp:1549
    std::vector<server_vbr_idle_refusal_witness>
        vbr_idle_admission_refusals;
    // fork server-context.cpp:1551
    server_vbr_prompt_cache_support_status vbr_idle_support_status =
        server_vbr_prompt_cache_support_status::supported;
    // fork server-context.cpp:1553
    std::array<uint8_t, 32> vbr_idle_capture_published_identity = {};
    // fork server-context.cpp:1554
    bool vbr_idle_capture_representation_valid = false;
    // fork server-context.cpp:1558
    std::array<uint8_t, 32> vbr_idle_stem_source_identity = {};
    // fork server-context.cpp:1559
    uint64_t vbr_idle_stem_coverage_tokens = 0;
    // fork server-context.cpp:1560
    llama_cache_acct_artifact_id vbr_idle_stem_host_artifact;
    // fork server-context.cpp:1561
    bool vbr_idle_stem_source_valid = false;
    // fork server-context.cpp:1562
    bool vbr_idle_stem_retry = false;
    // fork server-context.cpp:1567
    uint64_t vbr_imported_complete_frontier = 0;
    // NOTE (9l-d2d): 9l-a also declared the artifact store and the six
    // automatic-restore counters on this line. They are server_context_impl
    // members (fork 3562-3575), not server_slot members; nothing referenced
    // them until 9l-d2d landed their first reader, so the misplacement
    // compiled. They now live in the impl's member block, next to the knobs.
    // FORK-DIFF: the fork declares this inside the diffusion self-speculation
    // group (diff_draft_length, the mask/think token ids, diff_prev_logits, ...).
    // Only the flag itself is read by the SWA-window eligibility predicate, and
    // nothing in this tree ever sets it, so the group's other members stay out.
    // fork server-context.cpp:1587-1588
    // diffusion self-speculation
    bool                 diff_self_spec     = false;

    // fork server-context.cpp:1649
    std::list<server_swa_window_checkpoint> swa_windows;
    // fork server-context.cpp:1828
    using checkpoint_iterator = std::list<common_prompt_checkpoint>::iterator;
    // fork server-context.cpp:1830
    server_cache_checkpoint_authority_context checkpoint_authority_context() noexcept {
        return {
            id,
            prompt.checkpoints,
            lifecycle_authority,
            retention_obs,
            destruction_obs,
            lease_obs,
            checkpoint_attempts,
            checkpoint_seam_heuristic,
            checkpoint_floor_refusal,
            cache_debug_observability,
        };
    }

    // fork server-context.cpp:1845
    void checkpoint_ring_changed() noexcept {
        auto context = checkpoint_authority_context();
        server_cache_checkpoint_ring_changed(context);
    }

    // fork server-context.cpp:1855
    void vbr_idle_capture_source_reset() noexcept {
        vbr_idle_capture_attempt_identity = {};
        vbr_idle_capture_retry_after_ms = 0;
        vbr_idle_capture_terminal = false;
        vbr_idle_capture_isolated_retry = false;
        vbr_idle_admission_refusals.clear();
        vbr_idle_support_status =
            server_vbr_prompt_cache_support_status::supported;
        vbr_idle_capture_published_identity = {};
        vbr_idle_capture_representation_valid = false;
        vbr_idle_stem_source_identity = {};
        vbr_idle_stem_coverage_tokens = 0;
        vbr_idle_stem_host_artifact = {};
        vbr_idle_stem_source_valid = false;
        vbr_idle_stem_retry = false;
    }

    // fork server-context.cpp:1872
    void server_cache_slot_drop_impl(bool retire_retention = true) {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());
        if (retire_retention && retention_obs) {
            retention_obs->retire_slot(id);
        }

        // A default-constructed slot exists only in model-free ownership
        // tests. Production slots always carry ctx_tgt and remove their live
        // sequence here before clearing the scheduler image.
        if (mem.ctx_tgt) {
            mem.seq_rm(id, -1, -1);
        }
        common_speculative_sequence_transition(
            get_spec(), id,
            common_speculative_sequence_event::full_clear);

        if (lifecycle_authority && !prompt.checkpoints.empty()) {
            checkpoint_ring_changed();
        }
        prompt.clear();
        swa_windows.clear();
        cache_family = {};
        vbr_reuse_capture_frontier = 0;
        vbr_idle_capture_source_reset();
        vbr_imported_complete_frontier = 0;
        frontier_logits.clear();
    }

// fork 9l-d2a: server_slot::clear_frontier_logits (fork server-context.cpp 1994-1996)
    void clear_frontier_logits() noexcept {
        frontier_logits.clear();
    }

    // fork server-context.cpp:2054-2076
    bool prompt_clear_after_vbr_publication(
            server_cache_destruction_reason reason =
                server_cache_destruction_reason::idle_reclaim) {
        const auto admission = observe_full_slot(
            server_cache_destruction_class::slot_drop,
            reason);
        if (admission.verdict ==
                server_cache_destruction_verdict::would_refuse_hard_leased ||
            admission.verdict ==
                server_cache_destruction_verdict::unavailable) {
            return false;
        }
        // The VBR artifact owns every required companion at this frontier,
        // including the draft-context sequence when speculation is active.
        // Leaving that sequence live would make the otherwise-empty import
        // target fail its required-companion admission on the next request.
        if (ctx_dft && !llama_memory_seq_rm(
                llama_get_memory(ctx_dft), id, -1, -1)) {
            return false;
        }
        server_cache_slot_drop_impl();
        return true;
    }

    // fork server-context.cpp:2078-2113  (36 lines)
    server_cache_destruction_admission observe_live_range_drop(
            server_cache_destruction_reason reason,
            bool include_checkpoints = false) const {
        server_cache_destruction_admission admission;
        admission.cls    = server_cache_destruction_class::live_range_drop;
        admission.reason = reason;
        admission.issued = true;
        if (!destruction_obs) {
            return admission;
        }
        server_cache_destruction_request request;
        request.cls    = server_cache_destruction_class::live_range_drop;
        request.reason = reason;
        request.add_target(
            server_cache_destruction_target_kind::live_target,
            id,
            retention_obs ? retention_obs->artifact_id(
                server_retention_instance_key::for_slot(id))
                : llama_cache_acct_artifact_id{});
        if (ctx_dft) {
            request.add_target(
                server_cache_destruction_target_kind::live_draft, id);
        }
        request.add_target(
            server_cache_destruction_target_kind::token_ledger, id);
        if (include_checkpoints) {
            request.add_target(
                server_cache_destruction_target_kind::checkpoint_ring, id);
        }
        for (const auto category : {
                llama_cache_acct_category::live_attention_state,
                llama_cache_acct_category::live_recurrent_state }) {
            request.add_yield(category);
        }
        return server_cache_retention_admit(destruction_obs, request);
    }

    // fork server-context.cpp:2115
    server_cache_destruction_admission observe_mandatory_recovery_reset(
            server_cache_destruction_reason reason) const {
        return observe_full_slot(
            server_cache_destruction_class::mandatory_recovery_reset,
            reason);
    }

    // fork server-context.cpp:2142
    void mandatory_recovery_reset(server_cache_destruction_reason reason) {
        observe_mandatory_recovery_reset(reason);
        server_cache_slot_drop_impl();
    }

    // fork server-context.cpp:2161-2181
    checkpoint_iterator server_cache_checkpoint_drop_impl(
            checkpoint_iterator first,
            checkpoint_iterator last) {
        if (lifecycle_authority && first != last) {
            checkpoint_ring_changed();
        }
        return prompt.checkpoints.erase(first, last);
    }

    checkpoint_iterator checkpoint_drop_joined_impl(
            checkpoint_iterator first,
            checkpoint_iterator last) {
        if (retention_obs) {
            for (auto it = first; it != last; ++it) {
                retention_obs->retire(
                    server_retention_instance_key::for_checkpoint(
                        id, &*it));
            }
        }
        return server_cache_checkpoint_drop_impl(first, last);
    }

    // fork 9k-b2-iii: observe_checkpoint_drop + checkpoint_drop (fork 2183-2206)
    server_cache_destruction_admission observe_checkpoint_drop(
            server_cache_destruction_reason reason,
            llama_cache_acct_artifact_id artifact = {}) {
        auto context = checkpoint_authority_context();
        return server_cache_checkpoint_observe_drop(context, reason, artifact);
    }

    checkpoint_iterator checkpoint_drop(
            checkpoint_iterator first,
            checkpoint_iterator last,
            server_cache_destruction_reason reason) {
        if (destruction_obs && first != last) {
            (void) observe_checkpoint_drop(
                reason,
                retention_obs ? retention_obs->artifact_id(
                    server_retention_instance_key::for_checkpoint(
                        id, &*first))
                    : llama_cache_acct_artifact_id{});
        }
        // Legacy/pass-through release owner. retention accounting's certified path calls the
        // raw list eraser first, commits the prepared exact release, then
        // retires only the descriptor through retire_after_committed_release.
        return checkpoint_drop_joined_impl(first, last);
    }

    // fork server-context.cpp:2224-2226  — the token_ledger_truncate row's physical
    // choke point. The fallback arm of the restore/trim transaction truncates the
    // ledger to 0 when a rejected suffix trim has already cleared everything.
    void server_cache_token_ledger_truncate_impl(size_t n_tokens) {
        prompt.tokens.keep_first(n_tokens);
    }

    // fork server-context.cpp:2431-2483  (9 + 13 + 29 lines)
    void clear_retention_reuse() noexcept {
        if (retention_obs) {
            retention_obs->release_lineage_ticket(retention_reuse_source);
        } else {
            retention_reuse_source = {};
        }
        retention_reuse_pending = false;
        retention_reuse_tokens = 0;
    }

    void stage_active_prefix_reuse(
            const server_retention_instance_key & source, uint64_t n_tokens) noexcept {
        // Replace the discarded incumbent's receipt, not the destination lineage.
        // This runs only after the prefix has been installed successfully.
        clear_retention_reuse();
        retention_branch_pending = false;
        if (retention_obs && server_prompt_cache_retention_reuse_is_useful(
                    n_tokens, &task->params.message_spans) &&
            retention_obs->acquire_lineage_ticket(source, retention_reuse_source)) {
            retention_reuse_pending = true;
            retention_reuse_tokens = n_tokens;
        }
    }

    void commit_retention_reuse() noexcept {
        if (!retention_reuse_pending) {
            return;
        }
        retention_reuse_pending = false;
        // A later trim or cold replay can supersede a staged reuse receipt.
        retention_reuse_tokens = std::min(retention_reuse_tokens, stats.n_prompt_cached);
        const bool reused_branch = retention_branch_pending;
        common_retention_credit_result credit =
            common_retention_credit_result::unavailable;
        if (retention_obs && server_prompt_cache_retention_reuse_is_useful(
                    retention_reuse_tokens, &task->params.message_spans) &&
            retention_reuse_source.valid()) {
            credit = retention_obs->credit_reuse(retention_reuse_source);
        }
        if (retention_obs) {
            retention_obs->release_lineage_ticket(retention_reuse_source);
        }
        if (vbr_prompt_cache_enabled && reused_branch &&
            retention_reuse_tokens != 0 &&
            credit != common_retention_credit_result::unavailable) {
            vbr_reuse_capture_frontier = retention_reuse_tokens;
        }
        retention_request_succeeded = !retention_branch_pending ||
            !retention_destination.valid() ||
            (retention_obs && retention_obs->activate_lineage_ticket(
                retention_destination));
        retention_reuse_tokens = 0;
    }

    // fork 9l-d2e: abandon a staged launch when the scheduler drops the task (fork server-context.cpp 2485-2489)
    void abandon_retention_prepared_launch() noexcept {
        if (retention_obs) {
            retention_obs->abandon_prepared_launch(
                server_retention_instance_key::for_slot(id));
        }
    }

    // ---- group A slot state (verbatim declarations from the fork) ----

    common_retention_pool retention_pool = common_retention_pool::attention;
    json cache_plan_json;
    std::unique_ptr<common_cache_plan_record> cache_plan;

    // fork server-context.cpp:2289  (4 lines)
    // [obs] human-readable reason for how the last prompt's prefix was (or wasn't) reused, surfaced
    // in GET /slots so a user can see WHY a request cold-processed instead of guessing. Set at the
    // prompt-reuse decision points; greppable by tests (including typed VBR-lineage rejection).
    std::string cache_status;

};

// fork server-context.cpp:815 (definition)
// The dynamic VBR runtime controller flips KV tensor types in place as the context fills; state
// save/restore, context checkpoints and cache reuse all assume a fixed cache layout and would
// restore/reuse bytes under the wrong tier. Gate them off whenever the controller can arm.
// (This replaces the old VBR_STAGE2A env gate — Stage2A itself is gone.)
// fork 9l-d2a: occupied-restore failure classification + its test entry (fork server-context.cpp 3196-3210, 3243-3282)
static bool server_vbr_apply_occupied_restore_failure_to_slot(
        vbr_adopt_status adopt_status,
        vbr_adopt_recovery_outcome recovery,
        size_t incumbent_prefix,
        size_t & retained_prefix,
        server_slot & slot) noexcept {
    const bool quarantined = adopt_status == vbr_adopt_status::quarantined ||
        recovery == vbr_adopt_recovery_outcome::quarantined;
    retained_prefix = quarantined ? 0 : incumbent_prefix;
    if (quarantined) {
        slot.mandatory_recovery_reset(
            server_cache_destruction_reason::restore_failure);
    }
    return quarantined;
}

server_vbr_occupied_quarantine_reset_result
server_vbr_occupied_quarantine_reset_for_test() {
    const common_cache_family_binding family {
        common_cache_family_id { 41 }, common_cache_family_role::branch,
    };
    server_slot replay_slot {};
    replay_slot.id = 0;
    replay_slot.prompt.tokens = server_tokens(
        llama_tokens { 1, 2, 3 }, false);
    replay_slot.prompt.sequence_epoch = 7;
    replay_slot.cache_family = family;
    size_t retained_prefix = 3;
    server_vbr_occupied_quarantine_reset_result result;
    result.replay_preserved_prefix =
        !server_vbr_apply_occupied_restore_failure_to_slot(
            vbr_adopt_status::transfer_failed,
            vbr_adopt_recovery_outcome::replayed,
            retained_prefix, retained_prefix, replay_slot) &&
        retained_prefix == 3;
    result.replay_preserved_slot = replay_slot.prompt.n_tokens() == 3 &&
        replay_slot.prompt.sequence_epoch == 7 &&
        replay_slot.cache_family == family;

    server_slot slot {};
    slot.id = 0;
    slot.prompt.tokens = server_tokens(llama_tokens { 1, 2, 3 }, false);
    slot.prompt.sequence_epoch = 7;
    slot.cache_family = family;
    result.quarantined =
        server_vbr_apply_occupied_restore_failure_to_slot(
            vbr_adopt_status::transfer_failed,
            vbr_adopt_recovery_outcome::quarantined,
            retained_prefix, retained_prefix, slot);
    result.retained_prefix_zero = retained_prefix == 0;
    result.prompt_cleared = slot.prompt.tokens.empty() &&
        slot.prompt.sequence_epoch == 0;
    result.family_cleared = slot.cache_family ==
        common_cache_family_binding {};
    return result;
}

static bool server_vbr_dynamic_active(const common_params & params) {
    return params.vbr_dynamic();
}

// fork 9l-d4b7e: the per-credential VBR tenant key (fork server-context.cpp 823-847)
static std::string server_cache_capture_tenant_key(
        const server_http_req & req) {
    std::string credential =
        req.get_header_value("Authorization");
    if (credential.empty()) {
        credential = req.get_header_value("X-Api-Key");
    }
    if (credential.empty()) {
        credential = "authenticated-local";
    }
    llama_sha256_writer writer;
    static constexpr char domain_label[] =
        "buun.vbr.server-tenant/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.string(credential.data(), credential.size());
    const auto digest = writer.finish();
    static constexpr char HEX[] = "0123456789abcdef";
    std::string output;
    output.reserve(32);
    for (size_t i = 0; i < 16; ++i) {
        output.push_back(HEX[digest[i] >> 4]);
        output.push_back(HEX[digest[i] & 0x0f]);
    }
    return output;
}

// fork 9l-d4b7g: server_cache_import_route_consistency (fork server-context.cpp
// 849-861).  Both of its call sites are inside this file.
// ONE consistency derivation for both the import JSON result and the terminal
// log line — a failed import is 'unavailable' on both surfaces, never a
// fabricated live_rebased.
static server_cache_import_consistency server_cache_import_route_consistency(
        server_vbr_artifact_import_status status,
        vbr_artifact_consistency_kind kind) {
    if (status != server_vbr_artifact_import_status::ok) {
        return server_cache_import_consistency::unavailable;
    }
    return kind == vbr_artifact_consistency_kind::capture_exact
        ? server_cache_import_consistency::capture_exact
        : server_cache_import_consistency::live_rebased;
}

// A flip is deliberately per-boot: persisting migration confidence across a
// binary/model/config change would turn yesterday's evidence into authority for
// different code. Zero keeps the frontier path in permanent shadow mode.
static uint64_t server_frontier_ratchet_min_agreements() {
    static constexpr uint64_t DEFAULT_MIN_AGREEMENTS = 1024;

    const char * env = std::getenv("LLAMA_FRONTIER_RATCHET_MIN_AGREEMENTS");
    if (env == nullptr || env[0] == '\0') {
        return DEFAULT_MIN_AGREEMENTS;
    }

    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0') {
        SRV_WRN("invalid LLAMA_FRONTIER_RATCHET_MIN_AGREEMENTS='%s'; using %" PRIu64 "\n",
                env, DEFAULT_MIN_AGREEMENTS);
        return DEFAULT_MIN_AGREEMENTS;
    }

    return (uint64_t) parsed;
}

// fork 9l-d4b7h: server_reset_processing_slots_after_committed_failure
// (fork server-context.cpp 3094-3102).  Lands ahead of the decode() rewrite:
// both of its callers (the decode() body and the *_for_test layer) are still
// unported, so it is defined-but-uncalled until those land.
static void server_reset_processing_slots_after_committed_failure(
        std::vector<server_slot> & slots) {
    for (auto & slot : slots) {
        if (slot.is_processing()) {
            slot.mandatory_recovery_reset(
                server_cache_destruction_reason::restore_failure);
        }
    }
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                void * cb_data = slot.spec;
                static auto cb = [](llama_batch batch, void * user_data) {
                    common_speculative * spec = static_cast<common_speculative *>(user_data);
                    if (!common_speculative_process(spec, batch)) {
                        return 1;
                    }
                    return 0;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

// A projector GPU swap is a destructive two-phase transition: once the draft context has
// been torn down to free VRAM, every exit from the media interval must restore it exactly
// once. In particular, server_queue::yield_to_queue() may rethrow either the media work
// exception or an exception raised by a concurrently serviced callback.
class server_mmproj_restore_guard {
public:
    explicit server_mmproj_restore_guard(std::function<void()> restore) :
        restore_(std::move(restore)) {}

    server_mmproj_restore_guard(const server_mmproj_restore_guard &) = delete;
    server_mmproj_restore_guard & operator=(const server_mmproj_restore_guard &) = delete;

    ~server_mmproj_restore_guard() noexcept {
        if (!armed_) {
            return;
        }
        try {
            restore_now();
        } catch (const std::exception & e) {
            SRV_ERR("mmproj restore failed: %s\n", e.what());
        } catch (...) {
            SRV_ERR("%s", "mmproj restore failed\n");
        }
    }

    void arm() noexcept {
        armed_ = true;
    }

    void restore_now() {
        if (!armed_) {
            return;
        }
        // Clear before invoking the callback. If restoration itself throws, unwinding must
        // not attempt the same destructive transition twice.
        armed_ = false;
        restore_();
    }

private:
    std::function<void()> restore_;
    bool armed_ = false;
};

// fork 9l-d3c2: retention metadata wiring for standalone owners (fork server-context.cpp 3373-3410)
static void server_wire_live_retention_metadata(
        server_retention_sidecar_store & owner,
        std::vector<server_slot> & slots,
        common_retention_pool pool) {
    for (auto & slot : slots) {
        slot.retention_obs = &owner;
        slot.retention_pool = pool;
    }
}

static void server_request_retention_prefix_tracking(
        server_retention_sidecar_store & owner,
        server_prompt_cache * prompt_cache,
        const server_retention_owner_plan & plan) {
    bool workspace_ready = !plan.prompt_shadow_workspace;
    if (prompt_cache && plan.prompt_shadow_workspace) {
        workspace_ready = prompt_cache->enable_retention_shadow();
    }
    if (plan.prefix_tracking && workspace_ready) {
        (void) owner.enable_prefix_tracking();
    }
}

static void server_wire_standalone_retention_metadata(
        server_retention_sidecar_store & owner,
        std::vector<server_slot> & slots,
        server_prompt_cache * prompt_cache,
        common_retention_pool pool,
        const server_retention_owner_plan & plan) {
    GGML_ASSERT(plan.owner ==
        server_retention_owner_kind::standalone_metadata);
    owner.configure(nullptr, {}, nullptr);
    server_wire_live_retention_metadata(owner, slots, pool);
    if (prompt_cache) {
        prompt_cache->retention_obs = &owner;
    }
    server_request_retention_prefix_tracking(owner, prompt_cache, plan);
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    // In-process execution/lineage namespace for computation-frontier records.
    // Checkpoints are never portable across this random model-instance key;
    // slot-file restore explicitly invalidates them below.
    std::string frontier_execution_identity;
    // fork server-context.cpp:3521-3522
    std::array<uint8_t, 32> swa_window_execution_identity {};
    int32_t swa_window_checkpoint_limit = 0;
    // Stable across server restarts, unlike frontier_execution_identity.  It
    // exists only when slot-file persistence is enabled and binds the optional
    // logits companion to the exact model/execution configuration.
    server_slot_runtime_identity slot_file_runtime_identity;
    uint64_t frontier_next_sequence_epoch = 1;
    // fork server-context.cpp:3528
    uint64_t frontier_ratchet_threshold = 1024;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    // ---------------------------------------------------------------------
    // group C tip: the server_context_impl handles the cache-plan observation
    // face dereferences. The authority/retention machinery behind them is NOT
    // part of this landing -- only the handles, so the class compiles. They
    // stay null until the constructor wiring lands; every reader already
    // null-checks (the fork guarantees it, because an ordinary fixed-cache run
    // has no authority at all).
    // Deliberately absent: cache_retention_metadata (no reader in the spliced
    // set) and vbr_idle_frontier_queries (needs the frontier-query type, which
    // is only reachable through the retention wave, not through observation).
    // Verbatim declarations from the fork, coordinates on each line.
    // ---------------------------------------------------------------------
    // fork server-context.cpp:3534
    std::unique_ptr<server_cache_authority> cache_authority;
    // fork server-context.cpp:3481
    llama_context_ptr ctx_dft_shared;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

// fork 9l-d2b: prompt-admission geometry test door (fork server-context.cpp 5111-5120)
    struct slot_prompt_admission_geometry {
        bool can_split = true;
        int32_t n_ubatch = 0;
    };

    // Deterministic model-free door for the rejected-launch regression. No
    // production caller installs an override.
    const slot_prompt_admission_geometry * prompt_admission_geometry_for_test =
        nullptr;
    uint64_t prompt_selection_calls_for_test = 0;

    // MTP <-> mmproj GPU swap state: the MTP draft context is destroyed for the duration of
    // a media encode so the projector can use its VRAM, then recreated
    bool mmproj_gpu_swap   = false;
    bool mmproj_is_on_gpu  = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    // build the mtmd context params the same way load_model() does; the reload path only
    // parameterizes device placement, everything else has to match the initial load
    mtmd_context_params make_mmproj_params(bool use_gpu) const {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = use_gpu;
        mparams.device           = params_base.mmproj_device;
        mparams.print_timings    = false;
        mparams.n_threads        = params_base.cpuparams.n_threads;
        mparams.flash_attn_type  = params_base.flash_attn_type;
        mparams.warmup           = use_gpu; // only warm up when the projector actually lands on the GPU
        mparams.image_min_tokens = params_base.image_min_tokens;
        mparams.image_max_tokens = params_base.image_max_tokens;
        mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
        mparams.media_marker     = get_media_marker();
        return mparams;
    }

    // drop the current projector and load it again on the requested device. Slot state that
    // is derived from the projector (the mtmd batch is cached between chunks) dies with it.
    void reload_mmproj(bool use_gpu) {
        for (server_slot & slot : slots) {
            slot.mbatch.reset();
        }

        mtmd_free(mctx);
        mctx = nullptr;
        mctx = mtmd_init_from_file(params_base.mmproj.path.c_str(), model_tgt, make_mmproj_params(use_gpu));

        for (server_slot & slot : slots) {
            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;
        }
    }

    // point (or unpoint) every slot at a draft context and its speculative owner
    void bind_slots_to_draft_context(llama_context * draft_context, common_speculative * shared_spec) {
        for (server_slot & slot : slots) {
            slot.spec    = shared_spec;
            slot.ctx_dft = draft_context;
            slot.mem.init(slot.ctx_tgt, draft_context);
            // fork server-context.cpp:5336
            slot.vbr_prompt_cache_enabled = params_base.vbr_prompt_cache;
            if (draft_context == nullptr) {
                // Cached draft state only carries meaning for the context that produced it.
                // It has to go as a set: draft generation asserts that a reused partial draft
                // comes with its checkpoint, so clearing one without the other would abort.
                slot.spec_draft.clear();
                slot.spec_i_batch.clear();
                slot.spec_ckpt.clear();
                slot.spec_is_replay = false;
            }
        }
    }

    void unbind_slots_from_draft_context() {
        bind_slots_to_draft_context(nullptr, nullptr);
        spec.reset();
    }

    // recreate the MTP draft context and its speculative owner after the projector swap tore
    // them down. Mirrors the speculative init block in load_model().
    bool recreate_mtp_context() {
        common_params params_dft = common_base_params_to_speculative(params_base);

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        model_dft = spec_init->model();
        ctx_dft   = spec_init->context();

        if (ctx_dft == nullptr) {
            SRV_ERR("%s", "failed to recreate the MTP context after the mmproj swap\n");
            spec_init.reset();
            model_dft = nullptr;
            return false;
        }

        params_base.speculative.draft.ctx_tgt = ctx_tgt;
        params_base.speculative.draft.ctx_dft = ctx_dft;

        ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);

        try {
            spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
        } catch (const std::exception & e) {
            SRV_ERR("failed to reinitialize speculative decoding after the mmproj swap: %s\n", e.what());
            return false;
        }

        if (!spec) {
            SRV_ERR("%s", "failed to reinitialize speculative decoding after the mmproj swap\n");
            return false;
        }

        bind_slots_to_draft_context(ctx_dft, spec.get());

        return true;
    }

    // destroy the MTP draft context so the projector can have its VRAM, then load the
    // projector onto the GPU. Returns false if the projector could not be brought up at all.
    bool swap_spec_to_mmproj_gpu() noexcept {
        SRV_INF("%s", "swapping MTP out of VRAM, loading mmproj to GPU...\n");
        const int64_t t0 = ggml_time_us();

        mmproj_is_on_gpu = false;

        try {
            // every slot must stop referring to the draft context before it is destroyed
            unbind_slots_from_draft_context();
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
            params_base.speculative.draft.ctx_dft = nullptr;

            reload_mmproj(true);
            if (mctx == nullptr) {
                SRV_ERR("%s", "failed to load mmproj to GPU, falling back to CPU\n");
                reload_mmproj(false);
            } else {
                mmproj_is_on_gpu = true;
            }

            SRV_INF("MTP->mmproj swap done in %" PRId64 " ms\n", (ggml_time_us() - t0) / 1000);
            return mctx != nullptr;
        } catch (const std::exception & e) {
            SRV_ERR("mmproj swap-out failed: %s\n", e.what());
        } catch (...) {
            SRV_ERR("%s", "mmproj swap-out failed\n");
        }

        // the caller's armed restore guard reconstructs speculation; keep every slot
        // target-only until it runs
        unbind_slots_from_draft_context();
        mmproj_is_on_gpu = false;
        return false;
    }

    void swap_mmproj_to_spec() noexcept {
        SRV_INF("%s", "unloading mmproj from the GPU, recreating MTP...\n");
        const int64_t t0 = ggml_time_us();

        mmproj_is_on_gpu = false;

        try {
            reload_mmproj(false);
            if (!recreate_mtp_context()) {
                SRV_ERR("%s", "continuing without speculative decoding after the mmproj swap\n");
                return;
            }
            SRV_INF("mmproj->MTP swap done in %" PRId64 " ms\n", (ggml_time_us() - t0) / 1000);
        } catch (const std::exception & e) {
            SRV_ERR("mmproj swap-back failed: %s\n", e.what());
        } catch (...) {
            SRV_ERR("%s", "mmproj swap-back failed\n");
        }
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;
        // fork 9l-d3e: resolve the host prompt-cache policy from the CLI (fork server-context.cpp 6263-6271)
        const auto vbr_prompt_cache_mode =
            common_vbr_prompt_cache_mode_for(params_base);
        const bool vbr_prompt_cache_automatic =
            vbr_prompt_cache_mode ==
                common_vbr_prompt_cache_mode::enabled_automatic;
        params_base.vbr_prompt_cache =
            vbr_prompt_cache_automatic ||
            vbr_prompt_cache_mode ==
                common_vbr_prompt_cache_mode::enabled_explicit;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        const std::string & mmproj_path = params_base.mmproj.path;

        // The GPU swap destroys the draft context for the duration of an image encode, which
        // is only viable for MTP: it is recreated against the target model without reloading
        // anything. An external draft model would have to be loaded from disk again, so the
        // projector and such a drafter stay resident together.
        const bool mmproj_swap_applicable = params_base.mmproj_gpu_swap && !has_draft && spec_mtp;
        if (params_base.mmproj_gpu_swap && !mmproj_swap_applicable && has_mmproj) {
            SRV_WRN("%s", "mmproj GPU swap requires the MTP draft context; keeping the projector and the drafter resident\n");
        }

        // with the swap enabled the projector starts on the CPU and only moves to the GPU for
        // the duration of an image encode
        mtmd_context_params mparams = make_mmproj_params(
                mmproj_swap_applicable ? false : params_base.mmproj_use_gpu);
        if (has_mmproj) {
            if (!mmproj_swap_applicable) {
                mparams.warmup = params_base.warmup;
            }
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        // with the GPU swap the projector and the drafter are phase-exclusive, so the
        // projector does not need a permanent reservation
        if (has_mmproj && params_base.fit_params && !mmproj_swap_applicable) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }

            // the swap needs a live MTP context to tear down and recreate
            mmproj_gpu_swap = mmproj_swap_applicable && ctx_dft != nullptr;
            if (mmproj_gpu_swap) {
                SRV_INF("loaded multimodal model on CPU (GPU swap enabled), '%s'\n", mmproj_path.c_str());
            } else {
                SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());
            }

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        // cache receipt (§7.7 security contract): keyed by default; an unkeyed
        // chain leaks prompt-content comparability and requires the explicit
        // debug flag. Fail closed at startup, never silently downgrade.
        if (params_base.cache_receipt &&
            params_base.cache_receipt_key.empty() &&
            !params_base.cache_receipt_unkeyed_debug) {
            SRV_ERR("%s\n", "--cache-receipt requires --cache-receipt-key "
                    "(or --cache-receipt-unkeyed-debug for trusted local use)");
            return false;
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();
        frontier_execution_identity =
            "server-execution-v1:" + random_string();
        frontier_next_sequence_epoch = 1;
        frontier_ratchet_threshold =
            server_frontier_ratchet_min_agreements();
        SRV_INF(
            "computation-frontier ratchet: legacy reads authoritative, "
            "flip after %" PRIu64 " qualifying agreements%s\n",
            frontier_ratchet_threshold,
            frontier_ratchet_threshold == 0 ? " (shadow-only)" : "");
        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            // fork server-context.cpp:7423
            slot.vbr_prompt_cache_enabled = params_base.vbr_prompt_cache;
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel), n_embd);
        }

        if (server_vbr_dynamic_active(params_base)) {
            // Disabled: these mechanisms serialize/shift the ATTENTION KV, whose tensor tiers flip
            // in place at runtime under the dynamic VBR controller (a FLAGS_NONE state restore
            // would land bytes under the wrong tier or past the VMM watermark; cache_reuse needs
            // shifts, and get_can_shift() is false under VMM anyway).
            if (params_base.cache_ram_mib != 0 &&
                !params_base.vbr_prompt_cache) {
                params_base.cache_ram_mib = 0;
                SRV_WRN("%s\n", "dynamic VBR prompt-cache artifacts were explicitly disabled; fixed KV state images are unsafe and cache-ram will be disabled");
            } else if (params_base.cache_ram_mib != 0) {
                SRV_INF("%s\n", "dynamic VBR idle caching will store projected artifacts; fixed KV state images remain disabled");
            }
            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by dynamic VBR (KV tiers change at runtime), it will be disabled");
            }
            if (!params_base.slot_save_path.empty()) {
                // llama_state_seq_save_file carries full tier-typed attention KV: a save taken
                // after any degrade is refused at the lib level, and even entry-tier saves stop
                // restoring once the target cache has degraded — predictably off beats flaky
                params_base.slot_save_path.clear();
                SRV_WRN("%s\n", "slot save/restore (--slot-save-path) is not supported by dynamic VBR (KV tiers change at runtime), it will be disabled");
            }
        }

        // Dynamic VBR may have disabled slot files above.  Hash model shards,
        // controls, and LoRA sources only for an actually reachable slot-file
        // endpoint; ordinary servers and the disabled path do no file IO.
        if (!params_base.slot_save_path.empty()) {
            slot_file_runtime_identity =
                server_slot_file_runtime_identity_build(
                    model_tgt, ctx_tgt, params_base);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);

            // fork 9l-d3d: VBR quality-anchor budget (fork server-context.cpp 7683-7695)
            if (params_base.vbr_prompt_cache &&
                params_base.vbr_anchor_cache_mib > 0) {
                const uint64_t anchor_bytes =
                    uint64_t(params_base.vbr_anchor_cache_mib)*1024*1024;
                if (anchor_bytes > SIZE_MAX) {
                    SRV_ERR("%s", "--vbr-anchor-cache-mib exceeds this platform's addressable size\n");
                    return false;
                }
                prompt_cache->quality_anchor_budget_enabled = true;
                prompt_cache->limit_anchor_size = size_t(anchor_bytes);
                SRV_INF("VBR quality-anchor cache enabled, size limit: %d MiB\n",
                        params_base.vbr_anchor_cache_mib);
            }
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        // fork 9l-d3d: cache authority substrate, constructed and configured (fork server-context.cpp 7701-7827)
        // Retention-capacity policy is the normal fixed-cache owner. Explicit
        // --cache-lifecycle still enables the transaction substrate without a
        // host cache; cache-disabled defaults remain zero-state.
        params_base.cache_lifecycle = server_cache_lifecycle_default(
            params_base.cache_lifecycle, prompt_cache != nullptr,
            params_base.cache_control_api);
        const bool retention_capacity = prompt_cache != nullptr;
        const auto retention_owner_plan = server_retention_owner_plan_for(
            params_base.cache_debug,
            params_base.cache_lifecycle,
            prompt_cache != nullptr,
            server_vbr_dynamic_active(params_base),
            retention_capacity);
        const auto live_retention_pool =
            (llama_model_is_recurrent(model_tgt) ||
             llama_model_is_hybrid(model_tgt))
                ? common_retention_pool::recurrent
                : common_retention_pool::attention;

        // Cache authority substrate, constructed and configured under
        // (cache_debug || cache_lifecycle). Neither flag remains the strictly-zero-work legacy
        // path; debug alone observes the shared substrate, while lifecycle enables publication
        // admission without allocating or emitting a cache-plan observer.
        if (params_base.cache_debug) {
            cache_plan_obs = std::make_unique<server_cache_plan_observer>();
        }
        if (retention_owner_plan.owner ==
                server_retention_owner_kind::authority) {
            cache_authority = std::make_unique<server_cache_authority>();
        } else if (retention_owner_plan.owner ==
                server_retention_owner_kind::standalone_metadata) {
            cache_retention_metadata =
                std::make_unique<server_retention_sidecar_store>();
        }
        if (retention_owner_plan.owner !=
                server_retention_owner_kind::none) {
            // Live-slot retention is representation-neutral. Fixed and
            // dynamic-VBR memories both expose exact per-sequence exclusive
            // cell counts; unsupported composite topologies fail closed to
            // the protected LRU floor.
            vbr_retention_candidates.resize(slots.size());
            vbr_retention_candidate_by_seq.resize(slots.size());
            vbr_retention_seen_by_seq.resize(slots.size());
            vbr_exclusive_cells_by_seq.resize(slots.size());
        }

        // Cache-plan observer, constructed only under params_base.cache_debug: the observer
        // object and both surfaces exist iff the flag is set. Off = strictly zero observer work
        // It records the shared authority substrate above, owned by server_context.
        if (retention_owner_plan.owner ==
                server_retention_owner_kind::authority) {
            GGML_ASSERT(cache_authority);
            // Policy substrate is live under either gate. CACHE_PLAN JSON/log
            // emission remains below behind cache_plan_obs/cache_debug, but
            // lifecycle-only operation must still inspect WS-D leases before
            // every censused destructive primitive.
            cache_authority->destruction.lease_context = &cache_authority->leases;
            cache_authority->destruction.lease_evaluator =
                server_cache_lease_evaluate_request;
            server_wire_live_retention_metadata(
                cache_authority->retention, slots, live_retention_pool);
            for (auto & slot : slots) {
                slot.destruction_obs = &cache_authority->destruction;
                slot.lease_obs = &cache_authority->leases;
                slot.lease_execution_identity = &frontier_execution_identity;
                slot.lifecycle_authority = params_base.cache_lifecycle
                    ? cache_authority.get()
                    : nullptr;
                slot.cache_debug_observability = params_base.cache_debug;
            }
            if (prompt_cache) {
                prompt_cache->debug_observability = params_base.cache_debug;
                prompt_cache->retention_capacity_authority = retention_capacity;
                prompt_cache->destruction_obs = &cache_authority->destruction;
                prompt_cache->retention_obs = &cache_authority->retention;
                prompt_cache->lease_obs = &cache_authority->leases;
                prompt_cache->lease_execution_identity =
                    &frontier_execution_identity;
                // Debug shadow observes whichever fixed-host authority is
                // active. Lifecycle keeps its certified price selector; observation
                // only compares the decayed projection against that choice.
                server_request_retention_prefix_tracking(
                    cache_authority->retention, prompt_cache.get(),
                    retention_owner_plan);
                if (retention_capacity) {
                    SRV_INF("%s\n", "decayed-frequency host-cache authority is enabled");
                }
            } else if (retention_owner_plan.prefix_tracking) {
                server_request_retention_prefix_tracking(
                    cache_authority->retention, nullptr,
                    retention_owner_plan);
            }
            if (params_base.cache_lifecycle) {
                llama_get_memory(ctx_tgt)->vbr_hard_seal_guard_set(
                    vbr_hard_seal_guard {
                        [this]() {
                            return cache_authority != nullptr &&
                                server_cache_has_hard_lease(
                                    &cache_authority->leases);
                        },
                        [this](const vbr_hard_seal_subject &,
                               const std::vector<vbr_hard_seal_range> & ranges) {
                            for (const auto & range : ranges) {
                                const auto slot = std::find_if(
                                    slots.begin(), slots.end(),
                                    [&](const server_slot & value) {
                                        return value.id == range.sequence;
                                    });
                                if (slot != slots.end() &&
                                    slot->hard_lease_blocks_live_range(
                                        range.first_token, range.token_count)) {
                                    return vbr_hard_seal_guard_result::hard_lease_blocked;
                                }
                            }
                            return vbr_hard_seal_guard_result::allow;
                        },
                    });
            }
        }
        if (cache_retention_metadata) {
            // The retention sidecar owns payload-independent live lineage metadata for both the normal fixed
            // host-cache path and dynamic VBR. This does not enable lifecycle accounting,
            // leases, host-state serialization, or victim authority.
            server_wire_standalone_retention_metadata(
                *cache_retention_metadata, slots, prompt_cache.get(),
                live_retention_pool, retention_owner_plan);
        }

        // fork 9l-d3e: capture manifest + the VBR artifact store (fork server-context.cpp 7829-8365)
        std::vector<std::string> gpu_descs;
        std::vector<vbr_artifact_portable_topology>
            capture_topologies;
        std::vector<vbr_explicit_capture_runtime_pool>
            capture_runtime_pools;
        std::vector<vbr_explicit_capture_pool_binding>
            capture_pool_bindings;
        std::vector<vbr_capture_lane> capture_lanes;
        uint32_t capture_attention_children = 0;
        bool capture_manifest_enabled = false;
        size_t capture_resolved_split_count = 0;
        const char * capture_topology_failure =
            "cache_authority_unavailable";
        const bool capture_requested =
            params_base.cache_lifecycle &&
            server_vbr_dynamic_active(params_base);
        int ngl_eff = 0;
        if (cache_authority) {
            capture_topology_failure = "topology_not_built";
            const auto observer_domain = llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::not_applicable);
            const auto host_domain = llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host);

            // Manifest ordering: resolve effective placement, build/intern every
            // usable device domain, and retain its
            // runtime device binding BEFORE the one-shot producer manifest is configured.
            // The topology must use the loader's effective placement weights.
            // Raw CLI weights are all zero for automatic multi-device placement,
            // while the loader has already resolved them from device capacity.
            std::vector<ggml_backend_dev_t> gpu_devices;
            std::vector<std::string> gpu_identities;
            if (params_base.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                const int32_t n_model_devices =
                    llama_model_n_devices(model_tgt);
                for (int32_t i = 0; i < n_model_devices; ++i) {
                    auto * dev = llama_model_get_device(model_tgt, i);
                    if (dev != nullptr &&
                        ggml_backend_dev_type(dev) ==
                            GGML_BACKEND_DEVICE_TYPE_GPU) {
                        gpu_devices.push_back(dev);
                        gpu_descs.push_back(
                            ggml_backend_dev_description(dev));
                        gpu_identities.push_back(
                            std::string(ggml_backend_dev_name(dev)) + "\n" +
                            ggml_backend_dev_description(dev));
                    }
                }
            } else if (!params_base.devices.empty()) {
                // the explicit --device list is nullptr-TERMINATED (parse_device_list);
                // "none" is a single nullptr -> empty -> "cpu"
                for (auto * dev : params_base.devices) {
                    if (dev) {
                        gpu_devices.push_back(dev);
                        gpu_descs.push_back(ggml_backend_dev_description(dev));
                        gpu_identities.push_back(
                            std::string(ggml_backend_dev_name(dev)) + "\n" +
                            ggml_backend_dev_description(dev));
                    }
                }
            } else {
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    auto * dev = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        gpu_devices.push_back(dev);
                        gpu_descs.push_back(ggml_backend_dev_description(dev));
                        gpu_identities.push_back(
                            std::string(ggml_backend_dev_name(dev)) + "\n" +
                            ggml_backend_dev_description(dev));
                    }
                }
            }
            // Auto-fit writes the resolved layer count back into params. A remaining
            // negative sentinel means the loader's negative-is-all default.
            ngl_eff = params_base.n_gpu_layers >= 0
                ? params_base.n_gpu_layers
                : llama_model_n_layer(model_tgt) + 1;
            if (ngl_eff > 0 && !gpu_identities.empty()) {
                std::vector<float> resolved_split(gpu_identities.size());
                const size_t resolved_count =
                    llama_model_resolved_tensor_split(
                        model_tgt, resolved_split.data(),
                        resolved_split.size());
                capture_resolved_split_count = resolved_count;
                llama_cache_acct_shard_topology topology;
                const int32_t topology_main_device =
                    params_base.split_mode == LLAMA_SPLIT_MODE_NONE
                        ? 0
                        : params_base.main_gpu;
                if (resolved_count == gpu_identities.size() &&
                    llama_cache_acct_build_shard_topology(
                        gpu_identities,
                        int16_t(params_base.split_mode),
                        topology_main_device,
                        resolved_split.data(),
                        topology)) {
                    if (capture_requested) {
                        capture_topologies.push_back(topology);
                    }
                    for (size_t i = 0; i < gpu_identities.size(); ++i) {
                        llama_cache_acct_resource_domain domain;
                        if (!cache_authority->ledger.make_device_domain(
                                topology,
                                llama_cache_acct_device_ordinal{ uint16_t(i) },
                                domain)) {
                            cache_authority_config_failed(true);
                            cache_authority->live_device_domains.clear();
                            break;
                        }
                        cache_authority->live_device_domains.push_back({
                            gpu_devices[i], domain,
                        });
                    }
                    if (cache_authority->configured) {
                        capture_topology_failure =
                            "runtime_pool_discovery_failed";
                    } else {
                        capture_topology_failure =
                            "device_domain_unavailable";
                    }
                } else {
                    capture_topology_failure =
                        resolved_count == gpu_identities.size()
                            ? "invalid_resolved_split"
                            : "resolved_split_device_mismatch";
                    cache_authority_config_failed(true);
                }
            } else {
                capture_topology_failure = ngl_eff <= 0
                    ? "no_gpu_layers"
                    : "no_gpu_model_device";
            }

            // Capture construction discovery is read-only and occurs before the
            // one-shot C manifest. Every attention child and every pool must
            // be armed and exactly attributable to a topology device.
            if (capture_requested &&
                cache_authority->configured &&
                capture_topologies.size() == 1 &&
                vbr_explicit_capture_runtime_pools(
                    *llama_get_memory(ctx_tgt),
                    capture_runtime_pools,
                    capture_attention_children)) {
                capture_topology_failure =
                    "runtime_pool_binding_failed";
                for (const auto & pool : capture_runtime_pools) {
                    const auto domain = std::find_if(
                        cache_authority->live_device_domains.begin(),
                        cache_authority->live_device_domains.end(),
                        [&](const auto & binding) {
                            return binding.device ==
                                pool.backend_device;
                        });
                    if (domain ==
                        cache_authority->live_device_domains.end()) {
                        capture_pool_bindings.clear();
                        capture_lanes.clear();
                        break;
                    }
                    auto lane = std::find_if(
                        capture_lanes.begin(), capture_lanes.end(),
                        [&](const auto & candidate) {
                            return candidate.device ==
                                pool.backend_device;
                        });
                    uint32_t lane_index = 0;
                    if (lane == capture_lanes.end()) {
                        ggml_backend_t ring_backend = pool.backend;
                        if (ring_backend == nullptr) {
                            ring_backend =
                                ctx_tgt->backend_for_device(
                                    pool.backend_device);
                            if (ring_backend == nullptr) {
                                capture_pool_bindings.clear();
                                capture_lanes.clear();
                                break;
                            }
                        }
                        lane_index = uint32_t(capture_lanes.size());
                        capture_lanes.push_back({
                            pool.backend_device, ring_backend, false,
                        });
                    } else {
                        lane_index = uint32_t(
                            lane - capture_lanes.begin());
                    }
                    capture_pool_bindings.push_back({
                        pool.instance_id,
                        pool.device,
                        0,
                        domain->domain.device_ordinal.v,
                        lane_index,
                    });
                }
                capture_manifest_enabled =
                    !capture_pool_bindings.empty() &&
                    !capture_lanes.empty() &&
                    capture_pool_bindings.size() ==
                        capture_runtime_pools.size();
                if (capture_manifest_enabled) {
                    capture_topology_failure = "none";
                }
            }

            // C schema-v2 completeness manifest is owned by configuration, not by either
            // producer. Device/live-memory rows are part of this SAME one-shot call; no
            // producer is allowed to append its own domain after cells exist.
            // Artifact cells and the pinned domain therefore precede ring-budget
            // admission. If that later admission refuses the ring, these manifested
            // cells remain certified measured-zero. This is intentional, harmless,
            // limited to lifecycle+armed VBR, and must not become a second manifest.
            std::vector<llama_cache_acct_completeness_requirement> required;
            required.reserve(
                4 + cache_authority->live_device_domains.size());
            required.push_back({
                observer_domain, llama_cache_acct_producer::observer_init,
            });
            if (prompt_cache) {
                required.push_back({
                    host_domain, llama_cache_acct_producer::host_cache,
                });
            }
            required.push_back({
                host_domain, llama_cache_acct_producer::retention_sidecar,
            });
            const auto pinned_domain =
                llama_cache_acct_resource_domain::non_device(
                    llama_cache_acct_residency::pinned_host);
            if (capture_manifest_enabled) {
                required.push_back({
                    pinned_domain,
                    llama_cache_acct_producer::retention_sidecar,
                });
            }
            for (const auto & binding : cache_authority->live_device_domains) {
                required.push_back({
                    binding.domain, llama_cache_acct_producer::live_memory,
                });
            }
            if (!cache_authority->ledger.configure_required_producers(
                    required.data(), required.size())) {
                cache_authority_config_failed(false);
            }
            if (capture_manifest_enabled) {
                // Capture reservations span pageable metadata/companions and
                // topology-qualified device payloads. This runs BEFORE the
                // ordinary host/live producers below, so it co-initializes every
                // host-scope participating cell in this domain to measured-zero —
                // including the legacy host-cache cells (full_snapshot/checkpoint_state/
                // typed_accelerator) — so the host domain is complete when the
                // store's first host reservation prices it. The later host-accounting block
                // re-gauges those same cells to the same 0 (gauge_set is idempotent),
                // so this co-init is load-bearing but behavior-neutral.
                if (!server_vbr_artifact_store_observe_empty_accounting(
                        cache_authority->ledger, host_domain)) {
                    cache_authority_config_failed(true);
                }
                for (const auto & binding :
                        cache_authority->live_device_domains) {
                    if (!server_vbr_artifact_store_observe_empty_accounting(
                            cache_authority->ledger,
                            binding.domain)) {
                        cache_authority_config_failed(true);
                    }
                }
                if (!server_vbr_artifact_store_configure_pinned_accounting(
                        cache_authority->ledger, pinned_domain)) {
                    cache_authority_config_failed(true);
                }
            }
            cache_authority->retention.configure(
                &cache_authority->ledger, host_domain, &cache_authority->leases);
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::
                        artifact_descriptor_metadata,
                    host_domain, measure, 0);
            }

            // Host-cache absence is itself an observed zero. Initialize the
            // transactional leaves before an optional cache attaches; later C
            // transactions update these same resident/reserved cells.
            for (const auto cat : {
                    llama_cache_acct_category::full_snapshot_payload,
                    llama_cache_acct_category::checkpoint_state_payload,
                    llama_cache_acct_category::typed_accelerator_payload,
                    llama_cache_acct_category::transfer_staging }) {
                for (const auto measure : {
                        llama_cache_acct_measure::logical_payload,
                        llama_cache_acct_measure::resident_allocated,
                        llama_cache_acct_measure::reserved }) {
                    cache_authority->ledger.gauge_set(
                        cat, host_domain, measure, 0);
                }
            }
            if (prompt_cache) {
                // Empty-cache gauges are real observations. Certification follows them,
                // rather than merely proving that wiring code ran.
                (void) cache_authority->ledger.certify_complete(
                    host_domain, llama_cache_acct_producer::host_cache);
            }
            if (capture_manifest_enabled) {
                for (const auto category : {
                        llama_cache_acct_category::
                            artifact_reference_metadata,
                        llama_cache_acct_category::transfer_staging,
                        llama_cache_acct_category::codec_workspace }) {
                    for (const auto measure : {
                            llama_cache_acct_measure::logical_payload,
                            llama_cache_acct_measure::resident_allocated,
                            llama_cache_acct_measure::reserved }) {
                        cache_authority->ledger.gauge_set(
                            category, host_domain, measure, 0);
                    }
                }
                for (const auto & binding :
                        cache_authority->live_device_domains) {
                    for (const auto category : {
                            llama_cache_acct_category::unit_version_payload,
                            llama_cache_acct_category::clean_stash_payload,
                            llama_cache_acct_category::transfer_staging,
                            llama_cache_acct_category::codec_workspace }) {
                        for (const auto measure : {
                                llama_cache_acct_measure::logical_payload,
                                llama_cache_acct_measure::resident_allocated,
                                llama_cache_acct_measure::reserved }) {
                            cache_authority->ledger.gauge_set(
                                category, binding.domain, measure, 0);
                        }
                    }
                }
            }
            (void) cache_authority->ledger.certify_complete(
                host_domain,
                llama_cache_acct_producer::retention_sidecar);
            // the parked rolling-window tape's zero is OBSERVED, not implied: an explicit
            // measured-zero gauge, distinct from every unknown cell
            cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::rolling_window_tape,
                    observer_domain,
                    llama_cache_acct_measure::logical_payload, 0);
            (void) cache_authority->ledger.certify_complete(
                observer_domain, llama_cache_acct_producer::observer_init);

            // Observe real resident allocations before certifying live_memory. A failed
            // attribution leaves every affected domain unavailable; a measured zero is
            // valid (e.g. no recurrent layers on an attention-only target).
            if (!cache_plan_observe_live_memory(true)) {
                cache_authority_config_failed(true);
            }
            if (!cache_plan_configure_budget()) {
                cache_authority_config_failed(true);
            }
            if (capture_manifest_enabled) {
                std::vector<llama_cache_acct_resource_domain>
                    capture_accounting_domains;
                capture_accounting_domains.reserve(
                    2 + cache_authority->live_device_domains.size());
                capture_accounting_domains.push_back(host_domain);
                capture_accounting_domains.push_back(pinned_domain);
                for (const auto & binding :
                        cache_authority->live_device_domains) {
                    capture_accounting_domains.push_back(
                        binding.domain);
                }
                if (!server_vbr_artifact_store_verify_accounting(
                        cache_authority->ledger,
                        capture_accounting_domains)) {
                    cache_authority_config_failed(true);
                }
            }

            // Establish an initial coherent budget object so lifecycle-only is a fully configured
            // authority, not merely a set of producer pointers. Each admission re-samples physical
            // capacity before composing, because free memory is time-sensitive.
            if (!cache_authority->sample_budget(cache_authority->budget_config) ||
                !cache_authority->budget.reset(
                    cache_authority->ledger.snapshot(),
                    cache_authority->budget_config)) {
                cache_authority_config_failed(false);
            }

            if (capture_manifest_enabled &&
                cache_authority->configured) {
                server_vbr_artifact_store_config config;
                config.ledger = &cache_authority->ledger;
                config.pinned_domain = pinned_domain;
                config.topologies = capture_topologies;
                config.pool_bindings = capture_pool_bindings;
                config.lanes = capture_lanes;
                config.attention_children =
                    capture_attention_children;
                static constexpr uint64_t RING_FLOOR =
                    64ull*1024*1024;
                static constexpr size_t RING_CHUNK =
                    8ull*1024*1024;
                const uint64_t lane_floor =
                    uint64_t(config.lanes.size())*2*RING_CHUNK;
                config.ring_bytes = std::min<uint64_t>(
                    VBR_CAPTURE_PINNED_RING_MAX_BYTES,
                    std::max<uint64_t>(RING_FLOOR, lane_floor));
                config.chunk_bytes = RING_CHUNK;
                config.budget_context = cache_authority.get();
                config.sample_budget = [](
                        void * context,
                        llama_cache_budget_config & output) noexcept {
                    auto * authority =
                        static_cast<server_cache_authority *>(context);
                    return authority->sample_budget(output);
                };
                server_vbr_artifact_capture_status status;
                server_vbr_artifact_store_create_diagnostics diagnostics;
                vbr_artifact_store =
                    server_vbr_artifact_store::create(
                        config, status, &diagnostics);
                vbr_prompt_cache_store_status = status;
                vbr_prompt_cache_store_diagnostics = diagnostics;
                if (!vbr_artifact_store) {
                    SRV_WRN(
                        "VBR_ARTIFACT_CAPTURE store unavailable status=%s "
                        "failure=%s ring_status=%s ring_failure=%s "
                        "attention_children=%u lanes=%zu "
                        "requested_ring=%" PRIu64
                        " attempted_ring=%" PRIu64
                        " chunk=%zu\n",
                        server_vbr_artifact_capture_status_name(status),
                        server_vbr_artifact_store_create_failure_name(
                            diagnostics.failure),
                        vbr_capture_stream_status_name(
                            diagnostics.ring_status),
                        vbr_capture_ring_create_failure_name(
                            diagnostics.ring_failure),
                        diagnostics.attention_children,
                        diagnostics.lane_count,
                        diagnostics.requested_ring_bytes,
                        diagnostics.attempted_ring_bytes,
                        diagnostics.chunk_bytes);
                } else {
                    vbr_capture_readiness_estimator = std::make_unique<
                        server_vbr_capture_readiness_estimator>();
                    vbr_prompt_cache_support =
                        server_vbr_prompt_cache_support_status::supported;
                    SRV_INF(
                        "VBR_ARTIFACT_CAPTURE store ready "
                        "attention_children=%u lanes=%zu "
                        "requested_ring=%" PRIu64
                        " constructed_ring=%" PRIu64
                        " chunk=%zu\n",
                        diagnostics.attention_children,
                        diagnostics.lane_count,
                        diagnostics.requested_ring_bytes,
                        diagnostics.constructed_ring_bytes,
                        diagnostics.chunk_bytes);
                }
            }

            if (params_base.vbr_prompt_cache && !vbr_artifact_store) {
                // Classify failures that occur before the store constructor as
                // explicitly as constructor-owned failures. Automatic
                // activation continues live-only; an explicit request remains
                // a strict startup contract.
                if (!capture_manifest_enabled) {
                    vbr_prompt_cache_support =
                        server_vbr_prompt_cache_support_status::
                            artifact_topology_unavailable;
                    SRV_WRN(
                        "VBR_ARTIFACT_CAPTURE topology unavailable "
                        "reason=%s devices=%zu resolved_split=%zu "
                        "topologies=%zu runtime_pools=%zu bindings=%zu "
                        "lanes=%zu attention_children=%u\n",
                        capture_topology_failure,
                        gpu_identities.size(),
                        capture_resolved_split_count,
                        capture_topologies.size(),
                        capture_runtime_pools.size(),
                        capture_pool_bindings.size(),
                        capture_lanes.size(),
                        capture_attention_children);
                } else if (!cache_authority->configured) {
                    vbr_prompt_cache_support =
                        server_vbr_prompt_cache_support_status::
                            accounting_unavailable;
                } else {
                    vbr_prompt_cache_support =
                        server_vbr_prompt_cache_support_status::
                            artifact_store_unavailable;
                }
                const auto fallback =
                    server_vbr_prompt_cache_fallback_action_for(
                        vbr_prompt_cache_automatic,
                        vbr_prompt_cache_support);
                if (fallback ==
                        server_vbr_prompt_cache_fallback_action::live_only) {
                    params_base.vbr_prompt_cache = false;
                    params_base.cache_idle_slots = false;
                    params_base.cache_ram_mib = 0;
                    prompt_cache.reset();
                    SRV_WRN(
                        "automatic dynamic VBR host caching "
                        "fallback=live_only reason=%s store_status=%s "
                        "store_failure=%s; cache-ram disabled\n",
                        server_vbr_prompt_cache_support_status_name(
                            vbr_prompt_cache_support),
                        server_vbr_artifact_capture_status_name(
                            vbr_prompt_cache_store_status),
                        server_vbr_artifact_store_create_failure_name(
                            vbr_prompt_cache_store_diagnostics.failure));
                } else {
                    SRV_ERR(
                        "--vbr-prompt-cache is unavailable reason=%s "
                        "store_status=%s store_failure=%s\n",
                        server_vbr_prompt_cache_support_status_name(
                            vbr_prompt_cache_support),
                        server_vbr_artifact_capture_status_name(
                            vbr_prompt_cache_store_status),
                        server_vbr_artifact_store_create_failure_name(
                            vbr_prompt_cache_store_diagnostics.failure));
                    return false;
                }
            }

            if (prompt_cache) {
                prompt_cache->acct = &cache_authority->ledger;
                if (params_base.cache_lifecycle) {
                    prompt_cache->publish_authority =
                        cache_authority.get();
                }
            }
        }

        if (cache_plan_obs) {
            SRV_INF("%s", "cache-debug enabled: observed cache decisions per request (JSON log + /slots.cache_plan)\n");
        }

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {

// fork 9l-d2b: selection-call counter for the rejected-launch regression (fork server-context.cpp 9304-9306)
        if (prompt_admission_geometry_for_test) {
            prompt_selection_calls_for_test++;
        }
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache && fixed_host_cache_enabled();

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

// fork 9l-d2b: prompt-token preparation gate + slot admission (fork server-context.cpp 9522-9527, 9648-9695)
    void prepare_prompt_tokens_for_launch(
            const server_slot & slot,
            server_task & task) {
        if (task.prompt_tokens_prepared) {
            return;
        }
        // The fork continues with a diff-self-spec prompt rewrite (fork 9529-9645:
        // <think> block stripping + looped-content truncation).  That belongs to the
        // separate diff-self-spec feature: the port declares server_slot::diff_self_spec
        // but never sets it true, and the rewrite needs two members the port does not
        // have (diff_think_open_id / diff_think_close_id).  Landing it here would add
        // dead code plus unported members, so only the gate is ported.
        (void) slot;
        task.prompt_tokens_prepared = true;
    }

    bool admit_task_for_slot(
            const server_slot & slot,
            server_task & task) {
        const int32_t n_ubatch = prompt_admission_geometry_for_test
            ? prompt_admission_geometry_for_test->n_ubatch
            : llama_n_ubatch(ctx_tgt);
        const bool can_split = prompt_admission_geometry_for_test
            ? prompt_admission_geometry_for_test->can_split
            : (!task.need_embd() ||
               (llama_get_memory(ctx_tgt) &&
                llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST));
        switch (server_slot_prompt_admission_check(
                can_split, task.tokens.size(), n_ubatch, slot.n_ctx)) {
            case server_slot_prompt_admission::batch_too_large:
                send_error(
                    task,
                    string_format(
                        "input (%d tokens) is too large to process. increase the physical batch "
                        "size (current batch size: %d)",
                        task.n_tokens(), n_ubatch),
                    ERROR_TYPE_SERVER);
                return false;
            case server_slot_prompt_admission::context_too_large:
                if (!can_split) {
                    send_error(
                        task.id,
                        string_format(
                            "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                            task.n_tokens(), slot.n_ctx),
                        ERROR_TYPE_EXCEED_CONTEXT_SIZE,
                        task.n_tokens(), slot.n_ctx);
                } else {
                    send_error(
                        task.id,
                        string_format(
                            "request (%d tokens) exceeds the available context size (%d tokens), try increasing it",
                            task.n_tokens(), slot.n_ctx),
                        ERROR_TYPE_EXCEED_CONTEXT_SIZE,
                        task.n_tokens(), slot.n_ctx);
                }
                return false;
            case server_slot_prompt_admission::accepted:
                return true;
        }
        GGML_ABORT("invalid slot prompt admission");
    }

// fork 9l-d2d: the automatic VBR restore body (fork server-context.cpp 9698-10270)
    bool try_automatic_vbr_restore(
            server_slot & slot,
            const server_task & task,
            const common_cache_family_binding & incoming_family,
            size_t * restored_prefix = nullptr) noexcept {
        if (restored_prefix) {
            *restored_prefix = SIZE_MAX;
        }
        try {
            if (!params_base.vbr_prompt_cache || !prompt_cache ||
                !vbr_artifact_store ||
                task.type != SERVER_TASK_TYPE_COMPLETION || task.is_child() ||
                task.is_parent() || task.tokens.empty() ||
                !task.params.cache_prompt ||
                automatic_vbr_cache_support(
                    task.tokens, slot.lora, slot.can_speculate()) !=
                    server_vbr_prompt_cache_support_status::supported ||
                slot.state != SLOT_STATE_IDLE || slot.is_processing()) {
                return false;
            }
            const bool construction_empty =
                slot.prompt.tokens.empty() &&
                slot.prompt.checkpoints.empty() &&
                slot.prompt.sequence_epoch == 0;
            const bool occupied_candidate =
                !slot.prompt.tokens.empty() &&
                !slot.prompt.tokens.has_media() &&
                slot.prompt.sequence_epoch != 0;
            if (!construction_empty && !occupied_candidate) {
                return false;
            }
            server_prompt_cache_vbr_restore_candidate candidate;
            const auto adapter_identity = lora_config_identity(slot.lora);
            if (!prompt_cache->prepare_vbr_restore(
                    task.tokens, frontier_execution_identity,
                    adapter_identity, candidate,
                    !ctx_dft && !slot.can_speculate())) {
                if (occupied_candidate) {
                    SLT_INF(slot, "%s",
                            "automatic occupied VBR restore refused: no exact host candidate\n");
                }
                return false;
            }
            const auto restore_event_for = [](
                    const server_prompt_cache_vbr_owner & payload) noexcept {
                bool draft_image = false;
                bool accelerator_image = false;
                if (payload) {
                    for (const auto & companion :
                            payload->package().companions()) {
                        draft_image |= companion.descriptor.kind ==
                            vbr_artifact_companion_kind::required_spec_payload;
                        accelerator_image |= companion.descriptor.kind ==
                            vbr_artifact_companion_kind::typed_accelerator;
                    }
                }
                if (!draft_image) {
                    return common_speculative_sequence_event::
                        target_restored_without_draft;
                }
                return accelerator_image
                    ? common_speculative_sequence_event::
                        composite_image_restored
                    : common_speculative_sequence_event::draft_image_restored;
            };
            const int32_t plan_source_id = candidate.source_id();
            const uint64_t plan_prefix_tokens = candidate.prefix_tokens();
            const auto observe_vbr_delivery = [&] (
                    uint64_t delivered_payload_bytes) noexcept {
                if (!slot.cache_plan || plan_source_id < 0) {
                    return;
                }
                auto & rec = *slot.cache_plan;
                auto * row = rec.find_or_add(
                    common_cache_plan_provider::host_cache_entry,
                    plan_source_id, COMMON_CACHE_PLAN_PHASE_HOST_SCAN,
                    slot.id, rec.selection);
                if (!row) {
                    return;
                }
                row->payload_kind =
                    common_cache_plan_payload_kind::vbr_artifact;
                row->lcp_tokens = llama_cache_acct_value::measured(
                    plan_prefix_tokens);
                row->payload_bytes = llama_cache_acct_value::measured(
                    delivered_payload_bytes);
                row->accept();
                row->delivered = true;
                rec.select(common_cache_plan_provider::host_cache_entry, row);
                rec.note_inventory_complete(
                    common_cache_plan_provider::host_cache_entry);
            };
            llama_memory_i * memory = nullptr;
            bool cleared_for_empty_handoff = false;
            const bool occupied_prefix_projection =
                occupied_candidate && candidate.requires_prefix_projection();
            vbr_artifact_attention_prefix_projection occupied_projection;
            if (occupied_candidate) {
                const auto occupied_projection_status =
                    occupied_prefix_projection
                    ? vbr_artifact_store->prepare_host_prefix_projection(
                        candidate.payload(),
                        task.tokens.retention_token_ids(),
                        candidate.prefix_tokens(), occupied_projection)
                    : vbr_artifact_prefix_projection_status::projected;
                if (occupied_projection_status !=
                        vbr_artifact_prefix_projection_status::projected) {
                    SLT_DBG(
                        slot,
                        "automatic occupied VBR prefix projection refused: status=%u\n",
                        unsigned(occupied_projection_status));
                    vbr_automatic_restore_occupied_fallbacks++;
                    return false;
                }
                memory = llama_get_memory(ctx_tgt);
                if (!memory) {
                    SLT_DBG(slot, "%s",
                            "automatic occupied VBR restore refused: no memory\n");
                    return false;
                }
                if (!cache_plan_observe_live_memory(false)) {
                    SLT_DBG(slot, "%s",
                            "automatic occupied VBR restore refused: live accounting unavailable\n");
                    return false;
                }
                if (memory->seq_pos_min(slot.id) < 0 ||
                    memory->seq_pos_max(slot.id) < 0) {
                    vbr_automatic_restore_occupied_fallbacks++;
                    SLT_DBG(slot, "%s",
                            "automatic occupied VBR restore refused: live sequence absent\n");
                    return false;
                }

                // Exact rollback keeps the existing atomic occupied path. A
                // one-slot server may instead hand off through an empty target
                // when the idle incumbent and incoming request both have
                // useful durable host prefixes. This is the same destructive
                // rebind the subsequent cold path would perform, moved before
                // import so failure naturally falls through to cold prefill.
                server_vbr_empty_handoff_gate handoff_gate;
                handoff_gate.slot_count = slots.size();
                handoff_gate.incoming_prefix = candidate.prefix_tokens();
                handoff_gate.hard_lease =
                    slot.hard_lease_blocks_live_prefix();
                handoff_gate.deferred_task =
                    queue_tasks.has_deferred_for_slot(slot.id);
                handoff_gate.incumbent_supported =
                    automatic_vbr_cache_support(
                        slot.prompt.tokens, slot.lora,
                        slot.can_speculate()) ==
                    server_vbr_prompt_cache_support_status::supported;
                server_prompt_cache_vbr_restore_candidate durable_incumbent;
                bool durable_incumbent_prepared = false;
                if (server_vbr_empty_handoff_lookup_allowed(handoff_gate)) {
                    handoff_gate.incumbent_lcp =
                        slot.prompt.tokens.get_common_prefix(task.tokens);
                    if (candidate.prefix_tokens() >
                            handoff_gate.incumbent_lcp) {
                        // The incumbent witness must itself be a sealed host
                        // frontier. A projection from some longer artifact is
                        // not recovery authority for destructive handoff.
                        durable_incumbent_prepared =
                            prompt_cache->prepare_vbr_restore(
                                slot.prompt.tokens,
                                frontier_execution_identity,
                                adapter_identity, durable_incumbent, false,
                                &slot.cache_family);
                        handoff_gate.durable_incumbent_prefix =
                            durable_incumbent.prefix_tokens();
                        handoff_gate.exact_incumbent_durable =
                            durable_incumbent_prepared &&
                            durable_incumbent.prefix_tokens() ==
                                uint64_t(slot.prompt.n_tokens());
                        handoff_gate.family_matches =
                            durable_incumbent.cache_family() ==
                                slot.cache_family;
                    }
                }
                const bool handoff_eligible = durable_incumbent_prepared &&
                    server_vbr_empty_handoff_allowed(handoff_gate);
                if (handoff_eligible &&
                    slot.prompt_clear_after_vbr_publication(
                        server_cache_destruction_reason::live_prefix_replace)) {
                    // Hybrid/iSWA memory may contain payload-complete children
                    // which a per-sequence erase cannot empty. With one slot,
                    // clearing the whole tree cannot affect another request.
                    llama_memory_clear(memory, false);
                    memory->breathe();
                    cleared_for_empty_handoff = true;
                    SLT_DBG(
                        slot,
                        "automatic VBR host handoff cleared idle source "
                        "prefix=%" PRIu64 " incoming=%" PRIu64 "\n",
                        durable_incumbent.prefix_tokens(),
                        candidate.prefix_tokens());
                }
            }
            if (occupied_candidate && !cleared_for_empty_handoff) {
                server_prompt_cache_vbr_replacement_ticket ticket;
                server_prompt_cache_vbr_replacement_diagnostics
                    replacement_diagnostics;
                if (!prompt_cache->prepare_vbr_occupied_replacement(
                        std::move(candidate), slot.prompt, slot.cache_family,
                        incoming_family, slot.id,
                        frontier_execution_identity, adapter_identity,
                        ticket, &replacement_diagnostics)) {
                    vbr_automatic_restore_occupied_fallbacks++;
                    SLT_DBG(
                        slot,
                        "automatic occupied VBR ticket refused: states=%llu "
                        "digest_matches=%llu raw_compares=%llu\n",
                        (unsigned long long) replacement_diagnostics.
                            recovery_states_visited,
                        (unsigned long long) replacement_diagnostics.
                            recovery_digest_matches,
                        (unsigned long long) replacement_diagnostics.
                            recovery_raw_token_comparisons);
                    return false;
                }

                struct occupied_publish_state {
                    server_prompt_cache * cache = nullptr;
                    server_prompt_cache_vbr_replacement_ticket * ticket = nullptr;
                    bool published = false;
                } state { prompt_cache.get(), &ticket, false };

                server_vbr_artifact_import_target request;
                request.memory = memory;
                request.destination = slot.id;
                request.execution_identity = frontier_execution_identity;
                request.adapter_config_identity = adapter_identity;
                request.previously_observed = true;
                request.draft_context = ctx_dft;
                request.accelerator = slot.get_spec();
                request.frontier_logits = &slot.frontier_logits.values;
                request.frontier_logits_count = uint32_t(
                    llama_vocab_n_tokens(vocab));
                request.publish_context = &state;
                request.prepare_publish = [](
                    void * opaque,
                    const std::vector<llama_token> & tokens,
                    uint64_t sequence_epoch) noexcept {
                    auto * current = static_cast<occupied_publish_state *>(opaque);
                    return current && current->cache && current->ticket &&
                        current->ticket->ready() &&
                        tokens.size() ==
                            current->ticket->incoming_prefix_tokens() &&
                        sequence_epoch == current->ticket->
                            replacement_prompt().sequence_epoch &&
                        current->cache->prepare_vbr_occupied_replacement_publish(
                            *current->ticket);
                };
                request.publish = [](void * opaque) noexcept {
                    auto * current = static_cast<occupied_publish_state *>(opaque);
                    GGML_ASSERT(current && current->cache && current->ticket);
                    current->cache->publish_vbr_occupied_replacement(
                        *current->ticket);
                    current->published = true;
                };

                vbr_automatic_restore_attempts++;
                vbr_automatic_restore_occupied_attempts++;
                const int64_t started = ggml_time_us();
                const uint64_t prefix_tokens = ticket.incoming_prefix_tokens();
                const uint64_t incumbent_lcp = ticket.incumbent_live_lcp();
                const auto restore_event =
                    restore_event_for(ticket.incoming_payload());
                const bool incoming_frontier_logits = std::any_of(
                    ticket.incoming_payload()->package().companions().begin(),
                    ticket.incoming_payload()->package().companions().end(),
                    [](const auto & companion) {
                        return companion.descriptor.kind ==
                            vbr_artifact_companion_kind::frontier_logits;
                    });
                if (restored_prefix) {
                    *restored_prefix = size_t(incumbent_lcp);
                }
                const auto imported = occupied_prefix_projection
                    ? vbr_artifact_store->
                        import_host_occupied_prefix_replacement(
                            std::move(request), ticket.incoming_payload(),
                            ticket.recovery_payload(),
                            std::move(occupied_projection))
                    : vbr_artifact_store->import_host_occupied_replacement(
                            std::move(request), ticket.incoming_payload(),
                            ticket.recovery_payload());
                if (imported.status !=
                        server_vbr_artifact_import_status::ok) {
                    GGML_ASSERT(!state.published);
                    vbr_automatic_restore_occupied_fallbacks++;
                    size_t failure_prefix = size_t(incumbent_lcp);
                    const bool quarantined =
                        server_vbr_apply_occupied_restore_failure_to_slot(
                            imported.adopt_status, imported.recovery,
                            size_t(incumbent_lcp), failure_prefix, slot);
                    if (restored_prefix) {
                        *restored_prefix = failure_prefix;
                    }
                    SLT_DBG(
                        slot,
                        "automatic occupied VBR restore refused: status=%s "
                        "guard=%s schedule=%s destination=%s validation=%s "
                        "stage=%s adopt=%s phase=%s "
                        "recovery=%s reset=%s\n",
                        server_vbr_artifact_import_status_name(imported.status),
                        vbr_occupied_replacement_guard_status_name(
                            imported.occupied_guard_status),
                        vbr_import_schedule_status_name(
                            imported.schedule_status),
                        vbr_import_destination_status_name(
                            imported.destination_status),
                        vbr_manifest_validation_status_name(
                            imported.validation_status),
                        vbr_adopt_stage_status_name(imported.stage_status),
                        vbr_adopt_status_name(imported.adopt_status),
                        vbr_adopt_phase_name(imported.phase),
                        vbr_adopt_recovery_outcome_name(imported.recovery),
                        quarantined ? "true" : "false");
                    return false;
                }
                GGML_ASSERT(state.published);
                prompt_cache->commit_vbr_occupied_replacement(
                    ticket, slot.prompt, slot.cache_family, slot.id);
                if (!occupied_prefix_projection &&
                    (imported.decision == vbr_import_decision::native_import ||
                     imported.decision == vbr_import_decision::live_rebased)) {
                    mark_exact_vbr_restore_durable(slot, *memory);
                }
                common_speculative_sequence_transition(
                    slot.get_spec(), slot.id, restore_event);
                if (incoming_frontier_logits) {
                    slot.bind_frontier_logits_to_prompt();
                } else {
                    slot.clear_frontier_logits();
                }
                SLT_DBG(slot,
                        "restored frontier logits: count=%zu authenticated=%d\n",
                        slot.frontier_logits.values.size(),
                        slot.frontier_logits_matches());
                vbr_automatic_restore_succeeded++;
                vbr_automatic_restore_occupied_succeeded++;
                const uint64_t delivered_payload_bytes =
                    imported.payload_bytes <=
                            UINT64_MAX - imported.companion_bytes
                        ? imported.payload_bytes + imported.companion_bytes
                        : 0;
                observe_vbr_delivery(delivered_payload_bytes);
                if (restored_prefix) {
                    *restored_prefix = size_t(prefix_tokens);
                }
                SLT_INF(
                    slot,
                    "automatic occupied VBR restore old_lcp=%" PRIu64
                    " prefix=%" PRIu64 " status=%s h2d_bytes=%" PRIu64
                    " h2d_chunks=%" PRIu64 " duration_ms=%.3f "
                    "occupied=%" PRIu64 "/%" PRIu64
                    " fallbacks=%" PRIu64 "\n",
                    incumbent_lcp, prefix_tokens,
                    server_vbr_artifact_import_status_name(imported.status),
                    imported.h2d_bytes, imported.h2d_chunks,
                    (ggml_time_us() - started)/1000.0,
                    vbr_automatic_restore_occupied_succeeded,
                    vbr_automatic_restore_occupied_attempts,
                    vbr_automatic_restore_occupied_fallbacks);
                return true;
            }

            vbr_artifact_attention_prefix_projection prefix_projection =
                cleared_for_empty_handoff
                    ? std::move(occupied_projection)
                    : vbr_artifact_attention_prefix_projection {};
            bool prefix_projection_ready =
                cleared_for_empty_handoff && candidate.requires_prefix_projection();
            const auto prepare_prefix_projection = [&](bool refresh = false) noexcept {
                if (!candidate.requires_prefix_projection()) {
                    return true;
                }
                if (prefix_projection_ready && !refresh) {
                    return true;
                }
                prefix_projection_ready =
                    vbr_artifact_store->prepare_host_prefix_projection(
                           candidate.payload(),
                           task.tokens.retention_token_ids(),
                           candidate.prefix_tokens(), prefix_projection) ==
                    vbr_artifact_prefix_projection_status::projected;
                return prefix_projection_ready;
            };
            if (!prepare_prefix_projection()) {
                return false;
            }
            memory = llama_get_memory(ctx_tgt);
            if (!memory) {
                return false;
            }
            // An explicitly erased construction-empty slot has already
            // abandoned its old prompt lineage, but the physical memory tree
            // is normally cleared later by the cold prompt path.  Artifact
            // validation runs before that path and must not see stale
            // recurrent/QSA companion state.  Clear this destination across
            // every child now; failure remains a soft cold-replay fallback.
            // Other unified-KV sequences are untouched.
            const bool empty_destination =
                construction_empty || cleared_for_empty_handoff;
            if (empty_destination &&
                !memory->seq_rm(slot.id, -1, -1)) {
                return false;
            }
            if (empty_destination) {
                // seq_rm drops membership immediately; dynamic VBR releases
                // its now-phantom pool watermark at the idle boundary.  Run
                // that boundary before snapshotting so the target is truly an
                // empty import destination rather than merely cell-empty.
                memory->breathe();
            }
            if (!cache_plan_observe_live_memory(false)) {
                return false;
            }
            const bool target_ready =
                memory && memory->seq_pos_min(slot.id) < 0 &&
                memory->seq_pos_max(slot.id) < 0;
            if (!target_ready ||
                !prompt_cache->prepare_vbr_restore_destination(
                    candidate, slot.prompt, slot.id)) {
                return false;
            }

            struct publish_state {
                server_prompt_cache * cache = nullptr;
                server_prompt_cache_vbr_restore_candidate * candidate = nullptr;
                bool published = false;
            } state { prompt_cache.get(), &candidate, false };

            const auto make_request = [&]() {
                server_vbr_artifact_import_target request;
                request.memory = memory;
                request.destination = slot.id;
                request.execution_identity = frontier_execution_identity;
                request.adapter_config_identity = adapter_identity;
                request.previously_observed = false;
                request.draft_context = ctx_dft;
                request.accelerator = slot.get_spec();
                request.frontier_logits = &slot.frontier_logits.values;
                request.frontier_logits_count = uint32_t(
                    llama_vocab_n_tokens(vocab));
                request.publish_context = &state;
                request.prepare_publish = [](
                    void * opaque,
                    const std::vector<llama_token> & tokens,
                    uint64_t sequence_epoch) noexcept {
                const auto * current = static_cast<const publish_state *>(opaque);
                if (!current || !current->candidate ||
                    !current->candidate->ready() ||
                    !current->candidate->payload()) {
                    return false;
                }
                const auto & package = current->candidate->payload()->package();
                return package && tokens.size() ==
                           current->candidate->prefix_tokens() &&
                       sequence_epoch ==
                           package.manifest().identity.sequence_epoch;
                };
                request.publish = [](void * opaque) noexcept {
                    auto * current = static_cast<publish_state *>(opaque);
                    GGML_ASSERT(
                        current && current->cache && current->candidate);
                    current->published =
                        current->cache->publish_vbr_restore(
                            *current->candidate);
                    GGML_ASSERT(current->published);
                };
                return request;
            };

            vbr_automatic_restore_attempts++;
            const int64_t started = ggml_time_us();
            auto import_selected = [&]() {
                return candidate.requires_prefix_projection()
                    ? vbr_artifact_store->import_host_prefix_payload(
                        make_request(), candidate.payload(),
                        std::move(prefix_projection))
                    : vbr_artifact_store->import_host_payload(
                        make_request(), candidate.payload());
            };
            auto imported = import_selected();
            bool used_compact_fallback = false;
            if (server_vbr_artifact_import_variant_fallback_safe(imported) &&
                candidate.use_fallback_payload()) {
                GGML_ASSERT(!state.published);
                if (!prepare_prefix_projection(true)) {
                    return false;
                }
                imported = import_selected();
                used_compact_fallback =
                    imported.status == server_vbr_artifact_import_status::ok;
            }
            if (imported.status != server_vbr_artifact_import_status::ok) {
                SLT_DBG(
                    slot,
                    "automatic VBR host restore refused: status=%s "
                    "validation=%s schedule=%s decision=%s adopt=%s "
                    "phase=%s units=%u companions=%u rollback=%llu\n",
                    server_vbr_artifact_import_status_name(imported.status),
                    vbr_manifest_validation_status_name(
                        imported.validation_status),
                    vbr_import_schedule_status_name(imported.schedule_status),
                    vbr_import_decision_name(imported.decision),
                    vbr_adopt_status_name(imported.adopt_status),
                    vbr_adopt_phase_name(imported.phase), imported.units,
                    imported.companions,
                    static_cast<unsigned long long>(imported.rollback_count));
                return false;
            }
            GGML_ASSERT(state.published);
            const uint64_t prefix_tokens = candidate.prefix_tokens();
            const int32_t source_id = candidate.source_id();
            const bool exact_reusable_restore =
                !candidate.requires_prefix_projection() &&
                (imported.decision == vbr_import_decision::native_import ||
                 imported.decision == vbr_import_decision::live_rebased);
            const auto restore_event =
                restore_event_for(candidate.payload());
            if (candidate.requires_prefix_projection()) {
                slot.cache_family = incoming_family;
            }
            const bool committed = prompt_cache->commit_vbr_restore(
                candidate, slot.prompt, slot.cache_family, slot.id);
            GGML_ASSERT(committed);
            if (!committed) {
                return false;
            }
            if (exact_reusable_restore) {
                mark_exact_vbr_restore_durable(slot, *memory);
            }
            common_speculative_sequence_transition(
                slot.get_spec(), slot.id, restore_event);
            slot.bind_frontier_logits_to_prompt();
            SLT_DBG(slot,
                    "restored frontier logits: count=%zu authenticated=%d\n",
                    slot.frontier_logits.values.size(),
                    slot.frontier_logits_matches());
            if (used_compact_fallback) {
                vbr_automatic_restore_quality_fallbacks++;
            }
            vbr_automatic_restore_succeeded++;
            const uint64_t delivered_payload_bytes =
                imported.payload_bytes <= UINT64_MAX - imported.companion_bytes
                    ? imported.payload_bytes + imported.companion_bytes
                    : 0;
            observe_vbr_delivery(delivered_payload_bytes);
            if (restored_prefix) {
                *restored_prefix = size_t(prefix_tokens);
            }
            SLT_INF(
                slot,
                "automatic VBR host restore source=%d prefix=%" PRIu64
                " status=%s compact_fallback=%d duration_ms=%.3f "
                "total=%" PRIu64 "/%" PRIu64 " quality_fallbacks=%" PRIu64
                "\n",
                source_id, prefix_tokens,
                server_vbr_artifact_import_status_name(imported.status),
                int(used_compact_fallback),
                (ggml_time_us() - started)/1000.0,
                vbr_automatic_restore_succeeded,
                vbr_automatic_restore_attempts,
                vbr_automatic_restore_quality_fallbacks);
            return true;
        } catch (...) {
            // Automatic reuse is a cache optimization. Allocation or identity
            // preparation failure must remain a soft cold-launch fallback.
            return false;
        }
    }

    bool launch_slot_with_task(
            server_slot & slot,
            server_task && task,
            bool reclaim_before_launch = false) {
        // Defensive backstop for any future caller which does not pass through
        // process_single_task's group-wide preselection admission.
        prepare_prompt_tokens_for_launch(slot, task);
        if (!admit_task_for_slot(slot, task)) {
            return false;
        }

        common_cache_family_binding incoming_family;
        if (server_cache_family_resolve_for_launch(
                cache_control_authority.get(),
                task.cache_family_binding_token,
                incoming_family) != server_cache_control_status::ok) {
            send_error(
                task, "cache family binding is unavailable",
                ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        // process per-request lora adapters (fall back to the server's base adapters when the
        // request carries none). Identity is checked for both branches: a slot that computed
        // its prompt state under one adapter set must not silently continue it under a different
        // one. The previous code checked equality only for the per-request branch and rebound the
        // no-lora branch unconditionally, so a request with no `lora` field could keep the KV/
        // recurrent state built under a per-request adapter. Also use prompt_clear() (not the
        // struct-only prompt.clear()) so the recurrent state is reset too on hybrid models — a
        // token-only clear leaves stale recurrent state that the reprocess would continue from.
        const auto task_loras = task.params.lora.empty()
            ? params_base.lora_adapters
            : construct_lora_list(task.params.lora);


        if (!are_lora_equal(task_loras, slot.lora)) {
            // called only after establishing inequality, as lora_should_clear_cache requires
            if (lora_should_clear_cache(slot.lora, task_loras)) {
                SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task_loras.size());
                slot.prompt_clear();
            } else {
                SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
            }
            slot.lora = task_loras;
            slot.vbr_adapter_config_identity_valid = false;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // VBR exact-frontier reuse requires the complete raw vocabulary
            // row. Backend sampling exposes only its selected/candidate tail.
            use_backend_sampling &= !params_base.vbr_prompt_cache;

            // fork 9l-d2c FORK-DIFF: the fork also clears backend sampling
            // for speculation here -- "use_backend_sampling &= !slot.can_speculate();".
            // `git log --all -S` finds no upstream revision that ever carried
            // that guard, so it is a fork speculative-decoding addition rather
            // than a VBR one, and the upstream launch path this port inherits
            // keeps its own behaviour instead.
            // diffusion self-spec needs raw logits for acceptance checking
            use_backend_sampling &= !slot.diff_self_spec;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // The per-request generation limit takes priority over the global one.
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        // Complete the last ordinary fallible launch allocation before an
        // automatic import can atomically publish live KV/prompt metadata.
        auto launched_task =
            std::make_unique<const server_task>(std::move(task));
        size_t restored_prefix = SIZE_MAX;
        const bool automatically_restored = try_automatic_vbr_restore(
            slot, *launched_task, incoming_family, &restored_prefix);
        slot.vbr_imported_complete_frontier = automatically_restored
            ? uint64_t(restored_prefix) : 0;

        const size_t retained_prefix = restored_prefix != SIZE_MAX
            ? restored_prefix
            : slot.prompt.tokens.get_common_prefix(launched_task->tokens);

        if (reclaim_before_launch &&
            server_vbr_dynamic_active(params_base) &&
            launched_task->n_tokens() > 0) {
            vbr_reclaim_before_degrade(
                slot.id,
                uint32_t(launched_task->n_tokens() - retained_prefix),
                "launch");
        }

        // The binding follows retained immutable content. Only a full-prefix
        // append/resume keeps the existing lineage; a branch/replacement
        // adopts the incoming declaration even when BOS or a shared system
        // prefix overlaps. Children carry the opaque token and later copy the
        // parent's resolved slot binding.
        // A live-prefix lease binds the execution/adapter/media identity while
        // its proven token boundary lives in the lease frontier. Preserve the
        // sidecar artifact only when the entire decoded ledger is an exact
        // prefix of the incoming request. That is an append of the same
        // physical live copy: inspect must report partially_stale until renew.
        // A trim, branch, or zero-overlap replacement retires the artifact and
        // keeps subject_lost as the fail-closed identity-change terminal.
        const bool append_continuity =
            !slot.prompt.tokens.empty() &&
            retained_prefix == slot.prompt.tokens.size();
        const auto live_retention_key =
            server_retention_instance_key::for_slot(slot.id);
        const bool prepared_host_restore = slot.retention_obs &&
            slot.retention_obs->prepared_for_launch(live_retention_key);
        const bool useful_reused_prefix =
            server_prompt_cache_retention_reuse_is_useful(
                retained_prefix, &launched_task->params.message_spans);
        const bool reused_live_prefix =
            !prepared_host_restore && useful_reused_prefix &&
            slot.retention_obs &&
            slot.retention_obs->acquire_lineage_ticket(
                live_retention_key, slot.retention_reuse_source);
        slot.retention_reuse_pending = reused_live_prefix;
        slot.retention_reuse_tokens = reused_live_prefix
            ? uint64_t(retained_prefix) : 0;
        slot.retention_branch_pending =
            reused_live_prefix && !append_continuity;
        slot.retention_request_succeeded = false;
        slot.retention_geometry_failed = false;
        if (slot.retention_obs && !append_continuity &&
            !prepared_host_restore) {
            slot.retention_obs->retire(
                live_retention_key);
        }
        slot.cache_family = common_cache_family_follow_lineage(
            slot.cache_family, incoming_family, retained_prefix,
            slot.prompt.tokens.size());
        slot.task = std::move(launched_task);

        if (prepared_host_restore) {
            server_retention_lineage_ticket restored_source;
            if (slot.retention_obs->consume_prepared_launch(
                    live_retention_key, restored_source)) {
                if (useful_reused_prefix) {
                    (void) slot.retention_obs->credit_reuse(restored_source);
                }
                slot.retention_obs->release_lineage_ticket(restored_source);
                server_retention_lineage_ticket restored_destination;
                const bool acquired_destination =
                    slot.retention_obs->acquire_lineage_ticket(
                        live_retention_key, restored_destination);
                const bool admitted_destination = acquired_destination &&
                    slot.retention_obs->activate_lineage_ticket(
                        restored_destination);
                if (acquired_destination) {
                    slot.retention_obs->release_lineage_ticket(
                        restored_destination);
                }
                if (!admitted_destination) {
                    // Observation must never compromise a launched request.
                    // Drop the unpublishable provisional shadow state and
                    // continue with the already-restored live KV.
                    slot.retention_obs->retire_slot(slot.id);
                }
            }
        }

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
            res->stats.kv_bpv =
                llama_memory_kv_bpv(llama_get_memory(slot.ctx_tgt));
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        // Inspection and launch do not count as reuse. Publish the pending
        // live-prefix credit only at the successful final-response boundary.
        // fork server-context.cpp:10910
        slot.commit_retention_reuse();
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->stats.kv_bpv    =
            llama_memory_kv_bpv(llama_get_memory(slot.ctx_tgt));
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        // cache receipt (§7.7): keyed chained block-hash over the slot's cached
        // token prefix + frontier identity. Untrusted hint, never authorization.
        // Portable tokenizer/template digests are §7.2 scope; the chain is
        // versioned by hash_spec + model identity here.
        if (params_base.cache_receipt) {
            const llama_tokens receipt_tokens = slot.prompt.tokens.get_text_tokens();
            std::string identity = lora_config_identity(slot.lora);
            if (identity == "0") {
                identity = "base:no-lora";
            }
            const bool keyed = !params_base.cache_receipt_key.empty();
            res->cache_receipt = json {
                {"version",          1},
                {"hash_spec",        keyed ? "sha256-chain-trunc64/v1"
                                           : "sha256-chain-trunc64-unkeyed-debug/v1"},
                {"block_tokens",     1024},
                {"model",            slot.task->params.oaicompat_model.empty()
                                         ? params_base.model.path
                                         : slot.task->params.oaicompat_model},
                {"sequence_epoch",   slot.prompt.sequence_epoch},
                {"token_count",      (int64_t) receipt_tokens.size()},
                {"adapter_identity", identity},
                {"chain",            cache_receipt_chain(
                                         receipt_tokens, 1024,
                                         params_base.cache_receipt_key)},
            };
        }

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max,
                           bool is_replay_boundary = false, llama_state_seq_flags tgt_extra_flags = 0) {
        const int id_task = slot.task->id;
        const int64_t checkpoint_n_tokens = slot.prompt.n_tokens() - n_tokens_cur;

        // A device-backed checkpoint is a transient view into the context's own storage for one
        // exact prefix; at any other prefix it would alias the wrong device state. Drop stale ones.
        if (tgt_extra_flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                if (it->data_tgt_on_device && it->n_tokens != checkpoint_n_tokens) {
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task, or they are a deliberate replay boundary
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            if (it->id_task != id_task && !it->is_replay_boundary && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            // Prefer evicting an ordinary periodic checkpoint: a replay boundary marks a prefix the
            // agent is expected to come back to, so dropping it would defeat the point of keeping it.
            auto victim = slot.prompt.checkpoints.begin();
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ++it) {
                if (!it->is_replay_boundary) {
                    victim = it;
                    break;
                }
            }

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", replay_boundary = %d, size = %.3f MiB)\n",
                    victim->pos_min, victim->pos_max, victim->n_tokens, victim->is_replay_boundary ? 1 : 0,
                    (float) victim->size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(victim);
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        bool superseded_replay_boundary = false;
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if (it->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", it->n_tokens);
                    superseded_replay_boundary = superseded_replay_boundary || it->is_replay_boundary;
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;
        cur.is_replay_boundary = is_replay_boundary || superseded_replay_boundary;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | tgt_extra_flags);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        {
            std::vector<uint8_t> spec_state;
            if (common_speculative_get_state(spec.get(), slot.id, spec_state) &&
                !spec_state.empty()) {
                cur.accel.spec.overwrite(
                    spec_state.size(),
                    [&](uint8_t * data, size_t size) {
                        std::memcpy(data, spec_state.data(), size);
                    });
            }
        }

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", replay_boundary = %d, size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, cur.is_replay_boundary ? 1 : 0, (float) cur.size() / 1024 / 1024);
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    // fork 9l-d2e: unified-KV cell admission + retention abort paths (fork server-context.cpp 14423-14494)
                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        slot->abandon_retention_prepared_launch();
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            slot->cache_plan.reset();
                            slot->abandon_retention_prepared_launch();
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            slot->cache_plan.reset();
                            slot->abandon_retention_prepared_launch();
                            break; // drop the task
                        }
                    } else {
                        // Unified KV: check if launching this task would overflow the shared cell pool.
                        // Use max(current, planned) since a just-launched slot hasn't filled yet.
                        if (params_base.kv_unified && task.n_tokens() > 0) {
                            int64_t cells_committed = 0;
                            for (const auto & s : slots) {
                                if (s.is_processing() && s.task) {
                                    cells_committed += std::max((int64_t) s.prompt.n_tokens(), (int64_t) s.task->n_tokens());
                                }
                            }
                            const int64_t cells_available = (int64_t) slot->n_ctx - cells_committed;
                            if (cells_available < (int64_t) task.n_tokens()) {
                                // never-fits: reject instead of deferring forever (silent hang)
                                if (task.n_tokens() > slot->n_ctx) {
                                    send_error(task.id,
                                               string_format(
                                                   "request (%d tokens) exceeds the total context size (%d tokens), try increasing it",
                                                   task.n_tokens(), slot->n_ctx),
                                                   ERROR_TYPE_EXCEED_CONTEXT_SIZE, task.n_tokens(), slot->n_ctx);
                                    slot->cache_plan.reset();
                                    slot->abandon_retention_prepared_launch();
                                    break; // drop the task
                                }
                                SRV_DBG("defer task %d: needs %d tokens but only %" PRId64 " cells available (%" PRId64 " committed by active slots)\n",
                                        id_task, task.n_tokens(), cells_available, cells_committed);
                                slot->cache_plan.reset();
                                slot->abandon_retention_prepared_launch();
                                queue_tasks.defer(std::move(task));
                                break;
                            }
                        }

                        if (!launch_slot_with_task(
                                *slot, std::move(task), true)) {
                            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                            slot->cache_plan.reset();
                            slot->abandon_retention_prepared_launch();
                            break; // drop the task
                        }
                    }

                    if (!params_base.vbr_prompt_cache &&
                        params_base.cache_idle_slots &&
                        !server_vbr_dynamic_active(params_base)) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            // fork 9l-d4b7c: the cache-task dispatch -- plan preflight and
            // the cache-control group (fork server-context.cpp 14717-14869).
            // The fork's other two cache cases (CACHE_CAPTURE 14870-15048
            // and CACHE_IMPORT 15049-15282) are not ported yet; when they
            // land they belong here, before SLOT_RESTORE, in fork order.
            case SERVER_TASK_TYPE_CACHE_PLAN_PREFLIGHT:
                {
                    auto res = std::make_unique<
                        server_task_result_cache_plan_preflight>();
                    res->id = task.id;
                    try {
                        res->view = cache_plan_preflight(task);
                    } catch (...) {
                        res->view.status =
                            server_cache_plan_preflight_status::internal_fault;
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_CACHE_HOLDER_CREATE:
            case SERVER_TASK_TYPE_CACHE_HOLDER_CLOSE:
            case SERVER_TASK_TYPE_CACHE_HOLDER_REATTACH:
            case SERVER_TASK_TYPE_CACHE_FAMILY_REGISTER:
            case SERVER_TASK_TYPE_CACHE_FAMILY_BIND:
            case SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE:
            case SERVER_TASK_TYPE_CACHE_LEASE_INSPECT:
            case SERVER_TASK_TYPE_CACHE_LEASE_RENEW:
            case SERVER_TASK_TYPE_CACHE_LEASE_RELEASE:
            case SERVER_TASK_TYPE_CACHE_CONTROL_EVENTS:
                {
                    auto res = std::make_unique<server_task_result_cache_control>();
                    res->id = task.id;
                    static_assert(
                        SERVER_TASK_TYPE_CACHE_CONTROL_EVENTS -
                                SERVER_TASK_TYPE_CACHE_HOLDER_CREATE + 1 ==
                            int(server_cache_control_operation::_count));
                    const auto operation = static_cast<
                        server_cache_control_operation>(
                            task.type - SERVER_TASK_TYPE_CACHE_HOLDER_CREATE);
                    res->operation = operation;
                    const auto precheck = server_cache_control_task_precheck(
                        task.cache_control != nullptr,
                        params_base.cache_lifecycle,
                        cache_authority != nullptr);
                    if (precheck != server_cache_control_status::ok) {
                        res->result.status = precheck;
                    } else {
                        try {
                            if (!cache_control_authority) {
                                server_cache_control_config config;
                                config.leases = &cache_authority->leases;
                                config.retention = &cache_authority->retention;
                                config.artifacts = vbr_artifact_store.get();
                                config.refresh_context = this;
                                config.refresh_subject = [](void * context,
                                    const server_cache_control_selector & selector,
                                    server_cache_lease_identity & identity,
                                    server_cache_lease_frontier & frontier) noexcept {
                                    return static_cast<server_context_impl *>(context)
                                        ->cache_control_refresh_subject(
                                            selector, identity, frontier);
                                };
                                config.host_proof_context = prompt_cache.get();
                                config.acquire_host_proof = [](void * context,
                                    const server_cache_control_selector & selector) noexcept {
                                    auto * cache = static_cast<server_prompt_cache *>(context);
                                    return cache
                                        ? server_prompt_cache_host_fallback_proof(
                                            *cache, selector)
                                        : server_cache_durable_fallback_proof{};
                                };
                                config.selector_evidence_context = prompt_cache.get();
                                config.selector_evidence = [](void *,
                                    const server_cache_control_selector & selector,
                                    uint64_t & bytes, bool & shared) noexcept {
                                    bytes = 0;
                                    shared = false;
                                    if (selector.kind !=
                                            server_cache_control_subject_kind::host_snapshot ||
                                        selector.retention_key.kind !=
                                            common_retention_artifact_kind::host_entry ||
                                        selector.retention_key.instance == 0) {
                                        return false;
                                    }
                                    const auto * state = reinterpret_cast<
                                        const server_prompt_cache_state *>(
                                            selector.retention_key.instance);
                                    bytes = state->size();
                                    shared = state->recovery_pins > 1;
                                    return true;
                                };
                                cache_control_authority = std::make_unique<
                                    server_cache_control_authority>(config);
                            }
                            server_cache_control_request request =
                                *task.cache_control;
                            server_cache_control_status selector_status =
                                server_cache_control_status::ok;
                            if (operation ==
                                    server_cache_control_operation::lease_acquire) {
                                if (selector_status ==
                                        server_cache_control_status::ok) {
                                    selector_status =
                                        cache_control_resolve_semantic_selector(
                                            request.subject,
                                            task.cache_control_subject);
                                }
                                if (selector_status ==
                                        server_cache_control_status::ok &&
                                    request.requested_class ==
                                        server_cache_lease_class::hard) {
                                    selector_status =
                                        cache_control_resolve_semantic_selector(
                                            request.fallback,
                                            task.cache_control_fallback);
                                    if (selector_status !=
                                            server_cache_control_status::ok &&
                                        request.allow_soft_fallback) {
                                        // Keep the unresolved exact selector:
                                        // the authority remains the sole policy
                                        // point for an explicitly authorized
                                        // hard->soft downgrade and reports the
                                        // effective class.
                                        selector_status =
                                            server_cache_control_status::ok;
                                    } else if (selector_status ==
                                            server_cache_control_status::not_found) {
                                        selector_status =
                                            server_cache_control_status::fallback_unavailable;
                                    }
                                }
                            } else if (operation ==
                                    server_cache_control_operation::lease_renew &&
                                request.fallback.kind <
                                    server_cache_control_subject_kind::_count) {
                                selector_status =
                                    cache_control_resolve_semantic_selector(
                                        request.fallback,
                                        task.cache_control_fallback);
                                if (selector_status ==
                                        server_cache_control_status::not_found) {
                                    selector_status =
                                        server_cache_control_status::fallback_unavailable;
                                }
                            }
                            if (selector_status !=
                                    server_cache_control_status::ok) {
                                res->result.status = selector_status;
                            } else {
                                res->result = cache_control_authority->execute(
                                    operation, request);
                            }
                        } catch (...) {
                            res->result.status =
                                server_cache_control_status::internal_fault;
                        }
                    }
                    queue_results.send(std::move(res));
                } break;
            // fork 9l-d4b7f: CACHE_CAPTURE (fork server-context.cpp 14870-15048).
            // The CACHE_IMPORT case (fork 15049-15282) landed right below as 9l-d4b7g.
            case SERVER_TASK_TYPE_CACHE_CAPTURE:
                {
                    auto send_capture = [&](server_vbr_artifact_capture_status status) {
                        auto res =
                            std::make_unique<server_task_result_cache_capture>();
                        res->id = task.id;
                        res->id_slot = task.cache_capture.id_slot;
                        res->status = status;
                        queue_results.send(std::move(res));
                    };

                    // The route remains present so gated configurations return
                    // a typed result, but no slot/model inspection occurs when
                    // the full lifecycle+armed construction gate did not pass.
                    if (!vbr_artifact_store) {
                        send_capture(
                            server_vbr_artifact_capture_status::unsupported);
                        break;
                    }
                    try {
                    assert_scheduler_thread(
                        vbr_capture_scheduler_thread,
                        "VBR artifact work must run on scheduler thread");

                    server_slot * slot =
                        get_slot_by_id(task.cache_capture.id_slot);
                    if (slot == nullptr) {
                        send_capture(
                            server_vbr_artifact_capture_status::invalid_slot);
                        break;
                    }
                    if (slot->is_processing()) {
                        SRV_DBG(
                            "requested capture slot is processing; defer task, id_task = %d\n",
                            task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    if (slot->state != SLOT_STATE_IDLE ||
                        slot->prompt.n_tokens() <= 0) {
                        send_capture(
                            server_vbr_artifact_capture_status::stale_frontier);
                        break;
                    }

                    vbr_explicit_capture_request request;
                    server_vbr_artifact_capture_status request_status;
                    if (!build_capture_request(
                            *slot, request, request_status)) {
                        send_capture(request_status);
                        break;
                    }

                    if (!cache_plan_observe_live_memory(false)) {
                        send_capture(
                            server_vbr_artifact_capture_status::unavailable);
                        break;
                    }
                    SRV_INF(
                        "VBR_ARTIFACT_CAPTURE begin task=%d slot=%d controllers=%u\n",
                        task.id, slot->id,
                        vbr_artifact_store->attention_children());
                    const int64_t started = ggml_time_us();
                    const auto captured = vbr_artifact_store->capture(
                        *llama_get_memory(ctx_tgt), std::move(request),
                        task.cache_capture.tenant_key);
                    const auto & capture_totals =
                        vbr_artifact_store->counters();
                    const double duration_ms =
                        (ggml_time_us() - started)/1000.0;
                    SRV_INF(
                        "VBR_ARTIFACT_CAPTURE end task=%d slot=%d status=%s "
                        "library_status=%s phase=%s inner_status=%s "
                        "generation_failure=%s size_failure=%s "
                        "reservation_group=%s prepare_status=%s "
                        "admission_status=%s failed_leaf=%zu "
                        "consistency=%s units=%u payload=%" PRIu64
                        " stash=%" PRIu64 " companion=%" PRIu64
                        " chunks=%" PRIu64 " backpressure=%" PRIu64
                        " events=%" PRIu64 " sync_fallbacks=%" PRIu64
                        " dedup=%d duration_ms=%.3f"
                        " total_requested=%" PRIu64
                        " total_published=%" PRIu64
                        " total_refused=%" PRIu64
                        " total_unavailable=%" PRIu64
                        " total_internal=%" PRIu64
                        " total_payload=%" PRIu64
                        " total_stash=%" PRIu64
                        " total_companion=%" PRIu64
                        " pinned_bytes=%" PRIu64
                        " total_chunks=%" PRIu64
                        " total_events=%" PRIu64
                        " total_sync_fallbacks=%" PRIu64
                        " total_backpressure=%" PRIu64
                        " total_dedup_hits=%" PRIu64
                        " total_dedup_misses=%" PRIu64
                        " staging_overlap_refusals=%" PRIu64 "\n",
                        task.id, slot->id,
                        server_vbr_artifact_capture_status_name(
                            captured.status),
                        vbr_explicit_capture_status_name(
                            captured.library_status),
                        vbr_explicit_capture_phase_name(
                            captured.phase),
                        vbr_capture_stream_status_name(
                            captured.inner_stream_status),
                        vbr_explicit_generation_failure_name(
                            captured.generation_failure),
                        vbr_explicit_size_failure_name(
                            captured.size_failure),
                        vbr_capture_reservation_group_name(
                            captured.begin_diagnostics.
                                reservation_group),
                        llama_cache_prepare_status_name(
                            captured.begin_diagnostics.
                                prepare_status),
                        llama_cache_admission_status_name(
                            captured.begin_diagnostics.
                                admission_status),
                        captured.begin_diagnostics.failed_leaf,
                        captured.status ==
                                server_vbr_artifact_capture_status::ok
                            ? "capture_exact" : "unavailable",
                        captured.units, captured.payload_bytes,
                        captured.stash_bytes,
                        captured.companion_bytes, captured.chunks,
                        captured.backpressure_waits,
                        captured.event_completions,
                        captured.synchronous_fallbacks,
                        captured.dedup, duration_ms,
                        capture_totals.requested,
                        capture_totals.exact_published,
                        capture_totals.refused,
                        capture_totals.unavailable,
                        capture_totals.internal_error,
                        capture_totals.payload_bytes,
                        capture_totals.stash_bytes,
                        capture_totals.companion_bytes,
                        capture_totals.pinned_bytes,
                        capture_totals.chunks,
                        capture_totals.event_completions,
                        capture_totals.synchronous_fallbacks,
                        capture_totals.backpressure_waits,
                        capture_totals.dedup_hits,
                        capture_totals.dedup_misses,
                        capture_totals.staging_overlap_refusals);

                    auto res =
                        std::make_unique<server_task_result_cache_capture>();
                    res->id = task.id;
                    res->id_slot = slot->id;
                    res->status = captured.status;
                    if (captured.status ==
                            server_vbr_artifact_capture_status::ok) {
                        res->consistency =
                            server_cache_capture_consistency::capture_exact;
                    }
                    res->reference = captured.reference;
                    res->controllers = captured.controllers;
                    res->units = captured.units;
                    res->companions = captured.companions;
                    res->payload_bytes = captured.payload_bytes;
                    res->stash_bytes = captured.stash_bytes;
                    res->companion_bytes = captured.companion_bytes;
                    res->chunks = captured.chunks;
                    res->backpressure_waits =
                        captured.backpressure_waits;
                    res->event_completions =
                        captured.event_completions;
                    res->synchronous_fallbacks =
                        captured.synchronous_fallbacks;
                    res->dedup = captured.dedup;
                    queue_results.send(std::move(res));
                    } catch (...) {
                        send_capture(
                            server_vbr_artifact_capture_status::
                                internal_error);
                    }
                } break;
            // fork 9l-d4b7g: CACHE_IMPORT (fork server-context.cpp 15049-15282).
            case SERVER_TASK_TYPE_CACHE_IMPORT:
                {
                    const auto send_import = [&task, this](
                            const server_vbr_artifact_import_output & imported) {
                        auto res =
                            std::make_unique<server_task_result_cache_import>();
                        res->id = task.id;
                        res->id_slot = task.cache_import.id_slot;
                        res->status = imported.status;
                        res->validation_status = imported.validation_status;
                        res->stage_status = imported.stage_status;
                        res->downward_reserve_status =
                            imported.downward_reserve_status;
                        res->adopt_status = imported.adopt_status;
                        res->recovery = imported.recovery;
                        res->adopt_attempted = imported.adopt_attempted;
                        res->phase = imported.phase;
                        res->downward_subphase = imported.downward_subphase;
                        res->downward_edge = imported.downward_edge;
                        res->schedule_status = imported.schedule_status;
                        res->destination_status = imported.destination_status;
                        res->destination_policy_steps =
                            imported.destination_policy_steps;
                        res->destination_logical_bytes =
                            imported.destination_logical_bytes;
                        res->destination_physical_growth_bytes =
                            imported.destination_physical_growth_bytes;
                        res->destination_max_deficit =
                            imported.destination_max_deficit;
                        res->decision = imported.decision;
                        res->consistency = server_cache_import_route_consistency(
                            imported.status, imported.consistency);
                        res->units = imported.units;
                        res->companions = imported.companions;
                        res->payload_bytes = imported.payload_bytes;
                        res->companion_bytes = imported.companion_bytes;
                        queue_results.send(std::move(res));
                    };
                    const auto terminal = [&](server_vbr_artifact_import_status status) {
                        server_vbr_artifact_import_output output;
                        output.status = status;
                        send_import(output);
                    };

                    if (!vbr_artifact_store) {
                        terminal(server_vbr_artifact_import_status::unsupported);
                        break;
                    }
                    try {
                        assert_scheduler_thread(
                            vbr_capture_scheduler_thread,
                            "VBR artifact work must run on scheduler thread");

                        server_slot * slot = get_slot_by_id(
                            task.cache_import.id_slot);
                        llama_memory_i * memory = llama_get_memory(ctx_tgt);
                        const bool slot_empty = slot && memory &&
                            slot->state == SLOT_STATE_IDLE &&
                            slot->prompt.n_tokens() == 0 &&
                            slot->prompt.checkpoints.empty() &&
                            slot->prompt.sequence_epoch == 0 &&
                            memory->seq_pos_min(slot->id) < 0 &&
                            memory->seq_pos_max(slot->id) < 0;
                        const auto precheck =
                            server_vbr_artifact_import_route_precheck(
                                vbr_artifact_store != nullptr, slot != nullptr,
                                slot && slot->is_processing(), memory != nullptr,
                                slot_empty);
                        if (precheck ==
                                server_vbr_artifact_import_status::slot_processing) {
                            SRV_DBG(
                                "requested import slot is processing; defer task, id_task = %d\n",
                                task.id);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (precheck != server_vbr_artifact_import_status::ok) {
                            terminal(precheck);
                            break;
                        }
                        if (!cache_plan_observe_live_memory(false)) {
                            terminal(server_vbr_artifact_import_status::unavailable);
                            break;
                        }

                        struct import_publish_state {
                            server_slot * slot = nullptr;
                            server_prompt prompt;
                            bool ready = false;
                        } publish_state { slot, {}, false };
                        server_vbr_artifact_import_request request;
                        request.memory = memory;
                        request.destination = slot->id;
                        request.reference = task.cache_import.reference;
                        request.tenant_key = task.cache_import.tenant_key;
                        request.execution_identity = frontier_execution_identity;
                        request.adapter_config_identity =
                            lora_config_identity(slot->lora);
                        // An erased construction-empty slot has no prior import
                        // state to preserve. Normal decode history was removed by
                        // the explicit erase and does not force a rebase.
                        // Imports erase into a fresh logical target. A persistent
                        // receipt/re-import seam will supply this
                        // bit once prior-import evidence has durable authority.
                        request.previously_observed = false;
                        request.publish_context = &publish_state;
                        request.prepare_publish = [](
                                void * opaque,
                                const std::vector<llama_token> & tokens,
                                uint64_t sequence_epoch) noexcept {
                            try {
                                auto * state =
                                    static_cast<import_publish_state *>(opaque);
                                if (!state || !state->slot || state->ready ||
                                    tokens.empty() || sequence_epoch == 0 ||
                                    state->slot->prompt.n_tokens() != 0 ||
                                    !state->slot->prompt.checkpoints.empty() ||
                                    state->slot->prompt.sequence_epoch != 0) {
                                    return false;
                                }
                                // Match SLOT_RESTORE's decode-side establishment:
                                // populate the existing slot prompt rather than
                                // replacing it with a default-constructed one.
                                // In particular, has_mtmd is slot configuration,
                                // not artifact payload. Prepare its value off-side
                                // so the no-fail composite publication remains
                                // allocation-free.
                                state->prompt.tokens.has_mtmd =
                                    state->slot->prompt.tokens.has_mtmd;
                                state->prompt.tokens.insert(tokens);
                                state->prompt.sequence_epoch = sequence_epoch;
                                state->ready =
                                    state->prompt.n_tokens() ==
                                        int(tokens.size());
                                return state->ready;
                            } catch (...) {
                                return false;
                            }
                        };
                        request.publish = [](void * opaque) noexcept {
                            auto * state =
                                static_cast<import_publish_state *>(opaque);
                            GGML_ASSERT(state && state->slot && state->ready);
                            using std::swap;
                            // SLOT_RESTORE installs its decoded token ledger into
                            // the existing prompt object. Do the same here: a
                            // whole-prompt swap would overwrite slot-owned decode
                            // configuration with server_prompt defaults. The
                            // target was proved empty before prepare_publish, so
                            // checkpoints remain empty and these two no-throw
                            // assignments are the complete publication surface.
                            swap(state->slot->prompt.tokens,
                                 state->prompt.tokens);
                            state->slot->prompt.sequence_epoch =
                                state->prompt.sequence_epoch;
                            // F-reference family provenance is not part of the
                            // v1 artifact metadata. A foreign import therefore
                            // starts undeclared rather than inheriting a stale
                            // binding from the destination slot.
                            state->slot->cache_family = {};
                            state->ready = false;
                        };

                        SRV_INF(
                            "VBR_ARTIFACT_IMPORT begin task=%d slot=%d\n",
                            task.id, slot->id);
                        const int64_t started = ggml_time_us();
                        const auto imported =
                            vbr_artifact_store->import(std::move(request));
                        if (imported.status ==
                                server_vbr_artifact_import_status::ok) {
                            common_speculative_sequence_transition(
                                slot->get_spec(), slot->id,
                                common_speculative_sequence_event::
                                    target_restored_without_draft);
                        }
                        const auto & totals = vbr_artifact_store->counters();
                        const double duration_ms =
                            (ggml_time_us() - started)/1000.0;
                        const std::string edge_name =
                            imported.downward_edge == UINT32_MAX
                                ? "none"
                                : std::to_string(imported.downward_edge);
                        const char * phase_name = imported.adopt_attempted
                            ? vbr_adopt_phase_name(imported.phase) : "none";
                        const char * subphase_name = imported.adopt_attempted
                            ? vbr_downward_adopt_subphase_name(
                                  imported.downward_subphase)
                            : "none";
                        SRV_INF(
                            "VBR_ARTIFACT_IMPORT end task=%d slot=%d status=%s "
                            "validation=%s stage=%s downward_reserve=%s "
                            "adopt=%s recovery=%s phase=%s subphase=%s edge=%s "
                            "schedule=%s decision=%s consistency=%s "
                            "units=%u companions=%u "
                            "payload=%" PRIu64 " companion_bytes=%" PRIu64
                            " duration_ms=%.3f total_requested=%" PRIu64
                            " total_succeeded=%" PRIu64
                            " total_report_only=%" PRIu64
                            " total_not_found=%" PRIu64
                            " total_refused=%" PRIu64
                            " total_unavailable=%" PRIu64 "\n",
                            task.id, slot->id,
                            server_vbr_artifact_import_status_name(imported.status),
                            vbr_manifest_validation_status_name(
                                imported.validation_status),
                            vbr_adopt_stage_status_name(imported.stage_status),
                            vbr_downward_reserve_status_name(
                                imported.downward_reserve_status),
                            vbr_adopt_status_name(imported.adopt_status),
                            vbr_adopt_recovery_outcome_name(imported.recovery),
                            phase_name, subphase_name,
                            edge_name.c_str(),
                            vbr_import_schedule_status_name(
                                imported.schedule_status),
                            vbr_import_decision_name(imported.decision),
                            server_cache_import_consistency_name(
                                server_cache_import_route_consistency(
                                    imported.status, imported.consistency)),
                            imported.units, imported.companions,
                            imported.payload_bytes,
                            imported.companion_bytes, duration_ms,
                            totals.imports_requested,
                            totals.imports_succeeded,
                            totals.imports_report_only,
                            totals.imports_not_found,
                            totals.imports_refused,
                            totals.imports_unavailable);
                        send_import(imported);
                    } catch (...) {
                        terminal(
                            server_vbr_artifact_import_status::internal_error);
                    }
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;
                    // fork server-context.cpp:16283-16288.  Only the declarations that the
                    // restore/trim transaction below reads are landable now: the region that
                    // INSTALLS a recurrent-checkpoint row (fork 16294-17266, the can_split
                    // rework) is not ported yet, so the three checkpoint_* flags keep their
                    // fork initial values and every attention-only arm below is
                    // constant-false -- which is exactly the fork's state on a restore that
                    // installed no recurrent row.  vbr_restore_freeze is likewise never begun.
                    std::unique_ptr<server_vbr_retier_freeze_scope> vbr_restore_freeze;
                    bool return_after_vbr_restore_trim = false;
                    bool checkpoint_tgt_recurrent_installed = false;
                    bool checkpoint_dft_recurrent_installed = false;
                    llama_pos checkpoint_installed_pos = -1;
                    const bool starting_prompt = slot.state == SLOT_STATE_STARTED;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // fork 9k-b2-i: n_past_common/n_past_keep declarations (fork 16315-16318)
                        // keep track how many tokens we can reuse from the previous state
                        int    n_past        = 0;
                        size_t n_past_common = 0;
                        size_t n_past_keep   = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                        // fork 9k-b2-i: n_cached_before/n_content_lcp (fork 16339-16351)
                            // Gate-5 edit-regime observation. Capture the logical
                            // branch point before checkpoint restore or VBR policy
                            // can rewind the physical frontier. The raw token LCP
                            // and actually reusable LCP are intentionally separate:
                            // adapter/alora constraints can add replay work without
                            // changing where the user's content diverged.
                            const size_t n_cached_before = slot.prompt.tokens.size();
                            const size_t n_content_lcp =
                                slot.prompt.tokens.get_common_prefix(input_tokens);

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = (int) n_content_lcp;

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                        // fork 9k-b2-i: n_past_common/n_past_keep assignment (fork 16359-16365)
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                // Preserve the true token LCP before checkpoint restore
                                // rewinds n_past to a smaller physical frontier. n_past_keep is the
                                // separately bounded amount that prompt.tokens may continue to
                                // claim after restore; see the restore-site recurrent guard below.
                                // Keeping token bookkeeping never mutates KV/state by itself.
                                n_past_common = n_past;
                                n_past_keep   = n_past_common;

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                        // fork 9k-b2-i: cache-reuse splice invalidation (fork 16407-16434)
                                            // The splice rewrites live cell positions (and later
                                            // re-RoPEs their K bytes) beneath any retained context
                                            // checkpoint. On non-hybrid (e.g. pure-SWA) models the
                                            // checkpoint selector has no epoch guard, so a stale
                                            // checkpoint over spliced cells would restore silently
                                            // wrong state. Invalidate before the first mutation.
                                            slot.observe_live_range_drop(
                                                server_cache_destruction_reason::live_prefix_replace,
                                                true);
                                            if (!slot.prompt.checkpoints.empty()) {
                                                SLT_WRN(slot, "cache-reuse splice invalidates %zu context checkpoint(s)\n",
                                                        slot.prompt.checkpoints.size());
                                                (void) slot.checkpoint_drop_joined_impl(
                                                    slot.prompt.checkpoints.begin(),
                                                    slot.prompt.checkpoints.end());
                                            }

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            ::server_cache_live_range_drop_impl(
                                                ctx_tgt, slot.id, head_p, head_c);
                                            common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);

                                            if (ctx_dft) {
                                                ::server_cache_live_range_drop_impl(
                                                    ctx_dft, slot.id, head_p, head_c);
                                                common_context_seq_add(ctx_dft, slot.id, head_c, head_c + n_match, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                        // fork 9k-b2-i: edit/divergence sample + speculative rewind (fork 16455-16484)
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            const size_t n_reusable_lcp =
                                slot.task->params.cache_prompt ? n_past_common : 0;
                            const size_t n_rewind =
                                n_cached_before > n_content_lcp
                                    ? n_cached_before - n_content_lcp
                                    : 0;
                            const size_t n_append =
                                input_tokens.size() > n_content_lcp
                                    ? input_tokens.size() - n_content_lcp
                                    : 0;
                            SLT_INF(
                                slot,
                                "edit/divergence sample "
                                "(cached/incoming/lcp/reusable/rewind/append/cache_prompt) = "
                                "(%zu/%zu/%zu/%zu/%zu/%zu/%d)\n",
                                n_cached_before, input_tokens.size(),
                                n_content_lcp, n_reusable_lcp,
                                n_rewind, n_append,
                                slot.task->params.cache_prompt ? 1 : 0);

                            if (n_rewind > 0) {
                                // A deferred speculative carry belongs to the discarded
                                // suffix, even when its numeric positions happen to fall
                                // below the retained LCP. Preserve the reusable drafter
                                // prefix, but invalidate branch-local proposal state at
                                // the scheduler's semantic rewind boundary.
                                common_speculative_sequence_transition(
                                    slot.get_spec(), slot.id,
                                    common_speculative_sequence_event::prompt_rewind);
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                        // fork 9k-b2-i: complete_vbr_import + fail-closed desync (fork 16494-16521)
                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                const bool complete_vbr_import =
                                    slot.vbr_imported_complete_frontier >=
                                        uint64_t(n_past);
                                slot.vbr_imported_complete_frontier = 0;
                                if (pos_min == -1) {
                                    // Fail-closed coordinated restore. The token ledger claims a
                                    // prefix (n_past > 0) but this sequence's KV is empty (pos_min == -1) --
                                    // a token-ledger/KV desync. Upstream GGML_ABORTs here, taking down the
                                    // whole server (every client) on a "should not happen" invariant. The
                                    // Transactional save and exact-size validation close the known host-cache route (empty-main select + non-
                                    // consuming both-or-nothing load); this converts any remaining/latent
                                    // desync into a per-slot recovery, matching the fork's recoverable-failure
                                    // rule (llama-kv-cache.cpp VBR try_map: "fail this batch recoverably
                                    // instead of a process abort killing every client"). Coordinate the two
                                    // structures -- clear KV (tgt+dft) + the stale ledger via prompt_clear --
                                    // and full-reprocess. Loud ERR so a real desync is never silently masked.
                                    SLT_ERR(slot, "token-ledger/KV desync (n_past = %d, pos_min = -1, tokens = %d, seq_id = %d) -- failing closed: clearing slot and reprocessing from scratch\n",
                                            n_past, (int) slot.prompt.tokens.size(), slot.id);
                                    slot.mandatory_recovery_reset(
                                        server_cache_destruction_reason::restore_failure);
                                    n_past            = 0;
                                    n_past_common     = 0;
                                    n_past_keep       = 0;
                                    pos_next          = 0;
                                    slot.cache_status = "full reprocess: token-ledger/KV desync recovered";
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                        // fork 9k-b2-i: !complete_vbr_import guard (fork 16566)
                                // fork 9k-b2-ii: checkpoint selection rewrite (fork 16566-17144)
                                if (!complete_vbr_import && pos_min >= pos_min_thold) {
                                    // Current attention-content lineage epoch(s). A recurrent-only
                                    // checkpoint remains valid across a lossless in-place retier, but not
                                    // across occupied-cell reuse, clear/reset, or state adoption. All-zero
                                    // when VBR is inactive, so this rejects nothing then.
                                    const auto vbr_now = llama_memory_vbr_state(llama_get_memory(ctx_tgt), slot.id, 0);
                                    const bool recurrent =
                                        llama_model_is_recurrent(model_tgt) ||
                                        llama_model_is_hybrid(model_tgt);

                                    // [obs] count checkpoints rejected specifically for changed attention
                                    // lineage, to explain a resulting cold reprocess in /slots
                                    int n_ckpt_rejected_vbr = 0;

                                    // B/observer candidate transport: which scan is authoritative
                                    // (legacy vs frontier) is decided AFTER both run, so each
                                    // lambda notes its visits into a fixed stack buffer
                                    // (classification + scalars the scan itself touched —
                                    // noexcept, no allocation); once the ratchet picks the
                                    // authoritative scan, ONLY that buffer becomes candidate
                                    // rows. The declared domain stays the shipped-visited set.
                                    struct ckpt_obs_visit {
                                        common_cache_plan_reason cls;
                                        int32_t  pos_max;
                                        uint64_t bytes;
                                    };
                                    struct ckpt_obs_buf {
                                        // deliberately NO default member initializers: the
                                        // disabled path pays a stack reservation and nothing
                                        // else; fields are initialized
                                        // only under the observer flag below
                                        std::array<ckpt_obs_visit, COMMON_CACHE_PLAN_MAX_CANDIDATES> v;
                                        uint32_t n;
                                        bool     overflow;
                                        // cls NONE = eligible (the short-circuiting selection)
                                        void note(common_cache_plan_reason cls, int32_t pos_max, uint64_t bytes) noexcept {
                                            if (n >= v.size()) { overflow = true; return; }
                                            v[n++] = { cls, pos_max, bytes };
                                        }
                                    };
                                    ckpt_obs_buf obs_legacy_buf, obs_frontier_buf;
                                    const bool ckpt_obs_on = slot.cache_plan != nullptr;
                                    if (ckpt_obs_on) {
                                        obs_legacy_buf.n   = 0; obs_legacy_buf.overflow   = false;
                                        obs_frontier_buf.n = 0; obs_frontier_buf.overflow = false;
                                    }

                                    // Legacy selection remains authoritative while the
                                    // computation-frontier path runs in shadow. The shipped
                                    // predicate is instantiated once; observer bookkeeping
                                    // compiles away from the unobserved instantiation.
                                    const auto make_legacy_pred = [&](auto obs_c) {
                                        return [&, obs_c](const auto & cur) {
                                            constexpr bool Obs = decltype(obs_c)::value;
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // A capture is durable only after its target payload is
                                            // complete. Old/foreign half-filled entries must never
                                            // reach the mutating restore path.
                                            [[maybe_unused]] uint64_t obs_bytes = 0;
                                            if constexpr (Obs) { obs_bytes = (uint64_t) cur.size(); }
                                            const bool checkpoint_lineage_matches =
                                                !recurrent ||
                                                common_prompt_checkpoint_lineage_matches(cur, vbr_now);
                                            const auto evaluation =
                                                server_cache_plan_evaluate_checkpoint(
                                                    !cur.empty(), true, recurrent,
                                                    checkpoint_lineage_matches,
                                                    cur.pos_min, cur.pos_max, pos_next,
                                                    pos_min_thold, obs_bytes);
                                            const bool ok = server_cache_plan_viable(
                                                evaluation.reason);
                                            if (!cur.empty() && !checkpoint_lineage_matches) {
                                                n_ckpt_rejected_vbr++;
                                            }
                                            if constexpr (Obs) {
                                                obs_legacy_buf.note(
                                                    ok ? COMMON_CACHE_PLAN_REASON_NONE
                                                       : evaluation.reason,
                                                    cur.pos_max, obs_bytes);
                                            }
                                            return ok;
                                        };
                                    };
                                    const auto it_legacy = ckpt_obs_on
                                        ? std::find_if(slot.prompt.checkpoints.rbegin(),
                                                       slot.prompt.checkpoints.rend(),
                                                       make_legacy_pred(std::true_type{}))
                                        : std::find_if(slot.prompt.checkpoints.rbegin(),
                                                       slot.prompt.checkpoints.rend(),
                                                       make_legacy_pred(std::false_type{}));

                                    // Shadow the same choice from the dual-written logical
                                    // frontier. pos_min remains the physical attention-coverage
                                    // guard until plan_resume lands; logical eligibility comes
                                    // from next_position rather than pos_max.
                                    std::string frontier_adapter_identity;
                                    bool frontier_eligible = false;
                                    // Per-selector lineage-rejection count: after the frontier
                                    // selector becomes authoritative, this scan's epoch
                                    // rejections are distinct from the legacy scan's.
                                    int n_ckpt_rejected_vbr_frontier = 0;
                                    auto it_frontier =
                                        slot.prompt.checkpoints.rend();
                                    bool frontier_shadow_available = true;
                                    try {
                                        frontier_adapter_identity =
                                            lora_config_identity(slot.lora);
                                        const auto make_frontier_pred = [&](auto obs_c) {
                                            return [&, obs_c](const auto & cur) {
                                                constexpr bool Obs = decltype(obs_c)::value;
                                                [[maybe_unused]] uint64_t obs_bytes = 0;
                                                if constexpr (Obs) { obs_bytes = (uint64_t) cur.size(); }
                                                if (slot.frontier_ratchet_flipped &&
                                                    server_fault(
                                                        "frontier_disagree_after_flip")) {
                                                    if constexpr (Obs) { obs_frontier_buf.note(COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID, cur.pos_max, obs_bytes); }
                                                    return false;
                                                }
                                                if (cur.empty() ||
                                                    !checkpoint_frontier_is_current(
                                                        slot, cur,
                                                        frontier_adapter_identity)) {
                                                    if constexpr (Obs) {
                                                        obs_frontier_buf.note(cur.empty()
                                                            ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY
                                                            : COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID, cur.pos_max, obs_bytes);
                                                    }
                                                    return false;
                                                }
                                                frontier_eligible = true;
                                                if (recurrent) {
                                                    if (!common_prompt_checkpoint_lineage_matches(
                                                            cur, vbr_now)) {
                                                        n_ckpt_rejected_vbr_frontier++;
                                                        if constexpr (Obs) { obs_frontier_buf.note(COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED, cur.pos_max, obs_bytes); }
                                                        return false;
                                                    }
                                                    const bool ok = cur.computation_frontier.next_position <=
                                                           pos_next;
                                                    if constexpr (Obs) { obs_frontier_buf.note(ok ? COMMON_CACHE_PLAN_REASON_NONE : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT, cur.pos_max, obs_bytes); }
                                                    return ok;
                                                }
                                                if (cur.computation_frontier.next_position - 1 >
                                                    pos_next) {
                                                    if constexpr (Obs) { obs_frontier_buf.note(COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT, cur.pos_max, obs_bytes); }
                                                    return false;
                                                }
                                                const bool ok = cur.pos_min < pos_min_thold ||
                                                       cur.pos_min == 0;
                                                if constexpr (Obs) { obs_frontier_buf.note(ok ? COMMON_CACHE_PLAN_REASON_NONE : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT, cur.pos_max, obs_bytes); }
                                                return ok;
                                            };
                                        };
                                        it_frontier = ckpt_obs_on
                                            ? std::find_if(slot.prompt.checkpoints.rbegin(),
                                                           slot.prompt.checkpoints.rend(),
                                                           make_frontier_pred(std::true_type{}))
                                            : std::find_if(slot.prompt.checkpoints.rbegin(),
                                                           slot.prompt.checkpoints.rend(),
                                                           make_frontier_pred(std::false_type{}));
                                    } catch (const std::exception & e) {
                                        // Shadow bookkeeping must never make the
                                        // authoritative legacy selector less available.
                                        frontier_shadow_available = false;
                                        SLT_WRN(slot,
                                                "FRONTIER_RATCHET "
                                                "event=shadow_unavailable "
                                                "action=keep_legacy error=%s\n",
                                                e.what());
                                    }

                                    const auto choice_index = [&](const auto & choice) {
                                        return choice ==
                                                slot.prompt.checkpoints.rend()
                                            ? int64_t(-1)
                                            : (int64_t) std::distance(
                                                slot.prompt.checkpoints.rbegin(),
                                                choice);
                                    };
                                    const int64_t legacy_choice =
                                        choice_index(it_legacy);
                                    const int64_t frontier_choice =
                                        choice_index(it_frontier);
                                    bool use_frontier = false;
                                    if (frontier_shadow_available) {
                                        use_frontier =
                                            slot.frontier_ratchet_selection(
                                                frontier_eligible,
                                                it_legacy == it_frontier,
                                                legacy_choice,
                                                frontier_choice);
                                    } else {
                                        slot.frontier_ratchet_disagreement(
                                            "checkpoint_selection_shadow_unavailable",
                                            legacy_choice, -2);
                                    }
                                    auto it = use_frontier ? it_frontier : it_legacy;

                                    bool do_reset = it == slot.prompt.checkpoints.rend();
                                    bool vbr_preflight_rejected = false;

                                    // B candidate transport [observer, noexcept]: the AUTHORITATIVE
                                    // scan's visit buffer becomes checkpoint candidate rows.
                                    // Reverse visit positions are translated to the stable
                                    // forward container ordinal before row identity joins.
                                    // The short-circuit
                                    // leaves later siblings outside the declared domain —
                                    // recorded as truncation, never extrapolated.
                                    if (slot.cache_plan) {
                                        auto & rec = *slot.cache_plan;
                                        const auto & buf = use_frontier ? obs_frontier_buf : obs_legacy_buf;
                                        int32_t checkpoint_host_source = -1;
                                        if (const auto * host = rec.selected_row(
                                                common_cache_plan_provider::host_cache_entry);
                                            host && host->delivered && host->source_id >= 0) {
                                            checkpoint_host_source = host->source_id;
                                        }
                                        for (uint32_t i = 0; i < buf.n; i++) {
                                            const int32_t checkpoint_source =
                                                server_cache_plan_checkpoint_source_id_from_reverse(
                                                    slot.prompt.checkpoints.size(), i,
                                                    checkpoint_host_source);
                                            if (checkpoint_source < 0) {
                                                rec.inventory_states[size_t(
                                                    common_cache_plan_provider::live_context_checkpoint)] =
                                                    common_cache_plan_inventory_state::overflowed;
                                                break;
                                            }
                                            auto * row = rec.find_or_add(
                                                common_cache_plan_provider::live_context_checkpoint,
                                                checkpoint_source,
                                                COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,
                                                slot.id, rec.selection);
                                            if (!row) {
                                                break; // record inventory overflowed (state latched)
                                            }
                                            row->lcp_tokens    = llama_cache_acct_value::measured(
                                                (uint64_t) std::max<int32_t>(buf.v[i].pos_max, 0));
                                            row->payload_bytes = llama_cache_acct_value::measured(buf.v[i].bytes);
                                            row->component_only = checkpoint_host_source >= 0;
                                            row->dependent_host_source_id = checkpoint_host_source;
                                            if (buf.v[i].cls == COMMON_CACHE_PLAN_REASON_NONE) {
                                                // the scan's first eligible entry = the shipped selection;
                                                // restore failures below demote it via note_reject
                                                row->accept();
                                                rec.select(
                                                    common_cache_plan_provider::live_context_checkpoint, row);
                                            } else {
                                                row->note_reject(buf.v[i].cls);
                                            }
                                        }
                                        if (buf.overflow) {
                                            rec.inventory_states[size_t(common_cache_plan_provider::live_context_checkpoint)] =
                                                common_cache_plan_inventory_state::overflowed;
                                        } else if (!do_reset && buf.n < slot.prompt.checkpoints.size()) {
                                            if (!rec.authority_inventory_complete) {
                                                rec.note_inventory_truncated(common_cache_plan_provider::live_context_checkpoint);
                                            }
                                        } else {
                                            rec.note_inventory_complete(common_cache_plan_provider::live_context_checkpoint);
                                        }
                                    }

                                    if (!do_reset) {
                                        // A live_rebased PARTIAL_ONLY restore keeps the
                                        // current attention representation. Prove the current-tier
                                        // footprint fits before either the restore or its later
                                        // suffix trim mutates state, then hold a scoped retier
                                        // freeze across both. This consumer performs no bounded
                                        // replay inside the scope, so n_tokens_extra is exactly 0.
                                        const auto preflight =
                                            llama_memory_vbr_retier_preflight(
                                                llama_get_memory(ctx_tgt), 0);
                                        if (preflight.active) {
                                            const bool preflight_fault =
                                                server_fault("vbr_retier_preflight_fail");
                                            const bool preflight_fits =
                                                preflight.fits && !preflight_fault;
                                            SLT_INF(slot,
                                                    "VBR_RETIER_PREFLIGHT owner=server_checkpoint_restore "
                                                    "result=%s fault=%u pools=%u watermark=%u needed=%" PRIu64
                                                    " available=%" PRIu64 " physical_growth=%" PRIu64
                                                    " physical_free=%" PRIu64 " max_deficit=%" PRId64 "\n",
                                                    preflight_fits ? "fits" : "reject",
                                                    preflight_fault ? 1u : 0u,
                                                    preflight.pools,
                                                    preflight.watermark_cells,
                                                    preflight.bytes_needed,
                                                    preflight.bytes_available,
                                                    preflight.physical_growth_needed,
                                                    preflight.physical_growth_available,
                                                    preflight.max_deficit);
                                            if (!preflight_fits) {
                                                SLT_WRN(slot, "%s",
                                                        "checkpoint restore rejected before mutation: "
                                                        "current VBR tiers cannot cover the frozen footprint\n");
                                                slot.cache_status =
                                                    "full reprocess: VBR frozen-footprint preflight rejected";
                                                vbr_preflight_rejected = true;
                                                do_reset = true;
                                            } else {
                                                vbr_restore_freeze =
                                                    std::make_unique<server_vbr_retier_freeze_scope>(
                                                        llama_get_memory(ctx_tgt),
                                                        "server_checkpoint_restore");
                                                if (!vbr_restore_freeze->active()) {
                                                    SLT_ERR(slot, "%s",
                                                            "VBR preflight was active but scoped freeze acquisition failed\n");
                                                    vbr_restore_freeze.reset();
                                                    do_reset = true;
                                                    // cache-plan observer: an attempted restore failed on
                                                    // infrastructure, not eligibility
                                                    if (slot.cache_plan) {
                                                        if (auto * sel_row = slot.cache_plan->selected_row(
                                                                common_cache_plan_provider::live_context_checkpoint)) {
                                                            sel_row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY);
                                                        }
                                                        slot.cache_plan->restore_attempt_failed = true;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        // note: keep the raw size-checked restore (instead of it->load_tgt which aborts on
                                        //       mismatch) so a failed restore falls back to a full re-process (do_reset)
                                        const size_t checkpoint_size = it->data_tgt.size();
                                        const size_t n = do_reset ? 0 :
                                            llama_state_seq_set_data_ext(
                                                ctx_tgt,
                                                it->data_tgt.view().data(),
                                                checkpoint_size, slot.id,
                                                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        if (!do_reset && n != checkpoint_size) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, (float) checkpoint_size / 1024 / 1024);
                                            do_reset = true;
                                            if (slot.cache_plan) {
                                                if (auto * sel_row = slot.cache_plan->selected_row(
                                                        common_cache_plan_provider::live_context_checkpoint)) {
                                                    sel_row->note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);
                                                }
                                                slot.cache_plan->restore_attempt_failed = true;
                                            }
                                        } else if (!do_reset && !it->try_load_dft(
                                                       ctx_dft, slot.id,
                                                       LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                                            SLT_ERR(slot,
                                                    "failed to restore draft checkpoint "
                                                    "(n_tokens = %" PRId64 ", size = %.3f MiB) -- "
                                                    "falling back to full prompt processing\n",
                                                    it->n_tokens,
                                                    (float) it->data_dft.size() / 1024 / 1024);
                                            do_reset = true;
                                            if (slot.cache_plan) {
                                                if (auto * sel_row = slot.cache_plan->selected_row(
                                                        common_cache_plan_provider::live_context_checkpoint)) {
                                                    sel_row->note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);
                                                }
                                                slot.cache_plan->restore_attempt_failed = true;
                                            }
                                        } else if (!do_reset) {
                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                            if (slot.retention_reuse_pending &&
                                                slot.retention_reuse_tokens != 0) {
                                                // Credit only the stateful
                                                // frontier actually restored,
                                                // not a longer raw token LCP
                                                // whose recurrent companion
                                                // was unavailable.
                                                slot.retention_reuse_tokens =
                                                    std::min<uint64_t>(
                                                        slot.retention_reuse_tokens,
                                                        uint64_t(n_past));
                                            }

                                            // Attention/SWA follows upstream #24891: the
                                            // checkpoint's logical token count bounds the true LCP
                                            // retained in bookkeeping. Recurrent/hybrid PARTIAL_ONLY
                                            // restore is different: it installs recurrent state only
                                            // at the checkpoint frontier, so even a matching later
                                            // token prefix is not backed by that state. Cap to n_past,
                                            // which was derived from BOTH restored pos_next and
                                            // checkpoint.n_tokens. Thus keep_first below can never
                                            // claim a token beyond the restored recurrent frontier;
                                            // the conservative replay changes work only, never output.
                                            if (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt)) {
                                                n_past_keep = std::min(n_past_keep, (size_t) n_past);
                                            } else {
                                                n_past_keep = std::min(n_past_keep, (size_t) it->n_tokens);
                                            }

                                            // Invalidate branch-local state before applying the
                                            // optional accelerator image. This ordering lets legacy
                                            // DFlash clear an old ring while preserving the newly
                                            // authenticated checkpoint ring installed below.
                                            const bool draft_image_restored =
                                                !it->data_dft.empty() &&
                                                it->data_dft_full_sequence;
                                            common_speculative_sequence_transition(
                                                slot.get_spec(), slot.id,
                                                draft_image_restored
                                                    ? (!it->accel.spec.empty()
                                                        ? common_speculative_sequence_event::composite_image_restored
                                                        : common_speculative_sequence_event::draft_image_restored)
                                                    : common_speculative_sequence_event::target_restored_without_draft);

                                            // A sequence transition invalidates deferred branch
                                            // state. Apply the matching checkpoint state only after
                                            // that invalidation; doing this in the opposite order
                                            // immediately discards a restored MTP carry.
                                            const bool spec_state_ok =
                                                it->accel.spec.empty() ||
                                                common_speculative_set_state(
                                                    slot.get_spec(), slot.id,
                                                    it->accel.spec.view());

                                            bool checkpoint_aux_ok = spec_state_ok;
                                            if (!spec_state_ok) {
                                                SLT_ERR(slot,
                                                    "failed to restore checkpoint speculative state "
                                                    "(n_tokens = %" PRId64 ", state size = %zu) -- failing closed\n",
                                                    it->n_tokens, it->accel.spec.size());
                                            }
                                            if (slot.can_speculate() && !it->accel.ring.empty() &&
                                                (!common_speculative_ring_state_matches_frontier(
                                                     slot.get_spec(),
                                                     it->accel.ring.view().data(),
                                                     it->accel.ring.size(),
                                                     llama_pos(it->n_tokens)-1) ||
                                                 !common_speculative_ring_state_load(
                                                     slot.get_spec(),
                                                     it->accel.ring.view().data(),
                                                     it->accel.ring.size()))) {
                                                SLT_ERR(slot,
                                                        "failed to restore checkpoint DFlash ring state "
                                                        "(n_tokens = %" PRId64 ", ring size = %zu) -- failing closed\n",
                                                        it->n_tokens, it->accel.ring.size());
                                                checkpoint_aux_ok = false;
                                            }

                                            if (!checkpoint_aux_ok) {
                                                do_reset = true;
                                                if (slot.cache_plan) {
                                                    if (auto * sel_row = slot.cache_plan->selected_row(
                                                            common_cache_plan_provider::live_context_checkpoint)) {
                                                        sel_row->note_reject(COMMON_CACHE_PLAN_REASON_ACCELERATOR_UNRESTORABLE);
                                                    }
                                                    slot.cache_plan->restore_attempt_failed = true;
                                                }
                                            } else {
                                                checkpoint_tgt_recurrent_installed =
                                                    llama_model_is_hybrid(model_tgt);
                                                checkpoint_dft_recurrent_installed =
                                                    ctx_dft && model_dft &&
                                                    llama_model_is_hybrid(model_dft) &&
                                                    !it->data_dft.empty();
                                                checkpoint_installed_pos = it->pos_max;

                                                // The checkpoint that successfully
                                                // restored this task is now part of this task's active
                                                // lineage. Adopt it here, unconditionally, rather than
                                                // relying on a later create-time dedup: ordinary
                                                // mid-prompt batches suppress checkpoint creation, and
                                                // the next min-step thinning pass could otherwise erase
                                                // the exact anchor we just restored.
                                                it->id_task = slot.task->id;

                                                SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) checkpoint_size / 1024 / 1024);
                                                slot.cache_status = "restored context checkpoint";
                                                // Transport the shipped
                                                // restore's own values into the observer row.
                                                // This site IS the delivery point — recorded
                                                // as data, never inferred.
                                                if (slot.cache_plan) {
                                                    if (auto * sel_row = slot.cache_plan->selected_row(
                                                            common_cache_plan_provider::live_context_checkpoint)) {
                                                        sel_row->accept();
                                                        sel_row->delivered = true;
                                                        // logical restored token count, not the physical
                                                        // pos_max position; bytes = every applied component
                                                        // across target, draft, and accelerators.
                                                        sel_row->lcp_tokens =
                                                            llama_cache_acct_value::measured((uint64_t) std::max(n_past, 0));
                                                        sel_row->payload_bytes =
                                                            llama_cache_acct_value::measured((uint64_t) it->size());
                                                        // rows the AUTHORITATIVE shipped scan
                                                        // actually visited (frontier after a
                                                        // frontier-selector flip, else legacy) + that same
                                                        // selector's epoch rejections
                                                        sel_row->siblings_scanned =
                                                            (use_frontier
                                                                ? obs_frontier_buf.n
                                                                : obs_legacy_buf.n);
                                                        sel_row->siblings_rejected_epoch = (uint32_t)
                                                            (use_frontier ? n_ckpt_rejected_vbr_frontier
                                                                          : n_ckpt_rejected_vbr);
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if (do_reset) {
                                        // The restore transaction is being abandoned. Clear
                                        // speculative branch/ring state at the same semantic
                                        // boundary; the p0=0 memory trim below performs the
                                        // corresponding target/draft physical reset.
                                        common_speculative_sequence_transition(
                                            slot.get_spec(), slot.id,
                                            common_speculative_sequence_event::full_clear);
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        // [obs] explain the cold reprocess: a checkpoint rejected for a
                                        // changed VBR representation is the common surprising cause
                                        slot.cache_status = vbr_preflight_rejected
                                            ? "full reprocess: VBR frozen-footprint preflight rejected"
                                            : n_ckpt_rejected_vbr > 0
                                                ? "full reprocess: checkpoint(s) rejected -- attention lineage changed"
                                                : "full reprocess: no reusable context checkpoint";
                                        // cache-plan observer stage-3: the same shipped classification, as a
                                        // closed reason on the checkpoint candidate row. A
                                        // concrete restore-failure reason recorded at its
                                        // failure site takes precedence — the generic
                                        // eligibility classification applies only to a row
                                        // with no reason yet.
                                        if (slot.cache_plan) {
                                            const int n_rejected_sel = use_frontier
                                                ? n_ckpt_rejected_vbr_frontier
                                                : n_ckpt_rejected_vbr;
                                            // the selected row carries a concrete restore-failure
                                            // reason from its failure site; when the scan selected
                                            // NOTHING, the provider-level classification rides one
                                            // aggregate row (source -1). Per-sibling verdicts are
                                            // already in the inventory from the scan transport.
                                            auto * row = slot.cache_plan->selected_row(
                                                common_cache_plan_provider::live_context_checkpoint);
                                            if (!row) {
                                                row = slot.cache_plan->find_or_add(
                                                    common_cache_plan_provider::live_context_checkpoint,
                                                    COMMON_CACHE_PLAN_SOURCE_AGGREGATE,
                                                    COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,
                                                    slot.id, slot.cache_plan->selection);
                                            }
                                            if (row) {
                                                row->siblings_scanned =
                                                    (use_frontier
                                                        ? obs_frontier_buf.n
                                                        : obs_legacy_buf.n);
                                                row->siblings_rejected_epoch = (uint32_t) n_rejected_sel;
                                                if (row->reason == COMMON_CACHE_PLAN_REASON_NONE) {
                                                    if (vbr_preflight_rejected) {
                                                        // restoring would demand a retier under the frozen
                                                        // footprint: a representation-tier fit failure
                                                        row->note_reject(COMMON_CACHE_PLAN_REASON_REPRESENTATION_TIER_UNSUPPORTED);
                                                    } else if (n_rejected_sel > 0) {
                                                        row->note_reject(COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED);
                                                    } else if (slot.prompt.checkpoints.empty()) {
                                                        row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE);
                                                    } else {
                                                        row->note_reject(COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT);
                                                    }
                                                }
                                            }
                                            // This reset discards everything already
                                            // installed for the request (pos_next/n_past go
                                            // to zero) — revoke historical deliveries; cold
                                            // fallback dominates them
                                            slot.cache_plan->revoke_deliveries();
                                        }
                                        pos_next = 0;
                                        n_past = 0;
                                        n_past_keep = 0;
                                    }
                                }
                            }

                            // fork 9k-b2-iii: lineage-guarded checkpoint invalidation (fork 17147-17185)
                            {
                                // pos_next may have been rewound by restore and is not an
                                // invalidation frontier. Use the real physical end of the incoming
                                // prompt, as upstream #24891 does, plus the logical true LCP as a
                                // lineage guard. A checkpoint after the true LCP may describe the old
                                // branch even when its position fits inside the new prompt; recurrent
                                // state makes that especially dangerous because it summarizes every
                                // preceding token. Checkpoint.n_tokens is the architecture-neutral
                                // capture frontier (unlike recurrent pos_min), so checkpoints through
                                // the true LCP survive even when restore rewound n_past below them.
                                // This changes only checkpoint metadata retention; checkpoint selection
                                // (strict pos_max < pos_next) and epoch rejection above are
                                // untouched. Every retained checkpoint is on the verified common
                                // lineage and within the new physical prompt; erasure can only cause
                                // replay, never install a different state.
                                const llama_pos prompt_pos_end = input_tokens.pos_next();
                                const llama_pos common_pos_end = slot.prompt.tokens.pos_next(n_past_common);

                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    const bool beyond_prompt = cur.pos_max > prompt_pos_end;
                                    const bool beyond_lcp    = cur.n_tokens > (int64_t) n_past_common;
                                    const bool invalidated   = beyond_prompt || beyond_lcp;

                                    if (invalidated) {
                                        SLT_TRC(slot,
                                                "erased invalidated context checkpoint "
                                                "(pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                                                ", n_swa = %d, prompt_pos_end = %d, common_pos_end = %d, size = %.3f MiB)\n",
                                                cur.pos_min, cur.pos_max, cur.n_tokens, n_swa,
                                                prompt_pos_end, common_pos_end, (float) cur.size() / 1024 / 1024);
                                        it = slot.checkpoint_drop(
                                            it, std::next(it),
                                            server_cache_destruction_reason::checkpoint_invalidated);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // fork 9k-b2-i: low-LCP reset + exact prompt action (fork 17188-17233)
                        // dynamic VBR: trade a token-trivial reusable prefix for the lossless
                        // full reset (n_past -> 0 makes the seq_rm below a full clear; the
                        // cache empties and the next prepare restores every entry tier)
                        server_cache_destruction_admission
                            vbr_low_lcp_admission;
                        {
                            const int n_past_before_vbr_reset = n_past;
                            n_past = vbr_reset_on_low_lcp(
                                slot, n_past, vbr_low_lcp_admission);
                            if (n_past < n_past_before_vbr_reset) {
                                // A VBR reset physically cleared state; do not let true-LCP token
                                // retention claim the cleared prefix.
                                n_past_keep = std::min(n_past_keep, (size_t) n_past);
                                if (n_past == 0) {
                                    // The full reset physically discarded whatever any
                                    // provider installed — a policy reset, not a failed
                                    // attempt, but delivery is revoked either way
                                    if (slot.cache_plan) {
                                        slot.cache_plan->revoke_deliveries();
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            const auto exact_prompt_action =
                                server_slot_exact_prompt_action_resolve(
                                    size_t(slot.task->n_tokens()),
                                    slot.task->need_sampling(),
                                    slot.diff_self_spec,
                                    slot.frontier_logits_matches());
                            if (exact_prompt_action.use_frontier_logits) {
                                slot.frontier_logits.pending = true;
                                SLT_INF(slot,
                                    "using authenticated frontier logits for exact prompt hit (n_past = %d)\n",
                                    n_past);
                            } else {
                                SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                                n_past = int(exact_prompt_action.reusable_tokens);
                                // The last token must really be re-evaluated for logits; preserving it
                                // only in token bookkeeping would skip that decode.
                                n_past_keep = std::min(n_past_keep, (size_t) n_past);
                                SLT_WRN(slot, "n_past was set to %d\n", n_past);
                            }
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        // fork 9k-b2-i: admission-aware ledger truncate (fork 17238-17254)
                        metrics.add_prompt_cached(n_past);

                        const size_t n_tokens_keep =
                            std::max(n_past_keep, (size_t) n_past);
                        if (vbr_low_lcp_admission.covers(
                                server_cache_destruction_class::live_range_drop,
                                server_cache_destruction_reason::low_lcp_reset) &&
                            vbr_low_lcp_admission.execution ==
                                server_cache_destruction_execution::pass_through) {
                            // The joined low-LCP manifest was admitted before its checkpoint
                            // invalidation; carry that same authority to this later phase.
                            slot.server_cache_token_ledger_truncate_impl(n_tokens_keep);
                        } else {
                            slot.observe_live_range_drop(
                                server_cache_destruction_reason::live_prefix_replace);
                            slot.server_cache_token_ledger_truncate_impl(n_tokens_keep);
                        }

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    // Checkpoint restores reset recurrent rollback depth. In particular,
                    // [TAG_PROMPT_LOGITS] can move p0 one token behind the restored
                    // frontier, making this bounded trim legitimately fallible.
                    // A successful PARTIAL_ONLY hybrid restore has already installed the
                    // recurrent row at checkpoint_installed_pos. When the suffix begins at
                    // exactly the next position, trimming that row only to reinstall the same
                    // checkpoint bytes is redundant. Keep the old whole-memory path for every
                    // other case, especially [TAG_PROMPT_LOGITS]'s one-behind trim: recurrent
                    // rollback depth is then a real validity check and must still fail closed.
                    const bool trim_tgt_attn_only =
                        checkpoint_tgt_recurrent_installed &&
                        (int64_t) p0 == (int64_t) checkpoint_installed_pos + 1;
                    const bool trim_dft_attn_only =
                        checkpoint_dft_recurrent_installed &&
                        (int64_t) p0 == (int64_t) checkpoint_installed_pos + 1;
                    if (trim_tgt_attn_only || trim_dft_attn_only) {
                        SLT_INF(slot,
                                "CHECKPOINT_ATTN_ONLY_TRIM p0=%d target=%u draft=%u\n",
                                p0,
                                trim_tgt_attn_only ? 1u : 0u,
                                trim_dft_attn_only ? 1u : 0u);
                    }
                    const auto checkpoint_lineage_before_trim =
                        trim_tgt_attn_only
                            ? llama_memory_vbr_state(
                                llama_get_memory(ctx_tgt), slot.id, 0)
                            : llama_memory_vbr_state_data{};
                    bool trim_ok = ::server_cache_live_range_drop_impl(
                        llama_get_memory(ctx_tgt), slot.id, p0, -1,
                        trim_tgt_attn_only);
                    if (trim_ok && ctx_dft) {
                        // FORK-DIFF (fork server-context.cpp:17324-17326): the fork holds
                        // ctx_dft in a llama_context_ptr; this tree keeps the raw pointer
                        // (port server-context.cpp:5131 already notes the same divergence).
                        // Restoring `ctx_dft.get()` on the call line reproduces fork:17325
                        // byte-for-byte.
                        trim_ok = ::server_cache_live_range_drop_impl(
                            llama_get_memory(ctx_dft), slot.id, p0, -1,
                            trim_dft_attn_only);
                    }

                    if (trim_ok && trim_tgt_attn_only) {
                        const auto checkpoint_lineage_after_trim =
                            llama_memory_vbr_state(
                                llama_get_memory(ctx_tgt), slot.id, 0);
                        if (server_cache_checkpoint_rebase_preserved_suffix(
                                slot.prompt.checkpoints,
                                checkpoint_lineage_before_trim,
                                checkpoint_lineage_after_trim,
                                p0) > 0) {
                            slot.checkpoint_ring_changed();
                        }
                    }

                    if (!trim_ok) {
                        SLT_WRN(slot, "memory_seq_rm [%d, end) rejected; falling back to full prompt re-processing\n", p0);

                        // Full removal is infallible for memory implementations and retains
                        // common_context_seq_rm's asserting contract. Clear both target and
                        // draft so a failure on either side cannot leave them out of sync.
                        slot.observe_mandatory_recovery_reset(
                            server_cache_destruction_reason::trim_rejection);
                        ::server_cache_mandatory_recovery_reset_impl(
                            ctx_tgt, slot.id, -1, -1);
                        if (ctx_dft) {
                            // FORK-DIFF (fork server-context.cpp:17352-17355): same raw-pointer
                            // divergence as above. Restoring `ctx_dft.get()` on the call line
                            // reproduces fork:17354 byte-for-byte.
                            ::server_cache_mandatory_recovery_reset_impl(
                                ctx_dft, slot.id, -1, -1);
                        }
                        common_speculative_sequence_transition(
                            slot.get_spec(), slot.id,
                            common_speculative_sequence_event::full_clear);

                        slot.server_cache_token_ledger_truncate_impl(0);
                        slot.stats.n_prompt_cached = 0;
                        slot.stats.n_prompt_processed = 0;

                        // The suffix trim of the restore transaction was rejected and
                        // everything installed was cleared — the delivered rows failed to
                        // realize their frontier. Record the precise reason on each, revoke
                        // delivery, and mark the attempt failed: cold fallback dominates.
                        if (slot.cache_plan) {
                            for (const auto prov : { common_cache_plan_provider::host_cache_entry,
                                                     common_cache_plan_provider::live_context_checkpoint }) {
                                if (auto * c = slot.cache_plan->selected_row(prov)) {
                                    if (c->delivered) {
                                        c->note_reject(COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID);
                                    }
                                }
                            }
                            slot.cache_plan->revoke_deliveries();
                            slot.cache_plan->restore_attempt_failed = true;
                        }
                    }

                    // Retiering resumes only after the coordinated restore+trim transaction.
                    // The outer exit arms a fresh controller pass at the next safe boundary.
                    vbr_restore_freeze.reset();
                    if (return_after_vbr_restore_trim) {
                        return;
                    }

                    if (starting_prompt && trim_ok && slot.prompt.tokens.empty()) {
                        restore_active_prefix(slot);
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // swap the MTP draft context out of VRAM so that the projector can run on
                    // the GPU for the image encode, then restore it once the media interval ends
                    const bool needs_mmproj_swap = mmproj_gpu_swap && !mmproj_is_on_gpu
                        && slot.prompt.n_tokens() < slot.task->n_tokens()
                        && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL;

                    // most prompts never enter the projector/draft exchange, so the guard that
                    // owns the restore is only constructed for an actual swap
                    std::optional<server_mmproj_restore_guard> mmproj_restore;
                    if (needs_mmproj_swap) {
                        mmproj_restore.emplace([&]() { swap_mmproj_to_spec(); });
                        mmproj_restore->arm();
                        if (!swap_spec_to_mmproj_gpu()) {
                            mmproj_restore->restore_now();
                            SLT_ERR(slot, "%s", "failed to load the multimodal projector during the GPU swap\n");
                            send_error(slot, "failed to load multimodal projector", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }
                    }

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    // the media interval is over: bring the draft context back
                    if (mmproj_restore.has_value()) {
                        mmproj_restore->restore_now();
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            // Lossless prefill reuse may use a larger physical ubatch while
                            // preserving the baseline GEMM tile. Keep semantic checkpoint
                            // boundaries tied to that baseline tile too; otherwise enabling
                            // reuse changes prompt segmentation and therefore MTP/checkpoint
                            // trajectories even when the CUDA math itself is bit-identical.
                            const int checkpoint_ubatch = params_base.prefill_reuse > 0
                                ? std::min(n_ubatch, params_base.prefill_reuse)
                                : n_ubatch;
                            const uint64_t n_task_tokens = (uint64_t) std::max<int32_t>(0, slot.task->n_tokens());
                            const uint64_t n_prompt_new = n_task_tokens > slot.stats.n_prompt_cached
                                ? n_task_tokens - slot.stats.n_prompt_cached
                                : 0;
                            // With a preceding rollback snapshot available, the 4+ubatch boundary
                            // checkpoint is redundant: the snapshot lands 1 token before the end of
                            // the prompt, which is finer than that boundary. Skipping it also frees
                            // a checkpoint slot for the snapshot itself.
                            const bool snapshot_prev = params_base.checkpoint_recurrent_prev &&
                                    llama_n_rs_seq(ctx_tgt) > 0 && spec == nullptr && n_prompt_new > 64;
                            const int checkpoint_offsets[] = {4 + checkpoint_ubatch, 4};
                            const int n_offsets = snapshot_prev ? 1 : 2;

                            bool should_break = false;
                            for (int oi = 0; oi < n_offsets; ++oi) {
                                const int offset = checkpoint_offsets[oi];
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    // fork 9l-d4b7i: the decode() hard-seal rewrite (fork server-context.cpp 18455-18693).
    //  Five fork features are fused in that body.  This port takes three:
    //    speculative single-yield terminal / VBR hard seal / VBR capture-readiness growth.
    //  Two are deliberately left out:
    //    - diffusion cycle accounting (t_verify_total / cycle_has_output): the whole
    //      cycle family is absent here and post_cycle is not in the VBR queue;
    //    - DFlash flush (common_speculative_flush_prefill): DFlash is not ported here.
    //  Upstream-only lines the fork dropped are restored: is_prompt / homogeneous_phase /
    //  decode_phase + llama_set_decode_phase, and the raw `model_dft` pointer.
    //  The recurrent copy_state_to bool is not taken: this tree uses the newer
    //  mem.seq_rm/mem.seq_cp implementation, which cannot fail.
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        // Ensure causal attention is on for prompt eval (diffusion draft toggles it off)
        llama_set_causal_attn(ctx_tgt, true);

        bool has_output = false;
        bool is_prompt = batch.tokens[off].is_prompt;
        bool homogeneous_phase = true;
        for (int i = off; i < off + batch_view.n_tokens; ++i) {
            has_output |= batch.tokens[i].output;
            homogeneous_phase &=
                batch.tokens[i].is_prompt == is_prompt;
        }

        const enum llama_decode_phase decode_phase =
            !homogeneous_phase ? LLAMA_DECODE_PHASE_AUTOMATIC :
            is_prompt ? LLAMA_DECODE_PHASE_PROMPT :
                        LLAMA_DECODE_PHASE_GENERATION;

        // Keep target verification and the dependent speculative update inside
        // one queue yield. Re-arming the worker between these two operations
        // adds a second mutex/wake/condition rendezvous to every speculative
        // ubatch even when no task is queued.
        int ret = 0;
        bool speculative_ok = true;
        std::exception_ptr speculative_exception;
        const int64_t t_verify_start = ggml_time_us();
        int64_t t_verify_elapsed = 0;
        const std::exception_ptr yield_exception =
            queue_tasks.yield_to_queue_capture_exception([&]() {
            llama_set_decode_phase(ctx_tgt, decode_phase);
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }

            t_verify_elapsed = ggml_time_us() - t_verify_start;

            if (ret == 0 && spec) {
                try {
                    speculative_ok = common_speculative_process(
                        spec.get(), batch_view);
                } catch (...) {
                    // Preserve the existing decode accounting/error terminal:
                    // queue cleanup completes here, then the speculative
                    // exception is rethrown at its former publication point.
                    speculative_exception = std::current_exception();
                }
            }
            }, speculative_exception);
        SRV_DBG("  verify ubatch: %d tok, %.1fms (%.2fms/tok)\n",
                batch_view.n_tokens, t_verify_elapsed / 1e3,
                t_verify_elapsed / 1e3 / std::max(1, batch_view.n_tokens));

        if (vbr_capture_readiness_estimator && t_verify_elapsed > 0 &&
            batch_view.n_tokens > 0) {
            vbr_capture_readiness_estimator->observe_growth(
                uint64_t(batch_view.n_tokens), uint64_t(t_verify_elapsed));
        }

        auto * memory = llama_get_memory(ctx_tgt);
        std::vector<vbr_hard_seal_subject> hard_seal_evidence;
        memory->vbr_hard_seal_evidence_take(hard_seal_evidence);
        if (params_base.cache_debug) {
            for (const auto & step : hard_seal_evidence) {
                const json payload = {
                    {"evidence_event", "sealed_step"},
                    {"order_ordinal", step.order_ordinal},
                    {"layer", step.il},
                    {"side", step.is_v ? "v" : "k"},
                };
                SRV_INF("CACHE_VBR_HARD_SEAL %s\n", payload.dump().c_str());
            }
        }
        const bool hard_seal_terminal =
            memory->vbr_hard_seal_blocked_take(ret != 0);
        const auto decode_terminal = server_speculative_decode_terminal_resolve(
            ret, hard_seal_terminal, n_batch == 1,
            yield_exception != nullptr, speculative_ok);
        if (ret != 0) {
            if (decode_terminal ==
                    server_speculative_decode_terminal::preserve_hard_seal) {
                constexpr const char * refusal = "hard_lease_blocked";
                SRV_WRN("VBR pressure refused at the hard-lease seal; completing the in-flight request with %s\n",
                        refusal);
                for (auto & slot : slots) {
                    if (slot.is_processing()) {
                        send_error(slot, refusal, ERROR_TYPE_HARD_LEASE_BLOCKED);
                        // prepare() failed before committing this batch. Keep
                        // the proven live prefix and its lease intact; unlike
                        // an ordinary cache-placement failure this must never
                        // clear an idle leased slot or retry forever.
                        slot.release();
                    }
                }
                throw std::runtime_error(refusal);
            }
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (decode_terminal ==
                        server_speculative_decode_terminal::ordinary_ret_error) {
                    GGML_ASSERT(!err.empty());
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.mandatory_recovery_reset(
                                server_cache_destruction_reason::restore_failure);
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            GGML_ASSERT(decode_terminal ==
                server_speculative_decode_terminal::retry);
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch_view.n_tokens, has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (decode_terminal ==
                server_speculative_decode_terminal::reset_committed_then_throw) {
            // Target verification has already committed this batch, including
            // draft tokens tentatively appended to each live prompt. Never let
            // a queue/speculative failure publish that suffix as reusable state.
            server_reset_processing_slots_after_committed_failure(slots);

            if (yield_exception) {
                std::rethrow_exception(yield_exception);
            }

            SRV_ERR("%s", "failed to process speculative batch\n");
            throw std::runtime_error("failed to process speculative batch");
        }
        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    // fork 9l-d4b6w: capture_vbr_swa_prompt_frontier (fork server-context.cpp 18695-18760)
    bool capture_vbr_swa_prompt_frontier(
            server_slot & slot, int32_t output_index) noexcept {
        if (!params_base.vbr_prompt_cache || !prompt_cache ||
            !vbr_artifact_store || n_swa <= 0 || ctx_dft ||
            slot.can_speculate() || vbr_idle_exact_capture ||
            slot.state != SLOT_STATE_DONE_PROMPT || !slot.is_processing() ||
            slot.prompt.n_tokens() <
                int64_t(SERVER_PROMPT_CACHE_MIN_RETENTION_REUSE_TOKENS)) {
            return false;
        }

        // Bind the terminal row before the capture so an exact host hit can
        // sample immediately without re-running the final prompt token.  A
        // missing row only removes that shortcut; it must not invalidate an
        // otherwise complete prompt-frontier artifact. Successful output
        // decode synchronized the target immediately before this call.
        (void) slot.capture_synchronized_frontier_logits(output_index);

        // Normal live-slot retention publication happens at release(), after
        // generation.  Prompt-boundary sealing deliberately precedes that
        // terminal, so publish the same immutable lineage source early.  The
        // later release refresh is idempotent and remains the general owner.
        if (!prompt_cache->vbr_retention_source_available(slot.id)) {
            if (!slot.retention_obs || !slot.task || slot.task->is_child()) {
                return false;
            }
            (void) ensure_frontier_sequence_epoch(slot.prompt);
            const auto live_key =
                server_retention_instance_key::for_slot(slot.id);
            const std::string adapter_identity =
                lora_config_identity(slot.lora);
            server_cache_lease_identity identity;
            const bool identity_known = slot.lease_obs &&
                slot.lease_execution_identity &&
                server_cache_lease_build_identity(
                    *slot.lease_execution_identity, adapter_identity,
                    slot.prompt.tokens, slot.prompt.n_tokens(), identity);
            const bool published = slot.retention_obs->publish(
                live_key, slot.retention_pool,
                slot.task->params.message_spans,
                !slot.task->params.message_spans.spans.empty(),
                uint64_t(slot.prompt.n_tokens()),
                uint64_t(slot.prompt.n_tokens()), true,
                identity_known ? &identity : nullptr,
                nullptr, nullptr,
                slot.retention_destination.valid()
                    ? &slot.retention_destination : nullptr);
            if (!published ||
                !server_prompt_retention_publish_exact_prefix(
                    *slot.retention_obs, live_key, slot.prompt,
                    adapter_identity, slot.prompt.n_tokens())) {
                slot.retention_obs->retire(live_key);
                return false;
            }
        }

        vbr_prompt_boundary_capture_slot = slot.id;
        auto capture = queue_tasks.try_begin_prompt_boundary_capture();
        if (capture) {
            (void) publish_idle_vbr_batch(capture, true);
        }
        vbr_prompt_boundary_capture_slot = -1;

        return vbr_idle_exact_capture &&
            vbr_idle_exact_capture->candidate.active_prompt_frontier;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            // fork 9l-d4b6x: wait_for_prompt_capture declaration (fork server-context.cpp 18830)
            bool wait_for_prompt_capture = false;
            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                const bool snapshot_prev = params_base.checkpoint_recurrent_prev &&
                        params_base.n_ctx_checkpoints > 0 && spec == nullptr &&
                        llama_n_rs_seq(ctx_tgt) > 0 && slot.stats.n_prompt_processed > 64 &&
                        (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                         ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS || n_swa > 0);
                if (snapshot_prev && slot.prompt.n_tokens() > 0) {
                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
                    if (pos_min > 0 && pos_max > 0) {
                        create_checkpoint(slot, /* n_tokens_cur = */ 1, pos_min - 1, pos_max - 1,
                                          /* is_replay_boundary = */ true,
                                          LLAMA_STATE_SEQ_FLAGS_ON_DEVICE | LLAMA_STATE_SEQ_FLAGS_RECURRENT_PREV);
                    }
                }

                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // fork 9l-d4b6x: wait_for_prompt_capture assignment (fork server-context.cpp 18861-18862)
                wait_for_prompt_capture = capture_vbr_swa_prompt_frontier(
                    slot, slot.i_batch-off);

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // fork 9l-d4b6x: hold the slot for the prompt-frontier capture (fork server-context.cpp 18994-19003)
                if (wait_for_prompt_capture && vbr_idle_exact_capture) {
                    // The sampled stop token is not decoded into KV, but the
                    // prompt-frontier worker still owns this slot as its live
                    // source. Cancel and drain it before release tears that
                    // source down. A later idle publication can retry the
                    // same stable frontier if it remains useful.
                    slot.state = SLOT_STATE_WAIT_VBR_CAPTURE;
                    vbr_idle_exact_capture->session.cancel();
                    (void) finish_idle_exact_vbr_capture(false);
                }
                // release slot because of stop condition
                // fork 9l-d4b6y: frontier-logits release capture (fork server-context.cpp 19005-19015)
                if (params_base.vbr_prompt_cache ||
                    !params_base.slot_save_path.empty()) {
                    (void) ensure_frontier_sequence_epoch(slot.prompt);
                    const bool frontier_logits_retained =
                        slot.capture_synchronized_frontier_logits(tok_idx);
                    SLT_DBG(slot,
                            "frontier logits capture at output %d: %s\n",
                            tok_idx,
                            frontier_logits_retained
                                ? "retained" : "unavailable");
                }
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            // fork 9l-d4b6x: hold the slot at the immutable prompt frontier (fork server-context.cpp 19023-19029)
            if (wait_for_prompt_capture) {
                // The first token is already sampled and delivered, but has
                // not been decoded into KV yet. Hold only this source slot at
                // the immutable prompt frontier; scheduler maintenance will
                // publish or abandon the capture and re-arm generation.
                slot.state = SLOT_STATE_WAIT_VBR_CAPTURE;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                auto accepted = synth_probs.empty()
                    ? common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft)
                    : server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }

    // -------------------------------------------------------------------------
    // Computation-frontier identity (slice 7d part 3).
    //
    // Landed inert: the consumers are the capture/restore paths of 7d part 4/5.
    // Bodies are verbatim from the fork (server-context.cpp:4585 for the epoch,
    // :11322 and :11357 for the identity block), with the port's adapter digest
    // and lora_config_identity() as the only external dependency.
    //
    // The identity block is what an artifact reference is authenticated against:
    // which execution namespace, which adapter set, which media prefix, how many
    // tokens and at which position. Every field is derived, never passed in.
    // -------------------------------------------------------------------------

    uint64_t ensure_frontier_sequence_epoch(server_prompt & prompt) {
        if (prompt.sequence_epoch == 0) {
            GGML_ASSERT(frontier_next_sequence_epoch != 0);
            prompt.sequence_epoch = frontier_next_sequence_epoch++;
            GGML_ASSERT(frontier_next_sequence_epoch != 0);
        }
        return prompt.sequence_epoch;
    }

    bool build_capture_identity_fields(
            server_slot & slot,
            vbr_artifact_identity_block & identity,
            server_vbr_artifact_capture_status & status,
            int64_t coverage_tokens = -1) {
        const int64_t token_count = coverage_tokens < 0
            ? slot.prompt.n_tokens() : coverage_tokens;
        const llama_pos next_position = coverage_tokens < 0
            ? slot.prompt.tokens.pos_next()
            : slot.prompt.tokens.pos_next(size_t(coverage_tokens));
        identity.media_content_identity.clear();
        if (token_count <= 0 || token_count > slot.prompt.n_tokens() ||
            next_position <= 0 ||
            !slot.prompt.tokens.media_content_identity(
                token_count, identity.media_content_identity)) {
            status =
                server_vbr_artifact_capture_status::identity_unavailable;
            return false;
        }
        const uint64_t sequence_epoch =
            ensure_frontier_sequence_epoch(slot.prompt);
        if (!slot.vbr_adapter_config_identity_valid) {
            slot.vbr_adapter_config_identity =
                lora_config_identity(slot.lora);
            slot.vbr_adapter_config_identity_valid = true;
        }
        identity.execution_identity = frontier_execution_identity;
        identity.adapter_config_identity = slot.vbr_adapter_config_identity;
        identity.sequence_epoch = sequence_epoch;
        identity.token_count = token_count;
        identity.next_position = next_position;
        status = server_vbr_artifact_capture_status::ok;
        return true;
    }

    bool build_capture_identity(
            server_slot & slot,
            vbr_artifact_identity_block & identity,
            std::vector<llama_token> & token_block,
            server_vbr_artifact_capture_status & status) {
        token_block.clear();
        if (!build_capture_identity_fields(slot, identity, status)) {
            return false;
        }
        try {
            token_block.reserve(size_t(identity.token_count));
            for (int64_t i = 0; i < identity.token_count; ++i) {
                token_block.push_back(slot.prompt.tokens[size_t(i)]);
            }
        } catch (...) {
            token_block.clear();
            status = server_vbr_artifact_capture_status::internal_error;
            return false;
        }
        return true;
    }

// fork 9l-d2a: exact-restore durability marking (fork server-context.cpp 11379-11406, 11550-11587)
    static std::array<uint8_t, 32> vbr_idle_capture_attempt_digest(
            const vbr_artifact_identity_block & identity,
            const std::array<uint8_t, 32> & token_identity_digest,
            uint64_t tier_epoch,
            uint64_t tier_epoch_swa) {
        llama_sha256_writer attempt_hash;
        static constexpr char ATTEMPT_DOMAIN[] =
            "buun.vbr.idle-capture-attempt/v2";
        attempt_hash.string(
            ATTEMPT_DOMAIN, sizeof(ATTEMPT_DOMAIN) - 1);
        attempt_hash.string(
            identity.execution_identity.data(),
            identity.execution_identity.size());
        attempt_hash.string(
            identity.adapter_config_identity.data(),
            identity.adapter_config_identity.size());
        attempt_hash.string(
            identity.media_content_identity.data(),
            identity.media_content_identity.size());
        attempt_hash.u64(identity.sequence_epoch);
        attempt_hash.u64(uint64_t(identity.token_count));
        attempt_hash.u64(uint64_t(identity.next_position));
        attempt_hash.bytes(
            token_identity_digest.data(), token_identity_digest.size());
        attempt_hash.u64(tier_epoch);
        attempt_hash.u64(tier_epoch_swa);
        return attempt_hash.finish();
    }

    static std::array<uint8_t, 32> vbr_idle_publication_identity(
            const std::array<uint8_t, 32> & token_identity_digest,
            uint64_t tier_epoch,
            uint64_t tier_epoch_swa) {
        llama_sha256_writer published_hash;
        static constexpr char PUBLISHED_DOMAIN[] =
            "buun.vbr.idle-capture-published/v1";
        published_hash.string(
            PUBLISHED_DOMAIN, sizeof(PUBLISHED_DOMAIN) - 1);
        published_hash.bytes(
            token_identity_digest.data(), token_identity_digest.size());
        published_hash.u64(tier_epoch);
        published_hash.u64(tier_epoch_swa);
        return published_hash.finish();
    }

    void mark_exact_vbr_restore_durable(
            server_slot & slot, llama_memory_i & memory) {
        std::array<uint8_t, 32> token_digest = {};
        vbr_artifact_identity_block identity;
        server_vbr_artifact_capture_status status;
        if (!slot.prompt.tokens.retention_token_digest(token_digest) ||
            !build_capture_identity_fields(slot, identity, status)) {
            return;
        }
        const auto representation = memory.vbr_representation_identity();
        slot.vbr_idle_capture_source_reset();
        slot.vbr_idle_capture_published_identity =
            vbr_idle_publication_identity(
                token_digest, representation.tier_epoch,
                representation.tier_epoch_swa);
        slot.vbr_idle_capture_representation_valid = true;
        slot.vbr_idle_capture_attempt_identity =
            vbr_idle_capture_attempt_digest(
                identity, token_digest, representation.tier_epoch,
                representation.tier_epoch_swa);
        slot.vbr_idle_capture_terminal = true;
    }

    bool build_capture_request(
            server_slot & slot,
            vbr_explicit_capture_request & request,
            server_vbr_artifact_capture_status & status,
            const common_prompt_checkpoint * checkpoint = nullptr) {
        request = {};
        if (checkpoint) {
            if (!checkpoint->computation_frontier.valid() ||
                checkpoint->n_tokens <= 0 ||
                checkpoint->n_tokens >= slot.prompt.n_tokens() ||
                checkpoint->computation_frontier.token_count !=
                    checkpoint->n_tokens ||
                checkpoint->computation_frontier.next_position <= 0 ||
                checkpoint->computation_frontier.sequence_epoch !=
                    slot.prompt.sequence_epoch ||
                checkpoint->computation_frontier.execution_identity !=
                    frontier_execution_identity ||
                checkpoint->computation_frontier.adapter_config_identity !=
                    lora_config_identity(slot.lora) ||
                !build_capture_identity_fields(
                    slot, request.identity, status,
                    checkpoint->n_tokens) ||
                request.identity.next_position !=
                    checkpoint->computation_frontier.next_position ||
                request.identity.media_content_identity !=
                    checkpoint->computation_frontier.media_content_identity) {
                status = server_vbr_artifact_capture_status::
                    identity_unavailable;
                return false;
            }
            try {
                request.token_block.reserve(size_t(checkpoint->n_tokens));
                for (int64_t i = 0; i < checkpoint->n_tokens; ++i) {
                    request.token_block.push_back(
                        slot.prompt.tokens[size_t(i)]);
                }
            } catch (...) {
                request.token_block.clear();
                status = server_vbr_artifact_capture_status::internal_error;
                return false;
            }
        } else if (!build_capture_identity(
                       slot, request.identity, request.token_block, status)) {
            return false;
        }
        // The first durable capture mints a nonzero sequence epoch. A raw
        // frontier row retained at generation stop may therefore still be
        // bound to epoch zero even though its token digest is exact.
        slot.bind_frontier_logits_to_prompt();
        request.sequence = slot.id;
        request.frontier.execution_identity =
            request.identity.execution_identity.data();
        request.frontier.execution_identity_len =
            request.identity.execution_identity.size();
        request.frontier.adapter_config_identity =
            request.identity.adapter_config_identity.data();
        request.frontier.adapter_config_identity_len =
            request.identity.adapter_config_identity.size();
        request.frontier.media_content_identity =
            request.identity.media_content_identity.data();
        request.frontier.media_content_identity_len =
            request.identity.media_content_identity.size();
        request.frontier.sequence_epoch = request.identity.sequence_epoch;
        request.frontier.token_count = request.identity.token_count;
        request.frontier.next_position = request.identity.next_position;
        request.idle_decode_thread = true;

        const auto pageable_domain =
            vbr_artifact_portable_domain {
                llama_cache_acct_residency::pageable_host,
                llama_cache_acct_domain_kind::not_applicable,
                UINT32_MAX, UINT16_MAX,
            };
        const auto add_checkpoint_buffer = [&request, pageable_domain](
                const server_vbr_companion_codec & codec,
                const common_shared_byte_buffer & bytes) {
            if (bytes.empty()) {
                return false;
            }
            vbr_explicit_companion_provider provider;
            provider.kind = codec.kind;
            provider.format_version = codec.format_version;
            provider.build_identity_digest =
                codec.build_identity_digest;
            provider.domain = pageable_domain;
            provider.context = &bytes;
            provider.size = [](
                    const void * context,
                    llama_seq_id,
                    uint64_t & output) noexcept {
                const auto * source =
                    static_cast<const common_shared_byte_buffer *>(context);
                output = source ? uint64_t(source->size()) : 0;
                return output != 0;
            };
            provider.capture_stream = [](
                    const void * context,
                    llama_seq_id,
                    llama_io_write_i & output) {
                const auto * source =
                    static_cast<const common_shared_byte_buffer *>(context);
                if (!source || source->empty()) {
                    return false;
                }
                const auto view = source->view();
                output.write(view.data(), view.size());
                return true;
            };
            if (codec.kind == vbr_artifact_companion_kind::recurrent) {
                provider.terminal_position = [](
                        const void * context,
                        llama_seq_id,
                        llama_pos & output) noexcept {
                    const auto * source = static_cast<
                        const common_shared_byte_buffer *>(context);
                    if (!source || source->empty()) {
                        return false;
                    }
                    const auto view = source->view();
                    return vbr_explicit_recurrent_companion_terminal(
                        view.data(), view.size(), output);
                };
            } else if (codec.kind == vbr_artifact_companion_kind::qsa_index) {
                provider.terminal_position = [](
                        const void * context,
                        llama_seq_id,
                        llama_pos & output) noexcept {
                    const auto * source = static_cast<
                        const common_shared_byte_buffer *>(context);
                    if (!source || source->empty()) {
                        return false;
                    }
                    const auto view = source->view();
                    return vbr_qsa_index_companion_terminal(
                        view.data(), view.size(), output);
                };
            } else if (codec.kind ==
                    vbr_artifact_companion_kind::typed_accelerator) {
                provider.terminal_position = [](
                        const void * context,
                        llama_seq_id,
                        llama_pos & output) noexcept {
                    const auto * source = static_cast<
                        const common_shared_byte_buffer *>(context);
                    if (!source || source->empty()) {
                        return false;
                    }
                    const auto view = source->view();
                    return common_speculative_ring_state_serialized_terminal(
                        view.data(), view.size(), output);
                };
            }
            request.companions.push_back(provider);
            return true;
        };

        if (checkpoint) {
            const server_vbr_companion_codec recurrent_codec {
                vbr_artifact_companion_kind::recurrent, 1,
                vbr_explicit_recurrent_companion_build_identity(),
            };
            server_vbr_companion_codec draft_codec;
            server_vbr_companion_codec accelerator_codec;
            server_vbr_companion_codec qsa_codec;
            const bool draft_codec_ready = !ctx_dft ||
                server_vbr_companion_codec_for(
                    vbr_artifact_companion_kind::required_spec_payload,
                    draft_codec);
            const bool accelerator_codec_ready = !slot.can_speculate() ||
                server_vbr_companion_codec_for(
                    vbr_artifact_companion_kind::typed_accelerator,
                    accelerator_codec);
            std::vector<llama_memory_tree_child> tree;
            size_t recurrent_children = 0;
            bool payload_complete_attention = false;
            if (!llama_memory_tree_collect(
                    llama_get_memory(ctx_tgt), tree)) {
                status = server_vbr_artifact_capture_status::unsupported;
                return false;
            }
            const bool needs_qsa = std::any_of(
                tree.begin(), tree.end(), [](const auto & child) {
                    return child.qsa_index_owner != nullptr;
                });
            const bool qsa_codec_ready = !needs_qsa ||
                server_vbr_companion_codec_for(
                    vbr_artifact_companion_kind::qsa_index, qsa_codec);
            for (const auto & child : tree) {
                recurrent_children += child.recurrent != nullptr;
                payload_complete_attention |= child.attention != nullptr &&
                    child.dependency_mode ==
                        checkpoint_child_dependency_mode::payload_complete;
            }
            // SWA/iSWA checkpoint bytes contain tier-sensitive attention
            // state.  They remain unavailable until that state has its own
            // artifact owner; treating it as a recurrent blob would duplicate
            // or mismatch the VBR attention tree.
            if (!draft_codec_ready || !accelerator_codec_ready ||
                !qsa_codec_ready ||
                payload_complete_attention || recurrent_children > 1 ||
                (recurrent_children != 0 &&
                 !add_checkpoint_buffer(
                     recurrent_codec,
                     checkpoint->data_tgt)) ||
                (needs_qsa && !add_checkpoint_buffer(
                     qsa_codec, checkpoint->data_qsa)) ||
                (ctx_dft && !add_checkpoint_buffer(
                     draft_codec,
                     checkpoint->data_dft)) ||
                (slot.can_speculate() && !checkpoint->accel.ring.empty() &&
                 !add_checkpoint_buffer(
                     accelerator_codec,
                     checkpoint->accel.ring))) {
                status = server_vbr_artifact_capture_status::
                    required_companion_unavailable;
                return false;
            }
            status = server_vbr_artifact_capture_status::ok;
            return true;
        }

        std::vector<llama_memory_tree_child> live_tree;
        if (!llama_memory_tree_collect(llama_get_memory(ctx_tgt), live_tree)) {
            status = server_vbr_artifact_capture_status::unsupported;
            return false;
        }
        for (const auto & child : live_tree) {
            if (child.qsa_index_owner) {
                request.companions.push_back(
                    vbr_qsa_index_capture_provider(*child.qsa_index_owner));
            }
        }

        if (ctx_dft) {
            server_vbr_companion_codec codec;
            if (!server_vbr_companion_codec_for(
                    vbr_artifact_companion_kind::required_spec_payload,
                    codec)) {
                status = server_vbr_artifact_capture_status::
                    required_companion_unavailable;
                return false;
            }
            vbr_explicit_companion_provider draft;
            draft.kind = codec.kind;
            draft.format_version = codec.format_version;
            draft.build_identity_digest = codec.build_identity_digest;
            draft.domain = pageable_domain;
            // FORK-DIFF: the fork keeps the draft context in a llama_context_ptr and
            // passes ctx_dft.get() (fork server-context.cpp:3489); the port keeps a raw
            // llama_context * (see bind_slots_to_draft_context), so the pointer is the
            // value.
            draft.context = ctx_dft;
            draft.size = [](
                    const void * context,
                    llama_seq_id sequence,
                    uint64_t & output) noexcept {
                const auto size =
                    llama_state_seq_get_size_ext(
                        static_cast<llama_context *>(
                            const_cast<void *>(context)),
                        sequence,
                        LLAMA_STATE_SEQ_FLAGS_NONE);
                output = uint64_t(size);
                return size != 0;
            };
            draft.capture = [](
                    const void * context,
                    llama_seq_id sequence,
                    std::vector<uint8_t> & output) noexcept {
                try {
                    auto * ctx =
                        static_cast<llama_context *>(
                            const_cast<void *>(context));
                    const size_t size =
                        llama_state_seq_get_size_ext(
                            ctx, sequence,
                            LLAMA_STATE_SEQ_FLAGS_NONE);
                    if (size == 0) {
                        return false;
                    }
                    output.resize(size);
                    return llama_state_seq_get_data_ext(
                               ctx, output.data(), size, sequence,
                               LLAMA_STATE_SEQ_FLAGS_NONE) == size;
                } catch (...) {
                    output.clear();
                    return false;
                }
            };
            draft.capture_stream = [](
                    const void * context,
                    llama_seq_id sequence,
                    llama_io_write_i & output) {
                auto * ctx = static_cast<llama_context *>(
                    const_cast<void *>(context));
                if (!ctx) {
                    return false;
                }
                const size_t expected = llama_state_seq_get_size_ext(
                    ctx, sequence, LLAMA_STATE_SEQ_FLAGS_NONE);
                const size_t written = ctx->state_seq_write_data_stream(
                    output, sequence, LLAMA_STATE_SEQ_FLAGS_NONE);
                if (written != expected || output.n_bytes() != expected) {
                    SRV_DBG(
                        "VBR draft companion stream size mismatch "
                        "expected=%zu written=%zu writer=%zu\n",
                        expected, written, output.n_bytes());
                    return false;
                }
                return written != 0;
            };
            request.companions.push_back(draft);
        }
        if (slot.can_speculate()) {
            const size_t ring_size =
                common_speculative_ring_state_size(slot.get_spec());
            if (ring_size != 0) {
                server_vbr_companion_codec codec;
                if (!server_vbr_companion_codec_for(
                        vbr_artifact_companion_kind::typed_accelerator,
                        codec)) {
                    status = server_vbr_artifact_capture_status::
                        required_companion_unavailable;
                    return false;
                }
                vbr_explicit_companion_provider accelerator;
                accelerator.kind = codec.kind;
                accelerator.format_version = codec.format_version;
                accelerator.build_identity_digest =
                    codec.build_identity_digest;
                accelerator.domain = pageable_domain;
                accelerator.context = &slot;
                accelerator.size = [](
                        const void * context,
                        llama_seq_id,
                        uint64_t & output) noexcept {
                    const auto * owner =
                        static_cast<const server_slot *>(context);
                    const size_t size =
                        common_speculative_ring_state_size(
                            owner->get_spec());
                    output = uint64_t(size);
                    return size != 0;
                };
                accelerator.capture = [](
                        const void * context,
                        llama_seq_id,
                        std::vector<uint8_t> & output) noexcept {
                    try {
                        const auto * owner =
                            static_cast<const server_slot *>(context);
                        const size_t size =
                            common_speculative_ring_state_size(
                                owner->get_spec());
                        if (size == 0) {
                            return false;
                        }
                        output.resize(size);
                        common_speculative_ring_state_save(
                            owner->get_spec(),
                            output.data(), output.size());
                        return true;
                    } catch (...) {
                        output.clear();
                        return false;
                    }
                };
                accelerator.capture_stream = [](
                        const void * context,
                        llama_seq_id,
                        llama_io_write_i & output) {
                    const auto * owner =
                        static_cast<const server_slot *>(context);
                    return owner && common_speculative_ring_state_write(
                        owner->get_spec(), output);
                };
                accelerator.terminal_position = [](
                        const void * context,
                        llama_seq_id,
                        llama_pos & output) noexcept {
                    const auto * owner =
                        static_cast<const server_slot *>(context);
                    return owner && common_speculative_ring_state_terminal(
                        owner->get_spec(), output);
                };
                request.companions.push_back(accelerator);
            }
        }
        server_vbr_companion_codec logits_codec;
        if (!server_vbr_companion_codec_for(
                vbr_artifact_companion_kind::frontier_logits,
                logits_codec)) {
            status = server_vbr_artifact_capture_status::
                required_companion_unavailable;
            return false;
        }
        vbr_explicit_companion_provider frontier_logits;
        frontier_logits.kind = logits_codec.kind;
        frontier_logits.format_version = logits_codec.format_version;
        frontier_logits.build_identity_digest =
            logits_codec.build_identity_digest;
        frontier_logits.domain = pageable_domain;
        frontier_logits.required = false;
        frontier_logits.context = &slot;
        frontier_logits.size = [](
                const void * context,
                llama_seq_id,
                uint64_t & output) noexcept {
            const auto * owner = static_cast<const server_slot *>(context);
            if (!owner || !owner->frontier_logits_matches() ||
                owner->frontier_logits.values.size() >
                    UINT64_MAX/sizeof(float)) {
                output = 0;
                return false;
            }
            output = uint64_t(owner->frontier_logits.values.size())*sizeof(float);
            return output != 0;
        };
        frontier_logits.capture = [](
                const void * context,
                llama_seq_id,
                std::vector<uint8_t> & output) noexcept {
            try {
                const auto * owner = static_cast<const server_slot *>(context);
                if (!owner || !owner->frontier_logits_matches()) {
                    return false;
                }
                output.resize(owner->frontier_logits.values.size()*sizeof(float));
                std::memcpy(output.data(), owner->frontier_logits.values.data(),
                            output.size());
                return !output.empty();
            } catch (...) {
                output.clear();
                return false;
            }
        };
        frontier_logits.capture_stream = [](
                const void * context,
                llama_seq_id,
                llama_io_write_i & output) {
            const auto * owner = static_cast<const server_slot *>(context);
            if (!owner || !owner->frontier_logits_matches()) {
                return false;
            }
            output.write(
                owner->frontier_logits.values.data(),
                owner->frontier_logits.values.size()*sizeof(float));
            return true;
        };
        request.companions.push_back(frontier_logits);
        status = server_vbr_artifact_capture_status::ok;
        return true;
    }
    // ---------------------------------------------------------------------
    // group A phase 2: prereq members, verbatim from the fork
    // ---------------------------------------------------------------------

    // fork server-context.cpp:3548
    std::vector<server_prompt_cache_vbr_frontier_query>
        vbr_idle_frontier_queries;
    // fork server-context.cpp:3540
    std::unique_ptr<server_retention_sidecar_store> cache_retention_metadata;
    // fork server-context.cpp:3630-3631
    std::unique_ptr<server_vbr_capture_readiness_estimator>
        vbr_capture_readiness_estimator;

    // fork server-context.cpp:5464  (6 lines)
    server_retention_sidecar_store * live_retention_owner() noexcept {
        if (cache_authority) {
            return &cache_authority->retention;
        }
        return cache_retention_metadata.get();
    }

    // fork server-context.cpp:5556  (8 lines)
    server_slot * find_slot_by_id(int32_t slot_id) noexcept {
        const auto found = std::find_if(
            slots.begin(), slots.end(),
            [slot_id](const server_slot & slot) {
                return slot.id == slot_id;
            });
        return found == slots.end() ? nullptr : &*found;
    }

    // fork server-context.cpp:4704  (21 lines)
    bool checkpoint_frontier_is_current(
            const server_slot & slot,
            const common_prompt_checkpoint & checkpoint,
            const std::string & adapter_identity) const {
        const auto & frontier = checkpoint.computation_frontier;
        if (!frontier.valid() ||
            frontier.sequence_epoch != slot.prompt.sequence_epoch ||
            frontier.execution_identity != frontier_execution_identity ||
            frontier.adapter_config_identity != adapter_identity ||
            frontier.token_count != checkpoint.n_tokens ||
            checkpoint.pos_max < 0 ||
            frontier.next_position <= 0 ||
            frontier.next_position - 1 != checkpoint.pos_max) {
            return false;
        }

        std::string media_identity;
        return slot.prompt.tokens.media_content_identity(
                   frontier.token_count, media_identity) &&
               media_identity == frontier.media_content_identity;
    }

    // fork server-context.cpp:4726  (4 lines)
    bool active_prefix_enabled() const {
        return slots.size() > 1 && prompt_cache && params_base.kv_unified &&
            !params_base.ctx_shift && params_base.n_cache_reuse == 0;
    }

    // fork server-context.cpp:4731  (11 lines)
    // FORK-DIFF: the fork also accepts COMMON_SPECULATIVE_TYPE_DFLASH here. That enumerator
    // does not exist in this tree (common/common.h jumps from DRAFT_DFLASH to DRAFT_DSPARK),
    // so no value can reach this loop equal to it and the branch is dropped. Re-add it in the
    // same position if the enumerator is ever ported.
    std::optional<common_speculative_type> active_prefix_draft_type() const {
        auto selected = COMMON_SPECULATIVE_TYPE_NONE;
        for (auto type : params_base.speculative.types) {
            if (type == COMMON_SPECULATIVE_TYPE_NONE) { continue; }
            if (selected != COMMON_SPECULATIVE_TYPE_NONE ||
                (type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP &&
                 type != COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) { return {}; }
            selected = type;
        }
        return selected;
    }

    // fork server-context.cpp:4744  (9 lines)
    bool swa_window_eligible(const server_slot & slot) const {
        return swa_window_checkpoint_limit > 0 && active_prefix_enabled() &&
            !ctx_dft && !slot.can_speculate() && !slot.diff_self_spec && slot.lora.empty() &&
            slot.task && slot.task->type == SERVER_TASK_TYPE_COMPLETION && slot.task->params.cache_prompt &&
            !slot.task->is_parent() && !slot.task->is_child() &&
            !slot.task->tokens.has_media() && !slot.prompt.tokens.has_media() &&
            std::all_of(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                [](auto type) { return type == COMMON_SPECULATIVE_TYPE_NONE; });
    }

    // fork server-context.cpp:4754  (3 lines)
    static std::shared_ptr<void> reserve_swa_staging(void * opaque, size_t bytes) {
        return static_cast<server_prompt_cache *>(opaque)->reserve_active_storage(bytes);
    }

    // fork server-context.cpp:4758  (3 lines)
    vbr_explicit_representation_policy swa_window_representation() const {
        return {frontier_execution_identity.data(), frontier_execution_identity.size()};
    }

    // fork server-context.cpp:4762  (30 lines)
    void capture_swa_window(server_slot & slot, int64_t frontier) {
        if (frontier <= 0 || frontier > INT32_MAX || slot.prompt.tokens.pos_next(frontier) != frontier ||
            (!slot.swa_windows.empty() && slot.swa_windows.back().image->frontier() == frontier)) { return; }
        const int64_t start = ggml_time_us();
        try {
            struct reservation { server_prompt_cache * cache; size_t tokens; } budget {prompt_cache.get(), size_t(frontier)};
            vbr_swa_window_capture_request capture;
            capture.sequence = slot.id; capture.frontier = frontier;
            capture.sequence_epoch = ensure_frontier_sequence_epoch(slot.prompt);
            capture.execution_identity = swa_window_execution_identity;
            capture.representation = swa_window_representation();
            capture.capacity_context = &budget;
            capture.reserve = [](void * opaque, size_t bytes) -> std::shared_ptr<void> {
                auto & b = *static_cast<reservation *>(opaque);
                if (bytes > SIZE_MAX-sizeof(server_swa_window_checkpoint)) { return {}; }
                return b.cache->reserve_active_storage(bytes+sizeof(server_swa_window_checkpoint), b.tokens);
            };
            auto result = vbr_capture_swa_window(*ctx_tgt, capture);
            if (result.status != vbr_swa_window_status::ok) {
                SLT_DBG(slot, "SWA window capture skipped: status=%d frontier=%" PRId64 "\n", int(result.status), frontier);
                return;
            }
            slot.swa_windows.push_back({capture.sequence_epoch, std::move(result.image)});
            while (slot.swa_windows.size() > size_t(swa_window_checkpoint_limit)) { slot.swa_windows.pop_front(); }
            SLT_INF(slot, "SWA window captured: frontier=%" PRId64 " windows=%zu host_bytes=%zu capture_ms=%.3f\n",
                    frontier, slot.swa_windows.size(), prompt_cache->active_storage_bytes(), (ggml_time_us()-start)/1000.0);
        } catch (const std::bad_alloc &) {
            SLT_WRN(slot, "%s", "SWA window capture skipped: host allocation unavailable\n");
        }
    }


    // Late fallback only: normal live/host/checkpoint selection and its approved
    // clearing have already left an empty destination. Never displace a hit or
    // change a pre-mutation authoritative plan. The source and any borrowed
    // checkpoint remain alive for this entire non-yielding scheduler callback.
    void restore_active_prefix(server_slot & dst) {
        if (!active_prefix_enabled() ||
            (!llama_model_is_hybrid(model_tgt) && n_swa <= 0) || dst.diff_self_spec ||
            !dst.task || dst.task->type != SERVER_TASK_TYPE_COMPLETION ||
            !dst.task->params.cache_prompt || dst.task->is_parent() || dst.task->is_child() ||
            !dst.prompt.tokens.empty() || !dst.prompt.checkpoints.empty() ||
            !dst.lora.empty()) {
            return;
        }
        const auto draft_type = active_prefix_draft_type();
        if (!draft_type) { return; }
        const bool mtp = *draft_type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
        // FORK-DIFF (fork server-context.cpp:4809): the fork's native DFlash block-diffusion
        // mode is absent from this tree -- common_speculative_type has no
        // COMMON_SPECULATIVE_TYPE_DFLASH member -- so this comparison can never hold. This is
        // the exact result of that member being absent, not an approximation of it.
        const bool dflash = false;
        const bool shared_dflash = *draft_type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
        const bool drafting = mtp || dflash || shared_dflash;
        // FORK-DIFF (fork server-context.cpp:4813): ctx_dft is a raw llama_context * in this
        // tree (port:390) where the fork holds a llama_context_ptr, and the dflash limb above
        // is constant-false, so the shared-draft arm is unreachable here.
        llama_context * draft_ctx = ctx_dft;
        if (drafting != bool(draft_ctx) || drafting != dst.can_speculate()) { return; }
        // DFlash media/rebased positions need their own qualification. Existing
        // MTP/media and target-only SWA admission are unchanged.
        if ((dflash || shared_dflash) && dst.task->tokens.has_media()) { return; }
        // Retained attention windows use zero-copy sharing. Recycled windows
        // may use the separately qualified typed historical companion.
        const bool live_attention = server_vbr_dynamic_active(params_base) &&
            n_swa > 0 && !llama_model_is_hybrid(model_tgt) && !drafting;

        auto * rec = dst.cache_plan.get();
        common_cache_plan_candidate * row = nullptr;
        bool installed = false;
        try {
            const int64_t start = ggml_time_us();
            struct candidate {
                const server_slot * source = nullptr;
                const common_prompt_checkpoint * checkpoint = nullptr;
                int64_t n_tokens = 0;
                std::vector<llama_pos> rows;
                size_t window_bytes = 0;
            } best;
            std::vector<std::pair<const server_slot *, const server_swa_window_checkpoint *>> windows;
            std::optional<server_tokens> installed_prefix;
            const auto adapter = lora_config_identity(dst.lora);
            const auto mem = llama_get_memory(ctx_tgt);
            const auto provider = live_attention
                ? common_cache_plan_provider::active_attention_prefix
                : common_cache_plan_provider::active_context_checkpoint;
            std::vector<float> carry;

            for (const auto & source : slots) {
                if (source.id == dst.id || source.state != SLOT_STATE_GENERATING ||
                    !source.task || !source.task->params.cache_prompt ||
                    source.task->is_parent() || source.task->is_child() ||
                    !source.lora.empty() ||
                    source.can_speculate() != drafting ||
                    ((dflash || shared_dflash) && source.prompt.tokens.has_media()) ||
                    (!source.prompt.tokens.has_media() &&
                     source.prompt.tokens.pos_next() != source.prompt.n_tokens())) {
                    continue;
                }
                // Always replay at least one input token to produce A2's own logits.
                const size_t lcp = std::min(source.prompt.tokens.get_common_prefix(dst.task->tokens),
                                           dst.task->tokens.size() - 1);
                if (live_attention) {
                    if (lcp <= size_t(best.n_tokens) || lcp > size_t(INT32_MAX)) { continue; }
                    std::vector<llama_pos> rows;
                    if (source.prompt.tokens.has_media() || dst.task->tokens.has_media()) {
                        std::string source_media, destination_media;
                        if (!source.prompt.tokens.media_content_identity(lcp, source_media) ||
                            !dst.task->tokens.media_content_identity(lcp, destination_media) ||
                            source_media != destination_media ||
                            source.prompt.tokens.pos_next(lcp) != dst.task->tokens.pos_next(lcp)) {
                            continue;
                        }
                        rows = dst.task->tokens.prefix_row_positions(lcp);
                        if (!mem->can_share_live_prefix_rows(source.id, dst.id,
                                dst.task->tokens.pos_next(lcp), rows)) {
                            continue;
                        }
                    } else if (!mem->can_share_live_prefix(source.id, dst.id, llama_pos(lcp))) {
                        if (swa_window_eligible(dst) && swa_window_eligible(source)) {
                            for (auto it = source.swa_windows.rbegin(); it != source.swa_windows.rend(); ++it) {
                                if (it->image->frontier() > int64_t(lcp) || it->image->frontier() <= best.n_tokens ||
                                    it->sequence_epoch != source.prompt.sequence_epoch) { continue; }
                                windows.emplace_back(&source, &*it);
                            }
                        }
                        continue;
                    }
                    best = { &source, nullptr, int64_t(lcp), std::move(rows) };
                    continue;
                }
                // The shared rows may have been retiered since capture. That is
                // allowed, but a content edit/replacement invalidates the old RS
                // checkpoint even when the same token positions are present.
                std::optional<llama_memory_vbr_state_data> source_vbr;
                for (const auto & cp : source.prompt.checkpoints) {
                    if (cp.n_tokens <= 0 || uint64_t(cp.n_tokens) > lcp ||
                        cp.n_tokens <= best.n_tokens ||
                        cp.pos_min < 0 || cp.pos_min > cp.pos_max ||
                        (llama_model_is_hybrid(model_tgt) && cp.pos_min != cp.pos_max) ||
                        cp.data_tgt.empty() || !cp.data_qsa.empty() || (!dflash && !cp.accel.ring.empty()) ||
                        !checkpoint_frontier_is_current(source, cp, adapter) ||
                        (source.retention_obs && !source.retention_obs->clone_source_available(
                            server_retention_instance_key::for_checkpoint(source.id, &cp)))) {
                        continue;
                    }
                    std::vector<llama_pos> rows;
                    if (source.prompt.tokens.has_media() || dst.task->tokens.has_media()) {
                        std::string media_identity;
                        if (!dst.task->tokens.media_content_identity(cp.n_tokens, media_identity) ||
                            media_identity != cp.computation_frontier.media_content_identity ||
                            dst.task->tokens.pos_next(cp.n_tokens) != cp.computation_frontier.next_position) {
                            continue;
                        }
                        rows = dst.task->tokens.prefix_row_positions(cp.n_tokens);
                        if (!mem->can_share_attn_prefix_rows(source.id, dst.id,
                                cp.computation_frontier.next_position, rows)) {
                            continue;
                        }
                    } else if (int64_t(cp.pos_max) + 1 != cp.n_tokens ||
                               !llama_memory_can_share_attn_prefix(mem, source.id, dst.id, cp.n_tokens)) {
                        continue;
                    }
                    if (server_vbr_dynamic_active(params_base)) {
                        // The pressure snapshot also scans occupancy. Do not pay
                        // for it on unrelated-prefix or fixed-KV requests.
                        if (!source_vbr) {
                            source_vbr = llama_memory_vbr_state(mem, source.id, 0);
                        }
                        if (!common_prompt_checkpoint_lineage_matches(cp, *source_vbr)) {
                            continue;
                        }
                    }
                    if (mtp) {
                        if (carry.empty()) {
                            carry.resize(llama_model_n_embd_out(model_tgt));
                        }
                        common_speculative_mtp_carry_lifecycle scratch;
                        if (cp.data_dft.empty() || !cp.data_dft_full_sequence ||
                            !common_speculative_mtp_carry_state_load(scratch, carry, cp.accel.spec.view())) {
                            continue;
                        }
                    }
                    if (shared_dflash && (cp.data_dft.empty() || !cp.data_dft_full_sequence || !cp.accel.spec.empty())) {
                        continue;
                    }
                    if (dflash && (!cp.accel.spec.empty() ||
                        !common_speculative_ring_state_matches_frontier(dst.get_spec(), cp.accel.ring.view().data(),
                            cp.accel.ring.size(), cp.pos_max))) { continue; }
                    best = { &source, &cp, cp.n_tokens, std::move(rows) };
                }
            }
            // Rank metadata first. A failed/larger proposal must not retain a
            // reservation that prevents a better candidate, or discard an
            // already feasible live hit. Live sharing wins equal-length ties.
            std::stable_sort(windows.begin(), windows.end(), [](const auto & a, const auto & b) {
                return a.second->image->frontier() > b.second->image->frontier();
            });
            bool synchronized = false;
            for (const auto & candidate : windows) {
                const auto & source = *candidate.first;
                const auto & image = candidate.second->image;
                if (image->frontier() <= best.n_tokens) { break; }
                if (!synchronized) {
                    llama_synchronize(ctx_tgt);
                    synchronized = true;
                }
                auto prefix = dst.task->tokens.clone_cached_prefix(image->frontier());
                vbr_swa_window_plan_request request;
                request.source_epoch = source.prompt.sequence_epoch;
                request.destination = dst.id;
                request.destination_epoch = ensure_frontier_sequence_epoch(dst.prompt);
                request.execution_identity = swa_window_execution_identity;
                request.representation = swa_window_representation();
                request.capacity_context = prompt_cache.get(); request.reserve = reserve_swa_staging;
                auto plan = vbr_prepare_swa_window(*ctx_tgt, image, request);
                if (plan.status != vbr_swa_window_status::ok) { continue; }
                vbr_swa_window_install_request install {request.source_epoch, request.destination_epoch, request.execution_identity};
                install.capacity_context = prompt_cache.get(); install.reserve = reserve_swa_staging;
                if (vbr_install_swa_window(*ctx_tgt, std::move(plan.plan), install) != vbr_swa_window_status::ok) { continue; }
                installed = true;
                installed_prefix.emplace(std::move(prefix));
                best = {&source, nullptr, image->frontier(), {}, image->retained_bytes()};
                break;
            }
            if (rec) {
                // Only the selected active prefix is recorded, not a complete
                // cost-planner inventory of every sibling/source considered here.
                if (best.source) {
                    rec->note_inventory_truncated(provider);
                } else {
                    rec->note_inventory_complete(provider);
                }
            }
            if (!best.source) {
                return;
            }

            const auto * cp = best.checkpoint;
            const auto n_tokens = best.n_tokens;
            const auto next_pos = dst.task->tokens.pos_next(n_tokens);
            row = rec ? rec->find_or_add(provider, best.source->id, uint8_t(0),
                                         dst.id, rec->selection) : nullptr;
            if (row) {
                row->lcp_tokens = llama_cache_acct_value::measured(n_tokens);
                row->payload_bytes = llama_cache_acct_value::measured(cp ? cp->size() : best.window_bytes);
                rec->select(provider, row);
            }
            auto prefix = installed_prefix ? std::move(*installed_prefix) : dst.task->tokens.clone_cached_prefix(n_tokens);
            // Media capability belongs to the slot, not this request. A text
            // completion can be followed by an image after erase/ID reuse.
            prefix.has_mtmd = dst.prompt.tokens.has_mtmd;
            std::string status = live_attention ? "shared active attention prefix" : "restored active context checkpoint";
            if (best.window_bytes) { status = "restored historical SWA window"; }
            if (!synchronized) { llama_synchronize(ctx_tgt); }
            if (draft_ctx) {
                llama_synchronize(draft_ctx);
            }
            const bool shared = installed || (live_attention
                ? (best.rows.empty() ? mem->try_share_live_prefix(best.source->id, dst.id, next_pos)
                                    : mem->try_share_live_prefix_rows(best.source->id, dst.id, next_pos, best.rows))
                : !best.rows.empty()
                    ? mem->try_share_attn_prefix_rows(best.source->id, dst.id,
                        cp->computation_frontier.next_position, best.rows)
                    : llama_memory_try_share_attn_prefix(mem, best.source->id, dst.id, n_tokens));
            if (!shared) {
                if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT); }
                return;
            }
            installed = true;
            const bool target_ok = live_attention || llama_state_seq_set_data_ext(ctx_tgt,
                    cp->data_tgt.data(), cp->data_tgt.size(), dst.id,
                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == cp->data_tgt.size();
            const bool draft_ok = target_ok && (!(mtp || shared_dflash) || cp->try_load_dft(
                    draft_ctx, dst.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            if (!draft_ok || llama_memory_seq_pos_max(mem, dst.id) != (cp ? cp->pos_max : next_pos - 1)) {
                throw std::runtime_error("active prefix checkpoint restore failed");
            }
            if (shared_dflash && llama_memory_seq_pos_max(llama_get_memory(draft_ctx), dst.id) != cp->pos_max) {
                throw std::runtime_error("active prefix draft frontier mismatch");
            }
            common_speculative_sequence_transition(dst.get_spec(), dst.id,
                mtp ? common_speculative_sequence_event::composite_image_restored
                    : shared_dflash ? common_speculative_sequence_event::draft_image_restored
                    : common_speculative_sequence_event::target_restored_without_draft);
            if (mtp && !common_speculative_set_state(dst.get_spec(), dst.id, cp->accel.spec.view())) {
                throw std::runtime_error("active prefix MTP state restore failed");
            }
            if (dflash && !common_speculative_ring_state_load(dst.get_spec(),
                    cp->accel.ring.view().data(), cp->accel.ring.size())) {
                throw std::runtime_error("active prefix DFlash ring restore failed");
            }
            // Publish only the input prefix, not source task/sampler/output or its
            // checkpoint ring. The next ordinary decode establishes A2's lineage.
            dst.prompt.tokens.swap(prefix);
            ensure_frontier_sequence_epoch(dst.prompt);
            dst.stats.n_prompt_cached = n_tokens;
            metrics.add_prompt_cached(n_tokens);
            dst.cache_status.swap(status);
            dst.stage_active_prefix_reuse(
                cp ? server_retention_instance_key::for_checkpoint(best.source->id, cp)
                   : server_retention_instance_key::for_slot(best.source->id), n_tokens);
            if (row) {
                rec->revoke_deliveries();
                row->accept();
                row->delivered = true;
            } else if (rec) {
                // Reuse succeeded, but a full observer cannot represent its provider.
                // Drop the record instead of attributing these tokens to the old live slot.
                if (cache_plan_obs) { cache_plan_obs->shadow_unavailable++; }
                dst.cache_plan.reset();
                rec = nullptr;
            }
            SLT_INF(dst, "active prefix restored: source_slot=%d source_task=%d tokens=%" PRId64
                    " replay=%zu restore_ms=%.3f\n", best.source->id, best.source->task->id,
                    n_tokens, dst.task->tokens.size() - size_t(n_tokens),
                    (ggml_time_us() - start) / 1000.0);
            return;
        } catch (const std::exception & e) {
            if (installed) {
                dst.mandatory_recovery_reset(server_cache_destruction_reason::restore_failure);
                dst.stats.n_prompt_cached = 0;
            }
            if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT); }
            if (rec && installed) { rec->restore_attempt_failed = true; }
            SLT_WRN(dst, "active prefix unavailable, using cold prefill: %s\n", e.what());
            return;
        }
    }


    // fork server-context.cpp:11170  (11 lines)
    void assert_scheduler_thread(
            std::thread::id & pinned,
            const char * what) {
        const auto current = std::this_thread::get_id();
        if (pinned == std::thread::id{}) {
            pinned = current;
        }
        if (pinned != current) {
            GGML_ABORT("%s\n", what);
        }
    }

    // fork 9l-d4b6z: cache-control refresh members (fork server-context.cpp 11182-11292)
    bool cache_control_refresh_prompt(
            const server_prompt & prompt,
            const std::string & adapter_identity,
            server_cache_lease_identity & identity,
            server_cache_lease_frontier & frontier) noexcept {
        const int64_t tokens = prompt.n_tokens();
        if (!server_cache_lease_build_identity(
                frontier_execution_identity, adapter_identity,
                prompt.tokens, tokens, identity)) {
            return false;
        }
        frontier = {
            prompt.sequence_epoch, uint64_t(tokens), tokens,
        };
        return frontier.valid();
    }

    bool cache_control_refresh_subject(
            const server_cache_control_selector & selector,
            server_cache_lease_identity & identity,
            server_cache_lease_frontier & frontier) noexcept {
        if (selector.kind == server_cache_control_subject_kind::live_prefix) {
            const auto slot = std::find_if(
                slots.begin(), slots.end(), [&](const server_slot & value) {
                    return value.id == selector.retention_key.owner_slot;
                });
            if (slot == slots.end()) {
                return false;
            }
            return cache_control_refresh_prompt(
                slot->prompt, lora_config_identity(slot->lora),
                identity, frontier);
        }
        if (selector.kind == server_cache_control_subject_kind::host_snapshot) {
            if (!prompt_cache) {
                return false;
            }
            const auto * wanted =
                reinterpret_cast<const server_prompt_cache_state *>(
                    selector.retention_key.instance);
            const auto state = std::find_if(
                prompt_cache->states.begin(), prompt_cache->states.end(),
                [&](const server_prompt_cache_state & value) {
                    return &value == wanted;
                });
            if (state == prompt_cache->states.end()) {
                return false;
            }
            return cache_control_refresh_prompt(
                state->prompt, state->adapter_config_key,
                identity, frontier);
        }
        return false;
    }

    server_cache_control_status cache_control_resolve_semantic_selector(
            server_cache_control_selector & selector,
            const server_task::cache_control_semantic_selector & semantic) noexcept {
        if (selector.kind == server_cache_control_subject_kind::vbr_reference ||
            selector.kind == server_cache_control_subject_kind::live_checkpoint) {
            return server_cache_control_status::ok;
        }
        if (selector.kind == server_cache_control_subject_kind::live_prefix) {
            const auto slot = std::find_if(
                slots.begin(), slots.end(), [&](const server_slot & value) {
                    return value.id == semantic.slot_id;
                });
            if (slot == slots.end() || slot->prompt.n_tokens() == 0) {
                return server_cache_control_status::not_found;
            }
            if (slot->is_processing()) {
                return server_cache_control_status::subject_busy;
            }
            selector.retention_key =
                server_retention_instance_key::for_slot(slot->id);
            return cache_control_refresh_prompt(
                slot->prompt, lora_config_identity(slot->lora),
                selector.identity, selector.frontier)
                    ? server_cache_control_status::ok
                    : server_cache_control_status::identity_unavailable;
        }
        if (selector.kind == server_cache_control_subject_kind::host_snapshot) {
            if (!prompt_cache || !semantic.tokens) {
                return server_cache_control_status::not_found;
            }
            const auto requested_lora = semantic.lora.empty()
                ? params_base.lora_adapters
                : construct_lora_list(semantic.lora);
            const std::string adapter_identity =
                lora_config_identity(requested_lora);
            const auto state = std::find_if(
                prompt_cache->states.begin(), prompt_cache->states.end(),
                [&](const server_prompt_cache_state & value) {
                    return value.adapter_config_key == adapter_identity &&
                        value.prompt.tokens.size() == semantic.tokens->size() &&
                        value.prompt.tokens.get_common_prefix(*semantic.tokens) ==
                            semantic.tokens->size();
                });
            if (state == prompt_cache->states.end()) {
                return server_cache_control_status::not_found;
            }
            selector.retention_key =
                server_retention_instance_key::for_host_entry(&*state);
            return cache_control_refresh_prompt(
                state->prompt, state->adapter_config_key,
                selector.identity, selector.frontier)
                    ? server_cache_control_status::ok
                    : server_cache_control_status::identity_unavailable;
        }
        return server_cache_control_status::not_supported;
    }

    // ---- group A file-local types (verbatim from the fork) ----

    // fork server-context.cpp:4219
    struct cache_plan_host_source_registry {
        explicit cache_plan_host_source_registry(bool publish) noexcept
            : publish_(publish) {}

        void begin(server_prompt_cache * cache) noexcept {
            if (publish_ && cache) {
                cache->cache_plan_begin_inventory();
                published_.clear();
                published_.reserve(cache->states.size());
                for (auto & state : cache->states) {
                    published_.emplace(
                        server_retention_instance_key::for_host_entry(
                            &state).instance,
                        &state);
                }
            }
        }

        bool get(
                server_prompt_cache & cache,
                server_prompt_cache_state & state,
                int32_t & source_id) {
            if (publish_) {
                return cache.cache_plan_get_source_id(state, source_id);
            }
            return local_.get_or_assign(
                server_retention_instance_key::for_host_entry(
                    &state).instance,
                source_id);
        }

        bool find(uintptr_t instance, int32_t & source_id) const noexcept {
            if (!publish_) {
                return local_.find(instance, source_id);
            }
            const auto found = published_.find(instance);
            if (found == published_.end() ||
                found->second->cache_plan_source_id < 0) {
                source_id = -1;
                return false;
            }
            source_id = found->second->cache_plan_source_id;
            return true;
        }

    private:
        bool publish_ = false;
        server_cache_plan_local_source_registry local_;
        std::unordered_map<uintptr_t, const server_prompt_cache_state *>
            published_;
    };

    // fork server-context.cpp:5565
    enum class vbr_recovery_requirement : uint8_t {
        full_frontier = 0,
        full_or_stem,
        none_for_admission_progress,
    };

    // fork server-context.cpp:5571
    struct vbr_live_retention_filter {
        int except_id = -1;
        int spec_tier = -1;
        const std::vector<uint8_t> * refused_by_seq = nullptr;
        vbr_recovery_requirement recovery =
            vbr_recovery_requirement::full_frontier;
    };

    // fork server-context.cpp:8889
    struct cache_plan_stage1_selection {
        server_slot * target = nullptr;
        bool update_cache = false;
        bool selection_deferred_busy = false;
        std::unique_ptr<common_cache_plan_record> record;
    };

    // fork server-context.cpp:9249
    struct cache_plan_stage1_inventory {
        std::vector<common_adapter_lora_info> incoming_loras;
        std::string incoming_adapter;
        bool incoming_adapter_ready = false;
        bool incoming_adapter_matches = false;
        bool host_lookup_enabled = false;
    };

    // ---- group A data members (verbatim declarations from the fork) ----

    // fork server-context.cpp:4271
    bool fixed_host_cache_enabled() const noexcept {
        return prompt_cache && !server_vbr_dynamic_active(params_base);
    }

    std::unique_ptr<server_cache_plan_observer> cache_plan_obs;

    // fork 9l-d3e: the authority configuration failure hook (fork server-context.cpp 3657-3662)
    void cache_authority_config_failed(bool mirror_to_shadow) noexcept {
        cache_authority->configured = false;
        if (mirror_to_shadow && cache_plan_obs) {
            cache_plan_obs->shadow_unavailable++;
        }
    }
    std::thread::id cache_plan_preflight_scheduler_thread;
    std::vector<server_live_retention_candidate> vbr_retention_candidates;
    std::vector<int32_t> vbr_retention_candidate_by_seq;
    std::vector<uint8_t> vbr_retention_seen_by_seq;
    // fork 9l-d4b3 (fork server-context.cpp 3546-3547)
    std::vector<server_anchor_parent_rank> vbr_idle_parent_values;
    std::vector<int32_t> vbr_idle_durability_order;
    std::vector<uint8_t> vbr_durable_sources_by_seq;
    std::vector<uint8_t> vbr_durable_stems_by_seq;

// fork 9l-d2b: reclaim refusal ledger (fork server-context.cpp 3553)
    std::vector<uint8_t> vbr_reclaim_refused_by_seq;
    std::vector<uint32_t> vbr_exclusive_cells_by_seq;
    size_t vbr_retention_candidate_count = 0;
    uint64_t vbr_retention_projection_epoch = 0;
    const std::vector<uint32_t> * vbr_exclusive_cells_for_test = nullptr;

// fork 9l-d2b: reclaim/pool-state test knobs (fork server-context.cpp 3557-3559)
    int64_t vbr_reclaim_cells_needed_for_test = -1;
    uint64_t vbr_reclaim_initial_cells_for_test = 0;
    size_t vbr_pool_state_index_for_test = 0;

    // fork 9l-d2d: artifact store + automatic-restore counters, on the class that
    // owns them (fork server-context.cpp 3562-3575). 9l-a had declared these in
    // server_slot, where nothing could resolve them; they moved here when 9l-d2d
    // landed their first reader.
    // Artifact machinery is a lifecycle-authority sibling: it references
    // the frozen ledger but is destroyed before it. It exists only for an
    // armed dynamic-VBR memory after the one-shot manifest and ring admission.
    std::unique_ptr<server_vbr_artifact_store> vbr_artifact_store;
    server_vbr_prompt_cache_support_status vbr_prompt_cache_support =
        server_vbr_prompt_cache_support_status::supported;
    server_vbr_artifact_capture_status vbr_prompt_cache_store_status =
        server_vbr_artifact_capture_status::unavailable;
    server_vbr_artifact_store_create_diagnostics
        vbr_prompt_cache_store_diagnostics;
    uint64_t vbr_automatic_restore_attempts = 0;
    uint64_t vbr_automatic_restore_succeeded = 0;
    uint64_t vbr_automatic_restore_quality_fallbacks = 0;
    uint64_t vbr_automatic_restore_occupied_attempts = 0;
    uint64_t vbr_automatic_restore_occupied_succeeded = 0;
    uint64_t vbr_automatic_restore_occupied_fallbacks = 0;

    // fork 9l-d4b3 (fork server-context.cpp 3576-3599)
    static constexpr uint64_t VBR_AUTOMATIC_EXACT_CAPTURE_MAX_BYTES =
        16ull*1024ull*1024ull*1024ull;
    static constexpr int64_t VBR_READINESS_SCAN_INTERVAL_MS = 1000;
    // Prompt-boundary sealing lets first-token sampling and unrelated slots
    // proceed, but the source slot's next KV-mutating decode waits for the
    // exact iSWA rows to seal. Bound that inter-token hold even before the
    // estimator has a model-specific sample.
    static constexpr uint64_t VBR_PROMPT_BOUNDARY_CAPTURE_DEADLINE_US =
        5000000;
    struct vbr_idle_capture_candidate {
        int32_t slot_id = -1;
        uint64_t manifest_id = 0;
        std::array<uint8_t, 32> attempt_identity = {};
        std::array<uint8_t, 32> full_source_identity = {};
        uint64_t tier_epoch = 0;
        uint64_t tier_epoch_swa = 0;
        bool refresh = false;
        bool stem_requested = false;
        bool stemmed = false;
        bool prepressure = false;
        bool active_prompt_frontier = false;
        uint64_t selected_tokens = 0;
        server_prompt_cache_vbr_publication_metadata publication;
    };

    // fork 9l-d4b3 (fork server-context.cpp 3601-3629)
    uint64_t vbr_automatic_exact_capture_max_bytes() const noexcept {
        if (params_base.cache_ram_mib < 0) {
            return VBR_AUTOMATIC_EXACT_CAPTURE_MAX_BYTES;
        }
        const uint64_t configured =
            uint64_t(params_base.cache_ram_mib)*1024ull*1024ull;
        return std::min(configured, VBR_AUTOMATIC_EXACT_CAPTURE_MAX_BYTES);
    }
    struct vbr_idle_exact_capture_work {
        server_queue::idle_capture_session session;
        server_vbr_explicit_host_capture capture;
        server_vbr_artifact_capture_output transfer;
        vbr_idle_capture_candidate candidate;
        server_prompt_cache_vbr_capacity_claim capacity;
        server_vbr_capture_readiness_reservation readiness;
        std::thread worker;
        std::atomic<bool> complete { false };
        int64_t started_us = 0;
        int64_t transfer_started_us = 0;
        int64_t transfer_deadline_us = 0;

        ~vbr_idle_exact_capture_work() {
            if (worker.joinable()) {
                worker.join();
            }
        }
    };
    std::unique_ptr<vbr_idle_exact_capture_work>
        vbr_idle_exact_capture;

    // fork 9l-d4b3 (fork server-context.cpp 3632-3642)
    uint64_t vbr_capture_readiness_generation = 0;
    // Decode can cross hundreds of ubatch boundaries while the durable/live
    // frontier inventory is unchanged.  Bound the expensive catalog/retention
    // classification cadence; the ordinary idle callback and worker
    // completion wake remain immediate mutation-driven terminals.
    int64_t vbr_capture_readiness_next_scan_ms = 0;
    // SWA/iSWA cannot reconstruct an earlier prompt boundary after sliding
    // rows have been recycled.  While this is nonnegative, the exact-capture
    // selector is restricted to the named DONE_PROMPT slot so it can seal the
    // frontier before the first generated token mutates that state.
    int32_t vbr_prompt_boundary_capture_slot = -1;

    // fork 9l-d4b4 (fork server-context.cpp 11408-11458, 11605-11635)
    static bool vbr_idle_capture_attempt_suppressed(
            const std::array<uint8_t, 32> & recorded,
            const std::array<uint8_t, 32> & current,
            bool terminal,
            int64_t retry_after_ms,
            int64_t now_ms) noexcept {
        return recorded == current &&
            (terminal || now_ms < retry_after_ms);
    }
    static bool vbr_idle_admission_refusal_suppressed(
            const server_vbr_idle_refusal_witness & recorded,
            const std::array<uint8_t, 32> & current_attempt,
            uint64_t accounting_serial,
            uint64_t retention_epoch,
            const server_cache_lease_retry_witness & lease) noexcept {
        if (!recorded.valid ||
            recorded.attempt_identity != current_attempt ||
            recorded.accounting_serial == 0 ||
            recorded.accounting_serial != accounting_serial ||
            recorded.retention_epoch == 0 ||
            recorded.retention_epoch != retention_epoch) {
            return false;
        }
        if (!recorded.lease_tracked) {
            return true;
        }
        return lease.available &&
            lease.event_ordinal == recorded.lease_event_ordinal &&
            (recorded.lease_next_expiry_ns == 0 ||
             lease.now_ns < recorded.lease_next_expiry_ns);
    }
    static bool vbr_idle_admission_refusal_suppressed(
            const std::vector<server_vbr_idle_refusal_witness> & recorded,
            const std::array<uint8_t, 32> & current_attempt,
            uint64_t accounting_serial,
            uint64_t retention_epoch,
            const server_cache_lease_retry_witness & lease) noexcept {
        const auto found = std::lower_bound(
            recorded.begin(), recorded.end(), current_attempt,
            [](const server_vbr_idle_refusal_witness & value,
               const std::array<uint8_t, 32> & attempt) {
                return value.attempt_identity < attempt;
            });
        return found != recorded.end() &&
            found->attempt_identity == current_attempt &&
            vbr_idle_admission_refusal_suppressed(
                *found, current_attempt, accounting_serial,
                retention_epoch, lease);
    }

// fork 9l-d4b6r: vbr_idle_retry_immediately (fork server-context.cpp 11589-11603)
    static bool vbr_idle_retry_immediately(
            bool terminal,
            bool aggregate_over_cap,
            server_prompt_cache_vbr_capacity_status capacity_status,
            size_t candidate_count,
            bool transfer_retry_candidate,
            bool capacity_retry_candidate) noexcept {
        const bool transfer_retry = aggregate_over_cap &&
            transfer_retry_candidate;
        const bool capacity_retry =
            capacity_status == server_prompt_cache_vbr_capacity_status::
                pressure_batch_unsupported && capacity_retry_candidate;
        return !terminal && candidate_count > 1 &&
            (transfer_retry || capacity_retry);
    }

    static bool vbr_idle_release_isolated_retry(
            size_t candidate_count,
            size_t successful_captures,
            bool isolated_retry,
            bool terminal,
            bool idle,
            bool processing) noexcept {
        return candidate_count == 1 && successful_captures != 0 &&
            isolated_retry && !terminal && idle && !processing;
    }

    static uint64_t vbr_idle_capacity_retry_manifest(
            const uint64_t * ranked,
            size_t ranked_count,
            const uint64_t * admitted,
            size_t admitted_count) noexcept {
        if (!ranked || !admitted) {
            return 0;
        }
        for (size_t i = 0; i < ranked_count; ++i) {
            if (ranked[i] == 0) {
                continue;
            }
            for (size_t j = 0; j < admitted_count; ++j) {
                if (ranked[i] == admitted[j]) {
                    return ranked[i];
                }
            }
        }
        return 0;
    }

    // fork 9l-d4b6a (fork server-context.cpp 11460-11506, 11508-11548): the idle-admission-refusal recorder
    static bool vbr_store_idle_admission_refusal(
            std::vector<server_vbr_idle_refusal_witness> & refusals,
            size_t refusal_limit,
            const server_vbr_idle_refusal_witness & witness,
            const server_cache_lease_retry_witness & current_lease) noexcept {
        if (!witness.valid || refusal_limit == 0) {
            return false;
        }
        auto existing = std::lower_bound(
            refusals.begin(), refusals.end(), witness.attempt_identity,
            [](const server_vbr_idle_refusal_witness & value,
               const std::array<uint8_t, 32> & attempt) {
                return value.attempt_identity < attempt;
            });
        if (existing != refusals.end() &&
            existing->attempt_identity == witness.attempt_identity) {
            *existing = witness;
            return true;
        }
        if (refusals.size() >= refusal_limit) {
            existing = std::find_if(
                refusals.begin(), refusals.end(),
                [&](const server_vbr_idle_refusal_witness & value) {
                    return !vbr_idle_admission_refusal_suppressed(
                        value, value.attempt_identity,
                        witness.accounting_serial,
                        witness.retention_epoch, current_lease);
                });
            if (existing == refusals.end()) {
                existing = refusals.begin();
            }
            *existing = witness;
            std::sort(
                refusals.begin(), refusals.end(),
                [](const server_vbr_idle_refusal_witness & a,
                   const server_vbr_idle_refusal_witness & b) {
                    return a.attempt_identity < b.attempt_identity;
                });
            return true;
        }
        try {
            refusals.insert(existing, witness);
        } catch (...) {
            return false;
        }
        return true;
    }

    bool vbr_record_idle_admission_refusal(
            server_slot & slot,
            const std::array<uint8_t, 32> & attempt) noexcept {
        auto * retention = live_retention_owner();
        if (!cache_authority || !retention) {
            slot.vbr_idle_admission_refusals.clear();
            return false;
        }
        server_vbr_idle_refusal_witness witness;
        witness.attempt_identity = attempt;
        witness.accounting_serial = cache_authority->ledger.serial();
        witness.retention_epoch = retention->competition_epoch_value();
        witness.lease_tracked = slot.lease_obs != nullptr;
        if (witness.lease_tracked) {
            const auto lease = slot.lease_obs->retry_witness(true);
            if (!lease.available) {
                slot.vbr_idle_admission_refusals.clear();
                return false;
            }
            witness.lease_event_ordinal = lease.event_ordinal;
            witness.lease_next_expiry_ns = lease.next_expiry_ns;
        }
        witness.valid = witness.accounting_serial != 0 &&
            witness.retention_epoch != 0;
        if (!witness.valid) {
            return false;
        }

        // The checkpoint ring is itself bounded. Keep enough witnesses for
        // every currently sealed frontier plus the full-source/stem attempt,
        // while preferentially recycling evidence made stale by a currency
        // change.
        const size_t refusal_limit = std::max<size_t>(
            2, slot.prompt.checkpoints.size() + 2);
        const auto current_lease = slot.lease_obs
            ? slot.lease_obs->retry_witness(false)
            : server_cache_lease_retry_witness {};
        return vbr_store_idle_admission_refusal(
            slot.vbr_idle_admission_refusals, refusal_limit,
            witness, current_lease);
    }

    // fork 9l-d4b4 (fork server-context.cpp 12100-12171)
    server_vbr_capture_readiness_input vbr_capture_readiness_input_for(
            const server_slot & source,
            const vbr_explicit_capture_pretransfer_quote & quote,
            bool reserve_decode_runway) const
            noexcept {
        server_vbr_capture_readiness_input input;
        input.source_slot = source.id;
        input.candidate_transfer_bytes = quote.planned_packed_bytes;
        input.candidate_host_bytes =
            quote.conservative_host_resident_bytes;
        input.candidate_metadata_bytes = quote.metadata_bytes;
        input.publication_margin_us = 5000;
        // The prompt-cache capacity claim is the canonical host reservation
        // and may cite bounded victims. Do not second-guess that transaction
        // with a scalar current-size comparison here.
        input.host_capacity_bytes = 0;
        input.host_committed_bytes = prompt_cache ? prompt_cache->size() : 0;
        input.pinned_lane_slots_available = vbr_idle_exact_capture ? 0 : 1;
        input.queue_slots_available = vbr_idle_exact_capture ? 0 : 1;
        // Usage below is global across the unified memory tree, so capacity
        // must come from that same global authority rather than the selected
        // sequence's per-slot context limit.
        uint64_t admitted_logical_growth = 0;
        for (const auto & slot : slots) {
            const uint64_t live = uint64_t(std::max<int64_t>(
                0, slot.prompt.n_tokens()));
            if (slot.is_processing() && slot.task) {
                const uint64_t target = uint64_t(std::max<int32_t>(
                    0, slot.task->n_tokens()));
                const uint64_t growth = target > live ? target-live : 0;
                if (admitted_logical_growth <= UINT64_MAX-growth) {
                    admitted_logical_growth += growth;
                } else {
                    admitted_logical_growth = UINT64_MAX;
                }
            }
        }
        uint64_t projected_cells = 0;
        auto * memory = ctx_tgt ? llama_get_memory(ctx_tgt) : nullptr;
        if (!memory || !memory->vbr_capture_readiness_cells(
                admitted_logical_growth,
                input.device_committed_cells,
                projected_cells,
                input.device_capacity_cells) ||
            projected_cells < input.device_committed_cells) {
            input.device_capacity_cells = 0;
        } else {
            input.admitted_growth_cells =
                projected_cells-input.device_committed_cells;
        }
        // Decode-driven pre-pressure capture must leave enough device runway
        // for the active request to keep advancing while the background D2H
        // completes. A normal idle capture is cancelled before a new request
        // mutates the source, so reserving that same runway would permanently
        // reject every useful frontier in the upper half of a context.
        input.emergency_cells = reserve_decode_runway
            ? std::min<uint64_t>(
                SERVER_PROMPT_CACHE_MIN_RETENTION_REUSE_TOKENS,
                input.device_capacity_cells)
            : 0;
        input.conservative_bandwidth_bytes_per_second =
            vbr_capture_readiness_estimator
                ? vbr_capture_readiness_estimator->
                    conservative_bandwidth_bytes_per_second()
                : 0;
        input.conservative_growth_cells_per_second =
            reserve_decode_runway && vbr_capture_readiness_estimator
                ? vbr_capture_readiness_estimator->
                    conservative_growth_cells_per_second()
                : 0;
        return input;
    }

// fork 9l-d4b6r: vbr_capture_reserved_cells (fork server-context.cpp 12173-12179)
    uint32_t vbr_capture_reserved_cells() const noexcept {
        if (!vbr_idle_exact_capture || !vbr_idle_exact_capture->readiness) {
            return 0;
        }
        return uint32_t(std::min<uint64_t>(
            vbr_idle_exact_capture->readiness.device_cells, UINT32_MAX));
    }

    // fork 9l-d4b6u: finish_idle_exact_vbr_capture (fork server-context.cpp 12181-12546)
    size_t finish_idle_exact_vbr_capture(
            bool allow_publish,
            bool task_arrival = false) noexcept {
        auto * pending = vbr_idle_exact_capture.get();
        if (!pending || (allow_publish && !task_arrival &&
                !pending->complete.load(std::memory_order_acquire))) {
            return 0;
        }
        if (pending->worker.joinable()) {
            pending->worker.join();
        }
        const auto trace_terminal = [&](const char * event,
                                        const char * reason) noexcept {
            if (!params_base.cache_debug) {
                return;
            }
            const auto & identity = pending->candidate.attempt_identity;
            SRV_INF(
                "VBR_CAPTURE_TRACE event=%s reason=%s slot=%d manifest=%" PRIu64
                " attempt=%02x%02x%02x%02x%02x%02x%02x%02x "
                "allow_publish=%d task_arrival=%d session_current=%d "
                "transfer_status=%s library_status=%s phase=%s "
                "planned=%" PRIu64 "\n",
                event, reason, pending->candidate.slot_id,
                pending->candidate.manifest_id,
                unsigned(identity[0]), unsigned(identity[1]),
                unsigned(identity[2]), unsigned(identity[3]),
                unsigned(identity[4]), unsigned(identity[5]),
                unsigned(identity[6]), unsigned(identity[7]),
                int(allow_publish), int(task_arrival),
                int(pending->session.continue_capture()),
                server_vbr_artifact_capture_status_name(
                    pending->transfer.status),
                vbr_explicit_capture_status_name(
                    pending->transfer.library_status),
                vbr_explicit_capture_phase_name(pending->transfer.phase),
                pending->transfer.pretransfer.planned_packed_bytes);
        };
        const auto reset_pending = [&](size_t result) noexcept {
            const int32_t source_id = pending->candidate.slot_id;
            if (pending->candidate.active_prompt_frontier &&
                source_id >= 0 && size_t(source_id) < slots.size() &&
                slots[size_t(source_id)].id == source_id) {
                auto & source = slots[size_t(source_id)];
                if (source.is_processing() &&
                    source.state == SLOT_STATE_WAIT_VBR_CAPTURE) {
                    source.state = SLOT_STATE_GENERATING;
                }
            }
            vbr_idle_exact_capture.reset();
            return result;
        };
        if (vbr_capture_readiness_estimator &&
            pending->transfer.status ==
                server_vbr_artifact_capture_status::ok &&
            pending->transfer_started_us > 0 &&
            pending->transfer.pretransfer.planned_packed_bytes != 0) {
            const int64_t elapsed =
                ggml_time_us()-pending->transfer_started_us;
            if (elapsed > 0) {
                vbr_capture_readiness_estimator->observe_transfer(
                    pending->transfer.pretransfer.planned_packed_bytes,
                    uint64_t(elapsed));
            }
        }
        auto mark_retry = [&](bool terminal = false) {
            const int32_t id = pending->candidate.slot_id;
            if (id < 0 || size_t(id) >= slots.size() ||
                slots[size_t(id)].id != id) {
                return;
            }
            auto & source = slots[size_t(id)];
            source.vbr_idle_capture_attempt_identity =
                pending->candidate.attempt_identity;
            source.vbr_idle_capture_terminal = terminal;
            source.vbr_idle_capture_retry_after_ms =
                terminal ? 0 : ggml_time_ms() + 5000;
        };
        if (!allow_publish) {
            trace_terminal("abandon", "publication_disabled");
            mark_retry();
            return reset_pending(0);
        }
        if ((!task_arrival && !pending->session.continue_capture()) ||
            pending->transfer.status !=
                server_vbr_artifact_capture_status::ok ||
            !pending->capture.ready_for_publication()) {
            SRV_DBG(
                "VBR_IDLE_CAPTURE exact-background transfer refused "
                "status=%s library_status=%s phase=%s generation=%s "
                "companion=%u kind=%u\n",
                server_vbr_artifact_capture_status_name(
                    pending->transfer.status),
                vbr_explicit_capture_status_name(
                    pending->transfer.library_status),
                vbr_explicit_capture_phase_name(pending->transfer.phase),
                vbr_explicit_generation_failure_name(
                    pending->transfer.generation_failure),
                pending->transfer.companion_failure_index,
                unsigned(pending->transfer.companion_failure_kind));
            trace_terminal("refused", "transfer_not_publishable");
            mark_retry(
                pending->transfer.library_status ==
                    vbr_explicit_capture_status::unsupported_layout ||
                pending->transfer.library_status ==
                    vbr_explicit_capture_status::not_armed);
            return reset_pending(0);
        }

        std::shared_ptr<const server_prompt_cache_vbr_payload> payload;
        const auto captured = vbr_artifact_store->publish_host_payload(
            pending->capture, payload);
        if (captured.status != server_vbr_artifact_capture_status::ok ||
            !payload ||
            (!task_arrival && !pending->session.continue_capture())) {
            trace_terminal("refused", "store_publish_failed");
            mark_retry();
            return reset_pending(0);
        }
        if (!pending->candidate.refresh &&
            !pending->capacity.requires_publication_revalidation() &&
            !prompt_cache->consume_vbr_publication_capacity(
                pending->capacity)) {
            trace_terminal("refused", "capacity_commit_failed");
            mark_retry();
            return reset_pending(0);
        }
        const int32_t id = pending->candidate.slot_id;
        if (id < 0 || size_t(id) >= slots.size() ||
            slots[size_t(id)].id != id) {
            trace_terminal("refused", "source_slot_missing");
            return reset_pending(0);
        }
        auto & source = slots[size_t(id)];
        const auto representation =
            llama_get_memory(ctx_tgt)->vbr_representation_identity();
        const bool representation_stable =
            representation.tier_epoch == pending->candidate.tier_epoch &&
            representation.tier_epoch_swa ==
                pending->candidate.tier_epoch_swa;
        const bool source_state_stable =
            pending->candidate.active_prompt_frontier
                ? source.is_processing() &&
                    source.state == SLOT_STATE_WAIT_VBR_CAPTURE
                : !source.is_processing() &&
                    source.state == SLOT_STATE_IDLE;
        if (!representation_stable || !source_state_stable ||
            (!task_arrival && !pending->session.continue_capture())) {
            trace_terminal("refused", "source_currency_changed");
            mark_retry();
            return reset_pending(0);
        }
        const auto adapter_key =
            payload->package().manifest().identity.adapter_config_identity;
        std::array<uint8_t, 32> token_digest = {};
        const uint64_t published_tokens =
            payload->package().manifest().identity.token_count > 0
                ? uint64_t(payload->package().manifest().identity.token_count)
                : 0;
        const bool token_digest_ok = pending->candidate.stemmed
            ? published_tokens == pending->candidate.selected_tokens &&
                published_tokens <= uint64_t(source.prompt.n_tokens()) &&
                source.prompt.tokens.retention_token_prefix_digest(
                    size_t(published_tokens), token_digest)
            : source.prompt.tokens.retention_token_digest(token_digest);
        if (!token_digest_ok) {
            trace_terminal("refused", "token_digest_changed");
            mark_retry();
            return reset_pending(0);
        }
        const auto publication_identity = vbr_idle_publication_identity(
            token_digest, pending->candidate.tier_epoch,
            pending->candidate.tier_epoch_swa);
        if (pending->candidate.refresh) {
            const auto refresh = prompt_cache->refresh_vbr_compact(
                source.prompt, std::move(payload),
                frontier_execution_identity, adapter_key, source.id);
            if (refresh != server_prompt_cache_vbr_refresh_status::
                    updated_with_anchor &&
                refresh != server_prompt_cache_vbr_refresh_status::
                    updated_compact_only &&
                refresh != server_prompt_cache_vbr_refresh_status::unchanged) {
                trace_terminal("refused", "refresh_publish_failed");
                mark_retry();
                return reset_pending(0);
            }
            source.vbr_idle_capture_published_identity =
                publication_identity;
            source.vbr_idle_capture_representation_valid = true;
            source.vbr_idle_capture_attempt_identity =
                pending->candidate.attempt_identity;
            // Publication seals this immutable attempt. If retention later
            // evicts it, that is a cache-policy verdict rather than a capture
            // failure: do not fight the eviction by repeating the same D2H.
            // Token, sequence, or representation changes produce a different
            // attempt identity and reopen capture immediately.
            source.vbr_idle_capture_terminal = true;
            source.vbr_idle_capture_retry_after_ms = 0;
        } else {
            server_prompt_cache::iterator published_host =
                prompt_cache->states.end();
            if (!pending->candidate.publication.ready() ||
                !prompt_cache->publish_vbr(
                    pending->candidate.publication,
                    server_prompt_cache_payload::from_vbr(
                        std::move(payload)),
                    source.cache_family,
                    !source.task || !source.task->is_child(),
                    &published_host,
                    pending->capacity.requires_publication_revalidation()
                        ? &pending->capacity : nullptr)) {
                trace_terminal("refused", "prompt_cache_publish_failed");
                mark_retry();
                return reset_pending(0);
            }
            if (pending->candidate.stemmed) {
                source.vbr_idle_stem_source_identity =
                    pending->candidate.full_source_identity;
                source.vbr_idle_stem_coverage_tokens = published_tokens;
                source.vbr_idle_stem_source_valid = true;
                source.vbr_idle_stem_retry = false;
                source.vbr_idle_stem_host_artifact =
                    prompt_cache->vbr_host_artifact_id(published_host);
                if (source.vbr_idle_stem_host_artifact.v == 0) {
                    trace_terminal("refused", "stem_artifact_missing");
                    mark_retry();
                    return reset_pending(0);
                }
                source.vbr_idle_capture_attempt_identity =
                    pending->candidate.attempt_identity;
                source.vbr_idle_capture_terminal = true;
                source.vbr_idle_capture_retry_after_ms = 0;
                if (source.vbr_reuse_capture_frontier == published_tokens) {
                    source.vbr_reuse_capture_frontier = 0;
                }
            } else {
                source.vbr_idle_capture_published_identity =
                    publication_identity;
                source.vbr_idle_capture_representation_valid = true;
                source.vbr_idle_capture_attempt_identity =
                    pending->candidate.attempt_identity;
                source.vbr_idle_capture_terminal = true;
                source.vbr_idle_capture_retry_after_ms = 0;
                source.vbr_idle_stem_retry = false;
                source.vbr_idle_stem_source_valid = false;
                source.vbr_idle_stem_host_artifact = {};
            }
        }

        size_t displaced = 0;
        const bool coordinated = n_swa > 0 ||
            llama_model_is_recurrent(model_tgt) ||
            llama_model_is_hybrid(model_tgt);
        if (coordinated && !pending->candidate.prepressure &&
            !pending->candidate.stemmed) {
            std::vector<server_slot *> reclaim;
            bool all_durable = true;
            try {
                reclaim.reserve(slots.size());
                size_t query_count = 0;
                for (auto & live : slots) {
                    if (live.prompt.n_tokens() == 0) {
                        continue;
                    }
                    const auto admission = live.observe_full_slot(
                        server_cache_destruction_class::slot_drop,
                        server_cache_destruction_reason::idle_reclaim);
                    const bool safe = !live.is_processing() &&
                        live.state == SLOT_STATE_IDLE &&
                        automatic_vbr_cache_support(
                            live.prompt.tokens, live.lora,
                            live.can_speculate()) ==
                            server_vbr_prompt_cache_support_status::supported &&
                        !live.hard_lease_blocks_live_prefix() &&
                        !queue_tasks.has_deferred_for_slot(live.id) &&
                        admission.verdict !=
                            server_cache_destruction_verdict::
                                would_refuse_hard_leased &&
                        admission.verdict !=
                            server_cache_destruction_verdict::unavailable;
                    if (!safe) {
                        all_durable = false;
                        break;
                    }
                    if (query_count >= vbr_idle_frontier_queries.size()) {
                        all_durable = false;
                        break;
                    }
                    auto & query =
                        vbr_idle_frontier_queries[query_count++];
                    query = {};
                    query.slot_id = live.id;
                    query.prompt = &live.prompt;
                    server_vbr_artifact_capture_status identity_status;
                    if (!build_capture_identity_fields(
                            live, query.identity, identity_status)) {
                        all_durable = false;
                        break;
                    }
                    reclaim.push_back(&live);
                }
                if (all_durable &&
                    (!prompt_cache->mark_vbr_frontiers(
                         vbr_idle_frontier_queries.data(), query_count) ||
                     query_count != reclaim.size())) {
                    all_durable = false;
                }
                if (all_durable) {
                    for (size_t i = 0; i < query_count; ++i) {
                        if (!vbr_idle_frontier_queries[i].durable) {
                            all_durable = false;
                            break;
                        }
                    }
                }
            } catch (...) {
                all_durable = false;
            }
            // A task may arrive while the coordinated full-frontier worker is
            // completing. finish() already joins and publishes that exact
            // artifact before slot selection; once the batched proof says
            // every live source is durable, clearing here is both recoverable
            // and necessary for SWA/iSWA to restore an earlier prompt-boundary
            // artifact (those memories cannot trim the live suffix in place).
            const bool clear_authorized = task_arrival ||
                pending->session.continue_capture();
            if (all_durable && !reclaim.empty() && clear_authorized) {
                auto * memory = llama_get_memory(ctx_tgt);
                llama_memory_clear(memory, false);
                llama_memory_breathe(memory);
                for (auto * live : reclaim) {
                    const bool cleared =
                        live->prompt_clear_after_vbr_publication();
                    GGML_ASSERT(cleared);
                    displaced += cleared ? 1 : 0;
                }
            }
        }
        if (params_base.cache_debug) {
            const auto & identity = pending->candidate.attempt_identity;
            SRV_INF(
                "VBR_CAPTURE_TRACE event=published reason=ok slot=%d "
                "manifest=%" PRIu64
                " attempt=%02x%02x%02x%02x%02x%02x%02x%02x "
                "reference=%s planned=%" PRIu64 " payload=%" PRIu64
                " stash=%" PRIu64 " companion=%" PRIu64
                " chunks=%" PRIu64 " displaced=%zu duration_ms=%.3f\n",
                pending->candidate.slot_id,
                pending->candidate.manifest_id,
                unsigned(identity[0]), unsigned(identity[1]),
                unsigned(identity[2]), unsigned(identity[3]),
                unsigned(identity[4]), unsigned(identity[5]),
                unsigned(identity[6]), unsigned(identity[7]),
                captured.reference.c_str(),
                captured.pretransfer.planned_packed_bytes,
                captured.payload_bytes, captured.stash_bytes,
                captured.companion_bytes, captured.chunks, displaced,
                (ggml_time_us() - pending->started_us)/1000.0);
        }
        SRV_INF(
            "VBR_IDLE_CAPTURE exact-background status=ok bytes=%" PRIu64
            " displaced=%zu duration_ms=%.3f\n",
            captured.pretransfer.planned_packed_bytes, displaced,
            (ggml_time_us() - pending->started_us)/1000.0);
        return reset_pending(1);
    }

    // fork 9l-d4b6v: publish_idle_vbr_batch (fork server-context.cpp 12548-14368)
    size_t publish_idle_vbr_batch(
            server_queue::idle_capture_session & capture_session,
            bool readiness_only = false) noexcept {
        if (!params_base.vbr_prompt_cache || !prompt_cache ||
            !vbr_artifact_store ||
            !capture_session.continue_capture()) {
            return 0;
        }

        try {

        struct candidate {
            server_slot * slot = nullptr;
            const common_prompt_checkpoint * checkpoint = nullptr;
            uint64_t manifest_id = 0;
            std::array<uint8_t, 32> attempt_identity = {};
            std::array<uint8_t, 32> full_source_identity = {};
            uint64_t tier_epoch = 0;
            uint64_t tier_epoch_swa = 0;
            bool refresh = false;
            bool stem_requested = false;
            bool stemmed = false;
            uint64_t selected_tokens = 0;
            server_prompt_cache_vbr_publication_metadata publication;
        };
        std::vector<candidate> candidates;
        std::vector<vbr_projected_capture_manifest_request> manifests;
        size_t displaced_existing = 0;
        size_t displacement_attempts = 0;
        size_t capture_attempts = 0;
        bool refresh_selected = false;

        uint64_t total_token_references = 0;
        static constexpr uint64_t MAX_IDLE_TOKEN_REFERENCES = 1024ull*1024;
        auto * memory = llama_get_memory(ctx_tgt);
        if (!memory) {
            return 0;
        }
        const bool requires_coordinated_tree_clear = n_swa > 0 ||
            llama_model_is_recurrent(model_tgt) ||
            llama_model_is_hybrid(model_tgt);
        const bool may_displace_live_source =
            server_vbr_live_source_displacement_allowed(
                params_base.kv_unified, slots.size());
        bool projected_attention_layout_supported = false;
        {
            std::vector<llama_memory_tree_child> tree;
            if (llama_memory_tree_collect(memory, tree)) {
                bool has_attention = false;
                bool supported = true;
                for (const auto & node : tree) {
                    if (node.recurrent != nullptr) {
                        continue;
                    }
                    if (node.attention == nullptr ||
                        node.dependency_mode !=
                            checkpoint_child_dependency_mode::live_guarded) {
                        supported = false;
                        break;
                    }
                    has_attention = true;
                }
                projected_attention_layout_supported =
                    supported && has_attention;
            }
        }
        const auto representation =
            memory->vbr_representation_identity();
        const auto slot_by_id = [&](int32_t id) -> server_slot * {
            return id >= 0 && size_t(id) < slots.size() &&
                    slots[size_t(id)].id == id
                ? &slots[size_t(id)] : nullptr;
        };
        const auto base_eligible = [&](const server_slot & idle) {
            const bool prompt_boundary =
                vbr_prompt_boundary_capture_slot >= 0 &&
                idle.id == vbr_prompt_boundary_capture_slot;
            if (vbr_prompt_boundary_capture_slot >= 0 && !prompt_boundary) {
                return false;
            }
            const bool source_state_ready = prompt_boundary
                ? idle.is_processing() && idle.state == SLOT_STATE_DONE_PROMPT
                : !idle.is_processing() && idle.state == SLOT_STATE_IDLE;
            return source_state_ready &&
                idle.prompt.n_tokens() > 0 &&
                automatic_vbr_cache_support(
                    idle.prompt.tokens, idle.lora,
                    idle.can_speculate()) ==
                    server_vbr_prompt_cache_support_status::supported &&
                !idle.hard_lease_blocks_live_prefix() &&
                prompt_cache->vbr_retention_source_available(idle.id);
        };

        // A host-only cache is a normal steady state after successful
        // displacement. Avoid walking its complete retention catalog on every
        // idle tick when no live source can produce capture work.
        bool any_live_source = false;
        for (auto & idle : slots) {
            if (!capture_session.continue_capture()) {
                return 0;
            }
            const auto support = automatic_vbr_cache_support(
                idle.prompt.tokens, idle.lora,
                idle.can_speculate());
            if (support != idle.vbr_idle_support_status) {
                idle.vbr_idle_support_status = support;
                if (support !=
                        server_vbr_prompt_cache_support_status::supported) {
                    SLT_DBG(
                        idle,
                        "VBR_IDLE_CAPTURE fallback=live_only reason=%s\n",
                        server_vbr_prompt_cache_support_status_name(
                            support));
                }
            }
            if (base_eligible(idle)) {
                any_live_source = true;
            }
        }
        if (!any_live_source) {
            return 0;
        }

        // Decode-driven readiness is allowed to sample frequently, but a
        // stable already-published frontier or a cooling-down refusal must
        // not pay retention ranking and a host-catalog pass again. Token and
        // media digests are cached by server_tokens; this is one scalar slot
        // walk and leaves the ordinary idle/mutation terminal unchanged.
        if (readiness_only) {
            bool work_due = false;
            const int64_t now_ms = ggml_time_ms();
            for (auto & idle : slots) {
                if (!base_eligible(idle)) {
                    continue;
                }
                std::array<uint8_t, 32> token_digest = {};
                vbr_artifact_identity_block identity;
                server_vbr_artifact_capture_status identity_status;
                if (!idle.prompt.tokens.retention_token_digest(
                        token_digest) ||
                    !build_capture_identity_fields(
                        idle, identity, identity_status)) {
                    continue;
                }
                const auto publication = vbr_idle_publication_identity(
                    token_digest, representation.tier_epoch,
                    representation.tier_epoch_swa);
                if (idle.vbr_idle_capture_representation_valid &&
                    idle.vbr_idle_capture_published_identity == publication) {
                    continue;
                }
                const auto attempt = vbr_idle_capture_attempt_digest(
                    identity, token_digest, representation.tier_epoch,
                    representation.tier_epoch_swa);
                if (vbr_idle_capture_attempt_suppressed(
                        idle.vbr_idle_capture_attempt_identity, attempt,
                        idle.vbr_idle_capture_terminal,
                        idle.vbr_idle_capture_retry_after_ms, now_ms)) {
                    continue;
                }
                work_due = true;
                break;
            }
            if (!work_due) {
                return 0;
            }
        }
        try {
            candidates.reserve(VBR_PROJECTED_CAPTURE_MAX_MANIFESTS);
            manifests.reserve(VBR_PROJECTED_CAPTURE_MAX_MANIFESTS);
        } catch (...) {
            return 0;
        }

        // The bounded wave is a retention decision, not a slot-number
        // lottery. Snapshot the current competition epoch without aging it,
        // then rank all eligible live parents from coldest to hottest. The
        // reverse order feeds scarce capture bandwidth; the forward order is
        // retained for cheap displacement of already-durable sources after
        // capture completes. No token vectors are copied until selection.
        const vbr_live_retention_filter retention_filter { -1, -1, nullptr };
        bool ranked = vbr_prepare_retention_wave(
            retention_filter, false, false);
        if (ranked) {
            for (size_t i = 0; i < vbr_retention_candidate_count; ++i) {
                auto & value = vbr_retention_candidates[i];
                const auto * slot = slot_by_id(value.slot_id);
                value.eligible = slot && base_eligible(*slot);
            }
            ranked = server_anchor_parent_values_prepared(
                vbr_retention_candidates.data(),
                vbr_retention_candidate_count,
                vbr_retention_projection_epoch,
                vbr_idle_parent_values);
        }
        size_t durability_order_count = 0;
        if (ranked) {
            ranked = server_idle_durability_order(
                vbr_idle_parent_values,
                vbr_idle_durability_order.data(),
                vbr_idle_durability_order.size(),
                durability_order_count);
            if (ranked) {
                for (size_t i = 0; i < durability_order_count; ++i) {
                    if (!slot_by_id(vbr_idle_durability_order[i])) {
                        ranked = false;
                        break;
                    }
                }
            }
            if (!ranked) {
                durability_order_count = 0;
            }
        }
        if (!ranked) {
            durability_order_count = 0;
            for (const auto & idle : slots) {
                if (!base_eligible(idle)) {
                    continue;
                }
                if (durability_order_count >=
                        vbr_idle_durability_order.size()) {
                    return 0;
                }
                vbr_idle_durability_order[durability_order_count++] = idle.id;
            }
            std::sort(
                vbr_idle_durability_order.begin(),
                vbr_idle_durability_order.begin() +
                    durability_order_count,
                [&](int32_t lhs, int32_t rhs) {
                    const auto * left = slot_by_id(lhs);
                    const auto * right = slot_by_id(rhs);
                    return left && right
                        ? std::tie(left->t_last_used, left->id) <
                              std::tie(right->t_last_used, right->id)
                        : lhs < rhs;
                });
        }

        // Classify every ranked live frontier in one host-cache pass. Canonical
        // artifact identities narrow the scan; the cache then authenticates
        // the complete token block before calling a row durable.
        size_t frontier_query_count = 0;
        try {
            if (vbr_durable_stems_by_seq.size() != slots.size()) {
                return 0;
            }
            std::fill(
                vbr_durable_stems_by_seq.begin(),
                vbr_durable_stems_by_seq.end(), uint8_t(0));
            for (size_t order = 0; order < durability_order_count; ++order) {
                if (!capture_session.continue_capture()) {
                    return 0;
                }
                auto * selected = slot_by_id(
                    vbr_idle_durability_order[order]);
                if (!selected || !base_eligible(*selected)) {
                    continue;
                }
                if (frontier_query_count >=
                        vbr_idle_frontier_queries.size()) {
                    return 0;
                }
                auto & query =
                    vbr_idle_frontier_queries[frontier_query_count];
                query.slot_id = selected->id;
                query.prompt = &selected->prompt;
                server_vbr_artifact_capture_status identity_status;
                if (!build_capture_identity_fields(
                        *selected, query.identity, identity_status)) {
                    continue;
                }
                query.durable = false;
                query.expected_stem_artifact =
                    selected->vbr_idle_stem_source_valid
                    ? selected->vbr_idle_stem_host_artifact
                    : llama_cache_acct_artifact_id {};
                query.stem_durable = false;
                frontier_query_count++;
            }
            if (!prompt_cache->mark_vbr_frontiers(
                    vbr_idle_frontier_queries.data(),
                    frontier_query_count) ||
                !capture_session.continue_capture()) {
                return 0;
            }
            for (size_t i = 0; i < frontier_query_count; ++i) {
                const auto & query = vbr_idle_frontier_queries[i];
                if (query.slot_id < 0 || size_t(query.slot_id) >=
                        vbr_durable_sources_by_seq.size()) {
                    return 0;
                }
                vbr_durable_sources_by_seq[size_t(query.slot_id)] =
                    query.durable ? 1 : 0;
                vbr_durable_stems_by_seq[size_t(query.slot_id)] =
                    query.stem_durable ? 1 : 0;
            }
        } catch (...) {
            return 0;
        }

        const auto durable_at_snapshot = [&](const server_slot & idle) {
            return idle.id >= 0 &&
                size_t(idle.id) < vbr_durable_sources_by_seq.size() &&
                vbr_durable_sources_by_seq[size_t(idle.id)] != 0;
        };
        for (size_t order = durability_order_count; order > 0; --order) {
            auto * selected = slot_by_id(
                vbr_idle_durability_order[order - 1]);
            if (!selected || !base_eligible(*selected)) {
                continue;
            }
            auto & idle = *selected;
            if (capture_attempts + displacement_attempts ==
                    VBR_PROJECTED_CAPTURE_MAX_MANIFESTS) {
                break;
            }
            const bool durable = durable_at_snapshot(idle);
            std::array<uint8_t, 32> token_identity_digest = {};
            if (durable &&
                !idle.prompt.tokens.retention_token_digest(
                    token_identity_digest)) {
                continue;
            }
            const auto durable_identity = durable
                ? vbr_idle_publication_identity(
                    token_identity_digest,
                    representation.tier_epoch,
                    representation.tier_epoch_swa)
                : std::array<uint8_t, 32> {};
            const bool representation_changed = durable &&
                (!idle.vbr_idle_capture_representation_valid ||
                 idle.vbr_idle_capture_published_identity !=
                    durable_identity);
            // Refresh and fresh publication have different cache terminals.
            // Keep each projected wave homogeneous so neither class can make
            // the other fail admission forever on a stable ranking.
            if ((representation_changed && !manifests.empty()) ||
                (!representation_changed && refresh_selected)) {
                continue;
            }
            // Scalar refresh performs one exact physical-union admission.
            // Bound that global work to one row per synchronous idle wave;
            // remaining durable sources drain on later quiet ticks. Initial
            // publications still share the normal <=8-manifest projection.
            if (representation_changed && refresh_selected) {
                continue;
            }
            if (durable && !representation_changed) {
                // Existing durable rows are displaced only after the capture
                // wave, in ascending retention value. Clearing them here
                // would mutate the shared pool before the selected high-value
                // projections have transferred.
                continue;
            }
            const uint64_t token_references =
                uint64_t(idle.prompt.n_tokens());
            if (token_references > MAX_IDLE_TOKEN_REFERENCES -
                    total_token_references) {
                continue;
            }
            ++capture_attempts;
            if (!durable &&
                !idle.prompt.tokens.retention_token_digest(
                    token_identity_digest)) {
                continue;
            }
            vbr_projected_capture_manifest_request manifest;
            manifest.manifest_id = uint64_t(idle.id) + 1;
            manifest.sequence = idle.id;
            server_vbr_artifact_capture_status identity_status;
            if (!build_capture_identity_fields(
                    idle, manifest.identity, identity_status)) {
                continue;
            }
            manifest.text_only = !idle.prompt.tokens.has_media() &&
                manifest.identity.next_position ==
                    manifest.identity.token_count;
            const auto full_source_identity =
                vbr_idle_capture_attempt_digest(
                    manifest.identity,
                    token_identity_digest,
                    representation.tier_epoch,
                    representation.tier_epoch_swa);
            const auto * refusal_retention =
                idle.vbr_idle_admission_refusals.empty()
                    ? nullptr : live_retention_owner();
            const bool refusal_currency_available =
                refusal_retention && cache_authority;
            const uint64_t refusal_accounting_serial =
                refusal_currency_available
                    ? cache_authority->ledger.serial() : 0;
            const uint64_t refusal_retention_epoch =
                refusal_currency_available
                    ? refusal_retention->competition_epoch_value() : 0;
            const auto refusal_lease = refusal_currency_available &&
                    idle.lease_obs
                ? idle.lease_obs->retry_witness(false)
                : server_cache_lease_retry_witness {};
            const auto refusal_suppressed = [&] (
                    const std::array<uint8_t, 32> & attempt) noexcept {
                return refusal_currency_available &&
                    vbr_idle_admission_refusal_suppressed(
                        idle.vbr_idle_admission_refusals, attempt,
                        refusal_accounting_serial,
                        refusal_retention_epoch, refusal_lease);
            };

            // Stateful prefixes may only be cut at an already sealed context
            // checkpoint.  The checkpoint owns the exact recurrent/draft/
            // accelerator image for that frontier; the attention capture
            // below is then bound to the same token count.  iSWA remains
            // excluded here because its PARTIAL_ONLY checkpoint contains
            // tier-sensitive attention bytes rather than a recurrent-only
            // companion.
            const common_prompt_checkpoint * checkpoint_frontier = nullptr;
            bool checkpoint_candidate_seen = false;
            std::array<uint8_t, 32> attempt_identity =
                full_source_identity;
            const bool matching_stem_source =
                server_vbr_stem_matches_capture_source(
                    idle.vbr_idle_stem_source_valid,
                    idle.vbr_idle_stem_source_identity,
                    full_source_identity);
            const bool stem_already_durable = matching_stem_source &&
                idle.id >= 0 &&
                size_t(idle.id) < vbr_durable_stems_by_seq.size() &&
                vbr_durable_stems_by_seq[size_t(idle.id)] != 0;
            // Reuse demand reopens checkpoint selection when the durable stem
            // belongs to a different source or frontier. If the authenticated
            // host witness already seals this exact demanded frontier, consume
            // the demand instead of repeating an idempotent companion/D2H
            // capture on every later branch hit.
            if (idle.vbr_reuse_capture_frontier != 0 &&
                stem_already_durable &&
                idle.vbr_idle_stem_source_valid &&
                idle.vbr_idle_stem_source_identity ==
                    full_source_identity &&
                idle.vbr_idle_stem_coverage_tokens ==
                    idle.vbr_reuse_capture_frontier) {
                idle.vbr_reuse_capture_frontier = 0;
            }
            if (!durable &&
                (!stem_already_durable ||
                 idle.vbr_reuse_capture_frontier != 0) &&
                (!matching_stem_source ||
                 idle.vbr_reuse_capture_frontier != 0) &&
                n_swa == 0 &&
                (llama_model_is_hybrid(model_tgt) || ctx_dft ||
                 idle.can_speculate())) {
                bool demanded_checkpoint_seen = false;
                const auto consider_checkpoint = [&] (
                        const common_prompt_checkpoint & checkpoint,
                        bool demanded) {
                    if (checkpoint_frontier || checkpoint.empty() ||
                        !checkpoint.computation_frontier.valid() ||
                        checkpoint.n_tokens <= 0 ||
                        checkpoint.n_tokens >= idle.prompt.n_tokens() ||
                        checkpoint.computation_frontier.sequence_epoch !=
                            idle.prompt.sequence_epoch ||
                        (ctx_dft && checkpoint.data_dft.empty())) {
                        return;
                    }
                    checkpoint_candidate_seen = true;
                    demanded_checkpoint_seen |= demanded;
                    vbr_artifact_identity_block checkpoint_identity;
                    std::array<uint8_t, 32> checkpoint_token_digest = {};
                    if (!build_capture_identity_fields(
                            idle, checkpoint_identity, identity_status,
                            checkpoint.n_tokens) ||
                        checkpoint_identity.next_position !=
                            checkpoint.computation_frontier.next_position ||
                        !idle.prompt.tokens.retention_token_prefix_digest(
                            size_t(checkpoint.n_tokens),
                            checkpoint_token_digest)) {
                        return;
                    }
                    const auto checkpoint_attempt =
                        vbr_idle_capture_attempt_digest(
                            checkpoint_identity, checkpoint_token_digest,
                            representation.tier_epoch,
                            representation.tier_epoch_swa);
                    if (refusal_suppressed(checkpoint_attempt)) {
                        return;
                    }
                    checkpoint_frontier = &checkpoint;
                    manifest.identity = std::move(checkpoint_identity);
                    token_identity_digest = checkpoint_token_digest;
                    attempt_identity = checkpoint_attempt;
                };

                // Reuse demand wins over size/recency: for a stateful tree a
                // newer checkpoint cannot be projected backward across its
                // companion. Find the exact frontier that actually served a
                // prior branch before considering ordinary newest-first work.
                if (idle.vbr_reuse_capture_frontier != 0) {
                    for (auto it = idle.prompt.checkpoints.rbegin();
                         it != idle.prompt.checkpoints.rend(); ++it) {
                        if (uint64_t(std::max<int64_t>(0, it->n_tokens)) ==
                                idle.vbr_reuse_capture_frontier) {
                            consider_checkpoint(*it, true);
                            break;
                        }
                    }
                    if (demanded_checkpoint_seen && !checkpoint_frontier) {
                        continue;
                    }
                    if (!demanded_checkpoint_seen) {
                        idle.vbr_reuse_capture_frontier = 0;
                    }
                }
                if (!checkpoint_frontier) {
                    for (auto it = idle.prompt.checkpoints.rbegin();
                         it != idle.prompt.checkpoints.rend(); ++it) {
                        consider_checkpoint(*it, false);
                        if (checkpoint_frontier) {
                            break;
                        }
                    }
                }
                // A refusal belongs to one sealed frontier, not the whole
                // source. Walk newest-to-oldest until a frontier is not
                // suppressed; if all current frontiers have stable refusal
                // evidence, wait for capacity/lease/retention currency to
                // change instead of falling through to an unrelated live
                // companion capture.
                if (checkpoint_candidate_seen && !checkpoint_frontier) {
                    continue;
                }
            }
            bool checkpoint_stem = checkpoint_frontier != nullptr;

            bool stem_retry = false;
            const bool matching_attempt_stem_source =
                server_vbr_stem_matches_capture_source(
                    idle.vbr_idle_stem_source_valid,
                    idle.vbr_idle_stem_source_identity,
                    attempt_identity);
            const bool durable_stem = idle.id >= 0 &&
                size_t(idle.id) < vbr_durable_stems_by_seq.size() &&
                vbr_durable_stems_by_seq[size_t(idle.id)] != 0;
            if (!checkpoint_stem && !durable &&
                !idle.prompt.tokens.has_media() &&
                matching_attempt_stem_source) {
                if (idle.vbr_idle_stem_coverage_tokens != 0 &&
                    durable_stem) {
                    continue;
                }
                // A coordinated recurrent/SWA tree cannot manufacture an
                // attention-only retry. Drop the stale witness and let the
                // next quiet tick select a sealed checkpoint again; do not
                // fall through and immediately replace the useful frontier
                // with the full source under the same host-capacity pressure.
                if (requires_coordinated_tree_clear) {
                    idle.vbr_idle_stem_source_valid = false;
                    idle.vbr_idle_stem_host_artifact = {};
                    continue;
                }
                idle.vbr_idle_stem_source_valid = false;
                idle.vbr_idle_stem_host_artifact = {};
                idle.vbr_idle_stem_retry = true;
            }
            if (!checkpoint_stem && !durable &&
                !requires_coordinated_tree_clear &&
                !idle.prompt.tokens.has_media()) {
                stem_retry = idle.vbr_idle_stem_retry &&
                    idle.vbr_idle_stem_source_identity ==
                        attempt_identity;
            }

            if (refusal_suppressed(attempt_identity)) {
                continue;
            }

            if ((stem_retry || checkpoint_stem) &&
                ggml_time_ms() < idle.vbr_idle_capture_retry_after_ms) {
                continue;
            }

            // A stem retry owns a singleton projection because it changes the
            // frontier of the complete physical union. Earlier exact rows
            // finish first; a later quiet tick then isolates this source.
            if ((stem_retry || checkpoint_stem) && !manifests.empty()) {
                continue;
            }

            if (!stem_retry && !checkpoint_stem &&
                vbr_idle_capture_attempt_suppressed(
                    idle.vbr_idle_capture_attempt_identity,
                    attempt_identity, idle.vbr_idle_capture_terminal,
                    idle.vbr_idle_capture_retry_after_ms, ggml_time_ms())) {
                continue;
            }
            server_prompt_cache_vbr_publication_metadata publication;
            const bool prompt_boundary_stem =
                vbr_prompt_boundary_capture_slot == idle.id;
            if (!representation_changed) {
                const bool prepared = checkpoint_stem || prompt_boundary_stem
                    ? prompt_cache->prepare_vbr_stem_publication_metadata(
                        idle.prompt,
                        checkpoint_stem
                            ? checkpoint_frontier->n_tokens
                            : manifest.identity.token_count,
                        manifest.identity.execution_identity,
                        manifest.identity.adapter_config_identity,
                        idle.id, publication)
                    : !stem_retry &&
                        prompt_cache->prepare_vbr_publication_metadata(
                            idle.prompt,
                            manifest.identity.execution_identity,
                            manifest.identity.adapter_config_identity,
                            idle.id, publication);
                if (!prepared && !stem_retry) {
                    continue;
                }
            }
            const int64_t trace_token_count = manifest.identity.token_count;
            const llama_pos trace_next_position =
                manifest.identity.next_position;
            manifest.token_block.reserve(
                size_t(manifest.identity.token_count));
            for (int64_t i = 0; i < manifest.identity.token_count; ++i) {
                manifest.token_block.push_back(
                    idle.prompt.tokens[size_t(i)]);
            }
            try {
                manifests.push_back(std::move(manifest));
                candidates.push_back({
                    &idle, checkpoint_frontier,
                    uint64_t(idle.id) + 1, attempt_identity,
                    full_source_identity,
                    representation.tier_epoch,
                    representation.tier_epoch_swa,
                    representation_changed,
                    stem_retry || checkpoint_stem || prompt_boundary_stem,
                    checkpoint_stem || prompt_boundary_stem,
                    checkpoint_stem
                        ? uint64_t(checkpoint_frontier->n_tokens)
                        : prompt_boundary_stem
                            ? uint64_t(manifest.identity.token_count) : 0,
                    std::move(publication),
                });
                total_token_references += token_references;
                refresh_selected |= representation_changed;
                if (params_base.cache_debug) {
                    const auto & selected = candidates.back();
                    SRV_INF(
                        "VBR_CAPTURE_TRACE event=candidate slot=%d manifest=%" PRIu64
                        " attempt=%02x%02x%02x%02x%02x%02x%02x%02x "
                        "tokens=%" PRId64 " next=%d durable=%d refresh=%d "
                        "stem_requested=%d checkpoint=%d prepressure=%d "
                        "prior_same=%d prior_terminal=%d retry_in_ms=%" PRId64
                        " published_identity_valid=%d "
                        "tier_epoch=%" PRIu64 " tier_epoch_swa=%" PRIu64 "\n",
                        idle.id, selected.manifest_id,
                        unsigned(selected.attempt_identity[0]),
                        unsigned(selected.attempt_identity[1]),
                        unsigned(selected.attempt_identity[2]),
                        unsigned(selected.attempt_identity[3]),
                        unsigned(selected.attempt_identity[4]),
                        unsigned(selected.attempt_identity[5]),
                        unsigned(selected.attempt_identity[6]),
                        unsigned(selected.attempt_identity[7]),
                        trace_token_count, int(trace_next_position),
                        int(durable), int(representation_changed),
                        int(selected.stem_requested), int(checkpoint_stem),
                        int(readiness_only),
                        int(idle.vbr_idle_capture_attempt_identity ==
                            selected.attempt_identity),
                        int(idle.vbr_idle_capture_terminal),
                        std::max<int64_t>(
                            0, idle.vbr_idle_capture_retry_after_ms-
                                ggml_time_ms()),
                        int(idle.vbr_idle_capture_representation_valid),
                        selected.tier_epoch,
                        selected.tier_epoch_swa);
                }
                if (readiness_only) {
                    break;
                }
                // A payload-complete iSWA child is one exact artifact-wide
                // dependency. It cannot be partitioned across the projected
                // <=8-manifest union, so capture one ranked source per idle
                // wave through the exact host handoff below.
                if (stem_retry || checkpoint_stem ||
                    requires_coordinated_tree_clear || ctx_dft ||
                    idle.can_speculate()) {
                    break;
                }
            } catch (...) {
                return 0;
            }
        }
        const auto displace_existing_sources = [&]() {
            // iSWA owns one shared payload-complete child. Removing a single
            // sequence cannot empty that child, so its live aliases are
            // reclaimed atomically below only after every resident slot has
            // a durable host frontier.
            if (requires_coordinated_tree_clear ||
                !may_displace_live_source) {
                return;
            }
            for (size_t order = 0;
                 order < durability_order_count &&
                 capture_attempts + displacement_attempts <
                    VBR_PROJECTED_CAPTURE_MAX_MANIFESTS;
                 ++order) {
                auto * selected = slot_by_id(
                    vbr_idle_durability_order[order]);
                if (!selected || !base_eligible(*selected) ||
                    !durable_at_snapshot(*selected)) {
                    continue;
                }
                auto & idle = *selected;
                const std::string & adapter_key =
                    idle.vbr_adapter_config_identity;
                std::array<uint8_t, 32> token_identity_digest = {};
                if (!idle.prompt.tokens.retention_token_digest(
                        token_identity_digest)) {
                    continue;
                }
                const auto current_publication_identity =
                    vbr_idle_publication_identity(
                        token_identity_digest,
                        representation.tier_epoch,
                        representation.tier_epoch_swa);
                const bool representation_changed =
                    !idle.vbr_idle_capture_representation_valid ||
                    idle.vbr_idle_capture_published_identity !=
                        current_publication_identity;
                if (representation_changed ||
                    automatic_vbr_cache_support(
                        idle.prompt.tokens, idle.lora,
                        idle.can_speculate()) !=
                        server_vbr_prompt_cache_support_status::supported ||
                    !capture_session.continue_capture() ||
                    !prompt_cache->contains_vbr_frontier(
                        idle.prompt, frontier_execution_identity,
                        adapter_key)) {
                    continue;
                }
                ++displacement_attempts;
                if (idle.prompt_clear_after_vbr_publication()) {
                    ++displaced_existing;
                }
            }
        };
        if (manifests.empty()) {
            if (readiness_only) {
                return 0;
            }
            displace_existing_sources();
            return displaced_existing;
        }
        if (!cache_plan_observe_live_memory(false)) {
            return 0;
        }

        std::vector<server_vbr_projected_host_publish_result> captured;
        server_vbr_projected_host_capture_diagnostics diagnostics;
        server_prompt_cache_vbr_capacity_claim capacity_claim;
        bool has_refresh_candidate = false;
        bool has_fresh_candidate = false;
        for (const auto & candidate : candidates) {
            has_refresh_candidate |= candidate.refresh;
            has_fresh_candidate |= !candidate.refresh;
        }
        const int64_t started = ggml_time_us();

        const bool stateful_companion_capture =
            requires_coordinated_tree_clear || ctx_dft ||
            (!candidates.empty() && candidates.front().slot &&
             (candidates.front().slot->can_speculate() ||
              candidates.front().checkpoint != nullptr));
        const bool projected_stateful_capture =
            !readiness_only && stateful_companion_capture &&
            projected_attention_layout_supported &&
            candidates.size() == 1 && manifests.size() == 1;
        const bool exact_companion_capture = readiness_only ||
            (stateful_companion_capture && !projected_stateful_capture);
        // Bound every capture by configured durable host capacity (and the
        // library's 16 GiB per-wave ceiling), not by a small transport-window
        // constant that would silently truncate long reusable prefixes. The
        // transfer probes the queue-owned cancellation token between bounded
        // chunks, so newly posted work interrupts a large idle wave promptly.
        const uint64_t runway = vbr_automatic_exact_capture_max_bytes();
        if (runway == 0 || !capture_session.continue_capture()) {
            return 0;
        }
        if (projected_stateful_capture) {
            auto & candidate = candidates.front();
            vbr_explicit_capture_request stateful_request;
            server_vbr_artifact_capture_status request_status;
            if (!candidate.slot || !build_capture_request(
                    *candidate.slot, stateful_request, request_status,
                    candidate.checkpoint)) {
                return 0;
            }
            auto & manifest = manifests.front();
            const auto & expected = manifest.identity;
            const auto & observed = stateful_request.identity;
            if (stateful_request.sequence != manifest.sequence ||
                observed.execution_identity != expected.execution_identity ||
                observed.adapter_config_identity !=
                    expected.adapter_config_identity ||
                observed.media_content_identity !=
                    expected.media_content_identity ||
                observed.sequence_epoch != expected.sequence_epoch ||
                observed.token_count != expected.token_count ||
                observed.next_position != expected.next_position ||
                stateful_request.token_block != manifest.token_block) {
                return 0;
            }
            manifest.companions = std::move(stateful_request.companions);
        }
        if (exact_companion_capture) {
            if (candidates.size() != 1 || manifests.size() != 1 ||
                !capture_session.continue_capture()) {
                return 0;
            }
            auto & candidate = candidates.front();
            vbr_explicit_capture_request exact_request;
            server_vbr_artifact_capture_status request_status;
            if (!candidate.slot || !build_capture_request(
                    *candidate.slot, exact_request, request_status,
                    candidate.checkpoint)) {
                return 0;
            }
            auto background =
                std::make_unique<vbr_idle_exact_capture_work>();
            background->session = std::move(capture_session);
            background->started_us = started;
            struct exact_admission_context {
                server_context_impl * owner = nullptr;
                server_queue::idle_capture_session * session = nullptr;
                const server_slot * source = nullptr;
                server_prompt_cache * cache = nullptr;
                server_prompt_cache_vbr_publication_metadata * publication =
                    nullptr;
                server_prompt_cache_vbr_capacity_claim * capacity = nullptr;
                server_vbr_capture_readiness_reservation * readiness = nullptr;
                uint64_t generation = 0;
                server_prompt_cache_vbr_capacity_status capacity_status =
                    server_prompt_cache_vbr_capacity_status::invalid;
                server_vbr_capture_readiness_status readiness_status =
                    server_vbr_capture_readiness_status::invalid;
                bool refresh = false;
                bool prepressure = false;
                bool prompt_boundary = false;
            } exact_admission {
                this, &background->session, candidate.slot,
                prompt_cache.get(),
                &candidate.publication, &background->capacity,
                &background->readiness,
                ++vbr_capture_readiness_generation,
                server_prompt_cache_vbr_capacity_status::invalid,
                server_vbr_capture_readiness_status::invalid,
                candidate.refresh, readiness_only,
                vbr_prompt_boundary_capture_slot == candidate.slot->id,
            };
            // Prompt-boundary sealing protects the immutable prompt rows while
            // first-token sampling and unrelated slots continue. The source
            // slot alone waits before its next KV mutation, so use the explicit
            // inter-token deadline below instead of charging the ordinary
            // decode-growth runway twice.
            exact_admission.prepressure = readiness_only &&
                vbr_prompt_boundary_capture_slot < 0;
            if (vbr_capture_readiness_generation == 0) {
                exact_admission.generation =
                    ++vbr_capture_readiness_generation;
            }
            exact_request.max_packed_bytes = runway;
            exact_request.pretransfer_context = &exact_admission;
            exact_request.pretransfer_admit = [](
                    void * opaque,
                    const vbr_explicit_capture_pretransfer_quote & quote)
                    noexcept {
                auto * context =
                    static_cast<exact_admission_context *>(opaque);
                if (!context || !context->session ||
                    !context->session->continue_capture()) {
                    return false;
                }
                if (!context->owner || !context->source ||
                    !context->readiness) {
                    return false;
                }
                context->readiness_status =
                    server_vbr_capture_readiness_admit(
                        context->owner->vbr_capture_readiness_input_for(
                            *context->source, quote,
                            context->prepressure),
                        context->generation, *context->readiness);
                if (context->readiness_status !=
                        server_vbr_capture_readiness_status::ready) {
                    return false;
                }
                if (context->prompt_boundary &&
                    context->readiness->forecast_capture_us >
                        VBR_PROMPT_BOUNDARY_CAPTURE_DEADLINE_US) {
                    context->readiness_status =
                        server_vbr_capture_readiness_status::deadline_missed;
                    context->readiness->reset();
                    return false;
                }
                if (context->refresh) {
                    return context->cache && context->source &&
                        context->cache->preview_vbr_compact_refresh_capacity(
                            context->source->prompt,
                            context->owner->frontier_execution_identity,
                            lora_config_identity(context->source->lora),
                            quote.conservative_host_resident_bytes) &&
                        context->session->continue_capture();
                }
                auto * publication = context->publication;
                return context->cache && context->capacity && publication &&
                    publication->ready() &&
                    context->cache->prepare_vbr_publication_capacity(
                        &publication, 1,
                        quote.conservative_host_resident_bytes,
                        *context->capacity, &context->capacity_status) &&
                    context->session->continue_capture();
            };
            exact_request.continue_context = background.get();
            exact_request.continue_transfer = [](void * opaque) noexcept {
                const auto * work = static_cast<const
                    vbr_idle_exact_capture_work *>(opaque);
                return work && work->session.continue_capture() &&
                    (work->transfer_deadline_us <= 0 ||
                     ggml_time_us() <= work->transfer_deadline_us);
            };
            const auto exact = vbr_artifact_store->prepare_host_payload(
                *memory, std::move(exact_request), background->capture);
            const uint64_t exact_bytes =
                exact.pretransfer.planned_packed_bytes;
            if (params_base.cache_debug) {
                SRV_INF(
                    "VBR_CAPTURE_TRACE event=prepared slot=%d manifest=%" PRIu64
                    " attempt=%02x%02x%02x%02x%02x%02x%02x%02x "
                    "mode=exact status=%s library_status=%s phase=%s "
                    "payload=%" PRIu64 " stash=%" PRIu64
                    " companion=%" PRIu64 " metadata=%" PRIu64
                    " planned=%" PRIu64 " host=%" PRIu64
                    " controllers=%u units=%u companions=%u "
                    "readiness=%u capacity=%u\n",
                    candidate.slot->id, candidate.manifest_id,
                    unsigned(candidate.attempt_identity[0]),
                    unsigned(candidate.attempt_identity[1]),
                    unsigned(candidate.attempt_identity[2]),
                    unsigned(candidate.attempt_identity[3]),
                    unsigned(candidate.attempt_identity[4]),
                    unsigned(candidate.attempt_identity[5]),
                    unsigned(candidate.attempt_identity[6]),
                    unsigned(candidate.attempt_identity[7]),
                    server_vbr_artifact_capture_status_name(exact.status),
                    vbr_explicit_capture_status_name(exact.library_status),
                    vbr_explicit_capture_phase_name(exact.phase),
                    exact.pretransfer.payload_bytes,
                    exact.pretransfer.stash_bytes,
                    exact.pretransfer.companion_bytes,
                    exact.pretransfer.metadata_bytes,
                    exact.pretransfer.planned_packed_bytes,
                    exact.pretransfer.conservative_host_resident_bytes,
                    exact.pretransfer.controllers, exact.pretransfer.units,
                    exact.pretransfer.companions,
                    unsigned(exact_admission.readiness_status),
                    unsigned(exact_admission.capacity_status));
            }
            const bool exact_size_ok =
                vbr_explicit_capture_pretransfer_quote_admissible(
                    exact.pretransfer, runway);
            if (exact.status != server_vbr_artifact_capture_status::ok ||
                !background->capture.ready_for_transfer() ||
                !exact_size_ok ||
                !background->session.continue_capture()) {
                SRV_DBG(
                    "VBR_IDLE_CAPTURE exact-complete refused status=%s "
                    "library_status=%s phase=%s generation=%s bytes=%" PRIu64
                    " readiness=%u capacity=%u duration_ms=%.3f\n",
                    server_vbr_artifact_capture_status_name(exact.status),
                    vbr_explicit_capture_status_name(exact.library_status),
                    vbr_explicit_capture_phase_name(exact.phase),
                    vbr_explicit_generation_failure_name(
                        exact.generation_failure), exact_bytes,
                    unsigned(exact_admission.readiness_status),
                    unsigned(exact_admission.capacity_status),
                    (ggml_time_us() - started)/1000.0);
                candidate.slot->vbr_idle_capture_attempt_identity =
                    candidate.attempt_identity;
                candidate.slot->vbr_idle_capture_terminal =
                    exact.library_status ==
                        vbr_explicit_capture_status::unsupported_layout ||
                    exact.library_status ==
                        vbr_explicit_capture_status::not_armed;
                candidate.slot->vbr_idle_capture_retry_after_ms =
                    candidate.slot->vbr_idle_capture_terminal
                        ? 0 : ggml_time_ms() + 5000;
                return 0;
            }
            background->candidate.slot_id = candidate.slot->id;
            background->candidate.manifest_id = candidate.manifest_id;
            background->candidate.attempt_identity =
                candidate.attempt_identity;
            background->candidate.full_source_identity =
                candidate.full_source_identity;
            background->candidate.tier_epoch = candidate.tier_epoch;
            background->candidate.tier_epoch_swa =
                candidate.tier_epoch_swa;
            background->candidate.refresh = candidate.refresh;
            background->candidate.stem_requested =
                candidate.stem_requested;
            background->candidate.stemmed = candidate.stemmed;
            background->candidate.prepressure = readiness_only;
            background->candidate.active_prompt_frontier =
                vbr_prompt_boundary_capture_slot == candidate.slot->id;
            background->candidate.selected_tokens =
                candidate.selected_tokens;
            background->candidate.publication =
                std::move(candidate.publication);
            vbr_idle_exact_capture = std::move(background);
            auto * worker = vbr_idle_exact_capture.get();
            try {
                worker->transfer_started_us = ggml_time_us();
                if (worker->candidate.active_prompt_frontier) {
                    worker->transfer_deadline_us =
                        worker->transfer_started_us+
                        int64_t(VBR_PROMPT_BOUNDARY_CAPTURE_DEADLINE_US);
                }
                worker->worker = std::thread([this, worker]() noexcept {
                    worker->transfer =
                        vbr_artifact_store->transfer_host_payload(
                            worker->capture);
                    if (params_base.cache_debug) {
                        const auto & identity =
                            worker->candidate.attempt_identity;
                        SRV_INF(
                            "VBR_CAPTURE_TRACE event=transfer_complete slot=%d "
                            "manifest=%" PRIu64
                            " attempt=%02x%02x%02x%02x%02x%02x%02x%02x "
                            "status=%s library_status=%s phase=%s "
                            "payload=%" PRIu64 " stash=%" PRIu64
                            " companion=%" PRIu64 " chunks=%" PRIu64
                            " duration_ms=%.3f\n",
                            worker->candidate.slot_id,
                            worker->candidate.manifest_id,
                            unsigned(identity[0]), unsigned(identity[1]),
                            unsigned(identity[2]), unsigned(identity[3]),
                            unsigned(identity[4]), unsigned(identity[5]),
                            unsigned(identity[6]), unsigned(identity[7]),
                            server_vbr_artifact_capture_status_name(
                                worker->transfer.status),
                            vbr_explicit_capture_status_name(
                                worker->transfer.library_status),
                            vbr_explicit_capture_phase_name(
                                worker->transfer.phase),
                            worker->transfer.payload_bytes,
                            worker->transfer.stash_bytes,
                            worker->transfer.companion_bytes,
                            worker->transfer.chunks,
                            (ggml_time_us()-worker->transfer_started_us)/1000.0);
                    }
                    worker->complete.store(
                        true, std::memory_order_release);
                    queue_tasks.request_idle_maintenance();
                });
            } catch (...) {
                candidate.slot->vbr_idle_capture_attempt_identity =
                    candidate.attempt_identity;
                candidate.slot->vbr_idle_capture_terminal = false;
                candidate.slot->vbr_idle_capture_retry_after_ms =
                    ggml_time_ms() + 5000;
                vbr_idle_exact_capture.reset();
                return 0;
            }
            SRV_DBG(
                "VBR_IDLE_CAPTURE exact-background admitted bytes=%" PRIu64
                " prepare_ms=%.3f\n",
                exact_bytes, (ggml_time_us() - started)/1000.0);
            return 0;
        } else {
        std::vector<server_prompt_cache_vbr_publication_metadata *>
            admitted_publications;
        try {
            admitted_publications.reserve(candidates.size());
        } catch (...) {
            return 0;
        }
        struct idle_admission_context {
            server_context_impl * owner = nullptr;
            server_queue::idle_capture_session * session = nullptr;
            server_prompt_cache * cache = nullptr;
            std::vector<candidate> * candidates = nullptr;
            std::vector<server_prompt_cache_vbr_publication_metadata *> *
                admitted = nullptr;
            server_prompt_cache_vbr_capacity_claim * capacity = nullptr;
            bool has_refresh = false;
            bool has_fresh = false;
            uint64_t tier_epoch = 0;
            uint64_t tier_epoch_swa = 0;
            server_prompt_cache_vbr_capacity_status capacity_status =
                server_prompt_cache_vbr_capacity_status::invalid;
            bool capacity_refused = false;
            uint64_t capacity_retry_manifest_id = 0;
        } admission_state {
            this, &capture_session, prompt_cache.get(), &candidates,
            &admitted_publications, &capacity_claim,
            has_refresh_candidate, has_fresh_candidate,
            representation.tier_epoch, representation.tier_epoch_swa,
        };
        server_vbr_projected_capture_admission admission;
        admission.context = &admission_state;
        admission.frontier = candidates.size() == 1 &&
                candidates.front().stem_requested &&
                !candidates.front().stemmed
            ? vbr_projected_capture_frontier_policy {
                vbr_projected_capture_frontier_mode::
                    longest_attention_stem,
                SERVER_PROMPT_CACHE_MIN_RETENTION_REUSE_TOKENS,
                prompt_cache->limit_size,
                prompt_cache->limit_size == 0
                    ? prompt_cache->limit_tokens : 0,
            }
            : vbr_projected_capture_frontier_policy {};
        admission.prepare = [](
                void * opaque,
                const server_vbr_projected_capture_admission::quote & quote)
                noexcept {
            auto * context = static_cast<idle_admission_context *>(opaque);
            auto * session = context ? context->session : nullptr;
            if (!context || !context->cache || !context->candidates ||
                !session || !session->continue_capture()) {
                return false;
            }
            if (context->has_refresh) {
                return !context->has_fresh && session->continue_capture();
            }
            for (const auto & durable : quote.durable) {
                auto found = std::find_if(
                    context->candidates->begin(),
                    context->candidates->end(),
                    [&](const candidate & value) {
                        return !value.refresh &&
                            value.manifest_id == durable.manifest_id;
                    });
                if (found == context->candidates->end() || !found->slot ||
                    durable.selected_token_count == 0 ||
                    durable.selected_token_count > uint64_t(INT64_MAX) ||
                    durable.selected_token_count >
                        durable.requested_token_count) {
                    return false;
                }
                const uint64_t expected_requested_tokens = found->stemmed
                    ? found->selected_tokens
                    : uint64_t(found->slot->prompt.n_tokens());
                if (expected_requested_tokens == 0 ||
                    durable.requested_token_count !=
                        expected_requested_tokens) {
                    return false;
                }
                if (found->publication.ready()) {
                    if (durable.stemmed ||
                        (found->stem_requested && !found->stemmed)) {
                        return false;
                    }
                    continue;
                }
                if (!found->stem_requested) {
                    return false;
                }
                vbr_artifact_identity_block selected_identity;
                server_vbr_artifact_capture_status identity_status;
                if (!context->owner ||
                    !context->owner->build_capture_identity_fields(
                        *found->slot, selected_identity, identity_status,
                        int64_t(durable.selected_token_count)) ||
                    selected_identity.next_position !=
                        durable.selected_next_position) {
                    return false;
                }
                std::array<uint8_t, 32> selected_token_digest = {};
                if (!found->slot->prompt.tokens.
                        retention_token_prefix_digest(
                            size_t(durable.selected_token_count),
                            selected_token_digest)) {
                    return false;
                }
                found->attempt_identity = vbr_idle_capture_attempt_digest(
                    selected_identity, selected_token_digest,
                    context->tier_epoch, context->tier_epoch_swa);
                // `durable.stemmed` says the core planner shortened the
                // submitted request. A sealed checkpoint request is already a
                // shorter semantic frontier before it reaches that planner, so
                // an exact projection of it must retain the server-owned stem
                // classification. Otherwise the next idle wave treats the
                // checkpoint as a full source capture and may immediately
                // evict the useful branch frontier it just published.
                found->stemmed = found->stemmed || durable.stemmed;
                found->selected_tokens = durable.selected_token_count;
                const bool prepared = durable.stemmed
                    ? context->cache->prepare_vbr_stem_publication_metadata(
                        found->slot->prompt,
                        int64_t(durable.selected_token_count),
                        selected_identity.execution_identity,
                        selected_identity.adapter_config_identity,
                        found->slot->id, found->publication)
                    : context->cache->prepare_vbr_publication_metadata(
                        found->slot->prompt,
                        selected_identity.execution_identity,
                        selected_identity.adapter_config_identity,
                        found->slot->id, found->publication);
                if (!prepared) {
                    return false;
                }
            }
            return session->continue_capture();
        };
        admission.admit = [](
                void * opaque,
                const server_vbr_projected_capture_admission::quote & quote)
                noexcept {
            auto * context = static_cast<idle_admission_context *>(opaque);
            auto * session = context ? context->session : nullptr;
            // Planning can include controller settlement, projection and
            // recurrent sizing. The store now holds the exact transfer-
            // staging and durable claims plus the persistent ring operation
            // while this final queue check runs, so capture cannot begin
            // behind queued work or a competing transfer. Later arrivals
            // cancel between bounded recurrent writes or attention chunks.
            if (!context || !context->cache || !context->candidates ||
                !context->admitted ||
                !context->capacity || !session ||
                !session->continue_capture()) {
                return false;
            }
            // Refresh owns a separate in-place transaction. Do not partition
            // one physical quote between refresh and fresh publication in
            // this first bounded capacity-claim slice.
            if (context->has_refresh) {
                return !context->has_fresh &&
                    session->continue_capture();
            }
            context->admitted->clear();
            for (const auto & durable : quote.durable) {
                server_prompt_cache_vbr_publication_metadata * found =
                    nullptr;
                for (auto & candidate : *context->candidates) {
                    if (!candidate.refresh &&
                        candidate.manifest_id == durable.manifest_id) {
                        if (found) {
                            return false;
                        }
                        found = &candidate.publication;
                    }
                }
                if (!found) {
                    return false;
                }
                if (std::find(
                        context->admitted->begin(),
                        context->admitted->end(), found) !=
                    context->admitted->end()) {
                    return false;
                }
                context->admitted->push_back(found);
            }
            if (context->admitted->empty()) {
                return false;
            }
            if (!context->cache->prepare_vbr_publication_capacity(
                    context->admitted->data(), context->admitted->size(),
                    quote.projected_host_resident_bytes,
                    *context->capacity, &context->capacity_status)) {
                context->capacity_refused = true;
                if (context->capacity_status ==
                        server_prompt_cache_vbr_capacity_status::
                            pressure_batch_unsupported) {
                    std::array<uint64_t,
                        VBR_PROJECTED_CAPTURE_MAX_MANIFESTS> ranked = {};
                    std::array<uint64_t,
                        VBR_PROJECTED_CAPTURE_MAX_MANIFESTS> admitted = {};
                    size_t ranked_count = 0;
                    size_t admitted_count = 0;
                    for (const auto & candidate : *context->candidates) {
                        if (ranked_count < ranked.size()) {
                            ranked[ranked_count++] = candidate.manifest_id;
                        }
                    }
                    for (const auto & durable : quote.durable) {
                        if (admitted_count < admitted.size()) {
                            admitted[admitted_count++] = durable.manifest_id;
                        }
                    }
                    context->capacity_retry_manifest_id =
                        vbr_idle_capacity_retry_manifest(
                            ranked.data(), ranked_count,
                            admitted.data(), admitted_count);
                }
                return false;
            }
            if (!session->continue_capture()) {
                *context->capacity = {};
                return false;
            }
            return true;
        };
        admission.continue_capture = [](void * opaque) noexcept {
            auto * context = static_cast<idle_admission_context *>(opaque);
            auto * session = context ? context->session : nullptr;
            return session && session->continue_capture();
        };
        const bool projected_ok =
            vbr_artifact_store->capture_projected_host_batch(
                *memory, std::move(manifests), runway,
                captured, &admission, &diagnostics);
        if (params_base.cache_debug) {
            SRV_INF(
                "VBR_CAPTURE_TRACE event=projected_complete ok=%d "
                "manifests=%zu status=%s phase=%s frontier=%u "
                "requested_tokens=%" PRIu64 " selected_tokens=%" PRIu64
                " union_cells=%" PRIu64 " planned=%" PRIu64
                " companion=%" PRIu64 " survey_cells=%" PRIu64
                " survey_calls=%u recaptures=%u size_passes=%u "
                "projections=%u transfers=%u ring_acquires=%u "
                "capacity=%u duration_ms=%.3f\n",
                int(projected_ok), candidates.size(),
                vbr_explicit_capture_status_name(
                    diagnostics.capture_status),
                vbr_explicit_capture_phase_name(
                    diagnostics.capture_phase),
                unsigned(diagnostics.frontier_status),
                diagnostics.requested_frontier_tokens,
                diagnostics.selected_frontier_tokens,
                diagnostics.union_cells,
                diagnostics.planned_packed_bytes,
                diagnostics.companion_d2h_bytes,
                diagnostics.frontier_survey_cells,
                diagnostics.frontier_survey_calls,
                diagnostics.frontier_recapture_calls,
                diagnostics.size_pass_calls,
                diagnostics.projection_calls,
                diagnostics.unit_transfer_calls,
                diagnostics.ring_operation_acquires,
                unsigned(admission_state.capacity_status),
                (ggml_time_us() - started)/1000.0);
        }
        if (!projected_ok) {
            SRV_DBG(
                "VBR_IDLE_CAPTURE refused status=%s phase=%s planned=%" PRIu64
                " duration_ms=%.3f\n",
                vbr_explicit_capture_status_name(
                    diagnostics.capture_status),
                vbr_explicit_capture_phase_name(
                    diagnostics.capture_phase),
                diagnostics.planned_packed_bytes,
                (ggml_time_us() - started)/1000.0);
            const bool terminal =
                diagnostics.capture_status ==
                    vbr_explicit_capture_status::unsupported_layout ||
                diagnostics.capture_status ==
                    vbr_explicit_capture_status::not_armed;
            const bool aggregate_over_cap =
                diagnostics.capture_status ==
                    vbr_explicit_capture_status::accounting_failed &&
                diagnostics.planned_packed_bytes > runway;
            const bool incoming_over_host_cap =
                admission_state.capacity_status ==
                    server_prompt_cache_vbr_capacity_status::
                        incoming_exceeds_hard_limit;
            const bool stem_permanently_unavailable =
                diagnostics.frontier_status ==
                    vbr_projected_capture_frontier_status::
                        stem_below_minimum ||
                (diagnostics.frontier_status ==
                     vbr_projected_capture_frontier_status::stem_unsupported &&
                 (diagnostics.capture_status ==
                      vbr_explicit_capture_status::unsupported_layout ||
                  diagnostics.capture_status ==
                      vbr_explicit_capture_status::identity_unavailable ||
                  diagnostics.capture_status ==
                      vbr_explicit_capture_status::
                          required_companion_unavailable));
            const bool witnessable_admission_refusal =
                candidates.size() == 1 &&
                admission_state.capacity_refused &&
                diagnostics.capture_status ==
                    vbr_explicit_capture_status::admission_refused &&
                diagnostics.capture_phase ==
                    vbr_explicit_capture_phase::reservation_preparation;
            // Prepared prompt/retention metadata owns provisional accounting
            // operations. Roll it back before sampling the refusal currency,
            // otherwise its later destructor would make our own witness stale
            // and reopen the identical candidate on the next idle tick.
            if (witnessable_admission_refusal) {
                admitted_publications.clear();
                capacity_claim = {};
                for (auto & candidate : candidates) {
                    candidate.publication = {};
                }
            }
            const int64_t retry_after_ms = ggml_time_ms() + 5000;
            for (size_t i = 0; i < candidates.size(); ++i) {
                const auto & candidate = candidates[i];
                candidate.slot->vbr_idle_capture_attempt_identity =
                    candidate.attempt_identity;
                const bool refusal_witnessed =
                    witnessable_admission_refusal &&
                    vbr_record_idle_admission_refusal(
                        *candidate.slot, candidate.attempt_identity);
                const bool begin_stem_retry = !candidate.refresh &&
                    !candidate.stem_requested && candidates.size() == 1 &&
                    (aggregate_over_cap || incoming_over_host_cap) &&
                    !requires_coordinated_tree_clear &&
                    !candidate.slot->prompt.tokens.has_media();
                if (begin_stem_retry || candidate.stem_requested) {
                    candidate.slot->vbr_idle_stem_source_identity =
                        candidate.full_source_identity;
                    candidate.slot->vbr_idle_stem_source_valid = false;
                    candidate.slot->vbr_idle_stem_host_artifact = {};
                    candidate.slot->vbr_idle_stem_retry =
                        !stem_permanently_unavailable;
                    candidate.slot->vbr_idle_stem_coverage_tokens =
                        diagnostics.selected_frontier_tokens;
                }
                // Aggregate admission is not a property of any one frontier.
                // Reopen one row immediately and cool down its siblings so
                // the next idle tick isolates it. A singleton overflow also
                // remains transient: dynamic retiering can reduce its packed
                // representation without changing the semantic frontier.
                candidate.slot->vbr_idle_capture_terminal =
                    terminal ||
                    (candidate.stem_requested &&
                     stem_permanently_unavailable);
                const bool retry_immediately = vbr_idle_retry_immediately(
                    terminal, aggregate_over_cap,
                    admission_state.capacity_status,
                    candidates.size(),
                    candidate.manifest_id ==
                        diagnostics.first_available_manifest_id,
                    candidate.manifest_id ==
                        admission_state.capacity_retry_manifest_id);
                candidate.slot->vbr_idle_capture_retry_after_ms =
                    begin_stem_retry ? 0 :
                    candidate.slot->vbr_idle_capture_terminal ? 0 :
                    refusal_witnessed ? 0 :
                    retry_immediately ? 0 : retry_after_ms;
                candidate.slot->vbr_idle_capture_isolated_retry =
                    !candidate.slot->vbr_idle_capture_terminal &&
                    candidates.size() > 1 && !retry_immediately &&
                    (aggregate_over_cap ||
                     admission_state.capacity_status ==
                        server_prompt_cache_vbr_capacity_status::
                            pressure_batch_unsupported);
                if (retry_immediately || begin_stem_retry) {
                    queue_tasks.request_idle_maintenance();
                }
            }
            return 0;
        }
        }

        if (has_fresh_candidate &&
            !capacity_claim.requires_publication_revalidation() &&
            !prompt_cache->consume_vbr_publication_capacity(
                capacity_claim)) {
            const int64_t retry_after_ms = ggml_time_ms() + 5000;
            for (auto & candidate : candidates) {
                if (!candidate.refresh && candidate.slot) {
                    candidate.slot->vbr_idle_capture_attempt_identity =
                        candidate.attempt_identity;
                    candidate.slot->vbr_idle_capture_terminal = false;
                    candidate.slot->vbr_idle_capture_retry_after_ms =
                        retry_after_ms;
                    candidate.slot->vbr_idle_capture_isolated_retry = false;
                }
            }
            return displaced_existing;
        }

        size_t published_count = 0;
        size_t refreshed_count = 0;
        size_t displaced_count = 0;
        const auto representation_after =
            memory->vbr_representation_identity();
        for (auto & row : captured) {
            if (!capture_session.continue_capture()) {
                break;
            }
            if (!row.payload ||
                (row.status !=
                     vbr_projected_manifest_publish_status::published &&
                 row.status !=
                     vbr_projected_manifest_publish_status::adopted)) {
                const auto failed = std::find_if(
                    candidates.begin(), candidates.end(),
                    [&](const candidate & value) {
                        return value.manifest_id == row.manifest_id;
                    });
                if (failed != candidates.end()) {
                    failed->slot->vbr_idle_capture_attempt_identity =
                        failed->attempt_identity;
                    // Dependency availability includes live recurrent/source
                    // currency and can recover without a semantic frontier
                    // change. Keep it on the bounded transient cooldown.
                    failed->slot->vbr_idle_capture_terminal = false;
                    failed->slot->vbr_idle_capture_retry_after_ms =
                        ggml_time_ms() + 5000;
                    failed->slot->vbr_idle_capture_isolated_retry = false;
                }
                continue;
            }
            const auto found = std::find_if(
                candidates.begin(), candidates.end(),
                [&](const candidate & value) {
                    return value.manifest_id == row.manifest_id;
                });
            if (found == candidates.end() || !found->slot ||
                found->slot->is_processing() ||
                found->slot->state != SLOT_STATE_IDLE) {
                continue;
            }
            auto & source = *found->slot;
            const bool representation_stable =
                representation_after.tier_epoch == found->tier_epoch &&
                representation_after.tier_epoch_swa ==
                    found->tier_epoch_swa;
            const auto adapter_key =
                row.payload->package().manifest().identity.
                    adapter_config_identity;
            std::array<uint8_t, 32> publication_token_identity = {};
            if (!source.prompt.tokens.retention_token_digest(
                    publication_token_identity)) {
                continue;
            }
            const auto publication_identity =
                vbr_idle_publication_identity(
                    publication_token_identity,
                    found->tier_epoch, found->tier_epoch_swa);
            if (found->refresh) {
                const auto refresh = prompt_cache->refresh_vbr_compact(
                    source.prompt, std::move(row.payload),
                    frontier_execution_identity, adapter_key, source.id);
                if (refresh ==
                        server_prompt_cache_vbr_refresh_status::
                            updated_with_anchor ||
                    refresh ==
                        server_prompt_cache_vbr_refresh_status::
                            updated_compact_only ||
                    refresh ==
                        server_prompt_cache_vbr_refresh_status::unchanged) {
                    source.vbr_idle_capture_published_identity =
                        publication_identity;
                    source.vbr_idle_capture_representation_valid =
                        representation_stable;
                    source.vbr_idle_capture_attempt_identity =
                        found->attempt_identity;
                    source.vbr_idle_capture_terminal = true;
                    source.vbr_idle_capture_retry_after_ms = 0;
                    source.vbr_idle_capture_isolated_retry = false;
                    if (refresh !=
                            server_prompt_cache_vbr_refresh_status::unchanged) {
                        ++refreshed_count;
                    }
                    // Refresh publication never clears its live source. A
                    // later idle/pressure wave observes the now-current
                    // durable frontier and owns any displacement decision.
                    continue;
                }
                source.vbr_idle_capture_attempt_identity =
                    found->attempt_identity;
                source.vbr_idle_capture_terminal = false;
                source.vbr_idle_capture_retry_after_ms =
                    ggml_time_ms() + 5000;
                source.vbr_idle_capture_isolated_retry = false;
                continue;
            }
            server_prompt_cache::iterator published_host =
                prompt_cache->states.end();
            if (!found->publication.ready() ||
                !prompt_cache->publish_vbr(
                    found->publication,
                    server_prompt_cache_payload::from_vbr(
                        std::move(row.payload)),
                    source.cache_family,
                    !source.task || !source.task->is_child(),
                    &published_host,
                    capacity_claim.requires_publication_revalidation()
                        ? &capacity_claim : nullptr)) {
                source.vbr_idle_capture_attempt_identity =
                    found->attempt_identity;
                source.vbr_idle_capture_terminal = false;
                source.vbr_idle_capture_retry_after_ms =
                    ggml_time_ms() + 5000;
                source.vbr_idle_capture_isolated_retry = false;
                continue;
            }
            ++published_count;
            if (found->stemmed) {
                // A stem is an independently sealed, non-consuming host
                // frontier. It cannot certify or clear the longer live
                // source; remember only enough to suppress duplicate work
                // while that exact host stem remains present.
                source.vbr_idle_stem_source_identity =
                    found->full_source_identity;
                source.vbr_idle_stem_coverage_tokens =
                    found->selected_tokens;
                source.vbr_idle_stem_host_artifact =
                    prompt_cache->vbr_host_artifact_id(published_host);
                if (source.vbr_idle_stem_host_artifact.v == 0) {
                    source.vbr_idle_stem_source_valid = false;
                    source.vbr_idle_stem_retry = true;
                    source.vbr_idle_capture_terminal = false;
                    source.vbr_idle_capture_retry_after_ms =
                        ggml_time_ms() + 5000;
                    source.vbr_idle_capture_isolated_retry = false;
                    continue;
                }
                source.vbr_idle_stem_source_valid = true;
                source.vbr_idle_stem_retry = false;
                source.vbr_idle_capture_attempt_identity =
                    found->attempt_identity;
                source.vbr_idle_capture_terminal = true;
                source.vbr_idle_capture_retry_after_ms = 0;
                source.vbr_idle_capture_isolated_retry = false;
                if (source.vbr_reuse_capture_frontier ==
                        found->selected_tokens) {
                    source.vbr_reuse_capture_frontier = 0;
                }
                continue;
            }
            source.vbr_idle_stem_retry = false;
            source.vbr_idle_stem_source_valid = false;
            source.vbr_idle_stem_host_artifact = {};
            source.vbr_idle_capture_published_identity =
                publication_identity;
            source.vbr_idle_capture_representation_valid =
                representation_stable;
            source.vbr_idle_capture_attempt_identity =
                found->attempt_identity;
            source.vbr_idle_capture_terminal = true;
            source.vbr_idle_capture_retry_after_ms = 0;
            source.vbr_idle_capture_isolated_retry = false;

            // Publication is durable, but it is not authority to destroy a
            // different live frontier. Re-derive the complete semantic key
            // immediately before the canonical clear and repeat every live
            // protection check that can change while capture was in flight.
            // A refusal intentionally retains both the host artifact and the
            // live source; a later pressure wave may reconsider the latter.
            vbr_artifact_identity_block current_identity;
            std::array<uint8_t, 32> current_token_identity = {};
            server_vbr_artifact_capture_status identity_status;
            const bool source_unchanged =
                build_capture_identity_fields(
                    source, current_identity, identity_status) &&
                source.prompt.tokens.retention_token_digest(
                    current_token_identity) &&
                vbr_idle_capture_attempt_digest(
                    current_identity,
                    current_token_identity,
                    found->tier_epoch,
                    found->tier_epoch_swa) ==
                    found->attempt_identity && representation_stable;
            if (!source_unchanged || source.is_processing() ||
                source.state != SLOT_STATE_IDLE ||
                automatic_vbr_cache_support(
                    source.prompt.tokens, source.lora,
                    source.can_speculate()) !=
                    server_vbr_prompt_cache_support_status::supported ||
                source.hard_lease_blocks_live_prefix() ||
                !capture_session.continue_capture() ||
                queue_tasks.has_deferred_for_slot(source.id) ||
                !prompt_cache->contains_vbr_frontier(
                    source.prompt, frontier_execution_identity,
                    adapter_key)) {
                continue;
            }
            if (!requires_coordinated_tree_clear &&
                may_displace_live_source) {
                if (!source.prompt_clear_after_vbr_publication()) {
                    continue;
                }
                ++displaced_count;
            }
        }

        // Spend any remaining per-wave operation budget on the coldest
        // already-durable live aliases only after the selected capture batch
        // has completed. This preserves the source geometry of hotter rows
        // throughout projection and D2H.
        displace_existing_sources();

        // An oversized multi-frontier batch is isolated to one manifest so
        // the bounded staging ring is never overcommitted. After that
        // singleton publishes, release exactly one cooled sibling for the
        // next idle maintenance pass. This drains a saturated slot set one
        // bounded transfer at a time without imposing the generic five-second
        // transient-failure backoff between successful captures.
        if (candidates.size() == 1 &&
            published_count + refreshed_count != 0) {
            for (auto & pending : slots) {
                if (!vbr_idle_release_isolated_retry(
                        candidates.size(), published_count + refreshed_count,
                        pending.vbr_idle_capture_isolated_retry,
                        pending.vbr_idle_capture_terminal,
                        pending.state == SLOT_STATE_IDLE,
                        pending.is_processing())) {
                    continue;
                }
                pending.vbr_idle_capture_isolated_retry = false;
                pending.vbr_idle_capture_retry_after_ms = 0;
                queue_tasks.request_idle_maintenance();
                break;
            }
        }

        if (requires_coordinated_tree_clear &&
            may_displace_live_source) {
            std::vector<server_slot *> reclaim;
            bool all_durable = true;
            try {
                reclaim.reserve(slots.size());
                size_t query_count = 0;
                for (auto & live : slots) {
                    if (live.prompt.n_tokens() == 0) {
                        continue;
                    }
                    const auto admission = live.observe_full_slot(
                        server_cache_destruction_class::slot_drop,
                        server_cache_destruction_reason::idle_reclaim);
                    const bool safe = !live.is_processing() &&
                        live.state == SLOT_STATE_IDLE &&
                        automatic_vbr_cache_support(
                            live.prompt.tokens, live.lora,
                            live.can_speculate()) ==
                            server_vbr_prompt_cache_support_status::supported &&
                        !live.hard_lease_blocks_live_prefix() &&
                        !queue_tasks.has_deferred_for_slot(live.id) &&
                        admission.verdict !=
                            server_cache_destruction_verdict::
                                would_refuse_hard_leased &&
                        admission.verdict !=
                            server_cache_destruction_verdict::unavailable;
                    if (!safe) {
                        all_durable = false;
                        break;
                    }
                    if (query_count >= vbr_idle_frontier_queries.size()) {
                        all_durable = false;
                        break;
                    }
                    auto & query = vbr_idle_frontier_queries[query_count++];
                    query.slot_id = live.id;
                    query.prompt = &live.prompt;
                    server_vbr_artifact_capture_status identity_status;
                    if (!build_capture_identity_fields(
                            live, query.identity, identity_status)) {
                        all_durable = false;
                        break;
                    }
                    query.expected_stem_artifact = {};
                    query.durable = false;
                    query.stem_durable = false;
                    reclaim.push_back(&live);
                }
                if (all_durable && query_count != reclaim.size()) {
                    all_durable = false;
                }
                if (all_durable && query_count != 0) {
                    all_durable = prompt_cache->mark_vbr_frontiers(
                        vbr_idle_frontier_queries.data(), query_count);
                }
                if (all_durable) {
                    for (size_t i = 0; i < query_count; ++i) {
                        if (!vbr_idle_frontier_queries[i].durable) {
                            all_durable = false;
                            break;
                        }
                    }
                }
            } catch (...) {
                all_durable = false;
            }
            if (all_durable && !reclaim.empty() &&
                capture_session.continue_capture()) {
                // No per-sequence erase can empty the payload-complete SWA
                // child. The scheduler has authenticated every resident
                // sequence above, so clear the shared tree once and then
                // consume each already-admitted live alias without another
                // fallible operation in between.
                llama_memory_clear(memory, false);
                // Publish an immediately importable empty frontier. Without
                // this settle, the next idle tick eventually shrinks the VBR
                // watermarks, but a request arriving first sees the old
                // mapped frontier as target_not_empty and cold-prefills.
                llama_memory_breathe(memory);
                for (auto * live : reclaim) {
                    const bool cleared =
                        live->prompt_clear_after_vbr_publication();
                    GGML_ASSERT(cleared);
                    if (cleared) {
                        ++displaced_count;
                    }
                }
            }
        }

        SRV_INF(
            "VBR_IDLE_CAPTURE manifests=%zu published=%zu refreshed=%zu displaced=%zu "
            "union_cells=%" PRIu64 " planned=%" PRIu64
            " transfers=%u duration_ms=%.3f\n",
            candidates.size(), published_count, refreshed_count,
            displaced_existing + displaced_count,
            diagnostics.union_cells, diagnostics.planned_packed_bytes,
            diagnostics.unit_transfer_calls,
            (ggml_time_us() - started)/1000.0);
        return published_count;
        } catch (...) {
            return 0;
        }
    }

    // fork 9l-d2c: cache-control authority (fork server-context.cpp 3642-3648).
    // Lazily constructed by its scheduler-only cache-control task; the fork's
    // destroy() closes it before prompt_cache because host proofs point into
    // cache list nodes, and declaring it after the artifact store is the
    // secondary reverse-destruction guard for retained package proofs. With no
    // control route/flag ported yet nothing allocates it -- which is exactly
    // the fork's own production state ("production allocates and executes
    // nothing"). The declaration exists so that the launch prologue's family
    // resolution is verbatim (fork 10280-10290).
    std::unique_ptr<server_cache_control_authority> cache_control_authority;
    // fork 9l-d4b7f: the VBR capture scheduler pin (fork server-context.cpp 3649).
    // The fork has NO assignment site for it anywhere: the first
    // assert_scheduler_thread() call pins it.
    std::thread::id vbr_capture_scheduler_thread;
    // ---------------------------------------------------------------------
    // group A: cache-plan observation face + the retention wave it calls.
    // Bodies verbatim from the fork; the fork coordinates are on each header
    // line. Placed together rather than in the fork's own order: inside a
    // class body that is semantically irrelevant and it keeps the addition
    // additive, so a future rebase can move them mechanically.
    // ---------------------------------------------------------------------

    // fork server-context.cpp:3670  (128 lines)
    bool cache_plan_observe_live_memory(bool certify) noexcept {
        if (!cache_authority || cache_authority->live_device_domains.empty()) {
            return true;
        }

        try {
            using live_row = llama_live_memory_breakdown_data;
            std::vector<live_row> totals(cache_authority->live_device_domains.size());
            bool complete = true;

            const auto add_checked = [&complete](size_t & dst, size_t value) {
                if (value > std::numeric_limits<size_t>::max() - dst) {
                    complete = false;
                    return;
                }
                dst += value;
            };

            llama_context * contexts[] = {
                ctx_tgt,
                // FORK-DIFF: the fork holds ctx_dft in a llama_context_ptr (fork:3488) and writes ctx_dft.get();
                // the port keeps a raw llama_context * (see port:1231 and the note at 5131).
                ctx_dft,
                ctx_dft_shared.get(),
            };
            for (size_t i_ctx = 0; i_ctx < std::size(contexts); ++i_ctx) {
                const llama_context * ctx = contexts[i_ctx];
                if (!ctx || std::find(contexts, contexts + i_ctx, ctx) != contexts + i_ctx) {
                    continue;
                }
                const auto breakdown = llama_get_live_memory_breakdown(ctx);
                for (const auto & [raw_buft, row] : breakdown) {
                    if (!raw_buft || ggml_backend_buft_is_host(raw_buft)) {
                        continue;
                    }

                    ggml_backend_buffer_type_t buft = raw_buft;
                    if (ggml_backend_buft_is_meta(buft)) {
                        if (ggml_backend_meta_buft_n_bufts(buft) != 1) {
                            complete = false;
                            continue;
                        }
                        buft = ggml_backend_meta_buft_simple_buft(buft, 0);
                    }

                    ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
                    const auto it = std::find_if(
                        cache_authority->live_device_domains.begin(),
                        cache_authority->live_device_domains.end(),
                        [device](const auto & binding) {
                            return binding.device == device;
                        });
                    if (it == cache_authority->live_device_domains.end()) {
                        complete = false;
                        continue;
                    }

                    auto & total = totals[size_t(
                        it - cache_authority->live_device_domains.begin())];
                    add_checked(total.attention,           row.attention);
                    add_checked(total.recurrent,           row.recurrent);
                    add_checked(total.recurrent_rollback,  row.recurrent_rollback);
                    add_checked(total.rolling_window_tape, row.rolling_window_tape);
                }
            }

            // complete is fixed after accumulation; when unavailable, every domain is marked
            // the same way, so the fail path is hoisted out of the per-domain gauge loop.
            if (!complete) {
                for (const auto & binding : cache_authority->live_device_domains) {
                    for (const auto category : {
                            llama_cache_acct_category::live_attention_state,
                            llama_cache_acct_category::live_recurrent_state,
                            llama_cache_acct_category::recurrent_rollback_planes,
                            llama_cache_acct_category::rolling_window_tape }) {
                        cache_authority->ledger.mark_unavailable(
                            category, binding.domain,
                            llama_cache_acct_measure::resident_allocated);
                    }
                    cache_authority->ledger.mark_producer_unavailable(
                        binding.domain, llama_cache_acct_producer::live_memory);
                }
                return false;
            }

            for (size_t i = 0; i < totals.size(); ++i) {
                const auto & domain = cache_authority->live_device_domains[i].domain;
                const auto & total  = totals[i];
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::live_attention_state, domain,
                    llama_cache_acct_measure::resident_allocated, total.attention);
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::live_recurrent_state, domain,
                    llama_cache_acct_measure::resident_allocated, total.recurrent);
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::recurrent_rollback_planes, domain,
                    llama_cache_acct_measure::resident_allocated,
                    total.recurrent_rollback);
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::rolling_window_tape, domain,
                    llama_cache_acct_measure::resident_allocated,
                    total.rolling_window_tape);
                if (certify) {
                    (void) cache_authority->ledger.certify_complete(
                        domain, llama_cache_acct_producer::live_memory);
                }
            }

            // Imported VMM pages are conservatively charged as transfer
            // staging until this complete observation accounts for them as
            // live attention state. Retire that bridge only after every gauge
            // has been published, avoiding both a gap and double-accounting.
            for (size_t i_ctx = 0; i_ctx < std::size(contexts); ++i_ctx) {
                llama_context * ctx = contexts[i_ctx];
                if (!ctx || std::find(contexts, contexts + i_ctx, ctx) !=
                                contexts + i_ctx) {
                    continue;
                }
                ctx->vbr_import_accounting_observed();
            }

            return complete;
        } catch (...) {
            for (const auto & binding : cache_authority->live_device_domains) {
                cache_authority->ledger.mark_producer_unavailable(
                    binding.domain, llama_cache_acct_producer::live_memory);
            }
            return false;
        }
    }

    // fork server-context.cpp:3802  (85 lines)
    bool cache_plan_configure_budget() noexcept {
        if (!cache_authority) {
            return true;
        }
        try {
            cache_authority->budget_devices.clear();
            cache_authority->budget_devices.reserve(
                cache_authority->live_device_domains.size());
            for (const auto & binding : cache_authority->live_device_domains) {
                llama_cache_budget_device_input input;
                input.backend_device = binding.device;
                input.domain = binding.domain;
                input.phys_state = llama_cache_budget_capacity_state::unavailable;
                input.compute_state = llama_cache_budget_capacity_state::known;
                input.cache_cap_state = llama_cache_budget_capacity_state::unbounded;
                input.reserve_provenance =
                    llama_cache_budget_reserve_provenance::configured;
                cache_authority->budget_devices.push_back(input);
            }

            bool complete = true;
            const auto add_checked = [&complete](uint64_t & dst, size_t value) {
                if (uint64_t(value) >
                    std::numeric_limits<uint64_t>::max() - dst) {
                    complete = false;
                    return;
                }
                dst += uint64_t(value);
            };

            const llama_context * contexts[] = {
                ctx_tgt,
                // FORK-DIFF: the fork holds ctx_dft in a llama_context_ptr (fork:3488) and writes ctx_dft.get();
                // the port keeps a raw llama_context * (see port:1231 and the note at 5131).
                ctx_dft,
                ctx_dft_shared.get(),
            };
            for (size_t i_ctx = 0; i_ctx < std::size(contexts); ++i_ctx) {
                const llama_context * ctx = contexts[i_ctx];
                if (!ctx ||
                    std::find(contexts, contexts + i_ctx, ctx) !=
                        contexts + i_ctx) {
                    continue;
                }
                for (const auto & [raw_buft, row] :
                        llama_get_memory_breakdown(ctx)) {
                    if (row.compute == 0 || !raw_buft ||
                        ggml_backend_buft_is_host(raw_buft)) {
                        continue;
                    }
                    // The context owner expands Meta scheduler buffers into
                    // physical rows. A Meta row here is incomplete accounting,
                    // not a quantity that can safely be split by topology.
                    if (ggml_backend_buft_is_meta(raw_buft)) {
                        complete = false;
                        continue;
                    }
                    const ggml_backend_dev_t device =
                        ggml_backend_buft_get_device(raw_buft);
                    auto it = std::find_if(
                        cache_authority->budget_devices.begin(),
                        cache_authority->budget_devices.end(),
                        [device](const auto & input) {
                            return input.backend_device == device;
                        });
                    if (it == cache_authority->budget_devices.end()) {
                        complete = false;
                        continue;
                    }
                    add_checked(it->current_compute_allocated, row.compute);
                }
            }

            for (auto & input : cache_authority->budget_devices) {
                input.configured_compute_reserve =
                    input.current_compute_allocated;
                if (!complete) {
                    input.compute_state =
                        llama_cache_budget_capacity_state::unavailable;
                }
            }
            return complete;
        } catch (...) {
            cache_authority->budget_devices.clear();
            return false;
        }
    }

    // fork server-context.cpp:4031  (187 lines)
    bool cache_plan_assemble_yield_candidates(
            const std::vector<server_retention_candidate> & catalog,
            std::vector<server_cache_yield_candidate> & candidates) noexcept {
        try {
            const auto resolve = [&](const server_retention_candidate & source,
                                     server_cache_yield_candidate & candidate,
                                     server_cache_lease_identity & identity,
                                     bool & identity_known) {
                switch (source.instance_key.kind) {
                    case common_retention_artifact_kind::host_entry: {
                        if (prompt_cache &&
                            source.instance_key.owner_slot == -1 &&
                            source.instance_key.instance != 0) {
                            const auto backing = std::find_if(
                                prompt_cache->states.begin(),
                                prompt_cache->states.end(),
                                [&](const auto & entry) {
                                    return server_retention_instance_key::
                                               for_host_entry(&entry).instance ==
                                           source.instance_key.instance;
                                });
                            if (backing == prompt_cache->states.end()) {
                                candidate.availability =
                                    server_retention_candidate_availability::
                                        backing_missing_or_stale;
                                break;
                            }
                            const auto coverage =
                                source.record.stamp.coverage_tokens;
                            identity_known =
                                coverage <= uint64_t(INT64_MAX) &&
                                server_cache_lease_build_identity(
                                    frontier_execution_identity,
                                    backing->adapter_config_key,
                                    backing->prompt.tokens,
                                    int64_t(coverage),
                                    identity);
                            for (const auto op : backing->release_ops()) {
                                if (op) {
                                    candidate.release_ops.push_back(op);
                                }
                            }
                            if (candidate.release_ops.empty()) {
                                candidate.availability =
                                    server_retention_candidate_availability::
                                        backing_missing_or_stale;
                            }
                        } else {
                            candidate.availability =
                                server_retention_candidate_availability::
                                    backing_missing_or_stale;
                        }
                        break;
                    }
                    case common_retention_artifact_kind::live_slot: {
                        const auto slot_it = std::find_if(
                            slots.begin(), slots.end(),
                            [&](const auto & candidate_slot) {
                                return candidate_slot.id ==
                                    source.instance_key.owner_slot &&
                                    server_retention_instance_key::for_slot(
                                        candidate_slot.id).instance ==
                                        source.instance_key.instance;
                            });
                        if (slot_it != slots.end()) {
                            const auto coverage =
                                source.record.stamp.coverage_tokens;
                            identity_known =
                                coverage <= uint64_t(INT64_MAX) &&
                                server_cache_lease_build_identity(
                                    frontier_execution_identity,
                                    lora_config_identity(slot_it->lora),
                                    slot_it->prompt.tokens,
                                    int64_t(coverage),
                                    identity);
                            candidate.availability = slot_it->is_processing()
                                ? server_retention_candidate_availability::
                                      in_flight_mutation
                                : server_retention_candidate_availability::
                                      available;
                        } else {
                            candidate.availability =
                                server_retention_candidate_availability::
                                    backing_missing_or_stale;
                        }
                        // Sequence removal does not free the fixed pooled KV
                        // backing. The live descriptor therefore contributes
                        // a known-zero transactional release; independently
                        // owned checkpoint children in the same slot manifest
                        // contribute their real operations.
                        candidate.has_unsupported_host_spill = false;
                        break;
                    }
                    case common_retention_artifact_kind::checkpoint: {
                        // retention accounting live members publish exact independently releasable
                        // payload operations. Host-entry checkpoint clones remain
                        // aggregate-owned and therefore stay fail-closed here.
                        candidate.availability =
                            server_retention_candidate_availability::
                                backing_missing_or_stale;
                        candidate.has_unsupported_host_spill = true;
                        if (source.instance_key.instance == 0) {
                            break;
                        }

                        for (const auto & owner : slots) {
                            if (source.instance_key.owner_slot != owner.id) {
                                continue;
                            }
                            const auto checkpoint = std::find_if(
                                owner.prompt.checkpoints.begin(),
                                owner.prompt.checkpoints.end(),
                                [&](const auto & item) {
                                    return server_retention_instance_key::
                                               for_checkpoint(
                                                   owner.id, &item).instance ==
                                           source.instance_key.instance;
                                });
                            if (checkpoint !=
                                owner.prompt.checkpoints.end()) {
                                identity_known =
                                    server_cache_lease_build_identity(
                                        frontier_execution_identity,
                                        lora_config_identity(owner.lora),
                                        owner.prompt.tokens,
                                        checkpoint->n_tokens,
                                        identity);
                                if (owner.is_processing()) {
                                    candidate.availability =
                                        server_retention_candidate_availability::
                                            in_flight_mutation;
                                } else if (!source.release_ops.empty()) {
                                    candidate.availability =
                                        server_retention_candidate_availability::
                                            available;
                                    candidate.has_unsupported_host_spill = false;
                                }
                                break;
                            }
                        }
                        if (!identity_known &&
                            source.instance_key.owner_slot == -1 &&
                            prompt_cache) {
                            for (const auto & owner : prompt_cache->states) {
                                const auto checkpoint = std::find_if(
                                    owner.prompt.checkpoints.begin(),
                                    owner.prompt.checkpoints.end(),
                                    [&](const auto & item) {
                                        return server_retention_instance_key::
                                                   for_checkpoint(
                                                       -1, &item).instance ==
                                               source.instance_key.instance;
                                    });
                                if (checkpoint !=
                                    owner.prompt.checkpoints.end()) {
                                    identity_known =
                                        server_cache_lease_build_identity(
                                            frontier_execution_identity,
                                            owner.adapter_config_key,
                                            owner.prompt.tokens,
                                            checkpoint->n_tokens,
                                            identity);
                                    break;
                                }
                            }
                        }
                        break;
                    }
                    case common_retention_artifact_kind::_count:
                        candidate.availability =
                            server_retention_candidate_availability::
                                backing_missing_or_stale;
                        break;
                }
            };
            if (!server_cache_yield_assemble(
                    catalog, cache_authority->leases, resolve, candidates)) {
                candidates.clear();
                return false;
            }

            return true;
        } catch (...) {
            candidates.clear();
            return false;
        }
    }

    // fork server-context.cpp:5471  (11 lines)
    bool vbr_refresh_exclusive_cells() noexcept {
        if (vbr_exclusive_cells_for_test) {
            return vbr_exclusive_cells_for_test->size() >= slots.size();
        }
        std::fill(vbr_exclusive_cells_by_seq.begin(),
                  vbr_exclusive_cells_by_seq.end(), 0);
        auto * memory = llama_get_memory(ctx_tgt);
        return memory && memory->vbr_accumulate_exclusive_cells(
            vbr_exclusive_cells_by_seq.data(),
            vbr_exclusive_cells_by_seq.size());
    }

    // fork server-context.cpp:5483  (6 lines)
    uint32_t vbr_exclusive_cells(int32_t slot_id) const noexcept {
        const auto & cells = vbr_exclusive_cells_for_test
            ? *vbr_exclusive_cells_for_test : vbr_exclusive_cells_by_seq;
        return slot_id >= 0 && size_t(slot_id) < cells.size()
            ? cells[size_t(slot_id)] : 0;
    }

    // Durability classifier and the clear door it authorises.  The port already
    // lands the 5471/5483 accessors and the 5579 eligibility filter in fork order;
    // these two sat in the gap between them.
    // fork server-context.cpp:5490-5509  (20 lines)
    bool vbr_idle_source_durable(
            const server_slot & slot) const noexcept {
        if (!params_base.vbr_prompt_cache) {
            return true;
        }
        if (!prompt_cache || slot.prompt.n_tokens() <= 0) {
            return false;
        }
        try {
            return automatic_vbr_cache_support(
                       slot.prompt.tokens, slot.lora,
                       slot.can_speculate()) ==
                    server_vbr_prompt_cache_support_status::supported &&
                prompt_cache->contains_vbr_frontier(
                    slot.prompt, frontier_execution_identity,
                    lora_config_identity(slot.lora));
        } catch (...) {
            return false;
        }
    }

    // fork server-context.cpp:5537-5543  (7 lines)
// fork 9l-d2b: the reclaim pool-state probe (fork server-context.cpp 5511-5535)
    llama_memory_vbr_state_data vbr_pool_state(
            uint32_t n_tokens_extra) noexcept {
        if (vbr_reclaim_cells_needed_for_test >= 0 &&
            vbr_exclusive_cells_for_test) {
            uint64_t remaining = 0;
            for (const auto & slot : slots) {
                if (slot.prompt.n_tokens() > 0 && slot.id >= 0 &&
                    size_t(slot.id) < vbr_exclusive_cells_for_test->size()) {
                    remaining += (*vbr_exclusive_cells_for_test)[size_t(slot.id)];
                }
            }
            vbr_pool_state_index_for_test++;
            const uint64_t reclaimed =
                vbr_reclaim_initial_cells_for_test >= remaining
                ? vbr_reclaim_initial_cells_for_test - remaining : 0;
            llama_memory_vbr_state_data result;
            result.deficit_raw = std::max<int64_t>(
                0, vbr_reclaim_cells_needed_for_test -
                    int64_t(std::min<uint64_t>(reclaimed, INT64_MAX)));
            result.bpv_if_degraded = result.deficit_raw > 0 ? 4.0 : 16.0;
            return result;
        }
        return llama_memory_vbr_state(
            llama_get_memory(ctx_tgt), -1, n_tokens_extra);
    }

    bool vbr_clear_idle_source(server_slot & slot) {
        if (params_base.vbr_prompt_cache) {
            return slot.prompt_clear_after_vbr_publication();
        }
        slot.prompt_clear(server_cache_destruction_reason::idle_reclaim);
        return true;
    }

// fork 9l-d2b: the reclaim victim clearer (fork server-context.cpp 5545-5554)
    bool vbr_clear_reclaim_victim(server_slot & slot) {
        if (vbr_reclaim_cells_needed_for_test >= 0) {
            if (slot.retention_obs) {
                slot.retention_obs->retire_slot(slot.id);
            }
            slot.prompt.clear();
            return true;
        }
        return vbr_clear_idle_source(slot);
    }

    // fork server-context.cpp:5579  (25 lines)
    bool vbr_idle_reclaim_eligible(
            const server_slot & slot,
            const vbr_live_retention_filter & filter) noexcept {
        const bool full_durable = slot.id >= 0 &&
            size_t(slot.id) < vbr_durable_sources_by_seq.size() &&
            vbr_durable_sources_by_seq[size_t(slot.id)] != 0;
        const bool stem_durable =
            filter.recovery != vbr_recovery_requirement::full_frontier &&
            slot.id >= 0 &&
            size_t(slot.id) < vbr_durable_stems_by_seq.size() &&
            vbr_durable_stems_by_seq[size_t(slot.id)] != 0;
        const bool durable = !params_base.vbr_prompt_cache ||
            full_durable || stem_durable ||
            filter.recovery ==
                vbr_recovery_requirement::none_for_admission_progress;
        const bool refused = filter.refused_by_seq && slot.id >= 0 &&
            size_t(slot.id) < filter.refused_by_seq->size() &&
            (*filter.refused_by_seq)[size_t(slot.id)] != 0;
        return slot.id != filter.except_id && !slot.is_processing() &&
            slot.prompt.n_tokens() > 0 && durable && !refused &&
            !slot.hard_lease_blocks_live_prefix() &&
            !queue_tasks.has_deferred_for_slot(slot.id) &&
            (filter.spec_tier < 0 ||
                slot.can_speculate() == (filter.spec_tier != 0));
    }

    // fork server-context.cpp:5605  (178 lines)
    bool vbr_prepare_retention_wave(
            const vbr_live_retention_filter & filter,
            bool advance_epoch = true,
            bool sample_durability = true) noexcept {
        auto * owner = live_retention_owner();
        vbr_retention_candidate_count = 0;
        vbr_retention_projection_epoch = 0;
        if (params_base.vbr_prompt_cache) {
            if (vbr_durable_sources_by_seq.size() != slots.size() ||
                vbr_durable_stems_by_seq.size() != slots.size()) {
                return false;
            }
            std::fill(
                vbr_durable_sources_by_seq.begin(),
                vbr_durable_sources_by_seq.end(), uint8_t(0));
            std::fill(
                vbr_durable_stems_by_seq.begin(),
                vbr_durable_stems_by_seq.end(), uint8_t(0));
            if (sample_durability) {
                size_t query_count = 0;
                for (auto & slot : slots) {
                    if (slot.prompt.n_tokens() <= 0) {
                        continue;
                    }
                    if (slot.id < 0 || size_t(slot.id) >= slots.size() ||
                        query_count >= vbr_idle_frontier_queries.size()) {
                        return false;
                    }
                    auto & query = vbr_idle_frontier_queries[query_count++];
                    query = {};
                    query.slot_id = slot.id;
                    query.prompt = &slot.prompt;
                    server_vbr_artifact_capture_status identity_status;
                    if (!build_capture_identity_fields(
                            slot, query.identity, identity_status)) {
                        query_count--;
                        continue;
                    }
                    query.expected_stem_artifact =
                        slot.vbr_idle_stem_source_valid
                            ? slot.vbr_idle_stem_host_artifact
                            : llama_cache_acct_artifact_id {};
                }
                if (query_count != 0 &&
                    (!prompt_cache || !prompt_cache->mark_vbr_frontiers(
                        vbr_idle_frontier_queries.data(), query_count))) {
                    return false;
                }
                for (size_t i = 0; i < query_count; ++i) {
                    const auto & query = vbr_idle_frontier_queries[i];
                    if (query.slot_id < 0 ||
                        size_t(query.slot_id) >= slots.size()) {
                        return false;
                    }
                    vbr_durable_sources_by_seq[size_t(query.slot_id)] =
                        query.durable ? 1 : 0;
                    vbr_durable_stems_by_seq[size_t(query.slot_id)] =
                        query.stem_durable ? 1 : 0;
                }
            }
        }
        if (!owner || vbr_retention_candidates.size() != slots.size() ||
            vbr_retention_candidate_by_seq.size() != slots.size() ||
            vbr_retention_seen_by_seq.size() != slots.size()) {
            return false;
        }
        std::fill(
            vbr_retention_seen_by_seq.begin(),
            vbr_retention_seen_by_seq.end(), uint8_t(0));
        if (advance_epoch) {
            if (!owner->begin_competition_wave()) {
                return false;
            }
            vbr_retention_projection_epoch = owner->competition_epoch_value();
        } else {
            // Slot selection is a read-only forecast. A new lineage is not
            // admitted until a later checkpoint/final publication, and that
            // transition owns its competition epoch. Quote at the current
            // epoch here so retries, overlapping launches, and appends cannot
            // predict an epoch they do not commit.
            vbr_retention_projection_epoch =
                owner->competition_epoch_value();
        }
        struct inventory_context {
            server_context_impl * self;
            const vbr_live_retention_filter * filter;
            size_t observed = 0;
            bool complete = true;
        } context { this, &filter, 0, true };
        const auto snapshot = owner->value_snapshots(
            &context,
            +[](void * opaque,
                const server_retention_value_snapshot & row) noexcept {
                auto & state = *static_cast<inventory_context *>(opaque);
                state.observed++;
                // Host aliases are part of the complete retention snapshot
                // and their coverage is already folded into each live row's
                // external_shared_coverage_tokens. Validate and count them,
                // but materialize only scheduler-owned live slots here.
                if (row.instance_key.kind !=
                        common_retention_artifact_kind::live_slot) {
                    return true;
                }
                if (row.kind !=
                        common_retention_artifact_kind::live_slot ||
                    row.instance_key.owner_slot < 0 ||
                    state.self->vbr_retention_candidate_count >=
                        state.self->vbr_retention_candidates.size()) {
                    state.complete = false;
                    return false;
                }
                const int32_t slot_id = row.instance_key.owner_slot;
                if (size_t(slot_id) >= state.self->slots.size() ||
                    state.self->slots[size_t(slot_id)].id != slot_id ||
                    state.self->vbr_retention_seen_by_seq[
                        size_t(slot_id)] != 0) {
                    state.complete = false;
                    return false;
                }
                auto * slot = &state.self->slots[size_t(slot_id)];
                if (slot->prompt.n_tokens() == 0) {
                    state.complete = false;
                    return false;
                }
                state.self->vbr_retention_seen_by_seq[
                    size_t(slot_id)] = 1;
                auto & candidate = state.self->vbr_retention_candidates[
                    state.self->vbr_retention_candidate_count++];
                candidate = {};
                candidate.slot_id = slot->id;
                candidate.artifact_id = row.artifact_id;
                candidate.stamp = row.stamp;
                candidate.lineage = row.lineage;
                candidate.external_shared_coverage_tokens =
                    row.external_shared_coverage_tokens;
                candidate.present = true;
                candidate.eligible = state.self->vbr_idle_reclaim_eligible(
                    *slot, *state.filter);
                return true;
            });
        if (!context.complete ||
            snapshot.status !=
                server_retention_value_snapshot_status::complete ||
            snapshot.size != context.observed) {
            return false;
        }
        for (const auto & slot : slots) {
            if (slot.prompt.n_tokens() == 0) {
                continue;
            }
            if (slot.id < 0 || size_t(slot.id) >=
                    vbr_retention_seen_by_seq.size() ||
                vbr_retention_seen_by_seq[size_t(slot.id)] == 0) {
                return false;
            }
        }
        if (vbr_retention_candidate_count == 0 ||
            !server_live_retention_prepare(
                vbr_retention_candidates.data(),
                vbr_retention_candidate_count,
                vbr_retention_projection_epoch)) {
            return false;
        }
        std::fill(
            vbr_retention_candidate_by_seq.begin(),
            vbr_retention_candidate_by_seq.end(), int32_t(-1));
        for (size_t i = 0; i < vbr_retention_candidate_count; ++i) {
            const int32_t slot_id = vbr_retention_candidates[i].slot_id;
            if (slot_id < 0 || size_t(slot_id) >=
                    vbr_retention_candidate_by_seq.size() ||
                vbr_retention_candidate_by_seq[size_t(slot_id)] != -1 ||
                i > size_t(INT32_MAX)) {
                return false;
            }
            vbr_retention_candidate_by_seq[size_t(slot_id)] = int32_t(i);
        }
        return true;
    }

    // fork server-context.cpp:5784  (120 lines)
    bool vbr_refresh_retention_wave(
            const vbr_live_retention_filter & filter) noexcept {
        auto * owner = live_retention_owner();
        if (!owner || vbr_retention_candidate_count == 0) {
            return false;
        }
        bool needs_exclusive_cells = false;
        for (size_t i = 0; i < vbr_retention_candidate_count; ++i) {
            auto & candidate = vbr_retention_candidates[i];
            if (candidate.slot_id < 0 || size_t(candidate.slot_id) >=
                    slots.size() ||
                slots[size_t(candidate.slot_id)].id != candidate.slot_id) {
                return false;
            }
            auto * slot = &slots[size_t(candidate.slot_id)];
            candidate.present = slot->prompt.n_tokens() > 0;
            candidate.eligible = candidate.present &&
                vbr_idle_reclaim_eligible(*slot, filter);
            candidate.marginal_cells = 0;
            needs_exclusive_cells |= candidate.eligible;
            candidate.external_shared_coverage_tokens = candidate.present
                ? UINT64_MAX : 0;
        }
        if (needs_exclusive_cells && !vbr_refresh_exclusive_cells()) {
            return false;
        }
        for (size_t i = 0; i < vbr_retention_candidate_count; ++i) {
            auto & candidate = vbr_retention_candidates[i];
            if (candidate.eligible) {
                candidate.marginal_cells =
                    vbr_exclusive_cells(candidate.slot_id);
                // Singleton pricing cannot account for a physical alias group whose
                // members release cells only together. Keep the historical order until
                // compound live projection owns that case.
                if (candidate.marginal_cells == 0) {
                    return false;
                }
            }
        }
        struct refresh_context {
            server_context_impl * self;
            size_t observed = 0;
            size_t live_rows = 0;
            bool complete = true;
        } context { this, 0, 0, true };
        std::fill(
            vbr_retention_seen_by_seq.begin(),
            vbr_retention_seen_by_seq.end(), uint8_t(0));
        const auto snapshot = owner->value_snapshots(
            &context,
            +[](void * opaque,
                const server_retention_value_snapshot & row) noexcept {
                auto & state = *static_cast<refresh_context *>(opaque);
                state.observed++;
                if (row.instance_key.kind !=
                        common_retention_artifact_kind::live_slot) {
                    return true;
                }
                if (row.kind != common_retention_artifact_kind::live_slot) {
                    state.complete = false;
                    return false;
                }
                const int32_t slot_id = row.instance_key.owner_slot;
                if (slot_id < 0 || size_t(slot_id) >=
                        state.self->vbr_retention_candidate_by_seq.size()) {
                    state.complete = false;
                    return false;
                }
                const int32_t candidate_index =
                    state.self->vbr_retention_candidate_by_seq[
                        size_t(slot_id)];
                if (candidate_index < 0 || size_t(candidate_index) >=
                        state.self->vbr_retention_candidate_count ||
                    state.self->vbr_retention_seen_by_seq[
                        size_t(slot_id)] != 0) {
                    state.complete = false;
                    return false;
                }
                auto & found = state.self->vbr_retention_candidates[
                    size_t(candidate_index)];
                if (!found.present || found.slot_id != slot_id ||
                    found.artifact_id != row.artifact_id ||
                    found.stamp.pool != row.stamp.pool ||
                    found.stamp.stable_id != row.stamp.stable_id ||
                    found.stamp.lineage_id != row.stamp.lineage_id ||
                    found.stamp.recency_ordinal !=
                        row.stamp.recency_ordinal ||
                    found.stamp.coverage_tokens !=
                        row.stamp.coverage_tokens ||
                    found.lineage != row.lineage) {
                    state.complete = false;
                    return false;
                }
                found.external_shared_coverage_tokens =
                    row.external_shared_coverage_tokens;
                state.self->vbr_retention_seen_by_seq[
                    size_t(slot_id)] = 1;
                state.live_rows++;
                return true;
            });
        size_t present = 0;
        for (size_t i = 0; i < vbr_retention_candidate_count; ++i) {
            const auto & candidate = vbr_retention_candidates[i];
            if (candidate.present) {
                present++;
                if (candidate.slot_id < 0 || size_t(candidate.slot_id) >=
                        vbr_retention_seen_by_seq.size() ||
                    vbr_retention_seen_by_seq[
                        size_t(candidate.slot_id)] == 0 ||
                    candidate.external_shared_coverage_tokens == UINT64_MAX) {
                    return false;
                }
            }
        }
        return context.complete &&
            snapshot.status ==
                server_retention_value_snapshot_status::complete &&
            snapshot.size == context.observed &&
            context.live_rows == present;
    }

    // fork server-context.cpp:5905  (15 lines)
    server_slot * vbr_oldest_idle_slot(
            const vbr_live_retention_filter & filter) noexcept {
        server_slot * oldest = nullptr;
        for (auto & slot : slots) {
            if (!vbr_idle_reclaim_eligible(slot, filter)) {
                continue;
            }
            if (!oldest ||
                std::tie(slot.t_last_used, slot.id) <
                    std::tie(oldest->t_last_used, oldest->id)) {
                oldest = &slot;
            }
        }
        return oldest;
    }

    // fork server-context.cpp:5921  (23 lines)
    server_slot * vbr_retention_victim(
            bool & learned_available,
            const vbr_live_retention_filter & filter) noexcept {
        if (learned_available &&
            !vbr_refresh_retention_wave(filter)) {
            learned_available = false;
        }
        if (learned_available) {
            const auto projection = server_live_retention_project_prepared(
                vbr_retention_candidates.data(),
                vbr_retention_candidate_count,
                vbr_retention_projection_epoch);
            if (projection.complete) {
                auto * selected = find_slot_by_id(projection.slot_id);
                if (selected &&
                    vbr_idle_reclaim_eligible(*selected, filter)) {
                    return selected;
                }
            }
            learned_available = false;
        }
        return vbr_oldest_idle_slot(filter);
    }

    // fork server-context.cpp:5949  (36 lines)
    server_slot * vbr_retention_victim_prepared(
            bool & learned_available,
            const vbr_live_retention_filter & filter) noexcept {
        if (learned_available) {
            for (size_t i = 0; i < vbr_retention_candidate_count; ++i) {
                auto & candidate = vbr_retention_candidates[i];
                if (candidate.slot_id < 0 ||
                    size_t(candidate.slot_id) >= slots.size() ||
                    slots[size_t(candidate.slot_id)].id !=
                        candidate.slot_id) {
                    learned_available = false;
                    break;
                }
                candidate.eligible = candidate.present &&
                    vbr_idle_reclaim_eligible(
                        slots[size_t(candidate.slot_id)], filter);
            }
        }
        if (learned_available) {
            const auto projection = server_live_retention_project_prepared(
                vbr_retention_candidates.data(),
                vbr_retention_candidate_count,
                vbr_retention_projection_epoch);
            if (projection.complete) {
                auto * selected = find_slot_by_id(projection.slot_id);
                if (selected &&
                    vbr_idle_reclaim_eligible(*selected, filter)) {
                    return selected;
                }
                learned_available = false;
            } else if (projection.candidate_count != 0) {
                learned_available = false;
            }
        }
        return vbr_oldest_idle_slot(filter);
    }

    // The reclaim wave and the recovery reset that calls it.  Both sit between the
    // port's 5949 and 8607 blocks, so they are placed there to keep fork order.
    // fork server-context.cpp:5986-6018  (33 lines, includes the fork's doc comment)
    // Dynamic VBR clear-only reclaim of idle slots. With automatic host
    // caching enabled, an exact immutable frontier is required before clear;
    // otherwise the older clear-only behavior remains unchanged. Never touch
    // a processing slot or one pinned by an explicitly deferred task.
    int vbr_clear_idle_slots(int except_id, const char * reason) {
        int cleared = 0;
        for (auto & s : slots) {
            if (s.id == except_id || s.is_processing() || s.prompt.n_tokens() == 0) {
                continue;
            }
            // Approved guard, not destruction-authority certification: the legacy eraser
            // simply does not consider a hard-leased live prefix.
            if (s.hard_lease_blocks_live_prefix()) {
                SLT_INF(s, "vbr reclaim (%s): kept — hard lease seals the live prefix\n", reason);
                continue;
            }
            if (!vbr_idle_source_durable(s)) {
                SLT_INF(s, "vbr reclaim (%s): kept — no durable VBR frontier\n",
                        reason);
                continue;
            }
            if (queue_tasks.has_deferred_for_slot(s.id)) {
                SLT_INF(s, "vbr reclaim (%s): kept — a deferred id_slot task pins this slot\n", reason);
                continue;
            }
            SLT_WRN(s, "vbr reclaim (%s): clearing %d cached tokens\n", reason, (int) s.prompt.n_tokens());
            const bool removed = vbr_clear_idle_source(s);
            if (removed) {
                cleared++;
            }
        }
        return cleared;
    }

// fork 9l-d2b: reclaim-before-degrade (fork server-context.cpp 6020-6077)
    // reclaim-before-degrade: when the incoming work's projected footprint would push the
    // degrade ladder BELOW the quality floor, pay with idle caches instead of tiers. Above the
    // floor the trade inverts — a within-band degrade is near-lossless and cheaper than
    // destroying another conversation's re-prefillable cache, so caches are kept. Policy reads
    // deficit_raw only: page-exact and free-VRAM independent, so WHETHER a cache is erased
    // never depends on driver jitter or co-tenants (the cache's own live clamp still handles
    // those with tier degrades).
    void vbr_reclaim_before_degrade(int except_id, uint32_t n_tokens_extra, const char * reason) {
        if (!server_vbr_dynamic_active(params_base) || params_base.vbr_reclaim_floor_bpv <= 0.0f) {
            return;
        }
        auto st = vbr_pool_state(n_tokens_extra);
        if (st.deficit_raw <= 0 || st.bpv_if_degraded >= (double) params_base.vbr_reclaim_floor_bpv) {
            return;
        }
        if (params_base.vbr_prompt_cache) {
            std::fill(
                vbr_reclaim_refused_by_seq.begin(),
                vbr_reclaim_refused_by_seq.end(), uint8_t(0));
        }
        const vbr_live_retention_filter filter {
            except_id, -1,
            params_base.vbr_prompt_cache
                ? &vbr_reclaim_refused_by_seq : nullptr,
        };
        bool learned_available = vbr_prepare_retention_wave(filter);
        int cleared = 0;
        while (st.deficit_raw > 0 &&
               st.bpv_if_degraded <
                    (double) params_base.vbr_reclaim_floor_bpv) {
            auto * victim = vbr_retention_victim(
                learned_available, filter);
            if (!victim) {
                break;
            }
            SLT_WRN(*victim,
                "vbr reclaim (%s): clearing %d cached tokens (%s order)\n",
                reason, (int) victim->prompt.n_tokens(),
                learned_available ? "decayed-value" : "oldest-idle");
            if (!vbr_clear_reclaim_victim(*victim)) {
                SLT_INF(*victim,
                    "vbr reclaim (%s): kept — full-slot lease refused displacement\n",
                    reason);
                if (victim->id >= 0 &&
                    size_t(victim->id) <
                        vbr_reclaim_refused_by_seq.size()) {
                    vbr_reclaim_refused_by_seq[
                        size_t(victim->id)] = 1;
                    continue;
                }
                break;
            }
            cleared++;
            st = vbr_pool_state(n_tokens_extra);
        }
        if (cleared > 0) {
            SRV_WRN("vbr reclaim (%s): cleared %d idle slot(s), residual deficit %.2f MiB, projected %.3f bpv (floor %.3f)\n",
                    reason, cleared, st.deficit_raw / 1024.0 / 1024.0,
                    st.bpv_if_degraded,
                    (double) params_base.vbr_reclaim_floor_bpv);
        }
    }

    // fork server-context.cpp:6083-6134  (52 lines, includes the fork's doc comment)
    // reset-on-low-LCP (dynamic VBR): tiers are per-tensor and promotion cannot cross the tap
    // boundary, so the ONLY full-quality recovery is the lossless empty-cache reset. When a
    // DEGRADED conversation keeps a token-trivial prefix anyway (rolling-window agent
    // harnesses rewrite mid-context every turn), trade the prefix for the reset: clear the
    // idle slots (recovery reclaim — deliberately NOT deficit-gated: budget slack is exactly
    // the state that starves the pressure path, and worthless idle caches are the only
    // obstacle to recovery that benefits every stream), and if the pool then holds nothing
    // but this slot's cells, drop n_past to 0 — the full re-prefill re-enters at the entry
    // tier, so turn-N cache quality equals turn-1. Returns the (possibly zeroed) n_past.
    int vbr_reset_on_low_lcp(
            server_slot & slot,
            int n_past,
            server_cache_destruction_admission & admission) {
        if (!server_vbr_dynamic_active(params_base) || params_base.vbr_reset_keep_frac <= 0.0f) {
            return n_past;
        }
        const int n_prompt = slot.task->n_tokens();
        if (n_prompt <= 0 || (float) n_past >= params_base.vbr_reset_keep_frac * (float) n_prompt) {
            return n_past;
        }
        llama_memory_t mem = llama_get_memory(ctx_tgt);
        auto st = llama_memory_vbr_state(mem, slot.id, 0);
        if (st.cursor < 2) {
            return n_past; // pristine or one transient step — a full re-prefill buys ~nothing
        }
        // Approved guard, not a destructive capability. The range is
        // exactly the prefix this recovery path would discard.
        if (slot.hard_lease_blocks_live_range(
                0, static_cast<uint64_t>(std::max(0, n_past)))) {
            SLT_INF(slot, "%s", "vbr reset blocked: hard lease seals the reusable prefix\n");
            return n_past;
        }
        vbr_clear_idle_slots(slot.id, "reset-recovery");
        st = llama_memory_vbr_state(mem, slot.id, 0);
        if (st.used_cells_other > 0) {
            SLT_INF(slot, "vbr reset blocked: %u used cells belong to other sequences (pinned or processing slots)\n",
                    st.used_cells_other);
            return n_past;
        }
        SLT_WRN(slot, "vbr reset: cursor %d and only %d/%d prompt tokens reusable (< %.2f) — dropping the prefix; "
                "the full re-prefill re-enters at the entry tier\n",
                st.cursor, n_past, n_prompt, (double) params_base.vbr_reset_keep_frac);
        admission = slot.observe_live_range_drop(
            server_cache_destruction_reason::low_lcp_reset, true);
        if (admission.execution !=
                server_cache_destruction_execution::pass_through) {
            return n_past;
        }
        (void) slot.checkpoint_drop_joined_impl(
            slot.prompt.checkpoints.begin(), slot.prompt.checkpoints.end());
        return 0;
    }

    // fork server-context.cpp:8607  (20 lines)
    common_cache_plan_selection cache_plan_origin_for(
            const server_task & task,
            uint64_t lcp) const noexcept {
        if (task.id_slot != -1) {
            return common_cache_plan_selection::by_id;
        }
        const float sim = task.tokens.empty()
            ? 0.0f : float(lcp) / float(task.tokens.size());
        // Keep in lockstep with the shipped tier boundaries below. A weak
        // token prefix is not conversation identity: unrelated requests can
        // share a system prompt, and routing one onto that slot destroys the
        // longer live lineage. Only the established strict-similarity tier
        // selects by prompt overlap; everything else uses the retention-aware
        // LRU tier.
        if (common_cache_plan_strict_similarity(
                sim, slot_prompt_similarity)) {
            return common_cache_plan_selection::similarity;
        }
        return common_cache_plan_selection::lru;
    }

    // fork server-context.cpp:8628  (6 lines)
    bool cache_plan_target_in_domain(
            const server_task & task,
            const server_slot & slot) const noexcept {
        return !slot.is_processing() &&
               (task.id_slot == -1 || slot.id == task.id_slot);
    }

    // fork server-context.cpp:8635  (30 lines)
    bool cache_plan_inventory_live_rows(
            const server_task & task,
            const std::vector<uint64_t> & slot_lcps,
            common_cache_plan_record & rec) {
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto & slot = slots[i];
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }
            const auto origin = cache_plan_origin_for(task, slot_lcps[i]);
            auto * row = rec.find_or_add(
                common_cache_plan_provider::live_slot, slot.id,
                COMMON_CACHE_PLAN_PHASE_LRU, slot.id, origin);
            if (!row) {
                return false;
            }
            server_cache_plan_apply_live(row, server_cache_plan_evaluate_live(
                slot.is_processing(), !slot.prompt.tokens.empty(), slot_lcps[i],
                task.tokens.size(), slot.prompt.tokens.size()));
            // The shipped similarity scan visits below-threshold rows too;
            // authority membership is the first tier that can select the target.
            row->origin_tier = origin;
            row->t_last_used_us = llama_cache_acct_value::measured(
                uint64_t(std::max<int64_t>(slot.t_last_used, 0)));
            row->spec_capable = slot.can_speculate();
            row->spec_capable_known = true;
        }
        rec.note_inventory_complete(common_cache_plan_provider::live_slot);
        return true;
    }

    // fork server-context.cpp:8666  (97 lines)
    bool cache_plan_inventory_host_rows(
            const server_task & task,
            const std::vector<uint64_t> & slot_lcps,
            const std::string & incoming_adapter,
            bool recurrent,
            common_cache_plan_record & rec,
            cache_plan_host_source_registry & source_registry) {
        if (!prompt_cache) {
            rec.note_inventory_complete(common_cache_plan_provider::host_cache_entry);
            return true;
        }
        for (auto & state : prompt_cache->states) {
            // VBR nodes use the typed VBR restore selector, not the legacy
            // fixed-state inventory. Skip them before fixed source-ID allocation
            // and long-prefix work so they cannot exhaust that inventory.
            if (!state.payload.fixed_state_restorable()) {
                continue;
            }
            int32_t host_source_id = -1;
            if (!source_registry.get(
                    *prompt_cache, state, host_source_id)) {
                rec.inventory_states[size_t(
                    common_cache_plan_provider::host_cache_entry)] =
                    common_cache_plan_inventory_state::overflowed;
                return false;
            }
            const uint64_t lcp = state.prompt.tokens.get_common_prefix(task.tokens);
            const auto host_eval = server_cache_plan_evaluate_host(
                state.payload.fixed_state_restorable(),
                state.adapter_config_key == incoming_adapter,
                lcp, task.tokens.size(), state.prompt.tokens.size(),
                state.payload.size());
            std::vector<server_cache_plan_checkpoint_evaluation> checkpoint_evals;
            checkpoint_evals.reserve(state.prompt.checkpoints.size());
            for (const auto & checkpoint : state.prompt.checkpoints) {
                checkpoint_evals.push_back(server_cache_plan_evaluate_checkpoint(
                    !checkpoint.empty(), true, recurrent, true,
                    checkpoint.pos_min, checkpoint.pos_max,
                    int64_t(lcp), 0, checkpoint.size()));
            }

            for (size_t target_i = 0; target_i < slots.size(); ++target_i) {
                const auto & candidate_target = slots[target_i];
                if (!cache_plan_target_in_domain(task, candidate_target)) {
                    continue;
                }
                const auto origin = cache_plan_origin_for(task, slot_lcps[target_i]);
                auto * host = rec.find_or_add(
                    common_cache_plan_provider::host_cache_entry, host_source_id,
                    COMMON_CACHE_PLAN_PHASE_HOST_SCAN, candidate_target.id, origin);
                if (!host) {
                    return false;
                }
                host->payload_kind =
                    server_cache_plan_payload_kind(state.payload.kind());
                server_cache_plan_apply_host(host, host_eval);

                int32_t checkpoint_ordinal = 0;
                for (const auto & eval : checkpoint_evals) {
                    const int32_t checkpoint_source =
                        server_cache_plan_host_checkpoint_source_id(
                            host_source_id, checkpoint_ordinal++);
                    if (checkpoint_source < 0) {
                        rec.inventory_states[size_t(
                            common_cache_plan_provider::live_context_checkpoint)] =
                            common_cache_plan_inventory_state::overflowed;
                        return false;
                    }
                    auto * checkpoint = rec.find_or_add(
                        common_cache_plan_provider::live_context_checkpoint,
                        checkpoint_source,
                        COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,
                        candidate_target.id, origin);
                    if (!checkpoint) {
                        return false;
                    }
                    server_cache_plan_apply_checkpoint(checkpoint, eval);
                    if (server_cache_plan_viable(host_eval.reason) &&
                        server_cache_plan_viable(eval.reason)) {
                        checkpoint->component_only = true;
                        checkpoint->dependent_host_source_id = host_source_id;
                        auto * chain = rec.add_chain(
                            common_cache_plan_provider::host_cache_entry,
                            int32_t(host - rec.inventory.data()),
                            int32_t(checkpoint - rec.inventory.data()));
                        if (!chain) {
                            return false;
                        }
                        chain->disposition =
                            common_cache_plan_disposition::valid_not_chosen_cost;
                    }
                }
            }
        }
        rec.note_inventory_complete(common_cache_plan_provider::host_cache_entry);
        return true;
    }

    // fork server-context.cpp:8764  (56 lines)
    bool cache_plan_inventory_checkpoint_rows(
            const server_task & task,
            const std::vector<uint64_t> & slot_lcps,
            bool recurrent,
            common_cache_plan_record & rec) {
        // Complete diagnostic scan. The shipped restore keeps its reverse find_if
        // and its shipped-visited transport below.
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto & candidate_target = slots[i];
            if (!cache_plan_target_in_domain(task, candidate_target)) {
                continue;
            }
            const llama_pos pos_next = task.tokens.pos_next(slot_lcps[i]);
            const bool has_new_tokens = slot_lcps[i] < task.tokens.size();
            const llama_pos pos_min_threshold =
                std::max<llama_pos>(
                    0, pos_next - n_swa - (has_new_tokens ? 0 : 1));
            const auto vbr_now = llama_memory_vbr_state(
                llama_get_memory(ctx_tgt), candidate_target.id, 0);
            std::string adapter_identity;
            bool adapter_known = true;
            try {
                adapter_identity = lora_config_identity(candidate_target.lora);
            } catch (...) {
                adapter_known = false;
            }
            int32_t checkpoint_ordinal = 0;
            for (const auto & checkpoint : candidate_target.prompt.checkpoints) {
                const bool checkpoint_lineage_matches =
                    !recurrent ||
                    common_prompt_checkpoint_lineage_matches(checkpoint, vbr_now);
                const bool frontier_current =
                    !candidate_target.frontier_ratchet_flipped ||
                    (adapter_known && checkpoint_frontier_is_current(
                        candidate_target, checkpoint, adapter_identity));
                auto * row = rec.find_or_add(
                    common_cache_plan_provider::live_context_checkpoint,
                    checkpoint_ordinal++, COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,
                    candidate_target.id,
                    cache_plan_origin_for(task, slot_lcps[i]));
                if (!row) {
                    return false;
                }
                server_cache_plan_apply_checkpoint(row,
                    server_cache_plan_evaluate_checkpoint(
                        !checkpoint.empty(), frontier_current, recurrent,
                        checkpoint_lineage_matches, checkpoint.pos_min,
                        checkpoint.pos_max, pos_next, pos_min_threshold,
                        checkpoint.size()));
            }
        }
        rec.note_inventory_complete(
            common_cache_plan_provider::live_context_checkpoint);
        rec.authority_inventory_complete = true;
        return true;
    }

    // fork server-context.cpp:8821  (21 lines)
    bool cache_plan_inventory_cold_rows(
            const server_task & task,
            const std::vector<uint64_t> & slot_lcps,
            common_cache_plan_record & rec) {
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto & candidate_target = slots[i];
            if (!cache_plan_target_in_domain(task, candidate_target)) {
                continue;
            }
            auto * cold = rec.find_or_add(
                common_cache_plan_provider::cold_replay,
                COMMON_CACHE_PLAN_SOURCE_AGGREGATE, 0, candidate_target.id,
                cache_plan_origin_for(task, slot_lcps[i]));
            if (!cold) {
                return false;
            }
            cold->accept();
        }
        rec.note_inventory_complete(common_cache_plan_provider::cold_replay);
        return true;
    }

    // fork server-context.cpp:8879  (9 lines)
    static void cache_plan_host_restore_failed(
            server_slot & slot,
            common_cache_plan_record * rec) {
        if (rec) {
            rec->restore_attempt_failed = true;
        }
        slot.mandatory_recovery_reset(
            server_cache_destruction_reason::restore_failure);
    }

    // fork server-context.cpp:8896  (352 lines)
    cache_plan_stage1_selection cache_plan_select_before_mutation(
            const server_task & task,
            bool observe,
            bool is_preflight) {
        server_slot * ret = nullptr;

        bool update_cache = false;
        bool selection_deferred_busy = false;

        // best similarity seen even BELOW the threshold — feeds the vbr route-home tier and the
        // LRU log line (a hopping conversation was undiagnosable: "selected by LRU" never said
        // how close the rejected candidates came)
        float sim_best_any = 0;

        // Begin the multi-stage cache-plan record before any selection-side mutation. The
        // selection scan below is read-only; the record exists only under --cache-debug, and
        // every row write from here on is noexcept (fixed-array record), so the shipped path
        // cannot throw through the observer. An allocation fault is swallowed and counted.
        std::unique_ptr<common_cache_plan_record> plan_rec;
        if (observe) {
            try {
                plan_rec = std::make_unique<common_cache_plan_record>();
                plan_rec->id_task = task.id;
                // opaque identity evidence from already-computed keys (never raw values);
                // identities the server has not computed stay typed unknown
                plan_rec->identity.model_digest = llama_cache_acct_value::measured(
                    std::hash<std::string>{}(model_name));
                plan_rec->identity.execution_digest = llama_cache_acct_value::measured(
                    std::hash<std::string>{}(frontier_execution_identity));
                // SAMPLED-PREFIX TELEMETRY (not an identity): FNV-1a over at most the
                // prompt's first 64 token ids + that sampled count. Useful for OFFLINE
                // clustering of workload variants; it is NOT an exact family key and must
                // never drive a shipped decision — prompts sharing 64 leading tokens merge
                // deliberately, the folded count saturates at the window, and FNV collides.
                // Opaque + content-minimizing like every identity digest.
                {
                    uint64_t h = 1469598103934665603ull;
                    const auto fold = [&h](uint64_t v) {
                        h = (h ^ v) * 1099511628211ull;
                    };
                    const size_t n = std::min<size_t>(task.tokens.size(), 64);
                    for (size_t i = 0; i < n; i++) {
                        fold((uint64_t)(uint32_t) task.tokens[i]);
                    }
                    fold((uint64_t) n);
                    plan_rec->identity.prefix_token_digest = llama_cache_acct_value::measured(h);
                }
            } catch (...) {
                if (!is_preflight && cache_plan_obs) {
                    cache_plan_obs->shadow_unavailable++;
                }
            }
        }

        // one observer row per slot a loop classifies (merge key = slot id across all
        // loops); nullptr when the observer is off or the inventory overflowed [observer]
        const auto slot_row = [&](const server_slot & slot, uint8_t phase) {
            return plan_rec
                ? plan_rec->find_or_add(
                      common_cache_plan_provider::live_slot, slot.id, phase, slot.id,
                      common_cache_plan_origin_for_phase(phase))
                : (common_cache_plan_candidate *) nullptr;
        };

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                if (!is_preflight) {
                    SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
                }
                if (plan_rec) {
                    plan_rec->selection = common_cache_plan_selection::by_id;
                    // forced selection carries no reuse verdict, but the slot WAS observed —
                    // its row exists with disposition unavailable; finalize upgrades the
                    // selected row from realized prefix reuse
                    slot_row(*ret, COMMON_CACHE_PLAN_PHASE_BY_ID);
                }
            }
        }

        // find the slot that has at least n% prompt similarity.
        // Observer transport [observer, noexcept]: one row per slot this loop classifies (merge key
        // = slot id across all three loops); every evaluated survivor starts as a cost loser
        // and the shipped winner is promoted after the scan. A nullptr row (inventory
        // overflow) is skipped — the shipped scan never changes.
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            // shipped logic written ONCE; the unobserved instantiation compiles every
            // observer block away: literal zero work with --cache-debug off.
            const auto similarity_scan = [&](auto obs_c) {
                constexpr bool Obs = decltype(obs_c)::value;
                for (server_slot & slot : slots) {
                    if (task.id_slot != -1 && slot.id != task.id_slot) {
                        continue; // id filter, not a classification: no row
                    }

                    [[maybe_unused]] common_cache_plan_candidate * row = nullptr;
                    if constexpr (Obs) {
                        row = slot_row(slot, COMMON_CACHE_PLAN_PHASE_SIMILARITY);
                    }

                    // skip the slot if it is not available
                    if (slot.is_processing()) {
                        if constexpr (Obs) {
                            if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY); }
                        }
                        continue;
                    }

                    const auto & tokens = slot.prompt.tokens;

                    // skip the slot if it does not contains cached tokens
                    if (tokens.empty()) {
                        if constexpr (Obs) {
                            // truly stateless slot: unavailable, never coverage
                            if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE); }
                        }
                        continue;
                    }

                    // fraction of the Longest Common Prefix length with respect to the input prompt length
                    const size_t lcp_cur = tokens.get_common_prefix(task.tokens);
                    const float  sim_cur = float(lcp_cur) / task.tokens.size();

                    sim_best_any = std::max(sim_best_any, sim_cur);

                    if constexpr (Obs) {
                        server_cache_plan_apply_live(row,
                            server_cache_plan_evaluate_live(
                                false, true, lcp_cur, task.tokens.size()));
                    }

                    // select the current slot if the criteria match
                    if (sim_cur > f_sim_best &&
                        common_cache_plan_strict_similarity(
                            sim_cur, slot_prompt_similarity)) {
                        f_sim_best = sim_cur;

                        ret = &slot;
                    }
                }
            };
            plan_rec ? similarity_scan(std::true_type{}) : similarity_scan(std::false_type{});

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    if (!is_preflight) {
                        SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                                f_sim_best, slot_prompt_similarity, f_keep);
                    }
                    if (plan_rec) {
                        plan_rec->selection = common_cache_plan_selection::similarity;
                        if (auto * row = slot_row(*ret, COMMON_CACHE_PLAN_PHASE_SIMILARITY)) {
                            row->accept();
                            row->sim    = f_sim_best; row->sim_known    = true;
                            row->f_keep = f_keep;   row->f_keep_known = true;
                        }
                    }
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }

            if (plan_rec) {
                plan_rec->sim_best_any       = sim_best_any;
                plan_rec->sim_best_any_known = true;
            }
        }

        // find the slot that has been least recently used
        // prefer spec-capable (DFlash) slots so requests get speculative decoding
        if (ret == nullptr) {
            int64_t t_last = -1;
            bool retention_value_selected = false;

            const auto lru_scan = [&](auto obs_c) {
                constexpr bool Obs = decltype(obs_c)::value;
                for (server_slot & slot : slots) {
                    [[maybe_unused]] common_cache_plan_candidate * row = nullptr;
                    if constexpr (Obs) {
                        row = slot_row(slot, COMMON_CACHE_PLAN_PHASE_LRU);
                    }

                    // skip the slot if it is not available
                    if (slot.is_processing()) {
                        if constexpr (Obs) {
                            if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY); }
                        }
                        continue;
                    }

                    if constexpr (Obs) {
                        if (row) {
                            // scalars THIS loop computes; reuse verdicts (if any) came from the
                            // reuse-evaluating loops via the merged row
                            row->t_last_used_us = llama_cache_acct_value::measured(
                                (uint64_t) std::max<int64_t>(slot.t_last_used, 0));
                            row->spec_capable       = slot.can_speculate();
                            row->spec_capable_known = true;
                        }
                    }

                    // strongly prefer spec-capable slots: pick a spec slot over a non-spec
                    // slot regardless of LRU, then use LRU within the same capability tier
                    const bool curr_spec = ret && ret->can_speculate();
                    const bool slot_spec = slot.can_speculate();
                    if (!ret ||
                        (slot_spec && !curr_spec) ||
                        (slot_spec == curr_spec && slot.t_last_used < t_last)) {
                        t_last = slot.t_last_used;
                        ret = &slot;
                    }
                }
            };
            plan_rec ? lru_scan(std::true_type{}) : lru_scan(std::false_type{});

            // A genuinely new stream has no reusable home. Once its legacy capability tier
            // has no empty slot, choose which resident conversation pays for the slot by the
            // shared decayed value/cell policy. Fixed and dynamic-VBR KV use the same lineage
            // authority and exact exclusive-cell measure here.
            // Both preflight and selection quote the current epoch without mutating it.
            // A later checkpoint/final publication owns any distinct-lineage admission
            // epoch; retries, appends, and overlapping launches cannot age sleeping
            // conversations speculatively. Incomplete/domain-ambiguous evidence uses
            // the protection-aware historical order within the same capability tier.
            if (ret != nullptr && live_retention_owner() != nullptr &&
                ret->prompt.n_tokens() > 0) {
                const auto empty_in_tier = [&](bool spec_tier) {
                    server_slot * empty = nullptr;
                    for (auto & slot : slots) {
                        if (slot.is_processing() ||
                            slot.prompt.n_tokens() != 0 ||
                            slot.can_speculate() != spec_tier) {
                            continue;
                        }
                        if (!empty ||
                            std::tie(slot.t_last_used, slot.id) <
                                std::tie(empty->t_last_used, empty->id)) {
                            empty = &slot;
                        }
                    }
                    return empty;
                };

                const bool preferred_tier = ret->can_speculate();
                server_slot * selected = empty_in_tier(preferred_tier);
                if (!selected) {
                    selected = empty_in_tier(!preferred_tier);
                }
                bool selected_by_value = false;
                if (!selected) {
                    const vbr_live_retention_filter broad_filter {
                        -1, -1, nullptr,
                        vbr_recovery_requirement::none_for_admission_progress,
                    };
                    bool learned = vbr_prepare_retention_wave(
                        broad_filter, false);
                    if (learned &&
                        !vbr_refresh_retention_wave(broad_filter)) {
                        learned = false;
                    }
                    const auto select_tier = [&](bool spec_tier,
                            vbr_recovery_requirement recovery) {
                        const vbr_live_retention_filter filter {
                            -1, spec_tier ? 1 : 0, nullptr, recovery,
                        };
                        auto * tier = vbr_retention_victim_prepared(
                            learned, filter);
                        selected_by_value = learned && tier != nullptr;
                        return tier;
                    };

                    selected = select_tier(
                        preferred_tier,
                        vbr_recovery_requirement::full_or_stem);
                    if (!selected) {
                        // Speculation is a preference, not a hard partition.
                        selected = select_tier(
                            !preferred_tier,
                            vbr_recovery_requirement::full_or_stem);
                    }
                    if (!selected) {
                        // Host-cache capacity is protection authority, not
                        // scheduler admission authority. Only after neither
                        // tier has a recoverable victim may this final progress
                        // policy admit an undurable one.
                        selected = select_tier(
                            preferred_tier,
                            vbr_recovery_requirement::none_for_admission_progress);
                    }
                    if (!selected) {
                        selected = select_tier(
                            !preferred_tier,
                            vbr_recovery_requirement::none_for_admission_progress);
                    }
                }
                // The fallback returned by vbr_retention_victim is the
                // protection-aware LRU floor. Always adopt it: retaining the
                // raw scan winner here could overwrite a hard-leased or
                // deferred slot exactly when learned evidence is incomplete
                // (including iSWA/alias cases).
                ret = selected;
                if (ret) {
                    t_last = ret->t_last_used;
                }
                retention_value_selected =
                    selected_by_value && ret != nullptr;
            }

            if (ret != nullptr) {
                if (!is_preflight) {
                    SLT_INF(*ret,
                        "selected slot by %s, t_last = %" PRId64
                        " (best rejected sim = %.3f)\n",
                        retention_value_selected ? "decayed retention value" : "LRU",
                        t_last, sim_best_any);
                }

                if (plan_rec) {
                    plan_rec->selection = common_cache_plan_selection::lru;
                }

                update_cache = true;
            }
        }

        if (ret && ret->is_processing()) {
            // A by-id task targeting a busy slot is deferred by the caller.
            // Do not replace that slot's in-flight evidence record with this
            // retry's local record; shipped selection/mutation remains intact.
            // Preflight retains the request-local record so PROVIDER_BUSY
            // remains visible as typed evidence without arming the slot.
            selection_deferred_busy = true;
            if (!is_preflight) {
                plan_rec.reset();
            }
        }

        return {
            ret,
            update_cache,
            selection_deferred_busy,
            std::move(plan_rec),
        };
    }

    // fork server-context.cpp:11294  (27 lines)
    server_cache_plan_preflight_view cache_plan_preflight(
            const server_task & task) {
        assert_scheduler_thread(
            cache_plan_preflight_scheduler_thread,
            "cache-plan preflight must run on scheduler thread");
        server_cache_plan_preflight_view view;
        auto stage1 = cache_plan_select_before_mutation(
            task, true, true);
        if (!stage1.target) {
            view.status = server_cache_plan_preflight_status::no_target;
            return view;
        }
        if (!stage1.record) {
            view.status = server_cache_plan_preflight_status::internal_fault;
            return view;
        }

        cache_plan_stage1_inventory inventory;
        cache_plan_inventory_request(
            task, *stage1.target, stage1.update_cache, *stage1.record,
            inventory, true);
        if (!server_cache_plan_preflight_build_view(
                *stage1.record, stage1.target->id, view)) {
            view.status = server_cache_plan_preflight_status::internal_fault;
        }
        return view;
    }

    // fork server-context.cpp:11637-11648  (12 lines)
    static server_vbr_prompt_cache_support_status
    automatic_vbr_cache_support(
            const server_tokens & tokens,
            const std::vector<common_adapter_lora_info> & lora,
            bool speculative_slot) noexcept {
        // Media and aLoRA still need their own frontier authority. Draft-model
        // and DFlash state are exact companion payloads in the same adoption
        // transaction as target KV, so speculative slots are supported.
        return server_vbr_prompt_cache_support_for(
            false, speculative_slot, tokens.has_media(),
            lora_all_alora(lora));
    }

    // fork server-context.cpp:3890  (140 lines)
    void cache_plan_emit_budget(
            const llama_cache_acct_snapshot & snapshot) noexcept {
        if (!cache_plan_obs) {
            return;
        }
        try {
            llama_cache_budget_config config;
            if (!cache_authority->sample_budget(config)) {
                cache_plan_obs->shadow_unavailable++;
                return;
            }

            llama_cache_budget_plan baseline;
            baseline.accounting_serial = snapshot.serial;
            if (!cache_authority->budget.reset(snapshot, config)) {
                cache_plan_obs->shadow_unavailable++;
                return;
            }
            const auto result = cache_authority->budget.fits(baseline);

            const auto compute_json = [](
                    const llama_cache_budget_device_input & input,
                    uint64_t value) {
                const llama_cache_acct_value typed =
                    input.compute_state ==
                        llama_cache_budget_capacity_state::known
                        ? llama_cache_acct_value::measured(value)
                        : llama_cache_acct_value {
                            0, llama_cache_acct_known::unavailable,
                        };
                return server_json_from_ordered(
                    common_cache_plan_value_json(typed));
            };
            for (const auto & group : result.groups) {
                if (group.resource.kind !=
                    llama_cache_budget_resource_kind::physical_device) {
                    continue;
                }
                const auto input_it = std::find_if(
                    config.devices.begin(), config.devices.end(),
                    [&](const auto & input) {
                        return input.backend_device ==
                            group.resource.backend_device;
                    });
                if (input_it == config.devices.end()) {
                    cache_plan_obs->shadow_unavailable++;
                    continue;
                }

                std::string topology_digest;
                const auto topo_it = std::find_if(
                    snapshot.topologies.begin(), snapshot.topologies.end(),
                    [&](const auto & row) {
                        return row.id == input_it->domain.topology;
                    });
                if (topo_it != snapshot.topologies.end()) {
                    topology_digest = common_cache_plan_sha256_hex_digest(
                        topo_it->topology.digest.bytes());
                }

                const char * reason = "none";
                if (group.state ==
                    llama_cache_budget_fit_state::unavailable) {
                    reason = group.ceiling_state ==
                            llama_cache_budget_capacity_state::unavailable
                        ? "capacity_unavailable"
                        : "accounting_unavailable";
                } else if (group.state ==
                           llama_cache_budget_fit_state::exceeds) {
                    reason = "budget_exceeded";
                }

                const json line = {
                    { "accounting_serial", result.accounting_serial },
                    { "domain", {
                        { "residency", common_cache_acct_residency_name(
                            input_it->domain.residency) },
                        { "device_ordinal",
                            input_it->domain.device_ordinal.v },
                        { "topology_id", input_it->domain.topology.v },
                        { "topology_digest", topology_digest.empty()
                            ? json("unavailable")
                            : json(topology_digest) },
                    } },
                    { "device", {
                        { "name", ggml_backend_dev_name(
                            reinterpret_cast<ggml_backend_dev_t>(
                                const_cast<void *>(
                                    input_it->backend_device))) },
                        { "physical_total", input_it->physical_total },
                        { "physical_free", input_it->physical_free },
                        { "physical_state",
                            llama_cache_budget_capacity_state_name(
                                input_it->phys_state) },
                    } },
                    { "current_cache_resident",
                        server_json_from_ordered(common_cache_plan_value_json(
                            group.current_resident)) },
                    { "current_compute_allocated",
                        compute_json(
                            *input_it, input_it->current_compute_allocated) },
                    { "configured_compute_reserve",
                        compute_json(
                            *input_it, input_it->configured_compute_reserve) },
                    { "reserve_provenance",
                        llama_cache_budget_reserve_provenance_name(
                            input_it->reserve_provenance) },
                    { "derived_cache_ceiling",
                        group.ceiling_state ==
                            llama_cache_budget_capacity_state::known
                            ? server_json_from_ordered(
                                common_cache_plan_value_json(group.ceiling))
                            : json(llama_cache_budget_capacity_state_name(
                                group.ceiling_state)) },
                    { "fit", {
                        { "state", llama_cache_budget_fit_state_name(
                            group.state) },
                        { "reason", reason },
                        { "before",
                            server_json_from_ordered(
                                common_cache_plan_value_json(group.before)) },
                        { "after",
                            server_json_from_ordered(
                                common_cache_plan_value_json(group.after)) },
                        { "headroom_after",
                            group.headroom_state ==
                                llama_cache_budget_capacity_state::known
                                ? server_json_from_ordered(
                                    common_cache_plan_value_json(
                                        group.headroom_after))
                                : json(llama_cache_budget_capacity_state_name(
                                    group.headroom_state)) },
                    } },
                };
                SRV_INF("CACHE_BUDGET %s\n", line.dump().c_str());
            }
        } catch (...) {
            cache_plan_obs->shadow_unavailable++;
        }
    }

    // fork server-context.cpp:4275  (42 lines)
    void cache_plan_run_yield(
            const llama_cache_acct_snapshot & snapshot,
            std::vector<server_retention_candidate> catalog) noexcept {
        if (!cache_plan_obs) {
            return;
        }
        try {
            std::vector<server_cache_yield_candidate> candidates;
            if (!cache_plan_assemble_yield_candidates(catalog, candidates)) {
                cache_authority->last_yield = {};
                cache_authority->last_yield.accounting_serial =
                    snapshot.serial;
                return;
            }

            const auto preview = [&](const auto & ops,
                                     uint64_t expected_serial,
                                     auto & out) {
                return cache_authority->ledger.preview_release_set(
                    ops, expected_serial, out);
            };
            const auto fit = [&](const llama_cache_budget_plan & plan) {
                // The empty baseline has no prefix preview, so probe the live
                // ledger here to preserve the atomic-capture serial guard.
                llama_cache_acct_release_set_preview serial_probe;
                if (!cache_authority->ledger.preview_release_set(
                        {}, snapshot.serial, serial_probe)) {
                    llama_cache_budget_result unavailable;
                    unavailable.accounting_serial = snapshot.serial;
                    return unavailable;
                }
                return cache_authority->budget.fits(plan);
            };
            cache_authority->last_yield = server_cache_yield_plan(
                candidates, snapshot.serial, preview, fit);
        } catch (...) {
            cache_authority->last_yield = {};
            cache_authority->last_yield.accounting_serial =
                snapshot.serial;
            cache_plan_obs->shadow_unavailable++;
        }
    }

    // fork server-context.cpp:4349  (95 lines)
    void cache_plan_project_yield(
            common_cache_plan_record & rec,
            const server_cache_yield_result & result) noexcept {
        try {
            common_cache_plan_yield_record projected;
            const auto actual_state = rec.yield.actual_state;
            auto actual_domains = std::move(rec.yield.actual_domains);
            if (result.accounting_serial != rec.acct.serial) {
                throw std::runtime_error(
                    "cache-yield record/accounting serial mismatch");
            }
            switch (result.status) {
                case server_cache_yield_status::fits:
                    projected.status = common_cache_plan_yield_status::fits;
                    break;
                case server_cache_yield_status::insufficient_yield:
                    projected.status =
                        common_cache_plan_yield_status::insufficient_yield;
                    break;
                case server_cache_yield_status::unsupported_required:
                    projected.status =
                        common_cache_plan_yield_status::unsupported_required;
                    break;
                case server_cache_yield_status::unavailable:
                    projected.status = common_cache_plan_yield_status::unavailable;
                    break;
                case server_cache_yield_status::_count:
                    throw std::runtime_error("invalid cache-yield status");
            }

            projected.yield_policy_version = result.yield_policy_version;
            projected.accounting_serial = result.accounting_serial;
            projected.selected_attention =
                result.selected[size_t(common_retention_pool::attention)];
            projected.selected_recurrent =
                result.selected[size_t(common_retention_pool::recurrent)];
            projected.unsupported = result.unsupported;

            const bool selected_any =
                !projected.selected_attention.empty() ||
                !projected.selected_recurrent.empty();
            if (result.status == server_cache_yield_status::fits) {
                if (result.projected_fit.accounting_serial !=
                        result.accounting_serial ||
                    result.projected_fit.state !=
                        llama_cache_budget_fit_state::fits) {
                    throw std::runtime_error(
                        "cache-yield projection serial/state mismatch");
                }
                projected.plan_state = selected_any
                    ? common_cache_plan_yield_plan_state::planned
                    : common_cache_plan_yield_plan_state::not_required;
                // Baseline-fit rows remain in the server pre-image as serial-bound
                // evidence, but no-eviction wire records carry no projected action.
                if (selected_any) {
                    if (result.plan.empty() ||
                        result.projected_fit.domains.empty()) {
                        throw std::runtime_error(
                            "cache-yield planned projection is empty");
                    }
                    projected.projected_domains.reserve(
                        result.projected_fit.domains.size());
                    for (const auto & row : result.projected_fit.domains) {
                        common_cache_plan_yield_domain lowered;
                        if (!server_cache_yield_lower_domain(row, lowered)) {
                            throw std::runtime_error(
                                "non-domain cache-yield projection row");
                        }
                        projected.projected_domains.push_back(
                            std::move(lowered));
                    }
                }
            } else {
                projected.plan_state =
                    common_cache_plan_yield_plan_state::unavailable;
            }

            // Shadow yield never executes an eviction. Destruction authority may fill this already-versioned
            // slot only after it has authoritative post-mutation measurements.
            if (rec.destruction.state ==
                    common_cache_plan_destruction_state::executed) {
                projected.actual_state = actual_state;
                projected.actual_domains = std::move(actual_domains);
            } else {
                projected.actual_state =
                    common_cache_plan_yield_actual_state::not_observed;
            }
            rec.yield = std::move(projected);
        } catch (...) {
            rec.yield = {};
            if (cache_plan_obs) {
                cache_plan_obs->shadow_unavailable++;
            }
        }
    }

    // fork server-context.cpp:4320  (24 lines)
    void cache_plan_emit_yield(
            const server_cache_yield_result & result) noexcept {
        if (!cache_plan_obs) {
            return;
        }
        try {
            const json line = {
                { "status", server_cache_yield_status_name(result.status) },
                { "yield_policy_version", result.yield_policy_version },
                { "accounting_serial", result.accounting_serial },
                { "n_selected_attention",
                    result.selected[size_t(
                        common_retention_pool::attention)].size() },
                { "n_selected_recurrent",
                    result.selected[size_t(
                        common_retention_pool::recurrent)].size() },
                { "n_plan_entries", result.plan.size() },
                { "n_unsupported", result.unsupported.size() },
            };
            SRV_INF("CACHE_YIELD %s\n", line.dump().c_str());
        } catch (...) {
            cache_plan_obs->shadow_unavailable++;
        }
    }

    // fork server-context.cpp:4451  (133 lines)
    void cache_plan_finalize(server_slot & slot, bool ttft_known = true) {
        if (!slot.cache_plan) {
            return;
        }
        try {
            auto & rec = *slot.cache_plan;

            // once-only: outcome != unknown IS the finalized state; a second finalization
            // attempt on the same record is an observer fault, never a second emission
            if (rec.outcome != common_cache_plan_outcome::unknown) {
                if (cache_plan_obs) {
                    cache_plan_obs->shadow_unavailable++;
                }
                slot.cache_plan.reset();
                return;
            }

            rec.n_prompt_tokens   = llama_cache_acct_value::measured((uint64_t) slot.task->n_tokens());
            rec.n_reused_tokens   = llama_cache_acct_value::measured(slot.stats.n_prompt_cached);
            rec.n_replayed_tokens = llama_cache_acct_value::measured(slot.stats.n_prompt_processed);
            if (ttft_known) {
                rec.ttft_us = llama_cache_acct_value::measured(
                    (uint64_t) (slot.stats.t_prompt_ms() * 1000.0));
            }

            // live-slot evidence comes from the FINAL post-trim state, not the selection
            // heuristics: realized prefix reuse IS live-slot delivery (with the default zero
            // similarity threshold an LRU pick can still reuse its own prefix), and a
            // stage-1 heuristic reject is superseded by that fact. PROVENANCE guard (): a
            // host restore replaces the slot state wholesale, so a positive final reuse
            // count after host delivery is the HOST's prefix — never attributed to the
            // original live slot.
            {
                const auto * host = rec.selected_row(common_cache_plan_provider::host_cache_entry);
                const auto * active = rec.selected_row(common_cache_plan_provider::active_context_checkpoint);
                const auto * attention = rec.selected_row(common_cache_plan_provider::active_attention_prefix);
                auto *       live = rec.selected_row(common_cache_plan_provider::live_slot);
                if (slot.stats.n_prompt_cached > 0 && !(host && host->delivered) &&
                    !(active && active->delivered) && !(attention && attention->delivered) && live) {
                    live->disposition = common_cache_plan_disposition::accepted;
                    live->delivered   = true;
                    live->reason      = COMMON_CACHE_PLAN_REASON_NONE;
                    live->lcp_tokens  = llama_cache_acct_value::measured(slot.stats.n_prompt_cached);
                }
            }

            // cold replay is the always-valid floor: it always has a row, and it delivered
            // exactly when nothing else did
            if (auto * cold = rec.find_or_add(
                    common_cache_plan_provider::cold_replay,
                    COMMON_CACHE_PLAN_SOURCE_AGGREGATE, uint8_t(0), rec.id_slot,
                    rec.selection)) {
                cold->disposition = common_cache_plan_disposition::accepted;
                rec.select(common_cache_plan_provider::cold_replay, cold);
                rec.note_inventory_complete(common_cache_plan_provider::cold_replay);
            }

            // chosen = the TERMINAL delivered provider (delivery is data recorded at each
            // site; the causal chain is emitted separately, so composition is not lost)
            const common_cache_plan_provider terminal_order[] = {
                common_cache_plan_provider::active_attention_prefix,
                common_cache_plan_provider::active_context_checkpoint,
                common_cache_plan_provider::live_context_checkpoint,
                common_cache_plan_provider::host_cache_entry,
                common_cache_plan_provider::live_slot,
            };
            const common_cache_plan_candidate * delivered_row = nullptr;
            for (const auto prov : terminal_order) {
                const auto * c = rec.selected_row(prov);
                if (c && c->delivered) {
                    delivered_row = c;
                    rec.chosen    = prov;
                    break;
                }
            }

            if (delivered_row) {
                rec.outcome = common_cache_plan_outcome::restored;
            } else if (rec.restore_attempt_failed) {
                rec.outcome = common_cache_plan_outcome::restore_failed_fell_back_cold;
                rec.chosen  = common_cache_plan_provider::cold_replay;
            } else {
                rec.outcome = common_cache_plan_outcome::cold;
                rec.chosen  = common_cache_plan_provider::cold_replay;
            }
            if (rec.chosen == common_cache_plan_provider::cold_replay) {
                if (auto * cold = rec.selected_row(common_cache_plan_provider::cold_replay)) {
                    cold->delivered = true;
                }
            }

            // A composed plan is a first-class candidate: a
            // multi-provider delivery (host snapshot + checkpoint continuation) gets ONE
            // chain row — DELIVERED, since it is what actually ran — and the chain IS the
            // complete shipped plan. The bare checkpoint row becomes component-only: its
            // state was reachable only through the delivered host entry, so it can never
            // enter the root optimum as a standalone plan.
            common_cache_plan_compose_chains(rec);

            if (cache_plan_obs) {
                if (!cache_plan_observe_live_memory(false)) {
                    cache_plan_obs->shadow_unavailable++;
                }
                rec.acct = cache_authority->ledger.snapshot();
                auto retention_candidates =
                    cache_authority->retention.candidate_snapshot();
                cache_plan_emit_budget(rec.acct);
                cache_plan_run_yield(
                    rec.acct, std::move(retention_candidates));
                cache_plan_project_yield(rec, cache_authority->last_yield);
                cache_plan_emit_yield(cache_authority->last_yield);
            }

            if (cache_plan_obs) {
                json out = server_json_from_ordered(
                    common_cache_plan_record_json(rec));
                cache_plan_obs->records_finalized++;
                out["observer"] = json {
                    { "records_finalized",  cache_plan_obs->records_finalized },
                    { "shadow_unavailable", cache_plan_obs->shadow_unavailable },
                };

                SRV_INF("CACHE_PLAN %s\n", out.dump().c_str());
                slot.cache_plan_json = std::move(out); // built once; /slots reuses it verbatim
            }
            slot.cache_plan.reset();
        } catch (...) {
            if (cache_plan_obs) {
                cache_plan_obs->shadow_unavailable++;
            }
            slot.cache_plan.reset();
        }
    }

    // fork server-context.cpp:9257  (12 lines)
    void cache_plan_derive_incoming_adapter(
            const server_task & task,
            const server_slot & target,
            cache_plan_stage1_inventory & out) {
        out.incoming_loras = task.params.lora.empty()
            ? params_base.lora_adapters
            : construct_lora_list(task.params.lora);
        out.incoming_adapter = lora_config_identity(out.incoming_loras);
        out.incoming_adapter_ready = true;
        out.incoming_adapter_matches =
            are_lora_equal(out.incoming_loras, target.lora);
    }

    // fork server-context.cpp:8843  (35 lines)
    void cache_plan_inventory_before_mutation(
            const server_task & task,
            server_slot & target,
            const std::string & incoming_adapter,
            common_cache_plan_record & rec,
            cache_plan_host_source_registry & source_registry) {
        rec.id_slot = target.id;
        rec.n_prompt_tokens = llama_cache_acct_value::measured(task.tokens.size());
        rec.identity.adapter_config_digest = llama_cache_acct_value::measured(
            std::hash<std::string>{}(incoming_adapter));

        source_registry.begin(prompt_cache.get());

        std::vector<uint64_t> slot_lcps;
        slot_lcps.reserve(slots.size());
        for (const auto & slot : slots) {
            slot_lcps.push_back(slot.prompt.tokens.get_common_prefix(task.tokens));
        }
        const bool recurrent = llama_model_is_recurrent(model_tgt) ||
                               llama_model_is_hybrid(model_tgt);

        // A fixed record is an honest bounded surface, not a promise that a
        // cross-product always fits. Any failed append latches provider
        // overflow (or derived_plans_incomplete); stop paying work immediately.
        if (!cache_plan_inventory_live_rows(task, slot_lcps, rec) ||
            !cache_plan_inventory_host_rows(
                task, slot_lcps, incoming_adapter, recurrent, rec,
                source_registry) ||
            !cache_plan_inventory_checkpoint_rows(
                task, slot_lcps, recurrent, rec) ||
            !cache_plan_inventory_cold_rows(task, slot_lcps, rec)) {
            GGML_ASSERT(rec.inventory_saturated());
            return;
        }
    }

    // fork server-context.cpp:9270  (32 lines)
    void cache_plan_inventory_request(
            const server_task & task,
            server_slot & target,
            bool update_cache,
            common_cache_plan_record & rec,
            cache_plan_stage1_inventory & out,
            bool preflight) {
        cache_plan_host_source_registry source_registry(
            !preflight);
        cache_plan_derive_incoming_adapter(task, target, out);
        cache_plan_inventory_before_mutation(
            task, target, out.incoming_adapter, rec, source_registry);

        const auto semantics = server_cache_plan_preflight_semantics_for(
            preflight,
            task.type == SERVER_TASK_TYPE_COMPLETION,
            update_cache,
            prompt_cache != nullptr,
            out.incoming_adapter_matches);
        out.host_lookup_enabled = semantics.host_lookup_enabled;
        // Selection observes the pre-launch inventory. Do not advertise that
        // state when the request will clear it; preserve the aLoRA exception
        // using the same predicate as launch_slot_with_task.
        if (preflight && (!task.params.cache_prompt ||
                (!out.incoming_adapter_matches &&
                 lora_should_clear_cache(target.lora, out.incoming_loras)))) {
            rec.destruction_legacy_plan_candidate = -1;
            return;
        }
        rec.destruction_legacy_plan_candidate = server_cache_plan_legacy_candidate(
            rec, target.id, out.host_lookup_enabled);
    }

};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->n_ctx_slot(),
        /* vbr_enabled            */ impl->params_base.vbr_enabled(),
        /* vbr_dynamic            */ impl->params_base.vbr_dynamic(),
        /* vbr_type_k             */ impl->params_base.vbr_cache_type_k,
        /* vbr_type_v             */ impl->params_base.vbr_cache_type_v,
        /* vbr_codec              */ impl->params_base.vbr_codec == LLAMA_VBR_CODEC_CLASSIC ? "classic" : "turbo",
        /* vbr_entry_type_k       */ ggml_type_name(impl->params_base.cache_type_k),
        /* vbr_entry_type_v       */ ggml_type_name(impl->params_base.cache_type_v),
        /* vbr_min_bits           */ impl->params_base.vbr_min_bits_value,
        /* vbr_capacity_bits      */ impl->params_base.vbr_capacity_bits,
        /* vbr_selected_bpv       */ impl->params_base.vbr_selected_bpv,
        /* vbr_selected_kld       */ impl->params_base.vbr_selected_kld,
        /* vbr_vram_budget_bytes  */ impl->params_base.vbr_vram_budget_bytes,
        /* vbr_selected_family    */ impl->params_base.vbr_selected_family,
        /* vbr_selected_policy    */ impl->params_base.vbr_selected_policy,
        /* vbr_selected_schedule  */ impl->params_base.vbr_selected_schedule,
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"vbr",         server_vbr_meta_json(&meta)},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "vbr",                         server_vbr_meta_json(&meta) },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

static json server_vbr_meta_json(const server_context_meta * meta) {
    return json {
        {"enabled",           meta->vbr_enabled},
        {"dynamic",           meta->vbr_dynamic},
        {"type_k",            meta->vbr_type_k},
        {"type_v",            meta->vbr_type_v},
        {"codec",             meta->vbr_codec},
        {"entry_type_k",      meta->vbr_entry_type_k},
        {"entry_type_v",      meta->vbr_entry_type_v},
        {"floor_bpv",         meta->vbr_min_bits},
        {"capacity_floor_bpv", meta->vbr_capacity_bits},
        // realized bits/value is a fixed number only for static schedules; under the
        // dynamic runtime controller it varies with occupancy — null, not a fiction
        {"realized_bpv",      meta->vbr_dynamic ? json() : json(meta->vbr_capacity_bits)},
        {"selected_family",   meta->vbr_selected_family},
        {"selected_policy",   meta->vbr_selected_policy},
        {"selected_bpv",      meta->vbr_dynamic ? json() : json(meta->vbr_selected_bpv)},
        {"selected_kld",      meta->vbr_dynamic ? json() : json(meta->vbr_selected_kld)},
        {"selected_schedule", meta->vbr_dynamic ? json() : json(meta->vbr_selected_schedule)},
        {"vram_budget_bytes", meta->vbr_vram_budget_bytes},
    };
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

        // fork 9l-d4b7f: post_slots in the fork's own order (fork server-context.cpp
    // 21809-21851).  FORK-DIFF: the --slot-save-path gate moved from the top of the
    // lambda to below capture/erase, so `action=erase` is now reachable without the
    // flag (the fork's own comment further down explains why) and `capture` is never
    // gated; `save`/`restore` stay behind the gate.
    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "capture") {
            return handle_slots_capture(req, id_slot);
        }
        // fork 9l-d4b7g: the import branch (fork server-context.cpp 21826-21828).
        if (action == "import") {
            return handle_slots_import(req, id_slot);
        }
        // Erase does no state-file IO (prompt clear + seq_rm only), so it is
        // NOT gated on --slot-save-path. Dynamic VBR clears slot_save_path at
        // startup (legacy state files cannot carry tier-typed KV), and import
        // REQUIRES an explicit erase to produce its empty target — keeping
        // erase behind the path gate made import's precondition unreachable
        // on exactly the servers that support import.
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    // fork 9l-d4b7e: the cache-control HTTP handler (fork server-context.cpp
    // 21964-22143).  One handler serves every path in SERVER_CACHE_CONTROL_ROUTES;
    // server.cpp registers it once per route.  The fork's post_cache_plan
    // (21853-21962) is not ported and belongs here, above this one.
    this->post_cache_control = [this](const server_http_req & req) {
        auto res = create_response();
        res->headers["Cache-Control"] = "no-store";

        server_cache_control_operation operation =
            server_cache_control_operation::_count;
        GGML_ASSERT(params.cache_control_api &&
                    server_cache_control_operation_for_path(req.path, operation));

        auto request = std::make_shared<server_cache_control_request>();
        server_task task = server_task::cache_control_task(operation);
        task.id = res->rd.get_new_id();

        const auto typed_status = [&](server_cache_control_status status,
                                      int http_status = 200) {
            server_cache_control_result result;
            result.status = status;
            const auto body = server_json_from_ordered(
                server_cache_control_json(operation, result));
            if (http_status == 400) {
                res->status = 400;
                res->data = safe_json_to_str(body);
            } else {
                res->ok(body);
            }
        };
        const auto parse_selector = [&](const json & value,
                                        server_cache_control_selector & selector,
                                        server_task::cache_control_semantic_selector & semantic) {
            if (!value.is_object() || !value.contains("kind") ||
                !value.at("kind").is_string()) {
                return false;
            }
            const std::string kind = value.at("kind").get<std::string>();
            if (!server_cache_control_parse_subject_kind(kind, selector.kind)) {
                return false;
            }
            // Authorize the complete shape before prompt tokenization. No
            // rejected field may consume tokenizer/model work on the worker.
            for (const auto & field : value.items()) {
                if (!server_cache_control_selector_field_allowed(
                        selector.kind, field.key())) {
                    return false;
                }
            }
            if (selector.kind == server_cache_control_subject_kind::live_prefix) {
                if (!value.contains("slot_id") ||
                    !value.at("slot_id").is_number_integer()) {
                    return false;
                }
                const int64_t slot_id = value.at("slot_id").get<int64_t>();
                if (slot_id < std::numeric_limits<int32_t>::min() ||
                    slot_id > std::numeric_limits<int32_t>::max()) {
                    return false;
                }
                semantic.slot_id = int32_t(slot_id);
            } else if (selector.kind == server_cache_control_subject_kind::host_snapshot) {
                if (!value.contains("prompt")) {
                    return false;
                }
                auto inputs = tokenize_input_prompts(
                    ctx_server.vocab, ctx_server.mctx,
                    value.at("prompt"), true, true, ctx_server.init_opt);
                if (inputs.size() != 1) {
                    return false;
                }
                if (value.contains("lora")) {
                    if (!value.at("lora").is_array()) {
                        return false;
                    }
                    semantic.lora = parse_lora_request(value.at("lora"));
                }
                if (value.contains("message_delimiters") &&
                    !value.at("message_delimiters").is_array()) {
                    return false;
                }
                semantic.tokens = std::make_shared<const server_tokens>(
                    std::move(inputs.front()));
            } else if (selector.kind == server_cache_control_subject_kind::vbr_reference) {
                if (!value.contains("reference") ||
                    !value.at("reference").is_string()) {
                    return false;
                }
                selector.reference = value.at("reference").get<std::string>();
                selector.tenant_key = server_cache_capture_tenant_key(req);
            }
            return true;
        };

        try {
            if (!req.files.empty()) {
                throw std::runtime_error("cache-control routes do not accept multipart files");
            }
            const json body = json::parse(req.body);
            const auto prepared = server_cache_control_prepare_request(
                operation, server_ordered_from_json(body), *request);
            if (prepared != server_cache_control_status::ok) {
                typed_status(prepared, 400);
                return res;
            }
            bool valid = false;
            switch (operation) {
                case server_cache_control_operation::holder_create:
                case server_cache_control_operation::holder_reattach:
                case server_cache_control_operation::holder_close:
                case server_cache_control_operation::family_register:
                case server_cache_control_operation::family_bind:
                    valid = true;
                    break;
                case server_cache_control_operation::lease_acquire: {
                    valid = true;
                    if (valid) {
                        valid = parse_selector(
                            body.at("subject"), request->subject,
                            task.cache_control_subject);
                    }
                    if (valid && body.contains("fallback")) {
                        valid = parse_selector(
                            body.at("fallback"), request->fallback,
                            task.cache_control_fallback);
                    } else if (valid && request->requested_class ==
                            server_cache_lease_class::hard) {
                        valid = false;
                    } else if (valid) {
                        request->fallback.kind =
                            server_cache_control_subject_kind::_count;
                    }
                    if (valid) {
                        valid = !request->allow_soft_fallback ||
                            request->requested_class ==
                                server_cache_lease_class::hard;
                    }
                } break;
                case server_cache_control_operation::lease_inspect:
                case server_cache_control_operation::lease_release:
                    valid = true;
                    break;
                case server_cache_control_operation::lease_renew:
                    valid = true;
                    request->fallback.kind =
                        server_cache_control_subject_kind::_count;
                    if (valid && body.contains("fallback")) {
                        valid = parse_selector(
                            body.at("fallback"), request->fallback,
                            task.cache_control_fallback);
                    }
                    break;
                case server_cache_control_operation::events:
                    valid = true;
                    break;
                case server_cache_control_operation::_count:
                    valid = false;
                    break;
            }
            if (!valid) {
                typed_status(server_cache_control_status::invalid_request, 400);
                return res;
            }
            task.cache_control = std::move(request);
            res->rd.post_task(std::move(task));
        } catch (const std::exception &) {
            typed_status(server_cache_control_status::invalid_request, 400);
            return res;
        }

        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * controlled = dynamic_cast<
            server_task_result_cache_control *>(result.get());
        GGML_ASSERT(controlled != nullptr);
        res->ok(controlled->to_json());
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i], ctx_server.init_opt);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

// fork 9l-d4b7f: handle_slots_capture (fork server-context.cpp 22692-22720).
// The handle_slots_import definition (fork 22722-22761) landed right below as 9l-d4b7g.
std::unique_ptr<server_res_generator> server_routes::handle_slots_capture(
        const server_http_req & req,
        int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_CACHE_CAPTURE);
        task.id = rd.get_new_id();
        task.cache_capture.id_slot = id_slot;
        task.cache_capture.tenant_key =
            server_cache_capture_tenant_key(req);
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }
    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }
    auto * captured =
        dynamic_cast<server_task_result_cache_capture *>(result.get());
    GGML_ASSERT(captured != nullptr);
    res->ok(captured->to_json());
    return res;
}

// fork 9l-d4b7g: handle_slots_import (fork server-context.cpp 22722-22761).
std::unique_ptr<server_res_generator> server_routes::handle_slots_import(
        const server_http_req & req,
        int id_slot) {
    auto res = create_response();
    std::string reference;
    try {
        const json request_data = json::parse(req.body);
        reference = request_data.at("reference").get<std::string>();
    } catch (const std::exception &) {
        res->error(format_error_response(
            "Import requires a string reference",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_CACHE_IMPORT);
        task.id = rd.get_new_id();
        task.cache_import.id_slot = id_slot;
        task.cache_import.tenant_key =
            server_cache_capture_tenant_key(req);
        task.cache_import.reference = std::move(reference);
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }
    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }
    auto * imported =
        dynamic_cast<server_task_result_cache_import *>(result.get());
    GGML_ASSERT(imported != nullptr);
    res->ok(imported->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
