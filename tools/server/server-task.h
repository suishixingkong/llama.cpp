#pragma once

#include "common.h"
#include "llama.h"
#include "server-prompt-cache-payload.h"
#include "server-cache-control.h"
#include "server-cache-plan-preflight.h"
// The cache result types name the artifact-store statuses directly. The fork's
// server-task.h relies on its includers having pulled this in first; spelling the
// dependency out keeps the header self-contained.
#include "server-vbr-artifact-store.h"
#include "../../src/llama-vbr-artifact.h"

#include <string>
#include <unordered_set>
#include <list>
#include <map>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"
#include "server-cache-lease.h"
// fork 9l-d3c2: the shadow workspace constants live in the retention sidecar header
#include "server-retention-sidecar.h"


// group A phase 2: verbatim from the fork server-task.h

// fork server-task.h:29  (6 lines)
constexpr common_cache_plan_payload_kind server_cache_plan_payload_kind(
        server_prompt_cache_payload_kind kind) noexcept {
    return kind == server_prompt_cache_payload_kind::vbr_artifact
        ? common_cache_plan_payload_kind::vbr_artifact
        : common_cache_plan_payload_kind::fixed_state;
}


// Derived lease identity for one exact logical prefix. Fails (and leaves `out`
// unusable) when any component is missing, so a caller can never key a lease on
// a partially known identity.
bool server_cache_lease_build_identity(
    const std::string & execution_identity,
    const std::string & adapter_identity,
    const server_tokens & tokens,
    int64_t coverage_tokens,
    server_cache_lease_identity & out);


enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_GET,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    std::string        control_action;
    std::string        control_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    std::string generated_text; // append new chunks of generated text here
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(const common_chat_parser_params & chat_parser_params);

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // fork 9l-d2c: optional declared-family policy input (fork server-task.h
    // 186-190). The wire layer supplies only this opaque token; scheduler
    // launch resolves the strong binding locally, so an HTTP worker cannot
    // inject a policy value into a task.
    server_cache_control_token cache_family_binding_token;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

// fork 9l-d2b: one-shot prompt-preparation gate (fork server-task.h 201)
    bool prompt_tokens_prepared = false;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot
        copy.cache_family_binding_token = cache_family_binding_token;

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    task_result_state create_state() const {
        return task_result_state(params.chat_parser_params);
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

// Response-local extension of upstream's slot statistics. Keeping the VBR
// value here avoids widening every live slot while retaining one timings
// serializer for cached/prompt/generation/draft and KV representation data.
struct server_result_stats : server_slot_stats {
    double kv_bpv = -1.0;

    server_result_stats & operator=(const server_slot_stats & other) {
        static_cast<server_slot_stats &>(*this) = other;
        kv_bpv = -1.0;
        return *this;
    }

    json to_json() const;
};

struct server_task_result_cmpl_final : server_task_result {
    std::string content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    server_result_stats stats;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    // cache receipt (§7.7): serialized JSON attached verbatim when enabled;
    // empty = no receipt. Built in send_final_response from slot state.
    json cache_receipt;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    completion_token_output prob_output;
    server_result_stats stats;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

// used by /metrics API
struct server_task_result_metrics : server_task_result {
    // these are immediate stats, not accumulated (server_metrics is cumulative)
    int n_processing_slots = 0;
    int n_tasks_deferred = 0;

    server_metrics metrics;

    virtual json to_json() override;

    struct metric_item {
        std::string name;
        std::string description;
        double value; // prometheus values are always float64
    };
    std::string to_metrics();
};

// used by /slots API
struct server_task_result_slots : server_task_result {
    int n_idle_slots = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

enum class server_cache_capture_consistency : uint8_t {
    unavailable = 0,
    capture_exact,
    _count,
};

struct server_task_result_cache_capture : server_task_result {
    server_vbr_artifact_capture_status status {};
    server_cache_capture_consistency consistency =
        server_cache_capture_consistency::unavailable;
    std::string reference;
    uint32_t controllers = 0;
    uint32_t units = 0;
    uint32_t companions = 0;
    uint64_t payload_bytes = 0;
    uint64_t stash_bytes = 0;
    uint64_t companion_bytes = 0;
    uint64_t chunks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
    bool dedup = false;

    virtual json to_json() override;
};

enum class server_cache_import_consistency : uint8_t {
    unavailable = 0,
    capture_exact,
    live_rebased,
    _count,
};

const char * server_cache_import_consistency_name(
    server_cache_import_consistency consistency) noexcept;

struct server_task_result_cache_import : server_task_result {
    server_vbr_artifact_import_status status {};
    vbr_manifest_validation_status validation_status {};
    vbr_adopt_stage_status stage_status {};
    vbr_downward_reserve_status downward_reserve_status {};
    vbr_adopt_status adopt_status {};
    vbr_adopt_recovery_outcome recovery {};
    bool adopt_attempted = false;
    vbr_adopt_phase phase {};
    vbr_downward_adopt_subphase downward_subphase {};
    uint32_t downward_edge = UINT32_MAX;
    vbr_import_schedule_status schedule_status {};
    vbr_import_destination_status destination_status {};
    uint32_t destination_policy_steps = 0;
    uint64_t destination_logical_bytes = 0;
    uint64_t destination_physical_growth_bytes = 0;
    int64_t destination_max_deficit = 0;
    vbr_import_decision decision {};
    server_cache_import_consistency consistency =
        server_cache_import_consistency::unavailable;
    uint32_t units = 0;
    uint32_t companions = 0;
    uint64_t payload_bytes = 0;
    uint64_t companion_bytes = 0;

    virtual json to_json() override;
};

// Internal scheduler result. The public adapter adds the deliberately redacted wire
// serializer; until then no route can serialize this task result.
struct server_task_result_cache_plan_preflight : server_task_result {
    server_cache_plan_preflight_view view;

    virtual json to_json() override;
};

struct server_task_result_cache_control : server_task_result {
    server_cache_control_operation operation =
        server_cache_control_operation::_count;
    server_cache_control_result result;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt {
    server_tokens tokens;

    std::list<common_prompt_checkpoint> checkpoints;

    // Server-local lineage for computation-frontier migration.
    // It moves with host-cache/child-slot prompt clones and resets whenever
    // the prompt ledger is structurally cleared.
    uint64_t sequence_epoch = 0;

    friend void swap(server_prompt & lhs, server_prompt & rhs) noexcept {
        using std::swap;
        swap(lhs.tokens, rhs.tokens);
        lhs.checkpoints.swap(rhs.checkpoints);
        swap(lhs.sequence_epoch, rhs.sequence_epoch);
    }

    void clear() {
        tokens.clear();
        checkpoints.clear();
        sequence_epoch = 0;
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            sequence_epoch,
        };
    }
};

// fork 9l-d3c2: prompt-shadow workspace types (fork server-task.h 1137-1199)
constexpr size_t SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES =
    SERVER_RETENTION_MAX_CANDIDATES;

enum class server_prompt_cache_shadow_status : uint8_t {
    unavailable = 0,
    complete,
};

struct server_prompt_cache_shadow_event {
    server_prompt_cache_shadow_status status =
        server_prompt_cache_shadow_status::unavailable;
    server_cache_destruction_reason reason =
        server_cache_destruction_reason::host_capacity;
    uint64_t competition_epoch = 0;
    uint64_t candidate_count = 0;
    llama_cache_acct_artifact_id incumbent_artifact;
    llama_cache_acct_artifact_id proposed_artifact;
    uint64_t incumbent_lineage = 0;
    uint64_t proposed_lineage = 0;
    common_retention_pool proposed_pool = common_retention_pool::attention;
    uint64_t proposed_lost_work = 0;
    uint64_t proposed_resource = 0;
    bool agrees = false;
};

struct server_prompt_cache_shadow_snapshot {
    uint64_t pressure_waves = 0;
    uint64_t choices = 0;
    uint64_t complete = 0;
    uint64_t unavailable = 0;
    uint64_t agreements = 0;
    uint64_t disagreements = 0;
    server_prompt_cache_shadow_event last;
};

struct server_prompt_cache_shadow_row {
    llama_cache_acct_artifact_id artifact_id;
    server_retention_instance_key instance_key;
    common_retention_artifact_kind kind =
        common_retention_artifact_kind::live_slot;
    common_retention_stamp stamp;
    common_retention_lineage_record lineage;
    uint64_t external_shared_coverage_tokens = 0;
    uint64_t resource = 0;
    bool backing_known = false;
    bool releasable = false;
};

constexpr size_t SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY =
    2*SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES;

struct server_prompt_cache_shadow_artifact_slot {
    uint64_t artifact_id = 0;
    uint32_t row_index = 0;
};

struct server_prompt_cache_shadow_lineage_slot {
    uint64_t lineage_id = 0;
    uint64_t maximum_coverage = 0;
    uint64_t second_coverage = 0;
    uint32_t maximum_count = 0;
    common_retention_pool pool = common_retention_pool::attention;
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_cache_payload payload;

    // Canonical identity of the adapter configuration this state was computed under; a load is
    // only served from an entry whose key matches the requesting slot's current adapter config
    std::string adapter_config_key;

    // Exact model/execution identity for a sealed VBR payload. Fixed entries
    // leave this empty because their historical compatibility check remains
    // the server-local adapter/state codec contract.
    std::string vbr_execution_identity;

    // Exact accounting references held by this logical host node. Authority
    // mode stores one snapshot reference plus one reference per unique shared
    // checkpoint/accelerator allocation; lifecycle-off shadow mode retains
    // its historical three aggregate category references.
    std::vector<llama_cache_acct_op_id> acct_ops;
    bool accounting_complete = false;

    // recovery proof's non-policy recovery guard. List nodes are stable; authoritative
    // redundant eviction increments this before prepare and the raw eraser
    // refuses to destroy the cited survivor until capability close.
    uint32_t recovery_pins = 0;

    // Automatic pre- family signal. A save sourced from a parent/main slot
    // receives the provisional lease boundary retention weight; child-task saves do not.
    // Declared identity replaces this heuristic rather than stacking with it.
    bool main_family = false;
    common_cache_family_binding cache_family;

    const std::vector<llama_cache_acct_op_id> & release_ops() const noexcept {
        return acct_ops;
    }

    // Request-local observer identity. It lives on the list node so save-time
    // dedup/splice preserves surviving identities and an allocator-reused
    // address can never inherit a consumed entry's source id.
    int32_t cache_plan_source_id = -1;

    size_t size() const {
        size_t res = payload.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

// Caller-owned batch inventory for automatic durability classification. The
// canonical artifact identity narrows the state-list search; `prompt` supplies
// the exact token witness. The cache marks rows in one state-list pass.
struct server_prompt_cache_vbr_frontier_query {
    int32_t slot_id = -1;
    vbr_artifact_identity_block identity;
    const server_prompt * prompt = nullptr;
    std::array<uint8_t, 32> token_identity_digest = {};
    bool token_identity_ready = false;
    bool durable = false;
    // Optional immutable host-node witness for a previously published stem.
    // The scheduler separately binds this artifact to the unchanged full
    // source attempt and selected coverage; this batch only proves that exact
    // destination association remains live.
    llama_cache_acct_artifact_id expected_stem_artifact;
    bool stem_durable = false;
};

struct server_prompt_cache_vbr_frontier_batch_diagnostics {
    uint64_t states_visited = 0;
    uint64_t vbr_states_visited = 0;
    uint64_t stem_artifact_lookups = 0;
    uint64_t stem_matches = 0;
};


// Match the fork's established useful cache-reuse granularity. Tiny shared
// prefixes remain valid LCP hits, but cannot earn durable frequency credit or
// authorize copying a much larger displaced live state into host RAM.
constexpr size_t SERVER_PROMPT_CACHE_MIN_RETENTION_REUSE_TOKENS = 256;

inline bool server_prompt_cache_retention_reuse_is_useful(
        size_t reused_tokens,
        const common_chat_msg_spans * message_spans = nullptr) noexcept {
    if (reused_tokens >= SERVER_PROMPT_CACHE_MIN_RETENTION_REUSE_TOKENS) {
        return true;
    }
    if (reused_tokens == 0 || !message_spans) {
        return false;
    }
    // A complete system/instruction prefix reaching the start of user content
    // is semantically valuable even when a compact template tokenizes below
    // the generic anti-noise floor. The raw token LCP may continue into shared
    // user boilerplate; the later restore seam narrows credit to the frontier
    // that was actually installed. This admits short shared-system fleets
    // without promoting ubiquitous BOS/header overlaps.
    bool complete_system_prefix = false;
    for (const auto & span : message_spans->spans) {
        if (span.role == COMMON_CHAT_ROLE_SYSTEM &&
            span.pos <= reused_tokens &&
            span.len <= reused_tokens - span.pos) {
            complete_system_prefix = true;
        }
        if (complete_system_prefix &&
            span.role == COMMON_CHAT_ROLE_USER &&
            span.pos <= reused_tokens) {
            return true;
        }
    }
    return false;
// fork 9l-a: capacity enum + vbr restore candidate

}

// fork 9l-d2b: slot prompt-admission decision (fork server-task.h 1242-1264)
enum class server_slot_prompt_admission : uint8_t {
    accepted,
    batch_too_large,
    context_too_large,
};

// This admission check runs before launch mutates the selected slot's
// retention lineage. Keep the split/non-split boundaries identical to the
// decode path: split prompts need one cell of generation headroom, while an
// unsplit prompt may exactly fill the context.
inline server_slot_prompt_admission server_slot_prompt_admission_check(
        bool can_split,
        size_t n_tokens,
        size_t n_ubatch,
        size_t n_ctx) noexcept {
    if (!can_split && n_tokens > n_ubatch) {
        return server_slot_prompt_admission::batch_too_large;
    }
    if (n_tokens > n_ctx || (can_split && n_tokens == n_ctx)) {
        return server_slot_prompt_admission::context_too_large;
    }
    return server_slot_prompt_admission::accepted;
}

// fork 9l-d3c: retention owner plan + cache-lifecycle default (fork server-task.h 1266-1305)
inline bool server_cache_lifecycle_default(
        bool explicitly_enabled,
        bool prompt_cache_enabled,
        bool cache_control_api_enabled) noexcept {
    return explicitly_enabled || prompt_cache_enabled ||
        cache_control_api_enabled;
}

// Retention metadata is payload-independent. Fixed host caching and dynamic VBR both need the
// same lineage/prefix owner even though VBR cannot serialize tier-changing KV state yet.
enum class server_retention_owner_kind : uint8_t {
    none = 0,
    standalone_metadata,
    authority,
};

struct server_retention_owner_plan {
    server_retention_owner_kind owner = server_retention_owner_kind::none;
    bool prompt_shadow_workspace = false;
    bool prefix_tracking = false;
};

inline server_retention_owner_plan server_retention_owner_plan_for(
        bool cache_debug,
        bool cache_lifecycle,
        bool prompt_cache_enabled,
        bool vbr_dynamic,
        bool retention_capacity) noexcept {
    server_retention_owner_plan out;
    if (cache_debug || cache_lifecycle) {
        out.owner = server_retention_owner_kind::authority;
    } else if (prompt_cache_enabled || vbr_dynamic) {
        out.owner = server_retention_owner_kind::standalone_metadata;
    }
    out.prompt_shadow_workspace = prompt_cache_enabled &&
        (out.owner == server_retention_owner_kind::standalone_metadata ||
         cache_debug || retention_capacity);
    out.prefix_tracking = vbr_dynamic || out.prompt_shadow_workspace;
    return out;
}

struct server_prompt_cache;

enum class server_prompt_cache_vbr_capacity_status : uint8_t {
    invalid,
    fit,
    incoming_exceeds_hard_limit,
    pressure_cited,
    pressure_batch_unsupported,
};

// Move-only, non-consuming citation of one immutable VBR host entry. Exact
// stored prefixes are preferred; on an exact miss the capability may name a
// shorter projected LCP and its selected logical frontier. The highest-quality
// same-frontier owner is preferred; compact current remains available as a
// bounded fallback when destination negotiation refuses it.
// The source list node is pinned until commit or destruction; the shared
// payload owner independently keeps catalog bytes alive. Raw node identity is
// deliberately private so callers cannot forge or double-release a pin.
class server_prompt_cache_vbr_restore_candidate {
public:
    server_prompt_cache_vbr_restore_candidate() = default;
    ~server_prompt_cache_vbr_restore_candidate();
    server_prompt_cache_vbr_restore_candidate(
        const server_prompt_cache_vbr_restore_candidate &) = delete;
    server_prompt_cache_vbr_restore_candidate & operator=(
        const server_prompt_cache_vbr_restore_candidate &) = delete;
    server_prompt_cache_vbr_restore_candidate(
        server_prompt_cache_vbr_restore_candidate && other) noexcept;
    server_prompt_cache_vbr_restore_candidate & operator=(
        server_prompt_cache_vbr_restore_candidate && other) noexcept;

    bool ready() const noexcept;
    const server_prompt_cache_vbr_owner & payload() const noexcept;
    const server_prompt_cache_vbr_owner & fallback_payload() const noexcept;
    bool use_fallback_payload() noexcept;
    const common_cache_family_binding & cache_family() const noexcept;
    uint64_t prefix_tokens() const noexcept;
    uint64_t source_tokens() const noexcept;
    llama_pos selected_next_position() const noexcept;
    bool requires_prefix_projection() const noexcept;
    int32_t source_id() const noexcept;

private:
    void clear() noexcept;
    server_prompt_cache * cache_ = nullptr;
    server_prompt_cache_state * source_ = nullptr;
    server_prompt_cache_vbr_owner payload_;
    server_prompt_cache_vbr_owner fallback_payload_;
    common_cache_family_binding cache_family_;
    uint64_t prefix_tokens_ = 0;
    uint64_t source_tokens_ = 0;
    llama_pos selected_next_position_ = -1;
    bool requires_prefix_projection_ = false;
    int32_t source_id_ = -1;
    int32_t prepared_slot_ = -1;
    server_prompt * prepared_destination_ = nullptr;
    server_prompt * adopted_destination_ = nullptr;
    std::unique_ptr<server_prompt> prepared_prompt_;
    friend struct server_prompt_cache;
    friend class server_prompt_cache_vbr_replacement_ticket;
};

// fork 9l-b: replacement ticket + diagnostics (fork server-task.h 1022-1100)
class server_cache_recovery_pin;

// Move-only pre-import capability for replacing an occupied live prompt. It
// owns both durable host pins and the provisional prompt/launch association
// through the core import transaction. The adopter's no-fail callback performs
// publication; destruction or move-overwrite before then leaves the incumbent
// prompt and canonical sidecar association untouched.
class server_prompt_cache_vbr_replacement_ticket {
public:
    server_prompt_cache_vbr_replacement_ticket() = default;
    ~server_prompt_cache_vbr_replacement_ticket();
    server_prompt_cache_vbr_replacement_ticket(
        const server_prompt_cache_vbr_replacement_ticket &) = delete;
    server_prompt_cache_vbr_replacement_ticket & operator=(
        const server_prompt_cache_vbr_replacement_ticket &) = delete;
    server_prompt_cache_vbr_replacement_ticket(
        server_prompt_cache_vbr_replacement_ticket && other) noexcept;
    server_prompt_cache_vbr_replacement_ticket & operator=(
        server_prompt_cache_vbr_replacement_ticket && other) noexcept;

    // Strong recheck of the prompt, family, retention lineage, host owners,
    // recovery pin, and current lease classification bound by preparation.
    bool ready() const noexcept;
    const server_prompt & replacement_prompt() const noexcept;
    const server_prompt_cache_vbr_owner & incoming_payload() const noexcept;
    const server_prompt_cache_vbr_owner & recovery_payload() const noexcept;
    int32_t destination_slot() const noexcept;
    uint64_t incoming_prefix_tokens() const noexcept;
    uint64_t incumbent_tokens() const noexcept;
    uint64_t incumbent_live_lcp() const noexcept;
    llama_cache_acct_artifact_id incumbent_artifact() const noexcept;
    llama_cache_acct_artifact_id incoming_owner_artifact() const noexcept;
    llama_cache_acct_artifact_id recovery_owner_artifact() const noexcept;
    llama_cache_acct_artifact_id recovery_host_artifact() const noexcept;
    llama_cache_acct_artifact_id provisional_artifact() const noexcept;

private:
    void clear() noexcept;
    server_prompt_cache * cache_ = nullptr;
    server_prompt_cache_vbr_restore_candidate incoming_;
    server_prompt_cache_state * recovery_source_ = nullptr;
    server_prompt_cache_vbr_owner recovery_owner_;
    std::unique_ptr<server_cache_recovery_pin> recovery_pin_;
    std::vector<llama_cache_acct_op_id> recovery_ops_;
    server_prompt * incumbent_ = nullptr;
    common_cache_family_binding * incumbent_family_current_ = nullptr;
    common_cache_family_binding incumbent_family_;
    common_cache_family_binding incoming_family_;
    std::string execution_identity_;
    std::string adapter_config_key_;
    std::unique_ptr<server_prompt> replacement_prompt_;
    server_retention_instance_key provisional_key_;
    int32_t destination_slot_ = -1;
    uint64_t incoming_prefix_tokens_ = 0;
    uint64_t incumbent_tokens_ = 0;
    uint64_t incumbent_live_lcp_ = 0;
    uint64_t incumbent_sequence_epoch_ = 0;
    std::array<uint8_t, 32> incoming_token_digest_ = {};
    std::array<uint8_t, 32> incumbent_token_digest_ = {};
    server_cache_lease_identity incumbent_lease_identity_;
    llama_cache_acct_artifact_id incumbent_artifact_;
    uint64_t incumbent_lineage_ = 0;
    llama_cache_acct_artifact_id incoming_owner_artifact_;
    llama_cache_acct_artifact_id recovery_owner_artifact_;
    llama_cache_acct_artifact_id recovery_host_artifact_;
    llama_cache_acct_artifact_id provisional_artifact_;
    bool publish_prepared_ = false;
    bool published_ = false;
    friend struct server_prompt_cache;
};

// Bounded recovery-selection evidence for occupied replacement. Published
// host prompts carry a mutation-invalidated token digest, so a state-list scan
// performs raw token comparison only for the one unambiguous digest match.
struct server_prompt_cache_vbr_replacement_diagnostics {
    uint64_t recovery_states_visited = 0;
    uint64_t recovery_digest_matches = 0;
    uint64_t recovery_raw_token_comparisons = 0;
};


// fork 9l-c: server_prompt_retention_publish_exact_prefix decl (fork server-task.h 1335-1340)
bool server_prompt_retention_publish_exact_prefix(
    server_retention_sidecar_store & retention,
    const server_retention_instance_key & key,
    const server_prompt & prompt,
    const std::string & adapter_config_key,
    int64_t coverage_tokens) noexcept;

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    // The fork marks the state member and the detach hook below private;
    // this struct carries no access specifiers, so they stay public here.
    // active_storage_ is touched only from member bodies, so the
    // visibility cannot matter.
    // fork server-task.h:1345-1353
    // Scheduler-owned, non-evicting reservation for process-local active
    // companions and restore staging. Readers share one charge; the last
    // reader releases it, even after this cache has been destroyed. Ordinary
    // host publications include these reservations in their capacity checks.
    std::shared_ptr<void> reserve_active_storage(size_t bytes, size_t tokens = 0);
    size_t active_storage_bytes() const noexcept;
    size_t active_storage_tokens() const noexcept;
    bool fits_bytes(size_t host_bytes) const noexcept;
    size_t effective_host_token_limit(size_t host_bytes, size_t host_tokens) const noexcept;

    // fork server-task.h:1356-1358
    struct active_storage_state;
    std::shared_ptr<active_storage_state> active_storage_;
    void detach_active_storage_accounting() noexcept;

    std::list<server_prompt_cache_state> states;

// fork 9l-d1: iterator typedefs (fork server-task.h 1362-1363)
    using iterator = std::list<server_prompt_cache_state>::iterator;
    using const_iterator = std::list<server_prompt_cache_state>::const_iterator;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    void update();

    // ---------------------------------------------------------------------
    // group C tip: the only part of the server_prompt_cache rewrite that the
    // cache-plan observation face actually calls. Everything else in that
    // class (118 members, 69 out-of-line bodies, 26 types) stays out until a
    // consumer lands. Verbatim from the fork; coordinates on each line.
    // ---------------------------------------------------------------------

    // fork server-task.h:1371-1374
    // Independent quality-only pool, in bytes. Zero is the valid
    // compact-only configuration: anchors are optional and are stripped
    // without invalidating their compact current artifact.
    size_t limit_anchor_size = 0;
    // fork server-task.h:1375-1377
    // Explicit VBR restore product gate. Keeping this false preserves literal zero
    // anchor work unless a nonzero --vbr-anchor-cache-mib pool is requested.
    bool quality_anchor_budget_enabled = false;
    // fork server-task.h:1379
    int32_t cache_plan_next_source_id = 0;

    // fork server-task.h:1567 / server-task.cpp:2418
    void cache_plan_begin_inventory() noexcept;
    bool cache_plan_get_source_id(
        server_prompt_cache_state & state,
        int32_t & source_id) noexcept;
    // fork server-task.h:1696
// fork 9l-d1: destruction observer member (fork server-task.h 1695)
    server_cache_destruction_observer * destruction_obs = nullptr;

    server_retention_sidecar_store * retention_obs = nullptr;
// fork 9l-b: lease_obs member (fork server-task.h 1697)
    server_cache_lease_table * lease_obs = nullptr;

// fork 9l-c: lease execution identity (fork server-task.h 1698)
    const std::string * lease_execution_identity = nullptr;

// fork 9l-d1: debug observability members (fork server-task.h 1701, 1703)
    bool debug_observability = false;
    uint64_t debug_lifecycle_emissions = 0;
    // fork server-task.h:1651 (the fork marks this private; this struct has no access specifiers)
    // Reused by the scheduler-owned durability classifier. Durable stems
    // accumulate across capture waves, so this arena follows the full host
    // candidate bound rather than the <=8 manifests admitted by one wave.
    mutable std::vector<server_prompt_cache_vbr_frontier_query *>
        vbr_stem_witness_arena;

    // fork server-task.h:1391-1397  (7 lines)
    // Exact immutable VBR-frontier presence query. This alone is not a clear
    // authority: automatic displacement also revalidates the live semantic
    // frontier, scheduler state, pending work, and lease/recovery protection.
    bool contains_vbr_frontier(
        const server_prompt & prompt,
        const std::string & execution_identity,
        const std::string & adapter_config_key) const noexcept;
    // Read-only suppression check for an already-durable shorter frontier.
    // The host package must be exact for coverage and the current live prompt
    // must still carry that exact prefix under the same source epoch.
    // Sorts the populated prefix of the caller-owned query arena and marks
    // immutable identity matches in one bounded pass over host states. This
    // is selection evidence, not permission to clear a live source.
    bool mark_vbr_frontiers(
        server_prompt_cache_vbr_frontier_query * queries,
        size_t query_count,
        server_prompt_cache_vbr_frontier_batch_diagnostics * diagnostics =
            nullptr)
        const noexcept;

// fork 9l-a: prepare/release decls

    // Select and pin the longest VBR artifact whose complete token block is an
    // exact prefix of the incoming text-only request. On an exact miss, select
    // the eligible host artifact with the longest nonzero common prefix. At an
    // equal frontier, prefer a quality anchor while retaining compact-current
    // as a bounded pre-adoption fallback.
    // Media-bearing automatic restore is deliberately deferred until lookup
    // owns a frontier-media authority.
    // A non-null required_family constrains selection at the index owner so a
    // higher-ranked foreign-family row cannot mask a compatible candidate.
    // This is deliberately separate
    // from fixed-state load()/contains(): VBR restoration is an adopt
    // transaction, not a serialized state-image restore.
    bool prepare_vbr_restore(
        const server_tokens & request_tokens,
        const std::string & execution_identity,
        const std::string & adapter_config_key,
        server_prompt_cache_vbr_restore_candidate & candidate,
        bool allow_prefix_projection = true,
        const common_cache_family_binding * required_family = nullptr) noexcept;
    // Fallible retention/lease preparation for a construction-empty live
    // destination. Cancellation rolls this provisional association back;
    // commit consumes it only after successful artifact adoption.
    bool prepare_vbr_restore_destination(
        server_prompt_cache_vbr_restore_candidate & candidate,
        server_prompt & destination,
        int32_t id_slot) noexcept;

// fork 9l-d1: publish_vbr_restore + commit_vbr_restore decls (fork server-task.h 1504-1517)
    // The adopter's allocation-free final publish hook. Only this method can
    // mint the receipt consumed by commit: it installs the cache-prepared
    // exact-or-projected source prompt into the construction-empty destination.
    bool publish_vbr_restore(
        server_prompt_cache_vbr_restore_candidate & candidate) noexcept;
    // Scheduler metadata terminal after the store's atomic adopt transaction
    // has returned success. This must never be used as the adopt publish hook:
    // every fallible retention operation was completed by the destination
    // preparation above, outside the library no-fail region.
    bool commit_vbr_restore(
        server_prompt_cache_vbr_restore_candidate & candidate,
        server_prompt & destination,
        common_cache_family_binding & destination_family,
        int32_t id_slot) noexcept;

    void release_vbr_restore(
        server_prompt_cache_vbr_restore_candidate & candidate) noexcept;
    friend class server_prompt_cache_vbr_restore_candidate;
// fork 9l-b: release_vbr_occupied_replacement + ticket friend (fork server-task.h 1673-1674, 1677)
    void release_vbr_occupied_replacement(
        server_prompt_cache_vbr_replacement_ticket & ticket) noexcept;
    friend class server_prompt_cache_vbr_replacement_ticket;

// fork 9l-d1: occupied-replacement decls (fork server-task.h 1519-1549)
    // CPU-only preparation for an occupied destination. The incoming
    // candidate must be exact (not a parent projection) and must improve the
    // live common prefix. Preparation clones all replacement metadata into a
    // private provisional launch association without mutating or retiring the
    // incumbent slot. The occupied importer consumes the ticket through its
    // allocation-free composite KV/prompt/retention publication callback.
    bool prepare_vbr_occupied_replacement(
        server_prompt_cache_vbr_restore_candidate && incoming,
        server_prompt & incumbent,
        common_cache_family_binding & incumbent_family,
        const common_cache_family_binding & incoming_family,
        int32_t id_slot,
        const std::string & execution_identity,
        const std::string & adapter_config_key,
        server_prompt_cache_vbr_replacement_ticket & ticket,
        server_prompt_cache_vbr_replacement_diagnostics * diagnostics =
            nullptr) noexcept;
    // The scheduler calls the fallible read half before entering the adopter's
    // no-fail terminal. Publication swaps the already-existing sidecar
    // association and prompt storage without first clearing the incumbent.
    bool prepare_vbr_occupied_replacement_publish(
        server_prompt_cache_vbr_replacement_ticket & ticket) noexcept;
    void publish_vbr_occupied_replacement(
        server_prompt_cache_vbr_replacement_ticket & ticket) noexcept;
    // Post-adopt metadata terminal. The replacement is a request branch and
    // therefore publishes the separately bound incoming request family.
    void commit_vbr_occupied_replacement(
        server_prompt_cache_vbr_replacement_ticket & ticket,
        server_prompt & destination,
        common_cache_family_binding & destination_family,
        int32_t id_slot) noexcept;

    // fork server-task.h:1691
    llama_cache_acct_ledger * acct = nullptr;

    // FORK-DIFF: the fork's destructor body also calls clear_accounting(),
    // a different (not yet landed) member of the same rewrite. Only the
    // detach hook is reachable from what exists here; the omitted call
    // cleans accounting this tree never creates, so the observable result
    // is unchanged.
    // fork server-task.h:1712-1715
    ~server_prompt_cache() {
        detach_active_storage_accounting();
    }

// fork 9l-d3c2: destination gate for the shadow workspace (fork server-task.h 1702)
    bool retention_capacity_authority = false;

// fork 9l-d3c2: shadow workspace allocator + snapshot accessor (fork server-task.h 1730-1734)
    bool enable_retention_shadow() noexcept;

    server_prompt_cache_shadow_snapshot retention_shadow_snapshot() const noexcept {
        return retention_shadow;
    }

// fork 9l-d3c2: the four shadow members (fork server-task.h 1796-1801)
    std::unique_ptr<server_prompt_cache_shadow_row[]> retention_shadow_rows;
    std::unique_ptr<server_prompt_cache_shadow_artifact_slot[]>
        retention_shadow_artifacts;
    std::unique_ptr<server_prompt_cache_shadow_lineage_slot[]>
        retention_shadow_lineages;
    server_prompt_cache_shadow_snapshot retention_shadow;

};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};
