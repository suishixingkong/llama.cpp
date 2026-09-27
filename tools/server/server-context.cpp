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

#endif

constexpr int HTTP_POLLING_SECONDS = 1;

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

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
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

    // fork server-context.cpp:2303
    bool frontier_ratchet_flipped = false;
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
    // fork server-context.cpp:1527
    uint64_t vbr_reuse_capture_frontier = 0;
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
    // fork server-context.cpp:1649
    std::list<server_swa_window_checkpoint> swa_windows;
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

    // ---- group A slot state (verbatim declarations from the fork) ----

    common_retention_pool retention_pool = common_retention_pool::attention;
    json cache_plan_json;
    std::unique_ptr<common_cache_plan_record> cache_plan;

};

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
    uint64_t frontier_next_sequence_epoch = 1;

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

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

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
            update_cache = update_cache && prompt_cache;

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

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
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

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

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

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
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

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

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

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

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

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

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
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
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

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, it->accel.spec.view());

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

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

                    slot.mem.seq_rm(slot.id, p0, -1);

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

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            llama_set_decode_phase(ctx_tgt, decode_phase);
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });

        if (ret != 0) {
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

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
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
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch_view);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
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
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
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


    // ---- group A file-local types (verbatim from the fork) ----

    // fork server-context.cpp:1423
struct server_cache_plan_observer {
    uint64_t records_finalized  = 0;
    uint64_t shadow_unavailable = 0;
};

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

    std::unique_ptr<server_cache_plan_observer> cache_plan_obs;
    std::thread::id cache_plan_preflight_scheduler_thread;
    std::vector<server_live_retention_candidate> vbr_retention_candidates;
    std::vector<int32_t> vbr_retention_candidate_by_seq;
    std::vector<uint8_t> vbr_retention_seen_by_seq;
    std::vector<uint8_t> vbr_durable_sources_by_seq;
    std::vector<uint8_t> vbr_durable_stems_by_seq;
    std::vector<uint32_t> vbr_exclusive_cells_by_seq;
    size_t vbr_retention_candidate_count = 0;
    uint64_t vbr_retention_projection_epoch = 0;
    const std::vector<uint32_t> * vbr_exclusive_cells_for_test = nullptr;
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

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
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
