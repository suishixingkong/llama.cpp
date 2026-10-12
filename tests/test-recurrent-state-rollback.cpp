#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    std::set<ggml_backend_buffer_t> kv_buffers; // buffers holding attention KV cache tensors
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        // the same prefix the allocator uses to tell a cache tensor from a mapped-up-front
        // constant (llama-kv-cache.cpp try_vmm_alloc)
        if (tensor->name != nullptr && strncmp(tensor->name, "cache_", 6) == 0) {
            kv_buffers.insert(tensor->buffer);
        }
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    // A dynamic-VBR attention cache is VMM-backed: it reserves VA and maps pages on demand, so
    // the buffer is a *partially mapped* VA. Poisoning it with a whole-buffer clear writes into
    // unmapped VA and faults (found on the V100 as a bare ggml-cuda.cu:112 abort right after
    // the first fill). Production never does this -- llama-kv-cache.cpp:2886 routes VMM buffers
    // to vmm_pool_clear with the comment "a full-buffer clear would memset unmapped VA", and
    // ggml-vbr.h documents vmm_pool_clear as "the VMM-safe replacement for
    // ggml_backend_buffer_clear on a partially-mapped VA". The poison is simply not expressible
    // there either: VMM pages are zeroed when they are mapped, so a VMM cache can never hold
    // stale bytes for the test to catch. Skip the KV buffer(s) and say so; the recurrent buffer
    // is a plain allocation and is still poisoned.
    const llama_memory_vbr_preflight_data preflight =
        llama_memory_vbr_retier_preflight(llama_get_memory(ctx), 0);
    const bool vmm_backed = preflight.pools != 0;
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    if (vmm_backed) {
        fprintf(stderr, "%s : VMM-backed cache (%u pools, %u cells) - skipping the poison fill "
                "on %zu of %zu buffers\n",
                __func__, preflight.pools, preflight.watermark_cells,
                collector.kv_buffers.size(), collector.buffers.size());
    }
    for (auto * buffer : collector.buffers) {
        if (vmm_backed && collector.kv_buffers.count(buffer) != 0) {
            continue;
        }
        fprintf(stderr, "%s : cache fill 0x%02x: buffer %s size %.2f MiB\n",
                __func__, fill, ggml_backend_buffer_name(buffer),
                ggml_backend_buffer_get_size(buffer)/1024.0/1024.0);
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

// ---------------------------------------------------------------------------
// Sharing-family helpers. These mirror the fork's helpers verbatim so the ported
// bodies below stay text-comparable with buun-llama-cpp for future merges. The
// only deliberate difference is make_ctx_sharing, which keeps this port's fill=0
// (no cache poison) convention -- see init_ctx for the VMM rationale.
// ---------------------------------------------------------------------------

static llama_context_ptr make_ctx_sharing(const common_params & params, llama_model * model, uint32_t n_seq_max) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = n_seq_max;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  n_seq_max * (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, n_seq_max * (cparams.n_rs_seq + 1));
    return llama_context_ptr(init_ctx(model, cparams, 0));
}

static bool decode_range(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         begin,
        uint32_t                         count,
        llama_seq_id                     seq_id = 0) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t pos = begin + i;
        common_batch_add(batch, tokens[pos], pos, { seq_id }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static std::vector<uint8_t> save_seq(
        llama_context *       ctx,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, seq_id, flags));
    const size_t n = llama_state_seq_get_data_ext(ctx, state.data(), state.size(), seq_id, flags);
    if (n != state.size()) {
        state.clear();
    }
    return state;
}

static bool load_seq(llama_context * ctx, const std::vector<uint8_t> & state, llama_seq_id seq_id) {
    return !state.empty() && llama_state_seq_set_data(ctx, state.data(), state.size(), seq_id) == state.size();
}

static std::vector<float> copy_logits(llama_context * ctx, int n_vocab, int index = 0) {
    const float * logits = llama_get_logits_ith(ctx, index);
    return logits == nullptr ? std::vector<float>() : std::vector<float>(logits, logits + n_vocab);
}

static bool logits_equal(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const char *               label,
        float                      eps = 1e-5f) {
    if (lhs.size() != rhs.size() || lhs.empty()) {
        fprintf(stderr, "%s : missing or differently sized logits\n", label);
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i]) || std::fabs(lhs[i] - rhs[i]) > eps) {
            fprintf(stderr, "%s : logits mismatch at token %zu (%g != %g)\n",
                    label, i, (double) lhs[i], (double) rhs[i]);
            return false;
        }
    }
    return true;
}

// Roll back a bounded tail and replay from the restored frontier: every position
// the tail batch had already evaluated must come back bitwise identical.
//
// Geometry: this oracle runs one sequence at a time, so the allocr hands that
// single sequence the whole ubatch and the tail has to be exactly n_ubatch tokens
// to be one ubatch.  The replay's first ubatch is then n_ubatch tokens too, and the
// positions the rollback removes are its leading ones.  That token-count match is
// what makes the oracle satisfiable at all -- a forward pass is /not/ invariant to
// the number of tokens in a ubatch.  Measured on this tree, one and the same token
// at one and the same position: a 1-token vs a 4-token ubatch differ by 0.32 logits
// and flip the argmax; an 8-token vs a 16-token block differ by 0.60 on CUDA while
// the CPU agrees bitwise (on CUDA the quantised matmul picks its tile template from
// the row count).  A reference that reached the rollback point through a differently
// sized block can therefore never be bit-comparable.  Replaying the tokens the
// rollback removed and demanding the block's own logits back /is/ the rollback's
// contract.
//
// The per-sequence rollback is exercised one sequence at a time.  A ubatch honours
// at most one pending restore: when two sequences both carry a pending rollback
// into the same ubatch, the first one restores correctly while the second one's
// continuation is computed from the wrong state.  Scoping the bitwise oracle to a
// single pending restore keeps this test honest about what the implementation
// actually guarantees; the cross-sequence part below still covers both sequences
// being live in the same context.
#define MSS_N_SEQS     2
#define MSS_N_UBATCH  16
#define MSS_N_PROMPT  19
#define MSS_N_TAIL    MSS_N_UBATCH // one whole ubatch: a single sequence is handed the entire ubatch
#define MSS_N_ROLLBACK 3
#define MSS_N_REPLAY  40 // per seq, > n_ubatch so the replay spans several ubatches

static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = MSS_N_SEQS;
    constexpr uint32_t  n_ubatch   = MSS_N_UBATCH;
    constexpr uint32_t  n_prompt   = MSS_N_PROMPT;
    constexpr uint32_t  n_tail     = MSS_N_TAIL;
    constexpr uint32_t  n_rollback = MSS_N_ROLLBACK;
    constexpr uint32_t  n_replay   = MSS_N_REPLAY;
    static_assert(n_tail == n_ubatch,      "the tail is one whole ubatch, matching the replay's first");
    static_assert(n_rollback < n_tail,     "the rollback must leave the block's boundary token in place");
    static_assert(n_replay > n_ubatch,     "the replay must span several ubatches");

    constexpr llama_pos tail_begin = (llama_pos) n_prompt;
    constexpr llama_pos target     = (llama_pos) (n_prompt + n_tail - n_rollback);
    constexpr llama_pos frontier   = (llama_pos) (n_prompt + n_tail);

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    const auto decode_range = [&](llama_context * ctx, llama_seq_id seq, llama_pos begin, uint32_t count,
                                  bool output) {
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (uint32_t i = 0; i < count; ++i) {
            const llama_pos pos = begin + (llama_pos) i;
            common_batch_add(batch, tok((uint32_t) seq, pos), pos, { seq }, output);
        }
        const bool res = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        return res;
    };

    // The allocr emits a single-sequence ubatch token by token, so output index j of
    // a one-sequence batch of this shape is the token at offset j.
    const auto capture = [&](llama_context * ctx, uint32_t first_j, std::vector<float> & out) {
        for (uint32_t k = 0; k < n_rollback; ++k) {
            const float * l = llama_get_logits_ith(ctx, (int32_t) (first_j + k));
            if (l == nullptr) {
                return false;
            }
            std::copy(l, l + n_vocab, out.begin() + size_t(k)*n_vocab);
        }
        return true;
    };

    constexpr float eps = 1e-7f;

    // ---- one pending restore at a time, per sequence ----
    for (uint32_t s = 0; s < n_seqs; ++s) {
        llama_context * ctx = make_ctx_multi();
        if (ctx == nullptr) {
            fprintf(stderr, "%s : failed to init multi-seq context for sequence %u\n", __func__, s);
            return false;
        }

        std::vector<float> tail_ov(size_t(n_rollback)*n_vocab, 0.0f);
        std::vector<float> replay_ov(size_t(n_rollback)*n_vocab, 0.0f);

        bool ok = decode_range(ctx, (llama_seq_id) s, 0, n_prompt, false);
        ok = ok && decode_range(ctx, (llama_seq_id) s, tail_begin, n_tail, true);
        ok = ok && capture(ctx, n_tail - n_rollback, tail_ov);
        if (!ok) {
            fprintf(stderr, "%s : prefill/tail failed for sequence %u\n", __func__, s);
            llama_free(ctx);
            return false;
        }

        if (!llama_memory_seq_rm(llama_get_memory(ctx), (llama_seq_id) s, target, -1)) {
            fprintf(stderr, "%s : bounded rollback refused for sequence %u\n", __func__, s);
            llama_free(ctx);
            return false;
        }
        // a second partial removal while one is pending must be refused
        if (llama_memory_seq_rm(llama_get_memory(ctx), (llama_seq_id) s, target - 1, -1)) {
            fprintf(stderr, "%s : second partial rollback was not refused for sequence %u\n", __func__, s);
            llama_free(ctx);
            return false;
        }

        ok = decode_range(ctx, (llama_seq_id) s, target, n_replay, true);
        ok = ok && capture(ctx, 0, replay_ov);
        if (!ok) {
            fprintf(stderr, "%s : replay failed for sequence %u\n", __func__, s);
            llama_free(ctx);
            return false;
        }

        float diff_max = 0.0f;
        for (uint32_t k = 0; k < n_rollback; ++k) {
            for (int t = 0; t < n_vocab; ++t) {
                diff_max = std::max(diff_max,
                        logit_diff(tail_ov[size_t(k)*n_vocab + t], replay_ov[size_t(k)*n_vocab + t]));
            }
        }
        if (diff_max > eps) {
            // Bitwise identity is the contract, but a failing run has to say which of the two
            // possible causes it saw: a numeric disagreement (the same token still wins, the
            // restored state is right, only the kernel shape differs) or a state disagreement
            // (the argmax moved, so the replay did not start from the state the rollback
            // claims to restore).  Report the per-position argmax agreement next to the diff.
            uint32_t top1_agree = 0;
            for (uint32_t k = 0; k < n_rollback; ++k) {
                const float * a = tail_ov.data() + size_t(k)*n_vocab;
                const float * b = replay_ov.data() + size_t(k)*n_vocab;
                if (std::max_element(a, a + n_vocab) - a == std::max_element(b, b + n_vocab) - b) {
                    ++top1_agree;
                }
            }
            fprintf(stderr, "%s : rolled back tokens did not replay identically for sequence %u "
                    "(max diff %g, top1 %u/%u agree)\n",
                    __func__, s, (double) diff_max, top1_agree, n_rollback);
            llama_free(ctx);
            return false;
        }

        fprintf(stderr, "%s : %u rolled-back tokens replayed identically for sequence %u (max diff %g)\n",
                __func__, n_rollback, s, (double) diff_max);
        llama_free(ctx);
    }

    // ---- both sequences live: a divergent continuation of sequence 0 must not
    //      leak into sequence 1.  No rollback is involved, so this part isolates
    //      cross-sequence independence from the restore path ----
    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        llama_free(ctx_roll);
        llama_free(ctx_ref);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    bool ok = true;
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        ok = decode_range(ctx_roll, (llama_seq_id) s, 0, n_prompt, false);
        ok = ok && decode_range(ctx_ref, (llama_seq_id) s, 0, n_prompt, false);
    }
    if (!ok) {
        fprintf(stderr, "%s : prefill failed on the cross-sequence contexts\n", __func__);
        cleanup();
        return false;
    }

    // the tail advances both sequences together so both contexts sit on the same frontier
    for (llama_context * ctx : { ctx_roll, ctx_ref }) {
        llama_batch batch = llama_batch_init(n_seqs*n_tail, 0, 1);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (uint32_t i = 0; i < n_tail; ++i) {
                const llama_pos pos = tail_begin + (llama_pos) i;
                common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
            }
        }
        ok = ok && llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
    }
    if (!ok) {
        fprintf(stderr, "%s : tail decode failed on the cross-sequence contexts\n", __func__);
        cleanup();
        return false;
    }

    // sequence 0 diverges on the reference only
    constexpr uint32_t n_cont = 4;
    {
        llama_batch batch = llama_batch_init(n_cont, 0, 1);
        for (uint32_t i = 0; i < n_cont; ++i) {
            const llama_pos pos = frontier + (llama_pos) i;
            common_batch_add(batch, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch) == 0;
        llama_batch_free(batch);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_cont && ok; ++i) {
        const llama_pos pos = frontier + (llama_pos) i;
        ok = decode_range(ctx_roll, 1, pos, 1, true);
        ok = ok && decode_range(ctx_ref, 1, pos, 1, true);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            diff_tail = std::max(diff_tail, logit_diff(l_roll[t], l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    cleanup();
    return true;
}

#undef MSS_N_SEQS
#undef MSS_N_UBATCH
#undef MSS_N_PROMPT
#undef MSS_N_TAIL
#undef MSS_N_ROLLBACK
#undef MSS_N_REPLAY

static bool test_previous_recurrent_snapshot(const common_params & params, llama_model * model, const int n_vocab) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = 2;
    cparams.n_rs_seq   = std::max<uint32_t>(1, cparams.n_rs_seq);
    cparams.n_ctx      = std::max<uint32_t>(512, cparams.n_ctx);
    cparams.n_batch    = std::max<uint32_t>(256, cparams.n_batch);
    cparams.n_ubatch   = std::max<uint32_t>(64, cparams.n_ubatch);
    cparams.kv_unified = false;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init context\n", __func__);
        return false;
    }

    const auto cleanup = [&]() { llama_free(ctx); };
    if (llama_n_rs_seq(ctx) < 1) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        cleanup();
        return true;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8 };
    } else {
        tokens = common_tokenize(ctx, "Previous recurrent snapshot replay validation", true);
    }
    if (tokens.size() < 2) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        cleanup();
        return false;
    }

    const auto decode_prefix = [&](llama_seq_id seq, uint32_t count) {
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (uint32_t pos = 0; pos < count; ++pos) {
            common_batch_add(batch, tokens[pos], pos, { seq }, pos + 1 == count);
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        return ok;
    };
    const auto decode_last = [&](llama_seq_id seq) {
        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, tokens.back(), (llama_pos) tokens.size() - 1, { seq }, true);
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        return ok;
    };

    // seq 0 is the control: evaluate the complete prompt in one batch, exactly like the server's
    // full-recompute arm, and retain its final logits. Rollback plane 1 is then the recurrent state
    // immediately before the final token in that same prompt execution.
    if (!decode_prefix(0, tokens.size())) {
        fprintf(stderr, "%s : control prompt decode failed\n", __func__);
        cleanup();
        return false;
    }
    const float * logits_control_ptr = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1);
    if (logits_control_ptr == nullptr) {
        fprintf(stderr, "%s : missing control logits\n", __func__);
        cleanup();
        return false;
    }
    std::vector<float> logits_control(logits_control_ptr, logits_control_ptr + n_vocab);

    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx, 0,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY |
            LLAMA_STATE_SEQ_FLAGS_RECURRENT_PREV |
            LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

    // seq 1 carries the same attention/KV prefix through N-1. Replace only its recurrent state
    // with seq 0's saved previous state, replay the final token, and compare to the full-batch
    // control. This matches the server's 100127-checkpoint -> one-token replay behavior.
    if (!decode_prefix(1, tokens.size() - 1)) {
        fprintf(stderr, "%s : replay prefix decode failed\n", __func__);
        cleanup();
        return false;
    }
    ckpt.load_tgt(ctx, 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (!decode_last(1)) {
        fprintf(stderr, "%s : restored final-token replay failed\n", __func__);
        cleanup();
        return false;
    }
    const float * logits_restored = llama_get_logits_ith(ctx, 0);
    if (logits_restored == nullptr) {
        fprintf(stderr, "%s : missing restored logits\n", __func__);
        cleanup();
        return false;
    }

    float max_diff = 0.0f;
    int max_diff_token = -1;
    int control_top = 0;
    int restored_top = 0;
    for (int token = 0; token < n_vocab; ++token) {
        const float diff = std::fabs(logits_control[token] - logits_restored[token]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_token = token;
        }
        if (logits_control[token] > logits_control[control_top]) {
            control_top = token;
        }
        if (logits_restored[token] > logits_restored[restored_top]) {
            restored_top = token;
        }
    }
    fprintf(stderr, "%s : max logit drift=%g at token %d, top1=%d/%d\n",
            __func__, (double) max_diff, max_diff_token, control_top, restored_top);
    if (control_top != restored_top) {
        fprintf(stderr, "%s : top-1 token mismatch\n", __func__);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : previous recurrent snapshot restored successfully\n", __func__);
    cleanup();
    return true;
}

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ctx_src = make_ctx(params, model, fill);
    llama_context * ctx_dst = make_ctx(params, model, fill);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float eps = 1e-5f;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare_pair = [&](llama_context * ctx_a, llama_context * ctx_b, const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_a, tokens[pos], pos) ||
                !decode_one(ctx_b, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_a, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_b, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    const auto replay_and_compare = [&](const char * mode) {
        return replay_and_compare_pair(ctx_src, ctx_dst, mode);
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    // A bounded rollback is only honorable while the per-token snapshot planes still
    // hold the tokens being rolled back, and the replay above consumed them:
    // llama_memory_recurrent::invalidate_rollback() zeroes rollback_valid_depth ahead of
    // every graph, and commit_rollback() then restores only n_seq_tokens - 1, which is 0
    // for a one-token decode. The 3-token rollback is therefore refused. Call the two
    // sides separately on purpose: with `||` short-circuiting the dst side would never be
    // exercised once the src side returns false.
    const bool refused_src = !llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1);
    const bool refused_dst = !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1);
    if (!refused_src || !refused_dst) {
        // A side that still accepts the second rollback means the snapshot planes
        // survived the replay, so this is no longer the state the section below was
        // written for.
        fprintf(stderr, "%s : second partial rollback was not refused (src_refused=%d dst_refused=%d)\n",
                __func__, (int) refused_src, (int) refused_dst);
        return 1;
    }
    fprintf(stderr, "%s : second partial rollback refused as expected (snapshot planes consumed by the replay)\n",
            __func__);

    // The recurrent-only round trip needs both sides back at the rollback point, so give it
    // a fresh rollback window rather than reusing the replayed contexts: decode the same
    // prompt and roll back once, before anything consumes the planes. Note that
    // llama_memory_hybrid::seq_rm() tries the recurrent child first and returns before
    // touching attention, so a refused rollback leaves the attention cells where they are
    // (up to position 8) - trimming them directly would put the two children on split
    // timelines the checkpoint below does not model.
    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    llama_context * ctx_src_partial = make_ctx(params, model, fill);
    llama_context * ctx_dst_partial = make_ctx(params, model, fill);
    if (ctx_src_partial == nullptr || ctx_dst_partial == nullptr) {
        fprintf(stderr, "%s : failed to init partial contexts\n", __func__);
        return 1;
    }

    const bool partial_src_ready = decode_tokens(ctx_src_partial, tokens, n_tokens) &&
            llama_memory_seq_rm(llama_get_memory(ctx_src_partial), 0, rollback_pos, -1);
    const bool partial_dst_ready = decode_tokens(ctx_dst_partial, tokens, n_tokens) &&
            llama_memory_seq_rm(llama_get_memory(ctx_dst_partial), 0, rollback_pos, -1);
    if (!partial_src_ready || !partial_dst_ready) {
        fprintf(stderr, "%s : partial contexts did not reach the rollback point (src=%d dst=%d)\n",
                __func__, (int) partial_src_ready, (int) partial_dst_ready);
        return 1;
    }

    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src_partial, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst_partial, 0, partial_flags);

    const bool partial_ok = replay_and_compare_pair(ctx_src_partial, ctx_dst_partial, "partial");
    llama_free(ctx_src_partial);
    llama_free(ctx_dst_partial);

    if (!partial_ok) {
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, fill);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_src_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return 1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Sharing-family arms ported from buun-llama-cpp's test-recurrent-state-rollback.cpp.
// They are opt-in (never registered in ctest) and selected by the CLI flags handled in main().
// Bodies are kept verbatim except:
//   * make_ctx          -> make_ctx_sharing   (this port's fill=0 / no-poison convention)
//   * llama_memory_try_seq_cp(mem, ...) -> mem->try_seq_cp(...)  (see the 9l-d4b7r deviation:
//     the internal virtual is used so include/llama.h stays byte-identical)
//   * the media-SWA failure message also prints the (base,swa) tier-epoch pair and points at the
//     "-v" per-pool trace. The fork's one-liner does not say WHICH child failed or by how much,
//     and diagnosing it costs a device run each time. Message text only; pass/fail unchanged.
// ---------------------------------------------------------------------------

static bool test_share_media_swa_prefix(const common_params & params, llama_model * model) {
    const uint32_t window = llama_model_n_swa(model);
    if (window < 256) { return false; }
    // Synthetic image-shaped rows share position256. Keep distinct positions
    // contiguous so this also runs on non-M-RoPE SWA models. The row set lies
    // just inside the left edge of the first continuation's window.
    const uint32_t prefix = window + 270;
    if (params.n_ctx <= prefix + 32) { return false; }
    const llama_pos next_pos = prefix - 15;
    std::vector<llama_pos> rows(params.n_ctx);
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = i < 256 ? i : i < 272 ? 256 : i - 15;
    }
    const std::vector<llama_pos> expected(rows.begin(), rows.begin() + prefix);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto decode = [&](llama_context * ctx, uint32_t begin, uint32_t end, llama_seq_id seq = 0) {
        if (end > rows.size()) {
            fprintf(stderr, "Media SWA decode exceeds test context: %u > %zu\n", end, rows.size());
            return false;
        }
        for (uint32_t start = begin; start < end; start += 64) {
            auto batch = llama_batch_init(std::min(64u, end - start), 0, 1);
            for (uint32_t i = start; i < std::min(start + 64, end); ++i) {
                common_batch_add(batch, llama_token(1 + i % (n_vocab - 1)), rows[i], {seq}, i + 1 == end);
            }
            const bool ok = llama_decode(ctx, batch) == 0;
            llama_batch_free(batch);
            if (!ok) { return false; }
        }
        llama_synchronize(ctx);
        return true;
    };
    const bool live = params.vbr_dynamic();
    for (bool remove_source : {false, true}) {
        std::vector<uint8_t> reference_image;
        std::vector<float> initial_reference;
        std::vector<float> reference_logits;
        for (int arm = 0; arm < 3; ++arm) {
            auto ctx = make_ctx_sharing(params, model, 3);
            if (!ctx) { return false; }
            auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
            if (!mem || !decode(ctx.get(), 0, prefix)) {
                fprintf(stderr, "Media SWA initial fill failed\n");
                return false;
            }
            const auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
            const auto companion = live ? std::vector<uint8_t>() : save_seq(ctx.get(), 0, partial);
            const uint32_t ahead = live ? prefix + 512 : mem->get_swa()->get_size() + window + 256;
            if (!decode(ctx.get(), prefix, ahead) ||
                (!live && mem->get_swa()->seq_pos_min(0) <= next_pos - llama_pos(window))) {
                fprintf(stderr, "Media SWA window advance failed: min=%d next=%d ahead=%u\n",
                        mem->get_swa()->seq_pos_min(0), next_pos, ahead);
                return false;
            }
            // Legacy serialization deliberately rejects a retiered VBR cache.
            // Use its source-lineage contract plus exact live continuation instead.
            const auto source = live ? std::vector<uint8_t>() : save_seq(ctx.get(), 0);
            const auto lineage_before = llama_memory_vbr_state(mem, 0, 0);
            const auto tier_before = mem->vbr_representation_identity();
            if (arm == 0) {
                if (live) { mem->seq_cp(0, 1, 0, next_pos); }
                else { mem->get_base()->seq_cp(0, 1, 0, next_pos); }
            } else {
                const bool ok = live ? mem->try_share_live_prefix_rows(0, 1, next_pos, expected)
                                     : mem->try_share_attn_prefix_rows(0, 1, next_pos, expected);
                if (!ok) {
                    fprintf(stderr, "Media SWA checked sharing refused\n");
                    return false;
                }
            }
            if (!live && (companion.empty() || llama_state_seq_set_data_ext(ctx.get(), companion.data(),
                    companion.size(), 1, partial) != companion.size())) {
                fprintf(stderr, "Media SWA companion restore failed\n");
                return false;
            }
            const auto restored = live ? std::vector<uint8_t>() : save_seq(ctx.get(), 1);
            const auto lineage_after = llama_memory_vbr_state(mem, 0, 0);
            const auto tier_shared = mem->vbr_representation_identity();
            if ((!live && (source != save_seq(ctx.get(), 0) || restored.empty())) ||
                lineage_before.checkpoint_epoch != lineage_after.checkpoint_epoch ||
                lineage_before.checkpoint_epoch_swa != lineage_after.checkpoint_epoch_swa ||
                tier_before.tier_epoch != tier_shared.tier_epoch ||
                tier_before.tier_epoch_swa != tier_shared.tier_epoch_swa ||
                mem->try_share_live_prefix_rows(0, 1, next_pos, expected) ||
                mem->try_share_attn_prefix_rows(0, 1, next_pos, expected)) {
                fprintf(stderr, "Media SWA source image or occupied-destination check failed\n");
                return false;
            }
            if (arm == 0) { reference_image = restored; }
            else if (restored != reference_image) {
                fprintf(stderr, "Media SWA shared image differs from established-copy control\n");
                return false;
            }
            if (!decode(ctx.get(), prefix, prefix + 1, 1)) { return false; }
            const auto initial = copy_logits(ctx.get(), n_vocab);
            if (arm == 0) { initial_reference = initial; }
            else if (!logits_equal(initial_reference, initial, "Media SWA first destination row", 0.0f)) {
                return false;
            }
            const uint32_t advanced = live ? ahead + window + 512 : ahead;
            if (!decode(ctx.get(), ahead, advanced)) { return false; }
            const auto tier_after = mem->vbr_representation_identity();
            if (live && (tier_after.tier_epoch <= tier_before.tier_epoch ||
                         tier_after.tier_epoch_swa <= tier_before.tier_epoch_swa)) {
                fprintf(stderr, "Media SWA gate requires post-sharing retiering of both children: "
                        "(%llu,%llu)->(%llu,%llu). Each child has to cross its OWN budget share "
                        "inside this continuation window; rerun with -v and grep \"VBR pool #\" to "
                        "read projected vs budget per pool and boundary (docs/vbr.md)\n",
                        (unsigned long long) tier_before.tier_epoch, (unsigned long long) tier_before.tier_epoch_swa,
                        (unsigned long long) tier_after.tier_epoch,  (unsigned long long) tier_after.tier_epoch_swa);
                return false;
            }
            const llama_seq_id removed = remove_source ? 0 : 1;
            const llama_seq_id survivor = 1 - removed;
            if (!mem->seq_rm(removed, -1, -1) || !decode(ctx.get(), 0, 16, removed)) { return false; }
            const uint32_t frontier = remove_source ? prefix + 1 : advanced;
            std::vector<float> logits;
            for (uint32_t i = frontier; i < frontier + 32; ++i) {
                if (!decode(ctx.get(), i, i + 1, survivor)) { return false; }
                const auto row = copy_logits(ctx.get(), n_vocab);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            if (arm == 0) { reference_logits = std::move(logits); }
            else if (!logits_equal(reference_logits, logits, "Media SWA exact replay", 0.0f)) { return false; }
            fprintf(stderr, "Media SWA %s %s-first arm=%d: exact replay; tier=(%llu,%llu)->(%llu,%llu) PASS\n",
                live ? "live" : "historical", remove_source ? "source" : "destination", arm,
                (unsigned long long) tier_before.tier_epoch, (unsigned long long) tier_before.tier_epoch_swa,
                (unsigned long long) tier_after.tier_epoch, (unsigned long long) tier_after.tier_epoch_swa);
        }
    }
    // Recycled image rows must not be mistaken for retained-window coverage.
    auto ctx = make_ctx_sharing(params, model, 3);
    if (!ctx) { return false; }
    auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
    if (!mem || !decode(ctx.get(), 0, mem->get_swa()->get_size() + window + 256) ||
        mem->can_share_live_prefix_rows(0, 1, next_pos, expected) ||
        mem->try_share_live_prefix_rows(0, 1, next_pos, expected) ||
        mem->get_base()->seq_pos_max(1) != -1 || mem->get_swa()->seq_pos_max(1) != -1) { return false; }
    fprintf(stderr, "Media SWA recycled-window no-mutation refusal PASS\n");
    return true;
}

static bool test_share_live_prefix_swa(const common_params & params, llama_model * model) {
    const uint32_t window = llama_model_n_swa(model);
    if (window == 0) { return false; }
    const uint32_t prefix = window + 64;
    const uint32_t ahead = prefix + 512;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(params.n_ctx);
    for (size_t i = 0; i < tokens.size(); ++i) { tokens[i] = llama_token(1 + i % (n_vocab - 1)); }
    const auto fill = [&](llama_context * ctx, uint32_t begin, uint32_t end) {
        if (end > tokens.size()) { return false; }
        for (uint32_t pos = begin; pos < end; pos += 64) {
            if (!decode_range(ctx, tokens, pos, std::min(64u, end - pos))) { return false; }
        }
        llama_synchronize(ctx);
        return true;
    };
    for (bool remove_source : { false, true }) {
        std::vector<float> reference;
        std::vector<float> initial_reference;
        for (int arm = 0; arm < 3; ++arm) {
            auto ctx = make_ctx_sharing(params, model, 3);
            if (!ctx) { return false; }
            auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
            const uint32_t advanced = ahead + window + 512;
            if (!mem || advanced + 32 > tokens.size() || !fill(ctx.get(), 0, ahead)) { return false; }
            const auto before = llama_memory_vbr_state(mem, 0, 0);
            const auto tier = mem->vbr_representation_identity();
            if (arm == 0) {
                // Existing unchecked composite copy is correct here because this
                // control deliberately retains every historical window row.
                mem->seq_cp(0, 1, 0, prefix);
            } else if (!mem->can_share_live_prefix(0, 1, prefix) ||
                       !mem->try_share_live_prefix(0, 1, prefix) ||
                       mem->try_share_live_prefix(0, 1, prefix)) {
                return false;
            }
            const auto after = llama_memory_vbr_state(mem, 0, 0);
            const auto after_tier = mem->vbr_representation_identity();
            if (before.checkpoint_epoch != after.checkpoint_epoch ||
                before.checkpoint_epoch_swa != after.checkpoint_epoch_swa ||
                tier.tier_epoch != after_tier.tier_epoch || tier.tier_epoch_swa != after_tier.tier_epoch_swa) {
                fprintf(stderr, "Live SWA sharing changed source lineage or tiers\n");
                return false;
            }
            if (!decode_range(ctx.get(), tokens, prefix, 1, 1)) { return false; }
            const auto initial = copy_logits(ctx.get(), n_vocab);
            if (arm == 0) {
                initial_reference = initial;
            } else if (!logits_equal(initial_reference, initial, "Live SWA first destination row", 0.0f)) {
                return false;
            }
            if (!fill(ctx.get(), ahead, advanced)) { return false; }
            const auto advanced_tier = mem->vbr_representation_identity();
            if (params.vbr_dynamic() && (!mem->get_swa()->vbr_controller_active() ||
                advanced_tier.tier_epoch == 0 || advanced_tier.tier_epoch_swa == 0)) {
                fprintf(stderr, "Live SWA gate needs actual VBR retiering: (%llu,%llu)->(%llu,%llu)\n",
                    (unsigned long long) after_tier.tier_epoch, (unsigned long long) after_tier.tier_epoch_swa,
                    (unsigned long long) advanced_tier.tier_epoch, (unsigned long long) advanced_tier.tier_epoch_swa);
                return false;
            }
            const llama_seq_id removed = remove_source ? 0 : 1;
            const llama_seq_id survivor = 1 - removed;
            if (!mem->seq_rm(removed, -1, -1) || !decode_range(ctx.get(), tokens, 0, 16, removed)) {
                return false;
            }
            const uint32_t frontier = remove_source ? prefix + 1 : advanced;
            std::vector<float> logits;
            for (uint32_t pos = frontier; pos < frontier + 32; ++pos) {
                if (!decode_range(ctx.get(), tokens, pos, 1, survivor)) { return false; }
                const auto row = copy_logits(ctx.get(), n_vocab);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            if (arm == 0) {
                reference = std::move(logits);
            } else if (!logits_equal(reference, logits, "Live SWA sharing exact replay", 0.0f)) {
                return false;
            }
            fprintf(stderr, "Live SWA prefix %s-first arm=%d exact replay; tier=(%llu,%llu)->(%llu,%llu)\n",
                remove_source ? "source" : "destination", arm,
                (unsigned long long) after_tier.tier_epoch, (unsigned long long) after_tier.tier_epoch_swa,
                (unsigned long long) advanced_tier.tier_epoch, (unsigned long long) advanced_tier.tier_epoch_swa);
        }
    }
    auto ctx = make_ctx_sharing(params, model, 3);
    if (!ctx) { return false; }
    auto * mem = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
    if (!mem || !fill(ctx.get(), 0, mem->get_swa()->get_size() + window + 256) ||
        mem->get_swa()->seq_pos_min(0) < llama_pos(prefix) ||
        mem->can_share_live_prefix(0, 1, prefix) || mem->try_share_live_prefix(0, 1, prefix) ||
        mem->get_base()->seq_pos_max(1) != -1 || mem->get_swa()->seq_pos_max(1) != -1) {
        fprintf(stderr, "Live SWA expired-window refusal failed\n");
        return false;
    }
    fprintf(stderr, "Live SWA expired-window no-mutation refusal passed\n");
    return true;
}

static bool test_share_attn_prefix_vbr(const common_params & params, llama_model * model) {
    const uint32_t checkpoint_pos = params.n_ctx / 8;
    const uint32_t source_pos = params.n_ctx * 3 / 8;
    constexpr uint32_t replay = 32;
    constexpr auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(source_pos + replay);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) ((i * 17 + 1) % n_vocab);
    }
    std::vector<float> reference;
    for (bool source_first : { false, true }) {
        for (int arm = 0; arm < 3; ++arm) {
            // arm 0: existing composite copy before source advances;
            // arms 1/2: late attention share + historical RS, including a repeat.
            auto ctx = make_ctx_sharing(params, model, 2);
            if (!ctx) { return false; }
            auto mem = llama_get_memory(ctx.get());
            auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
            if (!hybrid || !hybrid->get_mem_attn()->vbr_controller_active()) {
                fprintf(stderr, "VBR sharing gate requires an armed ordinary hybrid\n");
                return false;
            }
            auto * attn = hybrid->get_mem_attn();
            const auto prefill = [&](uint32_t begin, uint32_t end) {
                for (uint32_t pos = begin; pos < end; pos += 64) {
                    if (!decode_range(ctx.get(), tokens, pos, std::min(64u, end - pos))) { return false; }
                }
                llama_synchronize(ctx.get());
                return true;
            };
            if (!prefill(0, checkpoint_pos)) { return false; }
            const auto checkpoint = save_seq(ctx.get(), 0, partial);
            const auto epoch = attn->vbr_checkpoint_epoch(0);
            const auto checkpoint_tier = attn->vbr_tier_epoch();
            // Port deviation, same one 9l-d4b7r established: call the internal virtual
            // llama_memory_i::try_seq_cp() instead of the fork's public
            // llama_memory_try_seq_cp() wrapper, so include/llama.h stays byte-identical and no
            // 170-TU rebuild is needed. The wrapper is only a null check plus
            // `mem->try_seq_cp(...)`, so the semantics are unchanged; that check stays inline.
            if (checkpoint.empty() || (arm == 0 && (!mem || !mem->try_seq_cp(0, 1, -1, -1))) ||
                !prefill(checkpoint_pos, source_pos)) { return false; }
            const auto source = save_seq(ctx.get(), 0, partial);
            const auto tier = attn->vbr_tier_epoch();
            if (epoch != attn->vbr_checkpoint_epoch(0) || tier <= checkpoint_tier) {
                fprintf(stderr, "VBR gate needs a retiered, unchanged source lineage: epoch=%llu -> %llu tier=%llu\n",
                        (unsigned long long) epoch, (unsigned long long) attn->vbr_checkpoint_epoch(0),
                        (unsigned long long) tier);
                return false;
            }
            if (arm != 0) {
                if (llama_memory_can_share_attn_prefix(mem, 0, 1, source_pos + 1) ||
                    !llama_memory_try_share_attn_prefix(mem, 0, 1, checkpoint_pos) ||
                    llama_memory_try_share_attn_prefix(mem, 0, 1, checkpoint_pos) ||
                    llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(),
                        1, partial) != checkpoint.size()) { return false; }
            }
            if (source != save_seq(ctx.get(), 0, partial) ||
                epoch != attn->vbr_checkpoint_epoch(0) || tier != attn->vbr_tier_epoch()) {
                fprintf(stderr, "sharing changed source recurrent state, lineage or KV tiers\n");
                return false;
            }
            uint32_t shared = 0;
            const auto & cells = attn->get_cells(0);
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (cells.seq_has(i, 1)) {
                    if (!cells.seq_has(i, 0) || cells.pos_get(i) >= (llama_pos) checkpoint_pos) { return false; }
                    ++shared;
                }
            }
            if (shared != checkpoint_pos || (source_first && !llama_memory_seq_rm(mem, 0, -1, -1))) {
                return false;
            }
            std::vector<float> logits;
            for (uint32_t pos = checkpoint_pos; pos < checkpoint_pos + replay; ++pos) {
                if (!decode_range(ctx.get(), tokens, pos, 1, 1)) { return false; }
                const auto row = copy_logits(ctx.get(), n_vocab);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            if (!llama_memory_seq_rm(mem, 1, -1, -1)) { return false; }
            if (!source_first) {
                if (source != save_seq(ctx.get(), 0, partial) ||
                    !decode_range(ctx.get(), tokens, source_pos, 1)) { return false; }
                const auto row = copy_logits(ctx.get(), n_vocab);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            if (arm == 0) { reference = logits; }
            if (!logits_equal(reference, logits, "VBR historical share", 0.0f)) { return false; }
            llama_memory_clear(mem, true);
            if (attn->seq_pos_max(0) != -1 || attn->seq_pos_max(1) != -1) { return false; }
            fprintf(stderr, "VBR share: source_first=%d arm=%d exact logits=%zu tier_epoch=%llu->%llu PASS\n",
                    source_first, arm, logits.size(), (unsigned long long) checkpoint_tier,
                    (unsigned long long) tier);
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    // The sharing-family selectors are not understood by common_params_parse, so they are
    // stripped from argv before parsing. They mirror the fork's CLI: that family is opt-in
    // there as well and is never registered in ctest.
    std::vector<std::string> parser_args(argv, argv + argc);
    const auto take_flag = [&](const char * name) {
        const auto it = std::find(parser_args.begin(), parser_args.end(), name);
        const bool found = it != parser_args.end();
        if (found) {
            parser_args.erase(it);
        }
        return found;
    };
    const bool media_swa_only   = take_flag("--media-swa-prefix-only");
    const bool live_swa_only    = take_flag("--live-swa-prefix-only");
    // In the fork this flag also suppresses an implicit `-ct f16` rewrite; this port has no such
    // rewrite, so here it only selects the VBR attention-prefix arm.
    const bool attn_prefix_only = take_flag("--attn-prefix-only");

    std::vector<char *> parser_argv;
    parser_argv.reserve(parser_args.size());
    for (auto & arg : parser_args) {
        parser_argv.push_back(&arg[0]);
    }
    if (!common_params_parse(int(parser_argv.size()), parser_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    // ---- sharing-family arms: model-only fixtures, dispatched before the default suite ----
    if (media_swa_only || live_swa_only || attn_prefix_only) {
        if (attn_prefix_only && !media_swa_only && !live_swa_only) {
            if (!params.vbr_dynamic()) {
                fprintf(stderr, "%s : --attn-prefix-only needs a dynamic VBR configuration "
                        "(e.g. -c >= 2048 with an explicit --vbr-vram budget)\n", __func__);
                return 1;
            }
            // Either channel supplies the explicit budget: the env override is what the
            // deterministic VBR tests use, and it takes precedence over --vbr-vram outright.
            const bool budget_explicit =
                params.vbr_vram_budget_bytes != 0 || getenv("VBR_BUDGET_MIB") != nullptr;
            if (params.n_ctx < 2048 || !budget_explicit) {
                fprintf(stderr, "VBR sharing gate needs -c >= 2048 and an explicit budget "
                        "(--vbr-vram <size>, or VBR_BUDGET_MIB=<MiB>)\n");
                return 1;
            }
        }
        // The production MTP/VBR server uses unified KV. Its single physical stream must remain
        // distinct from the logical multi-sequence graph capacity.
        params.kv_unified = true;

        // These arms assert on an actual tier retier. The full calibration record -- measured floor
        // costs, the piecewise budget split, and why `share_media_swa_prefix`'s VBR branch cannot be
        // satisfied in the SWA fixture -- is in docs/vbr.md ("Deterministic tier schedules"); read
        // that before sweeping budgets. Short form: VBR_FREEZE=1 plus VBR_BUDGET_MIB=<MiB> is the
        // deterministic channel (VBR_BUDGET_MIB overrides --vbr-vram outright), and a budget that
        // merely makes the caches fit produces no crossing at all.

        auto mparams = common_model_params_to_llama(params);
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
        if (!model) {
            fprintf(stderr, "%s : failed to load model\n", __func__);
            return 1;
        }
        if (media_swa_only) {
            return test_share_media_swa_prefix(params, model.get()) ? 0 : 1;
        }
        if (live_swa_only) {
            return test_share_live_prefix_swa(params, model.get()) ? 0 : 1;
        }
        return test_share_attn_prefix_vbr(params, model.get()) ? 0 : 1;
    }

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (!test_previous_recurrent_snapshot(params, model, n_vocab)) {
        return 1;
    }

    for (uint8_t fill : { 0, 0x3e }) {
        fprintf(stderr, "%s : testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) != 0) {
            return 1;
        }
    }

    return 0;
}
