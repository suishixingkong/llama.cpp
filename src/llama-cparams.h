#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    bool rs_rollback_prompt_only = false; // rollback snapshots are needed only during prompt processing
    uint32_t n_draft_max;     // max draft tokens a single generation ubatch can carry (speculative decoding)
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    // CUDA prefill GEMM tile for lossless weight reuse, 0 = disabled [EXPERIMENTAL, Volta only]
    uint32_t prefill_reuse = 0;

    int32_t  nextn_layer_offset = 0;

    enum llama_moe_cache_mode moe_cache_mode;
    size_t moe_cache_budget_mib;
    int32_t moe_cache_expert_parallel;
    int32_t moe_cache_cpu_overlap;
    std::string moe_cache_profile_path;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
    bool training;           // set by llama_opt_init()

    bool vbr_dynamic;
    enum llama_vbr_codec vbr_codec = LLAMA_VBR_CODEC_TURBO;

    double vbr_min_bits = 0.0;
    uint64_t vbr_vram_budget_bytes = 0;
    uint64_t vbr_growth_headroom_bytes = 0;
    bool vbr_budget_explicit = false;
    bool vbr_min_bits_explicit = false;
    // mixed-config side pins, see llama.h vbr_pin_k
    bool vbr_pin_k = false;
    bool vbr_pin_v = false;

    uint32_t kv_stream_arena_mib;

    size_t moe_cache_size;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
