#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;
class llama_io_write_i;
class llama_io_read_i;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // candidate distribution per drafted token; set it to make draft-simple, draft-mtp and draft-dflash (DFlash2 only) sample
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
// fork common/speculative.h:158
bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// fork: DFlash ring buffer state save/load
//
// FORK-DIFF: the fork implements this family by dispatching over the DFlash
// implementations in `spec->impls` (`common_speculative_impl_dflash`, a fork-only
// backend coupled to CUDA and to the fork's draft-model support). This tree has no
// such implementation, so every entry point below returns the value the fork's own
// loop produces for a spec that contains no DFLASH implementation - see the
// definitions in speculative.cpp for the per-function outcome. The one exception is
// common_speculative_ring_state_serialized_terminal(), which is a pure function of
// the serialized bytes and is ported verbatim.
size_t common_speculative_ring_state_size(const common_speculative * spec);
void   common_speculative_ring_state_save(const common_speculative * spec, uint8_t * buf, size_t size);
bool   common_speculative_ring_state_write(
        const common_speculative * spec,
        llama_io_write_i & output);
bool   common_speculative_ring_state_load(common_speculative * spec, const uint8_t * buf, size_t size);
// Restore a DFlash ring directly from a bounded reader. The reader is consumed
// exactly `size` bytes on success; implementations issue no read larger than
// 1 MiB so retained artifact chains do not need a contiguous materialization.
bool   common_speculative_ring_state_read(
        common_speculative * spec,
        llama_io_read_i & input,
        size_t size);

// Cheap post-install currency. A successful ring mutation changes the epoch;
// callers can therefore close a late publication gate without serializing the
// full ring again.
struct common_speculative_ring_state_currency {
    size_t serialized_bytes = 0;
    llama_pos terminal = -1;
    uint64_t mutation_epoch = 0;
};
bool   common_speculative_ring_state_get_currency(
        const common_speculative * spec,
        common_speculative_ring_state_currency & output);
// Authenticate a serialized DFlash ring against the exact committed prompt
// frontier before it is captured or installed.  The terminal is the logical
// position of the last committed token (committed_len - 1).
bool   common_speculative_ring_state_matches_frontier(
        const common_speculative * spec,
        const uint8_t * buf,
        size_t size,
        llama_pos expected_terminal);
bool   common_speculative_ring_state_terminal(
        const common_speculative * spec,
        llama_pos & terminal);
bool   common_speculative_ring_state_serialized_terminal(
        const uint8_t * buf,
        size_t size,
        llama_pos & terminal);
bool   common_speculative_ring_state_empty(const common_speculative * spec);
void   common_speculative_ring_state_reset(common_speculative * spec);

// fork common/speculative.h:160
enum class common_speculative_sequence_event : uint8_t {
    prompt_rewind = 0,          // retained draft prefix remains installed
    target_restored_without_draft, // target restored; draft must reconstruct
    draft_image_restored,       // complete target+draft sequence images restored
    composite_image_restored,   // draft plus typed accelerator state restored
    live_range_shift,           // live target/draft positions were renumbered
    target_replaced,            // target replaced; draft sequence was cleared
    full_clear,                 // target and draft sequences were cleared
};

// fork common/speculative.h:170-202
// MTP's pending target-hidden row is process-local state: sequence images do
// not serialize it. Keep the validity decision separate from the row storage
// so a restored/cloned sequence cannot consume a row from the previous branch.
class common_speculative_mtp_carry_lifecycle {
public:
    enum class process_mode {
        retained_carry,
        cold_zero,
        target_only,
    };

    bool draft_ready() const noexcept;
    const float * draft_carry(const float * pending_h) const noexcept;
    process_mode target_process_mode(llama_pos first_position) const noexcept;

    void target_process_refreshed() noexcept;
    void target_process_skipped() noexcept;
    void sequence_transition(common_speculative_sequence_event event) noexcept;

private:
    bool ready = false;
};

// Host-checkpoint codec for the deferred MTP hidden row. A complete draft
// sequence image is not usable without this carry at a nonzero frontier.
bool common_speculative_mtp_carry_state_save(
        const common_speculative_mtp_carry_lifecycle & lifecycle,
        const std::vector<float> & pending_h,
        std::vector<uint8_t> & data);
bool common_speculative_mtp_carry_state_load(
        common_speculative_mtp_carry_lifecycle & lifecycle,
        std::vector<float> & pending_h,
        const std::vector<uint8_t> & data);

// fork 9l-d4b8h: the checkpoint policy and the batch preflight both live
// between the MTP carry codec and the sequence-transition owner
// (fork common/speculative.h:204-228).
struct common_speculative_checkpoint_policy {
    bool capture_draft = false;
    bool require_complete_draft_and_state = false;
};

common_speculative_checkpoint_policy common_speculative_checkpoint_policy_resolve(
        bool has_draft_context,
        bool vbr_prompt_cache,
        bool can_speculate,
        bool mtp_primary) noexcept;

enum class common_speculative_mtp_process_preflight {
    cold_or_retained,
    target_only,
};

// Resolve every active sequence before process() mutates draft memory. A
// single nonzero frontier without an authenticated carry makes the whole
// multi-sequence batch target-only so no sequence advances on partial state.
common_speculative_mtp_process_preflight
common_speculative_mtp_process_preflight_resolve(
    const std::vector<common_speculative_mtp_carry_lifecycle> & lifecycles,
    const std::vector<int32_t> & active_batch_beg,
    const llama_pos * positions) noexcept;

// One owner for external sequence lifecycle mutations. Implementations discard
// branch-local state and apply the event-specific memory/ring policy.
void common_speculative_sequence_transition(
        common_speculative * spec,
        llama_seq_id         seq_id,
        common_speculative_sequence_event event);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
