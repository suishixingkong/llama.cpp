#pragma once

#include "server-cache-lease.h"
#include "server-cache-destruction-quote.h"
#include "server-cache-yield.h"
#include "server-retention-sidecar.h"
#include "../../common/common-cache-plan.h"
#include "../../common/common-cache-family.h"
#include "../../src/llama-cache-authority.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <list>
#include <vector>

struct server_prompt_cache_state;
struct common_prompt_checkpoint;
struct server_cache_authority;

struct server_cache_live_checkpoint_admission {
    llama_cache_acct_artifact_id artifact;
    const common_prompt_checkpoint * checkpoint = nullptr;
    std::vector<llama_cache_acct_op_id> committed;
};

enum class server_cache_checkpoint_protection : uint8_t {
    none = 0,
    seam_heuristic,
    mandatory_anchor,
    hard_lease,
    _count,
};


// A later checkpoint may be omitted only when the retained predecessor is a
// same-lineage recovery point within the
// configured marginal replay bound.
bool server_cache_checkpoint_bounded_replay(
    const common_prompt_checkpoint & recovery,
    const common_prompt_checkpoint & later,
    uint64_t max_replay_tokens) noexcept;

// An exact recurrent-checkpoint restore may remove only the live attention suffix after the
// installed frontier. Checkpoints wholly before that suffix still cite identical attention rows;
// rebase those matching the pre-trim lineage so dedup/thinning do not copy a replacement image.
size_t server_cache_checkpoint_rebase_preserved_suffix(
    std::list<common_prompt_checkpoint> & checkpoints,
    const llama_memory_vbr_state_data & before,
    const llama_memory_vbr_state_data & after,
    llama_pos suffix_begin) noexcept;

struct server_cache_checkpoint_floor_input {
    uint32_t ordinal = 0;
    server_cache_checkpoint_protection protection =
        server_cache_checkpoint_protection::none;
    bool recovery_pinned = false;
};

struct server_cache_checkpoint_floor_plan {
    bool selected = false;
    uint32_t ordinal = UINT32_MAX;
    common_cache_plan_destruction_reason reason =
        common_cache_plan_destruction_reason::mandatory_anchor;
};

// Capacity's legacy-order floor. Heuristic members remain eligible when every
// unprotected member is gone; hard/mandatory/current-task/pinned members never
// are. No selection means the incoming checkpoint publication must be skipped.
server_cache_checkpoint_floor_plan server_cache_plan_checkpoint_capacity_floor(
    const std::vector<server_cache_checkpoint_floor_input> & candidates) noexcept;

// A protected/fail-closed ring has no new evidence until its membership
// changes. Keep publication checks and capacity selection independently
// latched so an unchanged ring pays one integer comparison per pass rather than
// rebuilding identities and leases for every attempted publication.
// A committed member erase or publication advances the generation and re-arms
// every lane.
enum class server_cache_checkpoint_attempt_lane : uint8_t {
    publication_check = 0,
    capacity_floor,
    _count,
};

class server_cache_checkpoint_attempt_latch {
public:
    bool begin(server_cache_checkpoint_attempt_lane lane) noexcept {
        const size_t index = size_t(lane);
        if (index >= attempted_generation_.size() ||
            attempted_generation_[index] == generation_) {
            return false;
        }
        attempted_generation_[index] = generation_;
        return true;
    }

    bool refusal_changed(
            common_cache_plan_destruction_reason reason,
            bool publication_skip = false) noexcept {
        const size_t lane = publication_skip ? 1 : 0;
        if (refusal_generation_[lane] == generation_ &&
            refusal_reason_[lane] == reason) {
            return false;
        }
        refusal_generation_[lane] = generation_;
        refusal_reason_[lane] = reason;
        return true;
    }

    void ring_changed() noexcept {
        generation_++;
        if (generation_ == 0) {
            generation_ = 1;
            attempted_generation_ = {};
            refusal_generation_ = {};
        }
    }

    uint64_t generation() const noexcept {
        return generation_;
    }

private:
    uint64_t generation_ = 1;
    std::array<uint64_t,
        size_t(server_cache_checkpoint_attempt_lane::_count)>
        attempted_generation_ = {};
    std::array<uint64_t, 2> refusal_generation_ = {};
    std::array<common_cache_plan_destruction_reason, 2> refusal_reason_ = {
        common_cache_plan_destruction_reason::none,
        common_cache_plan_destruction_reason::none,
    };
};

// Checkpoint ownership is prompt-cache authority work even though its
// physical list belongs to a live slot. The slot supplies this narrow view and
// retains the physical eraser; protected capacity selection and publication
// accounting live beside the prompt-cache authority orchestration.
struct server_cache_checkpoint_authority_context {
    using checkpoint_list = std::list<common_prompt_checkpoint>;
    using checkpoint_iterator = checkpoint_list::iterator;

    int32_t slot_id = -1;
    checkpoint_list & checkpoints;
    server_cache_authority * authority = nullptr;
    server_retention_sidecar_store * retention = nullptr;
    server_cache_destruction_observer * destruction = nullptr;
    server_cache_lease_table * leases = nullptr;
    server_cache_checkpoint_attempt_latch & attempts;
    const common_prompt_checkpoint *& seam_heuristic;
    common_cache_plan_destruction_reason & floor_refusal;
    bool debug_observability = false;
};

void server_cache_checkpoint_ring_changed(
    server_cache_checkpoint_authority_context & context) noexcept;

bool server_cache_checkpoint_publication_attempt_begin(
    server_cache_checkpoint_authority_context & context) noexcept;

bool server_cache_checkpoint_refusal_state_changed(
    server_cache_checkpoint_authority_context & context,
    common_cache_plan_destruction_reason reason,
    bool publication_skip = false) noexcept;

server_cache_destruction_admission server_cache_checkpoint_observe_drop(
    const server_cache_checkpoint_authority_context & context,
    server_cache_destruction_reason reason,
    llama_cache_acct_artifact_id artifact = {}) noexcept;

// Called only inside the optional publication-attempt latch. This retains
// the existing bounded incoming suppression without deleting an incumbent.
bool server_cache_checkpoint_publication_redundant(
    const std::list<common_prompt_checkpoint> & checkpoints,
    const common_prompt_checkpoint & incoming,
    int checkpoint_task_id,
    uint64_t max_replay_tokens) noexcept;

bool server_cache_checkpoint_capacity_floor(
    server_cache_checkpoint_authority_context & context,
    int checkpoint_task_id,
    const common_prompt_checkpoint * seam_heuristic,
    server_cache_checkpoint_authority_context::checkpoint_iterator & victim,
    common_cache_plan_destruction_reason & refusal) noexcept;

void server_cache_checkpoint_publication_skipped(
    server_cache_checkpoint_authority_context & context,
    common_cache_plan_destruction_reason reason) noexcept;

struct server_cache_host_recovery_evidence {
    llama_cache_acct_artifact_id artifact;
    std::vector<llama_cache_acct_op_id> ops;
    server_cache_recovery_pin pin;
    common_cache_plan_displaced_fate fate =
        common_cache_plan_displaced_fate::unavailable;
};

using server_cache_host_recovery_fn = bool (*)(
    void * context,
    const server_prompt_cache_state & victim,
    server_cache_host_recovery_evidence & out) noexcept;

// Prompt-cache authority substrate. The debug observer is only a serialization layer over this
// independently-owned state; --cache-lifecycle can therefore enforce accounting with debug off.
// Member order is lifetime order: retention releases lease memberships and accounting operations,
// so the ledger and leases must outlive it.
struct server_cache_authority {
    struct device_binding {
        ggml_backend_dev_t               device = nullptr;
        llama_cache_acct_resource_domain domain;
    };

    llama_cache_acct_ledger ledger;
    server_cache_lease_table leases;
    server_retention_sidecar_store retention;
    server_cache_destruction_observer destruction;
    llama_cache_budget_coordinator budget;
    server_cache_yield_result last_yield;
    common_cache_plan_destruction_counters destruction_counters;

    // Immutable bridge from load-time placement to ledger-local device domains.
    std::vector<device_binding> live_device_domains;
    // Fixed-at-reserve-time compute rows. Physical capacity is sampled at observation/admission.
    std::vector<llama_cache_budget_device_input> budget_devices;
    llama_cache_budget_config budget_config;

    uint64_t admission_retries   = 0;
    uint64_t admission_refusals  = 0;
    uint64_t admission_commits   = 0;
    uint64_t admission_rollbacks = 0;
    uint64_t destruction_quote_sequence = 0;
    void * host_recovery_context = nullptr;
    server_cache_host_recovery_fn host_recovery = nullptr;
    bool configured = true;
    bool summary_emitted = false;

    // Construct a point-in-time budget input. pending_host_bytes are already allocated in the
    // detached host-cache node, so they are added back to the sampled CPU free-memory headroom.
    bool sample_budget(
            llama_cache_budget_config & config,
            uint64_t pending_host_bytes = 0) noexcept;

    // Lower one exact accounting union into the capacity domains that its
    // release would affect. Every destruction class shares this projection door.
    bool project_release(
            const llama_cache_acct_release_set_preview & release,
            std::vector<common_cache_plan_yield_domain> & out) noexcept;

    // Re-sample the affected accounting domains after a committed release.
    // Actual yield is derived from this post-mutation observation, never
    // relabeled from the quote-time projection.
    bool observe_release_domains(
            const std::vector<common_cache_plan_yield_domain> & projected,
            std::vector<common_cache_plan_yield_domain> & out) noexcept;

    // The cache plan's first authoritative producer: admit, stage, and commit all host-entry payload leaves as one
    // all-or-nothing server transaction. Publication itself remains the caller's no-throw splice.
    bool admit_host_entry(server_prompt_cache_state & entry) noexcept;

    // Charge independently owned live-checkpoint payloads. The caller
    // publishes sidecar identities first, then attaches these exact operations
    // in the same scheduler turn before any planner can observe the members.
    // The batch is one all-or-nothing reservation transaction; host-entry
    // checkpoint copies never call this door.
    bool admit_live_checkpoints(
        std::vector<server_cache_live_checkpoint_admission> & batch) noexcept;

    // Single-member creation adapter. Restore paths must use the batch door so
    // one ring incurs one budget sample and one reservation transaction.
    bool admit_live_checkpoint(
        llama_cache_acct_artifact_id artifact,
        const common_prompt_checkpoint & checkpoint,
        std::vector<llama_cache_acct_op_id> & committed) noexcept;

    // Bounded process-local receipt publication for destruction work that
    // occurs during host-cache maintenance rather than one B request record.
    void observe_host_destruction(
        common_cache_plan_destruction_receipt receipt,
        bool observe_classification = true) noexcept;
};
