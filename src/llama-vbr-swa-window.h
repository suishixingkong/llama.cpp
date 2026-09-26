#pragma once

#include "llama-vbr-explicit-capture.h"

#include <array>
#include <memory>
#include <vector>

struct llama_context;
class llama_kv_cache_iswa;

// Process-local partial window, deliberately not an artifact package or a
// payload_complete child. No serialized format and no live tensor pointers.
struct vbr_swa_window_row {
    uint32_t physical_cell;
    llama_pos position;
    llama_token token;
};

struct vbr_swa_window_unit {
    uint32_t logical_unit;
    uint32_t model_layer;
    vbr_unit_generation generation;
    vbr_explicit_representation_identity codec;
    int32_t meansub_model_id;
    int32_t meansub_layer;
    std::array<char, GGML_MAX_NAME> tensor_name {};
    uint64_t columns;
    size_t row_bytes;
    std::vector<uint8_t> payload;
};

struct vbr_swa_window_capture_request {
    llama_seq_id sequence = -1;
    llama_pos frontier = -1; // exclusive; all required prefix rows must still be committed and retained
    uint64_t sequence_epoch = 0; // caller's logical slot lifetime, not slot ID
    std::array<uint8_t, 32> execution_identity {};
    vbr_explicit_representation_policy representation;

    // Required admission adapter. Charge the quoted bytes once and return an
    // owning lease that releases the charge at destruction. The image retains
    // it across all shared readers. A null lease declines BEFORE payload reads.
    // Server integration must bind this to its host-cache budget, not a second
    // unaccounted allowance.
    void * capacity_context = nullptr;
    std::shared_ptr<void> (*reserve)(void *, size_t bytes) = nullptr;
    void * continue_context = nullptr;
    bool (*continue_capture)(void *) noexcept = nullptr;
};

enum class vbr_swa_window_status {
    ok,
    unsupported,
    unavailable,
    protected_rows,
    capacity_refused,
    cancelled,
    source_changed,
    allocation_failed,
    destination_unavailable,
    representation_mismatch,
    insufficient_cells,
    backing_unavailable,
    operation_refused,
    rolled_back,
    staging_unavailable,
};

class vbr_swa_window_image {
public:
    vbr_swa_window_image(const vbr_swa_window_image &) = delete;
    vbr_swa_window_image & operator=(const vbr_swa_window_image &) = delete;
    const std::vector<vbr_swa_window_row> & rows() const { return rows_; }
    const std::vector<vbr_swa_window_unit> & units() const { return units_; }
    // Payload plus owned metadata; allocator/control-block overhead excluded,
    // as with other logical host-cache byte charges.
    size_t retained_bytes() const { return retained_bytes_; }
    llama_pos frontier() const { return frontier_; }

    // Source identity/coverage ONLY, not authorization to install. Deliberately
    // ignores the live SWA content epoch: recycling that window is the use case.
    // Caller must still match execution and logical sequence lifetime, and the
    // destination transaction must validate its current representation/capacity.
    bool source_matches(llama_context & ctx, uint64_t sequence_epoch,
                        const std::array<uint8_t, 32> & execution_identity) const;

private:
    friend class vbr_swa_window_capture;
    friend class vbr_swa_window_planner;
    vbr_swa_window_image() = default;
    // Declared first so the charge outlives payload/metadata destruction.
    std::shared_ptr<void> capacity_;
    vbr_controller_instance_id base_instance_ {}, swa_instance_ {};
    uint64_t base_epoch_ = 0, sequence_epoch_ = 0;
    std::array<uint8_t, 32> execution_identity_ {};
    llama_seq_id sequence_ = -1;
    llama_pos frontier_ = -1;
    size_t retained_bytes_ = 0;
    std::vector<vbr_swa_window_row> rows_;
    std::vector<vbr_swa_window_unit> units_;
};

struct vbr_swa_window_capture_result {
    vbr_swa_window_status status = vbr_swa_window_status::unavailable;
    std::shared_ptr<const vbr_swa_window_image> image;
};

// Static topology qualification, without synchronization, transfer or allocation.
bool vbr_swa_window_supported(llama_context & ctx);

// Requires the caller's exclusive scheduler boundary for this context, as do
// other llama memory operations. Synchronizes submitted compute. Not a lock
// allowing concurrent decode; callbacks must not reenter the context.
vbr_swa_window_capture_result vbr_capture_swa_window(
    llama_context & ctx, const vbr_swa_window_capture_request & request);

struct vbr_swa_window_plan_request {
    uint64_t source_epoch = 0;
    llama_seq_id destination = -1;
    uint64_t destination_epoch = 0;
    std::array<uint8_t, 32> execution_identity {};
    vbr_explicit_representation_policy representation;
    void * capacity_context = nullptr;
    std::shared_ptr<void> (*reserve)(void *, size_t bytes) = nullptr;
};

struct vbr_swa_window_membership_removal {
    uint32_t cell;
    llama_seq_id sequence;
    llama_pos position;
};

// Read-only placement proposal, NOT a reservation or authority to write.
// Owns the immutable image and the complete older-prefix removal list, including
// owners other than the source. No VMM mapping or operation/recovery capacity is
// reserved yet. The transactional installer must acquire those before writing.
class vbr_swa_window_plan {
public:
    ~vbr_swa_window_plan();
    vbr_swa_window_plan(const vbr_swa_window_plan &) = delete;
    vbr_swa_window_plan & operator=(const vbr_swa_window_plan &) = delete;
    const std::vector<uint32_t> & destination_cells() const;
    const std::vector<uint32_t> & base_cells() const;
    const std::vector<vbr_swa_window_membership_removal> & removals() const;
    uint32_t required_watermark() const;
    size_t retained_bytes() const;
    // Caller supplies current logical slot lifetimes. Requires the same exclusive,
    // synchronized scheduler boundary as prepare; does not synchronize or mutate.
    bool current(llama_context & ctx, uint64_t source_epoch, uint64_t destination_epoch,
                 const std::array<uint8_t, 32> & execution_identity) const;
private:
    friend class vbr_swa_window_planner;
    vbr_swa_window_plan();
    struct impl;
    std::unique_ptr<impl> impl_;
};

struct vbr_swa_window_plan_result {
    vbr_swa_window_status status = vbr_swa_window_status::unavailable;
    std::unique_ptr<vbr_swa_window_plan> plan;
};

// Equal-representation or adjacent-downward planning. No live state edits, transfers, mapping
// or implicit settlement. The required reservation callback must not reenter the
// context. Unsettled/busy contexts decline. A successful plan is not proof that
// the separately reserved install workspace/backing is available.
vbr_swa_window_plan_result vbr_prepare_swa_window(
    llama_context & ctx, std::shared_ptr<const vbr_swa_window_image> image,
    const vbr_swa_window_plan_request & request);

struct vbr_swa_window_install_request {
    uint64_t source_epoch = 0, destination_epoch = 0;
    std::array<uint8_t, 32> execution_identity {};
    // Boundary cancellation only; must not reenter or mutate the context.
    void * continue_context = nullptr;
    bool (*continue_install)(void *) noexcept = nullptr;
    void * capacity_context = nullptr;
    std::shared_ptr<void> (*reserve)(void *, size_t bytes) = nullptr;
};

// Consumes the proposal on ALL outcomes. Equal-tier copies and adjacent downward
// conversion use the existing codec implementations. This implementation uses
// already mapped backing only; missing backing declines without mapping/growing
// the live pool. Cancellation after a write restores saved bytes before return
// (rolled_back); backend-fatal transfer errors retain their fatal semantics.
// Caller owns exclusive synchronized context access for the entire call.
vbr_swa_window_status vbr_install_swa_window(llama_context & ctx,
    std::unique_ptr<vbr_swa_window_plan> plan, const vbr_swa_window_install_request & request);
