#include "llama-vbr-swa-window.h"

#include "llama-hparams.h"
#include "llama-kv-cache-iswa.h"
#include "llama-vbr-downward.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <type_traits>

namespace {
bool nonzero(const std::array<uint8_t, 32> & value) {
    return std::any_of(value.begin(), value.end(), [](uint8_t b) { return b != 0; });
}

bool add_bytes(size_t & total, size_t count, size_t stride) {
    if (stride && count > (SIZE_MAX-total)/stride) { return false; }
    total += count*stride;
    return true;
}
}

class vbr_swa_window_capture {
    using status = vbr_swa_window_status;
    static bool ready(const llama_kv_cache & cache) {
        const auto * tracker = cache.vbr_generation_tracker_get();
        return cache.vbr_operation_armed() && tracker && tracker->stable() &&
            !tracker->shadow_unavailable();
    }

public:
    static bool supported(llama_context & ctx) {
        auto * tree = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(&ctx));
        if (!tree) { return false; }
        const auto & base = *tree->get_base();
        const auto & swa = *tree->get_swa();
        return !base.other && !swa.other && base.n_stream == 1 && swa.n_stream == 1 &&
            !swa.v_trans && swa.swa_type == LLAMA_SWA_TYPE_STANDARD && swa.n_swa > 0 &&
            base.n_swa == 0 && base.vbr_pools_.size() == 1 && swa.vbr_pools_.size() == 1 &&
            base.vbr_pools_[0].device >= 0 && base.vbr_pools_[0].device == swa.vbr_pools_[0].device &&
            base.vbr_vmm_active() && swa.vbr_vmm_active();
    }

    static bool matches(const vbr_swa_window_image & image, llama_context & ctx,
                        uint64_t epoch, const std::array<uint8_t, 32> & execution) {
        auto * tree = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(&ctx));
        if (!tree || epoch != image.sequence_epoch_ || execution != image.execution_identity_) { return false; }
        const auto & base = *tree->get_base();
        const auto & swa = *tree->get_swa();
        const vbr_controller_instance_id instances[] = {base.vbr_instance_id(), swa.vbr_instance_id()};
        return ready(base) && ready(swa) && base.vbr_instance_id() == image.base_instance_ &&
            swa.vbr_instance_id() == image.swa_instance_ &&
            vbr_operation_registry_quiescent_for(instances, 2) &&
            base.vbr_checkpoint_epoch(image.sequence_) == image.base_epoch_ &&
            base.v_cells.size() == 1 && base.v_cells[0].seq_has_prefix(image.sequence_, image.frontier_);
    }

    static vbr_swa_window_capture_result capture(
            llama_context & ctx, const vbr_swa_window_capture_request & request) {
        const auto fail = [](status s) { return vbr_swa_window_capture_result {s, nullptr}; };
        auto * tree = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(&ctx));
        if (!tree || request.sequence < 0 || request.sequence >= LLAMA_MAX_SEQ ||
            request.frontier <= 0 || request.sequence_epoch == 0 || !nonzero(request.execution_identity) ||
            !request.representation.build_identity || !request.representation.build_identity_len) {
            return fail(status::unsupported);
        }
        auto & base = *tree->get_base();
        auto & swa = *tree->get_swa();
        if (!supported(ctx)) { return fail(status::unsupported); }
        if (!request.reserve) { return fail(status::capacity_refused); }
        llama_synchronize(&ctx);
        const vbr_controller_instance_id instances[] = {base.vbr_instance_id(), swa.vbr_instance_id()};
        if (!ready(base) || !ready(swa) || !vbr_operation_registry_quiescent_for(instances, 2) ||
            !swa.vbr_capture_settle()) { return fail(status::unavailable); }
        const auto & cells = swa.v_cells[0];
        const llama_pos begin = std::max<int64_t>(0, int64_t(request.frontier)-swa.n_swa);
        // Attention-only KV rows are independently reusable after later tokens
        // have been evaluated. Capture an earlier prefix only while its ENTIRE
        // window is retained; never infer coverage from the sequence high-water mark.
        if (cells.seq_pos_max(request.sequence) < request.frontier-1 ||
            !cells.seq_has_range(request.sequence, begin, request.frontier) ||
            !base.v_cells[0].seq_has_prefix(request.sequence, request.frontier)) {
            return fail(status::unavailable);
        }
        const uint64_t base_serial = base.vbr_generation_tracker_get()->mutation_serial();
        const uint64_t swa_serial = swa.vbr_generation_tracker_get()->mutation_serial();
        const uint64_t base_repr = base.vbr_representation_epoch();
        const uint64_t swa_repr = swa.vbr_representation_epoch();
        auto image = std::shared_ptr<vbr_swa_window_image>(new vbr_swa_window_image);
        image->base_instance_ = instances[0]; image->swa_instance_ = instances[1];
        image->base_epoch_ = base.vbr_checkpoint_epoch(request.sequence);
        image->sequence_epoch_ = request.sequence_epoch;
        image->execution_identity_ = request.execution_identity;
        image->sequence_ = request.sequence; image->frontier_ = request.frontier;
        const size_t row_count = std::min<uint32_t>(swa.n_swa, request.frontier);
        image->rows_.reserve(row_count);
        for (uint32_t r = 0; r < cells.size(); ++r) {
            if (!cells.seq_has(r, request.sequence) || cells.pos_get(r) < begin ||
                cells.pos_get(r) >= request.frontier) { continue; }
            if (r < swa.vbr_stash_rows_) { return fail(status::protected_rows); }
            const auto & ext = cells.ext_get(r);
            if (ext.x != 0 || ext.y != 0 || ext.tok == LLAMA_TOKEN_NULL || image->rows_.size() == row_count) {
                return fail(status::unsupported);
            }
            image->rows_.push_back({r, cells.pos_get(r), ext.tok});
        }
        if (image->rows_.size() != row_count) { return fail(status::unavailable); }
        // Pack in logical order, keeping the physical map for actual transfers.
        std::sort(image->rows_.begin(), image->rows_.end(), [](const auto & a, const auto & b) {
            return a.position < b.position;
        });
        for (size_t i = 0; i < row_count; ++i) {
            if (image->rows_[i].position != request.frontier-llama_pos(row_count)+llama_pos(i)) {
                return fail(status::unsupported);
            }
        }
        const auto * tracker = swa.vbr_generation_tracker_get();
        if (tracker->unit_count() != 2*swa.layers.size()) { return fail(status::unsupported); }
        image->units_.reserve(tracker->unit_count());
        size_t bytes = sizeof(vbr_swa_window_image);
        if (!add_bytes(bytes, image->rows_.capacity(), sizeof(vbr_swa_window_row)) ||
            !add_bytes(bytes, image->units_.capacity(), sizeof(vbr_swa_window_unit))) { return fail(status::capacity_refused); }
        std::vector<ggml_tensor *> tensors;
        for (uint32_t id = 0; id < tracker->unit_count(); ++id) {
            const auto & layer = swa.layers[id/2];
            const auto extents = swa.vbr_units_of(id/2, (id&1) != 0);
            if (extents.size() != 1) { return fail(status::unsupported); }
            const auto [pool, extent] = extents.front();
            auto * t = extent->t;
            const auto generation = tracker->unit_generation(id);
            vbr_downward_recipe recipe;
            if (!t || t != ((id&1) ? layer.v : layer.k) || t->ne[2] != 1 || t->ne[3] != 1 ||
                vbr_downward_resolve_recipe(t->type, t->type, t->type, true, recipe) != vbr_downward_recipe_status::equal_tier ||
                t->nb[1] != ggml_row_size(t->type, t->ne[0]) ||
                vbr_explicit_capture_validate_extent_generation(pool->wm_cells, t->type, extent->promote_hops, generation) !=
                    vbr_explicit_size_failure::none) { return fail(status::unsupported); }
            for (const auto & row : image->rows_) {
                if (row.physical_cell >= pool->wm_cells) { return fail(status::unavailable); }
            }
            vbr_swa_window_unit unit {};
            unit.logical_unit = id; unit.model_layer = layer.il;
            unit.generation = generation;
            unit.meansub_model_id = layer.turbo_meansub_ref.model_id;
            unit.meansub_layer = layer.turbo_meansub_ref.layer;
            // Same capture-scoped key as the explicit-capture adapter. Reuse
            // identities already in this image; never cache an override across captures.
            const auto cached = std::find_if(image->units_.begin(), image->units_.end(),
                [&](const auto & prior) {
                    return prior.generation.current_type == t->type &&
                        (prior.logical_unit&1) == (id&1) && prior.meansub_model_id == unit.meansub_model_id;
                });
            if (cached != image->units_.end()) {
                unit.codec = cached->codec;
            } else if (!vbr_explicit_capture_representation_identity(&request.representation, t->type, (id&1) != 0,
                           unit.meansub_model_id, unit.codec)) { return fail(status::unavailable); }
            std::memcpy(unit.tensor_name.data(), t->name, unit.tensor_name.size());
            unit.columns = t->ne[0]; unit.row_bytes = t->nb[1];
            if (!add_bytes(bytes, row_count, unit.row_bytes)) { return fail(status::capacity_refused); }
            image->units_.push_back(std::move(unit));
            tensors.push_back(t);
        }
        image->capacity_ = request.reserve(request.capacity_context, bytes);
        if (!image->capacity_) { return fail(status::capacity_refused); }
        image->retained_bytes_ = bytes;
        const auto stable = [&] {
            return ready(base) && ready(swa) && base.vbr_generation_tracker_get()->mutation_serial() == base_serial &&
                swa.vbr_generation_tracker_get()->mutation_serial() == swa_serial &&
                base.vbr_representation_epoch() == base_repr && swa.vbr_representation_epoch() == swa_repr &&
                vbr_operation_registry_quiescent_for(instances, 2);
        };
        if (!stable()) { return fail(status::source_changed); }
        constexpr size_t chunk_bytes = 64*1024;
        for (size_t u = 0; u < image->units_.size(); ++u) {
            auto & unit = image->units_[u];
            unit.payload.resize(row_count*unit.row_bytes);
            for (size_t i = 0; i < row_count;) {
                uint32_t stride = 1;
                if (i+1 < row_count && unit.row_bytes &&
                    image->rows_[i+1].physical_cell > image->rows_[i].physical_cell) {
                    const auto delta = image->rows_[i+1].physical_cell-image->rows_[i].physical_cell;
                    if (delta <= chunk_bytes/unit.row_bytes) { stride = delta; }
                }
                size_t end = i+1;
                while (end < row_count &&
                    image->rows_[end].physical_cell == uint64_t(image->rows_[end-1].physical_cell)+stride) { ++end; }
                if (stride > 1) {
                    // Interleaved slots often yield a constant physical stride.
                    // Bound source span AND payload by the cancellation quantum;
                    // a wrap, stride change or large gap ends this 2D gather.
                    const size_t pitch = stride*unit.row_bytes;
                    const size_t rows_per_copy = 1+(chunk_bytes-unit.row_bytes)/pitch;
                    while (i < end) {
                        if (request.continue_capture && !request.continue_capture(request.continue_context)) { return fail(status::cancelled); }
                        if (!stable()) { return fail(status::source_changed); }
                        const size_t count = std::min(end-i, rows_per_copy);
                        ggml_backend_tensor_get_2d(tensors[u], unit.payload.data()+i*unit.row_bytes,
                            image->rows_[i].physical_cell*unit.row_bytes, unit.row_bytes, count, pitch, unit.row_bytes);
                        i += count;
                    }
                    continue;
                }
                const size_t run_bytes = (end-i)*unit.row_bytes;
                for (size_t offset = 0; offset < run_bytes;) {
                    if (request.continue_capture && !request.continue_capture(request.continue_context)) { return fail(status::cancelled); }
                    if (!stable()) { return fail(status::source_changed); }
                    const size_t size = std::min(chunk_bytes, run_bytes-offset);
                    auto * dst = unit.payload.data()+i*unit.row_bytes+offset;
                    ggml_backend_tensor_get(tensors[u], dst, image->rows_[i].physical_cell*unit.row_bytes+offset, size);
                    offset += size;
                }
                i = end;
            }
        }
        if (!stable()) { return fail(status::source_changed); }
        if (request.continue_capture && !request.continue_capture(request.continue_context)) { return fail(status::cancelled); }
        return {status::ok, std::move(image)};
    }
};

bool vbr_swa_window_image::source_matches(llama_context & ctx, uint64_t epoch,
                                       const std::array<uint8_t, 32> & execution) const {
    return vbr_swa_window_capture::matches(*this, ctx, epoch, execution);
}

bool vbr_swa_window_supported(llama_context & ctx) {
    return vbr_swa_window_capture::supported(ctx);
}

vbr_swa_window_capture_result vbr_capture_swa_window(
        llama_context & ctx, const vbr_swa_window_capture_request & request) {
    try {
        return vbr_swa_window_capture::capture(ctx, request);
    } catch (const std::bad_alloc &) {
        return {vbr_swa_window_status::allocation_failed, nullptr};
    }
}

struct vbr_swa_window_plan::impl {
    std::shared_ptr<void> capacity;
    size_t retained_bytes = 0;
    struct boundary {
        uint64_t controller, mutation, representation, destination_content;
        uint32_t head, watermark;
    };
    std::shared_ptr<const vbr_swa_window_image> image;
    llama_seq_id destination = -1;
    uint64_t destination_epoch = 0;
    std::string build_identity;
    std::array<boundary, 2> children {};
    std::vector<uint32_t> destination_cells, base_cells;
    std::vector<vbr_swa_window_membership_removal> removals;
    std::vector<vbr_downward_recipe> recipes;
    uint32_t required_watermark = 0;
};

class vbr_swa_window_planner {
    using status = vbr_swa_window_status;
    using state = vbr_swa_window_plan::impl;

    static state::boundary boundary(const llama_kv_cache & cache, llama_seq_id destination) {
        const auto * t = cache.vbr_generation_tracker_get();
        return {t->controller_generation(), t->mutation_serial(), cache.vbr_representation_epoch(),
            cache.vbr_checkpoint_epoch(destination), cache.v_heads[0], cache.vbr_pools_[0].wm_cells};
    }

    static bool same(const state::boundary & a, const state::boundary & b) {
        return a.controller == b.controller && a.mutation == b.mutation &&
            a.representation == b.representation && a.destination_content == b.destination_content &&
            a.head == b.head && a.watermark == b.watermark;
    }

    static bool same_codec(const vbr_explicit_representation_identity & a,
                           const vbr_explicit_representation_identity & b) {
        return a.codec_id == b.codec_id && a.codec_version == b.codec_version &&
            a.codebook_digest == b.codebook_digest && a.rotation_digest == b.rotation_digest &&
            a.meansub_digest == b.meansub_digest && a.meansub_baked == b.meansub_baked;
    }

    static bool representation_matches(const state & plan, const llama_kv_cache & swa,
                                       std::vector<vbr_downward_recipe> * recipes = nullptr) {
        const auto * tracker = swa.vbr_generation_tracker_get();
        if (plan.image->units().size() != tracker->unit_count()) { return false; }
        const vbr_explicit_representation_policy policy {plan.build_identity.data(), plan.build_identity.size()};
        for (const auto & unit : plan.image->units()) {
            const auto & layer = swa.layers.at(unit.logical_unit/2);
            const auto extents = swa.vbr_units_of(unit.logical_unit/2, (unit.logical_unit&1) != 0);
            if (extents.size() != 1) { return false; }
            const auto [pool, extent] = extents.front();
            const auto * t = extent->t;
            const auto gen = tracker->unit_generation(unit.logical_unit);
            const auto & saved = unit.generation;
            if (!t || gen.flags != saved.flags ||
                layer.il != unit.model_layer || t->ne[0] != int64_t(unit.columns) ||
                t->ne[2] != 1 || t->ne[3] != 1 ||
                ggml_row_size(t->type, t->ne[0]) != t->nb[1] ||
                std::strncmp(t->name, unit.tensor_name.data(), GGML_MAX_NAME) != 0 ||
                layer.turbo_meansub_ref.model_id != unit.meansub_model_id ||
                layer.turbo_meansub_ref.layer != unit.meansub_layer ||
                vbr_explicit_capture_validate_extent_generation(pool->wm_cells, t->type, extent->promote_hops, gen) !=
                    vbr_explicit_size_failure::none) { return false; }
            vbr_downward_recipe recipe;
            const auto resolved = vbr_downward_resolve_recipe(ggml_type(saved.current_type), t->type, t->type, true, recipe);
            if (resolved == vbr_downward_recipe_status::equal_tier) {
                if (gen.domain != saved.domain || gen.last_source_type != saved.last_source_type ||
                    gen.promote_hops != saved.promote_hops || gen.last_transition != saved.last_transition) { return false; }
            } else if (resolved == vbr_downward_recipe_status::resolved) {
                // No live unit-wide history adoption or promotion reconstruction.
                // The last adjacent edge must match the live destination's loss history.
                if (saved.promote_hops || gen.promote_hops ||
                    saved.domain != recipe.edges[0].source_domain ||
                    gen.domain != recipe.edges[recipe.n_edges-1].target_domain ||
                    gen.last_source_type != recipe.edges[recipe.n_edges-1].source_type ||
                    (gen.last_transition != vbr_repr_transition::degrade_other &&
                     gen.last_transition != vbr_repr_transition::degrade_f16_to_t8_admitted)) { return false; }
            } else { return false; }
            vbr_explicit_representation_identity codec;
            if (!vbr_explicit_capture_representation_identity(&policy, ggml_type(saved.current_type), (unit.logical_unit&1) != 0,
                    unit.meansub_model_id, codec) || !same_codec(codec, unit.codec)) { return false; }
            // Each destination/edge codec must exist in this executable too.
            for (size_t edge = 0; edge < recipe.n_edges; ++edge) {
                if (!vbr_explicit_capture_representation_identity(&policy, recipe.edges[edge].target_type,
                        (unit.logical_unit&1) != 0, unit.meansub_model_id, codec)) { return false; }
            }
            if (recipes) { recipes->push_back(recipe); }
            else if (unit.logical_unit >= plan.recipes.size() || !(plan.recipes[unit.logical_unit] == recipe)) { return false; }
        }
        return true;
    }

public:
    static status install(llama_context & ctx, vbr_swa_window_plan & proposal,
                          const vbr_swa_window_install_request & request) {
        auto & plan = *proposal.impl_;
        if (!current(plan, ctx, request.source_epoch, request.destination_epoch, request.execution_identity)) {
            return status::source_changed;
        }
        auto & tree = *dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(&ctx));
        auto & base = *tree.get_base();
        auto & swa = *tree.get_swa();
        if (plan.required_watermark > swa.vbr_pools_[0].wm_cells) { return status::backing_unavailable; }
        if (!base.vbr_ownership_ || !swa.vbr_ownership_ || base.vbr_representation_epoch_ == UINT64_MAX ||
            swa.vbr_representation_epoch_ == UINT64_MAX || base.vbr_checkpoint_epoch_ == UINT64_MAX ||
            swa.vbr_checkpoint_epoch_ == UINT64_MAX) { return status::unavailable; }
        const auto proceed = [&] { return !request.continue_install || request.continue_install(request.continue_context); };
        if (!proceed()) { return status::cancelled; }

        // Reserve the transaction's logical host footprint before cloning cells,
        // provenance, ownership or payload. GPU staging is separately recoverable
        // and temporary. The caller uses the same budget as the sealed image.
        size_t bytes = sizeof(vbr_cell_update_event)*3;
        for (const auto * cache : {&base, &swa}) {
            if (!add_bytes(bytes, 1, cache->v_cells[0].copy_storage_bytes()) ||
                !add_bytes(bytes, 1, cache->vbr_ownership_->clone_storage_bytes(0, plan.destination)) ||
                !add_bytes(bytes, 1, cache->vbr_generation_tracker_get()->cell_update_storage_bytes())) { return status::capacity_refused; }
        }
        if (!add_bytes(bytes, plan.base_cells.size()+plan.destination_cells.size()+plan.removals.size(), sizeof(vbr_cell_update_stamp)) ||
            !add_bytes(bytes, plan.base_cells.size()+plan.destination_cells.size(), sizeof(std::pair<llama_pos, uint32_t>)) ||
            !add_bytes(bytes, plan.image->units().size(), 2*sizeof(void *)+2*sizeof(std::vector<uint8_t>))) { return status::capacity_refused; }
        for (const auto & unit : plan.image->units()) {
            const auto & layer = swa.layers[unit.logical_unit/2];
            const auto * t = (unit.logical_unit&1) ? layer.v : layer.k;
            const size_t copies = plan.recipes[unit.logical_unit].n_edges ? 2 : 1;
            if (!add_bytes(bytes, copies*plan.destination_cells.size(), t->nb[1])) { return status::capacity_refused; }
        }
        auto capacity = request.reserve ? request.reserve(request.capacity_context, bytes) : nullptr;
        if (!capacity) { return status::capacity_refused; }

        // All allocating membership/index edits happen on copies. Preserve even
        // unavailable ownership views; rebuilding them would change other slots.
        auto base_cells = base.v_cells[0];
        auto swa_cells = swa.v_cells[0];
        auto base_owners = base.vbr_ownership_->clone();
        auto swa_owners = swa.vbr_ownership_->clone();
        std::array<bool, LLAMA_MAX_SEQ> swa_changed {};
        std::array<llama_pos, LLAMA_MAX_SEQ> purge_to;
        purge_to.fill(-1);
        vbr_cell_update_event share {vbr_mutation_registrant::seq_cp, vbr_operation_class::prompt_share, {}};
        vbr_cell_update_event remove {vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api, {}};
        vbr_cell_update_event import {vbr_mutation_registrant::window_install, vbr_operation_class::checkpoint_restore, {}};
        share.stamps.reserve(plan.base_cells.size());
        remove.stamps.reserve(plan.removals.size());
        import.stamps.reserve(plan.destination_cells.size());
        for (auto cell : plan.base_cells) {
            const auto pos = base_cells.pos_get(cell);
            base_cells.seq_add(cell, plan.destination);
            if (!base_owners->add_cell(0, plan.destination, cell, pos)) { return status::unavailable; }
            share.stamps.push_back({cell, plan.destination, pos});
        }
        for (const auto & edit : plan.removals) {
            swa_cells.seq_rm(edit.cell, edit.sequence);
            // An already unavailable source view remains unavailable. Its
            // retained rows still belong to that owner in the actual cell map.
            swa_owners->remove_cell(0, edit.sequence, edit.cell, edit.position);
            swa_changed[edit.sequence] = true;
            purge_to[edit.sequence] = std::max(purge_to[edit.sequence], edit.position);
            remove.stamps.push_back({edit.cell, edit.sequence, edit.position});
        }
        for (size_t i = 0; i < plan.destination_cells.size(); ++i) {
            const auto cell = plan.destination_cells[i];
            const auto & row = plan.image->rows()[i];
            GGML_ASSERT(swa_cells.is_empty(cell));
            swa_cells.pos_set(cell, row.position);
            llama_kv_cell_ext ext; ext.tok = row.token;
            swa_cells.ext_set(cell, ext);
            swa_cells.seq_add(cell, plan.destination);
            if (!swa_owners->add_cell(0, plan.destination, cell, row.position)) { return status::unavailable; }
            import.stamps.push_back({cell, plan.destination, row.position});
        }
        swa_changed[plan.destination] = true;
        for (llama_seq_id seq = 0; seq < LLAMA_MAX_SEQ; ++seq) {
            if (swa_changed[seq] && swa_cells.seq_pos_min(seq) < 0) { swa_owners->clear_seq(0, seq); }
        }
        std::vector<vbr_cell_update_event> base_events;
        base_events.reserve(1);
        base_events.push_back(std::move(share));
        std::vector<vbr_cell_update_event> swa_events;
        swa_events.reserve(2);
        // Empty placement needs no removal target or authenticated remove event.
        if (!remove.stamps.empty()) { swa_events.push_back(std::move(remove)); }
        swa_events.push_back(std::move(import));
        vbr_operation_binding binding;
        binding.kind = vbr_operation_kind::window_restore;
        binding.child_phase = vbr_operation_phase::mutate;
        const auto target = [&](vbr_controller_instance_id instance, vbr_operation_class cls,
                                llama_seq_id seq, llama_pos p0, llama_pos p1) {
            return vbr_binding_add_instance_target(binding, binding.kind, cls, instance, 0, seq, p0, p1);
        };
        if (!target(base.vbr_instance_id(), vbr_operation_class::prompt_share, plan.destination, 0, plan.image->frontier_) ||
            !target(swa.vbr_instance_id(), vbr_operation_class::checkpoint_restore, plan.destination,
                    plan.image->rows().front().position, plan.image->frontier_)) { return status::operation_refused; }
        for (llama_seq_id seq = 0; seq < LLAMA_MAX_SEQ; ++seq) {
            if (purge_to[seq] >= 0 && !target(swa.vbr_instance_id(), vbr_operation_class::state_api,
                    seq, 0, purge_to[seq]+1)) { return status::operation_refused; }
        }
        vbr_scoped_operation operation(binding);
        if (!operation) { return status::operation_refused; }
        struct recovery_owner {
            vbr_operation_id operation;
            std::array<int32_t, 2> slots {-1, -1};
            ~recovery_owner() {
                for (auto slot : slots) { if (slot >= 0) { GGML_ASSERT(vbr_recovery_release_unused(slot, operation)); } }
            }
        } recovery {operation.id()};
        recovery.slots[0] = vbr_recovery_reserve(operation.id(), base.vbr_instance_id());
        recovery.slots[1] = vbr_recovery_reserve(operation.id(), swa.vbr_instance_id());
        if (recovery.slots[0] < 0 || recovery.slots[1] < 0) { return status::operation_refused; }
        auto * base_tracker = base.vbr_generation_tracker_mut();
        auto * swa_tracker = swa.vbr_generation_tracker_mut();
        vbr_tracker_cell_update base_update, swa_update;
        if (!base_tracker->prepare_cell_update(base_events, operation.id(), base_update) ||
            !swa_tracker->prepare_cell_update(swa_events, operation.id(), swa_update)) { return status::operation_refused; }

        struct transfer_unit {
            ggml_tensor * tensor;
            const vbr_swa_window_unit * saved;
            std::vector<uint8_t> backup, converted;
        };
        std::vector<transfer_unit> units;
        units.reserve(plan.image->units().size());
        size_t staging_bytes = 0;
        uint64_t staging_columns = 0;
        for (const auto & saved : plan.image->units()) {
            const auto & layer = swa.layers[saved.logical_unit/2];
            auto * tensor = (saved.logical_unit&1) ? layer.v : layer.k;
            const size_t bytes = plan.image->rows().size()*tensor->nb[1];
            const bool convert = plan.recipes[saved.logical_unit].n_edges != 0;
            units.push_back({tensor, &saved, std::vector<uint8_t>(bytes),
                             std::vector<uint8_t>(convert ? bytes : 0)});
            if (convert) {
                staging_bytes = std::max(staging_bytes, saved.payload.size());
                staging_columns = std::max(staging_columns, saved.columns);
            }
        }
        if (staging_bytes) {
            const auto & pool = swa.vbr_pools_[0];
            const auto * be = pool.be;
            if (!be || !be->kv_transcode || !be->kv_transcode_workspace_reserve) { return status::staging_unavailable; }
            // Temporary backend/workspace: optional conversion leaves no grow-only
            // workspace charge on the live controller after a miss or completion.
            ggml_backend_ptr backend(be->backend_init(pool.device));
            if (!backend) { return status::staging_unavailable; }
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_buffer(backend.get(), staging_bytes));
            if (!buffer || !be->kv_transcode_workspace_reserve(backend.get(), plan.image->rows().size(), staging_columns, 0)) {
                return status::staging_unavailable;
            }
            for (auto & unit : units) {
                if (unit.converted.empty()) { continue; }
                if (!proceed()) { return status::cancelled; }
                const auto & recipe = plan.recipes[unit.saved->logical_unit];
                ggml_tensor scratch = *unit.tensor;
                scratch.type = recipe.source_type;
                scratch.ne[1] = plan.image->rows().size();
                scratch.nb[0] = ggml_type_size(scratch.type);
                scratch.nb[1] = unit.saved->row_bytes;
                scratch.nb[2] = scratch.nb[3] = scratch.nb[1]*scratch.ne[1];
                scratch.data = ggml_backend_buffer_get_base(buffer.get());
                scratch.buffer = buffer.get(); scratch.view_src = nullptr; scratch.view_offs = 0;
                ggml_backend_tensor_set(&scratch, unit.saved->payload.data(), 0, unit.saved->payload.size());
                uint8_t fence;
                ggml_backend_tensor_get(&scratch, &fence, 0, 1);
                for (size_t edge = 0; edge < recipe.n_edges; ++edge) {
                    if (!proceed()) { return status::cancelled; }
                    // Packed row zero is NOT a sink. Only unprotected source and
                    // destination rows are admitted; never inject a synthetic stash.
                    const ggml_vbr_transcode_params transcode {&scratch, recipe.edges[edge].target_type,
                        scratch.data, buffer.get(), scratch.ne[1], bool(unit.saved->logical_unit&1), nullptr, 0, 0};
                    be->kv_transcode(backend.get(), &transcode);
                    ggml_backend_synchronize(backend.get());
                    scratch.type = transcode.type_B;
                    scratch.nb[0] = ggml_type_size(scratch.type);
                    scratch.nb[1] = ggml_row_size(scratch.type, scratch.ne[0]);
                    scratch.nb[2] = scratch.nb[3] = scratch.nb[1]*scratch.ne[1];
                }
                ggml_backend_tensor_get(&scratch, unit.converted.data(), 0, unit.converted.size());
            }
        }
        // Read all rollback bytes before the first write. No allocations after
        // this point. Bounded transfers; no per-row GPU scratch or new mapping.
        const auto transfer = [&](transfer_unit & unit, bool upload, const uint8_t * bytes, bool cancellable) {
            const auto stride = unit.tensor->nb[1];
            for (size_t i = 0; i < plan.destination_cells.size();) {
                size_t end = i+1;
                while (end < plan.destination_cells.size() && plan.destination_cells[end] == plan.destination_cells[end-1]+1) { ++end; }
                const size_t run = (end-i)*stride;
                for (size_t offset = 0; offset < run;) {
                    if (cancellable && !proceed()) { return false; }
                    const auto count = std::min<size_t>(64*1024, run-offset);
                    const auto dst_offset = plan.destination_cells[i]*stride+offset;
                    if (upload) { ggml_backend_tensor_set(unit.tensor, bytes+i*stride+offset, dst_offset, count); }
                    else { ggml_backend_tensor_get(unit.tensor, unit.backup.data()+i*stride+offset, dst_offset, count); }
                    offset += count;
                }
                i = end;
            }
            if (upload) {
                // CUDA's tensor_get completes the same per-thread transfer
                // stream as tensor_set, unlike the compute backend fence.
                uint8_t fence;
                ggml_backend_tensor_get(unit.tensor, &fence, plan.destination_cells.back()*stride, 1);
            }
            return true;
        };
        for (auto & unit : units) { if (!transfer(unit, false, nullptr, true)) { return status::cancelled; } }
        if (!proceed()) { return status::cancelled; }
        bool copied = true;
        for (auto & unit : units) {
            const auto * bytes = unit.converted.empty() ? unit.saved->payload.data() : unit.converted.data();
            if (!transfer(unit, true, bytes, true)) { copied = false; break; }
        }
        if (!copied || !proceed() || !base_tracker->cell_update_installable(base_update, operation.id()) ||
            !swa_tracker->cell_update_installable(swa_update, operation.id())) {
            // Even the interrupted unit's earlier queued uploads precede these
            // restores on the same stream. Fence every restore before returning.
            for (auto & unit : units) { transfer(unit, true, unit.backup.data(), false); }
            return status::rolled_back;
        }
        static_assert(std::is_nothrow_swappable<llama_kv_cells>::value, "cell publication must not allocate");
        base_tracker->install_cell_update(base_update, operation.id());
        swa_tracker->install_cell_update(swa_update, operation.id());
        base.v_cells[0].swap(base_cells);
        swa.v_cells[0].swap(swa_cells);
        base.vbr_ownership_.swap(base_owners);
        swa.vbr_ownership_.swap(swa_owners);
        swa.v_heads[0] = plan.destination_cells.back()+1;
        base.vbr_attention_content_changed(plan.destination);
        swa.vbr_attention_content_changed(swa_changed);
        // Release recovery before the successful root close. A failed path
        // unwinds it before the root's failed close, after rollback completed.
        for (auto & slot : recovery.slots) {
            GGML_ASSERT(vbr_recovery_release_unused(slot, operation.id())); slot = -1;
        }
        GGML_ASSERT(operation.close(vbr_operation_outcome::committed));
        return status::ok;
    }

    static bool current(const state & plan, llama_context & ctx, uint64_t source_epoch,
                        uint64_t destination_epoch, const std::array<uint8_t, 32> & execution) {
        if (destination_epoch != plan.destination_epoch ||
            !plan.image->source_matches(ctx, source_epoch, execution)) { return false; }
        auto & tree = *dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(&ctx));
        const auto & base = *tree.get_base();
        const auto & swa = *tree.get_swa();
        return base.can_share_attn_prefix(plan.image->sequence_, plan.destination, plan.image->frontier_) &&
            swa.can_share_destination(plan.image->sequence_, plan.destination) &&
            !base.vbr_stash_dirty_ && !swa.vbr_stash_dirty_ &&
            same(plan.children[0], boundary(base, plan.destination)) &&
            same(plan.children[1], boundary(swa, plan.destination)) && representation_matches(plan, swa);
    }

    static vbr_swa_window_plan_result prepare(llama_context & ctx,
            std::shared_ptr<const vbr_swa_window_image> image, const vbr_swa_window_plan_request & request) {
        const auto fail = [](status s) { return vbr_swa_window_plan_result {s, nullptr}; };
        if (!image || request.destination_epoch == 0 ||
            !request.representation.build_identity || !request.representation.build_identity_len) {
            return fail(status::unsupported);
        }
        if (!image->source_matches(ctx, request.source_epoch, request.execution_identity)) {
            return fail(status::source_changed);
        }
        auto & tree = *dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(&ctx));
        const auto & base = *tree.get_base();
        const auto & swa = *tree.get_swa();
        if (!base.can_share_attn_prefix(image->sequence_, request.destination, image->frontier_) ||
            !swa.can_share_destination(image->sequence_, request.destination)) {
            return fail(status::destination_unavailable);
        }
        if (base.vbr_stash_dirty_ || swa.vbr_stash_dirty_) { return fail(status::unavailable); }
        size_t memberships = 0;
        for (uint32_t cell = 0; cell < swa.v_cells[0].size(); ++cell) {
            swa.v_cells[0].seq_for_each(cell, [&](llama_seq_id) { ++memberships; });
        }
        size_t bytes = sizeof(state)+sizeof(vbr_swa_window_plan);
        if (!add_bytes(bytes, request.representation.build_identity_len+1, 1) ||
            !add_bytes(bytes, image->units().size(), sizeof(vbr_downward_recipe)) ||
            !add_bytes(bytes, image->rows().size()+size_t(image->frontier()), sizeof(uint32_t)) ||
            !add_bytes(bytes, memberships, sizeof(vbr_swa_window_membership_removal))) { return fail(status::capacity_refused); }
        auto capacity = request.reserve ? request.reserve(request.capacity_context, bytes) : nullptr;
        if (!capacity) { return fail(status::capacity_refused); }
        auto result = std::unique_ptr<vbr_swa_window_plan>(new vbr_swa_window_plan);
        auto & plan = *result->impl_;
        plan.capacity = std::move(capacity);
        plan.retained_bytes = bytes;
        plan.image = std::move(image);
        plan.destination = request.destination;
        plan.destination_epoch = request.destination_epoch;
        plan.build_identity.assign(request.representation.build_identity, request.representation.build_identity_len);
        plan.children = {boundary(base, request.destination), boundary(swa, request.destination)};
        plan.recipes.reserve(plan.image->units().size());
        plan.removals.reserve(memberships);
        plan.base_cells.reserve(plan.image->frontier());
        if (!representation_matches(plan, swa, &plan.recipes)) { return fail(status::representation_mismatch); }

        const auto & cells = swa.v_cells[0];
        const uint32_t count = plan.image->rows().size();
        if (count > cells.size()) { return fail(status::insufficient_cells); }
        uint32_t head = swa.v_heads[0];
        if (uint64_t(head) > uint64_t(cells.get_used()) + 2*uint64_t(count)) { head = 0; }
        plan.destination_cells.reserve(count);
        for (uint32_t tested = 0; tested < cells.size() && plan.destination_cells.size() < count; ++tested) {
            const uint32_t cell = (uint64_t(head)+tested)%cells.size();
            if (cell >= swa.vbr_stash_rows_ && swa.can_reuse_cell(0, cell)) {
                plan.destination_cells.push_back(cell);
            }
        }
        if (plan.destination_cells.size() != count) { return fail(status::insufficient_cells); }

        std::array<llama_pos, LLAMA_MAX_SEQ> purge_to;
        purge_to.fill(-1);
        for (uint32_t cell : plan.destination_cells) {
            cells.seq_for_each(cell, [&](llama_seq_id seq) {
                purge_to[seq] = std::max(purge_to[seq], cells.pos_get(cell));
            });
        }
        // Like apply_ubatch, preserve each previous owner's contiguous suffix.
        // Enumerate ALL older-prefix edits, not merely the rows receiving bytes.
        uint32_t last_used = 0;
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            size_t remaining = 0;
            cells.seq_for_each(cell, [&](llama_seq_id seq) {
                const llama_pos pos = cells.pos_get(cell);
                if (pos <= purge_to[seq]) {
                    plan.removals.push_back({cell, seq, pos});
                } else {
                    ++remaining;
                }
            });
            // seq_rm marks a protected cell dirty when its last owner leaves.
            // Refuse even an INDIRECT stash eviction; skipping destination
            // rows alone would not prevent the later prefix purge doing this.
            if (!cells.is_empty(cell) && remaining == 0 && cell < swa.vbr_stash_rows_) {
                return fail(status::protected_rows);
            }
            if (remaining) { last_used = cell+1; }
        }
        for (uint32_t cell : plan.destination_cells) { last_used = std::max(last_used, cell+1); }
        const uint32_t pad = std::max(swa.n_pad, swa.get_pad_floor());
        plan.required_watermark = std::min<uint64_t>(cells.size(),
            ((uint64_t(last_used)+pad-1)/pad)*pad);
        // This is a required endpoint, NOT proof of mapping or byte capacity.
        // The installer must preflight actual backing at this endpoint.
        const auto & base_cells = base.v_cells[0];
        for (uint32_t cell = 0; cell < base_cells.size(); ++cell) {
            if (base_cells.pos_in(cell, 0, plan.image->frontier_) && base_cells.seq_has(cell, plan.image->sequence_)) {
                plan.base_cells.push_back(cell);
            }
        }
        if (!current(plan, ctx, request.source_epoch, request.destination_epoch, request.execution_identity)) {
            return fail(status::source_changed);
        }
        return {status::ok, std::move(result)};
    }
};

vbr_swa_window_plan::vbr_swa_window_plan() : impl_(new impl) {}
vbr_swa_window_plan::~vbr_swa_window_plan() = default;
const std::vector<uint32_t> & vbr_swa_window_plan::destination_cells() const { return impl_->destination_cells; }
const std::vector<uint32_t> & vbr_swa_window_plan::base_cells() const { return impl_->base_cells; }
const std::vector<vbr_swa_window_membership_removal> & vbr_swa_window_plan::removals() const { return impl_->removals; }
uint32_t vbr_swa_window_plan::required_watermark() const { return impl_->required_watermark; }
size_t vbr_swa_window_plan::retained_bytes() const { return impl_->retained_bytes; }
bool vbr_swa_window_plan::current(llama_context & ctx, uint64_t source_epoch, uint64_t destination_epoch,
                                const std::array<uint8_t, 32> & execution) const {
    try {
        return vbr_swa_window_planner::current(*impl_, ctx, source_epoch, destination_epoch, execution);
    } catch (const std::bad_alloc &) {
        return false;
    }
}
vbr_swa_window_plan_result vbr_prepare_swa_window(llama_context & ctx,
        std::shared_ptr<const vbr_swa_window_image> image, const vbr_swa_window_plan_request & request) {
    try {
        return vbr_swa_window_planner::prepare(ctx, std::move(image), request);
    } catch (const std::bad_alloc &) {
        return {vbr_swa_window_status::allocation_failed, nullptr};
    }
}

vbr_swa_window_status vbr_install_swa_window(llama_context & ctx,
        std::unique_ptr<vbr_swa_window_plan> plan, const vbr_swa_window_install_request & request) {
    if (!plan) { return vbr_swa_window_status::unsupported; }
    try {
        return vbr_swa_window_planner::install(ctx, *plan, request);
    } catch (const std::bad_alloc &) {
        // All allocating work is before the first upload.
        return vbr_swa_window_status::allocation_failed;
    }
}
