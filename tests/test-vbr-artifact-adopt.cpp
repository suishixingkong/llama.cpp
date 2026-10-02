// VBR artifact-adoption tests: the portable, device-free subset.
//
// Ported from buun-llama-cpp tests/test-vbr-artifact-adopt.cpp, which is 6139
// lines and is really two programs sharing one main().  The unargumented run
// exercises the adoption state machine on the CPU; a `--vbr-*-cuda` run drives
// real GGUF models through the CUDA H2D adapter.  Only the first is here.
//
// What is NOT here, and why:
//   * fork 4042-5872 (1831 lines) -- test_cuda_h2d_adapter and the two
//     model_backed_* drivers.  They need a GPU, and their first statement is
//     `CHECK(device != nullptr); if (!device) return;`, so ported verbatim they
//     would pass here while asserting nothing at all.
//   * fork 61-583 (623 lines) -- the model-adoption / occupied-budget harness.
//     It is referenced only from 4227-5844, i.e. only from the model-backed
//     program above, so it comes with it rather than with this slice.
//   * fork 5938-5962 -- test_dflash_ring_frontier_header.  DFlash is ruled out
//     of this port.
//   * fork 930-3903 -- namespace adoption_fixture, the fake-memory adoption
//     fixture.  Self-contained but 2974 lines; it is its own slice.
//
// Self-reporting (a failure counter and an explicit pass line), so NDEBUG
// cannot hollow it out and it needs no /UNDEBUG.  The counter is deliberate:
// these seven functions assert 100+ conditions and one run should report all
// of them, not stop at the first.

#include "llama-vbr-artifact-adopt.h"
#include "llama-bit-ops.h"

#include "llama-context.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-vbr-generation.h"
#include "llama-vbr-operation.h"
#include "llama-memory-tree.h"
#include "llama-vbr-artifact-catalog.h"
#include "llama-vbr-explicit-capture.h"
#include "llama-vbr-identity-digest.h"
#include "llama-vbr-upward.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                         __FILE__, __LINE__, #cond); \
            ++failures; \
        } \
    } while (0)

static void test_epoch_capacity_preflight() {
    CHECK(vbr_artifact_epoch_capacity(0, 0, 0, 0));
    CHECK(!vbr_artifact_epoch_capacity(0, UINT64_MAX, 0, 0));
    CHECK(!vbr_artifact_epoch_capacity(0, 0, UINT64_MAX, 0));
    CHECK(vbr_artifact_epoch_capacity(UINT64_MAX, 0, 0, 0));
    CHECK(!vbr_artifact_epoch_capacity(UINT64_MAX, 0, 0, 1));
    CHECK(vbr_artifact_epoch_capacity(UINT64_MAX - 1, 0, 0, 1));
    CHECK(!vbr_artifact_epoch_capacity(UINT64_MAX - 1, 0, 0, 2));
}
static vbr_checkpoint_generation_controller source_controller(
        vbr_lineage_uuid lineage) {
    vbr_checkpoint_generation_controller source;
    source.child_id = 0;
    source.dependency_mode =
        checkpoint_child_dependency_mode::payload_complete;
    source.lineage_uuid = lineage;
    source.global_generation = 9;
    source.units.push_back({
        17, GGML_TYPE_F16, GGML_TYPE_F16,
        vbr_repr_domain::full, 0, vbr_repr_transition::initial,
    });
    vbr_checkpoint_generation_stream stream;
    stream.stream_index = 0;
    stream.dependency_seq_id = 0;
    stream.computation_frontier = 11;
    stream.captured_dependency_count = 1;
    vbr_generation_page_ref page;
    page.page_index = 0;
    page.captured_page_gen = 23;
    page.covered_mask[0] = uint64_t(1) << 5;
    stream.pages.push_back(page);
    source.streams.push_back(stream);
    return source;
}

static vbr_artifact_stream_placement one_cell_placement() {
    vbr_artifact_stream_placement placement;
    placement.child_id = 0;
    placement.stream_index = 0;
    placement.source_sequence = 0;
    placement.cells.push_back({ 5, 10, 1, 1 });
    return placement;
}

static vbr_operation_binding import_binding(
        vbr_controller_instance_id instance) {
    vbr_operation_binding binding;
    binding.kind = vbr_operation_kind::state_import;
    binding.child_phase = vbr_operation_phase::mutate;
    CHECK(vbr_binding_add_instance_target(
        binding, vbr_operation_kind::state_import,
        vbr_operation_class::state_api, instance, VBR_STREAM_ANY,
        1, 0, std::numeric_limits<llama_pos>::max()));
    return binding;
}

static void install_and_check(
        vbr_generation_tracker & target,
        const vbr_tracker_install_child & plan,
        const vbr_checkpoint_generation_controller & source,
        bool native) {
    const auto runtime = target.runtime_instance();
    const auto target_lineage = target.lineage_identity();
    vbr_tracker_import_image image;
    CHECK(target.prepare_import_image(
        plan, source, 1, { one_cell_placement() }, image));
    CHECK(image.ready());
    CHECK(image.stable());

    auto binding = import_binding(runtime);
    vbr_scoped_operation operation(binding);
    CHECK(bool(operation));
    CHECK(target.import_image_installable(image, operation.id()));
    target.install_import_image_swap(image);
    CHECK(!image.ready());
    CHECK(target.runtime_instance() == runtime);
    CHECK(vbr_controller_instance_owned_by(runtime, &target));
    CHECK(target.stable());
    CHECK(target.controller_generation() == plan.global_generation);
    CHECK(target.lineage_identity() ==
          (native ? source.lineage_uuid : target_lineage));
    CHECK(target.page_generation(0, 0) == (native ? 23u : 1u));
    CHECK(target.dependency_generation(0, 5) ==
          (native ? 23u : 1u));
    CHECK(target.membership_generation(0, 5) ==
          (native ? 23u : 1u));
    CHECK(target.last_membership_seq(0, 5) == 1);
    const uint16_t import_provenance =
        uint16_t(vbr_mutation_family::import) |
        (uint16_t(vbr_operation_class::state_api) << 8);
    CHECK(target.dependency_provenance(0, 5) == import_provenance);
    CHECK(target.membership_provenance(0, 5) == import_provenance);
    const auto dep = target.dependency_extent(0, 5);
    const auto mem = target.membership_extent(0, 5);
    const auto * dep_extent = target.extent_store().lookup_committed(dep);
    const auto * mem_extent = target.extent_store().lookup_committed(mem);
    CHECK(dep_extent != nullptr);
    CHECK(mem_extent != nullptr);
    if (dep_extent) {
        CHECK(dep_extent->family == vbr_mutation_family::import);
        CHECK(dep_extent->operation_class == vbr_operation_class::state_api);
        CHECK(dep_extent->stream == 0);
        CHECK(dep_extent->seq_id == 1);
        CHECK(dep_extent->p0 == 0);
        CHECK(dep_extent->p1 == 11);
    }
    if (mem_extent) {
        CHECK(mem_extent->family == vbr_mutation_family::import);
        CHECK(mem_extent->operation_class == vbr_operation_class::state_api);
        CHECK(mem_extent->stream == 0);
        CHECK(mem_extent->seq_id == 1);
        CHECK(mem_extent->p0 == 0);
        CHECK(mem_extent->p1 == 11);
    }
    CHECK(target.unit_generation(0).repr_gen == plan.units[0].repr_gen);
    CHECK(target.unit_generation(0).last_transition ==
          plan.units[0].last_transition);
    operation.close(vbr_operation_outcome::committed);
}
static void test_native_tracker_image_preserves_lineage_and_runtime() {
    const vbr_lineage_uuid source_lineage { 0x1234, 0x5678 };
    const auto source = source_controller(source_lineage);
    vbr_generation_tracker target(1, 256, 1,
                                  vbr_lineage_uuid { 0xa1, 0xb2 });
    CHECK(target.active());
    CHECK(target.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full));

    vbr_tracker_install_child plan;
    plan.child_id = 0;
    plan.transition = vbr_tracker_install_transition::native_clone;
    plan.lineage_uuid = source_lineage;
    plan.target_instance = target.runtime_instance();
    plan.global_generation = source.global_generation;
    plan.units = source.units;
    install_and_check(target, plan, source, true);
}

static void test_real_tracker_import_settle_and_teardown() {
    const auto source = source_controller(
        vbr_lineage_uuid { 0x1234, 0x5678 });
    auto target = std::make_unique<vbr_generation_tracker>(
        1, 256, 1, vbr_lineage_uuid { 0xa3, 0xb4 });
    CHECK(target->active());
    CHECK(target->initialize_unit(
        0, GGML_TYPE_F16, vbr_repr_domain::full));

    vbr_tracker_install_child plan;
    plan.child_id = 0;
    plan.transition = vbr_tracker_install_transition::native_clone;
    plan.lineage_uuid = source.lineage_uuid;
    plan.target_instance = target->runtime_instance();
    plan.global_generation = source.global_generation;
    plan.units = source.units;

    vbr_tracker_import_image image;
    CHECK(target->prepare_import_image(
        plan, source, 1, { one_cell_placement() }, image));
    auto binding = import_binding(target->runtime_instance());
    vbr_scoped_operation operation(binding);
    CHECK(bool(operation));
    const int32_t recovery = vbr_recovery_reserve(
        operation.id(), target->runtime_instance());
    CHECK(recovery >= 0);
    CHECK(target->import_image_installable(image, operation.id()));
    target->install_import_image_swap(image);
    CHECK(target->stable());
    CHECK(vbr_recovery_release_unused(recovery, operation.id()));
    CHECK(operation.close(vbr_operation_outcome::committed));

    // Model the first post-import append under a normal decode operation.  Its
    // event must settle mutation parity before the imported tracker dies.
    vbr_operation_binding decode;
    decode.kind = vbr_operation_kind::decode;
    decode.child_phase = vbr_operation_phase::mutate;
    CHECK(vbr_binding_add_instance_target(
        decode, vbr_operation_kind::decode,
        vbr_operation_class::ordinary_decode,
        target->runtime_instance(), 0, 1, 11,
        std::numeric_limits<llama_pos>::max()));
    vbr_scoped_operation decode_operation(decode);
    CHECK(bool(decode_operation));
    const int32_t decode_recovery = vbr_recovery_reserve(
        decode_operation.id(), target->runtime_instance());
    CHECK(decode_recovery >= 0);
    {
        auto event = target->begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode, 0,
            vbr_generation_stamp_kind::dependency,
            decode_operation.id());
        CHECK(bool(event));
        CHECK(target->stamp_cell(event, 6, 1, 11));
    }
    CHECK(target->stable());
    vbr_operation_binding live_decode;
    CHECK(vbr_operation_registry_binding(
        decode_operation.id(), live_decode));
    CHECK(live_decode.kind == vbr_operation_kind::decode);
    CHECK(live_decode.n_targets == 1);
    CHECK(live_decode.targets[0].operation_class ==
          vbr_operation_class::ordinary_decode);
    CHECK(live_decode.targets[0].instance_id ==
          target->runtime_instance());
    CHECK(target->teardown_state() ==
          vbr_generation_teardown_state::operation_live);
    CHECK(vbr_recovery_release_unused(
        decode_recovery, decode_operation.id()));
    CHECK(decode_operation.close(vbr_operation_outcome::committed));
    CHECK(target->teardown_state() ==
          vbr_generation_teardown_state::clean);
    target.reset();
}

static void test_live_rebased_tracker_image_is_fresh() {
    const auto source = source_controller(vbr_lineage_uuid { 0x1234, 0x5678 });
    const vbr_lineage_uuid target_lineage { 0xc1, 0xd2 };
    vbr_generation_tracker target(1, 256, 1, target_lineage);
    CHECK(target.active());
    CHECK(target.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full));

    vbr_tracker_install_child plan;
    plan.child_id = 0;
    plan.transition = vbr_tracker_install_transition::whole_import;
    plan.lineage_uuid = target_lineage;
    plan.target_instance = target.runtime_instance();
    plan.global_generation = 1;
    plan.units.push_back({
        1, GGML_TYPE_F16, GGML_TYPE_F16,
        vbr_repr_domain::full, 0, vbr_repr_transition::whole_import,
    });
    install_and_check(target, plan, source, false);
}

static void test_native_tracker_rejects_uncovered_cell_and_tuple_splice() {
    const vbr_lineage_uuid source_lineage { 0x1234, 0x5678 };
    const auto source = source_controller(source_lineage);
    vbr_generation_tracker target(1, 256, 1,
                                  vbr_lineage_uuid { 0x91, 0x92 });
    CHECK(target.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full));
    vbr_tracker_install_child plan;
    plan.child_id = 0;
    plan.transition = vbr_tracker_install_transition::native_clone;
    plan.lineage_uuid = source_lineage;
    plan.target_instance = target.runtime_instance();
    plan.global_generation = source.global_generation;
    plan.units = source.units;

    auto foreign = one_cell_placement();
    foreign.cells[0].physical_cell = 6;
    vbr_tracker_import_image image;
    CHECK(!target.prepare_import_image(plan, source, 1, { foreign }, image));
    CHECK(!image.ready());
    CHECK(target.extent_store().live_entries() == 0);

    plan.global_generation++;
    CHECK(!target.prepare_import_image(
        plan, source, 1, { one_cell_placement() }, image));
    CHECK(!image.ready());
    CHECK(target.extent_store().live_entries() == 0);
}

static void test_closed_vocabularies() {
    for (uint8_t i = 0; i < uint8_t(vbr_adopt_phase::_count); ++i) {
        CHECK(std::string(vbr_adopt_phase_name(vbr_adopt_phase(i))) !=
              "invalid");
    }
    for (uint8_t i = 0; i < uint8_t(vbr_adopt_status::_count); ++i) {
        CHECK(std::string(vbr_adopt_status_name(vbr_adopt_status(i))) !=
              "invalid");
    }
    for (uint8_t i = 0;
         i < uint8_t(vbr_adopt_recovery_outcome::_count); ++i) {
        CHECK(std::string(vbr_adopt_recovery_outcome_name(
                  vbr_adopt_recovery_outcome(i))) != "invalid");
    }
    for (uint8_t i = 0;
         i < uint8_t(vbr_downward_reserve_status::_count); ++i) {
        CHECK(std::string(vbr_downward_reserve_status_name(
                  vbr_downward_reserve_status(i))) != "invalid");
    }
    vbr_adopt_stage_result native_stage_default;
    CHECK(native_stage_default.transform_status ==
          vbr_downward_reserve_status::not_attempted);
    CHECK(std::string(vbr_adopt_phase_name(vbr_adopt_phase::_count)) ==
          "invalid");
    CHECK(std::string(vbr_adopt_status_name(vbr_adopt_status::_count)) ==
          "invalid");
}
static bool fake_target_empty(const void * context) noexcept {
    return context && *static_cast<const bool *>(context);
}

static void test_complete_tree_barrier_fail_closed() {
    auto * attention0 = reinterpret_cast<llama_kv_cache *>(uintptr_t(0x1010));
    auto * attention1 = reinterpret_cast<llama_kv_cache *>(uintptr_t(0x2020));
    auto * recurrent = reinterpret_cast<llama_memory_recurrent *>(uintptr_t(0x3030));
    const std::vector<vbr_adopt_expected_attention> expected = {
        { 0, attention0 }, { 1, attention1 },
    };
    std::vector<llama_memory_tree_child> live = {
        { 0, attention0, nullptr,
          checkpoint_child_dependency_mode::payload_complete },
    };
    CHECK(vbr_adopt_check_complete_tree(expected, live, {}) ==
          vbr_adopt_status::target_drift);

    live.push_back({ 1, attention1, recurrent,
                     checkpoint_child_dependency_mode::payload_complete });
    CHECK(vbr_adopt_check_complete_tree(expected, live, {}) ==
          vbr_adopt_status::required_companion_unavailable);

    bool empty = false;
    vbr_companion_adoption_provider provider;
    provider.kind = vbr_artifact_companion_kind::recurrent;
    provider.target_cookie = recurrent;
    provider.context = &empty;
    provider.target_empty = fake_target_empty;
    CHECK(vbr_adopt_check_complete_tree(expected, live, { provider }) ==
          vbr_adopt_status::target_drift);

    empty = true;
    CHECK(vbr_adopt_check_complete_tree(expected, live, { provider }) ==
          vbr_adopt_status::adopted);
    CHECK(vbr_adopt_check_complete_tree(
              expected, live, { provider, provider }) ==
          vbr_adopt_status::required_companion_unavailable);

    auto * qsa = reinterpret_cast<llama_memory_hybrid_idx *>(
        uintptr_t(0x4040));
    live[0].qsa_index_owner = qsa;
    CHECK(vbr_adopt_check_complete_tree(expected, live, { provider }) ==
          vbr_adopt_status::required_companion_unavailable);
    vbr_companion_adoption_provider qsa_provider;
    qsa_provider.kind = vbr_artifact_companion_kind::qsa_index;
    qsa_provider.target_cookie = qsa;
    bool qsa_empty_after_prepare = false;
    qsa_provider.context = &qsa_empty_after_prepare;
    qsa_provider.target_empty = fake_target_empty;
    qsa_provider.attention_child_id = 0;
    // QSA prepares reversibly in-place, so it is expected to be non-empty at
    // the complete-tree barrier.  Its later recheck authenticates that state.
    CHECK(vbr_adopt_check_complete_tree(
              expected, live, { provider, qsa_provider }) ==
          vbr_adopt_status::adopted);
    CHECK(vbr_adopt_check_complete_tree(
              expected, live, { provider, qsa_provider }, true) ==
          vbr_adopt_status::adopted);
    CHECK(vbr_adopt_check_complete_tree(
              expected, live, { provider, qsa_provider, qsa_provider }) ==
          vbr_adopt_status::required_companion_unavailable);
}

int main() {
    test_epoch_capacity_preflight();
    test_native_tracker_image_preserves_lineage_and_runtime();
    test_real_tracker_import_settle_and_teardown();
    test_live_rebased_tracker_image_is_fresh();
    test_native_tracker_rejects_uncovered_cell_and_tuple_splice();
    test_closed_vocabularies();
    test_complete_tree_barrier_fail_closed();

    if (failures > 0) {
        std::fprintf(stderr, "%d VBR artifact-adoption test(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("VBR artifact adoption (portable subset): all tests passed\n");
    return EXIT_SUCCESS;
}
