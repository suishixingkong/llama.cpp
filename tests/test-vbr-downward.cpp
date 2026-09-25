#include "../src/llama-vbr-downward.h"
// The two consumer-side includes the upstream file carries are removed here: they are what
// makes the preflight and stash cases below unportable at this stage.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>


// Ported from buun-llama-cpp at the stage-2.6d slice: the thirteen cases that exercise
// llama-vbr-downward / llama-vbr-upward alone. Five upstream cases are deliberately absent
// because they are written against the *consumer* side, which does not exist in this tree yet:
//
//   test_iswa_physical_preflight_preserves_device_skew       } need llama_memory_vbr_preflight_data,
//   test_iswa_physical_preflight_shared_scratch_and_overflow } llama_memory_vbr_physical_growth and
//   test_incremental_preflight_tree_matches_full_fold        } llama_memory_vbr_preflight_tree, all
//                                                            three declared in src/llama-memory.h
//   llama_kv_cache_vbr_stash_batch_test + test_trusted_stash_batch_validation_is_linear
//                                                            need llama_kv_cache::vbr_pool and
//                                                            vbr_stash_requests_valid (llama-kv-cache.h)
//
// llama-memory.h and llama-kv-cache.h are the consumer work of stage 2.9, so those cases move there
// together with the upstream file itself. The helper structs only they used (the backend-identity
// constants and fake_preflight_source) are dropped with them; nothing else in this file referenced
// them, so the remaining coverage is unaffected rather than reduced.
//
// Assert-based like the stage-2.6a tests, which is also why upstream attaches /UNDEBUG to it.

static const auto HOST = llama_cache_acct_resource_domain::non_device(
    llama_cache_acct_residency::pageable_host);

static constexpr ggml_type tiers[] = {
    GGML_TYPE_F16, GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO4_0,
    GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO1_TCQ,
};


static void test_all_recipes_and_rejections() {
    size_t pairs = 0;
    for (size_t source = 0; source < 6; ++source) {
        for (size_t target = source + 1; target < 6; ++target) {
            vbr_downward_recipe recipe;
            assert(vbr_downward_resolve_recipe(
                tiers[source], tiers[target], GGML_TYPE_TURBO1_TCQ, true, recipe) ==
                vbr_downward_recipe_status::resolved);
            assert(recipe.n_edges == target - source);
            assert(recipe.edges[0].source_type == tiers[source]);
            assert(recipe.edges[recipe.n_edges - 1].target_type == tiers[target]);
            for (size_t i = 0; i < recipe.n_edges; ++i) {
                assert(recipe.edges[i].source_type == tiers[source + i]);
                assert(recipe.edges[i].target_type == tiers[source + i + 1]);
            }
            pairs++;
        }
    }
    assert(pairs == 15);

    vbr_downward_recipe out;
    assert(vbr_downward_resolve_recipe(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0,
        GGML_TYPE_TURBO1_TCQ, true, out) == vbr_downward_recipe_status::equal_tier);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO1_TCQ, true, out) == vbr_downward_recipe_status::upward_forbidden);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_Q4_0, GGML_TYPE_TURBO4_0,
        GGML_TYPE_TURBO1_TCQ, true, out) == vbr_downward_recipe_status::unsupported_type);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_F16, GGML_TYPE_TURBO2_TCQ,
        GGML_TYPE_TURBO3_TCQ, true, out) == vbr_downward_recipe_status::below_floor);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_F16, GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO1_TCQ, false, out) == vbr_downward_recipe_status::nonmovable);
}

struct destination_measure_fixture {
    size_t required_degraded_units = 0;
    uint32_t calls = 0;
    bool fail = false;
};

struct destination_last_step_fixture {
    uint64_t calls = 0;
};

static bool measure_destination_last_step(
        void * opaque,
        const std::vector<std::vector<ggml_type>> &,
        const llama_vbr_policy::selection *,
        vbr_import_destination_evidence & evidence) noexcept {
    auto * fixture = static_cast<destination_last_step_fixture *>(opaque);
    if (!fixture) {
        return false;
    }
    fixture->calls++;
    evidence.active = true;
    evidence.fits = fixture->calls ==
        uint64_t(VBR_IMPORT_DESTINATION_MAX_STEPS) + 1;
    return true;
}

static bool measure_destination_schedule(
        void * opaque,
        const std::vector<std::vector<ggml_type>> & types,
        const llama_vbr_policy::selection *,
        vbr_import_destination_evidence & evidence) noexcept {
    auto * fixture = static_cast<destination_measure_fixture *>(opaque);
    if (!fixture || fixture->fail) {
        return false;
    }
    fixture->calls++;
    size_t degraded = 0;
    for (const auto & child : types) {
        for (const auto type : child) {
            degraded += type == GGML_TYPE_TURBO8_0;
        }
    }
    evidence.active = true;
    evidence.fits = degraded >= fixture->required_degraded_units;
    evidence.pools = uint32_t(types.size());
    evidence.logical_bytes_needed = 100 - 20*degraded;
    evidence.logical_bytes_available =
        100 - 20*fixture->required_degraded_units;
    evidence.physical_growth_needed = 40 - 10*degraded;
    evidence.physical_growth_available =
        40 - 10*fixture->required_degraded_units;
    evidence.max_deficit = evidence.fits ? 0 : int64_t(std::max(
        evidence.logical_bytes_needed - evidence.logical_bytes_available,
        evidence.physical_growth_needed -
            evidence.physical_growth_available));
    return true;
}

static void test_import_destination_selects_shortest_controller_prefix() {
    std::vector<vbr_import_destination_child> children(2);
    for (size_t i = 0; i < children.size(); ++i) {
        children[i].initial_types = { GGML_TYPE_F16 };
        children[i].initial_cursor = 7 + i;
        children[i].watermark_cells = 256;
        children[i].policy.terminal_progress = 10;
        children[i].policy.steps.push_back({
            7 + i, 0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, 10,
        });
    }

    destination_measure_fixture current;
    const auto unchanged = vbr_select_import_destination(
        children, &current, measure_destination_schedule);
    assert(unchanged.status ==
        vbr_import_destination_status::feasible_current);
    assert(unchanged.prefix.empty());
    assert(current.calls == 1);

    destination_measure_fixture one { 1 };
    const auto selected = vbr_select_import_destination(
        children, &one, measure_destination_schedule);
    assert(selected.status ==
        vbr_import_destination_status::feasible_degraded);
    assert(selected.prefix.size() == 1);
    assert(selected.prefix[0].child_index == 0);
    assert(selected.final_types[0][0] == GGML_TYPE_TURBO8_0);
    assert(selected.final_types[1][0] == GGML_TYPE_F16);
    assert(selected.final_cursors[0] == 8);
    assert(one.calls == 2);
    assert(selected.pools == 2);
    assert(selected.logical_bytes_needed == 80);
    assert(selected.logical_bytes_available == 80);
    assert(selected.physical_growth_needed == 30);
    assert(selected.physical_growth_available == 30);
    assert(selected.max_deficit == 0);
    assert(std::any_of(
        selected.tree_digest.begin(), selected.tree_digest.end(),
        [](uint8_t value) { return value != 0; }));
    assert(vbr_import_destination_projection_coherent(children, selected));
    {
        auto bad = selected;
        bad.initial_types[0][0] = GGML_TYPE_TURBO8_0;
        assert(!vbr_import_destination_projection_coherent(children, bad));
    }
    {
        auto bad = selected;
        bad.initial_cursors[0]++;
        assert(!vbr_import_destination_projection_coherent(children, bad));
    }
    {
        auto bad = selected;
        bad.prefix[0].value.type_a = GGML_TYPE_TURBO8_0;
        assert(!vbr_import_destination_projection_coherent(children, bad));
    }
    {
        auto bad = selected;
        bad.final_cursors[0]++;
        assert(!vbr_import_destination_projection_coherent(children, bad));
    }
    {
        auto bad = selected;
        bad.child_type_digests[0][0] ^= 1;
        assert(!vbr_import_destination_projection_coherent(children, bad));
    }
    {
        auto bad = selected;
        bad.tree_digest[0] ^= 1;
        assert(!vbr_import_destination_projection_coherent(children, bad));
    }

    destination_measure_fixture both { 2 };
    const auto pair = vbr_select_import_destination(
        children, &both, measure_destination_schedule);
    assert(pair.status ==
        vbr_import_destination_status::feasible_degraded);
    assert(pair.prefix.size() == 2);
    assert(pair.final_types[0][0] == GGML_TYPE_TURBO8_0);
    assert(pair.final_types[1][0] == GGML_TYPE_TURBO8_0);
    assert(both.calls == 3);
    assert(vbr_import_destination_projection_coherent(children, pair));

    destination_measure_fixture impossible { 3 };
    const auto exhausted = vbr_select_import_destination(
        children, &impossible, measure_destination_schedule);
    assert(exhausted.status == vbr_import_destination_status::exhausted);
    assert(exhausted.prefix.size() == 2);
    assert(exhausted.logical_bytes_needed == 60);
    assert(exhausted.logical_bytes_available == 40);
    assert(exhausted.physical_growth_needed == 20);
    assert(exhausted.physical_growth_available == 10);
    assert(exhausted.max_deficit == 20);

    destination_measure_fixture failed;
    failed.fail = true;
    const auto invalid = vbr_select_import_destination(
        children, &failed, measure_destination_schedule);
    assert(invalid.status == vbr_import_destination_status::invalid);
    assert(invalid.prefix.empty());

    std::vector<vbr_import_destination_child> over_limit(1);
    over_limit[0].initial_types = { GGML_TYPE_F16 };
    over_limit[0].watermark_cells = 1;
    over_limit[0].policy.terminal_progress = 1;
    over_limit[0].policy.steps.resize(
        VBR_IMPORT_DESTINATION_MAX_STEPS + 1);
    destination_measure_fixture never_measured;
    const auto overflow = vbr_select_import_destination(
        over_limit, &never_measured, measure_destination_schedule);
    assert(overflow.status == vbr_import_destination_status::overflow);
    assert(overflow.prefix.empty());
    assert(never_measured.calls == 0);
}

static void test_policy_stream_max_shape_is_heap_bounded() {
    constexpr size_t children_count = 4096;
    constexpr size_t steps_per_child = 4;
    std::vector<llama_vbr_policy::child> children(children_count);
    for (size_t child = 0; child < children.size(); ++child) {
        children[child].terminal_progress = steps_per_child;
        children[child].steps.reserve(steps_per_child);
        for (size_t step = 0; step < steps_per_child; ++step) {
            children[child].steps.push_back({
                step, step, int32_t(step), int32_t(step + 1), 1,
            });
        }
    }
    llama_vbr_policy::shortest_prefix_stream stream(std::move(children));
    llama_vbr_policy::selection selected;
    size_t selected_count = 0;
    while (stream.next(selected) == llama_vbr_policy::result::selected) {
        selected_count++;
    }
    assert(selected_count == children_count*steps_per_child);
    // A scan selector needs ~67M comparisons here. The indexed heap stays
    // comfortably within the O((C+S) log C) envelope.
    assert(stream.comparison_count() < 1000000);
}

static void test_import_destination_exact_step_cap() {
    constexpr size_t units = 16384;
    static constexpr std::array<ggml_type, 6> ladder = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_TCQ,
        GGML_TYPE_TURBO2_TCQ,
    };
    std::vector<vbr_import_destination_child> children(1);
    auto & child = children[0];
    child.initial_types.assign(units, ladder[0]);
    child.watermark_cells = 1;
    child.policy.terminal_progress = VBR_IMPORT_DESTINATION_MAX_STEPS;
    child.policy.steps.reserve(VBR_IMPORT_DESTINATION_MAX_STEPS);
    for (size_t unit = 0; unit < units; ++unit) {
        for (size_t edge = 0; edge + 1 < ladder.size(); ++edge) {
            child.policy.steps.push_back({
                unit*5 + edge, unit, int32_t(ladder[edge]),
                int32_t(ladder[edge + 1]), 1,
            });
        }
    }
    destination_last_step_fixture fixture;
    const auto selected = vbr_select_import_destination(
        children, &fixture, measure_destination_last_step);
    assert(selected.status ==
        vbr_import_destination_status::feasible_degraded);
    assert(selected.prefix.size() == VBR_IMPORT_DESTINATION_MAX_STEPS);
    assert(fixture.calls ==
        uint64_t(VBR_IMPORT_DESTINATION_MAX_STEPS) + 1);
    assert(vbr_import_destination_projection_coherent(children, selected));
}

static llama_vbr_policy::child policy_child(
        std::initializer_list<llama_vbr_policy::step> steps) {
    llama_vbr_policy::child child;
    child.terminal_progress = 100;
    child.steps = steps;
    return child;
}

static void test_policy_projection_tree_and_ordinary() {
    vbr_downward_policy_child ordinary;
    ordinary.initial_types = { GGML_TYPE_F16, GGML_TYPE_TURBO8_0 };
    ordinary.target_types = { GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO4_0 };
    ordinary.policy = policy_child({
        { 0, 0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, 10 },
        { 1, 1, GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO4_0, 10 },
    });
    const auto one = vbr_downward_project_policy_prefix({ ordinary });
    assert(one.status == vbr_downward_policy_status::coherent);
    assert(one.prefix.size() == 2);
    assert(one.final_types[0] == ordinary.target_types);
    assert(one.final_cursors[0] == 2);

    vbr_downward_policy_child peer;
    peer.initial_types = { GGML_TYPE_TURBO4_0 };
    peer.target_types = { GGML_TYPE_TURBO3_TCQ };
    peer.policy = policy_child({
        { 0, 0, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_TCQ, 5 },
    });
    const auto tree = vbr_downward_project_policy_prefix({ ordinary, peer });
    assert(tree.status == vbr_downward_policy_status::coherent);
    assert(tree.prefix.size() == 3);
    assert(tree.final_types[0] == ordinary.target_types);
    assert(tree.final_types[1] == peer.target_types);
    assert(tree.final_cursors[0] == 2);
    assert(tree.final_cursors[1] == 1);
    assert(tree.tree_digest != one.tree_digest);

    // Cursor publication follows the absolute degrade-order index, not the
    // number of selected steps: skipped/non-applicable entries are normal in
    // straddled K/V schedules.
    vbr_downward_policy_child skipped;
    skipped.initial_types = { GGML_TYPE_F16 };
    skipped.target_types = { GGML_TYPE_TURBO8_0 };
    skipped.initial_cursor = 4;
    skipped.policy = policy_child({
        { 9, 0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, 10 },
    });
    const auto skipped_projection =
        vbr_downward_project_policy_prefix({ skipped });
    assert(skipped_projection.status ==
           vbr_downward_policy_status::coherent);
    assert(skipped_projection.final_cursors[0] == 10);

    peer.target_types = { GGML_TYPE_TURBO2_TCQ };
    const auto impossible = vbr_downward_project_policy_prefix({ ordinary, peer });
    assert(impossible.status == vbr_downward_policy_status::exhausted);

    vbr_downward_policy_child at_target;
    at_target.initial_types = { GGML_TYPE_TURBO4_0 };
    at_target.target_types = at_target.initial_types;
    at_target.policy = policy_child({
        { 0, 0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, -1 },
    });
    const auto already_coherent = vbr_downward_project_policy_prefix({ at_target });
    assert(already_coherent.status == vbr_downward_policy_status::coherent);
    assert(already_coherent.prefix.empty());
    assert(already_coherent.final_cursors[0] == 0);

    vbr_downward_policy_child bad_source;
    bad_source.initial_types = { GGML_TYPE_F16 };
    bad_source.target_types = { GGML_TYPE_TURBO8_0 };
    bad_source.policy = policy_child({
        { 0, 0, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO8_0, 1 },
    });
    assert(vbr_downward_project_policy_prefix({ bad_source }).status ==
        vbr_downward_policy_status::incoherent);

    vbr_downward_policy_child invalid = bad_source;
    invalid.policy.steps[0].type_a = GGML_TYPE_F16;
    invalid.policy.steps[0].logical_gain = -1;
    assert(vbr_downward_project_policy_prefix({ invalid }).status ==
        vbr_downward_policy_status::invalid);

    vbr_downward_policy_child overflow = bad_source;
    overflow.policy.steps[0].type_a = GGML_TYPE_F16;
    overflow.policy.initial_progress = std::numeric_limits<int64_t>::max();
    overflow.policy.steps[0].logical_gain = 1;
    assert(vbr_downward_project_policy_prefix({ overflow }).status ==
        vbr_downward_policy_status::overflow);
}

static bool live_capture(
        void *, ggml_type source, const std::vector<uint8_t> & bytes,
        std::vector<uint8_t> & stash) {
    stash.resize(std::min<size_t>(8, bytes.size()) + 1);
    stash[0] = uint8_t(source);
    std::copy_n(bytes.begin(), stash.size() - 1, stash.begin() + 1);
    return true;
}

static bool live_edge(
        void *, const vbr_downward_edge & edge, const std::vector<uint8_t> & source,
        const std::vector<uint8_t> * stash, std::vector<uint8_t> & target) {
    target.resize(source.size());
    const uint8_t salt = uint8_t(uint32_t(edge.source_type)*3u + uint32_t(edge.target_type)*5u);
    for (size_t i = 0; i < source.size(); ++i) {
        const uint8_t tap = stash && !stash->empty() ? (*stash)[i % stash->size()] : 0;
        target[i] = uint8_t((uint32_t(source[i])*17u + salt + tap) & 0xffu);
    }
    return true;
}

// Independent chaining/stash-policy oracle: the checking power is the explicit
// edge loop + capture-at-boundary rule; the per-edge byte math is shared with
// the fixture transforms so the two cannot drift apart.
static std::vector<uint8_t> explicit_live_oracle(
        const vbr_downward_recipe & recipe, const std::vector<uint8_t> & source,
        std::vector<uint8_t> * stash) {
    std::vector<uint8_t> current = source;
    for (size_t edge_index = 0; edge_index < recipe.n_edges; ++edge_index) {
        const auto & edge = recipe.edges[edge_index];
        if (edge.capture_stash_before && stash->empty()) {
            assert(live_capture(nullptr, edge.source_type, current, *stash));
        }
        std::vector<uint8_t> next;
        assert(live_edge(nullptr, edge, current,
            stash->empty() ? nullptr : stash, next));
        current = std::move(next);
    }
    return current;
}

static void test_edge_oracles_and_stash_boundaries() {
    const std::vector<uint8_t> source = { 1, 3, 5, 7, 9, 11, 13, 15, 17 };
    const vbr_downward_transform_iface iface { nullptr, live_capture, live_edge };
    for (size_t i = 0; i + 1 < 6; ++i) {
        vbr_downward_recipe recipe;
        assert(vbr_downward_resolve_recipe(tiers[i], tiers[i + 1],
            GGML_TYPE_TURBO1_TCQ, true, recipe) == vbr_downward_recipe_status::resolved);
        std::vector<uint8_t> oracle_stash;
        const auto oracle = explicit_live_oracle(recipe, source, &oracle_stash);
        const auto actual = vbr_downward_execute_recipe(recipe, source, nullptr, iface);
        assert(actual.status == vbr_downward_transform_status::transformed);
        assert(actual.bytes == oracle);
        assert(actual.stash == oracle_stash);
        assert(actual.stash_regenerated == (i >= 2));
    }

    vbr_downward_recipe boundary;
    assert(vbr_downward_resolve_recipe(GGML_TYPE_F16, GGML_TYPE_TURBO2_TCQ,
        GGML_TYPE_TURBO1_TCQ, true, boundary) == vbr_downward_recipe_status::resolved);
    const auto regenerated = vbr_downward_execute_recipe(boundary, source, nullptr, iface);
    assert(regenerated.status == vbr_downward_transform_status::transformed);
    assert(regenerated.stash_regenerated);
    assert(regenerated.stash[0] == uint8_t(GGML_TYPE_TURBO4_0));
}

static void test_straddled_kv_independent_chains() {
    const std::vector<uint8_t> k = { 2, 4, 6, 8 };
    const std::vector<uint8_t> v = { 1, 4, 9, 16 };
    const vbr_downward_transform_iface iface { nullptr, live_capture, live_edge };
    vbr_downward_recipe kr;
    vbr_downward_recipe vr;
    assert(vbr_downward_resolve_recipe(GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO3_TCQ,
        GGML_TYPE_TURBO1_TCQ, true, kr) == vbr_downward_recipe_status::resolved);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO1_TCQ,
        GGML_TYPE_TURBO1_TCQ, true, vr) == vbr_downward_recipe_status::resolved);
    assert(kr.n_edges == 2);
    assert(vr.n_edges == 2);
    assert(kr.source_type != vr.source_type && kr.target_type != vr.target_type);
    const auto ko = vbr_downward_execute_recipe(kr, k, nullptr, iface);
    const auto vo = vbr_downward_execute_recipe(vr, v, nullptr, iface);
    assert(ko.status == vbr_downward_transform_status::transformed);
    assert(vo.status == vbr_downward_transform_status::transformed);
    assert(ko.bytes != vo.bytes);
}

struct fake_workspace {
    size_t current = 0;
    size_t endpoint = 4096;
    bool fail_once = false;
    bool request_scaled_endpoint = false;
    size_t prices = 0;
    size_t reserves = 0;
};

static bool workspace_memory(
        ggml_backend_t backend, int, int64_t n_cells, int64_t, int64_t,
        size_t * now, size_t * endpoint) {
    auto * state = reinterpret_cast<fake_workspace *>(backend);
    state->prices++;
    *now = state->current;
    *endpoint = state->request_scaled_endpoint
        ? size_t(n_cells)*4096 : state->endpoint;
    return true;
}

static bool workspace_reserve(
        ggml_backend_t backend, int64_t n_cells, int64_t, int64_t) {
    auto * state = reinterpret_cast<fake_workspace *>(backend);
    state->reserves++;
    if (state->fail_once) {
        state->fail_once = false;
        state->current = state->endpoint / 2;
        return false;
    }
    state->current = state->request_scaled_endpoint
        ? size_t(n_cells)*4096 : state->endpoint;
    return true;
}

static bool workspace_memory_v2(
        ggml_backend_t backend, int device,
        const ggml_vbr_transcode_workspace_params_v2 * request,
        size_t * now, size_t * endpoint) {
    if (!request || !request->mean_addback) {
        return false;
    }
    if (!workspace_memory(
            backend, device, request->n_cells, request->ne0,
            request->stash_rows, now, endpoint)) {
        return false;
    }
    *endpoint += size_t(request->ne0)*sizeof(float);
    return true;
}

static bool workspace_reserve_v2(
        ggml_backend_t backend,
        const ggml_vbr_transcode_workspace_params_v2 * request) {
    if (!request || !request->mean_addback) {
        return false;
    }
    auto * state = reinterpret_cast<fake_workspace *>(backend);
    state->reserves++;
    state->current = size_t(request->n_cells)*4096 +
        size_t(request->ne0)*sizeof(float);
    return true;
}

static bool cross_domain_reconstruct(
        ggml_backend_t,
        const ggml_vbr_cross_domain_reconstruct_params *) {
    return true;
}

struct fake_stash {
    uint64_t current = 0;
    uint64_t endpoint = 8192;
    bool fail = false;
    uint64_t memory_calls = 0;
    uint64_t reserve_calls = 0;
};

static bool stash_memory(void * context, uint64_t & now, uint64_t & endpoint) {
    auto * state = static_cast<fake_stash *>(context);
    ++state->memory_calls;
    now = state->current;
    endpoint = state->endpoint;
    return true;
}

static bool stash_reserve(void * context) {
    auto * state = static_cast<fake_stash *>(context);
    ++state->reserve_calls;
    if (state->fail) {
        state->current = state->endpoint / 2;
        return false;
    }
    state->current = state->endpoint;
    return true;
}

static void configure_resource_ledger(llama_cache_acct_ledger & ledger) {
    const llama_cache_acct_completeness_requirement requirement = {
        HOST, llama_cache_acct_producer::host_cache,
    };
    assert(ledger.configure_required_producers(&requirement, 1));
    for (size_t i = 0; i < size_t(llama_cache_acct_category::_count); ++i) {
        const auto category = static_cast<llama_cache_acct_category>(i);
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, HOST, measure, 0);
        }
    }
    assert(ledger.certify_complete(HOST, llama_cache_acct_producer::host_cache));
}

static void test_projection_reserve_retry_and_stashless() {
    llama_cache_acct_ledger ledger;
    configure_resource_ledger(ledger);
    const uint64_t baseline_ops = ledger.snapshot().live_ops;
    fake_workspace workspace;
    fake_stash stash;
    ggml_vbr_backend_iface iface = {};
    iface.kv_transcode_workspace_memory = workspace_memory;
    iface.kv_transcode_workspace_reserve = workspace_reserve;
    const int workspace_owner = 1;
    const int stash_owner = 2;
    vbr_downward_workspace_endpoint w;
    w.owner = &workspace_owner;
    w.iface = &iface;
    w.backend = reinterpret_cast<ggml_backend_t>(&workspace);
    w.device = 0;
    w.domain = HOST;
    w.requests.push_back({ 128, 256, 32 });
    vbr_downward_stash_endpoint s;
    s.owner = &stash_owner;
    s.unit_ids = { 77 };
    s.domain = HOST;
    s.context = &stash;
    s.memory = stash_memory;
    s.reserve = stash_reserve;

    {
        vbr_downward_resource_receipts receipts(ledger);
        workspace.fail_once = true;
        auto first = receipts.reserve_resources({}, { w }, { s });
        assert(first.status == vbr_downward_reserve_status::workspace_reserve_failed);
        assert(first.workspace_growth == 4096);
        assert(first.stash_growth == 8192);
        const uint64_t charged_ops = ledger.snapshot().live_ops;
        assert(charged_ops == baseline_ops + 2);

        stash.fail = true;
        auto retry = receipts.reserve_resources({}, { w }, { s });
        assert(retry.status == vbr_downward_reserve_status::reserved_stashless);
        assert(retry.stashless_units == std::vector<uint64_t>({ 77 }));
        assert(retry.workspace_growth == 0);
        assert(retry.stash_growth == 0);
        assert(ledger.snapshot().live_ops == charged_ops);
    }
    assert(ledger.snapshot().live_ops == baseline_ops);
}

static void test_required_stash_reserve_fails_closed() {
    llama_cache_acct_ledger ledger;
    configure_resource_ledger(ledger);
    fake_workspace workspace;
    fake_stash stash;
    ggml_vbr_backend_iface iface = {};
    iface.kv_transcode_workspace_memory = workspace_memory;
    iface.kv_transcode_workspace_reserve = workspace_reserve;
    const int workspace_owner = 1;
    const int stash_owner = 2;
    vbr_downward_workspace_endpoint w;
    w.owner = &workspace_owner;
    w.iface = &iface;
    w.backend = reinterpret_cast<ggml_backend_t>(&workspace);
    w.device = 0;
    w.domain = HOST;
    w.requests.push_back({ 128, 256, 32 });
    vbr_downward_stash_endpoint s;
    s.owner = &stash_owner;
    s.unit_ids = { 77 };
    s.required = true;
    s.domain = HOST;
    s.context = &stash;
    s.memory = stash_memory;
    s.reserve = stash_reserve;
    stash.fail = true;

    vbr_downward_resource_receipts receipts(ledger);
    const auto result = receipts.reserve_resources({}, { w }, { s });
    assert(result.status ==
           vbr_downward_reserve_status::required_stash_reserve_failed);
    assert(result.stashless_units.empty());
    assert(stash.memory_calls == 1);
    assert(stash.reserve_calls == 1);
}

static void test_mixed_transform_and_stash_only_receipts() {
    llama_cache_acct_ledger ledger;
    configure_resource_ledger(ledger);
    fake_workspace workspace;
    fake_stash transformed_stash;
    fake_stash exact_stash;
    ggml_vbr_backend_iface iface = {};
    iface.kv_transcode_workspace_memory = workspace_memory;
    iface.kv_transcode_workspace_reserve = workspace_reserve;
    const int workspace_owner = 1;
    const int transformed_owner = 2;
    const int exact_owner = 3;
    vbr_downward_workspace_endpoint w;
    w.owner = &workspace_owner;
    w.iface = &iface;
    w.backend = reinterpret_cast<ggml_backend_t>(&workspace);
    w.device = 0;
    w.domain = HOST;
    w.requests.push_back({ 128, 256, 32 });
    vbr_downward_stash_endpoint transformed;
    transformed.owner = &transformed_owner;
    transformed.unit_ids = { 77 };
    transformed.required = true;
    transformed.domain = HOST;
    transformed.context = &transformed_stash;
    transformed.memory = stash_memory;
    transformed.reserve = stash_reserve;
    vbr_downward_stash_endpoint exact = transformed;
    exact.owner = &exact_owner;
    exact.unit_ids = { 78 };
    exact.context = &exact_stash;

    vbr_downward_resource_receipts receipts(ledger);
    const auto result = receipts.reserve_resources(
        {}, { w }, { transformed, exact });
    assert(result.status == vbr_downward_reserve_status::reserved);
    assert(transformed_stash.memory_calls == 1);
    assert(transformed_stash.reserve_calls == 1);
    assert(exact_stash.memory_calls == 1);
    assert(exact_stash.reserve_calls == 1);
}

static void test_workspace_prices_max_shape_but_reserves_once() {
    constexpr size_t request_count = 16384;
    llama_cache_acct_ledger ledger;
    configure_resource_ledger(ledger);
    fake_workspace workspace;
    workspace.request_scaled_endpoint = true;
    ggml_vbr_backend_iface iface = {};
    iface.kv_transcode_workspace_memory = workspace_memory;
    iface.kv_transcode_workspace_reserve = workspace_reserve;
    const int workspace_owner = 1;
    vbr_downward_workspace_endpoint endpoint;
    endpoint.owner = &workspace_owner;
    endpoint.iface = &iface;
    endpoint.backend = reinterpret_cast<ggml_backend_t>(&workspace);
    endpoint.device = 0;
    endpoint.domain = HOST;
    endpoint.requests.reserve(request_count);
    for (size_t i = 0; i < request_count; ++i) {
        // A full permutation keeps the largest tuple away from either edge,
        // catching first/last-request shortcuts while preserving exact scale.
        const int64_t n_cells = int64_t((i*8191) % request_count + 1);
        endpoint.requests.push_back({ n_cells, 256, 0 });
    }
    {
        vbr_downward_resource_receipts receipts(ledger);
        const auto result = receipts.reserve_resources({}, { endpoint }, {});
        assert(result.status == vbr_downward_reserve_status::reserved);
        assert(result.workspace_growth == request_count*4096);
        assert(workspace.prices == request_count);
        assert(workspace.reserves == 1);
        assert(workspace.current == request_count*4096);
    }
}

static void test_cross_domain_workspace_prices_mean_plane_once() {
    constexpr size_t request_count = 16384;
    llama_cache_acct_ledger ledger;
    configure_resource_ledger(ledger);
    fake_workspace workspace;
    workspace.request_scaled_endpoint = true;
    ggml_vbr_backend_iface iface = {};
    iface.kv_transcode_workspace_memory = workspace_memory;
    iface.kv_transcode_workspace_reserve = workspace_reserve;
    const ggml_vbr_cross_domain_iface_v1 cross_iface = {
        GGML_VBR_CROSS_DOMAIN_IFACE_V1_VERSION,
        sizeof(ggml_vbr_cross_domain_iface_v1),
        cross_domain_reconstruct,
        workspace_memory_v2,
        workspace_reserve_v2,
    };
    const int workspace_owner = 2;
    vbr_downward_workspace_endpoint endpoint;
    endpoint.owner = &workspace_owner;
    endpoint.iface = &iface;
    endpoint.cross_iface = &cross_iface;
    endpoint.backend = reinterpret_cast<ggml_backend_t>(&workspace);
    endpoint.device = 0;
    endpoint.domain = HOST;
    endpoint.requests.reserve(request_count);
    for (size_t i = 0; i < request_count; ++i) {
        const int64_t n_cells = int64_t((i*8191) % request_count + 1);
        endpoint.requests.push_back({ n_cells, 256, 0, true });
    }
    {
        vbr_downward_resource_receipts receipts(ledger);
        const auto result = receipts.reserve_resources({}, { endpoint }, {});
        assert(result.status == vbr_downward_reserve_status::reserved);
        assert(result.workspace_growth == request_count*4096 + 1024);
        assert(workspace.prices == request_count);
        assert(workspace.reserves == 1);
        assert(workspace.current == request_count*4096 + 1024);
    }
}

static void test_cross_domain_workspace_refuses_legacy_only_backend() {
    llama_cache_acct_ledger ledger;
    configure_resource_ledger(ledger);
    fake_workspace workspace;
    ggml_vbr_backend_iface legacy = {};
    legacy.kv_transcode_workspace_memory = workspace_memory;
    legacy.kv_transcode_workspace_reserve = workspace_reserve;
    const int workspace_owner = 3;
    vbr_downward_workspace_endpoint endpoint;
    endpoint.owner = &workspace_owner;
    endpoint.iface = &legacy;
    endpoint.backend = reinterpret_cast<ggml_backend_t>(&workspace);
    endpoint.device = 0;
    endpoint.domain = HOST;
    endpoint.requests.push_back({ 128, 256, 0, true });

    vbr_downward_resource_receipts receipts(ledger);
    const auto result = receipts.reserve_resources({}, { endpoint }, {});
    assert(result.status == vbr_downward_reserve_status::projection_unavailable);
    assert(workspace.prices == 0);
    assert(workspace.reserves == 0);
}

static void test_mean_addback_launch_shape_is_row_bounded() {
    ggml_vbr_mean_addback_launch_shape shape = {};
    assert(ggml_vbr_mean_addback_launch_shape_for(256, 6144, &shape));
    assert(shape.blocks == 256);
    assert(shape.threads == 256);
    assert(shape.blocks < (256u*6144u + shape.threads - 1)/shape.threads);
    assert(ggml_vbr_mean_addback_launch_shape_for(1048576, 6144, &shape));
    assert(shape.blocks == 1048576);
    assert(!ggml_vbr_mean_addback_launch_shape_for(0, 6144, &shape));
    assert(!ggml_vbr_mean_addback_launch_shape_for(1, 0, &shape));
}

int main() {
    test_all_recipes_and_rejections();
    test_import_destination_selects_shortest_controller_prefix();
    test_policy_stream_max_shape_is_heap_bounded();
    test_import_destination_exact_step_cap();
    test_policy_projection_tree_and_ordinary();
    test_edge_oracles_and_stash_boundaries();
    test_straddled_kv_independent_chains();
    test_projection_reserve_retry_and_stashless();
    test_required_stash_reserve_fails_closed();
    test_mixed_transform_and_stash_only_receipts();
    test_workspace_prices_max_shape_but_reserves_once();
    test_cross_domain_workspace_prices_mean_plane_once();
    test_cross_domain_workspace_refuses_legacy_only_backend();
    test_mean_addback_launch_shape_is_row_bounded();
    // The five consumer-side cases listed at the top of this file run at stage 2.9.
    std::puts("test-vbr-downward: OK (13 of the upstream 18 cases; see the note at the top)");
    return 0;
}
