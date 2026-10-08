// Tests for llama-vbr-operation.h/.cpp - the operation registry and its closed vocabularies.
//
// Hand-written rather than ported: upstream's test-vbr-representation-epoch.cpp covers the
// operation CPU sub-mode inside one 765-line function that also reaches llama-kv-cache.h, so it
// cannot be cut along function boundaries and its file-level includes pull in explicit-capture and
// artifact-capture types that only exist from stage 2.6f onwards. See MERGE_BUUN_PLAN §10.33.
//
// What is covered here is everything that decides whether an operation may exist at all:
//   1. the closed vocabularies (names, registries, bit helpers, kind->family and kind->registrant
//      maps) plus the properties the compile-time checks do NOT establish - notably that no kind's
//      registrant mask reaches outside the registrant vocabulary;
//   2. the target covering predicates, which are the single spelling shared by begin-time and
//      per-stamp authentication;
//   3. the closed per-kind range enumeration, including the two enumerated special forms;
//   4. binding construction, including the refusal paths;
//   5. the mint validation matrix - one rejection per documented rejection rule, asserted to leave
//      binding.operation_id untouched, plus the accepting cases (including recovery's documented
//      empty-instance exception);
//   6. the registry lifecycle and quiescence probes;
//   7. the recovery ring's reserve/release/record transitions and single-use minting.
//
// Self-reporting (status from main plus explicit PASS lines), so NDEBUG cannot hollow it out and it
// needs no /UNDEBUG - same shape as test-vbr-identity-digest and test-vbr-hard-seal.

#include "llama-vbr-operation.h"

// LLAMA_MAX_SEQ is the closed sequence domain the
// registry validates against; it is declared in llama-cparams.h rather than re-exported here.
#include "llama-cparams.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static constexpr size_t KIND_COUNT      = static_cast<size_t>(vbr_operation_kind::count);
static constexpr size_t CLASS_COUNT     = static_cast<size_t>(vbr_operation_class::count);
static constexpr size_t FAMILY_COUNT    = static_cast<size_t>(vbr_mutation_family::count);
static constexpr size_t PHASE_COUNT     = static_cast<size_t>(vbr_operation_phase::count);
static constexpr size_t REGISTRANT_COUNT =
        static_cast<size_t>(vbr_mutation_registrant::count);

// Instances are opaque 128-bit values that count as "set" only when both words are nonzero.
static vbr_controller_instance_id make_instance(uint64_t lo) {
    vbr_controller_instance_id instance;
    instance.hi = 0x0123456789abcdefull;
    instance.lo = lo;
    return instance;
}

// A binding that the registry accepts, so every rejection case can be built by perturbing one
// field of a known-good manifest rather than by assembling each one from scratch.
static vbr_operation_binding valid_binding(uint64_t instance_lo = 7) {
    return vbr_mutation_binding(vbr_operation_kind::decode, 3, 0, 10,
                                vbr_operation_class::ordinary_decode,
                                make_instance(instance_lo), 0);
}

// Every rejection must return an empty id AND leave the manifest's operation_id untouched: a
// rejected begin that half-stamped the binding would poison a later retry.
static void expect_rejected(const vbr_operation_binding & candidate, const char * what) {
    vbr_operation_binding binding = candidate;
    const vbr_operation_id id = vbr_operation_registry_begin(binding);
    if (id) {
        vbr_operation_registry_end(id);
        throw std::runtime_error(std::string("invalid manifest was accepted: ") + what);
    }
    check(!binding.operation_id, (std::string("rejected begin stamped operation_id: ") + what).c_str());
}

int main() {
    try {
        // [1] Closed vocabularies.
        {
            check(VBR_OPERATION_KIND_NAMES.size() == KIND_COUNT,
                  "kind-name table size does not match the kind enum");
            for (size_t i = 0; i < KIND_COUNT; ++i) {
                const char * name = VBR_OPERATION_KIND_NAMES[i];
                check(name != nullptr && name[0] != '\0', "operation kind name is empty");
                for (size_t j = i + 1; j < KIND_COUNT; ++j) {
                    check(std::strcmp(name, VBR_OPERATION_KIND_NAMES[j]) != 0,
                          "two operation kinds share a name");
                }
            }

            // The compile-time asserts prove coverage. What they cannot prove is that a *runtime*
            // call still agrees - these two are the functions the asserts call, so re-running them
            // here also catches a table that was edited into an inconsistent state.
            check(vbr_mutation_registry_is_exhaustive(),
                  "mutation registry is no longer exhaustive");
            check(vbr_stable_read_registry_is_exhaustive(),
                  "stable-read registry is no longer exhaustive");

            // Class bits are a 1:1 mapping onto distinct class slots.
            uint16_t seen_classes = 0;
            for (size_t i = 0; i < CLASS_COUNT; ++i) {
                const uint16_t bit = vbr_operation_class_bit(static_cast<vbr_operation_class>(i));
                check(bit == (uint16_t(1u) << i), "class bit is not 1u << class index");
                check((seen_classes & bit) == 0, "two operation classes share a bit");
                seen_classes |= bit;
            }

            // Registrant bits must stay disjoint and, unlike the class mask, they are 32-bit wide
            // because boundary recovery needs the whole vocabulary in one mask.
            uint32_t seen_registrants = 0;
            for (size_t i = 0; i < REGISTRANT_COUNT; ++i) {
                const uint32_t bit = vbr_registrant_bit(static_cast<vbr_mutation_registrant>(i));
                check(bit == (uint32_t(1u) << i), "registrant bit is not 1u << registrant index");
                check((seen_registrants & bit) == 0, "two registrants share a bit");
                seen_registrants |= bit;
            }
            check(REGISTRANT_COUNT <= 32, "registrant vocabulary no longer fits one 32-bit mask");

            // Every registry row must stay inside the closed domains. The exhaustiveness check
            // covers presence, not range, so an out-of-range enum value would sail past it.
            for (const auto & registration : VBR_MUTATION_REGISTRY) {
                check(static_cast<size_t>(registration.registrant) < REGISTRANT_COUNT,
                      "mutation registry row has an out-of-range registrant");
                check(static_cast<size_t>(registration.family) < FAMILY_COUNT,
                      "mutation registry row has an out-of-range family");
                check(static_cast<size_t>(registration.phase) < PHASE_COUNT,
                      "mutation registry row has an out-of-range phase");
                check(registration.allowed_classes != 0,
                      "mutation registry row allows no operation class");
                check((registration.allowed_classes & ~seen_classes) == 0,
                      "mutation registry row allows a class outside the class enum");
            }
            uint32_t seen_stable = 0;
            for (vbr_stable_read_registrant registrant : VBR_STABLE_READ_REGISTRY) {
                const uint32_t bit = uint32_t(1u) << static_cast<uint8_t>(registrant);
                check(static_cast<size_t>(registrant) <
                          static_cast<size_t>(vbr_stable_read_registrant::count),
                      "stable-read registry has an out-of-range registrant");
                check((seen_stable & bit) == 0, "stable-read registry repeats a registrant");
                seen_stable |= bit;
            }

            // kind -> family. swa_wrap is the one class whose provenance family is occupied_reuse
            // regardless of kind, and it must win over every single kind's own mapping.
            const vbr_mutation_family expected[KIND_COUNT] = {
                vbr_mutation_family::clear,    // retier_freeze (no explicit arm)
                vbr_mutation_family::append,   // decode
                vbr_mutation_family::trim,     // sequence_edit
                vbr_mutation_family::restore,  // checkpoint_restore
                vbr_mutation_family::import,   // state_import
                vbr_mutation_family::clear,    // state_export (no explicit arm)
                vbr_mutation_family::degrade,  // controller_retier
                vbr_mutation_family::recovery, // recovery
                vbr_mutation_family::import,   // window_restore
            };
            for (size_t i = 0; i < KIND_COUNT; ++i) {
                const auto kind = static_cast<vbr_operation_kind>(i);
                check(vbr_operation_kind_family(kind) == expected[i],
                      "operation kind maps to the wrong mutation family");
                check(vbr_operation_kind_family(kind, vbr_operation_class::swa_wrap) ==
                          vbr_mutation_family::occupied_reuse,
                      "swa_wrap did not override the kind's provenance family");
            }

            // kind -> registrant mask. Capacity boundary owns the whole vocabulary, so its mask is
            // the full 32-bit range; every other kind must stay inside the registrant vocabulary.
            check(vbr_operation_kind_registrants(vbr_operation_kind::recovery) == ~uint32_t(0),
                  "recovery kind must declare the full registrant mask");
            for (size_t i = 0; i < KIND_COUNT; ++i) {
                const auto kind = static_cast<vbr_operation_kind>(i);
                const uint32_t mask = vbr_operation_kind_registrants(kind);
                if (kind != vbr_operation_kind::recovery) {
                    check((mask & ~seen_registrants) == 0,
                          "operation kind declares a registrant outside the vocabulary");
                }
                check(mask != 0 || kind == vbr_operation_kind::retier_freeze ||
                          kind == vbr_operation_kind::state_export,
                      "operation kind declares no registrant although it has mutation sites");
            }
            std::puts("VBR operation PASS: closed vocabularies");
        }

        // [2] Target covering predicates.
        {
            const auto instance_a = make_instance(11);
            const auto instance_b = make_instance(22);
            const vbr_controller_instance_id unset {};

            // An empty instance is the declared any-controller wildcard; a set one matches exactly.
            vbr_operation_target wildcard = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                unset, 0, 5, 0, 10);
            vbr_operation_target exact = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                instance_a, 0, 5, 0, 10);
            check(wildcard.instance_matches(instance_a) && wildcard.instance_matches(instance_b),
                  "unset instance did not behave as an any-controller wildcard");
            check(exact.instance_matches(instance_a) && !exact.instance_matches(instance_b),
                  "set instance did not match exactly");

            // Stream wildcard is a protocol fact, not a per-site literal.
            vbr_operation_target any_stream = wildcard;
            any_stream.stream = VBR_STREAM_ANY;
            check(any_stream.stream_matches(0) && any_stream.stream_matches(7),
                  "VBR_STREAM_ANY did not match every stream");
            check(wildcard.stream_matches(0) && !wildcard.stream_matches(1),
                  "a concrete stream did not match exactly");

            // matches() is the event-constant half of the tuple: class equality, a registrant bit
            // actually present in the mask, and the mutate child phase.
            const uint32_t append_bit = vbr_registrant_bit(vbr_mutation_registrant::apply_ubatch_append);
            check(exact.matches(instance_a, 0, vbr_operation_class::ordinary_decode, append_bit),
                  "a fully matching tuple was rejected");
            check(!exact.matches(instance_b, 0, vbr_operation_class::ordinary_decode, append_bit),
                  "tuple matched the wrong controller instance");
            check(!exact.matches(instance_a, 1, vbr_operation_class::ordinary_decode, append_bit),
                  "tuple matched the wrong stream");
            check(!exact.matches(instance_a, 0, vbr_operation_class::swa_wrap, append_bit),
                  "tuple ignored the operation class");
            check(!exact.matches(instance_a, 0, vbr_operation_class::ordinary_decode,
                                 vbr_registrant_bit(vbr_mutation_registrant::window_install)),
                  "tuple matched a registrant outside the target's mask");
            vbr_operation_target non_mutate = exact;
            non_mutate.child_phase = vbr_operation_phase::prepare;
            check(!non_mutate.matches(instance_a, 0, vbr_operation_class::ordinary_decode, append_bit),
                  "tuple matched a non-mutate child phase");

            // Sequence coverage: a concrete target covers only its own sequence and never the
            // unknown (-1) one; -1 on the target is the declared any-sequence wildcard.
            check(exact.seq_covers(5), "concrete target did not cover its own sequence");
            check(!exact.seq_covers(4) && !exact.seq_covers(6),
                  "concrete target covered a different sequence");
            check(!exact.seq_covers(-1), "concrete target covered the unknown sequence");
            // A declared wildcard covers the unknown marker as well; a concrete value never does.
            // This matches pos_covers' shape: the wildcard form is total, the concrete form is not,
            // so a whole-cache edit can stamp a cell whose sequence is unknown while a concrete
            // target must name a real sequence.
            vbr_operation_target any_seq = exact;
            any_seq.seq_id = -1;
            check(any_seq.seq_covers(0) && any_seq.seq_covers(99),
                  "seq wildcard did not cover every sequence");
            check(any_seq.seq_covers(-1), "seq wildcard did not cover the unknown sequence");

            // Position coverage. A negative p0 is the kind-declared whole-range wildcard; an
            // unknown position (-1) is covered ONLY by a whole range, because whole-cache edits
            // stamp cells whose position is not consulted.
            vbr_operation_target whole = exact;
            whole.range = { 0, std::numeric_limits<llama_pos>::max() };
            check(whole.pos_covers(0) && whole.pos_covers(12345), "whole range missed a position");
            check(whole.pos_covers(-1), "whole range did not cover an unknown position");
            check(exact.pos_covers(0) && exact.pos_covers(9) && !exact.pos_covers(10),
                  "half-open range boundary is wrong");
            check(!exact.pos_covers(-1), "concrete range covered an unknown position");
            vbr_operation_target kind_wildcard = exact;
            kind_wildcard.range = { -1, -1 };
            check(kind_wildcard.pos_covers(-1) && kind_wildcard.pos_covers(0) &&
                      kind_wildcard.pos_covers(1 << 20),
                  "kind-declared wildcard range did not cover everything");

            // Covering-target search agrees with the predicates. Both targets share the instance,
            // so the two calls differ only by position: the first-match search returns targets[0]
            // while the per-stamp search has to skip it once the position falls outside its range.
            vbr_operation_binding binding;
            binding.kind = vbr_operation_kind::decode;
            binding.child_phase = vbr_operation_phase::mutate;
            binding.n_targets = 2;
            binding.targets[0] = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                instance_a, 0, 5, 0, 10);
            binding.targets[1] = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                instance_a, 0, 5, 0, 100);
            check(binding.find_covering_target(instance_a, 0, vbr_operation_class::ordinary_decode,
                                               append_bit) == &binding.targets[0],
                  "covering search did not return the first matching target");
            uint8_t index = 0xff;
            check(binding.find_covering_target_at(instance_a, 0, vbr_operation_class::ordinary_decode,
                                                  append_bit, 5, 5, &index) == &binding.targets[0],
                  "per-stamp search did not prefer the narrow target");
            check(index == 0, "per-stamp search reported the wrong narrow-target index");
            index = 0xff;
            check(binding.find_covering_target_at(instance_a, 0, vbr_operation_class::ordinary_decode,
                                                  append_bit, 5, 50, &index) == &binding.targets[1],
                  "per-stamp search ignored the position filter");
            check(index == 1, "per-stamp search reported the wrong wide-target index");
            index = 0xff;
            check(binding.find_covering_target_at(instance_a, 0, vbr_operation_class::ordinary_decode,
                                                  append_bit, 4, 5, &index) == nullptr,
                  "per-stamp search ignored the sequence filter");
            check(index == 0xff, "per-stamp search wrote an index for a miss");
            check(binding.find_covering_target(instance_b, 0, vbr_operation_class::ordinary_decode,
                                               append_bit) == nullptr,
                  "covering search matched an unrelated instance");
            std::puts("VBR operation PASS: target covering predicates");
        }

        // [3] Closed per-kind range enumeration.
        {
            llama_pos p0 = -1;
            llama_pos p1 = -1;
            vbr_normalize_edit_range(p0, p1);
            check(p0 == 0 && p1 == std::numeric_limits<llama_pos>::max(),
                  "normalize_edit_range did not clamp both sentinels");
            p0 = 0;
            p1 = -1;
            vbr_normalize_edit_range(p0, p1);
            check(p0 == 0 && p1 == std::numeric_limits<llama_pos>::max(),
                  "normalize_edit_range moved an already-canonical p0");
            p0 = -1;
            p1 = 5;
            vbr_normalize_edit_range(p0, p1);
            check(p0 == 0 && p1 == 5, "normalize_edit_range rewrote a concrete p1");
            p0 = 3;
            p1 = 9;
            vbr_normalize_edit_range(p0, p1);
            check(p0 == 3 && p1 == 9, "normalize_edit_range rewrote a fully concrete range");

            // The wildcard form is enumerated, not generic: exactly two kinds may carry it.
            for (size_t i = 0; i < KIND_COUNT; ++i) {
                const auto kind = static_cast<vbr_operation_kind>(i);
                const bool wildcard_allowed = kind == vbr_operation_kind::recovery ||
                                              kind == vbr_operation_kind::controller_retier;
                check(vbr_target_range_valid(kind, { -1, -1 }) == wildcard_allowed,
                      "{-1,-1} wildcard is allowed for the wrong kinds");
                // A half-wildcard is never a valid form.
                check(!vbr_target_range_valid(kind, { -1, 5 }), "{-1,5} was accepted as a range");
                check(!vbr_target_range_valid(kind, { 5, -1 }), "{5,-1} was accepted as a range");
                // The empty range is sequence_edit's alone.
                check(vbr_target_range_valid(kind, { 4, 4 }) ==
                          (kind == vbr_operation_kind::sequence_edit),
                      "empty range is allowed for the wrong kinds");
                // An inverted range is never valid, whatever the kind.
                check(!vbr_target_range_valid(kind, { 9, 4 }), "inverted range was accepted");
                check(vbr_target_range_valid(kind, { 4, 5 }), "ordinary range was rejected");
            }
            std::puts("VBR operation PASS: closed per-kind range enumeration");
        }

        // [4] Binding construction.
        {
            const auto instance = make_instance(33);
            const auto binding = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, 4, 2, 6,
                vbr_operation_class::state_api, instance, VBR_STREAM_ANY);
            check(binding.kind == vbr_operation_kind::sequence_edit, "binding lost its kind");
            check(binding.child_phase == vbr_operation_phase::mutate,
                  "binding did not carry the stated mutate child phase");
            check(binding.n_targets == 1, "binding did not create exactly one target");
            check(binding.seq_id() == 4 && binding.range().p0 == 2 && binding.range().p1 == 6,
                  "single-target accessors disagree with targets[0]");
            check(binding.targets[0].operation_class == vbr_operation_class::state_api,
                  "make_target lost the operation class");
            check(binding.targets[0].registrant_mask ==
                      vbr_operation_kind_registrants(vbr_operation_kind::sequence_edit),
                  "make_target did not derive the registrant mask from the kind");
            check(binding.targets[0].instance_id == instance, "make_target lost the instance");
            check(binding.targets[0].stream == VBR_STREAM_ANY, "make_target lost the stream");

            // An unarmed controller appends nothing; a full manifest refuses; refusal leaves the
            // target count alone in both cases.
            vbr_operation_binding appended = binding;
            check(!vbr_binding_add_instance_target(appended, vbr_operation_kind::sequence_edit,
                                                   vbr_operation_class::state_api,
                                                   vbr_controller_instance_id{}, 0, -1, 0, 10),
                  "an empty instance was appended as a target");
            check(appended.n_targets == 1, "refused append still bumped the target count");
            check(vbr_binding_add_instance_target(appended, vbr_operation_kind::sequence_edit,
                                                  vbr_operation_class::state_api,
                                                  instance, 0, -1, 0, 10),
                  "a set instance was not appended");
            check(appended.n_targets == 2, "successful append did not bump the target count");
            check(appended.targets[1].instance_id == instance,
                  "appended target lost its instance");

            vbr_operation_binding full;
            full.n_targets = vbr_operation_binding::MAX_TARGETS;
            check(!vbr_binding_add_instance_target(full, vbr_operation_kind::decode,
                                                   vbr_operation_class::ordinary_decode,
                                                   instance, 0, -1, 0, 10),
                  "a full manifest accepted another target");
            check(full.n_targets == vbr_operation_binding::MAX_TARGETS,
                  "refused append on a full manifest changed the target count");

            // Operation ids are opaque handles with value semantics.
            check(!vbr_operation_id{}, "a default operation id reports as set");
            check(vbr_operation_id{ 5 } == vbr_operation_id{ 5 }, "operation id equality is broken");
            check(vbr_operation_id{ 5 } != vbr_operation_id{ 6 }, "operation id inequality is broken");
            std::puts("VBR operation PASS: binding construction");
        }

        // [5] Mint validation matrix.
        {
            const auto instance = make_instance(44);

            // One rejection per documented rule.
            {
                auto bad = valid_binding();
                bad.kind = static_cast<vbr_operation_kind>(KIND_COUNT);
                expect_rejected(bad, "kind outside the closed enum");
            }
            {
                auto bad = valid_binding();
                bad.child_phase = static_cast<vbr_operation_phase>(PHASE_COUNT);
                expect_rejected(bad, "child phase outside the closed enum");
            }
            {
                auto bad = valid_binding();
                bad.n_targets = static_cast<uint8_t>(vbr_operation_binding::MAX_TARGETS + 1);
                expect_rejected(bad, "target count above MAX_TARGETS");
            }
            {
                auto bad = valid_binding();
                bad.child_phase = vbr_operation_phase::mutate;
                bad.n_targets = 0;
                expect_rejected(bad, "mutate-phase operation with no target");
            }
            {
                auto bad = valid_binding();
                bad.targets[0].operation_class = static_cast<vbr_operation_class>(CLASS_COUNT);
                expect_rejected(bad, "target class outside the closed enum");
            }
            {
                auto bad = valid_binding();
                bad.targets[0].child_phase = vbr_operation_phase::publish;
                expect_rejected(bad, "target with a non-mutate child phase");
            }
            {
                auto bad = valid_binding();
                bad.targets[0].registrant_mask = 0;
                expect_rejected(bad, "target with an empty registrant mask");
            }
            {
                // A mask bit that the kind's canonical set does not contain: the manifest must not
                // be able to authorize a registrant its kind has no business touching.
                auto bad = vbr_mutation_binding(vbr_operation_kind::sequence_edit, 1, 0, 10,
                                                vbr_operation_class::state_api, instance, 0);
                bad.targets[0].registrant_mask |=
                    vbr_registrant_bit(vbr_mutation_registrant::window_install);
                expect_rejected(bad, "registrant mask reaching outside the kind's set");
            }
            {
                auto bad = valid_binding();
                bad.targets[0].instance_id = vbr_controller_instance_id{};
                expect_rejected(bad, "non-recovery operation with an unset instance");
            }
            {
                auto bad = valid_binding();
                bad.targets[0].seq_id = -2;
                expect_rejected(bad, "sequence id below the declared wildcard");
            }
            {
                auto bad = valid_binding();
                bad.targets[0].seq_id = LLAMA_MAX_SEQ;
                expect_rejected(bad, "sequence id at LLAMA_MAX_SEQ");
            }
            {
                auto bad = valid_binding();
                bad.targets[0].range = { -1, -1 };
                expect_rejected(bad, "wildcard range on a range-bearing kind");
            }
            {
                auto bad = valid_binding();
                bad.targets[0].range = { 3, 3 };
                expect_rejected(bad, "empty range on a range-bearing kind");
            }
            {
                auto bad = valid_binding();
                bad.targets[0].range = { 10, 5 };
                expect_rejected(bad, "inverted range");
            }

            // Accepted: the boundary case just inside each rule.
            vbr_operation_binding accepted = valid_binding();
            const vbr_operation_id id = vbr_operation_registry_begin(accepted);
            check(bool(id), "a valid manifest was rejected");
            check(accepted.operation_id == id, "begin did not stamp the manifest's operation id");
            check(vbr_operation_registry_is_live(id), "a freshly minted operation is not live");
            check(vbr_operation_registry_has_capacity(), "registry reports no capacity when idle");

            // Re-begining a live manifest is refused and must not disturb the existing id.
            vbr_operation_binding &again = accepted;
            check(!vbr_operation_registry_begin(again), "a live binding was re-minted");
            check(again.operation_id == id, "refused re-mint overwrote the live operation id");

            // The registry retains the authenticated manifest, which is what stamps and extents
            // validate against instead of caller-supplied values.
            vbr_operation_binding retained;
            check(vbr_operation_registry_binding(id, retained), "live binding was not retrievable");
            check(retained.operation_id == id, "retained binding carries a different id");
            check(retained.kind == accepted.kind && retained.n_targets == accepted.n_targets,
                  "retained binding lost its kind or target count");
            check(retained.targets[0].seq_id == accepted.targets[0].seq_id &&
                      retained.targets[0].range.p0 == accepted.targets[0].range.p0 &&
                      retained.targets[0].range.p1 == accepted.targets[0].range.p1,
                  "retained binding lost its authenticated target");
            check(retained.targets[0].instance_id == accepted.targets[0].instance_id,
                  "retained binding lost its instance");

            // A zero id is never live and never retrievable.
            vbr_operation_binding out;
            check(!vbr_operation_registry_binding(vbr_operation_id{}, out),
                  "the empty operation id resolved to a binding");
            check(!vbr_operation_registry_is_live(vbr_operation_id{}),
                  "the empty operation id reported as live");
            check(!vbr_operation_registry_end(vbr_operation_id{}),
                  "the empty operation id was ended");

            // Lifecycle: end is a one-shot, and the id stops resolving afterwards.
            check(vbr_operation_registry_end(id), "a live operation refused to end");
            check(!vbr_operation_registry_is_live(id), "an ended operation is still live");
            check(!vbr_operation_registry_end(id), "an ended operation was ended twice");
            check(!vbr_operation_registry_binding(id, out),
                  "an ended operation still resolved to a binding");

            // Accepted: recovery's documented empty-instance exception, with the enumerated
            // wildcard range that only it and controller_retier may carry.
            vbr_operation_binding recovery = vbr_mutation_binding(
                vbr_operation_kind::recovery, -1, -1, -1,
                vbr_operation_class::controller, vbr_controller_instance_id{}, 0);
            const vbr_operation_id recovery_id = vbr_operation_registry_begin(recovery);
            check(bool(recovery_id), "recovery with an empty instance was rejected");
            vbr_operation_binding retained_recovery;
            check(vbr_operation_registry_binding(recovery_id, retained_recovery),
                  "recovery binding was not retrievable");
            check(!vbr_controller_instance_id_is_set(retained_recovery.targets[0].instance_id),
                  "recovery binding did not retain its wildcard instance");
            check(vbr_operation_registry_end(recovery_id), "recovery operation refused to end");

            std::puts("VBR operation PASS: mint validation matrix");
        }

        // [6] Quiescence probes.
        {
            const auto probe = make_instance(55);
            const auto other = make_instance(66);
            const auto wildcard_instance = vbr_controller_instance_id{};

            // Nothing is live here, so a probe over a real instance list is quiescent.
            check(vbr_operation_registry_quiescent_for(&probe, 1),
                  "registry was not quiescent with no live operations");

            vbr_operation_binding binding = vbr_mutation_binding(
                vbr_operation_kind::decode, 0, 0, 10,
                vbr_operation_class::ordinary_decode, probe, 0);
            const vbr_operation_id id = vbr_operation_registry_begin(binding);
            check(bool(id), "quiescence fixture could not be minted");

            check(!vbr_operation_registry_quiescent_for(&probe, 1),
                  "a live operation on the probed instance was reported as quiescent");
            check(vbr_operation_registry_quiescent_for(&other, 1),
                  "a live operation on a different instance blocked an unrelated probe");
            check(vbr_operation_registry_quiescent_for_except(&probe, 1, id),
                  "an allowed operation still blocked its own probe");
            check(!vbr_operation_registry_quiescent_for_except(&probe, 1, vbr_operation_id{}),
                  "an empty allow-list entry unblocked the probe");

            // The guards: an absent or empty instance list is not evidence of quiescence, so the
            // probe fails closed rather than reporting an idle registry.
            check(!vbr_operation_registry_quiescent_for(nullptr, 0),
                  "an empty instance list was treated as quiescent");
            check(!vbr_operation_registry_quiescent_for(nullptr, 1),
                  "a null instance array was treated as quiescent");
            check(vbr_operation_registry_end(id), "quiescence fixture refused to end");
            check(vbr_operation_registry_quiescent_for(&probe, 1),
                  "registry stayed busy after the last operation ended");

            // An unset-instance target is an any-controller wildcard, so it blocks every probe.
            vbr_operation_binding wild = vbr_mutation_binding(
                vbr_operation_kind::recovery, -1, -1, -1,
                vbr_operation_class::controller, wildcard_instance, 0);
            const vbr_operation_id wild_id = vbr_operation_registry_begin(wild);
            check(bool(wild_id), "wildcard-instance fixture could not be minted");
            check(!vbr_operation_registry_quiescent_for(&probe, 1),
                  "an any-controller operation did not block the probe");
            check(!vbr_operation_registry_quiescent_for(&other, 1),
                  "an any-controller operation did not block an unrelated instance");
            check(vbr_operation_registry_quiescent_for_except(&probe, 1, wild_id),
                  "an allowed any-controller operation still blocked its own probe");
            check(vbr_operation_registry_end(wild_id), "wildcard fixture refused to end");
            std::puts("VBR operation PASS: quiescence probes");
        }

        // [7] Recovery ring. Its records are process-global, so this section is self-contained:
        // every reservation it takes is either released or explicitly minted.
        {
            const auto owner = make_instance(77);
            const auto someone_else = make_instance(88);

            // A binding that was never minted owns no operation id, so it cannot reserve.
            const vbr_operation_binding unminted = valid_binding(99);
            check(vbr_recovery_reserve(unminted, owner) < 0,
                  "an unminted binding reserved a recovery record");
            check(vbr_recovery_reserve(vbr_operation_id{}, owner) < 0,
                  "the empty operation id reserved a recovery record");

            vbr_operation_binding binding = valid_binding(99);
            const vbr_operation_id id = vbr_operation_registry_begin(binding);
            check(bool(id), "recovery fixture could not be minted");

            // Reservation by id resolves the live manifest itself.
            const int32_t index = vbr_recovery_reserve(id, owner);
            check(index >= 0, "a live operation could not reserve a recovery record");
            vbr_failed_operation_record record;
            check(vbr_recovery_get_record(index, record), "the reserved record was not readable");
            check(record.state == vbr_recovery_state::reserved, "a fresh record is not reserved");
            check(record.binding.operation_id == id, "the record bound the wrong operation");
            check(record.owner_instance == owner, "the record lost its owner instance");
            check(record.owner_instance != someone_else, "the record adopted a foreign owner");

            // Ownership is enforced on every transition, and it is keyed on the operation id
            // rather than on the instance: a second live operation must not be able to release,
            // record, or set the source token on this one's reservation.
            vbr_operation_binding foreign_binding = valid_binding(202);
            const vbr_operation_id foreign_id = vbr_operation_registry_begin(foreign_binding);
            check(bool(foreign_id), "the foreign-operation fixture could not be minted");
            check(!vbr_recovery_release_unused(index, foreign_id),
                  "a foreign operation released someone else's reservation");
            check(!vbr_recovery_record_failure(index, foreign_id,
                                               vbr_operation_phase::mutate,
                                               vbr_recovery_failure_site::metadata_mutation, false),
                  "a foreign operation recorded a failure on someone else's reservation");
            check(!vbr_recovery_set_source_token(index, foreign_id, 0, 0, { 0, 1 }, { 1 }),
                  "a foreign operation wrote the source token");
            check(vbr_recovery_get_record(index, record),
                  "the rejected foreign writes disturbed the reservation");
            check(record.state == vbr_recovery_state::reserved && record.binding.operation_id == id,
                  "a rejected foreign write mutated the reservation");
            check(vbr_operation_registry_end(foreign_id), "the foreign fixture refused to end");

            // A record must be `recorded` before a capability can be minted from it.
            check(!vbr_recovery_mint(index), "a capability was minted from an unrecorded reservation");

            check(vbr_recovery_set_source_token(index, id, 0, 0, { 2, 9 }, { 3, 4 }),
                  "the owning operation could not set its source token");
            check(vbr_recovery_record_failure(index, id, vbr_operation_phase::publish,
                                              vbr_recovery_failure_site::deferred_byte_copy, true),
                  "the owning operation could not record its failure");
            check(!vbr_recovery_record_failure(index, id, vbr_operation_phase::publish,
                                               vbr_recovery_failure_site::publication, false),
                  "an already-recorded failure was recorded twice");
            check(vbr_recovery_get_record(index, record), "the recorded entry was not readable");
            check(record.state == vbr_recovery_state::recorded, "the failure did not record");
            check(record.phase_reached == vbr_operation_phase::publish,
                  "the recorded phase was not retained");
            check(record.failure_site == vbr_recovery_failure_site::deferred_byte_copy,
                  "the recorded failure site was not retained");
            check(record.dest_bytes_observable, "the observability flag was not retained");
            check(record.src_stream == 0 && record.dst_stream == 0 &&
                      record.src_range.p0 == 2 && record.src_range.p1 == 9 &&
                      record.src_page_gens == std::vector<uint32_t>({ 3, 4 }),
                  "the source-stability token was not retained");
            check(!vbr_recovery_release_unused(index, id),
                  "a recorded entry was released as if it were unused");

            // A non-committed close auto-records any still-reserved entry for that operation, so a
            // non-committed close can never orphan a reservation with no resolution path.
            const int32_t auto_index = vbr_recovery_reserve(id, owner);
            check(auto_index >= 0, "the second reservation failed");
            check(auto_index != index, "the ring handed out the same slot twice");
            check(vbr_operation_registry_close(id, vbr_operation_outcome::aborted),
                  "the operation refused to close as aborted");
            check(vbr_recovery_get_record(auto_index, record),
                  "the auto-recorded entry was not readable");
            check(record.state == vbr_recovery_state::recorded,
                  "an aborted close left the reservation reserved");
            check(record.failure_site == vbr_recovery_failure_site::exception_unwind,
                  "the auto-record did not name the unwind site");
            check(record.phase_reached == record.binding.child_phase,
                  "the auto-record did not cite the binding's child phase");

            // A committed close leaves the reservation alone; the owner releases it as unused.
            vbr_operation_binding second = valid_binding(101);
            const vbr_operation_id second_id = vbr_operation_registry_begin(second);
            check(bool(second_id), "the committed-close fixture could not be minted");
            const int32_t kept = vbr_recovery_reserve(second_id, owner);
            check(kept >= 0, "the committed-close fixture could not reserve");
            check(vbr_operation_registry_close(second_id, vbr_operation_outcome::committed),
                  "the fixture refused to close as committed");
            check(vbr_recovery_get_record(kept, record),
                  "a committed close auto-recorded the reservation");
            check(record.state == vbr_recovery_state::reserved,
                  "a committed close changed the reservation state");
            check(vbr_recovery_release_unused(kept, second_id),
                  "the owning operation could not release its unused reservation");
            check(!vbr_recovery_get_record(kept, record),
                  "a released slot is still readable");

            // Minting is single-use and only from a recorded entry; destroying an unresolved
            // capability fail-closes the record into quarantine, so this is the last check here.
            const auto capability = vbr_recovery_mint(index);
            check(bool(capability), "a recorded failure could not mint a capability");
            check(!vbr_recovery_mint(index),
                  "the same recorded failure minted a second capability");
            check(vbr_recovery_get_record(index, record), "the minted entry vanished");
            check(record.state == vbr_recovery_state::capability_minted,
                  "minting did not advance the record state");

            check(!vbr_recovery_get_record(-1, record), "a negative record index was readable");
            check(!vbr_recovery_get_record(1 << 20, record), "an out-of-range index was readable");
            std::puts("VBR operation PASS: recovery ring transitions");
        }

        std::puts("VBR operation tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "VBR operation FAIL: %s\n", error.what());
        return 1;
    }
}
