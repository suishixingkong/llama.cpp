// Tests for the header-only half of llama-vbr-artifact.h - the unit-artifact format vocabulary.
//
// Hand-written, and deliberately narrow. Two reasons:
//
//   1. The codec itself (vbr_artifact_prepare / encode / decode) lives in llama-vbr-artifact.cpp,
//      which does NOT link yet: it calls vbr_generation_production_covered_set(), declared in the
//      already-landed llama-vbr-generation-types.h but defined only in llama-vbr-generation.cpp -
//      which cannot land until artifact-validate, occupied-replacement and artifact-capture exist.
//      That is the third closure criterion found in BUUN_MERGE_PLAN §10.35: include closure is not
//      enough, LINK closure is a separate measurement. So this commit lands the header alone, inert
//      and with nothing resolving it, and the codec test arrives with its .cpp.
//
//   2. Upstream has no test for this file at all.
//
// What is covered is the format vocabulary that every later reader of the wire format depends on:
// the version gates, the default-constructed invariants of the envelope structs, the portable-domain
// value type, and the decode limits. The portable domain's ordering is checked against a
// std::tuple oracle rather than against the implementation's nested comparisons, so the ordering is
// pinned by an independent rule: lexicographic on (residency, kind, topology_index, device_ordinal).
//
// Self-reporting (status from main plus explicit PASS lines), so NDEBUG cannot hollow it out and it
// needs no /UNDEBUG.

#include "llama-vbr-artifact.h"

#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

using domain_key = std::tuple<llama_cache_acct_residency, llama_cache_acct_domain_kind,
                               uint32_t, uint16_t>;

static domain_key key_of(const vbr_artifact_portable_domain & domain) {
    return { domain.residency, domain.kind, domain.topology_index, domain.device_ordinal };
}

int main() {
    try {
        // [1] Format version gates. The two feature gates are thresholds on the wire version, and
        // they are what let an older envelope be read without silently inventing a field.
        {
            check(VBR_UNIT_ARTIFACT_FORMAT_VERSION_MIN == 1 &&
                      VBR_UNIT_ARTIFACT_FORMAT_VERSION == 3,
                  "the supported format version range changed");
            check(VBR_UNIT_ARTIFACT_FORMAT_VERSION_REFERENCE_PLACEMENT == 2 &&
                      VBR_UNIT_ARTIFACT_FORMAT_VERSION_MEANSUB_REFERENCE == 3,
                  "a feature gate no longer matches its introducing version");
            check(VBR_UNIT_ARTIFACT_FORMAT_VERSION >=
                      VBR_UNIT_ARTIFACT_FORMAT_VERSION_REFERENCE_PLACEMENT &&
                      VBR_UNIT_ARTIFACT_FORMAT_VERSION >=
                          VBR_UNIT_ARTIFACT_FORMAT_VERSION_MEANSUB_REFERENCE,
                  "the current format version predates a feature it must carry");
            // A version below the minimum is not a format at all, so both gates must read false
            // there rather than being decided by the threshold comparison alone.
            check(!artifact_has_reference_placement(0),
                  "version 0 was granted reference placement");
            check(!artifact_has_meansub_reference(0),
                  "version 0 was granted a mean-sub reference");
            for (uint32_t v = 1; v <= VBR_UNIT_ARTIFACT_FORMAT_VERSION; ++v) {
                check(artifact_has_reference_placement(v) ==
                          (v >= VBR_UNIT_ARTIFACT_FORMAT_VERSION_REFERENCE_PLACEMENT),
                      "the reference-placement gate is not the documented threshold");
                check(artifact_has_meansub_reference(v) ==
                          (v >= VBR_UNIT_ARTIFACT_FORMAT_VERSION_MEANSUB_REFERENCE),
                      "the mean-sub-reference gate is not the documented threshold");
            }
            // Beyond the current version the gates stay open: a future writer must not be able to
            // smuggle a field past a reader that already understands the newer version.
            check(artifact_has_reference_placement(VBR_UNIT_ARTIFACT_FORMAT_VERSION + 1),
                  "the reference-placement gate closed past the current version");
            check(artifact_has_meansub_reference(VBR_UNIT_ARTIFACT_FORMAT_VERSION + 1),
                  "the mean-sub-reference gate closed past the current version");
            check(VBR_ARTIFACT_TOKEN_BLOCK_CODEC_VERSION == 1,
                  "the token-block codec version changed");
            std::puts("VBR artifact PASS: format version gates");
        }

        // [2] Default-constructed envelope invariants. A package that was never populated must be
        // distinguishable from one that was, or an empty envelope would decode as a valid one.
        {
            vbr_artifact_package package;
            check(package.version == VBR_UNIT_ARTIFACT_FORMAT_VERSION,
                  "a fresh package does not start at the current format version");
            check(package.manifest.version == VBR_UNIT_ARTIFACT_FORMAT_VERSION,
                  "a fresh manifest does not start at the current format version");
            check(package.flags == 0 && package.topologies.empty() &&
                      package.unit_blobs.empty() && package.companions.empty(),
                  "a fresh package carries content");
            check(package.manifest.identity.next_position == -1,
                  "a fresh identity block does not use the unknown frontier");
            check(package.manifest.identity.sequence_epoch == 0 &&
                      package.manifest.identity.token_count == 0,
                  "a fresh identity block is not zeroed");
            check(package.manifest.consistency.kind ==
                      vbr_artifact_consistency_kind::capture_exact,
                  "a fresh manifest does not default to exact-capture consistency");

            vbr_artifact_token_block block;
            check(block.codec_version == VBR_ARTIFACT_TOKEN_BLOCK_CODEC_VERSION,
                  "a fresh token block does not start at the codec version");
            check(block.tokens.empty(), "a fresh token block carries tokens");

            // Placement sentinels: an unset field must be the "absent" value, not a plausible index.
            vbr_artifact_cell_placement cell;
            check(cell.physical_cell == UINT32_MAX && cell.logical_position == -1,
                  "a fresh cell placement uses a plausible index instead of a sentinel");
            vbr_artifact_stream_placement stream;
            check(stream.child_id == UINT32_MAX && stream.stream_index == UINT32_MAX &&
                      stream.source_sequence == -1 && stream.computation_frontier == -1,
                  "a fresh stream placement uses plausible indices instead of sentinels");
            std::puts("VBR artifact PASS: default envelope invariants");
        }

        // [3] Portable domain: equality and the canonical ordering, against a tuple oracle.
        {
            const auto host = vbr_artifact_payload_storage_domain();
            check(host.residency == llama_cache_acct_residency::pageable_host,
                  "payload storage is not pageable host");
            check(host.kind == llama_cache_acct_domain_kind::not_applicable,
                  "payload storage carries a domain kind it should not");
            check(host.topology_index == UINT32_MAX && host.device_ordinal == UINT16_MAX,
                  "payload storage is bound to a concrete topology");

            vbr_artifact_portable_domain a {};
            vbr_artifact_portable_domain b = a;
            check(a == b && !(a != b), "a domain does not equal a copy of itself");
            b.device_ordinal = 1;
            check(a != b && !(a == b), "a single differing field did not break equality");

            // The oracle: lexicographic on (residency, kind, topology_index, device_ordinal). Every
            // pair drawn from a spread of values must agree with vbr_artifact_portable_domain_less,
            // and the comparator must be a strict weak ordering (irreflexive, antisymmetric).
            std::vector<vbr_artifact_portable_domain> sample;
            for (int residency = 0; residency < 4; ++residency) {
                for (int kind = 0; kind < 3; ++kind) {
                    for (uint32_t topology : { 0u, 1u, UINT32_MAX }) {
                        for (uint16_t device : { uint16_t(0), uint16_t(7), UINT16_MAX }) {
                            sample.push_back({
                                static_cast<llama_cache_acct_residency>(residency),
                                static_cast<llama_cache_acct_domain_kind>(kind),
                                topology, device });
                        }
                    }
                }
            }
            for (const auto & lhs : sample) {
                check(!vbr_artifact_portable_domain_less(lhs, lhs),
                      "the domain ordering is not irreflexive");
                for (const auto & rhs : sample) {
                    const bool got = vbr_artifact_portable_domain_less(lhs, rhs);
                    const bool want = key_of(lhs) < key_of(rhs);
                    check(got == want, "the domain ordering disagrees with the tuple oracle");
                    if (lhs != rhs) {
                        check(got != vbr_artifact_portable_domain_less(rhs, lhs),
                              "the domain ordering is not antisymmetric");
                    }
                }
            }
            // The ordering is the one std::set would derive from it, which is what the canonical
            // serialisation relies on for a stable row order.
            std::set<domain_key> seen;
            for (const auto & domain : sample) {
                seen.insert(key_of(domain));
            }
            check(seen.size() == sample.size(),
                  "the sample contains duplicate domains, so the sweep proves less than it claims");
            std::puts("VBR artifact PASS: portable domain ordering against the oracle");
        }

        // [4] Decode limits and byte-source validity. The limits are the caller's preflight bound;
        // a source with bytes but no reader is an unusable source.
        {
            vbr_artifact_decode_limits limits;
            check(limits.max_total_bytes == 0,
                  "the caller-preflighted byte bound is not required to be set explicitly");
            check(limits.max_topologies > 0 && limits.max_unit_blobs > 0 &&
                      limits.max_companions > 0 && limits.max_controllers > 0 &&
                      limits.max_stream_placements > 0 && limits.max_token_ids > 0 &&
                      limits.max_string_bytes > 0,
                  "a decode limit defaults to zero, which would reject every envelope");

            vbr_artifact_byte_source empty_source;
            check(empty_source.valid(), "a zero-sized source without a reader is not valid");
            vbr_artifact_byte_source sized_but_unreadable;
            sized_but_unreadable.size = 64;
            check(!sized_but_unreadable.valid(),
                  "a source with bytes but no reader was accepted");
            vbr_artifact_byte_source readable;
            readable.size = 64;
            readable.read = [](const void *, uint64_t, uint8_t *, size_t) noexcept {
                return true;
            };
            check(readable.valid(), "a source with bytes and a reader was rejected");

            // A consumer without a finish callback is not a consumer: the decoder promises to call
            // finish exactly once on every attempt, so an absent finish would break that contract.
            vbr_artifact_payload_consumer consumer;
            check(consumer.consume == nullptr && consumer.finish == nullptr,
                  "a default payload consumer carries callbacks");
            std::puts("VBR artifact PASS: decode limits and byte-source validity");
        }

        std::puts("VBR artifact tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "VBR artifact FAIL: %s\n", error.what());
        return 1;
    }
}
