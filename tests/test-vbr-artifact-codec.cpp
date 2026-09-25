// Tests for the artifact codec's admission gates, now that the whole artifact strongly-connected
// component (llama-vbr-artifact.cpp, -generation.cpp, -occupied-replacement.cpp, -catalog.cpp,
// -validate.cpp) is in the tree together. See BUUN_MERGE_PLAN §10.37-§10.38: these five TUs are
// mutually link-dependent, so they could only land as one commit.
//
// Hand-written: upstream has no test for these files. The scope here is the *admission* half of the
// codec plus the two helpers other layers call, because a full prepare/encode/decode round trip
// needs a complete unit-artifact package (shards with real payload sources, a generation record, a
// token block) and that construction is its own project. What is pinned:
//
//   - the status vocabulary, including that the _count sentinel maps to a name rather than crashing;
//   - vbr_artifact_accounting_category(), the closed wire-role -> C-category mapping the format
//     validation and the transactional catalog both share;
//   - vbr_artifact_logical_unit_digest(), the canonical per-unit lineage key: deterministic, nonzero
//     and sensitive to every field that identifies a unit;
//   - vbr_artifact_validate_portable_accounting(), the shared row validator;
//   - vbr_artifact_prepare()'s first three admission layers, exercised with deliberately broken
//     packages so each gate is seen refusing rather than assumed to exist.
//
// The full round trip is NOT covered and is listed as a gap in BUUN_MERGE_PLAN §10.38.
//
// Self-reporting (status from main plus explicit PASS lines), so NDEBUG cannot hollow it out and it
// needs no /UNDEBUG.

#include "llama-vbr-artifact.h"
#include "llama-vbr-artifact-catalog.h"

#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static bool nonzero_digest(const std::array<uint8_t, 32> & digest) {
    for (const uint8_t byte : digest) {
        if (byte != 0) {
            return true;
        }
    }
    return false;
}

static llama_cache_acct_shard_topology make_topology(const char * device) {
    llama_cache_acct_shard_topology out;
    if (!llama_cache_acct_build_shard_topology({ device }, 0, 0, nullptr, out)) {
        throw std::runtime_error("the topology construction door refused a device set");
    }
    return out;
}

static vbr_artifact_portable_accounting_row row(vbr_artifact_accounting_role role, uint64_t bytes) {
    vbr_artifact_portable_accounting_row result;
    result.role           = role;
    result.domain         = vbr_artifact_payload_storage_domain();
    result.logical_bytes  = bytes;
    result.resident_bytes = bytes;
    return result;
}

int main() {
    try {
        // [1] The status vocabulary is closed and named, and the sentinel has a name too.
        {
            std::set<std::string> seen;
            for (int i = 0; i < static_cast<int>(vbr_artifact_status::_count); ++i) {
                const char * name = vbr_artifact_status_name(static_cast<vbr_artifact_status>(i));
                check(name != nullptr && name[0] != '\0', "an artifact status has no name");
                check(seen.insert(name).second, "two artifact statuses share a name");
            }
            check(seen.size() == static_cast<size_t>(vbr_artifact_status::_count),
                  "the status name table is smaller than the enum");
            check(vbr_artifact_status_name(vbr_artifact_status::_count) != nullptr,
                  "the _count sentinel has no name, so a wild enum value would dereference null");
            check(vbr_artifact_status_name(vbr_artifact_status::ok) ==
                      std::string("vbr_artifact_ok"),
                  "the success status is not named as expected");
            std::puts("VBR codec PASS: status vocabulary");
        }

        // [2] The closed wire-role -> C-category mapping, shared by format validation and the
        // transactional catalog. It must stay total over the role enum and must not invent
        // categories outside the ledger's vocabulary.
        {
            for (int i = 0; i < static_cast<int>(vbr_artifact_accounting_role::_count); ++i) {
                const auto role = static_cast<vbr_artifact_accounting_role>(i);
                const auto category = vbr_artifact_accounting_category(role);
                check(static_cast<int>(category) >= 0, "a role mapped to a negative category");
            }
            // Distinct roles may map to distinct categories; assert at least that the map is stable
            // across calls rather than recomputed per call.
            const auto first = vbr_artifact_accounting_category(
                vbr_artifact_accounting_role::unit_payload);
            const auto again = vbr_artifact_accounting_category(
                vbr_artifact_accounting_role::unit_payload);
            check(first == again, "the role->category mapping is not stable");
            check(vbr_artifact_accounting_category(vbr_artifact_accounting_role::unit_payload) !=
                      vbr_artifact_accounting_category(
                          vbr_artifact_accounting_role::clean_stash_payload),
                  "two payload roles collapsed onto one category");
            std::puts("VBR codec PASS: wire-role to category mapping");
        }

        // [3] The canonical per-unit lineage key. The recipe hashes the domain label, the lineage
        // uuid, logical_unit_id and repr_gen - and deliberately nothing else: child_id and
        // current_type are placement and type details, not unit identity, and the test pins that
        // exclusion so a future field cannot silently enter the key.
        {
            vbr_artifact_unit_descriptor descriptor;
            descriptor.child_id        = 3;
            descriptor.logical_unit_id = 7;
            descriptor.lineage_uuid    = { 0x1122334455667788ull, 0x99aabbccddeeff00ull };
            descriptor.repr_gen        = 11;
            descriptor.current_type    = 1;
            vbr_artifact_unit_descriptor changed_desc;
            const auto base = vbr_artifact_logical_unit_digest(descriptor);
            check(nonzero_digest(base), "the unit digest is all zero for a populated descriptor");
            check(vbr_artifact_logical_unit_digest(descriptor) == base,
                  "the unit digest is not deterministic");

            const auto differs = [&](const vbr_artifact_unit_descriptor & changed,
                                     const char * what) {
                check(vbr_artifact_logical_unit_digest(changed) != base, what);
            };
            const auto same = [&](const vbr_artifact_unit_descriptor & changed,
                                  const char * what) {
                check(vbr_artifact_logical_unit_digest(changed) == base, what);
            };

            changed_desc = descriptor;
            changed_desc.logical_unit_id = 8;
            differs(changed_desc, "the digest ignored logical_unit_id");
            changed_desc = descriptor;
            changed_desc.repr_gen = 12;
            differs(changed_desc, "the digest ignored repr_gen");
            changed_desc = descriptor;
            changed_desc.lineage_uuid.hi = 0x1122334455667789ull;
            differs(changed_desc, "the digest ignored the lineage uuid high word");
            changed_desc = descriptor;
            changed_desc.lineage_uuid.lo = 0x99aabbccddeeff01ull;
            differs(changed_desc, "the digest ignored the lineage uuid low word");
            // Word order in the lineage uuid is serialized, not a set.
            changed_desc = descriptor;
            std::swap(changed_desc.lineage_uuid.hi, changed_desc.lineage_uuid.lo);
            differs(changed_desc, "the digest treated the lineage uuid words as unordered");

            // The deliberate exclusions: these identify where a unit lives and what type it is,
            // not which unit it is.
            changed_desc = descriptor;
            changed_desc.child_id = 99;
            same(changed_desc, "the digest hashed child_id");
            changed_desc = descriptor;
            changed_desc.current_type = 2;
            same(changed_desc, "the digest hashed current_type");

            // There is no unset-uuid validation at this layer: the zero words are hashed as-is,
            // so an anonymous descriptor still produces a digest - just a different one.
            vbr_artifact_unit_descriptor anonymous;
            anonymous.logical_unit_id = 7;
            anonymous.repr_gen        = 11;
            const auto anonymous_digest = vbr_artifact_logical_unit_digest(anonymous);
            check(nonzero_digest(anonymous_digest),
                  "a descriptor with an unset lineage uuid produced an all-zero digest");
            check(anonymous_digest != base,
                  "an unset lineage uuid did not change the digest");
            std::puts("VBR codec PASS: canonical unit digest");
        }

        // [4] The shared portable-accounting validator.
        {
            const auto top = make_topology("cpu");
            const std::vector<vbr_artifact_portable_topology> topologies { top };
            check(vbr_artifact_validate_portable_accounting(
                      topologies, { row(vbr_artifact_accounting_role::unit_payload, 4096) }),
                  "a well-formed row was refused");
            check(vbr_artifact_validate_portable_accounting(
                      topologies,
                      { row(vbr_artifact_accounting_role::unit_payload, 4096),
                        row(vbr_artifact_accounting_role::descriptor_metadata, 512) }),
                  "two rows in different roles were refused");

            auto orphan = row(vbr_artifact_accounting_role::unit_payload, 4096);
            orphan.domain.topology_index = 5;
            check(!vbr_artifact_validate_portable_accounting(topologies, { orphan }),
                  "a row addressing an undeclared topology was accepted");

            // An artifact with no accounting rows at all is still refused: the envelope must
            // account for what it stores.
            check(!vbr_artifact_validate_portable_accounting(topologies, {}),
                  "an empty row set was accepted");
            check(!vbr_artifact_validate_portable_accounting({}, {}),
                  "an empty topology table was accepted");
            std::puts("VBR codec PASS: portable accounting validator");
        }

        // [5] prepare()'s admission gates, each seen refusing a deliberately broken package.
        {
            const auto top = make_topology("cpu");
            const auto valid_topology = [&](std::vector<vbr_artifact_portable_topology> tops) {
                // prepare() recomputes the digest, so a door-built topology needs no pre-hashing.
                return tops;
            };

            // An unpopulated envelope has neither topologies nor unit blobs.
            {
                vbr_artifact_package package;
                check(vbr_artifact_prepare(package) == vbr_artifact_status::invalid_argument,
                      "an empty package was not refused as invalid argument");
            }
            // Topologies without unit blobs is still an invalid argument: the envelope is not empty
            // but it carries nothing.
            {
                vbr_artifact_package package;
                package.topologies = valid_topology({ top });
                check(vbr_artifact_prepare(package) == vbr_artifact_status::invalid_argument,
                      "a package with topologies but no units was not refused");
            }
            // A wrong format version is refused first. Note the first gate lumps version, flags,
            // topologies and unit blobs into one invalid_argument, so unsupported_version is not
            // reachable from prepare - it belongs to the decode path, which this test does not
            // drive (a round trip needs a complete unit-artifact package).
            {
                vbr_artifact_package package;
                package.version = VBR_UNIT_ARTIFACT_FORMAT_VERSION + 1;
                package.topologies = valid_topology({ top });
                package.unit_blobs.resize(1);
                check(vbr_artifact_prepare(package) == vbr_artifact_status::invalid_argument,
                      "an unsupported format version was not refused");
            }
            // A unit blob with no shards is malformed rather than empty.
            {
                vbr_artifact_package package;
                package.topologies = valid_topology({ top });
                package.unit_blobs.resize(1);
                check(vbr_artifact_prepare(package) == vbr_artifact_status::malformed,
                      "a unit blob without shards was not refused as malformed");
            }
            // A topology with a wrong wire version poisons every unit, so the topology gate fires
            // even when the blobs are the malformed ones.
            {
                vbr_artifact_package package;
                auto broken = top;
                broken.version = LLAMA_CACHE_ACCT_TOPOLOGY_VERSION + 1;
                package.topologies = valid_topology({ broken });
                package.unit_blobs.resize(1);
                check(vbr_artifact_prepare(package) == vbr_artifact_status::topology_mismatch,
                      "a topology with a wrong wire version was not refused");
            }
            std::puts("VBR codec PASS: prepare admission gates");
        }

        std::puts("VBR codec tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "VBR codec FAIL: %s\n", error.what());
        return 1;
    }
}
