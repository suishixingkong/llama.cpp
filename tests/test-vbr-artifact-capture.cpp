// Tests for llama-vbr-artifact-capture.h/.cpp - the segment chain and its Merkle range proofs.
//
// Hand-written: upstream has no test for this file, and it is the first stage-2.6 file whose .cpp is
// both link-self-contained and free of consumer types (see BUUN_MERGE_PLAN §10.35).
//
// The oracle here is adversarial rather than arithmetic. Nothing in this test recomputes a hash or
// re-derives the Merkle tree; instead the tree is treated as a black box that must satisfy:
//
//   - determinism: identical bytes seal to an identical root, and a single differing byte does not;
//   - soundness: a proof verifies against the bytes it was made from and FAILS against any other
//     bytes of the same length, which is the property the whole proof exists to provide;
//   - fail-closed admission: a malformed range list (empty, unsorted, overlapping, out of extent)
//     never produces a proof;
//   - non-widening delegation: restricting a proof to ranges the parent did not authorise is
//     refused, and never silently grants authority.
//
// That last one is asserted as "status != restricted" rather than as one specific enum value, so the
// test pins the security property (authority cannot widen) instead of a particular failure spelling.
//
// Self-reporting (status from main plus explicit PASS lines), so NDEBUG cannot hollow it out and it
// needs no /UNDEBUG.

#include "llama-vbr-artifact-capture.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

// A sealed digest must be nonzero; the header's own vbr_digest_nonzero() lives in
// llama-vbr-identity-digest.h, which this test does not need to depend on.
static bool nonzero(const std::array<uint8_t, 32> & digest) {
    for (const uint8_t byte : digest) {
        if (byte != 0) {
            return true;
        }
    }
    return false;
}

// Range records are plain structs with no generated equality, so compare them field by field.
static bool same_ranges(const std::vector<vbr_capture_authenticated_range> & lhs,
                        const std::vector<vbr_capture_authenticated_range> & rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].offset != rhs[i].offset || lhs[i].size != rhs[i].size) {
            return false;
        }
    }
    return true;
}

// Deterministic pseudo-random content, so a failure is reproducible from the seed alone.
static std::vector<uint8_t> make_bytes(size_t size, uint32_t seed) {
    std::vector<uint8_t> bytes(size);
    uint32_t state = seed ? seed : 1u;
    for (size_t i = 0; i < size; ++i) {
        state = state * 1664525u + 1013904223u;
        bytes[i] = static_cast<uint8_t>(state >> 24);
    }
    return bytes;
}

// One chain seeded with `bytes`, chunked and appended in pieces so the chain has to span segments.
static bool fill(artifact_segment_chain & chain, const std::vector<uint8_t> & bytes,
                 size_t piece, uint32_t chunk_bytes, uint32_t max_chunks) {
    chain = artifact_segment_chain(chunk_bytes, max_chunks);
    for (size_t at = 0; at < bytes.size(); at += piece) {
        const size_t n = bytes.size() - at < piece ? bytes.size() - at : piece;
        if (!chain.append(bytes.data() + at, n)) {
            return false;
        }
    }
    return true;
}

static bool sealed(artifact_segment_chain & chain, vbr_capture_range_tree & tree) {
    return vbr_capture_range_seal(chain, uint64_t(64)*1024*1024, tree);
}

int main() {
    try {
        // Sealing only happens for a chain built with the fixed wire chunk size; the constructor
        // refuses any other value, and authenticated() then reports false.
        constexpr uint32_t CHUNK = VBR_CAPTURE_RANGE_CHUNK_BYTES;
        constexpr uint32_t MAX_CHUNKS = 16;
        constexpr size_t PIECE = 32 * 1024;
        constexpr size_t SIZE = 200 * 1024;   // four chunks, so tree and proof are non-trivial

        // [1] Segment chain: append, read back across segment boundaries, expose a source.
        {
            const auto bytes = make_bytes(SIZE, 0x1234u);
            artifact_segment_chain chain(CHUNK, MAX_CHUNKS);
            check(chain.size() == 0, "a fresh chain is not empty");
            check(chain.authenticated(), "an authenticated chain does not report authentication");
            
            for (size_t at = 0; at < bytes.size(); at += PIECE) {
                const size_t n = bytes.size() - at < PIECE ? bytes.size() - at : PIECE;
                check(chain.append(bytes.data() + at, n), "append refused a piece");
            }
            check(chain.size() == bytes.size(), "the chain lost track of its byte count");
            check(chain.segment_count() > 1,
                  "appending in pieces did not span segments, so the read path is untested");

            std::vector<uint8_t> reread(bytes.size(), 0);
            check(chain.read(0, reread.data(), reread.size()), "read across segments failed");
            check(reread == bytes, "the chain did not return the bytes it was given");

            // A read that straddles a boundary, and one at the very last byte.
            std::vector<uint8_t> middle(200, 0);
            check(chain.read(500, middle.data(), middle.size()), "a mid-chain read failed");
            check(std::memcmp(middle.data(), bytes.data() + 500, middle.size()) == 0,
                  "a mid-chain read returned the wrong bytes");
            uint8_t last = 0;
            check(chain.read(bytes.size() - 1, &last, 1), "a read at the last byte failed");
            check(last == bytes.back(), "the last byte did not round-trip");
            check(!chain.read(bytes.size(), &last, 1), "a read past the end succeeded");

            const auto source = chain.source();
            check(source.valid(), "the chain exposed an invalid byte source");
            check(source.size == bytes.size(), "the exposed source has the wrong size");
            std::puts("VBR capture PASS: segment chain append and read");
        }

        // [2] Sealing: the root is deterministic in the content and sensitive to one byte.
        {
            const auto bytes = make_bytes(SIZE, 0x5678u);
            artifact_segment_chain chain;
            check(fill(chain, bytes, PIECE, CHUNK, MAX_CHUNKS), "the chain could not be filled");
            vbr_capture_range_tree tree;
            check(sealed(chain, tree), "sealing a complete chain failed");
            check(bool(tree), "a sealed tree reports as unset");
            check(tree.total_bytes() == bytes.size(), "the tree bound the wrong byte count");
            check(tree.chunk_bytes() == CHUNK, "the tree bound the wrong chunk size");
            check(tree.metadata_bytes() > 0, "a sealed tree reports no metadata");
            check(nonzero(tree.root()), "a sealed tree has an all-zero root");

            // Same bytes, same chunking => same root. A second chain exercises the whole path again
            // rather than re-reading one object's field.
            artifact_segment_chain again;
            check(fill(again, bytes, PIECE, CHUNK, MAX_CHUNKS), "the duplicate chain could not be filled");
            vbr_capture_range_tree again_tree;
            check(sealed(again, again_tree), "sealing the duplicate chain failed");
            check(again_tree.root() == tree.root(),
                  "identical bytes sealed to different roots");

            // One differing byte must change the root, or the tree would not authenticate content.
            auto mutated = bytes;
            mutated[SIZE / 2] ^= 0x80;
            artifact_segment_chain changed;
            check(fill(changed, mutated, PIECE, CHUNK, MAX_CHUNKS), "the mutated chain could not be filled");
            vbr_capture_range_tree changed_tree;
            check(sealed(changed, changed_tree), "sealing the mutated chain failed");
            check(changed_tree.root() != tree.root(),
                  "a single differing byte did not change the sealed root");
            std::puts("VBR capture PASS: sealing binds content");
        }

        // [3] Prove and verify, with the soundness check that gives the proof its value.
        {
            const auto bytes = make_bytes(SIZE, 0x9abcu);
            artifact_segment_chain chain;
            check(fill(chain, bytes, PIECE, CHUNK, MAX_CHUNKS), "the chain could not be filled");
            vbr_capture_range_tree tree;
            check(sealed(chain, tree), "sealing failed");

            vbr_capture_range_proof proof;
            vbr_capture_range_proof_limits limits;
            const std::vector<vbr_capture_authenticated_range> whole { { 0, bytes.size() } };
            check(vbr_capture_range_prove(tree, whole, limits, proof),
                  "proving the whole extent failed");
            check(bool(proof), "a produced proof reports as unset");
            check(proof.root() == tree.root(), "the proof carries a different root than the tree");
            check(proof.total_bytes() == bytes.size(), "the proof bound the wrong byte count");
            check(proof.chunk_bytes() == CHUNK, "the proof bound the wrong chunk size");
            check(proof.selected_chunk_count() > 0, "a proof selected no chunks");
            check(same_ranges(proof.ranges(), whole),
                  "the proof did not retain its authorized ranges");


            uint64_t bytes_read = 0;
            check(vbr_capture_range_verify(proof, chain.source(), &bytes_read),
                  "a proof of the bytes it was made from did not verify");
            check(bytes_read > 0, "verification reported reading nothing");

            // Soundness: the same proof against bytes that differ inside the proven range.
            auto mutated = bytes;
            mutated[SIZE / 2] ^= 0x80;
            artifact_segment_chain changed;
            check(fill(changed, mutated, PIECE, CHUNK, MAX_CHUNKS), "the mutated chain could not be filled");
            check(!vbr_capture_range_verify(proof, changed.source(), nullptr),
                  "a proof verified against bytes it does not authenticate");

            // A partial proof has to carry the siblings that reconnect its leaves to the root; a
            // whole-extent proof legitimately carries none, which is why this is checked here.
            vbr_capture_range_proof partial;
            const std::vector<vbr_capture_authenticated_range> part { { CHUNK, CHUNK } };
            check(vbr_capture_range_prove(tree, part, limits, partial),
                  "proving a single interior chunk failed");
            check(partial.selected_chunk_count() > 0, "the partial proof selected no chunks");
            check(partial.proof_node_count() > 0, "a partial proof carries no sibling nodes");
            check(partial.selected_chunk_count() < proof.selected_chunk_count(),
                  "a partial proof selected as many chunks as the whole extent");
            check(vbr_capture_range_verify(partial, chain.source(), nullptr),
                  "the partial proof did not verify");
            // Soundness applies to the partial proof too: the mutated byte sits in the interior
            // chunk it selects.
            check(!vbr_capture_range_verify(partial, changed.source(), nullptr),
                  "a partial proof verified against bytes it does not authenticate");

            // A source that cannot satisfy the read must fail closed rather than verify.
            vbr_artifact_byte_source truncated { bytes.size(), nullptr,
                [](const void *, uint64_t, uint8_t *, size_t) noexcept { return false; } };
            check(!vbr_capture_range_verify(proof, truncated, nullptr),
                  "a proof verified against a source that never yields bytes");

            // A shorter source of otherwise identical bytes: the extent itself is authenticated.
            artifact_segment_chain shorter;
            check(fill(shorter, std::vector<uint8_t>(bytes.begin(), bytes.begin() + CHUNK),
                       PIECE, CHUNK, MAX_CHUNKS), "the short chain could not be filled");
            check(!vbr_capture_range_verify(proof, shorter.source(), nullptr),
                  "a proof verified against a truncated extent");
            std::puts("VBR capture PASS: prove and verify, with soundness");
        }

        // [4] Admission is fail-closed: a malformed range list never yields a proof.
        {
            const auto bytes = make_bytes(SIZE, 0xdef0u);
            artifact_segment_chain chain;
            check(fill(chain, bytes, PIECE, CHUNK, MAX_CHUNKS), "the chain could not be filled");
            vbr_capture_range_tree tree;
            check(sealed(chain, tree), "sealing failed");
            vbr_capture_range_proof_limits limits;

            auto refused = [&](const std::vector<vbr_capture_authenticated_range> & ranges,
                               const char * what) {
                vbr_capture_range_proof proof;
                if (vbr_capture_range_prove(tree, ranges, limits, proof)) {
                    throw std::runtime_error(std::string("a malformed range list was accepted: ") +
                                             what);
                }
                check(!proof, (std::string("a refused proof still reports as set: ") + what).c_str());
            };

            refused({}, "no ranges");
            refused({ { 0, 0 } }, "an empty range");
            refused({ { 200, 100 }, { 0, 100 } }, "unsorted ranges");
            refused({ { 0, 200 }, { 100, 100 } }, "overlapping ranges");
            refused({ { 0, bytes.size() + 1 } }, "a range past the sealed extent");
            refused({ { bytes.size(), 1 } }, "a range starting past the sealed extent");
            refused({ { 0, 100 }, { 0, 100 } }, "duplicate ranges");

            // A zero-size range adjacent to a real one is still malformed, not a no-op.
            refused({ { 0, 100 }, { 100, 0 } }, "a trailing empty range");

            // An unsealed tree has no authority at all.
            vbr_capture_range_tree empty_tree;
            vbr_capture_range_proof from_empty;
            check(!vbr_capture_range_prove(empty_tree, { { 0, 8 } }, limits, from_empty),
                  "an unsealed tree produced a proof");
            std::puts("VBR capture PASS: fail-closed proof admission");
        }

        // [5] Restriction may narrow authority but never widen it.
        {
            const auto bytes = make_bytes(SIZE, 0x2468u);
            artifact_segment_chain chain;
            check(fill(chain, bytes, PIECE, CHUNK, MAX_CHUNKS), "the chain could not be filled");
            vbr_capture_range_tree tree;
            check(sealed(chain, tree), "sealing failed");
            vbr_capture_range_proof_limits limits;

            vbr_capture_range_proof parent;
            check(vbr_capture_range_prove(tree, { { 0, bytes.size() } }, limits, parent),
                  "the parent proof could not be produced");

            // A subset of the parent's authorized extent is a legitimate narrowing.
            vbr_capture_range_proof child;
            const std::vector<vbr_capture_authenticated_range> subset { { 128, 256 } };
            const auto status = vbr_capture_range_restrict(parent, subset, limits, child);
            check(status == vbr_capture_range_restrict_status::restricted,
                  "restricting to a subset of the authorized extent was refused");
            check(bool(child), "a restricted proof reports as unset");
            check(child.root() == parent.root(), "the child carries a different root");
            check(vbr_capture_range_verify(child, chain.source(), nullptr),
                  "a restricted proof did not verify");
            check(same_ranges(child.ranges(), subset),
                  "the child did not retain its narrower ranges");

            // Widening is refused. Asserted as "not restricted" so the test pins the property
            // rather than a particular failure spelling.
            vbr_capture_range_proof widened;
            const auto outside = vbr_capture_range_restrict(
                parent, { { 0, bytes.size() + CHUNK } }, limits, widened);
            check(outside != vbr_capture_range_restrict_status::restricted,
                  "a proof was restricted to a range the parent never authorised");
            check(!widened, "a refused restriction still produced a proof");

            // So is a subset of a parent that itself only covered part of the extent.
            vbr_capture_range_proof narrow_parent;
            check(vbr_capture_range_prove(tree, { { 0, 256 } }, limits, narrow_parent),
                  "the narrow parent could not be produced");
            vbr_capture_range_proof beyond;
            const auto beyond_status = vbr_capture_range_restrict(
                narrow_parent, { { 512, 128 } }, limits, beyond);
            check(beyond_status != vbr_capture_range_restrict_status::restricted,
                  "a proof was restricted beyond its parent's authorised ranges");
            check(!beyond, "a refused restriction still produced a proof");

            // The same malformations as prove(): a restriction cannot launder them.
            vbr_capture_range_proof junk;
            check(vbr_capture_range_restrict(parent, {}, limits, junk) !=
                      vbr_capture_range_restrict_status::restricted,
                  "an empty range list was restricted");
            check(vbr_capture_range_restrict(parent, { { 200, 100 }, { 0, 100 } }, limits, junk) !=
                      vbr_capture_range_restrict_status::restricted,
                  "an unsorted range list was restricted");
            check(vbr_capture_range_restrict(vbr_capture_range_proof{}, subset, limits, junk) !=
                      vbr_capture_range_restrict_status::restricted,
                  "a proof was restricted from an unset parent");
            std::puts("VBR capture PASS: restriction narrows but never widens");
        }

        // [6] The canonical stream digest is a second, independent binding of the same bytes.
        {
            const auto bytes = make_bytes(SIZE, 0x1357u);
            artifact_segment_chain chain;
            check(fill(chain, bytes, PIECE, CHUNK, MAX_CHUNKS), "the chain could not be filled");
            const auto first = vbr_capture_stream_digest(chain);
            check(nonzero(first), "the stream digest is all zero");
            check(vbr_capture_stream_digest(chain) == first, "the stream digest is not stable");

            artifact_segment_chain same;
            check(fill(same, bytes, PIECE, CHUNK, MAX_CHUNKS), "the duplicate chain could not be filled");
            check(vbr_capture_stream_digest(same) == first,
                  "identical bytes produced different stream digests");

            artifact_segment_chain different;
            check(fill(different, make_bytes(SIZE, 0x8642u), PIECE, CHUNK, MAX_CHUNKS),
                  "the differing chain could not be filled");
            check(vbr_capture_stream_digest(different) != first,
                  "different bytes produced the same stream digest");

            // Length alone is authenticated: a prefix is a different stream.
            artifact_segment_chain prefix;
            check(fill(prefix, std::vector<uint8_t>(bytes.begin(), bytes.begin() + CHUNK),
                       PIECE, CHUNK, MAX_CHUNKS), "the prefix chain could not be filled");
            check(vbr_capture_stream_digest(prefix) != first,
                  "a prefix produced the same stream digest as the whole");
            std::puts("VBR capture PASS: canonical stream digest");
        }

        std::puts("VBR capture tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "VBR capture FAIL: %s\n", error.what());
        return 1;
    }
}
