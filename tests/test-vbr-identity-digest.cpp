// Tests for llama-vbr-identity-digest.h - the shared SHA-256 identity recipes that VBR's
// checkpoint bridge, downward validation and explicit capture all serialize through.
//
// Why this test is hand-written instead of ported: upstream's test-vbr-representation-identity.cpp
// reaches the recipes through vbr_explicit_capture_representation_identity(), so it includes
// llama-vbr-explicit-capture.h, whose own closure is artifact-catalog + generation +
// occupied-replacement. Those land in stages 2.6f/2.7, so the upstream test cannot be built yet.
// This file covers the three recipes that *are* in the header (type-vector, downward tree policy,
// checkpoint identity/policy/order) and deliberately pins their exact wire bytes; the rotation
// recipe only exists inside explicit capture and stays unverified until stage 2.6f brings it.
//
// Three independent oracles are used, because a test that re-derives its expectation from the code
// under test proves nothing:
//   1. SHA-256 known-answer vectors (NIST FIPS 180-4 examples) for the hash primitive itself.
//   2. A hand-built little-endian byte stream for each recipe, assembled in this file from the
//      literals (domain labels included) rather than by calling the writer API.
//   3. Frozen golden digests, so a silent change to a recipe cannot be absorbed by editing the
//      reference in the same commit.
//
// Self-reporting like test-vbr-hard-seal: it returns a status from main and prints explicit PASS
// markers, so NDEBUG cannot hollow it out and it needs no /UNDEBUG.

#include "llama-vbr-identity-digest.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

// ---------------------------------------------------------------- hex helpers

static std::string to_hex(const std::array<uint8_t, 32> & digest) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const uint8_t byte : digest) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

// Compare against a frozen expectation; on mismatch report both, so the value can be read off
// rather than guessed when a recipe legitimately changes.
static void expect_hex(const std::array<uint8_t, 32> & actual, const char * expected_hex,
                       const char * what) {
    if (to_hex(actual) != expected_hex) {
        throw std::runtime_error(std::string(what) + ": expected " + expected_hex +
                                 " but got " + to_hex(actual));
    }
}

// ------------------------------------------------- independent serialization

// The canonical byte format: little-endian fixed-width integers and u64-length-prefixed byte
// strings. Rebuilt here from scratch so a change in the writer's encoding shows up as a mismatch
// rather than as agreement.
//
// Note the two label conventions, which is a real asymmetry in the recipes rather than an
// oversight: the type-vector and tree recipes hash their domain label through writer.string(),
// i.e. length-prefixed, while the checkpoint identity/policy recipe hashes its (different) label as
// raw bytes. Both are reproduced here so the test fails if either one silently changes.
static void put_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        out.push_back(uint8_t(value >> (8*i)));
    }
}

static void put_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out.push_back(uint8_t(value >> (8*i)));
    }
}

static void put_label(std::vector<uint8_t> & out, const char * label) {
    out.insert(out.end(), label, label + std::strlen(label));
}

static void put_string(std::vector<uint8_t> & out, const char * data, size_t size) {
    put_u64(out, size);
    for (size_t i = 0; i < size; ++i) {
        out.push_back(uint8_t(data[i]));
    }
}

// Length taken from the literal rather than a hand-counted constant; a typo in the literal itself
// is still caught, by the frozen golden digests below.
static void put_string(std::vector<uint8_t> & out, const char * text) {
    put_string(out, text, std::strlen(text));
}

static std::array<uint8_t, 32> hash_stream(const std::vector<uint8_t> & bytes) {
    llama_sha256 hash;
    if (!bytes.empty()) {
        hash.update(bytes.data(), bytes.size());
    }
    return hash.finish();
}

// ------------------------------------------------------------ one-shot hashes

static std::array<uint8_t, 32> digest_of(const void * data, size_t size) {
    llama_sha256 hash;
    hash.update(data, size);
    return hash.finish();
}

static std::array<uint8_t, 32> digest_of(const char * text) {
    return digest_of(text, std::strlen(text));
}

// --------------------------------------------- references, assembled by hand

static std::array<uint8_t, 32> ref_type_vector(const ggml_type * types, size_t count) {
    std::vector<uint8_t> bytes;
    put_string(bytes, "buun.vbr.capture/type-vector");
    for (size_t i = 0; i < count; ++i) {
        put_u32(bytes, uint32_t(types[i]));
    }
    return hash_stream(bytes);
}

static std::array<uint8_t, 32> ref_tree(const std::vector<std::array<uint8_t, 32>> & children,
                                        uint32_t recipe_version) {
    std::vector<uint8_t> bytes;
    put_string(bytes, "buun.vbr.downward/tree-policy");
    put_u32(bytes, recipe_version);
    put_u64(bytes, children.size());
    for (const auto & child : children) {
        bytes.insert(bytes.end(), child.begin(), child.end());
    }
    return hash_stream(bytes);
}

static std::array<uint8_t, 32> ref_identity_policy(
        const vbr_checkpoint_frontier_fields & frontier,
        const std::vector<vbr_identity_policy_digest_row> & policy) {
    std::vector<uint8_t> bytes;
    put_label(bytes, "vbr checkpoint identity/policy/order digest v1");
    put_string(bytes, frontier.execution_identity, frontier.execution_identity_len);
    put_string(bytes, frontier.adapter_config_identity, frontier.adapter_config_identity_len);
    put_string(bytes, frontier.media_content_identity, frontier.media_content_identity_len);
    put_u64(bytes, frontier.sequence_epoch);
    put_u64(bytes, uint64_t(frontier.token_count));
    put_u64(bytes, uint64_t(int64_t(frontier.next_position)));
    put_u64(bytes, policy.size());
    for (const auto & row : policy) {
        put_u64(bytes, row.child_id);
        put_u64(bytes, uint64_t(static_cast<uint8_t>(row.mode)));
        put_u64(bytes, row.lineage_uuid.hi);
        put_u64(bytes, row.lineage_uuid.lo);
    }
    return hash_stream(bytes);
}

// --------------------------------------------------------------- fixed inputs

static std::vector<std::array<uint8_t, 32>> golden_children() {
    std::array<uint8_t, 32> first;
    std::array<uint8_t, 32> second;
    for (size_t i = 0; i < 32; ++i) {
        first[i]  = uint8_t(i);
        second[i] = uint8_t(255 - i);
    }
    return {first, second};
}

static vbr_checkpoint_frontier_fields golden_frontier() {
    vbr_checkpoint_frontier_fields frontier;
    frontier.execution_identity          = "exec";
    frontier.execution_identity_len      = 4;
    frontier.adapter_config_identity     = "adapt";
    frontier.adapter_config_identity_len = 5;
    frontier.media_content_identity      = "media";
    frontier.media_content_identity_len  = 5;
    frontier.sequence_epoch = 7;
    frontier.token_count    = 42;
    frontier.next_position  = 11;
    return frontier;
}

static std::vector<vbr_identity_policy_digest_row> golden_policy() {
    vbr_identity_policy_digest_row first;
    first.child_id    = 5;
    first.mode        = checkpoint_child_dependency_mode::live_guarded;
    first.lineage_uuid = {0x1122334455667788ull, 0x99aabbccddeeff00ull};
    vbr_identity_policy_digest_row second;
    second.child_id    = 6;
    second.mode        = checkpoint_child_dependency_mode::absent;
    second.lineage_uuid = {0x0123456789abcdefull, 0xfedcba9876543210ull};
    return {first, second};
}

// Frozen expectations. A recipe change must fail here even if the hand-written reference above is
// edited in the same commit. Values are read from the implementation's own --print output once and
// then frozen; the inputs are visible above and use no ggml_type numbering, so the constants do not
// depend on upstream enum ordering.
static const char * GOLDEN_EMPTY_SHA256 =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
static const char * GOLDEN_ABC_SHA256 =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
static const char * GOLDEN_TYPE_VECTOR =
    "8e57022b17a8371a23aadf93d76ef5106652c69f6fa0776aa8972cf085a15770";
static const char * GOLDEN_TREE =
    "cb82da1222ab89146a0035bfb0ea69cbc612bd601b0c5cd3e58373b9e04973b4";
static const char * GOLDEN_IDENTITY_POLICY =
    "ef2f3ffa2384ebfcb4f52c20b2dca50b94b8003388ef0fd8627df7624ba1a8b3";

// ------------------------------------------------------------------ --print

static int print_golden() {
    const std::vector<ggml_type> types = {
        static_cast<ggml_type>(1), static_cast<ggml_type>(2), static_cast<ggml_type>(3)};
    std::puts(to_hex(digest_of("")).c_str());
    std::puts(to_hex(digest_of("abc")).c_str());
    std::puts(to_hex(vbr_type_vector_digest(types)).c_str());
    std::puts(to_hex(vbr_type_tree_digest(golden_children(), 3)).c_str());
    std::puts(to_hex(vbr_identity_policy_digest(golden_frontier(), golden_policy())).c_str());
    return 0;
}

// ---------------------------------------------------------------------- main

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--print") == 0) {
        return print_golden();
    }
    try {
        // [1] The hash primitive against external truth. A wrong SHA-256 would make every digest
        // below self-consistently wrong, and no in-tree comparison could detect it.
        {
            expect_hex(digest_of(""), GOLDEN_EMPTY_SHA256, "sha256(\"\")");
            expect_hex(digest_of("abc"), GOLDEN_ABC_SHA256, "sha256(\"abc\")");
            // Two NIST 180-4 vectors that straddle the 64-byte block: a 56-byte message (one block
            // plus padding) and the 112-byte two-block message.
            expect_hex(digest_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
                       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
                       "sha256(56-byte vector)");
            expect_hex(digest_of("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
                                 "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
                       "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
                       "sha256(112-byte vector)");
            // Feeding the same message byte-by-byte must not change the digest: this is what proves
            // the block buffer and the length counter work across chunk boundaries.
            static const char message[] = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
            llama_sha256 split;
            for (size_t i = 0; i < sizeof(message) - 1; ++i) {
                split.update(message + i, 1);
            }
            expect_hex(split.finish(),
                       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
                       "sha256(56-byte vector, byte-at-a-time)");
            std::puts("VBR identity digest PASS: sha256 known-answer vectors");
        }

        // [2] llama_sha256_writer's wire format, checked against the by-hand encoder.
        {
            llama_sha256_writer writer;
            writer.u32(0x01020304u);
            writer.u64(0x1112131415161718ull);
            writer.string("ab", 2);
            writer.string(nullptr, 0);
            std::vector<uint8_t> reference;
            put_u32(reference, 0x01020304u);
            put_u64(reference, 0x1112131415161718ull);
            put_string(reference, "ab", 2);
            put_string(reference, nullptr, 0);
            check(writer.finish() == hash_stream(reference), "writer wire format drifted");

            // The length prefix is what stops ("ab","c") and ("a","bc") from colliding; assert it
            // directly instead of trusting the recipe that depends on it. Without the prefix both
            // streams would be the same five bytes "abc" in the same order.
            std::vector<uint8_t> split_left;
            put_string(split_left, "ab", 2);
            put_string(split_left, "c", 1);
            std::vector<uint8_t> split_right;
            put_string(split_right, "a", 1);
            put_string(split_right, "bc", 2);
            check(split_left.size() == split_right.size(),
                  "the two split references are not the same length");
            check(hash_stream(split_left) != hash_stream(split_right),
                  "length prefix did not separate a short string");
            check(digest_of("abc") != hash_stream(split_left),
                  "untagged bytes collided with a length-prefixed string");
            std::puts("VBR identity digest PASS: serialization wire format");
        }

        // [3] type-vector recipe: label, one u32 per element, order and count significant.
        {
            const std::vector<ggml_type> types = {
                GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0};
            check(vbr_type_vector_digest(types) == ref_type_vector(types.data(), types.size()),
                  "type-vector digest does not match the hand-built stream");
            check(vbr_type_vector_digest(types) ==
                      vbr_type_vector_digest(types.data(), types.size()),
                  "type-vector digest depends on which overload was called");

            const std::vector<ggml_type> swapped = {
                GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16};
            check(vbr_type_vector_digest(swapped) != vbr_type_vector_digest(types),
                  "type-vector digest ignored element order");

            const std::vector<ggml_type> truncated = {GGML_TYPE_F16, GGML_TYPE_Q8_0};
            check(vbr_type_vector_digest(truncated) != vbr_type_vector_digest(types),
                  "type-vector digest ignored the element count");

            const std::vector<ggml_type> empty;
            check(vbr_type_vector_digest(empty) == ref_type_vector(nullptr, 0),
                  "empty type-vector digest is not the bare label");
            check(vbr_type_vector_digest(empty) != vbr_type_vector_digest(truncated),
                  "empty and non-empty type vectors collided");
            std::puts("VBR identity digest PASS: type-vector recipe");
        }

        // [4] downward tree-policy recipe: label, recipe version, u64 child count, then the digests.
        {
            const std::vector<std::array<uint8_t, 32>> children = golden_children();
            check(vbr_type_tree_digest(children, 3) == ref_tree(children, 3),
                  "tree-policy digest does not match the hand-built stream");

            check(vbr_type_tree_digest(children, 4) != vbr_type_tree_digest(children, 3),
                  "tree-policy digest ignored the recipe version");

            auto reordered = children;
            std::swap(reordered[0], reordered[1]);
            check(vbr_type_tree_digest(reordered, 3) != vbr_type_tree_digest(children, 3),
                  "tree-policy digest ignored child order");

            const std::vector<std::array<uint8_t, 32>> fewer = {children[0]};
            check(vbr_type_tree_digest(fewer, 3) != vbr_type_tree_digest(children, 3),
                  "tree-policy digest ignored the child count");

            const std::vector<std::array<uint8_t, 32>> none;
            check(vbr_type_tree_digest(none, 3) == ref_tree(none, 3),
                  "empty-tree digest does not match the hand-built stream");
            check(vbr_type_tree_digest(none, 3) != vbr_type_tree_digest(fewer, 3),
                  "empty and one-child trees collided");
            std::puts("VBR identity digest PASS: tree-policy recipe");
        }

        // [5] checkpoint identity/policy/order recipe. Every field is bound individually: a field
        // that silently stops being hashed is exactly the failure this digest exists to prevent.
        {
            const auto frontier = golden_frontier();
            const auto policy   = golden_policy();
            const auto digest   = vbr_identity_policy_digest(frontier, policy);
            check(digest == ref_identity_policy(frontier, policy),
                  "identity-policy digest does not match the hand-built stream");

            const auto differs = [&](const vbr_checkpoint_frontier_fields & changed,
                                     const std::vector<vbr_identity_policy_digest_row> & rows,
                                     const char * what) {
                check(vbr_identity_policy_digest(changed, rows) != digest, what);
            };

            auto changed = frontier;
            changed.execution_identity = "exeD";
            differs(changed, policy, "execution identity is not bound");

            changed = frontier;
            changed.adapter_config_identity = "adapT";
            differs(changed, policy, "adapter config identity is not bound");

            changed = frontier;
            changed.media_content_identity = "mediA";
            differs(changed, policy, "media content identity is not bound");

            changed = frontier;
            changed.sequence_epoch = 8;
            differs(changed, policy, "sequence epoch is not bound");

            changed = frontier;
            changed.token_count = 43;
            differs(changed, policy, "token count is not bound");

            changed = frontier;
            changed.next_position = 12;
            differs(changed, policy, "next position is not bound");

            // next_position is llama_pos (int32_t) and reaches the writer through
            // uint64_t(int64_t(...)), so a "no frontier" -1 must not alias 0 or a positive value.
            changed = frontier;
            changed.next_position = -1;
            differs(changed, policy, "negative next position aliases its magnitude");
            auto negative = changed;
            changed.next_position = 0;
            differs(changed, policy, "next position -1 and 0 collide");
            check(vbr_identity_policy_digest(negative, policy) !=
                      vbr_identity_policy_digest(changed, policy),
                  "next position sign is not bound");

            changed = frontier;
            changed.token_count = -1;
            check(vbr_identity_policy_digest(changed, policy) != digest,
                  "negative token count is not bound");

            // The length-prefixed strings must not be able to be re-split across each other.
            auto re_split = frontier;
            re_split.execution_identity     = "execa";
            re_split.execution_identity_len = 5;
            re_split.adapter_config_identity     = "dapt";
            re_split.adapter_config_identity_len = 4;
            differs(re_split, policy, "identities were concatenated without lengths");

            auto row_changed = policy;
            row_changed[0].child_id = 7;
            differs(frontier, row_changed, "policy row child id is not bound");

            row_changed = policy;
            row_changed[0].mode = checkpoint_child_dependency_mode::payload_complete;
            differs(frontier, row_changed, "policy row dependency mode is not bound");

            row_changed = policy;
            row_changed[0].mode = checkpoint_child_dependency_mode::absent;
            differs(frontier, row_changed, "absent and live_guarded modes collided");

            row_changed = policy;
            row_changed[0].lineage_uuid.hi = 0x1122334455667789ull;
            differs(frontier, row_changed, "lineage uuid high word is not bound");

            row_changed = policy;
            row_changed[0].lineage_uuid.lo = 0x99aabbccddeeff01ull;
            differs(frontier, row_changed, "lineage uuid low word is not bound");

            // Swapping the two words must not be a no-op: they are serialized in order, not as a set.
            row_changed = policy;
            std::swap(row_changed[0].lineage_uuid.hi, row_changed[0].lineage_uuid.lo);
            differs(frontier, row_changed, "lineage uuid word order is not bound");

            std::vector<vbr_identity_policy_digest_row> reordered = policy;
            std::swap(reordered[0], reordered[1]);
            differs(frontier, reordered, "policy row order is not bound");

            const std::vector<vbr_identity_policy_digest_row> fewer = {policy[0]};
            differs(frontier, fewer, "policy length is not bound");

            const std::vector<vbr_identity_policy_digest_row> none;
            check(vbr_identity_policy_digest(frontier, none) == ref_identity_policy(frontier, none),
                  "empty-policy digest does not match the hand-built stream");

            // A frontier with no strings at all: the three empty strings must still be length-tagged.
            vbr_checkpoint_frontier_fields bare;
            bare.sequence_epoch = 0;
            bare.token_count    = 0;
            bare.next_position  = 0;
            check(vbr_identity_policy_digest(bare, none) == ref_identity_policy(bare, none),
                  "null identity pointers are not serialized as empty strings");
            std::puts("VBR identity digest PASS: checkpoint identity/policy/order recipe");
        }

        // [6] digest_nonzero, frozen goldens, and the recipes' freedom from hidden state.
        {
            std::array<uint8_t, 32> zero {};
            check(!vbr_digest_nonzero(zero), "all-zero digest reported as set");
            std::array<uint8_t, 32> first {};
            first[0] = 1;
            check(vbr_digest_nonzero(first), "leading non-zero byte missed");
            std::array<uint8_t, 32> last {};
            last[31] = 1;
            check(vbr_digest_nonzero(last), "trailing non-zero byte missed");
            for (size_t i = 0; i < 32; ++i) {
                std::array<uint8_t, 32> single {};
                single[i] = 0x80;
                check(vbr_digest_nonzero(single), "non-zero byte missed inside the digest");
            }

            const std::vector<ggml_type> types = {
                static_cast<ggml_type>(1), static_cast<ggml_type>(2), static_cast<ggml_type>(3)};
            const auto children = golden_children();
            const auto frontier = golden_frontier();
            const auto policy   = golden_policy();
            expect_hex(digest_of(""), GOLDEN_EMPTY_SHA256, "golden sha256(\"\")");
            expect_hex(digest_of("abc"), GOLDEN_ABC_SHA256, "golden sha256(\"abc\")");
            expect_hex(vbr_type_vector_digest(types), GOLDEN_TYPE_VECTOR, "golden type vector");
            expect_hex(vbr_type_tree_digest(children, 3), GOLDEN_TREE, "golden tree policy");
            expect_hex(vbr_identity_policy_digest(frontier, policy), GOLDEN_IDENTITY_POLICY,
                       "golden identity policy");

            // The recipes are pure functions over their arguments. Nothing in the header may cache a
            // digest, so concurrent first use must agree with the single-threaded answer above -
            // upstream's identity test exists precisely because its capture layer does cache.
            const auto expected_vector = vbr_type_vector_digest(types);
            const auto expected_policy = vbr_identity_policy_digest(frontier, policy);
            std::atomic<int>  ready {0};
            std::atomic<bool> go {false};
            std::atomic<bool> ok {true};
            std::vector<std::thread> workers;
            for (int i = 0; i < 8; ++i) {
                workers.emplace_back([&] {
                    ++ready;
                    while (!go.load()) { std::this_thread::yield(); }
                    for (int repeat = 0; repeat < 64; ++repeat) {
                        if (vbr_type_vector_digest(types) != expected_vector ||
                            vbr_identity_policy_digest(frontier, policy) != expected_policy) {
                            ok = false;
                        }
                    }
                });
            }
            while (ready.load() != 8) { std::this_thread::yield(); }
            go = true;
            for (auto & worker : workers) { worker.join(); }
            check(ok, "digest recipes are not reentrant");
            std::puts("VBR identity digest PASS: digest_nonzero, golden vectors and reentrancy");
        }

        std::puts("VBR identity digest tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "VBR identity digest FAIL: %s\n", error.what());
        return 1;
    }
}
