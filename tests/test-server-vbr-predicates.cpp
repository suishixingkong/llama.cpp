// Unit test for the pure decision helpers landed with the server cache/VBR
// consumer rework (slice 7d, part 1). These functions were copied verbatim from
// the fork, so the point of this test is to pin their truth tables: a
// transcription error in a predicate like
// `slot_count == 1 && !hard_lease && !deferred_task && incumbent_supported`
// would otherwise be invisible until the scheduler rework starts calling them.
//
// Deliberately NOT assert-based: the Release build defines NDEBUG, which turns
// assert() into a no-op and an assert-shaped test into a vacuous one. Failures
// are counted and printed, and the exit code is derived from that count.
//
// Model-free and CPU-only: no GGUF is loaded.

#include "server-context.h"
#include "server-task.h"
#include "server-common.h"

#include "../src/llama-adapter.h"

#include <array>
#include <cstdio>
#include <memory>
#include <string>

static int g_checks   = 0;
static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);         \
        }                                                                      \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!((a) == (b))) {                                                   \
            ++g_failures;                                                      \
            std::printf("FAIL %s:%d  %s != %s\n", __FILE__, __LINE__, #a, #b);  \
        }                                                                      \
    } while (0)

using terminal = server_speculative_decode_terminal;

// --- server_speculative_decode_terminal_resolve -----------------------------
// decode_result == 0 succeeds unless the selected implementation threw or its
// speculative bookkeeping is not ok, in which case the committed state must be
// reset before the exception propagates.
static void test_decode_terminal() {
    CHECK(server_speculative_decode_terminal_resolve(0, false, false, false, true) ==
          terminal::success);
    CHECK(server_speculative_decode_terminal_resolve(0, true, true, true, false) ==
          terminal::reset_committed_then_throw);
    // either trigger alone is enough, and they do not compose into success
    CHECK(server_speculative_decode_terminal_resolve(0, false, false, true, true) ==
          terminal::reset_committed_then_throw);
    CHECK(server_speculative_decode_terminal_resolve(0, false, false, false, false) ==
          terminal::reset_committed_then_throw);

    // a hard seal is checked before the retry bucket
    CHECK(server_speculative_decode_terminal_resolve(1, true, false, false, true) ==
          terminal::preserve_hard_seal);
    CHECK(server_speculative_decode_terminal_resolve(1, true, true, false, true) ==
          terminal::preserve_hard_seal);
    // ... but a hard seal never rescues a failure that is not decode_result == 1
    CHECK(server_speculative_decode_terminal_resolve(-1, true, false, false, true) ==
          terminal::ordinary_ret_error);
    CHECK(server_speculative_decode_terminal_resolve(-2, true, false, false, true) ==
          terminal::ordinary_ret_error);

    // ordinary ret error: -1, below -1, or +1 on a single-token batch
    CHECK(server_speculative_decode_terminal_resolve(1, false, true, false, true) ==
          terminal::ordinary_ret_error);
    CHECK(server_speculative_decode_terminal_resolve(-1, false, false, false, true) ==
          terminal::ordinary_ret_error);
    CHECK(server_speculative_decode_terminal_resolve(-7, false, false, false, true) ==
          terminal::ordinary_ret_error);
    // a positive multi-result without a hard seal and without a single-token
    // batch is retryable
    CHECK(server_speculative_decode_terminal_resolve(1, false, false, false, true) ==
          terminal::retry);
    CHECK(server_speculative_decode_terminal_resolve(3, false, false, false, true) ==
          terminal::retry);
    // the exception/speculative_ok guard only applies on the success path
    CHECK(server_speculative_decode_terminal_resolve(3, false, false, true, false) ==
          terminal::retry);
}

// --- server_vbr_empty_handoff_* ---------------------------------------------
static void test_empty_handoff() {
    server_vbr_empty_handoff_gate gate;
    // defaults: no incumbent, no durability, no family match
    CHECK(!server_vbr_empty_handoff_lookup_allowed(gate));
    CHECK(!server_vbr_empty_handoff_allowed(gate));

    // lookup requires exactly one slot, no hard lease, no deferred task, and a
    // supported incumbent
    gate.slot_count = 1;
    gate.incumbent_supported = true;
    CHECK(server_vbr_empty_handoff_lookup_allowed(gate));
    gate.slot_count = 2;
    CHECK(!server_vbr_empty_handoff_lookup_allowed(gate));
    gate.slot_count = 0;
    CHECK(!server_vbr_empty_handoff_lookup_allowed(gate));
    gate.slot_count = 1;

    gate.hard_lease = true;
    CHECK(!server_vbr_empty_handoff_lookup_allowed(gate));
    gate.hard_lease = false;
    gate.deferred_task = true;
    CHECK(!server_vbr_empty_handoff_lookup_allowed(gate));
    gate.deferred_task = false;
    gate.incumbent_supported = false;
    CHECK(!server_vbr_empty_handoff_lookup_allowed(gate));
    gate.incumbent_supported = true;

    // admission additionally requires strict progress over BOTH the incumbent
    // lcp and its durable prefix, an inexactly-durable incumbent, and a
    // matching declared family. A gate that passes lookup but fails any of
    // these must be refused.
    CHECK(!server_vbr_empty_handoff_allowed(gate));            // 0 > 0 false
    gate.incoming_prefix = 10;
    CHECK(!server_vbr_empty_handoff_allowed(gate));            // durable 0 > 0 false
    gate.durable_incumbent_prefix = 5;
    CHECK(!server_vbr_empty_handoff_allowed(gate));            // family_matches false
    gate.family_matches = true;
    CHECK(server_vbr_empty_handoff_allowed(gate));
    // equal-prefix is not strict progress
    gate.incumbent_lcp = 10;
    CHECK(!server_vbr_empty_handoff_allowed(gate));
    gate.incumbent_lcp = 4;
    CHECK(server_vbr_empty_handoff_allowed(gate));
    // an exactly durable incumbent means there is nothing to gain
    gate.exact_incumbent_durable = true;
    CHECK(!server_vbr_empty_handoff_allowed(gate));
    gate.exact_incumbent_durable = false;

    // the lookup preconditions are inherited by admission
    gate.slot_count = 2;
    CHECK(!server_vbr_empty_handoff_allowed(gate));
    gate.slot_count = 1;
    gate.hard_lease = true;
    CHECK(!server_vbr_empty_handoff_allowed(gate));
    gate.hard_lease = false;
    // durability must beat the incumbent lcp independently of incoming_prefix
    gate.incoming_prefix = 100;
    gate.durable_incumbent_prefix = 4;
    CHECK(!server_vbr_empty_handoff_allowed(gate));
}

// --- server_vbr_live_source_displacement_allowed ----------------------------
static void test_live_source_displacement() {
    // a non-unified tree keeps each sequence separable
    CHECK(server_vbr_live_source_displacement_allowed(false, 1));
    CHECK(server_vbr_live_source_displacement_allowed(false, 64));
    // a unified tree only when nothing else shares the rows
    CHECK(server_vbr_live_source_displacement_allowed(true, 0));
    CHECK(server_vbr_live_source_displacement_allowed(true, 1));
    CHECK(!server_vbr_live_source_displacement_allowed(true, 2));
    CHECK(!server_vbr_live_source_displacement_allowed(true, 64));
}

// --- server_vbr_stem_matches_capture_source ---------------------------------
static void test_stem_matches() {
    const std::array<uint8_t, 32> a = [] { std::array<uint8_t, 32> v{}; v[0] = 1; return v; }();
    const std::array<uint8_t, 32> b = [] { std::array<uint8_t, 32> v{}; v[0] = 2; return v; }();
    CHECK(server_vbr_stem_matches_capture_source(true,  a, a));
    CHECK(!server_vbr_stem_matches_capture_source(true,  a, b));
    // an unavailable stem is never a match, even against itself
    CHECK(!server_vbr_stem_matches_capture_source(false, a, a));
    CHECK(!server_vbr_stem_matches_capture_source(false, a, b));
}

// --- server_vbr_occupied_quarantine_reset_for_test ---------------------------
// A runtime consumer for server_vbr_apply_occupied_restore_failure_to_slot: the
// helper builds a live-looking slot, drives the classifier down both of its arms
// and reports what survived. Every one of the six fields must hold. The
// load-bearing half is `retained_prefix_zero`: the classifier is
// `retained_prefix = quarantined ? 0 : incumbent_prefix`, so a swapped ternary
// (or a missed `mandatory_recovery_reset`) flips exactly this field while the
// replay arm still passes.
static void test_occupied_quarantine_reset() {
    const auto result = server_vbr_occupied_quarantine_reset_for_test();

    CHECK(result.replay_preserved_prefix);  // a replayed (non-quarantined) failure keeps the prefix
    CHECK(result.replay_preserved_slot);    // ... and leaves the slot's tokens/epoch/family intact
    CHECK(result.quarantined);              // a quarantine-class failure is reported as such
    CHECK(result.retained_prefix_zero);     // ... and collapses the retained prefix to zero
    CHECK(result.prompt_cleared);           // mandatory_recovery_reset ran: tokens + epoch cleared
    CHECK(result.family_cleared);           // ... and the declared cache family was dropped too           // ... and the declared cache family was dropped too
}

// --- server_tokens::media_content_identity ----------------------------------
static void test_media_content_identity() {
    server_tokens tokens(llama_tokens { 1, 2, 3 }, false);

    // no media: the key is the frozen domain tag plus a zero chunk count
    std::string out;
    CHECK(tokens.media_content_identity(3, out));
    CHECK_EQ(out, std::string("server-media-prefix-v1|chunks:0"));
    // a shorter prefix yields the same key because no chunk is involved
    CHECK(tokens.media_content_identity(1, out));
    CHECK_EQ(out, std::string("server-media-prefix-v1|chunks:0"));
    // an empty prefix is still a coherent (media-free) frontier
    CHECK(tokens.media_content_identity(0, out));
    CHECK_EQ(out, std::string("server-media-prefix-v1|chunks:0"));

    // out-of-range prefixes must be refused, not clamped
    out = "sentinel";
    CHECK(!tokens.media_content_identity(-1, out));
    CHECK(!tokens.media_content_identity(4, out));
    CHECK(!tokens.media_content_identity(1LL << 40, out));
}

// --- server_cache_lease_build_identity --------------------------------------
static void test_lease_build_identity() {
    server_tokens tokens(llama_tokens { 1, 2, 3 }, false);
    server_cache_lease_identity id;

    // both opaque identities are required
    id = {};
    CHECK(!server_cache_lease_build_identity("", "adapter", tokens, 3, id));
    CHECK(!server_cache_lease_build_identity("exec", "", tokens, 3, id));
    CHECK(!server_cache_lease_build_identity("", "", tokens, 3, id));
    // negative coverage is refused
    CHECK(!server_cache_lease_build_identity("exec", "adapter", tokens, -1, id));
    // coverage beyond the token count is refused by the media key
    CHECK(!server_cache_lease_build_identity("exec", "adapter", tokens, 4, id));

    // the success path fills all three identity components
    id = {};
    CHECK(server_cache_lease_build_identity("exec", "adapter", tokens, 3, id));
    CHECK_EQ(id.execution_identity, std::string("exec"));
    CHECK_EQ(id.adapter_config_identity, std::string("adapter"));
    CHECK_EQ(id.media_content_identity, std::string("server-media-prefix-v1|chunks:0"));
    CHECK(id.valid());

    // an empty coverage is still a valid lease identity for this prefix
    id = {};
    CHECK(server_cache_lease_build_identity("exec", "adapter", tokens, 0, id));
    CHECK(id.valid());
}

// --- lora_config_identity ---------------------------------------------------
// The key that persisted prompt-cache entries are stored under. If this is not a
// pure function of the *active* adapter set, a cache entry built with one adapter
// set can be restored for a request with a different one, which is a correctness
// bug rather than a performance regression. The adapters are hand-built here (their
// digest field is written directly) so the test stays model-free; the digest itself
// is covered by test-lora-adapter-digest.
namespace {

struct fake_lora {
    std::unique_ptr<llama_adapter_lora> adapter;
    common_adapter_lora_info             info;
};

fake_lora make_fake_lora(uint8_t digest_fill, float scale) {
    fake_lora fake;
    fake.adapter = std::make_unique<llama_adapter_lora>(nullptr);
    fake.adapter->digest.fill(digest_fill);
    fake.info.scale = scale;
    fake.info.ptr   = fake.adapter.get();
    return fake;
}

} // namespace

static void test_lora_config_identity() {
    // empty set (base adapters only) is itself a well-defined identity
    CHECK_EQ(lora_config_identity({}), std::string("0"));

    // frozen golden: one active adapter, digest 0..31, scale 1.0f.
    // The entry is "digest bytes || scale bits (LE)" -- pinning the order here is
    // what catches a silent swap of the two fields.
    {
        fake_lora one = make_fake_lora(0x00, 1.0f);
        one.adapter->digest.fill(0x00);
        for (size_t i = 0; i < one.adapter->digest.size(); ++i) {
            one.adapter->digest[i] = uint8_t(i);
        }
        const std::string key = lora_config_identity({ one.info });
        std::string expected = "1";
        expected.append(reinterpret_cast<const char *>(one.adapter->digest.data()), 32);
        expected.push_back('\x00'); // 1.0f == 0x3f800000, little-endian
        expected.push_back('\x00');
        expected.push_back('\x80');
        expected.push_back('\x3f');
        CHECK_EQ(key, expected);
        CHECK_EQ(key.size(), size_t(1 + 32 + 4));
    }

    fake_lora a = make_fake_lora(0x11, 1.0f);
    fake_lora b = make_fake_lora(0x22, 0.5f);

    // order independence: the same set in either request order is one identity
    const std::string ab = lora_config_identity({ a.info, b.info });
    const std::string ba = lora_config_identity({ b.info, a.info });
    CHECK_EQ(ab, ba);
    CHECK(ab != lora_config_identity({ a.info }));
    CHECK(ab != lora_config_identity({ b.info }));

    // the count prefix keeps a duplicated adapter distinct from a single one
    CHECK(lora_config_identity({ a.info, a.info }) != lora_config_identity({ a.info }));
    // ... and the two entries are byte-identical, which is why the prefix is needed
    CHECK_EQ(lora_config_identity({ a.info, a.info }).size(), size_t(1 + 2*(32 + 4)));

    // inactive adapters do not affect execution, so they must not affect the key.
    // -0.0f compares equal to 0.0f, so both spellings of "off" are excluded.
    fake_lora off      = make_fake_lora(0x33, 0.0f);
    fake_lora off_neg  = make_fake_lora(0x44, -0.0f);
    CHECK_EQ(lora_config_identity({ off.info }), std::string("0"));
    CHECK_EQ(lora_config_identity({ off_neg.info }), std::string("0"));
    CHECK_EQ(lora_config_identity({ off.info, a.info }), lora_config_identity({ a.info }));
    CHECK_EQ(lora_config_identity({ a.info, off_neg.info }), lora_config_identity({ a.info }));

    // a null pointer is excluded too, even with a live scale (defensive: the
    // loader can leave ptr unset for an adapter that failed to load)
    fake_lora null_ptr = make_fake_lora(0x55, 1.0f);
    null_ptr.info.ptr = nullptr;
    CHECK_EQ(lora_config_identity({ null_ptr.info }), std::string("0"));
    CHECK_EQ(lora_config_identity({ null_ptr.info, a.info }), lora_config_identity({ a.info }));

    // both components of an active entry are load-bearing
    fake_lora a_same_digest_other_scale = make_fake_lora(0x11, 2.0f);
    CHECK(lora_config_identity({ a.info }) != lora_config_identity({ a_same_digest_other_scale.info }));
    fake_lora other_digest_same_scale = make_fake_lora(0x12, 1.0f);
    CHECK(lora_config_identity({ a.info }) != lora_config_identity({ other_digest_same_scale.info }));
}

// --- server_tokens retention identity --------------------------------------
// The scheduler/cache compares token sequences by this digest instead of by the
// token list, and caches the result, so two things have to hold: the digest is a
// pure function of the sequence, and every mutation that changes the sequence
// invalidates the cache. The second one is the interesting half -- a missed
// invalidation makes two different sequences compare equal, which is how a cache
// entry built for one request gets reused for another.
static std::string digest_to_hex(const std::array<uint8_t, 32> & digest) {
    static const char * digits = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint8_t byte : digest) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

static void test_retention_token_identity() {
    server_tokens base(llama_tokens { 1, 2, 3 }, false);

    // text-only tokens have no media, and the retention view is the token list
    CHECK(!base.has_media());
    CHECK_EQ(base.retention_token_ids().size(), size_t(3));
    CHECK_EQ(base.retention_token_ids()[0], llama_token(1));
    CHECK_EQ(base.retention_token_ids()[2], llama_token(3));

    std::array<uint8_t, 32> whole = {};
    CHECK(base.retention_token_digest(whole));

    // Frozen golden: SHA-256 over the length-delimited domain string, the u64 token
    // count and then one little-endian u32 per token id. Computed independently in
    // Python, so this pins the domain, the count prefix and the per-token encoding
    // against a silent reorder.
    CHECK_EQ(digest_to_hex(whole),
             std::string("d4b770e94ff2b3e358f2c99526ab1d9066c05d92e1b9ce34653aced3194d852c"));

    // ... and it is a function of the sequence, not of the object
    {
        server_tokens same(llama_tokens { 1, 2, 3 }, false);
        std::array<uint8_t, 32> other = {};
        CHECK(same.retention_token_digest(other));
        CHECK(other == whole);
    }
    {
        server_tokens longer(llama_tokens { 1, 2, 3, 4 }, false);
        std::array<uint8_t, 32> other = {};
        CHECK(longer.retention_token_digest(other));
        CHECK(other != whole);
        CHECK_EQ(digest_to_hex(other),
                 std::string("67507e78ba140b10ee3b6f46f8164ca5bbc75cc2937074c4cbf3d53c79b12c0f"));
    }
    {
        server_tokens empty(llama_tokens {}, false);
        std::array<uint8_t, 32> other = {};
        CHECK(empty.retention_token_digest(other));
        CHECK(other != whole);
    }

    // every mutator has to invalidate the cache. Each case reads the digest first,
    // mutates, then reads again -- a stale hit shows up as "unchanged".
    {
        server_tokens t(llama_tokens { 1, 2, 3 }, false);
        std::array<uint8_t, 32> before = {};
        std::array<uint8_t, 32> after = {};
        CHECK(t.retention_token_digest(before));
        t.push_back(llama_token(4));
        CHECK(t.retention_token_digest(after));
        CHECK(after != before);
        // growing then shrinking back to the original sequence restores the digest
        t.keep_first(3);
        CHECK(t.retention_token_digest(after));
        CHECK(after == before);
    }
    {
        server_tokens t(llama_tokens { 1, 2, 3 }, false);
        std::array<uint8_t, 32> before = {};
        std::array<uint8_t, 32> after = {};
        CHECK(t.retention_token_digest(before));
        t.insert(llama_tokens { 7, 8 });
        CHECK(t.retention_token_digest(after));
        CHECK(after != before);
    }
    {
        server_tokens t(llama_tokens { 1, 2, 3 }, false);
        std::array<uint8_t, 32> before = {};
        std::array<uint8_t, 32> after = {};
        CHECK(t.retention_token_digest(before));
        t.set_token(1, llama_token(99));
        CHECK(t.retention_token_digest(after));
        CHECK(after != before);
    }
    {
        server_tokens t(llama_tokens { 1, 2, 3 }, false);
        std::array<uint8_t, 32> before = {};
        std::array<uint8_t, 32> after = {};
        CHECK(t.retention_token_digest(before));
        t.keep_first(2);
        CHECK(t.retention_token_digest(after));
        CHECK(after != before);
    }
    {
        server_tokens t(llama_tokens { 1, 2, 3 }, false);
        std::array<uint8_t, 32> before = {};
        std::array<uint8_t, 32> after = {};
        CHECK(t.retention_token_digest(before));
        t.clear();
        CHECK(t.retention_token_digest(after));
        CHECK(after != before);
        CHECK(t.empty());
    }
    {
        // the cache is carried by clone(), so a clone must not re-derive a
        // different value; the sequence is unchanged, so the digest is unchanged
        server_tokens t(llama_tokens { 1, 2, 3 }, false);
        std::array<uint8_t, 32> before = {};
        CHECK(t.retention_token_digest(before));
        server_tokens copy = t.clone();
        std::array<uint8_t, 32> after = {};
        CHECK(copy.retention_token_digest(after));
        CHECK(after == before);
    }

    // --- prefix digest ---
    // A prefix identity is what a bounded source witness is published under, so it
    // must depend only on the covered prefix, and must refuse a boundary it cannot
    // describe exactly.
    {
        server_tokens p(llama_tokens { 1, 2, 3 }, false);
        std::array<uint8_t, 32> prefix = {};
        CHECK((p.retention_token_prefix_digest(3, prefix)));
        CHECK_EQ(digest_to_hex(prefix),
                 std::string("e65ed2be72e483d74022de4c419d9b9311802ba65ef5dc7ba53c1f9bfb786f0a"));
        // a different domain from the whole-token digest, even at equal coverage
        CHECK(prefix != whole);
    }
    {
        server_tokens p(llama_tokens { 1, 2, 3, 4 }, false);
        server_tokens q(llama_tokens { 1, 2, 3, 5 }, false);
        std::array<uint8_t, 32> dp = {};
        std::array<uint8_t, 32> dq = {};
        CHECK((p.retention_token_prefix_digest(3, dp)));
        CHECK((q.retention_token_prefix_digest(3, dq)));
        // same covered prefix, different suffix: the prefix identity must agree
        CHECK(dp == dq);
        std::array<uint8_t, 32> dp4 = {};
        CHECK((p.retention_token_prefix_digest(4, dp4)));
        CHECK(dp4 != dp);
    }
    {
        server_tokens p(llama_tokens { 1, 2, 3 }, false);
        std::array<uint8_t, 32> out = {};
        out.fill(0xAB);
        // coverage 0 and coverage beyond the sequence are both refused...
        CHECK((!p.retention_token_prefix_digest(0, out)));
        CHECK((out == std::array<uint8_t, 32>{}));
        out.fill(0xAB);
        CHECK((!p.retention_token_prefix_digest(4, out)));
        CHECK((out == std::array<uint8_t, 32>{}));
    }
}

// fork 9l-d2b: server_slot_prompt_admission_check is a pure predicate the launch
// consumer will call from 9l-d2c on, so nothing exercises it yet. A transcription
// error in `n_tokens > n_ctx || (can_split && n_tokens == n_ctx)` -- e.g. dropping
// the `can_split &&` guard, or checking the batch limit for splittable prompts --
// would still compile and would only show up as a wrong HTTP status much later.
static void test_prompt_admission_check() {
    using A = server_slot_prompt_admission;

    struct row {
        bool   can_split;
        size_t n_tokens;
        size_t n_ubatch;
        size_t n_ctx;
        A      want;
    };
    const row rows[] = {
        // splittable: the batch bound does not apply, and the context bound reserves
        // one cell of generation headroom (n_tokens == n_ctx is already too large).
        { true,   10,  10, 100, A::accepted           },
        { true,   99,  10, 100, A::accepted           },
        { true,  100,  10, 100, A::context_too_large  },
        { true,  101,  10, 100, A::context_too_large  },
        { true,    1,   1,   1, A::context_too_large  },
        // unsplittable: the batch bound is checked FIRST and wins, and an unsplit
        // prompt is allowed to exactly fill the context.
        { false,  10,  10, 100, A::accepted           },
        { false,  11,  10, 100, A::batch_too_large    },
        { false,  10,  10,  10, A::accepted           },
        { false,  11,  10,  10, A::batch_too_large    },
        { false, 200,  10, 100, A::batch_too_large    }, // batch beats context
        { false, 100, 200, 100, A::accepted           },
        { false, 101, 200, 100, A::context_too_large  },
    };
    for (const auto & r : rows) {
        const A got = server_slot_prompt_admission_check(
            r.can_split, r.n_tokens, r.n_ubatch, r.n_ctx);
        CHECK(got == r.want);
    }

    // A frozen pair of boundary identities, so a reordered condition that happens to
    // agree on the table above still gets caught: equality at the context bound is
    // refused ONLY when the prompt can be split.
    CHECK((server_slot_prompt_admission_check(true,  100, 100, 100) == A::context_too_large));
    CHECK((server_slot_prompt_admission_check(false, 100, 100, 100) == A::accepted));
}

int main() {
    test_decode_terminal();
    test_empty_handoff();
    test_live_source_displacement();
    test_stem_matches();
    test_occupied_quarantine_reset();
    test_prompt_admission_check();
    test_media_content_identity();
    test_lease_build_identity();
    test_lora_config_identity();
    test_retention_token_identity();

    // print the executed-check count so a silently-skipped body is visible
    std::printf("test-server-vbr-predicates: %d checks, %d failures\n",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
