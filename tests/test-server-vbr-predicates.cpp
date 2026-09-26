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

#include <cstdio>
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

int main() {
    test_decode_terminal();
    test_empty_handoff();
    test_live_source_displacement();
    test_stem_matches();
    test_media_content_identity();
    test_lease_build_identity();

    // print the executed-check count so a silently-skipped body is visible
    std::printf("test-server-vbr-predicates: %d checks, %d failures\n",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
